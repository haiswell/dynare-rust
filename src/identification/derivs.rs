// SPDX-License-Identifier: GPL-3.0-or-later
//! Parameter derivatives of the steady state, of the dynamic Jacobian and of the first-order
//! solution: port of `+identification/get_perturbation_params_derivs.m` at `order = 1`
//! (`analytic_derivation_mode` 0: implicit function theorem for the steady state, Sylvester
//! equation for `dghx`; `-1`: two-sided finite differences, `fjaco.m`).
//!
//! The preprocessor's `static_params_derivs` / `dynamic_params_derivs` files are replaced by
//! symbolic derivatives of the model's residual DAG (`rp_static = df/dp` of the static model,
//! `g1p = d g1 / dp` and the second derivatives `g2 = d g1 / dy` of the dynamic model).

use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};

use crate::linalg::schur::{ordschur, schur};
use crate::linalg::{Mat, mul};
use crate::model::Model;
use crate::model::expr::{Dag, DiffVar, Node};
use crate::model::tape::Tape;
use crate::solver::DecisionRule;

/// The parameters an identification analysis looks at, in Dynare's `xparam1` order:
/// stderr of shocks, then corr of shocks, then model parameters.
#[derive(Clone, Debug, Default)]
pub struct ParamSet {
    /// `indpstderr`: exogenous indices.
    pub stderr: Vec<usize>,
    /// `indpcorr`: pairs of exogenous indices.
    pub corr: Vec<(usize, usize)>,
    /// `indpmodel`: parameter indices.
    pub model: Vec<usize>,
    /// Display names (`SE_e`, `CC_e1_e2`, parameter names).
    pub names: Vec<String>,
}

impl ParamSet {
    pub fn n_stderr(&self) -> usize {
        self.stderr.len()
    }
    pub fn n_corr(&self) -> usize {
        self.corr.len()
    }
    pub fn n_model(&self) -> usize {
        self.model.len()
    }
    /// stderr + corr parameters (they come first).
    pub fn n_shock(&self) -> usize {
        self.stderr.len() + self.corr.len()
    }
    pub fn len(&self) -> usize {
        self.n_shock() + self.model.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `set_all_parameters.m`: the model parameters and `Sigma_e` at `theta`, starting from the
    /// calibration (`params`, `sigma_e`); stderr changes keep the calibrated correlations.
    pub fn apply(&self, theta: &[f64], params: &[f64], sigma_e: &Mat) -> (Vec<f64>, Mat) {
        let mut p = params.to_vec();
        let mut q = sigma_e.clone();
        let ns = self.n_stderr();
        for (k, &i) in self.stderr.iter().enumerate() {
            let v = theta[k];
            let old = q[(i, i)].sqrt();
            for j in 0..q.ncols() {
                if j != i && old > 0.0 {
                    q[(i, j)] *= v / old;
                    q[(j, i)] *= v / old;
                }
            }
            q[(i, i)] = v * v;
        }
        for (k, &(i, j)) in self.corr.iter().enumerate() {
            let c = theta[ns + k];
            let cov = c * (q[(i, i)] * q[(j, j)]).sqrt();
            q[(i, j)] = cov;
            q[(j, i)] = cov;
        }
        for (k, &i) in self.model.iter().enumerate() {
            p[i] = theta[self.n_shock() + k];
        }
        (p, q)
    }
}

/// Compiled parameter derivatives of the static and dynamic models (built once per model).
pub struct ModelParamDerivs {
    n: usize,
    n_exo: usize,
    /// `oo_.exo_steady_state`: where the derivatives are evaluated.
    x_ss: Vec<f64>,
    n_par: usize,
    /// Static model: outputs are `d f_i / d p` for the entries in `rp_entries`.
    static_tape: Tape,
    rp_entries: Vec<(usize, usize)>,
    /// Dynamic model at y3 = [y(-1); y; y(+1)]: outputs are the `g1p` entries then the `g2`
    /// entries (second column restricted to endogenous columns).
    dyn_tape: Tape,
    g1p_entries: Vec<(usize, usize, usize)>,
    g2_entries: Vec<(usize, usize, usize)>,
}

/// Steady-state and Jacobian derivatives at one point (all model parameters).
pub struct ModelDerivs {
    /// `dys`: n_endo x n_params, declaration order.
    pub dys: Mat,
    /// Total derivative of the dynamic Jacobian (full layout `[y(-1), y, y(+1), x]`,
    /// see `Model::dynamic_jacobian`) with respect to each model parameter.
    pub dg1: Vec<Mat>,
}

impl ModelParamDerivs {
    pub fn new(model: &Model) -> Self {
        let n = model.n_endo();
        let n_exo = model.n_exo();
        let n_par = model.n_params();
        let neq = model.n_equations();
        let mut dag: Dag = model.residual_node(0).0.clone();
        let resid: Vec<u32> = (0..neq).map(|i| model.residual_node(i).1).collect();
        let mut memos: HashMap<DiffVar, HashMap<u32, u32>> = HashMap::new();

        // static parameter derivatives
        let mut smemo = HashMap::new();
        let mut rp_entries = Vec::new();
        let mut rp_nodes = Vec::new();
        let mut leaves = Vec::new();
        let mut smemo_p: HashMap<DiffVar, HashMap<u32, u32>> = HashMap::new();
        for (i, &r) in resid.iter().enumerate() {
            let s = dag.to_static(r, &mut smemo);
            leaves.clear();
            dag.collect_leaves(s, &mut leaves);
            let mut ps: Vec<u32> =
                leaves.iter().filter_map(|l| if let Node::Param(q) = l { Some(*q) } else { None }).collect();
            ps.sort_unstable();
            for q in ps {
                let v = DiffVar::Param(q);
                let d = dag.diff(s, v, smemo_p.entry(v).or_default());
                if d != Dag::ZERO {
                    rp_entries.push((i, q as usize));
                    rp_nodes.push(d);
                }
            }
        }

        // dynamic Jacobian entries, then their parameter and variable derivatives
        let col_of = |nd: Node| -> Option<(DiffVar, usize)> {
            match nd {
                Node::Endo(j, l) if (-1..=1).contains(&l) => {
                    Some((DiffVar::Endo(j, l), (l + 1) as usize * n + j as usize))
                }
                Node::Exo(k, 0) => Some((DiffVar::Exo(k, 0), 3 * n + k as usize)),
                _ => None,
            }
        };
        let mut g1p_entries = Vec::new();
        let mut g2_entries = Vec::new();
        let mut g1p_nodes = Vec::new();
        let mut g2_nodes = Vec::new();
        for (i, &r) in resid.iter().enumerate() {
            leaves.clear();
            dag.collect_leaves(r, &mut leaves);
            let mut vars: Vec<(DiffVar, usize)> = leaves.iter().filter_map(|&l| col_of(l)).collect();
            vars.sort_by_key(|v| v.1);
            for &(v, c) in &vars {
                let e = dag.diff(r, v, memos.entry(v).or_default());
                if e == Dag::ZERO {
                    continue;
                }
                let mut el = Vec::new();
                dag.collect_leaves(e, &mut el);
                // parameters: STEADY_STATE(y) depends on all of them (`ss_param_deriv`)
                let mut items: Vec<(DiffVar, usize, bool)> =
                    dag.param_deps(e).into_iter().map(|q| (DiffVar::Param(q), q as usize, true)).collect();
                for &l in &el {
                    if let Some((v2, c2)) = col_of(l) {
                        if c2 < 3 * n {
                            items.push((v2, c2, false));
                        }
                    }
                }
                items.sort_by_key(|t| (t.2, t.1));
                for (v2, idx, is_par) in items {
                    let d = dag.diff(e, v2, memos.entry(v2).or_default());
                    if d == Dag::ZERO {
                        continue;
                    }
                    if is_par {
                        g1p_entries.push((i, c, idx));
                        g1p_nodes.push(d);
                    } else {
                        g2_entries.push((i, c, idx));
                        g2_nodes.push(d);
                    }
                }
            }
        }
        let st_y = |j: u32, _l: i32| j as usize;
        let dyn_y = move |j: u32, l: i32| (l + 1) as usize * n + j as usize;
        let x0 = |k: u32, _l: i32| k as usize;
        let static_tape = Tape::compile(&dag, &rp_nodes, &st_y, &x0);
        let mut roots = g1p_nodes;
        roots.extend(g2_nodes);
        let dyn_tape = Tape::compile(&dag, &roots, &dyn_y, &x0);
        ModelParamDerivs {
            n,
            n_exo,
            x_ss: vec![0.0; n_exo],
            n_par,
            static_tape,
            rp_entries,
            dyn_tape,
            g1p_entries,
            g2_entries,
        }
    }

    /// Evaluates the static and dynamic derivatives at the exogenous steady state `x_ss`
    /// (`oo_.exo_steady_state`, zero by default).
    pub fn at_exo_steady_state(mut self, x_ss: &[f64]) -> Self {
        self.x_ss = x_ss.to_vec();
        self
    }

    /// `rp_static` (static residual derivatives w.r.t. all parameters) at `ys`.
    pub fn static_param_jacobian(&self, ys: &[f64], params: &[f64]) -> Mat {
        let x = &self.x_ss;
        let mut s = vec![0.0; self.static_tape.scratch_len()];
        self.static_tape.eval(ys, x, params, &mut s);
        let mut rp = Mat::zeros(self.n, self.n_par);
        for (k, &(i, q)) in self.rp_entries.iter().enumerate() {
            rp[(i, q)] += self.static_tape.output(k, &s);
        }
        rp
    }

    /// `dys` (implicit function theorem, with the Schur fallback of
    /// get_perturbation_params_derivs.m for a singular static Jacobian) and the total parameter
    /// derivative of the dynamic Jacobian at the steady state `ys`:
    /// `dg1 = g1p + g2 * dyy0` (part (1) + part (2)).
    pub fn eval(&self, model: &Model, ys: &[f64], params: &[f64], qz_criterium: f64) -> Result<ModelDerivs> {
        let n = self.n;
        let g1s = model.static_jacobian(ys, &self.x_ss, params);
        let rp = self.static_param_jacobian(ys, params);
        // MATLAB's `\` returns Inf/NaN for an exactly singular static Jacobian (unit roots),
        // which sends Dynare to the Schur branch; LU rounding can leave a tiny non-zero pivot
        // instead, so a numerically singular matrix (rcond < eps) takes the Schur branch too.
        let (lu, _, info) = crate::linalg::lu(&g1s);
        let singular = info > 0 || crate::linalg::rcond_from_lu(&lu, crate::linalg::norm1(&g1s)) < f64::EPSILON;
        let direct = if singular { None } else { crate::linalg::solve(&g1s, &rp) };
        let dys = match direct {
            Some(d) if d.iter().all(|v| v.is_finite()) => -d,
            _ => {
                // [U,T] = schur(g1_static); unit (zero) roots first; dys = -U2 (T22 \ U2') rp
                let mut sch = schur(&g1s)?;
                let e1: Vec<bool> = sch.eig_abs().iter().map(|&e| e < qz_criterium - 1.0).collect();
                let k = e1.iter().filter(|&&b| b).count();
                ordschur(&mut sch, &e1)?;
                let t22 = sch.t.view((k, k), (n - k, n - k)).into_owned();
                let u2 = sch.u.columns(k, n - k).into_owned();
                let rhs = mul(&u2, true, &rp, false);
                let z = crate::linalg::solve(&t22, &rhs).ok_or_else(|| anyhow!("singular static Jacobian"))?;
                let d = -mul(&u2, false, &z, false);
                // The Schur projection solves g1_static dys = -rp only when rp is in the range of
                // g1_static; otherwise (parameters moving the unit-root directions) use the
                // minimum-norm least-squares solution, whose determined rows agree with the
                // finite-difference derivative of the steady state (and with Dynare's output).
                let res = (mul(&g1s, false, &d, false) + &rp).amax();
                if res <= 1e-8 * rp.amax().max(1.0) {
                    d
                } else {
                    let (u, sv, v) = super::la::svd_econ(&g1s);
                    let tol = n as f64 * super::la::eps_of(sv[0]);
                    let mut utr = mul(&u, true, &rp, false);
                    for (i, &si) in sv.iter().enumerate() {
                        let f = if si > tol { 1.0 / si } else { 0.0 };
                        for c in 0..utr.ncols() {
                            utr[(i, c)] *= f;
                        }
                    }
                    -mul(&v, false, &utr, false)
                }
            }
        };
        let mut y3 = Vec::with_capacity(3 * n);
        for _ in 0..3 {
            y3.extend_from_slice(ys);
        }
        // dynamic_params_derivs(y3n, exo, params, ys, dys, d2ys)
        let pext = model.params_with_steady_state_derivs(params, ys, &dys, None);
        let mut s = vec![0.0; self.dyn_tape.scratch_len()];
        self.dyn_tape.eval(&y3, &self.x_ss, &pext, &mut s);
        let ncol = 3 * n + self.n_exo;
        let neq = model.n_equations();
        let mut dg1 = vec![Mat::zeros(neq, ncol); self.n_par];
        for (k, &(i, c, q)) in self.g1p_entries.iter().enumerate() {
            dg1[q][(i, c)] += self.dyn_tape.output(k, &s);
        }
        let off = self.g1p_entries.len();
        for (k, &(i, c, c2)) in self.g2_entries.iter().enumerate() {
            let v = self.dyn_tape.output(off + k, &s);
            let var = c2 % n;
            for (q, m) in dg1.iter_mut().enumerate() {
                let d = dys[(var, q)];
                if d != 0.0 {
                    m[(i, c)] += v * d;
                }
            }
        }
        Ok(ModelDerivs { dys, dg1 })
    }
}

/// `dr.derivs` at order 1 (all in DR order, parameters in [`ParamSet`] order).
#[derive(Clone, Debug)]
pub struct PerturbationDerivs {
    /// n_endo x n_model.
    pub d_yss: Mat,
    /// Per model parameter, legacy dynamic g1 layout (equations x [lli columns, exo]).
    pub dg1_legacy: Vec<Mat>,
    /// Per parameter (all `tot`): n_endo x nspred, n_endo x n_exo, n_endo x n_endo, n_exo x n_exo.
    pub dghx: Vec<Mat>,
    pub dghu: Vec<Mat>,
    pub dom: Vec<Mat>,
    pub dsigma_e: Vec<Mat>,
}

/// Columns of the legacy (lead_lag_incidence-ordered) g1 in the full layout.
pub fn legacy_columns(model: &Model) -> Vec<usize> {
    let n = model.n_endo();
    let lli = &model.lead_lag_incidence;
    let nnz = lli.iter().flat_map(|r| r.iter()).filter(|&&k| k > 0).count();
    let mut cols = vec![0; nnz];
    for (v, r) in lli.iter().enumerate() {
        for t in 0..3 {
            if r[t] > 0 {
                cols[r[t] - 1] = t * n + v;
            }
        }
    }
    cols.extend((0..model.n_exo()).map(|k| 3 * n + k));
    cols
}

pub fn to_legacy(g: &Mat, cols: &[usize]) -> Mat {
    Mat::from_fn(g.nrows(), cols.len(), |i, j| g[(i, cols[j])])
}

/// `dSigma_e` for the stderr and corr parameters (zero for model parameters).
pub fn dsigma_e(set: &ParamSet, sigma_e: &Mat) -> Vec<Mat> {
    let ne = sigma_e.nrows();
    let mut out = vec![Mat::zeros(ne, ne); set.len()];
    let isdiag = (0..ne).all(|i| (0..ne).all(|j| i == j || sigma_e[(i, j)] == 0.0));
    for (jp, &k) in set.stderr.iter().enumerate() {
        out[jp][(k, k)] = 2.0 * sigma_e[(k, k)].sqrt();
        if !isdiag {
            for kk in 0..ne {
                if kk != k {
                    let corr = sigma_e[(k, kk)] / (sigma_e[(k, k)] * sigma_e[(kk, kk)]).sqrt();
                    let v = corr * sigma_e[(kk, kk)].sqrt();
                    out[jp][(k, kk)] = v;
                    out[jp][(kk, k)] = v;
                }
            }
        }
    }
    for (jc, &(a, b)) in set.corr.iter().enumerate() {
        let v = sigma_e[(a, a)].sqrt() * sigma_e[(b, b)].sqrt();
        out[set.n_stderr() + jc][(a, b)] = v;
        out[set.n_stderr() + jc][(b, a)] = v;
    }
    out
}

/// Port of `sylvester3.m` + `sylvester3a.m`: solves `A X + B X C = D` for every right-hand
/// side (`C` real Schur form, column by column / 2x2 blocks), then refines with the fixed-point
/// iteration `X = A^{-1} D - A^{-1} B X C` (at most 4 rounds of up to 500 iterations, stopped
/// at a max abs change of 1e-8 as Dynare does).
pub fn sylvester3(a: &Mat, b: &Mat, c: &Mat, d: &[Mat]) -> Result<Vec<Mat>> {
    let n = a.nrows();
    let m = c.nrows();
    let mut xs = Vec::with_capacity(d.len());
    if m == 0 {
        return Ok(d.iter().map(|_| Mat::zeros(n, 0)).collect());
    }
    let sch = schur(c)?;
    let (u, t) = (&sch.u, &sch.t);
    // [aa, bb, qq, zz] = qz(a, b, 'real'): aa = qq a zz, bb = qq b zz (LAPACK: S = Q' A Z)
    let qz = crate::linalg::qz::qz_unordered(a, b, 0.0);
    if qz.info > 0 {
        bail!("sylvester3: QZ failed (info {})", qz.info);
    }
    let (aa, bb) = (&qz.s, &qz.t);
    for dj in d {
        let dd = mul(&mul(&qz.q, true, dj, false), false, u, false);
        let mut x = Mat::zeros(n, m);
        let mut i = 0;
        while i < m {
            // bb * X(:,0..i) * T(0..i, col): the columns already solved
            let rhs_for = |x: &Mat, col: usize| -> nalgebra::DVector<f64> {
                let mut acc = nalgebra::DVector::zeros(n);
                for k in 0..i {
                    let tk = t[(k, col)];
                    if tk != 0.0 {
                        acc += x.column(k) * tk;
                    }
                }
                bb * acc
            };
            if i + 1 < m && t[(i + 1, i)] != 0.0 {
                let c0 = rhs_for(&x, i);
                let c1 = rhs_for(&x, i + 1);
                let mut big = Mat::zeros(2 * n, 2 * n);
                big.view_mut((0, 0), (n, n)).copy_from(&(aa + bb * t[(i, i)]));
                big.view_mut((0, n), (n, n)).copy_from(&(bb * t[(i + 1, i)]));
                big.view_mut((n, 0), (n, n)).copy_from(&(bb * t[(i, i + 1)]));
                big.view_mut((n, n), (n, n)).copy_from(&(aa + bb * t[(i + 1, i + 1)]));
                let mut rhs = Mat::zeros(2 * n, 1);
                for r in 0..n {
                    rhs[(r, 0)] = dd[(r, i)] - c0[r];
                    rhs[(n + r, 0)] = dd[(r, i + 1)] - c1[r];
                }
                let z = crate::linalg::solve(&big, &rhs).ok_or_else(|| anyhow!("sylvester3: singular system"))?;
                for r in 0..n {
                    x[(r, i)] = z[(r, 0)];
                    x[(r, i + 1)] = z[(n + r, 0)];
                }
                i += 2;
            } else {
                let c0 = rhs_for(&x, i);
                let rhs: Vec<f64> = (0..n).map(|r| dd[(r, i)] - c0[r]).collect();
                let z = quasi_triangular_solve(&(aa + bb * t[(i, i)]), &rhs)?;
                for r in 0..n {
                    x[(r, i)] = z[r];
                }
                i += 1;
            }
        }
        xs.push(mul(&mul(&qz.z, false, &x, false), false, u, true));
    }
    // sylvester3a refinement
    let ainv = crate::linalg::solve(a, &Mat::identity(n, n)).ok_or_else(|| anyhow!("sylvester3a: singular A"))?;
    let ab = mul(&ainv, false, b, false);
    for (x0, dj) in xs.iter_mut().zip(d) {
        let ad = mul(&ainv, false, dj, false);
        for _round in 0..4 {
            let mut e = f64::INFINITY;
            let mut iter = 1;
            let start = x0.clone();
            while e > 1e-8 && iter < 500 {
                let x = &ad - mul(&mul(&ab, false, x0, false), false, c, false);
                e = (&x - &*x0).amax();
                *x0 = x;
                iter += 1;
                if !e.is_finite() {
                    break;
                }
            }
            if !e.is_finite() || x0.iter().any(|v| !v.is_finite()) {
                // diverging fixed point: keep the direct solution
                *x0 = start;
                break;
            }
            if iter < 500 {
                break;
            }
        }
    }
    Ok(xs)
}

/// Port of the `analytic_derivation_mode = 0` branch of get_perturbation_params_derivs.m
/// (order 1): `dghx` from the Sylvester equation, `dghu`, `dOm`, `dSigma_e`, `dYss`.
pub fn perturbation_derivs(
    model: &Model,
    dr: &DecisionRule,
    g1: &Mat,
    md: &ModelDerivs,
    sigma_e: &Mat,
    set: &ParamSet,
) -> Result<PerturbationDerivs> {
    let n = model.n_endo();
    let ne = model.n_exo();
    let nspred = dr.nspred;
    let lli = &model.lead_lag_incidence;
    let ov = &dr.order_var;
    let states: Vec<usize> = (dr.nstatic..dr.nstatic + nspred).collect();
    let neq = g1.nrows();
    let cur = |j: usize| lli[ov[j]][1] > 0;
    let lead = |j: usize| lli[ov[j]][2] > 0;

    // A, B (DR-ordered columns)
    let mut a = Mat::zeros(neq, n);
    let mut b = Mat::zeros(neq, n);
    // all current columns first: the lead terms below add into state columns, which a later
    // `set_column` would otherwise overwrite (variables with both a lag and a lead)
    for j in 0..n {
        if cur(j) {
            a.set_column(j, &g1.column(n + ov[j]));
        }
    }
    for j in 0..n {
        let v = ov[j];
        if lead(j) {
            b.set_column(j, &g1.column(2 * n + v));
            for (k, &s) in states.iter().enumerate() {
                let gx = dr.ghx[(j, k)];
                if gx != 0.0 {
                    for r in 0..neq {
                        a[(r, s)] += g1[(r, 2 * n + v)] * gx;
                    }
                }
            }
        }
    }
    let ghx_s = Mat::from_fn(nspred, nspred, |i, k| dr.ghx[(states[i], k)]);
    // zx in the full layout (3n x nspred)
    let mut zx = Mat::zeros(3 * n, nspred);
    for (k, &s) in states.iter().enumerate() {
        zx[(ov[s], k)] = 1.0;
    }
    let ghx_lead = mul(&dr.ghx, false, &ghx_s, false);
    for j in 0..n {
        let v = ov[j];
        for k in 0..nspred {
            zx[(n + v, k)] = dr.ghx[(j, k)];
            zx[(2 * n + v, k)] = ghx_lead[(j, k)];
        }
    }
    let nm = set.n_model();
    let mut rhs = Vec::with_capacity(nm);
    for &p in &set.model {
        let g = md.dg1[p].columns(0, 3 * n).into_owned();
        rhs.push(-mul(&g, false, &zx, false));
    }
    let dghx_m = sylvester3(&a, &b, &ghx_s, &rhs)?;

    let ainv = crate::linalg::solve(&a, &Mat::identity(n, n)).ok_or_else(|| anyhow!("singular A in dghu"))?;
    let ns = set.n_shock();
    let tot = set.len();
    let mut dghx = vec![Mat::zeros(n, nspred); tot];
    let mut dghu = vec![Mat::zeros(n, ne); tot];
    let mut dom = vec![Mat::zeros(n, n); tot];
    let ghu_sig = mul(&dr.ghu, false, sigma_e, false);
    for (jm, &p) in set.model.iter().enumerate() {
        let dg = &md.dg1[p];
        let mut da = Mat::zeros(neq, n);
        for j in 0..n {
            let v = ov[j];
            if cur(j) {
                da.set_column(j, &dg.column(n + v));
            }
        }
        for j in 0..n {
            if !lead(j) {
                continue;
            }
            let v = ov[j];
            for (k, &s) in states.iter().enumerate() {
                let gx = dr.ghx[(j, k)];
                let dgx = dghx_m[jm][(j, k)];
                for r in 0..neq {
                    da[(r, s)] += dg[(r, 2 * n + v)] * gx + g1[(r, 2 * n + v)] * dgx;
                }
            }
        }
        let dgx_exo = dg.columns(3 * n, ne).into_owned();
        let t = dgx_exo + mul(&da, false, &dr.ghu, false);
        let du = -mul(&ainv, false, &t, false);
        let o = mul(&du, false, &ghu_sig, true);
        dom[ns + jm] = &o + o.transpose();
        dghx[ns + jm] = dghx_m[jm].clone();
        dghu[ns + jm] = du;
    }
    let dsig = dsigma_e(set, sigma_e);
    for jp in 0..ns {
        dom[jp] = mul(&mul(&dr.ghu, false, &dsig[jp], false), false, &dr.ghu, true);
    }
    let d_yss = Mat::from_fn(n, nm, |i, k| md.dys[(ov[i], set.model[k])]);
    let cols = legacy_columns(model);
    let dg1_legacy = set.model.iter().map(|&p| to_legacy(&md.dg1[p], &cols)).collect();
    Ok(PerturbationDerivs { d_yss, dg1_legacy, dghx, dghu, dom, dsigma_e: dsig })
}

/// MATLAB `M \ r` for the (quasi) upper triangular `M = aa + bb t`: back substitution when `M`
/// is upper triangular (what MATLAB's `\` detects), LU otherwise.
fn quasi_triangular_solve(m: &Mat, r: &[f64]) -> Result<Vec<f64>> {
    let n = m.nrows();
    let upper = (0..n).all(|j| (j + 1..n).all(|i| m[(i, j)] == 0.0));
    if upper {
        let mut x = r.to_vec();
        for i in (0..n).rev() {
            let mut s = x[i];
            for j in i + 1..n {
                s -= m[(i, j)] * x[j];
            }
            x[i] = s / m[(i, i)];
        }
        return Ok(x);
    }
    let z = crate::linalg::solve(m, &Mat::from_column_slice(n, 1, r))
        .ok_or_else(|| anyhow!("sylvester3: singular system"))?;
    Ok(z.iter().copied().collect())
}
