// SPDX-License-Identifier: GPL-3.0-or-later
//! Second-order parameter derivatives of the first-order solution: port of the `d2flag`
//! branches of `+identification/get_perturbation_params_derivs.m` (`analytic_derivation_mode
//! = 0`): `d2Yss` (implicit function theorem, Ratto and Iskrev 2012 eq. 15), `d2g1` (eq. 16,
//! the five parts), `d2KalmanA` (generalized Sylvester equations, eqs. 17-18) and `d2Om`.
//!
//! The preprocessor's `static_params_derivs` / `dynamic_params_derivs` / `static_g2` /
//! `dynamic_g3` files are replaced by symbolic derivatives of the model's residual DAG
//! (`rpp`, `g1p_static`, `g2_static`, `g1pp`, `g2p`, `g3`), compiled to tapes once.
//!
//! Pairs of parameters are stored in `dyn_vech` order (upper triangle, column-major:
//! (1,1), (1,2), (2,2), (1,3), ...), the `jcount` order of Dynare's loops.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)] // index-for-index port

use std::collections::HashMap;

use anyhow::{Result, anyhow};

use crate::identification::derivs::{ModelDerivs, ParamSet, PerturbationDerivs, sylvester3};
use crate::linalg::{Mat, mul};
use crate::model::Model;
use crate::model::expr::{Dag, DiffVar, Node};
use crate::model::tape::Tape;
use crate::solver::DecisionRule;

/// Compiled second- and third-order derivatives of the static and dynamic models.
pub struct SecondTapes {
    n: usize,
    n_exo: usize,
    static_tape: Tape,
    /// (eq, q1, q2), q1 <= q2: d2 f / dq1 dq2 (static)
    rpp: Vec<(usize, usize, usize)>,
    /// (eq, y, q): d2 f / dy dq (static)
    g1ps: Vec<(usize, usize, usize)>,
    /// (eq, y1, y2), y1 <= y2: d2 f / dy1 dy2 (static)
    g2s: Vec<(usize, usize, usize)>,
    dyn_tape: Tape,
    /// (eq, c, q1, q2), q1 <= q2
    g1pp: Vec<(usize, usize, usize, usize)>,
    /// (eq, c, c2, q): c2 endogenous
    g2p: Vec<(usize, usize, usize, usize)>,
    /// (eq, c, c2, c3), c2 <= c3 endogenous
    g3: Vec<(usize, usize, usize, usize)>,
    /// (eq, c, c2): c2 endogenous
    g2: Vec<(usize, usize, usize)>,
}

fn leaves_of(dag: &Dag, id: u32) -> Vec<Node> {
    let mut v = Vec::new();
    dag.collect_leaves(id, &mut v);
    v
}

impl SecondTapes {
    pub fn new(model: &Model) -> SecondTapes {
        let n = model.n_endo();
        let n_exo = model.n_exo();
        let neq = model.n_equations();
        let mut dag: Dag = model.residual_node(0).0.clone();
        let resid: Vec<u32> = (0..neq).map(|i| model.residual_node(i).1).collect();
        let mut memos: HashMap<DiffVar, HashMap<u32, u32>> = HashMap::new();
        let params_in = |dag: &Dag, id: u32| -> Vec<u32> {
            let mut ps: Vec<u32> = leaves_of(dag, id)
                .into_iter()
                .filter_map(|l| if let Node::Param(q) = l { Some(q) } else { None })
                .collect();
            ps.sort_unstable();
            ps.dedup();
            ps
        };
        // ---- static model
        let mut smemo = HashMap::new();
        let (mut rpp, mut g1ps, mut g2s) = (Vec::new(), Vec::new(), Vec::new());
        let (mut n_rpp, mut n_g1ps, mut n_g2s) = (Vec::new(), Vec::new(), Vec::new());
        let endo_static = |dag: &Dag, id: u32| -> Vec<u32> {
            let mut v: Vec<u32> = leaves_of(dag, id)
                .into_iter()
                .filter_map(|l| if let Node::Endo(j, _) = l { Some(j) } else { None })
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        for (i, &r) in resid.iter().enumerate() {
            let s = dag.to_static(r, &mut smemo);
            for q1 in params_in(&dag, s) {
                let d1 = dag.diff(s, DiffVar::Param(q1), memos.entry(DiffVar::Param(q1)).or_default());
                if d1 == Dag::ZERO {
                    continue;
                }
                for q2 in params_in(&dag, d1).into_iter().filter(|&q2| q2 >= q1) {
                    let d2 = dag.diff(d1, DiffVar::Param(q2), memos.entry(DiffVar::Param(q2)).or_default());
                    if d2 != Dag::ZERO {
                        rpp.push((i, q1 as usize, q2 as usize));
                        n_rpp.push(d2);
                    }
                }
            }
            for y1 in endo_static(&dag, s) {
                let v1 = DiffVar::Endo(y1, 0);
                let d1 = dag.diff(s, v1, memos.entry(v1).or_default());
                if d1 == Dag::ZERO {
                    continue;
                }
                for q in params_in(&dag, d1) {
                    let d2 = dag.diff(d1, DiffVar::Param(q), memos.entry(DiffVar::Param(q)).or_default());
                    if d2 != Dag::ZERO {
                        g1ps.push((i, y1 as usize, q as usize));
                        n_g1ps.push(d2);
                    }
                }
                for y2 in endo_static(&dag, d1).into_iter().filter(|&y2| y2 >= y1) {
                    let v2 = DiffVar::Endo(y2, 0);
                    let d2 = dag.diff(d1, v2, memos.entry(v2).or_default());
                    if d2 != Dag::ZERO {
                        g2s.push((i, y1 as usize, y2 as usize));
                        n_g2s.push(d2);
                    }
                }
            }
        }
        let mut sroots = n_rpp;
        sroots.extend(n_g1ps);
        sroots.extend(n_g2s);
        let st_y = |j: u32, _l: i32| j as usize;
        let x0 = |k: u32, _l: i32| k as usize;
        let static_tape = Tape::compile(&dag, &sroots, &st_y, &x0);

        // STEADY_STATE(y) in the dynamic model depends on every parameter (`ss_param_deriv`,
        // `ss_param_2nd_deriv`: `prepareForDerivation` adds all the parameter deriv ids)
        let params_dyn = |dag: &Dag, id: u32| -> Vec<u32> {
            let mut ps = dag.param_deps(id);
            ps.dedup();
            ps
        };
        // ---- dynamic model (columns of the full layout [y(-1), y, y(+1), x])
        let col_of = |nd: Node| -> Option<(DiffVar, usize)> {
            match nd {
                Node::Endo(j, l) if (-1..=1).contains(&l) => {
                    Some((DiffVar::Endo(j, l), (l + 1) as usize * n + j as usize))
                }
                Node::Exo(k, 0) => Some((DiffVar::Exo(k, 0), 3 * n + k as usize)),
                _ => None,
            }
        };
        let endo_cols = |dag: &Dag, id: u32| -> Vec<(DiffVar, usize)> {
            let mut v: Vec<(DiffVar, usize)> =
                leaves_of(dag, id).into_iter().filter_map(col_of).filter(|v| v.1 < 3 * n).collect();
            v.sort_by_key(|v| v.1);
            v.dedup_by_key(|v| v.1);
            v
        };
        let (mut g1pp, mut g2p, mut g3, mut g2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut n_g1pp, mut n_g2p, mut n_g3, mut n_g2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, &r) in resid.iter().enumerate() {
            let mut vars: Vec<(DiffVar, usize)> = leaves_of(&dag, r).into_iter().filter_map(col_of).collect();
            vars.sort_by_key(|v| v.1);
            vars.dedup_by_key(|v| v.1);
            for &(v, c) in &vars {
                let e = dag.diff(r, v, memos.entry(v).or_default());
                if e == Dag::ZERO {
                    continue;
                }
                for q1 in params_dyn(&dag, e) {
                    let e1 = dag.diff(e, DiffVar::Param(q1), memos.entry(DiffVar::Param(q1)).or_default());
                    if e1 == Dag::ZERO {
                        continue;
                    }
                    for q2 in params_dyn(&dag, e1).into_iter().filter(|&q2| q2 >= q1) {
                        let e2 = dag.diff(e1, DiffVar::Param(q2), memos.entry(DiffVar::Param(q2)).or_default());
                        if e2 != Dag::ZERO {
                            g1pp.push((i, c, q1 as usize, q2 as usize));
                            n_g1pp.push(e2);
                        }
                    }
                }
                for (v2, c2) in endo_cols(&dag, e) {
                    let e2 = dag.diff(e, v2, memos.entry(v2).or_default());
                    if e2 == Dag::ZERO {
                        continue;
                    }
                    g2.push((i, c, c2));
                    n_g2.push(e2);
                    for q in params_dyn(&dag, e2) {
                        let e3 = dag.diff(e2, DiffVar::Param(q), memos.entry(DiffVar::Param(q)).or_default());
                        if e3 != Dag::ZERO {
                            g2p.push((i, c, c2, q as usize));
                            n_g2p.push(e3);
                        }
                    }
                    for (v3, c3) in endo_cols(&dag, e2).into_iter().filter(|v| v.1 >= c2) {
                        let e3 = dag.diff(e2, v3, memos.entry(v3).or_default());
                        if e3 != Dag::ZERO {
                            g3.push((i, c, c2, c3));
                            n_g3.push(e3);
                        }
                    }
                }
            }
        }
        let mut droots = n_g1pp;
        droots.extend(n_g2p);
        droots.extend(n_g3);
        droots.extend(n_g2);
        let dyn_y = move |j: u32, l: i32| (l + 1) as usize * n + j as usize;
        let dyn_tape = Tape::compile(&dag, &droots, &dyn_y, &x0);
        SecondTapes { n, n_exo, static_tape, rpp, g1ps, g2s, dyn_tape, g1pp, g2p, g3, g2 }
    }
}

/// The `d2flag` outputs of get_perturbation_params_derivs.m (DR order, `set` parameter order).
pub struct SecondOrder {
    /// `d2Yss(r,:,:)` per DR row (model x model parameters).
    pub d2yss: Vec<Mat>,
    /// `d2KalmanA`: vec(n x n) x tot(tot+1)/2.
    pub d2kalman_a: Mat,
    /// `d2Om`: vech(n x n) x tot(tot+1)/2.
    pub d2om: Mat,
}

/// `d2ys` (eq. 15) and `d2g1` (eq. 16) from the symbolic second/third derivatives.
fn symbolic_d2(
    model: &Model,
    tp: &SecondTapes,
    md: &ModelDerivs,
    ys: &[f64],
    params: &[f64],
    set: &ParamSet,
) -> Result<D2Model> {
    let n = tp.n;
    let ne = tp.n_exo;
    let neq = model.n_equations();
    let dys = &md.dys; // n x n_params, declaration order
    let nm = set.n_model();
    let pm = &set.model;
    let pos: HashMap<usize, usize> = pm.iter().enumerate().map(|(a, &p)| (p, a)).collect();
    let x = vec![0.0; ne];
    // ---- d2ys (selected model parameters), Ratto and Iskrev (2012) eq. 15
    let mut s = vec![0.0; tp.static_tape.scratch_len()];
    tp.static_tape.eval(ys, &x, params, &mut s);
    let mut rhs = vec![Mat::zeros(n, nm); nm]; // rhs[b](i, a) = rpp + gam
    let mut out_k = 0;
    for &(i, q1, q2) in &tp.rpp {
        let v = tp.static_tape.output(out_k, &s);
        out_k += 1;
        if let (Some(&a), Some(&b)) = (pos.get(&q1), pos.get(&q2)) {
            rhs[b][(i, a)] += v;
            if a != b {
                rhs[a][(i, b)] += v;
            }
        }
    }
    for &(i, y, q) in &tp.g1ps {
        let v = tp.static_tape.output(out_k, &s);
        out_k += 1;
        if let Some(&a) = pos.get(&q) {
            // tmp(a,b) = g1p(i,y,a) dys(y,b), gam += tmp + tmp'
            for b in 0..nm {
                let dyb = dys[(y, pm[b])];
                rhs[b][(i, a)] += v * dyb;
                rhs[a][(i, b)] += v * dyb;
            }
        }
    }
    for &(i, y1, y2) in &tp.g2s {
        let v = tp.static_tape.output(out_k, &s);
        out_k += 1;
        for a in 0..nm {
            for b in 0..nm {
                let mut t = dys[(y1, pm[a])] * dys[(y2, pm[b])];
                if y1 != y2 {
                    t += dys[(y2, pm[a])] * dys[(y1, pm[b])];
                }
                rhs[b][(i, a)] += v * t;
            }
        }
    }
    let g1s = model.static_jacobian(ys, &x, params);
    let mut d2ys = vec![Mat::zeros(n, nm); nm]; // d2ys[b](var, a)
    for b in 0..nm {
        let sol = crate::linalg::solve(&g1s, &rhs[b])
            .ok_or_else(|| anyhow!("analytic Hessian: singular static Jacobian (unit roots) is not supported"))?;
        d2ys[b] = -sol;
    }

    // ---- d2g1 per model pair (full layout), eq. 16
    let mut dsn = vec![0.0; tp.dyn_tape.scratch_len()];
    let mut y3 = Vec::with_capacity(3 * n);
    for _ in 0..3 {
        y3.extend_from_slice(ys);
    }
    // dynamic_params_derivs(y3n, exo, params, ys, dys, d2ys): `d2ys(:,:,q2)` is `n_endo x n_params`,
    // non-zero only for the selected model parameters
    let np = model.n_params();
    let d2full: Vec<Mat> = (0..np)
        .map(|q2| {
            let mut m = Mat::zeros(n, np);
            if let Some(&b) = pos.get(&q2) {
                for a in 0..nm {
                    for v in 0..n {
                        m[(v, pm[a])] = d2ys[b][(v, a)];
                    }
                }
            }
            m
        })
        .collect();
    let pext = model.params_with_steady_state_derivs(params, ys, dys, Some(&d2full));
    tp.dyn_tape.eval(&y3, &x, &pext, &mut dsn);
    let dyy = |c: usize, a: usize| if c < 3 * n { dys[(c % n, pm[a])] } else { 0.0 };
    let ncol = 3 * n + ne;
    let pair = |a: usize, b: usize| if a <= b { b * (b + 1) / 2 + a } else { a * (a + 1) / 2 + b };
    let mut d2g1 = vec![Mat::zeros(neq, ncol); nm * (nm + 1) / 2];
    let mut k = 0;
    for &(i, c, q1, q2) in &tp.g1pp {
        let v = tp.dyn_tape.output(k, &dsn);
        k += 1;
        if let (Some(&a), Some(&b)) = (pos.get(&q1), pos.get(&q2)) {
            d2g1[pair(a, b)][(i, c)] += v;
        }
    }
    for &(i, c, c2, q) in &tp.g2p {
        let v = tp.dyn_tape.output(k, &dsn);
        k += 1;
        if let Some(&a) = pos.get(&q) {
            for b in 0..nm {
                let w = v * dyy(c2, b);
                // part 2 (a, b) and part 3 (b, a); on the diagonal both add
                d2g1[pair(a, b)][(i, c)] += w;
                if a == b {
                    d2g1[pair(a, b)][(i, c)] += w;
                }
            }
        }
    }
    for &(i, c, c2, c3) in &tp.g3 {
        let v = tp.dyn_tape.output(k, &dsn);
        k += 1;
        for b in 0..nm {
            for a in 0..=b {
                let mut t = dyy(c2, a) * dyy(c3, b);
                if c2 != c3 {
                    t += dyy(c3, a) * dyy(c2, b);
                }
                d2g1[pair(a, b)][(i, c)] += v * t;
            }
        }
    }
    for &(i, c, c2) in &tp.g2 {
        let v = tp.dyn_tape.output(k, &dsn);
        k += 1;
        let var = c2 % n;
        for b in 0..nm {
            for a in 0..=b {
                d2g1[pair(a, b)][(i, c)] += v * d2ys[b][(var, a)];
            }
        }
    }

    Ok((d2ys, d2g1))
}

/// `d2ys[b](var, a)` (declaration order, selected model parameters) and `d2g1` per model pair
/// (`a <= b`, full Jacobian layout).
pub type D2Model = (Vec<Mat>, Vec<Mat>);

/// Port of the `d2flag` computations of get_perturbation_params_derivs.m (order 1). `md`: the
/// first-order steady-state and Jacobian derivatives; `numeric`: `d2Yss`/`d2g1` from
/// `hessian_sparse.m` (`analytic_derivation_mode = -2`), else computed analytically from `tp`.
pub fn second_order(
    model: &Model,
    tp: Option<&SecondTapes>,
    md: &ModelDerivs,
    numeric: Option<D2Model>,
    dr: &DecisionRule,
    g1: &Mat,
    ys: &[f64],
    params: &[f64],
    sigma_e: &Mat,
    set: &ParamSet,
    d: &PerturbationDerivs,
) -> Result<SecondOrder> {
    let n = model.n_endo();
    let ne = model.n_exo();
    let neq = model.n_equations();
    let nm = set.n_model();
    let pm = &set.model;
    let pair = |a: usize, b: usize| if a <= b { b * (b + 1) / 2 + a } else { a * (a + 1) / 2 + b };
    let (d2ys, d2g1) = match numeric {
        Some(v) => v,
        None => symbolic_d2(model, tp.expect("second-order tapes"), md, ys, params, set)?,
    };
    // ---- GAM matrices in DR order
    let lli = &model.lead_lag_incidence;
    let ov = &dr.order_var;
    let gam = |g: &Mat, t: usize, sign: f64| {
        let mut m = Mat::zeros(neq, n);
        for j in 0..n {
            if lli[ov[j]][t] > 0 {
                let c = t * n + ov[j];
                for r in 0..neq {
                    m[(r, j)] = sign * g[(r, c)];
                }
            }
        }
        m
    };
    let gam3 = |g: &Mat| Mat::from_fn(neq, ne, |r, e| -g[(r, 3 * n + e)]);
    let gam0 = gam(g1, 1, 1.0);
    let gam1 = gam(g1, 2, -1.0);
    let dgam0: Vec<Mat> = pm.iter().map(|&p| gam(&md.dg1[p], 1, 1.0)).collect();
    let dgam1: Vec<Mat> = pm.iter().map(|&p| gam(&md.dg1[p], 2, -1.0)).collect();
    let d2gam0: Vec<Mat> = d2g1.iter().map(|m| gam(m, 1, 1.0)).collect();
    let d2gam1: Vec<Mat> = d2g1.iter().map(|m| gam(m, 2, -1.0)).collect();
    let d2gam2: Vec<Mat> = d2g1.iter().map(|m| gam(m, 0, -1.0)).collect();
    let d2gam3: Vec<Mat> = d2g1.iter().map(&gam3).collect();
    let states: Vec<usize> = (dr.nstatic..dr.nstatic + dr.nspred).collect();
    let mut ka = Mat::zeros(n, n);
    for (kk, &sv) in states.iter().enumerate() {
        ka.set_column(sv, &dr.ghx.column(kk));
    }
    let ns = set.n_shock();
    let dka: Vec<Mat> = (0..nm)
        .map(|jm| {
            let mut m = Mat::zeros(n, n);
            for (kk, &sv) in states.iter().enumerate() {
                m.set_column(sv, &d.dghx[ns + jm].column(kk));
            }
            m
        })
        .collect();
    let mm_ = &gam0 - mul(&gam1, false, &ka, false);
    let inv_mm =
        crate::linalg::solve(&mm_, &Mat::identity(n, n)).ok_or_else(|| anyhow!("analytic Hessian: singular MM"))?;
    // QQ for each model pair (j <= i), solved by sylvester3 + sylvester3a
    let mut qq = Vec::with_capacity(nm * (nm + 1) / 2);
    for i in 0..nm {
        for j in 0..=i {
            let p = pair(j, i);
            let elem1 = &d2gam0[p] - mul(&d2gam1[p], false, &ka, false);
            let elem1 = &d2gam2[p] - mul(&elem1, false, &ka, false);
            let elemj0 = &dgam0[j] - mul(&dgam1[j], false, &ka, false);
            let elemi0 = &dgam0[i] - mul(&dgam1[i], false, &ka, false);
            let mut elem2 = -mul(&elemj0, false, &dka[i], false) - mul(&elemi0, false, &dka[j], false);
            elem2 += mul(
                &(mul(&dgam1[j], false, &dka[i], false) + mul(&dgam1[i], false, &dka[j], false)),
                false,
                &ka,
                false,
            );
            elem2 +=
                mul(&gam1, false, &(mul(&dka[i], false, &dka[j], false) + mul(&dka[j], false, &dka[i], false)), false);
            qq.push(elem1 + elem2);
        }
    }
    let x2x = sylvester3(&mm_, &(-&gam1), &ka, &qq)?;

    // ---- d2Sigma_e with Dynare's indexing (see the module documentation)
    let tot = set.len();
    let tot2 = tot * (tot + 1) / 2;
    let corr_m = |a: usize, b: usize| sigma_e[(a, b)] / (sigma_e[(a, a)] * sigma_e[(b, b)]).sqrt();
    let is_diag = (0..ne).all(|a| (0..ne).all(|b| a == b || sigma_e[(a, b)] == 0.0));
    let mut d2sig_full: HashMap<usize, Mat> = HashMap::new(); // key: column-major index in tot x tot
    let nsd = set.n_stderr();
    for jp in 0..nsd {
        for ip in 0..=jp {
            let key = ip + jp * tot;
            let (kj, ki) = (set.stderr[jp], set.stderr[ip]);
            let m = d2sig_full.entry(key).or_insert_with(|| Mat::zeros(ne, ne));
            if jp == ip {
                m[(kj, kj)] = 2.0;
            } else if !is_diag {
                m[(kj, ki)] = corr_m(kj, ki);
                m[(ki, kj)] = corr_m(kj, ki);
            }
        }
    }
    for jp in 0..nsd {
        for ip in 0..set.n_corr() {
            let key = jp + (nsd + ip) * tot;
            let (c1, c2) = set.corr[ip];
            let sj = set.stderr[jp];
            let m = d2sig_full.entry(key).or_insert_with(|| Mat::zeros(ne, ne));
            if sj == c1 {
                m[(sj, c2)] = sigma_e[(c2, c2)].sqrt();
                m[(c2, sj)] = sigma_e[(c2, c2)].sqrt();
            }
            if sj == c2 {
                m[(sj, c1)] = sigma_e[(c1, c1)].sqrt();
                m[(c1, sj)] = sigma_e[(c1, c1)].sqrt();
            }
        }
    }
    // d2Sigma_e(:,:,indp2tottot2): dyn_vech (upper triangle, column-major) of the index matrix
    let mut d2sig = Vec::with_capacity(tot2);
    for c in 0..tot {
        for r in 0..=c {
            d2sig.push(d2sig_full.get(&(r + c * tot)).cloned().unwrap_or_else(|| Mat::zeros(ne, ne)));
        }
    }

    // ---- d2KalmanA and d2Om for every pair (j over all parameters, i <= j)
    let b = &dr.ghu;
    let dghu = &d.dghu;
    let dsig = &d.dsigma_e;
    let nvech = n * (n + 1) / 2;
    let mut d2ka = Mat::zeros(n * n, tot2);
    let mut d2om = Mat::zeros(nvech, tot2);
    let mut jcount = 0;
    let mut icount = 0;
    let m3 = |a: &Mat, s: &Mat, c: &Mat| mul(&mul(a, false, s, false), false, c, true);
    for j in 0..tot {
        for i in 0..=j {
            let y = if j < ns {
                m3(b, &d2sig[jcount], b)
            } else {
                let jind = j - ns;
                if i < ns {
                    m3(&dghu[j], &dsig[i], b) + m3(b, &dsig[i], &dghu[j])
                } else {
                    let iind = i - ns;
                    let dkaj = &x2x[icount];
                    icount += 1;
                    let p = pair(iind, jind);
                    let mut elem1 = &d2gam0[p] - mul(&d2gam1[p], false, &ka, false);
                    elem1 -= mul(&dgam1[jind], false, &dka[iind], false) + mul(&dgam1[iind], false, &dka[jind], false);
                    let elemj0 =
                        &dgam0[jind] - mul(&dgam1[jind], false, &ka, false) - mul(&gam1, false, &dka[jind], false);
                    let elemi0 =
                        &dgam0[iind] - mul(&dgam1[iind], false, &ka, false) - mul(&gam1, false, &dka[iind], false);
                    let elem0 =
                        mul(&elemj0, false, &dghu[ns + iind], false) + mul(&elemi0, false, &dghu[ns + jind], false);
                    let inner = &d2gam3[p] - elem0 - mul(&(elem1 - mul(&gam1, false, dkaj, false)), false, b, false);
                    let d2b = mul(&inv_mm, false, &inner, false);
                    for (r, v) in dkaj.iter().enumerate() {
                        d2ka[(r, jcount)] = *v;
                    }
                    m3(&d2b, sigma_e, b)
                        + m3(b, sigma_e, &d2b)
                        + m3(&dghu[ns + jind], sigma_e, &dghu[ns + iind])
                        + m3(&dghu[ns + iind], sigma_e, &dghu[ns + jind])
                }
            };
            d2om.column_mut(jcount).copy_from_slice(&super::filters::vech(&y));
            jcount += 1;
        }
    }
    // d2Yss in DR order
    let d2yss = (0..n).map(|r| Mat::from_fn(nm, nm, |a, bb| d2ys[bb][(ov[r], a)])).collect();
    Ok(SecondOrder { d2yss, d2kalman_a: d2ka, d2om })
}
