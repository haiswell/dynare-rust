// SPDX-License-Identifier: GPL-3.0-or-later
//! Higher-order (2nd and 3rd) perturbation solution (stream K).
//!
//! Computes what Dynare 7.2 stores in `oo_.dr` at `order = 2` (`dyn_second_order_solver.m`:
//! `ghxx`, `ghxu`, `ghuu`, `ghs2`) and at `order = 3` (`k_order_pert.m` / the dynare++ k-order
//! solver of Kamenik: additionally `ghxxx`, `ghxxu`, `ghxuu`, `ghuuu`, `ghxss`, `ghuss`, and the
//! folded tensors `g_0 .. g_3`).
//!
//! Method (Juillard & Kamenik 2004, Kamenik 2005). Write the decision rule as
//! `y_t - ys = g(x)`, `x = (y*_{t-1} - ys*, u_t, σ)` (`nspred` states in DR order, `n_exo`
//! shocks, the perturbation parameter), with derivative tensors `D_k = g_{x^k}` (rows in DR
//! order, unfolded, symmetric). With `v = (y*, u, σ, u')` (`u'` the shock at t+1) the
//! model is `h(v) = f(z(v))`,
//! `z = [y*; g^0(y*,u,σ); g^{**}(g^*(y*,u,σ), u', σ); u]` (the columns `kk1` of
//! `dyn_second_order_solver.m`: lagged states, current, forward, shocks), and
//! `E_{ε'} h(y*, u, σ, σ ε') = 0` identically. The order-k derivatives of `h` follow from Faà
//! di Bruno's formula with the sparse model derivatives (`model::higher`). The order-k
//! unknowns `g_{w^r σ^s}` (`w = (y*, u)`, `r + s = k`) are solved for `s = 0, 1, ..., k` in
//! turn: with every not-yet-solved order-k coefficient set to zero the expected derivative
//! `H_{w^r σ^s} = sum_j C(s,j) h_{w^r σ^{s-j} u'^j} E[ε^{⊗j}]` (Gaussian shocks: `j = 0, 2`)
//! is the right-hand side `R` of
//! `A X + B X^{**} (g^*_w)^{⊗r} = -R`, `A = f_0 + f_+ g^{**}_{y*}`, `B = f_+`,
//! a generalized Sylvester equation for the pure-state block (`gensylv`) followed by linear
//! solves with `A` for the blocks involving `u` (as `dyn_second_order_solver.m` does for
//! `ghxx`, `ghxu`, `ghuu`, `ghs2`).

pub mod pruned;
pub mod simulate;
pub mod stoch_simul;
pub mod sylvester;
pub mod tensor;

use crate::linalg::Mat;
use crate::model::Model;
use crate::model::higher::{DerivativeValues, DynamicDerivatives};
use crate::solver::{DecisionRule, FirstOrderSolver, SolverOptions};
use anyhow::{Result, anyhow, bail};
use tensor::{add_sym2, add_sym3_first, add_sym6, col_of, kron_apply, kron_power_apply, tuple_of};

/// Higher-order decision rule (`oo_.dr` at order 2 or 3). Rows in DR order (`order_var`);
/// columns of the unfolded tensors in MATLAB `kron` order over the states (`state_var`,
/// `nspred`) and shocks (`n_exo`).
#[derive(Debug, Clone)]
pub struct KOrderDr {
    pub order: usize,
    /// First-order part (`ghx`, `ghu`, `ys`, `order_var`, ...).
    pub first: DecisionRule,
    pub n_exo: usize,
    pub ghxx: Mat,
    pub ghxu: Mat,
    pub ghuu: Mat,
    /// `n x 1`.
    pub ghs2: Mat,
    pub ghxxx: Mat,
    pub ghxxu: Mat,
    pub ghxuu: Mat,
    pub ghuuu: Mat,
    pub ghxss: Mat,
    pub ghuss: Mat,
    /// `derivs[k-1] = g_{x^k}`, `n x (nspred + n_exo + 1)^k`, over `x = (y*, u, σ)`.
    pub derivs: Vec<Mat>,
}

impl KOrderDr {
    pub fn nspred(&self) -> usize {
        self.first.nspred
    }

    /// Dynare 7's k-order tensor `dr.g_j` (`j = 0..=order`): folded over `w = (y*, u)`
    /// (columns = sorted index tuples `i1 <= ... <= ij` in lexicographic order), with the
    /// Taylor factorials and the `σ = 1` terms folded in:
    /// `g_j[w-tuple] = sum_q g_{w^j σ^q} / (j! q!)`.
    pub fn g_k(&self, j: usize) -> Mat {
        fold_taylor(&self.derivs, self.order, self.nspred() + self.n_exo, j)
    }
}

/// The folded Taylor tensor `g_j` (see [`KOrderDr::g_k`]) of the unfolded derivatives
/// `derivs[k-1]` (over `x = (w, σ)`, `nw + 1` per index) of a rule of order `order`.
pub fn fold_taylor(derivs: &[Mat], order: usize, nw: usize, j: usize) -> Mat {
    let n = derivs.first().map_or(0, |d| d.nrows());
    let nx = nw + 1;
    let tuples = sorted_tuples(nw, j);
    let mut out = Mat::zeros(n, tuples.len());
    for q in 0..=(order - j.min(order)) {
        let k = j + q;
        if k == 0 || k > order {
            continue;
        }
        let fac = (factorial(j) * factorial(q)) as f64;
        let d = &derivs[k - 1];
        for (c, t) in tuples.iter().enumerate() {
            let mut idx: Vec<usize> = t.clone();
            idx.extend(std::iter::repeat_n(nw, q));
            let col = col_of(&idx, nx);
            for i in 0..n {
                out[(i, c)] += d[(i, col)] / fac;
            }
        }
    }
    out
}

fn factorial(k: usize) -> usize {
    (1..=k).product()
}

/// Sorted tuples `i1 <= ... <= ik` over `0..m`, lexicographic (dynare++ folded order).
pub fn sorted_tuples(m: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur = Vec::with_capacity(k);
    fn rec(m: usize, k: usize, start: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in start..m {
            cur.push(i);
            rec(m, k, i, cur, out);
            cur.pop();
        }
    }
    rec(m, k, 0, &mut cur, &mut out);
    out
}

/// Sparse model derivative entry over the `z` variables.
struct Entry<const K: usize> {
    eq: usize,
    z: [usize; K],
    val: f64,
}

/// Problem data shared by the steps of the solver.
struct Setup<'a> {
    n: usize,
    ns: usize,
    ne: usize,
    nx: usize,
    nv: usize,
    nz: usize,
    klag: Vec<usize>,
    kcurr: Vec<usize>,
    klead: Vec<usize>,
    g1z: Mat,
    e2: Vec<Entry<2>>,
    e3: Vec<Entry<3>>,
    sigma_e: &'a Mat,
}

/// Solves for the order-2 or order-3 decision rule.
///
/// `g1`: dynamic Jacobian at the steady state (`Model::dynamic_jacobian` layout); `deriv` /
/// `vals`: its 2nd/3rd derivatives (`Model::dynamic_derivatives`, evaluated at the steady
/// state); `dr1`: the first-order solution; `sigma_e`: `M_.Sigma_e`.
pub fn solve(
    dr1: &DecisionRule,
    lli: &[[usize; 3]],
    g1: &Mat,
    deriv: &DynamicDerivatives,
    vals: &DerivativeValues,
    sigma_e: &Mat,
    order: usize,
) -> Result<KOrderDr> {
    if !(2..=3).contains(&order) {
        bail!("korder::solve: order must be 2 or 3 (got {order})");
    }
    if order == 3 && deriv.order < 3 {
        bail!("korder::solve: third derivatives were not computed");
    }
    let n = dr1.ghx.nrows();
    let ns = dr1.nspred;
    let ne = dr1.ghu.ncols();
    if dr1.nsfwrd == 0 {
        bail!("2nd and 3rd order approximation not implemented for purely backward models");
    }
    if ns == 0 {
        bail!("2nd and 3rd order approximation not implemented for purely forward models");
    }
    let nx = ns + ne + 1;
    let nv = nx + ne;
    let ov = &dr1.order_var;
    let klag: Vec<usize> = (dr1.nstatic..dr1.nstatic + ns).collect();
    let kcurr: Vec<usize> = (0..n).filter(|&k| lli[ov[k]][1] != 0).collect();
    let klead: Vec<usize> = (dr1.nstatic + dr1.npred..n).collect();
    let (nc, nf) = (kcurr.len(), klead.len());
    let nz = ns + nc + nf + ne;
    // dynamic-Jacobian column -> z index
    let mut zmap = vec![usize::MAX; 3 * n + ne];
    for (p, &k) in klag.iter().enumerate() {
        zmap[ov[k]] = p;
    }
    for (q, &k) in kcurr.iter().enumerate() {
        zmap[n + ov[k]] = ns + q;
    }
    for (p, &k) in klead.iter().enumerate() {
        zmap[2 * n + ov[k]] = ns + nc + p;
    }
    for e in 0..ne {
        zmap[3 * n + e] = ns + nc + nf + e;
    }
    let zof = |c: usize| -> Result<usize> {
        let z = zmap[c];
        if z == usize::MAX { Err(anyhow!("korder: dynamic column {c} is not a model variable")) } else { Ok(z) }
    };
    let mut g1z = Mat::zeros(n, nz);
    for c in 0..g1.ncols() {
        if zmap[c] != usize::MAX {
            for i in 0..n {
                g1z[(i, zmap[c])] = g1[(i, c)];
            }
        } else if g1.column(c).iter().any(|&v| v != 0.0) {
            bail!("korder: dynamic Jacobian column {c} is not a model variable");
        }
    }
    let mut e2 = Vec::with_capacity(vals.g2.len());
    for (&(eq, a, b), &val) in deriv.g2_index.iter().zip(&vals.g2) {
        let mut z = [zof(a)?, zof(b)?];
        z.sort_unstable();
        e2.push(Entry { eq, z, val });
    }
    let mut e3 = Vec::with_capacity(vals.g3.len());
    if order >= 3 {
        for (&(eq, a, b, c), &val) in deriv.g3_index.iter().zip(&vals.g3) {
            let mut z = [zof(a)?, zof(b)?, zof(c)?];
            z.sort_unstable();
            e3.push(Entry { eq, z, val });
        }
    }
    for v in vals.g2.iter().chain(&vals.g3) {
        if v.is_nan() {
            bail!("korder: the higher-order derivatives of the dynamic model contain NaN (info 12)");
        }
        if v.is_infinite() {
            bail!("korder: the higher-order derivatives of the dynamic model contain Inf (info 11)");
        }
    }
    let su = Setup { n, ns, ne, nx, nv, nz, klag, kcurr, klead, g1z, e2, e3, sigma_e };

    // first-order tensor over x = (y*, u, σ)
    let mut d1 = Mat::zeros(n, nx);
    d1.columns_mut(0, ns).copy_from(&dr1.ghx);
    d1.columns_mut(ns, ne).copy_from(&dr1.ghu);
    let mut derivs = vec![d1];

    // A = f_0 + f_+ g**_y (columns in DR order), B~ = f_+ (n x nf)
    let mut a = Mat::zeros(n, n);
    for (q, &k) in su.kcurr.iter().enumerate() {
        a.set_column(k, &su.g1z.column(ns + q));
    }
    let fplus = su.g1z.columns(ns + nc, nf).into_owned();
    let ghx_lead = Mat::from_fn(nf, ns, |i, j| dr1.ghx[(su.klead[i], j)]);
    let corr = &fplus * &ghx_lead;
    for (p, &k) in su.klag.iter().enumerate() {
        for i in 0..n {
            a[(i, k)] += corr[(i, p)];
        }
    }
    let alu = a.clone().lu();
    let ainv_b = alu.solve(&fplus).ok_or_else(|| anyhow!("korder: singular matrix A (f_0 + f_+ g**_y)"))?;
    let s_mat = Mat::from_fn(nf, nf, |i, j| ainv_b[(su.klead[i], j)]);
    let c_mat = Mat::from_fn(ns, ns, |i, j| dr1.ghx[(su.klag[i], j)]);
    let nw = ns + ne;
    let gstar = Mat::from_fn(ns, nw, |i, j| derivs[0][(su.klag[i], j)]);

    for k in 2..=order {
        derivs.push(Mat::zeros(n, nx.pow(k as u32)));
        for s in 0..=k {
            let r = k - s;
            let z = build_z(&su, &derivs, k);
            let h = compose_model(&su, &z, k);
            let rhs = expected_rhs(&su, &h, k, r, s);
            // pure-state block -> generalized Sylvester equation
            let ny = ns.pow(r as u32);
            let mut dy = Mat::zeros(n, ny);
            let mut tup = vec![0; r];
            for c in 0..ny {
                tuple_of(c, ns, r, &mut tup);
                let cw = col_of(&tup, nw);
                for i in 0..n {
                    dy[(i, c)] = -rhs[(i, cw)];
                }
            }
            let ainv_d = alu.solve(&dy).ok_or_else(|| anyhow!("korder: singular A"))?;
            let f = Mat::from_fn(nf, ny, |i, j| ainv_d[(su.klead[i], j)]);
            let y = sylvester::gensylv(r, &s_mat, &c_mat, &f)?;
            // all blocks: A X = -R - B~ Y (g*_w)^{⊗r}
            let yw = kron_power_apply(&y, &gstar, r);
            let mut full = -&rhs - &fplus * &yw;
            full = alu.solve(&full).ok_or_else(|| anyhow!("korder: singular A"))?;
            store(&mut derivs[k - 1], &full, k, s, nx, nw);
        }
    }

    let ext = |k: usize, sets: &[(usize, usize)]| block(&derivs[k - 1], nx, sets);
    let (y, u, sg) = ((0, ns), (ns, ne), (nw, 1));
    let empty = || Mat::zeros(0, 0);
    let o3 = order >= 3;
    Ok(KOrderDr {
        order,
        first: dr1.clone(),
        n_exo: ne,
        ghxx: ext(2, &[y, y]),
        ghxu: ext(2, &[y, u]),
        ghuu: ext(2, &[u, u]),
        ghs2: ext(2, &[sg, sg]),
        ghxxx: if o3 { ext(3, &[y, y, y]) } else { empty() },
        ghxxu: if o3 { ext(3, &[y, y, u]) } else { empty() },
        ghxuu: if o3 { ext(3, &[y, u, u]) } else { empty() },
        ghuuu: if o3 { ext(3, &[u, u, u]) } else { empty() },
        ghxss: if o3 { ext(3, &[y, sg, sg]) } else { empty() },
        ghuss: if o3 { ext(3, &[u, sg, sg]) } else { empty() },
        derivs,
    })
}

/// Columns of tensor `d` (index size `m`) for the product of index ranges `sets`.
fn block(d: &Mat, m: usize, sets: &[(usize, usize)]) -> Mat {
    let ncols: usize = sets.iter().map(|s| s.1).product();
    let k = sets.len();
    let lens: Vec<usize> = sets.iter().map(|s| s.1).collect();
    let mut out = Mat::zeros(d.nrows(), ncols);
    let mut idx = vec![0; k];
    for c in 0..ncols {
        let mut cc = c;
        for l in (0..k).rev() {
            idx[l] = sets[l].0 + cc % lens[l];
            cc /= lens[l];
        }
        out.set_column(c, &d.column(col_of(&idx, m)));
    }
    out
}

/// Writes the solution `x` (columns over `w^r`) into every column of `d` (over `x^k`) whose
/// tuple has exactly `s` σ indices.
fn store(d: &mut Mat, x: &Mat, k: usize, s: usize, nx: usize, nw: usize) {
    let mut t = vec![0; k];
    let mut w = Vec::with_capacity(k);
    for c in 0..d.ncols() {
        tuple_of(c, nx, k, &mut t);
        w.clear();
        w.extend(t.iter().copied().filter(|&i| i < nw));
        if k - w.len() == s {
            d.set_column(c, &x.column(col_of(&w, nw)));
        }
    }
}

/// Re-indexes a tensor over `x` (size `nx`) as a tensor over `v` (size `nv`, `x` first).
fn embed(x: &Mat, k: usize, nx: usize, nv: usize) -> Mat {
    let mut out = Mat::zeros(x.nrows(), nv.pow(k as u32));
    let mut t = vec![0; k];
    for c in 0..x.ncols() {
        tuple_of(c, nx, k, &mut t);
        out.set_column(col_of(&t, nv), &x.column(c));
    }
    out
}

fn rows(m: &Mat, idx: &[usize]) -> Mat {
    Mat::from_fn(idx.len(), m.ncols(), |i, j| m[(idx[i], j)])
}

/// Faà di Bruno for a dense outer function: derivatives of `F(a(v))` up to order `k`,
/// given `F_l` (`p x m^l`) and `a_l` (`m x nv^l`).
fn compose_dense(f: &[Mat], a: &[Mat], k: usize) -> Vec<Mat> {
    let nv = a[0].ncols();
    let mut h = vec![&f[0] * &a[0]];
    if k >= 2 {
        h.push(kron_apply(&f[1], &[&a[0], &a[0]]) + &f[0] * &a[1]);
    }
    if k >= 3 {
        let mut h3 = kron_apply(&f[2], &[&a[0], &a[0], &a[0]]) + &f[0] * &a[2];
        let t = kron_apply(&f[1], &[&a[0], &a[1]]);
        add_sym3_first(&mut h3, &t, nv);
        h.push(h3);
    }
    h
}

/// The derivatives `z_{v^l}` (`nz x nv^l`, `l = 1..=k`) of the model arguments.
fn build_z(su: &Setup, d: &[Mat], k: usize) -> Vec<Mat> {
    let (ns, ne, nx, nv, nz) = (su.ns, su.ne, su.nx, su.nv, su.nz);
    let (nc, nf) = (su.kcurr.len(), su.klead.len());
    // a(v) = (g*(y*,u,σ), u', σ) over x
    let mut a = Vec::with_capacity(k);
    for l in 1..=k {
        let mut al = Mat::zeros(nx, nv.pow(l as u32));
        let gs = embed(&rows(&d[l - 1], &su.klag), l, nx, nv);
        al.rows_mut(0, ns).copy_from(&gs);
        if l == 1 {
            for e in 0..ne {
                al[(ns + e, nx + e)] = 1.0;
            }
            al[(ns + ne, ns + ne)] = 1.0;
        }
        a.push(al);
    }
    let flead: Vec<Mat> = (0..k).map(|l| rows(&d[l], &su.klead)).collect();
    let q = compose_dense(&flead, &a, k);
    let mut z = Vec::with_capacity(k);
    for l in 1..=k {
        let mut zl = Mat::zeros(nz, nv.pow(l as u32));
        if l == 1 {
            for i in 0..ns {
                zl[(i, i)] = 1.0;
            }
            for e in 0..ne {
                zl[(ns + nc + nf + e, ns + e)] = 1.0;
            }
        }
        let cur = embed(&rows(&d[l - 1], &su.kcurr), l, nx, nv);
        zl.rows_mut(ns, nc).copy_from(&cur);
        zl.rows_mut(ns + nc, nf).copy_from(&q[l - 1]);
        z.push(zl);
    }
    z
}

/// Order-`k` derivative of `h(v) = f(z(v))` with the sparse model derivatives.
fn compose_model(su: &Setup, z: &[Mat], k: usize) -> Mat {
    let (n, nv) = (su.n, su.nv);
    let z1 = &z[0];
    let z1rows: Vec<Vec<(usize, f64)>> =
        (0..su.nz).map(|i| (0..nv).filter(|&a| z1[(i, a)] != 0.0).map(|a| (a, z1[(i, a)])).collect()).collect();
    match k {
        2 => {
            let mut t = Mat::zeros(n, nv * nv);
            for e in &su.e2 {
                let [i, j] = e.z;
                let w = if i == j { 0.5 * e.val } else { e.val };
                for &(a, za) in &z1rows[i] {
                    for &(b, zb) in &z1rows[j] {
                        t[(e.eq, a * nv + b)] += w * za * zb;
                    }
                }
            }
            let mut h = &su.g1z * &z[1];
            add_sym2(&mut h, &t, nv);
            h
        }
        3 => {
            let nv3 = nv * nv * nv;
            let mut t3 = Mat::zeros(n, nv3);
            for e in &su.e3 {
                let [i, j, l] = e.z;
                let s = if i == j && j == l {
                    6.0
                } else if i == j || j == l {
                    2.0
                } else {
                    1.0
                };
                let w = e.val / s;
                for &(a, za) in &z1rows[i] {
                    for &(b, zb) in &z1rows[j] {
                        let wab = w * za * zb;
                        for &(c, zc) in &z1rows[l] {
                            t3[(e.eq, (a * nv + b) * nv + c)] += wab * zc;
                        }
                    }
                }
            }
            let mut t2 = Mat::zeros(n, nv3);
            let z2 = &z[1];
            for e in &su.e2 {
                let [i, j] = e.z;
                for (p, q) in if i == j { vec![(i, j)] } else { vec![(i, j), (j, i)] } {
                    for &(a, za) in &z1rows[p] {
                        let w = e.val * za;
                        for bc in 0..nv * nv {
                            let v = z2[(q, bc)];
                            if v != 0.0 {
                                t2[(e.eq, a * nv * nv + bc)] += w * v;
                            }
                        }
                    }
                }
            }
            let mut h = &su.g1z * &z[2];
            add_sym6(&mut h, &t3, nv);
            add_sym3_first(&mut h, &t2, nv);
            h
        }
        _ => unreachable!(),
    }
}

/// `R[:, w-tuple] = sum_j C(s,j) h[:, (w-tuple, σ^{s-j}, u'^j)] E[ε^{⊗j}]`, `j ∈ {0, 2}`.
fn expected_rhs(su: &Setup, h: &Mat, k: usize, r: usize, s: usize) -> Mat {
    let (n, ns, ne, nx, nv) = (su.n, su.ns, su.ne, su.nx, su.nv);
    let nw = ns + ne;
    let isig = ns + ne;
    let ncol = nw.pow(r as u32);
    let mut out = Mat::zeros(n, ncol);
    let mut wt = vec![0; r];
    let mut idx = vec![0; k];
    for c in 0..ncol {
        tuple_of(c, nw, r, &mut wt);
        idx[..r].copy_from_slice(&wt);
        for l in r..k {
            idx[l] = isig;
        }
        out.set_column(c, &h.column(col_of(&idx, nv)));
        if s >= 2 {
            let coef = (s * (s - 1) / 2) as f64;
            for e1 in 0..ne {
                for e2 in 0..ne {
                    let sig = su.sigma_e[(e1, e2)];
                    if sig == 0.0 {
                        continue;
                    }
                    idx[k - 2] = nx + e1;
                    idx[k - 1] = nx + e2;
                    let hc = h.column(col_of(&idx, nv));
                    for i in 0..n {
                        out[(i, c)] += coef * sig * hc[i];
                    }
                }
            }
        }
    }
    out
}

/// Convenience: steady state `ys` and parameters given, solves the model at `order`
/// (first order through [`FirstOrderSolver`]).
pub fn solve_model(
    model: &Model,
    ys: &[f64],
    params: &[f64],
    sigma_e: &Mat,
    order: usize,
    opts: SolverOptions,
) -> Result<KOrderDr> {
    solve_model_at(model, ys, &vec![0.0; model.n_exo()], params, sigma_e, order, opts)
}

/// [`solve_model`] with the derivatives taken at the exogenous steady state `x` (stochastic_solvers.m
/// evaluates them at `oo_.exo_steady_state`).
pub fn solve_model_at(
    model: &Model,
    ys: &[f64],
    x: &[f64],
    params: &[f64],
    sigma_e: &Mat,
    order: usize,
    opts: SolverOptions,
) -> Result<KOrderDr> {
    let x = x.to_vec();
    let g1 = model.dynamic_jacobian(ys, &x, params);
    let fo = FirstOrderSolver::new(&model.lead_lag_incidence, model.n_exo(), opts);
    let dr1 = fo.solve(&g1, ys).map_err(|e| anyhow!("{e}"))?;
    model.check_external_order(order, "k_order_solver")?;
    let deriv = model.dynamic_derivatives(order);
    let vals = deriv.eval(ys, &x, params);
    solve(&dr1, &model.lead_lag_incidence, &g1, &deriv, &vals, sigma_e, order)
}
