// SPDX-License-Identifier: GPL-3.0-or-later
//! Perturbation at any order (Dynare's `k_order_perturbation` at `order >= 4`).
//!
//! Same method and unknowns as the order-2/3 solver of the parent module (`g_{w^r σ^s}`, solved
//! for `s = 0..=k` at each order `k` from `A X + B X^{**} (g^*_w)^{⊗r} = -R`), but the
//! order-`k` derivatives of `h(v) = f(z(v))` are taken column by column (one sorted `v`-tuple at
//! a time) from the general Faà di Bruno formula, a sum over the set partitions of the tuple's
//! positions, with the sparse symbolic model derivatives of every order
//! (`Model::dynamic_derivatives_k`); the expected right-hand side uses every even Gaussian
//! moment `E[ε^{⊗j}]`, `j <= s` (Isserlis). The derivatives of `z` and of
//! `a(v) = (g^*(y^*, u, σ), u', σ)` on the shorter sub-tuples are memoized: they only involve
//! the orders already solved. Orders 2 and 3 keep the dedicated dense solver of the parent
//! module.

use super::{
    KOrderDr, block, sorted_tuples, store, sylvester, tensor::col_of, tensor::kron_power_apply, tensor::tuple_of,
};
use crate::linalg::Mat;
use crate::model::higher::DynamicDerivativesK;
use crate::solver::DecisionRule;
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;
use std::rc::Rc;

/// One term of a model derivative `f_{z^m}` over an ordered tuple of `z` indices.
struct FEntry {
    eq: usize,
    z: Vec<usize>,
    val: f64,
}

struct Ctx {
    n: usize,
    ns: usize,
    nx: usize,
    nw: usize,
    nz: usize,
    nc: usize,
    nf: usize,
    klag: Vec<usize>,
    kcurr: Vec<usize>,
    klead: Vec<usize>,
    /// `f[m - 1]`: every ordering of every non-zero `f_{z^m}` entry.
    f: Vec<Vec<FEntry>>,
    /// `parts[l]`: the set partitions of the positions `0..l` (blocks of sorted positions).
    parts: Vec<Rc<Vec<Vec<Vec<usize>>>>>,
    amemo: HashMap<Vec<usize>, Vec<f64>>,
    zmemo: HashMap<Vec<usize>, Vec<f64>>,
}

/// The set partitions of `0..l`.
fn set_partitions(l: usize) -> Vec<Vec<Vec<usize>>> {
    let mut out = Vec::new();
    let mut a = vec![0usize; l];
    fn rec(i: usize, mx: usize, a: &mut Vec<usize>, out: &mut Vec<Vec<Vec<usize>>>) {
        if i == a.len() {
            let nb = a.iter().max().map_or(0, |m| m + 1);
            let mut blocks = vec![Vec::new(); nb];
            for (p, &b) in a.iter().enumerate() {
                blocks[b].push(p);
            }
            out.push(blocks);
            return;
        }
        for b in 0..=mx {
            a[i] = b;
            rec(i + 1, mx.max(b + 1), a, out);
        }
    }
    if l == 0 {
        return vec![Vec::new()];
    }
    rec(1, 1, &mut a, &mut out);
    out
}

/// Distinct orderings of a sorted tuple.
fn permutations(t: &[usize]) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur = Vec::with_capacity(t.len());
    let mut used = vec![false; t.len()];
    fn rec(t: &[usize], used: &mut [bool], cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == t.len() {
            out.push(cur.clone());
            return;
        }
        for i in 0..t.len() {
            if used[i] || (i > 0 && t[i] == t[i - 1] && !used[i - 1]) {
                continue;
            }
            used[i] = true;
            cur.push(t[i]);
            rec(t, used, cur, out);
            cur.pop();
            used[i] = false;
        }
    }
    rec(t, &mut used, &mut cur, &mut out);
    out
}

fn multiplicity(t: &[usize]) -> f64 {
    let mut m: f64 = (1..=t.len()).product::<usize>() as f64;
    let mut i = 0;
    while i < t.len() {
        let j = t[i..].iter().take_while(|&&v| v == t[i]).count();
        m /= (1..=j).product::<usize>() as f64;
        i += j;
    }
    m
}

/// `E[ε_{e1} ... ε_{ej}]` for `ε ~ N(0, Σ)` (Isserlis).
fn gauss_moment(sigma: &Mat, e: &[usize]) -> f64 {
    if e.is_empty() {
        return 1.0;
    }
    if e.len() % 2 == 1 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 1..e.len() {
        let c = sigma[(e[0], e[i])];
        if c != 0.0 {
            let rest: Vec<usize> = e[1..].iter().enumerate().filter(|&(p, _)| p + 1 != i).map(|(_, &v)| v).collect();
            s += c * gauss_moment(sigma, &rest);
        }
    }
    s
}

fn binom(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

impl Ctx {
    /// `a_{v^l}` at the sorted tuple `t` (`nx` entries); `d` = the decision-rule derivatives,
    /// `k` = the order being solved (tuples of length `k` are not memoized).
    fn a_of(&mut self, d: &[Mat], k: usize, t: &[usize]) -> Vec<f64> {
        if t.len() < k
            && let Some(v) = self.amemo.get(t)
        {
            return v.clone();
        }
        let l = t.len();
        let mut a = vec![0.0; self.nx];
        if t.iter().all(|&i| i < self.nx) {
            let col = col_of(t, self.nx);
            for (i, &r) in self.klag.iter().enumerate() {
                a[i] = d[l - 1][(r, col)];
            }
        }
        if l == 1 {
            if t[0] >= self.nx {
                a[self.ns + t[0] - self.nx] = 1.0;
            } else if t[0] == self.nw {
                a[self.nw] = 1.0;
            }
        }
        if l < k {
            self.amemo.insert(t.to_vec(), a.clone());
        }
        a
    }

    /// `z_{v^l}` at the sorted tuple `t` (`nz` entries).
    fn z_of(&mut self, d: &[Mat], k: usize, t: &[usize]) -> Vec<f64> {
        if t.len() < k
            && let Some(v) = self.zmemo.get(t)
        {
            return v.clone();
        }
        let (ns, nc, nf, nx) = (self.ns, self.nc, self.nf, self.nx);
        let l = t.len();
        let mut z = vec![0.0; self.nz];
        if l == 1 {
            if t[0] < ns {
                z[t[0]] = 1.0;
            } else if t[0] < self.nw {
                z[ns + nc + nf + t[0] - ns] = 1.0;
            }
        }
        if t.iter().all(|&i| i < nx) {
            let col = col_of(t, nx);
            for (q, &r) in self.kcurr.iter().enumerate() {
                z[ns + q] = d[l - 1][(r, col)];
            }
        }
        // lead: g**(a(v)) by Faà di Bruno
        let parts = Rc::clone(&self.parts[l]);
        for p in parts.iter() {
            let avs: Vec<Vec<(usize, f64)>> = p
                .iter()
                .map(|b| {
                    let sub: Vec<usize> = b.iter().map(|&i| t[i]).collect();
                    self.a_of(d, k, &sub).into_iter().enumerate().filter(|&(_, v)| v != 0.0).collect()
                })
                .collect();
            let g = &d[p.len() - 1];
            let lead = &mut z[ns + nc..ns + nc + nf];
            for_each_combo(&avs, |xs, prod| {
                let col = col_of(xs, nx);
                for (zq, &r) in lead.iter_mut().zip(&self.klead) {
                    *zq += prod * g[(r, col)];
                }
            });
        }
        if l < k {
            self.zmemo.insert(t.to_vec(), z.clone());
        }
        z
    }

    /// `h_{v^k}` at the sorted tuple `t` (`n` entries).
    fn h_of(&mut self, d: &[Mat], k: usize, t: &[usize]) -> Vec<f64> {
        let l = t.len();
        let mut h = vec![0.0; self.n];
        let parts = Rc::clone(&self.parts[l]);
        for p in parts.iter() {
            let zs: Vec<Vec<f64>> = p
                .iter()
                .map(|b| {
                    let sub: Vec<usize> = b.iter().map(|&i| t[i]).collect();
                    self.z_of(d, k, &sub)
                })
                .collect();
            if zs.iter().any(|z| z.iter().all(|&v| v == 0.0)) {
                continue;
            }
            'e: for e in &self.f[p.len() - 1] {
                let mut prod = e.val;
                for (b, &zi) in e.z.iter().enumerate() {
                    let v = zs[b][zi];
                    if v == 0.0 {
                        continue 'e;
                    }
                    prod *= v;
                }
                h[e.eq] += prod;
            }
        }
        h
    }
}

/// Solves for the decision rule at any `order >= 2` (used at `order >= 4`). Arguments as
/// [`super::solve`], with the derivatives of every order (`deriv`, `vals = deriv.eval(..)`).
pub fn solve(
    dr1: &DecisionRule,
    lli: &[[usize; 3]],
    g1: &Mat,
    deriv: &DynamicDerivativesK,
    vals: &[Vec<f64>],
    sigma_e: &Mat,
    order: usize,
) -> Result<KOrderDr> {
    if order < 2 || deriv.order < order {
        bail!("korder::general::solve: order {order} needs the derivatives up to that order");
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
    let nw = ns + ne;
    let nx = nw + 1;
    let ov = &dr1.order_var;
    let klag: Vec<usize> = (dr1.nstatic..dr1.nstatic + ns).collect();
    let kcurr: Vec<usize> = (0..n).filter(|&k| lli[ov[k]][1] != 0).collect();
    let klead: Vec<usize> = (dr1.nstatic + dr1.npred..n).collect();
    let (nc, nf) = (kcurr.len(), klead.len());
    let nz = ns + nc + nf + ne;
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
    let mut f: Vec<Vec<FEntry>> = Vec::with_capacity(order);
    f.push(
        (0..n)
            .flat_map(|i| (0..nz).map(move |j| (i, j)))
            .filter(|&(i, j)| g1z[(i, j)] != 0.0)
            .map(|(i, j)| FEntry { eq: i, z: vec![j], val: g1z[(i, j)] })
            .collect(),
    );
    for m in 2..=order {
        let mut fm = Vec::new();
        for ((eq, cols), &val) in deriv.index[m - 2].iter().zip(&vals[m - 2]) {
            if val.is_nan() {
                bail!("korder: the higher-order derivatives of the dynamic model contain NaN (info 12)");
            }
            if val.is_infinite() {
                bail!("korder: the higher-order derivatives of the dynamic model contain Inf (info 11)");
            }
            if val == 0.0 {
                continue;
            }
            let mut z = Vec::with_capacity(m);
            for &c in cols {
                let zc = zmap[c];
                if zc == usize::MAX {
                    return Err(anyhow!("korder: dynamic column {c} is not a model variable"));
                }
                z.push(zc);
            }
            z.sort_unstable();
            for p in permutations(&z) {
                fm.push(FEntry { eq: *eq, z: p, val });
            }
        }
        f.push(fm);
    }
    let mut cx = Ctx {
        n,
        ns,
        nx,
        nw,
        nz,
        nc,
        nf,
        klag,
        kcurr,
        klead,
        f,
        parts: (0..=order).map(|l| Rc::new(set_partitions(l))).collect(),
        amemo: HashMap::new(),
        zmemo: HashMap::new(),
    };

    let mut d1 = Mat::zeros(n, nx);
    d1.columns_mut(0, ns).copy_from(&dr1.ghx);
    d1.columns_mut(ns, ne).copy_from(&dr1.ghu);
    let mut derivs = vec![d1];

    // A = f_0 + f_+ g**_y, B~ = f_+ (as the order-2/3 solver)
    let mut a = Mat::zeros(n, n);
    for (q, &k) in cx.kcurr.iter().enumerate() {
        a.set_column(k, &g1z.column(ns + q));
    }
    let fplus = g1z.columns(ns + nc, nf).into_owned();
    let ghx_lead = Mat::from_fn(nf, ns, |i, j| dr1.ghx[(cx.klead[i], j)]);
    let corr = &fplus * &ghx_lead;
    for (p, &k) in cx.klag.iter().enumerate() {
        for i in 0..n {
            a[(i, k)] += corr[(i, p)];
        }
    }
    let alu = a.clone().lu();
    let ainv_b = alu.solve(&fplus).ok_or_else(|| anyhow!("korder: singular matrix A (f_0 + f_+ g**_y)"))?;
    let s_mat = Mat::from_fn(nf, nf, |i, j| ainv_b[(cx.klead[i], j)]);
    let c_mat = Mat::from_fn(ns, ns, |i, j| dr1.ghx[(cx.klag[i], j)]);
    let gstar = Mat::from_fn(ns, nw, |i, j| derivs[0][(cx.klag[i], j)]);
    let isig = nw;

    for k in 2..=order {
        derivs.push(Mat::zeros(n, nx.pow(k as u32)));
        for s in 0..=k {
            let r = k - s;
            // R at the sorted w-tuples
            let wts = sorted_tuples(nw, r);
            let mut rs: HashMap<Vec<usize>, Vec<f64>> = HashMap::with_capacity(wts.len());
            for wt in &wts {
                let mut acc = vec![0.0; n];
                for j in (0..=s).step_by(2) {
                    let c = binom(s, j);
                    for et in sorted_tuples(ne, j) {
                        let mom = gauss_moment(sigma_e, &et) * multiplicity(&et);
                        if mom == 0.0 {
                            continue;
                        }
                        let mut t = wt.clone();
                        t.extend(std::iter::repeat_n(isig, s - j));
                        t.extend(et.iter().map(|&e| nx + e));
                        let h = cx.h_of(&derivs, k, &t);
                        for (av, hv) in acc.iter_mut().zip(h) {
                            *av += c * mom * hv;
                        }
                    }
                }
                rs.insert(wt.clone(), acc);
            }
            let ncol = nw.pow(r as u32);
            let mut rhs = Mat::zeros(n, ncol);
            let mut tup = vec![0; r];
            for c in 0..ncol {
                tuple_of(c, nw, r, &mut tup);
                let mut st = tup.clone();
                st.sort_unstable();
                let v = &rs[&st];
                for i in 0..n {
                    rhs[(i, c)] = v[i];
                }
            }
            // pure-state block -> generalized Sylvester equation, then all blocks
            let ny = ns.pow(r as u32);
            let mut dy = Mat::zeros(n, ny);
            for c in 0..ny {
                tuple_of(c, ns, r, &mut tup);
                let cw = col_of(&tup, nw);
                for i in 0..n {
                    dy[(i, c)] = -rhs[(i, cw)];
                }
            }
            let ainv_d = alu.solve(&dy).ok_or_else(|| anyhow!("korder: singular A"))?;
            let fy = Mat::from_fn(nf, ny, |i, j| ainv_d[(cx.klead[i], j)]);
            let y = sylvester::gensylv(r, &s_mat, &c_mat, &fy)?;
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

/// Calls `f(xs, prod)` for every tuple `xs` of indices drawn from the sparse vectors `avs`
/// (one index per vector), with `prod` the product of the picked values.
fn for_each_combo(avs: &[Vec<(usize, f64)>], mut f: impl FnMut(&[usize], f64)) {
    let m = avs.len();
    if avs.iter().any(|a| a.is_empty()) {
        return;
    }
    let mut ix = vec![0usize; m];
    let mut xs = vec![0usize; m];
    loop {
        let mut prod = 1.0;
        for b in 0..m {
            let (x, v) = avs[b][ix[b]];
            xs[b] = x;
            prod *= v;
        }
        f(&xs, prod);
        let mut b = m;
        loop {
            if b == 0 {
                return;
            }
            b -= 1;
            ix[b] += 1;
            if ix[b] < avs[b].len() {
                break;
            }
            ix[b] = 0;
        }
    }
}

/// `k_order_welfare` at any order: the unfolded derivatives `W_{x^k}` (`1 x nx^k`,
/// `k = 1..=order`, over `x = (y*, u, σ)` as [`KOrderDr::derivs`]) of the welfare
/// `W(x) = U(ys + g(x)) + β E_ε W(g*(x), σ ε, σ)`, solved order by order and, within an order,
/// by the number `s` of `σ` indices: the pure block `W_{y*^r σ^s}` from
/// `X - β X (g*_{y*})^{⊗r} = R` (`gensylv`), then the blocks with shocks explicitly.
/// `u_derivs[m - 1]` are the sorted derivatives of order `m` of the objective `U` over the
/// endogenous variables (declaration order, `Model::planner_objective_derivatives_k`).
pub fn k_order_welfare(
    kdr: &KOrderDr,
    u_derivs: &[Vec<(Vec<usize>, f64)>],
    beta: f64,
    sigma_e: &Mat,
) -> Result<Vec<Mat>> {
    let order = kdr.order;
    let dr = &kdr.first;
    let (ns, ne) = (dr.nspred, kdr.n_exo);
    let nw = ns + ne;
    let nx = nw + 1;
    let d = &kdr.derivs;
    let ivar = &dr.inv_order_var;
    let klag: Vec<usize> = (dr.nstatic..dr.nstatic + ns).collect();
    // every ordering of every objective derivative, rows in DR order
    let uf: Vec<Vec<(Vec<usize>, f64)>> = u_derivs
        .iter()
        .map(|um| {
            um.iter()
                .filter(|e| e.1 != 0.0)
                .flat_map(|(js, v)| {
                    let mut rows: Vec<usize> = js.iter().map(|&j| ivar[j]).collect();
                    rows.sort_unstable();
                    permutations(&rows).into_iter().map(move |p| (p, *v))
                })
                .collect()
        })
        .collect();
    let parts: Vec<Vec<Vec<Vec<usize>>>> = (0..=order).map(set_partitions).collect();
    // a(v) = (g*(x), u', σ) at the sorted v-tuple t (v = (x, u')), non-zero entries
    let a_of = |t: &[usize]| -> Vec<(usize, f64)> {
        let mut a = vec![0.0; nx];
        if t.iter().all(|&i| i < nx) {
            let col = col_of(t, nx);
            for (i, &r) in klag.iter().enumerate() {
                a[i] = d[t.len() - 1][(r, col)];
            }
        }
        if t.len() == 1 {
            if t[0] >= nx {
                a[ns + t[0] - nx] = 1.0;
            } else if t[0] == nw {
                a[nw] = 1.0;
            }
        }
        a.into_iter().enumerate().filter(|&(_, v)| v != 0.0).collect()
    };
    let sub = |t: &[usize], b: &[usize]| -> Vec<usize> { b.iter().map(|&i| t[i]).collect() };
    // [U∘(ys + g)]_{x^k} at the sorted x-tuple t
    let ug = |t: &[usize]| -> f64 {
        let mut acc = 0.0;
        for p in &parts[t.len()] {
            let cols: Vec<usize> = p.iter().map(|b| col_of(&sub(t, b), nx)).collect();
            'e: for (rows, v) in &uf[p.len() - 1] {
                let mut prod = *v;
                for (bi, &r) in rows.iter().enumerate() {
                    let g = d[p[bi].len() - 1][(r, cols[bi])];
                    if g == 0.0 {
                        continue 'e;
                    }
                    prod *= g;
                }
                acc += prod;
            }
        }
        acc
    };
    // [W∘a]_{v^k} at the sorted v-tuple t, with the welfare derivatives `w` (orders 1..=k)
    let wa = |w: &[Mat], t: &[usize]| -> f64 {
        let mut acc = 0.0;
        for p in &parts[t.len()] {
            let avs: Vec<Vec<(usize, f64)>> = p.iter().map(|b| a_of(&sub(t, b))).collect();
            let wm = &w[p.len() - 1];
            for_each_combo(&avs, |xs, prod| acc += prod * wm[(0, col_of(xs, nx))]);
        }
        acc
    };
    // R at the sorted x-tuple (w-tuple, σ^s): U∘g + β E[W∘a]
    let rhs = |w: &[Mat], wt: &[usize], s: usize| -> f64 {
        let mut t = wt.to_vec();
        t.extend(std::iter::repeat_n(nw, s));
        let mut acc = ug(&t);
        for j in (0..=s).step_by(2) {
            let c = binom(s, j);
            for et in sorted_tuples(ne, j) {
                let mom = gauss_moment(sigma_e, &et) * multiplicity(&et);
                if mom == 0.0 {
                    continue;
                }
                let mut tv = wt.to_vec();
                tv.extend(std::iter::repeat_n(nw, s - j));
                tv.extend(et.iter().map(|&e| nx + e));
                acc += beta * c * mom * wa(w, &tv);
            }
        }
        acc
    };
    let set_all = |wk: &mut Mat, t: &[usize], v: f64| {
        for p in permutations(t) {
            wk[(0, col_of(&p, nx))] = v;
        }
    };
    let gy = Mat::from_fn(ns, ns, |i, j| d[0][(klag[i], j)]);
    let s_beta = Mat::from_element(1, 1, -beta);
    let mut w: Vec<Mat> = Vec::with_capacity(order);
    for k in 1..=order {
        w.push(Mat::zeros(1, nx.pow(k as u32)));
        for s in 0..=k {
            let r = k - s;
            // pure block
            let nr = ns.pow(r as u32);
            let mut f = Mat::zeros(1, nr);
            let mut yt = vec![0; r];
            let mut cache: HashMap<Vec<usize>, f64> = HashMap::new();
            for c in 0..nr {
                tuple_of(c, ns, r, &mut yt);
                let mut st = yt.clone();
                st.sort_unstable();
                f[(0, c)] = *cache.entry(st.clone()).or_insert_with(|| rhs(&w, &st, s));
            }
            let x = if nr == 0 { Mat::zeros(1, 0) } else { sylvester::gensylv(r, &s_beta, &gy, &f)? };
            for st in sorted_tuples(ns, r) {
                let v = x[(0, col_of(&st, ns))];
                let mut t = st;
                t.extend(std::iter::repeat_n(nw, s));
                set_all(&mut w[k - 1], &t, v);
            }
            // blocks with shocks
            let vals: Vec<(Vec<usize>, f64)> = sorted_tuples(nw, r)
                .into_iter()
                .filter(|wt| wt.iter().any(|&i| i >= ns))
                .map(|wt| {
                    let v = rhs(&w, &wt, s);
                    let mut t = wt;
                    t.extend(std::iter::repeat_n(nw, s));
                    (t, v)
                })
                .collect();
            for (t, v) in vals {
                set_all(&mut w[k - 1], &t, v);
            }
        }
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitions_and_moments() {
        let bell = [1, 1, 2, 5, 15, 52, 203];
        for (l, &b) in bell.iter().enumerate() {
            assert_eq!(set_partitions(l).len(), b);
        }
        assert_eq!(permutations(&[0, 0, 1]).len(), 3);
        let s = Mat::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.0]);
        assert!((gauss_moment(&s, &[0, 0, 0, 0]) - 12.0).abs() < 1e-14); // 3 σ^4
        assert!((gauss_moment(&s, &[0, 0, 1, 1]) - (2.0 + 2.0 * 0.25)).abs() < 1e-14);
        assert!((gauss_moment(&s, &[0; 6]) - 15.0 * 8.0).abs() < 1e-12);
    }
}
