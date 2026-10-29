// SPDX-License-Identifier: GPL-3.0-or-later
//! Simulation with a higher-order decision rule: port of `simult_.m` (order 2, pruned or
//! not; order 3 pruned, Andreasen et al. 2018), of `k_order_simul` (order >= 3 without
//! pruning: the full Taylor expansion of the k-order rule at `σ = 1`; order >= 4 pruned), and of `irf.m` at
//! order > 1 (generalized impulse responses as averages over simulated shock paths).

use super::KOrderDr;
use crate::linalg::Mat;
use rand::Rng;
use rand_distr::StandardNormal;

fn kron(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for &x in a {
        for &y in b {
            out.push(x * y);
        }
    }
    out
}

fn mv(m: &Mat, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; m.nrows()];
    for (j, &x) in v.iter().enumerate() {
        if x != 0.0 {
            for (i, o) in out.iter_mut().enumerate() {
                *o += m[(i, j)] * x;
            }
        }
    }
    out
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// `simult_(M_, options_, y0, dr, ex_, order)` for `order = dr.order` (2 or 3) and a model
/// with `maximum_lag = 1`: `y0` (declaration order) is the initial value, `ex` is
/// `periods x n_exo`. Returns `n x (periods + 1)` in declaration order, column 0 = `y0`.
pub fn simult(dr: &KOrderDr, y0: &[f64], ex: &Mat, pruning: bool) -> Mat {
    let d = &dr.first;
    let n = d.ghx.nrows();
    let ns = d.nspred;
    let ov = &d.order_var;
    let iter = ex.nrows();
    let ipred: Vec<usize> = (d.nstatic..d.nstatic + ns).collect();
    let ys_dr: Vec<f64> = ov.iter().map(|&j| d.ys[j]).collect();
    let mut out = Mat::zeros(n, iter + 1);
    for i in 0..n {
        out[(i, 0)] = y0[i];
    }
    let shock = |t: usize| -> Vec<f64> { (0..ex.ncols()).map(|e| ex[(t, e)]).collect() };
    let put = |out: &mut Mat, t: usize, ydr: &[f64]| {
        for (k, &j) in ov.iter().enumerate() {
            out[(j, t)] = ydr[k];
        }
    };
    let state_dev = |y: &[f64]| -> Vec<f64> { ipred.iter().map(|&k| y[ov[k]] - ys_dr[k]).collect() };
    match (dr.order, pruning) {
        (2, _) => {
            let constant: Vec<f64> = (0..n).map(|k| ys_dr[k] + 0.5 * dr.ghs2[(k, 0)]).collect();
            let mut yfirst: Vec<f64> = y0.to_vec(); // y__ of simult_ (declaration order)
            for t in 1..=iter {
                let prev: Vec<f64> = (0..n).map(|i| out[(i, t - 1)]).collect();
                let yhat2 = state_dev(&prev);
                let yhat1 = if pruning { state_dev(&yfirst) } else { yhat2.clone() };
                let eps = shock(t - 1);
                let mut y = constant.clone();
                axpy(&mut y, 1.0, &mv(&d.ghx, &yhat2));
                axpy(&mut y, 1.0, &mv(&d.ghu, &eps));
                axpy(&mut y, 0.5, &mv(&dr.ghxx, &kron(&yhat1, &yhat1)));
                axpy(&mut y, 0.5, &mv(&dr.ghuu, &kron(&eps, &eps)));
                axpy(&mut y, 1.0, &mv(&dr.ghxu, &kron(&yhat1, &eps)));
                put(&mut out, t, &y);
                if pruning {
                    let mut y1 = ys_dr.clone();
                    axpy(&mut y1, 1.0, &mv(&d.ghx, &yhat1));
                    axpy(&mut y1, 1.0, &mv(&d.ghu, &eps));
                    for (k, &j) in ov.iter().enumerate() {
                        yfirst[j] = y1[k];
                    }
                }
            }
        }
        (3, true) => {
            let mut yhat1 = state_dev(y0);
            let mut yhat2 = vec![0.0; ns];
            let mut yhat3 = vec![0.0; ns];
            for t in 1..=iter {
                let u = shock(t - 1);
                let gyy = mv(&dr.ghxx, &kron(&yhat1, &yhat1));
                let guu = mv(&dr.ghuu, &kron(&u, &u));
                let gyu = mv(&dr.ghxu, &kron(&yhat1, &u));
                let gyy12 = mv(&dr.ghxx, &kron(&yhat1, &yhat2));
                let gy2u = mv(&dr.ghxu, &kron(&yhat2, &u));
                let y2a = kron(&yhat1, &yhat1);
                let gyyy = mv(&dr.ghxxx, &kron(&y2a, &yhat1));
                let u2a = kron(&u, &u);
                let guuu = mv(&dr.ghuuu, &kron(&u2a, &u));
                let yu = kron(&yhat1, &u);
                let gyyu = mv(&dr.ghxxu, &kron(&yhat1, &yu));
                let gyuu = mv(&dr.ghxuu, &kron(&yu, &u));
                let gxss = mv(&dr.ghxss, &yhat1);
                let guss = mv(&dr.ghuss, &u);
                let mut y3 = mv(&d.ghx, &yhat3);
                for k in 0..n {
                    y3[k] +=
                        gyy12[k] + gy2u[k] + (gyyy[k] + guuu[k] + 3.0 * (gyyu[k] + gyuu[k] + gxss[k] + guss[k])) / 6.0;
                }
                let mut y2 = mv(&d.ghx, &yhat2);
                for k in 0..n {
                    y2[k] += 0.5 * (gyy[k] + guu[k] + 2.0 * gyu[k] + dr.ghs2[(k, 0)]);
                }
                let mut y1 = mv(&d.ghx, &yhat1);
                axpy(&mut y1, 1.0, &mv(&d.ghu, &u));
                let y: Vec<f64> = (0..n).map(|k| ys_dr[k] + y1[k] + y2[k] + y3[k]).collect();
                put(&mut out, t, &y);
                yhat1 = ipred.iter().map(|&k| y1[k]).collect();
                yhat2 = ipred.iter().map(|&k| y2[k]).collect();
                yhat3 = ipred.iter().map(|&k| y3[k]).collect();
            }
        }
        (o, false) if o >= 3 => {
            // k_order_simul: y_t - ys = sum_k sum_{j+q=k} g_{w^j σ^q} w^{⊗j} / (j! q!)
            let g: Vec<Mat> = (0..=o).map(|j| dr.g_k(j)).collect();
            let nw = ns + dr.n_exo;
            let tuples: Vec<Vec<Vec<usize>>> = (0..=o).map(|j| super::sorted_tuples(nw, j)).collect();
            for t in 1..=iter {
                let prev: Vec<f64> = (0..n).map(|i| out[(i, t - 1)]).collect();
                let mut w = state_dev(&prev);
                w.extend(shock(t - 1));
                let mut y = ys_dr.clone();
                for j in 0..=o {
                    for (c, tup) in tuples[j].iter().enumerate() {
                        // folded tensor: each unique tuple times its multiplicity
                        let mult = multiplicity(tup) as f64;
                        let prod: f64 = tup.iter().map(|&i| w[i]).product();
                        if prod != 0.0 {
                            for k in 0..n {
                                y[k] += g[j][(k, c)] * mult * prod;
                            }
                        }
                    }
                }
                put(&mut out, t, &y);
            }
        }
        (o, true) if o >= 4 => {
            // k_order_simul's simulate_pruning (Andreasen et al. 2018 at any order): the order-q
            // component x_q sums, over the columns of g_l = D_l / l! (over (y*, u, σ), σ = 1),
            // the products where the state indices carry components whose orders add up to
            // q - (number of shock and σ indices)
            let nw = ns + dr.n_exo;
            let nx = nw + 1;
            let fact: Vec<f64> = (0..=o).map(|l| (1..=l).product::<usize>() as f64).collect();
            let tuples: Vec<Vec<Vec<usize>>> = (0..=o).map(|l| super::sorted_tuples(nx, l)).collect();
            let mut phat = vec![vec![0.0; ns]; o + 1]; // phat[q] (q = 1..=o)
            phat[1] = state_dev(y0);
            for t in 1..=iter {
                let u = shock(t - 1);
                let mut psim = vec![vec![0.0; n]; o + 1];
                for l in 1..=o {
                    let d = &dr.derivs[l - 1];
                    for tup in &tuples[l] {
                        let states: Vec<usize> = tup.iter().copied().filter(|&i| i < ns).collect();
                        let mut base = multiplicity(tup) as f64 / fact[l];
                        for &i in tup.iter().filter(|&&i| i >= ns && i < nw) {
                            base *= u[i - ns];
                        }
                        if base == 0.0 {
                            continue;
                        }
                        let col = super::tensor::col_of(tup, nx);
                        let nonstate = l - states.len();
                        for q in l.max(1)..=o {
                            let m = q - nonstate;
                            let f = if states.is_empty() {
                                if q == l { 1.0 } else { 0.0 }
                            } else {
                                compositions_sum(&states, m, &phat)
                            };
                            if f != 0.0 {
                                for k in 0..n {
                                    psim[q][k] += d[(k, col)] * base * f;
                                }
                            }
                        }
                    }
                }
                let mut y = ys_dr.clone();
                for q in 1..=o {
                    axpy(&mut y, 1.0, &psim[q]);
                    phat[q] = ipred.iter().map(|&k| psim[q][k]).collect();
                }
                put(&mut out, t, &y);
            }
        }
        _ => panic!("korder::simult: unsupported order {}", dr.order),
    }
    out
}

/// `sum over the compositions m = c_1 + ... + c_s (c_i >= 1) of prod_i phat[c_i][states[i]]`.
fn compositions_sum(states: &[usize], m: usize, phat: &[Vec<f64>]) -> f64 {
    let s = states.len();
    if s == 0 {
        return if m == 0 { 1.0 } else { 0.0 };
    }
    if m < s {
        return 0.0;
    }
    let (first, rest) = (states[0], &states[1..]);
    let mut acc = 0.0;
    for c in 1..=(m - rest.len()) {
        let v = phat.get(c).map_or(0.0, |p| p[first]);
        if v != 0.0 {
            acc += v * compositions_sum(rest, m - c, phat);
        }
    }
    acc
}

/// Number of distinct orderings of a sorted tuple.
fn multiplicity(t: &[usize]) -> usize {
    let k = t.len();
    let mut m: usize = (1..=k).product();
    let mut i = 0;
    while i < k {
        let mut j = i;
        while j < k && t[j] == t[i] {
            j += 1;
        }
        m /= (1..=(j - i)).product::<usize>();
        i = j;
    }
    m
}

/// Port of `irf.m` for order > 1: the generalized IRF to the impulse `e1` (n_exo), averaged over
/// the shock paths `draws` (each `(long + drop) x n_exo`, already scaled by `chol(Sigma_e)`),
/// starting at the steady state. Returns `n x long` (declaration order).
pub fn girf(dr: &KOrderDr, e1: &[f64], long: usize, drop: usize, draws: &[Mat], pruning: bool) -> Mat {
    let n = dr.first.ghx.nrows();
    let mut acc = Mat::zeros(n, long);
    let ys = &dr.first.ys;
    for ex1 in draws {
        let mut ex2 = ex1.clone();
        for (e, &v) in e1.iter().enumerate() {
            ex2[(drop, e)] += v;
        }
        let y1 = simult(dr, ys, ex1, pruning);
        let y2 = simult(dr, ys, &ex2, pruning);
        for t in 0..long {
            for i in 0..n {
                acc[(i, t)] += y2[(i, 1 + drop + t)] - y1[(i, 1 + drop + t)];
            }
        }
    }
    acc / draws.len().max(1) as f64
}

/// Draws `replic` shock paths as `irf.m` does (`randn(long+drop, nxs) * chol(Sigma_e(i,i))` on
/// the shocks with non-zero variance), from `rng` (not MATLAB's stream).
pub fn draw_shocks<R: Rng>(sigma_e: &Mat, long: usize, drop: usize, replic: usize, rng: &mut R) -> Vec<Mat> {
    let ne = sigma_e.nrows();
    let idx: Vec<usize> = (0..ne).filter(|&i| sigma_e[(i, i)] != 0.0).collect();
    let sub = Mat::from_fn(idx.len(), idx.len(), |i, j| sigma_e[(idx[i], idx[j])]);
    let chol = sub.cholesky().map(|c| c.l().transpose()).unwrap_or_else(|| Mat::zeros(idx.len(), idx.len()));
    (0..replic)
        .map(|_| {
            let z = Mat::from_fn(long + drop, idx.len(), |_, _| rng.sample::<f64, _>(StandardNormal));
            let zs = z * &chol;
            let mut ex = Mat::zeros(long + drop, ne);
            for (c, &i) in idx.iter().enumerate() {
                ex.set_column(i, &zs.column(c));
            }
            ex
        })
        .collect()
}

/// `sum_j g_j w^{⊗j}` for the folded tensors `g[j]` (`n x` sorted `j`-tuples over `w`),
/// `tuples[j] = sorted_tuples(w.len(), j)`, added to `y`.
fn add_folded(y: &mut [f64], g: &[Mat], tuples: &[Vec<Vec<usize>>], w: &[f64]) {
    for (j, gj) in g.iter().enumerate() {
        for (c, tup) in tuples[j].iter().enumerate() {
            let prod: f64 = tup.iter().map(|&i| w[i]).product();
            if prod != 0.0 {
                let f = multiplicity(tup) as f64 * prod;
                for (k, yk) in y.iter_mut().enumerate() {
                    *yk += gj[(k, c)] * f;
                }
            }
        }
    }
}

/// Port of the `k_order_mean` MEX (Dynare 7.2, as `evaluate_planner_objective.m` calls it):
/// simulates the folded k-order rule `g` (`g[j]`, rows in DR order, `nstatic` static rows then
/// the `nspred` states) without pruning, from the state deviation `ystart` (DR order) with
/// the shocks `shocks` (`n_exo x T`). Returns the mean (MATLAB's first output, see below) and
/// the simulated levels (`n x T`, column `t` = period `t + 1`). The mean skips the first period
/// and divides by `T - drop`: `sum_{t = max(drop, 1)}^{T-1} y_t / (T - drop)` (0-based columns),
/// as the MEX does (checked against MATLAB for drop = 0..3).
pub fn k_order_mean(
    g: &[Mat],
    nstatic: usize,
    nspred: usize,
    ysteady: &[f64],
    ystart: &[f64],
    shocks: &Mat,
    drop: usize,
) -> (Vec<f64>, Mat) {
    let n = ysteady.len();
    let ne = shocks.nrows();
    let periods = shocks.ncols();
    let tuples: Vec<Vec<Vec<usize>>> = (0..g.len()).map(|j| super::sorted_tuples(nspred + ne, j)).collect();
    let mut sim = Mat::zeros(n, periods);
    let mut dev: Vec<f64> = (0..nspred).map(|k| ystart[nstatic + k]).collect();
    for t in 0..periods {
        let mut w = dev.clone();
        w.extend((0..ne).map(|e| shocks[(e, t)]));
        let mut y = ysteady.to_vec();
        add_folded(&mut y, g, &tuples, &w);
        for (i, v) in y.iter().enumerate() {
            sim[(i, t)] = *v;
        }
        dev = (0..nspred).map(|k| y[nstatic + k] - ysteady[nstatic + k]).collect();
    }
    let mut mean = vec![0.0; n];
    for t in drop.max(1)..periods {
        for (i, m) in mean.iter_mut().enumerate() {
            *m += sim[(i, t)];
        }
    }
    let den = periods as f64 - drop as f64;
    mean.iter_mut().for_each(|m| *m /= den);
    (mean, sim)
}
