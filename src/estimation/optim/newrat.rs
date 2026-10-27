// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `optimization/newrat.m` (`mode_compute = 5`) with its helpers
//! `mr_hessian.m` and `mr_gstep.m`.

#![allow(clippy::needless_range_loop)] // index-for-index port of the MATLAB loops

use rayon::prelude::*;

use super::{OptimResult, bfgsi1, csminit1, norm, pen};
use crate::estimation::{Bounds, Objective, penalty_objective};
use crate::linalg::{Mat, solve, sym_eig};

/// `hess_info` of `newrat.m` / `mr_hessian.m`.
#[derive(Clone, Debug)]
pub struct HessInfo {
    /// `options_.gstep` (not used by `mr_hessian.m` in Dynare 7.2, kept for completeness).
    pub gstep: [f64; 2],
    /// `newrat.tolerance.gstep` (NaN = use `ftol`).
    pub htol: f64,
    /// `newrat.tolerance.gstep_rel` (NaN = absolute tolerance).
    pub htol_rel: f64,
    /// Finite-difference steps, initially `gradient_epsilon`.
    pub h1: Vec<f64>,
    /// `newrat.robust`.
    pub robust: bool,
}

/// Options of `mode_compute = 5`, defaults from `default_option_values.m`
/// as set up by `dynare_minimize_objective.m`.
#[derive(Clone, Debug)]
pub struct NewratOptions {
    /// `options_.newrat.tolerance.f` (1e-5).
    pub ftol: f64,
    /// `options_.newrat.maxiter` (1000).
    pub nit: usize,
    /// `options_.newrat.tolerance.gstep` (NaN = `ftol`).
    pub gstep_crit: f64,
    /// `options_.newrat.tolerance.gstep_rel` (NaN).
    pub gstep_crit_rel: f64,
    /// `options_.newrat.robust` (false).
    pub robust: bool,
    /// `options_.gstep` ([1e-2, 1.0]).
    pub gstep: [f64; 2],
    /// `options_.gradient_epsilon` (1e-6).
    pub gradient_epsilon: f64,
    pub verbose: bool,
    /// Evaluate `mr_hessian`'s coordinates in parallel.
    pub parallel: bool,
    /// ---- estimation completion ---- `analytic_derivation`: gradient and outer-product
    /// Hessian from the objective (`analytic_Hessian = 'opg'`), no `mr_hessian`.
    pub analytic: bool,
}

impl Default for NewratOptions {
    fn default() -> Self {
        NewratOptions {
            ftol: 1e-5,
            nit: 1000,
            gstep_crit: f64::NAN,
            gstep_crit_rel: f64::NAN,
            robust: false,
            gstep: [1e-2, 1.0],
            gradient_epsilon: 1e-6,
            verbose: true,
            parallel: true,
            analytic: false,
        }
    }
}

// ---- estimation completion ---- newrat's returned `hess_info` (`new_rat_hess_info`), read by
// the estimation driver for `mr_hessian` at the mode (`optim = ('Hessian', 0|2)`).
thread_local! {
    static LAST_HESS_INFO: std::cell::RefCell<Option<HessInfo>> = const { std::cell::RefCell::new(None) };
}

/// The `hess_info` returned by the last [`newrat`] call on this thread.
pub fn take_last_hess_info() -> Option<HessInfo> {
    LAST_HESS_INFO.with(|h| h.borrow_mut().take())
}
// ---- end estimation completion

/// Output of [`mr_hessian`].
#[derive(Clone, Debug)]
pub struct MrHessian {
    /// `hessian_mat` (n x n), None when the objective gives no contributions.
    pub hessian_mat: Option<Mat>,
    pub gg: Vec<f64>,
    /// `htol1`: per-coordinate tolerances (or the scalar tolerance when `htol_rel` is set).
    pub htol1: Vec<f64>,
    /// Inverse outer-product-gradient Hessian.
    pub ihh: Option<Mat>,
    /// Outer-product-gradient Hessian.
    pub hh_mat0: Option<Mat>,
    /// Final steps.
    pub hh1: Option<Vec<f64>>,
    pub hess_info: HessInfo,
    pub fcount: usize,
}

struct Coord {
    f1: f64,
    f_1: f64,
    ff1: Vec<f64>,
    ff_1: Vec<f64>,
    h1: f64,
    htol: f64,
    fcount: usize,
}

fn nan_mat(n: usize) -> Mat {
    Mat::from_element(n, n, f64::NAN)
}

fn min_eig(m: &Mat) -> f64 {
    let (w, _) = sym_eig(m);
    w.iter().cloned().fold(f64::INFINITY, f64::min)
}

/// Port of `optimization/mr_hessian.m`: adaptive-step finite-difference
/// gradient and outer-product-gradient Hessian (from the likelihood
/// contributions of [`Objective::eval_full`]). `hflag` 0/1/2 as in Dynare.
#[allow(clippy::too_many_arguments)]
pub fn mr_hessian(
    obj: &dyn Objective,
    x: &[f64],
    penalty: f64,
    hflag: u8,
    htol0: f64,
    hess_info: &HessInfo,
    bounds: &Bounds,
    prior_std: &[f64],
    parallel: bool,
) -> MrHessian {
    let n = x.len();
    let mut hi = hess_info.clone();
    let (f0, _, ff0) = penalty_objective(obj, x, penalty, true);
    let mut fcount = 1;
    let h2: Vec<f64> = (0..n).map(|i| bounds.ub[i] - bounds.lb[i]).collect();
    let hmax: Vec<f64> = (0..n).map(|i| (bounds.ub[i] - x[i]).min(x[i] - bounds.lb[i])).collect();
    let ff0 = ff0.filter(|v| !v.is_empty());
    let opg = ff0.is_some();
    let m = ff0.as_ref().map_or(0, |v| v.len());
    for i in 0..n {
        hi.h1[i] = hi.h1[i].min(0.9 * hmax[i]);
    }
    if htol0 < hi.htol {
        hi.htol = htol0;
    }
    let mut htol1_scalar = None;
    if !hi.htol_rel.is_nan() {
        htol1_scalar = Some(hi.htol);
        hi.htol = (hi.htol_rel * f0).abs();
    }
    let htol_init = hi.htol;
    let evalc = |z: &[f64]| -> (f64, Option<Vec<f64>>) {
        let (f, _, c) = penalty_objective(obj, z, penalty, opg);
        (f, c)
    };
    let fallback = |c: Option<Vec<f64>>, fx: f64| -> Vec<f64> {
        if !opg {
            return Vec::new();
        }
        match c {
            Some(v) if v.len() == m && !v.iter().any(|a| a.is_nan()) => v,
            _ => vec![fx / m as f64; m],
        }
    };
    let coord = |i: usize| -> Coord {
        let mut h1 = hi.h1[i];
        let mut htol = htol_init;
        let mut fcount = 0;
        let mut xh1 = x.to_vec();
        loop {
            let h10 = h1;
            let mut hcheck = false;
            xh1[i] = x[i] + h1;
            let (mut fx, mut ffx) = evalc(&xh1);
            fcount += 1;
            let mut it = 0usize;
            let mut dx = vec![fx - f0];
            let mut ic = false;
            let mut icount = 0;
            let mut h0 = vec![h1];
            let mut istoobig = vec![false];
            let mut istoosmall = vec![false];
            let mut give_it_up = false;
            let mask_min = |h0: &[f64], mask: &[bool]| {
                h0.iter().zip(mask).filter(|(_, b)| **b).map(|(h, _)| *h).fold(f64::INFINITY, f64::min)
            };
            let mask_max = |h0: &[f64], mask: &[bool]| {
                h0.iter().zip(mask).filter(|(_, b)| **b).map(|(h, _)| *h).fold(f64::NEG_INFINITY, f64::max)
            };
            while (dx[it].abs() < 0.5 * htol || dx[it].abs() > 3.0 * htol)
                && icount < 10
                && !ic
                && hmax[i] > 2.0 * 1e-10
                && !give_it_up
            {
                icount += 1;
                istoobig.resize(it + 1, false);
                istoosmall.resize(it + 1, false);
                istoobig[it] = false;
                istoosmall[it] = false;
                if dx[it].abs() < 0.5 * htol {
                    istoosmall[it] = true;
                    if h1 == 0.9 * hmax[i] {
                        give_it_up = true;
                    }
                    let mut htmp = if dx[it] != 0.0 { 0.9 * htol / dx[it].abs() * h1 } else { 2.1 * h1 };
                    htmp = htmp.min(0.9 * hmax[i]);
                    if istoobig[..=it].iter().any(|&b| b) {
                        htmp = 0.5 * (mask_min(&h0, &istoobig) + mask_max(&h0, &istoosmall));
                    }
                    h1 = htmp.max(1e-10);
                    xh1[i] = x[i] + h1;
                    (fx, ffx) = evalc(&xh1);
                    fcount += 1;
                }
                if dx[it].abs() > 3.0 * htol {
                    istoobig[it] = true;
                    let mut htmp = htol / dx[it].abs() * h1;
                    if istoosmall[..=it].iter().any(|&b| b) {
                        htmp = 0.5 * (mask_min(&h0, &istoobig) + mask_max(&h0, &istoosmall));
                    }
                    h1 = htmp.max(1e-10);
                    xh1[i] = x[i] + h1;
                    (fx, ffx) = evalc(&xh1);
                    fcount += 1;
                    let mut iter = 0;
                    while fx - f0 == 0.0 && iter < 50 {
                        h1 *= 2.0;
                        xh1[i] = x[i] + h1;
                        (fx, ffx) = evalc(&xh1);
                        fcount += 1;
                        ic = true;
                        iter += 1;
                    }
                }
                it += 1;
                dx.push(fx - f0);
                h0.push(h1);
                if h1 < 1e-12 * 1f64.min(h2[i]) && h1 < 0.9 * hmax[i] {
                    ic = true;
                    hcheck = true;
                }
            }
            if icount == 10 || h1 == 1e-10 {
                istoobig.resize(it + 1, false);
                istoosmall.resize(it + 1, false);
                istoobig[it] = false;
                istoosmall[it] = false;
                if dx[it].abs() < 0.5 * htol {
                    istoosmall[it] = true;
                }
                if dx[it].abs() > 3.0 * htol {
                    istoobig[it] = true;
                }
                if istoobig.iter().any(|&b| b) && (istoosmall[it] || istoobig[it]) {
                    // [ddx, ij] = min(dx(istoobig)); h1 = h0(ij) (ij indexes the subset, as in Dynare)
                    let sub: Vec<f64> = dx.iter().zip(&istoobig).filter(|(_, b)| **b).map(|(d, _)| *d).collect();
                    let mut ij = 0;
                    for (k, v) in sub.iter().enumerate() {
                        if *v < sub[ij] {
                            ij = k;
                        }
                    }
                    fx = f0 + sub[ij];
                    h1 = h0[ij];
                }
            }
            let f1 = fx;
            let ff1 = fallback(ffx, fx);
            xh1[i] = x[i] - h1;
            let (fxm, ffxm) = evalc(&xh1);
            fcount += 1;
            let ff_1 = fallback(ffxm, fxm);
            xh1[i] = x[i];
            if hcheck && htol < 1.0 {
                let mindx = dx.iter().map(|d| d.abs()).fold(f64::INFINITY, f64::min);
                htol = 1f64.min((mindx * 2.0).max(htol * 10.0));
                h1 = h10;
                continue;
            }
            return Coord { f1, f_1: fxm, ff1, ff_1, h1, htol, fcount };
        }
    };
    let coords: Vec<Coord> =
        if parallel { (0..n).into_par_iter().map(coord).collect() } else { (0..n).map(coord).collect() };
    fcount += coords.iter().map(|c| c.fcount).sum::<usize>();
    let hhtol: Vec<f64> = coords.iter().map(|c| c.htol).collect();
    if let Some(last) = hhtol.last() {
        hi.htol = *last;
    }
    for (i, c) in coords.iter().enumerate() {
        hi.h1[i] = c.h1;
    }
    let h1 = hi.h1.clone();
    let f1: Vec<f64> = coords.iter().map(|c| c.f1).collect();
    let f_1: Vec<f64> = coords.iter().map(|c| c.f_1).collect();
    let gg: Vec<f64> = (0..n).map(|i| (f1[i] - f_1[i]) / (2.0 * h1[i])).collect();
    let htol1 = match htol1_scalar {
        Some(v) => vec![v],
        None => hhtol.clone(),
    };
    if !opg {
        return MrHessian { hessian_mat: None, gg, htol1, ihh: None, hh_mat0: None, hh1: None, hess_info: hi, fcount };
    }
    // ggh(:, i) = (ff1 - ff_1) / (2 h1(i))
    let mut ggh = Mat::zeros(m, n);
    for (i, c) in coords.iter().enumerate() {
        for k in 0..m {
            ggh[(k, i)] = (c.ff1[k] - c.ff_1[k]) / (2.0 * h1[i]);
        }
    }
    let mut hessian_mat = Mat::zeros(n, n);
    if hflag == 2 {
        let temp: Vec<f64> = (0..n).map(|i| f1[i] + f_1[i] - f0).collect();
        let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| ((i + 1)..n).map(move |j| (i, j))).collect();
        let off = |&(i, j): &(usize, usize)| {
            let mut zp = x.to_vec();
            let mut zm = x.to_vec();
            zp[i] = x[i] + h1[i];
            zp[j] = x[j] + h1[j];
            zm[i] = x[i] - h1[i];
            zm[j] = x[j] - h1[j];
            let t1 = pen(obj, &zp, penalty);
            let t2 = pen(obj, &zm, penalty);
            -(-t1 - t2 + temp[i] + temp[j]) / (2.0 * h1[i] * h1[j])
        };
        let vals: Vec<f64> =
            if parallel { pairs.par_iter().map(off).collect() } else { pairs.iter().map(off).collect() };
        fcount += 2 * pairs.len();
        for i in 0..n {
            hessian_mat[(i, i)] = (f1[i] + f_1[i] - 2.0 * f0) / (h1[i] * h1[i]);
        }
        for (&(i, j), v) in pairs.iter().zip(vals) {
            hessian_mat[(i, j)] = v;
            hessian_mat[(j, i)] = v;
        }
    } else if hflag == 1 {
        for i in 0..n {
            let dum = (f1[i] + f_1[i] - 2.0 * f0) / (h1[i] * h1[i]);
            hessian_mat[(i, i)] = if dum <= f64::EPSILON { f64::EPSILON.max(gg[i] * gg[i]) } else { dum };
        }
    }
    let a_diag: Vec<f64> = h1.iter().map(|h| 2.0 * h).collect();
    let mut gga = ggh.clone();
    for i in 0..n {
        for k in 0..m {
            gga[(k, i)] *= a_diag[i];
        }
    }
    let mut hh_mat = gga.transpose() * &gga;
    let mut hh_mat0 = ggh.transpose() * &ggh;
    let a = Mat::from_diagonal(&crate::linalg::Vector::from_vec(a_diag.clone()));
    let left_div = |m: &Mat, b: &Mat| solve(m, b).unwrap_or_else(|| nan_mat(n));
    let mut ihh = &a * left_div(&hh_mat, &a);
    if hflag > 0 && min_eig(&hessian_mat) > 0.0 {
        let hh0 = &a * &hessian_mat * &a;
        let hh = hessian_mat.clone();
        let mut sd0: Vec<f64> = (0..n).map(|i| hh0[(i, i)].sqrt()).collect();
        let sd: Vec<f64> = (0..n).map(|i| hh_mat[(i, i)].sqrt()).collect();
        for j in 0..n {
            for i in 0..n {
                hh_mat[(i, j)] = hh_mat[(i, j)] / (sd[i] * sd[j]) * (sd0[i] * sd0[j]);
            }
        }
        ihh = &a * left_div(&hh_mat, &a);
        let sd: Vec<f64> = (0..n).map(|i| ihh[(i, i)].sqrt()).collect();
        let sdh: Vec<f64> = (0..n).map(|i| (1.0 / hh[(i, i)]).sqrt()).collect();
        for j in 0..n {
            let s = prior_std[j].min(sd[j]);
            sd0[j] = 10f64.powf(0.5 * (s.log10() + sdh[j].log10()));
        }
        let inv_a = Mat::from_diagonal(&crate::linalg::Vector::from_vec(a_diag.iter().map(|v| 1.0 / v).collect()));
        for j in 0..n {
            for i in 0..n {
                ihh[(i, j)] = ihh[(i, j)] / (sd[i] * sd[j]) * (sd0[i] * sd0[j]);
            }
        }
        let igg = inv_a.transpose() * &ihh * &inv_a;
        // hh_mat0 = inv_A' / igg * inv_A
        let igg_inv =
            solve(&igg.transpose(), &Mat::identity(n, n)).map(|m| m.transpose()).unwrap_or_else(|| nan_mat(n));
        hh_mat0 = inv_a.transpose() * igg_inv * &inv_a;
    }
    if hflag < 2 {
        hessian_mat = hh_mat0.clone();
    }
    if hessian_mat.iter().any(|v| v.is_nan()) {
        hh_mat0 = Mat::identity(n, n);
        ihh = hh_mat0.clone();
        hessian_mat = hh_mat0.clone();
    }
    MrHessian {
        hessian_mat: Some(hessian_mat),
        gg,
        htol1,
        ihh: Some(ihh),
        hh_mat0: Some(hh_mat0),
        hh1: Some(hi.h1.clone()),
        hess_info: hi,
        fcount,
    }
}

/// `check_bounds` subfunction (`newrat.m` uses `eps`, `mr_gstep.m` `1e-10`).
fn check_bounds(x: &[f64], bounds: &Bounds, margin: f64) -> Vec<f64> {
    let mut y = x.to_vec();
    for i in 0..y.len() {
        if y[i] >= bounds.ub[i] {
            y[i] = bounds.ub[i] - margin;
        }
    }
    for i in 0..y.len() {
        if y[i] <= bounds.lb[i] {
            y[i] = bounds.lb[i] + margin;
        }
    }
    y
}

fn diag_mat(v: &[f64]) -> Mat {
    Mat::from_diagonal(&crate::linalg::Vector::from_vec(v.to_vec()))
}

/// MATLAB `sign` (`sign(0) = 0`; `f64::signum` gives 1 for 0.0).
fn matlab_sign(x: f64) -> f64 {
    if x == 0.0 || x.is_nan() { x } else { x.signum() }
}

fn linspace7(a: f64, b: f64) -> [f64; 7] {
    let mut out = [0.0; 7];
    for (k, o) in out.iter_mut().enumerate() {
        *o = a + k as f64 * (b - a) / 6.0;
    }
    out[6] = b;
    out
}

/// Port of `optimization/mr_gstep.m`: univariate Newton steps along each
/// coordinate. Returns `(f0, x, ig, fcount)`.
#[allow(clippy::too_many_arguments)]
fn mr_gstep(
    obj: &dyn Objective,
    h1: Option<&[f64]>,
    x: &[f64],
    bounds: &Bounds,
    penalty: f64,
    htol0: &[f64],
    verbose: bool,
    gradient_epsilon: f64,
    robust: bool,
) -> (f64, Vec<f64>, Vec<bool>, usize) {
    let n = x.len();
    let h1: Vec<f64> = h1.map(|h| h.to_vec()).unwrap_or_else(|| vec![gradient_epsilon; n]);
    let htol: Vec<f64> = match htol0.len() {
        0 => vec![1e-6; n],
        1 => vec![htol0[0]; n],
        _ => htol0.to_vec(),
    };
    let mut x = x.to_vec();
    let mut f0 = pen(obj, &x, penalty);
    let mut fcount = 1;
    let mut xh1 = x.clone();
    let mut ig = vec![false; n];
    let cb = |v: &[f64]| check_bounds(v, bounds, 1e-10);
    for i in 0..n {
        xh1[i] = x[i] + h1[i];
        let f1 = pen(obj, &xh1, penalty);
        xh1[i] = x[i] - h1[i];
        let f_1 = pen(obj, &xh1, penalty);
        fcount += 2;
        let mut gg = vec![0.0; n];
        let mut hh = vec![0.0; n];
        gg[i] = (f1 - f_1) / (2.0 * h1[i]);
        hh[i] = 1.0 / 1e-9_f64.max(((f1 + f_1 - 2.0 * f0) / (h1[i] * h1[i])).abs());
        if gg[i] * (hh[i] * gg[i]) / 2.0 > htol[i] {
            let (mut ff, mut xx, fc, retcode) = csminit1(obj, &x, penalty, f0, &gg, false, &diag_mat(&hh), verbose);
            fcount += fc;
            if retcode != 0 && robust {
                let xa = if x[i].abs() < 1e-6 {
                    linspace7(x[i] / 2.0, matlab_sign(x[i]) * 1e-6 * 3.0 / 2.0)
                } else {
                    linspace7(x[i] / 2.0, x[i] * 3.0 / 2.0)
                };
                let mut fa = [0.0; 7];
                for k in 0..7 {
                    xh1[i] = xa[k];
                    fa[k] = pen(obj, &xh1, penalty);
                    fcount += 1;
                }
                let design = Mat::from_fn(7, 3, |r, c| match c {
                    0 => 1.0,
                    1 => xa[r],
                    _ => xa[r] * xa[r] / 2.0,
                });
                let rhs = Mat::from_column_slice(7, 1, &fa);
                let b = design.svd(true, true).solve(&rhs, 0.0).unwrap_or_else(|_| Mat::from_element(3, 1, f64::NAN));
                gg[i] = x[i] * b[2] + b[1];
                hh[i] = 1.0 / b[2];
                let (ff2, xx2, fc, _) = csminit1(obj, &x, penalty, f0, &gg, false, &diag_mat(&hh), verbose);
                fcount += fc;
                if ff2 < ff {
                    ff = ff2;
                    xx = xx2;
                }
                let (mut im, mut fmin) = (0, fa[0]);
                for (k, v) in fa.iter().enumerate() {
                    if *v < fmin {
                        fmin = *v;
                        im = k;
                    }
                }
                if fmin < ff {
                    ff = fmin;
                    xx[i] = xa[im];
                }
            }
            ig[i] = true;
            if robust {
                let adjust = |xx: &mut Vec<f64>| {
                    *xx = cb(xx);
                    if xx[i] < x[i] {
                        xx[i] = (xx[i] + h1[i]).min(0.5 * (xx[i] + x[i]));
                    } else {
                        xx[i] = (xx[i] - h1[i]).max(0.5 * (xx[i] + x[i]));
                    }
                };
                if xx != cb(&xx) {
                    adjust(&mut xx);
                    let (ffb, ok, _) = penalty_objective(obj, &xx, penalty, false);
                    fcount += 1;
                    if !ok {
                        if verbose {
                            println!("last step exited with bad status!");
                        }
                    } else if ffb < f0 {
                        f0 = ffb;
                        x = xx;
                    }
                } else {
                    if (f0 - ff).abs() < (gg[i] * (hh[i] * gg[i]) / 2.0 / 100.0).abs() || (x[i] - xx[i]).abs() < 1e-10 {
                        let mgg: Vec<f64> = gg.iter().map(|v| -v).collect();
                        let (mut ff1, mut xx1, fc, _) =
                            csminit1(obj, &x, penalty, f0, &mgg, false, &diag_mat(&hh), verbose);
                        fcount += fc;
                        if xx1 != cb(&xx1) {
                            adjust(&mut xx1);
                            let (v, ok, _) = penalty_objective(obj, &xx1, penalty, false);
                            ff1 = v;
                            fcount += 1;
                            if !ok && verbose {
                                println!("last step exited with bad status!");
                            }
                        }
                        if ff1 < ff {
                            ff = ff1;
                            xx = xx1;
                        }
                    }
                    f0 = ff;
                    x = xx;
                }
            } else {
                f0 = ff;
                x = cb(&xx);
            }
            if verbose {
                println!("Done for param {} = {:8.4}; f = {:8.4}", i + 1, x[i], f0);
            }
        }
        xh1 = x.clone();
    }
    (f0, x, ig, fcount)
}

/// Port of `optimization/newrat.m` as called by `dynare_minimize_objective.m`
/// for `mode_compute = 5` (numerical derivatives, `flagg = 0`, outer-product
/// gradient Hessian from the likelihood contributions when the objective
/// provides them, BFGS updates otherwise). `prior_std` is `bayestopt_.p2`
/// (NaN for ML). The returned `hessian` is newrat's `hh`.
pub fn newrat(obj: &dyn Objective, x: &[f64], bounds: &Bounds, prior_std: &[f64], opts: &NewratOptions) -> OptimResult {
    let nx = x.len();
    let verbose = opts.verbose;
    let ftol = opts.ftol;
    let crit = opts.ftol;
    let gstep_crit = if opts.gstep_crit.is_nan() { crit } else { opts.gstep_crit };
    let mut hess_info = HessInfo {
        gstep: opts.gstep,
        htol: gstep_crit,
        htol_rel: opts.gstep_crit_rel,
        h1: vec![opts.gradient_epsilon; nx],
        robust: opts.robust,
    };
    let flagit = 0u8;
    let mut penalty = 1e8;
    let mut fcount = 0;
    let mut xparam1 = x.to_vec();
    let htol_base = 1e-7_f64.max(hess_info.htol);
    let gtol = 1e-3;
    let mut htol = htol_base;
    // ---- estimation completion ---- analytic_derivation: [fval0, exit_flag, gg, hh] =
    // penalty_objective_function(x, ...) with the outer-product Hessian
    let opg_an = |x: &[f64], penalty: f64| {
        crate::estimation::analytic::penalty_objective_grad(
            obj,
            x,
            penalty,
            crate::estimation::analytic::AnalyticHessian::Opg,
        )
    };
    let (fv0, exit_flag, an0) = if opts.analytic {
        let (f, ok, g, h) = opg_an(x, penalty);
        (f, ok, Some((g, h)))
    } else {
        let (f, ok, _) = penalty_objective(obj, x, penalty, true);
        (f, ok, None)
    };
    fcount += 1;
    if !exit_flag {
        let msg = "Bad initial parameter.";
        if verbose {
            println!("{msg}");
        }
        return OptimResult {
            x: x.to_vec(),
            fval: fv0,
            hessian: None,
            inv_hessian: None,
            iterations: 0,
            func_count: fcount,
            exitflag: -2,
            message: msg.into(),
        };
    }
    let mut fval0 = vec![fv0];
    let mut fval = fv0;
    let mut opg = true;
    #[allow(clippy::type_complexity)]
    let (mut gg, mut htol0, mut igg, mut hhg, mut h1, mut dum, mut hh): (
        Vec<f64>,
        Vec<f64>,
        Mat,
        Option<Mat>,
        Option<Vec<f64>>,
        Option<Mat>,
        Option<Mat>,
    );
    if let Some((g, h)) = an0 {
        // hh nonempty: hh0 = hh; hhg = hh; igg = inv(hh); h1 = []
        let h = h.unwrap_or_else(|| Mat::identity(nx, nx));
        gg = g;
        htol0 = vec![htol_base; nx];
        igg = solve(&h, &Mat::identity(nx, nx)).unwrap_or_else(|| nan_mat(nx));
        hhg = Some(h.clone());
        h1 = None;
        dum = None;
        hh = Some(h);
    } else {
        penalty = fv0;
        let mr = mr_hessian(obj, x, penalty, flagit, htol, &hess_info, bounds, prior_std, opts.parallel);
        fcount += mr.fcount;
        gg = mr.gg;
        htol0 = mr.htol1;
        hhg = mr.hh_mat0;
        h1 = mr.hh1;
        hess_info = mr.hess_info;
        dum = mr.hessian_mat;
        hh = None;
        match &dum {
            None => {
                opg = false;
                igg = Mat::identity(nx, nx) * 1e-4;
            }
            Some(_) => {
                igg = mr.ihh.expect("ihh");
                hh = hhg.clone();
            }
        }
        if htol0.iter().cloned().fold(f64::NEG_INFINITY, f64::max) > htol && verbose {
            println!("\nNumerical noise in the likelihood\nTolerance has to be relaxed\n");
        }
    }
    let mut h_inv = igg.clone();
    if verbose {
        disp_eigenvalues_gradient(&gg, hh.as_ref());
    }
    let mut check = false;
    let mut igrad = true;
    let inx = Mat::identity(nx, nx);
    let mut jit = 0;
    let mut ig = vec![true; nx];
    let mut xhist = vec![xparam1.clone()];
    let mut ghist = vec![gg.clone()];
    let mut message = String::new();
    let mut exitflag = -3;
    let cbe = |v: &[f64]| check_bounds(v, bounds, f64::EPSILON);
    while norm(&gg) > gtol && !check && jit < opts.nit {
        jit += 1;
        let icount = jit; // 1-based iteration counter: fval0[icount-1] is fval0(icount)
        penalty = fval0[icount - 1];
        crate::progress::detail(format!("newrat iteration {icount}, f = {penalty:.6}"));
        if verbose {
            println!("\nIteration {icount}");
        }
        let (fv, mut x0, fc, _) = csminit1(obj, &xparam1, penalty, fval0[icount - 1], &gg, false, &h_inv, verbose);
        fval = fv;
        fcount += fc;
        if igrad {
            let (fval1, x01, fc, _) = csminit1(obj, &x0, penalty, fval, &gg, false, &inx, verbose);
            fcount += fc;
            if fval - fval1 > 1.0 {
                if verbose {
                    println!("Gradient step!!");
                }
            } else {
                igrad = false;
            }
            fval = fval1;
            x0 = x01;
        }
        let ig_pos: Vec<usize> = (0..nx).filter(|&i| ig[i]).collect();
        if ig_pos.len() < nx {
            let mut ggx = vec![0.0; nx];
            for &i in &ig_pos {
                ggx[i] = gg[i];
            }
            let hhx = if opts.analytic || !opg {
                hh.clone().unwrap_or_else(|| Mat::identity(nx, nx))
            } else {
                dum.clone().expect("dum")
            };
            let mut iggx = Mat::identity(nx, nx);
            let k = ig_pos.len();
            let sub = Mat::from_fn(k, k, |r, c| hhx[(ig_pos[r], ig_pos[c])]);
            let subinv = solve(&sub, &Mat::identity(k, k)).unwrap_or_else(|| nan_mat(k));
            for r in 0..k {
                for c in 0..k {
                    iggx[(ig_pos[r], ig_pos[c])] = subinv[(r, c)];
                }
            }
            let (_, xx, fc, _) = csminit1(obj, &x0, penalty, fval, &ggx, false, &iggx, verbose);
            x0 = xx;
            fcount += fc;
        }
        let clamp = |x0: &mut Vec<f64>, penalty: &mut f64, fcount: &mut usize| {
            let cbx = cbe(x0);
            if *x0 != cbx {
                *x0 = cbx;
                let (fvala, ok, _) = penalty_objective(obj, x0, *penalty, false);
                *fcount += 1;
                if ok {
                    *penalty = fvala;
                } else if verbose {
                    println!("last step exited with bad status!");
                }
            }
        };
        clamp(&mut x0, &mut penalty, &mut fcount);
        let (fvala, xg, igv, fc) = mr_gstep(
            obj,
            h1.as_deref(),
            &x0,
            bounds,
            penalty,
            &htol0,
            verbose,
            opts.gradient_epsilon,
            hess_info.robust,
        );
        x0 = xg;
        ig = igv;
        fcount += fc;
        clamp(&mut x0, &mut penalty, &mut fcount);
        if verbose {
            println!("Sequence of univariate steps!!");
        }
        fval = fvala;
        if fval0[icount - 1] - fval < ftol && flagit == 0 {
            if verbose {
                println!("Try diagonal Hessian");
            }
            let hd = hhg.as_ref().map(|m| m.diagonal().iter().map(|v| 1.0 / v).collect::<Vec<f64>>());
            let ihh = diag_mat(&hd.unwrap_or_else(|| vec![f64::NAN; nx]));
            let (fval2, xx, fc, _) = csminit1(obj, &x0, penalty, fval, &gg, false, &ihh, verbose);
            fcount += fc;
            x0 = cbe(&xx);
            if fval - fval2 >= ftol && verbose {
                println!("Diagonal Hessian successful");
            }
            fval = fval2;
        }
        if fval0[icount - 1] - fval < ftol && flagit == 0 {
            if verbose {
                println!("Try gradient direction");
            }
            let ihh0 = &inx * 1e-4;
            let (fval3, xx, fc, _) = csminit1(obj, &x0, penalty, fval, &gg, false, &ihh0, verbose);
            fcount += fc;
            x0 = cbe(&xx);
            if fval - fval3 >= ftol && verbose {
                println!("Gradient direction successful");
            }
            fval = fval3;
        }
        xparam1 = x0;
        xhist.push(xparam1.clone());
        fval0.push(fval);
        let improvement = fval0[icount - 1] - fval;
        if improvement < ftol {
            message = "No further improvement is possible!".into();
            if verbose {
                println!("{message}");
            }
            check = true;
            if opts.analytic {
                let (_, _, g, h) = opg_an(&xparam1, penalty);
                fcount += 1;
                gg = g;
                if let Some(h) = h {
                    h_inv = solve(&h, &inx).unwrap_or_else(|| nan_mat(nx));
                    hhg = Some(h.clone());
                    hh = Some(h);
                }
            }
            if verbose {
                println!("FVAL          {fval}");
                println!("Improvement   {improvement}");
                println!("Ftol          {ftol}");
                disp_eigenvalues_gradient(&gg, hh.as_ref());
            }
            ghist.push(gg.clone());
        } else {
            if verbose {
                let dxn: f64 =
                    xhist[icount].iter().zip(&xhist[icount - 1]).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                println!("Actual dxnorm {dxn}");
                println!("FVAL          {fval}");
                println!("Improvement   {improvement}");
                println!("Ftol          {ftol}");
                println!("Htol          {}", htol0.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
            }
            htol = htol_base;
            let prev = &xhist[icount - 1];
            let dn: f64 = prev.iter().zip(&xparam1).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            if opts.analytic {
                let (_, _, g, h) = opg_an(&xparam1, penalty);
                fcount += 1;
                gg = g;
                if let Some(h) = h {
                    h_inv = solve(&h, &inx).unwrap_or_else(|| nan_mat(nx));
                    hhg = Some(h.clone());
                    hh = Some(h);
                }
            } else if dn > 1e-12 {
                let mr = mr_hessian(obj, &xparam1, penalty, flagit, htol, &hess_info, bounds, prior_std, opts.parallel);
                fcount += mr.fcount;
                gg = mr.gg;
                htol0 = mr.htol1;
                hhg = mr.hh_mat0;
                h1 = mr.hh1;
                hess_info = mr.hess_info;
                dum = mr.hessian_mat;
                if let Some(i) = mr.ihh {
                    igg = i;
                }
                if dum.is_none() {
                    opg = false;
                }
                if htol0.iter().cloned().fold(f64::NEG_INFINITY, f64::max) > htol && verbose {
                    println!("\nNumerical noise in the likelihood\nTolerance has to be relaxed\n");
                }
                if !opg {
                    let dg: Vec<f64> = gg.iter().zip(&ghist[icount - 1]).map(|(a, b)| a - b).collect();
                    let dx: Vec<f64> = xparam1.iter().zip(prev).map(|(a, b)| a - b).collect();
                    h_inv = bfgsi1(&h_inv, &dg, &dx, verbose);
                    let inv = solve(&h_inv, &inx).unwrap_or_else(|| nan_mat(nx));
                    hh = Some(inv.clone());
                    hhg = Some(inv);
                } else {
                    hh = hhg.clone();
                    h_inv = igg.clone();
                }
            }
            ghist.push(gg.clone());
        }
    }
    if jit == opts.nit {
        if !message.is_empty() {
            message.push(' ');
        }
        message.push_str("Maximum number of iterations reached.");
        exitflag = 0;
    }
    if norm(&gg) <= gtol {
        if !message.is_empty() {
            message.push(' ');
        }
        message.push_str(&format!("Estimation ended: Gradient norm < {gtol}"));
        exitflag = -1;
    }
    if check {
        if !message.is_empty() {
            message.push(' ');
        }
        message.push_str("Estimation successful.");
        exitflag = 1;
    }
    if verbose {
        println!("{message}");
    }
    LAST_HESS_INFO.with(|h| *h.borrow_mut() = Some(hess_info.clone())); // ---- estimation completion
    OptimResult {
        x: xparam1,
        fval,
        hessian: hh,
        inv_hessian: Some(igg),
        iterations: jit,
        func_count: fcount,
        exitflag,
        message,
    }
}

fn disp_eigenvalues_gradient(gg: &[f64], hh: Option<&Mat>) {
    println!("Gradient norm  {}", norm(gg));
    if let Some(h) = hh {
        let (w, _) = sym_eig(h);
        println!("Minimum Hessian eigenvalue {}", w.iter().cloned().fold(f64::INFINITY, f64::min));
        println!("Maximum Hessian eigenvalue {}", w.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    }
}

#[cfg(test)]
mod tests {
    use super::matlab_sign;

    /// mr_gstep.m:103 `sign(x(i))`: MATLAB's `sign(0)` is 0 (`f64::signum(0.0)` is 1).
    #[test]
    fn matlab_sign_of_zero_is_zero() {
        assert_eq!(matlab_sign(0.0), 0.0);
        assert_eq!(matlab_sign(-0.0), 0.0);
        assert_eq!(matlab_sign(3e-7), 1.0);
        assert_eq!(matlab_sign(-3e-7), -1.0);
    }
}
