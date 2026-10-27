// SPDX-License-Identifier: GPL-3.0-or-later
//! Steady state of the static model.
//!
//! Port of `evaluate_steady_state.m` (non-block, non-bytecode, no steady-state
//! file branch, including the `model(linear)` shortcut) and of
//! `optimization/dynare_solve.m` for `solve_algo` = 1, 2, 4 (default), 9 and 12,
//! with `optimization/solve1.m`, `optimization/lnsrch1.m` and
//! `optimization/trust_region.m`.
//!
//! The static model is passed as a closure `f(y) -> (residuals, jacobian)`
//! (Dynare's `static_problem`). Auxiliary variables: `evaluate_steady_state.m`
//! first applies `<fname>.set_auxiliary_variables` to the initial guess; do
//! that before calling [`steady_state`] (the model module owns that function).

use super::dmperm::dmperm;

#[path = "steady_state_x.rs"]
mod steady_state_x; // [stream X] solve_algo 0, 3, 6-8, 13, 14
use super::SolveError;
use crate::linalg::{Mat, chol_upper, lu, lu_solve, mul, norm1, rcond_from_lu, trsm_left};

/// The static model: `y -> (residuals, Jacobian)`.
pub type StaticFn<'a> = &'a dyn Fn(&[f64]) -> (Vec<f64>, Mat);

/// The static model's residuals alone (`[fvec] = feval(func, x)` with one output): what
/// `lnsrch1.m` and the residual checks evaluate. It must return the residuals of
/// [`StaticFn`]; it only saves computing a Jacobian that would be thrown away.
pub type ResidFn<'a> = &'a dyn Fn(&[f64]) -> Vec<f64>;

/// Options of `evaluate_steady_state` / `dynare_solve`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteadyOptions {
    /// `options_.solve_algo` (Dynare default 4; sw07.mod's `steady` sets 1,
    /// which then holds for the estimation too).
    pub solve_algo: u8,
    /// `options_.steady.maxit` (default 50).
    pub maxit: usize,
    /// `options_.solve_tolf` (default `eps^(1/3)`).
    pub tolf: f64,
    /// `options_.solve_tolx` (default `eps^(2/3)`).
    pub tolx: f64,
    /// `options_.trust_region_initial_step_bound_factor` (default 1).
    pub trust_region_initial_step_bound_factor: f64,
    /// `options_.linear` (`model(linear)`): one Newton step instead of `dynare_solve`.
    pub linear: bool,
}

impl Default for SteadyOptions {
    fn default() -> Self {
        SteadyOptions {
            solve_algo: 4,
            maxit: 50,
            tolf: f64::EPSILON.cbrt(),
            tolx: f64::EPSILON.powf(2.0 / 3.0),
            trust_region_initial_step_bound_factor: 1.0,
            linear: false,
        }
    }
}

/// Outcome of [`dynare_solve`]: `(x, errorflag, errorcode)` of `dynare_solve.m`.
#[derive(Debug, Clone, PartialEq)]
pub struct SolveOutcome {
    pub x: Vec<f64>,
    pub errorflag: bool,
    /// Dynare's `errorcode` (-11: converged at the initial guess, solve1:
    /// -1/1 success, 2 maxit, 3 dX, 4 spurious, 5 zero Jacobian, ...).
    /// -10: non-finite residuals/Jacobian at the initial guess (Dynare would
    /// then randomize the initial guess, which is not reproducible here).
    /// -20: `lnsrch1.m` raised a MATLAB error (non-finite Newton direction
    /// or "Roundoff problem in nlsearch").
    pub errorcode: i32,
}

fn max_abs(v: &[f64]) -> f64 {
    // MATLAB max(abs(v)): NaN entries ignored unless all are NaN
    v.iter().map(|x| x.abs()).filter(|a| !a.is_nan()).reduce(f64::max).unwrap_or(f64::NAN)
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn sub_vec(v: &[f64], idx: &[usize]) -> Vec<f64> {
    idx.iter().map(|&i| v[i]).collect()
}

fn sub_mat(a: &Mat, rows: &[usize], cols: &[usize]) -> Mat {
    Mat::from_fn(rows.len(), cols.len(), |i, j| a[(rows[i], cols[j])])
}

/// Port of `optimization/dynare_solve.m` (`jacobian_flag = true`).
pub fn dynare_solve(f: StaticFn, x0: &[f64], opts: &SteadyOptions) -> SolveOutcome {
    dynare_solve_r(f, &|x: &[f64]| f(x).0, x0, opts)
}

/// [`dynare_solve`] with a residual-only function `r` for the evaluations that need no
/// Jacobian (line searches, residual checks). Same iterates as [`dynare_solve`].
pub fn dynare_solve_r(f: StaticFn, resid: ResidFn, x0: &[f64], opts: &SteadyOptions) -> SolveOutcome {
    let nn = x0.len();
    let (tolf, tolx, maxit) = (opts.tolf, opts.tolx, opts.maxit);
    let (fvec, fjac) = f(x0);
    let fvec_finite = fvec.iter().all(|x| x.is_finite());
    if fvec_finite && max_abs(&fvec) < tolf {
        return SolveOutcome { x: x0.to_vec(), errorflag: false, errorcode: -11 };
    }
    if !fvec_finite || fjac.iter().any(|x| !x.is_finite()) {
        // options_.solve_randomize_initial_guess: MATLAB's rand stream cannot be reproduced.
        return SolveOutcome { x: x0.to_vec(), errorflag: true, errorcode: -10 };
    }
    let all: Vec<usize> = (0..nn).collect();
    let mut x = x0.to_vec();
    match opts.solve_algo {
        1 | 12 => {
            let (errorflag, errorcode) = solve1(f, resid, &mut x, &all, &all, tolf, tolx, maxit);
            SolveOutcome { x, errorflag, errorcode }
        }
        9 => {
            let (errorflag, errorcode) = trust_region(f, &mut x, &all, &all, tolf, tolx, maxit, opts);
            SolveOutcome { x, errorflag, errorcode }
        }
        2 | 4 => {
            let solver = |x: &mut Vec<f64>, j1: &[usize], j2: &[usize]| {
                if opts.solve_algo == 2 {
                    solve1(f, resid, x, j1, j2, tolf, tolx, maxit)
                } else {
                    trust_region(f, x, j1, j2, tolf, tolx, maxit, opts)
                }
            };
            let dm = dmperm(&fjac);
            let (j1, j2, r, s) = (&dm.p, &dm.q, &dm.r, &dm.s);
            let (mut errorflag, mut errorcode) = (false, 0);
            for i in (0..r.len() - 1).rev() {
                let blocklength = r[i + 1] - r[i];
                let j: Vec<usize> = (r[i]..r[i + 1]).collect();
                let blockcolumns = s[i + 1] - s[i];
                if blockcolumns != blocklength {
                    let fval_check = resid(&x);
                    let rows: Vec<usize> = j.iter().map(|&k| j1[k]).collect();
                    if norm2(&sub_vec(&fval_check, &rows)) < tolf {
                        errorflag = false;
                        errorcode = 0;
                        continue;
                    }
                }
                if blockcolumns >= blocklength {
                    // dynare_solve.m passes j2(j), i.e. the row range also for the columns
                    let rows: Vec<usize> = j.iter().map(|&k| j1[k]).collect();
                    let cols: Vec<usize> = j.iter().map(|&k| j2[k]).collect();
                    (errorflag, errorcode) = solver(&mut x, &rows, &cols);
                } else {
                    errorflag = true;
                    errorcode = 0;
                }
                if errorflag {
                    return SolveOutcome { x, errorflag, errorcode };
                }
            }
            let fv = resid(&x);
            if max_abs(&fv) > tolf {
                (errorflag, errorcode) = solver(&mut x, &all, &all);
            }
            SolveOutcome { x, errorflag, errorcode }
        }
        // [stream X] solve_algo 0, 3, 6, 7, 8, 13, 14
        a => steady_state_x::dynare_solve_x(f, x0, opts).unwrap_or_else(|| {
            println!(
                "ERROR: dynare_solve: solve_algo = {a} is not supported by dynare-rust (supported: 0-4, 6-9, 12-14)"
            );
            SolveOutcome { x: x0.to_vec(), errorflag: true, errorcode: -30 }
        }),
    }
}

/// Port of `optimization/solve1.m` (Jacobian supplied). Returns `(errorflag, errorcode)`.
#[allow(clippy::too_many_arguments)]
fn solve1(
    f: StaticFn,
    r: ResidFn,
    x: &mut Vec<f64>,
    j1: &[usize],
    j2: &[usize],
    tolf: f64,
    tolx: f64,
    maxit: usize,
) -> (bool, i32) {
    let nn = j1.len();
    let stpmx = 100.0;
    let mut errorflag = false;
    let mut fvec = sub_vec(&r(x), j1);
    if fvec.iter().any(|v| !v.is_finite()) {
        return (true, 0);
    }
    let mut fval = 0.5 * fvec.iter().map(|v| v * v).sum::<f64>();
    if max_abs(&fvec) < tolf * tolf {
        return (errorflag, -1);
    }
    let stpmax = stpmx * norm2(x).max(nn as f64);
    let sqrt_eps = f64::EPSILON.sqrt();
    for _its in 1..=maxit {
        let (fv, fj) = f(x);
        fvec = sub_vec(&fv, j1);
        let fjac = sub_mat(&fj, j1, j2);
        let fvm = Mat::from_column_slice(nn, 1, &fvec);
        let g = mul(&fjac, true, &fvm, false); // (fvec' * fjac)'
        let (flu, piv, _) = lu(&fjac);
        let rcond_fjac = rcond_from_lu(&flu, norm1(&fjac));
        let p: Vec<f64> = if rcond_fjac < sqrt_eps {
            let mut fjac2 = mul(&fjac, true, &fjac, false);
            let temp = norm1(&fjac2);
            if temp > 0.0 {
                let lambda = (nn as f64 * f64::EPSILON).sqrt() * temp;
                for k in 0..nn {
                    fjac2[(k, k)] += lambda;
                }
                let mut rhs = g.clone(); // fjac' * fvec
                match chol_upper(&fjac2) {
                    Some(u) => {
                        trsm_left(true, true, 1.0, &u, &mut rhs);
                        trsm_left(true, false, 1.0, &u, &mut rhs);
                    }
                    None => {
                        let (fl, pv, _) = lu(&fjac2);
                        lu_solve(&fl, &pv, false, &mut rhs);
                    }
                }
                rhs.iter().map(|v| -v).collect()
            } else {
                return (true, 5);
            }
        } else {
            let mut rhs = fvm.clone();
            lu_solve(&flu, &piv, false, &mut rhs);
            rhs.iter().map(|v| -v).collect()
        };
        let xold = x.clone();
        let fold = fval;
        let g: Vec<f64> = g.iter().copied().collect();
        let ls = match lnsrch1(&xold, fold, &g, p, stpmax, &mut |xx: &[f64]| r(xx), j1, j2, tolx) {
            Some(ls) => ls,
            None => return (true, -20),
        };
        *x = ls.x;
        fval = ls.f;
        fvec = ls.fvec;
        if ls.check {
            errorflag = true;
            let den = fval.max(0.5 * nn as f64);
            let crit =
                j2.iter().enumerate().map(|(k, &j)| g[k].abs() * x[j].abs().max(1.0)).fold(f64::NEG_INFINITY, f64::max)
                    / den;
            if crit < tolx {
                let dx =
                    j2.iter().map(|&j| (x[j] - xold[j]).abs() / x[j].abs().max(1.0)).fold(f64::NEG_INFINITY, f64::max);
                if dx < tolx {
                    return (errorflag, 3);
                }
            } else {
                return (errorflag, 4);
            }
        } else if max_abs(&fvec) < tolf {
            return (errorflag, 1);
        }
    }
    (true, 2)
}

pub(crate) struct LineSearch {
    pub(crate) x: Vec<f64>,
    pub(crate) f: f64,
    pub(crate) fvec: Vec<f64>,
    pub(crate) check: bool,
}

/// Port of `optimization/lnsrch1.m`. `None` where MATLAB raises an error.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lnsrch1(
    xold: &[f64],
    fold: f64,
    g: &[f64],
    mut p: Vec<f64>,
    stpmax: f64,
    func: &mut dyn FnMut(&[f64]) -> Vec<f64>,
    j1: &[usize],
    j2: &[usize],
    tolx: f64,
) -> Option<LineSearch> {
    let alf = 1e-4;
    let mut alam = 1.0;
    let mut x = xold.to_vec();
    let summ = norm2(&p);
    if !summ.is_finite() {
        return None; // "Some element of Newton direction isn't finite"
    }
    if summ > stpmax {
        for v in &mut p {
            *v *= stpmax / summ;
        }
    }
    let slope: f64 = g.iter().zip(&p).map(|(a, b)| a * b).sum();
    let test = j2.iter().zip(&p).map(|(&j, pk)| pk.abs() / xold[j].abs().max(1.0)).fold(f64::NEG_INFINITY, f64::max);
    let mut alamin = tolx / test;
    if alamin > 0.1 {
        alamin = 0.1;
    }
    let (mut alam2, mut f2, mut fold2) = (0.0, 0.0, 0.0);
    let mut f = f64::NAN;
    let mut fvec = Vec::new();
    loop {
        if alam < alamin {
            return Some(LineSearch { x, f, fvec, check: true });
        }
        for (k, &j) in j2.iter().enumerate() {
            x[j] = xold[j] + alam * p[k];
        }
        fvec = sub_vec(&func(&x), j1);
        f = 0.5 * fvec.iter().map(|v| v * v).sum::<f64>();
        if fvec.iter().any(|v| v.is_nan()) {
            alam /= 2.0;
            alam2 = alam;
            f2 = f;
            fold2 = fold;
        } else if f <= fold + alf * alam * slope {
            return Some(LineSearch { x, f, fvec, check: false });
        } else {
            let mut tmplam;
            if alam == 1.0 {
                tmplam = -slope / (2.0 * (f - fold - slope));
            } else {
                let rhs1 = f - fold - alam * slope;
                let rhs2 = f2 - fold2 - alam2 * slope;
                let a = (rhs1 / (alam * alam) - rhs2 / (alam2 * alam2)) / (alam - alam2);
                let b = (-alam2 * rhs1 / (alam * alam) + alam * rhs2 / (alam2 * alam2)) / (alam - alam2);
                if a == 0.0 {
                    tmplam = -slope / (2.0 * b);
                } else {
                    let disc = b * b - 3.0 * a * slope;
                    if disc < 0.0 {
                        return None; // "Roundoff problem in nlsearch"
                    }
                    tmplam = (-b + disc.sqrt()) / (3.0 * a);
                }
                if tmplam > 0.5 * alam {
                    tmplam = 0.5 * alam;
                }
            }
            alam2 = alam;
            f2 = f;
            fold2 = fold;
            alam = tmplam.max(0.1 * alam); // MATLAB max ignores NaN
        }
    }
}

/// MATLAB `eps(x)` for finite `x >= 0`.
fn eps_of(x: f64) -> f64 {
    let x = x.abs();
    if x == 0.0 {
        return f64::from_bits(1);
    }
    if x < f64::MIN_POSITIVE {
        return f64::from_bits(1);
    }
    let e = x.log2().floor();
    // guard against log2 rounding at powers of two
    let mut s = 2f64.powf(e - 52.0);
    if 2f64.powf(e) > x {
        s /= 2.0;
    }
    s
}

/// Port of `optimization/trust_region.m` (Jacobian supplied). Returns `(errorflag, info)`.
#[allow(clippy::too_many_arguments)]
fn trust_region(
    f: StaticFn,
    x: &mut Vec<f64>,
    j1: &[usize],
    j2: &[usize],
    tolf: f64,
    tolx: f64,
    maxiter: usize,
    opts: &SteadyOptions,
) -> (bool, i32) {
    let factor = opts.trust_region_initial_step_bound_factor;
    let radiusfactor = 0.5;
    let actualreductionthreshold = 0.001;
    let relativereductionthreshold = 0.1;
    let n = j1.len();
    let mut errorflag = true;
    let mut iter = 1usize;
    let mut info = 0;
    let mut delta;
    let (mut ncsucc, mut ncslow) = (0usize, 0usize);
    let xinit = x.clone();
    let (fv, fj) = f(x);
    let mut fval = sub_vec(&fv, j1);
    let mut fjac = sub_mat(&fj, j1, j2);
    let mut fnorm = norm2(&fval);
    if fval.iter().any(|v| !v.is_finite()) || fjac.iter().any(|v| !v.is_finite()) {
        return (errorflag, info);
    }
    if fnorm < tolf {
        return (false, -1);
    }
    let mut fjacnorm_ = vec![0.0; n];
    let mut xnorm;
    delta = 0.0;
    let col_norms = |a: &Mat| -> Vec<f64> { a.column_iter().map(|c| c.norm()).collect() };
    let wnorm = |d: &[f64], v: &[f64]| -> f64 { norm2(&d.iter().zip(v).map(|(a, b)| a * b).collect::<Vec<_>>()) };
    while iter <= maxiter && info == 0 {
        let fjacnorm = col_norms(&fjac);
        let xj = sub_vec(x, j2);
        if iter == 1 {
            fjacnorm_ = fjacnorm.iter().map(|&v| if v < f64::EPSILON { 1.0 } else { v }).collect();
            xnorm = wnorm(&fjacnorm_, &xj);
            delta = if xnorm > 0.0 { xnorm * factor } else { factor };
        } else {
            fjacnorm_ = fjacnorm_.iter().zip(&fjacnorm).map(|(a, b)| (0.1 * a).max(*b)).collect();
            xnorm = wnorm(&fjacnorm_, &xj);
        }
        let p = dogleg(&fjac, &fval, &fjacnorm_, delta);
        let pnorm = wnorm(&fjacnorm_, &p);
        let mut xx = x.clone();
        let x0 = x.clone();
        for (k, &j) in j2.iter().enumerate() {
            xx[j] = x[j] - p[k];
        }
        let fjp = &fjac * Mat::from_column_slice(n, 1, &p);
        let ww: Vec<f64> = (0..n).map(|k| fval[k] - fjp[k]).collect();
        let fval1 = sub_vec(&f(&xx).0, j1);
        if fval1.iter().any(|v| !v.is_finite()) {
            delta *= radiusfactor;
            iter += 1;
            continue;
        }
        let fnorm1 = norm2(&fval1);
        if fnorm1 < tolf {
            *x = xx;
            errorflag = false;
            info = 1;
            continue;
        }
        let actualreduction = if fnorm1 < fnorm { 1.0 - (fnorm1 / fnorm).powi(2) } else { -1.0 };
        let tt = norm2(&ww);
        let predictedreduction = if tt < fnorm { 1.0 - (tt / fnorm).powi(2) } else { 0.0 };
        let ratio = if predictedreduction > 0.0 { actualreduction / predictedreduction } else { 0.0 };
        if iter == 1 {
            delta = delta.min(pnorm);
        }
        if ratio < relativereductionthreshold {
            ncsucc = 0;
            delta *= radiusfactor;
        } else {
            ncsucc += 1;
            if (ratio - 1.0).abs() < relativereductionthreshold {
                delta = pnorm / radiusfactor;
            } else if ratio >= radiusfactor || ncsucc > 1 {
                delta = delta.max(pnorm / radiusfactor);
            }
        }
        if ratio > 1e-4 {
            *x = xx;
            fval = fval1;
            xnorm = wnorm(&fjacnorm_, &sub_vec(x, j2));
            fnorm = fnorm1;
        }
        ncslow += 1;
        if actualreduction >= actualreductionthreshold {
            ncslow = 0;
        }
        iter += 1;
        if iter == maxiter {
            info = 2;
            *x = xinit.clone();
            continue;
        }
        if delta < tolx * xnorm {
            info = 3;
            *x = xinit.clone();
            errorflag = true;
            continue;
        }
        if tolx > 0.0 && (0.1 * delta).max(pnorm) <= 10.0 * eps_of(xnorm) * xnorm {
            info = 5;
            *x = xinit.clone();
            errorflag = true;
            continue;
        }
        if ncslow == 15 {
            info = 4;
            *x = xinit.clone();
            errorflag = true;
            continue;
        }
        let fjac0 = fjac.clone();
        fjac = sub_mat(&f(x).1, j1, j2);
        if fjac.iter().any(|v| !v.is_finite()) {
            *x = x0;
            fjac = fjac0;
            delta *= radiusfactor;
        }
    }
    (errorflag, info)
}

/// `dogleg` subfunction of `trust_region.m`.
fn dogleg(r: &Mat, b: &[f64], d: &[f64], delta: f64) -> Vec<f64> {
    let n = b.len();
    let bm = Mat::from_column_slice(n, 1, b);
    let mut xm = bm.clone();
    let (fl, pv, _) = lu(r);
    lu_solve(&fl, &pv, false, &mut xm);
    let mut x: Vec<f64> = xm.iter().copied().collect();
    let dn = |v: &[f64]| norm2(&v.iter().zip(d).map(|(a, b)| a * b).collect::<Vec<_>>());
    let qnorm = dn(&x);
    if qnorm <= delta {
        return x;
    }
    let rtb = mul(r, true, &bm, false);
    let mut s: Vec<f64> = rtb.iter().zip(d).map(|(a, b)| a / b).collect();
    let gnorm = norm2(&s);
    let (alpha, sgnorm);
    if gnorm > 0.0 {
        s = s.iter().zip(d).map(|(a, b)| (a / gnorm) / b).collect();
        let rs = r * Mat::from_column_slice(n, 1, &s);
        let temp0 = rs.norm();
        sgnorm = gnorm / (temp0 * temp0);
        if sgnorm < delta {
            let bnorm = norm2(b);
            let temp1 = delta / qnorm;
            let temp2 = sgnorm / delta;
            let mut t0 = bnorm * bnorm * temp2 / (gnorm * qnorm);
            t0 = t0 - temp1 * temp2 * temp2
                + ((t0 - temp1).powi(2) + (1.0 - temp1 * temp1) * (1.0 - temp2 * temp2)).sqrt();
            alpha = temp1 * (1.0 - temp2 * temp2) / t0;
        } else {
            alpha = 0.0;
        }
    } else {
        alpha = delta / qnorm;
        sgnorm = 0.0;
    }
    let w = sgnorm.min(delta);
    if alpha > 0.0 {
        for k in 0..n {
            x[k] = alpha * x[k] + (1.0 - alpha) * w * s[k];
        }
    } else {
        for k in 0..n {
            x[k] = w * s[k];
        }
    }
    x
}

/// Port of `evaluate_steady_state.m` for a model without steady-state file,
/// `block` or `bytecode`: `dynare_solve` (or one Newton step for
/// `model(linear)`), then the `info` checks (20: not found, with
/// `info(2) = resid'*resid`; 22: NaN).
pub fn steady_state(f: StaticFn, guess: &[f64], opts: &SteadyOptions) -> Result<Vec<f64>, SolveError> {
    steady_state_r(f, &|x: &[f64]| f(x).0, guess, opts)
}

/// [`steady_state`] with a residual-only function (see [`dynare_solve_r`]).
pub fn steady_state_r(f: StaticFn, r: ResidFn, guess: &[f64], opts: &SteadyOptions) -> Result<Vec<f64>, SolveError> {
    steady_state_r_aux(f, r, guess, opts, None)
}

/// [`steady_state_r`] with `set_aux`, `<fname>.set_auxiliary_variables` applied to the solver's
/// last iterate on failure before `info(2) = resid'*resid` (evaluate_steady_state.m:438-446).
pub fn steady_state_r_aux(
    f: StaticFn,
    r: ResidFn,
    guess: &[f64],
    opts: &SteadyOptions,
    set_aux: Option<&dyn Fn(&mut [f64])>,
) -> Result<Vec<f64>, SolveError> {
    let (mut ys, check) = if opts.linear {
        let (fvec, jacob) = f(guess);
        if fvec.iter().any(|v| !v.is_finite()) {
            (fvec, true)
        } else if max_abs(&fvec) > 1e-12 {
            let mut d = Mat::from_column_slice(fvec.len(), 1, &fvec);
            let (fl, pv, _) = lu(&jacob);
            lu_solve(&fl, &pv, false, &mut d);
            let ys: Vec<f64> = guess.iter().zip(d.iter()).map(|(g, s)| g - s).collect();
            let resid = r(&ys);
            let bad = max_abs(&resid) > 1e-6;
            (ys, bad)
        } else {
            (guess.to_vec(), false)
        }
    } else {
        let out = dynare_solve_r(f, r, guess, opts);
        (out.x, out.errorflag)
    };
    if check {
        if let Some(set_aux) = set_aux {
            set_aux(&mut ys);
        }
        let resid = r(&ys);
        let ss: f64 = resid.iter().map(|r| r * r).sum();
        return Err(SolveError::new(if ss.is_nan() { 22 } else { 20 }, ss));
    }
    if ys.iter().any(|v| v.is_nan()) {
        return Err(SolveError::new(22, f64::NAN));
    }
    Ok(ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nonlinear(y: &[f64]) -> (Vec<f64>, Mat) {
        // y0^2 - 2 = 0 ; y1 - y0 * y2 = 0 ; y2 - 3 = 0 (recursive + nonlinear)
        let r = vec![y[0] * y[0] - 2.0, y[1] - y[0] * y[2], y[2] - 3.0];
        let j = Mat::from_row_slice(3, 3, &[2.0 * y[0], 0.0, 0.0, -y[2], 1.0, -y[0], 0.0, 0.0, 1.0]);
        (r, j)
    }

    #[test]
    fn solve_algos_converge() {
        for algo in [1u8, 2, 4, 9] {
            let opts = SteadyOptions { solve_algo: algo, ..Default::default() };
            let ys = steady_state(&nonlinear, &[1.0, 1.0, 1.0], &opts).unwrap();
            assert!(max_abs(&nonlinear(&ys).0) < opts.tolf, "algo {algo}: {ys:?}");
        }
    }

    #[test]
    fn converged_guess_returns_immediately() {
        let s2 = 2f64.sqrt();
        let out = dynare_solve(&nonlinear, &[s2, 3.0 * s2, 3.0], &SteadyOptions::default());
        assert_eq!(out.errorcode, -11);
        assert!(!out.errorflag);
    }

    #[test]
    fn singular_unit_root_solve1() {
        // y0 = y0 (unit root: zero row), y1 = 0.5 y1 + a, y2 = y1 + y0
        let a = 1.0;
        let f = move |y: &[f64]| -> (Vec<f64>, Mat) {
            let r = vec![0.0 * y[0], 0.5 * y[1] - a, y[2] - y[1] - y[0]];
            let j = Mat::from_row_slice(3, 3, &[0.0, 0.0, 0.0, 0.0, 0.5, 0.0, -1.0, -1.0, 1.0]);
            (r, j)
        };
        let opts = SteadyOptions { solve_algo: 1, maxit: 1000, ..Default::default() };
        let ys = steady_state(&f, &[1.0, 0.0, 0.0], &opts).unwrap();
        assert!(max_abs(&f(&ys).0) < opts.tolf, "{ys:?}");
    }

    #[test]
    fn failure_is_info_20() {
        let f = |y: &[f64]| -> (Vec<f64>, Mat) { (vec![y[0] * y[0] + 1.0], Mat::from_element(1, 1, 2.0 * y[0])) };
        let opts = SteadyOptions { solve_algo: 1, maxit: 20, ..Default::default() };
        let err = steady_state(&f, &[0.5], &opts).unwrap_err();
        assert_eq!(err.info, 20);
        assert!(err.info2 >= 1.0);
    }

    #[test]
    fn linear_shortcut() {
        let f = |y: &[f64]| -> (Vec<f64>, Mat) {
            (vec![2.0 * y[0] - 1.0, y[1] - y[0]], Mat::from_row_slice(2, 2, &[2.0, 0.0, -1.0, 1.0]))
        };
        let opts = SteadyOptions { linear: true, ..Default::default() };
        let ys = steady_state(&f, &[0.0, 0.0], &opts).unwrap();
        assert!((ys[0] - 0.5).abs() < 1e-15 && (ys[1] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn eps_matches_matlab() {
        assert_eq!(eps_of(1.0), f64::EPSILON);
        assert_eq!(eps_of(2.0), 2.0 * f64::EPSILON);
        assert_eq!(eps_of(1.5), f64::EPSILON);
        assert_eq!(eps_of(0.75), f64::EPSILON / 2.0);
    }
}
