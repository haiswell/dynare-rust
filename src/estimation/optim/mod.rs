// SPDX-License-Identifier: GPL-3.0-or-later
//! Mode finding: ports of Dynare's optimizers (`optimization/`).
//!
//! * [`csminwel`] — `csminwel1.m` (`mode_compute = 4`)
//! * [`newrat`] — `newrat.m` + `mr_hessian.m` + `mr_gstep.m` (`mode_compute = 5`)
//! * [`gmhmaxlik`] — `gmhmaxlik.m` + `gmhmaxlik_core.m` (`mode_compute = 6`), which
//!   sees the raw objective (failures are rejected Metropolis proposals)
//! * [`fminsearch`] — MATLAB's Nelder–Mead `fminsearch` (`mode_compute = 7`)
//! * [`dynare_simplex`] — `simplex_optimization_routine.m` (`mode_compute = 8`)
//! * [`minimize`] — `dynare_minimize_objective.m`: every `mode_compute`, with `optim` overrides
//!
//! The others see the objective through [`penalty_objective`]
//! (`penalty_objective_function.m`). Finite-difference gradients and
//! Hessians evaluate coordinates in parallel (rayon) when `parallel` is set;
//! results are bit-identical to the sequential order.

mod cmaes;
mod csminwel;
mod dynare_simplex;
mod gmhmaxlik;
pub mod minimize;
mod newrat;
mod particleswarm;
mod quasi_newton;
mod sa;
mod simplex;
mod simpsa;
pub mod simulannealbnd; // mode_compute = 102
mod solveopt;

pub use cmaes::{CmaesOptions, cmaes, cmaes_insigma};
pub use csminwel::{CsminwelOptions, csminwel};
pub use dynare_simplex::{SimplexOptions, dynare_simplex};
pub use gmhmaxlik::{GmhmaxlikOptions, GmhmaxlikResult, VarInit, gmhmaxlik};
pub use minimize::{MinimizeContext, minimize};
pub use newrat::{HessInfo, MrHessian, NewratOptions, mr_hessian, newrat, take_last_hess_info}; // ---- estimation completion: take_last_hess_info
pub use particleswarm::{ParticleswarmOptions, particleswarm};
pub use quasi_newton::{QuasiNewtonOptions, quasi_newton};
pub use sa::{SaOptions, simulated_annealing};
pub use simplex::{FminsearchOptions, fminsearch};
pub use simpsa::{SimpsaOptions, simpsa};
pub use solveopt::{SolveoptOptions, solveopt};

use rayon::prelude::*;

use super::{Objective, penalty_objective};
use crate::linalg::Mat;

/// Result of a mode search.
#[derive(Clone, Debug)]
pub struct OptimResult {
    pub x: Vec<f64>,
    pub fval: f64,
    /// csminwel: `inv(inverse_hessian)` as `dynare_minimize_objective.m`
    /// returns it; newrat: its (outer-product-gradient) `hh`; simplex: None.
    /// Dynare replaces it with [`crate::estimation::hessian`] at the mode for
    /// mode_compute 4 and 5 (default `newrat.hess = 1`).
    pub hessian: Option<Mat>,
    /// csminwel: the final inverse Hessian approximation `H`.
    pub inv_hessian: Option<Mat>,
    pub iterations: usize,
    pub func_count: usize,
    /// csminwel: `retcodeh`; newrat: `exitflag`; fminsearch: `exitflag`.
    pub exitflag: i32,
    pub message: String,
}

/// Finite-difference gradient used by csminwel/newrat (`options_.gradient_method`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientMethod {
    /// `numgrad2.m` (forward difference) — Dynare's default (`gradient_method = 2`).
    Forward,
    /// `numgrad3.m` (central difference, `gradient_method = 3`).
    Central,
    /// `numgrad5.m` (five-point, `gradient_method = 5`).
    FivePoint,
    /// `numgrad3_.m` (`NumgradAlgorithm = 13`): central difference, zero when `f0` is below
    /// both neighbours.
    CentralZeroAtMin,
    /// `numgrad5_.m` (`NumgradAlgorithm = 15`): five-point, zero when `f0` is below the
    /// neighbours on both sides.
    FivePointZeroAtMin,
}

impl GradientMethod {
    /// `get_num_grad`'s cases of csminwel1.m (2, 3, 5, 13, 15).
    pub fn from_code(code: i32) -> Option<Self> {
        match code {
            2 => Some(GradientMethod::Forward),
            3 => Some(GradientMethod::Central),
            5 => Some(GradientMethod::FivePoint),
            13 => Some(GradientMethod::CentralZeroAtMin),
            15 => Some(GradientMethod::FivePointZeroAtMin),
            _ => None,
        }
    }

    fn evals_per_coordinate(self) -> usize {
        match self {
            GradientMethod::Forward => 1,
            GradientMethod::Central | GradientMethod::CentralZeroAtMin => 2,
            GradientMethod::FivePoint | GradientMethod::FivePointZeroAtMin => 4,
        }
    }
}

pub(crate) fn pen(obj: &dyn Objective, x: &[f64], penalty: f64) -> f64 {
    penalty_objective(obj, x, penalty, false).0
}

/// Ports of `numgrad2.m`, `numgrad3.m`, `numgrad5.m`, `numgrad3_.m` and `numgrad5_.m`.
/// Returns `(g, badg, fcount)`.
///
/// `numgrad3_`/`numgrad5_` take the step `step_length_correction(x(i), ones(n,1), i)*epsilon
/// = epsilon` and print their messages unconditionally. `numgrad5_` never restores `x(i)`, so
/// parameter `i` is perturbed around `x` with every earlier parameter `j` left at
/// `x(j) - 2*epsilon`. (`numgrad3_`'s "penalty on the right/left" messages test the third
/// output of `penalty_objective_function`, the objective's empty `DLIK`, and never print.)
pub(crate) fn numgrad(
    obj: &dyn Objective,
    method: GradientMethod,
    f0: f64,
    x: &[f64],
    penalty: f64,
    epsilon: f64,
    parallel: bool,
) -> (Vec<f64>, bool, usize) {
    let h = epsilon;
    // (g0, message printed before the |g0| test)
    let coord = |i: usize| -> (f64, Option<String>) {
        let mut z = x.to_vec();
        if method == GradientMethod::FivePointZeroAtMin {
            for zj in z.iter_mut().take(i) {
                *zj -= 2.0 * h;
            }
        }
        let xi = x[i];
        let mut at = |v: f64| {
            z[i] = v;
            pen(obj, &z, penalty)
        };
        match method {
            GradientMethod::Forward => ((at(xi + h) - f0) / h, None),
            GradientMethod::Central => {
                let f1 = at(xi + h);
                let f2 = at(xi - h);
                ((f1 - f2) / (2.0 * h), None)
            }
            GradientMethod::FivePoint => {
                let f1 = at(xi + h);
                let f2 = at(xi - h);
                let f3 = at(xi + 2.0 * h);
                let f4 = at(xi - 2.0 * h);
                ((8.0 * (f1 - f2) + f4 - f3) / (12.0 * h), None)
            }
            GradientMethod::CentralZeroAtMin => {
                let f1 = at(xi + h);
                let f2 = at(xi - h);
                if f0 < f1 && f0 < f2 {
                    let msg = format!(
                        "Gradient w.r.t. parameter number {:3} (x={xi:16.8},h={h:16.8},f0={f0:16.8},f1={f1:16.8},f2={f2:16.8},g0={:16.8})",
                        i + 1,
                        (f1 - f2) / (2.0 * h)
                    );
                    (0.0, Some(msg))
                } else {
                    ((f1 - f2) / (2.0 * h), None)
                }
            }
            GradientMethod::FivePointZeroAtMin => {
                let f1 = at(xi + h);
                let f2 = at(xi - h);
                let f3 = at(xi + 2.0 * h);
                let f4 = at(xi - 2.0 * h);
                if f0 < f1 && f1 < f3 && f0 < f2 && f2 < f4 {
                    (0.0, None)
                } else {
                    ((8.0 * (f1 - f2) + f4 - f3) / (12.0 * h), None)
                }
            }
        }
    };
    let raw: Vec<(f64, Option<String>)> =
        if parallel { (0..x.len()).into_par_iter().map(coord).collect() } else { (0..x.len()).map(coord).collect() };
    let zero_at_min = matches!(method, GradientMethod::CentralZeroAtMin | GradientMethod::FivePointZeroAtMin);
    let mut badg = false;
    let g = raw
        .into_iter()
        .enumerate()
        .map(|(i, (g0, msg))| {
            if let Some(m) = msg {
                println!("{m}");
            }
            if g0.abs() < 1e15 {
                g0
            } else {
                if zero_at_min {
                    println!("Bad gradient -----------------------------------");
                    println!("Gradient w.r.t. parameter number {:3} (x={:16.8},h={h:16.8},g0={g0:16.8})", i + 1, x[i]);
                }
                badg = true;
                0.0
            }
        })
        .collect();
    (g, badg, method.evals_per_coordinate() * x.len())
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|a| a * a).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn matvec(m: &Mat, v: &[f64]) -> Vec<f64> {
    let n = m.nrows();
    let mut out = vec![0.0; n];
    for (j, &vj) in v.iter().enumerate() {
        for (i, o) in out.iter_mut().enumerate() {
            *o += m[(i, j)] * vj;
        }
    }
    out
}

/// Port of `optimization/csminit1.m` (line search along `-H0*g0`).
/// Returns `(fhat, xhat, fcount, retcode)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn csminit1(
    obj: &dyn Objective,
    x0: &[f64],
    penalty: f64,
    f0: f64,
    g0: &[f64],
    badg: bool,
    h0: &Mat,
    verbose: bool,
) -> (f64, Vec<f64>, usize, i32) {
    const ANGLE: f64 = 0.005;
    const THETA: f64 = 0.3;
    const FCHANGE: f64 = 1000.0;
    const MINLAMB: f64 = 1e-9;
    const MINDFAC: f64 = 0.01;
    let mut fcount = 0;
    let mut lambda: f64 = 1.0;
    let mut xhat = x0.to_vec();
    let mut fhat = f0;
    let g = g0;
    let gnorm = norm(g);
    let retcode = if gnorm < 1e-12 && !badg {
        1
    } else {
        let mut dx: Vec<f64> = matvec(h0, g).into_iter().map(|v| -v).collect();
        let mut dxnorm = norm(&dx);
        if dxnorm > 1e12 {
            if verbose {
                println!("Near-singular H problem.");
            }
            for v in dx.iter_mut() {
                *v = *v * FCHANGE / dxnorm;
            }
        }
        let mut dfhat = dot(&dx, g0);
        if !badg {
            let a = -dfhat / (gnorm * dxnorm);
            if a < ANGLE {
                let c = ANGLE * dxnorm / gnorm + dfhat / (gnorm * gnorm);
                for (d, gi) in dx.iter_mut().zip(g) {
                    *d -= c * gi;
                }
                dfhat = dot(&dx, g);
                dxnorm = norm(&dx);
                if verbose {
                    println!("Correct for low angle: {a}");
                }
            }
        }
        let _ = dxnorm;
        if verbose {
            println!("Predicted improvement: {:18.9}", -dfhat / 2.0);
        }
        let mut done = false;
        let mut factor: f64 = 3.0;
        let mut shrink = true;
        let mut lambda_max = f64::INFINITY;
        let mut lambda_peak = 0.0;
        let mut f_peak = f0;
        let mut rc = 0;
        let mut dxtest = vec![0.0; x0.len()];
        while !done {
            for k in 0..x0.len() {
                dxtest[k] = x0[k] + dx[k] * lambda;
            }
            let f = pen(obj, &dxtest, penalty);
            if verbose {
                println!("lambda = {lambda:10.5e}; f = {f:20.7}");
            }
            if f < fhat {
                fhat = f;
                xhat.copy_from_slice(&dxtest);
            }
            fcount += 1;
            let shrink_signal = (!badg && (f0 - f < (-THETA * dfhat * lambda).max(0.0))) || (badg && (f0 - f) < 0.0);
            let grow_signal = !badg && (lambda > 0.0 && (f0 - f > -(1.0 - THETA) * dfhat * lambda));
            if shrink_signal && (lambda > lambda_peak || lambda < 0.0) {
                if lambda > 0.0 && (!shrink || lambda / factor <= lambda_peak) {
                    shrink = true;
                    factor = factor.powf(0.6);
                    while lambda / factor <= lambda_peak {
                        factor = factor.powf(0.6);
                    }
                    if (factor - 1.0).abs() < MINDFAC {
                        rc = if lambda.abs() < 4.0 { 2 } else { 7 };
                        done = true;
                    }
                }
                if lambda < lambda_max && lambda > lambda_peak {
                    lambda_max = lambda;
                }
                lambda /= factor;
                if lambda.abs() < MINLAMB {
                    if lambda > 0.0 && f0 <= fhat {
                        lambda = -lambda * factor.powi(6);
                    } else {
                        rc = if lambda < 0.0 { 6 } else { 3 };
                        done = true;
                    }
                }
            } else if (grow_signal && lambda > 0.0) || (shrink_signal && (lambda <= lambda_peak && lambda > 0.0)) {
                if shrink {
                    shrink = false;
                    factor = factor.powf(0.6);
                    if (factor - 1.0).abs() < MINDFAC {
                        rc = if lambda.abs() < 4.0 { 4 } else { 7 };
                        done = true;
                    }
                }
                if f < f_peak && lambda > 0.0 {
                    f_peak = f;
                    lambda_peak = lambda;
                    if lambda_max <= lambda_peak {
                        lambda_max = lambda_peak * factor * factor;
                    }
                }
                lambda *= factor;
                if lambda.abs() > 1e20 {
                    rc = 5;
                    done = true;
                }
            } else {
                done = true;
                rc = if factor < 1.2 { 7 } else { 0 };
            }
        }
        rc
    };
    (fhat, xhat, fcount, retcode)
}

/// Port of `optimization/bfgsi1.m` (BFGS update of the inverse Hessian).
pub(crate) fn bfgsi1(h0: &Mat, dg: &[f64], dx: &[f64], verbose: bool) -> Mat {
    let n = dg.len();
    let hdg = matvec(h0, dg);
    let dgdx = dot(dg, dx);
    if dgdx.abs() > 1e-12 {
        let s = 1.0 + dot(dg, &hdg) / dgdx;
        let mut h = h0.clone();
        for j in 0..n {
            for i in 0..n {
                h[(i, j)] = h0[(i, j)] + s * (dx[i] * dx[j]) / dgdx - (dx[i] * hdg[j] + hdg[i] * dx[j]) / dgdx;
            }
        }
        h
    } else {
        if verbose {
            println!("bfgs update failed.");
            println!("|dg| = {} |dx| = {}", norm(dg), norm(dx));
            println!("dg'*dx = {dgdx}");
            println!("|H*dg| = {}", dot(&hdg, &hdg));
        }
        h0.clone()
    }
}
