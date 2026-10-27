// SPDX-License-Identifier: GPL-3.0-or-later
//! Substitutes for the Optimization Toolbox optimizers that Dynare calls for
//! `mode_compute = 1` (`fmincon`) and `mode_compute = 3` (`fminunc`).
//!
//! **These are not MATLAB's algorithms** (whose sources are not part of
//! Dynare): a BFGS quasi-Newton method on the inverse Hessian with a
//! backtracking Armijo line search and finite-difference gradients
//! (`fminunc`'s default `quasi-newton` algorithm is also BFGS, but its line
//! search and safeguards differ), and for `fmincon` the same method projected
//! onto the box `[lb, ub]` (active-set projected BFGS, instead of `fmincon`'s
//! default interior-point algorithm). They take Dynare's defaults
//! (`MaxFunEvals = 100000`, `TolFun = 1e-8`, `TolX = 1e-6`; `MaxIter` 400 for
//! `fminunc`, 1000 for `fmincon` as in MATLAB) and the usual `optim` keys.
//! Results (iterates, counts) cannot be compared with MATLAB's; the mode
//! reached can.
//!
//! The finite-difference gradient evaluates its coordinates in parallel
//! (rayon) when `parallel` is set; the result is bitwise the same.

use anyhow::{Result, bail};
use rayon::prelude::*;

use super::OptimResult;
use super::minimize::{kv_num, kv_str};
use crate::estimation::Objective;
use crate::modfile::OptValue;

/// Options shared by the `fmincon`/`fminunc` substitutes (`optimoptions` names).
#[derive(Clone, Debug)]
pub struct QuasiNewtonOptions {
    /// `MaxIterations` (fminunc 400, fmincon 1000).
    pub max_iter: usize,
    /// `MaxFunctionEvaluations` (Dynare: 100000).
    pub max_fun_evals: usize,
    /// `TolFun` / `OptimalityTolerance` (Dynare: 1e-8): stop when the infinity
    /// norm of the (projected) gradient is below `tol_fun * max(1, |g0|_inf)`,
    /// or when the relative decrease of f is below `tol_fun` in two successive steps.
    pub tol_fun: f64,
    /// `TolX` / `StepTolerance` (Dynare: 1e-6): stop when `|dx| < tol_x * (1 + |x|)`.
    pub tol_x: f64,
    /// `FiniteDifferenceType`: forward (default) or central.
    pub central: bool,
    /// `FiniteDifferenceStepSize` (default sqrt(eps) forward, eps^(1/3) central),
    /// relative to `max(|x_i|, 1)`.
    pub fd_step: Option<f64>,
    /// `Display` != 'off'.
    pub verbose: bool,
    pub parallel: bool,
    /// ---- estimation completion ---- `analytic_derivation` (`GradObj = 'on'`, `TolX = 1e-7`
    /// for fmincon): the objective's analytic gradient instead of finite differences.
    pub analytic: bool,
}

impl QuasiNewtonOptions {
    /// Dynare's `optimoptions('fminunc','display','iter','MaxFunEvals',100000,'TolFun',1e-8,'TolX',1e-6)`.
    pub fn fminunc() -> Self {
        QuasiNewtonOptions {
            max_iter: 400,
            max_fun_evals: 100_000,
            tol_fun: 1e-8,
            tol_x: 1e-6,
            central: false,
            fd_step: None,
            verbose: true,
            parallel: true,
            analytic: false,
        }
    }

    /// Dynare's `optimoptions('fmincon','display','iter','MaxFunEvals',100000,'TolFun',1e-8,'TolX',1e-6)`.
    pub fn fmincon() -> Self {
        QuasiNewtonOptions { max_iter: 1000, ..Self::fminunc() }
    }

    /// `optimoptions(optim_options, <optim>)` for the keys the substitute understands.
    pub fn apply_optim(&mut self, kv: &[(String, OptValue)], who: &str) -> Result<()> {
        for (k, v) in kv {
            match k.as_str() {
                "MaxIter" | "MaxIterations" => self.max_iter = kv_num(k, v)? as usize,
                "MaxFunEvals" | "MaxFunctionEvaluations" => self.max_fun_evals = kv_num(k, v)? as usize,
                "TolFun" | "OptimalityTolerance" | "FunctionTolerance" => self.tol_fun = kv_num(k, v)?,
                "TolX" | "StepTolerance" => self.tol_x = kv_num(k, v)?,
                "FiniteDifferenceType" | "FinDiffType" => {
                    self.central = match kv_str(k, v)?.as_str() {
                        "central" => true,
                        "forward" => false,
                        s => bail!("{who}: unknown FiniteDifferenceType '{s}'"),
                    }
                }
                "FiniteDifferenceStepSize" | "FinDiffRelStep" => self.fd_step = Some(kv_num(k, v)?),
                "Display" | "display" => self.verbose = kv_str(k, v)? != "off",
                "Algorithm" | "GradObj" | "SpecifyObjectiveGradient" | "HessianApproximation" | "HessUpdate" => {
                    println!(
                        "WARNING: {who}: option {k} is ignored by dynare-rust's substitute (not MATLAB's toolbox algorithm)."
                    )
                }
                other => println!("WARNING: {who}: Unknown option ({other})!"),
            }
        }
        Ok(())
    }
}

struct Eval<'a> {
    obj: &'a dyn Objective,
    count: std::sync::atomic::AtomicUsize,
}

impl Eval<'_> {
    fn f(&self, x: &[f64]) -> f64 {
        self.count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (v, info) = self.obj.eval(x);
        if info != 0 || v.is_nan() { f64::INFINITY } else { v }
    }
    fn count(&self) -> usize {
        self.count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, a| m.max(a.abs()))
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|a| a * a).sum::<f64>().sqrt()
}

/// Projected BFGS (`lb`/`ub` = None: plain BFGS, the `fminunc` substitute).
/// Uses the raw objective; infinite values (Dynare's failures) are treated as
/// "step too long" by the line search and avoided by the gradient.
pub fn quasi_newton(
    obj: &dyn Objective,
    x0: &[f64],
    bounds: Option<(&[f64], &[f64])>,
    opts: &QuasiNewtonOptions,
) -> Result<OptimResult> {
    let n = x0.len();
    let ev = Eval { obj, count: Default::default() };
    // fmincon's interior point never touches the bounds; keep a tiny margin.
    let (lo, hi): (Vec<f64>, Vec<f64>) = match bounds {
        Some((lb, ub)) => (
            lb.iter()
                .zip(ub)
                .map(|(&l, &u)| if l.is_finite() { l + 1e-10 * (u - l).abs().min(1.0).max(1e-300) } else { l })
                .collect(),
            lb.iter()
                .zip(ub)
                .map(|(&l, &u)| if u.is_finite() { u - 1e-10 * (u - l).abs().min(1.0).max(1e-300) } else { u })
                .collect(),
        ),
        None => (vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n]),
    };
    let project = |x: &mut [f64]| {
        for i in 0..n {
            x[i] = x[i].clamp(lo[i], hi[i]);
        }
    };
    let mut x = x0.to_vec();
    project(&mut x);
    let mut f = ev.f(&x);
    if !f.is_finite() {
        bail!("quasi-Newton: the objective is not finite at the initial point");
    }
    let step_rel = opts.fd_step.unwrap_or(if opts.central { f64::EPSILON.cbrt() } else { f64::EPSILON.sqrt() });
    let grad = |x: &[f64], fx: f64| -> Vec<f64> {
        if opts.analytic {
            // ---- estimation completion ---- analytic_gradient_wrapper.m
            return crate::estimation::analytic::eval_with_gradient(
                obj,
                x,
                crate::estimation::analytic::AnalyticHessian::None,
            )
            .map(|e| e.grad)
            .unwrap_or_else(|| vec![f64::NAN; n]);
        }
        let coord = |i: usize| -> f64 {
            let h = step_rel * x[i].abs().max(1.0);
            let mut z = x.to_vec();
            // Step inwards when the forward point would leave the box.
            let sgn = if x[i] + h > hi[i] { -1.0 } else { 1.0 };
            z[i] = x[i] + sgn * h;
            let f1 = ev.f(&z);
            if opts.central && x[i] - sgn * h >= lo[i] && x[i] - sgn * h <= hi[i] {
                z[i] = x[i] - sgn * h;
                let f2 = ev.f(&z);
                if f1.is_finite() && f2.is_finite() {
                    return sgn * (f1 - f2) / (2.0 * h);
                }
                if f2.is_finite() {
                    return sgn * (fx - f2) / h;
                }
            }
            if f1.is_finite() {
                return sgn * (f1 - fx) / h;
            }
            z[i] = x[i] - sgn * h;
            let f2 = ev.f(&z);
            if f2.is_finite() && z[i] >= lo[i] && z[i] <= hi[i] { sgn * (fx - f2) / h } else { 0.0 }
        };
        if opts.parallel { (0..n).into_par_iter().map(coord).collect() } else { (0..n).map(coord).collect() }
    };
    // Coordinates at a bound with the gradient pushing outwards are held fixed.
    let active = |x: &[f64], g: &[f64]| -> Vec<bool> {
        (0..n).map(|i| (x[i] <= lo[i] && g[i] > 0.0) || (x[i] >= hi[i] && g[i] < 0.0)).collect()
    };
    let proj_grad = |x: &[f64], g: &[f64]| -> Vec<f64> {
        let a = active(x, g);
        (0..n).map(|i| if a[i] { 0.0 } else { g[i] }).collect()
    };
    let mut g = grad(&x, f);
    let g0 = inf_norm(&proj_grad(&x, &g)).max(1.0);
    let mut h = vec![0.0; n * n]; // inverse Hessian approximation, row-major
    for i in 0..n {
        h[i * n + i] = 1.0;
    }
    let mut first = true;
    let mut small_steps = 0;
    let mut iter = 0usize;
    let mut exitflag = 0;
    let mut message = String::from("Maximum number of iterations exceeded.");
    if opts.verbose {
        println!("                                                        First-order");
        println!(" Iteration  Func-count       f(x)        Step-size       optimality");
        println!("{:>6}{:>12}{:>18.6}{:>32.3e}", 0, ev.count(), f, inf_norm(&proj_grad(&x, &g)));
    }
    while iter < opts.max_iter {
        let pg = proj_grad(&x, &g);
        if inf_norm(&pg) <= opts.tol_fun * g0 {
            exitflag = 1;
            message = "Local minimum found: the first-order optimality measure is below OptimalityTolerance.".into();
            break;
        }
        if ev.count() >= opts.max_fun_evals {
            exitflag = 0;
            message = "Maximum number of function evaluations exceeded.".into();
            break;
        }
        let act = active(&x, &g);
        // d = -H g on the free coordinates.
        let mut d: Vec<f64> = (0..n)
            .map(|i| if act[i] { 0.0 } else { -(0..n).filter(|&j| !act[j]).map(|j| h[i * n + j] * g[j]).sum::<f64>() })
            .collect();
        let mut slope: f64 = d.iter().zip(&g).map(|(a, b)| a * b).sum();
        if !(slope < 0.0) {
            // Not a descent direction: restart from steepest descent.
            for i in 0..n {
                h[i * n..(i + 1) * n].iter_mut().for_each(|v| *v = 0.0);
                h[i * n + i] = 1.0;
                d[i] = -pg[i];
            }
            slope = -pg.iter().map(|a| a * a).sum::<f64>();
            first = true;
        }
        let mut alpha = if first { (1.0 / inf_norm(&d)).min(1.0) } else { 1.0 };
        let mut xn = vec![0.0; n];
        let mut fnew;
        let mut ok = false;
        loop {
            for i in 0..n {
                xn[i] = x[i] + alpha * d[i];
            }
            project(&mut xn);
            fnew = ev.f(&xn);
            let dec: f64 = (0..n).map(|i| g[i] * (xn[i] - x[i])).sum();
            if fnew.is_finite() && fnew < f && fnew <= f + 1e-4 * dec.min(0.0) {
                ok = true;
                break;
            }
            alpha *= if fnew.is_finite() {
                // Quadratic interpolation, safeguarded in [0.1, 0.5].
                let a = (fnew - f - alpha * slope) / (alpha * alpha);
                if a > 0.0 { ((-slope / (2.0 * a)) / alpha).clamp(0.1, 0.5) } else { 0.5 }
            } else {
                0.25
            };
            if alpha * inf_norm(&d) < 1e-16 * (1.0 + inf_norm(&x)) || ev.count() >= opts.max_fun_evals {
                break;
            }
        }
        iter += 1;
        if !ok {
            exitflag = 2;
            message = "Local minimum possible: the line search cannot decrease the objective further.".into();
            break;
        }
        let s: Vec<f64> = (0..n).map(|i| xn[i] - x[i]).collect();
        let gn = grad(&xn, fnew);
        let y: Vec<f64> = (0..n).map(|i| gn[i] - g[i]).collect();
        let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();
        let fold = f;
        x = xn.clone();
        f = fnew;
        g = gn;
        if opts.verbose {
            println!(
                "{:>6}{:>12}{:>18.6}{:>14.4e}{:>18.3e}",
                iter,
                ev.count(),
                f,
                norm2(&s),
                inf_norm(&proj_grad(&x, &g))
            );
        }
        if sy > 1e-12 * norm2(&s) * norm2(&y) {
            if first {
                let yy: f64 = y.iter().map(|a| a * a).sum();
                let sc = sy / yy;
                for i in 0..n {
                    h[i * n..(i + 1) * n].iter_mut().for_each(|v| *v = 0.0);
                    h[i * n + i] = sc;
                }
                first = false;
            }
            // H+ = (I - rho s y') H (I - rho y s') + rho s s'
            let rho = 1.0 / sy;
            let hy: Vec<f64> = (0..n).map(|i| (0..n).map(|j| h[i * n + j] * y[j]).sum()).collect();
            let yhy: f64 = y.iter().zip(&hy).map(|(a, b)| a * b).sum();
            for i in 0..n {
                for j in 0..n {
                    h[i * n + j] += -rho * (hy[i] * s[j] + s[i] * hy[j]) + (rho * rho * yhy + rho) * s[i] * s[j];
                }
            }
        }
        if norm2(&s) < opts.tol_x * (1.0 + norm2(&x)) {
            exitflag = 2;
            message = "Local minimum possible: the step is below StepTolerance.".into();
            break;
        }
        small_steps = if (fold - f).abs() < opts.tol_fun * (1.0 + fold.abs()) { small_steps + 1 } else { 0 };
        if small_steps >= 2 {
            exitflag = 3;
            message = "Local minimum possible: the change in the objective is below FunctionTolerance.".into();
            break;
        }
    }
    Ok(OptimResult {
        x,
        fval: f,
        hessian: None,
        inv_hessian: None,
        iterations: iter,
        func_count: ev.count(),
        exitflag,
        message,
    })
}
