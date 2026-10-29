// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `optimization/simplex_optimization_routine.m`, Dynare's own
//! Nelder–Mead variant (`mode_compute = 8`): optimal expansion step, restarts
//! from a larger simplex after `10*n` iterations without improvement, up to
//! three simplex "algorithm iterations" (`max_simplex_algo_iterations`).

use anyhow::{Result, bail};

use super::OptimResult;
use super::minimize::kv_num;
use crate::estimation::Objective;
use crate::modfile::OptValue;

/// `options_.simplex` (`default_option_values.m`).
#[derive(Clone, Debug)]
pub struct SimplexOptions {
    /// `tolerance.x` (1e-4).
    pub tol_x: f64,
    /// `tolerance.f` (1e-4).
    pub tol_f: f64,
    /// `maxiter` (10000).
    pub maxiter: usize,
    /// `maxfcallfactor` (500): `max_func_calls = maxfcallfactor * n` unless `maxfcall` is set.
    pub maxfcallfactor: usize,
    /// `maxfcall` (empty).
    pub maxfcall: Option<usize>,
    /// `verbosity` (2).
    pub verbosity: u32,
    /// `delta_factor` (0.05): relative size of the initial simplex.
    pub delta_factor: f64,
}

impl Default for SimplexOptions {
    fn default() -> Self {
        SimplexOptions {
            tol_x: 1e-4,
            tol_f: 1e-4,
            maxiter: 10000,
            maxfcallfactor: 500,
            maxfcall: None,
            verbosity: 2,
            delta_factor: 0.05,
        }
    }
}

impl SimplexOptions {
    /// The `optim` keys of `dynare_minimize_objective.m`, case 8.
    pub fn apply_optim(&mut self, kv: &[(String, OptValue)]) -> Result<()> {
        for (k, v) in kv {
            match k.as_str() {
                "MaxIter" => self.maxiter = kv_num(k, v)? as usize,
                "TolFun" => self.tol_f = kv_num(k, v)?,
                "TolX" => self.tol_x = kv_num(k, v)?,
                "MaxFunEvals" => self.maxfcall = Some(kv_num(k, v)? as usize),
                "MaxFunEvalFactor" => self.maxfcallfactor = kv_num(k, v)? as usize,
                "InitialSimplexSize" => self.delta_factor = kv_num(k, v)?,
                "verbosity" => self.verbosity = kv_num(k, v)? as u32,
                other => println!("WARNING: simplex: Unknown option ({other})!"),
            }
        }
        Ok(())
    }
}

/// MATLAB `eps(x)`.
pub(crate) fn eps_of(x: f64) -> f64 {
    let a = x.abs();
    if !a.is_finite() {
        return f64::NAN;
    }
    if a < f64::MIN_POSITIVE {
        return f64::from_bits(1);
    }
    f64::from_bits(a.to_bits() + 1) - a
}

/// MATLAB's ascending `sort` (stable, NaN last).
pub(crate) fn sort_simplex(v: &mut Vec<Vec<f64>>, fv: &mut Vec<f64>) {
    let mut idx: Vec<usize> = (0..fv.len()).collect();
    idx.sort_by(|&a, &b| match (fv[a].is_nan(), fv[b].is_nan()) {
        (false, false) => fv[a].partial_cmp(&fv[b]).unwrap(),
        (x, y) => x.cmp(&y),
    });
    *v = idx.iter().map(|&i| v[i].clone()).collect();
    *fv = idx.iter().map(|&i| fv[i]).collect();
}

struct Counter<'a> {
    obj: &'a dyn Objective,
    total: usize,
}

impl Counter<'_> {
    /// `[fval, ~, exit_flag] = feval(objective_function, x)`.
    fn eval(&mut self, x: &[f64]) -> (f64, bool) {
        self.total += 1;
        let (f, info) = self.obj.eval(x);
        (f, info == 0 && f.is_finite())
    }
}

/// `simplex_initialization` (with `check_delta = 1`). Returns `(v, fv, delta)`.
fn initialization(
    c: &mut Counter,
    point: &[f64],
    score: f64,
    delta: &[f64],
    zero_delta: f64,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = point.len();
    let mut delta = delta.to_vec();
    let mut zero_delta = zero_delta;
    let mut v = vec![point.to_vec()];
    let mut fv = vec![score];
    for j in 0..n {
        let make = |delta: &[f64], zero_delta: f64| {
            let mut y = point.to_vec();
            y[j] = if y[j] != 0.0 { (1.0 + delta[j]) * y[j] } else { zero_delta };
            y
        };
        let mut y = make(&delta, zero_delta);
        let (mut f, mut ok) = c.eval(&y);
        while !ok {
            if y[j] != 0.0 {
                delta[j] /= 1.1;
            } else {
                zero_delta /= 1.1;
            }
            y = make(&delta, zero_delta);
            (f, ok) = c.eval(&y);
        }
        v.push(y);
        fv.push(f);
    }
    sort_simplex(&mut v, &mut fv);
    (v, fv, delta)
}

/// Port of `simplex_optimization_routine.m` (raw objective, as Dynare calls it).
pub fn dynare_simplex(obj: &dyn Objective, x0: &[f64], opts: &SimplexOptions) -> Result<OptimResult> {
    let n = x0.len();
    let verbose = opts.verbosity > 0;
    let max_func_calls = opts.maxfcall.unwrap_or(opts.maxfcallfactor * n);
    let (x_tolerance, f_tolerance) = (opts.tol_x, opts.tol_f);
    let max_iterations = opts.maxiter;
    let (rho, chi, psi, sigma) = (1.0, 2.0, 0.5, 0.5);
    let optimize_expansion_parameter = true;
    let delta0 = opts.delta_factor;
    let zero_delta = delta0 / 200.0;
    let mut max_no_improvements = (n * 10) as f64;
    let mut c = Counter { obj, total: 0 };
    let mut iter_count_total = 0usize;

    if verbose {
        println!("\nSimplex initialization...");
    }
    let (initial_score, ok) = c.eval(x0);
    if !ok {
        println!("Cannot initialize the simplex with the provided initial guess.\n");
        bail!("simplex_optimization_routine:: Initial condition is wrong!");
    }
    let (mut v, mut fv, mut delta) = initialization(&mut c, x0, initial_score, &vec![delta0; n], zero_delta);
    let mut func_count = n + 1;
    let mut iter_count = 1usize;
    if verbose {
        println!("Done!\n\nObjective function value: {}\n", crate::gsa::stats::num2str(fv[0]));
    }
    let header = "#Iter.      #FnCalls.   Best Value      Worst Value     Norm(f)        Norm(x)        Move";
    let mut no_improvements = 0usize;
    let mut simplex_iterations = 1usize;
    let max_simplex_algo_iterations = 3usize;
    let mut simplex_algo_iterations = 1usize;
    let mut best_point: Vec<f64>;
    let mut best_point_score = fv[0];
    let mut convergence = false;
    let mut tooslow = false;
    let mut iter_no_improvement_break = 0usize;
    let max_no_improvement_break = 1usize;
    if verbose {
        println!("{header}");
    }
    let mut crit_f = 1.0_f64;
    let mut crit_x = 1.0_f64;
    let comb =
        |a: f64, xbar: &[f64], b: f64, w: &[f64]| -> Vec<f64> { (0..n).map(|k| a * xbar[k] + b * w[k]).collect() };

    while func_count < max_func_calls
        && iter_count < max_iterations
        && simplex_algo_iterations <= max_simplex_algo_iterations
    {
        let vmax = v[0].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if crit_f <= f_tolerance.max(10.0 * eps_of(fv[0])) && crit_x <= x_tolerance.max(10.0 * eps_of(vmax)) {
            convergence = true;
        }
        if crit_x <= x_tolerance * x_tolerance && crit_f > 1.0 {
            tooslow = true;
        }
        let xbar: Vec<f64> = (0..n).map(|k| v[..n].iter().map(|vj| vj[k]).sum::<f64>() / n as f64).collect();
        let worst = v[n].clone();
        // xbar + w*(xbar - worst)
        let along = |w: f64| -> Vec<f64> { (0..n).map(|k| xbar[k] + w * (xbar[k] - worst[k])).collect() };
        let xr = along(rho);
        let (fxr, _) = c.eval(&xr);
        func_count += 1;
        let mv: &str;
        if fxr < fv[0] {
            let mut xe = along(rho * chi);
            let (mut fxe, _) = c.eval(&xe);
            func_count += 1;
            if fxe < fxr {
                if optimize_expansion_parameter {
                    let xee = along(rho * chi * 1.01);
                    let (fxee, _) = c.eval(&xee);
                    func_count += 1;
                    let grow = fxee < fxe;
                    let mut weight = if grow { rho * chi * 1.02 } else { rho * chi };
                    let mut fxeee_old = fxee;
                    let mut xeee_old = xee;
                    loop {
                        weight = if grow { 1.02 * weight } else { weight / 1.02 };
                        let xeee = along(weight);
                        let (fxeee, _) = c.eval(&xeee);
                        func_count += 1;
                        if fxeee < fxeee_old && -(fxeee - fxeee_old) > f_tolerance * 10.0 * fxeee_old {
                            fxeee_old = fxeee;
                            xeee_old = xeee;
                        } else {
                            break;
                        }
                    }
                    xe = xeee_old;
                    fxe = fxeee_old;
                    mv = "expand-opt";
                } else {
                    mv = "expand";
                }
                v[n] = xe;
                fv[n] = fxe;
            } else {
                v[n] = xr;
                fv[n] = fxr;
                mv = "reflect-1";
            }
        } else if fxr < fv[n - 1] {
            v[n] = xr;
            fv[n] = fxr;
            mv = "reflect-0";
        } else {
            let mut shrink = false;
            if fxr < fv[n] {
                let xc = comb(1.0 + psi * rho, &xbar, -psi * rho, &worst);
                let (fxc, _) = c.eval(&xc);
                func_count += 1;
                if fxc <= fxr {
                    v[n] = xc;
                    fv[n] = fxc;
                    mv = "contract outside";
                } else {
                    mv = "shrink";
                    shrink = true;
                }
            } else {
                let xcc = comb(1.0 - psi, &xbar, psi, &worst);
                let (fxcc, _) = c.eval(&xcc);
                func_count += 1;
                if fxcc < fv[n] {
                    v[n] = xcc;
                    fv[n] = fxcc;
                    mv = "contract inside";
                } else {
                    mv = "shrink";
                    shrink = true;
                }
            }
            if shrink {
                for j in 1..=n {
                    let vj: Vec<f64> = (0..n).map(|k| v[0][k] + sigma * (v[j][k] - v[0][k])).collect();
                    fv[j] = c.eval(&vj).0;
                    v[j] = vj;
                }
                func_count += n;
            }
        }
        sort_simplex(&mut v, &mut fv);
        crit_f = fv[1..].iter().map(|f| (fv[0] - f).abs()).fold(f64::NEG_INFINITY, f64::max);
        crit_x = v[1..]
            .iter()
            .flat_map(|vj| vj.iter().zip(&v[0]).map(|(a, b)| (a - b).abs()))
            .fold(f64::NEG_INFINITY, f64::max);
        if verbose {
            if simplex_iterations.is_multiple_of(50) {
                println!("\n{header}");
            }
            // `%12.7E` of C (two-digit exponent with sign), as `dprintf` in the MATLAB routine
            let e = |x: f64| crate::analysis::table::fmt_e(x, 12, 7).replace('e', "E");
            if fv[n].is_finite() && fv[0].is_finite() {
                // the field separators depend on the signs of the best and worst values
                let (s1, s2) = if fv[n] < 0.0 {
                    ("  ", "  ")
                } else if fv[0] > 0.0 {
                    (" ", " ")
                } else {
                    ("  ", " ")
                };
                println!(
                    "{:<10}  {:<10}{s1}{}{s2}{}  {}  {}  {mv}",
                    simplex_iterations,
                    func_count,
                    e(fv[0]),
                    e(fv[n]),
                    e(crit_f),
                    e(crit_x)
                );
            } else if fv[0].is_finite() {
                println!(
                    "{:<10}  {:<10}  {}                                 {}  {mv}",
                    simplex_iterations,
                    func_count,
                    e(fv[0]),
                    e(crit_x)
                );
            } else {
                println!(
                    "{:<10}  {:<10}                                         {}  {mv}",
                    simplex_iterations,
                    func_count,
                    e(crit_x)
                );
            }
        }
        iter_count += 1;
        iter_count_total += 1;
        simplex_iterations += 1;
        if (best_point_score - fv[0]).abs() < f_tolerance {
            no_improvements += 1;
        } else {
            no_improvements = 0;
        }
        best_point = v[0].clone();
        best_point_score = fv[0];
        if no_improvements as f64 > max_no_improvements {
            if verbose {
                println!("\nNO SIGNIFICANT IMPROVEMENT AFTER {no_improvements} ITERATIONS!\n");
            }
            if simplex_algo_iterations <= max_simplex_algo_iterations {
                for d in delta.iter_mut() {
                    *d *= 1.05;
                }
                (v, fv, delta) = initialization(&mut c, &best_point, best_point_score, &delta, zero_delta);
                if verbose {
                    println!(
                        "(Re)Start with a lager simplex based on the best current values for the control variables.\n\n{header}"
                    );
                }
                no_improvements = 0;
                func_count += n;
                iter_count += 1;
                iter_count_total += 1;
                iter_no_improvement_break += 1;
                simplex_iterations += 1;
            }
        }
        if func_count == max_func_calls
            || iter_count == max_iterations
            || iter_no_improvement_break == max_no_improvement_break
            || convergence
            || tooslow
        {
            (v, fv, delta) = initialization(&mut c, &best_point, best_point_score, &vec![delta0; n], zero_delta);
            if func_count == max_func_calls {
                if verbose {
                    println!("\nMAXIMUM NUMBER OF OBJECTIVE FUNCTION CALLS EXCEEDED ({max_func_calls})!\n");
                }
            } else if iter_count == max_iterations {
                if verbose {
                    println!("\nMAXIMUM NUMBER OF ITERATIONS EXCEEDED ({max_iterations})!\n");
                }
            } else if iter_no_improvement_break == max_no_improvement_break {
                if verbose {
                    println!("\nMAXIMUM NUMBER OF SIMPLEX REINITIALIZATION EXCEEDED ({max_no_improvement_break})!\n");
                }
                iter_no_improvement_break = 0;
                if simplex_algo_iterations == max_simplex_algo_iterations {
                    max_no_improvements = f64::INFINITY;
                    continue;
                }
            } else if tooslow {
                if verbose {
                    println!(
                        "\nCONVERGENCE NOT ACHIEVED AFTER {simplex_iterations} ITERATIONS! IMPROVING TOO SLOWLY!\n"
                    );
                }
            } else if verbose {
                println!("\nCONVERGENCE ACHIEVED AFTER {simplex_iterations} ITERATIONS!\n");
            }
            if simplex_algo_iterations < max_simplex_algo_iterations {
                for d in delta.iter_mut() {
                    *d *= 1.05;
                }
                (v, fv, delta) = initialization(&mut c, &best_point, best_point_score, &delta, zero_delta);
                if verbose {
                    println!(
                        "(Re)Start with a lager simplex based on the best current values for the control variables.\n\n{header}"
                    );
                }
                convergence = false;
                no_improvements = 0;
                func_count = n;
                iter_count = 1;
                simplex_iterations += 1;
                simplex_algo_iterations += 1;
            } else {
                break;
            }
        }
    }
    let mut exitflag = 1;
    let mut message = String::new();
    if func_count >= max_func_calls {
        message.push_str("Maximum number of objective function calls has been exceeded!");
        exitflag = 0;
    }
    if iter_count >= max_iterations {
        message.push_str("Maximum number of iterations has been exceeded!");
        exitflag = 0;
    }
    if exitflag == 1 {
        message = "The function converged to a solution x.".into();
    }
    if verbose {
        println!("{message}");
    }
    Ok(OptimResult {
        x: v[0].clone(),
        fval: fv[0],
        hessian: None,
        inv_hessian: None,
        iterations: iter_count_total,
        func_count: c.total,
        exitflag,
        message,
    })
}
