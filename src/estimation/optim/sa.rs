// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `optimization/simulated_annealing.m` (Corana et al. 1987 /
//! Goffe, Ferrier and Rogers 1994), Dynare's `mode_compute = 2`, with the
//! defaults of `options_.saopt` and the `optim` keys of
//! `dynare_minimize_objective.m` (case 2).
//!
//! The random numbers come from a ChaCha8 stream (`seed`, `stream`), drawn in
//! the order of the MATLAB code (one uniform per trial coordinate, one more
//! when a trial falls out of bounds, one per Metropolis test), so the run has
//! the structure of Dynare's but not its draws.

use anyhow::{Result, bail};
use rand::Rng;

use super::OptimResult;
use super::minimize::{kv_num, unquote};
use crate::estimation::Objective;
use crate::estimation::mcmc::rng_stream;
use crate::modfile::OptValue;

/// `options_.saopt` (`default_option_values.m`).
#[derive(Clone, Debug)]
pub struct SaOptions {
    /// Number of final function values used to decide upon termination.
    pub neps: usize,
    /// 0 = minimize (Dynare's objective is minus the log posterior).
    pub maximizer_indicator: bool,
    /// Temperature reduction factor.
    pub rt: f64,
    /// Maximum number of function evaluations.
    pub max_iter: usize,
    pub verbosity: i32,
    pub tol_fun: f64,
    pub initial_temperature: f64,
    /// Cycles before a step-length adjustment.
    pub ns: usize,
    /// Step-length adjustments before a temperature reduction.
    pub nt: usize,
    /// Step-length adjustment factor (expanded to one per parameter).
    pub step_length_c: f64,
    /// Initial step length (expanded to one per parameter).
    pub initial_step_length: f64,
    pub seed: u64,
    pub stream: u64,
    /// Parameter names for the start table printed by `dynare_minimize_objective.m`.
    pub names: Vec<String>,
}

impl Default for SaOptions {
    fn default() -> Self {
        SaOptions {
            neps: 10,
            maximizer_indicator: false,
            rt: 0.10,
            max_iter: 100_000,
            verbosity: 1,
            tol_fun: 1e-8,
            initial_temperature: 15.0,
            ns: 10,
            nt: 10,
            step_length_c: 0.1,
            initial_step_length: 1.0,
            seed: 0,
            stream: 0,
            names: Vec::new(),
        }
    }
}

impl SaOptions {
    /// The `optim` keys of `dynare_minimize_objective.m`, case 2.
    pub fn apply_optim(&mut self, kv: &[(String, OptValue)]) -> Result<()> {
        for (k, v) in kv {
            let x = || kv_num(k, v);
            match unquote(k).as_str() {
                "neps" => self.neps = x()? as usize,
                "rt" => self.rt = x()?,
                "MaxIter" => self.max_iter = x()? as usize,
                "TolFun" => self.tol_fun = x()?,
                "verbosity" => self.verbosity = x()? as i32,
                "initial_temperature" => self.initial_temperature = x()?,
                "ns" => self.ns = x()? as usize,
                "nt" => self.nt = x()? as usize,
                "step_length_c" => self.step_length_c = x()?,
                "initial_step_length" => self.initial_step_length = x()?,
                other => println!("WARNING: simulated_annealing: Unknown option ({other})!"),
            }
        }
        Ok(())
    }
}

/// Port of `simulated_annealing.m` as `dynare_minimize_objective.m` calls it:
/// `lb`/`ub` already finite (`set_bounds_to_finite_values`), the raw objective
/// (`+inf` on failure) minimized. `iterations` and `func_count` are
/// `n_total_draws`; `exitflag` is 0 (termination criteria met) or 1 (too many
/// evaluations). Errors on a non-positive temperature or an out-of-bounds start
/// (Dynare's exitflags 3 and 2, after which its `fopt` is unassigned).
pub fn simulated_annealing(
    obj: &dyn Objective,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    opts: &SaOptions,
) -> Result<OptimResult> {
    let n = x0.len();
    let sign = if opts.maximizer_indicator { 1.0 } else { -1.0 };
    let fcn = |x: &[f64]| sign * obj.eval(x).0;
    let c = vec![opts.step_length_c; n];
    let mut t = opts.initial_temperature;
    let mut vm = vec![opts.initial_step_length; n];
    let mut rng = rng_stream(opts.seed, opts.stream);
    let mut x = x0.to_vec();
    let mut xp = vec![0.0; n];
    let (mut n_accepted, mut n_oob, mut n_total) = (0usize, 0usize, 0usize);
    let mut xopt = x.clone();
    let mut nacp = vec![0usize; n];
    let mut fstar = vec![1e20; opts.neps.max(1)];
    if opts.verbosity > 0 {
        println!("\nNumber of parameters= {n}, initial temperatur= {:4.3} ", opts.initial_temperature);
        println!("rt=  {:4.3}; TolFun=  {:4.3}; ns=  {:4.3};", opts.rt, opts.tol_fun, opts.ns as f64);
        println!("nt=  {:4.3}; neps=  {:4.3}; MaxIter=  {}", opts.nt as f64, opts.neps as f64, opts.max_iter);
        println!(
            "Initial step length(vm): {:4.3}; step_length_c: {:4.3}",
            opts.initial_step_length, opts.step_length_c
        );
        if opts.names.len() == n {
            println!("{:<20}  {:<6}    {:<6}    {:<6}", "Name:", "LB;", "Start;", "UB;");
            for i in 0..n {
                println!("{:<20}  {:6.4};   {:6.4};  {:6.4};", opts.names[i], lb[i], x0[i], ub[i]);
            }
        }
    }
    if t <= 0.0 {
        let m = "The initial temperature is not positive. Reset the variable t";
        println!("\n{m}");
        bail!("simulated_annealing: {m}");
    }
    if x.iter().zip(ub).any(|(a, b)| a > b) || x.iter().zip(lb).any(|(a, b)| a < b) {
        let m = "Initial condition out of bounds";
        println!("\n{m}");
        bail!("simulated_annealing: {m}");
    }
    let mut f = fcn(&x);
    n_total += 1;
    let mut fopt = f;
    fstar[0] = f;
    if opts.verbosity > 1 {
        println!("\ninitial x    {x:?}");
        println!("initial f    {}", sign * f);
    }
    let finish =
        |xopt: Vec<f64>, fopt: f64, n_total: usize, n_accepted: usize, n_oob: usize, exitflag: i32, message: &str| {
            if opts.verbosity > 0 {
                println!("Accepted draws = {n_accepted}\nTotal draws = {n_total}\nOut of bounds draws = {n_oob}");
            }
            OptimResult {
                x: xopt,
                fval: sign * fopt,
                hessian: None,
                inv_hessian: None,
                iterations: n_total,
                func_count: n_total,
                exitflag,
                message: message.to_string(),
            }
        };
    loop {
        let (mut nup, mut nrej, mut nnew, mut ndown, mut lnobds) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for _m in 0..opts.nt {
            for _j in 0..opts.ns {
                for h in 0..n {
                    // Trial value of x, varying coordinate h by up to vm(h).
                    for i in 0..n {
                        xp[i] = if i == h { x[i] + (rng.r#gen::<f64>() * 2.0 - 1.0) * vm[i] } else { x[i] };
                        if xp[i] < lb[i] || xp[i] > ub[i] {
                            xp[i] = lb[i] + (ub[i] - lb[i]) * rng.r#gen::<f64>();
                            lnobds += 1;
                            n_oob += 1;
                        }
                    }
                    let fp = fcn(&xp);
                    n_total += 1;
                    if opts.verbosity >= 3 {
                        println!("Current f   {}  Trial x {xp:?}  Resulting f {}", sign * f, sign * fp);
                    }
                    if n_total >= opts.max_iter {
                        let m = "Too many function evaluations; consider increasing optim.MaxIter or optim.TolFun or decreasing optim.nt or optim.rt. These results are likely to be poor";
                        println!("\n{m}");
                        return Ok(finish(xopt, fopt, n_total, n_accepted, n_oob, 1, m));
                    }
                    if fp >= f {
                        x.copy_from_slice(&xp);
                        f = fp;
                        n_accepted += 1;
                        nacp[h] += 1;
                        nup += 1;
                        if fp > fopt {
                            xopt.copy_from_slice(&xp);
                            fopt = fp;
                            nnew += 1;
                        }
                    } else {
                        // Metropolis criterion.
                        let p = ((fp - f) / t).exp();
                        let pp = rng.r#gen::<f64>();
                        if pp < p {
                            x.copy_from_slice(&xp);
                            f = fp;
                            n_accepted += 1;
                            nacp[h] += 1;
                            ndown += 1;
                        } else {
                            nrej += 1;
                        }
                    }
                }
            }
            // Adjust vm so that about half of the evaluations are accepted.
            for i in 0..n {
                let ratio = nacp[i] as f64 / opts.ns as f64;
                if ratio > 0.6 {
                    vm[i] *= 1.0 + c[i] * (ratio - 0.6) / 0.4;
                } else if ratio < 0.4 {
                    vm[i] /= 1.0 + c[i] * ((0.4 - ratio) / 0.4);
                }
                if vm[i] > ub[i] - lb[i] {
                    vm[i] = ub[i] - lb[i];
                }
            }
            if opts.verbosity >= 2 {
                println!("intermediate results after step length adjustment");
                println!("new step length(vm)  {vm:?}");
                println!("current optimal x    {xopt:?}");
                println!("current x            {x:?}");
            }
            nacp.iter_mut().for_each(|v| *v = 0);
        }
        if opts.verbosity >= 1 {
            let label = |a: &str, b: &str| if opts.maximizer_indicator { a.to_string() } else { b.to_string() };
            println!("\nIntermediate results before next temperature reduction");
            println!("current temperature         {}", num2str(&[t]));
            println!(
                "{}   {}",
                label("Max function value so far", "Min function value so far"),
                num2str(&[sign * fopt])
            );
            println!("Total moves                 {}", nup + ndown + nrej);
            println!("{}{nup}", label("Uphill                      ", "Downhill                    "));
            println!("{}{ndown}", label("Accepted downhill           ", "Accepted uphill             "));
            println!("{}{nrej}", label("Rejected downhill           ", "Rejected uphill             "));
            println!("Trials out of bounds        {lnobds}");
            println!("{}{nnew}", label("New maxima this temperature ", "New minima this temperature "));
            let half = (xopt.len() as f64 / 2.0).round() as usize;
            println!("Current optimal x1           {}", num2str(&xopt[..half]));
            println!("Current optimal x2           {}", num2str(&xopt[half..]));
            println!("Strength(vm)                {}", num2str(&vm));
            println!();
        }
        // Termination criteria.
        let mut quit = false;
        fstar[0] = f;
        if fopt - fstar[0] <= opts.tol_fun {
            quit = true;
        }
        if fstar.iter().any(|&fs| (f - fs).abs() > opts.tol_fun) {
            quit = false;
        }
        if quit {
            let m = "SA achieved termination criteria.";
            if opts.verbosity >= 1 {
                println!("{m}");
            }
            return Ok(finish(xopt, fopt, n_total, n_accepted, n_oob, 0, m));
        }
        t *= opts.rt;
        for i in (1..fstar.len()).rev() {
            fstar[i] = fstar[i - 1];
        }
        f = fopt;
        x.copy_from_slice(&xopt);
    }
}

/// MATLAB `num2str` of a real row vector: integers as `%{w}d` columns, otherwise
/// `%{ndgt+7+any(x<0)}.{ndgt}g` with `ndgt = max(floor(log10(max|x|)) + 5, 5)`, blanks trimmed.
fn num2str(x: &[f64]) -> String {
    use crate::analysis::table::fmt_g;
    if x.is_empty() {
        return String::new();
    }
    let neg = x.iter().any(|&v| v < 0.0) as usize;
    if x.iter().all(|v| v.is_finite() && v.fract() == 0.0) {
        let mx = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let ndgt = if mx == 0.0 { 1 } else { (mx.log10().floor() as usize) + 1 };
        let w = ndgt + 2 + neg;
        return x.iter().map(|v| format!("{:>w$}", *v as i64)).collect::<String>().trim().to_string();
    }
    let mx = x.iter().filter(|v| !v.is_infinite()).fold(0.0f64, |m, v| m.max(v.abs()));
    let mut ndgt = if mx == 0.0 || !mx.is_finite() { 0 } else { mx.log10().floor() as i64 };
    ndgt = (ndgt + 5).clamp(5, 16);
    let w = ndgt as usize + 7 + neg;
    x.iter().map(|&v| fmt_g(v, w, ndgt as usize)).collect::<String>().trim().to_string()
}
