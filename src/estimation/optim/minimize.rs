// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `optimization/dynare_minimize_objective.m`: one entry point per
//! `mode_compute` value, with Dynare's defaults (`default_option_values.m`)
//! and the `optim = ('Key', value, ...)` overrides each branch accepts
//! (`read_key_value_string.m`).
//!
//! Optimizers that Dynare delegates to MATLAB toolboxes (1 = `fmincon`,
//! 3 = `fminunc`, 12 = `particleswarm`) are replaced by documented substitutes
//! that are *not* MATLAB's algorithms (a note is printed when they run).

use anyhow::{Result, anyhow, bail};

use super::{
    CmaesOptions, CsminwelOptions, FminsearchOptions, GradientMethod, NewratOptions, OptimResult, ParticleswarmOptions,
    QuasiNewtonOptions, SaOptions, SimplexOptions, SimpsaOptions, SolveoptOptions, cmaes, cmaes_insigma, csminwel,
    dynare_simplex, fminsearch, newrat, particleswarm, quasi_newton, simpsa, simulated_annealing, solveopt,
};
use crate::estimation::{Bounds, Objective};
use crate::modfile::OptValue;

/// `options_.huge_number` (`set_bounds_to_finite_values`).
pub const HUGE_NUMBER: f64 = 1e7;

// ---- estimation completion: `huge_number = DOUBLE` of the estimation command
thread_local! {
    static HUGE: std::cell::Cell<f64> = const { std::cell::Cell::new(HUGE_NUMBER) };
}

/// The `options_.huge_number` [`minimize`] uses on this thread (mode_compute 2 and 10).
pub fn huge_number() -> f64 {
    HUGE.with(|h| h.get())
}

/// Sets `options_.huge_number` for the [`minimize`] calls of this thread (the estimation
/// driver's) until the guard is dropped; `None` keeps the current value.
pub struct HugeNumberGuard(f64);

impl HugeNumberGuard {
    pub fn set(v: Option<f64>) -> Self {
        let old = huge_number();
        if let Some(v) = v {
            HUGE.with(|h| h.set(v));
        }
        HugeNumberGuard(old)
    }
}

impl Drop for HugeNumberGuard {
    fn drop(&mut self) {
        HUGE.with(|h| h.set(self.0));
    }
}
// ---- end estimation completion

// ---- estimation completion: `analytic_derivation` of the estimation command
thread_local! {
    static ANALYTIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `options_.analytic_derivation` for the [`minimize`] calls of this thread.
pub fn analytic_derivation() -> bool {
    ANALYTIC.with(|a| a.get())
}

/// Sets `options_.analytic_derivation` for the [`minimize`] calls of this thread (the
/// estimation driver's mode finding) until the guard is dropped.
pub struct AnalyticGuard(bool);

impl AnalyticGuard {
    pub fn set(v: bool) -> Self {
        let old = analytic_derivation();
        ANALYTIC.with(|a| a.set(v));
        AnalyticGuard(old)
    }
}

impl Drop for AnalyticGuard {
    fn drop(&mut self) {
        ANALYTIC.with(|a| a.set(self.0));
    }
}
// ---- end estimation completion

/// `options_.gradient_method` and `options_.gradient_epsilon` (`options_.X = V;` statements in
/// the .mod, defaults 2 and 1e-6): csminwel's numerical gradient and newrat's initial steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientOptions {
    pub method: f64,
    pub epsilon: f64,
}

impl Default for GradientOptions {
    fn default() -> Self {
        GradientOptions { method: 2.0, epsilon: 1e-6 }
    }
}

thread_local! {
    static GRADIENT: std::cell::Cell<GradientOptions> = std::cell::Cell::new(GradientOptions::default());
}

/// The `options_.gradient_method/gradient_epsilon` [`minimize`] uses on this thread.
pub fn gradient_options() -> GradientOptions {
    GRADIENT.with(|g| g.get())
}

/// Sets [`gradient_options`] for the [`minimize`] calls of this thread until the guard is
/// dropped (the estimation driver's thread, and the MCMC chains' for the slice sampler's
/// `maximize` and TaRB's optimizer).
pub struct GradientGuard(GradientOptions);

impl GradientGuard {
    pub fn set(v: GradientOptions) -> Self {
        let old = gradient_options();
        GRADIENT.with(|g| g.set(v));
        GradientGuard(old)
    }
}

impl Drop for GradientGuard {
    fn drop(&mut self) {
        GRADIENT.with(|g| g.set(self.0));
    }
}

/// What `dynare_minimize_objective.m` receives besides the objective and the
/// start: bounds, names, prior information, the `optim` string, and (Rust
/// only) the random stream for the stochastic optimizers.
#[derive(Clone, Copy)]
pub struct MinimizeContext<'a> {
    /// `[bounds.lb bounds.ub]`.
    pub bounds: &'a Bounds,
    /// `bayestopt_.name`.
    pub names: &'a [String],
    /// `bayestopt_.p2` (prior standard deviations; NaN/inf when absent).
    pub prior_sd: &'a [f64],
    /// `options_.optim_opt` (None = empty).
    pub optim: Option<&'a OptValue>,
    /// `options_.silent_optimizer` is the negation.
    pub verbose: bool,
    /// Seed and stream of the ChaCha8 generator of stochastic optimizers
    /// (simulated annealing, CMA-ES, SIMPSA, particle swarm).
    pub seed: u64,
    pub stream: u64,
}

/// Port of `read_key_value_string.m` on the parsed `optim` option:
/// `('MaxIter', 100, 'TolFun', 1e-6)` -> `[("MaxIter", Num(100)), ("TolFun", Num(1e-6))]`.
pub fn key_values(optim: Option<&OptValue>, who: &str) -> Result<Vec<(String, OptValue)>> {
    let Some(optim) = optim else { return Ok(vec![]) };
    let items: Vec<&OptValue> = match optim {
        OptValue::List(v) => v.iter().collect(),
        other => vec![other],
    };
    if !items.len().is_multiple_of(2) {
        bail!("{who}: optim must be a list of key/value pairs");
    }
    items
        .chunks(2)
        .map(|kv| {
            let key = match kv[0] {
                OptValue::Str(s) => unquote(s),
                _ => bail!("{who}: optim keys must be strings"),
            };
            let val = match kv[1] {
                OptValue::Str(s) => OptValue::Str(unquote(s)),
                v => v.clone(),
            };
            Ok((key, val))
        })
        .collect()
}

pub(crate) fn unquote(s: &str) -> String {
    s.trim().trim_matches(|c| c == '\'' || c == '"').to_string()
}

/// A numeric `optim` value (MATLAB `eval`s strings such as `'1e-5'` too, and the logicals
/// `true`/`false` act as 1/0).
pub fn kv_num(key: &str, v: &OptValue) -> Result<f64> {
    match v {
        OptValue::Num(x) => Ok(*x),
        OptValue::Str(s) if s.trim() == "true" => Ok(1.0),
        OptValue::Str(s) if s.trim() == "false" => Ok(0.0),
        OptValue::Str(s) => s.trim().parse::<f64>().map_err(|_| anyhow!("optim: the value of {key} must be a number")),
        OptValue::Vec(x) if x.len() == 1 => Ok(x[0]),
        _ => bail!("optim: the value of {key} must be a number"),
    }
}

/// A string `optim` value.
pub fn kv_str(key: &str, v: &OptValue) -> Result<String> {
    match v {
        OptValue::Str(s) => Ok(unquote(s)),
        OptValue::Num(x) => Ok(format!("{x}")),
        _ => bail!("optim: the value of {key} must be a string"),
    }
}

/// csminwel's `NumgradAlgorithm`: dynare_minimize_objective.m takes any value, and
/// csminwel1.m's `get_num_grad` errors on one other than 2, 3, 5, 13 or 15 when it computes a
/// numerical gradient (never with an analytic gradient). `None`: keep the default.
pub(crate) fn numgrad_algorithm(key: &str, v: &OptValue, analytic: bool) -> Result<Option<GradientMethod>> {
    match GradientMethod::from_code(kv_num(key, v)? as i32) {
        Some(m) => Ok(Some(m)),
        None if analytic => Ok(None),
        None => bail!("csminwel1: Unknown method for gradient evaluation!"),
    }
}

/// csminwel's `InitialInverseHessian` (`H0 = eval(...)`): a number stays a scalar
/// ([`CsminwelOptions::h0_scalar`]); `A*eye(n)`, `eye(n)*A` and `eye(n)` (with `n` the number
/// of parameters or `n_params`) give the matrix. Other MATLAB expressions are not evaluated.
pub(crate) fn initial_inverse_hessian(o: &mut CsminwelOptions, v: &OptValue, n: usize) -> Result<()> {
    if let Ok(s) = kv_num("InitialInverseHessian", v) {
        o.h0_scalar = Some(s);
        o.h0 = None;
        return Ok(());
    }
    let unsupported = || {
        anyhow!(
            "csminwel: InitialInverseHessian: only a number or a matrix A*eye(n) is supported (dynare-rust does not \
             evaluate other MATLAB expressions)"
        )
    };
    let OptValue::Str(s) = v else { return Err(unsupported()) };
    let s: String = unquote(s).chars().filter(|c| !c.is_whitespace()).collect();
    let mut scale = 1.0;
    let mut eye = None;
    for factor in s.split('*') {
        if let Some(arg) = factor.strip_prefix("eye(").and_then(|r| r.strip_suffix(')')) {
            if eye.is_some() {
                return Err(unsupported());
            }
            let m = if arg == "n_params" { n } else { arg.parse::<usize>().map_err(|_| unsupported())? };
            eye = Some(m);
        } else {
            scale *= factor.parse::<f64>().map_err(|_| unsupported())?;
        }
    }
    let m = eye.ok_or_else(unsupported)?;
    if m != n {
        bail!("csminwel: InitialInverseHessian is {m}x{m} but there are {n} parameters");
    }
    o.h0 = Some(crate::linalg::Mat::identity(n, n) * scale);
    o.h0_scalar = None;
    Ok(())
}

/// `set_bounds_to_finite_values(bounds, options_.huge_number)`.
pub fn finite_bounds(bounds: &Bounds, huge_number: f64) -> (Vec<f64>, Vec<f64>) {
    let lb = bounds.lb.iter().map(|&v| if v.is_infinite() { -huge_number } else { v }).collect();
    let ub = bounds.ub.iter().map(|&v| if v.is_infinite() { huge_number } else { v }).collect();
    (lb, ub)
}

/// Port of the `switch minimizer_algorithm` of `dynare_minimize_objective.m`
/// for every `mode_compute` except 6 (gmhmaxlik, which also returns a scale and
/// is called by the driver directly) and 0. `obj` is the raw objective
/// (`dsge_likelihood`: +inf on failure); each optimizer wraps it the way its
/// MATLAB counterpart does (penalty function for csminwel/newrat/particle swarm).
pub fn minimize(optimizer: i32, obj: &dyn Objective, x0: &[f64], ctx: &MinimizeContext) -> Result<OptimResult> {
    let kv = |who: &str| key_values(ctx.optim, who);
    // `options_.analytic_derivation || options_.mom.analytic_jacobian == 1` (fmincon, fminunc,
    // csminwel and solvopt; newrat only follows analytic_derivation)
    let analytic = analytic_derivation() || obj.has_gradient();
    let note = |what: &str| {
        println!(
            "NOTE: mode_compute={optimizer}: Dynare calls MATLAB's {what} here; dynare-rust runs its own substitute \
             (see src/estimation/optim/), not the toolbox algorithm. Iterates differ from MATLAB's."
        )
    };
    match optimizer {
        1 => {
            note("fmincon (Optimization Toolbox)");
            let mut o = QuasiNewtonOptions::fmincon();
            o.verbose = ctx.verbose;
            if analytic {
                // ---- estimation completion ---- optimoptions(..., 'GradObj','on','TolX',1e-7)
                o.analytic = true;
                o.tol_x = 1e-7;
            }
            o.apply_optim(&kv("fmincon")?, "fmincon")?;
            quasi_newton(obj, x0, Some((&ctx.bounds.lb, &ctx.bounds.ub)), &o)
        }
        2 => {
            let mut o =
                SaOptions { seed: ctx.seed, stream: ctx.stream, names: ctx.names.to_vec(), ..Default::default() };
            o.apply_optim(&kv("simulated_annealing")?)?;
            if !ctx.verbose {
                o.verbosity = 0;
            }
            let (lb, ub) = finite_bounds(ctx.bounds, huge_number()); // ---- estimation completion
            simulated_annealing(obj, x0, &lb, &ub, &o)
        }
        3 => {
            note("fminunc (Optimization Toolbox)");
            let mut o = QuasiNewtonOptions::fminunc();
            o.verbose = ctx.verbose;
            o.analytic = analytic; // ---- estimation completion ---- 'GradObj','on'
            o.apply_optim(&kv("fminunc")?, "fminunc")?;
            quasi_newton(obj, x0, None, &o)
        }
        4 => {
            let mut o = CsminwelOptions { verbose: ctx.verbose, parallel: true, analytic, ..Default::default() };
            // numgrad = options_.gradient_method; epsilon = options_.gradient_epsilon
            let grad = gradient_options();
            if let Some(m) = numgrad_algorithm("gradient_method", &OptValue::Num(grad.method), o.analytic)? {
                o.gradient_method = m;
            }
            o.gradient_epsilon = grad.epsilon;
            for (k, v) in kv("csminwel")? {
                match k.as_str() {
                    "MaxIter" => o.nit = kv_num(&k, &v)? as usize,
                    "TolFun" => o.crit = kv_num(&k, &v)?,
                    "NumgradAlgorithm" => {
                        if let Some(m) = numgrad_algorithm(&k, &v, o.analytic)? {
                            o.gradient_method = m;
                        }
                    }
                    "NumgradEpsilon" => o.gradient_epsilon = kv_num(&k, &v)?,
                    "verbosity" => o.verbose = ctx.verbose && kv_num(&k, &v)? != 0.0,
                    "InitialInverseHessian" => initial_inverse_hessian(&mut o, &v, x0.len())?,
                    "SaveFiles" => {}
                    other => println!("WARNING: csminwel: Unknown option ({other})!"),
                }
            }
            Ok(csminwel(obj, x0, &o))
        }
        5 => {
            let mut o = NewratOptions {
                verbose: ctx.verbose,
                parallel: true,
                names: ctx.names.to_vec(),
                // hess_info.h1 = options_.gradient_epsilon*ones(n_params,1)
                gradient_epsilon: gradient_options().epsilon,
                ..Default::default()
            };
            if analytic_derivation() {
                // ---- estimation completion ---- crit = options_.newrat.tolerance.f_analytic
                o.analytic = true;
                o.ftol = 1e-7;
            }
            for (k, v) in kv("newrat")? {
                match k.as_str() {
                    "MaxIter" => o.nit = kv_num(&k, &v)? as usize,
                    "Hessian" => {
                        // ---- estimation completion ---- read by the driver (newratflag)
                        let flag = kv_num(&k, &v)?;
                        if o.analytic && flag != 0.0 {
                            bail!(
                                "newrat: analytic_derivation is incompatible with numerical Hessian. Using analytic Hessian"
                            );
                        }
                    }
                    "robust" => o.robust = kv_num(&k, &v)? != 0.0,
                    "TolFun" => o.ftol = kv_num(&k, &v)?,
                    "TolGstep" => o.gstep_crit = kv_num(&k, &v)?,
                    "TolGstepRel" => o.gstep_crit_rel = kv_num(&k, &v)?,
                    "verbosity" => o.verbose = ctx.verbose && kv_num(&k, &v)? != 0.0,
                    "SaveFiles" => {}
                    other => println!("WARNING: newrat: Unknown option ({other})!"),
                }
            }
            Ok(newrat(obj, x0, ctx.bounds, ctx.prior_sd, &o))
        }
        7 => {
            let mut o = FminsearchOptions::default();
            for (k, v) in kv("fminsearch")? {
                match k.as_str() {
                    "MaxFunEvals" => o.max_fun_evals = kv_num(&k, &v)? as usize,
                    "MaxIter" => o.max_iter = kv_num(&k, &v)? as usize,
                    "TolFun" => o.tol_fun = kv_num(&k, &v)?,
                    "TolX" => o.tol_x = kv_num(&k, &v)?,
                    "Display" | "display" => {}
                    other => println!("WARNING: fminsearch: option {other} is ignored."),
                }
            }
            Ok(fminsearch(obj, x0, &o))
        }
        8 => {
            let mut o = SimplexOptions::default();
            o.apply_optim(&kv("simplex")?)?;
            if !ctx.verbose {
                o.verbosity = 0;
            }
            dynare_simplex(obj, x0, &o)
        }
        9 => {
            let h0 = cmaes_insigma(ctx.bounds);
            let mut o = CmaesOptions { verbose: ctx.verbose, seed: ctx.seed, stream: ctx.stream, ..Default::default() };
            o.apply_optim(&kv("cmaes")?)?;
            cmaes(obj, x0, &h0, ctx.bounds, &o)
        }
        10 => {
            let mut o = SimpsaOptions { seed: ctx.seed, stream: ctx.stream, ..Default::default() };
            o.apply_optim(&kv("simpsa")?)?;
            if !ctx.verbose {
                o.display = false;
            }
            let (lb, ub) = finite_bounds(ctx.bounds, huge_number()); // ---- estimation completion
            simpsa(obj, x0, &lb, &ub, &o)
        }
        11 => bail!(
            "Online particle filter is no more available with mode_compute=11; use posterior_sampler with option online"
        ),
        12 => {
            let mut o =
                ParticleswarmOptions { verbose: ctx.verbose, seed: ctx.seed, stream: ctx.stream, ..Default::default() };
            o.apply_optim(&kv("particleswarm")?)?;
            particleswarm(obj, x0, ctx.bounds, &o)
        }
        13 => bail!("Option mode_compute=13 is available only for method_of_moments estimation."),
        101 => {
            let mut o = SolveoptOptions { analytic, ..Default::default() }; // ---- estimation completion
            o.apply_optim(&kv("solveopt")?)?;
            if !ctx.verbose {
                o.verbosity = -1.0;
            }
            solveopt(obj, x0, &o)
        }
        102 => {
            // simulannealbnd substitute (src/estimation/optim/simulannealbnd.rs)
            note("simulannealbnd (Global Optimization Toolbox)");
            let mut o = super::simulannealbnd::SimulannealbndOptions {
                verbose: ctx.verbose,
                seed: ctx.seed,
                stream: ctx.stream,
                ..Default::default()
            };
            o.apply_optim(&kv("simulannealbnd")?)?;
            super::simulannealbnd::simulannealbnd(obj, x0, ctx.bounds, &o)
        }
        6 => bail!("mode_compute=6 is run by the driver (gmhmaxlik)"),
        k => bail!("Optimization algorithm {k} is unknown!"),
    }
}
