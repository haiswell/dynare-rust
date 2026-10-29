// SPDX-License-Identifier: GPL-3.0-or-later
//! Deterministic (perfect-foresight) simulations: Dynare's `oo_` state for `initval` /
//! `endval` / `histval` / `steady`, `perfect_foresight_setup` (`make_ex_.m`, `make_y_.m`)
//! and `perfect_foresight_solver` (homotopy loop of `perfect_foresight_solver.m`,
//! `perfect_foresight_solver_core.m`, Newton on the stacked system of `sim1.m`, and
//! `sim1_linear.m` for `model(linear)` / `linear_approximation`), plus `rplot`.
//!
//! The driver keeps one [`DetState`] per run and forwards `steady`, `endval` and the
//! [`Command::Deterministic`] commands to it.

pub mod sparse_lu;
pub mod stacked;
pub mod steady;
// ---- iterative / LBJ / stacked-MCP solvers, lmmcp, complementarity conditions
pub mod block_solve;
pub mod diagnostics;
pub mod exo_det;
pub mod iterative;
pub mod lmmcp;
pub mod mcp;
pub mod periods; // ---- shocks/shock_paths ----
pub mod pf_q1;
pub mod shock_paths;
pub mod spmat;
pub mod workspace; // ---- shocks/shock_paths ----

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow, bail};

use crate::graphics::{self, Color, Figure, GraphFormat, GraphOptions, LineStyle, Panel, Series};
use crate::linalg::Mat;
use crate::model::Model;
use crate::modfile::{Command, DetShockKind, ModFile, OptValue, Options, opt};
use crate::solver::SteadyOptions;
pub use stacked::{StackedJacobian, StackedProblem};
pub use steady::{SteadyStateModel, evaluate_steady_state};

/// `options_` fields used by the deterministic commands (Dynare defaults). Options persist
/// across commands, as in Dynare.
#[derive(Clone, Debug)]
pub struct SimulOptions {
    /// `options_.periods` (0: the horizon comes from `first_simulation_period` and
    /// `last_simulation_period`).
    pub periods_option: usize,
    /// Number of simulation periods, `get_simulation_periods(options_)` (set by
    /// [`DetState::simulation_periods`]).
    pub periods: usize,
    /// `options_.simul.maxit` (50).
    pub maxit: usize,
    /// `options_.dynatol.f` / `.x` (1e-5).
    pub tolf: f64,
    pub tolx: f64,
    pub stack_solve_algo: u8,
    pub noprint: bool,
    pub verbosity: u8,
    pub no_homotopy: bool,
    pub homotopy_initial_step_size: f64,
    pub homotopy_min_step_size: f64,
    pub homotopy_step_size_increase_success_count: usize,
    pub homotopy_max_completion_share: f64,
    pub homotopy_linearization_fallback: bool,
    pub homotopy_exclude_varexo: Vec<String>,
    pub endval_steady: bool,
    pub endval_steady_nocheck: bool,
    pub linear_approximation: bool,
    pub endogenous_terminal_period: bool,
    /// `options_.simul.steady_*` (terminal steady state recomputation; defaults are the
    /// `solve_algo`/`steady.maxit`/`solve_tolf`/`solve_tolx` defaults, not later `steady` options).
    pub steady: SteadyOptions,
    /// `options_.steadystate.nocheck`.
    pub steadystate_nocheck: bool,
    /// iterative and robust linear solvers, lmmcp, check_jacobian_singularity, ...
    pub q1: pf_q1::PfQ1Options,
    // ---- shocks/shock_paths ----
    /// `options_.simul.first_simulation_period` / `last_simulation_period` (dates).
    pub first_simulation_period: Option<crate::dates::Date>,
    pub last_simulation_period: Option<crate::dates::Date>,
}

impl Default for SimulOptions {
    fn default() -> Self {
        SimulOptions {
            periods_option: 0,
            periods: 0,
            maxit: 50,
            tolf: 1e-5,
            tolx: 1e-5,
            stack_solve_algo: 0,
            noprint: false,
            verbosity: 1,
            no_homotopy: false,
            homotopy_initial_step_size: 1.0,
            homotopy_min_step_size: 1e-3,
            homotopy_step_size_increase_success_count: 3,
            homotopy_max_completion_share: 1.0,
            homotopy_linearization_fallback: false,
            homotopy_exclude_varexo: Vec::new(),
            endval_steady: false,
            endval_steady_nocheck: false,
            linear_approximation: false,
            endogenous_terminal_period: false,
            steady: SteadyOptions::default(),
            steadystate_nocheck: false,
            q1: Default::default(),
            first_simulation_period: None,
            last_simulation_period: None,
        }
    }
}

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

fn flag(o: &Options, k: &str) -> Option<bool> {
    match opt(o, k) {
        Some(OptValue::Flag) => Some(true),
        Some(OptValue::Num(v)) => Some(*v != 0.0),
        Some(OptValue::Str(s)) => Some(s != "false" && s != "0"),
        _ => None,
    }
}

const PF_OPTIONS: &[&str] = &[
    "periods",
    "maxit",
    "tolf",
    "tolx",
    "stack_solve_algo",
    "noprint",
    "print",
    "verbosity",
    "no_homotopy",
    "homotopy_initial_step_size",
    "homotopy_min_step_size",
    "homotopy_step_size_increase_success_count",
    "homotopy_max_completion_share",
    "homotopy_linearization_fallback",
    "homotopy_exclude_varexo",
    "endval_steady",
    "endval_steady_nocheck",
    "linear_approximation",
    "endogenous_terminal_period",
    "steady_solve_algo",
    "steady_maxit",
    "steady_tolf",
    "steady_tolx",
    "nograph",
    "graph_format",
    "nodisplay",
    "solve_algo",
    "markowitz",
    "steady_markowitz",
    "first_simulation_period",
    "last_simulation_period",
];

impl SimulOptions {
    /// `options_.periods = p` (the `periods` option, `initval_file`, the expectation-errors solver).
    pub fn set_periods(&mut self, p: usize) {
        self.periods_option = p;
        self.periods = p;
    }

    /// Applies the options of one command.
    pub fn update(&mut self, o: &Options) {
        for (k, _) in o {
            if !PF_OPTIONS.contains(&k.as_str()) && !pf_q1::PF_Q1_OPTIONS.contains(&k.as_str()) {
                println!(
                    "WARNING: option `{k}` is not supported by dynare-rust's perfect foresight solver and is ignored."
                );
            }
        }
        if let Some(v) = num(o, "periods") {
            self.set_periods(v as usize);
        }
        // ---- shocks/shock_paths: dates of the simulation (get_simulation_periods.m)
        for (k, slot) in [
            ("first_simulation_period", &mut self.first_simulation_period),
            ("last_simulation_period", &mut self.last_simulation_period),
        ] {
            match crate::dates::option(o, k) {
                Ok(Some(crate::dates::DateOrIndex::Date(d))) => *slot = Some(d),
                Ok(None) => {}
                _ => println!("WARNING: option `{k}` must be a date; ignored."),
            }
        }
        if let Some(v) = num(o, "maxit") {
            self.maxit = v as usize;
        }
        if let Some(v) = num(o, "tolf") {
            self.tolf = v;
        }
        if let Some(v) = num(o, "tolx") {
            self.tolx = v;
        }
        if let Some(v) = num(o, "stack_solve_algo") {
            // values outside 0..=10 are rejected by perfect_foresight_solver (check_input_arguments)
            self.stack_solve_algo = if v < 0.0 { u8::MAX } else { v.min(255.0) as u8 };
        }
        self.q1.update(o);
        if let Some(b) = flag(o, "noprint") {
            self.noprint = b;
        }
        if flag(o, "print") == Some(true) {
            self.noprint = false;
        }
        if let Some(v) = num(o, "verbosity") {
            self.verbosity = v as u8;
        }
        if let Some(b) = flag(o, "no_homotopy") {
            self.no_homotopy = b;
        }
        if let Some(v) = num(o, "homotopy_initial_step_size") {
            self.homotopy_initial_step_size = v;
        }
        if let Some(v) = num(o, "homotopy_min_step_size") {
            self.homotopy_min_step_size = v;
        }
        if let Some(v) = num(o, "homotopy_step_size_increase_success_count") {
            self.homotopy_step_size_increase_success_count = v as usize;
        }
        if let Some(v) = num(o, "homotopy_max_completion_share") {
            self.homotopy_max_completion_share = v;
        }
        if let Some(b) = flag(o, "homotopy_linearization_fallback") {
            self.homotopy_linearization_fallback = b;
        }
        match opt(o, "homotopy_exclude_varexo") {
            Some(OptValue::List(l)) => {
                self.homotopy_exclude_varexo =
                    l.iter().filter_map(|v| if let OptValue::Str(s) = v { Some(s.clone()) } else { None }).collect()
            }
            Some(OptValue::Str(s)) => self.homotopy_exclude_varexo = vec![s.clone()],
            _ => {}
        }
        if let Some(b) = flag(o, "endval_steady") {
            self.endval_steady = b;
        }
        if let Some(b) = flag(o, "endval_steady_nocheck") {
            self.endval_steady_nocheck = b;
        }
        if let Some(b) = flag(o, "linear_approximation") {
            self.linear_approximation = b;
        }
        if let Some(b) = flag(o, "endogenous_terminal_period") {
            self.endogenous_terminal_period = b;
        }
        if let Some(v) = num(o, "steady_solve_algo") {
            self.steady.solve_algo = v as u8;
        }
        if let Some(v) = num(o, "steady_maxit") {
            self.steady.maxit = v as usize;
        }
        if let Some(v) = num(o, "steady_tolf") {
            self.steady.tolf = v;
        }
        if let Some(v) = num(o, "steady_tolx") {
            self.steady.tolx = v;
        }
    }
}

/// Outcome of `perfect_foresight_solver` (`oo_.deterministic_simulation`).
#[derive(Clone, Debug, Default)]
pub struct SimulationStatus {
    pub status: bool,
    pub error: f64,
    pub homotopy_completion_share: f64,
    pub homotopy_iterations: usize,
    pub iterations: usize,
}

/// A raw simulation of a linearization fallback, `oo_.deterministic_simulation.sim1` (the
/// partial simulation) or `.sim2` (the extra simulation of the marginal linearization).
#[derive(Clone, Debug)]
pub struct RawSim {
    pub endo_simul: Mat,
    pub exo_simul: Mat,
    pub homotopy_completion_share: f64,
}

/// `marginal_linearization_previous_raw_sims` of `perfect_foresight_solver.m`.
#[derive(Clone, Debug)]
pub struct MarginalRawSims {
    pub sim1: RawSim,
    pub sim2: RawSim,
}

/// The parts of Dynare's `oo_` (and `M_.endo_histval`) that deterministic commands use.
pub struct DetState {
    pub params: Vec<f64>,
    pub steady_state: Vec<f64>,
    pub exo_steady_state: Vec<f64>,
    pub initial_steady_state: Option<Vec<f64>>,
    pub initial_exo_steady_state: Option<Vec<f64>>,
    /// `oo_.endo_simul`: n_endo x (maximum_lag + periods + maximum_lead).
    pub endo_simul: Option<Mat>,
    /// `oo_.exo_simul`: (maximum_lag + periods + maximum_lead) x n_exo.
    pub exo_simul: Option<Mat>,
    /// Global `options_.solve_algo` / `steady.maxit` / `solve_tolf` / `solve_tolx` (set by `steady`).
    pub steady_opts: SteadyOptions,
    /// `options_.fsolve_options` (lower-cased name, value) pairs of the meaningful MATLAB `fsolve` options.
    pub fsolve_overrides: Vec<(String, f64)>,
    /// `options_.steady.non_zero`.
    pub steady_non_zero: bool,
    /// Equation names for the failure printout of `steady` (set by the driver).
    pub eq_names: Vec<Option<String>>,
    pub simul: SimulOptions,
    pub ssm: Option<SteadyStateModel>,
    pub last: SimulationStatus,
    /// `oo_.deterministic_simulation.sim1` / `.sim2` (linearization fallbacks).
    pub sim1: Option<RawSim>,
    pub sim2: Option<RawSim>,
    pub maximum_lag: usize,
    pub maximum_lead: usize,
    /// Stream EP: `oo_.deterministic_simulation.controlled_paths_by_period`
    /// (`perfect_foresight_controlled_paths`), set by `setup`.
    pub controlled: Option<Vec<crate::ep::controlled::ControlledPeriod>>,
    /// Stream EP: `oo_.pfwee` (perfect foresight with expectation errors).
    pub pfwee: Option<crate::ep::pfwee::PfweeInfo>,
    /// Stream EP: `oo_.initval_series` from `initval_file` (n_endo x T endogenous incl.
    /// auxiliary variables, T x n_exo exogenous).
    pub initval_series: Option<(Mat, Mat)>,
    /// Stream EP: `M_.endo_histval` (n_endo x maximum_lag) / `M_.exo_histval` (n_exo x
    /// maximum_lag) from `histval_file`.
    pub histval_file: Option<(Mat, Mat)>,
    /// controlled paths of `shock_paths` blocks (exogenize / endogenize stanzas).
    pub q1_controlled: Vec<crate::ep::controlled::ControlledPath>,
    /// the file has `shock_paths` blocks.
    pub q1_shock_paths: bool,
}

/// Does the file need the deterministic machinery for `steady` (endval, exogenous initval,
/// steady_state_model or deterministic commands)?
pub fn uses_det_state(mf: &ModFile) -> bool {
    !mf.steady_state_model.is_empty()
        || mf.initval_exo.iter().any(|&(_, v)| v != 0.0)
        || mf.commands.iter().any(|c| matches!(c, Command::Endval(_) | Command::Deterministic(..)))
        // ---- stream EP: extended_path reads oo_.steady_state / M_.endo_histval from DetState
        || mf.commands.iter().any(|c| matches!(c, Command::Other(n, _) if crate::ep::handles(n)))
        // ---- complementarity conditions (steady with solve_algo = 10)
        || !mf.model.mcp.is_empty()
        || !mf.q1.shock_paths.is_empty()
}

/// MATLAB `%g` (6 significant digits).
pub fn fmt_g(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Inf".into() } else { "-Inf".into() };
    }
    if v == 0.0 {
        return "0".into();
    }
    let s = format!("{:.5e}", v);
    let (mant, exp) = s.split_once('e').unwrap();
    let e: i32 = exp.parse().unwrap();
    if !(-4..6).contains(&e) {
        let mant = trim_zeros(mant);
        format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    } else {
        let dec = (5 - e).max(0) as usize;
        trim_zeros(&format!("{:.*}", dec, v)).to_string()
    }
}

/// C `%e` (6 decimals, two-digit exponent).
pub fn fmt_e(v: f64) -> String {
    if !v.is_finite() {
        return fmt_g(v);
    }
    let s = format!("{:.6e}", v);
    let (mant, exp) = s.split_once('e').unwrap();
    let e: i32 = exp.parse().unwrap();
    format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
}

fn trim_zeros(s: &str) -> &str {
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.') } else { s }
}

fn inf_norm(v: &[f64]) -> f64 {
    // MATLAB norm(v, Inf): NaN if any NaN
    let mut m = 0.0f64;
    for &x in v {
        if x.is_nan() {
            return f64::NAN;
        }
        m = m.max(x.abs());
    }
    m
}

impl DetState {
    /// State after the global `initval` block (`oo_.steady_state` / `oo_.exo_steady_state`
    /// with the aux variables set, as the generated driver does).
    pub fn new(mf: &ModFile) -> Result<DetState> {
        let m = &mf.model;
        let n = m.n_endo();
        let mut ys = vec![0.0; n];
        for &(i, v) in &mf.initval {
            ys[i] = v;
        }
        let mut xs = vec![0.0; m.n_exo()];
        for &(k, v) in &mf.initval_exo {
            xs[k] = v;
        }
        m.set_auxiliary_variables(&mut ys, &xs, &m.params);
        let lli = &m.lead_lag_incidence;
        let maximum_lag = lli.iter().any(|r| r[0] > 0) as usize;
        let maximum_lead = lli.iter().any(|r| r[2] > 0) as usize;
        Ok(DetState {
            params: m.params.clone(),
            steady_state: ys,
            exo_steady_state: xs,
            initial_steady_state: None,
            initial_exo_steady_state: None,
            endo_simul: None,
            exo_simul: None,
            steady_opts: SteadyOptions { linear: m.linear, randomize_initial_guess: true, ..Default::default() },
            fsolve_overrides: Vec::new(),
            steady_non_zero: false,
            eq_names: Vec::new(),
            simul: SimulOptions::default(),
            ssm: SteadyStateModel::from_modfile(mf)?,
            last: SimulationStatus::default(),
            sim1: None,
            sim2: None,
            maximum_lag,
            maximum_lead,
            pfwee: None,
            controlled: None,
            initval_series: None,
            histval_file: None,
            q1_controlled: Vec::new(),
            q1_shock_paths: !mf.q1.shock_paths.is_empty(),
        })
    }

    /// The `steady` command (`steady.m`): solves from `oo_.steady_state` and prints it.
    pub fn steady(&mut self, model: &Model, o: &Options) -> Result<()> {
        if let Some(v) = num(o, "solve_algo") {
            self.steady_opts.solve_algo = v as u8;
        }
        if let Some(v) = num(o, "maxit") {
            self.steady_opts.maxit = v as usize;
        }
        if let Some(v) = num(o, "tolf") {
            self.steady_opts.tolf = v;
        }
        if let Some(v) = num(o, "tolx") {
            self.steady_opts.tolx = v;
        }
        // fsolve_options (options_.fsolve_options, persistent): applied last in dynare_solve.m, so
        // they override tolf/tolx/maxit of the MATLAB `fsolve` that solve_algo=0 uses. Here
        // solve_algo=0 is the trust-region solver of solve_algo=9: TolFun/FunctionTolerance,
        // TolX/StepTolerance and MaxIter/MaxIterations map onto it; Display (only 'off'/'final'/'iter'
        // output of fsolve) and MaxFunEvals/MaxFunctionEvaluations (50000, never binding here) have
        // no counterpart and are accepted.
        if let Some(OptValue::List(l)) = opt(o, "fsolve_options") {
            self.fsolve_overrides.clear();
            for kv in l.chunks(2) {
                if let [OptValue::Str(k), OptValue::Num(v)] = kv {
                    match k.to_ascii_lowercase().as_str() {
                        "tolfun" | "functiontolerance" | "tolx" | "steptolerance" | "maxiter" | "maxiterations" => {
                            self.fsolve_overrides.push((k.to_ascii_lowercase(), *v))
                        }
                        _ => {}
                    }
                }
            }
        }
        if flag(o, "non_zero").unwrap_or(false) {
            self.steady_non_zero = true; // options_.steady.non_zero persists
        }
        let mut sopts = self.steady_opts;
        if sopts.solve_algo == 0 {
            for (k, v) in &self.fsolve_overrides {
                match k.as_str() {
                    "tolfun" | "functiontolerance" => sopts.tolf = *v,
                    "tolx" | "steptolerance" => sopts.tolx = *v,
                    _ => sopts.maxit = *v as usize,
                }
            }
        }
        let nocheck = flag(o, "nocheck").unwrap_or(false);
        if nocheck {
            self.simul.steadystate_nocheck = true;
        }
        let noprint = flag(o, "noprint").unwrap_or(false);
        match evaluate_steady_state(
            model,
            self.ssm.as_ref(),
            &self.steady_state,
            &self.exo_steady_state,
            &self.params,
            &sopts,
            !self.simul.steadystate_nocheck,
        ) {
            Ok((ys, p)) => {
                self.steady_state = ys;
                self.params = p;
                if !noprint {
                    print_steady_state(model, &self.steady_state);
                }
                Ok(())
            }
            Err(e) => {
                if !noprint {
                    // steady.m: display_static_residuals (with options_.steady.non_zero)
                    let mut r =
                        steady::static_residuals(model, &self.steady_state, &self.exo_steady_state, &self.params);
                    let disp =
                        crate::misc::mcp_residuals(model, &self.steady_state, &self.params, sopts.solve_algo, &mut r);
                    print!(
                        "{}",
                        crate::misc::residuals_text_disp(
                            model,
                            &r,
                            &self.eq_names,
                            sopts.tolf,
                            self.steady_non_zero,
                            disp
                        )
                    );
                }
                bail!("steady: {e}")
            }
        }
    }

    /// The `endval` block at its position: `oo_.initial_steady_state` <- `oo_.steady_state` at
    /// the first `endval` block only (`if isempty(oo_.initial_steady_state)` in the generated
    /// driver), then the new values (expressions read the current values, as the generated driver).
    pub fn endval(&mut self, model: &Model, entries: &[(String, String)]) -> Result<()> {
        if self.initial_steady_state.is_none() {
            self.initial_steady_state = Some(self.steady_state.clone());
            self.initial_exo_steady_state = Some(self.exo_steady_state.clone());
        }
        for (name, src) in entries {
            let e = model.compile_expr(src).map_err(|e| anyhow!("endval: {name} = {src}: {e}"))?;
            let v = e.eval(&self.steady_state, &self.exo_steady_state, &self.params);
            if let Some(j) = model.endo_index(name) {
                self.steady_state[j] = v;
            } else if let Some(k) = model.exo_index(name) {
                self.exo_steady_state[k] = v;
            } else {
                bail!("endval: '{name}' is not a variable");
            }
        }
        model.set_auxiliary_variables(&mut self.steady_state, &self.exo_steady_state, &self.params);
        Ok(())
    }

    /// Runs one deterministic command.
    pub fn command(
        &mut self,
        mf: &ModFile,
        name: &str,
        o: &Options,
        vars: &[String],
        dir: &Path,
        fname: &str,
    ) -> Result<()> {
        let model = &mf.model;
        match name {
            "perfect_foresight_setup" => {
                self.update_options(o);
                self.setup(mf)
            }
            "perfect_foresight_solver" => {
                self.update_options(o);
                self.solve(model)?;
                self.write_results(model, dir, fname)
            }
            "simul" => {
                self.update_options(o);
                self.setup(mf)?;
                self.solve(model)?;
                self.write_results(model, dir, fname)
            }
            "rplot" => self.rplot(model, o, vars, dir, fname),
            // ---- the position of a shock_paths block: its exogenize stanzas become
            // M_.perfect_foresight_controlled_paths entries, evaluated here
            "shock_paths" => {
                let k = match o.iter().find(|(k, _)| k == "__index") {
                    Some((_, OptValue::Num(v))) => *v as usize,
                    _ => 0,
                };
                let b = &mf.q1.shock_paths[k];
                if b.overwrite {
                    // ---- shocks/shock_paths: a date never matches an integer learnt_in
                    self.q1_controlled.retain(|c| match (c.learnt_in_date, b.learnt_in_date) {
                        (Some(a), Some(d)) => a != d,
                        (None, None) => c.learnt_in != b.learnt_in,
                        _ => true,
                    });
                }
                let cps =
                    shock_paths::controlled_paths(mf, b, &self.params, &self.steady_state, &self.exo_steady_state)?;
                self.q1_controlled.extend(cps);
                Ok(())
            }
            "model_diagnostics" => {
                // full port of model_diagnostics.m (options_.order = 2 by default)
                let order = match o.iter().find(|(k, _)| k == "order") {
                    Some((_, OptValue::Num(v))) => *v as usize,
                    _ => 2,
                };
                diagnostics::model_diagnostics(
                    model,
                    self.ssm.as_ref(),
                    &self.steady_state,
                    &self.exo_steady_state,
                    &self.params,
                    &self.steady_opts,
                    self.simul.steadystate_nocheck,
                    order,
                );
                Ok(())
            }
            _ => bail!("unknown deterministic command {name}"),
        }
    }

    /// Applies the options of a perfect-foresight command. `solve_algo` is the global
    /// `options_.solve_algo` that `steady` also sets (the preprocessor writes
    /// `options_.solve_algo = ...` for both commands).
    pub fn update_options(&mut self, o: &Options) {
        self.simul.update(o);
        if let Some(v) = num(o, "solve_algo") {
            self.steady_opts.solve_algo = v as u8;
        }
    }

    // ---- shocks/shock_paths ----
    /// `get_simulation_periods(options_)`: (periods, first_simulation_period,
    /// last_simulation_period). The number of periods is kept in `simul.periods` for the
    /// solvers; `options_.periods` itself is not changed (a later command with other dates
    /// recomputes the horizon from them).
    pub fn simulation_periods(&mut self) -> Result<(usize, Option<crate::dates::Date>, Option<crate::dates::Date>)> {
        let s = &self.simul;
        if s.periods_option == 0 && (s.first_simulation_period.is_none() || s.last_simulation_period.is_none()) {
            bail!(
                "perfect_foresight_setup: the `periods` option must be set (or both first_simulation_period and last_simulation_period)"
            );
        }
        let r = periods::get_simulation_periods(s.periods_option, s.first_simulation_period, s.last_simulation_period)?;
        self.simul.periods = r.0;
        Ok(r)
    }

    fn ncols(&self) -> usize {
        self.maximum_lag + self.simul.periods + self.maximum_lead
    }

    /// Port of `perfect_foresight_setup.m` (`make_ex_.m`, `make_y_.m`).
    pub fn setup(&mut self, mf: &ModFile) -> Result<()> {
        let m = &mf.model;
        let (n, nx) = (m.n_endo(), m.n_exo());
        if let Some(k) = self.params.iter().position(|p| p.is_nan()) {
            println!(
                "WARNING: Some of the parameters have no value ({}) when using perfect_foresight_setup.",
                m.param_names[k]
            );
        }
        let missing: Vec<&str> =
            (0..n).filter(|&j| m.lead_lag_incidence[j][1] == 0).map(|j| m.endo_names[j].as_str()).collect();
        if !missing.is_empty() {
            bail!(
                "PERFECT_FORESIGHT_SETUP: error in model specification : the variable(s) {} don't appear as current period variables.",
                missing.join(", ")
            );
        }
        // ---- shocks/shock_paths: get_simulation_periods.m (dates of the simulation)
        let (periods, fsp, lsp) = self.simulation_periods()?;
        let mut bad = String::new();
        for s in &mf.det_shocks {
            if !s.dates.is_empty() && lsp.is_none() {
                bail!(
                    "PERFECT_FORESIGHT_SETUP: temporary shocks are specified using dates but neither first_simulation_period nor last_simulation_period option was passed"
                );
            }
            if s.sim_periods(fsp)?.iter().any(|&p| p > periods as i64) {
                bad.push_str(&format!(
                    "\n   At least one expected value for {} has been declared after the terminal period.",
                    m.exo_names[s.exo]
                ));
            }
        }
        if !bad.is_empty() {
            println!("\nPERFECT_FORESIGHT_SETUP: Problem with the declaration of the expected shocks:\n{bad}\n");
            bail!(
                "PERFECT_FORESIGHT_SETUP: Please check the declaration of the shocks or increase the number of periods in the simulation."
            );
        }
        if self.simul.endval_steady && self.maximum_lead == 0 {
            bail!("PERFECT_FORESIGHT_SETUP: Option endval_steady cannot be used on a purely backward or static model.");
        }
        shock_paths::check_setup(mf, lsp)?; // ---- shocks/shock_paths
        let (ml, ncols) = (self.maximum_lag, self.ncols());
        let (endo_hist, exo_hist) = if let Some((e, x)) = &self.histval_file {
            (Some(e.clone()), Some(x.clone())) // ---- stream EP: histval_file
        } else if mf.histval.is_empty() {
            (None, None)
        } else {
            Some(self.histval(mf)).unzip()
        };
        // ---- a permanent shock in shock_paths recreates endval's variables
        if mf.q1.shock_paths.iter().any(|b| b.contains_endval()) {
            self.initial_steady_state = Some(self.steady_state.clone());
            self.initial_exo_steady_state = Some(self.exo_steady_state.clone());
        }
        // make_ex_
        let mut ex = Mat::zeros(ncols, nx);
        for r in 0..ncols {
            for k in 0..nx {
                ex[(r, k)] = match (&exo_hist, &self.initial_exo_steady_state) {
                    (Some(h), _) if r < ml => h[(k, r)],
                    (None, Some(ix)) if r < ml => ix[k],
                    _ => self.exo_steady_state[k],
                };
            }
        }
        if exo_hist.is_some() && self.initial_exo_steady_state.is_some() && self.initval_series.is_none() {
            bail!("histval and endval cannot be used simultaneously");
        }
        // ---- stream EP: initval_file (make_ex_.m with oo_.initval_series)
        let series_off = m.orig_maximum_lag.saturating_sub(ml);
        if let Some((_, sx)) = &self.initval_series {
            if sx.nrows() < series_off + ncols {
                bail!(
                    "perfect_foresight_setup: the initval_file data has {} observations, {} are needed",
                    sx.nrows(),
                    series_off + ncols
                );
            }
            ex = sx.rows(series_off, ncols).clone_owned();
            if let Some(h) = &exo_hist {
                for r in 0..ml {
                    for k in 0..nx {
                        ex[(r, k)] = h[(k, r)];
                    }
                }
            }
        }
        for s in &mf.det_shocks {
            for p in s.sim_periods(fsp)? {
                let k = p + ml as i64; // 1-based row
                if k == 0 {
                    bail!("make_ex_: The model does not have lags, so you cannot set values for period 0");
                }
                if !s.dates.is_empty() && k <= 1 {
                    bail!("make_ex_: some shocks are set at a date before the first simulation period");
                }
                if k < 0 {
                    bail!("make_ex_: a shock is set before the initial period");
                }
                let r = (k - 1) as usize;
                ex[(r, s.exo)] = match s.kind {
                    DetShockKind::Level => s.value,
                    DetShockKind::MultiplySteadyState => self.exo_steady_state[s.exo] * s.value,
                    DetShockKind::MultiplyInitialSteadyState => {
                        let ix = self.initial_exo_steady_state.as_ref().ok_or_else(|| {
                            anyhow!(
                                "Option relative_to_initval of mshocks block cannot be used without an endval block"
                            )
                        })?;
                        ix[s.exo] * s.value
                    }
                };
            }
        }
        // ---- shock_paths (make_ex_.m)
        if !mf.q1.shock_paths.is_empty() {
            let endval = mf.q1.shock_paths.iter().any(|b| b.contains_endval());
            let np = periods + endval as usize;
            let mut paths = Mat::from_fn(nx, np, |k, c| ex[(ml + c, k)]);
            shock_paths::evaluate_paths(
                mf,
                &self.params,
                &self.steady_state,
                &self.exo_steady_state,
                periods,
                fsp,
                &mut paths,
            )?;
            for c in 0..periods {
                for k in 0..nx {
                    ex[(ml + c, k)] = paths[(k, c)];
                }
            }
            if endval {
                self.exo_steady_state = paths.column(periods).iter().copied().collect();
                for r in ml + periods..ncols {
                    for k in 0..nx {
                        ex[(r, k)] = self.exo_steady_state[k];
                    }
                }
            }
        }
        // make_y_
        let mut ey = Mat::zeros(n, ncols);
        for c in 0..ncols {
            for j in 0..n {
                ey[(j, c)] = match (&endo_hist, &self.initial_steady_state) {
                    (Some(h), _) if c < ml => h[(j, c)],
                    (None, Some(is)) if c < ml => is[j],
                    _ => self.steady_state[j],
                };
            }
        }
        // ---- stream EP: initval_file (make_y_.m with oo_.initval_series)
        if let Some((se, _)) = &self.initval_series {
            ey = se.columns(series_off, ncols).clone_owned();
            if let Some(h) = &endo_hist {
                if self.initial_steady_state.is_some() {
                    bail!("histval and endval cannot be used simultaneously");
                }
                for c in 0..ml {
                    for j in 0..n {
                        ey[(j, c)] = h[(j, c)];
                    }
                }
            }
        }
        self.exo_simul = Some(ex);
        self.endo_simul = Some(ey);
        // ---- stream EP: perfect_foresight_controlled_paths
        self.controlled = crate::ep::controlled::by_period_with(mf, &self.q1_controlled, periods, None, fsp)?;
        Ok(())
    }

    /// `M_.endo_histval` / `M_.exo_histval` as the generated histval code builds them: a
    /// zero-initialized series over `orig_maximum_lag_with_diffs_expanded` periods (original
    /// endogenous variables, Lagrange multipliers and exogenous variables), the histval
    /// entries, the auxiliary variables from their definitions
    /// (`<fname>.dynamic_set_auxiliary_series`, NaN where a lag or lead falls outside the
    /// series), NaN -> 0 for the endogenous variables, last `maximum_lag` periods.
    pub fn histval(&self, mf: &ModFile) -> (Mat, Mat) {
        let m = &mf.model;
        let (n, nx) = (m.n_endo(), m.n_exo());
        let ll = m.orig_maximum_lag_with_diffs_expanded.max(1) as i32;
        // series index: d in -(ll-1)..=0 -> d + ll - 1; one column per series date
        let mut endo = Mat::zeros(n, ll as usize);
        let mut exo = Mat::zeros(nx, ll as usize);
        let computed: Vec<usize> = m.aux_vars.iter().filter(|a| a.ty != 6).map(|a| a.endo_index).collect();
        for &j in &computed {
            endo.row_mut(j).fill(f64::NAN);
        }
        let mut direct = Vec::new(); // auxiliary rows set directly (smoother2histval)
        for &(is_endo, idx, lag, v) in &mf.histval {
            let d = lag + ll - 1;
            if is_endo && computed.contains(&idx) {
                direct.push((idx, lag, v));
            } else if d >= 0 {
                if is_endo {
                    endo[(idx, d as usize)] = v;
                } else {
                    exo[(idx, d as usize)] = v;
                }
            }
        }
        // dynamic_set_auxiliary_series: each auxiliary equation `AUX = f(...)` evaluated date by
        // date (repeated sweeps let an auxiliary variable use another one at the same date)
        let params = m.params_with_steady_state(&self.params, &self.steady_state);
        let mut y3 = vec![0.0; 3 * n];
        let mut res = vec![0.0; m.n_equations()];
        for s in 0..ll as usize {
            let x: Vec<f64> = exo.column(s).iter().copied().collect();
            for _ in 0..=computed.len() {
                let mut changed = false;
                // the equation of aux variable k follows the written equations, in order
                for (k, a) in m.aux_vars.iter().enumerate().filter(|(_, a)| a.ty != 6) {
                    let row = m.orig_eq_nbr + k;
                    if row >= res.len() {
                        continue;
                    }
                    for i in 0..n {
                        y3[i] = if s > 0 { endo[(i, s - 1)] } else { f64::NAN };
                        y3[n + i] = endo[(i, s)];
                        y3[2 * n + i] = if s + 1 < ll as usize { endo[(i, s + 1)] } else { f64::NAN };
                    }
                    // `AUX - f(...)`: the residual at AUX = 0 is -f
                    y3[n + a.endo_index] = 0.0;
                    m.dynamic_residuals(&y3, &x, &params, &mut res);
                    let v = -res[row];
                    if v.to_bits() != endo[(a.endo_index, s)].to_bits() {
                        endo[(a.endo_index, s)] = v;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        let ml = self.maximum_lag;
        let mut eh = Mat::zeros(n, ml);
        let mut xh = Mat::zeros(nx, ml);
        for c in 0..ml {
            let i = c as i32 - ml as i32 + ll; // date 1-ml .. 0
            if i < 0 {
                continue;
            }
            for j in 0..n {
                let v = endo[(j, i as usize)];
                eh[(j, c)] = if v.is_nan() { 0.0 } else { v };
            }
            for k in 0..nx {
                xh[(k, c)] = exo[(k, i as usize)];
            }
        }
        for (idx, lag, v) in direct {
            let c = ml as i32 - 1 + lag;
            if idx < n && c >= 0 && (c as usize) < ml {
                eh[(idx, c as usize)] = v;
            }
        }
        (eh, xh)
    }
    /// Port of `perfect_foresight_solver.m` (homotopy loop, linearization fallback).
    pub fn solve(&mut self, model: &Model) -> Result<()> {
        self.solve_with(model, None)
    }

    /// `perfect_foresight_solver(M_, options_, oo_, marginal_linearization_previous_raw_sims)`:
    /// with `prev`, the homotopy starts from the raw simulations of the previous informational
    /// period of `perfect_foresight_with_expectation_errors_solver` (`shareorig` < 1).
    pub fn solve_with(&mut self, model: &Model, prev: Option<&MarginalRawSims>) -> Result<()> {
        self.check_input_arguments(model)?;
        let o = self.simul.clone();
        let (endo0, exo0) = (self.endo_simul.clone().unwrap(), self.exo_simul.clone().unwrap());
        // Various sanity checks
        let mlf = o.q1.homotopy_marginal_linearization_fallback;
        if o.no_homotopy
            && (o.homotopy_initial_step_size != 1.0
                || o.homotopy_max_completion_share != 1.0
                || o.homotopy_linearization_fallback
                || mlf != 0.0)
        {
            bail!(
                "perfect_foresight_solver: the no_homotopy option is incompatible with homotopy_initial_step_size, homotopy_max_completion_share, homotopy_linearization_fallback and homotopy_marginal_linearization_fallback options"
            );
        }
        if o.homotopy_initial_step_size > 1.0 || o.homotopy_initial_step_size < 0.0 {
            bail!(
                "perfect_foresight_solver: The value given to homotopy_initial_step_size option must be in the [0,1] interval"
            );
        }
        if o.homotopy_min_step_size > 1.0 || o.homotopy_min_step_size < 0.0 {
            bail!(
                "perfect_foresight_solver: The value given to homotopy_min_step_size option must be in the [0,1] interval"
            );
        }
        if o.homotopy_max_completion_share > 1.0 || o.homotopy_max_completion_share < 0.0 {
            bail!(
                "perfect_foresight_solver: The value given to homotopy_max_completion_share option must be in the [0,1] interval"
            );
        }
        if !(0.0..=1.0).contains(&mlf) {
            bail!(
                "perfect_foresight_solver: The value given to homotopy_marginal_linearization_fallback option must be in the [0,1] interval"
            );
        }
        if o.homotopy_initial_step_size < o.homotopy_min_step_size {
            bail!(
                "perfect_foresight_solver: The value given to homotopy_initial_step_size option must be greater or equal to that given to homotopy_min_step_size option"
            );
        }
        if o.homotopy_linearization_fallback && mlf > 0.0 {
            bail!(
                "perfect_foresight_solver: Options homotopy_linearization_fallback and homotopy_marginal_linearization_fallback cannot be used together"
            );
        }
        if o.homotopy_max_completion_share < 1.0 && !o.homotopy_linearization_fallback && mlf == 0.0 {
            bail!(
                "perfect_foresight_solver: Option homotopy_max_completion_share has a value less than 1, so you must also specify either homotopy_linearization_fallback or homotopy_marginal_linearization_fallback"
            );
        }
        if let Some(cp) = &self.controlled
            && !o.homotopy_exclude_varexo.is_empty()
        {
            for c in cp.iter().filter(|c| !c.exogenize.is_empty()) {
                // [is_excluded, excluded_exo_ids] = ismember(homotopy_exclude_varexo, M_.exo_names(exo_ids))
                let loc = |v: &String| c.endogenize.iter().position(|&k| &model.exo_names[k] == v);
                if o.homotopy_exclude_varexo.iter().any(|v| loc(v).is_some()) {
                    // M_.exo_names{excluded_exo_ids(1)}: the position in exo_ids of the first
                    // excluded variable, used as an index into M_.exo_names
                    let Some(q) = loc(&o.homotopy_exclude_varexo[0]) else {
                        bail!(
                            "Index in position 1 is invalid. Array indices must be positive integers or logical values."
                        );
                    };
                    bail!(
                        "Exogenous {} cannot be in the homotopy_exclude_varexo option and a perfect_foresight_controlled_paths or shock_paths+exogenize block at the same time",
                        model.exo_names[q]
                    );
                }
            }
        }
        let (ml, ncols) = (self.maximum_lag, self.ncols());
        let periods = o.periods;
        let n = model.n_endo();
        let base_ss = self.initial_steady_state.clone().unwrap_or_else(|| self.steady_state.clone());
        let base_xs = self.initial_exo_steady_state.clone().unwrap_or_else(|| self.exo_steady_state.clone());
        let endobase = Mat::from_fn(n, ncols, |j, _| base_ss[j]);
        let exobase = Mat::from_fn(ncols, model.n_exo(), |_, k| base_xs[k]);
        let lastperiods: Vec<usize> = (ml + periods..ncols).collect();
        let recompute = if o.endval_steady || self.q1_shock_paths {
            true // `|| ~isempty(M_.shock_paths)`
        } else if self.initial_steady_state.is_some() {
            lastperiods.iter().all(|&j| {
                let y: Vec<f64> = endo0.column(j).iter().copied().collect();
                let x: Vec<f64> = exo0.row(j).iter().copied().collect();
                inf_norm(&steady::static_residuals(model, &y, &x, &self.params)) <= o.steady.tolf
            })
        } else {
            false
        };
        // Perform the homotopy loop
        let (shareorig, endoorig, exoorig) = self.orig_scenario(prev.map(|p| &p.sim1), &endo0, &exo0, &exobase);
        let h = self.homotopy_loop(
            model,
            &o,
            shareorig,
            &endoorig,
            &exoorig,
            &endobase,
            &exobase,
            recompute,
            &self.steady_state,
            &self.exo_steady_state,
        )?;
        self.last = SimulationStatus {
            status: false,
            error: h.maxerror,
            homotopy_completion_share: h.completed_share,
            homotopy_iterations: h.iteration,
            iterations: h.solver_iter,
        };
        if h.completed_share == 1.0 {
            if self.controlled.is_some() {
                self.exo_simul = Some(h.exo.clone()); // ---- stream EP: endogenized exogenous values
            }
            self.endo_simul = Some(h.endo);
            if recompute {
                self.steady_state = h.steady_state;
            }
            if !o.noprint {
                print!("Perfect foresight solution found.\n\n");
            }
            self.last.status = true;
        } else if o.homotopy_linearization_fallback && h.completed_share > 0.0 {
            self.sim1 = Some(RawSim {
                endo_simul: h.endo.clone(),
                exo_simul: h.exo.clone(),
                homotopy_completion_share: h.completed_share,
            });
            let s = h.completed_share;
            let e = Mat::from_fn(n, ncols, |j, c| endobase[(j, c)] + (h.endo[(j, c)] - endobase[(j, c)]) / s);
            if recompute {
                self.steady_state = e.column(ncols - 1).iter().copied().collect();
            }
            // ---- stream EP: with controlled paths the exogenous path is extrapolated too
            let x = if self.controlled.is_some() {
                let x = Mat::from_fn(exo0.nrows(), exo0.ncols(), |r, k| {
                    exobase[(r, k)] + (h.exo[(r, k)] - exobase[(r, k)]) / s
                });
                self.exo_simul = Some(x.clone());
                x
            } else {
                exo0.clone()
            };
            let pe = model.params_with_steady_state(&self.params, &self.steady_state); // stream PP
            self.last.error = compute_maxerror(model, &e, &x, &pe, ml, periods);
            self.endo_simul = Some(e);
            if !o.noprint {
                print!(
                    "Perfect foresight solution found for {:.1}% of the shock, then extrapolation was performed using linearization\n\n",
                    s * 100.0
                );
            }
            self.last.status = true;
        } else if mlf > 0.0 && h.completed_share > mlf {
            // ---- homotopy_marginal_linearization_fallback (perfect_foresight_solver.m)
            let completed = h.completed_share;
            self.sim1 = Some(RawSim {
                endo_simul: h.endo.clone(),
                exo_simul: h.exo.clone(),
                homotopy_completion_share: completed,
            });
            // Now compute extra simulation. First try using the first simulation as guess value.
            let (shareorig, endoorig, exoorig) = self.orig_scenario(prev.map(|p| &p.sim2), &endo0, &exo0, &exobase);
            let extra_share = completed - mlf;
            if !o.noprint {
                print!(
                    "Only {:.1}% of the shock could be simulated. Since marginal linearization was requested as a fallback,\nnow running an extra simulation for {:.1}% of the shock.\n\n",
                    completed * 100.0,
                    extra_share * 100.0
                );
                print!("{}\n\n", "*".repeat(80));
            }
            let t0 = Instant::now();
            let excluded: Vec<usize> = o.homotopy_exclude_varexo.iter().filter_map(|s| model.exo_index(s)).collect();
            let sc = self.create_scenario(
                model,
                &o,
                extra_share,
                shareorig,
                &endoorig,
                &exoorig,
                &endobase,
                &exobase,
                recompute,
                &h.endo,
                &h.exo,
                &h.steady_state,
                &h.exo_steady_state,
                &excluded,
            );
            let (mut ok, mut extra) = (false, None);
            if sc.steady_success {
                let (mut e, mut x) = (sc.endo, sc.exo);
                let r = self.core(
                    model,
                    &o,
                    &mut e,
                    &mut x,
                    &sc.steady_state,
                    &sc.exo_steady_state,
                    sc.cp_scaled.as_deref(),
                )?;
                ok = r.0;
                extra = Some((e, x));
            }
            if !ok {
                if !o.noprint {
                    print!(
                        "The extra simulation for {:.1}% of the shock did not run when using the first simulation as a guess value.\nNow trying a full homotopy loop to get that extra simulation working.\n\n",
                        extra_share * 100.0
                    );
                    print!("{}\n\n", "*".repeat(80));
                }
                let mut o2 = o.clone();
                o2.homotopy_max_completion_share = extra_share;
                let h2 = self.homotopy_loop(
                    model,
                    &o2,
                    shareorig,
                    &endoorig,
                    &exoorig,
                    &endobase,
                    &exobase,
                    recompute,
                    &self.steady_state,
                    &self.exo_steady_state,
                )?;
                ok = h2.completed_share == extra_share;
                extra = Some((h2.endo, h2.exo));
            }
            let el = t0.elapsed().as_secs_f64();
            if ok {
                let (ee, ex) = extra.unwrap();
                self.sim2 = Some(RawSim {
                    endo_simul: ee.clone(),
                    exo_simul: ex.clone(),
                    homotopy_completion_share: extra_share,
                });
                let f = (1.0 - completed) / mlf;
                let e = Mat::from_fn(n, ncols, |j, c| h.endo[(j, c)] + (h.endo[(j, c)] - ee[(j, c)]) * f);
                if recompute {
                    self.steady_state = e.column(ncols - 1).iter().copied().collect();
                }
                let x = if self.controlled.is_some() {
                    let x = Mat::from_fn(exo0.nrows(), exo0.ncols(), |r, k| {
                        h.exo[(r, k)] + (h.exo[(r, k)] - ex[(r, k)]) * f
                    });
                    self.exo_simul = Some(x.clone());
                    x
                } else {
                    exo0.clone()
                };
                let pe = model.params_with_steady_state(&self.params, &self.steady_state);
                self.last.error = compute_maxerror(model, &e, &x, &pe, ml, periods);
                self.endo_simul = Some(e);
                if !o.noprint {
                    print!(
                        "Perfect foresight solution found for {:.1}% of the shock, then extrapolation\nwas performed using marginal linearization (extra simulation took {:.1} seconds).\n\n",
                        completed * 100.0,
                        el
                    );
                }
            } else {
                self.endo_simul = Some(h.endo);
                self.exo_simul = Some(h.exo);
                self.steady_state = h.steady_state;
                self.exo_steady_state = h.exo_steady_state;
                print!(
                    "perfect_foresight_solver: marginal linearization failed, unable to\nfind solution for {:.1}% of the shock (extra simulation took {:.1} seconds).\nTry to modify the value of homotopy_marginal_linearization_fallback option.\n\n",
                    extra_share * 100.0,
                    el
                );
            }
            self.last.status = ok;
        } else {
            self.endo_simul = Some(h.endo);
            self.exo_simul = Some(h.exo);
            self.steady_state = h.steady_state;
            self.exo_steady_state = h.exo_steady_state;
            print!("Failed to solve perfect foresight model\n\n");
        }
        Ok(())
    }

    /// `check_input_arguments` of `perfect_foresight_solver.m` (also sets `simul.periods` from
    /// `get_simulation_periods`).
    fn check_input_arguments(&mut self, model: &Model) -> Result<()> {
        let (periods, _, _) = self.simulation_periods()?;
        let ssa = self.simul.stack_solve_algo;
        let block = model.block_solve.as_ref().is_some_and(|b| b.dynamic.is_some());
        if ssa > 10 {
            bail!("PERFECT_FORESIGHT_SOLVER: stack_solve_algo must be between 0 and 10");
        }
        if !block && !model.bytecode && matches!(ssa, 4 | 5) {
            bail!(
                "PERFECT_FORESIGHT_SOLVER: you must use stack_solve_algo={{0,1,2,3,6,7,8,9,10}} when not using block nor bytecode option"
            );
        }
        if block && !model.bytecode && ssa == 5 {
            bail!("PERFECT_FORESIGHT_SOLVER: you can't use stack_solve_algo = 5 without bytecode option");
        }
        let ncols = self.maximum_lag + periods + self.maximum_lead;
        let initval_file = self.initval_series.is_some();
        if !matches!(&self.endo_simul, Some(e) if e.nrows() == model.n_endo() && e.ncols() == ncols) {
            if initval_file {
                print!(
                    "PERFECT_FORESIGHT_SOLVER: 'oo_.endo_simul' has wrong size. Check whether your initval-file provides {} periods.",
                    ncols
                );
            }
            bail!("PERFECT_FORESIGHT_SOLVER: 'oo_.endo_simul' has wrong size. Did you run 'perfect_foresight_setup' ?");
        }
        if model.n_exo() > 0 && !matches!(&self.exo_simul, Some(x) if x.nrows() == ncols && x.ncols() == model.n_exo())
        {
            if initval_file {
                print!(
                    "PERFECT_FORESIGHT_SOLVER: 'oo_.exo_simul' has wrong size. Check whether your initval-file provides {} periods.",
                    ncols
                );
                bail!("PERFECT_FORESIGHT_SOLVER: 'oo_.exo_simul' has wrong size.");
            }
            bail!("PERFECT_FORESIGHT_SOLVER: 'oo_.exo_simul' has wrong size. Did you run 'perfect_foresight_setup' ?");
        }
        if self.exo_simul.is_none() {
            self.exo_simul = Some(Mat::zeros(ncols, 0));
        }
        Ok(())
    }

    /// `shareorig`, `endoorig` and `exoorig` of `perfect_foresight_solver.m`: the full shock
    /// (`oo_.endo_simul`, `oo_.exo_simul`) or, with a raw simulation of
    /// `marginal_linearization_previous_raw_sims`, its completion share and paths (the
    /// endogenized exogenous variables of controlled paths taken from the raw simulation).
    fn orig_scenario(&self, raw: Option<&RawSim>, endo0: &Mat, exo0: &Mat, exobase: &Mat) -> (f64, Mat, Mat) {
        let Some(raw) = raw else {
            return (1.0, endo0.clone(), exo0.clone());
        };
        let s = raw.homotopy_completion_share;
        let mut exoorig =
            Mat::from_fn(exo0.nrows(), exo0.ncols(), |r, k| s * exo0[(r, k)] + (1.0 - s) * exobase[(r, k)]);
        if let Some(cp) = &self.controlled {
            for (p, c) in cp.iter().enumerate().filter(|(_, c)| !c.exogenize.is_empty()) {
                for &k in &c.endogenize {
                    exoorig[(self.maximum_lag + p, k)] = raw.exo_simul[(self.maximum_lag + p, k)];
                }
            }
        }
        (s, raw.endo_simul.clone(), exoorig)
    }

    /// `create_scenario` of `perfect_foresight_solver.m`: exogenous paths and initial
    /// conditions at `share` of the shock (`endoorig`, `exoorig` being `shareorig` of it),
    /// terminal steady state recomputed (or interpolated), controlled paths scaled;
    /// `endo`/`exo` are the guesses, `ss`/`xs` the steady state kept without recomputation.
    #[allow(clippy::too_many_arguments)]
    fn create_scenario(
        &self,
        model: &Model,
        o: &SimulOptions,
        new_share: f64,
        shareorig: f64,
        endoorig: &Mat,
        exoorig: &Mat,
        endobase: &Mat,
        exobase: &Mat,
        recompute: bool,
        endo: &Mat,
        exo: &Mat,
        ss: &[f64],
        xs: &[f64],
        excluded: &[usize],
    ) -> Scenario {
        let ml = self.maximum_lag;
        let periods = o.periods;
        let ncols = self.ncols();
        let r = new_share / shareorig;
        let mut e2 = endo.clone();
        let mut x2 = Mat::from_fn(ncols, model.n_exo(), |t, k| {
            exoorig[(t, k)] * new_share / shareorig + exobase[(t, k)] * (1.0 - r)
        });
        for &k in excluded {
            for t in 0..ncols {
                x2[(t, k)] = exoorig[(t, k)];
            }
        }
        for c in 0..ml {
            for j in 0..model.n_endo() {
                e2[(j, c)] = r * endoorig[(j, c)] + (1.0 - r) * endobase[(j, c)];
            }
        }
        let mut steady_success = true;
        let (mut steady_state, mut exo_steady_state) = (ss.to_vec(), xs.to_vec());
        if recompute {
            let mut so = o.steady;
            so.linear = model.linear;
            let saved_ss: Vec<Vec<f64>> =
                (ml + periods..ncols).map(|j| e2.column(j).iter().copied().collect()).collect();
            for j in ml + periods..ncols {
                let guess: Vec<f64> = e2.column(j).iter().copied().collect();
                let xj: Vec<f64> = x2.row(j).iter().copied().collect();
                let check = !o.endval_steady_nocheck;
                let r = evaluate_steady_state(model, self.ssm.as_ref(), &guess, &xj, &self.params, &so, check).or_else(
                    |_| {
                        let g = self.initial_steady_state.as_ref().unwrap_or(&self.steady_state);
                        evaluate_steady_state(model, self.ssm.as_ref(), g, &xj, &self.params, &so, check)
                    },
                );
                match r {
                    Ok((y, _)) => e2.set_column(j, &crate::linalg::Vector::from_vec(y)),
                    Err(_) => {
                        for (k, j2) in (ml + periods..ncols).enumerate() {
                            e2.set_column(j2, &crate::linalg::Vector::from_vec(saved_ss[k].clone()));
                        }
                        steady_success = false;
                        break;
                    }
                }
            }
            // also when the recomputation failed (from the restored terminal condition)
            steady_state = e2.column(ncols - 1).iter().copied().collect();
            exo_steady_state = x2.row(ncols - 1).iter().copied().collect();
        } else {
            for c in ml + periods..ncols {
                for j in 0..model.n_endo() {
                    e2[(j, c)] = r * endoorig[(j, c)] + (1.0 - r) * endobase[(j, c)];
                }
            }
        }
        // ---- stream EP: controlled paths in create_scenario (values scaled by the share,
        // endogenized exogenous guesses kept from the current iterate)
        let cp_scaled =
            self.controlled.as_ref().map(|cp| crate::ep::controlled::scale(cp, new_share, endobase, exo, &mut x2, ml));
        Scenario { steady_success, endo: e2, exo: x2, steady_state, exo_steady_state, cp_scaled }
    }

    /// `homotopy_loop` of `perfect_foresight_solver.m` (target share
    /// `o.homotopy_max_completion_share`, from `shareorig` of the shock in `endoorig`/`exoorig`).
    #[allow(clippy::too_many_arguments)]
    fn homotopy_loop(
        &self,
        model: &Model,
        o: &SimulOptions,
        shareorig: f64,
        endoorig: &Mat,
        exoorig: &Mat,
        endobase: &Mat,
        exobase: &Mat,
        recompute: bool,
        steady_state: &[f64],
        exo_steady_state: &[f64],
    ) -> Result<Homotopy> {
        let ml = self.maximum_lag;
        let periods = o.periods;
        let max_share = o.homotopy_max_completion_share;
        let excluded: Vec<usize> = o
            .homotopy_exclude_varexo
            .iter()
            .map(|s| {
                model.exo_index(s).ok_or_else(|| {
                    anyhow!("perfect_foresight_solver: the variable list homotopy_exclude_varexo must only contain exogenous variable names.")
                })
            })
            .collect::<Result<_>>()?;
        let mut completed = 0.0;
        let mut step = o.homotopy_initial_step_size.min(max_share);
        let mut success_counter = 0;
        let mut iteration = 0;
        let mut endo = endoorig.clone();
        let mut exo = exoorig.clone();
        let mut ss = steady_state.to_vec();
        let mut xs = exo_steady_state.to_vec();
        let mut maxerror = f64::NAN;
        let mut solver_iter = 0;
        let mut table: Vec<String> = Vec::new();
        while step > o.homotopy_min_step_size {
            iteration += 1;
            let saved = (endo.clone(), exo.clone());
            let mut new_share = completed + step;
            if new_share > max_share {
                new_share = max_share;
                step = new_share - completed;
            }
            let t0 = Instant::now();
            let sc = self.create_scenario(
                model, o, new_share, shareorig, endoorig, exoorig, endobase, exobase, recompute, &endo, &exo, &ss, &xs,
                &excluded,
            );
            let steady_success = sc.steady_success;
            ss = sc.steady_state;
            xs = sc.exo_steady_state;
            let cp_scaled = sc.cp_scaled;
            endo = sc.endo;
            exo = sc.exo;
            let mut success = false;
            if steady_success {
                if completed == 0.0 {
                    if iteration == 1 && new_share == shareorig {
                        // initial guess from perfect_foresight_setup (or the raw simulation)
                    } else if self.maximum_lead > 0 {
                        let last = endo.column(ml + periods).clone_owned();
                        for c in ml..ml + periods {
                            endo.set_column(c, &last);
                        }
                    } else {
                        for c in ml..ml + periods {
                            endo.set_column(c, &endobase.column(c).clone_owned());
                        }
                    }
                    // ---- stream EP: with controlled paths, the exogenous guesses over the
                    // simulation periods restart from the base path (all of them, as Dynare)
                    if self.controlled.is_some() && !(iteration == 1 && new_share == shareorig) {
                        for r in ml..ml + periods {
                            for k in 0..model.n_exo() {
                                exo[(r, k)] = exobase[(r, k)];
                            }
                        }
                    }
                }
                let r = self.core(model, o, &mut endo, &mut exo, &ss, &xs, cp_scaled.as_deref())?;
                success = r.0;
                maxerror = r.1;
                solver_iter = r.2;
            } else {
                maxerror = f64::NAN;
            }
            let elapsed = t0.elapsed().as_secs_f64();
            if o.no_homotopy || (iteration == 1 && success && new_share == 1.0) {
                if success {
                    completed = new_share;
                }
                break;
            }
            if iteration == 1 && !o.noprint {
                println!("\nEntering the homotopy method iterations...");
                table.push("\nIter. \t | Share \t | Status \t | Max. residual\t | Duration (sec)\n".into());
                table
                    .push("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n".into());
            }
            if success {
                if !o.noprint {
                    table.push(format!(
                        "{} \t | {:.5} \t | succeeded \t | {} \t | {:.1}\n",
                        iteration,
                        new_share,
                        fmt_e(maxerror),
                        elapsed
                    ));
                }
                completed = new_share;
                if completed >= max_share {
                    print!("{}", table.concat());
                    break;
                }
                success_counter += 1;
                if o.homotopy_step_size_increase_success_count > 0
                    && success_counter >= o.homotopy_step_size_increase_success_count
                {
                    success_counter = 0;
                    step *= 2.0;
                }
            } else {
                endo = saved.0;
                exo = saved.1;
                success_counter = 0;
                step /= 2.0;
                if !o.noprint {
                    if !steady_success {
                        table.push(format!(
                            "{} \t | {:.5} \t | failed (in endval steady) \t\t | {:.1}\n",
                            iteration, new_share, elapsed
                        ));
                    } else {
                        table.push(format!(
                            "{} \t | {:.5} \t | failed \t | {} \t | {:.1}\n",
                            iteration,
                            new_share,
                            fmt_e(maxerror),
                            elapsed
                        ));
                    }
                }
            }
            print!("{}", table.concat());
        }
        println!();
        Ok(Homotopy {
            completed_share: completed,
            endo,
            exo,
            steady_state: ss,
            exo_steady_state: xs,
            iteration,
            maxerror,
            solver_iter,
        })
    }

    /// Port of `perfect_foresight_solver_core.m`: the option checks, then block-wise solving,
    /// the purely backward/forward/static simulations, `sim1_linear` for linear models /
    /// `linear_approximation`, `sim1` / `sim1_lbj` / `solve_stacked_problem` by
    /// `stack_solve_algo`. Returns (success, maxerror, iterations); errors are Dynare's
    /// `error()` calls, which end the command.
    #[allow(clippy::too_many_arguments)]
    fn core(
        &self,
        model: &Model,
        o: &SimulOptions,
        endo: &mut Mat,
        exo: &mut Mat,
        ss: &[f64],
        xs: &[f64],
        cp: Option<&[crate::ep::controlled::ControlledPeriod]>,
    ) -> Result<(bool, f64, usize)> {
        // options_.lmmcp.status sets stack_solve_algo = 7 and solve_algo = 10
        let (stack, solve_algo) = if o.q1.lmmcp { (7, 10) } else { (o.stack_solve_algo, self.steady_opts.solve_algo) };
        let block = model.block_solve.as_ref().and_then(|b| b.dynamic.as_ref());
        let purely = self.maximum_lead == 0 || self.maximum_lag == 0;
        if o.linear_approximation {
            if !matches!(stack, 0 | 7) {
                bail!(
                    "perfect_foresight_solver: option linear_approximation is only available with stack_solve_algo option equal to 0 or 7"
                );
            }
            if block.is_some() {
                bail!("perfect_foresight_solver: option linear_approximation is not available with the block option");
            }
            if purely {
                bail!(
                    "perfect_foresight_solver: option linear_approximation is not available with purely backward, purely forward or static models"
                );
            }
        }
        if o.endogenous_terminal_period {
            if stack != 0 {
                bail!(
                    "perfect_foresight_solver: option endogenous_terminal_period is only available with stack_solve_algo option equal to 0"
                );
            }
            if block.is_some() {
                bail!(
                    "perfect_foresight_solver: option endogenous_terminal_period is not available with the block option"
                );
            }
            if purely {
                bail!(
                    "perfect_foresight_solver: option endogenous_terminal_period is not available with purely backward, purely forward or static models"
                );
            }
        }
        if cp.is_some() {
            if !matches!(stack, 0..=3 | 6..=10) {
                bail!(
                    "perfect_foresight_controlled_paths or shock_paths+exogenize is only available with stack_solve_algo option equal to 0, 1, 2, 3, 6, 7, 8, 9, 10"
                );
            }
            if stack == 7 && matches!(solve_algo, 10 | 11) {
                bail!(
                    "perfect_foresight_controlled_paths or shock_paths+exogenize is not available for mixed-complementarity problems (LMMCP or PATH solvers)"
                );
            }
            if block.is_some() {
                bail!(
                    "perfect_foresight_controlled_paths or shock_paths+exogenize is not available with the block option"
                );
            }
            if purely {
                bail!(
                    "perfect_foresight_controlled_paths or shock_paths+exogenize is not available with purely backward, purely forward or static models"
                );
            }
        }
        let linear = o.linear_approximation || (model.linear && !purely && matches!(stack, 0 | 2 | 3 | 7..=10));
        // stream PP: STEADY_STATE() in the model reads oo_.steady_state (perfect_foresight_problem.m)
        let params = model.params_with_steady_state(&self.params, ss);
        // ---- model(block): block-wise solving (solve_block_decomposed_problem.m)
        if let Some(bs) = block {
            return Ok(block_solve::solve_block_decomposed(model, bs, o, endo, exo, &params, self.maximum_lag));
        }
        // ---- model(bytecode) without block: bytecode('dynamic', ...), the MEX's Newton iterations
        // on the stacked model with a sparse Gaussian elimination, i.e. those of sim1 (the MEX does
        // not take the linear shortcut)
        if model.bytecode && !matches!(stack, 1 | 6) {
            if matches!(stack, 8..=10) {
                bail!("perfect_foresight_solver: bytecode is not compatible with stack_solve_algo={stack}");
            }
            return Ok(sim1(model, o, endo, exo, &params, self.maximum_lag));
        }
        // ---- stream EP: perfect_foresight_controlled_paths (sim1 / sim1_linear / sim1_lbj /
        // solve_stacked_problem with a controlled_paths_by_period)
        if let Some(cp) = cp {
            let ml = self.maximum_lag;
            if stack == 7 {
                if linear {
                    bail!(
                        "The combination of the linear or linear_approximation option, stack_solve_algo=7 and perfect_foresight_controlled_paths or shock_paths+exogenize is not supported"
                    );
                }
                return Ok(crate::ep::controlled::solve_stacked_problem(
                    model, o, solve_algo, endo, exo, &params, ml, cp,
                ));
            }
            if matches!(stack, 1 | 6) && !linear {
                // sim1_lbj.m with the controlled paths
                return Ok(pf_q1::sim1_lbj_controlled(model, o, endo, exo, &params, ml, Some(cp)));
            }
            if o.endogenous_terminal_period && !linear {
                bail!(
                    "The endogenous_terminal_period option not available with perfect_foresight_controlled_paths or shock_paths+exogenize"
                );
            }
            // the Newton iterations of sim1_lbj (stack_solve_algo 1/6) and of the iterative
            // linear solvers are those of the direct solve, up to the linear solver's precision
            return crate::ep::controlled::sim1(
                model,
                o,
                endo,
                exo,
                ss,
                xs,
                &params,
                ml,
                cp,
                linear,
                self.steady_opts.tolf,
            );
        }
        // ---- stack_solve_algo 1-3 and 6-10, lmmcp, robust_lin_solve, purely
        // backward/forward/static models
        if let Some(r) = pf_q1::core(
            model,
            o,
            solve_algo,
            endo,
            exo,
            &params,
            &self.params,
            self.maximum_lag,
            self.maximum_lead,
            linear,
        ) {
            return Ok(r);
        }
        if linear {
            sim1_linear(model, o, endo, exo, ss, xs, &params, self.maximum_lag, self.steady_opts.tolf)
        } else {
            Ok(sim1(model, o, endo, exo, &params, self.maximum_lag))
        }
    }

    /// Writes `oo_.endo_simul` / `oo_.exo_simul` to `<dir>/<fname>/Output/<fname>_endo_simul.csv`
    /// and `_exo_simul.csv` (one row per period, period 1 = first simulated period).
    pub fn write_results(&self, model: &Model, dir: &Path, fname: &str) -> Result<()> {
        let out = dir.join(fname).join("Output");
        std::fs::create_dir_all(&out)?;
        let ml = self.maximum_lag as i64;
        if let Some(e) = &self.endo_simul {
            let mut s = String::from("period");
            for nme in &model.endo_names {
                s.push(',');
                s.push_str(nme);
            }
            s.push('\n');
            for c in 0..e.ncols() {
                s.push_str(&(c as i64 + 1 - ml).to_string());
                for j in 0..e.nrows() {
                    s.push_str(&format!(",{:.17e}", e[(j, c)]));
                }
                s.push('\n');
            }
            std::fs::write(out.join(format!("{fname}_endo_simul.csv")), s)?;
        }
        if let Some(x) = &self.exo_simul {
            let mut s = String::from("period");
            for nme in &model.exo_names {
                s.push(',');
                s.push_str(nme);
            }
            s.push('\n');
            for r in 0..x.nrows() {
                s.push_str(&(r as i64 + 1 - ml).to_string());
                for k in 0..x.ncols() {
                    s.push_str(&format!(",{:.17e}", x[(r, k)]));
                }
                s.push('\n');
            }
            std::fs::write(out.join(format!("{fname}_exo_simul.csv")), s)?;
        }
        Ok(())
    }

    /// `rplot.m` as `graphics::Figure`s: `options_.rplottype` 0 (all variables in one panel,
    /// default), 1 (one figure per variable) or 2 (subplots with the steady state in red).
    fn rplot(&self, model: &Model, o: &Options, vars: &[String], dir: &Path, fname: &str) -> Result<()> {
        let e = self.endo_simul.as_ref().ok_or_else(|| anyhow!("rplot: oo_.endo_simul is empty."))?;
        let ml = self.maximum_lag as i64;
        let first = (ml - 1).max(0) as usize; // max(1, maximum_lag), 0-based
        let xs: Vec<f64> = (first..e.ncols()).map(|c| (c as i64 + 1 - ml) as f64).collect();
        // (values, steady state) of each variable
        let mut ys: Vec<(Vec<f64>, f64)> = Vec::new();
        for v in vars {
            ys.push(if let Some(j) = model.endo_index(v) {
                ((first..e.ncols()).map(|c| e[(j, c)]).collect(), self.steady_state.get(j).copied().unwrap_or(0.0))
            } else if let (Some(kx), Some(x)) = (model.exo_index(v), &self.exo_simul) {
                (
                    (first..x.nrows()).map(|r| x[(r, kx)]).collect(),
                    self.exo_steady_state.get(kx).copied().unwrap_or(0.0),
                )
            } else {
                bail!("rplot: One of the variables specified does not exist");
            });
        }
        let line = |k: usize, y: &[f64], v: &str| Series::Line {
            x: xs.clone(),
            y: y.to_vec(),
            color: Color::Palette(k),
            style: LineStyle::Solid,
            width: 1.0,
            label: Some(v.to_string()),
        };
        let panel_of = |title: String, series: Vec<Series>, legend: bool| Panel {
            title,
            x_label: Some("Periods".into()),
            legend,
            series,
            ..Default::default()
        };
        let rplottype = match opt(o, "rplottype") {
            Some(OptValue::Num(v)) => *v as i64,
            _ => 0,
        };
        let name_of = |v: &str| format!("SimulatedTrajectory_{v}");
        let first_var = vars.first().map(String::as_str).unwrap_or("");
        let figs: Vec<Figure> = match rplottype {
            1 => vars
                .iter()
                .zip(&ys)
                .map(|(v, (y, _))| Figure {
                    name: name_of(v),
                    title: "Simulated Trajectory".into(),
                    rows: 1,
                    cols: 1,
                    panels: vec![panel_of(format!("Plot of {v}"), vec![line(0, y, v)], false)],
                })
                .collect(),
            2 => {
                let n = vars.len();
                let nl = (n / 4).max(1);
                let nc = n.div_ceil(nl);
                let panels = vars
                    .iter()
                    .zip(&ys)
                    .map(|(v, (y, ss))| {
                        let mut p = panel_of(
                            format!("Plot of {v}"),
                            vec![
                                line(0, y, v),
                                Series::Line {
                                    x: xs.clone(),
                                    y: vec![*ss; xs.len()],
                                    color: Color::Red,
                                    style: LineStyle::Dotted,
                                    width: 1.0,
                                    label: None,
                                },
                            ],
                            false,
                        );
                        p.y_label = Some(v.clone());
                        p
                    })
                    .collect();
                vec![Figure {
                    name: name_of(first_var),
                    title: "Simulated Trajectory".into(),
                    rows: nl,
                    cols: nc,
                    panels,
                }]
            }
            _ => {
                let series = vars.iter().zip(&ys).enumerate().map(|(k, (v, (y, _)))| line(k, y, v)).collect();
                let title = format!("Plot of {}", vars.join(" "));
                vec![Figure {
                    name: name_of(first_var),
                    title: "Simulated Trajectory".into(),
                    rows: 1,
                    cols: 1,
                    panels: vec![panel_of(title, series, vars.len() > 1)],
                }]
            }
        };
        let formats = match opt(o, "graph_format") {
            Some(OptValue::Str(s)) => GraphFormat::parse(s).into_iter().collect(),
            Some(OptValue::List(l)) => {
                l.iter().filter_map(|v| if let OptValue::Str(s) = v { GraphFormat::parse(s) } else { None }).collect()
            }
            _ => vec![GraphFormat::Pdf],
        };
        let gopts = GraphOptions {
            enabled: !flag(o, "nograph").unwrap_or(false),
            formats,
            dir: graphics::graph_dir(dir, fname),
        };
        for fig in &figs {
            graphics::save(fig, fname, &gopts)?;
        }
        Ok(())
    }
}

/// Result of `create_scenario`.
struct Scenario {
    steady_success: bool,
    endo: Mat,
    exo: Mat,
    /// Terminal steady state (endogenous, exogenous): the last column / row when it is
    /// recomputed, the input otherwise.
    steady_state: Vec<f64>,
    exo_steady_state: Vec<f64>,
    cp_scaled: Option<Vec<crate::ep::controlled::ControlledPeriod>>,
}

struct Homotopy {
    completed_share: f64,
    endo: Mat,
    exo: Mat,
    steady_state: Vec<f64>,
    exo_steady_state: Vec<f64>,
    iteration: usize,
    maxerror: f64,
    solver_iter: usize,
}

/// `disp_steady_state.m`.
pub fn print_steady_state(model: &Model, ys: &[f64]) {
    print!("{}", steady_state_text(model, ys));
}

/// The text `disp_steady_state.m` prints: a blank line, the title, a blank line, then
/// `fprintf('%s \t\t %g\n', ...)` per declared variable, with the names padded as
/// `char(M_.endo_names)` pads them (to the longest name, auxiliary variables included).
pub fn steady_state_text(model: &Model, ys: &[f64]) -> String {
    let w = model.endo_names.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut s = String::from("\nSTEADY-STATE RESULTS:\n\n");
    for i in 0..model.orig_endo_nbr {
        s += &format!("{:<w$} \t\t {}\n", model.endo_names[i], fmt_g(ys[i]));
    }
    s
}

fn exo_row(exo: &Mat, r: usize) -> Vec<f64> {
    exo.row(r).iter().copied().collect()
}

fn conditions(endo: &Mat, ml: usize, periods: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = endo.nrows();
    let y0: Vec<f64> = if ml > 0 { endo.column(ml - 1).iter().copied().collect() } else { vec![f64::NAN; n] };
    let yt: Vec<f64> = if endo.ncols() > ml + periods {
        endo.column(ml + periods).iter().copied().collect()
    } else {
        vec![f64::NAN; n]
    };
    let mut y = vec![0.0; n * periods];
    for t in 0..periods {
        for j in 0..n {
            y[t * n + j] = endo[(j, ml + t)];
        }
    }
    (y0, yt, y)
}

/// `compute_maxerror.m`.
pub fn compute_maxerror(model: &Model, endo: &Mat, exo: &Mat, params: &[f64], ml: usize, periods: usize) -> f64 {
    let (y0, yt, y) = conditions(endo, ml, periods);
    let mut p = StackedProblem::new(model);
    let mut res = vec![0.0; y.len()];
    p.eval(&y, &y0, &yt, &|t| exo_row(exo, t + ml), params, &mut res, None);
    inf_norm(&res)
}

/// Port of `sim1.m` (no controlled paths): Newton on the stacked system.
pub fn sim1(
    model: &Model,
    o: &SimulOptions,
    endo: &mut Mat,
    exo: &Mat,
    params: &[f64],
    ml: usize,
) -> (bool, f64, usize) {
    let verbose = o.verbosity > 0 && !o.noprint;
    let n = model.n_endo();
    let mut periods = o.periods;
    let (y0, mut yt, mut y) = conditions(endo, ml, periods);
    if verbose {
        println!();
        println!("{}", "-".repeat(56));
        println!("MODEL SIMULATION:");
        println!();
    }
    let h1 = Instant::now();
    let mut iter = 1;
    let mut converged = false;
    let mut prob = StackedProblem::new(model);
    let mut a = StackedJacobian::new(&prob.s, periods);
    let mut res = vec![0.0; n * periods];
    let mut err = f64::NAN;
    while !(converged || iter > o.maxit) {
        let h2 = Instant::now();
        prob.eval(&y, &y0, &yt, &|t| exo_row(exo, t + ml), params, &mut res, Some(&mut a));
        if o.endogenous_terminal_period && iter > 1 {
            for it in 0..periods {
                if inf_norm(&res[it * n..(it + 1) * n]) < o.tolf / 1e7 {
                    let itm = it + 1;
                    if itm < periods {
                        yt = y[itm * n..(itm + 1) * n].to_vec();
                        for t in itm..periods {
                            for j in 0..n {
                                endo[(j, ml + t)] = y[t * n + j];
                            }
                        }
                        y.truncate(itm * n);
                        periods = itm;
                        res.truncate(itm * n);
                        a = StackedJacobian::new(&prob.s, periods);
                        prob.eval(&y, &y0, &yt, &|t| exo_row(exo, t + ml), params, &mut res, Some(&mut a));
                    }
                    break;
                }
            }
        }
        err = inf_norm(&res);
        if err < o.tolf {
            converged = true;
        } else {
            let dy = lin_solve(&mut a, &prob.s, &res);
            for (v, d) in y.iter_mut().zip(&dy) {
                *v -= d;
            }
        }
        if verbose {
            println!("Iter: {},\t err. = {},\t time = {}", iter, fmt_g(err), fmt_g(h2.elapsed().as_secs_f64()));
        }
        iter += 1;
    }
    for t in 0..periods {
        for j in 0..n {
            endo[(j, ml + t)] = y[t * n + j];
        }
    }
    if o.endogenous_terminal_period {
        err = compute_maxerror(model, endo, exo, params, ml, o.periods);
    }
    let success;
    if converged {
        if y.iter().any(|v| !v.is_finite()) {
            success = false;
            if verbose {
                println!();
                println!("Total time of simulation: {}.", fmt_g(h1.elapsed().as_secs_f64()));
                println!("Simulation terminated with NaN or Inf in the residuals or endogenous variables.");
                println!(
                    "There is most likely something wrong with your model. Try model_diagnostics or another simulation method."
                );
                println!("{}", "-".repeat(105));
            }
        } else {
            if verbose {
                println!();
                println!("Total time of simulation: {}.", fmt_g(h1.elapsed().as_secs_f64()));
                println!("{}", "-".repeat(56));
            }
            success = true;
        }
    } else {
        if verbose {
            println!();
            println!("Total time of simulation: {}.", fmt_g(h1.elapsed().as_secs_f64()));
            println!("Maximum number of iterations is reached (modify option maxit).");
            println!("{}", "-".repeat(62));
        }
        success = false;
    }
    if verbose {
        println!();
    }
    (success, err, iter)
}

/// `lin_solve.m` with `stack_solve_algo = 0`: `A\b`, 0 if `norm(b) < sqrt(eps)`,
/// non-finite entries set to 0.
fn lin_solve(a: &mut StackedJacobian, s: &stacked::StackedStructure, b: &[f64]) -> Vec<f64> {
    let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if nb < f64::EPSILON.sqrt() {
        return vec![0.0; b.len()];
    }
    let mut x = b.to_vec();
    if !a.solve(s, &mut x) {
        println!("Warning: Matrix is singular to working precision.");
        x.iter_mut().for_each(|v| *v = f64::NAN);
    }
    for v in &mut x {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    x
}

/// Port of `sim1_linear.m` (no controlled paths): one solve of the system linearized at the
/// steady state `ss`, `xs`; `solve_tolf` (`options_.solve_tolf`) bounds the residuals at the
/// steady state.
#[allow(clippy::too_many_arguments)]
pub fn sim1_linear(
    model: &Model,
    o: &SimulOptions,
    endo: &mut Mat,
    exo: &Mat,
    ss: &[f64],
    xs: &[f64],
    params: &[f64],
    ml: usize,
    solve_tolf: f64,
) -> Result<(bool, f64, usize)> {
    let verbose = o.verbosity > 0;
    let n = model.n_endo();
    let periods = o.periods;
    if verbose {
        println!();
        println!("{}", "-".repeat(80));
        println!("MODEL SIMULATION:");
        println!();
    }
    let mut y3n = Vec::with_capacity(3 * n);
    for _ in 0..3 {
        y3n.extend_from_slice(ss);
    }
    let mut d1 = vec![0.0; n];
    model.dynamic_residuals(&y3n, xs, params, &mut d1);
    let prob = StackedProblem::new(model);
    let pattern = model.dynamic_jacobian_pattern();
    let mut vals = vec![0.0; pattern.len()];
    let mut ws = model.workspace();
    model.dynamic_jacobian_values_with(&y3n, xs, params, &mut ws, &mut vals);
    if !o.steadystate_nocheck && d1.iter().fold(0.0f64, |m, v| m.max(v.abs())) > solve_tolf {
        bail!("Jacobian is not evaluated at the steady state!");
    }
    // deviations from the steady state
    let mut dev = endo.clone();
    for c in 0..dev.ncols() {
        for j in 0..n {
            dev[(j, c)] -= ss[j];
        }
    }
    let xdev = Mat::from_fn(exo.nrows(), exo.ncols(), |r, k| exo[(r, k)] - xs[k]);
    let (y0, yt, mut yv) = conditions(&dev, ml, periods);
    let h2 = Instant::now();
    let mut a = StackedJacobian::new(&prob.s, periods);
    for t in 0..periods {
        a.set_period(t, &vals);
    }
    let linres = |yv: &[f64]| -> Vec<f64> {
        let mut res = vec![0.0; n * periods];
        let mut z = vec![0.0; 3 * n + model.n_exo()];
        for t in 0..periods {
            let prev = if t == 0 { &y0[..] } else { &yv[(t - 1) * n..t * n] };
            let next = if t + 1 == periods { &yt[..] } else { &yv[(t + 1) * n..(t + 2) * n] };
            z[..n].copy_from_slice(prev);
            z[n..2 * n].copy_from_slice(&yv[t * n..(t + 1) * n]);
            z[2 * n..3 * n].copy_from_slice(next);
            for k in 0..model.n_exo() {
                z[3 * n + k] = xdev[(t + ml, k)];
            }
            let r = &mut res[t * n..(t + 1) * n];
            for (q, &(i, c)) in pattern.iter().enumerate() {
                // 0 * NaN in the unused initial/terminal blocks is skipped as Dynare's sparse product
                if vals[q] != 0.0 {
                    r[i] += vals[q] * z[c];
                }
            }
        }
        res
    };
    let res0 = linres(&yv);
    let err0 = inf_norm(&res0);
    let mut bx = res0.clone();
    let ok = {
        let nb = bx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if nb < f64::EPSILON.sqrt() {
            bx.fill(0.0);
            true
        } else {
            a.solve(&prob.s, &mut bx)
        }
    };
    if !ok {
        if verbose {
            println!();
            println!("Singularity problem! The jacobian matrix of the stacked model cannot be inverted.");
        }
        return Ok((false, f64::NAN, 1));
    }
    for v in &mut bx {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    for (v, d) in yv.iter_mut().zip(&bx) {
        *v -= d;
    }
    let res1 = linres(&yv);
    let err = inf_norm(&res1);
    if verbose {
        println!(
            "Iter: 1,\t Initial err. = {},\t err. = {},\t time = {}",
            fmt_g(err0),
            fmt_g(err),
            fmt_g(h2.elapsed().as_secs_f64())
        );
        println!("{}", "-".repeat(80));
    }
    let success = !(res1.iter().any(|v| !v.is_finite()) || yv.iter().any(|v| !v.is_finite()));
    if !success && verbose {
        println!();
        println!("Simulation terminated with NaN or Inf in the residuals or endogenous variables.");
        println!(
            "There is most likely something wrong with your model. Try model_diagnostics or another simulation method."
        );
    }
    for t in 0..periods {
        for j in 0..n {
            endo[(j, ml + t)] = yv[t * n + j] + ss[j];
        }
    }
    if verbose {
        println!();
    }
    Ok((success, err, 1))
}

/// A light `model_diagnostics.m`: static Jacobian rank and collinear equations at the
/// current steady state.
pub fn model_diagnostics(model: &Model, ys: &[f64], xs: &[f64], p: &[f64]) {
    let j = model.static_jacobian(ys, xs, p);
    let sv = crate::linalg::singular_values(&j);
    let tol = sv.iter().cloned().fold(0.0, f64::max) * (j.nrows().max(j.ncols()) as f64) * f64::EPSILON;
    let rank = sv.iter().filter(|&&s| s > tol).count();
    if rank < j.nrows() {
        println!(
            "MODEL_DIAGNOSTICS: The Jacobian of the static model is singular\nMODEL_DIAGNOSTICS: there is {} colinear relationships between the variables and the equations",
            j.nrows() - rank
        );
    } else {
        println!("MODEL_DIAGNOSTICS: No obvious problems with this mod-file were detected.");
    }
}
