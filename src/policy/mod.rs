// SPDX-License-Identifier: GPL-3.0-or-later
//! Optimal policy (streams P, P2): `ramsey_model` / `ramsey_policy`, `evaluate_planner_objective`,
//! `discretionary_policy` and `osr`.
//!
//! The symbolic side (Lagrange multipliers, Ramsey FOCs, planner objective derivatives, the
//! `optimal_policy_discount_factor` parameter) is done by the preprocessor port in
//! [`crate::model::planner`]. This module holds the numerical side:
//!
//! * [`ramsey`]: the Ramsey steady state (`dyn_ramsey_static.m`, with or without
//!   `steady_state_model`, `csolve.m`; `ramsey_constraints` are parsed and ignored, as in
//!   Dynare 7.2);
//! * [`discretion`]: `discretionary_policy_1.m` / `discretionary_policy_engine.m`;
//! * [`osr`]: `+osr/run.m`, `+osr/objective.m` (orders 1 and 2, bounds);
//! * [`welfare`]: `evaluate_planner_objective.m` (order 1, order 2, linear-quadratic) and the
//!   theoretical variances it needs.
//!
//! [`Session`] runs the commands in file order (the driver hands it every
//! [`Command::Policy`] and, for Ramsey models, `stoch_simul`), keeping the policy parts of
//! `options_` and `oo_` (last decision rule, `oo_.exo_simul(1,:)`, the persistent `Hold` of
//! `discretionary_policy_1.m`) between commands as Dynare's globals do. `M_.params` and the
//! `stoch_simul` options are the driver's: a command reads them from the `ModFile` and from
//! [`Session::ss_opts`], and leaves the `M_.params` it returns (`osr`'s optimal parameters,
//! parameters set by a steady-state model) in [`Session::params`] for the driver to take
//! ([`Session::take_params`]).
//! The `stoch_simul` part of the output goes through stream S's `analysis::stoch_simul` (order
//! 1: model summary, decision rules, moments, decompositions, IRF files and figures) or stream
//! K's `korder::stoch_simul::run_at` (order >= 2), when the driver gave an output context.

pub mod discretion;
pub mod osr;
pub mod ramsey;
pub mod welfare;

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;

use crate::analysis::{LinearSolution, OutputContext, StochSimulOptions};
use crate::deterministic::SteadyStateModel;
use crate::linalg::Mat;
use crate::model::Model;
use crate::model::higher::DynamicDerivatives;
use crate::modfile::{Command, DetShockKind, ModFile, OptValue, Options, opt};
use crate::solver::korder::KOrderDr;
use crate::solver::{DecisionRule, FirstOrderSolver, SolveError, SolverOptions, SteadyOptions};
pub use discretion::DiscretionOptions;
pub use ramsey::RamseyConstraint;
pub use welfare::{InitialState, PlannerObjectiveValue};

/// Steady state + decision rule at a parameter vector (`resol.m` for the policy problems).
pub struct Resolver<'a> {
    pub model: &'a Model,
    pub solver: FirstOrderSolver,
    pub steady_opts: SteadyOptions,
    /// `oo_.steady_state` the steady-state search starts from (initval).
    pub ys_guess: Vec<f64>,
    pub qz_criterium: f64,
    /// `Some` for `discretionary_policy` (decision rule from the Dennis algorithm).
    pub discretion: Option<DiscretionOptions>,
    /// `steady_state_model` (`options_.steadystate_flag = 2`).
    pub ssm: Option<SteadyStateModel>,
    /// `oo_.exo_steady_state`.
    pub exo_ss: Vec<f64>,
}

impl<'a> Resolver<'a> {
    pub fn new(mf: &'a ModFile, discretion: bool) -> Self {
        let model = &mf.model;
        let mut ys_guess = vec![0.0; model.n_endo()];
        for &(i, v) in &mf.initval {
            ys_guess[i] = v;
        }
        Resolver {
            model,
            solver: FirstOrderSolver::new(&model.lead_lag_incidence, model.n_exo(), SolverOptions::default()),
            steady_opts: steady_options(mf),
            ys_guess,
            qz_criterium: 1.0 + 1e-6,
            discretion: discretion.then(DiscretionOptions::default),
            ssm: SteadyStateModel::from_modfile(mf).ok().flatten(),
            exo_ss: mf.exo_steady_state(),
        }
    }

    /// The first-order solver and the Lyapunov `qz_criterium` of `options_` (`resol` and
    /// `lyapunov_symm` read `qz_criterium`, `qz_zero_threshold` and `dr=` from `options_`).
    pub fn with_options(mut self, o: &StochSimulOptions) -> Self {
        let m = self.model;
        self.solver = FirstOrderSolver::new(&m.lead_lag_incidence, m.n_exo(), o.solver_options());
        self.qz_criterium = o.qz_criterium;
        if let Some(d) = self.discretion.as_mut() {
            d.qz_criterium = o.qz_criterium;
        }
        self
    }

    /// The steady state and the parameters (a steady-state model may change them) at
    /// parameters `p` (Ramsey: `dyn_ramsey_static`; otherwise `evaluate_steady_state`).
    pub fn steady_state_params(&self, p: &[f64]) -> Result<(Vec<f64>, Vec<f64>), SolveError> {
        let m = self.model;
        if m.planner().is_some_and(|pl| pl.ramsey.is_some()) {
            let rs = ramsey::RamseyStatic { model: m, ssm: self.ssm.as_ref(), exo_ss: &self.exo_ss };
            return rs.steady_state(p, &self.ys_guess, &self.steady_opts);
        }
        let x = self.exo_ss.clone();
        crate::deterministic::evaluate_steady_state(
            m,
            self.ssm.as_ref(),
            &self.ys_guess,
            &x,
            p,
            &self.steady_opts,
            true,
        )
    }

    /// The steady state at parameters `p`.
    pub fn steady_state(&self, p: &[f64]) -> Result<Vec<f64>, SolveError> {
        self.steady_state_params(p).map(|(y, _)| y)
    }

    /// The first-order decision rule at parameters `p`.
    pub fn resolve(&self, p: &[f64]) -> Result<DecisionRule, SolveError> {
        self.resolve_params(p).map(|(dr, _)| dr)
    }

    /// The first-order decision rule and the parameters it was computed with.
    pub fn resolve_params(&self, p: &[f64]) -> Result<(DecisionRule, Vec<f64>), SolveError> {
        if let Some(o) = &self.discretion {
            let steady = self.discretion_steady_state();
            return discretion::discretionary_policy_1(self.model, p, o, steady.as_ref(), self.steady_opts.tolf, None)
                .map(|(dr, p, _)| (dr, p))
                .map_err(|(info, info2)| SolveError::new(info, info2));
        }
        let (ys, p) = self.steady_state_params(p)?;
        let g1 = self.model.dynamic_jacobian(&ys, &self.exo_ss, &p);
        self.solver.solve(&g1, &ys).map(|dr| (dr, p))
    }

    /// The steady-state file `discretionary_policy_1.m` evaluates (`options_.steadystate_flag`).
    fn discretion_steady_state(&self) -> Option<discretion::DiscretionSteadyState<'_>> {
        let ssm = self.ssm.as_ref()?;
        Some(discretion::DiscretionSteadyState {
            ssm,
            ys_init: &self.ys_guess,
            opts: &self.steady_opts,
            exo_ss: &self.exo_ss,
        })
    }

    /// The decision rule at `order >= 2` (`deriv` = `Model::dynamic_derivatives(order)`).
    pub fn resolve_k(
        &self,
        p: &[f64],
        order: usize,
        deriv: &DynamicDerivatives,
        sigma_e: &Mat,
    ) -> Result<(KOrderDr, Vec<f64>), SolveError> {
        let (ys, p) = self.steady_state_params(p)?;
        let x = self.exo_ss.clone();
        let g1 = self.model.dynamic_jacobian(&ys, &x, &p);
        let dr1 = self.solver.solve(&g1, &ys)?;
        let vals = deriv.eval(&ys, &x, &p);
        crate::solver::korder::solve(&dr1, &self.model.lead_lag_incidence, &g1, deriv, &vals, sigma_e, order)
            .map(|k| (k, p))
            .map_err(|_| SolveError::new(5, f64::NAN))
    }
}

/// Options of the last `steady(...)` command (solve_algo, maxit, tolf, tolx).
fn steady_options(mf: &ModFile) -> SteadyOptions {
    let mut o = SteadyOptions { linear: mf.model.linear, ..Default::default() };
    for c in &mf.commands {
        if let Command::Steady(opts) = c {
            if let Some(v) = num(opts, "solve_algo") {
                o.solve_algo = v as u8;
            }
            if let Some(v) = num(opts, "maxit") {
                o.maxit = v as usize;
            }
            if let Some(v) = num(opts, "tolf") {
                o.tolf = v;
            }
            if let Some(v) = num(opts, "tolx") {
                o.tolx = v;
            }
        }
    }
    o
}

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

/// Result of an OSR run (`oo_.osr`).
#[derive(Clone, Debug)]
pub struct OsrResult {
    pub objective_function: f64,
    /// (parameter name, optimal value)
    pub optim_params: Vec<(String, f64)>,
    pub iterations: usize,
    pub func_count: usize,
}

/// Policy options that are not `stoch_simul` options (kept out of the stoch_simul warnings).
const POLICY_OPTIONS: &[&str] = &[
    "instruments",
    "planner_discount",
    "planner_discount_latex_name",
    "opt_algo",
    "discretionary_tol",
    "maxit",
    "tolf",
    "optim",
    "analytic_derivation",
    "analytic_derivation_mode",
    "huge_number",
];

/// `options_.osr` and `options_.optim_opt` (persistent across `osr` commands).
#[derive(Clone, Debug)]
pub struct OsrOptions {
    /// `options_.osr.opt_algo` (4).
    pub opt_algo: i32,
    /// `options_.osr.maxit`, `options_.osr.tolf` (deprecated; fields absent by default).
    pub maxit: Option<f64>,
    pub tolf: Option<f64>,
    /// `options_.optim_opt` (`optim` option).
    pub optim: Option<OptValue>,
    /// `options_.analytic_derivation` (false), `options_.analytic_derivation_mode` (0).
    pub analytic_derivation: bool,
    pub analytic_derivation_mode: i32,
    /// `options_.huge_number` (1e7 when `None`).
    pub huge_number: Option<f64>,
}

impl Default for OsrOptions {
    fn default() -> Self {
        OsrOptions {
            opt_algo: 4,
            maxit: None,
            tolf: None,
            optim: None,
            analytic_derivation: false,
            analytic_derivation_mode: 0,
            huge_number: None,
        }
    }
}

/// MATLAB `num2str` of a scalar: integers in full, other values with
/// `max(floor(log10(|x|)) + 5, 5)` significant digits (at most 16).
fn num2str(x: f64) -> f64 {
    if x == x.trunc() || !x.is_finite() {
        return x;
    }
    let dgt = (x.abs().log10().floor() as i64 + 5).clamp(5, 16) as usize;
    format!("{:.*e}", dgt - 1, x).parse().unwrap_or(x)
}

/// State carried between commands (the policy parts of Dynare's `options_` and `oo_`:
/// `oo_.dr`, `oo_.planner_objective_value`, `oo_.osr`, ...).
#[derive(Default)]
pub struct Session {
    /// `M_.params` as the last command returned it (`stoch_simul` / `osr` /
    /// `discretionary_policy` return `M_`), until the driver takes it into the `ModFile`
    /// ([`Session::take_params`]); while it is there, the next command reads it instead of
    /// `mf.model.params`.
    pub params: Option<Vec<f64>>,
    /// `oo_.endo_simul` of the last simulation (`periods > 0`), for `datatomfile`; taken by
    /// the driver.
    pub endo_simul: Option<Mat>,
    pub dr: Option<DecisionRule>,
    /// Second- or third-order decision rule of the last stoch_simul-like command.
    pub kdr: Option<KOrderDr>,
    /// `options_.order` after the last stoch_simul-like command (`stoch_simul.m` restores the
    /// order it lowers to 1 for `model(linear)` or a zero Hessian).
    pub order: usize,
    pub planner_objective_value: Option<PlannerObjectiveValue>,
    pub osr: Option<OsrResult>,
    /// Theoretical variance of the last var_list (declaration order of the list).
    pub var: Option<Mat>,
    pub var_list: Vec<usize>,
    pub quiet: bool,
    /// `options_` of the stoch_simul part: the driver's persistent options, which it copies
    /// in before [`Session::run`] and back afterwards.
    pub ss_opts: StochSimulOptions,
    /// `options_.osr`, `options_.optim_opt`.
    pub osr_opts: OsrOptions,
    /// `options_.dp.maxit`, `options_.discretionary_tol` (`qz_criterium` comes from `ss_opts`).
    pub dp_opts: DiscretionOptions,
    /// The persistent `Hold` of `discretionary_policy_1.m`: the last discretionary solution
    /// `H`, where the next Dennis iteration starts.
    pub hold: Option<Mat>,
    /// Where the stoch_simul output (files, figures) goes; `None`: no stoch_simul output.
    pub ctx: Option<OutputContext>,
    hessian_eq_zero: Option<bool>,
    /// `oo_.exo_simul(1,:)` after the last stoch_simul-like command (the first simulated shock
    /// when `periods > 0`, else `oo_.exo_steady_state`): the initial shock `u` of
    /// `evaluate_planner_objective`'s conditional welfare. `None`: `oo_.exo_simul` is empty.
    pub exo_simul_first: Option<Vec<f64>>,
    /// `options_.ramsey.periods` / `options_.ramsey.drop` (`evaluate_planner_objective(periods=,
    /// drop=)`; read by the order >= 3 simulated welfare). `None`: Dynare's defaults 10000/1000.
    pub ramsey_periods: Option<usize>,
    pub ramsey_drop: Option<usize>,
    /// Shocks of the order >= 3 simulated welfare (`n_exo x periods`) instead of drawing them
    /// (tests feeding MATLAB's draws).
    pub planner_shocks: Option<Mat>,
}

/// Whether [`Session::run`] handles this command (the driver calls it instead of its own arm).
pub fn handles(mf: &ModFile, c: &Command) -> bool {
    match c {
        Command::Policy(..) => true,
        Command::StochSimul(..) => mf.model.planner().is_some_and(|p| p.ramsey.is_some()),
        _ => false,
    }
}

impl Session {
    pub fn new(quiet: bool) -> Self {
        Session { quiet, ss_opts: StochSimulOptions::default(), ..Default::default() }
    }

    /// The driver's output context (`stoch_simul` printout through `analysis`, IRF files and
    /// figures, results JSON files).
    pub fn set_output(&mut self, ctx: OutputContext) {
        self.ctx = Some(ctx);
    }

    /// `M_.params`: the parameters the last command returned when the driver has not taken
    /// them yet, else the driver's.
    fn params(&self, mf: &ModFile) -> Vec<f64> {
        self.params.clone().unwrap_or_else(|| mf.model.params.clone())
    }

    /// The `M_.params` the last command returned (`None` if the driver already took them, or
    /// no command returned `M_`): the driver writes them into `mf.model.params`, so that the
    /// later commands of every area see them and the next policy command sees the parameter
    /// changes made in between.
    pub fn take_params(&mut self) -> Option<Vec<f64>> {
        self.params.take()
    }

    /// A resolver with the solver options of `options_`.
    fn resolver<'a>(&self, mf: &'a ModFile, discretion: bool) -> Resolver<'a> {
        Resolver::new(mf, discretion).with_options(&self.ss_opts)
    }

    /// `M_.hessian_eq_zero`.
    fn hessian_eq_zero(&mut self, m: &Model) -> bool {
        *self.hessian_eq_zero.get_or_insert_with(|| m.dynamic_derivatives(2).hessian_eq_zero())
    }

    /// Runs one command; `out` = `(output dir, fname)` for the results files.
    pub fn run(&mut self, mf: &ModFile, c: &Command, out: Option<(&Path, &str)>) -> Result<()> {
        match c {
            Command::StochSimul(o, vars) => {
                // the preprocessor writes options_.order = 2 when the command has no `order`
                // option (ramsey_policy and osr keep options_.order)
                if !o.iter().any(|(k, _)| k == "order") {
                    self.ss_opts.order = 2;
                }
                self.ramsey_stoch_simul(mf, o, vars, out)
            }
            Command::Policy(name, o, vars) => match name.as_str() {
                "ramsey_model" => self.apply_options(o, false),
                "osr_params" => Ok(()),
                "ramsey_policy" => {
                    self.ramsey_stoch_simul(mf, o, vars, out)?;
                    self.evaluate_planner_objective(mf, out, self.ss_opts.order)
                }
                "evaluate_planner_objective" => {
                    if let Some(v) = num(o, "periods") {
                        self.ramsey_periods = Some(v as usize);
                    }
                    if let Some(v) = num(o, "drop") {
                        self.ramsey_drop = Some(v as usize);
                    }
                    self.evaluate_planner_objective(mf, out, self.ss_opts.order)
                }
                "discretionary_policy" => self.discretionary_policy(mf, o, vars, out),
                "osr" => self.osr(mf, o, vars, out),
                _ => bail!("unknown policy command {name}"),
            },
            _ => Ok(()),
        }
    }

    /// Updates the persistent `stoch_simul` options with a command's options.
    fn apply_options(&mut self, o: &Options, warn: bool) -> Result<()> {
        let o2: Options = o.iter().filter(|(k, _)| !POLICY_OPTIONS.contains(&k.as_str())).cloned().collect();
        let unknown = self.ss_opts.apply_checked(&o2)?;
        if warn {
            for k in unknown {
                println!(
                    "WARNING: option `{k}` is not supported by dynare-rust's optimal policy commands and is ignored."
                );
            }
        }
        Ok(())
    }

    fn var_list(mf: &ModFile, vars: &[String]) -> Result<Vec<usize>> {
        let m = &mf.model;
        if vars.is_empty() {
            return Ok((0..m.orig_endo_nbr).collect());
        }
        let mut out = Vec::new();
        for v in vars {
            let j = m.endo_index(v).ok_or_else(|| anyhow!("unknown variable {v}"))?;
            if !out.contains(&j) {
                out.push(j);
            }
        }
        Ok(out)
    }

    /// The order `stoch_simul` solves at (`stoch_simul.m`: `model(linear)` and a zero Hessian
    /// force order 1 locally; `options_.order` is restored at the end).
    fn effective_order(&mut self, m: &Model) -> usize {
        let mut order = self.ss_opts.order.max(1);
        if order > 1 && m.linear {
            order = 1;
        }
        if order > 1 && self.hessian_eq_zero(m) {
            if !self.quiet {
                println!("WARNING: stoch_simul: using order = 1 because Hessian is equal to zero");
            }
            order = 1;
        }
        order
    }

    /// `stoch_simul` on a Ramsey model (`ramsey_model` + `stoch_simul`, or `ramsey_policy`).
    fn ramsey_stoch_simul(
        &mut self,
        mf: &ModFile,
        o: &Options,
        vars: &[String],
        out: Option<(&Path, &str)>,
    ) -> Result<()> {
        self.apply_options(o, true)?;
        self.exo_simul_first = Some(exo_steady_state(mf)); // oo_.exo_simul = exo_steady_state
        let m = &mf.model;
        ramsey::check_instruments(m, !mf.steady_state_model.is_empty())?;
        // ramsey_constraints: parsed (preprocessor errors), not used by dyn_ramsey_static.m
        ramsey::parse_constraints(m, &mf.policy.ramsey_constraints)?;
        if self.ss_opts.partial_information && self.ctx.is_some() {
            return self.ramsey_stoch_simul_pi(mf, vars);
        }
        let order = self.effective_order(m);
        self.order = self.ss_opts.order;
        let p0 = self.params(mf);
        let res = self.resolver(mf, false);
        let (ys, p) = res.steady_state_params(&p0).map_err(|e| anyhow!("Ramsey policy: {e}"))?;
        let iv = Self::var_list(mf, vars)?;
        if order == 1 {
            let g1 = m.dynamic_jacobian(&ys, &res.exo_ss, &p);
            let dr = res.solver.solve(&g1, &ys).map_err(|e| anyhow!("Ramsey policy: {e}"))?;
            self.params = Some(p);
            self.report(mf, &dr, &iv, vars, &res, out, "ramsey")?;
            self.dr = Some(dr);
            self.kdr = None;
        } else {
            let (kdr, first) = self.korder_stoch_simul(mf, order, vars, &ys, &p)?;
            self.params = Some(p);
            self.report_k(mf, &kdr, &iv, &res, out, "ramsey", first)?;
            self.dr = Some(kdr.first.clone());
            self.kdr = Some(kdr);
        }
        Ok(())
    }

    /// `stoch_simul(partial_information)` on a Ramsey model: `PCL_resol.m` at the Ramsey
    /// steady state (`evaluate_steady_state` -> `dyn_ramsey_static.m`), then `dr1_PI.m` on the
    /// model with the multipliers; `oo_.dr` is then the PCL solution, so the `_ramsey.json`
    /// decision rule is not written. `options_.partial_information` is reset afterwards.
    fn ramsey_stoch_simul_pi(&mut self, mf: &ModFile, vars: &[String]) -> Result<()> {
        let m = &mf.model;
        // (not noprint in quiet mode: PCL_varobs is only set when printing, and the IRFs and moments need it)
        let o = self.ss_opts.clone();
        self.ss_opts.partial_information = false; // stoch_simul.m: "temporary fix waiting for local options"
        crate::analysis::partial_info::stoch_simul_pi_prologue(m, &o)?;
        self.order = 1;
        let p0 = self.params(mf);
        let res = self.resolver(mf, false);
        let (ys, p) = res.steady_state_params(&p0).map_err(|e| anyhow!("Ramsey policy: {e}"))?;
        let ctx = self.ctx.as_ref().expect("checked by the caller");
        crate::analysis::partial_info::stoch_simul_pi_at(mf, &o, vars, ctx, ys.clone(), p.clone(), mf.sigma_e.clone())?;
        // the full-information rule, for the later commands that read this state
        let g1 = m.dynamic_jacobian(&ys, &res.exo_ss, &p);
        self.dr = res.solver.solve(&g1, &ys).ok();
        self.kdr = None;
        self.params = Some(p);
        self.exo_simul_first = Some(exo_steady_state(mf));
        Ok(())
    }

    /// The order >= 2 `stoch_simul` (stream K) at `(ys, p)`: prints and writes its files when
    /// there is an output context, else only solves. Also returns `oo_.exo_simul(1,:)` of its
    /// simulation (`periods > 0`).
    fn korder_stoch_simul(
        &self,
        mf: &ModFile,
        order: usize,
        vars: &[String],
        ys: &[f64],
        p: &[f64],
    ) -> Result<(KOrderDr, Option<Vec<f64>>)> {
        match &self.ctx {
            Some(ctx) => {
                let mut so = self.korder_options(mf, order);
                so.tex_ctx = Some(ctx.clone());
                let r = crate::solver::korder::stoch_simul::run_at(
                    mf,
                    &so,
                    vars,
                    &ctx.dir,
                    &ctx.fname,
                    ctx.seed,
                    Some((ys, p)),
                )?;
                let first = r.exo_simul.as_ref().filter(|e| e.nrows() > 0).map(|e| e.row(0).iter().copied().collect());
                Ok((r.dr, first))
            }
            None => crate::solver::korder::solve_model_at(
                &mf.model,
                ys,
                &mf.exo_steady_state(),
                p,
                &mf.sigma_e,
                order,
                self.ss_opts.solver_options(),
            )
            .map(|k| (k, None)),
        }
    }

    /// Stream K's options from the persistent `options_` (as the driver's `stoch_simul` at
    /// order >= 2: pruning, replic, filters, ... and `M_.H`), at `order`.
    pub fn korder_options(&self, mf: &ModFile, order: usize) -> crate::solver::korder::stoch_simul::StochSimulOptions {
        let mut a = self.ss_opts.clone();
        a.me = crate::analysis::MeasurementErrors::from_modfile(mf);
        a.order = order;
        a.noprint |= self.quiet;
        crate::solver::korder::stoch_simul::StochSimulOptions::from_analysis(&a)
    }

    /// Order-1 output: stream S's `stoch_simul` printout/files when there is an output
    /// context, the policy results JSON, and the theoretical variance of the var_list.
    #[allow(clippy::too_many_arguments)]
    fn report(
        &mut self,
        mf: &ModFile,
        dr: &DecisionRule,
        iv: &[usize],
        vars: &[String],
        res: &Resolver,
        out: Option<(&Path, &str)>,
        what: &str,
    ) -> Result<()> {
        let rows: Vec<usize> = iv.iter().map(|&j| dr.inv_order_var[j]).collect();
        let var = welfare::variance(dr, &rows, &mf.sigma_e, res.qz_criterium);
        let mut first = None;
        if let Some(ctx) = &self.ctx {
            let sol = LinearSolution { dr: dr.clone(), sigma_e: mf.sigma_e.clone(), params: self.params(mf) };
            let mut o = self.ss_opts.clone();
            o.order = 1;
            o.noprint |= self.quiet;
            let names: Vec<String> = if vars.is_empty() {
                iv.iter().map(|&j| mf.model.endo_names[j].clone()).collect()
            } else {
                vars.to_vec()
            };
            let r = crate::analysis::stoch_simul(&mf.model, &sol, &o, &names, ctx, None)?;
            if let Some(ex) = r.exo_simul.as_ref().filter(|e| e.nrows() > 0) {
                first = Some(ex.row(0).iter().copied().collect());
            }
            if r.endo_simul.is_some() {
                self.endo_simul = r.endo_simul;
            }
        } else if !self.quiet {
            let sol = LinearSolution { dr: dr.clone(), sigma_e: mf.sigma_e.clone(), params: self.params(mf) };
            print!("{}", crate::analysis::model_summary(&mf.model, &sol));
            print!("\n{}", crate::analysis::disp_dr(&mf.model, dr, iv, 1e-6));
            print_moments(&mf.model, dr, iv, &var);
        }
        self.exo_simul_first = Some(first.unwrap_or_else(|| exo_steady_state(mf)));
        self.var = Some(var.clone());
        self.var_list = iv.to_vec();
        if let Some((dir, fname)) = out {
            let m = &mf.model;
            let j = json!({
                "command": what,
                "order": 1,
                "endo_names": m.endo_names,
                "steady_state": dr.ys,
                "params": self.params(mf),
                "ghx": mat_rows(&dr.ghx),
                "ghu": mat_rows(&dr.ghu),
                "order_var": dr.order_var.iter().map(|v| v + 1).collect::<Vec<_>>(),
                "state_var": dr.state_var.iter().map(|v| v + 1).collect::<Vec<_>>(),
                "var_list": iv.iter().map(|&j| m.endo_names[j].clone()).collect::<Vec<_>>(),
                "var": mat_rows(&var),
            });
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            std::fs::write(dir.join(format!("{fname}_{what}.json")), serde_json::to_string_pretty(&j)?)?;
        }
        Ok(())
    }

    /// Order >= 2: the policy results JSON (the printout and IRFs come from stream K);
    /// `first` is `oo_.exo_simul(1,:)` of stream K's simulation.
    #[allow(clippy::too_many_arguments)]
    fn report_k(
        &mut self,
        mf: &ModFile,
        kdr: &KOrderDr,
        iv: &[usize],
        res: &Resolver,
        out: Option<(&Path, &str)>,
        what: &str,
        first: Option<Vec<f64>>,
    ) -> Result<()> {
        let dr = &kdr.first;
        let rows: Vec<usize> = iv.iter().map(|&j| dr.inv_order_var[j]).collect();
        let var = welfare::variance(dr, &rows, &mf.sigma_e, res.qz_criterium);
        self.exo_simul_first = Some(first.unwrap_or_else(|| exo_steady_state(mf)));
        self.var = Some(var.clone());
        self.var_list = iv.to_vec();
        if let Some((dir, fname)) = out {
            let m = &mf.model;
            let j = json!({
                "command": what,
                "order": kdr.order,
                "endo_names": m.endo_names,
                "steady_state": dr.ys,
                "params": self.params(mf),
                "ghx": mat_rows(&dr.ghx),
                "ghu": mat_rows(&dr.ghu),
                "ghxx": mat_rows(&kdr.ghxx),
                "ghxu": mat_rows(&kdr.ghxu),
                "ghuu": mat_rows(&kdr.ghuu),
                "ghs2": kdr.ghs2.as_slice(),
                "order_var": dr.order_var.iter().map(|v| v + 1).collect::<Vec<_>>(),
                "state_var": dr.state_var.iter().map(|v| v + 1).collect::<Vec<_>>(),
                "var_list": iv.iter().map(|&j| m.endo_names[j].clone()).collect::<Vec<_>>(),
                "var": mat_rows(&var),
            });
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            std::fs::write(dir.join(format!("{fname}_{what}.json")), serde_json::to_string_pretty(&j)?)?;
        }
        Ok(())
    }

    /// Port of `evaluate_planner_objective.m` on the last decision rule, at `options_.order =
    /// order`.
    fn evaluate_planner_objective(&mut self, mf: &ModFile, out: Option<(&Path, &str)>, order: usize) -> Result<()> {
        let m = &mf.model;
        let p = self.params(mf);
        let beta = m.planner_discount(&p).unwrap_or(1.0);
        let discretionary = m.planner().is_some_and(|pl| pl.discretionary);
        let qz = self.ss_opts.qz_criterium;
        let v = self.welfare(mf, &p, discretionary, qz, order)?;
        if beta >= 1.0 {
            println!(
                "evaluate_planner_objective: the planner discount factor is not strictly smaller than 1. Unconditional welfare will not be finite."
            );
        }
        if !self.quiet {
            let ramsey = m.planner().is_some_and(|pl| pl.ramsey.is_some());
            if ramsey {
                println!("\nApproximated value of unconditional welfare:  {:10.8}", v.unconditional);
                println!("\nApproximated value of conditional welfare:");
                println!(
                    "    - with initial Lagrange multipliers set to 0: {:10.8}",
                    v.conditional_zero_initial_multiplier
                );
                println!(
                    "    - with initial Lagrange multipliers set to steady state: {:10.8}\n",
                    v.conditional_steady_initial_multiplier
                );
            } else if discretionary {
                println!(
                    "\nApproximated value of unconditional welfare with discretionary policy:  {:10.8}",
                    v.unconditional
                );
                println!(
                    "\nApproximated value of conditional welfare with discretionary policy: {:10.8}",
                    v.conditional_steady_initial_multiplier
                );
            } else {
                println!("\nApproximated value of unconditional welfare:  {:10.8}", v.unconditional);
                println!(
                    "\nApproximated value of conditional welfare: {:10.8}",
                    v.conditional_steady_initial_multiplier
                );
            }
        }
        if let Some((dir, fname)) = out {
            let j = json!({
                "order": order,
                "unconditional": v.unconditional,
                "conditional_zero_initial_multiplier": v.conditional_zero_initial_multiplier,
                "conditional_steady_initial_multiplier": v.conditional_steady_initial_multiplier,
            });
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join(format!("{fname}_planner_objective.json")), serde_json::to_string_pretty(&j)?)?;
        }
        self.planner_objective_value = Some(v);
        Ok(())
    }

    /// The branch of `evaluate_planner_objective.m` for `options_.order = order` and the model.
    fn welfare(
        &mut self,
        mf: &ModFile,
        p: &[f64],
        discretionary: bool,
        qz: f64,
        order: usize,
    ) -> Result<PlannerObjectiveValue> {
        let m = &mf.model;
        let hz = order == 2 && self.hessian_eq_zero(m);
        if self.dr.is_none() {
            bail!(
                "evaluate_planner_objective requires decision rules to have previously been computed (e.g. by stoch_simul or discretionary_policy)"
            );
        }
        let init = self.initial_state(mf)?;
        let dr = self.dr.as_ref().expect("checked above");
        let none = || anyhow!("evaluate_planner_objective: no planner_objective");
        if order == 1 && !discretionary {
            return welfare::evaluate_planner_objective_from(m, p, dr, &mf.sigma_e, false, qz, &init).ok_or_else(none);
        }
        if order == 2 && !hz {
            // full second order approximation
            let kdr = self
                .kdr
                .as_ref()
                .ok_or_else(|| anyhow!("evaluate_planner_objective: no second-order decision rule"))?;
            return welfare::evaluate_planner_objective_order2_from(m, p, kdr, &mf.sigma_e, qz, &init).ok_or_else(none);
        }
        if hz || discretionary {
            // linear quadratic problem
            return welfare::evaluate_planner_objective_from(m, p, dr, &mf.sigma_e, true, qz, &init).ok_or_else(none);
        }
        // order > 2: k_order_welfare + simulated unconditional welfare
        let kdr = self.kdr.as_ref().ok_or_else(|| anyhow!("evaluate_planner_objective: no k-order decision rule"))?;
        let periods = self.ramsey_periods.unwrap_or(10000);
        let drop = self.ramsey_drop.unwrap_or(1000);
        let ex = match &self.planner_shocks {
            Some(e) => e.clone(),
            None => {
                // exo_simul(i_exo_var,:) = chol(Sigma_e(i_exo_var,i_exo_var))*randn(nxs, periods)
                use rand::SeedableRng;
                let se = &mf.sigma_e;
                let ne = m.n_exo();
                let iv: Vec<usize> = (0..ne).filter(|&i| se[(i, i)] != 0.0).collect();
                let sub = Mat::from_fn(iv.len(), iv.len(), |i, j| se[(iv[i], iv[j])]);
                let l = sub
                    .cholesky()
                    .ok_or_else(|| anyhow!("evaluate_planner_objective: Sigma_e is not positive definite"))?
                    .l();
                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(self.ctx.as_ref().map_or(0, |c| c.seed));
                let z = Mat::from_fn(iv.len(), periods, |_, _| {
                    rand::Rng::sample::<f64, _>(&mut rng, rand_distr::StandardNormal)
                });
                let x = l.transpose() * z; // chol(S) is upper: R'R = S
                let mut ex = Mat::zeros(ne, periods);
                for (a, &i) in iv.iter().enumerate() {
                    ex.set_row(i, &x.row(a));
                }
                ex
            }
        };
        welfare::evaluate_planner_objective_korder(m, p, kdr, &mf.sigma_e, &ex, drop, &init)
    }

    /// `get_initial_state` of `evaluate_planner_objective.m`: `M_.endo_histval` and the initial
    /// shock `u` (`oo_.exo_steady_state` with the period-1 values of `M_.det_shocks`, else
    /// `oo_.exo_simul(1,:)`, else `oo_.exo_steady_state`).
    fn initial_state(&self, mf: &ModFile) -> Result<InitialState> {
        let endo_histval = crate::analysis::stochastic_histval_of(&mf.model, &mf.histval);
        let u = if !mf.det_shocks.is_empty() {
            if self.exo_simul_first.as_ref().is_some_and(|u| u.iter().any(|&v| v != 0.0)) {
                println!(
                    "\nevaluate_planner_objective: oo_.exo_simul contains simulated values for the initial period.\n\
                     evaluate_planner_objective: Dynare will ignore them and use the provided initial condition."
                );
            }
            let mut u = exo_steady_state(mf);
            let periods: Vec<i64> = mf.det_shocks.iter().flat_map(|d| d.periods.iter().copied()).collect();
            if periods.contains(&0) {
                println!(
                    "\nevaluate_planner_objective: M_.det_shocks contains values for the predetermined t=0 period.\n\
                     evaluate_planner_objective: Dynare will ignore them. Use histval to set the value of lagged innovations."
                );
            }
            if periods.iter().any(|&t| t > 1) {
                println!(
                    "\nevaluate_planner_objective: Shock values for periods not contained in the initial information set (t=1) have been provided.\n\
                     evaluate_planner_objective: Note that they will be ignored."
                );
            }
            // shock_indices = find(periods == 1), which index M_.det_shocks itself
            let n = mf.det_shocks.len();
            let shocks = (periods.iter().enumerate().filter(|(_, t)| **t == 1))
                .map(|(k, _)| {
                    mf.det_shocks.get(k).ok_or_else(|| {
                        anyhow!(
                            "evaluate_planner_objective: Index exceeds the number of array elements. Index must not exceed {n}."
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if shocks.iter().any(|d| d.kind != DetShockKind::Level) {
                println!("\nevaluate_planner_objective: Shock values need to be specified in level.");
            }
            for d in shocks {
                u[d.exo] = d.value;
            }
            u
        } else {
            // first value of the simulation (set by simult if periods > 0), else the steady state
            self.exo_simul_first.clone().unwrap_or_else(|| exo_steady_state(mf))
        };
        Ok(InitialState { endo_histval, u })
    }

    /// Port of `discretionary_policy.m`.
    fn discretionary_policy(
        &mut self,
        mf: &ModFile,
        o: &Options,
        vars: &[String],
        out: Option<(&Path, &str)>,
    ) -> Result<()> {
        self.apply_options(o, true)?;
        // discretionary_policy_initialization.m
        if self.ss_opts.loglinear {
            bail!("discretionary_policy is not compatible with `loglinear` option set to 1");
        }
        let m = &mf.model;
        let pl = m.planner().ok_or_else(|| anyhow!("discretionary_policy: no planner_objective"))?;
        let instr_nbr = m.n_endo() - m.n_equations();
        if instr_nbr == 0 {
            bail!(
                "discretionary_policy:: There are no available instruments, because the model has as many equations as variables."
            );
        }
        if pl.instruments.len() < instr_nbr {
            bail!("discretionary_policy:: There are fewer declared instruments than omitted equations.");
        } else if pl.instruments.len() > instr_nbr {
            bail!("discretionary_policy:: There are more declared instruments than omitted equations.");
        }
        // options_.discretionary_tol, options_.dp.maxit (persistent)
        if let Some(v) = num(o, "discretionary_tol") {
            self.dp_opts.tol = v;
        }
        if let Some(v) = num(o, "maxit") {
            self.dp_opts.maxit = v as usize;
        }
        let p = self.params(mf);
        let res = self.resolver(mf, true);
        let dopts = DiscretionOptions { qz_criterium: self.ss_opts.qz_criterium, ..self.dp_opts };
        // stoch_simul at order 1: discretionary_policy_1, starting from the persistent Hold
        self.exo_simul_first = Some(exo_steady_state(mf));
        let steady = res.discretion_steady_state();
        let (dr, p, h) = discretion::discretionary_policy_1(
            m,
            &p,
            &dopts,
            steady.as_ref(),
            res.steady_opts.tolf,
            self.hold.as_ref(),
        )
        .map_err(|(info, info2)| anyhow!("discretionary_policy: {}", SolveError::new(info, info2)))?;
        self.hold = Some(h);
        self.order = self.ss_opts.order; // options_.order = origorder
        self.params = Some(p);
        let iv = Self::var_list(mf, vars)?;
        self.report(mf, &dr, &iv, vars, &res, out, "discretionary_policy")?;
        if !self.quiet && !self.ss_opts.noprint {
            // discretionary_policy.m: disp_steady_state, then the multipliers
            crate::deterministic::print_steady_state(m, &dr.ys);
            for j in m.orig_endo_nbr.saturating_sub(1)..m.endo_names.len() {
                if m.endo_names[j].starts_with("mult_") {
                    println!("{} \t\t {}", m.endo_names[j], crate::deterministic::fmt_g(dr.ys[j]));
                }
            }
        }
        self.dr = Some(dr);
        self.kdr = None;
        self.evaluate_planner_objective(mf, out, 1)
    }

    /// Port of `+osr/run.m`.
    fn osr(&mut self, mf: &ModFile, o: &Options, vars: &[String], out: Option<(&Path, &str)>) -> Result<()> {
        self.apply_options(o, true)?;
        // options_.osr.opt_algo / maxit / tolf and options_.optim_opt persist
        if let Some(v) = num(o, "opt_algo") {
            self.osr_opts.opt_algo = v as i32;
        }
        if let Some(v) = num(o, "maxit") {
            self.osr_opts.maxit = Some(v);
        }
        if let Some(v) = num(o, "tolf") {
            self.osr_opts.tolf = Some(v);
        }
        if let Some(v) = opt(o, "optim") {
            self.osr_opts.optim = Some(v.clone());
        }
        // options_.analytic_derivation / analytic_derivation_mode / huge_number persist too
        match opt(o, "analytic_derivation") {
            None => {}
            Some(OptValue::Flag) => self.osr_opts.analytic_derivation = true,
            Some(OptValue::Num(v)) => self.osr_opts.analytic_derivation = *v != 0.0,
            Some(OptValue::Str(v)) if v.eq_ignore_ascii_case("true") => self.osr_opts.analytic_derivation = true,
            Some(OptValue::Str(v)) if v.eq_ignore_ascii_case("false") => self.osr_opts.analytic_derivation = false,
            Some(_) => bail!("osr: analytic_derivation must be a boolean"),
        }
        if let Some(v) = num(o, "analytic_derivation_mode") {
            self.osr_opts.analytic_derivation_mode = v as i32;
        }
        if let Some(v) = num(o, "huge_number") {
            self.osr_opts.huge_number = Some(v);
        }
        let print = !self.quiet && !self.ss_opts.noprint;
        let m = &mf.model;
        let order = self.ss_opts.order.max(1);
        let pd = &mf.policy;
        let use_weights = !pd.optim_weights.is_empty();
        if use_weights && order != 1 {
            bail!("OSR: OSR without planner_objective only supports order=1.");
        }
        if !use_weights && self.osr_opts.analytic_derivation {
            bail!("OSR: analytic_derivation is only available at order=1 with optim_weights.");
        }
        if !use_weights && order > 2 {
            bail!(
                "OSR: osr with planner_objective at order={order} (simulated welfare) is not ported; use order = 1 or 2."
            );
        }
        if pd.osr_params.is_empty() {
            bail!("OSR: no osr_params declared");
        }
        if print {
            println!("\nOPTIMAL SIMPLE RULE");
        }
        if m.lead_lag_incidence.iter().all(|r| r[2] == 0) {
            bail!("OSR: Backward or static model: no point in using OSR");
        }
        let p0 = self.params(mf);
        if pd.osr_params.iter().any(|&i| p0[i].is_nan()) {
            bail!("OSR: At least one of the initial parameter values for osr_params is NaN");
        }
        // restore backward compatibility with maxit and tolf (appended to options_.optim_opt)
        if self.osr_opts.maxit.is_some() || self.osr_opts.tolf.is_some() {
            println!("Warning: OSR: The use of maxit and tolf is now deprecated. Please use the optim-option instead.");
            if self.osr_opts.opt_algo != 4 {
                bail!(
                    "OSR: The maxit and tolf options are not supported when not using the default opt_algo=4. Use the optim-option instead."
                );
            }
            let mut items: Vec<OptValue> = match self.osr_opts.optim.take() {
                Some(OptValue::List(l)) => l,
                Some(v) => vec![v],
                None => Vec::new(),
            };
            if let Some(v) = self.osr_opts.maxit {
                items.extend([OptValue::Str("'MaxIter'".into()), OptValue::Num(num2str(v))]);
            }
            if let Some(v) = self.osr_opts.tolf {
                items.extend([OptValue::Str("'TolFun'".into()), OptValue::Num(num2str(v))]);
            }
            self.osr_opts.optim = Some(OptValue::List(items));
        }
        let analytic = if use_weights && self.osr_opts.analytic_derivation {
            match self.osr_opts.analytic_derivation_mode {
                0 | 1 => {}
                // get_perturbation_params_derivs.m with estim_params_ = [] (as +osr/objective.m
                // passes it) reads estim_params_.param_vals, which does not exist
                -1 | -2 => bail!(
                    "OSR: analytic_derivation_mode={}: Unrecognized field name \"param_vals\" (Dynare 7.2's \
                     get_perturbation_params_derivs.m needs an estimated_params block for numerical derivatives; \
                     use analytic_derivation_mode=0)",
                    self.osr_opts.analytic_derivation_mode
                ),
                k => bail!("OSR: analytic_derivation_mode={k} is not 0, 1, -1 or -2"),
            }
            Some(crate::identification::derivs::ModelParamDerivs::new(m).at_exo_steady_state(&mf.exo_steady_state()))
        } else {
            None
        };
        let res = self.resolver(mf, false);
        let weights = if use_weights {
            // i_var = unique(i_var); positions in the decision rule
            let (iv, w) = osr::weight_block(m, &pd.optim_weights);
            let dr0 = res.resolve(&p0).ok();
            let rows = match &dr0 {
                Some(dr) => iv.iter().map(|&j| dr.inv_order_var[j]).collect(),
                None => {
                    let s = crate::solver::DrStructure::new(&m.lead_lag_incidence, m.n_exo());
                    iv.iter().map(|&j| s.inv_order_var[j]).collect()
                }
            };
            Some((rows, w))
        } else {
            None
        };
        // the welfare branch osr.objective takes (resol at options_.order, no stoch_simul reset)
        let welfare_kind = if use_weights || order == 1 {
            osr::WelfareKind::Order1
        } else if self.hessian_eq_zero(m) {
            osr::WelfareKind::LinearQuadratic
        } else {
            osr::WelfareKind::Order2(m.dynamic_derivatives(2))
        };
        let obj = osr::OsrObjective {
            resolver: &res,
            base_params: p0.clone(),
            i_params: pd.osr_params.clone(),
            weights,
            sigma_e: mf.sigma_e.clone(),
            welfare: welfare_kind,
            analytic,
        };
        let x0: Vec<f64> = pd.osr_params.iter().map(|&i| p0[i]).collect();
        let (l0, info0, _) = obj.loss(&x0);
        if info0 != 0 {
            // print_info: an error unless noprint
            if !self.ss_opts.noprint {
                bail!("OSR: {}", crate::solver::describe(info0));
            }
        } else if print {
            println!("\nOSR: Initial value of the objective function: {} \n", crate::analysis::table::fmt_g(l0, 0, 6));
        }
        if l0.is_infinite() {
            bail!("OSR: Initial likelihood is infinite");
        }
        let opt_algo = self.osr_opts.opt_algo;
        let bounds: Vec<(f64, f64)> = pd
            .osr_params
            .iter()
            .map(|&i| {
                pd.osr_bounds
                    .iter()
                    .rev()
                    .find(|b| b.0 == i)
                    .map(|b| (b.1, b.2))
                    .unwrap_or((f64::NEG_INFINITY, f64::INFINITY))
            })
            .collect();
        let names: Vec<String> = pd.osr_params.iter().map(|&i| m.param_names[i].clone()).collect();
        let r = osr::minimize(
            &obj,
            &x0,
            opt_algo,
            (!pd.osr_bounds.is_empty()).then_some(&bounds[..]),
            &names,
            self.osr_opts.optim.as_ref(),
            !self.quiet,
            self.ctx.as_ref().map_or(0, |c| c.seed), // --seed (stochastic optimizers 2, 9)
            self.osr_opts.huge_number,
        )?;
        let mut p = p0.clone();
        for (&i, &v) in pd.osr_params.iter().zip(&r.x) {
            p[i] = v;
        }
        let optim_params: Vec<(String, f64)> =
            pd.osr_params.iter().zip(&r.x).map(|(&i, &v)| (m.param_names[i].clone(), v)).collect();
        if print {
            // osr.run: dyntable(options_, title, {'Variables';'Value'}, labels, p, lh, 10, 6)
            let labels: Vec<String> = optim_params.iter().map(|(n, _)| n.clone()).collect();
            let vals = Mat::from_fn(labels.len(), 1, |i, _| optim_params[i].1);
            let headers = vec!["Variables".to_string(), "Value".to_string()];
            let lh = crate::analysis::table::max_len(&labels) + 2;
            print!(
                "{}",
                crate::analysis::table::dyntable(
                    "OPTIMAL VALUE OF THE PARAMETERS",
                    &headers,
                    &labels,
                    &vals,
                    lh,
                    10,
                    6
                )
            );
            println!("\nObjective function : {}\n", crate::analysis::table::fmt_g(r.fval, 16, 6));
        }
        self.osr = Some(OsrResult {
            objective_function: r.fval,
            optim_params: optim_params.clone(),
            iterations: r.iterations,
            func_count: r.func_count,
        });
        // stoch_simul at the optimum (M_.params(i_params) = p)
        self.exo_simul_first = Some(exo_steady_state(mf));
        let st_order = self.effective_order(m);
        self.order = self.ss_opts.order;
        let iv = Self::var_list(mf, vars)?;
        let (ys, p) =
            res.steady_state_params(&p).map_err(|e| anyhow!("osr: no solution at the optimal parameters: {e}"))?;
        self.params = Some(p.clone());
        if st_order == 1 {
            let dr = res.resolve(&p).map_err(|e| anyhow!("osr: no solution at the optimal parameters: {e}"))?;
            self.report(mf, &dr, &iv, vars, &res, out, "osr")?;
            self.dr = Some(dr);
            self.kdr = None;
        } else {
            let (kdr, first) = self.korder_stoch_simul(mf, st_order, vars, &ys, &p)?;
            self.report_k(mf, &kdr, &iv, &res, out, "osr", first)?;
            self.dr = Some(kdr.first.clone());
            self.kdr = Some(kdr);
        }
        if let Some((dir, fname)) = out {
            let j = json!({
                "objective_function": r.fval,
                "optim_params": optim_params.iter().map(|(n, v)| json!({"name": n, "value": v})).collect::<Vec<_>>(),
                "iterations": r.iterations,
                "func_count": r.func_count,
            });
            std::fs::write(dir.join(format!("{fname}_osr_results.json")), serde_json::to_string_pretty(&j)?)?;
        }
        Ok(())
    }
}

/// `oo_.exo_steady_state` (the exogenous values of the `initval` block, zero by default).
fn exo_steady_state(mf: &ModFile) -> Vec<f64> {
    let mut x = vec![0.0; mf.model.n_exo()];
    for &(i, v) in &mf.initval_exo {
        if let Some(e) = x.get_mut(i) {
            *e = v;
        }
    }
    x
}

fn mat_rows(m: &Mat) -> Vec<Vec<f64>> {
    (0..m.nrows()).map(|i| m.row(i).iter().copied().collect()).collect()
}

/// The mean / std / variance table of `disp_th_moments.m` (order 1; used without output context).
fn print_moments(m: &Model, dr: &DecisionRule, iv: &[usize], var: &Mat) {
    println!("\nTHEORETICAL MOMENTS");
    println!("{:<12}{:>10}{:>11}{:>11}", "VARIABLE", "MEAN", "STD. DEV.", "VARIANCE");
    for (k, &j) in iv.iter().enumerate() {
        let v = var[(k, k)];
        println!("{:<12}{:>10.4}{:>11.4}{:>11.4}", m.endo_names[j], dr.ys[j], v.max(0.0).sqrt(), v);
    }
}
