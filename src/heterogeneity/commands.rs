// SPDX-License-Identifier: GPL-3.0-or-later
// ---- heterogeneity ----
//! `heterogeneity_load_steady_state` and `heterogeneity_compute_steady_state`: option parsing
//! (`options_.heterogeneity`), steady-state input from a MAT file or from the session's
//! "workspace", ports of `+heterogeneity/load_steady_state.m` and `compute_steady_state.m`.

use super::check::{CheckOptions, DIn, FreeParameter, NdArray, PolIn, ShocksIn, SteadyStateData, SteadyStateInput};
use super::steady::{LoadContext, LoadedSteadyState, load_steady_state};
use super::time_iteration::{
    CalibrationOptions, CalibrationOutput, ForwardOptions, SteadySolver, TargetEquation, TimeIterationOptions,
    calibrate,
};
use crate::matfile::MatValue;
use crate::modfile::{ModFile, OptValue, Options};
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;
use std::path::Path;

/// `options_.heterogeneity` (the parts of the steady-state commands).
#[derive(Clone, Debug)]
pub struct HetOptions {
    pub steady_state_file_name: String,
    pub steady_state_variable_name: String,
    pub check: CheckOptions,
    pub forward: ForwardOptions,
    pub time_iteration: TimeIterationOptions,
    pub calibration: CalibrationOptions,
}

impl Default for HetOptions {
    fn default() -> Self {
        HetOptions {
            steady_state_file_name: String::new(),
            steady_state_variable_name: "steady_state".into(),
            check: CheckOptions::default(),
            forward: ForwardOptions::default(),
            time_iteration: TimeIterationOptions::default(),
            calibration: CalibrationOptions::default(),
        }
    }
}

fn num(v: &OptValue, name: &str) -> Result<f64> {
    match v {
        OptValue::Num(x) => Ok(*x),
        OptValue::Flag => Ok(1.0),
        _ => bail!("heterogeneity: option {name} expects a number"),
    }
}

fn string(v: &OptValue) -> String {
    match v {
        OptValue::Str(s) => s.trim_matches('\'').to_string(),
        OptValue::Num(x) => format!("{x}"),
        _ => String::new(),
    }
}

impl HetOptions {
    /// Reads the options of a `heterogeneity_*_steady_state` command on top of the defaults.
    pub fn from_options(opts: &Options) -> Result<HetOptions> {
        let mut o = HetOptions::default();
        o.apply(opts)?;
        Ok(o)
    }

    /// The assignments to `options_.heterogeneity` made by the options of a
    /// `heterogeneity_*_steady_state` command: the preprocessor writes only the options that are
    /// given, so the others keep the value an earlier command left (see [`HetSession`]).
    pub fn apply(&mut self, opts: &Options) -> Result<()> {
        let o = self;
        for (k, v) in opts {
            match k.as_str() {
                "filename" => o.steady_state_file_name = string(v),
                "variable" => o.steady_state_variable_name = string(v),
                "tolf" => o.check.tolf = num(v, k)?,
                "forward_max_iter" => o.forward.max_iter = num(v, k)? as usize,
                "forward_tol" => o.forward.tol = num(v, k)?,
                "forward_check_every" => o.forward.check_every = num(v, k)? as usize,
                "forward_verbosity" => o.forward.verbosity = num(v, k)? as i32,
                "time_iteration_max_iter" => o.time_iteration.max_iter = num(v, k)? as usize,
                "time_iteration_tol" => o.time_iteration.tol = num(v, k)?,
                "time_iteration_learning_rate" => o.time_iteration.learning_rate = num(v, k)?,
                "time_iteration_verbosity" => o.time_iteration.verbosity = num(v, k)? as i32,
                "time_iteration_solver_tolf" => o.time_iteration.solver_tolf = num(v, k)?,
                "time_iteration_solver_tolx" => o.time_iteration.solver_tolx = num(v, k)?,
                "time_iteration_solver_factor" => o.time_iteration.solver_factor = num(v, k)?,
                "time_iteration_solver_max_iter" => o.time_iteration.solver_max_iter = num(v, k)? as usize,
                "time_iteration_solver_stop_on_error" => o.time_iteration.solver_stop_on_error = true,
                "time_iteration_early_stopping" => o.time_iteration.early_stopping = num(v, k)? as usize,
                "calibration_tolf" => o.calibration.tolf = num(v, k)?,
                "calibration_max_iter" => o.calibration.max_iter = num(v, k)? as usize,
                "calibration_verbosity" => o.calibration.verbosity = num(v, k)? as i32,
                "calibration_target_equations" => {
                    o.calibration.target_equations.clear();
                    let items: Vec<&OptValue> = match v {
                        OptValue::List(l) => l.iter().collect(),
                        other => vec![other],
                    };
                    for it in items {
                        match it {
                            OptValue::Num(x) => {
                                o.calibration.target_equations.push(TargetEquation::Number(*x as usize))
                            }
                            OptValue::Str(s) => o
                                .calibration
                                .target_equations
                                .push(TargetEquation::Name(s.trim_matches('\'').to_string())),
                            _ => bail!("Calibration target equation must be a number or string"),
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// C `%.<prec>e`.
fn fmt_e(x: f64, prec: usize) -> String {
    let s = format!("{:.*e}", prec, x);
    match s.split_once('e') {
        Some((m, e)) => {
            let e: i32 = e.parse().unwrap_or(0);
            format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
        }
        None => s,
    }
}

/// Reads `steady_state` from the MAT file / workspace variable the options designate
/// (`options_het.steady_state_file_name`, `steady_state_variable_name`).
pub fn read_steady_state(
    cmd: &str,
    opts: &HetOptions,
    dir: &Path,
    workspace: &HashMap<String, MatValue>,
) -> Result<SteadyStateInput> {
    let var = &opts.steady_state_variable_name;
    let v = if opts.steady_state_file_name.is_empty() {
        // the MATLAB workspace is not available: the session workspace, else `<variable>.mat`
        // next to the .mod file (holding that variable)
        let from_file = dir.join(format!("{var}.mat"));
        let from_file = if from_file.is_file() {
            crate::matfile::read(&from_file).ok().and_then(|mut m| m.remove(var))
        } else {
            None
        };
        match workspace.get(var).cloned().or(from_file) {
            Some(v) => v,
            None => bail!(
                "{cmd}: variable '{var}' not found in workspace. Provide either 'filename' or 'variable' option. (if a `verbatim` block dynare-rust could not run builds it, save the structure in a MAT file, either named '{var}.mat' next to the .mod file or given with the 'filename' option)"
            ),
        }
    } else {
        let f = &opts.steady_state_file_name;
        let mut p = if Path::new(f).is_absolute() { Path::new(f).to_path_buf() } else { dir.join(f) };
        if p.extension().is_none() {
            let q = p.with_extension("mat");
            if !q.is_file() {
                bail!("{cmd}: Cannot find file: {}", q.display());
            }
            p = q;
        }
        if !p.is_file() {
            bail!("{cmd}: File not found: {}", p.display());
        }
        let mut all = crate::matfile::read(&p)?;
        match all.remove(var) {
            Some(v) => v,
            None => bail!("{cmd}: Variable '{var}' not found in file '{}'.", p.display()),
        }
    };
    SteadyStateInput::from_mat(&v)
}

/// `heterogeneity_load_steady_state`.
pub fn run_load_steady_state(mf: &ModFile, opts: &HetOptions, input: SteadyStateInput) -> Result<LoadedSteadyState> {
    if mf.het.model.is_none() {
        bail!("heterogeneity_load_steady_state: the model has no heterogeneity dimension");
    }
    let cx = LoadContext { model: &mf.model, het: &mf.het, params: &mf.model.params, check: &opts.check };
    load_steady_state(&cx, input, false, true)
}

/// A `SteadyStateData` as a `SteadyStateInput` (what MATLAB passes to the final load).
fn data_as_input(d: &SteadyStateData) -> SteadyStateInput {
    SteadyStateInput {
        agg: Some(d.agg.clone()),
        shocks: Some(ShocksIn { grids: Some(d.shocks_grids.clone()), pi: Some(d.shocks_pi.clone()) }),
        pol: Some(PolIn {
            values: Some(d.pol_values.clone()),
            grids: Some(d.pol_grids.clone()),
            order: Some(d.pol_order.clone()),
        }),
        d: Some(DIn { hist: d.d_hist.clone(), grids: Some(d.d_grids.clone()), order: Some(d.d_order.clone()) }),
        free_parameters: d.free_parameters.clone(),
    }
}

/// Result of `heterogeneity_compute_steady_state`.
pub struct ComputedSteadyState {
    pub steady_state: LoadedSteadyState,
    /// `M_.params` after the calibration.
    pub params: Vec<f64>,
    pub output: CalibrationOutput,
}

/// `heterogeneity_compute_steady_state`: port of `compute_steady_state.m`.
pub fn run_compute_steady_state(
    mf: &ModFile,
    opts: &HetOptions,
    input: SteadyStateInput,
) -> Result<ComputedSteadyState> {
    let model = &mf.model;
    let Some(h) = mf.het.model.as_ref() else {
        bail!("heterogeneity_compute_steady_state: the model has no heterogeneity dimension")
    };
    let free_in = input.free_parameters.clone();
    let cx = LoadContext { model, het: &mf.het, params: &model.params, check: &opts.check };
    let ss = load_steady_state(&cx, input, true, false)?;
    // ---- free parameters (in the order of M_.param_names)
    for (n, _) in &free_in {
        if !model.param_names.contains(n) {
            bail!(
                "Misspecified steady-state input `steady_state.free_parameters`. The following parameters are missing in `M_.param_names`: {n}."
            );
        }
    }
    let mut free: Vec<(usize, String, f64, f64, f64)> = Vec::new();
    for (i, pn) in model.param_names.iter().enumerate() {
        if let Some((_, fp)) = free_in.iter().find(|(n, _)| n == pn) {
            let FreeParameter { initial_guess, lower_bound, upper_bound } = fp.clone();
            let Some(g) = initial_guess else {
                bail!(
                    "Misspecified steady-state input `steady_state`: the `steady_state.free_parameters.{pn}.initial_guess` field is missing."
                )
            };
            let (lo, hi) = (lower_bound.unwrap_or(f64::NEG_INFINITY), upper_bound.unwrap_or(f64::INFINITY));
            if lo.is_finite() && hi.is_finite() && lo > hi {
                bail!(
                    "Misspecified steady-state input: incompatible values for `steady_state.free_parameters.{pn}.lower_bound` and `steady_state.free_parameters.{pn}.upper_bound`."
                );
            }
            free.push((i, pn.clone(), g, lo, hi));
        }
    }
    // ---- target equations (0-based aggregate equations)
    let n_agg_eq = model.orig_endo_nbr;
    let (targets, source): (Vec<usize>, &str) = if !opts.calibration.target_equations.is_empty() {
        let mut t = Vec::new();
        for eq in &opts.calibration.target_equations {
            let idx = match eq {
                TargetEquation::Number(k) => {
                    if *k < 1 || *k > n_agg_eq {
                        bail!("Invalid calibration target equation index {k}. Must be between 1 and {n_agg_eq}.");
                    }
                    *k
                }
                TargetEquation::Name(n) => {
                    let k = model
                        .equation_tags
                        .iter()
                        .position(|tags| tags.iter().any(|(a, b)| a == "name" && b == n))
                        .map(|i| i + 1)
                        .ok_or_else(|| anyhow!("Equation name '{n}' not found in M_.equations_tags."))?;
                    if k > n_agg_eq {
                        bail!("Equation '{n}' (index {k}) is not an aggregate equation (must be <= {n_agg_eq}).");
                    }
                    k
                }
            };
            t.push(idx);
        }
        t.sort_unstable();
        t.dedup();
        (t.into_iter().map(|k| k - 1).collect(), "user-specified")
    } else {
        let n = model.n_endo();
        let cols: Vec<usize> = mf.het.aggregate_aux.iter().map(|a| n + a.endo_index).collect();
        let mut rows: Vec<usize> = model
            .dynamic_jacobian_pattern()
            .iter()
            .filter(|&&(r, c)| cols.contains(&c) && r < n_agg_eq)
            .map(|&(r, _)| r)
            .collect();
        rows.sort_unstable();
        rows.dedup();
        (rows, "auto-detected (SUM operators)")
    };
    if !free.is_empty() && targets.len() != free.len() {
        bail!(
            "Number of calibration target equations ({}) must match number of unknown parameters ({}).\nTarget equations: [{}]\nUnknown parameters: [{}]",
            targets.len(),
            free.len(),
            targets.iter().map(|t| (t + 1).to_string()).collect::<Vec<_>>().join(", "),
            free.iter().map(|f| f.1.clone()).collect::<Vec<_>>().join(", ")
        );
    }
    if !free.is_empty() && opts.calibration.tolf < opts.time_iteration.tol {
        eprintln!(
            "Warning: Calibration tolerance (calibration_tolf={:.2e}) is smaller than time iteration tolerance (time_iteration_tol={:.2e}).\nThe calibration step may attempt to match noise in the inner solver. Consider increasing calibration_tolf or decreasing time_iteration_tol.",
            opts.calibration.tolf, opts.time_iteration.tol
        );
    }
    if opts.calibration.verbosity == 1 && !free.is_empty() {
        println!("\n=== Calibration Target Equations ===");
        for &t in &targets {
            let nm = model
                .equation_tags
                .get(t)
                .and_then(|g| g.iter().find(|(k, _)| k == "name"))
                .map(|x| format!(" ({})", x.1))
                .unwrap_or_default();
            println!("  Equation {:3}{nm}", t + 1);
        }
        println!("Source: {source}\n");
    }
    let mut solver = SteadySolver::new(
        h,
        model,
        &mf.het,
        &ss,
        &model.params,
        opts.time_iteration.clone(),
        opts.forward.clone(),
        opts.calibration.verbosity,
    )?;
    let output = calibrate(&mut solver, &free, &targets, &opts.calibration)?;
    if !output.converged {
        bail!(
            "Steady-state computation failed to converge after {} iterations.\nFinal residual norm: {:.6e}\nTime iteration: {} (iter={}, resid={:.3e})\nDistribution: {} (iter={}, resid={:.3e})",
            output.iterations,
            output.residual_norm,
            output.time_iteration.converged,
            output.time_iteration.iterations,
            output.time_iteration.residual_norm,
            output.distribution.converged,
            output.distribution.iterations,
            output.distribution.residual_norm
        );
    }
    let mut params = model.params.clone();
    if !free.is_empty() {
        for (k, f) in free.iter().enumerate() {
            params[f.0] = output.params[k];
        }
        if opts.calibration.verbosity >= 1 {
            println!("\n=== Calibrated Parameters ===");
            for (k, f) in free.iter().enumerate() {
                println!("  {:<20} = {:12.6}", f.1, output.params[k]);
            }
            println!();
        }
    }
    // ---- back to a steady-state structure
    let mut data = ss.steady_state.clone();
    let mut pol_dims: Vec<usize> = h.exo_names.iter().map(|s| ss.sizes.shock(s)).collect();
    let mut d_dims = pol_dims.clone();
    for s in &ss.indices.base.states {
        pol_dims.push(ss.sizes.pol_state(s));
        d_dims.push(ss.sizes.d_state(s));
    }
    for i in 0..h.orig_endo_nbr {
        let vals: Vec<f64> = (0..ss.sizes.n_sp).map(|p| output.time_iteration.policies[(i, p)]).collect();
        data.set_pol_value(&h.endo_names[i], NdArray::new(pol_dims.clone(), vals));
    }
    data.d_hist = Some(NdArray::new(d_dims, output.distribution.hist.clone()));
    if opts.calibration.verbosity == 1 {
        if free.is_empty() {
            println!(
                "\n=== Heterogeneous Block Computation ===\nAggregate residual norm: {}\n",
                fmt_e(output.residual_norm, 6)
            );
        } else {
            println!(
                "\n=== Steady-State Convergence ===\nOverall: {} after {} calibration iterations\nTarget equations residual norm: {}\n",
                output.converged,
                output.iterations,
                fmt_e(output.residual_norm, 6)
            );
        }
    }
    if !free.is_empty() && output.residual_norm > opts.calibration.tolf {
        bail!(
            "Target equations residual ({}) exceeds tolerance ({}).\nSteady state may not be accurate.",
            fmt_e(output.residual_norm, 6),
            fmt_e(opts.calibration.tolf, 6)
        );
    }
    let mut check = opts.check.clone();
    check.tolf = f64::INFINITY;
    let cx = LoadContext { model, het: &mf.het, params: &params, check: &check };
    let final_ss = load_steady_state(&cx, data_as_input(&data), false, true)?;
    // Dynare 7.2: the aggregate endogenous values are not recomputed, so check that the user gave
    // the right ones (`‖G‖∞ ≤ calibration_tolf`).
    if model.orig_endo_nbr > 0 && aggregate_residuals_too_large(&final_ss.mat.g, opts.calibration.tolf) {
        bail!(
            "heterogeneity_compute_steady_state: residuals too large on aggregate equations. This likely means that you did not give the right value for aggregate endogenous variables (they are not recomputed by this command). Alternatively, if you want to go ahead with those values, you can increase the value of the \"calibration_tolf\" option."
        );
    }
    Ok(ComputedSteadyState { steady_state: final_ss, params, output })
}

/// `norm(G, Inf) > tolf` (the Dynare 7.2 aggregate residual check of `heterogeneity_compute_steady_state`).
fn aggregate_residuals_too_large<'a>(g: impl IntoIterator<Item = &'a f64>, tolf: f64) -> bool {
    g.into_iter().fold(0.0f64, |a, &b| a.max(b.abs())) > tolf
}

/// Session state of the heterogeneity commands (`oo_.heterogeneity`, plus the variables of the
/// MATLAB workspace the commands can read).
#[derive(Default)]
pub struct HetSession {
    pub steady_state: Option<LoadedSteadyState>,
    /// `oo_.heterogeneity.dr` after `heterogeneity_solve`, with its `truncation_horizon`.
    pub dr: Option<super::solve::HetDr>,
    /// `oo_.irfs`, `oo_.endo_simul`, `oo_.exo_simul` of the last `heterogeneity_simulate`.
    pub sim: Option<super::simulate::SimResult>,
    /// Seed of the stochastic simulation draws.
    pub seed: u64,
    pub workspace: HashMap<String, MatValue>,
    /// `options_.heterogeneity` (steady-state, check, forward, time-iteration and calibration
    /// options): persistent, the commands write only the options they are given.
    pub het_options: HetOptions,
    /// `options_.heterogeneity.solve.truncation_horizon` (`None`: the default, 300).
    pub truncation_horizon: Option<usize>,
    /// `(oo_.initial_steady_state or oo_.initial_exo_steady_state non-empty, oo_.initval_series
    /// non-empty)` as the driver found them before the command (`endval` block, `initval_file`):
    /// news-shock simulations refuse both.
    pub pf_start: (bool, bool),
    /// The `options_` fields `heterogeneity_simulate` sets (`irf`, `periods`, `irf_shocks`, `relative_irf`,
    /// `impulse_responses.plot_threshold`, `add_tiny_number_to_cholesky`): the preprocessor writes them into
    /// `options_`, so they persist into the later `heterogeneity_simulate` calls of the run.
    pub sim_options: super::simulate::SimOptions,
}

pub fn handles(name: &str) -> bool {
    matches!(
        name,
        "heterogeneity_load_steady_state"
            | "heterogeneity_compute_steady_state"
            | "heterogeneity_solve"
            | "heterogeneity_simulate"
    )
}

/// Options of a `heterogeneity_simulate` call on top of those of the earlier calls (`prev`); the
/// variable list is the command's own.
pub fn sim_options(opts: &Options, prev: &super::simulate::SimOptions) -> Result<super::simulate::SimOptions> {
    let mut o = super::simulate::SimOptions { var_list: Vec::new(), ..prev.clone() };
    let names = |v: &OptValue| -> Vec<String> {
        match v {
            OptValue::List(l) => l.iter().map(string).collect(),
            other => vec![string(other)],
        }
    };
    for (k, v) in opts {
        match k.as_str() {
            "irf" => o.irf = Some(num(v, k)? as usize),
            "periods" => o.periods = Some(num(v, k)? as usize),
            "irf_shocks" => o.irf_shocks = Some(names(v)),
            "relative_irf" => o.relative_irf = Some(num(v, k)? != 0.0),
            "irf_plot_threshold" => o.irf_plot_threshold = Some(num(v, k)?),
            "add_tiny_number_to_cholesky" => o.add_tiny = Some(num(v, k)?),
            // `print` / `noprint` only set options_.print / noprint: the +heterogeneity code
            // prints nothing (checked: no output differs); `tex` is read at plot time
            "nograph" | "nodisplay" | "graph_format" | "noprint" | "print" | "tex" => {}
            // sets options_.drop, which simulate_stochastic_shocks.m does not read (its check
            // against periods is commented out)
            "drop" => {}
            k if k == crate::modfile::RAW_ARGS_KEY => o.var_list = names(v),
            _ => bail!("heterogeneity_simulate: unknown option {k}"),
        }
    }
    Ok(o)
}

/// The session workspace on top of the variables of the .mod file's `verbatim` blocks.
fn workspace(mf: &ModFile, st: &HetSession) -> HashMap<String, MatValue> {
    let mut w = mf.verbatim_workspace.clone();
    w.extend(st.workspace.iter().map(|(k, v)| (k.clone(), v.clone())));
    w
}

/// Runs a heterogeneity command of the .mod file; may update `mf.model.params`.
pub fn run_command(
    mf: &mut ModFile,
    name: &str,
    opts: &Options,
    dir: &Path,
    fname: &str,
    st: &mut HetSession,
) -> Result<()> {
    match name {
        "heterogeneity_load_steady_state" => {
            st.het_options.apply(opts)?;
            let o = st.het_options.clone();
            let input = read_steady_state(name, &o, dir, &workspace(mf, st))?;
            st.steady_state = Some(run_load_steady_state(mf, &o, input)?);
        }
        "heterogeneity_compute_steady_state" => {
            st.het_options.apply(opts)?;
            let o = st.het_options.clone();
            let input = read_steady_state(name, &o, dir, &workspace(mf, st))?;
            let c = run_compute_steady_state(mf, &o, input)?;
            mf.model.params = c.params;
            st.steady_state = Some(c.steady_state);
        }
        "heterogeneity_solve" => {
            for (k, v) in opts {
                match k.as_str() {
                    "truncation_horizon" => st.truncation_horizon = Some(num(v, k)? as usize),
                    // global display options (`dynare file nograph` puts them on every command)
                    "nograph" | "nodisplay" | "graph_format" | "noprint" => {}
                    _ => bail!("heterogeneity_solve: unknown option {k}"),
                }
            }
            let Some(ss) = st.steady_state.as_ref() else {
                bail!(
                    "heterogeneity_solve: no steady state; run heterogeneity_load_steady_state or heterogeneity_compute_steady_state first"
                );
            };
            let t = st.truncation_horizon.unwrap_or(300);
            st.dr = Some(super::solve::solve_hetero(&mf.model, &mf.het, ss, t)?);
        }
        "heterogeneity_simulate" => {
            let o = sim_options(opts, &st.sim_options)?;
            st.sim_options = o.clone();
            let (Some(dr), Some(ss)) = (st.dr.as_ref(), st.steady_state.as_ref()) else {
                bail!("No solution found. Run heterogeneity_solve first.");
            };
            // simulate_news_shocks.m: the simulation starts from the steady state
            if !mf.det_shocks.is_empty() {
                if mf.histval.iter().any(|h| h.0) {
                    bail!("histval not supported for news shock simulation. Simulation starts from steady state.");
                }
                if st.pf_start.0 {
                    bail!("endval not supported for news shock simulation. Simulation starts from steady state.");
                }
                if st.pf_start.1 {
                    bail!(
                        "initval_series not supported for news shock simulation. Simulation starts from steady state."
                    );
                }
            }
            let res = super::simulate::simulate(
                &mf.model,
                dr,
                &ss.mat.y[mf.model.n_endo()..2 * mf.model.n_endo()],
                &mf.sigma_e,
                &mf.det_shocks,
                &o,
                st.seed,
            )?;
            let gopts = crate::graphics::GraphOptions::from_options(opts, dir, fname);
            if gopts.enabled {
                super::simulate::plot_irfs(&mf.model, &res.irfs, &o, &gopts, fname)?;
                let tex = crate::modfile::opt(opts, "tex").is_some_and(|v| !matches!(v, OptValue::Num(x) if *x == 0.0))
                    && gopts.formats.contains(&crate::graphics::GraphFormat::Eps);
                if tex {
                    let stex: Vec<String> = (0..mf.model.n_exo()).map(|k| crate::latex::exo_tex(mf, k)).collect();
                    if mf.det_shocks.is_empty() {
                        super::tex::write_irf_tex(&mf.model, &stex, &res.irfs, &o, dir, fname)?;
                    }
                    if res.endo_simul.is_some() {
                        let news = !mf.det_shocks.is_empty();
                        super::tex::write_simulation_tex(&mf.model, &o.var_list, news, dir, fname)?;
                    }
                }
                if let Some(d) = res.endo_simul.as_ref() {
                    super::simulate::plot_simulation(
                        &mf.model,
                        d,
                        &o.var_list,
                        !mf.det_shocks.is_empty(),
                        &gopts,
                        fname,
                    )?;
                }
            }
            st.sim = Some(res);
        }
        _ => unreachable!(),
    }
    Ok(())
}

#[cfg(test)]
mod tests_72 {
    use super::aggregate_residuals_too_large;

    #[test]
    fn aggregate_residual_check_uses_inf_norm_vs_calibration_tolf() {
        assert!(!aggregate_residuals_too_large(&[1e-6, -5e-5], 1e-4));
        assert!(aggregate_residuals_too_large(&[1e-6, -2e-4], 1e-4));
        assert!(!aggregate_residuals_too_large(&[], 1e-4));
    }
}
