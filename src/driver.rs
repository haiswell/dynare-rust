// SPDX-License-Identifier: GPL-3.0-or-later
//! Runs the commands of a parsed .mod file, in order, the way Dynare's
//! generated `driver.m` does: `steady` prints the steady state, `estimation`
//! goes through `dynare_estimation_1.m` (initial checks, mode finding, Hessian,
//! mode file, Laplace approximation, Metropolis–Hastings, posterior statistics,
//! smoother).
//!
//! Output files go where Dynare puts them, `<dir of .mod>/<fname>/Output/`:
//! `<fname>_mode.mat` (same variables as Dynare's, loadable as `mode_file` by
//! either program), `<fname>_results.json`, and CSV files for the posterior
//! summary and the smoothed/filtered variables.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;

use crate::analysis::realtime_decomposition as rtd;
use crate::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
use crate::estimation::mh_session;
use crate::estimation::optim::{MinimizeContext, minimize};
use crate::estimation::samplers::{SamplerOptions, load_scale_file, prior_mode_start};
use crate::estimation::{
    GmhmaxlikOptions, MhOptions, MhProgress, Objective, PosteriorOptions, TuneOptions, gmhmaxlik, hessian,
    inverse_hessian, laplace_approximation, mcmc_bounds, mh_tune_jscale, posterior_stats,
};
use crate::forecast::commands as fc; // stream F
use crate::forecast::conditional as cfc; // stream C
use crate::kalman::KalmanOptions;
use crate::linalg::Mat;
use crate::matfile::{self, MatOut};
use crate::modfile::{Command, EstimatedKind, ModFile, OptValue, Options, opt};

/// Settings from the command line.
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    pub defines: Vec<(String, String)>,
    /// RNG seed for the MCMC (Dynare uses MATLAB's stream; results agree statistically only).
    pub seed: u64,
    /// Override `mh_replic` (e.g. to benchmark with fewer draws).
    pub mh_replic: Option<usize>,
    pub quiet: bool,
    /// Pre-fetching width of the MCMC chains and of mh_tune_jscale
    /// ([`MhOptions::prefetch`]; 0 = automatic, 1 = sequential).
    pub prefetch: usize,
    /// [stream X] Dynare's command-line options (`nograph`, `savemacro`, `-I`, ...).
    pub cli: crate::misc::cli::CliOptions,
}

/// Stack of the thread [`run`] works on: the passes over the expression trees are recursive,
/// and a long sum (`a1 + a2 + ... + a20000`) is a deep tree.
const RUN_STACK_SIZE: usize = 1 << 30;

pub fn run(path: &Path, ro: &RunOptions) -> Result<()> {
    std::thread::scope(|s| {
        let h = std::thread::Builder::new()
            .name("dynare-run".into())
            .stack_size(RUN_STACK_SIZE)
            .spawn_scoped(s, || {
                let r = run_on_this_thread(path, ro);
                (r, crate::dataset::export_globals())
            })
            .map_err(|e| anyhow!("cannot start the run thread: {e}"))?;
        match h.join() {
            Ok((r, g)) => {
                // the caller sees the run's `dataset_` (as after `dynare` in MATLAB)
                crate::dataset::import_globals(g);
                r
            }
            Err(p) => std::panic::resume_unwind(p),
        }
    })
}

fn run_on_this_thread(path: &Path, ro: &RunOptions) -> Result<()> {
    let t0 = Instant::now();
    let fname = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("bad file name {}", path.display()))?
        .to_string();
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    println!("Starting dynare-rust (Dynare 7.2 port) on {}", path.display());
    crate::progress::start(crate::progress::file_path(&dir, &fname), &fname);
    let tp = Instant::now();
    // [stream X] savemacro / onlymacro / -I / nograph / onlymodel
    if crate::misc::cli::before_parse(path, ro)? {
        return Ok(());
    }
    if ro.cli.json.is_some() {
        crate::model::dtree::json_capture(); // json=...
    }
    let mut mf = match crate::modfile::parse_mod_file_opts(path, &crate::misc::cli::parse_options(ro)) {
        Ok(mf) => mf,
        Err(e) => {
            for w in crate::modfile::take_error_warnings() {
                println!("{w}"); // the preprocessor prints its warnings before the error
            }
            return Err(e);
        }
    };
    for w in &mf.warnings {
        println!("{w}"); // the preprocessor's output comes first
    }
    if crate::q4::after_parse(&mf, &dir, &fname, ro)? {
        return Ok(()); // onlyjson
    }
    println!(
        "Preprocessing: {} endogenous ({} declared + {} auxiliary), {} shocks{}, {} parameters, {:.0} ms.",
        mf.model.n_endo(),
        mf.model.orig_endo_nbr,
        mf.model.n_endo() - mf.model.orig_endo_nbr,
        mf.model.n_exo(),
        match mf.model.exo_det_names.len() {
            0 => String::new(),
            k => format!(" + {k} varexo_det"),
        },
        mf.model.n_user_params(),
        tp.elapsed().as_secs_f64() * 1e3
    );
    let out_dir = dir.join(&fname).join("Output");
    // ---- stream F: what the estimation leaves in oo_/options_ for later commands ----
    let mut est_state: Option<fc::EstimationState> = None;
    let mut calib_smoothed: Option<crate::misc::SmoothedRecord> = None; // oo_.Smoothed* of calib_smoother
    let mut occbin = crate::occbin::Session::new(&mf, &fname); // OccBin: options_.occbin, oo_.occbin
    let mut mom = crate::mom::Session::new(); // MoM: matched_* blocks, method_of_moments
    // [graphics] chains of the last MCMC, for generate_trace_plots
    let mut mcmc_plots: Option<crate::estimation::plots::McmcPlotContext> = None;
    // stream S: Dynare's options_ persist across commands; the last estimation's point
    // (mode or posterior mean) becomes M_.params for later stoch_simul commands.
    let mut ss_opts = crate::analysis::StochSimulOptions::default();
    // oo_.dr: the decision rules the last stoch_simul / estimation left (read by forecast)
    let mut oo_dr = fc::DecisionRules::Empty;
    let mut steady_params: Option<Vec<f64>> = None;
    // oo_.steady_state as the commands leave it: initval, `steady`, the starting point of an
    // estimation (stoch_simul does not store it); zeros before
    let mut oo_ss: Vec<f64> = vec![0.0; mf.model.n_endo()];
    let mut last_endo_simul: Option<crate::linalg::Mat> = None; // oo_.endo_simul of stoch_simul
    let mut last_exo_simul: Option<crate::linalg::Mat> = None; // oo_.exo_simul of stoch_simul (order 1)
    let ss_ctx = crate::analysis::OutputContext {
        dir: dir.clone(),
        fname: fname.clone(),
        out_dir: out_dir.clone(),
        seed: ro.seed,
    };
    // stream D: oo_ state of deterministic commands (initval/endval/steady/perfect foresight)
    // The shocks / initval / histval blocks run at their position (driver.m): the file's final
    // state is dropped here and each block is replayed before the first command that follows it.
    // An `initval` block after a command may read the steady state that command left, so it
    // needs the deterministic state too.
    let uses_det = crate::deterministic::uses_det_state(&mf)
        || mf.state_blocks.iter().any(|(k, b)| {
            *k > 0 && matches!(b, crate::modfile::StateBlock::Initval { sources, .. } if !sources.is_empty())
        });
    let shock_paths_all = std::mem::take(&mut mf.q1.shock_paths); // revealed at their position
    let controlled_paths_all = std::mem::take(&mut mf.ep.controlled_paths);
    let mut next_cp = 0usize;
    mf.reset_state_blocks();
    mf.options.active = true;
    let mut next_block = 0usize;
    let mut det = if uses_det { Some(crate::deterministic::DetState::new(&mf)?) } else { None };
    if let Some(d) = det.as_mut() {
        d.q1_shock_paths = !shock_paths_all.is_empty();
    }
    let mut bvar_session = crate::bvar::Session::default(); // stream B
    let mut policy = crate::policy::Session::new(ro.quiet); // [stream P]
    policy.set_output(ss_ctx.clone()); // stoch_simul output of the policy commands
    let mut sc = rtd::DecompositionState::default(); // stream C: oo_ of the decompositions / conditional forecasts
    let mut xs = crate::misc::Session::default(); // [stream X]
    let mut reporting = crate::reporting::Session::default(); // report() objects, IRF dseries
    let mut q4 = crate::q4::Session::start(); // options_.dataset, options_.initial_period, ...
    let mut ro_x = ro.clone(); // [stream X] set_dynare_seed changes the seed
    let mut het_session = crate::heterogeneity::commands::HetSession { seed: ro.seed, ..Default::default() }; // ---- heterogeneity ----
    let ncommands = if ro.cli.onlymodel { 0 } else { mf.commands.len() }; // [stream X] onlymodel
    for ci in 0..ncommands {
        mf.options.write(&mf.commands[ci]); // the options this command writes into options_
        if matches!(mf.commands[ci], Command::Estimation(..)) && mf.options.slice_prior_trunc_raised {
            println!("\ndynare_estimation_init: slice requires bounded support. Setting options_.prior_trunc=1e-10.");
        }
        // the shocks / initval / histval / perfect_foresight_controlled_paths blocks written
        // before this command (driver.m runs them in file order)
        for b in mf.run_state_blocks(ci, &mut next_block, &mut oo_ss)? {
            apply_initval_to_det(&mut mf, det.as_mut(), &b)?;
        }
        mf.resolve_learnt(ci, &oo_ss)?; // learnt_in values with endogenous variables
        while next_cp < controlled_paths_all.len() && mf.controlled_paths_pos[next_cp] <= ci {
            mf.ep.controlled_paths.push(controlled_paths_all[next_cp].clone());
            next_cp += 1;
        }
        if let Command::Deterministic(n, o, _) = &mf.commands[ci]
            && n == "shock_paths"
            && let Some((_, OptValue::Num(k))) = o.iter().find(|(k, _)| k == "__index")
        {
            // M_.shock_paths grows at the block's position; `overwrite` drops the earlier blocks
            // (and controlled paths) with the same learnt_in
            let mut b = shock_paths_all[*k as usize].clone();
            b.active = true;
            if b.overwrite {
                for x in mf.q1.shock_paths.iter_mut() {
                    if x.same_learnt_in(b.learnt_in, b.learnt_in_date) {
                        x.active = false;
                    }
                }
                mf.ep.controlled_paths.retain(|c| match (c.learnt_in_date, b.learnt_in_date) {
                    (Some(a), Some(d)) => a != d,
                    (None, None) => c.learnt_in != b.learnt_in,
                    _ => true,
                });
            }
            let newk = mf.q1.shock_paths.len();
            mf.q1.shock_paths.push(b);
            if let Command::Deterministic(_, o, _) = &mut mf.commands[ci] {
                o.retain(|(k, _)| k != "__index");
                o.push(("__index".into(), OptValue::Num(newk as f64)));
            }
        }
        // estim_params_ as set by the last estimated_params block before this command
        if mf.estimated_params_history.len() > 1
            && let Some((_, e)) = mf.estimated_params_history.iter().rev().find(|(k, _)| *k <= ci)
        {
            mf.estimated_params = e.clone();
        }
        // ---- heterogeneity: heterogeneity_* commands (may change M_.params) ----
        if let Command::Other(name, o) = &mf.commands[ci]
            && crate::heterogeneity::commands::handles(name)
        {
            let (name, o) = (name.clone(), o.clone());
            het_session.pf_start = det.as_ref().map_or((false, false), |d| {
                (d.initial_steady_state.is_some() || d.initial_exo_steady_state.is_some(), d.initval_series.is_some())
            });
            crate::heterogeneity::commands::run_command(&mut mf, &name, &o, &dir, &fname, &mut het_session)?;
            if let Some(d) = det.as_mut() {
                d.params[..mf.model.params.len()].copy_from_slice(&mf.model.params);
            }
            continue;
        }
        // ---- end heterogeneity ----
        // ---- [stream X] misc commands, some of which change M_ (parameter assignments) ----
        // ---- var models: var_expectation.initialize / update, get_companion_matrix ----
        if let Command::Other(name, o) = &mf.commands[ci]
            && crate::varmodels::handles(name)
        {
            let (name, mut o) = (name.clone(), o.clone());
            o.push(("dir".into(), crate::modfile::OptValue::Str(dir.to_string_lossy().into_owned())));
            o.push(("out_dir".into(), crate::modfile::OptValue::Str(out_dir.to_string_lossy().into_owned())));
            o.push(("fname".into(), crate::modfile::OptValue::Str(fname.to_string())));
            crate::varmodels::run(&mut mf, &name, &o)?;
            if let Some(d) = det.as_mut() {
                d.params[..mf.model.params.len()].copy_from_slice(&mf.model.params);
            }
            continue;
        }
        if let Command::Other(name, o) = &mf.commands[ci]
            && crate::misc::handles(name)
        {
            let (name, o) = (name.clone(), o.clone());
            let ctx = crate::misc::Ctx {
                dir: &dir,
                fname: &fname,
                out_dir: &out_dir,
                ci,
                est_state: est_state.as_ref(),
                mcmc_plots: mcmc_plots.as_ref(),
                calib_smoothed: calib_smoothed.as_ref(),
                det_steady: det.as_ref().map(|d| (d.steady_state.as_slice(), d.exo_steady_state.as_slice())),
            };
            xs.run(&mut mf, &name, &o, &mut ro_x.seed, &ctx)?;
            if name == crate::modfile::SET_PARAM
                && let Some(d) = det.as_mut()
            {
                d.params[..mf.model.params.len()].copy_from_slice(&mf.model.params); // DetState's M_.params copy
            }
            continue;
        }
        // options_.homotopy_mode & co. are global: a later `steady` repeats the homotopy
        if matches!(mf.commands[ci], Command::Steady(_)) && crate::misc::homotopy::requested(&mf.options.steady) {
            let o = mf.options.steady.clone();
            crate::misc::homotopy::before_steady(&mut mf, det.as_mut(), &o)?; // steady(homotopy_mode=...)
        }
        if matches!(mf.commands[ci], Command::Estimation(..)) {
            // dynare_estimation_init.m computes the steady state at the initial point
            if let Ok(x0) = crate::dsge::initial_vector(&mf)
                && let Ok(dm) = DsgeModel::new(&mf, Some(&x0))
                && let Ok(ys) = crate::analysis::calibrated_steady_state(&mf, &dm.set_all_parameters_full(&x0).0)
            {
                oo_ss = ys;
            }
        }
        // an estimation sees the options earlier estimations left in options_
        let mut new_globals: Option<crate::dsge::GlobalState> = None; // M_ written by an estimation
        let est_opts: Option<Options> =
            matches!(mf.commands[ci], Command::Estimation(..)).then(|| mf.options.estimation_options());
        let (c, ro) = (&mf.commands[ci], &ro_x);
        crate::deterministic::exo_det::command_check(&mf, c)?; // varexo_det errors
        // ---- [end stream X] ----
        // [stream P] optimal policy commands (and stoch_simul on Ramsey models)
        if crate::policy::handles(&mf, c) {
            policy.ss_opts = ss_opts.clone(); // options_ is shared with stoch_simul / estimation
            policy.run(&mf, c, Some((&out_dir, &fname)))?;
            ss_opts = policy.ss_opts.clone();
            if let Some(y) = policy.endo_simul.take() {
                last_endo_simul = Some(y); // oo_.endo_simul (datatomfile)
            }
            if let Some(p) = policy.take_params() {
                mf.model.params = p; // osr: M_.params with the optimized values
                if let Some(d) = det.as_mut() {
                    d.params[..mf.model.params.len()].copy_from_slice(&mf.model.params);
                }
            }
            oo_dr = fc::DecisionRules::Computed { params: Some(mf.model.params.clone()), point: None };
            continue;
        }
        match c {
            // global sensitivity analysis (src/gsa)
            Command::Other(name, o) if name == "sensitivity" || name == "dynare_sensitivity" => {
                crate::gsa::run(&mf, &dir, &fname, o, ci)?;
            }
            Command::Steady(o) if det.is_some() => {
                let d = det.as_mut().unwrap();
                d.eq_names = crate::misc::equation_names(&mf);
                d.steady(&mf.model, o)?;
                // steady.m: M_.params assigned by the steady_state_model block
                let n = mf.model.params.len();
                if d.params[..n].iter().zip(&mf.model.params).any(|(a, b)| a != b && !(a.is_nan() && b.is_nan())) {
                    steady_params = Some(d.params[..n].to_vec());
                }
            }
            Command::Endval(v) => {
                let d = det.as_mut().expect("endval implies DetState");
                d.endval(&mf.model, v)?;
                let d = det.as_ref().unwrap();
                mf.initval_exo = d.exo_steady_state.iter().copied().enumerate().filter(|&(_, v)| v != 0.0).collect();
            }
            Command::Deterministic(name, o, vars) => {
                let d = det.as_mut().expect("deterministic command implies DetState");
                // rplot plots the latest oo_.endo_simul / oo_.exo_simul, also those of stoch_simul
                if name == "rplot"
                    && let Some(e) = &last_endo_simul
                {
                    d.endo_simul = Some(e.clone());
                    if let Some(x) = &last_exo_simul {
                        d.exo_simul = Some(x.clone());
                    }
                }
                // rplot after dsample (options_.smpl); the epilogue over the simulation
                if !crate::q4::rplot_dsample(&mf, name, o, vars, d, &dir, &fname, xs.dsample)? {
                    d.command(&mf, name, o, vars, &dir, &fname)?;
                }
                crate::q4::after_deterministic(&mf, name, d, &out_dir, &fname)?;
                if let Some(e) = &d.endo_simul {
                    last_endo_simul = Some(e.clone()); // oo_.endo_simul of the perfect foresight solver
                    last_exo_simul = d.exo_simul.clone();
                }
            }
            // ---- stream B: BVAR, model comparison, LaTeX
            Command::Other(name, _) if crate::msbvar::handles(name) => {
                crate::msbvar::run_command(&mf, c, &dir, &fname, ro.seed)?
            }
            Command::Other(name, _) if crate::bvar::handles(name) => {
                crate::bvar::run_command(&mf, c, &dir, &fname, ro.seed, &mut bvar_session)?
            }
            Command::Other(name, _) if name == "model_comparison" => {
                crate::estimation::model_comparison::run_command(&mf, c, &dir, &fname).map(|_| ())?
            }
            Command::Other(name, _) if crate::latex::handles(name) => crate::latex::run_command(&mf, c, &dir, &fname)?,
            // ---- end stream B
            Command::Steady(_) => steady_params = print_steady_state(&mf)?,
            // ---- stream PF: order >= 2 -> nonlinear filters (non_linear_dsge_likelihood) ----
            Command::Estimation(..) if dir.join(format!("{fname}_prior_restrictions.m")).exists() => {
                // dynare_estimation_init.m: options_.prior_restrictions.routine, called by
                // dsge_likelihood.m to add its value to the log prior
                bail!(
                    "{}: user-defined prior restrictions are MATLAB code and cannot be executed by dynare-rust \
                     (Dynare adds the value this function returns to the log prior density); \
                     remove or rename the file to estimate without them",
                    dir.join(format!("{fname}_prior_restrictions.m")).display()
                )
            }
            Command::Estimation(..) if crate::nonlinear::requested_order(est_opts.as_ref().unwrap()) >= 2 => {
                // options_.qz_criterium left by an earlier stoch_simul / estimation stays in
                // options_ (select_qz_criterium_value.m reads it, nonlinear branch included)
                let mut opts_qz = est_opts.as_ref().unwrap().clone();
                if !opts_qz.iter().any(|(k, _)| k == "qz_criterium") && ss_opts.qz_criterium != 1.0 + 1e-6 {
                    opts_qz.push(("qz_criterium".into(), crate::modfile::OptValue::Num(ss_opts.qz_criterium)));
                }
                let opts = &opts_qz;
                // ---- estimation completion: analytic_derivation at order > 1 (no gradient)
                crate::estimation::analytic::check_options(
                    opts,
                    0,
                    1,
                    false,
                    crate::nonlinear::requested_order(opts) as i32,
                )?;
                let point = crate::nonlinear::estimation::run(&mf, &dir, &fname, &out_dir, opts, ro)?;
                // dynare_estimation_1.m:587: M_ = set_all_parameters(xparam1, ...)
                let mut dm = DsgeModel::new(&mf, Some(&point))?;
                dm.adopt_point(&point);
                new_globals = Some(dm.global_state());
                oo_dr = fc::DecisionRules::Computed { params: None, point: Some(point) };
            }
            Command::Estimation(_, vars) => {
                let opts = est_opts.as_ref().unwrap();
                // the preprocessor writes options_.order = 1 when estimation has no order option
                if !opts.iter().any(|(k, _)| k == "order") {
                    ss_opts.order = 1;
                }
                ss_opts.apply_checked(opts)?;
                // dynare_estimation_init.m: options_.qz_criterium stays in options_
                let lik_init = if flag(opts, "diffuse_filter") { 3.0 } else { num(opts, "lik_init").unwrap_or(1.0) };
                ss_opts.qz_criterium = select_qz_criterium_value(ss_opts.qz_criterium, lik_init)?;
                let (mut st, ctx, point) = match crate::estimation::recursive::Recursive::new(opts, &fname)? {
                    None => estimation(&mf, &dir, &fname, &out_dir, opts, vars, ro, &ss_opts, &mut occbin)?,
                    Some(rec) => {
                        recursive_estimation(&rec, &mf, &dir, &fname, &out_dir, opts, vars, ro, &ss_opts, &mut occbin)?
                    }
                };
                // dynare_estimation_1.m:584-587: the mode / posterior mean becomes M_.params, M_.Sigma_e, M_.H
                st.dsge.adopt_point(&point);
                new_globals = Some(st.dsge.global_state());
                oo_dr = fc::DecisionRules::Computed { params: None, point: Some(point) };
                est_state = Some(st);
                calib_smoothed = None; // the estimation's smoother overwrites oo_.SmoothedVariables
                if ctx.is_some() {
                    mcmc_plots = ctx;
                }
            }
            // ---- stream F ----
            Command::Forecast(o, v) => {
                if let Some(p) = num(o, "periods") {
                    ss_opts.periods = p as usize; // options_.periods is global
                }
                let mut fo = fc::ForecastOrder::of(&mf, &ss_opts, ro.seed);
                fo.dr = oo_dr.clone();
                if let Some(c) = mf.options.forecast_conf_sig {
                    fo.conf_sig = c; // options_.forecasts.conf_sig
                }
                sc.forecast = Some(fc::forecast_command(&mf, est_state.as_ref(), o, v, &dir, &fname, &out_dir, &fo)?) // + stream C
            }
            Command::ShockDecomposition(o, v) => {
                // stream C: dispatches to stream F's command or the forecast_type branch, keeps oo_
                // options_.datafile of an earlier command, when no estimation supplies the data
                let o = &if est_state.is_none() { mf.options.with_data_options(o) } else { o.clone() };
                rtd::shock_decomposition_command(&mf, est_state.as_ref(), o, v, &mut sc, &dir, &fname, &out_dir)?
            }
            Command::CalibSmoother(o, v) if !mf.occbin_constraints.is_empty() => {
                // stream OE: occbin.DSGE_smoother at the calibration
                crate::occbin::estimation::calib_smoother_command(
                    &mf,
                    est_state.as_ref(),
                    &mut occbin,
                    o,
                    v,
                    &dir,
                    &fname,
                    &out_dir,
                )?
            }
            Command::CalibSmoother(o, v) => {
                calib_smoothed =
                    Some(fc::calib_smoother_command(&mf, est_state.as_ref(), o, v, &dir, &fname, &out_dir)?);
            }
            Command::ShockGroups(..) => {}
            // ---- end stream F ----
            // [graphics] generate_trace_plots(chains)
            Command::Other(name, opts) if name == "generate_trace_plots" => match &mcmc_plots {
                Some(ctx) => {
                    let raw = opts.first().map(|(k, _)| k.as_str()).unwrap_or("");
                    let n = crate::estimation::plots::generate_trace_plots(
                        ctx,
                        &crate::estimation::plots::parse_chain_numbers(raw),
                    )?;
                    println!("generate_trace_plots: {n} trace plots written to {}.", ctx.gopts.dir.display());
                }
                None => println!(
                    "WARNING: generate_trace_plots: no MCMC draws in memory (run an estimation with mh_replic > 0 first); skipped."
                ),
            },
            Command::Other(name, o) if name == "datatomfile" => {
                let y = last_endo_simul
                    .as_ref()
                    .ok_or_else(|| anyhow!("datatomfile: oo_.endo_simul is empty. Check why the simulation failed."))?;
                crate::misc::datatomfile(&mf, y, o, &dir, &fname)?;
            }
            Command::Check(o) => {
                crate::analysis::run_check_det(&mf, o, None, det.as_ref())?;
            }
            Command::StochSimul(o, vars) => {
                // ---- stoch_simul: the preprocessor writes options_.order = 2 when the command
                // has no `order` option (even after estimation) ----
                if !o.iter().any(|(k, _)| k == "order") {
                    ss_opts.order = 2;
                }
                let unsupported = ss_opts.apply_checked(o)?;
                for k in unsupported {
                    println!("WARNING: stoch_simul option `{k}` is not supported by dynare-rust and is ignored.");
                }
                // the k-order solver fails on varexo_det (order 3, k_order_solver)
                crate::deterministic::exo_det::k_order_check(&mf.model, ss_opts.order, ss_opts.k_order_solver)?;
                let pi = ss_opts.partial_information; // forces order 1 (stoch_simul.m)
                let order1 = mf.model.linear || pi || mf.model.dynamic_derivatives(2).hessian_eq_zero();
                // ---- end stoch_simul ----
                if ss_opts.order >= 2 && !order1 {
                    // ---- stream K: stoch_simul at order >= 2 (Dynare's default order is 2);
                    // with the persistent options_ ----
                    let r = crate::solver::korder::stoch_simul::run_q2(&mf, &ss_opts, vars, &ss_ctx);
                    if let Some(y) = noprint_info(r, ss_opts.noprint)?.flatten() {
                        last_endo_simul = Some(y); // oo_.endo_simul (datatomfile)
                        last_exo_simul = None;
                    }
                } else {
                    // ---- stoch_simul: order 1 (model(linear), Hessian equal to zero, partial_information)
                    let mut so = ss_opts.clone();
                    if !pi && so.order != 1 {
                        if !mf.model.linear {
                            println!("Warning: stoch_simul: using order = 1 because Hessian is equal to zero");
                        }
                        so.order = 1;
                    }
                    let r = crate::analysis::run_stoch_simul(&mf, &so, vars, &ss_ctx, None);
                    if let Some(r) = noprint_info(r, so.noprint)?.flatten()
                        && r.endo_simul.is_some()
                    {
                        last_endo_simul = r.endo_simul;
                        last_exo_simul = r.exo_simul;
                    }
                }
                ss_opts.partial_information = false; // stoch_simul.m: "temporary fix waiting for local options"
                // oo_.dr at the M_ in force (after an estimation M_ holds the estimated point)
                oo_dr = fc::DecisionRules::Computed {
                    params: Some(mf.model.params.clone()),
                    point: est_state
                        .as_ref()
                        .and_then(|_| fc::calibration_vector(&DsgeModel::new(&mf, None).ok()?).ok()),
                };
            }
            Command::Mom(c) => {
                // ---- calibration restrictions: default_option_mom_values.m drops them ----
                if matches!(c, crate::mom::MomCommand::MethodOfMoments(_)) {
                    crate::gsa::calibration::mom_warning(&mf, ci);
                }
                mom.command(&mf, c, &dir, &fname, None, ro_x.seed)? // MoM
            }
            Command::Occbin(oc) => {
                occbin.command(&mf, oc, &out_dir, &fname, &crate::graphics::graph_dir(&dir, &fname))?
            }
            Command::Policy(..) => {} // [stream P] handled by policy::handles above
            // ---- stream C ----
            Command::ConditionalForecastPaths(p) => {
                sc.cond_paths =
                    Some(cfc::ConstrainedPaths::from_block(&mf, p, &|| Ok((oo_ss.clone(), mf.exo_steady_state())))?)
            }
            Command::ConditionalForecast(o) => {
                sc.conditional_forecast = Some(cfc::conditional_forecast_command(
                    &mf,
                    est_state.as_ref(),
                    o,
                    sc.cond_paths.as_ref(),
                    ro.seed,
                    &out_dir,
                    &fname,
                )?)
            }
            Command::PlotConditionalForecast(o, v) => cfc::plot_conditional_forecast_command(
                &mf,
                est_state.as_ref(),
                sc.conditional_forecast.as_ref(),
                o,
                v,
                &dir,
                &fname,
            )?,
            Command::RealtimeShockDecomposition(o, v) => rtd::realtime_shock_decomposition_command(
                &mf,
                est_state.as_ref(),
                o,
                v,
                &mut sc,
                &dir,
                &fname,
                &out_dir,
            )?,
            Command::PlotShockDecomposition(o, v) => {
                rtd::plot_shock_decomposition_command(&mf, est_state.as_ref(), o, v, &mut sc, &dir, &fname, &out_dir)?
            }
            Command::InitialConditionDecomposition(o, v) => {
                // options_.datafile of an earlier command, as for shock_decomposition
                let o = &if est_state.is_none() { mf.options.with_data_options(o) } else { o.clone() };
                rtd::initial_condition_decomposition_command(
                    &mf,
                    est_state.as_ref(),
                    o,
                    v,
                    &mut sc,
                    &dir,
                    &fname,
                    &out_dir,
                )?
            }
            Command::SqueezeShockDecomposition(v) => rtd::squeeze_shock_decomposition(&mf, v, &mut sc)?,
            // ---- end stream C ----
            // stream I: identification
            Command::Other(name, o) if name == "identification" => {
                // parameter_set = posterior_* uses the previous estimation
                let post = est_state.as_ref().map(|e| crate::identification::run::PosteriorPoints {
                    mode: e.mode.clone(),
                    mean: e.posterior_mean.clone(),
                    median: e.posterior_median.clone(),
                });
                crate::identification::run::run_q2(&mf, &dir, &fname, o, ro.seed, post, ss_opts.drop)?
            }
            // stream EP: extended path, perfect foresight with expectation errors
            Command::Other(name, _) if crate::ep::handles(name) => {
                crate::ep::command(&mf, det.as_mut().expect("EP commands imply DetState"), ci, &dir, &fname, ro.seed)?
            }
            // ---- data, set_time, epilogue, ...
            Command::Other(name, o) if crate::q4::handles(name) => {
                crate::q4::command(&mf, name, o, &dir, &fname, &out_dir, &mut q4)?
            }
            // options_.endogenous_prior_restrictions (read by sensitivity and prior)
            // (read by estimation, prior, sensitivity: see gsa::calibration; Dynare prints nothing)
            Command::Other(name, o) if name == crate::reporting::REPORT_STATEMENT => {
                reporting.run(o, &dir, &fname, &out_dir)?
            }
            Command::Other(name, _) if name == "irf_calibration" || name == "moment_calibration" => {}
            Command::Other(name, _) => println!("WARNING: command `{name}` is not supported by dynare-rust; skipped."),
        }
        if matches!(mf.commands[ci], Command::CalibSmoother(..)) {
            // DsgeSmoother.m: dynare_resolve stores oo_.dr at the calibration
            oo_dr = fc::DecisionRules::Computed { params: Some(mf.model.params.clone()), point: None };
            // the driver writes options_.parameter_set = 'calibration' before
            // evaluate_smoother, and it stays for the later decompositions
            sc.parameter_set = Some("calibration".into());
        }
        if matches!(mf.commands[ci], Command::Steady(_)) {
            mf.options.steady_ran = true; // oo_.steady_state holds a computed steady state
            match det.as_ref() {
                Some(d) if d.steady_state.len() >= oo_ss.len() => oo_ss = d.steady_state[..oo_ss.len()].to_vec(),
                _ => {
                    if let Ok(ys) = crate::analysis::calibrated_steady_state(&mf, &mf.model.params) {
                        oo_ss = ys;
                    }
                }
            }
        }
        if let Some(g) = new_globals.take() {
            g.write(&mut mf); // M_ after the estimation (set_all_parameters)
            if let Some(d) = det.as_mut() {
                d.params[..mf.model.params.len()].copy_from_slice(&mf.model.params);
            }
        }
        if let Some(p) = steady_params.take() {
            mf.model.params = p; // steady.m: M_.params from the steady_state_model block
        }
    }
    {
        // the generated driver: fprintf("Total computing time : %dh%02dm%02ds\n", ...)
        let s = t0.elapsed().as_secs();
        println!("Total computing time : {}h{:02}m{:02}s", s / 3600, (s / 60) % 60, s % 60);
    }
    if !mf.warnings.is_empty() {
        println!("Note: {} warning(s) encountered in the preprocessor", mf.warnings.len());
    }
    crate::progress::finish(true, "");
    Ok(())
}

/// `select_qz_criterium_value.m` (linear filter): `current` is `options_.qz_criterium`, where the
/// default 1+1e-6 stands for the empty value. With `lik_init = 1` the default becomes 1-1e-6
/// (unit roots are explosive) and a larger value is an error; otherwise a value <= 1 is reset
/// to 1+1e-6.
fn select_qz_criterium_value(current: f64, lik_init: f64) -> Result<f64> {
    let empty = current == 1.0 + 1e-6;
    if lik_init == 1.0 {
        if empty {
            return Ok(1.0 - 1e-6);
        }
        if current > 1.0 - f64::EPSILON {
            bail!(
                "dynare_estimation_init: option qz_criterium is too large for estimating/smoothing a stationary model. If your model contains unit roots, use option diffuse_filter"
            );
        }
        Ok(current)
    } else if current <= 1.0 {
        println!(
            "
dynare_estimation_init:: diffuse filter is incompatible with a qz_criterium<=1. Resetting it to 1+1e-6."
        );
        Ok(1.0 + 1e-6)
    } else {
        Ok(current)
    }
}

/// An `initval` block (InitValStatement) on the deterministic state: the assignments
/// `oo_.steady_state(i) = expr` / `oo_.exo_steady_state(k) = expr` run in order on the current
/// values (the expressions may read the steady state an earlier command left), then the
/// auxiliary variables are set. The exogenous values are also what the stochastic commands read.
fn apply_initval_to_det(
    mf: &mut ModFile,
    det: Option<&mut crate::deterministic::DetState>,
    b: &crate::modfile::StateBlock,
) -> Result<()> {
    let Some(d) = det else { return Ok(()) };
    let crate::modfile::StateBlock::Initval { endo, exo, sources } = b else { return Ok(()) };
    let m = &mf.model;
    if sources.is_empty() {
        for &(i, v) in endo {
            d.steady_state[i] = v;
        }
        for &(k, v) in exo {
            d.exo_steady_state[k] = v;
        }
    }
    for (name, src) in sources {
        let e = m.compile_expr(src).map_err(|e| anyhow!("initval: {name} = {src}: {e}"))?;
        let v = e.eval(&d.steady_state, &d.exo_steady_state, &d.params);
        if let Some(j) = m.endo_index(name) {
            d.steady_state[j] = v;
        } else if let Some(k) = m.exo_index(name) {
            d.exo_steady_state[k] = v;
        }
    }
    m.set_auxiliary_variables(&mut d.steady_state, &d.exo_steady_state, &d.params);
    sync_exo_steady_state(mf, d);
    Ok(())
}

/// `oo_.exo_steady_state` of the deterministic state, as the stochastic commands read it.
fn sync_exo_steady_state(mf: &mut ModFile, d: &crate::deterministic::DetState) {
    mf.initval_exo = d.exo_steady_state.iter().copied().enumerate().filter(|&(_, v)| v != 0.0).collect();
}

/// The `steady` command of a stochastic model. Returns `M_.params` when the
/// `steady_state_model` block assigns parameters (steady.m stores the updated vector), so that
/// the later commands (`check`, `method_of_moments`, ...) see them; `None` otherwise.
fn print_steady_state(mf: &ModFile) -> Result<Option<Vec<f64>>> {
    // MoM: the calibrated steady state only (DsgeModel::new needs priors for every estimated
    // parameter, which frequentist method_of_moments runs do not have)
    let ys = crate::analysis::calibrated_steady_state(mf, &mf.model.params).context("steady")?;
    crate::deterministic::print_steady_state(&mf.model, &ys); // disp_steady_state.m (stream R)
    let Some(ssm) = crate::deterministic::SteadyStateModel::from_modfile(mf)? else { return Ok(None) };
    let x = mf.exo_steady_state();
    let (_, p) = crate::deterministic::evaluate_steady_state(
        &mf.model,
        Some(&ssm),
        &ys,
        &x,
        &mf.model.params,
        &crate::dsge::steady_options(mf),
        false,
    )
    .map_err(|e| anyhow!("steady: info {}", e.info))?;
    let same = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());
    Ok((!p.iter().zip(&mf.model.params).all(|(&a, &b)| same(a, b))).then_some(p))
}

// ---------------------------------------------------------------------------
// option helpers
// ---------------------------------------------------------------------------

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

fn flag(o: &Options, k: &str) -> bool {
    opt(o, k).is_some()
}

fn string(o: &Options, k: &str) -> Option<String> {
    match opt(o, k) {
        Some(OptValue::Str(s)) => Some(s.clone()),
        _ => None,
    }
}

/// `additional_optimizer_steps = [6]` / `= 6` / `= [4, 6]`.
fn optimizer_steps(o: &Options) -> Result<Vec<i32>> {
    match opt(o, "additional_optimizer_steps") {
        None => Ok(vec![]),
        Some(OptValue::Num(v)) => Ok(vec![*v as i32]),
        Some(OptValue::Vec(v)) => Ok(v.iter().map(|&x| x as i32).collect()),
        Some(OptValue::List(items)) => items
            .iter()
            .map(|it| match it {
                OptValue::Num(v) => Ok(*v as i32),
                _ => Err(anyhow!("additional_optimizer_steps: only numeric optimizers are supported")),
            })
            .collect(),
        Some(_) => bail!("additional_optimizer_steps: expected a vector of mode_compute values"),
    }
}

/// Options that only control figures/console output in Dynare; accepted silently.
const DISPLAY_ONLY: &[&str] = &[
    "graph_format",
    "nodisplay",
    "nograph",
    "plot_priors",
    "tex",
    "nointeractive",
    "console",
    "graph",
    "noprint",
    // [graphics]
    "mode_check_neighbourhood_size",
    "mode_check_symmetric_plots",
    "mode_check_number_of_points",
    "nodiagnostic",
];

const SUPPORTED: &[&str] = &[
    "datafile",
    "first_obs",
    "recursive_estimation_restart",
    "nobs",
    "presample",
    "diffuse_filter",
    "lik_init",
    "kalman_algo",
    "kalman_tol",
    "diffuse_kalman_tol",
    "riccati_tol",
    "qz_criterium",
    "prior_trunc",
    "mode_compute",
    "mode_file",
    "mh_replic",
    "mh_nblocks",
    "mh_jscale",
    "mh_tune_jscale",
    "mh_drop",
    "mh_conf_sig",
    "mh_init_scale_factor",
    "mh_init_scale",
    "smoother",
    "filtered_vars",
    "smoothed_state_uncertainty",
    "forecast",
    "mode_check",
    "optim",
    "use_univariate_filters_if_singularity_is_detected",
    "conf_sig",
    "additional_optimizer_steps",
    "mh_tune_guess",
    // stream S
    "bayesian_irf",
    "moments_varendo",
    "sub_draws",
    "irf",
    "irf_shocks",
    "relative_irf",
    "ar",
    "nodecomposition",
    "conditional_variance_decomposition",
    "irf_plot_threshold",
    "xls_sheet",
    "xls_range",
    "prefilter",
    "loglinear",
    "logdata",
    "endogenous_prior",
    "posterior_sampling_method",
    "posterior_sampler_options",
    // `options_.solve_tolf/solve_tolx = V;` before the command (GlobalOptions::write: the steady state)
    "solve_tolf",
    "solve_tolx",
    // `options_.gradient_epsilon/gradient_method = V;` (the optimizers' numerical gradients)
    "gradient_epsilon",
    "gradient_method",
];

// ---------------------------------------------------------------------------
// estimation
// ---------------------------------------------------------------------------

fn estimation(
    mf: &ModFile,
    dir: &Path,
    fname: &str,
    out_dir: &Path,
    o: &Options,
    var_list: &[String],
    ro: &RunOptions,
    ss_opts: &crate::analysis::StochSimulOptions,
    occbin: &mut crate::occbin::Session,
) -> Result<(fc::EstimationState, Option<crate::estimation::plots::McmcPlotContext>, Vec<f64>)> {
    let mut mcmc_plots = None;
    // ---- estimation completion: estimate_initial_states_endogenous_prior (dynare_estimation_init.m)
    let mf_is;
    let mf = if crate::estimation::est_completion::check_initial_states(o)? {
        mf_is = crate::estimation::init_states::augment(mf)?;
        &mf_is
    } else {
        mf
    };
    // dirname: M_.dname (metropolis/ and Output/ folders)
    let (dname, out_dir_q3) = crate::estimation::q3::dname(dir, fname, o, out_dir);
    let out_dir = out_dir_q3.as_path();
    // check_list_of_variables.m: consider_all_endogenous(_and_auxiliary), consider_only_observed
    let var_list_q3 = crate::estimation::q3::var_list(mf, o, var_list)?;
    let var_list = &var_list_q3[..];
    for (k, _) in o {
        // `order=1` is the linear estimation (the default); higher orders are run elsewhere
        if k == "order" {
            continue;
        }
        if !SUPPORTED.contains(&k.as_str()) && !DISPLAY_ONLY.contains(&k.as_str()) && !mh_session::OPTIONS.contains(&k.as_str())
            && !crate::estimation::q3::OPTIONS.contains(&k.as_str())
            && !crate::estimation::est_completion::OPTIONS.contains(&k.as_str()) // ---- estimation completion
            && !crate::estimation::analytic::OPTIONS.contains(&k.as_str())
        // ---- estimation completion
        {
            println!("WARNING: estimation option `{k}` is not supported by dynare-rust and is ignored.");
        }
    }
    let uses_gmhmaxlik = num(o, "mode_compute") == Some(6.0) || optimizer_steps(o).is_ok_and(|v| v.contains(&6));
    let _ = uses_gmhmaxlik; // [stream O] every optimizer reads its optim options (optim::minimize)

    // ---- data (makedataset.m) ---------------------------------------------
    crate::progress::phase("data and initial checks");
    // makedataset.m: datafile or the data command, first_obs / last_obs / nobs
    let mut ds = crate::dataset::Dataset::load(mf, dir, o)?;
    // DSGE-VAR: makedataset.m prepends dsge_varlag initial conditions
    let init_cond = crate::estimation::q3::initial_conditions(mf, o)?;
    if init_cond > 0 {
        ds.first_obs -= init_cond;
        ds.nobs += init_cond;
        ds.init = ds.init.add(-(init_cond as i64));
    }
    let (datapath, table, first_obs, sample) =
        (ds.path.clone(), ds.table.clone(), ds.first_obs, ds.sample(&mf.varobs)?);
    let nobs = sample.nrows();
    let mut kopts = KalmanOptions {
        presample: num(o, "presample").unwrap_or(0.0) as usize,
        lik_init: num(o, "lik_init").unwrap_or(1.0) as u8,
        kalman_algo: num(o, "kalman_algo").unwrap_or(0.0) as u8,
        ..Default::default()
    };
    if flag(o, "diffuse_filter") {
        kopts.lik_init = 3;
    }
    if let Some(v) = num(o, "kalman_tol") {
        kopts.kalman_tol = v;
    }
    if let Some(v) = num(o, "diffuse_kalman_tol") {
        kopts.diffuse_kalman_tol = v;
    }
    if let Some(v) = num(o, "riccati_tol") {
        kopts.riccati_tol = v;
    }
    kopts.qz_criterium = ss_opts.qz_criterium; // select_qz_criterium_value.m (done by the caller)
    crate::estimation::q3::kalman_options(o, &mut kopts);
    crate::estimation::est_completion::kalman_options(o, &mut kopts); // ---- estimation completion
    let mut sample = sample;
    if flag(o, "loglinear") && !flag(o, "logdata") {
        sample.iter_mut().for_each(|v| *v = v.ln()); // makedataset.m: dataset_.log()
    }
    let mut data = LikelihoodData::new(&sample, kopts.clone());
    data.first_obs = first_obs; // [stream X] observation_trends
    crate::estimation::q3::attach(mf, o, &mut data, first_obs, ds.init)?; // heteroskedastic_filter
    if num(o, "prefilter").unwrap_or(0.0) == 1.0 {
        data.prefilter();
    }
    if flag(o, "endogenous_prior") {
        if kopts.lik_init != 1 {
            bail!("Endogenous prior not supported with non-stationary models");
        }
        data.endogenous_prior = true;
    }
    // ---- calibration restrictions ----
    data.calib_restrictions = crate::gsa::calibration::restrictions_at_options(mf, o)?;
    // ---- end calibration restrictions ----
    let missing = data.missing.any_missing;
    crate::progress::note(format!(
        "Data: {} ({} observables, {} observations from first_obs = {}{}).",
        datapath.display(),
        mf.varobs.len(),
        nobs,
        first_obs,
        if missing { ", with missing values" } else { "" }
    ));

    // ---- initial values ----------------------------------------------------
    let mode_file = string(o, "mode_file");
    let names_dynare_order: Vec<String> =
        mf.estimated_params_dynare_order().iter().map(|e| mf.estimated_param_name(e)).collect();
    let (x_file, hh_file) = match &mode_file {
        Some(f) => {
            let mc = num(o, "mode_compute").unwrap_or(5.0) as i32;
            let (x, hh) = read_mode_file(dir, f, &names_dynare_order, mc, || crate::dsge::initial_vector(mf))?;
            (Some(x), hh)
        }
        None => (None, None),
    };
    let x_init = match &x_file {
        Some(x) => x.clone(),
        None => crate::dsge::initial_vector(mf)?,
    };
    let dsge = crate::estimation::est_completion::with_solver_options(DsgeModel::new(mf, Some(&x_init))?, ss_opts)?; // ---- estimation completion: dr=, qz_zero_threshold
    // set_prior.m: beta priors first, then gamma priors
    for k in 0..dsge.priors.len() {
        let p = &dsge.priors[k];
        if p.shape.code() == 1 && !(p.p6.is_nan() || p.p7.is_nan()) {
            if p.p6 < 1.0 || p.p7 < 1.0 {
                println!("Prior distribution for parameter {} has unbounded density!", dsge.estimated_names[k]);
            }
            if p.p6 < 1.0 && p.p7 < 1.0 {
                println!("Prior distribution for parameter {} has two modes!", dsge.estimated_names[k]);
            }
        }
    }
    for k in 0..dsge.priors.len() {
        let p = &dsge.priors[k];
        if p.shape.code() == 2 && p.p6 < 1.0 {
            println!("Prior distribution for parameter {} has unbounded density!", dsge.estimated_names[k]);
        }
    }
    if mode_file.is_some() {
        println!(); // check_mode_file.m (after set_prior.m) ends with skipline()
    }
    // initial_estimation_checks.m: check_prior_stderr_corr_skew
    let kinds_dyn: Vec<_> = mf.estimated_params_dynare_order().iter().map(|e| e.kind).collect();
    for w in crate::estimation::prior::check_prior_stderr_corr_skew(&kinds_dyn, &dsge.priors) {
        println!("Warning: {w}");
    }
    // dynare_estimation_init.m: prefiltered data need zero-mean measurement equations
    if data.prefilter_mean.is_some()
        && dsge.varobs.iter().any(|&v| dsge.level_constant(v, &dsge.ys_guess).abs() >= 1e-9)
    {
        bail!("The option \"prefilter\" is inconsistent with the non-zero mean measurement equations in the model.");
    }
    let lin_post = DsgePosterior { dsge: &dsge, data: &data, bounds: &dsge.bounds };
    // ---- stream OE: with occbin_constraints, OccBin's piecewise Kalman / inversion filter ----
    let occ = crate::occbin::estimation::OccbinEstimation::new(mf, occbin, &dsge, &data)?;
    if let Some(oe) = &occ {
        crate::occbin::estimation::check_estimation_options(oe, o, ss_opts)?;
    }
    let occ_post = occ.as_ref().map(|oe| crate::occbin::estimation::OccbinPosterior {
        dsge: &dsge,
        data: &data,
        bounds: &dsge.bounds,
        occ: oe,
    });
    // ---- end stream OE ----
    // dsge_var_likelihood instead of dsge_likelihood
    let q3x = crate::estimation::q3::Extras::new(mf, o, &dsge, &data)?;
    let q3_post = q3x.posterior(&dsge, &data, &dsge.bounds);
    let post: &dyn Objective = match (&occ_post, &q3_post) {
        (Some(p), _) => p,
        (_, Some(p)) => p.as_ref(),
        _ => &lin_post,
    };
    let n = dsge.n_estimated();
    // ---- estimation completion: analytic_derivation (dynare_estimation_init.m / dsge_likelihood.m checks)
    let is_dsge_likelihood = occ_post.is_none() && q3_post.is_none();
    let ad_opts =
        crate::estimation::analytic::check_options(o, kopts.kalman_algo, kopts.lik_init, data.missing.any_missing, 1)?;
    if ad_opts.is_some() && !is_dsge_likelihood {
        bail!(
            "analytic_derivation is only available with dsge_likelihood (no gradient from dsge_var_likelihood or the OccBin filters; Dynare fails in the optimizer)"
        );
    }
    let ad_ctx = match &ad_opts {
        Some(a) => Some(crate::estimation::analytic::AnalyticContext::new(&lin_post, *a)?),
        None => None,
    };
    let ad_post = ad_ctx.as_ref().map(|c| crate::estimation::analytic::AnalyticPosterior { post: &lin_post, ctx: c });
    let cova_compute = num(o, "cova_compute").unwrap_or(1.0) != 0.0;
    let mode_hessian = crate::estimation::mode_hessian::ModeHessian {
        obj: post,
        dsge_likelihood: is_dsge_likelihood.then_some(&lin_post),
        analytic: ad_post.as_ref(),
        cova_compute,
        penalized: flag(o, "use_penalized_objective_for_hessian"),
        newratflag: crate::estimation::mode_hessian::newratflag(opt(o, "optim"), ad_opts.is_some())?,
        bounds: &dsge.bounds,
        prior_sd: &[],
    };
    let prior_sd_mh = dsge.prior_sd();
    let mode_hessian = crate::estimation::mode_hessian::ModeHessian { prior_sd: &prior_sd_mh, ..mode_hessian };
    // ---- end estimation completion
    // [graphics] dynare_estimation_init.m: plot_priors (default 1) unless nograph
    let gopts = crate::graphics::GraphOptions::from_options(o, dir, fname);
    if gopts.enabled && num(o, "plot_priors").unwrap_or(1.0) != 0.0 {
        let figs = crate::estimation::plots::priors_figures(&dsge.priors, &dsge.estimated_names);
        let k = crate::estimation::plots::save_figures(&figs, fname, &gopts)?;
        crate::progress::note(format!(
            "plot_priors: {k} figures ({fname}_Priors*) written to {}.",
            gopts.dir.display()
        ));
    }

    // ---- estimation completion: initial_estimation_checks.m (stochastic singularity, no_init_estimation_check_first_obs)
    crate::estimation::est_completion::initial_estimation_checks(mf, &dsge, &sample, &x_init, o, data.hetero.as_ref())?;
    let _huge =
        crate::estimation::optim::minimize::HugeNumberGuard::set(crate::estimation::est_completion::huge_number(o));
    // `options_.gradient_epsilon = V;` / `options_.gradient_method = V;` before the command
    let _grad = {
        let d = crate::estimation::optim::minimize::GradientOptions::default();
        crate::estimation::optim::minimize::GradientGuard::set(crate::estimation::optim::minimize::GradientOptions {
            method: num(o, "gradient_method").unwrap_or(d.method),
            epsilon: num(o, "gradient_epsilon").unwrap_or(d.epsilon),
        })
    };
    // ---- end estimation completion
    // initial_estimation_checks.m
    let t = Instant::now();
    let (f0, info0) = post.eval(&x_init);
    if !f0.is_finite() {
        if info0 == 49 {
            // ---- calibration restrictions (print_info: get_error_message(49)) ----
            bail!(
                "initial_estimation_checks: the posterior is not finite at the initial values (info 49: The model violates one (many) endogenous prior restriction(s))."
            );
        }
        bail!("initial_estimation_checks: the posterior is not finite at the initial values (info {info0}).");
    }
    // initial_estimation_checks.m: disp(['Initial value of the log posterior (or likelihood): ' num2str(-fval)])
    if num(o, "mode_compute").unwrap_or(5.0) as i32 != 11 {
        println!("Initial value of the log posterior (or likelihood): {}", crate::gsa::stats::num2str(-f0));
    }
    crate::progress::note(format!("({:.2} ms per evaluation)", t.elapsed().as_secs_f64() * 1e3));

    // ---- mode finding (dynare_estimation_1.m) -------------------------------
    let mode_compute = num(o, "mode_compute").unwrap_or(5.0) as i32;
    // options_.mh_jscale: empty -> 2.38/sqrt(npar) (dynare_estimation_init.m); mode_compute=6
    // replaces it with its tuned Scale.
    let mut mh_jscale = num(o, "mh_jscale").unwrap_or(2.38 / (n as f64).sqrt());
    let mut gmh_scale: Option<f64> = None;
    // stream PF: posterior_sampling_method = 'dsmh' (dynare_estimation_1.m: issmc -> no mode
    // finding, no MCMC), on the linear (Kalman) posterior
    if crate::nonlinear::smc_run::smc_method(o).is_some() {
        // dsmh, hssmc, dime_mcmc, online (at order 1)
        let mut results =
            json!({ "model": fname, "parameter_names": dsge.estimated_names, "initial_minus_log_posterior": f0 });
        let online = crate::nonlinear::smc_run::online_model(mf, &x_init, o, &sample, ro.seed)?;
        let means = crate::nonlinear::smc_run::run_smc(
            mf,
            post,
            &dsge,
            nobs,
            o,
            ro,
            out_dir,
            fname,
            &mut results,
            online.as_ref(),
            &x_init,
        )?;
        let st = fc::EstimationState {
            dsge: DsgeModel::new(mf, Some(&x_init))?,
            data: LikelihoodData::new(&sample, kopts.clone()),
            dates: table.dates.as_ref().map(|d| d[first_obs - 1..first_obs - 1 + nobs].to_vec()),
            first_obs,
            var_list: var_list.to_vec(),
            mode: means.clone(),
            posterior_mean: Some(means.clone()),
            posterior_median: None,
            graphs: fc::graph_options(o, dir, fname),
            mean_forecast: None,
            point_forecast: None,
            forecast: None,
            smoothed_mean: None,
        };
        return Ok((st, None, means));
    }
    // [stream O] posterior_sampling_method / posterior_sampler_options (check_posterior_sampler_options.m).
    let mut sampler =
        SamplerOptions::from_options(opt(o, "posterior_sampling_method"), opt(o, "posterior_sampler_options"))?
            .with_estimation_optim(opt(o, "optim"))
            .with_priors(&dsge.priors);
    if let Some(f) = &sampler.scale_file {
        mh_jscale = load_scale_file(&dir.join(f).to_string_lossy())?;
        println!("mh_jscale = {mh_jscale} read from scale_file {f}");
    }
    // ---- MCMC files (load_mh_file, mh_recover, mh_initialize_from_previous_mcmc, ...) ----
    let mfo = mh_session::MhFileOptions::from_options(o, &sampler, dir)?;
    crate::estimation::samplers::check_maximize(&sampler, mfo.load_mh_file, mfo.mh_recover)?;
    // Slice without slice_initialize_with_mode, or mh_posterior_mode_estimation: no mode
    // finding, no Hessian; the MCMC starts at the prior mode (dynare_estimation_1.m).
    let mode_estim = sampler.mh_posterior_mode_estimation() || (mfo.mh_posterior_mode_estimation && !dsge.ml);
    // dynare_estimation_1.m: mode_compute=0 without a mode file only runs the smoother at
    // the initial values (not with mh_posterior_mode_estimation, e.g. the slice sampler's default)
    let smoother_only = mode_compute == 0 && mode_file.is_none() && flag(o, "smoother") && !mode_estim;
    let (xmode, fmode, hh) = if mode_estim {
        println!("No posterior maximization (mh_posterior_mode_estimation); the MCMC starts from the prior mode.");
        let x = prior_mode_start(&dsge.priors, &dsge.bounds);
        let f = post.eval(&x).0;
        // invhess = diag(prior variances), infinite ones set to 1
        let hh = if mfo.mh_posterior_mode_estimation {
            Mat::from_diagonal(&nalgebra::DVector::from_iterator(
                n,
                dsge.priors.iter().map(|p| {
                    let v = p.sd * p.sd;
                    if v.is_finite() { 1.0 / v } else { 1.0 }
                }),
            ))
        } else {
            Mat::identity(n, n)
        };
        (x, f, hh)
    } else if mode_compute == 0 {
        let hh = match hh_file {
            Some(h) => h,
            None if smoother_only => Mat::from_element(n, n, f64::NAN),
            None => {
                println!("mode_compute = 0 and no Hessian in the mode file: computing it with hessian.m.");
                crate::progress::phase("Hessian at the mode file's point");
                hessian(post, &x_init, [1e-2, 1.0])
            }
        };
        (x_init.clone(), f0, hh)
    } else {
        // optimizer_vec = [mode_compute; additional_optimizer_steps], each step starting
        // from the previous one's mode; `hh` (mode file, then each step's Hessian) is the
        // Initial_Hessian of gmhmaxlik.
        let mut optimizer_vec = vec![mode_compute];
        optimizer_vec.extend(optimizer_steps(o)?);
        let mut x = x_init.clone();
        let mut fval = f0;
        let mut hh = hh_file;
        for (step, &optimizer) in optimizer_vec.iter().enumerate() {
            let t = Instant::now();
            crate::progress::phase(format!(
                "mode_compute = {optimizer} (step {} of {})",
                step + 1,
                optimizer_vec.len()
            ));
            if optimizer == 6 {
                let prior_sd: Vec<f64> = dsge.prior_sd();
                let mut gopts = GmhmaxlikOptions {
                    silent: ro.quiet || flag(o, "silent_optimizer"),
                    seed: ro.seed,
                    stream: u64::MAX - 1 - step as u64,
                    prefetch: ro.prefetch,
                    ..Default::default()
                };
                if let Some(v) = opt(o, "optim") {
                    gopts.apply_optim(v)?;
                }
                let res = gmhmaxlik(post, &x, hh.as_ref(), mh_jscale, &dsge.bounds, &prior_sd, &gopts)?;
                crate::progress::note(format!(
                    "mode_compute = 6: Optimization successful. ({} function evaluations, {} per round, {}).",
                    res.func_count,
                    res.prefetch,
                    hms(t.elapsed().as_secs_f64())
                ));
                // dynare_estimation_1.m: save the scale, use it as mh_jscale; hh is
                // gmhmaxlik's (no hessian.m for optimizer 6).
                std::fs::create_dir_all(out_dir)?;
                let scale_path = out_dir.join(format!("{fname}_optimal_mh_scale_parameter.mat"));
                matfile::write(&scale_path, &[("Scale", MatOut::Scalar(res.scale))])?;
                crate::progress::note(format!("Optimal mh_jscale {:.6} saved to {}", res.scale, scale_path.display()));
                mh_jscale = res.scale;
                gmh_scale = Some(res.scale);
                x = res.x;
                fval = res.fval;
                hh = Some(res.hessian);
            } else {
                // [stream O] dynare_minimize_objective.m: every other mode_compute, with optim overrides.
                let prior_sd = dsge.prior_sd(); // NaN under ML (stream M)
                let ctx = MinimizeContext {
                    bounds: &dsge.bounds,
                    names: &dsge.estimated_names,
                    prior_sd: &prior_sd,
                    optim: opt(o, "optim"),
                    verbose: !ro.quiet && !flag(o, "silent_optimizer"), // silent_optimizer
                    seed: ro.seed,
                    stream: u64::MAX - 1 - step as u64,
                };
                // ---- estimation completion: analytic_derivation (analytic gradients in the optimizers)
                let res = {
                    let _ad = crate::estimation::optim::minimize::AnalyticGuard::set(ad_post.is_some());
                    // newrat's outer-product Hessian: dsge_likelihood's DLIK = [-lnprior; lik(:)],
                    // per observable with the univariate filters
                    let data_obs = (optimizer == 5 && is_dsge_likelihood && ad_post.is_none()).then(|| {
                        let mut d = data.clone();
                        d.kalman.obs_contributions = true;
                        d
                    });
                    let post_obs =
                        data_obs.as_ref().map(|d| DsgePosterior { dsge: &dsge, data: d, bounds: &dsge.bounds });
                    let obj: &dyn Objective = match (&ad_post, &post_obs) {
                        (Some(a), _) => a,
                        (None, Some(p)) => p,
                        _ => post,
                    };
                    minimize(optimizer, obj, &x, &ctx)?
                };
                crate::progress::note(format!(
                    "mode_compute = {optimizer}: {} ({} iterations, {} function evaluations, {}).",
                    res.message,
                    res.iterations,
                    res.func_count,
                    hms(t.elapsed().as_secs_f64())
                ));
                let t = Instant::now();
                crate::progress::phase(format!("Hessian at the mode (mode_compute = {optimizer})"));
                // ---- estimation completion: dynare_estimation_1.m's Hessian at the mode
                let optimizer_hh = if optimizer == 5 { res.hessian.clone() } else { None };
                let h = mode_hessian.compute(optimizer, &res.x, res.fval, optimizer_hh)?;
                if h.is_some() && cova_compute {
                    crate::progress::note(format!(
                        "Hessian at the mode computed in {}.",
                        hms(t.elapsed().as_secs_f64())
                    ));
                }
                x = res.x;
                fval = res.fval;
                hh = h;
            }
            // dynare_estimation_1.m: fprintf('\nFinal value of minus the log posterior (or likelihood):%f \n', fval)
            println!("\nFinal value of minus the log posterior (or likelihood):{fval:.6} ");
        }
        std::fs::create_dir_all(out_dir)?;
        let mode_path = out_dir.join(format!("{fname}_mode.mat"));
        // ---- estimation completion: cova_compute = 0 saves no hh (unless mode_compute 5/6)
        let last = *optimizer_vec.last().unwrap();
        let hh = match hh {
            Some(h) if cova_compute || last == 5 || last == 6 => {
                matfile::write(
                    &mode_path,
                    &[
                        ("xparam1", MatOut::Vector(&x)),
                        ("hh", MatOut::Matrix(&h)),
                        ("fval", MatOut::Scalar(fval)),
                        ("parameter_names", MatOut::Strings(&dsge.estimated_names)),
                    ],
                )?;
                h
            }
            _ => {
                matfile::write(
                    &mode_path,
                    &[
                        ("xparam1", MatOut::Vector(&x)),
                        ("fval", MatOut::Scalar(fval)),
                        ("parameter_names", MatOut::Strings(&dsge.estimated_names)),
                    ],
                )?;
                Mat::from_element(n, n, f64::NAN)
            }
        };
        // ---- end estimation completion
        crate::progress::note(format!("Mode saved to {}", mode_path.display()));
        (x, fval, hh)
    };

    // ---- calibration restrictions: check_hessian_at_the_mode.m (e.g. a mode on the boundary
    // of an irf/moment_calibration restriction, where hessian.m meets Inf values) ----
    if !mode_estim && !smoother_only && (hh.iter().any(|v| !v.is_finite()) || hh.clone().cholesky().is_none()) {
        println!("\nOPTIMIZATION PROBLEM!");
        println!(" (minus) the Hessian matrix at the \"mode\" is not positive definite!");
        println!("=> variance of the estimated parameters are not positive.");
        println!("You should try to change the initial values of the parameters using");
        println!("the estimated_params_init block, or use another optimization routine.");
        let names = crate::gsa::gsa_param_names(mf, &dsge);
        let at_bound: Vec<&str> = (0..n)
            .filter(|&k| (xmode[k] - dsge.bounds.ub[k]).abs() < 1e-10 || (xmode[k] - dsge.bounds.lb[k]).abs() < 1e-10)
            .map(|k| names[k].as_str())
            .collect();
        if !at_bound.is_empty() {
            println!("\nThe following parameters are at the bound: {}", at_bound.join(", "));
            println!("Some potential solutions are:");
            for l in [
                "Check your model for mistakes.",
                "Check whether model and data are consistent (correct observation equation).",
                "Shut off prior_trunc.",
                "Change the optimization bounds.",
                "Use a different mode_compute like 6 or 9.",
                "Check whether the parameters estimated are identified.",
                "Check prior shape (e.g. Inf density at bound(s)).",
                "Increase the informativeness of the prior.",
            ] {
                println!("   - {l}");
            }
        }
        println!("Warning: The results below are most likely wrong!");
    }
    // ---- end calibration restrictions ----

    // [graphics] mode_check.m (dynare_estimation_1.m, after the Hessian)
    if flag(o, "mode_check") {
        mode_check_step(post, &dsge, &xmode, fmode, &hh, o, fname, &gopts)?;
    }

    // ---- mode table + Laplace -------------------------------------------------
    let invhess = inverse_hessian(&hh);
    let sd_mode: Vec<f64> = match &invhess {
        // dynare_estimation_1.m: `if ~options_.cova_compute, stdh = NaN(length(xparam1),1)`
        _ if !cova_compute => vec![f64::NAN; n],
        // (NaN stays NaN, as MATLAB's sqrt: a non-finite Hessian gives NaN standard deviations)
        Some(v) => (0..n).map(|i| if v[(i, i)].is_nan() { f64::NAN } else { v[(i, i)].max(0.0).sqrt() }).collect(),
        None => vec![f64::NAN; n],
    };
    let laplace = if smoother_only || mode_estim {
        f64::NAN
    } else if dsge.ml {
        print_ml_table(mf, &dsge, &xmode, &sd_mode);
        f64::NAN
    } else {
        print_mode_table(mf, &dsge, &xmode, &sd_mode);
        if flag(o, "tex") {
            // stream B: TeX tables (display_estimation_results_table.m)
            crate::latex::estimation_tables::posterior_mode(
                mf,
                &dsge.estimated,
                &dsge.priors,
                &xmode,
                &sd_mode,
                dir,
                fname,
            )?;
        }
        if !cova_compute {
            f64::NAN // ---- estimation completion: no Laplace approximation without cova_compute
        } else {
            let laplace = laplace_approximation(&hh, fmode);
            println!("\nLog data density [Laplace approximation] is {laplace:.6}.\n");
            laplace
        }
    };

    let mut results = json!({
        "model": fname,
        "parameter_names": dsge.estimated_names,
        "mode": xmode,
        "dsge_var_posterior_mode": q3x.posterior_mode_json(&dsge, &xmode), // oo_.dsge_var.posterior_mode
        "mode_sd": sd_mode,
        "minus_log_posterior_at_mode": fmode,
        "log_data_density_laplace": laplace,
    });

    // ---- Metropolis–Hastings ----------------------------------------------------
    let mut mh_replic = ro.mh_replic.unwrap_or(num(o, "mh_replic").unwrap_or(20000.0) as usize);
    if dsge.ml || smoother_only {
        mh_replic = 0; // dynare_estimation_init.m: no Metropolis without priors
    }
    let mut point_for_smoother = xmode.clone();
    // ---- stream F ----
    let (mut posterior_mean, mut posterior_median): (Option<Vec<f64>>, Option<Vec<f64>>) = (None, None);
    let dates = table.dates.as_ref().map(|d| d[first_obs - 1..first_obs - 1 + nobs].to_vec());
    // (DsgeModel is not Clone; rebuilding it from the same inputs gives the same model)
    let state = |pm: Option<Vec<f64>>, pmed: Option<Vec<f64>>| -> Result<fc::EstimationState> {
        Ok(fc::EstimationState {
            dsge: crate::estimation::est_completion::with_solver_options(DsgeModel::new(mf, Some(&x_init))?, ss_opts)?, // ---- estimation completion
            data: {
                // ---- stream R: the later commands see makedataset's data (prefilter: demeaned) ----
                let mut d = LikelihoodData::new(&sample, kopts.clone());
                if data.prefilter_mean.is_some() {
                    d.prefilter();
                }
                d.endogenous_prior = data.endogenous_prior;
                d.hetero = data.hetero.clone();
                d.filter_initial_state = data.filter_initial_state.clone();
                d.filter_initial_state_expr = data.filter_initial_state_expr.clone();
                d.calib_restrictions = data.calib_restrictions.clone(); // ---- calibration restrictions ----
                d.skewed = data.skewed.clone(); // ---- estimation completion ----
                d
            },
            dates: dates.clone(),
            first_obs,
            var_list: var_list.to_vec(),
            mode: xmode.clone(),
            posterior_mean: pm,
            posterior_median: pmed,
            graphs: fc::graph_options(o, dir, fname),
            mean_forecast: None,
            point_forecast: None,
            forecast: None,
            smoothed_mean: None,
        })
    };
    // ---- end stream F ----
    // dynare_estimation_1.m: `options_.smoother || ~isempty(options_.filter_step_ahead) || options_.forecast`
    // (filtered_vars sets filter_step_ahead = 1; filter_covariance alone computes nothing)
    let post_stats = flag(o, "smoother")
        || flag(o, "filtered_vars")
        || crate::modfile::opt(o, "filter_step_ahead").is_some()
        || fc::forecast_horizon(o) > 0;
    let mut mean_forecast: Option<crate::forecast::Forecast> = None; // oo_.MeanForecast
    let mut point_forecast: Option<crate::forecast::Forecast> = None; // oo_.PointForecast
    let mut smoothed_mean: Option<(Mat, Mat)> = None; // oo_.SmoothedVariables.Mean, ...
    let has_mcmc = !dsge.ml && !smoother_only && (mh_replic > 0 || mfo.load_mh_file);
    if has_mcmc {
        // use_mh_covariance_matrix: the previous MCMC's posterior covariance
        let invhess = if mfo.load_mh_file && mfo.use_mh_covariance_matrix {
            mh_session::previous_mh_covariance(&dname, fname, out_dir, &dsge.estimated_names)?.0
        } else if mh_replic == 0 {
            invhess.unwrap_or_else(|| Mat::identity(n, n))
        } else if !cova_compute && !mode_estim && sampler.method != crate::estimation::samplers::SamplingMethod::Slice {
            // ---- estimation completion: cova_compute = 0 leaves invhess = [] (only a jumping
            // covariance that does not use it can start the MCMC)
            match opt(o, "mcmc_jumping_covariance") {
                Some(OptValue::Str(v)) if !v.contains("hessian") => Mat::identity(n, n),
                _ => {
                    println!(
                        "\ncheck_posterior_sampler_options:: I cannot start the MCMC because the Hessian of the posterior kernel at the mode was not computed"
                    );
                    println!("check_posterior_sampler_options:: or there is no previous MCMC to load ");
                    bail!("check_posterior_sampler_options:: MCMC cannot start")
                }
            }
        } else {
            invhess.ok_or_else(|| anyhow!("the Hessian at the mode is not invertible: cannot start the MCMC"))?
        };
        let invhess = crate::estimation::q3::jumping_covariance(o, invhess, &dsge.priors, dir)?; // mcmc_jumping_covariance
        let prior_trunc = crate::dsge::prior_trunc(mf);
        let mh_bounds = mcmc_bounds(&xmode, &dsge.priors, prior_trunc)?;
        // slice sampler's maximize_using_mh_bounds (check_posterior_sampler_options.m)
        let mh_drop_now = num(o, "mh_drop").unwrap_or(0.5);
        crate::estimation::samplers::maximize_bounds(&mut sampler, &dname, fname, mh_drop_now, &mh_bounds)?;
        let mh_lin = DsgePosterior { dsge: &dsge, data: &data, bounds: &mh_bounds };
        let mh_occ = occ.as_ref().map(|oe| crate::occbin::estimation::OccbinPosterior {
            dsge: &dsge,
            data: &data,
            bounds: &mh_bounds,
            occ: oe,
        }); // stream OE
        let mh_q3 = q3x.posterior(&dsge, &data, &mh_bounds);
        let mh_post: &dyn Objective = match (&mh_occ, &mh_q3) {
            (Some(p), _) => p,
            (_, Some(p)) => p.as_ref(),
            _ => &mh_lin,
        };
        let mut mh = MhOptions::new(n);
        mh.replic = mh_replic;
        mh.nblocks = num(o, "mh_nblocks").unwrap_or(2.0) as usize;
        mh.seed = ro.seed;
        mh.prefetch = ro.prefetch;
        if let Some(v) = num(o, "mh_init_scale_factor").or(num(o, "mh_init_scale")) {
            mh.init_scale_factor = v;
        }
        let mh_drop = num(o, "mh_drop").unwrap_or(0.5);
        mh.store_from = (mh_drop * mh_replic as f64).floor() as usize;
        // [graphics] mcmc_diagnostics and trace plots read the burn-in draws too
        if !flag(o, "nodiagnostic") && crate::estimation::plots::keep_all_draws(n, mh_replic, mh.nblocks) {
            mh.store_from = 0;
        }
        // Every 100 draws for the progress window; the console prints every print_every.
        mh.progress_every = 100;
        let print_every = (mh_replic / 20).max(1000).next_multiple_of(100);
        let mut jscale = mh_jscale;
        if flag(o, "mh_tune_jscale") && mh_replic > 0 {
            // Overrides mode_compute=6's Scale too; the tuning starts from mh_tune_guess
            // (or the user's mh_jscale), else 2.38/sqrt(npar), never from Scale.
            let guess = num(o, "mh_tune_guess").or(num(o, "mh_jscale")).unwrap_or(mh.mh_jscale);
            let t = Instant::now();
            let target = num(o, "mh_tune_jscale").unwrap_or(0.33);
            crate::progress::phase(format!("mh_tune_jscale (target acceptance {target})"));
            jscale = mh_tune_jscale(
                &mh_post,
                &invhess,
                &xmode,
                &mh_bounds,
                &TuneOptions { target, guess: Some(guess), seed: ro.seed, prefetch: ro.prefetch, ..Default::default() },
            )?;
            // dynare_estimation_1.m: fprintf('mh_jscale has been set equal to %s\n', num2str(options_.mh_jscale))
            println!("mh_jscale has been set equal to {}", crate::gsa::stats::num2str(jscale));
            crate::progress::note(format!("mh_jscale tuned in {}.", hms(t.elapsed().as_secs_f64())));
        }
        mh.mh_jscale = jscale;
        // bayestopt_.jscale: per-parameter estimated_params scale (set_prior.m); NaN entries
        // become mh_jscale (dynare_estimation_init.m:462). mode_compute=6, mh_tune_jscale and
        // scale_file overwrite every entry (dynare_estimation_1.m, check_posterior_sampler_options.m).
        let overwritten = gmh_scale.is_some() || flag(o, "mh_tune_jscale") || sampler.scale_file.is_some();
        mh.jscale = (0..n)
            .map(|i| match dsge.estimated.get(i).and_then(|e| e.jscale) {
                Some(v) if !v.is_nan() && !overwritten => v,
                _ => jscale,
            })
            .collect();
        crate::progress::note(format!(
            "Estimation::mcmc ({}): {} chains x {} draws, running in parallel on {} threads.",
            sampler.title(),
            mh.nblocks,
            mh.replic,
            rayon::current_num_threads()
        ));
        crate::progress::phase(format!("MCMC: {} chains x {} draws", mh.nblocks, mh.replic));
        crate::progress::chains(mh.nblocks, mh.replic);
        let t = Instant::now();
        let progress = |p: &MhProgress| {
            crate::progress::chain(p.chain, p.draws_done, p.acceptance);
            if p.chain == 0 && p.draws_done.is_multiple_of(print_every) {
                crate::progress::note(format!(
                    "  chain 1: {:>9}/{} draws, acceptance {:5.2}%  [{}]",
                    p.draws_done,
                    p.total,
                    100.0 * p.acceptance,
                    hms(t.elapsed().as_secs_f64())
                ));
            }
        };
        // posterior_sampler.m with Dynare's mh files (<fname>/metropolis/)
        let res = mh_session::run(
            mh_session::SessionInputs {
                disp: "Estimation::mcmc",
                obj: mh_post,
                xparam1: &xmode,
                invhess: &invhess,
                bounds: &mh_bounds,
                mh: mh.clone(),
                sampler: &sampler,
                names: &dsge.estimated_names,
                priors: &dsge.priors,
                dname: &dname,
                fname,
                mh_drop,
                keep_all: mh.store_from == 0,
                progress: Some(&progress),
            },
            &mfo,
        )?;
        let mcmc_secs = t.elapsed().as_secs_f64();
        crate::progress::note(format!(
            "Estimation::mcmc: done in {} ({} posterior evaluations per chain round).",
            hms(mcmc_secs),
            res.prefetch
        ));
        // mh_posterior_mode_estimation: the posterior mode from the MCMC, then stop
        if mfo.mh_posterior_mode_estimation {
            let (m, sd, kernel) = mh_session::posterior_mode_from_mcmc(&dname, fname, out_dir, &dsge.estimated_names)?;
            println!(
                "\nPosterior mode from the MCMC (log posterior kernel {kernel:.6}), saved to {fname}_mh_mode.mat:"
            );
            print_mode_table(mf, &dsge, &m, &sd);
            results["mh_posterior_mode"] = json!({"mode": m, "sd": sd, "log_posterior_kernel": kernel});
            std::fs::create_dir_all(out_dir)?;
            std::fs::write(out_dir.join(format!("{fname}_results.json")), serde_json::to_string_pretty(&results)?)?;
            return Ok((state(None, None)?, None, m));
        }
        let previous_results = if mfo.load_results_after_load_mh && mh_replic == 0 {
            let r = mh_session::load_previous_results(out_dir, fname).and_then(|v| v.get("mcmc").cloned());
            if r.is_none() {
                println!(
                    "\ndynare_estimation_init:: You specified the load_results_after_load_mh, but no _results file"
                );
                println!("dynare_estimation_init:: was found. Results will be recomputed.");
            }
            r
        } else {
            None
        };
        crate::progress::phase("posterior statistics (HPD, MHM)");
        let post_opts =
            PosteriorOptions { mh_drop, mh_conf_sig: num(o, "mh_conf_sig").unwrap_or(0.9), ..Default::default() };
        let stats = crate::estimation::mcmc::bounded(|| posterior_stats(&res.chains, &post_opts))?;
        let diagnostics = !flag(o, "nodiagnostic") && previous_results.is_none();
        let conv = crate::estimation::convergence::ConvergenceOptions {
            // options_.convergence
            last_file_lines: Some(res.record.last_line_number),
            ..crate::estimation::convergence::ConvergenceOptions::from_options(o)
        };
        if previous_results.is_none() {
            // dynare_estimation_1.m: mcmc_diagnostics (inefficiency factors, Geweke, Brooks-Gelman),
            // then marginal_density and GetPosteriorParametersStatistics
            if diagnostics {
                print_ifac(mf, &dsge, &stats.ifac);
                crate::estimation::mcmc::bounded(|| {
                    crate::estimation::plots::mcmc_diagnostics(
                        &res.chains,
                        &dsge.estimated_names,
                        mh_drop,
                        fname,
                        &gopts,
                        &conv,
                    )
                })?;
            }
            print_marginal_density_messages(&stats.mhm);
            // marginal_density.m: with the slice sampler, oo_.posterior_mode is the MCMC's best draw
            // (fill_mh_mode with NaN standard deviations)
            if mh_replic > 0 && sampler.method == crate::estimation::samplers::SamplingMethod::Slice {
                let (m, _, _) = mh_session::posterior_mode_from_mcmc(&dname, fname, out_dir, &dsge.estimated_names)?;
                results["mode"] = json!(m);
                results["mode_sd"] = json!(vec![f64::NAN; n]);
            }
            print_posterior_table(mf, &dsge, &stats);
        }
        if flag(o, "tex") {
            // stream B: TeX tables (GetPosteriorParametersStatistics.m)
            crate::latex::estimation_tables::posterior_mean(
                mf,
                &dsge.estimated,
                &dsge.priors,
                &stats.params,
                dir,
                fname,
            )?;
        }
        // stream S: bayesian_irf / moments_varendo (PosteriorIRF.m, compute_moments_varendo.m)
        // ---- estimation completion: posterior_nograph (PosteriorIRF.m: ~options_.no_graph.posterior)
        let ss_post = crate::analysis::StochSimulOptions {
            nograph: ss_opts.nograph || crate::estimation::est_completion::no_graph_posterior(o),
            ..ss_opts.clone()
        };
        crate::analysis::posterior::run(
            &dsge,
            &res.chains,
            &ss_post,
            var_list,
            dir,
            fname,
            out_dir,
            ro.seed,
            q3x.dsge_var.is_some(),
        )?;
        q3x.posterior_irf(&dsge, &mh_bounds, &res.chains, ss_opts, out_dir, fname, ro.seed)?; // DSGE-VAR IRFs
        write_posterior_csv(out_dir, fname, &dsge, &xmode, &sd_mode, &stats)?;
        // [graphics] mcmc_diagnostics.m (unless nodiagnostic), PlotPosteriorDistributions.m
        crate::progress::phase("posterior densities");
        crate::estimation::mcmc::bounded(|| {
            crate::estimation::plots::mcmc_figures(
                &res.chains,
                &stats,
                &dsge.priors,
                &dsge.estimated_names,
                mh_drop,
                false, // the diagnostics ran before the posterior table
                fname,
                &gopts,
                &conv,
                if mode_estim { None } else { Some(&xmode) },
            )
        })?;
        results["mcmc"] = json!({
            "replic": res.record.total_draws(), "new_draws": res.new_draws, "nblocks": res.chains.len(),
            "jscale": res.jscale.first().copied().unwrap_or(jscale), "seconds": mcmc_secs,
            "acceptance": stats.acceptance, "log_data_density_mhm": stats.mhm.log_marginal,
            "posterior_mean": stats.mean,
            "hpd_lower": stats.params.iter().map(|p| p.hpd.0).collect::<Vec<_>>(),
            "hpd_upper": stats.params.iter().map(|p| p.hpd.1).collect::<Vec<_>>(),
            "posterior_median": stats.params.iter().map(|p| p.median).collect::<Vec<_>>(),
            "posterior_sd": stats.params.iter().map(|p| p.var.sqrt()).collect::<Vec<_>>(),
            "metropolis": dname.join("metropolis"),
        });
        point_for_smoother = stats.mean.clone();
        posterior_mean = Some(stats.mean.clone());
        posterior_median = Some(stats.params.iter().map(|p| p.median).collect());
        // load_results_after_load_mh: oo_ fields of the previous run
        if let Some(prev) = &previous_results {
            let vecf = |k: &str| {
                prev.get(k).and_then(|v| serde_json::from_value::<Vec<f64>>(v.clone()).ok()).filter(|v| v.len() == n)
            };
            if let Some(m) = vecf("posterior_mean") {
                point_for_smoother = m.clone();
                posterior_mean = Some(m);
            }
            if let Some(m) = vecf("posterior_median") {
                posterior_median = Some(m);
            }
            let mut r = prev.clone();
            r["loaded_from_previous_results"] = json!(true);
            results["mcmc"] = r;
        }
        // ---- stream F: prior_posterior_statistics.m ----
        if let (Some(oe), true) = (&occ, post_stats) {
            // OccBin smoother per posterior draw
            crate::estimation::q3_occbin::posterior_objects(
                &dsge,
                &data,
                oe,
                o,
                &res.chains,
                stats.first_draw,
                stats.kept_per_chain,
                ro.seed,
                dates.as_deref(),
                out_dir,
                fname,
            )?;
        } else if post_stats {
            crate::progress::phase("posterior distributions of smoothed objects / forecasts");
            let st = state(posterior_mean.clone(), posterior_median.clone())?;
            let pf = fc::posterior_objects(
                mf,
                &st,
                o,
                &res.chains,
                stats.first_draw,
                stats.kept_per_chain,
                ro.seed,
                out_dir,
                fname,
            )?;
            (mean_forecast, point_forecast, smoothed_mean) = (pf.mean_forecast, pf.point_forecast, pf.smoothed_mean);
        }
        // ---- end stream F ----
        // [graphics] kept for generate_trace_plots
        mcmc_plots = Some(crate::estimation::plots::McmcPlotContext {
            chains: res.chains,
            names: dsge.estimated_names.clone(),
            gopts: gopts.clone(),
            trace_plot_ma: 200,
        });
        crate::progress::note("(Smoothed variables below are evaluated at the posterior mean.)");
    }

    // ---- smoother -----------------------------------------------------------------
    if crate::estimation::est_completion::classical_smoother(o, has_mcmc, smoother_only) {
        // ---- estimation completion: frequentist_smoother
        let t = Instant::now();
        crate::progress::phase("smoother");
        let sm = match &occ {
            // stream OE: occbin.DSGE_smoother / IVF smoother
            Some(oe) => {
                let (sm, rh) =
                    crate::occbin::estimation::smoother_series(&dsge, &data, oe, &point_for_smoother, out_dir, fname)?;
                occbin.smoother_regimes = Some(rh);
                sm
            }
            None => {
                // Dynare runs no classical smoother after an MCMC: keep its console quiet
                crate::kalman::SMOOTHER_QUIET.store(has_mcmc, std::sync::atomic::Ordering::Relaxed);
                let r = run_smoother(&dsge, &data, &point_for_smoother, flag(o, "smoothed_state_uncertainty"));
                crate::kalman::SMOOTHER_QUIET.store(false, std::sync::atomic::Ordering::Relaxed);
                r?
            }
        };
        let dates = table.dates.as_ref().map(|d| d[first_obs - 1..first_obs - 1 + nobs].to_vec());
        write_series_csv(
            &out_dir.join(format!("{fname}_smoothed_variables.csv")),
            &mf.model.endo_names[..mf.model.orig_endo_nbr],
            &sm.smoothed,
            dates.as_deref(),
        )?;
        if occ.is_none() {
            crate::estimation::q3::smoother_outputs(&dsge, &data, &point_for_smoother, o, out_dir, fname)?; // filter_step_ahead, ...
        }
        write_series_csv(
            &out_dir.join(format!("{fname}_smoothed_shocks.csv")),
            &mf.model.exo_names,
            &sm.shocks,
            dates.as_deref(),
        )?;
        crate::q4::smoothed_epilogue(
            mf,
            &dsge.set_all_parameters(&point_for_smoother).0,
            &sm.smoothed,
            dates.as_deref(),
            out_dir,
            fname,
        )?;
        if flag(o, "filtered_vars") {
            write_series_csv(
                &out_dir.join(format!("{fname}_filtered_variables.csv")),
                &mf.model.endo_names[..mf.model.orig_endo_nbr],
                &sm.filtered,
                dates.as_deref(),
            )?;
        }
        crate::progress::note(format!(
            "Smoother: smoothed variables/shocks{} written to {} ({:.0} ms).",
            if flag(o, "filtered_vars") { " and one-step-ahead filtered variables" } else { "" },
            out_dir.display(),
            t.elapsed().as_secs_f64() * 1e3
        ));
        // [graphics] plot_classical_smoother_results.m: only without an MCMC, as in Dynare
        if crate::estimation::est_completion::plot_classical_smoother(o, has_mcmc) {
            // ---- estimation completion: frequentist_smoother
            crate::estimation::plots::smoother_figures(
                &mf.model,
                &dsge.varobs,
                var_list,
                &sm.smoothed,
                &sm.shocks,
                flag(o, "filtered_vars").then_some(&sm.filtered),
                &sample,
                dates.as_deref(),
                fname,
                &gopts,
            )?;
        }
        // (dynare-rust's list of the last smoothed values is in the results json, not on the console)
        let show: Vec<&String> = if var_list.is_empty() { vec![] } else { var_list.iter().collect() };
        if !show.is_empty() {
            crate::progress::note("Last smoothed values of the requested variables:");
            for v in show {
                if let Some(i) = mf.model.endo_index(v).filter(|&i| i < mf.model.orig_endo_nbr) {
                    crate::progress::note(format!("  {:<24} {:>12.4}", v, sm.smoothed[(nobs - 1, i)]));
                }
            }
        }
    }

    // ---- stream F: forecasts.run(..., 'smoother') at the mode (mh_replic = 0) ----
    let mut st = state(posterior_mean, posterior_median)?;
    st.mean_forecast = mean_forecast;
    st.point_forecast = point_forecast;
    st.smoothed_mean = smoothed_mean;
    let horizon = fc::forecast_horizon(o);
    if horizon > 0 && !has_mcmc && occ.is_some() {
        println!("WARNING: OccBin: the forecast after the OccBin smoother is not ported; skipped."); // stream OE
    } else if horizon > 0 && !has_mcmc {
        st.forecast =
            Some(fc::forecast_at_mode(mf, &st, &xmode, horizon, num(o, "conf_sig").unwrap_or(0.9), out_dir, fname)?);
    }
    // ---- end stream F ----

    std::fs::create_dir_all(out_dir)?;
    let rpath = out_dir.join(format!("{fname}_results.json"));
    std::fs::write(&rpath, serde_json::to_string_pretty(&results)?)?;
    crate::progress::note(format!("Results written to {}", rpath.display()));
    Ok((st, mcmc_plots, point_for_smoother))
}

/// Port of the sample loop of dynare_estimation.m (`nobs` or `first_obs` vectors): one
/// estimation per sample in `<dirname>_<k>`, then `oo_.RecursiveForecast` (see
/// [`crate::estimation::recursive`]). Returns the last estimation, as `oo_` after the loop.
#[allow(clippy::too_many_arguments)]
fn recursive_estimation(
    rec: &crate::estimation::recursive::Recursive,
    mf: &ModFile,
    dir: &Path,
    fname: &str,
    out_dir: &Path,
    o: &Options,
    var_list: &[String],
    ro: &RunOptions,
    ss_opts: &crate::analysis::StochSimulOptions,
    occbin: &mut crate::occbin::Session,
) -> Result<(fc::EstimationState, Option<crate::estimation::plots::McmcPlotContext>, Vec<f64>)> {
    use crate::estimation::recursive::SampleForecast;
    let horizon = fc::forecast_horizon(o);
    let mut last = None;
    let mut forecasts = Vec::new();
    for i in 0..rec.keys.len() {
        for run in rec.runs(o, i, fname) {
            last = Some(estimation(mf, dir, fname, out_dir, &run, var_list, ro, ss_opts, occbin)?);
        }
        let st = &last.as_ref().unwrap().0;
        if rec.by_nobs && horizon > 0 {
            // mh_replic > 0: oo_recursive_{k}.MeanForecast / PointForecast, else .forecast
            let f = match (&st.mean_forecast, &st.point_forecast, &st.forecast) {
                (Some(m), Some(p), _) => Some(SampleForecast { mean: m.clone(), point: Some(p.clone()) }),
                (_, _, Some(f)) => Some(SampleForecast { mean: f.clone(), point: None }),
                _ => None,
            };
            forecasts.extend(f);
        }
    }
    let last = last.expect("at least one sample");
    if rec.by_nobs && horizon > 0 && forecasts.len() == rec.keys.len() {
        let st = &last.0;
        // dataset_info.rawdata of the last sample (before prefilter)
        let mut raw = st.data.y.transpose();
        if let Some(m) = &st.data.prefilter_mean {
            for (j, mj) in m.iter().enumerate() {
                raw.column_mut(j).add_scalar_mut(*mj);
            }
        }
        let n = mf.model.orig_endo_nbr;
        crate::estimation::recursive::recursive_forecast(
            rec,
            &forecasts,
            &st.var_list,
            &mf.model.endo_names[..n],
            &mf.varobs,
            &raw,
            horizon,
            &dir.join(rec.dname(rec.keys.len() - 1)).join("Output"),
            fname,
            &st.graphs,
        )?;
    }
    Ok(last)
}

/// [graphics] mode_check.m: slices of the posterior kernel around the mode (parallel
/// evaluations), `<fname>_CheckPlots<k>` and `<graphs>/<fname>_check_plot_data.json`.
#[allow(clippy::too_many_arguments)]
fn mode_check_step(
    post: &dyn Objective,
    dsge: &DsgeModel,
    xmode: &[f64],
    fmode: f64,
    hh: &Mat,
    o: &Options,
    fname: &str,
    gopts: &crate::graphics::GraphOptions,
) -> Result<()> {
    use crate::estimation::mode_check::{ModeCheckOptions, mode_check, print_mode_check_header};
    let mut mc = ModeCheckOptions::default();
    if let Some(v) = num(o, "mode_check_neighbourhood_size") {
        mc.neighbourhood_size = v;
    }
    if let Some(v) = num(o, "mode_check_symmetric_plots") {
        mc.symmetric_plots = v != 0.0;
    }
    if let Some(v) = num(o, "mode_check_number_of_points") {
        mc.number_of_points = v;
    }
    if let Some(v) = crate::estimation::est_completion::huge_number(o) {
        mc.huge_number = v; // ---- estimation completion: mode_check.m's options_.huge_number
    }
    let t = Instant::now();
    crate::progress::phase("mode_check");
    let names: Vec<String> =
        dsge.estimated_names.iter().map(|s| crate::estimation::plots::dynare_param_name(s)).collect();
    print_mode_check_header(fmode, Some(hh), &names, xmode);
    let prior_sd = dsge.prior_sd(); // NaN under ML (no priors)
    let log_prior = |x: &[f64]| dsge.log_prior(x);
    let slices = mode_check(post, &log_prior, xmode, &dsge.bounds, &prior_sd, &mc);
    let figs = crate::estimation::plots::mode_check_figures(&slices, &dsge.estimated_names, mc.nolik);
    let k = crate::estimation::plots::save_figures(&figs, fname, gopts)?;
    std::fs::create_dir_all(&gopts.dir)?;
    let data: serde_json::Map<String, serde_json::Value> = names
        .iter()
        .zip(&slices)
        .map(|(nm, s)| (nm.clone(), json!({"mode": s.mode, "z": s.z, "log_post": s.log_post, "log_lik": s.log_lik})))
        .collect();
    let p = gopts.dir.join(format!("{fname}_check_plot_data.json"));
    std::fs::write(&p, serde_json::to_string(&data)?)?;
    let npts: usize = slices.iter().map(|s| s.z.len()).sum();
    crate::progress::note(format!(
        "mode_check: {npts} evaluations, {k} figures ({fname}_CheckPlots*) and {} written ({}).",
        p.display(),
        hms(t.elapsed().as_secs_f64())
    ));
    Ok(())
}

/// Dynare looks for the datafile next to the .mod file; without an extension makedataset.m
/// tries m, mat, csv, xls, xlsx and requires exactly one to exist.
pub(crate) fn resolve_datafile(dir: &Path, f: &str) -> Result<PathBuf> {
    let p = dir.join(f);
    if p.extension().is_some() {
        return if p.exists() { Ok(p) } else { bail!("datafile {} not found", p.display()) };
    }
    let found: Vec<PathBuf> =
        crate::data::DATAFILE_EXTENSIONS.iter().map(|e| dir.join(format!("{f}.{e}"))).filter(|q| q.exists()).collect();
    match found.as_slice() {
        [one] => Ok(one.clone()),
        [] => bail!(
            "makedataset: I can't find a datafile {} (with allowed extension m, mat, csv, xls or xlsx)!",
            p.display()
        ),
        _ => bail!(
            "makedataset: no extension given for the datafile {} and more than one candidate is available",
            p.display()
        ),
    }
}

/// Reads a Dynare (or dynare-rust) mode file and maps it by parameter name onto our order
/// (port of utilities/estimation/check_mode_file.m): a file for another specification is
/// accepted with mode_compute = 0 only if it holds every estimated parameter; with
/// mode_compute > 0 the missing ones keep their initial values (`x_init`) and the Hessian
/// is dropped.
pub(crate) fn read_mode_file(
    dir: &Path,
    f: &str,
    names: &[String],
    mode_compute: i32,
    x_init: impl FnOnce() -> Result<Vec<f64>>,
) -> Result<(Vec<f64>, Option<Mat>)> {
    let mut p = dir.join(f);
    if p.extension().is_none() {
        p.set_extension("mat");
    }
    let vars = matfile::read(&p)?;
    let x = vars.get("xparam1").and_then(|v| v.as_vec()).ok_or_else(|| anyhow!("{}: no xparam1", p.display()))?;
    let file_names = vars.get("parameter_names").and_then(|v| v.as_strings());
    let hh = vars.get("hh").and_then(|v| v.as_matrix());
    let Some(file_names) = file_names else {
        if x.len() != names.len() {
            bail!("{}: {} parameters, the model estimates {}", p.display(), x.len(), names.len());
        }
        return Ok((x, hh));
    };
    // match_parameter_name_with_stderr: old mode files store stderr entries without the prefix
    let find = |nm: &str| {
        file_names
            .iter()
            .position(|f| f == nm)
            .or_else(|| nm.strip_prefix("stderr ").and_then(|bare| file_names.iter().position(|f| f == bare.trim())))
    };
    let pos: Vec<Option<usize>> = names.iter().map(|nm| find(nm)).collect();
    if pos.iter().all(Option::is_some) {
        let pos: Vec<usize> = pos.into_iter().flatten().collect();
        if file_names.len() != names.len() || pos.iter().enumerate().any(|(i, &k)| i != k) {
            println!(
                "The 'mode_file' {} has been generated using another specification of the model or another model!",
                p.display()
            );
        }
        let xm = pos.iter().map(|&i| x[i]).collect();
        let hh = hh
            .filter(|h| h.nrows() == file_names.len())
            .map(|h| Mat::from_fn(pos.len(), pos.len(), |i, j| h[(pos[i], pos[j])]));
        return Ok((xm, hh));
    }
    println!(
        "\nThe 'mode_file' {} has been generated using another specification of the model or another model!",
        p.display()
    );
    for (nm, k) in names.iter().zip(&pos) {
        if k.is_none() {
            println!(
                "--> Estimated parameter {nm} is not present in the loaded 'mode_file' (prior mean or initialized values will be used, if possible)."
            );
        }
    }
    for fname in &file_names {
        if !names.iter().any(|nm| nm == fname || nm.strip_prefix("stderr ") == Some(fname.as_str())) {
            println!("--> Parameter {fname} is not estimated according to the current mod file.");
        }
    }
    if mode_compute == 0 {
        bail!("Please change the 'mode_file' option, the list of estimated parameters or set 'mode_compute'>0.");
    }
    if pos.iter().all(Option::is_none) {
        bail!("Please remove the 'mode_file' option.");
    }
    let mut xm = x_init()?;
    for (i, k) in pos.iter().enumerate() {
        if let Some(k) = k {
            xm[i] = x[*k];
        }
    }
    Ok((xm, None))
}

fn prior_name(code: u8) -> &'static str {
    match code {
        1 => "beta",
        2 => "gamma",
        3 => "norm",
        4 => "invg",
        5 => "unif",
        6 => "invg2",
        8 => "weibl",
        _ => "?",
    }
}

/// `prior_dist_names.m`: the `pnames` of the console tables (`gamm`, `weibl`).
fn pnames(code: u8) -> &'static str {
    match code {
        1 => "beta",
        2 => "gamm",
        3 => "norm",
        4 => "invg",
        5 => "unif",
        6 => "invg2",
        8 => "weibl",
        _ => "",
    }
}

/// The names of the rows of the console tables (`display_estimation_results_table.m`,
/// `GetPosteriorParametersStatistics.m`): correlations are `x,y` without a space.
fn short_name(mf: &ModFile, dsge: &DsgeModel, k: usize) -> String {
    match dsge.estimated[k].kind {
        EstimatedKind::StdErr(i) => mf.model.exo_names[i].clone(),
        EstimatedKind::Corr(i, j) => format!("{},{}", mf.model.exo_names[i], mf.model.exo_names[j]),
        EstimatedKind::Param(i) => mf.model.param_names[i].clone(),
        EstimatedKind::MeStdErr(i) => mf.model.endo_names[i].clone(),
        EstimatedKind::MeCorr(i, j) => format!("{},{}", mf.model.endo_names[i], mf.model.endo_names[j]),
        EstimatedKind::Skew(i) => mf.model.exo_names[i].clone(),
        EstimatedKind::InitState(i) => mf.model.endo_names[i].clone(), // ---- estimation completion
    }
}

/// `row_header_width.m`.
fn row_header_width(mf: &ModFile, dsge: &DsgeModel) -> usize {
    let mut w = 0;
    if dsge.estimated.iter().any(|e| matches!(e.kind, EstimatedKind::Param(_))) {
        w = dsge.estimated_names.iter().map(|s| s.chars().count()).max().unwrap_or(0);
    }
    let m = &mf.model;
    for e in &dsge.estimated {
        let l = match e.kind {
            EstimatedKind::StdErr(i) | EstimatedKind::Skew(i) => m.exo_names[i].chars().count(),
            EstimatedKind::MeStdErr(i) => m.endo_names[i].chars().count(),
            EstimatedKind::Corr(i, j) => m.exo_names[i].chars().count() + m.exo_names[j].chars().count(),
            EstimatedKind::MeCorr(i, j) => m.endo_names[i].chars().count() + m.endo_names[j].chars().count(),
            _ => 0,
        };
        w = w.max(l);
    }
    w
}

/// The blocks of the tables in the order of `display_estimation_results_table.m` and
/// `GetPosteriorParametersStatistics.m`: parameters, shocks' standard deviations, measurement
/// errors' standard deviations, shocks' correlations, measurement errors' correlations,
/// skewness, initial states.
fn table_sections(dsge: &DsgeModel, init_title: &'static str) -> Vec<(&'static str, Vec<usize>)> {
    let pick = |f: fn(&EstimatedKind) -> bool| -> Vec<usize> {
        (0..dsge.estimated.len()).filter(|&k| f(&dsge.estimated[k].kind)).collect()
    };
    vec![
        ("parameters", pick(|k| matches!(k, EstimatedKind::Param(_)))),
        ("standard deviation of shocks", pick(|k| matches!(k, EstimatedKind::StdErr(_)))),
        ("standard deviation of measurement errors", pick(|k| matches!(k, EstimatedKind::MeStdErr(_)))),
        ("correlation of shocks", pick(|k| matches!(k, EstimatedKind::Corr(..)))),
        ("correlation of measurement errors", pick(|k| matches!(k, EstimatedKind::MeCorr(..)))),
        ("skewness of shocks", pick(|k| matches!(k, EstimatedKind::Skew(_)))),
        (init_title, pick(|k| matches!(k, EstimatedKind::InitState(_)))),
    ]
}

/// `display_estimation_results_table.m`: the posterior mode table (`bayes`) or the maximum
/// likelihood table.
fn print_estimates_table(mf: &ModFile, dsge: &DsgeModel, x: &[f64], sd: &[f64], bayes: bool) {
    use crate::analysis::table::fmt_f;
    println!("\nRESULTS FROM {} ESTIMATION", if bayes { "POSTERIOR" } else { "MAXIMUM LIKELIHOOD" });
    let w = row_header_width(mf, dsge);
    let tit1 = if bayes {
        format!("{:<w$} {:>10} {:>8} {:>7} {:>6} {:>6}", "", "prior mean", "mode", "s.d.", "prior", "pstdev")
    } else {
        format!("{:<w$} {:>10} {:>7} {:>6}", "", "Estimate", "s.d.", "t-stat")
    };
    for (title, idx) in table_sections(dsge, "initial states") {
        if idx.is_empty() {
            continue;
        }
        println!("{title}");
        println!("{tit1}\n");
        for k in idx {
            let name = short_name(mf, dsge, k);
            if bayes {
                let p = &dsge.priors[k];
                println!(
                    "{name:<w$} {} {} {} {:>6} {} ",
                    fmt_f(p.mean, 10, 4),
                    fmt_f(x[k], 8, 4),
                    fmt_f(sd[k], 7, 4),
                    pnames(p.shape.code()),
                    fmt_f(p.sd, 6, 4)
                );
            } else {
                println!(
                    "{name:<w$} {} {} {} ",
                    fmt_f(x[k], 10, 4),
                    fmt_f(sd[k], 7, 4),
                    fmt_f(x[k].abs() / sd[k], 7, 4)
                );
            }
        }
        println!();
    }
}

pub(crate) fn print_mode_table(mf: &ModFile, dsge: &DsgeModel, x: &[f64], sd: &[f64]) {
    print_estimates_table(mf, dsge, x, sd, true);
}

/// display_estimation_results_table.m, 'Maximum Likelihood' / 'mle'.
fn print_ml_table(mf: &ModFile, dsge: &DsgeModel, x: &[f64], sd: &[f64]) {
    print_estimates_table(mf, dsge, x, sd, false);
}

/// `GetPosteriorParametersStatistics.m`: the `ESTIMATION RESULTS` block with the log data
/// density (modified harmonic mean) and the posterior moments.
pub(crate) fn print_posterior_table(mf: &ModFile, dsge: &DsgeModel, s: &crate::estimation::PosteriorStats) {
    let sig = num(&mf.options.estimation_options(), "mh_conf_sig").unwrap_or(0.9);
    let short: Vec<String> = (0..dsge.estimated.len()).map(|k| short_name(mf, dsge, k)).collect();
    let spec = TableSpec {
        ifac_names: crate::gsa::gsa_param_names(mf, dsge),
        short,
        sections: table_sections(dsge, "initial state variables"),
        width: row_header_width(mf, dsge),
    };
    print_posterior_table_spec(&spec, &dsge.priors, sig, s);
}

/// The row names and blocks of the console tables of the posterior (`GetPosteriorParametersStatistics.m`,
/// `mcmc_diagnostics.m`) for estimations that have no [`DsgeModel`] (`method_of_moments`).
pub struct TableSpec {
    /// Names of the inefficiency factors' rows (`SE_ez`, `CC_a_b`, parameters).
    pub ifac_names: Vec<String>,
    /// Row names of the posterior table.
    pub short: Vec<String>,
    /// Blocks of the posterior table: title and indices.
    pub sections: Vec<(&'static str, Vec<usize>)>,
    /// `row_header_width`.
    pub width: usize,
}

pub(crate) fn print_posterior_table_spec(
    spec: &TableSpec,
    priors: &[crate::estimation::Prior],
    mh_conf_sig: f64,
    s: &crate::estimation::PosteriorStats,
) {
    let line = format!(
        "Log data density (Modified Harmonic Mean) is {:.6}.
",
        s.mhm.log_marginal
    );
    print_posterior_rows(spec, priors, mh_conf_sig, &line, &s.params);
}

/// `GetPosteriorParametersStatistics.m` for the SMC samplers (`hssmc`, `dsmh`, `dime_mcmc`,
/// `online`): `density` is `oo_.MarginalDensity.<sampler>`, `None` where Dynare prints no
/// log data density line.
pub(crate) fn print_smc_posterior_table(
    mf: &ModFile,
    dsge: &DsgeModel,
    params: &[crate::estimation::posterior::ParamMoments],
    mh_conf_sig: f64,
    density: Option<f64>,
) {
    let short: Vec<String> = (0..dsge.estimated.len()).map(|k| short_name(mf, dsge, k)).collect();
    let spec = TableSpec {
        ifac_names: Vec::new(),
        short,
        sections: table_sections(dsge, "initial state variables"),
        width: row_header_width(mf, dsge),
    };
    let line = density
        .map(|d| {
            format!(
                "Log data density is {d:.6}.

"
            )
        })
        .unwrap_or_else(|| {
            "
"
            .to_string()
        });
    print_posterior_rows(&spec, &dsge.priors, mh_conf_sig, &line, params);
}

fn print_posterior_rows(
    spec: &TableSpec,
    priors: &[crate::estimation::Prior],
    mh_conf_sig: f64,
    density_line: &str,
    params: &[crate::estimation::posterior::ParamMoments],
) {
    use crate::analysis::table::fmt_f;
    // num2str(options_.mh_conf_sig*100), '% HPD interval'
    let sig = mh_conf_sig * 100.0;
    let hpd = format!("{}% HPD interval", crate::gsa::stats::num2str((sig * 1e10).round() / 1e10));
    let w = spec.width;
    println!("\n\nESTIMATION RESULTS\n");
    print!("{density_line}");
    let tit2 =
        format!("{:<w$} {:>12} {:>12} {:>23} {:>8} {:>12}", "", "prior mean", "post. mean", hpd, "prior", "pstdev");
    let mut first = true;
    for (title, idx) in spec.sections.iter().cloned() {
        if idx.is_empty() {
            continue;
        }
        if !first {
            println!();
        }
        first = false;
        println!("{title}");
        println!("{tit2}\n");
        for k in idx {
            let (p, pm) = (&priors[k], &params[k]);
            println!(
                "{:<w$} {} {} {} {} {:>8} {}",
                spec.short[k],
                fmt_f(p.mean, 12, 3),
                fmt_f(pm.mean, 12, 4),
                fmt_f(pm.hpd.0, 11, 4),
                fmt_f(pm.hpd.1, 11, 4),
                pnames(p.shape.code()),
                fmt_f(p.sd, 12, 4)
            );
        }
    }
}

/// The console output of `marginal_density.m`, rebuilt from the final variance inflation of
/// the weighting density (each failed coverage check inflates it by 20%).
pub(crate) fn print_marginal_density_messages(m: &crate::estimation::posterior::MhmResult) {
    println!("marginal density: I'm computing the posterior mean and covariance...  Done!");
    print!("marginal density: I'm computing the posterior log marginal density (modified harmonic mean)... ");
    let failures = (m.increase.ln() / 1.2f64.ln()).round() as usize;
    for i in 0..failures {
        println!();
        if i == 0 {
            println!("marginal density: The support of the weighting density function is not large enough...");
            println!("marginal density: I increase the variance of this distribution.");
        } else {
            println!("marginal density: Let me try again.");
        }
    }
    if m.increase > 20.0 {
        println!("marginal density: There's probably a problem with the modified harmonic mean estimator.");
    } else {
        println!("Done!");
    }
}

/// `mcmc_diagnostics.m`: the inefficiency factors per block (`dyntable`).
pub(crate) fn print_ifac(mf: &ModFile, dsge: &DsgeModel, ifac: &Mat) {
    print_ifac_names(&crate::gsa::gsa_param_names(mf, dsge), ifac);
}

pub(crate) fn print_ifac_names(names: &[String], ifac: &Mat) {
    let mut headers = vec!["Parameter".to_string()];
    headers.extend((0..ifac.nrows()).map(|b| format!("Block {}", b + 1)));
    let lh = crate::analysis::table::max_len(names) + 2;
    print!(
        "{}",
        crate::analysis::table::dyntable(
            "MCMC Inefficiency factors per block",
            &headers,
            names,
            &ifac.transpose(),
            lh,
            12,
            3
        )
    );
    println!();
}

pub(crate) fn write_posterior_csv(
    out_dir: &Path,
    fname: &str,
    dsge: &DsgeModel,
    xmode: &[f64],
    sd_mode: &[f64],
    s: &crate::estimation::PosteriorStats,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let mut w = String::from(
        "parameter,prior,prior_mean,prior_sd,mode,mode_sd,post_mean,post_median,post_sd,hpd_lower,hpd_upper\n",
    );
    for (k, pm) in s.params.iter().enumerate() {
        let p = &dsge.priors[k];
        w += &format!(
            "{},{},{},{},{},{},{},{},{},{},{}\n",
            dsge.estimated_names[k],
            prior_name(p.shape.code()),
            p.mean,
            p.sd,
            xmode[k],
            sd_mode[k],
            pm.mean,
            pm.median,
            pm.var.sqrt(),
            pm.hpd.0,
            pm.hpd.1
        );
    }
    std::fs::write(out_dir.join(format!("{fname}_posterior.csv")), w)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// smoother (DsgeSmoother.m, full state space)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SmoothedSeries {
    /// nobs x orig_endo_nbr, levels (steady state added back).
    pub smoothed: Mat,
    /// nobs x n_exo.
    pub shocks: Mat,
    /// nobs x orig_endo_nbr, one-step-ahead predictions (Dynare's FilteredVariables).
    pub filtered: Mat,
    /// nobs x nvobs smoothed measurement errors (oo_.SmoothedMeasurementErrors), if H != 0.
    pub measurement_errors: Option<Mat>,
}

/// DsgeSmoother at parameter vector `x`: full (unrestricted) state space in DR order.
pub fn run_smoother(
    dsge: &DsgeModel,
    data: &LikelihoodData,
    x: &[f64],
    state_uncertainty: bool,
) -> Result<SmoothedSeries> {
    Ok(run_smoother_request(dsge, data, x, state_uncertainty, None)?.0)
}

/// [`run_smoother`] with the k-step-ahead objects of `req` (filter_step_ahead,
/// filter_covariance, filter_decomposition); also returns the order_var of the DR and the
/// level constant of each endogenous variable.
pub fn run_smoother_request(
    dsge: &DsgeModel,
    data: &LikelihoodData,
    x: &[f64],
    state_uncertainty: bool,
    req: Option<&crate::kalman::SmootherRequest>,
) -> Result<(SmoothedSeries, Option<crate::kalman::StepAhead>, Vec<usize>, Vec<f64>)> {
    // DsgeSmoother.m (crate::forecast) + store_smoother_results.m: + log(ys) with loglinear,
    // + the sample mean for the observables with prefilter, + the observation trend
    let mut sm = crate::forecast::dsge_smoother_request(dsge, data, x, state_uncertainty, req)?;
    let norig = dsge.model.orig_endo_nbr;
    let vars: Vec<usize> = (0..norig).collect();
    let series = SmoothedSeries {
        smoothed: sm.levels(&sm.out.alphahat, &vars),
        shocks: sm.out.etahat.transpose(),
        filtered: sm.filtered_levels(&vars),
        measurement_errors: sm.resolved.h.iter().any(|&v| v != 0.0).then(|| sm.out.epsilonhat.transpose()),
    };
    let step_ahead = sm.out.step_ahead.take();
    sm.constant.truncate(norig);
    Ok((series, step_ahead, sm.resolved.dr.inv_order_var.clone(), sm.constant))
}

fn write_series_csv(path: &Path, names: &[String], m: &Mat, dates: Option<&[String]>) -> Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut w = String::new();
    w += "date,";
    w += &names.join(",");
    w.push('\n');
    for i in 0..m.nrows() {
        w += dates.map_or(&(i + 1).to_string(), |d| &d[i]);
        for j in 0..m.ncols() {
            w += &format!(",{}", m[(i, j)]);
        }
        w.push('\n');
    }
    std::fs::write(path, w).with_context(|| format!("writing {}", path.display()))
}

pub(crate) fn hms(secs: f64) -> String {
    let s = secs.round() as u64;
    if secs < 60.0 { format!("{secs:.1}s") } else { format!("{}h{:02}m{:02}s", s / 3600, (s / 60) % 60, s % 60) }
}

/// stoch_simul.m: `if info(1), print_info(info, options_.noprint, options_); return; end`
/// (`print_info` raises the error only without `noprint`): with `noprint`, a failure of the
/// steady state or of the decision rule (a Dynare info code) ends the command silently.
fn noprint_info<T>(r: Result<T>, noprint: bool) -> Result<Option<T>> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(e)
            if noprint
                && e.chain().any(|c| {
                    c.downcast_ref::<crate::solver::SolveError>().is_some()
                        || c.to_string().starts_with("steady state at the calibration failed (info ")
                }) =>
        {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}
