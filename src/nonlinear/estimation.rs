// SPDX-License-Identifier: GPL-3.0-or-later
//! `estimation(order = 2|3, ...)`: the nonlinear-filter branch of `dynare_estimation_1.m`
//! (objective `non_linear_dsge_likelihood`): initial checks, mode finding
//! (`mode_compute` 0/4/5/6/7), Hessian, mode file, Laplace approximation, random-walk
//! Metropolis–Hastings or the DSMH sampler (`posterior_sampling_method = 'dsmh'`), posterior
//! statistics. Dynare runs no smoother after a nonlinear estimation (no particle smoother).

use super::dsmh::{DsmhOptions, dsmh};
use super::likelihood::{NonlinearModel, NonlinearPosterior};
use super::{FilterAlgorithm, ParticleOptions, requested_order};
use crate::driver::{
    RunOptions, hms, print_ifac, print_mode_table, print_posterior_table, read_mode_file, write_posterior_csv,
};
use crate::dsge::DsgeModel;
use crate::estimation::mh_session;
use crate::estimation::samplers::{SamplerOptions, prior_mode_start};
use crate::estimation::{
    GmhmaxlikOptions, MhOptions, MhProgress, Objective, PosteriorOptions, TuneOptions, gmhmaxlik, hessian,
    inverse_hessian, laplace_approximation, mcmc_bounds, mh_tune_jscale, posterior_moments, posterior_stats,
};
use crate::linalg::Mat;
use crate::matfile::{self, MatOut};
use crate::modfile::{ModFile, OptValue, Options, opt};
use anyhow::{Result, anyhow, bail};
use serde_json::json;
use std::path::Path;
use std::time::Instant;

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

fn flag(o: &Options, k: &str) -> bool {
    match opt(o, k) {
        Some(OptValue::Flag) => true,
        Some(OptValue::Num(v)) => *v != 0.0,
        _ => false,
    }
}

fn string(o: &Options, k: &str) -> Option<String> {
    match opt(o, k) {
        Some(OptValue::Str(s)) => Some(s.clone()),
        Some(OptValue::List(v)) if v.len() == 1 => match &v[0] {
            OptValue::Str(s) => Some(s.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Options this branch understands (others draw a warning).
const SUPPORTED: &[&str] = &[
    "datafile",
    "first_obs",
    "nobs",
    "presample",
    "order",
    "mode_compute",
    "mode_file",
    "mh_replic",
    "mh_nblocks",
    "mh_jscale",
    "mh_tune_jscale",
    "mh_tune_guess",
    "mh_drop",
    "mh_conf_sig",
    "mh_init_scale_factor",
    "mh_init_scale",
    "prior_trunc",
    "cova_compute",
    "optim",
    "filter_algorithm",
    "number_of_particles",
    "resampling",
    "resampling_threshold",
    "resampling_method",
    "proposal_approximation",
    "distribution_approximation",
    "cpf_weights",
    "particle_filter_options",
    "posterior_sampling_method",
    "posterior_sampler_options",
    "graph_format",
    "nodisplay",
    "nograph",
    "plot_priors",
    "nointeractive",
    "console",
    "noprint",
    "nodiagnostic",
    "qz_criterium",
    "nonlinear_filter_initialization", // ---- estimation completion
    "prefilter",                       // refused by initial_estimation_checks.m
    "loglinear",                       // refused by non_linear_dsge_likelihood.m
];

/// `posterior_sampler_options = ('H', 25, 'N', 200, ...)` for DSMH.
pub fn dsmh_options(o: &Options, seed: u64, quiet: bool) -> Result<DsmhOptions> {
    let mut d = DsmhOptions { seed, verbose: !quiet, ..Default::default() };
    if let Some(OptValue::List(items)) = opt(o, "posterior_sampler_options") {
        let mut it = items.iter();
        while let Some(k) = it.next() {
            let OptValue::Str(k) = k else { bail!("posterior_sampler_options: expected 'name', value pairs") };
            let k = k.trim_matches(|c| c == '\'' || c == '"');
            let v = match it.next() {
                Some(OptValue::Num(v)) => *v,
                _ => bail!("posterior_sampler_options: {k} needs a number"),
            };
            // check_posterior_sampler_options.m: case-sensitive keys
            match k {
                "H" => d.h = v as usize,
                "M" => d.m = v as usize,
                "N" => d.n = v as usize,
                "G" => d.g = v as usize,
                "K" => d.k = v as usize,
                "tau" => d.tau = v as usize,
                "alpha0" => d.alpha0 = v,
                "alpha1" => d.alpha1 = v,
                other => println!("WARNING: dsmh: Unknown option ({other})!"),
            }
        }
    }
    let ng = d.n * d.g;
    // MATLAB `mod(N*G, M)` (`mod(x, 0) = x`)
    let rem = if d.m == 0 { ng } else { ng % d.m };
    if rem != 0 {
        bail!("check_posterior_sampler_options:: DSMH requires M to divide N*G = {ng} evenly.");
    }
    Ok(d)
}

/// Runs the estimation; returns the point later commands use (posterior mean, else mode).
pub fn run(mf: &ModFile, dir: &Path, fname: &str, out_dir: &Path, o: &Options, ro: &RunOptions) -> Result<Vec<f64>> {
    for (k, _) in o {
        if !SUPPORTED.contains(&k.as_str()) && !mh_session::OPTIONS.contains(&k.as_str()) {
            println!("WARNING: estimation option `{k}` is not supported with a nonlinear filter and is ignored.");
        }
    }
    let order = requested_order(o);
    let particle = ParticleOptions::from_options(o)?;
    println!("\nEstimation using a non linear filter!\n");
    crate::progress::note(format!(
        "order = {order}, filter = {} ({}), {} particles",
        format!("{:?}", particle.algorithm).to_lowercase(),
        particle.algorithm.matlab_name(),
        particle.number_of_particles
    ));

    // ---- data -------------------------------------------------------------
    let ds = crate::dataset::Dataset::load(mf, dir, o)?; // makedataset.m
    let (datapath, first_obs, sample) = (ds.path.clone(), ds.first_obs, ds.sample(&mf.varobs)?);
    let nobs = sample.nrows();
    let presample = num(o, "presample").unwrap_or(0.0) as usize;
    crate::progress::note(format!(
        "Data: {} ({} observables, {nobs} observations from first_obs = {first_obs}).",
        datapath.display(),
        mf.varobs.len()
    ));

    // ---- model and initial values -----------------------------------------------
    let names: Vec<String> = mf.estimated_params_dynare_order().iter().map(|e| mf.estimated_param_name(e)).collect();
    let (x_file, hh_file) = match string(o, "mode_file") {
        Some(f) => {
            let mc = num(o, "mode_compute").unwrap_or(5.0) as i32; // default_option_values.m
            let (x, hh) = read_mode_file(dir, &f, &names, mc, || crate::dsge::initial_vector(mf))?; // check_mode_file.m (stream M)
            (Some(x), hh)
        }
        None => (None, None),
    };
    let x_init = match &x_file {
        Some(x) => x.clone(),
        None => crate::dsge::initial_vector(mf)?,
    };
    let dsge = DsgeModel::new(mf, Some(&x_init))?;
    // dynare_estimation_init.m: prefiltered data need zero-mean measurement equations
    let prefilter = num(o, "prefilter").is_some_and(|v| v != 0.0);
    if prefilter && dsge.varobs.iter().any(|&v| dsge.level_constant(v, &dsge.ys_guess).abs() >= 1e-9) {
        bail!("The option \"prefilter\" is inconsistent with the non-zero mean measurement equations in the model.");
    }
    let mut nm = NonlinearModel::new(dsge, order, particle, sample.transpose(), presample, ro.seed)?;
    // select_qz_criterium_value.m (non linear filter branch), for resol and lyapunov_symm
    nm.set_qz_criterium(num(o, "qz_criterium"))?;
    // initial_estimation_checks.m (missing observations: NonlinearModel::new)
    if num(o, "prefilter") == Some(1.0) {
        bail!("initial_estimation_checks:: particle filtering does not support the prefilter option");
    }
    if !mf.observation_trends.is_empty() {
        bail!("initial_estimation_checks:: particle filtering does not support trends");
    }
    // ---- estimation completion: non_linear_dsge_likelihood.m's `otherwise` branch
    if !(1..=3).contains(&nm.particle.initialization) {
        bail!("Unknown initialization option!");
    }
    // non_linear_dsge_likelihood.m, at its first call (initial_estimation_checks.m)
    if flag(o, "loglinear") {
        bail!("non_linear_dsge_likelihood: It is not possible to use a non linear filter with the option loglinear!");
    }
    let nm = nm;
    let dsge = &nm.dsge;
    let n = dsge.n_estimated();
    let post = NonlinearPosterior { model: &nm, bounds: &dsge.bounds };
    let t = Instant::now();
    let (f0, info0) = post.eval(&x_init);
    if !f0.is_finite() {
        bail!("initial_estimation_checks: the posterior is not finite at the initial values (info {info0}).");
    }
    println!("Initial value of the log posterior (or likelihood): {}", crate::gsa::stats::num2str(-f0));
    crate::progress::note(format!("({:.2} ms per evaluation)", t.elapsed().as_secs_f64() * 1e3));
    let mut results = json!({
        "model": fname,
        "order": order,
        "filter": nm.particle.algorithm.matlab_name(),
        "number_of_particles": nm.particle.number_of_particles,
        "parameter_names": dsge.estimated_names,
        "initial_minus_log_posterior": f0,
    });
    std::fs::create_dir_all(out_dir)?;
    let rpath = out_dir.join(format!("{fname}_results.json"));

    // ---- DSMH (dynare_estimation_1.m: issmc -> no mode finding, no MCMC) ---------
    if super::smc_run::smc_method(o).is_some() {
        // dsmh, hssmc, dime_mcmc, online
        return super::smc_run::run_smc(mf, &post, dsge, nobs, o, ro, out_dir, fname, &mut results, Some(&nm), &x_init);
    }

    // ---- mode finding ------------------------------------------------------------
    let mode_compute = num(o, "mode_compute").unwrap_or(5.0) as i32; // default_option_values.m (newrat)
    if [1, 3, 4].contains(&mode_compute) && nm.particle.algorithm != FilterAlgorithm::Gf {
        println!("You are using a gradient-based mode-finder. Particle filtering introduces discontinuities in the");
        println!("objective function w.r.t the parameters. Thus, should use a non-gradient based optimizer.");
    }
    let mut mh_jscale = num(o, "mh_jscale").unwrap_or(2.38 / (n as f64).sqrt());
    // every posterior sampler, and Dynare's mh files (load_mh_file, mh_recover, ...)
    let sampler =
        SamplerOptions::from_options(opt(o, "posterior_sampling_method"), opt(o, "posterior_sampler_options"))?
            .with_estimation_optim(opt(o, "optim"))
            .with_priors(&dsge.priors);
    let mfo = mh_session::MhFileOptions::from_options(o, &sampler, dir)?;
    let mode_estim = sampler.mh_posterior_mode_estimation() || mfo.mh_posterior_mode_estimation;
    // dynare_estimation_1.m: cova_compute = 0 computes no Hessian at the mode (stdh = NaN)
    let cova_compute = num(o, "cova_compute").unwrap_or(1.0) != 0.0;
    let (xmode, fmode, hh) = if mode_estim {
        println!("No posterior maximization (mh_posterior_mode_estimation); the MCMC starts from the prior mode.");
        let x = prior_mode_start(&dsge.priors, &dsge.bounds);
        let f = post.eval(&x).0;
        let hh = Mat::from_diagonal(&nalgebra::DVector::from_iterator(
            n,
            dsge.priors.iter().map(|p| if (p.sd * p.sd).is_finite() { 1.0 / (p.sd * p.sd) } else { 1.0 }),
        ));
        (x, f, hh)
    } else if mode_compute == 0 {
        let hh = match hh_file {
            Some(h) => h,
            None if !cova_compute => Mat::from_element(n, n, f64::NAN),
            None => {
                println!("mode_compute = 0 and no Hessian in the mode file: computing it with hessian.m.");
                hessian(&post, &x_init, [1e-2, 1.0])
            }
        };
        (x_init.clone(), f0, hh)
    } else {
        let t = Instant::now();
        crate::progress::phase(format!("mode_compute = {mode_compute}"));
        let (x, fval, hh) = if mode_compute == 6 {
            let prior_sd = dsge.prior_sd();
            let mut gopts =
                GmhmaxlikOptions { silent: ro.quiet, seed: ro.seed, prefetch: ro.prefetch, ..Default::default() };
            if let Some(v) = opt(o, "optim") {
                gopts.apply_optim(v)?;
            }
            let res = gmhmaxlik(&post, &x_init, hh_file.as_ref(), mh_jscale, &dsge.bounds, &prior_sd, &gopts)?;
            mh_jscale = res.scale;
            crate::progress::note(format!(
                "mode_compute = 6: done ({} function evaluations, {}).",
                res.func_count,
                hms(t.elapsed().as_secs_f64())
            ));
            (res.x, res.fval, res.hessian)
        } else {
            // [stream O] dynare_minimize_objective.m: every mode_compute but 6
            let prior_sd = dsge.prior_sd();
            let ctx = crate::estimation::optim::MinimizeContext {
                bounds: &dsge.bounds,
                names: &dsge.estimated_names,
                prior_sd: &prior_sd,
                optim: opt(o, "optim"),
                verbose: !ro.quiet,
                seed: ro.seed,
                stream: u64::MAX - 1,
            };
            let res = crate::estimation::optim::minimize(mode_compute, &post, &x_init, &ctx)?;
            crate::progress::note(format!(
                "mode_compute = {mode_compute}: {} ({} iterations, {} function evaluations, {}).",
                res.message,
                res.iterations,
                res.func_count,
                hms(t.elapsed().as_secs_f64())
            ));
            let h = if cova_compute { hessian(&post, &res.x, [1e-2, 1.0]) } else { Mat::from_element(n, n, f64::NAN) };
            (res.x, res.fval, h)
        };
        println!("\nFinal value of minus the log posterior (or likelihood):{fval:.6} ");
        let mode_path = out_dir.join(format!("{fname}_mode.mat"));
        if cova_compute || mode_compute == 6 {
            matfile::write(
                &mode_path,
                &[
                    ("xparam1", MatOut::Vector(&x)),
                    ("hh", MatOut::Matrix(&hh)),
                    ("fval", MatOut::Scalar(fval)),
                    ("parameter_names", MatOut::Strings(&dsge.estimated_names)),
                ],
            )?;
        } else {
            matfile::write(
                &mode_path,
                &[
                    ("xparam1", MatOut::Vector(&x)),
                    ("fval", MatOut::Scalar(fval)),
                    ("parameter_names", MatOut::Strings(&dsge.estimated_names)),
                ],
            )?;
        }
        crate::progress::note(format!("Mode saved to {}", mode_path.display()));
        (x, fval, hh)
    };
    let invhess = inverse_hessian(&hh);
    let sd_mode: Vec<f64> = match &invhess {
        _ if !cova_compute => vec![f64::NAN; n],
        Some(v) => (0..n).map(|i| v[(i, i)].max(0.0).sqrt()).collect(),
        None => vec![f64::NAN; n],
    };
    let laplace = if mode_estim {
        f64::NAN
    } else if !cova_compute {
        print_mode_table(mf, dsge, &xmode, &sd_mode);
        f64::NAN
    } else {
        print_mode_table(mf, dsge, &xmode, &sd_mode);
        let laplace = laplace_approximation(&hh, fmode);
        println!("\nLog data density [Laplace approximation] is {laplace:.6}.\n");
        laplace
    };
    results["mode"] = json!(xmode);
    results["mode_sd"] = json!(sd_mode);
    results["minus_log_posterior_at_mode"] = json!(fmode);
    results["log_data_density_laplace"] = json!(laplace);
    let mut point = xmode.to_vec();

    // ---- Metropolis–Hastings ------------------------------------------------------------
    let mh_replic = ro.mh_replic.unwrap_or(num(o, "mh_replic").unwrap_or(20000.0) as usize);
    if mh_replic > 0 || mfo.load_mh_file {
        let dname = dir.join(fname);
        let invhess = if mfo.load_mh_file && mfo.use_mh_covariance_matrix {
            mh_session::previous_mh_covariance(&dname, fname, out_dir, &dsge.estimated_names)?.0
        } else if mh_replic == 0 {
            invhess.unwrap_or_else(|| Mat::identity(n, n))
        } else {
            invhess.ok_or_else(|| anyhow!("the Hessian at the mode is not invertible: cannot start the MCMC"))?
        };
        let mh_bounds = mcmc_bounds(&xmode, &dsge.priors, crate::dsge::prior_trunc(mf))?;
        let mh_post = NonlinearPosterior { model: &nm, bounds: &mh_bounds };
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
        mh.progress_every = 100;
        let mut jscale = mh_jscale;
        if (flag(o, "mh_tune_jscale") || num(o, "mh_tune_jscale").is_some()) && mh_replic > 0 {
            let target = num(o, "mh_tune_jscale").unwrap_or(0.33);
            let guess = num(o, "mh_tune_guess").or(num(o, "mh_jscale")).unwrap_or(mh.mh_jscale);
            jscale = mh_tune_jscale(
                &mh_post,
                &invhess,
                &xmode,
                &mh_bounds,
                &TuneOptions { target, guess: Some(guess), seed: ro.seed, prefetch: ro.prefetch, ..Default::default() },
            )?;
            println!("mh_jscale has been set equal to {}", crate::gsa::stats::num2str(jscale));
        }
        mh.mh_jscale = jscale;
        mh.jscale = vec![jscale; n];
        crate::progress::note(format!(
            "Estimation::mcmc ({}): {} chains x {} draws.",
            sampler.title(),
            mh.nblocks,
            mh.replic
        ));
        crate::progress::phase(format!("MCMC: {} chains x {} draws", mh.nblocks, mh.replic));
        crate::progress::chains(mh.nblocks, mh.replic);
        let t = Instant::now();
        let print_every = (mh_replic / 20).max(100);
        let progress = |p: &MhProgress| {
            crate::progress::chain(p.chain, p.draws_done, p.acceptance);
            if p.chain == 0 && p.draws_done.is_multiple_of(print_every) {
                println!(
                    "  chain 1: {:>9}/{} draws, acceptance {:5.2}%  [{}]",
                    p.draws_done,
                    p.total,
                    100.0 * p.acceptance,
                    hms(t.elapsed().as_secs_f64())
                );
            }
        };
        let res = mh_session::run(
            mh_session::SessionInputs {
                disp: "Estimation::mcmc",
                obj: &mh_post,
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
                keep_all: false,
                progress: Some(&progress),
            },
            &mfo,
        )?;
        let secs = t.elapsed().as_secs_f64();
        crate::progress::note(format!("Estimation::mcmc: done in {}.", hms(secs)));
        if mfo.mh_posterior_mode_estimation {
            let (m, sd, kernel) = mh_session::posterior_mode_from_mcmc(&dname, fname, out_dir, &dsge.estimated_names)?;
            println!(
                "\nPosterior mode from the MCMC (log posterior kernel {kernel:.6}), saved to {fname}_mh_mode.mat:"
            );
            print_mode_table(mf, dsge, &m, &sd);
            results["mh_posterior_mode"] = json!({"mode": m, "sd": sd, "log_posterior_kernel": kernel});
            std::fs::write(&rpath, serde_json::to_string_pretty(&results)?)?;
            return Ok(m);
        }
        let stats = posterior_stats(
            &res.chains,
            &PosteriorOptions { mh_drop, mh_conf_sig: num(o, "mh_conf_sig").unwrap_or(0.9), ..Default::default() },
        )?;
        print_ifac(mf, dsge, &stats.ifac);
        crate::driver::print_marginal_density_messages(&stats.mhm);
        print_posterior_table(mf, dsge, &stats);
        write_posterior_csv(out_dir, fname, dsge, &xmode, &sd_mode, &stats)?;
        results["mcmc"] = json!({
            "replic": res.record.total_draws(), "new_draws": res.new_draws, "nblocks": res.chains.len(),
            "jscale": res.jscale.first().copied().unwrap_or(jscale), "seconds": secs, "sampler": sampler.title(),
            "acceptance": stats.acceptance, "log_data_density_mhm": stats.mhm.log_marginal,
            "posterior_mean": stats.mean,
            "hpd_lower": stats.params.iter().map(|p| p.hpd.0).collect::<Vec<_>>(),
            "hpd_upper": stats.params.iter().map(|p| p.hpd.1).collect::<Vec<_>>(),
            "posterior_sd": stats.params.iter().map(|p| p.var.sqrt()).collect::<Vec<_>>(),
        });
        point = stats.mean.clone();
    }
    std::fs::write(&rpath, serde_json::to_string_pretty(&results)?)?;
    crate::progress::note(format!("Results written to {}", rpath.display()));
    Ok(point)
}

/// DSMH (`estimation/smc/dsmh.m`) on any posterior (nonlinear filters, or the linear Kalman
/// posterior from the driver): prints the table, writes `<fname>_dsmh_particles.csv` and
/// `results["dsmh"]` to `<fname>_results.json`; returns the posterior means.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_dsmh(
    mf: &ModFile,
    post: &dyn crate::estimation::Objective,
    dsge: &DsgeModel,
    nobs: usize,
    o: &Options,
    ro: &RunOptions,
    out_dir: &Path,
    fname: &str,
    results: &mut serde_json::Value,
) -> Result<Vec<f64>> {
    if dsge.priors.is_empty() {
        bail!("posterior_sampling_method = dsmh needs priors (not available for maximum likelihood)");
    }
    let n = dsge.n_estimated();
    std::fs::create_dir_all(out_dir)?;
    let rpath = out_dir.join(format!("{fname}_results.json"));
    let d = dsmh_options(o, ro.seed, ro.quiet)?;
    crate::progress::phase("DSMH sampler");
    let t = Instant::now();
    let r = dsmh(post, &dsge.priors, &dsge.bounds, nobs, &d)?;
    crate::progress::note(format!(
        "dsmh done in {}. Log marginal data density: {:.6}",
        hms(t.elapsed().as_secs_f64()),
        r.log_marginal_density
    ));
    let conf = num(o, "mh_conf_sig").unwrap_or(0.9);
    let np = r.particles.ncols();
    let mut means = Vec::with_capacity(n);
    let mut table = Vec::new();
    let mut stats = Vec::with_capacity(n);
    for k in 0..n {
        let mut xs: Vec<f64> = r.particles.row(k).iter().copied().collect();
        let m = posterior_moments(&mut xs, conf);
        means.push(m.mean);
        table.push(json!({"name": dsge.estimated_names[k], "mean": m.mean, "median": m.median, "sd": m.var.sqrt(),
            "hpd_lower": m.hpd.0, "hpd_upper": m.hpd.1}));
        stats.push(m);
    }
    crate::driver::print_smc_posterior_table(mf, dsge, &stats, conf, Some(r.log_marginal_density));
    let csv: String = std::iter::once(dsge.estimated_names.join(","))
        .chain((0..np).map(|j| (0..n).map(|k| format!("{:.17e}", r.particles[(k, j)])).collect::<Vec<_>>().join(",")))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(out_dir.join(format!("{fname}_dsmh_particles.csv")), csv + "\n")?;
    // oo_.MarginalDensity.dsmh, read by model_comparison
    results["MarginalDensity"] = json!({ "dsmh": r.log_marginal_density });
    results["dsmh"] = json!({
        "log_marginal_density": r.log_marginal_density, "lambda": r.lambda, "ess": r.ess, "c": r.c,
        "acceptance": r.acceptance, "posterior": table, "seconds": t.elapsed().as_secs_f64(),
    });
    std::fs::write(&rpath, serde_json::to_string_pretty(&*results)?)?;
    crate::progress::note(format!("Results written to {}", rpath.display()));
    Ok(means)
}
