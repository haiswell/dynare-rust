// SPDX-License-Identifier: GPL-3.0-or-later
//! The MCMC part of `+mom/run.m` (Bayesian IRF matching with `mh_replic > 0`):
//! `set_mcmc_prior_bounds`, optional `mh_tune_jscale`, `posterior_sampler` (through stream
//! S2's `mh_session::run`, mh files in `<dirname>/metropolis/`), `mcmc_diagnostics`,
//! `marginal_density` and `GetPosteriorParametersStatistics` on any [`Objective`].

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::collapsible_if
)] // index loops mirror the MATLAB code

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow};

use crate::estimation::convergence::ConvergenceOptions;
use crate::estimation::mh_session::{self, MhFileOptions, SessionInputs};
use crate::estimation::samplers::SamplerOptions;
use crate::estimation::{
    MhOptions, Objective, PosteriorOptions, PosteriorStats, Prior, TuneOptions, inverse_hessian, mcmc_bounds,
    mh_tune_jscale, posterior_stats,
};
use crate::linalg::Mat;
use crate::modfile::{OptValue, Options, opt};

/// MCMC options of `method_of_moments` handled here.
pub const OPTIONS: &[&str] = &[
    "mh_nblocks",
    "mh_jscale",
    "mh_drop",
    "mh_conf_sig",
    "mh_tune_jscale",
    "mh_tune_guess",
    "mh_init_scale_factor",
    "mh_init_scale",
    "nodiagnostic",
    "posterior_sampling_method",
    "posterior_sampler_options",
    "sub_draws",
    "taper_steps",
    "geweke_interval",
    "raftery_lewis_diagnostics",
    "raftery_lewis_qrs",
    "brooks_gelman_plotrows",
    "load_mh_file",
    "mh_recover",
    "mh_posterior_mode_estimation",
    "load_results_after_load_mh",
    "posterior_max_subsample_draws",
];

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

/// Runs the MCMC from the mode `xmode` with the Hessian `hh` (`invhess = inv(hh)`) when
/// `mh_replic > 0` (or `load_mh_file`); prints the posterior table, returns the statistics.
#[allow(clippy::too_many_arguments)]
pub fn mcmc_after_mode(
    obj: &dyn Objective,
    xmode: &[f64],
    hh: &Mat,
    priors: &[Prior],
    names: &[String],
    o: &Options,
    dir: &Path,
    dname: &Path,
    fname: &str,
    seed: u64,
    prefetch: usize,
    title: &str,
    spec: &crate::driver::TableSpec,
) -> Result<Option<(PosteriorStats, Vec<crate::estimation::Chain>)>> {
    mcmc_after_mode_js(obj, xmode, hh, priors, names, o, dir, dname, fname, seed, prefetch, title, spec, &[])
}

/// As [`mcmc_after_mode`] with the per-parameter `bayestopt_.jscale` of `estimated_params`
/// (`set_prior.m`; `None`/NaN entries become `options_.mh_jscale`, `dynare_estimation_init.m`
/// lines 459-463; an empty slice means no per-parameter scale).
#[allow(clippy::too_many_arguments)]
pub fn mcmc_after_mode_js(
    obj: &dyn Objective,
    xmode: &[f64],
    hh: &Mat,
    priors: &[Prior],
    names: &[String],
    o: &Options,
    dir: &Path,
    dname: &Path,
    fname: &str,
    seed: u64,
    prefetch: usize,
    title: &str,
    spec: &crate::driver::TableSpec,
    param_jscale: &[Option<f64>],
) -> Result<Option<(PosteriorStats, Vec<crate::estimation::Chain>)>> {
    let n = xmode.len();
    let mh_replic = num(o, "mh_replic").unwrap_or(0.0) as usize;
    let sampler =
        SamplerOptions::from_options(opt(o, "posterior_sampling_method"), opt(o, "posterior_sampler_options"))?
            .with_estimation_optim(opt(o, "optim"))
            .with_priors(priors);
    let mfo = MhFileOptions::from_options(o, &sampler, dir)?;
    if mh_replic == 0 && !mfo.load_mh_file {
        return Ok(None);
    }
    let invhess = inverse_hessian(hh)
        .ok_or_else(|| anyhow!("{title}: the Hessian at the mode is not invertible: cannot start the MCMC"))?;
    let prior_trunc = num(o, "prior_trunc").unwrap_or(1e-10);
    let bounds = mcmc_bounds(xmode, priors, prior_trunc)?;
    let mut mh = MhOptions::new(n);
    mh.replic = mh_replic;
    mh.nblocks = num(o, "mh_nblocks").unwrap_or(2.0) as usize;
    mh.seed = seed;
    mh.prefetch = prefetch;
    if let Some(v) = num(o, "mh_init_scale_factor").or(num(o, "mh_init_scale")) {
        mh.init_scale_factor = v;
    }
    let mh_drop = num(o, "mh_drop").unwrap_or(0.5);
    let keep_all = !flag(o, "nodiagnostic");
    mh.store_from = if keep_all { 0 } else { (mh_drop * mh_replic as f64).floor() as usize };
    let mut jscale = num(o, "mh_jscale").unwrap_or(2.38 / (n as f64).sqrt());
    let mut per_param = (0..n)
        .map(|i| param_jscale.get(i).copied().flatten().filter(|v| !v.is_nan()).unwrap_or(jscale))
        .collect::<Vec<_>>();
    if flag(o, "mh_tune_jscale") && mh_replic > 0 {
        let target = num(o, "mh_tune_jscale").unwrap_or(0.33);
        let guess = num(o, "mh_tune_guess").or(num(o, "mh_jscale")).unwrap_or(jscale);
        jscale = mh_tune_jscale(
            &obj,
            &invhess,
            xmode,
            &bounds,
            &TuneOptions { target, guess: Some(guess), seed, prefetch, ..Default::default() },
        )?;
        println!("mh_tune_jscale: mh_jscale has been set equal to {}.", crate::gsa::stats::num2str(jscale));
        // dynare_estimation_1.m: bayestopt_.jscale(:) = options_.mh_jscale
        per_param = vec![jscale; n];
    }
    mh.mh_jscale = jscale;
    mh.jscale = per_param;
    crate::progress::note(format!("{title}::mcmc ({}): {} chains x {} draws.", sampler.title(), mh.nblocks, mh.replic));
    let disp = format!("{title}::mcmc");
    let t = Instant::now();
    let res = mh_session::run(
        SessionInputs {
            obj,
            xparam1: xmode,
            invhess: &invhess,
            bounds: &bounds,
            mh: mh.clone(),
            sampler: &sampler,
            names,
            priors,
            dname,
            fname,
            mh_drop,
            keep_all,
            progress: None,
            disp: &disp,
        },
        &mfo,
    )?;
    crate::progress::note(format!("{title}::mcmc: done in {:.1}s.", t.elapsed().as_secs_f64()));
    let stats = posterior_stats(
        &res.chains,
        &PosteriorOptions { mh_drop, mh_conf_sig: num(o, "mh_conf_sig").unwrap_or(0.9), ..Default::default() },
    )?;
    // mcmc_diagnostics (inefficiency factors, Geweke, Brooks-Gelman), marginal_density,
    // GetPosteriorParametersStatistics
    if keep_all {
        let conv = ConvergenceOptions {
            last_file_lines: Some(res.record.last_line_number),
            ..ConvergenceOptions::from_options(o)
        };
        crate::driver::print_ifac_names(&spec.ifac_names, &stats.ifac);
        let gopts = crate::graphics::GraphOptions::from_options(o, dir, fname);
        crate::estimation::plots::mcmc_diagnostics(&res.chains, names, mh_drop, fname, &gopts, &conv)?;
    }
    crate::driver::print_marginal_density_messages(&stats.mhm);
    crate::driver::print_posterior_table_spec(spec, priors, num(o, "mh_conf_sig").unwrap_or(0.9), &stats);
    Ok(Some((stats, res.chains)))
}
