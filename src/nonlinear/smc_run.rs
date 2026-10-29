// SPDX-License-Identifier: GPL-3.0-or-later
//! the driver side of the SMC samplers (`dynare_estimation_1.m`: `issmc` -> no mode
//! finding, no MCMC): `posterior_sampling_method = 'dsmh' | 'hssmc' | 'dime_mcmc' | 'online'`
//! with their `posterior_sampler_options`, the posterior table
//! (`GetPosteriorParametersStatistics.m` on the final particles), the results file
//! (`MarginalDensity.<sampler>` for `model_comparison`) and Dynare's files
//! (`<dname>/hssmc/particles-<i>-<n>.mat`, `<dname>/dime/chains.mat`,
//! `<dname>/online/parameters_particles_final.mat`).

use super::likelihood::NonlinearModel;
use super::online::{OnlineOptions, online_auxiliary_filter};
use super::smc::{DimeOptions, HssmcOptions, dime, hssmc};
use crate::driver::{RunOptions, hms};
use crate::dsge::DsgeModel;
use crate::estimation::{Objective, posterior_moments};
use crate::linalg::Mat;
use crate::matfile::{self, MatValue};
use crate::modfile::{OptValue, Options, opt};
use anyhow::{Result, bail};
use serde_json::json;
use std::path::Path;
use std::time::Instant;

/// The SMC sampler requested, if any (lower case).
pub fn smc_method(o: &Options) -> Option<String> {
    let s = match opt(o, "posterior_sampling_method")? {
        OptValue::Str(s) => s.clone(),
        OptValue::List(v) if v.len() == 1 => match &v[0] {
            OptValue::Str(s) => s.clone(),
            _ => return None,
        },
        _ => return None,
    };
    let s = s.trim_matches(|c| c == '\'' || c == '"').to_lowercase();
    ["dsmh", "hssmc", "dime_mcmc", "online"].contains(&s.as_str()).then_some(s)
}

/// The model of `posterior_sampling_method = 'online'` at order 1 (the linear estimation
/// path; `None` for the other samplers): `solve_model_for_online_filter.m`'s `order < 2`
/// branch on the data `y` (nobs x pp). `options_.particle.status` is false at order 1, so
/// `particle_filter_options` is not read (`check_particle_filter_options.m` does not run).
pub fn online_model(
    mf: &crate::modfile::ModFile,
    x_init: &[f64],
    o: &Options,
    y: &Mat,
    seed: u64,
) -> Result<Option<NonlinearModel>> {
    if smc_method(o).as_deref() != Some("online") {
        return Ok(None);
    }
    let particle = super::ParticleOptions::from_preprocessor_options(o)?;
    let presample = match opt(o, "presample") {
        Some(OptValue::Num(v)) => *v as usize,
        _ => 0,
    };
    let dsge = DsgeModel::new(mf, Some(x_init))?;
    let mut nm = NonlinearModel::new(dsge, 1, particle, y.transpose(), presample, seed)?;
    nm.set_qz_criterium(match opt(o, "qz_criterium") {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    })?;
    Ok(Some(nm))
}

/// `options_.prior_trunc` as the online filter reads it (`online_auxiliary_filter.m:45,84`):
/// `default_option_values.m:508` sets 0, the full prior support.
pub fn online_prior_trunc(o: &Options) -> f64 {
    match opt(o, "prior_trunc") {
        Some(OptValue::Num(v)) => *v,
        _ => 0.0,
    }
}

/// `posterior_sampler_options = ('name', value, ...)` as (name, number) pairs.
fn key_values(o: &Options) -> Result<Vec<(String, f64)>> {
    let mut out = vec![];
    if let Some(OptValue::List(items)) = opt(o, "posterior_sampler_options") {
        let mut it = items.iter();
        while let Some(k) = it.next() {
            let OptValue::Str(k) = k else { bail!("posterior_sampler_options: expected 'name', value pairs") };
            let k = k.trim_matches(|c| c == '\'' || c == '"').to_string();
            let v = match it.next() {
                Some(OptValue::Num(v)) => *v,
                Some(OptValue::Str(s)) if s.trim_matches('\'').eq_ignore_ascii_case("true") => 1.0,
                Some(OptValue::Str(s)) if s.trim_matches('\'').eq_ignore_ascii_case("false") => 0.0,
                _ => bail!("posterior_sampler_options: {k} needs a number"),
            };
            out.push((k, v));
        }
    }
    Ok(out)
}

/// `GetPosteriorParametersStatistics.m` on equally weighted draws (`draws[k]` = parameter k):
/// prints the table, returns the means and the JSON rows.
fn report(
    mf: &crate::modfile::ModFile,
    dsge: &DsgeModel,
    draws: &[Vec<f64>],
    conf: f64,
    density: Option<f64>,
) -> (Vec<f64>, Vec<serde_json::Value>) {
    let mut means = vec![];
    let mut table = vec![];
    let mut stats = vec![];
    for (k, d) in draws.iter().enumerate() {
        let mut xs = d.clone();
        let m = posterior_moments(&mut xs, conf);
        means.push(m.mean);
        table.push(json!({"name": dsge.estimated_names[k], "mean": m.mean, "median": m.median, "sd": m.var.sqrt(),
            "hpd_lower": m.hpd.0, "hpd_upper": m.hpd.1}));
        stats.push(m);
    }
    crate::driver::print_smc_posterior_table(mf, dsge, &stats, conf, density);
    (means, table)
}

fn write_csv(p: &Path, names: &[String], cols: usize, get: impl Fn(usize, usize) -> f64) -> Result<()> {
    let csv: String = std::iter::once(names.join(","))
        .chain((0..cols).map(|j| (0..names.len()).map(|k| format!("{:.17e}", get(k, j))).collect::<Vec<_>>().join(",")))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(p, csv + "\n")?;
    Ok(())
}

/// Runs the SMC sampler of `posterior_sampling_method` on `post` (the posterior kernel);
/// `nm` is the nonlinear model (needed by `online`), `x_init` the initial parameters.
/// Returns the posterior means.
#[allow(clippy::too_many_arguments)]
pub fn run_smc(
    mf: &crate::modfile::ModFile,
    post: &dyn Objective,
    dsge: &DsgeModel,
    nobs: usize,
    o: &Options,
    ro: &RunOptions,
    out_dir: &Path,
    fname: &str,
    results: &mut serde_json::Value,
    nm: Option<&NonlinearModel>,
    x_init: &[f64],
) -> Result<Vec<f64>> {
    let method = smc_method(o).unwrap_or_default();
    if method == "dsmh" {
        return super::estimation::run_dsmh(mf, post, dsge, nobs, o, ro, out_dir, fname, results);
    }
    if dsge.priors.is_empty() {
        bail!("posterior_sampling_method = {method} needs priors (not available for maximum likelihood)");
    }
    let n = dsge.n_estimated();
    let dname = out_dir.parent().unwrap_or(out_dir).to_path_buf();
    std::fs::create_dir_all(out_dir)?;
    let conf = match opt(o, "mh_conf_sig") {
        Some(OptValue::Num(v)) => *v,
        _ => 0.9,
    };
    let kv = key_values(o)?;
    let t = Instant::now();
    let names = &dsge.estimated_names;
    let means = match method.as_str() {
        "hssmc" => {
            let mut h = HssmcOptions { seed: ro.seed, verbose: !ro.quiet, ..Default::default() };
            for (k, v) in &kv {
                match k.as_str() {
                    "target" => h.target = *v,
                    "steps" => h.steps = *v as usize,
                    "scale" => h.scale = *v,
                    "particles" => h.particles = *v as usize,
                    "lambda" => h.lambda = *v,
                    other => println!("WARNING: hssmc: Unknown option ({other})!"),
                }
            }
            crate::progress::phase("HSSMC sampler");
            let r = hssmc(post, &dsge.priors, &dsge.bounds, &h)?;
            crate::progress::note(format!("hssmc done in {}", hms(t.elapsed().as_secs_f64())));
            let folder = dname.join("hssmc");
            std::fs::create_dir_all(&folder)?;
            let mut vars = vec![
                ("particles", MatValue::matrix(&r.particles)),
                ("tlogpostkernel", MatValue::column(&r.tlogpostkernel)),
                ("loglikelihood", MatValue::column(&r.loglikelihood)),
            ];
            if !r.weights.is_empty() {
                vars.push(("weights", MatValue::column(&r.weights)));
            }
            let refs: Vec<(&str, &MatValue)> = vars.iter().map(|(k, v)| (*k, v)).collect();
            matfile::write_values(&folder.join(format!("particles-{}-{}.mat", h.steps, h.steps)), &refs)?;
            let draws: Vec<Vec<f64>> = (0..n).map(|k| r.particles.row(k).iter().copied().collect()).collect();
            let (means, table) = report(mf, dsge, &draws, conf, Some(r.log_marginal_density));
            write_csv(&out_dir.join(format!("{fname}_hssmc_particles.csv")), names, r.particles.ncols(), |k, j| {
                r.particles[(k, j)]
            })?;
            results["MarginalDensity"] = json!({ "hssmc": r.log_marginal_density });
            results["hssmc"] = json!({
                "log_marginal_density": r.log_marginal_density, "phi": r.phi, "ess": r.ess, "acceptance": r.acceptance,
                "scale": r.scale, "posterior": table, "seconds": t.elapsed().as_secs_f64(),
            });
            means
        }
        "dime_mcmc" => {
            let mut d = DimeOptions::new(n);
            d.seed = ro.seed;
            d.verbose = !ro.quiet;
            let mut tune_set = false;
            for (k, v) in &kv {
                match k.as_str() {
                    "nchain" => d.nchain = *v as usize,
                    "niter" => d.niter = *v as usize,
                    "tune" => {
                        d.tune = *v as usize;
                        tune_set = true;
                    }
                    "aimh_prob" => d.aimh_prob = *v,
                    "sigma" => d.sigma = *v,
                    "gamma" => d.gamma = Some(*v),
                    "df_proposal_dist" => d.df_proposal_dist = *v,
                    "rho" => d.rho = *v,
                    other => println!("WARNING: dime: Unknown option ({other})!"),
                }
            }
            if !tune_set {
                d.tune = 50000usize.div_ceil(d.nchain.max(1));
                println!("dime: setting number of ensemble iterations to keep (`tune`) to {}.", d.tune);
            }
            if d.nchain < 6 {
                bail!("check_posterior_sampler_options:: DIME requires nchain to be at least 6 (got {}).", d.nchain);
            }
            crate::progress::phase("DIME sampler");
            let r = dime(post, &dsge.priors, &dsge.bounds, &d)?;
            crate::progress::note(format!("dime done in {}", hms(t.elapsed().as_secs_f64())));
            // <dname>/dime/chains.mat: chains (niter x nchain x ndim), lprobs, tune
            let folder = dname.join("dime");
            std::fs::create_dir_all(&folder)?;
            let (ni, nc) = (r.chains.len(), d.nchain);
            let mut data = vec![0.0; ni * nc * n];
            for it in 0..ni {
                for c in 0..nc {
                    for k in 0..n {
                        data[it + ni * (c + nc * k)] = r.chains[it][c][k];
                    }
                }
            }
            let lp = Mat::from_fn(ni, nc, |i, c| r.lprobs[i][c]);
            matfile::write_values(
                &folder.join("chains.mat"),
                &[
                    ("chains", &MatValue::Numeric { dims: vec![ni, nc, n], data }),
                    ("lprobs", &MatValue::matrix(&lp)),
                    ("tune", &MatValue::scalar(d.tune as f64)),
                ],
            )?;
            let post_draws = r.posterior_draws();
            let draws: Vec<Vec<f64>> = (0..n).map(|k| post_draws.iter().map(|v| v[k]).collect()).collect();
            // compute_posterior_covariance_matrix.m + fill_mh_mode.m: the mode among all draws
            let (mode, kernel) = r.mode();
            let all: Vec<&Vec<f64>> = r.chains.iter().flatten().collect();
            let mu: Vec<f64> = (0..n).map(|k| all.iter().map(|v| v[k]).sum::<f64>() / all.len() as f64).collect();
            let cov = Mat::from_fn(n, n, |a, b| {
                all.iter().map(|v| (v[a] - mu[a]) * (v[b] - mu[b])).sum::<f64>() / (all.len() as f64 - 1.0)
            });
            crate::estimation::mh_files::write_mh_mode(out_dir, fname, &mode, &cov, kernel, names)?;
            let (means, table) = report(mf, dsge, &draws, conf, None);
            write_csv(&out_dir.join(format!("{fname}_dime_draws.csv")), names, post_draws.len(), |k, j| {
                post_draws[j][k]
            })?;
            results["dime"] = json!({
                "nchain": d.nchain, "niter": d.niter, "tune": d.tune, "posterior_mode": mode,
                "log_posterior_at_mode": kernel, "posterior": table, "seconds": t.elapsed().as_secs_f64(),
            });
            means
        }
        "online" => {
            let Some(nm) = nm else {
                bail!("posterior_sampling_method = online: the online filter needs its model (see online_model)");
            };
            let mut on = OnlineOptions { seed: ro.seed, verbose: !ro.quiet, ..Default::default() };
            for (k, v) in &kv {
                match k.as_str() {
                    "particles" => on.particles = *v as usize,
                    "liu_west_delta" => on.liu_west_delta = *v,
                    "liu_west_max_resampling_tries" => on.liu_west_max_resampling_tries = *v as usize,
                    "second_resampling" => on.second_resampling = *v != 0.0,
                    other => println!("WARNING: online: Unknown option ({other})!"),
                }
            }
            println!(
                "Estimation:online: Liu & West filter with {} particles, delta = {}",
                on.particles, on.liu_west_delta
            );
            crate::progress::phase("online (Liu & West) filter");
            // online_auxiliary_filter.m: prior_bounds / dprior with options_.prior_trunc
            // (default_option_values.m: 0, the full prior support)
            let prior_trunc = online_prior_trunc(o);
            let bounds = crate::estimation::prior_bounds(&dsge.priors, prior_trunc);
            let r = online_auxiliary_filter(nm, &dsge.priors, &bounds, x_init, &on)?;
            println!("Estimation:online: done in {}.", hms(t.elapsed().as_secs_f64()));
            let folder = dname.join("online");
            std::fs::create_dir_all(&folder)?;
            matfile::write_values(
                &folder.join("parameters_particles_final.mat"),
                &[("param", &MatValue::matrix(&r.param))],
            )?;
            // parameter trajectories (the figure data of online_auxiliary_filter.m)
            let tl = r.mean.ncols();
            let mut traj = String::from("period,parameter,lb95,mean,median,ub95,std\n");
            for t in 0..tl {
                for k in 0..n {
                    traj += &format!(
                        "{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}\n",
                        t + 1,
                        names[k],
                        r.lb95[(k, t)],
                        r.mean[(k, t)],
                        r.median[(k, t)],
                        r.ub95[(k, t)],
                        r.std[(k, t)]
                    );
                }
            }
            std::fs::write(folder.join(format!("{fname}_param_traj.csv")), traj)?;
            let draws: Vec<Vec<f64>> = (0..n).map(|k| r.param.row(k).iter().copied().collect()).collect();
            let (means, table) = report(mf, dsge, &draws, conf, None);
            write_csv(&out_dir.join(format!("{fname}_online_particles.csv")), names, r.param.ncols(), |k, j| {
                r.param[(k, j)]
            })?;
            results["online"] = json!({
                "particles": on.particles, "posterior": table, "seconds": t.elapsed().as_secs_f64(),
                "final_mean": (0..n).map(|k| r.mean[(k, tl - 1)]).collect::<Vec<_>>(),
            });
            means
        }
        other => bail!("posterior_sampling_method = {other} is not an SMC sampler"),
    };
    let rpath = out_dir.join(format!("{fname}_results.json"));
    std::fs::write(&rpath, serde_json::to_string_pretty(&*results)?)?;
    crate::progress::note(format!("Results written to {}", rpath.display()));
    Ok(means)
}
