// SPDX-License-Identifier: GPL-3.0-or-later
//! Posterior distributions of the smoothed objects after an OccBin estimation
//! (`prior_posterior_statistics.m` with `options_.occbin.smoother.status`): for every posterior
//! sub-draw the OccBin smoother (`occbin.DSGE_smoother`, or the inversion-filter smoother) gives
//! the smoothed variables and shocks (and the one-step-ahead filtered variables with the PKF),
//! summarized as `pm3.m` does. With `forecast=h` the smoothed last state starts `occbin.forecast`
//! (`MeanForecast`: expected zero shocks; `PointForecast`: one draw of random shocks from
//! `Sigma_e`), as the OccBin branch of `prior_posterior_statistics_core.m` does (including its
//! use of `endo_histval - steady state` on a state that already is a deviation).

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::collapsible_if
)] // index loops mirror the MATLAB code

use std::path::Path;

use anyhow::{Result, anyhow, bail};
use rayon::prelude::*;

use crate::dsge::{DsgeModel, LikelihoodData};
use crate::estimation::mcmc::Chain;
use crate::estimation::prior_posterior::{number_of_subdraws, posterior_subsample, summarize, write_summary_csv};
use crate::linalg::Mat;
use crate::modfile::{OptValue, Options, opt};
use crate::occbin::estimation::{OccbinEstimation, OccbinPosterior};
use crate::occbin::pkf::PointModel;
use crate::solver::{StateSpaceIndex, kalman_transition_matrix};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// Smoothed variables (nobs x orig endo, levels), smoothed shocks (nobs x exo) and filtered
/// variables (None with the inversion filter) at `x`, without writing any file.
fn draw_series(
    dsge: &DsgeModel,
    data: &LikelihoodData,
    occ: &OccbinEstimation,
    x: &[f64],
) -> Result<(Mat, Mat, Option<Mat>, Vec<f64>)> {
    let norig = dsge.model.orig_endo_nbr;
    let nobs = data.y.ncols();
    if occ.smoother.inversion_filter {
        let post = OccbinPosterior { dsge, data, bounds: &dsge.bounds, occ };
        let res = dsge.resolve(x).map_err(|e| anyhow!("IVF smoother: no solution (info {})", e.info))?;
        let e = post.evaluate_ivf(x, res).map_err(|f| anyhow!("IVF smoother failed (info {})", f.info))?;
        let (states, innov) = e.ivf_states.ok_or_else(|| anyhow!("IVF smoother: no states"))?;
        let ys = &e.resolved.steady_state;
        let mut last = vec![0.0; ys.len()];
        for v in 0..states.nrows().min(last.len()) {
            last[v] = states[(v, nobs - 1)];
        }
        return Ok((Mat::from_fn(nobs, norig, |t, v| states[(v, t)] + ys[v]), innov.transpose(), None, last));
    }
    let sm = crate::occbin::smoother::occbin_smoother(dsge, data, occ, x)?;
    if sm.error_flag != 0 {
        bail!("OccBin smoother failed (error {})", sm.error_flag);
    }
    let mut inv = vec![0; sm.order_var.len()];
    for (k, &v) in sm.order_var.iter().enumerate() {
        inv[v] = k;
    }
    let ys = &sm.steady_state;
    // + trend_addition for the observables (one period ahead for the filtered variables)
    let trend_at = |ahead: bool, v: usize, t: usize| match (&sm.trend, dsge.varobs.iter().position(|&o| o == v)) {
        (Some((tr, tr1)), Some(i)) => {
            if ahead {
                tr1[(i, t)]
            } else {
                tr[(i, t)]
            }
        }
        _ => 0.0,
    };
    let level = |m: &Mat, ahead: bool| Mat::from_fn(nobs, norig, |t, v| m[(inv[v], t)] + ys[v] + trend_at(ahead, v, t));
    let mut last = vec![0.0; sm.order_var.len()];
    for (k, &v) in sm.order_var.iter().enumerate() {
        last[v] = sm.alphahat[(k, nobs - 1)];
    }
    Ok((level(&sm.alphahat, false), sm.etahat.transpose(), Some(level(&sm.filtered, true)), last))
}

/// `occbin.forecast` from the smoothed last state `last` (deviations, declaration order) at `x`:
/// the forecast without shocks (`yf`) and with one random-shock path (`yf1`), `horizon` x
/// original endogenous variables, in levels.
fn draw_forecast(
    dsge: &DsgeModel,
    occ: &OccbinEstimation,
    x: &[f64],
    last: &[f64],
    horizon: usize,
    seed: u64,
) -> Result<(Mat, Mat)> {
    let res = dsge.resolve(x).map_err(|e| anyhow!("forecast: no solution (info {})", e.info))?;
    let dr = &res.dr;
    let n = dr.order_var.len();
    let full = StateSpaceIndex::full(dr);
    let (t, r) = kalman_transition_matrix(dr, &full);
    let all: Vec<usize> = (0..n).collect();
    let ys = res.steady_state.clone();
    let mut pm =
        PointModel::new(&dsge.model, &occ.constraints, &occ.pswitch, res.params.clone(), ys.clone(), dr, &all, t, r);
    let ne = res.q.nrows();
    let mut so = occ.smoother_simul.clone();
    so.maxit = 30;
    so.check_ahead_periods = 30;
    so.periods = horizon;
    so.piecewise_only = false;
    so.init_regime = None;
    // M_.endo_histval = alphahat (a deviation); occbin.forecast then subtracts the steady state
    so.endo_init = (0..n).map(|v| last[v] - ys[v]).collect();
    so.shocks = Mat::zeros(horizon, ne);
    so.exo_pos = (0..ne).collect();
    let norig = dsge.model.orig_endo_nbr;
    // occbin.solver's paths already are levels, and the core adds the steady state again
    let lev = |m: &Mat| Mat::from_fn(horizon, norig, |t, v| m[(t, v)] + 2.0 * ys[v]);
    let mean = lev(&pm.solve(&so).map_err(|e| anyhow!("{e}"))?.piecewise);
    let ishock: Vec<usize> = (0..ne).filter(|&k| res.q[(k, k)].sqrt() != 0.0).collect();
    let k = ishock.len();
    let cov = Mat::from_fn(k, k, |i, j| res.q[(ishock[i], ishock[j])]);
    let svd = cov.svd(true, false);
    let mut us = svd.u.unwrap();
    for j in 0..k {
        let s = svd.singular_values[j].sqrt();
        for i in 0..k {
            us[(i, j)] *= s;
        }
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let draws = Mat::from_fn(k, horizon, |_, _| StandardNormal.sample(&mut rng));
    let add = &us * draws;
    so.exo_pos = ishock;
    so.shocks = add.transpose();
    let point = lev(&pm.solve(&so).map_err(|e| anyhow!("{e}"))?.piecewise);
    Ok((mean, point))
}

/// `<fname>_posterior_{smoothed_variables,smoothed_shocks,filtered_variables}.csv` after an
/// OccBin MCMC (failed draws are dropped, as `is_successful_draw` does).
#[allow(clippy::too_many_arguments)]
pub fn posterior_objects(
    dsge: &DsgeModel,
    data: &LikelihoodData,
    occ: &OccbinEstimation,
    o: &Options,
    chains: &[Chain],
    first: usize,
    nkeep: usize,
    seed: u64,
    dates: Option<&[String]>,
    out_dir: &Path,
    fname: &str,
) -> Result<()> {
    let sub = match opt(o, "sub_draws") {
        Some(OptValue::Num(v)) => Some(*v as usize),
        _ => None,
    };
    let b = number_of_subdraws(sub, nkeep, chains.len())?;
    let draws = posterior_subsample(chains, first, nkeep, b, seed);
    posterior_from_draws(dsge, data, occ, o, &draws, seed, dates, out_dir, fname)
}

/// The statistics of [`posterior_objects`] for given parameter draws.
#[allow(clippy::too_many_arguments)]
pub fn posterior_from_draws(
    dsge: &DsgeModel,
    data: &LikelihoodData,
    occ: &OccbinEstimation,
    o: &Options,
    draws: &[Vec<f64>],
    seed: u64,
    dates: Option<&[String]>,
    out_dir: &Path,
    fname: &str,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    let b = draws.len();
    println!("\nOccBin: posterior distributions of smoothed objects: {b} draws (sub_draws) ...");
    let horizon = crate::forecast::commands::forecast_horizon(o);
    type Draw = (Mat, Mat, Option<Mat>, Vec<f64>, Option<(Mat, Mat)>);
    let per: Vec<Option<Draw>> = draws
        .par_iter()
        .enumerate()
        .map(|(i, x)| {
            let (a, b, c, last) = draw_series(dsge, data, occ, x).ok()?;
            let fc = if horizon > 0 {
                Some(draw_forecast(dsge, occ, x, &last, horizon, seed.wrapping_add(i as u64)).ok()?)
            } else {
                None
            };
            Some((a, b, c, last, fc))
        })
        .collect();
    let ok: Vec<&Draw> = per.iter().flatten().collect();
    if ok.is_empty() {
        bail!("OccBin: the smoother failed at every posterior draw");
    }
    let conf = match opt(o, "mh_conf_sig") {
        Some(OptValue::Num(v)) => *v,
        _ => 0.9,
    };
    let norig = dsge.model.orig_endo_nbr;
    let names = &dsge.model.endo_names[..norig];
    let nobs = data.y.ncols();
    let dates0 = dates;
    let dates: Vec<String> = dates.map(|d| d.to_vec()).unwrap_or_else(|| (1..=nobs).map(|t| t.to_string()).collect());
    std::fs::create_dir_all(out_dir)?;
    let mut written = Vec::new();
    if opt(o, "smoother").is_some() {
        let s = summarize(
            "SmoothedVariables",
            "Smoothed variables",
            names,
            &ok.iter().map(|d| d.0.clone()).collect::<Vec<_>>(),
            conf,
        );
        write_summary_csv(&out_dir.join(format!("{fname}_posterior_smoothed_variables.csv")), &s, &dates)?;
        let s = summarize(
            "SmoothedShocks",
            "Smoothed shocks",
            &dsge.model.exo_names,
            &ok.iter().map(|d| d.1.clone()).collect::<Vec<_>>(),
            conf,
        );
        write_summary_csv(&out_dir.join(format!("{fname}_posterior_smoothed_shocks.csv")), &s, &dates)?;
        written.push("smoothed variables/shocks");
    }
    if opt(o, "filtered_vars").is_some() && ok.iter().all(|d| d.2.is_some()) {
        let s = summarize(
            "FilteredVariables",
            "Filtered variables",
            names,
            &ok.iter().map(|d| d.2.clone().unwrap()).collect::<Vec<_>>(),
            conf,
        );
        write_summary_csv(&out_dir.join(format!("{fname}_posterior_filtered_variables.csv")), &s, &dates)?;
        written.push("filtered variables");
    }
    println!(
        "OccBin posterior {} ({} draws, {} failed, {:.1}s) written to {}/{fname}_posterior_*.csv",
        written.join(", "),
        ok.len(),
        per.len() - ok.len(),
        t0.elapsed().as_secs_f64(),
        out_dir.display()
    );
    if horizon > 0 {
        let first_obs = match opt(o, "first_obs") {
            Some(OptValue::Num(v)) => *v as usize,
            _ => 1,
        };
        let fc_dates = crate::forecast::extend_dates(dates0.filter(|d| !d.is_empty()), first_obs - 1 + nobs, horizon);
        let fc_dates =
            if dates0.is_some() { fc_dates } else { (1..=horizon).map(|h| (nobs + h).to_string()).collect() };
        let s = summarize(
            "MeanForecast",
            "Forecasted variables (mean)",
            names,
            &ok.iter().map(|d| d.4.as_ref().unwrap().0.clone()).collect::<Vec<_>>(),
            conf,
        );
        write_summary_csv(&out_dir.join(format!("{fname}_posterior_mean_forecast.csv")), &s, &fc_dates)?;
        let s = summarize(
            "PointForecast",
            "Forecasted variables (point)",
            names,
            &ok.iter().map(|d| d.4.as_ref().unwrap().1.clone()).collect::<Vec<_>>(),
            conf,
        );
        write_summary_csv(&out_dir.join(format!("{fname}_posterior_point_forecast.csv")), &s, &fc_dates)?;
        println!(
            "OccBin posterior forecasts (horizon {horizon}) written to {}/{fname}_posterior_{{mean,point}}_forecast.csv",
            out_dir.display()
        );
    }
    Ok(())
}
