// SPDX-License-Identifier: GPL-3.0-or-later
//! Posterior distributions of smoothed objects and forecasts: port of
//! `prior_posterior_statistics.m` / `prior_posterior_statistics_core.m` (type 'posterior',
//! order 1, no OccBin) and the statistics part of `pm3.m`.
//!
//! For each of `sub_draws` parameter vectors taken from the MCMC chains after `mh_drop`
//! (`get_posterior_subsample.m`), the smoother is rerun (`DsgeSmoother`), and the smoothed
//! variables, smoothed shocks, trend coefficients, smoothed constant and trend, updated
//! variables, smoothed measurement errors (estimated ones only), one-step-ahead filtered
//! variables and the forecasts without (`MeanForecast`) and with simulated shocks
//! (`PointForecast`, plus `PointForecastME` with measurement errors) are stored, as are the
//! initial state (`Init_State`) and, with `smoothed_state_uncertainty` / `filter_covariance`,
//! the array statistics `StateUncertainty`, `InitStateUncertainty` (multivariate smoother
//! only) and `FilterCovariance` (diffuse periods are zero, as the k-step-ahead covariances).
//! `pm3` then reports, per variable and period, the mean, median, variance, HPD interval at
//! `mh_conf_sig` and deciles (`posterior_moments.m`). Draws run in parallel (rayon); each
//! draw's shocks come from its own ChaCha stream, so results do not depend on scheduling.

use std::path::Path;

use anyhow::{Context, Result, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

use super::mcmc::Chain;
use super::posterior::posterior_moments;
use crate::dsge::{DsgeModel, LikelihoodData};
use crate::forecast::{ShockLoading, dsge_smoother_request, matlab_quantile, simulate_posterior_forecasts};
use crate::graphics::{Color, Figure, LineStyle, Panel, Series};
use crate::linalg::{Mat, chol_upper};
use crate::modfile::EstimatedKind;

/// pm3's summary of one object: rows = periods (n2), columns = variables.
#[derive(Clone, Debug)]
pub struct PosteriorSummary {
    /// Dynare's `name3` (`SmoothedVariables`, `MeanForecast`, ...).
    pub name: String,
    /// pm3's `tit1` (console/figure title).
    pub title: String,
    pub names: Vec<String>,
    pub mean: Mat,
    pub median: Mat,
    pub var: Mat,
    pub hpd_inf: Mat,
    pub hpd_sup: Mat,
    /// deciles[k] = (k+1)*10 % quantile (`post_deciles` / `oo_.X.deciles`), each n2 x nvar.
    pub deciles: Vec<Mat>,
    /// ---- estimation completion: pm3's `Density(:,:,:,i)` per variable (512 x 2*n2, see
    /// [`crate::estimation::est_completion::pm3_density`]); empty with `no_posterior_kernel_density`.
    pub density: Vec<Mat>,
}

/// Port of pm3.m's three-dimensional branch: `stock[b]` is draw b's n2 x nvar matrix.
pub fn summarize(name: &str, title: &str, names: &[String], stock: &[Mat], mh_conf_sig: f64) -> PosteriorSummary {
    let (n2, nvar) = stock.first().map_or((0, names.len()), |m| (m.nrows(), m.ncols()));
    let cells: Vec<_> = (0..n2 * nvar)
        .into_par_iter()
        .map(|c| {
            let (j, i) = (c % n2, c / n2);
            let mut xx: Vec<f64> = stock.iter().map(|m| m[(j, i)]).collect();
            posterior_moments(&mut xx, mh_conf_sig)
        })
        .collect();
    let get = |f: &dyn Fn(&super::ParamMoments) -> f64| Mat::from_fn(n2, nvar, |j, i| f(&cells[i * n2 + j]));
    PosteriorSummary {
        name: name.into(),
        title: title.into(),
        names: names.to_vec(),
        mean: get(&|p| p.mean),
        median: get(&|p| p.median),
        var: get(&|p| p.var),
        hpd_inf: get(&|p| p.hpd.0),
        hpd_sup: get(&|p| p.hpd.1),
        deciles: (0..9).map(|k| get(&|p| p.deciles[k])).collect(),
        density: Vec::new(),
    }
}

/// ---- estimation completion: [`summarize`] plus pm3's kernel density estimates
/// (`options_.estimation.moments_posterior_density.indicator`).
pub fn summarize_with_density(
    name: &str,
    title: &str,
    names: &[String],
    stock: &[Mat],
    mh_conf_sig: f64,
    density: bool,
) -> PosteriorSummary {
    let mut s = summarize(name, title, names, stock, mh_conf_sig);
    if density {
        s.density = (0..names.len())
            .into_par_iter()
            .map(|i| crate::estimation::est_completion::pm3_density(stock, i))
            .collect();
    }
    s
}

/// What prior_posterior_statistics computes (`options_.smoother`, `filtered_vars`, `forecast`).
#[derive(Clone, Debug)]
pub struct PosteriorSmootherOptions {
    pub smoother: bool,
    pub filtered_vars: bool,
    /// `options_.forecast` (0 = none).
    pub horizon: usize,
    pub mh_conf_sig: f64,
    /// Declaration-order endogenous variables reported (and their names).
    pub vars: Vec<usize>,
    pub var_names: Vec<String>,
    pub exo_names: Vec<String>,
    /// Seed of the forecast shocks (draw b uses stream b of this seed).
    pub seed: u64,
    /// ---- estimation completion: pm3's densities (not `no_posterior_kernel_density`).
    pub kernel_density: bool,
    /// `options_.smoothed_state_uncertainty`.
    pub state_uncertainty: bool,
    /// `options_.filter_covariance`.
    pub filter_covariance: bool,
}

/// pm3's statistics of an array object (`oo_.Smoother.State_uncertainty`,
/// `oo_.Smoother.Init_State_uncertainty`, `oo_.FilterCovariance`): per period (one n x n matrix
/// per period) the mean, median, variance (N-1), the `(1-c)/2` and `(1-c)/2+c` quantiles
/// (`HPDinf`, `HPDsup`; NaN with at most 2 draws) and the deciles (NaN with at most 9 draws).
#[derive(Clone, Debug)]
pub struct PosteriorArraySummary {
    pub name: String,
    pub title: String,
    /// Names of the rows and columns (all endogenous variables, declaration order).
    pub names: Vec<String>,
    pub mean: Vec<Mat>,
    pub median: Vec<Mat>,
    pub var: Vec<Mat>,
    pub hpd_inf: Vec<Mat>,
    pub hpd_sup: Vec<Mat>,
    /// `deciles[t][k]`, k = 0..9 for the quantile (k+1)/10.
    pub deciles: Vec<Vec<Mat>>,
}

/// `stock[b][t]` is draw b's n x n matrix of period t; the statistics of pm3.m's
/// `draw_dimension` branches.
pub fn summarize_array(
    name: &str,
    title: &str,
    names: &[String],
    stock: &[Vec<Mat>],
    mh_conf_sig: f64,
) -> PosteriorArraySummary {
    let nper = stock.first().map_or(0, Vec::len);
    let n = names.len();
    let b = stock.len();
    let (p_lo, p_hi) = ((1.0 - mh_conf_sig) / 2.0, (1.0 - mh_conf_sig) / 2.0 + mh_conf_sig);
    let mut out = PosteriorArraySummary {
        name: name.into(),
        title: title.into(),
        names: names.to_vec(),
        mean: Vec::new(),
        median: Vec::new(),
        var: Vec::new(),
        hpd_inf: Vec::new(),
        hpd_sup: Vec::new(),
        deciles: Vec::new(),
    };
    for t in 0..nper {
        let (mut mean, mut median, mut var) = (Mat::zeros(n, n), Mat::zeros(n, n), Mat::zeros(n, n));
        let (mut lo, mut hi) = (Mat::from_element(n, n, f64::NAN), Mat::from_element(n, n, f64::NAN));
        let mut dec = vec![Mat::from_element(n, n, f64::NAN); 9];
        for i in 0..n {
            for j in 0..n {
                let mut x: Vec<f64> = stock.iter().map(|d| d[t][(i, j)]).collect();
                let m = x.iter().sum::<f64>() / b as f64;
                mean[(i, j)] = m;
                var[(i, j)] =
                    if b > 1 { x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (b as f64 - 1.0) } else { f64::NAN };
                x.sort_by(|a, c| a.total_cmp(c));
                median[(i, j)] = if b % 2 == 1 { x[b / 2] } else { (x[b / 2 - 1] + x[b / 2]) / 2.0 };
                if b > 2 {
                    lo[(i, j)] = matlab_quantile(&x, p_lo);
                    hi[(i, j)] = matlab_quantile(&x, p_hi);
                }
                if b > 9 {
                    for (k, d) in dec.iter_mut().enumerate() {
                        d[(i, j)] = matlab_quantile(&x, (k + 1) as f64 / 10.0);
                    }
                }
            }
        }
        out.mean.push(mean);
        out.median.push(median);
        out.var.push(var);
        out.hpd_inf.push(lo);
        out.hpd_sup.push(hi);
        out.deciles.push(dec);
    }
    out
}

#[derive(Clone, Debug, Default)]
pub struct PosteriorObjects {
    /// Number of draws used (failed draws are dropped, as `is_successful_draw` does).
    pub draws: usize,
    pub failed: usize,
    pub smoothed_variables: Option<PosteriorSummary>,
    pub smoothed_shocks: Option<PosteriorSummary>,
    /// `TrendCoeff` (one row; zero for the variables that are not observed).
    pub trend_coeff: Option<PosteriorSummary>,
    /// `Constant` (smoothed constant).
    pub constant: Option<PosteriorSummary>,
    /// `Trend` (smoothed trend; zero for the variables that are not observed).
    pub trend: Option<PosteriorSummary>,
    pub updated_variables: Option<PosteriorSummary>,
    /// `Init_State` (smoothed initial state, one row).
    pub init_state: Option<PosteriorSummary>,
    /// `oo_.Smoother.State_uncertainty` (`smoothed_state_uncertainty`).
    pub state_uncertainty: Option<PosteriorArraySummary>,
    /// `oo_.Smoother.Init_State_uncertainty` (`smoothed_state_uncertainty`).
    pub init_state_uncertainty: Option<PosteriorArraySummary>,
    /// `oo_.FilterCovariance` (`filter_covariance`), periods 1..nobs+1.
    pub filter_covariance: Option<PosteriorArraySummary>,
    /// `SmoothedMeasurementErrors` (`SE_EOBS_<varobs>`), when measurement errors are estimated.
    pub smoothed_measurement_errors: Option<PosteriorSummary>,
    pub filtered_variables: Option<PosteriorSummary>,
    pub mean_forecast: Option<PosteriorSummary>,
    pub point_forecast: Option<PosteriorSummary>,
    /// `PointForecastME` (observed variables), when `M_.H` is not zero.
    pub point_forecast_me: Option<PosteriorSummary>,
}

impl PosteriorObjects {
    /// The objects in the order of prior_posterior_statistics.m's `pm3` calls.
    pub fn all(&self) -> Vec<&PosteriorSummary> {
        [
            &self.smoothed_variables,
            &self.smoothed_shocks,
            &self.trend_coeff,
            &self.init_state,
            &self.constant,
            &self.trend,
            &self.updated_variables,
            &self.smoothed_measurement_errors,
            &self.filtered_variables,
            &self.mean_forecast,
            &self.point_forecast,
            &self.point_forecast_me,
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

/// `set_number_of_subdraws.m`: `sub_draws` or min(posterior_max_subsample_draws = 1200,
/// number of kept draws over all chains).
pub fn number_of_subdraws(sub_draws: Option<usize>, kept_per_chain: usize, nblck: usize) -> Result<usize> {
    let total = kept_per_chain * nblck;
    match sub_draws {
        None => Ok(crate::estimation::mh_session::posterior_max_subsample_draws().min(total)),
        Some(b) if b > total => bail!(
            "The value of option sub_draws ({b}) is greater than the number of available draws in the MCMC ({total})!"
        ),
        Some(b) => Ok(b),
    }
}

/// `get_posterior_subsample.m` (draws `first..first+nkeep` of each chain are kept): for each
/// of the `b` subdraws a chain and a kept draw are picked uniformly at random (with
/// replacement); when `b` equals the number of kept draws all of them are used
/// (`GetAllPosteriorDraws`, chain by chain).
pub fn posterior_subsample(chains: &[Chain], first: usize, nkeep: usize, b: usize, seed: u64) -> Vec<Vec<f64>> {
    if b == nkeep * chains.len() {
        return chains.iter().flat_map(|c| (first..first + nkeep).map(move |t| c.draw(t).to_vec())).collect();
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(u64::MAX - 7);
    let mut addr: Vec<(usize, usize)> =
        (0..b).map(|_| (rng.gen_range(0..chains.len()), first + rng.gen_range(0..nkeep))).collect();
    // sortrows(SampleAddress, [3 2]): grouped by chain (file order irrelevant here)
    addr.sort_by_key(|&(c, _)| c);
    addr.into_iter().map(|(c, t)| chains[c].draw(t).to_vec()).collect()
}

struct DrawResult {
    smooth: Mat,
    update: Mat,
    filter: Mat,
    innov: Mat,
    trend_coeff: Mat,
    init_state: Mat,
    constant: Mat,
    trend: Mat,
    error: Mat,
    /// State uncertainty per period, all endogenous variables (declaration order).
    state_unc: Vec<Mat>,
    init_unc: Option<Mat>,
    filter_cov: Vec<Mat>,
    fc_mean: Mat,
    fc_point: Mat,
    fc_point_me: Mat,
}

/// The loop of `prior_posterior_statistics_core.m` over `draws`, then `pm3` per object.
pub fn prior_posterior_statistics(
    dsge: &DsgeModel,
    data: &LikelihoodData,
    draws: &[Vec<f64>],
    o: &PosteriorSmootherOptions,
) -> Result<PosteriorObjects> {
    let vars = &o.vars;
    let nvobs = dsge.varobs.len();
    let gend = data.y.ncols();
    // estim_params_.nvn: number of estimated measurement errors
    let nvn = dsge.estimated.iter().filter(|e| matches!(e.kind, EstimatedKind::MeStdErr(_))).count();
    // varlist_forecast_ME = intersect(options_.varobs, varlist) (sorted), as varobs positions
    let mut me_obs: Vec<usize> = (0..nvobs).filter(|&i| vars.contains(&dsge.varobs[i])).collect();
    me_obs.sort_by(|&a, &b| dsge.model.endo_names[dsge.varobs[a]].cmp(&dsge.model.endo_names[dsge.varobs[b]]));
    let results: Vec<Option<DrawResult>> = draws
        .par_iter()
        .enumerate()
        .map(|(b, x)| {
            let req = o.filter_covariance.then_some(crate::kalman::SmootherRequest {
                nk: 1,
                filter_covariance: true,
                decomposition: false,
                a_initial: None,
            });
            let sm = dsge_smoother_request(dsge, data, x, o.state_uncertainty, req.as_ref()).ok()?;
            let dr = sm.dr();
            let ys = sm.ys();
            let y0 = sm.last_state();
            // DsgeSmoother's trend_coeff (zeros without observation_trends) and trend_addition
            let trend_coeff: Vec<f64> = dsge
                .trend_addition(&sm.resolved.params, 1, 1, false)
                .map_or_else(|| vec![0.0; nvobs], |m| m.column(0).iter().copied().collect());
            let trend_addition = dsge
                .trend_addition(&sm.resolved.params, data.first_obs, gend, data.prefilter_mean.is_some())
                .unwrap_or_else(|| Mat::zeros(nvobs, gend));
            // log(SteadyState) under loglinear, SteadyState otherwise (every variable)
            let ss: Vec<f64> = ys.iter().map(|&y| if dsge.loglinear { y.ln() } else { y }).collect();
            // stock_smoothed_constant: ss, with prefilter mean_varobs for the observed variables
            let mut constant = ss.clone();
            if let Some(m) = &data.prefilter_mean {
                for (i, &v) in dsge.varobs.iter().enumerate() {
                    constant[v] = m[i];
                }
            }
            // stock_trend_coeff(varobs_id) = trend_coeff, stock_smoothed_trend(IdObs,:) = trend_addition
            let mut stock_trend_coeff = Mat::zeros(1, vars.len());
            let mut stock_trend = Mat::zeros(gend, vars.len());
            for (j, &v) in vars.iter().enumerate() {
                if let Some(i) = dsge.varobs.iter().position(|&w| w == v) {
                    stock_trend_coeff[(0, j)] = trend_coeff[i];
                    for t in 0..gend {
                        stock_trend[(t, j)] = trend_addition[(i, t)];
                    }
                }
            }
            let (fc_mean, fc_point, fc_point_me) = if o.horizon > 0 {
                // yf and yf1: with prefilter + mean_varobs and the trend centered on the sample,
                // else + the trend (trend_coeff*(first_obs-1+gend+h) in forecast period h); then
                // + log(SteadyState) under loglinear, + SteadyState otherwise
                let mut offset = Mat::from_fn(o.horizon, ys.len(), |_, v| ss[v]);
                let first_obs = data.first_obs as f64;
                for (i, &v) in dsge.varobs.iter().enumerate() {
                    let c = trend_coeff[i];
                    for h in 0..o.horizon {
                        let mut add = (first_obs - 1.0 + (gend + h + 1) as f64) * c;
                        if let Some(m) = &data.prefilter_mean {
                            // mean(trend_coeff*(first_obs:first_obs+gend-1), 2)
                            add += m[i] - c * (first_obs + (gend as f64 - 1.0) / 2.0);
                        }
                        offset[(h, v)] += add;
                    }
                }
                let pick = |m: &Mat| Mat::from_fn(m.nrows(), vars.len(), |h, j| m[(h, vars[j])] + offset[(h, vars[j])]);
                let mean = simulate_posterior_forecasts::<ChaCha8Rng>(dr, &y0, o.horizon, None);
                let mut rng = ChaCha8Rng::seed_from_u64(o.seed);
                rng.set_stream(b as u64 + 1);
                let load = ShockLoading::new(dr, &sm.resolved.q).ok()?;
                let point = simulate_posterior_forecasts(dr, &y0, o.horizon, Some((&load, &mut rng)));
                // ~isequal(M_.H,0): yf1(:,IdObs)' + ME_shocks, ME_shocks(i_exo_var,:) =
                // chol(H(i_exo_var,i_exo_var))*randn(nxs,horizon)
                let h = &sm.resolved.h;
                let point_me = if h.iter().any(|&v| v != 0.0) && !me_obs.is_empty() {
                    let i_exo_var: Vec<usize> = (0..nvobs).filter(|&i| h[(i, i)] != 0.0).collect();
                    let nxs = i_exo_var.len();
                    let chol_h = chol_upper(&Mat::from_fn(nxs, nxs, |a, c| h[(i_exo_var[a], i_exo_var[c])]))?;
                    let e = Mat::from_fn(nxs, o.horizon, |_, _| rng.sample::<f64, _>(StandardNormal));
                    let shocks = &chol_h * e;
                    let mut me = Mat::zeros(nvobs, o.horizon);
                    for (a, &i) in i_exo_var.iter().enumerate() {
                        for t in 0..o.horizon {
                            me[(i, t)] = shocks[(a, t)];
                        }
                    }
                    Mat::from_fn(o.horizon, me_obs.len(), |t, j| {
                        let (i, v) = (me_obs[j], dsge.varobs[me_obs[j]]);
                        point[(t, v)] + offset[(t, v)] + me[(i, t)]
                    })
                } else {
                    Mat::zeros(0, 0)
                };
                (pick(&mean), pick(&point), point_me)
            } else {
                (Mat::zeros(0, 0), Mat::zeros(0, 0), Mat::zeros(0, 0))
            };
            // declaration-order copy of a DR-order n x n matrix (stock(dr.order_var, dr.order_var, ...))
            let n_endo = dr.inv_order_var.len();
            let decl = |m: &Mat| Mat::from_fn(n_endo, n_endo, |a, c| m[(dr.inv_order_var[a], dr.inv_order_var[c])]);
            let state_unc: Vec<Mat> =
                sm.out.state_uncertainty.as_ref().map_or_else(Vec::new, |v| v.iter().map(decl).collect());
            let init_unc = sm.out.state_uncertainty0.as_ref().map(decl);
            // P(:,:,1) = Pstar, P(:,:,t+1) = PK(1,:,:,t+1)
            let filter_cov: Vec<Mat> = match sm.out.step_ahead.as_ref().and_then(|s| s.pk.as_ref()) {
                Some(pk) => {
                    std::iter::once(decl(&sm.pstar_initial)).chain((1..=gend).map(|t| decl(&pk[0][t]))).collect()
                }
                None => Vec::new(),
            };
            let a0 = &sm.out.alphahat0;
            let init_state = sm.levels(&Mat::from_column_slice(a0.nrows(), 1, a0.as_slice()), vars);
            Some(DrawResult {
                init_state,
                state_unc,
                init_unc,
                filter_cov,
                smooth: sm.levels(&sm.out.alphahat, vars),
                update: sm.levels(&sm.out.ahat, vars),
                filter: sm.filtered_levels(vars), // trend shifted by one period
                innov: sm.out.etahat.transpose(),
                trend_coeff: stock_trend_coeff,
                constant: Mat::from_fn(gend, vars.len(), |_, j| constant[vars[j]]),
                trend: stock_trend,
                error: sm.out.epsilonhat.transpose(),
                fc_mean,
                fc_point,
                fc_point_me,
            })
        })
        .collect();
    let failed = results.iter().filter(|r| r.is_none()).count();
    let ok: Vec<DrawResult> = results.into_iter().flatten().collect();
    if ok.is_empty() {
        bail!("prior_posterior_statistics: the smoother failed for every draw");
    }
    if failed > 0 {
        println!(
            "prior_posterior_statistics: {failed} of the {} draws failed (no solution) and were dropped.",
            draws.len()
        );
    }
    let c = o.mh_conf_sig;
    let collect = |f: &dyn Fn(&DrawResult) -> Mat| ok.iter().map(f).collect::<Vec<Mat>>();
    let run = |name: &str, title: &str, names: &[String], f: &dyn Fn(&DrawResult) -> Mat| {
        println!("Estimation::mcmc: {title}");
        let s = summarize_with_density(name, title, names, &collect(f), c, o.kernel_density); // ---- estimation completion
        println!("Estimation::mcmc: {title}, done!");
        Some(s)
    };
    let mut out = PosteriorObjects { draws: ok.len(), failed, ..Default::default() };
    if o.smoother {
        out.smoothed_variables = run("SmoothedVariables", "Smoothed variables", &o.var_names, &|d| d.smooth.clone());
        out.smoothed_shocks = run("SmoothedShocks", "Smoothed shocks", &o.exo_names, &|d| d.innov.clone());
        out.trend_coeff = run("TrendCoeff", "Trend coefficients", &o.var_names, &|d| d.trend_coeff.clone());
        out.init_state = run("Init_State", "Init State", &o.var_names, &|d| d.init_state.clone());
        out.constant = run("Constant", "Smoothed constant", &o.var_names, &|d| d.constant.clone());
        out.trend = run("Trend", "Smoothed trend", &o.var_names, &|d| d.trend.clone());
        out.updated_variables = run("UpdatedVariables", "Updated Variables", &o.var_names, &|d| d.update.clone());
        if o.state_uncertainty && ok.iter().all(|d| !d.state_unc.is_empty()) {
            let names = dsge.model.endo_names.clone();
            let stock: Vec<Vec<Mat>> = ok.iter().map(|d| d.state_unc.clone()).collect();
            println!("Estimation::mcmc: State Uncertainty");
            out.state_uncertainty = Some(summarize_array("StateUncertainty", "State Uncertainty", &names, &stock, c));
            if ok.iter().all(|d| d.init_unc.is_some()) {
                let stock: Vec<Vec<Mat>> = ok.iter().map(|d| vec![d.init_unc.clone().unwrap()]).collect();
                out.init_state_uncertainty =
                    Some(summarize_array("InitStateUncertainty", "Initial State Uncertainty", &names, &stock, c));
            }
        }
        if nvn > 0 {
            // meas_error_names: SE_EOBS_<varobs>
            let names: Vec<String> =
                dsge.varobs.iter().map(|&v| format!("SE_EOBS_{}", dsge.model.endo_names[v])).collect();
            out.smoothed_measurement_errors =
                run("SmoothedMeasurementErrors", "Smoothed measurement errors", &names, &|d| d.error.clone());
        }
    }
    if o.filtered_vars {
        out.filtered_variables =
            run("FilteredVariables", "One step ahead forecast (filtered variables)", &o.var_names, &|d| {
                d.filter.clone()
            });
    }
    if o.horizon > 0 {
        out.mean_forecast = run("MeanForecast", "Forecasted variables (mean)", &o.var_names, &|d| d.fc_mean.clone());
        out.point_forecast =
            run("PointForecast", "Forecasted variables (point)", &o.var_names, &|d| d.fc_point.clone());
        if ok.iter().all(|d| d.fc_point_me.ncols() > 0) {
            let names: Vec<String> = me_obs.iter().map(|&i| dsge.model.endo_names[dsge.varobs[i]].clone()).collect();
            out.point_forecast_me =
                run("PointForecastME", "Forecasted variables (point) with ME", &names, &|d| d.fc_point_me.clone());
        }
    }
    if o.filter_covariance && ok.iter().all(|d| !d.filter_cov.is_empty()) {
        let names = dsge.model.endo_names.clone();
        let stock: Vec<Vec<Mat>> = ok.iter().map(|d| d.filter_cov.clone()).collect();
        println!("Estimation::mcmc: Filtered covariances");
        out.filter_covariance = Some(summarize_array("FilterCovariance", "Filtered covariances", &names, &stock, c));
    }
    Ok(out)
}

/// `<fname>_posterior_<object>.csv` of an array object, long format:
/// `period,variable1,variable2,mean,median,var,hpd_inf,hpd_sup,d10..d90`.
pub fn write_array_summary_csv(path: &Path, s: &PosteriorArraySummary) -> Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut w = String::from(
        "period,variable1,variable2,mean,median,var,hpd_inf,hpd_sup,d10,d20,d30,d40,d50,d60,d70,d80,d90\n",
    );
    for t in 0..s.mean.len() {
        for (i, n1) in s.names.iter().enumerate() {
            for (j, n2) in s.names.iter().enumerate() {
                w += &format!(
                    "{},{n1},{n2},{},{},{},{},{}",
                    t + 1,
                    s.mean[t][(i, j)],
                    s.median[t][(i, j)],
                    s.var[t][(i, j)],
                    s.hpd_inf[t][(i, j)],
                    s.hpd_sup[t][(i, j)]
                );
                for d in &s.deciles[t] {
                    w += &format!(",{}", d[(i, j)]);
                }
                w.push('\n');
            }
        }
    }
    std::fs::write(path, w).with_context(|| format!("writing {}", path.display()))
}

/// `<fname>_posterior_<object>.csv`, long format:
/// `date,variable,mean,median,sd,hpd_lower,hpd_upper,d10..d90`.
pub fn write_summary_csv(path: &Path, s: &PosteriorSummary, dates: &[String]) -> Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut w = String::from("date,variable,mean,median,sd,hpd_lower,hpd_upper,d10,d20,d30,d40,d50,d60,d70,d80,d90\n");
    for (i, name) in s.names.iter().enumerate() {
        for t in 0..s.mean.nrows() {
            w += &format!(
                "{},{},{},{},{},{},{}",
                dates.get(t).cloned().unwrap_or_else(|| (t + 1).to_string()),
                name,
                s.mean[(t, i)],
                s.median[(t, i)],
                s.var[(t, i)].sqrt(),
                s.hpd_inf[(t, i)],
                s.hpd_sup[(t, i)]
            );
            for d in &s.deciles {
                w += &format!(",{}", d[(t, i)]);
            }
            w.push('\n');
        }
    }
    std::fs::write(path, w).with_context(|| format!("writing {}", path.display()))
}

/// pm3_core.m's figures: 3x3 panels, the deciles as thin green lines and the mean in black;
/// file `<fname>_<name3>_<last variable of the figure>`. `select`: the variables to plot
/// (Dynare plots `options_.varlist`), all if empty. Panels whose mean is all NaN are skipped.
/// pm3.m draws no figure for the trend coefficients (`_trend_coeff`).
pub fn pm3_figures(s: &PosteriorSummary, select: &[String]) -> Vec<Figure> {
    if s.name == "TrendCoeff" {
        return Vec::new();
    }
    let idx: Vec<usize> = if select.is_empty() {
        (0..s.names.len()).collect()
    } else {
        select.iter().filter_map(|v| s.names.iter().position(|n| n == v)).collect()
    };
    let idx: Vec<usize> = idx.into_iter().filter(|&i| s.mean.column(i).iter().any(|v| !v.is_nan())).collect();
    let n2 = s.mean.nrows();
    let x: Vec<f64> = (1..=n2).map(|t| t as f64).collect();
    let mut figs = Vec::new();
    for (k, chunk) in idx.chunks(9).enumerate() {
        let panels = chunk
            .iter()
            .map(|&i| {
                let mut series: Vec<Series> = s
                    .deciles
                    .iter()
                    .map(|d| Series::Line {
                        x: x.clone(),
                        y: d.column(i).iter().copied().collect(),
                        color: Color::Green,
                        style: LineStyle::Solid,
                        width: 0.5,
                        label: None,
                    })
                    .collect();
                series.push(Series::Line {
                    x: x.clone(),
                    y: s.mean.column(i).iter().copied().collect(),
                    color: Color::Black,
                    style: LineStyle::Solid,
                    width: 1.0,
                    label: None,
                });
                Panel { title: s.names[i].clone(), series, ..Default::default() }
            })
            .collect();
        figs.push(Figure {
            name: format!("{}_{}", s.name, s.names[*chunk.last().unwrap()]),
            title: format!("{} {}", s.title, k + 1),
            rows: 3,
            cols: 3,
            panels,
        });
    }
    figs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_matches_posterior_moments() {
        let stock: Vec<Mat> = (0..50).map(|b| Mat::from_fn(2, 1, |t, _| (b as f64) * (t as f64 + 1.0))).collect();
        let s = summarize("X", "x", &["a".into()], &stock, 0.9);
        assert!((s.mean[(0, 0)] - 24.5).abs() < 1e-12);
        assert!((s.mean[(1, 0)] - 49.0).abs() < 1e-12);
        assert_eq!(s.deciles.len(), 9);
        assert!((s.deciles[4][(0, 0)] - 24.0).abs() < 1e-12);
    }

    #[test]
    fn subdraws() {
        assert_eq!(number_of_subdraws(None, 500, 2).unwrap(), 1000);
        assert_eq!(number_of_subdraws(None, 5000, 2).unwrap(), 1200);
        assert!(number_of_subdraws(Some(2000), 500, 2).is_err());
    }
}
