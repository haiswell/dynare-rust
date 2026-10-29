// SPDX-License-Identifier: GPL-3.0-or-later
//! Driver-facing entry points of stream F: what `dynare_estimation_1.m` does with
//! `forecast` / `smoother` / `filtered_vars` after the mode or the MCMC, and the
//! `forecast`, `shock_decomposition` and `calib_smoother` commands. The estimation leaves an
//! [`EstimationState`] behind (Dynare's `oo_`/`options_` globals) that later commands use.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

use super::{
    dsge_smoother, extend_dates, forecast_figures, save_figures, simult_constant, stochastic_classical,
    write_forecast_csv,
};
use crate::analysis::shock_decomposition::{self as sdmod, ShockGroups};
use crate::dsge::{DsgeModel, LikelihoodData};
use crate::estimation::Chain;
use crate::estimation::prior_posterior::{
    PosteriorSmootherOptions, number_of_subdraws, pm3_figures, posterior_subsample, prior_posterior_statistics,
    write_summary_csv,
};
use crate::graphics::{GraphFormat, GraphOptions, graph_dir};
use crate::kalman::KalmanOptions;
use crate::linalg::Mat;
use crate::modfile::{Command, EstimatedKind, ModFile, OptValue, Options, opt};

/// What an `estimation` command leaves for the commands after it.
pub struct EstimationState {
    pub dsge: DsgeModel,
    pub data: LikelihoodData,
    /// Dates of the estimation sample (None if the datafile has none).
    pub dates: Option<Vec<String>>,
    pub first_obs: usize,
    pub var_list: Vec<String>,
    pub mode: Vec<f64>,
    pub posterior_mean: Option<Vec<f64>>,
    pub posterior_median: Option<Vec<f64>>,
    /// Graph options of the estimation command (Dynare keeps `options_` between commands).
    pub graphs: GraphOptions,
    /// `oo_.MeanForecast` (the posterior mean of the forecasts without shocks, written by
    /// prior_posterior_statistics); read by `shock_decomposition(forecast_type=unconditional)`
    /// for the Bayesian parameter sets.
    pub mean_forecast: Option<super::Forecast>,
    /// `oo_.PointForecast` (posterior forecasts with shocks, after an MCMC with `forecast`).
    pub point_forecast: Option<super::Forecast>,
    /// `oo_.forecast` (forecasts at the mode, estimation without MCMC with `forecast`).
    pub forecast: Option<super::Forecast>,
    /// `oo_.SmoothedVariables.Mean`, `oo_.SmoothedShocks.Mean` (posterior mean of the smoothed
    /// draws, nobs x orig_endo and nobs x n_exo): what smoother2histval reads after an MCMC.
    pub smoothed_mean: Option<(Mat, Mat)>,
}

impl EstimationState {
    /// Parameters `M_` holds after the estimation: posterior mean after an MCMC, else the mode.
    pub fn current(&self) -> &[f64] {
        self.posterior_mean.as_deref().unwrap_or(&self.mode)
    }
}

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

fn string(o: &Options, k: &str) -> Option<String> {
    match opt(o, k) {
        Some(OptValue::Str(s)) => Some(s.clone()),
        _ => None,
    }
}

/// `nograph` / `graph_format` of a command (default pdf, as the rest of dynare-rust).
pub fn graph_options(o: &Options, dir: &Path, fname: &str) -> GraphOptions {
    let mut formats = Vec::new();
    match opt(o, "graph_format") {
        Some(OptValue::Str(s)) => formats.extend(GraphFormat::parse(s)),
        Some(OptValue::List(l)) => {
            for v in l {
                if let OptValue::Str(s) = v {
                    formats.extend(GraphFormat::parse(s));
                }
            }
        }
        _ => {}
    }
    if formats.is_empty() {
        formats.push(GraphFormat::Pdf);
    }
    GraphOptions { enabled: opt(o, "nograph").is_none(), formats, dir: graph_dir(dir, fname) }
}

/// `options_.forecast` of an estimation command.
pub fn forecast_horizon(o: &Options) -> usize {
    num(o, "forecast").unwrap_or(0.0).max(0.0) as usize
}

pub(crate) fn var_indices(mf: &ModFile, var_list: &[String]) -> Result<(Vec<usize>, Vec<String>)> {
    if var_list.is_empty() {
        let n = mf.model.orig_endo_nbr;
        return Ok(((0..n).collect(), mf.model.endo_names[..n].to_vec()));
    }
    let idx = var_list
        .iter()
        .map(|v| mf.model.endo_index(v).ok_or_else(|| anyhow!("{v} isn't an endogenous variable")))
        .collect::<Result<Vec<_>>>()?;
    Ok((idx, var_list.to_vec()))
}

/// `forecast=H` after a mode-only estimation: `forecasts.run(var_list_, ..., 'smoother')` at
/// `x` (the mode), from the last smoothed state; writes `<fname>_forecast.csv` and the
/// `forcst<k>` figures.
#[allow(clippy::too_many_arguments)]
pub fn forecast_at_mode(
    mf: &ModFile,
    st: &EstimationState,
    x: &[f64],
    horizon: usize,
    conf_sig: f64,
    out_dir: &Path,
    fname: &str,
) -> Result<super::Forecast> {
    let f = smoother_forecast(mf, st, x, horizon, conf_sig)?;
    let nobs = st.data.y.ncols();
    let dates = extend_dates(st.dates.as_deref(), st.first_obs - 1 + nobs, horizon);
    let path = out_dir.join(format!("{fname}_forecast.csv"));
    write_forecast_csv(&path, &f, &dates)?;
    // (Dynare prints nothing here: the forecasts are in oo_.forecast and the csv)
    crate::progress::note(format!("Forecasts ({:.0}% bands) written to {}", 100.0 * conf_sig, path.display()));
    save_figures(&forecast_figures(&f), fname, &st.graphs);
    Ok(f)
}

/// Port of `forecasts.run(var_list_, M_, options_, oo_, 'smoother', dataset_info)` (order 1,
/// no `varexo_det`) at `x`: [`stochastic_classical`] from the last smoothed state, the
/// estimation's var_list, plus the `observation_trends` trend for the observed variables
/// (mean only). Under `loglinear` the forecasts are in logs (`simult_` adds `log(ys)`). Under `prefilter` Dynare then moves the *mean* of the observed
/// variables from the steady state to the sample mean, as run.m does it:
/// `yf(i_var_obs,:) - ys(i_var_obs) + mean_varobs` (`log(ys(i_var_obs))` under `loglinear`),
/// with two quirks reproduced here: the bands `HPDinf`/`HPDsup` are not moved (they stay
/// around the steady state), and `i_var_obs` holds positions in the var_list but indexes
/// `oo_.dr.ys` as if they were declaration indices (harmless while the var_list starts with
/// the variables in declaration order, or when those steady states coincide). A var_list
/// without every observable makes the assignment fail in MATLAB, so it is an error here too.
pub fn smoother_forecast(
    mf: &ModFile,
    st: &EstimationState,
    x: &[f64],
    horizon: usize,
    conf_sig: f64,
) -> Result<super::Forecast> {
    let sm = dsge_smoother(&st.dsge, &st.data, x, false)?;
    let (vars, names) = var_indices(mf, &st.var_list)?;
    let mut f = stochastic_classical(
        sm.dr(),
        &sm.ys_constant,
        &sm.last_state(),
        horizon,
        &sm.resolved.q,
        &vars,
        &names,
        conf_sig,
    );
    // ---- varexo_det paths (simult_varexo_det.m, order 1; make_ex_.m with the
    // default options_.periods = 0)
    crate::deterministic::exo_det::add_to_forecast(mf, sm.dr(), sm.ys(), &sm.resolved.params, 0, horizon, &mut f);
    super::add_measurement_error_bands(&mut f, &sm.resolved.h, &st.dsge.varobs); // forecasts.run: HPDinf_ME / HPDsup_ME
    // observation_trends: y0 is the smoothed state without the trend (oo_.Smoother.Trend
    // subtracted); the trend `trend_coeffs*(first_obs+gend-1+h)` (minus its sample mean under
    // prefilter) is added back to the mean forecast of the observed variables of the var_list,
    // not to the bands (yf_CI)
    if sm.trend.is_some() {
        let (first_obs, gend) = (st.data.first_obs as f64, st.data.y.ncols() as f64);
        let center = if st.data.prefilter_mean.is_some() { first_obs + (gend - 1.0) / 2.0 } else { 0.0 };
        for (p, &v) in vars.iter().enumerate() {
            if st.dsge.varobs.contains(&v) {
                let c = sm.trend_coeff_of(v);
                for h in 0..horizon {
                    f.mean[(h, p)] += c * (first_obs + gend + h as f64 - center);
                }
            }
        }
    }
    if let Some(mean) = &st.data.prefilter_mean {
        let ys = sm.ys();
        // i_var_obs: for each observable (varobs order), its position in the var_list
        let i_var_obs: Vec<(usize, usize)> = st
            .dsge
            .varobs
            .iter()
            .enumerate()
            .filter_map(|(k, v)| vars.iter().position(|w| w == v).map(|p| (p, k)))
            .collect();
        let nvobs = st.dsge.varobs.len();
        if i_var_obs.len() != nvobs {
            let matlab = if i_var_obs.len() == 1 {
                format!(
                    "Unable to perform assignment because the size of the left side is 1-by-{} and the size of the right side is {nvobs}-by-{}.",
                    horizon + 1,
                    horizon + 1
                )
            } else {
                "Arrays have incompatible sizes for this operation.".to_string()
            };
            bail!(
                "forecasts.run: with prefilter, the forecast var_list must contain every observed variable \
                 (Dynare 7.2 stops in forecasts.run: {matlab})"
            );
        }
        for &(p, k) in &i_var_obs {
            // Dynare: oo_.dr.ys(i_var_obs), a var_list position used as a declaration index
            let c = if st.dsge.loglinear { ys[p].ln() } else { ys[p] };
            for h in 0..horizon {
                f.mean[(h, p)] += mean[k] - c;
            }
        }
    }
    Ok(f)
}

fn print_forecast(f: &super::Forecast, dates: &[String]) {
    println!("\nPOINT FORECAST (mean; {:.0}% band in the CSV):", 100.0 * f.conf_sig);
    print!("{:<12}", "");
    for n in &f.names {
        print!(" {:>14}", if n.len() > 14 { &n[..14] } else { n });
    }
    println!();
    for h in 0..f.mean.nrows() {
        print!("{:<12}", dates[h]);
        for j in 0..f.names.len() {
            print!(" {:>14.4}", f.mean[(h, j)]);
        }
        println!();
    }
}

/// After an MCMC with `smoother`, `filtered_vars` or `forecast`: prior_posterior_statistics
/// on `sub_draws` draws (after `mh_drop`); writes `<fname>_posterior_<object>.csv` and pm3's
/// figures.
#[allow(clippy::too_many_arguments)]
pub fn posterior_objects(
    mf: &ModFile,
    st: &EstimationState,
    o: &Options,
    chains: &[Chain],
    first: usize,
    nkeep: usize,
    seed: u64,
    out_dir: &Path,
    fname: &str,
) -> Result<PosteriorForecasts> {
    let t = std::time::Instant::now();
    let b = number_of_subdraws(num(o, "sub_draws").map(|v| v as usize), nkeep, chains.len())?;
    let draws = posterior_subsample(chains, first, nkeep, b, seed);
    let n = mf.model.orig_endo_nbr;
    let horizon = forecast_horizon(o);
    let po = PosteriorSmootherOptions {
        smoother: opt(o, "smoother").is_some(),
        filtered_vars: opt(o, "filtered_vars").is_some() || opt(o, "filter_step_ahead").is_some(),
        horizon,
        mh_conf_sig: num(o, "mh_conf_sig").unwrap_or(0.9),
        vars: (0..n).collect(),
        var_names: mf.model.endo_names[..n].to_vec(),
        exo_names: mf.model.exo_names.clone(),
        seed,
        kernel_density: crate::estimation::est_completion::kernel_density(o), // ---- estimation completion
        state_uncertainty: flag(o, "smoothed_state_uncertainty"),
        filter_covariance: flag(o, "filter_covariance"),
    };
    crate::progress::note(format!(
        "posterior distributions of smoothed objects and forecasts: {b} draws (sub_draws) ..."
    ));
    let p = prior_posterior_statistics(&st.dsge, &st.data, &draws, &po)?;
    let nobs = st.data.y.ncols();
    let sample_dates: Vec<String> = st.dates.clone().unwrap_or_else(|| (1..=nobs).map(|t| t.to_string()).collect());
    let fc_dates = extend_dates(st.dates.as_deref(), st.first_obs - 1 + nobs, horizon);
    // Dynare plots options_.varlist (sorted orig endo names if empty); shocks: all.
    let mut select = st.var_list.clone();
    if select.is_empty() {
        select = mf.model.endo_names[..n].to_vec();
        select.sort();
    }
    let array_files = [
        (&p.state_uncertainty, "state_uncertainty"),
        (&p.init_state_uncertainty, "init_state_uncertainty"),
        (&p.filter_covariance, "filter_covariance"),
    ];
    for (s, file) in array_files {
        if let Some(s) = s {
            crate::estimation::prior_posterior::write_array_summary_csv(
                &out_dir.join(format!("{fname}_posterior_{file}.csv")),
                s,
            )?;
        }
    }
    let as_forecast = |s: &crate::estimation::prior_posterior::PosteriorSummary| super::Forecast {
        vars: (0..n).collect(),
        names: s.names.clone(),
        mean: s.mean.clone(),
        lower: s.hpd_inf.clone(),
        upper: s.hpd_sup.clone(),
        conf_sig: po.mh_conf_sig,
        lower_me: None,
        upper_me: None,
    };
    let mean_forecast = p.mean_forecast.as_ref().map(as_forecast);
    let point_forecast = p.point_forecast.as_ref().map(as_forecast);
    for s in p.all() {
        let file = match s.name.as_str() {
            "SmoothedVariables" => "smoothed_variables",
            "SmoothedShocks" => "smoothed_shocks",
            "UpdatedVariables" => "updated_variables",
            "FilteredVariables" => "filtered_variables",
            "MeanForecast" => "mean_forecast",
            "PointForecast" => "point_forecast",
            "Init_State" => "init_state",
            other => other,
        };
        let dates = if s.name.contains("Forecast") { &fc_dates } else { &sample_dates };
        write_summary_csv(&out_dir.join(format!("{fname}_posterior_{file}.csv")), s, dates)?;
        let sel: &[String] = if s.name == "SmoothedShocks" { &[] } else { &select };
        // ---- estimation completion: posterior_nograph (pm3.m: ~options_.no_graph.posterior), densities
        if !crate::estimation::est_completion::no_graph_posterior(o) {
            save_figures(&pm3_figures(s, sel), fname, &st.graphs);
        }
        if !s.density.is_empty() {
            let vars: Vec<(&str, crate::matfile::MatOut)> =
                s.names.iter().zip(&s.density).map(|(n, d)| (n.as_str(), crate::matfile::MatOut::Matrix(d))).collect();
            crate::matfile::write(&out_dir.join(format!("{fname}_posterior_{file}_density.mat")), &vars)?;
        }
        // ---- end estimation completion
    }
    crate::progress::note(format!(
        "Posterior smoothed objects{} ({} draws, {:.1}s) written to {}/{fname}_posterior_*.csv",
        if horizon > 0 { " and forecasts" } else { "" },
        p.draws,
        t.elapsed().as_secs_f64(),
        out_dir.display()
    ));
    // oo_.SmoothedVariables.Mean / oo_.SmoothedShocks.Mean (smoother2histval after an MCMC)
    let smoothed_mean = match (&p.smoothed_variables, &p.smoothed_shocks) {
        (Some(v), Some(s)) => Some((v.mean.clone(), s.mean.clone())),
        _ => None,
    };
    Ok(PosteriorForecasts { mean_forecast, point_forecast, smoothed_mean })
}

/// What [`posterior_objects`] leaves in `oo_`.
pub struct PosteriorForecasts {
    /// `oo_.MeanForecast`.
    pub mean_forecast: Option<super::Forecast>,
    /// `oo_.PointForecast`.
    pub point_forecast: Option<super::Forecast>,
    /// `oo_.SmoothedVariables.Mean`, `oo_.SmoothedShocks.Mean`.
    pub smoothed_mean: Option<(Mat, Mat)>,
}

// ---------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------

/// The calibrated value of every estimated parameter (`parameter_set = calibration`).
pub fn calibration_vector(dsge: &DsgeModel) -> Result<Vec<f64>> {
    dsge.estimated
        .iter()
        .zip(&dsge.estimated_names)
        .enumerate()
        .map(|(k, (e, name))| {
            let q = &dsge.base_sigma_e;
            let v = match e.kind {
                EstimatedKind::Param(i) => dsge.base_params[i],
                EstimatedKind::StdErr(i) => q[(i, i)].sqrt(),
                EstimatedKind::Corr(i, j) => q[(i, j)] / (q[(i, i)] * q[(j, j)]).sqrt(),
                EstimatedKind::MeStdErr(_)
                | EstimatedKind::MeCorr(..)
                | EstimatedKind::Skew(_)
                | EstimatedKind::InitState(_) => dsge.calibrated_value(k), // stream M
            };
            if v.is_finite() {
                Ok(v)
            } else {
                Err(anyhow!("parameter_set = calibration: {name} has no calibrated value"))
            }
        })
        .collect()
}

/// Is option `k` set (flag, or a non-zero value)?
fn flag(o: &Options, k: &str) -> bool {
    match opt(o, k) {
        Some(OptValue::Flag) => true,
        Some(OptValue::Num(v)) => *v != 0.0,
        _ => false,
    }
}

/// Data of a command that takes the estimation's data options (`datafile`, `first_obs`,
/// `nobs`, `presample`, `diffuse_filter`, `lik_init`, `kalman_algo`, tolerances, `lyapunov`):
/// makedataset.m with `loglinear` (the data are logged unless `logdata`) and `prefilter`
/// (demeaned, the sample mean becomes the constant of the observables), and `first_obs` as
/// the time index of `observation_trends`.
pub fn load_data(mf: &ModFile, dir: &Path, o: &Options) -> Result<(LikelihoodData, Option<Vec<String>>, usize)> {
    let ds = crate::dataset::Dataset::load(mf, dir, o)?; // makedataset.m (datafile or data command)
    let (table, first_obs, sample) = (ds.table.clone(), ds.first_obs, ds.sample(&mf.varobs)?);
    let mut k = KalmanOptions {
        presample: num(o, "presample").unwrap_or(0.0) as usize,
        lik_init: num(o, "lik_init").unwrap_or(1.0) as u8,
        kalman_algo: num(o, "kalman_algo").unwrap_or(0.0) as u8,
        ..Default::default()
    };
    if opt(o, "diffuse_filter").is_some() {
        k.lik_init = 3;
    }
    for (key, f) in [
        ("kalman_tol", &mut k.kalman_tol),
        ("diffuse_kalman_tol", &mut k.diffuse_kalman_tol),
        ("riccati_tol", &mut k.riccati_tol),
        ("qz_criterium", &mut k.qz_criterium),
    ] {
        if let Some(v) = num(o, key) {
            *f = v;
        }
    }
    crate::estimation::q3::kalman_options(o, &mut k); // lyapunov, rescale_prediction_error_covariance
    crate::estimation::est_completion::kalman_options(o, &mut k); // use_univariate_*_if_singularity_is_detected
    let nobs = sample.nrows();
    let dates = table.dates.map(|d| d[first_obs - 1..first_obs - 1 + nobs].to_vec());
    let mut sample = sample;
    if flag(o, "loglinear") && !flag(o, "logdata") {
        sample.iter_mut().for_each(|v| *v = v.ln()); // makedataset.m: dataset_.log()
    }
    let mut data = LikelihoodData::new(&sample, k);
    data.first_obs = first_obs; // observation_trends
    crate::estimation::q3::attach(mf, o, &mut data, first_obs, ds.init)?; // heteroskedastic_filter
    if num(o, "prefilter").unwrap_or(0.0) == 1.0 {
        data.prefilter(); // makedataset.m: isequal(options_.prefilter, 1)
    }
    Ok((data, dates, first_obs))
}

/// `parameter_set` (evaluate_smoother.m / shock_decomposition.m): the default is
/// posterior_mean after an MCMC, else posterior_mode, else calibration.
pub(crate) fn parameter_vector(st: Option<&EstimationState>, dsge: &DsgeModel, set: &str) -> Result<Vec<f64>> {
    let need = || st.ok_or_else(|| anyhow!("parameter_set = {set} needs a previous estimation"));
    Ok(match set {
        "posterior_mode" | "mle_mode" => need()?.mode.clone(),
        "posterior_mean" => {
            need()?.posterior_mean.clone().ok_or_else(|| anyhow!("parameter_set = posterior_mean needs an MCMC"))?
        }
        "posterior_median" => {
            need()?.posterior_median.clone().ok_or_else(|| anyhow!("parameter_set = posterior_median needs an MCMC"))?
        }
        "prior_mean" => dsge.priors.iter().map(|p| p.mean).collect(),
        "prior_mode" => dsge.priors.iter().map(|p| p.p5).collect(),
        // evaluate_smoother('calibration') smooths at the current M_.params: the estimation's
        // point after an estimation (checked against Dynare in the conditional-forecast tests)
        "calibration" => match st {
            Some(s) => s.current().to_vec(),
            None => calibration_vector(dsge)?,
        },
        other => bail!("evaluate_smoother: unknown parameter_set option {other}."),
    })
}

/// The `parameter_set` of shock_decomposition.m, realtime_shock_decomposition.m and
/// initial_condition_decomposition.m when `options_.parameter_set` is empty: posterior_mean
/// after an MCMC, else mle_mode, else posterior_mode after an estimation. Without any of them
/// (`who` = the command's name) Dynare stops.
pub(crate) fn default_parameter_set(st: Option<&EstimationState>, who: &str) -> Result<&'static str> {
    match st {
        Some(s) if s.posterior_mean.is_some() => Ok("posterior_mean"),
        Some(s) if s.dsge.ml => Ok("mle_mode"), // isfield(oo_, 'mle_mode')
        Some(_) => Ok("posterior_mode"),
        None => bail!("{who}: option parameter_set is not specified and posterior mode is not available"),
    }
}

/// `oo_.dr` as the previous commands left it: `forecasts.run(..., 'simul')` uses the stored
/// decision rules, it does not solve the model again.
#[derive(Clone, Debug)]
pub enum DecisionRules {
    /// `oo_.dr = []` (global_initialization.m): no `stoch_simul` or estimation computed it, and
    /// forecasts.run stops.
    Empty,
    /// Computed by the last `stoch_simul` or estimation at `M_.params = params` (None: the
    /// parameters of the mod file as they are now) and, after an estimation, at the estimated
    /// parameters `point` (None: [`EstimationState::current`]).
    Computed { params: Option<Vec<f64>>, point: Option<Vec<f64>> },
}

/// What `forecasts.run` reads from `options_` / `oo_` as the previous commands left it: the
/// approximation order (`options_.order`, 1 under `model(linear)`), `pruning`,
/// `options_.periods` (the horizon when the command has no `periods`),
/// `options_.forecasts.conf_sig` (when the command has no `conf_sig`), `oo_.dr`, and the seed
/// of the band simulations at order > 1 (`options_.forecast_replic = 1000` draws).
#[derive(Clone, Debug)]
pub struct ForecastOrder {
    pub order: usize,
    pub pruning: bool,
    pub periods: usize,
    pub conf_sig: f64,
    pub dr: DecisionRules,
    pub replic: usize,
    pub seed: u64,
    /// MATLAB's `randn` draws to replay instead of the seeded stream (tests).
    pub replay: Option<Vec<f64>>,
}

impl ForecastOrder {
    pub fn of(mf: &ModFile, ss: &crate::analysis::StochSimulOptions, seed: u64) -> ForecastOrder {
        let mut order = if mf.model.linear { 1 } else { ss.order };
        if order > 1 && mf.model.dynamic_derivatives(2).hessian_eq_zero() {
            order = 1; // stoch_simul computed an order-1 rule
        }
        ForecastOrder { order, pruning: ss.pruning, periods: ss.periods, ..ForecastOrder::first() }.with_seed(seed)
    }
    /// Order 1, `options_.periods = 0`, `options_.forecasts.conf_sig = 0.9`
    /// (default_option_values.m), decision rules at the current parameters.
    pub fn first() -> ForecastOrder {
        ForecastOrder {
            order: 1,
            pruning: false,
            periods: 0,
            conf_sig: 0.9,
            dr: DecisionRules::Computed { params: None, point: None },
            replic: 1000,
            seed: 0,
            replay: None,
        }
    }
    fn with_seed(mut self, seed: u64) -> ForecastOrder {
        self.seed = seed;
        self
    }
}

/// `forecast(periods=, conf_sig=) varlist;` (`forecasts.run(..., 'simul')`): from the steady
/// state, or `M_.endo_histval` with a `histval` block, with the decision rules `oo_.dr` the last
/// `stoch_simul` or estimation left (the parameters they were computed at, [`DecisionRules`]; an
/// error when there are none), at `options_.order`
/// ([`ForecastOrder`]): order 1 with analytical bands (`stochastic_classical`), orders 2-3 by
/// simulation; with `varexo_det`, `simult_varexo_det.m` (orders 1-2). Returns `oo_.forecast`.
#[allow(clippy::too_many_arguments)]
pub fn forecast_command(
    mf: &ModFile,
    st: Option<&EstimationState>,
    o: &Options,
    vars: &[String],
    dir: &Path,
    fname: &str,
    out_dir: &Path,
    fo: &ForecastOrder,
) -> Result<super::Forecast> {
    let (params, point) = match &fo.dr {
        DecisionRules::Empty => {
            bail!("forecasts.run: the decision rules have not been computed. Did you forget a stoch_simul-command?")
        }
        DecisionRules::Computed { params, point } => (params, point),
    };
    if st.is_some_and(|s| s.data.prefilter_mean.is_some()) {
        // forecasts.run(..., 'simul') gets no dataset_info
        bail!("forecasts.run: The prefiltering option is not allowed without providing a dataset");
    }
    // forecasts.run's varexo_det checks
    if !mf.model.exo_det_names.is_empty() {
        if fo.order > 2 {
            bail!("forecasts.run: forecasting with varexo_det does not support order>2.");
        }
        if fo.order == 2 && fo.pruning {
            bail!("forecasts.run: forecasting with varexo_det does not support pruning.");
        }
    }
    // oo_.dr: the parameters of the stoch_simul / estimation that computed it
    let mut own;
    let (dsge, x) = match st {
        Some(s) if params.is_none() => (&s.dsge, point.clone().unwrap_or_else(|| s.current().to_vec())),
        _ => {
            own = DsgeModel::new(mf, None)?;
            if let Some(p) = params {
                own.base_params = p.clone();
            }
            let x = match point {
                Some(x) => x.clone(),
                None => calibration_vector(&own)?,
            };
            (&own, x)
        }
    };
    let res = dsge.resolve(&x).map_err(|e| anyhow!("forecast: no solution (info {})", e.info))?;
    // options_.periods / options_.forecasts.conf_sig: the command's (the preprocessor writes
    // them to options_ before calling forecasts.run), else the ones previous commands left
    let periods = num(o, "periods").map_or(fo.periods, |v| v as usize);
    let horizon = if periods == 0 { 5 } else { periods };
    let conf_sig = num(o, "conf_sig").unwrap_or(fo.conf_sig);
    let (idx, names) = var_indices(mf, vars)?;
    // y0 = dr.ys (log(dr.ys) under loglinear), or M_.endo_histval (levels)
    let histval = crate::analysis::stochastic_histval_of(&mf.model, &mf.histval);
    let f = if fo.order >= 2 {
        let y0 = histval.unwrap_or_else(|| res.steady_state.clone());
        simulated_forecast(mf, &res, &y0, periods, horizon, &idx, &names, conf_sig, fo, &dsge.varobs)?
    } else {
        let y0 = match &histval {
            // y0 = log_variable(M_.endo_histval) under loglinear, deviation from simult_'s constant
            Some(h) => {
                let c = simult_constant(dsge, &res.steady_state);
                let lv = crate::analysis::q2::log_variable(&mf.model, h);
                let y = if dsge.loglinear { &lv } else { h };
                (0..h.len()).map(|k| y[res.dr.order_var[k]] - c[res.dr.order_var[k]]).collect()
            }
            None => vec![0.0; res.dr.ghx.nrows()],
        };
        let mut f = stochastic_classical(
            &res.dr,
            &simult_constant(dsge, &res.steady_state),
            &y0,
            horizon,
            &res.q,
            &idx,
            &names,
            conf_sig,
        );
        // ---- varexo_det paths (simult_varexo_det.m)
        crate::deterministic::exo_det::add_to_forecast(
            mf,
            &res.dr,
            &res.steady_state,
            &res.params,
            periods,
            horizon,
            &mut f,
        );
        super::add_measurement_error_bands(&mut f, &res.h, &dsge.varobs); // forecasts.run: HPDinf_ME / HPDsup_ME
        f
    };
    let dates: Vec<String> = (1..=horizon).map(|h| h.to_string()).collect();
    let path = out_dir.join(format!("{fname}_forecast.csv"));
    write_forecast_csv(&path, &f, &dates)?;
    print_forecast(&f, &dates);
    let from = if mf.histval.is_empty() { "the steady state" } else { "histval" };
    println!("Forecasts (from {from}) written to {}", path.display());
    let mut g = graph_options(o, dir, fname);
    if let Some(s) = st {
        g.enabled &= s.graphs.enabled || opt(o, "nograph").is_none();
    }
    save_figures(&forecast_figures(&f), fname, &g);
    Ok(f)
}

/// `forecasts.run` at `options_.order` 2 or 3 (`simult_varexo_det.m` with `varexo_det`, order 2;
/// `forecasts.stochastic_classical` otherwise): the mean is the simulation without shocks
/// from `y0` (levels), the bands the quantiles of `fo.replic` simulations with
/// `randn(nxs, periods)'*chol(Sigma_e)` shocks.
#[allow(clippy::too_many_arguments)]
fn simulated_forecast(
    mf: &ModFile,
    res: &crate::dsge::Resolved,
    y0: &[f64],
    periods: usize,
    horizon: usize,
    vars: &[usize],
    names: &[String],
    conf_sig: f64,
    fo: &ForecastOrder,
    varobs: &[usize],
) -> Result<super::Forecast> {
    let model = &mf.model;
    let dr2 = crate::solver::korder::solve_model(
        model,
        &res.steady_state,
        &res.params,
        &res.q,
        fo.order,
        Default::default(),
    )?;
    // M_.H not zero: simulated measurement errors (stochastic_classical's `yf_CI_ME`)
    let me = res.h.iter().any(|&x| x != 0.0).then_some((&res.h, varobs));
    let mut seeded = crate::nonlinear::rng::SeededDraws::new(fo.seed);
    let mut replay = fo.replay.clone().map(|z| crate::nonlinear::rng::ReplayDraws::new(z, Vec::new()));
    let draws: &mut dyn crate::nonlinear::rng::Draws = match replay.as_mut() {
        Some(r) => r,
        None => &mut seeded,
    };
    if model.exo_det_names.is_empty() {
        let sim = |ex: &Mat| crate::solver::korder::simulate::simult(&dr2, y0, ex, fo.pruning);
        super::simulated_forecast(
            &sim,
            model.n_exo(),
            horizon,
            horizon,
            &res.q,
            vars,
            names,
            conf_sig,
            fo.replic,
            draws,
            me,
        )
    } else {
        use crate::deterministic::exo_det;
        let path = exo_det::ExoDetPath::new(mf, &res.params, periods, horizon);
        let xd = exo_det::second_order(model, &dr2, &res.steady_state, &res.params, path.length);
        let sim = |ex: &Mat| exo_det::simulate_order2(&dr2, &xd, &path, y0, ex);
        super::simulated_forecast(
            &sim,
            model.n_exo(),
            path.sim_periods,
            horizon,
            &res.q,
            vars,
            names,
            conf_sig,
            fo.replic,
            draws,
            me,
        )
    }
}

/// Data and smoother model for shock_decomposition / calib_smoother: the estimation's, or the
/// command's own data options (calibration) when there was no estimation or it sets `datafile`.
pub(crate) fn smoother_inputs<'a>(
    mf: &ModFile,
    st: Option<&'a EstimationState>,
    o: &Options,
    dir: &Path,
    own: &'a mut Option<(Option<DsgeModel>, LikelihoodData)>,
) -> Result<(&'a DsgeModel, &'a LikelihoodData, Option<Vec<String>>)> {
    match st {
        Some(s) if opt(o, "datafile").is_none() => {
            // command-level prefilter / use_univariate_*: evaluate_smoother.m re-runs
            // dynare_estimation_init with the command's options_
            let set = [
                "prefilter",
                "use_univariate_smoother_if_singularity_is_detected",
                "use_univariate_filters_if_singularity_is_detected",
            ]
            .iter()
            .any(|k| opt(o, k).is_some());
            if !set {
                return Ok((&s.dsge, &s.data, s.dates.clone()));
            }
            let mut data = s.data.clone();
            if num(o, "prefilter").unwrap_or(0.0) == 1.0 && data.prefilter_mean.is_none() {
                data.prefilter();
                if s.dsge.varobs.iter().any(|&v| s.dsge.level_constant(v, &s.dsge.ys_guess).abs() >= 1e-9) {
                    bail!(
                        "The option \"prefilter\" is inconsistent with the non-zero mean measurement equations in the model."
                    );
                }
            }
            crate::estimation::est_completion::kalman_options(o, &mut data.kalman);
            *own = Some((None, data));
            Ok((&s.dsge, &own.as_ref().unwrap().1, s.dates.clone()))
        }
        _ => {
            let (data, dates, _) = load_data(mf, dir, o).context("no estimation before this command")?;
            let mut dsge = DsgeModel::new(mf, None)?;
            if flag(o, "loglinear") {
                dsge.loglinear = true; // options_.loglinear of the command (dynare_resolve)
            }
            // dynare_estimation_init.m (called by evaluate_smoother.m)
            if data.prefilter_mean.is_some()
                && dsge.varobs.iter().any(|&v| dsge.level_constant(v, &dsge.ys_guess).abs() >= 1e-9)
            {
                bail!(
                    "The option \"prefilter\" is inconsistent with the non-zero mean measurement equations in the model."
                );
            }
            *own = Some((Some(dsge), data));
            let (d, l) = own.as_ref().unwrap();
            Ok((d.as_ref().unwrap(), l, dates))
        }
    }
}

/// `shock_decomposition(parameter_set=, use_shock_groups=, init_state=) varlist;`
/// Returns `oo_.shock_decomposition` (all endogenous variables), the parameter set and the
/// steady state.
#[allow(clippy::too_many_arguments)]
pub fn shock_decomposition_command(
    mf: &ModFile,
    st: Option<&EstimationState>,
    o: &Options,
    vars: &[String],
    dir: &Path,
    fname: &str,
    out_dir: &Path,
) -> Result<(sdmod::ShockDecomposition, String, Vec<f64>)> {
    let set = match string(o, "parameter_set") {
        Some(s) => s,
        None => default_parameter_set(st, "shock_decomposition")?.to_string(),
    };
    let mut own = None;
    let (dsge, data, dates) = smoother_inputs(mf, st, o, dir, &mut own)?;
    let x = parameter_vector(st, dsge, &set)?;
    let sm = dsge_smoother(dsge, data, &x, false)?;
    let init_state = num(o, "init_state").unwrap_or(0.0) as usize;
    let sd = sdmod::shock_decomposition(sm.dr(), &sm.smoothed_deviation(), &sm.out.etahat, init_state);
    let (idx, names) = var_indices(mf, vars)?;
    let gend = data.y.ncols();
    let dates: Vec<String> = dates.unwrap_or_else(|| (1..=gend).map(|t| t.to_string()).collect());
    let mut shock_labels = mf.model.exo_names.clone();
    shock_labels.push("Initial values".into());
    let rows: Vec<(String, Mat, Vec<String>)> =
        idx.iter().zip(&names).map(|(&v, n)| (n.clone(), sd.z[v].clone(), shock_labels.clone())).collect();
    let path = out_dir.join(format!("{fname}_shock_decomposition.csv"));
    sdmod::write_csv(&path, &rows, &dates)?;

    let groups = match string(o, "use_shock_groups") {
        Some(g) => Some(find_shock_groups(mf, &g)?),
        None => None,
    };
    let mut grouped_rows = Vec::new();
    let mut g = graph_options(o, dir, fname);
    if let Some(s) = st {
        g.enabled &= s.graphs.enabled || opt(o, "nograph").is_none();
    }
    let thin = (gend / 8).max(1);
    let ticks: Vec<(f64, String)> =
        dates.iter().enumerate().step_by(thin).map(|(t, d)| ((t + 1) as f64, d.clone())).collect();
    for (var, zv, labels) in &rows {
        let (zv, labels) = match &groups {
            Some(gr) => {
                let (zz, l) = sdmod::group(zv, sd.nshocks, gr);
                grouped_rows.push((var.clone(), zz.clone(), l.clone()));
                (zz, l)
            }
            None => (zv.clone(), labels.clone()),
        };
        if let Some(fig) = sdmod::decomposition_figure(
            var,
            &zv,
            &labels,
            &set,
            groups.as_ref().map(|g| g.name.as_str()),
            Some(ticks.clone()),
        ) {
            save_figures(&[fig], fname, &g);
        }
    }
    if let Some(gr) = &groups {
        sdmod::write_csv(&out_dir.join(format!("{fname}_shock_decomposition_{}.csv", gr.name)), &grouped_rows, &dates)?;
    }
    println!(
        "Shock decomposition ({}, {} variables{}) written to {}",
        sdmod::parameter_set_name(&set),
        rows.len(),
        groups.as_ref().map(|g| format!(", groups `{}`", g.name)).unwrap_or_default(),
        path.display()
    );
    Ok((sd, set, sm.ys().to_vec()))
}

pub(crate) fn find_shock_groups(mf: &ModFile, name: &str) -> Result<ShockGroups> {
    for c in &mf.commands {
        if let Command::ShockGroups(n, groups) = c
            && n == name
        {
            let groups = groups
                .iter()
                .map(|(label, shocks)| {
                    let idx = shocks
                        .iter()
                        .map(|s| mf.model.exo_index(s).ok_or_else(|| anyhow!("shock_groups {name}: unknown shock {s}")))
                        .collect::<Result<Vec<_>>>()?;
                    Ok((label.clone(), idx))
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(ShockGroups { name: n.clone(), groups });
        }
    }
    bail!("use_shock_groups = {name}: no shock_groups(name={name}) block")
}

/// `calib_smoother(datafile=, ...) varlist;`: evaluate_smoother('calibration'); writes the
/// smoothed variables / shocks (and filtered variables with `filtered_vars`) like the
/// estimation's smoother, under `<fname>_calib_*.csv`.
pub fn calib_smoother_command(
    mf: &ModFile,
    st: Option<&EstimationState>,
    o: &Options,
    vars: &[String],
    dir: &Path,
    fname: &str,
    out_dir: &Path,
) -> Result<crate::misc::SmoothedRecord> {
    let mut own = None;
    let (dsge, data, dates) = smoother_inputs(mf, st, o, dir, &mut own)?;
    let x = parameter_vector(st, dsge, "calibration")?;
    // oo_.SmoothedVariables / SmoothedShocks (store_smoother_results), read by smoother2histval
    let record = crate::misc::SmoothedRecord {
        series: crate::driver::run_smoother(dsge, data, &x, false)?,
        steady_state: dsge
            .resolve(&x)
            .map(|r| r.steady_state)
            .map_err(|e| anyhow!("calib_smoother: no steady state (info {})", e.info))?,
        loglinear: dsge.loglinear,
    };
    let sm = dsge_smoother(dsge, data, &x, false)?;
    let n = mf.model.orig_endo_nbr;
    let all: Vec<usize> = (0..n).collect();
    let names = &mf.model.endo_names[..n];
    let gend = data.y.ncols();
    let dates: Vec<String> = dates.unwrap_or_else(|| (1..=gend).map(|t| t.to_string()).collect());
    let write = |file: &str, cols: &[String], m: &Mat| -> Result<()> {
        let mut w = format!("date,{}\n", cols.join(","));
        for t in 0..m.nrows() {
            w += &dates[t];
            for j in 0..m.ncols() {
                w += &format!(",{}", m[(t, j)]);
            }
            w.push('\n');
        }
        std::fs::create_dir_all(out_dir)?;
        let p = out_dir.join(format!("{fname}_{file}.csv"));
        std::fs::write(&p, w).with_context(|| format!("writing {}", p.display()))
    };
    write("calib_smoothed_variables", names, &sm.levels(&sm.out.alphahat, &all))?;
    write("calib_smoothed_shocks", &mf.model.exo_names, &sm.out.etahat.transpose())?;
    write("calib_updated_variables", names, &sm.levels(&sm.out.ahat, &all))?;
    if opt(o, "filtered_vars").is_some() {
        write("calib_filtered_variables", names, &sm.filtered_levels(&all))?;
    }
    println!("calib_smoother: smoothed variables/shocks at the calibration written to {}", out_dir.display());
    crate::estimation::q3::smoother_outputs(dsge, data, &x, o, out_dir, &format!("{fname}_calib"))?; // filter_step_ahead, ...
    let (idx, show) = var_indices(mf, vars)?;
    if !vars.is_empty() {
        println!("Last smoothed values (calibration):");
        let lv = sm.levels(&sm.out.alphahat, &idx);
        for (j, v) in show.iter().enumerate() {
            println!("  {:<24} {:>12.4}", v, lv[(gend - 1, j)]);
        }
    }
    Ok(record)
}
