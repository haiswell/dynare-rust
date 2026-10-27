//! Parity fixes of the analysis-forecast group (Dynare 7.2): the single DsgeSmoother port
//! (observation_trends, lik_init 2/4/5, lyapunov) behind the post-estimation commands, the
//! forecast and conditional-forecast trends, `calib_smoother(loglinear, prefilter, first_obs)`,
//! the posterior objects (initial state, state uncertainty, filter covariance, loglinear and
//! prefilter forecasts), and the smaller fixes (check, moments_varendo, bandpass, ...).
//!
//! Expected values come from Dynare's reference output (reference/misc/ref_misc.json), from
//! equivalences between runs (a run with `observation_trends` on trended data against a run
//! without trends on detrended data) or from closed forms of AR(1) models.
mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use dynare::analysis::{self, OutputContext, StochSimulOptions, moments, vardec};
use dynare::driver::{RunOptions, run};
use dynare::dsge::{DsgeModel, LikelihoodData};
use dynare::estimation::mcmc::Chain;
use dynare::estimation::prior_posterior::{PosteriorSmootherOptions, prior_posterior_statistics};
use dynare::kalman::KalmanOptions;
use dynare::linalg::Mat;
use dynare::modfile::parse_mod_str;
use serde_json::Value;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn misc() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/misc")
}

fn csv(p: &Path) -> Vec<Vec<String>> {
    let s = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    s.lines().map(|l| l.split(',').map(str::to_string).collect()).collect()
}

fn col(h: &[String], n: &str) -> usize {
    h.iter().position(|s| s == n).unwrap_or_else(|| panic!("no column {n} in {h:?}"))
}

/// A numeric column of a CSV file.
fn column(p: &Path, name: &str) -> Vec<f64> {
    let rows = csv(p);
    let j = col(&rows[0], name);
    rows[1..].iter().map(|r| r[j].parse().unwrap_or(f64::NAN)).collect()
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

fn all_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (t, (x, y)) in a.iter().zip(b).enumerate() {
        close(*x, *y, tol, &format!("{what}[{t}]"));
    }
}

/// Runs `body` (a .mod file) in a fresh directory holding `files` (name, content); returns
/// the `<tag>/Output` directory.
fn run_mod(tag: &str, body: &str, files: &[(&str, String)], defines: &[(&str, &str)]) -> PathBuf {
    let dir = common::temp_dir(tag);
    for (n, c) in files {
        std::fs::write(dir.join(n), c).unwrap();
    }
    std::fs::write(dir.join(format!("{tag}.mod")), body).unwrap();
    let defines = defines.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    run(&dir.join(format!("{tag}.mod")), &RunOptions { quiet: true, defines, ..Default::default() })
        .unwrap_or_else(|e| panic!("{tag}: {e:#}"));
    dir.join(tag).join("Output")
}

fn run_err(tag: &str, body: &str, files: &[(&str, String)]) -> String {
    let dir = common::temp_dir(tag);
    for (n, c) in files {
        std::fs::write(dir.join(n), c).unwrap();
    }
    std::fs::write(dir.join(format!("{tag}.mod")), body).unwrap();
    match run(&dir.join(format!("{tag}.mod")), &RunOptions { quiet: true, ..Default::default() }) {
        Ok(()) => panic!("{tag}: Dynare stops here but the run succeeded"),
        Err(e) => format!("{e:#}"),
    }
}

fn xlik_files() -> Vec<(&'static str, String)> {
    ["x_lik_obs.csv", "x_lik_data.m"].iter().map(|f| (*f, std::fs::read_to_string(misc().join(f)).unwrap())).collect()
}

fn xlik_mod() -> String {
    std::fs::read_to_string(misc().join("x_lik.mod")).unwrap()
}

// ---------------------------------------------------------------------------
// items 1 and 4: the smoother of the post-estimation commands against Dynare's reference
// ---------------------------------------------------------------------------

/// x_lik.mod at its initial values (mode_compute = 0): `calib_smoother` on its own data
/// (levels, trend and prefilter mean included) is Dynare's `oo_.SmoothedVariables`, and the
/// 'Smoothed' column of `shock_decomposition` is `Smoothed_Variables_deviation_from_mean`:
/// the smoothed variables minus the constant (the sample mean of the observables under
/// prefilter) and minus the observation trend, which DsgeSmoother subtracts from the data.
/// lik_init 2/5, observation_trends and prefilter all change the smoother's output.
#[test]
fn post_estimation_smoother_matches_dynare() {
    let r: Value = common::json_ref(misc().join("ref_misc.json"));
    let data = column(&misc().join("x_lik_obs.csv"), "x");
    let dataz = column(&misc().join("x_lik_obs.csv"), "z");
    // the sample mean ignores the missing observations (x(31:32) is missing)
    let mean = |v: &[f64]| {
        let o: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
        o.iter().sum::<f64>() / o.len() as f64
    };
    let mut checked = 0;
    for run_ref in r["runs"].as_array().unwrap() {
        let name = run_ref["name"].as_str().unwrap();
        if !run_ref["error"].as_str().unwrap().is_empty() || name == "MFILE" || name.starts_with("UNITROOT") {
            continue;
        }
        let defs = run_ref["defines"].as_str().unwrap();
        let likinit: u8 =
            defs.split_whitespace().find_map(|d| d.strip_prefix("-DLIKINIT=")).map_or(1, |v| v.parse().unwrap());
        let trend = defs.contains("-DTREND");
        let prefilter = defs.contains("-DPREFILTER");
        let g = if trend { run_ref["xparam_init"].as_array().unwrap().last().unwrap().as_f64().unwrap() } else { 0.0 };
        let defines: Vec<(&str, &str)> = defs
            .split_whitespace()
            .map(|d| d.trim_start_matches("-D"))
            .map(|d| d.split_once('=').unwrap_or((d, "")))
            .collect();
        let extra = format!(
            "\ncalib_smoother(datafile='x_lik_obs.csv', lik_init={likinit}{}) x z w;\n\
             shock_decomposition(parameter_set=posterior_mode) x z w;\n",
            if prefilter { ", prefilter=1" } else { "" }
        );
        let tag = format!("pafx_{}", name.replace(['=', '_'], ""));
        let out = run_mod(&tag, &(xlik_mod() + &extra), &xlik_files(), &defines);
        let t: Vec<f64> = (1..=120).map(|t| t as f64).collect();
        for v in ["x", "z", "w"] {
            let want: Vec<f64> =
                run_ref["SmoothedVariables"][v].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
            // calib_smoother: levels
            let got = column(&out.join(format!("{tag}_calib_smoothed_variables.csv")), v);
            all_close(&got, &want, 1e-9, &format!("{name}: calib_smoother {v}"));
            // shock_decomposition: deviation from the mean, detrended
            let sd = csv(&out.join(format!("{tag}_shock_decomposition.csv")));
            let (iv, ic, ix) = (col(&sd[0], "variable"), col(&sd[0], "component"), col(&sd[0], "value"));
            let smoothed: Vec<f64> =
                sd[1..].iter().filter(|r| r[iv] == v && r[ic] == "Smoothed").map(|r| r[ix].parse().unwrap()).collect();
            let mut dev = want.clone();
            if prefilter && v != "w" {
                let m = mean(if v == "x" { &data } else { &dataz });
                dev.iter_mut().for_each(|d| *d -= m);
            }
            if trend && v == "x" {
                let tm = if prefilter { mean(&t) } else { 0.0 };
                for (d, tt) in dev.iter_mut().zip(&t) {
                    *d -= g * (tt - tm);
                }
            }
            all_close(&smoothed, &dev, 1e-9, &format!("{name}: shock_decomposition {v}"));
        }
        checked += 1;
    }
    assert!(checked >= 5, "only {checked} runs checked");
}

// ---------------------------------------------------------------------------
// items 2 and 3: forecast and conditional forecast with observation_trends
// ---------------------------------------------------------------------------

/// A model with an observed trend: `observation_trends x (g)` on data `x + g t` is the model
/// without trend on the detrended data. The smoothed variables (levels), the smoother
/// forecast (mean, `+ g (gend + h)`, run.m) and the conditional forecast (constrained path
/// and forecast shifted by the trend, run.m) must reproduce that.
#[test]
fn forecasts_with_observation_trends_match_detrended_run() {
    let g = 0.015;
    let obs = std::fs::read_to_string(misc().join("x_lik_obs.csv")).unwrap();
    let mut trended = String::from("x,z,w\n");
    let mut detrended = String::from("x,z,w\n");
    for (t, l) in obs.lines().skip(1).enumerate() {
        // x(31:32) is missing; the file is read as the trended data, the detrended file
        // removes g*t
        let f: Vec<f64> = l.split(',').map(|v| v.parse().unwrap_or(f64::NAN)).collect();
        let s = |v: f64| if v.is_nan() { String::new() } else { v.to_string() };
        trended += &format!("{},{},{}\n", s(f[0]), s(f[1]), s(f[2]));
        detrended += &format!("{},{},{}\n", s(f[0] - g * (t + 1) as f64), s(f[1]), s(f[2]));
    }
    let model = |trend: bool, data: &str| {
        format!(
            "var x z w;\nvarexo ex ez ew;\nparameters a b c g d;\na = 0.8; b = 0.3; c = 0.5; g = {g}; d = 0.1;\n\
             model(linear);\nx = a*x(-1) + b*z(-1) + ex;\nz = c*z(-1) + ez;\nw = 0.6*w(-1) + d*x + ew;\nend;\n\
             shocks;\nvar ex; stderr 1;\nvar ez; stderr 0.5;\nvar ew; stderr 0.1;\nend;\nsteady;\nvarobs x z;\n\
             {}\
             estimated_params;\na, 0.7, -0.99, 0.99;\nstderr ex, 0.8, 0.01, 10;\nend;\n\
             estimation(datafile='{data}', mode_compute=0, nograph, smoother, forecast=4);\n\
             conditional_forecast_paths;\nvar x;\nperiods 1:1, 2:2;\nvalues {};\nend;\n\
             conditional_forecast(parameter_set=posterior_mode, controlled_varexo=(ex), periods=5, replic=20) x z w;\n",
            if trend { "observation_trends;\nx (g);\nend;\n" } else { "" },
            if trend { "0.3+0.015*121, 0.3+0.015*122" } else { "0.3, 0.3" }
        )
    };
    let files = vec![("trended.csv", trended), ("detrended.csv", detrended)];
    let out_t = run_mod("paf_trend", &model(true, "trended.csv"), &files, &[]);
    let out_n = run_mod("paf_notrend", &model(false, "detrended.csv"), &files, &[]);
    let sm_t = |v: &str| column(&out_t.join("paf_trend_smoothed_variables.csv"), v);
    let sm_n = |v: &str| column(&out_n.join("paf_notrend_smoothed_variables.csv"), v);
    let tt: Vec<f64> = (1..=120).map(|t| t as f64).collect();
    // smoothed variables in levels: x carries the trend, the others do not
    let want: Vec<f64> = sm_n("x").iter().zip(&tt).map(|(v, t)| v + g * t).collect();
    all_close(&sm_t("x"), &want, 1e-9, "smoothed x");
    all_close(&sm_t("z"), &sm_n("z"), 1e-9, "smoothed z");
    all_close(&sm_t("w"), &sm_n("w"), 1e-9, "smoothed w");

    // smoother forecast: the mean of x gets g*(gend+h); the bands (yf_CI) are not shifted
    let fc = |out: &Path, tag: &str| {
        let rows = csv(&out.join(format!("{tag}_forecast.csv")));
        let (v, h) = (col(&rows[0], "variable"), col(&rows[0], "horizon"));
        let cols: Vec<String> = rows[0].clone();
        let mut m: HashMap<(String, usize, String), f64> = HashMap::new();
        for r in &rows[1..] {
            for (j, c) in cols.iter().enumerate() {
                if let Ok(x) = r[j].parse::<f64>() {
                    if j != h {
                        m.insert((r[v].clone(), r[h].parse().unwrap(), c.clone()), x);
                    }
                }
            }
        }
        m
    };
    let (ft, fnn) = (fc(&out_t, "paf_trend"), fc(&out_n, "paf_notrend"));
    for h in 1..=4usize {
        let k = |v: &str, c: &str| (v.to_string(), h, c.to_string());
        close(ft[&k("x", "mean")], fnn[&k("x", "mean")] + g * (120 + h) as f64, 1e-9, &format!("forecast x[{h}]"));
        close(ft[&k("z", "mean")], fnn[&k("z", "mean")], 1e-9, &format!("forecast z[{h}]"));
        close(ft[&k("x", "lower")], fnn[&k("x", "lower")], 1e-9, "forecast band x");
    }

    // conditional forecast: shifted by g*(gend+t) in x, for the paths and the unconditional path
    let (ct, cn) = (fc(&out_t, "paf_trend_conditional"), fc(&out_n, "paf_notrend_conditional"));
    for t in 0..=5usize {
        for v in ["x", "z", "w"] {
            for c in ["cond_path", "uncond_path"] {
                let k = (v.to_string(), t, c.to_string());
                let shift = if v == "x" { g * (120 + t) as f64 } else { 0.0 };
                close(ct[&k], cn[&k] + shift, 1e-9, &format!("conditional forecast {c} {v}[{t}]"));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// item 4: calib_smoother with loglinear, prefilter and first_obs
// ---------------------------------------------------------------------------

fn csv_of(v: &[f64]) -> String {
    format!("y\n{}\n", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("\n"))
}

/// One shock and one observed variable: the smoother reproduces the data exactly, so the
/// smoothed variable is the (logged) data and, for t >= 2, the smoothed shock is
/// `e_t = d_t - rho d_{t-1}` with `d` the data minus its constant (log ybar, or the sample
/// mean of log y under prefilter; makedataset.m logs before it demeans).
#[test]
fn calib_smoother_loglinear_and_prefilter() {
    let n = 40;
    let y: Vec<f64> = (1..=n).map(|t| (0.5 + 0.3 * (t as f64 / 3.0).sin() + 0.1 * (t as f64).cos()).exp()).collect();
    let data = csv_of(&y);
    let rho = 0.9f64;
    let logy: Vec<f64> = y.iter().map(|v| v.ln()).collect();
    let mean = logy.iter().sum::<f64>() / n as f64;
    for prefilter in [false, true] {
        // dynare_estimation_init.m: prefilter needs observed variables with a zero (log) steady state
        let ybar = if prefilter { 1.0 } else { 2.0f64 };
        let tag = format!("pafll{}", prefilter as u8);
        let body = format!(
            "var y;\nvarexo e;\nparameters rho ybar;\nrho = {rho}; ybar = {ybar};\n\
             model;\nlog(y) = (1-rho)*log(ybar) + rho*log(y(-1)) + e;\nend;\n\
             initval;\ny = ybar;\nend;\nsteady;\nshocks;\nvar e; stderr 0.1;\nend;\nvarobs y;\n\
             calib_smoother(datafile='d.csv', loglinear{}) y;\n",
            if prefilter { ", prefilter=1" } else { "" }
        );
        let out = run_mod(&tag, &body, &[("d.csv", data.clone())], &[]);
        let got = column(&out.join(format!("{tag}_calib_smoothed_variables.csv")), "y");
        all_close(&got, &logy, 1e-9, &format!("{tag}: smoothed y = log data"));
        let c = if prefilter { mean } else { ybar.ln() };
        let d: Vec<f64> = logy.iter().map(|v| v - c).collect();
        let shocks = column(&out.join(format!("{tag}_calib_smoothed_shocks.csv")), "e");
        let want: Vec<f64> = (1..n).map(|t| d[t] - rho * d[t - 1]).collect();
        all_close(&shocks[1..], &want, 1e-9, &format!("{tag}: smoothed shocks"));
    }
}

/// `first_obs` is the time index of `observation_trends`: the trend of the first sample row is
/// `g*first_obs` (compute_trend_coefficients.m), the sample starts at row `first_obs`.
#[test]
fn calib_smoother_first_obs_trend_index() {
    let n = 30;
    let y: Vec<f64> = (1..=n).map(|t| 0.04 * t as f64 + (t as f64).sin()).collect();
    let data = csv_of(&y);
    let (rho, g) = (0.9, 0.05);
    let tag = "pafobs";
    let body = format!(
        "var y;\nvarexo e;\nparameters rho g;\nrho = {rho}; g = {g};\n\
         model(linear);\ny = rho*y(-1) + e;\nend;\nshocks;\nvar e; stderr 0.5;\nend;\nvarobs y;\n\
         observation_trends;\ny (g);\nend;\ncalib_smoother(datafile='d.csv', first_obs=3) y;\n"
    );
    let out = run_mod(tag, &body, &[("d.csv", data)], &[]);
    let levels = column(&out.join(format!("{tag}_calib_smoothed_variables.csv")), "y");
    all_close(&levels, &y[2..], 1e-9, "smoothed y = data from row 3");
    // sample row s (1-based) has the time index first_obs + s - 1 = s + 2
    let d: Vec<f64> = y[2..].iter().enumerate().map(|(s, v)| v - g * (s + 3) as f64).collect();
    let shocks = column(&out.join(format!("{tag}_calib_smoothed_shocks.csv")), "e");
    let want: Vec<f64> = (1..d.len()).map(|t| d[t] - rho * d[t - 1]).collect();
    all_close(&shocks[1..], &want, 1e-9, "smoothed shocks of the detrended data");
}

// ---------------------------------------------------------------------------
// the posterior objects of prior_posterior_statistics (estimation-6) and loglinear /
// prefilter posterior forecasts (estimation-1)
// ---------------------------------------------------------------------------

const AR1: &str = "var y;\nvarexo e;\nparameters rho;\nrho = 0.9;\nmodel(linear);\ny = rho*y(-1) + e;\nend;\n\
                   shocks;\nvar e; stderr 0.5;\nend;\nvarobs y;\nestimated_params;\nstderr e, inv_gamma_pdf, 0.5, 2;\nend;\n";

/// AR(1) with y observed and no measurement error: the state is known after each observation,
/// so (DK 2012) P_1 = sigma^2/(1-rho^2), P_t = sigma^2 for t >= 2 (`FilterCovariance`), the
/// smoothed state variance is 0 (`StateUncertainty`), the smoothed initial state is
/// `E[y_0|y_1] = rho y_1` (`Init_State`) with variance `sigma^2` (`InitStateUncertainty`).
#[test]
fn posterior_init_state_uncertainty_and_filter_covariance() {
    let mf = parse_mod_str(AR1, &[]).unwrap();
    let dsge = DsgeModel::new(&mf, None).unwrap();
    let y = [1.0, 0.7, -0.3, 0.2, 0.9, 1.4, 0.8, 0.1, -0.5, -0.2, 0.4, 0.6];
    let sample = Mat::from_fn(y.len(), 1, |t, _| y[t]);
    let data = LikelihoodData::new(&sample, KalmanOptions::default());
    let draws = vec![vec![0.5]; 12];
    let o = PosteriorSmootherOptions {
        smoother: true,
        filtered_vars: false,
        horizon: 0,
        mh_conf_sig: 0.9,
        vars: vec![0],
        var_names: vec!["y".into()],
        exo_names: vec!["e".into()],
        seed: 1,
        kernel_density: false,
        state_uncertainty: true,
        filter_covariance: true,
    };
    let p = prior_posterior_statistics(&dsge, &data, &draws, &o).unwrap();
    let (rho, sig2) = (0.9, 0.25);
    close(p.init_state.as_ref().unwrap().mean[(0, 0)], rho * y[0], 1e-10, "Init_State");
    let iu = p.init_state_uncertainty.as_ref().unwrap();
    close(iu.mean[0][(0, 0)], sig2, 1e-10, "InitStateUncertainty");
    let fc = p.filter_covariance.as_ref().unwrap();
    assert_eq!(fc.mean.len(), y.len() + 1);
    close(fc.mean[0][(0, 0)], sig2 / (1.0 - rho * rho), 1e-10, "FilterCovariance(1)");
    for t in 1..=y.len() {
        close(fc.mean[t][(0, 0)], sig2, 1e-10, &format!("FilterCovariance({})", t + 1));
    }
    let su = p.state_uncertainty.as_ref().unwrap();
    assert_eq!(su.mean.len(), y.len());
    for (t, m) in su.mean.iter().enumerate() {
        assert!(m[(0, 0)].abs() < 1e-10, "StateUncertainty({t}) = {}", m[(0, 0)]);
    }
    // identical draws: no dispersion
    assert!(fc.var[1][(0, 0)].abs() < 1e-20 && fc.hpd_inf[1][(0, 0)] == fc.hpd_sup[1][(0, 0)]);
}

/// Loglinear posterior forecasts are in logs: with `rho` and `ybar` calibrated and the shock
/// the only estimated parameter, every draw smooths `log y_T` exactly, so
/// `MeanForecast(h) = log(ybar) + rho^h (log y_T - log ybar)` (prior_posterior_statistics_core.m
/// adds `log(SteadyState)` under loglinear). With `prefilter` the data are demeaned and
/// Dynare adds the sample mean *and* `log(SteadyState)`: `rho^h (log y_T - mean) + mean + log ybar`.
#[test]
fn posterior_forecast_loglinear_and_prefilter() {
    let n = 40;
    let y: Vec<f64> = (1..=n).map(|t| (0.5 + 0.3 * (t as f64 / 3.0).sin() + 0.1 * (t as f64).cos()).exp()).collect();
    let data = csv_of(&y);
    let rho = 0.9f64;
    let logy: Vec<f64> = y.iter().map(|v| v.ln()).collect();
    let mean = logy.iter().sum::<f64>() / n as f64;
    for prefilter in [false, true] {
        // dynare_estimation_init.m: prefilter needs observed variables with a zero (log) steady state
        let ybar = if prefilter { 1.0 } else { 2.0f64 };
        let tag = format!("pafpf{}", prefilter as u8);
        let body = format!(
            "var y;\nvarexo e;\nparameters rho ybar;\nrho = {rho}; ybar = {ybar};\n\
             model;\nlog(y) = (1-rho)*log(ybar) + rho*log(y(-1)) + e;\nend;\n\
             initval;\ny = ybar;\nend;\nsteady;\nshocks;\nvar e; stderr 0.1;\nend;\nvarobs y;\n\
             estimated_params;\nstderr e, inv_gamma_pdf, 0.1, 2;\nend;\n\
             estimation(datafile='d.csv', loglinear, mh_replic=300, mh_nblocks=1, mh_jscale=1.5, \
             nograph, smoother, forecast=4{});\n",
            if prefilter { ", prefilter=1" } else { "" }
        );
        let out = run_mod(&tag, &body, &[("d.csv", data.clone())], &[]);
        let rows = csv(&out.join(format!("{tag}_posterior_mean_forecast.csv")));
        let (iv, im) = (col(&rows[0], "variable"), col(&rows[0], "mean"));
        let got: Vec<f64> = rows[1..].iter().filter(|r| r[iv] == "y").map(|r| r[im].parse().unwrap()).collect();
        let yt = logy[n - 1];
        let want: Vec<f64> = (1..=4)
            .map(|h| {
                let rh = rho.powi(h);
                if prefilter { rh * (yt - mean) + mean + ybar.ln() } else { ybar.ln() + rh * (yt - ybar.ln()) }
            })
            .collect();
        all_close(&got, &want, 1e-8, &format!("{tag}: posterior MeanForecast"));
    }
}

/// `shock_decomposition(forecast_type=unconditional)` with a Bayesian parameter set extends
/// the decomposition with `oo_.MeanForecast` (the posterior mean forecast), not with the
/// point forecast at the mode.
#[test]
fn shock_decomposition_unconditional_uses_mean_forecast() {
    let y = [1.0, 0.7, -0.3, 0.2, 0.9, 1.4, 0.8, 0.1, -0.5, -0.2, 0.4, 0.6];
    let data = csv_of(&y);
    let tag = "pafmf";
    let body = "var y;\nvarexo e;\nparameters rho;\nrho = 0.8;\nmodel(linear);\ny = rho*y(-1) + e;\nend;\n\
                shocks;\nvar e; stderr 0.5;\nend;\nvarobs y;\n\
                estimated_params;\nstderr e, inv_gamma_pdf, 0.5, 2;\nrho, beta_pdf, 0.8, 0.1;\nend;\n\
                estimation(datafile='d.csv', mh_replic=400, mh_nblocks=1, mh_jscale=1.5, nograph, smoother, forecast=3);\n\
                shock_decomposition(parameter_set=posterior_mean, forecast_type=unconditional) y;\n"
        .to_string();
    let out = run_mod(tag, &body, &[("d.csv", data)], &[]);
    let rows = csv(&out.join(format!("{tag}_posterior_mean_forecast.csv")));
    let (iv, im) = (col(&rows[0], "variable"), col(&rows[0], "mean"));
    let mean_fc: Vec<f64> = rows[1..].iter().filter(|r| r[iv] == "y").map(|r| r[im].parse().unwrap()).collect();
    let sd = csv(&out.join(format!("{tag}_shock_decomposition_unconditional_forecast.csv")));
    let (v, c, x) = (col(&sd[0], "variable"), col(&sd[0], "component"), col(&sd[0], "value"));
    let smoothed: Vec<f64> =
        sd[1..].iter().filter(|r| r[v] == "y" && r[c] == "Smoothed").map(|r| r[x].parse().unwrap()).collect();
    assert_eq!(smoothed.len(), y.len() + 3);
    all_close(&smoothed[y.len()..], &mean_fc, 1e-12, "decomposition over the posterior mean forecast");
}

// ---------------------------------------------------------------------------
// items 12, 13, 14: error behaviour of the decompositions and conditional forecasts
// ---------------------------------------------------------------------------

#[test]
fn shock_decomposition_without_parameter_set_or_estimation_stops() {
    let data = csv_of(&[1.0, 0.7, -0.3, 0.2, 0.9, 1.4]);
    let model = "var y;\nvarexo e;\nparameters rho;\nrho = 0.8;\nmodel(linear);\ny = rho*y(-1) + e;\nend;\n\
                 shocks;\nvar e; stderr 0.5;\nend;\nvarobs y;\n";
    for cmd in ["shock_decomposition(datafile='d.csv') y;", "realtime_shock_decomposition(datafile='d.csv') y;"] {
        let e = run_err("pafsd", &format!("{model}{cmd}\n"), &[("d.csv", data.clone())]);
        assert!(e.contains("option parameter_set is not specified and posterior mode is not available"), "{cmd}: {e}");
    }
    // with parameter_set = calibration it runs
    run_mod(
        "pafsd2",
        &format!("{model}shock_decomposition(datafile='d.csv', parameter_set=calibration) y;\n"),
        &[("d.csv", data)],
        &[],
    );
}

const TWO: &str = "var x z;\nvarexo ex ez;\nparameters a c;\na = 0.8; c = 0.5;\nmodel(linear);\n\
                   x = a*x(-1) + z(-1) + ex;\nz = c*z(-1) + ez;\nend;\n\
                   shocks;\nvar ex; stderr 1;\nvar ez; stderr 0.5;\nend;\nvarobs x z;\n\
                   estimated_params;\nstderr ex, 0.8, 0.01, 10;\nend;\n";

fn two_data() -> String {
    let mut s = String::from("x,z\n");
    for t in 1..=30 {
        s += &format!("{},{}\n", (t as f64 / 2.0).sin(), (t as f64 / 3.0).cos() * 0.5);
    }
    s
}

/// shock_decomposition.m reads `cf.controlled_exo_variables.Mean.(name)` of every shock: MATLAB
/// errors unless all of them are controlled.
#[test]
fn conditional_shock_decomposition_needs_every_shock_controlled() {
    let body = format!(
        "{TWO}estimation(datafile='d.csv', mode_compute=0, nograph, smoother);\n\
         conditional_forecast_paths;\nvar x;\nperiods 1:2;\nvalues 0.5;\nend;\n\
         conditional_forecast(parameter_set=posterior_mode, controlled_varexo=(ex), periods=4, replic=10) x z;\n\
         shock_decomposition(parameter_set=posterior_mode, forecast_type=conditional) x z;\n"
    );
    let e = run_err("pafcsd", &body, &[("d.csv", two_data())]);
    assert!(e.contains("Reference to non-existent field 'ez'"), "{e}");
}

/// get_shock_paths.m: `inv()` of a singular impact matrix only warns (Inf/NaN follow and run.m
/// reports 'controlling of variables was not successful'); the run continues. Here the
/// controlled shock `ex` has no contemporaneous effect on the constrained `z`.
#[test]
fn conditional_forecast_singular_impact_matrix_does_not_stop() {
    let body = format!(
        "{TWO}estimation(datafile='d.csv', mode_compute=0, nograph, smoother);\n\
         conditional_forecast_paths;\nvar z;\nperiods 1:2;\nvalues 0.5;\nend;\n\
         conditional_forecast(parameter_set=posterior_mode, controlled_varexo=(ex), periods=4, replic=10) x z;\n"
    );
    let out = run_mod("pafsing", &body, &[("d.csv", two_data())], &[]);
    assert!(out.join("pafsing_conditional_forecast.csv").exists());
}

// ---------------------------------------------------------------------------
// item 7: the one-shock simulations of the simulated variance decomposition start at histval
// ---------------------------------------------------------------------------

#[test]
fn simulated_variance_decomposition_starts_from_histval() {
    let src = "var y;\nvarexo e1 e2;\nparameters rho;\nrho = 0.9;\nmodel(linear);\ny = rho*y(-1) + e1 + e2;\nend;\n\
               histval;\ny(0) = 5;\nend;\nshocks;\nvar e1; stderr 1;\nvar e2; stderr 2;\nend;\n";
    let mf = parse_mod_str(src, &[]).unwrap();
    let sol = analysis::solve_calibration(&mf, 1.0 + 1e-6).unwrap().unwrap();
    let periods = 30;
    let exo =
        Mat::from_fn(periods, 2, |t, i| if i == 0 { (t as f64 * 0.7).sin() } else { 2.0 * (t as f64 * 1.3).cos() });
    let dir = common::temp_dir("pafvd");
    let ctx = OutputContext { out_dir: dir.join("pafvd/Output"), dir, fname: "pafvd".into(), seed: 1 };
    let mut o = StochSimulOptions { periods, drop: 0, irf: 0, noprint: true, nograph: true, ..Default::default() };
    o.endo_histval = analysis::stochastic_histval_of(&mf.model, &mf.histval);
    assert_eq!(o.endo_histval, Some(vec![5.0]));
    let res = analysis::stoch_simul(&mf.model, &sol, &o, &[], &ctx, Some(&exo)).unwrap();
    let vd = res.simulated.unwrap().vardec.unwrap();
    // by hand: y_t = rho y_{t-1} + shocks, from y_0 = 5 (all shocks, then each shock alone)
    let sim = |e: &dyn Fn(usize) -> f64| -> f64 {
        let mut y = 5.0;
        let mut v = Vec::new();
        for t in 0..periods {
            y = 0.9 * y + e(t);
            v.push(y);
        }
        let m = v.iter().sum::<f64>() / periods as f64;
        v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (periods as f64 - 1.0)
    };
    // disp_moments.m: s2 = mean(y.^2) of the demeaned series (divides by T), var(...) by T-1
    let all = sim(&|t| exo[(t, 0)] + exo[(t, 1)]) * (periods as f64 - 1.0) / periods as f64;
    for i in 0..2 {
        let one = sim(&|t| exo[(t, i)]);
        close(vd[(0, i)], one / all * 100.0, 1e-9, &format!("variance decomposition, shock {i}"));
    }
}

// ---------------------------------------------------------------------------
// item 8: check on purely backward models
// ---------------------------------------------------------------------------

#[test]
fn check_backward_models() {
    // z = 1.5 z(-1) + y: an explosive root; Dynare prints the eigenvalues and the verdict
    // (dr.edim = 1, nsfwrd = 0: 'The order condition is NOT verified.'), it does not stop
    let explosive = "var y z;\nvarexo e;\nmodel;\ny = 0.9*y(-1) + e;\nz = 1.5*z(-1) + y;\nend;\n\
                     initval;\ny = 0;\nz = 0;\nend;\nsteady;\n";
    let stable = explosive.replace("1.5*z(-1)", "0.5*z(-1)");
    for (src, edim, ok, eig) in [(explosive, 1, false, [0.9, 1.5]), (stable.as_str(), 0, true, [0.5, 0.9])] {
        let mf = parse_mod_str(src, &[]).unwrap();
        let ys = analysis::calibrated_steady_state(&mf, &mf.model.params).unwrap();
        let c = analysis::check::check(&mf.model, &mf.model.params, &ys, Default::default()).unwrap();
        assert_eq!(c.edim, Some(edim));
        assert_eq!(c.nsfwrd, 0);
        assert_eq!(c.result, ok);
        let mut m: Vec<f64> = c.eigval.iter().map(|z| z.norm()).collect();
        m.sort_by(f64::total_cmp);
        all_close(&m, &eig, 1e-10, "eigenvalues");
        let text = analysis::check::format_check(&c);
        assert!(text.contains(&format!(
            "There are {edim} eigenvalue(s) larger than 1 in modulus for 0 forward-looking variable(s)."
        )));
        assert!(text.contains(if ok {
            "The order and rank conditions are verified."
        } else {
            "The order condition is NOT verified."
        }));
    }
}

// ---------------------------------------------------------------------------
// items 9 and 10: moments_varendo, conditional decomposition and measurement errors
// ---------------------------------------------------------------------------

const NK: &str = r#"
var y pi i a;
varexo e_a e_d e_i;
parameters beta sigma kappa phi_pi phi_y rho_a rho_i;
beta = 0.99; sigma = 1.5; kappa = 0.12; phi_pi = 1.6; phi_y = 0.4; rho_a = 0.8; rho_i = 0.6;
model(linear);
y = y(+1) - (1/sigma)*(i - pi(+1)) + e_d;
pi = beta*pi(+1) + kappa*y + a;
i = rho_i*i(-1) + (1-rho_i)*(phi_pi*pi + phi_y*y) + e_i;
a = rho_a*a(-1) + e_a;
end;
shocks;
var e_a; stderr 0.5;
var e_d; stderr 0.8;
var e_i; stderr 0.25;
var y; stderr 0.3;
end;
varobs y pi i;
estimated_params;
stderr e_a, inv_gamma_pdf, 0.5, 2;
stderr e_d, inv_gamma_pdf, 0.8, 2;
stderr e_i, inv_gamma_pdf, 0.25, 2;
rho_a, beta_pdf, 0.8, 0.1;
phi_pi, normal_pdf, 1.6, 0.2;
end;
"#;

fn chain_of(points: &[Vec<f64>]) -> Chain {
    Chain {
        npar: points[0].len(),
        draws: points.iter().flatten().copied().collect(),
        logpost: vec![0.0; points.len()],
        runs: Vec::new(),
        first_stored: 0,
        total_draws: points.len(),
        accepted: points.len(),
        init: points[0].clone(),
        init_logpost: 0.0,
    }
}

#[test]
fn posterior_moments_varendo_conditional_decomposition_and_measurement_errors() {
    let mf = parse_mod_str(NK, &[]).unwrap();
    let dsge = DsgeModel::new(&mf, None).unwrap();
    let x = vec![0.5, 0.8, 0.25, 0.8, 1.6];
    let r = dsge.resolve(&x).unwrap();
    let ivar = vec![0usize, 1, 2];
    let draws = analysis::posterior::posterior_subsample(&[chain_of(&vec![x.clone(); 20])], 12, 3).unwrap();
    let steps = vec![1usize, 4];
    // ---- nodecomposition: the conditional decomposition is still computed (compute_moments_varendo.m)
    let o = StochSimulOptions {
        nodecomposition: true,
        conditional_variance_decomposition: steps.clone(),
        ..Default::default()
    };
    let pm = analysis::posterior::posterior_moments_varendo(&dsge, &draws, &ivar, &o).unwrap();
    assert!(pm.variance_decomposition.is_none());
    let cvd = vardec::conditional_variance_decomposition(&r.dr, &r.q, &steps, &ivar).unwrap();
    let cond = pm.conditional.as_ref().expect("conditional decomposition with nodecomposition");
    for h in 0..2 {
        for i in 0..3 {
            for s in 0..3 {
                close(cond[h][i][s].mean, cvd[h][(i, s)], 1e-12, "conditional decomposition");
            }
        }
    }
    assert!(pm.variance_decomposition_me.is_none(), "no ME decomposition under nodecomposition");
    // ---- with measurement error on y: VarianceDecompositionME and its conditional version
    let o = StochSimulOptions { conditional_variance_decomposition: steps.clone(), ..Default::default() };
    let pm = analysis::posterior::posterior_moments_varendo(&dsge, &draws, &ivar, &o).unwrap();
    let th = moments::th_autocovariances(&r.dr, &r.q, &ivar, &o.moment_options()).unwrap();
    let vd = th.vardec.as_ref().unwrap();
    let me = pm.variance_decomposition_me.as_ref().expect("VarianceDecompositionME");
    assert_eq!(me.vars, ivar);
    let hvar = [0.09, 0.0, 0.0];
    for k in 0..3 {
        let v = th.gamma0[(k, k)];
        let parts: Vec<f64> = (0..3).map(|j| vd[(k, j)] * v / (v + hvar[k])).collect();
        for j in 0..3 {
            close(me.stats[k][j].mean, parts[j], 1e-12, &format!("ME decomposition ({k},{j})"));
        }
        close(me.stats[k][3].mean, 1.0 - parts.iter().sum::<f64>(), 1e-12, &format!("ME share of {k}"));
    }
    let cv = vardec::conditional_variances(&r.dr, &r.q, &steps, &ivar).unwrap();
    let cme = pm.conditional_me.as_ref().expect("ConditionalVarianceDecompositionME");
    for h in 0..2 {
        for k in 0..3 {
            let tot: f64 = (0..3).map(|j| cv[h][(k, j)]).sum::<f64>() + hvar[k];
            for j in 0..3 {
                close(cme.stats[h][k][j].mean, cv[h][(k, j)] / tot, 1e-12, "conditional ME decomposition");
            }
            close(cme.stats[h][k][3].mean, hvar[k] / tot, 1e-12, "conditional ME share");
        }
    }
}

// ---------------------------------------------------------------------------
// item 15: order >= 2 forecast with measurement errors; item 16: bandpass passbands
// ---------------------------------------------------------------------------

#[test]
fn forecast_order2_measurement_error_bands() {
    let body = "var y junk;\nvarexo e;\nparameters rho;\nrho = 0.8;\nmodel;\ny = 0.5 + rho*(y(-1)-0.5) + e + 0.2*(y(-1)-0.5)^2;\njunk = 0.5*junk(+1);\nend;\n\
                initval;\ny = 0.5;\njunk = 0;\nend;\nsteady;\nshocks;\nvar e; stderr 0.2;\nvar y; stderr 0.3;\nend;\nvarobs y;\n\
                stoch_simul(order=2, irf=0, nograph, noprint);\nforecast(periods=3);\n";
    let out = run_mod("pafme2", body, &[], &[]);
    let rows = csv(&out.join("pafme2_forecast.csv"));
    let (il, iu) = (col(&rows[0], "lower"), col(&rows[0], "upper"));
    let (ilm, ium) = (col(&rows[0], "lower_me"), col(&rows[0], "upper_me"));
    let iv = col(&rows[0], "variable");
    let mut n = 0;
    for r in rows[1..].iter().filter(|r| r[iv] == "y") {
        n += 1;
        let w: f64 = r[iu].parse::<f64>().unwrap() - r[il].parse::<f64>().unwrap();
        let wm: f64 = r[ium].parse::<f64>().unwrap() - r[ilm].parse::<f64>().unwrap();
        // the 90% band of (y + N(0, 0.09)) for a near-normal y: width^2 grows by (2*1.645*0.3)^2
        let want = (w * w + (2.0 * 1.6449 * 0.3f64).powi(2)).sqrt();
        assert!(wm > w && (wm / want - 1.0).abs() < 0.08, "ME band width {wm} vs {want} (without ME {w})");
    }
    assert_eq!(n, 3);
}

#[test]
fn bandpass_passband_is_not_checked_when_k_is_given() {
    // baxter_king_filter_.m only checks `high >= 2` and `high <= low` when K is omitted
    let y = Mat::from_fn(40, 1, |t, _| (t as f64 * 0.9).sin() + 0.1 * t as f64);
    let f = analysis::q2::baxter_king_filter(&y, 1.5, 8.0, 4).expect("passband with high < 2");
    assert_eq!(f.nrows(), 40 - 8);
    assert!(f.iter().all(|v| v.is_finite()));
    analysis::q2::baxter_king_filter(&y, 10.0, 8.0, 4).expect("passband with high > low");
}
