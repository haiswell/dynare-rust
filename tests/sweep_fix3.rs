//! Divergences found by running Dynare 7.2's own test suite (numbers from the MATLAB runs
//! named in reference/sweep_fix3/ref.json):
//! * csminwel perturbs its search direction on a cliff with MATLAB's `rand` (the global
//!   mt19937ar stream seeded by `set_dynare_seed('default')`): the mode of
//!   estimation/ls2003_endog_prior_restrict_estimation.mod, where the `irf_calibration` /
//!   `moment_calibration` restrictions make cliffs, depends on those draws;
//! * one-element `nobs = [76]` / `first_obs = [8]` are scalars, also in each sample of a
//!   rolling-window estimation (forecast/ls2003_rolling_window_forecast.mod).
mod common;

use dynare::driver::{RunOptions, run};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sweep_fix3")
}

fn setup(tag: &str, files: &[&str]) -> PathBuf {
    let dir = common::temp_dir(tag);
    for f in files {
        // data_ca1.m is shared with sweep_fix2
        let src = if *f == "data_ca1.m" { refdir().join("../sweep_fix2").join(f) } else { refdir().join(f) };
        std::fs::copy(&src, dir.join(f)).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
    }
    dir
}

fn quiet() -> RunOptions {
    RunOptions { quiet: true, ..Default::default() }
}

fn results(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn csminwel_cliff_perturbation_draws_matlab_rand() {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    let r = &r["endog_prior_restrict"];
    let dir = setup("fix3_endog_prior", &["endog_prior_restrict.mod", "data_ca1.m"]);
    run(&dir.join("endog_prior_restrict.mod"), &quiet()).unwrap();
    let res = results(&dir.join("endog_prior_restrict/Output/endog_prior_restrict_results.json"));
    let ld = -res["minus_log_posterior_at_mode"].as_f64().unwrap();
    let want = r["log_density"].as_f64().unwrap();
    assert!((ld - want).abs() < 1e-6, "log density at the mode: rust {ld} vs dynare {want}");
    for (i, (a, w)) in floats(&res["mode"]).iter().zip(floats(&r["mode"])).enumerate() {
        assert!((a - w).abs() <= 1e-7 * (1.0 + w.abs()), "mode[{i}]: rust {a} vs dynare {w}");
    }
}

#[test]
fn rolling_window_with_one_element_nobs() {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    let dir = setup("fix3_rolling_window", &["rolling_window.mod", "data_ca1.m"]);
    run(&dir.join("rolling_window.mod"), &quiet()).unwrap();
    for k in ["8", "11"] {
        let w = &r["rolling_window"][k];
        let out = dir.join(format!("rolling_window_{k}/Output"));
        let res = results(&out.join("rolling_window_results.json"));
        // newrat stops at slightly different points of a flat posterior (its numerical Hessian
        // is ill-conditioned); the log posterior at the mode agrees, the rest within the
        // differences of the modes
        // (1.2e-5 apart with OpenBLAS on Linux)
        let ld = -res["minus_log_posterior_at_mode"].as_f64().unwrap();
        let want = w["log_density"].as_f64().unwrap();
        let tol = if cfg!(target_os = "macos") { 1e-5 } else { 1e-4 };
        assert!((ld - want).abs() < tol, "sample {k}: log density rust {ld} vs dynare {want}");
        for (i, (a, w)) in floats(&res["mode"]).iter().zip(floats(&w["mode"])).enumerate() {
            assert!((a - w).abs() <= 1e-2 * w.abs(), "sample {k} mode[{i}]: rust {a} vs dynare {w}");
        }
        let lap = res["log_data_density_laplace"].as_f64().unwrap();
        let want = w["laplace"].as_f64().unwrap();
        assert!((lap - want).abs() < 0.05, "sample {k}: Laplace rust {lap} vs dynare {want}");
        let csv = std::fs::read_to_string(out.join("rolling_window_forecast.csv")).unwrap();
        for v in ["y_obs", "R_obs", "pie_obs"] {
            let got: Vec<f64> = csv
                .lines()
                .skip(1)
                .map(|l| l.split(',').collect::<Vec<_>>())
                .filter(|x| x[2] == v)
                .map(|x| x[3].parse().unwrap())
                .collect();
            let want = floats(&w["forecast_mean"][v]);
            assert_eq!(got.len(), 8);
            for (h, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!((a - b).abs() <= 1e-2 * (1.0 + b.abs()), "sample {k} {v}[{h}]: rust {a} vs dynare {b}");
            }
        }
    }
}
