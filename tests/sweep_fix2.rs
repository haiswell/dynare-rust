//! Divergences found by running Dynare 7.2's own test suite (numbers from the MATLAB runs
//! named in reference/sweep_fix2/ref.json):
//! * a `.m` datafile with `...` continuation, `reshape(x,m,n)'` and `data(:,k)` (gsa/ls2003.mod);
//! * endogenous variables in the values of `conditional_forecast_paths` are `oo_.steady_state`
//!   (shock_decomposition/fs2000_forecasts.mod);
//! * `rplot` after `stoch_simul(periods=...)` plots `oo_.endo_simul` / `oo_.exo_simul`, with
//!   `options_.rplottype` 0, 1 and 2 (TeX/fs2000_corr_ME.mod).
mod common;

use dynare::data::Table;
use dynare::driver::{RunOptions, run};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sweep_fix2")
}

fn setup(tag: &str, files: &[&str]) -> PathBuf {
    let dir = common::temp_dir(tag);
    for f in files {
        // fsdat_simul.m is shared with sweep_numfix
        let src = if *f == "fsdat_simul.m" { refdir().join("../sweep_numfix").join(f) } else { refdir().join(f) };
        std::fs::copy(&src, dir.join(f)).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
    }
    dir
}

fn quiet() -> RunOptions {
    RunOptions { quiet: true, ..Default::default() }
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn m_datafile_with_continuation_and_reshape() {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    let t = Table::read(&refdir().join("data_ca1.m"), &Default::default()).unwrap();
    // `data` is a matrix: only the column vectors are series
    assert_eq!(t.names, ["R_obs", "de", "dq", "pie_obs", "y_obs"]);
    assert_eq!(t.values.nrows(), r["data_ca1"]["n"].as_u64().unwrap() as usize);
    for k in ["y_obs", "R_obs", "de", "dq"] {
        let j = t.column(k).unwrap();
        for (i, w) in floats(&r["data_ca1"][k]).iter().enumerate() {
            assert_eq!(t.values[(i, j)], *w, "{k}[{i}]");
        }
    }
    // the second column of the first row is pie_obs(1)
    assert_eq!(t.values[(0, t.column("pie_obs").unwrap())], 11.8716889412);
}

/// The conditional-forecast CSV: the controlled path of `var` (horizons 1..5, `cond_mean`).
fn cond_mean(csv: &Path, var: &str) -> Vec<f64> {
    let s = std::fs::read_to_string(csv).unwrap();
    s.lines()
        .skip(1)
        .map(|l| l.split(',').collect::<Vec<_>>())
        .filter(|c| c[2] == var && c[1].parse::<usize>().unwrap() >= 1)
        .take(5)
        .map(|c| c[3].parse().unwrap())
        .collect()
}

fn check_paths(csv: &Path, r: &Value, prefix: &str) {
    for v in ["gy_obs", "gp_obs"] {
        let got = cond_mean(csv, v);
        let want = floats(&r[format!("{prefix}_cond_{v}")]);
        assert_eq!(got.len(), 5);
        for (h, (a, w)) in got.iter().zip(&want).enumerate() {
            assert!((a - w).abs() <= 1e-8, "{prefix} {v}[{h}]: rust {a} vs dynare {w}");
        }
    }
}

#[test]
fn endogenous_variables_in_conditional_forecast_paths_are_steady_state_values() {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    // stoch_simul leaves oo_.steady_state at zero: the paths are 0.01, -0.02, 0, 0, 0
    let dir = setup("sweepfix2_cf_calib", &["cf_paths_calib.mod", "fs2000.inc"]);
    run(&dir.join("cf_paths_calib.mod"), &quiet()).unwrap();
    check_paths(&dir.join("cf_paths_calib/Output/cf_paths_calib_conditional_forecast.csv"), &r, "calib");
    // after an estimation: the steady state at its initial point (gam = 0.0085)
    let dir = setup("sweepfix2_cf_est", &["cf_paths_est.mod", "fs2000.inc", "fsdat_simul.m"]);
    run(&dir.join("cf_paths_est.mod"), &quiet()).unwrap();
    check_paths(&dir.join("cf_paths_est/Output/cf_paths_est_conditional_forecast.csv"), &r, "est");
}

#[test]
fn rplot_after_stoch_simul() {
    let dir = setup("sweepfix2_rplot", &["rplot_stoch.mod"]);
    run(&dir.join("rplot_stoch.mod"), &quiet()).unwrap();
    let g = dir.join("rplot_stoch/graphs");
    let names: Vec<String> =
        std::fs::read_dir(&g).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    // type 0: one figure named after the first variable; type 1: one per variable
    for want in ["SimulatedTrajectory_y", "SimulatedTrajectory_a"] {
        assert!(names.iter().any(|n| n.contains(want)), "{want} not in {names:?}");
    }
}
