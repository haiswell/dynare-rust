//! shock_decomposition/ls2003_plot.mod (Dynare 7.2 tests) against MATLAB
//! (reference/sd_recompute/ref.json, written by reference/sd_recompute/dump.py):
//! * plot_shock_decomposition of a variable that squeeze_shock_decomposition dropped recomputes
//!   shock_decomposition, realtime_shock_decomposition and the initval decomposition for every
//!   variable (plot_shock_decomposition.m's catch branch), then type=aoa for y and the epilogue
//!   variable ya, realtime=1/2/3 with vintage, interactive (accepted, static figures);
//! * options_.shock_decomp persists between commands: with_epilogue of shock_decomposition
//!   reaches realtime_shock_decomposition and initial_condition_decomposition, and
//!   save_realtime / forecast reach the later fast_realtime run.
mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use dynare::driver::{RunOptions, run};
use serde_json::Value;

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sd_recompute")
}

/// A long csv (`[period,]date,variable,component,value`) as (variable, component) -> values,
/// plus the components in file order.
fn read_long(p: &Path) -> (HashMap<(String, String), Vec<f64>>, Vec<String>) {
    let text = common::read_ref(p);
    let mut lines = text.lines();
    let off = usize::from(lines.next().unwrap().starts_with("period,"));
    let mut m: HashMap<(String, String), Vec<f64>> = HashMap::new();
    let mut comps = Vec::new();
    for l in lines {
        let f: Vec<&str> = l.split(',').collect();
        let c = f[off + 2].to_string();
        if !comps.contains(&c) {
            comps.push(c.clone());
        }
        m.entry((f[off + 1].into(), c)).or_default().push(f[off + 3].parse().unwrap_or(f64::NAN));
    }
    (m, comps)
}

/// Compares `vars` of a csv with MATLAB's `[var][component][period]` array; MATLAB's
/// get_decomp_only array is before the plot_init_date window, so it is matched on its last
/// periods (type=aoa starts the plot in the first full year).
fn check(p: &Path, want: &Value, vars: &[&str]) -> usize {
    let (got, comps) = read_long(p);
    let mut n = 0;
    for (vi, v) in vars.iter().enumerate() {
        let wv = want[vi].as_array().unwrap();
        assert_eq!(wv.len(), comps.len(), "{}: components", p.display());
        for (ci, c) in comps.iter().enumerate() {
            let w: Vec<f64> = wv[ci].as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
            let h = &got[&(v.to_string(), c.clone())];
            assert!(
                h.len() <= w.len() && h.len() + 1 >= w.len(),
                "{} {v} {c}: {} vs {} periods",
                p.display(),
                h.len(),
                w.len()
            );
            let w = &w[w.len() - h.len()..];
            for (t, (a, b)) in h.iter().zip(w).enumerate() {
                let ok = (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * (1.0 + b.abs());
                assert!(ok, "{} {v} {c} t={t}: rust {a} vs dynare {b}", p.display());
                n += 1;
            }
        }
    }
    n
}

#[test]
fn plot_recomputes_squeezed_decompositions() {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    let dir = common::temp_dir("sd_recompute");
    std::fs::copy(refdir().join("../sweep_fix2/data_ca1.m"), dir.join("data_ca1.m")).unwrap();
    // the epilogue variable ya in the realtime var list (the preprocessor refuses it; dynare-rs
    // writes the listed rows)
    let modtext = common::read_ref(refdir().join("ls2003_recompute.mod"))
        .replace("(fast_realtime=75) y y_obs;", "(fast_realtime=75) y y_obs ya;");
    std::fs::write(dir.join("ls2003_recompute.mod"), modtext).unwrap();
    run(&dir.join("ls2003_recompute.mod"), &RunOptions { quiet: true, ..Default::default() }).unwrap();
    let out = dir.join("ls2003_recompute/Output");
    let f = |s: &str| out.join(format!("ls2003_recompute_{s}.csv"));
    let v5 = ["y_obs", "R_obs", "pie_obs", "dq", "de"];
    let zz = &r["zz"];
    let mut n = 0;
    n += check(&f("shock_decomposition_P1_calibration"), &zz[0], &v5);
    n += check(&f("shock_decomposition_P2_realtime_rolling_calibration"), &zz[1], &v5);
    // y was squeezed out: recomputed
    n += check(&f("shock_decomposition_aoa_P3_calibration"), &zz[2], &["y"]);
    n += check(&f("shock_decomposition_aoa_P4_calibration"), &zz[3], &["ya"]);
    n += check(&f("shock_decomposition_P5_realtime_29Y_calibration"), &zz[4], &["y"]);
    n += check(&f("shock_decomposition_P6_8step_ahead_conditional_forecast_29Y_calibration"), &zz[5], &["y"]);
    n += check(&f("shock_decomposition_P7_8step_ahead_forecast_29Y_calibration"), &zz[6], &["y"]);
    n += check(&f("shock_decomposition_P8_8step_ahead_forecast_29Y_calibration"), &zz[7], &v5);
    // initial_condition_decomposition R_obs (before the recomputation)
    n += check(&f("initval_decomposition"), &r["iv_R_obs"], &["R_obs"]);
    // fast_realtime=75 with forecast=8 and save_realtime of the first run; with_epilogue rows
    n += check(&f("realtime_shock_decomposition_pool"), &r["fast_pool"], &["y", "ya"]);
    n += check(&f("realtime_forecast_shock_decomposition_time_29"), &r["fast_f29"], &["y", "ya"]);
    assert!(n > 8000, "{n} values compared");
}
