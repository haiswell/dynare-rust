//! `conditional_forecast(parameter_set = calibration)` starts from `M_.endo_histval`
//! (conditional_forecasts/run.m, `estimated_model = 0`): after `smoother2histval`, the
//! zero-shock unconditional path starts at the last smoothed values and equals the order-1
//! `forecast` from the same histval (checked against Dynare in tests/smoother2histval.rs).
//!   - levels: fs2000 after `calib_smoother` (reference/fs2000_cf); `stoch_simul(order=1)`
//!     first, as `forecast` runs at `options_.order`;
//!   - loglinear: the rbc of reference/opts after `estimation(loglinear, mode_file=...)`
//!     (smoothed variables, forecasts and conditional forecasts in logs).
mod common;

use std::collections::HashMap;
use std::path::Path;

use dynare::driver::{RunOptions, run};

fn csv(p: &Path) -> Vec<Vec<String>> {
    let s = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    s.lines().map(|l| l.split(',').map(str::to_string).collect()).collect()
}

fn col(h: &[String], n: &str) -> usize {
    h.iter().position(|s| s == n).unwrap_or_else(|| panic!("no column {n}"))
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

/// Runs `body` (the commands after the model) on a copy of `files` from reference/`refdir`;
/// `smoothed` is the smoothed-variables CSV the histval comes from.
fn check(tag: &str, refdir: &str, files: &[&str], body: &str, smoothed: &str, vars: &[&str]) {
    let refdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference").join(refdir);
    let dir = common::temp_dir(tag);
    for f in files {
        std::fs::copy(refdir.join(f), dir.join(f)).unwrap();
    }
    std::fs::write(dir.join(format!("{tag}.mod")), body).unwrap();
    run(&dir.join(format!("{tag}.mod")), &RunOptions { quiet: true, ..Default::default() }).unwrap();
    let out = dir.join(format!("{tag}/Output"));

    let sm = csv(&out.join(format!("{tag}_{smoothed}.csv")));
    let last = sm.last().unwrap();
    let fc = csv(&out.join(format!("{tag}_forecast.csv")));
    let (fv, fh, fm) = (col(&fc[0], "variable"), col(&fc[0], "horizon"), col(&fc[0], "mean"));
    let forecast: HashMap<(String, usize), f64> =
        fc[1..].iter().map(|r| ((r[fv].clone(), r[fh].parse().unwrap()), r[fm].parse().unwrap())).collect();

    let cf = csv(&out.join(format!("{tag}_conditional_forecast.csv")));
    let (cv, ch, cu) = (col(&cf[0], "variable"), col(&cf[0], "horizon"), col(&cf[0], "uncond_path"));
    let mut n = 0;
    for r in &cf[1..] {
        let v = r[cv].as_str();
        if !vars.contains(&v) {
            continue;
        }
        let t: usize = r[ch].parse().unwrap();
        let u: f64 = r[cu].parse().unwrap();
        let want = if t == 0 { last[col(&sm[0], v)].parse().unwrap() } else { forecast[&(v.to_string(), t)] };
        close(u, want, 1e-10, &format!("{tag}: uncond {v}[{t}]"));
        n += 1;
    }
    assert_eq!(n, vars.len() * 11, "{tag}: rows");
}

#[test]
fn conditional_forecast_calibration_starts_from_histval() {
    let vars = ["m", "P", "c", "k", "y", "gp_obs"];
    let v = vars.join(" ");
    let body = format!(
        "@#include \"fs2000_model.inc\"\n\
         varobs gp_obs gy_obs;\n\
         stoch_simul(order=1, irf=0, noprint, nograph);\n\
         calib_smoother(datafile='fs2000_cf_data.csv') {v};\n\
         smoother2histval;\n\
         conditional_forecast_paths;\n\
         var gp_obs;\n\
         periods 1:2, 3:5;\n\
         values 1.012, 1.008;\n\
         end;\n\
         conditional_forecast(parameter_set=calibration, controlled_varexo=(e_m), periods=10, replic=50);\n\
         forecast(periods=10) {v};\n"
    );
    let files = ["fs2000_model.inc", "fs2000_cf_data.csv"];
    check("cfh_levels", "fs2000_cf", &files, &body, "calib_smoothed_variables", &vars);
}

#[test]
fn conditional_forecast_calibration_starts_from_histval_loglinear() {
    let src =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/opts/model_rbc.mod")).unwrap();
    // the declarations, model, priors and varobs of model_rbc.mod, without its STAGE commands
    let decl = &src[src.find("var c k a y;").unwrap()..src.find("@#if STAGE == 1").unwrap()];
    let body = format!(
        "{decl}\
         estimation(datafile='rbc_levels.csv', loglinear, mode_file='rbc_mode', mode_compute=0, mh_replic=0, nograph, smoother);\n\
         smoother2histval;\n\
         conditional_forecast_paths;\n\
         var y;\n\
         periods 1:2, 3;\n\
         values 1.12, 1.13;\n\
         end;\n\
         conditional_forecast(parameter_set=calibration, controlled_varexo=(e), periods=10, replic=50);\n\
         forecast(periods=10) c k a y;\n"
    );
    let files = ["rbc_levels.csv", "rbc_mode.mat"];
    check("cfh_loglin", "opts", &files, &body, "smoothed_variables", &["c", "k", "a", "y"]);
}
