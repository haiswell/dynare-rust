//!  against Dynare 7.2 (reference/misc/ref_misc.json from reference/misc/dump_misc.m):
//! `lik_init` 2/5, `observation_trends` (with and without `prefilter`), `.m` datafiles,
//! `estimated_params_bounds` / `estimated_params_remove` (x_lik.mod); `resid`, `model_info`,
//! `prior table`, `save_params_and_steady_state` and parameter changes between commands
//! (x_misc.mod); the series of Dynare's `schorfheide_2000_data.m`.
mod common;

use std::path::{Path, PathBuf};

use dynare::data::Table;
use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
use dynare::estimation::Objective;
use dynare::kalman::KalmanOptions;
use dynare::modfile::parse_mod_file;
use serde_json::Value;

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

fn src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/misc")
}

fn reference() -> Option<Value> {
    let text = common::read_ref(src().join("ref_misc.json"));
    Some(serde_json::from_str(&text).unwrap())
}

fn defines(s: &str) -> Vec<(String, String)> {
    s.split_whitespace()
        .map(|d| {
            let d = d.trim_start_matches("-D");
            match d.split_once('=') {
                Some((k, v)) => (k.to_string(), v.to_string()),
                None => (d.to_string(), String::new()),
            }
        })
        .collect()
}

#[test]
fn likelihood_lik_init_trends_mfile() {
    let r = reference().expect("reference/setup unavailable");
    for run in r["runs"].as_array().unwrap() {
        let name = run["name"].as_str().unwrap();
        let defs = defines(run["defines"].as_str().unwrap());
        let mf = parse_mod_file(&src().join("x_lik.mod"), &defs).unwrap();
        let lik_init = defs.iter().find(|(k, _)| k == "LIKINIT").map_or(1, |(_, v)| v.parse::<u8>().unwrap());
        if lik_init == 4 {
            // Dynare 7.1's kalman_steady_state MEX does not load on this platform (SLICOT symbol
            // missing): MATLAB falls back to lik_init=1 and then fails in the smoother. Checked
            // by the unit tests of kalman::lik_init instead.
            continue;
        }
        let dsge = DsgeModel::new(&mf, None).unwrap();
        let datafile = if defs.iter().any(|(k, _)| k == "MFILE") { "x_lik_data.m" } else { "x_lik_obs.csv" };
        let table = Table::read(&src().join(datafile), &Default::default()).unwrap();
        let sample = table.sample(&mf.varobs, 1, None).unwrap();
        let mut data = LikelihoodData::new(&sample, KalmanOptions { lik_init, ..Default::default() });
        if defs.iter().any(|(k, _)| k == "PREFILTER") {
            data.prefilter();
        }
        let post = DsgePosterior { dsge: &dsge, data: &data, bounds: &dsge.bounds };
        let err = run["error"].as_str().unwrap_or("");
        if !err.is_empty() {
            // UNITROOT with lik_init=5: dsge_likelihood.m's eye(nunit) Pstar has the wrong size
            let (f, info) = post.eval(&dsge.init_vector().unwrap());
            assert!(f.is_infinite() && info != 0, "{name}: Dynare fails ({err}), dynare-rust gives {f}");
            continue;
        }
        assert_eq!(dsge.bounds.lb, vecf(&run["bounds"]["lb"]), "{name}: lower bounds");
        assert_eq!(dsge.bounds.ub, vecf(&run["bounds"]["ub"]), "{name}: upper bounds");
        assert_eq!(dsge.init_vector().unwrap(), vecf(&run["xparam_init"]), "{name}");
        for p in run["points"].as_array().unwrap() {
            let (f, _) = post.eval(&vecf(&p["x"]));
            let want = p["minus_loglik"].as_f64().unwrap();
            assert!(((f - want) / want).abs() < 1e-12, "{name}: fval {f} vs Dynare {want}");
        }
        // smoothed variables at the initial values (mode_compute=0, smoother)
        let sm = dynare::driver::run_smoother(&dsge, &data, &dsge.init_vector().unwrap(), false).unwrap();
        let mut worst: f64 = 0.0;
        for (j, v) in ["x", "z", "w"].iter().enumerate() {
            let want = vecf(&run["SmoothedVariables"][v]);
            for (t, w) in want.iter().enumerate() {
                worst = worst.max((sm.smoothed[(t, j)] - w).abs());
            }
        }
        println!("{name}: smoothed variables max |diff| {worst:.1e}");
        assert!(worst < 1e-9, "{name}: smoothed variables differ by {worst}");
    }
}

/// The block of console output that starts with `start` (first occurrence at or after `from`)
/// and ends before the next line starting with one of `stops`.
fn block<'a>(text: &'a str, start: &str, from: usize) -> (&'a str, usize) {
    let s = from + text[from..].find(start).unwrap_or_else(|| panic!("`{start}` not found"));
    (&text[s..], s + start.len())
}

fn norm_lines(s: &str, n: usize) -> Vec<String> {
    s.lines().filter(|l| !l.trim().is_empty()).take(n).map(|l| l.trim_end().to_string()).collect()
}

#[test]
fn misc_commands_against_dynare() {
    let r = reference().expect("reference/setup unavailable");
    let console = r["misc_console"].as_str().unwrap();
    let mf = parse_mod_file(&src().join("x_misc.mod"), &[]).unwrap();
    let pos = |name: &str, nth: usize| {
        mf.commands
            .iter()
            .enumerate()
            .filter(|(_, c)| matches!(c, dynare::modfile::Command::Other(n, _) if n == name))
            .nth(nth)
            .unwrap()
            .0
    };
    let opts = |ci: usize| match &mf.commands[ci] {
        dynare::modfile::Command::Other(_, o) => o.clone(),
        _ => unreachable!(),
    };

    // resid at the initval values, then resid(non_zero) after steady
    let (b1, after1) = block(console, "Residuals of the static equations:", 0);
    let ours = dynare::misc::resid_text(&mf, &opts(pos("resid", 0)), pos("resid", 0)).unwrap();
    assert_eq!(norm_lines(&ours, 6), norm_lines(b1, 6));
    let (b2, _) = block(console, "Residuals of the static equations:", after1);
    let ours = dynare::misc::resid_text(&mf, &opts(pos("resid", 1)), pos("resid", 1)).unwrap();
    assert_eq!(norm_lines(&ours, 2), norm_lines(b2, 2));

    // model_info
    let (b, _) = block(console, "Information about x_misc", 0);
    let ours = dynare::misc::model_info_text(&mf, &opts(pos("model_info", 0)), "x_misc").unwrap();
    assert_eq!(norm_lines(&ours, 11), norm_lines(b, 11));

    // prior table (rows trimmed: MATLAB pads them to the widest line)
    let dsge = DsgeModel::new(&mf, None).unwrap();
    let names: Vec<String> =
        dsge.estimated_names.iter().map(|n| dynare::estimation::plots::dynare_param_name(n)).collect();
    let ours = dynare::misc::prior_cmd::table_prior_text(&dsge, &names);
    let start = console.find(&"-".repeat(100)).unwrap();
    assert_eq!(norm_lines(&ours, 10), norm_lines(&console[start..], 10));

    // the whole file, run by the driver: save_params_and_steady_state and the parameter
    // changes between the two stoch_simul commands
    let dir = std::env::temp_dir().join(format!("dynare_rs_misc_x_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(src().join("x_misc.mod"), dir.join("x_misc.mod")).unwrap();
    dynare::driver::run(&dir.join("x_misc.mod"), &dynare::driver::RunOptions { quiet: true, ..Default::default() })
        .unwrap();
    let saved = std::fs::read_to_string(dir.join("x_misc_ss.txt")).unwrap();
    let want = r["misc_ss_file"].as_str().unwrap();
    // %.16g of the steady state: the last digits depend on where the solver stopped
    for (a, b) in saved.lines().zip(want.lines()) {
        let (na, va) = a.split_once(' ').unwrap();
        let (nb, vb) = b.split_once(' ').unwrap();
        assert_eq!(na, nb);
        let (va, vb): (f64, f64) = (va.parse().unwrap(), vb.parse().unwrap());
        assert!((va - vb).abs() <= 1e-12 * vb.abs().max(1.0), "{na}: {va} vs {vb}");
    }
    assert_eq!(saved.lines().count(), want.lines().count());
    let ss: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("x_misc/Output/x_misc_stoch_simul.json")).unwrap())
            .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    // oo_.var of the second stoch_simul (rho = 0.5, alpha = 0.3): Dynare starts the steady
    // state from the previous one, so both stop at different points within solve_tolf
    let (ours, want) = (vecf(&ss["var"]), vecf(&r["misc_var"]));
    for (a, b) in ours.iter().zip(&want) {
        assert!(((a - b) / b).abs() < 1e-6, "var {a} vs {b}");
    }
    let p = vecf(&r["misc_params"]);
    assert_eq!(p, vec![0.3, 0.99, 0.025, 0.5, 1.0]);
}

#[test]
fn m_datafile_schorfheide() {
    let r = reference().expect("reference/setup unavailable");
    let t = Table::read(&src().join("schorfheide_2000_data.m"), &Default::default()).unwrap();
    for (name, key) in [("gy_obs", "schorfheide_gy_obs"), ("gp_obs", "schorfheide_gp_obs")] {
        let j = t.column(name).unwrap();
        let want = vecf(&r[key]);
        let got: Vec<f64> = t.values.column(j).iter().copied().take(want.len()).collect();
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-13, "{name}: {a} vs {b}"); // log() rounding
        }
    }
}

#[test]
fn macro_language_against_dynare_preprocessor() {
    // x_macro_macroexp.mod: `dynare-preprocessor x_macro.mod savemacro onlymacro` (Dynare 7.1)
    let want = common::read_ref(src().join("x_macro_macroexp.mod"));
    let ours = dynare::modfile::macro_proc::preprocess(&src().join("x_macro.mod"), &[]).unwrap();
    // comments are kept in the savemacro output as in the preprocessor
    let ours: Vec<String> = ours.iter().map(|l| l.text.trim().to_string()).filter(|t| !t.is_empty()).collect();
    let want: Vec<String> = want.lines().map(|l| l.trim().to_string()).filter(|t| !t.is_empty()).collect();
    assert_eq!(ours, want);
    // and the parser accepts the expanded file
    let mf = parse_mod_file(&src().join("x_macro.mod"), &[]).unwrap();
    assert_eq!(mf.model.params, vec![11.0]);
}

#[test]
fn steady_homotopy_against_dynare() {
    let text = common::read_ref(src().join("ref_homotopy.json"));
    let r: Value = serde_json::from_str(&text).unwrap();
    for run in r["runs"].as_array().unwrap() {
        let mode = run["mode"].as_i64().unwrap();
        let mut mf = parse_mod_file(&src().join("x_homotopy.mod"), &[("MODE".into(), mode.to_string())]).unwrap();
        let mut det = dynare::deterministic::DetState::new(&mf).unwrap();
        let o = match mf.commands.last() {
            Some(dynare::modfile::Command::Steady(o)) => o.clone(),
            c => panic!("{c:?}"),
        };
        dynare::misc::homotopy::before_steady(&mut mf, Some(&mut det), &o).unwrap();
        det.steady(&mf.model, &o).unwrap();
        let (ys, want) = (&det.steady_state, vecf(&run["steady_state"]));
        for (a, b) in ys.iter().zip(&want) {
            assert!((a - b).abs() < 1e-10 * b.abs(), "mode {mode}: {a} vs {b}");
        }
        assert_eq!(det.params, vecf(&run["params"]));
        assert_eq!(det.exo_steady_state, vecf(&run["exo_steady_state"]));
    }
}
