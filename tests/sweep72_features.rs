//! Features of Dynare 7.2 needed by its test suite:
//! - `ramsey_model` with `EXPECTATION` / `diff` in the model: the operators are substituted
//!   before the first-order conditions, so their aux variables get a FOC and their aux
//!   equations a multiplier (tests/optimal_policy/nk_ramsey_expectation.mod, and a variant with
//!   `diff` and initial values off the steady state; reference/sweep72/*.json from Dynare,
//!   `dump_ramsey_expectation.m`).
//! - recursive (`nobs = [76 79]`, with `forecast`) and rolling-window (`first_obs = [8 12]`)
//!   estimation, variants of tests/recursive/ls2003_bayesian_csv.mod (`dump_recursive.m`).
mod common;

use std::path::{Path, PathBuf};

use serde_json::Value;

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sweep72")
}

fn run_cli(dir: &Path, file: &str, extra: &[&str]) -> String {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_dynare-rs"))
        .args(extra)
        .args([file, "nograph", "nointeractive"])
        .current_dir(dir)
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{file}: {s}");
    s
}

fn floats(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(floats).collect(),
        Value::Number(x) => vec![x.as_f64().unwrap()],
        Value::Null => vec![f64::NAN],
        _ => panic!("not a number: {v}"),
    }
}

fn close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol * (1.0 + w.abs()), "{what}[{k}]: {g} vs Dynare {w}");
    }
}

fn check_ramsey(fname: &str) {
    let want: Value = common::json_ref(refdir().join(format!("{fname}.json")));
    let d = common::temp_dir(&format!("feat_{fname}"));
    std::fs::copy(refdir().join(format!("{fname}.mod")), d.join(format!("{fname}.mod"))).unwrap();
    // M_ fields
    let mf = dynare::modfile::parse_mod_file(&d.join(format!("{fname}.mod")), &[]).unwrap();
    let names: Vec<String> =
        want["endo_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().into()).collect();
    assert_eq!(mf.model.endo_names, names);
    let r = mf.model.planner().unwrap().ramsey.as_ref().unwrap();
    assert_eq!(r.orig_endo_nbr as u64, want["ramsey_orig_endo_nbr"].as_u64().unwrap());
    assert_eq!(r.orig_eq_nbr as u64, want["ramsey_orig_eq_nbr"].as_u64().unwrap());
    assert_eq!(
        mf.model.orig_maximum_lag_with_diffs_expanded as u64,
        want["orig_maximum_lag_with_diffs_expanded"].as_u64().unwrap()
    );
    // results
    run_cli(&d, &format!("{fname}.mod"), &[]);
    let out = d.join(fname).join("Output");
    let got: Value = common::json_ref(out.join(format!("{fname}_ramsey.json")));
    assert_eq!(floats(&got["order_var"]), floats(&want["order_var"]));
    close("ys", &floats(&got["steady_state"]), &floats(&want["ys"]), 1e-10);
    close("ghx", &floats(&got["ghx"]), &floats(&want["ghx"]), 1e-10);
    close("ghu", &floats(&got["ghu"]), &floats(&want["ghu"]), 1e-10);
    let po: Value = common::json_ref(out.join(format!("{fname}_planner_objective.json")));
    let w = &want["planner_objective_value"];
    close(
        "planner_objective_value",
        &[
            po["unconditional"].as_f64().unwrap(),
            po["conditional_zero_initial_multiplier"].as_f64().unwrap(),
            po["conditional_steady_initial_multiplier"].as_f64().unwrap(),
        ],
        &[
            w["unconditional"].as_f64().unwrap(),
            w["conditional"]["zero_initial_multiplier"].as_f64().unwrap(),
            w["conditional"]["steady_initial_multiplier"].as_f64().unwrap(),
        ],
        1e-10,
    );
}

#[test]
fn ramsey_with_expectation() {
    check_ramsey("nk_ramsey_expectation");
}

#[test]
fn ramsey_with_diff_and_expectation_off_the_steady_state() {
    check_ramsey("nk_ramsey_diff_expectation");
}

/// Runs `fname` and checks each sample's mode, Laplace approximation and the initial log
/// posterior (the second sample starts from the first sample's mode file) against Dynare.
/// The modes come from csminwel, whose path differs from MATLAB's at the 1e-4 level here.
fn check_samples(fname: &str) -> (PathBuf, Value) {
    let want: Value = common::json_ref(refdir().join(format!("{fname}.json")));
    let d = common::temp_dir(&format!("feat_{fname}"));
    for f in [format!("{fname}.mod"), "data_ca1_csv.csv".to_string()] {
        std::fs::copy(refdir().join(&f), d.join(&f)).unwrap();
    }
    let log = run_cli(&d, &format!("{fname}.mod"), &[]);
    let init: Vec<&str> =
        log.lines().filter_map(|l| l.strip_prefix("Initial value of the log posterior (or likelihood): ")).collect();
    let want_init: Vec<f64> = floats(&want["initial_log_posterior"]);
    assert_eq!(init.len(), want_init.len());
    // printed with 4 decimals; the last one can round the other way with OpenBLAS on Linux
    let tol = if cfg!(target_os = "macos") { 0.0 } else { 1.5e-4 };
    for (g, w) in init.iter().zip(&want_init) {
        let g = g.trim().parse::<f64>().unwrap();
        assert!((g - w).abs() <= tol, "initial log posterior: {g} vs {w}");
    }
    for k in floats(&want["keys"]) {
        let w = &want[format!("sample_{k}")];
        let r: Value = common::json_ref(d.join(format!("{fname}_{k}/Output/{fname}_results.json")));
        assert_eq!(r["parameter_names"], w["parameter_names"]);
        close(&format!("mode of sample {k}"), &floats(&r["mode"]), &floats(&w["xparam1"]), 1e-3);
        let (g, w) = (r["log_data_density_laplace"].as_f64().unwrap(), w["laplace"].as_f64().unwrap());
        assert!((g - w).abs() < 5e-3, "Laplace approximation of sample {k}: {g} vs Dynare {w}");
    }
    (d, want)
}

#[test]
fn recursive_estimation_with_forecasts() {
    let fname = "ls2003_recursive";
    let (d, want) = check_samples(fname);
    let got: Value = common::json_ref(d.join(format!("{fname}_79/Output/{fname}_recursive_forecast.json")));
    let (rf, wf) = (&got["RecursiveForecast"], &want["RecursiveForecast"]);
    let fields = wf.as_object().unwrap();
    assert_eq!(rf.as_object().unwrap().len(), fields.len());
    for (f, vars) in fields {
        for (v, w) in vars.as_object().unwrap() {
            let (g, w) = (floats(&rf[f][v]), floats(w));
            assert_eq!(g.len(), w.len(), "{f}.{v}");
            for (a, b) in g.iter().zip(&w) {
                assert!((a - b).abs() < 1e-3, "RecursiveForecast.{f}.{v}: {a} vs Dynare {b}");
            }
        }
    }
    // after an MCMC: MeanForecast plus the PointForecast bands (HPDTotalinf / HPDTotalsup)
    let d2 = common::temp_dir("feat_recursive_mcmc");
    for f in [format!("{fname}.mod"), "data_ca1_csv.csv".to_string()] {
        std::fs::copy(refdir().join(&f), d2.join(&f)).unwrap();
    }
    run_cli(&d2, &format!("{fname}.mod"), &["--mh-replic", "40"]);
    let got: Value = common::json_ref(d2.join(format!("{fname}_79/Output/{fname}_recursive_forecast.json")));
    for f in ["Mean", "HPDinf", "HPDsup", "HPDTotalinf", "HPDTotalsup"] {
        assert_eq!(floats(&got["RecursiveForecast"][f]["y_obs"]).len(), 2 * 8, "{f}");
    }
    assert!(d2.join(format!("{fname}_76/metropolis")).is_dir());
}

#[test]
fn rolling_window_estimation() {
    check_samples("ls2003_rolling");
}

/// The slice sampler's `maximize` with `maximize_using_mh_bounds` (third estimation of
/// tests/estimation/slice/fs2000_slice.mod): continues Dynare's MCMC files (slice_max/metropolis,
/// written by `-DPREP`) with one maximization per chain; the new draws are Dynare's.
#[test]
fn slice_maximize_continues_dynares_chains() {
    let src = refdir().join("slice_max");
    let want: Value = common::json_ref(src.join("fs2000_slice_max.json"));
    let d = common::temp_dir("feat_slice_max");
    for f in ["fs2000_slice_max.mod", "fs2000.inc"] {
        std::fs::copy(src.join(f), d.join(f)).unwrap();
    }
    std::fs::copy(refdir().join("../sweep_numfix/fsdat_simul.m"), d.join("fsdat_simul.m")).unwrap();
    let met = d.join("fs2000_slice_max/metropolis");
    std::fs::create_dir_all(&met).unwrap();
    for e in std::fs::read_dir(src.join("metropolis")).unwrap() {
        let p = e.unwrap().path();
        std::fs::copy(&p, met.join(p.file_name().unwrap())).unwrap();
    }
    run_cli(&d, "fs2000_slice_max.mod", &[]);
    let paths = dynare::estimation::mh_files::MhPaths::new(&d.join("fs2000_slice_max"), "fs2000_slice_max");
    let (rec, _) = dynare::estimation::mh_files::load_last_record(&paths).unwrap();
    let rows = |v: &Value| -> Vec<Vec<f64>> { v.as_array().unwrap().iter().map(floats).collect() };
    for (b, w) in rows(&want["last_parameters"]).iter().enumerate() {
        let g: Vec<f64> = rec.last_parameters.row(b).iter().copied().collect();
        close(&format!("draw of chain {}", b + 1), &g, w, 1e-4);
    }
    close("log posterior", &rec.last_log_post, &floats(&want["last_logpost"]), 1e-10);
}

#[test]
fn slice_maximize_option_checks() {
    use dynare::estimation::samplers::{SamplerOptions, check_maximize};
    use dynare::modfile::OptValue;
    let so = |kv: &[(&str, f64)]| {
        let l = kv.iter().flat_map(|(k, v)| [OptValue::Str(k.to_string()), OptValue::Num(*v)]).collect();
        SamplerOptions::from_options(Some(&OptValue::Str("slice".into())), Some(&OptValue::List(l))).unwrap()
    };
    let s = so(&[("maximize", 1.0), ("maximize_using_mh_bounds", 1.0), ("mode_compute", 4.0)]);
    assert!(s.slice.maximize && s.slice.maximize_using_mh_bounds && s.slice.mode_compute == 4);
    assert!(check_maximize(&s, true, false).is_ok());
    let e = check_maximize(&s, false, false).unwrap_err().to_string();
    assert!(e.contains("requires the load_mh_file option"), "{e}");
    assert!(check_maximize(&s, true, true).unwrap_err().to_string().contains("incompatible with the mh_recover"));
    let e = check_maximize(&so(&[("maximize_using_mh_bounds", 1.0)]), true, false).unwrap_err().to_string();
    assert!(e.contains("requires the maximize option"), "{e}");
}
