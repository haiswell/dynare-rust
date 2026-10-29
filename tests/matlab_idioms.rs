//! The MATLAB idioms of Dynare 7.2's own tests that dynare-rust interprets
//! (src/modfile/parser_matlab.rs, `options_.initial_date`):
//! * `if isoctave ... else ... end`: `isoctave` is false, the other branches are skipped (an
//!   `if isoctave` block without ';' in pac/trend-component-21 used to swallow a line);
//! * `options_.initial_date = dates('1989Q4')` dates plot_shock_decomposition
//!   (shock_decomposition/ls2003_plot.mod; reference/matlab_idioms/ref.json from MATLAB, see
//!   below);
//! * `load 'X.mat'` + `param_names = fieldnames(steady_state.params)` + a `for` loop calling
//!   `set_param_value(param, steady_state.params.(param))`: the heterogeneity tests
//!   (reference/matlab_idioms/het/, the .mod and MAT files of Dynare's tests/heterogeneity,
//!   `ref_<name>.json` written by `MDL=<name> matlab -batch "run('dump_tests.m')"`).
mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use dynare::driver::{RunOptions, run};
use dynare::heterogeneity::commands::{HetSession, handles, run_command};
use dynare::matfile::{MatValue, write_values};
use dynare::modfile::{Command, MATLAB_STATEMENT, ModFile, OptValue, SET_PARAM, parse_mod_file};
use serde_json::Value;

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/matlab_idioms")
}

fn param(mf: &ModFile, name: &str) -> f64 {
    let k = mf.model.param_names.iter().position(|p| p == name).unwrap();
    mf.model.params[k]
}

/// The skipped MATLAB statements (kept for the run-time warning).
fn skipped(mf: &ModFile) -> Vec<String> {
    mf.commands
        .iter()
        .filter_map(|c| match c {
            Command::Other(n, o) if n == MATLAB_STATEMENT => Some(o[0].0.clone()),
            _ => None,
        })
        .collect()
}

const MODEL: &str = "var y; varexo e; parameters a b c d;\na = 0.5; b = 0; c = 0; d = 0;\n\
                     model; y = a*y(-1) + e; end;\nshocks; var e; stderr 1; end;\n";

#[test]
fn if_isoctave_selects_the_matlab_branch() {
    let dir = common::temp_dir("ml_isoctave");
    let text = format!(
        "{MODEL}\
if ~isoctave % comment
  a = 0.7;
end
if isoctave
  // Octave only, no ';' after `if`
  options_.bnlms.set_dynare_seed_to_default = false;
  set_dynare_seed(4);
  a = 0.9;
end
b = 2;
if isoctave()
  c = 1;
elseif isoctave && options_.solve_algo==0
  c = 2;
else
  if ~isoctave
    c = 3;
  end
end
if (isoctave && user_has_octave_package('optim')) || ~(isoctave)
  d = 4;
else
  d = 5;
end
if ~isoctave && matlab.internal.parallel.isPCTInstalled
  d = d + 1;
end
"
    );
    std::fs::write(dir.join("t.mod"), text).unwrap();
    let mf = parse_mod_file(&dir.join("t.mod"), &[]).unwrap();
    assert_eq!(param(&mf, "a"), 0.7);
    assert_eq!(param(&mf, "b"), 2.0);
    assert_eq!(param(&mf, "c"), 3.0);
    assert_eq!(param(&mf, "d"), 4.0);
    // `d = d + 1` follows a skipped statement (a command): set where it stands
    assert!(mf.commands.iter().any(|c| matches!(c, Command::Other(n, o) if n == SET_PARAM
        && matches!(o.as_slice(), [(p, OptValue::Num(v))] if p == "d" && *v == 5.0))));
    // the condition isoctave does not decide: the `if` line and its `end` are skipped with a
    // warning, the body runs (as before)
    let s = skipped(&mf);
    assert_eq!(s.len(), 2, "{s:?}");
    assert!(s[0].starts_with("if ~ isoctave && matlab"), "{s:?}");
    assert_eq!(s[1], "end");
    assert!(!mf.commands.iter().any(|c| matches!(c, Command::Other(n, _) if n == "set_dynare_seed")));
}

#[test]
fn load_and_set_param_value_loop() {
    let dir = common::temp_dir("ml_load_loop");
    let params = MatValue::structure(vec![("a", MatValue::scalar(0.25)), ("c", MatValue::scalar(-3.0))]);
    let ss = MatValue::structure(vec![("params", params), ("agg", MatValue::structure(vec![]))]);
    write_values(&dir.join("p.mat"), &[("steady_state", &ss)]).unwrap();
    let text = format!(
        "{MODEL}\
load 'p.mat';

param_names = fieldnames(steady_state.params);
for i=1:numel(param_names)
    param = param_names{{i}};
    set_param_value(param, steady_state.params.(param));
end
stoch_simul(order=1, irf=0, noprint);
load('p.mat')
for i=1:length(param_names)
   p = param_names{{i}};
   set_param_value(p, steady_state.params.(p));
end
"
    );
    std::fs::write(dir.join("t.mod"), text).unwrap();
    let mf = parse_mod_file(&dir.join("t.mod"), &[]).unwrap();
    assert_eq!(param(&mf, "a"), 0.25);
    assert_eq!(param(&mf, "c"), -3.0);
    assert!(skipped(&mf).is_empty(), "{:?}", skipped(&mf));
    // a parameter missing from the model: set_param_value's error
    let bad = MatValue::structure(vec![("params", MatValue::structure(vec![("zz", MatValue::scalar(1.0))]))]);
    write_values(&dir.join("p.mat"), &[("steady_state", &bad)]).unwrap();
    let e = parse_mod_file(&dir.join("t.mod"), &[]).unwrap_err().to_string();
    assert!(e.contains("Parameter name zz doesn't exist"), "{e}");
}

/// A long csv (`period,date,variable,component,value`) as (variable, component) -> values, the
/// components in file order and the first period label.
fn read_long(p: &Path) -> (HashMap<(String, String), Vec<f64>>, Vec<String>, String) {
    let text = common::read_ref(p);
    let mut lines = text.lines();
    assert!(lines.next().unwrap().starts_with("period,"));
    let (mut m, mut comps, mut first) = (HashMap::<(String, String), Vec<f64>>::new(), Vec::new(), String::new());
    for l in lines {
        let f: Vec<&str> = l.split(',').collect();
        if first.is_empty() {
            first = f[0].to_string();
        }
        if !comps.contains(&f[3].to_string()) {
            comps.push(f[3].to_string());
        }
        m.entry((f[2].into(), f[3].into())).or_default().push(f[4].parse().unwrap_or(f64::NAN));
    }
    (m, comps, first)
}

/// Compares a plot csv with MATLAB's get_decomp_only `[var][component][period]` array (before
/// the plot_init_date window: matched on its last periods).
fn check(p: &Path, want: &Value, vars: &[&str]) -> (usize, String) {
    let (got, comps, first) = read_long(p);
    let mut n = 0;
    for (vi, v) in vars.iter().enumerate() {
        let wv = want[vi].as_array().unwrap();
        assert_eq!(wv.len(), comps.len(), "{}: components", p.display());
        for (ci, c) in comps.iter().enumerate() {
            let w: Vec<f64> = wv[ci].as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
            let h = &got[&(v.to_string(), c.clone())];
            assert!(h.len() <= w.len(), "{} {v} {c}: {} vs {} periods", p.display(), h.len(), w.len());
            for (t, (a, b)) in h.iter().zip(&w[w.len() - h.len()..]).enumerate() {
                let ok = (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * (1.0 + b.abs());
                assert!(ok, "{} {v} {c} t={t}: rust {a} vs dynare {b}", p.display());
                n += 1;
            }
        }
    }
    (n, first)
}

#[test]
fn options_initial_date_dates_the_decomposition_plots() {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    let dir = common::temp_dir("ml_initial_date");
    std::fs::copy(refdir().join("../sweep_fix2/data_ca1.m"), dir.join("data_ca1.m")).unwrap();
    std::fs::copy(refdir().join("ls2003_initial_date.mod"), dir.join("ls2003_initial_date.mod")).unwrap();
    run(&dir.join("ls2003_initial_date.mod"), &RunOptions { quiet: true, ..Default::default() }).unwrap();
    let out = dir.join("ls2003_initial_date/Output");
    let f = |s: &str| out.join(format!("ls2003_initial_date_{s}.csv"));
    let zz = &r["zz"];
    // aoa: the first full year after 1989Q4 is 1990, the plot starts after its Q4
    let (n1, first) = check(&f("shock_decomposition_aoa_P1_calibration"), &zz[0], &["y_obs", "R_obs"]);
    assert_eq!(first, "1991Y");
    // the plot_init_date window starts at 2004Q1 (Dynare 7.2 warns "nothing to plot" with the
    // default 1Q1 dates)
    let (n2, first) = check(&f("shock_decomposition_qoq_P2_calibration"), &zz[1], &["y_obs", "R_obs"]);
    assert_eq!(first, "2004Q1");
    // vintage 29 is 1989Q4 + 28
    let (n3, _) = check(&f("shock_decomposition_P3_8step_ahead_forecast_1996Q4_calibration"), &zz[2], &["y_obs"]);
    assert!(n1 > 50 && n2 > 50 && n3 > 10, "{n1} {n2} {n3}");
}

/// Runs the .mod file's heterogeneity commands in order (as the driver does).
fn run_het(dir: &Path, name: &str) -> (ModFile, HetSession) {
    let mut mf = parse_mod_file(&dir.join(format!("{name}.mod")), &[]).unwrap();
    let mut st = HetSession::default();
    for c in &mf.commands.clone() {
        if let Command::Other(n, o) = c
            && handles(n)
        {
            run_command(&mut mf, n, o, dir, name, &mut st).unwrap_or_else(|e| panic!("{name}: {n}: {e:#}"));
        }
    }
    (mf, st)
}

fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        Value::Null => vec![f64::NAN],
        x => vec![x.as_f64().unwrap()],
    }
}

fn assert_close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        let ok = (g.is_nan() && w.is_nan()) || (g - w).abs() <= tol * (1.0 + w.abs());
        assert!(ok, "{what}[{k}]: rust {g} vs dynare {w}");
    }
}

/// Parameters (from the MAT file, then calibrated), steady state and Jacobians against MATLAB.
fn check_het(dir: &Path, name: &str, tol: f64) {
    let r: Value = common::json_ref(dir.join(format!("ref_{name}.json")));
    let (mf, st) = run_het(dir, name);
    assert_close(&format!("{name} params"), &mf.model.params, &flat(&r["params"]), tol);
    let ss = st.steady_state.as_ref().unwrap();
    let ragg = r["het"]["agg"].as_object().unwrap();
    assert_eq!(ss.steady_state.agg.len(), ragg.len(), "{name}: agg size");
    for (n, v) in &ss.steady_state.agg {
        assert_close(&format!("{name} agg.{n}"), &[*v], &[ragg[n].as_f64().unwrap()], tol);
    }
    for (n, a) in &ss.steady_state.pol_values {
        let rp = &r["het"]["pol"][n];
        assert_close(&format!("{name} pol.{n} sum"), &[a.data.iter().sum()], &[rp["sum"].as_f64().unwrap()], tol);
        let first = flat(&rp["first"]);
        assert_close(&format!("{name} pol.{n} first"), &a.data[..first.len()], &first, tol);
    }
    let dr = st.dr.as_ref().unwrap();
    assert_eq!(dr.truncation_horizon as f64, r["het"]["T"].as_f64().unwrap());
    let mut n = 0;
    for (v, by_shock) in r["het"]["G"].as_object().unwrap() {
        for (s, g0) in by_shock.as_object().unwrap() {
            let want = flat(g0);
            let g = dr.g_block(v, s).unwrap_or_else(|| panic!("{name}: G.{v}.{s}"));
            let got: Vec<f64> = (0..want.len()).map(|i| g[(i, 0)]).collect();
            assert_close(&format!("{name} G.{v}.{s}"), &got, &want, tol);
            n += 1;
        }
    }
    assert!(n > 0);
}

#[test]
fn heterogeneity_parameters_from_a_mat_file() {
    // a copy: the commands write their figures next to the .mod file
    let dir = common::temp_dir("ml_het");
    for e in std::fs::read_dir(refdir().join("het")).unwrap() {
        let p = e.unwrap().path();
        std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
    }
    for name in ["ks_steady_state", "hank_1a_steady_state", "ks_two_shocks"] {
        check_het(&dir, name, 1e-6);
    }
    // the calibration loop stops within its tolerance: policies agree to ~2e-6, the Jacobians
    // to ~6e-5
    check_het(&dir, "hank_2a_steady_state", 1e-4);
}
