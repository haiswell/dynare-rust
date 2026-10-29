//! End-to-end checks of `dynare-rs` on Dynare 7.2's official stochastic-simulation examples
//! (`examples/stochastic_simulations/`, `examples/macroprocessor/bkk_1992.mod`,
//! `examples/reporting/collard_2001_reporting.mod`), against
//! `reference/examples72/<name>/ref.json` written by `reference/examples72/dump_stoch.m`
//! (`oo_.dr.ys`, `order_var`, `ghx ghu` (+ `ghxx ghxu ghuu ghs2` at order 2), `oo_.mean/var/
//! autocorr/variance_decomposition`, order-1 `oo_.irfs`, and `oo_.exo_simul` for the simulated
//! moments). Each test runs the binary on the example in a scratch directory and reads
//! `<fname>/Output/<fname>_stoch_simul.json`.
//!
//! Two examples need MATLAB code that dynare-rust cannot run: `nk_baseline` (an external
//! `nk_baseline_steadystate.m` with `fzero`) and `collard_2001_analytical_steady_state` (a
//! `steady_state_model` helper calling `fsolve`). Both are checked to fail with an explicit
//! message, and their dynamics are checked on variants (`*_noss.mod`, `*_noh.mod`) that give the
//! parameters / steady-state value Dynare computes with that code.

mod common;

use std::path::{Path, PathBuf};

use dynare::linalg::Mat;
use dynare::modfile::{Command, parse_mod_file};
use dynare::solver::korder::{simulate::simult, stoch_simul as ks};
use serde_json::Value;

fn refdir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/examples72").join(name)
}

fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        Value::Null => vec![f64::NAN],
        x => vec![x.as_f64().unwrap()],
    }
}

/// A MATLAB matrix as jsonencode writes it (array of rows, or a vector/scalar).
fn mat(v: &Value) -> Mat {
    match v {
        Value::Array(rows) if rows.first().is_some_and(|r| r.is_array()) => {
            let nr = rows.len();
            let nc = rows[0].as_array().unwrap().len();
            Mat::from_fn(nr, nc, |i, j| rows[i][j].as_f64().unwrap_or(f64::NAN))
        }
        _ => {
            let x = flat(v);
            Mat::from_fn(x.len(), 1, |i, _| x[i])
        }
    }
}

fn assert_close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length {} vs {}", got.len(), want.len());
    for (k, (a, b)) in got.iter().zip(want).enumerate() {
        let ok = (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol * (1.0 + b.abs());
        assert!(ok, "{what}[{k}]: dynare-rs {a} vs Dynare {b}");
    }
}

/// Runs `dynare-rs <file>` on a copy of the example directory; returns (scratch dir, stdout, ok).
fn run(name: &str, file: &str) -> (PathBuf, String, bool) {
    let dir = common::temp_dir(&format!("ex72_{}", file.trim_end_matches(".mod")));
    for e in std::fs::read_dir(refdir(name)).unwrap() {
        let p = e.unwrap().path();
        if p.is_file() && p.file_name().unwrap() != "ref.json" {
            std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
        }
    }
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_dynare-rs")).arg(file).current_dir(&dir).output().unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (dir, text, out.status.success())
}

fn results(dir: &Path, fname: &str) -> Value {
    let p = dir.join(fname).join("Output").join(format!("{fname}_stoch_simul.json"));
    common::json_ref(p)
}

/// Compares decision rules and (theoretical) moments; `moments = false` for simulated moments.
fn check(name: &str, file: &str, moments: bool) -> (Value, Value) {
    let r: Value = common::json_ref(refdir(name).join("ref.json"));
    let (dir, text, ok) = run(name, file);
    assert!(ok, "{file} failed:\n{text}");
    let fname = file.trim_end_matches(".mod");
    let j = results(&dir, fname);
    let dr = &j["dr"];
    let tol = 1e-8;
    assert_close("ys", &flat(&dr["ys"]), &flat(&r["ys"]), tol);
    assert_eq!(flat(&dr["order_var"]), flat(&r["order_var"]), "order_var");
    for k in ["ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghs2"] {
        if !r[k].is_null() {
            assert!(!dr[k].is_null(), "{k} missing");
            assert_close(k, &flat(&dr[k]), &flat(&r[k]), tol);
        }
    }
    if moments {
        for k in ["mean", "var", "variance_decomposition"] {
            if !r[k].is_null() {
                assert!(!j[k].is_null(), "{k} missing from the results file");
                assert_close(k, &flat(&j[k]), &flat(&r[k]), tol);
            }
        }
        assert!(!j["var"].is_null(), "var missing");
        let (ra, ja) = (r["autocorr"].as_array().unwrap(), j["autocorr"].as_array().unwrap());
        assert_eq!(ra.len(), ja.len(), "autocorr: number of lags");
        for (k, (a, b)) in ja.iter().zip(ra).enumerate() {
            assert_close(&format!("autocorr{{{}}}", k + 1), &flat(a), &flat(b), tol);
        }
    }
    // order 1: oo_.irfs (order 2: generalized IRFs, random draws -- names only)
    let ri = r["irfs"].as_object();
    let ji = j["irfs"].as_object().expect("irfs");
    if let Some(ri) = ri {
        assert_eq!(ri.len(), ji.len(), "number of IRFs");
        for (k, v) in ri {
            assert_close(&format!("irfs.{k}"), &flat(&ji[k]), &flat(v), tol);
        }
    }
    (r, j)
}

#[test]
fn collard_2001_theoretical_moments() {
    // stoch_simul; (order 2 by default): order-2 rules, second-order mean, first-order moments
    check("collard_2001_theoretical_moments", "collard_2001_theoretical_moments.mod", true);
}

#[test]
fn schorfheide_2000_nonstationary() {
    check("schorfheide_2000_nonstationary", "schorfheide_2000_nonstationary.mod", true);
}

#[test]
fn aguiar_gopinath_2007_trend() {
    check("aguiar_gopinath_2007_trend", "aguiar_gopinath_2007_trend.mod", true);
}

#[test]
fn bkk_1992_macroprocessor_hp_filter() {
    // @#for loops over countries and lags; stoch_simul(order=1, hp_filter=1600)
    check("bkk_1992", "bkk_1992.mod", true);
}

#[test]
fn collard_2001_analytical_steady_state() {
    // the fsolve helper of steady_state_model is MATLAB code: explicit error
    let (_, text, ok) = run("collard_2001_analytical_steady_state", "collard_2001_analytical_steady_state.mod");
    assert!(
        !ok && text.contains("collard_2001_analytical_steady_state_helper") && text.contains("MATLAB function"),
        "{text}"
    );
    // the rest of the steady_state_model block, with h at Dynare's value
    check("collard_2001_analytical_steady_state", "collard_2001_analytical_steady_state_noh.mod", true);
    // write_latex_dynamic_model / write_latex_static_model
    let (dir, _, _) = run("collard_2001_analytical_steady_state", "collard_2001_analytical_steady_state_noh.mod");
    for f in ["dynamic.tex", "static.tex"] {
        assert!(dir.join("collard_2001_analytical_steady_state_noh/latex").join(f).is_file(), "{f}");
    }
}

#[test]
fn nk_baseline() {
    // the external MATLAB steady-state file cannot run: explicit error
    let (_, text, ok) = run("nk_baseline", "nk_baseline.mod");
    assert!(!ok && text.contains("nk_baseline_steadystate.m"), "{text}");
    // the model with the parameters and steady state of that file (order 1, irf=20)
    check("nk_baseline", "nk_baseline_noss.mod", true);
}

/// `stoch_simul(periods=2000, drop=200)` at order 2: the rules from the binary; the simulated
/// moments on Dynare's own shocks (`oo_.exo_simul`) through `simult_` and `disp_moments`.
#[test]
fn collard_2001_simulated_moments() {
    let name = "collard_2001_simulated_moments";
    let (r, j) = check(name, &format!("{name}.mod"), false);
    // the binary's own draws: moments of the right size
    assert_eq!(flat(&j["mean"]).len(), 6);
    // Dynare's draws
    let mf = parse_mod_file(&refdir(name).join(format!("{name}.mod")), &[]).unwrap();
    let opts = mf
        .commands
        .iter()
        .find_map(|c| match c {
            Command::StochSimul(o, _) => Some(o.clone()),
            _ => None,
        })
        .unwrap();
    let mut so = ks::StochSimulOptions::from_options(&opts);
    let (periods, drop) = (so.periods, so.drop);
    assert_eq!((periods, drop), (2000, 200));
    so.periods = 0;
    so.irf = 0;
    so.nomoments = true;
    so.noprint = true;
    so.nograph = true;
    let out = ks::stoch_simul(&mf, &so, &[], 1).unwrap();
    let dr = &out.dr;
    let ex = mat(&r["exo_simul"]);
    assert_eq!((ex.nrows(), ex.ncols()), (periods, 2));
    let ys = dr.first.ys.clone();
    let y = simult(dr, &ys, &ex, false).columns(1, periods).into_owned();
    let ivar: Vec<usize> = (0..6).collect();
    let filter = dynare::analysis::q2::Filter::None;
    let sm =
        dynare::analysis::simulate::simulated_moments_q2(&mf.sigma_e, &y, &ex, &ivar, drop, 5, &filter, false, &|e| {
            simult(dr, &ys, e, false)
        })
        .unwrap();
    assert_close("mean", &sm.mean, &flat(&r["mean"]), 1e-8);
    assert_close("skewness", &sm.skewness, &flat(&r["skewness"]), 1e-8);
    assert_close("kurtosis", &sm.kurtosis, &flat(&r["kurtosis"]), 1e-8);
    assert_eq!(flat(&j["skewness"]).len(), 6, "skewness missing from the results file");
    assert_eq!(flat(&j["kurtosis"]).len(), 6, "kurtosis missing from the results file");
    let var: Vec<f64> = (0..6).flat_map(|i| (0..6).map(move |k| (i, k))).map(|(i, k)| sm.var[(i, k)]).collect();
    assert_close("var", &var, &flat(&r["var"]), 1e-8);
    for (l, a) in sm.autocorr.iter().enumerate() {
        let got: Vec<f64> = (0..6).flat_map(|i| (0..6).map(move |k| (i, k))).map(|(i, k)| a[(i, k)]).collect();
        assert_close(&format!("autocorr{{{}}}", l + 1), &got, &flat(&r["autocorr"][l]), 1e-8);
    }
    let vd = sm.vardec.as_ref().unwrap();
    let got: Vec<f64> = (0..6).flat_map(|i| (0..2).map(move |k| (i, k))).map(|(i, k)| vd[(i, k)]).collect();
    assert_close("variance_decomposition", &got, &flat(&r["variance_decomposition"]), 1e-8);
}

/// `examples/reporting/collard_2001_reporting.mod`: `send_irfs_to_workspace`, dseries of the
/// IRFs (`[shocke dseries(y_e, 2014q3, 'y')]`), a two-page `report()` with graphs and tables,
/// `r.write()`, `r.compile()`. The report builds (the same files as Dynare's reporting classes);
/// the table rows are the IRFs at the table dates. Compiling needs pdflatex with pgfplots (a
/// failed compilation is a warning).
#[test]
fn collard_2001_reporting() {
    let name = "collard_2001_reporting";
    let (dir, text, ok) = run(name, &format!("{name}.mod"));
    assert!(ok, "{text}");
    assert!(!text.contains("not supported"), "{text}");
    assert!(text.contains("Adding Page: 2") && text.contains("Finished Writing Report!"), "{text}");
    let main = std::fs::read_to_string(dir.join("report.tex")).unwrap();
    assert!(main.contains("\\input{tmpRepDir/report_document.tex}"));
    let rep = dir.join("tmpRepDir");
    let mut files: Vec<String> =
        std::fs::read_dir(&rep).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    files.sort();
    assert_eq!(files.len(), 18, "{files:?}");
    for pg in 1..=2 {
        assert!(files.contains(&format!("report_pg{pg}.tex")));
        assert!(files.contains(&format!("table_pg{pg}_sec2_row1_col1.tex")));
        for (row, col) in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)] {
            assert!(files.contains(&format!("graph_pg{pg}_sec1_row{row}_col{col}.tex")), "pg{pg} row{row} col{col}");
        }
    }
    // table of page 1 (shock e): rows y c k a h b over 2022q1:2024q1 = IRF periods 31..39
    let j = results(&dir, name);
    let table = std::fs::read_to_string(rep.join("table_pg1_sec2_row1_col1.tex")).unwrap();
    assert!(
        table.contains("\\multicolumn{4}{c}{2022} & \\multicolumn{4}{c}{2023} & \\multicolumn{1}{c}{2024}"),
        "{table}"
    );
    for v in ["y", "c", "k", "a", "h", "b"] {
        let line = table
            .lines()
            .find(|l| l.starts_with(&format!("\\rowcolor{{white}}\\noindent {v}&")))
            .unwrap_or_else(|| panic!("row {v}"));
        let cells: Vec<&str> = line.trim_end_matches("\\\\%").split('&').skip(1).collect();
        let irf = flat(&j["irfs"][format!("{v}_e")]);
        let want: Vec<String> = irf[30..39].iter().map(|x| format!("{x:.5}")).collect();
        assert_eq!(cells, want, "{v}");
    }
    // graph of y: 40 points from 2014q3
    let g = std::fs::read_to_string(rep.join("graph_pg1_sec1_row1_col1.tex")).unwrap();
    assert!(
        g.contains("xticklabels={2014q3,2024q2,}")
            && g.contains("title=y,")
            && g.contains("\\addplot[red,line width=.5,forget plot] coordinates {(1,0)(40,0)};"),
        "{g}"
    );
    if text.contains("Done compiling") {
        assert!(dir.join("report.pdf").is_file());
    }
}
