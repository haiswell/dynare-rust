//! stoch_simul completion against Dynare 7.2 (reference/stoch_simul2/, written by dump_ss2.m):
//! `partial_information` (dr1_PI / PI_gensys / PCL_Part_info_irf / PCL_Part_info_moments),
//! `aim_solver` / `dr=aim` (AIM_first_order_solver + matlab/AIM), the refusals of `loglinear`
//! / `aim_solver` / `dr=cycle_reduction` above order 1, `dr=cycle_reduction` at order 2 without
//! `k_order_solver`, the Hessian-equal-to-zero fallback, the `order = 2` default of every
//! `stoch_simul` without `order`, and `print`. The whole console output of `dynare-rs` is
//! compared with Dynare's log, character for character.

use std::path::{Path, PathBuf};

use dynare::analysis::{self, OutputContext, StochSimulOptions, partial_info};
use dynare::linalg::Mat;
use dynare::modfile::{Command, ModFile, parse_mod_file};
use serde_json::Value;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn refdir() -> PathBuf {
    root().join("reference/stoch_simul2")
}

fn load(model: &str, case: usize) -> Value {
    let s = std::fs::read_to_string(refdir().join(format!("ref_{model}_c{case}.json"))).unwrap();
    serde_json::from_str(&s).unwrap()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn vec_of(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vec_of).collect(),
        x => vec![num(x)],
    }
}

/// A MATLAB matrix as jsonencode writes it (rows of arrays, or a vector / scalar).
fn mat(v: &Value, nrows: usize, ncols: usize) -> Mat {
    match v {
        Value::Array(rows) if rows.first().is_some_and(|r| r.is_array()) => {
            Mat::from_fn(nrows, ncols, |i, j| num(&rows[i][j]))
        }
        _ => {
            let x = vec_of(v);
            Mat::from_fn(nrows, ncols, |i, j| x[i * ncols + j])
        }
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol * (1.0 + b.abs())
}

fn assert_vec(what: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(close(*x, *y, tol), "{what}[{k}]: rust {x} vs dynare {y}");
    }
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("dynare_ss2_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The file and the options in force at each stoch_simul command (they persist; every command
/// without `order` resets it to 2, as the preprocessor writes it).
fn setup(model: &str, case: usize) -> (ModFile, Vec<(StochSimulOptions, Vec<String>)>) {
    let mf = parse_mod_file(&refdir().join(format!("{model}.mod")), &[("CASE".into(), case.to_string())]).unwrap();
    let mut o = StochSimulOptions::default();
    let mut opts = Vec::new();
    for c in &mf.commands {
        if let Command::StochSimul(co, vars) = c {
            if !co.iter().any(|(k, _)| k == "order") {
                o.order = 2;
            }
            let unknown = o.apply(co);
            assert!(unknown.is_empty(), "unknown options {unknown:?}");
            opts.push((o.clone(), vars.clone()));
            o.partial_information = false;
        }
    }
    (mf, opts)
}

fn ctx(dir: &Path, fname: &str) -> OutputContext {
    OutputContext { dir: dir.to_path_buf(), fname: fname.into(), out_dir: dir.join(fname).join("Output"), seed: 1 }
}

/// IRFs (`oo_.irfs`) and the moments of the partial-information solution.
#[test]
fn partial_information_irfs_and_moments() {
    for case in [1, 2, 3, 4] {
        let r = load("pi_nk", case);
        let (mf, opts) = setup("pi_nk", case);
        let (o, vars) = &opts[0];
        assert!(o.partial_information);
        let dir = scratch(&format!("pi{case}"));
        let res = partial_info::stoch_simul_pi(&mf, o, vars, &ctx(&dir, "pi_nk"), None).unwrap().unwrap();
        let cmd = &r["cmds"][0];
        let shocks = analysis::irf::irf_shock_indices(&mf.model.exo_names, o.irf_shocks.as_deref()).unwrap();
        assert_eq!(res.irfs.len(), if o.irf > 0 { shocks.len() } else { 0 }, "case {case}: shocks");
        for s in &res.irfs {
            for (k, &v) in res.ivar.iter().enumerate() {
                let key = format!("{}_{}", mf.model.endo_names[v], mf.model.exo_names[s.shock]);
                let y: Vec<f64> = s.y.row(k).iter().copied().collect();
                assert_vec(&format!("case {case} irf {key}"), &y, &vec_of(&cmd["irfs"][&key]), 1e-9);
            }
        }
        // oo_.autocorr / oo_.contemporaneous_correlation from the JSON output
        let j: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("pi_nk/Output/pi_nk_stoch_simul.json")).unwrap())
                .unwrap();
        if let Some(ac) = cmd.get("autocorr").filter(|_| o.ar > 0 && !o.nomoments) {
            // (cells beyond `ar` are left over in oo_ from earlier runs of the dump)
            for k in 0..o.ar {
                assert_vec(
                    &format!("case {case} autocorr{{{}}}", k + 1),
                    &vec_of(&j["autocorr"][k]),
                    &vec_of(&ac[k]),
                    1e-9,
                );
            }
        }
        if let Some(cc) = cmd.get("contemporaneous_correlation").filter(|_| o.contemporaneous_correlation && !o.nocorr)
        {
            assert_vec(
                &format!("case {case} contemporaneous_correlation"),
                &vec_of(&j["contemporaneous_correlation"]),
                &vec_of(cc),
                1e-9,
            );
        }
        // Output/<fname>_PCL_moments.mat (COV_YR0) and the IRF .mat files
        if let Some(p) = r.get("PCL_moments") {
            let m = dynare::matfile::read(&dir.join("pi_nk/Output/pi_nk_PCL_moments.mat")).unwrap();
            let n = mf.model.n_endo();
            let cov = m["COV_YR0"].as_matrix().unwrap();
            let want = mat(&p["COV_YR0"], n, n);
            for i in 0..n {
                for jj in 0..n {
                    assert!(
                        close(cov[(i, jj)], want[(i, jj)], 1e-9),
                        "case {case} COV_YR0[{i},{jj}]: {} vs {}",
                        cov[(i, jj)],
                        want[(i, jj)]
                    );
                }
            }
        }
        if o.irf > 0 {
            assert!(
                dir.join(format!("pi_nk_PCL_PtInfoIRFs_{}_{}.mat", shocks[0] + 1, mf.model.exo_names[shocks[0]]))
                    .exists()
            );
        }
    }
}

/// `dr=aim` / `aim_solver`: AIM's decision rule is the QZ one, on a linear and a nonlinear model.
#[test]
fn aim_solver_decision_rules() {
    for (model, case) in [("pi_nk", 5), ("rbc_ll", 4)] {
        let r = load(model, case);
        let (mf, opts) = setup(model, case);
        for (k, (o, _)) in opts.iter().enumerate() {
            assert!(o.aim_solver);
            let mut o = o.clone();
            o.order = 1;
            let sol = analysis::solve_q2(&mf, &o, None).unwrap().unwrap();
            let cmd = &r["cmds"][k];
            let n = mf.model.n_endo();
            let ov: Vec<usize> = vec_of(&cmd["order_var"]).iter().map(|&v| v as usize - 1).collect();
            assert_eq!(ov, sol.dr.order_var, "{model}: order_var");
            let ghx = mat(&cmd["ghx"], n, sol.dr.nspred);
            let ghu = mat(&cmd["ghu"], n, mf.model.n_exo());
            for i in 0..n {
                for j in 0..sol.dr.nspred {
                    assert!(close(sol.dr.ghx[(i, j)], ghx[(i, j)], 1e-10), "{model} ghx[{i},{j}]");
                }
                for j in 0..mf.model.n_exo() {
                    assert!(close(sol.dr.ghu[(i, j)], ghu[(i, j)], 1e-10), "{model} ghu[{i},{j}]");
                }
            }
            let ev = vec_of(&cmd["eigval_re"]);
            let got: Vec<f64> = sol.dr.eigval.iter().map(|z| z.re).collect();
            assert_vec(&format!("{model} eigval"), &got, &ev, 1e-10);
        }
    }
}

/// The console output of `dynare-rs` on every case, against Dynare's log (from the end of the
/// preprocessing on; Dynare's errors end the log, dynare-rs reports them on stderr).
#[test]
fn printout_matches_dynare_logs() {
    let bin = env!("CARGO_BIN_EXE_dynare-rs");
    let cases: &[(&str, &[usize])] =
        &[("pi_nk", &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), ("rbc_ll", &[1, 2, 3, 4, 5, 6, 7, 8, 9])];
    for &(model, list) in cases {
        for &case in list {
            let dir = scratch(&format!("bin_{model}_{case}"));
            std::fs::copy(refdir().join(format!("{model}.mod")), dir.join(format!("{model}.mod"))).unwrap();
            let out = std::process::Command::new(bin)
                .arg(format!("{model}.mod"))
                .arg(format!("-DCASE={case}"))
                .current_dir(&dir)
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let lines: Vec<&str> = stdout.lines().collect();
            let i = lines.iter().position(|l| l.starts_with("Preprocessing:")).unwrap();
            let rust: Vec<&str> = lines[i + 1..]
                .iter()
                .copied()
                .filter(|l| {
                    !l.starts_with("Total computing time") && !(l.starts_with("Note: ") && l.contains("warning(s)"))
                })
                .collect();
            let log = std::fs::read_to_string(refdir().join(format!("{model}_c{case}_matlab.log"))).unwrap();
            let ml: Vec<&str> = log.lines().collect();
            let j = ml.iter().position(|l| l.starts_with("Preprocessing time")).unwrap();
            let matlab: Vec<&str> = ml[j + 1..]
                .iter()
                .copied()
                .filter(|l| {
                    !l.starts_with("Total computing time") && !(l.starts_with("Note: ") && l.contains("warning(s)"))
                })
                // MATLAB's warning() (bracketed, with a stack trace in -batch mode)
                .filter(|l| {
                    let t = l.trim_start_matches(['[', '\u{8}']);
                    !(t.starts_with("Warning: ") || t.starts_with("> In ") || l.starts_with("In "))
                })
                .collect();
            let rust: Vec<&str> = rust.into_iter().filter(|l| !l.starts_with("Warning: ")).collect();
            // simulated moments use dynare-rust's random numbers: compare up to them
            let cut = |v: &[&str]| -> String {
                let k = v.iter().position(|l| l.starts_with("MOMENTS OF SIMULATED VARIABLES")).unwrap_or(v.len());
                v[..k].join("\n")
            };
            let (rust, matlab) = (cut(&rust), cut(&matlab));
            assert_eq!(rust, matlab, "{model} case {case}: printout differs\nstderr: {stderr}");
            let err = load(model, case)["error"].as_str().unwrap_or("").to_string();
            if err.is_empty() {
                assert!(out.status.success(), "{model} case {case}: {stderr}");
            } else {
                assert!(!out.status.success(), "{model} case {case}: Dynare stops with {err}");
                assert!(
                    stderr.contains(&format!("ERROR: {err}")),
                    "{model} case {case}: expected {err:?}, got {stderr}"
                );
            }
            // tex + graph_format=eps: the eps-loader file (but its date line)
            let want = refdir().join(format!("{model}_c{case}_IRF.tex"));
            if want.exists() {
                let got = std::fs::read_to_string(dir.join(format!("{model}/graphs/{model}_IRF.tex"))).unwrap();
                let want = std::fs::read_to_string(want).unwrap();
                let skip_date = |s: &str| {
                    s.lines().enumerate().filter(|(k, _)| *k != 1).map(|(_, l)| l.to_string()).collect::<Vec<_>>()
                };
                assert_eq!(skip_date(&got), skip_date(&want), "{model} case {case}: _IRF.tex");
            }
        }
    }
}

/// `simul_replic`: every replication in `Output/<fname>_simul` (float64, n_endo x periods each).
#[test]
fn simul_replic_file() {
    let (mf, opts) = setup("pi_nk", 1);
    let mut o = opts[1].0.clone();
    o.periods = 120;
    o.drop = 20;
    o.simul_replic = 3;
    o.irf = 0;
    o.noprint = true;
    o.nograph = true;
    let dir = scratch("replic");
    let c = ctx(&dir, "pi_nk");
    let sol = analysis::solve_q2(&mf, &o, None).unwrap().unwrap();
    let res = analysis::stoch_simul_q2(&mf.model, &sol, &o, &[], &c, None, None).unwrap();
    let bytes = std::fs::read(dir.join("pi_nk/Output/pi_nk_simul")).unwrap();
    let n = mf.model.n_endo();
    assert_eq!(bytes.len(), 3 * n * 120 * 8);
    let first: Vec<f64> = bytes[..n * 120 * 8].chunks(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    let es = res.endo_simul.unwrap();
    assert_eq!(first, es.iter().copied().collect::<Vec<_>>());
}

/// `histval` sets the starting point of the simulation (`M_.endo_histval`).
#[test]
fn histval_starting_point() {
    let (mf, opts) = setup("pi_nk", 1);
    let mut o = opts[1].0.clone();
    o.periods = 150;
    o.irf = 0;
    o.noprint = true;
    o.nomoments = true;
    let r = mf.model.endo_index("r").unwrap();
    o.endo_histval = analysis::stochastic_histval_of(&mf.model, &[(true, r, 0, 2.0)]);
    let dir = scratch("histval");
    let sol = analysis::solve_q2(&mf, &o, None).unwrap().unwrap();
    let ex = Mat::zeros(150, mf.model.n_exo());
    let res = analysis::stoch_simul_q2(&mf.model, &sol, &o, &[], &ctx(&dir, "pi_nk"), Some(&ex), None).unwrap();
    let y = res.endo_simul.unwrap();
    // period 1 = ghx * (y0 - ys): r(0) = 2, every other state at 0
    let dr = &sol.dr;
    let k = dr.state_var.iter().position(|&v| v == r).unwrap();
    for v in 0..mf.model.n_endo() {
        let want = dr.ghx[(dr.inv_order_var[v], k)] * 2.0;
        assert!((y[(v, 0)] - want).abs() < 1e-12, "{v}: {} vs {want}", y[(v, 0)]);
    }
}

/// Measurement errors (`M_.H`): `oo_.variance_decomposition_ME` and
/// `oo_.conditional_variance_decomposition_ME` at orders 1 and 2.
#[test]
fn measurement_error_decompositions() {
    for (model, case, order) in [("pi_nk", 10, 1), ("rbc_ll", 9, 2)] {
        let r = load(model, case);
        let (mf, opts) = setup(model, case);
        let (o, vars) = &opts[0];
        let mut o = o.clone();
        o.me = analysis::MeasurementErrors::from_modfile(&mf);
        assert!(o.me.is_some());
        o.noprint = true;
        let sol = analysis::solve_q2(&mf, &StochSimulOptions { order: 1, ..o.clone() }, None).unwrap().unwrap();
        let ivar = analysis::var_indices(&mf.model, vars).unwrap();
        let rep = analysis::theoretical_moments_q2(&mf.model, &sol, &ivar, &o, order, None, None).unwrap();
        let cmd = &r["cmds"][0];
        let vme = rep.variance_decomposition_me.unwrap();
        let want = mat(&cmd["variance_decomposition_ME"], vme.nrows(), vme.ncols());
        for i in 0..vme.nrows() {
            for j in 0..vme.ncols() {
                assert!(
                    close(vme[(i, j)], want[(i, j)], 1e-9),
                    "{model} variance_decomposition_ME[{i},{j}]: {} vs {}",
                    vme[(i, j)],
                    want[(i, j)]
                );
            }
        }
        // MATLAB's array is nobs x nsteps x (exo_nbr+1); jsonencode nests it [obs][step][shock]

        let cme = rep.conditional_vardec_me.unwrap();
        let c = &cmd["conditional_variance_decomposition_ME"];
        for (h, m) in cme.iter().enumerate() {
            for i in 0..m.nrows() {
                for j in 0..m.ncols() {
                    let w = num(&c[i][h][j]);
                    assert!(close(m[(i, j)], w, 1e-9), "{model} cvd_ME[{i},{h},{j}]: {} vs {w}", m[(i, j)]);
                }
            }
        }
    }
}
