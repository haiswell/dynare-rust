//! : perfect-foresight solver variants and mixed complementarity problems against
//! Dynare 7.1 (reference/q1/ref_q1pf.json, written by reference/q1/dump_q1pf.m):
//! stack_solve_algo 1, 2 (first-iteration LU / block-diagonal LU preconditioners), 3 (incomplete
//! LU), 6, 7 (dynare_solve on the stacked system), robust_lin_solve, `lmmcp` (perfect foresight
//! and steady state with solve_algo = 10), purely backward / forward / static models.

use std::path::{Path, PathBuf};

use dynare::deterministic::DetState;
use dynare::modfile::{Command, ModFile, OptValue, parse_mod_file};
use serde_json::Value;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/q1")
}

fn reference() -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir().join("ref_q1pf.json")).unwrap()).unwrap()
}

fn run(mf: &ModFile) -> DetState {
    let mut det = DetState::new(mf).unwrap();
    for c in &mf.commands {
        match c {
            Command::Steady(o) => {
                let mut o = o.clone();
                o.push(("noprint".into(), OptValue::Flag));
                det.steady(&mf.model, &o).unwrap()
            }
            Command::Deterministic(name, o, _) if name == "shock_paths" => {
                det.command(mf, name, o, &[], &dir(), "q1").unwrap()
            }
            Command::Deterministic(name, o, _) => {
                det.update_options(o);
                det.simul.noprint = true;
                det.simul.verbosity = 0;
                match name.as_str() {
                    "perfect_foresight_setup" => det.setup(mf).unwrap(),
                    "perfect_foresight_solver" => det.solve(&mf.model).unwrap(),
                    _ => {}
                }
            }
            _ => {}
        }
    }
    det
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| match r {
            Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect(),
            x => vec![x.as_f64().unwrap_or(f64::NAN)],
        })
        .collect()
}

fn check(case: &str, file: &str, cs: &str, tol: f64) {
    let r = &reference()[case];
    let mf = parse_mod_file(&dir().join(file), &[("CASE".into(), cs.into())]).unwrap();
    let det = run(&mf);
    let ss: Vec<f64> = r["steady_state"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for (j, v) in ss.iter().enumerate() {
        assert!(
            (det.steady_state[j] - v).abs() <= tol * v.abs().max(1.0),
            "{case}: steady state {j}: {} vs {v}",
            det.steady_state[j]
        );
    }
    if r.get("endo_simul").is_none() {
        return;
    }
    assert!(det.last.status, "{case}: solver failed");
    let e = det.endo_simul.as_ref().unwrap();
    let re = rows(&r["endo_simul"]);
    let mut worst: f64 = 0.0;
    for j in 0..mf.model.orig_endo_nbr {
        for c in 0..e.ncols() {
            let d = (e[(j, c)] - re[j][c]).abs() / re[j][c].abs().max(1.0);
            worst = worst.max(d);
        }
    }
    assert!(worst < tol, "{case}: max relative difference {worst:e}");
    println!(
        "{case}: max rel. diff {worst:.2e}, error {:e} (Dynare {:e})",
        det.last.error,
        r["error"].as_f64().unwrap()
    );
}

#[test]
fn stack_solve_algo_1_and_6_lbj() {
    check("pf_ssa1", "q1_pf.mod", "1", 1e-9);
    check("pf_ssa6", "q1_pf.mod", "5", 1e-9);
}

#[test]
fn stack_solve_algo_2_3_iterative() {
    check("pf_ssa2", "q1_pf.mod", "2", 1e-6);
    check("pf_ssa3ilu", "q1_pf.mod", "3", 1e-8);
    check("pf_ssa2bd", "q1_pf.mod", "4", 1e-8);
    check("pf_ssa3", "q1_pf.mod", "8", 1e-6);
}

#[test]
fn stack_solve_algo_7_and_robust() {
    check("pf_ssa7", "q1_pf.mod", "6", 1e-8);
    check("pf_ssa7d", "q1_pf.mod", "9", 1e-5);
    check("pf_robust", "q1_pf.mod", "7", 1e-8);
    check("pf_mlf", "q1_pf.mod", "10", 1e-8); // homotopy_marginal_linearization_fallback
}

#[test]
fn lmmcp_perfect_foresight_and_steady() {
    check("mcp_pf", "q1_mcp.mod", "1", 1e-8);
    check("mcp_pf7", "q1_mcp.mod", "3", 1e-8);
    check("mcp_ss", "q1_mcp.mod", "2", 1e-8);
}

/// `extended_path(lmmcp)` (order 0 and the order-1 stochastic extended path) on MATLAB's shocks.
#[test]
fn extended_path_lmmcp() {
    use dynare::ep::{EpModel, EpOptions, extended_path};
    use dynare::linalg::Mat;
    for (name, case) in [("ep0", "4"), ("sep1", "5")] {
        let r: Value =
            serde_json::from_str(&std::fs::read_to_string(dir().join(format!("ref_q1ep_{name}.json"))).unwrap())
                .unwrap();
        let mf = parse_mod_file(&dir().join("q1_mcp.mod"), &[("CASE".into(), case.into())]).unwrap();
        let mut det = DetState::new(&mf).unwrap();
        let mut o = EpOptions::default();
        for c in &mf.commands {
            match c {
                Command::Steady(opts) => {
                    let mut opts = opts.clone();
                    opts.push(("noprint".into(), OptValue::Flag));
                    det.steady(&mf.model, &opts).unwrap();
                }
                Command::Other(n, opts) if n == "extended_path" => {
                    o.update(opts).unwrap();
                }
                _ => {}
            }
        }
        let ex = rows(&r["exo_simul"]);
        let shocks = Mat::from_fn(ex.len() - 1, ex[0].len(), |t, k| ex[t + 1][k]);
        let m = EpModel {
            model: &mf.model,
            params: &det.params,
            steady_state: &det.steady_state,
            exo_steady_state: &det.exo_steady_state,
            sigma_e: &mf.sigma_e,
        };
        let out = extended_path(&m, &o, &det.steady_state.clone(), &shocks).unwrap();
        assert!(out.status, "{name}: extended path failed");
        let want = rows(&r["endo_simul"]);
        let mut err = 0.0f64;
        for (v, row) in want.iter().enumerate() {
            for (t, w) in row.iter().enumerate() {
                err = err.max((out.endo_simul[(v, t)] - w).abs());
            }
        }
        println!("{name}: max |diff| {err:.2e}");
        assert!(err < 1e-8, "{name}: max diff {err}");
    }
}

/// `model_diagnostics` prints what Dynare 7.1 prints (reference/q1/q1diag_<case>.log, from
/// reference/q1/dump_q1diag.m): no problem, static singularity (Walras law), unit root.
#[test]
fn model_diagnostics_output() {
    let tmp = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tmp/q1_diag");
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::copy(dir().join("q1_diag.mod"), tmp.join("q1_diag.mod")).unwrap();
    let keep = |l: &str| -> bool {
        let t = l.trim();
        !t.is_empty()
            && (t.starts_with("MODEL_DIAGNOSTICS")
                || t.starts_with("===")
                || t.starts_with("---")
                || t.starts_with("The ")
                || t.starts_with("There ")
                || l.starts_with("    ")
                || t.starts_with("DYNARE_SOLVE"))
    };
    for case in 1..=3 {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_dynare-rs"))
            .arg(format!("-DCASE={case}"))
            .arg(tmp.join("q1_diag.mod"))
            .output()
            .unwrap();
        let ours: Vec<String> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| keep(l))
            .map(|l| l.trim_end().to_string())
            .collect();
        let log = std::fs::read_to_string(dir().join(format!("q1diag_{case}.log"))).unwrap();
        let theirs: Vec<String> = log
            .split("Preprocessing time")
            .nth(1)
            .unwrap()
            .lines()
            .filter(|l| keep(l))
            .map(|l| l.trim_end().to_string())
            .collect();
        assert_eq!(ours, theirs, "case {case}");
    }
}

/// `shock_paths` (temporary, permanent and controlled shocks, `initval.` / `self.` scopes).
#[test]
fn shock_paths_block() {
    check("sp", "q1_sp.mod", "0", 1e-8);
    let r = &reference()["sp"];
    let mf = parse_mod_file(&dir().join("q1_sp.mod"), &[]).unwrap();
    let det = run(&mf);
    let x = det.exo_simul.as_ref().unwrap();
    let rx = rows(&r["exo_simul"]);
    for t in 0..x.nrows() {
        for k in 0..x.ncols() {
            assert!((x[(t, k)] - rx[t][k]).abs() < 1e-12, "exo_simul({t},{k}): {} vs {}", x[(t, k)], rx[t][k]);
        }
    }
    let xs: Vec<f64> = r["exo_steady_state"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for (a, b) in det.exo_steady_state.iter().zip(&xs) {
        assert!((a - b).abs() < 1e-14, "exo_steady_state {a} vs {b}");
    }
}

#[test]
fn purely_backward_forward_static() {
    check("back", "q1_back.mod", "1", 1e-9);
    check("forward", "q1_back.mod", "2", 1e-9);
    check("static", "q1_back.mod", "3", 1e-7);
}
