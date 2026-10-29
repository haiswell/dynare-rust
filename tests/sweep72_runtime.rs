//! Runtime behaviours of Dynare 7.2 found by running its test suite:
//! - `steady` on a `ramsey_model` in a perfect-foresight file goes through `dyn_ramsey_static`
//!   (tests/optimal_policy/nk_ramsey_det.mod; reference/sweep72/nk_ramsey_det.json from Dynare);
//! - `check(nocheck)` writes `options_.steadystate.nocheck` (tests/fs2000/fs2000a.mod);
//! - `stoch_simul(noprint)` returns silently on a Dynare info code
//!   (tests/stochastic_simulations/example1_noprint.mod);
//! - nested model-local variables in a Ramsey model (tests/partial_information/Partial_ramsey.mod).
mod common;

use std::path::{Path, PathBuf};

use dynare::deterministic::DetState;
use dynare::modfile::{Command, OptValue, parse_mod_file};
use serde_json::Value;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sweep72_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run_cli(dir: &Path, file: &str) -> (bool, String) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_dynare-rs"))
        .args([file, "nograph", "nointeractive"])
        .current_dir(dir)
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    (out.status.success(), s)
}

#[test]
fn ramsey_steady_in_perfect_foresight() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sweep72");
    let r: Value = common::json_ref(dir.join("nk_ramsey_det.json"));
    let mf = parse_mod_file(&dir.join("nk_ramsey_det.mod"), &[]).unwrap();
    let mut det = DetState::new(&mf).unwrap();
    for c in &mf.commands {
        match c {
            Command::Steady(o) => {
                let mut o = o.clone();
                o.push(("noprint".into(), OptValue::Flag));
                det.steady(&mf.model, &o).unwrap();
            }
            Command::Deterministic(name, o, _) => {
                det.simul.update(o);
                det.simul.noprint = true;
                det.simul.verbosity = 0;
                match name.as_str() {
                    "perfect_foresight_setup" => det.setup(&mf).unwrap(),
                    "perfect_foresight_solver" => det.solve(&mf.model).unwrap(),
                    _ => {}
                }
            }
            _ => {}
        }
    }
    let want: Vec<f64> = r["steady_state"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert_eq!(det.steady_state.len(), want.len());
    for (g, w) in det.steady_state.iter().zip(&want) {
        assert!((g - w).abs() < 1e-8 * (1.0 + w.abs()), "steady state {g} vs Dynare {w}");
    }
    let es = det.endo_simul.as_ref().unwrap();
    for (t, row) in r["endo_simul"].as_array().unwrap().iter().enumerate() {
        for (i, w) in row.as_array().unwrap().iter().enumerate() {
            let (g, w) = (es[(i, t)], w.as_f64().unwrap());
            assert!((g - w).abs() < 1e-8 * (1.0 + w.abs()), "endo_simul({i},{t}) {g} vs Dynare {w}");
        }
    }
}

const RBC: &str = "var y k; varexo e; parameters a b;\na=0.36; b=0.95;\n\
model; y = k(-1)^a*exp(e); k = b*a*y; end;\n";

#[test]
fn check_nocheck_skips_the_steady_state_residual_check() {
    let d = tmp("nocheck");
    // a steady_state_model that does not solve the model: only check(nocheck) lets it through
    let ssm = "steady_state_model; k = 1; y = 1; end;\nshocks; var e; stderr 0.01; end;\n";
    std::fs::write(d.join("nc.mod"), format!("{RBC}{ssm}check(nocheck);\nstoch_simul(order=1, irf=0);\n")).unwrap();
    let (ok, out) = run_cli(&d, "nc.mod");
    assert!(ok, "{out}");
    std::fs::write(d.join("c.mod"), format!("{RBC}{ssm}check;\n")).unwrap();
    let (ok, out) = run_cli(&d, "c.mod");
    assert!(!ok && out.contains("info 19"), "{out}");
}

#[test]
fn stoch_simul_noprint_returns_on_info() {
    let d = tmp("noprint");
    // b > 1: explosive (Blanchard-Kahn) model
    let m = "var y; varexo e; parameters b; b = 1.5;\nmodel; y = b*y(-1) + 0.1*y(-1)^2 + e; end;\n\
             steady_state_model; y = 0; end;\nshocks; var e; stderr 0.01; end;\n";
    std::fs::write(d.join("p.mod"), format!("{m}stoch_simul(order=1);\n")).unwrap();
    let (_, out) = run_cli(&d, "p.mod");
    assert!(out.contains("Blanchard & Kahn"), "{out}");
    // order 2 (the default, k-order path) as example1_noprint.mod
    std::fs::write(d.join("np2.mod"), format!("{m}stoch_simul(order=2, noprint);\n")).unwrap();
    let (ok, out) = run_cli(&d, "np2.mod");
    assert!(ok && !out.contains("Blanchard & Kahn"), "{out}");
}

#[test]
fn ramsey_with_nested_model_local_variables() {
    let d = tmp("nested_mlv");
    let src = |locals: &str| {
        format!(
            "var pi x i u; varexo eps_u; parameters betta kappa0 siggma rho_u;\n\
             betta=0.99; kappa0=0.1; siggma=1; rho_u=0.5;\nmodel(linear);\n{locals}\n\
             pi=betta*pi(+1)+kappa*x + u;\nx=-1/siggma*(i-pi(+1))+x(+1);\nu=rho_u*u(-1)+eps_u;\nend;\n\
             shocks; var eps_u = 1; end;\nplanner_objective pi^2 + 0.1*x^2;\n\
             ramsey_model(instruments=(i),planner_discount=betta);\nstoch_simul(order=1,irf=0) x pi u;\n"
        )
    };
    std::fs::write(d.join("flat.mod"), src("#kappa=kappa0*siggma;")).unwrap();
    std::fs::write(d.join("nested.mod"), src("#lam=kappa0; #kappa=lam*siggma;")).unwrap();
    let (ok1, a) = run_cli(&d, "flat.mod");
    let (ok2, b) = run_cli(&d, "nested.mod");
    assert!(ok1 && ok2, "{a}\n{b}");
    // the same decision rule (the printed policy and transition rules)
    let rules =
        |s: &str| s.lines().filter(|l| l.starts_with("x ") || l.starts_with("pi ")).collect::<Vec<_>>().join("\n");
    assert!(!rules(&a).is_empty());
    assert_eq!(rules(&a), rules(&b));
}
