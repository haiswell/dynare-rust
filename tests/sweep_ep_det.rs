//! Divergences found by running Dynare 7.2's own test suite (numbers from the MATLAB runs
//! named in reference/sweep_ep_det/ref.json):
//! * `extended_path(order = 0)` is the deterministic extended path (the default order), not
//!   an invalid perturbation order; on a purely backward model each period starts from the
//!   previous one (`sim1_purely_backward`; ep/ar_backward.mod, ep/rbc_bytecode.mod);
//! * `hybrid = 4`: the correction is the constant `g_0` of the fourth-order rule
//!   (ep/hybrid/rs.mod);
//! * evaluate blocks of `model(block)` are computed in closed form equation by equation, even
//!   when their mfs is the whole block (deterministic_simulations/rbc_det_exo_lag_2c.mod: the
//!   simultaneous Newton step made `Efficiency = exp(efficiency)` linear);
//! * `model_remove(..., foo = 'eq:dummy2')`: a bare `tag = 'value'` selection
//!   (preprocessor_checks/ramst_model_edit.mod);
//! * an `initval` block after an `endval` block is rejected
//!   (deterministic_simulations/rbc_det4_endval_initval_xfail.mod).
mod common;

use dynare::deterministic::DetState;
use dynare::driver::{RunOptions, run};
use dynare::ep::{EpModel, EpOptions, extended_path};
use dynare::linalg::Mat;
use dynare::modfile::{Command, OptValue, parse_mod_file};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sweep_ep_det")
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| match r {
            Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap()).collect(),
            x => vec![x.as_f64().unwrap()],
        })
        .collect()
}

/// Runs the `extended_path` of `<name>.mod` on MATLAB's shocks; max |diff| of the paths.
fn ep_diff(name: &str) -> f64 {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    let r = &r[name];
    let mf = parse_mod_file(&refdir().join(format!("{name}.mod")), &[]).unwrap();
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
    let mut err = 0.0f64;
    for (v, row) in rows(&r["endo_simul"]).iter().enumerate() {
        for (t, w) in row.iter().enumerate() {
            err = err.max((out.endo_simul[(v, t)] - w).abs());
        }
    }
    err
}

#[test]
fn ep_order_zero_is_the_deterministic_extended_path() {
    // purely backward: the guess of each period is the previous period (sim1_purely_backward)
    let err = ep_diff("ar_backward");
    assert!(err < 1e-9, "max diff {err}");
    let err = ep_diff("rbc_bytecode");
    assert!(err < 1e-9, "rbc_bytecode: max diff {err}");
}

#[test]
fn sep_hybrid4_uses_the_fourth_order_constant() {
    let err = ep_diff("rs_hybrid4");
    assert!(err < 1e-8, "max diff {err}");
}

/// Runs `<name>.mod` (a perfect-foresight simulation) and compares the first periods of
/// `oo_.endo_simul` with Dynare's.
fn check_endo_simul(name: &str) {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    let want = rows(&r[name]["endo_simul"]);
    let dir = common::temp_dir(&format!("sweep_ep_det_{name}"));
    std::fs::copy(refdir().join(format!("{name}.mod")), dir.join(format!("{name}.mod"))).unwrap();
    run(&dir.join(format!("{name}.mod")), &RunOptions { quiet: true, ..Default::default() }).unwrap();
    let csv = std::fs::read_to_string(dir.join(format!("{name}/Output/{name}_endo_simul.csv"))).unwrap();
    // period, then the variables (declared + auxiliary) in M_.endo_names order
    let got: Vec<Vec<f64>> =
        csv.lines().skip(1).map(|l| l.split(',').skip(1).map(|x| x.parse().unwrap()).collect()).collect();
    for (v, row) in want.iter().enumerate() {
        for (t, w) in row.iter().enumerate() {
            let a = got[t][v];
            assert!((a - w).abs() < 1e-10 * (1.0 + w.abs()), "{name}: endo {v} period {t}: rust {a} vs dynare {w}");
        }
    }
}

#[test]
fn block_evaluate_blocks_are_solved_in_closed_form() {
    check_endo_simul("rbc_det_exo_lag_2c");
}

#[test]
fn model_remove_selects_by_tag_equals_value() {
    // model_remove('eq:dummy1', foo = 'eq:dummy2'): the bare `tag = 'value'` form
    check_endo_simul("ramst_model_edit");
}

#[test]
fn initval_after_endval_is_rejected() {
    let dir = common::temp_dir("sweep_ep_det_initval");
    let p = dir.join("t.mod");
    std::fs::write(
        &p,
        "var y; varexo e; parameters a; a = 0.5;\nmodel; y = a*y(-1) + e; end;\n\
         endval; e = 1; end;\ninitval; e = 0; end;\n",
    )
    .unwrap();
    let e = parse_mod_file(&p, &[]).unwrap_err().to_string();
    assert!(e.contains("an 'initval' block cannot appear after an 'endval' block"), "{e}");
}
