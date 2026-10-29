//! Endogenous and exogenous variables in the expressions of `shocks` / `mshocks` blocks: the
//! preprocessor compiles them to `oo_.steady_state(i)` / `oo_.exo_steady_state(k)`, read where
//! the block stands in the driver (initval before `steady`, the computed steady state after).
//! Numbers: Dynare 7.2 on the .mod files of reference/shocks_endo (ref.json).
mod common;

use dynare::driver::{RunOptions, run};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/shocks_endo")
}

fn run_ref(name: &str) -> PathBuf {
    let dir = common::temp_dir(&format!("shocks_endo_{name}"));
    std::fs::copy(refdir().join(format!("{name}.mod")), dir.join(format!("{name}.mod"))).unwrap();
    run(&dir.join(format!("{name}.mod")), &RunOptions { quiet: true, ..Default::default() }).unwrap();
    dir.join(name).join("Output")
}

fn matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()).collect()
}

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-9 * (1.0 + b.abs()), "{what}: rust {a} vs dynare {b}");
}

fn check_var(name: &str) {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    let out = run_ref(name);
    let j: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join(format!("{name}_stoch_simul.json"))).unwrap()).unwrap();
    let (got, want) = (matrix(&j["var"]), matrix(&r[format!("{name}_var")]));
    for i in 0..2 {
        for k in 0..2 {
            close(got[i][k], want[i][k], &format!("{name} var[{i}][{k}]"));
        }
    }
}

/// Before `steady`, oo_.steady_state holds the initval (y = 1, c = 2).
#[test]
fn stderr_and_corr_read_initval() {
    check_var("s1");
}

/// A second block after `steady` reads the computed steady state (y = 5, c = 2.5, u = 0.5).
#[test]
fn shocks_block_after_steady_reads_the_steady_state() {
    check_var("s2");
}

fn check_simul(name: &str) {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    let out = run_ref(name);
    let rows = |f: &str| -> Vec<Vec<f64>> {
        let s = std::fs::read_to_string(out.join(format!("{name}_{f}.csv"))).unwrap();
        s.lines().skip(1).map(|l| l.split(',').map(|v| v.parse().unwrap()).collect()).collect()
    };
    // MATLAB: one row per variable, columns are the periods; the csv: period, variables
    let endo = matrix(&r[format!("{name}_endo_simul")]);
    for (t, row) in rows("endo_simul").iter().enumerate() {
        for (k, e) in endo.iter().enumerate() {
            close(row[k + 1], e[t], &format!("endo_simul[{k}][{t}]"));
        }
    }
    let exo = matrix(&r[format!("{name}_exo_simul")]);
    for (t, row) in rows("exo_simul").iter().enumerate() {
        for k in 0..2 {
            close(row[k + 1], exo[t][k], &format!("exo_simul[{t}][{k}]"));
        }
    }
}

/// Deterministic values and mshocks, with `steady` between the blocks.
#[test]
fn deterministic_values_with_endogenous_variables() {
    check_simul("d");
}

/// `shocks(learnt_in=2)` and `endval(learnt_in=3)` after `steady`.
#[test]
fn learnt_in_values_with_endogenous_variables() {
    check_simul("l");
}
