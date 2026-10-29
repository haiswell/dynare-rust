//! Perturbation at any order (`k_order_perturbation` at order >= 4) vs Dynare 7.2:
//! - the general solver agrees with the dedicated order-2/3 solver;
//! - tests/optimal_policy/neo_growth_k_order.mod and neo_growth_ramsey_k_order.mod (order 6):
//!   `oo_.dr.g_0 .. g_6` and the conditional welfare of `evaluate_planner_objective`
//!   (reference/korder/ref_neo_growth_k_order.json, from Dynare's own runs of those tests);
//! - `k_order_welfare` at any order: the welfare `W = U + β W(+1)` written as a model equation
//!   (neo_growth_k_order.mod) has the same rule as the one solved from `U` alone;
//! - simulations at order 5 (`k_order_simul`, with and without pruning) with Dynare's shocks
//!   (reference/korder/ref_ng5.json, written by reference/korder/dump_ng5.m).
mod common;

use dynare::dsge::DsgeModel;
use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use dynare::solver::korder::{self, KOrderDr, general, simulate};
use dynare::solver::{FirstOrderSolver, SolverOptions};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/korder")
}

/// Matrix from jsonencode (array of rows, or a flat vector read as a column).
fn mat(v: &Value) -> Mat {
    let num = |v: &Value| v.as_f64().unwrap_or(f64::NAN);
    match v {
        Value::Array(rows) if rows.first().is_some_and(|x| x.is_array()) => {
            let c = rows[0].as_array().unwrap().len();
            Mat::from_fn(rows.len(), c, |i, j| num(&rows[i][j]))
        }
        Value::Array(a) => Mat::from_fn(a.len(), 1, |i, _| num(&a[i])),
        x => Mat::from_element(1, 1, num(x)),
    }
}

fn assert_close(what: &str, got: &Mat, want: &Mat, tol: f64) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let scale = 1.0 + want.abs().max();
    let err = (got - want).abs().max();
    assert!(err < tol * scale, "{what}: max abs diff {err}");
}

fn model(file: &str) -> DsgeModel {
    let mf = parse_mod_file(&dir().join(file), &[]).unwrap();
    DsgeModel::new(&mf, Some(&[])).unwrap()
}

fn solve(d: &DsgeModel, order: usize) -> KOrderDr {
    let m = &d.model;
    korder::solve_model(m, &d.ys_calib, &m.params, &d.base_sigma_e, order, SolverOptions::default()).unwrap()
}

#[test]
fn general_solver_matches_order_3_solver() {
    for model in ["rbc", "nk"] {
        let mf = parse_mod_file(&dir().join(format!("{model}.mod")), &[("ORDER".into(), "3".into())]).unwrap();
        let dsge = DsgeModel::new(&mf, Some(&[])).unwrap();
        let m = &dsge.model;
        let ys = dsge.ys_calib.clone();
        let x = vec![0.0; m.n_exo()];
        let want = solve(&dsge, 3);
        let g1 = m.dynamic_jacobian(&ys, &x, &m.params);
        let dr1 =
            FirstOrderSolver::new(&m.lead_lag_incidence, m.n_exo(), SolverOptions::default()).solve(&g1, &ys).unwrap();
        let d = m.dynamic_derivatives_k(3);
        let v = d.eval(&ys, &x, &m.params);
        let got = general::solve(&dr1, &m.lead_lag_incidence, &g1, &d, &v, &dsge.base_sigma_e, 3).unwrap();
        for k in 0..3 {
            assert_close(&format!("{model} order {}", k + 1), &got.derivs[k], &want.derivs[k], 1e-11);
        }
    }
}

fn run_cli(dir: &Path, file: &str) -> String {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_dynare-rs"))
        .args([file, "nograph", "nointeractive"])
        .current_dir(dir)
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{file}:\n{s}");
    s
}

#[test]
fn order_6_matches_dynare() {
    let r: Value = common::json_ref(dir().join("ref_neo_growth_k_order.json"));
    let tmp = common::temp_dir("korder_order6");
    for f in [
        "neo_growth_common.inc",
        "neo_growth_ramsey_common.inc",
        "neo_growth_k_order.mod",
        "neo_growth_ramsey_k_order.mod",
    ] {
        std::fs::copy(dir().join(f), tmp.join(f)).unwrap();
    }
    for (name, tol) in [("neo_growth_k_order", 1e-12), ("neo_growth_ramsey_k_order", 1e-10)] {
        run_cli(&tmp, &format!("{name}.mod"));
        let out = tmp.join(name).join("Output");
        let got: Value = common::json_ref(out.join(format!("{name}_stoch_simul.json")));
        let want = &r[name];
        assert_eq!(got["dr"]["order_var"], want["order_var"], "{name}: order_var");
        for j in 0..=6 {
            assert_close(&format!("{name} g_{j}"), &mat(&got["dr"][format!("g_{j}")]), &mat(&want["g"][j]), tol);
        }
        if let Some(w) = want["conditional_welfare"].as_f64() {
            let p: Value = common::json_ref(out.join(format!("{name}_planner_objective.json")));
            let g = p["conditional_zero_initial_multiplier"].as_f64().unwrap();
            assert!((g - w).abs() < 1e-10 * w.abs(), "{name}: conditional welfare {g} vs {w}");
        }
    }
}

#[test]
fn k_order_welfare_matches_the_welfare_equation() {
    // W = U + beta W(+1), U = ln(c): the W row of the decision rule is the k_order_welfare of U
    let d = model("neo_growth_k_order.mod");
    let m = &d.model;
    let (iw, ic) = (m.endo_index("W").unwrap(), m.endo_index("c").unwrap());
    let beta = m.params[m.param_names.iter().position(|p| p == "beta").unwrap()];
    let c = d.ys_calib[ic];
    for order in [3, 6] {
        let kdr = solve(&d, order);
        // d^m ln(c) / dc^m = (-1)^(m-1) (m-1)! / c^m
        let ud: Vec<Vec<(Vec<usize>, f64)>> = (1..=order)
            .map(|k| {
                let f: f64 = (1..k).product::<usize>() as f64;
                vec![(vec![ic; k], if k % 2 == 1 { 1.0 } else { -1.0 } * f / c.powi(k as i32))]
            })
            .collect();
        let w = general::k_order_welfare(&kdr, &ud, beta, &d.base_sigma_e).unwrap();
        let row = kdr.first.inv_order_var[iw];
        for k in 0..order {
            let want = kdr.derivs[k].rows(row, 1).into_owned();
            assert_close(&format!("order {order}: W_{}", k + 1), &w[k], &want, 1e-10);
        }
        if order == 3 {
            // the dedicated order-3 port
            let n = m.n_endo();
            let mut uy = vec![0.0; n];
            uy[ic] = 1.0 / c;
            let mut uyy = Mat::zeros(n, n);
            uyy[(ic, ic)] = -1.0 / (c * c);
            let u3 = vec![(ic, ic, ic, 2.0 / c.powi(3))];
            let w3 = dynare::policy::welfare::k_order_welfare(&kdr, &uy, &uyy, &u3, beta, &d.base_sigma_e).unwrap();
            for k in 0..3 {
                assert_close(&format!("order 3 port: W_{}", k + 1), &w[k], &w3[k], 1e-11);
            }
        }
    }
}

#[test]
fn order_5_simulations_match_dynare() {
    let r: Value = common::json_ref(dir().join("ref_ng5.json"));
    let d = model("ng5.mod");
    let kdr = solve(&d, 5);
    for j in 0..=5 {
        assert_close(&format!("g_{j}"), &kdr.g_k(j), &mat(&r["g"][j]), 1e-12);
    }
    let ex = mat(&r["exo_simul"]); // periods x n_exo
    let ys = &d.ys_calib;
    // stoch_simul(periods): oo_.endo_simul (k_order_simul without pruning) from the steady state
    let y = simulate::simult(&kdr, ys, &ex, false);
    assert_close("endo_simul", &y.columns(1, ex.nrows()).into_owned(), &mat(&r["endo_simul"]), 1e-12);
    // simult_ from another initial state
    let y0: Vec<f64> = mat(&r["y0"]).iter().copied().collect();
    assert_close("simult_ from y0", &simulate::simult(&kdr, &y0, &ex, false), &mat(&r["endo_simul_y0"]), 1e-12);
    // pruning at order > 3: k_order_simul's pruned state space
    assert_close("pruned simult_", &simulate::simult(&kdr, ys, &ex, true), &mat(&r["endo_simul_pruned"]), 1e-12);
}
