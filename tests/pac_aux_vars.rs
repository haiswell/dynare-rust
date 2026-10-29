//! VAR / trend component models whose variables are expressions of variables, and the
//! auxiliary variables of the PAC models, against Dynare 7.2 (tests/pac of Dynare,
//! reference/pac/aux_vars: the .mod files cut after the parameter update; ref_aux.json: `M_` and
//! `oo_` of MATLAB):
//! - `diff(diff(x))`, `diff(log(x))`, `log(x)` left-hand sides (unary-op `AUX_UOP_<n>` and nested
//!   `AUX_DIFF_<n>` variables: the companion matrix and the `h` parameters named after them);
//! - `component log(x)` and `growth diff(log(x(-2)))` in `pac_target_info`;
//! - the PAC variables created after the diff auxiliaries (`pac_expectation_<m>`, `vns`,
//!   `mce_Z1_<m>` with its `AUX_DIFF_LEAD` chain), the node numbering of the names;
//! - a PAC target `log(x(-1))`: `pac.estimate.iterative_ols` on the data simulated by MATLAB.

use dynare::modfile::{Command, ModFile, parse_mod_file};
use dynare::varmodels::pac_estimate::{PacData, estimate_iterative_ols};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/pac/aux_vars")
}

fn reference(name: &str) -> Value {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(refdir().join("ref_aux.json")).unwrap()).unwrap();
    r[name].clone()
}

fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        Value::Null => vec![f64::NAN],
        _ => vec![],
    }
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect()
}

fn close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x.is_nan() && y.is_nan()) || (x - y).abs() <= tol * (1.0 + y.abs()), "{what}[{i}]: {x} vs {y}");
    }
}

/// Parses `name.mod` and runs its `pac.*` statements in file order.
fn run_commands(name: &str) -> ModFile {
    let mut mf = parse_mod_file(&refdir().join(format!("{name}.mod")), &[]).unwrap();
    for ci in 0..mf.commands.len() {
        if let Command::Other(n, o) = mf.commands[ci].clone()
            && dynare::varmodels::handles(&n)
        {
            dynare::varmodels::run(&mut mf, &n, &o).unwrap();
        }
    }
    mf
}

/// `M_.endo_names`, `M_.param_names`, `M_.params` and the companion matrix.
fn check(name: &str, params: bool) -> ModFile {
    let r = reference(name);
    let mf = run_commands(name);
    assert_eq!(mf.model.endo_names, strings(&r["endo_names"]), "{name}: M_.endo_names");
    assert_eq!(mf.model.param_names, strings(&r["param_names"]), "{name}: M_.param_names");
    if params {
        close(&mf.model.params, &flat(&r["params"]), 1e-10, &format!("{name}: M_.params"));
    }
    if let Some(c) = r.get("companion") {
        let rows: Vec<Vec<f64>> = c.as_array().unwrap().iter().map(flat).collect();
        let comp = mf.varmodels.companions.values().next().unwrap();
        let m = &comp.matrix;
        assert_eq!(m.nrows(), rows.len(), "{name}: companion size");
        for (i, row) in rows.iter().enumerate() {
            let got: Vec<f64> = (0..m.ncols()).map(|j| m[(i, j)]).collect();
            close(&got, row, 1e-12, &format!("{name}: CompanionMatrix row {i}"));
        }
    }
    mf
}

#[test]
fn trend_component_with_diff_of_diff() {
    // diff(diff(x1)) = ... diff(x1(-1)) - diff(x1bar(-1)) ...: AUX_DIFF_97 of AUX_DIFF_9; the
    // companion variables are AUX_DIFF_9 (undiff), x2, AUX_DIFF_64 (target), x2bar
    let mf = check("tc2", true);
    let comp = mf.varmodels.companions.values().next().unwrap();
    assert_eq!(comp.variables, ["AUX_DIFF_9", "x2", "AUX_DIFF_64", "x2bar"]);
}

#[test]
fn trend_component_with_logs() {
    // diff(log(x2)), log(x2bar): AUX_UOP_32, AUX_UOP_50
    check("tc3", true);
}

#[test]
fn trend_component_term_outside_the_model_is_skipped() {
    // diff(diff(x1(-2))) in an equation of diff(diff(log(x1))): not an autoregressive term
    // (fillAutoregressiveRow skips it), and the PAC target log(x1) is found through the
    // AUX_DIFF variable whose original variable it is
    check("tc11", true);
}

#[test]
fn var_with_log_components_and_growth() {
    // component log(x) / log(w) (kind dd) of the target v = log(x) + .5*log(w) + d_y*y, growth
    // diff(log(x(-2))) (AUX_DIFF_LAG of AUX_DIFF of AUX_UOP)
    check("v5", true);
}

#[test]
fn model_consistent_expectations_variables_after_the_diff_ones() {
    // mce_Z1_pacman after the diff auxiliaries, AUX_DIFF_LEAD_163/168, AUX_ENDO_LEAD_152/159
    check("mce1", true);
}

#[test]
fn iterative_ols_with_a_log_target() {
    // e_c_m*(log(x1(-1)) - z(-1)): the target is AUX_UOP_22, computed from the data
    let r = reference("tc10");
    let mf = check("tc10", false);
    let names = strings(&r["sim"]["names"]);
    let rows = r["sim"]["data"].as_array().unwrap();
    let mut cols = vec![Vec::new(); names.len()];
    for row in rows {
        for (j, v) in flat(row).into_iter().enumerate() {
            cols[j].push(v);
        }
    }
    let data = PacData::new(names, cols);
    let mut params = mf.model.params.clone();
    let guess = [("e_c_m".to_string(), 0.9), ("c_z_1".to_string(), 0.5), ("c_z_2".to_string(), 0.2)];
    // 2005Q1:2000Q1+200 of data starting in 2000Q1
    let est =
        estimate_iterative_ols(&mf.varmodels, "zpac", &mf.model.param_names, &mut params, &guess, &data, (20, 200))
            .unwrap();
    let m = &r["ols"];
    assert_eq!(est.parnames, strings(&m["parnames"]));
    close(&est.estimator, &flat(&m["estimator"]), 1e-9, "OLS estimates");
    close(&[est.ssr], &flat(&m["ssr"]), 1e-9, "OLS ssr");
    close(&est.residual, &flat(&m["residual"]), 1e-8, "OLS residuals");
    // the h parameters are updated at the estimates
    close(&params, &flat(&r["params"]), 1e-8, "M_.params after OLS");
}

/// `init(:,[2,4]) = ones(10,2);` on a workspace array before `dseries(init, ...)` (the initial
/// conditions of tests/pac/trend-component-3, with logged variables at 1).
#[test]
fn indexed_assignment_of_a_workspace_array() {
    use dynare::deterministic::workspace::WsValue;
    let d = std::env::temp_dir().join(format!("pac_aux_ws_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join("ws.mod");
    std::fs::write(
        &f,
        "var y; varexo e; parameters a; a = 0.5;\nmodel; y = a*y(-1) + e; end;\n\
         init = zeros(3, 4);\ninit(:,[2,4]) = ones(3,2);\ninit(2,3) = 7;\ninit(5) = -1;\n",
    )
    .unwrap();
    let mf = parse_mod_file(&f, &[]).unwrap();
    let Some(WsValue::Num(m)) = mf.q1.workspace.vars.get("init") else { panic!("init is not a numeric array") };
    let want = [[0.0, 1.0, 0.0, 1.0], [0.0, -1.0, 7.0, 1.0], [0.0, 1.0, 0.0, 1.0]];
    for (i, row) in want.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            assert_eq!(m[(i, j)], *v, "init({}, {})", i + 1, j + 1);
        }
    }
}
