//! `var_model`, `trend_component_model`, `var_expectation_model`, `var_expectation(...)` and
//! `var_expectation.initialize/update` against Dynare 7.2 (reference/varmodels, dump.m):
//! the reduced-form parameters `var_expectation_model_<m>_*` that `+var_expectation/update.m`
//! writes into `M_.params`, the companion matrices (`oo_.var.<m>.CompanionMatrix`,
//! `oo_.trend_component.<m>.CompanionMatrix`), and the first-order solution of models that
//! use the operator.

use dynare::analysis::{calibrated_steady_state, solve_calibration};
use dynare::modfile::{Command, ModFile, parse_mod_file};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/varmodels")
}

fn reference(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(refdir().join(format!("ref_{name}.json"))).unwrap()).unwrap()
}

fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        Value::Null => vec![f64::NAN],
        _ => vec![],
    }
}

/// Parses `name.mod` and runs its `var_expectation.*` statements in file order.
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

fn close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{what}[{i}]: {x} vs {y}");
    }
}

fn check_params(name: &str, mf: &ModFile, r: &Value) {
    let names: Vec<String> =
        r["param_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(names, mf.model.param_names, "{name}: parameter names/order");
    close(&mf.model.params, &flat(&r["params"]), 1e-12, &format!("{name}: M_.params"));
}

/// Column-major companion matrix of MATLAB (jsonencode gives rows).
fn companion_row_major(m: &dynare::linalg::Mat) -> Vec<f64> {
    (0..m.nrows()).flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)])).collect()
}

#[test]
fn var_with_constant_and_lags() {
    let r = reference("vm1");
    let r = &r;
    let mf = run_commands("vm1");
    check_params("vm1", &mf, r);
    let c = mf.varmodels.companion("vv").unwrap();
    assert!(c.is_constant);
    close(&companion_row_major(&c.matrix), &flat(&r["var"]["vv"]["CompanionMatrix"]), 1e-13, "vm1 companion");
}

#[test]
fn structural_var_without_constant() {
    let r = reference("vm2");
    let r = &r;
    let mf = run_commands("vm2");
    check_params("vm2", &mf, r);
    let c = mf.varmodels.companion("vv").unwrap();
    assert!(!c.is_constant);
    close(&companion_row_major(&c.matrix), &flat(&r["var"]["vv"]["CompanionMatrix"]), 1e-13, "vm2 companion");
}

#[test]
fn trend_component_model() {
    let r = reference("vm3");
    let r = &r;
    let mf = run_commands("vm3");
    check_params("vm3", &mf, r);
    let c = mf.varmodels.companion("tc").unwrap();
    assert_eq!(c.variables, ["y", "x", "ys", "xs"]);
    close(
        &companion_row_major(&c.matrix),
        &flat(&r["trend_component"]["tc"]["CompanionMatrix"]),
        1e-13,
        "vm3 companion",
    );
}

fn solution_matches(name: &str) {
    let r = reference(name);
    let r = &r;
    let mf = run_commands(name);
    let ys = calibrated_steady_state(&mf, &mf.model.params).unwrap();
    close(&ys, &flat(&r["ys"]), 1e-9, &format!("{name}: steady state"));
    let sol = solve_calibration(&mf, 1.0 + 1e-6).unwrap().unwrap();
    let dr = &sol.dr;
    let order: Vec<f64> = dr.order_var.iter().map(|&i| (i + 1) as f64).collect();
    close(&order, &flat(&r["order_var"]), 0.0, &format!("{name}: order_var"));
    let rows = |m: &dynare::linalg::Mat| companion_row_major(m);
    close(&rows(&dr.ghx), &flat(&r["ghx"]), 1e-9, &format!("{name}: ghx"));
    close(&rows(&dr.ghu), &flat(&r["ghu"]), 1e-9, &format!("{name}: ghu"));
}

#[test]
fn first_order_solution_with_var_expectation_vm1() {
    solution_matches("vm1");
}

#[test]
fn first_order_solution_with_var_expectation_vm2() {
    solution_matches("vm2");
}

#[test]
fn errors_like_dynare() {
    let bad = |src: &str| dynare::modfile::parse_mod_str(src, &[]).err().map(|e| format!("{e:#}")).unwrap_or_default();
    let head = "var y x; varexo e; parameters a; a=0.5;\nvar_model(model_name=m, eqtags=['e1']);\n";
    // unknown equation tag
    let e = bad(&format!("{head}model;\n[name='zz']\ny = a*y(-1) + e;\nx = y;\nend;"));
    assert!(e.contains("not found"), "{e}");
    // a contemporaneous variable on the right-hand side of a non-structural VAR
    let e = bad(&format!("{head}model;\n[name='e1']\ny = a*x + e;\nx = y;\nend;"));
    assert!(e.contains("structural"), "{e}");
    // unknown expectation model
    let e = bad("var y; varexo e; parameters a; a=0.5;\nmodel;\ny = var_expectation(nope) + e;\nend;");
    assert!(e.contains("unknown var_expectation_model"), "{e}");
}

/// Dynare 7.2 tests/var-expectations/8: `expression = diff(log(x))` (the auxiliary variable of
/// the VAR's left-hand side), `var_expectation.initialize('varexp')` without ';' (a native
/// MATLAB statement ends with its line) and `var_expectation.print('varexp')`.
#[test]
fn var_expectation_of_diff_log_and_print() {
    let r = reference("ve_diff_log");
    let mut mf = parse_mod_file(&refdir().join("ve_diff_log.mod"), &[]).unwrap();
    let dir = std::env::temp_dir().join(format!("dynare_rs_ve_print_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut ran = Vec::new();
    for ci in 0..mf.commands.len() {
        if let Command::Other(n, mut o) = mf.commands[ci].clone()
            && dynare::varmodels::handles(&n)
        {
            o.push(("dir".into(), dynare::modfile::OptValue::Str(dir.to_string_lossy().into_owned())));
            o.push(("fname".into(), dynare::modfile::OptValue::Str("example1".into())));
            dynare::varmodels::run(&mut mf, &n, &o).unwrap();
            ran.push(n);
        }
    }
    assert_eq!(ran, ["var_expectation.initialize", "var_expectation.update", "var_expectation.print"]);
    let endo: Vec<String> = flat_names(&r["endo_names"]);
    assert_eq!(mf.model.endo_names, endo, "endo_names");
    check_params("ve_diff_log", &mf, &r);
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap();
    let pars = read("example1/model/var-expectations/varexp-parameters.inc");
    assert!(pars.contains("\nvar_expectation_model_varexp_AUX_DIFF_60_0 = 0.8823529411764706;\n"), "{pars}");
    let expr = read("example1/model/var-expectations/varexp-expression.inc");
    let body: Vec<&str> = expr.lines().skip(1).collect();
    assert_eq!(
        body,
        [
            "var_expectation_model_varexp_constant + var_expectation_model_varexp_AUX_DIFF_60_0*diff(log(x))",
            " + var_expectation_model_varexp_AUX_UOP_42_0*log(y)",
            " + var_expectation_model_varexp_AUX_DIFF_32_0*diff(z)",
            " + var_expectation_model_varexp_AUX_DIFF_60_1*diff(log(x(-1)))",
            " + var_expectation_model_varexp_AUX_UOP_42_1*log(y(-1))",
            " + var_expectation_model_varexp_AUX_DIFF_32_1*diff(z(-1))",
        ]
    );
    let ev = read("+example1/+var_expectations/+varexp/evaluate.m");
    assert!(
        ev.contains(
            "ds.varexp = 0.0000000000000000+0.8823529411764706*dbase.x.log().diff()+0.0000000000000000*dbase.y.log()"
        ),
        "{ev}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

fn flat_names(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect()
}
