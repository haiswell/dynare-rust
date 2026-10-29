//! PAC models against Dynare 7.2 (reference/pac, dump.m): `pac_model`, `pac_target_info`,
//! `pac_expectation`, `pac.initialize`, `pac.update.expectation`, `pac.mce.parameters`
//! (the h / growth neutrality / mce_alpha parameters and the first-order solution of the
//! models that use them), and `pac.estimate.nls` / `pac.estimate.iterative_ols` on data
//! simulated by `simul_backward_model` in MATLAB.

use dynare::analysis::{calibrated_steady_state, solve_calibration};
use dynare::modfile::{Command, ModFile, parse_mod_file};
use dynare::varmodels::pac_estimate::{PacData, estimate_iterative_ols, estimate_nls, pac_residuals};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/pac")
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

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect()
}

/// Parses `name.mod` and runs its `var_expectation.*` / `pac.*` statements in file order.
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

fn pac_param(n: &str) -> bool {
    n.starts_with("h_") || n.contains("pac_growth_neutrality") || n.starts_with("mce_alpha")
}

/// Declared parameters by name; the parameters created for the PAC models (whose names contain
/// Dynare's auxiliary variable numbering) by position.
fn check_params(name: &str, mf: &ModFile, r: &Value) {
    let rn = strings(&r["param_names"]);
    let rv = flat(&r["params"]);
    assert_eq!(rn.len(), mf.model.param_names.len(), "{name}: number of parameters");
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (i, n) in rn.iter().enumerate() {
        if pac_param(n) {
            a.push(mf.model.params[i]);
            b.push(rv[i]);
            assert!(
                pac_param(&mf.model.param_names[i]),
                "{name}: parameter {i} is {} not {n}",
                mf.model.param_names[i]
            );
        } else {
            assert_eq!(&mf.model.param_names[i], n, "{name}: parameter name");
            close(&[mf.model.params[i]], &[rv[i]], 1e-12, &format!("{name}: {n}"));
        }
    }
    assert!(!a.is_empty());
    close(&a, &b, 1e-10, &format!("{name}: PAC parameters"));
}

#[test]
fn var_composite_target_and_growth() {
    let mf = run_commands("pc1");
    check_params("pc1", &mf, &reference("pc1"));
}

#[test]
fn trend_component_growth() {
    let mf = run_commands("pc2");
    check_params("pc2", &mf, &reference("pc2"));
}

#[test]
fn var_simple_target() {
    let mf = run_commands("pc3");
    check_params("pc3", &mf, &reference("pc3"));
}

/// `pac.bgp.set` (+pac/+bgp/set.m): `s` with a nonzero mean in the optimizing part enters the
/// growth neutrality correction; numeric means of zero (`x`, `dx2`) leave it unchanged.
#[test]
fn bgp_set_growth_neutrality() {
    let mf = run_commands("pc10");
    check_params("pc10", &mf, &reference("pc10"));
    let p = mf.varmodels.pac("pacman").unwrap();
    let bgp = |g: &str, v: &str| p.bgp.iter().find(|t| t.group == g && t.var == v).map(|t| t.bgp);
    use dynare::varmodels::pac::PacBgp;
    assert_eq!(bgp("optim_additive", "s"), Some(PacBgp::Bool(true)));
    assert_eq!(bgp("optim_additive", "dv"), Some(PacBgp::Bool(false)));
    assert_eq!(bgp("non_optimizing_behaviour", "x"), Some(PacBgp::Num(0.0)));
    assert_eq!(bgp("non_optimizing_behaviour", "y"), Some(PacBgp::Bool(false)));
    assert_eq!(bgp("additive", "dx2"), Some(PacBgp::Num(0.0)));
    assert_eq!(bgp("additive", "u"), Some(PacBgp::Bool(false)));
    assert_eq!(bgp("additive", "ez"), Some(PacBgp::Bool(false)));
    assert_eq!(p.bgp.len(), 7);
}

#[test]
fn model_consistent_expectations_parameters() {
    let mf = run_commands("pc4");
    check_params("pc4", &mf, &reference("pc4"));
}

/// Steady state, ghu/ghx of the original variables and impulse responses (recursion on the
/// Rust decision rule) against stoch_simul.
fn solution_matches(name: &str) {
    let r = reference(name);
    let mf = run_commands(name);
    let endo = &mf.model.endo_names;
    let rendo = strings(&r["endo_names"]);
    let ys = calibrated_steady_state(&mf, &mf.model.params).unwrap();
    let rys = flat(&r["ys"]);
    let sol = solve_calibration(&mf, 1.0 + 1e-6).unwrap().unwrap();
    let dr = &sol.dr;
    let n = endo.len();
    let nexo = mf.model.exo_names.len();
    let orig = r["orig_endo_nbr"].as_u64().unwrap() as usize;
    for k in 0..orig {
        let j = endo.iter().position(|e| *e == rendo[k]).unwrap();
        close(&[ys[j]], &[rys[k]], 1e-9, &format!("{name}: steady state {}", rendo[k]));
    }
    let rorder = flat(&r["order_var"]);
    let rstate = flat(&r["state_var"]);
    let (rghx, rghu) = (flat(&r["ghx"]), flat(&r["ghu"]));
    let nsp = rstate.len();
    // jsonencode gives rows
    for (i, &oi) in rorder.iter().enumerate() {
        let vname = &rendo[oi as usize - 1];
        if vname.starts_with("AUX_") {
            continue;
        }
        let ri = dr.order_var.iter().position(|&e| endo[e] == *vname).unwrap();
        for s in 0..nexo {
            close(&[dr.ghu[(ri, s)]], &[rghu[i * nexo + s]], 1e-9, &format!("{name}: ghu {vname} {s}"));
        }
        for (c, &sv) in rstate.iter().enumerate() {
            let cname = &rendo[sv as usize - 1];
            if cname.starts_with("AUX_") {
                continue;
            }
            let rc = dr.state_var.iter().position(|&e| endo[e] == *cname).unwrap();
            close(&[dr.ghx[(ri, rc)]], &[rghx[i * nsp + c]], 1e-9, &format!("{name}: ghx {vname} {cname}"));
        }
    }
    let nsv = dr.state_var.len();
    let irfs = r["irfs"].as_object().unwrap();
    for s in 0..nexo {
        let mut yprev = vec![0.0; n];
        for t in 0..8 {
            let mut yt = vec![0.0; n];
            for i in 0..n {
                let mut v = 0.0;
                for c in 0..nsv {
                    v += dr.ghx[(i, c)] * yprev[dr.state_var[c]];
                }
                if t == 0 {
                    v += dr.ghu[(i, s)] * mf.sigma_e[(s, s)].sqrt(); // one standard deviation, as stoch_simul
                }
                yt[dr.order_var[i]] = v;
            }
            for k in 0..orig {
                let key = format!("{}_{}", rendo[k], mf.model.exo_names[s]);
                let j = endo.iter().position(|e| *e == rendo[k]).unwrap();
                let want = flat(&irfs[&key])[t];
                close(&[yt[j]], &[want], 1e-9, &format!("{name}: irf {key} t={t}"));
            }
            yprev = yt;
        }
    }
}

#[test]
fn first_order_solution_var_pac() {
    solution_matches("pc3");
}

#[test]
fn first_order_solution_mce_pac() {
    solution_matches("pc4");
}

fn sim_data(r: &Value) -> PacData {
    let names = strings(&r["sim"]["names"]);
    let rows = r["sim"]["data"].as_array().unwrap();
    let mut cols = vec![Vec::new(); names.len()];
    for row in rows {
        for (j, v) in row.as_array().unwrap().iter().enumerate() {
            cols[j].push(v.as_f64().unwrap());
        }
    }
    PacData::new(names, cols)
}

fn range(r: &Value) -> (usize, usize) {
    let v = flat(&r["range_rows"]);
    (v[0] as usize, v[1] as usize)
}

/// The simulated data satisfies the PAC equation (with its residual) under the h parameters
/// computed here: checks the whole reduced form of the expectation.
fn equation_holds(name: &str, tol: f64) {
    let r = reference(name);
    let mf = run_commands(name);
    let data = sim_data(&r);
    let res = pac_residuals(&mf.varmodels, "zpac", &mf.model.param_names, &mf.model.params, &data, range(&r)).unwrap();
    let worst = res.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let at = res.iter().position(|v| v.abs() == worst).unwrap();
    let bad = res.iter().filter(|v| v.abs() > 1e-9).count();
    assert!(
        worst < tol && bad <= 5,
        "{name}: PAC equation residual {bad} periods above 1e-9, {worst} at {at}: {:?}",
        &res[..6.min(res.len())]
    );
}

#[test]
fn simulated_data_satisfies_pac_equation() {
    // pc1 has contemporaneous links between the variables: simul_backward_model solves each
    // period with a nonlinear solver whose tolerance leaves an error of a few 1e-6 in some periods
    equation_holds("pc1", 1e-5);
    equation_holds("pc2", 1e-9);
    equation_holds("pc3", 1e-9);
}

/// Initial values of dump.m (pc6 also estimates the share of optimizing agents).
fn guess(name: &str) -> Vec<(String, f64)> {
    let mut g = vec![("e_c_m".into(), 0.9), ("c_z_1".into(), 0.5), ("c_z_2".into(), 0.2)];
    if name == "pc6" {
        g.push(("lambda".into(), 0.5));
    }
    if name == "pc8" {
        g.extend([("c_z_s".into(), 0.9), ("c_z_dx2".into(), 0.1), ("lambda".into(), 0.7)]);
    }
    if name == "pc9" {
        g.push(("c_z_dx2".into(), 0.5));
    }
    g
}

fn without_residual(mut d: PacData) -> PacData {
    let k = d.names.iter().position(|n| n == "ez").unwrap();
    d.names.remove(k);
    d.cols.remove(k);
    d
}

fn nls_matches(name: &str, tol: f64) {
    let r = reference(name);
    let mf = run_commands(name);
    let data = without_residual(sim_data(&r));
    let mut params = mf.model.params.clone();
    let est = estimate_nls(&mf.varmodels, "zpac", &mf.model.param_names, &mut params, &guess(name), &data, range(&r))
        .unwrap();
    let m = &r["nls"];
    assert_eq!(est.parnames, strings(&m["parnames"]));
    close(&est.estimator, &flat(&m["estimator"]), tol, &format!("{name}: NLS estimates"));
    close(&[est.ssr], &[m["ssr"].as_f64().unwrap()], tol, &format!("{name}: ssr"));
    close(&[est.s2], &[m["s2"].as_f64().unwrap()], tol, &format!("{name}: s2"));
    close(&[est.r2], &[m["R2"].as_f64().unwrap()], tol, &format!("{name}: R2"));
    close(&est.residual, &flat(&m["residual"]), 1e-7, &format!("{name}: residuals"));
    let c = est.covariance.as_ref().unwrap();
    let rc = flat(&m["covariance"]);
    let cv: Vec<f64> =
        (0..c.nrows()).flat_map(|i| (0..c.ncols()).map(move |j| (i, j))).map(|(i, j)| c[(i, j)]).collect();
    close(&cv, &rc, 1e-4, &format!("{name}: covariance"));
    close(&est.student, &flat(&m["student"]), 1e-4, &format!("{name}: student"));
    // the h parameters are updated at the estimates
    close(&params, &flat(&m["params_after"]), 1e-6, &format!("{name}: M_.params after NLS"));
}

fn ols_matches(name: &str) {
    let r = reference(name);
    let mf = run_commands(name);
    let data = without_residual(sim_data(&r));
    let mut params = mf.model.params.clone();
    let est = estimate_iterative_ols(
        &mf.varmodels,
        "zpac",
        &mf.model.param_names,
        &mut params,
        &guess(name),
        &data,
        range(&r),
    )
    .unwrap();
    let m = &r["ols"];
    assert_eq!(est.parnames, strings(&m["parnames"]));
    close(&est.estimator, &flat(&m["estimator"]), 1e-9, &format!("{name}: OLS estimates"));
    close(&[est.ssr], &[m["ssr"].as_f64().unwrap()], 1e-9, &format!("{name}: OLS ssr"));
    close(&est.residual, &flat(&m["residual"]), 1e-8, &format!("{name}: OLS residuals"));
    close(&params, &flat(&m["params_after"]), 1e-8, &format!("{name}: M_.params after OLS"));
}

#[test]
fn nls_var_simple_target() {
    nls_matches("pc3", 1e-7);
}

#[test]
fn nls_trend_component() {
    nls_matches("pc2", 1e-7);
}

#[test]
fn nls_composite_target() {
    nls_matches("pc1", 1e-7);
}

#[test]
fn iterative_ols_var_and_trend_component() {
    ols_matches("pc3");
    ols_matches("pc2");
}

#[test]
fn share_of_optimizing_agents_parameters() {
    // growth neutrality correction divided by the share (+pac/+update/parameters.m,
    // +pac/+mce/parameters.m)
    for name in ["pc5", "pc7"] {
        let mf = run_commands(name);
        check_params(name, &mf, &reference(name));
    }
}

#[test]
fn share_of_optimizing_agents_estimation() {
    equation_holds("pc5", 1e-9);
    nls_matches("pc5", 1e-7);
    nls_matches("pc6", 1e-7);
    ols_matches("pc5");
    ols_matches("pc6");
}

/// Dynare 7.2 tests/pac/trend-component-22 and -25: estimated parameters of additive terms
/// outside (pc9, calibrated share) and inside and outside (pc8, estimated share) the
/// optimizing part (`M_.pac.<m>.additive` / `optim_additive` in iterative_ols.m).
#[test]
fn iterative_ols_additive_terms() {
    equation_holds("pc8", 1e-9);
    equation_holds("pc9", 1e-9);
    ols_matches("pc9");
    ols_matches("pc8");
}

#[test]
fn errors_like_dynare() {
    let bad = |src: &str| dynare::modfile::parse_mod_str(src, &[]).err().map(|e| format!("{e:#}")).unwrap_or_default();
    let head = "var y x z; varexo e; parameters a b c d beta; a=.5; b=.1; c=.2; d=.3; beta=.9;\nvar_model(model_name=m, eqtags=['e1']);\npac_model(auxiliary_model_name=m, discount=beta, model_name=p);\n";
    // no equation uses pac_expectation(p)
    let e = bad(&format!("{head}model;\n[name='e1']\ny = a*y(-1) + e;\nx = y;\nz = x;\nend;"));
    assert!(e.contains("pac_expectation"), "{e}");
    // unknown PAC model in an equation
    let e = bad("var y z; varexo e; parameters a; a=.5;\nmodel;\ny = a*y(-1) + pac_expectation(zz) + e;\nz = y;\nend;");
    assert!(e.contains("unknown PAC model"), "{e}");
}

/// The decimal numbers of `text` (`0.0145990564819670`) and the text with each one replaced by `#`.
fn split_numbers(text: &str) -> (String, Vec<f64>) {
    let c: Vec<char> = text.chars().collect();
    let (mut t, mut v, mut i) = (String::new(), Vec::new(), 0);
    while i < c.len() {
        let after_name = i > 0 && (c[i - 1].is_alphanumeric() || c[i - 1] == '_');
        if c[i].is_ascii_digit() && !after_name {
            let j = (i..c.len()).find(|&j| !(c[j].is_ascii_digit() || c[j] == '.')).unwrap_or(c.len());
            let s: String = c[i..j].iter().collect();
            if s.contains('.') {
                v.push(s.parse().unwrap());
                t.push('#');
            } else {
                t.push_str(&s);
            }
            i = j;
        } else {
            t.push(c[i]);
            i += 1;
        }
    }
    (t, v)
}

fn run_commands_in(dir: &Path, name: &str) -> ModFile {
    let mut mf = parse_mod_file(&dir.join(format!("{name}.mod")), &[]).unwrap();
    for ci in 0..mf.commands.len() {
        if let Command::Other(n, o) = mf.commands[ci].clone()
            && dynare::varmodels::handles(&n)
            && n != "pac.print"
        {
            dynare::varmodels::run(&mut mf, &n, &o).unwrap();
        }
    }
    mf
}

/// `pac.print`: the files written by print_expectations.m (reference/pac/print, from var-12 of
/// Dynare's tests), but for the dates. example4: growth in pac_target_info (Dynare numbers the
/// auxiliary variable of diff(x) after the nodes of the growth expressions, AUX_DIFF_11);
/// example3: a component log(x) and a growth diff(log(x(-2))) (unary-op auxiliary variables).
#[test]
fn pac_print_files() {
    let dir = refdir().join("print");
    for name in ["example11", "example4", "example3"] {
        let mf = run_commands_in(&dir, name);
        let p = mf.varmodels.pac("pacman").unwrap();
        let (files, console) = p.print(&mf.varmodels, &mf.model.param_names, &mf.model.params, name, true).unwrap();
        assert!(console.contains(&format!(
            "Parameters declarations and calibrations are saved in {name}/model/pac-expectations/pacman-parameters.inc."
        )));
        assert_eq!(files.len(), 4);
        for (f, text) in files {
            let base = Path::new(&f).file_name().unwrap().to_str().unwrap().to_string();
            let want = std::fs::read_to_string(dir.join(name).join(&base)).unwrap();
            let norm =
                |s: &str| s.lines().filter(|l| !l.contains("generated by Dynare")).collect::<Vec<_>>().join("\n");
            let (a, va) = split_numbers(&norm(&text));
            let (b, vb) = split_numbers(&norm(&want));
            assert_eq!(a, b, "{name}/{base}");
            close(&va, &vb, 1e-13, &format!("{name}/{base}"));
        }
    }
}

/// Equal, or both NaN.
fn close_nan(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x.is_nan() && y.is_nan()) || (x - y).abs() <= tol * (1.0 + y.abs()), "{what}[{i}]: {x} vs {y}");
    }
}

/// Row-major elements (MATLAB's jsonencode gives rows).
fn rows(x: &dynare::linalg::Mat) -> Vec<f64> {
    (0..x.nrows()).flat_map(|i| (0..x.ncols()).map(move |j| x[(i, j)])).collect()
}

/// Dynare's tests/trend-component-and-var-models/tcm6-8.mod (reference/pac/dump_tcm.m): the
/// `trend_component_model` and `pac_model` declarations follow the model block, and the PAC
/// equations have the left hand side `diff(log(x))`. Same auxiliary variables and parameters,
/// `M_.params` after `pac.initialize` (NaN h parameters, as in MATLAB), the companion matrices
/// and `trend_component_ar_a0` of both models.
#[test]
fn trend_component_declared_after_the_model_block() {
    for name in ["tcm6c", "tcm7c", "tcm8c"] {
        let r = reference(name);
        let mf = run_commands(name);
        assert_eq!(mf.model.endo_names.len(), r["endo_names"].as_array().unwrap().len(), "{name}: endo_nbr");
        let rn = strings(&r["param_names"]);
        assert_eq!(rn.len(), mf.model.param_names.len(), "{name}: number of parameters");
        for (i, n) in rn.iter().enumerate() {
            if !pac_param(n) {
                assert_eq!(&mf.model.param_names[i], n, "{name}: parameter name");
            }
        }
        close_nan(&mf.model.params, &flat(&r["params"]), 1e-10, &format!("{name}: M_.params"));
        for m in ["toto", "titi"] {
            let c = mf.varmodels.companion(m).unwrap();
            close(&rows(&c.matrix), &flat(&r["trend_component"][m]["CompanionMatrix"]), 1e-13, &format!("{name} {m}"));
            let ev =
                |a: &dynare::modfile::Ast| dynare::varmodels::eval_coef(a, &mf.model.param_names, &mf.model.params);
            let t = mf.varmodels.trend(m).unwrap().matrices(&ev).unwrap();
            let ra = &r["ar_a0"][m];
            assert_eq!(t.ar.len(), 1);
            close(&rows(&t.ar[0]), &flat(&ra["AR"]), 1e-15, &format!("{name} {m} AR"));
            close(&rows(&t.a0), &flat(&ra["A0"]), 1e-15, &format!("{name} {m} A0"));
            close(&rows(&t.a0star), &flat(&ra["A0star"]), 1e-15, &format!("{name} {m} A0star"));
        }
    }
}
