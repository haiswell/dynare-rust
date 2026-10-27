//! Optimal policy (stream P) against Dynare 7.2: reference/policy/ref_<model>.json, written by
//! reference/policy/dump_policy.m.
mod common;

use dynare::linalg::Mat;
use dynare::modfile::{ModFile, parse_mod_file};
use dynare::policy::{Session, handles};
use serde_json::Value;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("reference/policy")
}

fn reference(name: &str) -> Option<Value> {
    Some(common::json_ref(root().join(format!("ref_{name}.json"))))
}

fn nums(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(nums).collect(),
        Value::Number(x) => vec![x.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

/// A MATLAB matrix from jsonencode (rows nested; a column/row vector or scalar is flat).
fn mat(v: &Value, nr: usize, nc: usize) -> Mat {
    let rows: Vec<Vec<f64>> = match v {
        Value::Array(a) if a.first().is_some_and(|x| x.is_array()) => a.iter().map(nums).collect(),
        _ => {
            let f = nums(v);
            if nc == 1 { f.iter().map(|&x| vec![x]).collect() } else { vec![f] }
        }
    };
    assert_eq!((rows.len(), rows.first().map_or(0, |r| r.len())), (nr, nc), "matrix shape");
    Mat::from_fn(nr, nc, |i, j| rows[i][j])
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

fn close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{what}[{k}]: {x} vs {y} (diff {:e})", (x - y).abs());
    }
}

fn run(name: &str) -> Option<(ModFile, Session, Value)> {
    let r = reference(name)?;
    let mf = parse_mod_file(&root().join(format!("{name}.mod")), &[]).unwrap();
    let mut s = Session::new(true);
    for c in &mf.commands {
        if handles(&mf, c) {
            s.run(&mf, c, None).unwrap();
        }
    }
    Some((mf, s, r))
}

/// Structure (names, incidence, parameters), steady state, Jacobians and decision rule.
fn check_common(mf: &ModFile, s: &Session, r: &Value, tol: f64) {
    let m = &mf.model;
    // lead auxiliaries are named after Dynare's internal expression index (AUX_ENDO_LEAD_<k>)
    let names = strings(&r["endo_names"]);
    assert_eq!(m.endo_names.len(), names.len());
    for (a, b) in m.endo_names.iter().zip(&names) {
        if b.starts_with("AUX_ENDO_LEAD_") {
            assert!(a.starts_with("AUX_ENDO_LEAD_"), "{a} vs {b}");
        } else {
            assert_eq!(a, b);
        }
    }
    assert_eq!(m.param_names, strings(&r["param_names"]));
    let n = m.n_endo();
    let nx = m.n_exo();
    let lli = mat(&r["lead_lag_incidence"], 3, n);
    for j in 0..n {
        for t in 0..3 {
            assert_eq!(m.lead_lag_incidence[j][t] as f64, lli[(t, j)], "lead_lag_incidence of {}", m.endo_names[j]);
        }
    }
    assert_eq!(m.n_equations(), r["eq_nbr"].as_u64().unwrap() as usize);
    let p = s.params.clone().unwrap();
    close(&p, &nums(&r["params"]), tol, "params");
    let dr = s.dr.as_ref().unwrap();
    close(&dr.ys, &nums(&r["ys"]), tol, "steady state");
    let ov: Vec<usize> = nums(&r["order_var"]).iter().map(|&x| x as usize - 1).collect();
    assert_eq!(dr.order_var, ov);
    if !r["state_var"].is_null() {
        let sv: Vec<usize> = nums(&r["state_var"]).iter().map(|&x| x as usize - 1).collect();
        assert_eq!(dr.state_var, sv);
    }
    let ghx = mat(&r["ghx"], n, dr.ghx.ncols());
    let ghu = mat(&r["ghu"], n, nx);
    close(dr.ghx.as_slice(), ghx.as_slice(), tol, "ghx");
    close(dr.ghu.as_slice(), ghu.as_slice(), tol, "ghu");
    // static residuals and dynamic Jacobian at the steady state (the Ramsey FOCs)
    let x = vec![0.0; nx];
    let mut res = vec![0.0; m.n_equations()];
    m.static_residuals(&dr.ys, &x, &p, &mut res);
    close(&res, &nums(&r["static_resid"]), tol, "static residuals");
    let g1 = m.dynamic_jacobian(&dr.ys, &x, &p);
    let g1r = mat(&r["dynamic_g1"], m.n_equations(), 3 * n + nx);
    close(g1.as_slice(), g1r.as_slice(), tol, "dynamic g1");
    // theoretical variances of the var_list
    let var = s.var.as_ref().unwrap();
    let k = s.var_list.len();
    assert_eq!(strings(&r["var_list"]), s.var_list.iter().map(|&j| m.endo_names[j].clone()).collect::<Vec<_>>());
    close(var.as_slice(), mat(&r["var"], k, k).as_slice(), tol, "variances");
}

#[test]
fn ramsey_nonlinear_cmr() {
    let (mf, s, r) = run("ramsey_cmr").expect("reference/setup unavailable");
    check_common(&mf, &s, &r, 1e-8);
    let v = s.planner_objective_value.as_ref().unwrap();
    let pr = &r["planner_objective_value"];
    close(&[v.unconditional], &[pr["unconditional"].as_f64().unwrap()], 1e-10, "unconditional welfare");
    close(
        &[v.conditional_zero_initial_multiplier, v.conditional_steady_initial_multiplier],
        &[
            pr["conditional"]["zero_initial_multiplier"].as_f64().unwrap(),
            pr["conditional"]["steady_initial_multiplier"].as_f64().unwrap(),
        ],
        1e-10,
        "conditional welfare",
    );
    // the multipliers are the type-6 aux variables MULT_1..MULT_11, one per original equation
    let ri = mf.model.planner().unwrap().ramsey.as_ref().unwrap();
    assert_eq!((ri.orig_endo_nbr, ri.orig_eq_nbr), (12, 11));
    assert_eq!(ri.mult_index, (12..23).collect::<Vec<_>>());
    assert_eq!(mf.model.equation_tags[0], vec![("name".to_string(), "Ramsey FOC w.r.t. C".to_string())]);
}

#[test]
fn ramsey_linear_quadratic_gali() {
    let (mf, s, r) = run("gali_ramsey").expect("reference/setup unavailable");
    check_common(&mf, &s, &r, 1e-10);
    let v = s.planner_objective_value.as_ref().unwrap();
    assert!(v.unconditional.abs() < 1e-14 && v.conditional_zero_initial_multiplier.abs() < 1e-14);
}

#[test]
fn discretionary_policy_gali() {
    let (mf, s, r) = run("gali_discretion").expect("reference/setup unavailable");
    check_common(&mf, &s, &r, 1e-10);
    let v = s.planner_objective_value.as_ref().unwrap();
    let pr = &r["planner_objective_value"];
    close(&[v.unconditional], &[pr["unconditional"].as_f64().unwrap()], 1e-10, "unconditional welfare");
    close(
        &[v.conditional_steady_initial_multiplier],
        &[pr["conditional"].as_f64().unwrap()],
        1e-10,
        "conditional welfare",
    );
}

#[test]
fn osr_gali() {
    let (mf, s, r) = run("gali_osr").expect("reference/setup unavailable");
    check_common(&mf, &s, &r, 1e-8);
    let o = s.osr.as_ref().unwrap();
    close(&[o.objective_function], &[r["osr_objective"].as_f64().unwrap()], 1e-8, "OSR loss");
    for (name, v) in &o.optim_params {
        close(&[*v], &[r["osr_params"][name].as_f64().unwrap()], 1e-8, name);
    }
    assert_eq!(o.func_count as f64, r["osr_info"]["funcCount"].as_f64().unwrap());
    assert_eq!(o.iterations as f64, r["osr_info"]["iterations"].as_f64().unwrap());
}

#[test]
fn ramsey_two_instruments_and_aux_multipliers() {
    let (mf, s, r) = run("ramsey_cmr2").expect("reference/setup unavailable");
    check_common(&mf, &s, &r, 1e-8);
    let v = s.planner_objective_value.as_ref().unwrap();
    let pr = &r["planner_objective_value"];
    close(&[v.unconditional], &[pr["unconditional"].as_f64().unwrap()], 1e-10, "unconditional welfare");
    close(
        &[v.conditional_zero_initial_multiplier, v.conditional_steady_initial_multiplier],
        &[
            pr["conditional"]["zero_initial_multiplier"].as_f64().unwrap(),
            pr["conditional"]["steady_initial_multiplier"].as_f64().unwrap(),
        ],
        1e-10,
        "conditional welfare",
    );
    // MULT_13(+1) and MULT_12(-1) get aux variables, like pi(+1) inside log(pi(+2)) and Z(-1)
    let aux: Vec<&str> = mf.model.aux_vars.iter().map(|a| a.orig_expr.as_str()).collect();
    assert_eq!(aux, ["MULT_13(1)", "log(pi(1))", "MULT_12(-1)", "Z(-1)"]);
}

#[test]
fn osr_simplex_gali() {
    let (mf, s, r) = run("gali_osr_simplex").expect("reference/setup unavailable");
    check_common(&mf, &s, &r, 1e-8);
    let o = s.osr.as_ref().unwrap();
    close(&[o.objective_function], &[r["osr_objective"].as_f64().unwrap()], 1e-10, "OSR loss");
    for (name, v) in &o.optim_params {
        close(&[*v], &[r["osr_params"][name].as_f64().unwrap()], 1e-8, name);
    }
    assert_eq!(o.func_count as f64, r["osr_info"]["funcCount"].as_f64().unwrap());
}

const SMALL: &str = r#"
var pi y i a;
varexo e;
parameters beta kappa rho phi;
beta = 0.99; kappa = 0.1; rho = 0.9; phi = 1.5;
model(linear);
pi = beta*pi(+1) + kappa*y + a;
y = y(+1) - (i - pi(+1));
a = rho*a(-1) + e;
i = phi*pi;
end;
shocks; var e; stderr 0.01; end;
optim_weights;
pi 1;
y 0.5*kappa;
pi, y 0.1;
end;
osr_params phi;
osr_params_bounds;
phi, 1, 2*beta;
end;
osr(order=1) pi;
"#;

#[test]
fn parses_policy_declarations() {
    let mf = dynare::modfile::parse_mod_str(SMALL, &[]).unwrap();
    let p = &mf.policy;
    assert_eq!(p.optim_weights, vec![(0, 0, 1.0), (1, 1, 0.05), (0, 1, 0.1)]);
    assert_eq!(p.osr_params, vec![3]);
    assert_eq!(p.osr_bounds, vec![(3, 1.0, 1.98)]);
    // optimal_policy_discount_factor is appended (1 when planner_discount is not given)
    assert_eq!(mf.model.param_names.last().unwrap(), "optimal_policy_discount_factor");
    assert_eq!(*mf.model.params.last().unwrap(), 1.0);
    // bounds need a constrained optimizer, as in Dynare
    let mut s = Session::new(true);
    let c = mf.commands.iter().find(|c| matches!(c, dynare::modfile::Command::Policy(n, ..) if n == "osr")).unwrap();
    let e = s.run(&mf, c, None).unwrap_err();
    assert!(format!("{e}").contains("constrained optimizer"), "{e}");
}

#[test]
fn ramsey_structure_matches_preprocessor() {
    // reference: Dynare 7.1's preprocessor on the same model (MULT_1..3, aux for MULT_1(+2)
    // -> AUX_ENDO_LEAD of MULT_1(1), pi(+2), then lags of MULT_1(-2) and a(-2))
    let src = "var pi y i a;\nvarexo e;\nparameters beta kappa rho;\nbeta=0.99; kappa=0.1; rho=0.9;\n\
               model;\npi = beta*pi(+2) + kappa*y + a(-2);\ny = y(+1) - (i - pi(+1));\na = rho*a(-1) + e;\nend;\n\
               planner_objective -(pi^2 + y^2 + exp(i));\nramsey_model(instruments=(i), planner_discount=beta);\nstoch_simul(order=1);\n";
    let mf = dynare::modfile::parse_mod_str(src, &[]).unwrap();
    let m = &mf.model;
    let names: Vec<&str> = m.endo_names.iter().map(String::as_str).collect();
    assert_eq!(names[..7], ["pi", "y", "i", "a", "MULT_1", "MULT_2", "MULT_3"]);
    assert!(names[7].starts_with("AUX_ENDO_LEAD_") && names[8].starts_with("AUX_ENDO_LEAD_"));
    assert_eq!(names[9..], ["AUX_ENDO_LAG_9_1", "AUX_ENDO_LAG_3_1"]);
    let aux: Vec<&str> = m.aux_vars.iter().map(|a| a.orig_expr.as_str()).collect();
    assert_eq!(aux, ["MULT_1(1)", "pi(1)", "MULT_1(-1)", "a(-1)"]);
    // M_.lead_lag_incidence' of the preprocessor
    let lli: [[usize; 3]; 11] = [
        [0, 6, 17],
        [0, 7, 18],
        [0, 8, 0],
        [1, 9, 0],
        [2, 10, 19],
        [3, 11, 0],
        [0, 12, 20],
        [0, 13, 21],
        [0, 14, 22],
        [4, 15, 0],
        [5, 16, 0],
    ];
    assert_eq!(m.lead_lag_incidence, lli);
    assert_eq!(m.param_names.last().unwrap(), "optimal_policy_discount_factor");
    // ramsey_multipliers_static_g1 (preprocessor values at any y: linear in the multipliers)
    let p = &m.params;
    let y = vec![0.0; m.n_endo()];
    let a = m.ramsey_multipliers_static_g1(&y, &[0.0], p).unwrap();
    let b = 0.99f64;
    assert!((a[(0, 0)] - (1.0 + b.powi(-2) * (-b))).abs() < 1e-15);
    assert!((a[(3, 0)] + b * b).abs() < 1e-15);
    assert!((a[(1, 1)] - (1.0 - 1.0 / b)).abs() < 1e-15);
    assert!((a[(3, 2)] - (1.0 - b * 0.9)).abs() < 1e-15);
}

// ---- second order, steady_state_model, constraints, bounded OSR ----

/// Second-order decision rule (`ghxx`, `ghxu`, `ghuu`, `ghs2`) against the reference.
fn check_order2(s: &Session, r: &Value, tol: f64) {
    let k = s.kdr.as_ref().expect("second-order decision rule");
    let n = k.first.ghx.nrows();
    for (name, m) in [("ghxx", &k.ghxx), ("ghxu", &k.ghxu), ("ghuu", &k.ghuu), ("ghs2", &k.ghs2)] {
        let rm = mat(&r[name], n, m.ncols());
        close(m.as_slice(), rm.as_slice(), tol, name);
    }
}

fn check_ramsey_welfare(s: &Session, r: &Value, tol: f64) {
    let v = s.planner_objective_value.as_ref().unwrap();
    let pr = &r["planner_objective_value"];
    close(&[v.unconditional], &[pr["unconditional"].as_f64().unwrap()], tol, "unconditional welfare");
    close(
        &[v.conditional_zero_initial_multiplier, v.conditional_steady_initial_multiplier],
        &[
            pr["conditional"]["zero_initial_multiplier"].as_f64().unwrap(),
            pr["conditional"]["steady_initial_multiplier"].as_f64().unwrap(),
        ],
        tol,
        "conditional welfare",
    );
}

/// Ramsey steady state from a steady_state_model conditional on the instrument (csolve),
/// second-order decision rules and welfare.
#[test]
fn ramsey_steady_state_model_order2() {
    let (mf, s, r) = run("ramsey_cmr_ssm").expect("reference/setup unavailable");
    assert_eq!(s.order, 2);
    check_common(&mf, &s, &r, 1e-9);
    check_order2(&s, &r, 1e-9);
    check_ramsey_welfare(&s, &r, 1e-11);
}

/// `ramsey_policy(order=2, planner_discount=0.98)`: numerical Ramsey steady state, second order.
#[test]
fn ramsey_policy_order2_planner_discount() {
    let (mf, s, r) = run("ramsey_cmr_o2").expect("reference/setup unavailable");
    let m = &mf.model;
    let p = s.params.as_ref().unwrap();
    assert_eq!(m.planner_discount(p), Some(0.98));
    check_common(&mf, &s, &r, 1e-9);
    // two numerically zero but nonzero roots (3e-8, 4e-8) enter the state space, which makes
    // ghxx in the lagged-multiplier directions sensitive to rounding (1.5e-9 relative)
    check_order2(&s, &r, 2e-9);
    check_ramsey_welfare(&s, &r, 1e-11);
}

/// OSR maximizing the second-order welfare within osr_params_bounds (both bounds bind).
#[test]
fn osr_order2_welfare_with_bounds() {
    let (mf, s, r) = run("cmr_osr_o2").expect("reference/setup unavailable");
    check_common(&mf, &s, &r, 1e-9);
    check_order2(&s, &r, 1e-9);
    let o = s.osr.as_ref().unwrap();
    for (name, v) in &o.optim_params {
        close(&[*v], &[r["osr_params"][name].as_f64().unwrap()], 1e-12, name);
    }
    close(&[o.objective_function], &[r["osr_objective"].as_f64().unwrap()], 1e-11, "OSR loss");
    close(
        &[o.objective_function],
        &[-r["osr_welfare_opt"]["unconditional"].as_f64().unwrap()],
        1e-11,
        "welfare at the optimum",
    );
    // second-order welfare at the initial (calibrated) rule
    let res = dynare::policy::Resolver::new(&mf, false);
    let mut p = mf.model.params.clone();
    let names = ["phi_pi", "phi_y"];
    for (k, n) in names.iter().enumerate() {
        let i = mf.model.param_names.iter().position(|x| x == n).unwrap();
        p[i] = nums(&r["osr_init_params"])[k];
    }
    let d = mf.model.dynamic_derivatives(2);
    let (kdr, p) = res.resolve_k(&p, 2, &d, &mf.sigma_e).unwrap();
    let n = kdr.first.ghx.nrows();
    close(
        kdr.ghxx.as_slice(),
        mat(&r["osr_init_ghxx"], n, kdr.ghxx.ncols()).as_slice(),
        1e-9,
        "ghxx at the calibration",
    );
    let v = dynare::policy::welfare::evaluate_planner_objective_order2(&mf.model, &p, &kdr, &mf.sigma_e, 1.0 + 1e-6)
        .unwrap();
    let w = &r["osr_welfare_init"];
    close(
        &[v.unconditional],
        &[w["unconditional"].as_f64().unwrap()],
        1e-11,
        "unconditional welfare at the calibration",
    );
    close(
        &[v.conditional_steady_initial_multiplier],
        &[w["conditional"].as_f64().unwrap()],
        1e-11,
        "conditional welfare at the calibration",
    );
}

/// discretionary_policy with discretionary_tol, maxit and a planner discount of its own.
#[test]
fn discretionary_policy_tol_and_planner_discount() {
    let (mf, s, r) = run("gali_discretion2").expect("reference/setup unavailable");
    assert_eq!(mf.model.planner_discount(s.params.as_ref().unwrap()), Some(0.97));
    check_common(&mf, &s, &r, 1e-10);
    let v = s.planner_objective_value.as_ref().unwrap();
    let pr = &r["planner_objective_value"];
    close(&[v.unconditional], &[pr["unconditional"].as_f64().unwrap()], 1e-10, "unconditional welfare");
    close(
        &[v.conditional_steady_initial_multiplier],
        &[pr["conditional"].as_f64().unwrap()],
        1e-10,
        "conditional welfare",
    );
}

/// `ramsey_constraints` have no effect on the static problem in Dynare 7.2 (`evaluate_steady_state.m`
/// takes the Ramsey branch, `dyn_ramsey_static.m` never reads `static_complementarity_conditions`):
/// the steady state is the reference one whether the constraint is slack or violated.
#[test]
fn ramsey_constraints_in_the_static_problem() {
    let r = reference("ramsey_cmr_ssm").expect("reference/setup unavailable");
    let src = std::fs::read_to_string(root().join("ramsey_cmr_ssm.mod")).unwrap();
    let dir = std::env::temp_dir().join(format!("dynare_rust_rc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (tag, bound) in [("loose", "R > 0"), ("upper", "R < 0.05"), ("violated", "R > 0.02")] {
        let text = src.replace("ramsey_model(", &format!("ramsey_constraints;\n{bound};\nend;\nramsey_model("));
        let path = dir.join(format!("rc_{tag}.mod"));
        std::fs::write(&path, text).unwrap();
        let mf = parse_mod_file(&path, &[]).unwrap();
        assert_eq!(mf.policy.ramsey_constraints.len(), 1);
        let mut s = Session::new(true);
        let mut err = None;
        for c in &mf.commands {
            if handles(&mf, c) {
                if let Err(e) = s.run(&mf, c, None) {
                    err = Some(e);
                    break;
                }
            }
        }
        assert!(err.is_none(), "{tag}: {err:?}");
        close(&s.dr.as_ref().unwrap().ys, &nums(&r["ys"]), 1e-9, "steady state");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn csolve_finds_a_root() {
    let f = |x: &[f64]| vec![x[0] * x[0] - 2.0, x[1] - x[0]];
    let (x, rc) = dynare::policy::ramsey::csolve(&f, &[1.0, 0.0], 1e-12, 100);
    assert_eq!(rc, 0);
    assert!((x[0] - 2f64.sqrt()).abs() < 1e-10 && (x[1] - 2f64.sqrt()).abs() < 1e-10);
}

/// `ramsey_model` + `stoch_simul(order=3)` + `evaluate_planner_objective(periods=300, drop=50)`:
/// the k_order_welfare tensors, the k_order_mean simulation (first period skipped, divided by
/// periods - drop) and the welfare with MATLAB's shocks (reference/policy/dump_ramsey_o3.m).
#[test]
fn ramsey_order3_simulated_welfare() {
    let name = "ramsey_cmr_o3";
    let r = reference(name).expect("reference unavailable");
    let mf = parse_mod_file(&root().join(format!("{name}.mod")), &[]).unwrap();
    let ne = mf.model.n_exo();
    let periods = r["ramsey_periods"].as_u64().unwrap() as usize;
    assert_eq!((periods, r["ramsey_drop"].as_u64().unwrap()), (300, 50));
    let mut s = Session::new(true);
    s.planner_shocks = Some(mat(&r["exo_simul"], ne, periods));
    for c in &mf.commands {
        if handles(&mf, c) {
            s.run(&mf, c, None).unwrap();
        }
    }
    let kdr = s.kdr.as_ref().unwrap();
    assert_eq!(kdr.order, 3);
    let nw = kdr.first.nspred + ne;
    let p = s.params.clone().unwrap();
    let m = &mf.model;
    let ys = &kdr.first.ys;
    let x0 = vec![0.0; ne];
    let obj = m.planner_objective(ys, &x0, &p).unwrap();
    let u3 = m.planner_objective_g3(ys, &x0, &p).unwrap();
    let beta = m.planner_discount(&p).unwrap();
    let w = dynare::policy::welfare::k_order_welfare(kdr, &obj.uy, &obj.uyy, &u3, beta, &mf.sigma_e).unwrap();
    let n = kdr.first.ghx.nrows();
    let mut g = Vec::new();
    for j in 0..=3 {
        let gj = kdr.g_k(j);
        close(gj.as_slice(), mat(&r[format!("g_{j}")], n, gj.ncols()).as_slice(), 1e-7, &format!("g_{j}"));
        let wj = dynare::solver::korder::fold_taylor(&w, 3, nw, j);
        let want = nums(&r[format!("W_{j}")]);
        // absolute tolerance relative to the largest coefficient (W_0 is ~1e-10)
        let scale = want.iter().fold(1.0f64, |a, &b| a.max(b.abs()));
        for (k, (x, y)) in wj.as_slice().iter().zip(&want).enumerate() {
            assert!((x - y).abs() <= 1e-7 * scale, "W_{j}[{k}]: {x} vs {y}");
        }
        let mut gw = Mat::zeros(n + 1, gj.ncols());
        gw.rows_mut(0, n).copy_from(&gj);
        gw.set_row(n, &wj.row(0));
        g.push(gw);
    }
    // k_order_mean on a short path from a perturbed state, drop = 0..3
    let ov = &kdr.first.order_var;
    let mut ysteady: Vec<f64> = ov.iter().map(|&j| ys[j]).collect();
    ysteady.push(obj.u / (1.0 - beta));
    let ex = mat(&r["probe_ex"], ne, 4);
    let start = nums(&r["probe_ystart"]);
    for d in 0..4 {
        let (mean, sim) = dynare::solver::korder::simulate::k_order_mean(
            &g,
            kdr.first.nstatic,
            kdr.first.nspred,
            &ysteady,
            &start,
            &ex,
            d,
        );
        close(&mean, &nums(&r[format!("probe_mean_drop{d}")]), 1e-9, &format!("k_order_mean drop={d}"));
        if d == 3 {
            close(sim.as_slice(), mat(&r["probe_sim"], n + 1, 4).as_slice(), 1e-9, "k_order_mean simulation");
        }
    }
    let v = s.planner_objective_value.as_ref().unwrap();
    let pv = &r["planner_objective_value"];
    close(&[v.unconditional], &[pv["unconditional"].as_f64().unwrap()], 1e-9, "unconditional welfare");
    close(
        &[v.conditional_steady_initial_multiplier],
        &[pv["conditional"]["steady_initial_multiplier"].as_f64().unwrap()],
        1e-9,
        "conditional welfare (steady multipliers)",
    );
    close(
        &[v.conditional_zero_initial_multiplier],
        &[pv["conditional"]["zero_initial_multiplier"].as_f64().unwrap()],
        1e-9,
        "conditional welfare (zero multipliers)",
    );
}
