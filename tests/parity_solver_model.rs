//! Parity fixes in the model and solver layer (solver-model group): derivatives of
//! `STEADY_STATE()` with respect to parameters, `getPowerDeriv`, the pruned state space's
//! unit-root threshold `2-qz_criterium`.
mod common;

use dynare::dsge::DsgeModel;
use dynare::identification::derivs::ModelParamDerivs;
use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use dynare::solver::korder::{self, pruned};

fn dsge_from(tag: &str, src: &str) -> DsgeModel {
    let dir = common::temp_dir(tag);
    let p = dir.join(format!("{tag}.mod"));
    std::fs::write(&p, src).unwrap();
    let mf = parse_mod_file(&p, &[]).unwrap();
    DsgeModel::new(&mf, Some(&[])).unwrap()
}

const SS_MODEL: &str = "
var y c;
varexo e;
parameters a b;
a = 1; b = 0.5;
model;
y = a + b*y(-1) + e;
c = STEADY_STATE(y)*y + STEADY_STATE(y)^2*y(-1)^2;
end;
steady_state_model;
y = a/(1-b);
c = y*y + y^2*y^2;
end;
shocks; var e; stderr 1; end;
";

/// Steady state of `SS_MODEL` at `(a, b)`.
fn ss_ys(a: f64, b: f64) -> Vec<f64> {
    let y = a / (1.0 - b);
    vec![y, y * y + y.powi(4)]
}

/// d STEADY_STATE(y)/d param (Dynare's `steadyStateParamDeriv` fed with `dys`): the total
/// derivative of the dynamic Jacobian with respect to the parameters equals the finite
/// difference of the Jacobian evaluated at the re-solved steady state.
#[test]
fn steady_state_parameter_derivatives_match_finite_differences() {
    let dsge = dsge_from("sm_ss_deriv", SS_MODEL);
    let m = &dsge.model;
    let x0 = vec![0.0; m.n_exo()];
    let p0 = vec![1.0, 0.5];
    let ys0 = ss_ys(p0[0], p0[1]);
    let pd = ModelParamDerivs::new(m);
    let md = pd.eval(m, &ys0, &p0, 1.0 + 1e-6).unwrap();
    // implicit function theorem: dy/da = 1/(1-b), dy/db = a/(1-b)^2
    assert!((md.dys[(0, 0)] - 2.0).abs() < 1e-12 && (md.dys[(0, 1)] - 4.0).abs() < 1e-12, "{}", md.dys);
    let h = 1e-6;
    for q in 0..2 {
        let g = |s: f64| {
            let mut p = p0.clone();
            p[q] += s;
            m.dynamic_jacobian(&ss_ys(p[0], p[1]), &x0, &p)
        };
        let fd: Mat = (g(h) - g(-h)) / (2.0 * h);
        let err = (&md.dg1[q] - &fd).amax();
        assert!(fd.amax() > 1.0, "the probe must have a steady-state channel");
        assert!(err < 1e-6 * fd.amax(), "param {q}: dg1\n{}\nvs finite differences\n{fd}", md.dg1[q]);
    }
}

fn ghx_row(dr: &korder::KOrderDr, var: usize) -> f64 {
    dr.first.ghx[(dr.first.inv_order_var[var], 0)]
}

const POW_MODEL: &str = "
var x y;
varexo e;
parameters ex;
ex = EXPONENT;
model;
x = 0.5*x(-1) + e;
y = x^ex + 0.1*y(+1);
end;
steady_state_model;
x = 0;
y = STEADY;
end;
shocks; var e; stderr 1; end;
";

fn solve_pow(tag: &str, exponent: &str, steady: &str, order: usize) -> korder::KOrderDr {
    let src = POW_MODEL.replace("EXPONENT", exponent).replace("STEADY", steady);
    let dsge = dsge_from(tag, &src);
    let m = &dsge.model;
    korder::solve_model(m, &dsge.ys_calib, &m.params, &dsge.base_sigma_e, order, Default::default()).unwrap()
}

/// getPowerDeriv.m: `x^p` with a parameter exponent differentiates to exactly 0 when
/// `|x| < 1e-12`, `p >= 0` is an integer and `k > p`. With `p = 1` the second derivative at
/// `x = 0` is 0 (the symbolic `p*(p-1)*x^(p-2)` is `0*Inf = NaN`: info 12 before the fix).
#[test]
fn power_derivative_with_parameter_exponent_order_2() {
    let dr = solve_pow("sm_pow1", "1", "0", 2);
    // y_t = G(x_t), G(x) = x + 0.1 E G(0.5 x + e): G' = 1/(1-0.05), no curvature; the state
    // is x(-1): ghx = 0.5 G'
    assert!((ghx_row(&dr, 1) - 0.5 / 0.95).abs() < 1e-12, "{}", ghx_row(&dr, 1));
    assert!(dr.ghxx.iter().all(|v| v.abs() < 1e-12), "{}", dr.ghxx);
    // p = 2: d2/dx2 x^2 = 2 (k = p is not > p): G'' = 2/(1 - 0.1*0.25), ghxx = 0.25 G''
    let dr = solve_pow("sm_pow2", "2", "0", 2);
    let row = dr.first.inv_order_var[1];
    assert!(ghx_row(&dr, 1).abs() < 1e-12);
    let col = dr.ghxx.ncols() - 1; // the only state pair
    assert!((dr.ghxx[(row, col)] - 0.5 / 0.975).abs() < 1e-10, "{}", dr.ghxx);
}

/// `p = 0`: `x^0 = 1`, and its derivatives at 0 are exactly 0 (`k > p = 0`; `0*0^(-1)` is NaN
/// symbolically).
#[test]
fn power_derivative_with_zero_exponent() {
    let dr = solve_pow("sm_pow0", "0", "1/0.9", 2);
    assert!(dr.ghxx.iter().all(|v| *v == 0.0), "{}", dr.ghxx);
    assert!(ghx_row(&dr, 1).abs() < 1e-14, "{}", ghx_row(&dr, 1));
    assert!((dr.first.ys[1] - 1.0 / 0.9).abs() < 1e-12);
}

const TWO_STATE: &str = "
var a b c;
varexo e1 e2;
parameters rho;
rho = 0.8;
model;
a = rho*a(-1) + e1;
b = 0.3*b(-1) + e2 + 0.1*a^2;
c = 0.5*c(+1) + 0.1*a;
end;
steady_state_model;
a = 0;
b = 0;
c = 0;
end;
shocks; var e1; stderr 1; var e2; stderr 1; end;
";

/// pruned_state_space_system.m: a first-order transition with `|eig| > 2-qz_criterium` is a
/// unit root, an error at higher order ('does not support unit roots'). Default
/// `qz_criterium = 1+1e-6` gives the threshold `1-1e-6`; the eigenvalues here are 0.8 and 0.3.
#[test]
fn pruned_unit_root_threshold_is_two_minus_qz_criterium() {
    let src = TWO_STATE.to_string();
    let dsge = dsge_from("sm_pruned_qz", &src);
    let m = &dsge.model;
    let dr = korder::solve_model(m, &dsge.ys_calib, &m.params, &dsge.base_sigma_e, 2, Default::default()).unwrap();
    let all: Vec<usize> = (0..3).collect();
    let ok = |qz: f64| pruned::pruned_state_space_system_qz(&dr, &dsge.base_sigma_e, &all, 1, qz).is_ok();
    assert!(ok(1.0 + 1e-6), "0.8 is stable at the default threshold 1-1e-6");
    assert!(ok(1.2 - 1e-3), "threshold 0.801 > 0.8");
    assert!(!ok(1.2 + 1e-3), "threshold 0.799 < 0.8: unit root error");
    assert!(!ok(1.7 - 1e-3), "threshold 0.301 < 0.8");
    assert!(pruned::pruned_state_space_system(&dr, &dsge.base_sigma_e, &all, 1).is_ok());
}

/// `analytic_derivation` through `STEADY_STATE()`: the score (first-order `dys`) and the full
/// Hessian (second-order `d2ys`, `ss_param_2nd_deriv`) of a model whose coefficient is
/// `0.3*STEADY_STATE(x)` with `x`'s steady state `2a`, against finite differences of the
/// posterior kernel and of the score.
#[test]
fn analytic_derivatives_through_steady_state_match_finite_differences() {
    use dynare::data::Table;
    use dynare::dsge::{DsgePosterior, LikelihoodData};
    use dynare::estimation::Objective;
    use dynare::estimation::analytic::{AnalyticContext, AnalyticHessian, AnalyticOptions, AnalyticPosterior};
    use dynare::kalman::KalmanOptions;

    let dir = common::temp_dir("sm_ss_est");
    let mut state = 987654321u64;
    let mut noise = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let mut csv = String::from("x,z\n");
    for _ in 0..60 {
        csv += &format!("{},{}\n", 2.0 + noise(), 3.0 + noise());
    }
    std::fs::write(dir.join("d.csv"), csv).unwrap();
    let p = dir.join("ss_est.mod");
    std::fs::write(
        &p,
        "var x z;
varexo e1 e2;
parameters a rho;
a = 1; rho = 0.6;
model;
x = a + 0.5*x(-1) + e1;
z = rho*z(-1) + 0.3*STEADY_STATE(x)*x + e2;
end;
steady_state_model;
x = 2*a;
z = 0.3*x*x/(1-rho);
end;
shocks; var e1; stderr 0.3; var e2; stderr 0.2; end;
varobs x z;
estimated_params;
a, 1, normal_pdf, 1, 0.5;
rho, 0.6, beta_pdf, 0.6, 0.1;
end;
",
    )
    .unwrap();
    let mf = parse_mod_file(&p, &[]).unwrap();
    let dsge = DsgeModel::new(&mf, None).unwrap();
    let table = Table::read_csv(&dir.join("d.csv")).unwrap();
    let data = LikelihoodData::new(&table.sample(&mf.varobs, 1, Some(60)).unwrap(), KalmanOptions::default());
    let post = DsgePosterior { dsge: &dsge, data: &data, bounds: &dsge.bounds };
    let ctx = AnalyticContext::new(&post, AnalyticOptions { mode: 0 }).unwrap();
    let ap = AnalyticPosterior { post: &post, ctx: &ctx };
    let x0 = [1.3, 0.55];
    let n = 2;
    let full = ap.eval_grad(&x0, AnalyticHessian::Full).unwrap();
    assert_eq!(full.info, 0);
    // the score against central differences of the objective
    let h = 1e-5;
    for k in 0..n {
        let f = |s: f64| {
            let mut x = x0;
            x[k] += s;
            post.eval(&x).0
        };
        let fd = (f(h) - f(-h)) / (2.0 * h);
        assert!((full.grad[k] - fd).abs() < 1e-5 * fd.abs().max(1.0), "score {k}: {} vs {fd}", full.grad[k]);
    }
    // the Hessian against central differences of the (analytic) score
    let hess = full.hess.expect("Hess");
    let h = 1e-6;
    for k in 0..n {
        let g = |s: f64| {
            let mut x = x0;
            x[k] += s;
            ap.eval_grad(&x, AnalyticHessian::None).unwrap().grad
        };
        let (gp, gm) = (g(h), g(-h));
        for j in 0..n {
            let fd = (gp[j] - gm[j]) / (2.0 * h);
            assert!(
                (hess[(j, k)] - fd).abs() < 1e-4 * fd.abs().max(1.0),
                "Hessian ({j},{k}): {} vs {fd}",
                hess[(j, k)]
            );
        }
    }
}

/// evaluate_steady_state.m:438-446: on failure `set_auxiliary_variables` is applied to the
/// solver's last iterate before `info(2) = resid'*resid`. `y = y(-2) + 1` has no steady state;
/// its static residuals are `-1` (the `y` equation) and `aux - y` (the lag auxiliary
/// `AUX_ENDO_LAG_1_1 = y(-1)`), so after the reset `info(2) = 1` whatever the iterate.
#[test]
fn steady_state_failure_penalty_applies_set_auxiliary_variables_first() {
    use dynare::dsge::solve_steady;
    use dynare::solver::SteadyOptions;
    let dir = common::temp_dir("sm_ss_fail");
    let p = dir.join("nosteady.mod");
    std::fs::write(&p, "var y; varexo e; model; y = y(-2) + 1 + e; end;\n").unwrap();
    let mf = parse_mod_file(&p, &[]).unwrap();
    let m = &mf.model;
    assert_eq!(m.n_endo(), 2, "y and its lag auxiliary variable");
    let err = solve_steady(m, None, &m.params, &[3.0, -2.0], &SteadyOptions::default()).unwrap_err();
    assert_eq!(err.info, 20);
    assert!((err.info2 - 1.0).abs() < 1e-12, "info(2) = {}", err.info2);
}

/// `M_.aux_vars` of a Ramsey model lists the multipliers first (type 6 with `eq_nbr`), then the
/// lead/lag auxiliary variables of the FOCs. Expected: the 7.2 preprocessor's modfile.json
/// (`dynare-preprocessor r1.mod json=compute`) for this model.
#[test]
fn ramsey_aux_vars_list_the_multipliers() {
    let dir = common::temp_dir("sm_ramsey_aux");
    let p = dir.join("r1.mod");
    std::fs::write(
        &p,
        "var pi y i a;
varexo e;
parameters beta kappa sigma lam rho;
beta=0.99; kappa=0.1; sigma=1; lam=0.5; rho=0.8;
model;
pi = beta*pi(+1)+kappa*y+0.1*pi(-2);
y = y(+2)-(1/sigma)*(i-pi(+1))+a;
a = rho*a(-1)+e;
end;
shocks; var e; stderr 1; end;
planner_objective pi^2+lam*y^2;
ramsey_model(planner_discount=beta, instruments=(i));
",
    )
    .unwrap();
    dynare::model::dtree::json_capture();
    let mf = dynare::modfile::parse_mod_file_opts(&p, &Default::default()).unwrap();
    let cap = dynare::model::dtree::json_take().unwrap();
    let jo = dynare::q4::json::JsonOptions { stage: "compute".into(), stdout: false, derivsimple: false };
    dynare::q4::json::write(&mf, &cap, &dir, "r1", &jo).unwrap();
    let got: serde_json::Value = common::json_ref(dir.join("r1/model/json/modfile.json"));
    let want: serde_json::Value = serde_json::from_str(
        r#"[{"endo_index": 5, "type": 6, "eq_nbr": 1}, {"endo_index": 6, "type": 6, "eq_nbr": 2},
            {"endo_index": 7, "type": 6, "eq_nbr": 3}, {"endo_index": 8, "type": 0, "orig_expr": "MULT_1(1)"},
            {"endo_index": 9, "type": 0, "orig_expr": "y(1)"},
            {"endo_index": 10, "type": 1, "orig_index": 6, "orig_lead_lag": -1, "orig_expr": "MULT_2(-1)"},
            {"endo_index": 11, "type": 1, "orig_index": 1, "orig_lead_lag": -1, "orig_expr": "pi(-1)"}]"#,
    )
    .unwrap();
    assert_eq!(got["aux_vars"], want);
}
