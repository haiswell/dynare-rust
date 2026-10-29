//! `osr(analytic_derivation, analytic_derivation_mode, huge_number)` against Dynare 7.2:
//! reference/policy/osr_ad/ref_osr_ad.json, written by dump_osr_ad.m (one .mod per case, which
//! `#include`s osr_ad.mod with the case's macro switches).
mod common;

use dynare::modfile::{ModFile, parse_mod_file};
use dynare::policy::osr::{OsrObjective, WelfareKind, weight_block};
use dynare::policy::{Resolver, Session, handles};
use serde_json::Value;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("reference/policy/osr_ad")
}

fn refs() -> Value {
    common::json_ref(root().join("ref_osr_ad.json"))
}

fn nums(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(nums).collect(),
        Value::Number(x) => vec![x.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b} (diff {:e})", (a - b).abs());
}

/// The case's .mod (as dump_osr_ad.m writes it), parsed.
fn parse_case(name: &str, r: &Value) -> ModFile {
    let dir = common::temp_dir(&format!("osr_ad_{name}"));
    std::fs::copy(root().join("osr_ad.mod"), dir.join("osr_ad.mod")).unwrap();
    let opts = r["options"].as_str().unwrap();
    let mut s = String::new();
    if r["const"].as_bool().unwrap() {
        s.push_str("@#define CONST = 1\n");
    }
    if opts.contains("analytic_derivation") {
        s.push_str("@#define ORDER2 = 1\n");
    }
    s.push_str(&format!("@#define OSR_OPTIONS = \"{opts}\"\n@#include \"osr_ad.mod\"\n"));
    let f = dir.join(format!("{name}.mod"));
    std::fs::write(&f, s).unwrap();
    parse_mod_file(&f, &[]).unwrap()
}

fn run_case(mf: &ModFile) -> anyhow::Result<Session> {
    let mut s = Session::new(true);
    for c in &mf.commands {
        if handles(mf, c) {
            s.run(mf, c, None)?;
        }
    }
    Ok(s)
}

/// `osr.objective` with `analytic_derivation` at the reference points: loss and gradient.
fn check_points(name: &str, mf: &ModFile, r: &Value) {
    let m = &mf.model;
    let res = Resolver::new(mf, false);
    let p0 = m.params.clone();
    let (iv, w) = weight_block(m, &mf.policy.optim_weights);
    let dr = res.resolve(&p0).unwrap();
    let rows = iv.iter().map(|&j| dr.inv_order_var[j]).collect();
    let pd = dynare::identification::derivs::ModelParamDerivs::new(m).at_exo_steady_state(&mf.exo_steady_state());
    let obj = OsrObjective {
        resolver: &res,
        base_params: p0,
        i_params: mf.policy.osr_params.clone(),
        weights: Some((rows, w)),
        sigma_e: mf.sigma_e.clone(),
        welfare: WelfareKind::Order1,
        analytic: Some(pd),
    };
    let pts = nums(&r["points"]);
    let np = pts.len() / 2;
    let loss = nums(&r["loss"]);
    let grad = nums(&r["grad"]);
    for q in 0..np {
        // MATLAB's jsonencode writes the 4x2 matrix by rows
        let x = [pts[2 * q], pts[2 * q + 1]];
        let (l, info, _, g) = obj.loss_grad(&x);
        assert_eq!(info, 0, "{name}: info at {x:?}");
        close(l, loss[q], 1e-9, &format!("{name}: loss at {x:?}"));
        let g = g.expect("gradient");
        for k in 0..2 {
            // gradients near the optimum are tiny: tolerance relative to the loss scale
            assert!(
                (g[k] - grad[2 * q + k]).abs() <= 1e-9 * (grad[2 * q + k].abs() + 1e-6),
                "{name}: df[{k}] at {x:?}: {} vs {}",
                g[k],
                grad[2 * q + k]
            );
        }
    }
}

fn optimum(s: &Session) -> (f64, [f64; 2]) {
    let o = s.osr.as_ref().unwrap();
    let get = |n: &str| o.optim_params.iter().find(|(k, _)| k == n).unwrap().1;
    (o.objective_function, [get("phi_pi"), get("phi_y")])
}

/// csminwel (opt_algo=4, Dynare's default) with the analytic gradient: loss and gradient at
/// fixed points, the attained optimum; with a non-zero steady state (`c`), the analytic loss
/// includes `E_y E_y'` and moves the optimum (Dynare's behaviour); modes 0 and 1 agree.
#[test]
fn osr_analytic_csminwel() {
    let r = refs();
    for name in ["ad4", "ad4c", "adm0", "adm1"] {
        let rc = &r[name];
        let mf = parse_case(name, rc);
        check_points(name, &mf, rc);
        let s = run_case(&mf).unwrap();
        let (f, x) = optimum(&s);
        let xr = nums(&rc["osr_params"]);
        close(f, rc["osr_objective"].as_f64().unwrap(), 1e-11, &format!("{name}: objective"));
        close(x[0], xr[0], 1e-7, &format!("{name}: phi_pi"));
        close(x[1], xr[1], 1e-7, &format!("{name}: phi_y"));
    }
}

/// fminunc / fmincon with 'GradObj','on': the loss is very flat in phi_pi, so the attained
/// points differ between implementations (Dynare's fmincon reaches phi_pi = 110); check the
/// gradient at the reference points, that the optimizer improves on the csminwel optimum
/// (as Dynare's does), and that the objective is near Dynare's.
#[test]
fn osr_analytic_fminunc_fmincon() {
    let r = refs();
    let f_cs = r["ad4"]["osr_objective"].as_f64().unwrap();
    for name in ["ad3", "ad1"] {
        let rc = &r[name];
        let mf = parse_case(name, rc);
        check_points(name, &mf, rc);
        let s = run_case(&mf).unwrap();
        let (f, _) = optimum(&s);
        let fr = rc["osr_objective"].as_f64().unwrap();
        assert!(f < f_cs, "{name}: {f} does not improve on csminwel's {f_cs}");
        assert!((f - fr).abs() < 0.03 * fr, "{name}: objective {f} vs Dynare's {fr}");
    }
}

/// Without analytic_derivation the loss ignores the steady state: the optimum with `c` is
/// the one without (and equals Dynare's).
#[test]
fn osr_numerical_ignores_mean() {
    let r = refs();
    let rc = &r["num4c"];
    let s = run_case(&parse_case("num4c", rc)).unwrap();
    let (f, x) = optimum(&s);
    let xr = nums(&rc["osr_params"]);
    close(f, rc["osr_objective"].as_f64().unwrap(), 1e-11, "num4c: objective");
    close(x[0], xr[0], 1e-6, "num4c: phi_pi");
    close(x[1], xr[1], 1e-6, "num4c: phi_y");
}

/// analytic_derivation_mode -1 / -2 fail in Dynare 7.2 (get_perturbation_params_derivs.m reads
/// estim_params_.param_vals of the empty estim_params_ +osr/objective.m passes).
#[test]
fn osr_analytic_numerical_modes_fail_as_in_dynare() {
    let r = refs();
    for name in ["admm1", "admm2"] {
        let rc = &r[name];
        assert!(rc["error"].as_str().unwrap().contains("param_vals"));
        let e = run_case(&parse_case(name, rc)).err().expect("an error");
        assert!(format!("{e:#}").contains("param_vals"), "{name}: {e:#}");
    }
}

/// opt_algo=2 (simulated annealing) without osr_params_bounds searches within
/// [-huge_number, huge_number]: with huge_number=2 phi_pi stops at the bound 2 (unconstrained
/// optimum 2.33), as in Dynare (stochastic: compared loosely).
#[test]
fn osr_huge_number_bounds_simulated_annealing() {
    let r = refs();
    let rc = &r["hn2"];
    let s = run_case(&parse_case("hn2", rc)).unwrap();
    let (f, x) = optimum(&s);
    let xr = nums(&rc["osr_params"]);
    assert!(x[0] <= 2.0 && x[0] > 1.95, "phi_pi = {}", x[0]);
    close(x[1], xr[1], 0.02, "hn2: phi_y");
    close(f, rc["osr_objective"].as_f64().unwrap(), 1e-10, "hn2: objective");
}

/// analytic_derivation needs optim_weights (+osr/run.m).
#[test]
fn osr_analytic_requires_optim_weights() {
    let dir = common::temp_dir("osr_ad_planner");
    let src = std::fs::read_to_string(root().join("osr_ad.mod")).unwrap();
    let start = src.find("optim_weights;").unwrap();
    let end = start + src[start..].find("end;").unwrap() + 4;
    let body = format!(
        "@#define OSR_OPTIONS = \"analytic_derivation\"\n{}planner_objective pi^2 + lambda*y_gap^2;\n{}",
        &src[..start],
        &src[end..]
    );
    let f = dir.join("planner.mod");
    std::fs::write(&f, body).unwrap();
    let mf = parse_mod_file(&f, &[]).unwrap();
    let e = run_case(&mf).err().expect("an error");
    assert!(format!("{e:#}").contains("only available at order=1 with optim_weights"), "{e:#}");
}
