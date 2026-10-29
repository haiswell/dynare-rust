//! Method of moments against Dynare 7.2: `reference/mom/dump_mom.m` ->
//! `ref_rbc_case<k>.json` for `reference/mom/rbc_mom.mod -DCASE=k`
//! (1 GMM order 1, 2 GMM order 2, 3 GMM order 3, 4 SMM order 1, 5 SMM order 2 pruned,
//! 6 penalized GMM with priors).
//!
//! Deterministic pieces (data moments, `m_data`, objective and model moments at fixed points,
//! the optimal weighting matrix, standard errors and J-test at Dynare's estimate) are checked
//! tightly; SMM is fed MATLAB's `randn` shock series. The optimizers' end points are compared
//! by the objective value (csminwel perturbs on cliffs with its own random draws).
mod common;

use std::path::{Path, PathBuf};

use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use dynare::mom::moments::optimal_weighting_matrix;
use dynare::mom::run::{Context, MomResults, jtest, method_of_moments, setup, standard_errors_at};
use dynare::mom::{MomInputs, blocks_of, last_mom_options};
use serde_json::Value;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/mom")
}

fn load(case: usize) -> Option<Value> {
    let p = dir().join(format!("ref_rbc_case{case}.json"));
    Some(common::json_ref(p))
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![],
    }
}

fn matf(v: &Value) -> Mat {
    let rows: Vec<Vec<f64>> = v.as_array().unwrap().iter().map(vecf).collect();
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-12)
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs() / y.abs().max(1e-8)).fold(0.0, f64::max)
}

struct Case {
    r: Value,
    s: dynare::mom::run::Setup,
    mf: dynare::modfile::ModFile,
    inputs: MomInputs,
    ctx: Context,
}

fn case(k: usize) -> Option<Case> {
    let r = load(k)?;
    let mf = parse_mod_file(&dir().join("rbc_mom.mod"), &[("CASE".into(), k.to_string())]).unwrap();
    let blocks = blocks_of(&mf);
    let opts = last_mom_options(&mf).unwrap();
    let mut inputs = MomInputs { quiet: true, ..Default::default() };
    if let Some(z) = r.get("shock_series") {
        inputs.shock_series = Some(matf(z));
    }
    let tmp = std::env::temp_dir().join(format!("dynare_rs_mom_case{k}"));
    std::fs::create_dir_all(&tmp).unwrap();
    for f in ["rbc_mom_data1.csv", "rbc_mom_data2.csv"] {
        std::fs::copy(dir().join(f), tmp.join(f)).unwrap();
    }
    let ctx = Context { dir: tmp, fname: "rbc_mom".into(), seed_override: None, stream_seed: 0 };
    let s = setup(&mf, &blocks, &opts, &ctx, &inputs).unwrap();
    Some(Case { r, s, mf, inputs, ctx })
}

/// Checks everything that does not depend on the optimizer path.
fn deterministic(k: usize, tol: f64) -> Option<Case> {
    let c = case(k)?;
    let (r, s) = (&c.r, &c.s);
    let mom = &r["mom"];
    // data moments and m_data
    let dm = vecf(&mom["data_moments"]);
    let e = max_rel(&s.data_moments, &dm);
    println!("case {k}: data moments {e:.2e}");
    assert!(e < 1e-12, "data moments");
    let md = matf(&mom["m_data"]);
    let m_data = s.m_data.as_ref().unwrap();
    let e = (m_data - &md).abs().max();
    assert!(e < 1e-10, "m_data {e}");
    // optimal weighting matrix at the data moments
    let lag = s.o.bartlett_kernel_lag;
    let w = optimal_weighting_matrix(m_data, &s.data_moments, lag).unwrap();
    let wr = matf(&r["W_opt_data"]);
    let e = (&w - &wr).abs().max() / wr.abs().max();
    println!("case {k}: optimal W {e:.2e}");
    assert!(e < 1e-8, "W_opt");
    // objective at fixed points
    let id = Mat::identity(s.mom_nbr, s.mom_nbr);
    let sw = matf(&mom["weighting_info"]["Sw"]);
    for p in r["points"].as_array().unwrap() {
        let x = vecf(&p["x"]);
        let ev = match s.problem.evaluate(&x, &s.data_moments, &id) {
            Ok(ev) => ev,
            Err(f) => {
                println!(
                    "case {k}: x={x:?}: info {} (Dynare: fval {}, info {})",
                    f.info, p["fval_identity"], p["info"]
                );
                assert!(p["fval_identity"].as_f64().is_none_or(|v| !v.is_finite()));
                assert_eq!(p["info"].as_f64(), Some(f.info as f64));
                continue;
            }
        };
        let e1 = rel(ev.fval, p["fval_identity"].as_f64().unwrap());
        let e2 = max_rel(&ev.model_moments, &vecf(&p["model_moments"]));
        let ef = s.problem.evaluate(&x, &s.data_moments, &sw).unwrap().fval;
        let e3 = rel(ef, p["fval_final"].as_f64().unwrap());
        println!("case {k}: x={x:?}: fval(I) {e1:.2e}, model moments {e2:.2e}, fval(final W) {e3:.2e}");
        assert!(e1 < tol && e2 < tol && e3 < tol);
    }
    // standard errors and J-test at Dynare's estimate with Dynare's final weighting matrix
    let xf = vecf(&r["points"][2]["x"]);
    let wopt = mom["weighting_info"]["Woptflag"].as_bool().unwrap();
    let (se, _) = standard_errors_at(&s.problem, &xf, m_data, &sw, wopt, s.nobs, s.o.se_tolx, lag).unwrap();
    // the dump evaluates standard_errors.m without bounds; in the run itself a bound hit by
    // the finite differences gives NaN (as dynare-rust, which uses the bounds)
    let method = r["mom_method"].as_str().unwrap().to_lowercase();
    let run_std = &mom[format!("{method}_std_at_mode")];
    let run_nan = run_std
        .as_object()
        .is_some_and(|o| o.values().flat_map(|g| g.as_object().unwrap().values()).any(|v| v.is_null()));
    let ser = if run_nan { vec![f64::NAN; se.len()] } else { vecf(&r["se_at_final"]) };
    let e = max_rel(&se, &ser);
    println!("case {k}: standard errors {se:?} vs {ser:?} ({e:.2e})");
    if ser.iter().all(|v| v.is_finite()) {
        assert!(e < 1e-6, "standard errors");
    } else {
        assert!(se.iter().all(|v| v.is_nan()));
    }
    if let Some(j) = mom.get("J_test").filter(|j| j.is_object()) {
        let jt = jtest(&s.problem, &xf, &s.data_moments, m_data, &sw, wopt, s.nobs, s.vcf, lag).unwrap();
        let e = rel(jt.j_stat, j["j_stat"].as_f64().unwrap());
        println!("case {k}: J {} vs {} ({e:.2e}), p {}", jt.j_stat, j["j_stat"], jt.p_val);
        assert!(e < 1e-8 && (jt.p_val - j["p_val"].as_f64().unwrap()).abs() < 1e-8);
    }
    Some(c)
}

/// Full run; the minimized objective must not be worse than Dynare's (beyond `ftol`).
/// Where the optimizer stops depends on last-bit libm differences, so off macOS `ftol` is at
/// least 1e-4 (up to 2.0e-5 above Dynare's: SMM case 4 on Linux, GMM case 1 on Windows).
fn end_to_end(c: &Case, k: usize, ftol: f64) -> MomResults {
    let ftol = if cfg!(target_os = "macos") { ftol } else { ftol.max(1e-4) };
    let blocks = blocks_of(&c.mf);
    let opts = last_mom_options(&c.mf).unwrap();
    let res = method_of_moments(&c.mf, &blocks, &opts, &c.ctx, &c.inputs).unwrap();
    let mom = &c.r["mom"];
    let method = c.r["mom_method"].as_str().unwrap().to_lowercase();
    let xr = vecf(&c.r["points"][2]["x"]);
    let qr = mom["Q"].as_f64().unwrap();
    println!(
        "case {k}: estimate {:?} (Dynare {xr:?}); Q {} (Dynare {qr}); J {:?}",
        res.xparam1,
        res.q,
        res.j_test.as_ref().map(|j| j.j_stat)
    );
    let _ = method;
    // compare the final-stage objective of both estimates under Rust's final weighting matrix
    let fr = c.s.problem.evaluate(&xr, &c.s.data_moments, &res.weighting).map(|e| e.fval).unwrap_or(f64::INFINITY);
    println!("case {k}: final-stage objective at Rust's estimate {} and at Dynare's {fr}", res.fval);
    assert!(res.fval <= fr * (1.0 + ftol) + 1e-10, "Rust's estimate is worse than Dynare's");
    assert!(c.ctx.dir.join("rbc_mom/Output/rbc_mom_mom_results.json").exists());
    res
}

#[test]
fn gmm_order1() {
    if let Some(c) = deterministic(1, 1e-9) {
        end_to_end(&c, 1, 1e-6);
    }
}

#[test]
fn gmm_order2() {
    if let Some(c) = deterministic(2, 1e-8) {
        end_to_end(&c, 2, 1e-6);
    }
}

#[test]
fn gmm_order3() {
    if let Some(c) = deterministic(3, 1e-8) {
        end_to_end(&c, 3, 1e-5);
    }
}

#[test]
fn smm_order1_matlab_shocks() {
    if let Some(c) = deterministic(4, 1e-9) {
        end_to_end(&c, 4, 1e-6);
    }
}

#[test]
fn smm_order2_matlab_shocks() {
    // csminwel crosses cliffs on this objective; off macOS (other libm and BLAS) its path
    // differs from the first cliff on and stops at another point (4% above Dynare's objective
    // on Linux), so the end-to-end comparison is macOS only
    if let Some(c) = deterministic(5, 1e-8)
        && cfg!(target_os = "macos")
    {
        end_to_end(&c, 5, 1e-6);
    }
}

#[test]
fn penalized_gmm() {
    if let Some(c) = deterministic(6, 1e-9) {
        end_to_end(&c, 6, 1e-6);
    }
}

// ---- IRF matching (reference/mom/nk_irf.mod, dump_irf.m) ------------------------------------

fn irf_case(k: usize) -> Option<(Value, Case)> {
    let r: Value = common::json_ref(dir().join(format!("ref_nk_irf_case{k}.json")));
    let mf = parse_mod_file(&dir().join("nk_irf.mod"), &[("CASE".into(), k.to_string())]).unwrap();
    let blocks = blocks_of(&mf);
    let opts = last_mom_options(&mf).unwrap();
    let inputs = MomInputs { quiet: true, ..Default::default() };
    let tmp = std::env::temp_dir().join(format!("dynare_rs_mom_irf{k}"));
    std::fs::create_dir_all(&tmp).unwrap();
    let ctx = Context { dir: tmp, fname: "nk_irf".into(), seed_override: None, stream_seed: 0 };
    let s = setup(&mf, &blocks, &opts, &ctx, &inputs).unwrap();
    Some((r.clone(), Case { r, s, mf, inputs, ctx }))
}

fn irf_deterministic(k: usize) -> Option<(Value, Case)> {
    let (r, c) = irf_case(k)?;
    let s = &c.s;
    let mom = &r["mom"];
    let e = max_rel(&s.data_moments, &vecf(&mom["data_moments"]));
    assert!(e < 1e-14, "IRF data");
    let idx: Vec<usize> = vecf(&r["irfIndex"]).iter().map(|v| *v as usize - 1).collect();
    assert_eq!(s.irf_targets.as_ref().unwrap().index, idx);
    let w = &s.irf_targets.as_ref().unwrap().w;
    let wr = matf(&mom["weighting_info"]["W"]);
    for i in 0..w.nrows() {
        for j in 0..w.ncols() {
            if w[(i, j)] != wr[(i, j)] {
                if (w[(i, j)] - wr[(i, j)]).abs() > 1e-12 * wr[(i, j)].abs() {
                    println!("W({i},{j}): {} vs Dynare {}", w[(i, j)], wr[(i, j)]);
                }
            }
        }
    }
    // Dynare's preprocessor rounds some of the 17-digit literals to the neighbouring double
    assert!((w - &wr).abs().max() / wr.abs().max() < 1e-15, "W");
    let ld = s.problem.irf.as_ref().unwrap().winv_logdet;
    let ldr = mom["weighting_info"]["Winv_logdet"].as_f64().unwrap();
    println!("irf case {k}: Winv_logdet {ld} vs {ldr}");
    assert!(rel(ld, ldr) < 1e-12);
    // case 3 (relative_irf against level IRFs) ends at a degenerate corner (rho = 1e-14): its mode is skipped
    for p in r["points"].as_array().unwrap().iter().take(if k == 3 { 2 } else { 3 }) {
        let x = vecf(&p["x"]);
        let ev = s.problem.evaluate(&x, &s.data_moments, w).unwrap();
        let e1 = rel(ev.fval, p["fval"].as_f64().unwrap());
        let e2 = max_rel(&ev.model_moments, &vecf(&p["model_moments"]));
        println!("irf case {k}: x={x:?}: fval {} ({e1:.2e}), model IRFs {e2:.2e}", ev.fval);
        assert!(e1 < 1e-11 && e2 < 1e-10);
    }
    Some((r, c))
}

fn run_case(c: &Case) -> MomResults {
    method_of_moments(&c.mf, &blocks_of(&c.mf), &last_mom_options(&c.mf).unwrap(), &c.ctx, &c.inputs).unwrap()
}

#[test]
fn irf_matching_frequentist() {
    let (r, c) = irf_deterministic(1).expect("reference/setup unavailable");
    // Hessian (hessian.m) at Dynare's mode
    let mfile = &r["mode_file"];
    let xm = vecf(&mfile["xparam1"]);
    let w = &c.s.irf_targets.as_ref().unwrap().w;
    let obj = dynare::mom::problem::MomObjective { p: &c.s.problem, data: &c.s.data_moments, w, analytic: false };
    let h = dynare::estimation::hessian(&obj, &xm, [1e-2, 1.0]);
    let hr = matf(&mfile["hh"]);
    let e = (&h - &hr).abs().max() / hr.abs().max();
    println!("irf case 1: hessian at Dynare's mode {e:.2e}");
    assert!(e < 1e-6);
    let res = run_case(&c);
    let e = max_rel(&res.xparam1, &xm);
    println!("irf case 1: mode {:?} vs {xm:?} ({e:.2e}); fval {} vs {}", res.xparam1, res.fval, mfile["fval"]);
    assert!(e < 1e-5);
    assert!(rel(res.fval, mfile["fval"].as_f64().unwrap()) < 1e-9);
    let sd = &r["mom"]["irf_matching_std_at_mode"];
    let sdr: Vec<f64> = std::iter::once(sd["shocks_std"]["e_i"].as_f64().unwrap())
        .chain(["kappa", "phi_pi", "rho_a", "rho_i"].iter().map(|n| sd["parameters"][n].as_f64().unwrap()))
        .collect();
    let e = max_rel(&res.stdh, &sdr);
    println!("irf case 1: s.d. {:?} vs {sdr:?} ({e:.2e})", res.stdh);
    assert!(e < 1e-4);
}

#[test]
fn irf_matching_bayesian() {
    let (r, c) = irf_deterministic(2).expect("reference/setup unavailable");
    let xm = vecf(&r["mode_file"]["xparam1"]);
    let res = run_case(&c);
    let e = max_rel(&res.xparam1, &xm);
    let lr = r["mom"]["MarginalDensity"]["LaplaceApproximation"].as_f64().unwrap();
    println!("irf case 2: mode {:?} vs {xm:?} ({e:.2e}); Laplace {:?} vs {lr}", res.xparam1, res.laplace);
    assert!(e < 1e-5);
    assert!((res.laplace.unwrap() - lr).abs() < 1e-5);
}

#[test]
fn irf_matching_relative_irf() {
    let _ = irf_deterministic(3);
}
