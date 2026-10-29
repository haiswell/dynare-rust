//! Estimation completion: the pruned skewed Kalman filter and smoother (`kalman_algo = 5`)
//! against Dynare 7.2 (reference/est_complete/skewed/dump_skewed.m, sk_nk.mod): smoother
//! (kalman_smoother_pruned_skewed.m), `skewed_kalman_rank_deficiency_transform`,
//! `skewed_kalman_mvnlogcdf = 'mvncdf'` (Dynare's own mvncdf), missing observations,
//! `lik_init` 2/4/5, measurement errors with `filter_initial_state`, the `lik_init = 3` error,
//! and the CDF / CSN helpers on fixed inputs.
mod common;

use std::path::{Path, PathBuf};

use dynare::linalg::{Mat, Vector};
use dynare::modfile::{Command, OptValue, opt};
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/est_complete/skewed")
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

/// A MATLAB matrix as jsonencode writes it (row arrays; a scalar or a vector for 1-row/column).
fn mat(v: &Value, r: usize, c: usize) -> Mat {
    let f = vecf(v);
    assert_eq!(f.len(), r * c);
    Mat::from_row_slice(r, c, &f)
}

fn load(name: &str) -> Option<Value> {
    Some(common::json_ref(root().join(name)))
}

struct Setup {
    mf: dynare::modfile::ModFile,
    est: dynare::modfile::Options,
    dsge: dynare::dsge::DsgeModel,
    data: dynare::dsge::LikelihoodData,
}

/// `nobs`: None = the estimation's `nobs` option (the smoother of the estimation run).
fn setup(stage: usize, x0: Option<&[f64]>, nobs: Option<usize>) -> anyhow::Result<Setup> {
    let defines = vec![("STAGE".to_string(), stage.to_string())];
    let mf = dynare::modfile::parse_mod_file(&root().join("sk_nk.mod"), &defines)?;
    let est = mf
        .commands
        .iter()
        .find_map(|c| if let Command::Estimation(o, _) = c { Some(o.clone()) } else { None })
        .unwrap();
    let num = |k: &str| match opt(&est, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    };
    let datafile = match opt(&est, "datafile") {
        Some(OptValue::Str(s)) => s.trim_matches(|c| c == '\'' || c == '"').to_string(),
        _ => unreachable!(),
    };
    let dsge = dynare::dsge::DsgeModel::new(&mf, x0)?;
    let table = dynare::data::Table::read(&root().join(format!("{datafile}.csv")), &Default::default())?;
    let sample = table.sample(&mf.varobs, 1, nobs.or(num("nobs").map(|v| v as usize)))?;
    let mut k = dynare::kalman::KalmanOptions {
        lik_init: if opt(&est, "diffuse_filter").is_some() { 3 } else { num("lik_init").unwrap_or(1.0) as u8 },
        kalman_algo: num("kalman_algo").unwrap_or(0.0) as u8,
        ..Default::default()
    };
    dynare::estimation::q3::kalman_options(&est, &mut k);
    let mut data = dynare::dsge::LikelihoodData::new(&sample, k);
    dynare::estimation::q3::attach(&mf, &est, &mut data, 1, dynare::dates::Date::parse("1Y").unwrap())?;
    Ok(Setup { mf, est, dsge, data })
}

/// `tol`: relative tolerance of the objective; `smoother_tol`: absolute, smoothed objects.
fn check_points(stage: usize, tol: f64, smoother_tol: f64) {
    let r = load(&format!("ref_skewed_{stage}.json")).expect("reference/setup unavailable");
    let x0 = vecf(&r["xparam_init"]);
    // dump_skewed.m evaluates the points after resetting options_.nobs (the whole sample)
    let s = setup(stage, Some(&x0), Some(r["nobs"].as_u64().unwrap() as usize)).unwrap();
    assert_eq!(s.data.y.ncols(), r["nobs"].as_u64().unwrap() as usize, "nobs");
    let q3x = dynare::estimation::q3::Extras::new(&s.mf, &s.est, &s.dsge, &s.data).unwrap();
    let post = q3x.posterior(&s.dsge, &s.data, &s.dsge.bounds).expect("kalman_algo = 5 objective");
    for (k, p) in r["points"].as_array().unwrap().iter().enumerate() {
        let x = vecf(&p["x"]);
        let f = post.eval(&x).0;
        let v = p["fval"].as_f64().unwrap();
        println!("stage {stage} point {k}: {f:.12} vs Dynare {v:.12} (rel {:.1e})", ((f - v) / v).abs());
        assert!(((f - v) / v).abs() < tol, "stage {stage} point {k}: {f} vs {v}");
    }
    if r.get("SmoothedVariables").is_some() {
        let s = setup(stage, Some(&x0), None).unwrap();
        let sm = dynare::driver::run_smoother(&s.dsge, &s.data, &x0, false).unwrap();
        for (j, name) in s.mf.model.endo_names[..s.mf.model.orig_endo_nbr].iter().enumerate() {
            let want = vecf(&r["SmoothedVariables"][name]);
            let d = (0..want.len()).map(|t| (sm.smoothed[(t, j)] - want[t]).abs()).fold(0.0, f64::max);
            println!("   smoothed {name}: {d:.1e}");
            assert!(d < smoother_tol, "smoothed {name} differs by {d}");
        }
        for (j, name) in s.mf.model.exo_names.iter().enumerate() {
            let want = vecf(&r["SmoothedShocks"][name]);
            let d = (0..want.len()).map(|t| (sm.shocks[(t, j)] - want[t]).abs()).fold(0.0, f64::max);
            println!("   smoothed shock {name}: {d:.1e}");
            assert!(d < smoother_tol, "smoothed shock {name} differs by {d}");
        }
        if let Some(me) = r.get("SmoothedMeasurementErrors") {
            let ours = sm.measurement_errors.as_ref().expect("measurement errors");
            for (j, name) in s.mf.varobs.iter().enumerate() {
                let want = vecf(&me[name]);
                let d = (0..want.len()).map(|t| (ours[(t, j)] - want[t]).abs()).fold(0.0, f64::max);
                println!("   smoothed measurement error {name}: {d:.1e}");
                assert!(d < smoother_tol, "measurement error {name} differs by {d}");
            }
        }
    }
}

#[test]
fn skewed_smoother() {
    check_points(1, 1e-10, 1e-12);
}

#[test]
fn rank_deficiency_transform() {
    check_points(2, 1e-10, 1e-12);
}

#[test]
fn mvncdf_cdfs() {
    check_points(3, 1e-10, 1e-12);
}

#[test]
fn missing_observations_gaussian_filter() {
    check_points(4, 1e-10, 1e-12);
}

#[test]
fn lik_init_2_smoother() {
    check_points(5, 1e-10, 1e-12);
}

/// lik_init = 4: Dynare's `kalman_steady_state` MEX does not load on this platform (SLICOT
/// symbol `sb02od_` missing), so MATLAB prints "I am not able to solve the Riccati equation,
/// so I switch to lik_init=1!" and its reference values are those of lik_init = 1 (stage 1).
/// dynare-rust solves the Riccati equation (kalman::lik_init), so its objective is the
/// skewed filter started from the Riccati steady state: finite and close to the lik_init = 1
/// value.
#[test]
fn lik_init_4() {
    let (r, r1) = (
        load("ref_skewed_6.json").expect("reference/setup unavailable"),
        load("ref_skewed_1.json").expect("reference/setup unavailable"),
    );
    for (p, p1) in r["points"].as_array().unwrap().iter().zip(r1["points"].as_array().unwrap()) {
        assert_eq!(p["fval"], p1["fval"], "MATLAB fell back to lik_init = 1");
    }
    let x0 = vecf(&r["xparam_init"]);
    let s = setup(6, Some(&x0), Some(r["nobs"].as_u64().unwrap() as usize)).unwrap();
    let q3x = dynare::estimation::q3::Extras::new(&s.mf, &s.est, &s.dsge, &s.data).unwrap();
    let post = q3x.posterior(&s.dsge, &s.data, &s.dsge.bounds).unwrap();
    for p in r["points"].as_array().unwrap() {
        let f = post.eval(&vecf(&p["x"])).0;
        let v = p["fval"].as_f64().unwrap();
        println!("lik_init = 4 (Riccati): {f:.10}; Dynare (lik_init = 1 fallback) {v:.10}");
        assert!(f.is_finite() && ((f - v) / v).abs() < 1e-3);
    }
}

#[test]
fn lik_init_5() {
    check_points(7, 1e-10, 1e-12);
}

/// Measurement error on y (pi and i observed exactly) and filter_initial_state. The likelihood
/// matches to rounding. The smoother does not, by construction of Dynare's algorithm: the
/// smoothed covariances of the exactly observed states are zero up to rounding noise (MATLAB
/// gets e.g. -3e-33 or 2e-16), and csn_prune_distribution.m keeps or drops a skewness
/// dimension according to `|P_ij| / sqrt(|P_ii| |P_jj|)` built on that noise. In the periods
/// where both programs keep the same dimensions the smoothed states agree to ~1e-15; in the
/// others the CSN means differ by up to ~3e-3 (an approximation choice, not a bug of either).
#[test]
fn measurement_error_filter_initial_state_smoother() {
    check_points(8, 1e-10, 5e-3);
}

#[test]
fn lik_init_3_is_an_error() {
    let r = load("ref_skewed_9.json").expect("reference/setup unavailable");
    let err = setup(9, None, None).err().map(|e| e.to_string()).unwrap_or_default();
    assert_eq!(err, r["error"].as_str().unwrap());
}

#[test]
fn rank_deficiency_mvncdf_lik_init_2() {
    check_points(10, 1e-10, 1e-12);
}

#[test]
fn mvncdf_and_csn_helpers() {
    use dynare::kalman::csn::{CsnDist, csn_mean, csn_statespace_linear_transform};
    use dynare::kalman::mvncdf::mvncdf;
    use dynare::kalman::skewed::Mvnlogcdf;
    let c = load("ref_skewed_mvncdf.json").expect("reference/setup unavailable");
    let mut worst = 0.0f64;
    for (key, d) in [("bv", 2), ("tv", 3)] {
        for e in c[key].as_array().unwrap() {
            let p = mvncdf(&vecf(&e["x"]), &vecf(&e["mu"]), &mat(&e["Sigma"], d, d)).unwrap();
            let want = e["p"].as_f64().unwrap();
            worst = worst.max((p - want).abs());
            assert!((p - want).abs() < 1e-13, "{key}: {p} vs {want}");
        }
    }
    println!("bivariate/trivariate mvncdf: max abs diff {worst:.1e}");
    // d = 5: randomized QMC (tolerance 1e-4 at 3.5 standard errors) against the mean of 20
    // MATLAB draws
    let q = &c["qmc"][0];
    let p = mvncdf(&vecf(&q["x"]), &[0.0; 5], &mat(&q["Sigma"], 5, 5)).unwrap();
    let draws = vecf(&q["p"]);
    let m = draws.iter().sum::<f64>() / draws.len() as f64;
    println!("QMC mvncdf: {p:.6} vs MATLAB mean {m:.6}");
    assert!((p - m).abs() < 1e-4);
    for e in c["csn"].as_array().unwrap() {
        let q = vecf(&e["nu"]).len();
        let (mu, sigma, gamma) = (Vector::from_vec(vecf(&e["mu"])), mat(&e["Sigma"], 3, 3), mat(&e["Gamma"], q, 3));
        let (nu, delta) = (Vector::from_vec(vecf(&e["nu"])), mat(&e["Delta"], q, q));
        let me = csn_mean(&mu, &sigma, &gamma, &nu, &delta, Mvnlogcdf::MendellElston).unwrap();
        let d = (me - Vector::from_vec(vecf(&e["mean_me"]))).amax();
        assert!(d < 1e-13, "csn_mean (Mendell-Elston), q = {q}: {d}");
        let mv = csn_mean(&mu, &sigma, &gamma, &nu, &delta, Mvnlogcdf::Mvncdf).unwrap();
        let d = (mv - Vector::from_vec(vecf(&e["mean_mvncdf"]))).amax();
        println!("csn_mean q = {q}: mvncdf diff {d:.1e}");
        // q - 1 <= 3: deterministic quadratures; q = 4: QMC denominators
        assert!(d < if q <= 3 { 1e-12 } else { 1e-3 }, "csn_mean (mvncdf), q = {q}: {d}");
    }
    for e in c["lt"].as_array().unwrap() {
        let gv = &e["G"];
        let rows = gv.as_array().unwrap().len();
        let g = mat(gv, rows, 4);
        let x = CsnDist {
            mu: Vector::from_vec(vecf(&e["mu"])),
            sigma: mat(&e["Sigma"], 4, 4),
            gamma: mat(&e["Gamma"], 2, 4),
            nu: Vector::from_vec(vecf(&e["nu"])),
            delta: mat(&e["Delta"], 2, 2),
        };
        let t = csn_statespace_linear_transform(&g, &x);
        let rel = |a: &Mat, b: &Mat| (a - b).amax() / b.amax().max(1.0);
        let dg = rel(&t.gamma, &mat(&e["Gamma_tr"], 2, rows));
        let dd = rel(&t.delta, &mat(&e["Delta_tr"], 2, 2));
        let ds = rel(&t.sigma, &mat(&e["Sigma_tr"], rows, rows));
        println!("linear transform ({rows}x4): Gamma {dg:.1e} Delta {dd:.1e} Sigma {ds:.1e}");
        assert!(dg < 1e-10 && dd < 1e-10 && ds < 1e-14);
    }
}
