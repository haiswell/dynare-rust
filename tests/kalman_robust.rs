//! The likelihood at ill-conditioned points: the 245 stable SW07 prior draws of the GSA RMSE
//! mapping (`reference/gsa/ref_gsa_sw07_stage21.json`, Dynare's `logpo2`) and how well Dynare
//! itself determines them (`reference/gsa/ref_gsa_sw07_kalman.json`, written by
//! `reference/gsa/dump_sw07_gsa_kalman.m`: dsge_likelihood at each draw and at three
//! perturbations `x .* (1 + 1e-14 randn)`).
//!
//! On the prior support some draws make the forecast-error covariance F badly conditioned:
//! the filter then amplifies rounding, and the value is decided by rounding in Dynare as here.
//! On SW07 Dynare's own value moves by more than 1e-6 under 1e-14 perturbations at 2 of the
//! 245 draws (by ~1 log point: draws 133 and 166, 1-based). Everywhere Dynare's value is
//! determined, dynare-rust's must agree with it to 1e-6.
mod common;

use std::path::{Path, PathBuf};

use dynare::dsge::{DsgeModel, DsgePosterior};
use dynare::gsa;
use dynare::modfile::{Command, parse_mod_file};
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn vec1(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(num).collect(),
        Value::Null => vec![],
        x => vec![num(x)],
    }
}

fn mat(v: &Value) -> Vec<Vec<f64>> {
    match v {
        Value::Array(a) if a.first().is_some_and(Value::is_array) => a.iter().map(vec1).collect(),
        Value::Array(a) => a.iter().map(|x| vec![num(x)]).collect(),
        _ => vec![],
    }
}

fn load(name: &str) -> Option<Value> {
    Some(common::json_ref(root().join("reference/gsa").join(name)))
}

/// Relative difference of log posteriors; NaN (JSON null: -Inf in MATLAB) matches any
/// non-finite value.
fn rel(a: f64, b: f64) -> f64 {
    match (a.is_finite(), b.is_finite()) {
        (false, false) => 0.0,
        (true, true) => (a - b).abs() / b.abs().max(1.0),
        _ => f64::INFINITY,
    }
}

#[test]
fn sw07_gsa_draws_ill_conditioned_filter() {
    let (r, sens) = (
        load("ref_gsa_sw07_stage21.json").expect("reference/setup unavailable"),
        load("ref_gsa_sw07_kalman.json").expect("reference/setup unavailable"),
    );
    let tmp = std::env::temp_dir().join(format!("dynare_rs_krob_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    for f in ["sw07.mod", "sw07_obs.csv"] {
        std::fs::copy(root().join("reference/sw07").join(f), tmp.join(f)).unwrap();
    }
    let mf = parse_mod_file(&tmp.join("sw07.mod"), &[("STAGE".into(), "21".into())]).unwrap();
    let o = mf
        .commands
        .iter()
        .find_map(|c| match c {
            Command::Other(n, o) if n == "sensitivity" => Some(o.clone()),
            _ => None,
        })
        .unwrap();
    let mut dsge = DsgeModel::new(&mf, None).unwrap();
    dsge.set_solver_options(dynare::solver::SolverOptions {
        qz_criterium: gsa::qz_criterium(&o, true),
        ..Default::default()
    });
    let data = gsa::rmse::likelihood_data(&mf, &tmp, &o, "sw07_obs.csv").unwrap();
    let _ = std::fs::remove_dir_all(&tmp);
    let post = DsgePosterior { dsge: &dsge, data: &data, bounds: &dsge.bounds };
    let (l, l0) = (mat(&r["lpmat"]), mat(&r["lpmat0"]));
    let ist: Vec<usize> = vec1(&r["istable"]).into_iter().map(|x| x as usize - 1).collect();
    let logpo2 = vec1(&r["logpo2"]);
    // Dynare's -fval at x and three perturbations of x
    let dyn_f: Vec<Vec<f64>> = mat(&sens["fval"]).into_iter().map(|v| v.into_iter().map(|f| -f).collect()).collect();
    assert_eq!(dyn_f.len(), ist.len());
    let (mut agree, mut determined, mut agree_determined) = (0, 0, 0);
    for (j, &k) in ist.iter().enumerate() {
        let x: Vec<f64> = l0[k].iter().chain(&l[k]).copied().collect();
        let ours = post.evaluate(&x).map_or(f64::NEG_INFINITY, |e| -e.fval);
        let theirs = logpo2[j];
        // the dump reproduces the GSA's logpo2 (to 2e-11: the GSA's options_ are not the estimation's)
        assert!(rel(dyn_f[j][0], theirs) < 1e-6, "draw {j}: {} vs {theirs}", dyn_f[j][0]);
        let ok = rel(ours, theirs) < 1e-6;
        agree += ok as usize;
        if dyn_f[j][1..].iter().all(|&f| rel(f, theirs) < 1e-6) {
            determined += 1;
            agree_determined += ok as usize;
        }
    }
    let n = ist.len();
    eprintln!(
        "log posterior within 1e-6 of Dynare: {agree}/{n} draws; Dynare's value is determined (stable to 1e-6 \
         under 1e-14 perturbations) at {determined} draws, of which dynare-rust agrees at {agree_determined}"
    );
    // 243/245, and every one of the 243 draws where Dynare's value is determined
    assert!(determined as f64 >= 0.98 * n as f64, "Dynare's value determined at only {determined}/{n} draws");
    assert_eq!(agree_determined, determined, "{agree_determined}/{determined}");
    assert!(agree as f64 >= 0.98 * n as f64, "{agree}/{n}");
}
