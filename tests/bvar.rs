//! BVAR (bvar_density / bvar_forecast / bvar_irf) against Dynare 7.2 on three SW07 observables: reference/bvar/ref_bvar.json
//! (written by reference/bvar/dump_bvar.m). The forecasts and IRFs replay MATLAB's recorded
//! randn stream, so they match to rounding, not just statistically.
mod common;

use std::path::Path;

use dynare::bvar::{BvarOptions, Identification, ReplaySource, density, forecast, irf, toolbox};
use dynare::data::Table;
use dynare::linalg::Mat;
use serde_json::Value;

fn load() -> Option<(Value, Mat)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let txt = common::read_ref(root.join("reference/bvar/ref_bvar.json"));
    let r: Value = serde_json::from_str(&txt).unwrap();
    let t = Table::read_csv(&root.join("reference/sw07/sw07_obs.csv")).unwrap();
    let varobs: Vec<String> = r["varobs"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let data = t.sample(&varobs, 1, None).unwrap();
    Some((r, data))
}

fn mat(v: &Value) -> Mat {
    let rows: Vec<Vec<f64>> = match v {
        Value::Array(a) if a.first().is_some_and(|x| x.is_array()) => {
            a.iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()).collect()
        }
        Value::Array(a) => vec![a.iter().map(|x| x.as_f64().unwrap()).collect()],
        Value::Number(n) => vec![vec![n.as_f64().unwrap()]],
        _ => panic!("not a matrix: {v}"),
    };
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => panic!("not numeric: {v}"),
    }
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    let d = (a - b).abs() / b.abs().max(1.0);
    assert!(d < tol, "{what}: {a} vs {b} (rel {d:e})");
}

fn close_mat(a: &Mat, b: &Mat, tol: f64, what: &str) {
    assert_eq!(a.shape(), b.shape(), "{what}: shape");
    let scale = b.iter().fold(1.0f64, |m, x| m.max(x.abs()));
    let d = (a - b).iter().fold(0.0f64, |m, x| m.max(x.abs())) / scale;
    assert!(d < tol, "{what}: max rel diff {d:e}");
}

fn options_of(c: &Value) -> BvarOptions {
    let o = &c["options"];
    let g = |k: &str| match &o[k] {
        Value::Bool(b) => *b as u8 as f64,
        v => v.as_f64().unwrap_or_else(|| panic!("option {k}: {v}")),
    };
    BvarOptions {
        first_obs: Some(c["first_obs"].as_f64().unwrap() as usize),
        nobs: c["nobs"].as_f64().map(|x| x as usize),
        presample: g("presample") as usize,
        prefilter: g("prefilter") != 0.0,
        noconstant: g("noconstant") != 0.0,
        tau: g("bvar_prior_tau"),
        decay: g("bvar_prior_decay"),
        lambda: g("bvar_prior_lambda"),
        mu: g("bvar_prior_mu"),
        omega: g("bvar_prior_omega"),
        flat: g("bvar_prior_flat") != 0.0,
        train: g("bvar_prior_train") as usize,
        ..Default::default()
    }
}

#[test]
fn bvar_density_matches_dynare() {
    let (r, data) = load().expect("reference/setup unavailable");
    for c in r["density"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let o = options_of(c);
        let maxnlags = c["maxnlags"].as_f64().unwrap() as usize;
        let res = density(&data, maxnlags, &o).unwrap();
        let want = vecf(&c["log_marginal_data_density"]);
        for (p, ((_, d), w)) in res.iter().zip(&want).enumerate() {
            close(*d, *w, 1e-10, &format!("{name}: log density, {} lags", p + 1));
        }
        let tb = &res.last().unwrap().0;
        for (niw, key) in [(&tb.posterior, "posterior"), (&tb.prior, "prior")] {
            let w = &c[key];
            assert_eq!(niw.df, w["df"].as_f64().unwrap(), "{name}: {key} df");
            close_mat(&niw.s, &mat(&w["S"]), 1e-10, &format!("{name}: {key} S"));
            close_mat(&niw.xxi, &mat(&w["XXi"]), 1e-9, &format!("{name}: {key} XXi"));
            close_mat(&niw.phi_hat, &mat(&w["PhiHat"]), 1e-10, &format!("{name}: {key} PhiHat"));
        }
    }
}

fn forecast_options(f: &Value) -> BvarOptions {
    BvarOptions {
        first_obs: Some(f["first_obs"].as_f64().unwrap() as usize),
        nobs: Some(f["nobs"].as_f64().unwrap() as usize),
        tau: f["bvar_prior_tau"].as_f64().unwrap(),
        train: f["bvar_prior_train"].as_f64().unwrap() as usize,
        forecast: f["forecast"].as_f64().unwrap() as usize,
        replic: f["bvar_replic"].as_f64().unwrap() as usize,
        conf_sig: f["conf_sig"].as_f64().unwrap(),
        ..Default::default()
    }
}

#[test]
fn bvar_forecast_matches_dynare_with_replayed_draws() {
    let (r, data) = load().expect("reference/setup unavailable");
    let varobs: Vec<String> = r["varobs"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let f = &r["forecast"];
    let o = forecast_options(f);
    let nlags = f["nlags"].as_f64().unwrap() as usize;
    let tb = toolbox(&data, nlags, &o).unwrap();
    let z = vecf(&f["normals"]);
    let mut src = ReplaySource { values: &z, pos: 0 };
    let res = forecast(&tb, nlags, &o, &mut src).unwrap();
    assert_eq!(src.pos, z.len(), "all recorded normals consumed");
    for (kind, st) in [("with_shocks", &res.with_shocks), ("no_shock", &res.no_shock)] {
        for (field, m) in [
            ("Mean", &st.mean),
            ("Median", &st.median),
            ("Var", &st.var),
            ("HPDsup", &st.hpd_sup),
            ("HPDinf", &st.hpd_inf),
        ] {
            for (j, v) in varobs.iter().enumerate() {
                let want = vecf(&f[kind][field][v]);
                for (t, w) in want.iter().enumerate() {
                    close(m[(t, j)], *w, 1e-9, &format!("forecast {kind}.{field}.{v}[{t}]"));
                }
            }
        }
    }
    let rmse = res.rmse.unwrap();
    for (j, v) in varobs.iter().enumerate() {
        close(rmse[j], f["rmse"][v].as_f64().unwrap(), 1e-10, &format!("rmse {v}"));
    }
}

#[test]
fn bvar_irf_matches_dynare_with_replayed_draws() {
    let (r, data) = load().expect("reference/setup unavailable");
    let varobs: Vec<String> = r["varobs"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let mut o = forecast_options(&r["forecast"]);
    for c in r["irf"].as_array().unwrap() {
        o.irf = c["irf"].as_f64().unwrap() as usize;
        o.replic = c["bvar_replic"].as_f64().unwrap() as usize;
        let nlags = c["nlags"].as_f64().unwrap() as usize;
        let ident = Identification::parse(c["identification"].as_str().unwrap()).unwrap();
        let tb = toolbox(&data, nlags, &o).unwrap();
        let z = vecf(&c["normals"]);
        let mut src = ReplaySource { values: &z, pos: 0 };
        let res = irf(&tb, nlags, ident, &o, &mut src).unwrap();
        assert_eq!(src.pos, z.len());
        for (field, s) in [
            ("Mean", &res.mean),
            ("Median", &res.median),
            ("Var", &res.var),
            ("Upper_bound", &res.upper),
            ("Lower_bound", &res.lower),
        ] {
            for (a, va) in varobs.iter().enumerate() {
                for (b, vb) in varobs.iter().enumerate() {
                    let want = vecf(&c[field][va][vb]);
                    assert_eq!(want.len(), s.len());
                    for (t, w) in want.iter().enumerate() {
                        close(s[t][(a, b)], *w, 1e-9, &format!("irf {ident:?} {field}.{va}.{vb}[{t}]"));
                    }
                }
            }
        }
    }
}

#[test]
fn bvar_commands_end_to_end() {
    let (r, _) = load().expect("reference/setup unavailable");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("bvar_sw07");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(root.join("reference/bvar/bvar_sw07.mod"), dir.join("bvar_sw07.mod")).unwrap();
    std::fs::copy(root.join("reference/sw07/sw07_obs.csv"), dir.join("sw07_obs.csv")).unwrap();
    dynare::driver::run(&dir.join("bvar_sw07.mod"), &dynare::driver::RunOptions { quiet: true, ..Default::default() })
        .unwrap();
    let out = dir.join("bvar_sw07/Output");
    let d: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("bvar_sw07_bvar_density.json")).unwrap()).unwrap();
    // bvar_sw07.mod's bvar_density is the `tau_train` configuration of the dump
    let c = r["density"].as_array().unwrap().iter().find(|c| c["name"] == "tau_train").unwrap();
    for (a, b) in vecf(&d["log_marginal_data_density"]).iter().zip(vecf(&c["log_marginal_data_density"])) {
        close(*a, b, 1e-10, "bvar_density via the driver");
    }
    // bvar_forecast (nobs = 60) and bvar_irf (inherits nobs, forecast, bvar_replic): own draws,
    // so only the deterministic RMSE is compared
    let f: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("bvar_sw07_bvar_forecast_2.json")).unwrap()).unwrap();
    for v in r["varobs"].as_array().unwrap() {
        let v = v.as_str().unwrap();
        close(f["rmse"][v].as_f64().unwrap(), r["forecast"]["rmse"][v].as_f64().unwrap(), 1e-10, "rmse via the driver");
    }
    let i: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("bvar_sw07_bvar_irf_2.json")).unwrap()).unwrap();
    assert_eq!(i["identification"], "SquareRoot");
    assert_eq!(vecf(&i["Mean"]["dy"]["robs"]).len(), 40);
}
