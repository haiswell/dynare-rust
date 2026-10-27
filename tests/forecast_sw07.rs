//!  against Dynare 7.2 on SW07 (STAGE 9: mode file, first_obs=5, presample=8,
//! diffuse filter):
//!   - point forecast at the mode with conf_sig bands (`oo_.forecast`, ref_forecast.json),
//!     and the `forecast` command's 'simul' task;
//!   - shock decomposition at the mode (`oo_.shock_decomposition`, ref_shock_decomp.json);
//!   - posterior distributions of smoothed/updated/filtered variables, smoothed shocks and
//!     forecasts on the exact draws Dynare used (ref_posterior.json): everything but the
//!     simulated-shock PointForecast must match to rounding.
mod common;

use std::path::Path;

use dynare::analysis::shock_decomposition::shock_decomposition;
use dynare::data::Table;
use dynare::dsge::{DsgeModel, LikelihoodData};
use dynare::estimation::prior_posterior::{PosteriorSmootherOptions, PosteriorSummary, prior_posterior_statistics};
use dynare::forecast::{dsge_smoother, stochastic_classical};
use dynare::kalman::KalmanOptions;
use dynare::modfile::{ModFile, parse_mod_file};
use serde_json::Value;

fn load(name: &str) -> Option<Value> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07").join(name);
    Some(common::json_ref(p))
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

struct Setup {
    mf: ModFile,
    dsge: DsgeModel,
    data: LikelihoodData,
    xm: Vec<f64>,
}

fn setup() -> Option<Setup> {
    let r = load("ref.json")?;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let mf = parse_mod_file(&dir.join("sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let xm = vecf(&r["xparam_mode"]);
    let dsge = DsgeModel::new(&mf, Some(&xm)).unwrap();
    let table = Table::read_csv(&dir.join("sw07_obs.csv")).unwrap();
    let sample = table.sample(&mf.varobs, 5, None).unwrap();
    let kopts = KalmanOptions { presample: 8, lik_init: 3, ..Default::default() };
    let data = LikelihoodData::new(&sample, kopts);
    Some(Setup { mf, dsge, data, xm })
}

fn names(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

#[test]
fn point_forecast_at_mode() {
    let (s, f) = (
        setup().expect("reference/setup unavailable"),
        load("ref_forecast.json").expect("reference/setup unavailable"),
    );
    let sm = dsge_smoother(&s.dsge, &s.data, &s.xm, false).unwrap();
    let endo = &s.mf.model.endo_names;
    // forecast start: last smoothed value of every endogenous variable, aux included
    let last = vecf(&f["last_smoothed"]);
    let y0 = sm.last_state();
    let dr = sm.dr();
    let d0 = (0..endo.len()).map(|v| (y0[dr.inv_order_var[v]] + sm.ys()[v] - last[v]).abs()).fold(0.0, f64::max);
    println!("last smoothed state: {d0:e}");
    assert!(d0 < 1e-9);

    let var_list = names(&f["var_list"]);
    let vars: Vec<usize> = var_list.iter().map(|v| s.mf.model.endo_index(v).unwrap()).collect();
    let h = f["horizon"].as_u64().unwrap() as usize;
    let fc = stochastic_classical(dr, sm.ys(), &y0, h, &sm.resolved.q, &vars, &var_list, 0.9);
    let mut worst: f64 = 0.0;
    for (j, v) in var_list.iter().enumerate() {
        for (field, m) in [("Mean", &fc.mean), ("HPDinf", &fc.lower), ("HPDsup", &fc.upper)] {
            let r = vecf(&f["forecast"][field][v]);
            for t in 0..h {
                worst = worst.max((m[(t, j)] - r[t]).abs());
            }
        }
    }
    println!("forecast at the mode: max abs diff {worst:e}");
    assert!(worst < 1e-9);

    // forecast command, task 'simul': from the steady state, periods = 8, conf_sig 0.8
    let fc = stochastic_classical(dr, sm.ys(), &vec![0.0; endo.len()], 8, &sm.resolved.q, &vars, &var_list, 0.8);
    let mut worst: f64 = 0.0;
    for (j, v) in var_list.iter().enumerate() {
        for (field, m) in [("Mean", &fc.mean), ("HPDinf", &fc.lower), ("HPDsup", &fc.upper)] {
            let r = vecf(&f["forecast_simul"][field][v]);
            for t in 0..8 {
                worst = worst.max((m[(t, j)] - r[t]).abs());
            }
        }
    }
    println!("forecast command (simul): max abs diff {worst:e}");
    assert!(worst < 1e-9);
}

#[test]
fn shock_decomposition_at_mode() {
    let (s, r) = (
        setup().expect("reference/setup unavailable"),
        load("ref_shock_decomp.json").expect("reference/setup unavailable"),
    );
    let sm = dsge_smoother(&s.dsge, &s.data, &s.xm, false).unwrap();
    let size: Vec<usize> = vecf(&r["size"]).iter().map(|&x| x as usize).collect();
    let (nv, nc, gend) = (size[0], size[1], size[2]);
    for (key, init_state) in [("z", 0), ("z_init_state2", 2)] {
        let zr = vecf(&r[key]);
        let sd = shock_decomposition(sm.dr(), &sm.smoothed_deviation(), &sm.out.etahat, init_state);
        let mut worst: f64 = 0.0;
        for v in 0..nv {
            for c in 0..nc {
                for t in 0..gend {
                    let refv = zr[v + nv * (c + nc * t)];
                    worst = worst.max((sd.z[v][(c, t)] - refv).abs());
                }
            }
        }
        println!("shock decomposition ({key}): max abs diff {worst:e}");
        assert!(worst < 1e-9, "{key}: {worst}");
    }
}

fn compare(s: &PosteriorSummary, r: &Value, exact: bool) -> f64 {
    let mut worst: f64 = 0.0;
    for (i, name) in s.names.iter().enumerate() {
        let Some(_) = r["Mean"].get(name) else { continue };
        let n2 = s.mean.nrows();
        let width: Vec<f64> = {
            let lo = vecf(&r["HPDinf"][name]);
            let hi = vecf(&r["HPDsup"][name]);
            lo.iter().zip(&hi).map(|(a, b)| (b - a).abs().max(1e-12)).collect()
        };
        let mut fields: Vec<(&str, Vec<f64>)> = vec![("Mean", s.mean.column(i).iter().copied().collect())];
        if exact {
            fields.push(("Median", s.median.column(i).iter().copied().collect()));
            fields.push(("HPDinf", s.hpd_inf.column(i).iter().copied().collect()));
            fields.push(("HPDsup", s.hpd_sup.column(i).iter().copied().collect()));
        }
        for (f, ours) in fields {
            let theirs = vecf(&r[f][name]);
            for t in 0..n2 {
                let d = (ours[t] - theirs[t]).abs();
                worst = worst.max(if exact { d } else { d / width[t] });
            }
        }
        if exact {
            // deciles: 9 x n2 (rows = deciles)
            let dec = r["deciles"][name].as_array().unwrap();
            for (k, row) in dec.iter().enumerate() {
                let row = vecf(row);
                for t in 0..n2 {
                    worst = worst.max((s.deciles[k][(t, i)] - row[t]).abs());
                }
            }
        }
    }
    worst
}

#[test]
fn posterior_objects_on_dynare_draws() {
    let (s, r) = (
        setup().expect("reference/setup unavailable"),
        load("ref_posterior.json").expect("reference/setup unavailable"),
    );
    let draws: Vec<Vec<f64>> = r["draws"].as_array().unwrap().iter().map(vecf).collect();
    assert_eq!(names(&r["param_names"]), s.dsge.estimated_names);
    let m = &s.mf.model;
    let vars: Vec<usize> = (0..m.orig_endo_nbr).collect();
    let o = PosteriorSmootherOptions {
        smoother: true,
        filtered_vars: true,
        horizon: r["horizon"].as_u64().unwrap() as usize,
        mh_conf_sig: r["mh_conf_sig"].as_f64().unwrap(),
        vars: vars.clone(),
        var_names: vars.iter().map(|&v| m.endo_names[v].clone()).collect(),
        exo_names: m.exo_names.clone(),
        seed: 1,
        kernel_density: false,
        state_uncertainty: false,
        filter_covariance: false,
    };
    let t = std::time::Instant::now();
    let p = prior_posterior_statistics(&s.dsge, &s.data, &draws, &o).unwrap();
    println!("{} draws in {:.2}s ({} failed)", p.draws, t.elapsed().as_secs_f64(), p.failed);
    assert_eq!(p.failed, 0);
    for (s, key) in [
        (&p.smoothed_variables, "SmoothedVariables"),
        (&p.smoothed_shocks, "SmoothedShocks"),
        (&p.updated_variables, "UpdatedVariables"),
        (&p.filtered_variables, "FilteredVariables"),
        (&p.mean_forecast, "MeanForecast"),
    ] {
        let d = compare(s.as_ref().unwrap(), &r[key], true);
        println!("{key}: max abs diff {d:e}");
        assert!(d < 1e-8, "{key}: {d}");
    }
    // simulated shocks: same distribution, different random numbers
    let d = compare(p.point_forecast.as_ref().unwrap(), &r["PointForecast"], false);
    println!("PointForecast: max |mean diff| / HPD width {d:.3}");
    assert!(d < 0.25, "PointForecast mean differs by {d} HPD widths");
}
