//! OccBin estimation against Dynare 7.2 (reference/occbin_est/, dump_occbin_est.m): the
//! piecewise Kalman filter likelihood (with and without filter_use_relaxation) and the
//! inversion-filter likelihood of a ZLB New Keynesian model on data simulated from the
//! piecewise-linear model, at six parameter points; the filter's regimes; the IVF's
//! filtered states and shocks.
mod common;

use std::path::{Path, PathBuf};

use dynare::data::Table;
use dynare::dsge::{DsgeModel, LikelihoodData};
use dynare::kalman::KalmanOptions;
use dynare::modfile::{Command, ModFile, parse_mod_file};
use dynare::occbin::estimation::{OccbinEstimation, OccbinPosterior};
use dynare::occbin::{OccbinCommand, Session};
use serde_json::Value;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/occbin_est")
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

fn reference() -> Option<Value> {
    let text = common::read_ref(dir().join("ref_occbin_est.json"));
    Some(serde_json::from_str(&text).unwrap())
}

struct Setup {
    _mf: ModFile,
    dsge: DsgeModel,
    data: LikelihoodData,
    occ: OccbinEstimation,
}

fn setup(defines: &[(&str, &str)], x_init: &[f64]) -> Setup {
    let defs: Vec<(String, String)> = defines.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
    let mf = parse_mod_file(&dir().join("nk_est.mod"), &defs).unwrap();
    let mut s = Session::new(&mf, "nk_est");
    for c in &mf.commands {
        if let Command::Occbin(OccbinCommand::Setup(o)) = c {
            s.setup(&mf, o).unwrap();
        }
    }
    let dsge = DsgeModel::new(&mf, Some(x_init)).unwrap();
    let table = Table::read_csv(&dir().join("nk_est_data.csv")).unwrap();
    let sample = table.sample(&mf.varobs, 1, None).unwrap();
    let data = LikelihoodData::new(&sample, KalmanOptions { lik_init: 1, ..Default::default() });
    let occ = OccbinEstimation::new(&mf, &s, &dsge, &data).unwrap().unwrap();
    Setup { _mf: mf, dsge, data, occ }
}

/// MATLAB regime/regimestart (scalar or array) -> Vec.
fn ints(v: &Value) -> Vec<usize> {
    vecf(v).iter().map(|&x| x as usize).collect()
}

fn check_likelihood(variant: &str, defines: &[(&str, &str)]) {
    let r = reference().expect("reference/setup unavailable");
    let r = &r[variant];
    let x_init = vecf(&r["xparam_init"]);
    let s = setup(defines, &x_init);
    let names: Vec<String> =
        r["estim_param_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().into()).collect();
    assert_eq!(s.dsge.estimated_names, names);
    let post = OccbinPosterior { dsge: &s.dsge, data: &s.data, bounds: &s.dsge.bounds, occ: &s.occ };
    for (k, p) in r["points"].as_array().unwrap().iter().enumerate() {
        let x = vecf(&p["x"]);
        let t = std::time::Instant::now();
        let e = post.evaluate(&x);
        let secs = t.elapsed().as_secs_f64();
        let want = p["fval"].as_f64().unwrap();
        match e {
            Ok(e) => {
                let rel = ((e.fval - want) / want).abs();
                println!(
                    "{variant} point {k}: fval {:.10} vs Dynare {want:.10} (rel {rel:.1e}; {:.1} ms vs MATLAB {:.0} ms)",
                    e.fval,
                    secs * 1e3,
                    p["seconds"].as_f64().unwrap() * 1e3
                );
                if let Some(fr) = p.get("filter_regime") {
                    let fs = &p["filter_regimestart"];
                    let fr = fr.as_array().unwrap();
                    let mut bad = 0;
                    for (t, g) in e.regimes.iter().enumerate() {
                        let (rw, sw) = (ints(&fr[t]), ints(&fs[t]));
                        let rg: Vec<usize> = g[0].regime.iter().map(|&b| b as usize).collect();
                        if rg != rw || g[0].start != sw {
                            if bad < 5 {
                                println!(
                                    "  period {}: regime {:?} {:?} vs Dynare {:?} {:?}",
                                    t + 1,
                                    rg,
                                    g[0].start,
                                    rw,
                                    sw
                                );
                            }
                            bad += 1;
                        }
                    }
                    assert_eq!(bad, 0, "{variant} point {k}: {bad} filter regimes differ");
                }
                assert!(rel < 1e-8, "{variant} point {k}");
            }
            Err(f) => panic!("{variant} point {k}: info {} (Dynare fval {want}, info {})", f.info, p["info"]),
        }
    }
}

#[test]
fn pkf_likelihood() {
    check_likelihood("pkf", &[]);
}

#[test]
fn pkf_likelihood_relaxation() {
    check_likelihood("relax", &[("RELAX", "1")]);
}

#[test]
fn ivf_likelihood() {
    check_likelihood("ivf", &[("IVF", "1")]);
}

/// occbin.irf (GIRFs from the steady state, given and default shock sizes, both signs) and
/// occbin.forecast (point forecast from an initial state with expected shocks).
#[test]
fn occbin_irf_and_forecast() {
    let text = common::read_ref(dir().join("ref_occbin_irf.json"));
    let r: Value = serde_json::from_str(&text).unwrap();
    let mf = parse_mod_file(&dir().join("nk_est.mod"), &[("GEN".into(), "1".into())]).unwrap();
    let mut s = Session::new(&mf, "nk_est");
    let names = mf.model.endo_names.clone();
    let mut worst: f64 = 0.0;
    for (shocks, size, key) in [(vec!["eps_d", "eps_s"], Some(vec![0.03, 0.002]), "irf"), (vec!["eps_m"], None, "irf2")]
    {
        let mut o = dynare::occbin::irf::IrfOptions::new(&mf);
        o.exo_names = shocks.iter().map(|x| x.to_string()).collect();
        o.shocksize = size;
        o.periods = 20;
        let irfs = s.irf(&mf, &o, None).unwrap();
        assert_eq!(irfs.len(), 2 * shocks.len());
        for g in &irfs {
            for (j, v) in names.iter().enumerate() {
                let field = format!("{v}_{}_{}", g.shock, g.sign);
                for (kind, m) in [("piecewise", &g.piecewise), ("linear", &g.linear)] {
                    let want = vecf(&r[format!("{key}_{kind}")][&field]);
                    assert_eq!(want.len(), m.nrows(), "{field}");
                    for (t, w) in want.iter().enumerate() {
                        worst = worst.max((m[(t, j)] - w).abs());
                    }
                }
            }
        }
    }
    println!("occbin.irf: max |diff| {worst:.1e}");
    assert!(worst < 1e-12);
    let hist = vecf(&r["endo_histval"]);
    let fo = dynare::occbin::irf::ForecastOptions {
        shocks0: vec![("eps_d".into(), [vec![-0.01, -0.005], vec![0.0; 13]].concat())],
        ..Default::default()
    };
    let f = s.forecast(&mf, &fo, &hist, 15).unwrap();
    let mut worst: f64 = 0.0;
    for (j, v) in names.iter().enumerate() {
        for (kind, m) in [("forecast_piecewise", &f.piecewise), ("forecast_linear", &f.linear)] {
            for (t, w) in vecf(&r[kind][v]).iter().enumerate() {
                worst = worst.max((m[(t, j)] - w).abs());
            }
        }
    }
    println!("occbin.forecast: max |diff| {worst:.1e}");
    assert!(worst < 1e-12);
    assert_eq!(regimes_match(&f.regimes, &r["forecast_regime"], &r["forecast_regimestart"]), 0);
}

/// The posterior kernel at Dynare's posterior mode (mode_compute=4, ref_occbin_mode.json), with
/// both filters. (csminwel itself follows Dynare's iterates for 4 iterations on the PKF, then
/// drifts on the non-smooth likelihood; both stop within 1e-3 of each other's fval.)
#[test]
fn likelihood_at_dynare_mode() {
    let text = common::read_ref(dir().join("ref_occbin_mode.json"));
    let m: Value = serde_json::from_str(&text).unwrap();
    let r = reference().unwrap();
    for (variant, defs) in [("pkf", vec![]), ("ivf", vec![("IVF", "1")])] {
        let s = setup(&defs, &vecf(&r[variant]["xparam_init"]));
        let post = OccbinPosterior { dsge: &s.dsge, data: &s.data, bounds: &s.dsge.bounds, occ: &s.occ };
        let x = vecf(&m[variant]["xparam1"]);
        let f = post.evaluate(&x).unwrap().fval;
        let want = m[variant]["fval"].as_f64().unwrap();
        println!("{variant} at Dynare's mode: {f:.10} vs {want:.10}");
        assert!(((f - want) / want).abs() < 1e-9);
    }
}

fn mat_rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(vecf).collect()
}

fn max_diff(got: &dynare::linalg::Mat, rows: &[usize], want: &[Vec<f64>]) -> f64 {
    let mut d: f64 = 0.0;
    for (i, w) in want.iter().enumerate() {
        for (j, v) in w.iter().enumerate() {
            d = d.max((got[(rows[i], j)] - v).abs());
        }
    }
    d
}

fn regimes_match(got: &[Vec<dynare::occbin::Regime>], reg: &Value, start: &Value) -> usize {
    let (reg, start) = (reg.as_array().unwrap(), start.as_array().unwrap());
    assert_eq!(got.len(), reg.len(), "number of regime periods");
    let mut bad = 0;
    for (t, g) in got.iter().enumerate() {
        let rg: Vec<usize> = g[0].regime.iter().map(|&b| b as usize).collect();
        if rg != ints(&reg[t]) || g[0].start != ints(&start[t]) {
            if bad < 5 {
                println!(
                    "  period {}: {:?} {:?} vs Dynare {:?} {:?}",
                    t + 1,
                    rg,
                    g[0].start,
                    ints(&reg[t]),
                    ints(&start[t])
                );
            }
            bad += 1;
        }
    }
    bad
}

/// occbin.DSGE_smoother at the six points: realtime smoother converged (points 1, 2, 4, 5),
/// conditional smoother + periodic regimes (point 3, warning 321), regime iteration converged
/// to a regime different from the realtime one (point 6, warning 320).
#[test]
fn occbin_smoother() {
    let r = reference().expect("reference/setup unavailable");
    let text = common::read_ref(dir().join("ref_occbin_smoother.json"));
    let sref: Value = serde_json::from_str(&text).unwrap();
    let x_init = vecf(&r["pkf"]["xparam_init"]);
    let s = setup(&[], &x_init);
    let pts = r["pkf"]["points"].as_array().unwrap();
    for (k, (p, want)) in pts.iter().zip(sref["pkf"]["points"].as_array().unwrap()).enumerate() {
        let x = vecf(&p["x"]);
        let t = std::time::Instant::now();
        let sm = dynare::occbin::smoother::occbin_smoother(&s.dsge, &s.data, &s.occ, &x).unwrap();
        let secs = t.elapsed().as_secs_f64();
        let a = mat_rows(&want["alphahat"]);
        let e = mat_rows(&want["etahat"]);
        let all: Vec<usize> = (0..a.len().max(e.len())).collect();
        let (da, de) = (max_diff(&sm.alphahat, &all, &a), max_diff(&sm.etahat, &all, &e));
        println!(
            "smoother point {k}: |d alphahat| {da:.1e}, |d etahat| {de:.1e}, error {} warning {} iterations {} (Dynare {} {} {}); {:.1} ms vs MATLAB {:.0} ms",
            sm.error_flag,
            sm.warning_flag,
            sm.iterations,
            want["error_flag"],
            want["warning_flag"],
            want["iterations"],
            secs * 1e3,
            want["seconds"].as_f64().unwrap_or(f64::NAN) * 1e3
        );
        let bad = regimes_match(&sm.regime_history, &want["regime"], &want["regimestart"]);
        let bad_rt =
            regimes_match(&sm.realtime_regime_history, &want["realtime_regime"], &want["realtime_regimestart"]);
        assert_eq!((bad, bad_rt), (0, 0), "point {k}: smoothed / realtime regimes differ");
        assert_eq!(sm.error_flag as f64, want["error_flag"].as_f64().unwrap());
        assert_eq!(sm.warning_flag as f64, want["warning_flag"].as_f64().unwrap());
        assert_eq!(sm.iterations as f64, want["iterations"].as_f64().unwrap());
        assert!(da < 1e-9 && de < 1e-9, "point {k}: smoothed variables/shocks differ");
    }
}

#[test]
fn ivf_smoother() {
    let r = reference().expect("reference/setup unavailable");
    let r = &r["ivf"];
    let x_init = vecf(&r["xparam_init"]);
    let s = setup(&[("IVF", "1")], &x_init);
    let post = OccbinPosterior { dsge: &s.dsge, data: &s.data, bounds: &s.dsge.bounds, occ: &s.occ };
    let order_var: Vec<usize> = vecf(&r["order_var"]).iter().map(|&v| v as usize - 1).collect();
    for (k, p) in r["points"].as_array().unwrap().iter().enumerate() {
        let e = post.evaluate(&vecf(&p["x"])).unwrap();
        let (states, innov) = e.ivf_states.unwrap();
        let a = mat_rows(&p["atT"]);
        let inn = mat_rows(&p["innov"]);
        let all: Vec<usize> = (0..inn.len()).collect();
        let d = max_diff(&states, &order_var, &a).max(max_diff(&innov, &all, &inn));
        let bad = regimes_match(&e.regimes, &p["regime"], &p["regimestart"]);
        println!("ivf point {k}: max |d states, shocks| {d:.1e}, {bad} regimes differ");
        assert!(d < 1e-9 && bad == 0);
    }
}

/// occbin_setup options that Dynare 7.2's +occbin/setup.m does not forward
/// (smoother_max_check_ahead_periods, smoother_plot, smoother_first_period_occbin_update,
/// smoother_max_number_of_iterations, posterior_importance_sampling*): nk_est.mod -DSETUP_OPTS
/// sets them to values that would change the smoother; Dynare keeps the defaults
/// (dump_occbin_setup_opts.m), and so does the port.
#[test]
fn occbin_setup_unforwarded_options() {
    let r = reference().expect("reference/setup unavailable");
    let want: Value = serde_json::from_str(&common::read_ref(dir().join("ref_occbin_setup_opts.json"))).unwrap();
    assert_eq!(want["smoother"]["max_number_of_iterations"].as_f64(), Some(10.0));
    assert_eq!(want["smoother"]["first_period_occbin_update"].as_f64(), Some(1.0));
    assert!(want["smoother"]["max_check_ahead_periods"].is_null()); // Inf
    assert_eq!(want["posterior_importance_sampling"]["status"].as_bool(), Some(false));
    let x_init = vecf(&r["pkf"]["xparam_init"]);
    let s = setup(&[("SETUP_OPTS", "1")], &x_init);
    let d = dynare::occbin::estimation::SmootherOptions::default();
    assert_eq!(s.occ.smoother.max_number_of_iterations, d.max_number_of_iterations);
    assert_eq!(s.occ.smoother.first_period_occbin_update, d.first_period_occbin_update);
    assert_eq!(s.occ.smoother.max_check_ahead_periods, None);
    let pts = r["pkf"]["points"].as_array().unwrap();
    for (k, (p, w)) in pts.iter().zip(want["points"].as_array().unwrap()).enumerate() {
        let sm = dynare::occbin::smoother::occbin_smoother(&s.dsge, &s.data, &s.occ, &vecf(&p["x"])).unwrap();
        assert_eq!(sm.iterations as f64, w["iterations"].as_f64().unwrap(), "point {k}: iterations");
        assert_eq!(sm.error_flag as f64, w["error_flag"].as_f64().unwrap());
        assert_eq!(regimes_match(&sm.regime_history, &w["regime"], &w["regimestart"]), 0, "point {k}: regimes");
    }
}
