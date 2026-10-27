//!  against Dynare 7.2 on SW07 (STAGE 9: mode file, first_obs=5, presample=8, diffuse
//! filter), reference values from reference/sw07/dump_conditional.m (ref_conditional.json),
//! which runs tests/data/sw07_conditional.mod:
//!   - conditional_forecast_paths parsing (constrained_vars_ / constrained_paths_);
//!   - conditional and unconditional forecast paths without free shocks (Dynare's
//!     conditional_forecasts.run with randn shadowed by zeros): to rounding, at the posterior
//!     mode (from the last smoothed state) and with parameter_set = calibration;
//!   - Monte Carlo means and bands (replic = 5000): statistically;
//!   - realtime_shock_decomposition (pools, vintages, forecast and conditional variants, and
//!     fast_realtime), initial_condition_decomposition, plot_shock_decomposition's
//!     transformations and shock_decomposition's forecast_type: to rounding;
//!   - the whole scenario through the driver.
mod common;

use std::path::{Path, PathBuf};

use dynare::analysis::realtime_decomposition::{self as rtd, DecompositionState, PlotSpec, Sources, Z3};
use dynare::data::Table;
use dynare::driver::{RunOptions, run};
use dynare::dsge::{DsgeModel, LikelihoodData};
use dynare::forecast::commands::{EstimationState, graph_options};
use dynare::forecast::conditional::{ConditionalForecast, ConstrainedPaths, conditional_forecast_command};
use dynare::kalman::KalmanOptions;
use dynare::modfile::{Command, ModFile, OptValue, Options, parse_mod_file};
use serde_json::Value;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn load(name: &str) -> Option<Value> {
    Some(common::json_ref(root().join("reference/sw07").join(name)))
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

/// Temp dir with sw07.mod, the data, the mode file and sw07_conditional.mod.
fn temp_dir(tag: &str) -> Option<PathBuf> {
    let refdir = root().join("reference/sw07");
    let mode = refdir.join("sw07/Output/sw07_mode.mat");
    assert!(mode.exists(), "missing reference {}", mode.display());
    let dir = std::env::temp_dir().join(format!("dynare_rs_cond_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(dir.join("sw07/Output")).unwrap();
    for f in ["sw07.mod", "sw07_obs.csv"] {
        std::fs::copy(refdir.join(f), dir.join(f)).unwrap();
    }
    std::fs::copy(refdir.join("sw07/Output/sw07_mode.mat"), dir.join("sw07/Output/sw07_mode.mat")).unwrap();
    std::fs::copy(root().join("tests/data/sw07_conditional.mod"), dir.join("sw07_conditional.mod")).unwrap();
    Some(dir)
}

struct Setup {
    dir: PathBuf,
    mf: ModFile,
    st: EstimationState,
    r: Value,
    /// declaration indices of R.vars
    iv: Vec<usize>,
}

fn setup(tag: &str) -> Option<Setup> {
    let r = load("ref_conditional.json")?;
    let base = load("ref.json")?;
    let dir = temp_dir(tag)?;
    let mf = parse_mod_file(&dir.join("sw07_conditional.mod"), &[]).unwrap();
    let xm = vecf(&base["xparam_mode"]);
    let dsge = DsgeModel::new(&mf, Some(&xm)).unwrap();
    let table = Table::read_csv(&dir.join("sw07_obs.csv")).unwrap();
    let sample = table.sample(&mf.varobs, 5, None).unwrap();
    let nobs = sample.nrows();
    let data = LikelihoodData::new(&sample, KalmanOptions { presample: 8, lik_init: 3, ..Default::default() });
    let dates = table.dates.map(|d| d[4..4 + nobs].to_vec());
    let mut graphs = graph_options(&Vec::new(), &dir, "sw07_conditional");
    graphs.enabled = false;
    let st = EstimationState {
        dsge,
        data,
        dates,
        first_obs: 5,
        var_list: ["dy", "pinfobs", "robs", "pinf4", "y", "lab", "pistar", "r"].iter().map(|s| s.to_string()).collect(),
        mode: xm,
        posterior_mean: None,
        posterior_median: None,
        graphs,
        mean_forecast: None,
        smoothed_mean: None,
    };
    let iv = r["vars"].as_array().unwrap().iter().map(|v| mf.model.endo_index(v.as_str().unwrap()).unwrap()).collect();
    Some(Setup { dir, mf, st, r, iv })
}

fn paths(mf: &ModFile) -> ConstrainedPaths {
    let block = mf
        .commands
        .iter()
        .find_map(|c| if let Command::ConditionalForecastPaths(p) = c { Some(p.clone()) } else { None })
        .unwrap();
    ConstrainedPaths::from_block(mf, &block).unwrap()
}

fn opts(kv: &[(&str, OptValue)]) -> Options {
    kv.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

fn s(x: &str) -> OptValue {
    OptValue::Str(x.into())
}

fn cond_opts(set: &str, replic: f64) -> Options {
    opts(&[
        ("parameter_set", s(set)),
        ("controlled_varexo", OptValue::List(vec![s("em"), s("e_lr")])),
        ("periods", OptValue::Num(12.0)),
        ("replic", OptValue::Num(replic)),
        ("conf_sig", OptValue::Num(0.8)),
    ])
}

/// max |rust - ref| over the variables of a MATLAB struct of per-variable columns.
fn worst_struct(cf: &ConditionalForecast, m: &dynare::linalg::Mat, refs: &Value) -> f64 {
    let mut worst: f64 = 0.0;
    for (j, name) in cf.names.iter().enumerate() {
        let r = vecf(&refs[name]);
        assert_eq!(r.len(), m.nrows(), "{name}");
        for t in 0..m.nrows() {
            worst = worst.max((m[(t, j)] - r[t]).abs());
        }
    }
    worst
}

#[test]
fn constrained_paths_parsed() {
    let su = setup("parse").expect("reference/setup unavailable");
    let p = paths(&su.mf);
    let rv: Vec<usize> = vecf(&su.r["constrained_vars"]).iter().map(|&v| v as usize - 1).collect();
    assert_eq!(p.vars, rv);
    let rp = vecf(&su.r["constrained_paths"]); // 2 x 8, row-major in the JSON
    for i in 0..2 {
        for t in 0..8 {
            // 0.8+0.05 is written as 0.85000000000000009, which serde_json's default (not
            // correctly rounded) parser may read one ulp off: allow that ulp, nothing more
            let (a, b) = (p.paths[(i, t)], rp[i * 8 + t]);
            assert!((a - b).abs() <= f64::EPSILON * b.abs(), "path {i} {t}: {a} vs {b}");
        }
    }
}

#[test]
fn conditional_paths_match_dynare() {
    let su = setup("paths").expect("reference/setup unavailable");
    let p = paths(&su.mf);
    let out = su.dir.join("out");
    for (set, key) in [("posterior_mode", "cf0"), ("calibration", "cf0_calibration")] {
        let cf =
            conditional_forecast_command(&su.mf, Some(&su.st), &cond_opts(set, 20.0), Some(&p), 1, &out, "t").unwrap();
        let r = &su.r[key];
        let c = worst_struct(&cf, &cf.cond.path, &r["cond"]["Mean"]);
        let u = worst_struct(&cf, &cf.uncond.path, &r["uncond"]["Mean"]);
        let mut e: f64 = 0.0;
        for (j, n) in cf.instrument_names.iter().enumerate() {
            let rr = vecf(&r["controlled_exo_variables"]["Mean"][n]);
            for t in 0..rr.len() {
                e = e.max((cf.exo.path[(t, j)] - rr[t]).abs());
            }
        }
        println!("{set}: conditional path {c:e}, unconditional path {u:e}, controlled shocks {e:e}");
        assert!(c < 1e-9 && u < 1e-9 && e < 1e-9);
        // the constrained variables follow their paths in every draw
        for (i, &v) in p.vars.iter().enumerate() {
            let j = cf.column(v).unwrap();
            for t in 0..8 {
                assert!((cf.cond.lower[(t + 1, j)] - p.paths[(i, t)]).abs() < 1e-9);
                assert!((cf.cond.upper[(t + 1, j)] - p.paths[(i, t)]).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn monte_carlo_bands_statistically() {
    let su = setup("mc").expect("reference/setup unavailable");
    let p = paths(&su.mf);
    let cf = conditional_forecast_command(
        &su.mf,
        Some(&su.st),
        &cond_opts("posterior_mode", 5000.0),
        Some(&p),
        7,
        &su.dir,
        "t",
    )
    .unwrap();
    let r = &su.r["cf"];
    // per variable/period: sd from the Rust 80% band (width = 2*1.2816 sd), MC error of the
    // mean ~ sd/sqrt(5000) per program, of a 10% quantile ~ 0.024 sd.
    let (mut zmax, mut qmax) = (0.0f64, 0.0f64);
    for (part, d) in [("cond", &cf.cond), ("uncond", &cf.uncond)] {
        for (j, name) in cf.names.iter().enumerate() {
            let m = vecf(&r[part]["Mean"][name]);
            let ci = vecf(&r[part]["ci"][name]); // 2 x (periods+1), row-major
            let h = m.len();
            for t in 1..h {
                let sd = (d.upper[(t, j)] - d.lower[(t, j)]) / (2.0 * 1.2816);
                if sd < 1e-8 {
                    assert!((d.mean[(t, j)] - m[t]).abs() < 1e-8, "{part} {name} {t}");
                    continue;
                }
                zmax = zmax.max((d.mean[(t, j)] - m[t]).abs() / (sd * (2.0f64 / 5000.0).sqrt()));
                qmax = qmax.max((d.lower[(t, j)] - ci[t]).abs() / sd).max((d.upper[(t, j)] - ci[h + t]).abs() / sd);
            }
        }
    }
    println!("MC: max |mean diff| = {zmax:.2} standard errors, max |band diff| = {qmax:.3} sd");
    assert!(zmax < 5.5, "means differ by {zmax} s.e.");
    assert!(qmax < 0.2, "bands differ by {qmax} sd");
}

/// Ref flattening: reshape(z(iv,:,:),[],1) with size [nv, c, t].
fn cmp_z3(z: &Z3, rows: &[usize], r: &Value, size: &Value, what: &str) -> f64 {
    let sz: Vec<usize> = vecf(size).iter().map(|&x| x as usize).collect();
    let rr = vecf(r);
    let (nv, c, t) = (sz[0], sz[1], *sz.get(2).unwrap_or(&1));
    assert_eq!((c, t), (z.c, z.t), "{what}: size");
    let mut worst: f64 = 0.0;
    for i in 0..t {
        for k in 0..c {
            for (a, &v) in rows.iter().enumerate().take(nv) {
                let x = z.at(v, k, i);
                let y = rr[a + nv * (k + c * i)];
                if x.is_nan() && y.is_nan() {
                    continue;
                }
                worst = worst.max((x - y).abs());
            }
        }
    }
    println!("{what}: {worst:e}");
    if worst > 1e-9 {
        for i in 0..t {
            let mut wi: f64 = 0.0;
            for k in 0..c {
                for (a, &v) in rows.iter().enumerate().take(nv) {
                    let d = (z.at(v, k, i) - rr[a + nv * (k + c * i)]).abs();
                    if d.is_finite() {
                        wi = wi.max(d);
                    }
                }
            }
            if wi > 1e-9 {
                println!("  period {}: {wi:e}", i + 1);
            }
        }
    }
    worst
}

fn realtime_state(su: &Setup, extra: &[(&str, OptValue)]) -> DecompositionState {
    let mut sc = DecompositionState::default();
    let vars: Vec<String> = su.r["vars"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let mut o = opts(&[("parameter_set", s("posterior_mode"))]);
    o.extend(opts(extra));
    rtd::realtime_shock_decomposition_command(
        &su.mf,
        Some(&su.st),
        &o,
        &vars,
        &mut sc,
        &su.dir,
        "t",
        &su.dir.join("out"),
    )
    .unwrap();
    sc
}

#[test]
fn realtime_decomposition_matches_dynare() {
    let su = setup("rt").expect("reference/setup unavailable");
    let sc = realtime_state(
        &su,
        &[("forecast", OptValue::Num(4.0)), ("save_realtime", OptValue::Vec(vec![80.0, 86.0, 90.0]))],
    );
    let (rt, set) = sc.realtime.as_ref().unwrap();
    assert_eq!(set, "posterior_mode");
    let r = &su.r["realtime"];
    let mut w: f64 = 0.0;
    w = w.max(cmp_z3(&rt.pool, &su.iv, &r["pool"], &r["pool_size"], "realtime pool"));
    w = w.max(cmp_z3(
        &rt.conditional_pool,
        &su.iv,
        &r["conditional_pool"],
        &r["conditional_pool_size"],
        "conditional pool",
    ));
    w = w.max(cmp_z3(
        rt.forecast_pool.as_ref().unwrap(),
        &su.iv,
        &r["forecast_pool"],
        &r["forecast_pool_size"],
        "forecast pool",
    ));
    for t in [80usize, 86, 90] {
        let f = format!("time_{t}");
        w = w.max(cmp_z3(&rt.vintages[&t], &su.iv, &r[&f], &r[format!("{f}_size")], &format!("realtime {f}")));
        let g = format!("forecast_{f}");
        w = w.max(cmp_z3(&rt.forecast_vintages[&t], &su.iv, &r[&g], &r[format!("{g}_size")], &g));
    }
    for t in [80usize, 86] {
        let f = format!("time_{t}");
        let g = format!("conditional_{f}");
        w = w.max(cmp_z3(&rt.conditional_vintages[&t], &su.iv, &r[&g], &r[format!("{g}_size")], &g));
        let g = format!("realtime_conditional_{f}");
        w = w.max(cmp_z3(&rt.realtime_conditional_vintages[&t], &su.iv, &r[&g], &r[format!("{g}_size")], &g));
    }
    assert!(w < 1e-9, "realtime decomposition: {w:e}");

    // plot_shock_decomposition transformations (get_decomp_only), on top of
    // shock_decomposition(parameter_set=posterior_mode)
    let mut sc = sc;
    let vars: Vec<String> = su.r["vars"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let o = opts(&[("parameter_set", s("posterior_mode")), ("nograph", OptValue::Flag)]);
    rtd::shock_decomposition_command(&su.mf, Some(&su.st), &o, &vars, &mut sc, &su.dir, "t", &su.dir.join("out"))
        .unwrap();
    let cases: &[(&str, &[(&str, OptValue)])] = &[
        (
            "rt1_v86_yoy_flip_main",
            &[
                ("realtime", OptValue::Num(1.0)),
                ("vintage", OptValue::Num(86.0)),
                ("type", OptValue::Str(String::new())),
                ("flip", OptValue::Flag),
                ("use_shock_groups", OptValue::Str(String::new())),
            ],
        ),
        ("rt3_v86", &[("realtime", OptValue::Num(3.0)), ("vintage", OptValue::Num(86.0))]),
        ("rt2_v86", &[("realtime", OptValue::Num(2.0)), ("vintage", OptValue::Num(86.0))]),
        ("rt2_pool", &[("realtime", OptValue::Num(2.0)), ("use_shock_groups", OptValue::Str(String::new()))]),
        (
            "rt1_pool_diff",
            &[("realtime", OptValue::Num(1.0)), ("type", OptValue::Str(String::new())), ("diff", OptValue::Flag)],
        ),
        ("rt0_aoa", &[("type", OptValue::Str(String::new()))]),
        (
            "rt0_yoy_main",
            &[
                ("type", OptValue::Str(String::new())),
                ("flip", OptValue::Flag),
                ("use_shock_groups", OptValue::Str(String::new())),
            ],
        ),
    ];
    let names: Vec<String> = vars.clone();
    let mut w: f64 = 0.0;
    for (name, kv) in cases {
        // the string values that depend on the case
        let fixed: Vec<(&str, OptValue)> = kv
            .iter()
            .map(|(k, v)| {
                let v = match (*k, v) {
                    ("type", _) => s(if name.contains("yoy") {
                        "yoy"
                    } else if name.contains("aoa") {
                        "aoa"
                    } else {
                        "qoq"
                    }),
                    ("use_shock_groups", _) => s("main"),
                    _ => v.clone(),
                };
                (*k, v)
            })
            .collect();
        let spec = PlotSpec::from_options(&su.mf, &opts(&fixed)).unwrap();
        let src = Sources {
            shock_decomposition: sc.shock_decomposition.as_ref(),
            initval: None,
            forecast_sd: &sc.forecast_sd,
            realtime: sc.realtime.as_ref(),
            sd_forecast: sc.sd_forecast,
            nobs: su.st.data.y.ncols(),
        };
        let p = rtd::transform(&spec, &src, &su.iv, &names, &su.mf.model.exo_names).unwrap();
        let rows: Vec<usize> = (0..su.iv.len()).collect();
        w = w.max(cmp_z3(&p.z, &rows, &su.r["plot"][name], &su.r["plot"][format!("{name}_size")], name));
        assert!(!p.figures(&spec, spec.realtime, spec.vintage).is_empty() || name.contains("diff"));
    }
    // aoa with the data's (daily, as Dynare reads ISO dates) first date
    let mut spec = PlotSpec::from_options(&su.mf, &opts(&[("type", s("aoa"))])).unwrap();
    spec.initial_date = rtd::Period::parse(&su.st.dates.as_ref().unwrap()[0]);
    let src = Sources {
        shock_decomposition: sc.shock_decomposition.as_ref(),
        initval: None,
        forecast_sd: &sc.forecast_sd,
        realtime: sc.realtime.as_ref(),
        sd_forecast: sc.sd_forecast,
        nobs: su.st.data.y.ncols(),
    };
    let e = rtd::transform(&spec, &src, &su.iv, &names, &su.mf.model.exo_names).unwrap_err();
    assert_eq!(e.to_string(), su.r["plot"]["rt0_aoa_dated"].as_str().unwrap());
    assert!(w < 1e-9, "plot_shock_decomposition transformations: {w:e}");
}

#[test]
fn fast_realtime_matches_dynare() {
    let su = setup("fast").expect("reference/setup unavailable");
    let sc = realtime_state(
        &su,
        &[
            ("forecast", OptValue::Num(0.0)),
            ("save_realtime", OptValue::Vec(vec![70.0, 85.0])),
            ("fast_realtime", OptValue::Vec(vec![60.0, 80.0])),
        ],
    );
    let (rt, _) = sc.realtime.as_ref().unwrap();
    let r = &su.r["fast"];
    let mut w: f64 = 0.0;
    w = w.max(cmp_z3(&rt.pool, &su.iv, &r["pool"], &r["pool_size"], "fast pool"));
    w = w.max(cmp_z3(&rt.vintages[&70], &su.iv, &r["time_70"], &r["time_70_size"], "fast time_70"));
    w = w.max(cmp_z3(&rt.vintages[&85], &su.iv, &r["time_85"], &r["time_85_size"], "fast time_85"));
    w = w.max(cmp_z3(&rt.conditional_pool, &su.iv, &r["conditional_pool"], &r["pool_size"], "fast conditional pool"));
    assert!(w < 1e-9, "fast realtime: {w:e}");
}

#[test]
fn initval_and_forecast_type_match_dynare() {
    let su = setup("iv").expect("reference/setup unavailable");
    let vars: Vec<String> = su.r["vars"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let mut sc = DecompositionState::default();
    let out = su.dir.join("out");
    rtd::initial_condition_decomposition_command(&su.mf, Some(&su.st), &Vec::new(), &vars, &mut sc, &su.dir, "t", &out)
        .unwrap();
    let w = cmp_z3(
        &sc.initval.as_ref().unwrap().z,
        &su.iv,
        &su.r["initval"],
        &su.r["initval_size"],
        "initval decomposition",
    );
    assert!(w < 1e-9);

    // forecast_type = unconditional at the calibration (= M_.params after the estimation),
    // over the estimation's forecast at the mode (oo_.forecast)
    let o = opts(&[
        ("parameter_set", s("calibration")),
        ("forecast_type", s("unconditional")),
        ("nograph", OptValue::Flag),
    ]);
    rtd::shock_decomposition_command(&su.mf, Some(&su.st), &o, &vars, &mut sc, &su.dir, "t", &out).unwrap();
    let f = &su.r["forecast_type"];
    let w = cmp_z3(
        &sc.forecast_sd["unconditional"].z,
        &su.iv,
        &f["unconditional_calibration"],
        &f["unconditional_calibration_size"],
        "forecast_type=unconditional",
    );
    assert!(w < 1e-9);
    // Dynare 7.2 fails on forecast_type=conditional unless every shock is controlled; we take
    // the uncontrolled ones at their mean (zero).
    assert!(f["conditional_posterior_mode"].as_str().unwrap().contains("Unrecognized field"));

    // squeeze keeps only the listed rows; plotting another variable is an error
    rtd::squeeze_shock_decomposition(&su.mf, &vars[..1], &mut sc).unwrap();
    assert_eq!(sc.initval.as_ref().unwrap().z.n, 1);
    let e = rtd::plot_shock_decomposition_command(
        &su.mf,
        Some(&su.st),
        &Vec::new(),
        &vars[1..2],
        &mut sc,
        &su.dir,
        "t",
        &out,
    );
    assert!(e.is_err());
}

#[test]
fn scenario_through_the_driver() {
    let (dir, r) = (
        temp_dir("driver").expect("reference/setup unavailable"),
        load("ref_conditional.json").expect("reference/setup unavailable"),
    );
    run(&dir.join("sw07_conditional.mod"), &RunOptions { quiet: true, seed: 3, ..Default::default() }).unwrap();
    let out = dir.join("sw07_conditional/Output");
    let rows: Vec<Vec<String>> = std::fs::read_to_string(out.join("sw07_conditional_conditional_forecast.csv"))
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| l.split(',').map(str::to_string).collect())
        .collect();
    // cond_path of every variable = Dynare's deterministic path
    let mut worst: f64 = 0.0;
    for row in &rows {
        let h: usize = row[1].parse().unwrap();
        let refv = vecf(&r["cf0"]["cond"]["Mean"][&row[2]])[h];
        worst = worst.max((row[9].parse::<f64>().unwrap() - refv).abs());
    }
    println!("driver: conditional path {worst:e}");
    assert!(worst < 1e-9);
    // the constrained policy rate in the last controlled quarter: the path exactly, the Monte
    // Carlo mean to rounding (a mean of 5000 copies of 1.3)
    let last = rows.iter().find(|x| x[0] == "2027-09-30" && x[2] == "robs").unwrap();
    assert_eq!(last[9].parse::<f64>().unwrap(), 1.3);
    assert!((last[3].parse::<f64>().unwrap() - 1.3).abs() < 1e-9);
    for f in [
        "sw07_conditional_conditional_forecast_shocks.csv",
        "sw07_conditional_realtime_shock_decomposition_pool.csv",
        "sw07_conditional_realtime_shock_decomposition_time_86.csv",
        "sw07_conditional_realtime_conditional_shock_decomposition_time_86.csv",
        "sw07_conditional_initval_decomposition.csv",
    ] {
        assert!(out.join(f).exists(), "{f}");
    }
    // the plot_shock_decomposition CSVs (names follow Dynare's figure names; standalone plot
    // commands have no initial date: vintage 86 of 1Q1 is 22Q2)
    let names: Vec<String> =
        std::fs::read_dir(&out).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    assert!(
        names
            .iter()
            .any(|n| n
                == "sw07_conditional_shock_decomposition_yoy_test_realtime_22Q2_group_main_flip_posterior_mode.csv"),
        "{names:?}"
    );
    assert!(
        names.iter().any(|n| n == "sw07_conditional_shock_decomposition_4step_ahead_forecast_86Y_posterior_mode.csv"),
        "{names:?}"
    );
}
