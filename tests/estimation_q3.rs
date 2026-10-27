//!  (estimation options) against Dynare 7.2 (reference/q3/dump_q3.m, q3_nk.mod):
//! DSGE-VAR (calibrated, estimated and infinite weight), heteroskedastic filter (likelihood
//! and smoother), rescale_prediction_error_covariance, conditional_likelihood, and the
//! Raftery-Lewis / Geweke convergence diagnostics.
mod common;

use std::path::{Path, PathBuf};

use dynare::estimation::Objective;
use dynare::linalg::Mat;
use dynare::modfile::{Command, OptValue, opt};
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/q3")
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

fn mat(v: &Value) -> Mat {
    let rows = v.as_array().unwrap();
    let c = rows[0].as_array().map_or(1, |x| x.len());
    Mat::from_fn(rows.len(), c, |i, j| rows[i].as_array().map_or_else(|| vecf(&rows[i])[0], |x| vecf(&x[j])[0]))
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

fn setup(stage: usize, x0: &[f64]) -> Setup {
    let defines = vec![("STAGE".to_string(), stage.to_string())];
    let mf = dynare::modfile::parse_mod_file(&root().join("q3_nk.mod"), &defines).unwrap();
    let est = mf
        .commands
        .iter()
        .find_map(|c| if let Command::Estimation(o, _) = c { Some(o.clone()) } else { None })
        .unwrap();
    let num = |k: &str| match opt(&est, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    };
    let dsge = dynare::dsge::DsgeModel::new(&mf, Some(x0)).unwrap();
    let table = dynare::data::Table::read(&root().join("nk_data.csv"), &Default::default()).unwrap();
    let init = dynare::estimation::q3::initial_conditions(&mf, &est).unwrap();
    let first_obs = num("first_obs").unwrap_or(1.0) as usize - init;
    let sample = table.sample(&mf.varobs, first_obs, None).unwrap();
    let mut k = dynare::kalman::KalmanOptions {
        lik_init: if opt(&est, "diffuse_filter").is_some() { 3 } else { 1 },
        kalman_algo: num("kalman_algo").unwrap_or(0.0) as u8,
        ..Default::default()
    };
    dynare::estimation::q3::kalman_options(&est, &mut k);
    let mut data = dynare::dsge::LikelihoodData::new(&sample, k);
    data.first_obs = first_obs;
    dynare::estimation::q3::attach(&mf, &est, &mut data, first_obs).unwrap();
    Setup { mf, est, dsge, data }
}

fn check_points(stage: usize) {
    let r = load(&format!("ref_q3_{stage}.json")).expect("reference/setup unavailable");
    let x0 = vecf(&r["xparam_init"]);
    let s = setup(stage, &x0);
    assert_eq!(s.data.y.ncols(), r["nobs"].as_u64().unwrap() as usize, "nobs");
    let ddata = (&s.data.y.transpose() - mat(&r["data"])).amax();
    assert!(ddata < 1e-14, "data differ by {ddata}");
    let q3x = dynare::estimation::q3::Extras::new(&s.mf, &s.est, &s.dsge, &s.data).unwrap();
    let lin = dynare::dsge::DsgePosterior { dsge: &s.dsge, data: &s.data, bounds: &s.dsge.bounds };
    let q3p = q3x.posterior(&s.dsge, &s.data, &s.dsge.bounds);
    let post: &dyn Objective = match &q3p {
        Some(p) => p.as_ref(),
        None => &lin,
    };
    for (k, p) in r["points"].as_array().unwrap().iter().enumerate() {
        let x = vecf(&p["x"]);
        let f = post.eval(&x).0;
        let v = p["fval"].as_f64().unwrap();
        println!("stage {stage} point {k}: {f:.12} vs Dynare {v:.12} (rel {:.1e})", ((f - v) / v).abs());
        assert!(((f - v) / v).abs() < 1e-10, "stage {stage} point {k}: {f} vs {v}");
        if let Some((m, w)) = &q3x.dsge_var {
            if k == 0 {
                let dm = (&m.m_yy - mat(&r["mYY"]))
                    .amax()
                    .max((&m.m_yx - mat(&r["mYX"])).amax())
                    .max((&m.m_xx - mat(&r["mXX"])).amax());
                assert!(dm < 1e-9, "var moments differ by {dm}");
            }
            let dv = dynare::estimation::dsge_var::DsgeVarPosterior {
                dsge: &s.dsge,
                moments: m,
                weight: *w,
                bounds: &s.dsge.bounds,
                qz_criterium: q3x.lyap.0,
                lyapunov_complex_threshold: q3x.lyap.1,
            };
            let e = dv.evaluate(&x).unwrap();
            let rel = |a: &Mat, b: &Mat| (a - b).amax() / b.amax();
            let (dphi, dsig, dixx) = (
                rel(&e.phi_tilde, &mat(&p["PHI_tilde"])),
                rel(&e.sigma_u_tilde, &mat(&p["SIGMA_u_tilde"])),
                rel(&e.ixx, &mat(&p["iXX"])),
            );
            println!("   PHI_tilde {dphi:.1e} SIGMA_u_tilde {dsig:.1e} iXX {dixx:.1e}");
            assert!(dphi < 1e-8 && dsig < 1e-8 && dixx < 1e-7);
        }
    }
    // smoother at the initial values (estimation with mode_compute = 0, smoother)
    if r.get("SmoothedVariables").is_some() {
        let sm = dynare::driver::run_smoother(&s.dsge, &s.data, &x0, false).unwrap();
        for (j, name) in s.mf.model.endo_names[..s.mf.model.orig_endo_nbr].iter().enumerate() {
            let want = vecf(&r["SmoothedVariables"][name]);
            let d = (0..want.len()).map(|t| (sm.smoothed[(t, j)] - want[t]).abs()).fold(0.0, f64::max);
            println!("   smoothed {name}: {d:.1e}");
            assert!(d < 1e-8, "smoothed {name} differs by {d}");
        }
        for (j, name) in s.mf.model.exo_names.iter().enumerate() {
            let want = vecf(&r["SmoothedShocks"][name]);
            let d = (0..want.len()).map(|t| (sm.shocks[(t, j)] - want[t]).abs()).fold(0.0, f64::max);
            println!("   smoothed shock {name}: {d:.1e}");
            assert!(d < 1e-8, "smoothed shock {name} differs by {d}");
        }
    }
}

#[test]
fn dsge_var_calibrated() {
    check_points(1);
}

#[test]
fn dsge_var_estimated_weight() {
    check_points(2);
}

#[test]
fn dsge_var_infinite_weight() {
    check_points(3);
}

#[test]
fn heteroskedastic_filter_multivariate() {
    check_points(4);
}

#[test]
fn heteroskedastic_filter_diffuse_univariate() {
    check_points(5);
}

#[test]
fn rescale_prediction_error_covariance() {
    check_points(6);
}

#[test]
fn conditional_likelihood() {
    check_points(7);
}

#[test]
fn raftery_lewis_and_geweke() {
    let c = load("ref_q3_convergence.json").expect("reference/setup unavailable");
    let x = mat(&c["x"]);
    let runs: Vec<Vec<f64>> = (0..2).map(|j| x.column(j).iter().copied().collect()).collect();
    let q = vecf(&c["q"]);
    assert!((dynare::estimation::convergence::matlab_quantile(&runs[0], 0.025) - q[0]).abs() < 1e-12);
    assert!((dynare::estimation::convergence::matlab_quantile(&runs[1], 0.3) - q[1]).abs() < 1e-12);
    for (key, qrs) in [("rl", [0.025, 0.005, 0.95]), ("rl2", [0.1, 0.02, 0.9])] {
        let rl = dynare::estimation::convergence::raftery_lewis(&runs, qrs[0], qrs[1], qrs[2]);
        let r = &c[key];
        for (name, got) in [
            ("M_burn", &rl.m_burn),
            ("N_prec", &rl.n_prec),
            ("k_thin", &rl.k_thin),
            ("k_ind", &rl.k_ind),
            ("N_total", &rl.n_total),
        ] {
            assert_eq!(got, &vecf(&r[name]), "{key}.{name}");
        }
        let istat = vecf(&r["I_stat"]);
        assert!(rl.i_stat.iter().zip(&istat).all(|(a, b)| (a - b).abs() < 1e-12), "{key}.I_stat");
        assert_eq!(rl.n_min, r["N_min"].as_f64().unwrap());
    }
    let gm = dynare::estimation::diagnostics::geweke_moments(&runs[0], &[2, 5, 10]);
    let want = vecf(&c["gm"]);
    let d = gm.iter().zip(&want).map(|(a, b)| (a - b).abs() / b.abs().max(1e-300)).fold(0.0, f64::max);
    assert!(d < 1e-10, "geweke_moments rel diff {d}");
}

#[test]
fn simulannealbnd_substitute_minimizes_a_quadratic() {
    let quad = |x: &[f64]| x.iter().enumerate().map(|(i, v)| (i as f64 + 1.0) * (v - 0.5).powi(2)).sum::<f64>();
    let obj = dynare::estimation::FnObjective { n: 3, f: quad };
    let bounds = dynare::estimation::Bounds { lb: vec![-2.0; 3], ub: vec![2.0; 3] };
    let o = dynare::estimation::optim::simulannealbnd::SimulannealbndOptions {
        verbose: false,
        seed: 7,
        ..Default::default()
    };
    let r = dynare::estimation::optim::simulannealbnd::simulannealbnd(&obj, &[1.5, -1.5, 1.0], &bounds, &o).unwrap();
    println!("simulannealbnd: f = {:.3e} after {} evaluations ({})", r.fval, r.func_count, r.message);
    assert!(r.fval < 1e-3, "{:?}", r.x);
    assert!(r.x.iter().zip(&bounds.lb).all(|(x, l)| x >= l));
}

#[test]
fn lyapunov_fixed_point() {
    check_points(8);
}

#[test]
fn lyapunov_doubling() {
    check_points(9);
}

fn scratch(name: &str, files: &[&str]) -> PathBuf {
    let d = std::env::temp_dir().join(format!("dynare_rust_q3_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    for f in files {
        std::fs::copy(root().join(f), d.join(f)).unwrap();
    }
    d
}

fn run_driver(dir: &Path, file: &str, defines: &[(&str, &str)]) {
    let ro = dynare::driver::RunOptions {
        defines: defines.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(),
        seed: 1,
        quiet: true,
        ..Default::default()
    };
    dynare::driver::run(&dir.join(file), &ro).unwrap();
}

/// Statistical agreement of two MCMC runs: |a - b| below `k` posterior standard deviations.
fn close(what: &str, a: f64, b: f64, sd: f64, k: f64) {
    println!("{what}: {a:.5} vs Dynare {b:.5} (sd {sd:.5})");
    assert!((a - b).abs() < k * sd, "{what}: {a} vs {b}");
}

/// DSGE-VAR with an estimated weight: mode (csminwel), MCMC and the bvar-dsge posterior IRFs.
#[test]
fn dsge_var_mcmc_and_bayesian_irf() {
    let r = load("ref_q3_10.json").expect("reference/setup unavailable");
    let d = scratch("dsgevar", &["q3_nk.mod", "nk_model.inc", "nk_data.csv"]);
    run_driver(&d, "q3_nk.mod", &[("STAGE", "10")]);
    let res: Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("q3_nk/Output/q3_nk_results.json")).unwrap()).unwrap();
    let names: Vec<String> =
        res["parameter_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let mean = vecf(&res["mcmc"]["posterior_mean"]);
    let sd = vecf(&res["mcmc"]["posterior_sd"]);
    for (k, n) in names.iter().enumerate() {
        let short = n.trim_start_matches("stderr ");
        let want = r["posterior_mean"]["parameters"][short]
            .as_f64()
            .or_else(|| r["posterior_mean"]["shocks_std"][short].as_f64())
            .unwrap();
        close(n, mean[k], want, sd[k], 0.5);
    }
    let mhm = res["mcmc"]["log_data_density_mhm"].as_f64().unwrap();
    // Loose on purpose: Rust's chain uses its own RNG, so this compares two independent
    // Monte Carlo estimates of the log data density (MHM over a short chain has a spread of
    // several tenths); the deterministic pieces are checked exactly elsewhere.
    close("MHM", mhm, r["mhm"].as_f64().unwrap(), 1.0, 1.0);
    assert!(res["dsge_var_posterior_mode"]["PHI_tilde"].is_array());
    // bvar-dsge IRFs: posterior mean of y and pi to e_y
    let csv = std::fs::read_to_string(d.join("q3_nk/Output/q3_nk_Bayesian_IRF_BVAR-DSGE_e_y.csv")).unwrap();
    for (var, key) in [("y", "y_e_y"), ("pi", "pi_e_y")] {
        let ours: Vec<f64> = csv
            .lines()
            .filter(|l| l.starts_with(&format!("{var},")))
            .map(|l| l.split(',').nth(2).unwrap().parse().unwrap())
            .collect();
        let want = vecf(&r["irf_bvardsge_mean"][key]);
        for (t, (a, b)) in ours.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 0.05 * want[0].abs().max(0.1), "bvar-dsge IRF {key}({}) {a} vs {b}", t + 1);
        }
    }
}

/// Bayesian IRF matching with an MCMC (method_of_moments, mh_replic > 0).
#[test]
fn method_of_moments_mcmc() {
    let r = load("ref_q3_11.json").expect("reference/setup unavailable");
    let d = scratch("mom", &["nk_irf_mcmc.mod", "nk_irf_data.inc"]);
    run_driver(&d, "nk_irf_mcmc.mod", &[("CASE", "2")]);
    let res: Value = serde_json::from_str(
        &std::fs::read_to_string(d.join("nk_irf_mcmc/method_of_moments/nk_irf_mcmc_mom_mcmc.json")).unwrap(),
    )
    .unwrap();
    let mean = vecf(&res["posterior_mean"]);
    let sd = vecf(&res["posterior_sd"]);
    let want = [
        r["posterior_mean"]["shocks_std"]["e_i"].as_f64().unwrap(),
        r["posterior_mean"]["parameters"]["kappa"].as_f64().unwrap(),
        r["posterior_mean"]["parameters"]["phi_pi"].as_f64().unwrap(),
        r["posterior_mean"]["parameters"]["rho_a"].as_f64().unwrap(),
        r["posterior_mean"]["parameters"]["rho_i"].as_f64().unwrap(),
    ];
    for k in 0..5 {
        close(&format!("param {k}"), mean[k], want[k], sd[k], 0.5);
    }
    // Loose on purpose: Rust's chain uses its own RNG, so this compares two independent
    // Monte Carlo estimates of the log data density (MHM over a short chain has a spread of
    // several tenths); the deterministic pieces are checked exactly elsewhere.
    close("MHM", res["log_data_density_mhm"].as_f64().unwrap(), r["mhm"].as_f64().unwrap(), 1.0, 0.5);
}

fn read_csv(p: &Path) -> Vec<Vec<f64>> {
    std::fs::read_to_string(p)
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| l.split(',').skip(1).map(|v| v.parse().unwrap()).collect())
        .collect()
}

/// filter_step_ahead = [1 3], filter_covariance, filter_decomposition (oo_.FilteredVariablesKStepAhead,
/// ...KStepAheadVariances, ...ShockDecomposition) with the multivariate (12) and univariate (13) smoothers.
fn check_step_ahead(stage: usize) {
    let r = load(&format!("ref_q3_{stage}.json")).expect("reference/setup unavailable");
    let d = scratch(&format!("kstep{stage}"), &["q3_nk.mod", "nk_model.inc", "nk_data.csv"]);
    run_driver(&d, "q3_nk.mod", &[("STAGE", &stage.to_string())]);
    let out = d.join("q3_nk/Output");
    let nexo = 3;
    for (ks, k) in [(0usize, 1usize), (1, 3)] {
        let a = read_csv(&out.join(format!("q3_nk_filtered_variables_{k}_step_ahead.csv")));
        let v = read_csv(&out.join(format!("q3_nk_filtered_variables_{k}_step_ahead_variances.csv")));
        let dec = read_csv(&out.join(format!("q3_nk_filtered_variables_{k}_step_ahead_shock_decomposition.csv")));
        let (mut da, mut dv, mut dd) = (0.0f64, 0.0f64, 0.0f64);
        for (c, row) in a.iter().enumerate() {
            for j in 0..3 {
                da = da.max((row[j] - r["KStepAhead"][ks][j][c].as_f64().unwrap()).abs());
                dv = dv.max((v[c][j] - r["KStepAheadVariances"][ks][j][j][c].as_f64().unwrap()).abs());
                for e in 0..nexo {
                    dd = dd.max((dec[c][j * nexo + e] - r["ShockDecomposition"][ks][j][e][c].as_f64().unwrap()).abs());
                }
            }
        }
        println!("stage {stage}, {k}-step ahead: states {da:.1e}, variances {dv:.1e}, decomposition {dd:.1e}");
        assert!(da < 1e-9 && dv < 1e-9 && dd < 1e-9);
    }
}

#[test]
fn filter_step_ahead_multivariate() {
    check_step_ahead(12);
}

#[test]
fn filter_step_ahead_univariate() {
    check_step_ahead(13);
}

#[test]
fn filter_initial_state_stationary() {
    check_points(14);
}

#[test]
fn filter_initial_state_diffuse() {
    check_points(15);
}

#[test]
fn pruned_skewed_kalman_filter() {
    check_points(16);
}

#[test]
fn pruned_skewed_kalman_filter_no_pruning_rescaled() {
    check_points(17);
}

/// Generalized priors: shifted beta / gamma / Weibull / inverse gammas (PRIOR_3RD_PARAMETER,
/// PRIOR_4TH_PARAMETER) and bounds on a normal prior.
#[test]
fn generalized_priors() {
    check_points(18);
}

/// Parsing of the stream's declarations.
#[test]
fn q3_declarations_parsed() {
    let mf = dynare::modfile::parse_mod_file(&root().join("q3_nk.mod"), &[("STAGE".into(), "16".into())]).unwrap();
    assert_eq!(mf.q3.skew_e, vec![(0, 0, 0, 0.6)]);
    assert!(mf.estimated_params.iter().any(|e| e.kind == dynare::modfile::EstimatedKind::Skew(1)));
    let mf = dynare::modfile::parse_mod_file(&root().join("q3_nk.mod"), &[("STAGE".into(), "4".into())]).unwrap();
    // e_y: 50:60 (scale 2) and 61 (scale 0.5); e_pi: 100 (0.1) and 101:103 (1.2); e_i: 120:125 (0.5)
    assert_eq!(mf.q3.heteroskedastic_shocks.len(), 5);
    let mf = dynare::modfile::parse_mod_file(&root().join("q3_nk.mod"), &[("STAGE".into(), "14".into())]).unwrap();
    assert_eq!(mf.q3.filter_initial_state.len(), 3);
    let mf = dynare::modfile::parse_mod_file(&root().join("q3_nk.mod"), &[("STAGE".into(), "2".into())]).unwrap();
    assert_eq!(mf.model.param_names.last().map(String::as_str), Some("dsge_prior_weight"));
}

/// Posterior distributions of the OccBin-smoothed variables after an OccBin MCMC
/// (occbin_post.mod -DMCMC: mode_compute = 6 + 2 x 3000 draws, ~1 min): posterior means of the
/// smoothed variables against Dynare's, statistically. `cargo test --release --test estimation_q3 -- --ignored`
#[test]
#[ignore]
fn occbin_posterior_smoothed_variables() {
    let r = load("ref_q3_occbin_post.json").expect("reference/setup unavailable");
    let d = scratch("occbin_post", &["occbin_post.mod", "nk_est_data.csv"]);
    run_driver(&d, "occbin_post.mod", &[("MCMC", "1")]);
    let csv =
        std::fs::read_to_string(d.join("occbin_post/Output/occbin_post_posterior_smoothed_variables.csv")).unwrap();
    for var in ["y", "pi", "i"] {
        let ours: Vec<f64> = csv
            .lines()
            .filter(|l| l.split(',').nth(1) == Some(var))
            .map(|l| l.split(',').nth(2).unwrap().parse().unwrap())
            .collect();
        let want = vecf(&r["SmoothedVariables_Mean"][var]);
        assert_eq!(ours.len(), want.len());
        let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-3);
        let d = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / scale;
        println!("posterior mean of smoothed {var}: max rel diff {d:.3}");
        assert!(d < 0.15, "{var}: {d}");
    }
}
