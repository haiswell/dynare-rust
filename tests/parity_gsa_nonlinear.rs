//! gsa-nonlinear parity fixes: `sensitivity` (src/gsa) and the particle-filter / SMC estimation
//! (src/nonlinear). Expected values are derived independently (sample moments of the
//! distribution a sampler must draw from, properties that hold by construction in Dynare such as
//! "pruning is ignored by the order-3 online filter") or come from the Dynare 7.2 sources named in
//! each test.
//!
//! Item 8 (nonlinear estimation at order > 3) is not implemented: the k-order solver stops at
//! order 3.
mod common;

use std::path::{Path, PathBuf};

use dynare::driver::{RunOptions, run};
use dynare::gsa;
use dynare::linalg::Mat;
use dynare::modfile::{Command, ModFile, OptValue, Options, parse_mod_file};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

// ---------------------------------------------------------------------------------------
// GSA
// ---------------------------------------------------------------------------------------

/// nk_gsa.mod up to its commands, then `tail`.
fn nk_text(tail: &str) -> String {
    let t = std::fs::read_to_string(root().join("reference/gsa2/nk_gsa.mod")).unwrap();
    let t = t.replace("@#ifndef STAGE\n@#define STAGE = 1\n@#endif\n", "");
    let cut = t.find("@#ifdef DYNARE_MATLAB").unwrap();
    format!("{}{tail}", &t[..cut])
}

fn nk_setup(tag: &str, tail: &str) -> (PathBuf, ModFile) {
    let dir = common::temp_dir(tag);
    std::fs::copy(root().join("reference/gsa2/nk_gsa_data.csv"), dir.join("nk_gsa_data.csv")).unwrap();
    let f = dir.join("nk_gsa.mod");
    std::fs::write(&f, nk_text(tail)).unwrap();
    let mf = parse_mod_file(&f, &[]).unwrap();
    (dir, mf)
}

fn command(mf: &ModFile, name: &str, nth: usize) -> (usize, Options) {
    mf.commands
        .iter()
        .enumerate()
        .filter_map(|(i, c)| match c {
            Command::Other(n, o) if n == name => Some((i, o.clone())),
            _ => None,
        })
        .nth(nth)
        .unwrap()
}

/// gsa-nonlinear-5: `+gsa/run.m` takes `dynare_estimation_init` (and so
/// `select_qz_criterium_value`: `1-1e-6` for `lik_init = 1`) only with a datafile, no `bayestopt_`
/// or `rmse`; after an estimation (bayestopt_ set, `options_.qz_criterium` back to `[]`) the
/// `else` branch sets `1+1e-6`.
#[test]
fn gsa_qz_criterium_follows_the_branch_of_run_m() {
    let est = "estimation(datafile=nk_gsa_data, mode_compute=0, nograph, nodisplay);\n";
    let sens = "sensitivity(nsam=16, stab=1, nograph);\n";
    let sens_df = "sensitivity(nsam=16, stab=1, datafile=nk_gsa_data, nograph);\n";
    let case = |tag: &str, tail: &str| {
        let (_d, mf) = nk_setup(tag, tail);
        let (i, o) = command(&mf, "sensitivity", 0);
        let g = gsa::GsaOptions::parse(&o, &mf).unwrap();
        gsa::runs_estimation_init(&mf, i, &g)
    };
    assert!(case("pgn_q1", sens), "no estimation before: bayestopt_ is empty");
    assert!(!case("pgn_q2", &format!("{est}{sens}")), "estimation before, no datafile: run.m's else branch");
    assert!(case("pgn_q3", &format!("{est}{sens_df}")), "a datafile always re-runs dynare_estimation_init");
    let eo: Options = vec![];
    assert_eq!(gsa::qz_criterium(&eo, true), 1.0 - 1e-6);
    assert_eq!(gsa::qz_criterium(&eo, false), 1.0 + 1e-6);
    let diffuse: Options = vec![("lik_init".into(), OptValue::Num(2.0))];
    assert_eq!(gsa::qz_criterium(&diffuse, true), 1.0 + 1e-6);
    let user: Options = vec![("qz_criterium".into(), OptValue::Num(0.9))];
    assert_eq!(gsa::qz_criterium(&user, false), 0.9);
    assert_eq!(gsa::qz_criterium(&user, true), 0.9);
}

/// gsa-nonlinear-10: `pprior = 0` without `neighborhood_width` samples `N(xparam1, inv(hh))` from
/// the mode file (`stability_mapping.m`: `d = chol(inv(hh)); lp = randn(2*Nsam, n)*d + xparam1'`),
/// keeping the rows strictly inside the bounds. The sample mean, standard deviations and the
/// correlation of two correlated parameters must be those of that distribution.
#[test]
fn posterior_sampling_around_the_mode_file() {
    let (dir, mf) =
        nk_setup("pgn_post", "sensitivity(nsam=800, stab=1, pprior=0, mode_file=nk_mode, nograph, redform=0);\n");
    // xparam1 of nk_gsa.mod: e_g e_u e_i stderr, sigma kappa phi_pi phi_y rho_i rho_g rho_u
    let x = vec![0.5, 0.3, 0.2, 1.0, 0.1, 1.5, 0.125, 0.7, 0.9, 0.5];
    let n = x.len();
    let sd = vec![0.02; n];
    let rho = 0.8; // between phi_pi (index 5) and phi_y (index 6)
    let mut cov = Mat::from_fn(n, n, |i, j| if i == j { sd[i] * sd[i] } else { 0.0 });
    cov[(5, 6)] = rho * sd[5] * sd[6];
    cov[(6, 5)] = cov[(5, 6)];
    let hh = cov.clone().try_inverse().unwrap();
    dynare::matfile::write_values(
        &dir.join("nk_mode.mat"),
        &[
            ("xparam1", &dynare::matfile::MatValue::Numeric { dims: vec![n, 1], data: x.clone() }),
            ("hh", &dynare::matfile::MatValue::matrix(&hh)),
        ],
    )
    .unwrap();
    let (i, o) = command(&mf, "sensitivity", 0);
    let out = gsa::run(&mf, &dir, "nk_gsa", &o, i).unwrap().expect("no output");
    let nshock = 3;
    let rows: Vec<Vec<f64>> =
        out.stab.lpmat0.iter().zip(&out.stab.lpmat).map(|(a, b)| a.iter().chain(b).copied().collect()).collect();
    assert!(rows.len() >= 700 && rows.len() <= 800, "{} rows", rows.len());
    assert_eq!(rows[0].len(), n);
    let m = rows.len() as f64;
    let mean = |j: usize| rows.iter().map(|r| r[j]).sum::<f64>() / m;
    let sdev = |j: usize| (rows.iter().map(|r| (r[j] - mean(j)).powi(2)).sum::<f64>() / m).sqrt();
    for j in 0..n {
        assert!((mean(j) - x[j]).abs() < 4.0 * sd[j] / m.sqrt(), "mean[{j}] = {} vs {}", mean(j), x[j]);
        assert!((sdev(j) / sd[j] - 1.0).abs() < 0.15, "sd[{j}] = {} vs {}", sdev(j), sd[j]);
    }
    let corr = rows.iter().map(|r| (r[5] - mean(5)) * (r[6] - mean(6))).sum::<f64>() / m / (sdev(5) * sdev(6));
    assert!((corr - rho).abs() < 0.06, "corr(phi_pi, phi_y) = {corr}");
    let _ = nshock;
    // every draw is inside the bounds (strictly)
    for r in &rows {
        for j in 0..n {
            assert!(r[j] > out.ctx.bounds.lb[j] && r[j] < out.ctx.bounds.ub[j]);
        }
    }
    // without a mode file and without neighborhood_width Dynare cannot load it: an error
    let (dir2, mf2) = nk_setup("pgn_post2", "sensitivity(nsam=50, stab=1, pprior=0, nograph);\n");
    let (i, o) = command(&mf2, "sensitivity", 0);
    assert!(gsa::run(&mf2, &dir2, "nk_gsa", &o, i).is_err());
}

/// gsa-nonlinear-16: the moments of `moment_calibration` and `identification` are computed with
/// the global `options_`, which keeps `hp_filter` (and `bandpass_filter`) of an earlier
/// `stoch_simul`.
#[test]
fn gsa_moments_use_the_persistent_filters() {
    let calib = "moment_calibration;\ny, y, [0, 10];\nend;\n";
    let (_d, mf) = nk_setup(
        "pgn_hp",
        &format!(
            "stoch_simul(order=1, irf=0, hp_filter=1600, nograph, nomoments);\n{calib}sensitivity(nsam=16, stab=1, nograph);\n"
        ),
    );
    let (i, _) = command(&mf, "sensitivity", 0);
    let r = gsa::calibration::restrictions_before(&mf, i).unwrap();
    assert_eq!(r.moment_options.hp_filter, 1600.0);
    assert_eq!(gsa::calibration::persistent_moment_options(&mf, i).hp_filter, 1600.0);
    // no filter without the stoch_simul option
    let (_d, mf0) = nk_setup("pgn_hp0", &format!("{calib}sensitivity(nsam=16, stab=1, nograph);\n"));
    let (i0, _) = command(&mf0, "sensitivity", 0);
    assert_eq!(gsa::calibration::restrictions_before(&mf0, i0).unwrap().moment_options.hp_filter, 0.0);
    // the HP-filtered variance of y is smaller than its raw variance: the restriction sees it
    let dsge = dynare::dsge::DsgeModel::new(&mf, None).unwrap();
    let x = dynare::dsge::initial_vector(&mf).unwrap();
    let sol = dsge.resolve(&x).unwrap();
    let with = gsa::calibration::endogenous_prior_restrictions(&dsge, &sol.dr, &sol.q, &r).unwrap();
    let r0 = gsa::calibration::restrictions_before(&mf0, i0).unwrap();
    let without = gsa::calibration::endogenous_prior_restrictions(&dsge, &sol.dr, &sol.q, &r0).unwrap();
    let (v1, v0) = (with.data_moment[0][0].1, without.data_moment[0][0].1);
    assert!(v1 > 0.0 && v1 < v0, "HP-filtered variance {v1} vs raw {v0}");
}

// ---------------------------------------------------------------------------------------
// nonlinear filters / SMC
// ---------------------------------------------------------------------------------------

use dynare::dsge::DsgeModel;
use dynare::nonlinear::online::{OnlineOptions, initial_particles, online_auxiliary_filter};
use dynare::nonlinear::{NonlinearModel, ParticleOptions, select_qz_criterium_value};

fn nl_dir() -> PathBuf {
    root().join("reference/nonlinear")
}

/// rbcnl.mod at `order` with the first `t` observations; `particle` options as given.
fn rbcnl(order: usize, t: usize, particle: ParticleOptions) -> (ModFile, NonlinearModel, Vec<f64>) {
    let mf = parse_mod_file(&nl_dir().join("rbcnl.mod"), &[("ORDER".into(), order.to_string())]).unwrap();
    let x0 = dynare::dsge::initial_vector(&mf).unwrap();
    let dsge = DsgeModel::new(&mf, Some(&x0)).unwrap();
    let table = dynare::data::Table::read_csv(&nl_dir().join("rbcnl_data.csv")).unwrap();
    let y = table.sample(&mf.varobs, 1, None).unwrap();
    let y = y.rows(0, t).into_owned().transpose();
    let nm = NonlinearModel::new(dsge, order, particle, y, 0, 0).unwrap();
    (mf, nm, x0)
}

fn pf(particles: usize) -> ParticleOptions {
    ParticleOptions { number_of_particles: particles, ..Default::default() }
}

/// An estimation command's options, parsed from a .mod line.
fn estimation_options(extra: &str) -> Options {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = common::temp_dir(&format!("pgn_estopt{}", N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)));
    let f = dir.join("m.mod");
    let src = std::fs::read_to_string(nl_dir().join("rbcnl.mod")).unwrap();
    let cut = src.find("@#if defined(ESTGF)").unwrap();
    let src = format!("{}estimation(datafile='rbcnl_data.csv', order=2, {extra});\n@#endif\n", &src[..cut]);
    std::fs::write(&f, src).unwrap();
    let mf = parse_mod_file(&f, &[]).unwrap();
    mf.commands.iter().find_map(|c| if let Command::Estimation(o, _) = c { Some(o.clone()) } else { None }).unwrap()
}

/// gsa-nonlinear-4: `select_qz_criterium_value.m`: with a particle filter and
/// `nonlinear_filter_initialization` 1 or 2 the default is `1-1e-6`, with 3 it is `1+1e-6` (a user
/// value <= 1 is reset); it reaches the first-order solver too, so a root in
/// `(1-1e-6, 1+1e-6]` is a Blanchard-Kahn failure where the linear default would accept it.
#[test]
fn particle_filter_qz_criterium() {
    assert_eq!(select_qz_criterium_value(None, Some(1), 1.0).unwrap(), 1.0 - 1e-6);
    assert_eq!(select_qz_criterium_value(None, Some(2), 1.0).unwrap(), 1.0 - 1e-6);
    assert_eq!(select_qz_criterium_value(None, Some(3), 1.0).unwrap(), 1.0 + 1e-6);
    assert_eq!(select_qz_criterium_value(Some(0.5), Some(3), 1.0).unwrap(), 1.0 + 1e-6);
    assert_eq!(select_qz_criterium_value(Some(1.1), Some(3), 1.0).unwrap(), 1.1);
    assert_eq!(select_qz_criterium_value(Some(0.99), Some(1), 1.0).unwrap(), 0.99);
    assert_eq!(select_qz_criterium_value(None, None, 1.0).unwrap(), 1.0 - 1e-6);
    assert_eq!(select_qz_criterium_value(None, None, 2.0).unwrap(), 1.0 + 1e-6);
    // through the model: z has the root rho = 1 - 5e-7
    let (_mf, mut nm, mut x) = rbcnl(2, 10, pf(10));
    assert_eq!(nm.qz_criterium, 1.0 - 1e-6);
    let irho = nm.dsge.estimated_names.iter().position(|n| n == "rho").unwrap();
    x[irho] = 1.0 - 5e-7;
    // the first-order solver (resol) classifies the root with the particle default too
    assert!(nm.dsge.resolve(&x).is_err(), "root 1-5e-7 is explosive for qz_criterium = 1-1e-6");
    assert!(nm.reduced_form(&x).is_err());
    nm.set_qz_criterium(Some(1.0 + 1e-6)).unwrap();
    assert!(nm.dsge.resolve(&x).is_ok(), "and accepted for 1+1e-6");
    x[irho] = 0.95;
    assert!(nm.reduced_form(&x).is_ok());
}

/// gsa-nonlinear-12: `check_particle_filter_options.m` has no `initialization` key (Dynare only
/// warns), accepts `liu_west_delta` and `posterior_sampler`, and errors on non-positive
/// `initial_state_prior_std`, `unscented_alpha`, mixture sizes, and a non-logical `pruning`.
#[test]
fn particle_filter_options_follow_check_particle_filter_options() {
    let ok =
        |extra: &str| ParticleOptions::from_options(&estimation_options(&format!("particle_filter_options=({extra})")));
    let p = ok("'initialization', 2").unwrap();
    assert_eq!(p.initialization, 1, "'initialization' is not a particle_filter_options key");
    let p = ParticleOptions::from_options(&estimation_options("nonlinear_filter_initialization=2")).unwrap();
    assert_eq!(p.initialization, 2);
    ok("'liu_west_delta', 0.9").unwrap();
    ok("'posterior_sampler', 'DSMH'").unwrap();
    assert_eq!(ok("'mixture_state_variables', 3").unwrap().mixture_state_variables, 3);
    assert_eq!(ok("'initial_state_prior_std', 0.5").unwrap().initial_state_prior_std, 0.5);
    for (bad, msg) in [
        ("'initial_state_prior_std', 0", "initial_state_prior_std"),
        ("'unscented_alpha', -1", "unscented_alpha"),
        ("'unscented_beta', -1", "unscented_beta"),
        ("'mixture_state_variables', 0", "mixture_state_variables"),
        ("'liu_west_delta', 0", "liu_west_delta"),
        ("'pruning', 1", "pruning"),
    ] {
        let e = format!("{:#}", ok(bad).unwrap_err());
        assert!(e.contains(msg), "{bad}: {e}");
    }
}

/// gsa-nonlinear-13: `check_posterior_sampler_options.m` (dsmh): keys are case sensitive and `M`
/// must divide `N*G`.
#[test]
fn dsmh_options_are_case_sensitive_and_check_divisibility() {
    use dynare::nonlinear::estimation::dsmh_options;
    let d = dsmh_options(&estimation_options("posterior_sampler_options=('H', 7, 'N', 12, 'G', 3, 'M', 4)"), 0, true)
        .unwrap();
    assert_eq!((d.h, d.n, d.g, d.m), (7, 12, 3, 4));
    let dflt = dsmh_options(&estimation_options("mode_compute=0"), 0, true).unwrap();
    let low = dsmh_options(&estimation_options("posterior_sampler_options=('h', 7, 'n', 5)"), 0, true).unwrap();
    assert_eq!((low.h, low.n), (dflt.h, dflt.n), "lower-case keys are unknown options");
    let e =
        dsmh_options(&estimation_options("posterior_sampler_options=('N', 10, 'G', 3, 'M', 4)"), 0, true).unwrap_err();
    assert!(format!("{e:#}").contains("DSMH requires M to divide N*G = 30"), "{e:#}");
}

/// Runs rbcnl.mod with its estimation commands replaced by `est` (in a private folder).
fn rbcnl_run(tag: &str, est: &str) -> anyhow::Result<(PathBuf, serde_json::Value)> {
    let work = common::temp_dir(tag);
    std::fs::copy(nl_dir().join("rbcnl_data.csv"), work.join("rbcnl_data.csv")).unwrap();
    let src = std::fs::read_to_string(nl_dir().join("rbcnl.mod")).unwrap();
    let cut = src.find("@#if defined(ESTGF)").unwrap();
    std::fs::write(work.join("rbcnl.mod"), format!("{}{est}\n@#endif\n", &src[..cut])).unwrap();
    run(&work.join("rbcnl.mod"), &RunOptions { quiet: true, seed: 3, ..Default::default() })?;
    let r =
        serde_json::from_str(&std::fs::read_to_string(work.join("rbcnl/Output/rbcnl_results.json")).unwrap()).unwrap();
    Ok((work, r))
}

const GF: &str = "datafile='rbcnl_data.csv', order=2, filter_algorithm=gf, proposal_approximation=cubature, \
                  distribution_approximation=cubature, mh_replic=0, nograph, nodisplay, nodiagnostic";

/// gsa-nonlinear-9: Dynare refuses a particle-filter estimation with `loglinear` and `prefilter=1`.
#[test]
fn nonlinear_estimation_refuses_loglinear_and_prefilter() {
    let e = rbcnl_run("pgn_loglin", &format!("estimation({GF}, mode_compute=0, loglinear);")).err().unwrap();
    assert!(format!("{e:#}").contains("not possible to use a non linear filter with the option loglinear"), "{e:#}");
    let e = rbcnl_run("pgn_prefilter", &format!("estimation({GF}, mode_compute=0, prefilter=1);")).err().unwrap();
    assert!(format!("{e:#}").to_lowercase().contains("prefilter"), "{e:#}");
}

/// gsa-nonlinear-1: `options_.mode_compute` defaults to 5 (newrat) with a nonlinear filter as
/// well: the run without the option is the run with `mode_compute=5`.
#[test]
fn nonlinear_estimation_default_mode_compute_is_newrat() {
    let est = |mc: &str| format!("estimation({GF}{mc});");
    let (_w, a) = rbcnl_run("pgn_mc_default", &est("")).unwrap();
    let (_w, b) = rbcnl_run("pgn_mc_5", &est(", mode_compute=5")).unwrap();
    let mode = |r: &serde_json::Value| -> Vec<f64> {
        r["mode"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
    };
    assert_eq!(mode(&a), mode(&b));
}

/// gsa-nonlinear-2: the initial particles of the online filter are centred on the deterministic
/// steady state of the states (`state_variables_steady_state`), not on `constant(mf0)` =
/// steady state + 0.5*ghs2.
#[test]
fn online_initial_particles_are_centred_on_the_steady_state() {
    let (_mf, nm, x0) = rbcnl(2, 10, pf(10));
    let rf = nm.reduced_form(&x0).unwrap();
    // the means of 20 independent clouds of 500000 particles (state 0: capital, whose
    // 0.5*ghs2 shift is a few standard errors)
    let (chunks, n) = (20, 500_000usize);
    let mut sum = vec![0.0; rf.ns()];
    for seed in 0..chunks {
        let p = initial_particles(&rf, n, seed as u64 + 1).unwrap();
        for (i, s) in sum.iter_mut().enumerate() {
            *s += p.row(i).iter().sum::<f64>();
        }
    }
    let total = (chunks * n) as f64;
    let mut discriminates = false;
    for i in 0..rf.ns() {
        let mean = sum[i] / total;
        let se = rf.state_variance[(i, i)].sqrt() / total.sqrt();
        assert!((mean - rf.state_ss[i]).abs() < 4.0 * se, "state {i}: {mean} vs {}", rf.state_ss[i]);
        // far from constant(mf0) = steady state + 0.5*ghs2 where that differs
        if (rf.state_mean[i] - rf.state_ss[i]).abs() > 5.0 * se {
            discriminates = true;
            assert!((mean - rf.state_mean[i]).abs() > 3.0 * se);
        }
    }
    assert!(discriminates, "constant(mf0) must differ from the steady state by several standard errors");
}

/// gsa-nonlinear-7: at order 3 the online filter takes the k-order solver path, which ignores
/// pruning (`use_k_order_solver`, `local_state_space_iteration_k`): the filter with and without
/// `pruning` is the same, bit for bit; at order 2 pruning does change the recursion.
#[test]
fn online_filter_ignores_pruning_at_order_three() {
    let run_at = |order: usize, pruning: bool| {
        let (mf, nm, x0) = rbcnl(order, 4, ParticleOptions { pruning, ..pf(10) });
        let dsge = DsgeModel::new(&mf, Some(&x0)).unwrap();
        let bounds = dynare::estimation::prior_bounds(&dsge.priors, 0.0);
        let o = OnlineOptions { particles: 30, seed: 5, verbose: false, ..Default::default() };
        online_auxiliary_filter(&nm, &dsge.priors, &bounds, &x0, &o).unwrap()
    };
    let (a, b) = (run_at(3, true), run_at(3, false));
    assert_eq!(a.mean, b.mean);
    assert_eq!(a.param, b.param);
    let (c, d) = (run_at(2, true), run_at(2, false));
    assert_ne!(c.mean, d.mean, "pruning matters at order 2");
}

/// gsa-nonlinear-6: the online sampler runs at order 1 (`ReducedForm.constant` = steady state, zero
/// `ghxx`, `ghuu`, `ghxu`).
#[test]
fn online_filter_runs_at_order_one() {
    let (mf, nm, x0) = rbcnl(1, 6, pf(10));
    let rf = nm.reduced_form(&x0).unwrap();
    assert_eq!(rf.constant, rf.steadystate);
    assert!(rf.ghxx.iter().chain(rf.ghuu.iter()).chain(rf.ghxu.iter()).all(|&v| v == 0.0));
    let dsge = DsgeModel::new(&mf, Some(&x0)).unwrap();
    let bounds = dynare::estimation::prior_bounds(&dsge.priors, 0.0);
    let o = OnlineOptions { particles: 40, seed: 5, verbose: false, ..Default::default() };
    let r = online_auxiliary_filter(&nm, &dsge.priors, &bounds, &x0, &o).unwrap();
    assert_eq!(r.mean.ncols(), 6);
    assert!(r.mean.iter().all(|v| v.is_finite()) && r.std.iter().all(|v| v.is_finite() && *v >= 0.0));
    // through the estimation command (linear path of the driver)
    let est = "estimation(datafile='rbcnl_data.csv', order=1, mode_compute=0, mh_replic=0, nobs=8, \
posterior_sampling_method='online', posterior_sampler_options=('particles', 40), nograph, nodisplay, nodiagnostic);";
    let (_w, r) = rbcnl_run("pgn_online1", est).unwrap();
    assert_eq!(r["online"]["particles"].as_u64(), Some(40));
}

/// gsa-nonlinear-11: the online filter's prior bounds and draws use `options_.prior_trunc`, whose
/// default is 0 (the full support), not 1e-10. A run with a large truncation draws other particles.
#[test]
fn online_prior_trunc_defaults_to_zero() {
    assert_eq!(dynare::nonlinear::smc_run::online_prior_trunc(&vec![]), 0.0);
    let o: Options = vec![("prior_trunc".into(), OptValue::Num(1e-10))];
    assert_eq!(dynare::nonlinear::smc_run::online_prior_trunc(&o), 1e-10);
    let est = |extra: &str| {
        format!(
            "estimation(datafile='rbcnl_data.csv', order=2, filter_algorithm=sis, number_of_particles=30, mode_compute=0, mh_replic=0, nobs=5, posterior_sampling_method='online', posterior_sampler_options=('particles', 40){extra}, nograph, nodisplay, nodiagnostic);"
        )
    };
    let draws = |tag: &str, extra: &str| {
        let (w, _) = rbcnl_run(tag, &est(extra)).unwrap();
        std::fs::read_to_string(w.join("rbcnl/Output/rbcnl_online_particles.csv")).unwrap()
    };
    let a = draws("pgn_pt_default", "");
    assert_eq!(a, draws("pgn_pt_0", ", prior_trunc=0"));
}
