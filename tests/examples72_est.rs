//! End-to-end checks of Dynare 7.2's official examples (estimation, occbin, perfect foresight,
//! semistructural groups) against MATLAB runs of Dynare 7.2 (`reference/examples72/<name>/ref.json`,
//! written by the scripts in `reference/examples72/dump_est/`). Each test runs the `dynare-rs`
//! binary on a copy of the example and reads its output files.
//!
//! Deviations from the shipped examples (see the final report of the validation):
//! - gali_2015: the dataset is the `sim_data.m` MATLAB wrote with `datatomfile` (the port's own
//!   simulation draws other shocks), the user MATLAB functions (`prior_function`,
//!   `posterior_function`, `gali_2015_prior_restrictions.m`) are left out;
//! - rbc_irf_matching: the first two `method_of_moments` runs (the third needs MATLAB
//!   anonymous functions and an `irf_matching_file`);
//! - pac_model: `simul_backward_model` draws its shocks with MATLAB's RNG; the deterministic
//!   check replays MATLAB's draws (`pac_model_shocks.csv`, passed as the `innovations` argument),
//!   the shipped file runs with dynare-rust's own draws (checked statistically).

mod common;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn refdir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/examples72").join(name)
}

fn floats(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(floats).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(floats).collect()
}

/// Copies `files` of the example's reference folder to a scratch folder and runs `dynare-rs
/// <modfile> nograph` there; returns the folder and the console output.
fn run(name: &str, files: &[&str], modfile: &str) -> (PathBuf, String) {
    let work = common::temp_dir(&format!("ex72_{name}_{}", modfile.trim_end_matches(".mod")));
    for f in files {
        std::fs::copy(refdir(name).join(f), work.join(f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    let out =
        Command::new(env!("CARGO_BIN_EXE_dynare-rs")).arg(modfile).arg("nograph").current_dir(&work).output().unwrap();
    let all = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{name}/{modfile} failed:\n{all}");
    (work, all)
}

/// `<period>,v1,v2,...` CSV written by dynare-rs: one row per period, without the period column.
fn csv_rows(p: &Path) -> Vec<Vec<f64>> {
    let s = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    s.lines().skip(1).map(|l| l.split(',').skip(1).map(|x| x.parse().unwrap()).collect()).collect()
}

fn json(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}

/// Max abs difference between a (variables x periods) reference and a (periods x variables) CSV.
fn simul_err(want: &[Vec<f64>], got: &[Vec<f64>]) -> f64 {
    assert_eq!(got.len(), want[0].len(), "number of periods");
    let mut err = 0.0f64;
    for (j, row) in want.iter().enumerate() {
        for (t, w) in row.iter().enumerate() {
            err = err.max((got[t][j] - w).abs());
        }
    }
    err
}

fn perfect_foresight(name: &str) {
    let r = common::json_ref::<Value>(refdir(name).join("ref.json"));
    let (work, _) = run(name, &[&format!("{name}.mod")], &format!("{name}.mod"));
    let got = csv_rows(&work.join(name).join("Output").join(format!("{name}_endo_simul.csv")));
    let err = simul_err(&rows(&r["endo_simul"]), &got);
    println!("{name}: endo_simul max abs error {err:.2e}");
    assert!(err < 1e-8, "{name}: endo_simul error {err}");
}

/// `check` after `steady` with a non-zero exogenous steady state from `initval` (z = 1).
#[test]
fn perfect_foresight_rbc() {
    perfect_foresight("perfect_foresight_rbc");
}

#[test]
fn perfect_foresight_expectation_errors() {
    perfect_foresight("perfect_foresight_expectation_errors");
}

#[test]
fn rbc_occbin() {
    let r = common::json_ref::<Value>(refdir("rbc_occbin").join("ref.json"));
    let (work, _) = run("rbc_occbin", &["rbc_occbin.mod"], "rbc_occbin.mod");
    for k in ["piecewise", "linear"] {
        let got = csv_rows(&work.join("rbc_occbin/Output").join(format!("rbc_occbin_occbin_{k}.csv")));
        let want = rows(&r[k]);
        assert_eq!(got.len(), want.len());
        let err =
            got.iter().zip(&want).flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs())).fold(0.0, f64::max);
        println!("rbc_occbin {k}: max abs error {err:.2e}");
        assert!(err < 1e-8, "rbc_occbin {k}: {err}");
    }
}

/// Mode (relative to its s.d.), log posterior at the mode, Laplace approximation, and the MCMC
/// posterior means / MHM within Monte Carlo error of a Dynare 7.2 run.
fn check_estimation(name: &str, res: &Value, r: &Value, laplace_tol: f64, mc_tol: f64, mhm_tol: f64) {
    assert_eq!(res["parameter_names"], r["names"], "{name}: parameter names");
    let (mode, sd) = (floats(&r["mode"]), floats(&r["mode_sd"]));
    let gmode = floats(&res["mode"]);
    let zmode = gmode.iter().zip(&mode).zip(&sd).map(|((a, b), s)| ((a - b) / s).abs()).fold(0.0, f64::max);
    let gsd = floats(&res["mode_sd"]);
    let rsd = gsd.iter().zip(&sd).map(|(a, b)| ((a - b) / b).abs()).fold(0.0, f64::max);
    let dlogp = (-res["minus_log_posterior_at_mode"].as_f64().unwrap() - r["log_density"].as_f64().unwrap()).abs();
    let dlap = (res["log_data_density_laplace"].as_f64().unwrap() - r["laplace"].as_f64().unwrap()).abs();
    let (pm, ps) = (floats(&r["posterior_mean"]), floats(&r["posterior_std"]));
    let gpm = floats(&res["mcmc"]["posterior_mean"]);
    let zpm = gpm.iter().zip(&pm).zip(&ps).map(|((a, b), s)| ((a - b) / s).abs()).fold(0.0, f64::max);
    let dmhm = (res["mcmc"]["log_data_density_mhm"].as_f64().unwrap() - r["mhm"].as_f64().unwrap()).abs();
    println!(
        "{name}: mode {zmode:.1e} sd, mode s.d. rel {rsd:.1e}, log post {dlogp:.1e}, Laplace {dlap:.1e}, post. mean {zpm:.2} sd, MHM {dmhm:.3}"
    );
    assert!(zmode < 1e-3, "{name}: mode differs by {zmode} s.d.");
    assert!(rsd < 1e-3, "{name}: mode s.d. relative error {rsd}");
    // Where the optimizer stops depends on last-bit libm differences: 1.02e-6 for gali_2015 on
    // Windows.
    let tol_logp = if cfg!(target_os = "macos") { 1e-6 } else { 1e-5 };
    assert!(dlogp < tol_logp, "{name}: log posterior at the mode differs by {dlogp}");
    assert!(dlap < laplace_tol, "{name}: Laplace approximation differs by {dlap}");
    assert!(zpm < mc_tol, "{name}: posterior means differ by {zpm} posterior s.d.");
    assert!(dmhm < mhm_tol, "{name}: MHM differs by {dmhm}");
}

/// mode_compute = 4, mode_check, loglinear/logdata, 2 x 20000 MH draws (as shipped, ~4 s).
#[test]
fn schorfheide_2000() {
    let r = common::json_ref::<Value>(refdir("schorfheide_2000").join("ref.json"));
    let (work, all) =
        run("schorfheide_2000", &["schorfheide_2000.mod", "schorfheide_2000_data.m"], "schorfheide_2000.mod");
    assert!(!all.contains("option `order`"), "order=1 is the default linear estimation:\n{all}");
    let res = json(&work.join("schorfheide_2000/Output/schorfheide_2000_results.json"));
    check_estimation("schorfheide_2000", &res, &r, 1e-3, 0.3, 0.3);
}

/// The two estimations (nobs = 1, then nobs = 100; diffuse_filter, mode_compute = 5) on the data
/// Dynare 7.2 simulated; the results are those of the second one.
#[test]
fn gali_2015_estimation() {
    let r = common::json_ref::<Value>(refdir("gali_2015").join("ref.json"));
    let (work, _) = run("gali_2015", &["gali_2015.mod", "sim_data.m"], "gali_2015.mod");
    let res = json(&work.join("gali_2015/Output/gali_2015_results.json"));
    check_estimation("gali_2015", &res, &r, 1e-3, 0.3, 0.3);
}

/// `datatomfile` after `stoch_simul(periods=100)`, MATLAB statements of the example
/// (`[a, b] = f(...)`, `figure(h)` / `title(...)` lines without ';'), and the refusal of a
/// `<fname>_prior_restrictions.m` file (MATLAB code that Dynare adds to the log prior).
#[test]
fn gali_2015_datatomfile_and_matlab_statements() {
    let src = std::fs::read_to_string(refdir("gali_2015").join("gali_2015.mod")).unwrap();
    let head = &src[..src.find("estimated_params;").unwrap()];
    let tail = "stoch_simul(periods=100,drop=0,irf=0) yhat;\n\
                datatomfile('sim_out',{'yhat'});\n\
                [density(:,1),density(:,2)] = kernel_density_estimate(v,512,length(v),bw,'gaussian');\n\
                figure(hh_fig)\nsubplot(3,1,1)\nplot(density(:,1),density(:,2));\ntitle('Prior')\n";
    let work = common::temp_dir("ex72_gali_datatomfile");
    std::fs::write(work.join("g.mod"), format!("{head}{tail}")).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_dynare-rs")).arg("g.mod").current_dir(&work).output().unwrap();
    let all = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "{all}{}", String::from_utf8_lossy(&out.stderr));
    assert!(all.contains("MATLAB statement `title ( 'Prior' );` is not supported"), "{all}");
    let m = std::fs::read_to_string(work.join("sim_out.m")).unwrap();
    let body = &m[m.find("yhat = [").unwrap() + 8..m.find("];").unwrap()];
    let vals: Vec<f64> = body.split_whitespace().map(|x| x.parse().unwrap()).collect();
    assert_eq!(vals.len(), 100, "{m}");
    assert!(vals.iter().all(|v| v.is_finite()) && vals.iter().any(|&v| v != 0.0));

    // the shipped prior restrictions file
    let work = common::temp_dir("ex72_gali_prior_restrictions");
    for f in ["gali_2015.mod", "sim_data.m"] {
        std::fs::copy(refdir("gali_2015").join(f), work.join(f)).unwrap();
    }
    std::fs::copy(
        refdir("gali_2015").join("gali_2015_prior_restrictions.m.txt"),
        work.join("gali_2015_prior_restrictions.m"),
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_dynare-rs")).arg("gali_2015.mod").current_dir(&work).output().unwrap();
    let all = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(!out.status.success() && all.contains("user-defined prior restrictions are MATLAB code"), "{all}");
}

/// Frequentist IRF matching: `steady_state_model` assigning parameters (M_.params after
/// `steady`), `values (xx)` / `weights (ww)` workspace vectors, newrat + additional csminwel step.
#[test]
fn rbc_irf_matching_frequentist() {
    let r = common::json_ref::<Value>(refdir("rbc_irf_matching").join("ref.json"));
    let p = &r["part1"];
    let (work, _) = run("rbc_irf_matching", &["rbc_irf_matching_part1.mod"], "rbc_irf_matching_part1.mod");
    let res = json(&work.join("rbc_irf_matching_part1/Output/rbc_irf_matching_part1_mom_results.json"));
    let (x, sd) = (floats(&res["xparam1"]), floats(&res["std_at_mode"]));
    let (rx, rsd) = (floats(&p["mode"]), floats(&p["sd"]));
    let zx = x.iter().zip(&rx).zip(&rsd).map(|((a, b), s)| ((a - b) / s).abs()).fold(0.0, f64::max);
    let esd = sd.iter().zip(&rsd).map(|(a, b)| ((a - b) / b).abs()).fold(0.0, f64::max);
    let emm = floats(&res["model_moments"])
        .iter()
        .zip(floats(&p["model_moments"]))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let eq = (res["Q"].as_f64().unwrap() - p["Q"].as_f64().unwrap()).abs();
    println!("rbc_irf_matching part 1: mode {zx:.1e} sd, s.d. rel {esd:.1e}, model IRFs {emm:.1e}, Q {eq:.1e}");
    // csminwel's stopping point moves with the libm: (8.7e-4, 3.8e-4, 8.1e-5, 9.3e-7) on Windows.
    let k = if cfg!(target_os = "macos") { 1.0 } else { 50.0 };
    assert!(zx < k * 1e-4 && esd < k * 1e-4 && emm < k * 1e-5 && eq < k * 1e-7, "{zx} {esd} {emm} {eq}");
}

/// Bayesian IRF matching on the CSV data (`importdata`, `s.data(2:80,1)`, `1./x`) with the slice
/// sampler (no mode finding, chains started at prior draws). The posterior is symmetric in the
/// two roots (rho_g_1 = r1 + r2, rho_g_2 = -r1*r2), so it has two modes, (0.963, 0.15) and
/// (0.15, 0.963), and each chain stays in the mode its initial prior draw falls into: which
/// labelling the run lands on is decided by the random draws, not by Dynare's choices (no mode,
/// both chains from `Prior.draw()` as in `posterior_sampler_initialization.m`, parameters in
/// declaration order, the same slice `W1`). Dynare 7.2 itself, run with `set_dynare_seed(s)` for
/// s = 0..63 (`reference/examples72/dump_est/seeds_irfm.m`), gives both chains on (0.963, 0.15)
/// for 23 seeds (the default stream, `ref.json`, is one), both on (0.15, 0.963) for 10, and one
/// chain on each for 31 (means ~0.55/0.55, MHM 204.75 +- 0.57). Over 300 `--seed`s dynare-rs gives
/// 34% / 13% / 53% and MHM 204.75 +- 0.70 for the split runs. When both chains share a mode
/// the MHM is 199.35 +- 0.20 (range 199.02-199.77) in Dynare and 199.44 +- 0.17 in dynare-rs, so
/// the 0.24 between the default runs (199.45 against 199.21, the low end of Dynare's spread) is
/// Monte Carlo error of the 2 x 25 kept draws. The default dynare-rs stream lands on the swapped
/// labelling; the sorted means are compared, the MHM within 3 Dynare seed-to-seed s.d.
#[test]
fn rbc_irf_matching_bayesian_slice() {
    let r = common::json_ref::<Value>(refdir("rbc_irf_matching").join("ref.json"));
    let p = &r["part12"];
    let (work, all) = run(
        "rbc_irf_matching",
        &["rbc_irf_matching_part12.mod", "rbc_irf_matching_data.csv"],
        "rbc_irf_matching_part12.mod",
    );
    assert_eq!(all.matches("Computing Hessian at the mode").count(), 1, "the slice run must skip the mode:\n{all}");
    let res = json(&work.join("rbc_irf_matching_part12/method_of_moments/rbc_irf_matching_part12_mom_mcmc.json"));
    // The data / model IRF comparison is at the posterior mean (Dynare 7.2: xparam1 =
    // get_posterior_parameters('mean')), not at the prior mode the chains start from: the model
    // response of ghat at t = 2 is rho_g_1 = root_g_1 + root_g_2.
    let row = all.lines().rev().find(|l| l.starts_with("ghat eps_g (2)")).expect("IRF comparison table");
    let model_irf2: f64 = row.split_whitespace().last().unwrap().parse().unwrap();
    let means = floats(&res["posterior_mean"]);
    assert!((model_irf2 - means.iter().sum::<f64>()).abs() < 1e-4, "{row} vs means {means:?}");
    let mut g = means;
    let mut w = floats(&p["posterior_mean"]);
    g.sort_by(f64::total_cmp);
    w.sort_by(f64::total_cmp);
    let d: Vec<f64> = g.iter().zip(&w).map(|(a, b)| (a - b).abs()).collect();
    let dmhm = (res["log_data_density_mhm"].as_f64().unwrap() - p["mhm"].as_f64().unwrap()).abs();
    println!("rbc_irf_matching part 2: sorted posterior means {g:?} vs {w:?}, MHM {dmhm:.3}");
    // Both chains in one mode (a split run has both means near 0.55).
    assert!(g[0] < 0.3 && g[1] > 0.9, "the chains split between the two modes: {g:?}");
    // Across Dynare seeds the small root's mean ranges 0.132-0.158 and the large one's 0.9627-0.9640.
    assert!(d[0] < 0.03 && d[1] < 0.002 && dmhm < 0.6, "{d:?} {dmhm}");
}

/// PAC model: `pac.initialize` / `pac.update.expectation` (h0/h1 and growth neutrality
/// parameters).
#[test]
fn pac_model_expectation_parameters() {
    let r = common::json_ref::<Value>(refdir("pac_model").join("ref.json"));
    let mut mf = dynare::modfile::parse_mod_file(&refdir("pac_model").join("pac_model.mod"), &[]).unwrap();
    for ci in 0..mf.commands.len() {
        if let dynare::modfile::Command::Other(n, o) = mf.commands[ci].clone()
            && dynare::varmodels::handles(&n)
        {
            if n == dynare::varmodels::SIMUL_BACKWARD {
                break; // the parameters after pac.update.expectation
            }
            dynare::varmodels::run(&mut mf, &n, &o).unwrap();
        }
    }
    let names: Vec<String> =
        r["param_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let vals = floats(&r["params"]);
    assert_eq!(names.len(), mf.model.param_names.len());
    let mut err = 0.0f64;
    for (i, n) in names.iter().enumerate() {
        // names of the generated parameters carry the auxiliary variables' numbering
        if !n.starts_with("h_") && !n.contains("growth_neutrality") {
            assert_eq!(&mf.model.param_names[i], n);
        }
        err = err.max((mf.model.params[i] - vals[i]).abs());
    }
    println!("pac_model: parameters max abs error {err:.2e}");
    assert!(err < 1e-10, "{err}");
}

/// `<period>,v1,...` CSV with its header: (names, rows).
fn csv_named(p: &Path) -> (Vec<String>, Vec<Vec<f64>>) {
    let s = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let names = s.lines().next().unwrap().split(',').skip(1).map(str::to_string).collect();
    (names, csv_rows(p))
}

/// PAC model, the whole example with MATLAB's shocks replayed: `simul_backward_model` (paths of
/// the 5000 periods) and `pac.estimate.nls` on the simulated data (fmincon in Dynare,
/// Gauss-Newton here: equal up to fmincon's tolerances).
#[test]
fn pac_model_simulation_and_nls_estimation() {
    let r = common::json_ref::<Value>(refdir("pac_model").join("ref_full.json"));
    let work = common::temp_dir("ex72_pac_model_replay");
    let src = std::fs::read_to_string(refdir("pac_model").join("pac_model.mod")).unwrap();
    let call = "simul_backward_model(initialconditions, 5000);";
    assert!(src.contains(call));
    let src = src.replace(call, "simul_backward_model(initialconditions, 5000, dseries('pac_model_shocks.csv'));");
    std::fs::write(work.join("pac_replay.mod"), src).unwrap();
    std::fs::copy(refdir("pac_model").join("pac_model_shocks.csv"), work.join("pac_model_shocks.csv")).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_dynare-rs"))
        .arg("pac_replay.mod")
        .arg("nograph")
        .current_dir(&work)
        .output()
        .unwrap();
    let all = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "pac_replay failed:\n{all}");
    // simulated paths
    let (names, got) = csv_named(&work.join("pac_replay/Output/pac_replay_TrueData.csv"));
    let want_names: Vec<&str> = r["sim_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(names, want_names);
    assert_eq!(got.len(), 5010);
    let mut err = 0.0f64;
    for (k, t) in floats(&r["sim_rows"]).iter().enumerate() {
        let w = floats(&r["sim_data"][k]);
        for j in 0..w.len() {
            err = err.max((got[*t as usize - 1][j] - w[j]).abs());
        }
    }
    for (k, w) in rows(&r["sim_every50"]).iter().enumerate() {
        for j in 0..w.len() {
            err = err.max((got[50 * (k + 1) - 1][j] - w[j]).abs());
        }
    }
    let colsum = floats(&r["sim_colsum"]);
    let mut cerr = 0.0f64;
    for (j, w) in colsum.iter().enumerate() {
        let g: f64 = got[10..].iter().map(|row| row[j]).sum();
        cerr = cerr.max((g - w).abs() / w.abs().max(1.0));
    }
    // estimates
    let res = json(&work.join("pac_replay/Output/pac_replay_pac_pacman.json"));
    assert_eq!(res["parnames"], r["parnames"]);
    let (g, w) = (floats(&res["estimator"]), floats(&r["estimator"]));
    let eerr = g.iter().zip(&w).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let (gs, ws) = (floats(&res["student"]), floats(&r["student"]));
    let serr = gs.iter().zip(&ws).map(|(a, b)| ((a - b) / b).abs()).fold(0.0, f64::max);
    let ssr_err = ((res["ssr"].as_f64().unwrap() - r["ssr"].as_f64().unwrap()) / r["ssr"].as_f64().unwrap()).abs();
    let r2_err = (res["R2"].as_f64().unwrap() - r["R2"].as_f64().unwrap()).abs();
    println!(
        "pac_model: paths {err:.2e} (column sums {cerr:.2e}), estimates {eerr:.2e}, t-stats {serr:.2e} (rel), SSR {ssr_err:.2e} (rel), R2 {r2_err:.2e}"
    );
    assert!(err < 1e-10 && cerr < 1e-12, "{err} {cerr}");
    assert!(eerr < 1e-6 && serr < 1e-5 && ssr_err < 1e-10 && r2_err < 1e-9, "{eerr} {serr} {ssr_err} {r2_err}");
}

/// PAC model, the shipped file (shocks drawn by dynare-rust): the estimates are close to the
/// calibration (as the comment of the example says) and the shocks have the calibrated variances.
#[test]
fn pac_model_native_draws() {
    let (work, log) = run("pac_model", &["pac_model.mod"], "pac_model.mod");
    // the table of pac.estimate.nls (dyn_table.m), layout as MATLAB's
    for l in [
        "                    NLS Estimation of equation 'zpac'",
        "    Dependent Variable: diff(z)",
        "    Observations: 4001 from 2005Q1 to 3005Q1",
        "                Estimates          t-statistic          Std. Error   ",
        "             ________________    ________________    ________________",
        "_____________________________________________________________________",
    ] {
        assert!(log.lines().any(|x| x == l), "missing line {l:?} in:\n{log}");
    }
    for k in ["    e_c_m ", "    c_z_1 ", "    c_z_2 ", "    R^2: ", "    s^2: "] {
        assert!(log.lines().any(|x| x.starts_with(k)), "missing {k:?}");
    }
    let (names, got) = csv_named(&work.join("pac_model/Output/pac_model_TrueData.csv"));
    assert_eq!(names, ["y", "x", "z", "v", "ex", "ey", "ez"]);
    assert_eq!(got.len(), 5010);
    for (j, var) in [(4, 0.10), (5, 0.15), (6, 0.05)] {
        let v = got[10..].iter().map(|r| r[j] * r[j]).sum::<f64>() / 5000.0;
        assert!((v / var - 1.0).abs() < 0.1, "variance of {}: {v} vs {var}", names[j]);
    }
    let res = json(&work.join("pac_model/Output/pac_model_pac_pacman.json"));
    let est = floats(&res["estimator"]);
    println!("pac_model (dynare-rust draws): estimates {est:?}");
    for (e, c) in est.iter().zip([0.1, 0.7, -0.3]) {
        assert!((e - c).abs() < 0.06, "{est:?}");
    }
}

/// nk_ramsey_osr.mod with -DRamsey=0: the OSR parameter table is `dyntable`'s (MATLAB: header
/// `Variables  Value`, `%10.6f` values).
#[test]
fn nk_ramsey_osr_parameter_table() {
    let work = common::temp_dir("ex72_nk_osr_table");
    std::fs::copy(refdir("nk_ramsey_osr").join("nk_ramsey_osr.mod"), work.join("nk_ramsey_osr.mod")).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_dynare-rs"))
        .args(["nk_ramsey_osr.mod", "nograph", "-DRamsey=0"])
        .current_dir(&work)
        .output()
        .unwrap();
    let log = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "{log}");
    let l: Vec<&str> = log.lines().collect();
    let i = l.iter().position(|x| *x == "OPTIMAL VALUE OF THE PARAMETERS").expect("title");
    assert_eq!(l[i + 1], "Variables        Value");
    assert_eq!(l[i + 2], "alpha       100.000000");
    assert_eq!(l[i + 4], "Objective function :          51.1572");
}
