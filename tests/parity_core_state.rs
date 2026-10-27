//! Positional blocks and global `options_` (parity audit group core-state): the generated
//! driver.m runs shocks / initval / histval / endval blocks and writes the options of each
//! command into the global `options_` at that position, so a command sees only what was
//! written before it. Expected values are derived by hand on small models (and checked against
//! the Dynare 7.2 preprocessor's driver.m ordering).
mod common;

use std::path::Path;

use dynare::driver::{RunOptions, run};

fn run_mod(tag: &str, body: &str) -> std::path::PathBuf {
    let dir = common::temp_dir(tag);
    let f = dir.join(format!("{tag}.mod"));
    std::fs::write(&f, body).unwrap();
    run(&f, &RunOptions { quiet: true, ..Default::default() }).unwrap();
    dir.join(tag).join("Output")
}

fn stoch_simul_json(out: &Path, tag: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(out.join(format!("{tag}_stoch_simul.json"))).unwrap()).unwrap()
}

/// `<fname>_endo_simul.csv`: rows (period, values by endogenous variable).
fn simul(out: &Path, tag: &str) -> Vec<Vec<f64>> {
    let s = std::fs::read_to_string(out.join(format!("{tag}_endo_simul.csv"))).unwrap();
    s.lines().skip(1).map(|l| l.split(',').map(|v| v.parse().unwrap()).collect()).collect()
}

fn exo_simul(out: &Path, tag: &str) -> Vec<Vec<f64>> {
    let s = std::fs::read_to_string(out.join(format!("{tag}_exo_simul.csv"))).unwrap();
    s.lines().skip(1).map(|l| l.split(',').map(|v| v.parse().unwrap()).collect()).collect()
}

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() < 1e-8 * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

const AR1: &str = "var y; varexo e; parameters a; a = 0.5;\nmodel; y = a*y(-1) + e; end;\n";

/// core-state-2: a shocks block written after a command does not reach it
/// (Sigma_e = 1 -> var(y) = 1/(1-0.25)).
#[test]
fn shocks_block_after_command_is_not_seen() {
    let body = format!("{AR1}shocks; var e = 1; end;\nstoch_simul(order=1, irf=0);\nshocks; var e = 4; end;\n");
    let out = run_mod("cs_shocks_after", &body);
    close(stoch_simul_json(&out, "cs_shocks_after")["var"][0][0].as_f64().unwrap(), 1.0 / 0.75, "var(y)");
}

/// ... and a block written before the command is seen (Sigma_e = 4).
#[test]
fn shocks_block_before_command_is_seen() {
    let body = format!(
        "{AR1}shocks; var e = 1; end;\nstoch_simul(order=1, irf=0);\nshocks; var e = 4; end;\nstoch_simul(order=1, irf=0);\n"
    );
    let out = run_mod("cs_shocks_both", &body);
    close(stoch_simul_json(&out, "cs_shocks_both")["var"][0][0].as_f64().unwrap(), 4.0 / 0.75, "var(y)");
}

/// core-state-3 / 8: a shocks(overwrite) block after a perfect-foresight simulation does not
/// change it (e = 1 at period 1: y = 1, 0.5, 0.25, ...).
#[test]
fn det_shocks_overwrite_after_simulation() {
    let body = format!(
        "{AR1}initval; y = 0; e = 0; end;\nshocks; var e; periods 1; values 1; end;\n\
perfect_foresight_setup(periods=5);\nperfect_foresight_solver;\n\
shocks(overwrite); var e; periods 2; values 10; end;\n"
    );
    let out = run_mod("cs_det_shocks", &body);
    let y = simul(&out, "cs_det_shocks");
    // columns: period, y; row 0 is period 0
    close(y[1][1], 1.0, "y(1)");
    close(y[2][1], 0.5, "y(2)");
    close(y[3][1], 0.25, "y(3)");
    let e = exo_simul(&out, "cs_det_shocks");
    close(e[1][1], 1.0, "e(1)");
    close(e[2][1], 0.0, "e(2)");
}

/// core-state-4 / 6: an initval block after the simulation does not change its initial value.
#[test]
fn initval_after_simulation() {
    let body = format!(
        "{AR1}initval; y = 1; e = 0; end;\nperfect_foresight_setup(periods=4);\nperfect_foresight_solver;\n\
initval; y = 5; end;\n"
    );
    let out = run_mod("cs_initval", &body);
    let y = simul(&out, "cs_initval");
    close(y[0][1], 1.0, "y(0)");
}

/// core-state-4: an initval expression reads the current oo_.steady_state (here the result of
/// the earlier steady: y = 2 with e = 1), so the second block sets y = 0.5 * 2 = 1.
#[test]
fn initval_expression_reads_current_steady_state() {
    let body = format!(
        "{AR1}initval; e = 1; y = 0; end;\nsteady;\ninitval; y = 0.5*y; end;\n\
perfect_foresight_setup(periods=3);\nperfect_foresight_solver;\n"
    );
    let out = run_mod("cs_initval_expr", &body);
    let y = simul(&out, "cs_initval_expr");
    close(y[0][1], 1.0, "y(0)");
}

/// core-state-5 / 7: each histval block replaces the previous one, and a block written after
/// the simulation does not apply to it.
#[test]
fn histval_blocks_reset_and_are_positional() {
    let body = "var y k; varexo e; parameters a; a = 0.5;\nmodel; y = a*y(-1) + e; k = 0.9*k(-1) + e; end;\n\
initval; y = 0; k = 0; e = 0; end;\nhistval; y(0) = 1; k(0) = 2; end;\nhistval; y(0) = 3; end;\n\
perfect_foresight_setup(periods=3);\nperfect_foresight_solver;\nhistval; y(0) = 10; end;\n";
    let out = run_mod("cs_histval", body);
    let s = simul(&out, "cs_histval");
    close(s[0][1], 3.0, "y(0)");
    close(s[0][2], 0.0, "k(0)");
}

/// core-state-10: stoch_simul works at the exogenous steady state set by initval
/// (y = e / (1 - 0.5) = 2 for e = 1).
#[test]
fn stoch_simul_uses_exo_steady_state() {
    let body =
        format!("{AR1}initval; e = 1; y = 0; end;\nsteady;\nshocks; var e = 0.1; end;\nstoch_simul(order=1, irf=0);\n");
    let out = run_mod("cs_exoss", &body);
    close(stoch_simul_json(&out, "cs_exoss")["dr"]["ys"][0].as_f64().unwrap(), 2.0, "dr.ys");
}

/// analysis-forecast-6: forecast(periods=3, conf_sig=0.5) leaves options_.periods = 3 and
/// options_.forecasts.conf_sig = 0.5 for the next forecast.
#[test]
fn forecast_options_persist() {
    let body = format!(
        "{AR1}initval; y = 0; e = 0; end;\nshocks; var e = 1; end;\nstoch_simul(order=1, irf=0);\n\
forecast(periods=3, conf_sig=0.5);\n"
    );
    let a = run_mod("cs_fc1", &body);
    let body2 = format!("{body}forecast;\n");
    let b = run_mod("cs_fc2", &body2);
    let (fa, fb) = (
        std::fs::read_to_string(a.join("cs_fc1_forecast.csv")).unwrap(),
        std::fs::read_to_string(b.join("cs_fc2_forecast.csv")).unwrap(),
    );
    assert_eq!(fa, fb, "the second forecast keeps periods = 3 and conf_sig = 0.5");
    assert_eq!(fb.lines().count() - 1, 3, "three forecast periods");
}

/// core-state-1 / 13 / 17: options written by an estimation stay in options_ for the next one
/// (first_obs = 11 and mode_compute = 4 are not repeated); the second estimation's smoother
/// covers the same 190 observations.
#[test]
fn estimation_options_persist() {
    let refdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/ml");
    let model: String = std::fs::read_to_string(refdir.join("model_ml.mod"))
        .unwrap()
        .lines()
        .take_while(|l| !l.starts_with("@#ifdef"))
        .map(|l| format!("{l}\n"))
        .collect();
    let dir = common::temp_dir("cs_est_persist");
    std::fs::copy(refdir.join("model_ml_obs.csv"), dir.join("model_ml_obs.csv")).unwrap();
    let f = dir.join("cs_est_persist.mod");
    let body = format!(
        "{model}estimation(datafile='model_ml_obs.csv', first_obs=11, mode_compute=4, nograph, smoother);\n\
estimation(datafile='model_ml_obs.csv', nograph, smoother);\n"
    );
    std::fs::write(&f, body).unwrap();
    run(&f, &RunOptions { quiet: true, ..Default::default() }).unwrap();
    let s = std::fs::read_to_string(dir.join("cs_est_persist/Output/cs_est_persist_smoothed_variables.csv")).unwrap();
    assert_eq!(s.lines().count() - 1, 190, "second estimation keeps first_obs = 11");
}

/// core-state-11: with the default lik_init = 1 estimation uses qz_criterium = 1-1e-6, so an
/// unobserved random walk is an explosive root (info 3) and the initial checks fail.
#[test]
fn estimation_counts_unit_root_as_explosive() {
    let dir = common::temp_dir("cs_unit_root");
    let mut data = String::from("y\n");
    let mut v = 0.0f64;
    for t in 0..60 {
        v = 0.5 * v + ((t * 7919 % 13) as f64 - 6.0) / 6.0;
        data.push_str(&format!("{v}\n"));
    }
    std::fs::write(dir.join("d.csv"), data).unwrap();
    let f = dir.join("cs_unit_root.mod");
    std::fs::write(
        &f,
        "var y z; varexo e u; parameters rho; rho = 0.5;\nmodel; y = rho*y(-1) + e; z = z(-1) + u; end;\n\
shocks; var e = 1; var u = 1; end;\nvarobs y;\nestimated_params; rho, 0.5, -0.9, 0.9; stderr e, 1, 0.01, 5; end;\n\
estimation(datafile='d.csv', mode_compute=4, nograph);\n",
    )
    .unwrap();
    let r = run(&f, &RunOptions { quiet: true, ..Default::default() });
    assert!(r.is_err(), "the unit root must be explosive under qz_criterium = 1-1e-6");
}

// ---------------------------------------------------------------------------
// core-state part 2: the estimated point is the global M_ afterwards (14, 15), oo_.dr
// ---------------------------------------------------------------------------

/// A deterministic AR(1) sample (rho = 0.8, LCG shocks) for the estimation tests.
fn ar1_csv() -> String {
    let (mut s, mut y) = (12345u64, 0.0f64);
    let mut out = String::from("y\n");
    for _ in 0..60 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 3.4641; // unit variance
        y = 0.8 * y + u;
        out += &format!("{y}\n");
    }
    out
}

fn run_est(tag: &str, body: &str) -> std::path::PathBuf {
    let dir = common::temp_dir(tag);
    std::fs::write(dir.join("ar1.csv"), ar1_csv()).unwrap();
    std::fs::write(dir.join(format!("{tag}.mod")), body).unwrap();
    run(&dir.join(format!("{tag}.mod")), &RunOptions { quiet: true, ..Default::default() })
        .unwrap_or_else(|e| panic!("{tag}: {e:#}"));
    dir.join(tag).join("Output")
}

fn est_mode(out: &Path, tag: &str) -> f64 {
    let r: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join(format!("{tag}_results.json"))).unwrap()).unwrap();
    r["mode"][0].as_f64().unwrap()
}

const EST_HEAD: &str = "var y z; varexo e; parameters a;\na = 0.5;\n\
    model; y = a*y(-1) + e; z = 0.3*z(-1) + 0.2*z(+1) + y^2; end;\n\
    shocks; var e = 1; end;\nvarobs y;\n\
    estimated_params; a, 0.5, 0, 0.99; end;\n";

/// 14: stoch_simul(order=2) after the estimation uses the estimated `a`
/// (var(y) = 1/(1-a^2) at every order; the calibration gives 1/0.75).
#[test]
fn estimated_point_reaches_order2_stoch_simul() {
    let body = format!(
        "{EST_HEAD}estimation(datafile=ar1.csv, mode_compute=4, nograph, nodisplay);\nstoch_simul(order=2, irf=0, nograph, noprint);\n"
    );
    let out = run_est("cs_est_o2", &body);
    let a = est_mode(&out, "cs_est_o2");
    assert!((a - 0.5).abs() > 0.05, "the estimate should differ from the calibration: {a}");
    close(stoch_simul_json(&out, "cs_est_o2")["var"][0][0].as_f64().unwrap(), 1.0 / (1.0 - a * a), "var(y)");
}

/// 15: a parameter assignment after the estimation overrides the estimated value.
#[test]
fn parameter_assignment_after_estimation_wins() {
    let body = format!(
        "{EST_HEAD}estimation(datafile=ar1.csv, mode_compute=4, nograph, nodisplay);\na = 0.5;\nstoch_simul(order=1, irf=0, nograph, noprint);\n"
    );
    let out = run_est("cs_est_assign", &body);
    close(stoch_simul_json(&out, "cs_est_assign")["var"][0][0].as_f64().unwrap(), 1.0 / 0.75, "var(y)");
}

/// oo_.dr is empty until a stoch_simul or an estimation computes it: forecast stops.
#[test]
fn forecast_without_decision_rules_errors() {
    let dir = common::temp_dir("cs_fc_empty");
    let body = format!("{AR1}shocks; var e = 1; end;\nforecast(periods=3);\n");
    let f = dir.join("cs_fc_empty.mod");
    std::fs::write(&f, body).unwrap();
    let e = run(&f, &RunOptions { quiet: true, ..Default::default() }).expect_err("oo_.dr is empty");
    assert!(format!("{e:#}").to_lowercase().contains("stoch_simul") || format!("{e:#}").contains("dr"), "{e:#}");
}

/// forecast uses oo_.dr of the last stoch_simul: a later parameter change does not alter it
/// (y_h = 1 + a^h with a = 0.5 from y(0) = 2 and mean m = 1).
#[test]
fn forecast_uses_stored_decision_rules() {
    let body = "var y; varexo e; parameters a; a = 0.5;\nmodel; y = a*y(-1) + (1-a) + e; end;\n\
        initval; y = 1; end;\nsteady;\nshocks; var e = 0.01; end;\nhistval; y(0) = 2; end;\n\
        stoch_simul(order=1, irf=0, nograph, noprint);\na = 0.9;\nforecast(periods=3);\n";
    let out = run_mod("cs_fc_stored", body);
    let s = std::fs::read_to_string(out.join("cs_fc_stored_forecast.csv")).unwrap();
    let head: Vec<&str> = s.lines().next().unwrap().split(',').collect();
    let im = head.iter().position(|h| *h == "mean").expect("mean column");
    let means: Vec<f64> = s.lines().skip(1).map(|l| l.split(',').nth(im).unwrap().parse().unwrap()).collect();
    for (h, m) in means.iter().enumerate() {
        close(*m, 1.0 + 0.5f64.powi(h as i32 + 1), &format!("forecast h={}", h + 1));
    }
}

/// 30: `==` binds looser than `<` (DynareBison.yy): `2 == 2 < 3` is `2 == (2 < 3)` = 0, so
/// a = 0.5 and var(y) = 1/0.75 (left to right it would be 1 and a = 1.5, explosive).
#[test]
fn equality_binds_looser_than_comparison() {
    let body = "var y; varexo e; parameters a; a = 0.5 + (2 == 2 < 3);\nmodel; y = a*y(-1) + e; end;\n\
        shocks; var e = 1; end;\nstoch_simul(order=1, irf=0, nograph, noprint);\n";
    let out = run_mod("cs_prec", body);
    close(stoch_simul_json(&out, "cs_prec")["var"][0][0].as_f64().unwrap(), 1.0 / 0.75, "var(y)");
}

/// 31: pow() / power() are not model-block functions in Dynare 7.2.
#[test]
fn pow_in_model_block_needs_declaration() {
    let dir = common::temp_dir("cs_pow");
    let f = dir.join("cs_pow.mod");
    std::fs::write(&f, "var y; varexo e;\nmodel; y = pow(y(-1), 2) + e; end;\n").unwrap();
    let e = run(&f, &RunOptions { quiet: true, ..Default::default() }).expect_err("pow is not declared");
    assert!(format!("{e:#}").contains("external_function"), "{e:#}");
}

/// 36: write_latex_prior_table uses the options_.prior_trunc an earlier estimation left.
#[test]
fn latex_prior_table_sees_persistent_prior_trunc() {
    let head = "var y; varexo e; parameters a;\na = 0.5;\nmodel; y = a*y(-1) + e; end;\n\
        shocks; var e = 1; end;\nvarobs y;\nestimated_params; a, beta_pdf, 0.5, 0.2; end;\n";
    let dir = common::temp_dir("cs_latex_trunc");
    std::fs::write(dir.join("ar1.csv"), ar1_csv()).unwrap();
    let f = dir.join("cs_latex_trunc.mod");
    std::fs::write(
        &f,
        format!("{head}estimation(datafile=ar1.csv, mode_compute=4, prior_trunc=1e-4, nograph, nodisplay);\nwrite_latex_prior_table;\n"),
    )
    .unwrap();
    run(&f, &RunOptions { quiet: true, ..Default::default() }).unwrap();
    let tex = dir.join("cs_latex_trunc").join("latex").join("cs_latex_trunc_priors_table.tex");
    let s = std::fs::read_to_string(tex).unwrap();
    assert!(s.contains("Bounds*") && s.contains("prior\\_trunc=1.00e-4"), "{s}");
}

// ---------------------------------------------------------------------------
// core-state part 3
// ---------------------------------------------------------------------------

fn run_res(tag: &str, body: &str) -> (std::path::PathBuf, anyhow::Result<()>) {
    let dir = common::temp_dir(tag);
    std::fs::write(dir.join("ar1.csv"), ar1_csv()).unwrap();
    let f = dir.join(format!("{tag}.mod"));
    std::fs::write(&f, body).unwrap();
    let r = run(&f, &RunOptions { quiet: true, ..Default::default() });
    (dir, r)
}

fn mcmc_jscale(tag: &str, entry: &str) -> f64 {
    let body = format!(
        "var y; varexo e; parameters a;\na = 0.5;\nmodel; y = a*y(-1) + e; end;\nshocks; var e = 1; end;\nvarobs y;\n\
         estimated_params; {entry} end;\n\
         estimation(datafile=ar1.csv, mode_compute=4, mh_replic=200, mh_nblocks=1, mh_jscale=0.3, nograph, nodisplay);\n"
    );
    let (dir, r) = run_res(tag, &body);
    r.unwrap();
    let j: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join(tag).join("Output").join(format!("{tag}_results.json"))).unwrap(),
    )
    .unwrap();
    j["mcmc"]["jscale"].as_f64().or_else(|| j["jscale"].as_f64()).expect("jscale in results")
}

/// 19: the 10th estimated_params field is bayestopt_.jscale (only NaN entries take mh_jscale).
#[test]
fn per_parameter_jscale_is_used() {
    close(mcmc_jscale("cs_js1", "a, 0.5, 0, 0.99, beta_pdf, 0.5, 0.2, , , 0.07;"), 0.07, "jscale");
    close(mcmc_jscale("cs_js2", "a, 0.5, 0, 0.99, beta_pdf, 0.5, 0.2;"), 0.3, "jscale default");
}

/// 23: with calibrated ME covariances and only an ME stderr estimated, H = [0.01 .05; .05 .01]
/// is checked at every evaluation (info 44, penalty = sum of the negative eigenvalues' moduli).
#[test]
fn h_check_with_calibrated_me_covariance() {
    let dir = common::temp_dir("cs_hcheck");
    let f = dir.join("cs_hcheck.mod");
    std::fs::write(
        &f,
        "var y1 y2; varexo e1 e2; model; y1 = 0.5*y1(-1) + e1; y2 = 0.3*y2(-1) + e2; end;\n\
         shocks; var e1 = 1; var e2 = 1; var y1 = 0.01; var y2 = 0.01; var y1, y2 = 0.05; end;\n\
         varobs y1 y2;\nestimated_params; stderr y1, 0.1, 0, 1; end;\n",
    )
    .unwrap();
    let mf = dynare::modfile::parse_mod_file(&f, &[]).unwrap();
    let dsge = dynare::dsge::DsgeModel::new(&mf, None).unwrap();
    let e = dsge.resolve(&[0.1]).expect_err("H not positive definite");
    assert_eq!(e.info, 44);
    close(e.penalty, 0.04, "penalty");
}

/// 24: stoch_simul at order 2 does not run the estimation set-up (an estimated parameter
/// whose initial value is outside its bounds is irrelevant) and accepts solve_algo.
#[test]
fn order2_stoch_simul_ignores_estimation_setup() {
    let body = "var y; varexo e; parameters a;\na = 0.5;\nmodel; y = a*y(-1) + e + 0.1*y(+1) + 0.1*y(-1)^2; end;\n\
        shocks; var e = 1; end;\nvarobs y;\nestimated_params; a, 1.5, 0, 0.99; end;\n\
        stoch_simul(order=2, irf=0, solve_algo=3, nograph, noprint);\n";
    let out = run_mod("cs_o2_algo", body);
    assert!(stoch_simul_json(&out, "cs_o2_algo")["var"][0][0].as_f64().unwrap() > 1.0);
}

/// 28: datatomfile after an order-2 stoch_simul writes its simulation.
#[test]
fn datatomfile_after_order2() {
    let body = "var y; varexo e; parameters a;\na = 0.5;\nmodel; y = a*y(-1) + e + 0.1*y(+1) + 0.1*y(-1)^2; end;\n\
        shocks; var e = 1; end;\nstoch_simul(order=2, periods=10, drop=0, irf=0, nograph, noprint);\n\
        datatomfile('sim_out', {'y'});\n";
    let dir = common::temp_dir("cs_dtf2");
    let f = dir.join("cs_dtf2.mod");
    std::fs::write(&f, body).unwrap();
    run(&f, &RunOptions { quiet: true, ..Default::default() }).unwrap();
    let s = std::fs::read_to_string(dir.join("sim_out.m")).unwrap();
    assert_eq!(s.lines().filter(|l| l.trim().parse::<f64>().is_ok()).count(), 10);
}

/// 10: the exogenous steady state of `initval` is where check / the solution are computed:
/// y = a*y(-1) + y(-1)*e has d/dy(-1) = a + e = 0.8 at e = 0.5.
#[test]
fn check_and_solution_at_exo_steady_state() {
    let dir = common::temp_dir("cs_exo_ss");
    let f = dir.join("cs_exo_ss.mod");
    std::fs::write(
        &f,
        "var y; varexo e; parameters a; a = 0.3;\nmodel; y = a*y(-1) + y(-1)*e; end;\n\
         initval; y = 0; e = 0.5; end;\nshocks; var e = 1; end;\n",
    )
    .unwrap();
    let mf = dynare::modfile::parse_mod_file(&f, &[]).unwrap();
    let sol = dynare::analysis::solve_calibration(&mf, 1.0 + 1e-6).unwrap().unwrap();
    close(sol.dr.ghx[(0, 0)], 0.8, "ghx");
    let o = vec![("noprint".to_string(), dynare::modfile::OptValue::Flag)];
    let c = dynare::analysis::run_check(&mf, &o, None).unwrap();
    close(c.eigval[0].norm(), 0.8, "eigenvalue");
}

/// 27 / 37: homotopy_setup(from_initval_to_endval) moves the exogenous variables from initval
/// to endval; save_params_and_steady_state after endval sees the endval steady state.
#[test]
fn homotopy_initval_to_endval_and_saved_state() {
    let head = "var y; varexo e; parameters a; a = 0.5;\nmodel; y = a*y(-1) + e; end;\n\
        initval; y = 2; e = 1; end;\nendval; y = 4; e = 2; end;\n";
    let dir = common::temp_dir("cs_hom");
    let f = dir.join("cs_hom.mod");
    std::fs::write(
        &f,
        format!(
            "{head}homotopy_setup(from_initval_to_endval); a, 0.4, 0.5; end;\n\
             steady(homotopy_mode=1, homotopy_steps=2);\nsave_params_and_steady_state('p.txt');\n"
        ),
    )
    .unwrap();
    run(&f, &RunOptions { quiet: true, ..Default::default() }).unwrap();
    let s = std::fs::read_to_string(dir.join("p.txt")).unwrap();
    assert!(s.contains("y 4\n") && s.contains("e 2\n"), "{s}");
    // without a steady command: endval overwrites oo_.steady_state
    let f = dir.join("cs_hom2.mod");
    std::fs::write(&f, format!("{head}save_params_and_steady_state('q.txt');\n")).unwrap();
    run(&f, &RunOptions { quiet: true, ..Default::default() }).unwrap();
    let s = std::fs::read_to_string(dir.join("q.txt")).unwrap();
    assert!(s.contains("y 4\n") && s.contains("e 2\n"), "{s}");
    // without endval the option is an error
    let f = dir.join("cs_hom3.mod");
    std::fs::write(
        &f,
        "var y; varexo e; parameters a; a = 0.5;\nmodel; y = a*y(-1) + e; end;\ninitval; y = 2; e = 1; end;\n\
         homotopy_setup(from_initval_to_endval); a, 0.4, 0.5; end;\nsteady(homotopy_mode=1);\n",
    )
    .unwrap();
    let e = run(&f, &RunOptions { quiet: true, ..Default::default() }).expect_err("no endval");
    assert!(format!("{e:#}").contains("from_initval_to_endval"), "{e:#}");
}

/// 38: modfile.json carries a param_init per parameter assignment, before the commands.
#[test]
fn modfile_json_has_param_init() {
    let dir = common::temp_dir("cs_pjson");
    let f = dir.join("cs_pjson.mod");
    std::fs::write(
        &f,
        "var y; varexo e; parameters a b; a = 0.5; b = 0.25;\nmodel; y = a*y(-1) + b*e; end;\n\
         shocks; var e = 1; end;\nsteady;\n",
    )
    .unwrap();
    let ro = RunOptions {
        quiet: true,
        cli: dynare::misc::cli::CliOptions { json: Some("compute".into()), ..Default::default() },
        ..Default::default()
    };
    run(&f, &ro).unwrap();
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("cs_pjson/model/json/modfile.json")).unwrap()).unwrap();
    let st: Vec<(&str, Option<&str>, Option<&str>)> = j["statements"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| (s["statementName"].as_str().unwrap(), s["name"].as_str(), s["value"].as_str()))
        .collect();
    assert_eq!(
        st,
        vec![
            ("param_init", Some("a"), Some("0.5")),
            ("param_init", Some("b"), Some("0.25")),
            ("shocks", None, None),
            ("steady", None, None)
        ]
    );
}

/// var-models 7: the var_expectation parameters of a VAR whose LHS is diff(z) carry the
/// preprocessor's AUX_DIFF name (dynare-preprocessor: var_expectation_model_ve_AUX_DIFF_10_0).
#[test]
fn var_expectation_names_use_aux_diff() {
    let dir = common::temp_dir("cs_vd");
    let f = dir.join("cs_vd.mod");
    std::fs::write(
        &f,
        "var z w q;\nvarexo e_z e_w e_q;\nparameters a b c d beta_;\na=0.3; b=0.1; c=0.5; d=0.2; beta_=0.9;\n\
         var_model(model_name=vv, eqtags=['v_a','v_b']);\n\
         var_expectation_model(model_name=ve, expression=w, auxiliary_model_name=vv, horizon=2, discount=beta_);\n\
         model;\n[name='v_a']\ndiff(z) = a*diff(z(-1)) + b*w(-1) + e_z;\n[name='v_b']\nw = c*w(-1) + d*z(-1) + e_w;\n\
         q = var_expectation(ve) + e_q;\nend;\n",
    )
    .unwrap();
    let mf = dynare::modfile::parse_mod_file(&f, &[]).unwrap();
    let names = &mf.model.param_names;
    assert!(names.iter().any(|n| n == "var_expectation_model_ve_AUX_DIFF_10_0"), "{names:?}");
    assert!(names.iter().any(|n| n == "var_expectation_model_ve_w_0"), "{names:?}");
    assert!(!names.iter().any(|n| n == "var_expectation_model_ve_z_0"), "{names:?}");
}

/// A PAC equation without autoregressive terms is accepted by the 7.2 preprocessor and so by
/// the parser (a2alpha then fails at evaluation, as in MATLAB).
#[test]
fn pac_equation_without_ar_terms_parses() {
    let dir = common::temp_dir("cs_pacn");
    let f = dir.join("cs_pacn.mod");
    std::fs::write(
        &f,
        "var y x z;\nvarexo ex ey ez;\nparameters a_y_1 b_y_1 b_x_1 beta e_c_m;\n\
         a_y_1 = .2; b_y_1 = .1; b_x_1 = -.1; beta = .9; e_c_m = .1;\n\
         var_model(model_name=toto, eqtags=['eq:x', 'eq:y']);\n\
         pac_model(auxiliary_model_name=toto, discount=beta, model_name=pacman);\nmodel;\n\
         [name='eq:y']\ny = a_y_1*y(-1) + ey;\n[name='eq:x']\ndiff(x) = b_x_1*y(-1) + ex;\n\
         [name='zpac']\ndiff(z) = e_c_m*(x(-1)-z(-1)) + pac_expectation(pacman) + ez;\nend;\n",
    )
    .unwrap();
    dynare::modfile::parse_mod_file(&f, &[]).expect("parses");
}

/// dynare_estimation_init.m:546-550 resets options_.noconstant = 0: the option has no effect
/// on an estimation with a non-zero steady state of the observable.
#[test]
fn estimation_resets_noconstant() {
    let mode = |tag: &str, extra: &str| {
        let body = format!(
            "var y; varexo e; parameters a;\na = 0.5;\nmodel; y = 1 + a*y(-1) + e; end;\nshocks; var e = 1; end;\nvarobs y;\n\
             estimated_params; a, 0.5, 0, 0.99; end;\n\
             estimation(datafile=ar1.csv, mode_compute=4, {extra}nograph, nodisplay);\n"
        );
        let out = run_est(tag, &body);
        est_mode(&out, tag)
    };
    close(mode("cs_nc1", "noconstant, "), mode("cs_nc0", ""), "mode");
}
