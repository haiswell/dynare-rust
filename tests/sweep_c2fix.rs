//! Dynare 7.2 tests that failed in dynare-rs after it skipped their MATLAB statements, although
//! the error was not on a MATLAB line (numbers from the MATLAB runs named in
//! reference/sweep_c2fix/ref.json):
//! * `steady` with a non-finite residual at the initial guess retries from `rand` guesses of
//!   MATLAB's global stream (`options_.solve_randomize_initial_guess`), and the flag
//!   `homotopy_marginal_linearization_fallback` is a step of 0.01
//!   (deterministic_simulations/pfwee_marginal_linearization.mod);
//! * `options_.parameter_set = 'calibration'` and the data options that calib_smoother writes
//!   stay for the later shock / initial condition decompositions
//!   (kalman_filter_smoother/rbc_non_stationary_obs.mod);
//! * a MATLAB assignment to a name that is not a Dynare symbol ends with its line
//!   (preprocessor_checks/comments.mod);
//! * calibrated parameters are MATLAB variables, `max`/`floor` in workspace expressions, and
//!   simul_backward_model solves with `options_.solve_algo`
//!   (stochastic-backward-models/solow_cd_with_steadystate.mod, checked as the test does);
//! * MATLAB statements that dynare-rs skips instead of failing on: `set_param_value` with MATLAB
//!   arguments (heterogeneity/*.mod), `box on` (estimation/univariate/bayesian.mod), a dseries
//!   passed to a MATLAB function (ecb/SURGibbs/fulton_fish.mod), `drop` of
//!   heterogeneity_simulate (heterogeneity/ks_commands.mod).
mod common;

use dynare::driver::{RunOptions, run};
use dynare::modfile::{Command, MATLAB_STATEMENT, OptValue, parse_mod_file};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sweep_c2fix")
}

fn setup(tag: &str, files: &[&str]) -> PathBuf {
    let dir = common::temp_dir(tag);
    for f in files {
        let src = refdir().join(f);
        std::fs::copy(&src, dir.join(f)).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
    }
    dir
}

fn quiet() -> RunOptions {
    RunOptions { quiet: true, ..Default::default() }
}

fn reference() -> Value {
    common::json_ref(refdir().join("ref.json"))
}

/// A csv with a header line and a first label column: (header, rows of values).
fn csv(p: &Path) -> (Vec<String>, Vec<Vec<f64>>) {
    let s = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let mut lines = s.lines();
    let head = lines.next().unwrap().split(',').skip(1).map(str::to_string).collect();
    let rows = lines.map(|l| l.split(',').skip(1).map(|x| x.parse().unwrap()).collect()).collect();
    (head, rows)
}

#[test]
fn pfwee_marginal_linearization_from_a_nan_initial_guess() {
    let r = reference();
    let want = r["pfwee_marglin"]["endo_simul"].as_array().unwrap();
    let dir = setup("c2fix_pfwee_marglin", &["pfwee_marglin.mod"]);
    run(&dir.join("pfwee_marglin.mod"), &quiet()).unwrap();
    let (head, rows) = csv(&dir.join("pfwee_marglin/Output/pfwee_marglin_endo_simul.csv"));
    assert_eq!(head, ["Consumption", "Capital", "LoggedProductivity"]);
    for (i, w) in want.iter().enumerate() {
        let w: Vec<f64> = w.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        assert_eq!(rows.len(), w.len(), "periods");
        for (t, (row, w)) in rows.iter().zip(&w).enumerate() {
            assert!((row[i] - w).abs() <= 1e-8 * w.abs().max(1.0), "{}[{t}]: rust {} vs dynare {w}", head[i], row[i]);
        }
    }
}

#[test]
fn shock_decomposition_after_calib_smoother_uses_the_calibration() {
    let r = reference();
    let r = &r["rbc_non_stationary_obs"];
    let comps: Vec<&str> = r["components"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    let dir = setup("c2fix_rbc_nso", &["rbc_non_stationary_obs.mod", "data_rbc_non_stationary_obs.m"]);
    run(&dir.join("rbc_non_stationary_obs.mod"), &quiet()).unwrap();
    let p = dir.join("rbc_non_stationary_obs/Output/rbc_non_stationary_obs_shock_decomposition.csv");
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let mut n = 0;
    for l in s.lines().skip(1) {
        let f: Vec<&str> = l.split(',').collect();
        if f[1] != "z" {
            continue;
        }
        let (t, j) = (f[0].parse::<usize>().unwrap() - 1, comps.iter().position(|c| *c == f[2]).unwrap());
        let (a, w) = (f[3].parse::<f64>().unwrap(), r["z"][j][t].as_f64().unwrap());
        assert!((a - w).abs() < 1e-12, "z {} period {}: rust {a} vs dynare {w}", f[2], t + 1);
        n += 1;
    }
    assert_eq!(n, 6 * 99);
}

#[test]
fn matlab_assignments_without_semicolon_end_with_their_line() {
    let r = reference();
    let dir = setup("c2fix_comments", &["comments.mod"]);
    run(&dir.join("comments.mod"), &quiet()).unwrap();
    let p = dir.join("comments/Output/comments_stoch_simul.json");
    let j: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    // y c k a h b: declaration order, as oo_.var
    for (i, row) in r["comments"]["var"].as_array().unwrap().iter().enumerate() {
        for (k, w) in row.as_array().unwrap().iter().enumerate() {
            let (a, w) = (j["var"][i][k].as_f64().unwrap(), w.as_f64().unwrap());
            assert!((a - w).abs() <= 1e-10 * w.abs().max(1e-3), "var[{i}][{k}]: rust {a} vs dynare {w}");
        }
    }
}

/// The checks of solow_cd_with_steadystate.mod itself: the simulation reaches the long run
/// levels `d.Efficiency*d.EfficiencyGrowth^(rho_x/(1-rho_x))` (0.5) and
/// `d.Population*d.PopulationGrowth^(rho_n/(1-rho_n))` (1.04^19) within 1e-10, over
/// T = 5*floor(log(1e-6)/log(max(rho_x, rho_n))) = 1345 periods (with options_.solve_algo = 9;
/// the default trust region on the block decomposition stops the growth rates early).
#[test]
fn backward_model_with_matlab_parameters_and_solve_algo() {
    let dir = setup("c2fix_solow_cd", &["solow_cd_with_steadystate.mod"]);
    run(&dir.join("solow_cd_with_steadystate.mod"), &quiet()).unwrap();
    let (head, rows) = csv(&dir.join("solow_cd_with_steadystate/Output/solow_cd_with_steadystate_simulations.csv"));
    assert_eq!(rows.len(), 1 + 1345);
    let last = rows.last().unwrap();
    let col = |n: &str| last[head.iter().position(|h| h == n).unwrap()];
    assert!((col("Efficiency") - 0.5).abs() < 1e-10, "Efficiency {}", col("Efficiency"));
    let pop = 1.04f64.powf(0.95 / 0.05);
    assert!((col("Population") - pop).abs() < 1e-10, "Population {} vs {pop}", col("Population"));
    let k = col("PhysicalCapitalStock") / (col("Efficiency") * col("Population"));
    let k_lr = (0.20f64 / (1.0 - 1.0 + 0.02)).powf(1.0 / (1.0 - 0.33));
    assert!((k - k_lr).abs() < 1e-3, "intensive capital {k} vs {k_lr}");
}

/// MATLAB statements that dynare-rs keeps for the "not supported" warning instead of failing.
#[test]
fn matlab_statements_are_skipped() {
    let dir = common::temp_dir("c2fix_skipped");
    let f = dir.join("skipped.mod");
    std::fs::write(
        &f,
        "var y; varexo e; parameters a b;\na = 0.5;\nmodel; y = a*y(-1) + e; end;\n\
s.params.b = 0.3;\nnames = fieldnames(s.params);\nfor i=1:numel(names)\n    param = names{i};\n    \
set_param_value(param, s.params.(param));\nend\n\
x = surgibbs(dseries('data.csv'), {'a'}, 0.5, 1, 10, 5, 1);\n\
plot(1:3)\nbox on\nhold off\n\
shocks; var e; stderr 1; end;\nstoch_simul(order=1, irf=0, nomoments);\n",
    )
    .unwrap();
    let mf = parse_mod_file(&f, &[]).unwrap();
    let skipped: Vec<String> = mf
        .commands
        .iter()
        .filter_map(|c| match c {
            Command::Other(n, o) if n == MATLAB_STATEMENT => Some(o[0].0.clone()),
            _ => None,
        })
        .collect();
    for want in ["set_param_value", "box on"] {
        assert!(skipped.iter().any(|s| s.starts_with(want)), "{want} not skipped: {skipped:?}");
    }
    // b stays uncalibrated, a keeps its value
    assert_eq!(mf.model.params[0], 0.5);
    assert!(mf.model.params[1].is_nan());
    run(&f, &quiet()).unwrap();
}

/// heterogeneity_simulate(drop = 50) sets options_.drop, which +heterogeneity does not read.
#[test]
fn heterogeneity_simulate_accepts_drop() {
    let o = vec![("periods".to_string(), OptValue::Num(300.0)), ("drop".to_string(), OptValue::Num(50.0))];
    let s = dynare::heterogeneity::commands::sim_options(&o, &Default::default()).unwrap();
    assert_eq!(s.periods, Some(300));
}
