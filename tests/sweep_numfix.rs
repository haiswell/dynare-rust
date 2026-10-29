//! Divergences found by running Dynare 7.2's own test suite (reference/sweep_numfix, numbers from
//! the MATLAB runs of the tests named in ref.json):
//! * histval of a lagged exogenous variable (`u(-1)` for `u(-2)` in the model) reaches the
//!   AUX_EXO_LAG variable of the forecast's starting point;
//! * `ramsey_constraints` are complementarity conditions of the Ramsey FOCs in the perfect
//!   foresight solver (solve_algo = 10, stack_solve_algo = 7);
//! * `filter_initial_state` expressions are evaluated with the current parameters;
//! * the slice sampler without `slice_initialize_with_mode` runs its MCMC (also with
//!   `smoother` and `mode_compute=0`), with `prior_trunc = 0` raised to 1e-10;
//! * `cova_compute = 0` gives NaN standard deviations at the mode (linear and nonlinear).
mod common;

use dynare::driver::{RunOptions, run};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sweep_numfix")
}

fn setup(tag: &str, files: &[&str]) -> PathBuf {
    let dir = common::temp_dir(tag);
    for f in files {
        std::fs::copy(refdir().join(f), dir.join(f)).unwrap_or_else(|e| panic!("{f}: {e}"));
    }
    dir
}

fn csv(p: &Path) -> Vec<Vec<String>> {
    let s = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    s.lines().map(|l| l.split(',').map(str::to_string).collect()).collect()
}

fn results(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}

fn quiet() -> RunOptions {
    RunOptions { quiet: true, ..Default::default() }
}

#[test]
fn histval_of_lagged_exogenous_in_forecast() {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    let dir = setup("numfix_histval_sto", &["histval_sto.mod"]);
    run(&dir.join("histval_sto.mod"), &quiet()).unwrap();
    let rows = csv(&dir.join("histval_sto/Output/histval_sto_forecast.csv"));
    assert_eq!(rows[0][..6], ["date", "horizon", "variable", "mean", "lower", "upper"]);
    let mut n = 0;
    for (col, field) in [(3, "Mean"), (4, "HPDinf"), (5, "HPDsup")] {
        for v in ["a", "b", "y", "k"] {
            let want = r["histval_sto_forecast"][field][v].as_array().unwrap();
            let got: Vec<f64> = rows[1..].iter().filter(|x| x[2] == v).map(|x| x[col].parse().unwrap()).collect();
            for (h, w) in want.iter().enumerate() {
                let w = w.as_f64().unwrap();
                assert!(
                    (got[h] - w).abs() <= 1e-10 * (1.0 + w.abs()),
                    "{field}.{v}[{h}]: rust {} vs dynare {w}",
                    got[h]
                );
                n += 1;
            }
        }
    }
    assert_eq!(n, 3 * 4 * 12);
}

#[test]
fn ramsey_constraints_in_perfect_foresight() {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    let dir = setup("numfix_ramsey_constraints", &["ramsey_constraints.mod"]);
    run(&dir.join("ramsey_constraints.mod"), &quiet()).unwrap();
    let rows = csv(&dir.join("ramsey_constraints/Output/ramsey_constraints_endo_simul.csv"));
    for v in ["i", "y", "pi", "MULT_1", "MULT_2"] {
        let j = rows[0].iter().position(|s| s == v).unwrap();
        for (t, w) in r["ramsey_constraints_endo_simul"][v].as_array().unwrap().iter().enumerate() {
            let (a, w): (f64, f64) = (rows[t + 1][j].parse().unwrap(), w.as_f64().unwrap());
            assert!((a - w).abs() <= 1e-8 * (1.0 + w.abs()), "{v}[{t}]: rust {a} vs dynare {w}");
        }
    }
    // i > 0 binds: without the constraint, i(1) is about -4.3
    let j = rows[0].iter().position(|s| s == "i").unwrap();
    assert!(rows[1..].iter().all(|x| x[j].parse::<f64>().unwrap() > -1e-8));
}

#[test]
fn filter_initial_state_with_estimated_parameters() {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    let dir = setup(
        "numfix_kalman_initial",
        &["kalman_initial.mod", "kalman_initial_2_lag.mod", "fs2000_common.inc", "fsdat_simul.m"],
    );
    for m in ["kalman_initial", "kalman_initial_2_lag"] {
        run(&dir.join(format!("{m}.mod")), &quiet()).unwrap();
        // mode_compute = 0: the log posterior at the initial point (Dynare prints 8 digits)
        let f =
            results(&dir.join(format!("{m}/Output/{m}_results.json")))["minus_log_posterior_at_mode"].as_f64().unwrap();
        let want = r["kalman_initial_log_posterior"][m].as_f64().unwrap();
        assert!((-f - want).abs() < 1e-3, "{m}: rust {} vs dynare {want}", -f);
    }
}

#[test]
fn slice_sampler_with_smoother_runs_the_mcmc() {
    let dir = setup("numfix_slice", &["slice_init_states.mod", "fs2000_slice.inc", "fsdat_simul.m"]);
    run(&dir.join("slice_init_states.mod"), &RunOptions { quiet: true, seed: 1, ..Default::default() }).unwrap();
    let res = results(&dir.join("slice_init_states/Output/slice_init_states_results.json"));
    assert!(res["mcmc"].is_object(), "no MCMC run");
    // marginal_density.m: oo_.posterior_mode is the best draw, with NaN standard deviations
    let n = res["parameter_names"].as_array().unwrap().len();
    assert_eq!(res["mode_sd"].as_array().unwrap().len(), n);
    assert!(res["mode_sd"].as_array().unwrap().iter().all(Value::is_null));
}

/// reference/nonlinear/rbcnl.mod with its estimation commands replaced by `est`.
fn rbcnl(work: &Path, est: &str) {
    let nl = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/nonlinear");
    let src = std::fs::read_to_string(nl.join("rbcnl.mod")).unwrap();
    let cut = src.find("@#if defined(ESTGF)").unwrap();
    std::fs::write(work.join("rbcnl.mod"), format!("{}{est}\n@#endif\n", &src[..cut])).unwrap();
    std::fs::copy(nl.join("rbcnl_data.csv"), work.join("rbcnl_data.csv")).unwrap();
}

#[test]
fn cova_compute_zero_gives_nan_standard_deviations() {
    for (tag, order, filter) in [
        ("numfix_cova_o1", 1, ""),
        (
            "numfix_cova_o2",
            2,
            ", filter_algorithm=gf, proposal_approximation=cubature, distribution_approximation=cubature",
        ),
    ] {
        let dir = common::temp_dir(tag);
        rbcnl(
            &dir,
            &format!(
                "estimation(datafile='rbcnl_data.csv', order={order}{filter}, mode_compute=0, cova_compute=0, mh_replic=0, nograph);"
            ),
        );
        run(&dir.join("rbcnl.mod"), &quiet()).unwrap();
        let res = results(&dir.join("rbcnl/Output/rbcnl_results.json"));
        let sd = res["mode_sd"].as_array().unwrap();
        assert!(!sd.is_empty() && sd.iter().all(Value::is_null), "order {order}: {sd:?}");
        assert!(res["log_data_density_laplace"].is_null(), "order {order}");
    }
}
