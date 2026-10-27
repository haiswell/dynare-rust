//! Perfect-foresight behaviour of Dynare 7.2 that the reference models do not exercise:
//!   - the generated driver keeps `oo_.initial_steady_state` from the first `endval` block;
//!   - `histval` fills a series over `M_.orig_maximum_lag_with_diffs_expanded` periods and
//!     computes the auxiliary variables with `<fname>.dynamic_set_auxiliary_series` (the
//!     expected values below follow the generated driver.m and that function);
//!   - the errors of `perfect_foresight_solver.m` (`check_input_arguments`, homotopy options),
//!     of `perfect_foresight_solver_core.m` (controlled paths, `linear_approximation`,
//!     `endogenous_terminal_period`) and of `sim1_linear.m`;
//!   - `get_simulation_periods.m` leaves `options_.periods` untouched;
//!   - `histvalf_initvalf.m` marks `first_obs` as given by `first_simulation_period`;
//!   - `perfect_foresight_with_expectation_errors_solver.m`'s option check;
//!   - controlled paths with `stack_solve_algo = 7` (`solve_stacked_problem.m`), against the
//!     Dynare reference of the same problem solved by `sim1`.
mod common;

use std::path::{Path, PathBuf};

use dynare::driver::{RunOptions, run};

/// Writes `body` to `<tag>.mod` in a fresh directory (plus `files`) and runs it.
fn run_mod(tag: &str, body: &str, files: &[(&str, &str)]) -> (PathBuf, anyhow::Result<()>) {
    let dir = common::temp_dir(tag);
    for (name, text) in files {
        std::fs::write(dir.join(name), text).unwrap();
    }
    let p = dir.join(format!("{tag}.mod"));
    std::fs::write(&p, body).unwrap();
    let r = run(&p, &RunOptions { quiet: true, ..Default::default() });
    (dir, r)
}

/// `oo_.endo_simul` as written by `perfect_foresight_solver`: (variable names, rows by period).
fn endo_simul(dir: &Path, tag: &str) -> (Vec<String>, Vec<Vec<f64>>) {
    let p = dir.join(format!("{tag}/Output/{tag}_endo_simul.csv"));
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let mut lines = s.lines();
    let names = lines.next().unwrap().split(',').skip(1).map(str::to_string).collect();
    let rows = lines.map(|l| l.split(',').skip(1).map(|v| v.parse().unwrap()).collect()).collect();
    (names, rows)
}

/// Value of `var` in `period` (0 = initial condition, 1 = first simulated period).
fn at(sim: &(Vec<String>, Vec<Vec<f64>>), var: &str, period: usize) -> f64 {
    let j = sim.0.iter().position(|n| n == var).unwrap_or_else(|| panic!("no variable {var}"));
    sim.1[period][j]
}

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-10 * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

fn expect_err(r: anyhow::Result<()>, msg: &str) {
    let e = format!("{:#}", r.expect_err(&format!("expected the error '{msg}'")));
    assert!(e.contains(msg), "expected '{msg}', got '{e}'");
}

#[test]
fn second_endval_keeps_the_first_initial_steady_state() {
    // y = 0.5 y(-1) + e: steady state 2e. The initial condition is the steady state at e = 1
    // (the initval block), not the one of the first endval block.
    let body = "var y; varexo e;\n\
                model; y = 0.5*y(-1) + e; end;\n\
                initval; e = 1; end; steady(noprint);\n\
                endval; e = 2; end; steady(noprint);\n\
                endval; e = 3; end; steady(noprint);\n\
                perfect_foresight_setup(periods=3);\n\
                perfect_foresight_solver(noprint);\n";
    let (dir, r) = run_mod("pde_endval2", body, &[]);
    r.unwrap();
    let s = endo_simul(&dir, "pde_endval2");
    close(at(&s, "y", 0), 2.0, "y(0)");
    close(at(&s, "y", 1), 0.5 * 2.0 + 3.0, "y(1)");
    close(at(&s, "y", 2), 0.5 * 4.0 + 3.0, "y(2)");
}

#[test]
fn histval_auxiliary_series_over_lags_with_diffs_expanded() {
    // M_.orig_maximum_lag = 1 but orig_maximum_lag_with_diffs_expanded = 2: the series covers
    // dates -1..0, so AUX_ENDO_LAG_0_1(0) = y(-1) = 0.5 and AUX_DIFF_11(0) = y(-1) - y(-2) is
    // NaN (y(-2) is outside the series), set to 0.
    let body = "var y x; varexo e;\n\
                model; y = 0.9*y(-1) + e; x = 0.5*diff(y(-1)) + 0.1*x(+1); end;\n\
                histval; y(0) = 1; y(-1) = 0.5; end;\n\
                perfect_foresight_setup(periods=3);\n\
                perfect_foresight_solver(noprint);\n";
    let (dir, r) = run_mod("pde_histval_diff", body, &[]);
    r.unwrap();
    let s = endo_simul(&dir, "pde_histval_diff");
    close(at(&s, "AUX_ENDO_LAG_0_1", 0), 0.5, "AUX_ENDO_LAG_0_1(0)");
    close(at(&s, "AUX_DIFF_11", 0), 0.0, "AUX_DIFF_11(0)");
    // diff(y(-1)) at t = y(t-1) - y(t-2): 1 - 0.5, 0.9 - 1, 0.81 - 0.9
    let d = [0.5, -0.1, 0.81 - 0.9];
    for (t, dv) in d.iter().enumerate() {
        close(at(&s, "AUX_DIFF_11", t + 1), *dv, "AUX_DIFF_11");
    }
    // x(t) = 0.5 d(t) + 0.1 x(t+1), x(4) = 0
    let x3 = 0.5 * d[2];
    let x2 = 0.5 * d[1] + 0.1 * x3;
    let x1 = 0.5 * d[0] + 0.1 * x2;
    close(at(&s, "x", 1), x1, "x(1)");
    close(at(&s, "x", 3), x3, "x(3)");
}

#[test]
fn histval_auxiliary_series_deep_lags() {
    // The model of the generated df/dynamic_set_auxiliary_series.m:
    //   ds.AUX_DIFF_11=ds.y(-1)-ds.y(-2); ds.AUX_ENDO_LAG_0_1=ds.y(-1);
    //   ds.AUX_ENDO_LAG_0_2=ds.AUX_ENDO_LAG_0_1(-1); ds.AUX_EXO_LAG_3_0=ds.e;
    //   ds.AUX_EXO_LAG_3_1=ds.AUX_EXO_LAG_3_0(-1);
    // over the dates -2..0 (orig_maximum_lag_with_diffs_expanded = 3).
    let body = "var y x z; varexo e;\n\
                model;\n\
                y = 0.9*y(-1) + e;\n\
                x = 0.5*diff(y(-1)) + 0.1*x(+1);\n\
                z = 0.2*z(-1) + 0.3*y(-3) + e(-2);\n\
                end;\n\
                histval; y(0)=1; y(-1)=0.5; y(-2)=0.25; z(0)=2; e(0)=0.7; e(-1)=0.3; end;\n\
                perfect_foresight_setup(periods=4);\n\
                perfect_foresight_solver(noprint);\n";
    let (dir, r) = run_mod("pde_histval_deep", body, &[]);
    r.unwrap();
    let s = endo_simul(&dir, "pde_histval_deep");
    close(at(&s, "AUX_DIFF_11", 0), 0.5 - 0.25, "AUX_DIFF_11(0)");
    close(at(&s, "AUX_ENDO_LAG_0_1", 0), 0.5, "AUX_ENDO_LAG_0_1(0)");
    close(at(&s, "AUX_ENDO_LAG_0_2", 0), 0.25, "AUX_ENDO_LAG_0_2(0)");
    close(at(&s, "AUX_EXO_LAG_3_0", 0), 0.7, "AUX_EXO_LAG_3_0(0)");
    close(at(&s, "AUX_EXO_LAG_3_1", 0), 0.3, "AUX_EXO_LAG_3_1(0)");
    // z(1) = 0.2 z(0) + 0.3 y(-2) + e(-1)
    close(at(&s, "z", 1), 0.2 * 2.0 + 0.3 * 0.25 + 0.3, "z(1)");
}

#[test]
fn simulation_dates_do_not_set_options_periods() {
    // get_simulation_periods.m returns the horizon without storing it in options_.periods,
    // so new dates give a new horizon.
    let body = "var y; varexo e;\n\
                model; y = 0.5*y(-1) + 0.2*y(+1) + e; end;\n\
                perfect_foresight_setup(first_simulation_period=2000Q1, last_simulation_period=2000Q4);\n\
                perfect_foresight_setup(first_simulation_period=2000Q1, last_simulation_period=2001Q2);\n\
                perfect_foresight_solver(noprint);\n";
    let (dir, r) = run_mod("pde_dates", body, &[]);
    r.unwrap();
    let s = endo_simul(&dir, "pde_dates");
    assert_eq!(s.1.len(), 1 + 6 + 1, "maximum_lag + 6 periods + maximum_lead");
}

const LINEAR: &str = "var y; varexo e;\nmodel; y = 0.5*y(-1) + 0.2*y(+1) + e; end;\n";

#[test]
fn invalid_stack_solve_algo_is_an_error() {
    let (_, r) = run_mod(
        "pde_ssa4",
        &format!("{LINEAR}perfect_foresight_setup(periods=5);\nperfect_foresight_solver(stack_solve_algo=4);\n"),
        &[],
    );
    expect_err(
        r,
        "PERFECT_FORESIGHT_SOLVER: you must use stack_solve_algo={0,1,2,3,6,7,8,9,10} when not using block nor bytecode option",
    );
    let (_, r) = run_mod(
        "pde_ssa11",
        &format!("{LINEAR}perfect_foresight_setup(periods=5);\nperfect_foresight_solver(stack_solve_algo=11);\n"),
        &[],
    );
    expect_err(r, "PERFECT_FORESIGHT_SOLVER: stack_solve_algo must be between 0 and 10");
}

#[test]
fn homotopy_option_checks() {
    let cases = [
        (
            "homotopy_initial_step_size=0.0005",
            "The value given to homotopy_initial_step_size option must be greater or equal to that given to homotopy_min_step_size option",
        ),
        ("homotopy_min_step_size=2", "The value given to homotopy_min_step_size option must be in the [0,1] interval"),
        (
            "homotopy_max_completion_share=-1, homotopy_linearization_fallback",
            "The value given to homotopy_max_completion_share option must be in the [0,1] interval",
        ),
    ];
    for (k, (opts, msg)) in cases.iter().enumerate() {
        let body = format!("{LINEAR}perfect_foresight_setup(periods=5);\nperfect_foresight_solver({opts});\n");
        let (_, r) = run_mod(&format!("pde_homotopy{k}"), &body, &[]);
        expect_err(r, msg);
    }
}

#[test]
fn core_option_checks() {
    // perfect_foresight_solver_core.m
    let (_, r) = run_mod(
        "pde_linapprox_lbj",
        &format!(
            "{LINEAR}perfect_foresight_setup(periods=5);\nperfect_foresight_solver(linear_approximation, stack_solve_algo=1);\n"
        ),
        &[],
    );
    expect_err(r, "option linear_approximation is only available with stack_solve_algo option equal to 0 or 7");
    let (_, r) = run_mod(
        "pde_linapprox_backward",
        "var y; varexo e;\nmodel; y = 0.5*y(-1) + e; end;\nperfect_foresight_setup(periods=5);\nperfect_foresight_solver(linear_approximation);\n",
        &[],
    );
    expect_err(r, "option linear_approximation is not available with purely backward, purely forward or static models");
    let (_, r) = run_mod(
        "pde_etp_ssa",
        &format!(
            "{LINEAR}perfect_foresight_setup(periods=5);\nperfect_foresight_solver(endogenous_terminal_period, stack_solve_algo=2);\n"
        ),
        &[],
    );
    expect_err(r, "option endogenous_terminal_period is only available with stack_solve_algo option equal to 0");
}

#[test]
fn sim1_linear_off_steady_state_is_an_error() {
    // model(linear) with an initval that is not a steady state (residual 0.3 > solve_tolf)
    let body = "var y; varexo e;\nmodel(linear); y = 0.5*y(-1) + 0.2*y(+1) + e; end;\n\
                initval; y = 1; end;\n\
                perfect_foresight_setup(periods=5);\nperfect_foresight_solver;\n";
    let (_, r) = run_mod("pde_linear_ss", body, &[]);
    expect_err(r, "Jacobian is not evaluated at the steady state!");
}

const CONTROLLED_BACKWARD: &str = "var y; varexo e;\nmodel; y = 0.5*y(-1) + e; end;\n\
    perfect_foresight_controlled_paths;\n  exogenize y;\n  periods 1;\n  values 2;\n  endogenize e;\nend;\n\
    perfect_foresight_setup(periods=3);\nperfect_foresight_solver;\n";

#[test]
fn controlled_paths_checks() {
    let (_, r) = run_mod("pde_cp_backward", CONTROLLED_BACKWARD, &[]);
    expect_err(
        r,
        "perfect_foresight_controlled_paths or shock_paths+exogenize is not available with purely backward, purely forward or static models",
    );
    let refmod = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/ep/controlled.mod"))
        .unwrap()
        .replacen("perfect_foresight_solver;", "perfect_foresight_solver(lmmcp);", 1);
    let (_, r) = run_mod("pde_cp_lmmcp", &refmod, &[]);
    expect_err(
        r,
        "perfect_foresight_controlled_paths or shock_paths+exogenize is not available for mixed-complementarity problems (LMMCP or PATH solvers)",
    );
}

#[test]
fn controlled_paths_with_stack_solve_algo_7() {
    // solve_stacked_problem.m (controlled_paths_wrapper + dynare_solve) finds the solution of
    // the same problem that sim1 solves (reference/ep/ref_controlled_newton.json). Dynare's
    // reference stops at its default tolf = 1e-5 (the residual of its last Newton step is not
    // zero), so it is only reproduced to that accuracy; the two solvers run to tolf = 1e-12
    // agree to rounding.
    let refdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/ep");
    let src = std::fs::read_to_string(refdir.join("controlled.mod")).unwrap();
    let solve = |name: &str, opts: &str| {
        let body = src.replacen("perfect_foresight_solver;", &format!("perfect_foresight_solver({opts});"), 1);
        let (dir, r) = run_mod(name, &body, &[]);
        r.unwrap();
        endo_simul(&dir, name)
    };
    let s7 = solve("pde_cp_ssa7", "stack_solve_algo=7, tolf=1e-12");
    let s0 = solve("pde_cp_ssa0", "tolf=1e-12");
    let want: serde_json::Value = common::json_ref(refdir.join("ref_controlled_newton.json"));
    let (mut err_ref, mut err_newton) = (0.0f64, 0.0f64);
    for (j, row) in want["endo_simul"].as_array().unwrap().iter().enumerate() {
        for (t, w) in row.as_array().unwrap().iter().enumerate() {
            err_ref = err_ref.max((s7.1[t][j] - w.as_f64().unwrap()).abs());
            err_newton = err_newton.max((s7.1[t][j] - s0.1[t][j]).abs());
        }
    }
    assert!(err_newton < 1e-9, "stack_solve_algo 7 vs 0: {err_newton}");
    assert!(err_ref < 1e-5, "endo_simul vs Dynare: {err_ref}");
}

#[test]
fn histval_file_first_simulation_period_counts_as_first_obs() {
    // first_obs = first_simulation_period - M_.orig_maximum_lag = 2; with last_obs = 4 the
    // HISTVAL consistency check fails (last_obs must be first_obs + orig_maximum_lag - 1)
    let csv = ",y,e\n1,0.1,0\n2,0.2,0\n3,0.3,0\n4,0.4,0\n5,0.5,0\n";
    let body = "var y; varexo e;\nmodel; y = 0.5*y(-1) + e; end;\n\
                histval_file(datafile='h.csv', first_simulation_period=3, last_obs=4);\n";
    let (_, r) = run_mod("pde_hvf", body, &[("h.csv", csv)]);
    expect_err(
        r,
        "HISTVAL_FILE: the number of observations given by FIRST_OBS and LAST_OBS differs from the maximum number of lags in the model (1).",
    );
}

#[test]
fn pfwee_constant_simulation_length_with_last_simulation_period() {
    let body = format!(
        "{LINEAR}shocks(learnt_in=2); var e; periods 3; values 1; end;\n\
         perfect_foresight_with_expectation_errors_setup(first_simulation_period=2000Q1, last_simulation_period=2001Q2);\n\
         perfect_foresight_with_expectation_errors_solver(constant_simulation_length);\n"
    );
    let (_, r) = run_mod("pde_pfwee_csl", &body, &[]);
    expect_err(r, "Options constant_simulation_length and last_simulation_period cannot be used together");
}

#[test]
fn controlled_paths_with_stack_solve_algo_1_and_6() {
    // perfect_foresight_solver_core.m dispatches controlled paths to sim1_lbj.m for
    // stack_solve_algo 1 and 6 (the Newton iterations of the same problem, stopped on the step
    // size dynatol.x): the solution is that of sim1 (ssa 0) to the iteration accuracy.
    let refdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/ep");
    let src = std::fs::read_to_string(refdir.join("controlled.mod")).unwrap();
    let solve = |name: &str, opts: &str| {
        let body = src.replacen("perfect_foresight_solver;", &format!("perfect_foresight_solver({opts});"), 1);
        let (dir, r) = run_mod(name, &body, &[]);
        r.unwrap();
        endo_simul(&dir, name)
    };
    let s0 = solve("pde_cp_lbj0", "tolf=1e-12, tolx=1e-12");
    for ssa in [1, 6] {
        let s = solve(&format!("pde_cp_lbj{ssa}"), &format!("stack_solve_algo={ssa}, tolx=1e-12"));
        let mut d = 0.0f64;
        for (t, row) in s.1.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                d = d.max((v - s0.1[t][j]).abs());
            }
        }
        println!("stack_solve_algo {ssa} vs 0: {d:.2e}");
        assert!(d < 1e-8, "stack_solve_algo {ssa}: {d}");
        // the exogenized consumption is held at the controlled values (1.6 in period 2)
        close(at(&s, "c", 2), 1.6, "controlled c");
    }
}

#[test]
fn pf_solve_algo_is_the_global_options_solve_algo() {
    // The preprocessor writes `options_.solve_algo = X` for perfect_foresight_solver(solve_algo=X)
    // and for steady(solve_algo=X): one global, so each command's value reaches the next.
    use dynare::deterministic::DetState;
    use dynare::modfile::{OptValue, parse_mod_file};
    let dir = common::temp_dir("pde_ssa_global");
    let p = dir.join("m.mod");
    std::fs::write(&p, "var y; varexo e;\nmodel; y = 0.5*y(-1) + e; end;\ninitval; y=0; e=0; end;\n").unwrap();
    let mf = parse_mod_file(&p, &[]).unwrap();
    let mut det = DetState::new(&mf).unwrap();
    det.update_options(&vec![("solve_algo".to_string(), OptValue::Num(3.0))]);
    assert_eq!(det.steady_opts.solve_algo, 3);
    det.update_options(&vec![("solve_algo".to_string(), OptValue::Num(2.0))]);
    assert_eq!(det.steady_opts.solve_algo, 2);
}
