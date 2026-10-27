//! Bad .mod input gives a clean error (no panic, no stack overflow, no silently ignored
//! statement): parser, macro processor and option-value checks.

use dynare::driver::{RunOptions, run};
use dynare::modfile::{Command, OPTIONS_ASSIGNMENT, OptValue, opt, parse_mod_file, parse_mod_str};
use std::path::PathBuf;

const AR1: &str = "var y; varexo e; parameters r; r=0.5; model; y = r*y(-1)+e; end;\n\
                   shocks; var e; stderr 1; end; steady_state_model; y=0; end;\n";

fn err(src: &str) -> String {
    match parse_mod_str(src, &[]) {
        Ok(_) => panic!("expected an error for:\n{src}"),
        Err(e) => format!("{e:#}"),
    }
}

fn assert_err(src: &str, needle: &str) {
    let e = err(src);
    assert!(e.contains(needle), "error `{e}` does not contain `{needle}`");
}

/// A fresh directory holding `name.mod` (and optionally other files).
fn write_mod(tag: &str, name: &str, src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("dynare_rs_robust_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!("{name}.mod"));
    std::fs::write(&p, src).unwrap();
    p
}

#[test]
fn options_assignment_is_not_swallowed() {
    let mf = parse_mod_str(&format!("{AR1}options_.solve_tolf = 1e-3;\nsteady;\n"), &[]).unwrap();
    let assigned = mf.commands.iter().any(|c| {
        matches!(c, Command::Other(n, o) if n == OPTIONS_ASSIGNMENT
            && o.iter().any(|(k, v)| k == "solve_tolf" && *v == OptValue::Num(1e-3)))
    });
    assert!(assigned, "options_.solve_tolf assignment dropped");
    let steady = mf.commands.iter().find_map(|c| if let Command::Steady(o) = c { Some(o) } else { None }).unwrap();
    assert_eq!(opt(steady, "solve_tolf"), Some(&OptValue::Num(1e-3)));
}

#[test]
fn shocks_overwrite_resets_correlations() {
    let src = "var y; varexo e u; model; y = e + u; end;\n\
               shocks; var e; stderr 1; var u; stderr 1; corr e, u = 0.5; end;\n\
               shocks(overwrite); var e; stderr 1; var u; stderr 2; end;\n";
    let mf = parse_mod_str(src, &[]).unwrap();
    assert_eq!(mf.corr_e[(0, 1)], 0.0);
    assert_eq!(mf.corr_e[(1, 0)], 0.0);
    assert_eq!(mf.sigma_e[(0, 1)], 0.0);
}

#[test]
fn steady_state_model_has_priority_over_steadystate_file() {
    let p = write_mod("ssfile", "ssm", &format!("{AR1}steady;\n"));
    std::fs::write(
        p.with_file_name("ssm_steadystate.m"),
        "function [ys,params,check]=ssm_steadystate(ys,exo,M_,options_)\n",
    )
    .unwrap();
    parse_mod_file(&p, &[]).unwrap();
    // without the block the MATLAB file is still refused
    let q = write_mod("ssfile2", "nossm", "var y; varexo e; model; y = e; end;\n");
    std::fs::write(q.with_file_name("nossm_steadystate.m"), "function x\n").unwrap();
    assert!(parse_mod_file(&q, &[]).is_err());
}

#[test]
fn long_sum_does_not_overflow_the_stack() {
    let terms = vec!["0.0001*e"; 20000].join("+");
    let p = write_mod("longsum", "longsum", &format!("var y; varexo e; model; y = {terms}; end;\nsteady;\n"));
    run(&p, &RunOptions::default()).unwrap();
}

#[test]
fn deep_nesting_is_an_error() {
    let n = 20000;
    let src = format!("var y; varexo e; model; y = {}e{}; end;\n", "(".repeat(n), ")".repeat(n));
    let p = write_mod("deepnest", "deepnest", &src);
    let e = format!("{:#}", run(&p, &RunOptions::default()).unwrap_err());
    assert!(e.contains("nested too deeply"), "{e}");
}

#[test]
fn macro_errors() {
    assert_err("@#define f(a) = f(a)\n@#define y = f(1)\nvar a; varexo e; model; a = e; end;\n", "nested calls");
    // 1/0 is inf in the macro language (the preprocessor does not reject it); a loop over an
    // infinite range never ends in the preprocessor, here it is an error
    assert_err(
        "@#define x = 1e400\n@#for i in 1:x\nvar a@{i};\n@#endfor\nvar a; varexo e; model; a = e; end;\n",
        "non-finite",
    );
    assert_err("var y; varexo e;\n/* unterminated\nmodel; y = e; end;\n", ":2: unterminated comment");
}

#[test]
fn invalid_option_values() {
    let est = |o: &str| format!("{AR1}varobs y; estimated_params; r, 0.5, 0, 1; end;\nestimation(datafile=d, {o});\n");
    for (o, key) in [
        ("lik_init=7", "lik_init"),
        ("lik_init=0", "lik_init"),
        ("mh_nblocks=0", "mh_nblocks"),
        ("mh_drop=1", "mh_drop"),
        ("mh_drop=-0.5", "mh_drop"),
        ("mh_conf_sig=2", "mh_conf_sig"),
        ("mh_conf_sig=0", "mh_conf_sig"),
        ("kalman_algo=6", "kalman_algo"),
        ("order=0", "order"),
        ("forecast=1e20", "forecast"),
    ] {
        assert_err(&est(o), &format!("invalid value for option {key}"));
    }
    for (o, key) in [
        ("irf=-5", "irf"),
        ("irf=1e20", "irf"),
        ("irf=2.5", "irf"),
        ("order=0", "order"),
        ("periods=1e15", "periods"),
        ("ar='x'", "ar"),
    ] {
        assert_err(&format!("{AR1}stoch_simul({o});\n"), &format!("invalid value for option {key}"));
    }
    // a global options_ value is checked at the next command
    assert_err(&format!("{AR1}options_.lik_init = 9;\nstoch_simul(order=1);\n"), "invalid value for option lik_init");
}

#[test]
fn invalid_known_option_type_at_run_time() {
    let p = write_mod("hpfilter", "hpf", &format!("{AR1}stoch_simul(order=1, irf=0, hp_filter='abc');\n"));
    let e = format!("{:#}", run(&p, &RunOptions::default()).unwrap_err());
    assert!(e.contains("invalid value for option hp_filter"), "{e}");
}

#[test]
fn self_correlations_are_errors() {
    assert_err("var y; varexo e; model; y = e; end; shocks; var e; stderr 1; var e, e = 2; end;\n", "with itself");
    assert_err("var y; varexo e; model; y = e; end; shocks; var e; stderr 1; corr e, e = 0.5; end;\n", "with itself");
    assert_err(
        "var y; varexo e u; model; y = e + u; end; varobs y; estimated_params; corr e, e, 0.1, -1, 1; end;\n",
        "with itself",
    );
}

/// Dynare 7.2 preprocessor texts: duplicate `histval` entries and
/// `+=`/`*=` lines in an `endval` block without `learnt_in` > 1 or a date.
#[test]
fn histval_twice_and_endval_learnt_only_lines() {
    let m = "var y; varexo e; parameters a; a=0.5; model; y = a*y(-1)+e; end;\n";
    assert_err(&format!("{m}histval; y(0)=1; y(0)=2; end;\n"), "histval: y(0) declared twice");
    assert_err(&format!("{m}histval; y(-1)=1; y(-1)=2; end;\n"), "histval: y(-1) declared twice");
    parse_mod_str(&format!("{m}histval; y(0)=1; end;\nhistval; y(0)=2; end;\n"), &[]).unwrap();
    for op in ["+", "*"] {
        assert_err(
            &format!("{m}endval; e {op}= 1; end;\n"),
            &format!(
                "endval: 'e {op}= ...' line not allowed unless 'learnt_in' option with value >1 or date is passed"
            ),
        );
    }
    parse_mod_str(&format!("{m}endval(learnt_in=2); e += 1; end;\n"), &[]).unwrap();
}
