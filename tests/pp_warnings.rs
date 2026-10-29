//! The warnings of Dynare 7.2's preprocessor (`WARNING: ...` lines, same text and order).
//! `tests/pp_warnings/*.expected` hold the output of the real preprocessor on the `.mod` files.

use dynare::modfile::parse_mod_file;
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/pp_warnings")
}

fn check(name: &str) {
    let mf = parse_mod_file(&dir().join(format!("{name}.mod")), &[]).unwrap();
    let expected = std::fs::read_to_string(dir().join(format!("{name}.expected"))).unwrap();
    let expected: Vec<&str> = expected.lines().collect();
    assert_eq!(mf.warnings, expected, "{name}");
}

#[test]
fn unused_parameters() {
    check("unused");
    check("all_used");
}

#[test]
fn deprecated_syntax() {
    check("deprecated");
}

#[test]
fn stochastic_context_and_osr_params() {
    check("stochastic_ctx");
}

#[test]
fn ramsey_policy_and_discount_factor() {
    check("ramsey_policy");
}

/// A failing parse still reports the warnings the preprocessor printed before its error.
fn check_before_error(name: &str) {
    assert!(parse_mod_file(&dir().join(format!("{name}.mod")), &[]).is_err(), "{name}");
    let expected = std::fs::read_to_string(dir().join(format!("{name}.expected"))).unwrap();
    let expected: Vec<&str> = expected.lines().collect();
    assert_eq!(dynare::modfile::take_error_warnings(), expected, "{name}");
}

#[test]
fn warnings_before_errors() {
    check_before_error("err_syntax"); // parser warning, then a syntax error
    check_before_error("err_bg"); // unused parameters, then the balanced-growth failure
}
