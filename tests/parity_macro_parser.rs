//! Parity fixes in the macro processor, expected values from `dynare-preprocessor savemacro
//! onlymacro` (Dynare 7.2).
mod common;

use dynare::driver::RunOptions;
use dynare::misc::cli::{CliOptions, before_parse};

fn expand(tag: &str, src: &str) -> String {
    let dir = common::temp_dir(tag);
    let p = dir.join(format!("{tag}.mod"));
    std::fs::write(&p, src).unwrap();
    let ro = RunOptions {
        cli: CliOptions { savemacro: Some(None), onlymacro: true, ..Default::default() },
        quiet: true,
        ..Default::default()
    };
    assert!(before_parse(&p, &ro).unwrap());
    std::fs::read_to_string(dir.join(format!("{tag}_macroexp.mod"))).unwrap()
}

/// `@#for ... when`: the array is filtered first with the loop variable bound to every element,
/// so afterwards the variable is the last element kept, or the last element when none is kept.
#[test]
fn for_when_loop_variable_after_the_loop() {
    let out = expand(
        "pmp_when",
        "@#define i = 10\n@#for i in 1:3 when i < 2\nz@{i};\n@#endfor\nlast@{i};\n\
         @#for j in 1:3 when j > 5\nw@{j};\n@#endfor\nlastj@{j};\n",
    );
    assert_eq!(out, "z1;\nlast1;\nlastj3;\n");
}

/// The preprocessor's output has no empty lines (inside @#for bodies or elsewhere);
/// whitespace-only lines stay.
#[test]
fn savemacro_drops_empty_lines() {
    let out = expand("pmp_blank", "@#for k in 1:2\nx@{k};\n\ny@{k};\n\n@#endfor\nend;\n");
    assert_eq!(out, "x1;\ny1;\nx2;\ny2;\nend;\n");
    let out = expand("pmp_blank2", "a;\n\n\nb;\n  \nc;\n@#define x=1\n\nd;\n// c\n\ne;\n");
    assert_eq!(out, "a;\nb;\n  \nc;\nd;\n// c\ne;\n");
}

// ---------------------------------------------------------------------------------------------
// Model-modification statements and the shocks block. Expected values are the output of
// dynare-preprocessor.exe 7.2 on the same files (driver.m: M_.endo_names, M_.exo_names,
// `M_.histval_dseries{'y'}(dates('0Y'))=1`).
// ---------------------------------------------------------------------------------------------

use dynare::modfile::{ParseOptions, parse_mod_file_opts};

fn parse_text(tag: &str, text: &str, o: &ParseOptions) -> anyhow::Result<dynare::modfile::ModFile> {
    let dir = common::temp_dir(tag);
    let p = dir.join(format!("{tag}.mod"));
    std::fs::write(&p, text).unwrap();
    parse_mod_file_opts(&p, o)
}

/// macro-parser-2: a histval block written before `model_remove` keeps its variable: after the
/// removal of k's equation y is the only endogenous variable (index 0), and the preprocessor
/// writes `M_.histval_dseries{'y'}(dates('0Y'))=1` (it used to be re-indexed as an exogenous
/// variable).
#[test]
fn histval_survives_model_remove() {
    let src = "var k y; varexo ek ey;
model;
[name='ek'] k = 0.9*k(-1) + ek;
[name='ey'] y = 0.5*y(-1) + ey + 0*ek;
end;
histval; y(0) = 1; end;
model_remove('ek');
";
    let o = ParseOptions { nostrict: true, ..Default::default() };
    let mf = parse_text("mp_histrm", src, &o).unwrap();
    assert_eq!(mf.model.endo_names, ["y"]);
    assert_eq!(mf.histval, vec![(true, 0, 0, 1.0)]);
}

/// macro-parser-9: `stderr u = 2;` in a shocks block is a syntax error (the preprocessor:
/// "syntax error, unexpected STDERR, expecting CORR or SKEW or VAR"), not a skipped statement.
#[test]
fn invalid_shocks_entry_is_a_syntax_error() {
    let src = "var y; varexo u;
model; y = 0.5*y(-1) + u; end;
shocks; stderr u = 2; end;
";
    let e = parse_text("mp_shk", src, &ParseOptions::default()).expect_err("syntax error");
    println!("{e:#}");
    assert!(format!("{e:#}").to_lowercase().contains("syntax error") || format!("{e:#}").contains("stderr"), "{e:#}");
    // the valid forms parse
    let ok = "var y; varexo u;
model; y = 0.5*y(-1) + u; end;
shocks; var u; stderr 2; end;
";
    let mf = parse_text("mp_shk_ok", ok, &ParseOptions::default()).unwrap();
    assert_eq!(mf.sigma_e[(0, 0)], 4.0);
}

/// macro-parser-10: include_eqs removes an equation without tags too, finding its variable from
/// the left-hand side. Preprocessor with `include_eqs=[ey]`: exo_names k ek ey, endo_names
/// y AUX_EXO_LAG_0_0 (k became exogenous).
#[test]
fn include_eqs_removes_untagged_equations() {
    let src = "var k y; varexo ek ey;
model;
k = 0.9*k(-1) + ek;
[name='ey'] y = 0.5*y(-1) + ey + 0.1*k(-1);
end;
";
    let o = ParseOptions { include_eqs: Some("[ey]".into()), ..Default::default() };
    let mf = parse_text("mp_inc", src, &o).unwrap();
    assert_eq!(mf.model.exo_names, ["k", "ek", "ey"]);
    assert_eq!(mf.model.endo_names, ["y", "AUX_EXO_LAG_0_0"]);
}
