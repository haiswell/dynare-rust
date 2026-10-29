//! Checks of the Dynare 7.2 preprocessor that make its `*_xfail` tests fail (parser actions and
//! `checkPass`): the error texts are those printed by `dynare-preprocessor` 7.2 on the same
//! statements (tests/ep/hybrid/burnside_xfail, estimation/estim_param_in_shock_value_xfail,
//! estimation/tune_mh_jscale/fs2000_{1,2}_xfail, occbin/.../dynrbc_token_xfail,
//! kalman_initial_state/fs2000_kalman_initial_xfail, preprocessor_checks/example1_extra_exo_xfail).
use dynare::modfile::{ParseOptions, parse_mod_file, parse_mod_file_opts};

const HEAD: &str = "var y c; varexo e u; parameters a b; a=0.5; b=0.1;\n\
                    model; y = a*y(-1) + e; c = b*c(+1) + y + u; end;\n";

fn write(tag: &str, src: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("ppc72_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(format!("{tag}.mod"));
    std::fs::write(&p, src).unwrap();
    p
}

fn err(tag: &str, tail: &str) -> String {
    match parse_mod_file(&write(tag, &format!("{HEAD}{tail}")), &[]) {
        Ok(_) => String::new(),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn unused_exogenous() {
    let src = "var y; varexo e extra; parameters a; a=0.5;\nmodel; y = a*y(-1) + e; end;\n";
    let e = match parse_mod_file(&write("extra_exo", src), &[]) {
        Ok(_) => String::new(),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        e.contains(
            "extra not used in model block. To bypass this error, use the `nostrict` option. This may lead to crashes or unexpected behavior."
        ),
        "{e}"
    );
    // nostrict: a warning
    let o = ParseOptions { nostrict: true, ..Default::default() };
    let mf = parse_mod_file_opts(&write("extra_exo_ns", src), &o).unwrap();
    assert!(
        mf.warnings
            .iter()
            .any(|w| w == "WARNING: extra not used in model block, removed by nostrict command-line option"),
        "{:?}",
        mf.warnings
    );
    // an exogenous variable used through a model-local variable counts; varexo_det does not need to
    let ok = "var y; varexo e; varexo_det d; parameters a; a=0.5;\nmodel; # m = e; y = a*y(-1) + m; end;\n";
    parse_mod_file(&write("local_exo", ok), &[]).unwrap();
    // nor an observed exogenous variable (tests/exogenous-observed-variables/preprocessor.mod)
    let ok = "var y; varexo e x; parameters a; a=0.5;\nmodel; y = a*y(-1) + e; end;\nvarobs y; varexobs x;\n";
    parse_mod_file(&write("varexobs", ok), &[]).unwrap();
    // nor the `growth` of a `pac_model`, which the `pac_expectation` substitution brings into the
    // model (tests/pac/trend-component-12/example1.mod)
    let pac = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sweep72/pac_growth_exo.mod");
    parse_mod_file(&pac, &[]).unwrap();
}

#[test]
fn estimated_parameter_in_shocks() {
    let e = err(
        "shocks_est",
        "shocks; var e; stderr b; var u = a^2; end;\nestimated_params; a, 0.5, 0, 1; b, 0.1, 0, 1; end;\nvarobs y;\n",
    );
    assert!(
        e.contains(
            "some estimated parameters (a, b) also appear in the expressions defining the variance/covariance matrix of shocks; this is not allowed."
        ),
        "{e}"
    );
}

#[test]
fn estimation_mh_tune_options() {
    let est = "estimated_params; a, 0.5, 0, 1; end;\nvarobs y;\n";
    let e = err("tune1", &format!("{est}estimation(datafile=x, mh_jscale=0.2, mh_tune_jscale=0.33);\n"));
    assert!(
        e.contains("The mh_tune_jscale and mh_jscale options of the estimation statement are incompatible."),
        "{e}"
    );
    let e = err("tune2", &format!("{est}estimation(datafile=x, mh_tune_guess=0.33);\n"));
    assert!(
        e.contains("The option mh_tune_guess in estimation statement cannot be used without option mh_tune_jscale."),
        "{e}"
    );
    assert_eq!(err("tune3", &format!("{est}estimation(datafile=x, mh_tune_jscale, mh_tune_guess=0.33);\n")), "");
}

#[test]
fn extended_path_hybrid() {
    let e = err("hybrid", "extended_path(periods=10, hybrid=3);\n");
    assert!(
        e.contains("line 3, cols 27-34: The 'hybrid' option in stochastic extended path must be an even integer"),
        "{e}"
    );
    assert_eq!(err("hybrid2", "extended_path(periods=10, hybrid=2);\n"), "");
}

#[test]
fn occbin_constraints_symbols() {
    let e = err("occ1", "occbin_constraints;\nname 'A'; bind y<-e; relax y>0;\nend;\n");
    assert!(e.contains("line 4, col 19: Exogenous variable e cannot be used in 'occbin_constraints'."), "{e}");
    let e = err("occ2", "occbin_constraints;\nname 'A'; bind y<-zz; relax y>0;\nend;\n");
    assert!(e.contains("cols 19-20: Exogenous variable zz cannot be used in 'occbin_constraints'."), "{e}");
    let e = err("occ3", "occbin_constraints;\nname 'A'; bind y<c(-1); relax y>0;\nend;\n");
    assert!(e.contains("cols 18-22: Leads and lags on variables are forbidden in 'occbin_constraints'."), "{e}");
}

#[test]
fn filter_initial_state_lag() {
    // c appears without a lag: c(0) (the state c(-1)) is not a state of the model
    let e = err("fis1", "filter_initial_state;\nc(0)=1;\nend;\n");
    assert!(
        e.contains(
            "line 4, cols 1-7: filter_initial_state: variable c does not appear in the model with the lag -1 (see the reference manual for the timing convention in 'filter_initial_state')"
        ),
        "{e}"
    );
    assert_eq!(err("fis2", "filter_initial_state;\ny(0)=1;\nend;\n"), "");
}
