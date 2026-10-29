//! .mod syntax that Dynare 7.2's preprocessor accepts (found by running Dynare's tests/):
//! expected values from `dynare-preprocessor` 7.2 on the same statements (driver.m).
mod common;

use dynare::modfile::{Command, MATLAB_STATEMENT, OptValue, PathValue, PriorShape, StateBlock, parse_mod_str};

const HEAD: &str = "var y c;\nvarexo e;\nparameters a b;\na = 0.5; b = 0.1;\n\
                    model;\ny = a*y(-1) + e;\nc = b*y;\nend;\n";

fn parse(body: &str) -> dynare::modfile::ModFile {
    parse_mod_str(&format!("{HEAD}{body}"), &[]).unwrap()
}

fn matlab_statements(mf: &dynare::modfile::ModFile) -> Vec<String> {
    mf.commands
        .iter()
        .filter_map(|c| match c {
            Command::Other(n, o) if n == MATLAB_STATEMENT => Some(o[0].0.clone()),
            _ => None,
        })
        .collect()
}

/// Native MATLAB statements end with their line even without ';' (tests/TeX/fs2000_corr_ME.mod,
/// histval_initval_file/ramst_initval_file_with_dseries.mod, loglinear/example4_exp.mod), and
/// the MATLAB functions dynare-rust runs take the arguments of their line only
/// (estimation/slice/fs2000_estimate_initial_states.mod).
#[test]
fn native_statements_end_with_their_line() {
    let mf = parse(
        "[status, cmdout] = system('true');\nif status\n    cmdout\n    error('TeX-File did not compile.')\nend\n\
         Simulated_time_series(2000Q3:2050Q2)\nsave results_exp.mat oo_exp\n\
         datatomfile('fsdat_simul',{'y'})\ndelete example1.data.mat\nstoch_simul(order=1, irf=0) y\n  c;\n",
    );
    let m = matlab_statements(&mf);
    // (`delete f.mat`: pac/var-10e/example2.mod)
    assert_eq!(m.len(), 8, "{m:?}");
    assert_eq!(m[7], "delete example1 . data . mat");
    assert!(m[1].starts_with("if status"));
    assert_eq!(m[2], "cmdout");
    let dt = mf.commands.iter().find_map(|c| match c {
        Command::Other(n, o) if n == "datatomfile" => Some(o.clone()),
        _ => None,
    });
    let dt = dt.expect("datatomfile");
    assert_eq!(dt[1].1, OptValue::List(vec![OptValue::Str("y".into())]));
    // a Dynare command continues on the next line
    assert!(mf.commands.iter().any(|c| matches!(c, Command::StochSimul(_, v) if v == &["y", "c"])));
}

/// datatomfile('f', []) and a column cell (discretionary_policy/dennis_1.mod, arima/mod1.mod).
#[test]
fn datatomfile_empty_list_and_column_cell() {
    let mf = parse("datatomfile('x', {'y'; 'c';});\ndatatomfile('x', []);\n");
    let lists: Vec<_> = mf
        .commands
        .iter()
        .filter_map(|c| match c {
            Command::Other(n, o) if n == "datatomfile" => Some(o[1].1.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(lists[0], OptValue::List(vec![OptValue::Str("y".into()), OptValue::Str("c".into())]));
    assert_eq!(lists[1], OptValue::List(vec![]));
}

/// set_dynare_seed takes no `options_` defaults (estimation/dsmh/dsge_base_DSMH.mod failed with
/// the command line's `nograph` appended to its arguments).
#[test]
fn set_dynare_seed_has_positional_arguments_only() {
    let mf = parse("options_.nograph = 1;\nset_dynare_seed(2000);\n");
    let o = mf.commands.iter().find_map(|c| match c {
        Command::Other(n, o) if n == "set_dynare_seed" => Some(o.clone()),
        _ => None,
    });
    assert_eq!(o.unwrap(), vec![("2000".to_string(), OptValue::Flag)]);
}

/// Keywords are case-insensitive: `Var` (optimal_policy/OSR/osr_example_obj_corr_non_stat_vars.mod),
/// `NORMAL_PDF`, `GAMMA_PDF`, `beta_PDF` (bvar_a_la_sims/bvar_and_dsge.mod, gsa/morris/nk_est.mod):
/// driver.m has prior shapes 1 and 3.
#[test]
fn keywords_are_case_insensitive() {
    let mf = parse_mod_str(
        "Var y c;\nVAREXO e;\nparameters a b;\na = 0.5; b = 0.1;\nModel;\ny = a*y(-1) + e;\nc = b*y;\nend;\n\
         estimated_params;\na, 0.5, 0, 1, BETA_pdf, 0.5, 0.1;\nb, 0.1, NORMAL_PDF, 0.1, 0.01;\nend;\n",
        &[],
    )
    .unwrap();
    assert_eq!(mf.model.endo_names, ["y", "c"]);
    let p: Vec<_> = mf.estimated_params.iter().map(|e| (e.prior, e.lb)).collect();
    assert_eq!(p, [(Some(PriorShape::Beta), Some(0.0)), (Some(PriorShape::Normal), None)]);
}

/// `values 0.01 -0.02 0;` is three values (conditional_forecasts/1/fs2000_cal.mod):
/// constrained_paths_(1,1:1)=0.01; (1,2:2)=(-0.02); (1,3:5)=0. `(xx)` with
/// `xx = [0.01 -0.02 0 0 0];` fills 1:5 element by element.
#[test]
fn conditional_forecast_paths_values() {
    let mf = parse(
        "conditional_forecast_paths;\nvar y;\nperiods 1 2 3:5;\nvalues 0.01 -0.02 0;\nend;\n\
         xx = [0.01 -0.02 0 0 0];\nconditional_forecast_paths;\nvar c;\nperiods 1:5;\nvalues (xx);\nend;\n",
    );
    let blocks: Vec<_> = mf
        .commands
        .iter()
        .filter_map(|c| match c {
            Command::ConditionalForecastPaths(p) => Some(p.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        blocks[0],
        vec![(
            "y".to_string(),
            vec![(1, 1, PathValue::Num(0.01)), (2, 2, PathValue::Num(-0.02)), (3, 5, PathValue::Num(0.0))]
        )]
    );
    let r: Vec<f64> = blocks[1][0].1.iter().map(|x| if let PathValue::Num(v) = x.2 { v } else { f64::NAN }).collect();
    assert_eq!(r, [0.01, -0.02, 0.0, 0.0, 0.0]);
}

/// `[endogenous='c']` declares c (moments/example1_order2_pruning.mod: M_.endo_names(6) = {'c'}).
#[test]
fn equation_tag_declares_a_variable() {
    let mf = parse_mod_str(
        "var y;\nvarexo e;\nparameters a;\na = 0.5;\nmodel;\n[endogenous='c', name='c eq']\nc = a*y;\ny = a*y(-1) + e;\nend;\n",
        &[],
    )
    .unwrap();
    assert_eq!(mf.model.endo_names, ["y", "c"]);
}

/// An undeclared MATLAB variable in initval (decision_rules/second_order/ds1.mod:
/// `betabar = omega*cbar^(-eta);` then `r_1 = 1/betabar;`).
#[test]
fn initval_reads_mod_file_local_variables() {
    let mf = parse("betabar = a*4;\ninitval;\ny = 1/betabar;\nend;\n");
    let src = mf.state_blocks.iter().find_map(|(_, b)| match b {
        StateBlock::Initval { sources, .. } => Some(sources.clone()),
        _ => None,
    });
    let src = src.unwrap();
    let e = mf.model.compile_expr(&src[0].1).unwrap();
    let v = e.eval(&[0.0, 0.0], &[0.0], &mf.model.params);
    assert_eq!(v, 0.5);
}

/// A factor to the power 0 is dropped (AnScho_MoM_common.inc: `YGR^1*INFL^0*INT^0`; driver.m
/// for `y^1*c^0*y^0;` has `[1,], [0,], [1,]`).
#[test]
fn matched_moments_power_zero() {
    let mf = parse("matched_moments;\ny^1*c^0*y^0;\nend;\n");
    let rows = mf.commands.iter().find_map(|c| match c {
        Command::Mom(dynare::mom::MomCommand::MatchedMoments(r)) => Some(r.clone()),
        _ => None,
    });
    let r = &rows.unwrap()[0];
    assert_eq!((r.vars.clone(), r.lags.clone(), r.powers.clone()), (vec![0], vec![0], vec![1]));
}

/// heteroskedastic_shocks with dates (estimation/heteroskedastic_shocks/fs2000_het_dates.mod),
/// placed as dynare_estimation_init.m does: obs_idx = periods - dataset_.dates(1) + 1 within
/// the sample.
#[test]
fn heteroskedastic_shocks_dates() {
    let mf = parse("heteroskedastic_shocks;\nvar e;\nperiods 2000Y:2020Y;\nvalues 0.01;\nend;\n");
    let h = &mf.q3.heteroskedastic_shocks;
    assert_eq!(h.len(), 1);
    assert_eq!(h[0].dates.len(), 21);
    let init = dynare::dates::Date::parse("1995Y").unwrap();
    let s = dynare::kalman::hetero::HeteroSpec::new(h, 1, 1, 10, &["e".to_string()], init).unwrap();
    let set: Vec<usize> = (0..11).filter(|&c| !s.qvalue[(0, c)].is_nan()).collect();
    assert_eq!(set, [5, 6, 7, 8, 9]);
    assert!((s.qvalue[(0, 5)] - 1e-4).abs() < 1e-18);
}

/// Macro functions see their arguments and the global variables, not the caller's arguments
/// (macro_processor/example1_macro.mod); `inf` and `nan` are numbers and a comparison with NaN
/// is false; a list of indices; `\\` continues a directive; (bool) casts of strings.
#[test]
fn macro_function_scoping_and_inf() {
    let src = "@#define a = 1\n@#define f(x) = x + a\n@#define a = 2\n@#define g(a) = f(1) + a\n\
               @#define a = 3\n@#define h(a) = g(2) + a\n\
               @#if f(1) != 4 || g(2) != 6 || h(1) != 7\n@#error \"Problem with functions\"\n@#endif\n\
               @#if log(0) != -inf || 1/0 != inf\n@#error \"Problem with inf\"\n@#endif\n\
               @#if 5 < nan || 5 >= nan\n@#error \"Problem with NaN comparison\"\n@#endif\n\
               @#define A = \"La crise economique\"\n\
               @#if A[1,8,3,7,4,10,13,2,5,16,12,9,11,14,15,6,17:19] != \"Le scenario comique\" \\\\\n\
               || 1:1 != [1] \\\\ // comment\n || 0:0 != [0]\n@#error \"Problem with indexing\"\n@#endif\n\
               @#if (bool)\"FaLse\" || !(bool)\"TRUE\" || (bool)\"0\" || !(bool)\"-3\"\n@#error \"bool\"\n@#endif\n";
    parse(src);
}

/// datatomfile after discretionary_policy(periods=200) (discretionary_policy/dennis_1.mod):
/// oo_.endo_simul of the policy's simulation, 200 periods per series.
#[test]
fn datatomfile_after_discretionary_policy() {
    let dir = common::temp_dir("pms_dennis");
    let src = "var y i pi pi_c q;\nvarexo g u e;\nparameters omega sigma beta kappa alpha;\n\
               alpha = 0.4;\nbeta = 0.99;\nsigma = 1;\nomega = 1.64;\nkappa = 0.3;\n\
               model(linear);\ny = y(+1) -(omega/sigma)*(i-pi(+1))+g;\npi = beta*pi(+1)+kappa*y+u;\n\
               pi_c = pi+(alpha/(1-alpha))*(q-q(-1));\nq = q(+1)-(1-alpha)*(i-pi(+1))+(1-alpha)*e;\nend;\n\
               shocks;\nvar g; stderr 1;\nvar u; stderr 1;\nvar e; stderr 1;\nend;\n\
               planner_objective pi_c^2 + y^2;\n\
               discretionary_policy(instruments=(i),irf=0,planner_discount=beta, periods=200);\n\
               datatomfile('dennis_simul', {'y'; 'i'; 'pi';});\n";
    let p = dir.join("pms_dennis.mod");
    std::fs::write(&p, src).unwrap();
    dynare::driver::run(&p, &dynare::driver::RunOptions { quiet: true, ..Default::default() }).unwrap();
    let out = std::fs::read_to_string(dir.join("dennis_simul.m")).unwrap();
    let values = out.lines().filter(|l| l.trim().parse::<f64>().is_ok()).count();
    assert_eq!(values, 3 * 200);
}
