//! Parity fixes in the method of moments (`+mom`).
mod common;

use std::path::Path;

use dynare::driver::{RunOptions, run};

/// SMM with estimated measurement errors: `display_estimation_results_table.m` stores them in
/// `<field>_mode (smm_mode).measurement_errors_std` / `measurement_errors_corr` (and `_std_at_mode`);
/// `write_results` used to hit `unreachable!` on them.
#[test]
fn smm_with_measurement_errors_writes_results() {
    let refdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/mom");
    let src = std::fs::read_to_string(refdir.join("rbc_mom.mod")).unwrap();
    let dir = common::temp_dir("mom_me");
    std::fs::copy(refdir.join("rbc_mom_data1.csv"), dir.join("rbc_mom_data1.csv")).unwrap();
    let text = src
        .replace("@#ifndef CASE\n@#define CASE = 1\n@#endif", "@#define CASE = 4")
        .replace("stderr eg, 0.006;\npsi, 2.1, 0.5, 5;", "stderr eg, 0.006;\nstderr yo, 0.01, 0.0001, 1;\nstderr co, 0.01, 0.0001, 1;\ncorr yo, co, 0.1, -0.9, 0.9;\npsi, 2.1, 0.5, 5;")
        .replace("simulation_multiple = 5, burnin = 200", "simulation_multiple = 2, burnin = 100, optim = ('MaxIter', 5)");
    assert!(text.contains("stderr yo"));
    let p = dir.join("rbc_mom.mod");
    std::fs::write(&p, text).unwrap();
    run(&p, &RunOptions { quiet: true, ..Default::default() }).unwrap();
    let r: serde_json::Value = common::json_ref(dir.join("rbc_mom/Output/rbc_mom_mom_results.json"));
    let mode = &r["smm_mode"];
    let std = &r["smm_std_at_mode"];
    for (g, n) in
        [("measurement_errors_std", "yo"), ("measurement_errors_std", "co"), ("measurement_errors_corr", "yo_co")]
    {
        assert!(mode[g][n].as_f64().is_some_and(f64::is_finite), "{g}.{n} missing in mode");
        // NaN (null) when the 5-iteration mode sits on a bound and the Hessian is singular,
        // which depends on the platform's BLAS
        assert!(std[g].get(n).is_some_and(|v| v.is_number() || v.is_null()), "{g}.{n} missing in std_at_mode");
    }
    assert!(mode["shocks_std"]["ez"].is_number() && mode["parameters"]["psi"].is_number());
}

// ---------------------------------------------------------------------------------------------
// identification
// ---------------------------------------------------------------------------------------------

use dynare::identification::derivs::ModelParamDerivs;
use dynare::identification::run::{IdentOptions, build_context};
use dynare::modfile::parse_mod_file;

fn write_mod(tag: &str, text: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = common::temp_dir(tag);
    let p = dir.join(format!("{tag}.mod"));
    std::fs::write(&p, text).unwrap();
    (dir, p)
}

fn run_quiet(p: &Path) -> anyhow::Result<()> {
    run(p, &RunOptions { quiet: true, ..Default::default() }).map(|_| ())
}

/// identification-mom-3: the order condition raises `ar` (Dynare sets `options_.ar = nlags`),
/// and the simulated moment uncertainty then uses the raised lag count; used to panic with an
/// index out of bounds.
#[test]
fn identification_order_condition_raises_ar() {
    let (_, p) = write_mod(
        "pim_ar1",
        "var x y; varexo e; parameters rho a c; rho=0.5; a=1; c=0.3;
model; x = rho*x(-1)+e; y = a*x+c; end;
initval; x=0; y=0.3; end; steady;
shocks; var e; stderr 0.1; end;
varobs y;
identification(no_identification_minimal, no_identification_spectrum, nograph, replic=10, periods=250);
",
    );
    run_quiet(&p).unwrap();
}

/// identification-mom-14: the simulated moments drop the global `options_.drop` (set here by
/// `stoch_simul(drop=...)`): the simulation of periods + 100 periods is shorter than the drop.
#[test]
fn identification_uses_global_drop() {
    let model = |drop: usize| {
        format!(
            "var x y; varexo e; parameters rho a c; rho=0.5; a=1; c=0.3;
model; x = rho*x(-1)+e; y = a*x+c; end;
initval; x=0; y=0.3; end; steady;
shocks; var e; stderr 0.1; end;
varobs y;
stoch_simul(order=1, irf=0, nomoments, nograph, noprint, periods=500, drop={drop});
identification(no_identification_minimal, no_identification_spectrum, nograph, replic=10, periods=300, ar=1);
"
        )
    };
    let (_, p) = write_mod("pim_drop_err", &model(500));
    let err = run_quiet(&p).unwrap_err().to_string();
    assert!(err.contains("horizon of simulation is shorter than the number of observations to be dropped"), "{err}");
    // drop = 100 (the default) runs
    let (_, p) = write_mod("pim_drop_ok", &model(100));
    run_quiet(&p).unwrap();
}

/// identification-mom-17: `tex` with the eps graph format writes the TeX eps-loader files of
/// identification.plot.m and the bruteforce collinearity tables.
#[test]
fn identification_tex_files() {
    let src =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/identification/nk_ident.mod"))
            .unwrap()
            .replace("@#ifndef CASE\n@#define CASE = 1\n@#endif", "@#define CASE = 1")
            .replace(
                "identification(advanced=1, max_dim_cova_group=2);",
                "identification(advanced=1, max_dim_cova_group=2, tex, graph_format=eps);",
            );
    let (dir, p) = write_mod("pim_tex", &src);
    run_quiet(&p).unwrap();
    let d = dir.join("pim_tex/identification");
    let strength = std::fs::read_to_string(d.join("pim_tex_ident_strength_prior_mean.tex")).unwrap();
    assert!(strength.starts_with("% TeX eps-loader file generated by identification.plot.m (Dynare).\n"));
    assert!(strength.contains(
        "\\includegraphics[width=0.8\\textwidth]{pim_tex/identification/pim_tex_ident_strength_prior_mean}\n"
    ));
    assert!(strength.contains(
        "\\caption{Prior mean  - Identification using info from observables.}\\label{Fig:ident:prior_mean}\n"
    ));
    assert!(strength.ends_with("\\end{figure}\n\n% End Of TeX file. \n"));
    let sens = std::fs::read_to_string(d.join("pim_tex_sensitivity_prior_mean.tex")).unwrap();
    assert!(sens.contains("\\caption{Prior mean  - Sensitivity plot.}\\label{Fig:sensitivity:prior_mean}"));
    let col = std::fs::read_to_string(d.join("pim_tex_ident_collinearity_prior_mean_1.tex")).unwrap();
    assert!(col.contains("Collinearity patterns with 1 parameter(s).}\\label{Fig:collinearity:prior_mean:1_pars}"));
    assert!(d.join("pim_tex_collin_patterns_prior_mean_1.tex").exists());
}

/// Steady state and Jacobians at `oo_.exo_steady_state` (initval value of the exogenous
/// variable): y = rho*y(-1) + exp(e) has ghu = exp(e_ss) at e_ss = 0.3, and the steady state
/// exp(0.3)/(1-rho).
#[test]
fn identification_evaluates_at_exo_steady_state() {
    let (_, p) = write_mod(
        "pim_exo",
        "var y; varexo e; parameters rho; rho=0.5;
model; y = rho*y(-1)+exp(e); end;
initval; e=0.3; y=exp(0.3)/(1-rho); end;
shocks; var e; stderr 0.1; end;
varobs y;
estimated_params; rho, beta_pdf, 0.5, 0.1; stderr e, inv_gamma_pdf, 0.1, 2; end;
",
    );
    let mf = parse_mod_file(&p, &[]).unwrap();
    let ctx = build_context(&mf, IdentOptions::default()).unwrap();
    let (theta, _, _) = ctx.point_vector().unwrap();
    let pt = ctx.solve(&theta).unwrap();
    let e = 0.3f64.exp();
    assert!((pt.ys[0] - e / 0.5).abs() < 1e-9, "{}", pt.ys[0]);
    assert!((pt.dr.ghu[(0, 0)] - e).abs() < 1e-9, "{}", pt.dr.ghu[(0, 0)]);
    assert!((pt.dr.ghx[(0, 0)] - 0.5).abs() < 1e-12);
}

/// The identification guard against STEADY_STATE() in the model is lifted: the analytic
/// derivatives of the dynamic Jacobian (with d STEADY_STATE(y) / d params) equal central finite
/// differences of the Jacobian at the steady state of the perturbed parameters.
#[test]
fn identification_steady_state_param_derivs_match_finite_differences() {
    let (_, p) = write_mod(
        "pim_ss",
        "var y c; varexo e; parameters a b; a=1; b=0.5;
model;
y = a + b*y(-1) + e;
c = STEADY_STATE(y)*y + STEADY_STATE(y)^2*y(-1)^2;
end;
shocks; var e; stderr 0.1; end;
varobs c;
estimated_params; a, normal_pdf, 1, 0.1; b, beta_pdf, 0.5, 0.1; stderr e, inv_gamma_pdf, 0.1, 2; end;
",
    );
    let mf = parse_mod_file(&p, &[]).unwrap();
    // the builder no longer refuses the model
    let ctx = build_context(&mf, IdentOptions::default()).unwrap();
    assert!(ctx.model.uses_steady_state());
    let model = &mf.model;
    let ss = |a: f64, b: f64| {
        let y = a / (1.0 - b);
        vec![y, y * y + y.powi(4)]
    };
    let x = vec![0.0];
    let (a, b) = (1.0, 0.5);
    let pd = ModelParamDerivs::new(model);
    let md = pd.eval(model, &ss(a, b), &[a, b], 1.0 + 1e-6).unwrap();
    let h = 1e-6;
    for (q, (da, db)) in [(1.0, 0.0), (0.0, 1.0)].into_iter().enumerate() {
        let gp = model.dynamic_jacobian(&ss(a + h * da, b + h * db), &x, &[a + h * da, b + h * db]);
        let gm = model.dynamic_jacobian(&ss(a - h * da, b - h * db), &x, &[a - h * da, b - h * db]);
        let fd = (&gp - &gm) / (2.0 * h);
        let d = (&md.dg1[q] - &fd).amax();
        println!("dg1[{q}] vs finite differences: {d:.2e}");
        assert!(d < 1e-6, "param {q}: {d:e}");
    }
}

// ---------------------------------------------------------------------------------------------
// method_of_moments
// ---------------------------------------------------------------------------------------------

/// reference/mom/rbc_mom.mod with `CASE` selected and the extra replacements applied.
fn rbc_mom(tag: &str, case: u32, replace: &[(&str, &str)]) -> (std::path::PathBuf, std::path::PathBuf) {
    let refdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/mom");
    let mut text = std::fs::read_to_string(refdir.join("rbc_mom.mod"))
        .unwrap()
        .replace("@#ifndef CASE\n@#define CASE = 1\n@#endif", &format!("@#define CASE = {case}"));
    for (a, b) in replace {
        assert!(text.contains(a), "no `{a}` in rbc_mom.mod");
        text = text.replace(a, b);
    }
    let dir = common::temp_dir(tag);
    for f in ["rbc_mom_data1.csv", "rbc_mom_data2.csv"] {
        std::fs::copy(refdir.join(f), dir.join(f)).unwrap();
    }
    let p = dir.join(format!("{tag}.mod"));
    std::fs::write(&p, text).unwrap();
    (dir, p)
}

/// macro-parser/identification-mom-4: with priors, an estimated parameter without an initial
/// value starts at the prior mean (set_prior.m: `xparam1(isnan(xparam1)) = bayestopt_.p1`), not
/// at its calibrated value (rho = 0.95 in the model, prior mean 0.9). `mode_compute = 0` keeps
/// the starting point.
#[test]
fn mom_bayesian_nan_initial_value_starts_at_prior_mean() {
    let (dir, p) = rbc_mom(
        "pm_nan_init",
        6,
        &[
            ("rho, 0.92, , , beta_pdf, 0.9, 0.05;", "rho, , , , beta_pdf, 0.9, 0.05;"),
            (
                "mode_compute = 4,\n                  weighting_matrix = ['IDENTITY_MATRIX', 'DIAGONAL']",
                "mode_compute = 0,\n                  weighting_matrix = ['IDENTITY_MATRIX']",
            ),
        ],
    );
    run_quiet(&p).unwrap();
    let r: serde_json::Value = common::json_ref(dir.join("pm_nan_init/Output/pm_nan_init_mom_results.json"));
    let key = r.as_object().unwrap().keys().find(|k| k.ends_with("_mode")).cloned().expect("a *_mode entry");
    let rho = r[&key]["parameters"]["rho"].as_f64().unwrap();
    assert!((rho - 0.9).abs() < 1e-12, "rho {rho}");
}

/// identification-mom-10: without a `datafile` option method_of_moments takes the data of a
/// preceding `data(file=...)` command (makedataset.m uses options_.dataset.file).
#[test]
fn mom_uses_the_data_command() {
    let opts = "method_of_moments(mom_method = GMM, datafile = 'rbc_mom_data1.csv', order = 1, mode_compute = 5,
                  weighting_matrix = ['DIAGONAL', 'OPTIMAL'], se_tolx = 1e-6, verbose, nograph@{AN});";
    let small = "method_of_moments(mom_method = GMM, order = 1, mode_compute = 5, optim = ('MaxIter', 5),
                  weighting_matrix = ['DIAGONAL'], verbose, nograph@{AN});";
    let with_datafile = opts
        .replace("['DIAGONAL', 'OPTIMAL']", "['DIAGONAL']")
        .replace("mode_compute = 5,", "mode_compute = 5, optim = ('MaxIter', 5),");
    let (d1, p1) = rbc_mom("pm_datafile", 1, &[(opts, &with_datafile)]);
    let (d2, p2) = rbc_mom("pm_datacmd", 1, &[(opts, &format!("data(file='rbc_mom_data1.csv');\n{small}"))]);
    run_quiet(&p1).unwrap();
    run_quiet(&p2).unwrap();
    let a: serde_json::Value = common::json_ref(d1.join("pm_datafile/Output/pm_datafile_mom_results.json"));
    let b: serde_json::Value = common::json_ref(d2.join("pm_datacmd/Output/pm_datacmd_mom_results.json"));
    let ka = a.as_object().unwrap().keys().find(|k| k.ends_with("_mode")).cloned().unwrap();
    assert_eq!(a[&ka]["parameters"], b[&ka]["parameters"]);
}

/// identification-mom-12: SMM with a filter (hp_filter) on data with missing values is an
/// error in +mom/run.m.
#[test]
fn mom_missing_data_with_filter_is_an_error() {
    let (dir, p) =
        rbc_mom("pm_missing_hp", 4, &[("simulation_multiple = 5,", "simulation_multiple = 5, hp_filter = 1600,")]);
    let csv = std::fs::read_to_string(dir.join("rbc_mom_data1.csv")).unwrap();
    let mut lines: Vec<String> = csv.lines().map(str::to_string).collect();
    let mut f: Vec<&str> = lines[10].split(',').collect();
    f[1] = "";
    lines[10] = f.join(",");
    std::fs::write(dir.join("rbc_mom_data1.csv"), lines.join("\n") + "\n").unwrap();
    let e = format!("{:#}", run_quiet(&p).unwrap_err());
    assert!(e.contains("missing data is incompatible with filtering data"), "{e}");
}

/// identification-mom-5: estimated measurement errors are rejected for GMM only
/// (+mom/run.m), with Dynare's message.
#[test]
fn mom_gmm_rejects_estimated_measurement_errors() {
    let (_, p) = rbc_mom(
        "pm_gmm_me",
        1,
        &[(
            "stderr eg, 0.006;\npsi, 2.1, 0.5, 5;",
            "stderr eg, 0.006;\nstderr yo, 0.01, 0.0001, 1;\npsi, 2.1, 0.5, 5;",
        )],
    );
    let e = format!("{:#}", run_quiet(&p).unwrap_err());
    println!("{e}");
    assert!(e.contains("GMM") && e.to_lowercase().contains("measurement error"), "{e}");
}
