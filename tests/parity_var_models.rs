//! var-models parity fixes (backward models, SBVAR/MS-SBVAR global options, BVAR datafiles,
//! PAC `a2alpha`). Expected values are hand-derived from the Dynare 7.2 sources named in each
//! test (`+backward_model/initialize.m`, `simul_backward_model.m`, `simul_static_model.m`,
//! `read_variables.m`, `initialize_ms_sbvar_options.m`, `a2alpha.m`).
mod common;

use std::path::{Path, PathBuf};

use dynare::bvar::commands::read_variables;
use dynare::data::DataOptions;
use dynare::driver::{RunOptions, run};
use dynare::modfile::parse_mod_file;
use dynare::msbvar::{SbvarOptions, prepare_sample};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// Runs `text` as `<tag>.mod` in a fresh directory holding the `files` of reference/; returns the
/// directory and the result.
fn run_mod(tag: &str, files: &[(&str, &str)], text: &str) -> (PathBuf, anyhow::Result<()>) {
    let dir = common::temp_dir(tag);
    for (src, dst) in files {
        std::fs::copy(root().join("reference").join(src), dir.join(dst)).unwrap();
    }
    let f = dir.join(format!("{tag}.mod"));
    std::fs::write(&f, text).unwrap();
    let r = run(&f, &RunOptions { quiet: true, ..Default::default() });
    (dir, r)
}

fn err(r: anyhow::Result<()>) -> String {
    format!("{:#}", r.expect_err("the run should fail"))
}

/// `<tag>/Output/<tag>_<name>.csv` as (header, rows of values).
fn series(dir: &Path, tag: &str, name: &str) -> (Vec<String>, Vec<(String, Vec<f64>)>) {
    let p = dir.join(tag).join("Output").join(format!("{tag}_{name}.csv"));
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let mut lines = s.lines();
    let head: Vec<String> = lines.next().unwrap().split(',').skip(1).map(str::to_string).collect();
    let rows = lines
        .map(|l| {
            let mut it = l.split(',');
            let d = it.next().unwrap().to_string();
            (d, it.map(|x| x.parse().unwrap()).collect())
        })
        .collect();
    (head, rows)
}

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() < 1e-12, "{what}: {a} vs {b}");
}

const BW_HEAD: &str = "var y z;\nvarexo e u;\nparameters a b;\na = 0.5; b = 0.2;\n";
const BW_SHOCKS: &str = "shocks; var e; stderr 1; var u; stderr 1; end;\n";

/// var-models-1 and -3: `simul_backward_model([], ...)` starts from `M_.endo_histval`, with
/// `orig_maximum_lag` rows (here 2: y(0)=1, y(-1)=3, z(0)=2, z(-1)=0), as in
/// `+backward_model/initialize.m`; zero innovations dated from 2Y.
/// y(t) = 0.5 y(t-1) + 0.1 y(t-2) + 0.2 z(t-1), z(t) = 0.9 z(t-1): the dseries returned starts
/// with the two initial rows (1Y: y=3 z=0, 2Y: y=1 z=2), then 3Y y=1.2 z=1.8, 4Y y=1.06 z=1.62.
#[test]
fn backward_model_starts_from_histval_with_two_lags() {
    let text = format!(
        "{BW_HEAD}model;\ny = a*y(-1) + 0.1*y(-2) + b*z(-1) + e;\nz = 0.9*z(-1) + u;\nend;\n\
histval;\ny(0) = 1;\ny(-1) = 3;\nz(0) = 2;\nend;\n{BW_SHOCKS}\
X = simul_backward_model([], 5, dseries(zeros(5,2),2Y,{{'e','u'}}));\n"
    );
    let (dir, r) = run_mod("pvm_bw2", &[], &text);
    r.unwrap();
    let (head, rows) = series(&dir, "pvm_bw2", "X");
    let (iy, iz) = (head.iter().position(|n| n == "y").unwrap(), head.iter().position(|n| n == "z").unwrap());
    // the simulation is dated from the first innovation date minus the initial rows
    let first = rows.iter().position(|(d, _)| d == "1Y").expect("starts at 1Y");
    let at = |k: usize| (rows[first + k].1[iy], rows[first + k].1[iz]);
    assert_eq!(rows[first].0, "1Y");
    let (y1, z1) = at(0);
    close(y1, 3.0, "1Y y");
    close(z1, 0.0, "1Y z");
    let (y2, z2) = at(1);
    close(y2, 1.0, "2Y y");
    close(z2, 2.0, "2Y z");
    // 3Y: y = 0.5*1 + 0.1*3 + 0.2*2 = 1.2 ; z = 0.9*2 = 1.8
    let (y3, z3) = at(2);
    close(y3, 1.2, "3Y y");
    close(z3, 1.8, "3Y z");
    // 4Y: y = 0.5*1.2 + 0.1*1 + 0.2*1.8 = 1.06 ; z = 1.62
    let (y4, z4) = at(3);
    close(y4, 1.06, "4Y y");
    close(z4, 1.62, "4Y z");
}

/// var-models-1: without histval the default initial conditions are zero, and a model with
/// two lags runs (var-models-3).
#[test]
fn backward_model_zero_default_initial_conditions() {
    let text = format!(
        "{BW_HEAD}model;\ny = a*y(-1) + 0.1*y(-2) + b*z(-1) + e;\nz = 0.9*z(-1) + u;\nend;\n{BW_SHOCKS}\
X = simul_backward_model([], 3, dseries([1;0;0],2Y,{{'e'}}));\n"
    );
    let (dir, r) = run_mod("pvm_bw0", &[], &text);
    r.unwrap();
    let (head, rows) = series(&dir, "pvm_bw0", "X");
    let iy = head.iter().position(|n| n == "y").unwrap();
    let first = rows.iter().position(|(d, _)| d == "3Y").unwrap();
    // e = 1 in the first simulated period only; u is zero (not available in the third argument)
    close(rows[first].1[iy], 1.0, "y(1)");
    close(rows[first + 1].1[iy], 0.5, "y(2)");
    close(rows[first + 2].1[iy], 0.5 * 0.5 + 0.1 * 1.0, "y(3)");
}

/// var-models-4: with `[]` as first argument the innovations need not start in 2Y (no
/// contiguity check; simul_backward_model.m only tests `innovations.nobs`).
#[test]
fn backward_model_innovation_dates_are_free_with_empty_initial_conditions() {
    let model = format!("{BW_HEAD}model;\ny = a*y(-1) + b*z(-1) + e;\nz = 0.9*z(-1) + u;\nend;\n{BW_SHOCKS}");
    let a = format!("{model}X = simul_backward_model([], 3, dseries([1;0;0],5Y,{{'e'}}));\n");
    let b = format!("{model}X = simul_backward_model([], 3, dseries([1;0;0],2Y,{{'e'}}));\n");
    let (da, ra) = run_mod("pvm_dates_a", &[], &a);
    ra.unwrap();
    let (db, rb) = run_mod("pvm_dates_b", &[], &b);
    rb.unwrap();
    let (ha, xa) = series(&da, "pvm_dates_a", "X");
    let (hb, xb) = series(&db, "pvm_dates_b", "X");
    assert_eq!(ha, hb);
    let ya: Vec<&Vec<f64>> = xa.iter().map(|r| &r.1).collect();
    let yb: Vec<&Vec<f64>> = xb.iter().map(|r| &r.1).collect();
    assert_eq!(ya, yb);
    // y(1) = e(1) = 1, y(2) = 0.5, y(3) = 0.25
    let iy = ha.iter().position(|n| n == "y").unwrap();
    let last = xa.len() - 3;
    close(xa[last].1[iy], 1.0, "y(1)");
    close(xa[last + 2].1[iy], 0.25, "y(3)");
}

/// var-models-4: a dseries as first argument still needs contiguous dates.
#[test]
fn backward_model_dseries_initial_conditions_must_be_contiguous() {
    let text = format!(
        "{BW_HEAD}model;\ny = a*y(-1) + b*z(-1) + e;\nz = 0.9*z(-1) + u;\nend;\n{BW_SHOCKS}\
ic = dseries([1 2], 1Y, {{'y','z'}});\n\
X = simul_backward_model(ic, 3, dseries([1;0;0],5Y,{{'e'}}));\n"
    );
    let (_d, r) = run_mod("pvm_contig", &[], &text);
    assert!(err(r).contains("contiguous"));
}

/// var-models-6: `backward_model.simul_static_model`: y = a + e solved period by period, dated
/// from the first innovation date; without innovations simul_backward_model passes an
/// undefined third argument on, so Dynare errors.
#[test]
fn static_model_simulation() {
    let model = "var y;\nvarexo e;\nparameters a;\na = 0.5;\nmodel;\ny = a + e;\nend;\nshocks; var e; stderr 1; end;\n";
    let (dir, r) =
        run_mod("pvm_static", &[], &format!("{model}X = simul_backward_model([], 3, dseries([1;2;3],4Y,{{'e'}}));\n"));
    r.unwrap();
    let (head, rows) = series(&dir, "pvm_static", "X");
    let (iy, ie) = (head.iter().position(|n| n == "y").unwrap(), head.iter().position(|n| n == "e").unwrap());
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].0, "4Y");
    for (t, r) in rows.iter().enumerate() {
        close(r.1[iy], 0.5 + (t as f64 + 1.0), "y");
        close(r.1[ie], t as f64 + 1.0, "e");
    }
    let (_d, r) = run_mod("pvm_static0", &[], &format!("{model}X = simul_backward_model([], 3);\n"));
    assert!(r.is_err());
}

/// var-models-8: `read_variables.m` reads `.mat`, `.xlsx` and `.csv` (the CSV is the
/// independent source of the numbers).
#[test]
fn read_variables_reads_csv_mat_and_xlsx() {
    let names: Vec<String> = ["dy", "pinfobs", "robs"].iter().map(|s| s.to_string()).collect();
    let dir = common::temp_dir("pvm_readvars");
    let csv = root().join("reference/sw07/sw07_obs.csv");
    let r = |f: &str, src: &Path, ext: &str| {
        let d = dir.join(ext);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::copy(src, d.join(format!("{f}.{ext}"))).unwrap();
        let xls = DataOptions { xls_sheet: Some("data".into()), xls_range: Some("B3:K98".into()) };
        read_variables(&d, f, &names, &xls).unwrap()
    };
    let want = r("sw07_obs", &csv, "csv");
    assert_eq!(want.nrows(), 95);
    for ext in ["mat", "xlsx"] {
        let got = r("sw07_obs", &root().join(format!("reference/data/sw07_obs.{ext}")), ext);
        assert_eq!(got.shape(), want.shape(), "{ext}");
        let d = (&got - &want).iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(d < 1e-9, "{ext}: max diff {d}");
    }
}

/// var-models-8 and -10 through the commands: `bvar_density` reads an `.xlsx` datafile; without
/// `first_obs` (Dynare's default is NaN) the toolbox fails; `estimation`/`calib_smoother` would
/// set it to 1.
#[test]
fn bvar_commands_read_spreadsheets_and_need_first_obs() {
    let head = "var dy pinfobs robs;\nvarobs dy pinfobs robs;\n";
    let (_d, r) = run_mod(
        "pvm_bv_ok",
        &[("data/sw07_obs.xlsx", "sw07_obs.xlsx")],
        &format!(
            "{head}bvar_density(datafile = sw07_obs, xls_sheet = data, xls_range = B3:K98, first_obs = 20, nobs = 60) 2;\n"
        ),
    );
    r.unwrap();
    let (_d, r) = run_mod(
        "pvm_bv_nofo",
        &[("data/sw07_obs.xlsx", "sw07_obs.xlsx")],
        &format!("{head}bvar_density(datafile = sw07_obs, xls_sheet = data, xls_range = B3:K98, nobs = 60) 2;\n"),
    );
    assert!(err(r).contains("first_obs"));
    // a first_obs from an earlier command is kept (global options_)
    let (_d, r) = run_mod(
        "pvm_bv_keep",
        &[("data/sw07_obs.xlsx", "sw07_obs.xlsx")],
        &format!(
            "{head}bvar_density(datafile = sw07_obs, xls_sheet = data, xls_range = B3:K98, first_obs = 20, nobs = 60) 2;\nbvar_density(nobs = 60) 3;\n"
        ),
    );
    r.unwrap();
}

const MS_MODEL: &str =
    "var y1 y2;\nvarexo e1;\nparameters a;\na = 0.5;\nmodel;\ny1 = a*y1(-1) + e1;\ny2 = y1;\nend;\nvarobs y1 y2;\n";

fn ms_commands(tag: &str, body: &str) -> (PathBuf, dynare::modfile::ModFile) {
    let dir = common::temp_dir(tag);
    std::fs::copy(root().join("reference/msbvar/ms_est/d2.csv"), dir.join("d2.csv")).unwrap();
    let f = dir.join(format!("{tag}.mod"));
    std::fs::write(&f, format!("{MS_MODEL}{body}")).unwrap();
    let mf = parse_mod_file(&f, &[]).unwrap();
    (dir, mf)
}

fn command_options<'a>(mf: &'a dynare::modfile::ModFile, name: &str, nth: usize) -> &'a dynare::modfile::Options {
    mf.commands
        .iter()
        .map(dynare::bvar::commands::command_parts)
        .filter(|(n, _)| *n == name)
        .nth(nth)
        .unwrap_or_else(|| panic!("no {name} #{nth}"))
        .1
}

/// var-models-2: the preprocessor emits `initialize_ms_sbvar_options` before every `ms_*`
/// command, resetting the `options_.ms.nlags` that `svar_identification` wrote; `sbvar` has no
/// reset, so it keeps the identification's largest lag.
#[test]
fn ms_estimation_resets_nlags_but_sbvar_keeps_the_identification_lag() {
    let ident = "svar_identification;\nexclusion lag 2;\nequation 1, y2;\nend;\nmarkov_switching(chain=1, number_of_regimes=2, duration=4);\nsvar(variances, chain=1);\n";
    // sbvar alone (no ms_* before it) uses the identification's largest lag
    let (dir, mf) =
        ms_commands("pvm_msnl2", &format!("{ident}sbvar(datafile=d2, freq=4, initial_year=1960, final_year=1988);\n"));
    let sb = prepare_sample(&mf, command_options(&mf, "sbvar", 0), &dir).unwrap();
    assert_eq!(sb.settings.nlags, 2);
    // lag-0 exclusions: ms_estimation estimates with one lag
    let (dir, mf) = ms_commands(
        "pvm_msnl3",
        "svar_identification;\nexclusion lag 0;\nequation 2, y1;\nend;\nmarkov_switching(chain=1, number_of_regimes=2, duration=4);\nsvar(variances, chain=1);\n\
ms_estimation(datafile=d2, freq=4, initial_year=1960, final_year=1988);\n",
    );
    let ms = prepare_sample(&mf, command_options(&mf, "ms_estimation", 0), &dir).unwrap();
    assert_eq!(ms.settings.nlags, 1);
}

/// var-models-9: `sbvar` sees the global `options_`: `forecast`, `datafile`, `freq`,
/// `initial_year` left by earlier commands (sbvar.m: forecast 0 -> 4 years).
#[test]
fn sbvar_inherits_global_options() {
    let ident = "svar_identification;\nupper_cholesky;\nend;\n";
    let (dir, mf) = ms_commands(
        "pvm_sbg",
        &format!(
            "{ident}sbvar(datafile=d2, freq=4, initial_year=1960, final_year=1988, nlags=3, forecast=8);\n\
sbvar(nlags=2);\n"
        ),
    );
    let first = prepare_sample(&mf, command_options(&mf, "sbvar", 0), &dir).unwrap();
    assert_eq!(first.settings.forecast_steps, 32);
    let second = prepare_sample(&mf, command_options(&mf, "sbvar", 1), &dir).unwrap();
    assert_eq!(second.settings.nlags, 2);
    assert_eq!(second.settings.forecast_steps, 32, "forecast = 8 years from the earlier sbvar");
    assert_eq!(second.settings.freq, 4);
    assert_eq!(second.opts.datafile.as_deref(), Some("d2"));
    assert_eq!(second.z.nrows(), first.z.nrows());
    // an unset forecast is 4 years
    let (dir, mf) = ms_commands(
        "pvm_sbg2",
        &format!("{ident}sbvar(datafile=d2, freq=4, initial_year=1960, final_year=1988, nlags=2);\n"),
    );
    let p = prepare_sample(&mf, command_options(&mf, "sbvar", 0), &dir).unwrap();
    assert_eq!(p.settings.forecast_steps, 16);
    let _ = SbvarOptions::default();
}
