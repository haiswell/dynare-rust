// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `+mom/run.m`: options (`default_option_mom_values.m`), parameters, priors and
//! bounds, data moments, the estimation stages with iterated weighting matrices
//! (`mode_compute_gmm_smm.m`) or the IRF-matching mode (`mode_compute_irf_matching.m`),
//! standard errors (`standard_errors.m`), the J-test (`Jtest.m`), the printed tables
//! (`print_info_on_estimation_settings.m`, `display_estimation_results_table.m`,
//! `display_comparison_moments_irfs.m`) and the results file
//! `<fname>/Output/<fname>_mom_results.json` (Dynare: `method_of_moments/<fname>_mom_results.mat`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, anyhow, bail};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;

use super::moments::{Method, matched_moments_block, optimal_weighting_matrix};
use super::problem::{IrfSetup, MomObjective, MomProblem, SmmSetup, fail_err};
use super::{Blocks, MomInputs};
use crate::analysis::table::{dyntable, fmt_f};
use crate::estimation::{Bounds, Objective, Prior, PriorShape, hessian};
use crate::linalg::{Mat, chol_upper, mul, solve};
use crate::modfile::{EstimatedKind, EstimatedParam, ModFile, OptValue, Options, opt};

/// Where the outputs go.
#[derive(Clone, Debug)]
pub struct Context {
    /// Directory of the .mod file.
    pub dir: PathBuf,
    pub fname: String,
    /// `--seed` from the command line (overrides the SMM `seed` option when set).
    pub seed_override: Option<u64>,
    /// Seed of the run's random streams (`--seed`, changed by `set_dynare_seed`): Dynare draws the
    /// order>1 IRF-matching shocks and the MCMC (initial prior draws, slice sampler) from MATLAB's
    /// global stream, which `set_dynare_seed` resets, unlike the SMM `seed` option.
    pub stream_seed: u64,
}

/// `options_mom_` (the options used here).
#[derive(Clone, Debug)]
pub struct MomOptions {
    pub method: Method,
    pub datafile: Option<String>,
    pub first_obs: usize,
    pub nobs: Option<usize>,
    pub order: usize,
    pub pruning: bool,
    pub prefilter: bool,
    pub hp_filter: f64,
    pub one_sided_hp_filter: f64,
    pub bandpass: Option<[f64; 2]>,
    pub weighting_matrix: Vec<String>,
    pub weighting_matrix_scaling_factor: f64,
    pub bartlett_kernel_lag: usize,
    pub penalized_estimator: bool,
    pub se_tolx: f64,
    pub burnin: usize,
    pub seed: u64,
    pub simulation_multiple: f64,
    pub bounded_shock_support: bool,
    pub analytic_standard_errors: bool,
    /// `analytic_jacobian` (GMM only; dismissed otherwise).
    pub analytic_jacobian: bool,
    pub optimizer_vec: Vec<i32>,
    pub verbose: bool,
    pub qz_criterium: f64,
    pub schur_vec_tol: f64,
    pub cova_compute: bool,
    pub prior_trunc: f64,
    pub huge_number: f64,
    pub relative_irf: bool,
    pub add_tiny_number_to_cholesky: f64,
    pub drop: usize,
    pub replic: usize,
    pub noprint: bool,
    pub silent_optimizer: bool,
    /// `options_mom_.optim_opt` (stream O's `optim::minimize` reads it).
    pub optim: Option<OptValue>,
    pub gstep: [f64; 2],
    /// `mode_file`: starting values (and Hessian) from a mode file (`check_mode_file.m`).
    pub mode_file: Option<String>,
    /// the raw options (MCMC after the mode: `crate::estimation::q3_mcmc`).
    pub raw: Options,
}

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        Some(OptValue::Str(s)) => s.parse().ok(),
        _ => None,
    }
}

fn flag_or(o: &Options, k: &str, default: bool) -> bool {
    match opt(o, k) {
        None => default,
        Some(OptValue::Flag) => true,
        Some(OptValue::Num(v)) => *v != 0.0,
        Some(OptValue::Str(s)) => !(s == "false" || s == "0"),
        _ => true,
    }
}

fn strings(v: &OptValue) -> Vec<String> {
    match v {
        OptValue::Str(s) => vec![s.clone()],
        OptValue::List(l) => l.iter().flat_map(strings).collect(),
        OptValue::Num(x) => vec![format!("{x}")],
        OptValue::Vec(x) => x.iter().map(|v| format!("{v}")).collect(),
        OptValue::Flag => vec![],
    }
}

const KNOWN: &[&str] = &[
    "mom_method",
    "one_sided_hp_filter",
    "bandpass_filter",
    "datafile",
    "first_obs",
    "nobs",
    "order",
    "pruning",
    "prefilter",
    "hp_filter",
    "weighting_matrix",
    "weighting_matrix_scaling_factor",
    "bartlett_kernel_lag",
    "penalized_estimator",
    "se_tolx",
    "burnin",
    "seed",
    "simulation_multiple",
    "bounded_shock_support",
    "analytic_standard_errors",
    "analytic_jacobian",
    "mode_compute",
    "additional_optimizer_steps",
    "verbose",
    "qz_criterium",
    "schur_vec_tol",
    "cova_compute",
    "prior_trunc",
    "huge_number",
    "relative_irf",
    "add_tiny_number_to_cholesky",
    "drop",
    "replic",
    "noprint",
    "silent_optimizer",
    "simulation_method",
    "irf_matching_file",
    "k_order_solver",
    "plot_priors",
    "mh_replic",
    "dirname",
    "optim",
    "mode_file",
    // display options (accepted, no effect)
    "nograph",
    "nodisplay",
    "graph_format",
    "tex",
    "console",
    "nointeractive",
    "logdata",
    "xls_sheet",
    "xls_range",
];

impl MomOptions {
    /// `default_option_mom_values.m`.
    pub fn parse(o: &Options) -> Result<Self> {
        let method = match opt(o, "mom_method")
            .map(strings)
            .and_then(|v| v.first().cloned())
            .map(|s| s.to_uppercase())
            .as_deref()
        {
            Some("GMM") => Method::Gmm,
            Some("SMM") => Method::Smm,
            Some("IRF_MATCHING") => Method::IrfMatching,
            Some(m) => bail!("method_of_moments: unknown mom_method = {m}"),
            None => bail!("method_of_moments: the mom_method option is required (GMM, SMM or IRF_MATCHING)"),
        };
        for (k, _) in o {
            if !KNOWN.contains(&k.as_str()) && !crate::estimation::q3_mcmc::OPTIONS.contains(&k.as_str()) {
                // +
                println!("WARNING: method_of_moments option `{k}` is not supported by dynare-rust and is ignored.");
            }
        }
        if opt(o, "irf_matching_file").is_some() {
            bail!(
                "method_of_moments: irf_matching_file (a MATLAB function transforming the model IRFs) is not supported by dynare-rust"
            );
        }
        if let Some(m) = opt(o, "simulation_method").map(strings).and_then(|v| v.first().cloned()) {
            if !m.eq_ignore_ascii_case("stoch_simul") {
                bail!("method_of_moments: simulation_method = {m} is not supported (only STOCH_SIMUL)");
            }
        }
        let order = num(o, "order").unwrap_or(1.0) as usize;
        if order < 1 {
            bail!("method_of_moments: The order of the Taylor approximation cannot be 0!");
        }
        let mut pruning = flag_or(o, "pruning", false);
        if method == Method::Gmm {
            if order > 1 && !pruning {
                println!("GMM at higher order only works with pruning, so we set pruning option to 1.");
                pruning = true;
            }
            if order > 3 {
                bail!(
                    "method_of_moments: Perturbation orders higher than 3 are not implemented for GMM estimation, try using SMM!"
                );
            }
        }
        let hp_filter = num(o, "hp_filter").unwrap_or(0.0);
        let one_sided_hp_filter = num(o, "one_sided_hp_filter").unwrap_or(0.0);
        let bandpass = match opt(o, "bandpass_filter") {
            None => None,
            Some(OptValue::Flag) => Some([6.0, 32.0]),
            Some(OptValue::Num(x)) if *x == 0.0 => None,
            Some(OptValue::Vec(x)) if x.len() == 2 => Some([x[0], x[1]]),
            Some(OptValue::List(x)) if x.len() == 2 => match (&x[0], &x[1]) {
                (OptValue::Num(a), OptValue::Num(b)) => Some([*a, *b]),
                _ => bail!("method_of_moments: bandpass_filter expects [HIGHEST_PERIODICITY LOWEST_PERIODICITY]"),
            },
            Some(_) => bail!("method_of_moments: bandpass_filter expects [HIGHEST_PERIODICITY LOWEST_PERIODICITY]"),
        };
        if [hp_filter != 0.0, one_sided_hp_filter != 0.0, bandpass.is_some()].iter().filter(|&&b| b).count() > 1 {
            bail!("disp_moments:: You cannot use more than one filter at the same time");
        }
        if (hp_filter != 0.0 || one_sided_hp_filter != 0.0 || bandpass.is_some()) && method != Method::Smm {
            bail!("method_of_moments: filtering of data is only supported for SMM.");
        }
        let mode_compute = match opt(o, "mode_compute") {
            None => match method {
                // Dynare: 13 (lsqnonlin) when the Optimization Toolbox is licensed, else 5
                Method::Gmm | Method::Smm => 13,
                Method::IrfMatching => 5,
            },
            Some(OptValue::Num(v)) => *v as i32,
            Some(v) => bail!("method_of_moments: mode_compute = {:?} is not supported", strings(v)),
        };
        let mut optimizer_vec = vec![mode_compute];
        match opt(o, "additional_optimizer_steps") {
            None => {}
            Some(OptValue::Num(v)) => optimizer_vec.push(*v as i32),
            Some(OptValue::Vec(v)) => optimizer_vec.extend(v.iter().map(|&x| x as i32)),
            Some(OptValue::List(l)) => {
                for it in l {
                    match it {
                        OptValue::Num(v) => optimizer_vec.push(*v as i32),
                        _ => bail!("additional_optimizer_steps: only numeric optimizers are supported"),
                    }
                }
            }
            Some(_) => bail!("additional_optimizer_steps: expected a vector of mode_compute values"),
        }
        if method == Method::IrfMatching && optimizer_vec.contains(&13) {
            bail!("method_of_moments: lsqnonlin (mode_compute=13) is not yet supported for IRF matching!");
        }
        let mut simulation_multiple = num(o, "simulation_multiple").unwrap_or(7.0);
        if method == Method::Smm && simulation_multiple < 1.0 {
            println!("The simulation horizon is shorter than the data. Dynare resets the simulation_multiple to 7.");
            simulation_multiple = 7.0;
        }
        Ok(MomOptions {
            method,
            datafile: opt(o, "datafile").map(strings).and_then(|v| v.first().cloned()),
            first_obs: num(o, "first_obs").unwrap_or(1.0) as usize,
            nobs: num(o, "nobs").map(|v| v as usize),
            order,
            pruning,
            prefilter: num(o, "prefilter").unwrap_or(0.0) != 0.0,
            hp_filter,
            one_sided_hp_filter,
            bandpass,
            weighting_matrix: opt(o, "weighting_matrix")
                .map(strings)
                .unwrap_or_else(|| vec!["DIAGONAL".into(), "OPTIMAL".into()]),
            weighting_matrix_scaling_factor: num(o, "weighting_matrix_scaling_factor").unwrap_or(1.0),
            bartlett_kernel_lag: num(o, "bartlett_kernel_lag").unwrap_or(20.0) as usize,
            penalized_estimator: flag_or(o, "penalized_estimator", false),
            se_tolx: num(o, "se_tolx").unwrap_or(1e-5),
            burnin: num(o, "burnin").unwrap_or(500.0) as usize,
            seed: num(o, "seed").unwrap_or(24051986.0) as u64,
            simulation_multiple,
            bounded_shock_support: flag_or(o, "bounded_shock_support", false),
            analytic_standard_errors: flag_or(o, "analytic_standard_errors", false),
            analytic_jacobian: flag_or(o, "analytic_jacobian", false) && method == Method::Gmm,
            optimizer_vec,
            verbose: flag_or(o, "verbose", false),
            qz_criterium: num(o, "qz_criterium").unwrap_or(1.0 - 1e-6),
            schur_vec_tol: num(o, "schur_vec_tol").unwrap_or(1e-11),
            cova_compute: num(o, "cova_compute").unwrap_or(1.0) != 0.0,
            prior_trunc: num(o, "prior_trunc").unwrap_or(1e-10),
            huge_number: num(o, "huge_number").unwrap_or(1e7),
            relative_irf: flag_or(o, "relative_irf", false),
            add_tiny_number_to_cholesky: num(o, "add_tiny_number_to_cholesky").unwrap_or(1e-14),
            drop: num(o, "drop").unwrap_or(100.0) as usize,
            replic: num(o, "replic").map(|v| v as usize).unwrap_or(if order == 1 { 1 } else { 50 }),
            noprint: flag_or(o, "noprint", false),
            silent_optimizer: flag_or(o, "silent_optimizer", false),
            optim: opt(o, "optim").cloned(),
            gstep: [1e-2, 1.0],
            mode_file: match opt(o, "mode_file") {
                Some(OptValue::Str(s)) if !s.is_empty() => Some(s.clone()),
                _ => None,
            },
            raw: o.clone(),
        })
    }
}

/// One estimation stage / optimizer iteration.
#[derive(Clone, Debug)]
pub struct StageResult {
    pub stage: usize,
    pub iteration: usize,
    pub optimizer: i32,
    pub x: Vec<f64>,
    pub fval: f64,
}

/// `oo_.mom.J_test`.
#[derive(Clone, Debug)]
pub struct JTest {
    pub j_stat: f64,
    pub degrees_freedom: usize,
    pub p_val: f64,
}

/// What `method_of_moments` leaves in `oo_.mom`.
#[derive(Clone, Debug)]
pub struct MomResults {
    pub method: Method,
    pub names: Vec<String>,
    pub x0: Vec<f64>,
    /// Objective at `x0` with the identity weighting matrix (GMM/SMM) or `W` (IRF matching).
    pub fval0: f64,
    pub data_moments: Vec<f64>,
    pub m_data: Option<Mat>,
    pub stages: Vec<StageResult>,
    pub xparam1: Vec<f64>,
    pub stdh: Vec<f64>,
    pub asympt_cov: Option<Mat>,
    /// Final weighting matrix: `Sw` (GMM/SMM) or `W` (IRF matching).
    pub weighting: Mat,
    pub woptflag: bool,
    pub q: f64,
    pub fval: f64,
    pub model_moments: Vec<f64>,
    pub j_test: Option<JTest>,
    pub labels: Vec<String>,
    /// IRF matching: Hessian at the mode and Laplace approximation (Bayesian).
    pub hessian: Option<Mat>,
    pub laplace: Option<f64>,
}

/// Prints unless `quiet`.
struct Out {
    quiet: bool,
}

impl Out {
    fn p(&self, s: impl AsRef<str>) {
        if !self.quiet {
            print!("{}", s.as_ref());
        }
    }
}

fn resolve(dir: &Path, f: &str, ext: &str) -> PathBuf {
    let p = Path::new(f);
    let p = if p.is_absolute() { p.to_path_buf() } else { dir.join(p) };
    if p.extension().is_none() && !p.exists() { p.with_extension(ext) } else { p }
}

/// Dynare's `prior_dist_names` short names.
fn prior_name(s: PriorShape) -> &'static str {
    match s.code() {
        1 => "beta",
        2 => "gamm",
        3 => "norm",
        4 => "invg",
        5 => "unif",
        6 => "invg2",
        8 => "weibl",
        _ => "",
    }
}

/// The row names and blocks of the posterior tables (`GetPosteriorParametersStatistics.m`,
/// `mcmc_diagnostics.m`).
fn table_spec(mf: &ModFile, est: &[EstimatedParam], names: &[String]) -> crate::driver::TableSpec {
    let m = &mf.model;
    let short: Vec<String> = est
        .iter()
        .enumerate()
        .map(|(k, e)| match e.kind {
            EstimatedKind::Param(_) => names[k].clone(),
            EstimatedKind::StdErr(i) => m.exo_names[i].clone(),
            EstimatedKind::Corr(a, b) => format!("{},{}", m.exo_names[a], m.exo_names[b]),
            EstimatedKind::MeStdErr(i) => m.endo_names[i].clone(),
            EstimatedKind::MeCorr(a, b) => format!("{},{}", m.endo_names[a], m.endo_names[b]),
            EstimatedKind::Skew(_) | EstimatedKind::InitState(_) => String::new(),
        })
        .collect();
    let ifac_names: Vec<String> = est
        .iter()
        .enumerate()
        .map(|(k, e)| match e.kind {
            EstimatedKind::Param(_) => names[k].clone(),
            EstimatedKind::StdErr(i) => format!("SE_{}", m.exo_names[i]),
            EstimatedKind::Corr(a, b) => format!("CC_{}_{}", m.exo_names[a], m.exo_names[b]),
            EstimatedKind::MeStdErr(i) => format!("SE_EOBS_{}", m.endo_names[i]),
            EstimatedKind::MeCorr(a, b) => format!("CC_EOBS_{}_{}", m.endo_names[a], m.endo_names[b]),
            EstimatedKind::Skew(_) | EstimatedKind::InitState(_) => String::new(),
        })
        .collect();
    let pick = |f: fn(&EstimatedKind) -> bool| -> Vec<usize> { (0..est.len()).filter(|&k| f(&est[k].kind)).collect() };
    let mut width = 0usize;
    if est.iter().any(|e| matches!(e.kind, EstimatedKind::Param(_))) {
        width = names.iter().map(|n| n.chars().count()).max().unwrap_or(0);
    }
    for (k, e) in est.iter().enumerate() {
        if !matches!(e.kind, EstimatedKind::Param(_)) {
            width = width.max(match e.kind {
                EstimatedKind::StdErr(i) => m.exo_names[i].chars().count(),
                EstimatedKind::MeStdErr(i) => m.endo_names[i].chars().count(),
                EstimatedKind::Corr(a, b) => m.exo_names[a].chars().count() + m.exo_names[b].chars().count(),
                EstimatedKind::MeCorr(a, b) => m.endo_names[a].chars().count() + m.endo_names[b].chars().count(),
                _ => short[k].chars().count(),
            });
        }
    }
    crate::driver::TableSpec {
        ifac_names,
        short,
        sections: vec![
            ("parameters", pick(|k| matches!(k, EstimatedKind::Param(_)))),
            ("standard deviation of shocks", pick(|k| matches!(k, EstimatedKind::StdErr(_)))),
            ("standard deviation of measurement errors", pick(|k| matches!(k, EstimatedKind::MeStdErr(_)))),
            ("correlation of shocks", pick(|k| matches!(k, EstimatedKind::Corr(..)))),
            ("correlation of measurement errors", pick(|k| matches!(k, EstimatedKind::MeCorr(..)))),
        ],
        width,
    }
}

/// The estimation table of `display_estimation_results_table.m`.
fn results_table(
    mf: &ModFile,
    est: &[EstimatedParam],
    names: &[String],
    x: &[f64],
    sd: &[f64],
    title: &str,
    posterior: Option<&[Prior]>,
) -> String {
    let m = &mf.model;
    let mut w = 0usize;
    if est.iter().any(|e| matches!(e.kind, EstimatedKind::Param(_))) {
        w = names.iter().map(|n| n.chars().count()).max().unwrap_or(0);
    }
    for e in est {
        match e.kind {
            EstimatedKind::StdErr(k) => w = w.max(m.exo_names[k].len()),
            EstimatedKind::Corr(a, b) => w = w.max(m.exo_names[a].len() + m.exo_names[b].len()),
            _ => {}
        }
    }
    for e in est {
        match e.kind {
            EstimatedKind::MeStdErr(k) => w = w.max(m.endo_names[k].len()),
            EstimatedKind::MeCorr(a, b) => w = w.max(m.endo_names[a].len() + m.endo_names[b].len()),
            _ => {}
        }
    }
    let tit1 = if posterior.is_some() {
        format!("{:<w$} {:>10} {:>8} {:>7} {:>6} {:>6}\n", " ", "prior mean", "mode", "s.d.", "prior", "pstdev")
    } else {
        format!("{:<w$} {:>10} {:>7} {:>6}\n", " ", "Estimate", "s.d.", "t-stat")
    };
    let mut out = format!("\nRESULTS FROM {} ESTIMATION\n", title.to_uppercase());
    for (section, pick) in [
        ("parameters", 2),
        ("standard deviation of shocks", 0),
        ("standard deviation of measurement errors", 3),
        ("correlation of shocks", 1),
        ("correlation of measurement errors", 4),
    ] {
        let idx: Vec<usize> = (0..est.len())
            .filter(|&k| match est[k].kind {
                EstimatedKind::Param(_) => pick == 2,
                EstimatedKind::StdErr(_) => pick == 0,
                EstimatedKind::Corr(..) => pick == 1,
                EstimatedKind::MeStdErr(_) => pick == 3,
                EstimatedKind::MeCorr(..) => pick == 4,
                EstimatedKind::Skew(_) | EstimatedKind::InitState(_) => unreachable!("rejected by run()"),
            })
            .collect();
        if idx.is_empty() {
            continue;
        }
        out += &format!("{section}\n{tit1}\n");
        for k in idx {
            let name = match est[k].kind {
                EstimatedKind::Param(_) => names[k].clone(),
                EstimatedKind::StdErr(i) => m.exo_names[i].clone(),
                EstimatedKind::Corr(a, b) => format!("{},{}", m.exo_names[a], m.exo_names[b]),
                EstimatedKind::MeStdErr(k) => m.endo_names[k].clone(),
                EstimatedKind::MeCorr(a, b) => format!("{},{}", m.endo_names[a], m.endo_names[b]),
                EstimatedKind::Skew(_) | EstimatedKind::InitState(_) => unreachable!("rejected by run()"),
            };
            match posterior {
                Some(pr) => {
                    out += &format!(
                        "{name:<w$} {} {} {} {:>6} {} \n",
                        fmt_f(pr[k].mean, 10, 4),
                        fmt_f(x[k], 8, 4),
                        fmt_f(sd[k], 7, 4),
                        prior_name(pr[k].shape),
                        fmt_f(pr[k].sd, 6, 4)
                    )
                }
                None => {
                    out += &format!(
                        "{name:<w$} {} {} {} \n",
                        fmt_f(x[k], 10, 4),
                        fmt_f(sd[k], 7, 4),
                        fmt_f(x[k].abs() / sd[k], 7, 4)
                    )
                }
            }
        }
        out.push('\n');
    }
    out
}

/// Labels of `display_comparison_moments_irfs.m` for the moments.
fn moment_labels(mf: &ModFile, rows: &[super::MatchedMoment]) -> Vec<String> {
    rows.iter()
        .map(|m| {
            let mut s = "E[".to_string();
            for j in 0..m.vars.len() {
                s += &mf.model.endo_names[m.vars[j]];
                if m.lags[j] != 0 {
                    s += &format!("({})", m.lags[j]);
                }
                if m.powers[j] > 1 {
                    s += &format!("^{}", m.powers[j]);
                }
                s += if j + 1 == m.vars.len() { "]" } else { "*" };
            }
            s
        })
        .collect()
}

/// Upper Cholesky factor or Dynare's error.
fn chol_or_err(w: &Mat) -> Result<Mat> {
    chol_upper(w).ok_or_else(|| anyhow!("method_of_moments: Specified 'weighting_matrix' is not positive definite. Check whether your model implies stochastic singularity!"))
}

/// Runs one optimizer of `optimizer_vec` (`dynare_minimize_objective.m`).
fn optimize(
    code: i32,
    p: &MomProblem,
    data: &[f64],
    w: &Mat,
    x0: &[f64],
    o: &MomOptions,
    quiet: bool,
    hess0: Option<&Mat>,
) -> Result<(Vec<f64>, f64, Option<Mat>)> {
    let analytic = o.analytic_jacobian && [1, 3, 4, 13, 101].contains(&code);
    let obj = MomObjective { p, data, w, analytic };
    let verbose = !quiet && !o.silent_optimizer;
    match code {
        0 => {
            let f = obj.eval(x0).0;
            Ok((x0.to_vec(), f, hess0.cloned()))
        }
        13 => {
            let f = |x: &[f64]| p.residual_vector(x, data, w, o.huge_number);
            let jac = |x: &[f64]| {
                p.residual_jacobian(x, w).map(|j| {
                    (0..j.ncols()).map(|c| j.column(c).iter().copied().collect::<Vec<f64>>()).collect::<Vec<_>>()
                })
            };
            let jac_opt: Option<&(dyn Fn(&[f64]) -> Option<Vec<Vec<f64>>> + Sync)> =
                if analytic { Some(&jac) } else { None };
            let r = super::lsq::lsq_bounded(
                &f,
                jac_opt,
                x0,
                &p.bounds.lb,
                &p.bounds.ub,
                &super::lsq::LsqOptions { verbose, ..Default::default() },
            );
            if verbose {
                println!(
                    "\nmode_compute = 13 (dynare-rust bounded Levenberg-Marquardt substitute for lsqnonlin): {}",
                    r.message
                );
            }
            Ok((r.x, r.fval, None))
        }
        // [stream O] every other mode_compute through dynare_minimize_objective.m's dispatcher
        6 => bail!("method_of_moments: mode_compute = 6 (gmhmaxlik) is not available for method_of_moments"),
        k => {
            let p2: Vec<f64> = match (&p.penalty, &p.priors) {
                (Some((_, s)), _) => s.clone(),
                (None, Some(pr)) => pr.iter().map(|q| q.sd).collect(),
                _ => vec![f64::NAN; x0.len()],
            };
            let ctx = crate::estimation::optim::MinimizeContext {
                bounds: &p.bounds,
                names: &p.names,
                prior_sd: &p2,
                optim: o.optim.as_ref(),
                verbose,
                seed: o.seed,
                stream: u64::MAX - 100 - k as u64,
            };
            let r = crate::estimation::optim::minimize(k, &obj, x0, &ctx)?;
            Ok((r.x, r.fval, r.hessian))
        }
    }
}

fn optimizer_name(k: i32) -> String {
    match k {
        0 => "no minimization".into(),
        1 => "fmincon".into(),
        2 => "continuous simulated annealing".into(),
        3 => "fminunc".into(),
        4 => "csminwel".into(),
        5 => "newrat".into(),
        6 => "gmhmaxlik".into(),
        7 => "fminsearch".into(),
        8 => "Dynare Nelder-Mead simplex".into(),
        9 => "CMA-ES".into(),
        10 => "simpsa".into(),
        12 => "particleswarm".into(),
        13 => "lsqnonlin".into(),
        101 => "SolveOpt".into(),
        102 => "simulannealbnd".into(),
        _ => "user-defined".into(),
    }
}

/// Everything `mom.run` sets up before the first objective evaluation.
pub struct Setup {
    pub o: MomOptions,
    pub problem: MomProblem,
    pub est: Vec<EstimatedParam>,
    pub names: Vec<String>,
    pub x0: Vec<f64>,
    pub data_moments: Vec<f64>,
    pub m_data: Option<Mat>,
    pub nobs: usize,
    /// SMM `variance_correction_factor` (1 otherwise).
    pub vcf: f64,
    pub irf_targets: Option<super::irf_matching::IrfTargets>,
    pub mom_nbr: usize,
    pub bayesian: bool,
    pub penalized: bool,
    pub priors: Option<Vec<Prior>>,
    /// `bayestopt_` of the MCMC: the priors, Laplace-approximated (normal) for GMM/SMM
    /// (`transform_prior_to_laplace_prior.m`).
    pub mcmc_priors: Option<Vec<Prior>>,
    pub rows: Vec<super::MatchedMoment>,
    /// `hessian_xparam0` from the `mode_file`.
    pub hess0: Option<Mat>,
}

/// `method_of_moments` (`mom.run`).
pub fn method_of_moments(
    mf: &ModFile,
    blocks: &Blocks,
    opts: &Options,
    ctx: &Context,
    inputs: &MomInputs,
) -> Result<MomResults> {
    let Setup {
        o,
        problem,
        est,
        names,
        x0,
        data_moments,
        m_data,
        nobs,
        vcf,
        irf_targets,
        mom_nbr,
        bayesian,
        penalized,
        priors,
        mcmc_priors,
        rows,
        hess0,
    } = setup(mf, blocks, opts, ctx, inputs)?;
    let out = Out { quiet: inputs.quiet || o.noprint };
    estimate(
        mf,
        ctx,
        &out,
        o,
        problem,
        est,
        names,
        x0,
        data_moments,
        m_data,
        nobs,
        vcf,
        irf_targets,
        mom_nbr,
        bayesian,
        penalized,
        priors,
        mcmc_priors,
        rows,
        hess0,
    )
}

/// The setup part of `mom.run` (up to the checks of the objective at the initial values).
pub fn setup(mf: &ModFile, blocks: &Blocks, opts: &Options, ctx: &Context, inputs: &MomInputs) -> Result<Setup> {
    let o = MomOptions::parse(opts)?;
    let out = Out { quiet: inputs.quiet || o.noprint };
    let model = &mf.model;
    out.p(format!("\n==== Method of Moments Estimation ({}) ====\n\n", o.method.name()));

    // ---- checks on the blocks ------------------------------------------------------------
    let est = mf.estimated_params_dynare_order();
    if est.iter().any(|e| matches!(e.kind, EstimatedKind::Skew(_))) {
        bail!("method_of_moments: skew parameters in estimated_params are not supported by dynare-rust");
    }
    // SMM and IRF matching take estimated measurement errors (M_.H through set_all_parameters)
    if o.method == Method::Gmm
        && est.iter().any(|e| matches!(e.kind, EstimatedKind::MeStdErr(_) | EstimatedKind::MeCorr(..)))
    {
        bail!(
            "method_of_moments: GMM estimation does not support measurement error(s) yet. Please specify them as a structural shock!"
        );
    }
    if est.is_empty() {
        bail!("method_of_moments: You need to provide an 'estimated_params' block!");
    }
    match o.method {
        Method::Gmm | Method::Smm if blocks.matched_moments.is_empty() => {
            bail!(
                "method_of_moments: You need to provide a 'matched_moments' block for 'mom_method={}'!",
                o.method.name()
            )
        }
        Method::IrfMatching if blocks.matched_irfs.is_empty() => {
            bail!("method_of_moments: You need to provide a 'matched_irfs' block for 'mom_method=IRF_MATCHING'!")
        }
        _ => {}
    }
    let n_prior = est.iter().filter(|e| e.prior.is_some()).count();
    if n_prior != 0 && n_prior != est.len() {
        bail!(
            "method_of_moments: Estimation must be either fully Frequentist or fully Bayesian. Maybe you forgot to specify a prior distribution!"
        );
    }
    let bayesian = n_prior > 0;
    if mf.varobs.is_empty() {
        bail!("method_of_moments: VAROBS statement is missing!");
    }
    let varobs: Vec<usize> = mf
        .varobs
        .iter()
        .map(|v| model.endo_index(v).ok_or_else(|| anyhow!("varobs {v} is not endogenous")))
        .collect::<Result<_>>()?;
    let names: Vec<String> = est.iter().map(|e| mf.estimated_param_name(e)).collect();
    let n = est.len();

    // ---- matched moments --------------------------------------------------------------------
    let mut rows = Vec::new();
    if o.method != Method::IrfMatching {
        let (r, removed) = matched_moments_block(&blocks.matched_moments, o.method)?;
        if !removed.is_empty() {
            let s: Vec<String> = removed.iter().map(|i| i.to_string()).collect();
            out.p(format!(
                "Duplicate declared moments found and removed in 'matched_moments' block in rows:\n {}.\n",
                s.join("  ")
            ));
            out.p("Dynare will continue with remaining moment conditions\n");
        }
        for m in &r {
            if let Some(&v) = m.vars.iter().find(|v| !varobs.contains(v)) {
                bail!(
                    "method_of_moments: You specified moments involving {}, but it is not a varobs!",
                    model.endo_names[v]
                );
            }
        }
        rows = r;
    }
    let ar = rows.iter().map(|m| m.lags.iter().max().unwrap() - m.lags.iter().min().unwrap()).max().unwrap_or(0).max(0)
        as usize;
    let irf_targets = if o.method == Method::IrfMatching {
        Some(super::irf_matching::matched_irfs_blocks(
            &blocks.matched_irfs,
            &blocks.matched_irfs_weights,
            &varobs,
            model,
        )?)
    } else {
        None
    };
    let mom_nbr = irf_targets.as_ref().map_or(rows.len(), |t| t.data.len());
    if mom_nbr < n {
        bail!(
            "method_of_moments: There must be at least as many moments as parameters for a {} estimation!",
            o.method.name()
        );
    }
    out.p("\n\n");

    // ---- priors, penalty, bounds (set_prior.m, prior_bounds.m) ------------------------------
    let priors: Option<Vec<Prior>> = if bayesian {
        Some(
            est.iter()
                .zip(&names)
                .map(|(e, name)| {
                    let shape = PriorShape::from_code(e.prior.unwrap().dynare_code()).unwrap();
                    Prior::new(shape, e.mean, e.sd, e.p3, e.p4).map_err(|err| anyhow!("{name}: {err}"))
                })
                .collect::<Result<_>>()?,
        )
    } else {
        None
    };
    let mut penalized = o.penalized_estimator && o.method != Method::IrfMatching;
    let mut bound_priors = priors.clone();
    if o.method != Method::IrfMatching {
        if let Some(pr) = &priors {
            if !penalized {
                out.p(format!(
                    "\nPriors were specified, but the penalized_estimator-option was not set.\nDynare sets penalized_estimator to 1. Conducting {} with penalty.\n",
                    o.method.name()
                ));
                penalized = true;
            }
            // transform_prior_to_laplace_prior.m
            if pr.iter().any(|p| p.shape != PriorShape::Normal) {
                out.p("\nNon-normal priors specified. Penalized estimation uses a Laplace type of approximation:");
                out.p("\nOnly the prior mean and standard deviation are relevant, all other shape information, except for the parameter bounds, is ignored.\n\n");
            }
            bound_priors = Some(
                pr.iter()
                    .map(|p| {
                        if p.shape == PriorShape::Normal {
                            Ok(p.clone())
                        } else {
                            Prior::new(PriorShape::Normal, p.mean, p.sd, None, None)
                        }
                    })
                    .collect::<Result<_>>()?,
            );
        } else if penalized {
            out.p("Penalized estimation turned off as you did not declare priors\n");
            penalized = false;
        }
    }
    if let Some(pr) = &bound_priors {
        // +mom/run.m: check_prior_stderr_corr_skew
        let kinds: Vec<_> = est.iter().map(|e| e.kind).collect();
        for w in crate::estimation::prior::check_prior_stderr_corr_skew(&kinds, pr) {
            out.p(format!("Warning: {w}\n"));
        }
    }
    let explicit: Vec<(f64, f64)> = est
        .iter()
        .map(|e| match e.kind {
            EstimatedKind::Corr(..) | EstimatedKind::MeCorr(..) => {
                (e.lb.unwrap_or(-1.0).clamp(-1.0, 1.0), e.ub.unwrap_or(1.0).clamp(-1.0, 1.0))
            }
            _ => (e.lb.unwrap_or(f64::NEG_INFINITY), e.ub.unwrap_or(f64::INFINITY)),
        })
        .collect();
    let mut bounds = match &bound_priors {
        Some(pr) => crate::estimation::prior_bounds(pr, o.prior_trunc),
        None => Bounds::unbounded(n),
    };
    for i in 0..n {
        bounds.lb[i] = bounds.lb[i].max(explicit[i].0);
        bounds.ub[i] = bounds.ub[i].min(explicit[i].1);
    }
    // set_correct_bounds_for_stderr_corr.m (its strict `<` leaves the last corr out)
    let nvx = est.iter().filter(|e| matches!(e.kind, EstimatedKind::StdErr(_) | EstimatedKind::MeStdErr(_))).count();
    let ncx = est.iter().filter(|e| matches!(e.kind, EstimatedKind::Corr(..) | EstimatedKind::MeCorr(..))).count();
    for i in 0..n {
        let p1 = i + 1;
        if p1 <= nvx && bounds.lb[i] < 0.0 {
            bounds.lb[i] = 0.0;
        }
        if p1 > nvx && p1 < nvx + ncx {
            bounds.lb[i] = bounds.lb[i].max(-1.0);
            bounds.ub[i] = bounds.ub[i].min(1.0);
        }
    }

    // ---- initial values ----------------------------------------------------------------------
    let me_rows: Vec<Option<(usize, usize)>> = est
        .iter()
        .map(|e| match e.kind {
            EstimatedKind::MeStdErr(k) => mf.varobs_position(k).map(|i| (i, i)),
            EstimatedKind::MeCorr(a, b) => mf.varobs_position(a).zip(mf.varobs_position(b)),
            _ => None,
        })
        .collect();
    if let Some(k) = (0..n).find(|&k| {
        matches!(est[k].kind, EstimatedKind::MeStdErr(_) | EstimatedKind::MeCorr(..)) && me_rows[k].is_none()
    }) {
        bail!("set_prior: {} has to be declared as observable since you assume a measurement error on it.", names[k]);
    }
    let h = &mf.h;
    let calib = |k: usize| match (est[k].kind, me_rows[k]) {
        (EstimatedKind::Param(i), _) => model.params[i],
        (EstimatedKind::StdErr(i), _) => mf.sigma_e[(i, i)].sqrt(),
        (EstimatedKind::Corr(i, j), _) => mf.sigma_e[(i, j)] / (mf.sigma_e[(i, i)] * mf.sigma_e[(j, j)]).sqrt(),
        (EstimatedKind::MeStdErr(_), Some((i, _))) => h[(i, i)].sqrt(),
        (EstimatedKind::MeCorr(..), Some((i, j))) => h[(i, j)] / (h[(i, i)] * h[(j, j)]).sqrt(),
        _ => unreachable!("rejected by run()"),
    };
    // set_prior.m: the init column, NaN replaced by the prior mean (bayestopt_.p1)
    let mut x0: Vec<f64> = est
        .iter()
        .enumerate()
        .map(|(k, e)| {
            let init = mf.estimated_params_init.iter().find(|i| i.kind == e.kind).map(|i| i.value).or(e.init);
            match (init, &priors) {
                (Some(v), _) => v,
                (None, Some(pr)) => pr[k].mean,
                (None, None) => f64::NAN,
            }
        })
        .collect();
    // check_mode_file.m (unless the mode comes from the MCMC: mh_posterior_mode_estimation)
    let mut hess0 = None;
    if let Some(f) = &o.mode_file {
        let mh_mode = match opt(opts, "mh_posterior_mode_estimation") {
            Some(OptValue::Flag) => true,
            Some(OptValue::Num(v)) => *v != 0.0,
            _ => false,
        };
        if !bayesian || !mh_mode {
            let x_init = x0.clone();
            let (xm, hh) = crate::driver::read_mode_file(&ctx.dir, f, &names, o.optimizer_vec[0], || Ok(x_init))?;
            x0 = xm;
            hess0 = hh;
        }
    }
    if mf.estimated_params_init_use_calibration {
        for (k, e) in est.iter().enumerate() {
            if !mf.estimated_params_init.iter().any(|i| i.kind == e.kind) {
                x0[k] = calib(k);
            }
        }
    }
    if !bayesian && x0.iter().any(|v| v.is_nan()) {
        bail!(
            "method_of_moments: Frequentist {} requires all estimated parameters to be initialized, either in an estimated_params or estimated_params_init-block!",
            o.method.name()
        );
    }
    for i in 0..n {
        if x0[i] < bounds.lb[i] || x0[i] > bounds.ub[i] {
            bail!(
                "method_of_moments: the initial value of {} ({}) violates its bounds [{}, {}]",
                names[i],
                x0[i],
                bounds.lb[i],
                bounds.ub[i]
            );
        }
    }

    // ---- steady state at the initial parameters ------------------------------------------------
    let steady_opts = crate::dsge::steady_options(mf);
    let ys_calib = crate::analysis::calibrated_steady_state(mf, &model.params)?;
    let mut problem = MomProblem::new(
        o.method,
        model,
        est.clone(),
        names.clone(),
        &mf.sigma_e,
        varobs.clone(),
        steady_opts,
        ys_calib.clone(),
        o.order,
        o.pruning,
        o.qz_criterium,
    );
    problem.ssm = crate::deterministic::SteadyStateModel::from_modfile(mf)?; // stream D
    problem.exo_ss = mf.exo_steady_state();
    // M_.H, M_.Correlation_matrix_ME and check_for_calibrated_covariances.m's
    // calibrated_covariances_ME (the off-diagonal entries of M_.H that are not estimated)
    problem.base_h = mf.h.clone();
    problem.base_corr_h = mf.corr_h.clone();
    problem.me_rows = me_rows.clone();
    let est_corr_h: Vec<(usize, usize)> =
        (0..n).filter(|&k| matches!(est[k].kind, EstimatedKind::MeCorr(..))).filter_map(|k| me_rows[k]).collect();
    for j in 0..mf.h.ncols() {
        for i in j + 1..mf.h.nrows() {
            if mf.h[(i, j)] != 0.0 && !est_corr_h.iter().any(|&(a, b)| (a, b) == (i, j) || (b, a) == (i, j)) {
                problem.cal_cov_h.push((i, j, mf.h[(i, j)]));
            }
        }
    }
    {
        let (params, _) = problem.set_all_parameters(&x0);
        let ys1 = crate::dsge::solve_steady_at(
            model,
            problem.ssm.as_ref(),
            &params,
            &ys_calib,
            &problem.exo_ss,
            &steady_opts,
            true,
        )
        .map(|(ys, _)| ys)
        .map_err(|e| {
            anyhow!(
                "method_of_moments: The steady state at the initial parameters cannot be computed (info {}).",
                e.info
            )
        })?;
        problem.ys_guess = crate::dsge::solve_steady_at(
            model,
            problem.ssm.as_ref(),
            &params,
            &ys1,
            &problem.exo_ss,
            &steady_opts,
            true,
        )
        .map(|(ys, _)| ys)
        .unwrap_or(ys1);
    }
    problem.bounds = bounds.clone();
    problem.rows = rows.clone();
    problem.prefilter = o.prefilter;
    problem.hp_filter = o.hp_filter;
    problem.one_sided_hp_filter = o.one_sided_hp_filter;
    problem.bandpass = o.bandpass;
    problem.nlags = ar;
    problem.schur_vec_tol = o.schur_vec_tol;
    problem.scaling = o.weighting_matrix_scaling_factor;
    if penalized {
        let pr = bound_priors.as_ref().unwrap();
        problem.penalty = Some((pr.iter().map(|p| p.mean).collect(), pr.iter().map(|p| p.sd).collect()));
    }
    if o.method == Method::IrfMatching {
        problem.priors = priors.clone();
    }

    // ---- data ------------------------------------------------------------------------------------
    let mut data_moments = Vec::new();
    let mut m_data = None;
    let mut nobs = 0;
    if o.method != Method::IrfMatching {
        // makedataset.m: the datafile option, or options_.dataset of a preceding `data` command
        // (options_mom_.dataset = options_.dataset)
        let sample = match (&o.datafile, crate::dataset::current_data_command()) {
            (Some(_), Some(_)) => bail!(
                "makedataset: You cannot simultaneously use the data command and the datafile option (in the estimation command)!"
            ),
            (Some(df), None) => {
                let path = resolve(&ctx.dir, df, "csv");
                if path.file_stem().and_then(|s| s.to_str()) == Some(ctx.fname.as_str()) {
                    bail!(
                        "method_of_moments: 'datafile' and mod file are not allowed to have the same name; change the name of the 'datafile'!"
                    );
                }
                let table = crate::data::Table::read(&path, &Default::default())
                    .with_context(|| format!("reading {}", path.display()))?;
                table.sample(&mf.varobs, o.first_obs, o.nobs)?
            }
            (None, Some(_)) => crate::dataset::Dataset::load(mf, &ctx.dir, opts)?.sample(&mf.varobs)?,
            (None, None) => bail!("makedataset: datafile option is missing!"),
        };
        nobs = sample.nrows();
        if ar > nobs + 1 {
            bail!("method_of_moments: Dataset is too short to compute higher than first moments!");
        }
        if sample.iter().any(|v| v.is_nan())
            && (o.hp_filter != 0.0 || o.one_sided_hp_filter != 0.0 || o.bandpass.is_some())
        {
            bail!("method_of_moments: missing data is incompatible with filtering data!");
        }
        out.p("Computing data moments. Note that NaN values in the moments (due to leads and lags or missing data) are replaced by the mean of the corresponding moment.\n");
        let (dm, md) = problem.series_moments(&sample, o.prefilter);
        data_moments = dm;
        m_data = Some(md);
    }

    // ---- SMM shocks / IRF draws ------------------------------------------------------------------

    let mut vcf = 1.0;
    if o.method == Method::Smm {
        let long = (o.simulation_multiple * nobs as f64).round() as usize;
        vcf = 1.0 + 1.0 / o.simulation_multiple;
        let ne = model.n_exo();
        let mut rng = ChaCha8Rng::seed_from_u64(ctx.seed_override.unwrap_or(o.seed));
        let mut z = match &inputs.shock_series {
            Some(z) => {
                if z.nrows() != long + o.burnin || z.ncols() != ne {
                    bail!(
                        "method_of_moments: the injected shock series is {}x{}, expected {}x{}",
                        z.nrows(),
                        z.ncols(),
                        long + o.burnin,
                        ne
                    );
                }
                z.clone()
            }
            None => Mat::from_fn(long + o.burnin, ne, |_, _| 0.0).map(|_| StandardNormal.sample(&mut rng)),
        };
        // temp_shocks_ME = randn(smmstream, long, length(M_.H)), after temp_shocks
        let nh = mf.varobs.len();
        let mut z_me = match &inputs.me_shock_series {
            Some(z) => {
                if z.nrows() != long || z.ncols() != nh {
                    bail!(
                        "method_of_moments: the injected measurement-error shock series is {}x{}, expected {long}x{nh}",
                        z.nrows(),
                        z.ncols()
                    );
                }
                z.clone()
            }
            None => Mat::from_fn(long, nh, |_, _| 0.0).map(|_| StandardNormal.sample(&mut rng)),
        };
        if o.bounded_shock_support {
            z.apply(|v| *v = v.clamp(-2.0, 2.0));
            // run.m truncates the measurement-error draws at -2 only (sic)
            z_me.apply(|v| *v = v.max(-2.0));
        }
        problem.smm = Some(SmmSetup { shock_series: z, me_shock_series: z_me, long });
    }
    if let Some(t) = &irf_targets {
        let winv = solve(&t.w, &Mat::identity(t.w.nrows(), t.w.nrows())).ok_or_else(|| {
            anyhow!("method_of_moments: Something wrong while computing inv(W), check your weighting matrix!")
        })?;
        let r = chol_upper(&winv).ok_or_else(|| {
            anyhow!("method_of_moments: Something wrong while computing log(det(inv(W))), check your weighting matrix!")
        })?;
        let winv_logdet = 2.0 * (0..r.nrows()).map(|i| r[(i, i)].ln()).sum::<f64>();
        let draws = if o.order > 1 {
            let mut rng = ChaCha8Rng::seed_from_u64(ctx.seed_override.unwrap_or(ctx.stream_seed));
            let ne = model.n_exo();
            crate::solver::korder::simulate::draw_shocks(&Mat::identity(ne, ne), t.horizon, o.drop, o.replic, &mut rng)
        } else {
            Vec::new()
        };
        data_moments = t.data.clone();
        problem.irf = Some(IrfSetup {
            targets: t.clone(),
            winv_logdet,
            relative_irf: o.relative_irf,
            add_tiny_number_to_cholesky: o.add_tiny_number_to_cholesky,
            drop: o.drop,
            draws,
        });
    }

    let mcmc_priors = if o.method == Method::IrfMatching { priors.clone() } else { bound_priors };
    Ok(Setup {
        o,
        problem,
        est,
        names,
        x0,
        data_moments,
        m_data,
        nobs,
        vcf,
        irf_targets,
        mom_nbr,
        bayesian,
        penalized,
        mcmc_priors,
        priors,
        rows,
        hess0,
    })
}

#[allow(clippy::too_many_arguments)]
fn estimate(
    mf: &ModFile,
    ctx: &Context,
    out: &Out,
    o: MomOptions,
    problem: MomProblem,
    est: Vec<EstimatedParam>,
    names: Vec<String>,
    x0: Vec<f64>,
    data_moments: Vec<f64>,
    m_data: Option<Mat>,
    nobs: usize,
    vcf: f64,
    irf_targets: Option<super::irf_matching::IrfTargets>,
    mom_nbr: usize,
    bayesian: bool,
    penalized: bool,
    priors: Option<Vec<Prior>>,
    mcmc_priors: Option<Vec<Prior>>,
    rows: Vec<super::MatchedMoment>,
    hess0: Option<Mat>,
) -> Result<MomResults> {
    let n = est.len();
    // ---- objective at the initial values ---------------------------------------------------------
    let identity = Mat::identity(mom_nbr, mom_nbr);
    let w_init = irf_targets.as_ref().map_or(identity.clone(), |t| t.w.clone());
    let t0 = Instant::now();
    let e0 = problem.evaluate(&x0, &data_moments, &w_init);
    let elapsed = t0.elapsed().as_secs_f64();
    let fval0 = match &e0 {
        Ok(e) => e.fval,
        Err(f) => {
            out.p("method_of_moments: Error in computing the objective function for initial parameter values\n");
            return Err(fail_err(*f, "the objective function cannot be computed at the initial parameter values"));
        }
    };
    out.p("Initial value of the moment objective function");
    if o.method != Method::IrfMatching {
        out.p(format!(" with {:4.1} times identity weighting matrix", o.weighting_matrix_scaling_factor));
    }
    out.p(format!(": {fval0:6.4} \n\n"));
    out.p(format!("Time required to compute objective function once: {elapsed:5.4} seconds \n"));

    // ---- print_info_on_estimation_settings.m ------------------------------------------------------
    out.p("\n---------------------------------------------------\n");
    match o.method {
        Method::Smm => out.p("Simulated method of moments with"),
        Method::Gmm => out.p("General method of moments with"),
        Method::IrfMatching => out.p(if bayesian {
            "Bayesian Impulse Response Function Matching with"
        } else {
            "Frequentist Impulse Response Function Matching with"
        }),
    }
    if o.method != Method::IrfMatching {
        out.p(if o.prefilter {
            "\n  - centered moments (prefilter=1)"
        } else {
            "\n  - uncentered moments (prefilter=0)"
        });
        if o.hp_filter != 0.0 {
            out.p(format!("\n  - HP filter with lambda={:4.1}", o.hp_filter));
        } else if o.one_sided_hp_filter != 0.0 {
            out.p(format!("\n  - One-sided HP filter with lambda={:4.1}", o.one_sided_hp_filter));
        } else if let Some([h, l]) = o.bandpass {
            out.p(format!("\n  - Bandpass filter with passband [{h:4.1},{l:4.1}]"));
        }
        if penalized {
            out.p("\n  - penalized estimation using deviation from prior mean and weighted with prior precision");
        }
    }
    for (i, &k) in o.optimizer_vec.iter().enumerate() {
        let s = if i == 0 { "- optimizer (mode_compute" } else { "            (additional_optimizer_steps" };
        out.p(format!("\n  {s}={k}): {}", optimizer_name(k)));
        if o.silent_optimizer {
            out.p(" (silent)");
        }
    }
    out.p(format!("\n  - stochastic simulations with perturbation order: {}", o.order));
    if o.order > 1 && o.pruning {
        out.p(" (with pruning)");
    }
    if o.method != Method::IrfMatching {
        out.p(if o.method == Method::Gmm && o.analytic_standard_errors {
            "\n  - standard errors: analytic derivatives"
        } else {
            "\n  - standard errors: numerical derivatives"
        });
        out.p(format!("\n  - number of matched moments: {mom_nbr}"));
    } else {
        out.p(format!("\n  - number of matched IRFs: {mom_nbr}"));
    }
    out.p(format!("\n  - number of parameters: {n}\n\n"));

    let labels: Vec<String> = match &irf_targets {
        Some(t) => t.labels.clone(),
        None => moment_labels(mf, &rows),
    };
    let field = o.method.name().to_lowercase();
    let mut stages = Vec::new();
    let mut xparam1 = x0.clone();
    let mut res = MomResults {
        method: o.method,
        names: names.clone(),
        x0: x0.clone(),
        fval0,
        data_moments: data_moments.clone(),
        m_data: m_data.clone(),
        stages: Vec::new(),
        xparam1: Vec::new(),
        stdh: vec![f64::NAN; n],
        asympt_cov: None,
        weighting: identity.clone(),
        woptflag: false,
        q: f64::NAN,
        fval: f64::NAN,
        model_moments: Vec::new(),
        j_test: None,
        labels: labels.clone(),
        hessian: None,
        laplace: None,
    };
    // check_posterior_sampler_options.m: the slice sampler without slice_initialize_with_mode
    // sets mode_compute = 0, cova_compute = 0 and mh_posterior_mode_estimation; for IRF matching
    // mom.run then skips the mode and starts the MCMC at the prior mode (prior mean where
    // undefined or outside the bounds), invhess = diag(prior variances)
    let slice_no_mode = bayesian
        && crate::modfile::opt(&o.raw, "mh_replic").is_some_and(|v| matches!(v, OptValue::Num(r) if *r > 0.0))
        && crate::estimation::samplers::SamplerOptions::from_options(
            opt(&o.raw, "posterior_sampling_method"),
            opt(&o.raw, "posterior_sampler_options"),
        )?
        .mh_posterior_mode_estimation();
    // GMM/SMM: hessian_xparam1 = inv(invhess) of standard_errors.m (the MCMC's jumping covariance)
    let mut hessian_gmm_smm: Option<Mat> = None;

    if o.method != Method::IrfMatching {
        // ---- mode_compute_gmm_smm.m ----------------------------------------------------------------
        let md = m_data.as_ref().unwrap();
        let wms: Vec<String> = o.weighting_matrix.iter().map(|s| s.to_lowercase()).collect();
        if wms.len() > 1 && !(wms.iter().any(|s| s == "diagonal") || wms.iter().any(|s| s == "optimal")) {
            out.p("\nYou did not specify the use of an optimal or diagonal weighting matrix. There is no point in running an iterated method of moments.\n");
        }
        let mut model_moments: Vec<f64> = Vec::new();
        let mut sw = identity.clone();
        let mut woptflag = false;
        for (stage, wm) in o.weighting_matrix.iter().enumerate() {
            let stage = stage + 1;
            out.p(format!("Estimation stage {stage}\n"));
            woptflag = false;
            let target = if stage == 1 { &data_moments } else { &model_moments };
            let w = match wm.to_lowercase().as_str() {
                "identity_matrix" => {
                    out.p("  - identity weighting matrix\n");
                    identity.clone()
                }
                "diagonal" => {
                    out.p(format!(
                        "  - diagonal of optimal weighting matrix (Bartlett kernel with {} lags)\n",
                        o.bartlett_kernel_lag
                    ));
                    out.p(if stage == 1 {
                        "    and using data-moments as initial estimate of model-moments\n"
                    } else {
                        "    and using previous stage estimate of model-moments\n"
                    });
                    let w = optimal_weighting_matrix(md, target, o.bartlett_kernel_lag)?;
                    Mat::from_diagonal(&w.diagonal())
                }
                "optimal" => {
                    out.p(format!(
                        "  - optimal weighting matrix (Bartlett kernel with {} lags)\n",
                        o.bartlett_kernel_lag
                    ));
                    out.p(if stage == 1 {
                        "    and using data-moments as initial estimate of model-moments\n"
                    } else {
                        "    and using previous stage estimate of model-moments\n"
                    });
                    woptflag = stage > 1;
                    optimal_weighting_matrix(md, target, o.bartlett_kernel_lag)?
                }
                _ => {
                    out.p("  - user-specified weighting matrix\n");
                    let path = resolve(&ctx.dir, wm, "mat");
                    let vars = crate::matfile::read(&path).with_context(|| format!("reading {}", path.display()))?;
                    let w = vars.get("weighting_matrix").and_then(|v| v.as_matrix()).ok_or_else(|| {
                        anyhow!(
                            "method_of_moments: No matrix named 'weighting_matrix' could be found in {}!",
                            path.display()
                        )
                    })?;
                    if w.nrows() != w.ncols() || w.nrows() != mom_nbr {
                        bail!(
                            "method_of_moments: 'weighting_matrix' must be square and have {mom_nbr} rows and columns!"
                        );
                    }
                    w
                }
            };
            sw = chol_or_err(&w)?;
            for (it, &code) in o.optimizer_vec.iter().enumerate() {
                let (x1, fval, _) = optimize(code, &problem, &data_moments, &sw, &xparam1, &o, out.quiet, None)?;
                out.p(format!("\nStage {stage} Iteration {}: Value of minimized moment distance objective function: {fval:12.10}.\n", it + 1));
                if o.verbose {
                    out.p("\n'verbose' option: ");
                    let title = format!("FREQUENTIST {} (STAGE {stage} ITERATION {}) VERBOSE", o.method.name(), it + 1);
                    out.p(results_table(mf, &est, &names, &x1, &vec![f64::NAN; n], &title, None));
                }
                stages.push(StageResult { stage, iteration: it + 1, optimizer: code, x: x1.clone(), fval });
                xparam1 = x1;
            }
            model_moments = problem
                .evaluate(&xparam1, &data_moments, &sw)
                .map_err(|f| fail_err(f, "objective at the stage estimate"))?
                .model_moments;
        }

        // ---- standard errors (standard_errors.m) -----------------------------------------------------
        let mm_final =
            problem.evaluate(&xparam1, &data_moments, &sw).map_err(|f| fail_err(f, "objective at the estimate"))?;
        let analytic_se = o.method == Method::Gmm && o.analytic_standard_errors;
        out.p(if analytic_se {
            "\nComputing standard errors using analytical derivatives of moments\n"
        } else {
            "\nComputing standard errors using numerical derivatives of moments\n"
        });
        let (stdh, cov) = standard_errors(
            &problem,
            &xparam1,
            &mm_final.model_moments,
            md,
            &sw,
            woptflag,
            nobs,
            o.se_tolx,
            o.bartlett_kernel_lag,
            &names,
            out,
            analytic_se,
        )?;
        res.stdh = stdh.clone();
        res.asympt_cov = cov.clone();
        out.p(results_table(mf, &est, &names, &xparam1, &stdh, o.method.name(), None));
        let pd = cov.as_ref().is_some_and(|c| chol_upper(c).is_some());
        if !pd {
            out.p("\nOPTIMIZATION PROBLEM!\n (minus) the Hessian matrix at the \"mode\" is not positive definite!\n=> variance of the estimated parameters are not positive.\nYou should try to change the initial values of the parameters using\nthe estimated_params_init block, or use another optimization routine.\n");
        }
        hessian_gmm_smm = cov.as_ref().and_then(|c| solve(c, &Mat::identity(n, n)));
        res.weighting = sw.clone();
        res.woptflag = woptflag;
    } else {
        // ---- mode_compute_irf_matching.m ------------------------------------------------------------
        let w = irf_targets.as_ref().unwrap().w.clone();
        let mut fval = fval0;
        // hessian_xparam0 (mode_file)
        let mut hh: Option<Mat> = hess0;
        if slice_no_mode {
            let pr = priors.as_deref().unwrap();
            xparam1 = crate::estimation::samplers::prior_mode_start(pr, &problem.bounds);
            let fin = problem
                .evaluate(&xparam1, &data_moments, &w)
                .map_err(|f| fail_err(f, "objective at the prior mode"))?;
            res.hessian = Some(Mat::from_diagonal(&nalgebra::DVector::from_iterator(
                n,
                pr.iter().map(|p| {
                    let v = p.sd * p.sd;
                    if v.is_finite() { 1.0 / v } else { 1.0 }
                }),
            )));
            res.stdh = vec![f64::NAN; n];
            res.q = fin.q;
            res.fval = fin.fval;
            res.model_moments = fin.model_moments;
            res.weighting = w.clone();
        }
        for (it, &code) in o.optimizer_vec.iter().enumerate().filter(|_| !slice_no_mode) {
            let (x1, f1, h1) = optimize(code, &problem, &data_moments, &w, &xparam1, &o, out.quiet, hh.as_ref())?;
            out.p(format!(
                "\nMode Compute Iteration {}: Value of minimized moment distance objective function: {f1:12.10}.\n",
                it + 1
            ));
            stages.push(StageResult { stage: 1, iteration: it + 1, optimizer: code, x: x1.clone(), fval: f1 });
            xparam1 = x1;
            fval = f1;
            hh = h1;
        }
        let obj = MomObjective { p: &problem, data: &data_moments, w: &w, analytic: false };
        if !slice_no_mode {
            if o.cova_compute {
                out.p("\nComputing Hessian at the mode.\n");
                hh = Some(hessian(&obj, &xparam1, o.gstep));
            }
            // mode file (method_of_moments/<fname>_mode.mat)
            let mdir = ctx.dir.join(&ctx.fname).join("method_of_moments");
            std::fs::create_dir_all(&mdir)?;
            let mut vars = vec![
                ("xparam1", crate::matfile::MatOut::Vector(&xparam1)),
                ("parameter_names", crate::matfile::MatOut::Strings(&names)),
                ("fval", crate::matfile::MatOut::Scalar(fval)),
            ];
            if let Some(h) = &hh {
                vars.push(("hh", crate::matfile::MatOut::Matrix(h)));
            }
            crate::matfile::write(&mdir.join(format!("{}_mode.mat", ctx.fname)), &vars)?;
            if let Some(h) = &hh {
                let invhess = crate::estimation::inverse_hessian(h);
                res.stdh = match &invhess {
                    Some(v) => (0..n).map(|i| v[(i, i)].sqrt()).collect(),
                    None => vec![f64::NAN; n],
                };
                res.asympt_cov = invhess;
            }
            res.hessian = hh.clone();
            let fin =
                problem.evaluate(&xparam1, &data_moments, &w).map_err(|f| fail_err(f, "objective at the mode"))?;
            if bayesian {
                out.p(results_table(mf, &est, &names, &xparam1, &res.stdh, "Posterior", priors.as_deref()));
                if let Some(h) = &hh {
                    let l = crate::estimation::laplace_approximation(h, fin.fval);
                    out.p(format!("\nLog data density [Laplace approximation] is {l:.6}.\n"));
                    res.laplace = Some(l);
                }
            } else {
                out.p(results_table(mf, &est, &names, &xparam1, &res.stdh, o.method.name(), None));
            }
            res.q = fin.q;
            res.fval = fin.fval;
            res.model_moments = fin.model_moments;
            res.weighting = w;
        }
    }

    // ---- Bayesian MCMC estimation (priors and mh_replic > 0 or load_mh_file, every method: for
    // GMM/SMM the penalized objective is the posterior kernel, with the Laplace priors) ----
    if bayesian {
        if let (Some(pr), Some(h)) = (mcmc_priors.as_deref(), hessian_gmm_smm.as_ref().or(res.hessian.as_ref())) {
            let obj = MomObjective { p: &problem, data: &data_moments, w: &res.weighting, analytic: false };
            let dname = match opt(&o.raw, "dirname") {
                Some(OptValue::Str(d)) => ctx.dir.join(d),
                _ => ctx.dir.join(&ctx.fname),
            };
            let seed = ctx.seed_override.unwrap_or(ctx.stream_seed);
            if let Some((stats, _)) = crate::estimation::q3_mcmc::mcmc_after_mode(
                &obj,
                &xparam1,
                h,
                pr,
                &names,
                &o.raw,
                &ctx.dir,
                &dname,
                &ctx.fname,
                seed,
                0,
                "method_of_moments",
                &table_spec(mf, &est, &names),
            )? {
                let mdir = ctx.dir.join(&ctx.fname).join("method_of_moments");
                std::fs::create_dir_all(&mdir)?;
                let j = serde_json::json!({
                    "posterior_mean": stats.mean, "log_data_density_mhm": stats.mhm.log_marginal, "acceptance": stats.acceptance,
                    "hpd_lower": stats.params.iter().map(|p| p.hpd.0).collect::<Vec<_>>(),
                    "hpd_upper": stats.params.iter().map(|p| p.hpd.1).collect::<Vec<_>>(),
                    "posterior_median": stats.params.iter().map(|p| p.median).collect::<Vec<_>>(),
                    "posterior_sd": stats.params.iter().map(|p| p.var.sqrt()).collect::<Vec<_>>(),
                });
                std::fs::write(mdir.join(format!("{}_mom_mcmc.json", ctx.fname)), serde_json::to_string_pretty(&j)?)?;
                // xparam1 = get_posterior_parameters('mean', ...): the final results (and the data /
                // model IRF comparison) are at the posterior mean, also after a slice run without
                // mode (check_posterior_sampler_options resets mh_posterior_mode_estimation)
                xparam1 = stats.mean.clone();
            }
        }
    }

    // ---- final estimation results: objective at xparam1, J-test (Jtest.m) -------------------------------
    let fin = problem
        .evaluate(&xparam1, &data_moments, &res.weighting)
        .map_err(|f| fail_err(f, "objective at the final estimate"))?;
    res.q = fin.q;
    res.fval = fin.fval;
    res.model_moments = fin.model_moments;
    if o.method != Method::IrfMatching && mom_nbr > n {
        let md = m_data.as_ref().unwrap();
        let j = jtest(
            &problem,
            &xparam1,
            &data_moments,
            md,
            &res.weighting,
            res.woptflag,
            nobs,
            vcf,
            o.bartlett_kernel_lag,
        )?;
        out.p(format!("\nValue of J-test statistic: {:.6}\n", j.j_stat));
        out.p(format!("p-value of J-test statistic: {:.6}\n", j.p_val));
        res.j_test = Some(j);
    }

    // ---- display_comparison_moments_irfs.m ------------------------------------------------------------
    let (title, headers) = if o.method == Method::IrfMatching {
        ("COMPARISON OF MATCHED DATA IRFS AND MODEL IRFS".to_string(), ["IRF", "Data", "Model"])
    } else {
        (
            format!("Comparison of matched data moments and model moments ({})", o.method.name()),
            ["Moment", "Data", "Model"],
        )
    };
    let vals = Mat::from_fn(mom_nbr, 2, |i, j| if j == 0 { data_moments[i] } else { res.model_moments[i] });
    let lw = labels.iter().map(|l| l.chars().count()).max().unwrap_or(0) + 2;
    let headers: Vec<String> = headers.iter().map(|s| s.to_string()).collect();
    out.p(dyntable(&title, &headers, &labels, &vals, lw, 10, 7));
    out.p(format!("\n==== Method of Moments Estimation ({}) Completed ====\n\n", o.method.name()));
    res.stages = stages;

    res.xparam1 = xparam1;
    write_results(ctx, &field, &res, mf, &est)?;
    Ok(res)
}

/// Port of `standard_errors.m` (numerical Jacobian of the moments, sandwich formula).
/// Note: Dynare looks for `options_mom_.variance_correction_factor`, which is never set (the
/// factor lives in `options_mom_.mom`), so SMM standard errors use `T = nobs` uncorrected.
/// Port of `Jtest.m` at `x` with the weighting matrix `sw` (its upper Cholesky factor):
/// with a non-optimal matrix the objective is re-evaluated with the optimal one at the model
/// moments. `vcf`: SMM variance correction factor (ignored for GMM).
#[allow(clippy::too_many_arguments)]
pub fn jtest(
    p: &MomProblem,
    x: &[f64],
    data: &[f64],
    m_data: &Mat,
    sw: &Mat,
    woptflag: bool,
    nobs: usize,
    vcf: f64,
    q_lag: usize,
) -> Result<JTest> {
    let e = p.evaluate(x, data, sw).map_err(|f| fail_err(f, "J-test objective"))?;
    let fval = if !woptflag {
        let w_opt = optimal_weighting_matrix(m_data, &e.model_moments, q_lag)?;
        let swo = chol_upper(&w_opt)
            .ok_or_else(|| anyhow!("Jtest: the optimal weighting matrix is not positive definite"))?;
        p.evaluate(x, data, &swo).map_err(|f| fail_err(f, "J-test objective"))?.fval
    } else {
        e.q
    };
    let factor = if p.method == Method::Smm { vcf } else { 1.0 };
    let j_stat = nobs as f64 * factor * fval / p.scaling;
    let dof = data.len() - x.len();
    let p_val = 1.0 - crate::estimation::special::gammainc(j_stat / 2.0, dof as f64 / 2.0);
    Ok(JTest { j_stat, degrees_freedom: dof, p_val })
}

/// [`standard_errors`] without console output, at `x` (model moments recomputed there).
#[allow(clippy::too_many_arguments)]
pub fn standard_errors_at(
    p: &MomProblem,
    x: &[f64],
    m_data: &Mat,
    sw: &Mat,
    woptflag: bool,
    nobs: usize,
    eps: f64,
    q_lag: usize,
) -> Result<(Vec<f64>, Option<Mat>)> {
    let (mm, _) = p.model_moments(x).map_err(|f| fail_err(f, "model moments"))?;
    standard_errors(p, x, &mm, m_data, sw, woptflag, nobs, eps, q_lag, &p.names, &Out { quiet: true }, false)
}

/// `standard_errors.m` with `analytic_standard_errors` (GMM).
pub fn standard_errors_analytic_at(
    p: &MomProblem,
    x: &[f64],
    m_data: &Mat,
    sw: &Mat,
    woptflag: bool,
    nobs: usize,
    q_lag: usize,
) -> Result<(Vec<f64>, Option<Mat>)> {
    let (mm, _) = p.model_moments(x).map_err(|f| fail_err(f, "model moments"))?;
    standard_errors(p, x, &mm, m_data, sw, woptflag, nobs, 0.0, q_lag, &p.names, &Out { quiet: true }, true)
}

#[allow(clippy::too_many_arguments)]
fn standard_errors(
    p: &MomProblem,
    x: &[f64],
    model_moments: &[f64],
    m_data: &Mat,
    sw: &Mat,
    woptflag: bool,
    nobs: usize,
    eps: f64,
    q_lag: usize,
    names: &[String],
    out: &Out,
    analytic: bool,
) -> Result<(Vec<f64>, Option<Mat>)> {
    let n = x.len();
    let nm = model_moments.len();
    let mut d = Mat::zeros(nm, n);
    if analytic {
        match p.moments_jacobian(x) {
            Ok(j) if j.iter().all(|v| !v.is_nan()) => d = j,
            _ => {
                out.p("Warning: There are NaN in the analytical Jacobian of Moments. Check your bounds and/or priors, or use a different optimizer.\n");
                return Ok((vec![f64::NAN; n], None));
            }
        }
    }
    for i in 0..if analytic { 0 } else { n } {
        let mut xp = x.to_vec();
        xp[i] += eps;
        let mut xm = x.to_vec();
        xm[i] -= eps;
        match (p.model_moments(&xp), p.model_moments(&xm)) {
            (Ok((mp, _)), Ok((mm, _))) => {
                for k in 0..nm {
                    d[(k, i)] = (mp[k] - mm[k]) / (2.0 * eps);
                }
            }
            (rp, rm) => {
                let up = matches!(rp, Err(f) if f.info == 42);
                let lo = matches!(rm, Err(f) if f.info == 41);
                if up {
                    out.p(format!("Warning: Cannot compute the Jacobian using finite differences for parameter {} due to hitting the upper bound - no standard errors available.\n", names[i]));
                }
                if lo {
                    out.p(format!("Warning: Cannot compute the Jacobian using finite differences for parameter {} due to hitting the lower bound - no standard errors available.\n", names[i]));
                }
                if !up && !lo {
                    out.p(format!("Warning: Cannot compute the Jacobian using finite differences for parameter {} - no standard errors available\n", names[i]));
                }
                return Ok((vec![f64::NAN; n], None));
            }
        }
    }
    let t = nobs as f64;
    let ww = mul(sw, true, sw, false);
    let dwd = mul(&mul(&d, true, &ww, false), false, &d, false);
    let Some(aa) = solve(&dwd, &Mat::identity(n, n)) else { return Ok((vec![f64::NAN; n], None)) };
    let cov = if woptflag {
        aa / t
    } else {
        let wopt = optimal_weighting_matrix(m_data, model_moments, q_lag)?;
        let s = solve(&wopt, &Mat::identity(nm, nm))
            .ok_or_else(|| anyhow!("standard_errors: singular optimal weighting matrix"))?;
        // 1/T*AA*D'*WW*S*WW*D*AA, evaluated left to right as MATLAB does
        let mut m = &aa / t;
        m = mul(&m, false, &d, true);
        m = mul(&m, false, &ww, false);
        m = mul(&m, false, &s, false);
        m = mul(&m, false, &ww, false);
        m = mul(&m, false, &d, false);
        mul(&m, false, &aa, false)
    };
    let sd = (0..n).map(|i| cov[(i, i)].sqrt()).collect();
    Ok((sd, Some(cov)))
}

fn mat_json(m: &Mat) -> serde_json::Value {
    json!((0..m.nrows()).map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect::<Vec<_>>()).collect::<Vec<_>>())
}

fn write_results(ctx: &Context, field: &str, r: &MomResults, mf: &ModFile, est: &[EstimatedParam]) -> Result<()> {
    let out_dir = ctx.dir.join(&ctx.fname).join("Output");
    std::fs::create_dir_all(&out_dir)?;
    let mut mode = serde_json::Map::new();
    let mut std = serde_json::Map::new();
    for (k, e) in est.iter().enumerate() {
        let (group, name) = match e.kind {
            EstimatedKind::Param(i) => ("parameters", mf.model.param_names[i].clone()),
            EstimatedKind::StdErr(i) => ("shocks_std", mf.model.exo_names[i].clone()),
            EstimatedKind::Corr(a, b) => {
                ("shocks_corr", format!("{}_{}", mf.model.exo_names[a], mf.model.exo_names[b]))
            }
            // display_estimation_results_table.m: measurement_errors_std / measurement_errors_corr
            EstimatedKind::MeStdErr(k) => ("measurement_errors_std", mf.model.endo_names[k].clone()),
            EstimatedKind::MeCorr(a, b) => {
                ("measurement_errors_corr", format!("{}_{}", mf.model.endo_names[a], mf.model.endo_names[b]))
            }
            EstimatedKind::Skew(_) | EstimatedKind::InitState(_) => unreachable!("rejected by run()"),
        };
        mode.entry(group).or_insert_with(|| json!({}))[&name] = json!(r.xparam1[k]);
        std.entry(group).or_insert_with(|| json!({}))[&name] = json!(r.stdh[k]);
    }
    let mut v = json!({
        "mom_method": r.method.name(),
        "parameter_names": r.names,
        "x0": r.x0,
        "fval_at_x0": r.fval0,
        "xparam1": r.xparam1,
        "std_at_mode": r.stdh,
        "data_moments": r.data_moments,
        "model_moments": r.model_moments,
        "labels": r.labels,
        "Q": r.q,
        "fval": r.fval,
        "stages": r.stages.iter().map(|s| json!({"stage": s.stage, "iteration": s.iteration, "mode_compute": s.optimizer, "x": s.x, "fval": s.fval})).collect::<Vec<_>>(),
        "weighting_info": { if r.method == Method::IrfMatching { "W" } else { "Sw" }: mat_json(&r.weighting), "Woptflag": r.woptflag },
    });
    v[format!("{field}_mode")] = serde_json::Value::Object(mode);
    v[format!("{field}_std_at_mode")] = serde_json::Value::Object(std);
    if let Some(c) = &r.asympt_cov {
        v["asympt_cov_mat"] = mat_json(c);
    }
    if let Some(j) = &r.j_test {
        v["J_test"] = json!({"j_stat": j.j_stat, "degrees_freedom": j.degrees_freedom, "p_val": j.p_val});
    }
    if let Some(m) = &r.m_data {
        v["m_data"] = mat_json(m);
    }
    if let Some(h) = &r.hessian {
        v["hessian"] = mat_json(h);
    }
    if let Some(l) = r.laplace {
        v["MarginalDensity"] = json!({"LaplaceApproximation": l});
    }
    let path = out_dir.join(format!("{}_mom_results.json", ctx.fname));
    std::fs::write(&path, serde_json::to_string_pretty(&v)?)?;
    Ok(())
}
