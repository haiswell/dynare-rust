// SPDX-License-Identifier: GPL-3.0-or-later
//! Global sensitivity analysis: Dynare's `sensitivity` (`dynare_sensitivity`) command,
//! a port of `matlab/+gsa/` (M. Ratto, *Analysing DSGE models with global sensitivity
//! analysis*, Computational Economics 31, 2008).
//!
//! * [`sampling`]: LPTAU (Sobol, [`sobol`]) / Latin hypercube samples of the prior ranges or
//!   of the priors, Morris trajectories; random draws through MATLAB's default stream
//!   ([`rng`]) so that the samples are Dynare's.
//! * [`stability`]: `stability_mapping.m` (unique / explosive / indeterminate / no solution).
//! * [`mcf`]: Monte Carlo filtering (Smirnov tests, correlation analysis, scatter plots).
//! * [`redform`]: reduced-form mapping with `threshold_redform` (`reduced_form_mapping.m`),
//!   Morris screening (`reduced_form_screening.m`), and a regression substitute for the
//!   SS-ANOVA mapping.
//! * [`rmse`]: RMSE / likelihood filtering against the data (`monte_carlo_filtering.m`).
//!
//! Solves run in parallel on the rayon pool ([`DsgeModel::resolve`] is a pure function).

// MATLAB-style index loops and NaN-aware `!(a > b)` tests mirror the ported .m files.
#![allow(
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::type_complexity,
    clippy::explicit_counter_loop
)]

pub mod calibration; // irf_calibration / moment_calibration (map_calibration.m)
pub mod glue; // glue
pub mod identification; // sensitivity(identification) (map_identification.m)
pub mod mcf;
pub mod posterior; // ppost
pub mod redform;
pub mod rmse;
pub mod rng;
pub mod sampling;
pub mod sobol;
pub mod stability;
pub mod stats;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};

use crate::dsge::DsgeModel;
use crate::estimation::{Bounds, estimation_bounds};
use crate::graphics::{self, Figure, GraphFormat, GraphOptions};
use crate::modfile::{Command, EstimatedKind, ModFile, OptValue, Options, opt};

/// Options of the `sensitivity` command, with `+gsa/run.m`'s defaults and overrides.
#[derive(Clone, Debug)]
pub struct GsaOptions {
    pub stab: bool,
    pub nsam: usize,
    pub ilptau: u8,
    pub pprior: bool,
    pub prior_range: bool,
    pub ppost: bool,
    pub neighborhood_width: f64,
    pub morris: u8,
    pub morris_nliv: usize,
    pub morris_ntra: usize,
    pub load_stab: bool,
    pub alpha2_stab: f64,
    pub pvalue_ks: f64,
    pub pvalue_corr: f64,
    pub redform: bool,
    pub logtrans_redform: bool,
    pub threshold_redform: Option<(f64, f64)>,
    pub ksstat_redform: f64,
    pub alpha2_redform: f64,
    pub namendo: Vec<String>,
    pub namlagendo: Vec<String>,
    pub namexo: Vec<String>,
    pub rmse: bool,
    pub lik_only: bool,
    pub var_rmse: Vec<String>,
    pub pfilt_rmse: f64,
    pub istart_rmse: Option<usize>,
    pub alpha_rmse: f64,
    pub alpha2_rmse: f64,
    pub identification: bool,
    /// `glue` (GLUE file from the RMSE analysis).
    pub glue: bool,
    /// `ar`, `useautocorr` of `identification=1` (defaults 1, 0).
    pub ident_ar: usize,
    pub ident_useautocorr: bool,
    /// `load_redform`, `load_rmse`, `load_ident_files`: reload the saved results of a previous run.
    pub load_redform: bool,
    pub load_rmse: bool,
    pub load_ident_files: bool,
    pub datafile: Option<String>,
    pub nograph: bool,
    pub graph_formats: Vec<GraphFormat>,
    pub mode_file: Option<String>,
    /// The raw command options (data/filter options are read from them).
    pub raw: Options,
}

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

fn strings(v: &OptValue) -> Vec<String> {
    match v {
        OptValue::Str(s) => vec![s.clone()],
        OptValue::List(items) => items.iter().flat_map(strings).collect(),
        _ => Vec::new(),
    }
}

impl GsaOptions {
    /// Parses the options of `sensitivity(...)`. `endo`/`exo`: names for the `:` shortcut.
    pub fn parse(o: &Options, mf: &ModFile) -> Result<GsaOptions> {
        let has = |k: &str| opt(o, k).is_some();
        let b = |k: &str, d: bool| {
            num(o, k).map_or(if opt(o, k) == Some(&OptValue::Flag) { true } else { d }, |v| v != 0.0)
        };
        let f = |k: &str, d: f64| num(o, k).unwrap_or(d);
        let m = &mf.model;
        let names = |k: &str, all: &[String]| -> Vec<String> {
            let v = opt(o, k).map(strings).unwrap_or_default();
            if v.iter().any(|s| s == ":") { all.to_vec() } else { v }
        };
        let endo_all: Vec<String> = m.endo_names[..m.orig_endo_nbr].to_vec();
        let threshold_redform = match opt(o, "threshold_redform") {
            Some(OptValue::Vec(v)) if v.len() == 2 => Some((v[0], v[1])),
            Some(OptValue::List(v)) if v.len() == 2 => match (&v[0], &v[1]) {
                (OptValue::Num(a), OptValue::Num(b)) => Some((*a, *b)),
                _ => bail!("threshold_redform: expected two numbers"),
            },
            Some(_) => bail!("threshold_redform: expected [lower upper]"),
            None => None,
        };
        let graph_formats: Vec<GraphFormat> = match opt(o, "graph_format") {
            Some(v) => strings(v).iter().filter_map(|s| GraphFormat::parse(s)).collect(),
            None => vec![GraphFormat::Pdf],
        };
        let mut g = GsaOptions {
            stab: b("stab", true),
            nsam: f("nsam", 2048.0) as usize,
            ilptau: f("ilptau", 1.0) as u8,
            pprior: b("pprior", true),
            prior_range: b("prior_range", true),
            ppost: b("ppost", false),
            neighborhood_width: f("neighborhood_width", 0.0),
            morris: f("morris", 0.0) as u8,
            morris_nliv: f("morris_nliv", 6.0) as usize,
            morris_ntra: f("morris_ntra", 20.0) as usize,
            load_stab: b("load_stab", false),
            alpha2_stab: f("alpha2_stab", 0.0),
            pvalue_ks: f("pvalue_ks", 0.001),
            pvalue_corr: f("pvalue_corr", 1e-5),
            redform: b("redform", false),
            logtrans_redform: b("logtrans_redform", false),
            threshold_redform,
            ksstat_redform: f("ksstat_redform", 0.001),
            alpha2_redform: f("alpha2_redform", 1e-5),
            namendo: names("namendo", &endo_all),
            namlagendo: names("namlagendo", &endo_all),
            namexo: names("namexo", &m.exo_names),
            rmse: b("rmse", false),
            lik_only: b("lik_only", false),
            var_rmse: opt(o, "var_rmse").map(strings).unwrap_or_else(|| mf.varobs.clone()),
            pfilt_rmse: f("pfilt_rmse", 0.1),
            istart_rmse: num(o, "istart_rmse").map(|v| v as usize),
            alpha_rmse: f("alpha_rmse", 0.001),
            alpha2_rmse: f("alpha2_rmse", 1e-5),
            identification: b("identification", false),
            glue: b("glue", false),
            ident_ar: f("ar", 1.0) as usize,
            ident_useautocorr: b("useautocorr", false),
            load_redform: b("load_redform", false),
            load_rmse: b("load_rmse", false),
            load_ident_files: b("load_ident_files", false),
            datafile: match opt(o, "datafile") {
                Some(OptValue::Str(s)) => Some(s.clone()),
                _ => None,
            },
            // graph_format=none: Dynare still runs every analysis, it just saves no file
            nograph: has("nograph"),
            graph_formats,
            mode_file: match opt(o, "mode_file") {
                Some(OptValue::Str(s)) => Some(s.clone()),
                _ => None,
            },
            raw: o.clone(),
        };
        // ---- run.m: consistency checks and overrides ----
        if g.neighborhood_width > 0.0 && (has("pprior") && g.pprior || has("ppost") && g.ppost) {
            bail!("sensitivity:: neighborhood_width is incompatible with prior/posterior sampling");
        }
        if g.morris == 1 {
            if g.ppost {
                bail!("sensitivity:: Morris is incompatible with posterior sampling");
            }
            if has("pprior") && !g.pprior && g.neighborhood_width == 0.0 {
                bail!("sensitivity:: Morris is incompatible with MC sampling with correlation matrix");
            }
            if g.rmse {
                bail!("sensitivity:: Morris is incompatible with rmse analysis");
            }
            if (has("alpha2_stab") && g.alpha2_stab < 1.0)
                || (has("pvalue_ks") && g.pvalue_ks != 0.0)
                || (has("pvalue_corr") && g.pvalue_corr != 0.0)
            {
                bail!("sensitivity:: Morris is incompatible with Monte Carlo filtering");
            }
        }
        if g.rmse && g.datafile.is_none() {
            println!("The data file and all relevant estimation options ");
            println!("[first_obs, nobs, presample, prefilter, loglinear, lik_init, kalman_algo, ...]");
            println!("must be specified for RMSE analysis!");
            bail!("Sensitivity anaysis error!");
        }
        if g.identification {
            g.redform = false;
            if !has("morris") {
                g.morris = 1;
            }
        }
        if g.neighborhood_width > 0.0 {
            g.pprior = false;
            g.ppost = false;
        }
        if g.redform && g.neighborhood_width == 0.0 && g.threshold_redform.is_none() {
            g.pprior = true;
            g.ppost = false;
        }
        if g.morris > 2 {
            println!("The option morris = 3 is no longer supported");
            println!("the option is reset at morris = 1 .");
            g.morris = 1;
        }
        if g.morris == 1 {
            if !g.identification {
                g.redform = true;
            }
            g.pprior = g.neighborhood_width == 0.0;
            g.ppost = false;
            g.glue = false;
            g.rmse = false;
            g.load_rmse = false;
            g.alpha2_stab = 1.0;
            g.pvalue_ks = 0.0;
            g.pvalue_corr = 0.0;
        }
        Ok(g)
    }
}

/// What the analyses share: names, bounds, output folder, graphics.
pub struct Ctx {
    pub fname: String,
    /// `<dir>/<fname>/gsa` (or `gsa/screen` for Morris).
    pub out_dir: PathBuf,
    pub graphs: GraphOptions,
    /// Number of shock parameters (stderr/corr) at the front of xparam1.
    pub nshock: usize,
    /// `get_the_name` of every estimated parameter (`SE_e`, `CC_e1_e2`, parameter names).
    pub param_names: Vec<String>,
    /// GSA bounds (`prior_bounds(bayestopt_, max(prior_trunc,1e-10))` and explicit bounds).
    pub bounds: Bounds,
    /// The calibration as an xparam1 vector (`get_all_parameters`).
    pub xparam_calib: Vec<f64>,
    /// `xparam1` and `hh` of the `mode_file` when the sample is drawn from the posterior
    /// (`pprior = 0`, no `neighborhood_width`, `stability_mapping.m`).
    pub posterior_mode: Option<(Vec<f64>, crate::linalg::Mat)>,
    /// The `options_` of earlier commands that `th_autocovariances` reads (`hp_filter`, ...).
    pub moment_base: crate::analysis::moments::MomentOptions,
}

impl Ctx {
    pub fn graphs_enabled(&self) -> bool {
        self.graphs.enabled
    }

    /// The same context writing its figures (and files) to `dir`.
    pub fn with_dir(&self, dir: &Path) -> Ctx {
        Ctx {
            fname: self.fname.clone(),
            out_dir: dir.to_path_buf(),
            graphs: GraphOptions { dir: dir.to_path_buf(), ..self.graphs.clone() },
            nshock: self.nshock,
            param_names: self.param_names.clone(),
            bounds: self.bounds.clone(),
            xparam_calib: self.xparam_calib.clone(),
            posterior_mode: self.posterior_mode.clone(),
            moment_base: self.moment_base.clone(),
        }
    }

    pub fn save(&self, fig: &Figure) {
        if self.graphs.formats.is_empty() {
            return;
        }
        if let Some(d) = self.graphs.enabled.then_some(&self.graphs.dir) {
            let _ = std::fs::create_dir_all(d);
        }
        if let Err(e) = graphics::save(fig, &self.fname, &self.graphs) {
            println!("WARNING: could not save figure {}: {e}", fig.name);
        }
    }
}

/// `get_the_name.m` without TeX: `SE_<exo>`, `CC_<exo1>_<exo2>`, or the parameter name.
pub fn gsa_param_names(mf: &ModFile, dsge: &DsgeModel) -> Vec<String> {
    let m = &mf.model;
    dsge.estimated
        .iter()
        .map(|e| match e.kind {
            EstimatedKind::Param(i) => m.param_names[i].clone(),
            EstimatedKind::StdErr(k) => format!("SE_{}", m.exo_names[k]),
            EstimatedKind::Corr(a, b) => format!("CC_{}_{}", m.exo_names[a], m.exo_names[b]),
            // measurement errors (stream M), as get_the_name.m names them
            EstimatedKind::MeStdErr(k) => format!("SE_EOBS_{}", m.endo_names[k]),
            EstimatedKind::MeCorr(a, b) => format!("CC_EOBS_{}_{}", m.endo_names[a], m.endo_names[b]),
            EstimatedKind::Skew(k) => format!("SKEW_{}", m.exo_names[k]),
            EstimatedKind::InitState(k) => format!("INIT_{}", m.endo_names[k]), // ---- estimation completion
        })
        .collect()
}

/// `prior_trunc` of the last estimation command before `sensitivity` (Dynare's options_).
fn previous_estimation_options(mf: &ModFile, upto: usize) -> Options {
    mf.commands[..upto]
        .iter()
        .rev()
        .find_map(|c| match c {
            Command::Estimation(o, _) => Some(o.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// What a `sensitivity` run produced (for tests and callers).
pub struct GsaOutput {
    pub options: GsaOptions,
    pub ctx: Ctx,
    pub dsge: DsgeModel,
    pub stab: stability::StabMap,
    pub rmse: Option<rmse::RmseResults>,
    /// The reduced-form indices of the regression mapping (`label`, `S_i`), computed or loaded.
    pub redform_si: Vec<(String, Vec<f64>)>,
    /// The Morris identification screening (`SAvdec`, `SAcc`, `SAnorm`), computed or loaded.
    pub ident_sa: Option<identification::IdentScreening>,
}

/// Runs `sensitivity(...)` (command index `cmd_index` in `mf.commands`).
pub fn run(mf: &ModFile, dir: &Path, fname: &str, o: &Options, cmd_index: usize) -> Result<Option<GsaOutput>> {
    let t0 = std::time::Instant::now();
    let g = GsaOptions::parse(o, mf)?;
    // options_ as left by a previous estimation command, then the sensitivity options
    let mut eo = previous_estimation_options(mf, cmd_index);
    for (k, v) in o {
        eo.retain(|(kk, _)| kk != k);
        eo.push((k.clone(), v.clone()));
    }
    let mut dsge = DsgeModel::new(mf, None).context("sensitivity")?;
    // run.m: options_.loglinear = options_gsa.loglinear (inside the estimation-init branch)
    if runs_estimation_init(mf, cmd_index, &g)
        && let Some(v) = opt(o, "loglinear")
    {
        dsge.loglinear = !matches!(v, OptValue::Num(x) if *x == 0.0);
    }
    let qz = qz_criterium(&eo, runs_estimation_init(mf, cmd_index, &g));
    dsge.set_solver_options(crate::solver::SolverOptions { qz_criterium: qz, ..Default::default() });
    if mf.model.n_exo() == 0 {
        bail!(
            "sensitivity does not support having no varexo in the model. As a workaround you could define a dummy exogenous variable."
        );
    }
    let nshock =
        dsge.estimated.iter().filter(|e| matches!(e.kind, EstimatedKind::StdErr(_) | EstimatedKind::Corr(..))).count();
    // stability_mapping.m: without priors (ML) the bounds are those of estimated_params
    let ml = dsge.priors.is_empty();
    let prior_trunc = match num(&eo, "prior_trunc") {
        Some(v) if v != 0.0 => v,
        _ if ml => 0.0,
        _ => {
            println!(
                "\nstability_mapping: GSA with priors requires bounded support. Setting options_.prior_trunc=1e-10."
            );
            1e-10
        }
    };
    let explicit: Vec<(Option<f64>, Option<f64>)> = dsge
        .estimated
        .iter()
        .map(|e| match e.kind {
            EstimatedKind::Corr(..) => (Some(e.lb.unwrap_or(-1.0).max(-1.0)), Some(e.ub.unwrap_or(1.0).min(1.0))),
            _ => (e.lb, e.ub),
        })
        .collect();
    let bounds = if ml {
        Bounds {
            lb: explicit.iter().map(|b| b.0.unwrap_or(f64::NEG_INFINITY)).collect(),
            ub: explicit.iter().map(|b| b.1.unwrap_or(f64::INFINITY)).collect(),
        }
    } else {
        estimation_bounds(&dsge.priors, &explicit, prior_trunc)
    };
    let mut xparam_calib = calibration_vector(&dsge); // get_all_parameters(estim_params_, M_)
    // stability_mapping.m: with neighborhood_width > 0 and a mode_file, the sample is centred
    // on xparam1 loaded from the mode file
    if g.neighborhood_width > 0.0
        && let Some(f) = g.mode_file.as_deref().filter(|f| !f.is_empty())
    {
        let (x, _) = crate::driver::read_mode_file(dir, f, &dsge.estimated_names, 0, || Ok(xparam_calib.clone()))?;
        xparam_calib = x;
    }
    let sub = if g.morris == 1 { "gsa/screen" } else { "gsa" };
    let out_dir = dir.join(fname).join(sub);
    std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let mut ctx = Ctx {
        fname: fname.to_string(),
        graphs: GraphOptions { enabled: !g.nograph, formats: g.graph_formats.clone(), dir: out_dir.clone() },
        out_dir,
        nshock,
        param_names: gsa_param_names(mf, &dsge),
        bounds,
        xparam_calib,
        posterior_mode: None,
        moment_base: calibration::persistent_moment_options(mf, cmd_index),
    };
    if !g.pprior && !g.ppost && !g.load_stab && g.stab && g.neighborhood_width <= 0.0 {
        ctx.posterior_mode = Some(read_posterior_mode(dir, &eo, &g, ctx.param_names.len())?);
    }

    // run.m: loading a saved prior sample needs its file
    if (g.load_stab || g.load_rmse || g.load_redform) && g.pprior {
        let f = ctx.out_dir.join(format!("{}_prior.mat", ctx.fname));
        if !f.exists() {
            println!("{} not found!", f.display());
            println!("You asked to load a non existent analysis");
            return Ok(None);
        }
    }
    let mut rng = rng::MatlabRng::new(0);
    // ---- calibration restrictions (options_.endogenous_prior_restrictions) ----
    let restrictions = calibration::restrictions_checked(mf, cmd_index)?;
    let stab = if g.ppost {
        // the posterior subsample of the previous MCMC (no stability mapping)
        let mh_drop = num(&eo, "mh_drop").unwrap_or(0.5);
        let sub_draws = num(&eo, "sub_draws").map(|v| v as usize);
        posterior::posterior_sample(&ctx, &dsge, dir, mh_drop, sub_draws, 0)?
    } else if g.stab && !g.load_stab {
        let m = stability::stability_mapping(&ctx, &dsge, &g, &mut rng, restrictions.as_ref())?;
        stability::save(&ctx, &g, &m)?;
        m
    } else {
        // load_stab=1, or stab=0: run.m skips the mapping and the later analyses read the
        // sample the previous call saved
        stability::load(&ctx, &dsge, &g, restrictions.as_ref())?
    };
    if stab.x0.is_none() {
        println!("\nSensitivity computations stopped: no parameter set provided a unique solution");
        return Ok(None);
    }
    // ---- map_identification.m ----
    let mut ident_sa = None;
    if g.identification {
        ident_sa = identification::map_identification(&ctx, mf, dir, &dsge, &g, &stab, 0)?;
    }
    // ---- map_calibration.m ----
    if let Some(r) = &restrictions {
        calibration::map_calibration(&ctx, &dsge, &g, &stab, r)?;
    }
    let mut redform_si = Vec::new();
    if g.redform && !g.namendo.is_empty() {
        if g.morris == 1 {
            redform::screening(&ctx, &dsge, &g, &stab)?;
        } else {
            redform_si = redform::mapping(&ctx, &dsge, &g, &stab)?;
        }
    }
    let rmse = if g.rmse { Some(rmse::run(&ctx, mf, dir, &dsge, &g, &eo, &stab)?) } else { None };
    // ---- glue ----
    if g.glue {
        glue::write_glue(&ctx, mf, dir, &dsge, &g, &eo, &stab, rmse.as_ref())?;
    }
    println!("Sensitivity analysis done in {:.1}s.", t0.elapsed().as_secs_f64());
    Ok(Some(GsaOutput { options: g, ctx, dsge, stab, rmse, redform_si, ident_sa }))
}

/// Whether `+gsa/run.m` takes its first branch (`~isempty(options_gsa.datafile) ||
/// isempty(bayestopt_) || options_gsa.rmse`: `dynare_estimation_init` fills `options_` from the
/// estimation options) for the `sensitivity` command number `cmd_index`. `bayestopt_` is filled by
/// the commands that run `dynare_estimation_init` or `set_prior` before it.
pub fn runs_estimation_init(mf: &ModFile, cmd_index: usize, g: &GsaOptions) -> bool {
    let bayestopt_set = mf.commands[..cmd_index].iter().any(|c| match c {
        Command::Estimation(..) | Command::CalibSmoother(..) => true,
        Command::Other(n, _) => matches!(n.as_str(), "identification"),
        _ => false,
    });
    g.datafile.is_some() || !bayestopt_set || g.rmse
}

/// `load([options_.mode_file '.mat'], 'hh', 'xparam1')` of `stability_mapping.m`.
fn read_posterior_mode(dir: &Path, eo: &Options, g: &GsaOptions, n: usize) -> Result<(Vec<f64>, crate::linalg::Mat)> {
    let f = match (&g.mode_file, opt(eo, "mode_file")) {
        (Some(f), _) => f.clone(),
        (None, Some(OptValue::Str(f))) => f.clone(),
        _ => bail!("sensitivity: sampling from the posterior (pprior=0) needs a mode_file (or neighborhood_width > 0)"),
    };
    let p = dir.join(if f.ends_with(".mat") { f.clone() } else { format!("{f}.mat") });
    let vars = crate::matfile::read(&p).with_context(|| format!("loading {}", p.display()))?;
    let x = vars.get("xparam1").and_then(|v| v.as_vec()).ok_or_else(|| anyhow!("{}: no xparam1", p.display()))?;
    let hh = vars.get("hh").and_then(|v| v.as_matrix()).ok_or_else(|| anyhow!("{}: no hh", p.display()))?;
    if x.len() != n || hh.nrows() != n || hh.ncols() != n {
        bail!("{}: xparam1/hh do not match the {n} estimated parameters", p.display());
    }
    Ok((x, hh))
}

/// `options_.qz_criterium` as `+gsa/run.m` leaves it: the user's value, else
/// `select_qz_criterium_value.m` (through `dynare_estimation_init`): `1-1e-6` with the
/// stationary filter (`lik_init=1`, unit roots count as explosive), `1+1e-6` otherwise.
///
/// `estimation_init`: `run.m` goes through `dynare_estimation_init` (`select_qz_criterium_value`)
/// when it has a datafile, no `bayestopt_` or the `rmse` analysis; otherwise (an estimation ran
/// before, and `dynare_estimation_1.m:620` restored `options_.qz_criterium` to its old, empty
/// value) it sets `1+1e-6` when `options_.qz_criterium` is empty.
pub fn qz_criterium(eo: &Options, estimation_init: bool) -> f64 {
    if let Some(v) = num(eo, "qz_criterium") {
        return v;
    }
    if !estimation_init {
        return 1.0 + 1e-6;
    }
    let lik_init = if opt(eo, "diffuse_filter").is_some() { 3.0 } else { num(eo, "lik_init").unwrap_or(1.0) };
    if lik_init == 1.0 { 1.0 - 1e-6 } else { 1.0 + 1e-6 }
}

/// `get_all_parameters.m`: the calibration in xparam1 order.
fn calibration_vector(dsge: &DsgeModel) -> Vec<f64> {
    dsge.estimated
        .iter()
        .enumerate()
        .map(|(k, e)| match e.kind {
            EstimatedKind::Param(i) => dsge.base_params[i],
            EstimatedKind::StdErr(i) => dsge.base_sigma_e[(i, i)].sqrt(),
            EstimatedKind::Corr(i, j) => {
                dsge.base_sigma_e[(i, j)] / (dsge.base_sigma_e[(i, i)] * dsge.base_sigma_e[(j, j)]).sqrt()
            }
            EstimatedKind::MeStdErr(_)
            | EstimatedKind::MeCorr(..)
            | EstimatedKind::Skew(_)
            | EstimatedKind::InitState(_) => dsge.calibrated_value(k), // M_.H (stream M)
        })
        .collect()
}

/// Looks up an endogenous variable by name.
pub(crate) fn endo_index(dsge: &DsgeModel, name: &str) -> Result<usize> {
    dsge.model.endo_index(name).ok_or_else(|| anyhow!("{name} is not an endogenous variable"))
}
