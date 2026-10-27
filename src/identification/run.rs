// SPDX-License-Identifier: GPL-3.0-or-later
//! The `identification(...)` command: port of `+identification/run.m` (options, parameter
//! selection, the point analysis at `parameter_set`, the Monte Carlo analysis over `prior_mc`
//! prior draws, display, figures and saved results).

use std::path::Path;

use anyhow::{Context as _, Result, anyhow, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;

use crate::dsge::DsgeModel;
use crate::estimation::{Bounds, prior_bounds, prior_draw};
use crate::graphics::{GraphFormat, GraphOptions};
use crate::linalg::Mat;
use crate::modfile::{EstimatedKind, ModFile, OptValue, Options, opt};
use crate::solver::{FirstOrderSolver, SolverOptions};

use super::checks::{CheckResult, vnorm};
use super::derivs::{ModelParamDerivs, ParamSet};
use super::display::{Criterion, display};
use super::{Analysis, Context, IdeBlock, Persistent, plots};

/// `options_ident` (defaults of run.m).
#[derive(Clone, Debug)]
pub struct IdentOptions {
    pub parameter_set: String,
    pub ar: usize,
    pub useautocorr: bool,
    pub prior_mc: usize,
    pub prior_range: bool,
    pub periods: usize,
    /// The global `options_.drop` (set by an earlier `stoch_simul(drop=...)`, default 100).
    pub drop: usize,
    pub replic: usize,
    pub advanced: bool,
    pub normalize_jacobians: bool,
    pub grid_nbr: usize,
    /// `None` = `'robust'`.
    pub tol_rank: Option<f64>,
    pub tol_deriv: f64,
    pub tol_sv: f64,
    pub schur_vec_tol: f64,
    pub no_strength: bool,
    pub no_reducedform: bool,
    pub no_moments: bool,
    pub no_minimal: bool,
    pub no_spectrum: bool,
    pub max_dim_cova_group: usize,
    pub analytic_derivation_mode: i32,
    pub lik_init: u8,
    pub diffuse_filter: bool,
    pub kalman_tol: f64,
    pub riccati_tol: f64,
    /// `kalman_algo`, `use_univariate_filters_if_singularity_is_detected`, `diffuse_kalman_tol`
    /// (run.m copies them to `options_` for dsge_likelihood in the strength analysis).
    pub kalman_algo: u8,
    pub use_univariate_filters_if_singularity_is_detected: bool,
    pub diffuse_kalman_tol: f64,
    /// `TeX`: the collinearity-pattern tables of the brute-force search (bruteforce.m), with
    /// the TeX names of the parameters (`options_ident.name_tex`, set by [`run`]).
    pub tex: bool,
    pub name_tex: Vec<String>,
    pub prior_trunc: Option<f64>,
    pub nograph: bool,
    pub graph_formats: Vec<GraphFormat>,
    /// `order` (1, 2 or 3: pruned state space).
    pub order: usize,
    /// `checks_via_subsets`, `max_dim_subsets_groups` (default `min(4, tot - 1)`).
    pub checks_via_subsets: bool,
    pub max_dim_subsets_groups: Option<usize>,
    /// `load_ident_files`, `gsa_sample_file` (0 = none, 1/2 = Dynare's files, or a name).
    pub load_ident_files: bool,
    pub gsa_sample_file: Option<OptValue>,
    /// the previous estimation's points for `parameter_set = posterior_*`.
    pub posterior: Option<PosteriorPoints>,
    /// an external Monte Carlo sample (`identification.run(..., pdraws0)`, as
    /// `+gsa/map_identification.m` passes the GSA sample).
    pub external_sample: Option<Vec<Vec<f64>>>,
}

/// `oo_.posterior.optimization.mode`, `oo_.posterior_mean`, `oo_.posterior_median`
/// of an estimation earlier in the file (xparam1 order).
#[derive(Clone, Debug, Default)]
pub struct PosteriorPoints {
    pub mode: Vec<f64>,
    pub mean: Option<Vec<f64>>,
    pub median: Option<Vec<f64>>,
}

impl Default for IdentOptions {
    fn default() -> Self {
        IdentOptions {
            parameter_set: "prior_mean".into(),
            ar: 1,
            useautocorr: false,
            prior_mc: 1,
            prior_range: false,
            periods: 300,
            drop: 100,
            replic: 100,
            advanced: false,
            normalize_jacobians: true,
            grid_nbr: 5000,
            tol_rank: None,
            tol_deriv: 1e-8,
            tol_sv: 1e-3,
            schur_vec_tol: 1e-11,
            no_strength: false,
            no_reducedform: false,
            no_moments: false,
            no_minimal: false,
            no_spectrum: false,
            max_dim_cova_group: 2,
            analytic_derivation_mode: 0,
            lik_init: 1,
            diffuse_filter: false,
            kalman_tol: 1e-10,
            riccati_tol: 1e-6,
            kalman_algo: 0,
            use_univariate_filters_if_singularity_is_detected: true,
            diffuse_kalman_tol: 1e-6,
            tex: false,
            name_tex: Vec::new(),
            prior_trunc: None,
            nograph: false,
            graph_formats: vec![GraphFormat::Pdf],
            order: 1,
            checks_via_subsets: false,
            max_dim_subsets_groups: None,
            load_ident_files: false,
            gsa_sample_file: None,
            posterior: None,
            external_sample: None,
        }
    }
}

const KNOWN: &[&str] = &[
    "parameter_set",
    "ar",
    "useautocorr",
    "prior_mc",
    "prior_range",
    "periods",
    "replic",
    "advanced",
    "normalize_jacobians",
    "grid_nbr",
    "tol_rank",
    "tol_deriv",
    "tol_sv",
    "schur_vec_tol",
    "no_identification_strength",
    "no_identification_reducedform",
    "no_identification_moments",
    "no_identification_minimal",
    "no_identification_spectrum",
    "max_dim_cova_group",
    "analytic_derivation_mode",
    "lik_init",
    "diffuse_filter",
    "kalman_tol",
    "prior_trunc",
    "nograph",
    "nodisplay",
    "graph_format",
    "order",
    "tex",
    "analytic_derivation",
    "kalman_algo",
    "use_univariate_filters_if_singularity_is_detected",
    "diffuse_kalman_tol",
    "checks_via_subsets",
    "max_dim_subsets_groups",
    "load_ident_files",
    "gsa_sample_file",
];

impl IdentOptions {
    pub fn from_options(o: &Options) -> Result<Self> {
        let num = |k: &str| match opt(o, k) {
            Some(OptValue::Num(v)) => Some(*v),
            _ => None,
        };
        let flag_or = |k: &str| match opt(o, k) {
            None => None,
            Some(OptValue::Num(v)) => Some(*v != 0.0),
            Some(_) => Some(true),
        };
        let mut s = IdentOptions::default();
        for (k, _) in o {
            if !KNOWN.contains(&k.as_str()) {
                println!("WARNING: identification option `{k}` is not supported by dynare-rust and is ignored.");
            }
        }
        if let Some(OptValue::Str(v)) = opt(o, "parameter_set") {
            s.parameter_set = v.clone();
        }
        if let Some(v) = num("ar") {
            s.ar = v as usize;
        }
        if let Some(v) = flag_or("useautocorr") {
            s.useautocorr = v;
        }
        if let Some(v) = num("prior_mc") {
            s.prior_mc = v as usize;
        }
        if let Some(v) = flag_or("prior_range") {
            s.prior_range = v;
        }
        if let Some(v) = num("periods") {
            s.periods = v as usize;
        }
        if let Some(v) = num("replic") {
            s.replic = v as usize;
        }
        if let Some(v) = flag_or("advanced") {
            s.advanced = v;
        }
        if let Some(v) = flag_or("normalize_jacobians") {
            s.normalize_jacobians = v;
        }
        if let Some(v) = num("grid_nbr") {
            s.grid_nbr = v as usize;
            if s.grid_nbr % 2 != 0 {
                s.grid_nbr += 1;
                println!("IDENTIFICATION: 'grid_nbr' needs to be even, hence it is reset to {}", s.grid_nbr);
            }
        }
        match opt(o, "tol_rank") {
            Some(OptValue::Num(v)) => s.tol_rank = Some(*v),
            Some(OptValue::Str(v)) if v == "robust" => s.tol_rank = None,
            _ => {}
        }
        if let Some(v) = num("tol_deriv") {
            s.tol_deriv = v;
        }
        if let Some(v) = num("tol_sv") {
            s.tol_sv = v;
        }
        if let Some(v) = num("schur_vec_tol") {
            s.schur_vec_tol = v;
        }
        // run.m: presence of the field sets the flag (whatever its value)
        s.no_strength = opt(o, "no_identification_strength").is_some();
        s.no_reducedform = opt(o, "no_identification_reducedform").is_some();
        s.no_moments = opt(o, "no_identification_moments").is_some();
        s.no_minimal = opt(o, "no_identification_minimal").is_some();
        s.no_spectrum = opt(o, "no_identification_spectrum").is_some();
        if !s.no_strength || s.advanced {
            s.no_reducedform = false;
            s.no_moments = false;
        }
        if let Some(v) = num("max_dim_cova_group") {
            s.max_dim_cova_group = v as usize;
        }
        if let Some(v) = num("analytic_derivation_mode") {
            s.analytic_derivation_mode = v as i32;
        }
        if opt(o, "diffuse_filter").is_some() {
            s.diffuse_filter = true;
            s.lik_init = 3;
        } else if let Some(v) = num("lik_init") {
            s.lik_init = v as u8;
        }
        if let Some(v) = num("kalman_tol") {
            s.kalman_tol = v;
        }
        if let Some(v) = num("kalman_algo") {
            s.kalman_algo = v as u8;
        }
        if let Some(v) = flag_or("use_univariate_filters_if_singularity_is_detected") {
            s.use_univariate_filters_if_singularity_is_detected = v;
        }
        if let Some(v) = num("diffuse_kalman_tol") {
            s.diffuse_kalman_tol = v;
        }
        s.tex = flag_or("tex").unwrap_or(false);
        s.prior_trunc = num("prior_trunc");
        s.nograph = opt(o, "nograph").is_some();
        match opt(o, "graph_format") {
            Some(OptValue::Str(f)) => s.graph_formats = GraphFormat::parse(f).into_iter().collect(),
            Some(OptValue::List(l)) => {
                s.graph_formats = l
                    .iter()
                    .filter_map(|v| if let OptValue::Str(f) = v { GraphFormat::parse(f) } else { None })
                    .collect()
            }
            _ => {}
        }
        // ---- order 2/3 (pruned state space, numerical Jacobians) ----
        if let Some(v) = num("order") {
            if !(1.0..=3.0).contains(&v) {
                bail!("identification: order must be 1, 2 or 3 (got {v})");
            }
            s.order = v as usize;
            if s.order > 1 && s.analytic_derivation_mode != -1 {
                println!(
                    "NOTE: identification(order={}): dynare-rust computes the Jacobians by two-sided finite differences (Dynare's analytic_derivation_mode = -1); Dynare's analytic derivatives agree to about 1e-6.",
                    s.order
                );
            }
        }
        s.checks_via_subsets = num("checks_via_subsets").is_some_and(|v| v != 0.0);
        s.max_dim_subsets_groups = num("max_dim_subsets_groups").map(|v| v as usize);
        s.load_ident_files = num("load_ident_files").is_some_and(|v| v != 0.0);
        s.gsa_sample_file = match opt(o, "gsa_sample_file") {
            Some(OptValue::Num(v)) if *v == 0.0 => None,
            Some(v) => Some(v.clone()),
            None => None,
        };
        Ok(s)
    }
}

/// Builds the parameter selection (run.m: the `estimated_params` block, or all stderr and model
/// parameters without one).
fn param_set(mf: &ModFile) -> Result<(ParamSet, bool)> {
    let m = &mf.model;
    let mut set = ParamSet::default();
    if mf.estimated_params.is_empty() {
        set.stderr = (0..m.n_exo()).collect();
        // M_.params: not the parameters standing for the varexo_det variables
        set.model = (0..m.n_user_params()).collect();
        set.names = m
            .exo_names
            .iter()
            .map(|e| format!("SE_{e}"))
            .chain(m.param_names[..m.n_user_params()].iter().cloned())
            .collect();
        return Ok((set, false));
    }
    let est = mf.estimated_params_dynare_order();
    for e in &est {
        match e.kind {
            EstimatedKind::StdErr(k) => {
                set.stderr.push(k);
                set.names.push(format!("SE_{}", m.exo_names[k]));
            }
            EstimatedKind::Corr(a, b) => {
                set.corr.push((a, b));
                set.names.push(format!("CC_{}_{}", m.exo_names[a], m.exo_names[b]));
            }
            EstimatedKind::Param(i) => {
                set.model.push(i);
                set.names.push(m.param_names[i].clone());
            }
            EstimatedKind::MeStdErr(_) | EstimatedKind::MeCorr(..) => {
                // run.m's message
                bail!(
                    "Identification does not (yet) support measurement errors. Instead, define them explicitly as varexo and provide measurement equations in the model definition."
                )
            }
            EstimatedKind::Skew(_) => {
                bail!("identification: skew parameters in estimated_params are not supported by dynare-rust")
            }
            EstimatedKind::InitState(_) => bail!("identification: estimated initial states are not supported"), // ---- estimation completion
        }
    }
    Ok((set, true))
}

fn hms(secs: f64) -> String {
    let s = secs.round() as u64;
    format!("{}h{:02}m{:02}s", s / 3600, (s / 60) % 60, s % 60)
}

/// `get_error_message(info)` for the codes the solver returns.
fn info_message(info: i32) -> &'static str {
    crate::solver::describe(info)
}

fn criterion(b: &Option<IdeBlock>) -> Option<Criterion<'_>> {
    b.as_ref().map(|b| Criterion { checks: vec![&b.check], minimal_state_space: None, subsets: None })
}

fn display_point(an: &Analysis, names: &[String], o: &IdentOptions) {
    let minimal = if o.no_minimal || !an.minimal_state_space {
        None
    } else {
        an.minimal.as_ref().map(|b| Criterion {
            checks: vec![&b.check],
            minimal_state_space: Some(vec![true]),
            subsets: None,
        })
    };
    let mut crits = [criterion(&an.reducedform), minimal, criterion(&an.spectrum), criterion(&an.moments)];
    // checks_via_subsets
    if let Some(sb) = &an.subsets {
        for (c, r) in crits.iter_mut().zip([&sb.reducedform, &sb.minimal, &sb.spectrum, &sb.moments]) {
            if let (Some(c), Some(r)) = (c.as_mut(), r.as_ref()) {
                c.subsets = Some(r);
            }
        }
    }
    display(1, crits, names, o, an.bruteforce.as_ref());
}

fn mat_json(m: &Mat) -> serde_json::Value {
    let rows: Vec<Vec<f64>> = (0..m.nrows()).map(|i| m.row(i).iter().copied().collect()).collect();
    json!(rows)
}

fn check_json(c: &CheckResult) -> serde_json::Value {
    json!({
        "cond": c.cond, "rank": c.rank, "ind0": c.ind0, "indno": c.indno, "ino": c.ino,
        "Mco": c.mco, "Pco": mat_json(&c.pco), "jweak": c.jweak, "jweak_pair": c.jweak_pair,
    })
}

fn block_json(b: &IdeBlock) -> serde_json::Value {
    json!({
        "jacobian": mat_json(&b.jacobian), "ind": b.ind, "values": b.values, "norm": b.norm,
        "checks": check_json(&b.check),
    })
}

/// The point analysis as JSON (`<fname>/identification/<fname>_identif.json`, the counterpart
/// of Dynare's `_identif.mat`).
pub fn analysis_json(an: &Analysis, names: &[String]) -> serde_json::Value {
    let mut v = json!({
        "names": names,
        "params": an.theta,
        "ar": an.nlags,
        "dynamic": block_json(&an.dynamic),
        "minimal_state_space": an.minimal_state_space,
        "moments_S": an.moments_s,
        "moments_V": mat_json(&an.moments_v),
    });
    let o = v.as_object_mut().unwrap();
    if let Some(b) = &an.reducedform {
        o.insert("reducedform".into(), block_json(b));
    }
    if let Some(b) = &an.moments {
        o.insert("moments".into(), block_json(b));
    }
    if let Some(b) = &an.spectrum {
        o.insert("spectrum".into(), block_json(b));
    }
    if let Some(b) = &an.minimal {
        o.insert("minimal".into(), block_json(b));
    }
    if let Some(s) = &an.strength {
        o.insert(
            "strength".into(),
            json!({
                "AHess": mat_json(&s.ahess), "ide_strength_dMOMENTS": s.ide_strength,
                "ide_strength_dMOMENTS_prior": s.ide_strength_prior, "deltaM": s.delta_m,
                "deltaM_prior": s.delta_m_prior, "identified_parameter_indices": s.identified,
                "flag_score": s.flag_score, "si_dMOMENTSnorm": s.si_dmoments_norm,
                "si_dREDUCEDFORMnorm": s.si_dreducedform_norm, "si_dDYNAMICnorm": s.si_ddynamic_norm,
                "checks": check_json(&s.check),
            }),
        );
    }
    if let Some((pars, cosn)) = &an.bruteforce {
        o.insert("pars".into(), json!(pars));
        o.insert("cosndMOMENTS".into(), mat_json(cosn));
    }
    v
}

/// Builds the identification context of a .mod file (dynare_estimation_init + the parameter
/// selection of run.m).
pub fn build_context<'a>(mf: &'a ModFile, opts: IdentOptions) -> Result<Context<'a>> {
    if mf.varobs.is_empty() {
        bail!("identification: no varobs block (the observed variables are required)");
    }
    let (set, prior_exist) = param_set(mf)?;
    let mut dsge = DsgeModel::new(mf, None).context("identification: dynare_estimation_init")?;
    let prior_sd = if prior_exist { dsge.prior_sd() } else { vec![f64::NAN; set.len()] }; // NaN under ML (stream M)
    let varobs = dsge.varobs.clone();
    // select_qz_criterium_value.m: 1-1e-6 for lik_init = 1 (unit roots then violate
    // Blanchard-Kahn), 1+1e-6 for every other initialization
    let qz_criterium = if opts.lik_init == 1 { 1.0 - 1e-6 } else { 1.0 + 1e-6 };
    let sopts = SolverOptions { qz_criterium, ..SolverOptions::default() };
    dsge.set_solver_options(sopts);
    let solver = FirstOrderSolver::new(&mf.model.lead_lag_incidence, mf.model.n_exo(), sopts);
    let pderivs = ModelParamDerivs::new(&mf.model).at_exo_steady_state(&dsge.exo_ss);
    let endo_histval = crate::analysis::stochastic_histval_of(&mf.model, &mf.histval);
    Ok(Context {
        model: &mf.model,
        dsge,
        solver,
        pderivs,
        set,
        prior_exist,
        prior_sd,
        varobs,
        qz_criterium,
        opts,
        endo_histval,
        analytic: std::sync::OnceLock::new(),
    })
}

/// `options_ident.name_tex` (run.m): `get_the_name(k, TeX, ...)` of the `estimated_params`
/// entries, or `$ SE_{e} $` and `$ p $` for all shocks and parameters without the block.
fn name_tex(mf: &ModFile, set: &ParamSet, prior_exist: bool) -> Vec<String> {
    if prior_exist {
        return mf.estimated_params_dynare_order().iter().map(|e| crate::latex::get_the_tex_name(mf, e)).collect();
    }
    let se = set.stderr.iter().map(|&k| format!("$ SE_{{{}}} $", crate::latex::exo_tex(mf, k)));
    se.chain(set.model.iter().map(|&i| format!("$ {} $", crate::latex::param_tex(mf, i)))).collect()
}

/// The TeX tables of bruteforce.m (`options_.TeX`): `<fname>_collin_patterns_<tittxt>_<ll>.tex`
/// in the identification folder, one per group size.
fn write_collin_patterns_tex(
    ident_dir: &Path,
    fname: &str,
    pars: &[Vec<Vec<Option<usize>>>],
    cosn: &Mat,
    name_tex: &[String],
    tittxt: &str,
) -> Result<()> {
    let tittxt1 = tittxt.replace(' ', "_").replace('.', "");
    for ll in 0..cosn.ncols() {
        let l = ll + 1;
        let mut s = String::new();
        s += "% TeX-table generated by ident_bruteforce (Dynare).\n";
        s += &format!("% Collinearity patterns with {l} parameter(s): {tittxt}\n");
        s += &format!("% {} \n \n", crate::latex::datestr_now());
        s += "{\\tiny \n";
        s += "\\begin{longtable}{llc} \n";
        s += &format!("\\caption{{Collinearity patterns with {l} parameter(s): {tittxt}}}\n ");
        s += &format!("\\label{{Table:CollinearityPatterns:{tittxt1}:{l}}}\\\\\n");
        s += "\\toprule \n";
        s += "  Parameter & Explanatory & cosn \\\\ \n";
        s += "            & parameter(s)   &  \\\\ \n";
        s += "\\midrule \\endfirsthead \n";
        s += "\\caption{(continued)}\\\\\n ";
        s += "\\bottomrule \n";
        s += "  Parameter & Explanatory & cosn \\\\ \n";
        s += "            & parameter(s)   &  \\\\ \n";
        s += "\\midrule \\endhead \n";
        s += "\\bottomrule \\multicolumn{3}{r}{(Continued on next page)}\\endfoot \n";
        s += "\\bottomrule\\endlastfoot \n";
        for (i, row) in pars.iter().enumerate() {
            let mut plist = String::new();
            for p in &row[ll] {
                match p {
                    Some(k) => plist += &format!(" {}\\;\\;  ", name_tex[*k]),
                    None => plist += " ---- ",
                }
            }
            s += &format!("{} & [{plist}] & {} \\\\ \n", name_tex[i], crate::latex::fmt_f(cosn[(i, ll)], 7, 3));
        }
        s += "\\bottomrule \n";
        s += "\\end{longtable}\n";
        s += "} \n";
        s += "% End of TeX file.\n";
        std::fs::write(ident_dir.join(format!("{fname}_collin_patterns_{tittxt1}_{l}.tex")), s)?;
    }
    Ok(())
}

impl Context<'_> {
    /// The parameter vector of `parameter_set` and its title (`parameters`, `parameters_TeX`).
    pub fn point_vector(&self) -> Result<(Vec<f64>, String, String)> {
        if !self.prior_exist {
            println!("Testing all current stderr and model parameter values");
            let mut v: Vec<f64> = (0..self.model.n_exo()).map(|i| self.dsge.base_sigma_e[(i, i)].sqrt()).collect();
            v.extend(self.dsge.base_params.iter().copied());
            return Ok((v, "Current_params".into(), "Current parameter values".into()));
        }
        if self.dsge.ml {
            // all(bayestopt_.pshape == 0): only bounds in estimated_params, whatever parameter_set
            // says, the starting values (set_prior's xparam1)
            println!("Testing ML Starting value");
            return Ok((self.dsge.init_vector()?, "ML_Starting_value".into(), "ML starting value".into()));
        }
        let (v, tex) = match self.opts.parameter_set.as_str() {
            "calibration" => {
                println!("Testing calibration");
                // get_all_parameters(estim_params_, M_): the calibration (shocks block and
                // parameter values), not the init column of estimated_params
                let (p, q) = (&self.dsge.base_params, &self.dsge.base_sigma_e);
                let v = self
                    .dsge
                    .estimated
                    .iter()
                    .map(|e| match e.kind {
                        EstimatedKind::Param(i) => p[i],
                        EstimatedKind::StdErr(i) => q[(i, i)].sqrt(),
                        EstimatedKind::Corr(i, j) => q[(i, j)] / (q[(i, i)] * q[(j, j)]).sqrt(),
                        // rejected by param_set before we get here
                        EstimatedKind::MeStdErr(_)
                        | EstimatedKind::MeCorr(..)
                        | EstimatedKind::Skew(_)
                        | EstimatedKind::InitState(_) => f64::NAN,
                    })
                    .collect();
                (v, "Calibration")
            }
            "prior_mode" => {
                println!("Testing prior mode");
                (self.dsge.priors.iter().map(|p| p.p5).collect(), "Prior mode")
            }
            "prior_mean" => {
                println!("Testing prior mean");
                (self.dsge.priors.iter().map(|p| p.mean).collect(), "Prior mean")
            }
            // ---- get_posterior_parameters(type, ...) of the previous estimation ----
            s @ ("posterior_mode" | "posterior_mean" | "posterior_median") => {
                let post = self.opts.posterior.as_ref();
                let (v, tex, what) = match s {
                    "posterior_mode" => (post.map(|p| p.mode.clone()), "Posterior mode", "mode"),
                    "posterior_mean" => (post.and_then(|p| p.mean.clone()), "Posterior mean", "mean"),
                    _ => (post.and_then(|p| p.median.clone()), "Posterior median", "median"),
                };
                let Some(v) = v else {
                    bail!(
                        "get_posterior_parameters: the posterior {what} is not available (run an estimation{} first)",
                        if what == "mode" { "" } else { " with mh_replic > 0" }
                    )
                };
                println!("Testing posterior {what}");
                (v, tex)
            }
            s => bail!(
                "The option parameter_set has to be equal to: 'calibration', 'posterior_mode', 'posterior_mean', 'posterior_median', 'prior_mode' or 'prior_mean' (got '{s}')."
            ),
        };
        Ok((v, self.opts.parameter_set.clone(), tex.into()))
    }

    /// `prior_bounds(bayestopt_, options_.prior_trunc)`.
    pub(super) fn prior_bounds(&self) -> Bounds {
        prior_bounds(&self.dsge.priors, self.opts.prior_trunc.unwrap_or(1e-10))
    }

    /// `Prior.draw()` (or a uniform draw over the prior bounds with `prior_range`).
    fn draw<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let b = self.prior_bounds();
        if self.opts.prior_range {
            b.lb.iter().zip(&b.ub).map(|(l, u)| l + (u - l) * rng.r#gen::<f64>()).collect()
        } else {
            prior_draw(&self.dsge.priors, &b, rng)
        }
    }
}

/// Per-draw results of the Monte Carlo analysis.
#[derive(Default)]
struct McStore {
    pdraws: Vec<Vec<f64>>,
    reducedform: Vec<CheckResult>,
    moments: Vec<CheckResult>,
    spectrum: Vec<CheckResult>,
    minimal: Vec<CheckResult>,
    minimal_state_space: Vec<bool>,
    moments_values: Vec<Vec<f64>>,
    moments_si: Vec<Mat>,
}

/// Port of `identification.run` (called by the driver for `identification(...)`).
pub fn run(mf: &ModFile, dir: &Path, fname: &str, o: &Options, seed: u64) -> Result<()> {
    run_q2(mf, dir, fname, o, seed, None, 100)
}

/// [`run`] with the posterior points of an earlier estimation (`parameter_set =
/// posterior_mode | posterior_mean | posterior_median`).
pub fn run_q2(
    mf: &ModFile,
    dir: &Path,
    fname: &str,
    o: &Options,
    seed: u64,
    posterior: Option<PosteriorPoints>,
    options_drop: usize,
) -> Result<()> {
    run_impl(mf, dir, fname, o, seed, posterior, None, options_drop)
}

/// `identification.run(M_, ..., options_ident, pdraws0)` on an external sample
/// (xparam1 vectors), as `sensitivity(identification=1, morris=2)` calls it.
pub fn run_external(
    mf: &ModFile,
    dir: &Path,
    fname: &str,
    o: &Options,
    seed: u64,
    sample: Vec<Vec<f64>>,
) -> Result<()> {
    run_impl(mf, dir, fname, o, seed, None, Some(sample), 100)
}

#[allow(clippy::too_many_arguments)]
fn run_impl(
    mf: &ModFile,
    dir: &Path,
    fname: &str,
    o: &Options,
    seed: u64,
    posterior: Option<PosteriorPoints>,
    sample: Option<Vec<Vec<f64>>>,
    options_drop: usize,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    let mut opts = IdentOptions::from_options(o)?;
    opts.drop = options_drop;
    opts.posterior = posterior;
    opts.external_sample = sample;
    let mut ctx = build_context(mf, opts)?;
    ctx.opts.name_tex = name_tex(mf, &ctx.set, ctx.prior_exist);
    let names = ctx.set.names.clone();
    let tot = names.len();
    let ident_dir = dir.join(fname).join("identification");
    std::fs::create_dir_all(&ident_dir)?;
    let go =
        GraphOptions { enabled: !ctx.opts.nograph, formats: ctx.opts.graph_formats.clone(), dir: ident_dir.clone() };
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x1de7_1f1c);

    // ---- run.m's notes on measurement errors and external samples ----
    if !ctx.prior_exist && mf.h.iter().any(|v| *v != 0.0) {
        println!(
            "\nidentification.run:: Identification does not support measurement errors (yet) and will ignore them in the following. To test their identifiability, instead define them explicitly as varexo and provide measurement equations in the model definition."
        );
    }
    let external = match ctx.opts.external_sample.take().map(Ok).or_else(|| ctx.opts.gsa_sample_file.clone().map(Err)) {
        Some(Ok(s)) => {
            ctx.opts.prior_mc = s.len();
            ctx.opts.load_ident_files = false;
            Some(s)
        }
        Some(Err(spec)) => {
            let s = super::files::gsa_sample(dir, fname, &spec)?;
            ctx.opts.prior_mc = s.len();
            ctx.opts.load_ident_files = false;
            Some(s)
        }
        None => None,
    };
    println!("\n======== Identification Analysis ========");
    if ctx.opts.order > 1 {
        println!("Based on Pruned State Space System (order={})", ctx.opts.order);
    }
    println!();
    if tot < 2 {
        ctx.opts.advanced = false;
        println!("There is only one parameter to study for identitification. The advanced option is re-set to 0.");
    }
    ctx.opts.max_dim_cova_group = ctx.opts.max_dim_cova_group.min(tot.saturating_sub(1));
    if !ctx.prior_exist && external.is_none() {
        ctx.opts.prior_mc = 1;
        ctx.opts.prior_range = false;
    } else if ctx.dsge.ml && external.is_none() {
        ctx.opts.prior_mc = 1; // no prior to draw from: a single point
    }
    let has_priors = ctx.prior_exist && !ctx.dsge.ml; // any(bayestopt_.pshape > 0)
    let sample_size = ctx.opts.prior_mc;
    let (mut theta, mut parameters, mut parameters_tex) = ctx.point_vector()?;
    // ---- load_ident_files: display the saved point analysis ----
    if ctx.opts.load_ident_files {
        if super::files::display_saved(&ident_dir, fname, &names, &ctx.opts)? {
            if sample_size > 1 {
                println!(
                    "NOTE: load_ident_files: the saved Monte Carlo results ({fname}_identif_mc.json) are not re-displayed by dynare-rust."
                );
            }
            println!("\n==== Identification analysis completed ====\n");
            return Ok(());
        }
        println!(
            "WARNING: load_ident_files: {} not found; computing the analysis.",
            ident_dir.join(format!("{fname}_identif.json")).display()
        );
    }
    let mut pers = Persistent::default();
    let mut an = ctx.analysis(&theta, ctx.opts.advanced, true, &mut pers, &mut rng)?;
    if let Err(info) = an {
        println!("-----------");
        println!("The model does not solve for {parameters} (info = {info}: {})", info_message(info));
        println!("-----------");
        if has_priors {
            println!("Try sampling up to 50 parameter sets from the prior.");
            let mut kk = 0;
            while kk < 50 && an.is_err() {
                kk += 1;
                theta = ctx.draw(&mut rng);
                an = ctx.analysis(&theta, ctx.opts.advanced, true, &mut pers, &mut rng)?;
            }
        }
        if an.is_err() {
            println!("\n-----------");
            println!("Identification stopped:");
            if has_priors {
                println!("The model did not solve for any of 50 attempts of random samples from the prior");
            }
            println!("-----------");
            return Ok(());
        }
        println!("Found a random draw from the priors that solves the model:");
        for (n, v) in names.iter().zip(&theta) {
            println!("{n:<w$} {v:10.6}", w = names.iter().map(|s| s.len()).max().unwrap_or(4) + 2);
        }
        println!("Identification now continues for this draw.");
        parameters = "Random_prior_params".into();
        parameters_tex = "Random prior parameter draw".into();
    }
    let mut an = an.unwrap();
    // ---- identification.checks_via_subsets (point analysis only, as Dynare) ----
    if ctx.opts.checks_via_subsets {
        let md = ctx.opts.max_dim_subsets_groups.unwrap_or(4.min(tot.saturating_sub(1)));
        an.subsets = Some(super::subsets::checks_via_subsets(&an, tot, ctx.set.n_model(), ctx.opts.tol_rank, md));
    }
    display_point(&an, &names, &ctx.opts);
    if let (true, Some((pars, cosn))) = (ctx.opts.tex, &an.bruteforce) {
        // options_ident.tittxt of the last analysis call
        write_collin_patterns_tex(&ident_dir, fname, pars, cosn, &ctx.opts.name_tex, &parameters)?;
    }
    let mut point_json = analysis_json(&an, &names);
    point_json["parameter_set"] = json!(parameters); // load_ident_files
    std::fs::write(ident_dir.join(format!("{fname}_identif.json")), serde_json::to_string(&point_json)?)?;
    std::fs::write(ident_dir.join(format!("{fname}_{parameters}_identif.json")), serde_json::to_string(&point_json)?)?;
    if !ctx.opts.no_strength && !ctx.opts.nograph && !an.error_strength && !an.error_moments {
        let tittxt = parameters.clone();
        let figs = plots::point_figures(&an, &names, &tittxt, ctx.opts.advanced);
        plots::save_all(&figs, fname, &go)?;
        if ctx.opts.tex {
            plots::write_eps_loaders(&figs, fname, &go, &tittxt, &parameters_tex)?;
        }
    }

    if sample_size > 1 {
        println!("\nMonte Carlo Testing");
        let mut mc_opts = ctx.opts.clone();
        mc_opts.advanced = false;
        mc_opts.checks_via_subsets = false; // run.m: not supported in the Monte Carlo analysis
        ctx.opts.checks_via_subsets = false;
        let saved = std::mem::replace(&mut ctx.opts, mc_opts);
        let mut st = McStore::default();
        let mut attempts = 0usize;
        while st.pdraws.len() < sample_size {
            attempts += 1;
            if attempts > 100 * sample_size {
                println!(
                    "WARNING: too many prior draws do not solve the model; stopping the Monte Carlo after {} draws.",
                    st.pdraws.len()
                );
                break;
            }
            // gsa_sample_file draws come from the GSA sample, in order
            let draw = match &external {
                Some(ext) => match ext.get(attempts - 1) {
                    Some(d) => d.clone(),
                    None => break,
                },
                None => ctx.draw(&mut rng),
            };
            let r = ctx.analysis(&draw, false, false, &mut pers, &mut rng)?;
            let Ok(a) = r else { continue };
            st.pdraws.push(draw);
            if let Some(b) = &a.reducedform {
                st.reducedform.push(b.check.clone());
            }
            if let Some(b) = &a.moments {
                st.moments.push(b.check.clone());
                st.moments_values.push(b.values.clone());
                st.moments_si.push(b.si.clone());
            }
            if let Some(b) = &a.spectrum {
                st.spectrum.push(b.check.clone());
            }
            if !ctx.opts.no_minimal {
                st.minimal_state_space.push(a.minimal_state_space);
                if let Some(b) = &a.minimal {
                    st.minimal.push(b.check.clone());
                }
            }
            if st.pdraws.len() % 10 == 0 {
                println!("MC identification checks {}/{}", st.pdraws.len(), sample_size);
            }
        }
        ctx.opts = saved;
        let n_eff = st.pdraws.len();
        println!("\nTesting MC sample");
        let mut dopts = ctx.opts.clone();
        dopts.advanced = false;
        fn crit(v: &[CheckResult]) -> Option<Criterion<'_>> {
            if v.is_empty() {
                None
            } else {
                Some(Criterion { checks: v.iter().collect(), minimal_state_space: None, subsets: None })
            }
        }
        // display.m skips the Komunjer-Ng criterion of the MC sample: IDE_MINIMAL has no
        // `dMINIMAL` field (the per-draw results are still saved)
        display(n_eff, [crit(&st.reducedform), None, crit(&st.spectrum), crit(&st.moments)], &names, &dopts, None);

        // si_dMOMENTSnorm over the MC sample (normalized by the std of moments and draws)
        let si_norm = mc_moment_sensitivities(&st);
        if !ctx.opts.nograph {
            if let (Some(si), Some(s)) = (&si_norm, an.strength.as_ref()) {
                plots::save_all(&plots::mc_figures(si, Some(&s.ide_strength), &names), fname, &go)?;
            }
        }
        let mc_json = json!({
            "pdraws": st.pdraws,
            "moments": st.moments.iter().map(check_json).collect::<Vec<_>>(),
            "reducedform": st.reducedform.iter().map(check_json).collect::<Vec<_>>(),
            "spectrum": st.spectrum.iter().map(check_json).collect::<Vec<_>>(),
            "minimal": st.minimal.iter().map(check_json).collect::<Vec<_>>(),
            "minimal_state_space": st.minimal_state_space,
            "si_dMOMENTSnorm": si_norm.as_ref().map(mat_json),
        });
        std::fs::write(ident_dir.join(format!("{fname}_identif_mc.json")), serde_json::to_string(&mc_json)?)?;

        if ctx.opts.advanced && !st.moments.is_empty() {
            advanced_mc(&ctx, &st, &names, fname, &go, &mut pers, &mut rng)?;
        }
    }
    println!("\n==== Identification analysis completed ====\n");
    println!("(identification: {})", hms(t0.elapsed().as_secs_f64()));
    Ok(())
}

/// `si_dMOMENTSnorm` of the MC sample: `vnorm(si_dMOMENTS ./ std(MOMENTS)) .* std(pdraws)`.
fn mc_moment_sensitivities(st: &McStore) -> Option<Mat> {
    let n = st.moments_si.len();
    if n < 2 {
        return None;
    }
    let tot = st.pdraws[0].len();
    let nm = st.moments_values[0].len();
    let std_of = |vals: &dyn Fn(usize) -> f64| -> f64 {
        let mu = (0..n).map(vals).sum::<f64>() / n as f64;
        ((0..n).map(|k| (vals(k) - mu).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt()
    };
    let sd_m: Vec<f64> = (0..nm).map(|r| std_of(&|k| st.moments_values[k][r])).collect();
    let sd_p: Vec<f64> = (0..tot).map(|j| std_of(&|k| st.pdraws[k][j])).collect();
    let mut out = Mat::zeros(n, tot);
    for k in 0..n {
        let si = &st.moments_si[k];
        let q = Mat::from_fn(si.nrows(), si.ncols(), |i, j| si[(i, j)] / sd_m[i]);
        let v = vnorm(&q);
        for j in 0..tot {
            out[(k, j)] = v[j] * sd_p[j];
        }
    }
    Some(out)
}

/// The `advanced` part of the MC analysis: the rank-deficient draws, or the draws with the
/// highest and the smallest condition number of J, analysed with `advanced = 1`.
#[allow(clippy::too_many_arguments)]
fn advanced_mc<R: Rng + ?Sized>(
    ctx: &Context,
    st: &McStore,
    names: &[String],
    fname: &str,
    go: &GraphOptions,
    pers: &mut Persistent,
    rng: &mut R,
) -> Result<()> {
    let jcrit: Vec<usize> = (0..st.moments.len()).filter(|&k| st.moments[k].ino > 0).collect();
    if jcrit.len() >= st.moments.len() {
        return Ok(());
    }
    let mut runs: Vec<(usize, String)> = Vec::new();
    if jcrit.is_empty() {
        let conds: Vec<f64> = st.moments.iter().map(|c| c.cond).collect();
        let jmax = (0..conds.len()).max_by(|&a, &b| conds[a].total_cmp(&conds[b])).unwrap();
        let jmin = (0..conds.len()).min_by(|&a, &b| conds[a].total_cmp(&conds[b])).unwrap();
        runs.push((jmax, "Draw with HIGHEST condition number".into()));
        runs.push((jmin, "Draw with SMALLEST condition number".into()));
    } else {
        for (j, &k) in jcrit.iter().enumerate() {
            runs.push((k, format!("Rank deficient draw nr. {}", j + 1)));
        }
    }
    for (k, tittxt) in runs {
        println!("\nTesting {tittxt}.");
        let an = ctx
            .analysis(&st.pdraws[k], true, true, pers, rng)?
            .map_err(|info| anyhow!("identification: the model does not solve at MC draw {k} (info {info})"))?;
        if let (true, Some((pars, cosn))) = (ctx.opts.tex, &an.bruteforce) {
            write_collin_patterns_tex(&go.dir, fname, pars, cosn, &ctx.opts.name_tex, &tittxt)?;
        }
        let mut o = ctx.opts.clone();
        o.advanced = true;
        display_point(&an, names, &o);

        if !ctx.opts.nograph && !an.error_strength && !an.error_moments {
            let figs = plots::point_figures(&an, names, &tittxt, true);
            plots::save_all(&figs, fname, go)?;
            if ctx.opts.tex {
                plots::write_eps_loaders(&figs, fname, go, &tittxt, &tittxt)?;
            }
        }
    }
    Ok(())
}
