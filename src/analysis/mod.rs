// SPDX-License-Identifier: GPL-3.0-or-later
//! `stoch_simul` (order 1), `check` and the theoretical analysis of a linear(ised) model:
//! decision-rule printout (`disp_dr.m`), IRFs (`irf.m`), theoretical moments with unit roots
//! (`th_autocovariances.m`, `disp_th_moments.m`), variance decompositions, simulated moments
//! (`simult.m`, `disp_moments.m`), and their Bayesian counterparts (`PosteriorIRF.m`,
//! `moments_varendo`) in [`posterior`].
//!
//! Dynare's options are global: a `stoch_simul` option stays set for the following commands
//! (e.g. `hp_filter`); [`StochSimulOptions`] is kept by the driver across commands and each
//! command's options are applied on top ([`StochSimulOptions::apply`]).

pub mod check;
pub mod irf;
pub mod moments;
pub mod posterior;
pub mod q2; // filters, spectral density, loglinear, TeX tables
pub mod realtime_decomposition; // stream C
pub mod shock_decomposition; // stream F
pub mod simulate;
pub mod skew;
pub mod table;
pub mod vardec; // ---- shocks/shock_paths: Skew_e in stoch_simul, M_.H table
// ---- stoch_simul ----
pub mod aim; // aim_solver / dr=aim (AIM_first_order_solver.m, matlab/AIM)
pub mod partial_info; // partial_information (matlab/partial_information)
// ---- end stoch_simul ----

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use serde_json::json;

use crate::graphics::{GraphFormat, GraphOptions};
use crate::linalg::Mat;
use crate::model::Model;
use crate::modfile::{ModFile, OptValue, Options};
use crate::solver::{DecisionRule, FirstOrderSolver, SolveError, SolverOptions, describe};
use moments::{MomentOptions, TheoreticalMoments};
use simulate::SimulatedMoments;
use table::{dyntable, fmt_e, fmt_f, max_len};

/// The `options_` fields used by stoch_simul and the theoretical analysis (Dynare defaults).
#[derive(Clone, Debug)]
pub struct StochSimulOptions {
    pub order: usize,
    /// `irf` (40); 0 disables IRFs.
    pub irf: usize,
    pub irf_shocks: Option<Vec<String>>,
    pub relative_irf: bool,
    /// `impulse_responses.plot_threshold` (`irf_plot_threshold`, 1e-10).
    pub irf_plot_threshold: f64,
    pub nograph: bool,
    pub graph_format: Vec<GraphFormat>,
    pub noprint: bool,
    pub nomoments: bool,
    pub nocorr: bool,
    pub nofunctions: bool,
    pub nodecomposition: bool,
    pub nomodelsummary: bool,
    pub ar: usize,
    pub hp_filter: f64,
    pub periods: usize,
    pub drop: usize,
    pub conditional_variance_decomposition: Vec<usize>,
    pub qz_criterium: f64,
    pub dr_display_tol: f64,
    pub schur_vec_tol: f64,
    pub lyapunov_complex_threshold: f64,
    pub filtered_theoretical_moments_grid: usize,
    pub add_tiny_number_to_cholesky: f64,
    /// Estimation: `bayesian_irf`, `moments_varendo`, `sub_draws`, `mh_conf_sig`.
    pub bayesian_irf: bool,
    pub moments_varendo: bool,
    pub sub_draws: Option<usize>,
    pub mh_conf_sig: f64,
    /// `loglinear` (persistent; decision rules for the logs, order 1 only).
    pub loglinear: bool,
    /// `one_sided_hp_filter` (0 = off).
    pub one_sided_hp_filter: f64,
    /// `bandpass_filter`: `options_.bandpass.passband` when `options_.bandpass.indicator`.
    pub bandpass: Option<[f64; 2]>,
    /// `options_.bandpass.K` (12).
    pub bandpass_k: usize,
    /// `spectral_density` (`options_.SpectralDensity.trigger`).
    pub spectral_density: bool,
    /// `contemporaneous_correlation` (stores `oo_.contemporaneous_correlation`).
    pub contemporaneous_correlation: bool,
    // ---- estimation completion ----
    /// `options_.estimation.moments_posterior_density.indicator` (`no_posterior_kernel_density`
    /// clears it): kernel density estimates of the posterior theoretical moments.
    pub posterior_kernel_density: bool,
    // ---- end estimation completion ----
    /// `dr=cycle_reduction|logarithmic_reduction|default`.
    pub dr_cycle_reduction: bool,
    pub dr_logarithmic_reduction: bool,
    pub dr_cycle_reduction_tol: f64,
    pub dr_cycle_reduction_maxiter: usize,
    pub dr_logarithmic_reduction_tol: f64,
    pub dr_logarithmic_reduction_maxiter: usize,
    /// `qz_zero_threshold` (1e-6).
    pub qz_zero_threshold: f64,
    /// `aim_solver`: solved with the QZ (dynare-rust has no AIM; same solution when it exists).
    pub aim_solver: bool,
    /// `partial_information` (not ported: dynare-rust stops with an error).
    pub partial_information: bool,
    /// `tex` (`options_.TeX`, persistent).
    pub tex: bool,
    /// `pruning`, `replic` (order >= 2).
    pub pruning: bool,
    pub replic: usize,
    /// ---- shocks/shock_paths: `M_.Skew_e`, `M_.H`, varobs (set by `run_stoch_simul`)
    pub shocks_info: Option<skew::ShocksInfo>,
    // ---- stoch_simul ----
    /// `k_order_solver` (persistent; implied by `order >= 3`, as the preprocessor writes it).
    pub k_order_solver: bool,
    /// `simul_replic` (1): extra simulated series saved to `Output/<fname>_simul`.
    pub simul_replic: usize,
    /// `solve_algo` given to stoch_simul / estimation (`options_.solve_algo`; None: the
    /// `steady` command's).
    pub solve_algo: Option<u8>,
    /// `M_.endo_histval(:, end)` (levels) when the file has a `histval` block: the starting
    /// point of the simulation ([`stochastic_histval_of`]; set by [`run_stoch_simul`]).
    pub endo_histval: Option<Vec<f64>>,
    /// `M_.H` and the observables when the file declares measurement errors (set by
    /// [`run_stoch_simul`]): MODEL SUMMARY and the decompositions WITH MEASUREMENT ERROR.
    pub me: Option<MeasurementErrors>,
    // ---- end stoch_simul ----
}

impl Default for StochSimulOptions {
    fn default() -> Self {
        StochSimulOptions {
            order: 2, // Dynare's default; `model(linear)` forces 1
            irf: 40,
            irf_shocks: None,
            relative_irf: false,
            irf_plot_threshold: 1e-10,
            nograph: false,
            graph_format: vec![GraphFormat::Pdf],
            noprint: false,
            nomoments: false,
            nocorr: false,
            nofunctions: false,
            nodecomposition: false,
            nomodelsummary: false,
            ar: 5,
            hp_filter: 0.0,
            periods: 0,
            drop: 100,
            conditional_variance_decomposition: Vec::new(),
            qz_criterium: 1.0 + 1e-6,
            dr_display_tol: 1e-6,
            schur_vec_tol: 1e-11,
            lyapunov_complex_threshold: 1e-15,
            filtered_theoretical_moments_grid: 512,
            add_tiny_number_to_cholesky: 1e-14,
            bayesian_irf: false,
            moments_varendo: false,
            sub_draws: None,
            mh_conf_sig: 0.9,
            loglinear: false,
            one_sided_hp_filter: 0.0,
            bandpass: None,
            bandpass_k: 12,
            spectral_density: false,
            contemporaneous_correlation: false,
            posterior_kernel_density: true, // ---- estimation completion ----
            dr_cycle_reduction: false,
            dr_logarithmic_reduction: false,
            dr_cycle_reduction_tol: 1e-7,
            dr_cycle_reduction_maxiter: 100,
            dr_logarithmic_reduction_tol: 1e-12,
            dr_logarithmic_reduction_maxiter: 100,
            qz_zero_threshold: 1e-6,
            aim_solver: false,
            partial_information: false,
            tex: false,
            pruning: false,
            replic: 50,
            shocks_info: None,
            k_order_solver: false, // stoch_simul
            simul_replic: 1,
            solve_algo: None,
            endo_histval: None,
            me: None,
        }
    }
}

/// Options that only concern display or are irrelevant at order 1; accepted silently.
const IGNORED: &[&str] = &["nodisplay", "console", "nointeractive"];

fn to_usize(v: &OptValue) -> Option<usize> {
    match v {
        OptValue::Num(x) if *x >= 0.0 && x.fract() == 0.0 && *x <= crate::modfile::option_check::MAX_HORIZON => {
            Some(*x as usize)
        }
        _ => None,
    }
}

/// `[1 4 8]`, `4`, `1:4`, `(1, 4)` -> steps.
fn steps(v: &OptValue) -> Option<Vec<usize>> {
    match v {
        OptValue::Num(x) => Some(vec![*x as usize]),
        OptValue::Vec(x) => Some(x.iter().map(|&s| s as usize).collect()),
        OptValue::List(items) => items.iter().map(to_usize).collect(),
        OptValue::Str(s) => {
            let parts: Vec<&str> = s.split(':').map(str::trim).collect();
            let nums: Option<Vec<f64>> = parts.iter().map(|p| p.parse().ok()).collect();
            match nums?.as_slice() {
                [a, b] => Some((*a as usize..=*b as usize).collect()),
                [a, st, b] => {
                    let mut out = vec![];
                    let mut x = *a;
                    while x <= *b + 1e-12 {
                        out.push(x as usize);
                        x += st;
                    }
                    Some(out)
                }
                _ => None,
            }
        }
        OptValue::Flag => None,
    }
}

fn names(v: &OptValue) -> Option<Vec<String>> {
    match v {
        OptValue::Str(s) => Some(vec![s.clone()]),
        OptValue::List(items) => {
            items.iter().map(|i| if let OptValue::Str(s) = i { Some(s.clone()) } else { None }).collect()
        }
        _ => None,
    }
}

impl StochSimulOptions {
    /// Applies a command's options (later commands inherit them, as Dynare's global
    /// `options_`). Returns the keys that are not understood (for a warning).
    pub fn apply(&mut self, o: &Options) -> Vec<String> {
        let (mut unknown, invalid) = self.apply_impl(o);
        unknown.extend(invalid);
        unknown
    }

    /// [`Self::apply`], with an error for a known option whose value is unusable
    /// (negative or non-integer horizon, `order=0`, a string for a number, ...).
    pub fn apply_checked(&mut self, o: &Options) -> anyhow::Result<Vec<String>> {
        let (unknown, invalid) = self.apply_impl(o);
        if let Some(k) = invalid.first() {
            let v = o.iter().find(|(kk, _)| kk == k).map(|(_, v)| format!("{v:?}")).unwrap_or_default();
            anyhow::bail!("invalid value for option {k}: {v}");
        }
        Ok(unknown)
    }

    /// (unknown keys, known keys with an invalid value)
    fn apply_impl(&mut self, o: &Options) -> (Vec<String>, Vec<String>) {
        let mut unknown = Vec::new();
        let mut invalid = Vec::new();
        for (k, v) in o {
            let mut known = true;
            let num = || if let OptValue::Num(x) = v { Some(*x) } else { None };
            let ok = match k.as_str() {
                "order" => num()
                    .filter(|x| *x >= 1.0 && x.fract() == 0.0 && *x <= 100.0)
                    .map(|x| {
                        self.order = x as usize;
                        // ---- stoch_simul: the preprocessor writes k_order_solver = true for order >= 3
                        if self.order >= 3 {
                            self.k_order_solver = true;
                        }
                    })
                    .is_some(),
                "irf" => to_usize(v).map(|x| self.irf = x).is_some(),
                "irf_shocks" => names(v).map(|x| self.irf_shocks = Some(x)).is_some(),
                "relative_irf" => {
                    self.relative_irf = true;
                    true
                }
                "irf_plot_threshold" => num().map(|x| self.irf_plot_threshold = x).is_some(),
                "nograph" => {
                    self.nograph = true;
                    true
                }
                "graph_format" => {
                    let f: Vec<GraphFormat> =
                        names(v).unwrap_or_default().iter().filter_map(|s| GraphFormat::parse(s)).collect();
                    if !f.is_empty() {
                        self.graph_format = f;
                    }
                    true
                }
                "noprint" => {
                    self.noprint = true;
                    true
                }
                "nomoments" => {
                    self.nomoments = true;
                    true
                }
                "nocorr" => {
                    self.nocorr = true;
                    true
                }
                "nofunctions" => {
                    self.nofunctions = true;
                    true
                }
                "nodecomposition" => {
                    self.nodecomposition = true;
                    true
                }
                "nomodelsummary" => {
                    self.nomodelsummary = true;
                    true
                }
                "ar" => to_usize(v).map(|x| self.ar = x).is_some(),
                "hp_filter" => num().map(|x| self.hp_filter = x).is_some(),
                "periods" => to_usize(v).map(|x| self.periods = x).is_some(),
                "drop" => to_usize(v).map(|x| self.drop = x).is_some(),
                "conditional_variance_decomposition" => {
                    steps(v).map(|x| self.conditional_variance_decomposition = x).is_some()
                }
                "qz_criterium" => num().map(|x| self.qz_criterium = x).is_some(),
                "dr_display_tol" => num().map(|x| self.dr_display_tol = x).is_some(),
                "schur_vec_tol" => num().map(|x| self.schur_vec_tol = x).is_some(),
                "lyapunov_complex_threshold" => num().map(|x| self.lyapunov_complex_threshold = x).is_some(),
                "filtered_theoretical_moments_grid" => {
                    to_usize(v).map(|x| self.filtered_theoretical_moments_grid = x).is_some()
                }
                "bayesian_irf" => {
                    self.bayesian_irf = true;
                    true
                }
                "moments_varendo" => {
                    self.moments_varendo = true;
                    true
                }
                "sub_draws" => to_usize(v).map(|x| self.sub_draws = Some(x)).is_some(),
                "mh_conf_sig" => num().map(|x| self.mh_conf_sig = x).is_some(),
                // ---- stoch_simul ----
                k if self.apply_ss(k, v) => true,
                k if self.apply_q2(k, v) => true,
                k if IGNORED.contains(&k) => true,
                _ => {
                    known = false;
                    false
                }
            };
            if !ok {
                if known { invalid.push(k.clone()) } else { unknown.push(k.clone()) }
            }
        }
        (unknown, invalid)
    }

    /// The Q2 options (`loglinear`, filters, `dr=`, `spectral_density`, ...); true if `k` is one.
    fn apply_q2(&mut self, k: &str, v: &OptValue) -> bool {
        let num = || if let OptValue::Num(x) = v { Some(*x) } else { None };
        let flag = || !matches!(v, OptValue::Num(x) if *x == 0.0);
        match k {
            "loglinear" => self.loglinear = flag(),
            "one_sided_hp_filter" => self.one_sided_hp_filter = num().unwrap_or(0.0),
            "bandpass_filter" => {
                self.bandpass = match v {
                    OptValue::Vec(x) if x.len() == 2 => Some([x[0], x[1]]),
                    OptValue::List(x) if x.len() == 2 => match (&x[0], &x[1]) {
                        (OptValue::Num(a), OptValue::Num(b)) => Some([*a, *b]),
                        _ => Some([6.0, 32.0]),
                    },
                    OptValue::Num(x) if *x == 0.0 => None,
                    _ => Some(self.bandpass.unwrap_or([6.0, 32.0])),
                }
            }
            "spectral_density" => self.spectral_density = flag(),
            "contemporaneous_correlation" => self.contemporaneous_correlation = flag(),
            "no_posterior_kernel_density" => self.posterior_kernel_density = !flag(), // ---- estimation completion ----
            // the preprocessor sets only the flag of the chosen method (dr=default clears all three)
            "dr" => match v {
                OptValue::Str(s) if s == "cycle_reduction" => self.dr_cycle_reduction = true,
                OptValue::Str(s) if s == "logarithmic_reduction" => self.dr_logarithmic_reduction = true,
                OptValue::Str(s) if s == "aim" => self.aim_solver = true, // stoch_simul
                _ => (self.dr_cycle_reduction, self.dr_logarithmic_reduction, self.aim_solver) = (false, false, false),
            },
            "dr_cycle_reduction_tol" => self.dr_cycle_reduction_tol = num().unwrap_or(self.dr_cycle_reduction_tol),
            "dr_cycle_reduction_maxiter" => {
                self.dr_cycle_reduction_maxiter = num().map(|x| x as usize).unwrap_or(self.dr_cycle_reduction_maxiter)
            }
            "dr_logarithmic_reduction_tol" => {
                self.dr_logarithmic_reduction_tol = num().unwrap_or(self.dr_logarithmic_reduction_tol)
            }
            "dr_logarithmic_reduction_maxiter" => {
                self.dr_logarithmic_reduction_maxiter =
                    num().map(|x| x as usize).unwrap_or(self.dr_logarithmic_reduction_maxiter)
            }
            "qz_zero_threshold" => self.qz_zero_threshold = num().unwrap_or(self.qz_zero_threshold),
            "aim_solver" => self.aim_solver = flag(),
            "partial_information" => self.partial_information = flag(),
            "tex" => self.tex = flag(),
            "pruning" => self.pruning = flag(),
            "replic" => self.replic = num().map(|x| x as usize).unwrap_or(self.replic),
            _ => return false,
        }
        true
    }

    // ---- stoch_simul ----
    /// `graph`/`print` (re-enable), `graph_format = none`, `k_order_solver`, `simul_replic`,
    /// `solve_algo`; true if `k` is one of them.
    fn apply_ss(&mut self, k: &str, v: &OptValue) -> bool {
        let num = || if let OptValue::Num(x) = v { Some(*x) } else { None };
        match k {
            "graph" => self.nograph = false,
            "print" => self.noprint = false,
            "graph_format" => {
                let f = names(v).unwrap_or_default();
                // `none`: graphs are displayed but not saved (nothing to do without a display)
                if f.iter().any(|s| s.eq_ignore_ascii_case("none")) {
                    self.graph_format = Vec::new();
                } else {
                    return false; // the stream S handler
                }
            }
            "k_order_solver" => self.k_order_solver = !matches!(v, OptValue::Num(x) if *x == 0.0),
            "simul_replic" => self.simul_replic = num().map(|x| x as usize).unwrap_or(self.simul_replic),
            "solve_algo" => self.solve_algo = num().map(|x| x as u8).or(self.solve_algo),
            _ => return false,
        }
        true
    }
    // ---- end stoch_simul ----

    /// The first-order solver options these settings ask for (`qz_criterium`,
    /// `qz_zero_threshold`, `dr=cycle_reduction|logarithmic_reduction`).
    pub fn solver_options(&self) -> SolverOptions {
        let method = if self.dr_cycle_reduction {
            crate::solver::FirstOrderMethod::CycleReduction {
                tol: self.dr_cycle_reduction_tol,
                maxiter: self.dr_cycle_reduction_maxiter,
            }
        } else if self.dr_logarithmic_reduction {
            crate::solver::FirstOrderMethod::LogarithmicReduction {
                tol: self.dr_logarithmic_reduction_tol,
                maxiter: self.dr_logarithmic_reduction_maxiter,
            }
        } else {
            crate::solver::FirstOrderMethod::Qz
        };
        SolverOptions { qz_criterium: self.qz_criterium, qz_zero_threshold: self.qz_zero_threshold, method }
    }

    pub fn moment_options(&self) -> MomentOptions {
        MomentOptions {
            bandpass: self.bandpass,
            ar: self.ar,
            hp_filter: self.hp_filter,
            nodecomposition: self.nodecomposition,
            qz_criterium: self.qz_criterium,
            schur_vec_tol: self.schur_vec_tol,
            lyapunov_complex_threshold: self.lyapunov_complex_threshold,
            grid: self.filtered_theoretical_moments_grid,
            add_tiny_number_to_cholesky: self.add_tiny_number_to_cholesky,
        }
    }

    pub fn graph_options(&self, dir: &Path, fname: &str) -> GraphOptions {
        GraphOptions {
            enabled: !self.nograph,
            formats: self.graph_format.clone(),
            dir: crate::graphics::graph_dir(dir, fname),
        }
    }
}

/// A first-order solution with the shock covariance it is analysed with.
#[derive(Clone, Debug)]
pub struct LinearSolution {
    pub dr: DecisionRule,
    /// `M_.Sigma_e`.
    pub sigma_e: Mat,
    /// `M_.params`.
    pub params: Vec<f64>,
}

/// Where outputs go and the seed of the simulations.
#[derive(Clone, Debug)]
pub struct OutputContext {
    /// Directory of the .mod file.
    pub dir: PathBuf,
    pub fname: String,
    /// `<dir>/<fname>/Output`.
    pub out_dir: PathBuf,
    pub seed: u64,
}

/// The calibrated steady state (the `steady` command from `initval`).
pub fn calibrated_steady_state(mf: &ModFile, params: &[f64]) -> Result<Vec<f64>> {
    let model = &mf.model;
    let mut ys0 = vec![0.0; model.n_endo()];
    for &(i, v) in &mf.initval {
        ys0[i] = v;
    }
    let x = mf.exo_steady_state(); // oo_.exo_steady_state (initval)
    model.set_auxiliary_variables(&mut ys0, &x, params);
    let ssm = crate::deterministic::SteadyStateModel::from_modfile(mf)?; // stream D
    // resol.m: evaluate_steady_state(..., ~options_.steadystate.nocheck)
    let check = !crate::dsge::steady_flag(mf, "nocheck");
    crate::dsge::solve_steady_at(model, ssm.as_ref(), params, &ys0, &x, &crate::dsge::steady_options(mf), check)
        .map(|(ys, _)| ys)
        .map_err(|e| anyhow!("steady state at the calibration failed (info {}: {})", e.info, describe(e.info)))
}

// ---- stoch_simul ----
/// `M_.endo_histval(:, end)` of a `histval` block in a stochastic model (the starting point of
/// `stoch_simul`'s simulation): unlisted variables are 0, auxiliary lag variables take the
/// value of their original variable at the corresponding lag. `None` without `histval`.
pub fn stochastic_histval_of(model: &Model, histval: &[(bool, usize, i32, f64)]) -> Option<Vec<f64>> {
    if histval.is_empty() {
        return None;
    }
    let at = |endo: bool, j: usize, lag: i32| -> f64 {
        histval.iter().rev().find(|&&(e, i, l, _)| e == endo && i == j && l == lag).map(|h| h.3).unwrap_or(0.0)
    };
    let mut y0: Vec<f64> =
        (0..model.n_endo()).map(|j| if j < model.orig_endo_nbr { at(true, j, 0) } else { 0.0 }).collect();
    // dynamic_set_auxiliary_series: AUX_ENDO_LAG (type 1) and AUX_EXO_LAG (type 3) carry their
    // variable at the lag (e.g. histval's u(-1) for u(-2) in the model)
    for a in &model.aux_vars {
        match (a.ty, a.orig_index, a.orig_lead_lag) {
            (1, Some(o), Some(l)) => y0[a.endo_index] = at(true, o, l),
            (3, Some(o), Some(l)) => y0[a.endo_index] = at(false, o, l),
            _ => {}
        }
    }
    // rows of auxiliary variables set directly (`smoother2histval` writes the AUX_EXO_LAG rows
    // of M_.endo_histval): (true, aux endo index, 0, value)
    for &(e, j, l, v) in histval {
        if e && j >= model.orig_endo_nbr && j < y0.len() && l == 0 {
            y0[j] = v;
        }
    }
    Some(y0)
}

/// Measurement errors of the observed variables (`M_.H`, `options_.varobs_id`).
#[derive(Clone, Debug)]
pub struct MeasurementErrors {
    /// `M_.H` (nvobs x nvobs, varobs order).
    pub h: Mat,
    pub varobs: Vec<String>,
    /// Declaration indices of the observed variables.
    pub varobs_id: Vec<usize>,
}

impl MeasurementErrors {
    /// `None` unless some measurement error has a nonzero variance.
    pub fn from_modfile(mf: &ModFile) -> Option<MeasurementErrors> {
        if mf.varobs.is_empty() || mf.h.nrows() != mf.varobs.len() || (0..mf.h.nrows()).all(|i| mf.h[(i, i)] == 0.0) {
            return None;
        }
        let varobs_id = mf.varobs.iter().map(|v| mf.model.endo_index(v)).collect::<Option<Vec<_>>>()?;
        Some(MeasurementErrors { h: mf.h.clone(), varobs: mf.varobs.clone(), varobs_id })
    }

    /// `intersect(ivar, options_.varobs_id, 'stable')`: (observed declaration indices,
    /// their positions in `ivar`, their positions in varobs).
    pub fn requested(&self, ivar: &[usize]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let mut out = (Vec::new(), Vec::new(), Vec::new());
        for (k, &v) in ivar.iter().enumerate() {
            if let Some(o) = self.varobs_id.iter().position(|&x| x == v) {
                out.0.push(v);
                out.1.push(k);
                out.2.push(o);
            }
        }
        out
    }

    /// `diag(M_.H)`.
    pub fn variances(&self) -> Vec<f64> {
        self.h.diagonal().iter().copied().collect()
    }
}

/// The measurement-error part of `disp_moments.m`: `oo_.variance_decomposition_ME` (percent,
/// observed requested variables x exo_nbr+1) from the simulated `endo_simul` (n_endo x
/// periods), measurement errors drawn from ChaCha8 (`seed`). `None` without measurement
/// errors on the requested variables or with `nodecomposition`.
#[allow(clippy::too_many_arguments)]
pub fn simulated_vardec_me(
    o: &StochSimulOptions,
    endo_simul: &Mat,
    ivar: &[usize],
    sm: &SimulatedMoments,
    n_exo: usize,
    seed: u64,
) -> Result<Option<(Vec<usize>, Mat)>> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let Some(me) = &o.me else { return Ok(None) };
    let (obs_pos, index_subset, index_obs) = me.requested(ivar);
    if obs_pos.is_empty() || o.nodecomposition || n_exo == 1 {
        return Ok(None);
    }
    let Some(vd) = &sm.vardec else { return Ok(None) };
    let tol = 1e-10;
    let periods = endo_simul.ncols();
    let drop = o.drop;
    let nme = me.h.nrows();
    let i_me: Vec<usize> = (0..nme).filter(|&i| me.h[(i, i)] != 0.0).collect();
    let sub = Mat::from_fn(i_me.len(), i_me.len(), |a, b| me.h[(i_me[a], i_me[b])]);
    let chol = crate::linalg::chol_upper(&sub).ok_or_else(|| anyhow!("chol: Matrix must be positive definite."))?;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(0x4d45_4153_4552_5200);
    let mut z = Mat::zeros(periods, i_me.len());
    for t in 0..periods {
        for k in 0..i_me.len() {
            z[(t, k)] = StandardNormal.sample(&mut rng);
        }
    }
    let e = &z * &chol;
    let mut shock = Mat::zeros(periods, nme);
    for (k, &i) in i_me.iter().enumerate() {
        shock.set_column(i, &e.column(k));
    }
    let t_eff = periods - drop;
    let y_me = Mat::from_fn(t_eff, index_subset.len(), |t, a| {
        endo_simul[(ivar[index_subset[a]], drop + t)] + shock[(drop + t, index_obs[a])]
    });
    let y_me_only = Mat::from_fn(t_eff, index_subset.len(), |t, a| shock[(drop + t, index_obs[a])]);
    let mean = |m: &Mat| -> Vec<f64> { (0..m.ncols()).map(|c| m.column(c).mean()).collect() };
    let filter = q2::Filter::from_options(o)?;
    let y_me_f = q2::filtered_series(&y_me, &mean(&y_me), &filter)?;
    let y_me_only_f = q2::filtered_series(&y_me_only, &mean(&y_me_only), &filter)?;
    let s2_me: Vec<f64> = (0..y_me_f.ncols())
        .map(|c| y_me_f.column(c).iter().map(|x| x * x).sum::<f64>() / y_me_f.nrows() as f64)
        .collect();
    // positions (1-based, as MATLAB compares them) of zero variances
    let zero_me: Vec<usize> =
        (0..index_subset.len()).filter(|&a| s2_me[a].sqrt().abs() < tol).map(|a| index_subset[a] + 1).collect();
    let zero_var: Vec<usize> = (0..ivar.len()).filter(|&k| sm.std[k].abs() < tol).map(|k| k + 1).collect();
    let var_sample = |c: usize| -> f64 {
        let col = y_me_only_f.column(c);
        let m = col.mean();
        col.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (col.len() as f64 - 1.0)
    };
    let mut vme = Mat::zeros(index_subset.len(), n_exo + 1);
    for (a, &k) in index_subset.iter().enumerate() {
        for j in 0..n_exo {
            vme[(a, j)] = vd[(k, j)] * (sm.variance[k] / s2_me[a]);
        }
        vme[(a, n_exo)] = var_sample(a) / s2_me[a] * 100.0;
        // disp_moments.m compares declaration indices (observable_pos_requested_vars) with
        // positions in var_list (the zero-variance indices), as here
        let decl = obs_pos[a] + 1;
        if zero_me.contains(&decl) && zero_var.contains(&decl) {
            for j in 0..=n_exo {
                vme[(a, j)] = f64::NAN;
            }
        } else if zero_var.contains(&decl) && !zero_me.contains(&decl) {
            for j in 0..n_exo {
                vme[(a, j)] = 0.0;
            }
            vme[(a, n_exo)] = 1.0;
        }
    }
    Ok(Some((index_subset, vme)))
}

/// Header of `stoch_simul.m`'s `<fname>_IRF.tex` (TeX eps-loader, `tex` with `graph_format=eps`).
fn irf_tex_header() -> String {
    format!("% TeX eps-loader file generated by stoch_simul.m (Dynare).\n% {}\n \n", crate::latex::datestr_now())
}

/// The `figure` entries of `<fname>_IRF.tex` for one shock (`options_.figures.textwidth` 0.8).
fn irf_tex_entries(
    ctx: &OutputContext,
    shock: &str,
    shock_tex: &str,
    figs: &[crate::graphics::Figure],
    relative: bool,
) -> String {
    let base = format!("{}/graphs/{}", ctx.fname, ctx.fname);
    let mut s = String::new();
    let tw = 0.8;
    let nb = figs.len();
    for (k, f) in figs.iter().enumerate() {
        let width = tw * (f.panels.len() as f64 / f.cols.max(1) as f64).min(1.0);
        s += "\\begin{figure}[H]\n\\centering \n";
        if nb == 1 {
            s += &format!("\\includegraphics[width={width:.2}\\textwidth]{{{base}_IRF_{shock}}}\n");
            s += &format!("\\caption{{Impulse response functions (orthogonalized shock to ${shock_tex}$).}}\n");
            s += &format!("\\label{{Fig:IRF:{shock}}}\n");
        } else {
            let k1 = k + 1;
            s += &format!("\\includegraphics[width={width:.2}\\textwidth]{{{base}_IRF_{shock}{k1}}}\n");
            let what = if relative { "Relative impulse response functions" } else { "Impulse response functions" };
            s += &format!("\\caption{{{what} (orthogonalized shock to ${shock_tex}$).}}");
            s += &format!("\\label{{Fig:IRF:{shock}:{k1}}}\n");
        }
        s += "\\end{figure}\n \n";
    }
    s
}

/// `simult.m` with `options_.simul_replic > 1`: every simulated series (n_endo x periods,
/// periods 1..T) appended as float64 to `<fname>/Output/<fname>_simul` (`fwrite(fh, y_, 'float64')`,
/// column-major; read back with `fread(fh, [n_endo, periods], 'float64')`).
pub fn write_simul_replic(ctx: &OutputContext, sims: &[Mat]) -> Result<PathBuf> {
    std::fs::create_dir_all(&ctx.out_dir)?;
    let path = ctx.out_dir.join(format!("{}_simul", ctx.fname));
    let mut bytes = Vec::with_capacity(sims.iter().map(|m| m.len() * 8).sum());
    for m in sims {
        for x in m.iter() {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(&path, bytes)?;
    Ok(path)
}

/// [`calibrated_steady_state`] with `options_.solve_algo` set by stoch_simul / estimation
/// (`None`: the `steady` command's).
pub fn calibrated_steady_state_ss(mf: &ModFile, params: &[f64], solve_algo: Option<u8>) -> Result<Vec<f64>> {
    calibrated_steady_state_params(mf, params, solve_algo).map(|(ys, _)| ys)
}

/// [`calibrated_steady_state_ss`] also returning `M_.params` as `evaluate_steady_state` leaves
/// them (parameters assigned in a `steady_state_model` block updated).
pub fn calibrated_steady_state_params(
    mf: &ModFile,
    params: &[f64],
    solve_algo: Option<u8>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let model = &mf.model;
    let mut ys0 = vec![0.0; model.n_endo()];
    for &(i, v) in &mf.initval {
        ys0[i] = v;
    }
    let x = mf.exo_steady_state();
    model.set_auxiliary_variables(&mut ys0, &x, params);
    let ssm = crate::deterministic::SteadyStateModel::from_modfile(mf)?;
    let mut so = crate::dsge::steady_options(mf);
    if let Some(sa) = solve_algo {
        so.solve_algo = sa;
    }
    let check = !crate::dsge::steady_flag(mf, "nocheck");
    crate::dsge::solve_steady_at(model, ssm.as_ref(), params, &ys0, &x, &so, check)
        .map_err(|e| anyhow!("steady state at the calibration failed (info {}: {})", e.info, describe(e.info)))
}
// ---- end stoch_simul ----

/// `resol` at the calibration: steady state, then the first-order solution.
pub fn solve_calibration(mf: &ModFile, qz_criterium: f64) -> Result<Result<LinearSolution, SolveError>> {
    let params = mf.model.params.clone();
    let ys = calibrated_steady_state(mf, &params)?;
    Ok(solve_at_x(&mf.model, &params, &ys, &mf.exo_steady_state(), &mf.sigma_e, qz_criterium))
}

/// First-order solution at `params`, steady state `ys`.
pub fn solve_at(
    model: &Model,
    params: &[f64],
    ys: &[f64],
    sigma_e: &Mat,
    qz_criterium: f64,
) -> Result<LinearSolution, SolveError> {
    solve_at_x(model, params, ys, &vec![0.0; model.n_exo()], sigma_e, qz_criterium)
}

/// [`solve_at`] at the exogenous steady state `x` (`oo_.exo_steady_state`).
pub fn solve_at_x(
    model: &Model,
    params: &[f64],
    ys: &[f64],
    x: &[f64],
    sigma_e: &Mat,
    qz_criterium: f64,
) -> Result<LinearSolution, SolveError> {
    let solver = FirstOrderSolver::new(
        &model.lead_lag_incidence,
        model.n_exo(),
        SolverOptions { qz_criterium, ..Default::default() },
    );
    let g1 = model.dynamic_jacobian(ys, x, params);
    let dr = solver.solve(&g1, ys)?;
    Ok(LinearSolution { dr, sigma_e: sigma_e.clone(), params: params.to_vec() })
}

/// Variable list -> declaration indices (`varlist_indices.m`: duplicates removed, first kept);
/// empty list -> the declared (non-auxiliary) variables.
pub fn var_indices(model: &Model, var_list: &[String]) -> Result<Vec<usize>> {
    if var_list.is_empty() {
        return Ok((0..model.orig_endo_nbr).collect());
    }
    let mut out = Vec::new();
    for v in var_list {
        let i = model.endo_index(v).ok_or_else(|| anyhow!("stoch_simul: {v} is not an endogenous variable"))?;
        if !out.contains(&i) {
            out.push(i);
        }
    }
    Ok(out)
}

/// Everything one `stoch_simul` call computes (Dynare's `oo_` fields).
#[derive(Clone, Debug, Default)]
pub struct StochSimulResults {
    pub ivar: Vec<usize>,
    /// `oo_.irfs` per shock (rows in `ivar` order).
    pub irfs: Vec<irf::ShockIrf>,
    pub theoretical: Option<TheoreticalMoments>,
    /// `oo_.conditional_variance_decomposition` per step (nvar x exo_nbr, fractions).
    pub conditional_vardec: Option<Vec<Mat>>,
    pub simulated: Option<SimulatedMoments>,
    /// `oo_.endo_simul` (n_endo x periods) and `oo_.exo_simul` (periods x exo_nbr).
    pub endo_simul: Option<Mat>,
    pub exo_simul: Option<Mat>,
    /// `oo_.contemporaneous_correlation`, `oo_.SpectralDensity`.
    pub contemporaneous_correlation: Option<Mat>,
    pub spectral_density: Option<q2::SpectralDensity>,
    /// stoch_simul with measurement errors: `oo_.variance_decomposition_ME` (percent) and
    /// `oo_.conditional_variance_decomposition_ME` (fractions per step).
    pub variance_decomposition_me: Option<Mat>,
    pub conditional_vardec_me: Option<Vec<Mat>>,
}

/// `print_info` for a solver failure.
pub fn print_info(e: &SolveError) {
    println!("\nERROR: {} (info = {}).\n", describe(e.info), e.info);
}

/// Port of `stoch_simul.m` at order 1 on a solved model; prints like Dynare and writes the
/// IRF CSV files / figures. `exo_simul` (periods x exo_nbr) replaces the random shocks when
/// given (tests against MATLAB's draws).
pub fn stoch_simul(
    model: &Model,
    sol: &LinearSolution,
    o: &StochSimulOptions,
    var_list: &[String],
    ctx: &OutputContext,
    exo_simul: Option<&Mat>,
) -> Result<StochSimulResults> {
    stoch_simul_q2(model, sol, o, var_list, ctx, exo_simul, None)
}

/// [`stoch_simul`] with the `tex` tables . Under `loglinear`, `sol.dr` must be the
/// log-linear rule with `dr.ys` logged ([`run_stoch_simul`] does that).
pub fn stoch_simul_q2(
    model: &Model,
    sol: &LinearSolution,
    o: &StochSimulOptions,
    var_list: &[String],
    ctx: &OutputContext,
    exo_simul: Option<&Mat>,
    tex: Option<&TexNames>,
) -> Result<StochSimulResults> {
    if model.n_exo() == 0 {
        bail!("stoch_simul:: does not support having no varexo in the model.");
    }
    if o.order != 1 && !model.linear {
        println!("WARNING: stoch_simul: order = {} is not ported; using order = 1.", o.order);
    }
    // ---- shocks/shock_paths: check_model.m
    if let Some(si) = &o.shocks_info {
        skew::check_model(&sol.sigma_e, &si.h, &si.skew_e)?;
    }
    let ivar = var_indices(model, var_list)?;
    let dr = &sol.dr;
    let mut res = StochSimulResults { ivar: ivar.clone(), ..Default::default() };
    let labels: Vec<String> = ivar.iter().map(|&i| model.endo_names[i].clone()).collect();
    let mut out = String::new();
    if !o.noprint {
        if !o.nomodelsummary {
            // shocks/shock_paths + stoch_simul: shock covariance, coskewness, then M_.H
            out +=
                &model_summary_full(model, sol, o.shocks_info.as_ref().map(|si| si.skew_e.as_slice()), o.me.as_ref());
            if let Some(t) = tex {
                let mut h = vec!["Variables".to_string()];
                h.extend(t.exo.iter().cloned());
                q2::latex_table(
                    o,
                    &t.ctx,
                    "stoch_simul.m",
                    "MATRIX OF COVARIANCE OF EXOGENOUS SHOCKS",
                    "covar_ex_shocks",
                    &h,
                    &t.exo,
                    &sol.sigma_e,
                    max_len(&t.exo) + 2,
                    10,
                    6,
                )?;
                // stoch_simul.m: M_.H under the exo TeX names (as Dynare does), when not all-zero diagonal
                if let Some(me) = o.me.as_ref().filter(|me| me.h.diagonal().iter().any(|&x| x != 0.0)) {
                    q2::latex_table(
                        o,
                        &t.ctx,
                        "stoch_simul.m",
                        "MATRIX OF COVARIANCE OF MEASUREMENT ERRORS",
                        "covar_ME",
                        &h,
                        &t.exo,
                        &me.h,
                        max_len(&t.exo) + 2,
                        10,
                        6,
                    )?;
                }
            }
        }
        out.push('\n');
        if !o.nofunctions {
            out += &disp_dr_q2(model, dr, &ivar, o.dr_display_tol, o.loglinear);
        }
    }
    print!("{out}");
    if o.periods > 0 {
        if o.periods <= o.drop {
            println!(
                "\nSTOCH_SIMUL error: The horizon of simulation is shorter than the number of observations to be dropped."
            );
            println!("STOCH_SIMUL error: Either increase options_.periods or decrease options_.drop.");
            return Ok(res);
        }
        let ex = match exo_simul {
            Some(e) => e.clone(),
            // ---- shocks/shock_paths: skewed shocks (simult.m, rand_multivariate_csn)
            None if o.shocks_info.as_ref().is_some_and(|si| !si.skew_e.is_empty()) => {
                let si = o.shocks_info.as_ref().unwrap();
                simulate::draw_skewed(&sol.sigma_e, &si.skew_e, o.periods, ctx.seed)?
            }
            None => simulate::draw_shocks(&sol.sigma_e, o.periods, ctx.seed)?,
        };
        // ---- stoch_simul: y0 = M_.endo_histval when histval is given (logged under loglinear)
        let y0 = match o.endo_histval.clone() {
            Some(h) if o.loglinear => q2::log_variable(model, &h),
            Some(h) => h,
            None => dr.ys.clone(),
        };
        let y = simulate::simult(dr, &y0, &ex);
        res.endo_simul = Some(y.columns(1, o.periods).into_owned());
        // ---- stoch_simul: simul_replic > 1 (simult.m writes every replication)
        if o.simul_replic > 1 {
            let mut sims = vec![res.endo_simul.clone().unwrap()];
            for r in 1..o.simul_replic {
                let sd = ctx.seed.wrapping_add(r as u64);
                let e = match o.shocks_info.as_ref() {
                    Some(si) if !si.skew_e.is_empty() => {
                        simulate::draw_skewed(&sol.sigma_e, &si.skew_e, o.periods, sd)?
                    }
                    _ => simulate::draw_shocks(&sol.sigma_e, o.periods, sd)?,
                };
                sims.push(simulate::simult(dr, &y0, &e).columns(1, o.periods).into_owned());
            }
            write_simul_replic(ctx, &sims)?;
        }
        res.exo_simul = Some(ex);
    }
    if !o.nomoments {
        if o.periods == 0 {
            let r = theoretical_moments_q2(model, sol, &ivar, o, 1, None, tex)?;
            print!("{}", r.text);
            res.theoretical = r.th;
            res.conditional_vardec = r.conditional_vardec;
            res.contemporaneous_correlation = r.contemporaneous_correlation;
            res.variance_decomposition_me = r.variance_decomposition_me;
            res.conditional_vardec_me = r.conditional_vardec_me;
        } else {
            let filter = q2::Filter::from_options(o)?;
            // disp_moments.m: y0 = M_.endo_histval (logged under loglinear), else oo_.dr.ys
            let y0_moments = match o.endo_histval.clone() {
                Some(h) if o.loglinear => q2::log_variable(model, &h),
                Some(h) => h,
                None => dr.ys.clone(),
            };
            let sm = simulate::simulated_moments_q2(
                &sol.sigma_e,
                res.endo_simul.as_ref().unwrap(),
                res.exo_simul.as_ref().unwrap(),
                &ivar,
                o.drop,
                o.ar,
                &filter,
                o.nodecomposition,
                &|ex| simulate::simult(dr, &y0_moments, ex), // disp_moments: one-shock runs start from endo_histval too
            )?;
            let me = simulated_vardec_me(o, res.endo_simul.as_ref().unwrap(), &ivar, &sm, model.n_exo(), ctx.seed)?; // stoch_simul
            print!("{}", simulated_moments_report_me(model, &sm, &labels, o, 1, tex, &ivar, me.as_ref())?);
            if o.contemporaneous_correlation && !o.nocorr {
                res.contemporaneous_correlation = Some(sm.corr.clone());
            }
            res.variance_decomposition_me = me.as_ref().map(|(_, v)| v.clone());
            res.simulated = Some(sm);
        }
    }
    if o.irf > 0 {
        let cs = vardec::lower_cholesky(&sol.sigma_e, o.add_tiny_number_to_cholesky);
        let shocks = irf::irf_shock_indices(&model.exo_names, o.irf_shocks.as_deref())?;
        res.irfs = irf::stoch_simul_irfs(dr, &cs, &ivar, &shocks, o.irf, o.relative_irf);
        let gopts = o.graph_options(&ctx.dir, &ctx.fname);
        let mut eps_loader =
            tex.filter(|_| o.graph_format.contains(&crate::graphics::GraphFormat::Eps)).map(|_| irf_tex_header()); // stoch_simul
        for s in &res.irfs {
            let shock = &model.exo_names[s.shock];
            irf::write_irf_csv(&ctx.out_dir, &ctx.fname, shock, &labels, &s.y)?;
            if !o.nograph {
                let figs = irf::irf_figures(shock, &labels, &s.y, o.relative_irf, o.irf_plot_threshold);
                if let (Some(l), Some(t)) = (eps_loader.as_mut(), tex) {
                    *l += &irf_tex_entries(ctx, shock, &t.exo[s.shock], &figs, o.relative_irf);
                }
                for fig in figs {
                    crate::graphics::save(&fig, &ctx.fname, &gopts)?;
                }
            }
        }
        if let Some(mut l) = eps_loader {
            l += " \n% End Of TeX file. \n";
            let d = crate::graphics::graph_dir(&ctx.dir, &ctx.fname);
            std::fs::create_dir_all(&d)?;
            std::fs::write(d.join(format!("{}_IRF.tex", ctx.fname)), l)?;
        }
    }
    // ---- UnivariateSpectralDensity.m ----
    if o.spectral_density {
        let sd = q2::univariate_spectral_density(dr, &sol.sigma_e, &ivar, o)?;
        q2::spectral_density_figures(model, &sd, o, ctx)?;
        res.spectral_density = Some(sd);
    }
    write_results_json(model, &res, dr, ctx)?;
    Ok(res)
}
/// `<out_dir>/<fname>_stoch_simul.json`: the `oo_` fields of the last stoch_simul.
fn write_results_json(model: &Model, r: &StochSimulResults, dr: &DecisionRule, ctx: &OutputContext) -> Result<()> {
    let names: Vec<&String> = r.ivar.iter().map(|&i| &model.endo_names[i]).collect();
    let m2v = |m: &Mat| -> Vec<Vec<f64>> { (0..m.nrows()).map(|i| m.row(i).iter().copied().collect()).collect() };
    // `oo_.dr` at order 1 (`order_var` 1-based, as the order-2 file writes it)
    let drj = json!({
        "order": 1,
        "ys": dr.ys,
        "order_var": dr.order_var.iter().map(|v| v + 1).collect::<Vec<_>>(),
        "state_var": dr.state_var.iter().map(|v| v + 1).collect::<Vec<_>>(),
        "ghx": m2v(&dr.ghx),
        "ghu": m2v(&dr.ghu),
    });
    let mut j = json!({ "var_list": names, "exo_names": model.exo_names, "dr": drj });
    if !r.irfs.is_empty() {
        let mut irfs = serde_json::Map::new();
        for s in &r.irfs {
            for (k, n) in names.iter().enumerate() {
                irfs.insert(
                    format!("{}_{}", n, model.exo_names[s.shock]),
                    json!(s.y.row(k).iter().copied().collect::<Vec<_>>()),
                );
            }
        }
        j["irfs"] = serde_json::Value::Object(irfs);
    }
    if let Some(t) = &r.theoretical {
        j["mean"] = json!(t.mean(dr));
        j["var"] = json!(m2v(&t.gamma0));
        j["autocorr"] = json!(t.autocorr.iter().map(m2v).collect::<Vec<_>>());
        if let Some(v) = &t.vardec {
            j["variance_decomposition"] = json!(m2v(&(v * 100.0)));
        }
    }
    if let Some(c) = &r.conditional_vardec {
        j["conditional_variance_decomposition"] = json!(c.iter().map(|m| m2v(&(m * 100.0))).collect::<Vec<_>>());
    }
    if let Some(s) = &r.simulated {
        j["mean"] = json!(s.mean);
        j["var"] = json!(m2v(&s.var));
        j["skewness"] = json!(s.skewness);
        j["kurtosis"] = json!(s.kurtosis);
        j["autocorr"] = json!(s.autocorr.iter().map(m2v).collect::<Vec<_>>());
        if let Some(v) = &s.vardec {
            j["variance_decomposition"] = json!(m2v(v));
        }
    }
    // stoch_simul with measurement errors
    if let Some(v) = &r.variance_decomposition_me {
        j["variance_decomposition_ME"] = json!(m2v(v));
    }
    if let Some(c) = &r.conditional_vardec_me {
        j["conditional_variance_decomposition_ME"] = json!(c.iter().map(|m| m2v(&(m * 100.0))).collect::<Vec<_>>());
    }

    if let Some(c) = &r.contemporaneous_correlation {
        j["contemporaneous_correlation"] = json!(m2v(c));
    }
    if let Some(sd) = &r.spectral_density {
        j["SpectralDensity"] = json!({ "freqs": sd.freqs, "density": m2v(&sd.density) });
    }
    std::fs::create_dir_all(&ctx.out_dir)?;
    std::fs::write(ctx.out_dir.join(format!("{}_stoch_simul.json", ctx.fname)), serde_json::to_string_pretty(&j)?)?;
    Ok(())
}

/// MODEL SUMMARY and the covariance of the shocks (stoch_simul.m, after `skipline()`).
pub fn model_summary(model: &Model, sol: &LinearSolution) -> String {
    model_summary_me(model, sol, None)
}

/// [`model_summary`] followed, with measurement errors, by their covariance matrix
/// (`SE_<varobs>` labels).
pub fn model_summary_me(model: &Model, sol: &LinearSolution, me: Option<&MeasurementErrors>) -> String {
    model_summary_full(model, sol, None, me)
}

/// [`model_summary_me`] with the `COSKEWNESS OF EXOGENOUS SHOCKS` table (skewed shocks) between
/// the shock covariance and the measurement errors.
pub fn model_summary_full(
    model: &Model,
    sol: &LinearSolution,
    skew: Option<&skew::SkewEntries>,
    me: Option<&MeasurementErrors>,
) -> String {
    let dr = &sol.dr;
    let mut s = String::from("\nMODEL SUMMARY\n\n");
    s += &format!("  Number of variables:         {}\n", model.n_endo());
    s += &format!("  Number of stochastic shocks: {}\n", model.n_exo());
    s += &format!("  Number of state variables:   {}\n", dr.nspred);
    s += &format!("  Number of jumpers:           {}\n", dr.nsfwrd);
    s += &format!("  Number of static variables:  {}\n", dr.nstatic);
    let labels = &model.exo_names;
    let mut headers = vec!["Variables".to_string()];
    headers.extend(labels.iter().cloned());
    s += &dyntable(
        "MATRIX OF COVARIANCE OF EXOGENOUS SHOCKS",
        &headers,
        labels,
        &sol.sigma_e,
        max_len(labels) + 2,
        10,
        6,
    );
    if let Some(sk) = skew {
        s += &skew::skew_table(model, sk); // ---- shocks/shock_paths
    }
    // ---- stoch_simul: measurement errors
    if let Some(me) = me {
        s += &me_table(&me.h, &me.varobs);
    }
    s
}

/// `MATRIX OF COVARIANCE OF MEASUREMENT ERRORS` (`SE_<varobs>` labels), empty when `M_.H` is zero.
pub fn me_table(h: &Mat, varobs: &[String]) -> String {
    if !h.iter().any(|&x| x != 0.0) {
        return String::new();
    }
    let l: Vec<String> = varobs.iter().map(|v| format!("SE_{v}")).collect();
    let mut hd = vec!["Variables".to_string()];
    hd.extend(l.iter().cloned());
    dyntable("MATRIX OF COVARIANCE OF MEASUREMENT ERRORS", &hd, &l, h, max_len(&l) + 2, 10, 6)
}

/// `subst_auxvar(var_index, -1, M_)`: the name a state variable is printed with.
fn state_name(model: &Model, v: usize, lag: i32) -> String {
    if v < model.orig_endo_nbr {
        return format!("{}({lag})", model.endo_names[v]);
    }
    if let Some(a) = model.aux_vars.iter().find(|a| a.endo_index == v) {
        match (a.ty, a.orig_index, a.orig_lead_lag) {
            (1, Some(o), Some(l)) => return format!("{}({})", model.endo_names[o], l + lag),
            (3, Some(o), Some(l)) => {
                let k = if l == 0 { l + lag - 1 } else { l + lag };
                return format!("{}({k})", model.exo_names[o]);
            }
            _ => {}
        }
    }
    format!("{}({lag})", model.endo_names[v])
}

/// Port of `disp_dr.m` at order 1 (POLICY AND TRANSITION FUNCTIONS).
pub fn disp_dr(model: &Model, dr: &DecisionRule, ivar: &[usize], dr_display_tol: f64) -> String {
    disp_dr_q2(model, dr, ivar, dr_display_tol, false)
}

/// [`disp_dr`] with `options_.loglinear` (`log(...)` labels, wider columns).
pub fn disp_dr_q2(model: &Model, dr: &DecisionRule, ivar: &[usize], dr_display_tol: f64, loglinear: bool) -> String {
    let names: Vec<String> = ivar.iter().map(|&i| model.endo_names[i].clone()).collect();
    let rows: Vec<usize> = ivar.iter().map(|&i| dr.inv_order_var[i]).collect();
    let ll = if loglinear { 5 } else { 0 };
    let hl = names.iter().map(|n| n.chars().count() + 2).fold(16, usize::max) + ll;
    let aux_extra = if model.aux_vars.is_empty() { 0 } else { 3 };
    let vw = ((max_len(&names) + ll).max(max_len(&model.exo_names)) + aux_extra).max(20);
    let mut s = String::from("POLICY AND TRANSITION FUNCTIONS\n");
    s += &" ".repeat(vw);
    for n in &names {
        let n = if loglinear { format!("log({n})") } else { n.clone() };
        s += &format!("{n:>hl$}");
    }
    s.push('\n');
    let cell = |x: f64| -> String {
        if x.abs() >= dr_display_tol {
            let t = fmt_f(x, hl, 6);
            if t.chars().count() > hl { fmt_e(x, hl, 6) } else { t }
        } else {
            format!("{:>hl$}", 0)
        }
    };
    let mut line = |label: String, vals: Vec<f64>| {
        if vals.iter().any(|x| x.abs() >= dr_display_tol) {
            s += &format!("{label:<vw$}");
            for v in vals {
                s += &cell(v);
            }
            s.push('\n');
        }
    };
    line("Constant".into(), ivar.iter().map(|&i| dr.ys[i]).collect());
    for k in 0..dr.ghx.ncols() {
        let sn = state_name(model, dr.state_var[k], -1);
        let sn = if loglinear { format!("log({sn})") } else { sn };
        line(sn, rows.iter().map(|&r| dr.ghx[(r, k)]).collect());
    }
    for k in 0..dr.ghu.ncols() {
        line(model.exo_names[k].clone(), rows.iter().map(|&r| dr.ghu[(r, k)]).collect());
    }
    s
}

/// `add_filter_subtitle.m` (every filter).
pub(crate) fn filter_subtitle(title: &str, o: &StochSimulOptions) -> String {
    q2::add_filter_subtitle(title, o)
}

/// `M_.endo_names_tex` / `M_.exo_names_tex` and the output location of the `tex` tables.
#[derive(Clone, Debug)]
pub struct TexNames {
    pub ctx: OutputContext,
    /// All endogenous variables (declaration order).
    pub endo: Vec<String>,
    pub exo: Vec<String>,
}

impl TexNames {
    pub fn from_modfile(mf: &ModFile, ctx: &OutputContext) -> TexNames {
        TexNames {
            ctx: ctx.clone(),
            endo: (0..mf.model.n_endo()).map(|j| crate::latex::endo_tex(mf, j)).collect(),
            exo: (0..mf.model.n_exo()).map(|k| crate::latex::exo_tex(mf, k)).collect(),
        }
    }
}

/// What [`theoretical_moments_q2`] computes and prints.
#[derive(Clone, Debug, Default)]
pub struct ThMomentsReport {
    pub th: Option<TheoreticalMoments>,
    /// `oo_.conditional_variance_decomposition` (fractions), per step.
    pub conditional_vardec: Option<Vec<Mat>>,
    /// `oo_.mean` (with the second-order correction at order 2).
    pub mean: Vec<f64>,
    /// `oo_.contemporaneous_correlation` (with `contemporaneous_correlation`).
    pub contemporaneous_correlation: Option<Mat>,
    pub text: String,
    /// stoch_simul: `oo_.variance_decomposition_ME` (percent) and
    /// `oo_.conditional_variance_decomposition_ME` (fractions, per step, last column ME).
    pub variance_decomposition_me: Option<Mat>,
    pub conditional_vardec_me: Option<Vec<Mat>>,
}

/// Port of `disp_th_moments.m` at order 1: computes the theoretical moments, the
/// conditional variance decomposition, and the printout (empty under `noprint`).
#[allow(clippy::type_complexity)]
pub fn theoretical_moments_report(
    model: &Model,
    sol: &LinearSolution,
    ivar: &[usize],
    o: &StochSimulOptions,
) -> Result<(Option<TheoreticalMoments>, Option<Vec<Mat>>, String)> {
    let r = theoretical_moments_q2(model, sol, ivar, o, 1, None, None)?;
    Ok((r.th, r.conditional_vardec, r.text))
}

/// Port of `disp_th_moments.m` (`loglinear` labels, bandpass, `tex` tables,
/// `contemporaneous_correlation`, and `order = 2` without pruning: `mean_correction` is
/// `gamma_y{ar+3}` per `ivar` entry, the second moments are the first-order ones).
pub fn theoretical_moments_q2(
    model: &Model,
    sol: &LinearSolution,
    ivar: &[usize],
    o: &StochSimulOptions,
    order: usize,
    mean_correction: Option<&[f64]>,
    tex: Option<&TexNames>,
) -> Result<ThMomentsReport> {
    if o.one_sided_hp_filter != 0.0 {
        bail!(
            "disp_th_moments:: theoretical moments incompatible with one-sided HP filter. Use simulated moments instead"
        );
    }
    if order > 1 && (o.hp_filter != 0.0 || o.bandpass.is_some()) {
        bail!("Theoretical filtered moments not implemented above 1st order");
    }
    let dr = &sol.dr;
    let th = moments::th_autocovariances(dr, &sol.sigma_e, ivar, &o.moment_options())?;
    let all_names: Vec<String> = ivar.iter().map(|&i| model.endo_names[i].clone()).collect();
    let labels_of = |pos: &[usize]| -> Vec<String> {
        q2::transformed_labels(&pos.iter().map(|&k| all_names[k].clone()).collect::<Vec<_>>(), o.loglinear, false)
    };
    let tex_of = |pos: &[usize]| -> Vec<String> {
        let t = tex.map(|t| pos.iter().map(|&k| t.endo[ivar[k]].clone()).collect::<Vec<_>>()).unwrap_or_default();
        q2::transformed_labels(&t, o.loglinear, true)
    };
    let all: Vec<usize> = (0..ivar.len()).collect();
    let labels = labels_of(&all);
    let approx = |t: &str| if order == 2 { format!("APPROXIMATED {t}") } else { t.to_string() };
    #[allow(clippy::too_many_arguments)]
    fn latex(
        o: &StochSimulOptions,
        tex: Option<&TexNames>,
        title: &str,
        key: &str,
        by: &str,
        headers: &[String],
        l: &[String],
        z: &Mat,
        lh: usize,
        w: usize,
        p: usize,
    ) -> Result<()> {
        match tex {
            Some(t) => q2::latex_table(o, &t.ctx, by, title, key, headers, l, z, lh, w, p),
            None => Ok(()),
        }
    }
    let mut s = String::new();
    let p = !o.noprint;
    let nexo = model.n_exo();
    let mut vardec_me = None; // stoch_simul
    let mut cvd_me = None;
    let mut mean = th.mean(dr);
    if let Some(c) = mean_correction {
        for (m, ck) in mean.iter_mut().zip(c) {
            *m += ck;
        }
    }
    if !th.stationary.is_empty() {
        if p {
            let z = Mat::from_fn(ivar.len(), 3, |i, j| match j {
                0 => mean[i],
                1 => th.gamma0[(i, i)].sqrt(),
                _ => th.gamma0[(i, i)],
            });
            let headers: Vec<String> =
                ["VARIABLE", "MEAN", "STD. DEV.", "VARIANCE"].iter().map(|s| s.to_string()).collect();
            let title = filter_subtitle(&approx("THEORETICAL MOMENTS"), o);
            s += &dyntable(&title, &headers, &labels, &z, max_len(&labels) + 2, 11, 4);
            let tl = tex_of(&all);
            latex(o, tex, &title, "th_moments", "disp_th_moments.m", &headers, &tl, &z, max_len(&tl) + 2, 11, 4)?;
        }
        if let (true, Some(vd)) = (p && nexo > 1 && !o.nodecomposition, &th.vardec) {
            let st = &th.stationary;
            let sl = labels_of(st);
            let m = Mat::from_fn(st.len(), nexo, |i, j| 100.0 * vd[(st[i], j)]);
            let mut headers = vec![" ".to_string()];
            headers.extend(model.exo_names.iter().cloned());
            s.push('\n');
            let title = filter_subtitle(&approx("VARIANCE DECOMPOSITION (in percent)"), o);
            s += &dyntable(&title, &headers, &sl, &m, max_len(&sl) + 2, 8, 2);
            if let Some(t) = tex {
                let mut h = vec![" ".to_string()];
                h.extend(t.exo.iter().cloned());
                let tl = tex_of(st);
                latex(
                    o,
                    tex,
                    &title,
                    "th_var_decomp_uncond",
                    "display_unconditional_variance_decomposition.m",
                    &h,
                    &tl,
                    &m,
                    max_len(&tl) + 2,
                    8,
                    2,
                )?;
            }
        }
        // ---- stoch_simul: oo_.variance_decomposition_ME (disp_th_moments.m) and its table
        if let (Some(me), Some(vd), false) = (&o.me, &th.vardec, o.nodecomposition) {
            let (_, index_subset, index_obs) = me.requested(ivar);
            if !index_subset.is_empty() {
                let mev = me.variances();
                let mut vme = Mat::zeros(index_subset.len(), nexo + 1);
                for (a, (&k, &ob)) in index_subset.iter().zip(&index_obs).enumerate() {
                    let v = th.gamma0[(k, k)];
                    for j in 0..nexo {
                        vme[(a, j)] = 100.0 * vd[(k, j)] * (v / (v + mev[ob]));
                    }
                    vme[(a, nexo)] = 100.0 - (0..nexo).map(|j| vme[(a, j)]).sum::<f64>();
                }
                if p && nexo > 1 {
                    let st = &th.stationary;
                    let pos: Vec<usize> = (0..index_subset.len()).filter(|&a| st.contains(&index_subset[a])).collect();
                    let so: Vec<usize> = pos.iter().map(|&a| index_subset[a]).collect();
                    let lh = max_len(&labels_of(st)) + 2;
                    let sl = labels_of(&so);
                    let mut h = vec![" ".to_string()];
                    h.extend(model.exo_names.iter().cloned());
                    h.push("ME".into());
                    let z = Mat::from_fn(pos.len(), nexo + 1, |a, j| vme[(pos[a], j)]);
                    let title =
                        filter_subtitle(&approx("VARIANCE DECOMPOSITION (in percent)"), o) + " WITH MEASUREMENT ERROR";
                    s += &dyntable(&title, &h, &sl, &z, lh, 8, 2);
                    if let Some(t) = tex {
                        let mut h = vec![" ".to_string()];
                        h.extend(t.exo.iter().cloned());
                        h.push("ME".into());
                        let tl = tex_of(&so);
                        latex(
                            o,
                            tex,
                            &title,
                            "th_var_decomp_uncond_ME",
                            "display_unconditional_variance_decomposition.m",
                            &h,
                            &tl,
                            &z,
                            lh,
                            8,
                            2,
                        )?;
                    }
                }
                vardec_me = Some(vme);
            }
        }
    }
    let mut cvd = None;
    if !o.conditional_variance_decomposition.is_empty() {
        let steps = &o.conditional_variance_decomposition;
        let c = vardec::conditional_variance_decomposition(dr, &sol.sigma_e, steps, ivar)?;
        if p {
            let title = approx("CONDITIONAL VARIANCE DECOMPOSITION (in percent)");
            s += &format!("\n{title}\n");
            let mut headers = vec![" ".to_string()];
            headers.extend(model.exo_names.iter().cloned());
            // display_conditional_variance_decomposition.m: with TeX, lh comes from the TeX labels
            // (built from M_.endo_names, sic)
            let tl = q2::transformed_labels(&all_names, o.loglinear, true);
            let lh = if o.tex { max_len(&tl) + 2 } else { max_len(&labels) + 2 };
            for (k, m) in c.iter().enumerate() {
                s += &format!("Period {}:\n", steps[k]);
                let z = m * 100.0;
                s += &dyntable("", &headers, &labels, &z, lh, 8, 2);
                if let Some(t) = tex {
                    let mut h = vec![" ".to_string()];
                    h.extend(t.exo.iter().cloned());
                    let key = format!("th_var_decomp_cond_h{}", steps[k]);
                    latex(
                        o,
                        tex,
                        &format!("{title}; Period {}", steps[k]),
                        &key,
                        "display_conditional_variance_decomposition.m",
                        &h,
                        &tl,
                        &z,
                        lh,
                        8,
                        2,
                    )?;
                }
            }
        }
        // ---- stoch_simul: oo_.conditional_variance_decomposition_ME
        if let Some(me) = &o.me {
            let (obs_pos, index_subset, index_obs) = me.requested(ivar);
            let cv = vardec::conditional_variances(dr, &sol.sigma_e, steps, ivar)?;
            let mev = me.variances();
            let cme: Vec<Mat> = cv
                .iter()
                .map(|c| {
                    let mut m = Mat::zeros(index_subset.len(), nexo + 1);
                    for (a, (&k, &ob)) in index_subset.iter().zip(&index_obs).enumerate() {
                        let tot: f64 = (0..nexo).map(|j| c[(k, j)]).sum::<f64>() + mev[ob];
                        for j in 0..nexo {
                            m[(a, j)] = c[(k, j)] / tot;
                        }
                        m[(a, nexo)] = 1.0 - (0..nexo).map(|j| m[(a, j)]).sum::<f64>();
                    }
                    m
                })
                .collect();
            if p && !obs_pos.is_empty() {
                let title = approx("CONDITIONAL VARIANCE DECOMPOSITION (in percent)") + " - WITH MEASUREMENT ERROR";
                s += &format!("\n{title}\n");
                let mut headers = vec![" ".to_string()];
                headers.extend(model.exo_names.iter().cloned());
                headers.push("ME".into());
                let names: Vec<String> = obs_pos.iter().map(|&v| model.endo_names[v].clone()).collect();
                let l = q2::transformed_labels(&names, o.loglinear, false);
                let tl = q2::transformed_labels(&names, o.loglinear, true);
                let lh = if o.tex { max_len(&tl) + 2 } else { max_len(&l) + 2 };
                for (k, m) in cme.iter().enumerate() {
                    s += &format!("Period {}:\n", steps[k]);
                    let z = m * 100.0;
                    s += &dyntable("", &headers, &l, &z, lh, 8, 2);
                    if let Some(t) = tex {
                        let mut h = vec![" ".to_string()];
                        h.extend(t.exo.iter().cloned());
                        h.push("ME".into());
                        let key = format!("th_var_decomp_cond_h{}", steps[k]);
                        latex(
                            o,
                            tex,
                            &format!("{title}; Period {}", steps[k]),
                            &key,
                            "display_conditional_variance_decomposition.m",
                            &h,
                            &tl,
                            &z,
                            lh,
                            8,
                            2,
                        )?;
                    }
                }
            }
            if !obs_pos.is_empty() {
                cvd_me = Some(cme);
            }
        }
        cvd = Some(c);
    }
    let mut out = ThMomentsReport {
        mean,
        conditional_vardec: cvd,
        variance_decomposition_me: vardec_me,
        conditional_vardec_me: cvd_me,
        ..Default::default()
    };
    let i1 = th.nonzero_variance();
    if i1.is_empty() {
        if p {
            s += "\nAll endogenous are constant or non stationary, not displaying correlations and auto-correlations\n\n";
        }
        out.th = Some(th);
        out.text = s;
        return Ok(out);
    }
    let l1 = labels_of(&i1);
    if !o.nocorr && !th.stationary.is_empty() {
        let corr = th.correlation();
        if o.contemporaneous_correlation {
            out.contemporaneous_correlation = Some(corr.clone());
        }
        if p {
            let m = Mat::from_fn(i1.len(), i1.len(), |a, b| corr[(i1[a], i1[b])]);
            let mut headers = vec!["Variables".to_string()];
            headers.extend(l1.iter().cloned());
            s.push('\n');
            let title = filter_subtitle(&approx("MATRIX OF CORRELATIONS"), o);
            s += &dyntable(&title, &headers, &l1, &m, max_len(&l1) + 2, 8, 4);
            let tl = tex_of(&i1);
            let mut h = vec!["Variables".to_string()];
            h.extend(tl.iter().cloned());
            latex(o, tex, &title, "th_corr_matrix", "disp_th_moments.m", &h, &tl, &m, max_len(&tl) + 2, 8, 4)?;
        }
    }
    if o.ar > 0 && !th.stationary.is_empty() && p {
        let z = Mat::from_fn(i1.len(), o.ar, |a, k| th.autocorr[k][(i1[a], i1[a])]);
        let mut headers = vec!["Order ".to_string()];
        headers.extend((1..=o.ar).map(|k| k.to_string()));
        s.push('\n');
        let title = filter_subtitle(&approx("COEFFICIENTS OF AUTOCORRELATION"), o);
        s += &dyntable(&title, &headers, &l1, &z, max_len(&l1) + 2, 8, 4);
        let tl = tex_of(&i1);
        latex(o, tex, &title, "th_autocorr_matrix", "disp_th_moments.m", &headers, &tl, &z, max_len(&tl) + 2, 8, 4)?;
    }
    out.th = Some(th);
    out.text = s;
    Ok(out)
}

/// The printout of `disp_moments.m`.
pub fn simulated_moments_report(
    model: &Model,
    sm: &SimulatedMoments,
    labels: &[String],
    o: &StochSimulOptions,
) -> String {
    simulated_moments_report_q2(model, sm, labels, o, 1, None, &[]).unwrap_or_default()
}

/// The printout of `disp_moments.m` (`loglinear` labels, every filter, `tex`
/// tables, the order > 1 note). `names` are the endogenous names of the rows; `ivar` their
/// declaration indices (for the TeX names, may be empty without `tex`).
pub fn simulated_moments_report_q2(
    model: &Model,
    sm: &SimulatedMoments,
    names: &[String],
    o: &StochSimulOptions,
    order: usize,
    tex: Option<&TexNames>,
    ivar: &[usize],
) -> Result<String> {
    simulated_moments_report_me(model, sm, names, o, order, tex, ivar, None)
}

/// [`simulated_moments_report_q2`] with the measurement-error decomposition of
/// [`simulated_vardec_me`] (stoch_simul).
#[allow(clippy::too_many_arguments)]
pub fn simulated_moments_report_me(
    model: &Model,
    sm: &SimulatedMoments,
    names: &[String],
    o: &StochSimulOptions,
    order: usize,
    tex: Option<&TexNames>,
    ivar: &[usize],
    me: Option<&(Vec<usize>, Mat)>,
) -> Result<String> {
    let mut s = String::new();
    let labels = q2::transformed_labels(names, o.loglinear, false);
    let tl = match tex {
        Some(t) => {
            q2::transformed_labels(&ivar.iter().map(|&i| t.endo[i].clone()).collect::<Vec<_>>(), o.loglinear, true)
        }
        None => Vec::new(),
    };
    let latex = |title: &str, key: &str, headers: &[String], z: &Mat, lh: usize, w: usize, p: usize| -> Result<()> {
        match tex {
            Some(t) => q2::latex_table(o, &t.ctx, "disp_moments.m", title, key, headers, &tl, z, lh, w, p),
            None => Ok(()),
        }
    };
    let n = labels.len();
    let lh = max_len(&labels) + 2;
    if !o.nomoments {
        let z = Mat::from_fn(n, 5, |i, j| [sm.mean[i], sm.std[i], sm.variance[i], sm.skewness[i], sm.kurtosis[i]][j]);
        let headers: Vec<String> = ["VARIABLE", "MEAN", "STD. DEV.", "VARIANCE", "SKEWNESS", "KURTOSIS"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let title = filter_subtitle("MOMENTS OF SIMULATED VARIABLES", o);
        if !o.noprint {
            s += &dyntable(&title, &headers, &labels, &z, lh, 16, 6);
        }
        latex(&title, "sim_moments", &headers, &z, lh, 16, 6)?;
    }
    if o.noprint {
        return Ok(s);
    }
    if !o.nocorr {
        let mut h = vec!["VARIABLE".to_string()];
        h.extend(names.iter().cloned());
        let title = filter_subtitle("CORRELATION OF SIMULATED VARIABLES", o);
        s += &dyntable(&title, &h, &labels, &sm.corr, lh, 8, 4);
        if let Some(t) = tex {
            let mut h = vec!["VARIABLE".to_string()];
            h.extend(ivar.iter().map(|&i| t.endo[i].clone()));
            latex(&title, "sim_corr_matrix", &h, &sm.corr, lh, 8, 4)?;
        }
    }
    if !o.conditional_variance_decomposition.is_empty() {
        s += "\nSTOCH_SIMUL: conditional_variance_decomposition requires theoretical moments, i.e. periods=0.\n";
    }
    if !sm.autocorr.is_empty() {
        let z = Mat::from_fn(n, sm.autocorr.len(), |i, k| sm.autocorr[k][(i, i)]);
        let mut h = vec!["VARIABLE".to_string()];
        h.extend((1..=sm.autocorr.len()).map(|k| k.to_string()));
        let title = filter_subtitle("AUTOCORRELATION OF SIMULATED VARIABLES", o);
        s += &dyntable(&title, &h, &labels, &z, lh, 8, 4);
        latex(&title, "sim_autocorr_matrix", &h, &z, max_len(&tl) + 2, 8, 4)?;
    }
    if let (Some(vd), true) = (&sm.vardec, model.n_exo() > 1) {
        let nexo = model.n_exo();
        let m =
            Mat::from_fn(n, nexo + 1, |i, j| if j < nexo { vd[(i, j)] } else { (0..nexo).map(|k| vd[(i, k)]).sum() });
        let mut h = vec![" ".to_string()];
        h.extend(model.exo_names.iter().cloned());
        h.push("Tot. lin. contr.".into());
        s.push('\n');
        let title = filter_subtitle("VARIANCE DECOMPOSITION SIMULATING ONE SHOCK AT A TIME (in percent)", o);
        s += &dyntable(&title, &h, &labels, &m, max_len(names) + 2, 8, 2);
        if let Some(t) = tex {
            let mut h = vec![" ".to_string()];
            h.extend(t.exo.iter().cloned());
            h.push("Tot. lin. contr.".into());
            let lt: Vec<String> = ivar.iter().map(|&i| t.endo[i].clone()).collect();
            latex(&title, "sim_var_decomp", &h, &m, max_len(&lt) + 2, 8, 2)?;
        }
        // ---- stoch_simul: WITH MEASUREMENT ERROR
        if let Some((index_subset, vme)) = me {
            let mut h = vec![" ".to_string()];
            h.extend(model.exo_names.iter().cloned());
            h.push("ME".into());
            h.push("Tot. lin. contr.".into());
            let lo = q2::transformed_labels(
                &index_subset.iter().map(|&k| names[k].clone()).collect::<Vec<_>>(),
                o.loglinear,
                false,
            );
            let nc = vme.ncols();
            let z = Mat::from_fn(vme.nrows(), nc + 1, |i, j| {
                if j < nc { vme[(i, j)] } else { (0..nc).map(|k| vme[(i, k)]).sum() }
            });
            s += &dyntable(&format!("{title} WITH MEASUREMENT ERROR"), &h, &lo, &z, max_len(names) + 2, 8, 2);
            if let Some(t) = tex {
                let mut h = vec![" ".to_string()];
                h.extend(t.exo.iter().cloned());
                h.push("ME".into());
                h.push("Tot. lin. contr.".into());
                let lt: Vec<String> = ivar.iter().map(|&i| t.endo[i].clone()).collect();
                let lot = q2::transformed_labels(
                    &index_subset.iter().map(|&k| lt[k].clone()).collect::<Vec<_>>(),
                    o.loglinear,
                    true,
                );
                q2::latex_table(
                    o,
                    &t.ctx,
                    "disp_moments.m",
                    &format!("{title} WITH MEASUREMENT ERROR"),
                    "sim_var_decomp_ME",
                    &h,
                    &lot,
                    &z,
                    max_len(&lt) + 2,
                    8,
                    2,
                )?;
            }
        }
        if order == 1 {
            s += "Note: numbers do not add up to 100 due to non-zero correlation of simulated shocks in small samples\n\n";
        } else {
            s += "Note: numbers do not add up to 100 due to i) non-zero correlation of simulated shocks in small samples and ii) nonlinearity\n\n";
        }
    }
    Ok(s)
}

/// The `check` command at the calibration, or at the estimated parameter vector `point` that
/// an earlier `estimation` left in `M_.params`.
pub fn run_check(mf: &ModFile, o: &Options, point: Option<&[f64]>) -> Result<check::CheckResult> {
    run_check_det(mf, o, point, None)
}

/// [`run_check`] in a deterministic model: `det` holds `oo_.steady_state` and a non-zero
/// `oo_.exo_steady_state` (from `initval`), at which `resol` recomputes the steady state.
pub fn run_check_det(
    mf: &ModFile,
    o: &Options,
    point: Option<&[f64]>,
    det: Option<&crate::deterministic::DetState>,
) -> Result<check::CheckResult> {
    let mut qz = 1.0 + 1e-6;
    if let Some(OptValue::Num(v)) = crate::modfile::opt(o, "qz_criterium") {
        qz = *v;
    }
    let params = match point {
        Some(x) => crate::dsge::DsgeModel::new(mf, Some(x))?.set_all_parameters(x).0,
        None => mf.model.params.clone(),
    };
    let det = det.filter(|d| point.is_none() && d.exo_steady_state.iter().any(|&v| v != 0.0));
    let (ys, x) = match det {
        Some(d) => {
            let (ys, _) = crate::deterministic::evaluate_steady_state(
                &mf.model,
                d.ssm.as_ref(),
                &d.steady_state,
                &d.exo_steady_state,
                &params,
                &d.steady_opts,
                !crate::dsge::steady_flag(mf, "nocheck"),
            )
            .map_err(|e| anyhow!("steady state at the calibration failed (info {}: {})", e.info, describe(e.info)))?;
            (ys, d.exo_steady_state.clone())
        }
        None => (calibrated_steady_state(mf, &params)?, mf.exo_steady_state()),
    };
    // options_.qz_zero_threshold (mjdgges' 0/0 test)
    let qz0 = match crate::modfile::opt(o, "qz_zero_threshold") {
        Some(OptValue::Num(v)) => *v,
        _ => 1e-6,
    };
    let c = check::check_at(
        &mf.model,
        &params,
        &ys,
        &x,
        SolverOptions { qz_criterium: qz, qz_zero_threshold: qz0, ..Default::default() },
    )?;
    if crate::modfile::opt(o, "noprint").is_none() {
        print!("{}", check::format_check(&c));
    }
    Ok(c)
}

/// The `stoch_simul` command: solves at the calibration, or at `point` (the parameter
/// vector an earlier `estimation` left, as Dynare's `M_.params`), then [`stoch_simul`].
pub fn run_stoch_simul(
    mf: &ModFile,
    o: &StochSimulOptions,
    var_list: &[String],
    ctx: &OutputContext,
    point: Option<&[f64]>,
) -> Result<Option<StochSimulResults>> {
    // ---- stoch_simul: option checks of stoch_simul.m, partial_information ----
    if o.order == 0 {
        bail!("stoch_simul:: The order of the Taylor approximation cannot be 0!");
    }
    if mf.model.n_exo() == 0 {
        bail!(
            "stoch_simul:: does not support having no varexo in the model. As a workaround you could define a dummy exogenous variable."
        );
    }
    if matches!(o.solve_algo, Some(10 | 11)) {
        bail!(
            "stoch_simul:: perturbation solutions are not compatible with mixed complementarity problems and their solvers"
        );
    }
    if o.partial_information {
        return partial_info::stoch_simul_pi(mf, o, var_list, ctx, point);
    }
    // ---- end stoch_simul ----
    match solve_q2(mf, o, point)? {
        Ok(sol) => {
            let tex = o.tex.then(|| TexNames::from_modfile(mf, ctx));
            let mut o = o.clone(); // ---- shocks/shock_paths
            o.shocks_info = Some(skew::ShocksInfo::from_modfile(mf));
            let mut o = o.clone(); // stoch_simul: M_.endo_histval, M_.H
            o.endo_histval = stochastic_histval_of(&mf.model, &mf.histval);
            o.me = MeasurementErrors::from_modfile(mf);
            Ok(Some(stoch_simul_q2(&mf.model, &sol, &o, var_list, ctx, None, tex.as_ref())?))
        }
        Err(e) => {
            print_info(&e);
            Ok(None)
        }
    }
}

/// `resol` for `stoch_simul` (order 1) with the Q2 options: `dr=cycle_reduction|
/// logarithmic_reduction`, `qz_zero_threshold`, `aim_solver` (solved by the QZ), and
/// `loglinear` (positive steady state, info 26; log-linear rule with `dr.ys` logged as
/// `stoch_simul.m` leaves it). At the calibration, or at `point` (an estimation's `M_.params`).
pub fn solve_q2(
    mf: &ModFile,
    o: &StochSimulOptions,
    point: Option<&[f64]>,
) -> Result<Result<LinearSolution, SolveError>> {
    if o.partial_information {
        bail!("stoch_simul: partial_information goes through partial_info::stoch_simul_pi");
    }
    let sopts = o.solver_options();
    let model = &mf.model;
    if !matches!(sopts.method, crate::solver::FirstOrderMethod::Qz) {
        let solver = FirstOrderSolver::new(&model.lead_lag_incidence, model.n_exo(), sopts);
        if !solver.reduction_applicable() {
            if o.dr_cycle_reduction {
                bail!(
                    "The cycle reduction algorithm can't be used when the coefficient matrix for current variables isn't invertible"
                );
            }
            bail!(
                "The logarithmic reduction algorithm can't be used when the coefficient matrix for current variables isn't invertible"
            );
        }
    }
    let (mut sol, ys, already_log) = match point {
        None => {
            let params = model.params.clone();
            let ys = calibrated_steady_state_ss(mf, &params, o.solve_algo)?;
            if o.loglinear {
                if let Err(v) = q2::loglinear_check(model, &ys) {
                    return Ok(Err(SolveError::new(26, v)));
                }
            }
            let solver = FirstOrderSolver::new(&model.lead_lag_incidence, model.n_exo(), sopts);
            let g1 = model.dynamic_jacobian(&ys, &mf.exo_steady_state(), &params);
            // ---- stoch_simul: aim_solver (AIM_first_order_solver.m instead of dyn_first_order_solver.m)
            let solved = if o.aim_solver {
                aim::aim_first_order_solver(&solver, &g1, &ys, o.qz_criterium)?
            } else {
                solver.solve(&g1, &ys)
            };
            match solved {
                Ok(dr) => (LinearSolution { dr, sigma_e: mf.sigma_e.clone(), params }, ys, false),
                Err(e) => return Ok(Err(e)),
            }
        }
        Some(x) => {
            let mut dsge = crate::dsge::DsgeModel::new(mf, Some(x))?;
            dsge.set_solver_options(sopts);
            match dsge.resolve(x) {
                Ok(r) => {
                    if o.loglinear && !dsge.loglinear {
                        if let Err(v) = q2::loglinear_check(model, &r.steady_state) {
                            return Ok(Err(SolveError::new(26, v)));
                        }
                    }
                    let ys = r.steady_state.clone();
                    // ---- stoch_simul: aim_solver at the estimation's parameters
                    if o.aim_solver {
                        let solver = FirstOrderSolver::new(&model.lead_lag_incidence, model.n_exo(), sopts);
                        let g1 = model.dynamic_jacobian(&ys, &dsge.exo_ss, &r.params);
                        match aim::aim_first_order_solver(&solver, &g1, &ys, o.qz_criterium)? {
                            Ok(dr) => (LinearSolution { dr, sigma_e: r.q, params: r.params }, ys, false),
                            Err(e) => return Ok(Err(e)),
                        }
                    } else {
                        (LinearSolution { dr: r.dr, sigma_e: r.q, params: r.params }, ys, dsge.loglinear)
                    }
                }
                Err(e) => return Ok(Err(SolveError::new(e.info, e.penalty))),
            }
        }
    };
    if o.loglinear {
        if !already_log {
            q2::loglinear_decision_rule(model, &mut sol.dr, &ys);
        }
        sol.dr.ys = q2::log_variable(model, &ys);
    }
    Ok(Ok(sol))
}
