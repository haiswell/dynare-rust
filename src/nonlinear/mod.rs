// SPDX-License-Identifier: GPL-3.0-or-later
//! Nonlinear estimation with particle filters (stream PF): port of Dynare 7.2's
//! `estimation/non_linear_dsge_likelihood.m` and `nonlinear-filters/`.
//!
//! * [`ReducedForm`] is the `ReducedForm` structure that `non_linear_dsge_likelihood.m` builds
//!   from the order-2/3 decision rule (rows restricted to `oo_.dr.restrict_var_list`), with
//!   the initial state distribution (`options_.particle.initialization`).
//! * [`law_of_motion`] is `iterate_law_of_motion.m` (`local_state_space_iteration_2/3`, pruned
//!   or not), evaluated in parallel over the particles.
//! * the filters: [`sis`] (`sequential_importance_particle_filter.m`), [`apf`]
//!   (`auxiliary_particle_filter.m`), [`gf`] (`gaussian_filter.m` + `gaussian_filter_bank.m`),
//!   [`gmf`] (`gaussian_mixture_filter.m` + bank + densities), [`cpf`]
//!   (`conditional_particle_filter.m`), [`nlkf`] (`nonlinear_kalman_filter.m`); the
//!   resampling routines in [`resampling`], sigma points in [`sigma_points`].
//! * [`likelihood::NonlinearModel`] / [`likelihood::NonlinearPosterior`] is
//!   `non_linear_dsge_likelihood.m` as an [`crate::estimation::Objective`].
//! * [`dsmh`] is the dynamic striated Metropolis–Hastings sampler (`estimation/smc/dsmh.m`).
//! * [`estimation`] runs an `estimation(order = 2|3, ...)` command.
//!
//! Random numbers: every evaluation starts a fresh stream (Dynare resets the seed to its
//! default in every filter call, `set_dynare_seed_local_options([],false,'default')`), so the
//! likelihood is a deterministic function of the parameters. The stream is either ChaCha8
//! ([`rng::SeededDraws`]) or a replay of MATLAB's draws ([`rng::ReplayDraws`], tests). Draws
//! are always taken sequentially in MATLAB's order (column-major matrices); the per-particle
//! work then runs in parallel and the reductions are sequential, so results do not depend on
//! the number of threads.

pub mod apf;
pub mod cpf;
pub mod dsmh;
pub mod estimation;
pub mod gf;
pub mod gmf;
pub mod law_of_motion;
pub mod likelihood;
pub mod linalg;
pub mod nlkf;
pub mod online; // online_auxiliary_filter
pub mod resampling;
pub mod rng;
pub mod sigma_points;
pub mod sis;
pub mod smc; // hssmc, dime
pub mod smc_run; // SMC samplers in the estimation command

use crate::linalg::Mat;
use crate::modfile::{OptValue, Options, opt};
use anyhow::{Result, bail};

pub use likelihood::{NonlinearModel, NonlinearPosterior};

/// `options_.particle.filter_algorithm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterAlgorithm {
    /// `sis`: sequential importance (bootstrap) particle filter.
    Sis,
    /// `apf`: auxiliary particle filter.
    Apf,
    /// `gf`: Gaussian (particle) filter.
    Gf,
    /// `gmf`: Gaussian-mixture filter.
    Gmf,
    /// `cpf`: conditional particle filter.
    Cpf,
    /// `nlkf`: nonlinear (unscented / cubature) Kalman filter.
    Nlkf,
}

impl FilterAlgorithm {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "sis" => Self::Sis,
            "apf" => Self::Apf,
            "gf" => Self::Gf,
            "gmf" => Self::Gmf,
            "cpf" => Self::Cpf,
            "nlkf" => Self::Nlkf,
            _ => bail!("Estimation: Unknown filter {s}"),
        })
    }
    /// The MATLAB function (`options_.particle.algorithm`).
    pub fn matlab_name(self) -> &'static str {
        match self {
            Self::Sis => "sequential_importance_particle_filter",
            Self::Apf => "auxiliary_particle_filter",
            Self::Gf => "gaussian_filter",
            Self::Gmf => "gaussian_mixture_filter",
            Self::Cpf => "conditional_particle_filter",
            Self::Nlkf => "nonlinear_kalman_filter",
        }
    }
}

/// `options_.particle.resampling.status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplingStatus {
    Systematic,
    None,
    Generic,
}

/// `options_.particle.resampling.method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplingMethod {
    Kitagawa,
    Stratified,
    Smooth,
}

/// `proposal_approximation` / `distribution_approximation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approximation {
    Cubature,
    Unscented,
    MonteCarlo,
}

impl Approximation {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "cubature" => Self::Cubature,
            "unscented" => Self::Unscented,
            "montecarlo" => Self::MonteCarlo,
            _ => bail!("unknown approximation `{s}` (cubature, unscented, montecarlo)"),
        })
    }
}

/// `options_.particle.cpf_weights_method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpfWeights {
    AmisanoTristani,
    MurrayJonesParslow,
}

/// `options_.particle.unscented`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Unscented {
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
}

/// `options_.particle` (defaults of `default_option_values.m`).
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleOptions {
    pub algorithm: FilterAlgorithm,
    /// `initialization`: 1 = ergodic variance of the first-order approximation, 2 = Monte
    /// Carlo estimate from a simulation, 3 = `initial_state_prior_std * I`.
    pub initialization: u8,
    pub initial_state_prior_std: f64,
    pub number_of_particles: usize,
    pub pruning: bool,
    pub unscented: Unscented,
    pub resampling: ResamplingStatus,
    pub resampling_threshold: f64,
    pub resampling_method: ResamplingMethod,
    pub proposal: Approximation,
    pub distribution: Approximation,
    pub mixture_state_variables: usize,
    pub mixture_structural_shocks: usize,
    pub mixture_measurement_shocks: usize,
    pub cpf_weights: CpfWeights,
}

impl Default for ParticleOptions {
    fn default() -> Self {
        ParticleOptions {
            algorithm: FilterAlgorithm::Sis,
            initialization: 1,
            initial_state_prior_std: 0.1,
            number_of_particles: 5000,
            pruning: false,
            unscented: Unscented { alpha: 1.0, beta: 2.0, kappa: 1.0 },
            resampling: ResamplingStatus::Systematic,
            resampling_threshold: 0.5,
            resampling_method: ResamplingMethod::Kitagawa,
            proposal: Approximation::Unscented,
            distribution: Approximation::Unscented,
            mixture_state_variables: 5,
            mixture_structural_shocks: 1,
            mixture_measurement_shocks: 1,
            cpf_weights: CpfWeights::AmisanoTristani,
        }
    }
}

fn str_opt(o: &Options, k: &str) -> Option<String> {
    match opt(o, k) {
        Some(OptValue::Str(s)) => Some(s.clone()),
        Some(OptValue::List(v)) if v.len() == 1 => match &v[0] {
            OptValue::Str(s) => Some(s.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn num_opt(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

impl ParticleOptions {
    /// The estimation options the preprocessor maps to `options_.particle`, then
    /// `check_particle_filter_options.m` (`particle_filter_options`), as
    /// `dynare_estimation_1.m` runs it when `options_.particle.status` (order > 1).
    pub fn from_options(o: &Options) -> Result<Self> {
        let mut p = Self::from_preprocessor_options(o)?;
        p.check_particle_filter_options(o)?;
        Ok(p)
    }

    /// The estimation options the preprocessor maps to `options_.particle`, without
    /// `check_particle_filter_options.m` (at order 1, where `options_.particle.status` is
    /// false, `particle_filter_options` is never read).
    pub fn from_preprocessor_options(o: &Options) -> Result<Self> {
        let mut p = ParticleOptions::default();
        if let Some(s) = str_opt(o, "filter_algorithm") {
            p.algorithm = FilterAlgorithm::parse(&s)?;
        }
        if let Some(v) = num_opt(o, "number_of_particles") {
            p.number_of_particles = v as usize;
        }
        if let Some(s) = str_opt(o, "resampling") {
            p.resampling = match s.as_str() {
                "systematic" => ResamplingStatus::Systematic,
                "none" => ResamplingStatus::None,
                "generic" => ResamplingStatus::Generic,
                _ => bail!("resampling = {s}: expected systematic, none or generic"),
            };
        }
        if let Some(v) = num_opt(o, "resampling_threshold") {
            p.resampling_threshold = v;
        }
        if let Some(s) = str_opt(o, "resampling_method") {
            p.resampling_method = match s.as_str() {
                "kitagawa" => ResamplingMethod::Kitagawa,
                "stratified" => ResamplingMethod::Stratified,
                "smooth" => ResamplingMethod::Smooth,
                _ => bail!("resampling_method = {s}: expected kitagawa, stratified or smooth"),
            };
        }
        if let Some(s) = str_opt(o, "proposal_approximation") {
            p.proposal = Approximation::parse(&s)?;
        }
        if let Some(s) = str_opt(o, "distribution_approximation") {
            p.distribution = Approximation::parse(&s)?;
        }
        if let Some(s) = str_opt(o, "cpf_weights") {
            p.cpf_weights = match s.as_str() {
                "amisanotristani" => CpfWeights::AmisanoTristani,
                "murrayjonesparslow" => CpfWeights::MurrayJonesParslow,
                _ => bail!("cpf_weights = {s}: expected amisanotristani or murrayjonesparslow"),
            };
        }
        // ---- estimation completion: nonlinear_filter_initialization (options_.particle.initialization)
        if let Some(v) = num_opt(o, "nonlinear_filter_initialization") {
            p.initialization = v as u8;
        }
        // ---- end estimation completion
        Ok(p)
    }

    /// `check_particle_filter_options.m`: `particle_filter_options = ('key', value, ...)`,
    /// with its argument checks. Keys are case sensitive; `posterior_sampler` and
    /// `liu_west_delta` are checked and stored by Dynare but read by no filter.
    fn check_particle_filter_options(&mut self, o: &Options) -> Result<()> {
        let Some(v) = opt(o, "particle_filter_options") else { return Ok(()) };
        let items: Vec<OptValue> = match v {
            OptValue::List(l) => l.clone(),
            other => vec![other.clone()],
        };
        let mut it = items.into_iter();
        while let Some(k) = it.next() {
            let OptValue::Str(k) = k else { bail!("particle_filter_options: expected 'key', value pairs") };
            let val =
                it.next().ok_or_else(|| anyhow::anyhow!("read_key_value_string: Wrong number of Key-Value pairs!"))?;
            let num = match &val {
                OptValue::Num(x) => Some(*x),
                _ => None,
            };
            let need = |x: Option<f64>| x.ok_or_else(|| anyhow::anyhow!("particle_filter_options: {k} needs a number"));
            // `options_list{i,2} <= 0` (or `< 0`) -> error(msg)
            let positive = |x: f64, strict: bool, msg: &str| -> Result<f64> {
                if (strict && x <= 0.0) || (!strict && x < 0.0) {
                    bail!("check_particle_filter_options:: {msg}")
                } else {
                    Ok(x)
                }
            };
            match k.as_str() {
                "posterior_sampler" => {
                    let ok = matches!(&val, OptValue::Str(s)
                        if s.eq_ignore_ascii_case("Herbst_Schorfheide") || s.eq_ignore_ascii_case("DSMH"));
                    if !ok {
                        bail!(
                            "check_particle_filter_options:: the proposal_distribution option to estimation takes either Herbst_Schorfheide or Herbst_Schorfheide as options"
                        );
                    }
                }
                "initial_state_prior_std" => {
                    self.initial_state_prior_std =
                        positive(need(num)?, true, "the initial_state_prior_std option takes a positive argument")?
                }
                // `~islogical(value)`: only `true` / `false` (str2num of a number is a double)
                "pruning" => match &val {
                    OptValue::Str(s) if s == "true" || s == "false" => self.pruning = s == "true",
                    _ => bail!("check_particle_filter_options:: the pruning options takes only true or false"),
                },
                "unscented_alpha" => {
                    self.unscented.alpha =
                        positive(need(num)?, true, "the unscented_alpha option takes a positive argument")?
                }
                "unscented_beta" => {
                    self.unscented.beta =
                        positive(need(num)?, false, "the unscented_beta option takes a positive argument")?
                }
                "unscented_kappa" => {
                    self.unscented.kappa =
                        positive(need(num)?, false, "the unscented_kappa option takes a positive argument")?
                }
                "mixture_state_variables" => {
                    self.mixture_state_variables =
                        positive(need(num)?, true, "the mixture_state_variables option takes a positive integer")?
                            as usize
                }
                "mixture_structural_shocks" => {
                    self.mixture_structural_shocks =
                        positive(need(num)?, true, "the mixture_structural_shocks option takes a positive integer")?
                            as usize
                }
                "mixture_measurement_shocks" => {
                    self.mixture_measurement_shocks =
                        positive(need(num)?, true, "the mixture_measurement_shocks option takes a positive integer")?
                            as usize
                }
                "liu_west_delta" => {
                    positive(need(num)?, true, "the liu_west_delta option takes a positive argument")?;
                }
                _ => println!("WARNING: check_particle_filter_options: Unknown option ({k})!"),
            }
        }
        Ok(())
    }
}

/// `select_qz_criterium_value.m`: `options_.qz_criterium` for an estimation, from the user's
/// value (`None` when `options_.qz_criterium` is empty). `particle_initialization` is
/// `Some(options_.particle.initialization)` when `options_.particle.status` (order > 1),
/// `None` for the linear filters, which use `lik_init`.
pub fn select_qz_criterium_value(user: Option<f64>, particle_initialization: Option<u8>, lik_init: f64) -> Result<f64> {
    Ok(match particle_initialization {
        // non linear filter
        Some(3) => match user {
            None => 1.0 + 1e-6,
            Some(v) if v <= 1.0 => {
                println!(
                    "\ndynare_estimation_init.m:: You set nonlinear_filter_initialization equal to 3, it is assumed that you try to estimate a non stationary model. Resetting it to 1+1e-6."
                );
                1.0 + 1e-6
            }
            Some(v) => v,
        },
        Some(_) => user.unwrap_or(1.0 - 1e-6),
        // linear filter
        None if lik_init == 1.0 => match user {
            None => 1.0 - 1e-6,
            Some(v) if v > 1.0 - f64::EPSILON => bail!(
                "dynare_estimation_init.m: option qz_criterium is too large for estimating/smoothing a stationary model. If your model contains unit roots, use option diffuse_filter"
            ),
            Some(v) => v,
        },
        None => match user {
            None => 1.0 + 1e-6,
            Some(v) if v <= 1.0 => {
                println!(
                    "\ndynare_estimation_init.m:: diffuse filter is incompatible with a qz_criterium<=1. Resetting it to 1+1e-6."
                );
                1.0 + 1e-6
            }
            Some(v) => v,
        },
    })
}

/// `estimation(order = k)` with `k >= 2` (Dynare then requires a nonlinear filter).
pub fn requested_order(o: &Options) -> usize {
    num_opt(o, "order").map_or(1, |v| v as usize)
}

/// The `ReducedForm` structure of `non_linear_dsge_likelihood.m` (rows restricted to
/// `restrict_var_list`, in DR order; `m` rows, `ns` states, `nu` shocks).
#[derive(Debug, Clone)]
pub struct ReducedForm {
    pub order: usize,
    /// `dr.ys(dr.order_var(restrict_var_list))`.
    pub steadystate: Vec<f64>,
    /// `steadystate + 0.5 * ghs2`.
    pub constant: Vec<f64>,
    /// Steady state of the state variables (`mf0` rows).
    pub state_ss: Vec<f64>,
    pub q: Mat,
    /// Measurement-error covariance (`M_.H`, varobs order).
    pub h: Mat,
    /// 0-based positions of the states / observables in the restricted rows.
    pub mf0: Vec<usize>,
    pub mf1: Vec<usize>,
    pub ghx: Mat,
    pub ghu: Mat,
    pub ghxx: Mat,
    pub ghuu: Mat,
    pub ghxu: Mat,
    pub ghs2: Vec<f64>,
    pub ghxxx: Mat,
    pub ghuuu: Mat,
    pub ghxxu: Mat,
    pub ghxuu: Mat,
    pub ghxss: Mat,
    pub ghuss: Mat,
    pub state_mean: Vec<f64>,
    pub state_variance: Mat,
}

impl ReducedForm {
    pub fn m(&self) -> usize {
        self.ghx.nrows()
    }
    pub fn ns(&self) -> usize {
        self.mf0.len()
    }
    pub fn nu(&self) -> usize {
        self.q.nrows()
    }
    pub fn pp(&self) -> usize {
        self.mf1.len()
    }
}

/// Output of a filter: `LIK = -sum(lik(start:end))` and the per-period `lik`.
#[derive(Debug, Clone)]
pub struct FilterOutput {
    pub lik_total: f64,
    pub lik: Vec<f64>,
}

impl FilterOutput {
    pub fn nan(t: usize) -> Self {
        FilterOutput { lik_total: f64::NAN, lik: vec![f64::NAN; t] }
    }
    /// `LIK = -sum(lik(start:end))`, `start` 0-based.
    pub fn from_lik(lik: Vec<f64>, start: usize) -> Self {
        let s: f64 = lik[start.min(lik.len())..].iter().sum();
        FilterOutput { lik_total: -s, lik }
    }
}

/// Runs the filter selected by `opts.algorithm` (`feval(options_.particle.algorithm, ...)`).
/// `y`: pp x T observations (not demeaned), `start`: 0-based first period of the likelihood
/// (`presample`).
pub fn run_filter(
    rf: &ReducedForm,
    y: &Mat,
    start: usize,
    opts: &ParticleOptions,
    draws: &mut dyn rng::Draws,
) -> Result<FilterOutput> {
    match opts.algorithm {
        FilterAlgorithm::Sis => sis::filter(rf, y, start, opts, draws),
        FilterAlgorithm::Apf => apf::filter(rf, y, start, opts, draws),
        FilterAlgorithm::Gf => gf::filter(rf, y, start, opts, draws),
        FilterAlgorithm::Gmf => gmf::filter(rf, y, start, opts, draws),
        FilterAlgorithm::Cpf => cpf::filter(rf, y, start, opts, draws),
        FilterAlgorithm::Nlkf => nlkf::filter(rf, y, start, opts, draws),
    }
}
