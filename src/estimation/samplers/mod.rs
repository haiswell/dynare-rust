// SPDX-License-Identifier: GPL-3.0-or-later
//! Posterior samplers other than the plain Gaussian random walk: ports of the
//! branches of `estimation/posterior_sampler_iteration.m`
//! (`random_walk_metropolis_hastings` with a Student proposal,
//! `independent_metropolis_hastings`, `slice` = `slice_sampler.m` +
//! `rotated_slice_sampler.m`, `tailored_random_block_metropolis_hastings` +
//! `TaRB_optimizer_wrapper.m`), of the option handling of
//! `check_posterior_sampler_options.m` and of the per-sampler differences of
//! `posterior_sampler_initialization.m`.
//!
//! Chains run in parallel (one thread per chain, ChaCha stream `b+1` of the
//! seed for chain `b`, stream 0 for the initial values) as in
//! [`crate::estimation::rw_metropolis`], which still runs the default Gaussian
//! random walk. The draws do not depend on the number of threads: the
//! independent sampler and the slice sampler evaluate speculative points in
//! parallel (see [`imh`], [`slice`]) and keep exactly the sequential path.

pub mod chol_se;
mod imh;
mod rwmh;
mod slice;
mod tarb;

use anyhow::{Result, anyhow, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

pub use chol_se::chol_se;

use super::mcmc::{MhOptions, MhProgress, MhResult, initial_values, prefetch_width, rw_metropolis};
use super::optim::minimize::{key_values, kv_num, kv_str};
use super::{Bounds, Chain, Objective, Prior, prior_draw};
use crate::linalg::{Mat, chol_upper};
use crate::modfile::OptValue;

/// `options_.posterior_sampler_options.posterior_sampling_method`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingMethod {
    RandomWalk,
    Independent,
    Slice,
    TaRB,
}

/// `proposal_distribution`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proposal {
    /// `rand_multivariate_normal` (default).
    Normal,
    /// `rand_multivariate_student` with `student_degrees_of_freedom`.
    Student,
}

/// `options_.posterior_sampler_options.slice` (`default_option_values.m`).
#[derive(Clone, Debug)]
pub struct SliceOptions {
    /// `rotated` (0).
    pub rotated: bool,
    /// `mode_files`: a .mat file with `xparams` (npar x nmodes, >= 2 modes).
    pub mode_files: Option<String>,
    /// `mode`: the modes of the multimodal rotated sampler (loaded from `mode_files`).
    pub modes: Vec<Vec<f64>>,
    /// `slice_initialize_with_mode` (0): without it Dynare skips mode finding
    /// (`mh_posterior_mode_estimation`) and draws the starts from the prior.
    pub initialize_with_mode: bool,
    /// `initial_step_size` (0.8): `W1 = initial_step_size*(ub-lb)`.
    pub initial_step_size: f64,
    /// `use_prior_draws` (false).
    pub use_prior_draws: bool,
    /// `fast_likelihood_evaluation_for_rejection` (false): accepted; the
    /// objective is always evaluated in full (same draws, no early rejection).
    pub fast_likelihood_evaluation_for_rejection: bool,
    pub fast_likelihood_evaluation_for_rejection_penalty: f64,
    /// `maximize` (false): each iteration maximizes the posterior from the last draw
    /// (`posterior_sampler_iteration.m`) instead of drawing a slice step.
    pub maximize: bool,
    /// `maximize_using_mh_bounds` (false): the maximization stays within the range of the
    /// previous MCMC's draws (`load_mh_file`), see [`maximize_bounds`].
    pub maximize_using_mh_bounds: bool,
    /// `mode_compute` (5) of `maximize`.
    pub mode_compute: i32,
    /// The estimation's `optim` (`options_.optim_opt`), read by the `maximize` optimizer.
    pub optim: Option<OptValue>,
    /// The bounds of `maximize_using_mh_bounds` (set by [`maximize_bounds`]).
    pub maximize_bounds: Option<Bounds>,
}

impl Default for SliceOptions {
    fn default() -> Self {
        SliceOptions {
            rotated: false,
            mode_files: None,
            modes: vec![],
            initialize_with_mode: false,
            initial_step_size: 0.8,
            use_prior_draws: false,
            fast_likelihood_evaluation_for_rejection: false,
            fast_likelihood_evaluation_for_rejection_penalty: 10.0,
            maximize: false,
            maximize_using_mh_bounds: false,
            mode_compute: 5,
            optim: None,
            maximize_bounds: None,
        }
    }
}

/// `options_.posterior_sampler_options.tarb`.
#[derive(Clone, Debug)]
pub struct TarbOptions {
    /// `mode_compute` of the block optimizer (4 = csminwel).
    pub mode_compute: i32,
    /// `new_block_probability` (0.25).
    pub new_block_probability: f64,
    /// `optim`: optimizer options for the block optimizer (`options_.optim_opt` of the
    /// estimation when not given, see [`SamplerOptions::with_estimation_optim`]).
    pub optim: Option<OptValue>,
}

impl Default for TarbOptions {
    fn default() -> Self {
        TarbOptions { mode_compute: 4, new_block_probability: 0.25, optim: None }
    }
}

/// Posterior sampler settings (`check_posterior_sampler_options.m`).
#[derive(Clone, Debug)]
pub struct SamplerOptions {
    pub method: SamplingMethod,
    /// RW, IMH and TaRB proposal.
    pub proposal: Proposal,
    /// `student_degrees_of_freedom` (3).
    pub student_df: f64,
    /// `use_mh_covariance_matrix` (only meaningful with `load_mh_file`, which is
    /// not supported: kept for the driver).
    pub use_mh_covariance_matrix: bool,
    /// `scale_file`: .mat file whose `Scale` becomes `mh_jscale` (see [`load_scale_file`]).
    pub scale_file: Option<String>,
    pub slice: SliceOptions,
    pub tarb: TarbOptions,
    /// Priors, for the samplers that draw from them (slice initial values without
    /// `slice_initialize_with_mode`, slice restarts, `use_prior_draws`) and the
    /// prior standard deviations of TaRB's newrat. Set with [`SamplerOptions::with_priors`].
    pub priors: Option<Vec<Prior>>,
}

impl Default for SamplerOptions {
    fn default() -> Self {
        SamplerOptions {
            method: SamplingMethod::RandomWalk,
            proposal: Proposal::Normal,
            student_df: 3.0,
            use_mh_covariance_matrix: false,
            scale_file: None,
            slice: SliceOptions::default(),
            tarb: TarbOptions::default(),
            priors: None,
        }
    }
}

impl SamplerOptions {
    /// From the estimation options `posterior_sampling_method` and
    /// `posterior_sampler_options` (key/value list, as `sampling_opt`).
    /// Unknown keys print Dynare's warning; invalid values are errors.
    pub fn from_options(method: Option<&OptValue>, sampler_options: Option<&OptValue>) -> Result<Self> {
        let mut so = SamplerOptions::default();
        let name = match method {
            None => "random_walk_metropolis_hastings".to_string(),
            Some(v) => kv_str("posterior_sampling_method", v)?,
        };
        so.method = match name.as_str() {
            "random_walk_metropolis_hastings" => SamplingMethod::RandomWalk,
            "independent_metropolis_hastings" => SamplingMethod::Independent,
            "slice" => SamplingMethod::Slice,
            "tailored_random_block_metropolis_hastings" => SamplingMethod::TaRB,
            "hssmc" | "dime_mcmc" | "dsmh" | "online" => {
                bail!("posterior_sampling_method = {name} is not ported to dynare-rust")
            }
            other => bail!("check_posterior_sampler_options:: Unknown posterior_sampling_method option {other}"),
        };
        let who = match so.method {
            SamplingMethod::RandomWalk => "rwmh_sampler",
            SamplingMethod::Independent => "imh_sampler",
            SamplingMethod::Slice => "slice_sampler",
            SamplingMethod::TaRB => "tarb_sampler",
        };
        for (k, v) in key_values(sampler_options, "posterior_sampler_options")? {
            let m = so.method;
            let flag = |v: &OptValue| -> Result<bool> { Ok(kv_num(&k, v)? != 0.0) };
            match (k.as_str(), m) {
                (
                    "proposal_distribution",
                    SamplingMethod::RandomWalk | SamplingMethod::Independent | SamplingMethod::TaRB,
                ) => {
                    let s = kv_str(&k, &v)?;
                    so.proposal = if s.eq_ignore_ascii_case("rand_multivariate_student") {
                        Proposal::Student
                    } else if s.eq_ignore_ascii_case("rand_multivariate_normal") {
                        Proposal::Normal
                    } else {
                        bail!(
                            "initial_estimation_checks:: the proposal_distribution option to estimation takes either \
                             rand_multivariate_student or rand_multivariate_normal as options"
                        )
                    };
                }
                (
                    "student_degrees_of_freedom",
                    SamplingMethod::RandomWalk | SamplingMethod::Independent | SamplingMethod::TaRB,
                ) => {
                    let d = kv_num(&k, &v)?;
                    if d <= 0.0 {
                        bail!(
                            "initial_estimation_checks:: the student_degrees_of_freedom takes a positive integer argument"
                        );
                    }
                    so.student_df = d;
                }
                (
                    "use_mh_covariance_matrix",
                    SamplingMethod::RandomWalk | SamplingMethod::Independent | SamplingMethod::Slice,
                ) => so.use_mh_covariance_matrix = flag(&v)?,
                ("scale_file", SamplingMethod::RandomWalk | SamplingMethod::TaRB) => {
                    so.scale_file = Some(kv_str(&k, &v)?)
                }
                ("save_tmp_file", _) | ("save_iter_info_file", SamplingMethod::Slice) => {} // no tmp files
                ("mode_compute", SamplingMethod::TaRB) => so.tarb.mode_compute = kv_num(&k, &v)? as i32,
                ("optim", SamplingMethod::TaRB) => so.tarb.optim = Some(v.clone()),
                ("new_block_probability", SamplingMethod::TaRB) => {
                    let p = kv_num(&k, &v)?;
                    if !(0.0..=1.0).contains(&p) {
                        bail!(
                            "check_posterior_sampler_options:: The tarb new_block_probability must be between 0 and 1!"
                        );
                    }
                    so.tarb.new_block_probability = p;
                }
                ("rotated", SamplingMethod::Slice) => so.slice.rotated = flag(&v)?,
                ("mode_files", SamplingMethod::Slice) => so.slice.mode_files = Some(kv_str(&k, &v)?),
                ("mode", SamplingMethod::Slice) => match &v {
                    OptValue::Vec(x) => so.slice.modes = vec![x.clone()],
                    OptValue::List(cols) => {
                        so.slice.modes = cols
                            .iter()
                            .map(|c| match c {
                                OptValue::Vec(x) => Ok(x.clone()),
                                _ => Err(anyhow!("slice: option mode must be a list of vectors")),
                            })
                            .collect::<Result<_>>()?
                    }
                    _ => bail!("slice: option mode must be a list of vectors (one per mode)"),
                },
                ("slice_initialize_with_mode", SamplingMethod::Slice) => so.slice.initialize_with_mode = flag(&v)?,
                ("initial_step_size", SamplingMethod::Slice) => {
                    let s = kv_num(&k, &v)?;
                    if s <= 0.0 || s >= 1.0 {
                        bail!("check_posterior_sampler_options:: slice initial_step_size must be between 0 and 1");
                    }
                    so.slice.initial_step_size = s;
                }
                ("fast_likelihood_evaluation_for_rejection", SamplingMethod::Slice) => {
                    so.slice.fast_likelihood_evaluation_for_rejection = flag(&v)?
                }
                ("fast_likelihood_evaluation_for_rejection_penalty", SamplingMethod::Slice) => {
                    so.slice.fast_likelihood_evaluation_for_rejection_penalty = kv_num(&k, &v)?
                }
                ("use_prior_draws", SamplingMethod::Slice) => so.slice.use_prior_draws = flag(&v)?,
                // ---- estimation completion: draw_init_state_from_smoother is the default with
                // estimate_initial_states_endogenous_prior (check_posterior_sampler_options.m)
                ("draw_init_state_from_smoother", SamplingMethod::Slice) => {
                    flag(&v)?;
                }
                ("draw_init_state_with_rotated_slice", SamplingMethod::Slice) => {
                    if flag(&v)? {
                        bail!(
                            "slice: option {k} (rotated slice steps along the initial-state prior, get_init_state_prior.m) is not ported to dynare-rust"
                        );
                    }
                }
                ("maximize", SamplingMethod::Slice) => so.slice.maximize = flag(&v)?,
                ("maximize_using_mh_bounds", SamplingMethod::Slice) => so.slice.maximize_using_mh_bounds = flag(&v)?,
                ("mode_compute", SamplingMethod::Slice) => so.slice.mode_compute = kv_num(&k, &v)? as i32,
                _ => println!("WARNING: {who}: Unknown option ({k})!"),
            }
        }
        if so.method == SamplingMethod::Slice {
            if let Some(f) = &so.slice.mode_files {
                so.slice.modes = load_modes(f)?;
            }
            if !so.slice.modes.is_empty() {
                if so.slice.modes.len() < 2 {
                    bail!("check_posterior_sampler_options:: multimodal slice sampling needs at least two modes");
                }
                so.slice.rotated = true;
            }
        }
        Ok(so)
    }

    /// Supplies the priors (see [`SamplerOptions::priors`]).
    pub fn with_priors(mut self, priors: &[Prior]) -> Self {
        self.priors = Some(priors.to_vec());
        self
    }

    /// The estimation's `optim` (`options_.optim_opt`): posterior_sampler.m replaces it by the
    /// TaRB sampler's own `optim` only when that one is non-empty, so the block optimizer
    /// (`posterior_sampler_iteration.m`) runs with the estimation's options otherwise.
    pub fn with_estimation_optim(mut self, optim: Option<&OptValue>) -> Self {
        self.slice.optim = optim.cloned();
        if self.tarb.optim.is_none() {
            self.tarb.optim = optim.cloned();
        }
        self
    }

    /// `options_.mh_posterior_mode_estimation` set by the slice sampler without
    /// `slice_initialize_with_mode`: Dynare then skips mode finding and the
    /// Hessian (`mode_compute = 0`, `cova_compute = 0`), starts a single chain
    /// at the prior mode ([`prior_mode_start`]) and draws several chains' starts
    /// from the prior.
    pub fn mh_posterior_mode_estimation(&self) -> bool {
        self.method == SamplingMethod::Slice && !self.slice.initialize_with_mode
    }

    /// Short name as in Dynare's progress bars.
    pub fn title(&self) -> &'static str {
        match self.method {
            SamplingMethod::RandomWalk => "RW Metropolis-Hastings",
            SamplingMethod::Independent => "Ind. Metropolis-Hastings",
            SamplingMethod::Slice => "SLICE",
            SamplingMethod::TaRB => "TaRB Metropolis-Hastings",
        }
    }
}

/// `check_posterior_sampler_options.m` on `maximize_using_mh_bounds` (the slice sampler's
/// `maximize` within the range of the previous MCMC's draws): the option needs `maximize` and
/// `load_mh_file`, and excludes `mh_recover`.
pub fn check_maximize(so: &SamplerOptions, load_mh_file: bool, mh_recover: bool) -> Result<()> {
    if so.method != SamplingMethod::Slice || !so.slice.maximize_using_mh_bounds {
        return Ok(());
    }
    if !so.slice.maximize {
        bail!("check_posterior_sampler_options:: the maximize_using_mh_bounds requires the maximize option.");
    }
    if !load_mh_file {
        bail!("check_posterior_sampler_options:: the maximize_using_mh_bounds requires the load_mh_file option.");
    }
    if mh_recover {
        bail!(
            "check_posterior_sampler_options:: the maximize_using_mh_bounds is incompatible with the mh_recover option."
        );
    }
    Ok(())
}

/// `check_posterior_sampler_options.m` with `maximize` and `maximize_using_mh_bounds`
/// (after [`check_maximize`]): `GetAllPosteriorDraws(options_, dname, fname, 'all')` (the
/// kept draws of the previous MCMC, in a matrix of `NumberOfDraws*nblck` rows preallocated
/// with zeros, `NumberOfDraws` following the current `mh_drop`), and the bounds
/// `lb = max(min(lb*0.99, lb-1e-6), (lb+mh_lb)/2)`, `ub = min(max(ub*1.01, ub+1e-6), (ub+mh_ub)/2)`
/// around their range.
pub fn maximize_bounds(
    so: &mut SamplerOptions,
    dname: &std::path::Path,
    fname: &str,
    mh_drop: f64,
    mh_bounds: &Bounds,
) -> Result<()> {
    use super::mh_files::{MhFile, MhPaths, load_last_record};
    if so.method != SamplingMethod::Slice || !(so.slice.maximize && so.slice.maximize_using_mh_bounds) {
        return Ok(());
    }
    let paths = MhPaths::new(dname, fname);
    let (rec, _) = load_last_record(&paths)?;
    let kd =
        rec.keeped_draws.clone().ok_or_else(|| anyhow!("GetAllPosteriorDraws: the mh history has no KeepedDraws"))?;
    let npar = rec.npar();
    let total = rec.total_draws();
    let number_of_draws = total - (mh_drop * total as f64).floor() as usize;
    let (mut lo, mut hi) = (vec![f64::INFINITY; npar], vec![f64::NEG_INFINITY; npar]);
    let mut rows = 0;
    for b in 1..=rec.nblck {
        let mut first_line = kd.first_line;
        for k in kd.first_mh_file..=rec.total_files() {
            let f = MhFile::read(&paths.mh_file(k, b))?;
            for t in first_line.saturating_sub(1)..f.rows() {
                for (i, &v) in f.row(t).iter().enumerate() {
                    lo[i] = lo[i].min(v);
                    hi[i] = hi[i].max(v);
                }
                rows += 1;
            }
            first_line = 1;
        }
    }
    if rows < number_of_draws * rec.nblck {
        // rows of the preallocated zeros(NumberOfDraws*nblcks, npar) left unfilled
        for i in 0..npar {
            lo[i] = lo[i].min(0.0);
            hi[i] = hi[i].max(0.0);
        }
    }
    let lb = (0..npar).map(|i| (lo[i] * 0.99).min(lo[i] - 1e-6).max(0.5 * (lo[i] + mh_bounds.lb[i]))).collect();
    let ub = (0..npar).map(|i| (hi[i] * 1.01).max(hi[i] + 1e-6).min(0.5 * (hi[i] + mh_bounds.ub[i]))).collect();
    so.slice.maximize_bounds = Some(Bounds { lb, ub });
    Ok(())
}

/// `load(mode_files, 'xparams')`: one mode per column.
fn load_modes(path: &str) -> Result<Vec<Vec<f64>>> {
    let p = std::path::Path::new(path);
    let p = if p.exists() { p.to_path_buf() } else { p.with_extension("mat") };
    let vars = crate::matfile::read(&p)?;
    let x = vars
        .get("xparams")
        .and_then(|v| v.as_matrix())
        .ok_or_else(|| anyhow!("slice: no matrix xparams in {}", p.display()))?;
    if x.ncols() < 2 {
        bail!(
            "check_posterior_sampler_options:: Variable xparams loaded in file <{path}> has size [{}x{}]: it must contain \
             at least two columns, to allow multi-modal sampling.",
            x.nrows(),
            x.ncols()
        );
    }
    Ok((0..x.ncols()).map(|j| x.column(j).iter().copied().collect()).collect())
}

/// `scale_file`: the `Scale` saved by `mode_compute = 6`.
pub fn load_scale_file(path: &str) -> Result<f64> {
    let p = std::path::Path::new(path);
    let p = if p.exists() { p.to_path_buf() } else { p.with_extension("mat") };
    if !p.exists() {
        bail!("initial_estimation_checks:: The specified scale_file does not exist.");
    }
    crate::matfile::read(&p)?
        .get("Scale")
        .and_then(|v| v.as_scalar())
        .ok_or_else(|| anyhow!("scale_file {}: no scalar Scale", p.display()))
}

/// `dynare_estimation_1.m` under `mh_posterior_mode_estimation`: the start is
/// the prior mode (`p5`), the prior mean where it is NaN or outside the bounds.
pub fn prior_mode_start(priors: &[Prior], bounds: &Bounds) -> Vec<f64> {
    priors
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let x = if p.p5.is_nan() { p.mean } else { p.p5 };
            if x < bounds.lb[i] || x > bounds.ub[i] { p.mean } else { x }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// proposal distributions (distributions/*.m)
// ---------------------------------------------------------------------------

/// A proposal `N(mean, U'U)` or multivariate Student with `df` degrees of freedom.
#[derive(Clone, Debug)]
pub(crate) struct ProposalDist {
    pub kind: Proposal,
    pub df: f64,
    /// Upper triangular Cholesky factor (`Sigma_upper_chol`).
    pub u: Mat,
}

impl ProposalDist {
    /// A multivariate Student proposal with upper Cholesky factor `u` (DIME).
    pub(crate) fn student(u: Mat, df: f64) -> Self {
        ProposalDist { kind: Proposal::Student, df, u }
    }

    /// `rand_multivariate_normal(mean, U, n)` = `mean + randn(1,n)*U`, or
    /// `rand_multivariate_student(mean, U, df)` =
    /// `mean + (randn(1,dim)*U) .* sqrt(df/sum(randn(1,df).^2))`.
    pub fn draw<R: Rng>(&self, rng: &mut R, mean: &[f64], out: &mut [f64]) {
        let n = mean.len();
        let z: Vec<f64> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
        let scale = match self.kind {
            Proposal::Normal => 1.0,
            Proposal::Student => {
                let k = self.df.round().max(1.0) as usize;
                let s: f64 = (0..k).map(|_| rng.sample::<f64, _>(StandardNormal).powi(2)).sum();
                (self.df / s).sqrt()
            }
        };
        for j in 0..n {
            let mut s = 0.0;
            for i in 0..=j {
                s += z[i] * self.u[(i, j)];
            }
            out[j] = mean[j] + s * scale;
        }
    }

    /// `log(multivariate_normal_pdf(x, mean, U, n))` with `n = dim`, or
    /// `log(multivariate_student_pdf(x, mean, U, df))`, computed in logs.
    pub fn logpdf(&self, x: &[f64], mean: &[f64]) -> f64 {
        let n = x.len();
        // z = (x-mean)/U: z*U = d  <=>  U' z' = d' (forward substitution).
        let mut z = vec![0.0; n];
        for j in 0..n {
            let mut s = x[j] - mean[j];
            for i in 0..j {
                s -= z[i] * self.u[(i, j)];
            }
            z[j] = s / self.u[(j, j)];
        }
        let q: f64 = z.iter().map(|v| v * v).sum();
        let logdet: f64 = (0..n).map(|i| self.u[(i, i)].ln()).sum();
        let nf = n as f64;
        match self.kind {
            Proposal::Normal => -0.5 * nf * (2.0 * std::f64::consts::PI).ln() - logdet - 0.5 * q,
            Proposal::Student => {
                let df = self.df;
                let t1 = crate::estimation::special::gammaln(0.5 * (nf + df))
                    - crate::estimation::special::gammaln(0.5 * df)
                    - 0.5 * nf * (df * std::f64::consts::PI).ln();
                t1 - logdet - 0.5 * (nf + df) * (1.0 + q / df).ln()
            }
        }
    }
}

/// `d*diag(jscale)`.
pub(crate) fn scaled_chol(d: &Mat, jscale: &[f64]) -> Mat {
    let mut u = d.clone();
    for j in 0..u.ncols() {
        for i in 0..u.nrows() {
            u[(i, j)] *= jscale[j];
        }
    }
    u
}

/// MATLAB `randperm(n)` as `[~, p] = sort(rand(1, n))` (0-based).
pub(crate) fn randperm<R: Rng>(rng: &mut R, n: usize) -> Vec<usize> {
    let u: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>()).collect();
    let mut p: Vec<usize> = (0..n).collect();
    p.sort_by(|&a, &b| u[a].total_cmp(&u[b]));
    p
}

#[inline]
pub(crate) fn strictly_inside(p: &[f64], b: &Bounds) -> bool {
    p.iter().zip(&b.lb).zip(&b.ub).all(|((v, l), u)| v > l && v < u)
}

pub(crate) fn rng_stream(seed: u64, stream: u64) -> ChaCha8Rng {
    let mut r = ChaCha8Rng::seed_from_u64(seed);
    r.set_stream(stream);
    r
}

// ---------------------------------------------------------------------------
// chain runner (posterior_sampler_core.m bookkeeping)
// ---------------------------------------------------------------------------

/// One chain of a sampler: `advance` runs between 1 and `max_steps`
/// iterations, calling `on_step(current draw, its log posterior, accepted
/// (fraction for TaRB), function evaluations)` after each.
pub(crate) trait Stepper {
    fn advance(&mut self, max_steps: usize, on_step: &mut dyn FnMut(&[f64], f64, f64, usize)) -> Result<usize>;
    /// The chain's generator after the last iteration (saved in the mh files).
    fn rng(&self) -> &ChaCha8Rng;
}

/// Output of [`posterior_sampler_full`].
#[derive(Clone, Debug)]
pub struct SamplerResult {
    pub mh: MhResult,
    /// `record.AcceptanceRatio` per chain (fractional for TaRB, 1 for the slice).
    pub acceptance: Vec<f64>,
    /// `record.FunctionEvalPerIteration` per chain (Dynare's count, which
    /// includes the slice sampler's re-evaluation of the current point).
    pub evals_per_draw: Vec<f64>,
}

type MakeStepper<'a> = dyn Fn(usize, &[f64], f64) -> Result<Box<dyn Stepper + 'a>> + Sync + 'a;

fn run_chains<'a>(
    ix2: &[Vec<f64>],
    ilogpo2: &[f64],
    opts: &MhOptions,
    progress: Option<&(dyn Fn(&MhProgress) + Sync)>,
    make: &MakeStepper<'a>,
) -> Result<(Vec<Chain>, Vec<f64>, Vec<f64>)> {
    let npar = ix2[0].len();
    let nkeep = opts.replic.saturating_sub(opts.store_from);
    let ses = opts.session.as_deref();
    // options_.gradient_epsilon/method of the slice sampler's maximize and TaRB's optimizer
    let grad = crate::estimation::optim::minimize::gradient_options();
    let run_chain = |b: usize| -> Result<(Chain, f64, f64)> {
        let _grad = crate::estimation::optim::minimize::GradientGuard::set(grad);
        let nruns = ses.map_or(opts.replic, |s| s.chain_rng(b, opts.seed, opts.replic).1);
        let mut writer = ses.and_then(|s| s.take_writer(b));
        let mut st = make(b, &ix2[b], ilogpo2[b])?;
        let mut draws = Vec::with_capacity(nkeep * npar);
        let mut logpost = Vec::with_capacity(nkeep);
        let (mut it, mut acc_sum, mut nevals) = (0usize, 0.0f64, 0usize);
        let (mut last, mut last_lp) = (ix2[b].clone(), ilogpo2[b]);
        let mut done = 0;
        while done < nruns {
            let lim = writer.as_ref().map_or(usize::MAX, |wr| wr.until_flush());
            let mut on_step = |x: &[f64], lp: f64, acc: f64, ne: usize| {
                acc_sum += acc;
                nevals += ne;
                if let Some(wr) = writer.as_mut() {
                    wr.push(x, lp, acc, ne as f64);
                }
                if it >= opts.store_from {
                    draws.extend_from_slice(x);
                    logpost.push(lp);
                }
                if ses.is_some() {
                    last.copy_from_slice(x);
                    last_lp = lp;
                }
                it += 1;
                if let Some(cb) = progress
                    && opts.progress_every > 0
                    && it.is_multiple_of(opts.progress_every)
                {
                    cb(&MhProgress { chain: b, draws_done: it, total: nruns, acceptance: acc_sum / it as f64 });
                }
            };
            done += st.advance((nruns - done).min(lim), &mut on_step)?;
            if let Some(wr) = writer.as_mut() {
                wr.after(st.rng());
            }
        }
        if let Some(s) = ses {
            s.finish_chain(b, writer, st.rng(), &last, last_lp, acc_sum, nevals as f64, nruns);
        }
        let chain = Chain {
            npar,
            draws,
            logpost,
            runs: Vec::new(),
            first_stored: opts.store_from.min(nruns),
            total_draws: nruns,
            accepted: acc_sum.round() as usize,
            init: ix2[b].clone(),
            init_logpost: ilogpo2[b],
        };
        let n = nruns.max(1) as f64;
        Ok((chain, acc_sum / n, nevals as f64 / n))
    };
    let out: Vec<Result<(Chain, f64, f64)>> = if opts.nblocks > 1 && opts.nblocks <= rayon::current_num_threads() {
        std::thread::scope(|sc| {
            let run_chain = &run_chain;
            let hs: Vec<_> = (0..opts.nblocks)
                .map(|b| {
                    sc.spawn(move || {
                        crate::parallel::set_current_thread_high_qos();
                        run_chain(b)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("MCMC chain panicked")).collect()
        })
    } else {
        use rayon::prelude::*;
        (0..opts.nblocks).into_par_iter().map(run_chain).collect()
    };
    let mut chains = Vec::new();
    let mut acc = Vec::new();
    let mut ev = Vec::new();
    for r in out {
        let (c, a, e) = r?;
        chains.push(c);
        acc.push(a);
        ev.push(e);
    }
    Ok((chains, acc, ev))
}

/// `posterior_sampler_initialization.m` when `d = chol(vv)` is empty (slice
/// without `slice_initialize_with_mode`): several chains start at prior draws
/// inside the bounds with a finite posterior, one chain at `xparam1`.
pub(crate) fn prior_initial_values(
    obj: &dyn Objective,
    xparam1: &[f64],
    priors: Option<&[Prior]>,
    bounds: &Bounds,
    opts: &MhOptions,
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    let npar = xparam1.len();
    let inside = |c: &[f64]| (0..npar).all(|i| c[i] >= bounds.lb[i] && c[i] <= bounds.ub[i]);
    // ---- estimation completion: estimate_initial_states_endogenous_prior draws the initial
    // states of each start (draw_init_state_from_smoother.m with init = true)
    let init_states = |c: Vec<f64>, rng: &mut rand_chacha::ChaCha8Rng| match obj.init_state_ops() {
        Some(ops) => ops.draw_init_state(true, 1, &c, f64::NAN, bounds, rng).0,
        None => c,
    };
    if opts.nblocks <= 1 {
        if !inside(xparam1) {
            bail!("posterior_sampler_initialization: The prior mode lies outside the prior bounds.");
        }
        let x = init_states(xparam1.to_vec(), &mut rng_stream(opts.seed, 0));
        return Ok((vec![x.clone()], vec![-obj.eval(&x).0]));
    }
    let priors = priors.ok_or_else(|| anyhow!("slice sampler: the priors are needed to draw the initial values"))?;
    let mut rng = rng_stream(opts.seed, 0);
    let (mut ix2, mut lp) = (Vec::new(), Vec::new());
    for j in 0..opts.nblocks {
        let (mut validate, mut init_iter, mut trial) = (false, 0, 1);
        while !validate && trial <= 10 {
            let cand = prior_draw(priors, bounds, &mut rng);
            if inside(&cand) {
                let cand = init_states(cand, &mut rng); // ---- estimation completion
                let l = -obj.eval(&cand).0;
                if l.is_finite() {
                    ix2.push(cand);
                    lp.push(l);
                    validate = true;
                }
            }
            init_iter += 1;
            if !validate && init_iter > 100 {
                trial += 1;
            }
        }
        if !validate {
            bail!("posterior_sampler_initialization: I'm unable to find a starting value for block {}", j + 1);
        }
    }
    Ok((ix2, lp))
}

/// [`posterior_sampler_full`] without the per-chain records.
pub fn posterior_sampler(
    obj: &dyn Objective,
    xparam1: &[f64],
    invhess: &Mat,
    bounds: &Bounds,
    mh: &MhOptions,
    so: &SamplerOptions,
    progress: Option<&(dyn Fn(&MhProgress) + Sync)>,
) -> Result<MhResult> {
    posterior_sampler_full(obj, xparam1, invhess, bounds, mh, so, progress).map(|r| r.mh)
}

/// Port of `posterior_sampler.m` for every ported `posterior_sampling_method`.
/// `invhess` is the proposal covariance (`d = chol(invhess)`, scaled by
/// `mh.jscale`); the slice sampler without `slice_initialize_with_mode` ignores it.
/// `bounds` are the MCMC bounds ([`crate::estimation::mcmc_bounds`]).
pub fn posterior_sampler_full(
    obj: &dyn Objective,
    xparam1: &[f64],
    invhess: &Mat,
    bounds: &Bounds,
    mh: &MhOptions,
    so: &SamplerOptions,
    progress: Option<&(dyn Fn(&MhProgress) + Sync)>,
) -> Result<SamplerResult> {
    let npar = xparam1.len();
    if mh.jscale.len() != npar {
        bail!("posterior_sampler: jscale has {} entries for {npar} parameters", mh.jscale.len());
    }
    if so.method == SamplingMethod::RandomWalk && so.proposal == Proposal::Normal {
        let res = rw_metropolis(obj, xparam1, invhess, bounds, mh, progress)?;
        let acceptance = res.chains.iter().map(|c| c.acceptance_ratio()).collect();
        let n = res.chains.len();
        return Ok(SamplerResult { mh: res, acceptance, evals_per_draw: vec![1.0; n] });
    }
    let uses_hessian = so.method != SamplingMethod::Slice || so.slice.initialize_with_mode;
    let ses = mh.session.as_deref();
    let d = if uses_hessian {
        match ses.and_then(|s| s.proposal_chol.clone()) {
            Some(d) => Some(d),
            None => Some(
                chol_upper(invhess)
                    .ok_or_else(|| anyhow!("posterior_sampler: the proposal covariance is not positive definite"))?,
            ),
        }
    } else {
        None
    };
    let t0 = std::time::Instant::now();
    let (ix2, ilogpo2, factor, eval_secs) = match ses.and_then(|s| s.start_points()) {
        Some((x, l)) => (x, l, mh.init_scale_factor, ses.unwrap().eval_secs),
        None => {
            let (x, l, f) = match &d {
                Some(d) => initial_values(obj, xparam1, d, bounds, mh)?,
                None => {
                    let (x, l) = prior_initial_values(obj, xparam1, so.priors.as_deref(), bounds, mh)?;
                    (x, l, mh.init_scale_factor)
                }
            };
            (x, l, f, t0.elapsed().as_secs_f64() / mh.nblocks.max(1) as f64)
        }
    };
    // chain b's generator: a saved state when continuing, else stream b+1 of the seed
    let chain_rng =
        |b: usize| ses.map_or_else(|| rng_stream(mh.seed, b as u64 + 1), |s| s.chain_rng(b, mh.seed, mh.replic).0);
    let width = prefetch_width(mh.prefetch, mh.nblocks, eval_secs);
    let prop = d.as_ref().map(|d| ProposalDist { kind: so.proposal, df: so.student_df, u: scaled_chol(d, &mh.jscale) });
    let (chains, acceptance, evals_per_draw) = match so.method {
        SamplingMethod::RandomWalk => {
            let prop = prop.unwrap();
            let make = |b: usize, x: &[f64], lp: f64| -> Result<Box<dyn Stepper + '_>> {
                Ok(Box::new(rwmh::RwStudent::new(obj, bounds, chain_rng(b), x, lp, prop.clone())))
            };
            run_chains(&ix2, &ilogpo2, mh, progress, &make)?
        }
        SamplingMethod::Independent => {
            let prop = prop.unwrap();
            let make = |b: usize, x: &[f64], lp: f64| -> Result<Box<dyn Stepper + '_>> {
                Ok(Box::new(imh::Imh::new(obj, bounds, chain_rng(b), x, lp, xparam1, prop.clone(), width)))
            };
            run_chains(&ix2, &ilogpo2, mh, progress, &make)?
        }
        SamplingMethod::Slice => {
            let loaded = so.use_mh_covariance_matrix && ses.is_some_and(|s| s.start_points().is_some());
            let setup = slice::SliceSetup::new(so, invhess, bounds, npar, loaded)?;
            if so.slice.maximize {
                let mb = so.slice.maximize_bounds.as_ref().unwrap_or(bounds);
                let make = |b: usize, x: &[f64], _lp: f64| -> Result<Box<dyn Stepper + '_>> {
                    Ok(Box::new(slice::Maximize::new(obj, mb, so, chain_rng(b), x, b)))
                };
                return finish(run_chains(&ix2, &ilogpo2, mh, progress, &make)?, factor, width);
            }
            let make = |b: usize, x: &[f64], lp: f64| -> Result<Box<dyn Stepper + '_>> {
                Ok(Box::new(slice::Slice::new(obj, &setup, so.priors.as_deref(), chain_rng(b), x, lp, width)))
            };
            run_chains(&ix2, &ilogpo2, mh, progress, &make)?
        }
        SamplingMethod::TaRB => {
            let make = |b: usize, x: &[f64], lp: f64| -> Result<Box<dyn Stepper + '_>> {
                Ok(Box::new(tarb::Tarb::new(obj, bounds, so, &mh.jscale, chain_rng(b), x, lp, b)))
            };
            run_chains(&ix2, &ilogpo2, mh, progress, &make)?
        }
    };
    finish((chains, acceptance, evals_per_draw), factor, width)
}

fn finish(
    (chains, acceptance, evals_per_draw): (Vec<Chain>, Vec<f64>, Vec<f64>),
    factor: f64,
    width: usize,
) -> Result<SamplerResult> {
    Ok(SamplerResult {
        mh: MhResult { chains, init_scale_factor: factor, prefetch: width },
        acceptance,
        evals_per_draw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn student_pdf_matches_dynare_unit_test() {
        // multivariate_student_pdf([1 2],0,chol([1 0.5; 0.5 1]),10) = 0.02440738691918476
        let c = chol_upper(&Mat::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0])).unwrap();
        let p = ProposalDist { kind: Proposal::Student, df: 10.0, u: c };
        let v = p.logpdf(&[1.0, 2.0], &[0.0, 0.0]).exp();
        assert!((v - 0.02440738691918476).abs() < 1e-12, "{v}");
    }

    #[test]
    fn options_parsing() {
        let s = |x: &str| OptValue::Str(format!("'{x}'"));
        let so = SamplerOptions::from_options(
            Some(&s("tailored_random_block_metropolis_hastings")),
            Some(&OptValue::List(vec![
                s("proposal_distribution"),
                s("rand_multivariate_student"),
                s("student_degrees_of_freedom"),
                OptValue::Num(5.0),
                s("new_block_probability"),
                OptValue::Num(0.4),
                s("mode_compute"),
                OptValue::Num(5.0),
            ])),
        )
        .unwrap();
        assert_eq!(so.method, SamplingMethod::TaRB);
        assert_eq!(so.proposal, Proposal::Student);
        assert_eq!(so.student_df, 5.0);
        assert_eq!(so.tarb.new_block_probability, 0.4);
        assert_eq!(so.tarb.mode_compute, 5);
        let so = SamplerOptions::from_options(
            Some(&s("slice")),
            Some(&OptValue::List(vec![s("rotated"), OptValue::Num(1.0), s("initial_step_size"), OptValue::Num(0.5)])),
        )
        .unwrap();
        assert!(so.slice.rotated && so.slice.initial_step_size == 0.5 && so.mh_posterior_mode_estimation());
        assert!(
            SamplerOptions::from_options(
                Some(&s("slice")),
                Some(&OptValue::List(vec![s("initial_step_size"), OptValue::Num(1.5)]))
            )
            .is_err()
        );
        assert!(SamplerOptions::from_options(Some(&s("foo")), None).is_err());
        assert_eq!(SamplerOptions::from_options(None, None).unwrap().method, SamplingMethod::RandomWalk);
    }
}
