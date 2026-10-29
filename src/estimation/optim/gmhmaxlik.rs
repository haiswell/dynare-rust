// SPDX-License-Identifier: GPL-3.0-or-later
//! `mode_compute = 6`: ports of `optimization/gmhmaxlik.m` and
//! `optimization/gmhmaxlik_core.m` (Dynare 7.2), the "(dirty) global
//! minimization routine" built on random-walk Metropolis–Hastings:
//!
//! 1. tune the jump scale towards the target acceptance rate (windows of 500 draws),
//! 2. run a one-block Metropolis chain of `number` draws, updating the posterior mean
//!    and covariance recursively,
//! 3. (last of the `iterations` calls) re-tune the scale with the new covariance
//!    (windows of 1000 draws), then
//! 4. climb the hill: random-walk proposals around the best point, kept only when
//!    they improve the posterior, cooling the scale when progress stalls.
//!
//! Returns the best point visited, the inverse of the last estimated covariance
//! (`HessianMatrix = inv(PostVariance)`, not rescaled), and the tuned scale, which
//! `dynare_estimation_1.m` stores as `options_.mh_jscale` for the MCMC.
//!
//! Every Metropolis phase runs on the pre-fetching walker of
//! [`crate::estimation::mcmc`] and the climb pre-fetches the next proposals, so
//! the posterior evaluations run in parallel while every proposal, uniform and
//! decision is exactly that of the sequential algorithm (whatever
//! [`GmhmaxlikOptions::prefetch`]). The random stream is ChaCha8 (stream
//! [`GmhmaxlikOptions::stream`] of [`GmhmaxlikOptions::seed`]), not MATLAB's.

use anyhow::{Result, bail};
use rand::SeedableRng;
use rayon::prelude::*;

use crate::estimation::mcmc::{Walker, logpost_of, prefetch_width, rand_mvn, rng_stream};
use crate::estimation::{Bounds, Objective};
use crate::linalg::{Mat, chol_upper, solve};
use crate::modfile::OptValue;

/// Initial covariance of the jumping distribution (`gmhmaxlikOptions.varinit`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarInit {
    /// `inv(Hinit)` (the default when an initial Hessian is given).
    Previous,
    /// `diag(priorstd.^2)`, infinite standard deviations replaced by `sqrt(10)`
    /// (the default without initial Hessian).
    Prior,
    /// `0.1*eye(n)`.
    Identity,
}

/// `options_.gmhmaxlik` (defaults from `default_option_values.m`) plus the
/// random stream and the pre-fetching width.
#[derive(Clone, Debug)]
pub struct GmhmaxlikOptions {
    /// Calls of `gmhmaxlik_core` (`iterations`, 3).
    pub iterations: usize,
    /// Draws of the covariance-estimation chain (`number`, 20000).
    pub number: usize,
    /// Maximum draws of each scale-tuning phase (`nscale`, 200000).
    pub nscale: usize,
    /// Maximum draws of the hill climb (`nclimb`, 200000; 0 = no climb).
    pub nclimb: usize,
    /// Target acceptance rate (`target`, 1/3).
    pub target: f64,
    /// `silent`: no progress output.
    pub silent: bool,
    /// Override of the initial covariance (`optim` option `InitialCovarianceMatrix`);
    /// None = Previous if an initial Hessian is given, else Prior (`gmhmaxlik.m`).
    pub varinit: Option<VarInit>,
    pub seed: u64,
    /// ChaCha8 stream of the run (distinct per optimizer step).
    pub stream: u64,
    /// Posterior evaluations run in parallel by pre-fetching
    /// (as [`crate::estimation::MhOptions::prefetch`]: 1 = sequential, 0 = automatic).
    pub prefetch: usize,
}

impl Default for GmhmaxlikOptions {
    fn default() -> Self {
        GmhmaxlikOptions {
            iterations: 3,
            number: 20000,
            nscale: 200_000,
            nclimb: 200_000,
            target: 1.0 / 3.0,
            silent: false,
            varinit: None,
            seed: 0,
            stream: u64::MAX - 1,
            prefetch: 0,
        }
    }
}

impl GmhmaxlikOptions {
    /// The `optim` key/value pairs `gmhmaxlik.m` understands (`NumberOfMh`,
    /// `ncov-mh`, `nscale-mh`, `nclimb-mh`, `InitialCovarianceMatrix`,
    /// `AcceptanceRateTarget`), as parsed from `optim = ('key', value, ...)`.
    pub fn apply_optim(&mut self, optim: &OptValue) -> Result<()> {
        let items: Vec<&OptValue> = match optim {
            OptValue::List(v) => v.iter().collect(),
            other => vec![other],
        };
        if !items.len().is_multiple_of(2) {
            bail!("gmhmaxlik: optim must be a list of key/value pairs");
        }
        for kv in items.chunks(2) {
            let key = match kv[0] {
                OptValue::Str(s) => s.trim_matches(|c| c == '\'' || c == '"').to_string(),
                _ => bail!("gmhmaxlik: optim keys must be strings"),
            };
            let num = || match kv[1] {
                OptValue::Num(v) => Ok(*v),
                _ => Err(anyhow::anyhow!("gmhmaxlik: optim value of {key} must be a number")),
            };
            match key.as_str() {
                "NumberOfMh" => self.iterations = num()? as usize,
                "ncov-mh" => self.number = num()? as usize,
                "nscale-mh" => self.nscale = num()? as usize,
                "nclimb-mh" => self.nclimb = num()? as usize,
                "AcceptanceRateTarget" => {
                    let t = num()?;
                    if t > 1.0 || t < f64::EPSILON {
                        bail!(
                            "gmhmaxlik: The value of option AcceptanceRateTarget should be a double between 0 and 1!"
                        );
                    }
                    self.target = t;
                }
                "InitialCovarianceMatrix" => {
                    let v = match kv[1] {
                        OptValue::Str(s) => s.trim_matches(|c| c == '\'' || c == '"').to_string(),
                        _ => String::new(),
                    };
                    self.varinit = Some(match v.as_str() {
                        "previous" => VarInit::Previous,
                        "prior" => VarInit::Prior,
                        "identity" => VarInit::Identity,
                        _ => bail!("gmhmaxlik: Unknown value for option 'InitialCovarianceMatrix'!"),
                    });
                }
                other => println!("WARNING: gmhmaxlik: Unknown option ({other})!"),
            }
        }
        Ok(())
    }
}

/// Output of [`gmhmaxlik`] (`[PostMode, HessianMatrix, Scale, ModeValue, fcount]`).
#[derive(Clone, Debug)]
pub struct GmhmaxlikResult {
    /// `PostMode`: best point of the last call.
    pub x: Vec<f64>,
    /// `ModeValue`: objective at `x`.
    pub fval: f64,
    /// `HessianMatrix = inv(PostVariance)`.
    pub hessian: Mat,
    /// `PostVariance`: covariance estimated by the last call's Metropolis chain.
    pub post_variance: Mat,
    /// `PostMean` of the last call.
    pub post_mean: Vec<f64>,
    /// `Scale`: tuned jump scale (before the climb's cooling).
    pub scale: f64,
    /// `fcount`: objective evaluations of the sequential algorithm.
    pub func_count: usize,
    /// Pre-fetching width used.
    pub prefetch: usize,
}

/// Port of `gmhmaxlik.m`. `obj` is minimized (`fun` = minus log posterior; +inf
/// marks a failure, which the Metropolis steps reject). `hinit` is
/// `Initial_Hessian` (`hh` of the previous optimizer or of the mode file),
/// `iscale` is `options_.mh_jscale`, `bounds` the estimation bounds (proposals
/// must be strictly inside), `priorstd` is `bayestopt_.p2`.
pub fn gmhmaxlik(
    obj: &dyn Objective,
    xinit: &[f64],
    hinit: Option<&Mat>,
    iscale: f64,
    bounds: &Bounds,
    priorstd: &[f64],
    opts: &GmhmaxlikOptions,
) -> Result<GmhmaxlikResult> {
    let n = xinit.len();
    let varinit = match (opts.varinit, hinit) {
        (Some(VarInit::Previous), None) => bail!(
            "gmhmaxlik: No previous estimate of the Hessian matrix available! You cannot use the InitialCovarianceMatrix option!"
        ),
        (Some(v), _) => v,
        (None, Some(_)) => VarInit::Previous,
        (None, None) => VarInit::Prior,
    };
    let mut fcount = 0usize;
    let t0 = std::time::Instant::now();
    let mut old_mode_value = obj.eval(xinit).0;
    let eval_secs = t0.elapsed().as_secs_f64();
    fcount += 1;
    let cov_jump = match varinit {
        VarInit::Previous => {
            let h = hinit.expect("checked above");
            solve(h, &Mat::identity(n, n))
                .ok_or_else(|| anyhow::anyhow!("gmhmaxlik: the initial Hessian is singular"))?
        }
        VarInit::Prior => Mat::from_diagonal(&nalgebra::DVector::from_iterator(
            n,
            priorstd.iter().map(|&s| if s.is_infinite() { 10.0 } else { s * s }),
        )),
        VarInit::Identity => Mat::identity(n, n) * 0.1,
    };
    let width = prefetch_width(opts.prefetch, 1, eval_secs);
    let mut core = Core { obj, bounds, opts, rng: Some(rng_stream(opts.seed, opts.stream)), width, fcount: 0 };
    let mut old_post_variance = cov_jump;
    let mut old_post_mean = xinit.to_vec();
    let mut scale = iscale;
    let mut last = None;
    for i in 1..=opts.iterations {
        let last_call = i == opts.iterations;
        crate::progress::detail(format!(
            "round {i} of {}: tune scale, {} draws for the covariance{}",
            opts.iterations,
            opts.number,
            if last_call { ", re-tune, hill climb" } else { "" }
        ));
        // gmhmaxlik.m passes OldPostMode = xinit and MeanPar = xinit to every call.
        let out = core.run(xinit, scale, last_call, xinit, &old_post_variance)?;
        scale = out.scale;
        let mode_value = obj.eval(&out.mode).0;
        fcount += 1;
        let d_variance = (&out.variance - &old_post_variance).abs().max();
        let d_mean = out.mean.iter().zip(&old_post_mean).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        if !opts.silent {
            println!();
            println!("{}", "=".repeat(58));
            println!("   Change in the posterior covariance matrix = {}.", crate::gsa::stats::num2str(d_variance));
            println!("   Change in the posterior mean = {}.", crate::gsa::stats::num2str(d_mean));
            println!("   Current mode = {}", crate::gsa::stats::num2str(mode_value));
            println!("   Mode improvement = {}", crate::gsa::stats::num2str((old_mode_value - mode_value).abs()));
            println!("   New value of jscale = {}", crate::gsa::stats::num2str(scale));
            println!("{}", "=".repeat(58));
        }
        old_mode_value = mode_value;
        old_post_mean = out.mean.clone();
        old_post_variance = out.variance.clone();
        last = Some((out, mode_value));
    }
    let Some((out, mode_value)) = last else {
        bail!("gmhmaxlik: options_.gmhmaxlik.iterations must be positive");
    };
    let hessian = solve(&out.variance, &Mat::identity(n, n))
        .ok_or_else(|| anyhow::anyhow!("gmhmaxlik: the estimated posterior covariance matrix is singular"))?;
    if !opts.silent {
        println!("\nOptimal value of the scale parameter = {}\n", crate::gsa::stats::num2str(scale));
    }
    Ok(GmhmaxlikResult {
        x: out.mode,
        fval: mode_value,
        hessian,
        post_variance: out.variance,
        post_mean: out.mean,
        scale,
        func_count: fcount + core.fcount,
        prefetch: width,
    })
}

struct CoreOut {
    mode: Vec<f64>,
    variance: Mat,
    scale: f64,
    mean: Vec<f64>,
}

/// State shared by the calls of `gmhmaxlik_core.m`: the random stream (MATLAB's
/// global stream carries over from call to call) and the evaluation count.
struct Core<'a> {
    obj: &'a dyn Objective,
    bounds: &'a Bounds,
    opts: &'a GmhmaxlikOptions,
    rng: Option<rand_chacha::ChaCha8Rng>,
    width: usize,
    fcount: usize,
}

/// Best point visited (`ModePar`, `mlogpo2`).
struct Best {
    x: Vec<f64>,
    lp: f64,
}

impl Best {
    fn update(&mut self, last: &[f64], lp: f64, acc: bool) {
        if acc && lp > self.lp {
            self.x.copy_from_slice(last);
            self.lp = lp;
        }
    }
}

impl<'a> Core<'a> {
    fn walker(&mut self, x: &[f64], lp: f64) -> Walker<'a> {
        let rng = self.rng.take().unwrap_or_else(|| rand_chacha::ChaCha8Rng::seed_from_u64(0));
        Walker::new(self.obj, self.bounds, rng, x, lp, self.width)
    }

    fn finish(&mut self, w: Walker<'a>) -> (Vec<f64>, f64) {
        self.fcount += w.evaluated;
        self.rng = Some(w.rng);
        (w.last, w.last_lp)
    }

    fn eval_logpost(&mut self, x: &[f64]) -> f64 {
        self.fcount += 1;
        logpost_of(self.obj, x)
    }

    /// Scale tuning of `gmhmaxlik_core.m` (steps [1] and [3]): Metropolis draws
    /// from `ix2` with `iscale*dd`, the scale multiplied every `window` draws by
    /// (acceptance in the window)/target; stops after more than 4 (cumulated)
    /// windows with a factor in (0.9, 1.1). `guard_zero` is step [1]'s
    /// `if cfactor>0 ... else iScale = iScale/10`.
    #[allow(clippy::too_many_arguments)]
    fn tune(
        &mut self,
        ix2: &[f64],
        ilogpo2: f64,
        dd: &Mat,
        mut iscale: f64,
        window: usize,
        guard_zero: bool,
        best: &mut Best,
    ) -> (Vec<f64>, f64, f64) {
        let nscale = self.opts.nscale;
        let target = self.opts.target;
        let mut w = self.walker(ix2, ilogpo2);
        let (mut j, mut jj, mut jsux, mut test) = (1usize, 1usize, 0usize, 0usize);
        let mut u = dd * iscale;
        while j <= nscale {
            let to_boundary = window - (j - 1) % window;
            let max_steps = to_boundary.min(nscale - j + 1);
            let mut acc_batch = 0usize;
            let s = w.advance(&u, max_steps, &mut |last, lp, acc| {
                acc_batch += acc as usize;
                best.update(last, lp, acc);
            });
            jsux += acc_batch;
            j += s - 1;
            jj += s - 1;
            if j.is_multiple_of(window) {
                let test1 = jsux as f64 / jj as f64;
                let cfactor = test1 / target;
                if !guard_zero || cfactor > 0.0 {
                    iscale *= cfactor;
                } else {
                    iscale /= 10.0;
                }
                u = dd * iscale;
                jsux = 0;
                jj = 0;
                if cfactor > 0.90 && cfactor < 1.10 {
                    test += 1;
                }
                if test > 4 {
                    break;
                }
            }
            j += 1;
            jj += 1;
        }
        let (x, lp) = self.finish(w);
        (x, lp, iscale)
    }

    /// Port of `gmhmaxlik_core.m`.
    fn run(
        &mut self,
        xparam1: &[f64],
        iscale: f64,
        last_call: bool,
        mean_par: &[f64],
        var_cov: &Mat,
    ) -> Result<CoreOut> {
        let n = xparam1.len();
        let mut cov_jump = var_cov.clone();
        let mut best = Best { x: xparam1.to_vec(), lp: 0.0 };

        // [1] tune the scale parameter.
        let ilogpo2 = self.eval_logpost(xparam1);
        best.lp = ilogpo2;
        let dd = chol_upper(&cov_jump).unwrap_or_else(|| Mat::identity(n, n));
        let (ix2, _, mut iscale) = self.tune(xparam1, ilogpo2, &dd, iscale, 500, true, &mut best);

        // [2] one-block Metropolis, updating the mean and covariance (with the
        // jump distribution of step [1]).
        let ilogpo2 = self.eval_logpost(&ix2);
        let mut mean = mean_par.to_vec();
        let mut old_mean = vec![0.0; n];
        let mut w = self.walker(&ix2, ilogpo2);
        let u = &dd * iscale;
        let number = self.opts.number;
        let mut j = 1usize;
        while j <= number {
            let s = w.advance(&u, number - j + 1, &mut |x, lp, acc| {
                best.update(x, lp, acc);
                j += 1;
                let wt = 1.0 / j as f64;
                old_mean.copy_from_slice(&mean);
                for k in 0..n {
                    mean[k] = old_mean[k] + wt * (x[k] - old_mean[k]);
                }
                // CovJump + oldMean*oldMean' - Mean*Mean' + (1/j)*(x*x' - CovJump - oldMean*oldMean')
                for c in 0..n {
                    for r in 0..n {
                        let om = old_mean[r] * old_mean[c];
                        let cj = cov_jump[(r, c)];
                        cov_jump[(r, c)] = cj + om - mean[r] * mean[c] + wt * (x[r] * x[c] - cj - om);
                    }
                }
            });
            debug_assert!(s >= 1);
        }
        let (ix2, _) = self.finish(w);
        let post_var = cov_jump.clone();
        let post_mean = mean;

        let scale = if last_call {
            // [3] re-tune the scale with the new covariance.
            let ilogpo2 = self.eval_logpost(&ix2);
            let Some(dd) = chol_upper(&cov_jump) else {
                bail!(
                    "gmhmaxlik_core: the estimated posterior covariance matrix is not positive definite (chol failed)"
                );
            };
            let (_, _, s) = self.tune(&ix2, ilogpo2, &dd, iscale, 1000, false, &mut best);
            iscale = s;
            // [4] climb the hill (Scale is the value before the cooling).
            if self.opts.nclimb > 0 {
                self.climb(&dd, iscale, &cov_jump, &mut best);
            }
            s
        } else {
            iscale
        };
        Ok(CoreOut { mode: best.x, variance: post_var, scale, mean: post_mean })
    }

    /// Step [4] of `gmhmaxlik_core.m`: proposals `ModePar + iScale*dd*randn`, kept
    /// when they improve the posterior; every 200 draws, more than 4 consecutive
    /// windows with at most one improvement divide the scale by 1.1, and the climb
    /// stops once `sqrt(max(diag(iScale*CovJump))) < 1e-4`.
    ///
    /// Pre-fetching: the next proposals (all from the current best point, i.e.
    /// assuming no improvement, the likely outcome) are drawn ahead from a copy of
    /// the stream and evaluated in parallel; an improvement invalidates the rest of
    /// the batch, so the climb is exactly the sequential one.
    fn climb(&mut self, dd: &Mat, mut iscale: f64, cov_jump: &Mat, best: &mut Best) {
        let n = best.x.len();
        let nclimb = self.opts.nclimb;
        let maxdiag = (0..n).map(|i| cov_jump[(i, i)]).fold(f64::NEG_INFINITY, f64::max);
        let mut rng = self.rng.take().unwrap_or_else(|| rand_chacha::ChaCha8Rng::seed_from_u64(0));
        let inside = |p: &[f64]| p.iter().zip(&self.bounds.lb).zip(&self.bounds.ub).all(|((v, l), u)| v > l && v < u);
        let (mut j, mut jsux, mut test) = (1usize, 0usize, 0usize);
        let mut u = dd * iscale;
        let mut z = vec![0.0; n];
        'outer: while j <= nclimb {
            let to_boundary = 200 - (j - 1) % 200;
            let max_steps = to_boundary.min(nclimb - j + 1);
            // Draw ahead until `width` proposals need an evaluation.
            let mut props: Vec<(Vec<f64>, bool, rand_chacha::ChaCha8Rng)> = Vec::new();
            let mut ahead = rng.clone();
            let mut evals = 0;
            while props.len() < max_steps && evals < self.width {
                let mut p = vec![0.0; n];
                rand_mvn(&mut ahead, &best.x, &u, &mut z, &mut p);
                let ins = inside(&p);
                evals += ins as usize;
                props.push((p, ins, ahead.clone()));
            }
            let obj = self.obj;
            let lps: Vec<f64> =
                props.par_iter().map(|(p, ins, _)| if *ins { logpost_of(obj, p) } else { f64::NEG_INFINITY }).collect();
            for ((p, ins, r), lp) in props.into_iter().zip(lps) {
                rng = r;
                self.fcount += ins as usize;
                let improved = lp > best.lp;
                if improved {
                    best.x = p;
                    best.lp = lp;
                    jsux += 1;
                }
                if j.is_multiple_of(200) {
                    if jsux <= 1 {
                        test += 1;
                    } else {
                        test = 0;
                    }
                    jsux = 0;
                    if test > 4 {
                        iscale /= 1.10;
                        u = dd * iscale;
                    }
                    if (iscale * maxdiag).sqrt() < 1e-4 {
                        break 'outer;
                    }
                }
                j += 1;
                if improved {
                    continue 'outer; // the rest of the batch started from the old best point
                }
            }
        }
        self.rng = Some(rng);
    }
}
