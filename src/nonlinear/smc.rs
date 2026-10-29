// SPDX-License-Identifier: GPL-3.0-or-later
//! the SMC samplers of `estimation/smc/` other than DSMH, generic over the
//! posterior kernel ([`Objective`]: `dsge_likelihood` or the nonlinear filters):
//!
//! - [`hssmc`]: port of `hssmc.m` (Herbst & Schorfheide 2014 sequential Monte Carlo with a
//!   tempered likelihood, adaptive random-walk mutation), with `smc_samplers_initialization.m`;
//! - [`dime`]: port of `dime.m` (Boehl's differential-independence mixture ensemble MCMC).
//!
//! The particles (chains of the ensemble) are moved in parallel, each with its own ChaCha8
//! stream (seed, step, particle): results do not depend on the number of threads; they are
//! not MATLAB's draws (posteriors agree statistically).

use super::linalg::chol_lower;
use super::resampling::kitagawa;
use crate::estimation::samplers::chol_se;
use crate::estimation::{Bounds, Objective, Prior, log_prior_density, prior_draw};
use crate::linalg::{Mat, chol_upper};
use anyhow::{Result, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

fn stream(seed: u64, tag: u64, a: u64, b: u64) -> ChaCha8Rng {
    let mut r = ChaCha8Rng::seed_from_u64(seed ^ tag);
    r.set_stream(a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    r
}

fn admissible(x: &[f64], b: &Bounds) -> bool {
    x.iter().zip(&b.lb).all(|(v, l)| v >= l) && x.iter().zip(&b.ub).all(|(v, u)| v <= u)
}

/// `tempered_likelihood.m`: (tempered log posterior, log likelihood).
fn tempered(obj: &dyn Objective, priors: &[Prior], x: &[f64], lambda: f64) -> (f64, f64) {
    let logpost = -obj.eval(x).0;
    let logprior = log_prior_density(x, priors);
    let loglik = logpost - logprior;
    (lambda * loglik + logprior, loglik)
}

/// `smc_samplers_initialization.m`: `n` prior draws (inside `bounds`) with a finite
/// likelihood, in parallel; returns (particles npar x n, tempered log posterior at 0, log lik).
pub fn initialization(
    obj: &dyn Objective,
    priors: &[Prior],
    bounds: &Bounds,
    n: usize,
    seed: u64,
    tag: u64,
    sampler: &str,
    verbose: bool,
) -> (Mat, Vec<f64>, Vec<f64>) {
    let npar = priors.len();
    if verbose {
        println!("Estimation:{sampler}: Initialization...");
        println!("Estimation:{sampler}: Searching for initial values...");
    }
    let t0 = std::time::Instant::now();
    let init: Vec<(Vec<f64>, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|j| {
            let mut rng = stream(seed, tag, 0, j as u64 + 1);
            loop {
                let cand = prior_draw(priors, bounds, &mut rng);
                if admissible(&cand, bounds) {
                    let (tl, ll) = tempered(obj, priors, &cand, 0.0);
                    if ll.is_finite() {
                        return (cand, tl, ll);
                    }
                }
            }
        })
        .collect();
    if verbose {
        println!("Estimation:{sampler}: Initial values found ({:.2}s)", t0.elapsed().as_secs_f64());
        println!();
    }
    (Mat::from_fn(npar, n, |i, j| init[j].0[i]), init.iter().map(|v| v.1).collect(), init.iter().map(|v| v.2).collect())
}

// ---------------------------------------------------------------------------
// hssmc
// ---------------------------------------------------------------------------

/// `options_.posterior_sampler_options.hssmc` (`default_option_values.m`).
#[derive(Debug, Clone)]
pub struct HssmcOptions {
    pub steps: usize,
    pub lambda: f64,
    pub particles: usize,
    pub scale: f64,
    pub acpt: f64,
    pub target: f64,
    pub seed: u64,
    pub verbose: bool,
}

impl Default for HssmcOptions {
    fn default() -> Self {
        HssmcOptions {
            steps: 25,
            lambda: 2.0,
            particles: 20000,
            scale: 0.5,
            acpt: 1.0,
            target: 0.25,
            seed: 0,
            verbose: true,
        }
    }
}

/// Result of [`hssmc`].
#[derive(Debug, Clone)]
pub struct HssmcResult {
    /// npar x particles: the final (resampled) particles.
    pub particles: Mat,
    pub tlogpostkernel: Vec<f64>,
    pub loglikelihood: Vec<f64>,
    /// Weights of the last step when it did not resample (else empty).
    pub weights: Vec<f64>,
    /// Log marginal data density (`oo_.MarginalDensity.hssmc`).
    pub log_marginal_density: f64,
    pub phi: Vec<f64>,
    pub ess: Vec<f64>,
    pub acceptance: Vec<f64>,
    pub scale: Vec<f64>,
}

const HSSMC_TAG: u64 = 0x6873_736d_6300;

/// Port of `hssmc.m`. `bounds` is the admissible region (`Prior.admissible`).
pub fn hssmc(obj: &dyn Objective, priors: &[Prior], bounds: &Bounds, o: &HssmcOptions) -> Result<HssmcResult> {
    let npar = priors.len();
    let np = o.particles;
    if o.steps < 2 {
        bail!("hssmc: steps must be at least 2");
    }
    let mlogit = |x: f64| 0.95 + 0.1 / (1.0 + (-16.0 * x).exp());
    let phi: Vec<f64> = (0..o.steps).map(|i| (i as f64 / (o.steps - 1) as f64).powf(o.lambda)).collect();
    let mut mdd = 0.0;
    let mut scl = vec![0.0; o.steps];
    let mut ess = vec![0.0; o.steps];
    let mut acpt = vec![0.0; o.steps];
    let t0 = std::time::Instant::now();
    let (mut particles, mut tlogpost, mut loglik) =
        initialization(obj, priors, bounds, np, o.seed, HSSMC_TAG, "hssmc", o.verbose);
    if o.verbose {
        println!("#Iter.       lambda        ESS          Acceptance rate   scale   resample   seconds");
        println!(
            "{:3}          {:5.4}     {}         {:5.4}        {:4.3}     {:>3}       {:5.2}",
            1,
            0.0,
            crate::analysis::table::fmt_e(0.0, 9, 5).to_uppercase(),
            0.0,
            0.0,
            "no",
            t0.elapsed().as_secs_f64()
        );
    }
    let mut weights = vec![1.0 / np as f64; np];
    let mut resampled = false;
    let (mut scale, mut acc_rate) = (o.scale, o.acpt);
    let mut last_weights = vec![];
    for i in 1..o.steps {
        let t0 = std::time::Instant::now();
        let dphi = phi[i] - phi[i - 1];
        for (w, l) in weights.iter_mut().zip(&loglik) {
            *w *= (dphi * l).exp();
        }
        let sw: f64 = weights.iter().sum();
        weights.iter_mut().for_each(|w| *w /= sw);
        mdd += sw.ln();
        ess[i] = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
        let mut rs = stream(o.seed, HSSMC_TAG, i as u64, u64::MAX);
        if 2.0 * ess[i] < np as f64 {
            resampled = true;
            let idx = kitagawa(&weights, rs.r#gen(), np);
            particles = super::resampling::gather_columns(&particles, &idx);
            loglik = idx.iter().map(|&j| loglik[j]).collect();
            tlogpost = idx.iter().map(|&j| tlogpost[j]).collect();
            weights = vec![1.0 / np as f64; np];
        }
        scale *= mlogit(acc_rate - o.target);
        scl[i] = scale;
        // weighted mean and covariance
        let mu: Vec<f64> = (0..npar).map(|r| (0..np).map(|j| particles[(r, j)] * weights[j]).sum()).collect();
        let mut rmat = Mat::zeros(npar, npar);
        for j in 0..np {
            for a in 0..npar {
                let za = particles[(a, j)] - mu[a];
                for b in 0..=a {
                    rmat[(a, b)] += za * (particles[(b, j)] - mu[b]) * weights[j];
                }
            }
        }
        for a in 0..npar {
            for b in 0..a {
                rmat[(b, a)] = rmat[(a, b)];
            }
        }
        let Some(rr) = chol_lower(&rmat) else {
            bail!("hssmc: the weighted covariance of the particles is not positive definite at step {}", i + 1);
        };
        for (t, l) in tlogpost.iter_mut().zip(&loglik) {
            *t += dphi * l;
        }
        // randomwalk (parfor)
        let moved: Vec<(bool, Vec<f64>, f64, f64)> = (0..np)
            .into_par_iter()
            .map(|j| {
                let mut rng = stream(o.seed, HSSMC_TAG, i as u64, j as u64);
                let x: Vec<f64> = (0..npar).map(|r| particles[(r, j)]).collect();
                let z: Vec<f64> = (0..npar).map(|_| rng.sample(StandardNormal)).collect();
                let cand: Vec<f64> =
                    (0..npar).map(|r| x[r] + scale * (0..=r).map(|c| rr[(r, c)] * z[c]).sum::<f64>()).collect();
                if admissible(&cand, bounds) {
                    let (tl, ll) = tempered(obj, priors, &cand, phi[i]);
                    if ll.is_finite() && rng.r#gen::<f64>() < (tl - tlogpost[j]).exp() {
                        return (true, cand, ll, tl);
                    }
                }
                (false, x, loglik[j], tlogpost[j])
            })
            .collect();
        let mut nacc = 0usize;
        for (j, (a, x, ll, tl)) in moved.into_iter().enumerate() {
            nacc += a as usize;
            for r in 0..npar {
                particles[(r, j)] = x[r];
            }
            loglik[j] = ll;
            tlogpost[j] = tl;
        }
        acc_rate = nacc as f64 / np as f64;
        acpt[i] = acc_rate;
        if o.verbose {
            println!(
                "{:3}          {:5.4}     {}         {:5.4}        {:4.3}     {:>3}       {:5.2}",
                i + 1,
                phi[i],
                crate::analysis::table::fmt_e(ess[i], 9, 5).to_uppercase(),
                acpt[i],
                scl[i],
                if resampled { "yes" } else { "no" },
                t0.elapsed().as_secs_f64()
            );
        }
        if i == o.steps - 1 && !resampled {
            last_weights = weights.clone();
            let idx = kitagawa(&weights, rs.r#gen(), np);
            particles = super::resampling::gather_columns(&particles, &idx);
        }
        resampled = false;
    }
    Ok(HssmcResult {
        particles,
        tlogpostkernel: tlogpost,
        loglikelihood: loglik,
        weights: last_weights,
        log_marginal_density: mdd,
        phi,
        ess,
        acceptance: acpt,
        scale: scl,
    })
}

// ---------------------------------------------------------------------------
// DIME
// ---------------------------------------------------------------------------

/// `options_.posterior_sampler_options.dime` (`default_option_values.m`,
/// `dynare_estimation_init.m`: `nchain = 5*npar`; `tune = ceil(50000/nchain)`).
#[derive(Debug, Clone)]
pub struct DimeOptions {
    pub nchain: usize,
    pub niter: usize,
    pub tune: usize,
    pub aimh_prob: f64,
    pub sigma: f64,
    /// `gamma` (default `2.38/sqrt(2*ndim)`).
    pub gamma: Option<f64>,
    pub df_proposal_dist: f64,
    pub rho: f64,
    pub seed: u64,
    pub verbose: bool,
}

impl DimeOptions {
    pub fn new(npar: usize) -> Self {
        let nchain = 5 * npar;
        DimeOptions {
            nchain,
            niter: 1500,
            tune: 50000usize.div_ceil(nchain.max(1)),
            aimh_prob: 0.1,
            sigma: 1e-5,
            gamma: None,
            df_proposal_dist: 10.0,
            rho: 0.999,
            seed: 0,
            verbose: true,
        }
    }
}

/// Result of [`dime`]: `chains[iter][chain]` (constrained parameters) and their log posterior.
#[derive(Debug, Clone)]
pub struct DimeResult {
    pub chains: Vec<Vec<Vec<f64>>>,
    pub lprobs: Vec<Vec<f64>>,
    pub tune: usize,
}

impl DimeResult {
    /// `GetAllPosteriorDraws` for DIME: iterations `end-tune:end` (tune+1 of them) of every
    /// chain, iteration-major within each chain as MATLAB's `reshape` orders them.
    pub fn posterior_draws(&self) -> Vec<Vec<f64>> {
        let niter = self.chains.len();
        let first = niter.saturating_sub(self.tune + 1);
        let nchain = self.chains.first().map_or(0, Vec::len);
        let mut out = Vec::with_capacity((niter - first) * nchain);
        for c in 0..nchain {
            for it in first..niter {
                out.push(self.chains[it][c].clone());
            }
        }
        out
    }

    /// The draw with the highest log posterior (`compute_posterior_covariance_matrix.m`).
    pub fn mode(&self) -> (Vec<f64>, f64) {
        let mut best = (vec![], f64::NEG_INFINITY);
        for (it, row) in self.lprobs.iter().enumerate() {
            for (c, &l) in row.iter().enumerate() {
                if l > best.1 {
                    best = (self.chains[it][c].clone(), l);
                }
            }
        }
        best
    }
}

/// `check_bounds` of `dime.m` (as written there: only coordinates with a finite upper bound
/// are checked, and `x -/+ 1e-16` shifts every coordinate).
fn dime_bounds(priors: &[Prior], b: &Bounds) -> Bounds {
    let mut nb = b.clone();
    let mean: Vec<f64> = priors.iter().map(|p| p.mean).collect();
    let at_mean = log_prior_density(&mean, priors);
    let thr = 1e-16f64.ln();
    for j in 0..mean.len() {
        if b.ub[j].is_finite() {
            let mut x = mean.clone();
            x[j] = b.ub[j];
            let xm: Vec<f64> = x.iter().map(|v| v - 1e-16).collect();
            if log_prior_density(&xm, priors) - at_mean > thr {
                nb.ub[j] += 1e-2;
            }
            let mut x = mean.clone();
            x[j] = b.lb[j];
            let xp: Vec<f64> = x.iter().map(|v| v + 1e-16).collect();
            if log_prior_density(&xp, priors) - at_mean > thr {
                nb.lb[j] -= 1e-2;
            }
        }
    }
    nb
}

/// `ptransform`: constrained -> unconstrained (`con2unc`) or back.
fn ptransform(x: &mut [f64], b: &Bounds, con2unc: bool) {
    for j in 0..x.len() {
        let (lb, ub) = (b.lb[j], b.ub[j]);
        if lb.is_finite() && !ub.is_finite() {
            x[j] = if con2unc { (x[j] - lb).ln() } else { lb + x[j].exp() };
        } else if lb.is_finite() && ub.is_finite() {
            x[j] = if con2unc {
                let y = (x[j] - lb) / (ub - lb);
                (y / (1.0 - y)).ln()
            } else {
                lb + (ub - lb) / (1.0 + (-x[j]).exp())
            };
        }
    }
}

/// `logsumexp.m`.
pub fn logsumexp(x: &[f64]) -> f64 {
    let (k, &xmax) = x.iter().enumerate().fold((0, &f64::NEG_INFINITY), |b, (i, v)| if *v > *b.1 { (i, v) } else { b });
    if !xmax.is_finite() {
        return xmax;
    }
    let s: f64 = x.iter().enumerate().filter(|(i, _)| *i != k).map(|(_, v)| (v - xmax).exp()).sum();
    xmax + s.ln_1p()
}

const DIME_TAG: u64 = 0x6469_6d65_0000;

/// Port of `dime.m`. `bounds`: the estimation bounds (`mh_bounds`). As in Dynare, the chains'
/// initial log posterior is the initialization's log likelihood (`[x, ~, lprob]`), the
/// ensemble moves in the unconstrained space without a Jacobian term, and the posterior
/// draws are the last `tune + 1` iterations.
pub fn dime(obj: &dyn Objective, priors: &[Prior], bounds: &Bounds, o: &DimeOptions) -> Result<DimeResult> {
    let ndim = priors.len();
    let nchain = o.nchain;
    if o.tune > o.niter {
        bail!(
            "dime: number of ensemble iterations to keep (tune={}) exceeds number of iterations (niter={})!",
            o.tune,
            o.niter
        );
    }
    if nchain < 6 {
        bail!("dime: nchain must be at least 6 (got {nchain}).");
    }
    if nchain < 2 * ndim {
        println!(
            "WARNING: dime: nchain={nchain} is less than 2*ndim={}. The ensemble covariance will be rank-deficient, which may slow convergence of the AIMH proposals. Recommended: nchain >= 5*ndim = {}.",
            2 * ndim,
            5 * ndim
        );
    }
    let g0 = o.gamma.unwrap_or(2.38 / (2.0 * ndim as f64).sqrt());
    let dft = o.df_proposal_dist;
    let b = dime_bounds(priors, bounds);
    let t0 = std::time::Instant::now();
    let (p0, _tl, ll) = initialization(obj, priors, bounds, nchain, o.seed, DIME_TAG, "dime", o.verbose);
    let mut x: Vec<Vec<f64>> = (0..nchain)
        .map(|j| {
            let mut v: Vec<f64> = (0..ndim).map(|i| p0[(i, j)]).collect();
            ptransform(&mut v, &b, true);
            v
        })
        .collect();
    let mut lprob = ll;
    if o.verbose {
        println!(
            "Estimation:dime: log-posterior standard deviation (post.std) of a multivariate normal would be {:.2}.\n",
            (0.5 * ndim as f64).sqrt()
        );
    }
    let mut prop_cov = Mat::identity(ndim, ndim);
    let mut prop_mean = vec![0.0; ndim];
    let mut naccepted = 1usize;
    let mut cumlweight = f64::NEG_INFINITY;
    let isplit = nchain / 2 + 1;
    let mut chains = Vec::with_capacity(o.niter);
    let mut lprobs = Vec::with_capacity(o.niter);
    let mut rng = stream(o.seed, DIME_TAG, u64::MAX, 0);
    for iter in 0..o.niter {
        for complementary in [false, true] {
            let (idcur, idref): (Vec<usize>, Vec<usize>) = if complementary {
                ((0..isplit).collect(), (isplit..nchain).collect())
            } else {
                ((isplit..nchain).collect(), (0..isplit).collect())
            };
            let cursize = idcur.len();
            let refsize = nchain - cursize;
            // i1 = (0:cursize-1) + randi([1 cursize-1]); i2 = ... + randi([1 cursize-2]), skipping i1
            let i1: Vec<usize> = (0..cursize).map(|k| k + rng.gen_range(1..=cursize - 1)).collect();
            let i2: Vec<usize> = (0..cursize)
                .map(|k| {
                    let v = k + rng.gen_range(1..=cursize - 2);
                    if v >= i1[k] { v + 1 } else { v }
                })
                .collect();
            let f: Vec<f64> = (0..cursize).map(|_| o.sigma * rng.sample::<f64, _>(StandardNormal)).collect();
            let mut q: Vec<Vec<f64>> = (0..cursize)
                .map(|k| {
                    let (a, bb) = (&x[idref[i1[k] % refsize]], &x[idref[i2[k] % refsize]]);
                    (0..ndim).map(|d| x[idcur[k]][d] + g0 * (a[d] - bb[d]) + f[k]).collect()
                })
                .collect();
            let mut factors = vec![0.0; cursize];
            let xchnge: Vec<bool> = (0..cursize).map(|_| rng.r#gen::<f64>() <= o.aimh_prob).collect();
            let scaled = &prop_cov * ((dft - 2.0) / dft);
            let psd = &scaled + Mat::identity(ndim, ndim) * 1e-16;
            let u = match chol_upper(&psd) {
                Some(u) => u,
                None => {
                    println!(
                        "WARNING: dime: proposal covariance is not positive definite at iteration {}. Using modified Cholesky (chol_SE) to proceed.",
                        iter + 1
                    );
                    chol_se(&scaled)?.0
                }
            };
            let prop = crate::estimation::samplers::ProposalDist::student(u, dft);
            for k in 0..cursize {
                if xchnge[k] {
                    let mut cand = vec![0.0; ndim];
                    prop.draw(&mut rng, &prop_mean, &mut cand);
                    let lold = prop.logpdf(&x[idcur[k]], &prop_mean);
                    let lnew = prop.logpdf(&cand, &prop_mean);
                    q[k] = cand;
                    factors[k] = lold - lnew;
                }
            }
            let newlprob: Vec<f64> = q
                .par_iter()
                .map(|v| {
                    let mut c = v.clone();
                    ptransform(&mut c, &b, false);
                    -obj.eval(&c).0
                })
                .collect();
            for k in 0..cursize {
                let lnpdiff = factors[k] + newlprob[k] - lprob[idcur[k]];
                if lnpdiff > rng.r#gen::<f64>().ln() {
                    naccepted += 1;
                    x[idcur[k]] = q[k].clone();
                    lprob[idcur[k]] = newlprob[k];
                }
            }
        }
        chains.push(
            x.iter()
                .map(|v| {
                    let mut c = v.clone();
                    ptransform(&mut c, &b, false);
                    c
                })
                .collect::<Vec<_>>(),
        );
        lprobs.push(lprob.clone());
        let lweight = logsumexp(&lprob) + (naccepted as f64).ln() - (nchain as f64).ln();
        let nmean: Vec<f64> = (0..ndim).map(|d| x.iter().map(|v| v[d]).sum::<f64>() / nchain as f64).collect();
        let mut ncov = Mat::zeros(ndim, ndim);
        for v in &x {
            for a in 0..ndim {
                for bb in 0..ndim {
                    ncov[(a, bb)] += (v[a] - nmean[a]) * (v[bb] - nmean[bb]) / (nchain - 1) as f64;
                }
            }
        }
        let newcum = logsumexp(&[cumlweight, lweight]);
        let (wa, wb) = ((cumlweight - newcum).exp(), (lweight - newcum).exp());
        prop_cov = &prop_cov * wa + &ncov * wb;
        for d in 0..ndim {
            prop_mean[d] = wa * prop_mean[d] + wb * nmean[d];
        }
        cumlweight = newcum + o.rho.ln();
        if o.verbose {
            if iter == 0 || (iter + 1) % 15 == 0 {
                println!("     #iter.      post.mode     post.std    %accept     lapsed");
                println!("     ------      ---------     --------    -------     ------");
            }
            let mx = lprob.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let m = lprob.iter().sum::<f64>() / nchain as f64;
            let sd = (lprob.iter().map(|l| (l - m) * (l - m)).sum::<f64>() / (nchain - 1) as f64).sqrt();
            println!(
                "{:>11}     {:10.3}     {}     {:5.2}%     {:5.2}s",
                format!("{}/{}", iter + 1, o.niter),
                mx,
                crate::analysis::table::fmt_e(sd, 1, 2),
                100.0 * naccepted as f64 / nchain as f64,
                t0.elapsed().as_secs_f64()
            );
        }
        naccepted = 0;
    }
    Ok(DimeResult { chains, lprobs, tune: o.tune })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::PriorShape;

    /// The prior as the posterior (flat likelihood): both samplers must reproduce it.
    struct PriorObj(Vec<Prior>);
    impl Objective for PriorObj {
        fn dim(&self) -> usize {
            self.0.len()
        }
        fn eval(&self, x: &[f64]) -> (f64, i32) {
            (-log_prior_density(x, &self.0), 0)
        }
    }

    /// N(mu, s^2) likelihood on one parameter with a N(0, 1) prior: posterior N(m, v) and the
    /// marginal density are known.
    struct GaussLik {
        priors: Vec<Prior>,
        y: f64,
        s: f64,
    }
    impl Objective for GaussLik {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, x: &[f64]) -> (f64, i32) {
            let ll =
                -0.5 * (2.0 * std::f64::consts::PI * self.s * self.s).ln() - 0.5 * ((self.y - x[0]) / self.s).powi(2);
            (-(ll + log_prior_density(x, &self.priors)), 0)
        }
    }

    #[test]
    fn hssmc_marginal_density_of_a_gaussian() {
        let priors = vec![Prior::new(PriorShape::Normal, 0.0, 1.0, None, None).unwrap()];
        let obj = GaussLik { priors: priors.clone(), y: 1.5, s: 0.5 };
        let b = crate::estimation::prior_bounds(&priors, 0.0);
        for seed in [4u64, 9, 11] {
            let o = HssmcOptions { particles: 20000, steps: 50, seed, verbose: false, ..Default::default() };
            let r = hssmc(&obj, &priors, &b, &o).unwrap();
            // posterior N(1.5/(1+0.25), 0.25/1.25); evidence N(1.5; 0, 1 + 0.25)
            let (pm, pv) = (1.5 / 1.25, 0.25 / 1.25);
            let lml = -0.5 * (2.0 * std::f64::consts::PI * 1.25).ln() - 0.5 * 1.5 * 1.5 / 1.25;
            let xs: Vec<f64> = r.particles.row(0).iter().copied().collect();
            let m = xs.iter().sum::<f64>() / xs.len() as f64;
            let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64;
            assert!((m - pm).abs() < 0.03, "{m} vs {pm}");
            assert!((v - pv).abs() < 0.03, "{v} vs {pv}");
            assert!((r.log_marginal_density - lml).abs() < 0.02, "{} vs {lml}", r.log_marginal_density);
            println!("seed {seed}: log mdd {} vs {lml}", r.log_marginal_density);
        }
    }

    #[test]
    fn dime_recovers_the_prior() {
        let priors = vec![
            Prior::new(PriorShape::Gamma, 2.0, 0.5, None, None).unwrap(),
            Prior::new(PriorShape::Beta, 0.6, 0.15, None, None).unwrap(),
            Prior::new(PriorShape::Normal, -1.0, 0.3, None, None).unwrap(),
        ];
        let obj = PriorObj(priors.clone());
        let b = crate::estimation::prior_bounds(&priors, 0.0);
        let mut o = DimeOptions::new(3);
        o.nchain = 30;
        o.niter = 2000;
        o.tune = 1500;
        o.seed = 5;
        o.verbose = false;
        let r = dime(&obj, &priors, &b, &o).unwrap();
        let d = r.posterior_draws();
        // DIME moves in the unconstrained space without a Jacobian: the target there is the
        // prior density at the transformed point, so only the unbounded normal is the prior.
        let m: f64 = d.iter().map(|v| v[2]).sum::<f64>() / d.len() as f64;
        let s = (d.iter().map(|v| (v[2] - m).powi(2)).sum::<f64>() / d.len() as f64).sqrt();
        assert!((m + 1.0).abs() < 0.05 && (s - 0.3).abs() < 0.05, "{m} {s}");
        assert!(d.iter().all(|v| v[0] > 0.0 && v[1] > 0.0 && v[1] < 1.0));
    }
}
