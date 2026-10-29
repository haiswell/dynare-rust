// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `estimation/smc/dsmh.m` (dynamic striated Metropolis–Hastings, Waggoner, Wu &
//! Zha 2016) with `smc_samplers_initialization.m`, `tempered_likelihood.m` and `kitagawa.m`.
//!
//! Generic over the posterior kernel ([`Objective`]): works with `dsge_likelihood` and with
//! the nonlinear filters. The `G` striated chains of the tuning and mutation steps are
//! independent within a sweep, so they run in parallel, each with its own ChaCha8 stream
//! (seed, stage, step, chain): the result does not depend on the number of threads, but the
//! draws are not MATLAB's (the posterior agrees statistically).

use super::linalg::chol_lower;
use super::resampling::kitagawa;
use crate::estimation::{Bounds, Objective, Prior, log_prior_density, prior_draw};
use crate::linalg::Mat;
use anyhow::{Result, anyhow};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// `options_.posterior_sampler_options.dsmh` (defaults of `default_option_values.m`).
#[derive(Debug, Clone)]
pub struct DsmhOptions {
    /// Number of tempering stages.
    pub h: usize,
    /// Number of striation levels.
    pub m: usize,
    /// Particles per chain (`N`), chains (`G`), MH steps per chain when tuning `c` (`K`).
    pub n: usize,
    pub g: usize,
    pub k: usize,
    /// Target acceptance band for `c`.
    pub alpha0: f64,
    pub alpha1: f64,
    /// Thinning of the mutation step.
    pub tau: usize,
    pub seed: u64,
    pub verbose: bool,
}

impl Default for DsmhOptions {
    fn default() -> Self {
        DsmhOptions { h: 25, m: 20, n: 200, g: 10, k: 10, alpha0: 0.2, alpha1: 0.3, tau: 10, seed: 0, verbose: true }
    }
}

/// Final particles and diagnostics.
#[derive(Debug, Clone)]
pub struct DsmhResult {
    /// npar x (N*G) posterior particles.
    pub particles: Mat,
    /// Log posterior kernel of each particle.
    pub logpost: Vec<f64>,
    pub loglik: Vec<f64>,
    /// `zhat`: log marginal data density.
    pub log_marginal_density: f64,
    pub lambda: Vec<f64>,
    pub ess: Vec<f64>,
    /// Tuned `c` and acceptance rate per stage.
    pub c: Vec<f64>,
    pub acceptance: Vec<f64>,
}

/// `tempered_likelihood.m`: (tempered log posterior, log likelihood).
fn tempered(obj: &dyn Objective, priors: &[Prior], x: &[f64], lambda: f64) -> (f64, f64) {
    let logpost = -obj.eval(x).0;
    let logprior = log_prior_density(x, priors);
    let loglik = logpost - logprior;
    (lambda * loglik + logprior, loglik)
}

fn stream(seed: u64, a: u64, b: u64, c: u64) -> ChaCha8Rng {
    let mut r = ChaCha8Rng::seed_from_u64(seed ^ 0x6473_6d68);
    r.set_stream(a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ c);
    r
}

fn in_bounds(x: &[f64], b: &Bounds) -> bool {
    x.iter().zip(&b.lb).all(|(v, l)| v >= l) && x.iter().zip(&b.ub).all(|(v, u)| v <= u)
}

/// `candidate = x + sqrt(c) * Omegachol * randn(npar, 1)`.
fn propose(x: &[f64], c: f64, chol: &Mat, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let npar = x.len();
    let z: Vec<f64> = (0..npar).map(|_| rng.sample(StandardNormal)).collect();
    (0..npar).map(|i| x[i] + c.sqrt() * (0..npar).map(|j| chol[(i, j)] * z[j]).sum::<f64>()).collect()
}

/// Runs DSMH. `nobs`: number of observations `T` (sets the first temperature
/// `1/(10 npar T)`); `bounds`: `mh_bounds` (admissible region, also used to redraw the prior).
pub fn dsmh(
    obj: &dyn Objective,
    priors: &[Prior],
    bounds: &Bounds,
    nobs: usize,
    o: &DsmhOptions,
) -> Result<DsmhResult> {
    let npar = obj.dim();
    let np = o.n * o.g;
    let lambda1 = 1.0 / (10.0 * npar as f64 * nobs as f64);
    let hh = o.h as f64;
    let lambda: Vec<f64> = (1..=o.h).map(|i| ((hh - i as f64) / (hh - 1.0) * lambda1.ln()).exp()).collect();
    // ---- smc_samplers_initialization: prior draws with a finite likelihood
    if o.verbose {
        // smc_samplers_initialization.m
        println!("Estimation:dsmh: Initialization...");
        println!("Estimation:dsmh: Searching for initial values...");
    }
    let t0 = std::time::Instant::now();
    let init: Vec<(Vec<f64>, f64, f64)> = (0..np)
        .into_par_iter()
        .map(|j| {
            let mut rng = stream(o.seed, 0, 0, j as u64 + 1);
            loop {
                let cand = prior_draw(priors, bounds, &mut rng);
                if in_bounds(&cand, bounds) {
                    let (tl, ll) = tempered(obj, priors, &cand, 0.0);
                    if ll.is_finite() {
                        return (cand, tl, ll);
                    }
                }
            }
        })
        .collect();
    if o.verbose {
        println!("Estimation:dsmh: Initial values found ({:.2}s)", t0.elapsed().as_secs_f64());
        println!();
    }
    let mut param = Mat::from_fn(npar, np, |i, j| init[j].0[i]);
    let mut tlogpost: Vec<f64> = init.iter().map(|v| v.1).collect();
    let mut loglik: Vec<f64> = init.iter().map(|v| v.2).collect();
    let mut ess = vec![0.0; o.h];
    let mut cs = Vec::with_capacity(o.h);
    let mut accs = Vec::with_capacity(o.h);
    let mut zhat = 0.0;
    let mut c = 1.0;
    for i in 0..o.h {
        let t0 = std::time::Instant::now();
        // sort_matrices
        let mut ord: Vec<usize> = (0..np).collect();
        ord.sort_by(|&a, &b| tlogpost[a].partial_cmp(&tlogpost[b]).unwrap_or(std::cmp::Ordering::Equal));
        param = Mat::from_fn(npar, np, |r, j| param[(r, ord[j])]);
        tlogpost = ord.iter().map(|&j| tlogpost[j]).collect();
        loglik = ord.iter().map(|&j| loglik[j]).collect();
        // compute_IS_weights_and_moments
        let dl = if i == 0 { lambda[0] } else { lambda[i] - lambda[i - 1] };
        let tlogpost_i: Vec<f64> = tlogpost.iter().zip(&loglik).map(|(t, l)| t + l * dl).collect();
        let mut w: Vec<f64> = tlogpost_i.iter().zip(&tlogpost).map(|(a, b)| (a - b).exp()).collect();
        zhat += (w.iter().sum::<f64>() / np as f64).ln();
        let sw: f64 = w.iter().sum();
        w.iter_mut().for_each(|v| *v /= sw);
        ess[i] = 1.0 / w.iter().map(|v| v * v).sum::<f64>();
        let mu: Vec<f64> = (0..npar).map(|r| (0..np).map(|j| param[(r, j)] * w[j]).sum()).collect();
        let mut omega = Mat::zeros(npar, npar);
        for j in 0..np {
            for a in 0..npar {
                let za = param[(a, j)] - mu[a];
                for b in 0..npar {
                    omega[(a, b)] += za * w[j] * (param[(b, j)] - mu[b]);
                }
            }
        }
        let omega = (&omega + omega.transpose()) * 0.5;
        let chol = chol_lower(&omega)
            .ok_or_else(|| anyhow!("dsmh: the particles' covariance is not positive definite (stage {})", i + 1))?;
        // tune_c
        let lower = (0.5 * (o.alpha0 + o.alpha1)).powi(5);
        let upper = (0.5 * (o.alpha0 + o.alpha1)).powf(0.2);
        let mut attempt = 0u64;
        if o.verbose {
            // disp_verbose(newline) prints two line feeds
            println!("\n");
            println!("Estimation:dsmh: Tuning c_i, initial value = {c:6.3}.");
        }
        let acpt = loop {
            let mut rng = stream(o.seed, i as u64 + 1, 1 << 32 | attempt, 0);
            let idx = kitagawa(&w, rng.r#gen::<f64>(), o.g);
            let acc: usize = (0..o.g)
                .into_par_iter()
                .map(|j| {
                    let mut rng = stream(o.seed, i as u64 + 1, 1 << 32 | attempt, j as u64 + 1);
                    let mut x: Vec<f64> = (0..npar).map(|r| param[(r, idx[j])]).collect();
                    let mut tl0 = tlogpost_i[idx[j]];
                    let mut a = 0;
                    for _ in 0..o.k {
                        let cand = propose(&x, c, &chol, &mut rng);
                        if in_bounds(&cand, bounds) {
                            let (tl, ll) = tempered(obj, priors, &cand, lambda[i]);
                            if ll.is_finite() && rng.r#gen::<f64>() < (tl - tl0).exp() {
                                a += 1;
                                x = cand;
                                tl0 = tl;
                            }
                        }
                    }
                    a
                })
                .sum();
            // `acpt = acpt + 1/(opts.G*opts.K)` at every acceptance
            let step = 1.0 / (o.g * o.k) as f64;
            let acpt = (0..acc).fold(0.0, |s, _| s + step);
            attempt += 1;
            if o.verbose {
                println!("Acceptation rate = {acpt:5.3}.");
            }
            if o.alpha0 <= acpt && acpt <= o.alpha1 {
                if o.verbose {
                    println!("Tuning c_i done.");
                }
                break acpt;
            }
            c = if acpt < lower {
                c / 5.0
            } else if acpt <= upper {
                c * (0.5 * (o.alpha0 + o.alpha1)).ln() / acpt.ln()
            } else {
                5.0 * c
            };
            if o.verbose {
                println!("Trying with c_i  = {c:6.3}.");
            }
            // (no cap on the attempts: dsmh.m loops until the rate is in [alpha0, alpha1])
        };
        // mutation_DSMH
        if o.verbose {
            println!("\n");
            println!("Estimation:dsmh: Striated MH step...");
        }
        let levels_idx: Vec<usize> = (1..o.m).map(|k| k * o.n * o.g / o.m).collect();
        let mut levels = vec![f64::NEG_INFINITY];
        levels.extend(levels_idx.iter().map(|&k| tlogpost[k - 1]));
        let p_jump = 0.1 / o.tau as f64;
        let mut rng0 = stream(o.seed, i as u64 + 1, 2 << 32, 0);
        let idx = kitagawa(&w, rng0.r#gen::<f64>(), o.g);
        let per_level = o.n * o.g / o.m;
        let results: Vec<Vec<(Vec<f64>, f64, f64)>> = (0..o.g)
            .into_par_iter()
            .map(|j| {
                let mut rng = stream(o.seed, i as u64 + 1, 2 << 32, j as u64 + 1);
                let mut x: Vec<f64> = (0..npar).map(|r| param[(r, idx[j])]).collect();
                let mut tm1 = tlogpost[idx[j]];
                let mut ti = tlogpost_i[idx[j]];
                let mut ll0 = loglik[idx[j]];
                let mut kept = Vec::with_capacity(o.n);
                for l in 1..=o.n * o.tau {
                    let u1: f64 = rng.r#gen();
                    let u2: f64 = rng.r#gen();
                    if u1 < p_jump {
                        let mut k = o.m;
                        for mm in 1..o.m {
                            if tm1 < levels[mm] {
                                k = mm;
                                break;
                            }
                        }
                        let indx =
                            (((k - 1) * per_level + 1) as f64 + u2 * (per_level as f64 - 1.0)).floor() as usize - 1;
                        let alp = (loglik[indx] - ll0) * dl;
                        if u2 < alp.exp() {
                            x = (0..npar).map(|r| param[(r, indx)]).collect();
                            ti = tlogpost_i[indx];
                            ll0 = loglik[indx];
                            tm1 = tlogpost[indx];
                        }
                    } else {
                        let cand = propose(&x, c, &chol, &mut rng);
                        if in_bounds(&cand, bounds) {
                            let (tl, ll) = tempered(obj, priors, &cand, lambda[i]);
                            if ll.is_finite() && u2 < (tl - ti).exp() {
                                x = cand;
                                ti = tl;
                                ll0 = ll;
                                tm1 = tl - ll * dl;
                            }
                        }
                    }
                    if l % o.tau == 0 {
                        kept.push((x.clone(), ti, ll0));
                    }
                }
                kept
            })
            .collect();
        // out_param(:, (l/tau-1)*G + j) = chain j's state after l steps
        let mut out = Mat::zeros(npar, np);
        let mut out_t = vec![0.0; np];
        let mut out_l = vec![0.0; np];
        for (j, kept) in results.iter().enumerate() {
            for (s, (x, t, l)) in kept.iter().enumerate() {
                let col = s * o.g + j;
                for r in 0..npar {
                    out[(r, col)] = x[r];
                }
                out_t[col] = *t;
                out_l[col] = *l;
            }
        }
        param = out;
        tlogpost = out_t;
        loglik = out_l;
        cs.push(c);
        accs.push(acpt);
        if o.verbose {
            println!("Estimation:dsmh: Striated MH step done.");
            println!("Iter \t lambda_i \t ESS           \t c_i    \t Acc. rate \t scale  \t sec.  ");
            println!(
                "{:4} \t {:5.4} \t {} \t {:5.4} \t {:5.4} \t {:+5.4} \t {:5.2}",
                i + 1,
                lambda[i],
                crate::analysis::table::fmt_e(ess[i], 9, 5).to_uppercase(),
                c,
                acpt,
                zhat,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    Ok(DsmhResult {
        particles: param,
        logpost: tlogpost,
        loglik,
        log_marginal_density: zhat,
        lambda,
        ess,
        c: cs,
        acceptance: accs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::{FnObjective, PriorShape};

    /// Normal prior N(0, 1) and a normal likelihood N(1, 0.5^2) for one observation: the
    /// posterior is N(0.8, 0.2), the marginal density N(1; 0, 1.25).
    #[test]
    fn conjugate_normal() {
        let prior = Prior::new(PriorShape::Normal, 0.0, 1.0, None, None).unwrap();
        let priors = vec![prior.clone()];
        let obj = FnObjective {
            n: 1,
            f: |x: &[f64]| {
                let ll = -0.5 * (2.0 * std::f64::consts::PI * 0.25).ln() - 0.5 * (1.0 - x[0]).powi(2) / 0.25;
                -(ll + prior.logpdf(x[0]))
            },
        };
        let b = Bounds { lb: vec![-10.0], ub: vec![10.0] };
        let o = DsmhOptions { h: 10, n: 100, g: 10, tau: 4, m: 10, verbose: false, seed: 3, ..Default::default() };
        let r = dsmh(&obj, &priors, &b, 1, &o).unwrap();
        let n = r.particles.ncols() as f64;
        let mean = r.particles.row(0).sum() / n;
        let var = r.particles.row(0).iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let mdd = -0.5 * (2.0 * std::f64::consts::PI * 1.25).ln() - 0.5 / 1.25;
        assert!((mean - 0.8).abs() < 0.05, "mean {mean}");
        assert!((var - 0.2).abs() < 0.05, "var {var}");
        assert!((r.log_marginal_density - mdd).abs() < 0.1, "mdd {} vs {mdd}", r.log_marginal_density);
    }
}
