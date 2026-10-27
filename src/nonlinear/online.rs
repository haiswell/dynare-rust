// SPDX-License-Identifier: GPL-3.0-or-later
//! port of `estimation/online/online_auxiliary_filter.m`
//! (`posterior_sampling_method = 'online'`): Liu & West (2001) auxiliary particle filter over
//! the states and the parameters jointly, at order 1, 2 or 3 (dynare-rust's nonlinear
//! reduced form; `solve_model_for_online_filter.m`). The particles are solved and moved in parallel, each with its own ChaCha8 stream
//! (seed, period, particle), so the result does not depend on the number of threads.
//!
//! Ported as Dynare 7.2 has it: the first-stage weights use a Student-like kernel of the
//! squared Mahalanobis distance (`(1 + z^2/3)^-2`), `variance_update` stops once
//! `neff < resampling_threshold * T`, and with pruning the pruned states are only resampled
//! (the second stage does not store them).

use super::law_of_motion::iterate;
use super::likelihood::NonlinearModel;
use super::linalg::{chol_lower, det, neff};
use super::resampling::{gather_columns, kitagawa};
use super::sis::{mahalanobis_rows, pruned_init};
use crate::estimation::{Bounds, Prior, prior_draw};
use crate::linalg::Mat;
use anyhow::{Result, anyhow, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// `options_.posterior_sampler_options.online`.
#[derive(Debug, Clone)]
pub struct OnlineOptions {
    pub particles: usize,
    pub liu_west_delta: f64,
    pub liu_west_max_resampling_tries: usize,
    pub second_resampling: bool,
    pub seed: u64,
    pub verbose: bool,
}

impl Default for OnlineOptions {
    fn default() -> Self {
        OnlineOptions {
            particles: 5000,
            liu_west_delta: 0.99,
            liu_west_max_resampling_tries: 5000,
            second_resampling: false,
            seed: 0,
            verbose: true,
        }
    }
}

/// Parameter statistics per period (npar x T each) and the final particles.
#[derive(Debug, Clone)]
pub struct OnlineResult {
    pub mean: Mat,
    pub median: Mat,
    pub std: Mat,
    pub lb95: Mat,
    pub ub95: Mat,
    /// npar x particles: `parameters_particles_final.mat`'s `param` (resampled).
    pub param: Mat,
}

const TAG: u64 = 0x6f6e_6c69_6e65;

fn stream(seed: u64, t: u64, i: u64) -> ChaCha8Rng {
    let mut r = ChaCha8Rng::seed_from_u64(seed ^ TAG);
    r.set_stream(t.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i);
    r
}

fn randn(rng: &mut ChaCha8Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.sample(StandardNormal)).collect()
}

fn col(m: &Mat, j: usize) -> Mat {
    m.columns(j, 1).into_owned()
}

/// The initial state particles (ns x n): draws from `N(0, state_variance)` centred on
/// `solve_model_for_online_filter.m`'s `StateVectorMean = state_variables_steady_state` (the
/// deterministic steady state of the states, not `constant(mf0)`, for every initialization).
pub fn initial_particles(rf0: &super::ReducedForm, n: usize, seed: u64) -> Result<Mat> {
    let s0 =
        chol_lower(&rf0.state_variance).ok_or_else(|| anyhow!("online: chol of the initial state variance failed"))?;
    let mut rng0 = stream(seed, u64::MAX, 0);
    let z = Mat::from_fn(s0.ncols(), n, |_, _| rng0.sample(StandardNormal));
    let mut states = &s0 * z;
    for c in 0..n {
        for i in 0..rf0.ns() {
            states[(i, c)] += rf0.state_ss[i];
        }
    }
    Ok(states)
}

/// Runs the filter from `xparam1` (initial state distribution) on `model.y`.
pub fn online_auxiliary_filter(
    model: &NonlinearModel,
    priors: &[Prior],
    bounds: &Bounds,
    xparam1: &[f64],
    o: &OnlineOptions,
) -> Result<OnlineResult> {
    let npar = priors.len();
    let n = o.particles;
    // the pruned recursions run at order 2 only: at order >= 3 Dynare takes the k-order
    // solver path (`ReducedForm.use_k_order_solver`, local_state_space_iteration_k), which
    // ignores pruning (the pruned states are still drawn and resampled, then never read)
    let pruning = model.particle.pruning && model.order == 2;
    let rf0 = model
        .reduced_form(xparam1)
        .map_err(|e| anyhow!("online: the model has no solution at the initial values (info {})", e.info))?;
    let y = &model.y;
    let t_len = y.ncols();
    let pp = rf0.pp();
    let nu = rf0.nu();
    let ns = rf0.ns();
    let delta = o.liu_west_delta;
    let small_a = (3.0 * delta - 1.0) / (2.0 * delta);
    let b_square = 1.0 - small_a * small_a;
    // initial states
    let mut states = initial_particles(&rf0, n, o.seed)?;
    let (mut states_, _mf0_) = pruned_init(&rf0, &states);
    // parameter particles from the prior (those the model solves)
    let draws: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut r = stream(o.seed, u64::MAX - 1, i as u64);
            loop {
                let c = prior_draw(priors, bounds, &mut r);
                if model.reduced_form(&c).is_ok() {
                    return c;
                }
            }
        })
        .collect();
    let mut xparam = Mat::from_fn(npar, n, |r, c| draws[c][r]);
    let mut weights = vec![1.0 / n as f64; n];
    let const_lik = (2.0 * std::f64::consts::PI).ln() * pp as f64;
    let (mut mean, mut median, mut std, mut lb95, mut ub95) = (
        Mat::zeros(npar, t_len),
        Mat::zeros(npar, t_len),
        Mat::zeros(npar, t_len),
        Mat::zeros(npar, t_len),
        Mat::zeros(npar, t_len),
    );
    let mut variance_update = true;
    let mut chol_sigma_bar = Mat::zeros(npar, npar);
    let ddl = 3.0_f64;
    let tcoef =
        (crate::estimation::special::gammaln((ddl + 1.0) / 2.0) - crate::estimation::special::gammaln(ddl / 2.0)).exp()
            / (ddl * std::f64::consts::PI).sqrt();
    let mut param_final = Mat::zeros(npar, n);
    let t0 = std::time::Instant::now();
    for t in 0..t_len {
        let yt: Vec<f64> = (0..pp).map(|i| y[(i, t)]).collect();
        let m_bar: Vec<f64> = (0..npar).map(|r| (0..n).map(|c| xparam[(r, c)] * weights[c]).sum()).collect();
        let mut sigma_bar = Mat::zeros(npar, npar);
        for c in 0..n {
            for a in 0..npar {
                let da = xparam[(a, c)] - m_bar[a];
                for b in 0..npar {
                    sigma_bar[(a, b)] += weights[c] * da * (xparam[(b, c)] - m_bar[b]);
                }
            }
        }
        if variance_update {
            let s = &sigma_bar * b_square;
            chol_sigma_bar = match chol_lower(&s) {
                Some(l) => l,
                None => crate::estimation::samplers::chol_se(&s)?.0.transpose(),
            };
        }
        let fore = Mat::from_fn(npar, n, |r, c| (1.0 - small_a) * m_bar[r] + small_a * xparam[(r, c)]);
        // first stage: predicted observables at the forecast parameters, zero shocks
        let tau: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let x: Vec<f64> = fore.column(i).iter().copied().collect();
                match model.reduced_form(&x) {
                    Ok(rf) => {
                        let eps = Mat::zeros(nu, 1);
                        let tmp = iterate(&rf, &col(&states, i), &eps, pruning.then(|| col(&states_, i)).as_ref()).y;
                        let z = mahalanobis_rows(&yt, &tmp, &rf.mf1, &rf.h)[0];
                        weights[i] * (tcoef / (1.0 + z * z / ddl).powf((ddl + 1.0) / 2.0) + 1e-99)
                    }
                    Err(_) => 0.0,
                }
            })
            .collect();
        let st: f64 = tau.iter().sum();
        let tau: Vec<f64> = tau.iter().map(|v| v / st).collect();
        let mut rs = stream(o.seed, t as u64, u64::MAX);
        let indx = kitagawa(&tau, rs.r#gen(), n);
        states = gather_columns(&states, &indx);
        if pruning {
            states_ = gather_columns(&states_, &indx);
        }
        xparam = gather_columns(&fore, &indx);
        let w_stage1: Vec<f64> = indx.iter().map(|&j| weights[j] / tau[j]).collect();
        // second stage: parameters jittered (Liu-West), states propagated with shocks
        let moved: Vec<Result<(Vec<f64>, Vec<f64>, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut r = stream(o.seed, t as u64, i as u64);
                let xi: Vec<f64> = xparam.column(i).iter().copied().collect();
                for _ in 0..o.liu_west_max_resampling_tries {
                    let z = randn(&mut r, npar);
                    let cand: Vec<f64> = (0..npar).map(|a| xi[a] + (0..=a).map(|b| chol_sigma_bar[(a, b)] * z[b]).sum::<f64>()).collect();
                    if !cand.iter().zip(&bounds.lb).all(|(v, l)| v >= l) || !cand.iter().zip(&bounds.ub).all(|(v, u)| v <= u) {
                        continue;
                    }
                    let Ok(rf) = model.reduced_form(&cand) else { continue };
                    let qc = chol_lower(&rf.q).ok_or_else(|| anyhow!("online: chol(Q) failed"))?;
                    let e = randn(&mut r, nu);
                    let eps = Mat::from_fn(nu, 1, |a, _| (0..=a).map(|b| qc[(a, b)] * e[b]).sum());
                    let tmp = iterate(&rf, &col(&states, i), &eps, pruning.then(|| col(&states_, i)).as_ref()).y;
                    let newstate: Vec<f64> = rf.mf0.iter().map(|&k| tmp[(k, 0)]).collect();
                    let q = mahalanobis_rows(&yt, &tmp, &rf.mf1, &rf.h)[0];
                    let w = w_stage1[i] * (-0.5 * (const_lik + det(&rf.h).ln() + q)).exp();
                    return Ok((cand, newstate, w));
                }
                bail!(
                    "Liu & West online filter: I haven't been able to solve the model in {} tries (increase liu_west_max_resampling_tries).",
                    o.liu_west_max_resampling_tries
                )
            })
            .collect();
        let mut wtilde = vec![0.0; n];
        for (i, m) in moved.into_iter().enumerate() {
            let (cand, s, w) = m?;
            for a in 0..npar {
                xparam[(a, i)] = cand[a];
            }
            for a in 0..ns {
                states[(a, i)] = s[a];
            }
            wtilde[i] = w;
        }
        let sw: f64 = wtilde.iter().sum();
        weights = wtilde.iter().map(|w| w / sw).collect();
        if variance_update && neff(&weights) < model.particle.resampling_threshold * t_len as f64 {
            variance_update = false;
        }
        if o.second_resampling {
            let idx = kitagawa(&weights, rs.r#gen(), n);
            states = gather_columns(&states, &idx);
            if pruning {
                states_ = gather_columns(&states_, &idx);
            }
            xparam = gather_columns(&xparam, &idx);
            weights = vec![1.0 / n as f64; n];
            for a in 0..npar {
                let mut v: Vec<f64> = xparam.row(a).iter().copied().collect();
                let m = v.iter().sum::<f64>() / n as f64;
                mean[(a, t)] = m;
                std[(a, t)] = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64).sqrt();
                v.sort_by(f64::total_cmp);
                let q = |p: f64| v[((p * n as f64) as usize).saturating_sub(1).min(n - 1)];
                lb95[(a, t)] = q(0.025);
                median[(a, t)] = q(0.5);
                ub95[(a, t)] = q(0.975);
            }
            if t + 1 == t_len {
                param_final = xparam.clone();
            }
        } else {
            for a in 0..npar {
                let m: f64 = (0..n).map(|c| xparam[(a, c)] * weights[c]).sum();
                mean[(a, t)] = m;
                std[(a, t)] = (0..n).map(|c| weights[c] * (xparam[(a, c)] - m).powi(2)).sum::<f64>().sqrt();
                let mut v: Vec<(f64, f64)> = (0..n).map(|c| (xparam[(a, c)], weights[c])).collect();
                v.sort_by(|x, y| x.0.total_cmp(&y.0));
                let (mut cw, mut p1, mut p2, mut p3) = (0.0, false, false, false);
                for (x, w) in v {
                    cw += w;
                    if !p1 && cw >= 0.025 {
                        lb95[(a, t)] = x;
                        p1 = true;
                    }
                    if !p2 && cw >= 0.5 {
                        median[(a, t)] = x;
                        p2 = true;
                    }
                    if !p3 && cw >= 0.975 {
                        ub95[(a, t)] = x;
                        p3 = true;
                    }
                }
            }
            if t + 1 == t_len {
                let idx = kitagawa(&weights, rs.r#gen(), n);
                param_final = gather_columns(&xparam, &idx);
            }
        }
        if o.verbose && ((t + 1) % 10 == 0 || t + 1 == t_len) {
            println!("online: {} of {t_len} observations ({:.1}s)", t + 1, t0.elapsed().as_secs_f64());
        }
    }
    Ok(OnlineResult { mean, median, std, lb95, ub95, param: param_final })
}
