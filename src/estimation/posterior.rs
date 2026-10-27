// SPDX-License-Identifier: GPL-3.0-or-later
//! Posterior statistics from the MCMC draws: `CutSample.m` (burn-in),
//! `posterior_moments.m` (mean, median, variance, HPD interval, deciles),
//! `compute_mh_covariance_matrix.m`, `marginal_density.m` (modified harmonic
//! mean), the Laplace approximation of `dynare_estimation_1.m`,
//! `mcmc_diagnostics.m` / `mcmc_ifac.m` (inefficiency factors per block) and
//! `mcmc_diagnostics_core.m` (Brooks–Gelman univariate diagnostics).

use anyhow::{Result, bail};
use rayon::prelude::*;

use super::mcmc::Chain;
use super::special::chi2inv;
use crate::linalg::{Mat, chol_upper, lu, solve};

/// Options of the posterior analysis (defaults from `default_option_values.m`).
#[derive(Clone, Debug)]
pub struct PosteriorOptions {
    /// `options_.mh_drop` (0.5).
    pub mh_drop: f64,
    /// `options_.mh_conf_sig` (0.9).
    pub mh_conf_sig: f64,
    /// `options_.marginal_data_density.harmonic_mean.tolerance` (0.01).
    pub mhm_tolerance: f64,
}

impl Default for PosteriorOptions {
    fn default() -> Self {
        PosteriorOptions { mh_drop: 0.5, mh_conf_sig: 0.9, mhm_tolerance: 0.01 }
    }
}

/// Output of [`posterior_moments`] for one parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamMoments {
    pub mean: f64,
    pub median: f64,
    /// Variance (`var`, n-1 normalization).
    pub var: f64,
    /// HPD interval (NaN when fewer than 3 draws fall outside it).
    pub hpd: (f64, f64),
    /// Deciles 10%..90% (NaN with fewer than 10 draws).
    pub deciles: [f64; 9],
}

/// Port of `estimation/posterior_moments.m` (without the kernel density
/// estimate, which only feeds plots). `xx` is sorted in place.
pub fn posterior_moments(xx: &mut [f64], mh_conf_sig: f64) -> ParamMoments {
    xx.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xx.len();
    let nf = n as f64;
    let mean = xx.iter().sum::<f64>() / nf;
    let median = if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        xx[n / 2]
    } else {
        (xx[n / 2 - 1] + xx[n / 2]) / 2.0
    };
    let var = if n > 1 { xx.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (nf - 1.0) } else { 0.0 };
    let hpd_draws = ((1.0 - mh_conf_sig) * nf).round() as usize;
    let hpd = if hpd_draws > 2 {
        let mut kmin = f64::INFINITY;
        let mut idx = 0;
        // kk(ii) = xx(jj) - xx(ii), jj = number_of_draws - hpd_draws + ii - 1 (1-based)
        for ii in 0..hpd_draws {
            let k = xx[n - hpd_draws - 1 + ii] - xx[ii];
            if k < kmin {
                kmin = k;
                idx = ii;
            }
        }
        (xx[idx], xx[idx] + kmin)
    } else {
        (f64::NAN, f64::NAN)
    };
    let mut deciles = [f64::NAN; 9];
    if n > 9 {
        let q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        for (d, qk) in deciles.iter_mut().zip(q) {
            *d = xx[(qk * nf).round() as usize - 1];
        }
    }
    ParamMoments { mean, median, var, hpd, deciles }
}

/// Port of `convergence_diagnostics/mcmc_ifac.m` (Parzen-window inefficiency
/// factor, autocorrelations from `dyn_autocorr`).
pub fn mcmc_ifac(x: &[f64], nc: f64) -> f64 {
    let len = x.len();
    let mut nc = nc.min(len as f64 / 2.0).floor() as usize;
    if nc % 2 == 1 {
        nc -= 1;
    }
    let acf = dyn_autocorr(x, nc);
    let mut parzen = vec![0.0; nc + 1];
    let ncf = nc as f64;
    for i in 1..=(nc / 2 + 1) {
        let r = i as f64 / ncf;
        parzen[i - 1] = 1.0 - 6.0 * r * r + 6.0 * r * r * r;
    }
    for i in (nc / 2 + 1)..=(nc + 1) {
        parzen[i - 1] = 2.0 * (1.0 - i as f64 / ncf).powi(3);
    }
    1.0 + 2.0 * parzen.iter().zip(&acf).map(|(p, a)| p * a).sum::<f64>()
}

fn dot4(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (mut s0, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
    let chunks = n / 4;
    for k in 0..chunks {
        let i = 4 * k;
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    let mut s = (s0 + s1) + (s2 + s3);
    for i in 4 * chunks..n {
        s += a[i] * b[i];
    }
    s
}

/// `dyn_autocorr` subfunction of `mcmc_ifac.m`: autocorrelations at lags 0..ar.
fn dyn_autocorr(y: &[f64], ar: usize) -> Vec<f64> {
    let n = y.len();
    let nf = n as f64;
    let m = y.iter().sum::<f64>() / nf;
    let yc: Vec<f64> = y.iter().map(|v| v - m).collect();
    let sd = (yc.iter().map(|v| v * v).sum::<f64>() / nf).sqrt();
    let denom = nf * sd * sd;
    let mut acf = vec![1.0; ar + 1];
    for i in 1..=ar {
        acf[i] = dot4(&yc[i..], &yc[..n - i]) / denom;
    }
    acf
}

/// Kept draws (after `CutSample.m`): index of the first kept draw and the
/// number of kept draws per chain.
fn kept_range(chains: &[Chain], mh_drop: f64) -> Result<(usize, usize)> {
    let Some(c0) = chains.first() else { bail!("no MCMC chains") };
    let total = c0.total_draws;
    let first = (mh_drop * total as f64).floor() as usize; // FirstDraw - 1
    for c in chains {
        if c.total_draws != total {
            bail!("chains have different lengths");
        }
        if c.first_stored > first {
            bail!("draw {} is needed but only draws from {} were stored", first + 1, c.first_stored + 1);
        }
    }
    Ok((first, total - first))
}

/// Inefficiency factors per block (`mcmc_diagnostics.m`, table
/// "MCMC Inefficiency factors per block"): `nblck x npar`, computed on the
/// kept draws with `Nc = min(1000, NumberOfDraws/2)`.
pub fn inefficiency_factors(chains: &[Chain], mh_drop: f64) -> Result<Mat> {
    let (first, nkeep) = kept_range(chains, mh_drop)?;
    let npar = chains[0].npar;
    let nblck = chains.len();
    let nc = 1000f64.min(nkeep as f64 / 2.0);
    let vals: Vec<f64> = (0..nblck * npar)
        .into_par_iter()
        .map(|k| {
            let (b, j) = (k % nblck, k / nblck);
            let c = &chains[b];
            let col: Vec<f64> = (first..first + nkeep).map(|t| c.draw(t)[j]).collect();
            mcmc_ifac(&col, nc)
        })
        .collect();
    Ok(Mat::from_column_slice(nblck, npar, &vals))
}

/// Dynare's `invhess` from the Hessian at the mode
/// (`dynare_estimation_1.m`: `inv(hh./(hsd*hsd'))./(hsd*hsd')`).
pub fn inverse_hessian(hh: &Mat) -> Option<Mat> {
    let n = hh.nrows();
    let hsd: Vec<f64> = (0..n).map(|i| hh[(i, i)].sqrt()).collect();
    let scaled = Mat::from_fn(n, n, |i, j| hh[(i, j)] / (hsd[i] * hsd[j]));
    let inv = solve(&scaled, &Mat::identity(n, n))?;
    Some(Mat::from_fn(n, n, |i, j| inv[(i, j)] / (hsd[i] * hsd[j])))
}

/// Laplace approximation of the log marginal density, as printed by
/// `dynare_estimation_1.m` ("Log data density [Laplace approximation]"):
/// `.5*n*log(2*pi) + .5*log(det(invhess)) - fval` with `hh` the Hessian of
/// the minus log posterior at the mode and `fval` its value there. NaN when
/// `invhess` is not positive definite.
pub fn laplace_approximation(hh: &Mat, fval: f64) -> f64 {
    let n = hh.nrows();
    let Some(invhess) = inverse_hessian(hh) else {
        return f64::NAN;
    };
    if chol_upper(&invhess).is_none() {
        return f64::NAN;
    }
    let stdh: Vec<f64> = (0..n).map(|i| invhess[(i, i)].sqrt()).collect();
    let normalized = Mat::from_fn(n, n, |i, j| invhess[(i, j)] / (stdh[i] * stdh[j]));
    let (f, ipiv, _) = lu(&normalized);
    let mut det = 1.0;
    for i in 0..n {
        det *= f[(i, i)];
        if ipiv[i] as usize != i + 1 {
            det = -det;
        }
    }
    let log_det_invhess = det.ln() + 2.0 * stdh.iter().map(|s| s.ln()).sum::<f64>();
    0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln() + 0.5 * log_det_invhess - fval
}

/// Values of MATLAB's `0.1:0.1:0.9` (colon operator, computed from both ends).
fn colon_p() -> [f64; 9] {
    let mut p = [0.0; 9];
    for (k, v) in p.iter_mut().enumerate() {
        *v = if k <= 4 { 0.1 + k as f64 * 0.1 } else { 0.9 - (8 - k) as f64 * 0.1 };
    }
    p
}

/// Result of [`modified_harmonic_mean`].
#[derive(Clone, Debug)]
pub struct MhmResult {
    /// `oo_.MarginalDensity.ModifiedHarmonicMean` (mean over the 9 truncations).
    pub log_marginal: f64,
    /// `[p, estimate]` for p = 0.1..0.9.
    pub table: Vec<(f64, f64)>,
    /// Final variance inflation of the weighting density.
    pub increase: f64,
}

/// Port of `estimation/marginal_density.m` (Geweke's modified harmonic mean
/// with a truncated normal weighting density, inflated by 20% while the
/// estimates for p = 0.1 and 0.9 differ by more than `tolerance`).
/// `mean`/`covariance` are the posterior moments of the kept draws and
/// `lpost_mode` their maximal log posterior kernel.
pub fn modified_harmonic_mean(
    chains: &[Chain],
    mh_drop: f64,
    mean: &[f64],
    covariance: &Mat,
    lpost_mode: f64,
    tolerance: f64,
) -> Result<MhmResult> {
    let (first, nkeep) = kept_range(chains, mh_drop)?;
    let npar = mean.len();
    let nblck = chains.len();
    let upper = chol_upper(covariance);
    let logdet = match &upper {
        Some(u) => 2.0 * (0..npar).map(|i| u[(i, i)].ln()).sum::<f64>(),
        None => f64::NAN,
    };
    let inv = solve(covariance, &Mat::identity(npar, npar));
    // deviations (x-MU)*inv(SIGMA)*(x-MU)' and log posteriors of every kept draw, chain by chain
    let per_chain: Vec<(Vec<f64>, Vec<f64>)> = chains
        .par_iter()
        .map(|c| {
            let mut dev = Vec::with_capacity(nkeep);
            let mut lp = Vec::with_capacity(nkeep);
            let mut w = vec![0.0; npar];
            for t in first..first + nkeep {
                let x = c.draw(t);
                for k in 0..npar {
                    w[k] = x[k] - mean[k];
                }
                let d = match (&upper, &inv) {
                    (Some(u), _) => {
                        // solve U' v = w (forward substitution), dev = |v|^2
                        for k in 0..npar {
                            let mut s = w[k];
                            for i in 0..k {
                                s -= u[(i, k)] * w[i];
                            }
                            w[k] = s / u[(k, k)];
                        }
                        w.iter().map(|v| v * v).sum()
                    }
                    (None, Some(m)) => {
                        let mut s = 0.0;
                        for j in 0..npar {
                            for i in 0..npar {
                                s += w[i] * m[(i, j)] * w[j];
                            }
                        }
                        s
                    }
                    (None, None) => f64::NAN,
                };
                dev.push(d);
                lp.push(c.logpost_at(t));
            }
            (dev, lp)
        })
        .collect();
    let ps = colon_p();
    let nf = npar as f64;
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    let count = (nkeep * nblck) as f64;
    let mut increase = 1.0_f64;
    let mut table: Vec<(f64, f64)>;
    loop {
        table = ps
            .iter()
            .map(|&p| {
                let critval = chi2inv(p, nf);
                let mut tmp = 0.0;
                let mut coverage = Vec::new();
                for (dev, lp) in &per_chain {
                    for (d, l) in dev.iter().zip(lp) {
                        let deviation = d / increase;
                        if deviation <= critval {
                            let lftheta = -p.ln() - (nf * ln2pi + (nf * increase.ln() + logdet) + deviation) / 2.0;
                            let term = lftheta - l + lpost_mode;
                            tmp += term.exp();
                            coverage.push(term);
                        }
                    }
                }
                // Dynare 7.2: renormalize the log-sum when `tmp` overflows to Inf.
                let mut norm_proba = 0.0;
                if tmp.is_infinite() {
                    norm_proba = coverage.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    tmp = coverage.iter().map(|c| (c - norm_proba).exp()).sum();
                }
                (p, lpost_mode - (tmp / count).ln() - norm_proba)
            })
            .collect();
        let (m1, m9) = (table[0].1, table[8].1);
        // Dynare 7.2: retry when any row is infinite (7.1 tested row 1 only).
        if ((m9 - m1) / m9).abs() > tolerance || table.iter().any(|t| t.1.is_infinite()) {
            increase *= 1.2;
            if increase > 20.0 {
                break;
            }
        } else {
            break;
        }
    }
    let log_marginal = table.iter().map(|t| t.1).sum::<f64>() / 9.0;
    Ok(MhmResult { log_marginal, table, increase })
}

/// Posterior summary of a run.
#[derive(Clone, Debug)]
pub struct PosteriorStats {
    /// 0-based index of the first kept draw (`FirstDraw - 1` of `CutSample.m`).
    pub first_draw: usize,
    /// Kept draws per chain.
    pub kept_per_chain: usize,
    /// Per-parameter moments on the pooled kept draws.
    pub params: Vec<ParamMoments>,
    /// Posterior mean and covariance (1/N normalization, `recursive_moments`).
    pub mean: Vec<f64>,
    pub covariance: Mat,
    /// Kept draw with the highest log posterior kernel, and that kernel.
    pub mode: Vec<f64>,
    pub kernel_at_mode: f64,
    /// Modified harmonic mean estimate of the log marginal density.
    pub mhm: MhmResult,
    /// Inefficiency factors per block (`nblck x npar`).
    pub ifac: Mat,
    /// Acceptance ratio per chain.
    pub acceptance: Vec<f64>,
}

/// Burn-in, moments/HPD per parameter (`GetPosteriorParametersStatistics.m`),
/// posterior mean/covariance/mode (`compute_mh_covariance_matrix.m`), MHM
/// marginal density (`marginal_density.m`) and inefficiency factors.
pub fn posterior_stats(chains: &[Chain], opts: &PosteriorOptions) -> Result<PosteriorStats> {
    let (first, nkeep) = kept_range(chains, opts.mh_drop)?;
    let npar = chains[0].npar;
    let ntot = nkeep * chains.len();
    let params: Vec<ParamMoments> = (0..npar)
        .into_par_iter()
        .map(|j| {
            let mut col: Vec<f64> = Vec::with_capacity(ntot);
            for c in chains {
                col.extend((first..first + nkeep).map(|t| c.draw(t)[j]));
            }
            posterior_moments(&mut col, opts.mh_conf_sig)
        })
        .collect();
    // mean and covariance (two passes, per chain in parallel then combined in order)
    let sums: Vec<Vec<f64>> = chains
        .par_iter()
        .map(|c| {
            let mut s = vec![0.0; npar];
            for t in first..first + nkeep {
                for (sk, x) in s.iter_mut().zip(c.draw(t)) {
                    *sk += x;
                }
            }
            s
        })
        .collect();
    let mut mean = vec![0.0; npar];
    for s in &sums {
        for (m, v) in mean.iter_mut().zip(s) {
            *m += v;
        }
    }
    for m in mean.iter_mut() {
        *m /= ntot as f64;
    }
    let covs: Vec<Mat> = chains
        .par_iter()
        .map(|c| {
            // blocks of draws so the deviations never take more than a few MB
            const BLOCK: usize = 4096;
            let mut acc = Mat::zeros(npar, npar);
            let mut start = first;
            while start < first + nkeep {
                let len = BLOCK.min(first + nkeep - start);
                let mut dev = Mat::zeros(npar, len);
                for r in 0..len {
                    for (k, x) in c.draw(start + r).iter().enumerate() {
                        dev[(k, r)] = x - mean[k];
                    }
                }
                crate::linalg::gemm(1.0, &dev, false, &dev, true, 1.0, &mut acc);
                start += len;
            }
            acc
        })
        .collect();
    let mut covariance = Mat::zeros(npar, npar);
    for c in &covs {
        covariance += c;
    }
    covariance /= ntot as f64;
    let mut kernel_at_mode = f64::NEG_INFINITY;
    let mut mode = vec![f64::NAN; npar];
    for c in chains {
        for t in first..first + nkeep {
            let lp = c.logpost_at(t);
            if lp > kernel_at_mode {
                kernel_at_mode = lp;
                mode = c.draw(t).to_vec();
            }
        }
    }
    let mhm = modified_harmonic_mean(chains, opts.mh_drop, &mean, &covariance, kernel_at_mode, opts.mhm_tolerance)?;
    let ifac = inefficiency_factors(chains, opts.mh_drop)?;
    let acceptance = chains.iter().map(|c| c.acceptance_ratio()).collect();
    Ok(PosteriorStats {
        first_draw: first,
        kept_per_chain: nkeep,
        params,
        mean,
        covariance,
        mode,
        kernel_at_mode,
        mhm,
        ifac,
        acceptance,
    })
}

/// Brooks–Gelman (1998) univariate diagnostics of `mcmc_diagnostics_core.m`.
#[derive(Clone, Debug)]
pub struct BrooksGelman {
    /// Window end points (`Origin:StepSize:NumberOfDraws`, 1-based draw counts).
    pub iterations: Vec<usize>,
    /// Per parameter, per window: `UDIAG` columns
    /// [pooled 80% interval, mean within-chain interval, pooled variance,
    ///  mean within variance, pooled 3rd abs moment, mean within 3rd abs moment].
    pub udiag: Vec<Vec<[f64; 6]>>,
}

/// Port of the Brooks–Gelman part of `mcmc_diagnostics.m` /
/// `mcmc_diagnostics_core.m` (RW-MH: `Origin = 1000`,
/// `StepSize = ceil((N-Origin)/100)`). Needs all draws stored and at least
/// two chains of 1000 draws; returns None otherwise (Dynare skips it too).
pub fn brooks_gelman(chains: &[Chain], mh_drop: f64, slice: bool) -> Option<BrooksGelman> {
    let nblck = chains.len();
    if nblck < 2 || chains.iter().any(|c| c.first_stored != 0) {
        return None;
    }
    let ndraws = chains[0].total_draws;
    // mcmc_diagnostics.m:218-223: slice with fewer than 2000 draws uses Origin = StepSize = 1
    let (origin, step) =
        if slice && ndraws < 2000 { (1, 1) } else { (1000, (ndraws - 1000.min(ndraws)).div_ceil(100).max(1)) };
    if ndraws < origin {
        return None;
    }
    let iterations: Vec<usize> = (origin..=ndraws).step_by(step).collect();
    let npar = chains[0].npar;
    const ALPHA: f64 = 0.2;
    let udiag = (0..npar)
        .into_par_iter()
        .map(|j| {
            // (value, chain, time) sorted by value (stable, like sortrows)
            let mut tmp: Vec<(f64, u32, u32)> = Vec::with_capacity(ndraws * nblck);
            for (b, c) in chains.iter().enumerate() {
                for t in 0..ndraws {
                    tmp.push((c.draw(t)[j], b as u32, (t + 1) as u32));
                }
            }
            tmp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut out = Vec::with_capacity(iterations.len());
            let mut pooled: Vec<f64> = Vec::new();
            let mut within: Vec<Vec<f64>> = vec![Vec::new(); nblck];
            for &iter in &iterations {
                let linea = (mh_drop * iter as f64).ceil() as usize;
                let n = iter - linea + 1;
                let nf = n as f64;
                let cinf = 1usize.max((nf * ALPHA / 2.0).round() as usize);
                let csup = (nf * (1.0 - ALPHA / 2.0)).round() as usize;
                let cinf_p = 1usize.max((nblck as f64 * nf * ALPHA / 2.0).round() as usize);
                let csup_p = (nblck as f64 * nf * (1.0 - ALPHA / 2.0)).round() as usize;
                pooled.clear();
                for w in within.iter_mut() {
                    w.clear();
                }
                for &(v, b, t) in &tmp {
                    let t = t as usize;
                    if t >= linea && t <= iter {
                        pooled.push(v);
                        within[b as usize].push(v);
                    }
                }
                let mut u = [0.0; 6];
                u[0] = pooled[csup_p - 1] - pooled[cinf_p - 1];
                let pm = pooled.iter().sum::<f64>() / pooled.len() as f64;
                let denom = (nblck * n) as f64 - 1.0;
                u[2] = pooled.iter().map(|v| (v - pm).powi(2)).sum::<f64>() / denom;
                u[4] = pooled.iter().map(|v| (v - pm).abs().powi(3)).sum::<f64>() / denom;
                for w in &within {
                    u[1] += w[csup - 1] - w[cinf - 1];
                    let wm = w.iter().sum::<f64>() / w.len() as f64;
                    u[3] += w.iter().map(|v| (v - wm).powi(2)).sum::<f64>() / (nf - 1.0);
                    u[5] += w.iter().map(|v| (v - wm).abs().powi(3)).sum::<f64>() / (nf - 1.0);
                }
                for k in [1, 3, 5] {
                    u[k] /= nblck as f64;
                }
                out.push(u);
            }
            out
        })
        .collect();
    Some(BrooksGelman { iterations, udiag })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hpd_of_uniform_grid() {
        let mut xx: Vec<f64> = (0..1000).map(|k| k as f64).collect();
        let m = posterior_moments(&mut xx, 0.9);
        assert_eq!(m.mean, 499.5);
        assert_eq!(m.median, 499.5);
        // hpd_draws = 100: kk(ii) = xx(900+ii-1) - xx(ii) (1-based) = 899 constant -> first
        assert_eq!(m.hpd, (0.0, 899.0));
        assert_eq!(m.deciles[0], 99.0);
    }

    #[test]
    fn ifac_includes_lag_zero_like_dynare() {
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let x: Vec<f64> = (0..20000).map(|_| StandardNormal.sample(&mut rng)).collect();
        let f = mcmc_ifac(&x, 1000.0);
        // mcmc_ifac.m weights lag 0 too (Parzen(1) ~ 1): iid draws give ~3, not 1
        assert!((f - 3.0).abs() < 0.3, "{f}");
    }

    /// Dynare 7.2 marginal_density.m: when the sum of exp(lftheta - logpo2 + lpost_mode)
    /// overflows, the log-sum is renormalized (log posteriors shifted by -800 instead of -600,
    /// which overflows, shift the estimate by exactly -200 instead of giving -Inf).
    #[test]
    fn mhm_overflow_renormalized() {
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let n = 4000;
        let draws: Vec<f64> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        let chain = |shift: f64| Chain {
            npar: 1,
            draws: draws.clone(),
            logpost: draws.iter().map(|x| -0.5 * x * x + shift).collect(),
            runs: Vec::new(),
            first_stored: 0,
            total_draws: n,
            accepted: n,
            init: vec![0.0],
            init_logpost: 0.0,
        };
        let kept = &draws[n / 2..];
        let mean = kept.iter().sum::<f64>() / kept.len() as f64;
        let var = kept.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (kept.len() - 1) as f64;
        let cov = Mat::from_element(1, 1, var);
        let base = modified_harmonic_mean(&[chain(-600.0)], 0.5, &[mean], &cov, 0.0, 0.01).unwrap();
        let shifted = modified_harmonic_mean(&[chain(-800.0)], 0.5, &[mean], &cov, 0.0, 0.01).unwrap();
        assert!(base.log_marginal.is_finite() && shifted.log_marginal.is_finite());
        assert_eq!(base.increase, shifted.increase);
        assert!(
            (shifted.log_marginal - (base.log_marginal - 200.0)).abs() < 1e-9,
            "{} vs {}",
            shifted.log_marginal,
            base.log_marginal
        );
    }

    #[test]
    fn laplace_gaussian() {
        // f(x) = 0.5 x'Ax + c: integral of exp(-f) = exp(-c) (2pi)^{n/2} det(A)^{-1/2}
        let a = Mat::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.0]);
        let c = 3.0;
        let l = laplace_approximation(&a, c);
        let det: f64 = 2.0 - 0.25;
        let truth = -c + (2.0 * std::f64::consts::PI).ln() - 0.5 * det.ln();
        assert!((l - truth).abs() < 1e-12);
    }
}
