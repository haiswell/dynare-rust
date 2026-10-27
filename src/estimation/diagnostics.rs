// SPDX-License-Identifier: GPL-3.0-or-later
//! MCMC convergence diagnostics and posterior densities for the plots:
//! - Geweke (1992) tests of `mcmc_diagnostics.m` (`geweke_moments.m`, `geweke_chi2_test.m`),
//!   printed as Dynare prints them;
//! - the multivariate Brooks–Gelman diagnostic on the log posterior kernel (`MDIAG`); the
//!   univariate one is [`super::brooks_gelman`];
//! - the kernel density estimate of the posterior marginals (`posterior_moments.m` ->
//!   `mh_optimal_bandwidth.m` + `kernel_density_estimate.m`, Gaussian kernel, rule-of-thumb
//!   bandwidth, 512 grid points).

use rayon::prelude::*;

use super::mcmc::Chain;
use super::special::gammainc;

// ---------------------------------------------------------------------------
// kernel density
// ---------------------------------------------------------------------------

fn gaussian(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// `correction_for_repeated_draws` of `mh_optimal_bandwidth.m` (`draws` sorted).
fn correction_for_repeated_draws(draws: &[f64]) -> f64 {
    let n = draws.len();
    let mut c = 0.0;
    let mut j = 0;
    for i in 0..n {
        // j only moves forward: the run of draws within 2 eps of draws[i]
        j = j.max(i);
        while j < n && draws[j] - draws[i] < 2.0 * f64::EPSILON {
            j += 1;
        }
        c += 2.0 * (j - i) as f64 - 1.0;
    }
    c / n as f64
}

/// Port of `mh_optimal_bandwidth.m` for `bandwidth = 0` (Silverman's rule of thumb with
/// the correction for repeated draws) and the Gaussian kernel; `data` sorted.
pub fn mh_optimal_bandwidth(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mu02 = 1.0 / (2.0 * std::f64::consts::PI.sqrt());
    let mu21 = 1.0;
    let correction = correction_for_repeated_draws(data);
    let mean = data.iter().sum::<f64>() / n;
    let sigma = (data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let h = 2.0 * sigma * (std::f64::consts::PI.sqrt() * mu02 / (12.0 * mu21 * mu21 * n)).powf(0.2);
    h * correction.powf(0.2)
}

/// Port of `kernel_density_estimate.m` (Gaussian kernel): the binned data convolved with
/// the kernel. The FFT circular convolution of the MATLAB code is a plain linear
/// convolution here (identical up to rounding). Returns (abscissa, density).
pub fn kernel_density_estimate(data: &[f64], grid_points: usize, bandwidth: f64) -> (Vec<f64>, Vec<f64>) {
    let n = grid_points;
    let (min, max) = data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let lower = min - (max - min) / 3.0;
    let upper = max + (max - min) / 3.0;
    let abscissa = linspace(lower, upper, n);
    let inc = abscissa[1] - abscissa[0];
    let mut xi = vec![0.0; n + 2];
    for &d in data {
        let xa = (d - lower) / (upper - lower) * n as f64;
        let indx = xa.floor();
        let t = xa - indx;
        let i = indx as usize;
        xi[i] += 1.0 - t;
        xi[i + 1] += t;
    }
    // kk over xk = (-n..n-1)*inc, normalized
    let kk: Vec<f64> = (0..2 * n).map(|j| gaussian((j as f64 - n as f64) * inc / bandwidth)).collect();
    let norm = kk.iter().sum::<f64>() * inc * data.len() as f64;
    let f: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut s = 0.0;
            for (j, &x) in xi.iter().enumerate().take(n) {
                if x != 0.0 {
                    s += x * kk[i + n - j];
                }
            }
            s / norm
        })
        .collect();
    (abscissa, f)
}

/// MATLAB `linspace(a, b, n)`.
pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![b];
    }
    let d = (b - a) / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|k| a + k as f64 * d).collect();
    v[n - 1] = b;
    v
}

/// Posterior density of each parameter from the pooled kept draws
/// (`oo_.posterior_density`), as `GetPosteriorParametersStatistics.m` computes it;
/// None for a parameter whose posterior variance is <= 1e-12.
pub fn posterior_densities(chains: &[Chain], first_draw: usize) -> Vec<Option<(Vec<f64>, Vec<f64>)>> {
    let npar = chains.first().map_or(0, |c| c.npar);
    (0..npar)
        .into_par_iter()
        .map(|j| {
            let mut xx: Vec<f64> = Vec::new();
            for c in chains {
                xx.extend((first_draw.max(c.first_stored)..c.total_draws).map(|t| c.draw(t)[j]));
            }
            if xx.len() < 2 {
                return None;
            }
            xx.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = xx.len() as f64;
            let m = xx.iter().sum::<f64>() / n;
            let var = xx.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0);
            if var <= 1e-12 {
                return None;
            }
            let h = mh_optimal_bandwidth(&xx);
            Some(kernel_density_estimate(&xx, 512, h))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Geweke
// ---------------------------------------------------------------------------

/// `options_.convergence.geweke`.
#[derive(Clone, Debug)]
pub struct GewekeOptions {
    pub taper_steps: Vec<usize>,
    pub interval: [f64; 2],
}

impl Default for GewekeOptions {
    fn default() -> Self {
        GewekeOptions { taper_steps: vec![4, 8, 15], interval: [0.2, 0.5] }
    }
}

/// Port of `geweke_moments.m`: `[mean, std, NSE_iid, RNE_iid, (NSE_taper, RNE_taper)...]`.
pub fn geweke_moments(draws: &[f64], taper_steps: &[usize]) -> Vec<f64> {
    let n_groups = 100;
    let ns = draws.len() / n_groups;
    let used = ns * n_groups;
    let mut out = vec![0.0; 4 + 2 * taper_steps.len()];
    let mut wm = vec![0.0; n_groups];
    let mut wv = vec![0.0; n_groups];
    for g in 0..n_groups {
        let w = &draws[g * ns..(g + 1) * ns];
        wm[g] = w.iter().sum::<f64>() / ns as f64;
        wv[g] = w.iter().map(|v| v * v).sum::<f64>() / ns as f64;
    }
    let total_mean = wm.iter().sum::<f64>() / n_groups as f64;
    let total_var = wv.iter().sum::<f64>() / n_groups as f64 - total_mean * total_mean;
    out[0] = total_mean;
    out[1] = total_var.sqrt();
    let u = &draws[..used];
    let m = u.iter().sum::<f64>() / used as f64;
    let nse = (u.iter().map(|v| (v - m).powi(2)).sum::<f64>() / used as f64).sqrt() / (used as f64).sqrt();
    out[2] = nse;
    out[3] = total_var / (used as f64 * nse * nse);
    let c: Vec<f64> = wm.iter().map(|v| v - total_mean).collect();
    let autocov: Vec<f64> =
        (0..n_groups).map(|lag| (lag..n_groups).map(|i| c[i] * c[i - lag]).sum::<f64>() / n_groups as f64).collect();
    for (ti, &taper) in taper_steps.iter().enumerate() {
        let mut s = autocov[0];
        for lag in 1..taper {
            s += 2.0 * (1.0 - lag as f64 / taper as f64) * autocov[lag];
        }
        let nse_t = (s / n_groups as f64).sqrt();
        out[4 + 2 * ti] = nse_t;
        out[5 + 2 * ti] = total_var / (used as f64 * nse_t * nse_t);
    }
    out
}

/// `1 - chi2cdf(x, 1)`.
fn chi2_1_sf(x: f64) -> f64 {
    1.0 - gammainc(x / 2.0, 0.5)
}

/// Port of `geweke_chi2_test.m`: p-values (no taper, then each taper).
pub fn geweke_chi2_test(r1: &[f64], r2: &[f64], ntaper: usize) -> Vec<f64> {
    (0..=ntaper)
        .map(|k| {
            let (n1, n2) = (r1[2 + 2 * k], r2[2 + 2 * k]);
            let stat = (r1[0] - r2[0]).powi(2) / (n1 * n1 + n2 * n2);
            chi2_1_sf(stat)
        })
        .collect()
}

/// One chain's Geweke table: per parameter `[post. mean, post. std, p-values...]` and the
/// sample bounds (1-based) `(first_begin, last_begin, first_end, n)`.
pub struct GewekeChain {
    pub rows: Vec<Vec<f64>>,
    pub samples: (usize, usize, usize, usize),
}

/// The Geweke part of `mcmc_diagnostics.m` for every chain (all draws stored).
pub fn geweke(chains: &[Chain], mh_drop: f64, opts: &GewekeOptions) -> Option<Vec<GewekeChain>> {
    if chains.iter().any(|c| c.first_stored != 0) || chains.is_empty() {
        return None;
    }
    let n = chains[0].total_draws;
    let nf = n as f64;
    let first_begin = 1usize.max((mh_drop * nf).ceil() as usize);
    let last_begin = first_begin + (opts.interval[0] * nf * (1.0 - mh_drop)).round() as usize;
    let first_end = first_begin + (opts.interval[1] * nf * (1.0 - mh_drop)).round() as usize;
    if last_begin > n || first_end > n {
        return None;
    }
    let npar = chains[0].npar;
    Some(
        chains
            .par_iter()
            .map(|c| {
                let rows = (0..npar)
                    .map(|j| {
                        let d: Vec<f64> = (0..n).map(|t| c.draw(t)[j]).collect();
                        let all = geweke_moments(&d, &opts.taper_steps);
                        let r1 = geweke_moments(&d[first_begin - 1..last_begin], &opts.taper_steps);
                        let r2 = geweke_moments(&d[first_end - 1..], &opts.taper_steps);
                        let mut row = vec![all[0], all[1]];
                        row.extend(geweke_chi2_test(&r1, &r2, opts.taper_steps.len()));
                        row
                    })
                    .collect();
                GewekeChain { rows, samples: (first_begin, last_begin, first_end, n) }
            })
            .collect(),
    )
}

/// Dynare's `dyntable` (label column, then `%W.Pf` columns).
pub fn dyntable(
    title: &str,
    headers: &[String],
    labels: &[String],
    values: &[Vec<f64>],
    label_width: usize,
    val_width: usize,
    precision: usize,
) {
    let lw = label_width.max(labels.iter().chain(std::iter::once(&headers[0])).map(|s| s.len()).max().unwrap_or(0) + 2);
    let finite: Vec<f64> = values.iter().flatten().copied().filter(|v| v.is_finite()).collect();
    let mut values_length = if finite.is_empty() {
        4
    } else {
        (finite.iter().map(|v| v.abs().log10()).fold(f64::NEG_INFINITY, f64::max).ceil() as i64).max(1) as usize
            + precision
            + 1
    };
    if values.iter().flatten().any(|v| *v < 0.0) {
        values_length += 1;
    }
    let headers_length = headers[1..].iter().map(|s| s.len()).max().unwrap_or(0);
    let vw = val_width.max(headers_length.max(values_length) + 2);
    if !title.is_empty() {
        println!("\n\n{title}");
    }
    let mut h = format!("{:<lw$}", headers[0]);
    for x in &headers[1..] {
        h += &format!("{x:>vw$}");
    }
    println!("{h}");
    for (l, row) in labels.iter().zip(values) {
        let mut s = format!("{l:<lw$}");
        for v in row {
            s += &format!("{v:>vw$.precision$}");
        }
        println!("{s}");
    }
}

/// Prints the Geweke tables like `mcmc_diagnostics.m`.
pub fn print_geweke(res: &[GewekeChain], names: &[String], opts: &GewekeOptions) {
    let mut headers: Vec<String> =
        ["Parameter", "Post. Mean", "Post. Std", "p-val No Taper"].iter().map(|s| s.to_string()).collect();
    for t in &opts.taper_steps {
        headers.push(format!("p-val {t}% Taper"));
    }
    for (b, g) in res.iter().enumerate() {
        let (fb, lb, fe, n) = g.samples;
        println!("\n\nConvergence diagnostics results for chain {}.", b + 1);
        println!(
            "\nGeweke (1992) Convergence Tests, based on means of draws {fb} to {lb} vs {fe} to {n} for chain {}.",
            b + 1
        );
        println!("p-values are for Chi2-test for equality of means.");
        dyntable("", &headers, names, &g.rows, 3, 12, 3);
        println!("\n");
    }
}

// ---------------------------------------------------------------------------
// multivariate Brooks–Gelman
// ---------------------------------------------------------------------------

/// `MDIAG` of `mcmc_diagnostics.m`: the Brooks–Gelman criteria (80% interval, m2, m3;
/// pooled and mean within-chain) of the log posterior kernel, on the windows
/// `Origin:StepSize:NumberOfDraws` (Origin 1000). Needs all draws stored and two chains.
pub fn brooks_gelman_multivariate(chains: &[Chain], mh_drop: f64, slice: bool) -> Option<(Vec<usize>, Vec<[f64; 6]>)> {
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
    let mut tmp: Vec<(f64, u32, u32)> = Vec::with_capacity(ndraws * nblck);
    for (b, c) in chains.iter().enumerate() {
        for t in 0..ndraws {
            tmp.push((c.logpost_at(c.first_stored + t), b as u32, (t + 1) as u32));
        }
    }
    tmp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    const ALPHA: f64 = 0.2;
    let out = iterations
        .par_iter()
        .map(|&iter| {
            let linea = (mh_drop * iter as f64).ceil() as usize;
            let n = iter - linea + 1;
            let nf = n as f64;
            let cinf = 1usize.max((nf * ALPHA / 2.0).round() as usize);
            let csup = (nf * (1.0 - ALPHA / 2.0)).round() as usize;
            let cinf_p = 1usize.max((nblck as f64 * nf * ALPHA / 2.0).round() as usize);
            let csup_p = (nblck as f64 * nf * (1.0 - ALPHA / 2.0)).round() as usize;
            let mut pooled = Vec::with_capacity(n * nblck);
            let mut within: Vec<Vec<f64>> = vec![Vec::with_capacity(n); nblck];
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
            u
        })
        .collect();
    Some((iterations, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    #[test]
    fn kde_of_normal_draws() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let mut x: Vec<f64> = (0..20000).map(|_| StandardNormal.sample(&mut rng)).collect();
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let h = mh_optimal_bandwidth(&x);
        // Silverman: 1.06 sigma n^-1/5
        assert!((h - 1.0592 * 20000f64.powf(-0.2)).abs() < 0.01, "{h}");
        let (a, f) = kernel_density_estimate(&x, 512, h);
        let integral: f64 = f.iter().sum::<f64>() * (a[1] - a[0]);
        assert!((integral - 1.0).abs() < 1e-3, "{integral}");
        let i0 = a.iter().position(|v| *v >= 0.0).unwrap();
        assert!((f[i0] - 0.3989).abs() < 0.02, "{}", f[i0]);
        // tripled draws (rejections) carry no extra information: same bandwidth
        let rep: Vec<f64> = x.iter().flat_map(|v| [*v, *v, *v]).collect();
        assert!((mh_optimal_bandwidth(&rep) / h - 1.0).abs() < 1e-3);
    }

    #[test]
    fn geweke_iid_and_chi2() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        let x: Vec<f64> = (0..10000).map(|_| StandardNormal.sample(&mut rng)).collect();
        let r = geweke_moments(&x, &[4, 8, 15]);
        assert!(r[0].abs() < 0.05 && (r[1] - 1.0).abs() < 0.05);
        assert!((r[3] - 1.0).abs() < 1e-9, "rne_iid {}", r[3]);
        assert!((chi2_1_sf(3.841458820694124) - 0.05).abs() < 1e-9);
        let p = geweke_chi2_test(&r, &r, 3);
        assert!(p.iter().all(|v| (*v - 1.0).abs() < 1e-12));
    }
}
