// SPDX-License-Identifier: GPL-3.0-or-later
//! `options_.convergence.*` of `mcmc_diagnostics.m`: the Geweke options
//! (`taper_steps`, `geweke_interval`), the Raftery–Lewis (1992) diagnostic
//! (`raftery_lewis_diagnostics`, `raftery_lewis_qrs`; port of
//! `convergence_diagnostics/raftery_lewis.m`) and `brooks_gelman_plotrows`.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::collapsible_if
)] // index loops mirror the MATLAB code

use super::diagnostics::{GewekeOptions, dyntable};
use super::special::norminv;
use crate::modfile::{OptValue, Options, opt};

/// `options_.convergence`.
#[derive(Clone, Debug)]
pub struct ConvergenceOptions {
    pub geweke: GewekeOptions,
    /// `rafterylewis.indicator` / `rafterylewis.qrs` (`[q r s]`).
    pub raftery_lewis: bool,
    pub raftery_lewis_qrs: [f64; 3],
    /// `brooksgelman.plotrows`.
    pub brooks_gelman_plotrows: usize,
    /// Lines of the last `_mh<k>_blck<b>.mat` file of each chain: `mcmc_diagnostics.m` runs
    /// `raftery_lewis(x2, ...)` on the `x2` left by its loading loop, i.e. on the last mh file
    /// only (`record.MhDraws(end,3)`). `None`: all draws.
    pub last_file_lines: Option<usize>,
    /// `posterior_sampling_method = 'slice'`: no 2000-draw cutoff, `Origin = StepSize = 1`
    /// below 2000 draws (mcmc_diagnostics.m:110, 218-223).
    pub slice: bool,
}

impl Default for ConvergenceOptions {
    fn default() -> Self {
        ConvergenceOptions {
            geweke: GewekeOptions::default(),
            raftery_lewis: false,
            raftery_lewis_qrs: [0.025, 0.005, 0.95],
            brooks_gelman_plotrows: 3,
            last_file_lines: None,
            slice: false,
        }
    }
}

fn numbers(v: &OptValue) -> Option<Vec<f64>> {
    match v {
        OptValue::Num(x) => Some(vec![*x]),
        OptValue::Vec(x) => Some(x.clone()),
        OptValue::List(l) => l.iter().map(|x| if let OptValue::Num(v) = x { Some(*v) } else { None }).collect(),
        _ => None,
    }
}

impl ConvergenceOptions {
    /// From the `estimation` options (the preprocessor's `taper_steps`, `geweke_interval`,
    /// `raftery_lewis_diagnostics`, `raftery_lewis_qrs`, `brooks_gelman_plotrows`).
    pub fn from_options(o: &Options) -> Self {
        let mut c = ConvergenceOptions::default();
        c.slice = matches!(opt(o, "posterior_sampling_method"), Some(OptValue::Str(s)) if s.trim_matches(|ch| ch == '\'' || ch == '"') == "slice");
        if let Some(v) = opt(o, "taper_steps").and_then(numbers) {
            c.geweke.taper_steps = v.iter().map(|&x| x as usize).collect();
        }
        if let Some(v) = opt(o, "geweke_interval").and_then(numbers) {
            // mcmc_diagnostics.m: an invalid interval falls back to the default
            if v.len() == 2 && v.iter().all(|&x| (0.0..=1.0).contains(&x)) && v[1] >= v[0] {
                c.geweke.interval = [v[0], v[1]];
            } else {
                println!(
                    "\nCONVERGENCE DIAGNOSTICS: Invalid option for geweke_interval. Using the default of [0.2 0.5]."
                );
            }
        }
        c.raftery_lewis = match opt(o, "raftery_lewis_diagnostics") {
            Some(OptValue::Flag) => true,
            Some(OptValue::Num(v)) => *v != 0.0,
            _ => false,
        };
        if let Some(v) = opt(o, "raftery_lewis_qrs").and_then(numbers) {
            if v.len() == 3 && v.iter().all(|&x| (0.0..=1.0).contains(&x)) && v[0] - v[1] > 0.0 {
                c.raftery_lewis_qrs = [v[0], v[1], v[2]];
            } else {
                println!("\nInvalid option for raftery_lewis_qrs. Using the default of [0.025 0.005 0.95].");
            }
        }
        if let Some(OptValue::Num(v)) = opt(o, "brooks_gelman_plotrows") {
            c.brooks_gelman_plotrows = (*v as usize).max(1);
        }
        c
    }
}

/// `oo_.convergence.raftery_lewis(b)` for one chain (one entry per parameter).
#[derive(Clone, Debug, Default)]
pub struct RafteryLewis {
    pub m_burn: Vec<f64>,
    pub n_prec: Vec<f64>,
    pub k_thin: Vec<f64>,
    pub k_ind: Vec<f64>,
    pub i_stat: Vec<f64>,
    pub n_total: Vec<f64>,
    pub n_min: f64,
}

/// `quantile(x, p)` as Dynare 7.2 runs it: `missing/stats/quantile.m` (method R-8,
/// `h = (N+1/3) p + 1/3`, linear interpolation, clamped), which dynare_config.m puts on the
/// path when the Statistics Toolbox is not available (as with the reference installation).
pub fn matlab_quantile(x: &[f64], p: f64) -> f64 {
    let mut s: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    match n {
        0 => return f64::NAN,
        1 => return s[0],
        _ => {}
    }
    let nf = n as f64;
    if p < (2.0 / 3.0) / (nf + 1.0 / 3.0) {
        return s[0];
    }
    if p >= (nf - 1.0 / 3.0) / (nf + 1.0 / 3.0) {
        return s[n - 1];
    }
    let h = (nf + 1.0 / 3.0) * p + 1.0 / 3.0;
    let lo = h.floor();
    let i = lo as usize - 1;
    s[i] + (h - lo) * (s[i + 1] - s[i])
}

fn first_vs_second_order_bic(d: &[u8]) -> f64 {
    let n = d.len();
    let mut tran = [[[0.0f64; 2]; 2]; 2];
    for t in 2..n {
        tran[d[t - 2] as usize][d[t - 1] as usize][d[t] as usize] += 1.0;
    }
    let mut g2 = 0.0;
    for i1 in 0..2 {
        for i2 in 0..2 {
            for i3 in 0..2 {
                if tran[i1][i2][i3] != 0.0 {
                    let fitted = (tran[i1][i2][0] + tran[i1][i2][1]) * (tran[0][i2][i3] + tran[1][i2][i3])
                        / (tran[0][i2][0] + tran[0][i2][1] + tran[1][i2][0] + tran[1][i2][1]);
                    let focus = tran[i1][i2][i3];
                    g2 += (focus / fitted).ln() * focus;
                }
            }
        }
    }
    2.0 * g2 - ((n as f64) - 2.0).ln() * 2.0
}

fn independence_bic(d: &[u8]) -> f64 {
    let n = d.len();
    let mut tr = [[0.0f64; 2]; 2];
    for t in 1..n {
        tr[d[t - 1] as usize][d[t] as usize] += 1.0;
    }
    let dcm1 = n as f64 - 1.0;
    let mut g2 = 0.0;
    for i1 in 0..2 {
        for i2 in 0..2 {
            if tr[i1][i2] != 0.0 {
                let fitted = ((tr[i1][0] + tr[i1][1]) * (tr[0][i2] + tr[1][i2])) / dcm1;
                let focus = tr[i1][i2];
                g2 += (focus / fitted).ln() * focus;
            }
        }
    }
    2.0 * g2 - dcm1.ln()
}

fn fix(x: f64) -> f64 {
    x.trunc()
}

/// Port of `convergence_diagnostics/raftery_lewis.m`; `runs[v]` are the draws of parameter `v`.
pub fn raftery_lewis(runs: &[Vec<f64>], q: f64, r: f64, s: f64) -> RafteryLewis {
    let phi = norminv((s + 1.0) / 2.0, 0.0, 1.0);
    let mut out = RafteryLewis { n_min: fix(phi * phi * (1.0 - q) * q / (r * r) + 1.0), ..Default::default() };
    let epss = 0.001;
    for x in runs {
        let n_runs = x.len();
        let qv = matlab_quantile(x, q);
        let work: Vec<u8> = x.iter().map(|&v| u8::from(v <= qv)).collect();
        let thin = |k: usize| -> Vec<u8> { work.iter().step_by(k).copied().collect() };
        let mut k_thin = 1usize;
        let mut thinned;
        loop {
            thinned = thin(k_thin);
            let bic = first_vs_second_order_bic(&thinned);
            k_thin += 1;
            if !(bic > 0.0) {
                break;
            }
        }
        k_thin -= 1;
        let mut tm = [[0.0f64; 2]; 2];
        for i in 1..thinned.len() {
            tm[thinned[i - 1] as usize][thinned[i] as usize] += 1.0;
        }
        let alpha = tm[0][1] / (tm[0][0] + tm[0][1]);
        let beta = tm[1][0] / (tm[1][0] + tm[1][1]);
        let mut kmind = k_thin;
        let mut bic = independence_bic(&thinned);
        while bic > 0.0 {
            thinned = thin(kmind);
            bic = independence_bic(&thinned);
            kmind += 1;
        }
        let _ = n_runs;
        let m_star = ((alpha + beta) * epss / alpha.max(beta)).ln() / (1.0 - alpha - beta).abs().ln();
        let m_burn = fix((m_star + 1.0) * k_thin as f64);
        let n_star = (2.0 - (alpha + beta)) * alpha * beta * phi * phi / ((alpha + beta).powi(3) * r * r);
        let n_prec = fix(n_star + 1.0) * k_thin as f64;
        let i_stat = (m_burn + n_prec) / out.n_min;
        out.m_burn.push(m_burn);
        out.n_prec.push(n_prec);
        out.i_stat.push(i_stat);
        out.k_ind.push(fix(i_stat + 1.0).max(kmind as f64));
        out.k_thin.push(k_thin as f64);
        out.n_total.push(m_burn + n_prec);
    }
    out
}

/// The Raftery–Lewis table of `mcmc_diagnostics.m` for chain `b` (1-based).
pub fn print_raftery_lewis(rl: &RafteryLewis, names: &[String], qrs: [f64; 3], b: usize) {
    let title = format!(
        "Raftery/Lewis (1992) Convergence Diagnostics, based on quantile q={:.3} with precision r={:.3} with probability s={:.3} for chain {b}.",
        qrs[0], qrs[1], qrs[2]
    );
    let headers: Vec<String> = ["Variables", "M (burn-in)", "N (req. draws)", "N+M (total draws)", "k (thinning)"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut rows: Vec<Vec<f64>> =
        (0..rl.m_burn.len()).map(|i| vec![rl.m_burn[i], rl.n_prec[i], rl.n_total[i], rl.k_thin[i]]).collect();
    let max: Vec<f64> = (0..4).map(|j| rows.iter().map(|r| r[j]).fold(f64::NEG_INFINITY, f64::max)).collect();
    rows.push(max);
    let mut labels = names.to_vec();
    labels.push("Maximum".into());
    let lh = labels.iter().map(|s| s.len()).max().unwrap_or(0) + 2;
    dyntable(&title, &headers, &labels, &rows, lh, 10, 0);
}

/// Raftery–Lewis for every chain (`chains[b].draw(k)`, all draws stored), on the last
/// `opts.last_file_lines` draws as Dynare; prints the tables, returns them.
pub fn raftery_lewis_chains(
    chains: &[super::mcmc::Chain],
    names: &[String],
    opts: &ConvergenceOptions,
) -> Vec<RafteryLewis> {
    let [q, r, s] = opts.raftery_lewis_qrs;
    chains
        .iter()
        .enumerate()
        .map(|(b, c)| {
            let n = c.total_draws;
            let from = opts.last_file_lines.map_or(c.first_stored, |l| n.saturating_sub(l).max(c.first_stored));
            let runs: Vec<Vec<f64>> = (0..c.npar).map(|j| (from..n).map(|t| c.draw(t)[j]).collect()).collect();
            let rl = raftery_lewis(&runs, q, r, s);
            print_raftery_lewis(&rl, names, opts.raftery_lewis_qrs, b + 1);
            rl
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_r8() {
        // R-8: quantile(1:10, 0.5) = 5.5, quantile(1:10, 0.025) = 1 (lower limit)
        let x: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(matlab_quantile(&x, 0.025), 1.0);
        assert!((matlab_quantile(&x, 0.5) - 5.5).abs() < 1e-15);
    }
}
