// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `optimization/cmaes.m` (N. Hansen's CMA-ES, version 3.x as shipped
//! with Dynare 7.2), `mode_compute = 9` of `dynare_minimize_objective.m`.
//!
//! The path ported is the one Dynare's options exercise: no restarts, no noise
//! handling, no active CMA, full covariance matrix (`DiagonalOnly` is 0 below
//! 1000 parameters), `StopOnWarnings = 'yes'` (warnings stop the run),
//! box constraints handled by `cmaes.m`'s adaptive quadratic boundary penalty
//! (`bnd.*`), objective values that are NaN resampled, `+inf` values kept (the
//! raw `dsge_likelihood` is minimized, as in Dynare). File logging, plotting,
//! `signals.par`, resume and the variables file are not ported.
//!
//! Random numbers come from a ChaCha8 stream (`seed`, `stream`), not MATLAB's
//! `randn`, so runs are reproducible but not MATLAB's draws. Each generation's
//! `lambda` candidates are drawn first (column by column, as `randn(N,lambda)`
//! with `EvalParallel`), then evaluated — in parallel with rayon when
//! `parallel` is set — so the result does not depend on `parallel`.

use anyhow::{Result, bail};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

use super::OptimResult;
use super::minimize::{kv_num, kv_str};
use crate::estimation::mcmc::rng_stream;
use crate::estimation::{Bounds, Objective};
use crate::linalg::{Mat, sym_eig};
use crate::modfile::OptValue;

/// Options of `cmaes.m` as Dynare sets them (`options_.cmaes` in
/// `default_option_values.m` over `cmaes.m`'s own `defopts`). `None` fields
/// take `cmaes.m`'s formula defaults, evaluated at run time.
#[derive(Clone, Debug)]
pub struct CmaesOptions {
    /// `MaxIter` (None = `1e3*(N+5)^2/sqrt(popsize)`).
    pub max_iter: Option<f64>,
    /// `MaxFunEvals` (Inf).
    pub max_fun_evals: f64,
    /// `TolFun` (Dynare: 1e-7).
    pub tol_fun: f64,
    /// `TolX` (Dynare: 1e-7).
    pub tol_x: f64,
    /// `TolUpX` (None = `1e3*max(insigma)`).
    pub tol_up_x: Option<f64>,
    /// `TolHistFun` (1e-11).
    pub tol_hist_fun: f64,
    /// `StopFitness` (-Inf).
    pub stop_fitness: f64,
    /// `StopOnStagnation` ('on').
    pub stop_on_stagnation: bool,
    /// `PopSize` (None = `4 + floor(3*log(N))`).
    pub pop_size: Option<usize>,
    /// `ParentNumber` (None = `floor(popsize/2)`).
    pub parent_number: Option<usize>,
    /// `RecombinationWeights`: "superlinear" (default), "linear" or "equal".
    pub recombination_weights: String,
    /// `DiagMaxChange`/`DiffMinChange` (Inf / 0, scalars).
    pub diff_max_change: f64,
    pub diff_min_change: f64,
    /// `DispFinal`/`DispModulo` ('on' / '10' in Dynare): printing.
    pub verbose: bool,
    pub disp_modulo: usize,
    pub seed: u64,
    pub stream: u64,
    /// Evaluate each generation's candidates in parallel (same result).
    pub parallel: bool,
}

impl Default for CmaesOptions {
    fn default() -> Self {
        CmaesOptions {
            max_iter: None,
            max_fun_evals: f64::INFINITY,
            tol_fun: 1e-7,
            tol_x: 1e-7,
            tol_up_x: None,
            tol_hist_fun: 1e-11,
            stop_fitness: f64::NEG_INFINITY,
            stop_on_stagnation: true,
            pop_size: None,
            parent_number: None,
            recombination_weights: "superlinear".into(),
            diff_max_change: f64::INFINITY,
            diff_min_change: 0.0,
            verbose: true,
            disp_modulo: 10,
            seed: 0,
            stream: 9,
            parallel: true,
        }
    }
}

impl CmaesOptions {
    /// The `optim` keys of `dynare_minimize_objective.m`, case 9.
    pub fn apply_optim(&mut self, kv: &[(String, OptValue)]) -> Result<()> {
        for (k, v) in kv {
            match k.as_str() {
                "MaxIter" => self.max_iter = Some(kv_num(k, v)?),
                "TolFun" => self.tol_fun = kv_num(k, v)?,
                "TolX" => self.tol_x = kv_num(k, v)?,
                "MaxFunEvals" => self.max_fun_evals = kv_num(k, v)?,
                "verbosity" => {
                    if kv_num(k, v)? == 0.0 {
                        self.verbose = false;
                        self.disp_modulo = 0;
                    }
                }
                "SaveFiles" => {
                    let _ = kv_str(k, v);
                }
                "CMAESResume" => {
                    if kv_num(k, v)? == 1.0 {
                        bail!("cmaes: CMAESResume is not supported by dynare-rust");
                    }
                }
                other => println!("WARNING: cmaes: Unknown option ({other})!"),
            }
        }
        Ok(())
    }
}

/// `H0` of `dynare_minimize_objective.m`, case 9: `(ub-lb)*0.2`, non-finite
/// entries 0.01, the largest entries shrunk by 0.9 while `max/min > 1e6`.
pub fn cmaes_insigma(bounds: &Bounds) -> Vec<f64> {
    let mut h: Vec<f64> = bounds
        .lb
        .iter()
        .zip(&bounds.ub)
        .map(|(l, u)| {
            let v = (u - l) * 0.2;
            if v.is_finite() { v } else { 0.01 }
        })
        .collect();
    if h.is_empty() {
        return h;
    }
    loop {
        let mx = h.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mn = h.iter().cloned().fold(f64::INFINITY, f64::min);
        if mx / mn <= 1e6 {
            break;
        }
        for v in h.iter_mut() {
            if *v == mx {
                *v *= 0.9;
            }
        }
    }
    h
}

/// MATLAB `max`/`min` ignoring NaN (NaN if all NaN).
fn nanmax(x: impl Iterator<Item = f64>) -> f64 {
    x.filter(|v| !v.is_nan()).fold(f64::NAN, |a, b| if a.is_nan() || b > a { b } else { a })
}
fn nanmin(x: impl Iterator<Item = f64>) -> f64 {
    x.filter(|v| !v.is_nan()).fold(f64::NAN, |a, b| if a.is_nan() || b < a { b } else { a })
}
/// `myrange`: `max(x) - min(x)` (NaN-ignoring).
fn myrange(x: impl Iterator<Item = f64> + Clone) -> f64 {
    nanmax(x.clone()) - nanmin(x)
}

fn cmp_nan_last(a: &f64, b: &f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        _ => a.partial_cmp(b).unwrap(),
    }
}

/// MATLAB `[s, idx] = sort(x)` (stable, ascending, NaN last).
fn sort_idx(x: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| cmp_nan_last(&x[a], &x[b]));
    (idx.iter().map(|&i| x[i]).collect(), idx)
}

/// MATLAB `median` (of a vector without NaN).
fn median(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let mut s = x.to_vec();
    s.sort_by(cmp_nan_last);
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// `myprctile(inar, perc)`.
fn myprctile(inar: &[f64], perc: &[f64]) -> Vec<f64> {
    let n = inar.len();
    let mut sar = inar.to_vec();
    sar.sort_by(cmp_nan_last);
    let nf = n as f64;
    perc.iter()
        .map(|&p| {
            if p <= 100.0 * (0.5 / nf) {
                sar[0]
            } else if p >= 100.0 * ((nf - 0.5) / nf) {
                sar[n - 1]
            } else {
                let avail = |i: usize| 100.0 * ((i + 1) as f64 - 0.5) / nf;
                let i = (0..n).filter(|&i| p > avail(i)).max().unwrap();
                sar[i] + (sar[i + 1] - sar[i]) * (p - avail(i)) / (avail(i + 1) - avail(i))
            }
        })
        .collect()
}

/// `xintobounds`: clipped point and `ti` (+1 above ub, -1 below lb, 0 inside).
fn xintobounds(x: &[f64], lb: &[f64], ub: &[f64]) -> (Vec<f64>, Vec<i32>) {
    let mut out = x.to_vec();
    let mut ti = vec![0; x.len()];
    for i in 0..x.len() {
        let below = out[i] < lb[i];
        if below {
            out[i] = lb[i];
        }
        let above = out[i] > ub[i];
        if above {
            out[i] = ub[i];
        }
        ti[i] = above as i32 - below as i32;
    }
    (out, ti)
}

/// Port of `cmaes.m` (see the module documentation). Returns `BESTEVER`
/// (`x`, `fval`), `iterations = countiter`, `func_count = out.evals + 1`,
/// `exitflag` 1 (tolfun/tolx), 2 (warnconditioncov) or -1, and the stop flags
/// joined by spaces as `message`.
pub fn cmaes(
    obj: &dyn Objective,
    x0: &[f64],
    insigma: &[f64],
    bounds: &Bounds,
    opts: &CmaesOptions,
) -> Result<OptimResult> {
    let n = x0.len();
    let nf = n as f64;
    if insigma.len() != n && insigma.len() != 1 {
        bail!("cmaes: input parameter SIGMA should be a scalar or a vector of size {n}");
    }
    let insigma: Vec<f64> = if insigma.len() == 1 { vec![insigma[0]; n] } else { insigma.to_vec() };
    if insigma.iter().any(|&s| s <= 0.0) {
        bail!("cmaes: Initial search volume (SIGMA) must be greater than zero");
    }
    let max_insigma = insigma.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_insigma = insigma.iter().cloned().fold(f64::INFINITY, f64::min);
    if max_insigma / min_insigma > 1e6 {
        bail!("cmaes: Initial search volume (SIGMA) badly conditioned");
    }
    let (lb, ub) = (&bounds.lb, &bounds.ub);
    if lb.iter().zip(ub).any(|(l, u)| l > u) {
        bail!("cmaes: lower bound found to be greater than upper bound");
    }
    let f = |x: &[f64]| obj.eval(x).0;
    let mut rng: ChaCha8Rng = rng_stream(opts.seed, opts.stream);

    let lambda = opts.pop_size.unwrap_or(4 + (3.0 * nf.ln()).floor() as usize);
    let stop_max_iter = opts.max_iter.unwrap_or(1e3 * (nf + 5.0).powi(2) / (lambda as f64).sqrt());
    let stop_tol_x = opts.tol_x;
    let stop_tol_up_x = opts.tol_up_x.unwrap_or(1e3 * max_insigma);
    let stop_on_equal_function_values = 2.0 + nf / 3.0;
    let mut arr_equal_funvals = vec![0.0f64; 10 + n];

    // Initialization
    let mut maxdx = vec![opts.diff_max_change; n];
    let mindx = vec![opts.diff_min_change; n];
    let mut sigma = max_insigma;
    let mut pc = vec![0.0; n];
    let mut ps = vec![0.0; n];
    let mut diag_d: Vec<f64> = insigma.iter().map(|s| s / max_insigma).collect();
    let mut diag_c: Vec<f64> = diag_d.iter().map(|d| d * d).collect();
    let mut b = Mat::identity(n, n);
    let mut bd = Mat::from_fn(n, n, |i, j| b[(i, j)] * diag_d[j]);
    let mut c = Mat::from_diagonal(&nalgebra::DVector::from_vec(diag_c.clone()));

    let hist_len = 10 + (30.0 * nf / lambda as f64).ceil() as usize;
    let mut hist = vec![f64::NAN; hist_len];
    let mut histsel = vec![f64::NAN; hist_len];
    let mut histbest: Vec<f64> = vec![];
    let mut histmedian: Vec<f64> = vec![];

    // Boundary handling
    let bnd_active = lb.iter().any(|v| *v > f64::NEG_INFINITY) || ub.iter().any(|v| *v < f64::INFINITY);
    let mut xmean = x0.to_vec();
    let mut bnd_weights = vec![0.0; n];
    let bnd_scale = vec![1.0; n];
    let isbounded: Vec<bool> = (0..n).map(|i| lb[i] > f64::NEG_INFINITY || ub[i] < f64::INFINITY).collect();
    let mut dfithist: Vec<f64> = vec![1.0];
    let mut validfitval = false;
    let mut iniphase = true;
    if bnd_active {
        let (xm, ti) = xintobounds(&xmean, lb, ub);
        if ti.iter().any(|&t| t != 0) && opts.verbose {
            println!("WARNING: cmaes: Initial point was out of bounds, corrected");
        }
        xmean = xm;
        for i in 0..n {
            maxdx[i] = maxdx[i].min((ub[i] - lb[i]) / 2.0);
        }
        if (0..n).any(|i| sigma * diag_c[i].sqrt() > maxdx[i]) {
            sigma = (0..n).map(|i| maxdx[i] / diag_c[i].sqrt()).fold(f64::INFINITY, f64::min);
        }
    }

    // Initial evaluation (EvalInitialX = 'yes').
    let mut counteval: usize = 0;
    let mut best_x = xmean.clone();
    let mut best_f = f64::INFINITY;
    hist[0] = f(&xmean);
    histsel[0] = hist[0];
    counteval += 1;
    if hist[0] < best_f {
        best_x = xmean.clone();
        best_f = hist[0];
    }

    let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

    // Strategy parameters (set once: lambda never changes without restarts).
    let mu = opts.parent_number.unwrap_or(lambda / 2);
    let mut weights: Vec<f64> = match opts.recombination_weights.to_lowercase().get(..3) {
        Some("equ") => vec![1.0; mu],
        Some("lin") => (1..=mu).map(|i| mu as f64 + 0.5 - i as f64).collect(),
        Some("sup") => (1..=mu).map(|i| (mu as f64 + 0.5).ln() - (i as f64).ln()).collect(),
        _ => bail!("cmaes: Recombination weights to be \"{}\" is not implemented", opts.recombination_weights),
    };
    let sw: f64 = weights.iter().sum();
    let mueff = sw * sw / weights.iter().map(|w| w * w).sum::<f64>();
    for w in weights.iter_mut() {
        *w /= sw;
    }
    if mueff == lambda as f64 {
        bail!("cmaes: Combination of values for PopSize, ParentNumber and RecombinationWeights is not reasonable");
    }
    let cc = (4.0 + mueff / nf) / (nf + 4.0 + 2.0 * mueff / nf);
    let cs = (mueff + 2.0) / (nf + mueff + 3.0);
    let ccov1 = 2.0 / ((nf + 1.3).powi(2) + mueff);
    let ccovmu = (1.0 - ccov1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((nf + 2.0).powi(2) + mueff));
    let damps = 1.0 + 2.0 * (((mueff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + cs;
    if opts.verbose {
        // cmaes.m: the weights as `strw`, then `mu_eff` and the objective's name
        use crate::analysis::table::fmt_g;
        let w100: Vec<f64> = weights.iter().map(|w| 100.0 * w).collect();
        let strw = if mu == 1 {
            "100".to_string()
        } else if mu < 8 {
            let mut s = format!("{:.0}", w100[0]);
            for w in &w100[1..] {
                s += &format!(" {w:.0}");
            }
            s
        } else {
            let g = |x: f64| fmt_g(x, 0, 2);
            format!(
                "{} {} {}... {} {}]%, ",
                g(w100[0]),
                g(w100[1]),
                g(w100[2]),
                g(w100[w100.len() - 2]),
                g(w100[w100.len() - 1])
            )
        };
        println!("  n={n}: ({mu},{lambda})-CMA-ES(w=[{strw}]%, mu_eff={mueff:.1}) on function dsge_likelihood");
    }

    let mut countiter: usize = 0;
    let mut stopflag: Vec<&'static str> = vec![];
    let mut xold = xmean.clone();
    let mut arz = vec![vec![0.0; n]; lambda];
    let mut arx = vec![vec![0.0; n]; lambda];
    let mut arxvalid = vec![vec![0.0; n]; lambda];
    let mut out_evals = counteval;
    let disp_modulo = if opts.verbose { opts.disp_modulo } else { 0 };

    let sample = |rng: &mut ChaCha8Rng, xmean: &[f64], bd: &Mat, sigma: f64, z: &mut [f64], x: &mut [f64]| {
        for zi in z.iter_mut() {
            *zi = rng.sample(StandardNormal);
        }
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += bd[(i, j)] * z[j];
            }
            x[i] = xmean[i] + sigma * s;
        }
    };
    let clip = |x: &[f64]| if bnd_active { xintobounds(x, lb, ub).0 } else { x.to_vec() };

    while stopflag.is_empty() {
        countiter += 1;
        // Sample and evaluate (Eq. 1).
        for k in 0..lambda {
            sample(&mut rng, &xmean, &bd, sigma, &mut arz[k], &mut arx[k]);
            arxvalid[k] = clip(&arx[k]);
        }
        let mut raw: Vec<f64> = if opts.parallel {
            arxvalid.par_iter().map(|x| f(x)).collect()
        } else {
            arxvalid.iter().map(|x| f(x)).collect()
        };
        counteval += raw.iter().filter(|v| !v.is_nan()).count();
        for k in 0..lambda {
            if !raw[k].is_nan() {
                continue;
            }
            let mut tries = 0;
            while raw[k].is_nan() {
                sample(&mut rng, &xmean, &bd, sigma, &mut arz[k], &mut arx[k]);
                arxvalid[k] = clip(&arx[k]);
                raw[k] = f(&arxvalid[k]);
                tries += 1;
                if tries % 100 == 0 && opts.verbose {
                    println!("WARNING: cmaes: {tries} NaN objective function values at evaluation {counteval}");
                }
            }
            counteval += 1;
        }

        let mut sel = raw.clone();
        if bnd_active {
            let val = myprctile(&raw, &[25.0, 75.0]);
            let mean_diag_c = diag_c.iter().sum::<f64>() / nf;
            let mut val = Some((val[1] - val[0]) / nf / mean_diag_c / (sigma * sigma));
            let v = val.unwrap();
            if !v.is_finite() {
                val = Some(nanmax(dfithist.iter().cloned()));
            } else if v == 0.0 {
                let pos: Vec<f64> = dfithist.iter().cloned().filter(|&d| d > 0.0).collect();
                val = if pos.is_empty() { None } else { Some(nanmin(pos.into_iter())) };
            } else if !validfitval {
                dfithist.clear();
                validfitval = true;
            }
            if (dfithist.len() as f64) < 20.0 + (3.0 * nf) / lambda as f64 {
                dfithist.extend(val);
            } else {
                dfithist.remove(0);
                dfithist.extend(val);
            }
            let (tx, ti) = xintobounds(&xmean, lb, ub);
            let any_ti = ti.iter().any(|&t| t != 0);
            if iniphase && any_ti {
                let w0 = 2.0002 * median(&dfithist);
                let idx: Vec<usize> = (0..n).filter(|&i| isbounded[i]).collect();
                let mean_dd = diag_c.iter().sum::<f64>() / nf;
                for &i in &idx {
                    bnd_weights[i] = w0 / (diag_c[i] / mean_dd);
                }
                if validfitval && countiter > 2 {
                    iniphase = false;
                }
            }
            if any_ti {
                let fac = 1.2f64.powf((mueff / 10.0 / nf).min(1.0));
                let lim = 3.0 * 1f64.max(nf.sqrt() / mueff) * sigma;
                for i in 0..n {
                    let d = xmean[i] - tx[i];
                    let sgn = |v: f64| {
                        if v > 0.0 {
                            1
                        } else if v < 0.0 {
                            -1
                        } else {
                            0
                        }
                    };
                    if ti[i] != 0 && d.abs() > lim * diag_c[i].sqrt() && sgn(d) == sgn(xmean[i] - xold[i]) {
                        bnd_weights[i] *= fac;
                    }
                }
            }
            for k in 0..lambda {
                let pen: f64 =
                    (0..n).map(|i| bnd_weights[i] / bnd_scale[i] * (arxvalid[k][i] - arx[k][i]).powi(2)).sum();
                sel[k] = raw[k] + pen;
            }
        }

        let (rs, ri) = sort_idx(&raw);
        let (ss, si) = sort_idx(&sel);
        hist.rotate_right(1);
        hist[0] = rs[0];
        let med = median(&rs);
        if (histbest.len() as f64) < 120.0 + (30.0 * nf / lambda as f64).ceil()
            || (countiter.is_multiple_of(5) && histbest.len() < 20000)
        {
            histbest.insert(0, rs[0]);
            histmedian.insert(0, med);
        } else {
            histbest.rotate_right(1);
            histmedian.rotate_right(1);
            histbest[0] = rs[0];
            histmedian[0] = med;
        }
        histsel.rotate_right(1);
        histsel[0] = ss[0];

        // Recombination (Eq. 2-5).
        xold.copy_from_slice(&xmean);
        let mut zmean = vec![0.0; n];
        for i in 0..n {
            xmean[i] = (0..mu).map(|k| arx[si[k]][i] * weights[k]).sum();
            zmean[i] = (0..mu).map(|k| arz[si[k]][i] * weights[k]).sum();
        }
        let csf = (cs * (2.0 - cs) * mueff).sqrt();
        for i in 0..n {
            let bz: f64 = (0..n).map(|j| b[(i, j)] * zmean[j]).sum();
            ps[i] = (1.0 - cs) * ps[i] + csf * bz;
        }
        let ps_norm = ps.iter().map(|v| v * v).sum::<f64>().sqrt();
        let hsig = ps_norm / (1.0 - (1.0 - cs).powf(2.0 * countiter as f64)).sqrt() / chi_n < 1.4 + 2.0 / (nf + 1.0);
        let hs = hsig as i32 as f64;
        let ccf = hs * (cc * (2.0 - cc) * mueff).sqrt() / sigma;
        for i in 0..n {
            pc[i] = (1.0 - cc) * pc[i] + ccf * (xmean[i] - xold[i]);
        }
        if ccov1 + ccovmu > 0.0 {
            let arpos: Vec<Vec<f64>> =
                (0..mu).map(|k| (0..n).map(|i| (arx[si[k]][i] - xold[i]) / sigma).collect()).collect();
            let a = 1.0 - ccov1 - ccovmu + (1.0 - hs) * ccov1 * cc * (2.0 - cc);
            for j in 0..n {
                for i in 0..n {
                    let mut rmu = 0.0;
                    for k in 0..mu {
                        rmu += arpos[k][i] * weights[k] * arpos[k][j];
                    }
                    c[(i, j)] = a * c[(i, j)] + ccov1 * pc[i] * pc[j] + ccovmu * rmu;
                }
            }
            for i in 0..n {
                diag_c[i] = c[(i, i)];
            }
        }
        sigma *= ((ps_norm / chi_n - 1.0) * cs / damps).min(1.0).exp();

        // Eigendecomposition every 1/(ccov1+ccovmu)/N/10 generations.
        let ccsum = ccov1 + ccovmu;
        if ccsum > 0.0 {
            let period = 1.0 / ccsum / nf / 10.0;
            let m = countiter as f64 % period;
            if m < 1.0 {
                for j in 0..n {
                    for i in j + 1..n {
                        c[(i, j)] = c[(j, i)];
                    }
                }
                let (w, v) = sym_eig(&c);
                let mut dd: Vec<f64> = w.iter().cloned().collect();
                if dd.iter().any(|v| !v.is_finite()) || v.iter().any(|v| !v.is_finite()) {
                    bail!("cmaes: function eig returned non-finite eigenvalues or eigenvectors");
                }
                b = v;
                let mn = dd.iter().cloned().fold(f64::INFINITY, f64::min);
                let mx = dd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if mn <= 0.0 {
                    stopflag.push("warnconditioncov");
                }
                if mx > 1e14 * mn {
                    stopflag.push("warnconditioncov");
                }
                for i in 0..n {
                    diag_c[i] = c[(i, i)];
                }
                for d in dd.iter_mut() {
                    *d = d.sqrt();
                }
                diag_d = dd;
                bd = Mat::from_fn(n, n, |i, j| b[(i, j)] * diag_d[j]);
            }
        }
        // Numerical stability: sigma vs D.
        let max_d = diag_d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if sigma > 1e10 * max_d {
            let fac = sigma / max_d;
            sigma /= fac;
            for v in pc.iter_mut() {
                *v *= fac;
            }
            for v in diag_d.iter_mut() {
                *v *= fac;
            }
            c *= fac * fac;
            bd = Mat::from_fn(n, n, |i, j| b[(i, j)] * diag_d[j]);
            for v in diag_c.iter_mut() {
                *v *= fac * fac;
            }
        }
        // Step-size limits.
        if (0..n).any(|i| sigma * diag_c[i].sqrt() > maxdx[i]) {
            sigma = (0..n).map(|i| maxdx[i] / diag_c[i].sqrt()).fold(f64::INFINITY, f64::min);
        }
        if (0..n).any(|i| sigma * diag_c[i].sqrt() < mindx[i]) {
            sigma = (0..n).map(|i| mindx[i] / diag_c[i].sqrt()).fold(f64::NEG_INFINITY, f64::max)
                * (0.05 + cs / damps).exp();
        }
        // StopOnWarnings = 'yes': warnings stop.
        if (0..n).any(|i| xmean[i] == xmean[i] + 0.2 * sigma * diag_c[i].sqrt()) {
            stopflag.push("warnnoeffectcoord");
        }
        let col = countiter % n;
        if (0..n).all(|i| xmean[i] == xmean[i] + 0.1 * sigma * bd[(i, col)]) {
            stopflag.push("warnnoeffectaxis");
        }
        let jeq = (0.1 + lambda as f64 / 4.0).ceil() as usize; // 1-based 1+ceil(..) -> 0-based ceil(..)
        if ss[0] == ss[jeq] && stop_on_equal_function_values != 0.0 {
            arr_equal_funvals.rotate_right(1);
            arr_equal_funvals[0] = countiter as f64;
            let last = *arr_equal_funvals.last().unwrap();
            if last > countiter as f64 - 3.0 * arr_equal_funvals.len() as f64 {
                stopflag.push("equalfunvals");
            }
        }
        if countiter > 2 && myrange(hist.iter().cloned().chain(std::iter::once(ss[0]))) == 0.0 {
            stopflag.push("warnequalfunvalhist");
        }

        out_evals = counteval;
        if hist[0] < best_f {
            best_f = hist[0];
            best_x = arxvalid[ri[0]].clone();
        }

        // Termination.
        if rs[0] <= opts.stop_fitness {
            stopflag.push("fitness");
        }
        if counteval as f64 >= opts.max_fun_evals {
            stopflag.push("maxfunevals");
        }
        if countiter as f64 >= stop_max_iter {
            stopflag.push("maxiter");
        }
        if (0..n).all(|i| sigma * pc[i].abs().max(diag_c[i].sqrt()) < stop_tol_x) {
            stopflag.push("tolx");
        }
        if (0..n).any(|i| sigma * diag_c[i].sqrt() > stop_tol_up_x) {
            stopflag.push("tolupx");
        }
        if sigma * diag_d.iter().cloned().fold(f64::NEG_INFINITY, f64::max) == 0.0 {
            stopflag.push("bug");
        }
        if countiter > 2 && myrange(ss.iter().cloned().chain(hist.iter().cloned())) <= opts.tol_fun {
            stopflag.push("tolfun");
        }
        if countiter >= hist.len() && myrange(hist.iter().cloned()) <= opts.tol_hist_fun {
            stopflag.push("tolhistfun");
        }
        let l = histbest.len() / 3;
        if opts.stop_on_stagnation
            && countiter as f64 > nf * (5.0 + 100.0 / lambda as f64)
            && histbest.len() > 100
            && median(&histmedian[..l]) >= median(&histmedian[histmedian.len() - l - 1..])
            && median(&histbest[..l]) >= median(&histbest[histbest.len() - l - 1..])
        {
            stopflag.push("stagnation");
        }

        if disp_modulo > 0 {
            if countiter == 1 || (countiter as f64 % (10.0 * disp_modulo as f64)) < 1.0 {
                println!("Iterat, #Fevals:   Function Value    (median,worst) |Axis Ratio|idx:Min SD idx:Max SD");
            }
            if countiter.is_multiple_of(disp_modulo) || countiter < 3 || !stopflag.is_empty() {
                let sds: Vec<f64> = diag_c.iter().map(|v| sigma * v.sqrt()).collect();
                let (mut mi, mut ma) = (0, 0);
                for i in 0..n {
                    if sds[i] < sds[mi] {
                        mi = i;
                    }
                    if sds[i] > sds[ma] {
                        ma = i;
                    }
                }
                let dmax = diag_d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let dmin = diag_d.iter().cloned().fold(f64::INFINITY, f64::min);
                // num2str(x, '%.Ne'): C exponent (two digits), Inf and NaN as MATLAB prints them
                let e = |x: f64, p: usize| crate::analysis::table::fmt_e(x, 0, p);
                println!(
                    "{countiter:>5} , {counteval:>6} : {} +({},{}) | {} | {:>2}:{} {:>2}:{}",
                    e(hist[0], 13),
                    e(med - hist[0], 0),
                    e(rs[lambda - 1] - hist[0], 0),
                    e(dmax / dmin, 2),
                    mi + 1,
                    e(sds[mi], 1),
                    ma + 1,
                    e(sds[ma], 1)
                );
            }
        }
    }
    let _ = &histsel;

    // Final evaluation of the (clipped) mean.
    let xm = clip(&xmean);
    let fm = f(&xm);
    if fm < best_f {
        best_f = fm;
        best_x = xm;
    }
    let exitflag = if stopflag.iter().any(|s| *s == "tolfun" || *s == "tolx") {
        1
    } else if stopflag.iter().all(|s| *s == "warnconditioncov") {
        2
    } else {
        -1
    };
    let message = stopflag.join(" ");
    if opts.verbose {
        println!("#Fevals:   f(returned x)   |    bestever.f     | stopflag");
        let e = |x: f64, p: usize| crate::analysis::table::fmt_e(x, 0, p);
        let ce = counteval + 1;
        let pad = (6 - (ce as f64).log10().floor() as i64).max(0) as usize;
        println!("{:pad$}{ce}: {} | {} | {}", "", e(best_f, 11), e(best_f, 11), stopflag.join(""));
        if n < 102 {
            let sig = |x: f64| format!(" {}{}", if x < 0.0 { "-" } else { "+" }, e(x.abs(), 1));
            println!("mean solution:{}", xmean.iter().map(|&x| sig(x)).collect::<String>());
            println!(
                "std deviation:{}",
                diag_c.iter().map(|v| format!("  {}", e(sigma * v.sqrt(), 1))).collect::<String>()
            );
            println!("use plotcmaesdat.m for plotting the output at any time (option LogModulo must not be zero)");
        }
    }
    Ok(OptimResult {
        x: best_x,
        fval: best_f,
        hessian: None,
        inv_hessian: None,
        iterations: countiter,
        func_count: out_evals + 1,
        exitflag,
        message,
    })
}
