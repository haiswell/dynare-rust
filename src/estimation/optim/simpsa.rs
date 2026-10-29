// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `optimization/simpsa.m` (Cardoso, Salcedo and Feyo de Azevedo 1996:
//! Nelder–Mead simplex with simulated-annealing fluctuations), Dynare's
//! `mode_compute = 10`, with the defaults of `options_.simpsa` and the `optim`
//! keys of `dynare_minimize_objective.m` (case 10).
//!
//! The algorithm is transcribed as it is, including two quirks of the MATLAB
//! code that shape its behaviour: the "multiple contraction" re-evaluates
//! vertices `2..NDIM` only (not the last one), and the cooling schedule uses the
//! standard deviation of a `COSTS` vector that holds the final simplex costs
//! once and zeros elsewhere.
//!
//! The random numbers come from a ChaCha8 stream (`seed`, `stream`), drawn in
//! the order of the MATLAB code. The vertices of a new initial simplex and of a
//! multiple contraction are evaluated in parallel; the best point is then
//! updated in the sequential order, so the result does not depend on it.

use std::time::Instant;

use anyhow::Result;
use rand::Rng;
use rayon::prelude::*;

use super::OptimResult;
use super::minimize::{kv_num, unquote};
use crate::estimation::Objective;
use crate::estimation::mcmc::rng_stream;
use crate::modfile::OptValue;

/// `options_.simpsa` (`default_option_values.m`), on top of simpsa's own defaults.
#[derive(Clone, Debug)]
pub struct SimpsaOptions {
    /// `TEMP_START` (None: estimated during the first temperature loop).
    pub temp_start: Option<f64>,
    pub temp_end: f64,
    pub cool_rate: f64,
    pub initial_acceptance_ratio: f64,
    pub min_cooling_factor: f64,
    pub max_iter_temp_first: usize,
    pub max_iter_temp_last: usize,
    pub max_iter_temp: usize,
    pub max_iter_total: usize,
    /// Minutes.
    pub max_time: f64,
    pub max_fun_evals: usize,
    /// None: simpsa's default (1e-6), as Dynare's `TolX < 0`.
    pub tolx: Option<f64>,
    pub tolfun: f64,
    /// `DISPLAY = 'iter'`.
    pub display: bool,
    pub seed: u64,
    pub stream: u64,
    /// Evaluate the vertices of new simplices in parallel (same result).
    pub parallel: bool,
}

impl Default for SimpsaOptions {
    fn default() -> Self {
        SimpsaOptions {
            temp_start: None,
            temp_end: 0.1,
            cool_rate: 10.0,
            initial_acceptance_ratio: 0.95,
            min_cooling_factor: 0.9,
            max_iter_temp_first: 50,
            max_iter_temp_last: 2000,
            max_iter_temp: 10,
            max_iter_total: 5000,
            max_time: 2500.0,
            max_fun_evals: 20000,
            tolx: Some(1e-4),
            tolfun: 1e-4,
            display: true,
            seed: 0,
            stream: 0,
            parallel: true,
        }
    }
}

impl SimpsaOptions {
    /// The `optim` keys of `dynare_minimize_objective.m`, case 10.
    pub fn apply_optim(&mut self, kv: &[(String, OptValue)]) -> Result<()> {
        for (k, v) in kv {
            let x = || kv_num(k, v);
            match unquote(k).as_str() {
                "MaxIter" => self.max_iter_total = x()? as usize,
                "TolFun" => self.tolfun = x()?,
                "TolX" => {
                    let t = x()?;
                    self.tolx = if t < 0.0 { None } else { Some(t) };
                }
                "EndTemperature" => self.temp_end = x()?,
                "MaxFunEvals" => self.max_fun_evals = x()? as usize,
                "verbosity" => self.display = x()? != 0.0,
                other => println!("WARNING: simpsa: Unknown option ({other})!"),
            }
        }
        Ok(())
    }
}

/// `CALCULATE_COST` without its side effects: the penalty outside the bounds
/// (no evaluation), else the objective. Returns `(cost, evaluated)`.
fn cost(obj: &dyn Objective, p: &[f64], lb: &[f64], ub: &[f64]) -> (f64, bool) {
    for i in 0..p.len() {
        if p[i] < lb[i] {
            return (1e12 + (lb[i] - p[i]) * 1e6, false);
        }
        if p[i] > ub[i] {
            return (1e12 + (p[i] - ub[i]) * 1e6, false);
        }
    }
    (obj.eval(p).0, true)
}

struct Best {
    y: f64,
    p: Vec<f64>,
    nfun: usize,
}

impl Best {
    /// The side effects of `CALCULATE_COST`.
    fn record(&mut self, p: &[f64], y: f64, evaluated: bool) {
        if evaluated {
            self.nfun += 1;
            if y < self.y {
                self.y = y;
                self.p.copy_from_slice(p);
            }
        }
    }
}

/// Port of `simpsa.m` as `dynare_minimize_objective.m` calls it (raw objective,
/// finite bounds). `iterations` = `nITERATIONS`, `func_count` = `nFUN_EVALS`,
/// `exitflag` 1 (TOLFUN), 2 (TOLX), 0 (iterations/evaluations), -1 (time).
pub fn simpsa(obj: &dyn Objective, x0: &[f64], lb: &[f64], ub: &[f64], opts: &SimpsaOptions) -> Result<OptimResult> {
    let ndim = x0.len();
    anyhow::ensure!(lb.len() == ndim, "'LB' and 'X0' have incompatible dimensions in 'SIMPSA'");
    anyhow::ensure!(ub.len() == ndim, "'UB' and 'X0' have incompatible dimensions in 'SIMPSA'");
    let tolx = opts.tolx.unwrap_or(1e-6);
    let mut rng = rng_stream(opts.seed, opts.stream);
    let mut exitflag = -2;
    let mut message = String::from("something wrong");
    let nv = ndim + 1;
    // P: rows are vertices.
    let mut p: Vec<Vec<f64>> = vec![vec![0.0; ndim]; nv];
    let mut y = vec![0.0; nv];
    let (y0, _) = cost(obj, x0, lb, ub);
    let mut best = Best { y: y0, p: x0.to_vec(), nfun: 0 };
    let mut temp_loop_number = 1usize;
    let mut temp = opts.temp_start.unwrap_or(best.y.abs() * 1e5);
    let tic = Instant::now();
    best.nfun = 1;
    let mut n_iterations = 0usize;
    let (mut successful, mut unsuccessful, mut unsuccessful_costs) = (0usize, 0usize, 0.0f64);
    let eval_many = |pts: &[Vec<f64>]| -> Vec<(f64, bool)> {
        if opts.parallel {
            pts.par_iter().map(|q| cost(obj, q, lb, ub)).collect()
        } else {
            pts.iter().map(|q| cost(obj, q, lb, ub)).collect()
        }
    };
    let dprint = |it: usize, nf: usize, ymin: f64, yb: f64, t: f64, step: &str| {
        // dprintf('%5.0f      %5.0f       %12.6g     %15.6g      %12.6g       %s', ...)
        use crate::analysis::table::fmt_g;
        println!(
            "{it:5}      {nf:5}       {}     {}      {}       {step}",
            fmt_g(ymin, 12, 6),
            fmt_g(yb, 15, 6),
            fmt_g(t, 12, 6)
        );
    };
    loop {
        if exitflag != -2 {
            break;
        }
        let maxitertemp = if temp_loop_number == 1 {
            if opts.temp_start.is_none() {
                successful = 0;
                unsuccessful = 0;
                unsuccessful_costs = 0.0;
            }
            opts.max_iter_temp_first * ndim
        } else if temp < opts.temp_end {
            temp = 0.0;
            opts.max_iter_temp_last * ndim
        } else {
            opts.max_iter_temp * ndim
        };
        // New simplex around the best point.
        p[0].copy_from_slice(&best.p);
        let (c0, e0) = cost(obj, &p[0], lb, ub);
        y[0] = c0;
        best.record(&p[0].clone(), c0, e0);
        for k in 0..ndim {
            let mut q = p[0].clone();
            q[k] = lb[k] + rng.r#gen::<f64>() * (ub[k] - lb[k]);
            p[k + 1] = q;
        }
        let ys = eval_many(&p[1..]);
        for (k, (c, e)) in ys.into_iter().enumerate() {
            y[k + 1] = c;
            best.record(&p[k + 1], c, e);
        }
        let mut algostep = "initial simplex";
        if opts.display {
            if n_iterations == 0 {
                println!(
                    " Nr Iter  Nr Fun Eval    Min function       Best function        TEMP           Algorithm Step"
                );
            } else {
                dprint(n_iterations, best.nfun, y[0], best.y, temp, "best point");
            }
        }
        let mut costs = vec![0.0; nv * maxitertemp];
        let mut itertemp = 0usize;
        let mut broke = false;
        for it in 1..=maxitertemp {
            itertemp = it;
            n_iterations += 1;
            // Fluctuated costs, then sortrows on them (stable, NaN last).
            let yfluct0: Vec<f64> = (0..nv).map(|i| y[i] + temp * rng.r#gen::<f64>().ln().abs()).collect();
            let mut idx: Vec<usize> = (0..nv).collect();
            idx.sort_by(|&a, &b| sort_key(yfluct0[a]).total_cmp(&sort_key(yfluct0[b])));
            let yfluct: Vec<f64> = idx.iter().map(|&i| yfluct0[i]).collect();
            y = idx.iter().map(|&i| y[i]).collect();
            p = idx.iter().map(|&i| p[i].clone()).collect();
            if opts.display {
                dprint(n_iterations, best.nfun, y[0], best.y, temp, algostep);
            }
            let ymax = y.iter().cloned().fold(f64::NAN, f64::max);
            let ymin = y.iter().cloned().fold(f64::NAN, f64::min);
            if (ymax - ymin).abs() < opts.tolfun && temp_loop_number != 1 {
                message = "Change in the objective function value less than the specified tolerance (TOLFUN).".into();
                exitflag = 1;
            } else if temp_loop_number != 1 && {
                let mut m = f64::NAN;
                for r in 1..nv {
                    for c in 0..ndim {
                        m = m.max((p[r][c] - p[r - 1][c]).abs());
                    }
                }
                m < tolx
            } {
                message = "Change in X less than the specified tolerance (TOLX).".into();
                exitflag = 2;
            } else if n_iterations >= opts.max_iter_total * ndim || best.nfun >= opts.max_fun_evals * ndim * (ndim + 1)
            {
                message = "Maximum number of function evaluations (MAX_FUN_EVALS*NDIM*(NDIM+1)) or iterations (MAX_ITER_TOTAL*NDIM) reached.".into();
                exitflag = 0;
            } else if tic.elapsed().as_secs_f64() / 60.0 > opts.max_time {
                message = "Exceeded maximum time.".into();
                exitflag = -1;
            }
            if exitflag != -2 {
                if opts.display {
                    println!("{message}");
                }
                broke = true;
                break;
            }
            // AMOTRY(fac): psum*(1-fac) + P(end,:)*fac, psum the mean of the other vertices.
            let amotry = |p: &[Vec<f64>], fac: f64, best: &mut Best, rng: &mut rand_chacha::ChaCha8Rng| {
                let ptry: Vec<f64> = (0..ndim)
                    .map(|c| {
                        let psum = p[..ndim].iter().map(|r| r[c]).sum::<f64>() / ndim as f64;
                        psum * (1.0 - fac) + p[ndim][c] * fac
                    })
                    .collect();
                let (ytry, e) = cost(obj, &ptry, lb, ub);
                best.record(&ptry, ytry, e);
                let yftry = ytry - temp * rng.r#gen::<f64>().ln().abs();
                (yftry, ytry, ptry)
            };
            let (yftry, ytry, ptry) = amotry(&p, -1.0, &mut best, &mut rng);
            if yftry <= yfluct[0] {
                let (yfexp, yexp, pexp) = amotry(&p, -2.0, &mut best, &mut rng);
                if yfexp < yftry {
                    p[ndim] = pexp;
                    y[ndim] = yexp;
                    algostep = "reflection and expansion";
                } else {
                    p[ndim] = ptry;
                    y[ndim] = ytry;
                    algostep = "reflection";
                }
            } else if yftry >= yfluct[ndim - 1] {
                let (yfc, yc, pc) = amotry(&p, -0.5, &mut best, &mut rng);
                if yfc < yfluct[ndim] {
                    p[ndim] = pc;
                    y[ndim] = yc;
                    algostep = "one dimensional contraction";
                } else {
                    let p0 = p[0].clone();
                    for r in p.iter_mut().skip(1) {
                        for c in 0..ndim {
                            r[c] = 0.5 * (r[c] + p0[c]);
                        }
                    }
                    // for k=2:NDIM (MATLAB): vertices 1..ndim-1, 0-based.
                    if ndim >= 2 {
                        let ys = eval_many(&p[1..ndim]);
                        for (k, (c, e)) in ys.into_iter().enumerate() {
                            y[k + 1] = c;
                            best.record(&p[k + 1], c, e);
                        }
                    }
                    algostep = "multiple contraction";
                }
            } else {
                p[ndim] = ptry;
                y[ndim] = ytry;
                algostep = "reflection";
            }
            if temp_loop_number == 1 && opts.temp_start.is_none() {
                if y[0] > y[ndim] {
                    successful += 1;
                } else if y[0] <= y[ndim] {
                    unsuccessful += 1;
                    unsuccessful_costs += y[ndim] - y[0];
                }
            }
        }
        if broke && itertemp < maxitertemp {
            break;
        }
        // COSTS((ITERTEMP-1)*NDIM+1:ITERTEMP*NDIM+1) = Y
        if itertemp >= 1 {
            let start = (itertemp - 1) * ndim;
            costs[start..start + nv].copy_from_slice(&y);
        }
        if temp_loop_number == 1 && opts.temp_start.is_none() {
            let tot = (successful + unsuccessful) as f64;
            temp = -(unsuccessful_costs / tot)
                / ((tot * opts.initial_acceptance_ratio - successful as f64) / unsuccessful as f64).ln();
        } else {
            let stdev = std(&costs);
            let cooling = 1.0 / (1.0 + temp * (1.0 + opts.cool_rate).ln() / (3.0 * stdev));
            temp *= opts.min_cooling_factor.min(cooling);
        }
        temp_loop_number += 1;
    }
    Ok(OptimResult {
        x: best.p,
        fval: best.y,
        hessian: None,
        inv_hessian: None,
        iterations: n_iterations,
        func_count: best.nfun,
        exitflag,
        message,
    })
}

/// `sortrows` ascending order: NaN after everything.
fn sort_key(v: f64) -> f64 {
    if v.is_nan() { f64::INFINITY } else { v }
}

/// MATLAB `std` (n-1 normalization).
fn std(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    if v.len() < 2 {
        return 0.0;
    }
    let m = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)).sqrt()
}
