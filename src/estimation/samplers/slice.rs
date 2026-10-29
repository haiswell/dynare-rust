// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `estimation/slice_sampler.m` and `estimation/rotated_slice_sampler.m`
//! (Neal 2003 univariate slice sampling with stepping out and shrinkage, per
//! parameter or along rotated directions: the eigenvectors of the inverse
//! Hessian, or the directions between modes in the multimodal case). The
//! endogenous initial states (`estimate_initial_states_endogenous_prior`, default
//! `draw_init_state_from_smoother`) are handled through [`Objective::init_state_ops`]; the
//! `draw_init_state_with_rotated_slice` variant is not ported.
//!
//! Parallelism without changing the draws: the stepping-out points
//! `L, L-W, ...` and `R, R+W, ...` do not depend on the random stream, and the
//! shrinkage candidates along the path of successive rejections are fixed by
//! the stream (each rejection shrinks the interval on a known side); these
//! points are evaluated `width` at a time in parallel and consumed in the
//! sequential order. The current point's log posterior, which Dynare
//! re-evaluates at each coordinate, is reused (an evaluation is a pure
//! function of the parameters); Dynare's evaluation count is kept.

use std::collections::VecDeque;

use anyhow::{Result, anyhow, bail};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::{SamplerOptions, Stepper, randperm};
use crate::estimation::{Bounds, Objective, Prior, prior_draw};
use crate::linalg::{Mat, sym_eig};

enum Rotation {
    None,
    /// Rotated unimodal: `[V1, D] = eig(invhess)`, `WR = 3*sqrt(diag(D))`.
    Unimodal {
        v1: Vec<Vec<f64>>,
        wr: Vec<f64>,
    },
    /// Multimodal: directions between the modes.
    Modes(Vec<Vec<f64>>),
}

/// What `check_posterior_sampler_options.m` derives once for all chains.
pub(crate) struct SliceSetup {
    /// `thetaprior = [mh_bounds.lb mh_bounds.ub]`.
    bounds: Bounds,
    /// `W1 = initial_step_size*(ub-lb)`.
    w1: Vec<f64>,
    rotation: Rotation,
    use_prior_draws: bool,
}

impl SliceSetup {
    /// `mh_cov_loaded`: `load_mh_file` with `use_mh_covariance_matrix` (`invhess` is then the
    /// previous MCMC's posterior covariance).
    pub fn new(so: &SamplerOptions, invhess: &Mat, bounds: &Bounds, npar: usize, mh_cov_loaded: bool) -> Result<Self> {
        if bounds.lb.iter().chain(&bounds.ub).any(|v| !v.is_finite()) {
            bail!(
                "some priors are unbounded and prior_trunc is set to zero: the option \"slice\" is inconsistent with prior_trunc=0."
            );
        }
        let s = &so.slice;
        let rotation = if !s.modes.is_empty() {
            if s.modes.iter().any(|m| m.len() != npar) {
                bail!("slice: the modes must have {npar} parameters");
            }
            Rotation::Modes(s.modes.clone())
        } else if s.rotated {
            // ~cova_compute && ~(load_mh_file && use_mh_covariance_matrix) (slice sets
            // cova_compute = 0 unless slice_initialize_with_mode)
            if !s.initialize_with_mode && !mh_cov_loaded {
                bail!(
                    "check_posterior_sampler_options:: I cannot start rotated slice sampler because there is no previous \
                     MCMC to load or the Hessian at the mode is not computed (set slice_initialize_with_mode)."
                );
            }
            let (d, v) = sym_eig(invhess);
            Rotation::Unimodal {
                v1: (0..npar).map(|j| v.column(j).iter().copied().collect()).collect(),
                wr: d.iter().map(|x| x.sqrt() * 3.0).collect(),
            }
        } else {
            Rotation::None
        };
        if s.fast_likelihood_evaluation_for_rejection {
            println!(
                "NOTE: slice: fast_likelihood_evaluation_for_rejection: the posterior is always evaluated in full."
            );
        }
        Ok(SliceSetup {
            w1: bounds.lb.iter().zip(&bounds.ub).map(|(l, u)| s.initial_step_size * (u - l)).collect(),
            bounds: bounds.clone(),
            rotation,
            use_prior_draws: s.use_prior_draws,
        })
    }
}

pub(crate) struct Slice<'a> {
    obj: &'a dyn Objective,
    setup: &'a SliceSetup,
    priors: Option<&'a [Prior]>,
    rng: ChaCha8Rng,
    theta: Vec<f64>,
    /// Log posterior of `theta`.
    fx: f64,
    width: usize,
    /// `sampler_options.use_prior_draws.mh_blck(curr_block)`.
    try_prior_draws: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Side {
    Left,
    Right,
}

impl<'a> Slice<'a> {
    pub fn new(
        obj: &'a dyn Objective,
        setup: &'a SliceSetup,
        priors: Option<&'a [Prior]>,
        rng: ChaCha8Rng,
        x: &[f64],
        lp: f64,
        width: usize,
    ) -> Self {
        Slice {
            obj,
            setup,
            priors,
            rng,
            theta: x.to_vec(),
            fx: lp,
            width: width.max(1),
            try_prior_draws: setup.use_prior_draws,
        }
    }

    fn f(&self, x: &[f64]) -> f64 {
        -self.obj.eval(x).0
    }

    fn eval_many(&self, pts: Vec<Vec<f64>>) -> Vec<f64> {
        if pts.len() > 1 {
            pts.par_iter().map(|p| self.f(p)).collect()
        } else {
            pts.iter().map(|p| self.f(p)).collect()
        }
    }

    fn prior_draw(&mut self) -> Result<Vec<f64>> {
        let pr = self.priors.ok_or_else(|| anyhow!("slice sampler: the priors are needed to draw from the prior"))?;
        Ok(prior_draw(pr, &self.setup.bounds, &mut self.rng))
    }

    /// Stepping-out points from `x` (the next ones the loop would evaluate
    /// absent a stop), at most `max`; with Dynare's 30-evaluation cap
    /// (`standard`), only points reachable before it.
    #[allow(clippy::too_many_arguments)]
    fn step_points(side: Side, x: f64, bound: f64, w: f64, count: usize, cap: Option<usize>, max: usize) -> Vec<f64> {
        let mut out = Vec::new();
        let mut p = x;
        let open = |p: f64| if side == Side::Left { p > bound } else { p < bound };
        while out.len() < max && open(p) {
            let j = out.len();
            if j >= 1 && cap.is_some_and(|c| count + j > c) {
                break;
            }
            out.push(p);
            p = if side == Side::Left { bound.max(p - w) } else { bound.min(p + w) };
        }
        out
    }

    /// One univariate slice update of `make(x)` from `x = xold` within
    /// `[xlb, xub]`. Returns `(x, logpost, islow, neval)` with Dynare's count
    /// (the current point's evaluation included). `standard` = `slice_sampler.m`
    /// (caps, tight-interval exit, reset on an infinite posterior); otherwise
    /// `rotated_slice_sampler.m`.
    #[allow(clippy::too_many_arguments)]
    fn slice_1d(
        &mut self,
        make: &dyn Fn(f64) -> Vec<f64>,
        xold: f64,
        xlb: f64,
        xub: f64,
        w: f64,
        fxold: f64,
        standard: bool,
    ) -> (f64, f64, bool, usize) {
        let mut nev = 1usize;
        let z = fxold + self.rng.r#gen::<f64>().ln();
        let u: f64 = self.rng.r#gen();
        let mut l = xlb.max(xold - w * u);
        let mut r = xub.min(l + w);
        let (capl, capr) = if standard { (Some(30), Some(30)) } else { (None, None) };
        let mut ql: VecDeque<(f64, f64)> = VecDeque::new();
        let mut qr: VecDeque<(f64, f64)> = VecDeque::new();
        if self.width > 1 {
            // First round: both stepping-out directions together.
            let lp = Self::step_points(Side::Left, l, xlb, w, nev, capl, self.width.div_ceil(2));
            let rp = Self::step_points(Side::Right, r, xub, w, 0, capr, (self.width - lp.len()).max(1));
            let vals = self.eval_many(lp.iter().chain(&rp).map(|&x| make(x)).collect());
            ql.extend(lp.iter().copied().zip(vals[..lp.len()].iter().copied()));
            qr.extend(rp.iter().copied().zip(vals[lp.len()..].iter().copied()));
        }
        let take = |s: &mut Self,
                    q: &mut VecDeque<(f64, f64)>,
                    side: Side,
                    x: f64,
                    bound: f64,
                    count: usize,
                    cap: Option<usize>|
         -> f64 {
            if let Some(&(px, v)) = q.front()
                && px == x
            {
                q.pop_front();
                return v;
            }
            q.clear();
            let pts = Self::step_points(side, x, bound, w, count, cap, s.width);
            let vals = s.eval_many(pts.iter().map(|&p| make(p)).collect());
            q.extend(pts.into_iter().zip(vals));
            q.pop_front().map(|(_, v)| v).unwrap_or_else(|| s.f(&make(x)))
        };
        let eps_sqrt = f64::EPSILON.sqrt();
        while l > xlb {
            let fxl = take(self, &mut ql, Side::Left, l, xlb, nev, capl);
            nev += 1;
            if fxl <= z {
                break;
            }
            l = xlb.max(l - w);
            if standard && nev > 30 {
                ql.clear();
                l = xlb;
                let mut fxl = self.f(&make(l));
                let mut icount = 0;
                while (fxl.is_infinite() || fxl.is_nan()) && icount < 300 {
                    icount += 1;
                    l += eps_sqrt;
                    fxl = self.f(&make(l));
                }
            }
        }
        let nev1 = nev;
        while r < xub {
            let fxr = take(self, &mut qr, Side::Right, r, xub, nev - nev1, capr);
            nev += 1;
            if fxr <= z {
                break;
            }
            r = xub.min(r + w);
            if standard && nev > nev1 + 30 {
                qr.clear();
                r = xub;
                let mut fxr = self.f(&make(r));
                let mut icount = 0;
                while (fxr.is_infinite() || fxr.is_nan()) && icount < 300 {
                    icount += 1;
                    r -= eps_sqrt;
                    fxr = self.f(&make(r));
                }
            }
        }
        // Shrinkage, `width` candidates at a time along the all-rejected path.
        let mut x;
        let mut fxsim;
        let mut islow = false;
        'shrink: loop {
            let mut rr = self.rng.clone();
            let (mut ll, mut rg) = (l, r);
            let mut pts = Vec::new();
            let mut snaps = Vec::new();
            while pts.len() < self.width {
                let u: f64 = rr.r#gen();
                let xs = ll + u * (rg - ll);
                pts.push(xs);
                snaps.push(rr.clone());
                if xs > xold {
                    rg = xs;
                } else {
                    ll = xs;
                }
                if standard && rg - ll < 1e-6 {
                    break;
                }
            }
            let vals = self.eval_many(pts.iter().map(|&p| make(p)).collect());
            for ((xs, v), snap) in pts.into_iter().zip(vals).zip(snaps) {
                self.rng = snap;
                x = xs;
                fxsim = v;
                nev += 1;
                if xs > xold {
                    r = xs;
                } else {
                    l = xs;
                }
                // Dynare 7.2: the "sampling set too tight" exit also requires fxsim < Z.
                if standard && r - l < 1e-6 && fxsim < z {
                    islow = true;
                    x = xold;
                    fxsim = fxold;
                    break 'shrink;
                }
                if !(fxsim < z) {
                    break 'shrink;
                }
            }
        }
        if standard && (fxsim.is_infinite() || fxsim.is_nan()) {
            println!("SLICE: posterior density is infinite. Reset values at initial ones.");
            x = xold;
            fxsim = fxold;
        }
        (x, fxsim, islow, nev)
    }

    /// `rotated_slice_sampler.m`; returns Dynare's evaluation count.
    fn rotated(&mut self) -> usize {
        let (v1, wr): (Vec<Vec<f64>>, Option<Vec<f64>>) = match &self.setup.rotation {
            Rotation::Unimodal { v1, wr } => (v1.clone(), Some(wr.clone())),
            Rotation::Modes(mm) => {
                let dist: Vec<f64> = mm
                    .iter()
                    .map(|m| m.iter().zip(&self.theta).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt())
                    .collect();
                let im = (0..mm.len()).fold(0, |best, j| if dist[j] < dist[best] { j } else { best });
                let dirs: Vec<Vec<f64>> = (0..mm.len())
                    .filter(|&j| j != im)
                    .map(|j| {
                        let t: Vec<f64> = mm[j].iter().zip(&mm[im]).map(|(a, b)| a - b).collect();
                        let nrm = t.iter().map(|v| v * v).sum::<f64>().sqrt();
                        t.into_iter().map(|v| v / nrm).collect()
                    })
                    .collect();
                let perm = randperm(&mut self.rng, dirs.len());
                (perm.into_iter().map(|k| dirs[k].clone()).collect(), None)
            }
            Rotation::None => unreachable!(),
        };
        let b = &self.setup.bounds;
        let mut neval = 0;
        for (it, v) in v1.iter().enumerate() {
            let theta0 = self.theta.clone();
            let mut xlb = f64::NEG_INFINITY;
            let mut xub = f64::INFINITY;
            for i in 0..theta0.len() {
                let a = (b.lb[i] - theta0[i]) / v[i];
                let c = (b.ub[i] - theta0[i]) / v[i];
                xlb = xlb.max(a.min(c));
                xub = xub.min(a.max(c));
            }
            let w = wr.as_ref().map_or(xub - xlb, |wr| wr[it]);
            let make = |x: f64| -> Vec<f64> { theta0.iter().zip(v).map(|(t, d)| t + x * d).collect() };
            let fxold = self.fx;
            let (x, fx, _, nev) = self.slice_1d(&make, 0.0, xlb, xub, w, fxold, false);
            self.theta = make(x);
            self.fx = fx;
            neval += nev;
        }
        neval
    }

    /// One iteration of `slice_sampler.m`; returns Dynare's evaluation count.
    fn iterate(&mut self) -> Result<usize> {
        let mut neval_r = 0;
        if !matches!(self.setup.rotation, Rotation::None) {
            neval_r = self.rotated();
            if !matches!(self.setup.rotation, Rotation::Modes(_)) {
                return Ok(neval_r);
            }
        }
        // ---- estimation completion: estimated initial states (slice_sampler.m's endo_init_state)
        let hook = self.obj.init_state_ops();
        // draw_init_state_from_smoother.m checks its proposals against sampler_options.bounds
        let setup: &SliceSetup = self.setup;
        if hook.is_some() && !matches!(self.setup.rotation, Rotation::None) {
            bail!("slice: the rotated slice sampler with estimated initial states is not ported to dynare-rust");
        }
        let b_is;
        let b = match hook {
            Some(h) => {
                // thetaprior(index_init_state, :) = theta(index_init_state)
                let mut bb = self.setup.bounds.clone();
                for i in h.index_init_state() {
                    bb.lb[i] = self.theta[i];
                    bb.ub[i] = self.theta[i];
                }
                b_is = bb;
                &b_is
            }
            None => &self.setup.bounds,
        };
        let within = |c: &[f64], b: &Bounds| c.iter().zip(&b.lb).zip(&b.ub).all(|((v, l), u)| v >= l && v <= u);
        if self.try_prior_draws {
            let z1 = self.fx + self.rng.r#gen::<f64>().ln();
            let mut ilogpo2 = f64::NEG_INFINITY;
            let mut best = None;
            let mut nattempts = 0;
            while ilogpo2 < z1 && nattempts < 10 {
                nattempts += 1;
                let (cand, itest) = loop {
                    let cand = self.prior_draw()?;
                    if within(&cand, b) {
                        let itest = self.f(&cand);
                        if itest.is_finite() {
                            break (cand, itest);
                        }
                    }
                };
                if itest > ilogpo2 {
                    ilogpo2 = itest;
                    best = Some(cand);
                }
            }
            if ilogpo2 >= z1 {
                self.theta = best.unwrap();
                self.fx = ilogpo2;
            } else {
                self.try_prior_draws = false;
            }
        }
        let npar = self.theta.len();
        let mut neval = vec![0usize; npar];
        let mut islow = vec![false; npar];
        let mut it = 0;
        while it < npar {
            let mut i = it;
            neval[i] = 0;
            if b.lb[i] == b.ub[i] {
                it += 1;
                continue;
            }
            let mut fxold = self.fx;
            if !fxold.is_finite() {
                println!("slice_sampler:: Iteration {} started with bad parameter set (fval is inf or nan)", i + 1);
                let mut icount = 0;
                while !fxold.is_finite() && icount < 1000 {
                    icount += 1;
                    self.theta = self.prior_draw()?;
                    if within(&self.theta, b) {
                        fxold = self.f(&self.theta);
                    }
                }
                self.fx = fxold;
                i = 0;
                it = 0;
                neval[0] = 0;
            }
            let xold = self.theta[i];
            let theta = self.theta.clone();
            // ---- estimation completion: set_init_state(theta0, ys0) at each candidate
            let ys0 = hook.and_then(|h| h.steady_state_at(&theta));
            let make = |x: f64| -> Vec<f64> {
                let mut t = theta.clone();
                t[i] = x;
                match (hook, &ys0) {
                    (Some(h), Some(ys0)) => h.set_init_state(&t, ys0).0,
                    _ => t,
                }
            };
            let (x, fx, low, nev) = self.slice_1d(&make, xold, b.lb[i], b.ub[i], self.setup.w1[i], fxold, true);
            if hook.is_some() && x != xold {
                self.theta = make(x);
            } else {
                self.theta[i] = x;
            }
            self.fx = fx;
            islow[i] = low;
            neval[i] += nev;
            if let (Some(h), Some(ys0)) = (hook, &ys0) {
                let mut t = theta.clone();
                t[i] = x;
                if x != xold && h.set_init_state(&t, ys0).1 {
                    let (th, f, ne) = h.draw_init_state(false, 1, &self.theta, self.fx, &setup.bounds, &mut self.rng);
                    self.theta = th;
                    self.fx = f;
                    neval[i] += ne;
                }
            }
            it += 1;
        }
        if islow.iter().any(|&s| s) {
            let z1 = self.fx + self.rng.r#gen::<f64>().ln();
            let mut ilogpo2 = f64::NEG_INFINITY;
            let mut best = self.theta.clone();
            let mut nattempts = 0;
            while ilogpo2 < z1 && nattempts < 10 {
                nattempts += 1;
                let (cand, itest) = loop {
                    let mut cand = self.theta.clone();
                    let my = self.prior_draw()?;
                    for k in 0..npar {
                        if islow[k] {
                            cand[k] = my[k];
                        }
                    }
                    if (0..npar).filter(|&k| islow[k]).all(|k| cand[k] >= b.lb[k] && cand[k] <= b.ub[k]) {
                        if let Some(h) = hook {
                            cand = h.draw_init_state(true, 1, &cand, f64::NAN, &setup.bounds, &mut self.rng).0; // ---- estimation completion
                        }
                        let itest = self.f(&cand);
                        if itest.is_finite() {
                            break (cand, itest);
                        }
                    }
                };
                if itest > ilogpo2 {
                    ilogpo2 = itest;
                    best = cand;
                }
            }
            // Dynare 7.2: accept the prior draw only if it beats the slice level Z1.
            if ilogpo2 > z1 {
                self.theta = best;
                self.fx = ilogpo2;
            }
        }
        if let Some(h) = hook {
            // ---- estimation completion: draw_init_state_from_smoother([false 5], ...)
            let (th, f, ne) = h.draw_init_state(false, 5, &self.theta, self.fx, &setup.bounds, &mut self.rng);
            self.theta = th;
            self.fx = f;
            neval[0] += ne;
        }
        Ok(neval.iter().sum::<usize>() + neval_r)
    }
}

impl Stepper for Slice<'_> {
    fn rng(&self) -> &ChaCha8Rng {
        &self.rng
    }

    fn advance(&mut self, _max_steps: usize, on_step: &mut dyn FnMut(&[f64], f64, f64, usize)) -> Result<usize> {
        let ne = self.iterate()?;
        on_step(&self.theta, self.fx, 1.0, ne);
        Ok(1)
    }
}

/// The `maximize` branch of `posterior_sampler_iteration.m` for the slice sampler: each
/// iteration is `dynare_minimize_objective` (the slice options' `mode_compute`, newrat by
/// default, with the estimation's `optim`) of minus the log posterior from the last draw within
/// `bounds` (the MCMC bounds, or those of `maximize_using_mh_bounds`); the optimum is the new
/// draw, always accepted, one function evaluation counted.
pub(crate) struct Maximize<'a> {
    obj: &'a dyn Objective,
    bounds: &'a Bounds,
    so: &'a SamplerOptions,
    rng: ChaCha8Rng,
    last: Vec<f64>,
    chain: usize,
    iter: u64,
}

impl<'a> Maximize<'a> {
    pub fn new(
        obj: &'a dyn Objective,
        bounds: &'a Bounds,
        so: &'a SamplerOptions,
        rng: ChaCha8Rng,
        x: &[f64],
        chain: usize,
    ) -> Self {
        Maximize { obj, bounds, so, rng, last: x.to_vec(), chain, iter: 0 }
    }
}

impl Stepper for Maximize<'_> {
    fn rng(&self) -> &ChaCha8Rng {
        &self.rng
    }

    fn advance(&mut self, _max_steps: usize, on_step: &mut dyn FnMut(&[f64], f64, f64, usize)) -> Result<usize> {
        let n = self.last.len();
        let names: Vec<String> = (0..n).map(|i| format!("param_{}", i + 1)).collect();
        let sd: Vec<f64> = match &self.so.priors {
            Some(p) => p.iter().map(|p| p.sd).collect(),
            None => vec![f64::NAN; n],
        };
        let ctx = crate::estimation::optim::MinimizeContext {
            bounds: self.bounds,
            names: &names,
            prior_sd: &sd,
            optim: self.so.slice.optim.as_ref(),
            verbose: false,
            seed: (self.chain as u64) << 48 ^ self.iter,
            stream: 0,
        };
        let r = crate::estimation::optim::minimize(self.so.slice.mode_compute, self.obj, &self.last, &ctx)?;
        self.last = r.x;
        self.iter += 1;
        on_step(&self.last, -r.fval, 1.0, 1);
        Ok(1)
    }
}
