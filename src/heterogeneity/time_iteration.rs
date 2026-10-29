// SPDX-License-Identifier: GPL-3.0-or-later
// ---- heterogeneity ----
//! Port of the `compute_steady_state_tensor` MEX of Dynare 7.2
//! (`mex/sources/heterogeneity/`): time iteration on the policy functions (Fischer-Burmeister
//! treatment of the complementarity conditions, trust-region solve at each grid point),
//! forward iteration of the distribution, aggregation and the Broyden calibration of the free
//! parameters.
//!
//! The multilinear interpolation is written with explicit corner enumeration instead of the
//! MEX's Gray-code updates; the mathematical result is the same (corners of zero weight are
//! skipped).

use super::HetDecl;
use super::model::{HetModel, HetWorkspace};
use super::steady::{LoadedSteadyState, bracket_linear_weight};
use crate::linalg::Mat;
use crate::model::Model;
use crate::solver::block_trust_region::trust_region_solve;
use anyhow::{Result, bail};
use std::cell::RefCell;

/// `options_.heterogeneity.time_iteration`.
#[derive(Clone, Debug)]
pub struct TimeIterationOptions {
    pub max_iter: usize,
    pub tol: f64,
    pub learning_rate: f64,
    pub early_stopping: usize,
    pub verbosity: i32,
    pub solver_tolf: f64,
    pub solver_tolx: f64,
    pub solver_factor: f64,
    pub solver_max_iter: usize,
    pub solver_stop_on_error: bool,
}

impl Default for TimeIterationOptions {
    fn default() -> Self {
        TimeIterationOptions {
            max_iter: 1000,
            tol: 1e-8,
            learning_rate: 1.0,
            early_stopping: 3,
            verbosity: 2,
            solver_tolf: 1e-10,
            solver_tolx: 1e-10,
            solver_factor: 100.0,
            solver_max_iter: 1000,
            solver_stop_on_error: false,
        }
    }
}

/// `options_.heterogeneity.forward`.
#[derive(Clone, Debug)]
pub struct ForwardOptions {
    pub max_iter: usize,
    pub tol: f64,
    pub check_every: usize,
    pub verbosity: i32,
}

impl Default for ForwardOptions {
    fn default() -> Self {
        ForwardOptions { max_iter: 10000, tol: 1e-10, check_every: 100, verbosity: 1 }
    }
}

/// A calibration target equation: an aggregate equation by `name` tag or by (1-based) number.
#[derive(Clone, Debug, PartialEq)]
pub enum TargetEquation {
    Number(usize),
    Name(String),
}

/// `options_.heterogeneity.calibration`.
#[derive(Clone, Debug)]
pub struct CalibrationOptions {
    pub tolf: f64,
    pub max_iter: usize,
    pub verbosity: i32,
    pub target_equations: Vec<TargetEquation>,
}

impl Default for CalibrationOptions {
    fn default() -> Self {
        CalibrationOptions { tolf: 1e-4, max_iter: 50, verbosity: 2, target_equations: Vec::new() }
    }
}

const FB_EPS: f64 = 1.0e-14;

fn fb_phi(a: f64, b: f64) -> f64 {
    a + b - (a * a + b * b + FB_EPS).sqrt()
}

fn fb_phi_derivative(a: f64, b: f64) -> (f64, f64) {
    let r = (a * a + b * b + FB_EPS).sqrt();
    (1.0 - a / r, 1.0 - b / r)
}

#[derive(Clone, Copy, PartialEq)]
enum Bound {
    None,
    Lower,
    Upper,
    Both,
}

/// `fb_transform`.
fn fb_transform(x: &[f64], f: &[f64], lb: &[f64], ub: &[f64], bt: &[Bound]) -> Vec<f64> {
    (0..x.len())
        .map(|i| match bt[i] {
            Bound::None => -f[i],
            Bound::Lower => fb_phi(x[i] - lb[i], f[i]),
            Bound::Upper => fb_phi(ub[i] - x[i], -f[i]),
            Bound::Both => {
                let inner = fb_phi(ub[i] - x[i], -f[i]);
                fb_phi(x[i] - lb[i], inner)
            }
        })
        .collect()
}

/// `fb_transform_jacobian`.
fn fb_transform_jacobian(x: &[f64], f: &[f64], jac: &Mat, lb: &[f64], ub: &[f64], bt: &[Bound]) -> Mat {
    let n = x.len();
    let mut out = Mat::zeros(n, n);
    for i in 0..n {
        match bt[i] {
            Bound::None => {
                for j in 0..n {
                    out[(i, j)] = -jac[(i, j)];
                }
            }
            Bound::Lower => {
                let (da, db) = fb_phi_derivative(x[i] - lb[i], f[i]);
                for j in 0..n {
                    out[(i, j)] = if i == j { da + db * jac[(i, j)] } else { db * jac[(i, j)] };
                }
            }
            Bound::Upper => {
                let (da, db) = fb_phi_derivative(ub[i] - x[i], -f[i]);
                for j in 0..n {
                    out[(i, j)] = if i == j { -da + db * (-jac[(i, j)]) } else { db * (-jac[(i, j)]) };
                }
            }
            Bound::Both => {
                let (ai, bi) = (ub[i] - x[i], -f[i]);
                let phi_inner = fb_phi(ai, bi);
                let (dia, dib) = fb_phi_derivative(ai, bi);
                let (doa, dob) = fb_phi_derivative(x[i] - lb[i], phi_inner);
                for j in 0..n {
                    let dinner = if i == j { -dia + dib * (-jac[(i, j)]) } else { dib * (-jac[(i, j)]) };
                    out[(i, j)] = if i == j { doa + dob * dinner } else { dob * dinner };
                }
            }
        }
    }
    out
}

/// `compute_stationary` of `markov.f08`: power iteration from the uniform distribution.
pub fn compute_stationary(p_mat: &Mat, tol: f64, maxiter: usize) -> (Vec<f64>, bool) {
    let n = p_mat.nrows();
    let mut p = vec![1.0 / n as f64; n];
    for _ in 0..maxiter {
        let temp: Vec<f64> = (0..n).map(|j| (0..n).map(|i| p[i] * p_mat[(i, j)]).sum()).collect();
        let diff = (0..n).map(|i| (temp[i] - p[i]).abs()).fold(0.0, f64::max);
        p = temp;
        if diff < tol {
            return (p, true);
        }
    }
    (p, false)
}

/// Result of the time iteration at one evaluation.
#[derive(Clone, Debug, Default)]
pub struct TimeIterationOutput {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: f64,
    /// `n_het_endo x N_sp`.
    pub policies: Mat,
}

#[derive(Clone, Debug, Default)]
pub struct DistributionOutput {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: f64,
    /// `N_om` (shocks fastest).
    pub hist: Vec<f64>,
}

/// `calibration_output` of the MEX (packed into a structure in MATLAB).
#[derive(Clone, Debug, Default)]
pub struct CalibrationOutput {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: f64,
    pub params: Vec<f64>,
    pub param_names: Vec<String>,
    pub time_iteration: TimeIterationOutput,
    pub distribution: DistributionOutput,
    /// `aggregates.Ix` (integral of every heterogeneous variable) and `aggregates.residuals`
    /// (aggregate equations).
    pub ix: Vec<f64>,
    pub residuals: Vec<f64>,
}

/// State of one steady-state computation (the MEX `calibration_input`).
pub struct SteadySolver<'a> {
    pub h: &'a HetModel,
    pub model: &'a Model,
    pub het: &'a HetDecl,
    pub ss: &'a LoadedSteadyState,
    pub ti: TimeIterationOptions,
    pub fwd: ForwardOptions,
    pub cal_verbosity: i32,
    pub params: Vec<f64>,
    /// `3 M_.endo_nbr`, the current block is updated with the aggregates.
    y: Vec<f64>,
    old_pol: Mat,
    // derived
    n_orig: usize,
    n_het: usize,
    n_e: usize,
    n_sp: usize,
    n_a_sp: usize,
    n_a_om: usize,
    n_om: usize,
    n_states: usize,
    state_var: Vec<usize>,
    bound_type: Vec<Bound>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    mu: Mat,
    mu_t: Mat,
    p_stat: Vec<f64>,
    eq_reordering: Vec<usize>,
    eq_reordering_inv: Vec<usize>,
    mult_in_het: Vec<usize>,
    pol_strides: Vec<usize>,
    d_strides: Vec<usize>,
    // output of the last evaluation
    ti_state: TimeIterationOutput,
    policies_om: Mat,
    // forward-iteration cache: corner target index and weight per (point, corner)
    fwd_targets: Vec<usize>,
    fwd_weights: Vec<f64>,
    fwd_n_corners: usize,
}

impl<'a> SteadySolver<'a> {
    pub fn new(
        h: &'a HetModel,
        model: &'a Model,
        het: &'a HetDecl,
        ss: &'a LoadedSteadyState,
        params: &[f64],
        ti: TimeIterationOptions,
        fwd: ForwardOptions,
        cal_verbosity: i32,
    ) -> Result<Self> {
        let sizes = &ss.sizes;
        let n_het = h.endo_nbr();
        let n_orig = h.orig_endo_nbr;
        let n_e = sizes.n_e_total;
        let mu = ss.mat.mu.clone();
        let (p_stat, ok) = compute_stationary(&mu, fwd.tol, fwd.max_iter);
        if !ok {
            bail!("Stationary distribution of heterogeneous shocks: convergence failed!");
        }
        let mut bound_type = Vec::new();
        let (lo, hi) = (&ss.mat.lower_bounds, &ss.mat.upper_bounds);
        for i in 0..n_orig {
            bound_type.push(match (lo[i].is_finite(), hi[i].is_finite()) {
                (true, true) => Bound::Both,
                (true, false) => Bound::Lower,
                (false, true) => Bound::Upper,
                _ => Bound::None,
            });
        }
        let reorder = h.mcp_equations_reordering.clone();
        let mut inv = vec![0; reorder.len()];
        for (i, &e) in reorder.iter().enumerate() {
            inv[e] = i;
        }
        let mut old_pol = ss.mat.x_bar.clone();
        for &m in &ss.indices.mult_in_het {
            for p in 0..old_pol.ncols() {
                old_pol[(m, p)] = 0.0;
            }
        }
        let strides = |dims: &[usize]| {
            let mut s = vec![1usize; dims.len()];
            for k in 1..dims.len() {
                s[k] = s[k - 1] * dims[k - 1];
            }
            s
        };
        let n_states = ss.mat.pol.dims.len();
        Ok(SteadySolver {
            h,
            model,
            het,
            ss,
            ti,
            fwd,
            cal_verbosity,
            params: params.to_vec(),
            y: ss.mat.y.clone(),
            old_pol,
            n_orig,
            n_het,
            n_e,
            n_sp: sizes.n_sp,
            n_a_sp: sizes.n_sp / n_e,
            n_a_om: sizes.d_n_a,
            n_om: sizes.n_om,
            n_states,
            state_var: h.state_var.clone(),
            bound_type,
            lower: lo.clone(),
            upper: hi.clone(),
            mu_t: mu.transpose(),
            mu,
            p_stat,
            eq_reordering: reorder,
            eq_reordering_inv: inv,
            mult_in_het: ss.indices.mult_in_het.clone(),
            pol_strides: strides(&ss.mat.pol.dims),
            d_strides: strides(&ss.mat.d.dims),
            ti_state: TimeIterationOutput { policies: Mat::zeros(n_het, sizes.n_sp), ..Default::default() },
            policies_om: Mat::zeros(n_het, sizes.n_om),
            fwd_targets: Vec::new(),
            fwd_weights: Vec::new(),
            fwd_n_corners: 1 << n_states,
        })
    }

    /// `set_het_input`: fills `ws` for the grid point `j` (shock index `j_e`) and the policy
    /// guess; returns `dpol_next` (`n_het x n_states`).
    fn set_het_input(&self, ws: &mut HetWorkspace, policy: &[f64], j: usize, j_e: usize, weighted: &Mat) -> Mat {
        let (nh, ns) = (self.n_het, self.n_states);
        let n_exo = self.h.exo_nbr();
        {
            let yh = ws.yh_mut();
            yh.fill(0.0);
            for (k, &sv) in self.state_var.iter().enumerate() {
                yh[sv] = self.ss.mat.sm[(n_exo + k, j)];
            }
            yh[nh..nh + self.n_orig].copy_from_slice(&policy[..self.n_orig]);
        }
        if self.h.set_auxiliary_variables {
            self.h.set_auxiliary_variables(ws, 0);
        }
        if !self.mult_in_het.is_empty() {
            for &m in &self.mult_in_het {
                ws.yh_mut()[nh + m] = 0.0;
            }
        }
        for k in 0..n_exo {
            ws.xf[k] = self.ss.mat.sm[(k, j)];
        }
        // brackets of the next-period states on the policy grids
        let mut ind = vec![0usize; ns];
        let mut w = vec![0.0; ns];
        let mut inv_h = vec![0.0; ns];
        for k in 0..ns {
            let grid = &self.ss.mat.pol.grids_array[k];
            let (i, wk) = bracket_linear_weight(grid, policy[self.state_var[k]]);
            ind[k] = i - 1;
            w[k] = wk;
            inv_h[k] = 1.0 / (grid[i] - grid[i - 1]);
        }
        let mut base = 0usize;
        for k in 0..ns {
            base += self.pol_strides[k] * ind[k];
        }
        let mut pol_next = vec![0.0; nh];
        let mut dpol = Mat::zeros(nh, ns);
        for corner in 0..(1usize << ns) {
            let mut wt = 1.0;
            let mut lin = base;
            for k in 0..ns {
                if corner >> k & 1 == 1 {
                    wt *= 1.0 - w[k];
                    lin += self.pol_strides[k];
                } else {
                    wt *= w[k];
                }
            }
            let l = lin * self.n_e + j_e;
            if wt != 0.0 {
                for v in 0..nh {
                    pol_next[v] += wt * weighted[(l, v)];
                }
            }
            // derivative with respect to each next-period state
            for k in 0..ns {
                let mut dw = if corner >> k & 1 == 1 { inv_h[k] } else { -inv_h[k] };
                let mut zero = false;
                for m in 0..ns {
                    if m == k {
                        continue;
                    }
                    let f = if corner >> m & 1 == 1 { 1.0 - w[m] } else { w[m] };
                    if f == 0.0 {
                        zero = true;
                        break;
                    }
                    dw *= f;
                }
                if !zero {
                    for v in 0..nh {
                        dpol[(v, k)] += dw * weighted[(l, v)];
                    }
                }
            }
        }
        ws.yh_mut()[2 * nh..3 * nh].copy_from_slice(&pol_next);
        dpol
    }

    /// One sweep of `solve_household_all_points`; returns the new policies.
    fn household_sweep(&self) -> Result<Mat> {
        let (nh, ne) = (self.n_het, self.n_e);
        // weighted_old_pol[(a, e), v] = sum_e' Mu[e, e'] old_pol[v, (a, e')]
        let mut weighted = Mat::zeros(self.n_sp, nh);
        for a in 0..self.n_a_sp {
            for e in 0..ne {
                for v in 0..nh {
                    let mut s = 0.0;
                    for e2 in 0..ne {
                        s += self.mu[(e, e2)] * self.old_pol[(v, a * ne + e2)];
                    }
                    weighted[(a * ne + e, v)] = s;
                }
            }
        }
        let mut new_pol = Mat::zeros(nh, self.n_sp);
        let ws = RefCell::new(self.h.workspace(&self.params));
        {
            let mut w = ws.borrow_mut();
            w.y_agg_mut().copy_from_slice(&self.y);
            let n_exo = self.h.exo_nbr();
            w.xf[n_exo..].copy_from_slice(&self.ss.mat.x);
        }
        let jac_pat = self.h.jacobian_pattern();
        let jv = RefCell::new(vec![0.0; jac_pat.len()]);
        let n_orig = self.n_orig;
        for ja in 0..self.n_a_sp {
            for je in 0..ne {
                let j = ne * ja + je;
                let ev = |x1: &[f64]| -> (Vec<f64>, Mat) {
                    let mut w = ws.borrow_mut();
                    let dpol = self.set_het_input(&mut w, x1, j, je, &weighted);
                    let mut resid = vec![0.0; nh];
                    self.h.resid(&mut w, &mut resid);
                    let resid_orig: Vec<f64> = (0..n_orig).map(|ii| resid[self.eq_reordering[ii]]).collect();
                    let fvec = fb_transform(x1, &resid_orig, &self.lower, &self.upper, &self.bound_type);
                    let mut jv = jv.borrow_mut();
                    self.h.jacobian_values(&mut w, &mut jv);
                    let mut jac = Mat::zeros(nh, nh);
                    let mut jac_next = Mat::zeros(nh, nh);
                    for (k, &(row, col)) in jac_pat.iter().enumerate() {
                        let ii = self.eq_reordering_inv[row];
                        if col >= nh && col < 2 * nh {
                            jac[(ii, col - nh)] = jv[k];
                        } else if col >= 2 * nh && col < 3 * nh {
                            jac_next[(ii, col - 2 * nh)] = jv[k];
                        }
                    }
                    for kk in 0..self.n_states {
                        for r in 0..nh {
                            let mut s = 0.0;
                            for c in 0..nh {
                                s += jac_next[(r, c)] * dpol[(c, kk)];
                            }
                            jac[(r, self.state_var[kk])] += s;
                        }
                    }
                    let jac_orig = Mat::from_fn(n_orig, n_orig, |a, b| jac[(a, b)]);
                    let fjac =
                        fb_transform_jacobian(x1, &resid_orig, &jac_orig, &self.lower, &self.upper, &self.bound_type);
                    (fvec, fjac)
                };
                let mut x: Vec<f64> = (0..n_orig).map(|i| self.old_pol[(i, j)]).collect();
                let info = trust_region_solve(
                    &ev,
                    &mut x,
                    self.ti.solver_tolx,
                    self.ti.solver_tolf,
                    self.ti.solver_max_iter,
                    self.ti.solver_factor,
                );
                if info != 1 && info != -1 && self.ti.solver_stop_on_error {
                    bail!(
                        "Time iteration failed: Household solver did not converge (trust region info {info} at grid point j_e={}, j_a={})",
                        je + 1,
                        ja + 1
                    );
                }
                let mut w = ws.borrow_mut();
                let nh_ = nh;
                w.yh_mut()[nh_..nh_ + n_orig].copy_from_slice(&x);
                if self.h.set_auxiliary_variables {
                    for lv in 0..self.h.het_aux_levels.len() {
                        self.h.set_auxiliary_variables(&mut w, lv);
                    }
                }
                for v in 0..nh {
                    new_pol[(v, j)] = w.yh()[nh + v];
                }
                for &m in &self.mult_in_het {
                    new_pol[(m, j)] = 0.0;
                }
            }
        }
        Ok(new_pol)
    }

    /// `solve_time_iteration_tensor`.
    fn solve_time_iteration(&mut self) -> Result<()> {
        if self.ti.verbosity == 2 {
            println!("  === Time Iteration ===");
        }
        let mut prev = f64::MAX;
        let mut n_inc = 0usize;
        let mut diff = f64::NAN;
        for iter in 1..=self.ti.max_iter {
            let new_pol = self.household_sweep()?;
            diff = (0..self.n_sp)
                .flat_map(|j| (0..self.n_het).map(move |i| (i, j)))
                .fold(0.0f64, |m, (i, j)| m.max((new_pol[(i, j)] - self.old_pol[(i, j)]).abs()));
            if self.ti.verbosity == 2 {
                println!("  Iter {iter:4}: ||Δpolicy|| = {}, tol = {}", ee(diff, 12, 4), ee(self.ti.tol, 12, 4));
            }
            if diff > prev {
                n_inc += 1;
            } else {
                n_inc = 0;
            }
            prev = diff;
            if self.ti.early_stopping > 0 && n_inc >= self.ti.early_stopping {
                if self.ti.verbosity == 2 {
                    println!("  Early stopping: {n_inc:3} consecutive increases in ||Δpolicy||");
                }
                self.ti_state = TimeIterationOutput {
                    converged: false,
                    iterations: iter,
                    residual_norm: diff,
                    policies: new_pol.clone(),
                };
                self.old_pol = new_pol;
                break;
            }
            if diff < self.ti.tol {
                if self.ti.verbosity == 2 {
                    println!("  Converged in {iter:4} iterations");
                }
                self.ti_state = TimeIterationOutput {
                    converged: true,
                    iterations: iter,
                    residual_norm: diff,
                    policies: new_pol.clone(),
                };
                self.old_pol = new_pol;
                break;
            }
            let lr = self.ti.learning_rate;
            for j in 0..self.n_sp {
                for i in 0..self.n_het {
                    self.old_pol[(i, j)] = lr * new_pol[(i, j)] + (1.0 - lr) * self.old_pol[(i, j)];
                }
            }
            if iter == self.ti.max_iter {
                // not converged
                if self.ti_state.iterations == 0 {
                    self.ti_state = TimeIterationOutput {
                        converged: false,
                        iterations: self.ti.max_iter,
                        residual_norm: diff,
                        policies: new_pol.clone(),
                    };
                    self.old_pol = new_pol;
                } else {
                    self.ti_state.converged = false;
                }
                if self.ti.verbosity == 2 {
                    println!("    Warning: Time iteration did not converge");
                }
            }
        }
        let _ = diff;
        // snap to the bounds
        let eps = FB_EPS.sqrt();
        for i in 0..self.n_orig {
            for j in 0..self.n_sp {
                if matches!(self.bound_type[i], Bound::Lower | Bound::Both)
                    && self.old_pol[(i, j)] - self.lower[i] < eps
                {
                    self.old_pol[(i, j)] = self.lower[i];
                }
                if matches!(self.bound_type[i], Bound::Upper | Bound::Both)
                    && self.upper[i] - self.old_pol[(i, j)] < eps
                {
                    self.old_pol[(i, j)] = self.upper[i];
                }
            }
        }
        self.ti_state.policies = self.old_pol.clone();
        // policies on the distribution grid
        let (ne, nh) = (self.n_e, self.n_het);
        let ns = self.n_states;
        let mut pom = Mat::zeros(nh, self.n_om);
        let n_pts = self.n_a_om;
        let d_ind = &self.ss.mat.d.ind;
        let d_w = &self.ss.mat.d.w;
        for a in 0..n_pts {
            let mut base = 0usize;
            for k in 0..ns {
                base += self.pol_strides[k] * (d_ind[a + k * n_pts] as usize - 1);
            }
            for corner in 0..(1usize << ns) {
                let mut wt = 1.0;
                let mut lin = base;
                for k in 0..ns {
                    let wk = d_w[a + k * n_pts];
                    if corner >> k & 1 == 1 {
                        wt *= 1.0 - wk;
                        lin += self.pol_strides[k];
                    } else {
                        wt *= wk;
                    }
                }
                if wt == 0.0 {
                    continue;
                }
                for e in 0..ne {
                    for v in 0..nh {
                        pom[(v, a * ne + e)] += wt * self.old_pol[(v, lin * ne + e)];
                    }
                }
            }
        }
        self.policies_om = pom;
        // brackets of the next-period states on the distribution grid
        let nc = self.fwd_n_corners;
        self.fwd_targets = vec![0; self.n_om * nc];
        self.fwd_weights = vec![0.0; self.n_om * nc];
        for p in 0..self.n_om {
            let e = p % ne;
            let a = p / ne;
            let _ = a;
            let mut ind = vec![0usize; ns];
            let mut w = vec![0.0; ns];
            for k in 0..ns {
                let (i, wk) =
                    bracket_linear_weight(&self.ss.mat.d.grids_array[k], self.policies_om[(self.state_var[k], p)]);
                ind[k] = i - 1;
                w[k] = wk;
            }
            let mut base = 0usize;
            for k in 0..ns {
                base += self.d_strides[k] * ind[k];
            }
            for corner in 0..nc {
                let mut wt = 1.0;
                let mut lin = base;
                for k in 0..ns {
                    if corner >> k & 1 == 1 {
                        wt *= 1.0 - w[k];
                        lin += self.d_strides[k];
                    } else {
                        wt *= w[k];
                    }
                }
                self.fwd_targets[p * nc + corner] = lin * ne + e;
                self.fwd_weights[p * nc + corner] = wt;
            }
        }
        Ok(())
    }

    /// `solve_distribution_tensor`.
    fn solve_distribution(&mut self) -> DistributionOutput {
        let (ne, n_a, n_om) = (self.n_e, self.n_a_om, self.n_om);
        let mut dist = vec![0.0; n_om];
        for a in 0..n_a {
            for e in 0..ne {
                dist[a * ne + e] = self.p_stat[e] / n_a as f64;
            }
        }
        let nc = self.fwd_n_corners;
        let mut out = DistributionOutput::default();
        if self.fwd.verbosity == 2 {
            println!("  === Forward Iteration ===");
        }
        let mut acc = vec![0.0; n_om];
        let mut next = vec![0.0; n_om];
        let mut diff = f64::NAN;
        for iter in 1..=self.fwd.max_iter {
            acc.iter_mut().for_each(|v| *v = 0.0);
            for p in 0..n_om {
                let d = dist[p];
                for c in 0..nc {
                    let w = self.fwd_weights[p * nc + c];
                    if w != 0.0 {
                        acc[self.fwd_targets[p * nc + c]] += w * d;
                    }
                }
            }
            for a in 0..n_a {
                for e in 0..ne {
                    let mut s = 0.0;
                    for e2 in 0..ne {
                        s += self.mu_t[(e, e2)] * acc[a * ne + e2];
                    }
                    next[a * ne + e] = s;
                }
            }
            if iter % self.fwd.check_every == 0 || iter == self.fwd.max_iter {
                diff = (0..n_om).map(|i| (dist[i] - next[i]).abs()).fold(0.0, f64::max);
                if self.fwd.verbosity == 2 {
                    println!(
                        "  Forward iter {iter:6}: ||ΔD|| = {}, tol = {}",
                        ee(diff, 10, 3),
                        ee(self.fwd.tol, 10, 3)
                    );
                }
                if diff < self.fwd.tol {
                    out.converged = true;
                    out.iterations = iter;
                    out.residual_norm = diff;
                    if self.fwd.verbosity == 2 {
                        println!("  Distribution converged in {iter} iterations (norm: {})", ee(diff, 10, 3));
                    }
                    dist.copy_from_slice(&next);
                    break;
                }
            }
            dist.copy_from_slice(&next);
        }
        if !out.converged {
            out.iterations = self.fwd.max_iter;
            out.residual_norm = diff;
            if self.fwd.verbosity >= 1 {
                println!(
                    "    WARNING: Distribution did not converge after {} iterations (norm: {})",
                    self.fwd.max_iter,
                    ee(diff, 10, 3)
                );
            }
        }
        out.hist = dist;
        out
    }

    /// `agg_resid_tensor`: time iteration, distribution, aggregation and aggregate residuals.
    fn agg_resid(&mut self) -> Result<(DistributionOutput, Vec<f64>, Vec<f64>)> {
        self.solve_time_iteration()?;
        let dist = self.solve_distribution();
        // aggregates
        let nh = self.n_het;
        let ix: Vec<f64> =
            (0..nh).map(|v| (0..self.n_om).map(|p| self.policies_om[(v, p)] * dist.hist[p]).sum()).collect();
        let n_endo = self.model.n_endo();
        let yagg: Vec<f64> = self.ss.indices.ix_in_het.iter().map(|&i| ix[i]).collect();
        for (k, &ia) in self.ss.indices.ix_in_agg.iter().enumerate() {
            self.y[n_endo + ia] = yagg[k];
        }
        let mut resid = vec![0.0; self.model.n_equations()];
        if n_endo > 0 {
            self.het.agg_residuals(self.model, &self.y, &self.ss.mat.x, &self.params, &yagg, &mut resid);
        }
        if self.cal_verbosity == 2 {
            println!("  === Aggregate variables and residuals ===");
            println!("  Aggregate residuals:");
            for (i, r) in resid.iter().enumerate().take(self.model.orig_endo_nbr) {
                println!("    {}: {}", self.equation_name(i), ee(*r, 15, 6));
            }
        }
        Ok((dist, ix, resid))
    }

    fn equation_name(&self, i: usize) -> String {
        self.model
            .equation_tags
            .get(i)
            .and_then(|t| t.iter().find(|(k, _)| k == "name"))
            .map(|x| x.1.clone())
            .unwrap_or_else(|| format!("{}", i + 1))
    }
}

/// `calibrate`: runs the steady-state computation; `free` = (parameter index in `M_.params`,
/// name, initial value, lower bound, upper bound), `targets` = 0-based aggregate equations.
pub fn calibrate(
    sv: &mut SteadySolver<'_>,
    free: &[(usize, String, f64, f64, f64)],
    targets: &[usize],
    cal: &CalibrationOptions,
) -> Result<CalibrationOutput> {
    let n = free.len();
    let mut out = CalibrationOutput::default();
    if n == 0 {
        let (dist, ix, resid) = sv.agg_resid()?;
        out.converged = dist.converged;
        out.iterations = 1;
        out.residual_norm = resid.iter().take(sv.model.n_endo()).fold(0.0, |m: f64, v| m.max(v.abs()));
        out.time_iteration = sv.ti_state.clone();
        out.distribution = dist;
        out.ix = ix;
        out.residuals = resid;
        if cal.verbosity == 2 {
            println!("No calibration parameters - computing residuals only");
        }
        return Ok(out);
    }
    if cal.verbosity == 2 {
        println!("=== Parameter Calibration (Broyden) ===");
        println!("  Initial parameter guesses:");
        for f in free {
            println!("    {} = {} (lb={}, ub={})", f.1, ee(f.2, 12, 5), ee(f.3, 12, 5), ee(f.4, 12, 5));
        }
    }
    let lb: Vec<f64> = free.iter().map(|f| f.3).collect();
    let ub: Vec<f64> = free.iter().map(|f| f.4).collect();
    let mut x: Vec<f64> = free.iter().map(|f| f.2).collect();
    let last: RefCell<Option<(DistributionOutput, Vec<f64>, Vec<f64>)>> = RefCell::new(None);
    let mut fcn = |x: &[f64], sv: &mut SteadySolver<'_>| -> Option<Vec<f64>> {
        for (k, f) in free.iter().enumerate() {
            sv.params[f.0] = x[k];
        }
        let r = match sv.agg_resid() {
            Ok(r) => r,
            Err(_) => return None,
        };
        let res: Vec<f64> = targets.iter().map(|&t| r.2[t]).collect();
        if sv.cal_verbosity == 2 {
            let max_norm = res.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            println!("  Max norm: {}, tol = {}", ee(max_norm, 15, 6), ee(cal.tolf, 15, 6));
            println!("  Calibration iteration complete");
            for f in free {
                println!("  {} = {}", f.1, ee(sv.params[f.0], 12, 5));
            }
        }
        *last.borrow_mut() = Some(r);
        Some(res)
    };
    let (info, nfev, fvec) = broyden_solve(&mut x, &mut fcn, sv, cal.tolf, cal.max_iter, &lb, &ub, cal.verbosity);
    out.converged = info == 0;
    if cal.verbosity == 2 {
        if out.converged {
            println!("Calibration converged successfully!");
            println!("Calibrated parameters:");
            for (k, f) in free.iter().enumerate() {
                println!("  {} = {}", f.1, ee(x[k], 12, 5));
            }
            println!("Converged in {nfev:4} function evaluations");
            println!("Calibration complete");
        } else {
            println!("Calibration failed to converge");
            println!(
                "    Broyden: {}",
                match info {
                    1 => "Maximum iterations exceeded",
                    2 => "Singular Jacobian encountered",
                    3 => "Too many backtracks",
                    4 => "Function evaluation error",
                    _ => "Unknown error code",
                }
            );
        }
    }
    out.params = x;
    out.param_names = free.iter().map(|f| f.1.clone()).collect();
    out.residual_norm = fvec.iter().fold(0.0, |m: f64, v| m.max(v.abs()));
    out.iterations = nfev;
    out.time_iteration = sv.ti_state.clone();
    if let Some((dist, ix, resid)) = last.into_inner() {
        out.distribution = dist;
        out.ix = ix;
        out.residuals = resid;
    }
    Ok(out)
}

/// `broyden_solve` (`utils/broyden_solver.f08`): returns `(info, nfev, fvec)`.
#[allow(clippy::too_many_arguments)]
fn broyden_solve<C>(
    x: &mut Vec<f64>,
    fcn: &mut C,
    ctx: &mut SteadySolver<'_>,
    tol: f64,
    max_iter: usize,
    lb: &[f64],
    ub: &[f64],
    verbosity: i32,
) -> (i32, usize, Vec<f64>)
where
    C: FnMut(&[f64], &mut SteadySolver<'_>) -> Option<Vec<f64>>,
{
    let n = x.len();
    let (h, bc, max_backtrack) = (1.0e-5, 0.5, 30usize);
    let mut nfev = 0usize;
    let Some(mut fvec) = fcn(x, ctx) else { return (4, 1, vec![f64::NAN; n]) };
    nfev += 1;
    let mut jac = Mat::zeros(n, n);
    for iter in 1..=max_iter {
        let maxres = fvec.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if maxres < tol {
            return (0, nfev, fvec);
        }
        if iter == 1 {
            if verbosity == 2 {
                println!("  Computing initial Jacobian via finite differences...");
            }
            for i in 0..n {
                let mut xp = x.clone();
                xp[i] += h;
                let Some(df) = fcn(&xp, ctx) else { return (4, nfev, fvec) };
                nfev += 1;
                for r in 0..n {
                    jac[(r, i)] = (df[r] - fvec[r]) / h;
                }
                if verbosity == 2 {
                    println!(
                        "    FD[{:2}]: x={}, h={}, f={}, f_pert={}",
                        i + 1,
                        ee(x[i], 12, 5),
                        ee(h, 12, 5),
                        ee(fvec[0], 12, 5),
                        ee(df[0], 12, 5)
                    );
                    println!("           J={}", ee(jac[(0, i)], 12, 5));
                }
            }
        }
        // dgesv: solve J dx = -f
        let mut dx = match solve_dense(&jac, &fvec.iter().map(|v| -v).collect::<Vec<_>>()) {
            Some(d) => d,
            None => return (2, nfev, fvec),
        };
        let mut accepted = false;
        for _ in 0..max_backtrack {
            let mut xn: Vec<f64> = (0..n).map(|i| x[i] + dx[i]).collect();
            for i in 0..n {
                if lb[i].is_finite() {
                    xn[i] = xn[i].max(lb[i]);
                }
                if ub[i].is_finite() {
                    xn[i] = xn[i].min(ub[i]);
                }
            }
            nfev += 1;
            match fcn(&xn, ctx) {
                None => {
                    for d in dx.iter_mut() {
                        *d *= bc;
                    }
                }
                Some(fnew) => {
                    let df: Vec<f64> = (0..n).map(|i| fnew[i] - fvec[i]).collect();
                    let dxa: Vec<f64> = (0..n).map(|i| xn[i] - x[i]).collect();
                    fvec = fnew;
                    *x = xn;
                    // broyden_update
                    let nrm: f64 = dxa.iter().map(|v| v * v).sum();
                    if nrm >= 1.0e-30 {
                        let jdx: Vec<f64> = (0..n).map(|r| (0..n).map(|c| jac[(r, c)] * dxa[c]).sum()).collect();
                        let u: Vec<f64> = (0..n).map(|r| (df[r] - jdx[r]) * (1.0 / nrm)).collect();
                        for r in 0..n {
                            for c in 0..n {
                                jac[(r, c)] += u[r] * dxa[c];
                            }
                        }
                    }
                    accepted = true;
                    break;
                }
            }
        }
        if !accepted {
            return (3, nfev, fvec);
        }
    }
    (1, nfev, fvec)
}

/// Gaussian elimination with partial pivoting (LAPACK `dgesv`); `None` for a singular matrix.
fn solve_dense(a: &Mat, b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m = a.clone();
    let mut x = b.to_vec();
    for k in 0..n {
        let (mut p, mut best) = (k, m[(k, k)].abs());
        for i in k + 1..n {
            if m[(i, k)].abs() > best {
                best = m[(i, k)].abs();
                p = i;
            }
        }
        if best == 0.0 {
            return None;
        }
        if p != k {
            for c in 0..n {
                let t = m[(k, c)];
                m[(k, c)] = m[(p, c)];
                m[(p, c)] = t;
            }
            x.swap(k, p);
        }
        for i in k + 1..n {
            let f = m[(i, k)] / m[(k, k)];
            m[(i, k)] = f;
            for c in k + 1..n {
                let v = m[(k, c)];
                m[(i, c)] -= f * v;
            }
            x[i] -= f * x[k];
        }
    }
    for k in (0..n).rev() {
        let mut s = x[k];
        for c in k + 1..n {
            s -= m[(k, c)] * x[c];
        }
        x[k] = s / m[(k, k)];
    }
    Some(x)
}

/// C `%W.PE` (as the solver prints it: `1.2345E+01`).
fn ee(x: f64, width: usize, prec: usize) -> String {
    if x.is_infinite() {
        return format!("{:>width$}", if x > 0.0 { "Infinity" } else { "-Infinity" });
    }
    crate::analysis::table::fmt_e(x, width, prec).to_uppercase()
}
