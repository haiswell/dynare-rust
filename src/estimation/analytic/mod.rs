// SPDX-License-Identifier: GPL-3.0-or-later
//! `estimation(analytic_derivation)`: the analytic score (and Hessian) of the likelihood.
//!
//! * [`AnalyticContext`] + [`AnalyticPosterior`]: the `analytic_derivation` branches of
//!   `estimation/dsge_likelihood.m` (`compute_analytic_derivation_info`, the filter calls with
//!   `analytic_deriv_info`, the univariate expanded state with its derivative arrays, the
//!   prior's derivatives, `analytic_Hessian = '' | 'opg' | 'full' | 'asymptotic'`).
//! * [`filters`]: `kalman_filter.m` / `univariate_kalman_filter.m` with derivatives.
//! * [`derivs2`]: the second-order parameter derivatives of the solution
//!   (`get_perturbation_params_derivs.m` with `d2flag`) for the `'full'` Hessian.
//! * [`prior`]: `priordens.m` with 2-3 outputs.
//!
//! The first-order parameter derivatives of the solution reuse `src/identification/derivs.rs`
//! (port of `get_perturbation_params_derivs.m`, `analytic_derivation_mode = 0`).

#![allow(clippy::needless_range_loop, clippy::type_complexity, clippy::ptr_arg, clippy::too_many_arguments)] // index-for-index ports

pub mod derivs2;
pub mod filters;
pub mod prior;

use std::sync::OnceLock;

use anyhow::{Result, anyhow, bail};

use crate::dsge::{DsgePosterior, Resolved};
use crate::estimation::{Evaluation, Objective};
use crate::identification::derivs::{ModelParamDerivs, ParamSet, PerturbationDerivs, perturbation_derivs};
use crate::linalg::{Mat, mul};
use crate::modfile::{EstimatedKind, OptValue, Options, opt};
use filters::{DerivInfo, FilterOut, Meas, SecondDerivs, unvech, vech};

/// `options_.analytic_Hessian`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalyticHessian {
    /// `''`: score only.
    None,
    /// `'opg'`: outer product of the per-period scores (and the prior's), newrat's iterations.
    Opg,
    /// `'full'`: the exact Hessian (second-order derivatives), the Hessian at the mode.
    Full,
    /// `'asymptotic'`: the expected-information form of the filters.
    Asymptotic,
}

/// `dsge_likelihood.m` with `nargout >= 5` under `analytic_derivation`.
#[derive(Clone, Debug)]
pub struct GradEvaluation {
    /// `fval` (+inf on failure).
    pub fval: f64,
    pub info: i32,
    pub info4: f64,
    /// `DLIK`: gradient of `fval` (NaN on failure).
    pub grad: Vec<f64>,
    /// `Hess` (`None` for `AnalyticHessian::None` or on failure).
    pub hess: Option<Mat>,
}

/// `[fval, info, exit_flag, DLIK, Hess] = fcn(x)` as `analytic_gradient_wrapper.m` and
/// `penalty_objective_function.m` call it: [`Objective::eval_grad`] or, for an objective
/// that only provides [`Objective::gradient`] (method of moments with `analytic_jacobian`),
/// its evaluation and gradient (`df = NaN` on failure, as in `mom.objective_function`).
pub fn eval_with_gradient(obj: &dyn Objective, x: &[f64], hess: AnalyticHessian) -> Option<GradEvaluation> {
    if let Some(g) = obj.eval_grad(x, hess) {
        return Some(g);
    }
    if !obj.has_gradient() {
        return None;
    }
    let e = obj.eval_full(x, false);
    let grad = if e.info == 0 { obj.gradient(x) } else { None }.unwrap_or_else(|| vec![f64::NAN; x.len()]);
    Some(GradEvaluation { fval: e.fval, info: e.info, info4: e.info4, grad, hess: None })
}

/// `[fval, exit_flag, grad, hess] = penalty_objective_function(x, fcn, base_penalty)` with an
/// analytic objective: on failure `fval = base_penalty + info(4)`.
pub fn penalty_objective_grad(
    obj: &dyn Objective,
    x: &[f64],
    base_penalty: f64,
    hess: AnalyticHessian,
) -> (f64, bool, Vec<f64>, Option<Mat>) {
    match eval_with_gradient(obj, x, hess) {
        Some(g) => {
            if g.info != 0 || !g.fval.is_finite() {
                let info4 = if g.info != 0 { g.info4 } else { 0.1 };
                (base_penalty + info4, false, g.grad, g.hess)
            } else {
                (g.fval, true, g.grad, g.hess)
            }
        }
        None => panic!("analytic_derivation: the objective provides no analytic gradient"),
    }
}

/// Settings of `analytic_derivation` read from the estimation options.
#[derive(Clone, Copy, Debug)]
pub struct AnalyticOptions {
    /// `options_.analytic_derivation_mode` (0: Sylvester equations; 1: Kronecker products,
    /// computed here with the Sylvester solution, identical up to rounding; -1/-2: two-sided
    /// finite differences of the solution / of the steady state and Jacobian).
    pub mode: i32,
}

/// Estimation options handled by this module and [`crate::estimation::mode_hessian`].
pub const OPTIONS: &[&str] =
    &["analytic_derivation", "analytic_derivation_mode", "cova_compute", "use_penalized_objective_for_hessian"];

/// Is `analytic_derivation` set (flag or non-zero value)?
pub fn requested(o: &Options) -> bool {
    match opt(o, "analytic_derivation") {
        Some(OptValue::Flag) => true,
        Some(OptValue::Num(v)) => *v != 0.0,
        Some(OptValue::Str(s)) => s.trim() == "true" || s.trim() == "1",
        _ => false,
    }
}

/// The checks Dynare runs on `analytic_derivation` before the estimation
/// (dynare_estimation_init.m, dsge_likelihood.m, dsge_conditional_likelihood_1.m), with its
/// error messages; `None` when the option is off.
pub fn check_options(
    o: &Options,
    kalman_algo: u8,
    lik_init: u8,
    missing: bool,
    order: i32,
) -> Result<Option<AnalyticOptions>> {
    if !requested(o) {
        return Ok(None);
    }
    let flag = |k: &str| match opt(o, k) {
        Some(OptValue::Flag) => true,
        Some(OptValue::Num(v)) => *v != 0.0,
        _ => false,
    };
    if flag("estimate_initial_states_endogenous_prior") {
        bail!(
            "estimation option conflict: estimate_initial_states_endogenous_prior isn't available for analytic_derivation"
        );
    }
    if flag("fast_kalman_filter") {
        bail!("estimation option conflict: analytic_derivation isn't available for fast_kalman_filter");
    }
    if flag("heteroskedastic_filter") {
        bail!("estimation option conflict: analytic_derivation isn't available for heteroskedastic_filter");
    }
    if kalman_algo == 5 {
        bail!("analytic derivation is incompatible with pruned skewed Kalman filter");
    }
    if lik_init == 3 {
        bail!("analytic derivation is incompatible with diffuse filter");
    }
    if flag("loglinear") {
        bail!("The analytic_derivation and loglinear options are not compatible");
    }
    if flag("endogenous_prior") {
        bail!("The analytic_derivation and endogenous_prior options are not compatible");
    }
    if flag("conditional_likelihood") {
        bail!("The analytic_derivation and conditional_likelihood are not compatible!");
    }
    if order > 1 {
        bail!(
            "analytic_derivation is only available at order=1: non_linear_dsge_likelihood.m returns no gradient (Dynare fails \
             in the optimizer)"
        );
    }
    if missing && kalman_algo != 2 {
        bail!(
            "analytic_derivation with missing observations needs kalman_algo=2: dsge_likelihood.m calls \
             missing_observations_kalman_filter.m, which returns no derivatives (Dynare 7.2 fails with \
             'Brace indexing is not supported for variables of this type.')"
        );
    }
    let mode = match opt(o, "analytic_derivation_mode") {
        Some(OptValue::Num(v)) => *v as i32,
        _ => 0,
    };
    if ![-2, -1, 0, 1].contains(&mode) {
        bail!("analytic_derivation_mode must be 0, 1, -1 or -2");
    }
    Ok(Some(AnalyticOptions { mode }))
}

/// Everything the analytic derivatives need besides the parameter vector (built once).
pub struct AnalyticContext {
    /// Symbolic parameter derivatives of the static and dynamic models.
    pderivs: ModelParamDerivs,
    /// Second-order derivative tapes (built on first use: `'full'` Hessian).
    second: OnceLock<derivs2::SecondTapes>,
    /// stderr / corr / model parameters of the estimation (`indpstderr`, `indpcorr`, `indpmodel`).
    set: ParamSet,
    /// For each `xparam1` entry: its position in `set` (None for measurement errors).
    to_set: Vec<Option<usize>>,
    /// nvx, nvn, ncx, ncn, np.
    counts: [usize; 5],
    /// Rows of `M_.H` of the measurement-error entries.
    me_rows: Vec<Option<(usize, usize)>>,
    pub opts: AnalyticOptions,
}

impl AnalyticContext {
    pub fn new(post: &DsgePosterior, opts: AnalyticOptions) -> Result<AnalyticContext> {
        let dsge = post.dsge;
        let mut set = ParamSet::default();
        let mut counts = [0usize; 5];
        for e in &dsge.estimated {
            match e.kind {
                EstimatedKind::StdErr(i) => {
                    set.stderr.push(i);
                    counts[0] += 1;
                }
                EstimatedKind::MeStdErr(_) => counts[1] += 1,
                EstimatedKind::Corr(i, j) => {
                    set.corr.push((i, j));
                    counts[2] += 1;
                }
                EstimatedKind::MeCorr(..) => counts[3] += 1,
                EstimatedKind::Param(i) => {
                    set.model.push(i);
                    counts[4] += 1;
                }
                EstimatedKind::Skew(_) => bail!("analytic derivation is incompatible with pruned skewed Kalman filter"),
                EstimatedKind::InitState(_) => bail!(
                    "estimation option conflict: estimate_initial_states_endogenous_prior isn't available for analytic_derivation"
                ),
            }
        }
        if opts.mode == -1 && counts[1] + counts[3] > 0 {
            bail!(
                "analytic_derivation_mode = -1 with estimated measurement errors: Dynare 7.2 fails in \
                 get_perturbation_params_derivs.m ('Error using reshape: Number of elements must not change')"
            );
        }
        let (mut is, mut ic, mut ip) = (0, 0, 0);
        let (ns, nc) = (set.stderr.len(), set.corr.len());
        let to_set = dsge
            .estimated
            .iter()
            .map(|e| match e.kind {
                EstimatedKind::StdErr(_) => {
                    is += 1;
                    Some(is - 1)
                }
                EstimatedKind::Corr(..) => {
                    ic += 1;
                    Some(ns + ic - 1)
                }
                EstimatedKind::Param(_) => {
                    ip += 1;
                    Some(ns + nc + ip - 1)
                }
                _ => None,
            })
            .collect();
        Ok(AnalyticContext {
            pderivs: ModelParamDerivs::new(&dsge.model).at_exo_steady_state(&dsge.exo_ss),
            second: OnceLock::new(),
            set,
            to_set,
            counts,
            me_rows: dsge.me_rows().to_vec(),
            opts,
        })
    }

    /// First-order derivatives of the solution (`get_perturbation_params_derivs.m`).
    fn first_order(&self, post: &DsgePosterior, res: &Resolved, g1: &Mat) -> Result<PerturbationDerivs> {
        let model = &post.dsge.model;
        if self.opts.mode < 0 {
            return self.numerical_first_order(post, res);
        }
        let md = self.pderivs.eval(model, &res.steady_state, &res.params, post.data.kalman.qz_criterium)?;
        perturbation_derivs(model, &res.dr, g1, &md, &res.q, &self.set)
    }

    /// `xparam1` restricted to the stderr / corr / model parameters (`set` order) at `res`.
    fn theta(&self, res: &Resolved) -> Vec<f64> {
        let ns = self.set.n_shock();
        (0..self.set.len())
            .map(|j| {
                if j < self.set.n_stderr() {
                    res.q[(self.set.stderr[j], self.set.stderr[j])].sqrt()
                } else if j < ns {
                    let (a, b) = self.set.corr[j - self.set.n_stderr()];
                    res.q[(a, b)] / (res.q[(a, a)] * res.q[(b, b)]).sqrt()
                } else {
                    res.params[self.set.model[j - ns]]
                }
            })
            .collect()
    }

    /// The model at `th` (`set` order): (ys, full dynamic Jacobian, Sigma_e, decision rule).
    fn solve_at(
        &self,
        post: &DsgePosterior,
        res: &Resolved,
        th: &[f64],
    ) -> Result<(Vec<f64>, Mat, Mat, crate::solver::DecisionRule)> {
        let dsge = post.dsge;
        let model = &dsge.model;
        let (p, q) = self.set.apply(th, &res.params, &res.q);
        let ys = dsge.steady_state(&p).map_err(|f| anyhow!("fjaco.m: cannot solve the model (info = {})", f.info))?;
        let g1 = model.dynamic_jacobian(&ys, &dsge.exo_ss, &p);
        let dr =
            dsge.solve_first_order(&g1, &ys).map_err(|e| anyhow!("fjaco.m: cannot solve the model (info = {e})"))?;
        Ok((ys, g1, q, dr))
    }

    /// `analytic_derivation_mode = -2` with `d2flag`: the first and second derivatives of the
    /// steady state and of the dynamic Jacobian w.r.t. the model parameters by finite
    /// differences (`fjaco.m` and `hessian_sparse.m` on the `'dynamic_model'` objective).
    fn numerical_model_d2(
        &self,
        post: &DsgePosterior,
        res: &Resolved,
    ) -> Result<(crate::identification::derivs::ModelDerivs, derivs2::D2Model)> {
        let model = &post.dsge.model;
        let n = model.n_endo();
        let ns = self.set.n_shock();
        let nm = self.set.n_model();
        let theta = self.theta(res);
        let f = |thm: &[f64]| -> Result<Vec<f64>> {
            let mut th = theta.clone();
            th[ns..].copy_from_slice(thm);
            let (ys, g1, ..) = self.solve_at(post, res, &th)?;
            let mut v = ys;
            v.extend_from_slice(g1.as_slice());
            Ok(v)
        };
        let thm = theta[ns..].to_vec();
        // fjaco: first derivatives
        let tol = 1e-5;
        let ncol = 3 * n + model.n_exo();
        let neq = model.n_equations();
        let mut dys = Mat::zeros(n, model.n_params());
        let mut dg1 = vec![Mat::zeros(neq, ncol); model.n_params()];
        for jm in 0..nm {
            let h = tol * thm[jm].abs().max(1.0);
            let (mut xp, mut xm) = (thm.clone(), thm.clone());
            xp[jm] += h;
            xm[jm] -= h;
            let hh = xp[jm] - xm[jm];
            let (fp, fm) = (f(&xp)?, f(&xm)?);
            let p = self.set.model[jm];
            for i in 0..n {
                dys[(i, p)] = (fp[i] - fm[i]) / hh;
            }
            dg1[p] = Mat::from_fn(neq, ncol, |r, c| (fp[n + r + c * neq] - fm[n + r + c * neq]) / hh);
        }
        let h2 = hessian_sparse(&f, &thm, [1e-2, 1.0])?;
        let mut d2ys = vec![Mat::zeros(n, nm); nm];
        let mut d2g1 = vec![Mat::zeros(neq, ncol); nm * (nm + 1) / 2];
        for b in 0..nm {
            for a in 0..=b {
                let v = &h2[b * (b + 1) / 2 + a];
                for i in 0..n {
                    d2ys[b][(i, a)] = v[i];
                    d2ys[a][(i, b)] = v[i];
                }
                d2g1[b * (b + 1) / 2 + a] = Mat::from_fn(neq, ncol, |r, c| v[n + r + c * neq]);
            }
        }
        Ok((crate::identification::derivs::ModelDerivs { dys, dg1 }, (d2ys, d2g1)))
    }

    /// `analytic_derivation_mode = -1` with `d2flag`: `hessian_sparse.m` on the
    /// `'Kalman_Transition'` objective `[Yss; vec(KalmanA); dyn_vech(Om)]` (DR order) w.r.t.
    /// all stderr / corr / model parameters.
    fn numerical_second_kalman(&self, post: &DsgePosterior, res: &Resolved) -> Result<derivs2::SecondOrder> {
        let n = post.dsge.model.n_endo();
        let ns = self.set.n_shock();
        let nm = self.set.n_model();
        let tot = self.set.len();
        let f = |th: &[f64]| -> Result<Vec<f64>> {
            let (ys, _, q, dr) = self.solve_at(post, res, th)?;
            let mut v: Vec<f64> = dr.order_var.iter().map(|&o| ys[o]).collect();
            let mut ka = Mat::zeros(n, n);
            for k in 0..dr.nspred {
                ka.set_column(dr.nstatic + k, &dr.ghx.column(k));
            }
            v.extend_from_slice(ka.as_slice());
            v.extend(vech(&mul(&mul(&dr.ghu, false, &q, false), false, &dr.ghu, true)));
            Ok(v)
        };
        let h2 = hessian_sparse(&f, &self.theta(res), [1e-2, 1.0])?;
        let nvech = n * (n + 1) / 2;
        let tot2 = tot * (tot + 1) / 2;
        let mut d2ka = Mat::zeros(n * n, tot2);
        let mut d2om = Mat::zeros(nvech, tot2);
        for (c, v) in h2.iter().enumerate() {
            for r in 0..n * n {
                d2ka[(r, c)] = v[n + r];
            }
            for r in 0..nvech {
                d2om[(r, c)] = v[n + n * n + r];
            }
        }
        let d2yss = (0..n)
            .map(|r| {
                Mat::from_fn(nm, nm, |a, b| {
                    let (i, j) = (ns + a.min(b), ns + a.max(b));
                    h2[j * (j + 1) / 2 + i][r]
                })
            })
            .collect();
        Ok(derivs2::SecondOrder { d2yss, d2kalman_a: d2ka, d2om })
    }

    /// `analytic_derivation_mode = -1 / -2`: two-sided finite differences (`fjaco.m`, step
    /// `options_.dynatol.x = 1e-5` relative) of ghx, ghu, Sigma_e (mode -1) or of the steady
    /// state and the dynamic Jacobian followed by the analytic Sylvester solution (mode -2).
    fn numerical_first_order(&self, post: &DsgePosterior, res: &Resolved) -> Result<PerturbationDerivs> {
        let dsge = post.dsge;
        let model = &dsge.model;
        let tol = 1e-5;
        let ns = self.set.n_shock();
        let tot = self.set.len();
        let n = model.n_endo();
        let theta: Vec<f64> = (0..tot)
            .map(|j| {
                if j < self.set.n_stderr() {
                    res.q[(self.set.stderr[j], self.set.stderr[j])].sqrt()
                } else if j < ns {
                    let (a, b) = self.set.corr[j - self.set.n_stderr()];
                    res.q[(a, b)] / (res.q[(a, a)] * res.q[(b, b)]).sqrt()
                } else {
                    res.params[self.set.model[j - ns]]
                }
            })
            .collect();
        let x_ss = dsge.exo_ss.clone();
        let solve = |th: &[f64]| -> Result<(Vec<f64>, Mat, Mat, crate::solver::DecisionRule, Vec<f64>)> {
            let (p, q) = self.set.apply(th, &res.params, &res.q);
            let ys =
                dsge.steady_state(&p).map_err(|f| anyhow!("fjaco.m: cannot solve the model (info = {})", f.info))?;
            let g1 = model.dynamic_jacobian(&ys, &x_ss, &p);
            let dr = dsge
                .solve_first_order(&g1, &ys)
                .map_err(|e| anyhow!("fjaco.m: cannot solve the model (info = {})", e))?;
            Ok((ys, g1, q, dr, p))
        };
        if self.opts.mode == -2 {
            // numerical dYss and dg1 (model parameters), analytic rest
            let nm = self.set.n_model();
            let ncol = 3 * n + model.n_exo();
            let mut dys = Mat::zeros(n, model.n_params());
            let mut dg1 = vec![Mat::zeros(model.n_equations(), ncol); model.n_params()];
            for jm in 0..nm {
                let j = ns + jm;
                let h = tol * theta[j].abs().max(1.0);
                let (mut xp, mut xm) = (theta.clone(), theta.clone());
                xp[j] += h;
                xm[j] -= h;
                let hh = xp[j] - xm[j];
                let (ys1, g11, ..) = solve(&xp)?;
                let (ys0, g10, ..) = solve(&xm)?;
                let p = self.set.model[jm];
                for i in 0..n {
                    dys[(i, p)] = (ys1[i] - ys0[i]) / hh;
                }
                dg1[p] = (g11 - g10) / hh;
            }
            let md = crate::identification::derivs::ModelDerivs { dys, dg1 };
            let g1 = model.dynamic_jacobian(&res.steady_state, &x_ss, &res.params);
            return perturbation_derivs(model, &res.dr, &g1, &md, &res.q, &self.set);
        }
        let mut dghx = Vec::with_capacity(tot);
        let mut dghu = Vec::with_capacity(tot);
        let mut dsig = Vec::with_capacity(tot);
        let mut d_yss = Mat::zeros(n, self.set.n_model());
        for j in 0..tot {
            let h = tol * theta[j].abs().max(1.0);
            let (mut xp, mut xm) = (theta.clone(), theta.clone());
            xp[j] += h;
            xm[j] -= h;
            let hh = xp[j] - xm[j];
            let (ys1, _, q1, dr1, _) = solve(&xp)?;
            let (ys0, _, q0, dr0, _) = solve(&xm)?;
            dghx.push((&dr1.ghx - &dr0.ghx) / hh);
            dghu.push((&dr1.ghu - &dr0.ghu) / hh);
            dsig.push((q1 - q0) / hh);
            if j >= ns {
                let ov = &res.dr.order_var;
                for i in 0..n {
                    d_yss[(i, j - ns)] = (ys1[ov[i]] - ys0[ov[i]]) / hh;
                }
            }
        }
        let ghu = &res.dr.ghu;
        let dom = (0..tot)
            .map(|j| {
                if j < ns {
                    mul(&mul(ghu, false, &dsig[j], false), false, ghu, true)
                } else {
                    let o = mul(&mul(&dghu[j], false, &res.q, false), false, ghu, true);
                    &o + o.transpose()
                }
            })
            .collect();
        Ok(PerturbationDerivs { d_yss, dg1_legacy: Vec::new(), dghx, dghu, dom, dsigma_e: dsig })
    }

    /// Port of `compute_analytic_derivation_info` (dsge_likelihood.m): DT, DYss, DOm, DH, DP
    /// (and their second derivatives with `full`) on the restricted state space.
    pub fn deriv_info(&self, post: &DsgePosterior, res: &Resolved, pstar: &Mat, full: bool) -> Result<DerivInfo> {
        let dsge = post.dsge;
        let model = &dsge.model;
        let x_ss = dsge.exo_ss.clone();
        let g1 = model.dynamic_jacobian(&res.steady_state, &x_ss, &res.params);
        let d = self.first_order(post, res, &g1)?;
        let iv = &dsge.ss_index.restrict_var_list;
        let rc = &dsge.ss_index.restrict_columns;
        let mm = iv.len();
        let k = dsge.n_estimated();
        let pp = dsge.varobs.len();
        let [nvx, nvn, ncx, ncn, np] = self.counts;
        let offset = nvx + nvn + ncx + ncn;
        let restrict_t = |dghx: &Mat| {
            let mut t = Mat::zeros(mm, mm);
            for (kk, &c) in rc.iter().enumerate() {
                for (i, &v) in iv.iter().enumerate() {
                    t[(i, c)] = dghx[(v, kk)];
                }
            }
            t
        };
        let restrict = |m: &Mat| Mat::from_fn(mm, mm, |i, j| m[(iv[i], iv[j])]);
        let mut dt = vec![Mat::zeros(mm, mm); k];
        let mut dom = vec![Mat::zeros(mm, mm); k];
        for (j, s) in self.to_set.iter().enumerate() {
            if let Some(s) = *s {
                dt[j] = restrict_t(&d.dghx[s]);
                dom[j] = restrict(&d.dom[s]);
            }
        }
        let mut dyss = Mat::zeros(mm, k);
        for jm in 0..np {
            for (i, &v) in iv.iter().enumerate() {
                dyss[(i, offset + jm)] = d.d_yss[(v, jm)];
            }
        }
        let h = &res.h;
        let mut dh = vec![Mat::zeros(pp, pp); k];
        for (j, e) in dsge.estimated.iter().enumerate() {
            match (e.kind, self.me_rows[j]) {
                (EstimatedKind::MeStdErr(_), Some((kk, _))) => {
                    let sigma = h[(kk, kk)].sqrt();
                    for c in 0..pp {
                        dh[j][(kk, c)] = h[(kk, c)] / sigma;
                        dh[j][(c, kk)] = h[(c, kk)] / sigma;
                    }
                    dh[j][(kk, kk)] = 2.0 * sigma;
                }
                (EstimatedKind::MeCorr(..), Some((k1, k2))) => {
                    let v = h[(k1, k1)].sqrt() * h[(k2, k2)].sqrt();
                    dh[j][(k1, k2)] = v;
                    dh[j][(k2, k1)] = v;
                }
                _ => {}
            }
        }
        let opts = &post.data.kalman;
        let t = res.t.clone();
        let lyap = |b: &Mat| crate::kalman::lyapunov_symm(&t, b, opts.qz_criterium, opts.lyapunov_complex_threshold);
        let mut dp = vec![Mat::zeros(mm, mm); k];
        if opts.lik_init == 1 {
            for j in 0..nvx {
                dp[j] = lyap(&dom[j])?;
            }
            for i in 0..ncx {
                let j = nvx + nvn + i;
                dp[j] = lyap(&dom[j])?;
            }
            for i in 0..np {
                let j = offset + i;
                let b = mul(&mul(&dt[j], false, pstar, false), false, &t, true)
                    + mul(&mul(&t, false, pstar, false), false, &dt[j], true)
                    + &dom[j];
                dp[j] = lyap(&b)?;
            }
        }
        let second = if full {
            let s2 = match self.opts.mode {
                -1 => self.numerical_second_kalman(post, res)?,
                -2 => {
                    let (md, d2) = self.numerical_model_d2(post, res)?;
                    derivs2::second_order(
                        model,
                        None,
                        &md,
                        Some(d2),
                        &res.dr,
                        &g1,
                        &res.steady_state,
                        &res.params,
                        &res.q,
                        &self.set,
                        &d,
                    )?
                }
                _ => {
                    model.check_external_order(3, "second-order derivatives with respect to the parameters")?;
                    let tapes = self.second.get_or_init(|| derivs2::SecondTapes::new(model));
                    let md = self.pderivs.eval(model, &res.steady_state, &res.params, opts.qz_criterium)?;
                    derivs2::second_order(
                        model,
                        Some(tapes),
                        &md,
                        None,
                        &res.dr,
                        &g1,
                        &res.steady_state,
                        &res.params,
                        &res.q,
                        &self.set,
                        &d,
                    )?
                }
            };
            Some(self.second_info(post, res, pstar, &t, &s2, &dt, &dp, h)?)
        } else {
            None
        };
        Ok(DerivInfo { k, dt, dyss, dom, dh, dp, second })
    }

    /// The `full_Hess` parts of `compute_analytic_derivation_info`.
    #[allow(clippy::too_many_arguments)]
    fn second_info(
        &self,
        post: &DsgePosterior,
        res: &Resolved,
        pstar: &Mat,
        t: &Mat,
        s2: &derivs2::SecondOrder,
        dt: &[Mat],
        dp: &[Mat],
        h: &Mat,
    ) -> Result<SecondDerivs> {
        let dsge = post.dsge;
        let iv = &dsge.ss_index.restrict_var_list;
        let n = dsge.model.n_endo();
        let mm = iv.len();
        let k = dsge.n_estimated();
        let pp = dsge.varobs.len();
        let [nvx, nvn, ncx, ncn, np] = self.counts;
        let offset0 = nvx + nvn + ncx + ncn;
        let tot = self.set.len();
        let npairs = k * (k + 1) / 2;
        let nvech = mm * (mm + 1) / 2;
        // restricted rows of the DERIVS arrays: vec index (iv,iv) of n x n, vech index of (iv,iv)
        let vec_rows: Vec<usize> =
            (0..mm).flat_map(|j| (0..mm).map(move |i| (i, j))).map(|(i, j)| iv[i] + iv[j] * n).collect();
        let vech_index = |a: usize, b: usize| {
            // position of (min, max) in dyn_vech (upper triangle, column-major) of an n x n matrix
            let (r, c) = if a <= b { (a, b) } else { (b, a) };
            c * (c + 1) / 2 + r
        };
        let _ = n;
        let mut vech_rows = Vec::with_capacity(nvech);
        for j in 0..mm {
            for i in 0..=j {
                vech_rows.push(vech_index(iv[i], iv[j]));
            }
        }
        // estimation pair -> DERIVS pair (jcount_derivs), None for nvn/ncn
        let pair_derivs = |a: usize, b: usize| -> Option<usize> {
            let (da, db) = (self.to_set[a]?, self.to_set[b]?);
            let (lo, hi) = if da <= db { (da, db) } else { (db, da) };
            Some(hi * (hi + 1) / 2 + lo)
        };
        let _ = tot;
        let mut d2t = Mat::zeros(mm * mm, npairs);
        let mut d2om = Mat::zeros(nvech, npairs);
        let mut jc = 0;
        for jj in 0..k {
            for ii in 0..=jj {
                if let Some(c) = pair_derivs(ii, jj) {
                    for (r, &src) in vec_rows.iter().enumerate() {
                        d2t[(r, jc)] = s2.d2kalman_a[(src, c)];
                    }
                    for (r, &src) in vech_rows.iter().enumerate() {
                        d2om[(r, jc)] = s2.d2om[(src, c)];
                    }
                }
                jc += 1;
            }
        }
        // D2Yss: zero block for the offset parameters
        let d2yss: Vec<Mat> = iv
            .iter()
            .map(|&v| {
                let mut m = Mat::zeros(k, k);
                for a in 0..np {
                    for b in 0..np {
                        m[(offset0 + a, offset0 + b)] = s2.d2yss[v][(a, b)];
                    }
                }
                m
            })
            .collect();
        let mut d2h = vec![Mat::zeros(pp, pp); k * k];
        let mut d2p = Mat::zeros(nvech, npairs);
        let opts = &post.data.kalman;
        let lyap = |b: &Mat| crate::kalman::lyapunov_symm(t, b, opts.qz_criterium, opts.lyapunov_complex_threshold);
        let li1 = opts.lik_init == 1;
        let mut jcount = 0;
        // stderr
        for i in 0..nvx {
            for _j in 0..=i {
                if li1 {
                    let x = lyap(&unvech(d2om.column(jcount).as_slice()))?;
                    d2p.column_mut(jcount).copy_from_slice(&vech(&x));
                }
                jcount += 1;
            }
        }
        // measurement-error stderr
        let mut offset = nvx;
        for i in 0..nvn {
            let ip = i + offset;
            let (kk, _) = self.me_rows[ip].expect("measurement error row");
            let sigma_k = h[(kk, kk)].sqrt();
            d2h[ip + ip * k][(kk, kk)] = 2.0;
            for j in 0..=ip {
                if j >= nvx && j < nvx + nvn && j != ip {
                    let (kj, _) = self.me_rows[j].expect("measurement error row");
                    if kj != kk {
                        let sigma_kj = h[(kj, kj)].sqrt();
                        let c = h[(kk, kj)] / (sigma_k * sigma_kj);
                        for (a, b) in [(ip, j), (j, ip)] {
                            d2h[a + b * k][(kk, kj)] = c;
                            d2h[a + b * k][(kj, kk)] = c;
                        }
                    }
                }
                jcount += 1;
            }
        }
        offset += nvn;
        // shock correlations
        for i in 0..ncx {
            let ip = i + offset;
            for j in 0..=ip {
                let d2omij = unvech(d2om.column(jcount).as_slice());
                if li1 && d2omij.iter().any(|&v| v != 0.0) {
                    let tmpv = mul(&mul(&dt[j], false, &dp[ip], false), false, t, true)
                        + mul(&mul(t, false, &dp[ip], false), false, &dt[j], true)
                        + &d2omij;
                    let x = lyap(&tmpv)?;
                    d2p.column_mut(jcount).copy_from_slice(&vech(&x));
                }
                jcount += 1;
            }
        }
        offset += ncx;
        // measurement-error correlations
        for i in 0..ncn {
            let ip = i + offset;
            let (k1, k2) = self.me_rows[ip].expect("measurement error rows");
            for j in 0..=ip {
                if j >= nvx && j < nvx + nvn {
                    let (kj, _) = self.me_rows[j].expect("measurement error row");
                    if kj == k1 {
                        let v = h[(k2, k2)].sqrt();
                        for (a, b) in [(j, ip), (ip, j)] {
                            d2h[a + b * k][(k1, k2)] = v;
                            d2h[a + b * k][(k2, k1)] = v;
                        }
                    }
                    if kj == k2 {
                        let v = d2h[j + ip * k][(k1, k2)] + h[(k1, k1)].sqrt();
                        for (a, b) in [(j, ip), (ip, j)] {
                            d2h[a + b * k][(k1, k2)] = v;
                            d2h[a + b * k][(k2, k1)] = v;
                        }
                    }
                }
                jcount += 1;
            }
        }
        offset += ncn;
        // model parameters
        if li1 {
            for j in 0..np {
                let jp = j + offset;
                for i in 0..=jp {
                    let (dti, dpi, dtj, dpj) = (&dt[i], &dp[i], &dt[jp], &dp[jp]);
                    let d2tij = filters::unvec(&d2t, jcount, mm);
                    let d2omij = unvech(d2om.column(jcount).as_slice());
                    let m3 = |a: &Mat, b: &Mat, c: &Mat| mul(&mul(a, false, b, false), false, c, true);
                    let tmp = m3(&d2tij, pstar, t)
                        + m3(t, pstar, &d2tij)
                        + m3(dti, dpj, t)
                        + m3(dtj, dpi, t)
                        + m3(t, dpj, dti)
                        + m3(t, dpi, dtj)
                        + m3(dti, pstar, dtj)
                        + m3(dtj, pstar, dti)
                        + d2omij;
                    let x = lyap(&tmp)?;
                    d2p.column_mut(jcount).copy_from_slice(&vech(&x));
                    jcount += 1;
                }
            }
        }
        let _ = res;
        Ok(SecondDerivs { d2t, d2yss, d2om, d2h, d2p })
    }
}

/// Port of `+identification/hessian_sparse.m`: second derivatives of the vector function `f`
/// by finite differences (steps `max(|x|, sqrt(gstep(1))) eps^(1/6) gstep(2)`); the entry of
/// the pair `(j, i)`, `j <= i`, is at `i(i+1)/2 + j`.
fn hessian_sparse(f: &dyn Fn(&[f64]) -> Result<Vec<f64>>, x: &[f64], gstep: [f64; 2]) -> Result<Vec<Vec<f64>>> {
    let n = x.len();
    let e6 = f64::EPSILON.powf(1.0 / 6.0);
    let h1: Vec<f64> = (0..n).map(|i| (x[i] + x[i].abs().max(gstep[0].sqrt()) * e6 * gstep[1]) - x[i]).collect();
    let h_1: Vec<f64> = (0..n).map(|i| x[i] - (x[i] - x[i].abs().max(gstep[0].sqrt()) * e6 * gstep[1])).collect();
    let f0 = f(x)?;
    let m = f0.len();
    let mut f1 = Vec::with_capacity(n);
    let mut f_1 = Vec::with_capacity(n);
    let mut xh = x.to_vec();
    for i in 0..n {
        xh[i] = x[i] + h1[i];
        f1.push(f(&xh)?);
        xh[i] = x[i] - h_1[i];
        f_1.push(f(&xh)?);
        xh[i] = x[i];
    }
    let mut out = vec![Vec::new(); n * (n + 1) / 2];
    for i in 0..n {
        out[i * (i + 1) / 2 + i] = (0..m).map(|r| (f1[i][r] + f_1[i][r] - 2.0 * f0[r]) / (h1[i] * h_1[i])).collect();
        for j in 0..i {
            let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
            xp[i] = x[i] + h1[i];
            xp[j] = x[j] + h_1[j];
            xm[i] = x[i] - h1[i];
            xm[j] = x[j] - h_1[j];
            let (fp, fm) = (f(&xp)?, f(&xm)?);
            out[i * (i + 1) / 2 + j] = (0..m)
                .map(|r| {
                    let ti = f1[i][r] + f_1[i][r] - f0[r];
                    let tj = f1[j][r] + f_1[j][r] - f0[r];
                    -(-fp[r] - fm[r] + ti + tj) / (2.0 * h1[i] * h_1[j])
                })
                .collect();
        }
    }
    Ok(out)
}

/// `dsge_likelihood` with `analytic_derivation` as an [`Objective`]: values come from the
/// regular [`DsgePosterior`]; [`Objective::eval_grad`] adds DLIK and Hess.
pub struct AnalyticPosterior<'a> {
    pub post: &'a DsgePosterior<'a>,
    pub ctx: &'a AnalyticContext,
}

impl Objective for AnalyticPosterior<'_> {
    fn dim(&self) -> usize {
        self.post.dim()
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        self.post.eval(x)
    }
    fn eval_full(&self, x: &[f64], contributions: bool) -> Evaluation {
        self.post.eval_full(x, contributions)
    }
    fn eval_grad(&self, x: &[f64], hess: AnalyticHessian) -> Option<GradEvaluation> {
        Some(match self.evaluate(x, hess) {
            Ok(g) => g,
            Err(e) => {
                GradEvaluation { fval: f64::INFINITY, info: e.0, info4: e.1, grad: vec![f64::NAN; x.len()], hess: None }
            }
        })
    }
}

impl AnalyticPosterior<'_> {
    /// One `dsge_likelihood` evaluation with derivatives; Err((info, info4)).
    pub fn evaluate(&self, x: &[f64], hess_kind: AnalyticHessian) -> std::result::Result<GradEvaluation, (i32, f64)> {
        let post = self.post;
        let e = post.evaluate(x).map_err(|f| (f.info, f.penalty))?;
        crate::progress::eval(e.fval);
        let res = &e.resolved;
        let dsge = post.dsge;
        let idx = &dsge.ss_index;
        let opts = &post.data.kalman;
        let n = x.len();
        // Y = data - constant - trend (as DsgePosterior::evaluate)
        let mut y = post.data.y.clone();
        for (i, &v) in idx.mfys.iter().enumerate() {
            let c = res.steady_state[v];
            for j in 0..y.ncols() {
                y[(i, j)] -= c;
            }
        }
        if let Some(tr) =
            dsge.trend_addition(&res.params, post.data.first_obs, y.ncols(), post.data.prefilter_mean.is_some())
        {
            y -= tr;
        }
        let pstar = match opts.lik_init {
            1 => crate::kalman::lyapunov::lyapunov_solver_opts(&res.t, &res.r, &res.q, opts).map_err(|_| (50, 0.1))?,
            k => crate::kalman::lik_init::initial_pstar(
                k,
                &res.t,
                &res.r,
                &res.q,
                Some(&res.h),
                &idx.mf,
                opts.qz_criterium,
                opts.lyapunov_complex_threshold,
                false,
            )
            .map_err(|_| (50, 0.1))?,
        };
        let full = hess_kind == AnalyticHessian::Full;
        let info = self.ctx.deriv_info(post, res, &pstar, full).map_err(|err| {
            eprintln!("analytic_derivation: {err}");
            (50, 0.1)
        })?;
        let mm = res.t.nrows();
        let a0 = if post.data.filter_initial_state.is_empty() {
            vec![0.0; mm]
        } else {
            crate::estimation::q3::kalman_a0(dsge, res, idx, &post.data.filter_initial_state)
        };
        let qq = mul(&mul(&res.r, false, &res.q, false), false, &res.r, true);
        let mut kalman_algo = if opts.kalman_algo == 2 { 2 } else { 1 };
        let mut out: Option<FilterOut> = None;
        if kalman_algo == 1 {
            let o = filters::kalman_filter(
                &y,
                &a0,
                &pstar,
                opts.kalman_tol,
                opts.riccati_tol,
                opts.rescale_prediction_error_covariance,
                opts.presample,
                &res.t,
                &qq,
                &res.h,
                &idx.mf,
                &info,
                hess_kind,
            );
            if o.lik.is_infinite() {
                if opts.use_univariate_filters_if_singularity_is_detected {
                    kalman_algo = 2;
                } else {
                    return Err((50, 0.1));
                }
            } else {
                out = Some(o);
            }
        }
        if kalman_algo == 2 {
            out = Some(univariate(post, &y, &a0, &pstar, &res.t, &res.r, &res.q, &res.h, &idx.mf, info, hess_kind));
        }
        let out = out.expect("a filter ran");
        if out.lik.is_nan() {
            return Err((45, 0.1));
        }
        if out.lik.is_infinite() {
            return Err((50, 0.1));
        }
        // prior
        let (dlnprior, d2lnprior) =
            if dsge.ml { (vec![0.0; n], vec![0.0; n]) } else { prior::log_prior_derivatives(x, &dsge.priors) };
        let grad: Vec<f64> = (0..n).map(|i| out.dlik[i] - dlnprior[i]).collect();
        let hess = match hess_kind {
            AnalyticHessian::None => None,
            AnalyticHessian::Full => {
                let mut h = -out.hess.expect("full Hessian");
                for i in 0..n {
                    h[(i, i)] -= d2lnprior[i];
                }
                Some(h)
            }
            AnalyticHessian::Asymptotic => out.hess,
            AnalyticHessian::Opg => {
                let start = opts.presample;
                let rows = out.dlikk.nrows().saturating_sub(start);
                let mut g = Mat::zeros(rows + 1, n);
                for j in 0..n {
                    g[(0, j)] = -dlnprior[j];
                    for r in 0..rows {
                        g[(r + 1, j)] = out.dlikk[(start + r, j)];
                    }
                }
                Some(mul(&g, true, &g, false))
            }
        };
        Ok(GradEvaluation { fval: e.fval, info: 0, info4: 0.0, grad, hess })
    }
}

/// The univariate branch of dsge_likelihood.m with derivatives: diagonal H (`H1 = diag(H)`,
/// `DH(j,:) = DH(j,j,:)`) or the state vector expanded with the measurement errors.
#[allow(clippy::too_many_arguments)]
fn univariate(
    post: &DsgePosterior,
    y: &Mat,
    a0: &[f64],
    pstar: &Mat,
    t: &Mat,
    r: &Mat,
    q: &Mat,
    h: &Mat,
    mf: &[usize],
    mut info: DerivInfo,
    hess_kind: AnalyticHessian,
) -> FilterOut {
    let opts = &post.data.kalman;
    let miss = &post.data.missing;
    let (mm, pp, k) = (t.nrows(), y.nrows(), info.k);
    let data_index: Vec<Vec<usize>> =
        if miss.any_missing { miss.aindex.clone() } else { vec![(0..pp).collect(); y.ncols()] };
    let nmmo = if miss.any_missing { miss.no_more_missing_observations } else { 0 };
    let diagonal = (0..pp).all(|i| (0..pp).all(|j| i == j || h[(i, j)].abs() < 1e-14));
    if diagonal {
        let h1: Vec<f64> = (0..pp).map(|i| h[(i, i)]).collect();
        let dh = Mat::from_fn(pp, k, |i, j| info.dh[j][(i, i)]);
        let d2h: Option<Vec<Mat>> = info
            .second
            .as_ref()
            .map(|s| (0..pp).map(|i| Mat::from_fn(k, k, |a, b| s.d2h[a + b * k][(i, i)])).collect());
        let qq = mul(&mul(r, false, q, false), false, r, true);
        return filters::univariate_kalman_filter(
            &data_index,
            nmmo,
            y,
            a0,
            pstar,
            opts.kalman_tol,
            opts.riccati_tol,
            opts.presample,
            t,
            &qq,
            &h1,
            &Meas::Index(mf),
            &info,
            &dh,
            d2h.as_deref(),
            hess_kind,
        );
    }
    // expanded state vector: Z = [Z1 I], T = blkdiag(T, 0), Q = blkdiag(Q, H), R = blkdiag(R, I)
    let m2 = mm + pp;
    let mut z = Mat::zeros(pp, m2);
    for (i, &zi) in mf.iter().enumerate() {
        z[(i, zi)] = 1.0;
        z[(i, mm + i)] = 1.0;
    }
    let blk = |a: &Mat, b: &Mat| {
        let mut m = Mat::zeros(a.nrows() + b.nrows(), a.ncols() + b.ncols());
        m.view_mut((0, 0), (a.nrows(), a.ncols())).copy_from(a);
        m.view_mut((a.nrows(), a.ncols()), (b.nrows(), b.ncols())).copy_from(b);
        m
    };
    let t2 = blk(t, &Mat::zeros(pp, pp));
    let qq2 = blk(&mul(&mul(r, false, q, false), false, r, true), h);
    let p2 = blk(pstar, h);
    let mut a2 = a0.to_vec();
    a2.extend(std::iter::repeat_n(0.0, pp));
    let dh3 = info.dh.clone();
    info.dt = info.dt.iter().map(|d| blk(d, &Mat::zeros(pp, pp))).collect();
    let mut dyss = Mat::zeros(m2, k);
    dyss.view_mut((0, 0), (mm, k)).copy_from(&info.dyss);
    info.dyss = dyss;
    info.dom = info.dom.iter().zip(&dh3).map(|(d, hh)| blk(d, hh)).collect();
    info.dp = info.dp.iter().zip(&dh3).map(|(d, hh)| blk(d, hh)).collect();
    if let Some(s) = info.second.as_mut() {
        let npairs = k * (k + 1) / 2;
        let mut d2t = Mat::zeros(m2 * m2, npairs);
        for c in 0..npairs {
            for jj in 0..mm {
                for ii in 0..mm {
                    d2t[(ii + jj * m2, c)] = s.d2t[(ii + jj * mm, c)];
                }
            }
        }
        s.d2t = d2t;
        let mut d2y = s.d2yss.clone();
        d2y.extend(std::iter::repeat_n(Mat::zeros(k, k), pp));
        s.d2yss = d2y;
        let aug = |old: &Mat, d2h: &[Mat]| {
            let mut out = Mat::zeros(m2 * (m2 + 1) / 2, npairs);
            let mut jc = 0;
            for jj in 0..k {
                for ii in 0..=jj {
                    let o = unvech(old.column(jc).as_slice());
                    let hm = &d2h[jj + ii * k];
                    let m = blk(&o, hm);
                    out.column_mut(jc).copy_from_slice(&vech(&m));
                    jc += 1;
                }
            }
            out
        };
        s.d2om = aug(&s.d2om, &s.d2h);
        s.d2p = aug(&s.d2p, &s.d2h);
        s.d2h = vec![Mat::zeros(pp, pp); k * k];
    }
    let dh = Mat::zeros(pp, k);
    let d2h: Option<Vec<Mat>> = info.second.as_ref().map(|_| vec![Mat::zeros(k, k); pp]);
    let h1 = vec![0.0; pp];
    filters::univariate_kalman_filter(
        &data_index,
        nmmo,
        y,
        &a2,
        &p2,
        opts.kalman_tol,
        opts.riccati_tol,
        opts.presample,
        &t2,
        &qq2,
        &h1,
        &Meas::Matrix(&z),
        &info,
        &dh,
        d2h.as_deref(),
        hess_kind,
    )
}
