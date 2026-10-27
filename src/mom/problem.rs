// SPDX-License-Identifier: GPL-3.0-or-later
//! The method-of-moments estimation problem and its objective: port of
//! `+mom/objective_function.m` (with `set_all_parameters.m`,
//! `check_bounds_and_definiteness_estimation.m`, `compute_decision_rules.m`).
//!
//! [`MomProblem::evaluate`] is a pure function of the parameter vector (the SMM shocks and
//! the GIRF draws are fixed in the problem), so the optimizers may evaluate in parallel.

use anyhow::{Result, anyhow};

use super::moments::{
    Method, StateSpaceMoments, first_order_moments, get_data_moments, gmm_model_moments, is_first_moment,
};
use super::{MatchedMoment, irf_matching::IrfTargets};
use crate::estimation::{Bounds, Evaluation, Objective, Prior, log_prior_density};
use crate::linalg::{Mat, chol_upper, mul, sym_eig};
use crate::model::Model;
use crate::model::higher::DynamicDerivatives;
use crate::modfile::{EstimatedKind, EstimatedParam};
use crate::solver::korder::{self, KOrderDr};
use crate::solver::{DecisionRule, FirstOrderSolver, SolverOptions, SteadyOptions};

/// IRF matching settings fixed for the whole estimation.
pub struct IrfSetup {
    pub targets: IrfTargets,
    /// `log(det(inv(W)))` (`Winv_logdet`).
    pub winv_logdet: f64,
    pub relative_irf: bool,
    pub add_tiny_number_to_cholesky: f64,
    pub drop: usize,
    /// Standard-normal draws for the GIRFs at order > 1 (`(irf + drop) x exo_nbr` each),
    /// fixed across evaluations (common random numbers; Dynare redraws on every call).
    pub draws: Vec<Mat>,
}

/// SMM settings.
pub struct SmmSetup {
    /// `options_mom_.mom.shock_series`: `(long + burnin) x exo_nbr` standard normals.
    pub shock_series: Mat,
    /// `options_mom_.mom.ME_shock_series`: `long x length(M_.H)` standard normals for the
    /// measurement errors, drawn after `shock_series` from the same stream.
    pub me_shock_series: Mat,
    pub long: usize,
}

pub struct MomProblem {
    pub method: Method,
    pub model: Model,
    pub order: usize,
    pub pruning: bool,
    solver: FirstOrderSolver,
    deriv: Option<DynamicDerivatives>,
    pub(super) pderivs: std::sync::OnceLock<crate::identification::derivs::ModelParamDerivs>,
    /// Estimated parameters in `xparam1` order and their names (`bayestopt_.name`).
    pub estimated: Vec<EstimatedParam>,
    pub names: Vec<String>,
    pub base_params: Vec<f64>,
    pub base_sigma_e: Mat,
    /// `M_.H` (`nobs x nobs`, varobs order) and `M_.Correlation_matrix_ME` from the `shocks` block.
    pub base_h: Mat,
    pub base_corr_h: Mat,
    /// `estim_params_.calibrated_covariances_ME` (row, column, value).
    pub cal_cov_h: Vec<(usize, usize, f64)>,
    /// Rows of `M_.H` (varobs positions) of each estimated measurement-error entry.
    pub me_rows: Vec<Option<(usize, usize)>>,
    pub steady_opts: SteadyOptions,
    /// `steady_state_model` block, if any (stream D's closed-form steady state).
    pub ssm: Option<crate::deterministic::SteadyStateModel>,
    /// `oo_.steady_state` at the initial parameters (guess of every steady-state solve).
    pub ys_guess: Vec<f64>,
    /// `oo_.exo_steady_state`: the steady state and the Jacobians are evaluated there.
    pub exo_ss: Vec<f64>,
    /// `BoundsInfo`.
    pub bounds: Bounds,
    /// Observed variables (endogenous indices, varobs order).
    pub varobs: Vec<usize>,
    /// Matched moments as used in estimation (after `matched_moments_block`).
    pub rows: Vec<MatchedMoment>,
    pub prefilter: bool,
    pub hp_filter: f64,
    /// `one_sided_hp_filter` lambda (0 = off).
    pub one_sided_hp_filter: f64,
    /// `bandpass_filter` passband `[high low]` (`options_.bandpass`, K = 12).
    pub bandpass: Option<[f64; 2]>,
    /// `options_mom_.ar`: largest lag in the moments.
    pub nlags: usize,
    pub qz_criterium: f64,
    pub schur_vec_tol: f64,
    /// `weighting_matrix_scaling_factor`.
    pub scaling: f64,
    /// Penalized estimator: prior means and standard deviations (`bayestopt_.p1`, `p2`).
    pub penalty: Option<(Vec<f64>, Vec<f64>)>,
    /// Priors entering the IRF-matching posterior kernel (`priordens`).
    pub priors: Option<Vec<Prior>>,
    pub smm: Option<SmmSetup>,
    pub irf: Option<IrfSetup>,
}

/// One successful evaluation of `mom.objective_function`.
#[derive(Clone, Debug)]
pub struct Eval {
    pub fval: f64,
    /// The quadratic form (`Q`).
    pub q: f64,
    pub model_moments: Vec<f64>,
    /// GMM/SMM: `sqrt(scaling) * Sw * (data - model)` (the lsqnonlin residuals, without penalty).
    pub residuals: Vec<f64>,
}

/// A failed evaluation: Dynare's `info(1)` and `info(4)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fail {
    pub info: i32,
    pub info4: f64,
}

fn fail(info: i32, info4: f64) -> Fail {
    Fail { info, info4 }
}

/// `objective_function.m`: `info(4) = info(2)` for the solver codes with a meaningful penalty.
fn solver_fail(info: i32, info2: f64) -> Fail {
    const MEANINGFUL: &[i32] = &[3, 4, 5, 6, 19, 20, 21, 23, 26, 81, 84, 85, 86];
    let p = if MEANINGFUL.contains(&info) && info2.is_finite() { info2 } else { 0.1 };
    fail(info, p)
}

/// The solved model at one parameter vector.
pub enum Solution {
    First(DecisionRule),
    Higher(KOrderDr),
}

impl Solution {
    pub fn first(&self) -> &DecisionRule {
        match self {
            Solution::First(d) => d,
            Solution::Higher(k) => &k.first,
        }
    }
}

#[allow(clippy::too_many_arguments)]
impl MomProblem {
    /// Builds the structural part; the method-specific parts (`rows`, `smm`, `irf`, ...) are
    /// filled by the caller.
    pub fn new(
        method: Method,
        model: &Model,
        estimated: Vec<EstimatedParam>,
        names: Vec<String>,
        sigma_e: &Mat,
        varobs: Vec<usize>,
        steady_opts: SteadyOptions,
        ys_guess: Vec<f64>,
        order: usize,
        pruning: bool,
        qz_criterium: f64,
    ) -> Self {
        let solver = FirstOrderSolver::new(
            &model.lead_lag_incidence,
            model.n_exo(),
            SolverOptions { qz_criterium, ..Default::default() },
        );
        let deriv = (order >= 2).then(|| model.dynamic_derivatives(order));
        let n = estimated.len();
        let nobs = varobs.len();
        MomProblem {
            method,
            model: model.clone(),
            order,
            pruning,
            solver,
            deriv,
            pderivs: std::sync::OnceLock::new(),
            estimated,
            names,
            base_params: model.params.clone(),
            base_sigma_e: sigma_e.clone(),
            base_h: Mat::zeros(nobs, nobs),
            base_corr_h: Mat::identity(nobs, nobs),
            cal_cov_h: Vec::new(),
            me_rows: vec![None; n],
            steady_opts,
            ssm: None,
            ys_guess,
            exo_ss: vec![0.0; model.n_exo()],
            bounds: Bounds::unbounded(n),
            varobs,
            rows: Vec::new(),
            prefilter: false,
            hp_filter: 0.0,
            one_sided_hp_filter: 0.0,
            bandpass: None,
            nlags: 0,
            qz_criterium,
            schur_vec_tol: 1e-11,
            scaling: 1.0,
            penalty: None,
            priors: None,
            smm: None,
            irf: None,
        }
    }

    pub fn dim(&self) -> usize {
        self.estimated.len()
    }

    /// Port of `set_all_parameters.m` (stderr, corr, structural parameters).
    pub fn set_all_parameters(&self, x: &[f64]) -> (Vec<f64>, Mat) {
        let mut params = self.base_params.clone();
        let mut q = self.base_sigma_e.clone();
        let mut corrs = Vec::new();
        for (e, &v) in self.estimated.iter().zip(x) {
            match e.kind {
                EstimatedKind::Param(i) => params[i] = v,
                EstimatedKind::StdErr(i) => {
                    let old = q[(i, i)].sqrt();
                    for j in 0..q.ncols() {
                        if j != i && old > 0.0 {
                            q[(i, j)] *= v / old;
                            q[(j, i)] *= v / old;
                        }
                    }
                    q[(i, i)] = v * v;
                }
                EstimatedKind::Corr(i, j) => corrs.push((i, j, v)),
                // M_.H: see set_measurement_errors
                EstimatedKind::MeStdErr(_) | EstimatedKind::MeCorr(..) => {}
                EstimatedKind::Skew(_) | EstimatedKind::InitState(_) => unreachable!("rejected by mom::run"),
            }
        }
        for (i, j, c) in corrs {
            let cov = c * (q[(i, i)] * q[(j, j)]).sqrt();
            q[(i, j)] = cov;
            q[(j, i)] = cov;
        }
        (params, q)
    }

    /// The `M_.H` part of `set_all_parameters.m`: with an estimated stderr or corr of a
    /// measurement error, `H = diag(sqrt(diag(H))) * Correlation_matrix_ME * diag(sqrt(diag(H)))`
    /// with the calibrated covariances restored; otherwise the calibrated `M_.H`.
    pub fn set_measurement_errors(&self, x: &[f64]) -> Mat {
        let mut h = self.base_h.clone();
        let mut c = self.base_corr_h.clone();
        let mut set = false;
        for ((e, &v), me) in self.estimated.iter().zip(x).zip(&self.me_rows) {
            match (e.kind, me) {
                (EstimatedKind::MeStdErr(_), Some((i, _))) => {
                    h[(*i, *i)] = v * v;
                    set = true;
                }
                (EstimatedKind::MeCorr(..), Some((i, j))) => {
                    c[(*i, *j)] = v;
                    c[(*j, *i)] = v;
                    set = true;
                }
                _ => {}
            }
        }
        if set {
            let var: Vec<f64> = (0..h.nrows()).map(|i| h[(i, i)]).collect();
            for i in 0..h.nrows() {
                for j in 0..h.ncols() {
                    h[(i, j)] = if i == j { var[i] } else { c[(i, j)] * (var[i] * var[j]).sqrt() };
                }
            }
            for &(i, j, v) in &self.cal_cov_h {
                h[(i, j)] = v;
                h[(j, i)] = v;
            }
        }
        h
    }

    /// `check_bounds_and_definiteness_estimation.m`.
    fn check_bounds(&self, x: &[f64], sigma_e: &Mat) -> Result<(), Fail> {
        let b = &self.bounds;
        if x.iter().zip(&b.lb).any(|(v, l)| v < l) {
            let p = x.iter().zip(&b.lb).filter(|(v, l)| v < l).map(|(v, l)| (l - v).powi(2)).sum();
            return Err(fail(41, p));
        }
        if x.iter().zip(&b.ub).any(|(v, u)| v > u) {
            let p = x.iter().zip(&b.ub).filter(|(v, u)| v > u).map(|(v, u)| (v - u).powi(2)).sum();
            return Err(fail(42, p));
        }
        if self.estimated.iter().any(|e| matches!(e.kind, EstimatedKind::Corr(..))) {
            let (ev, _) = sym_eig(sigma_e);
            let neg: f64 = ev.iter().filter(|&&v| v < 0.0).sum();
            if neg < 0.0 {
                return Err(fail(43, neg.abs()));
            }
        }
        // `ncn > 0` or calibrated covariances of the measurement errors: ispd on
        // H(H_entries_to_check_for_positive_definiteness) (get_matrix_entries_for_psd_check.m)
        if self.estimated.iter().any(|e| matches!(e.kind, EstimatedKind::MeCorr(..))) || !self.cal_cov_h.is_empty() {
            let h = self.set_measurement_errors(x);
            let mut rows: Vec<usize> =
                (0..self.base_h.nrows()).filter(|&i| self.base_h.row(i).iter().any(|&v| v != 0.0)).collect();
            for (e, me) in self.estimated.iter().zip(&self.me_rows) {
                if let (EstimatedKind::MeStdErr(_), Some((i, _))) = (e.kind, me) {
                    rows.push(*i);
                }
            }
            rows.sort_unstable();
            rows.dedup();
            let sub = Mat::from_fn(rows.len(), rows.len(), |i, j| h[(rows[i], rows[j])]);
            if !rows.is_empty() && chol_upper(&sub).is_none() {
                let (ev, _) = sym_eig(&sub);
                return Err(fail(44, ev.iter().filter(|&&v| v < 0.0).map(|v| -v).sum()));
            }
        }
        Ok(())
    }

    /// Steady state and decision rules at `x` (`compute_decision_rules.m`); the returned
    /// parameters are `M_.params` as updated by a `steady_state_model` block that assigns
    /// parameters (`[dr, info, M_.params] = compute_decision_rules(...)`).
    pub fn solve(&self, x: &[f64]) -> Result<(Vec<f64>, Mat, Solution), Fail> {
        let (params, sigma_e) = self.set_all_parameters(x);
        self.check_bounds(x, &sigma_e)?;
        let (ys, params) = crate::identification::steady_state_with_params(
            &self.model,
            self.ssm.as_ref(),
            &params,
            &self.ys_guess,
            &self.exo_ss,
            &self.steady_opts,
        )
        .map_err(|e| solver_fail(e.info, e.info2))?;
        let x0 = self.exo_ss.clone();
        let g1 = self.model.dynamic_jacobian(&ys, &x0, &params);
        // at order 3 Dynare uses k_order_solver, whose failures are all info 9
        let dr1 = self
            .solver
            .solve(&g1, &ys)
            .map_err(|e| if self.order > 2 { fail(9, 0.1) } else { solver_fail(e.info, e.info2) })?;
        let sol = match &self.deriv {
            None => Solution::First(dr1),
            Some(d) => {
                let vals = d.eval(&ys, &x0, &params);
                let k = korder::solve(&dr1, &self.model.lead_lag_incidence, &g1, d, &vals, &sigma_e, self.order)
                    .map_err(|_| fail(9, 0.1))?;
                Solution::Higher(k)
            }
        };
        Ok((params, sigma_e, sol))
    }

    /// DR positions of the observed variables (`options_mom_.mom.obs_var`).
    pub fn obs_var(&self, dr: &DecisionRule) -> Vec<usize> {
        self.varobs.iter().map(|&v| dr.order_var.iter().position(|&o| o == v).unwrap()).collect()
    }

    fn obs_pos(&self, v: usize) -> usize {
        self.varobs.iter().position(|&o| o == v).expect("matched moment on a non-observed variable")
    }

    /// GMM: moments of the observed variables from the pruned state space.
    pub fn state_space_moments(&self, sol: &Solution, sigma_e: &Mat) -> Result<StateSpaceMoments, Fail> {
        let dr = sol.first();
        let indy = self.obs_var(dr);
        match sol {
            Solution::First(dr) => {
                first_order_moments(dr, sigma_e, &indy, self.nlags, self.qz_criterium, self.schur_vec_tol)
                    .map_err(|_| fail(9, 0.1))
            }
            Solution::Higher(k) => {
                let p = korder::pruned::pruned_state_space_system_qz(k, sigma_e, &indy, self.nlags, self.qz_criterium)
                    .map_err(|_| fail(9, 0.1))?;
                Ok(StateSpaceMoments { e_y: p.e_y, var_y: p.var_y, var_yi: p.var_yi })
            }
        }
    }

    /// SMM: simulated observables (`long x nobs`, varobs order) with the fixed shocks, plus the
    /// measurement errors `ME_shock_series * chol(H)` (on the observables with non-zero
    /// variance) when `diag(M_.H)` is not zero.
    pub fn simulate(&self, sol: &Solution, sigma_e: &Mat, h: &Mat) -> Result<Mat, Fail> {
        let smm = self.smm.as_ref().expect("SMM setup");
        let scaled = scale_shocks(&smm.shock_series, sigma_e).ok_or(fail(180, 0.1))?;
        let dr = sol.first();
        let ys = dr.ys.clone();
        let sim = match sol {
            Solution::First(dr) => crate::analysis::simulate::simult(dr, &ys, &scaled),
            Solution::Higher(k) => korder::simulate::simult(k, &ys, &scaled, self.pruning),
        };
        let nc = sim.ncols();
        if sim.iter().any(|v| !v.is_finite()) {
            return Err(fail(180, 0.1));
        }
        let mut y = Mat::from_fn(smm.long, self.varobs.len(), |t, j| sim[(self.varobs[j], nc - smm.long + t)]);
        if (0..h.nrows()).any(|i| h[(i, i)] != 0.0) {
            y += scale_shocks(&smm.me_shock_series, h).ok_or(fail(180, 0.1))?;
        }
        Ok(y)
    }

    /// Moments of a `T x nobs` series of the observables as `get_data_moments` on the raw
    /// series (first moments) and on the demeaned (`prefilter`) / HP-filtered series.
    pub fn series_moments(&self, y: &Mat, demean: bool) -> (Vec<f64>, Mat) {
        let col = |v: usize| self.obs_pos(v);
        let first: Vec<MatchedMoment> = self.rows.iter().filter(|m| is_first_moment(m)).cloned().collect();
        let other: Vec<MatchedMoment> = self.rows.iter().filter(|m| !is_first_moment(m)).cloned().collect();
        let (m1, d1) = get_data_moments(y, &col, &first);
        let mut yf = y.clone();
        if demean {
            for c in 0..yf.ncols() {
                let (s, n) =
                    yf.column(c).iter().filter(|v| !v.is_nan()).fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
                let m = s / n as f64;
                yf.column_mut(c).add_scalar_mut(-m);
            }
        }
        if self.hp_filter != 0.0 {
            yf = crate::analysis::simulate::sample_hp_filter(&yf, self.hp_filter);
        } else if self.one_sided_hp_filter != 0.0 {
            yf = crate::analysis::q2::one_sided_hp_filter(&yf, self.one_sided_hp_filter).1;
        } else if let Some([high, low]) = self.bandpass {
            // get_filtered_time_series.m: Baxter-King cycle padded with K NaN rows at each end
            let k = 12;
            let c = crate::analysis::q2::baxter_king_filter(&yf, high, low, k).expect("bandpass_filter");
            yf = Mat::from_fn(yf.nrows(), yf.ncols(), |t, j| {
                if t < k || t >= yf.nrows() - k { f64::NAN } else { c[(t - k, j)] }
            });
        }
        let (m2, d2) = get_data_moments(&yf, &col, &other);
        let mut moments = vec![0.0; self.rows.len()];
        let mut m_data = Mat::zeros(y.nrows(), self.rows.len());
        let (mut a, mut b) = (0, 0);
        for (j, m) in self.rows.iter().enumerate() {
            if is_first_moment(m) {
                moments[j] = m1[a];
                m_data.set_column(j, &d1.column(a));
                a += 1;
            } else {
                moments[j] = m2[b];
                m_data.set_column(j, &d2.column(b));
                b += 1;
            }
        }
        (moments, m_data)
    }

    /// IRF matching: the model IRFs of the observables (`irf x obs_nbr x exo_nbr`, column-major).
    pub fn model_irfs(&self, sol: &Solution, sigma_e: &Mat) -> Result<Vec<f64>, Fail> {
        let s = self.irf.as_ref().expect("IRF setup");
        let h = s.targets.horizon;
        let nobs = self.varobs.len();
        let nexo = self.model.n_exo();
        let cs = crate::analysis::vardec::lower_cholesky(sigma_e, s.add_tiny_number_to_cholesky);
        let mut out = vec![0.0; h * nobs * nexo];
        for i in 0..nexo {
            let ci = cs[(i, i)];
            let y = match sol {
                Solution::First(dr) => {
                    let e: Vec<f64> = cs.column(i).iter().copied().collect();
                    let mut y = crate::analysis::irf::irf(dr, &e, h);
                    if s.relative_irf {
                        y = y * 100.0 / ci;
                    }
                    y
                }
                Solution::Higher(k) => {
                    let e: Vec<f64> = if s.relative_irf {
                        cs.column(i).iter().map(|v| v / ci / 100.0).collect()
                    } else {
                        cs.column(i).iter().copied().collect()
                    };
                    let draws: Vec<Mat> =
                        s.draws.iter().map(|d| scale_shocks(d, sigma_e).unwrap_or_else(|| d.clone())).collect();
                    let y = korder::simulate::girf(k, &e, h, s.drop, &draws, self.pruning);
                    if y.iter().any(|v| v.is_nan()) && !self.pruning {
                        return Err(fail(181, 0.1));
                    }
                    y
                }
            };
            for (o, &v) in self.varobs.iter().enumerate() {
                for t in 0..h {
                    out[t + h * (o + nobs * i)] = y[(v, t)];
                }
            }
        }
        Ok(out)
    }

    /// Model moments (or IRFs) at `x`; also returns the full IRF array for IRF matching.
    pub fn model_moments(&self, x: &[f64]) -> Result<(Vec<f64>, Option<Vec<f64>>), Fail> {
        let (_, sigma_e, sol) = self.solve(x)?;
        match self.method {
            Method::Gmm => {
                let ss = self.state_space_moments(&sol, &sigma_e)?;
                Ok((gmm_model_moments(&ss, &self.rows, &|v| self.obs_pos(v), self.prefilter), None))
            }
            Method::Smm => {
                let y = self.simulate(&sol, &sigma_e, &self.set_measurement_errors(x))?;

                Ok((self.series_moments(&y, self.prefilter).0, None))
            }
            Method::IrfMatching => {
                let all = self.model_irfs(&sol, &sigma_e)?;
                let idx = &self.irf.as_ref().unwrap().targets.index;
                Ok((idx.iter().map(|&i| all[i]).collect(), Some(all)))
            }
        }
    }

    /// `mom.objective_function`: `w` is the upper Cholesky factor `Sw` of the weighting matrix
    /// for GMM/SMM and the weighting matrix `W` itself for IRF matching.
    pub fn evaluate(&self, x: &[f64], data: &[f64], w: &Mat) -> Result<Eval, Fail> {
        let (mm, _) = self.model_moments(x)?;
        let diff: Vec<f64> = data.iter().zip(&mm).map(|(d, m)| d - m).collect();
        let dv = Mat::from_column_slice(diff.len(), 1, &diff);
        match self.method {
            Method::IrfMatching => {
                let q = mul(&mul(&dv, true, w, false), false, &dv, false)[(0, 0)];
                let s = self.irf.as_ref().unwrap();
                let n = mm.len() as f64;
                let lnlik = n / 2.0 * (1.0 / 2.0 / std::f64::consts::PI).ln() - 0.5 * s.winv_logdet - 0.5 * q;
                if lnlik.is_infinite() {
                    return Err(fail(50, 0.1));
                }
                if lnlik.is_nan() {
                    return Err(fail(45, 0.1));
                }
                let lnprior = self.priors.as_ref().map_or(0.0, |p| log_prior_density(x, p));
                if lnprior.is_infinite() {
                    return Err(fail(40, 0.1));
                }
                if lnprior.is_nan() {
                    return Err(fail(47, 0.1));
                }
                Ok(Eval { fval: -(lnlik + lnprior), q, model_moments: mm, residuals: Vec::new() })
            }
            _ => {
                let sws = w * self.scaling.sqrt();
                let r = mul(&sws, false, &dv, false);
                let residuals: Vec<f64> = r.iter().copied().collect();
                let q: f64 = residuals.iter().map(|v| v * v).sum();
                if q.is_infinite() {
                    return Err(fail(177, 0.1));
                }
                if q.is_nan() {
                    return Err(fail(178, 0.1));
                }
                let mut fval = q;
                if let Some((p1, p2)) = &self.penalty {
                    fval += x.iter().zip(p1).zip(p2).map(|((xi, m), s)| (xi - m) * (xi - m) / (s * s)).sum::<f64>();
                }
                Ok(Eval { fval, q, model_moments: mm, residuals })
            }
        }
    }

    /// The lsqnonlin residual vector (`vector_output`): residuals, then `(x - p1) ./ p2` under
    /// the penalized estimator; `huge_number` entries on failure.
    pub fn residual_vector(&self, x: &[f64], data: &[f64], sw: &Mat, huge_number: f64) -> Vec<f64> {
        match self.evaluate(x, data, sw) {
            Ok(e) => {
                let mut r = e.residuals;
                if let Some((p1, p2)) = &self.penalty {
                    r.extend(x.iter().zip(p1).zip(p2).map(|((xi, m), s)| (xi - m) / s));
                }
                r
            }
            Err(_) => vec![huge_number; data.len() + if self.penalty.is_some() { x.len() } else { 0 }],
        }
    }
}

/// `shocks(:, i) * chol(Sigma_e(i, i))` on the shocks with non-zero variance.
pub fn scale_shocks(z: &Mat, sigma_e: &Mat) -> Option<Mat> {
    let ne = sigma_e.nrows();
    let idx: Vec<usize> = (0..ne).filter(|&i| sigma_e[(i, i)] != 0.0).collect();
    let sub = Mat::from_fn(idx.len(), idx.len(), |i, j| sigma_e[(idx[i], idx[j])]);
    let chol = chol_upper(&sub)?;
    let zs = mul(&z.select_columns(&idx), false, &chol, false);
    let mut out = Mat::zeros(z.nrows(), ne);
    for (c, &i) in idx.iter().enumerate() {
        out.set_column(i, &zs.column(c));
    }
    Some(out)
}

/// `mom.objective_function` as an [`Objective`] for the optimizers.
pub struct MomObjective<'a> {
    pub p: &'a MomProblem,
    pub data: &'a [f64],
    pub w: &'a Mat,
    /// `analytic_jacobian` (gradient-based optimizers only).
    pub analytic: bool,
}

impl Objective for MomObjective<'_> {
    fn dim(&self) -> usize {
        self.p.dim()
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        match self.p.evaluate(x, self.data, self.w) {
            Ok(e) => (e.fval, 0),
            Err(f) => (f64::INFINITY, f.info),
        }
    }
    fn gradient(&self, x: &[f64]) -> Option<Vec<f64>> {
        if self.analytic && self.p.method == Method::Gmm {
            self.p.objective_gradient(x, self.data, self.w)
        } else {
            None
        }
    }
    fn has_gradient(&self) -> bool {
        self.analytic && self.p.method == Method::Gmm
    }
    fn eval_full(&self, x: &[f64], _contributions: bool) -> Evaluation {
        match self.p.evaluate(x, self.data, self.w) {
            Ok(e) => Evaluation { fval: e.fval, info: 0, info4: 0.0, contributions: None },
            Err(f) => Evaluation { fval: f64::INFINITY, info: f.info, info4: f.info4, contributions: None },
        }
    }
}

/// Error for an evaluation that fails where Dynare errors out.
pub fn fail_err(f: Fail, what: &str) -> anyhow::Error {
    anyhow!("method_of_moments: {what} (info = {})", f.info)
}
