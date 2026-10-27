// SPDX-License-Identifier: GPL-3.0-or-later
//! The structural half of `dsge_likelihood.m`: estimated parameter vector ->
//! model parameters and shock covariance (`set_all_parameters.m`) -> bounds
//! check (`check_bounds_and_definiteness_estimation.m`) -> steady state
//! (`evaluate_steady_state.m`) -> first-order decision rule -> restricted
//! state space (`dynare_resolve(..., 'restrict')`).
//!
//! Everything that depends only on the model structure is built once in
//! [`DsgeModel::new`]; [`DsgeModel::resolve`] is then a pure function of the
//! parameter vector (`&self`), so MCMC chains can share one `DsgeModel`.

use anyhow::{Result, anyhow, bail};

use crate::estimation::{Bounds, Evaluation, Objective, Prior, PriorShape, estimation_bounds, log_prior_density};
use crate::kalman::{KalmanOptions, KalmanWorkspace, LikOutput, MissingData, try_compute_pinf_pstar};
use crate::linalg::Mat;
use crate::model::Model;
use crate::modfile::{Command, EstimatedKind, EstimatedParam, ModFile, OptValue, opt};
use crate::solver::{
    DecisionRule, FirstOrderSolver, SolverOptions, StateSpaceIndex, SteadyOptions, kalman_transition_matrix,
    state_space_index, steady_state_r_aux,
};

/// A failed evaluation, as Dynare's `info` vector: `info(1)` code and the
/// `info(4)` penalty that `penalty_objective_function` adds to the last finite value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalFailure {
    pub info: i32,
    pub penalty: f64,
}

/// The state space at one parameter vector.
#[derive(Debug, Clone)]
pub struct Resolved {
    /// Model parameters after `set_all_parameters`.
    pub params: Vec<f64>,
    /// Steady state (all endogenous, declaration order, aux variables included).
    pub steady_state: Vec<f64>,
    pub dr: DecisionRule,
    /// Restricted transition matrices: a_{t+1} = T a_t + R e_t.
    pub t: Mat,
    pub r: Mat,
    /// Shock covariance `M_.Sigma_e`.
    pub q: Mat,
    /// Measurement-error covariance `M_.H` (nvobs x nvobs, `varobs` order; zero without
    /// measurement errors).
    pub h: Mat,
}

pub struct DsgeModel {
    pub model: Model,
    /// The estimated parameters in Dynare's `xparam1` order.
    pub estimated: Vec<EstimatedParam>,
    pub estimated_names: Vec<String>,
    /// Priors (`bayestopt_`), `xparam1` order; empty for maximum likelihood ([`Self::ml`]).
    pub priors: Vec<Prior>,
    /// Maximum-likelihood estimation (no priors): the log prior is 0 (priordens with no
    /// pshape), the bounds are the explicit ones and there is no MCMC.
    pub ml: bool,
    /// `options_.loglinear` of the estimation: decision rules for the logs of the variables
    /// (stochastic_solvers.m), constant = log(steady state) in the measurement equation.
    pub loglinear: bool,
    /// Endogenous variables that are not aux variables for lagged/leaded exogenous ones
    /// (`get_all_variables_but_lagged_leaded_exogenous`).
    not_exo_aux: Vec<bool>,
    /// Mode-finding bounds (`BoundsInfo`): prior support intersected with explicit bounds.
    pub bounds: Bounds,
    /// Calibrated parameters and shock covariance that `set_all_parameters` starts from.
    pub base_params: Vec<f64>,
    pub base_sigma_e: Mat,
    /// Calibrated measurement-error covariance `M_.H` (nvobs x nvobs, `varobs` order).
    pub base_h: Mat,
    /// `M_.Correlation_matrix` and `M_.Correlation_matrix_ME` (see `ModFile::corr_e`).
    pub base_corr: Mat,
    pub base_corr_h: Mat,
    /// Rows of `M_.H` of each estimated parameter that is a measurement error (`xparam1`
    /// order; `(k, k)` for a stderr, `(k1, k2)` for a corr), `None` otherwise
    /// (`estim_params_.nvn/ncn_observable_correspondence`).
    me_rows: Vec<Option<(usize, usize)>>,
    /// `estim_params_.calibrated_covariances` / `calibrated_covariances_ME`
    /// (check_for_calibrated_covariances.m): calibrated off-diagonal entries `(i, j, value)`
    /// (lower triangle) that set_all_parameters restores after the diag(sd)*Corr*diag(sd) rebuild.
    cal_cov: Vec<(usize, usize, f64)>,
    cal_cov_h: Vec<(usize, usize, f64)>,
    /// `H_entries_to_check_for_positive_definiteness` / Sigma_e's when
    /// check_bounds_and_definiteness_estimation runs the test (ncn > 0 / ncx > 0).
    h_check: Option<Vec<usize>>,
    /// [`initial_vector`] of the mod file (Err: message).
    x_default: std::result::Result<Vec<f64>, String>,
    q_check: Option<Vec<usize>>,
    /// Observed variables as endogenous indices, `varobs` order.
    pub varobs: Vec<usize>,
    pub steady_opts: SteadyOptions,
    /// `oo_.steady_state` after the `steady` command (calibration), aux variables set.
    pub ys_calib: Vec<f64>,
    /// `oo_.exo_steady_state` (the exogenous values of `initval`/`endval`): the steady state
    /// and the Jacobians are evaluated there (resol.m, stochastic_solvers.m).
    pub exo_ss: Vec<f64>,
    /// `~options_.steadystate.nocheck`: check the residuals of a `steady_state_model`
    /// solution at every evaluation (resol.m, info 19).
    pub steady_check: bool,
    /// The initial guess every likelihood evaluation starts `evaluate_steady_state` from:
    /// `oo_.steady_state` as `dynare_estimation_init` leaves it (steady state at the
    /// initial parameter vector, solved from `ys_calib`).
    pub ys_guess: Vec<f64>,
    solver: FirstOrderSolver,
    /// `steady_state_model` block, if any (closed-form steady state, stream D).
    pub steady_state_model: Option<crate::deterministic::SteadyStateModel>,
    /// Restriction indices; they depend only on the DR ordering, so they are fixed.
    pub ss_index: StateSpaceIndex,
    /// [stream X] `observation_trends`: trend coefficient of each observable (`varobs` order).
    pub trend_coeffs: Vec<Option<crate::model::CompiledExpr>>,
    /// ---- estimation completion ---- `estimate_initial_states_endogenous_prior`: the estimated
    /// initial states as (position in `xparam1`, endogenous index); empty otherwise.
    pub init_states: Vec<(usize, usize)>,
}

impl DsgeModel {
    /// `x0`: the initial parameter vector of the estimation (estimated_params init values,
    /// or the mode file's `xparam1`); `None` = the `init` column of `estimated_params`.
    pub fn new(mf: &ModFile, x0: Option<&[f64]>) -> Result<DsgeModel> {
        let model = mf.model.clone();
        let estimated = mf.estimated_params_dynare_order();
        let estimated_names: Vec<String> = estimated.iter().map(|e| mf.estimated_param_name(e)).collect();
        // Maximum likelihood: no estimated parameter has a prior (bayestopt_.pshape all 0).
        let ml = !estimated.is_empty() && estimated.iter().all(|e| e.prior.is_none());
        if !ml && estimated.iter().any(|e| e.prior.is_none()) {
            bail!(
                "estimated_params: either all parameters have a prior (Bayesian estimation) or none (maximum likelihood)"
            );
        }
        let priors = if ml {
            Vec::new()
        } else {
            estimated
                .iter()
                .zip(&estimated_names)
                .map(|(e, name)| {
                    let shape = e.prior.and_then(|p| PriorShape::from_code(p.dynare_code())).unwrap();
                    Prior::new(shape, e.mean, e.sd, e.p3, e.p4).map_err(|err| anyhow!("{name}: {err}"))
                })
                .collect::<Result<Vec<_>>>()?
        };
        let explicit: Vec<(Option<f64>, Option<f64>)> = estimated
            .iter()
            .map(|e| match e.kind {
                EstimatedKind::Corr(..) | EstimatedKind::MeCorr(..) => {
                    (Some(e.lb.unwrap_or(-1.0).max(-1.0)), Some(e.ub.unwrap_or(1.0).min(1.0)))
                }
                // set_prior.m: skew bounds within the skew normal's theoretical bound
                EstimatedKind::Skew(_) => {
                    let b = crate::kalman::skewed::sn_skewness_bound();
                    (Some(e.lb.unwrap_or(-0.995).clamp(-b, b)), Some(e.ub.unwrap_or(0.995).clamp(-b, b)))
                }
                _ => (e.lb, e.ub),
            })
            .collect();
        // dynare_estimation_init.m: without priors the bounds are the explicit ones.
        let bounds = if ml {
            Bounds {
                lb: explicit.iter().map(|b| b.0.unwrap_or(f64::NEG_INFINITY)).collect(),
                ub: explicit.iter().map(|b| b.1.unwrap_or(f64::INFINITY)).collect(),
            }
        } else {
            estimation_bounds(&priors, &explicit, prior_trunc(mf))
        };
        let varobs = mf
            .varobs
            .iter()
            .map(|v| model.endo_index(v).ok_or_else(|| anyhow!("varobs {v} is not an endogenous variable")))
            .collect::<Result<Vec<_>>>()?;
        let steady_opts = steady_options(mf);
        let solver = FirstOrderSolver::new(&model.lead_lag_incidence, model.n_exo(), SolverOptions::default());

        // The `steady` command at the calibration, from initval.
        let n = model.n_endo();
        let mut ys0 = vec![0.0; n];
        for &(i, v) in &mf.initval {
            ys0[i] = v;
        }
        let x_ss = mf.exo_steady_state();
        model.set_auxiliary_variables(&mut ys0, &x_ss, &model.params);
        let ssm = crate::deterministic::SteadyStateModel::from_modfile(mf)?;
        let nocheck = steady_flag(mf, "nocheck");
        // dynare_estimation_init.m: no residual check at the initial parameters with
        // diffuse_filter or nocheck
        let diffuse = mf.options.active && opt(&mf.options.estimation_options(), "diffuse_filter").is_some();
        let init_check = !(nocheck || diffuse);
        let calib = solve_steady_at(&model, ssm.as_ref(), &model.params, &ys0, &x_ss, &steady_opts, !nocheck);
        let ys_calib = match calib {
            Ok((ys, _)) => ys,
            // dynare_estimation_init.m evaluates the steady state at the initial parameters
            // only, from oo_.steady_state: the calibration need not have one (e.g. estimated
            // parameters left uncalibrated)
            Err(_) if mf.options.active && !mf.options.steady_ran => ys0.clone(),
            Err(e) => bail!("steady state at the calibration failed (info {})", e.info),
        };

        let me_rows: Vec<Option<(usize, usize)>> = estimated.iter().map(|e| me_rows_of(mf, e)).collect();
        // get_matrix_entries_for_psd_check.m
        let nonzero_rows = |m: &Mat, extra: Vec<usize>| {
            let mut v: Vec<usize> = (0..m.nrows()).filter(|&i| m.row(i).iter().any(|&x| x != 0.0)).collect();
            v.extend(extra);
            v.sort_unstable();
            v.dedup();
            v
        };
        let q_check = estimated.iter().any(|e| matches!(e.kind, EstimatedKind::Corr(..))).then(|| {
            let stds =
                estimated.iter().filter_map(|e| if let EstimatedKind::StdErr(i) = e.kind { Some(i) } else { None });
            nonzero_rows(&mf.sigma_e, stds.collect())
        });
        // check_for_calibrated_covariances.m: Sigma_e's are recorded only when a corr among shocks is
        // estimated (ncx > 0); M_.H's always. Entries that are themselves estimated are dropped.
        let calibrated = |m: &Mat, est: &[(usize, usize)]| -> Vec<(usize, usize, f64)> {
            let mut v = Vec::new();
            for j in 0..m.ncols() {
                for i in j + 1..m.nrows() {
                    if m[(i, j)] != 0.0 && !est.iter().any(|&(a, b)| (a, b) == (i, j) || (b, a) == (i, j)) {
                        v.push((i, j, m[(i, j)]));
                    }
                }
            }
            v
        };
        let est_corr: Vec<(usize, usize)> = estimated
            .iter()
            .filter_map(|e| if let EstimatedKind::Corr(i, j) = e.kind { Some((i, j)) } else { None })
            .collect();
        let cal_cov = if est_corr.is_empty() { Vec::new() } else { calibrated(&mf.sigma_e, &est_corr) };
        let est_corr_h: Vec<(usize, usize)> = estimated
            .iter()
            .zip(&me_rows)
            .filter_map(|(e, r)| if matches!(e.kind, EstimatedKind::MeCorr(..)) { *r } else { None })
            .collect();
        let cal_cov_h = calibrated(&mf.h, &est_corr_h);
        // check_bounds_and_definiteness_estimation.m: H is checked when ncn > 0 or
        // estim_params_.calibrated_covariances_ME exists (calibrated ME covariances, even with
        // only ME standard errors estimated)
        let h_check = (!est_corr_h.is_empty() || !cal_cov_h.is_empty()).then(|| {
            let stds = estimated.iter().zip(&me_rows).filter(|(e, _)| matches!(e.kind, EstimatedKind::MeStdErr(_)));
            nonzero_rows(&mf.h, stds.filter_map(|(_, r)| r.map(|r| r.0)).collect())
        });
        let mut this = DsgeModel {
            cal_cov,
            cal_cov_h,
            base_params: model.params.clone(),
            base_sigma_e: mf.sigma_e.clone(),
            base_h: mf.h.clone(),
            base_corr: mf.corr_e.clone(),
            base_corr_h: mf.corr_h.clone(),
            x_default: initial_vector(mf).map_err(|e| e.to_string()),
            me_rows,
            h_check,
            q_check,
            model,
            estimated,
            estimated_names,
            priors,
            ml,
            loglinear: if mf.options.active { mf.options.loglinear } else { estimation_flag(mf, "loglinear") },
            not_exo_aux: {
                let mut v = vec![true; mf.model.n_endo()];
                for a in &mf.model.aux_vars {
                    if a.ty == 2 || a.ty == 3 {
                        v[a.endo_index] = false;
                    }
                }
                v
            },
            bounds,
            varobs,
            steady_opts,
            ys_guess: ys_calib.clone(),
            ys_calib,
            exo_ss: x_ss.clone(),
            steady_check: !nocheck,
            solver,
            steady_state_model: ssm,
            ss_index: StateSpaceIndex::default(),
            trend_coeffs: mf
                .varobs
                .iter()
                .map(|v| {
                    mf.observation_trends
                        .iter()
                        .find(|(n, _)| n == v)
                        .map(|(_, e)| mf.model.compile_expr(e))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?,
            init_states: Vec::new(), // ---- estimation completion
        };
        this.init_states = crate::estimation::init_states::positions(&this); // ---- estimation completion
        let x0 = match x0 {
            Some(x) => x.to_vec(),
            None => this.init_vector()?,
        };
        if x0.len() != this.estimated.len() {
            bail!("initial vector has {} entries, {} parameters are estimated", x0.len(), this.estimated.len());
        }
        // dynare_estimation_init: steady state at the initial parameters, from the calibrated one.
        let (params, _) = this.set_all_parameters(&x0);
        // initial_estimation_checks -> check_steady_state_changes_parameters then solves once
        // more from that result; with a singular static Jacobian (unit roots) the second pass
        // moves the solution by ~1e-8, so both are needed to land on Dynare's oo_.steady_state.
        let solve = |guess: &[f64], check: bool| {
            let ssm = this.steady_state_model.as_ref();
            solve_steady_at(&this.model, ssm, &params, guess, &x_ss, &this.steady_opts, check)
                .map(|(ys, _)| ys)
                .map_err(|e| anyhow!("steady state at the initial parameters failed (info {})", e.info))
        };
        let ys_init = solve(&this.ys_calib, init_check)?;
        this.ys_guess = solve(&ys_init, this.steady_check)?;
        // The restriction indices, from a first solution.
        let g1 = this.model.dynamic_jacobian(&this.ys_guess, &x_ss, &params);
        let dr = this
            .solver
            .solve(&g1, &this.ys_guess)
            .map_err(|e| anyhow!("no solution at the initial parameters: {e}"))?;
        this.ss_index = state_space_index(&dr, &this.varobs);
        Ok(this)
    }

    /// [stream X] `compute_trend_coefficients.m`: `trend_coeff * (first_obs .. first_obs+nobs-1)`
    /// (demeaned with `prefilter`), nvobs x nobs; `None` without `observation_trends`.
    pub fn trend_addition(&self, params: &[f64], first_obs: usize, nobs: usize, prefilter: bool) -> Option<Mat> {
        if self.trend_coeffs.iter().all(Option::is_none) {
            return None;
        }
        let (y0, x0) = (vec![0.0; self.model.n_endo()], vec![0.0; self.model.n_exo()]);
        let c: Vec<f64> =
            self.trend_coeffs.iter().map(|e| e.as_ref().map_or(0.0, |e| e.eval(&y0, &x0, params))).collect();
        let mean_t = if prefilter { first_obs as f64 + (nobs as f64 - 1.0) / 2.0 } else { 0.0 };
        Some(Mat::from_fn(c.len(), nobs, |i, t| c[i] * ((first_obs + t) as f64 - mean_t)))
    }

    pub fn n_estimated(&self) -> usize {
        self.estimated.len()
    }

    // ---- estimation completion ----
    /// Rows of `M_.H` of each estimated parameter (`(k, k)` for a measurement-error stderr,
    /// `(k1, k2)` for a corr, `None` otherwise): `estim_params_.nvn/ncn_observable_correspondence`.
    pub fn me_rows(&self) -> &[Option<(usize, usize)>] {
        &self.me_rows
    }

    /// The first-order solution at `g1` / `ys` (the estimation's solver); Err: Dynare's info.
    pub fn solve_first_order(&self, g1: &Mat, ys: &[f64]) -> Result<DecisionRule, i32> {
        self.solver.solve(g1, ys).map_err(|e| e.info)
    }
    // ---- end estimation completion ----

    /// The initial vector of the mod file ([`initial_vector`]).
    pub fn init_vector(&self) -> Result<Vec<f64>> {
        self.x_default.clone().map_err(|e| anyhow!("{e}"))
    }

    /// `priordens.m` at `x`.
    pub fn log_prior(&self, x: &[f64]) -> f64 {
        if self.ml { 0.0 } else { log_prior_density(x, &self.priors) }
    }

    /// The constant added back to the (smoothed, filtered) deviations of variable `v` at
    /// steady state `ys`: `log(ys)` with `loglinear` (except for aux variables of
    /// lagged/leaded exogenous variables), else `ys` (store_smoother_results.m).
    pub fn level_constant(&self, v: usize, ys: &[f64]) -> f64 {
        if self.loglinear && self.not_exo_aux[v] { ys[v].ln() } else { ys[v] }
    }

    /// `bayestopt_.p2` (prior standard deviations; NaN for maximum likelihood).
    pub fn prior_sd(&self) -> Vec<f64> {
        if self.ml { vec![f64::NAN; self.estimated.len()] } else { self.priors.iter().map(|p| p.sd).collect() }
    }

    /// Port of `set_all_parameters.m` (stderr, corr and structural parameters), without `M_.H`.
    pub fn set_all_parameters(&self, x: &[f64]) -> (Vec<f64>, Mat) {
        let (p, q, _) = self.set_all_parameters_h(x);
        (p, q)
    }

    /// Port of `set_all_parameters.m`: model parameters, `M_.Sigma_e` and `M_.H`.
    pub fn set_all_parameters_h(&self, x: &[f64]) -> (Vec<f64>, Mat, Mat) {
        let (p, q, h, _, _) = self.set_all_parameters_full(x);
        (p, q, h)
    }

    /// [`Self::set_all_parameters_h`] with the updated `M_.Correlation_matrix` / `M_.Correlation_matrix_ME`.
    pub fn set_all_parameters_full(&self, x: &[f64]) -> (Vec<f64>, Mat, Mat, Mat, Mat) {
        let mut params = self.base_params.clone();
        let mut q = self.base_sigma_e.clone();
        let mut h = self.base_h.clone();
        let (mut cq, mut ch) = (self.base_corr.clone(), self.base_corr_h.clone());
        let (mut set_q, mut set_h) = (false, false);
        for ((e, &v), me) in self.estimated.iter().zip(x).zip(&self.me_rows) {
            match e.kind {
                EstimatedKind::MeStdErr(_) => {
                    let (i, _) = me.expect("measurement error on a non-observed variable");
                    h[(i, i)] = v * v;
                    set_h = true;
                }
                EstimatedKind::MeCorr(..) => {
                    let (i, j) = me.expect("measurement error on a non-observed variable");
                    ch[(i, j)] = v;
                    ch[(j, i)] = v;
                    set_h = true;
                }
                EstimatedKind::Param(i) => params[i] = v,
                EstimatedKind::StdErr(i) => {
                    q[(i, i)] = v * v;
                    set_q = true;
                }
                EstimatedKind::Corr(i, j) => {
                    cq[(i, j)] = v;
                    cq[(j, i)] = v;
                    set_q = true;
                }
                EstimatedKind::Skew(_) => {} // M_.Skew_e: crate::kalman::skewed (kalman_algo = 5)
                EstimatedKind::InitState(_) => {} // ---- estimation completion: M_.endo_initial_state (crate::estimation::init_states)
            }
        }
        // set_all_parameters.m: Sigma_e = diag(sd) * M_.Correlation_matrix * diag(sd) as soon as a stderr or a
        // corr is estimated; covariances set in `shocks` with `var a, b = x` are not in the correlation matrix,
        // so they are zeroed unless recorded in estim_params_.calibrated_covariances(_ME) (likewise for M_.H).
        for (m, c, set, cal) in [(&mut q, &cq, set_q, &self.cal_cov), (&mut h, &ch, set_h, &self.cal_cov_h)] {
            if set {
                let var: Vec<f64> = (0..m.nrows()).map(|i| m[(i, i)]).collect();
                for i in 0..m.nrows() {
                    for j in 0..m.ncols() {
                        m[(i, j)] = if i == j { var[i] } else { c[(i, j)] * (var[i] * var[j]).sqrt() };
                    }
                }
                for &(i, j, v) in cal.iter() {
                    m[(i, j)] = v;
                    m[(j, i)] = v;
                }
            }
        }
        (params, q, h, cq, ch)
    }

    /// `dynare_estimation_1.m:584-587`: `M_ = set_all_parameters(xparam1, estim_params_, M_)`.
    /// The point becomes the model's `M_.params`, `M_.Sigma_e`, `M_.H` and correlation matrices
    /// (the calibration later commands and later estimations start from).
    pub fn adopt_point(&mut self, x: &[f64]) {
        let (p, q, h, cq, ch) = self.set_all_parameters_full(x);
        self.base_params = p;
        self.base_sigma_e = q;
        self.base_h = h;
        self.base_corr = cq;
        self.base_corr_h = ch;
    }

    /// `M_.params`, `M_.Sigma_e`, `M_.H` and the correlation matrices of this model (after
    /// [`Self::adopt_point`]), to be written into the mod file's global state.
    pub fn global_state(&self) -> GlobalState {
        GlobalState {
            params: self.base_params.clone(),
            sigma_e: self.base_sigma_e.clone(),
            h: self.base_h.clone(),
            corr_e: self.base_corr.clone(),
            corr_h: self.base_corr_h.clone(),
        }
    }

    /// Calibrated value of the `k`-th estimated parameter (`get_all_parameters.m` on the
    /// calibration: `M_.params`, `M_.Sigma_e`, `M_.H`), NaN when it has none.
    pub fn calibrated_value(&self, k: usize) -> f64 {
        calibrated_value(&self.estimated[k], &self.base_params, &self.base_sigma_e, &self.base_h, self.me_rows[k])
    }

    /// `evaluate_steady_state` at the model parameters `params` (from `ys_guess`),
    /// as [`Self::resolve`] runs it.
    pub fn steady_state(&self, params: &[f64]) -> Result<Vec<f64>, EvalFailure> {
        self.steady_state_params(params).map(|(ys, _)| ys)
    }

    /// [`Self::steady_state`] with `M_.params` as `evaluate_steady_state` returns them
    /// (parameters assigned by the `steady_state_model` block updated).
    pub fn steady_state_params(&self, params: &[f64]) -> Result<(Vec<f64>, Vec<f64>), EvalFailure> {
        solve_steady_at(
            &self.model,
            self.steady_state_model.as_ref(),
            params,
            &self.ys_guess,
            &self.exo_ss,
            &self.steady_opts,
            self.steady_check,
        )
        .map_err(|e| EvalFailure { info: e.info, penalty: penalty(e.info, e.info2) })
    }

    /// The first-order solver options (`options_.qz_criterium`, ...).
    pub fn solver_options(&self) -> SolverOptions {
        self.solver.opts
    }

    /// Replaces the first-order solver options (`options_.qz_criterium`; used by `src/gsa`,
    /// where Dynare's `select_qz_criterium_value` gives 1-1e-6 with a stationary filter).
    pub fn set_solver_options(&mut self, opts: SolverOptions) {
        self.solver = FirstOrderSolver::new(&self.model.lead_lag_incidence, self.model.n_exo(), opts);
    }

    /// `dynare_resolve(M_, options_, dr, ys, ..., 'restrict')` at parameter vector `x`,
    /// after `check_bounds_and_definiteness_estimation` (bounds checked by the caller,
    /// which owns the prior).
    pub fn resolve(&self, x: &[f64]) -> Result<Resolved, EvalFailure> {
        let (params, q, h) = self.set_all_parameters_h(x);
        // check_bounds_and_definiteness_estimation.m: Sigma_e (ncx > 0) and H (ncn > 0)
        // and the consistency of the calibrated covariances (info 71 / 72)
        for (m, rows, cal, info) in [(&q, &self.q_check, &self.cal_cov, 43), (&h, &self.h_check, &self.cal_cov_h, 44)] {
            if let Some(rows) = rows {
                if let Some(penalty) = not_positive_definite(m, rows) {
                    return Err(EvalFailure { info, penalty });
                }
                // check_consistency_covariances.m (only with calibrated covariances)
                if !cal.is_empty() {
                    let n = m.nrows();
                    let bad = (0..n).any(|i| (0..i).any(|j| m[(i, j)].abs() > m[(i, i)].sqrt() * m[(j, j)].sqrt()));
                    if bad {
                        let penalty = cal.iter().map(|&(i, j, _)| 2.0 * m[(i, j)] * m[(i, j)]).sum();
                        return Err(EvalFailure { info: info + 28, penalty });
                    }
                }
            }
        }
        // resol.m: [dr.ys, M_.params, info] = evaluate_steady_state(...)
        let (ys, params) = self.steady_state_params(&params)?;
        if self.loglinear {
            // resol.m: the steady state must be positive (info 26), before the solver; the
            // penalty sums over all the variables
            if ys.iter().zip(&self.not_exo_aux).any(|(&v, &k)| k && v < 1e-16) {
                let penalty = ys.iter().filter(|&&v| v < 1e-16).map(|v| v * v).sum();
                return Err(EvalFailure { info: 26, penalty });
            }
        }
        let g1 = self.model.dynamic_jacobian(&ys, &self.exo_ss, &params);
        let mut dr =
            self.solver.solve(&g1, &ys).map_err(|e| EvalFailure { info: e.info, penalty: penalty(e.info, e.info2) })?;
        if self.loglinear {
            loglinear_decision_rule(&mut dr, &ys, &self.not_exo_aux);
        }
        let (t, r) = kalman_transition_matrix(&dr, &self.ss_index);
        Ok(Resolved { params, steady_state: ys, dr, t, r, q, h })
    }
}

/// stochastic_solvers.m, `options_.loglinear` (order 1): rescales ghx/ghu to the logs of the
/// variables; aux variables of lagged/leaded exogenous variables (`il`) stay in levels.
fn loglinear_decision_rule(dr: &mut DecisionRule, ys: &[f64], not_exo_aux: &[bool]) {
    let n = dr.ghx.nrows();
    for i in 0..n {
        let vi = dr.order_var[i];
        for j in 0..dr.ghx.ncols() {
            let vj = dr.state_var[j];
            let scale_col = if not_exo_aux[vj] { ys[vj] } else { 1.0 };
            let scale_row = if not_exo_aux[vi] { 1.0 / ys[vi] } else { 1.0 };
            dr.ghx[(i, j)] *= scale_row * scale_col;
        }
        if not_exo_aux[vi] {
            for k in 0..dr.ghu.ncols() {
                dr.ghu[(i, k)] /= ys[vi];
            }
        }
    }
}

/// Is `name` set (flag or non-zero value) in the last `estimation` command?
fn estimation_flag(mf: &ModFile, name: &str) -> bool {
    mf.commands
        .iter()
        .rev()
        .find_map(|c| match c {
            Command::Estimation(o, _) => Some(match opt(o, name) {
                Some(OptValue::Flag) => true,
                Some(OptValue::Num(v)) => *v != 0.0,
                _ => false,
            }),
            _ => None,
        })
        .unwrap_or(false)
}

/// Is `name` set in `options_.steady*` by the `steady` commands run so far (without the
/// driver: by any `steady` command of the file)?
fn steady_flag(mf: &ModFile, name: &str) -> bool {
    let set = |o: &crate::modfile::Options| match opt(o, name) {
        Some(OptValue::Flag) => true,
        Some(OptValue::Num(v)) => *v != 0.0,
        _ => false,
    };
    if mf.options.active {
        return set(&mf.options.steady);
    }
    mf.commands.iter().any(|c| matches!(c, Command::Steady(o) if set(o)))
}

/// Rows of `M_.H` (positions in varobs) of a measurement-error entry.
fn me_rows_of(mf: &ModFile, e: &EstimatedParam) -> Option<(usize, usize)> {
    match e.kind {
        EstimatedKind::MeStdErr(k) => mf.varobs_position(k).map(|i| (i, i)),
        EstimatedKind::MeCorr(a, b) => mf.varobs_position(a).zip(mf.varobs_position(b)),
        _ => None,
    }
}

/// `ispd.m` on `m(rows, rows)`: `None` if positive definite (chol succeeds), else the
/// penalty `sum(-negative eigenvalues)`.
fn not_positive_definite(m: &Mat, rows: &[usize]) -> Option<f64> {
    let n = rows.len();
    let a = Mat::from_fn(n, n, |i, j| m[(rows[i], rows[j])]);
    if n == 0 || a.clone().cholesky().is_some() {
        return None;
    }
    let ev = a.symmetric_eigenvalues();
    Some(ev.iter().filter(|&&l| l < 0.0).map(|l| -l).sum())
}

/// Calibrated value of an estimated parameter (`get_all_parameters.m`); `me` are the rows
/// of `M_.H` of a measurement error.
fn calibrated_value(e: &EstimatedParam, params: &[f64], sigma_e: &Mat, h: &Mat, me: Option<(usize, usize)>) -> f64 {
    match (e.kind, me) {
        (EstimatedKind::Param(i), _) => params[i],
        (EstimatedKind::StdErr(i), _) => sigma_e[(i, i)].sqrt(),
        (EstimatedKind::Corr(i, j), _) => sigma_e[(i, j)] / (sigma_e[(i, i)] * sigma_e[(j, j)]).sqrt(),
        (EstimatedKind::MeStdErr(_), Some((i, _))) => h[(i, i)].sqrt(),
        (EstimatedKind::MeCorr(..), Some((i, j))) => h[(i, j)] / (h[(i, i)] * h[(j, j)]).sqrt(),
        // M_.endo_initial_state.values = zeros (dynare_estimation_init.m); Dynare 7.2's
        // do_parameter_initialization uses it as the init-state calibration (block six).
        (EstimatedKind::InitState(_), _) => 0.0,
        _ => f64::NAN,
    }
}

/// The initial parameter vector of an estimation, `xparam1` order (set_prior.m +
/// dynare_estimation_init.m): the `estimated_params` init column, overridden by
/// `estimated_params_init`; entries still empty take the calibration with
/// `estimated_params_init(use_calibration)` (do_parameter_initialization.m), else the prior
/// mean (`bayestopt_.p1`, e.g. (a+b)/2 for a uniform prior).
pub fn initial_vector(mf: &ModFile) -> Result<Vec<f64>> {
    let order = mf.estimated_params_dynare_order();
    let mut x: Vec<Option<f64>> = order.iter().map(|e| e.init).collect();
    for ini in &mf.estimated_params_init {
        if let Some(k) = order.iter().position(|e| e.kind == ini.kind) {
            x[k] = Some(ini.value);
        }
    }
    order
        .iter()
        .zip(x)
        .map(|(e, v)| {
            let v = v.unwrap_or_else(|| {
                let calib = calibrated_value(e, &mf.model.params, &mf.sigma_e, &mf.h, me_rows_of(mf, e));
                // a zero calibrated std cannot start an inverse gamma prior (do_parameter_initialization.m)
                let ig_zero = calib == 0.0 && e.prior == Some(crate::modfile::PriorShape::InvGamma1);
                if mf.estimated_params_init_use_calibration && calib.is_finite() && !ig_zero {
                    calib
                } else {
                    prior_mean(e)
                }
            });
            if v.is_finite() { Ok(v) } else { Err(anyhow!("no initial value for {}", mf.estimated_param_name(e))) }
        })
        .collect()
}

/// `bayestopt_.p1` (a uniform's mean is (p3+p4)/2 when it is given by its bounds); NaN without prior.
fn prior_mean(e: &EstimatedParam) -> f64 {
    e.prior
        .and_then(|p| PriorShape::from_code(p.dynare_code()))
        .and_then(|shape| Prior::new(shape, e.mean, e.sd, e.p3, e.p4).ok())
        .map_or(f64::NAN, |p| p.mean)
}

/// dsge_likelihood.m: `info(4) = info(2)` for the codes whose `info(2)` measures the
/// violation (3, 4, 5, 6, 19, 20, 21, 23, 26, 81, 84, 85, 86, 401-403, 411-413), 0.1 for
/// the others (and when `info(2)` is not finite).
fn penalty(info: i32, info2: f64) -> f64 {
    let measured = matches!(info, 3..=6 | 19..=21 | 23 | 26 | 81 | 84..=86 | 401..=403 | 411..=413);
    if measured && info2.is_finite() { info2 } else { 0.1 }
}

/// `evaluate_steady_state` from `guess` at `params` (aux variables set first); also used by
/// `analysis` for the calibration (`steady`, `check`, `stoch_simul`).
pub fn solve_steady(
    model: &Model,
    ssm: Option<&crate::deterministic::SteadyStateModel>,
    params: &[f64],
    guess: &[f64],
    opts: &SteadyOptions,
) -> Result<Vec<f64>, crate::solver::SolveError> {
    let x_ss = vec![0.0; model.n_exo()];
    solve_steady_at(model, ssm, params, guess, &x_ss, opts, true).map(|(ys, _)| ys)
}

/// `evaluate_steady_state(guess, x_ss, M_, options_, check)`: the steady state at the
/// exogenous steady state `x_ss`, with `M_.params` as returned (parameters assigned by the
/// `steady_state_model` block updated, evaluate_steady_state_file.m); `check` is the residual
/// check of a `steady_state_model` solution (info 19).
pub fn solve_steady_at(
    model: &Model,
    ssm: Option<&crate::deterministic::SteadyStateModel>,
    params: &[f64],
    guess: &[f64],
    x_ss: &[f64],
    opts: &SteadyOptions,
    check: bool,
) -> Result<(Vec<f64>, Vec<f64>), crate::solver::SolveError> {
    if let Some(ssm) = ssm {
        // steadystate_flag = 2: closed form (+ residual check)
        return crate::deterministic::evaluate_steady_state(model, Some(ssm), guess, x_ss, params, opts, check);
    }
    let n = model.n_endo();
    let f = |y: &[f64]| {
        let mut r = vec![0.0; n];
        model.static_residuals(y, x_ss, params, &mut r);
        (r, model.static_jacobian(y, x_ss, params))
    };
    // residuals alone for the line searches and checks (no Jacobian thrown away)
    let r = |y: &[f64]| {
        let mut r = vec![0.0; n];
        model.static_residuals(y, x_ss, params, &mut r);
        r
    };
    let mut g = guess.to_vec();
    model.set_auxiliary_variables(&mut g, x_ss, params);
    let set_aux = |y: &mut [f64]| model.set_auxiliary_variables(y, x_ss, params);
    steady_state_r_aux(&f, &r, &g, opts, Some(&set_aux)).map(|ys| (ys, params.to_vec()))
}

/// `options_.prior_trunc` as the estimation sees it (Dynare default 0); without the driver,
/// from the file's last estimation command.
fn prior_trunc(mf: &ModFile) -> f64 {
    if mf.options.active {
        return mf.options.prior_trunc.unwrap_or(0.0);
    }
    mf.commands
        .iter()
        .rev()
        .find_map(|c| match c {
            Command::Estimation(o, _) => match opt(o, "prior_trunc") {
                Some(OptValue::Num(v)) => Some(*v),
                _ => None,
            },
            _ => None,
        })
        .unwrap_or(0.0)
}

/// `options_.solve_algo`, `options_.steady.maxit`, `options_.solve_tolf` and
/// `options_.solve_tolx` as the commands run so far left them (`solve_algo` is also written
/// by `stoch_simul` and `estimation`); without the driver, from the file's `steady` commands.
pub fn steady_options(mf: &ModFile) -> SteadyOptions {
    let mut o = SteadyOptions { linear: mf.model.linear, ..Default::default() };
    if mf.options.active {
        let g = &mf.options;
        if let Some(v) = g.solve_algo {
            o.solve_algo = v as u8;
        }
        if let Some(OptValue::Num(v)) = opt(&g.steady, "maxit") {
            o.maxit = *v as usize;
        }
        if let Some(OptValue::Num(v)) = opt(&g.steady, "tolf") {
            o.tolf = *v;
        }
        if let Some(OptValue::Num(v)) = opt(&g.steady, "tolx") {
            o.tolx = *v;
        }
        return o;
    }
    for c in &mf.commands {
        if let Command::Steady(opts) = c {
            if let Some(OptValue::Num(v)) = opt(opts, "solve_algo") {
                o.solve_algo = *v as u8;
            }
            if let Some(OptValue::Num(v)) = opt(opts, "maxit") {
                o.maxit = *v as usize;
            }
            if let Some(OptValue::Num(v)) = opt(opts, "tolf") {
                o.tolf = *v;
            }
            if let Some(OptValue::Num(v)) = opt(opts, "tolx") {
                o.tolx = *v;
            }
        }
    }
    o
}

// ---------------------------------------------------------------------------
// The posterior kernel (dsge_likelihood.m)
// ---------------------------------------------------------------------------

/// Data and filter settings of an `estimation(...)` call, as `dynare_estimation_init`
/// leaves them in `dataset_`, `dataset_info` and `options_`.
#[derive(Clone)] // ---- estimation completion ---- (mr_hessian's univariate copy)
pub struct LikelihoodData {
    /// Raw observations, pp x nobs (varobs order), NaN = missing.
    pub y: Mat,
    pub missing: MissingData,
    pub kalman: KalmanOptions,
    /// `options_.endogenous_prior` (Christiano, Trabandt and Walentin 2011; lik_init = 1 only).
    pub endogenous_prior: bool,
    /// With `prefilter`: the sample means removed from `y` (`dataset_info.descriptive.mean`),
    /// added back to the smoothed observables.
    pub prefilter_mean: Option<Vec<f64>>,
    /// [stream X] `options_.first_obs` (1-based), for the `observation_trends` time index.
    pub first_obs: usize,
    /// `heteroskedastic_filter`: `M_.heteroskedastic_shocks` for this sample.
    pub hetero: Option<crate::kalman::hetero::HeteroSpec>,
    /// `filter_initial_state` (`M_.filter_initial_state`): (endogenous index, value).
    pub filter_initial_state: Vec<(usize, f64)>,
    // ---- calibration restrictions ----
    /// `options_.endogenous_prior_restrictions` (`irf_calibration` / `moment_calibration`),
    /// checked after `dynare_resolve` (info 49); `None` without the blocks.
    pub calib_restrictions: Option<crate::gsa::calibration::Restrictions>,
    // ---- end calibration restrictions ----
    // ---- estimation completion ----
    /// `kalman_algo = 5`: calibrated skewness and the skewed filter's options (likelihood and
    /// smoother, [`crate::estimation::skewed`]).
    pub skewed: Option<crate::estimation::skewed::SkewedSpec>,
    // ---- end estimation completion ----
}

impl LikelihoodData {
    /// `y`: nobs x pp sample from the datafile (see [`crate::data::Table::sample`]).
    pub fn new(sample: &Mat, kalman: KalmanOptions) -> Self {
        let y = sample.transpose();
        LikelihoodData {
            missing: MissingData::from_data(&y),
            y,
            kalman,
            endogenous_prior: false,
            prefilter_mean: None,
            first_obs: 1,
            hetero: None,
            filter_initial_state: Vec::new(),
            calib_restrictions: None,
            skewed: None,
        }
    }

    /// `prefilter=1` (makedataset.m: `dataset_.detrend()`): removes the mean of each
    /// observable (missing values ignored).
    pub fn prefilter(&mut self) {
        let mut means = Vec::with_capacity(self.y.nrows());
        for i in 0..self.y.nrows() {
            let obs: Vec<f64> = self.y.row(i).iter().copied().filter(|v| !v.is_nan()).collect();
            let m = obs.iter().sum::<f64>() / obs.len() as f64;
            self.y.row_mut(i).add_scalar_mut(-m);
            means.push(m);
        }
        self.prefilter_mean = Some(means);
    }
}

/// Port of estimation/endogenous_prior.m: the log density of the sample variances of the
/// observables (from the first period after which none is missing) given the model's
/// unconditional variances `diag(Z Pstar Z' + H)`. `y`: demeaned data, pp x nobs.
pub fn endogenous_prior(y: &Mat, missing: &MissingData, pstar: &Mat, mf: &[usize], h: &Mat) -> f64 {
    let start = missing.no_more_missing_observations;
    let n = y.nrows();
    let ts = y.ncols() - start;
    let tf = ts as f64;
    let mut yd = Mat::from_fn(ts, n, |t, j| y[(j, start + t)]);
    for j in 0..n {
        let m = yd.column(j).sum() / tf;
        yd.column_mut(j).add_scalar_mut(-m);
    }
    let fhat: Vec<f64> = (0..n).map(|j| yd.column(j).norm_squared() / tf).collect();
    let hmat = Mat::from_fn(n, ts, |j, t| yd[(t, j)] * yd[(t, j)] - fhat[j]);
    let cov = |lag: usize| {
        let mut c = Mat::zeros(n, n);
        for t in lag..ts {
            c += hmat.column(t) * hmat.column(t - lag).transpose();
        }
        c / (ts - lag) as f64
    };
    let (c0, c1, c2) = (cov(0), cov(1), cov(2));
    let shat = &c0 + (&c1 + c1.transpose()) * (1.0 - 1.0 / 3.0) + (&c2 + c2.transpose()) * (1.0 - 2.0 / 3.0);
    let d = crate::linalg::Vector::from_fn(n, |j, _| fhat[j] - (pstar[(mf[j], mf[j])] + h[(j, j)]));
    let logdet = shat.clone().determinant().ln();
    let sd = crate::linalg::solve(&shat, &Mat::from_column_slice(n, 1, d.as_slice()))
        .expect("endogenous_prior: singular Shat");
    0.5 * n as f64 * (tf / (2.0 * std::f64::consts::PI)).ln() - 0.5 * logdet - 0.5 * tf * d.dot(&sd.column(0))
}

/// `dsge_likelihood.m` as an [`Objective`]: `fval = -(loglik + logprior)`.
pub struct DsgePosterior<'a> {
    pub dsge: &'a DsgeModel,
    pub data: &'a LikelihoodData,
    /// `BoundsInfo`: [`DsgeModel::bounds`] for mode finding; the MCMC passes the prior
    /// support instead (`set_mcmc_prior_bounds.m`), as Dynare does.
    pub bounds: &'a Bounds,
}

/// Everything one evaluation produces (for reporting, the smoother, tests).
pub struct Evaluated {
    pub fval: f64,
    pub minus_loglik: f64,
    pub log_prior: f64,
    pub lik: LikOutput,
    pub resolved: Resolved,
}

thread_local! {
    static KALMAN_WS: std::cell::RefCell<Option<KalmanWorkspace>> = const { std::cell::RefCell::new(None) };
}

impl DsgePosterior<'_> {
    /// One evaluation of `dsge_likelihood.m`; `Err` carries `info(1)` and `info(4)`.
    pub fn evaluate(&self, x: &[f64]) -> Result<Evaluated, EvalFailure> {
        let b = self.bounds;
        // check_bounds_and_definiteness_estimation.m
        let below: f64 = x.iter().zip(&b.lb).filter(|(v, l)| v < l).map(|(v, l)| (l - v).powi(2)).sum();
        if x.iter().zip(&b.lb).any(|(v, l)| v < l) {
            return Err(EvalFailure { info: 41, penalty: below });
        }
        if x.iter().zip(&b.ub).any(|(v, u)| v > u) {
            let above = x.iter().zip(&b.ub).filter(|(v, u)| v > u).map(|(v, u)| (v - u).powi(2)).sum();
            return Err(EvalFailure { info: 42, penalty: above });
        }
        let res = self.dsge.resolve(x)?;
        // ---- calibration restrictions ----
        if let Some(rs) = &self.data.calib_restrictions {
            crate::gsa::calibration::likelihood_check(self.dsge, &res, rs)?;
        }
        // ---- end calibration restrictions ----
        let idx = &self.dsge.ss_index;
        let opts = &self.data.kalman;

        // Y = data - steady state of the observables (options_.noconstant = false; its log
        // with loglinear).
        let mut y = self.data.y.clone();
        for (i, &v) in idx.mfys.iter().enumerate() {
            let c = if self.dsge.loglinear { res.steady_state[v].ln() } else { res.steady_state[v] };
            for j in 0..y.ncols() {
                y[(i, j)] -= c;
            }
        }
        // [stream X] observation_trends: Y = data - constant - trend
        if let Some(tr) =
            self.dsge.trend_addition(&res.params, self.data.first_obs, y.ncols(), self.data.prefilter_mean.is_some())
        {
            y -= tr;
        }
        // ---- estimation completion: estimate_initial_states_endogenous_prior (dsge_likelihood.m)
        let init_states = if self.dsge.init_states.is_empty() {
            None
        } else {
            Some(crate::estimation::init_states::kalman_init(self.dsge, &res, x, opts)?)
        };
        let (pstar, pinf) = match opts.lik_init {
            _ if init_states.is_some() => {
                let p = init_states.as_ref().unwrap().pstar.clone();
                let z = Mat::zeros(p.nrows(), p.ncols());
                (p, z)
            }
            3 => try_compute_pinf_pstar(&idx.mf, &res.t, &res.r, &res.q, opts.qz_criterium, &idx.restrict_columns)
                .map_err(|_| EvalFailure { info: 50, penalty: 0.1 })?,
            1 => {
                let p = crate::kalman::lyapunov::lyapunov_solver_opts(&res.t, &res.r, &res.q, opts) // lyapunov option
                    .map_err(|_| EvalFailure { info: 50, penalty: 0.1 })?;
                let z = Mat::zeros(p.nrows(), p.ncols());
                (p, z)
            }
            // [stream X] lik_init 2/4/5
            k if crate::kalman::lik_init::handles(k) => {
                let p = crate::kalman::lik_init::initial_pstar(
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
                .map_err(|_| EvalFailure { info: 50, penalty: 0.1 })?;
                let z = Mat::zeros(p.nrows(), p.ncols());
                (p, z)
            }
            k => panic!("lik_init = {k} is not supported"),
        };
        let lik = KALMAN_WS.with(|cell| {
            let mut cell = cell.borrow_mut();
            let (mm, pp, rr) = (res.t.nrows(), y.nrows(), res.q.nrows());
            let ws = match cell.as_mut() {
                Some(ws) if ws.dims() == (mm, pp, rr) => ws,
                _ => cell.insert(KalmanWorkspace::new(mm, pp, rr)),
            };
            ws.set_heteroskedastic(self.data.hetero.as_ref().map(|hs| hs.qvec(&res.q, y.ncols())));
            ws.set_initial_state((!self.data.filter_initial_state.is_empty()).then(|| {
                crate::estimation::q3::kalman_a0(self.dsge, &res, &self.dsge.ss_index, &self.data.filter_initial_state)
            }));
            if let Some(is) = &init_states {
                ws.set_initial_state(Some(is.a_tm1.clone())); // ---- estimation completion
            }
            ws.likelihood(&y, &self.data.missing, &idx.mf, &res.t, &res.r, &res.q, Some(&res.h), &pstar, &pinf, opts)
        });
        if lik.info != 0 {
            return Err(EvalFailure { info: lik.info, penalty: 0.1 });
        }
        let log_prior = self.dsge.log_prior(x);
        if log_prior.is_nan() {
            return Err(EvalFailure { info: 47, penalty: 0.1 });
        }
        if !log_prior.is_finite() {
            return Err(EvalFailure { info: 40, penalty: 0.1 });
        }
        // dsge_likelihood.m: fval = likelihood - lnprior - lnpriormom
        let lnpriormom = if self.data.endogenous_prior {
            endogenous_prior(&y, &self.data.missing, &pstar, &idx.mf, &res.h)
        } else {
            0.0
        };
        let init_shift = init_states.as_ref().map_or(0.0, |is| is.fval_shift); // ---- estimation completion
        Ok(Evaluated {
            fval: lik.minus_loglik - log_prior - lnpriormom - init_shift,
            minus_loglik: lik.minus_loglik,
            log_prior,
            lik,
            resolved: res,
        })
    }
}

impl Objective for DsgePosterior<'_> {
    fn dim(&self) -> usize {
        self.dsge.n_estimated()
    }

    fn eval(&self, x: &[f64]) -> (f64, i32) {
        let r = match self.evaluate(x) {
            Ok(e) => (e.fval, 0),
            Err(f) => (f64::INFINITY, f.info),
        };
        crate::progress::eval(r.0);
        r
    }

    fn init_state_ops(&self) -> Option<&dyn crate::estimation::init_states::InitStateOps> {
        (!self.dsge.init_states.is_empty()).then_some(self as &dyn crate::estimation::init_states::InitStateOps) // ---- estimation completion
    }

    fn eval_full(&self, x: &[f64], contributions: bool) -> Evaluation {
        let r = self.evaluate(x);
        crate::progress::eval(r.as_ref().map_or(f64::INFINITY, |e| e.fval));
        match r {
            Ok(e) => {
                // DLIK = [-lnprior; lik(start:end)]
                let contributions = contributions.then(|| {
                    let start = self.data.kalman.presample;
                    // ---- estimation completion ---- univariate filters: lik(start:end,:)
                    if let Some(lo) = &e.lik.lik_by_obs {
                        let pp = self.data.y.nrows();
                        return std::iter::once(-e.log_prior)
                            .chain(lo[(start * pp).min(lo.len())..].iter().copied())
                            .collect();
                    }
                    std::iter::once(-e.log_prior).chain(e.lik.lik[start..].iter().copied()).collect()
                });
                Evaluation { fval: e.fval, info: 0, info4: 0.0, contributions }
            }
            Err(f) => Evaluation { fval: f64::INFINITY, info: f.info, info4: f.penalty, contributions: None },
        }
    }
}

/// The parts of Dynare's global `M_` that `set_all_parameters` writes.
#[derive(Debug, Clone)]
pub struct GlobalState {
    pub params: Vec<f64>,
    pub sigma_e: Mat,
    pub h: Mat,
    pub corr_e: Mat,
    pub corr_h: Mat,
}

impl GlobalState {
    /// Makes this the mod file's `M_.params`, `M_.Sigma_e`, `M_.H` and correlation matrices.
    pub fn write(self, mf: &mut ModFile) {
        if mf.model.params.len() == self.params.len() {
            mf.model.params = self.params;
        }
        mf.sigma_e = self.sigma_e;
        mf.h = self.h;
        mf.corr_e = self.corr_e;
        mf.corr_h = self.corr_h;
    }
}
