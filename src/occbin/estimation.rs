// SPDX-License-Identifier: GPL-3.0-or-later
//! OccBin estimation: `options_.occbin.likelihood/smoother/filter` (set by `occbin_setup`,
//! `+occbin/setup.m`), and the posterior kernel with OccBin's filters as an [`Objective`]:
//! `dsge_likelihood.m` with `options_.occbin.likelihood.status` (piecewise Kalman filter,
//! [`super::pkf`]) and `+occbin/IVF_posterior.m` (inversion filter, [`super::ivf`]).
//!
//! As in Dynare, a model with an `occbin_constraints` block is estimated with these filters
//! (`occbin.set_default_options` sets `likelihood.status` and `smoother.status` when the
//! preprocessor sees the constraints).

use anyhow::{Result, bail};
use nalgebra::DVector;

use super::constraints::Constraints;
use super::ivf::{self, MatchSolverOptions};
use super::pkf::{self, PkfOptions, PointModel};
use super::regimes::Regimes;
use super::solver::SimulOptions;
use super::{Session, boolean, count, num};
use crate::dsge::{DsgeModel, EvalFailure, LikelihoodData, Resolved, endogenous_prior};
use crate::estimation::{Bounds, Evaluation, Objective};
use crate::kalman::lik_init::{HARVEY_SCALE_FACTOR, initial_pstar};
use crate::kalman::lyapunov::lyapunov_solver_opts;
use crate::linalg::Mat;
use crate::modfile::{ModFile, OptValue};
use crate::solver::StateSpaceIndex;

/// `options_.occbin.likelihood` (defaults of `occbin.set_default_options`).
#[derive(Clone, Debug)]
pub struct LikelihoodOptions {
    pub periods: usize,
    pub curb_retrench: bool,
    pub maxit: usize,
    pub check_ahead_periods: usize,
    /// None = Inf.
    pub max_check_ahead_periods: Option<usize>,
    pub periodic_solution: bool,
    /// `max_number_of_iterations` (`likelihood_max_kalman_iterations`).
    pub max_number_of_iterations: usize,
    pub inversion_filter: bool,
    pub brute_force_regime_guess: bool,
    pub brute_force_extra_regime_guess: bool,
    pub first_period_binding_regime_allowed: usize,
    pub number_of_initial_periods_with_extra_regime_guess: usize,
}

impl Default for LikelihoodOptions {
    fn default() -> Self {
        LikelihoodOptions {
            periods: 3,
            curb_retrench: false,
            maxit: 30,
            check_ahead_periods: 200,
            max_check_ahead_periods: None,
            periodic_solution: false,
            max_number_of_iterations: 10,
            inversion_filter: false,
            brute_force_regime_guess: true,
            brute_force_extra_regime_guess: true,
            first_period_binding_regime_allowed: 1,
            number_of_initial_periods_with_extra_regime_guess: 0,
        }
    }
}

/// `options_.occbin.smoother`.
#[derive(Clone, Debug)]
pub struct SmootherOptions {
    pub periods: usize,
    pub curb_retrench: bool,
    pub maxit: usize,
    pub check_ahead_periods: usize,
    pub max_check_ahead_periods: Option<usize>,
    pub periodic_solution: bool,
    pub max_number_of_iterations: usize,
    pub inversion_filter: bool,
    pub debug: bool,
    pub linear_smoother: bool,
    pub fast: bool,
    pub first_period_occbin_update: usize,
}

impl Default for SmootherOptions {
    fn default() -> Self {
        SmootherOptions {
            periods: 3,
            curb_retrench: false,
            maxit: 30,
            check_ahead_periods: 200,
            max_check_ahead_periods: None,
            periodic_solution: false,
            max_number_of_iterations: 10,
            inversion_filter: false,
            debug: false,
            linear_smoother: true,
            fast: false,
            first_period_occbin_update: 1,
        }
    }
}

/// `options_.occbin.filter`.
#[derive(Clone, Debug)]
pub struct FilterOptions {
    pub use_relaxation: bool,
    pub use_relaxation_tol_period: i64,
    pub periodic_solution: bool,
}

impl Default for FilterOptions {
    fn default() -> Self {
        FilterOptions { use_relaxation: false, use_relaxation_tol_period: 1, periodic_solution: true }
    }
}

/// One non-`simul_` option of `occbin_setup`, as `+occbin/setup.m` forwards it; false if
/// setup.m does not forward it (the preprocessor accepts more options than setup.m uses).
pub fn setup_option(s: &mut Session, k: &str, v: &OptValue) -> Result<bool> {
    let l = &mut s.likelihood;
    let m = &mut s.smoother;
    match k {
        "likelihood_periods" => l.periods = count(v, k)?,
        "likelihood_curb_retrench" => l.curb_retrench = boolean(v),
        "likelihood_maxit" => l.maxit = count(v, k)?,
        "likelihood_check_ahead_periods" => l.check_ahead_periods = count(v, k)?,
        "likelihood_periodic_solution" => l.periodic_solution = boolean(v),
        "likelihood_max_kalman_iterations" => l.max_number_of_iterations = count(v, k)?,
        "likelihood_inversion_filter" => l.inversion_filter = boolean(v),
        "likelihood_piecewise_kalman_filter" => l.inversion_filter = !boolean(v),
        "smoother_periods" => m.periods = count(v, k)?,
        "smoother_curb_retrench" => m.curb_retrench = boolean(v),
        "smoother_maxit" => m.maxit = count(v, k)?,
        "smoother_check_ahead_periods" => m.check_ahead_periods = count(v, k)?,
        "smoother_debug" => m.debug = boolean(v),
        "smoother_periodic_solution" => m.periodic_solution = boolean(v),
        "smoother_inversion_filter" => m.inversion_filter = boolean(v),
        "smoother_piecewise_kalman_filter" => m.inversion_filter = !boolean(v),
        "filter_use_relaxation" => s.filter.use_relaxation = boolean(v),
        _ => {
            let _ = num(v);
            return Ok(false);
        }
    }
    Ok(true)
}

/// What the OccBin likelihood needs besides the linear model (built once per estimation).
pub struct OccbinEstimation {
    pub constraints: Constraints,
    pub pswitch: Vec<usize>,
    pub pkf: PkfOptions,
    pub inversion_filter: bool,
    /// `options_.occbin.likelihood.IVF_shock_observable_mapping` (0-based exogenous indices,
    /// one per observable).
    pub err_index: Vec<usize>,
    pub solver: MatchSolverOptions,
    pub smoother: SmootherOptions,
    /// The solver options of `occbin.DSGE_smoother` (`options_.occbin.simul` with the
    /// smoother's settings).
    pub smoother_simul: SimulOptions,
}

/// `set_occbin_options` (dsge_likelihood.m) / IVF_core.m: `options_.occbin.simul` with the
/// likelihood's settings.
fn likelihood_simul(s: &Session) -> SimulOptions {
    let l = &s.likelihood;
    let mut o = s.simul.clone();
    o.curb_retrench = l.curb_retrench;
    o.maxit = l.maxit;
    o.periods = l.periods;
    o.check_ahead_periods = l.check_ahead_periods;
    o.max_check_ahead_periods = l.max_check_ahead_periods;
    o.periodic_solution = l.periodic_solution;
    o.piecewise_only = true;
    // dynare_estimation_init.m deletes the surprise shocks; the filters set SHOCKS/endo_init
    o.endo_init = Vec::new();
    if let Some(mx) = o.max_check_ahead_periods
        && o.check_ahead_periods > mx
    {
        o.check_ahead_periods = mx;
    }
    o
}

fn smoother_simul(s: &Session) -> SimulOptions {
    let m = &s.smoother;
    let mut o = s.simul.clone();
    o.curb_retrench = m.curb_retrench;
    o.maxit = m.maxit;
    o.periods = m.periods;
    o.check_ahead_periods = m.check_ahead_periods;
    o.max_check_ahead_periods = m.max_check_ahead_periods;
    o.periodic_solution = m.periodic_solution;
    o.piecewise_only = true;
    o.endo_init = Vec::new();
    o
}

impl OccbinEstimation {
    /// `None` when the model has no `occbin_constraints` block (linear estimation).
    pub fn new(mf: &ModFile, s: &Session, dsge: &DsgeModel, data: &LikelihoodData) -> Result<Option<OccbinEstimation>> {
        if mf.occbin_constraints.is_empty() {
            return Ok(None);
        }
        let m = &mf.model;
        let constraints = Constraints::compile(&mf.occbin_constraints, m)?;
        let pswitch = super::pswitch(m, &mf.occbin_constraints)?;
        let l = &s.likelihood;
        // initial_estimation_checks.m
        if data.prefilter_mean.is_some() {
            bail!(
                "initial_estimation_checks:: OccBin is incompatible with the prefilter option due to the sample mean generally not corresponding to the steady state with an occasionally binding constraint."
            );
        }
        if !l.inversion_filter && (data.kalman.kalman_algo == 2 || data.kalman.kalman_algo == 4) {
            bail!("initial_estimation_checks:: OccBin is incompatible with the selected univariate Kalman filter.");
        }
        if dsge.loglinear {
            // +occbin has no loglinear handling: occbin.get_deriv evaluates the level dynamic
            // function of the regimes while stochastic_solvers.m:349 rescales the base Jacobian
            // by the steady state, so Dynare 7.2 mixes levels and logs for such a model
            bail!(
                "OccBin estimation with loglinear is not supported (Dynare 7.2's +occbin mixes the log-linearised base solution with level Jacobians for the regimes)"
            );
        }
        if !l.inversion_filter && data.kalman.lik_init == 3 {
            bail!("OccBin's piecewise Kalman filter: lik_init = 3 (diffuse_filter) is not ported");
        }
        if !l.inversion_filter && data.endogenous_prior && data.kalman.lik_init == 2 {
            // dsge_likelihood.m (raised at the first evaluation)
            bail!("Endogenous prior not supported with non-stationary models");
        }
        // dynare_estimation_init.m: IVF_shock_observable_mapping = find(diag(Sigma_e) ~= 0)
        let mut err_index = Vec::new();
        if l.inversion_filter || s.smoother.inversion_filter {
            err_index = (0..m.n_exo()).filter(|&k| dsge.base_sigma_e[(k, k)] != 0.0).collect();
            if dsge.base_h.iter().any(|&v| v != 0.0) {
                bail!("IVF-filter: Measurement errors are not allowed with the inversion filter.");
            }
            if err_index.len() != mf.varobs.len() {
                println!("initial_estimation_checks:: The IVF requires exactly as many shocks as observables.");
                if err_index.len() < mf.varobs.len() {
                    bail!(
                        "IVF: fewer shocks with a non-zero variance ({}) than observables ({})",
                        err_index.len(),
                        mf.varobs.len()
                    );
                }
            }
            err_index.truncate(mf.varobs.len());
        }
        let pkf = PkfOptions {
            simul: likelihood_simul(s),
            max_number_of_iterations: l.max_number_of_iterations,
            periodic_solution: s.filter.periodic_solution,
            use_relaxation: s.filter.use_relaxation,
            use_relaxation_tol_period: s.filter.use_relaxation_tol_period,
            brute_force_regime_guess: l.brute_force_regime_guess,
            brute_force_extra_regime_guess: l.brute_force_extra_regime_guess,
            number_of_initial_periods_with_extra_regime_guess: l.number_of_initial_periods_with_extra_regime_guess,
            first_period_binding_regime_allowed: l.first_period_binding_regime_allowed,
            rescale_prediction_error_covariance: data.kalman.rescale_prediction_error_covariance,
        };
        Ok(Some(OccbinEstimation {
            constraints,
            pswitch,
            pkf,
            inversion_filter: l.inversion_filter,
            err_index,
            solver: MatchSolverOptions::default(),
            smoother: s.smoother.clone(),
            smoother_simul: smoother_simul(s),
        }))
    }

    /// The OccBin model at a resolved parameter point, on the filter's restricted states.
    pub fn point_model<'a>(&'a self, dsge: &DsgeModel, res: &Resolved) -> PointModel<'a> {
        PointModel::new(
            &dsge.model,
            &self.constraints,
            &self.pswitch,
            res.params.clone(),
            res.steady_state.clone(),
            &res.dr,
            &dsge.ss_index.restrict_var_list,
            res.t.clone(),
            res.r.clone(),
        )
    }
}

/// Port of `set_Kalman_starting_values` (`dsge_likelihood.m`) and of
/// `set_Kalman_smoother_starting_values` (`DsgeSmoother.m`, with the full state space) for
/// `M_.filter_initial_state`: `a(restrict_columns) = value - ys` for the states given in
/// `entries` (endogenous index, level), zero elsewhere; the state before any prediction.
pub(super) fn kalman_starting_values(res: &Resolved, idx: &StateSpaceIndex, entries: &[(usize, f64)]) -> DVector<f64> {
    let mut a = DVector::zeros(idx.restrict_var_list.len());
    for &pos in &idx.restrict_columns {
        let endo = res.dr.order_var[idx.restrict_var_list[pos]];
        if let Some(&(_, v)) = entries.iter().find(|(e, _)| *e == endo) {
            a[pos] = v - res.steady_state[endo];
        }
    }
    a
}

/// Everything one OccBin evaluation produces.
pub struct OccbinEvaluated {
    pub fval: f64,
    pub minus_loglik: f64,
    pub log_prior: f64,
    /// Per-period likelihood contributions.
    pub lik: Vec<f64>,
    /// Filter's regimes (PKF: `regimes_` of periods 1..nobs; IVF: `regime_history`).
    pub regimes: Vec<Regimes>,
    pub resolved: Resolved,
    /// IVF only: filtered states (n_endo x nobs, deviations, declaration order) and shocks
    /// (n_exo x nobs).
    pub ivf_states: Option<(Mat, Mat)>,
}

/// The posterior kernel with OccBin's filters (`dsge_likelihood.m` with
/// `options_.occbin.likelihood.status`, or `occbin.IVF_posterior`).
pub struct OccbinPosterior<'a> {
    pub dsge: &'a DsgeModel,
    pub data: &'a LikelihoodData,
    pub bounds: &'a Bounds,
    pub occ: &'a OccbinEstimation,
}

fn bounds_check(x: &[f64], b: &Bounds) -> Result<(), EvalFailure> {
    if x.iter().zip(&b.lb).any(|(v, l)| v < l) {
        let p = x.iter().zip(&b.lb).filter(|(v, l)| v < l).map(|(v, l)| (l - v).powi(2)).sum();
        return Err(EvalFailure { info: 41, penalty: p });
    }
    if x.iter().zip(&b.ub).any(|(v, u)| v > u) {
        let p = x.iter().zip(&b.ub).filter(|(v, u)| v > u).map(|(v, u)| (v - u).powi(2)).sum();
        return Err(EvalFailure { info: 42, penalty: p });
    }
    Ok(())
}

impl OccbinPosterior<'_> {
    pub fn evaluate(&self, x: &[f64]) -> Result<OccbinEvaluated, EvalFailure> {
        bounds_check(x, self.bounds)?;
        let res = self.dsge.resolve(x)?;
        // ---- calibration restrictions ----
        if let Some(rs) = &self.data.calib_restrictions {
            crate::gsa::calibration::likelihood_check(self.dsge, &res, rs)?;
        }
        // ---- end calibration restrictions ----
        if self.occ.inversion_filter {
            return self.evaluate_ivf(x, res);
        }
        let idx = &self.dsge.ss_index;
        let opts = &self.data.kalman;
        // Y = data - constant - trend (observation_trends); the constant is zero with
        // `noconstant` (dsge_likelihood.m:204)
        let mut y = self.data.y.clone();
        if !opts.noconstant {
            for (i, &v) in idx.mfys.iter().enumerate() {
                let c = res.steady_state[v];
                for j in 0..y.ncols() {
                    y[(i, j)] -= c;
                }
            }
        }
        if let Some(tr) = self.dsge.trend_addition(&res.params, self.data.first_obs, y.ncols(), false) {
            y -= tr;
        }
        // initial condition of the Kalman filter: a = set_Kalman_starting_values(...),
        // a_0_given_tm1 = T*a, except with lik_init = 2 (first_period_occbin_update = 1)
        let a = kalman_starting_values(&res, idx, &self.data.filter_initial_state);
        let (a_0_given_tm1, pstar) = match opts.lik_init {
            1 => {
                let p = lyapunov_solver_opts(&res.t, &res.r, &res.q, opts)
                    .map_err(|_| EvalFailure { info: 50, penalty: 0.1 })?;
                (&res.t * a, p)
            }
            2 => {
                let mm = res.t.nrows();
                (a, Mat::identity(mm, mm) * HARVEY_SCALE_FACTOR)
            }
            k => {
                let p = initial_pstar(
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
                (&res.t * a, p)
            }
        };
        let mut pm = self.occ.point_model(self.dsge, &res);
        let out = pkf::pkf(
            &mut pm,
            &y,
            &self.data.missing.aindex,
            &idx.mf,
            &res.q,
            &res.h,
            a_0_given_tm1,
            &pstar,
            opts.presample,
            &self.occ.pkf,
        );
        if out.minus_loglik.is_infinite() {
            return Err(EvalFailure { info: 320, penalty: 0.1 });
        }
        if out.minus_loglik.is_nan() {
            return Err(EvalFailure { info: 45, penalty: 0.1 });
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
        let nobs = self.data.y.ncols();
        Ok(OccbinEvaluated {
            fval: out.minus_loglik - log_prior - lnpriormom,
            minus_loglik: out.minus_loglik,
            log_prior,
            lik: out.lik,
            regimes: out.regimes[..nobs.min(out.regimes.len())].to_vec(),
            resolved: res,
            ivf_states: None,
        })
    }

    /// `occbin.IVF_posterior` (after the bounds check and `dynare_resolve`).
    pub fn evaluate_ivf(&self, x: &[f64], res: Resolved) -> Result<OccbinEvaluated, EvalFailure> {
        let occ = self.occ;
        let mut pm = occ.point_model(self.dsge, &res);
        let obs = self.data.y.transpose();
        let core = ivf::ivf_core(&mut pm, &occ.pkf.simul, &occ.solver, &obs, &self.dsge.varobs, &occ.err_index);
        if core.info != 0 {
            return Err(EvalFailure { info: core.info, penalty: core.info4 });
        }
        let k = occ.err_index.len();
        let cov = Mat::from_fn(k, k, |i, j| res.q[(occ.err_index[i], occ.err_index[j])]);
        let (like, likei) = ivf::ivf_likelihood(&core, &obs, &self.dsge.varobs, &cov, self.data.kalman.presample)
            .map_err(|(info, penalty)| EvalFailure { info, penalty })?;
        let log_prior = self.dsge.log_prior(x);
        if log_prior.is_infinite() {
            return Err(EvalFailure { info: 40, penalty: 1000.0 });
        }
        if log_prior.is_nan() {
            return Err(EvalFailure { info: 47, penalty: 1000.0 });
        }
        let n_exo = res.q.nrows();
        let nobs = obs.nrows();
        // innov(diag(M_.Sigma_e)~=0,:) = filtered_errs'
        let nz: Vec<usize> = (0..n_exo).filter(|&e| res.q[(e, e)] != 0.0).collect();
        let mut innov = Mat::zeros(n_exo, nobs);
        for (c, &e) in nz.iter().enumerate().take(core.filtered_errs.ncols()) {
            for t in 0..nobs {
                innov[(e, t)] = core.filtered_errs[(t, c)];
            }
        }
        Ok(OccbinEvaluated {
            fval: like - log_prior,
            minus_loglik: like,
            log_prior,
            lik: likei,
            regimes: core.regime_history.clone(),
            resolved: res,
            ivf_states: Some((core.stateval.transpose(), innov)),
        })
    }
}

impl Objective for OccbinPosterior<'_> {
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

    fn eval_full(&self, x: &[f64], contributions: bool) -> Evaluation {
        let r = self.evaluate(x);
        crate::progress::eval(r.as_ref().map_or(f64::INFINITY, |e| e.fval));
        match r {
            Ok(e) => {
                let contributions = contributions.then(|| {
                    let start = self.data.kalman.presample;
                    std::iter::once(-e.log_prior).chain(e.lik[start.min(e.lik.len())..].iter().copied()).collect()
                });
                Evaluation { fval: e.fval, info: 0, info4: 0.0, contributions }
            }
            Err(f) => Evaluation { fval: f64::INFINITY, info: f.info, info4: f.penalty, contributions: None },
        }
    }
}

/// The estimation's smoother with OccBin (dynare_estimation_1.m): `occbin.IVF_posterior` with
/// `smoother_inversion_filter`, else `occbin.DSGE_smoother`. Returns the series as
/// `driver::run_smoother` does (levels, declaration order) and the smoothed regimes; writes
/// `<fname>_occbin_smoother.json` (`oo_.occbin.smoother`: regimes, flags) to `out_dir`.
pub fn smoother_series(
    dsge: &DsgeModel,
    data: &LikelihoodData,
    occ: &OccbinEstimation,
    x: &[f64],
    out_dir: &std::path::Path,
    fname: &str,
) -> Result<(crate::driver::SmoothedSeries, Vec<Regimes>)> {
    let norig = dsge.model.orig_endo_nbr;
    let nobs = data.y.ncols();
    let regime_json = |rh: &[Regimes]| -> serde_json::Value {
        rh.iter()
            .map(|g| {
                serde_json::Value::Array(
                    g.iter()
                        .map(|r| serde_json::json!({"regime": r.regime.iter().map(|&b| b as u8).collect::<Vec<_>>(), "regimestart": r.start}))
                        .collect(),
                )
            })
            .collect()
    };
    std::fs::create_dir_all(out_dir)?;
    if occ.smoother.inversion_filter {
        let post = OccbinPosterior { dsge, data, bounds: &dsge.bounds, occ };
        let res = dsge.resolve(x).map_err(|e| anyhow::anyhow!("IVF smoother: no solution (info {})", e.info))?;
        let e = post
            .evaluate_ivf(x, res)
            .map_err(|f| anyhow::anyhow!("IVF: smoother did not succeed (info {}).", f.info))?;
        let (states, innov) = e.ivf_states.unwrap();
        let ys = &e.resolved.steady_state;
        let smoothed = Mat::from_fn(nobs, norig, |t, v| states[(v, t)] + ys[v]);
        let j = serde_json::json!({"inversion_filter": true, "regime_history": regime_json(&e.regimes)});
        std::fs::write(out_dir.join(format!("{fname}_occbin_smoother.json")), serde_json::to_string(&j)?)?;
        let series = crate::driver::SmoothedSeries {
            smoothed,
            shocks: innov.transpose(),
            filtered: Mat::from_element(nobs, norig, f64::NAN),
            measurement_errors: None,
        };
        return Ok((series, e.regimes));
    }
    let sm = super::smoother::occbin_smoother(dsge, data, occ, x)?;
    if sm.error_flag != 0 {
        bail!("OccBin: smoother did not succeed (error {}).", sm.error_flag);
    }
    let n = sm.order_var.len();
    let mut inv = vec![0; n];
    for (k, &v) in sm.order_var.iter().enumerate() {
        inv[v] = k;
    }
    let ys = &sm.steady_state;
    // store_smoother_results.m: + the observation trend for the observed variables (shifted by
    // one period for the one-step-ahead filtered variables)
    let trend_at = |ahead: bool, v: usize, t: usize| match (&sm.trend, dsge.varobs.iter().position(|&o| o == v)) {
        (Some((tr, tr1)), Some(i)) => {
            if ahead {
                tr1[(i, t)]
            } else {
                tr[(i, t)]
            }
        }
        _ => 0.0,
    };
    let level = |m: &Mat, ahead: bool| Mat::from_fn(nobs, norig, |t, v| m[(inv[v], t)] + ys[v] + trend_at(ahead, v, t));
    let me = dsge.base_h.iter().any(|&v| v != 0.0).then(|| {
        Mat::from_fn(nobs, dsge.varobs.len(), |t, i| {
            let v = dsge.varobs[i];
            data.y[(i, t)] - (sm.alphahat[(inv[v], t)] + ys[v] + trend_at(false, v, t))
        })
    });
    let j = serde_json::json!({
        "error_flag": sm.error_flag,
        "warning_flag": sm.warning_flag,
        "iterations": sm.iterations,
        "regime_history": regime_json(&sm.regime_history),
        "realtime_regime_history": regime_json(&sm.realtime_regime_history),
    });
    std::fs::write(out_dir.join(format!("{fname}_occbin_smoother.json")), serde_json::to_string(&j)?)?;
    match sm.warning_flag {
        320 => println!("occbin.DSGE_smoother: WARNING: algorithm converged to a different regime wrt realtime regime"),
        321 => println!(
            "occbin.DSGE_smoother: smoother did not converge: the smoothed regimes loop; results are consistent with the last regimes."
        ),
        322 => println!("occbin.DSGE_smoother: We provide results consistent with realtime regime"),
        _ => println!("occbin.DSGE_smoother: smoother converged."),
    }
    let series = crate::driver::SmoothedSeries {
        smoothed: level(&sm.alphahat, false),
        shocks: sm.etahat.transpose(),
        filtered: level(&sm.filtered, true),
        measurement_errors: me,
    };
    Ok((series, sm.regime_history))
}

/// `calib_smoother` on a model with `occbin_constraints` (evaluate_smoother.m:
/// `occbin.DSGE_smoother` or the IVF at the current parameters): writes the
/// `<fname>_calib_smoothed_variables.csv` / `_calib_smoothed_shocks.csv` files of the linear
/// `calib_smoother`, and `<fname>_occbin_smoother.json`; the regimes are kept for
/// `occbin_write_regimes(smoother)`.
#[allow(clippy::too_many_arguments)]
pub fn calib_smoother_command(
    mf: &ModFile,
    st: Option<&crate::forecast::commands::EstimationState>,
    session: &mut Session,
    o: &crate::modfile::Options,
    vars: &[String],
    dir: &std::path::Path,
    fname: &str,
    out_dir: &std::path::Path,
) -> Result<()> {
    let mut own = None;
    let (dsge, data, dates) = crate::forecast::commands::smoother_inputs(mf, st, o, dir, &mut own)?;
    let x = crate::forecast::commands::parameter_vector(st, dsge, "calibration")?;
    let occ = OccbinEstimation::new(mf, session, dsge, data)?.expect("calib_smoother: occbin_constraints");
    let (sm, rh) = smoother_series(dsge, data, &occ, &x, out_dir, fname)?;
    session.smoother_regimes = Some(rh);
    let n = mf.model.orig_endo_nbr;
    let gend = data.y.ncols();
    let dates: Vec<String> = dates.unwrap_or_else(|| (1..=gend).map(|t| t.to_string()).collect());
    let write = |file: &str, cols: &[String], m: &Mat| -> Result<()> {
        let mut w = format!("date,{}\n", cols.join(","));
        for t in 0..m.nrows() {
            w += &dates[t];
            for j in 0..m.ncols() {
                w += &format!(",{}", m[(t, j)]);
            }
            w.push('\n');
        }
        std::fs::write(out_dir.join(format!("{fname}_{file}.csv")), w)?;
        Ok(())
    };
    write("calib_smoothed_variables", &mf.model.endo_names[..n], &sm.smoothed)?;
    write("calib_smoothed_shocks", &mf.model.exo_names, &sm.shocks)?;
    println!("calib_smoother (OccBin): smoothed variables/shocks at the calibration written to {}", out_dir.display());
    if !vars.is_empty() {
        println!("Last smoothed values (calibration):");
        for v in vars {
            if let Some(i) = mf.model.endo_index(v).filter(|&i| i < n) {
                println!("  {:<24} {:>12.4}", v, sm.smoothed[(gend - 1, i)]);
            }
        }
    }
    Ok(())
}

/// The OccBin part of `initial_estimation_checks.m`.
pub fn check_estimation_options(
    _occ: &OccbinEstimation,
    o: &crate::modfile::Options,
    ss: &crate::analysis::StochSimulOptions,
) -> Result<()> {
    let flag = |k: &str| match crate::modfile::opt(o, k) {
        Some(OptValue::Flag) => true,
        Some(v) => num(v).is_some_and(|x| x != 0.0),
        None => false,
    };
    if flag("fast_kalman_filter") {
        bail!("initial_estimation_checks:: OccBin is incompatible with the fast Kalman filter.");
    }
    if ss.bayesian_irf || flag("bayesian_irf") {
        bail!("initial_estimation_checks:: OccBin is incompatible with the bayesian_irf option.");
    }
    if ss.moments_varendo || flag("moments_varendo") {
        bail!("initial_estimation_checks:: OccBin is incompatible with the moments_varendo option.");
    }
    Ok(())
}
