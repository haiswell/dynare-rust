// SPDX-License-Identifier: GPL-3.0-or-later
//! `dsge_likelihood.m` with `kalman_algo = 5` (pruned skewed Kalman filter,
//! [`crate::kalman::skewed`]): the unrestricted state space (`dynare_resolve` without
//! `'restrict'`), `M_.csn = csn_update_specification(Sigma_e, Skew_e)` with the calibrated
//! `skew` entries of the `shocks` blocks and the estimated `skew` parameters
//! (`set_all_parameters.m`).
//!
//! [estimation completion] As Dynare 7.2 does:
//! * every `lik_init` but 3 (Lyapunov / `Harvey_scale_factor` / Riccati / partially diffuse
//!   `Pstar` as the covariance of the period-0 state, the filter predicting from `a`);
//!   `lik_init = 3` is the error of dsge_likelihood.m;
//! * missing observations: dsge_likelihood.m runs the *Gaussian*
//!   `missing_observations_kalman_filter` (from `T*a`, `Pstar`, no univariate fallback since
//!   dynare_estimation_init.m resets `use_univariate_filters_if_singularity_is_detected`);
//! * `filter_initial_state`: set_Kalman_starting_values.m writes the values at the positions
//!   `bayestopt_.mf0` of the *restricted* state vector into the unrestricted one (reproduced);
//! * `skewed_kalman_rank_deficiency_transform`, `skewed_kalman_mvnlogcdf`;
//! * the smoother: `DsgeSmoother.m` -> `kalman_smoother_pruned_skewed.m` ([`smoother`]).

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::collapsible_if
)] // index loops mirror the MATLAB code

use anyhow::{Result, anyhow, bail};

use super::{Bounds, Evaluation, Objective};
use crate::dsge::{DsgeModel, EvalFailure, LikelihoodData, Resolved};
use crate::kalman::skewed::{Csn, Mvnlogcdf, SkewedOptions, kalman_filter_pruned_skewed};
use crate::kalman::{KalmanOptions, SmootherOutput};
use crate::linalg::{Mat, Vector};
use crate::modfile::{EstimatedKind, ModFile, OptValue, Options, opt};
use crate::solver::{StateSpaceIndex, kalman_transition_matrix};

/// [estimation completion] `kalman_algo = 5`: the calibrated `Skew_e(k,k,k)` and
/// `options_.skewed_kalman` (carried by [`LikelihoodData::skewed`]).
#[derive(Clone, Debug)]
pub struct SkewedSpec {
    pub base_skew: Vec<f64>,
    pub opts: SkewedOptions,
}

impl SkewedSpec {
    /// dynare_estimation_init.m's checks for `kalman_algo = 5` and the skewed filter options.
    pub fn new(mf: &ModFile, o: &Options, k: &KalmanOptions) -> Result<SkewedSpec> {
        let flag = |name: &str| match opt(o, name) {
            Some(OptValue::Flag) => true,
            Some(OptValue::Num(v)) => *v != 0.0,
            _ => false,
        };
        let num = |name: &str| match opt(o, name) {
            Some(OptValue::Num(v)) => Some(*v),
            _ => None,
        };
        for name in ["smoother_redux", "smoothed_state_uncertainty"] {
            if flag(name) {
                bail!(
                    "dynare_estimation_init: pruned skewed Kalman filter is not yet compatible with the {name} option."
                );
            }
        }
        if num("forecast").unwrap_or(0.0) > 0.0 {
            bail!(
                "dynare_estimation_init: pruned skewed Kalman filter is not yet compatible with the forecast option."
            );
        }
        for name in ["filter_covariance", "filter_decomposition", "filtered_vars"] {
            if flag(name) {
                bail!(
                    "dynare_estimation_init: pruned skewed Kalman filter is not yet compatible with the {name} option."
                );
            }
        }
        if opt(o, "filter_step_ahead").is_some() {
            bail!(
                "dynare_estimation_init: pruned skewed Kalman filter is not yet compatible with the filter_step_ahead option."
            );
        }
        if flag("heteroskedastic_filter") {
            bail!(
                "dynare_estimation_init: pruned skewed Kalman filter is not yet compatible with the heteroskedastic_filter option."
            );
        }
        if k.lik_init == 3 {
            bail!(
                "The model requires Diffuse filter, but you specified a different Kalman filter. You must set options_.kalman_algo to 0 (default), 3 or 4"
            );
        }
        let mvnlogcdf = match opt(o, "skewed_kalman_mvnlogcdf") {
            Some(OptValue::Str(s)) => Mvnlogcdf::parse(s)?,
            Some(_) => bail!("skewed_kalman_mvnlogcdf: expected a quoted string"),
            None => Mvnlogcdf::MendellElston,
        };
        Ok(SkewedSpec {
            base_skew: base_skew(mf)?,
            opts: SkewedOptions {
                kalman_tol: k.kalman_tol,
                rescale_prediction_error_covariance: k.rescale_prediction_error_covariance,
                prune_tol: num("skewed_kalman_prune_tol").unwrap_or(0.01),
                presample: k.presample,
                mvnlogcdf,
                rank_deficiency_transform: flag("skewed_kalman_rank_deficiency_transform"),
            },
        })
    }

    /// `Skew_e(k,k,k)` at `x`: the calibration overridden by the estimated `skew` parameters.
    pub fn skew_vector(&self, dsge: &DsgeModel, x: &[f64]) -> Vec<f64> {
        let mut s = self.base_skew.clone();
        for (e, &v) in dsge.estimated.iter().zip(x) {
            if let EstimatedKind::Skew(k) = e.kind {
                s[k] = v;
            }
        }
        s
    }
}

pub struct SkewedPosterior<'a> {
    pub dsge: &'a DsgeModel,
    pub data: &'a LikelihoodData,
    pub bounds: &'a Bounds,
    /// Calibrated `Skew_e(k,k,k)` per shock.
    pub base_skew: Vec<f64>,
    pub opts: SkewedOptions,
}

/// The calibrated skewness coefficients (`M_.Skew_e`), checking that there is no co-skewness.
pub fn base_skew(mf: &ModFile) -> Result<Vec<f64>> {
    let mut s = vec![0.0; mf.model.n_exo()];
    for &(i, j, k, v) in &mf.q3.skew_e {
        if i != j || j != k {
            bail!(
                "csn_update_specification: Skewness is currently only supported for independent skew normally distributed shocks, i.e. there cannot be co-skewness between shocks."
            );
        }
        s[i] = v;
    }
    Ok(s)
}

/// The demeaned data of the unrestricted state space: `Y - constant - trend`.
fn demeaned(dsge: &DsgeModel, data: &LikelihoodData, res: &Resolved) -> Mat {
    let mut y = data.y.clone();
    for (i, &v) in dsge.varobs.iter().enumerate() {
        let c = dsge.level_constant(v, &res.steady_state);
        for j in 0..y.ncols() {
            y[(i, j)] -= c;
        }
    }
    if let Some(tr) = dsge.trend_addition(&res.params, data.first_obs, y.ncols(), data.prefilter_mean.is_some()) {
        y -= tr;
    }
    y
}

/// `filter_initial_state` value of endogenous variable `endo` minus its steady state (logs
/// with `loglinear`), if set.
fn initial_value(dsge: &DsgeModel, res: &Resolved, entries: &[(usize, f64)], endo: usize) -> Option<f64> {
    entries.iter().find(|(e, _)| *e == endo).map(|&(_, v)| {
        let ys = res.steady_state[endo];
        if dsge.loglinear { v.ln() - ys.ln() } else { v - ys }
    })
}

impl SkewedPosterior<'_> {
    /// `dynare_estimation_init.m`'s checks for `kalman_algo = 5` (see [`SkewedSpec::new`]).
    pub fn check(dsge: &DsgeModel, data: &LikelihoodData) -> Result<()> {
        // ---- estimation completion: missing observations and lik_init 2/4/5 are run as Dynare does ----
        if data.kalman.lik_init == 3 {
            bail!(
                "The model requires Diffuse filter, but you specified a different Kalman filter. You must set options_.kalman_algo to 0 (default), 3 or 4"
            );
        }
        if data.hetero.is_some() {
            bail!(
                "dynare_estimation_init: pruned skewed Kalman filter is not yet compatible with the heteroskedastic_filter option."
            );
        }
        let _ = dsge;
        Ok(())
    }

    fn skew_vector(&self, x: &[f64]) -> Vec<f64> {
        let mut s = self.base_skew.clone();
        for (e, &v) in self.dsge.estimated.iter().zip(x) {
            if let EstimatedKind::Skew(k) = e.kind {
                s[k] = v;
            }
        }
        s
    }

    pub fn evaluate(&self, x: &[f64]) -> Result<(f64, f64, f64), EvalFailure> {
        let b = self.bounds;
        if x.iter().zip(&b.lb).any(|(v, l)| v < l) {
            let p = x.iter().zip(&b.lb).filter(|(v, l)| v < l).map(|(v, l)| (l - v).powi(2)).sum();
            return Err(EvalFailure { info: 41, penalty: p });
        }
        if x.iter().zip(&b.ub).any(|(v, u)| v > u) {
            let p = x.iter().zip(&b.ub).filter(|(v, u)| v > u).map(|(v, u)| (v - u).powi(2)).sum();
            return Err(EvalFailure { info: 42, penalty: p });
        }
        let res = self.dsge.resolve(x)?;
        // ---- calibration restrictions ----
        if let Some(rs) = &self.data.calib_restrictions {
            crate::gsa::calibration::likelihood_check(self.dsge, &res, rs)?;
        }
        // ---- end calibration restrictions ----
        let csn = Csn::from_skew(&res.q, &self.skew_vector(x)).map_err(|_| EvalFailure { info: 43, penalty: 0.1 })?;
        let full = StateSpaceIndex::full(&res.dr);
        let (t, r) = kalman_transition_matrix(&res.dr, &full);
        let z: Vec<usize> = self.dsge.varobs.iter().map(|&v| res.dr.inv_order_var[v]).collect();
        let y = demeaned(self.dsge, self.data, &res);
        let k = &self.data.kalman;
        let mm = t.nrows();
        let h = if res.h.nrows() == y.nrows() { res.h.clone() } else { Mat::zeros(y.nrows(), y.nrows()) };
        let fail50 = |_| EvalFailure { info: 50, penalty: 0.1 };
        // dsge_likelihood.m, section 3 (kalman_algo = 5 keeps its filter for lik_init 1, 2, 4, 5)
        let pstar = match k.lik_init {
            1 => crate::kalman::lyapunov::lyapunov_solver_opts(&t, &r, &res.q, k).map_err(fail50)?,
            li => crate::kalman::lik_init::initial_pstar(
                li,
                &t,
                &r,
                &res.q,
                Some(&h),
                &z,
                k.qz_criterium,
                k.lyapunov_complex_threshold,
                false,
            )
            .map_err(fail50)?,
        };
        // a = set_Kalman_starting_values(zeros(mm,1), ...): bayestopt_.mf0 positions of the
        // restricted state space, written into the unrestricted vector
        let mut a0 = Vector::zeros(mm);
        if !self.data.filter_initial_state.is_empty() {
            let fis = self.data.filter_initial_state_at(&res.params);
            let idx = &self.dsge.ss_index;
            for &pos in &idx.restrict_columns {
                let endo = res.dr.order_var[idx.restrict_var_list[pos]];
                if let Some(v) = initial_value(self.dsge, &res, &fis, endo) {
                    a0[pos] = v;
                }
            }
        }
        let lik = if self.data.missing.any_missing {
            // missing_observations_kalman_filter (Gaussian), from a_0_given_tm1 = T*a and Pstar
            let gauss = KalmanOptions {
                lik_init: 1,
                kalman_algo: 1,
                use_univariate_filters_if_singularity_is_detected: false,
                ..k.clone()
            };
            let mut ws = crate::kalman::KalmanWorkspace::new(mm, y.nrows(), res.q.nrows());
            ws.set_initial_state(Some((&t * &a0).iter().copied().collect()));
            let out = ws.likelihood(
                &y,
                &self.data.missing,
                &z,
                &t,
                &r,
                &res.q,
                Some(&h),
                &pstar,
                &Mat::zeros(mm, mm),
                &gauss,
            );
            if out.info != 0 {
                return Err(EvalFailure { info: out.info, penalty: 0.1 });
            }
            out.minus_loglik
        } else {
            kalman_filter_pruned_skewed(&y, &a0, &pstar, &t, &r, &z, &csn, &h, &self.opts).0
        };
        if lik.is_nan() {
            return Err(EvalFailure { info: 45, penalty: 0.1 });
        }
        if lik.is_infinite() {
            return Err(EvalFailure { info: 50, penalty: 0.1 });
        }
        let lp = self.dsge.log_prior(x);
        if lp.is_nan() {
            return Err(EvalFailure { info: 47, penalty: 0.1 });
        }
        if lp.is_infinite() {
            return Err(EvalFailure { info: 40, penalty: 0.1 });
        }
        Ok((lik - lp, lik, lp))
    }
}

/// [estimation completion] `DsgeSmoother.m` with `kalman_algo = 5` at `x`: the initial
/// covariance of DsgeSmoother (`lik_init` 1: Lyapunov, 2: `Harvey_scale_factor*I`, 4: Riccati,
/// 5: partially diffuse), `a_initial` from set_Kalman_smoother_starting_values (not
/// premultiplied by T) and [`crate::kalman::skewed_smoother::kalman_smoother_pruned_skewed`].
/// `res`, `y`, `t`, `r`, `mf`: the caller's unrestricted state space and demeaned data.
pub fn smoother(
    dsge: &DsgeModel,
    data: &LikelihoodData,
    spec: &SkewedSpec,
    x: &[f64],
    res: &Resolved,
    y: &Mat,
    t: &Mat,
    r: &Mat,
    mf: &[usize],
) -> Result<SmootherOutput> {
    let k = &data.kalman;
    let csn = Csn::from_skew(&res.q, &spec.skew_vector(dsge, x))?;
    let h = if res.h.nrows() == y.nrows() { res.h.clone() } else { Mat::zeros(y.nrows(), y.nrows()) };
    let pstar = match k.lik_init {
        1 => crate::kalman::lyapunov::lyapunov_solver_opts(t, r, &res.q, k)?,
        3 => bail!(
            "The model requires Diffuse filter, but you specified a different Kalman filter. You must set options_.kalman_algo to 0 (default), 3 or 4"
        ),
        li => crate::kalman::lik_init::initial_pstar(
            li,
            t,
            r,
            &res.q,
            Some(&h),
            mf,
            k.qz_criterium,
            k.lyapunov_complex_threshold,
            true,
        )?,
    };
    let mm = t.nrows();
    let mut a0 = Vector::zeros(mm);
    let fis = data.filter_initial_state_at(&res.params);
    for pos in res.dr.nstatic..res.dr.nstatic + res.dr.nspred {
        // set_Kalman_smoother_starting_values: dr.restrict_columns of the full state space
        if let Some(v) = initial_value(dsge, res, &fis, res.dr.order_var[pos]) {
            a0[pos] = v;
        }
    }
    let s = crate::kalman::skewed_smoother::kalman_smoother_pruned_skewed(
        y,
        &a0,
        &pstar,
        t,
        r,
        mf,
        &csn,
        &h,
        k.kalman_tol,
        spec.opts.prune_tol,
        spec.opts.mvnlogcdf,
    )
    .map_err(|e| anyhow!("DsgeSmoother (kalman_algo = 5): {e}"))?;
    let n = y.ncols();
    Ok(SmootherOutput {
        alphahat: s.alphahat,
        etahat: s.etahat,
        epsilonhat: s.epsilonhat,
        ahat: Mat::from_element(mm, n, f64::NAN),
        filtered: Mat::from_element(mm, n, f64::NAN),
        alphahat0: s.alphahat0,
        state_uncertainty: None,
        state_uncertainty0: None,
        d: 0,
        univariate: false,
        info: 0,
        step_ahead: None,
    })
}

impl Objective for SkewedPosterior<'_> {
    fn dim(&self) -> usize {
        self.dsge.n_estimated()
    }

    fn eval(&self, x: &[f64]) -> (f64, i32) {
        let r = match self.evaluate(x) {
            Ok(e) => (e.0, 0),
            Err(f) => (f64::INFINITY, f.info),
        };
        crate::progress::eval(r.0);
        r
    }

    fn eval_full(&self, x: &[f64], _contributions: bool) -> Evaluation {
        let r = self.evaluate(x);
        crate::progress::eval(r.as_ref().map_or(f64::INFINITY, |e| e.0));
        match r {
            Ok(e) => Evaluation { fval: e.0, info: 0, info4: 0.0, contributions: None },
            Err(f) => Evaluation { fval: f64::INFINITY, info: f.info, info4: f.penalty, contributions: None },
        }
    }
}
