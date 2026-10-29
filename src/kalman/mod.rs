// SPDX-License-Identifier: GPL-3.0-or-later
//! Kalman filtering for the likelihood and the smoother. See DESIGN.md, section `kalman`.
//!
//! * [`compute_pinf_pstar`] / [`lyapunov_solver`]: initial covariances (lik_init 3 / 1).
//! * [`likelihood`] / [`KalmanWorkspace::likelihood`]: the Kalman part of
//!   `estimation/dsge_likelihood.m` with all its filter branches and fallbacks.
//! * [`smoother`]: the filtering/smoothing part of `kalman/DsgeSmoother.m`.
//!
//! Indices are 0-based; `mf` are the observed rows of T (Dynare's `bayestopt_.mf1 - 1`).

mod augment;
mod filters;
mod init;
pub mod lik_init; // [stream X] lik_init 2/4/5
mod smoother;

pub use filters::{DiffuseResult, KalmanWorkspace};
pub(crate) use init::stein_quasi_triangular; // used by analysis::moments (lyapunov_symm with unit-root vectors)
pub use init::{
    compute_pinf_pstar, compute_pinf_pstar_with_schur, lyapunov_solver, lyapunov_symm, try_compute_pinf_pstar,
};
pub use smoother::{
    QUIET as SMOOTHER_QUIET, SmootherOutput, SmootherRequest, StepAhead, smoother, smoother_heteroskedastic,
    smoother_request,
};
pub mod hetero; // heteroskedastic_filter
pub mod lyapunov; // lyapunov = fixed_point, doubling, square_root_solver
pub mod skewed; // pruned skewed Kalman filter (kalman_algo = 5)
// ---- estimation completion ----
pub mod csn; // CSN mean / linear transform (skewed filter and smoother)
pub mod mvncdf; // Dynare's mvncdf (skewed_kalman_mvnlogcdf = 'mvncdf')
pub mod skewed_smoother; // kalman_smoother_pruned_skewed.m
// ---- end estimation completion ----

use crate::linalg::Mat;

/// Options of the Kalman filters (Dynare `options_` fields, same defaults).
#[derive(Clone, Debug)]
pub struct KalmanOptions {
    pub kalman_tol: f64,
    pub diffuse_kalman_tol: f64,
    pub riccati_tol: f64,
    pub presample: usize,
    pub qz_criterium: f64,
    /// 1 = stationary (Lyapunov) initialisation, 3 = diffuse (Durbin-Koopman).
    pub lik_init: u8,
    /// 0 = default, 1 = multivariate, 2 = univariate, 3 = multivariate diffuse, 4 = univariate diffuse.
    pub kalman_algo: u8,
    pub use_univariate_filters_if_singularity_is_detected: bool,
    pub rescale_prediction_error_covariance: bool,
    pub lyapunov_complex_threshold: f64,
    /// `lyapunov = ...` and its tolerances (lik_init = 1).
    pub lyapunov: lyapunov::LyapunovOptions,
    /// `options_.noconstant`: the constant of the measurement equation is zero (the data are
    /// not demeaned by the steady state of the observables; dsge_likelihood.m:204,
    /// DsgeSmoother.m:155).
    pub noconstant: bool,
    // ---- estimation completion ----
    /// DsgeSmoother.m: the multivariate smoother's error 422 (singular F in a stationary
    /// period) is an error (`print_info`) unless `use_univariate_smoother_if_singularity_is_detected`
    /// or `noprint` is set; false (switch to the univariate smoother silently) by default here,
    /// the estimation sets it as Dynare does (`estimation::est_completion::kalman_options`).
    pub smoother_singularity_error: bool,
    /// newrat's `mr_hessian` at the mode (`Hessian` 0/2): keep the univariate filters'
    /// per-observable densities (`lik(:)` of dsge_likelihood) in [`LikOutput::lik_by_obs`].
    pub obs_contributions: bool,
    // ---- end estimation completion ----
}

impl Default for KalmanOptions {
    fn default() -> Self {
        Self {
            kalman_tol: 1e-10,
            diffuse_kalman_tol: 1e-6,
            riccati_tol: 1e-6,
            presample: 0,
            qz_criterium: 1.0 + 1e-6,
            lik_init: 1,
            kalman_algo: 0,
            use_univariate_filters_if_singularity_is_detected: true,
            rescale_prediction_error_covariance: false,
            lyapunov_complex_threshold: 1e-15,
            lyapunov: Default::default(),
            noconstant: false,
            smoother_singularity_error: false, // ---- estimation completion ----
            obs_contributions: false,          // ---- estimation completion ----
        }
    }
}

/// Port of `describe_missing_data.m` (as stored in `dataset_info.missing`).
#[derive(Clone, Debug)]
pub struct MissingData {
    /// observed variables (0-based rows of y) per period
    pub aindex: Vec<Vec<usize>>,
    pub number_of_observations: usize,
    /// 0-based first period from which all later periods are fully observed
    /// (Dynare's 1-based `no_more_missing_observations` minus one).
    pub no_more_missing_observations: usize,
    pub any_missing: bool,
}

impl MissingData {
    /// From the data (pp x nobs, NaN = missing).
    pub fn from_data(y: &Mat) -> Self {
        let (pp, nobs) = (y.nrows(), y.ncols());
        let mut aindex = Vec::with_capacity(nobs);
        let mut cum = Vec::with_capacity(nobs);
        let mut c = 0usize;
        let mut n = 0usize;
        for t in 0..nobs {
            let idx: Vec<usize> = (0..pp).filter(|&i| !y[(i, t)].is_nan()).collect();
            c += pp - idx.len();
            n += idx.len();
            cum.push(c);
            aindex.push(idx);
        }
        let total_missing = pp * nobs - n;
        let no_more = if total_missing == 0 { 0 } else { cum.iter().position(|&x| x >= total_missing).unwrap() + 1 };
        Self {
            aindex,
            number_of_observations: n,
            no_more_missing_observations: no_more,
            any_missing: total_missing > 0,
        }
    }
}

/// Measurement-error covariance as seen by the filters.
#[derive(Clone, Debug)]
pub enum HMatrix {
    Zero,
    Diag(Vec<f64>),
    Full(Mat),
}

impl HMatrix {
    /// `H = 0` -> Zero; a matrix with |off-diagonal| < 1e-14 (dsge_likelihood's test) -> Diag.
    pub fn from_option(h: Option<&Mat>, pp: usize) -> Self {
        match h {
            None => HMatrix::Zero,
            Some(h) if h.iter().all(|&x| x == 0.0) => HMatrix::Zero,
            Some(h) => {
                assert_eq!((h.nrows(), h.ncols()), (pp, pp), "H must be pp x pp");
                let diag = (0..pp).all(|i| (0..pp).all(|j| i == j || h[(i, j)].abs() < 1e-14));
                if diag { HMatrix::Diag((0..pp).map(|i| h[(i, i)]).collect()) } else { HMatrix::Full(h.clone()) }
            }
        }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        match self {
            HMatrix::Zero => 0.0,
            HMatrix::Diag(d) => {
                if i == j {
                    d[i]
                } else {
                    0.0
                }
            }
            HMatrix::Full(h) => h[(i, j)],
        }
    }

    /// `H1` of the univariate filters (None if H is not diagonal: Dynare would augment the state).
    pub fn diag(&self, pp: usize) -> Option<Vec<f64>> {
        match self {
            HMatrix::Zero => Some(vec![0.0; pp]),
            HMatrix::Diag(d) => Some(d.clone()),
            HMatrix::Full(_) => None,
        }
    }
}

/// Kalman part of the likelihood.
#[derive(Clone, Debug)]
pub struct LikOutput {
    /// = LIK (+ dLIK) of dsge_likelihood, i.e. minus the log-likelihood without the prior,
    /// summed over the periods after `presample`; +Inf on failure.
    pub minus_loglik: f64,
    /// per-period minus log-density, all `nobs` periods (diffuse periods first, presample
    /// included; univariate filters: sum over the observables of each period).
    pub lik: Vec<f64>,
    /// 0, or dsge_likelihood's info code: 45 (NaN), 46 (complex), 50 (singular / Inf).
    pub info: i32,
    /// true if the (final) regular filter was univariate.
    pub univariate: bool,
    pub diffuse_periods: usize,
    /// ---- estimation completion ---- with `KalmanOptions::obs_contributions` and a univariate
    /// final filter: the per-observable densities (diffuse periods first), `None` otherwise.
    pub lik_by_obs: Option<Vec<f64>>,
}

impl LikOutput {
    fn fail(info: i32, diffuse_periods: usize) -> Self {
        Self {
            minus_loglik: f64::INFINITY,
            lik: Vec::new(),
            info,
            univariate: false,
            diffuse_periods,
            lik_by_obs: None,
        }
    }
}

/// The Kalman part of dsge_likelihood.m (allocating a workspace; prefer
/// [`KalmanWorkspace::likelihood`] in loops).
///
/// `y` is the demeaned data (pp x nobs, NaN = missing), `mf` the 0-based observed rows of T.
/// For lik_init = 3 pass Pstar/Pinf from [`compute_pinf_pstar`]; for lik_init = 1 pass the
/// [`lyapunov_solver`] Pstar (Pinf is ignored).
#[allow(clippy::too_many_arguments)]
pub fn likelihood(
    y: &Mat,
    missing: &MissingData,
    mf: &[usize],
    t: &Mat,
    r: &Mat,
    q: &Mat,
    h: Option<&Mat>,
    pstar: &Mat,
    pinf: &Mat,
    opts: &KalmanOptions,
) -> LikOutput {
    let mut ws = KalmanWorkspace::new(t.nrows(), y.nrows(), q.nrows());
    ws.likelihood(y, missing, mf, t, r, q, h, pstar, pinf, opts)
}

impl KalmanWorkspace {
    /// Port of the Kalman part of estimation/dsge_likelihood.m (sections 3 and 4):
    ///
    /// * lik_init = 1: kalman_algo 0/1 -> multivariate `kalman_filter` /
    ///   `missing_observations_kalman_filter`; 2 -> univariate.
    /// * lik_init = 3: kalman_algo 0/3 -> multivariate diffuse filter (`kalman_filter_d` /
    ///   `missing_observations_kalman_filter_d`), falling back to `univariate_kalman_filter_d`
    ///   (with the original Pstar) when it is singular (dLIK = Inf); 4 -> univariate diffuse.
    /// * then the multivariate regular filter (algo 1/3); if it returns Inf and
    ///   `use_univariate_filters_if_singularity_is_detected`, the univariate filter
    ///   (`univariate_kalman_filter`) from the same starting point (algo 2/4).
    ///
    /// With a non-diagonal H the univariate branches augment the state vector with the
    /// measurement errors as Dynare does (see [`augment`]); Dynare then keeps the augmented
    /// system (and H in the multivariate filter) for the rest of the evaluation.
    #[allow(clippy::too_many_arguments)]
    pub fn likelihood(
        &mut self,
        y: &Mat,
        missing: &MissingData,
        mf: &[usize],
        t: &Mat,
        r: &Mat,
        q: &Mat,
        h: Option<&Mat>,
        pstar: &Mat,
        pinf: &Mat,
        opts: &KalmanOptions,
    ) -> LikOutput {
        let (mm, pp) = (self.mm, self.pp);
        assert_eq!(y.nrows(), pp, "likelihood: y must be pp x nobs");
        assert_eq!(mf.len(), pp, "likelihood: mf length");
        let nobs = y.ncols();
        if nobs == 0 {
            return LikOutput::fail(50, 0);
        }
        let last = nobs - 1;
        let h = HMatrix::from_option(h, pp);
        let miss = if missing.any_missing { Some(missing) } else { None };
        self.set_system(t, r, q);
        // MATLAB's log(det(F)) of a negative det makes dLIK / LIK complex (info 46), unless
        // that filter's result is discarded (singular -> univariate fallback)
        let mut complex = false;
        let zeros = self.a_init.clone().unwrap_or_else(|| vec![0.0; mm]); // filter_initial_state
        let zeros_h = vec![0.0; pp];
        // expanded_state_vector_for_univariate_filter: (workspace, observed rows)
        let mut aug: Option<(Box<KalmanWorkspace>, Vec<usize>)> = None;
        // heteroskedastic_filter: the augmented system uses blkdiag(Qvec(:,:,k), H)
        let hetero_q: Option<Vec<Mat>> = self.hetero_qvec();
        let new_aug = |a0: &[f64], p0: &Mat, pinf0: Option<&Mat>| {
            let HMatrix::Full(hm) = &h else { unreachable!() };
            let s = augment::system(t, r, q, hm, mf);
            let mut ws = Box::new(KalmanWorkspace::new(mm + pp, pp, q.nrows() + pp));
            if let Some(qv) = &hetero_q {
                ws.set_heteroskedastic(Some(qv.iter().map(|qk| augment::system(t, r, qk, hm, mf).q).collect()));
            }
            ws.set_system(&s.t, &s.r, &s.q);
            let pinf_c = pinf0.map(|p| augment::cov(p, None, mf));
            ws.set_state(&augment::state(a0, mf), &augment::cov(p0, Some(hm), mf), pinf_c.as_ref());
            (ws, s.mf)
        };

        let mut kalman_algo = opts.kalman_algo;
        let mut diffuse_periods = 0usize;
        let mut dlik_total = 0.0;
        // per-period densities of the diffuse stage
        let mut dlik_periods: Vec<f64> = Vec::new();
        let mut dlik_obs_periods: Vec<f64> = Vec::new(); // ---- estimation completion ----
        match opts.lik_init {
            1 | 2 | 4 | 5 => {
                // [stream X] lik_init 2/4/5: standard filter from kalman::lik_init's Pstar
                if kalman_algo != 2 {
                    kalman_algo = 1;
                }
                self.set_state(&zeros, pstar, None);
            }
            3 => {
                if kalman_algo == 0 {
                    kalman_algo = 3;
                } else if kalman_algo != 3 && kalman_algo != 4 {
                    panic!(
                        "The model requires Diffuse filter, but you specified a different Kalman filter (kalman_algo must be 0, 3 or 4)"
                    );
                }
                let mut singular_diffuse = false;
                if kalman_algo == 3 {
                    self.set_state(&zeros, pstar, Some(pinf));
                    let (d, n) = self.diffuse_multivariate(
                        y,
                        miss,
                        mf,
                        &h,
                        0,
                        last,
                        opts.kalman_tol,
                        opts.diffuse_kalman_tol,
                        opts.presample,
                    );
                    dlik_total = d;
                    diffuse_periods = n;
                    dlik_periods.clone_from(&self.dlik);
                    if d.is_infinite() {
                        singular_diffuse = true;
                    } else {
                        complex = self.complex;
                    }
                }
                if singular_diffuse || kalman_algo == 4 {
                    let (d, n) = match h.diag(pp) {
                        Some(h1) => {
                            self.set_state(&zeros, pstar, Some(pinf));
                            let r = self.diffuse_univariate(
                                y,
                                miss,
                                mf,
                                &h1,
                                0,
                                last,
                                opts.kalman_tol,
                                opts.diffuse_kalman_tol,
                                opts.presample,
                            );
                            dlik_periods.clone_from(&self.dlik);
                            if opts.obs_contributions {
                                dlik_obs_periods.clone_from(&self.dlik_by_obs);
                            }
                            r
                        }
                        None => {
                            let (mut ws, mfc) = new_aug(&zeros, pstar, Some(pinf));
                            let r = ws.diffuse_univariate(
                                y,
                                miss,
                                &mfc,
                                &zeros_h,
                                0,
                                last,
                                opts.kalman_tol,
                                opts.diffuse_kalman_tol,
                                opts.presample,
                            );
                            dlik_periods.clone_from(&ws.dlik);
                            if opts.obs_contributions {
                                dlik_obs_periods.clone_from(&ws.dlik_by_obs);
                            }
                            aug = Some((ws, mfc));
                            r
                        }
                    };
                    dlik_total = d;
                    diffuse_periods = n;
                }
                if dlik_total.is_nan() {
                    return LikOutput::fail(45, diffuse_periods);
                }
            }
            other => panic!("likelihood: lik_init = {other} is not supported (only 1 and 3)"),
        }

        let start = diffuse_periods;
        let mut lik_total = f64::NAN;
        let mut univariate = false;
        if kalman_algo == 1 || kalman_algo == 3 {
            let (w, mfu): (&mut KalmanWorkspace, &[usize]) = match aug.as_mut() {
                Some((ws, m)) => (ws, m),
                None => (&mut *self, mf),
            };
            w.save_state();
            let l = w.multivariate(
                y,
                miss,
                mfu,
                &h,
                start,
                last,
                opts.kalman_tol,
                opts.riccati_tol,
                opts.rescale_prediction_error_covariance,
                opts.presample,
                diffuse_periods,
            );
            if !l.is_infinite() {
                complex |= w.complex;
            }
            if l.is_infinite() {
                if opts.use_univariate_filters_if_singularity_is_detected {
                    kalman_algo = if kalman_algo == 1 { 2 } else { 4 };
                    w.restore_state();
                } else {
                    return LikOutput::fail(50, diffuse_periods);
                }
            } else {
                lik_total = l;
                if opts.lik_init == 3 {
                    lik_total += dlik_total;
                }
            }
        }
        if kalman_algo == 2 || kalman_algo == 4 {
            if start > last {
                return LikOutput::fail(50, diffuse_periods);
            }
            let h1 = match (h.diag(pp), aug.is_some()) {
                (_, true) => zeros_h.clone(),
                (Some(h1), false) => h1,
                (None, false) => {
                    // augment from the current state (after the diffuse periods)
                    let (a0, p0) = (self.state(), self.cov());
                    aug = Some(new_aug(&a0, &p0, None));
                    zeros_h.clone()
                }
            };
            let (w, mfu): (&mut KalmanWorkspace, &[usize]) = match aug.as_mut() {
                Some((ws, m)) => (ws, m),
                None => (&mut *self, mf),
            };
            univariate = true;
            lik_total = w.univariate(
                y,
                miss,
                mfu,
                &h1,
                start,
                last,
                opts.kalman_tol,
                opts.riccati_tol,
                opts.presample,
                diffuse_periods,
            );
            if opts.lik_init == 3 {
                lik_total += dlik_total;
            }
        }
        if lik_total.is_nan() {
            return LikOutput::fail(45, diffuse_periods);
        }
        if complex {
            return LikOutput::fail(46, diffuse_periods);
        }
        if lik_total.is_infinite() {
            return LikOutput::fail(50, diffuse_periods);
        }
        let final_lik = match &aug {
            Some((ws, _)) => &ws.lik,
            None => &self.lik,
        };
        let mut lik = Vec::with_capacity(nobs);
        lik.extend_from_slice(&dlik_periods[..diffuse_periods.min(dlik_periods.len())]);
        lik.extend_from_slice(final_lik);
        // ---- estimation completion ---- per-observable densities (newrat's OPG Hessian)
        let lik_by_obs = (opts.obs_contributions && univariate).then(|| {
            let (ws, pp_ws) = match &aug {
                Some((ws, _)) => (&**ws, ws.pp),
                None => (&*self, self.pp),
            };
            let mut v = Vec::with_capacity(nobs * pp_ws);
            let nd = diffuse_periods.min(dlik_periods.len());
            if nd > 0 {
                if dlik_obs_periods.len() >= nd * pp_ws {
                    v.extend_from_slice(&dlik_obs_periods[..nd * pp_ws]);
                } else {
                    v.extend_from_slice(&dlik_periods[..nd]);
                }
            }
            v.extend_from_slice(&ws.lik_by_obs);
            v
        });
        LikOutput { minus_loglik: lik_total, lik, info: 0, univariate, diffuse_periods, lik_by_obs }
    }
}

#[cfg(test)]
mod tests;
