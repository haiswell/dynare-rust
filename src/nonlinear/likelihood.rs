// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `estimation/non_linear_dsge_likelihood.m` as an [`Objective`].

use super::rng::{Draws, SeededDraws};
use super::{FilterOutput, ParticleOptions, ReducedForm, run_filter};
use crate::dsge::{DsgeModel, EvalFailure};
use crate::estimation::{Bounds, Evaluation, Objective};
use crate::kalman::lyapunov_symm;
use crate::linalg::Mat;
use crate::model::higher::DynamicDerivatives;

use crate::solver::korder;
use anyhow::{Result, bail};

/// The model, data and filter settings of a nonlinear estimation.
pub struct NonlinearModel {
    pub dsge: DsgeModel,
    /// Approximation order: 2 or 3 (`non_linear_dsge_likelihood.m`), or 1 for the online
    /// filter (`solve_model_for_online_filter.m`'s `order < 2` branch).
    pub order: usize,
    /// Second (and third) derivatives of the dynamic model (`None` at order 1).
    deriv: Option<DynamicDerivatives>,
    pub particle: ParticleOptions,
    /// Observations, pp x T (varobs order, not demeaned).
    pub y: Mat,
    /// `options_.presample` (the likelihood sums `lik(presample+1:end)`).
    pub presample: usize,
    /// Seed of the per-evaluation random stream.
    pub seed: u64,
    /// `options_.qz_criterium` (`select_qz_criterium_value.m`), used by the first-order solver
    /// (`resol`) and by the Lyapunov equation of `initialization = 1`; see
    /// [`Self::set_qz_criterium`].
    pub qz_criterium: f64,
    pub lyapunov_complex_threshold: f64,
}

/// One evaluation.
pub struct NlEvaluated {
    pub fval: f64,
    pub minus_loglik: f64,
    pub log_prior: f64,
    pub filter: FilterOutput,
}

impl NonlinearModel {
    /// `y`: pp x T observations (varobs order). `options_.qz_criterium` is Dynare's default for
    /// the filter ([`Self::set_qz_criterium`] without a user value); order 1 only serves the
    /// online filter.
    pub fn new(
        dsge: DsgeModel,
        order: usize,
        particle: ParticleOptions,
        y: Mat,
        presample: usize,
        seed: u64,
    ) -> Result<Self> {
        if !(1..=3).contains(&order) {
            // Dynare 7.2 runs order > 3 through `local_state_space_iteration_k`, which
            // dynare-rust does not port (its particle filters stop at order 3)
            bail!("non_linear_dsge_likelihood: dynare-rust supports order 2 or 3 (got {order})");
        }
        let pp = dsge.varobs.len();
        if y.nrows() != pp {
            bail!("non_linear_dsge_likelihood: the data has {} rows, {pp} observables", y.nrows());
        }
        if y.iter().any(|v| v.is_nan()) {
            bail!("non_linear_dsge_likelihood: the nonlinear filters do not handle missing observations");
        }
        let deriv = (order >= 2).then(|| dsge.model.dynamic_derivatives(order));
        let mut nm = NonlinearModel {
            dsge,
            order,
            deriv,
            particle,
            y,
            presample,
            seed,
            qz_criterium: 1.0 + 1e-6,
            lyapunov_complex_threshold: 1e-15,
        };
        nm.set_qz_criterium(None)?;
        Ok(nm)
    }

    /// `options_.qz_criterium` after `select_qz_criterium_value.m` (`user`: the
    /// `qz_criterium` option), given to `resol` (Blanchard-Kahn conditions) and to the
    /// ergodic variance of `initialization = 1`. At order 1 (online filter) Dynare takes the
    /// linear-filter branch (`options_.particle.status` is false, `lik_init = 1`).
    pub fn set_qz_criterium(&mut self, user: Option<f64>) -> Result<()> {
        let init = (self.order > 1).then_some(self.particle.initialization);
        self.qz_criterium = super::select_qz_criterium_value(user, init, 1.0)?;
        self.dsge
            .set_solver_options(crate::solver::SolverOptions { qz_criterium: self.qz_criterium, ..Default::default() });
        Ok(())
    }

    /// `M_.H` at parameter vector `x`.
    pub fn h_matrix(&self, x: &[f64]) -> Mat {
        // stream M's M_.H (calibration from the shocks block, estimated stderr/corr entries)
        self.dsge.set_all_parameters_h(x).2
    }

    /// The `ReducedForm` of `non_linear_dsge_likelihood.m` at `x` (after `resol`), including the
    /// initial state distribution; `draws` is only used by `initialization = 2`. At order 1,
    /// the `order < 2` branch of `solve_model_for_online_filter.m`: `constant = steadystate`
    /// and zero `ghxx`, `ghuu`, `ghxu`.
    pub fn reduced_form(&self, x: &[f64]) -> Result<ReducedForm, EvalFailure> {
        let res = self.dsge.resolve(x)?;
        let model = &self.dsge.model;
        let kdr = match &self.deriv {
            Some(deriv) => {
                let x_ss = vec![0.0; model.n_exo()];
                let g1 = model.dynamic_jacobian(&res.steady_state, &x_ss, &res.params);
                let vals = deriv.eval(&res.steady_state, &x_ss, &res.params);
                let kdr = korder::solve(&res.dr, &model.lead_lag_incidence, &g1, deriv, &vals, &res.q, self.order)
                    .map_err(|_| EvalFailure { info: 9, penalty: 0.1 })?;
                Some(kdr)
            }
            None => None,
        };
        let idx = &self.dsge.ss_index;
        let rv = &idx.restrict_var_list;
        let dr = kdr.as_ref().map_or(&res.dr, |k| &k.first);
        let rows = |m: &Mat| -> Mat { super::linalg::select_rows(m, rv) };
        let steadystate: Vec<f64> = rv.iter().map(|&k| dr.ys[dr.order_var[k]]).collect();
        let ghs2: Vec<f64> = match &kdr {
            Some(k) => rv.iter().map(|&j| k.ghs2[(j, 0)]).collect(),
            None => vec![0.0; rv.len()],
        };
        let constant: Vec<f64> = steadystate.iter().zip(&ghs2).map(|(s, g)| s + 0.5 * g).collect();
        let mf0 = idx.mf0.clone();
        let state_ss: Vec<f64> = mf0.iter().map(|&j| steadystate[j]).collect();
        let empty = Mat::zeros(rv.len(), 0);
        let o3 = self.order == 3;
        // order >= 2 terms (zero matrices of the right width at order 1)
        let (nx, nu) = (dr.ghx.ncols(), dr.ghu.ncols());
        let higher = |f: fn(&korder::KOrderDr) -> &Mat, cols: usize| -> Mat {
            kdr.as_ref().map_or_else(|| Mat::zeros(rv.len(), cols), |k| rows(f(k)))
        };
        let mut rf = ReducedForm {
            order: self.order,
            steadystate,
            state_mean: mf0.iter().map(|&j| constant[j]).collect(),
            constant,
            state_ss,
            q: res.q.clone(),
            h: self.h_matrix(x),
            mf0: mf0.clone(),
            mf1: idx.mf.clone(),
            ghx: rows(&dr.ghx),
            ghu: rows(&dr.ghu),
            ghxx: higher(|k| &k.ghxx, nx * nx),
            ghuu: higher(|k| &k.ghuu, nu * nu),
            ghxu: higher(|k| &k.ghxu, nx * nu),
            ghs2,
            ghxxx: if o3 { higher(|k| &k.ghxxx, 0) } else { empty.clone() },
            ghuuu: if o3 { higher(|k| &k.ghuuu, 0) } else { empty.clone() },
            ghxxu: if o3 { higher(|k| &k.ghxxu, 0) } else { empty.clone() },
            ghxuu: if o3 { higher(|k| &k.ghxuu, 0) } else { empty.clone() },
            ghxss: if o3 { higher(|k| &k.ghxss, 0) } else { empty.clone() },
            ghuss: if o3 { higher(|k| &k.ghuss, 0) } else { empty },
            state_variance: Mat::zeros(0, 0),
        };
        let ns = mf0.len();
        rf.state_variance = match self.particle.initialization {
            1 => {
                let rqr = &res.r * &res.q * res.r.transpose();
                let v = lyapunov_symm(&res.t, &rqr, self.qz_criterium, self.lyapunov_complex_threshold)
                    .map_err(|_| EvalFailure { info: 201, penalty: 0.1 })?;
                Mat::from_fn(ns, ns, |i, j| v[(mf0[i], mf0[j])])
            }
            2 => {
                // Monte Carlo estimate of the ergodic variance (simult, 5000 periods, 3000 kept)
                // order 1 (online filter): simult of the first-order rule, i.e. the order-2
                // recursion with zero second-order terms
                let first_order = || {
                    let (n, nx, nu) = (res.dr.ghx.nrows(), res.dr.ghx.ncols(), res.dr.ghu.ncols());
                    let zeros = |c: usize| Mat::zeros(n, c);
                    korder::KOrderDr {
                        order: 2,
                        first: res.dr.clone(),
                        n_exo: nu,
                        ghxx: zeros(nx * nx),
                        ghxu: zeros(nx * nu),
                        ghuu: zeros(nu * nu),
                        ghs2: zeros(1),
                        ghxxx: zeros(0),
                        ghxxu: zeros(0),
                        ghxuu: zeros(0),
                        ghuuu: zeros(0),
                        ghxss: zeros(0),
                        ghuss: zeros(0),
                        derivs: Vec::new(),
                    }
                };
                let kdr1;
                let kdr = match &kdr {
                    Some(k) => k,
                    None => {
                        kdr1 = first_order();
                        &kdr1
                    }
                };
                let mut draws = SeededDraws::new(self.seed.wrapping_add(0x5157));
                let nu = res.q.nrows();
                let cq = super::linalg::chol_upper(&res.q).ok_or(EvalFailure { info: 201, penalty: 0.1 })?;
                let ex = draws.randn_mat(5000, nu) * cq;
                let sim = korder::simulate::simult(kdr, &res.steady_state, &ex, self.particle.pruning);
                let decl: Vec<usize> = mf0.iter().map(|&j| dr.order_var[rv[j]]).collect();
                let cols: Vec<usize> = (2000..5000).collect();
                // `any(isnan(y_)) || any(isinf(y_)) && ~options_.pruning`: && binds tighter than
                // ||, so a NaN always fails and pruning only excuses an Inf
                let any = |f: fn(f64) -> bool| cols.iter().any(|&c| decl.iter().any(|&d| f(sim[(d, c)])));
                if initial_states_fail(any(f64::is_nan), any(f64::is_infinite), self.particle.pruning) {
                    return Err(EvalFailure { info: 202, penalty: 0.1 });
                }
                let nk = cols.len() as f64;
                let mean: Vec<f64> =
                    decl.iter().map(|&d| cols.iter().map(|&c| sim[(d, c)]).sum::<f64>() / nk).collect();
                Mat::from_fn(ns, ns, |i, j| {
                    cols.iter().map(|&c| (sim[(decl[i], c)] - mean[i]) * (sim[(decl[j], c)] - mean[j])).sum::<f64>()
                        / (nk - 1.0)
                })
            }
            3 => Mat::identity(ns, ns) * self.particle.initial_state_prior_std,
            k => panic!("Unknown initialization option {k}!"),
        };
        if super::linalg::chol_upper(&rf.state_variance).is_none() {
            return Err(EvalFailure { info: 201, penalty: 0.1 });
        }
        Ok(rf)
    }

    /// `non_linear_dsge_likelihood(x, ...)` with the random stream `draws`.
    pub fn evaluate_with(&self, x: &[f64], bounds: &Bounds, draws: &mut dyn Draws) -> Result<NlEvaluated, EvalFailure> {
        // check_bounds_and_definiteness_estimation.m
        if x.iter().zip(&bounds.lb).any(|(v, l)| v < l) {
            let p = x.iter().zip(&bounds.lb).filter(|(v, l)| v < l).map(|(v, l)| (l - v).powi(2)).sum();
            return Err(EvalFailure { info: 41, penalty: p });
        }
        if x.iter().zip(&bounds.ub).any(|(v, u)| v > u) {
            let p = x.iter().zip(&bounds.ub).filter(|(v, u)| v > u).map(|(v, u)| (v - u).powi(2)).sum();
            return Err(EvalFailure { info: 42, penalty: p });
        }
        let rf = self.reduced_form(x)?;
        let out = match run_filter(&rf, &self.y, self.presample, &self.particle, draws) {
            Ok(o) => o,
            // a MATLAB error inside the filter (non-positive-definite square roots, ...)
            Err(_) => return Err(EvalFailure { info: 45, penalty: 0.1 }),
        };
        let lik = out.lik_total;
        if lik.is_nan() {
            return Err(EvalFailure { info: 45, penalty: 0.1 });
        }
        if lik.is_infinite() {
            return Err(EvalFailure { info: 50, penalty: 0.1 });
        }
        let lnprior = self.dsge.log_prior(x);
        if lnprior.is_nan() {
            return Err(EvalFailure { info: 47, penalty: 0.1 });
        }
        if lnprior.is_infinite() {
            return Err(EvalFailure { info: 40, penalty: 0.1 });
        }
        Ok(NlEvaluated { fval: lik - lnprior, minus_loglik: lik, log_prior: lnprior, filter: out })
    }

    /// One evaluation with a fresh seeded stream (Dynare's per-call seed reset).
    pub fn evaluate(&self, x: &[f64], bounds: &Bounds) -> Result<NlEvaluated, EvalFailure> {
        let mut d = SeededDraws::new(self.seed);
        self.evaluate_with(x, bounds, &mut d)
    }
}

/// `non_linear_dsge_likelihood.m` as an objective (`fval = -(loglik + logprior)`).
pub struct NonlinearPosterior<'a> {
    pub model: &'a NonlinearModel,
    pub bounds: &'a Bounds,
}

impl Objective for NonlinearPosterior<'_> {
    fn dim(&self) -> usize {
        self.model.dsge.n_estimated()
    }

    fn eval(&self, x: &[f64]) -> (f64, i32) {
        let r = match self.model.evaluate(x, self.bounds) {
            Ok(e) => (e.fval, 0),
            Err(f) => (f64::INFINITY, f.info),
        };
        crate::progress::eval(r.0);
        r
    }

    fn eval_full(&self, x: &[f64], contributions: bool) -> Evaluation {
        let r = self.model.evaluate(x, self.bounds);
        crate::progress::eval(r.as_ref().map_or(f64::INFINITY, |e| e.fval));
        match r {
            Ok(e) => {
                let contributions = contributions.then(|| {
                    let start = self.model.presample.min(e.filter.lik.len());
                    std::iter::once(-e.log_prior).chain(e.filter.lik[start..].iter().map(|v| -v)).collect()
                });
                Evaluation { fval: e.fval, info: 0, info4: 0.0, contributions }
            }
            Err(f) => Evaluation { fval: f64::INFINITY, info: f.info, info4: f.penalty, contributions: None },
        }
    }
}

/// `any(any(isnan(y_))) || any(any(isinf(y_))) && ~options_.pruning` of
/// `non_linear_dsge_likelihood.m` (`initialization = 2`): `&&` binds tighter than `||`, so a NaN
/// always fails and pruning only excuses an Inf.
pub fn initial_states_fail(any_nan: bool, any_inf: bool, pruning: bool) -> bool {
    any_nan || (any_inf && !pruning)
}

#[cfg(test)]
mod tests {
    use super::initial_states_fail;

    #[test]
    fn initialization_two_failure_follows_matlab_precedence() {
        // (nan, inf, pruning) -> info 202
        for (nan, inf, pruning, fail) in [
            (false, false, false, false),
            (false, false, true, false),
            (true, false, false, true),
            (true, false, true, true),
            (false, true, false, true),
            (false, true, true, false),
            (true, true, true, true),
        ] {
            assert_eq!(initial_states_fail(nan, inf, pruning), fail, "nan={nan} inf={inf} pruning={pruning}");
        }
    }
}
