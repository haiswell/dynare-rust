// SPDX-License-Identifier: GPL-3.0-or-later
//! Bayesian estimation core: priors, bounds, mode finding (csminwel, newrat,
//! gmhmaxlik, simplex), numerical Hessian, random-walk Metropolis–Hastings and posterior
//! statistics. See DESIGN.md, section `estimation`.
//!
//! Everything here is generic over an [`Objective`]: Dynare's *minus* log
//! posterior kernel (`dsge_likelihood.m`'s `fval`). The likelihood itself is
//! assembled in `likelihood.rs` (integration).

pub mod analytic; // ---- estimation completion ---- analytic_derivation
pub mod conditional_likelihood;
pub mod convergence;
pub mod diagnostics;
pub mod dsge_var;
pub mod dsge_var_irf;
pub mod est_completion; // ---- estimation completion ----
pub mod hessian;
pub mod init_states; // ---- estimation completion ---- estimate_initial_states_endogenous_prior
pub mod mcmc;
pub mod mh_files;
pub mod mh_session;
pub mod mode_check;
pub mod mode_hessian; // ---- estimation completion ---- Hessian at the mode (cova_compute, use_penalized_objective_for_hessian, newrat Hessian)
pub mod model_comparison; // stream B
pub mod optim;
pub mod plots;
pub mod posterior;
pub mod prior;
pub mod prior_posterior; // stream F
pub mod q3;
pub mod q3_mcmc;
pub mod q3_occbin;
pub mod samplers;
pub mod skewed;
pub mod special;

pub use hessian::hessian;
pub use mcmc::{
    Chain, MhOptions, MhProgress, MhResult, TuneOptions, calibrate_mh_scale_parameter, mh_tune_jscale, rw_metropolis,
};
pub use optim::{
    CsminwelOptions, FminsearchOptions, GmhmaxlikOptions, GmhmaxlikResult, GradientMethod, NewratOptions, OptimResult,
    SimplexOptions, VarInit, csminwel, dynare_simplex, fminsearch, gmhmaxlik, newrat,
};
pub use posterior::{
    BrooksGelman, MhmResult, ParamMoments, PosteriorOptions, PosteriorStats, brooks_gelman, inefficiency_factors,
    inverse_hessian, laplace_approximation, mcmc_ifac, modified_harmonic_mean, posterior_moments, posterior_stats,
};
pub use prior::{
    Bounds, Prior, PriorShape, estimation_bounds, log_prior_density, mcmc_bounds, prior_bounds, prior_draw,
};

/// Full result of one objective evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// Minus log posterior kernel (+inf on failure).
    pub fval: f64,
    /// Dynare's `info(1)` (0 = success).
    pub info: i32,
    /// Dynare's `info(4)`: the endogenous penalty that
    /// `penalty_objective_function.m` adds to the base penalty on failure.
    pub info4: f64,
    /// `DLIK = [-lnprior; lik(:)]` of `dsge_likelihood.m` (prior and
    /// per-observation contributions, summing to `fval`), only when requested
    /// and available. Used by newrat's outer-product-gradient Hessian.
    pub contributions: Option<Vec<f64>>,
}

/// An objective to *minimize*: Dynare's minus log posterior kernel
/// `-(loglik + logprior)`; `+inf` and a nonzero info code on failure.
pub trait Objective: Sync {
    /// Number of estimated parameters.
    fn dim(&self) -> usize;

    /// `(fval, info)`.
    fn eval(&self, x: &[f64]) -> (f64, i32);

    /// Analytic gradient of the objective (method of moments with `analytic_jacobian`);
    /// `None` when [`Objective::has_gradient`] is false or the evaluation fails.
    fn gradient(&self, _x: &[f64]) -> Option<Vec<f64>> {
        None
    }

    /// `options_.mom.analytic_jacobian == 1`: the objective returns its gradient
    /// ([`Objective::gradient`]), and `dynare_minimize_objective.m` runs fmincon, fminunc,
    /// csminwel and solvopt with analytic gradients, as under `analytic_derivation`.
    fn has_gradient(&self) -> bool {
        false
    }

    /// Evaluation with Dynare's `info(4)` and, if `contributions` is true and
    /// the objective can provide them, the likelihood contributions. The
    /// default wraps [`Objective::eval`] with `info(4) = 0.1` (the value
    /// `dsge_likelihood.m` uses for most failures) and no contributions.
    fn eval_full(&self, x: &[f64], contributions: bool) -> Evaluation {
        let _ = contributions;
        let (fval, info) = self.eval(x);
        Evaluation { fval, info, info4: if info != 0 { 0.1 } else { 0.0 }, contributions: None }
    }

    // ---- estimation completion ----
    /// `analytic_derivation`: the evaluation with Dynare's `DLIK` (and `Hess`), `None` when
    /// the objective has no analytic derivatives (the default).
    fn eval_grad(&self, x: &[f64], hess: analytic::AnalyticHessian) -> Option<analytic::GradEvaluation> {
        let _ = (x, hess);
        None
    }

    /// `estimate_initial_states_endogenous_prior`: the slice sampler's initial-state steps
    /// (`None`: no estimated initial states, the default).
    fn init_state_ops(&self) -> Option<&dyn init_states::InitStateOps> {
        None
    }
    // ---- end estimation completion ----
}

impl<T: Objective + ?Sized> Objective for &T {
    fn dim(&self) -> usize {
        (**self).dim()
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        (**self).eval(x)
    }
    fn gradient(&self, x: &[f64]) -> Option<Vec<f64>> {
        (**self).gradient(x)
    }
    fn has_gradient(&self) -> bool {
        (**self).has_gradient()
    }
    fn eval_full(&self, x: &[f64], contributions: bool) -> Evaluation {
        (**self).eval_full(x, contributions)
    }
    fn eval_grad(&self, x: &[f64], hess: analytic::AnalyticHessian) -> Option<analytic::GradEvaluation> {
        (**self).eval_grad(x, hess) // ---- estimation completion ----
    }
    fn init_state_ops(&self) -> Option<&dyn init_states::InitStateOps> {
        (**self).init_state_ops() // ---- estimation completion ----
    }
}

/// A closure as an objective (`info` = 0 when finite, 1 otherwise).
pub struct FnObjective<F: Fn(&[f64]) -> f64 + Sync> {
    pub n: usize,
    pub f: F,
}

impl<F: Fn(&[f64]) -> f64 + Sync> Objective for FnObjective<F> {
    fn dim(&self) -> usize {
        self.n
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        let v = (self.f)(x);
        (v, if v.is_finite() { 0 } else { 1 })
    }
}

/// Port of `optimization/penalty_objective_function.m`:
/// on failure (`info != 0`) returns `base_penalty + info(4)`.
/// Returns `(fval, exit_flag, contributions)`; `exit_flag` is false on failure
/// (`dsge_likelihood`'s `exit_flag = 0`). A non-finite value with `info == 0`
/// is also treated as a failure (generic objectives do not set `info`).
pub fn penalty_objective(
    obj: &dyn Objective,
    x: &[f64],
    base_penalty: f64,
    contributions: bool,
) -> (f64, bool, Option<Vec<f64>>) {
    let e = obj.eval_full(x, contributions);
    if e.info != 0 || !e.fval.is_finite() {
        let info4 = if e.info != 0 { e.info4 } else { 0.1 };
        (base_penalty + info4, false, None)
    } else {
        (e.fval, true, e.contributions)
    }
}
