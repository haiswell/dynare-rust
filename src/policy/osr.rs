// SPDX-License-Identifier: GPL-3.0-or-later
//! Optimal simple rules: port of `+osr/run.m`, `+osr/objective.m` and
//! `+osr/get_variance_of_endogenous_variables.m` (order-1 `optim_weights` loss, or minus the
//! unconditional `planner_objective` welfare at order 1, order 2 or in the linear-quadratic
//! case), with `osr_params_bounds`; `analytic_derivation` (the analytic gradient of the
//! `optim_weights` loss, `+osr/objective.m` with `get_perturbation_params_derivs.m` and
//! `pruned_state_space_system.m`) and `huge_number`.

use anyhow::{Result, bail};

use super::{Resolver, welfare};
use crate::estimation::{Bounds, Evaluation, Objective, OptimResult};
use crate::identification::derivs::{ModelParamDerivs, ParamSet, perturbation_derivs};
use crate::identification::moments::pruned_state_space;
use crate::linalg::Mat;
use crate::model::Model;
use crate::model::higher::DynamicDerivatives;

/// The branch of `evaluate_planner_objective.m` that `osr.objective` reaches: `resol` solves at
/// `options_.order`, and no `stoch_simul` resets the order when the Hessian is zero.
pub enum WelfareKind {
    /// `options_.order = 1`.
    Order1,
    /// `options_.order = 2`, zero model Hessian.
    LinearQuadratic,
    /// `options_.order = 2` (second-order decision rule from these derivatives).
    Order2(DynamicDerivatives),
}

/// The OSR loss as a function of the `osr_params`.
pub struct OsrObjective<'a> {
    pub resolver: &'a Resolver<'a>,
    pub base_params: Vec<f64>,
    pub i_params: Vec<usize>,
    /// `optim_weights` loss: DR positions `inv_order_var(i_var)` and `W(i_var, i_var)`;
    /// `None` = minus the unconditional planner welfare.
    pub weights: Option<(Vec<usize>, Mat)>,
    pub sigma_e: Mat,
    /// Which `evaluate_planner_objective` branch the welfare loss takes.
    pub welfare: WelfareKind,
    /// `options_.analytic_derivation` (with `optim_weights` only): the symbolic parameter
    /// derivatives of the static and dynamic models.
    pub analytic: Option<ModelParamDerivs>,
}

/// `options_.schur_vec_tol` (pruned_state_space_system.m's test for non-stationary variables).
const SCHUR_VEC_TOL: f64 = 1e-11;

impl OsrObjective<'_> {
    /// `osr.objective`: `(loss, info, info(4))`.
    pub fn loss(&self, x: &[f64]) -> (f64, i32, f64) {
        let (l, i, i4, _) = self.loss_grad(x);
        (l, i, i4)
    }

    /// `[loss, info, ~, df] = osr.objective(...)`: `(loss, info, info(4), df)`, `df` = `None`
    /// unless `analytic_derivation` (and the evaluation succeeded).
    ///
    /// Under `analytic_derivation` the loss is `W(:)' * (Var_y + E_y E_y')(:)` from
    /// `pruned_state_space_system.m` (order 1: `E_y` = the steady state of the targets, so a
    /// non-zero steady state adds to the loss, unlike the `lyapunov_symm` loss of
    /// `get_variance_of_endogenous_variables.m`), and `df(j) = W(:)' * dE_yy(:, j)` with
    /// `dE_yy = dVar_y + dE_y E_y' + E_y dE_y'`. `analytic_derivation_mode` 0 and 1 (Sylvester or
    /// Kronecker solution of the same linear equations) give the same `dr.derivs`.
    pub fn loss_grad(&self, x: &[f64]) -> (f64, i32, f64, Option<Vec<f64>>) {
        let mut p = self.base_params.clone();
        for (&i, &v) in self.i_params.iter().zip(x) {
            p[i] = v;
        }
        let res = match &self.welfare {
            WelfareKind::Order2(d) if self.weights.is_none() => {
                self.resolver.resolve_k(&p, 2, d, &self.sigma_e).map(|(k, p)| (k.first.clone(), Some(k), p))
            }
            _ => self.resolver.resolve_params(&p).map(|(d, p)| (d, None, p)),
        };
        let (dr, kdr, p) = match res {
            Ok(x) => x,
            Err(e) => {
                let i4 = if [3, 4, 5, 6, 19, 20, 21, 23, 26, 81, 84, 85].contains(&e.info) && e.info2.is_finite() {
                    e.info2
                } else {
                    0.1
                };
                return (1e8, e.info, i4, None);
            }
        };
        let mut grad = None;
        let loss = match &self.weights {
            Some((rows, w)) if self.analytic.is_some() => match self.analytic_loss(&dr, &p, rows, w) {
                Ok((l, g)) => {
                    grad = Some(g);
                    l
                }
                Err(_) => f64::NAN,
            },
            Some((rows, w)) => {
                let vx = welfare::variance(&dr, rows, &self.sigma_e, self.resolver.qz_criterium);
                w.iter().zip(vx.iter()).map(|(a, b)| a * b).sum::<f64>()
            }
            None => {
                let m = self.resolver.model;
                let qz = self.resolver.qz_criterium;
                let v = match (&self.welfare, &kdr) {
                    (WelfareKind::Order2(_), Some(k)) => {
                        welfare::evaluate_planner_objective_order2(m, &p, k, &self.sigma_e, qz)
                    }
                    (WelfareKind::LinearQuadratic, _) => {
                        welfare::evaluate_planner_objective(m, &p, &dr, &self.sigma_e, true, qz)
                    }
                    _ => welfare::evaluate_planner_objective(m, &p, &dr, &self.sigma_e, false, qz),
                };
                v.map(|v| -v.unconditional).unwrap_or(f64::NAN)
            }
        };
        if loss.is_infinite() {
            return (1e8, 174, 0.1, None);
        }
        if loss.is_nan() {
            return (1e8, 175, 0.1, None);
        }
        (loss, 0, 0.0, grad)
    }

    /// The `analytic_derivation` branch of `osr.objective` (see [`OsrObjective::loss_grad`]).
    fn analytic_loss(
        &self,
        dr: &crate::solver::DecisionRule,
        p: &[f64],
        rows: &[usize],
        w: &Mat,
    ) -> Result<(f64, Vec<f64>)> {
        let pd = self.analytic.as_ref().expect("analytic derivatives");
        let m = self.resolver.model;
        let qz = self.resolver.qz_criterium;
        let set = ParamSet { model: self.i_params.clone(), ..Default::default() };
        let g1 = m.dynamic_jacobian(&dr.ys, &self.resolver.exo_ss, p);
        let md = pd.eval(m, &dr.ys, p, qz)?;
        let derivs = perturbation_derivs(m, dr, &g1, &md, &self.sigma_e, &set)?;
        let ps = pruned_state_space(dr, &derivs, &self.sigma_e, rows, 0, false, 0, qz, SCHUR_VEC_TOL)?;
        let ny = rows.len();
        let e = &ps.e_y;
        let mut loss = 0.0;
        for i in 0..ny {
            for j in 0..ny {
                loss += w[(i, j)] * (ps.var_y[(i, j)] + e[i] * e[j]);
            }
        }
        let grad = (0..self.i_params.len())
            .map(|k| {
                let mut g = 0.0;
                for i in 0..ny {
                    for j in 0..ny {
                        let d = ps.dvar_y[k][(i, j)] + ps.de_y[(i, k)] * e[j] + e[i] * ps.de_y[(j, k)];
                        g += w[(i, j)] * d;
                    }
                }
                g
            })
            .collect();
        Ok((loss, grad))
    }
}

impl Objective for OsrObjective<'_> {
    fn dim(&self) -> usize {
        self.i_params.len()
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        let (l, i, _) = self.loss(x);
        (l, i)
    }
    fn eval_full(&self, x: &[f64], _contributions: bool) -> Evaluation {
        let (fval, info, info4) = self.loss(x);
        Evaluation { fval, info, info4, contributions: None }
    }
    /// `analytic_derivation`: `dynare_minimize_objective.m` hands the gradient to fmincon,
    /// fminunc (`'GradObj','on'`), csminwel and solvopt.
    fn has_gradient(&self) -> bool {
        self.analytic.is_some()
    }
    fn gradient(&self, x: &[f64]) -> Option<Vec<f64>> {
        self.loss_grad(x).3
    }
}

/// Builds `W(i_var, i_var)` from the `optim_weights` entries (declaration-order indices):
/// returns the sorted unique `i_var` and the weight block.
pub fn weight_block(model: &Model, entries: &[(usize, usize, f64)]) -> (Vec<usize>, Mat) {
    let n = model.n_endo();
    let mut w = Mat::zeros(n, n);
    let mut iv = Vec::new();
    for &(i, j, v) in entries {
        w[(i, j)] = v;
        iv.push(i);
        iv.push(j);
    }
    iv.sort_unstable();
    iv.dedup();
    let blk = Mat::from_fn(iv.len(), iv.len(), |a, b| w[(iv[a], iv[b])]);
    (iv, blk)
}

/// Runs the optimizer of `opt_algo` through `dynare_minimize_objective.m`
/// ([`crate::estimation::optim::minimize`], stream O), as `+osr/run.m` does: `bounds` =
/// `M_.osr.param_bounds` (per osr parameter; unbounded without `osr_params_bounds`), no prior
/// standard deviations, `optim` = `options_.optim_opt`, `huge_number` = `options_.huge_number`
/// (the finite bounds of opt_algo 2; `None` = 1e7).
#[allow(clippy::too_many_arguments)]
pub fn minimize(
    obj: &OsrObjective,
    x0: &[f64],
    opt_algo: i32,
    bounds: Option<&[(f64, f64)]>,
    names: &[String],
    optim: Option<&crate::modfile::OptValue>,
    verbose: bool,
    seed: u64,
    huge_number: Option<f64>,
) -> Result<OptimResult> {
    if [5, 6, 10, 11].contains(&opt_algo) {
        bail!("OSR: OSR does not support opt_algo={opt_algo}.");
    }
    if bounds.is_some() && ![1, 2, 5, 9].contains(&opt_algo) {
        bail!("OSR: OSR with bounds on parameters requires a constrained optimizer, i.e. opt_algo= 1,2 or 9.");
    }
    let n = x0.len();
    let b = match bounds {
        Some(b) => Bounds { lb: b.iter().map(|v| v.0).collect(), ub: b.iter().map(|v| v.1).collect() },
        None => Bounds::unbounded(n),
    };
    let prior_sd = vec![f64::NAN; n];
    let ctx = crate::estimation::optim::MinimizeContext {
        bounds: &b,
        names,
        prior_sd: &prior_sd,
        optim,
        verbose,
        seed,
        stream: u64::MAX - 200,
    };
    let _huge = crate::estimation::optim::minimize::HugeNumberGuard::set(huge_number);
    crate::estimation::optim::minimize(opt_algo, obj, x0, &ctx)
}
