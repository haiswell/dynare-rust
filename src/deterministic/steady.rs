// SPDX-License-Identifier: GPL-3.0-or-later
//! `evaluate_steady_state.m` with exogenous steady-state values and the closed-form
//! `steady_state_model` branch (`evaluate_steady_state_file.m`, `steadystate_flag = 2`).

use anyhow::{Result, anyhow};

use crate::linalg::Mat;
use crate::model::{CompiledExpr, Model, SymbolKind};
use crate::modfile::ModFile;
use crate::solver::{SolveError, SteadyOptions, steady_state};

/// Target of one `steady_state_model` assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SsTarget {
    Endo(usize),
    Param(usize),
}

/// The compiled `steady_state_model` block (Dynare's `<fname>.steadystate`): assignments in
/// order, each reading the current values (`ys_` starts from the initial guess).
#[derive(Clone, Debug)]
pub struct SteadyStateModel {
    pub assignments: Vec<(SsTarget, CompiledExpr)>,
}

impl SteadyStateModel {
    /// `None` when the file has no `steady_state_model` block.
    pub fn from_modfile(mf: &ModFile) -> Result<Option<SteadyStateModel>> {
        if mf.steady_state_model.is_empty() {
            return Ok(None);
        }
        let m = &mf.model;
        let assignments = mf
            .steady_state_model
            .iter()
            .map(|(lhs, rhs)| {
                let target = match m.symbol(lhs) {
                    Some(s) if s.kind == SymbolKind::Endo => SsTarget::Endo(s.index),
                    Some(s) if s.kind == SymbolKind::Param => SsTarget::Param(s.index),
                    _ => {
                        return Err(anyhow!(
                            "steady_state_model: '{lhs}' is not an endogenous variable or a parameter"
                        ));
                    }
                };
                let e = m.compile_expr(rhs).map_err(|e| anyhow!("steady_state_model: {lhs} = {rhs}: {e}"))?;
                Ok((target, e))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(SteadyStateModel { assignments }))
    }

    /// Runs the assignments on `ys` (in place) and `params`; aux variables are then set from
    /// the static model, as the generated file does.
    pub fn eval(&self, model: &Model, ys: &mut [f64], x: &[f64], params: &mut [f64]) {
        for (t, e) in &self.assignments {
            let v = e.eval(ys, x, params);
            match *t {
                SsTarget::Endo(j) => ys[j] = v,
                SsTarget::Param(q) => params[q] = v,
            }
        }
        model.set_auxiliary_variables(ys, x, params);
    }

    pub fn changes_params(&self) -> bool {
        self.assignments.iter().any(|(t, _)| matches!(t, SsTarget::Param(_)))
    }
}

/// Port of `evaluate_steady_state.m` (non-block, non-bytecode) at exogenous steady state `x`:
/// the `steady_state_model` if any (residual check with `solve_tolf` unless `nocheck`,
/// info 19), else `dynare_solve` / the linear shortcut from `ys_init` (aux variables set first).
/// Returns the steady state and the (possibly updated) parameters.
pub fn evaluate_steady_state(
    model: &Model,
    ssm: Option<&SteadyStateModel>,
    ys_init: &[f64],
    x: &[f64],
    params: &[f64],
    opts: &SteadyOptions,
    check: bool,
) -> Result<(Vec<f64>, Vec<f64>), SolveError> {
    if model.planner().is_some_and(|pl| pl.ramsey.is_some()) {
        // options_.ramsey_policy: dyn_ramsey_static (the steady-state model, if any, gives the
        // variables conditional on the instruments)
        let rs = crate::policy::ramsey::RamseyStatic { model, ssm, exo_ss: x };
        return rs.steady_state(params, ys_init, opts);
    }
    let mut p = params.to_vec();
    if let Some(ssm) = ssm {
        let mut ys = ys_init.to_vec();
        ssm.eval(model, &mut ys, x, &mut p);
        if check {
            let mut r = vec![0.0; model.n_equations()];
            model.static_residuals(&ys, x, &p, &mut r);
            let max = r.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let ss: f64 = r.iter().map(|v| v * v).sum();
            if max > opts.tolf {
                return Err(SolveError::new(19, ss));
            }
            if r.iter().any(|v| v.is_nan()) {
                return Err(SolveError::new(22, f64::NAN));
            }
        }
        return Ok((ys, p));
    }
    if let Some(bs) = model.block_solve.as_ref().and_then(|b| b.statik.as_ref()) {
        // model(block): block-by-block dynare_solve
        let ys = super::block_solve::steady_blocks(model, bs, ys_init, x, &p, opts)?;
        return Ok((ys, p));
    }
    let n = model.n_endo();
    let f = |y: &[f64]| {
        let mut r = vec![0.0; n];
        model.static_residuals(y, x, &p, &mut r);
        (r, model.static_jacobian(y, x, &p))
    };
    let mut g = ys_init.to_vec();
    model.set_auxiliary_variables(&mut g, x, &p);
    // ---- solve_algo = 10 (lmmcp on static_mcp_problem; 11, PATH, is not shipped
    // with Dynare and also runs lmmcp here)
    if model.bytecode && opts.solve_algo == 5 {
        let r = |y: &[f64]| f(y).0;
        let ys = crate::solver::steady_state_bytecode(&f, &r, &g, opts)?;
        return Ok((ys, p));
    }
    if matches!(opts.solve_algo, 10 | 11) && !opts.linear {
        return steady_mcp(model, &g, x, p, opts);
    }
    let ys = steady_state(&f, &g, opts)?;
    Ok((ys, p))
}

/// `evaluate_steady_state.m`'s `solve_algo = 10` branch: `dynare_solve` -> `lmmcp`
/// with the bounds of `static_complementarity_conditions` and the residuals in
/// `static_mcp_equations_reordering` order.
fn steady_mcp(
    model: &Model,
    g: &[f64],
    x: &[f64],
    p: Vec<f64>,
    _opts: &SteadyOptions,
) -> Result<(Vec<f64>, Vec<f64>), SolveError> {
    use super::lmmcp::{LmmcpOptions, lmmcp};
    use super::spmat::Csr;
    let cc = super::mcp::Complementarity::new(model, &p).map_err(|_| SolveError::new(20, f64::NAN))?;
    let n = model.n_endo();
    let mut f = |y: &[f64], jac: bool| {
        let mut r = vec![0.0; n];
        model.static_residuals(y, x, &p, &mut r);
        let j = jac.then(|| {
            let full = model.static_jacobian(y, x, &p);
            Csr::from_dense(&Mat::from_fn(n, n, |i, k| full[(cc.eq_index[i], k)]))
        });
        (cc.reorder(&r), j)
    };
    let r = lmmcp(&mut f, g, &cc.lb, &cc.ub, &LmmcpOptions::default());
    if r.exitflag != 1 {
        let mut fv = f(&r.x, false).0;
        super::lmmcp::ignore_bound_residuals(&r.x, &mut fv, &cc.lb, &cc.ub);
        let ss: f64 = fv.iter().map(|v| v * v).sum();
        return Err(SolveError::new(if ss.is_nan() { 22 } else { 20 }, ss));
    }
    if r.x.iter().any(|v| v.is_nan()) {
        return Err(SolveError::new(22, f64::NAN));
    }
    Ok((r.x, p))
}

/// Static residuals at `(y, x)`.
pub fn static_residuals(model: &Model, y: &[f64], x: &[f64], p: &[f64]) -> Vec<f64> {
    let mut r = vec![0.0; model.n_equations()];
    model.static_residuals(y, x, p, &mut r);
    r
}
