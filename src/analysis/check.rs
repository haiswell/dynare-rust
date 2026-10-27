// SPDX-License-Identifier: GPL-3.0-or-later
//! The `check` command: port of `check.m` (eigenvalues of the first-order pencil and the
//! Blanchard–Kahn order and rank conditions, printed as Dynare prints them).

use anyhow::{Result, bail};
use nalgebra::Complex;

use super::table::fmt_g;
use crate::model::Model;
use crate::solver::{FirstOrderSolver, SolverOptions, describe};

/// What `check.m` returns: `eigenvalues_`, `result` (BK conditions verified) and `info`.
#[derive(Clone, Debug)]
pub struct CheckResult {
    /// `dr.eigval` (mjdgges order).
    pub eigval: Vec<Complex<f64>>,
    /// `dr.edim`.
    pub edim: Option<usize>,
    pub nsfwrd: usize,
    pub full_rank: bool,
    /// BK conditions verified.
    pub result: bool,
}

/// `check(M_, options_, oo_)` at parameters `params` and steady state `ys`.
pub fn check(model: &Model, params: &[f64], ys: &[f64], opts: SolverOptions) -> Result<CheckResult> {
    check_at(model, params, ys, &vec![0.0; model.n_exo()], opts)
}

/// [`check`] at the exogenous steady state `x` (`oo_.exo_steady_state`, set by `initval`
/// in a deterministic model).
pub fn check_at(model: &Model, params: &[f64], ys: &[f64], x: &[f64], opts: SolverOptions) -> Result<CheckResult> {
    let solver = FirstOrderSolver::new(&model.lead_lag_incidence, model.n_exo(), opts);
    let g1 = model.dynamic_jacobian(ys, x, params);
    let nsfwrd = solver.structure.nsfwrd;
    match solver.check_eigenvalues(&g1) {
        Ok(c) => Ok(CheckResult {
            result: nsfwrd == c.edim && c.full_rank,
            eigval: c.eigval,
            edim: Some(c.edim),
            nsfwrd,
            full_rank: c.full_rank,
        }),
        Err(e) => bail!("check: {} (info {})", describe(e.info), e.info),
    }
}

/// The printout of `check.m` (preceded by `skipline()`).
pub fn format_check(c: &CheckResult) -> String {
    let mut out = String::from("\nEIGENVALUES:\n");
    out += &format!("{:>16} {:>16} {:>16}\n", "Modulus", "Real", "Imaginary");
    let mut idx: Vec<usize> = (0..c.eigval.len()).collect();
    // [m_lambda, i] = sort(abs(eigenvalues_)): stable, NaN last
    idx.sort_by(|&a, &b| c.eigval[a].norm().total_cmp(&c.eigval[b].norm()));
    for &i in &idx {
        let z = c.eigval[i];
        out += &format!("{} {} {}\n", fmt_g(z.norm(), 16, 4), fmt_g(z.re, 16, 4), fmt_g(z.im, 16, 4));
    }
    if let Some(edim) = c.edim {
        out += &format!("\nThere are {edim} eigenvalue(s) larger than 1 in modulus ");
        out += &format!("for {} forward-looking variable(s).", c.nsfwrd);
        out.push('\n');
        if c.result {
            out += "The order and rank conditions are verified.\n";
        } else if c.nsfwrd != edim {
            out += "The order condition is NOT verified.\n";
        } else {
            out += "The rank condition is NOT verified.\n";
        }
    }
    out.push('\n');
    out
}
