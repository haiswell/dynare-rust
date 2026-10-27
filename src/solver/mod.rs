// SPDX-License-Identifier: GPL-3.0-or-later
//! Steady state, first-order solution and restricted state space
//! (DESIGN.md, section `solver`).
//!
//! Mirrors Dynare's `resol.m` -> `evaluate_steady_state.m` / `stochastic_solvers.m`
//! -> `dyn_first_order_solver.m` chain and the `dynare_resolve(...,'restrict')`
//! assembly used by `dsge_likelihood.m`. Indices are 0-based throughout.

mod block_schur;
pub mod block_trust_region;
mod dmperm;
pub mod first_order;
// stream K: higher-order perturbation
pub mod korder;
pub mod reduction;
pub mod state_space;
pub mod steady_state;
pub mod structure;
pub use reduction::FirstOrderMethod;

pub use dmperm::{Dmperm, dmperm};
pub use first_order::{CheckEigen, DecisionRule, FirstOrderSolver, first_order};
pub use state_space::{StateSpaceIndex, kalman_transition_matrix, kalman_transition_matrix_into, state_space_index};
pub use steady_state::{
    SolveOutcome, SteadyOptions, dynare_solve, dynare_solve_r, steady_state, steady_state_r, steady_state_r_aux,
};
pub use structure::DrStructure; // model_diagnostics, block decomposition
/// Options of the first-order solver (`options_.qz_criterium`, `options_.qz_zero_threshold`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverOptions {
    /// Eigenvalues with modulus `< qz_criterium` are stable (default `1 + 1e-6`).
    pub qz_criterium: f64,
    /// Threshold for detecting 0/0 generalized eigenvalues (default `1e-6`).
    pub qz_zero_threshold: f64,
    /// `dr=cycle_reduction|logarithmic_reduction` (default: QZ).
    pub method: FirstOrderMethod,
}

impl Default for SolverOptions {
    fn default() -> Self {
        SolverOptions { qz_criterium: 1.0 + 1e-6, qz_zero_threshold: 1e-6, method: FirstOrderMethod::Qz }
    }
}

/// A Dynare error code (`info(1)`) with its `info(2)` companion, which
/// `dsge_likelihood.m` turns into the endogenous penalty (`info(4)`).
///
/// Codes produced here: 2 (QZ failure, `info2` = dgges info), 3 (BK: no stable
/// equilibrium), 4 (BK: indeterminacy), 5 (BK rank failure, `info2 = -log(rcond)`),
/// 7 (0/0 eigenvalue), 8 (NaN in the Jacobian), 10 (Inf in the Jacobian),
/// 20 (steady state not found, `info2 = |resid|^2`), 22 (NaN steady state).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("Dynare info {info} ({}), info(2) = {info2}", describe(*.info))]
pub struct SolveError {
    pub info: i32,
    pub info2: f64,
}

impl SolveError {
    pub fn new(info: i32, info2: f64) -> Self {
        SolveError { info, info2 }
    }
}

/// Short text for a Dynare info code (subset of `get_error_message.m`).
pub fn describe(info: i32) -> &'static str {
    match info {
        1 => "the model doesn't determine the current variables uniquely",
        2 => "MJDGGES returned an error code",
        3 => "Blanchard & Kahn conditions are not satisfied: no stable equilibrium",
        4 => "Blanchard & Kahn conditions are not satisfied: indeterminacy",
        5 => "Blanchard & Kahn conditions are not satisfied: indeterminacy due to rank failure",
        7 => {
            "one of the eigenvalues is close to 0/0 (the absolute value of numerator and denominator is smaller than qz_zero_threshold)"
        }
        8 => "the Jacobian of the dynamic model contains NaN",
        10 => "the Jacobian of the dynamic model contains Inf",
        20 => "impossible to find the steady state",
        22 => "the steady state contains NaN",
        26 => {
            "The loglinearization of the model cannot be performed, because the steady state is not strictly positive."
        }
        401 => "Cycle reduction reached the iteration limit. Try increasing maxit.",
        402 => "Cycle reduction terminated with NaN/Inf.",
        411 => "Logarithmic reduction reached the iteration limit. Try increasing maxit.",
        412 => "Logarithmic reduction terminated with NaN/Inf.",
        _ => "solver error",
    }
}
