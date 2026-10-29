// SPDX-License-Identifier: GPL-3.0-or-later
// ---- heterogeneity ----
//! Heterogeneous-agent models (Dynare 7, `+heterogeneity/`), part 1: declarations, the
//! heterogeneous model block, the steady-state input checks and loading
//! (`heterogeneity_load_steady_state`), the steady-state computation
//! (`heterogeneity_compute_steady_state`) and `rouwenhorst`.
//!
//! Data structures a later stage (`heterogeneity_solve`, `heterogeneity_simulate`) builds on:
//!
//! * [`HetDecl`] (`ModFile::het`): the heterogeneous model block ([`HetModel`] =
//!   `M_.heterogeneity(1)`, with compiled evaluators of `dynamic_het1_resid`, `dynamic_het1_g1`,
//!   `dynamic_het1_set_auxiliary_variables` and `dynamic_het1_complementarity_conditions`) and
//!   `M_.heterogeneity_aggregates`. The aggregate model is the ordinary [`crate::model::Model`];
//!   its aggregation variables `SUM_x` ([`HetDecl::aggregate_aux`], `M_.aux_vars` of type 14)
//!   are set from the aggregates `yagg`, see [`HetDecl::agg_residuals`] and
//!   [`HetDecl::set_auxiliary_variables`].
//! * [`SteadyStateInput`]: `steady_state` as the user provides it (agg, pol, shocks, d,
//!   free_parameters); [`check_steady_state_input`] gives the normalised data with the
//!   [`Sizes`] and [`Indices`] of the MATLAB code.
//! * [`LoadedSteadyState`] (`oo_.heterogeneity`): `steady_state`, `sizes`, `indices` and
//!   `mat` ([`SteadyMat`]: grids, interpolation brackets, transition matrix `Mu`, the LU
//!   factors of the basis matrix, `x_bar`, `x_bar_dash`, `F`, `dF`, `G`, `dG`, ...).

pub mod check;
pub mod commands;
pub mod model;
pub mod rouwenhorst;
pub mod simulate;
pub mod solve;
pub mod steady;
pub mod tex;
pub mod time_iteration;

pub use check::{CheckOptions, Indices, Sizes, SteadyStateData, SteadyStateInput, check_steady_state_input};
pub use commands::{HetOptions, HetSession, run_compute_steady_state, run_load_steady_state};
pub use model::{HetModel, HetModelDecl, HetSymbolDecl, HetWorkspace};
pub use rouwenhorst::rouwenhorst;
pub use simulate::{SimOptions, SimResult, simulate};
pub use solve::{HetDr, solve_hetero};
pub use steady::{LoadedSteadyState, SteadyMat};
pub use time_iteration::{CalibrationOptions, ForwardOptions, TargetEquation, TimeIterationOptions};

use crate::linalg::Mat;
use crate::model::{AuxVar, Model};

/// One row of `M_.heterogeneity_aggregates` (`'sum', dimension, index in the heterogeneous
/// endogenous variables`), all indices 0-based.
#[derive(Clone, Debug, PartialEq)]
pub struct HetAggregate {
    pub op: String,
    pub dimension: usize,
    pub het_endo: usize,
}

/// Heterogeneity part of a parsed .mod file.
#[derive(Clone, Debug, Default)]
pub struct HetDecl {
    pub model: Option<HetModel>,
    /// `M_.heterogeneity_aggregates`; row `k` is the aggregate auxiliary variable `aggregate_aux[k]`.
    pub aggregates: Vec<HetAggregate>,
    /// The aggregation auxiliary variables (`M_.aux_vars(k)`, type 14, listed first in Dynare):
    /// `endo_index` is the 0-based position in `M_.endo_names`.
    pub aggregate_aux: Vec<AuxVar>,
}

impl HetDecl {
    /// Row of the equation `SUM_x = yagg` of aggregate `k` (the equations are: those written,
    /// then the aggregation equations in order, then the other auxiliary equations).
    pub fn aggregate_row(&self, model: &Model, k: usize) -> usize {
        model.orig_eq_nbr + k
    }

    /// `dynamic_resid(y, x, params, steady_state, yagg)` of the aggregate model: `y` is
    /// `[y_{t-1}; y_t; y_{t+1}]` (3 * `endo_nbr`). The equation of `SUM_x` is `SUM_x - yagg`.
    pub fn agg_residuals(&self, model: &Model, y: &[f64], x: &[f64], params: &[f64], yagg: &[f64], out: &mut [f64]) {
        model.dynamic_residuals(y, x, params, out);
        for k in 0..self.aggregate_aux.len() {
            out[self.aggregate_row(model, k)] -= yagg[k];
        }
    }

    /// `set_auxiliary_variables(y, x, yagg, params)` of the aggregate model: the aggregation
    /// variables take the aggregates, then the other auxiliary variables are set.
    pub fn set_auxiliary_variables(&self, model: &Model, y: &mut [f64], x: &[f64], params: &[f64], yagg: &[f64]) {
        for (k, a) in self.aggregate_aux.iter().enumerate() {
            y[a.endo_index] = yagg[k];
        }
        model.set_auxiliary_variables(y, x, params);
    }

    /// Dynamic Jacobian of the aggregate model with `y_{t-1} = y_t = y_{t+1} = y` (`mat.dG`);
    /// `yagg` is a constant, so it does not enter.
    pub fn agg_jacobian(&self, model: &Model, y: &[f64], x: &[f64], params: &[f64]) -> Mat {
        model.dynamic_jacobian(y, x, params)
    }
}
