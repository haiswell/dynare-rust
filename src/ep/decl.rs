// SPDX-License-Identifier: GPL-3.0-or-later
//! Declarations of the .mod file read by the expectation-errors solver (`M_.learnt_shocks`,
//! `M_.learnt_endval`), filled by the parser (`ModFile::ep`).

/// `M_.learnt_shocks(i).type` / `M_.learnt_endval(i).type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LearntKind {
    /// `values` / `x = v`.
    Level,
    /// `add` / `x += v`.
    Add,
    /// `multiply` / `x *= v`.
    Multiply,
    /// `mshocks(learnt_in=...)`: value times the terminal exogenous steady state anticipated
    /// in the learning period.
    MultiplySteadyState,
    /// `mshocks(learnt_in=... relative_to_initval)` (`multiply_initial_steady_state`, which
    /// perfect_foresight_with_expectation_errors_setup.m rejects as an unknown type).
    MultiplyInitialSteadyState,
}

/// One entry of `M_.learnt_shocks` (`shocks(learnt_in=p); var e; periods ...; values ...;`).
#[derive(Clone, Debug, PartialEq)]
pub struct LearntShock {
    pub learnt_in: usize,
    pub exo: usize,
    /// 1-based simulation periods.
    pub periods: Vec<i64>,
    pub kind: LearntKind,
    pub value: f64,
    /// The value when it mentions endogenous variables, and the number of commands before the
    /// block (`ModFile::resolve_learnt` evaluates it there).
    pub expr: Option<crate::modfile::Ast>,
    pub pos: usize,
    // ---- shocks/shock_paths ----
    /// `learnt_in = DATE` (then `learnt_in` is 0).
    pub learnt_in_date: Option<crate::dates::Date>,
    /// Periods given as dates (then `periods` is empty).
    pub dates: Vec<crate::dates::Date>,
}

/// One entry of `M_.learnt_endval` (`endval(learnt_in=p); x = v; y += v; z *= v; end;`).
#[derive(Clone, Debug, PartialEq)]
pub struct LearntEndval {
    pub learnt_in: usize,
    pub exo: usize,
    pub kind: LearntKind,
    pub value: f64,
    pub expr: Option<crate::modfile::Ast>,
    pub pos: usize,
    /// `learnt_in = DATE` (then `learnt_in` is 0). ---- shocks/shock_paths ----
    pub learnt_in_date: Option<crate::dates::Date>,
}

/// What the parser collects for stream EP.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EpDecl {
    pub learnt_shocks: Vec<LearntShock>,
    pub learnt_endval: Vec<LearntEndval>,
    /// `M_.perfect_foresight_controlled_paths`.
    pub controlled_paths: Vec<super::controlled::ControlledPath>,
}
