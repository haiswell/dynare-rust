// SPDX-License-Identifier: GPL-3.0-or-later
//! Method of moments (stream MoM): port of Dynare 7.2's `+mom` package (`mom.run`) for
//! `method_of_moments(mom_method = GMM | SMM | IRF_MATCHING, ...)`.
//!
//! * [`moments`] — `matched_moments_block.m`, `get_data_moments.m`,
//!   `optimal_weighting_matrix.m`, the order-1 part of `pruned_state_space_system.m`;
//! * [`problem`] — the estimation problem and `mom.objective_function` (model solution at
//!   order 1/2/3, GMM moments from the pruned state space, SMM moments from simulations,
//!   IRF matching), an [`crate::estimation::Objective`] for the optimizers;
//! * [`irf_matching`] — `matched_irfs_blocks.m`;
//! * [`run`] — `mom.run` itself: stages and iterated weighting matrices
//!   (`mode_compute_gmm_smm.m`), `mode_compute_irf_matching.m`, standard errors
//!   (`standard_errors.m`), J-test (`Jtest.m`), printed tables and the results JSON;
//! * [`lsq`] — a bounded Levenberg–Marquardt substitute for `mode_compute = 13` (`lsqnonlin`).
//!
//! The driver keeps one [`Session`]: the `matched_*` blocks are commands in file order (a later
//! `matched_irfs(overwrite)` replaces the earlier one), `method_of_moments` runs with the
//! blocks seen so far.

pub mod analytic;
pub mod irf_matching;
pub mod lsq;
pub mod moments;
pub mod problem;
pub mod run;

use std::path::Path;

use anyhow::Result;

use crate::linalg::Mat;
use crate::modfile::{ModFile, Options};

/// One row of `M_.matched_moments`: declaration indices, leads/lags, powers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MatchedMoment {
    pub vars: Vec<usize>,
    pub lags: Vec<i32>,
    pub powers: Vec<u32>,
}

/// One statement of a `matched_irfs` block (`M_.matched_irfs` row): the endogenous variable,
/// the shock, and the groups of periods with their values and weights (one entry per
/// comma-separated item; a single value/weight is used for all periods of the group).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MatchedIrf {
    pub endo: Option<usize>,
    pub exo: Option<usize>,
    pub periods: Vec<Vec<usize>>,
    pub values: Vec<Vec<f64>>,
    pub weights: Vec<Vec<f64>>,
}

/// One row of `matched_irfs_weights` (`M_.matched_irfs_weights`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IrfWeight {
    pub endo1: usize,
    pub exo1: usize,
    pub periods1: Vec<usize>,
    pub endo2: usize,
    pub exo2: usize,
    pub periods2: Vec<usize>,
    pub weights: Vec<f64>,
}

/// MoM statements of the .mod file, in file order.
#[derive(Clone, Debug, PartialEq)]
pub enum MomCommand {
    MatchedMoments(Vec<MatchedMoment>),
    MatchedIrfs { overwrite: bool, entries: Vec<MatchedIrf> },
    MatchedIrfsWeights { overwrite: bool, entries: Vec<IrfWeight> },
    MethodOfMoments(Options),
}

/// The blocks in force (`M_.matched_moments`, `M_.matched_irfs`, `M_.matched_irfs_weights`).
#[derive(Clone, Debug, Default)]
pub struct Blocks {
    pub matched_moments: Vec<MatchedMoment>,
    pub matched_irfs: Vec<MatchedIrf>,
    pub matched_irfs_weights: Vec<IrfWeight>,
}

/// Inputs that tests can inject instead of dynare-rust's own random draws.
#[derive(Clone, Debug, Default)]
pub struct MomInputs {
    /// SMM: the standard-normal draws `options_mom_.mom.shock_series`
    /// (`(long + burnin) x exo_nbr`), e.g. MATLAB's `randn` from a dump.
    pub shock_series: Option<Mat>,
    /// SMM: the measurement-error draws `options_mom_.mom.ME_shock_series` (`long x nobs`).
    pub me_shock_series: Option<Mat>,
    /// Print the console output (tables, optimizer progress).
    pub quiet: bool,
}

/// State kept by the driver across commands.
#[derive(Default)]
pub struct Session {
    pub blocks: Blocks,
    /// Results of the last `method_of_moments` command.
    pub last: Option<run::MomResults>,
}

impl Session {
    pub fn new() -> Self {
        Self::default()
    }

    /// Runs one MoM statement.
    pub fn command(
        &mut self,
        mf: &ModFile,
        c: &MomCommand,
        dir: &Path,
        fname: &str,
        seed_override: Option<u64>,
        stream_seed: u64,
    ) -> Result<()> {
        match c {
            MomCommand::MatchedMoments(rows) => self.blocks.matched_moments = rows.clone(),
            MomCommand::MatchedIrfs { overwrite, entries } => {
                if *overwrite {
                    self.blocks.matched_irfs = entries.clone();
                } else {
                    self.blocks.matched_irfs.extend(entries.iter().cloned());
                }
            }
            MomCommand::MatchedIrfsWeights { overwrite, entries } => {
                if *overwrite {
                    self.blocks.matched_irfs_weights = entries.clone();
                } else {
                    self.blocks.matched_irfs_weights.extend(entries.iter().cloned());
                }
            }
            MomCommand::MethodOfMoments(opts) => {
                let ctx = run::Context { dir: dir.to_path_buf(), fname: fname.to_string(), seed_override, stream_seed };
                let res = run::method_of_moments(mf, &self.blocks, opts, &ctx, &MomInputs::default())?;
                self.last = Some(res);
            }
        }
        Ok(())
    }
}

/// The blocks in force after all statements of `mf` (convenience for tests).
pub fn blocks_of(mf: &ModFile) -> Blocks {
    let mut s = Session::new();
    for c in &mf.commands {
        if let crate::modfile::Command::Mom(m) = c {
            if !matches!(m, MomCommand::MethodOfMoments(_)) {
                // block statements never fail
                let _ = s.command(mf, m, Path::new("."), "", None, 0);
            }
        }
    }
    s.blocks
}

/// The options of the last `method_of_moments` command of `mf`.
pub fn last_mom_options(mf: &ModFile) -> Option<Options> {
    mf.commands.iter().rev().find_map(|c| match c {
        crate::modfile::Command::Mom(MomCommand::MethodOfMoments(o)) => Some(o.clone()),
        _ => None,
    })
}
