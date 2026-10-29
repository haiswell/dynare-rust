// SPDX-License-Identifier: GPL-3.0-or-later
//! dynare-rust: a Rust port of the parts of Dynare (MATLAB) needed to estimate
//! linear(ised) DSGE and semi-structural models such as reference/sw07/sw07.mod.
//!
//! Pipeline (mirrors Dynare's): .mod file -> macro preprocessor + parser
//! (`modfile`) -> symbolic model with aux variables and derivatives (`model`)
//! -> steady state + first-order solution (`solver`) -> state space and
//! diffuse Kalman likelihood (`kalman`) -> priors, mode finding, MCMC and
//! posterior analysis (`estimation`). `driver` runs the commands of a .mod file.
//! See DESIGN.md for the module contracts.

pub mod analysis;
pub mod data;
pub mod data_m; // [stream X] .m datafiles
pub mod deterministic;
pub mod driver;
pub mod dsge;
pub mod estimation;
pub mod graphics;
pub mod kalman;
pub mod linalg;
pub mod matfile;
mod matfile_h5;
pub mod model;
pub mod modfile;
pub mod parallel;
pub mod progress;
pub mod solver;
// ---- stream F ----
pub mod forecast;
// ---- end stream F ----
pub mod occbin;
// ---- stream B: BVAR, LaTeX output
pub mod bvar;
pub mod latex;
pub mod policy; // [stream P] optimal policy
// global sensitivity analysis (`sensitivity` / `dynare_sensitivity`)
pub mod gsa;
pub mod misc; // [stream X] small commands, MATLAB-level statements
// stream I: identification
pub mod identification;
pub mod mom; // stream MoM
// ---- stream PF: nonlinear estimation with particle filters ----
pub mod nonlinear;
// stream EP: extended path / stochastic extended path
pub mod dataset;
pub mod dates;
pub mod ep;
pub mod q4;
pub mod reporting; // report() objects (matlab/reporting)
// ---- var models ----
pub mod heterogeneity; // ---- heterogeneity ----
pub mod msbvar;
pub mod varmodels; // MS-SBVAR part 1
