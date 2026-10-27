# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

- Windows support: builds with the MSVC toolchain against the prebuilt OpenBLAS
  (`scripts/setup-windows.ps1`, `DYNARE_RS_LAPACK_DIR`), `src/model/aux.rs` renamed
  `aux_subst.rs` (`aux` is a reserved file name on Windows), `.gitattributes` keeps LF
  checkouts for the byte-for-byte reference comparisons, `--window` opens the progress
  viewer in a new console window, and CI builds, lints and tests on `windows-latest`.
- Tests pass on Linux and Windows: comparisons tuned to macOS's Accelerate and libm use a
  documented, wider bound elsewhere (macOS keeps its tolerances), and `mex_direct_cases`
  accepts either mirror-image root of `chain_f`, which starts on the symmetry line.
- Test threads get a 64 MiB stack (`.cargo/config.toml`): four tests overflowed libtest's
  2 MiB default on Linux and aborted their test binaries.
- `--low-memory`: MCMC chains stored run-length encoded (`Chain::runs`; read through
  `Chain::draw` / `Chain::logpost_at`), MH files read back one chain at a time, and the
  post-MCMC statistics run on four threads. Identical results.
- `--openblas-threads N`: OpenBLAS's thread count, set before the library loads (dynare-rs
  restarts itself with `OPENBLAS_NUM_THREADS`: `exec` on Unix; on Windows dynare-rs.exe
  delay-loads libopenblas.dll so the waiting process never loads it), which also drops the
  other threads' buffers.
  Changes results at the rounding level.
- MH files are read without an extra copy of their draws; the progress viewer shows a
  separate "MCMC diagnostics and posterior densities" phase.
- `conditional_forecast(parameter_set=calibration)` starts from `M_.endo_histval` when a
  `histval` is set (e.g. by `smoother2histval`), as Dynare's `conditional_forecasts.run`,
  instead of always from the steady state.

## [0.1.0] - 2026-10-22

Initial public release.

- Port of Dynare 7.2: 153 of the 173 commands, blocks and macro directives of the reference
  manual; the other 20 are not applicable (MATLAB-only or code-generation features).
- Preprocessor and macro language, steady state, perturbation at orders 1-3, perfect
  foresight, extended path, OccBin, optimal policy, Kalman filters and smoothers, ML and
  Bayesian estimation (mode finders, MCMC samplers, SMC), forecasts, shock decompositions,
  identification, global sensitivity analysis, method of moments, particle filters, BVAR,
  MS-SBVAR, VAR/PAC models, heterogeneous-agent models, LaTeX output and figures.
- Validated against Dynare 7.2 reference output committed under `reference/`.
- Builds on macOS (Accelerate) and Linux (OpenBLAS).
