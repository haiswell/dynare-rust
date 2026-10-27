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
- Parity with Dynare 7.2, from an audit of every module against the 7.2 sources and the 7.2
  preprocessor (about 160 divergences fixed, `tests/parity_*.rs`):
  - The driver runs the .mod file as Dynare's `driver.m` does. `shocks`, deterministic shocks,
    `shock_paths`, controlled paths, `initval`, `histval` and `endval` take effect where they
    stand. Each `histval` block replaces the previous one, and `endval` saves the initial
    steady state only once.
  - Options persist between commands as in Dynare's global `options_`: estimation options,
    `forecast(periods, conf_sig)`, `loglinear` (taken per command, also from `stoch_simul`),
    `solve_algo`, homotopy and the heterogeneity options. `mode_compute` defaults to 5.
    Estimation selects `qz_criterium` as `select_qz_criterium_value.m` does.
  - The estimated point becomes `M_.params`/`M_.Sigma_e` for every later command, and a later
    parameter assignment overrides it. Parameters set in `steady_state_model` reach the
    likelihood, identification and the method of moments, and `osr` writes its parameters back.
    Steady states and Jacobians are evaluated at `oo_.exo_steady_state`. `forecast` uses the
    stored `oo_.dr`.
  - One port of `DsgeSmoother.m` for every smoother: observation trends, `lik_init` 2/4/5,
    `lyapunov`, `filter_initial_state`, and `prefilter`/`loglinear`/`first_obs` for
    `calib_smoother`. Forecasts and conditional forecasts add the trends. Posterior forecasts
    use `log(ss)` under `loglinear`, and add the prefilter mean and the trend. New posterior
    objects: `Init_State`, `StateUncertainty`, `InitStateUncertainty`, `FilterCovariance`, and
    the variance decompositions with measurement errors.
  - `smoother2histval` after an MCMC uses the posterior mean of the smoothed draws.
  - Macro processor and parser:
    - a `-D` define can be overridden by `@#define`;
    - `@#for` keeps its variable and handles `when` as Dynare does;
    - fractional ranges, `mod`, `[1:3]` and bool/real typing follow the preprocessor, and
      `savemacro` drops empty lines;
    - an invalid `shocks` entry is a syntax error, and `==` binds looser than `<`;
    - `include_eqs` handles equations that have no tag.
  - Derivatives: `STEADY_STATE()` terms have parameter derivatives, so identification accepts
    such models and analytic derivatives include them. A parameter exponent differentiates
    through `getPowerDeriv`.
  - Estimation and samplers: per-parameter `jscale`; the H check and the info 71/72 checks;
    TaRB inherits `optim`; NumgradAlgorithm 13/15; analytic gradients in the mode finders.
  - OccBin: `filter_initial_state`, observation trends, `lyapunov`, `endogenous_prior` and
    `lik_init` 2/4/5. The conditional smoother starts from all `nspred` states.
  - Identification and method of moments:
    - identification strength uses the analytic `dsge_likelihood` Hessian;
    - simulations start from histval and drop `options_.drop`;
    - SMM adds the calibrated measurement errors, and accepts estimated ones.
  - Policy:
    - LQ welfare at order 2;
    - conditional welfare from histval and the period-1 deterministic shocks;
    - `osr(maxit, tolf)`;
    - `ramsey_constraints` are ignored in the Ramsey steady state, as in 7.2.
  - Nonlinear and GSA:
    - the online filter runs at orders 1 and 3 and starts from the steady state;
    - particle-filter `qz_criterium`;
    - GSA centres `neighborhood_width` on the mode file and samples from the mode with
      `pprior=0`.
  - Other modules:
    - `simul_backward_model` starts from histval;
    - BVAR, sbvar and MS-SBVAR read the data like `read_variables`, and `ms_estimation`
      defaults to `nlags=1`;
    - controlled paths follow `stack_solve_algo` 1/6/7;
    - histval covers the lags of `diff()` auxiliaries;
    - the pruned state space rejects roots above `2-qz_criterium`.
  - Remaining known differences are listed in docs/REFERENCES.md.

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
