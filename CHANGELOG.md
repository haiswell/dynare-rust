# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-10-29

- Three MATLAB idioms of Dynare's own tests are interpreted: `if isoctave` / `if ~isoctave`
  (conditions that `isoctave`, false, decides with `~`, `&&`, `||` select the branch; the other
  branches are skipped instead of all running), `options_.initial_date = dates('...')` (read by
  `plot_shock_decomposition` and `initial_condition_decomposition`: aoa years, `plot_init_date`
  windows and vintage labels), and the heterogeneity tests' calibration from a MAT file
  (`load 'X.mat'`, `fieldnames`, a `for` loop calling `set_param_value(param, s.params.(param))`).
  `options_.X = V` without `;` ends with its line.

- csminwel (`mode_compute = 4`) perturbs its search direction on a cliff with MATLAB's
  global stream (mt19937ar, seed 0 at the start of a run or `set_dynare_seed`'s, carried over
  between calls) instead of ChaCha8, so modes that pass through cliffs (e.g. under
  `irf_calibration` / `moment_calibration` restrictions) are Dynare's.
- One-element `nobs = [n]` / `first_obs = [n]` are scalars (a rolling-window estimation with
  `nobs = [76]` no longer stops on "recursive estimation is not supported").
- `ramsey_model` with `EXPECTATION` / `diff` in the model: as Dynare's preprocessor, the
  operators are substituted before the first-order conditions (their aux variables get a FOC,
  their aux equations a multiplier; `M_.ramsey_orig_endo_nbr` / `ramsey_orig_eq_nbr` count
  them), and `dyn_ramsey_static` tells `M_.orig_endo_nbr` from `M_.ramsey_orig_endo_nbr`.
- Recursive and rolling-window estimation (`nobs = [a b ...]`, `nobs = [a:b]`,
  `first_obs = [...]`, `recursive_estimation_restart`): one estimation per sample in
  `<dirname>_<k>`, chained through the first sample's mode file, and with `forecast`
  `oo_.RecursiveForecast` (`<fname>_recursive_forecast.json`) and its figures.
- The slice sampler's `maximize` and `maximize_using_mh_bounds` (`posterior_sampler_options`):
  each iteration maximizes the posterior from the last draw (`mode_compute`, newrat by
  default), within the range of the previous MCMC's draws.
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
- Every option of the 7.2 manual is now handled (docs/COVERAGE.md lists none missing):
  - `osr`: `analytic_derivation` (the analytic loss gradient of `+osr/objective.m`, modes 0
    and 1; -1/-2 fail with Dynare's own error) and `huge_number`, checked against Dynare
    (`tests/policy_osr_ad.rs`).
  - `sensitivity`, `shock_decomposition`, `calib_smoother`: `prefilter`, `loglinear` and
    `use_univariate_*_if_singularity_is_detected` reach the smoother (also after an
    estimation); `nodisplay` and `conf_sig` (unused by Dynare's `+gsa`) are accepted.
  - `initval`/`endval`/`histval(all_values_required)` with the preprocessor's errors;
    `heterogeneity_simulate(tex)` writes the eps-loader TeX files (`print` accepted);
    `initial_condition_decomposition` honours `steadystate` and `write_xls`.
  - `ms_compute_mdd` accepts the manual's `mdd_proposal_draws`/`mdd_use_mean_center`
    spellings next to the preprocessor's `proposal_draws`/`use_mean_center`.
  - `occbin_setup`: `particle_filtering`, `filter_init_periods_using_particles`,
    `filter_particle_*`, `smoother_max_check_ahead_periods`, `smoother_plot`,
    `smoother_first_period_occbin_update`, `smoother_max_number_of_iterations` and
    `posterior_importance_sampling*` are accepted with a warning and no effect, as in Dynare
    7.2, whose `occbin.setup` does not forward them; checked against MATLAB
    (`tests/occbin_particle_options.rs`, `tests/occbin_est.rs`).
- Audit of the 23 official Dynare 7.2 examples (fresh MATLAB runs, every printout and result
  compared):
  - Console output matches Dynare 7.2 line by line for `estimation` (result tables, section
    order, inefficiency factors before the convergence diagnostics, MCMC and optimizer
    messages, `Inf`/`NaN`), the heterogeneity steady-state tables and calibration log, the
    `osr` parameter table and `pac.estimate.nls` (Dynare's NLS table with t-statistics);
    `Total computing time` uses Dynare's `0h00m03s`. Regression tests against stored MATLAB
    logs (`tests/console_layout.rs`, `tests/examples72_het.rs`).
  - dynare-rs's own notes (files written, timings, MCMC progress) go to stderr on a terminal
    only, so stdout is Dynare's output; `DYNARE_RS_VERBOSE=1` keeps them in redirected logs.
  - The preprocessor's warnings are printed with Dynare's text and order: unused parameters,
    deprecated `simul`/`ramsey_policy`/`mcp`/`aim_solver`/`bytecode`, max/min/abs/sign and
    comparisons in stochastic models, repeated `osr_params`, and the closing
    `Note: N warning(s) encountered in the preprocessor` (`tests/pp_warnings.rs`).
  - Fixed: the IRF-matching comparison table after a slice-sampler run without a mode is
    evaluated at the posterior mean, as `+mom/run.m`; `estimation(order=1)` no longer warns
    that `order` is unsupported.
  - Results JSON: `mean` at order 1, `skewness`/`kurtosis` with simulated moments. A model
    calling a MATLAB helper (`fsolve`, `fzero`) gets an explicit error.
- The classical smoother runs after mode-only, ML and MCMC estimation by default
  (`frequentist_smoother`, true by default, as `dynare_estimation_1.m`); preprocessor warnings
  are printed before a parse or model-building error, as the preprocessor does.
- Console parity for DSGE-VAR, the OccBin smoother iteration messages, `mh_recover` (incomplete
  chains and Dynare's "mh_recover option not required!" error), the
  `check_prior_stderr_corr_skew` warning, and Dynare's console for the dsmh, hssmc and dime
  samplers (with the SMC results table) and the `mode_compute=2`/`10` optimizers; 15 new
  console-layout tests against MATLAB logs (`reference/console/`).
- Dynare 7.2's own test suite (708 entries of `tests/meson.build`) run under MATLAB and dynare-rs and
  compared (`scripts/sweep/`, results in `scripts/sweep/triage.md`). Fixes:
  - `histval` of a lagged exogenous variable reaches its AUX_EXO_LAG variable (forecast,
    stoch_simul starting point); `ramsey_constraints` are complementarity conditions of the
    Ramsey FOCs in the perfect foresight solver (solve_algo = 10); `filter_initial_state`
    expressions follow the estimated parameters; the slice sampler's default (no
    `slice_initialize_with_mode`) runs its MCMC with `smoother` and `mode_compute=0`, raises
    `prior_trunc = 0` to 1e-10 and reports the MCMC's best draw as the posterior mode;
    `cova_compute = 0` gives NaN standard deviations at the mode and skips the Hessian in the
    nonlinear estimation (`tests/sweep_numfix.rs`).
  - PAC equations with rule-of-thumb agents (`share*(...) + (1-share)*(...)`): the share enters
    the growth-neutrality correction, MCE and `pac.estimate.iterative_ols` (calibrated or
    estimated); `pac.print` writes Dynare's expectation files; h-parameters on `diff(x(-k))`
    carry Dynare's names (`tests/pac.rs`).
  - VAR and trend-component models whose variables are expressions (`diff(diff(x))`,
    `diff(log(x))`, `log(x)` in `pac_target_info`) use Dynare's auxiliary variables; the PAC
    expectation, non-stationary and MCE variables are auxiliary variables created by the
    preprocessor's transform pass, with Dynare's numbering (`tests/pac_aux_vars.rs`).
  - .mod syntax (`tests/parity_mod_syntax.rs`): native MATLAB statements end with their line as
    in Dynare's lexer; case-insensitive keywords (`NORMAL_PDF`); signed and `(expr)` values in
    `conditional_forecast_paths`; `set_dynare_seed`, `datatomfile` and `set_param_value` forms;
    `[endogenous='c']`; mod-file local variables in `initval`; `matched_moments` powers of 0;
    `heteroskedastic_shocks` with dates; macro functions (scoping, `inf`/`nan`, index lists,
    `\\` continuation, bool casts).
  - Runtime (`tests/sweep72_runtime.rs`): `steady` on a deterministic `ramsey_model` solves the
    Ramsey static problem; `check(nocheck)` and `estimation(diffuse_filter)` switch off the
    steady-state residual check for later commands; `stoch_simul(noprint)` returns quietly on a
    solver error; nested model-local variables in Ramsey models; the rotated slice sampler starts
    from a loaded MH covariance; GSA without priors samples within the `estimated_params` bounds.
  - Dynare's errors (`tests/preprocessor_checks72.rs`): an exogenous variable unused in the
    model (a warning under `nostrict`), an estimated parameter in a `shocks` variance,
    `mh_tune_jscale` with `mh_jscale`/`mh_tune_guess`, `filter_algorithm=gmf` incompatibilities,
    an odd `hybrid` in the extended path, invalid symbols or leads/lags in
    `occbin_constraints`, and too many lags in `filter_initial_state`.
  - Estimation and policy (`tests/sweep_estfix.rs`): `options_.solve_tolf = V;` (also
    `solve_tolx`, `steady.maxit`, `solve_algo`) reaches the steady-state solver of the later
    commands (fs2000.mod's likelihood moved by 3e-4); `stoch_simul(partial_information)` after
    `ramsey_model` runs the PCL solver at the Ramsey steady state, and `oo_.dr.ghx/ghu` of a
    partial-information run are the PCL state space (`dr1_PI.m`).
  - The unused-exogenous check skips `varexobs` variables and the `growth` of `pac_model`
    (`tests/preprocessor_checks72.rs`). A positive `time_shift` of `var_expectation_model` is
    Dynare's error. Further fixes in `tests/sweep72_fixes.rs`:
    - `steady(solve_algo=10)` also where dynare-rs recomputes the steady state (e.g. for `resid`),
      and `resid` zeroes the residuals of the equations whose variable is at a bound ("accounting
      for / ignoring complementarity conditions").
    - `model(bytecode)` with `steady(solve_algo=5)` and `stack_solve_algo=5` (Newton iterations,
      as the bytecode MEX).
    - `var(log, deflator=...)`.
    - `dseries('file')` in MATLAB-like statements.
    - v7.3 (HDF5) MAT-files (`src/matfile_h5.rs`).
    - `homotopy_setup(from_initval_to_endval)` with no line in the block.
    - On-the-fly declarations (`c|e`, `e|x`, `beta|p`), with symbols ranked by first
      appearance as in Dynare.
    - `reshape` in .m datafiles, and .m/.mat/.xls datafiles for `sensitivity` and the method
      of moments; `sensitivity(stab=0)` loads the saved sample (`tests/sweep_fix2.rs`).
    - An endogenous variable in a `conditional_forecast_paths` value stands for its
      `oo_.steady_state` entry, as the preprocessor compiles it; `shock_decomposition` keeps the
      datafile options of an earlier call.
    - The same in `shocks`, `mshocks`, `shocks(learnt_in=...)` and `endval(learnt_in=...)`
      (`var`, `stderr`, `corr`, covariances, `values`): an endogenous variable reads
      `oo_.steady_state` and an exogenous one `oo_.exo_steady_state` where the block stands
      (`tests/shocks_endo.rs`).
    - `rplot` after `stoch_simul(periods > 0)` plots the simulation, with
      `options_.rplottype` 0, 1 and 2.
    - `plot_shock_decomposition` of a variable that `squeeze_shock_decomposition` dropped reruns
      the decompositions, as Dynare does; `options_.shock_decomp` (`with_epilogue`,
      `init_state`, `save_realtime`, `fast_realtime`, `presample`) persists between commands
      (`tests/shock_decomposition_recompute.rs`).
    - `var_expectation_model(expression=diff(x))` and `diff(log(x))`, and
      `var_expectation.print`; `pac.estimate.iterative_ols` estimates the parameters of additive
      terms (inside and outside the optimizing part), as iterative_ols.m; `delete f.mat` and
      `var_expectation.initialize(...)` without `;` end with their line (`tests/varmodels.rs`,
      `tests/pac.rs`).
    - `dynare_solve` retries from random guesses when the residuals at the initial guess are not
      finite (`solve_randomize_initial_guess`, deterministic simulations); the bare
      `homotopy_marginal_linearization_fallback` is 0.01 as the preprocessor writes it;
      `calib_smoother` leaves `parameter_set=calibration` and its data options for later
      decompositions; `simul_backward_model` uses `solve_algo`; calibrated parameters,
      `max`/`min`/`floor`/`ceil`/`round` in MATLAB-like statements; `heterogeneity_simulate(drop)`;
      parse errors in report statements carry file and line (`tests/sweep_c2fix.rs`).
    - `pac.bgp.set`, with the PAC equation's additive, optimizing-additive and non-optimizing
      terms in the growth-neutrality correction (`tests/pac.rs`); `true`/`false` in `optim`.
    - `extended_path(order=0)` (the default) and `hybrid > 2` (the constant of the order-`hybrid`
      rule); a purely backward model starts each period from the previous one
      (`sim1_purely_backward`); with `model(block)`, evaluate blocks are computed equation by
      equation (a whole-block Newton step gave `1 + x` for `exp(x)`); an `initval` after `endval`
      is the preprocessor's error; `model_remove`/`model_replace` take a bare `tag = 'value'`
      (`tests/sweep_ep_det.rs`).
    - `var_model`, `trend_component_model`, `var_expectation_model` and `pac_model` declared after
      the model block; PAC equations with a `diff(log(x))` left-hand side; A0star and the
      companion matrix follow the order of `targets` (`tests/pac.rs`).

- Perturbation at any order (`k_order_perturbation`, order >= 4): symbolic derivatives of any
  order, the decision rule `oo_.dr.g_0..g_k`, simulation with and without pruning (Dynare's
  `simulate_pruning`), and `k_order_welfare` for `evaluate_planner_objective`; orders 2 and 3 are
  unchanged. A Ramsey model with more or fewer instruments than omitted equations warns instead of
  failing, as in Dynare (`tests/korder_general.rs`, checked against MATLAB at orders 5 and 6).

- `mode_compute=5` (newrat): `mr_hessian` uses MATLAB's linear algebra (`A'*A` through dsyrk,
  Cholesky for `\` on symmetric positive-diagonal matrices, LU inverse), every stage was checked
  call by call against MATLAB (`reference/estimation/ref_newrat_stages.json`), and the console
  prints newrat.m's last-iteration lines. `options_.gradient_epsilon = V;` and
  `options_.gradient_method = V;` reach csminwel and newrat. The remaining iterate drift (about
  1e-4 after ten iterations on fs2000) is LAPACK rounding amplified by newrat's finite
  differences: MATLAB moves further when the starting point changes by one ulp.

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
