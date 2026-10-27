# Coverage of the Dynare 7.2 reference manual

Every command, block, MATLAB command and macro directive of the Dynare 7.2 reference manual
(`doc/dynare-manual.html/_sources/the-model-file.rst.txt`, `dynare-misc-commands.rst.txt`,
`running-dynare.rst.txt`) with its status in dynare-rust.

Status: **done** (ported, checked against Dynare where numbers are involved), **partial**
(see the note), **missing** (not ported), **n/a** (MATLAB-only or code-generation features
without a counterpart; accepted with a note where harmless). Unknown commands are skipped with
a warning.

Option status is derived from the code: an option is *done* when the modules that implement
the command read it (string literal search), corrected by hand where needed; *missing* options
of a *done* command are accepted and ignored (usually with a warning). Regenerate with
`python3 docs/gen_coverage.py` (the hand annotations are in that script).

## Summary

| status | commands / blocks / directives |
|---|---|
| done | 153 |
| partial | 0 |
| missing | 0 |
| n/a | 20 |

## Missing


Missing options of ported commands (highlights):
`estimation` `draw_init_state_with_rotated_slice` of `estimate_initial_states_endogenous_prior`;
(none besides the entries marked n/a.)

## Commands and blocks

| command | kind | status | notes |
|---|---|---|---|
| `var` | command | done | `$tex$`, `long_name`; `log` (LOG_ aux variables, `substituteLogTransform`), `deflator`/`log_deflator` (detrending, exact aux names; checked against the Dynare 7.2 preprocessor and MATLAB) |
| `varexo_det` | command | done | out of `M_.exo_names`/`Sigma_e`, parameters of the model at `oo_.exo_det_steady_state` (steady state, `check`, IRFs, `periods` simulations, estimation as Dynare); `dr.ghud` and order-2 `ghxud`/`ghuud`/`ghudud` (`stochastic_solvers.m`), `simult_varexo_det.m` forecasts at orders 1-2 (also `forecast=H` after an estimation, with the zeros `make_ex_.m` leaves); `histval`/`initval`/`endval`, JSON, LaTeX; every Dynare error (leads/lags, perfect-foresight solvers, `endval`, `shocks` moments, variable lists, `irf_shocks`, epilogue, purely forward/backward models, order 3 / k_order_solver, pruning, date-valued shocks, `extended_path`); `reference/exo_det/`, `tests/exo_det.rs` |
| `var_remove` | command | done | removes a declared symbol (`DynamicModel` port in src/model/mod.rs, Dynare's errors); checked against Dynare in reference/pp/pp_edit.mod (tests/preprocessor_pp.rs) |
| `predetermined_variables` | command | done |  |
| `trend_var` | command | done | `growth_factor`/`log_growth_factor`, detrending (`detrendEquations`, `removeTrendLeadLag`) and the balanced-growth test (`balanced_growth_test_tol`), checked against the Dynare 7.2 preprocessor and MATLAB (schorfheide_2000_nonstationary) |
| `model_local_variable` | command | done | `$tex$` names of `#` locals in the LaTeX output (`write_latex_*`, as the preprocessor); `long_name` is not accepted by the 7.2 grammar either |
| `external_function` | command | done | the model calls a function through a registry (`model::external`): a Rust closure registered with `register_external_function[_with_derivatives]` under the function name, else `NAME.m` next to the .mod, run by a small interpreter for simple MATLAB function files (scalar arithmetic, elementary functions, `if`/`elseif`/`else`, matrix literals and indexed assignments for the derivative outputs, `nargin`/`nargout`; loops, other function calls, strings, cell arrays and local functions are refused with the file and line). New DAG node `Node::External` (`ExternalFunctionNode` in the `dtree` port, `First/SecondDerivExternalFunctionNode` for the derivatives) handled by the tapes, symbolic derivatives, static/dynamic conversion, aux-variable substitutions, LaTeX/JSON printers. `first_deriv_provided`/`second_deriv_provided` without a name use the 2nd/3rd output of the function, with a name a separate function; otherwise Dynare's `jacob_element.m`/`hess_element.m` central differences (h = 1e-6). Third-order derivatives are refused (as the preprocessor). Checked against Dynare 7.2 (steady state, order-2 decision rules, all three derivative settings) in reference/external_function/ (tests/external_function.rs) |
| `get_param_by_name` | matcomm | n/a | MATLAB accessor |
| `set_param_value` | matcomm | done | all Dynare setters (`set_param_value`, `set_shock_stderr_value`, `set_shock_skew_value`) are applied at parse time or at their place after a command (tests/ss_partials.rs); the getters are MATLAB accessors and the `*_locally` variants are internal functions |
| `model` | block | done | `linear` (and `model(linear)`); equation tags, `#` locals, `STEADY_STATE()`; `use_dll`/`bytecode` are code-generation options with no counterpart (accepted); `block`/`mfs`/`static_mfs`/`cutoff`/`no_static` set the block decomposition (`M_.block_structure`), used by `steady` and `perfect_foresight_solver` for block-wise solving (`solve_block_decomposed_problem`, evaluate/solve/two-boundary blocks; checked against Dynare 7.2, reference/block_solve; `stack_solve_algo=4` runs the `lnsrch1` path-length search on the stacked two-boundary blocks, checked against Dynare 7.2 (reference/block_solve/ref_block_solve_a4.json); bytecode is n/a: no bytecode path) |
| `model_options` | command | done | applies `linear`, `block`, `mfs`, `static_mfs`, `cutoff`, `no_static`, `differentiate_forward_vars`, `balanced_growth_test_tol` exactly as the options of the `model` block (parser `apply_model_options`; equivalence tested in tests/model_steady_partials.rs); `use_dll`/`bytecode`/`parallel_local_files` accepted (no counterpart) |
| `model_remove` | command | done | tag selection, `endogenous` tags, port of `DynamicModel::removeEquations` with Dynare's errors; checked against Dynare in reference/pp/pp_edit.mod (tests/preprocessor_pp.rs) |
| `model_replace` | block | done | tag selection, `endogenous` tags, port of `DynamicModel::removeEquations` with Dynare's errors; checked against Dynare in reference/pp/pp_edit.mod (tests/preprocessor_pp.rs) |
| `write_latex_original_model` | command | done |  |
| `write_latex_dynamic_model` | command | done |  |
| `write_latex_static_model` | command | done |  |
| `write_latex_steady_state_model` | command | done |  |
| `initval` | block | done | `all_values_required` is not checked |
| `endval` | block | done | `all_values_required` is not checked |
| `histval` | block | done | `all_values_required` is not checked |
| `resid` | command | done | `display_static_residuals.m` output, incl. auto equation names; `non_zero` |
| `initval_file` | command | done | DATE values of `first_obs`/`last_obs`/`first_simulation_period`/`last_simulation_period` (positions in the datafile's dates, as `histvalf_initvalf.m`), checked against Dynare; `histval_file` takes no `nobs`/`last_simulation_period` and reads `M_.orig_maximum_lag` observations (as Dynare 7.2) |
| `histval_file` | command | done | DATE values of `first_obs`/`last_obs`/`first_simulation_period`/`last_simulation_period` (positions in the datafile's dates, as `histvalf_initvalf.m`), checked against Dynare; `histval_file` takes no `nobs`/`last_simulation_period` and reads `M_.orig_maximum_lag` observations (as Dynare 7.2) |
| `shocks` | block | done | stderr/var/cov/corr as expressions (correlations evaluated at the end of their block, as the generated code), measurement errors, `overwrite` (resets `Sigma_e`/`H`/det shocks, not `Skew_e`); deterministic `periods` with integers, ranges, lists and dates (`first_simulation_period`/`last_simulation_period`), `values` with expressions and MATLAB workspace vectors `(xx)`; `learnt_in = INTEGER / DATE` with `values`/`add`/`multiply`; `surprise` (OccBin); `skew`: pruned skewed Kalman filter and, in `stoch_simul`, the `check_model.m` bound, the COSKEWNESS table and closed-skew-normal draws for `periods > 0` (`rand_multivariate_csn.m`; Dynare uses `Skew_e` nowhere else: theoretical moments, IRFs/GIRFs and decision rules ignore it); all checked against Dynare 7.2 (reference/shocks). `shocks(heterogeneity=NAME)`: `M_.heterogeneity(1).Sigma_e` of the heterogeneous shocks (see `heterogeneity_dimension`) |
| `mshocks` | block | done | `relative_to_initval`, `overwrite`, `learnt_in = INTEGER / DATE`, dates in `periods`; options separated by spaces as in the 7.2 grammar (commas also accepted) |
| `heteroskedastic_shocks` | block | done | `values`/`scales`, `overwrite`; with `heteroskedastic_filter` in the likelihood (all filters), the smoother and `calib_smoother`; integer periods only (no dates). Checked against Dynare (fval 1e-15, smoother 1e-15) |
| `get_shock_stderr_by_name` | matcomm | n/a | MATLAB accessor |
| `set_shock_stderr_value` | matcomm | done | `M_.Sigma_e(i,i) = v^2` (before the first command at parse time, else at its place) |
| `get_shock_skew_by_name` | matcomm | n/a | MATLAB accessor |
| `set_shock_skew_value` | matcomm | done | `set_shock_skew_value('e', v)` (and the co-skewness form) applied at parse time |
| `dsample` | command | done | records `options_.smpl`; `rplot` plots that sample, as `rplot.m` (`dyn2vec`/`dynatype`/`dynasave` are MATLAB helpers) |
| `steady` | command | done | `homotopy_mode` 1/2/3 + `homotopy_steps` + `homotopy_force_continue` and `solve_algo` 0/3/6/7/8/13/14 (substitute for 0 = MATLAB fsolve; 13/14 = exact port of the block_trust_region MEX (Dulmage-Mendelsohn blocks / whole system, dogleg trust region, exit codes), checked against MATLAB; 7/8 = Newton with GMRES/BiCGStab preconditioned by an ILU with droptol 1e-10, i.e. the direct solve); `solve_algo = 10` (port of lmmcp.m with the complementarity conditions), `fsolve_options` (TolFun/FunctionTolerance, TolX/StepTolerance, MaxIter/MaxIterations override the substitute solver of solve_algo=0; Display and MaxFunEvals accepted, no counterpart); `non_zero` limits the failure printout (`display_static_residuals` text, shared with `resid`) to non-zero residuals; printout `disp_steady_state` |
| `get_mean` | matcomm | n/a | MATLAB accessor (order-2 mean: `stoch_simul` writes it) |
| `homotopy_setup` | block | done | with `steady(homotopy_mode=...)`; `perfect_foresight_solver` has its own homotopy |
| `steady_state_model` | block | done | temporary variables of the block; multi-output `[a, b] = f()` assignments and an external `<fname>_steadystate.m` call MATLAB code and are refused with an explicit error naming the alternative (inline `steady_state_model` assignments); deviation by design |
| `check` | command | done |  |
| `model_diagnostics` | command | done | port of `model_diagnostics.m` (static/dynamic/contemporaneous Jacobian ranks, collinear variables and equations via `null(A, "rational")`, unit-root hint, Inf/NaN, Hessian), output checked against Dynare; no Ramsey-instrument checks |
| `model_info` | command | done | the non-block output (states, forward-looking, static variables); model_info: `block_static`/`block_dynamic`/`incidence` print the preprocessor's block decomposition (`M_.block_structure(_stat)`: Boost matching with libstdc++ tie order, prologue/epilogue, Tarjan SCCs, feedback sets, block types; identical to the Dynare 7.2 preprocessor on 416 models, text identical to MATLAB incl. SW07/QPM/GM05) |
| `print_bytecode_dynamic_model` | command | n/a | bytecode is a MATLAB/C++ code-generation backend; accepted with a note |
| `print_bytecode_static_model` | command | n/a | bytecode is a MATLAB/C++ code-generation backend; accepted with a note |
| `perfect_foresight_setup` | command | done | `first_simulation_period`/`last_simulation_period` (dates, get_simulation_periods.m: with `periods` or both dates), mapping the dates of `shocks`/`mshocks`/`shock_paths`/controlled paths to periods |
| `perfect_foresight_solver` | command | done | `stack_solve_algo` 1/6 (sim1_lbj), 2/3 (GMRES/BiCGStab with first_iter_lu / block_diagonal_lu / incomplete_lu preconditioners; ILU = complete LU), 7 (stacked dynare_solve, `lmmcp`), 8/9 (ParU/Pardiso: sparse LU), 10 (LU then CGS), `robust_lin_solve`, `check_jacobian_singularity`, `homotopy_marginal_linearization_fallback`, purely backward/forward/static models, all checked against MATLAB; `minimal_solving_periods` is unused in Dynare 7.2 |
| `simul` | command | done | legacy alias of setup + solver |
| `shocks` | block | done | stderr/var/cov/corr as expressions (correlations evaluated at the end of their block, as the generated code), measurement errors, `overwrite` (resets `Sigma_e`/`H`/det shocks, not `Skew_e`); deterministic `periods` with integers, ranges, lists and dates (`first_simulation_period`/`last_simulation_period`), `values` with expressions and MATLAB workspace vectors `(xx)`; `learnt_in = INTEGER / DATE` with `values`/`add`/`multiply`; `surprise` (OccBin); `skew`: pruned skewed Kalman filter and, in `stoch_simul`, the `check_model.m` bound, the COSKEWNESS table and closed-skew-normal draws for `periods > 0` (`rand_multivariate_csn.m`; Dynare uses `Skew_e` nowhere else: theoretical moments, IRFs/GIRFs and decision rules ignore it); all checked against Dynare 7.2 (reference/shocks). `shocks(heterogeneity=NAME)`: `M_.heterogeneity(1).Sigma_e` of the heterogeneous shocks (see `heterogeneity_dimension`) |
| `endval` | block | done | `all_values_required` is not checked |
| `mshocks` | block | done | `relative_to_initval`, `overwrite`, `learnt_in = INTEGER / DATE`, dates in `periods`; options separated by spaces as in the 7.2 grammar (commas also accepted) |
| `perfect_foresight_with_expectation_errors_setup` | command | done | (`learnt_in` shocks/endval); integer or date `learnt_in`, `first_simulation_period`/`last_simulation_period`, `shock_paths(learnt_in=...)` information sets |
| `perfect_foresight_with_expectation_errors_solver` | command | done | (`learnt_in` shocks/endval); integer or date `learnt_in`, `first_simulation_period`/`last_simulation_period`, `shock_paths(learnt_in=...)` information sets |
| `perfect_foresight_controlled_paths` | block | done | `learnt_in` and periods as dates |
| `shock_paths` | block | done | temporary/permanent (`end`)/controlled (`exogenize`/`endogenize`) stanzas with `initval.`/`init.`/`self.` scopes; dates in `periods` (also `DATE:end`), `DATABASE.x(lag)` (dseries, multi-row and single-row tables: extract_from_database.m), `prev.`/`learnt_in(INTEGER / DATE).` scopes, `learnt_in = INTEGER / DATE` blocks evaluated per informational period by perfect_foresight_with_expectation_errors_setup, `overwrite` (options separated by spaces as in the 7.2 grammar, commas also accepted); all checked against Dynare 7.2 (reference/shocks) |
| `database` | command | done | `M_.database` for `shock_paths`; the databases are the .mod file's MATLAB assignments that dynare-rust evaluates: `dseries(DATA, DATE, NAMES)`, `dseries('file')` (csv/xls(x)/mat/m), `table(..., 'VariableNames', ...)`, `array2table(...)` over numeric expressions (`[...]`, `linspace`, `transpose`, `ones`, `zeros`, `repmat`, colon ranges, elementwise arithmetic, indexing) |
| `stoch_simul` | command | done | order 1 (S) and 2-3 (K); `loglinear` (order 1), filters, `spectral_density`, `dr=cycle_reduction/logarithmic_reduction`, `qz_zero_threshold`, `contemporaneous_correlation`, `tex` tables, filters/variance decompositions/`relative_irf`/`irf_plot_threshold` at order 2-3; `partial_information` (`dr1_PI`/`PI_gensys`/`PCL_Part_info_irf`/`PCL_Part_info_moments`, order forced to 1, `.mat` files), `aim_solver`/`dr=aim` (AIM_first_order_solver + matlab/AIM), the 7.2 refusals at order > 1 (`loglinear`: two messages with/without `k_order_solver`; `aim_solver`; `dr=cycle_reduction` only with `k_order_solver`), `order` reset to 2 by every command without it (as the preprocessor), Hessian-equal-to-zero fallback, `print`/`graph` re-enable, `graph_format=none`, `simul_replic` (`Output/<fname>_simul`; skew-normal draws in every replication with `skew`), the `covar_ME` TeX table and the `variance_decomposition_ME` JSON fields at orders 1-2, `solve_algo` (10/11 refused), `histval` start of simulations, measurement errors (`MATRIX OF COVARIANCE OF MEASUREMENT ERRORS`, decompositions WITH MEASUREMENT ERROR), IRF eps-loader TeX file; printouts checked character for character (tests/stoch_simul2.rs). Simulations draw from ChaCha8, not MATLAB's randn; graphs are never displayed (`nodisplay` is the only behaviour) |
| `IRF_MATRIX=get_irf` | matcomm | n/a | MATLAB accessor; IRFs are in `Output/<fname>_IRF_<shock>.csv` |
| `extended_path` | command | done | (incl. stochastic extended path); `lmmcp` at order 0 and on the stochastic tree (`ep.solve_algo = 10`, `solve_algo=10` here; `reference/ep_lmmcp/`, `tests/ep_lmmcp.rs`); not ported: `hybrid` > 2, `innovation_distribution`, `extended_path_mc` (not reachable from the preprocessor / no defined result) |
| `occbin_constraints` | block | done |  |
| `shocks` | block | done | stderr/var/cov/corr as expressions (correlations evaluated at the end of their block, as the generated code), measurement errors, `overwrite` (resets `Sigma_e`/`H`/det shocks, not `Skew_e`); deterministic `periods` with integers, ranges, lists and dates (`first_simulation_period`/`last_simulation_period`), `values` with expressions and MATLAB workspace vectors `(xx)`; `learnt_in = INTEGER / DATE` with `values`/`add`/`multiply`; `surprise` (OccBin); `skew`: pruned skewed Kalman filter and, in `stoch_simul`, the `check_model.m` bound, the COSKEWNESS table and closed-skew-normal draws for `periods > 0` (`rand_multivariate_csn.m`; Dynare uses `Skew_e` nowhere else: theoretical moments, IRFs/GIRFs and decision rules ignore it); all checked against Dynare 7.2 (reference/shocks). `shocks(heterogeneity=NAME)`: `M_.heterogeneity(1).Sigma_e` of the heterogeneous shocks (see `heterogeneity_dimension`) |
| `occbin_setup` | command | done | `simul_*` (OccBin) and `smoother_*`/`likelihood_*` (PKF/IVF likelihoods, OccBin smoother); posterior smoothed variables/shocks after an OccBin MCMC; posterior MeanForecast/PointForecast after it (`occbin.forecast` from each draw's smoothed last state, tests/occbin_posterior_forecast.rs) |
| `occbin_solver` | command | done |  |
| `occbin_graph` | command | done |  |
| `occbin_write_regimes` | command | done |  |
| `varobs` | command | done |  |
| `observation_trends` | block | done | likelihood and smoother (with `prefilter`, `first_obs`); checked against Dynare (fval 1e-12, smoother 1e-15) |
| `estimated_params` | block | done | all prior shapes, bounds, stderr/corr of shocks and measurement errors, `overwrite`; `skew` and `dsge_prior_weight`; generalized priors (PRIOR_3RD/4TH_PARAMETER) checked against Dynare |
| `estimated_params_init` | block | done | `use_calibration` |
| `estimated_params_bounds` | block | done | (checked against Dynare's bounds) |
| `estimated_params_remove` | block | done | (checked against Dynare's bounds) |
| `estimation` | command | done | see the option table (optimizers and samplers, particle filters, OccBin estimation, Kalman robustness, `lik_init` 2/4/5, `.m` datafiles, `observation_trends`, `analytic_derivation`, Hessian options, skewed Kalman smoother; checked against Dynare in reference/est_complete/) |
| `data` | command | done | `options_.dataset` for later data commands (makedataset.m: `first_obs`/`last_obs` dates, `nobs`, consistency errors); `series=` needs a MATLAB-workspace dseries (error with a note); `set_time` sets `options_.initial_period` |
| `filter_initial_state` | block | done | initial state of the likelihood filters and of the smoother (`x(0)`, `x(-k)` through the lag aux variables); checked against Dynare |
| `bvar_density` | command | done |  |
| `bvar_irf` | command | done |  |
| `varobs` | command | done |  |
| `matched_moments` | block | done |  |
| `matched_irfs` | block | done |  |
| `matched_irfs_weights` | block | done |  |
| `estimated_params` | block | done | all prior shapes, bounds, stderr/corr of shocks and measurement errors, `overwrite`; `skew` and `dsge_prior_weight`; generalized priors (PRIOR_3RD/4TH_PARAMETER) checked against Dynare |
| `estimated_params_init` | block | done | `use_calibration` |
| `estimated_params_bounds` | block | done | (checked against Dynare's bounds) |
| `method_of_moments` | command | done | (GMM, SMM, IRF matching), analytic_standard_errors / analytic_jacobian (order 1 analytic, order 2-3 Richardson differences), MCMC after Bayesian IRF matching (`mh_replic > 0`); `hp_filter`, `one_sided_hp_filter`, `bandpass_filter` (SMM only, as Dynare; get_filtered_time_series.m, filters shared with stoch_simul, no MATLAB reference run for the last two because SMM draws differ); `use_pct` (Parallel Computing Toolbox) is n/a |
| `model_comparison` | command | done |  |
| `shock_decomposition` | command | done | `with_epilogue` (epilogue_shock_decomposition.m, checked against Dynare) and `colormap` (MATLAB colormaps by name) |
| `shock_groups` | block | done |  |
| `realtime_shock_decomposition` | command | done | `colormap` through plot_shock_decomposition; `with_epilogue` inside the realtime recursion (vintages, pools, conditional / forecast / realtime conditional; tests/realtime_epilogue.rs) |
| `plot_shock_decomposition` | command | done | `init2shocks`, epilogue variables (`with_epilogue`), `steadystate` (zero line of the levels and the xls column; the y-tick relabelling is not reproduced), `write_xls` (WriteShockDecomp2Excel.m as a real .xlsx), `colormap`, `screen_shocks`, `fig_name` (`SCREEN`, `init2shocks`, `...initval` detail labels); `interactive` (MATLAB uimenus) accepted with a note |
| `init2shocks` | block | done | (`M_.init2shocks.NAME`, used by plot_shock_decomposition(init2shocks), checked against Dynare) |
| `initial_condition_decomposition` | command | done |  |
| `squeeze_shock_decomposition` | command | done |  |
| `calib_smoother` | command | done | `filter_step_ahead`, `filter_decomposition`, `filter_covariance`, `smoother_redux`, `heteroskedastic_filter` |
| `forecast` | command | done | (the `forecast` command starts from `M_.endo_histval` with `histval`, runs at `options_.order` 1-3 with simulated bands at order > 1, and takes its horizon from `options_.periods`; `conditional_forecast(parameter_set=calibration)` starts from `M_.endo_histval` with `histval`, e.g. after `smoother2histval`, `tests/conditional_forecast_histval.rs`); `varexo_det` paths in `forecast` (orders 1-2, `tests/exo_det.rs`) |
| `conditional_forecast` | command | done | (the `forecast` command starts from `M_.endo_histval` with `histval`, runs at `options_.order` 1-3 with simulated bands at order > 1, and takes its horizon from `options_.periods`; `conditional_forecast(parameter_set=calibration)` starts from `M_.endo_histval` with `histval`, e.g. after `smoother2histval`, `tests/conditional_forecast_histval.rs`); `varexo_det` paths in `forecast` (orders 1-2, `tests/exo_det.rs`) |
| `conditional_forecast_paths` | block | done | (the `forecast` command starts from `M_.endo_histval` with `histval`, runs at `options_.order` 1-3 with simulated bands at order > 1, and takes its horizon from `options_.periods`; `conditional_forecast(parameter_set=calibration)` starts from `M_.endo_histval` with `histval`, e.g. after `smoother2histval`, `tests/conditional_forecast_histval.rs`); `varexo_det` paths in `forecast` (orders 1-2, `tests/exo_det.rs`) |
| `plot_conditional_forecast` | command | done | (the `forecast` command starts from `M_.endo_histval` with `histval`, runs at `options_.order` 1-3 with simulated bands at order > 1, and takes its horizon from `options_.periods`; `conditional_forecast(parameter_set=calibration)` starts from `M_.endo_histval` with `histval`, e.g. after `smoother2histval`, `tests/conditional_forecast_histval.rs`); `varexo_det` paths in `forecast` (orders 1-2, `tests/exo_det.rs`) |
| `bvar_forecast` | command | done |  |
| `smoother2histval` | command | done | histval from the smoothed variables of the last estimation/calib_smoother at its point (posterior mean after an MCMC: the smoother at the posterior mean rather than the mean of the smoothed draws); `outfile` writes a dseries-layout .mat; as Dynare 7.2, the whole histval is reset first (exogenous variables to their steady state). the default `invars` set the `AUX_ENDO_LAG`/`AUX_EXO_LAG` rows of `M_.endo_histval` as MATLAB 7.2 (tests/smoother2histval.rs, reference/s2h) |
| `planner_objective` | command | done | (order-2 welfare, Ramsey with steady_state_model, bounded/order-2 OSR); order >= 3 welfare (k_order_welfare tensors, k_order_mean simulated unconditional welfare with `periods`/`drop`; shocks from ChaCha8 seeded with `--seed`, not MATLAB's randn; tests/policy.rs ramsey_order3_simulated_welfare feeds MATLAB's draws); Ramsey FOCs are built in the preprocessor DataTree emulation, so `AUX_ENDO_LEAD_*` names, equation order and incidence match Dynare |
| `evaluate_planner_objective` | command | done | (order-2 welfare, Ramsey with steady_state_model, bounded/order-2 OSR); order >= 3 welfare (k_order_welfare tensors, k_order_mean simulated unconditional welfare with `periods`/`drop`; shocks from ChaCha8 seeded with `--seed`, not MATLAB's randn; tests/policy.rs ramsey_order3_simulated_welfare feeds MATLAB's draws); Ramsey FOCs are built in the preprocessor DataTree emulation, so `AUX_ENDO_LEAD_*` names, equation order and incidence match Dynare |
| `ramsey_model` | command | done | (order-2 welfare, Ramsey with steady_state_model, bounded/order-2 OSR); order >= 3 welfare (k_order_welfare tensors, k_order_mean simulated unconditional welfare with `periods`/`drop`; shocks from ChaCha8 seeded with `--seed`, not MATLAB's randn; tests/policy.rs ramsey_order3_simulated_welfare feeds MATLAB's draws); Ramsey FOCs are built in the preprocessor DataTree emulation, so `AUX_ENDO_LEAD_*` names, equation order and incidence match Dynare |
| `ramsey_constraints` | block | done |  |
| `ramsey_policy` | command | done | (order-2 welfare, Ramsey with steady_state_model, bounded/order-2 OSR); order >= 3 welfare (k_order_welfare tensors, k_order_mean simulated unconditional welfare with `periods`/`drop`; shocks from ChaCha8 seeded with `--seed`, not MATLAB's randn; tests/policy.rs ramsey_order3_simulated_welfare feeds MATLAB's draws); Ramsey FOCs are built in the preprocessor DataTree emulation, so `AUX_ENDO_LEAD_*` names, equation order and incidence match Dynare |
| `discretionary_policy` | command | done | (order-2 welfare, Ramsey with steady_state_model, bounded/order-2 OSR); order >= 3 welfare (k_order_welfare tensors, k_order_mean simulated unconditional welfare with `periods`/`drop`; shocks from ChaCha8 seeded with `--seed`, not MATLAB's randn; tests/policy.rs ramsey_order3_simulated_welfare feeds MATLAB's draws); Ramsey FOCs are built in the preprocessor DataTree emulation, so `AUX_ENDO_LEAD_*` names, equation order and incidence match Dynare |
| `osr` | command | done | (order-2 welfare, Ramsey with steady_state_model, bounded/order-2 OSR); order >= 3 welfare (k_order_welfare tensors, k_order_mean simulated unconditional welfare with `periods`/`drop`; shocks from ChaCha8 seeded with `--seed`, not MATLAB's randn; tests/policy.rs ramsey_order3_simulated_welfare feeds MATLAB's draws); Ramsey FOCs are built in the preprocessor DataTree emulation, so `AUX_ENDO_LEAD_*` names, equation order and incidence match Dynare |
| `osr_params` | command | done | (order-2 welfare, Ramsey with steady_state_model, bounded/order-2 OSR); order >= 3 welfare (k_order_welfare tensors, k_order_mean simulated unconditional welfare with `periods`/`drop`; shocks from ChaCha8 seeded with `--seed`, not MATLAB's randn; tests/policy.rs ramsey_order3_simulated_welfare feeds MATLAB's draws); Ramsey FOCs are built in the preprocessor DataTree emulation, so `AUX_ENDO_LEAD_*` names, equation order and incidence match Dynare |
| `optim_weights` | block | done | (order-2 welfare, Ramsey with steady_state_model, bounded/order-2 OSR); order >= 3 welfare (k_order_welfare tensors, k_order_mean simulated unconditional welfare with `periods`/`drop`; shocks from ChaCha8 seeded with `--seed`, not MATLAB's randn; tests/policy.rs ramsey_order3_simulated_welfare feeds MATLAB's draws); Ramsey FOCs are built in the preprocessor DataTree emulation, so `AUX_ENDO_LEAD_*` names, equation order and incidence match Dynare |
| `osr_params_bounds` | block | done | (order-2 welfare, Ramsey with steady_state_model, bounded/order-2 OSR); order >= 3 welfare (k_order_welfare tensors, k_order_mean simulated unconditional welfare with `periods`/`drop`; shocks from ChaCha8 seeded with `--seed`, not MATLAB's randn; tests/policy.rs ramsey_order3_simulated_welfare feeds MATLAB's draws); Ramsey FOCs are built in the preprocessor DataTree emulation, so `AUX_ENDO_LEAD_*` names, equation order and incidence match Dynare |
| `sensitivity` | command | done | stability, redform, Morris, RMSE; `ppost` (posterior subsample of the MCMC files, also with `rmse`), `load_stab`/`load_redform`/`load_rmse`/`load_ident_files`, `glue` (also as `sensitivity(glue=1)`, which Dynare 7.2's preprocessor rejects), `identification` (map_identification: Morris screening, `morris=2`), the irf/moment calibration mapping |
| `irf_calibration` | block | done | mapped by `sensitivity` (map_calibration.m) and applied by `prior moments(distribution)`/`irfs(distribution)` (endogenous_prior_restrictions.m); applied wherever Dynare 7.2 reads `options_.endogenous_prior_restrictions`: dsge_likelihood.m (and the conditional-likelihood, OccBin and skewed-Kalman branches; info 49, `info(4)` = the summed squared distances, so the mode finders, mode_check, the MCMC and its initialization see them), prior_sampler.m (`prior simulate`: Endogenous prior violation share), `prior moments` (info 49 at the prior mode), stability_mapping.m (infox 49, `irestriction`, the `prior_calib` filtering, `x0`), `method_of_moments` (dropped with Dynare's warning); not used by non_linear_dsge_likelihood.m, DSGE-VAR or identification, as in Dynare (tests/calib_est.rs, reference/calib_est) |
| `moment_calibration` | block | done | mapped by `sensitivity` (map_calibration.m) and applied by `prior moments(distribution)`/`irfs(distribution)` (endogenous_prior_restrictions.m); applied wherever Dynare 7.2 reads `options_.endogenous_prior_restrictions`: dsge_likelihood.m (and the conditional-likelihood, OccBin and skewed-Kalman branches; info 49, `info(4)` = the summed squared distances, so the mode finders, mode_check, the MCMC and its initialization see them), prior_sampler.m (`prior simulate`: Endogenous prior violation share), `prior moments` (info 49 at the prior mode), stability_mapping.m (infox 49, `irestriction`, the `prior_calib` filtering, `x0`), `method_of_moments` (dropped with Dynare's warning); not used by non_linear_dsge_likelihood.m, DSGE-VAR or identification, as in Dynare (tests/calib_est.rs, reference/calib_est) |
| `identification` | command | done | `order` 2/3 (pruned state space, Jacobians by finite differences as `analytic_derivation_mode=-1`, 1e-6 from Dynare's analytic ones), `checks_via_subsets`, `gsa_sample_file`, `parameter_set=posterior_*`, `load_ident_files` (the point analysis; Dynare 7.2 itself fails without a saved Monte Carlo) |
| `markov_switching` | command | done | src/msbvar: parsed and validated as the preprocessor does (exclusion lag/equation, constants, upper/lower_cholesky, chains with regimes/durations/restrictions/parameters/number_of_lags, svar variances/coefficients/equations) into `MsDecl`; vector `duration` and `restrictions` are accepted although the 7.2 preprocessor aborts on a vector duration; `restriction equation N, [EXPR*]coeff(VAR,LAG) +/- ... = ...;` lines are parsed into rows of `Qi{N}`/`Ri{N}` exactly as the 7.2 preprocessor writes them (shared per-equation row counters with the exclusion groups, right-hand side negated, parameters at their calibrated values, "lag L used more than once", homogeneity and Qi-or-Ri errors; checked on its driver.m output, tests/msbvar.rs) |
| `svar` | command | done | src/msbvar: parsed and validated as the preprocessor does (exclusion lag/equation, constants, upper/lower_cholesky, chains with regimes/durations/restrictions/parameters/number_of_lags, svar variances/coefficients/equations) into `MsDecl`; vector `duration` and `restrictions` are accepted although the 7.2 preprocessor aborts on a vector duration; `restriction equation N, [EXPR*]coeff(VAR,LAG) +/- ... = ...;` lines are parsed into rows of `Qi{N}`/`Ri{N}` exactly as the 7.2 preprocessor writes them (shared per-equation row counters with the exclusion groups, right-hand side negated, parameters at their calibrated values, "lag L used more than once", homogeneity and Qi-or-Ri errors; checked on its driver.m output, tests/msbvar.rs) |
| `sbvar` | command | done | src/msbvar/sbvar.rs: Sims-Zha prior (dummy observations), exclusion/Cholesky restrictions, posterior mode of A0 (analytic-gradient BFGS in place of csminwel), A+ mode, impulse responses, point forecast and the marginal-density terms, outm.mat; checked to 1e-5 against MATLAB. The rest of sbvar (src/msbvar/sbvar_extra.rs): the Gibbs-sampled terms of the log marginal data density (`ms_mardd.m`: 1000 + 10000 draws per equation; deterministic pieces to 1e-2, `log mdd` within Monte Carlo error of Dynare: the term of the first equation is a heavy-tailed average with seed-to-seed std about 0.15), `outactqmygdata.prn` / `outyrqm.prn` (`fn_datana`, `vlistlog`; identical to the Dynare files), impulse-response, unconditional-forecast and growth-rate figures (`nograph`, `graph_format`; layout is ours, not that of MATLAB). Linear restrictions (`restriction equation` lines, src/msbvar/identification.rs): `Ui = null(Qi)`, `Vi = null(Ri)` (orthonormal), checked against Dynare 7.2 on two models (reference/msbvar/msbvar_lin*.mod, dump_msbvar_lin.m; mode, A+, IRFs, forecast, posterior matrices, mdd terms to 1e-5; Gibbs log mdd -469.84 as Dynare). `cross_restrictions`: Dynare 7.2 errors (msstart2.m:415 brace-indexes the NaN `ixmC0Pres` returned by exclusions.m; the preprocessor also rejects any line mixing lag 0 and lags > 0); here a mixed line with `cross_restrictions` is the restriction `R a+_j + Q a_j = 0`, and sbvar returns the exact restricted posterior mode (A0 objective plus the quadratic from integrating A+ over the restriction set, A+ by the conditional formula of msstart2.m), validated against a direct maximisation of the hand-built joint posterior (tests/msbvar.rs); no mdd with cross restrictions. Conditional forecasts do not exist in `sbvar` of Dynare: `msstart2.m` hard-codes `nconstr=0`, so only the unconditional forecast is produced |
| `svar_identification` | block | done | src/msbvar: parsed and validated as the preprocessor does (exclusion lag/equation, constants, upper/lower_cholesky, chains with regimes/durations/restrictions/parameters/number_of_lags, svar variances/coefficients/equations) into `MsDecl`; vector `duration` and `restrictions` are accepted although the 7.2 preprocessor aborts on a vector duration; `restriction equation N, [EXPR*]coeff(VAR,LAG) +/- ... = ...;` lines are parsed into rows of `Qi{N}`/`Ri{N}` exactly as the 7.2 preprocessor writes them (shared per-equation row counters with the exclusion groups, right-hand side negated, parameters at their calibrated values, "lag L used more than once", homogeneity and Qi-or-Ri errors; checked on its driver.m output, tests/msbvar.rs) |
| `ms_estimation` | command | done | MS-SBVAR (src/msbvar/ms_estimation.rs, ms_optim.rs, ms_files.rs): the Sims-Waggoner-Zha model (any number of chains, svar coefficients/variances per equation); `specification` random walk / `none` (Dynare's W_j) / `sims_zha` (lambda, psi), `contemp_reduced_form`, two-chain prior scaling, all pinned down against Dynare 7.2 (initial prior/likelihood to 1e-6, posterior at Dynare's mode = its objective); Dynare's mode search (block iterations VAR/Q with the convergence schedule, repeated runs, large/small perturbation stages from Metropolis draws; stochastic, attained mode compared with Dynare to 0.05, tests/msbvar_ms_est.rs); init_<file_tag>.dat / est_final_<output_file_tag>.out in Dynare's format (later ms_* commands read est_final_<file_tag>.out, also Dynare's own; `file_tag`, `output_file_tag`, `no_create_init`). `restriction equation` lines give the same general Ui/Vi as for sbvar; `cross_restrictions` fails as in Dynare 7.2 (undefined Fhat in ms_sbvar_setup.m); data from csv only (no xls_sheet/xls_range) |
| `ms_simulation` | command | done | src/msbvar/ms_commands.rs: adaptive random-walk Metropolis (mdd::rwmh) from the mode -> simulation_<tag>.out (log post, log lik, free parameters) with mh_replic/drop/thinning_factor/adaptive_mh_draws; Dynare's C sampler and its random numbers are not reproduced. `save_draws` writes draws_<tag>.out / draws_header_<tag>.out (est_flat layout) |
| `ms_compute_mdd` | command | done | src/msbvar/mdd.rs: port of dw_mdd_switch.c / dw_elliptical.c / dw_dirichlet_restrictions.c. The first 20% of the posterior draws fit the proposal, the other 80% are the posterior sample; proposal = elliptical on theta (centre = best fitting draw, scale = second moment about it, radius quantile bounds, `proposal_type` 1-6: Gaussian, power, truncated power, step, truncated Gaussian, uniform; power from E[r^2] = dim) x independent Dirichlet on the transition probabilities (Dynare's alpha rule); Muller, bridge and Waggoner-Zha (no cut) estimators. Dynare's mdd program switches on the Waggoner-Zha sign normalisation for the proposal draws only (nvars*ln 2 and -inf outside the region) while the posterior draws keep the kernel of the simulation file; reproduced, `MddOptions::consistent` gives the true mdd. On Dynare's own draws: proposal parameters and per-draw log proposal densities equal Dynare's (5e-7), Muller/bridge/WZ on its proposal matrix equal its printed values (1e-4); with our proposal draws the means over 20 seeds match those of 20 Dynare runs to 0.035 (Monte Carlo). use_mean_center is ignored as in Dynare (its MEX never sees the flag) |
| `ms_compute_probabilities` | command | done | src/msbvar/hamilton.rs: filtered and Kim-smoothed regime probabilities of the latest ms_estimation of the run (filtered_/smoothed_<fname>.out, checked to 1e-6 against Dynare). `real_time_smoothed` (real_time_smoothed_<tag>.out, checked to 2e-6) and the per-chain probability figures (src/msbvar/ms_graphs.rs; nograph/graph_format honoured) |
| `ms_irf` | command | done | src/msbvar/ms_sim.rs: regime-fixed impulse responses (regime, regimes, ergodic default, filtered_probabilities), `error_band_percentiles`/`median`, `parameter_uncertainty` from simulation_<tag>.out, `free_parameters`; IRF files checked against Dynare 7.2 to 2e-5 (the percentiles are those of the exact regime mixture, not of sampled draws). regime-specific figures (ms_graphs.rs); `shock_draws` regime sampling is replaced by exact integration |
| `ms_forecast` | command | done | src/msbvar/ms_sim.rs: simulated forecast percentiles (regime-fixed per regime / `regime`; otherwise start from the filtered probabilities with the regime moving by Q), `data_obs_nbr`, `parameter_uncertainty`; reproduces Dynare 7.2 (incl. its first forecast row being the one-step forecast of the last observation) to Monte-Carlo accuracy (0.05 on percentiles); figures via ms_graphs.rs |
| `ms_variance_decomposition` | command | done | src/msbvar/ms_sim.rs: shares of the shocks in the forecast-error variance per regime / ergodic / filtered, mean (`no_error_bands`) or percentiles; regime-specific means equal Dynare's to 2e-5, the mixtures to 3e-3 (Dynare samples its regime weights). Percentile version differs from Dynare's by ~4e-4 (unexplained); figures via ms_graphs.rs |
| `epilogue` | block | done | `epilogue_dynamic`/`epilogue_static` (incl. the 6-decimal rounding of dseries' `from ... do`), used by `with_epilogue` decompositions as Dynare; also written over the smoothed variables (`Output/<fname>_epilogue_smoothed_variables.csv`) and perfect-foresight paths (`_epilogue_endo_simul.csv`) |
| `heterogeneity_dimension` | command | done | (src/heterogeneity, parser_het.rs): one dimension (Dynare 7.2 allows one too); `M_.heterogeneity(1)` (names, tex/long names, aux_vars of type 15/16, `state_var`, sparse Jacobian structure, `dynamic_mcp_equations_reordering`, `het_aux_levels`, `Sigma_e`, equation tags), `M_.heterogeneity_aggregates` and the type-14 `SUM_x` variables are checked against the Dynare 7.2 preprocessor and MATLAB (reference/heterogeneity, tests/heterogeneity.rs) |
| `var` | command | done | `$tex$`, `long_name`; `log` (LOG_ aux variables, `substituteLogTransform`), `deflator`/`log_deflator` (detrending, exact aux names; checked against the Dynare 7.2 preprocessor and MATLAB) |
| `model` | block | done | `linear` (and `model(linear)`); equation tags, `#` locals, `STEADY_STATE()`; `use_dll`/`bytecode` are code-generation options with no counterpart (accepted); `block`/`mfs`/`static_mfs`/`cutoff`/`no_static` set the block decomposition (`M_.block_structure`), used by `steady` and `perfect_foresight_solver` for block-wise solving (`solve_block_decomposed_problem`, evaluate/solve/two-boundary blocks; checked against Dynare 7.2, reference/block_solve; `stack_solve_algo=4` runs the `lnsrch1` path-length search on the stacked two-boundary blocks, checked against Dynare 7.2 (reference/block_solve/ref_block_solve_a4.json); bytecode is n/a: no bytecode path) |
| `shocks` | block | done | stderr/var/cov/corr as expressions (correlations evaluated at the end of their block, as the generated code), measurement errors, `overwrite` (resets `Sigma_e`/`H`/det shocks, not `Skew_e`); deterministic `periods` with integers, ranges, lists and dates (`first_simulation_period`/`last_simulation_period`), `values` with expressions and MATLAB workspace vectors `(xx)`; `learnt_in = INTEGER / DATE` with `values`/`add`/`multiply`; `surprise` (OccBin); `skew`: pruned skewed Kalman filter and, in `stoch_simul`, the `check_model.m` bound, the COSKEWNESS table and closed-skew-normal draws for `periods > 0` (`rand_multivariate_csn.m`; Dynare uses `Skew_e` nowhere else: theoretical moments, IRFs/GIRFs and decision rules ignore it); all checked against Dynare 7.2 (reference/shocks). `shocks(heterogeneity=NAME)`: `M_.heterogeneity(1).Sigma_e` of the heterogeneous shocks (see `heterogeneity_dimension`) |
| `heterogeneity_load_steady_state` | command | done | port of `+heterogeneity/load_steady_state.m` and `check_steady_state_input.m` (+ the `find_bracket_linear_weight` and `compute_Phi_tilde_e` MEX): `oo_.heterogeneity.{steady_state,sizes,indices,mat}` (grids, brackets, Mu, Phi, Phi_e, x_bar, d.x_bar, aggregates, G/dG, F/dF) checked to 1e-12 against MATLAB. The steady state comes from a MAT file (`filename`, `variable`) or from `<variable>.mat` next to the .mod file: a `verbatim` MATLAB block cannot be run |
| `heterogeneity_compute_steady_state` | command | done | port of `compute_steady_state.m` and of the `compute_steady_state_tensor` MEX (time iteration with Fischer-Burmeister complementarity and the trust-region solver of `block_trust_region`, forward iteration of the distribution, aggregation, Broyden calibration of free parameters), same iteration path as MATLAB (policies and distribution to 1e-11, calibrated parameters to 1e-12; Krusell-Smith and HANK one asset, reference/heterogeneity). Same MAT-file input rule as `heterogeneity_load_steady_state` |
| `rouwenhorst` | matcomm | done | `rouwenhorst(rho, sigma, N, tol, maxiter)` (`mex/sources/heterogeneity/rouwenhorst.f08`), identical to Dynare to 1e-14 (src/heterogeneity/rouwenhorst.rs) |
| `heterogeneity_solve` | command | done | port of `+heterogeneity/solve.m` (sequence-space Jacobian method: policy impulse responses, `curlyDs`, `curlyYs`, expectation vectors, fake-news matrices `F`, heterogeneous Jacobians `J_ha`, general-equilibrium Jacobians `G`, the aggregate Jacobian `dr.J`) and of the MEX `compute_expected_dx`, `compute_curlyDs`, `compute_expected_y`, `compute_transition_matrices` (sources not distributed; semantics pinned against MATLAB). `dr.G`, `F`, `J_ha`, `curlyYs`, `curlyDs`, `J` and the policy impulse responses match Dynare 7.2 to 1e-12 (Krusell-Smith, HANK one asset; reference/heterogeneity/dump_solve.m, tests/heterogeneity.rs). Result in `HetSession::dr` |
| `heterogeneity_simulate` | command | done | port of `+heterogeneity/simulate.m`, `simulate_irfs.m`, `simulate_stochastic_shocks.m`, `simulate_news_shocks.m`: IRFs (one-standard-deviation or `relative_irf` unit shocks, `irf_shocks`, variable list, `irf` horizon check), stochastic simulation (`periods`) and news-shock sequences (`shocks` block with `periods`/`values`, `make_ex_` rules, option-compatibility error); IRFs, the convolution given the draws and news paths match Dynare 7.2 to 1e-12. Deviations: no graphs (`plot_irfs.m`/`plot_simulation.m`: `<fname>_IRF_<shock>[k]`, `<fname>_simulation[k]`, `<fname>_det_simulation[k]` via the stoch_simul IRF figures, honouring `nograph`/`graph_format`; no TeX eps-loader file), stochastic draws come from dynare-rust's ChaCha8 stream (MATLAB's `randn` stream is not reproduced); results kept in `HetSession::sim`, not written to files |
| `var_model` | command | done | VAR family (src/varmodels, parser_vm.rs): AR/A0/constants of the tagged equations (`varmatrices.m`, `trend_component_ar_a0.m`), companion matrices, `var_expectation_model_<m>_*` parameters and the `var_expectation(m)` operator (also `model_name = m`), `time_shift`, `horizon = n / a:b / a:Inf`, `structural`; checked against MATLAB to 1e-12 (`reference/varmodels`, `tests/varmodels.rs`). Dynare 7.2 quirk reproduced: `update_parameters.m` puts the *index* of a parameter in `expression` into alpha |
| `trend_component_model` | command | done | VAR family (src/varmodels, parser_vm.rs): AR/A0/constants of the tagged equations (`varmatrices.m`, `trend_component_ar_a0.m`), companion matrices, `var_expectation_model_<m>_*` parameters and the `var_expectation(m)` operator (also `model_name = m`), `time_shift`, `horizon = n / a:b / a:Inf`, `structural`; checked against MATLAB to 1e-12 (`reference/varmodels`, `tests/varmodels.rs`). Dynare 7.2 quirk reproduced: `update_parameters.m` puts the *index* of a parameter in `expression` into alpha |
| `var_expectation_model` | command | done | VAR family (src/varmodels, parser_vm.rs): AR/A0/constants of the tagged equations (`varmatrices.m`, `trend_component_ar_a0.m`), companion matrices, `var_expectation_model_<m>_*` parameters and the `var_expectation(m)` operator (also `model_name = m`), `time_shift`, `horizon = n / a:b / a:Inf`, `structural`; checked against MATLAB to 1e-12 (`reference/varmodels`, `tests/varmodels.rs`). Dynare 7.2 quirk reproduced: `update_parameters.m` puts the *index* of a parameter in `expression` into alpha |
| `var_expectation.initialize` | matcomm | done | statements in the .mod, run in file order (src/varmodels); also `get_companion_matrix(NAME)` |
| `var_expectation.update` | matcomm | done | statements in the .mod, run in file order (src/varmodels); also `get_companion_matrix(NAME)` |
| `pac_model` | command | done | PAC models (src/varmodels/pac.rs, modfile/parser_pac.rs): the PAC equation is decomposed as the preprocessor does (error correction `ec*(target(-1)-z(-1))`, `ar_k*diff(z(-k))`, with or without `pac_target_nonstationary`); `pac_expectation(m)` becomes the auxiliary endogenous variable `pac_expectation_<m>` (or the `auxname`s of `pac_target_info`, with `target`, `auxname_target_nonstationary`, `component`/`kind`/`auxname`/`growth`) defined by the linear form in the `h_<m>_*` parameters; `growth` adds the `<m>_pac_growth_neutrality_correction` term; `kind = dd / dl / ll`; without `auxiliary_model_name` the model consistent expectations recursion `mce_Z1_<m>` with the `mce_alpha_<m>_*` parameters. Checked against MATLAB (VAR with composite target and growth, trend component with growth, VAR simple target, MCE; `reference/pac`, `tests/pac.rs`). Not ported: rule of thumb agents and additive-term bgp (`pac.bgp.*`, `pac.print`, `pac.check`), composite targets with MCE; the auxiliary variables count in the `stoch_simul` tables, as declared variables |
| `pac_target_info` | block | done | PAC models (src/varmodels/pac.rs, modfile/parser_pac.rs): the PAC equation is decomposed as the preprocessor does (error correction `ec*(target(-1)-z(-1))`, `ar_k*diff(z(-k))`, with or without `pac_target_nonstationary`); `pac_expectation(m)` becomes the auxiliary endogenous variable `pac_expectation_<m>` (or the `auxname`s of `pac_target_info`, with `target`, `auxname_target_nonstationary`, `component`/`kind`/`auxname`/`growth`) defined by the linear form in the `h_<m>_*` parameters; `growth` adds the `<m>_pac_growth_neutrality_correction` term; `kind = dd / dl / ll`; without `auxiliary_model_name` the model consistent expectations recursion `mce_Z1_<m>` with the `mce_alpha_<m>_*` parameters. Checked against MATLAB (VAR with composite target and growth, trend component with growth, VAR simple target, MCE; `reference/pac`, `tests/pac.rs`). Not ported: rule of thumb agents and additive-term bgp (`pac.bgp.*`, `pac.print`, `pac.check`), composite targets with MCE; the auxiliary variables count in the `stoch_simul` tables, as declared variables |
| `pac.initialize` | matcomm | done | `pac.initialize(NAME)`, `pac.update.expectation(NAME)` (h0/h1 parameters and growth neutrality correction: `hVectors.m`, `+pac/+update/parameters.m`; `pac.update.equation` is the deprecated alias) and `pac.mce.parameters(NAME)` run as statements in file order; parameters checked to 1e-10 against MATLAB (`tests/pac.rs`) |
| `pac.update` | matcomm | done | `pac.initialize(NAME)`, `pac.update.expectation(NAME)` (h0/h1 parameters and growth neutrality correction: `hVectors.m`, `+pac/+update/parameters.m`; `pac.update.equation` is the deprecated alias) and `pac.mce.parameters(NAME)` run as statements in file order; parameters checked to 1e-10 against MATLAB (`tests/pac.rs`) |
| `pac.estimate.nls` | matcomm | done | statements in the .mod (`tests/pac_estimate_mod.rs`): `eparams.x = v;` structure, `edata = dseries('file')`, `dates(..):dates(..)+N` range; Gauss-Newton as `gauss_newton.m`, Newey-West covariance, iterative OLS, estimates and h parameters written to the model parameters, checked against MATLAB (`tests/pac.rs`, `tests/pac_estimate_mod.rs`). Optimizers `lsqnonlin`/`fmincon`/`annealing`/`particleswarm` bound the error correction parameter to [0,1] via projected Gauss-Newton, not the MATLAB algorithms; no `oo_.pac` structure; data only from a dseries file |
| `pac.estimate.iterative_ols` | matcomm | done | statements in the .mod (`tests/pac_estimate_mod.rs`): `eparams.x = v;` structure, `edata = dseries('file')`, `dates(..):dates(..)+N` range; Gauss-Newton as `gauss_newton.m`, Newey-West covariance, iterative OLS, estimates and h parameters written to the model parameters, checked against MATLAB (`tests/pac.rs`, `tests/pac_estimate_mod.rs`). Optimizers `lsqnonlin`/`fmincon`/`annealing`/`particleswarm` bound the error correction parameter to [0,1] via projected Gauss-Newton, not the MATLAB algorithms; no `oo_.pac` structure; data only from a dseries file |
| `rplot` | command | done |  |
| `dynatype` | command | n/a | accepted with a note; the paths are in `Output/<fname>_endo_simul.csv` |
| `dynasave` | command | n/a | accepted with a note; the paths are in `Output/<fname>_endo_simul.csv` |
| `@#includepath` | macrodir | done | checked against dynare-preprocessor |
| `@#include` | macrodir | done | checked against dynare-preprocessor |
| `@#define` | macrodir | done | checked against dynare-preprocessor |
| `@#if` | macrodir | done | checked against dynare-preprocessor |
| `@#for` | macrodir | done | tuples and `when` filters |
| `@#echo` | macrodir | done |  |
| `@#error` | macrodir | done |  |
| `@#echomacrovars` | macrodir | done | displays the variables and functions; `(save)`: `options_.macrovars_line_<n>` written to `Output/<fname>_macrovars.json` (functions shown with their body as written) |
| `verbatim` | block | done | skipped as in Dynare (MATLAB code) |
| `set_dynare_seed` | command | done | seeds dynare-rust's ChaCha8 streams (the algorithm name is ignored) |
| `save_params_and_steady_state` | command | done | (load: read at preprocessing time, as the preprocessor does); checked against Dynare's file |
| `load_params_and_steady_state` | command | done | (load: read at preprocessing time, as the preprocessor does); checked against Dynare's file |
| `compilation_setup` | command | n/a | no external compiler; accepted with a note |
| `dynare_version` | matcomm | done |  |
| `write_latex_definitions` | matcomm | done |  |
| `write_latex_parameter_table` | matcomm | done |  |
| `write_latex_prior_table` | matcomm | done |  |
| `collect_latex_files` | matcomm | done |  |
| `send_endogenous_variables_to_workspace` | matcomm | n/a | MATLAB workspace / development helpers; accepted with a note |
| `send_exogenous_variables_to_workspace` | matcomm | n/a | MATLAB workspace / development helpers; accepted with a note |
| `send_irfs_to_workspace` | matcomm | n/a | MATLAB workspace / development helpers; accepted with a note |
| `clean_current_folder` | matcomm | n/a | MATLAB workspace / development helpers; accepted with a note |
| `prior_function` | command | n/a | run a user MATLAB function on prior/posterior draws: no Rust counterpart (would need a plug-in mechanism) |
| `posterior_function` | command | n/a | run a user MATLAB function on prior/posterior draws: no Rust counterpart (would need a plug-in mechanism) |
| `generate_trace_plots` | matcomm | done | `trace_plot(options_, M_, estim_params_, TYPE, BLOCK, NAME)` too |
| `internals` | matcomm | n/a | MATLAB workspace / development helpers; accepted with a note |
| `prior` | matcomm | done | `table` (checked against Dynare's printout), `plot`, `simulate` (prior mass and BK shares on dynare-rust's draws), `moments` (theoretical moments at the prior mode); `moments(distribution)`, `irfs(distribution)` (dynare-rust's draws, same printout and .mat file), `optimize` |
| `search` | matcomm | n/a | MATLAB workspace / development helpers; accepted with a note |
| `dplot` | matcomm | n/a | MATLAB workspace / development helpers; accepted with a note |
| `set_dynare_threads` | matcomm | n/a | MATLAB workspace / development helpers; accepted with a note |

## Options

### `var` (done)

| option | status |
|---|---|
| `log` | done |
| `deflator = MODEL_EXPR` | done |
| `log_deflator = MODEL_EXPR` | done |
| `long_name = QUOTED_STRING` | done |
| `NAME = QUOTED_STRING` | done |

### `varexo_det` (done)

| option | status |
|---|---|
| `long_name = QUOTED_STRING` | done |
| `NAME = QUOTED_STRING` | done |

### `external_function` (done)

| option | status |
|---|---|
| `name = NAME` | done |
| `nargs = INTEGER` | done |
| `first_deriv_provided [= NAME]` | done |
| `second_deriv_provided [= NAME]` | done |

### `model` (done)

| option | status |
|---|---|
| `linear` | done |
| `use_dll` | n/a |
| `block` | done (decomposition and block-wise steady / perfect-foresight solving) |
| `bytecode` | n/a |
| `cutoff = DOUBLE` | done |
| `mfs = INTEGER` | done |
| `static_mfs` | done |
| `no_static` | done |
| `differentiate_forward_vars` | done |
| `parallel_local_files = ( FILENAME [, FILENAME]... )` | n/a |
| `balanced_growth_test_tol = DOUBLE` | done |

### `write_latex_original_model` (done)

| option | status |
|---|---|
| `write_equation_tags` | done |

### `write_latex_dynamic_model` (done)

| option | status |
|---|---|
| `write_equation_tags` | done |

### `write_latex_static_model` (done)

| option | status |
|---|---|
| `write_equation_tags` | done |

### `initval` (done)

| option | status |
|---|---|
| `all_values_required` | missing |

### `endval` (done)

| option | status |
|---|---|
| `all_values_required` | missing |

### `histval` (done)

| option | status |
|---|---|
| `all_values_required` | missing |

### `resid` (done)

| option | status |
|---|---|
| `non_zero` | done |

### `initval_file` (done)

| option | status |
|---|---|
| `datafile = FILENAME` | done |
| `first_obs = {INTEGER \| DATE}` | done |
| `first_simulation_period = {INTEGER \| DATE}` | done |
| `last_simulation_period = {INTEGER \| DATE}` | done |
| `last_obs = {INTEGER \| DATE}` | done |
| `nobs = INTEGER` | done |
| `series = DSERIES NAME` | done |

### `histval_file` (done)

| option | status |
|---|---|
| `datafile = FILENAME` | done |
| `first_obs = {INTEGER \| DATE}` | done |
| `first_simulation_period = {INTEGER \| DATE}` | done |
| `last_obs = {INTEGER \| DATE}` | done |
| `series = DSERIES NAME` | done |

### `mshocks` (done)

| option | status |
|---|---|
| `overwrite` | done |
| `relative_to_initval` | done |

### `steady` (done)

| option | status |
|---|---|
| `maxit = INTEGER` | done |
| `tolf = DOUBLE` | done |
| `tolx = DOUBLE` | done |
| `non_zero` | done |
| `solve_algo = INTEGER` | done |
| `homotopy_mode = INTEGER` | done |
| `homotopy_steps = INTEGER` | done |
| `homotopy_force_continue = INTEGER` | done |
| `nocheck` | done |
| `noprint` | done |
| `markowitz = DOUBLE` | n/a |
| `fsolve_options = (NAME, VALUE, ...)` | done (tolerances/iterations; Display, MaxFunEvals n/a) |

### `check` (done)

| option | status |
|---|---|
| `solve_algo = INTEGER` | done |
| `qz_zero_threshold = DOUBLE` | done |

### `model_info` (done)

| option | status |
|---|---|
| `block_static` | done |
| `block_dynamic` | done |
| `incidence` | done |

### `perfect_foresight_setup` (done)

| option | status |
|---|---|
| `periods = INTEGER` | done |
| `first_simulation_period = DATE` | done |
| `last_simulation_period = DATE` | done |
| `datafile = FILENAME` | done |
| `endval_steady` | done |
| `endval_steady_nocheck` | done |

### `perfect_foresight_solver` (done)

| option | status |
|---|---|
| `maxit = INTEGER` | done |
| `tolf = DOUBLE` | done |
| `tolx = DOUBLE` | done |
| `noprint` | done |
| `print` | done |
| `stack_solve_algo = INTEGER` | done |
| `preconditioner = OPTION` | done |
| `iter_tol = DOUBLE` | done |
| `iter_maxit = INTEGER` | done |
| `gmres_restart = INTEGER` | done |
| `block_diagonal_lu_maxlu = INTEGER` | done |
| `block_diagonal_lu_nperiods = INTEGER` | done |
| `block_diagonal_lu_nlu = INTEGER` | done |
| `block_diagonal_lu_relu = DOUBLE` | done |
| `robust_lin_solve` | done |
| `allow_nonfinite_values` | done |
| `check_jacobian_singularity` | done |
| `solve_algo` | done |
| `no_homotopy` | done |
| `homotopy_initial_step_size = DOUBLE` | done |
| `homotopy_min_step_size = DOUBLE` | done |
| `homotopy_step_size_increase_success_count = INTEGER` | done |
| `homotopy_linearization_fallback` | done |
| `homotopy_marginal_linearization_fallback [= DOUBLE]` | done |
| `homotopy_max_completion_share = DOUBLE` | done |
| `homotopy_exclude_varexo = (VARIABLE_NAME...)` | done |
| `markowitz = DOUBLE` | done |
| `minimal_solving_periods = INTEGER` | n/a |
| `lmmcp` | done |
| `endogenous_terminal_period` | done |
| `linear_approximation` | done |
| `steady_solve_algo = INTEGER` | done |
| `steady_tolf = DOUBLE` | done |
| `steady_tolx = DOUBLE` | done |
| `steady_maxit = INTEGER` | done |
| `steady_markowitz = DOUBLE` | done |

### `mshocks` (done)

| option | status |
|---|---|
| `overwrite` | done |
| `relative_to_initval` | done |

### `perfect_foresight_with_expectation_errors_setup` (done)

| option | status |
|---|---|
| `periods = INTEGER` | done |
| `first_simulation_period = DATE` | done |
| `last_simulation_period = DATE` | done |
| `datafile = FILENAME` | done |
| `endval_steady_nocheck` | done |

### `perfect_foresight_with_expectation_errors_solver` (done)

| option | status |
|---|---|
| `constant_simulation_length` | done |

### `perfect_foresight_controlled_paths` (done)

| option | status |
|---|---|
| `learnt_in = INTEGER \| DATE` | done |

### `shock_paths` (done)

| option | status |
|---|---|
| `learnt_in = INTEGER \| DATE` | done |
| `overwrite` | done |

### `stoch_simul` (done)

| option | status |
|---|---|
| `ar = INTEGER` | done |
| `drop = INTEGER` | done |
| `hp_filter = DOUBLE` | done |
| `one_sided_hp_filter = DOUBLE` | done |
| `bandpass_filter` | done |
| `bandpass_filter = [HIGHEST_PERIODICITY LOWEST_PERIODICITY]` | done |
| `filtered_theoretical_moments_grid = INTEGER` | done |
| `irf = INTEGER` | done |
| `irf_shocks = ( VARIABLE_NAME [[,] VARIABLE_NAME ...] )` | done |
| `relative_irf` | done |
| `irf_plot_threshold = DOUBLE` | done |
| `nocorr` | done |
| `nodecomposition` | done |
| `nofunctions` | done |
| `nomoments` | done |
| `nomodelsummary` | done |
| `nograph` | done |
| `graph` | done |
| `nodisplay` | done |
| `graph_format = FORMAT` | done |
| `noprint` | done |
| `print` | done |
| `order = INTEGER` | done |
| `k_order_solver` | done |
| `periods = INTEGER` | done |
| `qz_criterium = DOUBLE` | done |
| `qz_zero_threshold = DOUBLE` | done |
| `replic = INTEGER` | done |
| `simul_replic = INTEGER` | done |
| `solve_algo = INTEGER` | done |
| `aim_solver` | done |
| `conditional_variance_decomposition = INTEGER` | done |
| `pruning` | done |
| `partial_information` | done |
| `dr = OPTION` | done |
| `dr_cycle_reduction_tol = DOUBLE` | done |
| `dr_cycle_reduction_maxiter = INTEGER` | done |
| `dr_logarithmic_reduction_tol = DOUBLE` | done |
| `dr_logarithmic_reduction_maxiter = INTEGER` | done |
| `loglinear` | done |
| `tex` | done |
| `dr_display_tol = DOUBLE` | done |
| `contemporaneous_correlation` | done |
| `spectral_density` | done |

### `extended_path` (done)

| option | status |
|---|---|
| `periods = INTEGER` | done |
| `solver_periods = INTEGER` | done |
| `order = INTEGER` | done |
| `hybrid [ = INTEGER ]` | done |
| `lmmcp` | done |
| `use_first_order_solution` | done |
| `integration = OPTION` | done |
| `tree = OPTION` | done |
| `number_of_quadrature_nodes = INTEGER` | done |

### `occbin_setup` (done)

| option | status |
|---|---|
| `simul_periods = INTEGER` | done |
| `simul_maxit = INTEGER` | done |
| `simul_check_ahead_periods = INTEGER` | done |
| `simul_reset_check_ahead_periods` | done |
| `simul_reset_regime_in_new_period` | done |
| `simul_max_check_ahead_periods = INTEGER` | done |
| `simul_curb_retrench` | done |
| `simul_periodic_solution` | done |
| `simul_periodic_solution_threshold = INTEGER` | done |
| `simul_periodic_solution_strict` | done |
| `simul_debug` | done |
| `smoother_periods = INTEGER` | done |
| `smoother_maxit = INTEGER` | done |
| `smoother_check_ahead_periods = INTEGER` | done |
| `smoother_max_check_ahead_periods = INTEGER` | missing |
| `smoother_curb_retrench` | done |
| `smoother_periodic_solution` | done |
| `smoother_inversion_filter` | done |
| `smoother_piecewise_kalman_filter` | done |
| `smoother_plot` | missing |
| `smoother_first_period_occbin_update = INTEGER` | missing |
| `smoother_max_number_of_iterations = INTEGER` | missing |
| `likelihood_periods = INTEGER` | done |
| `likelihood_maxit = INTEGER` | done |
| `likelihood_check_ahead_periods = INTEGER` | done |
| `likelihood_curb_retrench` | done |
| `likelihood_periodic_solution` | done |
| `likelihood_inversion_filter` | done |
| `likelihood_piecewise_kalman_filter` | done |
| `likelihood_max_kalman_iterations` | done |
| `filter_use_relaxation` | done |
| `filter_particle_draw_states_from_empirical_density` | missing |
| `filter_particle_state_importance_sampling_pkf_init` | missing |
| `filter_particle_state_importance_sampling_logpost_crit_threshold = DOUBLE` | missing |
| `filter_particle_state_importance_sampling_slice_override_iteration = INTEGER` | missing |
| `filter_particle_state_importance_sampling_slice_burnin = INTEGER` | missing |
| `filter_particle_initial_state_ergodic_simul` | missing |
| `filter_particle_diagnostics` | missing |
| `filter_particle_diagnostics_graph_periods = INTEGER_VECTOR` | missing |
| `filter_particle_diagnostics_nograph` | missing |
| `filter_particle_number_of_particles  = INTEGER` | missing |
| `filter_particle_number_of_shocks_per_particle = INTEGER` | missing |
| `filter_particle_state_draws   = NUMERICAL_VECTOR` | missing |
| `particle_filtering` | missing |
| `filter_particle_use_pkf_updated_state_threshold = INTEGER` | missing |
| `filter_init_periods_using_particles` | missing |
| `posterior_importance_sampling` | missing |
| `posterior_importance_sampling_filter = QUOTED_STRING` | missing |
| `posterior_importance_sampling_orig_filter = QUOTED_STRING` | missing |
| `posterior_importance_sampling_sub_draws = INTEGER` | missing |
| `posterior_importance_sampling_orig_dname = FILENAME` | missing |
| `posterior_importance_sampling_orig_fname = FILENAME` | missing |

### `occbin_solver` (done)

| option | status |
|---|---|
| `simul_periods = INTEGER` | done |
| `simul_maxit = INTEGER` | done |
| `simul_check_ahead_periods = INTEGER` | done |
| `simul_reset_check_ahead_periods` | done |
| `simul_max_check_ahead_periods` | done |
| `simul_curb_retrench` | done |
| `simul_debug` | done |

### `occbin_graph` (done)

| option | status |
|---|---|
| `noconstant` | done |

### `occbin_write_regimes` (done)

| option | status |
|---|---|
| `periods = INTEGER` | done |
| `filename = FILENAME` | done |
| `simul` | done |
| `smoother` | done |

### `estimated_params` (done)

| option | status |
|---|---|
| `INITIAL_VALUE` | done |
| `LOWER_BOUND` | done |
| `UPPER_BOUND` | done |
| `PRIOR_SHAPE` | done |
| `PRIOR_MEAN` | done |
| `PRIOR_STANDARD_ERROR` | done |
| `PRIOR_3RD_PARAMETER` | done |
| `PRIOR_4TH_PARAMETER` | done |
| `SCALE_PARAMETER` | done |

### `estimated_params_init` (done)

| option | status |
|---|---|
| `use_calibration` | done |

### `estimation` (done)

| option | status |
|---|---|
| `datafile = FILENAME` | done |
| `dirname = FILENAME` | done |
| `xls_sheet = QUOTED_STRING` | done |
| `xls_range = RANGE` | done |
| `nobs = INTEGER` | done |
| `nobs = [INTEGER1:INTEGER2]` | done |
| `first_obs = INTEGER` | done |
| `first_obs = [INTEGER1:INTEGER2]` | done |
| `prefilter = INTEGER` | done |
| `presample = INTEGER` | done |
| `loglinear` | done |
| `logdata` | done |
| `plot_priors = INTEGER` | done |
| `nograph` | done |
| `posterior_nograph` | done (options_.no_graph.posterior: pm3/PosteriorIRF figures) |
| `posterior_graph` | done |
| `nodisplay` | done |
| `graph_format = FORMAT` | done |
| `tex` | done |
| `mode_compute = INTEGER \| FUNCTION_NAME` | done (102: simulannealbnd substitute, 13 outside method_of_moments errors as in Dynare 7.2) |
| `mode_file = FILENAME` | done |
| `additional_optimizer_steps = [INTEGER]` | done |
| `optim = (NAME, VALUE, ...)` | done (newrat `Hessian` 0/1/2: outer product / hessian.m / mixed, mr_hessian on the univariate filter at the mode as Dynare) |
| `silent_optimizer` | done |
| `cova_compute = INTEGER` | done (no Hessian, mode file without hh, Dynare's MCMC error) |
| `use_penalized_objective_for_hessian` | done (hessian.m on penalty_objective_function) |
| `analytic_derivation` | done (analytic score and Hessians of dsge_likelihood: kalman_filter/univariate_kalman_filter with derivatives, expanded state for correlated measurement errors, analytic_Hessian opg/full/asymptotic, second-order parameter derivatives of the solution; analytic gradients in mode_compute 1/3 substitutes, 4, 5 (OPG Hessian per iteration), 101; the exact Hessian at the mode; analytic_derivation_mode 0/1/-1/-2; Dynare's errors; gradients/Hessians 1e-12, modes/Hessians vs MATLAB runs, tests/est_complete_ad.rs) |
| `mode_check` | done |
| `mode_check_neighbourhood_size = DOUBLE` | done |
| `mode_check_symmetric_plots = INTEGER` | done |
| `mode_check_number_of_points = INTEGER` | done |
| `no_init_estimation_check_first_obs` | done (initial_estimation_checks.m stochastic-singularity checks, checked against Dynare) |
| `lik_init = INTEGER` | done |
| `conditional_likelihood` | done |
| `kalman_algo = INTEGER` | done (5: pruned skewed Kalman filter and smoother kalman_smoother_pruned_skewed.m; missing data via the Gaussian missing-observations filter and lik_init 1/2/4/5 as Dynare, lik_init=3 Dynare's error) |
| `fast_kalman_filter` | done |
| `kalman_tol = DOUBLE` | done |
| `diffuse_kalman_tol = DOUBLE` | done |
| `filter_covariance` | done |
| `filter_step_ahead = [INTEGER1:INTEGER2]` | done |
| `filter_decomposition` | done |
| `diffuse_filter` | done |
| `heteroskedastic_filter` | done |
| `use_univariate_filters_if_singularity_is_detected = INTEGER` | done |
| `use_univariate_smoother_if_singularity_is_detected` | done (DsgeSmoother.m: error 422 unless set or noprint) |
| `rescale_prediction_error_covariance` | done |
| `lyapunov = OPTION` | done (square_root_solver: the Schur solution) |
| `lyapunov_fixed_point_tol = DOUBLE` | done |
| `lyapunov_doubling_tol = DOUBLE` | done |
| `skewed_kalman_prune_tol = DOUBLE` | done |
| `skewed_kalman_rank_deficiency_transform` | done (csn_statespace_linear_transform.m + csn_singular_to_csn_proper.m; checked against Dynare, 1e-16) |
| `skewed_kalman_mvnlogcdf = QUOTED_STRING` | done (gaussian_log_mvncdf_mendell_elston and mvncdf: Dynare's missing/stats mvncdf with bvncdf/tvncdf/mvtcdfqmc ports; d<=3 bit-level vs MATLAB, d>=4 randomized QMC with fixed-seed shifts, within its 1e-4 tolerance) |
| `moments_varendo` | done |
| `ar = INTEGER` | done |
| `contemporaneous_correlation` | done (posterior moments: covariance_mc_analysis.m) |
| `no_posterior_kernel_density` | done (posterior_moments.m densities of pm3 objects and posterior moments, checked against Dynare) |
| `conditional_variance_decomposition = INTEGER` | done |
| `filtered_vars` | done |
| `frequentist_smoother = BOOLEAN` | done (explicit true/false as Dynare; without the option the classical smoother runs only with smoother/filtered_vars) |
| `smoother` | done |
| `smoother_redux` | done (same results, computed on the full state space) |
| `smoothed_state_uncertainty` | done |
| `forecast = INTEGER` | done |
| `conf_sig = DOUBLE` | done |
| `bayesian_irf` | done |
| `relative_irf` | done |
| `irf = INTEGER` | done |
| `irf_shocks = ( VARIABLE_NAME [[,] VARIABLE_NAME ...] )` | done |
| `irf_plot_threshold = DOUBLE` | done |
| `consider_all_endogenous` | done |
| `consider_all_endogenous_and_auxiliary` | done |
| `consider_only_observed` | done |
| `selected_variables_only` | done (accepted; all variables are written) |
| `solve_algo = INTEGER` | done |
| `order = INTEGER` | done |
| `dr = OPTION` | done (cycle_reduction/logarithmic_reduction in the likelihood, checked against Dynare) |
| `dr_cycle_reduction_tol = DOUBLE` | done |
| `dr_cycle_reduction_maxiter = INTEGER` | done |
| `dr_logarithmic_reduction_tol = DOUBLE` | done |
| `dr_logarithmic_reduction_maxiter = INTEGER` | done |
| `qz_zero_threshold = DOUBLE` | done (likelihood, checked against Dynare) |
| `posterior_sampling_method = NAME` | done |
| `posterior_sampler_options = (NAME, VALUE, ...)` | done |
| `sub_draws = INTEGER` | done |
| `posterior_max_subsample_draws = INTEGER` | done |
| `prior_trunc = DOUBLE` | done |
| `huge_number = DOUBLE` | done (mode_compute 2/10 bounds, mode_check) |
| `endogenous_prior` | done |
| `estimate_initial_states_endogenous_prior` | done (init <var> parameters sixth in xparam1, lik_init=2 with Harvey_scale_factor=0, init_state_endogenous_prior.m, set_init_state.m, draw_init_state_from_smoother.m in the slice sampler; likelihood, projection and smoothed initial state checked against Dynare, MCMC statistically; draw_init_state_with_rotated_slice not ported) |
| `mh_conf_sig = DOUBLE` | done |
| `mh_replic = INTEGER` | done |
| `mh_nblocks = INTEGER` | done |
| `mh_drop = DOUBLE` | done |
| `mh_jscale = DOUBLE` | done |
| `mh_init_scale = DOUBLE (deprecated)` | done |
| `mh_init_scale_factor = DOUBLE` | done |
| `mh_tune_jscale [= DOUBLE]` | done |
| `mh_tune_guess = DOUBLE` | done |
| `mh_recover` | done |
| `mh_posterior_mode_estimation` | done |
| `load_mh_file` | done |
| `load_results_after_load_mh` | done |
| `mh_initialize_from_previous_mcmc` | done |
| `mh_initialize_from_previous_mcmc_directory = FILENAME` | done |
| `mh_initialize_from_previous_mcmc_record = FILENAME` | done |
| `mh_initialize_from_previous_mcmc_prior = FILENAME` | done |
| `mcmc_jumping_covariance = OPTION` | done |
| `nodiagnostic` | done |
| `taper_steps = [INTEGER1 INTEGER2 ...]` | done |
| `brooks_gelman_plotrows = INTEGER` | done |
| `geweke_interval = [DOUBLE DOUBLE]` | done |
| `raftery_lewis_diagnostics` | done |
| `raftery_lewis_qrs = [DOUBLE DOUBLE DOUBLE]` | done |
| `dsge_var` | done |
| `dsge_varlag = INTEGER` | done |
| `pruning` | done |
| `number_of_particles = INTEGER` | done |
| `filter_algorithm = OPTION` | done |
| `resampling = OPTION` | done |
| `resampling_threshold = DOUBLE` | done |
| `resampling_method = OPTION` | done |
| `proposal_approximation = OPTION` | done |
| `distribution_approximation = OPTION` | done |
| `cpf_weights = OPTION` | done |
| `nonlinear_filter_initialization = INTEGER` | done (options_.particle.initialization; 3 checked against Dynare) |
| `particle_filter_options = (NAME, VALUE, ...)` | done |

### `data` (done)

| option | status |
|---|---|
| `file = FILENAME` | done |
| `series = DSERIES` | n/a |
| `xls_sheet = QUOTED_STRING` | done |
| `xls_range = RANGE` | done |
| `nobs = INTEGER` | done |
| `first_obs = DATE` | done |
| `last_obs = DATE` | done |

### `method_of_moments` (done)

| option | status |
|---|---|
| `mom_method = SMM\|GMM\|IRF_MATCHING` | done |
| `datafile = FILENAME` | done |
| `order = INTEGER` | done |
| `pruning` | done |
| `verbose` | done |
| `penalized_estimator` | done |
| `weighting_matrix = ['WM1','WM2',...,'WMn']` | done |
| `weighting_matrix_scaling_factor = DOUBLE` | done |
| `bartlett_kernel_lag = INTEGER` | done |
| `se_tolx = DOUBLE` | done |
| `burnin = INTEGER` | done |
| `bounded_shock_support` | done |
| `seed = INTEGER` | done |
| `simulation_multiple = INTEGER` | done |
| `hp_filter = DOUBLE` | done |
| `one_sided_hp_filter = DOUBLE` | done |
| `bandpass_filter` | done |
| `bandpass_filter = [HIGHEST_PERIODICITY LOWEST_PERIODICITY]` | done |
| `analytic_standard_errors` | done |
| `simulation_method = METHOD` | done |
| `irf_matching_file = FILENAME` | done |
| `add_tiny_number_to_cholesky = DOUBLE` | done |
| `drop = INTEGER` | done |
| `relative_irf` | done |
| `replic = INTEGER` | done |
| `dirname = FILENAME` | done |
| `graph_format = FORMAT` | done |
| `nodisplay` | done |
| `nograph` | done |
| `noprint` | done |
| `plot_priors = INTEGER` | done |
| `prior_trunc = DOUBLE` | done |
| `tex` | done |
| `prefilter = INTEGER` | done |
| `first_obs = INTEGER` | done |
| `nobs = INTEGER` | done |
| `logdata` | done |
| `xls_sheet = QUOTED_STRING` | done |
| `xls_range = RANGE` | done |
| `mode_file = FILENAME` | done |
| `mode_compute = INTEGER \| FUNCTION_NAME` | done |
| `additional_optimizer_steps = [INTEGER]` | done |
| `optim = (NAME, VALUE, ...)` | done |
| `analytic_jacobian` | done |
| `huge_number = DOUBLE` | done |
| `silent_optimizer` | done |
| `use_penalized_objective_for_hessian` | done |
| `posterior_sampling_method = NAME` | done |
| `posterior_sampler_options = (NAME, VALUE, ...)` | done |
| `mh_posterior_mode_estimation` | done |
| `cova_compute = INTEGER` | done |
| `mcmc_jumping_covariance = OPTION` | done |
| `mh_replic = INTEGER` | done |
| `mh_nblocks = INTEGER` | done |
| `mh_jscale = DOUBLE` | done |
| `mh_tune_jscale [= DOUBLE]` | done |
| `mh_tune_guess = DOUBLE` | done |
| `mh_conf_sig = DOUBLE` | done |
| `mh_drop = DOUBLE` | done |
| `mh_init_scale_factor = DOUBLE` | done |
| `no_posterior_kernel_density` | done |
| `posterior_max_subsample_draws = INTEGER` | done |
| `sub_draws = INTEGER` | done |
| `use_pct` | n/a |
| `load_mh_file` | done |
| `load_results_after_load_mh` | done |
| `mh_initialize_from_previous_mcmc` | done |
| `mh_initialize_from_previous_mcmc_directory = FILENAME` | done |
| `mh_initialize_from_previous_mcmc_prior = FILENAME` | done |
| `mh_initialize_from_previous_mcmc_record = FILENAME` | done |
| `mh_recover` | done |
| `nodiagnostic` | done |
| `brooks_gelman_plotrows = INTEGER` | done |
| `geweke_interval = [DOUBLE DOUBLE]` | done |
| `taper_steps = [INTEGER1 INTEGER2 ...]` | done |
| `raftery_lewis_diagnostics` | done |
| `raftery_lewis_qrs = [DOUBLE DOUBLE DOUBLE]` | done |
| `k_order_solver` | done |
| `dr = OPTION` | done |
| `dr_cycle_reduction_tol = DOUBLE` | done |
| `dr_cycle_reduction_maxiter = INTEGER` | done |
| `dr_logarithmic_reduction_tol = DOUBLE` | done |
| `dr_logarithmic_reduction_maxiter = INTEGER` | done |
| `lyapunov = OPTION` | done |
| `lyapunov_complex_threshold = DOUBLE` | done |
| `lyapunov_fixed_point_tol = DOUBLE` | done |
| `lyapunov_doubling_tol = DOUBLE` | done |
| `qz_criterium = DOUBLE` | done |
| `qz_zero_threshold = DOUBLE` | done |
| `schur_vec_tol = DOUBLE` | done |
| `mode_check` | done |
| `mode_check_neighbourhood_size = DOUBLE` | done |
| `mode_check_symmetric_plots = INTEGER` | done |
| `mode_check_number_of_points = INTEGER` | done |

### `model_comparison` (done)

| option | status |
|---|---|
| `marginal_density = ESTIMATOR` | done |

### `shock_decomposition` (done)

| option | status |
|---|---|
| `parameter_set = OPTION` | done |
| `datafile = FILENAME` | done |
| `first_obs = INTEGER` | done |
| `nobs = INTEGER` | done |
| `prefilter = INTEGER` | missing |
| `loglinear` | done |
| `diffuse_kalman_tol = DOUBLE` | done |
| `diffuse_filter` | done |
| `kalman_algo = INTEGER` | done |
| `kalman_tol = DOUBLE` | done |
| `use_univariate_filters_if_singularity_is_detected` | missing |
| `xls_sheet = QUOTED_STRING` | done |
| `xls_range = RANGE` | done |
| `use_shock_groups [= NAME]` | done |
| `colormap = VARIABLE_NAME` | done |
| `nograph` | done |
| `init_state = BOOLEAN` | done |
| `with_epilogue` | done |
| `forecast_type = unconditional \| conditional` | done |

### `shock_groups` (done)

| option | status |
|---|---|
| `name = NAME` | done |

### `realtime_shock_decomposition` (done)

| option | status |
|---|---|
| `parameter_set = OPTION` | done |
| `datafile = FILENAME` | done |
| `first_obs = INTEGER` | done |
| `nobs = INTEGER` | done |
| `use_shock_groups [= NAME]` | done |
| `colormap = VARIABLE_NAME` | done |
| `nograph` | done |
| `presample = INTEGER` | done |
| `forecast = INTEGER` | done |
| `save_realtime = INTEGER_VECTOR` | done |
| `fast_realtime = INTEGER` | done |
| `with_epilogue` | done |
| `kalman_algo = INTEGER` | done |
| `kalman_tol = DOUBLE` | done |

### `plot_shock_decomposition` (done)

| option | status |
|---|---|
| `use_shock_groups [= NAME]` | done |
| `colormap = VARIABLE_NAME` | done |
| `nodisplay` | done |
| `nograph` | done |
| `graph_format = FORMAT` | done |
| `detail_plot` | done |
| `interactive` | n/a |
| `screen_shocks` | done |
| `steadystate` | done |
| `type = qoq \| yoy \| aoa` | done |
| `fig_name = STRING` | done |
| `write_xls` | done |
| `realtime = INTEGER` | done |
| `vintage = INTEGER` | done |
| `plot_init_date = DATE` | done |
| `plot_end_date = DATE` | done |
| `diff` | done |
| `flip` | done |
| `max_nrows` | done |
| `with_epilogue` | done |
| `init2shocks` | done |

### `init2shocks` (done)

| option | status |
|---|---|
| `name = NAME` | done |

### `initial_condition_decomposition` (done)

| option | status |
|---|---|
| `colormap = VARIABLE_NAME` | missing |
| `nodisplay` | done |
| `graph_format = FORMAT` | done |
| `detail_plot` | done |
| `steadystate` | missing |
| `type = qoq \| yoy \| aoa` | done |
| `fig_name = STRING` | done |
| `write_xls` | missing |
| `plot_init_date = DATE` | done |
| `plot_end_date = DATE` | done |
| `diff` | done |
| `flip` | done |

### `calib_smoother` (done)

| option | status |
|---|---|
| `datafile = FILENAME` | done |
| `filtered_vars` | done |
| `filter_step_ahead = [INTEGER1:INTEGER2]` | done |
| `prefilter = INTEGER` | done |
| `parameter_set = OPTION` | done |
| `loglinear` | done |
| `first_obs = INTEGER` | done |
| `filter_decomposition` | done |
| `filter_covariance` | done |
| `smoother_redux` | done |
| `kalman_algo = INTEGER` | done |
| `diffuse_filter = INTEGER` | done |
| `diffuse_kalman_tol = DOUBLE` | done |
| `use_univariate_smoother_if_singularity_is_detected` | missing |
| `xls_sheet = QUOTED_STRING` | done |
| `xls_range = RANGE` | done |
| `heteroskedastic_filter` | done |
| `nobs = INTEGER` | done |

### `forecast` (done)

| option | status |
|---|---|
| `periods = INTEGER` | done |
| `conf_sig = DOUBLE` | done |
| `nograph` | done |
| `nodisplay` | done |
| `graph_format = FORMAT` | done |

### `conditional_forecast` (done)

| option | status |
|---|---|
| `parameter_set = OPTION` | done |
| `controlled_varexo = (VARIABLE_NAME...)` | done |
| `periods = INTEGER` | done |
| `replic = INTEGER` | done |
| `conf_sig = DOUBLE` | done |

### `plot_conditional_forecast` (done)

| option | status |
|---|---|
| `periods = INTEGER` | done |

### `smoother2histval` (done)

| option | status |
|---|---|
| `period = INTEGER` | done |
| `invars = ( VARIABLE_NAME [VARIABLE_NAME ...] )` | done |
| `outfile = FILENAME` | done |
| `outvars = ( VARIABLE_NAME [VARIABLE_NAME ...] )` | done |

### `evaluate_planner_objective` (done)

| option | status |
|---|---|
| `periods = INTEGER` | done (options_.ramsey.periods, order >= 3 simulated welfare) |
| `drop = INTEGER` | done (options_.ramsey.drop, order >= 3 simulated welfare) |

### `ramsey_model` (done)

| option | status |
|---|---|
| `planner_discount = EXPRESSION` | done |
| `planner_discount_latex_name = LATEX_NAME` | done |
| `instruments = (VARIABLE_NAME,...)` | done |

### `ramsey_policy` (done)

| option | status |
|---|---|
| `planner_discount = EXPRESSION` | done |
| `instruments = (VARIABLE_NAME,...)` | done |

### `discretionary_policy` (done)

| option | status |
|---|---|
| `discretionary_tol = NON-NEGATIVE DOUBLE` | done |
| `maxit = INTEGER` | done |

### `osr` (done)

| option | status |
|---|---|
| `opt_algo = INTEGER` | done |
| `optim = (NAME, VALUE, ...)` | done |
| `maxit = INTEGER` | done |
| `tolf = DOUBLE` | done |
| `analytic_derivation` | missing |
| `analytic_derivation_mode = INTEGER` | missing |
| `silent_optimizer` | done |
| `huge_number = DOUBLE` | missing |

### `sensitivity` (done)

| option | status |
|---|---|
| `Nsam = INTEGER` | done |
| `ilptau = INTEGER` | done |
| `pprior = INTEGER` | done |
| `prior_range = INTEGER` | done |
| `morris = INTEGER` | done |
| `morris_nliv = INTEGER` | done |
| `morris_ntra = INTEGER` | done |
| `ppost = INTEGER` | done |
| `neighborhood_width = DOUBLE` | done |
| `stab = INTEGER` | done |
| `load_stab = INTEGER` | done |
| `alpha2_stab = DOUBLE` | done |
| `pvalue_ks = DOUBLE` | done |
| `pvalue_corr = DOUBLE` | done |
| `redform = INTEGER` | done |
| `load_redform = INTEGER` | done |
| `logtrans_redform = INTEGER` | done |
| `threshold_redform = [DOUBLE DOUBLE]` | done |
| `ksstat_redform = DOUBLE` | done |
| `alpha2_redform = DOUBLE` | done |
| `namendo = (VARIABLE_NAME...)` | done |
| `namlagendo = (VARIABLE_NAME...)` | done |
| `namexo = (VARIABLE_NAME...)` | done |
| `rmse = INTEGER` | done |
| `load_rmse = INTEGER` | done |
| `lik_only = INTEGER` | done |
| `var_rmse = (VARIABLE_NAME...)` | done |
| `pfilt_rmse = DOUBLE` | done |
| `istart_rmse = INTEGER` | done |
| `alpha_rmse = DOUBLE` | done |
| `alpha2_rmse = DOUBLE` | done |
| `datafile = FILENAME` | done |
| `nobs = INTEGER` | done |
| `first_obs = INTEGER` | done |
| `prefilter = INTEGER` | missing |
| `presample = INTEGER` | done |
| `nograph` | done |
| `nodisplay` | missing |
| `graph_format = FORMAT` | done |
| `conf_sig = DOUBLE` | missing |
| `loglinear` | missing |
| `mode_file = FILENAME` | done |
| `kalman_algo = INTEGER` | done |
| `identification = INTEGER` | done |
| `load_ident_files = INTEGER` | done |
| `useautocorr = INTEGER` | done |
| `ar = INTEGER` | done |
| `diffuse_filter = INTEGER` | done |
| `kalman_tol = DOUBLE` | done |
| `diffuse_kalman_tol = DOUBLE` | done |
| `use_univariate_smoother_if_singularity_is_detected` | missing |

### `irf_calibration` (done)

| option | status |
|---|---|
| `relative_irf` | done |

### `identification` (done)

| option | status |
|---|---|
| `order = 1\|2\|3` | done |
| `parameter_set = OPTION` | done |
| `prior_mc = INTEGER` | done |
| `prior_range = INTEGER` | done |
| `advanced = INTEGER` | done |
| `max_dim_cova_group = INTEGER` | done |
| `gsa_sample_file = INTEGER\|FILENAME` | done |
| `diffuse_filter` | done |
| `lik_init = INTEGER` | done |
| `kalman_algo = INTEGER` | done |
| `kalman_tol = DOUBLE` | done |
| `diffuse_kalman_tol = DOUBLE` | done |
| `use_univariate_filters_if_singularity_is_detected` | done |
| `analytic_derivation_mode = INTEGER` | done |
| `normalize_jacobians = INTEGER` | done |
| `tol_rank = DOUBLE` | done |
| `tol_deriv = DOUBLE` | done |
| `tol_sv = DOUBLE` | done |
| `schur_vec_tol = DOUBLE` | done |
| `no_identification_strength` | done |
| `periods = INTEGER` | done |
| `replic = INTEGER` | done |
| `no_identification_moments` | done |
| `ar = INTEGER` | done |
| `useautocorr = INTEGER` | done |
| `no_identification_spectrum` | done |
| `grid_nbr = INTEGER` | done |
| `no_identification_minimal` | done |
| `nograph` | done |
| `nodisplay` | done |
| `graph_format = FORMAT` | done |
| `tex` | done |
| `load_ident_files = INTEGER` | partial |
| `no_identification_reducedform` | done |
| `checks_via_subsets = INTEGER` | done |
| `max_dim_subsets_groups = INTEGER` | done |

### `markov_switching` (done)

| option | status |
|---|---|
| `chain = INTEGER` | done |
| `number_of_regimes = INTEGER` | done |
| `duration = DOUBLE \| [ROW VECTOR OF DOUBLES]` | done |
| `restrictions = [[ROW VECTOR OF 3 DOUBLES],[ROW VECTOR OF 3 DOUBLES],...]` | done |
| `parameters = [LIST OF PARAMETERS]` | done |
| `number_of_lags = DOUBLE` | done |

### `svar` (done)

| option | status |
|---|---|
| `coefficients` | done |
| `variances` | done |
| `equations` | done |
| `chain = INTEGER` | done |

### `ms_estimation` (done)

| option | status |
|---|---|
| `file_tag = FILENAME` | done |
| `output_file_tag = FILENAME` | done |
| `no_create_init` | done |
| `coefficients_prior_hyperparameters = [DOUBLE1 DOUBLE2 ... DOUBLE6]` | done |
| `freq = INTEGER \| monthly \| quarterly \| yearly` | done |
| `initial_year = INTEGER` | done |
| `initial_subperiod = INTEGER` | done |
| `final_year = INTEGER` | done |
| `final_subperiod = INTEGER` | done |
| `datafile = FILENAME` | done |
| `xls_sheet = QUOTED_STRING` | missing |
| `xls_range = RANGE` | missing |
| `nlags = INTEGER` | done |
| `cross_restrictions` | done |
| `contemp_reduced_form` | done |
| `no_bayesian_prior` | done |
| `alpha = INTEGER` | done |
| `beta = INTEGER` | done |
| `gsig2_lmdm = INTEGER` | done |
| `specification = sims_zha \| none` | done |
| `convergence_starting_value = DOUBLE` | done |
| `convergence_ending_value = DOUBLE` | done |
| `convergence_increment_value = DOUBLE` | done |
| `max_iterations_starting_value = INTEGER` | done |
| `max_iterations_increment_value = DOUBLE` | done |
| `max_block_iterations = INTEGER` | done |
| `max_repeated_optimization_runs = INTEGER` | done |
| `function_convergence_criterion = DOUBLE` | done |
| `parameter_convergence_criterion = DOUBLE` | done |
| `number_of_large_perturbations = INTEGER` | done |
| `number_of_small_perturbations = INTEGER` | done |
| `number_of_posterior_draws_after_perturbation = INTEGER` | done |
| `max_number_of_stages = INTEGER` | done |
| `random_function_convergence_criterion = DOUBLE` | done |
| `random_parameter_convergence_criterion = DOUBLE` | done |

### `ms_simulation` (done)

| option | status |
|---|---|
| `file_tag = FILENAME` | done |
| `output_file_tag = FILENAME` | done |
| `mh_replic = INTEGER` | done |
| `drop = INTEGER` | done |
| `thinning_factor = INTEGER` | done |
| `adaptive_mh_draws = INTEGER` | done |
| `save_draws` | done |

### `ms_compute_mdd` (done)

| option | status |
|---|---|
| `file_tag = FILENAME` | done |
| `output_file_tag = FILENAME` | done |
| `simulation_file_tag = FILENAME` | done |
| `proposal_type = INTEGER` | done |
| `proposal_lower_bound = DOUBLE` | done |
| `proposal_upper_bound = DOUBLE` | done |
| `mdd_proposal_draws = INTEGER` | missing |
| `mdd_use_mean_center` | missing |

### `ms_compute_probabilities` (done)

| option | status |
|---|---|
| `file_tag = FILENAME` | done |
| `output_file_tag = FILENAME` | done |
| `filtered_probabilities` | done |
| `real_time_smoothed` | done |

### `ms_irf` (done)

| option | status |
|---|---|
| `file_tag = FILENAME` | done |
| `output_file_tag = FILENAME` | done |
| `simulation_file_tag = FILENAME` | done |
| `horizon = INTEGER` | done |
| `filtered_probabilities` | done |
| `error_band_percentiles = [DOUBLE1 ...]` | done |
| `shock_draws = INTEGER` | done |
| `shocks_per_parameter = INTEGER` | done |
| `thinning_factor = INTEGER` | done |
| `free_parameters = NUMERICAL_VECTOR` | done |
| `parameter_uncertainty` | done |
| `regime = INTEGER` | done |
| `regimes` | done |
| `median` | done |

### `ms_forecast` (done)

| option | status |
|---|---|
| `file_tag = FILENAME` | done |
| `output_file_tag = FILENAME` | done |
| `simulation_file_tag = FILENAME` | done |
| `data_obs_nbr = INTEGER` | done |
| `error_band_percentiles = [DOUBLE1 ...]` | done |
| `shock_draws = INTEGER` | done |
| `shocks_per_parameter = INTEGER` | done |
| `thinning_factor = INTEGER` | done |
| `free_parameters = NUMERICAL_VECTOR` | done |
| `parameter_uncertainty` | done |
| `regime = INTEGER` | done |
| `regimes` | done |
| `median` | done |
| `horizon = INTEGER` | done |

### `ms_variance_decomposition` (done)

| option | status |
|---|---|
| `file_tag = FILENAME` | done |
| `output_file_tag = FILENAME` | done |
| `simulation_file_tag = FILENAME` | done |
| `horizon = INTEGER` | done |
| `filtered_probabilities` | done |
| `no_error_bands` | done |
| `error_band_percentiles = [DOUBLE1 ...]` | done |
| `shock_draws = INTEGER` | done |
| `shocks_per_parameter = INTEGER` | done |
| `thinning_factor = INTEGER` | done |
| `free_parameters = NUMERICAL_VECTOR` | done |
| `parameter_uncertainty` | done |
| `regime = INTEGER` | done |
| `regimes` | done |

### `heterogeneity_load_steady_state` (done)

| option | status |
|---|---|
| `filename = FILENAME` | done |
| `variable = STRING` | done |
| `tolf = DOUBLE` | done |

### `heterogeneity_compute_steady_state` (done)

| option | status |
|---|---|
| `filename = FILENAME` | done |
| `variable = STRING` | done |
| `calibration_target_equations = [QUOTED_STRING \| INTEGER[, QUOTED_STRING \| INTEGER[, ...]]]` | done |
| `calibration_tolf = DOUBLE` | done |
| `calibration_max_iter = INTEGER` | done |
| `calibration_verbosity = INTEGER` | done |
| `time_iteration_max_iter = INTEGER` | done |
| `time_iteration_tol = DOUBLE` | done |
| `time_iteration_learning_rate = DOUBLE` | done |
| `time_iteration_early_stopping = INTEGER` | done |
| `time_iteration_verbosity = INTEGER` | done |
| `time_iteration_solver_tolf = DOUBLE` | done |
| `time_iteration_solver_tolx = DOUBLE` | done |
| `time_iteration_solver_factor = DOUBLE` | done |
| `time_iteration_solver_max_iter = INTEGER` | done |
| `time_iteration_solver_stop_on_error` | done |
| `forward_max_iter = INTEGER` | done |
| `forward_tol = DOUBLE` | done |
| `forward_check_every = INTEGER` | done |
| `forward_verbosity = INTEGER` | done |

### `heterogeneity_solve` (done)

| option | status |
|---|---|
| `truncation_horizon = INTEGER` | done |

### `heterogeneity_simulate` (done)

| option | status |
|---|---|
| `irf = INTEGER` | done |
| `periods = INTEGER` | done |
| `irf_shocks = ( SHOCK_NAME [[,] SHOCK_NAME ...] )` | done |
| `relative_irf` | done |
| `nograph` | partial |
| `nodisplay` | done |
| `graph_format = FORMAT` | done |
| `tex` | missing |
| `irf_plot_threshold = DOUBLE` | done |
| `print` | missing |
| `noprint` | done |

### `var_model` (done)

| option | status |
|---|---|
| `model_name = STRING` | done |
| `eqtags = [QUOTED_STRING[, QUOTED_STRING[, ...]]]` | done |
| `structural` | done |

### `trend_component_model` (done)

| option | status |
|---|---|
| `model_name = STRING` | done |
| `eqtags = [QUOTED_STRING[, QUOTED_STRING[, ...]]]` | done |
| `targets = [QUOTED_STRING[, QUOTED_STRING[, ...]]]` | done |

### `var_expectation_model` (done)

| option | status |
|---|---|
| `model_name = STRING` | done |
| `auxiliary_model = STRING` | done |
| `expression = VARIABLE_NAME \| EXPRESSION` | done |
| `discount = PARAMETER_NAME \| DOUBLE` | done |
| `horizon = INTEGER \| [INTEGER:INTEGER]` | done |
| `time_shift = INTEGER` | done |

### `pac_model` (done)

| option | status |
|---|---|
| `model_name = STRING` | done |
| `auxiliary_model = STRING` | done |
| `discount = PARAMETER_NAME \| DOUBLE` | done |
| `growth = PARAMETER_NAME \| VARIABLE_NAME \| EXPRESSION \| DOUBLE` | done |
| `kind = dd \| dl` | done |
| `auxname = STRING` | done |

### `compilation_setup` (n/a)

| option | status |
|---|---|
| `compiler = FILENAME` | n/a |
| `substitute_flags = QUOTED_STRING` | n/a |
| `add_flags = QUOTED_STRING` | n/a |
| `substitute_libs = QUOTED_STRING` | n/a |
| `add_libs = QUOTED_STRING` | n/a |

### `prior_function` (n/a)

| option | status |
|---|---|
| `function = FUNCTION_NAME` | n/a |
| `sampling_draws = INTEGER` | n/a |

### `posterior_function` (n/a)

| option | status |
|---|---|
| `function = FUNCTION_NAME` | n/a |
| `sampling_draws = INTEGER` | n/a |

### `prior` (done)

| option | status |
|---|---|
| `table` | done |
| `moments` | done |
| `moments(distribution)` | done |
| `irfs(distribution)` | missing |
| `optimize` | done |
| `simulate` | done |
| `plot` | done |

### `dplot` (n/a)

| option | status |
|---|---|
| `--expression EXPRESSION` | n/a |
| `--dseries NAME` | n/a |
| `--range DATE1:DATE2` | n/a |
| `--style MATLAB_SCRIPT_NAME` | n/a |
| `--title MATLAB_STRING` | n/a |
| `--with-legend` | n/a |

## Command-line options (`dynare-rs model.mod [OPTIONS]`)

| option | status |
|---|---|
| `noclearall` | n/a (accepted) |
| `onlyclearglobals` | n/a (accepted) |
| `debug` | n/a (accepted) |
| `notmpterms` | n/a (accepted) |
| `savemacro[=FILENAME]` | done |
| `onlymacro` | done |
| `linemacro` | done |
| `onlymodel` | done |
| `nolog` | n/a (accepted) |
| `output=first\|second\|third` | n/a (accepted) |
| `language=matlab\|julia` | n/a (accepted) |
| `params_derivs_order=0\|1\|2` | n/a (accepted with a note) |
| `nowarn` | n/a (accepted) |
| `notime` | n/a (accepted) |
| `transform_unary_ops` | n/a (accepted) |
| `json = parse\|check\|transform\|compute` | partial (modfile/modfile-original/static/dynamic.json; model, aux_vars, AST, declarations, variable_mapping, sparsity, model_info as the preprocessor; derivative strings without temporary terms; a subset of statements) |
| `jsonstdout` | done |
| `onlyjson` | done |
| `jsonderivsimple` | done |
| `warn_uninit` | n/a (accepted) |
| `console` | n/a (accepted) |
| `nograph` | done |
| `nointeractive` | n/a (accepted) |
| `nopathchange` | n/a (accepted) |
| `nopreprocessoroutput` | n/a (accepted) |
| `mexext=mex\|mexw64\|mexmaci64\|mexmaca64\|mexa64` | n/a (accepted) |
| `matlabroot=<<path>>` | n/a (accepted) |
| `parallel[=CLUSTER_NAME]` | n/a (accepted) |
| `conffile=FILENAME` | n/a (accepted) |
| `parallel_follower_open_mode` | n/a (accepted) |
| `parallel_test` | n/a (accepted) |
| `parallel_use_psexec=true\|false` | n/a (accepted) |
| `-DMACRO_VARIABLE[=MACRO_EXPRESSION]` | done |
| `-I<<path>>` | done |
| `nostrict` | partial (undeclared model symbols become exogenous; more variables than equations still an error) |
| `fast` | n/a (accepted: no generated files to skip) |
| `minimal_workspace` | n/a (accepted: no MATLAB workspace) |
| `compute_xrefs` | n/a (accepted) |
| `stochastic` | done |
| `exclude_eqs=<<equation_tags_to_exclude>>` | done (variables turned exogenous by symbol id) |
| `include_eqs=<<equation_tags_to_include>>` | done (untagged equations cannot be excluded) |
| `use_dll` | n/a (accepted) |
| `nocommutativity` | n/a (accepted) |

Also `--seed`, `--mh-replic`, `--threads`, `--prefetch`, `--blas`, `--quiet`, `--window` (dynare-rust's own).
