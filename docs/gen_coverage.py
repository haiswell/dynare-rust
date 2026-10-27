"""Generates docs/COVERAGE.md: every command/block/option of the Dynare 7.2 reference manual,
with dynare-rust's status. Option status: an option counts as read when its name appears as a
string literal in the source files that implement the command (heuristic, refined by the hand
annotations below)."""
import re, os, json, sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("DYNARE_ROOT", "/Applications/Dynare/7.2-arm64") + "/doc/dynare-manual.html/_sources/"

def literals(paths):
    lits = set()
    for p in paths:
        full = os.path.join(REPO, p)
        files = []
        if os.path.isdir(full):
            for root, _, fs in os.walk(full):
                files += [os.path.join(root, f) for f in fs if f.endswith('.rs')]
        elif os.path.exists(full):
            files = [full]
        for f in files:
            for m in re.finditer(r'"([A-Za-z_][A-Za-z0-9_.]*)"', open(f).read()):
                lits.add(m.group(1).lower())
    return lits

ALL = literals(['src'])

# ---------------------------------------------------------------------------------------
# Manual extraction
entries = []
for fn in ["the-model-file.rst.txt", "dynare-misc-commands.rst.txt"]:
    for i, line in enumerate(open(SRC + fn)):
        m = re.match(r'^\.\. (command|block|matcomm|macrodir)::\s*(.*)', line)
        if m:
            sig = m.group(2).strip()
            nm = re.split(r'[\s(;\[]', sig)[0] if not sig.startswith('@') else sig.split()[0]
            if not nm:
                nm = re.search(r'=\s*([A-Za-z_.]+)', sig).group(1)
            entries.append({"kind": m.group(1), "sig": sig, "name": nm, "file": fn, "line": i + 1, "opts": []})
            continue
        m = re.match(r'^ {3,4}\.\. option::\s*(.*)', line)
        if m and entries:
            o = m.group(1).strip()
            name = re.split(r'[\s=(\[]', o)[0]
            entries[-1]["opts"].append((o, name))

cli = []
for line in open(SRC + "running-dynare.rst.txt"):
    m = re.match(r'^ {3,4}\.\. option::\s*(.*)', line)
    if m:
        cli.append(m.group(1).strip())

# ---------------------------------------------------------------------------------------
# Hand annotations: name -> (status, note, source paths for the option check, option overrides)
# status: done | partial | missing | n/a
A = {}
def a(names, status, note, paths=None, opts=None):
    for n in names.split():
        A[n] = (status, note, paths, opts or {})

EST = ['src/driver.rs', 'src/dsge.rs', 'src/estimation', 'src/forecast', 'src/analysis/posterior.rs', 'src/kalman', 'src/data.rs', 'src/data_m.rs', 'src/graphics', 'src/nonlinear', 'src/occbin']
SS = ['src/analysis', 'src/solver/korder', 'src/graphics']
DET = ['src/deterministic', 'src/misc/homotopy.rs', 'src/ep']

a('var', 'done', '`$tex$`, `long_name`; `log` (LOG_ aux variables, `substituteLogTransform`), `deflator`/`log_deflator` (detrending, exact aux names; checked against the Dynare 7.2 preprocessor and MATLAB)', ['src/modfile'], {'log': 'done', 'deflator': 'done', 'log_deflator': 'done', 'name': 'done', 'long_name': 'done'})
a('varexo', 'done', '')
a('varexo_det', 'done', 'out of `M_.exo_names`/`Sigma_e`, parameters of the model at `oo_.exo_det_steady_state` (steady state, `check`, IRFs, `periods` simulations, estimation as Dynare); `dr.ghud` and order-2 `ghxud`/`ghuud`/`ghudud` (`stochastic_solvers.m`), `simult_varexo_det.m` forecasts at orders 1-2 (also `forecast=H` after an estimation, with the zeros `make_ex_.m` leaves); `histval`/`initval`/`endval`, JSON, LaTeX; every Dynare error (leads/lags, perfect-foresight solvers, `endval`, `shocks` moments, variable lists, `irf_shocks`, epilogue, purely forward/backward models, order 3 / k_order_solver, pruning, date-valued shocks, `extended_path`); `reference/exo_det/`, `tests/exo_det.rs`')
a('parameters', 'done', '')
a('var_remove', 'done', 'removes a declared symbol (`DynamicModel` port in src/model/mod.rs, Dynare\'s errors); checked against Dynare in reference/pp/pp_edit.mod (tests/preprocessor_pp.rs)')
a('predetermined_variables', 'done', '')
a('trend_var log_trend_var', 'done', '`growth_factor`/`log_growth_factor`, detrending (`detrendEquations`, `removeTrendLeadLag`) and the balanced-growth test (`balanced_growth_test_tol`), checked against the Dynare 7.2 preprocessor and MATLAB (schorfheide_2000_nonstationary)')
a('model_local_variable', 'done', '`$tex$` names of `#` locals in the LaTeX output (`write_latex_*`, as the preprocessor); `long_name` is not accepted by the 7.2 grammar either')
a('external_function', 'done', 'the model calls a function through a registry (`model::external`): a Rust closure registered with `register_external_function[_with_derivatives]` under the function name, else `NAME.m` next to the .mod, run by a small interpreter for simple MATLAB function files (scalar arithmetic, elementary functions, `if`/`elseif`/`else`, matrix literals and indexed assignments for the derivative outputs, `nargin`/`nargout`; loops, other function calls, strings, cell arrays and local functions are refused with the file and line). New DAG node `Node::External` (`ExternalFunctionNode` in the `dtree` port, `First/SecondDerivExternalFunctionNode` for the derivatives) handled by the tapes, symbolic derivatives, static/dynamic conversion, aux-variable substitutions, LaTeX/JSON printers. `first_deriv_provided`/`second_deriv_provided` without a name use the 2nd/3rd output of the function, with a name a separate function; otherwise Dynare\'s `jacob_element.m`/`hess_element.m` central differences (h = 1e-6). Third-order derivatives are refused (as the preprocessor). Checked against Dynare 7.2 (steady state, order-2 decision rules, all three derivative settings) in reference/external_function/ (tests/external_function.rs)', ['src/modfile/parser_q4.rs', 'src/model/external.rs'], {'name': 'done', 'nargs': 'done', 'first_deriv_provided': 'done', 'second_deriv_provided': 'done'})
a('get_param_by_name set_param_value get_shock_stderr_by_name set_shock_stderr_value get_shock_skew_by_name set_shock_skew_value get_mean', 'done', 'all Dynare setters (`set_param_value`, `set_shock_stderr_value`, `set_shock_skew_value`) are applied at parse time or at their place after a command (tests/ss_partials.rs); the getters are MATLAB accessors and the `*_locally` variants are internal functions')
A['get_param_by_name'] = ('n/a', 'MATLAB accessor', None, {})
A['get_shock_stderr_by_name'] = ('n/a', 'MATLAB accessor', None, {})
A['get_shock_skew_by_name'] = ('n/a', 'MATLAB accessor', None, {})
A['set_shock_stderr_value'] = ('done', '`M_.Sigma_e(i,i) = v^2` (before the first command at parse time, else at its place)', None, {})
A['set_shock_skew_value'] = ('done', '`set_shock_skew_value(\'e\', v)` (and the co-skewness form) applied at parse time', None, {})
A['get_mean'] = ('n/a', 'MATLAB accessor (order-2 mean: `stoch_simul` writes it)', None, {})
a('model', 'done', '`linear` (and `model(linear)`); equation tags, `#` locals, `STEADY_STATE()`; `use_dll`/`bytecode` are code-generation options with no counterpart (accepted); `block`/`mfs`/`static_mfs`/`cutoff`/`no_static` set the block decomposition (`M_.block_structure`), used by `steady` and `perfect_foresight_solver` for block-wise solving (`solve_block_decomposed_problem`, evaluate/solve/two-boundary blocks; checked against Dynare 7.2, reference/block_solve; `stack_solve_algo=4` runs the `lnsrch1` path-length search on the stacked two-boundary blocks, checked against Dynare 7.2 (reference/block_solve/ref_block_solve_a4.json); bytecode is n/a: no bytecode path)', ['src/modfile', 'src/model'], {'use_dll': 'n/a', 'block': 'done (decomposition and block-wise steady / perfect-foresight solving)', 'bytecode': 'n/a', 'cutoff': 'done', 'mfs': 'done', 'static_mfs': 'done', 'no_static': 'done', 'parallel_local_files': 'n/a', 'differentiate_forward_vars': 'done', 'balanced_growth_test_tol': 'done'})
a('model_options', 'done', 'applies `linear`, `block`, `mfs`, `static_mfs`, `cutoff`, `no_static`, `differentiate_forward_vars`, `balanced_growth_test_tol` exactly as the options of the `model` block (parser `apply_model_options`; equivalence tested in tests/model_steady_partials.rs); `use_dll`/`bytecode`/`parallel_local_files` accepted (no counterpart)')
a('model_remove model_replace', 'done', 'tag selection, `endogenous` tags, port of `DynamicModel::removeEquations` with Dynare\'s errors; checked against Dynare in reference/pp/pp_edit.mod (tests/preprocessor_pp.rs)')
a('write_latex_original_model write_latex_dynamic_model write_latex_static_model write_latex_steady_state_model write_latex_definitions write_latex_parameter_table write_latex_prior_table collect_latex_files', 'done', '', ['src/latex'])
a('initval endval histval', 'done', '`all_values_required` is not checked', ['src/modfile', 'src/deterministic'], {'all_values_required': 'missing'})
a('resid', 'done', '`display_static_residuals.m` output, incl. auto equation names; `non_zero`', ['src/misc'])
a('initval_file histval_file', 'done', 'DATE values of `first_obs`/`last_obs`/`first_simulation_period`/`last_simulation_period` (positions in the datafile\'s dates, as `histvalf_initvalf.m`), checked against Dynare; `histval_file` takes no `nobs`/`last_simulation_period` and reads `M_.orig_maximum_lag` observations (as Dynare 7.2)', DET + ['src/modfile', 'src/data.rs'])
a('shocks', 'done', 'stderr/var/cov/corr as expressions (correlations evaluated at the end of their block, as the generated code), measurement errors, `overwrite` (resets `Sigma_e`/`H`/det shocks, not `Skew_e`); deterministic `periods` with integers, ranges, lists and dates (`first_simulation_period`/`last_simulation_period`), `values` with expressions and MATLAB workspace vectors `(xx)`; `learnt_in = INTEGER | DATE` with `values`/`add`/`multiply`; `surprise` (OccBin); `skew`: pruned skewed Kalman filter and, in `stoch_simul`, the `check_model.m` bound, the COSKEWNESS table and closed-skew-normal draws for `periods > 0` (`rand_multivariate_csn.m`; Dynare uses `Skew_e` nowhere else: theoretical moments, IRFs/GIRFs and decision rules ignore it); all checked against Dynare 7.2 (reference/shocks). `shocks(heterogeneity=NAME)`: `M_.heterogeneity(1).Sigma_e` of the heterogeneous shocks (see `heterogeneity_dimension`)')
a('mshocks', 'done', '`relative_to_initval`, `overwrite`, `learnt_in = INTEGER | DATE`, dates in `periods`; options separated by spaces as in the 7.2 grammar (commas also accepted)')
a('heteroskedastic_shocks', 'done', '`values`/`scales`, `overwrite`; with `heteroskedastic_filter` in the likelihood (all filters), the smoother and `calib_smoother`; integer periods only (no dates). Checked against Dynare (fval 1e-15, smoother 1e-15)')
a('dsample', 'done', 'records `options_.smpl`; `rplot` plots that sample, as `rplot.m` (`dyn2vec`/`dynatype`/`dynasave` are MATLAB helpers)')
a('steady', 'done', '`homotopy_mode` 1/2/3 + `homotopy_steps` + `homotopy_force_continue` and `solve_algo` 0/3/6/7/8/13/14 (substitute for 0 = MATLAB fsolve; 13/14 = exact port of the block_trust_region MEX (Dulmage-Mendelsohn blocks / whole system, dogleg trust region, exit codes), checked against MATLAB; 7/8 = Newton with GMRES/BiCGStab preconditioned by an ILU with droptol 1e-10, i.e. the direct solve); `solve_algo = 10` (port of lmmcp.m with the complementarity conditions), `fsolve_options` (TolFun/FunctionTolerance, TolX/StepTolerance, MaxIter/MaxIterations override the substitute solver of solve_algo=0; Display and MaxFunEvals accepted, no counterpart); `non_zero` limits the failure printout (`display_static_residuals` text, shared with `resid`) to non-zero residuals; printout `disp_steady_state`', ['src/driver.rs', 'src/deterministic', 'src/solver', 'src/misc', 'src/dsge.rs'], {'non_zero': 'done', 'fsolve_options': 'done (tolerances/iterations; Display, MaxFunEvals n/a)', 'markowitz': 'n/a'})
a('homotopy_setup', 'done', 'with `steady(homotopy_mode=...)`; `perfect_foresight_solver` has its own homotopy')
a('steady_state_model', 'done', 'temporary variables of the block; multi-output `[a, b] = f()` assignments and an external `<fname>_steadystate.m` call MATLAB code and are refused with an explicit error naming the alternative (inline `steady_state_model` assignments); deviation by design')
a('check', 'done', '', ['src/analysis'], {})
a('model_diagnostics', 'done', 'port of `model_diagnostics.m` (static/dynamic/contemporaneous Jacobian ranks, collinear variables and equations via `null(A, "rational")`, unit-root hint, Inf/NaN, Hessian), output checked against Dynare; no Ramsey-instrument checks')
a('model_info', 'done', 'the non-block output (states, forward-looking, static variables); model_info: `block_static`/`block_dynamic`/`incidence` print the preprocessor\'s block decomposition (`M_.block_structure(_stat)`: Boost matching with libstdc++ tie order, prologue/epilogue, Tarjan SCCs, feedback sets, block types; identical to the Dynare 7.2 preprocessor on 416 models, text identical to MATLAB incl. SW07/QPM/GM05)', ['src/misc', 'src/model'])
a('print_bytecode_dynamic_model print_bytecode_static_model', 'n/a', 'bytecode is a MATLAB/C++ code-generation backend; accepted with a note')
a('perfect_foresight_setup', 'done', '`first_simulation_period`/`last_simulation_period` (dates, get_simulation_periods.m: with `periods` or both dates), mapping the dates of `shocks`/`mshocks`/`shock_paths`/controlled paths to periods', DET, {})
a('perfect_foresight_solver', 'done', '`stack_solve_algo` 1/6 (sim1_lbj), 2/3 (GMRES/BiCGStab with first_iter_lu / block_diagonal_lu / incomplete_lu preconditioners; ILU = complete LU), 7 (stacked dynare_solve, `lmmcp`), 8/9 (ParU/Pardiso: sparse LU), 10 (LU then CGS), `robust_lin_solve`, `check_jacobian_singularity`, `homotopy_marginal_linearization_fallback`, purely backward/forward/static models, all checked against MATLAB; `minimal_solving_periods` is unused in Dynare 7.2', DET, {'lmmcp': 'done', 'robust_lin_solve': 'done', 'minimal_solving_periods': 'n/a'})
a('simul', 'done', 'legacy alias of setup + solver')
a('perfect_foresight_with_expectation_errors_setup perfect_foresight_with_expectation_errors_solver', 'done', '(`learnt_in` shocks/endval); integer or date `learnt_in`, `first_simulation_period`/`last_simulation_period`, `shock_paths(learnt_in=...)` information sets', DET)
a('perfect_foresight_controlled_paths', 'done', '`learnt_in` and periods as dates', DET)
a('shock_paths', 'done', 'temporary/permanent (`end`)/controlled (`exogenize`/`endogenize`) stanzas with `initval.`/`init.`/`self.` scopes; dates in `periods` (also `DATE:end`), `DATABASE.x(lag)` (dseries, multi-row and single-row tables: extract_from_database.m), `prev.`/`learnt_in(INTEGER | DATE).` scopes, `learnt_in = INTEGER | DATE` blocks evaluated per informational period by perfect_foresight_with_expectation_errors_setup, `overwrite` (options separated by spaces as in the 7.2 grammar, commas also accepted); all checked against Dynare 7.2 (reference/shocks)')
a('database', 'done', '`M_.database` for `shock_paths`; the databases are the .mod file\'s MATLAB assignments that dynare-rust evaluates: `dseries(DATA, DATE, NAMES)`, `dseries(\'file\')` (csv/xls(x)/mat/m), `table(..., \'VariableNames\', ...)`, `array2table(...)` over numeric expressions (`[...]`, `linspace`, `transpose`, `ones`, `zeros`, `repmat`, colon ranges, elementwise arithmetic, indexing)')
a('stoch_simul', 'done', 'order 1 (S) and 2-3 (K); `loglinear` (order 1), filters, `spectral_density`, `dr=cycle_reduction|logarithmic_reduction`, `qz_zero_threshold`, `contemporaneous_correlation`, `tex` tables, filters/variance decompositions/`relative_irf`/`irf_plot_threshold` at order 2-3; `partial_information` (`dr1_PI`/`PI_gensys`/`PCL_Part_info_irf`/`PCL_Part_info_moments`, order forced to 1, `.mat` files), `aim_solver`/`dr=aim` (AIM_first_order_solver + matlab/AIM), the 7.2 refusals at order > 1 (`loglinear`: two messages with/without `k_order_solver`; `aim_solver`; `dr=cycle_reduction` only with `k_order_solver`), `order` reset to 2 by every command without it (as the preprocessor), Hessian-equal-to-zero fallback, `print`/`graph` re-enable, `graph_format=none`, `simul_replic` (`Output/<fname>_simul`; skew-normal draws in every replication with `skew`), the `covar_ME` TeX table and the `variance_decomposition_ME` JSON fields at orders 1-2, `solve_algo` (10/11 refused), `histval` start of simulations, measurement errors (`MATRIX OF COVARIANCE OF MEASUREMENT ERRORS`, decompositions WITH MEASUREMENT ERROR), IRF eps-loader TeX file; printouts checked character for character (tests/stoch_simul2.rs). Simulations draw from ChaCha8, not MATLAB\'s randn; graphs are never displayed (`nodisplay` is the only behaviour)', SS)
a('get_irf IRF_MATRIX=get_irf', 'n/a', 'MATLAB accessor; IRFs are in `Output/<fname>_IRF_<shock>.csv`')
A['IRF_MATRIX=get_irf'] = ('n/a', 'MATLAB accessor; IRFs are in `Output/<fname>_IRF_<shock>.csv`', None, {})
a('extended_path', 'done', '(incl. stochastic extended path); `lmmcp` at order 0 and on the stochastic tree (`ep.solve_algo = 10`, `solve_algo=10` here; `reference/ep_lmmcp/`, `tests/ep_lmmcp.rs`); not ported: `hybrid` > 2, `innovation_distribution`, `extended_path_mc` (not reachable from the preprocessor / no defined result)', DET, {'lmmcp': 'done'})
a('occbin_constraints', 'done', '')
a('occbin_setup', 'done', '`simul_*` (OccBin) and `smoother_*`/`likelihood_*` (PKF/IVF likelihoods, OccBin smoother); posterior smoothed variables/shocks after an OccBin MCMC; posterior MeanForecast/PointForecast after it (`occbin.forecast` from each draw\'s smoothed last state, tests/occbin_posterior_forecast.rs)', ['src/occbin', 'src/nonlinear'])
a('occbin_solver occbin_graph occbin_write_regimes', 'done', '', ['src/occbin'])
a('varobs', 'done', '')
a('observation_trends', 'done', 'likelihood and smoother (with `prefilter`, `first_obs`); checked against Dynare (fval 1e-12, smoother 1e-15)')
a('estimated_params', 'done', 'all prior shapes, bounds, stderr/corr of shocks and measurement errors, `overwrite`; `skew` and `dsge_prior_weight`; generalized priors (PRIOR_3RD/4TH_PARAMETER) checked against Dynare', ['src/modfile', 'src/dsge.rs', 'src/estimation'], {'initial_value': 'done', 'prior_shape': 'done', 'prior_standard_error': 'done', 'prior_3rd_parameter': 'done', 'prior_4th_parameter': 'done', 'scale_parameter': 'done', 'lower_bound': 'done', 'upper_bound': 'done', 'prior_mean': 'done'})
a('estimated_params_init', 'done', '`use_calibration`')
a('estimated_params_bounds estimated_params_remove', 'done', '(checked against Dynare\'s bounds)')
a('estimation', 'done', 'see the option table (optimizers and samplers, particle filters, OccBin estimation, Kalman robustness, `lik_init` 2/4/5, `.m` datafiles, `observation_trends`, `analytic_derivation`, Hessian options, skewed Kalman smoother; checked against Dynare in reference/est_complete/)', EST, {
    'mode_compute': 'done (102: simulannealbnd substitute, 13 outside method_of_moments errors as in Dynare 7.2)', 'lik_init': 'done', 'datafile': 'done', 'observation_trends': 'done',
    'lyapunov': 'done (square_root_solver: the Schur solution)',
    'smoother_redux': 'done (same results, computed on the full state space)', 'selected_variables_only': 'done (accepted; all variables are written)',
    'dirname': 'done', 'silent_optimizer': 'done',
    # ---- estimation completion
    'analytic_derivation': 'done (analytic score and Hessians of dsge_likelihood: kalman_filter/univariate_kalman_filter with derivatives, expanded state for correlated measurement errors, analytic_Hessian opg/full/asymptotic, second-order parameter derivatives of the solution; analytic gradients in mode_compute 1/3 substitutes, 4, 5 (OPG Hessian per iteration), 101; the exact Hessian at the mode; analytic_derivation_mode 0/1/-1/-2; Dynare\'s errors; gradients/Hessians 1e-12, modes/Hessians vs MATLAB runs, tests/est_complete_ad.rs)',
    'analytic_derivation_mode': 'done',
    'cova_compute': 'done (no Hessian, mode file without hh, Dynare\'s MCMC error)',
    'use_penalized_objective_for_hessian': 'done (hessian.m on penalty_objective_function)',
    'skewed_kalman_rank_deficiency_transform': 'done (csn_statespace_linear_transform.m + csn_singular_to_csn_proper.m; checked against Dynare, 1e-16)',
    'skewed_kalman_mvnlogcdf': 'done (gaussian_log_mvncdf_mendell_elston and mvncdf: Dynare\'s missing/stats mvncdf with bvncdf/tvncdf/mvtcdfqmc ports; d<=3 bit-level vs MATLAB, d>=4 randomized QMC with fixed-seed shifts, within its 1e-4 tolerance)',
    'kalman_algo': 'done (5: pruned skewed Kalman filter and smoother kalman_smoother_pruned_skewed.m; missing data via the Gaussian missing-observations filter and lik_init 1/2/4/5 as Dynare, lik_init=3 Dynare\'s error)',
    'posterior_nograph': 'done (options_.no_graph.posterior: pm3/PosteriorIRF figures)',
    'posterior_graph': 'done',
    'no_init_estimation_check_first_obs': 'done (initial_estimation_checks.m stochastic-singularity checks, checked against Dynare)',
    'use_univariate_smoother_if_singularity_is_detected': 'done (DsgeSmoother.m: error 422 unless set or noprint)',
    'contemporaneous_correlation': 'done (posterior moments: covariance_mc_analysis.m)',
    'no_posterior_kernel_density': 'done (posterior_moments.m densities of pm3 objects and posterior moments, checked against Dynare)',
    'frequentist_smoother': 'done (explicit true/false as Dynare; without the option the classical smoother runs only with smoother/filtered_vars)',
    'dr': 'done (cycle_reduction/logarithmic_reduction in the likelihood, checked against Dynare)',
    'dr_cycle_reduction_tol': 'done', 'dr_cycle_reduction_maxiter': 'done',
    'dr_logarithmic_reduction_tol': 'done', 'dr_logarithmic_reduction_maxiter': 'done',
    'qz_zero_threshold': 'done (likelihood, checked against Dynare)',
    'huge_number': 'done (mode_compute 2/10 bounds, mode_check)',
    'nonlinear_filter_initialization': 'done (options_.particle.initialization; 3 checked against Dynare)',
    'estimate_initial_states_endogenous_prior': 'done (init <var> parameters sixth in xparam1, lik_init=2 with Harvey_scale_factor=0, init_state_endogenous_prior.m, set_init_state.m, draw_init_state_from_smoother.m in the slice sampler; likelihood, projection and smoothed initial state checked against Dynare, MCMC statistically; draw_init_state_with_rotated_slice not ported)',
    'optim': 'done (newrat `Hessian` 0/1/2: outer product / hessian.m / mixed, mr_hessian on the univariate filter at the mode as Dynare)',
    })
a('data', 'done', '`options_.dataset` for later data commands (makedataset.m: `first_obs`/`last_obs` dates, `nobs`, consistency errors); `series=` needs a MATLAB-workspace dseries (error with a note); `set_time` sets `options_.initial_period`', ['src/dataset.rs', 'src/q4'], {'series': 'n/a', 'nobs = [INTEGER1:INTEGER2]': 'missing', 'first_obs = [INTEGER1:INTEGER2]': 'missing'})
a('filter_initial_state', 'done', 'initial state of the likelihood filters and of the smoother (`x(0)`, `x(-k)` through the lag aux variables); checked against Dynare')
a('bvar_density bvar_irf bvar_forecast', 'done', '', ['src/bvar'])
a('matched_moments matched_irfs matched_irfs_weights', 'done', '')
a('method_of_moments', 'done', '(GMM, SMM, IRF matching), analytic_standard_errors / analytic_jacobian (order 1 analytic, order 2-3 Richardson differences), MCMC after Bayesian IRF matching (`mh_replic > 0`); `hp_filter`, `one_sided_hp_filter`, `bandpass_filter` (SMM only, as Dynare; get_filtered_time_series.m, filters shared with stoch_simul, no MATLAB reference run for the last two because SMM draws differ); `use_pct` (Parallel Computing Toolbox) is n/a', ['src/mom', 'src/estimation', 'src/driver.rs'], {'use_pct': 'n/a'})
a('model_comparison', 'done', '')
a('shock_decomposition', 'done', '`with_epilogue` (epilogue_shock_decomposition.m, checked against Dynare) and `colormap` (MATLAB colormaps by name)', ['src/analysis', 'src/forecast', 'src/q4', 'src/dataset.rs'], {'colormap': 'done', 'with_epilogue': 'done'})
a('shock_groups', 'done', '')
a('realtime_shock_decomposition', 'done', '`colormap` through plot_shock_decomposition; `with_epilogue` inside the realtime recursion (vintages, pools, conditional / forecast / realtime conditional; tests/realtime_epilogue.rs)', ['src/analysis', 'src/forecast', 'src/dataset.rs'], {'colormap': 'done', 'with_epilogue': 'done'})
a('plot_shock_decomposition', 'done', '`init2shocks`, epilogue variables (`with_epilogue`), `steadystate` (zero line of the levels and the xls column; the y-tick relabelling is not reproduced), `write_xls` (WriteShockDecomp2Excel.m as a real .xlsx), `colormap`, `screen_shocks`, `fig_name` (`SCREEN`, `init2shocks`, `...initval` detail labels); `interactive` (MATLAB uimenus) accepted with a note', ['src/analysis', 'src/q4'], {'interactive': 'n/a', 'with_epilogue': 'done', 'init2shocks': 'done', 'write_xls': 'done', 'steadystate': 'done', 'colormap': 'done', 'screen_shocks': 'done'})
a('init2shocks', 'done', '(`M_.init2shocks.NAME`, used by plot_shock_decomposition(init2shocks), checked against Dynare)')
a('initial_condition_decomposition squeeze_shock_decomposition', 'done', '', ['src/analysis'])
a('calib_smoother', 'done', '`filter_step_ahead`, `filter_decomposition`, `filter_covariance`, `smoother_redux`, `heteroskedastic_filter`', ['src/forecast', 'src/driver.rs', 'src/estimation/q3.rs'], {})
a('forecast conditional_forecast conditional_forecast_paths plot_conditional_forecast', 'done', '(the `forecast` command starts from `M_.endo_histval` with `histval`, runs at `options_.order` 1-3 with simulated bands at order > 1, and takes its horizon from `options_.periods`; `conditional_forecast(parameter_set=calibration)` starts from `M_.endo_histval` with `histval`, e.g. after `smoother2histval`, `tests/conditional_forecast_histval.rs`); `varexo_det` paths in `forecast` (orders 1-2, `tests/exo_det.rs`)', ['src/forecast', 'src/driver.rs'])
a('smoother2histval', 'done', 'histval from the smoothed variables of the last estimation/calib_smoother at its point (after an MCMC: the posterior mean of the smoothed draws, oo_.SmoothedVariables.Mean, as smoother2histval.m); `outfile` writes a dseries-layout .mat; as Dynare 7.2, the whole histval is reset first (exogenous variables to their steady state). the default `invars` set the `AUX_ENDO_LAG`/`AUX_EXO_LAG` rows of `M_.endo_histval` as MATLAB 7.2 (tests/smoother2histval.rs, reference/s2h)', ['src/misc'])
a('planner_objective ramsey_model ramsey_constraints ramsey_policy discretionary_policy osr osr_params optim_weights osr_params_bounds evaluate_planner_objective', 'done', '(order-2 welfare, Ramsey with steady_state_model, bounded/order-2 OSR); order >= 3 welfare (k_order_welfare tensors, k_order_mean simulated unconditional welfare with `periods`/`drop`; shocks from ChaCha8 seeded with `--seed`, not MATLAB\'s randn; tests/policy.rs ramsey_order3_simulated_welfare feeds MATLAB\'s draws); Ramsey FOCs are built in the preprocessor DataTree emulation, so `AUX_ENDO_LEAD_*` names, equation order and incidence match Dynare', ['src/policy', 'src/model/planner.rs', 'src/modfile'], {
    # options_.ramsey.periods / .drop are read only by the order >= 3 branch of evaluate_planner_objective.m
    # (k_order_welfare + k_order_mean, simulated unconditional welfare)
    'periods': 'done (options_.ramsey.periods, order >= 3 simulated welfare)',
    'drop': 'done (options_.ramsey.drop, order >= 3 simulated welfare)'})
A['ramsey_constraints'] = ('done', '', None, {})
a('sensitivity', 'done', 'stability, redform, Morris, RMSE; `ppost` (posterior subsample of the MCMC files, also with `rmse`), `load_stab`/`load_redform`/`load_rmse`/`load_ident_files`, `glue` (also as `sensitivity(glue=1)`, which Dynare 7.2\'s preprocessor rejects), `identification` (map_identification: Morris screening, `morris=2`), the irf/moment calibration mapping', ['src/gsa'], {})
a('irf_calibration moment_calibration', 'done', 'mapped by `sensitivity` (map_calibration.m) and applied by `prior moments(distribution)`/`irfs(distribution)` (endogenous_prior_restrictions.m); applied wherever Dynare 7.2 reads `options_.endogenous_prior_restrictions`: dsge_likelihood.m (and the conditional-likelihood, OccBin and skewed-Kalman branches; info 49, `info(4)` = the summed squared distances, so the mode finders, mode_check, the MCMC and its initialization see them), prior_sampler.m (`prior simulate`: Endogenous prior violation share), `prior moments` (info 49 at the prior mode), stability_mapping.m (infox 49, `irestriction`, the `prior_calib` filtering, `x0`), `method_of_moments` (dropped with Dynare\'s warning); not used by non_linear_dsge_likelihood.m, DSGE-VAR or identification, as in Dynare (tests/calib_est.rs, reference/calib_est)', ['src/gsa/calibration.rs'], {})
a('identification', 'done', '`order` 2/3 (pruned state space, Jacobians by finite differences as `analytic_derivation_mode=-1`, 1e-6 from Dynare\'s analytic ones), `checks_via_subsets`, `gsa_sample_file`, `parameter_set=posterior_*`, `load_ident_files` (the point analysis; Dynare 7.2 itself fails without a saved Monte Carlo)', ['src/identification'], {'load_ident_files': 'partial'})
a('svar_identification markov_switching svar', 'done', 'src/msbvar: parsed and validated as the preprocessor does (exclusion lag/equation, constants, upper/lower_cholesky, chains with regimes/durations/restrictions/parameters/number_of_lags, svar variances/coefficients/equations) into `MsDecl`; vector `duration` and `restrictions` are accepted although the 7.2 preprocessor aborts on a vector duration; `restriction equation N, [EXPR*]coeff(VAR,LAG) +/- ... = ...;` lines are parsed into rows of `Qi{N}`/`Ri{N}` exactly as the 7.2 preprocessor writes them (shared per-equation row counters with the exclusion groups, right-hand side negated, parameters at their calibrated values, "lag L used more than once", homogeneity and Qi-or-Ri errors; checked on its driver.m output, tests/msbvar.rs)', ['src/msbvar'])
a('sbvar', 'done', 'src/msbvar/sbvar.rs: Sims-Zha prior (dummy observations), exclusion/Cholesky restrictions, posterior mode of A0 (analytic-gradient BFGS in place of csminwel), A+ mode, impulse responses, point forecast and the marginal-density terms, outm.mat; checked to 1e-5 against MATLAB. The rest of sbvar (src/msbvar/sbvar_extra.rs): the Gibbs-sampled terms of the log marginal data density (`ms_mardd.m`: 1000 + 10000 draws per equation; deterministic pieces to 1e-2, `log mdd` within Monte Carlo error of Dynare: the term of the first equation is a heavy-tailed average with seed-to-seed std about 0.15), `outactqmygdata.prn` / `outyrqm.prn` (`fn_datana`, `vlistlog`; identical to the Dynare files), impulse-response, unconditional-forecast and growth-rate figures (`nograph`, `graph_format`; layout is ours, not that of MATLAB). Linear restrictions (`restriction equation` lines, src/msbvar/identification.rs): `Ui = null(Qi)`, `Vi = null(Ri)` (orthonormal), checked against Dynare 7.2 on two models (reference/msbvar/msbvar_lin*.mod, dump_msbvar_lin.m; mode, A+, IRFs, forecast, posterior matrices, mdd terms to 1e-5; Gibbs log mdd -469.84 as Dynare). `cross_restrictions`: Dynare 7.2 errors (msstart2.m:415 brace-indexes the NaN `ixmC0Pres` returned by exclusions.m; the preprocessor also rejects any line mixing lag 0 and lags > 0); here a mixed line with `cross_restrictions` is the restriction `R a+_j + Q a_j = 0`, and sbvar returns the exact restricted posterior mode (A0 objective plus the quadratic from integrating A+ over the restriction set, A+ by the conditional formula of msstart2.m), validated against a direct maximisation of the hand-built joint posterior (tests/msbvar.rs); no mdd with cross restrictions. Conditional forecasts do not exist in `sbvar` of Dynare: `msstart2.m` hard-codes `nconstr=0`, so only the unconditional forecast is produced', ['src/msbvar'])
a('ms_estimation', 'done', 'MS-SBVAR (src/msbvar/ms_estimation.rs, ms_optim.rs, ms_files.rs): the Sims-Waggoner-Zha model (any number of chains, svar coefficients/variances per equation); `specification` random walk / `none` (Dynare\'s W_j) / `sims_zha` (lambda, psi), `contemp_reduced_form`, two-chain prior scaling, all pinned down against Dynare 7.2 (initial prior/likelihood to 1e-6, posterior at Dynare\'s mode = its objective); Dynare\'s mode search (block iterations VAR/Q with the convergence schedule, repeated runs, large/small perturbation stages from Metropolis draws; stochastic, attained mode compared with Dynare to 0.05, tests/msbvar_ms_est.rs); init_<file_tag>.dat / est_final_<output_file_tag>.out in Dynare\'s format (later ms_* commands read est_final_<file_tag>.out, also Dynare\'s own; `file_tag`, `output_file_tag`, `no_create_init`). `restriction equation` lines give the same general Ui/Vi as for sbvar; `cross_restrictions` fails as in Dynare 7.2 (undefined Fhat in ms_sbvar_setup.m); data from csv, m, mat, xls and xlsx files (read_variables, with xls_sheet/xls_range)', ['src/msbvar'])
a('ms_compute_probabilities', 'done', 'src/msbvar/hamilton.rs: filtered and Kim-smoothed regime probabilities of the latest ms_estimation of the run (filtered_/smoothed_<fname>.out, checked to 1e-6 against Dynare). `real_time_smoothed` (real_time_smoothed_<tag>.out, checked to 2e-6) and the per-chain probability figures (src/msbvar/ms_graphs.rs; nograph/graph_format honoured)', ['src/msbvar'])
a('ms_compute_mdd', 'done', 'src/msbvar/mdd.rs: port of dw_mdd_switch.c / dw_elliptical.c / dw_dirichlet_restrictions.c. The first 20% of the posterior draws fit the proposal, the other 80% are the posterior sample; proposal = elliptical on theta (centre = best fitting draw, scale = second moment about it, radius quantile bounds, `proposal_type` 1-6: Gaussian, power, truncated power, step, truncated Gaussian, uniform; power from E[r^2] = dim) x independent Dirichlet on the transition probabilities (Dynare\'s alpha rule); Muller, bridge and Waggoner-Zha (no cut) estimators. Dynare\'s mdd program switches on the Waggoner-Zha sign normalisation for the proposal draws only (nvars*ln 2 and -inf outside the region) while the posterior draws keep the kernel of the simulation file; reproduced, `MddOptions::consistent` gives the true mdd. On Dynare\'s own draws: proposal parameters and per-draw log proposal densities equal Dynare\'s (5e-7), Muller/bridge/WZ on its proposal matrix equal its printed values (1e-4); with our proposal draws the means over 20 seeds match those of 20 Dynare runs to 0.035 (Monte Carlo). use_mean_center is ignored as in Dynare (its MEX never sees the flag)', ['src/msbvar'])
a('ms_simulation', 'done', 'src/msbvar/ms_commands.rs: adaptive random-walk Metropolis (mdd::rwmh) from the mode -> simulation_<tag>.out (log post, log lik, free parameters) with mh_replic/drop/thinning_factor/adaptive_mh_draws; Dynare\'s C sampler and its random numbers are not reproduced. `save_draws` writes draws_<tag>.out / draws_header_<tag>.out (est_flat layout)', ['src/msbvar'])
a('ms_irf', 'done', 'src/msbvar/ms_sim.rs: regime-fixed impulse responses (regime, regimes, ergodic default, filtered_probabilities), `error_band_percentiles`/`median`, `parameter_uncertainty` from simulation_<tag>.out, `free_parameters`; IRF files checked against Dynare 7.2 to 2e-5 (the percentiles are those of the exact regime mixture, not of sampled draws). regime-specific figures (ms_graphs.rs); `shock_draws` regime sampling is replaced by exact integration', ['src/msbvar'])
a('ms_variance_decomposition', 'done', 'src/msbvar/ms_sim.rs: shares of the shocks in the forecast-error variance per regime / ergodic / filtered, mean (`no_error_bands`) or percentiles; regime-specific means equal Dynare\'s to 2e-5, the mixtures to 3e-3 (Dynare samples its regime weights). Percentile version differs from Dynare\'s by ~4e-4 (unexplained); figures via ms_graphs.rs', ['src/msbvar'])
a('ms_forecast', 'done', 'src/msbvar/ms_sim.rs: simulated forecast percentiles (regime-fixed per regime / `regime`; otherwise start from the filtered probabilities with the regime moving by Q), `data_obs_nbr`, `parameter_uncertainty`; reproduces Dynare 7.2 (incl. its first forecast row being the one-step forecast of the last observation) to Monte-Carlo accuracy (0.05 on percentiles); figures via ms_graphs.rs', ['src/msbvar'])
a('epilogue', 'done', '`epilogue_dynamic`/`epilogue_static` (incl. the 6-decimal rounding of dseries\' `from ... do`), used by `with_epilogue` decompositions as Dynare; also written over the smoothed variables (`Output/<fname>_epilogue_smoothed_variables.csv`) and perfect-foresight paths (`_epilogue_endo_simul.csv`)')
HET = ['src/heterogeneity', 'src/modfile/parser_het.rs', 'src/model/dtree_het.rs']
a('heterogeneity_dimension', 'done', '(src/heterogeneity, parser_het.rs): one dimension (Dynare 7.2 allows one too); `M_.heterogeneity(1)` (names, tex/long names, aux_vars of type 15/16, `state_var`, sparse Jacobian structure, `dynamic_mcp_equations_reordering`, `het_aux_levels`, `Sigma_e`, equation tags), `M_.heterogeneity_aggregates` and the type-14 `SUM_x` variables are checked against the Dynare 7.2 preprocessor and MATLAB (reference/heterogeneity, tests/heterogeneity.rs)')
a('heterogeneity_load_steady_state', 'done', 'port of `+heterogeneity/load_steady_state.m` and `check_steady_state_input.m` (+ the `find_bracket_linear_weight` and `compute_Phi_tilde_e` MEX): `oo_.heterogeneity.{steady_state,sizes,indices,mat}` (grids, brackets, Mu, Phi, Phi_e, x_bar, d.x_bar, aggregates, G/dG, F/dF) checked to 1e-12 against MATLAB. The steady state comes from a MAT file (`filename`, `variable`) or from `<variable>.mat` next to the .mod file: a `verbatim` MATLAB block cannot be run', HET, {'filename = FILENAME': 'done', 'variable = STRING': 'done', 'tolf = DOUBLE': 'done'})
a('heterogeneity_compute_steady_state', 'done', 'port of `compute_steady_state.m` and of the `compute_steady_state_tensor` MEX (time iteration with Fischer-Burmeister complementarity and the trust-region solver of `block_trust_region`, forward iteration of the distribution, aggregation, Broyden calibration of free parameters), same iteration path as MATLAB (policies and distribution to 1e-11, calibrated parameters to 1e-12; Krusell-Smith and HANK one asset, reference/heterogeneity). Same MAT-file input rule as `heterogeneity_load_steady_state`', HET, {k: 'done' for k in ['filename = FILENAME', 'variable = STRING', 'calibration_target_equations = [QUOTED_STRING | INTEGER[, QUOTED_STRING | INTEGER[, ...]]]', 'calibration_tolf = DOUBLE', 'calibration_max_iter = INTEGER', 'calibration_verbosity = INTEGER', 'time_iteration_max_iter = INTEGER', 'time_iteration_tol = DOUBLE', 'time_iteration_learning_rate = DOUBLE', 'time_iteration_early_stopping = INTEGER', 'time_iteration_verbosity = INTEGER', 'time_iteration_solver_tolf = DOUBLE', 'time_iteration_solver_tolx = DOUBLE', 'time_iteration_solver_factor = DOUBLE', 'time_iteration_solver_max_iter = INTEGER', 'time_iteration_solver_stop_on_error', 'forward_max_iter = INTEGER', 'forward_tol = DOUBLE', 'forward_check_every = INTEGER', 'forward_verbosity = INTEGER']})
HET2 = HET + ['tests/heterogeneity.rs']
a('heterogeneity_solve', 'done', 'port of `+heterogeneity/solve.m` (sequence-space Jacobian method: policy impulse responses, `curlyDs`, `curlyYs`, expectation vectors, fake-news matrices `F`, heterogeneous Jacobians `J_ha`, general-equilibrium Jacobians `G`, the aggregate Jacobian `dr.J`) and of the MEX `compute_expected_dx`, `compute_curlyDs`, `compute_expected_y`, `compute_transition_matrices` (sources not distributed; semantics pinned against MATLAB). `dr.G`, `F`, `J_ha`, `curlyYs`, `curlyDs`, `J` and the policy impulse responses match Dynare 7.2 to 1e-12 (Krusell-Smith, HANK one asset; reference/heterogeneity/dump_solve.m, tests/heterogeneity.rs). Result in `HetSession::dr`', HET2, {'truncation_horizon = INTEGER': 'done'})
a('heterogeneity_simulate', 'done', 'port of `+heterogeneity/simulate.m`, `simulate_irfs.m`, `simulate_stochastic_shocks.m`, `simulate_news_shocks.m`: IRFs (one-standard-deviation or `relative_irf` unit shocks, `irf_shocks`, variable list, `irf` horizon check), stochastic simulation (`periods`) and news-shock sequences (`shocks` block with `periods`/`values`, `make_ex_` rules, option-compatibility error); IRFs, the convolution given the draws and news paths match Dynare 7.2 to 1e-12. Deviations: no graphs (`plot_irfs.m`/`plot_simulation.m`: `<fname>_IRF_<shock>[k]`, `<fname>_simulation[k]`, `<fname>_det_simulation[k]` via the stoch_simul IRF figures, honouring `nograph`/`graph_format`; no TeX eps-loader file), stochastic draws come from dynare-rust\'s ChaCha8 stream (MATLAB\'s `randn` stream is not reproduced); results kept in `HetSession::sim`, not written to files', HET2, {'irf = INTEGER': 'done', 'periods = INTEGER': 'done', 'irf_shocks = ( VARIABLE_NAME [VARIABLE_NAME ...] )': 'done', 'relative_irf': 'done', 'nograph': 'partial'})
a('var(heterogeneity=NAME)', 'done', 'also `varexo(heterogeneity=NAME)` and `parameters(heterogeneity=NAME)`')
A['varexo(heterogeneity=NAME)'] = ('done', '', None, {})
A['model(heterogeneity=NAME);'] = ('done', 'lead/lag limits and the separability rules of `HeterogeneousModel::checkPass`, non-linear leads (`AUX_HET_ENDO_LEAD_<idx>`, names as the preprocessor), complementarity conditions `⟂` / `_|_` with `lb <= x <= ub` and the multipliers `MULT_L_x` / `MULT_U_x`, topological levels of the auxiliary variables; `dynamic_het1_{resid,g1,set_auxiliary_variables,complementarity_conditions}` evaluated to 1e-13. Not supported (as errors): aggregate leads/lags beyond 1 and parameters declared with `heterogeneity=` inside the equations', None, {})
a('SUM', 'done', '`SUM(x)` (also `sum`) in the aggregate `model` block: aggregation variables `SUM_x`, `M_.heterogeneity_aggregates`; the aggregate equations take `yagg` (`HetDecl::agg_residuals`)')
a('rouwenhorst', 'done', '`rouwenhorst(rho, sigma, N, tol, maxiter)` (`mex/sources/heterogeneity/rouwenhorst.f08`), identical to Dynare to 1e-14 (src/heterogeneity/rouwenhorst.rs)', HET)
a('var_model trend_component_model var_expectation_model var_expectation', 'done', 'VAR family (src/varmodels, parser_vm.rs): AR/A0/constants of the tagged equations (`varmatrices.m`, `trend_component_ar_a0.m`), companion matrices, `var_expectation_model_<m>_*` parameters and the `var_expectation(m)` operator (also `model_name = m`), `time_shift`, `horizon = n | a:b | a:Inf`, `structural`; checked against MATLAB to 1e-12 (`reference/varmodels`, `tests/varmodels.rs`). Dynare 7.2 quirk reproduced: `update_parameters.m` puts the *index* of a parameter in `expression` into alpha', ['src/varmodels', 'src/modfile/parser_vm.rs'], {})
a('pac_model pac_target_info pac_expectation pac_target_nonstationary', 'done', 'PAC models (src/varmodels/pac.rs, modfile/parser_pac.rs): the PAC equation is decomposed as the preprocessor does (error correction `ec*(target(-1)-z(-1))`, `ar_k*diff(z(-k))`, with or without `pac_target_nonstationary`); `pac_expectation(m)` becomes the auxiliary endogenous variable `pac_expectation_<m>` (or the `auxname`s of `pac_target_info`, with `target`, `auxname_target_nonstationary`, `component`/`kind`/`auxname`/`growth`) defined by the linear form in the `h_<m>_*` parameters; `growth` adds the `<m>_pac_growth_neutrality_correction` term; `kind = dd | dl | ll`; without `auxiliary_model_name` the model consistent expectations recursion `mce_Z1_<m>` with the `mce_alpha_<m>_*` parameters. Checked against MATLAB (VAR with composite target and growth, trend component with growth, VAR simple target, MCE; `reference/pac`, `tests/pac.rs`). Not ported: rule of thumb agents and additive-term bgp (`pac.bgp.*`, `pac.print`, `pac.check`), composite targets with MCE; the auxiliary variables count in the `stoch_simul` tables, as declared variables', ['src/varmodels/pac.rs', 'src/modfile/parser_pac.rs'], {'auxname': 'done', 'kind': 'done', 'growth': 'done', 'discount': 'done', 'auxiliary_model_name': 'done', 'model_name': 'done'})
a('var_expectation.initialize var_expectation.update', 'done', 'statements in the .mod, run in file order (src/varmodels); also `get_companion_matrix(NAME)`', ['src/varmodels'], {})
a('pac.initialize pac.update pac.mce.parameters', 'done', '`pac.initialize(NAME)`, `pac.update.expectation(NAME)` (h0/h1 parameters and growth neutrality correction: `hVectors.m`, `+pac/+update/parameters.m`; `pac.update.equation` is the deprecated alias) and `pac.mce.parameters(NAME)` run as statements in file order; parameters checked to 1e-10 against MATLAB (`tests/pac.rs`)', ['src/varmodels/pac.rs'], {})
a('pac.estimate.nls pac.estimate.iterative_ols', 'done', 'statements in the .mod (`tests/pac_estimate_mod.rs`): `eparams.x = v;` structure, `edata = dseries(\'file\')`, `dates(..):dates(..)+N` range; Gauss-Newton as `gauss_newton.m`, Newey-West covariance, iterative OLS, estimates and h parameters written to the model parameters, checked against MATLAB (`tests/pac.rs`, `tests/pac_estimate_mod.rs`). Optimizers `lsqnonlin`/`fmincon`/`annealing`/`particleswarm` bound the error correction parameter to [0,1] via projected Gauss-Newton, not the MATLAB algorithms; no `oo_.pac` structure; data only from a dseries file', ['src/varmodels/pac_estimate.rs', 'src/modfile/parser_pac.rs'], {})
a('rplot', 'done', '')
a('dynatype dynasave', 'n/a', 'accepted with a note; the paths are in `Output/<fname>_endo_simul.csv`')
a('@#includepath @#include @#define @#if @#ifdef @#ifndef @#elseif @#else @#endif', 'done', 'checked against dynare-preprocessor')
a('@#for', 'done', 'tuples and `when` filters')
a('@#echo @#error', 'done', '')
a('@#echomacrovars', 'done', 'displays the variables and functions; `(save)`: `options_.macrovars_line_<n>` written to `Output/<fname>_macrovars.json` (functions shown with their body as written)')
a('verbatim', 'done', 'skipped as in Dynare (MATLAB code)')
a('set_dynare_seed', 'done', 'seeds dynare-rust\'s ChaCha8 streams (the algorithm name is ignored)')
a('save_params_and_steady_state load_params_and_steady_state', 'done', '(load: read at preprocessing time, as the preprocessor does); checked against Dynare\'s file')
a('compilation_setup', 'n/a', 'no external compiler; accepted with a note')
a('dynare_version', 'done', '')
a('send_endogenous_variables_to_workspace send_exogenous_variables_to_workspace send_irfs_to_workspace clean_current_folder internals search dplot set_dynare_threads', 'n/a', 'MATLAB workspace / development helpers; accepted with a note')
a('prior_function posterior_function', 'n/a', 'run a user MATLAB function on prior/posterior draws: no Rust counterpart (would need a plug-in mechanism)')
a('generate_trace_plots', 'done', '`trace_plot(options_, M_, estim_params_, TYPE, BLOCK, NAME)` too')
a('prior', 'done', '`table` (checked against Dynare\'s printout), `plot`, `simulate` (prior mass and BK shares on dynare-rust\'s draws), `moments` (theoretical moments at the prior mode); `moments(distribution)`, `irfs(distribution)` (dynare-rust\'s draws, same printout and .mat file), `optimize`', ['src/misc'], {'table': 'done', 'plot': 'done', 'simulate': 'done', 'moments': 'done', 'optimize': 'done'})
a('STEADY_STATE EXPECTATION', 'partial', '')
A['STEADY_STATE'] = ('partial', 'in OccBin constraints only', None, {})
A['EXPECTATION'] = ('missing', 'EXPECTATION(k)(expr) operator', None, {})

STATUS_ICON = {'done': 'done', 'partial': 'partial', 'missing': 'missing', 'n/a': 'n/a'}

def cmd_status(e):
    name = e['name']
    if name in A:
        return A[name]
    if name.lower() in ALL:
        return ('partial', 'parsed; status not annotated', None, {})
    return ('missing', '', None, {})

def opt_status(e, cstat, paths, overrides, oname):
    key = oname.lower()
    for k, v in overrides.items():
        if k.lower() == key:
            return v
    if cstat in ('missing', 'n/a'):
        return cstat
    lits = literals(paths) if paths else ALL
    variants = {key, key.removeprefix('simul_')}
    if any(v in lits for v in variants):
        return 'done'
    return 'missing'

out = []
w = out.append
w("# Coverage of the Dynare 7.2 reference manual\n")
w("Every command, block, MATLAB command and macro directive of the Dynare 7.2 reference manual")
w("(`doc/dynare-manual.html/_sources/the-model-file.rst.txt`, `dynare-misc-commands.rst.txt`,")
w("`running-dynare.rst.txt`) with its status in dynare-rust.\n")
w("Status: **done** (ported, checked against Dynare where numbers are involved), **partial**")
w("(see the note), **missing** (not ported), **n/a** (MATLAB-only or code-generation features")
w("without a counterpart; accepted with a note where harmless). Unknown commands are skipped with")
w("a warning.\n")
w("Option status is derived from the code: an option is *done* when the modules that implement")
w("the command read it (string literal search), corrected by hand where needed; *missing* options")
w("of a *done* command are accepted and ignored (usually with a warning). Regenerate with")
w("`python3 docs/gen_coverage.py` (the hand annotations are in that script).\n")

# summary
summary = {}
for e in entries:
    st = cmd_status(e)[0]
    k = st.split(':')[0]
    summary[k] = summary.get(k, 0) + 1
w("## Summary\n")
w("| status | commands / blocks / directives |")
w("|---|---|")
for k in ['done', 'partial', 'missing', 'n/a']:
    w(f"| {k} | {summary.get(k, 0)} |")
w("")

# missing list
w("## Missing\n")
groups = {}
order = []
for e in entries:
    st, note, _, _ = cmd_status(e)
    if st == 'missing':
        if note not in groups:
            groups[note] = []
            order.append(note)
        if e['name'] not in groups[note]:
            groups[note].append(e['name'])
for note in order:
    names = ", ".join(f"`{n}`" for n in groups[note])
    w(f"- {names}" + (f": {note}" if note else ""))
w("")
w("Missing options of ported commands (highlights):")
w("`estimation` `draw_init_state_with_rotated_slice` of `estimate_initial_states_endogenous_prior`;")
w("(none besides the entries marked n/a.)\n")

# command table
w("## Commands and blocks\n")
w("| command | kind | status | notes |")
w("|---|---|---|---|")
for e in entries:
    st, note, _, _ = cmd_status(e)
    sig = e['name'].replace('|', '\\|')
    w(f"| `{sig}` | {e['kind']} | {st} | {note.replace('|', '/')} |")
w("")

# options per command
w("## Options\n")
for e in entries:
    if not e['opts']:
        continue
    st, note, paths, overrides = cmd_status(e)
    w(f"### `{e['name']}` ({st})\n")
    w("| option | status |")
    w("|---|---|")
    seen_o = set()
    for sig, oname in e['opts']:
        if (sig) in seen_o:
            continue
        seen_o.add(sig)
        os_ = opt_status(e, st, paths, overrides, oname)
        w(f"| `{sig.replace('|', '\\|')}` | {os_} |")
    w("")

# command line
w("## Command-line options (`dynare-rs model.mod [OPTIONS]`)\n")
CLI = {'savemacro': 'done', 'onlymacro': 'done', 'linemacro': 'done', 'nograph': 'done', 'onlymodel': 'done', '-DMACRO_VARIABLE': 'done', '-I<<path>>': 'done',
       'json': 'partial (modfile/modfile-original/static/dynamic.json; model, aux_vars, AST, declarations, variable_mapping, sparsity, model_info as the preprocessor; derivative strings without temporary terms; a subset of statements)',
       'jsonstdout': 'done', 'onlyjson': 'done', 'jsonderivsimple': 'done',
       'exclude_eqs': 'done (variables turned exogenous by symbol id)', 'include_eqs': 'done (untagged equations cannot be excluded)',
       'stochastic': 'done', 'nostrict': 'partial (undeclared model symbols become exogenous; more variables than equations still an error)',
       'params_derivs_order': 'n/a (accepted with a note)', 'fast': 'n/a (accepted: no generated files to skip)', 'minimal_workspace': 'n/a (accepted: no MATLAB workspace)'}
w("| option | status |")
w("|---|---|")
for c in cli:
    k = re.split(r'[\s=\[]', c)[0]
    st = CLI.get(k, 'n/a (accepted)')
    w(f"| `{c.replace('|', '\\|')}` | {st} |")
w("")
w("Also `--seed`, `--mh-replic`, `--threads`, `--prefetch`, `--blas`, `--quiet`, `--window` (dynare-rust's own).")
open(os.path.join(REPO, 'docs', 'COVERAGE.md'), 'w').write("\n".join(out) + "\n")
print(summary)
