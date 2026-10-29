# Dynare 7.2 test-suite triage of dynare-rs

Entries: 708 (from `mod_and_m_tests` in meson.build). Regenerate with scripts/sweep (see README.md).

| category | entries |
|---|---|
| (a) both succeed, numbers agree | 165 |
| (a0) both succeed, no comparable output mapped | 200 |
| (b) both succeed, numbers differ | 0 |
| (p) optimizer endpoint: same log posterior at the mode, the mode differs | 17 |
| (s) stochastic-only differences (RNG driven) | 74 |
| (c) Dynare succeeds, dynare-rs fails | 0 |
| (c2) dynare-rs fails after skipping MATLAB statements (error not on a MATLAB line) | 0 |
| (d) dynare-rs succeeds where Dynare fails / xfail not failing | 0 |
| (e) needs MATLAB code dynare-rs cannot run by design | 125 |
| (f) MATLAB timeout / infrastructure | 2 |
| (g) both fail unexpectedly | 0 |
| (x) both fail, as expected (xfail) | 38 |
| (r) examined: explained difference, not a dynare-rs bug | 87 |

## (c) grouped by error message

| #tests | normalised error | example tests | suspected area |
|---|---|---|---|

### (c) Dynare succeeds, dynare-rs fails

| test | symptom | suspected src area |
|---|---|---|

### (c2) dynare-rs fails after skipping MATLAB statements (error not on a MATLAB line)

| test | symptom | suspected src area |
|---|---|---|

### (b) both succeed, numbers differ

| test | symptom | suspected src area |
|---|---|---|

### (d) dynare-rs succeeds where Dynare fails / xfail not failing

| test | symptom | suspected src area |
|---|---|---|

### (r) examined: explained difference, not a dynare-rs bug

| test | symptom | suspected src area |
|---|---|---|
| ../examples/estimation/rbc_irf_matching.mod | MATLAB code: @(j) anonymous function |  |
| analytic_derivatives/burnside_3_order_PertParamsDerivs.mod | MATLAB code: error(['Error in matrix oo_.dr.g_' num2str(ord)]) in derivative checks |  |
| analytic_derivatives/fs2000_analytic_derivation.mod | MATLAB code: my_data.mat is saved after adding randn measurement error in verbatim |  |
| analytic_derivatives/fs2000_analytic_derivation_diag_H.mod | MATLAB code: my_data.mat is saved after adding randn measurement error in verbatim |  |
| analytic_derivatives/fs2000_analytic_derivation_newrat.mod | MATLAB code: my_data.mat is saved after adding randn measurement error in verbatim |  |
| block_bytecode/lola_solve_one_boundary.mod (+3) | MATLAB code: NBR_iss etc. come from load lola_data.mat into the workspace |  |
| decision_rules/first_order/walsh.mod | MATLAB code: calls the external function matlab.namespace.y_k |  |
| deterministic_simulations/lbj/rbc.mod | MATLAB code: histval uses CapitalSS = oo_.steady_state(ik), a MATLAB variable |  |
| deterministic_simulations/purely_forward/nk.mod | MATLAB code: tmp = inv(eye(3)-...)*... checks the analytical solution |  |
| deterministic_simulations/rbc_det.mod (+3) | MATLAB code: histval uses CapitalSS = oo_.steady_state(ik), a MATLAB variable |  |
| deterministic_simulations/rbc_det1.mod | MATLAB code: histval uses CapitalSS = oo_.steady_state(ik), a MATLAB variable |  |
| deterministic_simulations/rbc_det6.mod | MATLAB code: histval uses CapitalSS = oo_.steady_state(ik), a MATLAB variable |  |
| ep/burnside.mod | MATLAB code: set_dynare_seed(seed) takes a MATLAB variable (seed = 31415;) |  |
| estimation/MH_recover/fs2000_recover.mod (+1) | MATLAB code: MH files are copied and deleted with copyfile/delete before mh_recover |  |
| estimation/MH_recover/fs2000_recover_2.mod | MATLAB code: MH files are copied and deleted with copyfile/delete before mh_recover |  |
| estimation/MH_recover/fs2000_recover_3.mod | MATLAB code: MH files are copied and deleted with copyfile/delete before mh_recover |  |
| estimation/MH_recover/fs2000_recover_tmp.mod | MATLAB code: loads the MH files and compares temp1.x2/temp2.x2 |  |
| estimation/MH_recover/fs2000_slice_recover.mod | MATLAB code: loads the MH files and compares temp1.x2/temp2.x2 |  |
| estimation/fs2000_MCMC_jumping_covariance.mod | MATLAB code: test_matrix.mat is built from the mode file with load/save |  |
| estimation/method_of_moments/CET/cet_imh.mod | MATLAB code: @(j) anonymous function in cet_model.inc |  |
| estimation/method_of_moments/CET/cet_interface.mod | MATLAB code: @(j) anonymous function in cet_model.inc |  |
| estimation/method_of_moments/CET/cet_mh_posterior_mode.mod | MATLAB code: @(j) anonymous function in cet_model.inc |  |
| estimation/method_of_moments/CET/cet_mle.mod | MATLAB code: @(j) anonymous function in cet_model.inc |  |
| estimation/method_of_moments/CET/cet_rwmh.mod | MATLAB code: @(j) anonymous function in cet_model.inc |  |
| estimation/method_of_moments/CET/cet_slice.mod | MATLAB code: @(j) anonymous function in cet_model.inc |  |
| estimation/method_of_moments/CET/cet_tarb.mod | MATLAB code: @(j) anonymous function in cet_model.inc |  |
| estimation/univariate/nls/dynamicmodel1.mod | MATLAB code: estimate.nls(...) package call inside an if block |  |
| estimation/univariate/nls/staticmodel.mod | MATLAB code: estimate.nls(...) package call inside an if block |  |
| fs2000/fs2000.mod | optimizer endpoint on a flat ridge: log posteriors at MATLAB's mode agree to 3e-13 |  |
| heterogeneity/large_file_download/hank_1a_ssj.mod | MATLAB code: hank_1a_ssj.mat is downloaded with websave |  |
| heterogeneity/large_file_download/ks_ssj.mod | MATLAB code: ks_ssj.mat is downloaded with websave |  |
| kalman_initial_state/fs2000_kalman_initial_2_lag.mod | optimizer endpoint: log posteriors at MATLAB's mode agree to 1e-15 |  |
| measurement_errors/fs2000_corr_me_ml_mcmc/fs2000_corr_ME.mod | MATLAB code: fsdat_simul_corr.mat is built in verbatim with randn |  |
| model-inversion/bk-3a-nlsolve/varmodel.mod | MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P)) |  |
| model-inversion/bk-3a/varmodel.mod | MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P)) |  |
| model-inversion/bk-3b-nlsolve/varmodel.mod | MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P)) |  |
| model-inversion/bk-3b/varmodel.mod | MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P)) |  |
| model-inversion/bk-4-nlsolve/varmodel.mod | MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P)) |  |
| model-inversion/bk-4/varmodel.mod | MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P)) |  |
| model-inversion/bk-5-nlsolve/varmodel.mod | MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P)) |  |
| model-inversion/bk-5/varmodel.mod | MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P)) |  |
| model-inversion/bk-6/varmodel.mod | MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P)) |  |
| model-inversion/bk-7/varmodel.mod | MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P)) |  |
| model-inversion/bk-8/varmodel.mod | MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P)) |  |
| model-inversion/bk-diff-1/varmodel.mod | MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P)) |  |
| model-inversion/bk-diff-2/varmodel.mod | MATLAB code: M_ fields and model_inversion calls |  |
| model-inversion/bk-diff-3/varmodel.mod | MATLAB code: initialconditions dseries built from MATLAB matrices |  |
| model-inversion/bk-diff-4/varmodel.mod | MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P)) |  |
| model-inversion/bk-diff-5/varmodel.mod | MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P)) |  |
| observation_trends_and_prefiltering/ML/Trend_no_prefilter_selected_var.mod | MATLAB code: the data file is written by generate_trend_stationary_AR1.m |  |
| occbin/model_irrcap_twoconstraints/dynrbc.mod (+1) | MATLAB code: shockssequence drawn with randn('seed',1) inside a MATLAB if |  |
| pac/trend-component-19-growth-lin-comb/example1.mod | MATLAB code: eparams initial values read from M_.params |  |
| pac/trend-component-21/example1.mod | RNG: the estimated share goes to zero on dynare-rs's simulated data (MATLAB fails the same way on that data) |  |
| pac/trend-component-23/example1.mod | RNG: the estimated share goes to zero on dynare-rs's simulated data (MATLAB fails the same way on that data) |  |
| pac/trend-component-25/example1.mod | RNG: the estimated share goes to zero on dynare-rs's simulated data (MATLAB fails the same way on that data) |  |
| pac/trend-component-28/example2.mod | MATLAB code: initialconditions built from init = rand(...) |  |
| pac/trend-component-28/example3.mod | MATLAB code: initialconditions built from init = rand(...) |  |
| pac/var-10e/example1.mod (+1) | MATLAB code: example1.data.mat is written by the MATLAB method TrueData.save(...) |  |
| particle/dsge_base_sis.mod | RNG: generic resampling without propagation (as in Dynare's sequential_importance_particle_filter.m) underflows for some streams (4 of 8 ChaCha seeds at 5000 particles); MATLAB's stream passes |  |
| particle/first_spec.mod (+2) | MATLAB code: save('my_data.mat') of a simulation drawn with MATLAB's randn |  |
| particle/first_spec_MCMC.mod | MATLAB code: save('my_data_MCMC.mat') of a simulation drawn with MATLAB's randn |  |
| particle/first_spec_hssmc.mod | MATLAB code: save('my_data.mat') of a simulation drawn with MATLAB's randn |  |
| smoother2histval/fs2000_smooth_stoch_simul.mod | MATLAB code in the entry |  |
| solve_algo_12_14/simul_backward_reference.mod (+2) | MATLAB code: rand, randn, save |  |
| steady_state/walsh1_initval.mod | Dynare bug: `n=log(1/3);` with n endogenous makes the 7.2 preprocessor overwrite M_.params(1) |  |
| stochastic_simulations/pruning/fs2000_pruning.mod | MATLAB code: the ' transpose operator |  |
| trend-component-and-var-models/legacy/tcm1.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/legacy/tcm2.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/legacy/tcm3.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/legacy/tcm4.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/legacy/tcm5.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/legacy/vm1.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/legacy/vm2.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/legacy/vm3.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/tcm1.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/tcm10.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/tcm11.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/tcm12.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/tcm2.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/tcm3.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/tcm4.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/tcm5.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/tcm9.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/vm1.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/vm2.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/vm3.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |
| trend-component-and-var-models/vm5.mod | MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1)) |  |

### (p) optimizer endpoint: same log posterior at the mode, the mode differs

| test | symptom | suspected src area |
|---|---|---|
| estimation/fs2000_estimated_params_remove.mod | same log posterior, 2 fields at the mode differ (max 1.1e-03): SmoothedVariables.P, SmoothedVariables.c |  |
| estimation/fs2000_with_weibull_prior.mod | same log posterior, 3 fields at the mode differ (max 3.3e-03): posterior_std_at_mode.shocks_std.e_m, posterior_mode.parameters.rho, posterior_std_at_mode.parameters.rho |  |
| estimation/heteroskedastic_shocks/fs2000_het_sample_restriction.mod | same log posterior, 3 fields at the mode differ (max 9.4e-01): dr.ghu, dr.ghx, dr.ys |  |
| fs2000/fs2000_missing_data.mod | same log posterior, 3 fields at the mode differ (max 1.6e-03): posterior_std_at_mode.shocks_std.e_m, posterior_mode.parameters.rho, posterior_std_at_mode.parameters.rho |  |
| gradient/fs2000_numgrad_13.mod | same log posterior, 2 fields at the mode differ (max 1.1e-03): SmoothedVariables.P, SmoothedVariables.c |  |
| gradient/fs2000_numgrad_15.mod | same log posterior, 2 fields at the mode differ (max 1.1e-03): SmoothedVariables.P, SmoothedVariables.c |  |
| gradient/fs2000_numgrad_2.mod | same log posterior, 2 fields at the mode differ (max 1.1e-03): SmoothedVariables.P, SmoothedVariables.c |  |
| gradient/fs2000_numgrad_3.mod | same log posterior, 2 fields at the mode differ (max 1.1e-03): SmoothedVariables.P, SmoothedVariables.c |  |
| gradient/fs2000_numgrad_5.mod | same log posterior, 2 fields at the mode differ (max 1.1e-03): SmoothedVariables.P, SmoothedVariables.c |  |
| kalman/block/fs2000.mod | same log posterior, 3 fields at the mode differ (max 1.8e-03): posterior_std_at_mode.shocks_std.e_m, posterior_mode.parameters.rho, posterior_std_at_mode.parameters.rho |  |
| kalman_initial_state/fs2000_kalman_initial.mod | same log posterior, 1 fields at the mode differ (max 1.2e-03): posterior_std_at_mode.parameters.del |  |
| optimizers/fs2000_1.mod | same log posterior, 2 fields at the mode differ (max 1.6e-03): posterior_mode.parameters.del, posterior_std_at_mode.parameters.del |  |
| shock_decomposition/fs2000_est.mod | same log posterior, 2 fields at the mode differ (max 1.1e-03): SmoothedVariables.P, SmoothedVariables.c |  |
| shock_decomposition/fs2000_est_varlist.mod | same log posterior, 2 fields at the mode differ (max 1.1e-03): SmoothedVariables.P, SmoothedVariables.c |  |
| shock_decomposition/fs2000_forecasts.mod | same log posterior, 6 fields at the mode differ (max 1.7e+00): posterior_std_at_mode.shocks_std.e_m, posterior_mode.parameters.rho, posterior_std_at_mode.parameters.rho, dr.ghu |  |
| shock_decomposition/ls2003_plot.mod | same log posterior, 3 fields at the mode differ (max 1.2e-03): posterior_mode.parameters.alpha, posterior_mode.parameters.rho_pies, SmoothedShocks.e_ys |  |
| smoother2histval/fs2000_smooth_forecast.mod | same log posterior, 4 fields at the mode differ (max 1.3e-03): forecast.Mean.k, forecast.HPDinf.k, forecast.HPDsup.k, SmoothedVariables.k |  |

### (s) stochastic-only differences (RNG driven)

| test | symptom | suspected src area |
|---|---|---|
| ../examples/estimation/schorfheide_2000.mod | 19 stochastic fields differ: posterior_mean.parameters.gam, posterior_mean.parameters.logmst, posterior_mean.parameters.rho, posterior_mean.parameters.phi |  |
| ../examples/optimal_policy/nk_ramsey_osr.mod | 9 stochastic fields differ: var, autocorr{1}, autocorr{2}, autocorr{3} |  |
| ../examples/optimal_policy/nk_ramsey_steady_file.mod | 9 stochastic fields differ: var, autocorr{1}, autocorr{2}, autocorr{3} |  |
| ../examples/stochastic_simulations/collard_2001_simulated_moments.mod | 19 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| ../examples/stochastic_simulations/collard_2001_theoretical_moments.mod | 8 stochastic fields differ: irfs.c_e, irfs.c_u, irfs.h_e, irfs.h_u |  |
| ../examples/stochastic_simulations/schorfheide_2000_nonstationary.mod | 19 stochastic fields differ: irfs.P_e_a, irfs.P_e_m, irfs.R_e_m, irfs.W_e_a |  |
| arima/mod1.mod (+3) | 14 stochastic fields differ: posterior_mode.shocks_std.e_x, posterior_std_at_mode.shocks_std.e_x, posterior_mode.shocks_std.e_y, posterior_std_at_mode.shocks_std.e_y |  |
| arima/mod2.mod (+3) | 22 stochastic fields differ: posterior_mode.shocks_std.e_x, posterior_std_at_mode.shocks_std.e_x, posterior_mode.shocks_std.e_y, posterior_std_at_mode.shocks_std.e_y |  |
| bvar_a_la_sims/bvar_and_dsge.mod | 3 stochastic fields differ: posterior_mean.parameters.rho_x, posterior_mean.parameters.rho_y, SmoothedShocks.e_y |  |
| conditional_forecasts/2/fs2000_est.mod | 19 stochastic fields differ: posterior_mean.parameters.alp, posterior_mean.parameters.gam, posterior_mean.parameters.rho, posterior_mean.parameters.psi |  |
| conditional_variance_decomposition/example1.mod | 11 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| data/mod1a.mod | 4 stochastic fields differ: posterior_mean.parameters.rho_x, posterior_mean.parameters.rho_y, SmoothedShocks.e_x, SmoothedShocks.e_y |  |
| decision_rules/example1.mod (+1) | 8 stochastic fields differ: irfs.c_e, irfs.c_u, irfs.h_e, irfs.h_u |  |
| decision_rules/k_order/burnside_k_order.mod | 10 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| decision_rules/k_order/fs2000k2a.mod (+6) | 24 stochastic fields differ: irfs.P_e_a, irfs.P_e_m, irfs.R_e_m, irfs.W_e_a |  |
| discretionary_policy/dennis_1.mod (+2) | 10 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| dsge-var/simul_hybrid.mod (+2) | 10 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| estimation/TaRB/fs2000_tarb.mod | 21 stochastic fields differ: MarginalDensity.ModifiedHarmonicMean, posterior_mean.parameters.alp, posterior_mean.parameters.bet, posterior_mean.parameters.gam |  |
| estimation/conditional-likelihood/1/fs2000_estimation_exact.mod (+1) | 19 stochastic fields differ: posterior_mean.parameters.alp, posterior_mean.parameters.gam, posterior_mean.parameters.rho, posterior_mean.parameters.psi |  |
| estimation/fs2000_fast.mod | 5 stochastic fields differ: posterior_mean.parameters.alp, posterior_mean.parameters.gam, posterior_mean.parameters.rho, posterior_mean.parameters.psi |  |
| estimation/independent_mh/fs2000_independent_mh.mod | 20 stochastic fields differ: MarginalDensity.ModifiedHarmonicMean, posterior_mean.parameters.alp, posterior_mean.parameters.gam, posterior_mean.parameters.rho |  |
| estimation/ls2003_endog_prior_restrict_estimation.mod | 22 stochastic fields differ: MarginalDensity.ModifiedHarmonicMean, posterior_mean.parameters.psi1, posterior_mean.parameters.psi2, posterior_mean.parameters.psi3 |  |
| estimation/slice/bf.mod | 9 stochastic fields differ: posterior_mode.shocks_std.epsu, posterior_mode.shocks_std.epsr, posterior_mode.parameters.phip, posterior_mode.parameters.rhop |  |
| estimation/slice/fs2000_estimate_initial_states.mod | 16 stochastic fields differ: posterior_mode.shocks_std.e_a, posterior_mode.shocks_std.e_m, posterior_mode.parameters.alp, posterior_mode.parameters.bet |  |
| estimation/slice/fs2000_estimate_initial_states_log.mod | 13 stochastic fields differ: posterior_mode.shocks_std.e_a, posterior_mode.shocks_std.e_m, posterior_mode.parameters.alp, posterior_mode.parameters.bet |  |
| estimation/slice/fs2000_slice.mod | 24 stochastic fields differ: posterior_mode.shocks_std.e_m, posterior_mode.parameters.gam, posterior_mode.parameters.rho, posterior_mode.parameters.psi |  |
| estimation/slice/fs2000_slice_use_prior.mod | 27 stochastic fields differ: posterior_mode.shocks_std.e_a, posterior_mode.shocks_std.e_m, posterior_mode.parameters.alp, posterior_mode.parameters.bet |  |
| estimation/t_proposal/fs2000_student.mod | 20 stochastic fields differ: posterior_mean.parameters.alp, posterior_mean.parameters.bet, posterior_mean.parameters.gam, posterior_mean.parameters.rho |  |
| estimation/tune_mh_jscale/fs2000.mod | 17 stochastic fields differ: posterior_mean.parameters.alp, posterior_mean.parameters.gam, posterior_mean.parameters.psi, posterior_mean.parameters.del |  |
| external_function/benchmark.mod (+9) | 8 stochastic fields differ: irfs.c_e, irfs.c_u, irfs.h_e, irfs.h_u |  |
| filter_step_ahead/fs2000_filter_step_ahead_bayesian.mod | 5 stochastic fields differ: posterior_mean.parameters.alp, posterior_mean.parameters.gam, posterior_mean.parameters.rho, posterior_mean.parameters.psi |  |
| filter_step_ahead/trend_cycle_decomposition.mod | 2 stochastic fields differ: MarginalDensity.ModifiedHarmonicMean, posterior_mean.parameters.alpha |  |
| forecast/example1_varexo_det.mod | 12 stochastic fields differ: forecast.HPDinf.y, forecast.HPDsup.y, forecast.HPDinf.c, forecast.HPDsup.c |  |
| forecast/forecast_test_AR1_varexo_det_order_2.mod | 2 stochastic fields differ: forecast.HPDinf.A, forecast.HPDsup.A |  |
| fs2000/fs2000a.mod | 20 stochastic fields differ: posterior_mean.parameters.alp, posterior_mean.parameters.gam, posterior_mean.parameters.rho, posterior_mean.parameters.psi |  |
| fs2000/fs2000c.mod | 19 stochastic fields differ: irfs.P_e_a, irfs.P_e_m, irfs.R_e_m, irfs.W_e_a |  |
| gsa/ls2003.mod | 10 stochastic fields differ: posterior_mean.parameters.psi1, posterior_mean.parameters.psi2, posterior_mean.parameters.psi3, posterior_mean.parameters.rho_R |  |
| gsa/morris/nk_est.mod | 2 stochastic fields differ: posterior_mean.parameters.ETA, posterior_mean.parameters.KAPPA |  |
| irfs/example1_unit_std.mod | 8 stochastic fields differ: irfs.y_e, irfs.c_e, irfs.k_e, irfs.h_e |  |
| kalman_filter_smoother/check_variable_dimensions/fs2000.mod | 6 stochastic fields differ: posterior_mean.parameters.alp, posterior_mean.parameters.bet, posterior_mean.parameters.gam, posterior_mean.parameters.rho |  |
| kalman_filter_smoother/gen_data.mod (+13) | 10 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| kalman_filter_smoother/test_compute_Pinf_Pstar_data.mod (+1) | 10 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| log_transform/example1.mod | 8 stochastic fields differ: irfs.c_e, irfs.c_u, irfs.h_e, irfs.h_u |  |
| log_transform/fs2000_nonstationary.mod | 19 stochastic fields differ: irfs.P_e_a, irfs.P_e_m, irfs.R_e_m, irfs.W_e_a |  |
| loglinear/example4_exp.mod (+1) | 10 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| loglinear/example4_exp_histval.mod (+1) | 10 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| macro_processor/example1_macro.mod | 8 stochastic fields differ: irfs.c_e, irfs.c_u, irfs.h_e, irfs.h_u |  |
| moments/example1_bp_test.mod | 9 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| moments/example1_one_sided_hp_test.mod | 10 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| moments/example1_order2_pruning.mod | 8 stochastic fields differ: irfs.c_e, irfs.c_u, irfs.h_e, irfs.h_u |  |
| moments/example1_theoretical_simulated.mod | 10 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| moments/example1_var_decomp.mod | 19 stochastic fields differ: irfs.y_e, irfs.c_e, irfs.k_e, irfs.h_e |  |
| occbin/filter/NKM.mod | 32 stochastic fields differ: posterior_mean.parameters.varphip, posterior_mean.parameters.phipi, posterior_mean.parameters.phiy, posterior_mean.parameters.h |  |
| pac/trend-component-1-mce/example_sto.mod | 10 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| pac/trend-component-2-mce/example_sto.mod | 10 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| particle/dsge_base2.mod | 4 stochastic fields differ: posterior_mode.shocks_std.y, posterior_mode.shocks_std.l, posterior_mode.shocks_std.i, posterior.optimization.log_density |  |
| preprocessor_checks/comments.mod | 8 stochastic fields differ: irfs.c_e, irfs.c_u, irfs.h_e, irfs.h_u |  |
| preprocessor_checks/example1_mlv.mod | 8 stochastic fields differ: irfs.c_e, irfs.c_u, irfs.h_e, irfs.h_u |  |
| prior_posterior_function/fs2000_prior_posterior_function.mod | 19 stochastic fields differ: posterior_mean.parameters.alp, posterior_mean.parameters.bet, posterior_mean.parameters.gam, posterior_mean.parameters.rho |  |
| shock_decomposition/example1_calib_shock_decomp.mod | 10 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| smoother2histval/fs2000_smooth.mod (+1) | 5 stochastic fields differ: posterior_mean.parameters.alp, posterior_mean.parameters.gam, posterior_mean.parameters.rho, posterior_mean.parameters.psi |  |
| stat_functions/example1_abs_sign.mod | 8 stochastic fields differ: irfs.c_e, irfs.c_u, irfs.h_e, irfs.h_u |  |
| stochastic_purely_forward/stochastic_purely_forward.mod | 9 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| stochastic_purely_forward/stochastic_purely_forward_with_static.mod | 9 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| stochastic_simulations/example1.mod | 19 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| stochastic_simulations/example1_irf_shocks.mod | 8 stochastic fields differ: irfs.c_eps, irfs.c_u, irfs.h_eps, irfs.h_u |  |
| stochastic_simulations/example1long.mod (+2) | 10 stochastic fields differ: irfs.c_e, irfs.c_u, irfs.h_e, irfs.h_u |  |
| stochastic_simulations/example2_simul_replic.mod | 19 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| stochastic_simulations/example_seeds.mod | 1 stochastic fields differ: endo_simul |  |
| stochastic_simulations/histval_predetermined.mod | 10 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| stochastic_simulations/histval_sto.mod | 10 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| stochastic_simulations/pruning/fs2000k3_p.mod | 35 stochastic fields differ: autocorr{1}, autocorr{2}, autocorr{3}, autocorr{4} |  |
| trend_var/fs2000_log_nonstationary.mod | 16 stochastic fields differ: irfs.P_e_a, irfs.P_e_m, irfs.W_e_a, irfs.W_e_m |  |
| trend_var/fs2000_nonstationary.mod | 19 stochastic fields differ: irfs.P_e_a, irfs.P_e_m, irfs.R_e_m, irfs.W_e_a |  |

### (g) both fail unexpectedly

| test | symptom | suspected src area |
|---|---|---|

### (f) MATLAB timeout / infrastructure

| test | symptom | suspected src area |
|---|---|---|
| ../examples/reporting/collard_2001_reporting.mod | MATLAB TIMEOUT |  |
| heterogeneity/large_file_download/hank_2a_ssj.mod | MATLAB TIMEOUT |  |

