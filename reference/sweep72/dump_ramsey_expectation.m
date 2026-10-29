% Dumps Dynare 7.2's results for nk_ramsey_expectation.mod (tests/optimal_policy) and
% nk_ramsey_diff_expectation.mod into JSON (tests/sweep72_features.rs).
% Run from this folder: matlab -batch "addpath('<dynare>/matlab'); dump_ramsey_expectation"
for f = {'nk_ramsey_expectation', 'nk_ramsey_diff_expectation'}
    dynare(f{1}, 'nograph', 'nointeractive');
    r = struct();
    r.endo_names = M_.endo_names;
    r.ramsey_orig_endo_nbr = M_.ramsey_orig_endo_nbr;
    r.ramsey_orig_eq_nbr = M_.ramsey_orig_eq_nbr;
    r.orig_maximum_lag_with_diffs_expanded = M_.orig_maximum_lag_with_diffs_expanded;
    r.aux_types = [M_.aux_vars.type];
    r.ys = oo_.dr.ys;
    r.order_var = oo_.dr.order_var;
    r.ghx = oo_.dr.ghx;
    r.ghu = oo_.dr.ghu;
    r.planner_objective_value = oo_.planner_objective_value;
    fid = fopen([f{1} '.json'], 'w');
    fprintf(fid, '%s', jsonencode(r, 'ConvertInfAndNaN', false));
    fclose(fid);
    clearvars -except f
end
