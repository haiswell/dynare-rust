% Dumps Dynare 7.2's recursive (ls2003_recursive.mod: nobs = [76 79], forecast) and rolling-window
% (ls2003_rolling.mod: first_obs = [8 12]) estimations into JSON (tests/sweep72_features.rs):
% the mode and Laplace approximation of each sample, and oo_.RecursiveForecast.
% initial_log_posterior (the console's "Initial value of the log posterior" of each sample) was
% added to the JSON from the MATLAB log.
% Run from a copy of this folder: matlab -batch "addpath('<dynare>/matlab'); dump_recursive"
for f = {'ls2003_recursive', 'ls2003_rolling'}
    dynare(f{1}, 'nograph', 'nointeractive');
    r = struct();
    keys = find(~cellfun(@isempty, oo_recursive_));
    r.keys = keys;
    for k = keys
        m = load([f{1} '_' int2str(k) filesep 'Output' filesep f{1} '_mode.mat']);
        s.xparam1 = m.xparam1;
        s.parameter_names = m.parameter_names;
        s.laplace = oo_recursive_{k}.MarginalDensity.LaplaceApproximation;
        r.(['sample_' int2str(k)]) = s;
    end
    if isfield(oo_, 'RecursiveForecast')
        r.RecursiveForecast = oo_.RecursiveForecast;
    end
    fid = fopen([f{1} '.json'], 'w');
    fprintf(fid, '%s', jsonencode(r, 'ConvertInfAndNaN', false));
    fclose(fid);
    clearvars -except f
end
