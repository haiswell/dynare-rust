% dump_occbin_setup_opts.m -- Dynare 7.2 reference for the occbin_setup options that
% +occbin/setup.m does not forward to options_.occbin (smoother_max_check_ahead_periods,
% smoother_plot, smoother_first_period_occbin_update, smoother_max_number_of_iterations,
% posterior_importance_sampling*; occbin.posterior_importance_sampling is never called):
% nk_est.mod -DSETUP_OPTS sets them to values that would change the OccBin smoother, and this
% records options_.occbin after occbin_setup and the smoother at the points of
% ref_occbin_est.json, written to ref_occbin_setup_opts.json. Run from a PRIVATE COPY of this
% folder:  /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_occbin_setup_opts.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
ref = jsondecode(fileread('ref_occbin_est.json'));
dynare nk_est -DNOEST -DSETUP_OPTS noclearall nograph nolog
global M_ options_ oo_ estim_params_ bayestopt_
out = struct();
sm = options_.occbin.smoother;
out.smoother = struct('max_check_ahead_periods', sm.max_check_ahead_periods, 'plot', sm.plot, ...
    'first_period_occbin_update', sm.first_period_occbin_update, 'max_number_of_iterations', sm.max_number_of_iterations);
out.posterior_importance_sampling = options_.occbin.posterior_importance_sampling;
options_.smoother = true;
options_.mh_replic = 0;
options_.datafile = 'nk_est_data.csv';
options_.nograph = true;
options_.occbin.smoother.plot = false;
[dataset_, dataset_info, xparam_init, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
pts = ref.pkf.points;
out.points = {};
for k = 1:numel(pts)
    if iscell(pts), p = pts{k}; else, p = pts(k); end
    % alphahat/etahat equal those of ref_occbin_smoother.json (max diff 6e-11), so not stored
    [~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, oo1] = occbin.DSGE_smoother(p.x, dataset_.nobs, transpose(dataset_.data), ...
        dataset_info.missing.aindex, dataset_info.missing.state, M_, oo_, options_, bayestopt_, estim_params_, dataset_, dataset_info);
    s = struct();
    r = oo1.occbin.smoother;
    s.error_flag = r.error_flag;
    s.regime = arrayfun(@(x) double(x.regime), r.regime_history, 'UniformOutput', false);
    s.regimestart = arrayfun(@(x) double(x.regimestart), r.regime_history, 'UniformOutput', false);
    if isfield(r, 'regime_history_iter'), s.iterations = size(r.regime_history_iter, 1); end
    out.points{end+1} = s;
end
fid = fopen(fullfile(here, 'ref_occbin_setup_opts.json'), 'w');
fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true));
fclose(fid);
disp('wrote ref_occbin_setup_opts.json');
