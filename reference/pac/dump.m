% Run from a private copy of reference/pac, ONE fresh MATLAB session per model:
%   for m in pc1 pc2 pc3 pc4 pc5 pc6 pc7 pc8 pc9; do PAC_NAME=$m /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump.m')" > log_$m.txt 2>&1; done
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end % DYNARE_VERSION=7.1 for the 7.1 baseline
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
name = getenv('PAC_NAME');
dynare(name, 'nolog', 'json=compute', 'stochastic');
r = struct();
r.param_names = M_.param_names;
r.params = M_.params;
r.endo_names = M_.endo_names;
r.exo_names = M_.exo_names;
r.orig_endo_nbr = M_.orig_endo_nbr;
if isfield(oo_, 'dr') && isfield(oo_.dr, 'ghx')
    r.ys = oo_.dr.ys;
    r.ghx = oo_.dr.ghx;
    r.ghu = oo_.dr.ghu;
    r.order_var = oo_.dr.order_var;
    r.state_var = M_.state_var;
    r.irfs = oo_.irfs;
end
% pc6: the share of optimizing agents (lambda) is estimated too; pc8, pc9: additive terms
% (pc8: the share too, initial values of tests/pac/trend-component-25 and -22)
share = {};
if strcmp(name, 'pc6'), share = {'lambda'}; end
extra = {};
if strcmp(name, 'pc8'), extra = {'c_z_s', .9, 'c_z_dx2', .1, 'lambda', .7}; end
if strcmp(name, 'pc9'), extra = {'c_z_dx2', .5}; end
if ~ismember(name, {'pc4', 'pc7'})
    rng(1234);
    T = 600;
    initialconditions = dseries(zeros(10, M_.endo_nbr+M_.exo_nbr), dates('2000Q1'), vertcat(M_.endo_names, M_.exo_names));
    TrueData = simul_backward_model(initialconditions, T);
    keep = vertcat(M_.endo_names(1:M_.orig_endo_nbr), M_.exo_names);
    d = struct();
    d.names = keep;
    d.data = TrueData{keep{:}}.data;
    d.first = TrueData.dates(1).char;
    r.sim = d;
    rg = dates('2005Q1'):dates('2005Q1')+500;
    r.range_rows = [rg(1)-TrueData.dates(1), rg(end)-TrueData.dates(1)];
    % NLS (Gauss-Newton)
    clear eparams
    eparams.e_c_m = .9;
    eparams.c_z_1 = .5;
    eparams.c_z_2 = .2;
    if ~isempty(share), eparams.(share{1}) = .5; end
    for k = 1:2:numel(extra), eparams.(extra{k}) = extra{k+1}; end
    edata = TrueData;
    edata.ez = dseries(NaN, dates('2000Q1'));
    params_before = M_.params;
    pac.estimate.nls('zpac', eparams, edata, rg, 'GaussNewton');
    p = oo_.pac.pacman;
    r.nls = struct('parnames', {p.parnames}, 'estimator', p.estimator, 'ssr', p.ssr, 's2', p.s2, 'R2', p.R2, ...
                   'covariance', p.covariance, 'student', p.student, 'residual', p.residual.data, 'params_after', M_.params);
    % iterative OLS (restarts from the calibration)
    M_.params = params_before;
    clear eparams
    eparams.e_c_m = .9;
    eparams.c_z_1 = .5;
    eparams.c_z_2 = .2;
    if ~isempty(share), eparams.(share{1}) = .5; end
    for k = 1:2:numel(extra), eparams.(extra{k}) = extra{k+1}; end
    try  % iterative_ols does not support composite targets (no h_param_indices)
        pac.estimate.iterative_ols('zpac', eparams, edata, rg);
        p = oo_.pac.pacman;
        r.ols = struct('parnames', {p.parnames}, 'estimator', p.estimator, 'ssr', p.ssr, 'residual', p.residual, 'params_after', M_.params);
    catch ME
        r.ols_error = ME.message;
    end
end
fid = fopen(['ref_' name '.json'], 'w'); fprintf(fid, '%s', jsonencode(r)); fclose(fid);
