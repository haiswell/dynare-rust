% Reference values for osr(analytic_derivation, analytic_derivation_mode, huge_number) with
% Dynare 7.2: writes ref_osr_ad.json. Run from a private copy of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_osr_ad.m')"
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = '/Applications/Dynare/7.2-arm64'; end
addpath(fullfile(droot, 'matlab'));
% name, CONST, osr options
cases = {
    'ad4',     false, 'opt_algo=4, analytic_derivation';
    'ad4c',    true,  'opt_algo=4, analytic_derivation';
    'num4c',   true,  'opt_algo=4';
    'ad3',     false, 'opt_algo=3, analytic_derivation';
    'ad1',     false, 'opt_algo=1, analytic_derivation';
    'adm0',    true,  'opt_algo=4, analytic_derivation, analytic_derivation_mode=0';
    'adm1',    true,  'opt_algo=4, analytic_derivation, analytic_derivation_mode=1';
    'admm1',   true,  'opt_algo=4, analytic_derivation, analytic_derivation_mode=-1';
    'admm2',   true,  'opt_algo=4, analytic_derivation, analytic_derivation_mode=-2';
    'hn2',     false, 'opt_algo=2, huge_number=2';
};
out = struct();
for k = 1:size(cases, 1)
    name = cases{k, 1};
    fid = fopen([name '.mod'], 'w');
    if cases{k, 2}
        fprintf(fid, '@#define CONST = 1\n');
    end
    if contains(cases{k, 3}, 'analytic_derivation')
        fprintf(fid, '@#define ORDER2 = 1\n');
    end
    fprintf(fid, '@#define OSR_OPTIONS = "%s"\n@#include "osr_ad.mod"\n', cases{k, 3});
    fclose(fid);
    clearvars -global
    global M_ oo_ options_
    r = struct();
    r.options = cases{k, 3};
    r.const = cases{k, 2};
    try
        eval(['dynare ' name ' noclearall nolog']);
        r.error = '';
        r.osr_objective = oo_.osr.objective_function;
        r.osr_params = [oo_.osr.optim_params.phi_pi; oo_.osr.optim_params.phi_y];
        r.analytic_derivation = options_.analytic_derivation;
        r.huge_number = options_.huge_number;
        % loss and gradient of osr.objective at a few points
        i_params = [strmatch('phi_pi', M_.param_names, 'exact'); strmatch('phi_y', M_.param_names, 'exact')];
        i_var = unique(M_.osr.variable_indices);
        W = M_.osr.variable_weights(i_var, i_var);
        dr = set_state_space(oo_.dr, M_);
        oo1 = oo_; oo1.dr = dr;
        pts = [1.5 0.125; 2.0 0.3; 1.2 -0.1; r.osr_params'];
        r.points = pts;
        r.loss = zeros(size(pts, 1), 1);
        r.grad = zeros(size(pts, 1), 2);
        for q = 1:size(pts, 1)
            [l, ~, ~, df] = osr.objective(pts(q, :)', M_, oo1, options_, i_params, dr.inv_order_var(i_var), W);
            r.loss(q) = l;
            r.grad(q, :) = df';
        end
    catch err
        r.error = err.message;
    end
    out.(name) = r;
end
fid = fopen('ref_osr_ad.json', 'w');
fprintf(fid, '%s', jsonencode(out, 'PrettyPrint', true));
fclose(fid);
