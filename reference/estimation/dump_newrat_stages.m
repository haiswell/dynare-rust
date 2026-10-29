% dump_newrat_stages.m -- reference values for the unit tests of src/estimation/optim/newrat.rs:
% the inputs and outputs of the calls newrat.m makes to mr_hessian.m, mr_gstep.m and csminit1.m
% while minimizing ratfit_loop.m (an objective Rust evaluates bit for bit the same), every
% double as num2hex, written to ref_newrat_stages.json. Given MATLAB's inputs, these steps
% reproduce MATLAB's outputs bit for bit (the inverse outer-product Hessian of mr_hessian.m
% to rounding: MATLAB's LAPACK solves differ from Accelerate's in the last bits).
% Run from a private copy of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_newrat_stages.m')"
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
dynare_config;
% recording wrappers around Dynare's mr_hessian, mr_gstep and csminit1 (renamed *_orig)
wdir = fullfile(pwd, 'newrat_stage_wrappers');
if ~exist(wdir, 'dir'), mkdir(wdir); end
for f = {'mr_hessian', 'mr_gstep', 'csminit1'}
    src = fileread(fullfile(droot, 'matlab', 'optimization', [f{1} '.m']));
    src = regexprep(src, ['=\s*' f{1} '\('], ['= ' f{1} '_orig('], 'once');
    fid = fopen(fullfile(wdir, [f{1} '_orig.m']), 'w'); fprintf(fid, '%s', src); fclose(fid);
end
H = 'function c = hx(v)\nc = cellstr(num2hex(double(v(:))));\nif isempty(c), c = {}; end\n';
fid = fopen(fullfile(wdir, 'hx.m'), 'w'); fprintf(fid, H); fclose(fid);
W = {
 'function [hessian_mat, gg, htol1, ihh, hh_mat0, hh1, hess_info, fcount] = mr_hessian(x,func,penalty,hflag,htol0,hess_info,bounds,prior_std,Save_files,varargin)'
 'hi0 = hess_info;'
 '[hessian_mat, gg, htol1, ihh, hh_mat0, hh1, hess_info, fcount] = mr_hessian_orig(x,func,penalty,hflag,htol0,hess_info,bounds,prior_std,Save_files,varargin{:});'
 'global NRSTAGES'
 'if numel(NRSTAGES.mr_hessian) < 8'
 '    NRSTAGES.mr_hessian{end+1} = struct(''x'',{hx(x)},''penalty'',{hx(penalty)},''hflag'',hflag,''htol0'',{hx(htol0)},''h1_in'',{hx(hi0.h1)},''htol_in'',{hx(hi0.htol)},''htol_rel'',{hx(hi0.htol_rel)},''lb'',{hx(bounds(:,1))},''ub'',{hx(bounds(:,2))},''gg'',{hx(gg)},''htol1'',{hx(htol1)},''ihh'',{hx(ihh)},''hh_mat0'',{hx(hh_mat0)},''hessian_mat'',{hx(hessian_mat)},''hh1'',{hx(hh1)},''htol_out'',{hx(hess_info.htol)},''fcount'',fcount);'
 'end'
 };
fid = fopen(fullfile(wdir, 'mr_hessian.m'), 'w'); fprintf(fid, '%s\n', W{:}); fclose(fid);
W = {
 'function [f0, x, ig, fcount] = mr_gstep(h1,x,bounds,func0,penalty,htol0,Verbose,Save_files,gradient_epsilon,parameter_names,robust,varargin)'
 'x_in = x;'
 '[f0, x, ig, fcount] = mr_gstep_orig(h1,x,bounds,func0,penalty,htol0,Verbose,Save_files,gradient_epsilon,parameter_names,robust,varargin{:});'
 'global NRSTAGES'
 'if numel(NRSTAGES.mr_gstep) < 60'
 '    NRSTAGES.mr_gstep{end+1} = struct(''h1'',{hx(h1)},''x_in'',{hx(x_in)},''lb'',{hx(bounds(:,1))},''ub'',{hx(bounds(:,2))},''penalty'',{hx(penalty)},''htol0'',{hx(htol0)},''gradient_epsilon'',{hx(gradient_epsilon)},''robust'',double(robust),''f0'',{hx(f0)},''x'',{hx(x)},''ig'',ig(:)'',''fcount'',fcount);'
 'end'
 };
fid = fopen(fullfile(wdir, 'mr_gstep.m'), 'w'); fprintf(fid, '%s\n', W{:}); fclose(fid);
W = {
 'function [fhat,xhat,fcount,retcode] = csminit1(fcn,x0,penalty,f0,g0,badg,H0,Verbose,varargin)'
 '[fhat,xhat,fcount,retcode] = csminit1_orig(fcn,x0,penalty,f0,g0,badg,H0,Verbose,varargin{:});'
 'global NRSTAGES'
 'if numel(NRSTAGES.csminit1) < 12'
 '    NRSTAGES.csminit1{end+1} = struct(''x0'',{hx(x0)},''penalty'',{hx(penalty)},''f0'',{hx(f0)},''g0'',{hx(g0)},''badg'',double(badg),''H0'',{hx(H0)},''fhat'',{hx(fhat)},''xhat'',{hx(xhat)},''fcount'',fcount,''retcode'',retcode);'
 'end'
 };
fid = fopen(fullfile(wdir, 'csminit1.m'), 'w'); fprintf(fid, '%s\n', W{:}); fclose(fid);
addpath(wdir, '-begin');
rehash;
% name, x0, lb, ub, with_dlik, robust, gradient_epsilon, htol_rel, ftol (as in dump_newrat_iterates.m)
x0 = [1; 1; 0.1; 0.3];
lb = [-5; -5; -0.5; -5]; ub = [5; 5; 2; 5];
lbb = [-5; -5; 0; -5]; ubb = [5; 1.5; 2; 5];
cases = {'base',   x0, lb,  ub,  1, false, 1e-6, NaN,  1e-5;
         'robust', x0, lbb, ubb, 1, true,  1e-5, NaN,  1e-4;
         'rel',    x0, lb,  ub,  1, false, 1e-6, 1e-6, 1e-5;
         'bfgs',   x0, lb,  ub,  0, false, 1e-6, NaN,  1e-5;
         'robust2', [2; 0.5; 0.5; 0.1], lb, ub, 1, true, 1e-5, NaN, 1e-7};
global NRSTAGES
out = struct();
for c = 1:size(cases, 1)
    [name, x0, lb, ub, wd, robust, geps, hrel, ftol] = cases{c,:};
    NRSTAGES = struct('mr_hessian', {{}}, 'mr_gstep', {{}}, 'csminit1', {{}});
    hess_info = struct();
    hess_info.gstep = [1e-2; 1]; hess_info.htol = ftol; hess_info.htol_rel = hrel;
    hess_info.h1 = geps*ones(4,1); hess_info.robust = robust;
    newrat(@ratfit_loop, x0, [lb ub], 0, ftol, 1000, 0, 0, 0, hess_info, NaN(4,1), geps, {'a','b','c','d'}, wd);
    out.(name) = struct('with_dlik', wd, 'mr_hessian', {NRSTAGES.mr_hessian}, 'mr_gstep', {NRSTAGES.mr_gstep}, 'csminit1', {NRSTAGES.csminit1});
end
rmpath(wdir);
fid = fopen('ref_newrat_stages.json', 'w'); fprintf(fid, '%s', jsonencode(out)); fclose(fid);
