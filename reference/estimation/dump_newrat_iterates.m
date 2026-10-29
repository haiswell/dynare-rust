% dump_newrat_iterates.m -- reference values for tests/estimation_newrat_iterates.rs: Dynare's
% newrat.m (mode_compute=5) stopped after k = 1, 2, ... iterations (optim = ('MaxIter', k))
% and run to convergence, on ratfit_loop.m (an objective that Rust evaluates bit for bit
% the same), written to ref_newrat_iterates.json.
% Run from a private copy of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_newrat_iterates.m')"
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
dynare_config;
% name, x0, lb, ub, with_dlik, robust, gradient_epsilon, htol_rel, ftol
% (the unconstrained optimum is about [0.207; 2.032; -0.0625; 0.478])
x0 = [1; 1; 0.1; 0.3];
lb = [-5; -5; -0.5; -5]; ub = [5; 5; 2; 5];
lbb = [-5; -5; 0; -5]; ubb = [5; 1.5; 2; 5];
cases = {'base',   x0, lb,  ub,  1, false, 1e-6, NaN,  1e-5;
         'bound',  x0, lbb, ubb, 1, false, 1e-6, NaN,  1e-5;
         'robust', x0, lbb, ubb, 1, true,  1e-5, NaN,  1e-4;
         'rel',    x0, lb,  ub,  1, false, 1e-6, 1e-6, 1e-5;
         'bfgs',   x0, lb,  ub,  0, false, 1e-6, NaN,  1e-5};
kmax = [1 2 3 4 5 6 8 10 15 1000];
out = struct();
for c = 1:size(cases, 1)
    [name, x0, lb, ub, wd, robust, geps, hrel, ftol] = cases{c,:};
    n = length(x0);
    runs = {};
    for k = kmax
        hess_info = struct();
        hess_info.gstep = [1e-2; 1]; hess_info.htol = ftol; hess_info.htol_rel = hrel;
        hess_info.h1 = geps*ones(n,1); hess_info.robust = robust;
        [xp, hh, gg, fval, igg, hi, jit, fcount, exitflag, msg] = newrat(@ratfit_loop, x0, [lb ub], 0, ftol, k, 0, 0, 0, hess_info, NaN(n,1), geps, {'a','b','c','d'}, wd);
        runs{end+1} = struct('maxiter', k, 'x', xp, 'f', fval, 'g', gg, 'hh', hh, 'igg', igg, 'h1', hi.h1, 'htol', hi.htol, 'it', jit, 'fc', fcount, 'ef', exitflag, 'message', msg);
    end
    out.(name) = struct('x0', x0, 'lb', lb, 'ub', ub, 'with_dlik', wd, 'robust', robust, 'gradient_epsilon', geps, 'htol_rel', hrel, 'ftol', ftol, 'runs', {runs});
end
fid = fopen('ref_newrat_iterates.json', 'w'); fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true)); fclose(fid);
