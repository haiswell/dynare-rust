% tcm6c/tcm7c/tcm8c: Dynare's tests/trend-component-and-var-models/tcm6-8.mod with the randn
% calibration of tcm6.mod written out. Run from a private copy of reference/pac:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_tcm.m')"
addpath('/Applications/Dynare/7.2-arm64/matlab');
for f = {'tcm6c','tcm7c','tcm8c'}
    name = f{1};
    clear M_ oo_ options_
    dynare(name, 'nolog', 'nograph', 'nointeractive');
    r = struct();
    r.param_names = M_.param_names;
    r.params = M_.params;
    r.endo_names = M_.endo_names;
    r.trend_component = oo_.trend_component;
    for m = {'toto','titi'}
        [AR, A0, A0star] = feval([M_.fname '.trend_component_ar_a0'], m{1}, M_.params);
        r.ar_a0.(m{1}) = struct('AR', AR, 'A0', A0, 'A0star', A0star);
    end
    fid = fopen(['ref_' name '.json'], 'w'); fprintf(fid, '%s', jsonencode(r)); fclose(fid);
end
