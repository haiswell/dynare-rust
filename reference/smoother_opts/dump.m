% Reference for tests/smoother_cmd_options.rs (Dynare 7.2): calib_smoother / shock_decomposition
% with prefilter=1 and the use_univariate_* options on a linear NK model.
% Run from a private copy of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump.m')"
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', '7.2'); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath')); cd(here);
dynare nk_sm nointeractive nograph
R.smoothed = struct();
for v = {'y','pi','i','g','u'}
  R.smoothed.(v{1}) = oo_.SmoothedVariables.(v{1});
end
R.shock_decomposition = oo_.shock_decomposition;   % endo x (shocks+2) x T, DR order of M_.endo_names
R.endo_names = M_.endo_names;
R.exo_names = M_.exo_names;
fid = fopen('ref.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
