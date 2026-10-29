% dump_occbin_particle.m -- Dynare 7.2 reference for the OccBin particle-filter options of
% occbin_setup (tests/occbin_particle_options.rs). nk_est -DPARTICLE passes every
% filter_particle_* option, particle_filtering and filter_init_periods_using_particles: the
% preprocessor writes them into options_occbin_, but occbin.setup forwards only its fixed list
% (filter.use_relaxation for the filter), so options_.occbin.filter keeps its defaults and the
% PKF likelihood is the default one. Records options_occbin_.filter, options_.occbin.filter
% and dsge_likelihood at the points of dump_occbin_est.m. Written to ref_occbin_particle.json.
%
% Run from a PRIVATE COPY of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_occbin_particle.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare nk_est -DPARTICLE noclearall nograph nolog
global M_ options_ oo_ estim_params_ bayestopt_
out = struct();
out.options_occbin_filter = options_occbin_.filter;
out.options_occbin_filter_particle = options_.occbin.filter.particle;
out.options_occbin_filter_scalars = rmfield(options_.occbin.filter, 'particle');
[dataset_, dataset_info, xparam_init, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
out.xparam_init = xparam_init;
pts = {xparam_init};
rng(7);
for k = 1:5
    pts{end+1} = xparam_init .* (1 + 0.1*randn(size(xparam_init)));
end
out.points = {};
for k = 1:numel(pts)
    x = pts{k};
    [fval, info] = dsge_likelihood(x, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
        oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    out.points{end+1} = struct('x', x, 'fval', fval, 'info', info(1));
    fprintf('point %d: fval %.12g info %d\n', k, fval, info(1));
end
fid = fopen(fullfile(here, 'ref_occbin_particle.json'), 'w');
fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true));
fclose(fid);
disp('wrote ref_occbin_particle.json');
