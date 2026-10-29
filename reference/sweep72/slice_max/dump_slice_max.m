% The slice sampler's maximize option (tests/sweep72_features.rs): runs the two estimations of
% fs2000_slice_max.mod with -DPREP, keeps their MCMC files (metropolis/, the test's starting
% point), then the third estimation, and dumps the chains' last draws before and after it.
% Run from a copy of this folder with ../../sweep_numfix/fsdat_simul.m next to the .mod:
% (optim converged, MaxIter 200 / TolFun 1e-9: newrat's early iterates differ from MATLAB's at ~1e-4)
% matlab -batch "addpath('<dynare>/matlab'); dump_slice_max"
dynare('fs2000_slice_max', '-DPREP', 'nograph', 'nointeractive');
copyfile(['fs2000_slice_max' filesep 'metropolis'], 'metropolis');
r = struct();
record = load_last_mh_history_file(['fs2000_slice_max' filesep 'metropolis'], 'fs2000_slice_max');
r.start_parameters = record.LastParameters;
r.start_logpost = record.LastLogPost;
dynare('fs2000_slice_max', 'nograph', 'nointeractive');
record = load_last_mh_history_file(['fs2000_slice_max' filesep 'metropolis'], 'fs2000_slice_max');
r.last_parameters = record.LastParameters;
r.last_logpost = record.LastLogPost;
fid = fopen('fs2000_slice_max.json', 'w');
fprintf(fid, '%s', jsonencode(r, 'ConvertInfAndNaN', false));
fclose(fid);
