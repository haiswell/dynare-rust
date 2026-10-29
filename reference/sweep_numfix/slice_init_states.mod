// estimation/slice/fs2000_estimate_initial_states_log.mod of Dynare 7.2's tests with fewer draws:
// the slice sampler without slice_initialize_with_mode (mode_compute=0, smoother) runs its MCMC,
// prior_trunc = 0 raised to 1e-10 by dynare_estimation_init.m
@#include "fs2000_slice.inc"

estimation(order=1,datafile=fsdat_simul,nobs=192,loglinear,
    mh_replic=10,mh_nblocks=2,mh_drop=0.2,
    mode_compute=0,
    posterior_sampling_method='slice',
    estimate_initial_states_endogenous_prior,
    smoother, filtered_vars, consider_all_endogenous
);
