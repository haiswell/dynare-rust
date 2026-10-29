// The slice sampler's maximize option (third estimation of tests/estimation/slice/fs2000_slice.mod).
// With -DPREP: the two estimations before it, which write the MCMC files (metropolis/) it continues.
@#include "fs2000.inc"

@#ifdef PREP
estimation(order=1,datafile='fsdat_simul',nobs=192,silent_optimizer,loglinear,mh_replic=50,mh_nblocks=2,mh_drop=0.2,
posterior_sampling_method='slice'
);
estimation(order=1,datafile='fsdat_simul',silent_optimizer,nobs=192,loglinear,mh_replic=100,mh_nblocks=2,mh_drop=0.5,load_mh_file,
posterior_sampling_method='slice',
posterior_sampler_options=('rotated',1,'use_mh_covariance_matrix',1)
);
@#else
estimation(order=1,datafile='fsdat_simul',silent_optimizer,nobs=192,loglinear,mh_replic=1,mh_nblocks=2,mh_drop=0.5,load_mh_file,
posterior_sampling_method='slice',
posterior_sampler_options=('maximize',1,'maximize_using_mh_bounds',1),
optim = ('MaxIter', 200, 'TolFun',1.e-9, 'robust',1));
@#endif
