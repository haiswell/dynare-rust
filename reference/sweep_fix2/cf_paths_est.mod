// the same after an estimation: oo_.steady_state holds the steady state at the initial point
// of the estimation (the prior means), here exp(0.0085) for gy_obs
@#include "fs2000.inc"

estimated_params;
alp, beta_pdf, 0.356, 0.02;
bet, beta_pdf, 0.993, 0.002;
gam, normal_pdf, 0.0085, 0.003;
mst, normal_pdf, 1.0002, 0.007;
rho, beta_pdf, 0.129, 0.05;
psi, beta_pdf, 0.65, 0.05;
del, beta_pdf, 0.01, 0.005;
stderr e_a, inv_gamma_pdf, 0.035449, inf;
stderr e_m, inv_gamma_pdf, 0.008862, inf;
end;

varobs gp_obs gy_obs;

estimation(order=1,datafile=fsdat_simul,nobs=192,mode_compute=0,mh_replic=0,silent_optimizer,plot_priors=0) gy_obs gp_obs;

conditional_forecast_paths;
var gy_obs;
periods  1  2  3:5;
values   (gy_obs+0.01) (gy_obs-0.02) (gy_obs);
var gp_obs;
periods 1:5;
values  (gp_obs+0.05);
end;

conditional_forecast(periods=10, parameter_set=posterior_mode, controlled_varexo=(e_a,e_m));
