// Small New Keynesian model for the global sensitivity analysis reference
// (dump_gsa.m). The prior ranges contain determinate, indeterminate
// (phi_pi < 1) and explosive (rho_g > 1) regions.
// STAGE: 0 simulate data; 1 stability map; 2 stability map, prior_range=0;
// 3 reduced form, threshold; 4 Morris screening; 5 RMSE; 6 RMSE lik_only;
// 7 reduced form mapping without threshold (needs SS-ANOVA-R in MATLAB).

@#ifndef STAGE
@#define STAGE = 1
@#endif

var y pi i g u;
varexo e_g e_u e_i;
parameters sigma kappa beta phi_pi phi_y rho_i rho_g rho_u;

sigma = 1;
kappa = 0.1;
beta = 0.99;
phi_pi = 1.5;
phi_y = 0.125;
rho_i = 0.7;
rho_g = 0.9;
rho_u = 0.5;

model(linear);
y = y(+1) - sigma*(i - pi(+1)) + g;
pi = beta*pi(+1) + kappa*y + u;
i = rho_i*i(-1) + (1-rho_i)*(phi_pi*pi + phi_y*y) + e_i;
g = rho_g*g(-1) + e_g;
u = rho_u*u(-1) + e_u;
end;

shocks;
var e_g; stderr 0.5;
var e_u; stderr 0.3;
var e_i; stderr 0.2;
end;

steady;

varobs y pi i;

estimated_params;
stderr e_g, inv_gamma_pdf, 0.5, inf;
stderr e_u, inv_gamma_pdf, 0.3, inf;
stderr e_i, inv_gamma_pdf, 0.2, inf;
sigma, gamma_pdf, 1, 0.3;
kappa, gamma_pdf, 0.1, 0.05;
phi_pi, uniform_pdf, , , 0.5, 3;
phi_y, normal_pdf, 0.125, 0.05;
rho_i, beta_pdf, 0.7, 0.1;
rho_g, uniform_pdf, , , 0.5, 1.05;
rho_u, beta_pdf, 0.5, 0.2;
end;

@#ifdef DYNARE_MATLAB
// Dynare 7.1's reduced_form_mapping/screening index namendo_tex, which run.m only fills
// with TeX on (otherwise "Index exceeds array bounds"); dump_gsa.m defines DYNARE_MATLAB.
options_.TeX = 1;
@#endif

@#if STAGE == 0
stoch_simul(order=1, periods=250, drop=50, irf=0, nomoments, noprint, nograph);
@#elseif STAGE == 1
sensitivity(nodisplay, graph_format=none);
@#elseif STAGE == 2
sensitivity(prior_range=0, Nsam=1024, alpha2_stab=0.3, pvalue_ks=0.01, pvalue_corr=0.01, nodisplay, graph_format=none);
@#elseif STAGE == 3
sensitivity(redform=1, namendo=(y,pi), namlagendo=(i,g), namexo=(e_i,e_g), threshold_redform=[-2 -0.5], ksstat_redform=0.01, nodisplay, graph_format=none);
@#elseif STAGE == 4
sensitivity(morris=1, namendo=(y,pi), namlagendo=(i,g,u), namexo=(e_g,e_u,e_i), nodisplay, graph_format=none);
@#elseif STAGE == 5
sensitivity(datafile='nk_gsa_data.csv', rmse=1, Nsam=512, nodisplay, graph_format=none);
@#elseif STAGE == 6
sensitivity(datafile='nk_gsa_data.csv', rmse=1, lik_only=1, Nsam=512, nodisplay, graph_format=none);
@#elseif STAGE == 8
sensitivity(datafile='nk_gsa_data.csv', rmse=1, Nsam=512, prefilter=1, conf_sig=0.8, use_univariate_smoother_if_singularity_is_detected, use_univariate_filters_if_singularity_is_detected=0, nodisplay, graph_format=none);
@#elseif STAGE == 7
sensitivity(redform=1, namendo=(y,pi), namlagendo=(i), namexo=(e_i), nodisplay, graph_format=none);
@#endif
