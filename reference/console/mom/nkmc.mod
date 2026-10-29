@#define CASE = 2
// Small New Keynesian model for the IRF-matching checks of dynare-rust (reference/mom/dump_irf.m).
//   -DGEN      : IRFs at the "true" calibration (dump_irf.m turns them into nk_irf_data.inc)
//   -DCASE=1   : frequentist IRF matching (newrat, then fminsearch), diagonal and off-diagonal weights
//   -DCASE=2   : Bayesian IRF matching (posterior mode, Laplace approximation)
//   -DCASE=3   : frequentist, relative_irf
@#ifndef CASE
@#define CASE = 1
@#endif
var y pi i a d;
varexo e_a e_d e_i;
parameters beta sigma kappa phi_pi phi_y rho_a rho_d rho_i;
beta = 0.99; sigma = 1.5; kappa = 0.12; phi_pi = 1.6; phi_y = 0.4; rho_a = 0.8; rho_d = 0.7; rho_i = 0.6;
model(linear);
y = y(+1) - (1/sigma)*(i - pi(+1)) + d;
pi = beta*pi(+1) + kappa*y + a;
i = rho_i*i(-1) + (1-rho_i)*(phi_pi*pi + phi_y*y) + e_i;
a = rho_a*a(-1) + e_a;
d = rho_d*d(-1) + e_d;
end;
shocks;
var e_a; stderr 0.5;
var e_d; stderr 0.8;
var e_i; stderr 0.25;
end;
steady;

@#ifdef GEN
stoch_simul(order=1, irf=12, nograph, noprint, nomoments);
@#else
varobs y pi i;

@#if CASE == 1 || CASE == 3
estimated_params;
stderr e_i, 0.2, 0.01, 2;
kappa, 0.1, 0.001, 1;
phi_pi, 1.3, 1.01, 3;
rho_a, 0.7, 0, 0.99;
rho_i, 0.5, 0, 0.99;
end;
@#endif
@#if CASE == 2
estimated_params;
stderr e_i, 0.2, inv_gamma_pdf, 0.25, 1;
kappa, 0.1, gamma_pdf, 0.12, 0.05;
phi_pi, 1.3, normal_pdf, 1.5, 0.25;
rho_a, 0.7, beta_pdf, 0.7, 0.1;
rho_i, 0.5, beta_pdf, 0.6, 0.1;
end;
@#endif

@#include "nk_irf_data.inc"

matched_irfs_weights;
y(1), e_i, pi(1), e_i, 2;
y(2:3), e_a, y(3:4), e_a, (0.5);
end;

@#if CASE == 1
method_of_moments(mom_method = irf_matching, mode_compute = 5, additional_optimizer_steps = [7], nograph);
@#endif
@#if CASE == 2
method_of_moments(mom_method = irf_matching, mode_compute = 5, nograph, plot_priors = 0, mh_replic = 1000, mh_nblocks = 2, mh_jscale = 0.5, raftery_lewis_diagnostics);
@#endif
@#if CASE == 3
method_of_moments(mom_method = irf_matching, mode_compute = 5, relative_irf, nograph);
@#endif
@#endif
