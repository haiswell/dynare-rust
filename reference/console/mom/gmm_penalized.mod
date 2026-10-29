@#define CASE = 6
// RBC model for the method-of-moments checks of dynare-rust (reference/mom/dump_mom.m).
// The model of reference/korder/rbc.mod; yo, co, lo, io are 100*log of y, c, l, i relative to
// their steady state at the calibration (fixed constants). lo is not observed: at order 1 the
// labour FOC makes yo, co, lo collinear (stochastic singularity of the moment conditions).
//   -DGEN=<1|2>    : simulate 400 periods at order 1 / 2 (saved as rbc_mom_data<GEN>.csv by dump_mom.m)
//   -DCASE=<n>     : one method_of_moments run: 1 GMM order 1, 2 GMM order 2, 3 GMM order 3,
//                  4 SMM order 1, 5 SMM order 2, 6 penalized GMM with priors
@#ifndef CASE
@#define CASE = 1
@#endif
// -DANALYTIC=1 adds analytic_standard_errors (and analytic_jacobian for the mode_compute = 4 GMM
// cases 3 and 6) to the GMM method_of_moments calls (dump_analytic.m).
@#define AN = ""
@#ifdef ANALYTIC
@#if CASE == 1 || CASE == 2
@#define AN = ", analytic_standard_errors"
@#endif
@#if CASE == 3 || CASE == 6
@#define AN = ", analytic_standard_errors, analytic_jacobian"
@#endif
@#endif
var y c k l i z g yo co lo io;
varexo ez eg;
parameters beta alpha delta sigma psi rho rhog gbar;
beta = 0.99; alpha = 0.33; delta = 0.025; sigma = 2; psi = 2;
rho = 0.95; rhog = 0.9; gbar = 0.2;
model;
c^(-sigma) = beta*c(+1)^(-sigma)*(alpha*exp(z(+1))*k^(alpha-1)*l(+1)^(1-alpha) + 1 - delta);
psi*c^sigma/(1-l) = (1-alpha)*exp(z)*k(-1)^alpha*l^(-alpha);
y = exp(z)*k(-1)^alpha*l^(1-alpha);
k = i + (1-delta)*k(-1);
y = c + i + g;
z = rho*z(-1) + ez;
g = (1-rhog)*gbar + rhog*g(-1) + eg;
yo = 100*log(y/1.2630296059803614);
co = 100*log(c/0.7661722158553375);
lo = 100*log(l/0.4188697641102882);
io = 100*log(i/0.29685739012502393);
end;
initval;
l = 0.4188697641102882; k = 11.874295605000956; y = 1.2630296059803614;
i = 0.29685739012502393; c = 0.7661722158553375; z = 0; g = 0.2;
yo = 0; co = 0; lo = 0; io = 0;
end;
steady;
shocks;
var ez; stderr 0.01;
var eg; stderr 0.005;
end;

@#ifdef GEN
stoch_simul(order=@{GEN}, periods=400, drop=200, irf=0, nograph, noprint, nomoments);
@#else

varobs yo co io;

@#if CASE == 1 || CASE == 2 || CASE == 3 || CASE == 4 || CASE == 5
estimated_params;
rho, 0.92;
stderr ez, 0.009;
stderr eg, 0.006;
psi, 2.1, 0.5, 5;
end;
@#endif

@#if CASE == 6
estimated_params;
rho, 0.92, , , beta_pdf, 0.9, 0.05;
stderr ez, 0.009, , , inv_gamma_pdf, 0.01, 0.01;
psi, 2.1, , , normal_pdf, 2, 0.5;
end;
@#endif

matched_moments;
yo;
co;
io;
yo*yo;
co*co;
io*io;
yo*co;
yo*io;
co*io;
yo*yo(-1);
co*co(-1);
io(-1)*io;
@#if CASE == 5
yo^3;
co*io(1)^2;
@#endif
end;

@#if CASE == 1
method_of_moments(mom_method = GMM, datafile = 'rbc_mom_data1.csv', order = 1, mode_compute = 5,
                  weighting_matrix = ['DIAGONAL', 'OPTIMAL'], se_tolx = 1e-6, nograph@{AN});
@#endif
@#if CASE == 2
method_of_moments(mom_method = GMM, datafile = 'rbc_mom_data2.csv', order = 2, pruning, prefilter = 1,
                  mode_compute = 5, weighting_matrix = ['OPTIMAL'], bartlett_kernel_lag = 10, nograph@{AN});
@#endif
@#if CASE == 3
method_of_moments(mom_method = GMM, datafile = 'rbc_mom_data2.csv', order = 3, pruning, prefilter = 1,
                  mode_compute = 4, weighting_matrix = ['DIAGONAL'], first_obs = 21, nobs = 300, nograph@{AN});
@#endif
@#if CASE == 4
method_of_moments(mom_method = SMM, datafile = 'rbc_mom_data1.csv', order = 1, mode_compute = 5,
                  weighting_matrix = ['DIAGONAL', 'OPTIMAL'], simulation_multiple = 5, burnin = 200, seed = 1234,
                  nograph);
@#endif
@#if CASE == 5
method_of_moments(mom_method = SMM, datafile = 'rbc_mom_data2.csv', order = 2, pruning, prefilter = 1, mode_compute = 4,
                  weighting_matrix = ['IDENTITY_MATRIX', 'OPTIMAL'], weighting_matrix_scaling_factor = 100,
                  simulation_multiple = 3, bounded_shock_support, nograph);
@#endif
@#if CASE == 6
method_of_moments(mom_method = GMM, datafile = 'rbc_mom_data1.csv', order = 1, mode_compute = 4,
                  weighting_matrix = ['IDENTITY_MATRIX', 'DIAGONAL'], weighting_matrix_scaling_factor = 10,
                  bartlett_kernel_lag = 5, nograph@{AN});
@#endif
@#endif
