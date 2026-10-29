// New Keynesian model with a zero lower bound on the policy rate (one OccBin constraint),
// estimated with OccBin's piecewise Kalman filter (PKF) or inversion filter (IVF) on data
// simulated from the piecewise-linear model itself (nk_est_data.csv, written by
// dump_occbin_est.m). Macro variables:
//   GEN       only the model (used to simulate the data)
//   IVF       observables y, pi, inot (as many as shocks, no measurement error) and
//             occbin_setup(likelihood_inversion_filter, smoother_inversion_filter);
//             otherwise y, pi, i with a measurement error on i and the PKF
//   RELAX     filter_use_relaxation
//   PARTICLE  the filter_particle_* options of occbin_setup (ignored by Dynare 7.2)

var y pi i inot r_nat u_s;
varexo eps_d eps_s eps_m;
parameters beta sigma kappa phi_pi phi_y rho_i rho_d rho_s istar ilb;

beta   = 0.99;
sigma  = 1;
kappa  = 0.1;
phi_pi = 1.5;
phi_y  = 0.125;
rho_i  = 0.6;
rho_d  = 0.85;
rho_s  = 0.5;
istar  = 1/beta - 1;
ilb    = 0;

model;
[name='IS']
y = y(+1) - 1/sigma*(i - pi(+1) - istar - r_nat);
[name='PC']
pi = beta*pi(+1) + kappa*y + u_s;
r_nat = rho_d*r_nat(-1) + eps_d;
u_s = rho_s*u_s(-1) + eps_s;
[name='notional rate']
inot = rho_i*i(-1) + (1-rho_i)*(istar + phi_pi*pi + phi_y*y) + eps_m;
[name='mp', relax='zlb']
i = inot;
[name='mp', bind='zlb']
i = ilb;
end;

occbin_constraints;
name 'zlb'; bind inot < ilb; relax inot > ilb;
end;

initval;
y = 0; pi = 0; i = istar; inot = istar; r_nat = 0; u_s = 0;
end;
steady;

shocks;
var eps_d; stderr 0.01;
var eps_s; stderr 0.002;
var eps_m; stderr 0.002;
end;

@#ifndef GEN
@#ifdef IVF
varobs y pi inot;
@#else
varobs y pi i;
shocks;
var i; stderr 0.0005;
end;
@#endif

estimated_params;
kappa, gamma_pdf, 0.1, 0.03;
phi_pi, normal_pdf, 1.5, 0.2;
rho_i, beta_pdf, 0.6, 0.1;
rho_d, beta_pdf, 0.85, 0.05;
stderr eps_d, inv_gamma_pdf, 0.01, inf;
stderr eps_s, inv_gamma_pdf, 0.002, inf;
stderr eps_m, inv_gamma_pdf, 0.002, inf;
@#ifndef IVF
@#ifndef MCMC
stderr i, inv_gamma_pdf, 0.0005, inf;
@#endif
@#endif
end;

@#ifdef IVF
occbin_setup(likelihood_inversion_filter, smoother_inversion_filter);
@#else
@#ifdef RELAX
occbin_setup(filter_use_relaxation);
@#elseif defined(PARTICLE)
// the OccBin particle-filter options: parsed by the preprocessor into options_occbin_, but
// not forwarded to options_.occbin by Dynare 7.2's occbin.setup (dump_occbin_particle.m)
occbin_setup(particle_filtering, filter_init_periods_using_particles,
             filter_particle_number_of_particles=50, filter_particle_number_of_shocks_per_particle=3,
             filter_particle_draw_states_from_empirical_density=false,
             filter_particle_state_importance_sampling_pkf_init=false,
             filter_particle_state_importance_sampling_logpost_crit_threshold=7,
             filter_particle_state_importance_sampling_slice_override_iteration=20,
             filter_particle_state_importance_sampling_slice_burnin=3,
             filter_particle_initial_state_ergodic_simul, filter_particle_diagnostics,
             filter_particle_diagnostics_graph_periods=[2 3], filter_particle_diagnostics_nograph,
             filter_particle_state_draws=[1 2 3], filter_particle_use_pkf_updated_state_threshold=2);
@#elseif defined(SETUP_OPTS)
// options +occbin/setup.m does not forward in Dynare 7.2 (no effect; dump_occbin_setup_opts.m)
occbin_setup(smoother_max_check_ahead_periods=2, smoother_plot, smoother_first_period_occbin_update=5, smoother_max_number_of_iterations=1,
             posterior_importance_sampling, posterior_importance_sampling_filter='ppf', posterior_importance_sampling_orig_filter='pkf',
             posterior_importance_sampling_sub_draws=50, posterior_importance_sampling_orig_dname='orig', posterior_importance_sampling_orig_fname='orig_est');
@#else
occbin_setup;
@#endif
@#endif

@#ifdef MODE
// mode finding (csminwel) with the OccBin likelihood, compared in tests/occbin_est.rs
estimation(datafile='nk_est_data.csv', mode_compute=4, mh_replic=0, nograph, plot_priors=0) y pi i inot r_nat u_s;
@#elseif defined(MCMC)
// mode finding + a short Metropolis-Hastings run with the OccBin likelihood (dynare-rust demo)
estimation(datafile='nk_est_data.csv', mode_compute=6, optim=('nclimb-mh', 2000, 'ncov-mh', 2000, 'nscale-mh', 2000), mh_replic=2000, mh_nblocks=2, smoother, nograph, plot_priors=0) y pi i inot r_nat u_s;
occbin_write_regimes(smoother);
@#else
@#ifndef NOEST
estimation(datafile='nk_est_data.csv', mode_compute=0, mh_replic=0, smoother, nograph) y pi i inot r_nat u_s;
@#endif
@#endif
@#endif
