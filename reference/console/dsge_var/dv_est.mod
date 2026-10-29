var y $y$ (long_name='output gap') pi $\pi$ (long_name='inflation') i $i$ (long_name='nominal_rate');
varexo e_y $\varepsilon_y$ (long_name='demand shock') e_pi $\varepsilon_\pi$ e_i;
parameters rho_y $\rho_y$ (long_name='output persistence') beta $\beta$ kappa $\kappa$ phi_pi $\phi_\pi$ (long_name='Taylor 100% rule') rho_i;
rho_y = 0.7;
beta = 0.99;
kappa = 0.1;
phi_pi = 1.5;
rho_i = 0.6;

model(linear);
# r = i - pi(+1);
[name='IS curve']
y = rho_y*y(-1) + 0.1*y(-2) - 0.1*r + e_y;
[name='Phillips curve', mcp='pi>0']
pi = 0.5*beta*pi(+1) + 0.4*pi(-1) + kappa*y + e_pi;
[name='Taylor rule']
i = rho_i*i(-1) + (1-rho_i)*(phi_pi*pi + 0.5*y) + e_i;
end;

shocks;
var e_y; stderr 1;
var e_pi; stderr 0.5;
var e_i; stderr 0.25;
end;
estimated_params;
rho_y, beta_pdf, 0.6, 0.15;
kappa, gamma_pdf, 0.15, 0.05;
phi_pi, 1.4, 1, 3, normal_pdf, 1.5, 0.25;
rho_i, beta_pdf, 0.5, 0.2;
stderr e_y, inv_gamma_pdf, 0.8, inf;
stderr e_pi, inv_gamma_pdf, 0.5, inf;
stderr e_i, uniform_pdf, , , 0, 2;
dsge_prior_weight, 1.5, uniform_pdf, , , 0.5, 4;
end;
varobs y pi i;
estimation(datafile = nk_data, first_obs = 5, dsge_var, dsge_varlag = 2, mode_compute = 4, mh_replic = 1500, mh_nblocks = 2, mh_jscale = 0.5, bayesian_irf, irf = 8, moments_varendo, smoother, filtered_vars, nodiagnostic) y;
