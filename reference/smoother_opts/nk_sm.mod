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

calib_smoother(datafile=nk_gsa_data, prefilter=1, use_univariate_smoother_if_singularity_is_detected) y pi i g u;
shock_decomposition(datafile=nk_gsa_data, prefilter=1, use_univariate_filters_if_singularity_is_detected=0, parameter_set=calibration) y pi i;
