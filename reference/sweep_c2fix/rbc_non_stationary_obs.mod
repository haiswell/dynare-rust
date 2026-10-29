// tests correctly exiting the diffuse filter; prevents regression against https://git.dynare.org/Dynare/dynare/-/merge_requests/2433

var log_y_q_obs yNE l mu y c k i a z;
var a_innovation yNE_innovation;

varexo eps_a eps_z eps_yNE yNE_noise;

varobs log_y_q_obs;

parameters beta psi delta alpha  rho_a  rho_z GR_a  GR_yNE rho_yNE yNE_start;

%----------------------------------------------------------------
% 2. Calibration
%----------------------------------------------------------------

alpha   = 0.33;
beta    = 0.99;
delta   = 0.023;
psi     = 1.75;
aa = 0.00403679164929413; % average growth rate in the data
GR_a   = aa-0.001;
GR_yNE = aa - GR_a;
rho_a = 0.7;
rho_z = 0.8;
yNE_start = 6.4061;
rho_yNE = 0.9;

%----------------------------------------------------------------
% 3. Model
%----------------------------------------------------------------

observation_trends;
    log_y_q_obs(GR_a + GR_yNE);
end;

model;
    [name = 'Euler']
    (1/c) = beta*(1/c(+1)/mu(+1))*(1-delta+alpha*y(+1)*mu(+1)/k);

    [name = 'Labor supply']
    psi*c/(1-l) = (1-alpha)*y/l;

    [name = 'budget constraint']
    c + i = y;

    [name = 'output']
    y = mu^(-alpha)*exp(z)*(k(-1)^alpha)*l^(1-alpha); %

    [name = 'investment']
    i = k-(1-delta)*k(-1)/mu;   %exp(eps_i)*

    [name = 'persistent productivity process']
    z = rho_z*z(-1) + eps_z;

    [name = 'productivity innovation']
    a_innovation = (1-rho_a)*GR_a + rho_a*a_innovation(-1) + eps_a;

    [name = 'level_productivity']
    a = a(-1) + (1-rho_a)*GR_a + rho_a*a_innovation(-1) + eps_a;

    [name = 'level_productivity_gr']
    log(mu) = a_innovation;

%% measurement equations
    [name = 'obs: y']
    log_y_q_obs  =  y  + a + yNE + yNE_start + yNE_noise; %

%% NotExplained components' equations
    % yNE
    [name = 'yNE innovation']
    yNE_innovation = (1-rho_yNE)*GR_yNE + rho_yNE*yNE_innovation(-1) + eps_yNE;

    [name = 'level_yNE']
    yNE = yNE(-1) + (1-rho_yNE)*GR_yNE + rho_yNE*yNE_innovation(-1) + eps_yNE;

end;


steady_state_model;
    mu_ss = exp(GR_a);
    l_over_k_ss = 1/mu_ss* (  (mu_ss-beta*(1-delta))/(alpha*beta) )^(1/(1-alpha));
    const0 = mu_ss^(-alpha)*(l_over_k_ss^(1-alpha));
    const1 = (mu_ss-1+delta)/mu_ss;
    const2 = (const0-const1)/const0;
    const3 = (1-alpha)/psi;
    const4 = const3/(const2+const3);
    l = const4;
    k = l/l_over_k_ss;
    i = (mu_ss-1+delta)/mu_ss*k;
    y = mu_ss^(-alpha)*k^alpha*l^(1-alpha);
    c = y-i;
    a_innovation = GR_a;
    z = 0;
    a = 0;
    mu =  exp(GR_a);
    yNE = 0;
    log_y_q_obs = y+yNE_start;
end;

steady(nocheck);

shocks;
    var eps_a;
    stderr 0.02;
    var eps_z;
    stderr 0.02;
    var yNE_noise;
    stderr 0.001;
    var eps_yNE;
    stderr 0.001;
end;

options_.plot_priors=false;
calib_smoother(datafile='data_rbc_non_stationary_obs',
    first_obs=2, diffuse_filter,kalman_algo = 3,nobs =99) z;

z_smoothed_3=oo_.SmoothedVariables.z;

initial_condition_decomposition  z;

shock_decomposition(diffuse_filter,kalman_algo = 3) z;
z_shock_decomposition_initial_3=squeeze(oo_.shock_decomposition(strmatch('z',M_.endo_names,'exact'),M_.exo_nbr+1,:));


calib_smoother(datafile='data_rbc_non_stationary_obs',
    first_obs=2, diffuse_filter,kalman_algo = 4,nobs =99) z;

z_smoothed_4=oo_.SmoothedVariables.z;
initial_condition_decomposition  z;

shock_decomposition(diffuse_filter,kalman_algo = 4) z;
z_shock_decomposition_initial_4=squeeze(oo_.shock_decomposition(strmatch('z',M_.endo_names,'exact'),M_.exo_nbr+1,:));

if max(abs(z_smoothed_3-z_smoothed_4))>1e-10
    error('Shock decomposition does not match')
end

if max(abs(z_shock_decomposition_initial_3-z_shock_decomposition_initial_4))>1e-10
    error('Initial decomposition does not match')
end

if max(oo_.SmoothedVariables.z(2:end)-(rho_z*oo_.SmoothedVariables.z(1:end-1)+oo_.SmoothedShocks.eps_z(2:end)))>1e-10
    error('AR process is not recovered')
end


