// shock_decomposition/ls2003_plot.mod (Dynare 7.2 tests), calibrated at the posterior mode as in
// reference/sd_recompute/ls2003_recompute.mod: `options_.initial_date = dates('1989Q4')` dates
// the decompositions of plot_shock_decomposition (aoa from the first full year, plot_init_date
// windows, vintage labels); shock_decomposition itself keeps its own copy of options_.
var y y_s R pie dq pie_s de A y_obs pie_obs R_obs;
varexo e_R e_q e_ys e_pies e_A;

parameters psi1 psi2 psi3 rho_R tau alpha rr k rho_q rho_A rho_ys rho_pies;

psi1 = 2.0902;
psi2 = 0.1312;
psi3 = 0.2437;
rho_R = 0.7177;
alpha = 0.1308;
rr = 2.0906;
k = 0.5092;
tau = 0.4012;
rho_q = 0.3204;
rho_A = 0.4729;
rho_ys = 0.8956;
rho_pies = 0.3609;

model(linear);
y = y(+1) - (tau +alpha*(2-alpha)*(1-tau))*(R-pie(+1))-alpha*(tau +alpha*(2-alpha)*(1-tau))*dq(+1) + alpha*(2-alpha)*((1-tau)/tau)*(y_s-y_s(+1))-A(+1);
pie = exp(-rr/400)*pie(+1)+alpha*exp(-rr/400)*dq(+1)-alpha*dq+(k/(tau+alpha*(2-alpha)*(1-tau)))*y+alpha*(2-alpha)*(1-tau)/(tau*(tau+alpha*(2-alpha)*(1-tau)))*y_s;
pie = de+(1-alpha)*dq+pie_s;
R = rho_R*R(-1)+(1-rho_R)*(psi1*pie+psi2*(y+alpha*(2-alpha)*((1-tau)/tau)*y_s)+psi3*de)+e_R;
dq = rho_q*dq(-1)+e_q;
y_s = rho_ys*y_s(-1)+e_ys;
pie_s = rho_pies*pie_s(-1)+e_pies;
A = rho_A*A(-1)+e_A;
y_obs = y-y(-1)+A;
pie_obs = 4*pie;
R_obs = 4*R;
end;

shocks;
var e_R = 0.3811^2;
var e_q = 1.2854^2;
var e_A = 0.7340^2;
var e_ys = 0.8418^2;
var e_pies = 1.6151^2;
end;

varobs y_obs R_obs pie_obs dq de;

estimated_params;
psi1, 2.0902, gamma_pdf,1.5,0.5;
psi2, 0.1312, gamma_pdf,0.25,0.125;
psi3, 0.2437, gamma_pdf,0.25,0.125;
rho_R, 0.7177, beta_pdf,0.5,0.2;
alpha, 0.1308, beta_pdf,0.3,0.1;
rr, 2.0906, gamma_pdf,2.5,1;
k, 0.5092, gamma_pdf,0.5,0.25;
tau, 0.4012, gamma_pdf,0.5,0.2;
rho_q, 0.3204, beta_pdf,0.4,0.2;
rho_A, 0.4729, beta_pdf,0.5,0.2;
rho_ys, 0.8956, beta_pdf,0.8,0.1;
rho_pies, 0.3609, beta_pdf,0.7,0.15;
stderr e_R, 0.3811, inv_gamma_pdf,1.2533,0.6551;
stderr e_q, 1.2854, inv_gamma_pdf,2.5066,1.3103;
stderr e_A, 0.7340, inv_gamma_pdf,1.2533,0.6551;
stderr e_ys, 0.8418, inv_gamma_pdf,1.2533,0.6551;
stderr e_pies, 1.6151, inv_gamma_pdf,1.88,0.9827;
end;

estimation(datafile=data_ca1,mode_compute=0,first_obs=8,nobs=79,prefilter=1,mh_replic=0);

options_.initial_date=dates('1989Q4') % date arbitrarily set for testing purposes
shock_decomposition(nograph, parameter_set=calibration);
plot_shock_decomposition(fig_name=P1, detail_plot, type = aoa) y_obs R_obs;
plot_shock_decomposition(fig_name=P2, type = qoq, plot_init_date=2004Q1) y_obs R_obs;
initial_condition_decomposition(detail_plot, type=aoa, plot_init_date=1991Q1, plot_end_date=1995Q4) R_obs;
realtime_shock_decomposition(forecast=8, save_realtime=[5 9 13 17 21 25 29 33]) y_obs R_obs;
plot_shock_decomposition(fig_name=P3, realtime = 3, vintage = 29) y_obs;
