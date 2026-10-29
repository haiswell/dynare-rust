heterogeneity_dimension households;

var(heterogeneity=households)
   b
   a
   c
   Va
   Vb
   u
;

varexo(heterogeneity=households)
   e
;

var
    piw
    psiw
    rb
    ra
    tax
    i
    psip
    I
    Q
    w
    N
    K
    div
    p
    pi
    mc
    r
    Y
;

varexo
    rstar
    markup
    G
    beta
    Z
    rinv_shock
    markup_w
;

shocks;
    var rstar; stderr 0.01;
    var markup; stderr 0.01;
    var G; stderr 0.01;
    var beta; stderr 0.01;
    var Z; stderr 0.01;
    var rinv_shock; stderr 0.01;
    var markup_w; stderr 0.01;
end;

parameters
    kappap
    alpha
    epsI
    muw
    phi
    omega
    Bg
    pshare
    delta
    kappaw
    frisch
    mup
    vphi
    eis
    chi0
    chi1
    chi2
    Z_ss
    beta_ss
    r_ss
    G_ss
;


model(heterogeneity=households);
   // Euler equations
   c^(-1/eis) - (beta_ss+beta) * Vb(+1) = 0 ⟂ b >= 0;
   c^(-1/eis)*(1 + chi1 * sign(a-(1+ra)*a(-1)) * (abs(a-(1+ra)*a(-1))/((1+ra)*a(-1)+chi0))^(chi2-1)) - (beta_ss+beta) * Va(+1) = 0 ⟂ a >= 0;

   // Budget constraint
   (1 + ra) * a(-1) + (1 + rb) * b(-1) - (chi1 / chi2) * abs(a-(1+ra)*a(-1))^chi2 * ((1+ra)*a(-1)+chi0)^(1-chi2) + (1-tax) * w * N * e - c - a - b;

   // Effective labor definition
   u = e * c^(-1/eis);

   // Envelope conditions
   Va = (1 + ra)*(1 - (chi1/chi2) * ( - chi2 * sign(a-(1+ra)*a(-1)) * (abs(a-(1+ra)*a(-1))/((1+ra)*a(-1)+chi0))^(chi2-1) + (1-chi2) * (abs(a-(1+ra)*a(-1))/((1+ra)*a(-1)+chi0))^chi2 )) * c^(-1/eis);
   Vb = (1 + rb) * c^(-1/eis);
end;

model;
   // NKPC
   kappap * (mc - 1 / mup) + Y(+1) / Y * log(1 + pi(+1)) / (1 + r(+1)) + markup - log(1 + pi);

   // Equity price
   div(+1) + p(+1) - p * (1 + r(+1));

   // Production function
   N = (Y / (Z_ss+Z) / K(-1) ^ alpha) ^ (1 / (1 - alpha));

   // Labor demand
   mc = w * N / (1 - alpha) / Y;

   // Tobin's Q
   (K / K(-1) - 1) / (delta * epsI) + 1 - Q;

   // Valuation equation
   alpha * (Z_ss+Z(+1)) * (N(+1) / K) ^ (1 - alpha) * mc(+1) - (K(+1) / K -
   (1 - delta) + (K(+1) / K - 1) ^ 2 / (2 * delta * epsI)) + K(+1) / K * Q(+1) - (1 + r(+1) + rinv_shock) * Q;

   // Price adjustment cost
   mup / (mup - 1) / 2 / kappap * log(1 + pi) ^ 2 * Y - psip;

   // Aggregate investment
   K - (1 - delta) * K(-1) + K(-1) * (K / K(-1) - 1) ^ 2 / (2 * delta * epsI) - I;

   // Taylor rule
   rstar + r_ss + phi * pi - i;

   // Fiscal policy
   (r * Bg + G_ss + G) / w / N - tax;

   // Returns on liquid asset
   r - omega - rb;

   // Returns on illiquid asset
   pshare * (div + p) / p(-1) + (1 - pshare) * (1 + r) - 1 - ra;

   // Fisher equation
   1 + i(-1) - (1 + r) * (1 + pi);

   // Wage inflation
   (1 + pi) * w / w(-1) - 1 - piw;

   // Wage adjustment cost
   muw / (1 - muw) / 2 / kappaw * log(1 + piw) ^ 2 * N - psiw;

   // Wage NKPC
   [name='wage_nkpc']
   kappaw * (vphi * N ^ (1 + 1 / frisch) - (1 - tax) * w * N * SUM(u) / muw) + (beta_ss+beta) * log(1 + piw(+1)) + markup_w - log(1 + piw);

   // Total asset market clearing
   [name='illiquid_asset_market_clearing']
   p - SUM(a);

   // Liquid asset clearing
   [name='liquid_asset_market_clearing']
   Bg - SUM(b);
end;

load 'hank_2a_sp.mat';

param_names = fieldnames(steady_state.params);
for i=1:numel(param_names)
   param = param_names{i};
   set_param_value(param, steady_state.params.(param));
end

heterogeneity_compute_steady_state(filename=hank_2a_sp, calibration_target_equations=['wage_nkpc', 'liquid_asset_market_clearing', 'illiquid_asset_market_clearing'], time_iteration_tol=1e-10, time_iteration_learning_rate=0.79, time_iteration_solver_tolf=1e-12, time_iteration_solver_tolx=1e-14,time_iteration_verbosity=0,forward_verbosity=0,calibration_verbosity=0);

heterogeneity_solve;

if max(abs(oo_.heterogeneity.mat.G(:))) > 1e-4
    error('Aggregate-block residuals are too big!');
end
if max(abs(oo_.heterogeneity.mat.F(:))) > 6e-6
    error('Heterogeneous-block residuals are too big');
end

heterogeneity_simulate;
