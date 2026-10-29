heterogeneity_dimension households;

var(heterogeneity=households)
   c      // consumption
   n      // labor supply
   ns     // effective labor supply
   a      // assets
;

varexo(heterogeneity=households)
   e      // idiosyncratic efficiency
;

var
   Y L w pi Div Tax r
;

varexo G markup rstar;

shocks;
    var G; stderr 0.01;
    var markup; stderr 0.01;
    var rstar; stderr 0.01;
end;

parameters
   beta vphi
   eis frisch
   rho_e sig_e
   rho_Z sig_Z
   mu kappa phi
   Z B r_ss
;

model(heterogeneity=households);
   // Euler equation with borrowing constraint
   c^(-1/eis) - beta * (1 + r(+1)) * c(+1)^(-1/eis) = 0 ⟂ a >= 0;

   // Budget constraint
   (1 + r) * a(-1) + w * n * e + (Div-Tax) * e - c - a;

   // Intratemporal FOC for labor supply
   vphi*n^(1/frisch) - w*e*c^(-1/eis);

   // Definition of the effective labor supply
   ns = n * e;
end;

model;
   // Firm
   L - Y / Z;
   Div - (Y - w * L - mu / (mu - 1) / (2 * kappa) * log(1 + pi)^2 * Y);

   // Monetary policy (Taylor rule)
   (1 + r_ss + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1 - r;

   // Fiscal policy
   Tax - (r * B) - G;

   // NKPC
   kappa * (w / Z - 1 / mu)
   + Y(+1)/Y * log(1 + pi(+1)) / (1 + r(+1))
   + markup
   - log(1 + pi);

   // Market clearing
   [name='capital_market_clearing']
   SUM(a) - B;  // asset market
   sum(ns) - L; // labor market
end;

load 'hank_1a_sp.mat';

param_names = fieldnames(steady_state.params);
for i=1:numel(param_names)
    param = param_names{i};
    set_param_value(param, steady_state.params.(param));
end

heterogeneity_compute_steady_state(filename=hank_1a_sp, calibration_target_equations=['capital_market_clearing'], time_iteration_solver_stop_on_error,time_iteration_verbosity=0,forward_verbosity=0);

heterogeneity_solve;

if max(abs(oo_.heterogeneity.mat.G(:))) > 1e-4
    error('Aggregate-block residuals are too big!');
end
if max(abs(oo_.heterogeneity.mat.F(:))) > 1e-8
    error('Heterogeneous-block residuals are too big');
end

heterogeneity_simulate;