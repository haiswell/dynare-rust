heterogeneity_dimension households;

var(heterogeneity=households)
    c  // Consumption
    a  // Assets
    Va // Derivative of the value function w.r.t assets
;

varexo(heterogeneity=households)
    e u
;

var
    Y // Aggregate output
    r // Rate of return on capital net of depreciation
    w // Wage rate
    K // Aggregate capital
;

varexo Z; // Aggregate productivity shock

parameters
    L     // Labor
    alpha // Share of capital in production fuction
    beta  // Subjective discount rate of houselholds
    delta // Capital depreciation rate
    eis   // Elasticity of intertemporal substitution
    Z_ss  // Aggregate TFP shock average value
;

model(heterogeneity=households);
    c^(-1/eis)-beta*Va(+1)=0 ⟂ a>=0;
    (1+r)*a(-1)+w*e*u-c-a;
    Va = (1+r)*c^(-1/eis);
end;

model;
    [name='production']
    (Z_ss+Z) * K(-1)^alpha * L^(1 - alpha) - Y;
    alpha * (Z_ss+Z) * (K(-1) / L)^(alpha - 1) - delta - r;
    (1 - alpha) * (Z_ss+Z) * (K(-1) / L)^alpha - w;
    [name='capital_market_clearing']
    K - SUM(a);
end;

shocks;
    var Z; stderr 0.01;
end;

load 'ks_two_shocks.mat';

param_names = fieldnames(steady_state.params);
for i=1:numel(param_names)
    param = param_names{i};
    set_param_value(param, steady_state.params.(param));
end

heterogeneity_compute_steady_state(filename=ks_two_shocks);

heterogeneity_solve;

if max(abs(oo_.heterogeneity.mat.G(:))) > 1e-4
    error('Aggregate-block residuals are too big!');
end
if max(abs(oo_.heterogeneity.mat.F(:))) > 1e-8
    error('Heterogeneous-block residuals are too big');
end

heterogeneity_simulate;