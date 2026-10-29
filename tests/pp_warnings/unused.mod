// parameters used only in steady_state_model count; those only in shocks, estimated_params,
// initval or an unused model-local variable do not
var y c;
varexo e u;
parameters a b p_ss p_shk p_init p_mlv p_est;
a=0.9; b=0.5; p_ss=0; p_shk=0.1; p_init=1; p_mlv=1; p_est=1;
model;
# unused_local = p_mlv*2;
y = a*y(-1) + e;
c = b*c(-1) + u;
end;
steady_state_model;
y = p_ss; c = 0;
end;
initval; y=p_init; c=0; end;
shocks; var e; stderr p_shk; var u=1; end;
stoch_simul(order=1, irf=0);
