// endogenous (and exogenous) variables in the expressions of shocks blocks: oo_.steady_state(i)
// / oo_.exo_steady_state(k) when the block stands in the driver
var y c;
varexo e u;
parameters a;
a = 0.5;
model;
c = 0.4*c(-1) + 1 + u;
y = a*y(-1) + c + e;
end;
initval;
y = 1;
c = 2;
e = 0;
u = 0.5;
end;
// before steady: oo_.steady_state = initval
shocks;
var e; stderr y+1;
var u; stderr c/10;
corr e, u = y/10;
end;
stoch_simul(order=1, irf=0, nomoments, noprint, nograph);
steady;
// after steady: the computed steady state
shocks;
var e; stderr y/10;
var u = c/100+u;
var e; periods 1:2; values (y/10);
var u; periods 3; values (c+u);
corr e, u = c/10;
end;
stoch_simul(order=1, irf=0, noprint, nograph);
perfect_foresight_setup(periods=8);
perfect_foresight_solver;
