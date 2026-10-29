// Dynare 7.2 tests/optimal_policy/nk_ramsey_det.mod (comments, write_latex_static_model and rplot dropped)
var pai, c, n, r, a;
varexo u;
parameters beta, rho, epsilon, omega, phi, gamma;
beta=0.99;
gamma=3;       //Frish elasticity
omega=17;        //price stickyness
epsilon=8;      //elasticity for each variety of consumption
phi=1;           //coefficient associated to labor effort disutility
rho=0.95;  //coefficient associated to productivity shock
model;
a=rho*(a(-1))+u;
1/c=beta*(1/(c(+1)))*(r/(pai(+1)));               //euler
omega*pai*(pai-1)=beta*omega*(c/(c(+1)))*(pai(+1))*(pai(+1)-1)+epsilon*exp(a)*n*(c/exp(a)*phi*n^gamma-(epsilon-1)/epsilon);  //NK pc
(exp(a))*n=c+(omega/2)*((pai-1)^2);
end;
initval;
pai=1;
r=1/beta;
c=0.9671684882;
n=0.9671684882;
a=0;
end;
shocks;
var u;
periods 1;
values 0.008;
end;
planner_objective(ln(c)-phi*((n^(1+gamma))/(1+gamma)));
ramsey_model(planner_discount=0.99);
steady;
perfect_foresight_setup(periods=200);
perfect_foresight_solver(maxit=20);
