// tests/lmmcp/SMOPEC.mod of Dynare 7.2 (a model by Johannes Pfeifer, GPL-3+), without its comments
var  c d y;

parameters  r_bar ${\bar r}$
            beta;

r_bar    = 0.04; %world interest rate
beta=0.9;
d_bar=2;

model;
    [name='Evolution of debt']
    y + d= c + (1+r_bar)*d(-1);
    [name='Endowment']
    y = 1;
    [name='Euler equation']
    beta*(1+r_bar)*1/c(+1)-1/c=0 ⟂ d<2;
end;

initval;
d=d_bar;
y=1;
c=y-r_bar*d;
end;
resid;
steady(solve_algo=10);
resid;
