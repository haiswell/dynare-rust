function [fval, info, exit_flag, dlik, hess] = ratfit_loop(x, with_dlik)
% Least-squares fit of y = (x1 + x2 t)/(1 + x3 t + x4^2 t^2) to 12 points, as a sum of
% squared residuals (nonzero at the optimum, so newrat's outer-product Hessian stays
% regular). Only +, -, *, / and a plain loop sum, so that the Rust objective in
% tests/estimation_newrat_iterates.rs is bit for bit the same. with_dlik = 0 returns no
% contributions (newrat's BFGS branch).
y = [1.2 1.9 2.3 2.4 2.35 2.2 2.0 1.9 1.75 1.6 1.5 1.45];
r = zeros(12,1);
for k=1:12
    t = 0.5*k;
    m = (x(1) + x(2)*t)/(1 + x(3)*t + x(4)*x(4)*t*t);
    e = y(k) - m;
    r(k) = e*e;
end
fval = 0;
for k=1:12
    fval = fval + r(k);
end
if with_dlik
    dlik = r;
else
    dlik = [];
end
info = zeros(4,1);
exit_flag = 1;
hess = [];
