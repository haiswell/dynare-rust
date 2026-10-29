// Maximum-likelihood test model for dynare-rust (estimated_params without priors),
// stationary (lik_init = 1) with a measurement error on x. Data: make_data.py.
var x z;
varexo ex ez;
parameters a b c;
a = 0.8; b = 0.3; c = 0.5;

model(linear);
x = a*x(-1) + b*z(-1) + ex;
z = c*z(-1) + ez;
end;

shocks;
var ex; stderr 1;
var ez; stderr 0.5;
end;

steady;

varobs x z;

estimated_params;
a, 0.7, -0.99, 0.99;
b, 0.2, -2, 2;
c, 0.4, -0.99, 0.99;
stderr ex, 0.8, 0.01, 10;
stderr ez, 0.6, 0.01, 10;
stderr x, 0.2, 0.001, 5;
corr ex, ez, 0.1, -0.9, 0.9;
end;
estimation(datafile='model_ml_obs.csv', mode_compute=4, nograph, smoother, forecast=4, filtered_vars) x z;
