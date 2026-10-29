// trend component auxiliary model, growth neutrality, share of optimizing agents (rule of thumb agents)
var x1 x2 x1bar x2bar z y w;
varexo ex1 ex2 ex1bar ex2bar ez ey ew;
parameters a_x1_0 a_x1_1 a_x1_2 a_x1_x2_1 a_x1_x2_2 a_x2_0 a_x2_1 a_x2_2 a_x2_x1_1 a_x2_x1_2
 beta e_c_m c_z_1 c_z_2 gg lambda;
a_x1_0=-.2; a_x1_1=.3; a_x1_2=.1; a_x1_x2_1=.05; a_x1_x2_2=.02;
a_x2_0=-.15; a_x2_1=.04; a_x2_2=.03; a_x2_x1_1=.2; a_x2_x1_2=.1;
beta=.9; e_c_m=.1; c_z_1=.7; c_z_2=-.3; gg=0.01; lambda=.6;
trend_component_model(model_name=toto, eqtags=['eq:x1', 'eq:x2', 'eq:x1bar', 'eq:x2bar'], targets=['eq:x1bar', 'eq:x2bar']);
pac_model(auxiliary_model_name=toto, discount=beta, model_name=pacman, growth=gg);
model;
[name='eq:x1']
diff(x1) = a_x1_0*(x1(-1)-x1bar(-1)) + a_x1_1*diff(x1(-1)) + a_x1_2*diff(x1(-2)) + a_x1_x2_1*diff(x2(-1)) + a_x1_x2_2*diff(x2(-2)) + ex1;
[name='eq:x2']
diff(x2) = a_x2_0*(x2(-1)-x2bar(-1)) + a_x2_1*diff(x1(-1)) + a_x2_2*diff(x1(-2)) + a_x2_x1_1*diff(x2(-1)) + a_x2_x1_2*diff(x2(-2)) + ex2;
[name='eq:x1bar']
x1bar = x1bar(-1) + ex1bar;
[name='eq:x2bar']
x2bar = x2bar(-1) + ex2bar;
[name='zpac']
diff(z) = lambda*(e_c_m*(x1(-1)-z(-1)) + c_z_1*diff(z(-1))  + c_z_2*diff(z(-2)) + pac_expectation(pacman)) + (1-lambda)*w + ez;
w = .8*w(-1) + ew;
y = .5*y(-1) + z + ey;
end;
shocks; var ex1=.01; var ex2=.01; var ex1bar=.01; var ex2bar=.01; var ez=.02; var ey=.01; var ew=.01; end;
pac.initialize('pacman');
pac.update.expectation('pacman');
