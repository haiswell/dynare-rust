var y; varexo e; parameters a b;
a=0.5; b=1;
model;
y = a*y(-1) + e;
end;
shocks; var e; stderr 1; end;
simul(periods=10);
stoch_simul(order=1 periods=3);
