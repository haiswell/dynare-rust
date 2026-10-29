var y c;
varexo e u;
parameters a b;
a=0.9; b=0.5;
model(bytecode);
y = a*y(-1) + e;
[mcp = 'c > 0']
c = b*c(-1) + u
  + 0;
end;
shocks; var e=1; var u=1; end;
simul(periods=10);
