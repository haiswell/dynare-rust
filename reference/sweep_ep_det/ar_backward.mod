var Efficiency, efficiency;

varexo EfficiencyInnovation;

parameters rho, effstar, sigma;

/*
** Calibration
*/


rho     =  0.950;
effstar =  1.000;
sigma   =  0.1;

model;

  // Eq. n°1:
  efficiency = rho*efficiency(-1) + sigma*EfficiencyInnovation;

  // Eq. n°2:
  Efficiency = effstar*exp(efficiency);

end;

shocks;
var EfficiencyInnovation = 1;
end;

steady_state_model;
efficiency=0;
Efficiency=effstar;
end;

steady(nocheck);

extended_path(order=0,periods=100);
