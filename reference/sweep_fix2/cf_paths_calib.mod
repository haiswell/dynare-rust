// conditional_forecast_paths with endogenous variables in the values: they stand for
// oo_.steady_state, which stoch_simul does not fill (zeros), so the paths are 0.01, -0.02, 0...
@#include "fs2000.inc"

varobs gp_obs gy_obs;

stoch_simul(order=1,nomoments,irf=0);

conditional_forecast_paths;
var gy_obs;
periods  1  2  3:5;
values   (gy_obs+0.01) (gy_obs-0.02) (gy_obs);
var gp_obs;
periods 1:5;
values  (gp_obs+0.05+k/100);
end;

conditional_forecast(periods=10, parameter_set=calibration, controlled_varexo=(e_a,e_m));
