// neo_growth (Dynare 7.2 tests/optimal_policy) at order 5: decision rule and simulations
@#include "neo_growth_common.inc"

shocks;
var e;
stderr 1;
end;

steady;

stoch_simul(order=5, irf=0, periods=40, drop=0, nomoments, noprint);
