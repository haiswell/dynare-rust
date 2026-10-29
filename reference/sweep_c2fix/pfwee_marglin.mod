/* Test that perfect foresight simulation with expectation errors gives correct
   results with marginal linearization. Homotopy is not really triggered, but
   the use of homotopy_max_completion_share is enough to test the computations.

   This test is similar to pfwee_homotopy_marginal_linearization.mod, except that
   the numerical values are different (the shock is smaller to facilitate computations).
   */

var Consumption, Capital, LoggedProductivity;

varexo LoggedProductivityInnovation;

parameters beta, alpha, delta, rho;

beta = .985;
alpha = 1/3;
delta = alpha/10;
rho = .4;

model;
  1/Consumption = beta/Consumption(1)*(alpha*exp(LoggedProductivity(1))*Capital^(alpha-1)+1-delta);
  Capital = exp(LoggedProductivity)*Capital(-1)^alpha+(1-delta)*Capital(-1)-Consumption;
  LoggedProductivity = rho*LoggedProductivity(-1)+LoggedProductivityInnovation;
end;

initval;
  LoggedProductivityInnovation = 0.1;
end;

steady;

endval;
  LoggedProductivityInnovation = 0.3;
end;

shocks(learnt_in = 5);
  var LoggedProductivityInnovation;
  periods 7;
  values -0.5;
end;

endval(learnt_in = 10);
  LoggedProductivityInnovation = 0.6;
end;

// Save initial steady state (it will be modified by pfwee)
orig_steady_state = oo_.steady_state;
orig_exo_steady_state = oo_.exo_steady_state;

perfect_foresight_with_expectation_errors_setup(periods = 50);
perfect_foresight_with_expectation_errors_solver(homotopy_max_completion_share = 0.8, homotopy_marginal_linearization_fallback, steady_solve_algo = 13);
