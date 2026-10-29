/*
 * Copyright © 2021 Dynare Team
 *
 * This file is part of Dynare.
 *
 * Dynare is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Dynare is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Dynare.  If not, see <https://www.gnu.org/licenses/>.
 */
/*
 * This file computes a kth-order approximation of the neo-classical growth model.
 * It assesses the conditional welfare and the derivatives of the felicity and welfare functions computed by the k_order_welfare function
 * and compares them to a by-hand assessment stemming from the results the model neo_growth_k_order.mod incur.
 *
 * dynare-rs: Dynare 7.2's tests/optimal_policy/neo_growth_ramsey_k_order.mod up to
 * evaluate_planner_objective (the MATLAB checks that follow it are left out).
 */

@#include "neo_growth_ramsey_common.inc"

shocks;
var e;
stderr 1;
end;

stoch_simul(order=6, irf=0);

evaluate_planner_objective(periods=10000,drop=1000);
