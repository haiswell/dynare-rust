addpath('/Applications/Dynare/7.2-arm64/matlab');
% Writes ref_ng5.json (orders 4+: decision rule, simulations with and without pruning) for
% tests/korder_general.rs; run from reference/korder (ref_ng5.json is then trimmed to the
% fields the test reads). Dynare 7.2.
dynare ng5 nograph nointeractive
r = struct();
r.order_var = oo_.dr.order_var';
r.ys = oo_.dr.ys';
for i = 0:5
    r.(sprintf('g_%d', i)) = oo_.dr.(sprintf('g_%d', i));
end
r.exo_simul = oo_.exo_simul;
r.endo_simul = oo_.endo_simul;
% the same shocks with pruning (k_order_simul's pruned recursion)
options_.pruning = true;
try
    [dr2, info] = k_order_pert(oo_.dr, M_, options_);
    r.pruning_fields = fieldnames(dr2.pruning)';
    for i = 1:5
        r.(sprintf('pruning_g_%d', i)) = dr2.pruning.(sprintf('g_%d', i));
    end
    y0 = oo_.dr.ys;
    yp = simult_(M_, options_, y0, dr2, oo_.exo_simul, 5);
    r.endo_simul_pruned = yp;
    r.pruning_error = '';
catch err
    r.pruning_error = err.message;
end
% a start away from the steady state
y0 = oo_.dr.ys; y0(2) = y0(2) * 1.05; y0(3) = 0.03;
options_.pruning = false;
r.y0 = y0';
r.endo_simul_y0 = simult_(M_, options_, y0, oo_.dr, oo_.exo_simul, 5);
fid = fopen('ref_ng5.json', 'w'); fprintf(fid, '%s', jsonencode(r)); fclose(fid);
