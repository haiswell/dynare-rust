# Writes ls2003_recompute_m.mod: ls2003_recompute.mod with MATLAB lines that store what
# plot_shock_decomposition plots (get_decomp_only), the decompositions and the bookkeeping
# (plot_shock_decomposition_info.i_var / shock_decomposition_info.i_var) in ref.json.
# Run: python3 dump.py; then in MATLAB with Dynare 7.2 on the path and getiv.m (below) next to it:
#   dynare ls2003_recompute_m nograph nointeractive
# getiv.m:
#   function v = getiv(o, f)
#   v = -1;
#   if isfield(o,f) && isfield(o.(f),'i_var'), v = o.(f).i_var(:)'; end
#   end
src = open('ls2003_recompute.mod').read().split('\n')
out = []
book = ("bk{end+1} = struct('line', %d, 'piv', getiv(oo_,'plot_shock_decomposition_info'), "
        "'siv', getiv(oo_,'shock_decomposition_info'), 'nsd', size(oo_.shock_decomposition,1));")
for i, l in enumerate(src, 1):
    if l.startswith('squeeze_shock_decomposition') and i > 90:
        # before the last squeeze: the fast_realtime results (y = 1, ya = 12)
        out.append("RES_.fast_pool = oo_.realtime_shock_decomposition.pool([1 12],:,:);")
        out.append("RES_.fast_fields = fieldnames(oo_.realtime_forecast_shock_decomposition);")
        out.append("RES_.fast_f29 = oo_.realtime_forecast_shock_decomposition.time_29([1 12],:,:);")
    if l.startswith('shock_decomposition('):
        out.append('zz={}; bk={};')
    out.append(l)
    if l.startswith('plot_shock_decomposition'):
        out.append("zz{end+1} = plot_shock_decomposition(M_, oo_, options_, var_list_, true);")
    if l.startswith('initial_condition_decomposition'):
        out.append("RES_.iv_size = size(oo_.initval_decomposition); RES_.iv_R_obs = oo_.initval_decomposition(11,:,:);")
    if l.split('(')[0].split(' ')[0] in ('shock_decomposition', 'plot_shock_decomposition',
                                          'initial_condition_decomposition', 'realtime_shock_decomposition',
                                          'squeeze_shock_decomposition;'):
        out.append(book % i)
out.append("RES_.zz = zz; RES_.bk = bk;")
out.append("fid = fopen('ref.json', 'w'); fprintf(fid, '%s', jsonencode(RES_)); fclose(fid);")
open('ls2003_recompute_m.mod', 'w').write('\n'.join(out) + '\n')
