% Reference values of Dynare 7.2's tests/heterogeneity files whose parameters come from
% `load 'X.mat'` + `for ... set_param_value(param, steady_state.params.(param))`.
addpath('/Applications/Dynare/7.2-arm64/matlab');
here = fileparts(mfilename('fullpath')); cd(here);
mdl = getenv('MDL');
  try

    eval(sprintf('dynare %s nointeractive nograph', mdl));
    global M_ options_ oo_
    R = struct();
    mdl = getenv('MDL');
    R.params = M_.params; R.param_names = M_.param_names;
    h = oo_.heterogeneity;
    ss = h.steady_state;
    R.het.agg = ss.agg;
    fn = fieldnames(ss.pol.values);
    for i=1:numel(fn), v = ss.pol.values.(fn{i}); R.het.pol.(fn{i}) = struct('size',size(v),'sum',sum(v(:)),'first',v(1:min(10,end))); end
    if isfield(ss,'d'), d = ss.d.hist; R.het.d = struct('size',size(d),'sum',sum(d(:)), 'first', d(1:min(10,end))); end
    if isfield(h,'dr') && isfield(h.dr,'G')
      R.het.T = options_.heterogeneity.solve.truncation_horizon;
      G = h.dr.G; fn = fieldnames(G);
      for i=1:numel(fn), fs = fieldnames(G.(fn{i})); for j=1:numel(fs), g = full(G.(fn{i}).(fs{j})); R.het.G.(fn{i}).(fs{j}) = g(1:min(5,end),1)'; end, end
    end
    fid = fopen(sprintf('ref_%s.json', mdl), 'w');
    fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
    fprintf('DUMPED %s\n', mdl);
  catch e
    fprintf('FAILED %s: %s\n', mdl, e.message);
  end
