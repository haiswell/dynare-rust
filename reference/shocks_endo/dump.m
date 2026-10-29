function dump(fname)
global M_ oo_
s.sigma_e = M_.Sigma_e;
s.corr = M_.Correlation_matrix;
if isfield(oo_,'var'), s.var = oo_.var; end
if isfield(oo_,'endo_simul'), s.endo_simul = oo_.endo_simul; end
if isfield(oo_,'exo_simul'), s.exo_simul = oo_.exo_simul; end
s.steady_state = oo_.steady_state;
fid = fopen([fname '.json'],'w'); fprintf(fid,'%s',jsonencode(s,'PrettyPrint',true)); fclose(fid);
end
