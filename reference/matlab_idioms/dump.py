import sys
src, dst = sys.argv[1], sys.argv[2]
out = ['zz = {};']
for l in open(src).read().split('\n'):
    out.append(l)
    if l.startswith('plot_shock_decomposition'):
        out.append("zz{end+1} = plot_shock_decomposition(M_, oo_, options_, var_list_, true);")
out.append("RES_.zz = zz;")
out.append("fid = fopen('ref.json', 'w'); fprintf(fid, '%s', jsonencode(RES_)); fclose(fid);")
open(dst, 'w').write('\n'.join(out) + '\n')
