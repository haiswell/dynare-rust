function sweep_run_one(modfile, varargin)
% Run `dynare <modfile> nograph nointeractive [extra args]` (graphs on when the .mod runs pdflatex) in the current directory and
% dump oo_ / key M_ fields as JSON to the last argument (outjson), plus <outjson>.status.json.
global M_ oo_ options_
outjson = varargin{end};
extra = varargin(1:end-1);
st = struct('ok', false, 'error', '', 'elapsed', 0);
t0 = tic;
try
    % tests that compile their LaTeX output (pdflatex on the TeX binder) need the graphs
    if contains(fileread(modfile), 'pdflatex')
        dynare(modfile, 'nointeractive', extra{:});
    else
        dynare(modfile, 'nograph', 'nointeractive', extra{:});
    end
    st.ok = true;
catch ME
    st.error = [ME.identifier ': ' ME.message];
    if ~isempty(ME.stack), st.error = [st.error ' @' ME.stack(1).name ':' num2str(ME.stack(1).line)]; end
end
st.elapsed = toc(t0);
try
    out = struct();
    if exist('oo_', 'var'), out.oo_ = sweep_clean(oo_, 0); end
    if exist('M_', 'var')
        keep = {'fname','endo_names','exo_names','param_names','params','endo_nbr','exo_nbr','param_nbr','orig_endo_nbr','nstatic','nfwrd','npred','nboth','nspred','Sigma_e','H','maximum_lag','maximum_lead'};
        mm = struct();
        for i = 1:numel(keep)
            if isfield(M_, keep{i}), mm.(keep{i}) = sweep_clean(M_.(keep{i}), 0); end
        end
        out.M_ = mm;
    end
    fid = fopen(outjson, 'w'); fwrite(fid, jsonencode(out, 'ConvertInfAndNaN', false)); fclose(fid);
catch ME2
    st.dump_error = ME2.message;
end
fid = fopen([outjson '.status.json'], 'w'); fwrite(fid, jsonencode(st)); fclose(fid);
end

function y = sweep_clean(x, depth)
% generic JSON-friendly copy: numerics as {size,data(col-major)}, skip huge/non-numeric
y = [];
if depth > 7, return; end
if isnumeric(x) || islogical(x)
    if numel(x) > 1e6 || ~isreal(x), y = struct('skipped', numel(x)); return; end
    y = struct('size', size(x), 'data', full(double(x(:)))');  % jsonencode rejects sparse matrices
elseif ischar(x)
    y = x(1:min(end, 2000)); if size(x,1) > 1, y = cellstr(x); end
elseif isstring(x)
    y = cellstr(x);
elseif iscell(x)
    if numel(x) > 20000, y = []; return; end
    y = cell(1, numel(x));
    for i = 1:numel(x), y{i} = sweep_clean(x{i}, depth+1); end
elseif isstruct(x)
    if numel(x) ~= 1
        if numel(x) > 5000, y = []; return; end
        y = cell(1, numel(x));
        for i = 1:numel(x), y{i} = sweep_clean(x(i), depth+1); end
        return
    end
    y = struct();
    f = fieldnames(x);
    for i = 1:numel(f)
        y.(f{i}) = sweep_clean(x.(f{i}), depth+1);
    end
end
end
