#!/usr/bin/env python3
"""Compare dynare-rs outputs with MATLAB Dynare's oo_ dumps for each swept entry.

usage: python3 -I compare.py [RESULTS_DIR]     (default: $SWEEP_OUT, see sweep.py)
Writes RESULTS_DIR/compare.json: {"<id>.<k>": {"fields": [...], "unmapped": [...], "stochastic_test": bool,
"tol": float}}.  A field is {name, rel, status, stoch} with status in ok|diff|shape|nan.
"""
import csv, glob, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa: E402  (only for DEF_OUT / DEF_TESTS)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit('numpy required')

DR_KEYS = {'ghx', 'ghu', 'ghxx', 'ghuu', 'ghxu', 'ghs2', 'ghxxx', 'ghuuu', 'ghxxu', 'ghxuu', 'ghxss', 'ghuss', 'ghxu',
           'ghxss', 'order_var', 'state_var', 'ys', 'g_0', 'g_1', 'g_2', 'g_3', 'ghuuu'}
OO_TOP = {'mean', 'var', 'skewness', 'kurtosis', 'steady_state', 'endo_simul', 'exo_simul', 'autocorr',
          'variance_decomposition', 'variance_decomposition_ME', 'gamma_y', 'irfs'}
TOL_DET = 1e-6
TOL_EST = 1e-3


def arr(x):
    """dynare-rs nested lists -> ndarray; None -> nan."""
    try:
        return np.array(x, dtype=float)
    except Exception:
        return None


def flatten_matlab(o, p=''):
    """MATLAB dump -> {path: ndarray}; {size,data} leaves, cells as path[i]."""
    out = {}
    if isinstance(o, dict):
        if set(o) == {'size', 'data'}:
            d = o['data'] if isinstance(o['data'], list) else [o['data']]
            sz = o['size'] if isinstance(o['size'], list) else [o['size']]
            try:
                out[p] = np.array(d, dtype=float).reshape([int(s) for s in sz], order='F')
            except Exception:
                pass
        else:
            for k, v in o.items():
                out.update(flatten_matlab(v, p + '.' + k if p else k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(flatten_matlab(v, '%s{%d}' % (p, i + 1)))
    return out


def flatten_rs_json(o, p):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flatten_rs_json(v, p + '.' + k))
    elif isinstance(o, (list, int, float)) and not isinstance(o, bool):
        a = arr(o)
        if a is not None:
            out[p] = a
    return out


def read_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def num(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def rs_fields(odir, M):
    """Collect dynare-rs outputs mapped onto oo_ paths. Returns (fields, unmapped_files)."""
    F, unm = {}, []
    endo = [s.strip() for s in M.get('endo_names', [])] if isinstance(M.get('endo_names'), list) else []
    for f in sorted(glob.glob(os.path.join(odir, '*'))):
        b = os.path.basename(f)
        try:
            if b.endswith('_progress.json') or b.endswith('.mat'):
                continue
            if b.endswith('_stoch_simul.json'):
                d = json.load(open(f))
                for k, v in d.items():
                    if k == 'autocorr' and isinstance(v, list):
                        for i, m in enumerate(v):
                            F.update(flatten_rs_json(m, 'oo_.autocorr{%d}' % (i + 1)))
                    else:
                        F.update(flatten_rs_json(v, 'oo_.' + k))
            elif b.endswith('_ramsey.json'):
                d = json.load(open(f))
                for k, v in d.items():
                    if k in DR_KEYS:
                        F.update(flatten_rs_json(v, 'oo_.dr.' + k))
                    elif k == 'steady_state':
                        # MATLAB's stoch_simul never writes oo_.steady_state (it keeps the initval values
                        # when the .mod has no steady command); the Ramsey steady state is oo_.dr.ys
                        F.update(flatten_rs_json(v, 'oo_.dr.ys'))
                    elif k in OO_TOP:
                        F.update(flatten_rs_json(v, 'oo_.' + k))
            elif b.endswith('_results.json') and not b.startswith('sbvar'):
                d = json.load(open(f))
                names = d.get('parameter_names') or []
                for i, n in enumerate(names):
                    m = re.match(r'(?:stderr|corr)\s+(.*)', n)
                    if d.get('mode') and i < len(d['mode']):
                        if m and n.startswith('stderr'):
                            key = 'oo_.posterior_mode.shocks_std.' + m.group(1)
                        elif m:
                            key = 'oo_.posterior_mode.shocks_corr.' + m.group(1).replace(', ', '_').replace(',', '_')
                        else:
                            key = 'oo_.posterior_mode.parameters.' + n
                        F[key] = np.array(d['mode'][i], dtype=float)
                    if d.get('mode_sd') and i < len(d['mode_sd']):
                        key = ('oo_.posterior_std_at_mode.shocks_std.' + m.group(1)) if (m and n.startswith('stderr')) \
                            else 'oo_.posterior_std_at_mode.parameters.' + n
                        F[key] = np.array(d['mode_sd'][i], dtype=float)
                if d.get('minus_log_posterior_at_mode') is not None:
                    F['oo_.posterior.optimization.log_density'] = np.array(-d['minus_log_posterior_at_mode'])
                if d.get('log_data_density_laplace') is not None:
                    F['oo_.MarginalDensity.LaplaceApproximation'] = np.array(d['log_data_density_laplace'])
                mc = d.get('mcmc') or {}
                if mc.get('log_data_density_mhm') is not None:
                    F['oo_.MarginalDensity.ModifiedHarmonicMean'] = np.array(mc['log_data_density_mhm'])
                    for i, n in enumerate(names):
                        if i < len(mc.get('posterior_mean', [])) and not n.startswith(('stderr', 'corr')):
                            F['oo_.posterior_mean.parameters.' + n] = np.array(mc['posterior_mean'][i], dtype=float)
            elif b.endswith('_endo_simul.csv'):
                h, r = read_csv(f)
                A = np.array([[num(x) for x in row[1:]] for row in r])
                for j, n in enumerate(h[1:]):
                    pass
                F['oo_.endo_simul'] = ('rows', h[1:], A.T)
            elif re.search(r'_(calib_)?(smoothed|filtered|updated)_(variables|shocks)\.csv$', b):
                m = re.search(r'_(calib_)?(smoothed|filtered|updated)_(variables|shocks)\.csv$', b)
                kind = {'smoothed': 'Smoothed', 'filtered': 'Filtered', 'updated': 'Updated'}[m.group(2)] + \
                       ('Variables' if m.group(3) == 'variables' else 'Shocks')
                h, r = read_csv(f)
                A = np.array([[num(x) for x in row] for row in r])
                for j, n in enumerate(h[1:]):
                    F['oo_.%s.%s' % (kind, n)] = A[:, j + 1]
            elif re.search(r'_IRF_(.+)\.csv$', b) and '_Bayesian_' not in b:
                sh = re.search(r'_IRF_(.+)\.csv$', b).group(1)
                h, r = read_csv(f)
                A = np.array([[num(x) for x in row] for row in r])
                for j, n in enumerate(h[1:]):
                    F['oo_.irfs.%s_%s' % (n, sh)] = A[:, j + 1]
            elif b.endswith('_forecast.csv') and not b.startswith('conditional'):
                h, r = read_csv(f)
                if h[:3] == ['date', 'horizon', 'variable']:
                    byvar = {}
                    for row in r:
                        byvar.setdefault(row[2], []).append(row)
                    for v, rows in byvar.items():
                        F['oo_.forecast.Mean.' + v] = np.array([num(x[3]) for x in rows])
                        F['oo_.forecast.HPDinf.' + v] = np.array([num(x[4]) for x in rows])
                        F['oo_.forecast.HPDsup.' + v] = np.array([num(x[5]) for x in rows])
                else:
                    unm.append(b)
            else:
                unm.append(b)
        except Exception as ex:
            unm.append(b + ' (read error %s)' % ex)
    return F, unm


STOCH_PAT = re.compile(r'mh_replic|particle|sequential_importance|smc|dsmh|hssmc|bayesian_irf|nonlinear_filter|'
                       r'filter_algorithm|dime|mcmc|simul_replic|posterior_sampler|ms_estimation|sbvar|'
                       r'online|gsa|rand|shock_decomposition', re.I)


SIMUL_PAT = r'\b(stoch_simul|discretionary_policy|ramsey_policy)\s*\([^;]*periods\s*=\s*[1-9]'


def uncommented(text):
    return '\n'.join(l.split('%')[0].split('//')[0] for l in text.split('\n'))


def particle_estimation(text):
    """An estimation command at order >= 2 whose filter draws random numbers (all but the nonlinear Kalman
    filter with a cubature / unscented proposal)."""
    for m in re.finditer(r'\bestimation\s*\(([^;]*)\)\s*;', uncommented(text)):
        o = m.group(1)
        if not re.search(r'\border\s*=\s*[23]', o):
            continue
        if re.search(r'filter_algorithm\s*=\s*nlkf', o) and not re.search(r'proposal_approximation\s*=\s*montecarlo', o):
            continue
        return True
    return False


def simulated_datafiles(text):
    """Data files a test writes with datatomfile after simulating with random shocks."""
    if not re.search(SIMUL_PAT, text):
        return set()
    return set(re.findall(r"datatomfile\s*\(\s*'(\w+)'", uncommented(text)))


def reads_datafile(text, names):
    return any(re.search(r"datafile\s*=\s*'?(?:[\w.]*/)*%s\b" % re.escape(n), text) for n in names)


def classify_field(name, text, ML=None):
    """True if the field is RNG driven in this test."""
    # stoch_simul IRFs at order >= 2 are simulated (irf.m draws replic shock paths): oo_.dr.ghxx exists then
    if ML is not None and name.startswith('oo_.irfs.') and 'oo_.dr.ghxx' in ML:
        return True
    # forecast with varexo_det at order 2: simult_varexo_det.m's bands are quantiles of randn simulations
    if ML is not None and re.match(r'oo_\.forecast\.HPD', name) and 'oo_.dr.ghxx' in ML \
            and any(k.startswith('oo_.dr.ghud') for k in ML):
        return True
    if re.match(r'oo_\.(endo_simul|exo_simul)', name) and re.search(SIMUL_PAT, text):
        return True
    # moments of a simulated series (disp_moments.m; discretionary_policy / ramsey_policy call stoch_simul.m)
    if re.search(SIMUL_PAT, text) and re.match(
            r'oo_\.(mean|var|skewness|kurtosis|autocorr|gamma_y)', name):
        return True
    # estimation at order >= 2 with a particle filter: the likelihood is a randn/resampling estimate
    if re.match(r'oo_\.(posterior_mode|posterior_std_at_mode|posterior\.optimization|MarginalDensity|Smoothed|'
                r'Updated|Filtered)', name) and particle_estimation(text):
        return True
    # slice sampler: marginal_density.m stores the MCMC's best draw as oo_.posterior_mode
    if re.match(r'oo_\.posterior_mode\.', name) and re.search(r"posterior_sampling_method\s*=\s*'slice'", text):
        return True
    # after an MCMC the classical smoother runs at the posterior mean (dynare_estimation_1.m)
    if re.match(r'oo_\.(Smoothed|Updated|Filtered)', name) and re.search(r'mh_replic\s*=\s*[1-9]', text):
        return True
    if re.match(r'oo_\.(posterior_mean|posterior_median|posterior_hpdinf|posterior_hpdsup|MarginalDensity\.ModifiedHarmonicMean)', name):
        return True
    return False


def cmp_arrays(a, b, scale_floor=0.0):
    """rel diff (normwise max) between rs array a and matlab array b."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size != b.size:
        # dr matrices may differ by trailing singleton; otherwise shape mismatch
        return None, 'shape', '%s vs %s' % (a.shape, b.shape)
    a, b = a.ravel(), b.ravel()
    if a.shape != b.shape:
        return None, 'shape', ''
    # dynare-rs JSON writes every non-finite value as null (-> nan) while MATLAB keeps +-inf (e.g. autocorr of a
    # zero-variance variable): compare the non-finite masks, not the nan masks
    bad = np.isfinite(a) != np.isfinite(b)
    if bad.any():
        return float('inf'), 'nan', ''
    m = np.isfinite(b)
    if not m.any():
        return 0.0, 'ok', ''
    a, b = a[m], b[m]
    scale = max(np.max(np.abs(b)), scale_floor, 1e-12)
    rel = float(np.max(np.abs(a - b)) / scale)
    return rel, 'ok', ''


def sign_insensitive(name):
    return False


ENDPOINT_FIELDS = re.compile(r'oo_\.(posterior_mode|posterior_std_at_mode|posterior_variance_at_mode|MarginalDensity\.Laplace|'
                             r'posterior\.optimization\.(mode|Variance)|(Smoothed|Updated|Filtered)\w*|FilteredVariablesKStepAhead\w*|'
                             r'forecast|irfs|dr\.|mean|var|autocorr|gamma_y|variance_decomposition|steady_state)')


def mark_optimizer_endpoint(r):
    """Estimations whose log posterior at the mode agrees with MATLAB's (1e-7 relative, about 1e-4 on fs2000; a 3e-4 gap from a wrong solve_tolf was 2.5e-7)
    while the mode differs: the optimizers stop at different points of a flat ridge, and everything
    evaluated at the mode moves with it. Such fields get note 'endpoint' (triage category p)."""
    ld = [f for f in r.get('fields', []) if f['name'] == 'oo_.posterior.optimization.log_density']
    if not ld or ld[0]['rel'] is None or ld[0]["rel"] > 1e-7:
        return
    for f in r['fields']:
        if not f['stoch'] and f['status'] == 'ok' and ENDPOINT_FIELDS.match(f['name']):
            f['endpoint'] = True



def compare_run(out, eid, k, mod, text, simdata=False):
    """`simdata`: the test estimates on data an earlier test of the entry simulated with random shocks."""
    mp = os.path.join(out, 'matlab', str(eid), str(k), 'oo.json')
    od = os.path.join(out, 'rs', str(eid), str(k), 'Output')
    if not (os.path.exists(mp) and os.path.isdir(od)):
        return None
    D = json.load(open(mp))
    ML = flatten_matlab(D)
    M = D.get('M_', {})
    F, unm = rs_fields(od, M)
    endo = M.get('endo_names') or []
    exo = M.get('exo_names') or []
    # largest |value| of the MATLAB IRFs / forecasts (scale floors below)
    MX = {kk: float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 0.0 for kk, v in ML.items()
          if v.size and re.match(r'oo_\.(irfs|forecast|MeanForecast|PointForecast)\.', kk)}
    exo = [exo] if isinstance(exo, str) else [x.strip() for x in exo]
    est = bool(re.search(r'\bestimation\s*\(|\bmode_compute|\bosr\b|\bmethod_of_moments\b', text))
    tol = TOL_EST if est else TOL_DET
    res = []
    for name, a in F.items():
        if isinstance(a, tuple):  # endo_simul rows by name
            _, names, A = a
            mname = 'oo_.endo_simul'
            if mname not in ML:
                continue
            B = ML[mname]
            idx = {n.strip(): i for i, n in enumerate(endo)}
            try:
                sel = [idx[n] for n in names]
            except KeyError:
                res.append(dict(name=mname, rel=None, status='shape', note='names', stoch=False))
                continue
            rel, st, note = cmp_arrays(A, B[sel, :])
            res.append(dict(name=mname, rel=rel, status=st, note=note, stoch=classify_field(mname, text, ML)))
            continue
        cands = [name]
        if name.startswith('oo_.dr.ys'):
            cands.append('oo_.steady_state')
        if name == 'oo_.dr.ys' or name == 'oo_.steady_state':
            pass
        if '.shocks_std.' in name:
            cands.append(name.replace('.shocks_std.', '.measurement_errors_std.'))
        b = None
        for c in cands:
            if c in ML:
                b = ML[c]
                break
        if b is None:
            # name might be nested differently (variance_decomposition etc); try suffix match
            unm.append(name)
            continue
        if name == 'oo_.dr.ys' and 'oo_.dr.ys' not in ML and b is not None:
            pass
        if b.size == 0:
            # MATLAB left the field empty (e.g. oo_.posterior.optimization.log_density = [] with mode_compute=0)
            unm.append(name + ' (empty in MATLAB)')
            continue
        floor = 0.0
        if name.startswith('oo_.irfs.'):
            # an IRF is judged against the largest response to the same shock (an all-but-zero response, e.g.
            # 1e-8 under an optimal simple rule, otherwise turns optimizer noise into huge relative errors)
            sh = [x for x in exo if name.endswith('_' + x)]
            if sh:
                sh = max(sh, key=len)
                floor = max([v for kk, v in MX.items() if kk.startswith('oo_.irfs.') and kk.endswith('_' + sh)]
                            or [0.0])
            # a shock the model all but neutralizes (1e-8 responses to a shock that optimal policy offsets,
            # next to O(1) responses to the others) is judged against the largest response of all
            gmax = max([v for kk, v in MX.items() if kk.startswith('oo_.irfs.')] or [0.0])
            if floor < 1e-6 * gmax:
                floor = gmax
        m = re.match(r'(oo_\.(?:forecast|MeanForecast|PointForecast)\.[A-Za-z_]+)\.', name)
        if m:
            # likewise a forecast (Mean, HPDinf, ...) against the largest of all variables' (a variable at its
            # steady state, 1e-16, is rounding noise)
            pre = m.group(1) + '.'
            floor = max([v for kk, v in MX.items() if kk.startswith(pre)] or [0.0])
        if name == 'oo_.variance_decomposition' and len(exo) == 1 and b.size > np.size(a) and np.size(a) > 0:
            # compute_variance_decomposition.m with a single shock returns ones(size(ghu,1),1): endo_nbr rows
            # (all 100), not one per requested variable
            b = b.ravel()[:np.size(a)]
        if re.match(r'oo_\.dr\.ghx$', name) and re.search(r'partial_information', text):
            # dr1_PI.m stores the PCL state space G1pi (ghu = [I; 0]), whose basis comes from svd(a0)'s null
            # spaces: a similarity transform of MATLAB's (the IRFs and moments are what is compared)
            unm.append(name + ' (PCL basis-dependent)')
            continue
        rel, st, note = cmp_arrays(a, b, floor)
        res.append(dict(name=name, rel=rel, status=st, note=note, stoch=simdata or classify_field(name, text, ML)))
    # Modes that differ by a few 1e-3 with log posteriors equal to 1e-4 are usually optimizer endpoints on a flat
    # ridge, but not always (fs2000.mod's ignored options_.solve_tolf looked exactly like that): no tolerance is
    # relaxed here; check by evaluating the log posterior at MATLAB's mode (mode_compute=0, see triage.md)
    return dict(fields=res, unmapped=sorted(set(unm)), tol=tol, est=est)


def main():
    out = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(sweep.DEF_OUT)
    entries = json.load(open(os.path.join(out, 'entries.json')))
    allres = {}
    for e in entries:
        rp = os.path.join(out, 'rs', str(e['id']), 'record.json')
        mp = os.path.join(out, 'matlab', str(e['id']), 'record.json')
        if not (os.path.exists(rp) and os.path.exists(mp)):
            continue
        simfiles = set()
        for k, t in enumerate(e['test']):
            if not t.endswith('.mod'):
                continue
            try:
                text = open(os.path.join(sweep.DEF_EXAMPLES, t[len('../examples/'):]) if t.startswith('../examples/') else os.path.join(sweep.DEF_TESTS, t)).read()
            except Exception:
                text = ''
            if STOCH_PAT.search(text):
                pass
            simdata = reads_datafile(text, simfiles)
            simfiles |= simulated_datafiles(text)
            try:
                r = compare_run(out, e['id'], k, t, text, simdata)
            except Exception as ex:
                r = dict(error='compare crashed: %r' % ex)
            if r is not None:
                r['stoch_test'] = bool(STOCH_PAT.search(text))
                mark_optimizer_endpoint(r)
                allres['%d.%d' % (e['id'], k)] = r
    json.dump(allres, open(os.path.join(out, 'compare.json'), 'w'), indent=1, default=float)
    print(len(allres), 'compared runs')


if __name__ == '__main__':
    main()
