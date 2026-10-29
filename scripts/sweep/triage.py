#!/usr/bin/env python3
"""Aggregate sweep records + compare.json into categories and write triage.md.

usage: python3 -I triage.py [RESULTS_DIR] [OUT_MD]
Categories: a agree, a0 both ok/nothing comparable, b numbers differ, s stochastic-only diffs,
c rs fails (MATLAB ok), d rs ok where MATLAB fails / xfail not failing, e needs MATLAB code,
f MATLAB timeout/infra, g both fail unexpectedly, x both fail as expected (xfail).
"""
import collections, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa

out = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(sweep.DEF_OUT)
md_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'triage.md')
entries = json.load(open(os.path.join(out, 'entries.json')))
cmp_ = json.load(open(os.path.join(out, 'compare.json'))) if os.path.exists(os.path.join(out, 'compare.json')) else {}

AREAS = [  # (regex on error text or test path, suspected src area)
    (r'verbatim|MATLAB|\.m\b', 'user MATLAB code (by design)'),
    (r'declared twice|unknown symbol|statement not terminated|unexpected token|parse|syntax|preprocess|macro|@#', 'src/modfile (parser/macro processor)'),
    (r'Blanchard|BK|eigenvalue|singular|rank|no steady state|steady|Jacobian', 'src/solver, src/model (steady state / DR)'),
    (r'estimation|prior|mode|Hessian|optimi|mcmc|posterior', 'src/estimation'),
    (r'perfect|det|simul|endval|histval|homotopy', 'src/deterministic'),
    (r'occbin|OccBin', 'src/occbin'),
    (r'smoother|kalman|filter', 'src/kalman'),
    (r'ramsey|osr|optimal|planner|discretion|policy', 'src/policy'),
    (r'ep\b|extended path', 'src/ep'),
    (r'forecast|conditional', 'src/forecast'),
    (r'bvar|var_model|trend_component|pac|var_expectation', 'src/varmodels, src/bvar'),
    (r'heterogen|hank|krusell|sequence', 'src/heterogeneity'),
    (r'identif', 'src/identification'),
    (r'moment|gmm|smm|method_of_moments', 'src/mom'),
    (r'ms_|sbvar|markov', 'src/msbvar'),
    (r'particle|nonlinear|sequential', 'src/nonlinear'),
    (r'report|latex|graph', 'src/reporting, src/latex, src/graphics'),
    (r'sensitiv|gsa', 'src/gsa'),
]


def area(txt, path=''):
    for pat, a in AREAS:
        if re.search(pat, txt, re.I):
            return a
    for pat, a in AREAS[1:]:
        if re.search(pat, path, re.I):
            return a
    return '?'


def load(eng, i):
    p = os.path.join(out, eng, str(i), 'record.json')
    return json.load(open(p)) if os.path.exists(p) else None


def norm_err(s):
    s = re.sub(r'ERROR:\s*', '', s or '')
    s = re.sub(r'[A-Za-z0-9_./-]+\.mod:\d+:?', '<mod>:', s)
    s = re.sub(r"'[^']*'", "'X'", s)
    s = re.sub(r'\d+(\.\d+)?', 'N', s)
    return s[:140]


def read_mod(t):
    try:
        p = os.path.join(sweep.DEF_EXAMPLES, t[len('../examples/'):]) if t.startswith('../examples/') else os.path.join(sweep.DEF_TESTS, t)
        return open(p).read()
    except Exception:
        return ''


MATLABISH = re.compile(r'isoctave|ismac|ispc|isunix|\boo_\b|\bM_\b|\boptions_\b|~|\berror\s*\(|\bdisp\s*\(|fprintf|\bif\b|\bfor\b|\bwhile\b|\bcd\b|\bclear|\bassert|\bcontains\(|\bmkdir|\bdelete\(|\brmdir|\bexist\(|\bstruct\(|\brand|\bnorm\(|\bmax\(|\bmin\(|\babs\(|\bsum\(|\bcell|\bzeros\(|\bones\(|\beye\(|\bsize\(|\bnumel|\bplot|\bfigure|\bfieldnames|\bisfield|\bstrcmp|\bsprintf|\bnum2str|\bfunction\b|\bglobal\b|\bwarning\(|\bpause|\bkron\(|\binv\(|\bdiag\(|\bchol\(|\bcumsum|\bmean\(|\bstd\(|\bsqrt\(|\blength\(|\bisequal|\bany\(|\ball\(|\bnnz|\bfind\(|\bsort|\bunique|\bevalin|\beval\(|\bfeval|\btic\b|\btoc\b|\bdatestr|\bclock|\bnow\b|\bsetenv|\bgetenv|\bsystem\(|\bunix\(|\bstr2|\bupper|\blower|\bisnan|\bisinf|\bfix\(|\bfloor|\bceil|\bround|\bmod\(|\brem\(|\bdet\(|\brank\(|\beig\(|\bsvd|\bexpm|\bpinv|\btrace|\btril|\btriu|\breshape|\brepmat|\bcat\(|\bhorzcat|\bvertcat|\bisempty|\bcolon|\bnumel|\bcorr|\bcov\(|\bsubplot|\bgca|\bgcf|\bsaveas')


def matlabish_error(err, t):
    m = re.match(r'(?:ERROR:\s*)?([^\s:]*):(\d+):', err or '')
    if not m:
        return False
    fn, ln = m.group(1), int(m.group(2))
    base = os.path.join(sweep.DEF_EXAMPLES, t[len('../examples/'):]) if t.startswith('../examples/') else os.path.join(sweep.DEF_TESTS, t)
    d = os.path.dirname(base)
    cands = [base] if not fn or fn == os.path.basename(base) else []
    for dd in (d, os.path.dirname(d), os.path.join(d, '..', '..')):
        cands.append(os.path.join(dd, fn))
    for c in cands:
        if os.path.isfile(c):
            try:
                line = open(c, errors='replace').read().split('\n')[ln - 1]
            except Exception:
                return False
            line = re.sub(r'%.*', '', line)
            return bool(MATLABISH.search(line))
    return False


# Entries examined by hand: the outcome differs from Dynare's for a reason that is not a dynare-rs bug.
REVIEWED = {
    'ep/burnside.mod': 'MATLAB code: set_dynare_seed(seed) takes a MATLAB variable (seed = 31415;)',
    'deterministic_simulations/rbc_det.mod': 'MATLAB code: histval uses CapitalSS = oo_.steady_state(ik), a MATLAB variable',
    'deterministic_simulations/rbc_det1.mod': 'MATLAB code: histval uses CapitalSS = oo_.steady_state(ik), a MATLAB variable',
    'deterministic_simulations/rbc_det6.mod': 'MATLAB code: histval uses CapitalSS = oo_.steady_state(ik), a MATLAB variable',
    'deterministic_simulations/lbj/rbc.mod': 'MATLAB code: histval uses CapitalSS = oo_.steady_state(ik), a MATLAB variable',
    'block_bytecode/lola_solve_one_boundary.mod': 'MATLAB code: NBR_iss etc. come from load lola_data.mat into the workspace',
    'deterministic_simulations/purely_forward/nk.mod': 'MATLAB code: tmp = inv(eye(3)-...)*... checks the analytical solution',
    'analytic_derivatives/burnside_3_order_PertParamsDerivs.mod': "MATLAB code: error(['Error in matrix oo_.dr.g_' num2str(ord)]) in derivative checks",
    'estimation/univariate/nls/dynamicmodel1.mod': 'MATLAB code: estimate.nls(...) package call inside an if block',
    'estimation/univariate/nls/staticmodel.mod': 'MATLAB code: estimate.nls(...) package call inside an if block',
    'estimation/MH_recover/fs2000_recover_tmp.mod': 'MATLAB code: loads the MH files and compares temp1.x2/temp2.x2',
    'estimation/MH_recover/fs2000_slice_recover.mod': 'MATLAB code: loads the MH files and compares temp1.x2/temp2.x2',
    'estimation/MH_recover/fs2000_recover.mod': 'MATLAB code: MH files are copied and deleted with copyfile/delete before mh_recover',
    'estimation/MH_recover/fs2000_recover_2.mod': 'MATLAB code: MH files are copied and deleted with copyfile/delete before mh_recover',
    'estimation/MH_recover/fs2000_recover_3.mod': 'MATLAB code: MH files are copied and deleted with copyfile/delete before mh_recover',
    'analytic_derivatives/fs2000_analytic_derivation.mod': 'MATLAB code: my_data.mat is saved after adding randn measurement error in verbatim',
    'analytic_derivatives/fs2000_analytic_derivation_diag_H.mod': 'MATLAB code: my_data.mat is saved after adding randn measurement error in verbatim',
    'particle/first_spec_hssmc.mod': "MATLAB code: save('my_data.mat') of a simulation drawn with MATLAB's randn",
    'heterogeneity/large_file_download/hank_1a_ssj.mod': 'MATLAB code: hank_1a_ssj.mat is downloaded with websave',
    'model-inversion/bk-diff-3/varmodel.mod': 'MATLAB code: initialconditions dseries built from MATLAB matrices',
    'heterogeneity/hank_1a.mod': "MATLAB code: parameters set from load 'X.mat' and a for loop over set_param_value",
    'heterogeneity/hank_2a.mod': "MATLAB code: parameters set from load 'X.mat' and a for loop over set_param_value",
    'heterogeneity/ks.mod': "MATLAB code: parameters set from load 'X.mat' and a for loop over set_param_value",
    'heterogeneity/ks_two_shocks.mod': "MATLAB code: parameters set from load 'X.mat' and a for loop over set_param_value",
    'heterogeneity/hank_1a_steady_state.mod': "MATLAB code: parameters set from load 'X.mat' and a for loop over set_param_value",
    'heterogeneity/hank_2a_steady_state.mod': "MATLAB code: parameters set from load 'X.mat' and a for loop over set_param_value",
    'heterogeneity/ks_steady_state.mod': "MATLAB code: parameters set from load 'X.mat' and a for loop over set_param_value",
    'heterogeneity/ks_steady_state_dense.mod': "MATLAB code: parameters set from load 'X.mat' and a for loop over set_param_value",
    'heterogeneity/large_file_download/ks_ssj.mod': 'MATLAB code: ks_ssj.mat is downloaded with websave',
    'occbin/model_irrcap_twoconstraints/dynrbc.mod': "MATLAB code: shockssequence drawn with randn('seed',1) inside a MATLAB if",
    'ecb/SURGibbs/fulton_fish.mod': 'MATLAB code: surgibbs (MATLAB randn) is not ported',
    'estimation/fs2000_MCMC_jumping_covariance.mod': 'MATLAB code: test_matrix.mat is built from the mode file with load/save',
    'analytic_derivatives/fs2000_analytic_derivation_newrat.mod': 'MATLAB code: my_data.mat is saved after adding randn measurement error in verbatim',
    'measurement_errors/fs2000_corr_me_ml_mcmc/fs2000_corr_ME.mod': 'MATLAB code: fsdat_simul_corr.mat is built in verbatim with randn',
    'observation_trends_and_prefiltering/ML/Trend_no_prefilter_selected_var.mod': 'MATLAB code: the data file is written by generate_trend_stationary_AR1.m',
    'particle/first_spec.mod': "MATLAB code: save('my_data.mat') of a simulation drawn with MATLAB's randn",
    'particle/first_spec_MCMC.mod': "MATLAB code: save('my_data_MCMC.mat') of a simulation drawn with MATLAB's randn",
    'estimation/method_of_moments/CET/cet_interface.mod': 'MATLAB code: @(j) anonymous function in cet_model.inc',
    'trend-component-and-var-models/tcm1.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/tcm2.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/tcm3.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/tcm4.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/tcm5.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/tcm9.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/tcm10.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/tcm11.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/tcm12.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/vm1.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/vm2.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/vm3.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/vm5.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/legacy/tcm1.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/legacy/tcm2.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/legacy/tcm3.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/legacy/tcm4.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/legacy/tcm5.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/legacy/vm1.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/legacy/vm2.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'trend-component-and-var-models/legacy/vm3.mod': 'MATLAB code: parameters calibrated from MATLAB matrices (A1 = randn(n); A111 = A1(1,1))',
    'model-inversion/bk-3a/varmodel.mod': 'MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P))',
    'model-inversion/bk-3b/varmodel.mod': 'MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P))',
    'model-inversion/bk-4/varmodel.mod': 'MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P))',
    'model-inversion/bk-5/varmodel.mod': 'MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P))',
    'model-inversion/bk-6/varmodel.mod': 'MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P))',
    'model-inversion/bk-7/varmodel.mod': 'MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P))',
    'model-inversion/bk-8/varmodel.mod': 'MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P))',
    'model-inversion/bk-diff-1/varmodel.mod': 'MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P))',
    'model-inversion/bk-diff-4/varmodel.mod': 'MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P))',
    'model-inversion/bk-diff-5/varmodel.mod': 'MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P))',
    'model-inversion/bk-3a-nlsolve/varmodel.mod': 'MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P))',
    'model-inversion/bk-3b-nlsolve/varmodel.mod': 'MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P))',
    'model-inversion/bk-4-nlsolve/varmodel.mod': 'MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P))',
    'model-inversion/bk-5-nlsolve/varmodel.mod': 'MATLAB code: parameters a11 = A(1,1) from a MATLAB matrix (P = randn(3,3); A = P*D*inv(P))',
    'pac/trend-component-19-growth-lin-comb/example1.mod': 'MATLAB code: eparams initial values read from M_.params',
    'pac/trend-component-21/example1.mod': "RNG: the estimated share goes to zero on dynare-rs's simulated data (MATLAB fails the same way on that data)",
    'pac/trend-component-23/example1.mod': "RNG: the estimated share goes to zero on dynare-rs's simulated data (MATLAB fails the same way on that data)",
    'pac/trend-component-25/example1.mod': "RNG: the estimated share goes to zero on dynare-rs's simulated data (MATLAB fails the same way on that data)",
    'pac/var-10e/example1.mod': 'MATLAB code: example1.data.mat is written by the MATLAB method TrueData.save(...)',
    'pac/trend-component-28/example2.mod': 'MATLAB code: initialconditions built from init = rand(...)',
    'pac/trend-component-28/example3.mod': 'MATLAB code: initialconditions built from init = rand(...)',
    'decision_rules/first_order/walsh.mod': 'MATLAB code: calls the external function matlab.namespace.y_k',
    'steady_state/walsh1_initval.mod': "Dynare bug: `n=log(1/3);` with n endogenous makes the 7.2 preprocessor overwrite M_.params(1)",
    'stochastic_simulations/pruning/fs2000_pruning.mod': "MATLAB code: the ' transpose operator",
    '../examples/estimation/rbc_irf_matching.mod': 'MATLAB code: @(j) anonymous function',
    'estimation/method_of_moments/CET/cet_imh.mod': 'MATLAB code: @(j) anonymous function in cet_model.inc',
    'estimation/method_of_moments/CET/cet_mh_posterior_mode.mod': 'MATLAB code: @(j) anonymous function in cet_model.inc',
    'estimation/method_of_moments/CET/cet_mle.mod': 'MATLAB code: @(j) anonymous function in cet_model.inc',
    'estimation/method_of_moments/CET/cet_slice.mod': 'MATLAB code: @(j) anonymous function in cet_model.inc',
    'estimation/method_of_moments/CET/cet_tarb.mod': 'MATLAB code: @(j) anonymous function in cet_model.inc',
    'estimation/method_of_moments/CET/cet_rwmh.mod': 'MATLAB code: @(j) anonymous function in cet_model.inc',
    'decision_rules/ar_qz_test.mod': 'MATLAB code: try/catch',
    'model-inversion/bk-diff-2/varmodel.mod': 'MATLAB code: M_ fields and model_inversion calls',
    'solve_algo_12_14/simul_backward_reference.mod': 'MATLAB code: rand, randn, save',
    'particle/dsge_base_sis.mod': "RNG: generic resampling without propagation (as in Dynare's sequential_importance_particle_filter.m) "
                                  'underflows for some streams (4 of 8 ChaCha seeds at 5000 particles); MATLAB\'s stream passes',
    'block_bytecode/ireland.mod': 'ill-conditioned last periods: MATLAB itself stalls at 8e-11 with dynatol.f=1e-13; both within tolerance',
    'fs2000/fs2000.mod': 'optimizer endpoint on a flat ridge: log posteriors at MATLAB\'s mode agree to 3e-13',
    'kalman_initial_state/fs2000_kalman_initial_2_lag.mod': 'optimizer endpoint: log posteriors at MATLAB\'s mode agree to 1e-15',
    'smoother2histval/fs2000_smooth_stoch_simul.mod': 'MATLAB code in the entry',
}

cat = collections.defaultdict(list)  # cat -> list of dict rows
for e in entries:
    i = e['id']
    rs, ml = load('rs', i), load('matlab', i)
    name = e['test'][0] + ('' if len(e['test']) == 1 else ' (+%d)' % (len(e['test']) - 1))
    if e['dynare_args']:
        name += ' ' + ' '.join(e['dynare_args'])
    row = dict(id=i, test=name)
    if any(not t.endswith('.mod') for t in e['test']):
        cat['e'].append(dict(row, why='.m test script'))
        continue
    if rs is None or ml is None:
        cat['f'].append(dict(row, why='no result (not run)', area=''))
        continue
    mr, rr = ml['runs'], rs['runs']
    # evaluate the first run that differs in outcome; overall = worst run
    fin = None
    for k, t in enumerate(e['test']):
        m = mr[k] if k < len(mr) else None
        r = rr[k] if k < len(rr) else None
        if m is None:
            break
        if m.get('infra'):
            fin = ('f', dict(row, why='MATLAB ' + m['error'][:100], area=''))
            break
        text = read_mod(t)
        has_verb = bool(re.search(r'^\s*verbatim\b', text, re.M)) or (r and r.get('unsupported_matlab', 0) > 0)
        if not m['ok'] and 'TeX-File did not compile' in m['error']:
            # the test's own pdflatex step, after every Dynare command ran (a local LaTeX installation
            # issue): oo_ was dumped, so compare as if MATLAB succeeded
            m = dict(m, ok=True)
        if not m['ok']:
            if e['should_fail']:
                if r and not r['ok']:
                    fin = fin or ('x', dict(row, why='both fail (xfail)'))
                else:
                    fin = ('d', dict(row, why='xfail test: dynare-rs does NOT fail', area=area('', t)))
                    break
            else:
                if r and r['ok']:
                    fin = ('d', dict(row, why='MATLAB fails (%s); dynare-rs succeeds' % m['error'][:90], area=area('', t)))
                else:
                    fin = ('g', dict(row, why='MATLAB: %s | rs: %s' % (m['error'][:90], (r or {}).get('error', '')[:90]), area=area('', t)))
                break
            continue
        if r is None or not r['ok']:
            err = (r or {}).get('error', 'not run')
            if e['should_fail']:
                fin = ('d', dict(row, why='xfail test: MATLAB succeeds', area=''))
            elif re.search(r'\.m\b|MATLAB|verbatim', err) or matlabish_error(err, t):
                fin = ('e', dict(row, why='MATLAB code in/with mod: ' + err[:100]))
            elif has_verb:
                # dynare-rs skipped MATLAB statements earlier in the file, but the error is not on one
                fin = ('c2', dict(row, why=err[:160], area=area(err, t)))
            else:
                fin = ('c', dict(row, why=err, key=norm_err(err), area=area(err, t)))
            break
        # both ok
        if e['should_fail']:
            fin = ('d', dict(row, why='xfail test: both succeed', area=area('', t)))
            break
        c = cmp_.get('%d.%d' % (i, k))
        if not c or c.get('error') or not c['fields']:
            fin = fin or ('a0', dict(row, why='nothing comparable' if not c else str(c.get('error', ''))))
            continue
        tol = c['tol']
        bad = [f for f in c['fields'] if f['status'] != 'ok' or (f['rel'] or 0) > tol]
        det = [f for f in bad if not f['stoch'] and not f.get('endpoint')]
        sto = [f for f in bad if f['stoch']]
        endp = [f for f in bad if f.get('endpoint') and not f['stoch']]
        if det:
            if has_verb:
                fin = ('e', dict(row, why='numbers differ but model uses verbatim MATLAB'))
            else:
                worst = sorted(det, key=lambda f: -(f['rel'] if f['rel'] is not None and f['rel'] == f['rel'] else 1e99))[:4]
                fin = ('b', dict(row, why='; '.join('%s %s' % (f['name'].replace('oo_.', ''), ('%.1e' % f['rel']) if f['rel'] is not None else f['status'] + ' ' + f.get('note', '')) for f in worst) + (' [+%d more]' % (len(det) - 4) if len(det) > 4 else ''),
                                  area=area(' '.join(f['name'] for f in det), t), nfields=len(det)))
            break
        if endp:
            fin = ('p', dict(row, why='same log posterior, %d fields at the mode differ (max %.1e): %s' % (
                len(endp), max(f['rel'] or 0 for f in endp), ', '.join(f['name'].replace('oo_.', '') for f in endp[:4]))))
            break
        if sto:
            fin = ('s', dict(row, why='%d stochastic fields differ: %s' % (len(sto), ', '.join(f['name'].replace('oo_.', '') for f in sto[:4]))))
            break
        fin = ('a', dict(row, why='%d fields, max rel %.1e' % (len(c['fields']), max((f['rel'] or 0) for f in c['fields']))))
    if fin is None:
        fin = ('f', dict(row, why='incomplete', area=''))
    if fin[0] in ('b', 'c', 'c2', 'd', 'p') and e['test'][0] in REVIEWED:
        fin = ('r', dict(row, why=REVIEWED[e['test'][0]], area=''))
    cat[fin[0]].append(fin[1])

names = [('a', '(a) both succeed, numbers agree'), ('a0', '(a0) both succeed, no comparable output mapped'),
         ('b', '(b) both succeed, numbers differ'), ('p', '(p) optimizer endpoint: same log posterior at the mode, the mode differs'), ('s', '(s) stochastic-only differences (RNG driven)'),
         ('c', '(c) Dynare succeeds, dynare-rs fails'), ('c2', '(c2) dynare-rs fails after skipping MATLAB statements (error not on a MATLAB line)'), ('d', '(d) dynare-rs succeeds where Dynare fails / xfail not failing'),
         ('e', '(e) needs MATLAB code dynare-rs cannot run by design'), ('f', '(f) MATLAB timeout / infrastructure'),
         ('g', '(g) both fail unexpectedly'), ('x', '(x) both fail, as expected (xfail)'),
         ('r', '(r) examined: explained difference, not a dynare-rs bug')]
L = ['# Dynare 7.2 test-suite triage of dynare-rs', '', 'Entries: %d (from `mod_and_m_tests` in meson.build). Regenerate with scripts/sweep (see README.md).' % len(entries), '',
     '| category | entries |', '|---|---|']
for k, n in names:
    L.append('| %s | %d |' % (n, len(cat[k])))
L.append('')


def table(k, cols):
    rows = cat[k]
    L.append('### ' + dict(names)[k])
    L.append('')
    L.append('| test | symptom | suspected src area |')
    L.append('|---|---|---|')
    for r in sorted(rows, key=lambda r: (r.get('area', ''), r['test'])):
        L.append('| %s | %s | %s |' % (r['test'], r['why'].replace('|', '/').replace('\n', ' ')[:200], r.get('area', '')))
    L.append('')


# (c) grouped by message
L.append('## (c) grouped by error message')
L.append('')
g = collections.defaultdict(list)
for r in cat['c']:
    g[r['key']].append(r)
L.append('| #tests | normalised error | example tests | suspected area |')
L.append('|---|---|---|---|')
for key, rs_ in sorted(g.items(), key=lambda kv: -len(kv[1])):
    L.append('| %d | %s | %s | %s |' % (len(rs_), key.replace('|', '/'), ', '.join(r['test'] for r in rs_[:3]), rs_[0]['area']))
L.append('')
table('c', None)
table('c2', None)
table('b', None)
table('d', None)
table('r', None)
table('p', None)
for k in ('s', 'g', 'f'):
    table(k, None)
open(md_path, 'w').write('\n'.join(L) + '\n')
json.dump({k: v for k, v in cat.items()}, open(os.path.join(out, 'categories.json'), 'w'), indent=1)
print({k: len(v) for k, v in cat.items()})
