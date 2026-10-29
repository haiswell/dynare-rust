#!/usr/bin/env python3
"""Run Dynare 7.2's test suite entries under MATLAB Dynare or dynare-rs.

usage: python3 -I sweep.py {matlab|rs} --out RESULTS_DIR [--ids 1,2,5-9] [--jobs N] [--timeout S]

RESULTS_DIR must contain entries.json (see entries.py).  Per entry it clones tests/ and
examples/ (APFS clone, cheap) into a scratch case dir, runs each .mod of the entry in
order from its directory, saves the result (record.json, plus the oo_ dump for matlab or
the <fname>/Output dir for rs) under RESULTS_DIR/<engine>/<id>/<k>/ and deletes the case
dir.  Entries containing .m tests are recorded as skipped (category e).
"""
import argparse, json, os, shutil, subprocess, sys, time, signal
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_TESTS = os.environ.get('SWEEP_TESTS', 'dynare-7.2/tests')  # Dynare 7.2 source tree's tests/
DEF_OUT = os.environ.get('SWEEP_OUT', 'sweep-out')
DEF_EXAMPLES = '/Applications/Dynare/7.2-arm64/examples'
MATLAB = '/Applications/MATLAB_R2025b.app/bin/matlab'
DYNARE_M = '/Applications/Dynare/7.2-arm64/matlab'


def parse_ids(s, n):
    if not s:
        return list(range(n))
    out = []
    for p in s.split(','):
        if '-' in p:
            a, b = p.split('-')
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(p))
    return out


def make_case(a, e):
    case = os.path.join(a.scratch, a.engine, str(e['id']))
    shutil.rmtree(case, ignore_errors=True)
    os.makedirs(case)
    subprocess.run(['cp', '-Rc', a.tests, os.path.join(case, 'tests')], check=True)
    subprocess.run(['cp', '-Rc', a.examples, os.path.join(case, 'examples')], check=True)
    return case


def run_cmd(cmd, cwd, timeout):
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         errors='replace', start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        to = False
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        to = True
    return p.returncode, out or '', time.time() - t0, to


def do_entry(a, e):
    rdir = os.path.join(a.out, a.engine, str(e['id']))
    shutil.rmtree(rdir, ignore_errors=True)
    os.makedirs(rdir)
    rec = dict(id=e['id'], test=e['test'], should_fail=e['should_fail'], runs=[])
    if any(not t.endswith('.mod') for t in e['test']):
        rec['skipped'] = 'm-test'
        json.dump(rec, open(os.path.join(rdir, 'record.json'), 'w'), indent=1)
        return rec
    case = make_case(a, e)
    try:
        for k, t in enumerate(e['test']):
            d = os.path.join(case, 'tests', os.path.dirname(t))
            mod = os.path.basename(t)
            stem = mod[:-4]
            rk = os.path.join(rdir, str(k))
            os.makedirs(rk)
            r = dict(mod=t)
            if a.engine == 'rs':
                cmd = [a.bin, mod, 'nograph', 'nointeractive', '--threads', str(a.threads), '--quiet']
                for da in e['dynare_args']:
                    cmd += ['-D', da[2:] if da.startswith('-D') else da]
                rc, out, el, to = run_cmd(cmd, d, a.timeout)
                r.update(rc=rc, elapsed=el, timeout=to, ok=(rc == 0 and not to))
                lines = out.strip().split('\n')
                errs = [l for l in lines if l.startswith('ERROR') or 'panicked' in l]
                r['error'] = ('TIMEOUT' if to else (errs[0] if errs else (lines[-1] if not r['ok'] else '')))[:500]
                r['unsupported_matlab'] = sum('MATLAB statement' in l and 'not supported' in l for l in lines)
                r['tail'] = '\n'.join(lines[-15:])[-3000:]
                for root, dirs, files in os.walk(d):
                    if os.path.basename(root) == 'Output' and os.path.basename(os.path.dirname(root)) == stem:
                        shutil.copytree(root, os.path.join(rk, 'Output'))
                        break
            else:
                script = os.path.join(rk, 'sweep_case.m')
                dargs = ''.join(", '%s'" % x for x in e['dynare_args'])
                open(script, 'w').write(
                    "addpath('%s'); addpath('%s'); addpath('%s');\ncd('%s');\nsweep_run_one('%s'%s, '%s');\n"
                    % (DYNARE_M, HERE, os.path.join(os.path.abspath(a.tests), 'utils'), d, mod, dargs,
                       os.path.join(rk, 'oo.json')))  # tests/utils: fataltest.m, as Dynare's test runner
                cmd = [MATLAB, '-batch', "run('%s')" % script]
                rc, out, el, to = run_cmd(cmd, rk, a.timeout)
                st = {}
                sp = os.path.join(rk, 'oo.json.status.json')
                if os.path.exists(sp):
                    st = json.load(open(sp))
                r.update(rc=rc, timeout=to, elapsed=st.get('elapsed', el))
                if to:
                    r.update(ok=False, error='TIMEOUT', infra=True)
                elif st:
                    r.update(ok=st['ok'], error=st.get('error', '')[:500])
                else:
                    r.update(ok=False, error='NO STATUS (matlab crashed?): ' + out.strip()[-300:], infra=True)
                lines = out.strip().split('\n')
                r['tail'] = '\n'.join(lines[-15:])[-3000:]
            rec['runs'].append(r)
            if not r['ok'] and not e['should_fail']:
                break
    finally:
        shutil.rmtree(case, ignore_errors=True)
    json.dump(rec, open(os.path.join(rdir, 'record.json'), 'w'), indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('engine', choices=['matlab', 'rs'])
    ap.add_argument('--out', default=DEF_OUT)
    ap.add_argument('--entries', default=None)
    ap.add_argument('--ids', default='')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--timeout', type=int, default=600)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--bin', default=os.path.join(HERE, '../../target/release/dynare-rs'))
    ap.add_argument('--tests', default=DEF_TESTS)
    ap.add_argument('--examples', default=DEF_EXAMPLES)
    ap.add_argument('--scratch', default=None)
    a = ap.parse_args()
    a.bin = os.path.abspath(a.bin)
    a.out = os.path.abspath(a.out)
    a.scratch = a.scratch or os.path.join(a.out, '_scratch')
    entries = json.load(open(a.entries or os.path.join(a.out, 'entries.json')))
    todo = [entries[i] for i in parse_ids(a.ids, len(entries))]
    done = [0]

    def work(e):
        try:
            r = do_entry(a, e)
        except Exception as ex:
            r = dict(id=e['id'], crash=str(ex))
        done[0] += 1
        ok = [x.get('ok') for x in r.get('runs', [])]
        print('[%d/%d] %s %s %s' % (done[0], len(todo), e['id'], e['test'][0], ok or r.get('skipped') or r.get('crash')), flush=True)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(work, todo))


if __name__ == '__main__':
    main()
