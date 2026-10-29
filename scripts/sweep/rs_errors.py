#!/usr/bin/env python3
"""Cluster dynare-rs failures by normalised error message (independent of the MATLAB side). Debug helper."""
import collections, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep

out = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(sweep.DEF_OUT)
g = collections.defaultdict(list)
n = ok = 0
for p in glob.glob(os.path.join(out, 'rs', '*', 'record.json')):
    r = json.load(open(p))
    for run in r.get('runs', []):
        n += 1
        if run['ok']:
            ok += 1
            continue
        e = re.sub(r'ERROR:\s*', '', run['error'])
        e = re.sub(r'[A-Za-z0-9_./-]+\.mod:\d+:?', '<mod>:', e)
        e = re.sub(r"'[^']*'", "'X'", e)
        e = re.sub(r'\d+(\.\d+)?', 'N', e)[:150]
        g[e].append(run['mod'])
print('runs', n, 'ok', ok)
for e, m in sorted(g.items(), key=lambda kv: -len(kv[1]))[:int(sys.argv[2]) if len(sys.argv) > 2 else 40]:
    print(len(m), e, '|', m[0])
