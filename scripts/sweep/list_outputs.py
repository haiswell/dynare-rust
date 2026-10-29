#!/usr/bin/env python3
"""Histogram of dynare-rs output-file suffixes (after the <fname>_ prefix) across a sweep. Debug helper."""
import glob, json, os, sys, collections

out = sys.argv[1]
cnt = collections.Counter()
ex = {}
for rec in glob.glob(os.path.join(out, 'rs', '*', 'record.json')):
    r = json.load(open(rec))
    for k, run in enumerate(r.get('runs', [])):
        stem = os.path.basename(run['mod'])[:-4]
        for f in glob.glob(os.path.join(os.path.dirname(rec), str(k), 'Output', '*')):
            b = os.path.basename(f)
            if b.startswith(stem + '_'):
                b = '<fname>' + b[len(stem):]
            cnt[b] += 1
            ex.setdefault(b, f)
for b, n in cnt.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 60):
    print(n, b, ex[b])
