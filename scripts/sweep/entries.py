#!/usr/bin/env python3 -I
"""Parse mod_and_m_tests out of Dynare's meson.build into entries.json."""
import re, json, sys
src, out = sys.argv[1], sys.argv[2]
txt = open(src).read()
a = txt.index('mod_and_m_tests = [')
b = txt.index('base_test_driver_args')
body = txt[a:b]
# strip comments
body = '\n'.join(l.split('  #')[0] if l.strip().startswith('#') is False else '' for l in body.split('\n'))
parts = re.split(r'\n  \{ (?=\'test\')', body)[1:]
entries = []
for i, p in enumerate(parts):
    test = re.search(r"'test'\s*:\s*\[([^\]]*)\]", p).group(1)
    test = re.findall(r"'([^']*)'", test)
    m = re.search(r"'extra'\s*:\s*\[([^\]]*)\]", p)
    extra = re.findall(r"'([^']*)'", m.group(1)) if m else []
    m = re.search(r"'dynare_args'\s*:\s*\[([^\]]*)\]", p)
    dargs = re.findall(r"'([^']*)'", m.group(1)) if m else []
    sf = bool(re.search(r"'should_fail'\s*:\s*true", p))
    entries.append(dict(id=i, test=test, extra=extra, dynare_args=dargs, should_fail=sf))
json.dump(entries, open(out, 'w'), indent=1)
n_mod = sum(1 for e in entries if e['test'][-1].endswith('.mod') and len(e['test']) == 1)
print(len(entries), 'entries;', n_mod, 'single .mod;', sum(e['should_fail'] for e in entries), 'should_fail;',
      sum(len(e['test']) > 1 for e in entries), 'multi-file')
