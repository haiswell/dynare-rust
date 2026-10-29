#!/usr/bin/env python3
"""Print the structure of a JSON file (debug helper). usage: show_json.py FILE [maxdepth]"""
import json, sys

d = json.load(open(sys.argv[1]))
maxdep = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def show(o, p='', dep=0):
    if isinstance(o, dict):
        if set(o) == {'size', 'data'}:
            return
        for k, v in o.items():
            if isinstance(v, dict) and set(v) == {'size', 'data'}:
                print(p + k, 'ARR', v['size'])
            else:
                n = len(v) if hasattr(v, 'len') or isinstance(v, (list, dict, str)) else v
                print(p + k, type(v).__name__, n)
                if dep < maxdep:
                    show(v, p + '  ', dep + 1)


show(d)
