# Dynare test-suite sweep

Runs Dynare 7.2's own `tests/` (the `mod_and_m_tests` list in `meson.build`) under MATLAB
Dynare and under `dynare-rs`, compares the numbers and classifies every entry.
Raw logs and dumps live outside the repo: set `SWEEP_TESTS` (Dynare 7.2's `tests/`) and `SWEEP_OUT` (results dir).

Inputs: the Dynare 7.2 source tarball (`tests/`, `meson.build`), the installed
`/Applications/Dynare/7.2-arm64` (its `examples/` is used for the `../examples/...` entries and its
`matlab/` is the MATLAB path), MATLAB R2025b, a release build of dynare-rs (`cargo build --release`).

```
S=<results dir>                      # default: see DEF_OUT in sweep.py
python3 -I entries.py <dynare-7.2>/meson.build $S/entries.json   # parse the test list
python3 -I sweep.py rs     --jobs 4 --threads 2 --timeout 600     # dynare-rs side (also: --ids 3,10-20)
python3 -I sweep.py matlab --jobs 6 --timeout 600                 # MATLAB side, one `matlab -batch` per .mod
python3 -I compare.py                                             # -> $S/compare.json
python3 -I triage.py                                              # -> triage.md, $S/categories.json
```

* `entries.py`   parse meson.build into entries (test files, extra files, `dynare_args`, `should_fail`).
* `sweep.py`     per entry: APFS-clone `tests/` + `examples/` into a scratch dir, run each `.mod` of the entry
  in order from its own directory (`dynare <mod> nograph nointeractive [-Dx=y]` / `dynare-rs <mod> nograph nointeractive`),
  record ok/error/elapsed in `<engine>/<id>/record.json`; MATLAB side also dumps `oo_` and key `M_` fields
  (`sweep_run_one.m`, numerics as `{size,data}` column-major, >1e6 elements skipped); rs side keeps `<fname>/Output`.
  Entries with `.m` test scripts are recorded as skipped (category e). Timeout kills the process group.
* `compare.py`   maps dynare-rs outputs (`*_stoch_simul.json`, `*_results.json`, `*_endo_simul.csv`,
  smoothed/filtered/updated csv, `*_IRF_<shock>.csv`, forecast csv, ramsey json) onto `oo_` paths and reports the
  normwise max relative difference per field (max|a-b| / max|b|). Tolerance 1e-6, 1e-3 for estimation/optimizer tests.
  Fields that are RNG driven (simulated moments, MCMC posterior) are flagged `stoch`.
* `triage.py`    categorises (see header of triage.md); a failure whose offending `.mod` line is MATLAB code
  (`if ~isoctave`, `oo_.x`, `error(...)`, ...) or whose model uses `verbatim` is put in (e).
* Debug helpers: `show_json.py`, `list_outputs.py`, `rs_errors.py`.

Caveats: the comparator only sees fields dynare-rs writes to `Output/` and that it knows how to map; tests that
produce neither are `(a0)`. Parallel MATLAB workers are sensitive to machine load: the 600 s timeout may report
`(f)` on slow/loaded runs; rerun those with `--ids` and a larger `--timeout`.

## Test data

The `.mat` files under Dynare's `tests/` are stored with git LFS. Without `git lfs`, a clone has 130-byte pointer
files instead (MATLAB: `load:notBinaryFile`); fetch them from
`https://git.dynare.org/Dynare/dynare/-/raw/<commit>/<path>` and check the `size` line of each pointer.

Tests that compile their LaTeX output run with graphs on; with a minimal TeX installation (BasicTeX lacks
`morefloats.sty`) their final pdflatex step fails in MATLAB, and triage puts them in (f).

## Comparator

IRFs at order >= 2 are simulated and flagged `stoch` (MATLAB's `oo_.dr.ghxx` exists). IRFs are scaled by the largest
response to the same shock, forecasts by the largest value over all variables. Non-finite values compare by finiteness
(dynare-rs writes +-inf as JSON null); fields MATLAB leaves empty are skipped; a single-shock variance decomposition is
truncated as Dynare returns `ones(endo_nbr,1)`; a Ramsey `steady_state` is compared with `oo_.dr.ys`.

## Status

Full run 2026-10-28 (MATLAB all 708 entries, LFS data fetched); triage.md is the result before the fixes merged after it.
