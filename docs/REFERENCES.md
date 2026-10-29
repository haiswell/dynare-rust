# Reference models and outputs

dynare-rust is validated against output of MATLAB + Dynare 7.2 committed under `reference/`.
The tests read only these files; MATLAB is needed only to regenerate them.

## Reference models

| directory | model | used for |
|---|---|---|
| `reference/sw07/` | Smets-Wouters (2007) with a random-walk inflation target, a random-walk hours trend and two survey observables; data simulated from the model (`gen_data.m`, fixed seed) | the main estimation benchmark: likelihood, solver, Kalman filter and smoother, forecasts, decompositions, `mode_check`, MCMC, identification, GSA, timings |
| `reference/qpm/` | Berg-Karam-Laxton (2006) quarterly projection model with a foreign block; simulated data | a semi-structural gap model with unit-root trends and long leads and lags |
| `reference/gm05/` | Gali-Monacelli (2005) small open economy in the Lubik-Schorfheide (2007) form; simulated data | a small forward-looking model with a unit-root observable |
| `reference/examples72/` | the example models shipped with Dynare 7.2 | end-to-end runs |
| `reference/<area>/` | small test models per area (OccBin, perfect foresight, k-order, MoM, BVAR, ...) | the area's integration tests |

None of the data are real: each `gen_data.m` simulates the observables at the model's
calibration. SW07 has 50 endogenous variables (45 declared + 5 auxiliary), 11 shocks, 9
observables (some missing in the first quarters), 40 estimated parameters and two unit roots
(`diffuse_filter`; at the mode the diffuse filter falls back to the univariate filter).

The SW07, QPM and GM05 files select the estimation call with a `STAGE` macro: 1 = mode search
from the calibration, 2 = refined mode search from the stage-1 mode, 3 (default) = MCMC from the
mode file with smoother and forecasts. SW07 adds 9 = the reference run (mode file, Laplace,
smoother, no MCMC) and 20/21 = GSA stability map / RMSE mapping.

## Check run

From a copy of `reference/sw07/` (dynare-rs writes its outputs next to the `.mod` file):

```sh
cp -R reference/sw07/. /tmp/sw07 && cd /tmp/sw07
dynare-rs sw07.mod -DSTAGE=3 --mh-replic 2000 --seed 1
```

prints `Log data density (Modified Harmonic Mean) is -447.590190.` (`mh_tune_jscale` gives
`mh_jscale` = 0.29819; Laplace at the mode: -442.187409, Dynare -442.187384).

## Generating the references

Each `reference/<area>/` folder holds the test models, the scripts that dump Dynare's results
(`dump_*.m`, `gen_*.py`) and their output (`ref*.json`, logs, `.mat` files). The scripts need
MATLAB and Dynare and read two environment variables:

- `DYNARE_VERSION` (default `7.2`): the Dynare version to run;
- `DYNARE_ROOT` (default `/Applications/Dynare/<DYNARE_VERSION>-arm64`): the installation (its
  `matlab/` and `preprocessor/` folders are used).

Run a script from a private copy of its folder (Dynare writes `+<fname>/` and `<fname>/` next to
the `.mod`), e.g. `DYNARE_ROOT=/opt/dynare-7.2 matlab -batch "run('dump_occbin.m')"`, and copy
the JSON back. The header of each script says what it writes. `sh reference/sw07/run_dumps.sh`
reruns all SW07 dumps in dependency order; `reference/sw07/timing.sh` is the timing protocol of
the README. Indices in the JSON files are MATLAB's (1-based).

## Known differences from Dynare 7.2

- **Random numbers.** MCMC chains, simulations, particle filters, extended path, SMM,
  conditional-forecast bands, `PointForecast` and the `sub_draws` selection use dynare-rust's
  seeded ChaCha8 streams (`--seed`, one stream per chain), not MATLAB's Mersenne twister. They
  agree with Dynare statistically, and to rounding when fed the same draws (the tests replay
  MATLAB's draws where possible). Continuing Dynare's chains (`load_mh_file` on MATLAB files)
  seeds each chain with a hash of MATLAB's saved generator state. `mh_recover` resumes every
  chain from its last saved file.
- **Rounding-decided results.** When the transition matrix has a numerically multiple unit
  root, `compute_Pinf_Pstar` may keep or drop a diffuse direction depending on the Schur basis
  LAPACK returns (MATLAB ships its own LAPACK, dynare-rust uses Accelerate's or OpenBLAS); when
  the forecast-error covariance stays near `kalman_tol`, whether the multivariate filter returns
  a value, falls back to the univariate filter or fails with info 46 can differ. `check` may
  print numerically defective multiple roots at different rounding-level values (the counts and
  the Blanchard-Kahn verdict agree).
- **`analytic_derivation`.** newrat follows a slightly different path (the final value agrees to
  ~5e-7, the Hessian at the mode is exact); a steady-state model that changes parameters is not
  supported by the estimation. The skewed filter's `mvncdf` above 3 dimensions uses a fixed seed.
- **Method of moments.** `mode_compute=13` (`lsqnonlin`) is a bounded Levenberg-Marquardt that
  reaches the same minimum by another path; IRF matching at order > 1 keeps its draws fixed
  across evaluations. Dynare 7.2 itself cannot run GMM at `order=3` (its reference was produced
  with a patched copy, see `reference/mom/dump_mom.m`).
- **Particle filters.** Dynare 7.2's quirks are reproduced (`stratified` resampling,
  `generic` resampling threshold, `nlkf` likelihood); particle smoothers and the MATLAB draws of
  `initialization=2` are not.
- **Samplers.** DSMH, SMC and the online filter use their own streams. `mode_compute=6` keeps
  Dynare's behaviour (its tuned scale becomes `mh_jscale`, its `hh` is the inverse of its
  simulated covariance). As in Dynare, the MCMC drops explicit `estimated_params` bounds and
  keeps only the prior support.
- **Remaining gaps found by the 7.2 parity audit.**
  - The Lyapunov solver does not switch to the Schur method for the rest of the session after one
    fixed-point failure (Dynare's persistent switch would make a posterior evaluation depend on the
    previous ones and break the MCMC pre-fetching).
  - OccBin estimation refuses `lik_init=3` and `loglinear`.
  - Nonlinear estimation stops at `order=3`, because the k-order solver does.
  - The numbering of the Ramsey `AUX_ENDO_LEAD` variables can differ from the preprocessor's;
    the model is the same.
  - The `var_expectation_model` parameter name of a diff-LHS VAR counts only the variables of
    the `expression` option.
  - `param_init` in `modfile.json` holds values, not the expressions.
- **Not supported.** `<fname>_steadystate.m` files other than Dynare's template, MATLAB external
  functions beyond simple scalar code, multi-output assignments in `steady_state_model`,
  welfare at order 3, MEX-only and
  code-generation options (`use_dll`, `bytecode`). `docs/COVERAGE.md` lists every command and
  option with its status.
