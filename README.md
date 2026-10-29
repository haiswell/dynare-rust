# dynare-rust

[![CI](https://github.com/haiswell/dynare-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/haiswell/dynare-rust/actions/workflows/ci.yml)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

An unofficial Rust port of [Dynare 7.2](https://www.dynare.org) for solving, simulating and
estimating DSGE models from `.mod` files. `dynare-rs` reads Dynare `.mod` files and runs them
without MATLAB or Octave, translating Dynare's algorithms (same branches, tolerances, defaults
and fallbacks) so that it reproduces Dynare's numbers, not just its methods. This project is
not affiliated with or endorsed by the Dynare Team.

## Features

- Preprocessor: macro language, auxiliary variables, symbolic derivatives, `steady_state_model`,
  model editing, JSON output, block decomposition.
- Steady state, perturbation at orders 1-3 (pruning, GIRFs, theoretical and simulated moments).
- Perfect foresight, expectation errors, controlled paths, extended path, OccBin, optimal
  policy (Ramsey, discretion, OSR).
- Kalman filters and smoothers (diffuse, univariate, measurement errors, missing data).
- Maximum-likelihood and Bayesian estimation: all mode finders, MCMC samplers (with parallel
  chains), SMC, `mode_check`, Laplace and harmonic-mean marginal densities.
- Forecasts, conditional forecasts, shock and realtime decompositions.
- Identification, global sensitivity analysis, method of moments, particle filters.
- BVAR, MS-SBVAR, VAR/PAC models, heterogeneous-agent models, LaTeX output and figures.

Coverage: of the 173 commands, blocks and macro directives of the Dynare 7.2 reference manual,
153 are ported and 20 are not applicable (MATLAB-only or code-generation features). See
[docs/COVERAGE.md](docs/COVERAGE.md) for every command and option, and
[docs/FEATURES.md](docs/FEATURES.md) for the details per area.

## Build

Requirements: Rust 1.89 or later. On macOS, Apple's Accelerate framework provides BLAS/LAPACK;
on Linux, install OpenBLAS (`sudo apt-get install libopenblas-dev`; another library can be
named with `DYNARE_RS_LAPACK=lapack,blas`, and a library directory added with
`DYNARE_RS_LAPACK_DIR`). On Windows, see [Windows](#windows) below.

```sh
git clone https://github.com/haiswell/dynare-rust
cd dynare-rust
cargo build --release          # target/release/dynare-rs
cargo install --path .         # optional: puts dynare-rs on the PATH
cargo test --release           # the test suite (no MATLAB needed)
```

macOS on Apple silicon is the validated platform. Linux and Windows build and run.

### Windows

Requirements: Rust with the `x86_64-pc-windows-msvc` toolchain ([rustup](https://rustup.rs),
or `winget install Rustlang.Rustup`) and the Visual Studio C++ Build Tools (MSVC linker).
`scripts/setup-windows.ps1` downloads the prebuilt OpenBLAS (`OpenBLAS-<version>-x64.zip`,
32-bit integers) into `%LOCALAPPDATA%\OpenBLAS` and sets, for the current user,
`DYNARE_RS_LAPACK=libopenblas`, `DYNARE_RS_LAPACK_DIR=<prefix>\lib` and `<prefix>\bin` on
`PATH` (where `dynare-rs.exe` finds `libopenblas.dll` at run time):

```powershell
git clone https://github.com/haiswell/dynare-rust
cd dynare-rust
powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1
# open a new terminal so the variables are picked up, then
cargo build --release          # target\release\dynare-rs.exe
```

`.gitattributes` checks text out with LF on every platform, as the tests compare outputs byte
for byte with the references. `--window` opens the progress viewer in a new console window. The test
suite passes on Windows with OpenBLAS 0.3.34 and the MSVC CRT.

The references come from MATLAB on macOS, and the tests keep their tightest tolerances there.
On Linux and Windows a few comparisons use a wider bound, given at each test with the cause:
OpenBLAS's Schur vectors, glibc's or the MSVC CRT's last-bit libm differences carried through
Newton iterations and finite differences, or an optimizer that stops at a slightly different
point (1e-6 to 1e-14 relative).

## Usage

```sh
dynare-rs [OPTIONS] model.mod [DYNARE_OPTIONS]...
```

`DYNARE_OPTIONS` are Dynare's own options (`nograph`, `savemacro`, `nolog`, `json=compute`,
`-DNAME=VALUE`, `-I<path>`, ...). Extra options include `--seed N`, `--mh-replic N`,
`--threads N`, `--blas accelerate|rust` and `--quiet`; `dynare-rs --help` lists them all.

Standard output is Dynare's console output. dynare-rs's own notes (files written, timings, MCMC
progress) go to stderr when it is a terminal; set `DYNARE_RS_VERBOSE=1` to keep them when the
output is redirected to a log file. `dynare-progress <dir>` follows a run from its progress
file either way.

Long MCMC runs can be memory-bound. `--low-memory` stores the chains run-length encoded (a
rejected proposal repeats the previous draw, so about the acceptance rate's share of the draws
is kept), reads the MH files back one chain at a time and runs the post-MCMC statistics on four
threads; the results are identical, the posterior statistics slower. With OpenBLAS (Linux,
Windows), `--openblas-threads 1` also removes OpenBLAS's per-thread buffers (about 2 GB on 28
threads) and is usually faster on models of this size, but it changes results at the rounding
level. A 6 x 1M-draw post-processing of a 56-parameter model went from 10.6 GB to 6.3 GB with
`--low-memory` and to 3.4 GB with both. Lowering `sub_draws` reduces the memory of the posterior
smoothed objects and forecasts.

Outputs go where Dynare puts them (`<dir of .mod>/<fname>/Output/`, `<fname>/metropolis/`,
`<fname>/graphs/`); mode files and MCMC files use Dynare's format, so either program can read
the other's.

Example, with a model shipped with Dynare 7.2:

```sh
mkdir -p /tmp/bkk && cp reference/examples72/bkk_1992/bkk_1992.mod /tmp/bkk/
target/release/dynare-rs /tmp/bkk/bkk_1992.mod nograph
```

This prints the steady state, eigenvalues, decision rules, moments and IRFs, as Dynare does.

## Accuracy

The test suite (about 1,000 tests) compares dynare-rust with Dynare 7.2 reference outputs
committed under `reference/`: decision rules, likelihoods, smoothers, forecasts and most other
outputs agree to 1e-10 or better. Results that depend on random draws (MCMC, simulations,
particle filters) use dynare-rust's seeded ChaCha8 streams instead of MATLAB's generators, so
they agree statistically, and to rounding when fed the same draws.
Dynare 7.2's own test suite (the 708 entries of its `tests/meson.build`) is run under both
programs by `scripts/sweep/`; `scripts/sweep/triage.md` classifies every entry (agreeing, random
draws, optimizer endpoints, MATLAB code in the .mod, examined differences).
[docs/REFERENCES.md](docs/REFERENCES.md) describes the reference models, how the references
are generated and the known differences from Dynare.

## Performance

Apple M5 Pro (18 cores), MATLAB R2025b + Dynare 7.2, on the public Smets-Wouters (2007)
reference model `reference/sw07/sw07.mod`:

| | MATLAB + Dynare 7.2 | dynare-rs |
|---|---|---|
| one posterior evaluation at the mode, 1 thread | 1.64 ms | 0.48 ms |
| full estimation run from mode file (Laplace, smoother) | 3.1 s (incl. MATLAB start) | 0.01 s |
| `mh_tune_jscale` | ~136 s | 69 s |
| 6 chains × 10,000 MH draws | 157 s | 32 s |

dynare-rs runs the MCMC chains in parallel (MATLAB runs them one after the other) and
pre-fetches likely proposals on spare cores; the draws do not depend on the thread count. To
reproduce, build the release binary and run `sh reference/sw07/timing.sh [JSCALE]` (needs MATLAB
and Dynare 7.2; see the script header).

## Limitations

- MATLAB code is not executed: `<fname>_steadystate.m` files are accepted only when they follow
  Dynare's template (they are translated into a `steady_state_model` block), external
  functions written in MATLAB work for simple scalar code only, and MATLAB statements and
  `verbatim` blocks are interpreted for the common cases only (`options_.X = V`,
  `if isoctave ... else ... end` with `isoctave` false, `load 'X.mat'` followed by a `for` loop
  of `set_param_value(param, s.params.(param))` over `fieldnames(s.params)`).
- MEX-only and code-generation features (`use_dll`, `bytecode`, ...) are not applicable;
  `mode_compute=13` (`lsqnonlin`) is replaced by a bounded Levenberg-Marquardt.
- Random draws come from dynare-rust's own streams, not MATLAB's Mersenne twister.
- A few results that Dynare decides by rounding (numerically multiple unit roots, forecast-error
  covariances near `kalman_tol`) can differ.
- Options that Dynare 7.2 itself ignores (e.g. the OccBin particle-filter and posterior
  importance sampling options) are accepted with a warning; [docs/COVERAGE.md](docs/COVERAGE.md)
  gives the status of every option and [docs/REFERENCES.md](docs/REFERENCES.md) the known
  differences.
- The library API (`dynare::...`) is not stable; the command-line interface follows Dynare's.

## License and credits

GPL-3.0-or-later, the license of Dynare; see [LICENSE](LICENSE).

dynare-rust is a port of [Dynare](https://www.dynare.org), Copyright (C) 1996-2026 Dynare Team.
The algorithms are translated from Dynare's MATLAB and C++ sources, the reference data under
`reference/` was produced by running Dynare, and the example models under
`reference/examples72/` are Dynare's. See [NOTICE](NOTICE) for details and third-party content.

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md), [DESIGN.md](DESIGN.md) and
[CHANGELOG.md](CHANGELOG.md).
