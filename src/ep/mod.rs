// SPDX-License-Identifier: GPL-3.0-or-later
//! Extended path and stochastic extended path: port of `ep/extended_path.m`,
//! `extended_path_initialization.m`, `extended_path_shocks.m`, `extended_path_core.m`,
//! `extended_path_homotopy.m`, `setup_stochastic_perfect_foresight_model_solver.m`,
//! `solve_stochastic_perfect_foresight_model_{0,1}.m` and `ep_problem_{0,1}.m`.
//!
//! Every period `t` solves a perfect-foresight problem over `ep.periods` (`solver_periods`)
//! periods from the state of `t - 1` with the shock of `t` and no future shocks (order 0),
//! or, at `order = S > 0`, with the expectations of the next `S` periods integrated over a
//! tree of quadrature nodes (Gauss-Hermite tensor rule, Stroud cubature or the unscented
//! transform; full tree or Dynare's sparse "fishbone" tree), whose residuals are evaluated in
//! parallel. Every problem is solved by `dynare_solve`'s trust region (`ep.solve_algo = 9`,
//! the default; `ep.stack_solve_algo = 7`) on the sparse stacked Jacobian.

pub mod controlled;
pub mod datafiles;
pub mod decl;
pub mod pfwee;
pub mod quadrature;
pub mod sparse;
pub mod tree;
pub mod trust_region;

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::deterministic::DetState;
use crate::deterministic::stacked::StackedStructure;
use crate::linalg::{Mat, chol_upper};
use crate::model::Model;
use crate::modfile::{Command, ModFile, OptValue, Options, opt};
pub use decl::EpDecl;
use quadrature::Integration;
use sparse::{LuPlan, SparseJac};
use tree::{TreeKind, TreeProblem};
use trust_region::{System, dynare_solve};

/// `options_.ep` (and the `options_` fields the extended path reads), Dynare's defaults.
#[derive(Clone, Debug)]
pub struct EpOptions {
    /// `options_.ep.periods` (`solver_periods`, default 200).
    pub periods: usize,
    /// `options_.ep.maxit` (500).
    pub maxit: usize,
    /// `options_.ep.stochastic.order` (0).
    pub order: usize,
    /// `options_.ep.stochastic.algo` (0 = `tree=perfect`, 1 = `tree=sparse`).
    pub tree: TreeKind,
    pub integration: Integration,
    /// `options_.ep.stochastic.quadrature.nodes` (5).
    pub quadrature_nodes: usize,
    /// `options_.ep.stochastic.hybrid_order` (0; `hybrid` = 2).
    pub hybrid_order: usize,
    /// `options_.ep.use_first_order_solution_as_initial_guess`.
    pub use_first_order_solution: bool,
    /// `options_.dynatol.f` / `.x` (1e-5).
    pub tolf: f64,
    pub tolx: f64,
    /// `options_.trust_region_initial_step_bound_factor` (1).
    pub trust_region_factor: f64,
    /// `options_.no_homotopy` (false; forced to true at order > 0).
    pub no_homotopy: bool,
    /// `options_.lmmcp.status` (the `lmmcp` option: `lmmcp.m` at order 0).
    pub lmmcp: bool,
    /// `options_.ep.solve_algo` (9; 10 = `lmmcp.m` on the stacked problem or the stochastic tree,
    /// `solve_stochastic_perfect_foresight_model_{0,1}.m`). Dynare's preprocessor has no option for
    /// it (it is set by `options_.ep.solve_algo = 10` in a `verbatim` block); here the
    /// `extended_path` option `solve_algo=10` does it.
    pub solve_algo: usize,
}

impl Default for EpOptions {
    fn default() -> Self {
        EpOptions {
            periods: 200,
            maxit: 500,
            order: 0,
            tree: TreeKind::Full,
            integration: Integration::Tensor,
            quadrature_nodes: 5,
            hybrid_order: 0,
            use_first_order_solution: false,
            tolf: 1e-5,
            tolx: 1e-5,
            trust_region_factor: 1.0,
            no_homotopy: false,
            lmmcp: false,
            solve_algo: 9,
        }
    }
}

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

fn word(o: &Options, k: &str) -> Option<String> {
    match opt(o, k) {
        Some(OptValue::Str(s)) => Some(s.to_lowercase()),
        _ => None,
    }
}

impl EpOptions {
    /// Applies the options of one `extended_path` command (the preprocessor's mapping to
    /// `options_.ep`). Returns the `periods` option (the sample size), if given.
    pub fn update(&mut self, o: &Options) -> Result<Option<usize>> {
        for (k, v) in o {
            match k.as_str() {
                "periods" => {}
                "solver_periods" => {
                    self.periods =
                        num(o, k).ok_or_else(|| anyhow!("extended_path: solver_periods must be an integer"))? as usize
                }
                "order" => {
                    self.order = num(o, k).ok_or_else(|| anyhow!("extended_path: order must be an integer"))? as usize
                }
                "hybrid" => {
                    self.hybrid_order = match v {
                        OptValue::Num(x) => *x as usize,
                        _ => 2,
                    }
                }
                "lmmcp" => self.lmmcp = true,
                "solve_algo" => {
                    self.solve_algo =
                        num(o, k).map(|v| v as usize).filter(|v| *v == 9 || *v == 10).ok_or_else(|| {
                            anyhow!("extended_path: solve_algo must be 9 or 10 (11, PATH, is not available)")
                        })?
                }
                "use_first_order_solution" => self.use_first_order_solution = true,
                "integration" => {
                    self.integration = match word(o, k).as_deref() {
                        Some("tensor_gaussian_quadrature") => Integration::Tensor,
                        Some("stroud") => Integration::Stroud5,
                        Some("unscented") => Integration::Unscented,
                        _ => bail!(
                            "extended_path: unknown integration method (tensor_gaussian_quadrature, stroud, unscented)"
                        ),
                    }
                }
                "tree" => {
                    self.tree = match word(o, k).as_deref() {
                        Some("perfect") => TreeKind::Full,
                        Some("sparse") => TreeKind::Sparse,
                        _ => bail!("extended_path: unknown tree (perfect, sparse)"),
                    }
                }
                "number_of_quadrature_nodes" => {
                    self.quadrature_nodes = num(o, k)
                        .ok_or_else(|| anyhow!("extended_path: number_of_quadrature_nodes must be an integer"))?
                        as usize
                }
                _ => println!("WARNING: extended_path: option `{k}` is not supported by dynare-rust and is ignored."),
            }
        }
        Ok(num(o, "periods").map(|v| v as usize))
    }
}

/// Inputs of an extended-path simulation (Dynare's `M_`, `oo_` fields).
pub struct EpModel<'a> {
    pub model: &'a Model,
    pub params: &'a [f64],
    pub steady_state: &'a [f64],
    pub exo_steady_state: &'a [f64],
    pub sigma_e: &'a Mat,
}

/// Result of [`extended_path`].
#[derive(Clone, Debug)]
pub struct EpOutput {
    /// `oo_.endo_simul`: n_endo x (samplesize + 1), column 0 = initial condition.
    pub endo_simul: Mat,
    /// `oo_.exo_simul`: (samplesize + 1) x n_exo, row 0 = exogenous steady state.
    pub exo_simul: Mat,
    /// `oo_.extended_path.status`.
    pub status: bool,
    /// Trust-region iterations per period.
    pub iterations: Vec<usize>,
}

/// Port of `extended_path_shocks.m` (`innovation_distribution = 'gaussian'`):
/// `randn(effective_number_of_shocks, T)` (filled period by period) times `Omega`, the upper
/// Cholesky factor of the covariance of the shocks with positive variance, from ChaCha8.
pub fn draw_gaussian_shocks(sigma_e: &Mat, samplesize: usize, seed: u64) -> Result<Mat> {
    let nx = sigma_e.nrows();
    let pos: Vec<usize> = (0..nx).filter(|&i| sigma_e[(i, i)] > 0.0).collect();
    let mut shocks = Mat::zeros(samplesize, nx);
    if pos.is_empty() {
        return Ok(shocks);
    }
    let sig = Mat::from_fn(pos.len(), pos.len(), |i, j| sigma_e[(pos[i], pos[j])]);
    let omega = chol_upper(&sig)
        .ok_or_else(|| anyhow!("extended_path: the covariance matrix of the shocks is not positive definite"))?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(0xE9);
    let z = Mat::from_fn(samplesize, pos.len(), |_, _| 0.0);
    let mut z = z;
    for t in 0..samplesize {
        for k in 0..pos.len() {
            z[(t, k)] = StandardNormal.sample(&mut rng);
        }
    }
    let e = &z * &omega; // (Omega' * Z')'
    for (c, &i) in pos.iter().enumerate() {
        shocks.set_column(i, &e.column(c));
    }
    Ok(shocks)
}

/// The per-period problem and its solver state (Dynare's `pfm`).
struct Solver<'a> {
    prob: TreeProblem<'a>,
    jac: SparseJac,
    plan: LuPlan,
    ny: usize,
    periods: usize,
    kind: TreeKind,
    opts: EpOptions,
    /// `lmmcp` (complementarity conditions of the model)
    mcp: Option<crate::deterministic::mcp::Complementarity>,
}

impl<'a> Solver<'a> {
    fn new(
        m: &EpModel<'a>,
        o: &EpOptions,
        order: usize,
        nodes: Mat,
        weights: Vec<f64>,
        pos: Vec<usize>,
        h: Option<Vec<f64>>,
    ) -> Result<Self> {
        let model = m.model;
        let ny = model.n_endo();
        let periods = o.periods;
        let kind = if order == 0 { TreeKind::Full } else { o.tree };
        let nnodes = weights.len().max(1);
        let blocks = tree::build_tree(kind, order, nnodes, periods, &weights, h.is_some())?;
        let exo = Mat::from_fn(periods + 2, model.n_exo(), |_, k| m.exo_steady_state[k]);
        // stream PP: STEADY_STATE() reads oo_.steady_state
        let prob = TreeProblem::new(
            model,
            model.params_with_steady_state(m.params, m.steady_state),
            blocks,
            exo,
            pos,
            nodes,
            h,
        );
        let jac = prob.new_jacobian();
        let plan = LuPlan::new(&jac, ny, &StackedStructure::new(model));
        // ---- extended_path(lmmcp)
        // (order 0: perfect_foresight_solver_core with options_.lmmcp.status; the stochastic
        // extended path only uses lmmcp with ep.solve_algo = 10, which dynare-rust does not set)
        let mcp = if (o.lmmcp && order == 0) || o.solve_algo == 10 {
            Some(crate::deterministic::mcp::Complementarity::new(model, m.params)?)
        } else {
            None
        };
        Ok(Solver { prob, jac, plan, ny, periods, kind, opts: o.clone(), mcp })
    }

    fn solve(&mut self, x: &mut [f64]) -> trust_region::SolveStatus {
        let o = &self.opts;
        // ---- solve_algo = 10 (lmmcp)
        if let Some(cc) = &self.mcp {
            let (errorflag, iterations) = crate::deterministic::pf_q1::ep_lmmcp(&mut self.prob, &mut self.jac, x, cc);
            return trust_region::SolveStatus { errorflag, errorcode: if errorflag { 0 } else { 1 }, iterations };
        }
        dynare_solve(&mut self.prob, &mut self.jac, &mut self.plan, x, o.maxit, o.tolf, o.tolx, o.trust_region_factor)
    }

    /// `perfect_foresight_solver_core` (stacked problem, `stack_solve_algo = 7`,
    /// `solve_algo = 9`: `solve_stacked_problem.m`) on `endo` (n x (periods+2), solved in
    /// place) and `exo` ((periods+2) x n_exo). Returns (success, iterations).
    fn pf_core(&mut self, endo: &mut Mat, exo: &Mat) -> (bool, usize) {
        let (ny, periods) = (self.ny, self.periods);
        self.prob.init = endo.column(0).iter().copied().collect();
        self.prob.terminal = endo.column(periods + 1).iter().copied().collect();
        self.prob.exo.copy_from(exo);
        let mut x: Vec<f64> = (1..=periods).flat_map(|c| endo.column(c).iter().copied().collect::<Vec<_>>()).collect();
        let st = self.solve(&mut x);
        for c in 0..periods {
            for v in 0..ny {
                endo[(v, c + 1)] = x[c * ny + v];
            }
        }
        (!st.errorflag, st.iterations)
    }

    /// `solve_stochastic_perfect_foresight_model_{0,1}.m`: solves the tree from `endo`'s
    /// initial (and, full tree, terminal) condition with the realized shock in row 1 of
    /// `exo`, starting from `y`; sets column 1 of `endo`. Returns (success, iterations).
    fn sep(&mut self, endo: &mut Mat, exo: &Mat, y: &mut [f64], steady_state: &[f64]) -> (bool, usize) {
        let ny = self.ny;
        self.prob.init = endo.column(0).iter().copied().collect();
        self.prob.terminal = match self.kind {
            TreeKind::Full => endo.column(endo.ncols() - 1).iter().copied().collect(),
            TreeKind::Sparse => steady_state.to_vec(),
        };
        self.prob.exo.copy_from(exo);
        let st = self.solve(y);
        for v in 0..ny {
            endo[(v, 1)] = y[v];
        }
        (!st.errorflag, st.iterations)
    }

    /// Port of `extended_path_homotopy.m` (method 2, order 0): returns (convergence, endo).
    fn homotopy(&mut self, endo_simul: &Mat, exo_simul: &Mat, ss: &[f64], xs: &[f64]) -> (bool, Mat) {
        let blend = |w: f64| {
            Mat::from_fn(exo_simul.nrows(), exo_simul.ncols(), |r, k| w * exo_simul[(r, k)] + (1.0 - w) * xs[k])
        };
        let endo0 = endo_simul.clone();
        let mut endo = endo_simul.clone();
        let mut noconvergence = true;
        let mut iteration = 0;
        let mut weight: f64 = 0.1;
        let maxiter = 100;
        let mut increase_flag = false;
        let mut increase_factor = 1.2;
        let decrease_factor = 1.1;
        let mut state = [false; 5];
        let mut oldweight = weight;
        while noconvergence {
            iteration += 1;
            let mut e = endo.clone();
            for v in 0..self.ny {
                e[(v, 0)] = ss[v] + weight * (endo0[(v, 0)] - ss[v]);
            }
            let (flag, _) = self.pf_core(&mut e, &blend(weight));
            state.copy_within(0..4, 1);
            state[0] = flag;
            if flag {
                if weight == 1.0 {
                    // (Dynare returns the previous weight's paths here)
                    noconvergence = false;
                    break;
                }
                if state.iter().all(|&s| s) {
                    increase_factor = 1.0 + (increase_factor - 1.0) * 1.1;
                    state = [false; 5];
                }
                oldweight = weight;
                weight = (weight * increase_factor).min(1.0);
                increase_flag = true;
                endo = e;
            } else if increase_flag {
                weight = oldweight + (weight - oldweight) / 100.0;
            } else {
                weight = (weight / decrease_factor).min(1.0);
            }
            if iteration > maxiter || weight < 1e-9 {
                break;
            }
        }
        if noconvergence {
            endo = endo0;
            let mut index = 0usize;
            while noconvergence {
                // MATLAB's 0:(1/1000):1
                let weight = if index <= 500 { index as f64 * 0.001 } else { 1.0 - (1000 - index) as f64 * 0.001 };
                let mut e = endo.clone();
                let (flag, _) = self.pf_core(&mut e, &blend(weight));
                if flag {
                    if weight == 1.0 {
                        noconvergence = false;
                        continue;
                    }
                    index += 1;
                    endo = e;
                } else {
                    break;
                }
            }
        }
        (!noconvergence, endo)
    }
}

/// Port of `extended_path.m` (with `extended_path_initialization.m` and
/// `extended_path_core.m`): simulates `shocks.nrows()` periods from `init`.
pub fn extended_path(m: &EpModel, opts: &EpOptions, init: &[f64], shocks: &Mat) -> Result<EpOutput> {
    let model = m.model;
    let (ny, nx) = (model.n_endo(), model.n_exo());
    let lli = &model.lead_lag_incidence;
    let maximum_lag = lli.iter().any(|r| r[0] > 0);
    let maximum_lead = lli.iter().any(|r| r[2] > 0);
    let mut o = opts.clone();
    if !maximum_lag {
        bail!("(Stochastic) Extended path is not supported for purely forward-looking models.");
    }
    if !maximum_lead {
        if o.order > 0 {
            bail!("Stochastic extended path does not support purely backward-looking models.");
        }
        o.periods = 1;
    }
    if let Some(j) = (0..ny).find(|&j| lli[j][1] == 0) {
        bail!("extended_path: variable {} does not appear at the current period", model.endo_names[j]);
    }
    if shocks.ncols() != nx {
        bail!("extended_path: the shocks must have {nx} columns");
    }
    let order = o.order;
    // setup_stochastic_perfect_foresight_model_solver
    let pos: Vec<usize> = (0..nx).filter(|&i| m.sigma_e[(i, i)] > 0.0).collect();
    let sig = Mat::from_fn(pos.len(), pos.len(), |i, j| m.sigma_e[(pos[i], pos[j])]);
    let omega = if pos.is_empty() {
        Mat::zeros(0, 0)
    } else {
        chol_upper(&sig).ok_or_else(|| anyhow!("extended_path: Sigma_e is not positive definite"))?
    };
    if order > 0 {
        o.no_homotopy = true;
    }
    let (nodes, weights) = if order > 0 {
        if pos.is_empty() {
            bail!("Stochastic extended path: no shock has a positive variance");
        }
        let (nodes, weights) = quadrature::setup_integration_nodes(o.integration, o.quadrature_nodes, &omega)?;
        // solve_stochastic_perfect_foresight_model_{0,1}: the zero node first
        let Some(k) = (0..nodes.nrows()).find(|&r| nodes.row(r).iter().map(|v| v.abs()).sum::<f64>() < 1e-12) else {
            bail!("solve_stochastic_perfect_foresight_model: there is no node equal to zero");
        };
        let mut idx = vec![k];
        idx.extend((0..nodes.nrows()).filter(|&r| r != k));
        (
            Mat::from_fn(nodes.nrows(), nodes.ncols(), |r, c| nodes[(idx[r], c)]),
            idx.iter().map(|&r| weights[r]).collect(),
        )
    } else {
        (Mat::zeros(0, pos.len()), Vec::new())
    };
    // hybrid correction (constant of the second-order reduced form)
    if o.hybrid_order == 1 {
        println!(
            "Warning: extended_path:: hybrid=1 is equivalent to hybrid=0 (option value must be an integer greater than 1 to be effective)."
        );
    }
    let h = if order > 0 && o.hybrid_order > 1 {
        // extended_path_initialization: resol at order = hybrid_order; the correction is
        // 0.5*ghs2 at order 2, the constant g_0 of the k-order rule above
        let drh = crate::solver::korder::solve_model(
            model,
            m.steady_state,
            m.params,
            m.sigma_e,
            o.hybrid_order,
            Default::default(),
        )?;
        let iov = &drh.first.inv_order_var;
        let g0 = if o.hybrid_order == 2 { drh.ghs2.clone() * 0.5 } else { drh.g_k(0) };
        Some((0..ny).map(|v| g0[(iov[v], 0)]).collect::<Vec<f64>>())
    } else if order > 0 && o.hybrid_order == 1 {
        Some(vec![0.0; ny])
    } else {
        None
    };
    let dr1 = if o.use_first_order_solution {
        let g1 = model.dynamic_jacobian(m.steady_state, m.exo_steady_state, m.params);
        Some(
            crate::solver::first_order(&g1, lli, nx, m.steady_state, &Default::default())
                .map_err(|e| anyhow!("extended_path: first-order solution: {e}"))?,
        )
    } else {
        None
    };
    let mut solver = Solver::new(m, &o, order, nodes, weights, pos, h)?;
    let periods = o.periods;
    let ss = m.steady_state;
    let samplesize = shocks.nrows();
    let mut paths = Mat::from_element(ny, samplesize + 1, f64::NAN);
    for v in 0..ny {
        paths[(v, 0)] = init[v];
    }
    let exo_simul =
        Mat::from_fn(samplesize + 1, nx, |r, k| if r == 0 { m.exo_steady_state[k] } else { shocks[(r - 1, k)] });
    let mut spfm = Mat::from_fn(periods + 2, nx, |_, k| m.exo_steady_state[k]);
    let mut prev: Option<Mat> = None;
    let mut y: Option<Vec<f64>> = None;
    let mut status = true;
    let mut iterations = Vec::with_capacity(samplesize);
    for t in 1..=samplesize {
        for k in 0..nx {
            spfm[(1, k)] = shocks[(t - 1, k)];
        }
        // extended_path_core
        let mut endo = if let Some(dr) = &dr1 {
            let ex = Mat::from_fn(periods + 1, nx, |r, k| spfm[(r + 1, k)]);
            let y0: Vec<f64> = paths.column(t - 1).iter().copied().collect();
            crate::analysis::simulate::simult(dr, &y0, &ex)
        } else if let Some(p) = &prev {
            Mat::from_fn(ny, periods + 2, |v, c| if c + 1 < periods + 2 { p[(v, c + 1)] } else { ss[v] })
        } else {
            Mat::from_fn(ny, periods + 2, |v, c| if c == 0 { paths[(v, t - 1)] } else { ss[v] })
        };
        if !maximum_lead && !o.lmmcp {
            // sim1_purely_backward: the guess is the previous period's value
            for v in 0..ny {
                endo[(v, 1)] = endo[(v, 0)];
            }
        }
        let (mut ok, it) = if order == 0 {
            solver.pf_core(&mut endo, &spfm)
        } else {
            let mut yv = y.take().unwrap_or_else(|| (0..solver.prob.dim()).map(|q| ss[q % ny]).collect());
            let r = solver.sep(&mut endo, &spfm, &mut yv, ss);
            y = Some(yv);
            r
        };
        iterations.push(it);
        if !ok && !o.no_homotopy {
            let (c, e) = solver.homotopy(&endo, &spfm, ss, m.exo_steady_state);
            ok = c;
            endo = e;
        }
        if !ok {
            println!(
                "Warning: extended_path: No convergence of the (stochastic) perfect foresight solver (in period {})!",
                t + 1
            );
            status = false;
            break;
        }
        for v in 0..ny {
            paths[(v, t)] = endo[(v, 1)];
        }
        prev = Some(endo);
    }
    Ok(EpOutput { endo_simul: paths, exo_simul, status, iterations })
}

/// Commands handled by this module (the driver's `Command::Other` names).
pub fn handles(name: &str) -> bool {
    matches!(
        name,
        "extended_path"
            | "perfect_foresight_with_expectation_errors_setup"
            | "perfect_foresight_with_expectation_errors_solver"
            | "initval_file"
            | "histval_file"
    )
}

/// Runs command `ci` of `mf` (one of [`handles`]).
pub fn command(mf: &ModFile, det: &mut DetState, ci: usize, dir: &Path, fname: &str, seed: u64) -> Result<()> {
    let Command::Other(name, o) = &mf.commands[ci] else { bail!("ep::command: not an EP command") };
    match name.as_str() {
        "extended_path" => run(mf, det, ci, dir, fname, seed),
        "perfect_foresight_with_expectation_errors_setup" => pfwee::setup(mf, det, o, dir),
        "perfect_foresight_with_expectation_errors_solver" => pfwee::solver(mf, det, o, dir, fname),
        "initval_file" => datafiles::initval_file(mf, det, o, dir),
        "histval_file" => datafiles::histval_file(mf, det, o, dir),
        _ => bail!("ep::command: unknown command {name}"),
    }
}

/// The `extended_path` command (`[Simulated_time_series, oo_] = extended_path([], periods, [],
/// options_, M_, oo_)`): draws the Gaussian shocks (dynare-rust's ChaCha8 stream, `--seed`),
/// simulates from `M_.endo_histval` (or the steady state), stores `oo_.endo_simul` /
/// `oo_.exo_simul` in `det` and writes them to `<fname>/Output/<fname>_{endo,exo}_simul.csv`.
/// `ci` is the command's index (options of earlier `extended_path` commands persist).
pub fn run(mf: &ModFile, det: &mut DetState, ci: usize, dir: &Path, fname: &str, seed: u64) -> Result<()> {
    let model = &mf.model;
    let mut o = EpOptions::default();
    let mut samplesize = None;
    for (k, c) in mf.commands[..=ci].iter().enumerate() {
        if let Command::Other(n, opts) = c
            && n == "extended_path"
        {
            if k == ci {
                samplesize = o.update(opts)?;
            } else {
                let quiet: Options = opts.iter().filter(|(k, _)| k != "periods").cloned().collect();
                let _ = o.update(&quiet);
            }
        }
    }
    let samplesize =
        samplesize.ok_or_else(|| anyhow!("ERROR: the 'periods' option of 'extended_path' is mandatory"))?;
    let init: Vec<f64> = if let Some((eh, _)) = &det.histval_file {
        eh.column(eh.ncols() - 1).iter().copied().collect()
    } else if mf.histval.is_empty() {
        det.steady_state.clone()
    } else {
        let (eh, _) = det.histval(mf);
        eh.column(eh.ncols() - 1).iter().copied().collect()
    };
    let shocks = draw_gaussian_shocks(&mf.sigma_e, samplesize, seed)?;
    let m = EpModel {
        model,
        params: &det.params,
        steady_state: &det.steady_state,
        exo_steady_state: &det.exo_steady_state,
        sigma_e: &mf.sigma_e,
    };
    let t0 = Instant::now();
    let out = extended_path(&m, &o, &init, &shocks)?;
    let el = t0.elapsed().as_secs_f64();
    let what = if o.order == 0 {
        "Extended path".to_string()
    } else {
        format!(
            "Stochastic extended path (order {}, {} tree, {}{})",
            o.order,
            if o.tree == TreeKind::Full { "full" } else { "sparse" },
            o.integration.dynare_name(),
            if o.integration == Integration::Tensor {
                format!(" with {} nodes", o.quadrature_nodes)
            } else {
                String::new()
            }
        )
    };
    println!(
        "{what}: {samplesize} periods, {} solver periods, {:.3} s ({} trust-region iterations).",
        o.periods,
        el,
        out.iterations.iter().sum::<usize>()
    );
    if !out.status {
        println!("oo_.extended_path.status = false");
    }
    det.endo_simul = Some(out.endo_simul);
    det.exo_simul = Some(out.exo_simul);
    det.write_results(model, dir, fname)
}
