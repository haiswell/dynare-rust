// SPDX-License-Identifier: GPL-3.0-or-later
//! OccBin: occasionally binding constraints solved as a piecewise-linear model
//! (Guerrieri & Iacoviello 2015), a port of Dynare 7.2's `matlab/+occbin/`.
//!
//! The pieces, as in Dynare:
//! - preprocessor: `[name='..', bind='c', relax='c']` equation tags are merged into one equation
//!   per name, switched by the parameters `occbin_<c>_bind` (`ModelBuilder::add_occbin_equation`);
//!   the `occbin_constraints` block ([`OccbinConstraint`], compiled in [`constraints`]) and
//!   `shocks(surprise)` blocks ([`OccbinCommand::SurpriseShocks`]) are parsed by `modfile`;
//! - `occbin_setup` (`+occbin/setup.m`), `occbin_solver` (`+occbin/solver.m` ->
//!   [`solver::solve_one_constraint`] / [`solver::solve_two_constraints`]),
//!   `occbin_write_regimes` (`write_regimes_to_xls.m`, written as CSV) and `occbin_graph`
//!   (`+occbin/graph.m`), run by a [`Session`] that keeps `options_.occbin` and the solver's
//!   `persistent` state between commands.
//!
//! - estimation ([`estimation`]): the piecewise Kalman filter ([`pkf`]) and inversion filter
//!   ([`ivf`]) likelihoods, the OccBin smoother ([`smoother`]); GIRFs and forecasts ([`irf`]).
//!
//! Not ported yet: `STEADY_STATE()` inside model equations.

pub mod constraints;
pub mod estimation;
pub mod irf;
pub mod ivf;
pub mod pkf;
pub mod regimes;
pub mod smoother;
pub mod solver;

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;

use crate::graphics::{self, Color, Figure, GraphFormat, GraphOptions, LineStyle, Panel, Series};
use crate::linalg::Mat;
use crate::model::Model;
use crate::modfile::{Ast, Command, ModFile, OptValue, Options};
use crate::solver::{FirstOrderSolver, SolverOptions, SteadyOptions, steady_state};
pub use constraints::Constraints;
pub use solver::{Dm, OccbinError, Regime, RegimeHistory, SimulOptions, SolveOutput, StateSpace};

/// One constraint of the `occbin_constraints` block (expressions unresolved).
#[derive(Clone, Debug, PartialEq)]
pub struct OccbinConstraint {
    pub name: String,
    pub bind: Option<Ast>,
    pub relax: Option<Ast>,
    pub error_bind: Option<Ast>,
    pub error_relax: Option<Ast>,
}

impl OccbinConstraint {
    pub fn new(name: String) -> Self {
        OccbinConstraint { name, bind: None, relax: None, error_bind: None, error_relax: None }
    }
}

/// One `var/periods/values` group of a `shocks(surprise)` block (`M_.surprise_shocks` entry).
#[derive(Clone, Debug, PartialEq)]
pub struct SurpriseShock {
    /// 0-based exogenous index.
    pub exo: usize,
    /// 1-based periods.
    pub periods: Vec<usize>,
    pub value: f64,
}

/// OccBin statements, in file order among the other commands.
#[derive(Clone, Debug, PartialEq)]
pub enum OccbinCommand {
    /// `shocks(surprise[,overwrite])`.
    SurpriseShocks {
        overwrite: bool,
        shocks: Vec<SurpriseShock>,
    },
    Setup(Options),
    Solver(Options),
    WriteRegimes(Options),
    Graph(Options, Vec<String>),
}

const COMMANDS: &[&str] = &["occbin_setup", "occbin_solver", "occbin_write_regimes", "occbin_graph"];

pub fn is_occbin_command(name: &str) -> bool {
    COMMANDS.contains(&name)
}

impl OccbinCommand {
    pub fn new(name: &str, opts: Options, vars: Vec<String>) -> Self {
        match name {
            "occbin_setup" => OccbinCommand::Setup(opts),
            "occbin_solver" => OccbinCommand::Solver(opts),
            "occbin_write_regimes" => OccbinCommand::WriteRegimes(opts),
            _ => OccbinCommand::Graph(opts, vars),
        }
    }
}

/// `M_.occbin.pswitch`: the parameter `occbin_<name>_bind` of each constraint (block order).
pub fn pswitch(m: &Model, decl: &[OccbinConstraint]) -> Result<Vec<usize>> {
    decl.iter()
        .map(|c| {
            let p = format!("occbin_{}_bind", c.name);
            m.param_index(&p)
                .ok_or_else(|| anyhow!("occbin constraint '{}' is not used by any bind/relax equation tag", c.name))
        })
        .collect()
}

/// `oo_.occbin.simul`: paths in levels (steady state added, as `occbin.solver` returns them).
#[derive(Clone, Debug)]
pub struct SimulResult {
    pub piecewise: Mat,
    pub linear: Option<Mat>,
    pub regime_history: RegimeHistory,
    pub shocks_sequence: Mat,
    /// 0-based exogenous indices of the columns of `shocks_sequence`.
    pub exo_pos: Vec<usize>,
    pub ys: Vec<f64>,
    /// `SS_out`: the state space of each shock period.
    pub ss: Vec<StateSpace>,
}

/// The decision rule and the regimes' derivatives, kept while the parameters do not change
/// (`occbin.solver`'s `persistent sto_M sto_dr`, the solvers' `persistent DM` and the
/// `mkdatap_*` dictionaries).
struct Cache {
    params: Vec<f64>,
    ys: Vec<f64>,
    dm: Dm,
    dict: Option<solver::Dictionary>,
}

/// `options_.occbin.write_regimes`.
#[derive(Clone, Debug)]
pub struct WriteRegimesOptions {
    pub periods: Option<Vec<usize>>,
    /// "simul" or "smoother".
    pub kind: String,
    pub filename: String,
}

/// OccBin state of a .mod run: `options_.occbin`, `M_.surprise_shocks`, `oo_.occbin`.
pub struct Session {
    pub simul: SimulOptions,
    /// `options_.occbin.likelihood` / `.smoother` / `.filter` (OccBin estimation, see
    /// [`estimation`]).
    pub likelihood: estimation::LikelihoodOptions,
    pub smoother: estimation::SmootherOptions,
    pub filter: estimation::FilterOptions,
    pub write_regimes: WriteRegimesOptions,
    surprise: Vec<SurpriseShock>,
    cache: Option<Cache>,
    pub last: Option<SimulResult>,
    /// `oo_.occbin.smoother.regime_history` of the last estimation's smoother.
    pub smoother_regimes: Option<Vec<regimes::Regimes>>,
}

pub(crate) fn num(v: &OptValue) -> Option<f64> {
    match v {
        OptValue::Num(x) => Some(*x),
        OptValue::Vec(x) if x.len() == 1 => Some(x[0]),
        OptValue::Str(s) => s.parse().ok(),
        _ => None,
    }
}

pub(crate) fn boolean(v: &OptValue) -> bool {
    match v {
        OptValue::Flag => true,
        OptValue::Str(s) => s == "true",
        other => num(other).is_some_and(|x| x != 0.0),
    }
}

pub(crate) fn count(v: &OptValue, k: &str) -> Result<usize> {
    let x = num(v).ok_or_else(|| anyhow!("option {k}: expected an integer"))?;
    if x < 0.0 || x.fract() != 0.0 {
        bail!("option {k}: expected a non-negative integer");
    }
    Ok(x as usize)
}

impl SimulOptions {
    /// Sets `options_.occbin.simul.<key>` from a `simul_<key>` option; false if `key` is unknown.
    fn set(&mut self, key: &str, v: &OptValue) -> Result<bool> {
        match key {
            "periods" => self.periods = count(v, key)?,
            "maxit" => self.maxit = count(v, key)?,
            "check_ahead_periods" => self.check_ahead_periods = count(v, key)?,
            "max_check_ahead_periods" => {
                self.max_check_ahead_periods = match num(v) {
                    Some(x) if x.is_infinite() => None,
                    _ => Some(count(v, key)?),
                }
            }
            "curb_retrench" => self.curb_retrench = boolean(v),
            "debug" => self.debug = boolean(v),
            "periodic_solution" => self.periodic_solution = boolean(v),
            "periodic_solution_threshold" => {
                self.periodic_solution_threshold = num(v).ok_or_else(|| anyhow!("option {key}: expected a number"))?
            }
            "periodic_solution_strict" => self.periodic_solution_strict = boolean(v),
            "reset_check_ahead_periods" | "reset_check_ahead_periods_in_new_period" => {
                self.reset_check_ahead_periods_in_new_period = boolean(v)
            }
            "reset_regime_in_new_period" => self.reset_regime_in_new_period = boolean(v),
            "piecewise_only" => self.piecewise_only = boolean(v),
            _ => return Ok(false),
        }
        Ok(true)
    }
}

impl Session {
    pub fn new(mf: &ModFile, fname: &str) -> Session {
        Session {
            simul: SimulOptions::new(mf.model.n_exo()),
            likelihood: estimation::LikelihoodOptions::default(),
            smoother: estimation::SmootherOptions::default(),
            filter: estimation::FilterOptions::default(),
            write_regimes: WriteRegimesOptions {
                periods: None,
                kind: "simul".into(),
                filename: format!("{fname}_occbin_regimes"),
            },
            surprise: Vec::new(),
            cache: None,
            last: None,
            smoother_regimes: None,
        }
    }

    /// Runs one OccBin command. `out_dir` is `<dir>/<fname>/Output`, `graph_dir` the folder
    /// for figures.
    pub fn command(
        &mut self,
        mf: &ModFile,
        cmd: &OccbinCommand,
        out_dir: &Path,
        fname: &str,
        graph_dir: &Path,
    ) -> Result<()> {
        match cmd {
            OccbinCommand::SurpriseShocks { overwrite, shocks } => {
                if *overwrite {
                    self.surprise.clear();
                }
                self.surprise.extend(shocks.iter().cloned());
            }
            OccbinCommand::Setup(opts) => self.setup(mf, opts)?,
            OccbinCommand::Solver(opts) => {
                let r = self.run_solver(mf, opts)?;
                print_summary(mf, &r);
                write_results(mf, &r, out_dir, fname)?;
                self.last = Some(r);
            }
            OccbinCommand::WriteRegimes(opts) => {
                for (k, v) in opts {
                    match k.as_str() {
                        "periods" => {
                            self.write_regimes.periods = Some(match v {
                                OptValue::Vec(x) => x.iter().map(|&p| p as usize).collect(),
                                other => (1..=count(other, k)?).collect(),
                            })
                        }
                        "filename" => {
                            if let OptValue::Str(s) = v {
                                self.write_regimes.filename = s.clone();
                            }
                        }
                        "simul" => self.write_regimes.kind = "simul".into(),
                        "smoother" => self.write_regimes.kind = "smoother".into(),
                        _ => println!("WARNING: occbin_write_regimes: option `{k}` is not supported; ignored."),
                    }
                }
                let rh = if self.write_regimes.kind == "smoother" {
                    self.smoother_regimes.as_ref().ok_or_else(|| anyhow!("write_regimes_to_xls: the required field does not exist (run an estimation with the smoother first)"))?
                } else {
                    &self
                        .last
                        .as_ref()
                        .ok_or_else(|| {
                            anyhow!("write_regimes_to_xls: the required field does not exist (run occbin_solver first)")
                        })?
                        .regime_history
                };
                let p = write_regimes_csv(rh, &self.write_regimes, out_dir)?;
                println!("OccBin regimes written to {}", p.display());
            }
            OccbinCommand::Graph(opts, vars) => {
                let r = self.last.as_ref().ok_or_else(|| anyhow!("occbin_graph: run occbin_solver first"))?;
                let steady_state = !opts.iter().any(|(k, _)| k == "noconstant");
                let figs = graph(mf, r, vars, steady_state)?;
                let go = GraphOptions { enabled: true, formats: vec![GraphFormat::Pdf], dir: graph_dir.to_path_buf() };
                for f in &figs {
                    graphics::save(f, fname, &go)?;
                }
            }
        }
        Ok(())
    }

    /// `occbin_solver(options)`: sets `options_.occbin.simul.*` (they persist), then solves.
    pub fn run_solver(&mut self, mf: &ModFile, opts: &Options) -> Result<SimulResult> {
        for (k, v) in opts {
            let key = k.strip_prefix("simul_").unwrap_or(k);
            if !self.simul.set(key, v)? {
                println!("WARNING: occbin_solver: option `{k}` is not supported; ignored.");
            }
        }
        self.solve(mf).context("occbin_solver")
    }

    /// Port of `+occbin/setup.m`: only the options it forwards (`occbin.set_option`) reach
    /// `options_.occbin`; the preprocessor accepts more, which Dynare 7.2 then ignores.
    pub fn setup(&mut self, mf: &ModFile, opts: &Options) -> Result<()> {
        const FORWARDED: &[&str] =
            &["periods", "curb_retrench", "maxit", "check_ahead_periods", "debug", "periodic_solution"];
        for (k, v) in opts {
            if let Some(key) = k.strip_prefix("simul_") {
                if FORWARDED.contains(&key) {
                    self.simul.set(key, v)?;
                } else {
                    println!(
                        "WARNING: occbin_setup: option `{k}` is not forwarded by Dynare 7.2's occbin.setup; ignored."
                    );
                }
                continue;
            }
            if !estimation::setup_option(self, k, v)? {
                println!("WARNING: occbin_setup: option `{k}` is not forwarded by Dynare 7.2's occbin.setup; ignored.");
            }
        }
        if !self.surprise.is_empty() {
            let n_exo = mf.model.n_exo();
            let tmax = self.surprise.iter().flat_map(|s| s.periods.iter().copied()).max().unwrap_or(0);
            let mut temp = Mat::zeros(tmax, n_exo);
            for s in &self.surprise {
                for &p in &s.periods {
                    temp[(p - 1, s.exo)] = s.value;
                }
            }
            let pos: Vec<usize> = (0..n_exo).filter(|&k| (0..tmax).any(|t| temp[(t, k)] != 0.0)).collect();
            let mut sh = Mat::zeros(tmax, pos.len());
            for (c, &k) in pos.iter().enumerate() {
                sh.set_column(c, &temp.column(k));
            }
            self.simul.shocks = sh;
            self.simul.exo_pos = pos;
        }
        Ok(())
    }

    /// Port of `+occbin/solver.m`.
    pub fn solve(&mut self, mf: &ModFile) -> Result<SimulResult> {
        let o = self.simul.clone();
        self.solve_with(mf, o)
    }

    /// `occbin.solver` with the solver options `o` (instead of `options_.occbin.simul`).
    pub fn solve_with(&mut self, mf: &ModFile, mut o: SimulOptions) -> Result<SimulResult> {
        let m = &mf.model;
        let nc = mf.occbin_constraints.len();
        if nc == 0 || nc > 2 {
            bail!("OccBin needs one or two constraints in an occbin_constraints block (found {nc})");
        }
        let cons = Constraints::compile(&mf.occbin_constraints, m)?;
        let psw = pswitch(m, &mf.occbin_constraints)?;
        let params = m.params.clone();
        if self.cache.as_ref().is_none_or(|c| !same_params(&c.params, &params)) {
            // resol(0, ...): steady state (from oo_.steady_state) and first-order solution
            let ys = steady(mf, &params)?;
            let x = vec![0.0; m.n_exo()];
            let g1 = m.dynamic_jacobian(&ys, &x, &params);
            let dr = FirstOrderSolver::new(&m.lead_lag_incidence, m.n_exo(), SolverOptions::default())
                .solve(&g1, &ys)
                .map_err(|e| anyhow!("{e}"))?;
            let dm = Dm::new(m, &ys, &params, &psw, &dr);
            self.cache = Some(Cache { params: params.clone(), ys, dm, dict: None });
        }
        if let Some(mx) = o.max_check_ahead_periods
            && o.check_ahead_periods > mx
        {
            println!("occbin::options::simul_check_ahead_periods cannot exceed simul_max_check_ahead_periods");
            println!(
                "occbin::options::simul_check_ahead_periods is re-set to be equal to simul_max_check_ahead_periods"
            );
            o.check_ahead_periods = mx;
        }
        let cache = self.cache.as_mut().unwrap();
        let pb = solver::Problem { dm: &cache.dm, constraints: &cons, params: &params, ys: &cache.ys };
        let out = if nc == 1 {
            solver::solve_one_constraint(&pb, &mut cache.dict, &o)
        } else {
            solver::solve_two_constraints(&pb, &mut cache.dict, &o)
        }
        .map_err(|e| anyhow!("{e} (shock period {})", e.period))?;
        let ys = cache.ys.clone();
        let add_ys = |mut z: Mat| {
            for t in 0..z.nrows() {
                for j in 0..z.ncols() {
                    z[(t, j)] += ys[j];
                }
            }
            z
        };
        Ok(SimulResult {
            piecewise: add_ys(out.piecewise),
            linear: out.linear.map(add_ys),
            regime_history: out.regime_history,
            shocks_sequence: o.shocks.clone(),
            exo_pos: o.exo_pos.clone(),
            ys: ys.clone(),
            ss: out.ss,
        })
    }
}

fn same_params(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

/// `oo_.steady_state` as `steady` leaves it (from initval), then `resol`'s
/// `evaluate_steady_state` from there.
pub(crate) fn steady(mf: &ModFile, params: &[f64]) -> Result<Vec<f64>> {
    let m = &mf.model;
    let n = m.n_endo();
    let x = vec![0.0; m.n_exo()];
    let mut o = SteadyOptions { linear: m.linear, ..Default::default() };
    for c in &mf.commands {
        if let Command::Steady(opts) = c {
            for (k, v) in opts {
                match (k.as_str(), num(v)) {
                    ("solve_algo", Some(a)) => o.solve_algo = a as u8,
                    ("maxit", Some(a)) => o.maxit = a as usize,
                    ("tolf", Some(a)) => o.tolf = a,
                    ("tolx", Some(a)) => o.tolx = a,
                    _ => {}
                }
            }
        }
    }
    let f = |y: &[f64]| {
        let mut r = vec![0.0; n];
        m.static_residuals(y, &x, params, &mut r);
        (r, m.static_jacobian(y, &x, params))
    };
    let mut g = vec![0.0; n];
    for &(i, v) in &mf.initval {
        g[i] = v;
    }
    m.set_auxiliary_variables(&mut g, &x, params);
    // stream PP: a steady_state_model gives the steady state (Dynare's examples/occbin/rbc_occbin.mod)
    if let Some(ssm) = crate::deterministic::SteadyStateModel::from_modfile(mf)? {
        let (ys, _) = crate::deterministic::evaluate_steady_state(m, Some(&ssm), &g, &x, params, &o, true)
            .map_err(|e| anyhow!("steady state: {e}"))?;
        return Ok(ys);
    }
    let has_steady = mf.commands.iter().any(|c| matches!(c, Command::Steady(_)));
    if has_steady {
        g = steady_state(&f, &g, &o).map_err(|e| anyhow!("steady: {e}"))?;
    }
    let mut ys = steady_state(&f, &g, &o).map_err(|e| anyhow!("steady state: {e}"))?;
    m.set_auxiliary_variables(&mut ys, &x, params);
    Ok(ys)
}

fn regime_string(v: &[bool]) -> String {
    v.iter().map(|&b| if b { "1" } else { "0" }).collect::<Vec<_>>().join("  ")
}

fn start_string(v: &[usize]) -> String {
    v.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("  ")
}

/// dynare-rust's one-line summary (Dynare prints nothing here): a progress note.
fn print_summary(mf: &ModFile, r: &SimulResult) {
    let names: Vec<&str> = mf.occbin_constraints.iter().map(|c| c.name.as_str()).collect();
    let binding: Vec<usize> = r
        .regime_history
        .iter()
        .enumerate()
        .filter(|(_, h)| h.iter().any(|g| g.regime.iter().any(|&b| b)))
        .map(|(t, _)| t + 1)
        .collect();
    crate::progress::note(format!(
        "\nOccBin: {} periods simulated ({} shock periods, constraint{} {}); a regime other than the baseline is expected after the shocks of {} period(s){}.",
        r.piecewise.nrows(),
        r.shocks_sequence.nrows(),
        if names.len() > 1 { "s" } else { "" },
        names.join(", "),
        binding.len(),
        if binding.is_empty() { String::new() } else { format!(": {}", compress(&binding)) }
    ));
}

/// `[1 2 3 7 8]` -> "1-3, 7-8".
fn compress(v: &[usize]) -> String {
    let mut out = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
            j += 1;
        }
        out.push(if i == j { v[i].to_string() } else { format!("{}-{}", v[i], v[j]) });
        i = j + 1;
    }
    out.join(", ")
}

fn mat_rows(m: &Mat) -> Vec<Vec<f64>> {
    (0..m.nrows()).map(|i| m.row(i).iter().copied().collect()).collect()
}

/// `<fname>_occbin_simul.json` (oo_.occbin.simul) and CSV files of the piecewise / linear paths.
fn write_results(mf: &ModFile, r: &SimulResult, out_dir: &Path, fname: &str) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let m = &mf.model;
    let regimes: Vec<serde_json::Value> = r
        .regime_history
        .iter()
        .map(|h| {
            if h.len() == 1 {
                json!({"regime": h[0].regime.iter().map(|&b| b as u8).collect::<Vec<_>>(), "regimestart": h[0].start})
            } else {
                json!({
                    "regime1": h[0].regime.iter().map(|&b| b as u8).collect::<Vec<_>>(), "regimestart1": h[0].start,
                    "regime2": h[1].regime.iter().map(|&b| b as u8).collect::<Vec<_>>(), "regimestart2": h[1].start,
                })
            }
        })
        .collect();
    let j = json!({
        "endo_names": m.endo_names,
        "exo_names": r.exo_pos.iter().map(|&k| m.exo_names[k].clone()).collect::<Vec<_>>(),
        "exo_pos": r.exo_pos.iter().map(|k| k + 1).collect::<Vec<_>>(),
        "ys": r.ys,
        "piecewise": mat_rows(&r.piecewise),
        "linear": r.linear.as_ref().map(mat_rows),
        "shocks_sequence": mat_rows(&r.shocks_sequence),
        "regime_history": regimes,
    });
    std::fs::write(out_dir.join(format!("{fname}_occbin_simul.json")), serde_json::to_string(&j)?)?;
    let write_csv = |name: &str, z: &Mat| -> Result<()> {
        let mut s = String::from("period,");
        s.push_str(&m.endo_names.join(","));
        s.push('\n');
        for t in 0..z.nrows() {
            s.push_str(&(t + 1).to_string());
            for j in 0..z.ncols() {
                s.push_str(&format!(",{:.17e}", z[(t, j)]));
            }
            s.push('\n');
        }
        std::fs::write(out_dir.join(format!("{fname}_occbin_{name}.csv")), s)?;
        Ok(())
    };
    write_csv("piecewise", &r.piecewise)?;
    if let Some(l) = &r.linear {
        write_csv("linear", l)?;
    }
    Ok(())
}

/// Port of `write_regimes_to_xls.m`, as CSV (`<filename>.csv` in `Output/`).
pub fn write_regimes_csv(
    regime_history: &RegimeHistory,
    o: &WriteRegimesOptions,
    out_dir: &Path,
) -> Result<std::path::PathBuf> {
    std::fs::create_dir_all(out_dir)?;
    let periods: Vec<usize> = o.periods.clone().unwrap_or_else(|| (1..=regime_history.len()).collect());
    let two = regime_history.first().is_some_and(|h| h.len() == 2);
    let mut s = if two {
        String::from(
            "time,regime_sequence_1,starting_period_of_regime_1,regime_sequence_2,starting_period_of_regime_2\n",
        )
    } else {
        String::from("time,regime_sequence,starting_period_of_regime\n")
    };
    for &t in &periods {
        let h = regime_history
            .get(t.wrapping_sub(1))
            .ok_or_else(|| anyhow!("occbin_write_regimes: period {t} is out of range"))?;
        s.push_str(&t.to_string());
        for g in h {
            s.push_str(&format!(",{},{}", regime_string(&g.regime), start_string(&g.start)));
        }
        s.push('\n');
    }
    let p = out_dir.join(format!("{}.csv", o.filename));
    std::fs::write(&p, s)?;
    Ok(p)
}

/// Port of `+occbin/graph.m`: piecewise (blue, solid) vs linear (red, dashed) paths, 3x3
/// panels per figure; shocks in the list are plotted from `shocks_sequence`.
pub fn graph(mf: &ModFile, r: &SimulResult, vars: &[String], steady_state: bool) -> Result<Vec<Figure>> {
    let m = &mf.model;
    let list: Vec<String> = if vars.is_empty() { m.endo_names[..m.orig_endo_nbr].to_vec() } else { vars.to_vec() };
    let np = r.piecewise.nrows();
    let x: Vec<f64> = (1..=np).map(|t| t as f64).collect();
    let mut panels = Vec::new();
    // endogenous first, then exogenous (varlist_indices order within each group)
    for name in &list {
        if let Some(j) = m.endo_index(name) {
            let off = if steady_state { 0.0 } else { r.ys[j] };
            let mut series = vec![Series::HLine { y: 0.0, color: Color::Black, style: LineStyle::Dashed }];
            series.push(Series::Line {
                x: x.clone(),
                y: (0..np).map(|t| r.piecewise[(t, j)] - off).collect(),
                color: Color::Blue,
                style: LineStyle::Solid,
                width: 2.0,
                label: Some("Piecewise Linear".into()),
            });
            if let Some(l) = &r.linear {
                series.push(Series::Line {
                    x: x.clone(),
                    y: (0..np).map(|t| l[(t, j)] - off).collect(),
                    color: Color::Red,
                    style: LineStyle::Dashed,
                    width: 2.0,
                    label: Some("Linear".into()),
                });
            }
            panels.push(Panel { title: name.clone(), series, legend: panels.is_empty(), ..Default::default() });
        } else if m.exo_index(name).is_none() {
            bail!("occbin_graph: unknown variable '{name}'");
        }
    }
    for name in &list {
        if let Some(k) = m.exo_index(name) {
            let c = r
                .exo_pos
                .iter()
                .position(|&e| e == k)
                .ok_or_else(|| anyhow!("{name} was not part of the shocks for OccBin."))?;
            let y: Vec<f64> =
                (0..np).map(|t| if t < r.shocks_sequence.nrows() { r.shocks_sequence[(t, c)] } else { 0.0 }).collect();
            let series = vec![
                Series::HLine { y: 0.0, color: Color::Black, style: LineStyle::Dashed },
                Series::Line { x: x.clone(), y, color: Color::Blue, style: LineStyle::Solid, width: 2.0, label: None },
            ];
            panels.push(Panel { title: name.clone(), series, ..Default::default() });
        }
    }
    let (nbplt, nr, nc, nstar) = graphics::pltorg(panels.len());
    let mut figs = Vec::new();
    for f in 0..nbplt {
        let chunk: Vec<Panel> = panels.iter().skip(f * nstar).take(nstar).cloned().collect();
        let (rows, cols) = if f + 1 == nbplt && chunk.len() < nstar {
            let (_, r2, c2, _) = graphics::pltorg(chunk.len());
            (r2, c2)
        } else {
            (nr, nc)
        };
        figs.push(Figure {
            name: format!("occbin_{}", f + 1),
            title: format!("OccBin simulated paths, figure {}", f + 1),
            rows,
            cols,
            panels: chunk,
        });
    }
    Ok(figs)
}
