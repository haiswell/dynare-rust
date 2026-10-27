// SPDX-License-Identifier: GPL-3.0-or-later
//! `perfect_foresight_controlled_paths`: port of `controlled_paths_by_period.m`,
//! `controlled_paths_substitute_stacked_jacobian.m` and of the controlled-paths branches of
//! `sim1.m`, `sim1_linear.m` and `perfect_foresight_solver.m` (`create_scenario`). In each
//! controlled period the exogenized endogenous variables are fixed to their values and the
//! endogenized exogenous variables become unknowns: the stacked Jacobian's column of the
//! exogenized variable is replaced by the column of the endogenized shock, and the Newton
//! step on that column updates the shock.
//!
//! Stream D's `DetState` keeps the per-period controls (`DetState::controlled`) and calls
//! [`sim1`] from its solver core when they are set.

use std::time::Instant;

use anyhow::{Result, bail};

use super::sparse::{LuPlan, SparseJac};
use crate::deterministic::stacked::StackedStructure;
use crate::deterministic::{SimulOptions, fmt_g};
use crate::linalg::Mat;
use crate::model::Model;
use crate::modfile::ModFile;

/// One entry of `M_.perfect_foresight_controlled_paths`.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledPath {
    pub learnt_in: usize,
    /// Endogenous variable whose value is imposed.
    pub exogenize: usize,
    /// Exogenous variable left free.
    pub endogenize: usize,
    /// 1-based simulation periods.
    pub periods: Vec<i64>,
    pub value: f64,
    // ---- shocks/shock_paths ----
    /// `learnt_in = DATE` (then `learnt_in` is unused).
    pub learnt_in_date: Option<crate::dates::Date>,
    /// Periods given as dates (then `periods` is empty).
    pub dates: Vec<crate::dates::Date>,
}

/// One period of `controlled_paths_by_period` (pairs `exogenize[q]` / `endogenize[q]`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ControlledPeriod {
    pub exogenize: Vec<usize>,
    pub endogenize: Vec<usize>,
    pub values: Vec<f64>,
    pub learnt_in: Vec<usize>,
}

/// Port of `controlled_paths_by_period.m` (`info_period = None`: every block, as
/// `perfect_foresight_setup`). `None` when the file has no controlled paths.
pub fn by_period(
    mf: &ModFile,
    periods: usize,
    info_period: Option<usize>,
    fsp: Option<crate::dates::Date>,
) -> Result<Option<Vec<ControlledPeriod>>> {
    by_period_with(mf, &[], periods, info_period, fsp)
}

/// [`by_period`] with extra entries (the `exogenize` stanzas of `shock_paths`).
pub fn by_period_with(
    mf: &ModFile,
    extra: &[ControlledPath],
    periods: usize,
    info_period: Option<usize>,
    fsp: Option<crate::dates::Date>,
) -> Result<Option<Vec<ControlledPeriod>>> {
    use crate::deterministic::periods::{learnt_in_period, to_sim_periods};
    let all: Vec<ControlledPath> = mf.ep.controlled_paths.iter().chain(extra).cloned().collect();
    let cps = &all;
    if cps.is_empty() {
        return Ok(None);
    }
    let m = &mf.model;
    let mut out = vec![ControlledPeriod::default(); periods];
    for c in cps {
        let learnt_in = learnt_in_period(c.learnt_in, c.learnt_in_date, fsp)?;
        if let Some(ip) = info_period
            && learnt_in > ip
        {
            continue;
        }
        let prange = to_sim_periods(&c.periods, &c.dates, fsp)?;
        if info_period.is_some() && prange.iter().any(|&p| p < learnt_in as i64) {
            bail!(
                "perfect_foresight_controlled_paths(learnt_in={0})/shock_paths(learnt_in={0})+exogenize: the periods are inconsistent with the learnt_in value",
                learnt_in
            );
        }
        for &p in &prange {
            if p < 1 || p as usize > periods {
                bail!("perfect_foresight_controlled_paths: period {p} is outside the simulation (1..{periods})");
            }
            let cp = &mut out[p as usize - 1];
            let idx = (0..cp.exogenize.len()).find(|&q| {
                cp.exogenize[q] == c.exogenize && cp.endogenize[q] == c.endogenize && cp.learnt_in[q] < learnt_in
            });
            match idx {
                None => {
                    cp.exogenize.push(c.exogenize);
                    cp.endogenize.push(c.endogenize);
                    cp.values.push(c.value);
                    cp.learnt_in.push(learnt_in);
                }
                Some(q) => {
                    cp.values[q] = c.value;
                    cp.learnt_in[q] = learnt_in;
                }
            }
        }
    }
    for (p, cp) in out.iter().enumerate() {
        for (q, j) in cp.exogenize.iter().enumerate() {
            if cp.exogenize[..q].contains(j) {
                bail!(
                    "perfect_foresight_controlled_paths/shock_paths+exogenize: variable {} is exogenized two times in period {}",
                    m.endo_names[*j],
                    p + 1
                );
            }
        }
        for (q, k) in cp.endogenize.iter().enumerate() {
            if cp.endogenize[..q].contains(k) {
                bail!(
                    "perfect_foresight_controlled_paths/shock_paths+exogenize: variable {} is endogenized two times in period {}",
                    m.exo_names[*k],
                    p + 1
                );
            }
        }
    }
    for s in &mf.det_shocks {
        for p in s.sim_periods(fsp)? {
            if p >= 1 && (p as usize) <= periods && out[p as usize - 1].endogenize.contains(&s.exo) {
                bail!(
                    "Exogenous variable {} is present in both the shocks and the perfect_foresight_controlled_paths in period {p}",
                    m.exo_names[s.exo]
                );
            }
        }
    }
    // ---- shocks/shock_paths: the learnt shocks on endogenized variables (as Dynare 7.2, every
    // entry is skipped or checked according to the learnt_in of the first one:
    // `learnt_in = M_.learnt_shocks.learnt_in`)
    if let (Some(ip), Some(first)) = (info_period, mf.ep.learnt_shocks.first()) {
        let learnt_in = learnt_in_period(first.learnt_in, first.learnt_in_date, fsp)?;
        if learnt_in <= ip {
            for s in &mf.ep.learnt_shocks {
                for p in to_sim_periods(&s.periods, &s.dates, fsp)? {
                    if p >= 1 && (p as usize) <= periods && out[p as usize - 1].endogenize.contains(&s.exo) {
                        bail!(
                            "Exogenous variable {} is present in both the shocks(learnt_in={learnt_in}) and the perfect_foresight_controlled_paths in period {p}",
                            m.exo_names[s.exo]
                        );
                    }
                }
            }
        }
    }
    Ok(Some(out))
}

/// `create_scenario`'s controlled-paths part: values scaled by `share` towards the base
/// path, and the guesses of the endogenized exogenous variables (rows of `x2`) taken from
/// the current iterate `exo`.
pub fn scale(
    cp: &[ControlledPeriod],
    share: f64,
    endobase: &Mat,
    exo: &Mat,
    x2: &mut Mat,
    ml: usize,
) -> Vec<ControlledPeriod> {
    let mut out = cp.to_vec();
    for (p, c) in out.iter_mut().enumerate() {
        if c.exogenize.is_empty() {
            continue;
        }
        for (q, &j) in c.exogenize.iter().enumerate() {
            c.values[q] = c.values[q] * share + endobase[(j, p)] * (1.0 - share);
        }
        for &k in &c.endogenize {
            x2[(ml + p, k)] = exo[(ml + p, k)];
        }
    }
    out
}

fn inf_norm(v: &[f64]) -> f64 {
    let mut m = 0.0f64;
    for &x in v {
        if x.is_nan() {
            return f64::NAN;
        }
        m = m.max(x.abs());
    }
    m
}

/// The stacked problem with controlled paths (`perfect_foresight_problem` + the Jacobian
/// substitution); `vals_fixed` = the dynamic Jacobian at the steady state for `sim1_linear`.
struct Stacked<'a> {
    model: &'a Model,
    n: usize,
    periods: usize,
    ml: usize,
    cp: &'a [ControlledPeriod],
    rows: Vec<u32>,
    cols: Vec<u32>,
    vals: Vec<f64>,
}

impl<'a> Stacked<'a> {
    fn column(&self, t: usize, c: usize) -> Option<usize> {
        let n = self.n;
        if c < 3 * n {
            let (l, v) = (c / n, c % n);
            let tt = t as i64 + l as i64 - 1;
            if tt < 0 || tt >= self.periods as i64 {
                return None;
            }
            let tt = tt as usize;
            if self.cp[tt].exogenize.contains(&v) {
                return None; // column replaced by the endogenized shock's
            }
            Some(tt * n + v)
        } else {
            let k = c - 3 * n;
            self.cp[t].endogenize.iter().position(|&e| e == k).map(|q| t * n + self.cp[t].exogenize[q])
        }
    }

    /// Residuals (`linear`: `J z` with `fixed` the Jacobian values at the steady state and
    /// deviations in `y`, `y0`, `yt`, `exo`) and Jacobian triplet values.
    #[allow(clippy::too_many_arguments)]
    fn eval(
        &mut self,
        y: &[f64],
        y0: &[f64],
        yt: &[f64],
        exo: &Mat,
        params: &[f64],
        res: &mut [f64],
        jac: bool,
        fixed: Option<&[f64]>,
    ) {
        let (n, periods, ml, model) = (self.n, self.periods, self.ml, self.model);
        let pattern = model.dynamic_jacobian_pattern();
        let mut ws = model.workspace();
        let mut y3 = vec![0.0; 3 * n];
        let mut jv = vec![0.0; pattern.len()];
        let nx = model.n_exo();
        let mut x = vec![0.0; nx];
        let record = self.rows.is_empty();
        let mut q = 0;
        for t in 0..periods {
            y3[..n].copy_from_slice(if t == 0 { y0 } else { &y[(t - 1) * n..t * n] });
            y3[n..2 * n].copy_from_slice(&y[t * n..(t + 1) * n]);
            y3[2 * n..].copy_from_slice(if t + 1 == periods { yt } else { &y[(t + 1) * n..(t + 2) * n] });
            for k in 0..nx {
                x[k] = exo[(t + ml, k)];
            }
            let r = &mut res[t * n..(t + 1) * n];
            match fixed {
                None => model.dynamic_residuals(&y3, &x, params, r),
                Some(f) => {
                    r.iter_mut().for_each(|v| *v = 0.0);
                    for (k, &(i, c)) in pattern.iter().enumerate() {
                        let z = if c < 3 * n { y3[c] } else { x[c - 3 * n] };
                        if f[k] != 0.0 {
                            r[i] += f[k] * z;
                        }
                    }
                }
            }
            if !jac {
                continue;
            }
            let vals: &[f64] = match fixed {
                None => {
                    model.dynamic_jacobian_values_with(&y3, &x, params, &mut ws, &mut jv);
                    &jv
                }
                Some(f) => f,
            };
            for (k, &(i, c)) in pattern.iter().enumerate() {
                if let Some(col) = self.column(t, c) {
                    if record {
                        self.rows.push((t * n + i) as u32);
                        self.cols.push(col as u32);
                        self.vals.push(vals[k]);
                    } else {
                        self.vals[q] = vals[k];
                    }
                    q += 1;
                }
            }
        }
    }
}

/// `sim1.m` / `sim1_linear.m` with a `controlled_paths_by_period` (stacked Newton; one linear
/// solve at the steady state when `linear`, whose residuals there are bounded by `solve_tolf`).
/// Updates `endo` and the endogenized entries of `exo`; returns (success, maxerror, iterations).
#[allow(clippy::too_many_arguments)]
pub fn sim1(
    model: &Model,
    o: &SimulOptions,
    endo: &mut Mat,
    exo: &mut Mat,
    ss: &[f64],
    xs: &[f64],
    params: &[f64],
    ml: usize,
    cp: &[ControlledPeriod],
    linear: bool,
    solve_tolf: f64,
) -> Result<(bool, f64, usize)> {
    let n = model.n_endo();
    let periods = o.periods;
    for (p, c) in cp.iter().enumerate().take(periods) {
        for (q, &j) in c.exogenize.iter().enumerate() {
            endo[(j, p + ml)] = c.values[q];
        }
    }
    let verbose = if linear { o.verbosity > 0 } else { o.verbosity > 0 && !o.noprint };
    if verbose {
        println!();
        println!("{}", "-".repeat(if linear { 80 } else { 56 }));
        println!("MODEL SIMULATION:");
        println!();
    }
    // linear: deviations from the steady state, Jacobian at the steady state
    let fixed = if linear {
        let mut y3n = Vec::with_capacity(3 * n);
        for _ in 0..3 {
            y3n.extend_from_slice(ss);
        }
        let mut d1 = vec![0.0; n];
        model.dynamic_residuals(&y3n, xs, params, &mut d1);
        if !o.steadystate_nocheck && d1.iter().fold(0.0f64, |m, v| m.max(v.abs())) > solve_tolf {
            bail!("Jacobian is not evaluated at the steady state!");
        }
        let mut vals = vec![0.0; model.dynamic_jacobian_pattern().len()];
        model.dynamic_jacobian_values_with(&y3n, xs, params, &mut model.workspace(), &mut vals);
        for c in 0..endo.ncols() {
            for j in 0..n {
                endo[(j, c)] -= ss[j];
            }
        }
        for r in 0..exo.nrows() {
            for k in 0..exo.ncols() {
                exo[(r, k)] -= xs[k];
            }
        }
        Some(vals)
    } else {
        None
    };
    let col = |e: &Mat, c: usize| -> Vec<f64> {
        if c < e.ncols() { e.column(c).iter().copied().collect() } else { vec![f64::NAN; n] }
    };
    let y0 = if ml > 0 { col(endo, ml - 1) } else { vec![f64::NAN; n] };
    let yt = col(endo, ml + periods);
    let mut y: Vec<f64> = (0..periods).flat_map(|t| col(endo, ml + t)).collect();
    let mut st = Stacked { model, n, periods, ml, cp, rows: vec![], cols: vec![], vals: vec![] };
    let mut res = vec![0.0; n * periods];
    let s = StackedStructure::new(model);
    let mut jac: Option<(SparseJac, LuPlan)> = None;
    let h1 = Instant::now();
    let mut iter = 1;
    let mut converged = false;
    let mut err = f64::NAN;
    let maxit = if linear { 1 } else { o.maxit };
    while !(converged || iter > maxit) {
        let h2 = Instant::now();
        st.eval(&y, &y0, &yt, exo, params, &mut res, true, fixed.as_deref());
        let (a, plan) = jac.get_or_insert_with(|| {
            let a = SparseJac::from_triplets(n * periods, &st.rows, &st.cols);
            let plan = LuPlan::new(&a, n, &s);
            (a, plan)
        });
        a.fill(&st.rows, &st.cols, &st.vals);
        err = inf_norm(&res);
        if !linear && err < o.tolf {
            converged = true;
        } else {
            // lin_solve.m: A\b, 0 if norm(b) < sqrt(eps), non-finite entries set to 0
            let nb = res.iter().map(|v| v * v).sum::<f64>().sqrt();
            let mut dy = if nb < f64::EPSILON.sqrt() {
                vec![0.0; res.len()]
            } else {
                match plan.factor(a) {
                    Some(f) => f.solve(&res),
                    None => {
                        if linear {
                            if verbose {
                                println!();
                                println!(
                                    "Singularity problem! The jacobian matrix of the stacked model cannot be inverted."
                                );
                            }
                            return Ok((false, f64::NAN, 1));
                        }
                        println!("Warning: Matrix is singular to working precision.");
                        vec![f64::NAN; res.len()]
                    }
                }
            };
            for v in &mut dy {
                if !v.is_finite() {
                    *v = 0.0;
                }
                *v = -*v;
            }
            for (v, d) in y.iter_mut().zip(&dy) {
                *v += d;
            }
            for (p, c) in cp.iter().enumerate().take(periods) {
                for (q, (&j, &k)) in c.exogenize.iter().zip(&c.endogenize).enumerate() {
                    y[p * n + j] = if linear { c.values[q] - ss[j] } else { c.values[q] };
                    exo[(p + ml, k)] += dy[p * n + j];
                }
            }
            if linear {
                let err0 = err;
                st.eval(&y, &y0, &yt, exo, params, &mut res, false, fixed.as_deref());
                err = inf_norm(&res);
                if verbose {
                    println!(
                        "Iter: 1,\t Initial err. = {},\t err. = {},\t time = {}",
                        fmt_g(err0),
                        fmt_g(err),
                        fmt_g(h2.elapsed().as_secs_f64())
                    );
                    println!("{}", "-".repeat(80));
                }
            }
        }
        if verbose && !linear {
            println!("Iter: {},\t err. = {},\t time = {}", iter, fmt_g(err), fmt_g(h2.elapsed().as_secs_f64()));
        }
        iter += 1;
    }
    for t in 0..periods {
        for j in 0..n {
            endo[(j, ml + t)] = y[t * n + j];
        }
    }
    if linear {
        for c in 0..endo.ncols() {
            for j in 0..n {
                endo[(j, c)] += ss[j];
            }
        }
        for r in 0..exo.nrows() {
            for k in 0..exo.ncols() {
                exo[(r, k)] += xs[k];
            }
        }
        let ok = !(res.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()));
        if !ok && verbose {
            println!();
            println!("Simulation terminated with NaN or Inf in the residuals or endogenous variables.");
        }
        if verbose {
            println!();
        }
        return Ok((ok, err, 1));
    }
    let success = converged && y.iter().all(|v| v.is_finite());
    if verbose {
        println!();
        println!("Total time of simulation: {}.", fmt_g(h1.elapsed().as_secs_f64()));
        if !converged {
            println!("Maximum number of iterations is reached (modify option maxit).");
        } else if !success {
            println!("Simulation terminated with NaN or Inf in the residuals or endogenous variables.");
        }
        println!();
    }
    Ok((success, err, iter))
}

/// `solve_stacked_problem.m` with a `controlled_paths_by_period` (`stack_solve_algo = 7`):
/// `dynare_solve` (`options_.solve_algo`) on `controlled_paths_wrapper`, whose unknowns are
/// the endogenous paths with the endogenized exogenous values in place of the exogenized
/// variables. Updates `endo` and `exo`; returns (success, maxerror, 0).
#[allow(clippy::too_many_arguments)]
pub fn solve_stacked_problem(
    model: &Model,
    o: &SimulOptions,
    solve_algo: u8,
    endo: &mut Mat,
    exo: &mut Mat,
    params: &[f64],
    ml: usize,
    cp: &[ControlledPeriod],
) -> (bool, f64, usize) {
    let n = model.n_endo();
    let periods = o.periods;
    let col = |e: &Mat, c: usize| -> Vec<f64> {
        if c < e.ncols() { e.column(c).iter().copied().collect() } else { vec![f64::NAN; n] }
    };
    let y0 = if ml > 0 { col(endo, ml - 1) } else { vec![f64::NAN; n] };
    let yt = col(endo, ml + periods);
    let mut z: Vec<f64> = (0..periods).flat_map(|t| col(endo, ml + t)).collect();
    for (p, c) in cp.iter().enumerate().take(periods) {
        for (&j, &k) in c.exogenize.iter().zip(&c.endogenize) {
            z[p * n + j] = exo[(p + ml, k)];
        }
    }
    // controlled_paths_wrapper: the exogenized values in y, the unknowns in exogenousvariables
    let substitute = |z: &[f64], x: &mut Mat| -> Vec<f64> {
        let mut y = z.to_vec();
        for (p, c) in cp.iter().enumerate().take(periods) {
            for (q, (&j, &k)) in c.exogenize.iter().zip(&c.endogenize).enumerate() {
                x[(p + ml, k)] = z[p * n + j];
                y[p * n + j] = c.values[q];
            }
        }
        y
    };
    let st = std::cell::RefCell::new(Stacked { model, n, periods, ml, cp, rows: vec![], cols: vec![], vals: vec![] });
    let x0 = exo.clone();
    let f = |z: &[f64]| {
        let mut x = x0.clone();
        let y = substitute(z, &mut x);
        let mut res = vec![0.0; n * periods];
        let mut st = st.borrow_mut();
        st.eval(&y, &y0, &yt, &x, params, &mut res, true, None);
        let mut a = Mat::zeros(n * periods, n * periods);
        for ((&r, &c), &v) in st.rows.iter().zip(&st.cols).zip(&st.vals) {
            a[(r as usize, c as usize)] += v;
        }
        (res, a)
    };
    let so =
        crate::solver::SteadyOptions { solve_algo, maxit: o.maxit, tolf: o.tolf, tolx: o.tolx, ..Default::default() };
    let out = crate::solver::dynare_solve(&f, &z, &so);
    let res = f(&out.x).0;
    let y = substitute(&out.x, exo);
    for t in 0..periods {
        for j in 0..n {
            endo[(j, ml + t)] = y[t * n + j];
        }
    }
    (!out.errorflag, inf_norm(&res), 0)
}
