// SPDX-License-Identifier: GPL-3.0-or-later
//! `model(block)`: block-wise solving.
//!
//! * Perfect foresight: `solve_block_decomposed_problem.m`. The blocks of
//!   `M_.block_structure` are solved one after the other (each only depends on the blocks
//!   before it): `evaluate{Forward,Backward}` and `solve{Forward,Backward}{Simple,Complete}`
//!   blocks period by period (forward or backward in time, `solve_one_boundary.m`: Newton to
//!   `dynatol.f`, at most `simul.maxit` iterations), `solveTwoBoundaries{Simple,Complete}`
//!   blocks by a Newton iteration on the block's stacked system over all periods
//!   (`solve_two_boundaries_stacked.m`; `stack_solve_algo` 1/6, LBJ, solves the same Newton
//!   systems with the sparse LU as every other value; no path-length search for 4).
//!   Dynare iterates only on the `mfs` feedback variables of a block after evaluating its
//!   recursive equations in closed form (normalized equations); here all variables of the block
//!   are Newton unknowns, which gives the same solution (the recursive equations are
//!   solved in the first iteration when they are linear in their variable).
//! * Steady state: `evaluate_steady_state.m`'s block branch (`dynare_solve` on each block).

use super::SimulOptions;
use super::sparse_lu::{LuWork, SparseLu};
use super::spmat::Csr;
use crate::linalg::Mat;
use crate::model::Model;
use crate::model::blocks::{BlockStructure, sim};
use crate::solver::{SolveError, SteadyOptions, dynare_solve};

/// `evaluate_steady_state.m`, `options_.block` branch: block by block, `dynare_solve` on the
/// block's variables with the block's equations (other variables held at their current value).
pub fn steady_blocks(
    model: &Model,
    bs: &BlockStructure,
    ys_init: &[f64],
    x: &[f64],
    p: &[f64],
    opts: &SteadyOptions,
) -> Result<Vec<f64>, SolveError> {
    let n = model.n_endo();
    let mut ys = ys_init.to_vec();
    model.set_auxiliary_variables(&mut ys, x, p);
    for blk in &bs.blocks {
        let (eqs, vars) = (&blk.equations, &blk.variables);
        let base = ys.clone();
        let f = |z: &[f64]| {
            let mut y = base.clone();
            for (k, &v) in vars.iter().enumerate() {
                y[v] = z[k];
            }
            let mut r = vec![0.0; model.n_equations()];
            model.static_residuals(&y, x, p, &mut r);
            let jac = model.static_jacobian(&y, x, p);
            (
                eqs.iter().map(|&e| r[e]).collect::<Vec<_>>(),
                Mat::from_fn(eqs.len(), vars.len(), |i, j| jac[(eqs[i], vars[j])]),
            )
        };
        let z0: Vec<f64> = vars.iter().map(|&v| ys[v]).collect();
        let out = dynare_solve(&f, &z0, opts);
        let bad = out.errorflag || out.x.iter().any(|v| !v.is_finite());
        for (k, &v) in vars.iter().enumerate() {
            ys[v] = out.x[k];
        }
        if bad {
            let mut r = vec![0.0; model.n_equations()];
            model.static_residuals(&ys, x, p, &mut r);
            let ss: f64 = eqs.iter().map(|&e| r[e] * r[e]).sum();
            return Err(SolveError::new(if ss.is_nan() { 22 } else { 20 }, ss));
        }
    }
    debug_assert_eq!(ys.len(), n);
    Ok(ys)
}

struct Ctx<'a> {
    model: &'a Model,
    ws: crate::model::Workspace,
    n: usize,
    y3: Vec<f64>,
    res: Vec<f64>,
    vals: Vec<f64>,
}

impl Ctx<'_> {
    /// Residuals (all equations) and Jacobian values at column `c` of `endo`.
    fn eval(&mut self, endo: &Mat, c: usize, x: &[f64], params: &[f64], jac: bool) {
        let n = self.n;
        for j in 0..n {
            self.y3[j] = if c > 0 { endo[(j, c - 1)] } else { f64::NAN };
            self.y3[n + j] = endo[(j, c)];
            self.y3[2 * n + j] = if c + 1 < endo.ncols() { endo[(j, c + 1)] } else { f64::NAN };
        }
        self.model.dynamic_residuals(&self.y3, x, params, &mut self.res);
        if jac {
            self.model.dynamic_jacobian_values_with(&self.y3, x, params, &mut self.ws, &mut self.vals);
        }
    }
}

fn block_residual_norm(res: &[f64], eqs: &[usize]) -> f64 {
    // norm(r, 'Inf'): NaN propagates
    let mut m = 0.0f64;
    for &e in eqs {
        let a = res[e].abs();
        if a.is_nan() {
            return f64::NAN;
        }
        m = m.max(a);
    }
    m
}

/// `solve_block_decomposed_problem.m`: `(success, maxerror, iterations)`.
#[allow(clippy::too_many_arguments)]
pub fn solve_block_decomposed(
    model: &Model,
    bs: &BlockStructure,
    o: &SimulOptions,
    endo: &mut Mat,
    exo: &Mat,
    params: &[f64],
    ml: usize,
) -> (bool, f64, usize) {
    let verbose = o.verbosity > 0 && !o.noprint;
    let n = model.n_endo();
    let periods = o.periods;
    let mthd = match o.stack_solve_algo {
        0 => "Sparse LU on stacked system, using mldivide".to_string(),
        1 | 6 => "LBJ with LU solver".to_string(),
        2 => format!("GMRES on stacked system with '{}' preconditioner", o.q1.preconditioner),
        3 => format!("BiCGStab on stacked system with '{}' preconditioner", o.q1.preconditioner),
        4 => "Sparse LU solver with optimal path length on stacked system".to_string(),
        7 => "Solver from solve_algo option on stacked system".to_string(),
        8 => "Sparse LU on stacked system, using ParU".to_string(),
        9 => "Sparse LU on stacked system, using PARDISO".to_string(),
        10 => "CGS on stacked system with preconditioner computed at first iteration with PARDISO".to_string(),
        _ => {
            println!("Unsupported stack_solve_algo value");
            return (false, f64::NAN, 0);
        }
    };
    if verbose {
        println!("{}", "-".repeat(41));
        println!("MODEL SIMULATION (method={}):\n", mthd);
    }
    let mut cx = Ctx {
        model,
        ws: model.workspace(),
        n,
        y3: vec![0.0; 3 * n],
        res: vec![0.0; model.n_equations()],
        vals: vec![0.0; model.dynamic_jacobian_pattern().len()],
    };
    let pattern: Vec<(usize, usize)> = model.dynamic_jacobian_pattern().to_vec();
    let exo_row = |c: usize| -> Vec<f64> { exo.row(c).iter().copied().collect() };
    let mut maxerror = 0.0f64;
    let mut total_iter = 0;
    for (b, blk) in bs.blocks.iter().enumerate() {
        let nb = blk.variables.len();
        let mut eq_local = vec![usize::MAX; model.n_equations()];
        for (k, &e) in blk.equations.iter().enumerate() {
            eq_local[e] = k;
        }
        let mut var_local = vec![usize::MAX; n];
        for (k, &v) in blk.variables.iter().enumerate() {
            var_local[v] = k;
        }
        let forward = matches!(
            blk.simulation_type,
            sim::EVALUATE_FORWARD | sim::SOLVE_FORWARD_SIMPLE | sim::SOLVE_FORWARD_COMPLETE
        );
        let (mut success, mut blkerr, mut iters) = (true, 0.0f64, 0usize);
        match blk.simulation_type {
            sim::SOLVE_TWO_BOUNDARIES_SIMPLE | sim::SOLVE_TWO_BOUNDARIES_COMPLETE => {
                // stacked Newton over the block's variables and all periods
                let mut iter = 0;
                let mut cvg = false;
                while !cvg && iter <= o.maxit {
                    let mut trip: Vec<(usize, usize, f64)> = Vec::new();
                    let mut r = vec![0.0; nb * periods];
                    let mut maxres = 0.0f64;
                    for t in 0..periods {
                        let c = ml + t;
                        cx.eval(endo, c, &exo_row(c), params, true);
                        for (k, &e) in blk.equations.iter().enumerate() {
                            r[t * nb + k] = cx.res[e];
                        }
                        let m = block_residual_norm(&cx.res, &blk.equations);
                        maxres = if m.is_nan() || maxres.is_nan() { f64::NAN } else { maxres.max(m) };
                        for (kk, &(i, col)) in pattern.iter().enumerate() {
                            if col >= 3 * n || eq_local[i] == usize::MAX {
                                continue;
                            }
                            let v = col % n;
                            if var_local[v] == usize::MAX {
                                continue;
                            }
                            let tt = t as isize + (col / n) as isize - 1;
                            if tt < 0 || tt >= periods as isize {
                                continue;
                            }
                            trip.push((t * nb + eq_local[i], tt as usize * nb + var_local[v], cx.vals[kk]));
                        }
                    }
                    blkerr = maxres;
                    if maxres.is_nan() {
                        success = false;
                        break;
                    }
                    cvg = (blk.linear && iter > 0) || maxres < o.tolf;
                    if !cvg && o.stack_solve_algo == 4 {
                        // solve_two_boundaries_stacked.m, stack_solve_algo=4: Newton direction p = -J\r
                        // with the lnsrch1 path-length search (stpmx = 100)
                        let a = Csr::from_triplets(nb * periods, nb * periods, trip.clone()).to_csc();
                        let prefer: Vec<usize> = (0..nb * periods).collect();
                        let ya: Vec<f64> = (0..periods)
                            .flat_map(|t| blk.variables.iter().map(move |&v| (t, v)))
                            .map(|(t, v)| endo[(v, ml + t)])
                            .collect();
                        let mut g = vec![0.0; nb * periods];
                        for &(i, j, v) in &trip {
                            g[j] += v * r[i];
                        }
                        let f0 = 0.5 * r.iter().map(|v| v * v).sum::<f64>();
                        let pdir = SparseLu::factor(&a, Some(&prefer), 0.001, &mut LuWork::default()).map(|f| {
                            let mut x = r.clone();
                            f.solve(&mut x);
                            x.iter().map(|v| -v).collect::<Vec<f64>>()
                        });
                        let Some(pdir) = pdir else {
                            println!("The Jacobian matrix is singular in block {}.", b + 1);
                            success = false;
                            break;
                        };
                        let stpmax = 100.0 * ya.iter().map(|v| v * v).sum::<f64>().sqrt().max(nb as f64);
                        let nn: Vec<usize> = (0..nb * periods).collect();
                        let mut work = endo.clone();
                        let mut func = |xv: &[f64]| -> Vec<f64> {
                            for t in 0..periods {
                                for (k, &v) in blk.variables.iter().enumerate() {
                                    work[(v, ml + t)] = xv[t * nb + k];
                                }
                            }
                            let mut rr = vec![0.0; nb * periods];
                            for t in 0..periods {
                                let c = ml + t;
                                cx.eval(&work, c, &exo_row(c), params, false);
                                for (k, &e) in blk.equations.iter().enumerate() {
                                    rr[t * nb + k] = cx.res[e];
                                }
                            }
                            rr
                        };
                        let Some(ls) = crate::solver::steady_state::lnsrch1(
                            &ya, f0, &g, pdir, stpmax, &mut func, &nn, &nn, o.tolx,
                        ) else {
                            println!(
                                "Some element of Newton direction isn't finite. Jacobian maybe singular or there is a problem with initial values"
                            );
                            success = false;
                            break;
                        };
                        for t in 0..periods {
                            for (k, &v) in blk.variables.iter().enumerate() {
                                endo[(v, ml + t)] = ls.x[t * nb + k];
                            }
                        }
                    } else if !cvg {
                        // threshold pivoting preferring the block's equation <-> variable pairing
                        // (the diagonal), as the matched pivots of the stacked solver
                        let a = Csr::from_triplets(nb * periods, nb * periods, trip).to_csc();
                        let prefer: Vec<usize> = (0..nb * periods).collect();
                        let d = SparseLu::factor(&a, Some(&prefer), 0.001, &mut LuWork::default()).map(|f| {
                            let mut x = r.clone();
                            f.solve(&mut x);
                            x
                        });
                        let Some(d) = d else {
                            println!("The Jacobian matrix is singular in block {}.", b + 1);
                            success = false;
                            break;
                        };
                        for t in 0..periods {
                            for (k, &v) in blk.variables.iter().enumerate() {
                                endo[(v, ml + t)] -= d[t * nb + k];
                            }
                        }
                    }
                    iter += 1;
                    if verbose {
                        println!("iteration: {} error: {}", iter, super::fmt_e(maxres));
                    }
                }
                iters = iter;
                if success && !cvg {
                    println!(
                        "Convergence not achieved in block {}, after {} iterations.\n Increase \"maxit\" or set \"cutoff=0\" in model options.",
                        b + 1,
                        iter
                    );
                    success = false;
                }
            }
            _ => {
                // evaluate / solve forward or backward: period by period (solve_one_boundary.m)
                let range: Vec<usize> =
                    if forward { (ml..ml + periods).collect() } else { (ml..ml + periods).rev().collect() };
                // evaluate blocks: every equation in closed form (the block function with
                // evaluate = true), whatever their mfs
                let nrec = if matches!(blk.simulation_type, sim::EVALUATE_FORWARD | sim::EVALUATE_BACKWARD) {
                    nb
                } else {
                    nb - blk.mfs.min(nb)
                };
                'periods: for c in range {
                    let x = exo_row(c);
                    let mut iter = 0;
                    let mut cvg = false;
                    while !(cvg || iter > o.maxit) {
                        // recursive equations in closed form (Dynare's normalized equations)
                        for k in 0..nrec {
                            let (e, v) = (blk.equations[k], blk.variables[k]);
                            for _ in 0..100 {
                                cx.eval(endo, c, &x, params, true);
                                let r = cx.res[e];
                                let d = (0..pattern.len())
                                    .find(|&kk| pattern[kk] == (e, n + v))
                                    .map(|kk| cx.vals[kk])
                                    .unwrap_or(0.0);
                                if r == 0.0 || !r.is_finite() || d == 0.0 {
                                    break;
                                }
                                let step = r / d;
                                endo[(v, c)] -= step;
                                if step.abs() <= 1e-15 * (1.0 + endo[(v, c)].abs()) {
                                    break;
                                }
                            }
                        }
                        cx.eval(endo, c, &x, params, true);
                        let fb_eqs = &blk.equations[nrec..];
                        let fb_vars = &blk.variables[nrec..];
                        let maxres = block_residual_norm(&cx.res, fb_eqs);
                        blkerr = maxres;
                        cvg = if maxres.is_nan() {
                            false
                        } else if blk.linear && iter > 0 {
                            true
                        } else {
                            maxres < o.tolf
                        };
                        if maxres.is_nan() {
                            break;
                        }
                        if !cvg {
                            // Newton step on the feedback variables with the Jacobian reduced
                            // by the recursive part (Schur complement)
                            let nf = fb_vars.len();
                            let mut jm = Mat::zeros(nb, nb);
                            for (kk, &(i, col)) in pattern.iter().enumerate() {
                                if col >= n
                                    && col < 2 * n
                                    && eq_local[i] != usize::MAX
                                    && var_local[col - n] != usize::MAX
                                {
                                    jm[(eq_local[i], var_local[col - n])] = cx.vals[kk];
                                }
                            }
                            let rf = Mat::from_fn(nf, 1, |k, _| cx.res[fb_eqs[k]]);
                            let mut jred = jm.view((nrec, nrec), (nf, nf)).into_owned();
                            if nrec > 0 {
                                let lu = jm.view((0, 0), (nrec, nrec)).into_owned().lu();
                                let rcols = jm.view((0, nrec), (nrec, nf)).into_owned();
                                if let Some(z) = lu.solve(&rcols) {
                                    jred -= jm.view((nrec, 0), (nf, nrec)).into_owned() * z;
                                }
                            }
                            let Some(dx) = jred.lu().solve(&rf) else {
                                println!("The Jacobian matrix is singular in block {}, at time {}.", b + 1, c + 1);
                                success = false;
                                break 'periods;
                            };
                            for (k, &v) in fb_vars.iter().enumerate() {
                                endo[(v, c)] -= dx[k];
                            }
                        }
                        iter += 1;
                    }
                    iters = iters.max(iter);
                    if !cvg {
                        println!(
                            "Convergence not achieved in block {}, at time {}, after {} iterations.\n Increase \"maxit\" or set \"cutoff=0\" in model options.",
                            b + 1,
                            c + 1,
                            iter
                        );
                        success = false;
                        break 'periods;
                    }
                }
            }
        }
        if success {
            // `any(isnan | isinf)` on the block's variables
            let bad = blk.variables.iter().any(|&v| (ml..ml + periods).any(|c| !endo[(v, c)].is_finite()));
            if bad {
                println!("Inf or Nan value during the resolution of block {}", b + 1);
                success = false;
            }
        }
        maxerror = if blkerr.is_nan() { f64::NAN } else { maxerror.max(blkerr) };
        total_iter += iters;
        if !success {
            return (false, maxerror, total_iter);
        }
    }
    (true, maxerror, total_iter)
}
