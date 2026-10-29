// SPDX-License-Identifier: GPL-3.0-or-later
//! Ramsey steady state: port of `optimal_policy/dyn_ramsey_static.m` and of the
//! `options_.ramsey_policy` branch of `evaluate_steady_state.m`.
//!
//! Without steady-state file the unknowns are the declared endogenous variables. For a
//! candidate `x`, the multipliers are the least-squares solution of the static FOCs, which are
//! linear in them: `mult = -A \ y` with `y` the FOC residuals at zero multipliers (`dU/dy` plus
//! the constant parts) and `A = ramsey_multipliers_static_g1`; the residuals handed to
//! `dynare_solve` are the original equations plus the part of the FOCs the multipliers cannot
//! absorb (`norm(y + A*mult)` for one instrument, the last row of the QR factor of `[A y]'`
//! otherwise). As in Dynare, `dynare_solve` runs without analytic Jacobian (forward
//! differences with `gstep = 1e-2`, the same steps as `trust_region.m`/`solve1.m` take with
//! `jacobian_flag = 0`).
//!
//! With a `steady_state_model` (`steadystate_flag = 2`) the unknowns are the instruments only:
//! the block computes every other variable conditional on the instruments' values, and only the
//! FOC part of the residual remains; one instrument is solved by `csolve.m`, several by
//! `dynare_solve`.
//!
//! `ramsey_constraints`: the preprocessor turns them into `static_complementarity_conditions`,
//! which Dynare 7.2 only reads in the non-Ramsey branch of `evaluate_steady_state.m` (and in
//! perfect foresight); `dyn_ramsey_static.m` ignores them, and so does this port (they are
//! parsed, see [`parse_constraints`], and not used).

use anyhow::{Result, anyhow, bail};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::deterministic::SteadyStateModel;
use crate::linalg::{Mat, mul, qr_qt_mul, solve, trsm_left};
use crate::model::{CompiledExpr, Model};
use crate::solver::{SolveError, SteadyOptions, dynare_solve};

/// `options_.ramsey.maxit`.
pub const RAMSEY_MAXIT: usize = 500;

/// Least-squares solution of `a x = b` (MATLAB `a\b`: LU for square, QR otherwise).
pub(crate) fn backslash(a: &Mat, b: &Mat) -> Mat {
    if a.is_square() {
        return solve(a, b).unwrap_or_else(|| Mat::from_element(a.ncols(), b.ncols(), f64::NAN));
    }
    let (m, k) = (a.nrows(), a.ncols());
    let mut r = a.clone();
    qr_qt_mul(a, &mut r);
    let mut qb = b.clone();
    qr_qt_mul(a, &mut qb);
    let kk = m.min(k);
    let rr = r.view((0, 0), (kk, kk)).into_owned();
    let mut x = qb.rows(0, kk).into_owned();
    trsm_left(true, false, 1.0, &rr, &mut x);
    if k > kk {
        let mut full = Mat::zeros(k, b.ncols());
        full.rows_mut(0, kk).copy_from(&x);
        return full;
    }
    x
}

/// One `ramsey_constraints` line: `var > bound`, `var >= bound`, `var < bound`, `var <= bound`.
#[derive(Clone, Debug)]
pub struct RamseyConstraint {
    pub var: usize,
    /// `true` for a lower bound (`>`, `>=`).
    pub lower: bool,
    pub bound: CompiledBound,
}

/// The right-hand side of a constraint (an expression in the parameters).
#[derive(Clone, Debug)]
pub struct CompiledBound(CompiledExpr);

impl CompiledBound {
    pub fn eval(&self, p: &[f64]) -> f64 {
        self.0.eval(&[], &[], p)
    }
}

/// Parses the `ramsey_constraints` lines kept by the parser (`ModFile::policy.ramsey_constraints`);
/// the errors are those the preprocessor reports on the block.
pub fn parse_constraints(model: &Model, lines: &[String]) -> Result<Vec<RamseyConstraint>> {
    let mut out = Vec::new();
    for l in lines {
        let s = l.trim().trim_end_matches(';').trim();
        let (pos, op) = [">=", "<=", ">", "<"]
            .iter()
            .filter_map(|op| s.find(op).map(|p| (p, *op)))
            .min_by_key(|(p, op)| (*p, std::cmp::Reverse(op.len())))
            .ok_or_else(|| anyhow!("ramsey_constraints: cannot read '{s}' (expected VAR > VALUE or VAR < VALUE)"))?;
        let name = s[..pos].trim();
        let rhs = s[pos + op.len()..].trim();
        let var = model
            .endo_index(name)
            .filter(|&j| j < model.orig_endo_nbr)
            .ok_or_else(|| anyhow!("ramsey_constraints: '{name}' is not a declared endogenous variable"))?;
        let e = model.compile_expr(rhs).map_err(|e| anyhow!("ramsey_constraints: {s}: {e}"))?;
        out.push(RamseyConstraint { var, lower: op.starts_with('>'), bound: CompiledBound(e) });
    }
    Ok(out)
}

/// The Ramsey static problem (`dyn_ramsey_static.m` inputs).
pub struct RamseyStatic<'a> {
    pub model: &'a Model,
    /// The `steady_state_model` (`options_.steadystate_flag = 2`), if any.
    pub ssm: Option<&'a SteadyStateModel>,
    /// `oo_.exo_steady_state` (`dyn_ramsey_static` evaluates the static model there).
    pub exo_ss: &'a [f64],
}

struct Eval {
    resids: Vec<f64>,
    steady_state: Vec<f64>,
    params: Vec<f64>,
}

impl RamseyStatic<'_> {
    fn info(&self) -> &crate::model::planner::RamseyInfo {
        self.model.planner().and_then(|pl| pl.ramsey.as_ref()).expect("Ramsey model")
    }

    /// `options_.instruments` as endogenous indices (`k_inst`).
    fn k_inst(&self) -> Vec<usize> {
        self.model.planner().map(|pl| pl.instruments.clone()).unwrap_or_default()
    }

    /// `dyn_ramsey_static_1(x, exo_ss, ys_init, M_, options_)`.
    fn static_1(&self, p: &[f64], x: &[f64], ys_init: &[f64]) -> Eval {
        let model = self.model;
        let r = self.info();
        let n = model.n_endo();
        // M_.ramsey_orig_endo_nbr (the FOCs: declared variables and diff / EXPECTATION aux
        // variables) and M_.orig_endo_nbr (declared variables)
        let (n0, nd) = (r.orig_endo_nbr, model.orig_endo_nbr);
        let inst_nbr = n0 - r.orig_eq_nbr;
        let exo = self.exo_ss.to_vec();
        let mut params = p.to_vec();
        let mut full = x.to_vec();
        if let Some(ssm) = self.ssm {
            // ys_init(k_inst) = x; evaluate_steady_state_file (no check)
            let mut yi = ys_init.to_vec();
            for (&k, &v) in self.k_inst().iter().zip(x) {
                yi[k] = v;
            }
            ssm.eval(model, &mut yi, &exo, &mut params);
            if yi[..nd].iter().any(|v| !v.is_finite()) {
                // complex / NaN steady state: penalty (dynare-rust's NaN for MATLAB's complex)
                let pen = 1.0 + yi[..nd].iter().filter(|v| v.is_finite()).map(|v| v.abs()).sum::<f64>();
                return Eval { resids: vec![pen; inst_nbr], steady_state: vec![f64::NAN; n], params };
            }
            full = yi;
        }
        let nm = r.mult_index.len();
        let mut xx = vec![0.0; n];
        xx[..nd].copy_from_slice(&full[..nd]);
        let has_aux = !model.aux_vars.is_empty();
        if has_aux {
            model.set_auxiliary_variables(&mut xx, &exo, &params);
        }
        let mut res = vec![0.0; model.n_equations()];
        model.static_residuals(&xx, &exo, &params, &mut res);
        let a = model.ramsey_multipliers_static_g1(&xx, &exo, &params).expect("Ramsey model");
        let y = Mat::from_column_slice(n0, 1, &res[..n0]);
        let mult = -backslash(&a, &y);
        let resids1 = &y + mul(&a, false, &mult, false);
        let r1: Vec<f64> = if inst_nbr == 1 {
            vec![resids1.iter().map(|v| v * v).sum::<f64>().sqrt()]
        } else {
            // [~, r] = qr([A y]'); r1 = r(end, size(A,1)+(1-inst_nbr:0))'
            let mut ay = Mat::zeros(n0, nm + 1);
            ay.columns_mut(0, nm).copy_from(&a);
            ay.set_column(nm, &y.column(0));
            let m = ay.transpose();
            let mut rm = m.clone();
            qr_qt_mul(&m, &mut rm);
            let last = rm.nrows() - 1;
            (0..inst_nbr).map(|q| rm[(last, n0 - inst_nbr + q)]).collect()
        };
        let resids = if self.ssm.is_some() {
            r1
        } else {
            let mut v: Vec<f64> = res[n0..n0 + (nd - inst_nbr)].to_vec();
            v.extend(r1);
            v
        };
        let mut ss = vec![0.0; n];
        ss[..n0].copy_from_slice(&xx[..n0]);
        for (k, &j) in r.mult_index.iter().enumerate() {
            ss[j] = mult[(k, 0)];
        }
        if has_aux {
            model.set_auxiliary_variables(&mut ss, &exo, &params);
        }
        Eval { resids, steady_state: ss, params }
    }

    /// The Ramsey steady state (all endogenous variables, multipliers included) and the
    /// parameters (a steady-state model may update them), from the initial guess `ys_init`
    /// (`oo_.steady_state`: initval, multipliers 0 unless given).
    /// Errors are Dynare's info codes: 81 (not found), 82/83 (NaN), 84/85 (the steady-state
    /// model gives NaN / does not solve the original equations at the initial instruments),
    /// 87 (`csolve` reached `ramsey.maxit`).
    pub fn steady_state(
        &self,
        p: &[f64],
        ys_init: &[f64],
        opts: &SteadyOptions,
    ) -> Result<(Vec<f64>, Vec<f64>), SolveError> {
        let model = self.model;
        let r = self.info();
        let n0 = r.orig_endo_nbr;
        let exo = self.exo_ss.to_vec();
        let mut res = vec![0.0; model.n_equations()];
        let (ys, params) = if let Some(ssm) = self.ssm {
            // evaluate_steady_state.m: the file at the initial instruments must solve the
            // original equations
            let mut y1 = ys_init.to_vec();
            let mut p1 = p.to_vec();
            ssm.eval(model, &mut y1, &exo, &mut p1);
            model.static_residuals(&y1, &exo, &p1, &mut res);
            let orig = &res[n0..n0 + r.orig_eq_nbr];
            let ss: f64 = res.iter().map(|v| v * v).sum();
            if orig.iter().any(|v| v.is_nan()) {
                return Err(SolveError::new(84, ss));
            }
            if orig.iter().map(|v| v.abs()).fold(0.0, f64::max) > opts.tolf {
                return Err(SolveError::new(85, ss));
            }
            let k_inst = self.k_inst();
            if k_inst.is_empty() {
                return Err(SolveError::new(81, f64::NAN));
            }
            let x0: Vec<f64> = k_inst.iter().map(|&k| ys_init[k]).collect();
            let nl = |x: &[f64]| self.static_1(p, x, ys_init).resids;
            let x = if k_inst.len() == 1 {
                let (x, rc) = csolve(&nl, &x0, opts.tolf, RAMSEY_MAXIT);
                match rc {
                    1 | 3 => return Err(SolveError::new(81, f64::NAN)),
                    4 => return Err(SolveError::new(87, f64::NAN)),
                    _ => x,
                }
            } else {
                let out = self.solve_fd(&nl, &x0, opts);
                if out.errorflag {
                    return Err(SolveError::new(81, f64::NAN));
                }
                out.x
            };
            let e = self.static_1(p, &x, ys_init);
            (e.steady_state, e.params)
        } else {
            let mut ys0 = ys_init.to_vec();
            if !model.aux_vars.is_empty() {
                model.set_auxiliary_variables(&mut ys0, &exo, p);
            }
            model.static_residuals(&ys0, &exo, p, &mut res);
            if res.iter().map(|v| v * v).sum::<f64>().sqrt() < opts.tolf {
                (ys0, p.to_vec()) // check_static_model: the initial values already solve the model
            } else {
                let nl = |x: &[f64]| self.static_1(p, x, ys_init).resids;
                let out = self.solve_fd(&nl, &ys0[..model.orig_endo_nbr], opts);
                if out.errorflag {
                    return Err(SolveError::new(81, f64::NAN));
                }
                let e = self.static_1(p, &out.x, ys_init);
                (e.steady_state, e.params)
            }
        };
        model.static_residuals(&ys, &exo, &params, &mut res);
        if res[n0..].iter().any(|v| v.is_nan()) {
            return Err(SolveError::new(82, f64::NAN));
        }
        if res[..n0].iter().any(|v| v.is_nan()) {
            return Err(SolveError::new(83, f64::NAN));
        }
        let max = res.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if max > opts.tolf {
            return Err(SolveError::new(81, res.iter().map(|v| v * v).sum()));
        }
        Ok((ys, params))
    }

    /// `dynare_solve` with `jacobian_flag = 0` and `maxit = ramsey.maxit`.
    fn solve_fd(
        &self,
        nl: &dyn Fn(&[f64]) -> Vec<f64>,
        x0: &[f64],
        opts: &SteadyOptions,
    ) -> crate::solver::SolveOutcome {
        let f = |x: &[f64]| {
            let f0 = nl(x);
            let j = fd_jacobian(nl, x, &f0);
            (f0, j)
        };
        let o = SteadyOptions { maxit: RAMSEY_MAXIT, linear: false, ..*opts };
        dynare_solve(&f, x0, &o)
    }
}

/// Forward-difference Jacobian with Dynare's steps `max(|x|, gstep) * eps^(1/3)`.
fn fd_jacobian(f: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64], f0: &[f64]) -> Mat {
    let n = x.len();
    let mut j = Mat::zeros(f0.len(), n);
    let e3 = f64::EPSILON.cbrt();
    for c in 0..n {
        let dh = x[c].abs().max(1e-2) * e3;
        let mut xd = x.to_vec();
        xd[c] += dh;
        let f1 = f(&xd);
        for r in 0..f0.len() {
            j[(r, c)] = (f1[r] - f0[r]) / dh;
        }
    }
    j
}

/// Port of `optimization/csolve.m` (Sims) with numerical derivatives (`gradfun = ''`):
/// returns `(x, rc)`, `rc = 0` converged (`sum|f| < crit`), `4` after `itmax` iterations.
/// The random search MATLAB takes when Newton stalls draws from a seeded generator here.
pub fn csolve(fun: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64], crit: f64, itmax: usize) -> (Vec<f64>, i32) {
    let delta = 1e-6;
    let alpha = 1e-3;
    let nv = x0.len();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let mut x = x0.to_vec();
    let mut f0 = fun(&x);
    let asum = |v: &[f64]| v.iter().map(|z| z.abs()).sum::<f64>();
    let mut af0 = asum(&f0);
    let mut af00 = af0;
    let mut itct = 0usize;
    let mut rc;
    loop {
        let randomize = itct > 3 && af00 - af0 < crit * af0.max(1.0) && itct % 2 == 1;
        let dx0: Vec<f64> = if !randomize {
            let mut grad = Mat::zeros(f0.len(), nv);
            for i in 0..nv {
                let mut xi = x.clone();
                xi[i] += delta;
                let fi = fun(&xi);
                for r in 0..f0.len() {
                    grad[(r, i)] = (fi[r] - f0[r]) / delta;
                }
            }
            if rcond(&grad) < 1e-12 {
                for i in 0..nv.min(grad.nrows()) {
                    grad[(i, i)] += delta;
                }
            }
            let rhs = Mat::from_column_slice(f0.len(), 1, &f0);
            backslash(&grad, &rhs).iter().map(|v| -v).collect()
        } else {
            let nx = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            (0..nv)
                .map(|_| {
                    let z: f64 = StandardNormal.sample(&mut rng);
                    nx / z
                })
                .collect()
        };
        let mut lambda: f64 = 1.0;
        let mut fmin = f0.clone();
        let mut xmin = x.clone();
        let mut afmin = af0;
        let dx_size = dx0.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut factor: f64 = 0.6;
        let mut shrink = true;
        loop {
            let dx: Vec<f64> = dx0.iter().map(|v| lambda * v).collect();
            let xt: Vec<f64> = x.iter().zip(&dx).map(|(a, b)| a + b).collect();
            let f = fun(&xt);
            let af = asum(&f);
            if af < afmin {
                afmin = af;
                fmin = f.clone();
                xmin = xt;
            }
            if (lambda > 0.0 && af0 - af < alpha * lambda * af0) || (lambda < 0.0 && af0 - af < 0.0) {
                if !shrink {
                    factor = factor.powf(0.6);
                    shrink = true;
                }
                if (lambda * (1.0 - factor)).abs() * dx_size > 0.1 * delta {
                    lambda *= factor;
                } else if lambda > 0.0 && factor == 0.6 {
                    lambda = -0.3;
                } else {
                    rc = if lambda > 0.0 { if factor == 0.6 { 2 } else { 1 } } else { 3 };
                    break;
                }
            } else if lambda > 0.0 && af - af0 > (1.0 - alpha) * lambda * af0 {
                if shrink {
                    factor = factor.powf(0.6);
                    shrink = false;
                }
                lambda /= factor;
            } else {
                rc = 0;
                break;
            }
        }
        itct += 1;
        x = xmin;
        f0 = fmin;
        af00 = af0;
        af0 = afmin;
        if itct >= itmax {
            rc = 4;
            break;
        } else if af0 < crit {
            rc = 0;
            break;
        }
        let _ = rc;
    }
    (x, rc)
}

/// MATLAB `rcond` (1-norm reciprocal condition number; computed exactly from the inverse).
fn rcond(a: &Mat) -> f64 {
    if !a.is_square() || a.nrows() == 0 {
        return 0.0;
    }
    let norm1 = |m: &Mat| (0..m.ncols()).map(|j| m.column(j).iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max);
    match solve(a, &Mat::identity(a.nrows(), a.nrows())) {
        Some(inv) if inv.iter().all(|v| v.is_finite()) => 1.0 / (norm1(a) * norm1(&inv)),
        _ => 0.0,
    }
}

/// The Ramsey steady state without steady-state model (`dyn_ramsey_static.m`).
pub fn ramsey_steady_state(
    model: &Model,
    p: &[f64],
    ys_init: &[f64],
    opts: &SteadyOptions,
) -> Result<Vec<f64>, SolveError> {
    if model.planner().and_then(|pl| pl.ramsey.as_ref()).is_none() {
        return Err(SolveError::new(81, f64::NAN));
    }
    RamseyStatic { model, ssm: None, exo_ss: &vec![0.0; model.n_exo()] }.steady_state(p, ys_init, opts).map(|(y, _)| y)
}

/// Checks of `model_diagnostics.m` / `dyn_ramsey_static.m` on the `instruments` option.
pub fn check_instruments(model: &Model, has_ssm: bool) -> Result<()> {
    let Some(pl) = model.planner() else { return Ok(()) };
    let Some(r) = pl.ramsey.as_ref() else { return Ok(()) };
    let omitted = r.orig_endo_nbr - r.orig_eq_nbr;
    if has_ssm && pl.instruments.is_empty() {
        bail!(
            "Ramsey: with a steady-state model, the steady state is computed conditional on the instruments: declare them with the `instruments` option of ramsey_model / ramsey_policy"
        );
    }
    if !pl.instruments.is_empty() && pl.instruments.len() != omitted {
        println!(
            "WARNING: You have specified {} steady state instruments than there are omitted equations. While there are use cases for this setup, it is rather unusual. Check whether this is desired.",
            if pl.instruments.len() > omitted { "more" } else { "fewer" }
        );
    }
    Ok(())
}
