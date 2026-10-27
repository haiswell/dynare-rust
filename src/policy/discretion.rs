// SPDX-License-Identifier: GPL-3.0-or-later
//! Optimal discretionary policy in linear-quadratic models: port of
//! `discretionary_policy/discretionary_policy_1.m` and `discretionary_policy_engine.m`
//! (Dennis 2007, "Optimal policy in rational expectations models: new solution algorithms").

use crate::deterministic::SteadyStateModel;
use crate::linalg::{Mat, eig_general, mul, solve};
use crate::model::Model;
use crate::solver::{DecisionRule, DrStructure, SteadyOptions};

/// `options_.dp.maxit`, `options_.discretionary_tol`.
#[derive(Clone, Copy, Debug)]
pub struct DiscretionOptions {
    pub maxit: usize,
    pub tol: f64,
    pub qz_criterium: f64,
}

impl Default for DiscretionOptions {
    fn default() -> Self {
        DiscretionOptions { maxit: 3000, tol: 1e-7, qz_criterium: 1.0 + 1e-6 }
    }
}

fn sel(a: &Mat, rows: &[usize], cols: &[usize]) -> Mat {
    Mat::from_fn(rows.len(), cols.len(), |i, j| a[(rows[i], cols[j])])
}

/// MATLAB `max(max(abs(a)))` over several matrices: NaN entries are ignored unless all are NaN.
fn max_abs(mats: &[&Mat]) -> f64 {
    let mut m = f64::NAN;
    for a in mats {
        for v in a.iter() {
            if !v.is_nan() && (m.is_nan() || v.abs() > m) {
                m = v.abs();
            }
        }
    }
    m
}

/// MATLAB `norm(a, 1)` (largest column sum).
fn norm1(a: &Mat) -> f64 {
    (0..a.ncols()).map(|j| a.column(j).iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max)
}

/// `SylvesterDoubling`: solves `v = d + g v h` by doubling.
fn sylvester_doubling(d: &Mat, g: &Mat, h: &Mat, tol: f64, maxit: usize) -> Mat {
    let mut v = d.clone();
    let mut g = g.clone();
    let mut h = h.clone();
    for _ in 0..maxit {
        let vadd = mul(&mul(&g, false, &v, false), false, &h, false);
        v += &vadd;
        if norm1(&vadd) <= tol * norm1(&v) {
            break;
        }
        g = mul(&g, false, &g, false);
        h = mul(&h, false, &h, false);
    }
    v
}

/// Dense fallback for `v = d + g v h` (Dynare: `SylvesterHessenbergSchur`):
/// `(I - h' (x) g) vec(v) = vec(d)`.
fn sylvester_dense(d: &Mat, g: &Mat, h: &Mat) -> Mat {
    let (m, n) = (d.nrows(), d.ncols());
    let mut k = Mat::identity(m * n, m * n);
    for j in 0..n {
        for l in 0..n {
            let hlj = h[(l, j)];
            if hlj == 0.0 {
                continue;
            }
            for i in 0..m {
                for q in 0..m {
                    k[(j * m + i, l * m + q)] -= hlj * g[(i, q)];
                }
            }
        }
    }
    let rhs = Mat::from_column_slice(m * n, 1, d.as_slice());
    match solve(&k, &rhs) {
        Some(x) => Mat::from_column_slice(m, n, x.as_slice()),
        None => Mat::from_element(m, n, f64::NAN),
    }
}

/// Port of `discretionary_policy_engine.m`: `y_t = H y_{t-1} + G e_t` for all endogenous
/// variables, iterating from `H00` (the previous solution) or from zero. `Err` carries
/// Dynare's retcode (2 Sylvester failure, 61 maxit, 62 unstable, 63 NaN) and `retcode(2)`.
#[allow(clippy::too_many_arguments)]
pub fn discretionary_policy_engine(
    aalag: &Mat,
    aa0: &Mat,
    aalead: &Mat,
    bb: &Mat,
    bigw: &Mat,
    instr_id: &[usize],
    beta: f64,
    opts: &DiscretionOptions,
    h00: Option<&Mat>,
) -> Result<(Mat, Mat), (i32, f64)> {
    // GetDennisMatrices
    let (eq_nbr, endo_nbr) = aalag.shape();
    let exo_nbr = bb.ncols();
    let rows: Vec<usize> = (0..eq_nbr).collect();
    let y: Vec<usize> = (0..endo_nbr).filter(|j| !instr_id.contains(j)).collect();
    let instr_nbr = instr_id.len();
    let a0 = sel(aa0, &rows, &y);
    let a1 = -sel(aalag, &rows, &y);
    let a2 = -sel(aalead, &rows, &y);
    let a3 = -sel(aa0, &rows, instr_id);
    let a4 = -sel(aalead, &rows, instr_id);
    let a5 = -bb.clone();
    let w = sel(bigw, &y, &y);
    let q = sel(bigw, instr_id, instr_id);
    let a6 = -sel(aalag, &rows, instr_id);
    let aux: Vec<bool> = (0..instr_nbr).map(|c| a6.column(c).iter().any(|&v| v != 0.0)).collect();
    let aux_idx: Vec<usize> = (0..instr_nbr).filter(|&c| aux[c]).collect();
    let naux = aux_idx.len();
    let ny = eq_nbr;
    let m = eq_nbr + naux;
    let mut aa00 = Mat::zeros(m, m);
    aa00.view_mut((0, 0), (ny, ny)).copy_from(&a0);
    for k in 0..naux {
        aa00[(ny + k, ny + k)] = 1.0;
    }
    let mut aa11 = Mat::zeros(m, m);
    aa11.view_mut((0, 0), (ny, ny)).copy_from(&a1);
    for (k, &c) in aux_idx.iter().enumerate() {
        for r in 0..ny {
            aa11[(r, ny + k)] = a6[(r, c)];
        }
    }
    let mut aa22 = Mat::zeros(m, m);
    aa22.view_mut((0, 0), (ny, ny)).copy_from(&a2);
    let mut aa33 = Mat::zeros(m, instr_nbr);
    aa33.view_mut((0, 0), (ny, instr_nbr)).copy_from(&a3);
    for (k, &c) in aux_idx.iter().enumerate() {
        aa33[(ny + k, c)] = 1.0;
    }
    let mut aa44 = Mat::zeros(m, instr_nbr);
    aa44.view_mut((0, 0), (ny, instr_nbr)).copy_from(&a4);
    let mut aa55 = Mat::zeros(m, exo_nbr);
    aa55.view_mut((0, 0), (ny, exo_nbr)).copy_from(&a5);
    let mut ww = Mat::zeros(m, m);
    ww.view_mut((0, 0), (ny, ny)).copy_from(&w);
    let endo_augm_id: Vec<usize> = (0..endo_nbr + naux).filter(|j| !instr_id.contains(j)).collect();

    // H0 = zeros(endo_nbr + AuxiliaryVariables_nbr); H0(1:endo_nbr, 1:endo_nbr) = H00
    let mut h0 = Mat::zeros(endo_nbr + naux, endo_nbr + naux);
    if let Some(h) = h00 {
        h0.view_mut((0, 0), (endo_nbr, endo_nbr)).copy_from(h);
    }
    let mut h10 = sel(&h0, &endo_augm_id, &endo_augm_id);
    let mut f10 = sel(&h0, instr_id, &endo_augm_id);
    let mut h1 = h10.clone();
    let mut f1 = f10.clone();
    let mut iter = 0usize;
    let rcode;
    let (mut dinv, mut a3dpd);
    loop {
        iter += 1;
        let d = &ww + beta * mul(&mul(&f1, true, &q, false), false, &f1, false);
        let g = beta * h1.transpose();
        let mut p = sylvester_doubling(&d, &g, &h1, opts.tol, opts.maxit);
        if p.iter().any(|v| !v.is_finite()) {
            p = sylvester_dense(&d, &g, &h1);
            if p.iter().any(|v| !v.is_finite()) {
                return Err((2, 0.0));
            }
        }
        let dd = &aa00 - mul(&aa22, false, &h1, false) - mul(&aa44, false, &f1, false);
        dinv = match dd.clone().try_inverse() {
            Some(x) => x,
            None => return Err((63, 10000.0)),
        };
        a3dpd = mul(&mul(&mul(&aa33, true, &dinv, true), false, &p, false), false, &dinv, false);
        let lhs = &q + mul(&a3dpd, false, &aa33, false);
        let rhs = mul(&a3dpd, false, &aa11, false);
        f1 = -solve(&lhs, &rhs).unwrap_or_else(|| Mat::from_element(rhs.nrows(), rhs.ncols(), f64::NAN));
        h1 = mul(&dinv, false, &(&aa11 + mul(&aa33, false, &f1, false)), false);
        let nq = max_abs(&[&(&h1 - &h10), &(&f1 - &f10)]);
        if nq.is_nan() {
            rcode = 3;
            break;
        } else if iter > opts.maxit {
            rcode = 2;
            break;
        } else if nq < opts.tol {
            rcode = 1;
            break;
        }
        h10 = h1.clone();
        f10 = f1.clone();
    }
    match rcode {
        3 => return Err((63, 10000.0)),
        2 => return Err((61, 0.0)),
        _ => {
            let emax = eig_general(&h1).iter().map(|z| z.norm()).fold(0.0, f64::max);
            if emax > opts.qz_criterium {
                return Err((62, 100.0 * emax));
            }
        }
    }
    let lhs = &q + mul(&a3dpd, false, &aa33, false);
    let rhs = mul(&a3dpd, false, &aa55, false);
    let f2 = -solve(&lhs, &rhs).unwrap_or_else(|| Mat::from_element(rhs.nrows(), rhs.ncols(), f64::NAN));
    let h2 = mul(&dinv, false, &(&aa55 + mul(&aa33, false, &f2, false)), false);
    let nt = endo_nbr + naux;
    let mut hh = Mat::zeros(nt, nt);
    let mut gg = Mat::zeros(nt, exo_nbr);
    for (a, &ra) in endo_augm_id.iter().enumerate() {
        for (b, &cb) in endo_augm_id.iter().enumerate() {
            hh[(ra, cb)] = h1[(a, b)];
        }
        for c in 0..exo_nbr {
            gg[(ra, c)] = h2[(a, c)];
        }
    }
    for (k, &ri) in instr_id.iter().enumerate() {
        for (b, &cb) in endo_augm_id.iter().enumerate() {
            hh[(ri, cb)] = f1[(k, b)];
        }
        for c in 0..exo_nbr {
            gg[(ri, c)] = f2[(k, c)];
        }
    }
    // H(:, instr_id(aux)) = H(:, end-(naux-1:-1:0))
    for (k, &c) in aux_idx.iter().enumerate() {
        let src = nt - naux + k;
        let col = hh.column(src).into_owned();
        hh.set_column(instr_id[c], &col);
    }
    let h = hh.view((0, 0), (endo_nbr, endo_nbr)).into_owned();
    let g = gg.rows(0, endo_nbr).into_owned();
    Ok((h, g))
}

/// The steady-state file of `discretionary_policy_1.m` (`options_.steadystate_flag`): the
/// `steady_state_model`, `oo_.steady_state` (its initial values) and the solver options
/// (`options_.solve_tolf`).
pub struct DiscretionSteadyState<'a> {
    pub ssm: &'a SteadyStateModel,
    pub ys_init: &'a [f64],
    pub opts: &'a SteadyOptions,
    /// `oo_.exo_steady_state`, where the steady-state file is evaluated (the Jacobian is taken
    /// at zero exogenous values, `discretionary_policy_1.m:82`).
    pub exo_ss: &'a [f64],
}

/// Port of `discretionary_policy_1.m`: the decision rule, the parameters (a steady-state model
/// may update them) and `H`, the solution the persistent `Hold` keeps for the next call
/// (`hold`: the previous one). Without steady-state model the model is in deviations
/// (`ys = 0`). Errors: Dynare info codes (19/22 from the steady-state model, 64 NaN objective
/// derivatives, 65 non-zero residuals at the steady state, 66 non-zero objective gradient,
/// engine codes).
pub fn discretionary_policy_1(
    model: &Model,
    p: &[f64],
    opts: &DiscretionOptions,
    steady: Option<&DiscretionSteadyState>,
    tolf: f64,
    hold: Option<&Mat>,
) -> Result<(DecisionRule, Vec<f64>, Mat), (i32, f64)> {
    let pl = model.planner().ok_or((66, 0.0))?;
    let beta = model.planner_discount(p).unwrap_or(1.0);
    let n = model.n_endo();
    let nx = model.n_exo();
    let x = vec![0.0; nx];
    // call steady_state_file if present to update parameters
    let (ys, p) = match steady {
        Some(s) => {
            crate::deterministic::evaluate_steady_state(model, Some(s.ssm), s.ys_init, s.exo_ss, p, s.opts, true)
                .map_err(|e| (e.info, e.info2))?
        }
        None => (vec![0.0; n], p.to_vec()),
    };
    // the objective's derivatives at y = 0
    let obj = model.planner_objective(&vec![0.0; n], &x, &p).ok_or((64, 0.0))?;
    if obj.uy.iter().any(|v| v.is_nan()) || obj.uyy.iter().any(|v| v.is_nan()) {
        return Err((64, 0.0));
    }
    if obj.uy.iter().any(|&v| v != 0.0) {
        return Err((66, 0.0));
    }
    // the Jacobian at the steady state
    let y3: Vec<f64> = ys.iter().chain(&ys).chain(&ys).copied().collect();
    let mut resid = vec![0.0; model.n_equations()];
    model.dynamic_residuals(&y3, &x, &p, &mut resid);
    if resid.iter().any(|r| r.is_nan() || r.abs() > tolf) {
        return Err((65, 0.0));
    }
    let g1 = model.dynamic_jacobian(&ys, &x, &p);
    let neq = model.n_equations();
    let lag = g1.view((0, 0), (neq, n)).into_owned();
    let cur = g1.view((0, n), (neq, n)).into_owned();
    let lead = g1.view((0, 2 * n), (neq, n)).into_owned();
    let b = g1.view((0, 3 * n), (neq, nx)).into_owned();
    let (h, g) = discretionary_policy_engine(&lag, &cur, &lead, &b, &obj.uyy, &pl.instruments, beta, opts, hold)?;
    let s = DrStructure::new(&model.lead_lag_incidence, nx);
    let ov = &s.order_var;
    let t = Mat::from_fn(n, n, |i, j| h[(ov[i], ov[j])]);
    let ghu = Mat::from_fn(n, nx, |i, j| g[(ov[i], j)]);
    // Selection = lead_lag_incidence(1, order_var) > 0
    let cols: Vec<usize> = (0..n).filter(|&k| model.lead_lag_incidence[ov[k]][0] != 0).collect();
    let ghx = Mat::from_fn(n, cols.len(), |i, j| t[(i, cols[j])]);
    let dr = DecisionRule {
        ys,
        ghx,
        ghu,
        order_var: s.order_var.clone(),
        inv_order_var: s.inv_order_var.clone(),
        state_var: s.state_var.clone(),
        nstatic: s.nstatic,
        nfwrd: s.nfwrd,
        npred: s.npred,
        nboth: s.nboth,
        nspred: s.nspred,
        nsfwrd: s.nsfwrd,
        eigval: Vec::new(),
        eigval_abs: Vec::new(),
        sdim: 0,
    };
    Ok((dr, p, h))
}
