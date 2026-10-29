// SPDX-License-Identifier: GPL-3.0-or-later
//! Welfare: port of `optimal_policy/evaluate_planner_objective.m` (first-order branch and
//! linear-quadratic branch), plus the theoretical variances it needs (`th_autocovariances`
//! at order 1) and `osr.get_variance_of_endogenous_variables`.

use crate::kalman::lyapunov_symm;
use crate::linalg::{Mat, mul};
use crate::model::Model;
use crate::solver::DecisionRule;
use crate::solver::korder::KOrderDr;

/// `options_.lyapunov_fixed_point_tol`, `options_.lyapunov_complex_threshold`, `options_.huge_number`.
const LYAPUNOV_FIXED_POINT_TOL: f64 = 1e-10;
const LYAPUNOV_COMPLEX_THRESHOLD: f64 = 1e-15;
const HUGE_NUMBER: f64 = 1e7;

/// `oo_.planner_objective_value`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannerObjectiveValue {
    pub unconditional: f64,
    /// Ramsey: (with initial multipliers at zero, at their steady state);
    /// otherwise both entries hold the conditional welfare.
    pub conditional_zero_initial_multiplier: f64,
    pub conditional_steady_initial_multiplier: f64,
}

/// What `get_initial_state` (end of `evaluate_planner_objective.m`) starts the conditional
/// welfare from: `M_.endo_histval` and the initial shock `u` (from `M_.det_shocks`,
/// `oo_.exo_simul(1,:)` or `oo_.exo_steady_state`, as chosen by the caller).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InitialState {
    /// `M_.endo_histval` (levels, every endogenous variable); `None` without `histval`.
    pub endo_histval: Option<Vec<f64>>,
    /// `u` (`exo_nbr`; missing entries are zero).
    pub u: Vec<f64>,
}

impl InitialState {
    /// `(yhat_L_SS, yhat_L_0)` of `get_initial_state` over the state variables
    /// `order_var(nstatic + (1:nspred))`: the initial state minus the steady state, with the
    /// Lagrange multipliers at their steady state (`yhat_L_SS`) or at zero (`yhat_L_0`); the
    /// other variables start from `M_.endo_histval`, or from the steady state without histval.
    fn yhat(&self, model: &Model, dr: &DecisionRule) -> (Vec<f64>, Vec<f64>) {
        let ys = &dr.ys;
        let mults: Vec<usize> =
            model.planner().and_then(|pl| pl.ramsey.as_ref()).map(|r| r.mult_index.clone()).unwrap_or_default();
        let mut l_ss = ys.clone();
        let mut l_0 = vec![0.0; ys.len()];
        for j in (0..ys.len()).filter(|j| !mults.contains(j)) {
            match &self.endo_histval {
                Some(h) => {
                    l_ss[j] = h[j];
                    l_0[j] = h[j];
                }
                None => l_0[j] = ys[j],
            }
        }
        let states = (0..dr.nspred).map(|k| dr.order_var[dr.nstatic + k]);
        states.map(|j| (l_ss[j] - ys[j], l_0[j] - ys[j])).unzip()
    }

    /// `u` as a vector of length `nx`.
    fn shock(&self, nx: usize) -> Vec<f64> {
        (0..nx).map(|i| self.u.get(i).copied().unwrap_or(0.0)).collect()
    }
}

/// `lyapunov_symm(a, b, ..., method = 3)`: fixed-point iteration `X = a X a' + b` from
/// `X = b`, falling back to the Schur method when it does not converge in 2000 steps.
pub fn lyapunov_fixed_point(a: &Mat, b: &Mat, qz_criterium: f64) -> Mat {
    let mut x = b.clone();
    let mut evol = 100.0;
    let mut it = 0;
    while evol > LYAPUNOV_FIXED_POINT_TOL && it < 2000 {
        let old = x.clone();
        x = mul(&mul(a, false, &x, false), false, a, true) + b;
        evol = (0..x.ncols())
            .map(|j| (0..x.nrows()).map(|i| (x[(i, j)] - old[(i, j)]).abs()).sum::<f64>())
            .fold(f64::NAN, |m, v| if m.is_nan() || v > m { v } else { m });
        it += 1;
    }
    if it >= 2000 {
        return lyapunov_symm(a, b, qz_criterium, LYAPUNOV_COMPLEX_THRESHOLD)
            .unwrap_or_else(|_| Mat::from_element(b.nrows(), b.ncols(), f64::NAN));
    }
    x
}

/// `Gy = ghx(nstatic+(1:nspred),:)`, `Gu = ghu(nstatic+(1:nspred),:)`.
pub fn state_blocks(dr: &DecisionRule) -> (Mat, Mat) {
    let (s, k) = (dr.nstatic, dr.nspred);
    (dr.ghx.rows(s, k).into_owned(), dr.ghu.rows(s, k).into_owned())
}

/// Unconditional variance of the state variables (`lyapunov_symm(Gy, Gu Sigma Gu')`);
/// NaN on failure.
pub fn state_variance(dr: &DecisionRule, sigma_e: &Mat, qz_criterium: f64) -> Mat {
    let (gy, gu) = state_blocks(dr);
    if gy.nrows() == 0 {
        return Mat::zeros(0, 0);
    }
    let b = mul(&mul(&gu, false, sigma_e, false), false, &gu, true);
    lyapunov_symm(&gy, &b, qz_criterium, LYAPUNOV_COMPLEX_THRESHOLD)
        .unwrap_or_else(|_| Mat::from_element(b.nrows(), b.ncols(), f64::NAN))
}

/// Variance of the variables at DR positions `rows` (`ghx vx ghx' + ghu Sigma ghu'`), as
/// `th_autocovariances` (order 1) and `osr.get_variance_of_endogenous_variables`.
pub fn variance(dr: &DecisionRule, rows: &[usize], sigma_e: &Mat, qz_criterium: f64) -> Mat {
    let vx = state_variance(dr, sigma_e, qz_criterium);
    let ghx = Mat::from_fn(rows.len(), dr.ghx.ncols(), |i, j| dr.ghx[(rows[i], j)]);
    let ghu = Mat::from_fn(rows.len(), dr.ghu.ncols(), |i, j| dr.ghu[(rows[i], j)]);
    let mut v = mul(&mul(&ghu, false, sigma_e, false), false, &ghu, true);
    if vx.nrows() > 0 {
        v += mul(&mul(&ghx, false, &vx, false), false, &ghx, true);
    }
    v
}

/// `b' a c` for vectors/matrices: `A_times_B_kronecker_C(H, B, C)` reshaped, i.e.
/// `(B' H C)(a, b) = sum_ij H(i,j) B(i,a) C(j,b)`.
fn quad(h: &Mat, b: &Mat, c: &Mat) -> Mat {
    mul(&mul(b, true, h, false), false, c, false)
}

/// `sum(sum(a .* b))`.
fn dot(a: &Mat, b: &Mat) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Port of `evaluate_planner_objective.m`, `order = 1` (`lq = false`) or the
/// linear-quadratic branch (`lq = true`: `discretionary_policy`, or `order = 2` with a zero
/// model Hessian), from the steady state with a zero initial shock.
pub fn evaluate_planner_objective(
    model: &Model,
    p: &[f64],
    dr: &DecisionRule,
    sigma_e: &Mat,
    lq: bool,
    qz_criterium: f64,
) -> Option<PlannerObjectiveValue> {
    evaluate_planner_objective_from(model, p, dr, sigma_e, lq, qz_criterium, &InitialState::default())
}

/// [`evaluate_planner_objective`] from the initial state `init` of `get_initial_state`.
pub fn evaluate_planner_objective_from(
    model: &Model,
    p: &[f64],
    dr: &DecisionRule,
    sigma_e: &Mat,
    lq: bool,
    qz_criterium: f64,
    init: &InitialState,
) -> Option<PlannerObjectiveValue> {
    let pl = model.planner()?;
    let beta = model.planner_discount(p).unwrap_or(1.0);
    let n = model.n_endo();
    let nx = model.n_exo();
    let ys = &dr.ys;
    let obj = model.planner_objective(ys, &vec![0.0; nx], p)?;
    let ns = dr.nspred;
    let (gys, gus) = state_blocks(dr);
    let mut gy = Mat::zeros(n, ns);
    let mut gu = Mat::zeros(n, nx);
    for (k, &v) in dr.order_var.iter().enumerate() {
        gy.set_row(v, &dr.ghx.row(k));
        gu.set_row(v, &dr.ghu.row(k));
    }
    let uy = Mat::from_row_slice(1, n, &obj.uy);
    let u0 = obj.u;
    let wbar = u0 / (1.0 - beta);
    // Wy = Uy*gy / (eye(nspred) - beta*Gy)
    let uygy = mul(&uy, false, &gy, false);
    let ima = Mat::identity(ns, ns) - beta * &gys;
    let wy = if ns > 0 {
        crate::linalg::solve(&ima.transpose(), &uygy.transpose())
            .map(|x| x.transpose())
            .unwrap_or_else(|| Mat::from_element(1, ns, f64::NAN))
    } else {
        Mat::zeros(1, 0)
    };
    let wu = mul(&uy, false, &gu, false) + beta * mul(&wy, false, &gus, false);

    // get_initial_state
    let (yhat_ss, yhat_0) = init.yhat(model, dr);
    let yhat_ss = Mat::from_column_slice(ns, 1, &yhat_ss);
    let yhat_0 = Mat::from_column_slice(ns, 1, &yhat_0);
    let u = Mat::from_column_slice(nx, 1, &init.shock(nx));
    let lin = |yh: &Mat| wbar + dot(&wy, yh) + dot(&wu, &u);

    let (unconditional, w_ss, w_0) = if !lq {
        (u0 / (1.0 - beta), lin(&yhat_ss), lin(&yhat_0))
    } else {
        let h = &obj.uyy;
        let uyygygy = quad(h, &gy, &gy);
        let uyygugu = quad(h, &gu, &gu);
        let uyygugy = quad(h, &gu, &gy); // nx x ns
        let mut var = state_variance(dr, sigma_e, qz_criterium);
        var.iter_mut().for_each(|v| {
            if v.is_nan() {
                *v = HUGE_NUMBER
            }
        });
        let mut eu = u0 + 0.5 * dot(&uyygugu, sigma_e);
        if dr.nspred > 0 && model.lead_lag_incidence.iter().any(|r| r[0] != 0) {
            // Eyhat = 0 at order 1
            eu += 0.5 * dot(&uyygygy, &var);
        }
        let ew = eu / (1.0 - beta);
        let sb = beta.sqrt();
        let wyy = if ns > 0 {
            lyapunov_fixed_point(&(sb * gys.transpose()), &uyygygy.transpose(), qz_criterium)
        } else {
            Mat::zeros(0, 0)
        };
        let wyygugu = quad(&wyy, &gus, &gus);
        let wyygugy = quad(&wyy, &gus, &gys);
        let wuu = &uyygugu + beta * &wyygugu;
        let wss = beta * dot(&wuu, sigma_e) / (1.0 - beta);
        let wyu = &uyygugy + beta * &wyygugy; // nx x ns, Dynare's (exo, state) column order
        let second = |yh: &Mat| {
            // A_times_B_kronecker_C(Wyu, yhat, u): Wyu(1, (a-1)*ns+b) * yhat(a') u(b') with Dynare's index pairing
            let mut wyu_yu = 0.0;
            for s in 0..ns {
                for e in 0..nx {
                    let idx = s * nx + e; // kron(yhat, u)
                    let (a, b) = (idx / ns, idx % ns); // Wyu as row vector over (exo a, state b)
                    wyu_yu += wyu[(a, b)] * yh[(s, 0)] * u[(e, 0)];
                }
            }
            let wyy_yy = if ns > 0 { mul(&mul(yh, true, &wyy, false), false, yh, false)[(0, 0)] } else { 0.0 };
            let wuu_uu = mul(&mul(&u, true, &wuu, false), false, &u, false)[(0, 0)];
            lin(yh) + wyu_yu + 0.5 * (wss + wyy_yy + wuu_uu)
        };
        (ew, second(&yhat_ss), second(&yhat_0))
    };
    Some(if pl.ramsey.is_some() {
        PlannerObjectiveValue {
            unconditional,
            conditional_zero_initial_multiplier: w_0,
            conditional_steady_initial_multiplier: w_ss,
        }
    } else {
        PlannerObjectiveValue {
            unconditional,
            conditional_zero_initial_multiplier: w_ss,
            conditional_steady_initial_multiplier: w_ss,
        }
    })
}

// ---- second-order welfare ----

/// `A_times_B_kronecker_C(a, b, c)` for a row vector `a` (MATLAB layout, length
/// `b.nrows() * c.nrows()`, index `i * c.nrows() + j`): `a * kron(b, c)`, a row vector of
/// length `b.ncols() * c.ncols()` (index `p * c.ncols() + q`).
pub fn a_kron(a: &[f64], b: &Mat, c: &Mat) -> Vec<f64> {
    let (rb, rc, cb, cc) = (b.nrows(), c.nrows(), b.ncols(), c.ncols());
    debug_assert_eq!(a.len(), rb * rc);
    let am = Mat::from_fn(rb, rc, |i, j| a[i * rc + j]);
    let t = mul(&mul(b, true, &am, false), false, c, false); // cb x cc
    let mut out = vec![0.0; cb * cc];
    for p in 0..cb {
        for q in 0..cc {
            out[p * cc + q] = t[(p, q)];
        }
    }
    out
}

/// `kron(b, c)` of two column vectors.
fn kron_vec(b: &[f64], c: &[f64]) -> Vec<f64> {
    b.iter().flat_map(|x| c.iter().map(move |y| x * y)).collect()
}

/// `uy * g` for a row vector `uy` (length `g.nrows()`).
fn row_mul(uy: &[f64], g: &Mat) -> Vec<f64> {
    (0..g.ncols()).map(|c| (0..g.nrows()).map(|r| uy[r] * g[(r, c)]).sum()).collect()
}

fn vdot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Rows of `m` in DR order put back in declaration order (`g(dr.order_var,:) = m`).
fn decl_rows(m: &Mat, order_var: &[usize]) -> Mat {
    let mut g = Mat::zeros(m.nrows(), m.ncols());
    for (k, &v) in order_var.iter().enumerate() {
        g.set_row(v, &m.row(k));
    }
    g
}

/// Rows `nstatic .. nstatic + nspred` of `m` (the state rows, DR order).
fn state_rows(m: &Mat, dr: &DecisionRule) -> Mat {
    m.rows(dr.nstatic, dr.nspred).into_owned()
}

/// `disp_th_moments` / `th_autocovariances` at order 2 for the state variables (`var_list =
/// endo_names(order_var(nstatic + (1:nspred)))`): `(oo_.mean, oo_.var)`, NaN replaced by
/// `options_.huge_number` as `evaluate_planner_objective.m` does.
pub fn second_order_state_moments(kdr: &KOrderDr, sigma_e: &Mat, qz_criterium: f64) -> (Vec<f64>, Mat) {
    let dr = &kdr.first;
    let ns = dr.nspred;
    let a = state_rows(&dr.ghx, dr);
    let b = state_rows(&dr.ghu, dr);
    let bsb = mul(&mul(&b, false, sigma_e, false), false, &b, true);
    let vs = if ns > 0 {
        lyapunov_symm(&a, &bsb, qz_criterium, LYAPUNOV_COMPLEX_THRESHOLD)
            .unwrap_or_else(|_| Mat::from_element(ns, ns, f64::NAN))
    } else {
        Mat::zeros(0, 0)
    };
    // Gamma_y{1} of the states: aa*vs*aa' + bb*Sigma*bb' (aa = A), small entries zeroed
    let mut v = mul(&mul(&a, false, &vs, false), false, &a, true) + &bsb;
    v.iter_mut().for_each(|x| {
        if x.abs() < 1e-12 {
            *x = 0.0
        }
    });
    let vsv = vs.as_slice();
    let sv = sigma_e.as_slice();
    let corr: Vec<f64> = (0..dr.ghx.nrows())
        .map(|i| {
            kdr.ghs2[(i, 0)]
                + (0..vsv.len()).map(|k| kdr.ghxx[(i, k)] * vsv[k]).sum::<f64>()
                + (0..sv.len()).map(|k| kdr.ghuu[(i, k)] * sv[k]).sum::<f64>()
        })
        .collect();
    let ex_rhs = Mat::from_fn(ns, 1, |k, _| corr[dr.nstatic + k] / 2.0);
    let ex = crate::linalg::solve(&(Mat::identity(ns, ns) - &a), &ex_rhs)
        .unwrap_or_else(|| Mat::from_element(ns, 1, f64::NAN));
    let mut mean = vec![0.0; ns];
    for (k, m) in mean.iter_mut().enumerate() {
        let i = dr.nstatic + k;
        let gx: f64 = (0..ns).map(|j| dr.ghx[(i, j)] * ex[(j, 0)]).sum();
        *m = dr.ys[dr.order_var[i]] + gx + corr[i] / 2.0;
    }
    let huge = |x: &mut f64| {
        if x.is_nan() {
            *x = HUGE_NUMBER
        }
    };
    if mean.iter().any(|x| x.is_nan()) || v.iter().any(|x| x.is_nan()) {
        println!("evaluate_planner_objective: encountered NaN moments in the endogenous variables often associated");
        println!("evaluate_planner_objective: with either non-stationary variables or singularity due e.g. including");
        println!("evaluate_planner_objective: the planner objective function (or additive parts of it) in the model.");
        println!("evaluate_planner_objective: I will replace the NaN with a large number, but tread carefully,");
        println!("evaluate_planner_objective: check your model, and watch out for strange results.");
    }
    mean.iter_mut().for_each(huge);
    v.iter_mut().for_each(huge);
    (mean, v)
}

/// Port of the `options_.order == 2 && ~M_.hessian_eq_zero` branch of
/// `evaluate_planner_objective.m`: second-order approximation of the unconditional and
/// conditional welfare from the second-order decision rule `kdr` (`ghxx`, `ghxu`, `ghuu`,
/// `ghs2`) and the planner objective's Hessian, from the steady state with a zero initial shock.
pub fn evaluate_planner_objective_order2(
    model: &Model,
    p: &[f64],
    kdr: &KOrderDr,
    sigma_e: &Mat,
    qz_criterium: f64,
) -> Option<PlannerObjectiveValue> {
    evaluate_planner_objective_order2_from(model, p, kdr, sigma_e, qz_criterium, &InitialState::default())
}

/// [`evaluate_planner_objective_order2`] from the initial state `init` of `get_initial_state`.
pub fn evaluate_planner_objective_order2_from(
    model: &Model,
    p: &[f64],
    kdr: &KOrderDr,
    sigma_e: &Mat,
    qz_criterium: f64,
    init: &InitialState,
) -> Option<PlannerObjectiveValue> {
    let pl = model.planner()?;
    let dr = &kdr.first;
    let beta = model.planner_discount(p).unwrap_or(1.0);
    let nx = model.n_exo();
    let ns = dr.nspred;
    let ys = &dr.ys;
    let obj = model.planner_objective(ys, &vec![0.0; nx], p)?;
    let n = model.n_endo();
    let uy = &obj.uy;
    let u0 = obj.u;
    let uyy: Vec<f64> = (0..n * n).map(|k| obj.uyy[(k / n, k % n)]).collect();

    let gys = state_rows(&dr.ghx, dr);
    let gus = state_rows(&dr.ghu, dr);
    let gyy_s = state_rows(&kdr.ghxx, dr);
    let gyu_s = state_rows(&kdr.ghxu, dr);
    let guu_s = state_rows(&kdr.ghuu, dr);
    let gss_s = state_rows(&kdr.ghs2, dr);
    let ov = &dr.order_var;
    let gy = decl_rows(&dr.ghx, ov);
    let gu = decl_rows(&dr.ghu, ov);
    let gyy = decl_rows(&kdr.ghxx, ov);
    let gyu = decl_rows(&kdr.ghxu, ov);
    let guu = decl_rows(&kdr.ghuu, ov);
    let gss = decl_rows(&kdr.ghs2, ov);

    let uyygygy = a_kron(&uyy, &gy, &gy);
    let uyygugu = a_kron(&uyy, &gu, &gu);
    let uyygugy = a_kron(&uyy, &gu, &gy);
    let uy_gy = row_mul(uy, &gy);
    let uy_gu = row_mul(uy, &gu);
    let uy_gyy = row_mul(uy, &gyy);
    let uy_guu = row_mul(uy, &guu);
    let uy_gyu = row_mul(uy, &gyu);
    let uy_gss = row_mul(uy, &gss)[0];

    // unconditional welfare
    let (ey, var) = second_order_state_moments(kdr, sigma_e, qz_criterium);
    let eyhat: Vec<f64> = (0..ns).map(|k| ey[k] - ys[ov[dr.nstatic + k]]).collect();
    let eyy = var.as_slice();
    let euu = sigma_e.as_slice();
    let sum2 = |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
    let eu = u0
        + vdot(&uy_gy, &eyhat)
        + 0.5 * (vdot(&sum2(&uyygygy, &uy_gyy), eyy) + vdot(&sum2(&uyygugu, &uy_guu), euu) + uy_gss);
    let unconditional = eu / (1.0 - beta);

    // conditional welfare
    let wbar = u0 / (1.0 - beta);
    let wy: Vec<f64> = if ns > 0 {
        let ima = Mat::identity(ns, ns) - beta * &gys;
        let r = Mat::from_column_slice(ns, 1, &uy_gy);
        crate::linalg::solve(&ima.transpose(), &r)
            .map(|x| x.iter().copied().collect())
            .unwrap_or_else(|| vec![f64::NAN; ns])
    } else {
        Vec::new()
    };
    let wu: Vec<f64> = uy_gu.iter().zip(row_mul(&wy, &gus)).map(|(a, b)| a + beta * b).collect();
    let wy_gyy = row_mul(&wy, &gyy_s);
    let rhs: Vec<f64> = (0..ns * ns).map(|k| uyygygy[k] + uy_gyy[k] + beta * wy_gyy[k]).collect();
    // Wyy = gy'*Uyy*gy + Uy*gyy + beta*Wy*Gyy + beta*Gy'*Wyy*Gy (lyapunov_symm, method 3)
    let wyy: Vec<f64> = if ns > 0 {
        let b = Mat::from_column_slice(ns, ns, &rhs);
        lyapunov_fixed_point(&(beta.sqrt() * gys.transpose()), &b, qz_criterium).as_slice().to_vec()
    } else {
        Vec::new()
    };
    let wyygugu = a_kron(&wyy, &gus, &gus);
    let wyygugy = a_kron(&wyy, &gus, &gys);
    let wy_guu = row_mul(&wy, &guu_s);
    let wy_gyu = row_mul(&wy, &gyu_s);
    let wy_gss = row_mul(&wy, &gss_s)[0];
    let wuu: Vec<f64> = (0..nx * nx).map(|k| uyygugu[k] + uy_guu[k] + beta * (wyygugu[k] + wy_guu[k])).collect();
    let wss = (uy_gss + beta * (wy_gss + vdot(&wuu, euu))) / (1.0 - beta);
    // as Dynare: Uyygugy (shock-major) and Uy*gyu (state-major) are added element by element
    let wyu: Vec<f64> = (0..nx * ns).map(|k| uyygugy[k] + uy_gyu[k] + beta * (wyygugy[k] + wy_gyu[k])).collect();

    // get_initial_state
    let (yhat_ss, yhat_0) = init.yhat(model, dr);
    let u = init.shock(nx);
    let w = |yh: &[f64]| {
        let wyu_yu = vdot(&wyu, &kron_vec(yh, &u));
        let wyy_yy = vdot(&wyy, &kron_vec(yh, yh));
        let wuu_uu = vdot(&wuu, &kron_vec(&u, &u));
        wbar + vdot(&wy, yh) + vdot(&wu, &u) + wyu_yu + 0.5 * (wss + wyy_yy + wuu_uu)
    };
    let (w_ss, w_0) = (w(&yhat_ss), w(&yhat_0));
    Some(if pl.ramsey.is_some() {
        PlannerObjectiveValue {
            unconditional,
            conditional_zero_initial_multiplier: w_0,
            conditional_steady_initial_multiplier: w_ss,
        }
    } else {
        PlannerObjectiveValue {
            unconditional,
            conditional_zero_initial_multiplier: w_ss,
            conditional_steady_initial_multiplier: w_ss,
        }
    })
}

// ---- order >= 3: k_order_welfare + simulated unconditional welfare ----

/// Port of the `k_order_welfare` MEX: the unfolded derivatives `W_{x^k}` (`1 x nx^k`,
/// `k = 1..=order`, `x = (y*, u, σ)` as [`KOrderDr::derivs`]) of the welfare function
/// `W(y*_{t-1}, u_t, σ) = U(ys + g(x)) + β E_ε W(g*(x), σ ε, σ)` (`W(0) = U(ys)/(1-β)`),
/// solved order by order as the k-order solver: for each number `s` of `σ` indices, the
/// pure block `W_{y*^r σ^s}` from `X - β X (g*_{y*})^{⊗r} = R` (`gensylv`), then the blocks
/// with shocks explicitly. `E_ε` adds `σ² K(g*(x))`, `K(z) = tr(W_{uu}(z) Σ_e)/2`, to order 3.
/// `u3` are the sorted third derivatives of `U` (`objective.static_g3`).
pub fn k_order_welfare(
    kdr: &KOrderDr,
    uy: &[f64],
    uyy: &Mat,
    u3: &[(usize, usize, usize, f64)],
    beta: f64,
    sigma_e: &Mat,
) -> anyhow::Result<Vec<Mat>> {
    use crate::solver::korder::sylvester::gensylv;
    use crate::solver::korder::tensor::{add_sym3_first, col_of, kron_power_apply, tuple_of};
    let order = kdr.order;
    if order > 3 {
        anyhow::bail!("k_order_welfare: order > 3 goes through korder::general::k_order_welfare");
    }
    let dr = &kdr.first;
    let (ns, ne) = (dr.nspred, kdr.n_exo);
    let nw = ns + ne;
    let nx = nw + 1;
    let n = dr.ghx.nrows();
    let g: Vec<Mat> = kdr.derivs.iter().map(|d| decl_rows(d, &dr.order_var)).collect();
    // ψ_k: derivatives of x' = (g*(x), σε, σ) without the shock part (rows y*', u' = 0, σ')
    let psi: Vec<Mat> = kdr
        .derivs
        .iter()
        .enumerate()
        .map(|(k, d)| {
            let mut p = Mat::zeros(nx, d.ncols());
            p.rows_mut(0, ns).copy_from(&d.rows(dr.nstatic, ns));
            if k == 0 {
                p[(nw, nw)] = 1.0;
            }
            p
        })
        .collect();
    let uym = Mat::from_row_slice(1, uy.len(), uy);
    // U∘(ys + g): Faà di Bruno
    let mut a: Vec<Mat> = vec![&uym * &g[0]];
    let mut nz2: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if uyy[(i, j)] != 0.0 {
                nz2.push((i, j, uyy[(i, j)]));
            }
        }
    }
    if order >= 2 {
        let mut a2 = &uym * &g[1];
        for &(i, j, v) in &nz2 {
            for p in 0..nx {
                let gp = v * g[0][(i, p)];
                if gp != 0.0 {
                    for q in 0..nx {
                        a2[(0, p * nx + q)] += gp * g[0][(j, q)];
                    }
                }
            }
        }
        a.push(a2);
    }
    if order >= 3 {
        let mut a3 = &uym * &g[2];
        // Uyy (g1 ⊗ g2), symmetrized over the position of the g1 index
        let mut t = Mat::zeros(1, nx * nx * nx);
        for &(i, j, v) in &nz2 {
            for p in 0..nx {
                let gp = v * g[0][(i, p)];
                if gp != 0.0 {
                    for bc in 0..nx * nx {
                        t[(0, p * nx * nx + bc)] += gp * g[1][(j, bc)];
                    }
                }
            }
        }
        add_sym3_first(&mut a3, &t, nx);
        for &(i, j, l, v) in u3 {
            let mut perms = vec![[i, j, l], [i, l, j], [j, i, l], [j, l, i], [l, i, j], [l, j, i]];
            perms.sort_unstable();
            perms.dedup();
            for [i, j, l] in perms {
                for p in 0..nx {
                    let gp = v * g[0][(i, p)];
                    if gp == 0.0 {
                        continue;
                    }
                    for q in 0..nx {
                        let gq = gp * g[0][(j, q)];
                        if gq == 0.0 {
                            continue;
                        }
                        for r in 0..nx {
                            a3[(0, (p * nx + q) * nx + r)] += gq * g[0][(l, r)];
                        }
                    }
                }
            }
        }
        a.push(a3);
    }
    let mut w: Vec<Mat> = Vec::new();
    let gy = psi[0].view((0, 0), (ns, ns)).into_owned();
    let s_beta = Mat::from_element(1, 1, -beta);
    // 0.5 * sum_{b,c} f(u_b, u_c) Sigma_e(b, c)
    let tr = |f: &dyn Fn(usize, usize) -> f64| -> f64 {
        let mut s = 0.0;
        for b in 0..ne {
            for c in 0..ne {
                s += f(ns + b, ns + c) * sigma_e[(b, c)];
            }
        }
        0.5 * s
    };
    for k in 1..=order {
        // A_k + β (W_k ψ1^{⊗k} + lower Faà di Bruno terms of W∘ψ + σ² K terms)
        let p_of = |w: &[Mat], wk: &Mat| -> Mat {
            let mut l = kron_power_apply(wk, &psi[0], k);
            if k >= 2 {
                l += &w[0] * &psi[k - 1];
            }
            if k == 3 {
                let w2 = Mat::from_fn(nx, nx, |p, q| w[1][(0, p * nx + q)]);
                let m = psi[0].transpose() * w2 * &psi[1]; // nx x nx^2: (a, bc)
                let t = Mat::from_fn(1, nx * nx * nx, |_, c| m[(c / (nx * nx), c % (nx * nx))]);
                add_sym3_first(&mut l, &t, nx);
            }
            if k == 2 {
                let k0 = tr(&|b, c| wk[(0, b * nx + c)]);
                l[(0, nw * nx + nw)] += 2.0 * k0;
            }
            if k == 3 {
                let k1 = Mat::from_fn(1, nx, |_, a0| tr(&|b, c| wk[(0, (a0 * nx + b) * nx + c)]));
                let cv = &k1 * &psi[0];
                for a0 in 0..nx {
                    if a0 == nw {
                        l[(0, (nw * nx + nw) * nx + nw)] += 6.0 * cv[(0, nw)];
                    } else {
                        for idx in [[nw, nw, a0], [nw, a0, nw], [a0, nw, nw]] {
                            l[(0, col_of(&idx, nx))] += 2.0 * cv[(0, a0)];
                        }
                    }
                }
            }
            &a[k - 1] + beta * l
        };
        let ncol = nx.pow(k as u32);
        let mut wk = Mat::zeros(1, ncol);
        let mut tup = vec![0; k];
        let count =
            |t: &[usize]| (t.iter().filter(|&&i| i == nw).count(), t.iter().filter(|&&i| i >= ns && i < nw).count());
        for s in 0..=k {
            let r = k - s;
            let p = p_of(&w, &wk);
            // pure block (y*^r, σ^s), σ at the last s positions
            let nr = ns.pow(r as u32);
            let mut f = Mat::zeros(1, nr);
            let mut yt = vec![0; r];
            for c in 0..nr {
                tuple_of(c, ns, r, &mut yt);
                let mut idx = yt.clone();
                idx.extend(std::iter::repeat_n(nw, s));
                f[(0, c)] = p[(0, col_of(&idx, nx))];
            }
            let x = if nr == 0 { Mat::zeros(1, 0) } else { gensylv(r, &s_beta, &gy, &f)? };
            for c in 0..ncol {
                tuple_of(c, nx, k, &mut tup);
                if count(&tup) == (s, 0) {
                    let yi: Vec<usize> = tup.iter().copied().filter(|&i| i < ns).collect();
                    wk[(0, c)] = x[(0, col_of(&yi, ns))];
                }
            }
            // blocks with shocks
            let p = p_of(&w, &wk);
            for c in 0..ncol {
                tuple_of(c, nx, k, &mut tup);
                let (sc, uc) = count(&tup);
                if sc == s && uc > 0 {
                    wk[(0, c)] = p[(0, c)];
                }
            }
        }
        w.push(wk);
    }
    Ok(w)
}

/// Port of the `options_.order > 2` branch of `evaluate_planner_objective.m`: the welfare
/// rule ([`k_order_welfare`]) appended to the folded decision rule, the unconditional welfare
/// as the mean of a `k_order_mean` simulation from the steady state with the shocks
/// `exo_simul` (`n_exo x options_.ramsey.periods`, `chol(Sigma_e)*randn` in Dynare) after
/// `drop` periods (`options_.ramsey.drop`), and the conditional welfare as the first period
/// of a simulation from the initial state `init` (`get_initial_state`).
pub fn evaluate_planner_objective_korder(
    model: &Model,
    p: &[f64],
    kdr: &KOrderDr,
    sigma_e: &Mat,
    exo_simul: &Mat,
    drop: usize,
    init: &InitialState,
) -> anyhow::Result<PlannerObjectiveValue> {
    use crate::solver::korder::{fold_taylor, simulate::k_order_mean};
    let none = || anyhow::anyhow!("evaluate_planner_objective: no planner_objective");
    let pl = model.planner().ok_or_else(none)?;
    let dr = &kdr.first;
    let beta = model.planner_discount(p).unwrap_or(1.0);
    let nx = model.n_exo();
    let ns = dr.nspred;
    let ys = &dr.ys;
    let x0 = vec![0.0; nx];
    let obj = model.planner_objective(ys, &x0, p).ok_or_else(none)?;
    let u3 = model.planner_objective_g3(ys, &x0, p).unwrap_or_default();
    let w = if kdr.order > 3 {
        let ud = model.planner_objective_derivatives_k(kdr.order, ys, &x0, p).ok_or_else(none)?;
        crate::solver::korder::general::k_order_welfare(kdr, &ud, beta, sigma_e)?
    } else {
        k_order_welfare(kdr, &obj.uy, &obj.uyy, &u3, beta, sigma_e)?
    };
    let nw = ns + kdr.n_exo;
    let g: Vec<Mat> = (0..=kdr.order)
        .map(|j| {
            let gj = kdr.g_k(j);
            let wj = fold_taylor(&w, kdr.order, nw, j);
            let mut m = Mat::zeros(gj.nrows() + 1, gj.ncols());
            m.rows_mut(0, gj.nrows()).copy_from(&gj);
            m.set_row(gj.nrows(), &wj.row(0));
            m
        })
        .collect();
    let ov = &dr.order_var;
    let mut ysteady: Vec<f64> = ov.iter().map(|&j| ys[j]).collect();
    ysteady.push(obj.u / (1.0 - beta));
    let n1 = ysteady.len();
    let (moment, _) = k_order_mean(&g, dr.nstatic, ns, &ysteady, &vec![0.0; n1], exo_simul, drop);
    let unconditional = moment[n1 - 1];
    // get_initial_state; yhat_start(nstatic + (1:npred + nboth)) = yhat
    let (yhat_ss, yhat_0) = init.yhat(model, dr);
    let u = Mat::from_column_slice(nx, 1, &init.shock(nx));
    let cond = |yhat: &[f64]| -> f64 {
        let mut start = vec![0.0; n1];
        start[dr.nstatic..dr.nstatic + ns].copy_from_slice(yhat);
        let (_, sim) = k_order_mean(&g, dr.nstatic, ns, &ysteady, &start, &u, 0);
        sim[(n1 - 1, 0)]
    };
    let w_ss = cond(&yhat_ss);
    Ok(if pl.ramsey.is_some() {
        let w_0 = cond(&yhat_0);
        PlannerObjectiveValue {
            unconditional,
            conditional_zero_initial_multiplier: w_0,
            conditional_steady_initial_multiplier: w_ss,
        }
    } else {
        PlannerObjectiveValue {
            unconditional,
            conditional_zero_initial_multiplier: w_ss,
            conditional_steady_initial_multiplier: w_ss,
        }
    })
}
