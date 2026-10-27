// SPDX-License-Identifier: GPL-3.0-or-later
//! `lyapunov = fixed_point | doubling | square_root_solver` of
//! `matrix_solver/lyapunov_solver.m` (stationary initialisation, lik_init = 1), with
//! `lyapunov_fixed_point_tol` and `lyapunov_doubling_tol`.
//!
//! * fixed point (`lyapunov_symm.m`, method 3): X = A X A' + B iterated from X = B until the
//!   largest column sum of |X - X_old| is below the tolerance (at most 2000 iterations, then the
//!   Schur method). Dynare warm-starts from the solution of the previous call (a `persistent`
//!   variable, at most 300 iterations then); here every call starts from B, so that an
//!   evaluation stays a pure function of the parameters (MCMC pre-fetching), which changes the
//!   result by at most about the tolerance. For the same reason the switch to the Schur method
//!   after a failed iteration applies to that call only, where Dynare's `persistent method1`
//!   keeps the Schur method for every later call of the session.
//! * doubling (`disclyap_fast.m`), falling back to the Schur method when it does not converge.
//! * square root solver (`dlyapchol` of the Control System Toolbox): the Cholesky factor of the
//!   solution; computed here as the Schur-method solution (the same matrix up to rounding).

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::collapsible_if
)] // index loops mirror the MATLAB code

use anyhow::Result;

use super::KalmanOptions;
use super::init::{lyapunov_solver, lyapunov_symm};
use crate::linalg::{Mat, mul};

/// `options_.lyapunov_fp / lyapunov_db / lyapunov_srs` and the tolerances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LyapunovOptions {
    /// 0 = Schur (default), 1 = fixed point, 2 = doubling, 3 = square root solver.
    pub method: u8,
    pub fixed_point_tol: f64,
    pub doubling_tol: f64,
}

impl Default for LyapunovOptions {
    fn default() -> Self {
        LyapunovOptions { method: 0, fixed_point_tol: 1e-10, doubling_tol: 1e-16 }
    }
}

/// Port of `disclyap_fast.m` (`max_iter = 2000`, no check): `None` on failure.
pub fn disclyap_fast(g: &Mat, v: &Mat, tol: f64) -> Option<Mat> {
    let max_iter = 2000;
    let (mut p0, mut a0) = (v.clone(), g.clone());
    let mut matd = 1.0f64;
    let mut iter = 1;
    while matd > tol && iter < max_iter {
        let p1 = &p0 + mul(&mul(&a0, false, &p0, false), false, &a0, true);
        let a1 = mul(&a0, false, &a0, false);
        matd = (&p1 - &p0).amax();
        p0 = p1;
        a0 = a1;
        iter += 1;
    }
    if iter == max_iter || matd.is_nan() {
        return None;
    }
    Some((&p0 + p0.transpose()) * 0.5)
}

/// Fixed-point branch of `lyapunov_symm.m` (method 3), from X = B: `None` if it does not
/// converge in 2000 iterations.
pub fn lyapunov_fixed_point(a: &Mat, b: &Mat, tol: f64) -> Option<Mat> {
    let mut x = b.clone();
    let mut evol = 100.0f64;
    let mut it = 0;
    while evol > tol && it < 2000 {
        let xn = mul(&mul(a, false, &x, false), false, a, true) + b;
        evol = (0..x.ncols())
            .map(|j| (0..x.nrows()).map(|i| (xn[(i, j)] - x[(i, j)]).abs()).sum::<f64>())
            .fold(0.0, f64::max);
        x = xn;
        it += 1;
    }
    (it < 2000).then_some(x)
}

/// `lyapunov_solver.m` with the `lyapunov` option of `opts` (the default is exactly
/// [`lyapunov_solver`]).
pub fn lyapunov_solver_opts(t: &Mat, r: &Mat, q: &Mat, opts: &KalmanOptions) -> Result<Mat> {
    let l = &opts.lyapunov;
    let schur = || lyapunov_solver(t, r, q, opts.qz_criterium, opts.lyapunov_complex_threshold);
    match l.method {
        1 => {
            let b = mul(&mul(r, false, q, false), false, r, true);
            match lyapunov_fixed_point(t, &b, l.fixed_point_tol) {
                Some(x) => Ok(x),
                None => {
                    println!(
                        "lyapunov_symm:: convergence not achieved in solution of Lyapunov equation after 2000 iterations, switching method from 3 to 0"
                    );
                    lyapunov_symm(t, &b, opts.qz_criterium, opts.lyapunov_complex_threshold)
                }
            }
        }
        2 => {
            let b = mul(&mul(r, false, q, false), false, r, true);
            match disclyap_fast(t, &b, l.doubling_tol) {
                Some(x) => Ok(x),
                None => lyapunov_symm(t, &b, opts.qz_criterium, opts.lyapunov_complex_threshold),
            }
        }
        _ => schur(),
    }
}
