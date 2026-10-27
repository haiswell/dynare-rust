// SPDX-License-Identifier: GPL-3.0-or-later
//! [stream X] The other `solve_algo` values of `dynare_solve.m`:
//!
//! - 0 (`fsolve`, MATLAB's Optimization Toolbox, trust-region-dogleg): the trust-region solver
//!   of `solve_algo = 9` on the whole system instead (same method family, not MATLAB's code);
//! - 3: port of `csolve.m` (Sims), without its random search when the Newton step stalls
//!   (reported as a failure, `rc = 4`, instead);
//! - 6, 7, 8: port of `newton_solve.m` (6: Newton with an LU solve; 7/8 use GMRES/BiCGStab
//!   preconditioned by an incomplete LU in MATLAB, here the same Newton step solved exactly);
//! - 13, 14: Dynare's `block_trust_region` MEX (see `block_trust_region.rs`): Dulmage-Mendelsohn
//!   blocks (13) or the whole system (14).
//!
//! 10 and 11 (LMMCP / PATH mixed complementarity problems) are not ported.

use super::{SolveOutcome, StaticFn, SteadyOptions, max_abs, trust_region};
use crate::linalg::{Mat, lu, lu_solve, norm1, rcond_from_lu};

/// `dynare_solve` for `solve_algo` 0, 3, 6, 7, 8, 13, 14 (`None` for the others).
pub(super) fn dynare_solve_x(f: StaticFn, x0: &[f64], opts: &SteadyOptions) -> Option<SolveOutcome> {
    let nn = x0.len();
    let all: Vec<usize> = (0..nn).collect();
    let mut x = x0.to_vec();
    Some(match opts.solve_algo {
        0 => {
            let (errorflag, errorcode) = trust_region(f, &mut x, &all, &all, opts.tolf, opts.tolx, opts.maxit, opts);
            SolveOutcome { x, errorflag, errorcode }
        }
        // ---- block_trust_region ----
        13 | 14 => crate::solver::block_trust_region::block_trust_region(f, x0, opts, opts.solve_algo == 13),
        3 => {
            let rc = csolve(f, &mut x, opts.tolf, opts.maxit);
            SolveOutcome { x, errorflag: rc != 0, errorcode: rc }
        }
        6..=8 => {
            let (errorflag, errorcode) = newton_solve(f, &mut x, opts.tolf, opts.maxit);
            SolveOutcome { x, errorflag, errorcode }
        }
        _ => return None,
    })
}

fn sum_abs(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).sum()
}

/// `grad \ f` (MATLAB backslash on a square matrix: LU).
fn backslash(a: &Mat, b: &[f64]) -> Option<Vec<f64>> {
    let (fct, ipiv, info) = lu(a);
    if info < 0 {
        return None;
    }
    let mut rhs = Mat::from_column_slice(b.len(), 1, b);
    lu_solve(&fct, &ipiv, false, &mut rhs);
    let v: Vec<f64> = rhs.iter().copied().collect();
    v.iter().all(|x| x.is_finite()).then_some(v)
}

/// Port of `optimization/csolve.m` with the analytic Jacobian; returns `rc` (0 converged,
/// 1-3 line search could not improve, 4 iteration limit or random search needed).
pub(super) fn csolve(f: StaticFn, x: &mut Vec<f64>, crit: f64, itmax: usize) -> i32 {
    let delta = 1e-6;
    let alpha = 1e-3;
    let nv = x.len();
    let (mut f0, _) = f(x);
    let mut af0 = sum_abs(&f0);
    let mut af00 = af0;
    let mut itct = 0usize;
    loop {
        if itct > 3 && af00 - af0 < crit * af0.max(1.0) && itct % 2 == 1 {
            return 4; // csolve.m draws a random direction here
        }
        let (fx, mut grad) = f(x);
        f0 = fx;
        let (fct, _, _) = lu(&grad);
        if rcond_from_lu(&fct, norm1(&grad)) < 1e-12 {
            for i in 0..nv {
                grad[(i, i)] += delta;
            }
        }
        let Some(mut dx0) = backslash(&grad, &f0) else { return 4 };
        dx0.iter_mut().for_each(|v| *v = -*v);
        let mut lambda: f64 = 1.0;
        let mut fmin = f0.clone();
        let mut xmin = x.clone();
        let mut afmin = af0;
        let dx_size = dx0.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut factor: f64 = 0.6;
        let mut shrink = true;
        let rc;
        loop {
            let xt: Vec<f64> = x.iter().zip(&dx0).map(|(a, d)| a + lambda * d).collect();
            let (ft, _) = f(&xt);
            // csolve.m: af=sum(abs(f)); a NaN residual makes af NaN, every test below false
            // ('good value found', rc=0, xmin unchanged)
            let af = sum_abs(&ft);
            if af < afmin {
                afmin = af;
                fmin = ft;
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
        *x = xmin;
        drop(fmin); // csolve.m: f0 = fmin, recomputed with the Jacobian below
        af00 = af0;
        af0 = afmin;
        if itct >= itmax {
            return 4;
        } else if af0 < crit {
            return 0;
        }
        let _ = rc;
    }
}

/// Port of `optimization/newton_solve.m` (`solve_algo = 6`, analytic Jacobian).
pub(super) fn newton_solve(f: StaticFn, x: &mut [f64], tolf: f64, maxit: usize) -> (bool, i32) {
    let (mut fvec, mut fjac) = f(x);
    if fvec.iter().any(|v| !v.is_finite()) {
        println!(
            "newton_solve: during the resolution of the non-linear system, the evaluation of some equation(s) resulted in a non-finite number"
        );
        return (true, 0);
    }
    if max_abs(&fvec) < tolf {
        return (false, -1);
    }
    for _ in 0..maxit {
        let Some(p) = backslash(&fjac, &fvec) else { return (true, 2) };
        x.iter_mut().zip(&p).for_each(|(a, d)| *a -= d);
        (fvec, fjac) = f(x);
        if max_abs(&fvec) < tolf {
            return (false, 1);
        }
    }
    (true, 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn system(y: &[f64]) -> (Vec<f64>, Mat) {
        // x^2 + y - 3 = 0, x - y^2 + 1 = 0 (root (1, 2)... check: 1+2-3=0, 1-4+1=-2 no)
        // use x^2 - 2 = 0, x*y - 1 = 0: root (sqrt 2, 1/sqrt 2)
        let (a, b) = (y[0], y[1]);
        (vec![a * a - 2.0, a * b - 1.0], Mat::from_row_slice(2, 2, &[2.0 * a, 0.0, b, a]))
    }

    #[test]
    fn other_solve_algos_converge() {
        for algo in [0u8, 3, 6, 7, 8, 13, 14] {
            let opts = SteadyOptions { solve_algo: algo, ..Default::default() };
            let out = dynare_solve_x(&system, &[1.0, 1.0], &opts).unwrap();
            assert!(!out.errorflag, "solve_algo {algo}: {out:?}");
            assert!(
                (out.x[0] - 2f64.sqrt()).abs() < 1e-5 && (out.x[1] - 0.5f64.sqrt()).abs() < 1e-5,
                "{algo}: {:?}",
                out.x
            );
        }
    }
}
