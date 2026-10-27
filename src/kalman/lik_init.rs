// SPDX-License-Identifier: GPL-3.0-or-later
//! [stream X] Initial state covariance for `lik_init` = 2, 4 and 5 (`dsge_likelihood.m`,
//! `DsgeSmoother.m`). All three run the standard (non-diffuse) filter from `Pstar` with
//! `Pinf = 0`, like `lik_init = 1`:
//!
//! - 2: `Pstar = options_.Harvey_scale_factor * I` (10; with a zero factor the likelihood
//!   starts from `R Q R'`, the conditional likelihood);
//! - 4: the steady state of the Riccati equation, `kalman_steady_state(T', RQR', Z', H)`
//!   (prediction-error form `P = T (P - P Z' (Z P Z' + H)^-1 Z P) T' + RQR'`, solved here by
//!   iterating the filter recursion to convergence; `dsge_likelihood.m` switches to
//!   `lik_init = 1` when it cannot be solved, `DsgeSmoother.m` errors);
//! - 5: the Lyapunov covariance on the variables that do not load on unit roots, and
//!   `Harvey_scale_factor` on the diagonal elsewhere.

use anyhow::{Result, anyhow, bail};

use super::init::lyapunov_solver;
use crate::linalg::{Mat, mul, solve};

/// `options_.Harvey_scale_factor`.
pub const HARVEY_SCALE_FACTOR: f64 = 10.0;

/// Whether the filter is the standard one started from a `Pstar` built here.
pub fn handles(lik_init: u8) -> bool {
    matches!(lik_init, 2 | 4 | 5)
}

/// Initial `Pstar` for `lik_init` 2, 4 or 5. `mf`: observed rows of T; `h`: `M_.H`
/// (`None` or zero = no measurement error); `smoother`: `DsgeSmoother.m`'s variant (for 5:
/// `Harvey_scale_factor * eye(np)` as the base, where `dsge_likelihood.m` uses
/// `eye(nunit)`).
#[allow(clippy::too_many_arguments)]
pub fn initial_pstar(
    lik_init: u8,
    t: &Mat,
    r: &Mat,
    q: &Mat,
    h: Option<&Mat>,
    mf: &[usize],
    qz_criterium: f64,
    lyapunov_complex_threshold: f64,
    smoother: bool,
) -> Result<Mat> {
    let mm = t.nrows();
    let rqr = mul(&mul(r, false, q, false), false, r, true);
    match lik_init {
        2 => {
            if HARVEY_SCALE_FACTOR == 0.0 && !smoother {
                Ok(rqr)
            } else {
                Ok(Mat::identity(mm, mm) * HARVEY_SCALE_FACTOR)
            }
        }
        4 => match kalman_steady_state(t, &rqr, mf, h) {
            Ok(p) => Ok(p),
            // DsgeSmoother.m calls kalman_steady_state without a try/catch
            Err(e) if smoother => Err(e),
            Err(e) => {
                println!("{e}");
                println!("dsge_likelihood:: I am not able to solve the Riccati equation, so I switch to lik_init=1!");
                lyapunov_solver(t, r, q, qz_criterium, lyapunov_complex_threshold)
            }
        },
        5 => {
            let (vals, vecs) = eig_with_vectors(t)?;
            let unit: Vec<usize> = (0..mm).filter(|&k| (vals[k].0.hypot(vals[k].1) - 1.0).abs() < 1e-7).collect();
            let nunit = unit.len();
            // sum(abs(V), 2) over the unit-root eigenvectors
            let stable: Vec<usize> = (0..mm)
                .filter(|&i| unit.iter().map(|&k| vecs[k][i].0.hypot(vecs[k][i].1)).sum::<f64>() < 1e-5)
                .collect();
            let size = if smoother { mm } else { nunit.max(stable.last().map_or(0, |&s| s + 1)) };
            if size != mm {
                bail!(
                    "lik_init=5: Pstar has {size} rows for {mm} states (dsge_likelihood.m's eye(nunit) with {nunit} unit roots); \
                     MATLAB fails on this model too"
                );
            }
            let base = if smoother { mm } else { nunit };
            let mut p = Mat::zeros(mm, mm);
            for i in 0..base {
                p[(i, i)] = HARVEY_SCALE_FACTOR;
            }
            if !stable.is_empty() {
                let ts = Mat::from_fn(stable.len(), stable.len(), |i, j| t[(stable[i], stable[j])]);
                let rs = Mat::from_fn(stable.len(), r.ncols(), |i, j| r[(stable[i], j)]);
                let ps = lyapunov_solver(&ts, &rs, q, qz_criterium, lyapunov_complex_threshold)?;
                for (a, &i) in stable.iter().enumerate() {
                    for (b, &j) in stable.iter().enumerate() {
                        p[(i, j)] = ps[(a, b)];
                    }
                }
            }
            Ok(p)
        }
        k => bail!("lik_init = {k}: not handled here"),
    }
}

/// Eigenvalues `(re, im)` and right eigenvectors (one `Vec<(re, im)>` per eigenvalue,
/// unit 2-norm as MATLAB's `eig`) of a real matrix (LAPACK dgeev).
pub fn eig_with_vectors(a: &Mat) -> Result<(Vec<(f64, f64)>, Vec<Vec<(f64, f64)>>)> {
    use crate::linalg::lapack;
    use std::os::raw::{c_char, c_int};
    let n = a.nrows();
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    let ni = n as c_int;
    let mut f = a.clone();
    let (mut wr, mut wi) = (vec![0.0; n], vec![0.0; n]);
    let mut vr = vec![0.0; n * n];
    let lwork = (8 * n + 64) as c_int;
    let mut work = vec![0.0; lwork as usize];
    let mut info: c_int = 0;
    let one: c_int = 1;
    let mut dummy = [0.0f64; 1];
    unsafe {
        lapack::dgeev_(
            &(b'N' as c_char),
            &(b'V' as c_char),
            &ni,
            f.as_mut_ptr(),
            &ni,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            dummy.as_mut_ptr(),
            &one,
            vr.as_mut_ptr(),
            &ni,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    if info != 0 {
        bail!("eig: dgeev failed (info {info})");
    }
    let mut vecs = Vec::with_capacity(n);
    let mut k = 0;
    while k < n {
        let col = |j: usize| &vr[j * n..(j + 1) * n];
        if wi[k] == 0.0 {
            vecs.push(col(k).iter().map(|&x| (x, 0.0)).collect());
            k += 1;
        } else {
            let (re, im) = (col(k), col(k + 1));
            vecs.push(re.iter().zip(im).map(|(&x, &y)| (x, y)).collect());
            vecs.push(re.iter().zip(im).map(|(&x, &y)| (x, -y)).collect());
            k += 2;
        }
    }
    Ok((wr.into_iter().zip(wi).collect(), vecs))
}

/// Steady state of the prediction-error Riccati equation
/// `P = T (P - P Z' (Z P Z' + H)^-1 Z P) T' + RQR'` (`kalman_steady_state` MEX, SLICOT SB02OD
/// on the dual problem), by iterating the Riccati map from `RQR'` until the relative change
/// is below 1e-15; `Err` when it does not converge.
pub fn kalman_steady_state(t: &Mat, rqr: &Mat, mf: &[usize], h: Option<&Mat>) -> Result<Mat> {
    let mm = t.nrows();
    let pp = mf.len();
    let h = match h {
        Some(h) if h.iter().any(|&v| v != 0.0) => h.clone(),
        _ => Mat::zeros(pp, pp),
    };
    // Plain iteration of the filter recursion, from RQR'.
    let mut p = rqr.clone();
    let tol = 1e-15;
    for _ in 0..200_000 {
        let zpz = Mat::from_fn(pp, pp, |i, j| p[(mf[i], mf[j])] + h[(i, j)]);
        let pz = Mat::from_fn(mm, pp, |i, j| p[(i, mf[j])]);
        // K = P Z' F^-1
        let k = solve(&zpz, &pz.transpose())
            .ok_or_else(|| anyhow!("kalman_steady_state: singular Z P Z' + H"))?
            .transpose();
        let pupd = &p - mul(&k, false, &pz, true);
        let mut pn = mul(&mul(t, false, &pupd, false), false, t, true) + rqr;
        pn = (&pn + pn.transpose()) * 0.5;
        if pn.iter().any(|v| !v.is_finite()) {
            bail!("kalman_steady_state: the Riccati iteration diverged");
        }
        let scale = pn.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        let diff = pn.iter().zip(p.iter()).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        p = pn;
        if diff <= tol * scale {
            return Ok(p);
        }
    }
    bail!("kalman_steady_state: no convergence of the Riccati iteration")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_riccati() {
        // x' = a x + e (var s2), y = x + u (var h): P = a^2 (P - P^2/(P+h)) + s2
        let (a, s2, h) = (0.9, 1.0, 0.5);
        let p = kalman_steady_state(
            &Mat::from_element(1, 1, a),
            &Mat::from_element(1, 1, s2),
            &[0],
            Some(&Mat::from_element(1, 1, h)),
        )
        .unwrap()[(0, 0)];
        let rhs = a * a * (p - p * p / (p + h)) + s2;
        assert!((p - rhs).abs() < 1e-12, "{p} {rhs}");
        // no measurement error: P = s2
        let p0 =
            kalman_steady_state(&Mat::from_element(1, 1, a), &Mat::from_element(1, 1, s2), &[0], None).unwrap()[(0, 0)];
        assert!((p0 - s2).abs() < 1e-14);
    }

    #[test]
    fn eigenvectors() {
        let t = Mat::from_row_slice(2, 2, &[1.0, 0.5, 0.0, 0.5]);
        let (vals, vecs) = eig_with_vectors(&t).unwrap();
        for (k, (re, _)) in vals.iter().enumerate() {
            let v: Vec<f64> = vecs[k].iter().map(|c| c.0).collect();
            let tv = [t[(0, 0)] * v[0] + t[(0, 1)] * v[1], t[(1, 0)] * v[0] + t[(1, 1)] * v[1]];
            assert!((tv[0] - re * v[0]).abs() < 1e-14 && (tv[1] - re * v[1]).abs() < 1e-14);
        }
    }
}
