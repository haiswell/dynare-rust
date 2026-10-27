// SPDX-License-Identifier: GPL-3.0-or-later
//! Identification strength (Ratto and Iskrev 2011): the asymptotic information matrix from the
//! analytic score of the Kalman filter on simulated data (the `try` branch of
//! `+identification/analysis.m`: `stoch_simul` + `dsge_likelihood` with
//! `analytic_Hessian = 'asymptotic'`, computed by [`super::Context::asymptotic_hessian_on`]
//! through `crate::estimation::analytic`), or the moments information matrix from simulated
//! moment uncertainty (the `catch` branch, `simulated_moment_uncertainty.m`).

use rand::Rng;
use rand_distr::StandardNormal;

use crate::linalg::{Mat, chol_upper, mul};
use crate::solver::DecisionRule;

use super::checks::{CheckResult, checks, vnorm};
use super::la::{inv, rank, solve};

/// `simult.m` at order 1 from `y0` (levels, declaration order: `oo_.dr.ys` or
/// `M_.endo_histval`), the initial value removed: `periods` draws of `e ~ N(0, Sigma_e)`
/// (`randn * chol(Sigma_e)`); returns n_endo x periods (declaration order, levels).
pub fn simulate<R: Rng + ?Sized>(dr: &DecisionRule, sigma_e: &Mat, y0: &[f64], periods: usize, rng: &mut R) -> Mat {
    let n = dr.ghx.nrows();
    let ne = sigma_e.nrows();
    let chol = chol_upper(sigma_e).unwrap_or_else(|| {
        // singular covariance (zero variances): use the square roots of the diagonal
        Mat::from_fn(ne, ne, |i, j| if i == j { sigma_e[(i, i)].max(0.0).sqrt() } else { 0.0 })
    });
    let states: Vec<usize> = (dr.nstatic..dr.nstatic + dr.nspred).collect();
    let mut y = Mat::zeros(n, periods);
    // deviations, DR order
    let mut prev: Vec<f64> = (0..n).map(|i| y0[dr.order_var[i]] - dr.ys[dr.order_var[i]]).collect();
    let mut e = vec![0.0; ne];
    let mut z = vec![0.0; ne];
    for t in 0..periods {
        for v in z.iter_mut() {
            *v = rng.sample(StandardNormal);
        }
        for j in 0..ne {
            e[j] = (0..=j).map(|i| z[i] * chol[(i, j)]).sum();
        }
        let mut cur = vec![0.0; n];
        for (i, c) in cur.iter_mut().enumerate() {
            let mut s = 0.0;
            for (k, &st) in states.iter().enumerate() {
                s += dr.ghx[(i, k)] * prev[st];
            }
            for j in 0..ne {
                s += dr.ghu[(i, j)] * e[j];
            }
            *c = s;
        }
        for i in 0..n {
            y[(dr.order_var[i], t)] = cur[i] + dr.ys[dr.order_var[i]];
        }
        prev = cur;
    }
    y
}

/// The identification-strength part of `identification.analysis` (`ide_hess` and the
/// normalized sensitivities).
#[derive(Clone, Debug)]
pub struct Strength {
    pub ahess: Mat,
    pub check: CheckResult,
    pub ide_strength: Vec<f64>,
    pub ide_strength_prior: Vec<f64>,
    pub delta_m: Vec<f64>,
    pub delta_m_prior: Vec<f64>,
    pub sensitivity_zero_pos: Vec<usize>,
    pub identified: Vec<usize>,
    /// 1: asymptotic information matrix; 0: moments information matrix.
    pub flag_score: bool,
    pub si_dmoments_norm: Vec<f64>,
    pub si_dreducedform_norm: Vec<f64>,
    pub si_ddynamic_norm: Vec<f64>,
    /// Rows of si_dREDUCEDFORM / si_dDYNAMIC kept (`iy`).
    pub keep_reducedform: Vec<usize>,
    pub keep_dynamic: Vec<usize>,
}

fn sel_cols(m: &Mat, c: &[usize]) -> Mat {
    Mat::from_fn(m.nrows(), c.len(), |i, j| m[(i, c[j])])
}

fn norm_rows(si: &Mat, d: &[f64], scale: &[f64]) -> Vec<f64> {
    let q = Mat::from_fn(si.nrows(), si.ncols(), |i, j| si[(i, j)] / d[i].sqrt());
    let base: Vec<f64> = if q.nrows() == 1 { q.row(0).iter().map(|v| v.abs()).collect() } else { vnorm(&q) };
    base.iter().zip(scale).map(|(a, b)| a * b).collect()
}

/// `diag(S * (H \ S'))`.
fn quad_diag(s: &Mat, h: &Mat) -> Vec<f64> {
    let x = solve(h, &s.transpose());
    (0..s.nrows()).map(|r| (0..s.ncols()).map(|c| s[(r, c)] * x[(c, r)]).sum()).collect()
}

/// Strength measures from an information matrix `ahess` (`flag_score`: from the asymptotic
/// Hessian; otherwise `ahess` is the moments information matrix MIM).
#[allow(clippy::too_many_arguments)]
pub fn strength_measures(
    ahess: Mat,
    flag_score: bool,
    params: &[f64],
    prior_sd: &[f64],
    si_dmoments: &Mat,
    cmm_diag: Option<Vec<f64>>,
    si_dreducedform: &Mat,
    si_ddynamic: &Mat,
    n_shock: usize,
    tol_rank: Option<f64>,
    tol_sv: f64,
) -> Strength {
    let tot = params.len();
    let dm: Vec<f64> = (0..tot).map(|i| ahess[(i, i)].sqrt()).collect();
    let iflag = dm.iter().any(|d| d * d == 0.0);
    let tilda = Mat::from_fn(tot, tot, |i, j| ahess[(i, j)] / (dm[i] * dm[j]));
    let check = if iflag || rank(&ahess, None) > rank(&tilda, None) {
        checks(&ahess, 0, tol_rank, tol_sv, tot)
    } else {
        checks(&tilda, 0, tol_rank, tol_sv, tot)
    };
    let indok: Vec<usize> = (0..tot).filter(|&j| check.indno.iter().all(|r| !r[j])).collect();
    let mut uncert = vec![f64::NAN; tot];
    if !indok.is_empty() {
        if flag_score {
            let h = Mat::from_fn(indok.len(), indok.len(), |i, j| ahess[(indok[i], indok[j])]);
            let hi = inv(&h);
            for (a, &i) in indok.iter().enumerate() {
                uncert[i] = hi[(a, a)].sqrt();
            }
        } else {
            let h = Mat::from_fn(indok.len(), indok.len(), |i, j| tilda[(indok[i], indok[j])]);
            let hi = inv(&h);
            for (a, &i) in indok.iter().enumerate() {
                uncert[i] = hi[(a, a)].sqrt() / dm[i];
            }
        }
    }
    let ind1: Vec<usize> = (0..tot).filter(|&j| check.ind0[j]).collect();
    let h11 = Mat::from_fn(ind1.len(), ind1.len(), |i, j| ahess[(ind1[i], ind1[j])]);
    let cmm_d = cmm_diag.unwrap_or_else(|| quad_diag(&sel_cols(si_dmoments, &ind1), &h11));
    let diag_chh = quad_diag(&sel_cols(si_dreducedform, &ind1), &h11);
    let ind1m: Vec<usize> = ind1.iter().copied().filter(|&i| i >= n_shock).collect();
    let hm = Mat::from_fn(ind1m.len(), ind1m.len(), |i, j| ahess[(ind1m[i], ind1m[j])]);
    let cols_m: Vec<usize> = ind1m.iter().map(|i| i - n_shock).collect();
    let diag_cdyn = quad_diag(&sel_cols(si_ddynamic, &cols_m), &hm);

    let mut ide_strength = vec![f64::NAN; tot];
    let mut ide_strength_prior = vec![f64::NAN; tot];
    for &i in &indok {
        ide_strength[i] = 1.0 / (uncert[i] / params[i].abs());
        ide_strength_prior[i] = 1.0 / (uncert[i] / prior_sd[i]);
    }
    let sensitivity_zero_pos: Vec<usize> = (0..tot).filter(|&i| dm[i].is_infinite()).collect();
    let delta_m_prior: Vec<f64> = (0..tot).map(|i| dm[i] * prior_sd[i].abs()).collect();
    let delta_m: Vec<f64> = (0..tot).map(|i| dm[i] * params[i].abs()).collect();
    let si_dmoments_norm = norm_rows(si_dmoments, &cmm_d, prior_sd);
    let keep_reducedform: Vec<usize> = (0..diag_chh.len()).filter(|&i| diag_chh[i] != 0.0).collect();
    let si_dreducedform_norm = if keep_reducedform.is_empty() {
        vec![]
    } else {
        let s = Mat::from_fn(keep_reducedform.len(), tot, |i, j| si_dreducedform[(keep_reducedform[i], j)]);
        let d: Vec<f64> = keep_reducedform.iter().map(|&i| diag_chh[i]).collect();
        norm_rows(&s, &d, prior_sd)
    };
    let keep_dynamic: Vec<usize> = (0..diag_cdyn.len()).filter(|&i| diag_cdyn[i] != 0.0).collect();
    let si_ddynamic_norm = if keep_dynamic.is_empty() {
        vec![]
    } else {
        let nm = si_ddynamic.ncols();
        let s = Mat::from_fn(keep_dynamic.len(), nm, |i, j| si_ddynamic[(keep_dynamic[i], j)]);
        let d: Vec<f64> = keep_dynamic.iter().map(|&i| diag_cdyn[i]).collect();
        norm_rows(&s, &d, &prior_sd[n_shock..])
    };
    Strength {
        ahess,
        check,
        ide_strength,
        ide_strength_prior,
        delta_m,
        delta_m_prior,
        sensitivity_zero_pos,
        identified: indok,
        flag_score,
        si_dmoments_norm,
        si_dreducedform_norm,
        si_ddynamic_norm,
        keep_reducedform,
        keep_dynamic,
    }
}

/// `simulated_moment_uncertainty(ind_dMOMENTS, periods, replic)`: covariance of the selected
/// simulated moments `[mean; vech(var); vec(autocorr_i .* sd sd')]` over `replic` simulations
/// of `periods` periods from `y0` (the first `drop` dropped, `disp_moments.m`), with the
/// autocovariances up to lag `ar` (`options_.ar`).
#[allow(clippy::too_many_arguments)]
pub fn simulated_moment_uncertainty<R: Rng + ?Sized>(
    dr: &DecisionRule,
    sigma_e: &Mat,
    y0: &[f64],
    varobs: &[usize],
    ind: &[usize],
    periods: usize,
    drop: usize,
    replic: usize,
    ar: usize,
    rng: &mut R,
) -> Mat {
    let nv = varobs.len();
    let mut mm = Mat::zeros(ind.len(), replic);
    for rep in 0..replic {
        let ys = simulate(dr, sigma_e, y0, periods, rng);

        let nt = periods - drop;
        let y = Mat::from_fn(nt, nv, |t, j| ys[(varobs[j], drop + t)]);
        let mean: Vec<f64> = (0..nv).map(|j| y.column(j).sum() / nt as f64).collect();
        let yc = Mat::from_fn(nt, nv, |t, j| y[(t, j)] - mean[j]);
        let var = mul(&yc, true, &yc, false) / nt as f64;
        let sd: Vec<f64> = (0..nv).map(|j| var[(j, j)].sqrt()).collect();
        let mut dum = mean.clone();
        dum.extend(super::moments::vech(&var));
        let std_of = |m: &Mat| -> Vec<f64> {
            let n = m.nrows() as f64;
            (0..m.ncols())
                .map(|j| {
                    let mu = m.column(j).sum() / n;
                    (m.column(j).iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / (n - 1.0)).sqrt()
                })
                .collect()
        };
        for i in 1..=ar {
            let a = yc.rows(ar, nt - ar).into_owned();
            let b = yc.rows(ar - i, nt - ar).into_owned();
            let (sa, sb) = (std_of(&a), std_of(&b));
            let c = mul(&a, true, &b, false);
            for q in 0..nv {
                for p in 0..nv {
                    let ac = c[(p, q)] / ((nt - ar) as f64 * sa[p] * sb[q]);
                    dum.push(ac * sd[p] * sd[q]);
                }
            }
        }
        for (r, &k) in ind.iter().enumerate() {
            mm[(r, rep)] = dum[k];
        }
    }
    // cov(mm')
    let nr = ind.len();
    let mu: Vec<f64> = (0..nr).map(|r| mm.row(r).sum() / replic as f64).collect();
    let c = Mat::from_fn(nr, replic, |r, j| mm[(r, j)] - mu[r]);
    mul(&c, false, &c, true) / (replic as f64 - 1.0)
}

/// The moments information matrix of the `catch` branch:
/// `MIM = (si'V) D^{-1} (W'si)` with the eigen-decomposition of the moments correlation.
pub fn moments_information_matrix(si_dmoments: &Mat, cmm: &Mat, tol_deriv: f64) -> Mat {
    let n = cmm.nrows();
    let sd: Vec<f64> = (0..n).map(|i| cmm[(i, i)].sqrt()).collect();
    let cc = Mat::from_fn(n, n, |i, j| cmm[(i, j)] / (sd[i] * sd[j]));
    let (w, v) = crate::linalg::sym_eig(&cc);
    let id: Vec<usize> = (0..n).filter(|&i| w[i] > tol_deriv).collect();
    let si = Mat::from_fn(si_dmoments.nrows(), si_dmoments.ncols(), |i, j| si_dmoments[(i, j)] / sd[i]);
    let vid = Mat::from_fn(n, id.len(), |i, j| v[(i, id[j])]);
    let left = mul(&si, true, &vid, false);
    let mut right = mul(&vid, true, &si, false);
    for (r, &k) in id.iter().enumerate() {
        for c in 0..right.ncols() {
            right[(r, c)] /= w[k];
        }
    }
    mul(&left, false, &right, false)
}
