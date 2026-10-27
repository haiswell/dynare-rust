// SPDX-License-Identifier: GPL-3.0-or-later
//! DSGE-VAR (Del Negro and Schorfheide 2004): port of
//! `estimation/dsge_var_likelihood.m`, `var_sample_moments.m` and `check_dsge_var_model.m`.
//!
//! The VAR(p) with a constant on the observables gets the DSGE model's population moments
//! as a prior with weight `dsge_prior_weight` (a calibrated value `dsge_var = w`, possibly
//! `Inf`, or the estimated parameter `dsge_prior_weight`). The objective is minus the
//! marginal likelihood of the VAR (formula (A.2) of DS 2004) minus the log prior.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::collapsible_if
)] // index loops mirror the MATLAB code

use anyhow::{Result, bail};

use super::{Bounds, Evaluation, Objective};
use crate::dsge::{DsgeModel, EvalFailure};
use crate::estimation::special::gammaln;
use crate::kalman::lyapunov_symm;
use crate::linalg::{Mat, mul};
use crate::modfile::EstimatedKind;

/// `dataset_info.mYY/mYX/mXX` (var_sample_moments.m with a constant, `var_trend_order = 0`).
#[derive(Clone, Debug)]
pub struct VarMoments {
    pub m_yy: Mat,
    pub m_yx: Mat,
    pub m_xx: Mat,
    /// `dataset_.nobs`: all the periods of the dataset, the `nlags` initial ones included
    /// (as `dsge_var_likelihood.m` uses it).
    pub nobs: usize,
    pub nlags: usize,
    pub nvobs: usize,
}

impl VarMoments {
    /// `data`: nobs x nvobs (the dataset with its `nlags` initial conditions first), no NaN.
    pub fn new(data: &Mat, nlags: usize) -> Result<Self> {
        let (n, k) = (data.nrows(), data.ncols());
        if data.iter().any(|v| v.is_nan()) {
            bail!("Estimation::DsgeVarLikelihood: I cannot estimate a DSGE-VAR model with missing observations!");
        }
        if n <= nlags {
            bail!("DSGE-VAR: the sample has {n} periods, not more than dsge_varlag = {nlags}");
        }
        let t = n - nlags;
        let y = Mat::from_fn(t, k, |i, j| data[(nlags + i, j)]);
        let x = Mat::from_fn(t, k * nlags + 1, |i, c| {
            if c == k * nlags {
                1.0
            } else {
                let (lag, j) = (c / k + 1, c % k);
                data[(nlags + i - lag, j)]
            }
        });
        Ok(VarMoments {
            m_yy: mul(&y, true, &y, false),
            m_yx: mul(&y, true, &x, false),
            m_xx: mul(&x, true, &x, false),
            nobs: n,
            nlags,
            nvobs: k,
        })
    }
}

/// Weight of the DSGE prior.
#[derive(Clone, Copy, Debug)]
pub enum PriorWeight {
    /// `dsge_var = w` (calibrated; may be `Inf`).
    Calibrated(f64),
    /// `dsge_var` with `dsge_prior_weight` estimated: index of that parameter in `M_.params`.
    Estimated(usize),
}

/// DSGE-VAR posterior quantities at one parameter vector (the extra outputs of
/// dsge_var_likelihood.m: `PHI_tilde`, `SIGMA_u_tilde`, `iXX`, `prior`).
#[derive(Clone, Debug)]
pub struct DsgeVarEval {
    pub fval: f64,
    pub lik: f64,
    pub log_prior: f64,
    pub weight: f64,
    pub phi_tilde: Mat,
    pub sigma_u_tilde: Mat,
    pub ixx: Mat,
    pub phi_star: Mat,
    pub sigma_u_star: Mat,
    pub igxx_star: Mat,
}

/// dsge_var_likelihood.m as an [`Objective`].
pub struct DsgeVarPosterior<'a> {
    pub dsge: &'a DsgeModel,
    pub moments: &'a VarMoments,
    pub weight: PriorWeight,
    pub bounds: &'a Bounds,
    /// `options_.qz_criterium` / `options_.lyapunov_complex_threshold` for `lyapunov_symm`
    /// (dsge_var_likelihood.m:180).
    pub qz_criterium: f64,
    pub lyapunov_complex_threshold: f64,
}

/// MATLAB det (LU).
fn det(a: &Mat) -> f64 {
    let (lu, ipiv, _) = crate::linalg::lu(a);
    let mut d = 1.0;
    for i in 0..a.nrows() {
        d *= lu[(i, i)];
        if ipiv[i] as usize != i + 1 {
            d = -d;
        }
    }
    d
}

fn inv(a: &Mat) -> Mat {
    crate::linalg::solve(a, &Mat::identity(a.nrows(), a.nrows()))
        .unwrap_or_else(|| Mat::from_element(a.nrows(), a.ncols(), f64::INFINITY))
}

/// `ispd.m`: (positive definite, penalty = sum of -negative eigenvalues).
fn ispd(a: &Mat) -> (bool, f64) {
    if a.clone().cholesky().is_some() {
        return (true, 0.0);
    }
    let ev = a.clone().symmetric_eigenvalues();
    (false, ev.iter().filter(|&&l| l < 0.0).map(|l| -l).sum())
}

/// MATLAB `log` of a real number (NaN for a negative one, as the complex result fails
/// `dsge_var_likelihood`'s `imag(lik) ~= 0` test: info 46).
fn ln_checked(x: f64, complex: &mut bool) -> f64 {
    if x < 0.0 {
        *complex = true;
        (-x).ln()
    } else {
        x.ln()
    }
}

impl DsgeVarPosterior<'_> {
    /// `check_dsge_var_model.m`.
    pub fn check_model(dsge: &DsgeModel) -> Result<()> {
        if dsge.estimated.iter().any(|e| matches!(e.kind, EstimatedKind::MeStdErr(_) | EstimatedKind::MeCorr(..)))
            || dsge.base_h.iter().any(|&v| v != 0.0)
        {
            bail!("Estimation::DsgeVarLikelihood: Measurement errors are not allowed!");
        }
        let sig = &dsge.base_sigma_e;
        let offdiag = (0..sig.nrows()).any(|i| (0..i).any(|j| sig[(i, j)] != 0.0));
        if dsge.estimated.iter().any(|e| matches!(e.kind, EstimatedKind::Corr(..))) || (sig.nrows() > 1 && offdiag) {
            bail!(
                "Estimation::DsgeVarLikelihood: Structural innovations cannot be correlated using Dynare's interface! Introduce the correlations in the model block instead."
            );
        }
        if dsge.trend_coeffs.iter().any(Option::is_some) {
            bail!("Estimation::DsgeVarLikelihood: Linear trend is not yet implemented!");
        }
        Ok(())
    }

    /// One evaluation of dsge_var_likelihood.m; `Err` carries `info(1)` and `info(4)`.
    pub fn evaluate(&self, x: &[f64]) -> Result<DsgeVarEval, EvalFailure> {
        let b = self.bounds;
        if x.iter().zip(&b.lb).any(|(v, l)| v < l) {
            let p = x.iter().zip(&b.lb).filter(|(v, l)| v < l).map(|(v, l)| (l - v).powi(2)).sum();
            return Err(EvalFailure { info: 41, penalty: p });
        }
        if x.iter().zip(&b.ub).any(|(v, u)| v > u) {
            let p = x.iter().zip(&b.ub).filter(|(v, u)| v > u).map(|(v, u)| (v - u).powi(2)).sum();
            return Err(EvalFailure { info: 42, penalty: p });
        }
        let m = self.moments;
        let (nv, nl, nobs) = (m.nvobs, m.nlags, m.nobs as f64);
        let npar = (nv * nl + 1) as f64; // NumberOfParameters (with the constant)
        let weight = match self.weight {
            PriorWeight::Calibrated(w) => w,
            PriorWeight::Estimated(i) => self.dsge.set_all_parameters(x).0[i],
        };
        if weight < (npar + nv as f64) / nobs {
            return Err(EvalFailure { info: 51, penalty: (nobs * weight - (npar + nv as f64)).abs() });
        }
        let res = self.dsge.resolve(x)?;
        let idx = &self.dsge.ss_index;
        let constant: Vec<f64> = idx
            .mfys
            .iter()
            .map(|&v| if self.dsge.loglinear { res.steady_state[v].ln() } else { res.steady_state[v] })
            .collect();
        let rqr = mul(&mul(&res.r, false, &res.q, false), false, &res.r, true);
        let mut tmp0 = lyapunov_symm(&res.t, &rqr, self.qz_criterium, self.lyapunov_complex_threshold)
            .map_err(|_| EvalFailure { info: 50, penalty: 0.1 })?;
        let mf = &idx.mf;
        let cc = Mat::from_fn(nv, nv, |i, j| constant[i] * constant[j]);
        let sel = |p: &Mat| Mat::from_fn(nv, nv, |i, j| p[(mf[i], mf[j])]);
        let mut acov: Vec<Mat> = vec![sel(&tmp0) + &cc];
        for _ in 0..nl {
            tmp0 = mul(&res.t, false, &tmp0, false);
            acov.push(sel(&tmp0) + &cc);
        }
        let k = nv * nl + 1;
        let mut gyx = Mat::zeros(nv, k);
        for i in 0..nl {
            gyx.view_mut((0, i * nv), (nv, nv)).copy_from(&acov[i + 1]);
        }
        for r in 0..nv {
            gyx[(r, k - 1)] = constant[r];
        }
        let mut gxx = Mat::zeros(k, k);
        for i in 0..nl {
            for j in 0..nl {
                let blk = if i == j {
                    acov[0].clone()
                } else if j > i {
                    acov[j - i].clone() // kron(diag(ones,i), Gamma_i) above the diagonal
                } else {
                    acov[i - j].transpose()
                };
                gxx.view_mut((i * nv, j * nv), (nv, nv)).copy_from(&blk);
            }
        }
        for i in 0..nl {
            for r in 0..nv {
                gxx[(i * nv + r, k - 1)] = constant[r];
                gxx[(k - 1, i * nv + r)] = constant[r];
            }
        }
        gxx[(k - 1, k - 1)] = 1.0;
        let gyy = &acov[0];
        let igxx = inv(&gxx);
        let phi_star = mul(&igxx, false, &gyx, true);
        let sigma_u_star = gyy - mul(&gyx, false, &phi_star, false);
        let (pd, pen) = ispd(&sigma_u_star);
        if !pd {
            return Err(EvalFailure { info: 53, penalty: pen });
        }
        let mut complex = false;
        let (lik, sigma_u_tilde, phi_tilde, ixx) = if weight.is_finite() {
            let tn = weight * nobs;
            let t0 = acov[0].scale(tn) + &m.m_yy;
            let t1 = gyx.scale(tn) + &m.m_yx;
            let a2 = gxx.scale(tn) + &m.m_xx;
            let t2 = inv(&a2);
            let s = &t0 - mul(&mul(&t1, false, &t2, false), false, &t1, true);
            let (pd, pen) = ispd(&s);
            if !pd {
                return Err(EvalFailure { info: 52, penalty: pen });
            }
            let s = s / (nobs * (1.0 + weight));
            let phi_tilde = mul(&t2, false, &t1, true);
            let prodlng1: f64 = (1..=nv).map(|i| gammaln(0.5 * ((1.0 + weight) * nobs - npar + 1.0 - i as f64))).sum();
            let prodlng2: f64 = (1..=nv).map(|i| gammaln(0.5 * (weight * nobs - npar + 1.0 - i as f64))).sum();
            let nvf = nv as f64;
            let lik = 0.5 * nvf * ln_checked(det(&a2), &mut complex)
                + 0.5 * ((weight + 1.0) * nobs - npar) * ln_checked(det(&s.scale((weight + 1.0) * nobs)), &mut complex)
                - 0.5 * nvf * ln_checked(det(&gxx.scale(tn)), &mut complex)
                - 0.5 * (tn - npar) * ln_checked(det(&sigma_u_star.scale(tn)), &mut complex)
                + 0.5 * nvf * nobs * (2.0 * std::f64::consts::PI).ln()
                - 0.5 * 2f64.ln() * nvf * ((weight + 1.0) * nobs - npar)
                + 0.5 * 2f64.ln() * nvf * (tn - npar)
                - prodlng1
                + prodlng2;
            (lik, s, phi_tilde, t2)
        } else {
            let isu = inv(&sigma_u_star);
            let myx_phi = mul(&m.m_yx, false, &phi_star, false);
            let inner = &m.m_yy - myx_phi.transpose() - &myx_phi
                + mul(&mul(&phi_star, true, &m.m_xx, false), false, &phi_star, false);
            let tr = mul(&isu, false, &inner, false).trace() / nobs;
            let lik = 0.5
                * nobs
                * (ln_checked(det(&sigma_u_star), &mut complex) + nv as f64 * (2.0 * std::f64::consts::PI).ln() + tr);
            (lik, sigma_u_star.clone(), phi_star.clone(), igxx.clone())
        };
        if lik.is_infinite() {
            return Err(EvalFailure { info: 50, penalty: 0.1 });
        }
        if lik.is_nan() {
            return Err(EvalFailure { info: 45, penalty: 0.1 });
        }
        if complex {
            return Err(EvalFailure { info: 46, penalty: 0.1 });
        }
        let log_prior = self.dsge.log_prior(x);
        if log_prior.is_infinite() {
            return Err(EvalFailure { info: 40, penalty: 0.1 });
        }
        if log_prior.is_nan() {
            return Err(EvalFailure { info: 47, penalty: 0.1 });
        }
        Ok(DsgeVarEval {
            fval: lik - log_prior,
            lik,
            log_prior,
            weight,
            phi_tilde,
            sigma_u_tilde,
            ixx,
            phi_star,
            sigma_u_star,
            igxx_star: igxx,
        })
    }
}

impl Objective for DsgeVarPosterior<'_> {
    fn dim(&self) -> usize {
        self.dsge.n_estimated()
    }

    fn eval(&self, x: &[f64]) -> (f64, i32) {
        let r = match self.evaluate(x) {
            Ok(e) => (e.fval, 0),
            Err(f) => (f64::INFINITY, f.info),
        };
        crate::progress::eval(r.0);
        r
    }

    fn eval_full(&self, x: &[f64], _contributions: bool) -> Evaluation {
        let r = self.evaluate(x);
        crate::progress::eval(r.as_ref().map_or(f64::INFINITY, |e| e.fval));
        match r {
            Ok(e) => Evaluation { fval: e.fval, info: 0, info4: 0.0, contributions: None },
            Err(f) => Evaluation { fval: f64::INFINITY, info: f.info, info4: f.penalty, contributions: None },
        }
    }
}
