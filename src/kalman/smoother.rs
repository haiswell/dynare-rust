// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of the filtering/smoothing part of kalman/DsgeSmoother.m (no OccBin, no
//! `smoother_redux`, `nk = 1`, no filter covariance / decomposition):
//! `missing_DiffuseKalmanSmootherH1_Z.m` (multivariate) with the switch to
//! `missing_DiffuseKalmanSmootherH3_Z.m` (univariate) when the multivariate smoother hits a
//! singular forecast-error variance.
//!
//! The smoother is not on the MCMC hot path; it uses plain nalgebra arithmetic.

use super::{HMatrix, KalmanOptions, MissingData};
use crate::linalg::{Mat, Vector, mul};

/// Smoother results (state space of the T passed in; Dynare's DsgeSmoother runs it on the
/// `smoother_var_list` state space, i.e. all endogenous variables in DR order).
#[derive(Clone, Debug)]
pub struct SmootherOutput {
    /// smoothed states E[alpha_t | Y_n], mm x nobs
    pub alphahat: Mat,
    /// smoothed shocks, rr x nobs
    pub etahat: Mat,
    /// smoothed measurement errors Y - Z*alphahat, pp x nobs (NaN where data are missing)
    pub epsilonhat: Mat,
    /// updated states E[alpha_t | Y_t], mm x nobs (`atilde` of H1, `a` of H3)
    pub ahat: Mat,
    /// one-step-ahead predictions E[alpha_{t+1} | Y_t], mm x nobs: column t is Dynare's
    /// `aK(1,:,t+1)` (1-based t), i.e. what `oo_.FilteredVariables` stores for period t.
    pub filtered: Mat,
    /// smoothed initial state (period 0)
    pub alphahat0: Vector,
    /// smoothed state covariances V_t (if requested)
    pub state_uncertainty: Option<Vec<Mat>>,
    /// `state_uncertainty0`: covariance of the smoothed initial state (period 0), when
    /// `state_uncertainty` was requested (multivariate smoother only).
    pub state_uncertainty0: Option<Mat>,
    /// number of diffuse periods
    pub d: usize,
    /// true if the univariate smoother (H3) produced the results
    pub univariate: bool,
    /// `filter_error_flag` of the multivariate smoother when it failed (420, 421, 422), else 0
    pub info: i32,
    /// k-step-ahead objects when requested ([`smoother_request`]).
    pub step_ahead: Option<StepAhead>,
}

/// DsgeSmoother.m's choice of smoother and fallback, for lik_init 1 (pass the Lyapunov Pstar,
/// `pinf = None`) and 3 (Pstar/Pinf of `compute_pinf_pstar(mf, T, R, Q, qz, &[])`, the
/// 5-argument call DsgeSmoother makes).
///
/// `y`: demeaned data (pp x nobs, NaN = missing); `mf`: 0-based observed rows of T.
/// A non-diagonal H with the univariate smoother augments the state vector (see `augment`).
#[allow(clippy::too_many_arguments)]
pub fn smoother(
    y: &Mat,
    missing: &MissingData,
    mf: &[usize],
    t: &Mat,
    r: &Mat,
    q: &Mat,
    h: Option<&Mat>,
    pstar: &Mat,
    pinf: Option<&Mat>,
    opts: &KalmanOptions,
    state_uncertainty: bool,
) -> anyhow::Result<SmootherOutput> {
    smoother_heteroskedastic(y, missing, mf, t, r, q, h, pstar, pinf, opts, state_uncertainty, None)
}

/// [`smoother`] with `heteroskedastic_filter`'s `Qvec` (k = 0..=nobs, see
/// [`super::hetero`]): the transitions after period t (0-based) use `R*Qvec(:,:,t+1)*R'`, the
/// smoothed shocks of period t `Qvec(:,:,t)*R'`, as missing_DiffuseKalmanSmootherH1_Z.m /
/// H3_Z.m do.
#[allow(clippy::too_many_arguments)]
pub fn smoother_heteroskedastic(
    y: &Mat,
    missing: &MissingData,
    mf: &[usize],
    t: &Mat,
    r: &Mat,
    q: &Mat,
    h: Option<&Mat>,
    pstar: &Mat,
    pinf: Option<&Mat>,
    opts: &KalmanOptions,
    state_uncertainty: bool,
    qvec: Option<&[Mat]>,
) -> anyhow::Result<SmootherOutput> {
    smoother_request(y, missing, mf, t, r, q, h, pstar, pinf, opts, state_uncertainty, qvec, None)
}

/// `filter_step_ahead` / `filter_covariance` / `filter_decomposition` of
/// DsgeSmoother.m: what the smoother computes besides the smoothed states.
#[derive(Clone, Debug, Default)]
pub struct SmootherRequest {
    /// `options_.nk = max(filter_step_ahead)`.
    pub nk: usize,
    pub filter_covariance: bool,
    pub decomposition: bool,
    /// `filter_initial_state`: `a_initial = T*a` (DsgeSmoother.m), None = 0.
    pub a_initial: Option<Vec<f64>>,
}

/// `aK`, `PK`, `decomp` of missing_DiffuseKalmanSmootherH1_Z.m / H3_Z.m. Index
/// `c` of the time axis (length nobs+nk) is MATLAB's `c+1`.
#[derive(Clone, Debug, Default)]
pub struct StepAhead {
    /// `aK(j,:,:)`, j = 1..nk: mm x (nobs+nk) each.
    pub ak: Vec<Mat>,
    /// `PK(j,:,:,c)` (mm x mm, zero in the diffuse periods), with `filter_covariance`.
    pub pk: Option<Vec<Vec<Mat>>>,
    /// `decomp(j,:,:,c)` (mm x rr), with `filter_decomposition`.
    pub decomp: Option<Vec<Vec<Mat>>>,
}

/// [`smoother_heteroskedastic`] that also returns the k-step-ahead objects of
/// `req` (in `SmootherOutput::step_ahead`).
#[allow(clippy::too_many_arguments)]
pub fn smoother_request(
    y: &Mat,
    missing: &MissingData,
    mf: &[usize],
    t: &Mat,
    r: &Mat,
    q: &Mat,
    h: Option<&Mat>,
    pstar: &Mat,
    pinf: Option<&Mat>,
    opts: &KalmanOptions,
    state_uncertainty: bool,
    qvec: Option<&[Mat]>,
    req: Option<&SmootherRequest>,
) -> anyhow::Result<SmootherOutput> {
    let pp = y.nrows();
    let hm = HMatrix::from_option(h, pp);
    let mut kalman_algo = opts.kalman_algo;
    let pinf = match opts.lik_init {
        1 | 2 | 4 | 5 => {
            // [stream X] lik_init 2/4/5: Pstar from kalman::lik_init
            if kalman_algo != 2 {
                kalman_algo = 1;
            }
            None
        }
        3 => {
            if kalman_algo != 4 {
                kalman_algo = 3;
            }
            pinf
        }
        other => anyhow::bail!("smoother: lik_init = {other} not supported"),
    };
    let data_index: Vec<Vec<usize>> =
        if missing.any_missing { missing.aindex.clone() } else { vec![(0..pp).collect(); y.ncols()] };
    let hetero = |r: &Mat, qv: &[Mat]| -> (Vec<Mat>, Vec<Mat>) {
        (
            qv.iter().map(|qk| mul(&mul(r, false, qk, false), false, r, true)).collect(),
            qv.iter().map(|qk| mul(qk, false, r, true)).collect(),
        )
    };
    let (qq_v, qrt_v) = match qvec {
        Some(qv) => {
            let (a, b) = hetero(r, qv);
            (Some(a), Some(b))
        }
        None => (None, None),
    };
    let sys = Sys {
        t,
        r,
        q,
        mf,
        qq: mul(&mul(r, false, q, false), false, r, true),
        qrt: mul(q, false, r, true),
        qq_v,
        qrt_v,
    };
    let mut info = 0;
    if kalman_algo == 1 || kalman_algo == 3 {
        match smoother_h1(&sys, &hm, pstar, pinf, y, &data_index, opts, state_uncertainty, req) {
            Ok(out) => return Ok(out),
            Err(flag) => {
                info = flag;
                // ---- estimation completion: use_univariate_smoother_if_singularity_is_detected (DsgeSmoother.m)
                if flag == 422 && opts.smoother_singularity_error {
                    println!(
                        "DsgeSmoother: Stochastic singularity in stationary period detected. If you are sure the model is correctly specified,"
                    );
                    println!(
                        "DsgeSmoother: you may want to enable use_univariate_smoother_if_singularity_is_detected to deal with pathological draws."
                    );
                    anyhow::bail!(
                        "Kalman filter: F is singular in stationary period. Stochastic singularity detected."
                    );
                }
                // ---- end estimation completion
                kalman_algo = if kalman_algo == 1 { 2 } else { 4 };
            }
        }
    }
    debug_assert!(kalman_algo == 2 || kalman_algo == 4);
    let Some(h1) = hm.diag(pp) else {
        // DsgeSmoother.m: augment the state vector with the measurement errors (see
        // kalman::augment), H = 0, then drop the extra states; the measurement errors are
        // the last pp smoothed shocks.
        let HMatrix::Full(hfull) = &hm else { unreachable!() };
        let (mm, rr) = (t.nrows(), q.nrows());
        let s = super::augment::system(t, r, q, hfull, mf);
        let pstar_c = super::augment::cov(pstar, Some(hfull), mf);
        let pinf_c = pinf.map(|p| super::augment::cov(p, None, mf));
        // heteroskedastic: blkdiag(Qvec(:,:,k), H)
        let qv_c: Option<Vec<Mat>> =
            qvec.map(|qv| qv.iter().map(|qk| super::augment::system(t, r, qk, hfull, mf).q).collect());
        let (qq_v, qrt_v) = match &qv_c {
            Some(qv) => {
                let (a, b) = hetero(&s.r, qv);
                (Some(a), Some(b))
            }
            None => (None, None),
        };
        // filter_initial_state: Dynare's a_initial = ST*a_initial with the augmented
        // ST = blkdiag(T, 0) predicts the measurement errors at 0, i.e. [a; Z a] here
        let req_c;
        let req = match req.and_then(|q| q.a_initial.as_ref().map(|ai| (q, ai))) {
            Some((q0, ai)) if ai.len() == mm => {
                let mut q1 = q0.clone();
                q1.a_initial = Some(super::augment::state(ai, mf));
                req_c = q1;
                Some(&req_c)
            }
            _ => req,
        };
        let sys_c = Sys {
            t: &s.t,
            r: &s.r,
            q: &s.q,
            mf: &s.mf,
            qq: mul(&mul(&s.r, false, &s.q, false), false, &s.r, true),
            qrt: mul(&s.q, false, &s.r, true),
            qq_v,
            qrt_v,
        };
        let o = smoother_h3(
            &sys_c,
            &vec![0.0; pp],
            &pstar_c,
            pinf_c.as_ref(),
            y,
            &data_index,
            opts,
            state_uncertainty,
            req,
        );
        return Ok(SmootherOutput {
            alphahat: o.alphahat.rows(0, mm).into_owned(),
            etahat: o.etahat.rows(0, rr).into_owned(),
            epsilonhat: o.etahat.rows(rr, pp).into_owned(),
            ahat: o.ahat.rows(0, mm).into_owned(),
            filtered: o.filtered.rows(0, mm).into_owned(),
            alphahat0: o.alphahat0.rows(0, mm).into_owned(),
            state_uncertainty: o
                .state_uncertainty
                .map(|v| v.iter().map(|m| m.view((0, 0), (mm, mm)).into_owned()).collect()),
            state_uncertainty0: None,
            d: o.d,
            univariate: true,
            info,
            step_ahead: o.step_ahead.map(|sa| StepAhead {
                ak: sa.ak.iter().map(|m| m.rows(0, mm).into_owned()).collect(),
                pk: sa.pk.map(|v| {
                    v.iter().map(|w| w.iter().map(|m| m.view((0, 0), (mm, mm)).into_owned()).collect()).collect()
                }),
                decomp: sa.decomp.map(|v| {
                    v.iter().map(|w| w.iter().map(|m| m.view((0, 0), (mm, rr)).into_owned()).collect()).collect()
                }),
            }),
        });
    };
    let mut out = smoother_h3(&sys, &h1, pstar, pinf, y, &data_index, opts, state_uncertainty, req);
    out.info = info;
    Ok(out)
}

struct Sys<'a> {
    t: &'a Mat,
    r: &'a Mat,
    q: &'a Mat,
    mf: &'a [usize],
    qq: Mat,
    qrt: Mat,
    /// heteroskedastic_filter: R*Qvec(:,:,k)*R' and Qvec(:,:,k)*R', k = 0..=nobs
    qq_v: Option<Vec<Mat>>,
    qrt_v: Option<Vec<Mat>>,
}

impl Sys<'_> {
    fn r_mat(&self) -> &Mat {
        self.r
    }
    /// QQ for the transition after 0-based period `k - 1` (Dynare's `Qvec(:,:,t+1)`).
    fn qq(&self, k: usize) -> &Mat {
        self.qq_v.as_ref().map_or(&self.qq, |v| &v[k.min(v.len() - 1)])
    }
    /// QRt for the smoothed shocks of 0-based period `k` (Dynare's `Qvec(:,:,t)`).
    fn qrt(&self, k: usize) -> &Mat {
        self.qrt_v.as_ref().map_or(&self.qrt, |v| &v[k.min(v.len() - 1)])
    }
    fn tpt(&self, p: &Mat) -> Mat {
        mul(&mul(self.t, false, p, false), false, self.t, true)
    }
    fn tv(&self, a: &Vector) -> Vector {
        self.t * a
    }
}

/// rank(Z*Pinf*Z', tol) > 0 (MATLAB `rank`: number of singular values > tol).
fn rank_zpz(pinf: &Mat, mf: &[usize], tol: f64) -> bool {
    let zpz = pinf.select_rows(mf).select_columns(mf);
    crate::linalg::singular_values(&zpz).iter().any(|&s| s > tol)
}

fn rcond(f: &Mat) -> f64 {
    let (lu, _, info) = crate::linalg::lu(f);
    if info > 0 {
        return 0.0;
    }
    crate::linalg::rcond_from_lu(&lu, crate::linalg::norm1(f))
}

fn inv(f: &Mat) -> Mat {
    crate::linalg::solve(f, &Mat::identity(f.nrows(), f.nrows()))
        .unwrap_or_else(|| Mat::from_element(f.nrows(), f.nrows(), f64::INFINITY))
}

/// Port of kalman/missing_DiffuseKalmanSmootherH1_Z.m. Err(filter_error_flag) where Dynare
/// returns `alphahat = Inf`.
#[allow(clippy::too_many_arguments)]
fn smoother_h1(
    s: &Sys,
    h: &HMatrix,
    pstar1: &Mat,
    pinf1: Option<&Mat>,
    y: &Mat,
    data_index: &[Vec<usize>],
    o: &KalmanOptions,
    su: bool,
    req: Option<&SmootherRequest>,
) -> Result<SmootherOutput, i32> {
    let (mm, pp, smpl) = (s.t.nrows(), y.nrows(), y.ncols());
    let rr = s.q.nrows();
    let (ktol, dtol) = (o.kalman_tol, o.diffuse_kalman_tol);
    let mf = s.mf;
    let zz = |di: &[usize]| -> Vec<usize> { di.iter().map(|&i| mf[i]).collect() };
    let hsub = |di: &[usize]| Mat::from_fn(di.len(), di.len(), |i, j| h.get(di[i], di[j]));

    let mut a: Vec<Vector> = vec![Vector::zeros(mm); smpl + 1];
    let mut a0 = Vector::zeros(mm); // a_initial = T*0
    if let Some(ai) = req.and_then(|r| r.a_initial.as_ref()) {
        a0 = Vector::from_column_slice(ai); // filter_initial_state
        a[0] = a0.clone();
    }
    let mut atilde: Vec<Vector> = vec![Vector::zeros(mm); smpl];
    let mut ak = Mat::zeros(mm, smpl + 1);
    let mut v = Mat::zeros(pp, smpl);
    let mut p: Vec<Mat> = vec![Mat::zeros(0, 0); smpl + 1];
    let mut pstar: Vec<Mat> = vec![Mat::zeros(0, 0); smpl + 1];
    let mut pinf: Vec<Mat> = vec![Mat::zeros(0, 0); smpl + 1];
    let mut i_f: Vec<Mat> = vec![Mat::zeros(0, 0); smpl];
    let mut fstar: Vec<Mat> = vec![Mat::zeros(0, 0); smpl];
    let mut ifstar: Vec<Mat> = vec![Mat::zeros(0, 0); smpl];
    let mut ifinf: Vec<Mat> = vec![Mat::zeros(0, 0); smpl];
    let mut kmat: Vec<Mat> = vec![Mat::zeros(0, 0); smpl]; // K(:,di,t) (mm x |di|)
    let mut kstar: Vec<Mat> = vec![Mat::zeros(0, 0); smpl];
    let mut linf: Vec<Mat> = vec![Mat::zeros(0, 0); smpl];
    let mut lstar: Vec<Mat> = vec![Mat::zeros(0, 0); smpl];
    let mut finf_singular = vec![false; smpl];
    let p0 = pstar1.clone();
    pstar[0] = pstar1.clone();
    let have_pinf = pinf1.is_some();
    pinf[0] = pinf1.cloned().unwrap_or_else(|| Mat::zeros(mm, mm));

    let mut t = 0usize; // number of processed periods (MATLAB t)
    let mut new_rank = have_pinf && rank_zpz(&pinf[0], mf, dtol);
    let (mut pinf_init, mut pstar_init, mut a_init) = (Mat::zeros(mm, mm), Mat::zeros(mm, mm), Vector::zeros(mm));
    if new_rank {
        pinf_init = pinf[0].clone();
        pstar_init = pstar[0].clone();
        a_init = a[0].clone();
        a[0] = s.tv(&a[0]);
        pstar[0] = s.tpt(&pstar[0]) + s.qq(0);
    }
    while new_rank && t < smpl {
        let ti = t; // 0-based period
        t += 1;
        let di = &data_index[ti];
        if di.is_empty() {
            atilde[ti] = a[ti].clone();
            a[ti + 1] = s.tv(&atilde[ti]);
            linf[ti] = s.t.clone();
            pstar[ti + 1] = s.tpt(&pstar[ti]) + s.qq(ti + 1);
            pinf[ti + 1] = s.tpt(&pinf[ti]);
        } else {
            let z = zz(di);
            for (j, &d) in di.iter().enumerate() {
                v[(d, ti)] = y[(d, ti)] - a[ti][z[j]];
            }
            let vv = Vector::from_iterator(di.len(), di.iter().map(|&d| v[(d, ti)]));
            let finf = pinf[ti].select_rows(&z).select_columns(&z);
            if rcond(&finf) < dtol {
                if !finf.iter().all(|x| x.abs() < dtol) {
                    return Err(420);
                }
                finf_singular[ti] = true;
                fstar[ti] = pstar[ti].select_rows(&z).select_columns(&z) + hsub(di);
                if rcond(&fstar[ti]) < ktol {
                    if !fstar[ti].iter().all(|x| x.abs() < ktol) {
                        return Err(421);
                    }
                    a[ti + 1] = s.tv(&a[ti]);
                    pstar[ti + 1] = s.tpt(&pstar[ti]) + s.qq(ti + 1);
                    pinf[ti + 1] = s.tpt(&pinf[ti]);
                } else {
                    ifstar[ti] = inv(&fstar[ti]);
                    kstar[ti] = pstar[ti].select_columns(&z) * &ifstar[ti];
                    pinf[ti + 1] = s.tpt(&pinf[ti]);
                    let tk = s.t * &kstar[ti];
                    let mut ls = s.t.clone();
                    for (j, &zj) in z.iter().enumerate() {
                        for rrow in 0..mm {
                            ls[(rrow, zj)] -= tk[(rrow, j)];
                        }
                    }
                    pstar[ti + 1] = s.t * &pstar[ti] * ls.transpose() + s.qq(ti + 1);
                    lstar[ti] = ls;
                    a[ti + 1] = s.tv(&(&a[ti] + &kstar[ti] * &vv));
                }
            } else {
                ifinf[ti] = inv(&finf);
                let kinf = pinf[ti].select_columns(&z) * &ifinf[ti];
                atilde[ti] = &a[ti] + &kinf * &vv;
                let tk = s.t * &kinf;
                let mut li = s.t.clone();
                for (j, &zj) in z.iter().enumerate() {
                    for rrow in 0..mm {
                        li[(rrow, zj)] -= tk[(rrow, j)];
                    }
                }
                fstar[ti] = pstar[ti].select_rows(&z).select_columns(&z) + hsub(di);
                kstar[ti] = (pstar[ti].select_columns(&z) - &kinf * &fstar[ti]) * &ifinf[ti];
                pstar[ti + 1] =
                    s.t * &pstar[ti] * li.transpose() - &tk * &finf * (s.t * &kstar[ti]).transpose() + s.qq(ti + 1);
                pinf[ti + 1] = s.t * &pinf[ti] * li.transpose();
                linf[ti] = li;
            }
            // As Dynare: a(:,t+1) = T*atilde(:,t) also after the Finf = 0 branch (atilde = 0 there).
            a[ti + 1] = s.tv(&atilde[ti]);
            ak.set_column(ti + 1, &a[ti + 1]);
        }
        if new_rank {
            new_rank = rank_zpz(&pinf[ti + 1], mf, dtol);
        }
    }
    let d = t;
    p[d] = pstar[d].clone();
    while t < smpl {
        let ti = t;
        t += 1;
        // P = tril(P) + tril(P,-1)'
        {
            let pt = &mut p[ti];
            for c in 0..mm {
                for rrow in 0..c {
                    pt[(rrow, c)] = pt[(c, rrow)];
                }
            }
        }
        let di = &data_index[ti];
        if di.is_empty() {
            atilde[ti] = a[ti].clone();
            p[ti + 1] = s.tpt(&p[ti]) + s.qq(ti + 1);
        } else {
            let z = zz(di);
            for (j, &dd) in di.iter().enumerate() {
                v[(dd, ti)] = y[(dd, ti)] - a[ti][z[j]];
            }
            let vv = Vector::from_iterator(di.len(), di.iter().map(|&dd| v[(dd, ti)]));
            let f = p[ti].select_rows(&z).select_columns(&z) + hsub(di);
            let n = di.len();
            let sig: Vec<f64> = (0..n).map(|i| f[(i, i)].sqrt()).collect();
            let fs = Mat::from_fn(n, n, |i, j| f[(i, j)] / (sig[i] * sig[j]));
            if (0..n).any(|i| f[(i, i)] < ktol) || rcond(&fs) < ktol {
                return Err(422);
            }
            let ifs = inv(&fs);
            let iff = Mat::from_fn(n, n, |i, j| ifs[(i, j)] / (sig[i] * sig[j]));
            let pzi = p[ti].select_columns(&z) * &iff;
            atilde[ti] = &a[ti] + &pzi * &vv;
            let k = s.t * &pzi;
            let mut l = s.t.clone();
            for (j, &zj) in z.iter().enumerate() {
                for rrow in 0..mm {
                    l[(rrow, zj)] -= k[(rrow, j)];
                }
            }
            p[ti + 1] = s.t * &p[ti] * l.transpose() + s.qq(ti + 1);
            i_f[ti] = iff;
            kmat[ti] = k;
        }
        a[ti + 1] = s.tv(&atilde[ti]);
        ak.set_column(ti + 1, &a[ti + 1]);
    }
    // L(:,:,t) = T - K(:,di,t)*Z(di,:), rebuilt in the backward pass
    let l_of = |ti: usize| -> Mat {
        let di = &data_index[ti];
        let mut l = s.t.clone();
        if !di.is_empty() {
            let z = zz(di);
            for (j, &zj) in z.iter().enumerate() {
                for rrow in 0..mm {
                    l[(rrow, zj)] -= kmat[ti][(rrow, j)];
                }
            }
        }
        l
    };
    let mut alphahat = Mat::zeros(mm, smpl);
    let mut etahat = Mat::zeros(rr, smpl);
    let mut rvec: Vec<Vector> = vec![Vector::zeros(mm); smpl + 1];
    let mut nmat: Vec<Mat> = if su { vec![Mat::zeros(mm, mm); smpl + 1] } else { Vec::new() };
    let mut vmat: Vec<Mat> = if su { vec![Mat::zeros(mm, mm); smpl] } else { Vec::new() };
    let mut tt = smpl;
    while tt > d {
        tt -= 1;
        let di = &data_index[tt];
        let l = l_of(tt);
        let mut rt = l.transpose() * &rvec[tt + 1];
        if !di.is_empty() {
            let z = zz(di);
            let vv = Vector::from_iterator(di.len(), di.iter().map(|&dd| v[(dd, tt)]));
            let w = &i_f[tt] * vv;
            for (j, &zj) in z.iter().enumerate() {
                rt[zj] += w[j];
            }
        }
        if su {
            let mut nt = mul(&mul(&l, true, &nmat[tt + 1], false), false, &l, false);
            if !di.is_empty() {
                let z = zz(di);
                for (j, &zj) in z.iter().enumerate() {
                    for (k, &zk) in z.iter().enumerate() {
                        nt[(zj, zk)] += i_f[tt][(j, k)];
                    }
                }
            }
            nmat[tt] = nt;
            vmat[tt] = &p[tt] - mul(&mul(&p[tt], false, &nmat[tt], false), false, &p[tt], false);
        }
        rvec[tt] = rt;
        alphahat.set_column(tt, &(&a[tt] + &p[tt] * &rvec[tt]));
        etahat.set_column(tt, &(s.qrt(tt) * &rvec[tt]));
    }
    let (alphahat0, v0) = if d == 0 {
        let r0 = s.t.transpose() * &rvec[0];
        // V0 = P0 - P0*N0*P0 with N0 = L0'*N(:,:,1)*L0, L0 = T (DK (2012), eq. 4.43)
        let v0 = su.then(|| {
            let n0 = s.t.transpose() * &nmat[0] * s.t;
            &p0 - &p0 * n0 * &p0
        });
        (&a0 + &p0 * &r0, v0)
    } else {
        let mut r0: Vec<Vector> = vec![Vector::zeros(mm); d + 1];
        let mut r1: Vec<Vector> = vec![Vector::zeros(mm); d + 1];
        r0[d] = rvec[d].clone();
        let mut n1: Vec<Mat> = if su { vec![Mat::zeros(mm, mm); d + 1] } else { Vec::new() };
        let mut n2: Vec<Mat> = if su { vec![Mat::zeros(mm, mm); d + 1] } else { Vec::new() };
        for tt in (0..d).rev() {
            let di = &data_index[tt];
            if di.is_empty() {
                r1[tt] = linf[tt].transpose() * &r1[tt + 1];
            } else {
                let z = zz(di);
                let vv = Vector::from_iterator(di.len(), di.iter().map(|&dd| v[(dd, tt)]));
                if !finf_singular[tt] {
                    r0[tt] = linf[tt].transpose() * &r0[tt + 1];
                    let inner = &ifinf[tt] * &vv - kstar[tt].transpose() * (s.t.transpose() * &r0[tt + 1]);
                    let mut r1t = linf[tt].transpose() * &r1[tt + 1];
                    for (j, &zj) in z.iter().enumerate() {
                        r1t[zj] += inner[j];
                    }
                    r1[tt] = r1t;
                    if su {
                        let zmat = selection(&z, mm);
                        let l1 = -(s.t * &kstar[tt] * &zmat);
                        let li = &linf[tt];
                        nmat[tt] = li.transpose() * &nmat[tt + 1] * li;
                        n1[tt] = zmat.transpose() * &ifinf[tt] * &zmat
                            + li.transpose() * &n1[tt + 1] * li
                            + l1.transpose() * &nmat[tt + 1] * li;
                        n2[tt] = zmat.transpose() * (-(&ifinf[tt] * &fstar[tt] * &ifinf[tt])) * &zmat
                            + li.transpose() * &n2[tt + 1] * li
                            + li.transpose() * &n1[tt + 1] * &l1
                            + l1.transpose() * n1[tt + 1].transpose() * li
                            + l1.transpose() * &nmat[tt + 1] * &l1;
                    }
                } else {
                    let w = &ifstar[tt] * &vv;
                    let mut r0t = -(lstar[tt].transpose() * &r0[tt + 1]);
                    for (j, &zj) in z.iter().enumerate() {
                        r0t[zj] += w[j];
                    }
                    r0[tt] = r0t;
                    r1[tt] = s.t.transpose() * &r1[tt + 1];
                    if su {
                        let zmat = selection(&z, mm);
                        let ls = &lstar[tt];
                        nmat[tt] = zmat.transpose() * &ifstar[tt] * &zmat + ls.transpose() * &nmat[tt + 1] * ls;
                        n1[tt] = s.t.transpose() * &n1[tt + 1] * ls;
                        n2[tt] = s.t.transpose() * &n2[tt + 1] * s.t.transpose();
                    }
                }
            }
            alphahat.set_column(tt, &(&a[tt] + &pstar[tt] * &r0[tt] + &pinf[tt] * &r1[tt]));
            etahat.set_column(tt, &(s.qrt(tt) * &r0[tt]));
            if su {
                let ps = &pstar[tt];
                let pi = &pinf[tt];
                let pnp = pi * &n1[tt] * ps;
                vmat[tt] = ps - ps * &nmat[tt] * ps - pnp.transpose() - &pnp - pi * &n2[tt] * pi;
            }
        }
        let r10 = s.t.transpose() * &r1[0];
        let r00 = s.t.transpose() * &r0[0];
        // V0 (DK (2012), eq. 5.30) from N0 = T'*N(:,:,1)*T, N_10, N_20
        let v0 = su.then(|| {
            let n0 = s.t.transpose() * &nmat[0] * s.t;
            let n10 = s.t.transpose() * &n1[0] * s.t;
            let n20 = s.t.transpose() * &n2[0] * s.t;
            let pnp = &pinf_init * &n10 * &pstar_init;
            &pstar_init - &pstar_init * &n0 * &pstar_init - pnp.transpose() - &pnp - &pinf_init * &n20 * &pinf_init
        });
        (&a_init + &pstar_init * &r00 + &pinf_init * &r10, v0)
    };
    let ahat = Mat::from_columns(&atilde);
    // aK / PK / decomp
    let step = req.map(|rq| {
        let p_after = |ti: usize| (ti >= d).then(|| &p[ti + 1]);
        let mut sa = step_ahead(s, &ak, &p_after, smpl, rq);
        if rq.decomposition && rq.nk > 0 {
            let rr = s.q.nrows();
            let qq_last = s.qq(smpl);
            let zqz = Mat::from_fn(mf.len(), mf.len(), |i, j| qq_last[(mf[i], mf[j])]);
            let zrqinv = inv(&zqz);
            let r_mat = s.r_mat();
            let mut dec = vec![vec![Mat::zeros(mm, rr); smpl + rq.nk]; rq.nk];
            for ti in d..smpl {
                let di = &data_index[ti];
                if di.is_empty() {
                    continue;
                }
                let z = zz(di);
                let vv = Vector::from_iterator(di.len(), di.iter().map(|&dd| v[(dd, ti)]));
                let w = &i_f[ti] * &vv;
                let mut zw = Vector::zeros(mm);
                for (j, &zj) in z.iter().enumerate() {
                    zw[zj] += w[j];
                }
                let eta = s.qrt(ti) * zw;
                // AAA = P(:,:,t)*Z(di,:)'*ZRQinv(di,di)*(Z(di,:)*R .* eta')
                let zinv = Mat::from_fn(di.len(), di.len(), |a, b| zrqinv[(di[a], di[b])]);
                let zr = Mat::from_fn(di.len(), rr, |a, j| r_mat[(z[a], j)] * eta[j]);
                let pz = p[ti].select_columns(&z);
                let mut aaa = pz * zinv * zr;
                dec[0][ti + 1] = aaa.clone();
                for hh in 1..rq.nk {
                    aaa = s.t * &aaa;
                    dec[hh][ti + 1 + hh] = aaa.clone();
                }
            }
            sa.decomp = Some(dec);
        }
        sa
    });
    let mut out = finish(s, y, alphahat, etahat, ahat, ak, alphahat0, if su { Some(vmat) } else { None }, d, false);
    out.state_uncertainty0 = v0;
    out.step_ahead = step;
    Ok(out)
}

/// `aK(j,:,t+j) = T*aK(j-1,:,t+j-1)` from the one-step-ahead predictions
/// `ak(:, t+1)`, and `PK(j,:,:,t+j) = T*PK(j-1)*T' + QQ` from the predicted covariances
/// after each non-diffuse period (`p_after(t)`).
fn step_ahead<'m>(
    s: &Sys,
    ak: &Mat,
    p_after: &dyn Fn(usize) -> Option<&'m Mat>,
    smpl: usize,
    rq: &SmootherRequest,
) -> StepAhead {
    let (mm, nk) = (s.t.nrows(), rq.nk);
    let mut aks = vec![Mat::zeros(mm, smpl + nk); nk];
    let mut pks = rq.filter_covariance.then(|| vec![vec![Mat::zeros(mm, mm); smpl + nk]; nk]);
    for ti in 0..smpl {
        if nk == 0 {
            break;
        }
        let mut a = ak.column(ti + 1).into_owned();
        aks[0].set_column(ti + 1, &a);
        for j in 1..nk {
            a = s.t * &a;
            aks[j].set_column(ti + 1 + j, &a);
        }
        if let (Some(pk), Some(p1)) = (pks.as_mut(), p_after(ti)) {
            let mut pf = p1.clone();
            pk[0][ti + 1] = pf.clone();
            for j in 1..nk {
                pf = s.tpt(&pf) + s.qq(ti + 1);
                pk[j][ti + 1 + j] = pf.clone();
            }
        }
    }
    StepAhead { ak: aks, pk: pks, decomp: None }
}

/// Selection matrix Z(di,:) (|z| x mm).
fn selection(z: &[usize], mm: usize) -> Mat {
    let mut m = Mat::zeros(z.len(), mm);
    for (j, &zj) in z.iter().enumerate() {
        m[(j, zj)] = 1.0;
    }
    m
}

#[allow(clippy::too_many_arguments)]
fn finish(
    s: &Sys,
    y: &Mat,
    alphahat: Mat,
    etahat: Mat,
    ahat: Mat,
    ak: Mat,
    alphahat0: Vector,
    v: Option<Vec<Mat>>,
    d: usize,
    univariate: bool,
) -> SmootherOutput {
    let (pp, smpl) = (y.nrows(), y.ncols());
    let epsilonhat = Mat::from_fn(pp, smpl, |i, t| y[(i, t)] - alphahat[(s.mf[i], t)]);
    let filtered = ak.columns(1, smpl).into_owned();
    SmootherOutput {
        alphahat,
        etahat,
        epsilonhat,
        ahat,
        filtered,
        alphahat0,
        state_uncertainty: v,
        state_uncertainty0: None,
        d,
        univariate,
        info: 0,
        step_ahead: None,
    }
}

/// Port of kalman/missing_DiffuseKalmanSmootherH3_Z.m (univariate, H diagonal).
#[allow(clippy::too_many_arguments)]
fn smoother_h3(
    s: &Sys,
    h: &[f64],
    pstar_in: &Mat,
    pinf_in: Option<&Mat>,
    y: &Mat,
    data_index: &[Vec<usize>],
    o: &KalmanOptions,
    su: bool,
    req: Option<&SmootherRequest>,
) -> SmootherOutput {
    let (mm, pp, smpl) = (s.t.nrows(), y.nrows(), y.ncols());
    let rr = s.q.nrows();
    let (ktol, dtol) = (o.kalman_tol, o.diffuse_kalman_tol);
    let mf = s.mf;
    let mut v = Mat::zeros(pp, smpl);
    let mut a: Vec<Vector> = vec![Vector::zeros(mm); smpl];
    let mut a1: Vec<Vector> = vec![Vector::zeros(mm); smpl + 1];
    if let Some(ai) = req.and_then(|r| r.a_initial.as_ref()).filter(|ai| ai.len() == mm) {
        a1[0] = Vector::from_column_slice(ai); // filter_initial_state
    }
    let mut ak = Mat::zeros(mm, smpl + 1);
    let mut fstar = Mat::zeros(pp, smpl);
    let mut finf = Mat::zeros(pp, smpl);
    let mut fi = Mat::zeros(pp, smpl);
    let mut ki: Vec<Mat> = vec![Mat::zeros(0, 0); smpl]; // mm x pp per period
    let mut kstar: Vec<Mat> = Vec::new();
    let mut kinf: Vec<Mat> = Vec::new();
    let pinit = pstar_in.clone();
    let mut pstar: Vec<Mat> = vec![pstar_in.clone()];
    let mut pinf: Vec<Mat> = vec![pinf_in.cloned().unwrap_or_else(|| Mat::zeros(mm, mm))];
    let mut pstar1: Vec<Mat> = Vec::new();
    let mut pinf1: Vec<Mat> = Vec::new();
    let mut p1: Vec<Mat> = vec![Mat::zeros(0, 0); smpl + 1];
    let ainit = a1[0].clone();

    let mut t = 0usize;
    let mut new_rank = pinf_in.is_some() && rank_zpz(&pinf[0], mf, dtol);
    let (mut pinf_init, mut pstar_init) = (Mat::zeros(mm, mm), Mat::zeros(mm, mm));
    if new_rank {
        pinf_init = pinf[0].clone();
        pstar_init = pstar[0].clone();
        pstar[0] = s.tpt(&pstar[0]) + s.qq(0);
    }
    while new_rank && t < smpl {
        let ti = t;
        t += 1;
        a[ti] = a1[ti].clone();
        pstar1.push(pstar[ti].clone());
        pinf1.push(pinf[ti].clone());
        kstar.push(Mat::zeros(mm, pp));
        kinf.push(Mat::zeros(mm, pp));
        let (mut ps, mut pi) = (pstar[ti].clone(), pinf[ti].clone());
        for &i in &data_index[ti] {
            let z = mf[i];
            v[(i, ti)] = y[(i, ti)] - a[ti][z];
            fstar[(i, ti)] = ps[(z, z)] + h[i];
            finf[(i, ti)] = pi[(z, z)];
            let ks = ps.column(z).into_owned();
            kstar[ti].set_column(i, &ks);
            if finf[(i, ti)] > dtol && new_rank {
                let kin = pi.column(z).into_owned();
                kinf[ti].set_column(i, &kin);
                let fin = finf[(i, ti)];
                let kf = &kin / fin;
                a[ti] += &kf * v[(i, ti)];
                ps += &kin * kf.transpose() * (fstar[(i, ti)] / fin) - &ks * kf.transpose() - &kf * ks.transpose();
                pi -= &kin * kin.transpose() / fin;
            } else if fstar[(i, ti)] > ktol {
                let fs = fstar[(i, ti)];
                a[ti] += &ks * (v[(i, ti)] / fs);
                ps -= &ks * ks.transpose() / fs;
            }
        }
        pstar[ti] = ps;
        pinf[ti] = pi;
        a1[ti + 1] = s.tv(&a[ti]);
        ak.set_column(ti + 1, &a1[ti + 1]);
        pstar.push(s.tpt(&pstar[ti]) + s.qq(ti + 1));
        pinf.push(s.tpt(&pinf[ti]));
        if new_rank {
            new_rank = rank_zpz(&pinf[ti + 1], mf, dtol);
        }
    }
    let d = t;
    let mut p = pstar[d].clone();
    while t < smpl {
        let ti = t;
        t += 1;
        a[ti] = a1[ti].clone();
        p1[ti] = p.clone();
        let mut kt = Mat::zeros(mm, pp);
        for &i in &data_index[ti] {
            let z = mf[i];
            v[(i, ti)] = y[(i, ti)] - a[ti][z];
            fi[(i, ti)] = p[(z, z)] + h[i];
            let k = p.column(z).into_owned();
            if fi[(i, ti)] > ktol {
                let f = fi[(i, ti)];
                a[ti] += &k * (v[(i, ti)] / f);
                p -= &k * k.transpose() / f;
                p = (&p + p.transpose()) * 0.5; // Dynare 7.2
            }
            kt.set_column(i, &k);
        }
        ki[ti] = kt;
        a1[ti + 1] = s.tv(&a[ti]);
        p = s.tpt(&p) + s.qq(ti + 1);
        p = (&p + p.transpose()) * 0.5; // Dynare 7.2
        ak.set_column(ti + 1, &a1[ti + 1]);
    }
    p1[t] = p;

    let mut alphahat = Mat::zeros(mm, smpl);
    let mut etahat = Mat::zeros(rr, smpl);
    let mut ri = Vector::zeros(mm);
    let mut ni = if su { Mat::zeros(mm, mm) } else { Mat::zeros(0, 0) };
    let mut vmat: Vec<Mat> = if su { vec![Mat::zeros(mm, mm); smpl] } else { Vec::new() };
    let mut tt = smpl;
    while tt > d {
        tt -= 1;
        for &i in data_index[tt].iter().rev() {
            let f = fi[(i, tt)];
            if f > ktol {
                let z = mf[i];
                let k = ki[tt].column(i);
                // Li = I - Ki*Zi/Fi;  ri = Zi'/Fi*v + Li'*ri
                let kr = k.dot(&ri);
                ri[z] += v[(i, tt)] / f - kr / f;
                if su {
                    // Ni = Zi'/Fi*Zi + Li'*Ni*Li
                    let kf = k / f;
                    let nk = &ni * &kf; // N*k
                    let knk = kf.dot(&nk);
                    let kn = ni.transpose() * &kf; // (k'N)'
                    for c in 0..mm {
                        ni[(z, c)] -= kn[c];
                    }
                    for rrow in 0..mm {
                        ni[(rrow, z)] -= nk[rrow];
                    }
                    ni[(z, z)] += knk + 1.0 / f;
                }
            }
        }
        alphahat.set_column(tt, &(&a1[tt] + &p1[tt] * &ri));
        etahat.set_column(tt, &(s.qrt(tt) * &ri));
        ri = s.t.transpose() * &ri;
        if su {
            vmat[tt] = &p1[tt] - mul(&mul(&p1[tt], false, &ni, false), false, &p1[tt], false);
            ni = mul(&mul(s.t, true, &ni, false), false, s.t, false);
        }
    }
    let alphahat0 = if d == 0 {
        &ainit + &pinit * &ri
    } else {
        let mut r0 = ri;
        let mut r1 = Vector::zeros(mm);
        let mut n0 = ni;
        let mut n1 = if su { Mat::zeros(mm, mm) } else { Mat::zeros(0, 0) };
        let mut n2 = n1.clone();
        let mut r00 = Vector::zeros(mm);
        let mut r10 = Vector::zeros(mm);
        for tt in (0..d).rev() {
            for &i in data_index[tt].iter().rev() {
                let z = mf[i];
                let (fin, fs) = (finf[(i, tt)], fstar[(i, tt)]);
                if fin > dtol {
                    let kin = kinf[tt].column(i).into_owned();
                    let ks = kstar[tt].column(i).into_owned();
                    // Linf = I - Kinf*Zi/Finf; L0 = (Kinf*(Fstar/Finf) - Kstar)*Zi/Finf
                    let l0c = (&kin * (fs / fin) - &ks) / fin; // L0 = l0c * e_z'
                    let kinf_f = &kin / fin; // Linf = I - kinf_f * e_z'
                    // r1 = Zi'*v/Finf + L0'*r0 + Linf'*r1
                    let new_r1z = v[(i, tt)] / fin + l0c.dot(&r0) - kinf_f.dot(&r1);
                    r1[z] += new_r1z;
                    // r0 = Linf'*r0
                    let kr = kinf_f.dot(&r0);
                    r0[z] -= kr;
                    if su {
                        let mut linf = Mat::identity(mm, mm);
                        let mut l0 = Mat::zeros(mm, mm);
                        for rrow in 0..mm {
                            linf[(rrow, z)] -= kinf_f[rrow];
                            l0[(rrow, z)] = l0c[rrow];
                        }
                        let zz = {
                            let mut m = Mat::zeros(mm, mm);
                            m[(z, z)] = 1.0;
                            m
                        };
                        let new_n2 = -&zz / (fin * fin) * fs
                            + linf.transpose() * &n2 * &linf
                            + linf.transpose() * &n1 * &l0
                            + l0.transpose() * n1.transpose() * &linf
                            + l0.transpose() * &n0 * &l0;
                        let new_n1 = &zz / fin + linf.transpose() * &n1 * &linf + l0.transpose() * &n0 * &linf;
                        let new_n0 = linf.transpose() * &n0 * &linf;
                        n2 = new_n2;
                        n1 = new_n1;
                        n0 = new_n0;
                    }
                } else if fs > ktol {
                    let ks = kstar[tt].column(i).into_owned();
                    let kr = ks.dot(&r0) / fs;
                    r0[z] += v[(i, tt)] / fs - kr;
                    if su {
                        let kf = &ks / fs;
                        let nk = &n0 * &kf;
                        let knk = kf.dot(&nk);
                        let kn = n0.transpose() * &kf;
                        for c in 0..mm {
                            n0[(z, c)] -= kn[c];
                        }
                        for rrow in 0..mm {
                            n0[(rrow, z)] -= nk[rrow];
                        }
                        n0[(z, z)] += knk + 1.0 / fs;
                    }
                }
            }
            alphahat.set_column(tt, &(&a1[tt] + &pstar1[tt] * &r0 + &pinf1[tt] * &r1));
            etahat.set_column(tt, &(s.qrt(tt) * &r0));
            if su {
                let (ps, pi) = (&pstar1[tt], &pinf1[tt]);
                let pnp = pi * &n1 * ps;
                vmat[tt] = ps - ps * &n0 * ps - pnp.transpose() - &pnp - pi * &n2 * pi;
            }
            let r0n = s.t.transpose() * &r0;
            let r1n = s.t.transpose() * &r1;
            if su {
                n0 = s.t.transpose() * &n0 * s.t;
                n1 = s.t.transpose() * &n1 * s.t;
                n2 = s.t.transpose() * &n2 * s.t;
            }
            if tt > 0 {
                r0 = r0n;
                r1 = r1n;
            } else {
                r00 = r0n;
                r10 = r1n;
            }
        }
        &ainit + &pstar_init * &r00 + &pinf_init * &r10
    };
    let ahat = Mat::from_columns(&a);
    // aK / PK / decomp
    let step = req.map(|rq| {
        let p_after = |ti: usize| (ti >= d).then(|| &p1[ti + 1]);
        let mut sa = step_ahead(s, &ak, &p_after, smpl, rq);
        if rq.decomposition && rq.nk > 0 {
            let qq_last = s.qq(smpl);
            let zqz = Mat::from_fn(mf.len(), mf.len(), |i, j| qq_last[(mf[i], mf[j])]);
            let zrqinv = inv(&zqz);
            let r_mat = s.r_mat();
            // Z'*ZRQinv*Z*R (mm x rr)
            let mut zzr = Mat::zeros(mm, rr);
            for (a, &za) in mf.iter().enumerate() {
                for (b, &zb) in mf.iter().enumerate() {
                    for j in 0..rr {
                        zzr[(za, j)] += zrqinv[(a, b)] * r_mat[(zb, j)];
                    }
                }
            }
            let mut dec = vec![vec![Mat::zeros(mm, rr); smpl + rq.nk]; rq.nk];
            for ti in d..smpl {
                let mut ri_d = Vector::zeros(mm);
                for &i in data_index[ti].iter().rev() {
                    let f = fi[(i, ti)];
                    if f > ktol {
                        let z = mf[i];
                        let kr = ki[ti].column(i).dot(&ri_d);
                        ri_d[z] += v[(i, ti)] / f - kr / f;
                    }
                }
                let eta = s.qrt(ti) * ri_d;
                let aaa = &p1[ti] * &zzr;
                let mut ttok = Mat::identity(mm, mm);
                for hh in 0..rq.nk {
                    let bbb = &ttok * &aaa;
                    let mut m = Mat::zeros(mm, rr);
                    for j in 0..rr {
                        m.set_column(j, &(bbb.column(j) * eta[j]));
                    }
                    dec[hh][ti + 1 + hh] = m;
                    ttok = s.t * ttok;
                }
            }
            sa.decomp = Some(dec);
        }
        sa
    });
    let mut out = finish(s, y, alphahat, etahat, ahat, ak, alphahat0, if su { Some(vmat) } else { None }, d, true);
    out.step_ahead = step;
    out
}
