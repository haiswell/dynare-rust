// SPDX-License-Identifier: GPL-3.0-or-later
//! Posterior IRFs of a DSGE-VAR: the `MAX_nirfs_dsgevar` branch of
//! `PosteriorIRF_core1.m` and its statistics in `PosteriorIRF.m` (`oo_.PosteriorIRF.bvardsge`).
//!
//! For each posterior sub-draw: (SIGMA_u, PHI) drawn from the DSGE-VAR posterior
//! (`rand_inverse_wishart.m`, `rand_matrix_normal.m`, explosive VARs rejected), identified
//! with the rotation of the DSGE impact matrix (`qr2(A0')`, Del Negro and Schorfheide 2004).
//! The random numbers come from ChaCha8 (seed `--seed`), so results agree with Dynare's
//! statistically only.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::collapsible_if
)] // index loops mirror the MATLAB code

use std::path::Path;

use anyhow::{Result, anyhow, bail};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use super::dsge_var::DsgeVarPosterior;
use crate::analysis::StochSimulOptions;
use crate::estimation::mcmc::Chain;
use crate::linalg::{Mat, mul};

const STREAM: u64 = 0x4253_5641_5244_5347;

fn inv(a: &Mat) -> Mat {
    crate::linalg::solve(a, &Mat::identity(a.nrows(), a.nrows()))
        .unwrap_or_else(|| Mat::from_element(a.nrows(), a.ncols(), f64::NAN))
}

fn chol_lower(a: &Mat) -> Option<Mat> {
    let s = (a + a.transpose()) * 0.5;
    s.cholesky().map(|c| c.l())
}

/// `qr2.m`: Q of the QR decomposition, columns flipped so that diag(R) >= 0.
fn qr2_q(a: &Mat) -> Mat {
    let qr = a.clone().qr();
    let (mut q, r) = (qr.q(), qr.r());
    for i in 0..r.nrows().min(r.ncols()) {
        if r[(i, i)] < 0.0 {
            for k in 0..q.nrows() {
                q[(k, i)] = -q[(k, i)];
            }
        }
    }
    q
}

/// `stock_irf_bvardsge` of every draw: `[shock][observable x period]`.
pub fn posterior_irf_draws(
    post: &DsgeVarPosterior,
    draws: &[Vec<f64>],
    periods: usize,
    seed: u64,
) -> Result<Vec<Vec<Mat>>> {
    let m = post.moments;
    let (nv, nl, nobs) = (m.nvobs, m.nlags, m.nobs as f64);
    let k = nv * nl + 1;
    let n_exo = post.dsge.model.n_exo();
    if n_exo != nv {
        bail!("DSGE-VAR posterior IRFs need as many shocks ({n_exo}) as observed variables ({nv})");
    }
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(STREAM);
    let mut randn = |n: usize| -> Vec<f64> { (0..n).map(|_| StandardNormal.sample(&mut rng)).collect() };
    let mut out = Vec::with_capacity(draws.len());
    for x in draws {
        let e =
            post.evaluate(x).map_err(|f| anyhow!("PosteriorIRF :: dsge_var_likelihood failed (info {})", f.info))?;
        let w = e.weight;
        if !w.is_finite() {
            bail!("DSGE-VAR posterior IRFs are not available with dsge_var = Inf");
        }
        let dof = (nobs * (1.0 + w)).floor() as usize;
        let v = dof
            .checked_sub(k)
            .ok_or_else(|| anyhow!("DSGE-VAR: not enough degrees of freedom for the posterior draws"))?;
        // SIGMA_inv_upper_chol = chol(inv(SIGMAu*nobs*(w+1)))
        let h_upper = chol_lower(&inv(&e.sigma_u_tilde.scale(nobs * (w + 1.0))))
            .ok_or_else(|| anyhow!("DSGE-VAR: SIGMA_u_tilde is not positive definite"))?
            .transpose();
        let ixx_l = chol_lower(&e.ixx).ok_or_else(|| anyhow!("DSGE-VAR: iXX is not positive definite"))?;
        let mut companion = Mat::zeros(nv * nl, nv * nl);
        for i in nv..nv * nl {
            companion[(i, i - nv)] = 1.0;
        }
        let sig_l = loop {
            // rand_inverse_wishart(nv, v, H_inv_upper_chol): inv(X'X), X = randn(v, nv)*H
            let xr = Mat::from_vec(v, nv, randn(v * nv)) * &h_upper;
            let g = inv(&mul(&xr, true, &xr, false));
            let Some(gl) = chol_lower(&g) else { continue };
            // rand_matrix_normal(k, nv, PHI, chol(SIGMAu_draw)', chol(iXX)') = PHI + L_iXX * B1 * L_SIG'
            let b1 = Mat::from_vec(k, nv, randn(k * nv));
            let phi = &e.phi_tilde + &ixx_l * b1 * gl.transpose();
            for r in 0..nv {
                for c in 0..nv * nl {
                    companion[(r, c)] = phi[(c, r)];
                }
            }
            if !crate::linalg::eig_general(&companion).iter().any(|z| z.norm() > 1.000000001) {
                break gl;
            }
        };
        // rotation: Atheta(order_var,:) = ghu*sqrt(Sigma_e); A0 = Atheta(mfys,:); OMEGAstar = qr2(A0')
        let res = post.dsge.resolve(x).map_err(|f| anyhow!("PosteriorIRF :: no solution (info {})", f.info))?;
        let dr = &res.dr;
        let sq = res.q.map(f64::sqrt);
        let ghu_s = &dr.ghu * &sq;
        let mut atheta = Mat::zeros(post.dsge.model.n_endo(), n_exo);
        for i in 0..ghu_s.nrows() {
            atheta.row_mut(dr.order_var[i]).copy_from(&ghu_s.row(i));
        }
        let a0 = Mat::from_fn(nv, n_exo, |i, j| atheta[(post.dsge.ss_index.mfys[i], j)]);
        let omega = qr2_q(&a0.transpose());
        let st = &sig_l * omega.transpose();
        let mut per_shock: Vec<Mat> = vec![Mat::zeros(nv, periods); n_exo];
        let mut power = Mat::identity(nv * nl, nv * nl);
        for t in 0..periods {
            if t > 0 {
                power = &companion * &power;
            }
            let tmp3 = power.view((0, 0), (nv, nv)) * &st;
            for (s, ps) in per_shock.iter_mut().enumerate() {
                for o in 0..nv {
                    ps[(o, t)] = tmp3[(o, s)];
                }
            }
        }
        for ps in per_shock.iter_mut() {
            for o in 0..nv {
                let (mn, mx) =
                    ps.row(o).iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
                if mx - mn <= 1e-10 {
                    ps.row_mut(o).fill(0.0);
                }
            }
        }
        out.push(per_shock);
    }
    Ok(out)
}

/// `oo_.PosteriorIRF.bvardsge` (bayesian_irf with dsge_var): statistics over the sub-draws,
/// written to `<fname>_Bayesian_IRF_BVAR-DSGE_<shock>.csv` (observables x periods).
#[allow(clippy::too_many_arguments)]
pub fn run(
    post: &DsgeVarPosterior,
    chains: &[Chain],
    o: &StochSimulOptions,
    out_dir: &Path,
    fname: &str,
    seed: u64,
) -> Result<()> {
    if !o.bayesian_irf {
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    let model = &post.dsge.model;
    let periods = if o.irf == 0 { 40 } else { o.irf };
    let b = crate::analysis::posterior::number_of_subdraws(chains, o.sub_draws)?;
    if b == 0 {
        return Ok(());
    }
    let draws = crate::analysis::posterior::posterior_subsample(chains, b, seed)?;
    let stock = posterior_irf_draws(post, &draws, periods, seed)?;
    let shocks = crate::analysis::irf::irf_shock_indices(&model.exo_names, o.irf_shocks.as_deref())?;
    std::fs::create_dir_all(out_dir)?;
    for &s in &shocks {
        let mut w =
            String::from("variable,period,mean,median,var,hpd_inf,hpd_sup,d10,d20,d30,d40,d50,d60,d70,d80,d90\n");
        for (j, &v) in post.dsge.varobs.iter().enumerate() {
            for t in 0..periods {
                let m =
                    crate::analysis::posterior::moments_of(stock.iter().map(|d| d[s][(j, t)]).collect(), o.mh_conf_sig);
                w += &format!(
                    "{},{},{},{},{},{},{}",
                    model.endo_names[v],
                    t + 1,
                    m.mean,
                    m.median,
                    m.var,
                    m.hpd.0,
                    m.hpd.1
                );
                for d in m.deciles {
                    w += &format!(",{d}");
                }
                w.push('\n');
            }
        }
        std::fs::write(out_dir.join(format!("{fname}_Bayesian_IRF_BVAR-DSGE_{}.csv", model.exo_names[s])), w)?;
    }
    crate::progress::note(format!(
        "Posterior (bvar-dsge) IRFs: {b} draws, {:.1}s; {fname}_Bayesian_IRF_BVAR-DSGE_<shock>.csv in {}",
        t0.elapsed().as_secs_f64(),
        out_dir.display()
    ));
    Ok(())
}
