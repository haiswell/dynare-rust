// SPDX-License-Identifier: GPL-3.0-or-later
//! Parameter samples of the GSA (`stability_mapping.m`, first part): LPTAU (Sobol) or Latin
//! hypercube draws in the unit cube, mapped to the prior ranges (`prior_range=1`) or through
//! the prior quantile functions (`+gsa/prior_draw.m`, `prior_range=0`), and Morris
//! trajectories (`+gsa/Sampling_Function_2.m`).

use anyhow::{Result, bail};

use super::rng::MatlabRng;
use super::sobol::qmc_sequence;
use crate::estimation::special::{betainc, betainv, gaminv, gammainc, norminv, wblinv};
use crate::estimation::{Bounds, Prior, PriorShape};

/// Row-major sample: `rows[k][j]`.
pub type Sample = Vec<Vec<f64>>;

/// MATLAB colon `a:d:b` (same element formula as MATLAB: first half from `a`, second half
/// back from `a + n*d`).
pub fn colon(a: f64, d: f64, b: f64) -> Vec<f64> {
    let q = (b - a) / d;
    let tol = 3.0 * f64::EPSILON * q.abs().max(1.0);
    let n = if (q - q.round()).abs() <= tol { q.round() } else { q.floor() };
    if n < 0.0 {
        return Vec::new();
    }
    let n = n as usize;
    let c = a + n as f64 * d;
    (0..=n).map(|k| if 2 * k < n { a + k as f64 * d } else { c - (n - k) as f64 * d }).collect()
}

/// Port of `+gsa/Sampling_Function_2.m` without groups: `r` Morris trajectories of
/// `k + 1` points each on a `p`-level grid of [LB, UB] (here the unit cube).
pub fn morris_sampling(p: usize, k: usize, r: usize, rng: &mut MatlabRng) -> Sample {
    let pf = p as f64;
    let delta = pf / (2.0 * pf - 2.0);
    let sizea = k;
    let sizeb = k + 1;
    let my_int = colon(0.0, 1.0 / (pf - 1.0), 1.0 - delta);
    let int_used = colon(0.0, 1.0 / my_int.len() as f64, 1.0);
    let mut out = Vec::with_capacity(r * sizeb);
    for _ in 0..r {
        let v = rng.rand_vec(k);
        let randmult: Vec<f64> = v.iter().map(|&x| if x < 0.5 { -1.0 } else { 1.0 }).collect();
        let perm_e = rng.randperm(sizea);
        // P0(perm_e(j), j) = 1: column j of the permutation has its 1 in row perm_e[j]
        let v2 = rng.rand_vec(k);
        let x0: Vec<f64> = v2
            .iter()
            .map(|&vi| {
                // w = last index with IntUsed - v < 0
                let w = int_used.iter().rposition(|&u| u - vi < 0.0).expect("Sampling_Function_2: no grid point");
                my_int[w]
            })
            .collect();
        // AuxMat = Delta*0.5*((2*B - A)*DD0 + A), B(r, c) = 1 if r > c
        for row in 0..sizeb {
            // B0 = (A0*x0' + AuxMat) * P0 -> B0(row, j) = (x0 + aux)(row, perm_e[j])
            let pre: Vec<f64> = (0..k)
                .map(|c| {
                    let b = if row > c { 1.0 } else { 0.0 };
                    let aux = delta * 0.5 * ((2.0 * b - 1.0) * randmult[c] + 1.0);
                    x0[c] + aux
                })
                .collect();
            let b0: Vec<f64> = (0..k).map(|j| pre[perm_e[j]]).collect();
            // In = LB + B0 .* (UB - LB) with LB = 0, UB = 1
            out.push(b0.iter().map(|&x| 0.0 + x * (1.0 - 0.0)).collect());
        }
    }
    out
}

/// What `stability_mapping.m` needs to draw the sample.
pub struct SamplingSpec<'a> {
    pub nsam: usize,
    pub nshock: usize,
    pub np: usize,
    pub ilptau: u8,
    pub morris: bool,
    pub morris_nliv: usize,
    pub morris_ntra: usize,
    pub pprior: bool,
    pub prior_range: bool,
    /// GSA bounds: `prior_bounds(bayestopt_, prior_trunc)` intersected with explicit bounds.
    pub bounds: &'a Bounds,
    pub priors: &'a [Prior],
    /// `neighborhood_width` sampling around `xparam1` (pprior = 0).
    pub neighborhood: Option<(f64, &'a [f64])>,
    /// `xparam1` and `hh` of the mode file (`pprior = 0` without `neighborhood_width`): the sample
    /// is drawn from `N(xparam1, inv(hh))`.
    pub posterior: Option<(&'a [f64], &'a crate::linalg::Mat)>,
}

/// The sample of `stability_mapping.m` (`fload == 0` branch), consuming `rng` in Dynare's
/// order. Returns `(lpmat0, lpmat)`: shock (stderr/corr) parameters and structural parameters.
pub fn draw_sample(spec: &SamplingSpec, rng: &mut MatlabRng) -> Result<(Sample, Sample)> {
    let (nshock, np) = (spec.nshock, spec.np);
    let mut nsam = spec.nsam;
    let mut lpmat0: Sample;
    let mut lpmat: Sample;
    if spec.morris {
        let m = morris_sampling(spec.morris_nliv, np + nshock, spec.morris_ntra, rng);
        let nliv = spec.morris_nliv as f64;
        let m: Sample = m
            .into_iter()
            .map(|row| row.into_iter().map(|x| x * (nliv - 1.0) / nliv + 1.0 / nliv / 2.0).collect())
            .collect();
        nsam = m.len();
        lpmat0 = m.iter().map(|r| r[..nshock].to_vec()).collect();
        lpmat = m.iter().map(|r| r[nshock..].to_vec()).collect();
    } else {
        lpmat0 = vec![vec![0.0; nshock]; nsam];
        if np < 1112 && spec.ilptau > 0 {
            lpmat = if np > 0 { qmc_sequence(np, 1, nsam)? } else { vec![Vec::new(); nsam] };
            if np > 30 || spec.ilptau == 2 {
                for j in 0..np {
                    let perm = rng.randperm(nsam);
                    let col: Vec<f64> = perm.iter().map(|&i| lpmat[i][j]).collect();
                    for (row, v) in lpmat.iter_mut().zip(col) {
                        row[j] = v;
                    }
                }
            }
        } else {
            lpmat = vec![vec![0.0; np]; nsam];
            for j in 0..np {
                let perm = rng.randperm(nsam);
                for (row, &p) in lpmat.iter_mut().zip(&perm) {
                    row[j] = (p + 1) as f64 / (nsam as f64 + 1.0);
                }
            }
        }
    }
    let b = spec.bounds;
    if spec.pprior {
        for j in 0..nshock {
            if !spec.morris {
                let perm = rng.randperm(nsam);
                for (row, &p) in lpmat0.iter_mut().zip(&perm) {
                    row[j] = (p + 1) as f64 / (nsam as f64 + 1.0);
                }
            }
            if spec.prior_range {
                for row in lpmat0.iter_mut() {
                    row[j] = row[j] * (b.ub[j] - b.lb[j]) + b.lb[j];
                }
            }
        }
        if spec.prior_range {
            const HUGE: f64 = 1e7; // options_.huge_number
            for j in 0..np {
                let lo = (-HUGE).max(b.lb[j + nshock]);
                let hi = HUGE.min(b.ub[j + nshock]);
                for row in lpmat.iter_mut() {
                    row[j] = row[j] * (hi - lo) + lo;
                }
            }
        } else {
            let pd = PriorDraw::new(spec.priors, b);
            for k in 0..nsam {
                for j in 0..nshock {
                    lpmat0[k][j] = pd.draw(j, lpmat0[k][j]);
                }
                for j in 0..np {
                    lpmat[k][j] = pd.draw(j + nshock, lpmat[k][j]);
                }
            }
        }
    } else if let Some((width, x)) = spec.neighborhood {
        for j in 0..nshock {
            if !spec.morris {
                let perm = rng.randperm(nsam);
                for (row, &p) in lpmat0.iter_mut().zip(&perm) {
                    row[j] = (p + 1) as f64 / (nsam as f64 + 1.0);
                }
            }
            let ub = b.ub[j].min(x[j] * (1.0 + width));
            let lb = b.lb[j].max(x[j] * (1.0 - width));
            for row in lpmat0.iter_mut() {
                row[j] = row[j] * (ub - lb) + lb;
            }
        }
        for j in 0..np {
            let xj = x[j + nshock];
            let mut ub = xj * (1.0 + xj.signum() * width);
            let mut lb = xj * (1.0 - xj.signum() * width);
            if b.ub[j + nshock] >= xj && b.lb[j + nshock] <= xj {
                ub = b.ub[j + nshock].min(ub);
                lb = b.lb[j + nshock].max(lb);
            } else {
                println!(
                    "\nstab_map_:: the calibrated value of param {} for neighborhood_width sampling is outside prior bounds.\nWe allow violation of bounds for this parameter, but if this was not done on purpose, please change calibration before running neighborhood_width sampling",
                    j + nshock + 1
                );
            }
            for row in lpmat.iter_mut() {
                row[j] = row[j] * (ub - lb) + lb;
            }
        }
    } else if let Some((x, hh)) = spec.posterior {
        // d = chol(inv(hh)); lp = randn(Nsam*2, nshock+np)*d + kron(ones(Nsam*2, 1), xparam1')
        // (MATLAB's randn ziggurat is not ported: the normal draws are the inverse normal CDF of the
        // MATLAB uniform stream, filled column by column like randn)
        let n = nshock + np;
        let Some(inv) = hh.clone().try_inverse() else {
            bail!("sensitivity: the Hessian hh of the mode file is singular")
        };
        let sym = (&inv + inv.transpose()) * 0.5;
        let Some(ch) = nalgebra::Cholesky::new(sym) else {
            bail!("sensitivity: chol(inv(hh)): the Hessian of the mode file is not positive definite")
        };
        let l = ch.l();
        let m2 = 2 * nsam;
        let mut z = vec![vec![0.0; n]; m2];
        for j in 0..n {
            for row in z.iter_mut() {
                row[j] = norminv(rng.rand(), 0.0, 1.0);
            }
        }
        let mut lp: Sample = Vec::new();
        for zr in &z {
            // row vector z*d = (L z')' with d = L'
            let row: Vec<f64> = (0..n).map(|c| x[c] + (0..n).map(|k| l[(c, k)] * zr[k]).sum::<f64>()).collect();
            // lnprior(j) = any(lp(j,:)' <= bounds.lb | lp(j,:)' >= bounds.ub)
            if !(0..n).any(|c| row[c] <= b.lb[c] || row[c] >= b.ub[c]) {
                lp.push(row);
            }
        }
        nsam = nsam.min(lp.len());
        lpmat0 = lp[..nsam].iter().map(|r| r[..nshock].to_vec()).collect();
        lpmat = lp[..nsam].iter().map(|r| r[nshock..].to_vec()).collect();
    } else {
        bail!("sensitivity: sampling around the posterior mode (pprior=0) needs a mode_file or neighborhood_width");
    }
    Ok((lpmat0, lpmat))
}

/// `+gsa/prior_draw.m`: inverse prior CDF on the truncated range `[lbcum, ubcum]`.
pub struct PriorDraw<'a> {
    priors: &'a [Prior],
    lbcum: Vec<f64>,
    ubcum: Vec<f64>,
    p3: Vec<f64>,
    p4: Vec<f64>,
}

/// MATLAB `gamcdf(x, a, b)`.
fn gamcdf(x: f64, a: f64, b: f64) -> f64 {
    gammainc(x / b, a)
}

fn wblcdf(x: f64, scale: f64, shape: f64) -> f64 {
    if x <= 0.0 { 0.0 } else { 1.0 - (-(x / scale).powf(shape)).exp() }
}

impl<'a> PriorDraw<'a> {
    pub fn new(priors: &'a [Prior], bounds: &Bounds) -> Self {
        let n = priors.len();
        let (mut lbcum, mut ubcum) = (vec![0.0; n], vec![1.0; n]);
        let mut p3: Vec<f64> = priors.iter().map(|p| p.p3).collect();
        let mut p4: Vec<f64> = priors.iter().map(|p| p.p4).collect();
        for (i, p) in priors.iter().enumerate() {
            let (lb, ub) = (bounds.lb[i], bounds.ub[i]);
            let (p6, p7) = (p.p6, p.p7);
            match p.shape {
                PriorShape::Beta => {
                    lbcum[i] = betainc((lb - p3[i]) / (p4[i] - p3[i]), p6, p7);
                    ubcum[i] = betainc((ub - p3[i]) / (p4[i] - p3[i]), p6, p7);
                }
                PriorShape::Gamma => {
                    lbcum[i] = gamcdf(lb - p3[i], p6, p7);
                    ubcum[i] = gamcdf(ub - p3[i], p6, p7);
                }
                PriorShape::Normal => {
                    lbcum[i] = 0.5 * statrs::function::erf::erfc(-(lb - p6) / p7 / std::f64::consts::SQRT_2);
                    ubcum[i] = 0.5 * statrs::function::erf::erfc(-(ub - p6) / p7 / std::f64::consts::SQRT_2);
                }
                PriorShape::InvGamma1 => {
                    lbcum[i] = gamcdf(1.0 / (ub - p3[i]).powi(2), p7 / 2.0, 2.0 / p6);
                    ubcum[i] = gamcdf(1.0 / (lb - p3[i]).powi(2), p7 / 2.0, 2.0 / p6);
                }
                PriorShape::Uniform => {
                    p4[i] = p4[i].min(ub);
                    p3[i] = p3[i].max(lb);
                }
                PriorShape::InvGamma2 => {
                    lbcum[i] = gamcdf(1.0 / (ub - p3[i]), p7 / 2.0, 2.0 / p6);
                    ubcum[i] = gamcdf(1.0 / (lb - p3[i]), p7 / 2.0, 2.0 / p6);
                }
                PriorShape::Weibull => {
                    lbcum[i] = wblcdf(lb - p3[i], p7, p6);
                    ubcum[i] = wblcdf(ub - p3[i], p7, p6);
                }
            }
        }
        PriorDraw { priors, lbcum, ubcum, p3, p4 }
    }

    /// Parameter `i` at unit-cube coordinate `u`.
    pub fn draw(&self, i: usize, u: f64) -> f64 {
        let p = &self.priors[i];
        let r = u * (self.ubcum[i] - self.lbcum[i]) + self.lbcum[i];
        let (p3, p4, p6, p7) = (self.p3[i], self.p4[i], p.p6, p.p7);
        match p.shape {
            PriorShape::Uniform => r * (p4 - p3) + p3,
            PriorShape::Normal => norminv(r, p6, p7),
            PriorShape::Gamma => gaminv(r, p6, p7) + p3,
            PriorShape::Beta => betainv(r, p6, p7) * (p4 - p3) + p3,
            PriorShape::InvGamma1 => (1.0 / gaminv(r, p7 / 2.0, 2.0 / p6)).sqrt() + p3,
            PriorShape::InvGamma2 => 1.0 / gaminv(r, p7 / 2.0, 2.0 / p6) + p3,
            PriorShape::Weibull => wblinv(r, p7, p6) + p3,
        }
    }
}

/// `+gsa/priorcdf.m` for one parameter.
pub fn priorcdf(p: &Prior, x: f64) -> f64 {
    let (p3, p4, p6, p7) = (p.p3, p.p4, p.p6, p.p7);
    match p.shape {
        PriorShape::Beta => betainc((x - p3) / (p4 - p3), p6, p7),
        PriorShape::Gamma => gamcdf(x - p3, p6, p7),
        PriorShape::Normal => 0.5 * statrs::function::erf::erfc(-(x - p6) / p7 / std::f64::consts::SQRT_2),
        PriorShape::InvGamma1 => gamcdf(1.0 / (x - p3).powi(2), p7 / 2.0, 2.0 / p6),
        PriorShape::Uniform => (x - p3) / (p4 - p3),
        PriorShape::InvGamma2 => gamcdf(1.0 / (x - p3), p7 / 2.0, 2.0 / p6),
        PriorShape::Weibull => wblcdf(x - p3, p7, p6),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colon_like_matlab() {
        assert_eq!(colon(0.0, 0.2, 0.4), vec![0.0, 0.2, 0.4]);
        assert_eq!(colon(0.0, 1.0 / 3.0, 1.0).len(), 4);
        assert_eq!(*colon(0.0, 1.0 / 3.0, 1.0).last().unwrap(), 1.0);
    }
}
