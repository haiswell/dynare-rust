// SPDX-License-Identifier: GPL-3.0-or-later
//! `heteroskedastic_filter`: per-period shock covariances.
//!
//! * [`HeteroSpec::new`]: `M_.heteroskedastic_shocks.Qvalue` / `Qscale` as
//!   `dynare_estimation_init.m` builds them from the `heteroskedastic_shocks` block
//!   (exo_nbr x (nobs+1); integer periods of the original dataset, column
//!   `period - first_obs`, values and scales squared).
//! * [`HeteroSpec::qvec`]: port of `kalman/get_Qvec_heteroskedastic_filter.m`.
//!
//! The filters ([`super::KalmanWorkspace::set_heteroskedastic`]) and the smoother
//! ([`super::smoother_heteroskedastic`]) take the resulting `Qvec` (k = 0..=nobs).

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::collapsible_if
)] // index loops mirror the MATLAB code

use anyhow::{Result, bail};

use crate::linalg::Mat;
use crate::modfile::HeteroShock;

/// `M_.heteroskedastic_shocks.{Qvalue, Qscale}` (NaN = not set).
#[derive(Clone, Debug)]
pub struct HeteroSpec {
    pub qvalue: Mat,
    pub qscale: Mat,
}

impl HeteroSpec {
    /// From the declarations, for a sample of `nobs` periods starting at `first_obs` (1-based).
    pub fn new(
        decl: &[HeteroShock],
        n_exo: usize,
        first_obs: usize,
        nobs: usize,
        exo_names: &[String],
        init: crate::dates::Date,
    ) -> Result<Self> {
        let mut qvalue = Mat::from_element(n_exo, nobs + 1, f64::NAN);
        let mut qscale = Mat::from_element(n_exo, nobs + 1, f64::NAN);
        let fo = first_obs as i64;
        for d in decl {
            // mask_periods = periods >= first_obs & periods <= nobs+first_obs
            let mut cols: Vec<usize> =
                d.periods.iter().filter(|&&p| p >= fo && p <= nobs as i64 + fo).map(|&p| (p - fo) as usize).collect();
            // dates: periods >= dataset_.dates(1) & periods <= dataset_.dates(end)
            for &t in &d.dates {
                let Some(c) = init.diff(t) else {
                    bail!(
                        "heteroskedastic_shocks: dates in \"periods\" do not have the same frequency as those in the dataset"
                    )
                };
                if c >= 0 && c < nobs as i64 {
                    cols.push(c as usize);
                }
            }
            for c in cols {
                if d.is_value {
                    qvalue[(d.exo, c)] = d.value * d.value;
                } else {
                    qscale[(d.exo, c)] = d.value * d.value;
                }
            }
        }
        let zero_first: Vec<&str> =
            (0..n_exo).filter(|&i| qscale[(i, 0)] == 0.0).map(|i| exo_names[i].as_str()).collect();
        if !zero_first.is_empty() {
            println!(
                "\ndynare_estimation_init: WARNING: The scale for shock(s) {} is set to zero in the first period.",
                zero_first.join(", ")
            );
            println!(
                "dynare_estimation_init: During Kalman filter initialization, these shocks will still be considered non-zero (using the base variance from M_.Sigma_e)."
            );
            println!("dynare_estimation_init: Check whether that is desired behavior.");
        }
        if qvalue.iter().zip(qscale.iter()).any(|(v, s)| !v.is_nan() && !s.is_nan()) {
            println!("\ndynare_estimation_init: With the option \"heteroskedastic_shocks\" you cannot define");
            println!("dynare_estimation_init: the scale and the value for the same shock ");
            println!("dynare_estimation_init: in the same period!");
            bail!("Scale and value defined for the same shock in the same period with \"heteroskedastic_shocks\".");
        }
        Ok(HeteroSpec { qvalue, qscale })
    }

    /// Port of get_Qvec_heteroskedastic_filter.m: `Qvec(:,:,k)`, k = 0..=smpl.
    pub fn qvec(&self, q: &Mat, smpl: usize) -> Vec<Mat> {
        let n = q.nrows();
        let isqdiag = (0..n).all(|i| (0..n).all(|j| i == j || q[(i, j)].abs() < 1e-14));
        (0..=smpl)
            .map(|k| {
                let mut qk = q.clone();
                let col = |m: &Mat, i: usize| if k < m.ncols() { m[(i, k)] } else { f64::NAN };
                for i in 0..n {
                    let v = col(&self.qvalue, i);
                    if v.is_nan() {
                        continue;
                    }
                    if isqdiag {
                        qk[(i, i)] = v;
                    } else if q[(i, i)] > 1e-14 {
                        let s = (v / q[(i, i)]).sqrt();
                        for j in 0..n {
                            qk[(i, j)] *= s;
                        }
                        for j in 0..n {
                            qk[(j, i)] *= s;
                        }
                    } else {
                        qk[(i, i)] = v;
                    }
                }
                for i in 0..n {
                    let s2 = col(&self.qscale, i);
                    if s2.is_nan() {
                        continue;
                    }
                    if isqdiag {
                        qk[(i, i)] *= s2;
                    } else {
                        let s = s2.sqrt();
                        for j in 0..n {
                            qk[(i, j)] *= s;
                        }
                        for j in 0..n {
                            qk[(j, i)] *= s;
                        }
                    }
                }
                qk
            })
            .collect()
    }
}
