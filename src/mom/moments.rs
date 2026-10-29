// SPDX-License-Identifier: GPL-3.0-or-later
//! Moment bookkeeping of the `+mom` package: `matched_moments_block.m`,
//! `get_data_moments.m`, `optimal_weighting_matrix.m`, and the first-order branch of
//! `+pruned_SS/pruned_state_space_system.m` (GMM at `order = 1`).

use anyhow::{Result, bail};

use super::MatchedMoment;
use crate::analysis::moments::LyapunovSchur;
use crate::linalg::{Mat, chol_upper, mul, solve, trsm_left};
use crate::solver::DecisionRule;

/// Which estimator (`options_mom_.mom.mom_method`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Gmm,
    Smm,
    IrfMatching,
}

impl Method {
    pub fn name(self) -> &'static str {
        match self {
            Method::Gmm => "GMM",
            Method::Smm => "SMM",
            Method::IrfMatching => "IRF_MATCHING",
        }
    }
}

/// Port of `matched_moments_block.m`. Returns the rows used in estimation (GMM: expanded
/// powers, leads/lags shifted so the largest is 0 and sorted in descending order; SMM: the
/// original rows) and the 1-based indices of the removed duplicates.
pub fn matched_moments_block(rows: &[MatchedMoment], method: Method) -> Result<(Vec<MatchedMoment>, Vec<usize>)> {
    // the preprocessor hands over the factors of a product in its own canonical order (observed
    // with `dynare-preprocessor`: by |lag|, then lag, then variable; equal factors keep the order
    // written), which shows in the labels `E[io*io(-1)]` and decides which duplicates are found
    let canon: Vec<MatchedMoment> = rows
        .iter()
        .map(|m| {
            let mut idx: Vec<usize> = (0..m.vars.len()).collect();
            idx.sort_by_key(|&i| (m.lags[i].abs(), m.lags[i], m.vars[i]));
            MatchedMoment {
                vars: idx.iter().map(|&i| m.vars[i]).collect(),
                lags: idx.iter().map(|&i| m.lags[i]).collect(),
                powers: idx.iter().map(|&i| m.powers[i]).collect(),
            }
        })
        .collect();
    let rows = &canon[..];
    let psum = |m: &MatchedMoment| m.powers.iter().map(|&p| p as usize).sum::<usize>();
    if method == Method::Gmm && rows.iter().any(|m| psum(m) > 2) {
        bail!(
            "method_of_moments: GMM does not yet support product moments higher than 2. Change your 'matched_moments' block!"
        );
    }
    let transformed: Vec<MatchedMoment> = rows
        .iter()
        .map(|m| {
            let mut t = if m.powers.iter().any(|&p| p > 1) {
                let mut e = MatchedMoment::default();
                for j in 0..m.vars.len() {
                    for _ in 0..m.powers[j] {
                        e.vars.push(m.vars[j]);
                        e.lags.push(m.lags[j]);
                        e.powers.push(1);
                    }
                }
                e
            } else {
                m.clone()
            };
            let mx = t.lags.iter().copied().max().unwrap_or(0);
            for l in &mut t.lags {
                *l -= mx;
            }
            // MATLAB sort(...,'descend') is stable
            let mut idx: Vec<usize> = (0..t.lags.len()).collect();
            idx.sort_by(|&a, &b| t.lags[b].cmp(&t.lags[a]));
            MatchedMoment {
                vars: idx.iter().map(|&i| t.vars[i]).collect(),
                lags: idx.iter().map(|&i| t.lags[i]).collect(),
                powers: idx.iter().map(|&i| t.powers[i]).collect(),
            }
        })
        .collect();
    // unique rows within each power group (first occurrence kept)
    let mut keep = vec![false; rows.len()];
    for i in 0..rows.len() {
        let dup = (0..i)
            .any(|j| keep[j] && psum(&transformed[j]) == psum(&transformed[i]) && transformed[j] == transformed[i]);
        keep[i] = !dup;
    }
    let removed: Vec<usize> = (0..rows.len()).filter(|&i| !keep[i]).map(|i| i + 1).collect();
    let out = (0..rows.len())
        .filter(|&i| keep[i])
        .map(|i| if method == Method::Smm { rows[i].clone() } else { transformed[i].clone() })
        .collect();
    Ok((out, removed))
}

/// A moment is a first moment when its powers sum to 1 (`first_moment_indicator`).
pub fn is_first_moment(m: &MatchedMoment) -> bool {
    m.powers.iter().map(|&p| p as i64).sum::<i64>() == 1
}

/// Port of `get_data_moments.m`: `data` is T x nobs, `col[v]` the data column of
/// endogenous variable `v` (declaration index). Returns the moments (means omitting NaN)
/// and `m_data` (T x nmom, NaN replaced by the mean).
pub fn get_data_moments(data: &Mat, col: &dyn Fn(usize) -> usize, rows: &[MatchedMoment]) -> (Vec<f64>, Mat) {
    let t = data.nrows() as i64;
    let mut moments = vec![f64::NAN; rows.len()];
    let mut m_data = Mat::from_element(data.nrows(), rows.len(), f64::NAN);
    for (jm, m) in rows.iter().enumerate() {
        let mut tmp = vec![0.0; t as usize];
        for jv in 0..m.vars.len() {
            let c = col(m.vars[jv]);
            let ll = m.lags[jv] as i64;
            let mut y = vec![f64::NAN; t as usize];
            let (lo, hi) = (1 - ll.min(0), t - ll.max(0)); // 1-based target range
            let src0 = 1 + ll.max(0);
            let mut k = 0;
            let mut r = lo;
            while r <= hi {
                let v = data[((src0 + k - 1) as usize, c)];
                y[(r - 1) as usize] = v.powi(m.powers[jv] as i32);
                r += 1;
                k += 1;
            }
            if jv == 0 {
                tmp = y;
            } else {
                for (a, b) in tmp.iter_mut().zip(&y) {
                    *a *= b;
                }
            }
        }
        let (s, n) = tmp.iter().filter(|v| !v.is_nan()).fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
        let mean = s / n as f64;
        moments[jm] = mean;
        for (r, v) in tmp.iter().enumerate() {
            m_data[(r, jm)] = if v.is_nan() { mean } else { *v };
        }
    }
    (moments, m_data)
}

/// MATLAB `S \ eye(n)` for a symmetric matrix with positive diagonal: Cholesky when it
/// succeeds (what `mldivide` tries first), LU otherwise.
pub fn inv_sym(s: &Mat) -> Option<Mat> {
    let n = s.nrows();
    let diag_pos = (0..n).all(|i| s[(i, i)] > 0.0);
    if diag_pos && *s == s.transpose() {
        if let Some(r) = chol_upper(s) {
            let mut x = Mat::identity(n, n);
            trsm_left(true, true, 1.0, &r, &mut x);
            trsm_left(true, false, 1.0, &r, &mut x);
            return Some(x);
        }
    }
    solve(s, &Mat::identity(n, n))
}

/// Port of `optimal_weighting_matrix.m` (Newey–West with a Bartlett kernel of `q_lag` lags).
pub fn optimal_weighting_matrix(m_data: &Mat, moments: &[f64], q_lag: usize) -> Result<Mat> {
    let (t, nm) = (m_data.nrows(), m_data.ncols());
    let h = Mat::from_fn(t, nm, |r, c| m_data[(r, c)] - moments[c]);
    let corr = |v: usize| -> Mat {
        let mut g = Mat::zeros(nm, nm);
        for tt in v..t {
            for j in 0..nm {
                let b = h[(tt, j)];
                for i in 0..nm {
                    g[(i, j)] += h[(tt - v, i)] * b;
                }
            }
        }
        g / t as f64
    };
    let mut s = corr(0);
    for ii in 1..=q_lag {
        let g = corr(ii);
        let w = 1.0 - ii as f64 / (q_lag as f64 + 1.0);
        s += (&g + g.transpose()) * w;
    }
    let Some(w) = inv_sym(&s) else {
        bail!("optimal_weighting_matrix: the long-run covariance of the moments is singular")
    };
    Ok((&w + w.transpose()) / 2.0)
}

/// First and second moments of the selected variables (`pruned_state_space_system.m`).
#[derive(Clone, Debug)]
pub struct StateSpaceMoments {
    pub e_y: Vec<f64>,
    pub var_y: Mat,
    /// `var_yi[i-1](a, b) = Cov(y_a(t), y_b(t-i))`.
    pub var_yi: Vec<Mat>,
}

/// Port of `pruned_state_space_system(M_, options_, dr, indy, nlags, 0, 0)` at `order = 1`:
/// `indy` are DR positions. Unit roots are handled as Dynare does (Schur vectors of
/// `lyapunov_symm`; nonstationary variables get NaN moments).
pub fn first_order_moments(
    dr: &DecisionRule,
    sigma_e: &Mat,
    indy: &[usize],
    nlags: usize,
    qz_criterium: f64,
    schur_vec_tol: f64,
) -> Result<StateSpaceMoments> {
    let indx: Vec<usize> = (dr.nstatic..dr.nstatic + dr.nspred).collect();
    let rows = |m: &Mat, r: &[usize]| m.select_rows(r);
    let hx = rows(&dr.ghx, &indx);
    let hu = rows(&dr.ghu, &indx);
    let gx = rows(&dr.ghx, indy);
    let gu = rows(&dr.ghu, indy);
    let ny = indy.len();
    let om_z = mul(&mul(&hu, false, sigma_e, false), false, &hu, true);
    let ly = LyapunovSchur::new(&hx, qz_criterium)?;
    let var_z = ly.solve(&om_z, 1e-15);
    let stationary: Vec<usize> = if ly.u_unit.ncols() > 0 {
        let x = mul(&gx, false, &ly.u_unit, false);
        (0..ny).filter(|&i| (0..x.ncols()).all(|j| x[(i, j)].abs() < schur_vec_tol)).collect()
    } else {
        (0..ny).collect()
    };
    let st = &stationary;
    let cs = gx.select_rows(st);
    let ds = gu.select_rows(st);
    let mut v = mul(&mul(&cs, false, &var_z, false), false, &cs, true)
        + mul(&mul(&ds, false, sigma_e, false), false, &ds, true);
    v.apply(|x| {
        if x.abs() < 1e-12 {
            *x = 0.0
        }
    });
    let mut var_y = Mat::from_element(ny, ny, f64::NAN);
    for (a, &i) in st.iter().enumerate() {
        for (b, &j) in st.iter().enumerate() {
            var_y[(i, j)] = v[(a, b)];
        }
    }
    let tmp = mul(&mul(&hx, false, &var_z, false), false, &cs, true)
        + mul(&mul(&hu, false, sigma_e, false), false, &ds, true);
    let nz = hx.nrows();
    let mut ai = Mat::identity(nz, nz);
    let mut var_yi = Vec::with_capacity(nlags);
    for _ in 0..nlags {
        let vi = mul(&mul(&cs, false, &ai, false), false, &tmp, false);
        let mut m = Mat::from_element(ny, ny, f64::NAN);
        for (a, &i) in st.iter().enumerate() {
            for (b, &j) in st.iter().enumerate() {
                m[(i, j)] = vi[(a, b)];
            }
        }
        var_yi.push(m);
        ai = mul(&ai, false, &hx, false);
    }
    let e_y = (0..ny).map(|i| if st.contains(&i) { dr.ys[dr.order_var[indy[i]]] } else { f64::NAN }).collect();
    Ok(StateSpaceMoments { e_y, var_y, var_yi })
}

/// Model moments from the (pruned) state-space moments, as `mom.objective_function` (GMM):
/// `pos[v]` is the position in the selected (observed) variables of endogenous variable `v`.
pub fn gmm_model_moments(
    ss: &StateSpaceMoments,
    rows: &[MatchedMoment],
    pos: &dyn Fn(usize) -> usize,
    prefilter: bool,
) -> Vec<f64> {
    rows.iter()
        .map(|m| {
            let p: usize = m.powers.iter().map(|&p| p as usize).sum();
            if p == 1 {
                return ss.e_y[pos(m.vars[0])];
            }
            let (i1, i2) = (pos(m.vars[0]), pos(m.vars[1]));
            let mean = if prefilter { 0.0 } else { ss.e_y[i1] * ss.e_y[i2] };
            if m.lags.iter().all(|&l| l == 0) {
                ss.var_y[(i1, i2)] + mean
            } else {
                let lag = (-m.lags[1]) as usize;
                ss.var_yi[lag - 1][(i1, i2)] + mean
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mm(v: &[usize], l: &[i32], p: &[u32]) -> MatchedMoment {
        MatchedMoment { vars: v.to_vec(), lags: l.to_vec(), powers: p.to_vec() }
    }

    #[test]
    fn block_transform_and_duplicates() {
        // the manual's example: c*c(3), and y(1)^2*c(-4)^3 == c(-5)^3*y(0)^2 up to a shift
        let rows = vec![
            mm(&[0], &[0], &[1]),
            mm(&[0, 0], &[0, 3], &[1, 1]),
            mm(&[1, 0], &[1, -4], &[2, 3]),
            mm(&[0, 1], &[-5, 0], &[3, 2]),
        ];
        let (g, removed) = matched_moments_block(&rows[..2], Method::Gmm).unwrap();
        assert!(removed.is_empty());
        assert_eq!(g[1], mm(&[0, 0], &[0, -3], &[1, 1]));
        let (s, removed) = matched_moments_block(&rows, Method::Smm).unwrap();
        assert_eq!(removed, vec![4]);
        assert_eq!(s.len(), 3);
        assert_eq!(s[2], rows[2]); // SMM keeps the original rows
        assert!(matched_moments_block(&rows, Method::Gmm).is_err());
    }

    #[test]
    fn data_moments_with_lags() {
        let data = Mat::from_column_slice(4, 1, &[1.0, 2.0, 3.0, 4.0]);
        let (m, md) = get_data_moments(&data, &|_| 0, &[mm(&[0, 0], &[0, -1], &[1, 1])]);
        assert_eq!(m[0], (2.0 + 6.0 + 12.0) / 3.0);
        assert_eq!(md[(0, 0)], m[0]);
        assert_eq!(md[(3, 0)], 12.0);
    }
}
