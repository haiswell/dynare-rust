// SPDX-License-Identifier: GPL-3.0-or-later
//! Analytic Jacobian of the GMM model moments with respect to the estimated parameters
//! (`analytic_standard_errors`, `analytic_jacobian`): port of the `compute_derivs` branch of
//! `+mom/objective_function.m` (`model_moments_params_derivs`, `df`).
//!
//! At order 1 the derivatives of `E_y`, `Var_y` and `Var_yi` of `pruned_state_space_system`
//! (`compute_derivs = 1`) are computed analytically from `dr.derivs`
//! (`identification::derivs::perturbation_derivs`, `analytic_derivation_mode = 0`) and the
//! derivative of the Lyapunov equation. At order 2 and 3 the pruned-state-space moments depend
//! on the higher-order solution through code that has no analytic parameter derivative here;
//! the Jacobian is then a Richardson-extrapolated central difference (error O(h^4), agreeing
//! with Dynare's analytic derivatives to about 1e-9), the same substitute as
//! `identification/korder.rs`.

use super::MatchedMoment;
use super::problem::{Fail, MomProblem, Solution};
use crate::analysis::moments::LyapunovSchur;
use crate::identification::derivs::{ModelParamDerivs, ParamSet, perturbation_derivs};
use crate::linalg::{Mat, mul};
use crate::modfile::EstimatedKind;

/// `A B C'`.
fn abct(a: &Mat, b: &Mat, c: &Mat) -> Mat {
    mul(&mul(a, false, b, false), false, c, true)
}

const NUM_FAIL: Fail = Fail { info: 9, info4: 0.1 };

impl MomProblem {
    /// `d model_moments / d x` (`mom_nbr x dim`), GMM only.
    pub fn moments_jacobian(&self, x: &[f64]) -> Result<Mat, Fail> {
        if let Some(j) = self.moments_jacobian_order1(x)? {
            return Ok(j);
        }
        self.moments_jacobian_richardson(x)
    }

    fn moments_jacobian_richardson(&self, x: &[f64]) -> Result<Mat, Fail> {
        let n = x.len();
        let central = |h: &[f64]| -> Result<Mat, Fail> {
            let mut cols = Vec::new();
            for i in 0..n {
                let mut xp = x.to_vec();
                xp[i] += h[i];
                let mut xm = x.to_vec();
                xm[i] -= h[i];
                let (mp, _) = self.model_moments(&xp)?;
                let (mm, _) = self.model_moments(&xm)?;
                cols.push(mp.iter().zip(&mm).map(|(a, b)| (a - b) / (2.0 * h[i])).collect::<Vec<f64>>());
            }
            Ok(Mat::from_fn(cols[0].len(), n, |r, c| cols[c][r]))
        };
        let h: Vec<f64> = x.iter().map(|v| 5e-4 * v.abs().max(1e-2)).collect();
        let h2: Vec<f64> = h.iter().map(|v| v / 2.0).collect();
        let (d1, d2) = (central(&h)?, central(&h2)?);
        Ok((&d2 * 4.0 - d1) / 3.0)
    }

    /// Order-1 analytic Jacobian; `None` when not applicable (order > 1, unit roots).
    fn moments_jacobian_order1(&self, x: &[f64]) -> Result<Option<Mat>, Fail> {
        if self.order > 1 {
            return Ok(None);
        }
        let (params, sigma_e, sol) = self.solve(x)?;
        let Solution::First(dr) = &sol else { return Ok(None) };
        let model = &self.model;
        let nx = x.len();
        // ParamSet in Dynare's order (stderr, corr, model) and the xparam position of each column
        let mut set = ParamSet::default();
        let (mut cs, mut cc, mut cm) = (Vec::new(), Vec::new(), Vec::new());
        for (k, e) in self.estimated.iter().enumerate() {
            match e.kind {
                EstimatedKind::StdErr(i) => {
                    set.stderr.push(i);
                    cs.push(k)
                }
                EstimatedKind::Corr(i, j) => {
                    set.corr.push((i, j));
                    cc.push(k)
                }
                EstimatedKind::Param(i) => {
                    set.model.push(i);
                    cm.push(k)
                }
                _ => return Ok(None),
            }
        }
        let colmap: Vec<usize> = cs.into_iter().chain(cc).chain(cm).collect();
        let g1 = model.dynamic_jacobian(&dr.ys, &self.exo_ss, &params);
        let pd = self.pderivs.get_or_init(|| ModelParamDerivs::new(model).at_exo_steady_state(&self.exo_ss));
        let md = pd.eval(model, &dr.ys, &params, self.qz_criterium).map_err(|_| NUM_FAIL)?;
        let d = perturbation_derivs(model, dr, &g1, &md, &sigma_e, &set).map_err(|_| NUM_FAIL)?;
        let nm_off = set.n_shock();

        let indy = self.obs_var(dr);
        let indx: Vec<usize> = (dr.nstatic..dr.nstatic + dr.nspred).collect();
        let hx = dr.ghx.select_rows(&indx);
        let hu = dr.ghu.select_rows(&indx);
        let gx = dr.ghx.select_rows(&indy);
        let gu = dr.ghu.select_rows(&indy);
        let ly = LyapunovSchur::new(&hx, self.qz_criterium).map_err(|_| NUM_FAIL)?;
        if ly.u_unit.ncols() > 0 {
            return Ok(None);
        }
        let om_z = abct(&hu, &sigma_e, &hu);
        let vz = ly.solve(&om_z, 1e-15);
        let tmp = abct(&hx, &vz, &gx) + abct(&hu, &sigma_e, &gu);
        let ny = indy.len();
        let e_y: Vec<f64> = (0..ny).map(|i| dr.ys[dr.order_var[indy[i]]]).collect();
        let nlags = self.nlags;
        let nz = hx.nrows();
        let tot = set.len();
        let mut de = vec![vec![0.0; ny]; nx];
        let mut dvar = vec![Mat::zeros(ny, ny); nx];
        let mut dvari = vec![vec![Mat::zeros(ny, ny); nlags]; nx];
        for c in 0..tot {
            let xi = colmap[c];
            let dhx = d.dghx[c].select_rows(&indx);
            let dhu = d.dghu[c].select_rows(&indx);
            let dgx = d.dghx[c].select_rows(&indy);
            let dgu = d.dghu[c].select_rows(&indy);
            let ds = &d.dsigma_e[c];
            if c >= nm_off {
                for i in 0..ny {
                    de[xi][i] = d.d_yss[(indy[i], c - nm_off)];
                }
            }
            let hsd = abct(&dhu, &sigma_e, &hu);
            let d_om = &hsd + hsd.transpose() + abct(&hu, ds, &hu);
            let t1 = abct(&dhx, &vz, &hx);
            let rhs = &t1 + t1.transpose() + d_om;
            let dvz = ly.solve(&rhs, 1e-15);
            let a = abct(&dgx, &vz, &gx);
            let b = abct(&dgu, &sigma_e, &gu);
            dvar[xi] = &a + a.transpose() + abct(&gx, &dvz, &gx) + &b + b.transpose() + abct(&gu, ds, &gu);
            let dtmp = abct(&dhx, &vz, &gx)
                + abct(&hx, &dvz, &gx)
                + abct(&hx, &vz, &dgx)
                + abct(&dhu, &sigma_e, &gu)
                + abct(&hu, ds, &gu)
                + abct(&hu, &sigma_e, &dgu);
            let mut ai = Mat::identity(nz, nz);
            let mut dai = Mat::zeros(nz, nz);
            for l in 0..nlags {
                dvari[xi][l] = mul(&mul(&dgx, false, &ai, false), false, &tmp, false)
                    + mul(&mul(&gx, false, &dai, false), false, &tmp, false)
                    + mul(&mul(&gx, false, &ai, false), false, &dtmp, false);
                dai = mul(&dai, false, &hx, false) + mul(&ai, false, &dhx, false);
                ai = mul(&ai, false, &hx, false);
            }
        }
        let rows: &[MatchedMoment] = &self.rows;
        let pos = |v: usize| self.varobs.iter().position(|&o| o == v).expect("observed variable");
        let mut jac = Mat::zeros(rows.len(), nx);
        for (r, m) in rows.iter().enumerate() {
            let p: usize = m.powers.iter().map(|&p| p as usize).sum();
            let i1 = pos(m.vars[0]);
            for xi in 0..nx {
                jac[(r, xi)] = if p == 1 {
                    de[xi][i1]
                } else {
                    let i2 = pos(m.vars[1]);
                    let mean = if self.prefilter { 0.0 } else { de[xi][i1] * e_y[i2] + e_y[i1] * de[xi][i2] };
                    if m.lags.iter().all(|&l| l == 0) {
                        dvar[xi][(i1, i2)] + mean
                    } else {
                        dvari[xi][(-m.lags[1]) as usize - 1][(i1, i2)] + mean
                    }
                };
            }
        }
        Ok(Some(jac))
    }

    /// Gradient of the GMM objective (`df` of `mom.objective_function`, `vector_output = 0`):
    /// `d fval / dx_j = 2 res' dres_j`, plus the penalty term of the penalized estimator.
    pub fn objective_gradient(&self, x: &[f64], data: &[f64], sw: &Mat) -> Option<Vec<f64>> {
        let e = self.evaluate(x, data, sw).ok()?;
        let jac = self.moments_jacobian(x).ok()?;
        let dres = -mul(&(sw * self.scaling.sqrt()), false, &jac, false);
        let mut g: Vec<f64> = (0..x.len())
            .map(|j| 2.0 * (0..e.residuals.len()).map(|k| dres[(k, j)] * e.residuals[k]).sum::<f64>())
            .collect();
        if let Some((p1, p2)) = &self.penalty {
            for j in 0..x.len() {
                g[j] += 2.0 * (x[j] - p1[j]) / (p2[j] * p2[j]);
            }
        }
        g.iter().all(|v| v.is_finite()).then_some(g)
    }

    /// Jacobian of the lsqnonlin residual vector (`df` with `vector_output = 1`).
    pub fn residual_jacobian(&self, x: &[f64], sw: &Mat) -> Option<Mat> {
        let jac = self.moments_jacobian(x).ok()?;
        let dres = -mul(&(sw * self.scaling.sqrt()), false, &jac, false);
        let n = x.len();
        Some(match &self.penalty {
            None => dres,
            Some((_, p2)) => Mat::from_fn(dres.nrows() + n, n, |r, c| {
                if r < dres.nrows() {
                    dres[(r, c)]
                } else if r - dres.nrows() == c {
                    1.0 / p2[c]
                } else {
                    0.0
                }
            }),
        })
    }
}
