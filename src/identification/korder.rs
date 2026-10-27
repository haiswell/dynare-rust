// SPDX-License-Identifier: GPL-3.0-or-later
//! identification at `order = 2 | 3` (`identification.get_jacobians` on the pruned
//! state space system, Mutschler 2015).
//!
//! The Jacobians are the two-sided finite differences of Dynare's `analytic_derivation_mode =
//! -1` (`identification.fjaco` with `options_.dynatol.x = 1e-5`, `numerical_objective.m`,
//! `get_perturbation_params_derivs_numerical_objective.m`) of
//! - the reduced form `[Yss; vec(ghx); vech(Om); vec(ghxx); vec(ghxu); vec(ghuu); vec(ghs2)
//!   (; vec(ghxxx); ...; vec(ghuss))]`,
//! - the dynamic model `[Yss; vec(g1); vec(g2) (; vec(g2), sic: get_jacobians.m repeats dg2 at
//!   order 3, while the values `DYNAMIC` end with `vec(g3)`)]` in the legacy
//!   (lead_lag_incidence) column layout,
//! - the moments of the pruned state space `[E_y; vech(Var_y); vec(Var_yi or Corr_yi)]`,
//! - its spectral density on `0 : pi/(grid_nbr/2) : pi` (`Gbar`, Gram matrix of the complex
//!   Jacobian),
//! - the first-order minimal system (Komunjer and Ng) with the k-th order accurate mean.
//!
//! Dynare computes the mean's Jacobian (and, with `analytic_derivation_mode = 0`, all others)
//! analytically; the finite differences agree with those to O(h^2) (~1e-9 relative here).

use anyhow::{Result, anyhow};
use nalgebra::{Complex, DMatrix};

use super::derivs::legacy_columns;
use super::moments::{pruned_state_space, vech};
use super::{Context, Jacobians, Point};
use crate::linalg::{Mat, mul};
use crate::solver::SolverOptions;
use crate::solver::korder::pruned::pruned_state_space_system_qz;
use crate::solver::korder::{KOrderDr, solve_model};

type C64 = Complex<f64>;

/// Everything differentiated at one parameter vector.
struct Objects {
    reducedform: Vec<f64>,
    dynamic: Vec<f64>,
    /// `[E_y]` and `[vech(Var_y); lags]` (MOMENTS without and with the mean).
    mean: Vec<f64>,
    moments: Vec<f64>,
    /// Per frequency, `vec(g_omega)` (obs^2), complex.
    spectrum: Vec<Vec<C64>>,
}

fn vec_of(m: &Mat) -> impl Iterator<Item = f64> + '_ {
    m.iter().copied()
}

impl Context<'_> {
    /// The k-order decision rule at `params`, `sigma_e`, with the steady state and the
    /// parameters as updated by a `steady_state_model` block.
    fn solve_k(&self, params: &[f64], sigma_e: &Mat) -> std::result::Result<(Vec<f64>, Vec<f64>, KOrderDr), i32> {
        let d = &self.dsge;
        let (ys, params) = super::steady_state_with_params(
            self.model,
            d.steady_state_model.as_ref(),
            params,
            &d.ys_guess,
            &d.exo_ss,
            &d.steady_opts,
        )
        .map_err(|e| e.info)?;
        let sopts = SolverOptions { qz_criterium: self.qz_criterium, ..SolverOptions::default() };
        let dr = solve_model(self.model, &ys, &params, sigma_e, self.opts.order, sopts)
            .map_err(|e| e.downcast_ref::<crate::solver::SolveError>().map(|s| s.info).unwrap_or(9))?;
        Ok((ys, params, dr))
    }

    /// `[g1; g2]` (and `g2` again at order 3) of the dynamic model at the steady state, legacy
    /// layout, declaration-order rows, vectorised column-major. With `g3_values` (order 3) the
    /// last block holds `vec(g3)` instead (`DYNAMIC` of get_jacobians.m, `legacy_dynamic_g3.m`:
    /// the unique derivatives only, not symmetrized), cut to the length of the `dDYNAMIC` rows
    /// it is indexed with.
    fn dynamic_k(
        &self,
        ys: &[f64],
        params: &[f64],
        deriv: &crate::model::higher::DynamicDerivatives,
        g3_values: bool,
    ) -> Vec<f64> {
        let n = self.model.n_endo();
        let cols = legacy_columns(self.model);
        let nv = cols.len();
        let mut inv = vec![usize::MAX; 3 * n + self.model.n_exo()];
        for (l, &c) in cols.iter().enumerate() {
            inv[c] = l;
        }
        let x = &self.dsge.exo_ss;
        let g1 = self.model.dynamic_jacobian(ys, x, params);
        let mut out: Vec<f64> = Vec::with_capacity(n * nv * (1 + nv));
        for &c in &cols {
            for i in 0..n {
                out.push(g1[(i, c)]);
            }
        }
        let vals = deriv.eval(ys, x, params);
        let mut g2 = vec![0.0; n * nv * nv];
        for (k, &(eq, v1, v2)) in deriv.g2_index.iter().enumerate() {
            let (l1, l2) = (inv[v1], inv[v2]);
            if l1 == usize::MAX || l2 == usize::MAX {
                continue;
            }
            // legacy_dynamic_g2.m: column var1*nvar + var2 (+ the symmetric element)
            g2[eq + n * (l1 * nv + l2)] = vals.g2[k];
            g2[eq + n * (l2 * nv + l1)] = vals.g2[k];
        }
        out.extend(g2.iter().copied());
        if self.opts.order == 3 {
            if g3_values {
                let d3 = self.model.dynamic_derivatives(3);
                let vals3 = d3.eval(ys, x, params);
                let mut g3 = vec![0.0; n * nv * nv];
                for (k, &(eq, v1, v2, v3)) in d3.g3_index.iter().enumerate() {
                    let (l1, l2, l3) = (inv[v1], inv[v2], inv[v3]);
                    if l1 == usize::MAX || l2 == usize::MAX || l3 == usize::MAX {
                        continue;
                    }
                    // legacy_dynamic_g3.m: column (var1*nvar + var2)*nvar + var3
                    let col = (l1 * nv + l2) * nv + l3;
                    if col < nv * nv {
                        g3[eq + n * col] = vals3.g3[k];
                    }
                }
                out.extend(g3);
            } else {
                // get_jacobians.m stacks dg2 twice at order 3 (sic)
                out.extend(g2.iter().copied());
            }
        }
        out
    }

    fn objects(
        &self,
        theta: &[f64],
        nlags: usize,
        spectrum: bool,
        deriv: &crate::model::higher::DynamicDerivatives,
        g3_values: bool,
    ) -> std::result::Result<Objects, i32> {
        let (params, sigma_e) = self.set.apply(theta, &self.dsge.base_params, &self.dsge.base_sigma_e);
        let (ys, params, dr) = self.solve_k(&params, &sigma_e)?;
        let f = &dr.first;
        let yss: Vec<f64> = f.order_var.iter().map(|&v| ys[v]).collect();
        let om = mul(&mul(&f.ghu, false, &sigma_e, false), false, &f.ghu, true);
        let mut rf = yss;
        rf.extend(vec_of(&f.ghx));
        rf.extend(vech(&om));
        for m in [&dr.ghxx, &dr.ghxu, &dr.ghuu, &dr.ghs2] {
            rf.extend(vec_of(m));
        }
        if self.opts.order == 3 {
            for m in [&dr.ghxxx, &dr.ghxxu, &dr.ghxuu, &dr.ghuuu, &dr.ghxss, &dr.ghuss] {
                rf.extend(vec_of(m));
            }
        }
        let mut dynamic: Vec<f64> = f.order_var.iter().map(|&v| ys[v]).collect();
        dynamic.extend(self.dynamic_k(&ys, &params, deriv, g3_values));
        let indy: Vec<usize> = self.varobs.iter().map(|&v| f.inv_order_var[v]).collect();
        let p = pruned_state_space_system_qz(&dr, &sigma_e, &indy, nlags, self.qz_criterium).map_err(|_| 9)?;
        let mut moments = vech(&p.var_y);
        let lags = if self.opts.useautocorr { &p.corr_yi } else { &p.var_yi };
        for l in lags.iter().take(nlags) {
            moments.extend(vec_of(l));
        }
        let mut spec = Vec::new();
        if spectrum {
            let pi = std::f64::consts::PI;
            let grid = self.opts.grid_nbr as f64;
            let freqs = crate::analysis::q2::matlab_colon(0.0, pi / (grid / 2.0), pi);
            let cz = |m: &Mat| DMatrix::<C64>::from_fn(m.nrows(), m.ncols(), |i, j| C64::new(m[(i, j)], 0.0));
            let (a, b, c, d, v) = (cz(&p.a), cz(&p.b), cz(&p.cmat), cz(&p.dmat), cz(&p.var_inov));
            let na = p.a.nrows();
            for w in freqs {
                let tpos = C64::new(w.cos(), w.sin());
                let m = DMatrix::<C64>::identity(na, na) * tpos - &a;
                let sol = m.lu().solve(&b).ok_or(9)?;
                let t = &d + &c * sol;
                let g = (&t * &v * t.adjoint()) / C64::new(2.0 * pi, 0.0);
                spec.push(g.iter().copied().collect());
            }
        }
        Ok(Objects { reducedform: rf, dynamic, mean: p.e_y.clone(), moments, spectrum: spec })
    }

    /// `identification.get_jacobians` at `order = 2 | 3` (see the module doc).
    pub(crate) fn jacobians_k(&self, pt: &Point, nlags: usize, spectrum: bool, minimal: bool) -> Result<Jacobians> {
        let o = &self.opts;
        let tot = self.set.len();
        let ns = self.set.n_shock();
        let nm = self.set.n_model();
        let deriv = self.model.dynamic_derivatives(2);
        let fail = |j: usize, info: i32| {
            anyhow!(
                "fjaco.m: numerical two-sided difference method yields errors in solution algorithm (cannot solve the model for {}, info = {info})",
                self.set.names.get(j).cloned().unwrap_or_default()
            )
        };
        let x0 = &pt.theta;
        let f0 = self.objects(x0, nlags, spectrum, &deriv, true).map_err(|i| fail(0, i))?;
        let tol = 1e-5; // options_.dynatol.x
        let nrf = f0.reducedform.len();
        let ndyn = f0.dynamic.len();
        let ny = f0.mean.len();
        let nmom = f0.moments.len();
        let mut d_rf = Mat::zeros(nrf, tot);
        let mut d_dyn = Mat::zeros(ndyn, nm);
        let mut d_mean = Mat::zeros(ny, tot);
        let mut d_mom = Mat::zeros(nmom, tot);
        let nf = f0.spectrum.len();
        let no2 = ny * ny;
        let mut d_spec: Vec<DMatrix<C64>> = vec![DMatrix::zeros(no2, tot); nf];
        for j in 0..tot {
            let h = tol * x0[j].abs().max(1.0);
            let (mut xp, mut xm) = (x0.clone(), x0.clone());
            xp[j] += h;
            xm[j] -= h;
            let hh = xp[j] - xm[j];
            let f1 = self.objects(&xp, nlags, spectrum, &deriv, false).map_err(|i| fail(j, i))?;
            let fm = self.objects(&xm, nlags, spectrum, &deriv, false).map_err(|i| fail(j, i))?;

            for r in 0..nrf {
                d_rf[(r, j)] = (f1.reducedform[r] - fm.reducedform[r]) / hh;
            }
            for r in 0..ny {
                d_mean[(r, j)] = (f1.mean[r] - fm.mean[r]) / hh;
            }
            for r in 0..nmom {
                d_mom[(r, j)] = (f1.moments[r] - fm.moments[r]) / hh;
            }
            if j >= ns {
                for r in 0..ndyn {
                    d_dyn[(r, j - ns)] = (f1.dynamic[r] - fm.dynamic[r]) / hh;
                }
            }
            for (k, ds) in d_spec.iter_mut().enumerate() {
                for r in 0..no2 {
                    ds[(r, j)] = (f1.spectrum[k][r] - fm.spectrum[k][r]) / C64::new(hh, 0.0);
                }
            }
        }
        // the stderr / corr rows of Yss are zero by construction (dREDUCEDFORM's first block)
        let n = self.model.n_endo();
        for j in 0..ns {
            for r in 0..n {
                d_rf[(r, j)] = 0.0;
            }
        }
        // MOMENTS = [MEAN; vech(E_yy); lags], dMOMENTS = [dMEAN; ...]
        let mut moments = f0.mean.clone();
        moments.extend(f0.moments.iter().copied());
        let mut d_moments = Mat::zeros(ny + nmom, tot);
        d_moments.rows_mut(0, ny).copy_from(&d_mean);
        d_moments.rows_mut(ny, nmom).copy_from(&d_mom);
        let (d_spectrum, d_spectrum_no_mean) = if spectrum {
            let mut g = DMatrix::<C64>::zeros(tot, tot);
            for (k, ds) in d_spec.iter().enumerate() {
                let w = if k == 0 { 1.0 } else { 2.0 };
                g += (ds.adjoint() * ds) * C64::new(w, 0.0);
            }
            let scale = 2.0 * std::f64::consts::PI / (2.0 * nf as f64 - 1.0);
            let g = Mat::from_fn(tot, tot, |a, b| g[(a, b)].re * scale);
            (Some(&g + mul(&d_mean, true, &d_mean, false)), Some(g))
        } else {
            (None, None)
        };
        // first-order objects for the minimal system (and the Jacobians struct)
        let derivs = self.perturbation_derivs(pt)?;
        let indy: Vec<usize> = self.varobs.iter().map(|&v| pt.dr.inv_order_var[v]).collect();
        let pruned1 = pruned_state_space(
            &pt.dr,
            &derivs,
            &pt.sigma_e,
            &indy,
            nlags,
            o.useautocorr,
            ns,
            self.qz_criterium,
            o.schur_vec_tol,
        )?;
        let d_minimal = if minimal {
            if indy.len() < self.model.n_exo() {
                print!(
                    "WARNING: Komunjer and Ng (2011) failed:\n       There are more shocks and measurement errors than observables, this is not implemented (yet).\n       Skip identification analysis based on minimal state space system.\n"
                );
                None
            } else {
                match super::minimal::minimal_state_representation(&pruned1) {
                    None => {
                        print!(
                            "WARNING: Komunjer and Ng (2011) failed:\n         Conditions for minimality are not fullfilled:\n         Skip identification analysis based on minimal state space system.\n"
                        );
                        None
                    }
                    Some(ms) => Some(super::minimal::d_minimal(&ms, &d_mean, &derivs.dsigma_e, &pt.sigma_e)),
                }
            }
        } else {
            None
        };
        Ok(Jacobians {
            dynamic: f0.dynamic,
            d_dynamic: d_dyn,
            reducedform: f0.reducedform,
            d_reducedform: d_rf,
            moments,
            d_moments,
            d_spectrum,
            d_spectrum_no_mean,
            d_minimal,
            derivs,
            pruned: pruned1,
        })
    }
}
