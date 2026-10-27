// SPDX-License-Identifier: GPL-3.0-or-later
//! The `identification` command: port of Dynare 7.2's `+identification` package at order 1
//! (`run.m`, `analysis.m`, `get_jacobians.m`, `get_perturbation_params_derivs.m`,
//! `checks.m`, `bruteforce.m`, `display.m`, `plot.m`).
//!
//! For a parameter vector (`parameter_set`, then optionally `prior_mc` draws from the prior)
//! the module computes the Jacobians of
//! - the steady state and reduced form (`Tau`: Yss, ghx, Om = ghu Sigma ghu'),
//! - the first two moments of the observables (`J`, Iskrev 2010; `ar` lags, `useautocorr`),
//! - the mean and spectral density (`Gbar`, Qu and Tkachenko 2012),
//! - the minimal state space system (`Deltabar`, Komunjer and Ng 2011),
//! - the steady state and dynamic Jacobian (the "LRE model"),
//!
//! with analytic parameter derivatives (`analytic_derivation_mode = 0`, Sylvester equations;
//! `-1`/`-2` finite differences), checks their rank, finds non-identified parameters and
//! collinear pairs (nullspace + multicorrelation, `checks.m`), measures the identification
//! strength from the asymptotic information matrix of the likelihood on simulated data
//! (Ratto and Iskrev 2011), and with `advanced` runs the brute-force collinearity search.
//!
//! Layout: [`derivs`] (perturbation derivatives), [`moments`] (pruned state space, moments,
//! spectrum), [`minimal`] (Komunjer–Ng), [`checks`], [`strength`], [`display`] (console),
//! [`plots`] (figures), [`run`] (the command).

pub mod checks;
pub mod derivs;
pub mod display;
pub mod files; // load_ident_files, gsa_sample_file
pub mod korder; // order 2/3 (pruned state space)
pub mod la;
pub mod minimal;
pub mod moments;
pub mod plots;
pub mod run;
pub mod strength;
pub mod subsets; // checks_via_subsets

pub use run::{IdentOptions, run};

use anyhow::{Result, anyhow, bail};
use rand::Rng;

use crate::dsge::DsgeModel;
use crate::linalg::{Mat, mul};
use crate::model::Model;
use crate::solver::{DecisionRule, FirstOrderSolver};

use checks::{CheckResult, bruteforce, checks};
use derivs::{ModelParamDerivs, ParamSet, PerturbationDerivs, legacy_columns, perturbation_derivs, to_legacy};
use moments::{PrunedStateSpace, pruned_state_space, spectrum_gram, vech};
use strength::{Strength, strength_measures};

/// `evaluate_steady_state` at `params` as `compute_decision_rules.m` (`resol.m`) calls it: the
/// steady state and `M_.params`, which a `steady_state_model` block may update by assigning
/// parameters (`evaluate_steady_state_file.m`: `params = params1`), as the dynamic Jacobian
/// and the decision rules then use them.
pub(crate) fn steady_state_with_params(
    model: &Model,
    ssm: Option<&crate::deterministic::SteadyStateModel>,
    params: &[f64],
    guess: &[f64],
    x_ss: &[f64],
    opts: &crate::solver::SteadyOptions,
) -> std::result::Result<(Vec<f64>, Vec<f64>), crate::solver::SolveError> {
    // evaluate_steady_state at `oo_.exo_steady_state`
    crate::dsge::solve_steady_at(model, ssm, params, guess, x_ss, opts, true)
}

/// The model at one parameter vector.
pub struct Point {
    pub theta: Vec<f64>,
    pub params: Vec<f64>,
    pub sigma_e: Mat,
    pub ys: Vec<f64>,
    pub g1: Mat,
    pub dr: DecisionRule,
}

/// Everything that does not depend on the parameter vector.
pub struct Context<'a> {
    pub model: &'a Model,
    pub dsge: DsgeModel,
    pub solver: FirstOrderSolver,
    pub pderivs: ModelParamDerivs,
    pub set: ParamSet,
    /// Whether the parameters come from an `estimated_params` block (`prior_exist`).
    pub prior_exist: bool,
    /// Prior standard deviations (`bayestopt_.p2`), NaN without priors.
    pub prior_sd: Vec<f64>,
    /// Observed variables (declaration indices, varobs order).
    pub varobs: Vec<usize>,
    pub qz_criterium: f64,
    pub opts: IdentOptions,
    /// `M_.endo_histval` (levels) of a `histval` block: the starting point of the simulations
    /// of the identification-strength analysis (`stoch_simul`, `simulated_moment_uncertainty.m`).
    pub endo_histval: Option<Vec<f64>>,
    /// The analytic derivatives of `dsge_likelihood` (built on first use).
    pub analytic: std::sync::OnceLock<crate::estimation::analytic::AnalyticContext>,
}

impl Context<'_> {
    /// `set_all_parameters` + `compute_decision_rules`; Err = Dynare's info code.
    pub fn solve(&self, theta: &[f64]) -> std::result::Result<Point, i32> {
        let (params, sigma_e) = if self.prior_exist {
            self.set.apply(theta, &self.dsge.base_params, &self.dsge.base_sigma_e)
        } else {
            (self.dsge.base_params.clone(), self.dsge.base_sigma_e.clone())
        };
        self.solve_at(theta, params, sigma_e)
    }

    /// As [`Self::solve`], always applying `theta` (finite differences).
    fn solve_applied(&self, theta: &[f64]) -> std::result::Result<Point, i32> {
        let (params, sigma_e) = self.set.apply(theta, &self.dsge.base_params, &self.dsge.base_sigma_e);
        self.solve_at(theta, params, sigma_e)
    }

    fn solve_at(&self, theta: &[f64], params: Vec<f64>, sigma_e: Mat) -> std::result::Result<Point, i32> {
        // analysis.m: [oo_.dr, info, M_.params] = compute_decision_rules(...)
        let d = &self.dsge;
        let (ys, params) = steady_state_with_params(
            self.model,
            d.steady_state_model.as_ref(),
            &params,
            &d.ys_guess,
            &d.exo_ss,
            &d.steady_opts,
        )
        .map_err(|e| e.info)?;

        let g1 = self.model.dynamic_jacobian(&ys, &d.exo_ss, &params);
        let dr = self.solver.solve(&g1, &ys).map_err(|e| e.info)?;
        Ok(Point { theta: theta.to_vec(), params, sigma_e, ys, g1, dr })
    }

    /// `dr.derivs` (get_perturbation_params_derivs.m).
    pub fn perturbation_derivs(&self, pt: &Point) -> Result<PerturbationDerivs> {
        if self.opts.analytic_derivation_mode < 0 {
            return self.numerical_derivs(pt);
        }
        let md = self.pderivs.eval(self.model, &pt.ys, &pt.params, self.qz_criterium)?;
        perturbation_derivs(self.model, &pt.dr, &pt.g1, &md, &pt.sigma_e, &self.set)
    }

    /// `analytic_derivation_mode = -1`: two-sided finite differences (`fjaco.m` with
    /// `options_.dynatol.x = 1e-5`) of `[Sigma_e; ghx; ghu]` w.r.t. all selected parameters
    /// and of `[Yss; g1]` w.r.t. the model parameters.
    fn numerical_derivs(&self, pt: &Point) -> Result<PerturbationDerivs> {
        let tol = 1e-5;
        let tot = self.set.len();
        let ns = self.set.n_shock();
        let n = self.model.n_endo();
        let cols = legacy_columns(self.model);
        let mut dghx = Vec::with_capacity(tot);
        let mut dghu = Vec::with_capacity(tot);
        let mut dsig = Vec::with_capacity(tot);
        let mut d_yss = Mat::zeros(n, self.set.n_model());
        let mut dg1_legacy = Vec::new();
        for j in 0..tot {
            let h = tol * pt.theta[j].abs().max(1.0);
            let (mut xp, mut xm) = (pt.theta.clone(), pt.theta.clone());
            xp[j] += h;
            xm[j] -= h;
            let hh = xp[j] - xm[j];
            let fail = |info: i32| anyhow!("fjaco.m: cannot solve the model for {} (info = {info})", self.set.names[j]);
            let p1 = self.solve_applied(&xp).map_err(fail)?;
            let p0 = self.solve_applied(&xm).map_err(fail)?;
            dghx.push((&p1.dr.ghx - &p0.dr.ghx) / hh);
            dghu.push((&p1.dr.ghu - &p0.dr.ghu) / hh);
            dsig.push((&p1.sigma_e - &p0.sigma_e) / hh);
            if j >= ns {
                let ov = &pt.dr.order_var;
                for i in 0..n {
                    d_yss[(i, j - ns)] = (p1.ys[ov[i]] - p0.ys[ov[i]]) / hh;
                }
                dg1_legacy.push((to_legacy(&p1.g1, &cols) - to_legacy(&p0.g1, &cols)) / hh);
            }
        }
        let ghu = &pt.dr.ghu;
        let mut dom = Vec::with_capacity(tot);
        for j in 0..tot {
            if j < ns {
                dom.push(mul(&mul(ghu, false, &dsig[j], false), false, ghu, true));
            } else {
                let o = mul(&mul(&dghu[j], false, &pt.sigma_e, false), false, ghu, true);
                dom.push(&o + o.transpose());
            }
        }
        Ok(PerturbationDerivs { d_yss, dg1_legacy, dghx, dghu, dom, dsigma_e: dsig })
    }
}

/// One identification criterion (`ide_reducedform`, `ide_moments`, ...).
#[derive(Clone, Debug, Default)]
pub struct IdeBlock {
    /// The full Jacobian (`dREDUCEDFORM`, `dMOMENTS`, `dSPECTRUM`, `dMINIMAL`, `dDYNAMIC`).
    pub jacobian: Mat,
    /// The values the Jacobian is taken of, restricted to `ind` (empty for spectrum/minimal).
    pub values: Vec<f64>,
    /// Rows with non-zero derivatives (`ind_d*`).
    pub ind: Vec<usize>,
    /// `si_d* = jacobian(ind, :)`.
    pub si: Mat,
    /// Row normalization (`norm_d*`, max abs of each row), 1 when not normalizing.
    pub norm: Vec<f64>,
    pub check: CheckResult,
}

/// Output of `identification.analysis`.
#[derive(Clone, Debug, Default)]
pub struct Analysis {
    pub theta: Vec<f64>,
    pub dynamic: IdeBlock,
    pub reducedform: Option<IdeBlock>,
    pub moments: Option<IdeBlock>,
    pub spectrum: Option<IdeBlock>,
    /// The spectrum Gram matrix without the mean (`dSPECTRUM_NO_MEAN`).
    pub spectrum_no_mean: Option<Mat>,
    pub minimal: Option<IdeBlock>,
    /// The minimal state space could be computed (`ide_minimal.minimal_state_space`).
    pub minimal_state_space: bool,
    pub strength: Option<Strength>,
    /// Singular values (first 4 and last 4 when more than 8 parameters) and vectors of
    /// `dMOMENTS(ind, :)`.
    pub moments_s: Vec<f64>,
    pub moments_v: Mat,
    /// `advanced`: `(pars, cosndMOMENTS)` from the brute-force search.
    pub bruteforce: Option<(Vec<Vec<Vec<Option<usize>>>>, Mat)>,
    pub nlags: usize,
    pub error_strength: bool,
    pub error_moments: bool,
    pub error_minimal: bool,
    pub error_spectrum: bool,
    /// `checks_via_subsets` results of the point analysis.
    pub subsets: Option<subsets::Subsets>,
}

/// The `persistent` row selections of analysis.m, fixed at the first (init) call.
#[derive(Clone, Debug, Default)]
pub struct Persistent {
    pub ind_dmoments: Vec<usize>,
    pub ind_dreducedform: Vec<usize>,
    pub ind_ddynamic: Vec<usize>,
    pub nlags: usize,
}

fn nonzero_rows(j: &Mat, tol: f64) -> Vec<usize> {
    (0..j.nrows()).filter(|&i| (0..j.ncols()).map(|c| j[(i, c)].abs()).fold(f64::NAN, f64::max) > tol).collect()
}

fn rows(j: &Mat, r: &[usize]) -> Mat {
    Mat::from_fn(r.len(), j.ncols(), |i, c| j[(r[i], c)])
}

fn row_max_abs(m: &Mat) -> Vec<f64> {
    (0..m.nrows()).map(|i| (0..m.ncols()).map(|c| m[(i, c)].abs()).fold(0.0, f64::max)).collect()
}

fn normalized(m: &Mat, norm: &[f64]) -> Mat {
    Mat::from_fn(m.nrows(), m.ncols(), |i, c| m[(i, c)] / norm[i])
}

fn has_nan(m: &Mat) -> bool {
    m.iter().any(|v| v.is_nan())
}

/// Jacobians of one parameter vector (`identification.get_jacobians`, order 1).
pub struct Jacobians {
    pub dynamic: Vec<f64>,
    pub d_dynamic: Mat,
    pub reducedform: Vec<f64>,
    pub d_reducedform: Mat,
    pub moments: Vec<f64>,
    pub d_moments: Mat,
    pub d_spectrum: Option<Mat>,
    pub d_spectrum_no_mean: Option<Mat>,
    pub d_minimal: Option<Mat>,
    pub derivs: PerturbationDerivs,
    pub pruned: PrunedStateSpace,
}

impl Context<'_> {
    /// `identification.get_jacobians`.
    pub fn jacobians(&self, pt: &Point, nlags: usize, spectrum: bool, minimal: bool) -> Result<Jacobians> {
        if self.opts.order > 1 {
            return self.jacobians_k(pt, nlags, spectrum, minimal);
        }
        let o = &self.opts;
        let n = self.model.n_endo();
        let ns = self.set.n_shock();
        let tot = self.set.len();
        let derivs = self.perturbation_derivs(pt)?;
        let dr = &pt.dr;
        let yss: Vec<f64> = dr.order_var.iter().map(|&v| pt.ys[v]).collect();
        let cols = legacy_columns(self.model);
        let g1l = to_legacy(&pt.g1, &cols);
        let mut dynamic = yss.clone();
        dynamic.extend(g1l.iter().copied());
        let nm = self.set.n_model();
        let mut d_dynamic = Mat::zeros(n + g1l.len(), nm);
        for j in 0..nm {
            for i in 0..n {
                d_dynamic[(i, j)] = derivs.d_yss[(i, j)];
            }
            for (k, v) in derivs.dg1_legacy[j].iter().enumerate() {
                d_dynamic[(n + k, j)] = *v;
            }
        }
        let om = mul(&mul(&dr.ghu, false, &pt.sigma_e, false), false, &dr.ghu, true);
        let mut reducedform = yss;
        reducedform.extend(dr.ghx.iter().copied());
        reducedform.extend(vech(&om));
        let mut d_reducedform = Mat::zeros(reducedform.len(), tot);
        for j in 0..tot {
            let mut col: Vec<f64> =
                if j < ns { vec![0.0; n] } else { derivs.d_yss.column(j - ns).iter().copied().collect() };
            col.extend(derivs.dghx[j].iter().copied());
            col.extend(vech(&derivs.dom[j]));
            for (r, v) in col.into_iter().enumerate() {
                d_reducedform[(r, j)] = v;
            }
        }
        let indy: Vec<usize> = self.varobs.iter().map(|&v| dr.inv_order_var[v]).collect();
        let pruned = pruned_state_space(
            dr,
            &derivs,
            &pt.sigma_e,
            &indy,
            nlags,
            o.useautocorr,
            ns,
            self.qz_criterium,
            o.schur_vec_tol,
        )?;
        let (moments, d_moments) =
            if o.no_moments { (vec![], Mat::zeros(0, tot)) } else { moments::moments(&pruned, nlags) };
        let (d_spectrum, d_spectrum_no_mean) = if spectrum {
            let g = spectrum_gram(&pruned, ns, o.grid_nbr);
            let dm = &pruned.de_y;
            (Some(&g + mul(dm, true, dm, false)), Some(g))
        } else {
            (None, None)
        };
        let d_minimal = if minimal {
            let obs = indy.len();
            let ne = self.model.n_exo();
            if obs < ne {
                print!(
                    "WARNING: Komunjer and Ng (2011) failed:\n       There are more shocks and measurement errors than observables, this is not implemented (yet).\n       Skip identification analysis based on minimal state space system.\n"
                );
                None
            } else {
                match minimal::minimal_state_representation(&pruned) {
                    None => {
                        print!(
                            "WARNING: Komunjer and Ng (2011) failed:\n         Conditions for minimality are not fullfilled:\n         Skip identification analysis based on minimal state space system.\n"
                        );
                        None
                    }
                    Some(ms) => Some(minimal::d_minimal(&ms, &pruned.de_y, &derivs.dsigma_e, &pt.sigma_e)),
                }
            }
        } else {
            None
        };
        Ok(Jacobians {
            dynamic,
            d_dynamic,
            reducedform,
            d_reducedform,
            moments,
            d_moments,
            d_spectrum,
            d_spectrum_no_mean,
            d_minimal,
            derivs,
            pruned,
        })
    }

    /// `identification.analysis(..., params, ..., init)`. `Ok(Err(info))` when the model does
    /// not solve at `theta`.
    pub fn analysis<R: Rng + ?Sized>(
        &self,
        theta: &[f64],
        advanced: bool,
        init: bool,
        pers: &mut Persistent,
        rng: &mut R,
    ) -> Result<std::result::Result<Analysis, i32>> {
        let o = &self.opts;
        let tot = self.set.len();
        let nm = self.set.n_model();
        let pt = match self.solve(theta) {
            Ok(p) => p,
            Err(info) => return Ok(Err(info)),
        };
        let mut nlags = if init { o.ar } else { pers.nlags };
        let jac = self.jacobians(&pt, nlags, !o.no_spectrum, !o.no_minimal)?;
        let Jacobians { dynamic, d_dynamic, reducedform, d_reducedform, mut moments, mut d_moments, .. } = jac;
        let mut an = Analysis { theta: theta.to_vec(), ..Default::default() };
        an.error_minimal = jac.d_minimal.is_none();
        if init {
            if !o.no_moments {
                if moments.iter().any(|v| v.is_nan()) {
                    if o.diffuse_filter {
                        bail!(
                            "There are NaN's in the theoretical moments. Make sure that for non-stationary models stationary transformations of non-stationary observables are used for checking identification. [TIP: use first differences]."
                        );
                    } else {
                        bail!(
                            "There are NaN's in the theoretical moments. Please check whether your model has units roots, and you forgot to set diffuse_filter=1."
                        );
                    }
                }
                pers.ind_dmoments = nonzero_rows(&d_moments, o.tol_deriv);
                if pers.ind_dmoments.is_empty() && has_nan(&d_moments) {
                    bail!("There are NaN in the dMOMENTS matrix.");
                }
            }
            if let Some(s) = &jac.d_spectrum {
                if has_nan(s) {
                    print!(
                        "WARNING: There are NaN in the dSPECTRUM matrix. Note that identification based on spectrum does not support non-stationary models (yet).\n         Skip identification analysis based on spectrum.\n"
                    );
                    an.error_spectrum = true;
                }
            }
            if let Some(m) = &jac.d_minimal {
                if has_nan(m) {
                    print!(
                        "WARNING: There are NaN in the dMINIMAL matrix. Note that identification based on minimal system does not support non-stationary models (yet).\n         Skip identification analysis based on minimal system.\n"
                    );
                    an.error_minimal = true;
                }
            }
            if !o.no_moments && !an.error_moments {
                while pers.ind_dmoments.len() < tot && nlags < 10 {
                    println!("The number of moments with non-zero derivative is smaller than the number of parameters");
                    println!("Try increasing ar = {}", nlags + 1);
                    nlags += 1;
                    let j2 = self.jacobians(&pt, nlags, false, false)?;
                    moments = j2.moments;
                    d_moments = j2.d_moments;
                    pers.ind_dmoments = nonzero_rows(&d_moments, o.tol_deriv);
                }
                if pers.ind_dmoments.len() < tot {
                    print!(
                        "WARNING: Order condition for dMOMENTS failed: There are not enough moments and too many parameters.\n         The number of moments with non-zero derivative is smaller than the number of parameters up to 10 lags.\n         Either reduce the list of parameters, or further increase ar, or increase number of observables.\n         Skip identification analysis based on moments.\n         Skip identification strength analysis.\n"
                    );
                    an.error_moments = true;
                    an.error_strength = true;
                }
            }
            if !o.no_minimal && !an.error_minimal {
                let m = jac.d_minimal.as_ref().unwrap();
                if nonzero_rows(m, o.tol_deriv).len() < m.ncols() {
                    print!(
                        "WARNING: Order condition for dMINIMAL failed: There are too many parameters or too few observable variables.\n         The number of minimal system elements with non-zero derivative is smaller than the number of parameters.\n         Either reduce the list of parameters, or increase number of observables.\n         Skip identification analysis based on minimal state space system.\n"
                    );
                    an.error_minimal = true;
                }
            }
            if an.error_moments && an.error_minimal && an.error_spectrum {
                bail!("identification_analyis: Order condition(s) failed");
            }
            if !o.no_reducedform {
                pers.ind_dreducedform = nonzero_rows(&d_reducedform, o.tol_deriv);
            }
            pers.ind_ddynamic = nonzero_rows(&d_dynamic, o.tol_deriv);
            pers.nlags = nlags;
        }
        // Monte Carlo draws (init = 0): error_indicator is reset on every call and the order
        // conditions are not checked, the moments use the persistent ind_dMOMENTS
        an.nlags = nlags;
        let mut ind_ddynamic = pers.ind_ddynamic.clone();
        let mut si_ddynamic = rows(&d_dynamic, &ind_ddynamic);
        let dyn_values: Vec<f64> = ind_ddynamic.iter().map(|&i| dynamic[i]).collect();
        let mut ind_drf = pers.ind_dreducedform.clone();
        let mut si_drf = rows(&d_reducedform, &ind_drf);
        let rf_values: Vec<f64> = ind_drf.iter().map(|&i| reducedform[i]).collect();
        let do_moments = !o.no_moments && !an.error_moments;
        let ind_dm = pers.ind_dmoments.clone();
        let si_dm = if do_moments { rows(&d_moments, &ind_dm) } else { Mat::zeros(0, tot) };

        // identification strength
        if do_moments && !o.no_strength && !an.error_strength && init {
            let st = self.strength(&pt, &si_dm, &si_drf, &si_ddynamic, &ind_dm, nlags, rng)?;
            ind_drf = st.keep_reducedform.iter().map(|&i| ind_drf[i]).collect();
            si_drf = rows(&si_drf, &st.keep_reducedform);
            ind_ddynamic = st.keep_dynamic.iter().map(|&i| ind_ddynamic[i]).collect();
            si_ddynamic = rows(&si_ddynamic, &st.keep_dynamic);
            an.strength = Some(st);
        }

        let norm_of = |si: &Mat| if o.normalize_jacobians { row_max_abs(si) } else { vec![1.0; si.nrows()] };
        // dynamic model
        let norm_dyn = norm_of(&si_ddynamic);
        let chk = checks(&normalized(&rows(&d_dynamic, &ind_ddynamic), &norm_dyn), 1, o.tol_rank, o.tol_sv, nm);
        an.dynamic = IdeBlock {
            jacobian: d_dynamic,
            values: dyn_values,
            ind: ind_ddynamic,
            si: si_ddynamic,
            norm: norm_dyn,
            check: chk,
        };
        if !o.no_reducedform {
            let norm = norm_of(&si_drf);
            let chk = checks(&normalized(&rows(&d_reducedform, &ind_drf), &norm), 1, o.tol_rank, o.tol_sv, tot);
            an.reducedform = Some(IdeBlock {
                jacobian: d_reducedform,
                values: rf_values,
                ind: ind_drf,
                si: si_drf,
                norm,
                check: chk,
            });
        }
        if do_moments {
            let norm = norm_of(&si_dm);
            if advanced {
                an.bruteforce = Some(bruteforce(&si_dm, o.max_dim_cova_group, o.tol_deriv));
            }
            let (s, v) = la::svd_v_full(&si_dm);
            let mut s = s;
            s.resize(tot, 0.0);
            if tot > 8 {
                let keep: Vec<usize> = (0..4).chain(tot - 4..tot).collect();
                an.moments_s = keep.iter().map(|&k| s[k]).collect();
                an.moments_v = Mat::from_fn(v.nrows(), 8, |i, j| v[(i, keep[j])]);
            } else {
                an.moments_s = s;
                an.moments_v = v;
            }
            let chk = checks(&normalized(&si_dm, &norm), 1, o.tol_rank, o.tol_sv, tot);
            let values = ind_dm.iter().map(|&i| moments[i]).collect();
            an.moments = Some(IdeBlock { jacobian: d_moments, values, ind: ind_dm, si: si_dm, norm, check: chk });
        }
        if !o.no_minimal {
            if !an.error_minimal {
                let m = jac.d_minimal.clone().unwrap();
                let ind = nonzero_rows(&m, o.tol_deriv);
                let si = rows(&m, &ind);
                let norm = norm_of(&si);
                let chk = checks(&normalized(&si, &norm), 2, o.tol_rank, o.tol_sv, tot);
                an.minimal = Some(IdeBlock { jacobian: m, values: vec![], ind, si, norm, check: chk });
                an.minimal_state_space = true;
            } else {
                an.minimal_state_space = false;
            }
        }
        if !o.no_spectrum && !an.error_spectrum {
            let g = jac.d_spectrum.clone().unwrap();
            let ind = nonzero_rows(&g, o.tol_deriv);
            let tilda = if o.normalize_jacobians {
                let mut t = Mat::zeros(g.nrows(), g.ncols());
                let delta: Vec<f64> = ind.iter().map(|&i| g[(i, i)].sqrt()).collect();
                for (a, &i) in ind.iter().enumerate() {
                    for (b, &j) in ind.iter().enumerate() {
                        t[(i, j)] = g[(i, j)] / (delta[a] * delta[b]);
                    }
                }
                t
            } else {
                g.clone()
            };
            let si = rows(&g, &ind);
            let norm = norm_of(&si);
            let chk = checks(&tilda, 3, o.tol_rank, o.tol_sv, tot);
            an.spectrum = Some(IdeBlock { jacobian: g, values: vec![], ind, si: tilda, norm, check: chk });
            an.spectrum_no_mean = jac.d_spectrum_no_mean.clone();
        }
        Ok(Ok(an))
    }

    /// The identification-strength block of analysis.m: the asymptotic Hessian of
    /// `dsge_likelihood` on `stoch_simul` data (`periods + 100` periods from `M_.endo_histval`
    /// or the steady state, the first 100 dropped), simulated moment uncertainty (moments up to
    /// lag `nlags`, the `options_.ar` raised by the order condition) as fallback.
    #[allow(clippy::too_many_arguments)]
    fn strength<R: Rng + ?Sized>(
        &self,
        pt: &Point,
        si_dm: &Mat,
        si_drf: &Mat,
        si_ddyn: &Mat,
        ind_dm: &[usize],
        nlags: usize,
        rng: &mut R,
    ) -> Result<Strength> {
        let o = &self.opts;
        let ns = self.set.n_shock();
        let y0 = self.endo_histval.clone().unwrap_or_else(|| pt.dr.ys.clone());
        // Without an estimated_params block dsge_likelihood cannot be called (Dynare's `try`
        // fails), without priors (ML) prior_bounds errors on pshape 0 inside the same `try`, and
        // the diffuse filter (lik_init = 3) has no analytic derivatives: all go to the
        // simulated-moments branch. order > 1 has no analytic Hessian (analysis.m errors inside
        // the `try`).
        let ahess = if o.lik_init != 3 && self.prior_exist && !self.dsge.ml && o.order == 1 {
            let data = strength::simulate(&pt.dr, &pt.sigma_e, &y0, o.periods + 100, rng);
            self.asymptotic_hessian_on(pt, &data.columns(100, o.periods).into_owned())
        } else {
            Err(anyhow!("no analytic Hessian (diffuse filter, order > 1 or no priors)"))
        };
        let ahess = ahess.and_then(|h| {
            let (w, _) = crate::linalg::sym_eig(&h);
            let mineig = w.iter().copied().fold(f64::INFINITY, f64::min);
            // `min(eig(AHess)) < -tol_rank`; with tol_rank = 'robust' MATLAB compares with the
            // negated character codes, i.e. the test only fires below -117.
            let thr = o.tol_rank.unwrap_or(117.0);
            if mineig < -thr {
                Err(anyhow!("identification.analysis: Analytic Hessian is not positive semi-definite!"))
            } else {
                Ok(h)
            }
        });
        match ahess {
            Ok(h) => Ok(strength_measures(
                h,
                true,
                &pt.theta,
                &self.prior_sd,
                si_dm,
                None,
                si_drf,
                si_ddyn,
                ns,
                o.tol_rank,
                o.tol_sv,
            )),
            Err(_) => {
                let replic = o.replic.max(3 * ind_dm.len());
                println!("Evaluating simulated moment uncertainty ... please wait");
                println!("Doing {replic} replicas of length {} periods.", o.periods);
                // options_.periods is periods + 100 here (set before the `try` failed, or in the
                // `catch` at order > 1); disp_moments drops the global options_.drop
                let total = o.periods + 100;
                let drop = o.drop;
                if total <= drop {
                    bail!(
                        "simulated_moment_uncertainty: The horizon of simulation is shorter than the number of observations to be dropped. Either increase options_.periods or decrease options_.drop."
                    );
                }
                if total - drop <= nlags {
                    // too few kept periods for the lagged moments: Dynare's autocorrelations are NaN
                    bail!(
                        "simulated_moment_uncertainty: periods + 100 = {total} leaves {} simulated periods after dropping options_.drop = {drop}, too few for {nlags} lags (Dynare's simulated moments are NaN).",
                        total - drop
                    );
                }
                let cmm = strength::simulated_moment_uncertainty(
                    &pt.dr,
                    &pt.sigma_e,
                    &y0,
                    &self.varobs,
                    ind_dm,
                    total,
                    drop,
                    replic,
                    nlags,
                    rng,
                );
                println!("Simulated moment uncertainty ... done!");
                let mim = strength::moments_information_matrix(si_dm, &cmm, o.tol_deriv);
                let d: Vec<f64> = (0..cmm.nrows()).map(|i| cmm[(i, i)]).collect();
                Ok(strength_measures(
                    mim,
                    false,
                    &pt.theta,
                    &self.prior_sd,
                    si_dm,
                    Some(d),
                    si_drf,
                    si_ddyn,
                    ns,
                    o.tol_rank,
                    o.tol_sv,
                ))
            }
        }
    }

    /// `AHess` of the `try` branch of analysis.m: `-Hess` of `dsge_likelihood(params, dataset_,
    /// ...)` with `analytic_derivation` and `analytic_Hessian = 'asymptotic'` on `data`
    /// (n_endo x nobs, levels, declaration-order rows), through the estimation's port
    /// ([`crate::estimation::analytic`]): the `lik_init` initializations (with `DP = 0` unless
    /// `lik_init = 1`), the calibrated `M_.H` in `F = P(Z,Z) + H`, and the switch to the
    /// univariate filter when the forecast-error variance is singular
    /// (`use_univariate_filters_if_singularity_is_detected`). `Err` where the `try` fails.
    pub fn asymptotic_hessian_on(&self, pt: &Point, data: &Mat) -> Result<Mat> {
        use crate::estimation::analytic::{AnalyticContext, AnalyticHessian, AnalyticOptions, AnalyticPosterior};
        let o = &self.opts;
        let sample = Mat::from_fn(data.ncols(), self.varobs.len(), |t, j| data[(self.varobs[j], t)]);
        let kalman = crate::kalman::KalmanOptions {
            kalman_tol: o.kalman_tol,
            diffuse_kalman_tol: o.diffuse_kalman_tol,
            riccati_tol: o.riccati_tol,
            qz_criterium: self.qz_criterium,
            lik_init: o.lik_init,
            // analysis.m: `if options_.kalman_algo > 2, options_.kalman_algo = 1; end`
            kalman_algo: if o.kalman_algo > 2 { 1 } else { o.kalman_algo },
            use_univariate_filters_if_singularity_is_detected: o.use_univariate_filters_if_singularity_is_detected,
            ..Default::default()
        };
        let data = crate::dsge::LikelihoodData::new(&sample, kalman);
        // bounds = prior_bounds(bayestopt_, options_.prior_trunc)
        let bounds = self.prior_bounds();
        let post = crate::dsge::DsgePosterior { dsge: &self.dsge, data: &data, bounds: &bounds };
        let ctx = match self.analytic.get() {
            Some(c) => c,
            None => {
                let c = AnalyticContext::new(&post, AnalyticOptions { mode: o.analytic_derivation_mode })?;
                self.analytic.get_or_init(|| c)
            }
        };
        let g = AnalyticPosterior { post: &post, ctx }
            .evaluate(&pt.theta, AnalyticHessian::Asymptotic)
            .map_err(|(info, _)| anyhow!("dsge_likelihood failed (info = {info})"))?;
        Ok(-g.hess.ok_or_else(|| anyhow!("dsge_likelihood: no asymptotic Hessian"))?)
    }
}
