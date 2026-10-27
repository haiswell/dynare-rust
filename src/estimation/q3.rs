// SPDX-License-Identifier: GPL-3.0-or-later
//! Estimation options: option list and the `estimation` hooks of the stream
//! (DSGE-VAR, heteroskedastic filter, conditional likelihood, Lyapunov solvers, smoother
//! outputs, convergence diagnostics, extra optimizers).

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::collapsible_if
)] // index loops mirror the MATLAB code

use anyhow::{Result, bail};

use crate::dsge::{DsgeModel, LikelihoodData};
use crate::estimation::dsge_var::{DsgeVarPosterior, PriorWeight, VarMoments};
use crate::estimation::{Bounds, Objective};
use crate::kalman::KalmanOptions;
use crate::linalg::Mat;
use crate::modfile::{ModFile, OptValue, Options, opt};

/// `estimation` options handled by .
pub const OPTIONS: &[&str] = &[
    // convergence diagnostics (mcmc_diagnostics.m)
    "taper_steps",
    "geweke_interval",
    "raftery_lewis_diagnostics",
    "raftery_lewis_qrs",
    "brooks_gelman_plotrows",
    // DSGE-VAR
    "dsge_var",
    "dsge_varlag",
    // Kalman filter
    "heteroskedastic_filter",
    "rescale_prediction_error_covariance",
    "conditional_likelihood",
    "lyapunov",
    "lyapunov_fixed_point_tol",
    "lyapunov_doubling_tol",
    "lyapunov_complex_threshold",
    // output and optimizers
    "dirname",
    "silent_optimizer",
    // smoother outputs (DsgeSmoother.m / store_smoother_results.m)
    "filter_step_ahead",
    "filter_covariance",
    "filter_decomposition",
    "smoother_redux",
    // variables of the posterior objects (check_list_of_variables.m)
    "consider_all_endogenous",
    "consider_all_endogenous_and_auxiliary",
    "consider_only_observed",
    "selected_variables_only",
    // pruned skewed Kalman filter (kalman_algo = 5)
    "skewed_kalman_prune_tol",
    "skewed_kalman_rank_deficiency_transform",
    "skewed_kalman_mvnlogcdf",
    // MCMC proposal (set_mcmc_jumping_covariance.m)
    "mcmc_jumping_covariance",
];

/// Port of `set_mcmc_jumping_covariance.m` (`mcmc_jumping_covariance = hessian | prior_variance |
/// identity_matrix | FILENAME`). The file case loads `jumping_covariance` and, as Dynare does,
/// uses its inverse (computed with the diagonal scaling of the MATLAB code).
pub fn jumping_covariance(
    o: &Options,
    invhess: Mat,
    priors: &[crate::estimation::Prior],
    dir: &std::path::Path,
) -> Result<Mat> {
    let n = invhess.nrows();
    let scaled_inv = |hess: &Mat| -> Result<Mat> {
        let hsd: Vec<f64> = (0..n).map(|i| hess[(i, i)].sqrt()).collect();
        let s = Mat::from_fn(n, n, |i, j| hess[(i, j)] / (hsd[i] * hsd[j]));
        let si = crate::linalg::solve(&s, &Mat::identity(n, n))
            .ok_or_else(|| anyhow::anyhow!("mcmc_jumping_covariance: singular matrix"))?;
        Ok(Mat::from_fn(n, n, |i, j| si[(i, j)] / (hsd[i] * hsd[j])))
    };
    let Some(OptValue::Str(v)) = opt(o, "mcmc_jumping_covariance") else { return Ok(invhess) };
    let v = v.trim_matches(|c| c == '\'' || c == '"');
    match v {
        "hessian" => Ok(invhess),
        "prior_variance" => {
            if priors.iter().any(|p| p.sd.is_infinite()) {
                bail!(
                    "dynare_estimation_1: Infinite prior variances detected. You cannot use the prior variances as the proposal density, if some variances are Inf."
                );
            }
            let hess = Mat::from_diagonal(&crate::linalg::Vector::from_iterator(
                n,
                priors.iter().map(|p| 1.0 / (p.sd * p.sd)),
            ));
            scaled_inv(&hess)
        }
        "identity_matrix" => Ok(Mat::identity(n, n)),
        file => {
            let mut p = dir.join(file);
            if p.extension().is_none() {
                p.set_extension("mat");
            }
            let vars = crate::matfile::read(&p)?;
            let hess = vars.get("jumping_covariance").and_then(|m| m.as_matrix()).ok_or_else(|| {
                anyhow::anyhow!(
                    "dynare_estimation_1: No matrix named 'jumping_covariance' could be found in {}!",
                    p.display()
                )
            })?;
            if hess.nrows() != n || hess.ncols() != n {
                bail!(
                    "dynare_estimation_1: 'jumping_covariance' matrix (loaded from {}) must be square and have {n} rows and columns!",
                    p.display()
                );
            }
            if hess.clone().cholesky().is_none() {
                bail!("dynare_estimation_1: Specified 'MCMC_jumping_covariance' is not positive definite!");
            }
            scaled_inv(&hess)
        }
    }
}

/// Port of `check_list_of_variables.m` (the non-interactive part): the variables of the
/// posterior IRFs, moments, smoothed variables and forecasts. `selected_variables_only` only
/// makes Dynare's smoother skip the other variables; the results for the listed ones are the
/// same, so it is accepted (dynare-rust writes all variables).
pub fn var_list(mf: &ModFile, o: &Options, var_list: &[String]) -> Result<Vec<String>> {
    if dsge_var_weight(mf, o)?.is_some() && flag(o, "bayesian_irf") {
        if !var_list.is_empty()
            && (var_list.len() != mf.varobs.len() || var_list.iter().any(|v| !mf.varobs.contains(v)))
        {
            for v in var_list.iter().filter(|v| !mf.varobs.contains(v)) {
                println!("{v} is not an observed variable!");
            }
            println!("\nPosterior IRFs will be computed for all observed variables.\n");
        }
        return Ok(mf.varobs.clone());
    }
    let which = if flag(o, "consider_all_endogenous") {
        Some("all_endogenous_variables")
    } else if flag(o, "consider_all_endogenous_and_auxiliary") {
        Some("all_endogenous_and_auxiliary_variables")
    } else if flag(o, "consider_only_observed") {
        Some("only_observed_variables")
    } else {
        None
    };
    match which {
        Some(_) if !var_list.is_empty() => bail!(
            "You cannot use the consider_all_endogenous, consider_all_endogenous_and_auxiliary or consider_all_observed options when listing variables after the estimation command"
        ),
        Some("all_endogenous_variables") => Ok(mf.model.endo_names[..mf.model.orig_endo_nbr].to_vec()),
        Some("all_endogenous_and_auxiliary_variables") => Ok(mf.model.endo_names.clone()),
        Some(_) => Ok(mf.varobs.clone()),
        None => Ok(var_list.to_vec()),
    }
}

fn steps(o: &Options) -> Vec<usize> {
    match opt(o, "filter_step_ahead") {
        Some(OptValue::Num(v)) => vec![*v as usize],
        Some(OptValue::Vec(v)) => v.iter().map(|&x| x as usize).collect(),
        Some(OptValue::List(l)) => {
            l.iter().filter_map(|x| if let OptValue::Num(v) = x { Some(*v as usize) } else { None }).collect()
        }
        _ => Vec::new(),
    }
}

/// `filter_step_ahead`, `filter_covariance`, `filter_decomposition` (DsgeSmoother.m +
/// store_smoother_results.m at the smoother's parameter vector): writes
/// `<fname>_filtered_variables_<k>_step_ahead.csv` (`oo_.FilteredVariablesKStepAhead`, levels),
/// `..._variances.csv` (diagonal of `oo_.FilteredVariablesKStepAheadVariances`) and
/// `..._shock_decomposition.csv` (`oo_.FilteredVariablesShockDecomposition`, `var:shock`
/// columns), periods 1..nobs+max(k) of the MATLAB arrays. `smoother_redux` gives the same
/// smoothed objects (it only changes how DsgeSmoother computes them), so it is accepted.
pub fn smoother_outputs(
    dsge: &DsgeModel,
    data: &LikelihoodData,
    x: &[f64],
    o: &Options,
    out_dir: &std::path::Path,
    fname: &str,
) -> Result<()> {
    let st = steps(o);
    if st.is_empty() && !flag(o, "filter_covariance") && !flag(o, "filter_decomposition") {
        return Ok(());
    }
    // dynare_estimation_init.m: filtered_vars => filter_step_ahead = 1 when empty
    let st = if st.is_empty() { vec![1] } else { st };
    let nk = st.iter().copied().max().unwrap_or(1);
    let req = crate::kalman::SmootherRequest {
        nk,
        filter_covariance: flag(o, "filter_covariance"),
        decomposition: flag(o, "filter_decomposition"),
        a_initial: None,
    };
    let (_, sa, inv_order_var, constant) = crate::driver::run_smoother_request(dsge, data, x, false, Some(&req))?;
    let Some(sa) = sa else { return Ok(()) };
    let model = &dsge.model;
    let norig = model.orig_endo_nbr;
    std::fs::create_dir_all(out_dir)?;
    let ncol = sa.ak.first().map_or(0, |m| m.ncols());
    for &k in &st {
        if k == 0 || k > nk {
            continue;
        }
        let mut w = format!("period,{}\n", model.endo_names[..norig].join(","));
        for c in 0..ncol {
            w += &format!("{}", c + 1);
            for v in 0..norig {
                w += &format!(",{}", sa.ak[k - 1][(inv_order_var[v], c)] + constant[v]);
            }
            w.push('\n');
        }
        std::fs::write(out_dir.join(format!("{fname}_filtered_variables_{k}_step_ahead.csv")), w)?;
        if let Some(pk) = &sa.pk {
            let mut w = format!("period,{}\n", model.endo_names[..norig].join(","));
            for (c, m) in pk[k - 1].iter().enumerate() {
                w += &format!("{}", c + 1);
                for v in 0..norig {
                    let i = inv_order_var[v];
                    w += &format!(",{}", m[(i, i)]);
                }
                w.push('\n');
            }
            std::fs::write(out_dir.join(format!("{fname}_filtered_variables_{k}_step_ahead_variances.csv")), w)?;
        }
        if let Some(dec) = &sa.decomp {
            let mut head = String::from("period");
            for v in 0..norig {
                for e in &model.exo_names {
                    head += &format!(",{}:{e}", model.endo_names[v]);
                }
            }
            let mut w = head + "\n";
            for (c, m) in dec[k - 1].iter().enumerate() {
                w += &format!("{}", c + 1);
                for v in 0..norig {
                    for j in 0..model.n_exo() {
                        w += &format!(",{}", m[(inv_order_var[v], j)]);
                    }
                }
                w.push('\n');
            }
            std::fs::write(
                out_dir.join(format!("{fname}_filtered_variables_{k}_step_ahead_shock_decomposition.csv")),
                w,
            )?;
        }
    }
    println!(
        "Smoother: {}-step-ahead filtered variables{}{} written to {}.",
        st.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(","),
        if req.filter_covariance { ", their variances" } else { "" },
        if req.decomposition { ", their shock decomposition" } else { "" },
        out_dir.display()
    );
    Ok(())
}

/// `options_.dirname` (dynare_estimation.m: `M_.dname`): the folder of `metropolis/` and
/// `Output/`, relative to the .mod file's folder; default `<fname>` (then `out_dir` is kept).
pub fn dname(
    dir: &std::path::Path,
    fname: &str,
    o: &Options,
    out_dir: &std::path::Path,
) -> (std::path::PathBuf, std::path::PathBuf) {
    match opt(o, "dirname") {
        Some(OptValue::Str(d)) if !d.is_empty() => {
            let dn = dir.join(d);
            let out = dn.join("Output");
            (dn, out)
        }
        _ => (dir.join(fname), out_dir.to_path_buf()),
    }
}

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

fn flag(o: &Options, k: &str) -> bool {
    match opt(o, k) {
        Some(OptValue::Flag) => true,
        Some(OptValue::Num(v)) => *v != 0.0,
        _ => false,
    }
}

/// Kalman options of the stream (`rescale_prediction_error_covariance`).
pub fn kalman_options(o: &Options, k: &mut KalmanOptions) {
    if flag(o, "rescale_prediction_error_covariance") {
        k.rescale_prediction_error_covariance = true;
    }
    // lyapunov = fixed_point / doubling / square_root_solver (the preprocessor's lyapunov_fp/db/srs)
    if let Some(OptValue::Str(m)) = opt(o, "lyapunov") {
        k.lyapunov.method = match m.as_str() {
            "fixed_point" => 1,
            "doubling" => 2,
            "square_root_solver" => 3,
            _ => 0,
        };
    }
    if let Some(v) = num(o, "lyapunov_fixed_point_tol") {
        k.lyapunov.fixed_point_tol = v;
    }
    if let Some(v) = num(o, "lyapunov_doubling_tol") {
        k.lyapunov.doubling_tol = v;
    }
    if let Some(v) = num(o, "lyapunov_complex_threshold") {
        k.lyapunov_complex_threshold = v;
    }
}

/// The weight of a DSGE-VAR, if the estimation is one (dynare_estimation.m: `dsge_var`, or
/// a parameter named `dsge_prior_weight`).
pub fn dsge_var_weight(mf: &ModFile, o: &Options) -> Result<Option<PriorWeight>> {
    if let Some(i) = mf.model.param_names.iter().position(|n| n == "dsge_prior_weight") {
        return Ok(Some(PriorWeight::Estimated(i)));
    }
    Ok(match opt(o, "dsge_var") {
        None => None,
        Some(OptValue::Num(w)) => Some(PriorWeight::Calibrated(*w)),
        Some(OptValue::Str(s)) if s.eq_ignore_ascii_case("inf") => Some(PriorWeight::Calibrated(f64::INFINITY)),
        Some(OptValue::Flag) => bail!(
            "dsge_var: the weight of the DSGE prior is estimated, so dsge_prior_weight must be in the estimated_params block"
        ),
        Some(v) => bail!("dsge_var: unexpected value {v:?}"),
    })
}

/// Number of initial conditions makedataset.m prepends to the sample
/// (`options_.dsge_var*options_.dsge_varlag`).
pub fn initial_conditions(mf: &ModFile, o: &Options) -> Result<usize> {
    if dsge_var_weight(mf, o)?.is_none() {
        return Ok(0);
    }
    if num(o, "presample").unwrap_or(0.0) != 0.0 {
        bail!("DSGE-VAR does not support the presample option.");
    }
    let lags = num(o, "dsge_varlag").unwrap_or(4.0) as usize;
    let first_obs = num(o, "first_obs").unwrap_or(1.0) as usize;
    if lags > 0 && first_obs <= lags {
        bail!(
            "makedataset: first_obs ({first_obs}) - {lags} cannot be less than the first date in the dataset (1)!\nReduce the number of lags in the VAR model or increase the value of first_obs\nto at least first_obs={}.",
            lags + 1
        );
    }
    Ok(lags)
}

/// `heteroskedastic_filter` (dynare_estimation_init.m): `M_.heteroskedastic_shocks`
/// for the sample of `data`, which starts at `first_obs` (1-based, of the original dataset).
pub fn attach(mf: &ModFile, o: &Options, data: &mut LikelihoodData, first_obs: usize) -> Result<()> {
    data.filter_initial_state = filter_initial_state(mf)?;
    if flag(o, "heteroskedastic_filter") {
        if flag(o, "analytic_derivation") {
            bail!("estimation option conflict: analytic_derivation isn't available for heteroskedastic_filter");
        }
        data.hetero = Some(crate::kalman::hetero::HeteroSpec::new(
            &mf.q3.heteroskedastic_shocks,
            mf.model.n_exo(),
            first_obs,
            data.y.ncols(),
            &mf.model.exo_names,
        )?);
    }
    // ---- estimation completion: kalman_algo = 5 (likelihood and smoother) ----
    if data.kalman.kalman_algo == 5 && crate::estimation::q3::dsge_var_weight(mf, o)?.is_none() {
        data.skewed = Some(crate::estimation::skewed::SkewedSpec::new(mf, o, &data.kalman)?);
    }
    // ---- end estimation completion ----
    Ok(())
}

/// `set_Kalman_starting_values.m`: `T*a` with `a(mf0)` = the `filter_initial_state` values of
/// the state variables minus their steady state (logs with `loglinear`), on the state space of
/// `idx` (restricted for the likelihood, [`crate::solver::StateSpaceIndex::full`] for the smoother).
pub fn kalman_a0(
    dsge: &DsgeModel,
    res: &crate::dsge::Resolved,
    idx: &crate::solver::StateSpaceIndex,
    entries: &[(usize, f64)],
) -> Vec<f64> {
    let (t, _) = crate::solver::kalman_transition_matrix(&res.dr, idx);
    let mut a = crate::linalg::Vector::zeros(t.nrows());
    for &pos in &idx.restrict_columns {
        // bayestopt_.mf0 (likelihood) = dr.restrict_columns (smoother)
        let endo = res.dr.order_var[idx.restrict_var_list[pos]];
        if let Some(&(_, v)) = entries.iter().find(|(e, _)| *e == endo) {
            let ys = res.steady_state[endo];
            a[pos] = if dsge.loglinear { v.ln() - ys.ln() } else { v - ys };
        }
    }
    (&t * a).iter().copied().collect()
}

/// `M_.filter_initial_state` entries as (endogenous index, value): `x(0)` is `x`, `x(-k)` the
/// aux variable of the k-th lag (the preprocessor's `searchAuxiliaryVars`).
pub fn filter_initial_state(mf: &ModFile) -> Result<Vec<(usize, f64)>> {
    mf.q3
        .filter_initial_state
        .iter()
        .map(|&(e, lag, v)| {
            if lag == 0 {
                return Ok((e, v));
            }
            mf.model
                .aux_vars
                .iter()
                .find(|a| a.ty == 1 && a.orig_index == Some(e) && a.orig_lead_lag == Some(lag))
                .map(|a| (a.endo_index, v))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "filter_initial_state: {}({lag}) is not a state of the model",
                        mf.model.endo_names[e]
                    )
                })
        })
        .collect()
}

/// The objectives of the stream that replace `dsge_likelihood.m` (dynare_estimation_1.m's
/// `objective_function`): `dsge_var_likelihood`.
pub struct Extras {
    /// `kalman_algo = 5`: calibrated skewness and the filter options.
    pub skewed: Option<(Vec<f64>, crate::kalman::skewed::SkewedOptions)>,
    pub dsge_var: Option<(VarMoments, PriorWeight)>,
    /// `conditional_likelihood` (dsge_conditional_likelihood_1.m).
    pub conditional: bool,
    /// `options_.qz_criterium`, `options_.lyapunov_complex_threshold` (`dsge_var_likelihood.m`).
    pub lyap: (f64, f64),
}

impl Extras {
    pub fn new(mf: &ModFile, o: &Options, dsge: &DsgeModel, data: &LikelihoodData) -> Result<Self> {
        let dsge_var = match dsge_var_weight(mf, o)? {
            Some(w) => {
                DsgeVarPosterior::check_model(dsge)?;
                let lags = num(o, "dsge_varlag").unwrap_or(4.0) as usize;
                Some((VarMoments::new(&data.y.transpose(), lags)?, w))
            }
            None => None,
        };
        let conditional = flag(o, "conditional_likelihood") && dsge_var.is_none();
        if conditional {
            crate::estimation::conditional_likelihood::ConditionalPosterior::check(dsge, data)?;
        }
        // kalman_algo = 5 (dynare_estimation_init.m checks)
        let skew0 = crate::estimation::skewed::base_skew(mf)?;
        let has_skew = skew0.iter().any(|&v| v != 0.0)
            || dsge.estimated.iter().any(|e| matches!(e.kind, crate::modfile::EstimatedKind::Skew(_)));
        // ---- estimation completion: checks and options in SkewedSpec::new (q3::attach) ----
        let skewed = if data.kalman.kalman_algo == 5 && dsge_var.is_none() {
            crate::estimation::skewed::SkewedPosterior::check(dsge, data)?;
            let spec = match &data.skewed {
                Some(s) => s.clone(),
                None => crate::estimation::skewed::SkewedSpec::new(mf, o, &data.kalman)?,
            };
            let _ = skew0;
            Some((spec.base_skew, spec.opts))
        } else {
            if has_skew {
                bail!(
                    "dynare_estimation_init: Skewed shocks have been declared (M_.Skew_e ≠ 0) or skewness parameters are estimated, but estimation with skewed shocks is only supported with the pruned skewed Kalman filter. You need to set kalman_algo=5."
                );
            }
            None
        };
        let lyap = (data.kalman.qz_criterium, data.kalman.lyapunov_complex_threshold);
        Ok(Extras { skewed, dsge_var, conditional, lyap })
    }

    /// The objective with the given bounds (mode finding: `dsge.bounds`; MCMC: the prior support).
    pub fn posterior<'a>(
        &'a self,
        dsge: &'a DsgeModel,
        data: &'a LikelihoodData,
        bounds: &'a Bounds,
    ) -> Option<Box<dyn Objective + 'a>> {
        if let Some((s, so)) = &self.skewed {
            return Some(Box::new(crate::estimation::skewed::SkewedPosterior {
                dsge,
                data,
                bounds,
                base_skew: s.clone(),
                opts: so.clone(),
            }));
        }
        if self.conditional {
            return Some(Box::new(crate::estimation::conditional_likelihood::ConditionalPosterior {
                dsge,
                data,
                bounds,
            }));
        }
        self.dsge_var.as_ref().map(|(m, w)| {
            Box::new(DsgeVarPosterior {
                dsge,
                moments: m,
                weight: *w,
                bounds,
                qz_criterium: self.lyap.0,
                lyapunov_complex_threshold: self.lyap.1,
            }) as Box<dyn Objective + 'a>
        })
    }

    /// `bayesian_irf` with `dsge_var`: `oo_.PosteriorIRF.bvardsge` (see [`crate::estimation::dsge_var_irf`]).
    #[allow(clippy::too_many_arguments)]
    pub fn posterior_irf(
        &self,
        dsge: &DsgeModel,
        bounds: &Bounds,
        chains: &[crate::estimation::mcmc::Chain],
        o: &crate::analysis::StochSimulOptions,
        out_dir: &std::path::Path,
        fname: &str,
        seed: u64,
    ) -> Result<()> {
        let Some((m, w)) = &self.dsge_var else { return Ok(()) };
        let post = DsgeVarPosterior {
            dsge,
            moments: m,
            weight: *w,
            bounds,
            qz_criterium: self.lyap.0,
            lyapunov_complex_threshold: self.lyap.1,
        };
        crate::estimation::dsge_var_irf::run(&post, chains, o, out_dir, fname, seed)
    }

    /// `oo_.dsge_var.posterior_mode` (PHI_tilde, SIGMA_u_tilde, iXX, prior) as JSON.
    pub fn posterior_mode_json(&self, dsge: &DsgeModel, x: &[f64]) -> Option<serde_json::Value> {
        let (m, w) = self.dsge_var.as_ref()?;
        let p = DsgeVarPosterior {
            dsge,
            moments: m,
            weight: *w,
            bounds: &dsge.bounds,
            qz_criterium: self.lyap.0,
            lyapunov_complex_threshold: self.lyap.1,
        };
        let e = p.evaluate(x).ok()?;
        let rows = |a: &Mat| (0..a.nrows()).map(|i| a.row(i).iter().copied().collect::<Vec<f64>>()).collect::<Vec<_>>();
        let k = (m.nvobs * m.nlags + 1) as f64;
        let ass = (e.weight * m.nobs as f64).trunc();
        Some(serde_json::json!({
            "PHI_tilde": rows(&e.phi_tilde), "SIGMA_u_tilde": rows(&e.sigma_u_tilde), "iXX": rows(&e.ixx),
            "prior": {"SIGMA_u_star": rows(&e.sigma_u_star), "PHI_star": rows(&e.phi_star), "ArtificialSampleSize": ass,
                      "DF": ass - k - m.nvobs as f64, "iGXX_star": rows(&e.igxx_star)},
            "dsge_prior_weight": e.weight,
        }))
    }
}
