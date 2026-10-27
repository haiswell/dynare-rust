// SPDX-License-Identifier: GPL-3.0-or-later
//! Estimation completion: the remaining `estimation` options of Dynare 7.2.
//!
//! * `no_init_estimation_check_first_obs` and the stochastic-singularity checks of
//!   `initial_estimation_checks.m` (fewer shocks than observables, shocks calibrated with a
//!   zero variance, in the first period and in the rest of the sample);
//! * `posterior_nograph` / `posterior_graph` (`options_.no_graph.posterior`: pm3.m and
//!   PosteriorIRF.m figures);
//! * `frequentist_smoother` (dynare_estimation_1.m: the classical smoother after the mode
//!   or the MCMC);
//! * `use_univariate_smoother_if_singularity_is_detected` (DsgeSmoother.m: error 422 unless
//!   set, see [`crate::kalman::KalmanOptions::smoother_singularity_error`]);
//! * `dr = cycle_reduction | logarithmic_reduction`, `dr_*_tol`, `dr_*_maxiter`,
//!   `qz_zero_threshold` in the likelihood (dyn_first_order_solver.m through
//!   dynare_resolve.m);
//! * `no_posterior_kernel_density` (`options_.estimation.moments_posterior_density.indicator`:
//!   the kernel density estimates of posterior_moments.m for pm3's objects and the posterior
//!   theoretical moments);
//! * `huge_number` (dynare_minimize_objective.m's finite bounds, mode_check.m);
//! * `nonlinear_filter_initialization` (`options_.particle.initialization`, read by
//!   [`crate::nonlinear::ParticleOptions::from_options`]).

use anyhow::{Result, bail};

use crate::dsge::DsgeModel;
use crate::kalman::KalmanOptions;
use crate::linalg::Mat;
use crate::modfile::{EstimatedKind, ModFile, OptValue, Options, opt};

/// The options of this module (accepted by the estimation driver without a warning).
pub const OPTIONS: &[&str] = &[
    "no_init_estimation_check_first_obs",
    "posterior_nograph",
    "posterior_graph",
    "frequentist_smoother",
    "use_univariate_smoother_if_singularity_is_detected",
    "dr",
    "dr_cycle_reduction_tol",
    "dr_cycle_reduction_maxiter",
    "dr_logarithmic_reduction_tol",
    "dr_logarithmic_reduction_maxiter",
    "qz_zero_threshold",
    "no_posterior_kernel_density",
    "contemporaneous_correlation",
    "huge_number",
    "nonlinear_filter_initialization",
    "estimate_initial_states_endogenous_prior",
];

/// A flag or BOOLEAN option: `None` when absent.
pub fn bool_opt(o: &Options, k: &str) -> Option<bool> {
    opt(o, k).map(|v| match v {
        OptValue::Flag => true,
        OptValue::Num(x) => *x != 0.0,
        OptValue::Str(s) => !matches!(s.trim_matches(|c| c == '\'' || c == '"'), "false" | "0"),
        _ => true,
    })
}

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        Some(OptValue::Str(s)) => s.trim().parse().ok(),
        _ => None,
    }
}

/// `options_.no_graph.posterior`: `posterior_nograph` sets it, `posterior_graph` clears it
/// (the last one given wins, as the preprocessor writes them in order).
pub fn no_graph_posterior(o: &Options) -> bool {
    let mut v = false;
    for (k, _) in o {
        match k.as_str() {
            "posterior_nograph" => v = true,
            "posterior_graph" => v = false,
            _ => {}
        }
    }
    v
}

/// `options_.estimation.moments_posterior_density.indicator` (default true).
pub fn kernel_density(o: &Options) -> bool {
    !bool_opt(o, "no_posterior_kernel_density").unwrap_or(false)
}

/// DsgeSmoother.m: `print_info(422)` errors unless
/// `use_univariate_smoother_if_singularity_is_detected` (or `noprint`) is set.
pub fn kalman_options(o: &Options, k: &mut KalmanOptions) {
    if bool_opt(o, "estimate_initial_states_endogenous_prior").unwrap_or(false) {
        k.lik_init = 2; // dynare_estimation_init.m (with Harvey_scale_factor = 0)
    }
    // dynare_estimation_init.m:546-550 resets options_.noconstant = 0 in both branches: every
    // estimation / smoother call runs with the constant of the measurement equation
    k.noconstant = false;
    k.smoother_singularity_error = !bool_opt(o, "use_univariate_smoother_if_singularity_is_detected").unwrap_or(false)
        && !bool_opt(o, "noprint").unwrap_or(false);
}

/// dynare_estimation_1.m: does the classical (frequentist) smoother run after the mode
/// finding / MCMC? Dynare runs it unless the Bayesian smoother ran (`smoother` with an
/// MCMC), when `frequentist_smoother` is true (its default). dynare-rust keeps its own default
/// (the classical smoother only when `smoother` or `filtered_vars` is requested, plus at the
/// posterior mean after an MCMC with `smoother`) unless `frequentist_smoother` is given:
/// `false` skips it (except the `mode_compute = 0` smoother-only run, which Dynare does before
/// this test), `true` runs it whatever `smoother` says.
pub fn classical_smoother(o: &Options, has_mcmc: bool, smoother_only: bool) -> bool {
    let requested = flag(o, "smoother") || flag(o, "filtered_vars");
    if smoother_only || (has_mcmc && flag(o, "smoother")) {
        return requested;
    }
    bool_opt(o, "frequentist_smoother").unwrap_or(requested)
}

/// plot_classical_smoother_results.m runs with the frequentist smoother (Dynare's branch);
/// dynare-rust's default plots it without an MCMC when `smoother` is set.
pub fn plot_classical_smoother(o: &Options, has_mcmc: bool) -> bool {
    let fs = bool_opt(o, "frequentist_smoother");
    if has_mcmc {
        !flag(o, "smoother") && fs == Some(true)
    } else {
        fs.unwrap_or(true) && (flag(o, "smoother") || fs == Some(true))
    }
}

fn flag(o: &Options, k: &str) -> bool {
    bool_opt(o, k).unwrap_or(false)
}

/// `options_.huge_number` of the estimation (default 1e7).
pub fn huge_number(o: &Options) -> Option<f64> {
    num(o, "huge_number")
}

/// `dr = cycle_reduction | logarithmic_reduction` and `qz_zero_threshold` in the likelihood
/// (stochastic_solvers.m -> dyn_first_order_solver.m with `options_.dr_cycle_reduction`,
/// `options_.dr_logarithmic_reduction`, `options_.qz_zero_threshold`); `ss` carries Dynare's
/// global options as the commands so far set them. The model is returned unchanged with the
/// defaults (QZ, qz_zero_threshold = 1e-6).
pub fn with_solver_options(mut dsge: DsgeModel, ss: &crate::analysis::StochSimulOptions) -> Result<DsgeModel> {
    let s = ss.solver_options();
    let qz = matches!(s.method, crate::solver::FirstOrderMethod::Qz);
    let default = crate::solver::SolverOptions::default();
    if qz && s.qz_zero_threshold == default.qz_zero_threshold && s.qz_criterium == default.qz_criterium {
        return Ok(dsge);
    }
    // options_.qz_criterium as select_qz_criterium_value.m left it (1-1e-6 for lik_init = 1)
    let opts = s;
    if !qz {
        let solver = crate::solver::FirstOrderSolver::new(&dsge.model.lead_lag_incidence, dsge.model.n_exo(), opts);
        if !solver.reduction_applicable() {
            if ss.dr_cycle_reduction {
                bail!(
                    "The cycle reduction algorithm can't be used when the coefficient matrix for current variables isn't invertible"
                );
            }
            bail!(
                "The logarithmic reduction algorithm can't be used when the coefficient matrix for current variables isn't invertible"
            );
        }
    }
    dsge.set_solver_options(opts);
    Ok(dsge)
}

/// dynare_estimation_init.m's checks of `estimate_initial_states_endogenous_prior`; true when
/// the option is set (the joint estimation of the initial states: [`crate::estimation::init_states`]).
pub fn check_initial_states(o: &Options) -> Result<bool> {
    if !bool_opt(o, "estimate_initial_states_endogenous_prior").unwrap_or(false) {
        return Ok(false);
    }
    if bool_opt(o, "analytic_derivation").unwrap_or(false) {
        bail!(
            "estimation option conflict: estimate_initial_states_endogenous_prior isn't available for analytic_derivation"
        );
    }
    let slice = matches!(opt(o, "posterior_sampling_method"), Some(OptValue::Str(s)) if s.trim_matches(|c| c == '\'' || c == '"') == "slice");
    if !slice {
        bail!("Init state estimation with endogenous prior is only compatible with slice sampler");
    }
    Ok(true)
}

/// The stochastic-singularity checks of initial_estimation_checks.m. `sample`: nobs x pp
/// (NaN = missing), `x`: the initial parameter vector (`M_` after `set_all_parameters`),
/// `hetero`: `M_.heteroskedastic_shocks` with `heteroskedastic_filter`.
pub fn initial_estimation_checks(
    mf: &ModFile,
    dsge: &DsgeModel,
    sample: &Mat,
    x: &[f64],
    o: &Options,
    hetero: Option<&crate::kalman::hetero::HeteroSpec>,
) -> Result<()> {
    let (nobs, pp) = (sample.nrows(), sample.ncols());
    let count = |t: usize| (0..pp).filter(|&j| !sample[(t, j)].is_nan()).count();
    // max(sum(~isnan(dataset_.data(2:end,:)),2)): empty (no test) with one observation
    let max_obs = (1..nobs).map(count).max();
    let init_obs = if nobs > 0 { count(0) } else { 0 };
    // estim_params_.H_entries_to_check_for_positive_definiteness (get_matrix_entries_for_psd_check.m)
    let mut me_rows: Vec<usize> =
        (0..dsge.base_h.nrows()).filter(|&i| dsge.base_h.row(i).iter().any(|&v| v != 0.0)).collect();
    for e in &dsge.estimated {
        if let EstimatedKind::MeStdErr(k) = e.kind
            && let Some(i) = mf.varobs_position(k)
        {
            me_rows.push(i);
        }
    }
    me_rows.sort_unstable();
    me_rows.dedup();
    let non_zero_me = me_rows.len();
    let n_exo = dsge.model.n_exo();
    let (_, sigma_e) = dsge.set_all_parameters(x);
    let nonzero_shocks = (0..n_exo).filter(|&i| sigma_e[(i, i)] != 0.0).count();
    let first_obs_check = bool_opt(o, "no_init_estimation_check_first_obs").unwrap_or(false);
    let mut warn = false;
    if max_obs.is_some_and(|m| m > n_exo + non_zero_me) {
        bail!(
            "initial_estimation_checks:: Estimation can't take place because there are less declared shocks than observed variables!"
        );
    }
    if init_obs > n_exo + non_zero_me {
        if first_obs_check {
            warn = true;
        } else {
            bail!(
                "initial_estimation_checks:: Estimation can't take place because there are less declared shocks than observed variables in first period!"
            );
        }
    }
    match hetero {
        Some(hs) => {
            let base: Vec<usize> = (0..n_exo).filter(|&i| sigma_e[(i, i)] != 0.0).collect();
            let zero: Vec<usize> = (0..n_exo).filter(|&i| sigma_e[(i, i)] == 0.0).collect();
            for t in 0..nobs {
                let at = |m: &Mat, i: usize| if t < m.ncols() { m[(i, t)] } else { f64::NAN };
                let mut nz = base.len() as i64;
                nz -= base.iter().filter(|&&i| at(&hs.qscale, i) == 0.0).count() as i64;
                nz -= base.iter().filter(|&&i| at(&hs.qvalue, i) == 0.0).count() as i64;
                nz += zero.iter().filter(|&&i| at(&hs.qvalue, i) > 0.0).count() as i64;
                if count(t) as i64 > nz + non_zero_me as i64 {
                    bail!(
                        "initial_estimation_checks:: Estimation can't take place because too many shocks have been calibrated with a zero variance: Check heteroskedastic block and shocks calibration!"
                    );
                }
            }
        }
        None => {
            if max_obs.is_some_and(|m| m > nonzero_shocks + non_zero_me) {
                bail!(
                    "initial_estimation_checks:: Estimation can't take place because too many shocks have been calibrated with a zero variance!"
                );
            }
        }
    }
    if init_obs > nonzero_shocks + non_zero_me {
        if first_obs_check {
            warn = true;
        } else {
            bail!(
                "initial_estimation_checks:: Estimation can't take place because too many shocks have been calibrated with a zero variance in first period!"
            );
        }
    }
    if warn {
        println!("ESTIMATION_CHECKS: You decided to ignore test of stochastic singularity in first_obs.");
        println!(
            "ESTIMATION_CHECKS: If this was not done on purpose (typically when observing a stock variable [capital] in first period, on top of its flow [investment]),"
        );
        println!("ESTIMATION_CHECKS: it may lead to a crash or provide undesired/wrong results later on!");
    }
    Ok(())
}

/// The `density` output of posterior_moments.m (`kernel_density_estimate` on 2^9 points with
/// the rule-of-thumb bandwidth, Gaussian kernel) of the draws `xx`: 512 x 2 (abscissa,
/// density), NaN when the posterior variance is <= 1e-12.
pub fn posterior_density(xx: &[f64]) -> Mat {
    const NGRID: usize = 512;
    let mut v = xx.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = if v.len() > 1 { v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0) } else { 0.0 };
    if var.is_nan() || var <= 1e-12 {
        return Mat::from_element(NGRID, 2, f64::NAN);
    }
    let h = crate::estimation::diagnostics::mh_optimal_bandwidth(&v);
    let (a, f) = crate::estimation::diagnostics::kernel_density_estimate(&v, NGRID, h);
    Mat::from_fn(NGRID, 2, |i, j| if j == 0 { a[i] } else { f[i] })
}

/// pm3.m's `Density(:,:,:,i)` of one variable (draws `stock[b][(t, i)]`), as a 512 x (2*n2)
/// matrix whose columns are [abscissa, density] of period 1, 2, ... (MATLAB's 512x2xn2
/// array in column-major order).
pub fn pm3_density(stock: &[Mat], i: usize) -> Mat {
    let n2 = stock.first().map_or(0, |m| m.nrows());
    let mut out = Mat::zeros(512, 2 * n2);
    for t in 0..n2 {
        let xx: Vec<f64> = stock.iter().map(|m| m[(t, i)]).collect();
        let d = posterior_density(&xx);
        out.columns_mut(2 * t, 2).copy_from(&d);
    }
    out
}
