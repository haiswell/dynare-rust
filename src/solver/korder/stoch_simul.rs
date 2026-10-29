// SPDX-License-Identifier: GPL-3.0-or-later
//! `stoch_simul` at `order = 2 | 3` (stream K): port of the higher-order branches of
//! `stochastic_solver/stoch_simul.m` — decision rules (`disp_dr.m` at order 2), simulation
//! (`simult.m`), moments (`disp_th_moments.m` at order 2, `disp_th_moments_pruned_state_space.m`
//! with `pruning`, `disp_moments.m` on simulated data when `periods > 0`) and generalized
//! IRFs (`irf.m`). Order 1 belongs to `analysis` (stream S); the driver calls [`run`] when the
//! requested order is at least 2 (Dynare's default order is 2).
//!
//! Random draws (simulations, GIRFs) come from ChaCha8 seeded with `--seed`, not MATLAB's
//! stream, so simulated moments and GIRFs agree with Dynare statistically only (with the same
//! draws they agree to rounding, see `tests/korder.rs`).

use super::pruned::{pruned_state_space_system_qz, second_order_mean};
use super::simulate::{draw_shocks, girf, simult};
use super::{KOrderDr, solve_model_at};
use crate::graphics::{self, GraphFormat, GraphOptions};
use crate::linalg::Mat;
use crate::modfile::{ModFile, OptValue, Options, opt};
use crate::solver::SolverOptions;
use anyhow::{Context, Result, anyhow, bail};
use rand::SeedableRng;
use serde_json::json;
use std::path::Path;

/// The options of `stoch_simul` this branch reads (Dynare defaults).
#[derive(Debug, Clone)]
pub struct StochSimulOptions {
    pub order: usize,
    pub pruning: bool,
    pub irf: usize,
    pub replic: usize,
    pub drop: usize,
    pub periods: usize,
    pub ar: usize,
    pub nomoments: bool,
    pub noprint: bool,
    pub nofunctions: bool,
    pub nograph: bool,
    pub irf_shocks: Vec<String>,
    pub graph_formats: Vec<GraphFormat>,
    pub qz_criterium: f64,
    /// The persistent `options_` of the analysis (filters, `relative_irf`, decompositions,
    /// `tex`, ...); `None`: Dynare's defaults (the fields above take precedence).
    pub q2: Option<crate::analysis::StochSimulOptions>,
    /// Where `tex` tables go (`<dir>/<fname>/latex`).
    pub tex_ctx: Option<crate::analysis::OutputContext>,
}

impl Default for StochSimulOptions {
    fn default() -> Self {
        StochSimulOptions {
            order: 2,
            pruning: false,
            irf: 40,
            replic: 50,
            drop: 100,
            periods: 0,
            ar: 5,
            nomoments: false,
            noprint: false,
            nofunctions: false,
            nograph: false,
            irf_shocks: Vec::new(),
            graph_formats: vec![GraphFormat::Pdf],
            qz_criterium: 1.0 + 1e-6,
            q2: None,
            tex_ctx: None,
        }
    }
}

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

fn names(v: &OptValue) -> Vec<String> {
    match v {
        OptValue::Str(s) => vec![s.clone()],
        OptValue::List(l) => l.iter().flat_map(names).collect(),
        _ => Vec::new(),
    }
}

/// Requested approximation order (`order` option, default 2).
pub fn requested_order(o: &Options) -> usize {
    num(o, "order").map(|v| v as usize).unwrap_or(2)
}

const SUPPORTED: &[&str] = &[
    "order",
    "pruning",
    "irf",
    "replic",
    "drop",
    "periods",
    "ar",
    "nomoments",
    "noprint",
    "nofunctions",
    "nograph",
    "irf_shocks",
    "graph_format",
    "nodisplay",
    "k_order_solver",
    "qz_criterium",
    "nocorr",
    "nodecomposition",
    "tex",
    "nomodelsummary",
    "print",
];

impl StochSimulOptions {
    pub fn from_options(o: &Options) -> StochSimulOptions {
        let mut s = StochSimulOptions { order: requested_order(o), ..Default::default() };
        for (k, _) in o {
            if !SUPPORTED.contains(&k.as_str()) {
                println!(
                    "WARNING: stoch_simul option `{k}` is not supported at order {} by dynare-rust and is ignored.",
                    s.order
                );
            }
        }
        s.pruning = opt(o, "pruning").is_some();
        let us = |k: &str, d: usize| num(o, k).map(|v| v as usize).unwrap_or(d);
        s.irf = us("irf", s.irf);
        s.replic = us("replic", s.replic);
        s.drop = us("drop", s.drop);
        s.periods = us("periods", s.periods);
        s.ar = us("ar", s.ar);
        s.nomoments = opt(o, "nomoments").is_some();
        s.noprint = opt(o, "noprint").is_some();
        s.nofunctions = opt(o, "nofunctions").is_some();
        s.nograph = opt(o, "nograph").is_some();
        if let Some(v) = opt(o, "irf_shocks") {
            s.irf_shocks = names(v);
        }
        if let Some(v) = opt(o, "graph_format") {
            let f: Vec<GraphFormat> = names(v).iter().filter_map(|n| GraphFormat::parse(n)).collect();
            if !f.is_empty() {
                s.graph_formats = f;
            }
        }
        if let Some(v) = num(o, "qz_criterium") {
            s.qz_criterium = v;
        }
        s
    }
}

impl StochSimulOptions {
    /// Stream K's options from the persistent analysis options (Dynare's global `options_`).
    pub fn from_analysis(a: &crate::analysis::StochSimulOptions) -> StochSimulOptions {
        StochSimulOptions {
            order: a.order,
            pruning: a.pruning,
            irf: a.irf,
            replic: a.replic,
            drop: a.drop,
            periods: a.periods,
            ar: a.ar,
            nomoments: a.nomoments,
            noprint: a.noprint,
            nofunctions: a.nofunctions,
            nograph: a.nograph,
            irf_shocks: a.irf_shocks.clone().unwrap_or_default(),
            graph_formats: a.graph_format.clone(),
            qz_criterium: a.qz_criterium,
            q2: Some(a.clone()),
            tex_ctx: None,
        }
    }

    /// The analysis options in force, with this struct's fields.
    pub fn analysis(&self) -> crate::analysis::StochSimulOptions {
        let mut a = self.q2.clone().unwrap_or_default();
        a.order = self.order;
        a.pruning = self.pruning;
        a.irf = self.irf;
        a.replic = self.replic;
        a.drop = self.drop;
        a.periods = self.periods;
        a.ar = self.ar;
        a.nomoments = self.nomoments;
        a.noprint = self.noprint;
        a.nofunctions = self.nofunctions;
        a.nograph = self.nograph;
        a.graph_format = self.graph_formats.clone();
        a.qz_criterium = self.qz_criterium;
        a
    }
}

/// `disp_th_moments_pruned_state_space.m` (`pruning`, order 2 or 3).
#[allow(clippy::too_many_arguments)]
fn pruned_moments(
    m: &crate::model::Model,
    dr: &KOrderDr,
    sol1: &crate::analysis::LinearSolution,
    var_list: &[usize],
    names: &[String],
    a: &crate::analysis::StochSimulOptions,
    order: usize,
    tex: Option<&crate::analysis::TexNames>,
    out: &mut StochSimulOutput,
) -> Result<()> {
    use crate::analysis::table::{dyntable, max_len};
    if a.one_sided_hp_filter != 0.0 || a.hp_filter != 0.0 || a.bandpass.is_some() {
        bail!("disp_th_moments:: theoretical moments incompatible with filtering. Use simulated moments instead");
    }
    let f = &dr.first;
    let sigma_e = &sol1.sigma_e;
    let obs: Vec<usize> = var_list.iter().map(|&j| f.inv_order_var[j]).collect();
    let p = pruned_state_space_system_qz(dr, sigma_e, &obs, a.ar, a.qz_criterium)?;
    let nv = var_list.len();
    let tl: Vec<String> = tex.map(|t| var_list.iter().map(|&j| t.endo[j].clone()).collect()).unwrap_or_default();
    let latex = |title: &str,
                 key: &str,
                 by: &str,
                 h: &[String],
                 l: &[String],
                 z: &Mat,
                 lh: usize,
                 w: usize,
                 pr: usize|
     -> Result<()> {
        match tex {
            Some(t) => crate::analysis::q2::latex_table(a, &t.ctx, by, title, key, h, l, z, lh, w, pr),
            None => Ok(()),
        }
    };
    let mut s = String::new();
    let pr = !a.noprint;
    let z = Mat::from_fn(nv, 3, |i, j| match j {
        0 => p.e_y[i],
        1 => p.var_y[(i, i)].sqrt(),
        _ => p.var_y[(i, i)],
    });
    if pr {
        let title = "THEORETICAL MOMENTS BASED ON PRUNED STATE SPACE";
        let h: Vec<String> = ["VARIABLE", "MEAN", "STD. DEV.", "VARIANCE"].iter().map(|x| x.to_string()).collect();
        s += &dyntable(title, &h, names, &z, max_len(names) + 2, 11, 4);
        latex(title, "th_moments", "disp_th_moments_pruned_state_space.m", &h, &tl, &z, max_len(&tl) + 2, 11, 4)?;
    }
    out.mean = p.e_y.clone();
    out.var = p.var_y.clone();
    out.autocorr = p.corr_yi.clone();
    let i1: Vec<usize> = (0..nv).filter(|&i| p.var_y[(i, i)].abs() > 1e-12).collect();
    if i1.is_empty() {
        if pr {
            s += "\nAll endogenous are constant or non stationary, not displaying correlations and auto-correlations\n\n";
        }
        print!("{s}");
        return Ok(());
    }
    if !a.nocorr {
        if a.contemporaneous_correlation {
            out.contemporaneous_correlation = Some(p.corr_y.clone());
        }
        if pr {
            let title = "MATRIX OF CORRELATIONS BASED ON PRUNED STATE SPACE";
            let mut h = vec!["Variables".to_string()];
            h.extend(names.iter().cloned());
            s.push('\n');
            s += &dyntable(title, &h, names, &p.corr_y, max_len(names) + 2, 8, 4);
            let mut ht = vec!["Variables".to_string()];
            ht.extend(tl.iter().cloned());
            latex(
                title,
                "th_corr_matrix",
                "disp_th_moments_pruned_state_space.m",
                &ht,
                &tl,
                &p.corr_y,
                max_len(&tl) + 2,
                8,
                4,
            )?;
        }
    }
    if a.ar > 0 && pr {
        let z = Mat::from_fn(i1.len(), a.ar, |r, k| p.corr_yi[k][(i1[r], i1[r])]);
        let l1: Vec<String> = i1.iter().map(|&k| names[k].clone()).collect();
        let mut h = vec!["Order ".to_string()];
        h.extend((1..=a.ar).map(|k| k.to_string()));
        s.push('\n');
        let title = "COEFFICIENTS OF AUTOCORRELATION BASED ON PRUNED STATE SPACE";
        s += &dyntable(title, &h, &l1, &z, max_len(&l1) + 2, 8, 4);
        let t1: Vec<String> = if tex.is_some() { i1.iter().map(|&k| tl[k].clone()).collect() } else { Vec::new() };
        latex(
            title,
            "th_autocorr_matrix",
            "disp_th_moments_pruned_state_space.m",
            &h,
            &t1,
            &z,
            max_len(&t1) + 2,
            8,
            4,
        )?;
    }
    if order == 2 && !a.nodecomposition && m.n_exo() > 1 {
        // first-order shares (compute_variance_decomposition.m; no aggregate-variance check
        // under pruning) and conditional decomposition, titled APPROXIMATED
        let mut a1 = a.clone();
        a1.order = 2;
        a1.conditional_variance_decomposition.clear();
        a1.nodecomposition = false;
        let th = crate::analysis::moments::th_autocovariances(&sol1.dr, sigma_e, var_list, &a1.moment_options())?;
        let vd = th.vardec.clone().unwrap_or_else(|| Mat::from_element(nv, m.n_exo(), f64::NAN)) * 100.0;
        if pr {
            let mut h = vec![" ".to_string()];
            h.extend(m.exo_names.iter().cloned());
            s.push('\n');
            let title = "APPROXIMATED VARIANCE DECOMPOSITION (in percent)";
            s += &dyntable(title, &h, names, &vd, max_len(names) + 2, 8, 2);
            if let Some(t) = tex {
                let mut ht = vec![" ".to_string()];
                ht.extend(t.exo.iter().cloned());
                latex(
                    title,
                    "th_var_decomp_uncond",
                    "display_unconditional_variance_decomposition.m",
                    &ht,
                    &tl,
                    &vd,
                    max_len(&tl) + 2,
                    8,
                    2,
                )?;
            }
        }
        out.variance_decomposition = Some(vd);
        let steps = &a.conditional_variance_decomposition;
        if !steps.is_empty() {
            let c = crate::analysis::vardec::conditional_variance_decomposition(&sol1.dr, sigma_e, steps, var_list)?;
            if pr {
                s += "\nAPPROXIMATED CONDITIONAL VARIANCE DECOMPOSITION (in percent)\n";
                let mut h = vec![" ".to_string()];
                h.extend(m.exo_names.iter().cloned());
                for (k, cm) in c.iter().enumerate() {
                    s += &format!("Period {}:\n", steps[k]);
                    s += &dyntable("", &h, names, &(cm * 100.0), max_len(names) + 2, 8, 2);
                }
            }
            out.conditional_variance_decomposition = Some(c);
        }
    }
    print!("{s}");
    Ok(())
}

/// Results of [`stoch_simul`] (declaration order for variables).
#[derive(Debug, Clone)]
pub struct StochSimulOutput {
    pub dr: KOrderDr,
    /// Indices (declaration order) of the variables in `var_list`.
    pub var_list: Vec<usize>,
    /// `oo_.irfs`: (`<var>_<shock>`, response over `irf` periods).
    pub irfs: Vec<(String, Vec<f64>)>,
    /// `oo_.mean`, `oo_.var`, `oo_.autocorr` for `var_list` (empty if not computed).
    pub mean: Vec<f64>,
    pub var: Mat,
    pub autocorr: Vec<Mat>,
    /// `oo_.skewness`, `oo_.kurtosis` of the simulated moments (empty otherwise).
    pub skewness: Vec<f64>,
    pub kurtosis: Vec<f64>,
    /// `oo_.endo_simul` when `periods > 0` (`n x periods`, declaration order).
    pub endo_simul: Option<Mat>,
    /// `oo_.exo_simul` of that simulation (`periods x n_exo`).
    pub exo_simul: Option<Mat>,
    /// `oo_.variance_decomposition` (percent), `oo_.conditional_variance_decomposition`
    /// (fractions per step), `oo_.contemporaneous_correlation`.
    pub variance_decomposition: Option<Mat>,
    pub conditional_variance_decomposition: Option<Vec<Mat>>,
    pub contemporaneous_correlation: Option<Mat>,
    /// `oo_.variance_decomposition_ME` (percent) and `oo_.conditional_variance_decomposition_ME`.
    pub variance_decomposition_me: Option<Mat>,
    pub conditional_variance_decomposition_me: Option<Vec<Mat>>,
}

/// Runs `stoch_simul(order >= 2)` on the calibrated model, printing like Dynare unless `noprint`.
pub fn stoch_simul(mf: &ModFile, so: &StochSimulOptions, var_names: &[String], seed: u64) -> Result<StochSimulOutput> {
    stoch_simul_at(mf, so, var_names, seed, None)
}

/// [`stoch_simul`] at a given steady state and parameter vector (`at = (ys, params)`) instead
/// of the calibrated steady state (the Ramsey steady state of a Ramsey model).
pub fn stoch_simul_at(
    mf: &ModFile,
    so: &StochSimulOptions,
    var_names: &[String],
    seed: u64,
    at: Option<(&[f64], &[f64])>,
) -> Result<StochSimulOutput> {
    if so.order < 2 {
        bail!("stoch_simul: dynare-rust supports order >= 2 here (got {})", so.order);
    }
    if mf.model.n_exo() == 0 {
        bail!("stoch_simul:: does not support having no varexo in the model.");
    }
    if mf.model.dynamic_derivatives(2).hessian_eq_zero() {
        // Dynare switches to order 1; here the order-2 machinery gives the same (zero)
        // corrections, so the output is the first-order one.
        println!("WARNING: stoch_simul: the Hessian is equal to zero (order 1 solution; corrections are zero).");
    }
    let (m, ys, params) = match at {
        Some((ys, p)) => (&mf.model, ys.to_vec(), p.to_vec()),
        None => {
            // stoch_simul.m: resol(0, M_, options_, ...) at the calibration with the
            // stoch_simul solve_algo (no estimation-specific steps)
            let (ys, params) =
                crate::analysis::calibrated_steady_state_params(mf, &mf.model.params.clone(), so.analysis().solve_algo)
                    .context("stoch_simul: steady state")?;
            (&mf.model, ys, params)
        }
    };
    let sigma_e = &mf.sigma_e;
    crate::analysis::skew::check_model(sigma_e, &mf.h, &mf.q3.skew_e)?; // ---- shocks/shock_paths: check_model.m
    // ---- k_order_pert.m's option checks ----
    let a = so.analysis();
    if a.partial_information {
        bail!(
            "stoch_simul: partial_information (PCL_resol / dr1_PI, which forces order = 1) is not ported by dynare-rust"
        );
    }
    // ---- stoch_simul: stochastic_solvers.m checks aim_solver before the k-order branch;
    // without k_order_solver (order 2) the first-order step honours dr=cycle_reduction|
    // logarithmic_reduction and loglinear fails after the second-order solution ----
    if a.aim_solver {
        bail!("Option \"aim_solver\" is incompatible with order >= 2");
    }
    let k_order = a.k_order_solver || so.order >= 3;
    if k_order {
        if a.loglinear {
            bail!("The loglinear-option currently only works at order 1");
        }
        if a.dr_cycle_reduction {
            bail!("Option dr=cycle_reduction is not compatible with k_order_solver");
        }
        if a.dr_logarithmic_reduction {
            bail!("Option dr=logarithmic_reduction is not compatible with k_order_solver");
        }
    }
    let sopts = if k_order {
        SolverOptions { qz_criterium: so.qz_criterium, qz_zero_threshold: a.qz_zero_threshold, ..Default::default() }
    } else {
        SolverOptions { qz_criterium: so.qz_criterium, ..a.solver_options() }
    };
    // ---- end stoch_simul ----
    let dr = solve_model_at(m, &ys, &mf.exo_steady_state(), &params, sigma_e, so.order, sopts)?;
    if a.loglinear {
        // stochastic_solvers.m, after dyn_second_order_solver
        bail!("Loglinear options currently only works at order 1");
    }
    let var_list: Vec<usize> = if var_names.is_empty() {
        (0..m.orig_endo_nbr).collect()
    } else {
        var_names
            .iter()
            .map(|v| m.endo_index(v).ok_or_else(|| anyhow!("stoch_simul: {v} is not an endogenous variable")))
            .collect::<Result<_>>()?
    };
    let f = &dr.first;
    if !so.noprint {
        // Dynare's MODEL SUMMARY (stoch_simul.m, dyntable)
        let s1 = crate::analysis::LinearSolution { dr: f.clone(), sigma_e: sigma_e.clone(), params: params.clone() };
        if !so.analysis().nomodelsummary {
            print!("{}", crate::analysis::model_summary_full(m, &s1, Some(&mf.q3.skew_e), so.analysis().me.as_ref())); // coskewness + M_.H
        }
        println!();
        if so.order == 2 && !so.nofunctions {
            print_dr(mf, &dr, &var_list, so.analysis().dr_display_tol);
        }
    }

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut out = StochSimulOutput {
        dr: dr.clone(),
        var_list: var_list.clone(),
        irfs: Vec::new(),
        mean: Vec::new(),
        var: Mat::zeros(0, 0),
        autocorr: Vec::new(),
        endo_simul: None,
        exo_simul: None,
        variance_decomposition: None,
        conditional_variance_decomposition: None,
        contemporaneous_correlation: None,
        variance_decomposition_me: None,
        conditional_variance_decomposition_me: None,
        skewness: vec![],
        kurtosis: vec![],
    };
    let names: Vec<String> = var_list.iter().map(|&j| m.endo_names[j].clone()).collect();

    // ---- the analysis options at order >= 2 (filters, variance decompositions,
    // relative_irf, irf_plot_threshold, contemporaneous_correlation, tex) ----
    let tex = match (&so.tex_ctx, a.tex) {
        (Some(c), true) => Some(crate::analysis::TexNames::from_modfile(mf, c)),
        _ => None,
    };
    let sol1 = crate::analysis::LinearSolution { dr: f.clone(), sigma_e: sigma_e.clone(), params: params.clone() };

    // simult.m
    let mut exo_simul = None;
    if so.periods > 0 {
        if so.periods <= so.drop {
            println!(
                "\nSTOCH_SIMUL error: The horizon of simulation is shorter than the number of observations to be dropped."
            );
            return Ok(out);
        }
        // ---- shocks/shock_paths: skewed shocks (simult.m, rand_multivariate_csn)
        let ex = if mf.q3.skew_e.is_empty() {
            draw_shocks(sigma_e, so.periods, 0, 1, &mut rng).remove(0)
        } else {
            use rand_distr::Distribution;
            crate::analysis::skew::draw_skewed_shocks(sigma_e, &mf.q3.skew_e, so.periods, &mut || {
                rand_distr::StandardNormal.sample(&mut rng)
            })?
        };
        // stoch_simul: y0 = M_.endo_histval with a histval block
        let y0 = crate::analysis::stochastic_histval_of(m, &mf.histval).unwrap_or_else(|| ys.clone());
        let y = simult(&dr, &y0, &ex, so.pruning);
        out.endo_simul = Some(y.columns(1, so.periods).into_owned());
        // ---- stoch_simul: simul_replic > 1 (simult.m saves every replication) ----
        if a.simul_replic > 1 {
            if let Some(ctx) = &so.tex_ctx {
                let mut sims = vec![out.endo_simul.clone().unwrap()];
                for _ in 1..a.simul_replic {
                    let e = if mf.q3.skew_e.is_empty() {
                        draw_shocks(sigma_e, so.periods, 0, 1, &mut rng).remove(0)
                    } else {
                        use rand_distr::Distribution;
                        crate::analysis::skew::draw_skewed_shocks(sigma_e, &mf.q3.skew_e, so.periods, &mut || {
                            rand_distr::StandardNormal.sample(&mut rng)
                        })?
                    };
                    sims.push(simult(&dr, &y0, &e, so.pruning).columns(1, so.periods).into_owned());
                }
                crate::analysis::write_simul_replic(ctx, &sims)?;
            }
        }
        out.exo_simul = Some(ex.clone());
        exo_simul = Some(ex);
    }

    // moments
    if !so.nomoments {
        if let (Some(y), Some(ex)) = (&out.endo_simul, &exo_simul) {
            // disp_moments.m (simult_ at the solution's order for the one-shock-at-a-time decomposition)
            let filter = crate::analysis::q2::Filter::from_options(&a)?;
            let sm = crate::analysis::simulate::simulated_moments_q2(
                sigma_e,
                y,
                ex,
                &var_list,
                so.drop,
                so.ar,
                &filter,
                a.nodecomposition,
                &|e| {
                    simult(
                        &dr,
                        &crate::analysis::stochastic_histval_of(m, &mf.histval).unwrap_or_else(|| ys.clone()),
                        e,
                        so.pruning,
                    )
                }, // disp_moments: y0 = endo_histval
            )?;
            let me = crate::analysis::simulated_vardec_me(&a, y, &var_list, &sm, m.n_exo(), seed)?; // stoch_simul
            print!(
                "{}",
                crate::analysis::simulated_moments_report_me(
                    m,
                    &sm,
                    &names,
                    &a,
                    so.order,
                    tex.as_ref(),
                    &var_list,
                    me.as_ref()
                )?
            );
            if a.contemporaneous_correlation && !a.nocorr {
                out.contemporaneous_correlation = Some(sm.corr.clone());
            }
            out.mean = sm.mean.clone();
            out.var = sm.var.clone();
            out.autocorr = sm.autocorr.clone();
            out.skewness = sm.skewness.clone();
            out.kurtosis = sm.kurtosis.clone();
            out.variance_decomposition = sm.vardec.clone();
            out.variance_decomposition_me = me.as_ref().map(|(_, v)| v.clone());
        } else if so.pruning && so.order <= 3 {
            pruned_moments(m, &dr, &sol1, &var_list, &names, &a, so.order, tex.as_ref(), &mut out)?;
        } else if so.order == 2 {
            // disp_th_moments.m at order 2: first-order second moments, second-order mean
            let mean_all = second_order_mean(&dr, sigma_e)?;
            let corr: Vec<f64> = var_list.iter().map(|&j| mean_all[j] - ys[j]).collect();
            let mut a2 = a.clone();
            a2.order = 2;
            let r = crate::analysis::theoretical_moments_q2(m, &sol1, &var_list, &a2, 2, Some(&corr), tex.as_ref())?;
            print!("{}", r.text);
            out.mean = r.mean;
            if let Some(th) = &r.th {
                out.var = th.gamma0.clone();
                out.autocorr = th.autocorr.clone();
                out.variance_decomposition = th.vardec.as_ref().map(|v| v * 100.0);
            }
            out.conditional_variance_decomposition = r.conditional_vardec;
            out.variance_decomposition_me = r.variance_decomposition_me;
            out.conditional_variance_decomposition_me = r.conditional_vardec_me;
            out.contemporaneous_correlation = r.contemporaneous_correlation;
        }
        // order 3 without pruning, order >= 4: no theoretical moments in Dynare either
    }

    // irf.m (generalized IRFs)
    if so.irf > 0 {
        let cs = lower_cholesky(sigma_e);
        // stoch_simul: getIrfShocksIndx.m (unique, sorted)
        let shocks = crate::analysis::irf::irf_shock_indices(
            &m.exo_names,
            (!so.irf_shocks.is_empty()).then_some(&so.irf_shocks[..]),
        )?;
        for &i in &shocks {
            if cs[(i, i)] <= 5e-7 {
                continue;
            }
            let draws = draw_shocks(sigma_e, so.irf, so.drop, so.replic, &mut rng);
            // relative_irf at order > 1: the shock is normalised to 0.01 (stoch_simul.m)
            let e1: Vec<f64> =
                cs.column(i).iter().map(|&v| if a.relative_irf { v / cs[(i, i)] / 100.0 } else { v }).collect();
            let y = girf(&dr, &e1, so.irf, so.drop, &draws, so.pruning);
            if !so.noprint && y.iter().any(|v| v.is_nan()) && !so.pruning {
                println!(
                    "\nstoch_simul:: The simulations conducted for generating IRFs to {} were explosive.",
                    m.exo_names[i]
                );
                println!("stoch_simul:: No IRFs will be displayed. Either reduce the shock size, ");
                print!("stoch_simul:: use pruning, or set the approximation order to 1.");
                println!("\n");
            }
            for &j in &var_list {
                out.irfs.push((format!("{}_{}", m.endo_names[j], m.exo_names[i]), y.row(j).iter().copied().collect()));
            }
        }
    }

    Ok(out)
}

/// Driver entry: runs [`stoch_simul`], writes `<dir>/<fname>/Output/<fname>_stoch_simul.json`
/// and the IRF figures (`<fname>_IRF_<shock>`).
pub fn run(mf: &ModFile, o: &Options, var_names: &[String], dir: &Path, fname: &str, seed: u64) -> Result<()> {
    let so = StochSimulOptions::from_options(o);
    run_at(mf, &so, var_names, dir, fname, seed, None).map(|_| ())
}

/// the driver's entry with the persistent analysis options (Dynare's `options_`).
pub fn run_q2(
    mf: &ModFile,
    a: &crate::analysis::StochSimulOptions,
    var_names: &[String],
    ctx: &crate::analysis::OutputContext,
) -> Result<Option<Mat>> {
    let mut a = a.clone(); // stoch_simul: M_.H
    a.me = crate::analysis::MeasurementErrors::from_modfile(mf);
    let mut so = StochSimulOptions::from_analysis(&a);
    so.tex_ctx = Some(ctx.clone());
    // oo_.endo_simul, for datatomfile
    run_at(mf, &so, var_names, &ctx.dir, &ctx.fname, ctx.seed, None).map(|r| r.endo_simul)
}

/// [`run`] with explicit options and, optionally, a steady state and parameters (see
/// [`stoch_simul_at`]); returns the results (Ramsey `stoch_simul(order >= 2)`).
pub fn run_at(
    mf: &ModFile,
    so: &StochSimulOptions,
    var_names: &[String],
    dir: &Path,
    fname: &str,
    seed: u64,
    at: Option<(&[f64], &[f64])>,
) -> Result<StochSimulOutput> {
    let so = so.clone();
    let res = stoch_simul_at(mf, &so, var_names, seed, at)?;
    let m = &mf.model;
    let dr = &res.dr;
    let out_dir = dir.join(fname).join("Output");
    std::fs::create_dir_all(&out_dir)?;
    let mj = |a: &Mat| -> serde_json::Value {
        (0..a.nrows()).map(|i| a.row(i).iter().copied().collect::<Vec<f64>>()).collect()
    };
    let mut drj = json!({
        "order": dr.order,
        "ys": dr.first.ys,
        "order_var": dr.first.order_var.iter().map(|v| v + 1).collect::<Vec<_>>(),
        "state_var": dr.first.state_var.iter().map(|v| v + 1).collect::<Vec<_>>(),
        "ghx": mj(&dr.first.ghx), "ghu": mj(&dr.first.ghu),
        "ghxx": mj(&dr.ghxx), "ghxu": mj(&dr.ghxu), "ghuu": mj(&dr.ghuu),
        "ghs2": dr.ghs2.as_slice(),
    });
    for j in 0..=dr.order {
        drj[format!("g_{j}")] = mj(&dr.g_k(j));
    }
    if dr.order >= 3 {
        for (k, v) in [
            ("ghxxx", &dr.ghxxx),
            ("ghxxu", &dr.ghxxu),
            ("ghxuu", &dr.ghxuu),
            ("ghuuu", &dr.ghuuu),
            ("ghxss", &dr.ghxss),
            ("ghuss", &dr.ghuss),
        ] {
            drj[k] = mj(v);
        }
    }
    let irfs: serde_json::Map<String, serde_json::Value> =
        res.irfs.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
    let mut j = json!({
        "dr": drj,
        "var_list": res.var_list.iter().map(|&i| m.endo_names[i].clone()).collect::<Vec<_>>(),
        "irfs": irfs,
        "mean": res.mean,
        "var": mj(&res.var),
        "autocorr": res.autocorr.iter().map(&mj).collect::<Vec<_>>(),
    });
    if let Some(y) = &res.endo_simul {
        j["endo_simul"] = mj(y);
    }
    let path = out_dir.join(format!("{fname}_stoch_simul.json"));
    std::fs::write(&path, serde_json::to_string(&j)?)?;

    // IRF figures, as stoch_simul.m (irf_plot_threshold, relative_irf titles,
    // irf_shocks order)
    let a = so.analysis();
    let gopts =
        GraphOptions { enabled: !so.nograph, formats: so.graph_formats.clone(), dir: graphics::graph_dir(dir, fname) };
    if gopts.enabled && so.irf > 0 {
        let names: Vec<String> = res.var_list.iter().map(|&v| m.endo_names[v].clone()).collect();
        for shock in &m.exo_names {
            let rows: Vec<Vec<f64>> = res
                .var_list
                .iter()
                .filter_map(|&v| {
                    let key = format!("{}_{}", m.endo_names[v], shock);
                    res.irfs.iter().find(|(k, _)| *k == key).map(|(_, y)| y.clone())
                })
                .collect();
            if rows.len() != names.len() {
                continue;
            }
            let y = Mat::from_fn(rows.len(), so.irf, |i, t| rows[i][t]);
            for fig in crate::analysis::irf::irf_figures(shock, &names, &y, a.relative_irf, a.irf_plot_threshold) {
                graphics::save(&fig, fname, &gopts)?;
            }
        }
    }
    // the decompositions in the results file
    {
        let path = out_dir.join(format!("{fname}_stoch_simul.json"));
        let mut j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        if !res.skewness.is_empty() {
            j["skewness"] = json!(res.skewness);
            j["kurtosis"] = json!(res.kurtosis);
        }
        if let Some(v) = &res.variance_decomposition {
            j["variance_decomposition"] = mj(v);
        }
        if let Some(c) = &res.conditional_variance_decomposition {
            j["conditional_variance_decomposition"] = c.iter().map(|m| mj(&(m * 100.0))).collect::<Vec<_>>().into();
        }
        if let Some(c) = &res.contemporaneous_correlation {
            j["contemporaneous_correlation"] = mj(c);
        }
        if let Some(v) = &res.variance_decomposition_me {
            j["variance_decomposition_ME"] = mj(v);
        }
        if let Some(c) = &res.conditional_variance_decomposition_me {
            j["conditional_variance_decomposition_ME"] = c.iter().map(|m| mj(&(m * 100.0))).collect::<Vec<_>>().into();
        }
        std::fs::write(&path, serde_json::to_string(&j)?)?;
    }
    Ok(res)
}

/// `get_lower_cholesky_covariance`: lower Cholesky factor on the shocks with non-zero variance.
fn lower_cholesky(s: &Mat) -> Mat {
    let n = s.nrows();
    let idx: Vec<usize> = (0..n).filter(|&i| s[(i, i)] != 0.0).collect();
    let sub = Mat::from_fn(idx.len(), idx.len(), |i, j| s[(idx[i], idx[j])]);
    let mut out = Mat::zeros(n, n);
    if let Some(c) = sub.cholesky() {
        let l = c.l();
        for (a, &i) in idx.iter().enumerate() {
            for (b, &j) in idx.iter().enumerate() {
                out[(i, j)] = l[(a, b)];
            }
        }
    }
    out
}

/// `disp_dr.m` at order 2.
fn print_dr(mf: &ModFile, dr: &KOrderDr, var_list: &[usize], tol: f64) {
    let m = &mf.model;
    let f = &dr.first;
    let ivar: Vec<usize> = var_list.iter().map(|&j| f.inv_order_var[j]).collect();
    let hl = var_list.iter().map(|&j| m.endo_names[j].len() + 2).max().unwrap_or(0).max(16);
    let aux_extra = if m.aux_vars.is_empty() { 0 } else { 3 };
    let base =
        var_list.iter().map(|&j| m.endo_names[j].len()).chain(m.exo_names.iter().map(|s| s.len())).max().unwrap_or(0);
    let vw = (2 * (base + aux_extra) + 2).max(20);
    let state_name = |k: usize| -> String {
        let j = f.state_var[k];
        match m.aux_vars.iter().find(|a| a.endo_index == j && a.ty == 1) {
            Some(a) => match (a.orig_index, a.orig_lead_lag) {
                (Some(o), Some(l)) => format!("{}({})", m.endo_names[o], l - 1),
                _ => format!("{}(-1)", m.endo_names[j]),
            },
            None => format!("{}(-1)", m.endo_names[j]),
        }
    };
    let fmt = |x: f64| -> String {
        if x.abs() >= tol {
            let s = format!("{x:>hl$.6}");
            if s.len() > hl { format!("{x:>hl$.6e}") } else { s }
        } else {
            format!("{:>hl$}", 0)
        }
    };
    let row = |label: &str, vals: &dyn Fn(usize) -> f64| {
        let xs: Vec<f64> = ivar.iter().map(|&i| vals(i)).collect();
        if xs.iter().any(|x| x.abs() >= tol) {
            let mut s = format!("{label:<vw$}");
            for x in xs {
                s.push_str(&fmt(x));
            }
            println!("{s}");
        }
    };
    println!("POLICY AND TRANSITION FUNCTIONS"); // stoch_simul.m: skipline() then disp_dr
    let mut h = " ".repeat(vw);
    for &j in var_list {
        h.push_str(&format!("{:>hl$}", m.endo_names[j]));
    }
    println!("{h}");
    row("Constant", &|i| f.ys[f.order_var[i]] + dr.ghs2[(i, 0)] / 2.0);
    row("(correction)", &|i| dr.ghs2[(i, 0)] / 2.0);
    let (nx, nu) = (f.nspred, dr.n_exo);
    for k in 0..nx {
        row(&state_name(k), &|i| f.ghx[(i, k)]);
    }
    for k in 0..nu {
        row(&m.exo_names[k], &|i| f.ghu[(i, k)]);
    }
    for k in 0..nx {
        for j in 0..=k {
            let lab = format!("{},{}", state_name(k), state_name(j));
            row(&lab, &|i| {
                let v = dr.ghxx[(i, k * nx + j)];
                if k == j { v / 2.0 } else { v }
            });
        }
    }
    for k in 0..nu {
        for j in 0..=k {
            let lab = format!("{},{}", m.exo_names[k], m.exo_names[j]);
            row(&lab, &|i| {
                let v = dr.ghuu[(i, k * nu + j)];
                if k == j { v / 2.0 } else { v }
            });
        }
    }
    for k in 0..nx {
        for j in 0..nu {
            let lab = format!("{},{}", state_name(k), m.exo_names[j]);
            row(&lab, &|i| dr.ghxu[(i, k * nu + j)]);
        }
    }
}
