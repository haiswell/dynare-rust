// SPDX-License-Identifier: GPL-3.0-or-later
//! Bayesian IRFs and posterior moments of the endogenous variables from the MCMC draws:
//! ports of `estimation/PosteriorIRF.m` (+ `PosteriorIRF_core1/2.m`), `get_posterior_subsample.m`,
//! `set_number_of_subdraws.m` and the `moments_varendo` path (`compute_moments_varendo.m`,
//! `posterior_analysis.m`, `dsge_simulated_theoretical_{covariance,variance_decomposition,
//! conditional_variance_decomposition}.m`, `*_mc_analysis.m`).
//!
//! Each sub-draw is re-solved with [`DsgeModel::resolve`] (steady state + first order, as
//! `compute_decision_rules`), in parallel on the rayon pool. The sub-sample is drawn with
//! ChaCha8 (seed `--seed`, stream [`SUBSAMPLE_STREAM`]) instead of MATLAB's `rand`.

use std::path::Path;

use anyhow::{Result, anyhow, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::{StochSimulOptions, irf, moments, table::dyntable, table::max_len, vardec};
use crate::dsge::DsgeModel;
use crate::estimation::mcmc::Chain;
use crate::estimation::posterior::{ParamMoments, posterior_moments};
use crate::graphics::{Color, Figure, LineStyle, Panel, Series};
use crate::linalg::Mat;
use crate::solver::describe;

pub const SUBSAMPLE_STREAM: u64 = 0x5355_4244_5241_5753;

/// `set_number_of_subdraws.m`: `sub_draws`, or min(posterior_max_subsample_draws = 1200,
/// number of kept draws over all chains).
pub fn number_of_subdraws(chains: &[Chain], sub_draws: Option<usize>) -> Result<usize> {
    let total: usize = chains.iter().map(Chain::len).sum();
    match sub_draws {
        None => Ok(crate::estimation::mh_session::posterior_max_subsample_draws().min(total)),
        Some(b) if b > total => bail!(
            "The value of option sub_draws ({b}) is greater than the number of available draws in the MCMC ({total})!"
        ),
        Some(b) => Ok(b),
    }
}

/// `get_posterior_subsample.m`: `b` draws with replacement, chain `ceil(rand*nblck)`, draw
/// `ceil(rand*NumberOfDraws)` among the kept draws of that chain.
pub fn posterior_subsample(chains: &[Chain], b: usize, seed: u64) -> Result<Vec<Vec<f64>>> {
    if chains.is_empty() || chains.iter().any(Chain::is_empty) {
        bail!("posterior subsample: no MCMC draws kept");
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(SUBSAMPLE_STREAM);
    let ndraws = chains.iter().map(Chain::len).min().unwrap();
    Ok((0..b)
        .map(|_| {
            let c = &chains[rng.gen_range(0..chains.len())];
            let k = rng.gen_range(0..ndraws);
            c.draw(c.first_stored + k).to_vec()
        })
        .collect())
}

/// `posterior_moments` of one statistic over the draws; all NaN if any draw is NaN
/// (nonstationary variable).
pub fn moments_of(mut v: Vec<f64>, mh_conf_sig: f64) -> ParamMoments {
    if v.iter().any(|x| x.is_nan()) {
        return ParamMoments {
            mean: f64::NAN,
            median: f64::NAN,
            var: f64::NAN,
            hpd: (f64::NAN, f64::NAN),
            deciles: [f64::NAN; 9],
        };
    }
    posterior_moments(&mut v, mh_conf_sig)
}

/// Posterior IRF statistics (`oo_.PosteriorIRF.dsge`).
pub struct PosteriorIrf {
    pub ivar: Vec<usize>,
    /// Shocks with IRFs (`irf_shocks`).
    pub shocks: Vec<usize>,
    /// `stats[s][var][period]` for `shocks[s]`.
    pub stats: Vec<Vec<Vec<ParamMoments>>>,
}

/// `PosteriorIRF_core1.m` + the statistics of `PosteriorIRF.m`.
pub fn posterior_irf(
    dsge: &DsgeModel,
    draws: &[Vec<f64>],
    ivar: &[usize],
    o: &StochSimulOptions,
) -> Result<PosteriorIrf> {
    let periods = if o.irf == 0 { 40 } else { o.irf };
    let shocks = irf::irf_shock_indices(&dsge.model.exo_names, o.irf_shocks.as_deref())?;
    // per draw: [shock][var][period]
    let per_draw: Vec<Vec<Mat>> = draws
        .par_iter()
        .map(|x| {
            let r = dsge.resolve(x).map_err(|e| {
                anyhow!(
                    "PosteriorIRF :: Dynare is unable to solve the model ({}) with sample posterior",
                    describe(e.info)
                )
            })?;
            let cs = vardec::lower_cholesky(&r.q, o.add_tiny_number_to_cholesky);
            Ok(shocks
                .iter()
                .map(|&i| {
                    let mut y = Mat::zeros(ivar.len(), periods);
                    if cs[(i, i)] > 5e-7 {
                        let e: Vec<f64> = cs.column(i).iter().copied().collect();
                        let mut full = irf::irf(&r.dr, &e, periods);
                        if o.relative_irf {
                            full *= 100.0 / cs[(i, i)];
                        }
                        for (j, &v) in ivar.iter().enumerate() {
                            let row = full.row(v);
                            let (mn, mx) =
                                row.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
                            if mx - mn > 1e-12 {
                                y.row_mut(j).copy_from(&row);
                            }
                        }
                    }
                    y
                })
                .collect())
        })
        .collect::<Result<_>>()?;
    let stats = (0..shocks.len())
        .map(|s| {
            (0..ivar.len())
                .map(|j| {
                    (0..periods)
                        .map(|t| moments_of(per_draw.iter().map(|d| d[s][(j, t)]).collect(), o.mh_conf_sig))
                        .collect()
                })
                .collect()
        })
        .collect();
    Ok(PosteriorIrf { ivar: ivar.to_vec(), shocks, stats })
}

/// A decomposition with measurement error and the variables it covers
/// (`check_measurement_error_requested_vars.m`).
pub struct MeDecomposition<T> {
    /// Declaration indices of the observed requested variables (`observable_pos_requested_vars`).
    pub vars: Vec<usize>,
    pub stats: T,
}

/// Posterior distribution of the theoretical moments (`oo_.PosteriorTheoreticalMoments.dsge`).
pub struct PosteriorTheoreticalMoments {
    pub ivar: Vec<usize>,
    /// `covariance[i][j]`, j >= i (lower part empty).
    pub covariance: Vec<Vec<Option<ParamMoments>>>,
    /// `correlation[lag-1][i][j]`: corr(y_i,t, y_j,t-lag).
    pub correlation: Vec<Vec<Vec<ParamMoments>>>,
    /// `variance_decomposition[i][shock]` (fractions), when computed.
    pub variance_decomposition: Option<Vec<Vec<ParamMoments>>>,
    /// `conditional[step][i][shock]` (fractions).
    pub conditional: Option<Vec<Vec<Vec<ParamMoments>>>>,
    /// `VarianceDecompositionME` (dsge_simulated_theoretical_variance_decomposition.m): for
    /// the requested observed variables, `[obs][shock or ME]` (fractions, last column ME).
    pub variance_decomposition_me: Option<MeDecomposition<Vec<Vec<ParamMoments>>>>,
    /// `ConditionalVarianceDecompositionME`: `[step][obs][shock or ME]`.
    pub conditional_me: Option<MeDecomposition<Vec<Vec<Vec<ParamMoments>>>>>,
    // ---- estimation completion ----
    /// `contemporaneous_correlation`: `contemporeaneous_correlation[i][j]`, j >= i
    /// (covariance_mc_analysis.m: cov / (sd_i sd_j), 0 where the covariance is 0).
    pub contemporaneous_correlation: Option<Vec<Vec<Option<ParamMoments>>>>,
    /// The `density` fields (posterior_moments.m, 512 x 2) unless `no_posterior_kernel_density`:
    /// (statistic, variable1, variable2 or shock, index, density).
    pub densities: Vec<(&'static str, usize, usize, usize, Mat)>,
    // ---- end estimation completion ----
}

/// `compute_moments_varendo('posterior', ...)` at order 1.
pub fn posterior_moments_varendo(
    dsge: &DsgeModel,
    draws: &[Vec<f64>],
    ivar: &[usize],
    o: &StochSimulOptions,
) -> Result<PosteriorTheoreticalMoments> {
    let me_ids: Option<Vec<usize>> = Some(dsge.varobs.clone());
    let nexo = dsge.model.n_exo();
    let mut mo = o.moment_options();
    mo.nodecomposition = true;
    let do_vd = nexo > 1 && !o.nodecomposition;
    let steps = o.conditional_variance_decomposition.clone();
    // compute_moments_varendo.m: `if Steps` sits outside `if ~NoDecomposition`
    let do_cvd = nexo > 1 && !steps.is_empty();
    struct One {
        g: moments::TheoreticalMoments,
        vd: Option<Mat>,
        cvd: Option<Vec<Mat>>,
        /// diag(M_.H) at the draw (set_measurement_errors).
        h: Vec<f64>,
        /// Conditional variances of the requested variables (nvar x nexo per step).
        cv: Option<Vec<Mat>>,
    }
    let per: Vec<One> = draws
        .par_iter()
        .map(|x| {
            let r = dsge
                .resolve(x)
                .map_err(|e| anyhow!("moments_varendo: no solution at a posterior draw ({})", describe(e.info)))?;
            let g = moments::th_autocovariances(&r.dr, &r.q, ivar, &mo)?;
            let vd = if do_vd {
                let mut m2 = mo.clone();
                m2.nodecomposition = false;
                m2.ar = 0;
                moments::th_autocovariances(&r.dr, &r.q, ivar, &m2)?.vardec
            } else {
                None
            };
            let cvd = if do_cvd {
                Some(vardec::conditional_variance_decomposition(&r.dr, &r.q, &steps, ivar)?)
            } else {
                None
            };
            let cv = if do_cvd { Some(vardec::conditional_variances(&r.dr, &r.q, &steps, ivar)?) } else { None };
            Ok(One { g, vd, cvd, h: r.h.diagonal().iter().copied().collect(), cv })
        })
        .collect::<Result<_>>()?;
    let n = ivar.len();
    let sig = o.mh_conf_sig;
    let covariance = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (j >= i).then(|| moments_of(per.iter().map(|p| p.g.gamma0[(i, j)]).collect(), sig)))
                .collect()
        })
        .collect();
    let correlation = (0..o.ar)
        .map(|k| {
            (0..n)
                .map(|i| {
                    (0..n).map(|j| moments_of(per.iter().map(|p| p.g.autocorr[k][(i, j)]).collect(), sig)).collect()
                })
                .collect()
        })
        .collect();
    let variance_decomposition = do_vd.then(|| {
        (0..n)
            .map(|i| {
                (0..nexo)
                    .map(|s| {
                        moments_of(per.iter().map(|p| p.vd.as_ref().map_or(f64::NAN, |m| m[(i, s)])).collect(), sig)
                    })
                    .collect()
            })
            .collect()
    });
    let conditional = do_cvd.then(|| {
        (0..steps.len())
            .map(|h| {
                (0..n)
                    .map(|i| {
                        (0..nexo)
                            .map(|s| moments_of(per.iter().map(|p| p.cvd.as_ref().unwrap()[h][(i, s)]).collect(), sig))
                            .collect()
                    })
                    .collect()
            })
            .collect()
    });
    // ---- with measurement error: dsge_simulated_theoretical_(conditional_)variance_decomposition.m
    // (M_.H of each draw; the observed requested variables, `intersect(ivar, varobs_id, 'stable')`)
    let (me_vars, index_subset, index_obs) = match &me_ids {
        Some(ids) if per.iter().any(|p| p.h.iter().any(|&x| x != 0.0)) => {
            let me = super::MeasurementErrors { h: Mat::zeros(0, 0), varobs: vec![], varobs_id: ids.clone() };
            me.requested(ivar)
        }
        _ => (vec![], vec![], vec![]),
    };
    let me_present = !me_vars.is_empty();
    let variance_decomposition_me = (do_vd && me_present).then(|| MeDecomposition {
        vars: me_vars.clone(),
        stats: (0..me_vars.len())
            .map(|a| {
                let (k, ob) = (index_subset[a], index_obs[a]);
                let draw_me = |p: &One, j: usize| -> f64 {
                    let v = p.g.gamma0[(k, k)];
                    let vdm = p.vd.as_ref().unwrap();
                    let part = |s: usize| vdm[(k, s)] * (v / (v + p.h[ob]));
                    if j < nexo { part(j) } else { 1.0 - (0..nexo).map(part).sum::<f64>() }
                };
                (0..=nexo).map(|j| moments_of(per.iter().map(|p| draw_me(p, j)).collect(), sig)).collect()
            })
            .collect(),
    });
    let conditional_me = (do_cvd && me_present).then(|| MeDecomposition {
        vars: me_vars.clone(),
        stats: (0..steps.len())
            .map(|h| {
                (0..me_vars.len())
                    .map(|a| {
                        let (k, ob) = (index_subset[a], index_obs[a]);
                        let draw_me = |p: &One, j: usize| -> f64 {
                            let c = &p.cv.as_ref().unwrap()[h];
                            let tot: f64 = (0..nexo).map(|i| c[(k, i)]).sum::<f64>() + p.h[ob];
                            if j < nexo {
                                c[(k, j)] / tot
                            } else {
                                1.0 - (0..nexo).map(|i| c[(k, i)] / tot).sum::<f64>()
                            }
                        };
                        (0..=nexo).map(|j| moments_of(per.iter().map(|p| draw_me(p, j)).collect(), sig)).collect()
                    })
                    .collect()
            })
            .collect(),
    });
    // ---- estimation completion: contemporaneous_correlation, kernel densities ----
    let corr_of = |p: &One, i: usize, j: usize| {
        let g = &p.g.gamma0;
        if g[(i, j)] == 0.0 { 0.0 } else { g[(i, j)] / (g[(i, i)].sqrt() * g[(j, j)].sqrt()) }
    };
    let contemporaneous_correlation = o.contemporaneous_correlation.then(|| {
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| (j >= i).then(|| moments_of(per.iter().map(|p| corr_of(p, i, j)).collect(), sig)))
                    .collect()
            })
            .collect()
    });
    let mut densities: Vec<(&'static str, usize, usize, usize, Mat)> = Vec::new();
    if o.posterior_kernel_density {
        use crate::estimation::est_completion::posterior_density;
        for i in 0..n {
            for j in i..n {
                densities.push((
                    "covariance",
                    i,
                    j,
                    0,
                    posterior_density(&per.iter().map(|p| p.g.gamma0[(i, j)]).collect::<Vec<_>>()),
                ));
                if o.contemporaneous_correlation {
                    densities.push((
                        "contemporaneous_correlation",
                        i,
                        j,
                        0,
                        posterior_density(&per.iter().map(|p| corr_of(p, i, j)).collect::<Vec<_>>()),
                    ));
                }
            }
        }
        for k in 0..o.ar {
            for i in 0..n {
                for j in 0..n {
                    densities.push((
                        "correlation",
                        i,
                        j,
                        k + 1,
                        posterior_density(&per.iter().map(|p| p.g.autocorr[k][(i, j)]).collect::<Vec<_>>()),
                    ));
                }
            }
        }
        if do_vd {
            for i in 0..n {
                for s in 0..nexo {
                    let v: Vec<f64> = per.iter().map(|p| p.vd.as_ref().map_or(f64::NAN, |m| m[(i, s)])).collect();
                    densities.push(("variance_decomposition", i, s, 0, posterior_density(&v)));
                }
            }
        }
        if do_cvd {
            for (h, &st) in steps.iter().enumerate() {
                for i in 0..n {
                    for s in 0..nexo {
                        let v: Vec<f64> = per.iter().map(|p| p.cvd.as_ref().unwrap()[h][(i, s)]).collect();
                        densities.push(("conditional_variance_decomposition", i, s, st, posterior_density(&v)));
                    }
                }
            }
        }
    }
    Ok(PosteriorTheoreticalMoments {
        ivar: ivar.to_vec(),
        covariance,
        correlation,
        variance_decomposition,
        conditional,
        variance_decomposition_me,
        conditional_me,
        contemporaneous_correlation,
        densities,
    })
    // ---- end estimation completion ----
}

fn stat_row(m: &ParamMoments) -> String {
    let mut s = format!("{},{},{},{},{}", m.mean, m.median, m.var, m.hpd.0, m.hpd.1);
    for d in m.deciles {
        s += &format!(",{d}");
    }
    s
}

const STAT_HEADER: &str = "mean,median,var,hpd_inf,hpd_sup,d10,d20,d30,d40,d50,d60,d70,d80,d90";

/// The estimation hook: `bayesian_irf` and/or `moments_varendo` after an MCMC. Prints the
/// tables Dynare prints, writes `<fname>_Bayesian_IRF_<shock>.csv` and
/// `<fname>_posterior_moments.csv` to `out_dir`, and the `Bayesian_IRF_<shock>_<k>` figures.
#[allow(clippy::too_many_arguments)]
pub fn run(
    dsge: &DsgeModel,
    chains: &[Chain],
    o: &StochSimulOptions,
    var_list: &[String],
    dir: &Path,
    fname: &str,
    out_dir: &Path,
    seed: u64,
    bvar_dsge_irfs: bool,
) -> Result<()> {
    if !o.bayesian_irf && !o.moments_varendo {
        return Ok(());
    }
    let model = &dsge.model;
    // varlist = options_.varlist, or varobs when empty
    let ivar: Vec<usize> = if var_list.is_empty() { dsge.varobs.clone() } else { super::var_indices(model, var_list)? };
    let names: Vec<String> = ivar.iter().map(|&i| model.endo_names[i].clone()).collect();
    let b = number_of_subdraws(chains, o.sub_draws)?;
    if b == 0 {
        println!("Estimation: sub_draws was set to 0. Skipping posterior computations.");
        return Ok(());
    }
    if ((1.0 - o.mh_conf_sig) * b as f64).round() < 2.0 {
        println!(
            "\nPosteriorIRF:: options_.mh_conf_sig times options_.sub_draws is too small to generate HPDIs. I am omitting them."
        );
    }
    let draws = posterior_subsample(chains, b, seed)?;
    std::fs::create_dir_all(out_dir)?;
    if o.bayesian_irf {
        let t = std::time::Instant::now();
        println!("Estimation::mcmc: Posterior (dsge) IRFs...");
        if bvar_dsge_irfs {
            // PosteriorIRF.m prints this before the DSGE-VAR IRFs, which are computed afterwards here
            println!("Estimation::mcmc: Posterior (bvar-dsge) IRFs...");
        }
        let pirf = posterior_irf(dsge, &draws, &ivar, o)?;
        let mut gopts = o.graph_options(dir, fname);
        gopts.dir = out_dir.to_path_buf(); // PosteriorIRF_core2 saves to Output/
        for (s, &shock) in pirf.shocks.iter().enumerate() {
            let sname = &model.exo_names[shock];
            let mut w = format!("variable,period,{STAT_HEADER}\n");
            for (j, n) in names.iter().enumerate() {
                for (t, m) in pirf.stats[s][j].iter().enumerate() {
                    w += &format!("{n},{},{}\n", t + 1, stat_row(m));
                }
            }
            std::fs::write(out_dir.join(format!("{fname}_Bayesian_IRF_{sname}.csv")), w)?;
            if !o.nograph {
                for fig in bayesian_irf_figures(sname, &names, &pirf.stats[s], o.relative_irf, o.irf_plot_threshold) {
                    crate::graphics::save(&fig, fname, &gopts)?;
                }
            }
        }
        println!("Estimation::mcmc: Posterior IRFs, done!");
        crate::progress::note(format!(
            "Posterior IRFs: {} draws, {:.1}s; {}_Bayesian_IRF_<shock>.csv in {}",
            b,
            t.elapsed().as_secs_f64(),
            fname,
            out_dir.display()
        ));
    }
    if o.moments_varendo {
        let t = std::time::Instant::now();
        println!("Estimation::compute_moments_varendo: I'm computing endogenous moments (this may take a while)... ");
        let pm = posterior_moments_varendo(dsge, &draws, &ivar, o)?;
        let mut w = format!("statistic,variable1,variable2,index,{STAT_HEADER}\n");
        for i in 0..names.len() {
            for j in i..names.len() {
                if let Some(m) = &pm.covariance[i][j] {
                    w += &format!("covariance,{},{},0,{}\n", names[i], names[j], stat_row(m));
                }
            }
        }
        // ---- estimation completion: contemporaneous_correlation
        if let Some(cc) = &pm.contemporaneous_correlation {
            for i in 0..names.len() {
                for j in i..names.len() {
                    if let Some(m) = &cc[i][j] {
                        w += &format!("contemporaneous_correlation,{},{},0,{}\n", names[i], names[j], stat_row(m));
                    }
                }
            }
        }
        // ---- end estimation completion
        for (k, c) in pm.correlation.iter().enumerate() {
            for i in 0..names.len() {
                for j in 0..names.len() {
                    w += &format!("correlation,{},{},{},{}\n", names[i], names[j], k + 1, stat_row(&c[i][j]));
                }
            }
        }
        let mut headers = vec![" ".to_string()];
        headers.extend(model.exo_names.iter().cloned());
        let lh = max_len(&names) + 2;
        if let Some(vd) = &pm.variance_decomposition {
            for (i, n) in names.iter().enumerate() {
                for (s, e) in model.exo_names.iter().enumerate() {
                    w += &format!("variance_decomposition,{n},{e},0,{}\n", stat_row(&vd[i][s]));
                }
            }
            let m = Mat::from_fn(names.len(), model.n_exo(), |i, s| 100.0 * vd[i][s].mean);
            if !o.noprint {
                print!(
                    "{}",
                    dyntable(
                        &super::filter_subtitle("Posterior mean variance decomposition (in percent)", o),
                        &headers,
                        &names,
                        &m,
                        lh,
                        8,
                        2
                    )
                );
                println!("\n");
            }
        }
        // ---- with measurement error (compute_moments_varendo.m)
        let me_headers: Vec<String> = headers.iter().cloned().chain(std::iter::once("ME".to_string())).collect();
        let me_names = |vars: &[usize]| vars.iter().map(|&v| model.endo_names[v].clone()).collect::<Vec<_>>();
        if let (Some(me), true) = (&pm.variance_decomposition_me, pm.variance_decomposition.is_some()) {
            let nm = me_names(&me.vars);
            for (a, n) in nm.iter().enumerate() {
                for (s, e) in model.exo_names.iter().map(String::as_str).chain(["ME"]).enumerate() {
                    w += &format!("variance_decomposition_ME,{n},{e},0,{}\n", stat_row(&me.stats[a][s]));
                }
            }
            if !o.noprint {
                let m = Mat::from_fn(nm.len(), model.n_exo() + 1, |a, s| 100.0 * me.stats[a][s].mean);
                let title = super::filter_subtitle(
                    "Posterior mean variance decomposition (in percent) with measurement error",
                    o,
                );
                print!("{}", dyntable(&title, &me_headers, &nm, &m, max_len(&names) + 2, 8, 2));
                println!();
            }
        }
        if let Some(c) = &pm.conditional {
            let steps = &o.conditional_variance_decomposition;
            for (h, ch) in c.iter().enumerate() {
                for (i, n) in names.iter().enumerate() {
                    for (s, e) in model.exo_names.iter().enumerate() {
                        w += &format!(
                            "conditional_variance_decomposition_{},{n},{e},{},{}\n",
                            steps[h],
                            steps[h],
                            stat_row(&ch[i][s])
                        );
                    }
                }
                let m = Mat::from_fn(names.len(), model.n_exo(), |i, s| 100.0 * ch[i][s].mean);
                if !o.noprint {
                    let title =
                        format!("Posterior mean conditional variance decomposition (in percent) Period {}", steps[h]);
                    print!("{}", dyntable(&title, &headers, &names, &m, lh, 8, 2));
                }
            }
            if !o.noprint {
                println!();
            }
        }
        if let Some(me) = &pm.conditional_me {
            let steps = &o.conditional_variance_decomposition;
            let nm = me_names(&me.vars);
            for (h, ch) in me.stats.iter().enumerate() {
                for (a, n) in nm.iter().enumerate() {
                    for (s, e) in model.exo_names.iter().map(String::as_str).chain(["ME"]).enumerate() {
                        w += &format!(
                            "conditional_variance_decomposition_ME_{},{n},{e},{},{}\n",
                            steps[h],
                            steps[h],
                            stat_row(&ch[a][s])
                        );
                    }
                }
                if !o.noprint {
                    let m = Mat::from_fn(nm.len(), model.n_exo() + 1, |a, s| 100.0 * ch[a][s].mean);
                    let title = format!(
                        "Posterior mean conditional variance decomposition (in percent) with measurement error Period {}",
                        steps[h]
                    );
                    print!("{}", dyntable(&title, &me_headers, &nm, &m, max_len(&names) + 2, 8, 2));
                }
            }
            if !o.noprint {
                println!();
            }
        }
        std::fs::write(out_dir.join(format!("{fname}_posterior_moments.csv")), w)?;
        // ---- estimation completion: oo_.PosteriorTheoreticalMoments.dsge.*.density
        if !pm.densities.is_empty() {
            let mut d = String::from("statistic,variable1,variable2,index,abscissa,density\n");
            for (stat, i, j, k, m) in &pm.densities {
                let second = if stat.contains("variance_decomposition") { &model.exo_names[*j] } else { &names[*j] };
                for r in 0..m.nrows() {
                    d += &format!("{stat},{},{second},{k},{},{}\n", names[*i], m[(r, 0)], m[(r, 1)]);
                }
            }
            std::fs::write(out_dir.join(format!("{fname}_posterior_moments_density.csv")), d)?;
        }
        // ---- end estimation completion
        println!("Done!\n");
        crate::progress::note(format!(
            "Estimation::compute_moments_varendo: done ({:.1}s; {}_posterior_moments.csv).",
            t.elapsed().as_secs_f64(),
            fname
        ));
    }
    Ok(())
}

/// `PosteriorIRF_core2.m`: 3x3 panels per figure, HPD band (gray) and posterior mean (black),
/// variables whose mean IRF reaches the plot threshold; `Bayesian_IRF_<shock>_<k>`.
pub fn bayesian_irf_figures(
    shock: &str,
    names: &[String],
    stats: &[Vec<ParamMoments>],
    relative: bool,
    threshold: f64,
) -> Vec<Figure> {
    let mut figs = Vec::new();
    let mut panels = Vec::new();
    let title = if relative {
        format!("Relative response to orthogonalized shock to {shock}")
    } else {
        format!("Orthogonalized shock to {shock}")
    };
    let flush = |panels: &mut Vec<Panel>, figs: &mut Vec<Figure>| {
        if !panels.is_empty() {
            figs.push(Figure {
                name: format!("Bayesian_IRF_{shock}_{}", figs.len() + 1),
                title: title.clone(),
                rows: 3,
                cols: 3,
                panels: std::mem::take(panels),
            });
        }
    };
    for (j, st) in stats.iter().enumerate() {
        let mean: Vec<f64> = st.iter().map(|m| m.mean).collect();
        if mean.iter().fold(0.0f64, |a, v| a.max(v.abs())) < threshold {
            continue;
        }
        let x: Vec<f64> = (1..=st.len()).map(|t| t as f64).collect();
        let mut series = Vec::new();
        let lower: Vec<f64> = st.iter().map(|m| m.hpd.0).collect();
        if !lower.iter().any(|v| v.is_nan()) {
            series.push(Series::Band {
                x: x.clone(),
                lower,
                upper: st.iter().map(|m| m.hpd.1).collect(),
                color: Color::Gray,
                alpha: 0.3,
                label: None,
            });
        }
        series.push(Series::Line { x, y: mean, color: Color::Black, style: LineStyle::Solid, width: 3.0, label: None });
        panels.push(Panel { title: names[j].clone(), series, ..Default::default() });
        if panels.len() == 9 {
            flush(&mut panels, &mut figs);
        }
    }
    flush(&mut panels, &mut figs);
    figs
}
