// SPDX-License-Identifier: GPL-3.0-or-later
//! The estimation figures, built as [`graphics::Figure`]s (rendering is `graphics`' job):
//! - `<fname>_Priors<k>` (`plot_priors.m`, with `draw_prior_density.m`),
//! - `<fname>_CheckPlots<k>` (`mode_check.m`),
//! - `<fname>_PriorsAndPosteriors<k>` (`PlotPosteriorDistributions.m`),
//! - `<fname>_udiag<k>`, `<fname>_mdiag` (`mcmc_diagnostics.m`),
//! - `TracePlot_<name>_blck_<b>` (`trace_plot.m` / `generate_trace_plots.m`),
//! - `<fname>_SmoothedShocks<k>`, `<fname>_HistoricalAndSmoothedVariables<k>`
//!   (`plot_classical_smoother_results.m`) and `<fname>_SmoothedVariables<k>`.
//!
//! Colors and line styles follow the MATLAB code (gray prior, black posterior, green dashed
//! posterior mode, cyan mode line in the check plots, blue/red pooled/within diagnostics).

use crate::graphics::{Color, Figure, LineStyle, Panel, Series, pltorg};
use crate::linalg::Mat;

use super::diagnostics::linspace;
use super::mode_check::ModeCheckSlice;
use super::posterior::BrooksGelman;
use super::prior::{Prior, PriorShape, lpdfgam, lpdfgbeta, lpdfgweibull, lpdfig1, lpdfig2, lpdfnorm};
use super::special::{betainv, gaminv, norminv, wblinv};

/// `get_the_name.m` (TeX off) from dynare-rust's estimated-parameter names:
/// `stderr e` -> `SE_e`, `corr a, b` -> `CC_a_b`, a parameter keeps its name.
pub fn dynare_param_name(name: &str) -> String {
    if let Some(e) = name.strip_prefix("stderr ") {
        format!("SE_{}", e.trim())
    } else if let Some(p) = name.strip_prefix("corr ") {
        let v: Vec<&str> = p.split(',').map(str::trim).collect();
        format!("CC_{}", v.join("_"))
    } else {
        name.to_string()
    }
}

/// Port of `draw_prior_density.m`: 200 points between the 0.1% and 99.9% quantiles (1% /
/// 99% of the inverse gamma's gamma), the density, with values > 10 dropped (NaN) when the
/// density peaks at an end of the grid.
pub fn draw_prior_density(p: &Prior) -> (Vec<f64>, Vec<f64>) {
    let trunc = 1e-3;
    let steps = 200;
    let (p3, p4, p6, p7) = (p.p3, p.p4, p.p6, p.p7);
    let (inf, sup) = match p.shape {
        PriorShape::Beta => (betainv(trunc, p6, p7) * (p4 - p3) + p3, betainv(1.0 - trunc, p6, p7) * (p4 - p3) + p3),
        PriorShape::Gamma => (gaminv(trunc, p6, p7) + p3, gaminv(1.0 - trunc, p6, p7) + p3),
        PriorShape::Normal => (p3.max(norminv(trunc, p6, p7)), p4.min(norminv(1.0 - trunc, p6, p7))),
        PriorShape::InvGamma1 => (
            1.0 / gaminv(1.0 - 10.0 * trunc, p7 / 2.0, 2.0 / p6).sqrt() + p3,
            1.0 / gaminv(10.0 * trunc, p7 / 2.0, 2.0 / p6).sqrt() + p3,
        ),
        PriorShape::Uniform => (p6, p7),
        PriorShape::InvGamma2 => (
            1.0 / gaminv(1.0 - 10.0 * trunc, p7 / 2.0, 2.0 / p6) + p3,
            1.0 / gaminv(10.0 * trunc, p7 / 2.0, 2.0 / p6) + p3,
        ),
        PriorShape::Weibull => (p3 + wblinv(trunc, p7, p6), p3 + wblinv(1.0 - trunc, p7, p6)),
    };
    let x = linspace(inf, sup, steps);
    let mut f: Vec<f64> = x
        .iter()
        .map(|&v| match p.shape {
            PriorShape::Beta => lpdfgbeta(v, p6, p7, p3, p4).exp(),
            PriorShape::Gamma => lpdfgam(v - p3, p6, p7).exp(),
            PriorShape::Normal => lpdfnorm(v, p6, p7).exp(),
            PriorShape::InvGamma1 => lpdfig1(v - p3, p6, p7).exp(),
            PriorShape::Uniform => 1.0 / (sup - inf),
            PriorShape::InvGamma2 => lpdfig2(v - p3, p6, p7).exp(),
            PriorShape::Weibull => lpdfgweibull(v, p6, p7, p3).exp(),
        })
        .collect();
    if p.shape != PriorShape::Uniform {
        // [~,k1] = max(dens): first maximum, NaN ignored
        let mut k1 = 0;
        for (k, v) in f.iter().enumerate() {
            if *v > f[k1] || f[k1].is_nan() {
                k1 = k;
            }
        }
        if k1 == 0 || k1 == f.len() - 1 {
            for v in f.iter_mut() {
                if *v > 10.0 {
                    *v = f64::NAN;
                }
            }
        }
    }
    (x, f)
}

fn line(x: Vec<f64>, y: Vec<f64>, color: Color, style: LineStyle, width: f64, label: Option<&str>) -> Series {
    Series::Line { x, y, color, style, width, label: label.map(str::to_string) }
}

/// Splits `panels` into figures of `per` panels on an `nr x nc` grid named `<stem><k>`.
fn paginate(panels: Vec<Panel>, stem: &str, title: &str, nr: usize, nc: usize, per: usize) -> Vec<Figure> {
    let mut out = Vec::new();
    let mut it = panels.into_iter().peekable();
    let mut k = 1;
    while it.peek().is_some() {
        let page: Vec<Panel> = it.by_ref().take(per).collect();
        out.push(Figure { name: format!("{stem}{k}"), title: title.to_string(), rows: nr, cols: nc, panels: page });
        k += 1;
    }
    out
}

/// `plot_priors.m`: `Priors<k>`.
pub fn priors_figures(priors: &[Prior], names: &[String]) -> Vec<Figure> {
    let (_, nr, nc, nstar) = pltorg(priors.len());
    let panels = priors
        .iter()
        .zip(names)
        .map(|(p, n)| {
            let (x, f) = draw_prior_density(p);
            Panel {
                title: dynare_param_name(n),
                series: vec![line(x, f, Color::Gray, LineStyle::Solid, 2.0, None)],
                ..Default::default()
            }
        })
        .collect();
    paginate(panels, "Priors", "Priors", nr, nc, nstar)
}

/// `mode_check.m`: `CheckPlots<k>`: log posterior and log likelihood kernels (MATLAB's first
/// two line colors), the mode (cyan), failed points (red dots on the lower axis limit),
/// a binding bound (red dashed).
pub fn mode_check_figures(slices: &[ModeCheckSlice], names: &[String], nolik: bool) -> Vec<Figure> {
    let (_, nr, nc, nstar) = pltorg(slices.len());
    let panels = slices
        .iter()
        .zip(names)
        .map(|(s, n)| {
            let mut series =
                vec![line(s.z.clone(), s.log_post.clone(), Color::Palette(0), LineStyle::Solid, 1.0, Some("log-post"))];
            if !nolik {
                series.push(line(
                    s.z.clone(),
                    s.log_lik.clone(),
                    Color::Palette(1),
                    LineStyle::Solid,
                    1.0,
                    Some("log-lik kernel"),
                ));
            }
            series.push(Series::VLine { x: s.mode, color: Color::Rgb(0, 191, 191), style: LineStyle::Solid });
            let failed: Vec<f64> = s.z.iter().zip(&s.log_post).filter(|(_, v)| v.is_nan()).map(|(z, _)| *z).collect();
            if !failed.is_empty() {
                let lo = s
                    .log_post
                    .iter()
                    .chain(if nolik { [].iter() } else { s.log_lik.iter() })
                    .copied()
                    .filter(|v| v.is_finite())
                    .fold(f64::INFINITY, f64::min);
                let lo = if lo.is_finite() { lo } else { 0.0 };
                series.push(Series::Points { y: vec![lo; failed.len()], x: failed, color: Color::Red, label: None });
            }
            if let Some((b, _)) = s.binding_bound {
                series.push(Series::VLine { x: b, color: Color::Red, style: LineStyle::Dashed });
            }
            Panel { title: dynare_param_name(n), series, legend: true, ..Default::default() }
        })
        .collect();
    paginate(panels, "CheckPlots", "Mode check plots", nr, nc, nstar)
}

/// `PlotPosteriorDistributions.m`: `PriorsAndPosteriors<k>` (3x3): prior (gray), posterior
/// kernel density (black), posterior mode (green dashed).
pub fn priors_and_posteriors_figures(
    priors: &[Prior],
    names: &[String],
    densities: &[Option<(Vec<f64>, Vec<f64>)>],
    mode: Option<&[f64]>,
) -> Vec<Figure> {
    let panels = priors
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let (x2, f2) = draw_prior_density(p);
            let mut series = vec![line(x2, f2, Color::Gray, LineStyle::Solid, 2.0, None)];
            if let Some((x1, f1)) = &densities[k] {
                series.push(line(x1.clone(), f1.clone(), Color::Black, LineStyle::Solid, 2.0, None));
            }
            if let Some(m) = mode {
                series.push(Series::VLine { x: m[k], color: Color::Green, style: LineStyle::Dashed });
            }
            Panel { title: dynare_param_name(&names[k]), series, ..Default::default() }
        })
        .collect();
    paginate(panels, "PriorsAndPosteriors", "Priors and posteriors", 3, 3, 9)
}

/// `mcmc_diagnostics.m`: `udiag<k>`, three parameters per page (`brooksgelman.plotrows = 3`),
/// columns Interval / m2 / m3, pooled (blue) and mean within-chain (red).
pub fn udiag_figures(bg: &BrooksGelman, names: &[String]) -> Vec<Figure> {
    udiag_figures_rows(bg, names, 3)
}

/// `udiag_figures` with `options_.convergence.brooksgelman.plotrows` rows.
pub fn udiag_figures_rows(bg: &BrooksGelman, names: &[String], npardisp: usize) -> Vec<Figure> {
    let x: Vec<f64> = bg.iterations.iter().map(|&v| v as f64).collect();
    let panels_of = |k: usize| -> Vec<Panel> {
        let nam = dynare_param_name(&names[k]);
        [("Interval", 0), ("m2", 2), ("m3", 4)]
            .iter()
            .map(|(crit, c)| Panel {
                title: format!("{nam} ({crit})"),
                series: vec![
                    line(
                        x.clone(),
                        bg.udiag[k].iter().map(|u| u[*c]).collect(),
                        Color::Blue,
                        LineStyle::Solid,
                        0.8,
                        None,
                    ),
                    line(
                        x.clone(),
                        bg.udiag[k].iter().map(|u| u[c + 1]).collect(),
                        Color::Red,
                        LineStyle::Solid,
                        0.8,
                        None,
                    ),
                ],
                ..Default::default()
            })
            .collect()
    };
    let npar = names.len();
    let pages = npar / npardisp;
    let title = "MCMC univariate convergence diagnostic (Brooks and Gelman, 1998)";
    let mut out = Vec::new();
    for i in 0..pages {
        let panels = (i * npardisp..(i + 1) * npardisp).flat_map(panels_of).collect();
        out.push(Figure { name: format!("udiag{}", i + 1), title: title.into(), rows: npardisp, cols: 3, panels });
    }
    let reste = npar - pages * npardisp;
    if reste > 0 {
        let (nr, nc) = if reste == 1 { (3, 1) } else { (npardisp, 3) };
        let panels = (pages * npardisp..npar).flat_map(panels_of).collect();
        out.push(Figure { name: format!("udiag{}", pages + 1), title: title.into(), rows: nr, cols: nc, panels });
    }
    out
}

/// `mcmc_diagnostics.m`: `mdiag` (3x1) from [`super::diagnostics::brooks_gelman_multivariate`].
pub fn mdiag_figure(iterations: &[usize], mdiag: &[[f64; 6]]) -> Figure {
    let x: Vec<f64> = iterations.iter().map(|&v| v as f64).collect();
    let panels = [("Interval", 0), ("m2", 2), ("m3", 4)]
        .iter()
        .map(|(t, c)| Panel {
            title: t.to_string(),
            series: vec![
                line(x.clone(), mdiag.iter().map(|u| u[*c]).collect(), Color::Blue, LineStyle::Solid, 0.8, None),
                line(x.clone(), mdiag.iter().map(|u| u[c + 1]).collect(), Color::Red, LineStyle::Solid, 0.8, None),
            ],
            ..Default::default()
        })
        .collect();
    Figure { name: "mdiag".into(), title: "Multivariate convergence diagnostic".into(), rows: 3, cols: 1, panels }
}

/// Reduces a long series to at most `2*buckets` points (min and max of each bucket, in
/// order) so a million-draw trace stays a small file and looks the same.
pub fn decimate(y: &[f64], buckets: usize) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    if n <= 2 * buckets {
        return ((1..=n).map(|v| v as f64).collect(), y.to_vec());
    }
    let mut xs = Vec::with_capacity(2 * buckets);
    let mut ys = Vec::with_capacity(2 * buckets);
    for b in 0..buckets {
        let (s, e) = (b * n / buckets, (b + 1) * n / buckets);
        let (mut imin, mut imax) = (s, s);
        for i in s..e {
            if y[i] < y[imin] {
                imin = i;
            }
            if y[i] > y[imax] {
                imax = i;
            }
        }
        for i in if imin <= imax { [imin, imax] } else { [imax, imin] } {
            xs.push((i + 1) as f64);
            ys.push(y[i]);
        }
    }
    (xs, ys)
}

/// `trace_plot.m` for one chain: the draws (gray) and their centred moving average over
/// `2*ma+1` draws (black). `what` is e.g. "parameter rho" or "the posterior density".
pub fn trace_plot_figure(draws: &[f64], what: &str, block: usize, nblck: usize, ma: usize, stem: &str) -> Figure {
    let n = draws.len();
    let mut avg = vec![f64::NAN; n];
    if n > 2 * ma {
        let mut s: f64 = draws[..=2 * ma].iter().sum();
        for t in ma..n - ma {
            if t > ma {
                s += draws[t + ma] - draws[t - ma - 1];
            }
            avg[t] = s / (2 * ma + 1) as f64;
        }
    }
    let (x1, y1) = decimate(draws, 4000);
    let (x2, y2) = decimate(&avg, 4000);
    let mut title = format!("Trace plot for {what}");
    if nblck > 1 {
        title += &format!(" (block number {block}).");
    }
    Figure {
        name: stem.into(),
        title,
        rows: 1,
        cols: 1,
        panels: vec![Panel {
            series: vec![
                line(x1, y1, Color::Gray, LineStyle::Solid, 0.5, Some("MCMC draw")),
                line(x2, y2, Color::Black, LineStyle::Solid, 2.0, Some(&format!("{ma} period moving average"))),
            ],
            legend: true,
            ..Default::default()
        }],
    }
}

/// `trace_plot.m` for several chains: one line per chain.
pub fn trace_plot_chains_figure(draws: &[Vec<f64>], blocks: &[usize], what: &str, stem: &str) -> Figure {
    let series = draws
        .iter()
        .zip(blocks)
        .enumerate()
        .map(|(k, (d, b))| {
            let (x, y) = decimate(d, 3000);
            line(x, y, Color::Palette(k), LineStyle::Solid, 0.5, Some(&format!("Chain {b}")))
        })
        .collect();
    Figure {
        name: stem.into(),
        title: format!("Trace plot for {what}"),
        rows: 1,
        cols: 1,
        panels: vec![Panel { series, legend: true, ..Default::default() }],
    }
}

/// Time-series panels of one `nobs x nvars` matrix per variable.
fn series_panels(
    names: &[String],
    ticks: Option<&[(f64, String)]>,
    build: impl Fn(usize, &[f64]) -> Vec<Series>,
) -> Vec<Panel> {
    names
        .iter()
        .enumerate()
        .map(|(k, n)| Panel {
            title: n.clone(),
            series: build(k, &[]),
            x_ticks: ticks.map(<[_]>::to_vec),
            ..Default::default()
        })
        .collect()
}

fn col(m: &Mat, j: usize) -> Vec<f64> {
    (0..m.nrows()).map(|i| m[(i, j)]).collect()
}

/// `plot_classical_smoother_results.m`: `SmoothedShocks<k>` (zero line red, shock black).
pub fn smoothed_shocks_figures(exo_names: &[String], shocks: &Mat, ticks: Option<&[(f64, String)]>) -> Vec<Figure> {
    let gend = shocks.nrows();
    let t: Vec<f64> = (1..=gend).map(|v| v as f64).collect();
    let (_, nr, nc, nstar) = pltorg(exo_names.len());
    let panels = series_panels(exo_names, ticks, |k, _| {
        vec![
            line(vec![1.0, gend as f64], vec![0.0, 0.0], Color::Red, LineStyle::Solid, 0.5, None),
            line(t.clone(), col(shocks, k), Color::Black, LineStyle::Solid, 1.0, None),
        ]
    });
    paginate(panels, "SmoothedShocks", "Smoothed shocks", nr, nc, nstar)
}

/// `plot_classical_smoother_results.m`: `HistoricalAndSmoothedVariables<k>`: smoothed
/// observables (red) and the data (black dashed).
pub fn historical_and_smoothed_figures(
    varobs: &[String],
    smoothed: &Mat,
    data: &Mat,
    ticks: Option<&[(f64, String)]>,
) -> Vec<Figure> {
    let gend = smoothed.nrows();
    let t: Vec<f64> = (1..=gend).map(|v| v as f64).collect();
    let (_, nr, nc, nstar) = pltorg(varobs.len());
    let panels = series_panels(varobs, ticks, |k, _| {
        vec![
            line(t.clone(), col(smoothed, k), Color::Red, LineStyle::Solid, 1.0, None),
            line(t.clone(), col(data, k), Color::Black, LineStyle::Dashed, 1.0, None),
        ]
    });
    paginate(panels, "HistoricalAndSmoothedVariables", "Historical and smoothed variables", nr, nc, nstar)
}

/// `SmoothedVariables<k>`: the smoothed endogenous variables of the estimation's var list
/// (black), with the one-step-ahead filtered values (blue dashed) when available.
pub fn smoothed_variables_figures(
    names: &[String],
    smoothed: &Mat,
    filtered: Option<&Mat>,
    ticks: Option<&[(f64, String)]>,
) -> Vec<Figure> {
    let gend = smoothed.nrows();
    let t: Vec<f64> = (1..=gend).map(|v| v as f64).collect();
    let (_, nr, nc, nstar) = pltorg(names.len());
    let mut panels = series_panels(names, ticks, |k, _| {
        let mut s = vec![line(t.clone(), col(smoothed, k), Color::Black, LineStyle::Solid, 1.0, Some("smoothed"))];
        if let Some(f) = filtered {
            s.push(line(t.clone(), col(f, k), Color::Blue, LineStyle::Dashed, 0.8, Some("filtered (one step ahead)")));
        }
        s
    });
    if filtered.is_some() {
        for p in panels.iter_mut() {
            p.legend = true;
        }
    }
    paginate(panels, "SmoothedVariables", "Smoothed variables", nr, nc, nstar)
}

// ---------------------------------------------------------------------------
// driver glue
// ---------------------------------------------------------------------------

/// Saves every figure (`graphics::save`), returning how many were written.
pub fn save_figures(figs: &[Figure], fname: &str, gopts: &crate::graphics::GraphOptions) -> anyhow::Result<usize> {
    let mut n = 0;
    for f in figs {
        n += usize::from(!crate::graphics::save(f, fname, gopts)?.is_empty());
    }
    Ok(n)
}

/// Whether the MCMC may keep the burn-in draws in memory for the diagnostics and trace plots
/// (Dynare reads them back from its `_mh*_blck*.mat` files): at most 8 GB of draws.
pub fn keep_all_draws(npar: usize, replic: usize, nblocks: usize) -> bool {
    let bytes = (npar + 1) as f64 * replic as f64 * nblocks as f64 * 8.0;
    if bytes > 8e9 {
        println!(
            "WARNING: {:.1} GB of draws: only the draws after mh_drop are kept, so the Brooks-Gelman and Geweke diagnostics and the trace plots are skipped.",
            bytes / 1e9
        );
        return false;
    }
    true
}

/// `mcmc_diagnostics.m` (after the inefficiency factors): the Geweke tables, the Raftery-Lewis
/// tables and the Brooks-Gelman diagnostics (`udiag<k>`, `mdiag` figures with graphs).
pub fn mcmc_diagnostics(
    chains: &[super::mcmc::Chain],
    names: &[String],
    mh_drop: f64,
    fname: &str,
    gopts: &crate::graphics::GraphOptions,
    conv: &super::convergence::ConvergenceOptions, // options_.convergence
) -> anyhow::Result<()> {
    use super::diagnostics::{brooks_gelman_multivariate, geweke, print_geweke};
    let dyn_names: Vec<String> = names.iter().map(|s| dynare_param_name(s)).collect();
    if chains.is_empty() {
        return Ok(());
    }
    let ndraws = chains[0].total_draws;
    // mcmc_diagnostics.m:110: the 2000-draw cutoff does not apply to the slice sampler
    if !conv.slice && ndraws <= 2000 {
        // a MATLAB warning (stderr)
        eprintln!(
            "Warning: MCMC convergence diagnostics are not computed because the total number of iterations is not bigger than 2000!"
        );
    } else if chains.iter().any(|c| c.first_stored != 0) {
        crate::progress::note("MCMC convergence diagnostics skipped: the burn-in draws were not kept.");
    } else {
        let gw = &conv.geweke; // taper_steps, geweke_interval
        if let Some(g) = geweke(chains, mh_drop, gw) {
            print_geweke(&g, &dyn_names, gw);
        }
        if conv.raftery_lewis {
            super::convergence::raftery_lewis_chains(chains, &dyn_names, conv);
        }
        if chains.len() > 1 {
            println!("Univariate convergence diagnostic, Brooks and Gelman (1998):");
            let t = std::time::Instant::now();
            let mut k = 0;
            if let Some(bg) = super::posterior::brooks_gelman(chains, mh_drop, conv.slice) {
                for j in 1..=names.len() {
                    println!("    Parameter {j}...  Done! ");
                }
                println!();
                if gopts.enabled {
                    k += save_figures(&udiag_figures_rows(&bg, names, conv.brooks_gelman_plotrows), fname, gopts)?;
                }
            }
            if let Some((it, md)) = brooks_gelman_multivariate(chains, mh_drop, conv.slice)
                && gopts.enabled
            {
                k += save_figures(&[mdiag_figure(&it, &md)], fname, gopts)?;
            }
            crate::progress::note(format!(
                "mcmc_diagnostics: {k} figures ({fname}_udiag*, {fname}_mdiag) written to {} ({:.1}s).",
                gopts.dir.display(),
                t.elapsed().as_secs_f64()
            ));
        }
    }
    Ok(())
}

/// After the MCMC: `mcmc_diagnostics.m` when `diagnostics`, and `PlotPosteriorDistributions.m`
/// (`PriorsAndPosteriors<k>`).
#[allow(clippy::too_many_arguments)]
pub fn mcmc_figures(
    chains: &[super::mcmc::Chain],
    stats: &super::posterior::PosteriorStats,
    priors: &[Prior],
    names: &[String],
    mh_drop: f64,
    diagnostics: bool,
    fname: &str,
    gopts: &crate::graphics::GraphOptions,
    conv: &super::convergence::ConvergenceOptions, // options_.convergence
    xmode: Option<&[f64]>, // oo_.posterior_mode.parameters (None: smc or mh_posterior_mode_estimation)
) -> anyhow::Result<()> {
    use super::diagnostics::posterior_densities;
    if diagnostics {
        mcmc_diagnostics(chains, names, mh_drop, fname, gopts, conv)?;
    }
    if gopts.enabled && !chains.is_empty() {
        let dens = posterior_densities(chains, stats.first_draw);
        let figs = priors_and_posteriors_figures(priors, names, &dens, xmode);
        let k = save_figures(&figs, fname, gopts)?;
        crate::progress::note(format!(
            "{k} figures ({fname}_PriorsAndPosteriors*) written to {}.",
            gopts.dir.display()
        ));
    }
    Ok(())
}

/// `plot_classical_smoother_results.m` (`SmoothedShocks<k>`, `HistoricalAndSmoothedVariables<k>`)
/// plus `SmoothedVariables<k>` for the endogenous variables of the estimation's var list.
/// `smoothed`/`filtered` are `nobs x orig_endo_nbr` levels, `data` the `nobs x nvobs` sample.
#[allow(clippy::too_many_arguments)]
pub fn smoother_figures(
    model: &crate::model::Model,
    varobs: &[usize],
    var_list: &[String],
    smoothed: &Mat,
    shocks: &Mat,
    filtered: Option<&Mat>,
    data: &Mat,
    dates: Option<&[String]>,
    fname: &str,
    gopts: &crate::graphics::GraphOptions,
) -> anyhow::Result<()> {
    if !gopts.enabled {
        return Ok(());
    }
    let ticks = dates.map(crate::graphics::date_ticks);
    let ticks = ticks.as_deref();
    let mut figs = smoothed_shocks_figures(&model.exo_names, shocks, ticks);
    let obs_names: Vec<String> = varobs.iter().map(|&v| model.endo_names[v].clone()).collect();
    let yf = Mat::from_fn(smoothed.nrows(), varobs.len(), |t, i| smoothed[(t, varobs[i])]);
    figs.extend(historical_and_smoothed_figures(&obs_names, &yf, data, ticks));
    let vars: Vec<usize> =
        var_list.iter().filter_map(|v| model.endo_index(v)).filter(|&i| i < model.orig_endo_nbr).collect();
    if !vars.is_empty() {
        let pick = |m: &Mat| Mat::from_fn(m.nrows(), vars.len(), |t, i| m[(t, vars[i])]);
        let names: Vec<String> = vars.iter().map(|&v| model.endo_names[v].clone()).collect();
        let f = filtered.map(pick);
        figs.extend(smoothed_variables_figures(&names, &pick(smoothed), f.as_ref(), ticks));
    }
    let k = save_figures(&figs, fname, gopts)?;
    println!(
        "Smoother plots: {k} figures ({fname}_SmoothedShocks*, {fname}_HistoricalAndSmoothedVariables*{}) written to {}.",
        if vars.is_empty() { "" } else { ", SmoothedVariables*" },
        gopts.dir.display()
    );
    Ok(())
}

/// What `generate_trace_plots` needs from the last estimation with an MCMC.
pub struct McmcPlotContext {
    pub chains: Vec<super::mcmc::Chain>,
    /// dynare-rust estimated names (`stderr e`, `rho`, ...), in xparam order.
    pub names: Vec<String>,
    pub gopts: crate::graphics::GraphOptions,
    /// `options_.trace_plot_ma` (200).
    pub trace_plot_ma: usize,
}

/// `generate_trace_plots(chain_numbers)`: trace plots of the posterior kernel and of every
/// estimated parameter for the given chains (1-based, e.g. `[1:4]`; all draws, no burn-in
/// dropped). Files `TracePlot_<name>_blck_<b>[_<b2>...]` as in `trace_plot.m`.
pub fn generate_trace_plots(ctx: &McmcPlotContext, chain_numbers: &[usize]) -> anyhow::Result<usize> {
    let nblck = ctx.chains.len();
    if chain_numbers.is_empty() {
        anyhow::bail!("generate_trace_plots: no chain number given");
    }
    if let Some(&b) = chain_numbers.iter().find(|&&b| b == 0 || b > nblck) {
        anyhow::bail!("generate_trace_plots:: chain number is bigger than existing number of chains ({b} > {nblck})");
    }
    if ctx.chains.iter().any(|c| c.first_stored != 0) {
        println!(
            "WARNING: generate_trace_plots: the burn-in draws were not kept in memory; the trace plots start at the first kept draw."
        );
    }
    let series = |b: usize, j: Option<usize>| -> Vec<f64> {
        let c = &ctx.chains[b - 1];
        (c.first_stored..c.total_draws).map(|t| j.map_or(c.logpost_at(t), |j| c.draw(t)[j])).collect()
    };
    // (what, plot name, column): posterior first, then parameters, shocks' stderr, corr
    let mut items: Vec<(String, String, Option<usize>)> =
        vec![("the posterior density".into(), "Posterior".into(), None)];
    let order =
        |pred: &dyn Fn(&str) -> bool| -> Vec<usize> { (0..ctx.names.len()).filter(|&k| pred(&ctx.names[k])).collect() };
    for k in order(&|n| !n.starts_with("stderr ") && !n.starts_with("corr ")) {
        items.push((format!("parameter {}", ctx.names[k]), dynare_param_name(&ctx.names[k]), Some(k)));
    }
    for k in order(&|n| n.starts_with("stderr ")) {
        let e = ctx.names[k].trim_start_matches("stderr ");
        items.push((
            format!("the standard deviation of structural shock {e}"),
            dynare_param_name(&ctx.names[k]),
            Some(k),
        ));
    }
    for k in order(&|n| n.starts_with("corr ")) {
        let v: Vec<&str> = ctx.names[k].trim_start_matches("corr ").split(',').map(str::trim).collect();
        items.push((
            format!("the correlation between structural shocks {}", v.join(" and ")),
            dynare_param_name(&ctx.names[k]),
            Some(k),
        ));
    }
    let mut written = 0;
    for (what, plot_name, j) in items {
        let fig = if chain_numbers.len() == 1 {
            let b = chain_numbers[0];
            trace_plot_figure(
                &series(b, j),
                &what,
                b,
                nblck,
                ctx.trace_plot_ma,
                &format!("TracePlot_{plot_name}_blck_{b}"),
            )
        } else {
            let tag: Vec<String> = chain_numbers.iter().map(|b| b.to_string()).collect();
            let d: Vec<Vec<f64>> = chain_numbers.iter().map(|&b| series(b, j)).collect();
            trace_plot_chains_figure(&d, chain_numbers, &what, &format!("TracePlot_{plot_name}_blck_{}", tag.join("_")))
        };
        written += usize::from(!crate::graphics::save_as(&fig, &fig.name, &ctx.gopts)?.is_empty());
    }
    Ok(written)
}

/// Chain numbers of `generate_trace_plots([1:4])` / `(2)` / `([1 3])` from the raw text the
/// parser keeps for an unknown command's argument.
pub fn parse_chain_numbers(raw: &str) -> Vec<usize> {
    let s = raw.trim().trim_start_matches(['(', '[']).trim_end_matches([')', ']']);
    let mut out = Vec::new();
    for part in s.split([',', ' ']).filter(|p| !p.is_empty()) {
        let nums: Vec<usize> = part.split(':').filter_map(|v| v.trim().parse().ok()).collect();
        match nums.as_slice() {
            [a] => out.push(*a),
            [a, b] => out.extend(*a..=*b),
            [a, st, b] if *st > 0 => out.extend((*a..=*b).step_by(*st)),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_numbers() {
        assert_eq!(parse_chain_numbers("[1:4]"), vec![1, 2, 3, 4]);
        assert_eq!(parse_chain_numbers("2"), vec![2]);
        assert_eq!(parse_chain_numbers("[1 3]"), vec![1, 3]);
    }

    #[test]
    fn names() {
        assert_eq!(dynare_param_name("stderr ea"), "SE_ea");
        assert_eq!(dynare_param_name("corr a, b"), "CC_a_b");
        assert_eq!(dynare_param_name("rho"), "rho");
    }

    #[test]
    fn prior_density_integrates_to_one() {
        for (shape, m, s) in [
            (PriorShape::Beta, 0.6, 0.1),
            (PriorShape::Gamma, 1.0, 0.3),
            (PriorShape::Normal, 0.0, 1.0),
            (PriorShape::InvGamma1, 0.5, 0.5),
            (PriorShape::Uniform, 0.0, 1.0),
        ] {
            let p = Prior::new(shape, m, s, None, None).unwrap();
            let (x, f) = draw_prior_density(&p);
            assert_eq!(x.len(), 200);
            let mass: f64 = x.windows(2).zip(f.windows(2)).map(|(a, b)| (a[1] - a[0]) * (b[0] + b[1]) / 2.0).sum();
            let expect = if shape == PriorShape::InvGamma1 {
                0.98
            } else if shape == PriorShape::Uniform {
                1.0
            } else {
                0.998
            };
            assert!((mass - expect).abs() < 0.01, "{shape:?}: {mass}");
        }
    }

    #[test]
    fn pages() {
        let names: Vec<String> = (0..73).map(|k| format!("p{k}")).collect();
        let p = Prior::new(PriorShape::Normal, 0.0, 1.0, None, None).unwrap();
        let figs = priors_figures(&vec![p; 73], &names);
        assert_eq!(figs.len(), 9);
        assert_eq!(figs[8].panels.len(), 1);
        assert_eq!((figs[8].rows, figs[8].cols), (3, 3));
        assert_eq!(figs[0].name, "Priors1");
        let (x, y) = decimate(&(0..100000).map(|v| (v as f64).sin()).collect::<Vec<_>>(), 1000);
        assert_eq!(x.len(), 2000);
        assert!(y.iter().all(|v| v.abs() <= 1.0));
    }
}
