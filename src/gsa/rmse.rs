// SPDX-License-Identifier: GPL-3.0-or-later
//! RMSE mapping, port of `+gsa/monte_carlo_filtering.m` (with the per-draw quantities that
//! `prior_posterior_statistics(type='gsa')` computes): for every stable point of the sample,
//! the posterior kernel and the one-step-ahead forecast errors of the observed variables;
//! then Smirnov tests of the best `pfilt_rmse` share of the sample against the rest, per
//! observed series (or, with `lik_only`, by posterior kernel and likelihood).

use std::path::Path;

use anyhow::{Result, anyhow, bail};
use rayon::prelude::*;

use super::mcf::{McfSpec, analysis, cdf_series};
use super::stability::StabMap;
use super::stats::{argsort, dyntable, mean, num2str, smirnov_test, std};
use super::{Ctx, GsaOptions};
use crate::data::Table;
use crate::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
use crate::graphics::{Color, Figure, LineStyle, Panel};
use crate::kalman::{KalmanOptions, compute_pinf_pstar, smoother};
use crate::linalg::Mat;
use crate::modfile::{ModFile, OptValue, Options, opt};
use crate::solver::{StateSpaceIndex, kalman_transition_matrix};

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

/// `dynare_estimation_init`'s data and filter options for the GSA (`sensitivity` options
/// over those of a previous `estimation`).
pub fn likelihood_data(mf: &ModFile, dir: &Path, o: &Options, datafile: &str) -> Result<LikelihoodData> {
    let p = dir.join(datafile);
    let path = if p.exists() { p } else { dir.join(format!("{datafile}.csv")) };
    if !path.exists() {
        bail!("datafile {} not found (only .csv files are supported)", dir.join(datafile).display());
    }
    let table = Table::read_csv(&path)?;
    let first_obs = num(o, "first_obs").unwrap_or(1.0) as usize;
    let sample = table.sample(&mf.varobs, first_obs, num(o, "nobs").map(|v| v as usize))?;
    let mut k = KalmanOptions {
        presample: num(o, "presample").unwrap_or(0.0) as usize,
        lik_init: num(o, "lik_init").unwrap_or(1.0) as u8,
        kalman_algo: num(o, "kalman_algo").unwrap_or(0.0) as u8,
        ..Default::default()
    };
    if opt(o, "diffuse_filter").is_some() {
        k.lik_init = 3;
    }
    if let Some(v) = num(o, "kalman_tol") {
        k.kalman_tol = v;
    }
    if let Some(v) = num(o, "diffuse_kalman_tol") {
        k.diffuse_kalman_tol = v;
    }
    if let Some(v) = num(o, "riccati_tol") {
        k.riccati_tol = v;
    }
    k.qz_criterium = super::qz_criterium(o, true); // rmse always runs dynare_estimation_init
    Ok(LikelihoodData::new(&sample, k))
}

/// Per-draw output of `prior_posterior_statistics_core` that the RMSE mapping uses.
struct DrawOut {
    logpo: f64,
    /// rmse and r2 per `var_rmse` variable (empty with lik_only).
    rmse: Vec<f64>,
    r2: Vec<f64>,
}

/// RMSE and R2 of the one-step-ahead predictions of `vars` (endogenous indices) at `x`,
/// periods `istart..` (1-based, as monte_carlo_filtering.m: `yobs(istart:end) - y0(istart:end)`
/// with `y0 = aK(1, :, t)` and `yobs` the updated variables).
fn fit_at(
    dsge: &DsgeModel,
    data: &LikelihoodData,
    x: &[f64],
    vars: &[usize],
    istart: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let res = dsge.resolve(x).map_err(|e| anyhow!("no solution (info {})", e.info))?;
    let dr = &res.dr;
    let full = StateSpaceIndex::full(dr);
    let (t, r) = kalman_transition_matrix(dr, &full);
    let mf: Vec<usize> = dsge.varobs.iter().map(|&v| dr.inv_order_var[v]).collect();
    let mut y = data.y.clone();
    for (i, &v) in dsge.varobs.iter().enumerate() {
        let c = res.steady_state[v];
        for j in 0..y.ncols() {
            y[(i, j)] -= c;
        }
    }
    let opts = &data.kalman;
    let (pstar, pinf) = compute_pinf_pstar(&mf, &t, &r, &res.q, opts.qz_criterium, &[]);
    let pinf = (opts.lik_init == 3).then_some(&pinf);
    let out = smoother(&y, &data.missing, &mf, &t, &r, &res.q, None, &pstar, pinf, opts, false)?;
    let gend = y.ncols();
    let mut rmse = Vec::with_capacity(vars.len());
    let mut r2 = Vec::with_capacity(vars.len());
    for &v in vars {
        let row = dr.inv_order_var[v];
        let (mut se, mut so, mut n) = (0.0, 0.0, 0.0);
        for tt in istart - 1..gend {
            // y0(t) = aK(1, :, t) = filtered(:, t-1) (1-based); yobs(t) = updated(:, t)
            let y0 = out.filtered[(row, tt - 1)];
            let yo = out.ahat[(row, tt)];
            se += (yo - y0).powi(2);
            so += yo * yo;
            n += 1.0;
        }
        rmse.push((se / n).sqrt());
        r2.push(1.0 - (se / n) / (so / n));
    }
    Ok((rmse, r2))
}

fn char_block(names: &[String]) -> String {
    let w = names.iter().map(|s| s.len()).max().unwrap_or(0);
    names.iter().map(|s| format!("{s:<w$}\n")).collect()
}

/// `monte_carlo_filtering.m` (`ppost = 0`).
pub fn run(
    ctx: &Ctx,
    mf: &ModFile,
    dir: &Path,
    dsge: &DsgeModel,
    o: &GsaOptions,
    eo: &Options,
    stab: &StabMap,
) -> Result<RmseResults> {
    let datafile = o.datafile.as_deref().ok_or_else(|| anyhow!("rmse needs a datafile"))?;
    let data = likelihood_data(mf, dir, eo, datafile)?;
    let istart = o.istart_rmse.unwrap_or(data.kalman.presample + 1).max(2);
    let vnames = o.var_rmse.clone();
    let vars: Vec<usize> = vnames.iter().map(|n| super::endo_index(dsge, n)).collect::<Result<_>>()?;
    let names = &ctx.param_names;
    let npar = names.len();

    println!("\nStarting sensitivity analysis");
    println!("for the fit of EACH observed series ...\n");
    if !o.nograph {
        println!("Deleting old SA figures...");
        println!("done !");
    }
    // monte_carlo_filtering.m: ppost uses the posterior draws and ignores lik_only
    let ppost = o.ppost;
    let pprior = o.pprior && !ppost;
    let lik_only = o.lik_only && !ppost;
    let tag = if ppost {
        "post"
    } else if pprior {
        "prior"
    } else {
        "mc"
    };
    let mut x: Vec<Vec<f64>> = stab.istable.iter().map(|&k| stab.point(k)).collect();
    let compute_fit = !lik_only;
    let mut loaded: Option<Saved> = None;
    if o.load_rmse {
        loaded = load_saved(ctx, tag, &vnames, lik_only);
    }
    let (logpo2, likelihood, rmse_mc, r2_mc, lnprior, rmse_txt);
    let mut nruns = x.len();
    if let Some(sv) = loaded {
        // load_rmse: x, logpo2, likelihood, rmse_MC, r2_MC (and the mode / posterior mean fits) of the saved file
        x = sv.x;
        nruns = x.len();
        lnprior = sv.logpo2.iter().zip(&sv.likelihood).map(|(a, b)| a - b).collect::<Vec<f64>>();
        logpo2 = sv.logpo2;
        likelihood = sv.likelihood;
        rmse_mc = sv.rmse_mc;
        r2_mc = sv.r2_mc;
        rmse_txt = pick_txt(ppost, pprior, sv.pmean, sv.mode, vars.len());
    } else {
        let post = DsgePosterior { dsge, data: &data, bounds: &dsge.bounds };
        if compute_fit {
            println!("\nComputing RMSE's...");
        }
        let outs: Vec<DrawOut> = x
            .par_iter()
            .map(|xi| {
                let logpo = post.evaluate(xi).map_or(f64::NEG_INFINITY, |e| -e.fval);
                let (rmse, r2) = if compute_fit {
                    fit_at(dsge, &data, xi, &vars, istart)
                        .unwrap_or_else(|_| (vec![f64::NAN; vars.len()], vec![f64::NAN; vars.len()]))
                } else {
                    (Vec::new(), Vec::new())
                };
                DrawOut { logpo, rmse, r2 }
            })
            .collect();
        logpo2 = outs.iter().map(|d| d.logpo).collect::<Vec<f64>>();
        lnprior = x.iter().map(|xi| dsge.log_prior(xi)).collect::<Vec<f64>>();
        likelihood = logpo2.iter().zip(&lnprior).map(|(a, b)| a - b).collect::<Vec<f64>>();
        rmse_mc = outs.iter().map(|d| d.rmse.clone()).collect();
        r2_mc = outs.iter().map(|d| d.r2.clone()).collect();
        // the fit at the posterior mode (mode_file) and, for ppost / a mean file, at the posterior mean
        let mode = match &o.mode_file {
            Some(f) if compute_fit => Some(fit_at(dsge, &data, &read_xparam1(dir, f)?, &vars, istart)?),
            _ => None,
        };
        let pmean = if compute_fit {
            match read_mean(dir, &ctx.fname, o, ppost, &x) {
                Some(xm) => Some(fit_at(dsge, &data, &xm, &vars, istart)?),
                None => None,
            }
        } else {
            None
        };
        rmse_txt = pick_txt(ppost, pprior, pmean.clone(), mode.clone(), vars.len());
        let res = RmseResults {
            logpo2: logpo2.clone(),
            likelihood: likelihood.clone(),
            rmse_mc: rmse_mc.clone(),
            r2_mc: r2_mc.clone(),
        };
        write_results(ctx, tag, &vnames, &x, &res, mode.as_ref(), pmean.as_ref())?;
    }
    let nfilt = (o.pfilt_rmse * nruns as f64).floor() as usize;
    println!("... done!");
    let neg = |v: &[f64]| v.iter().map(|a| -a).collect::<Vec<f64>>();
    let ipost = argsort(&neg(&logpo2));
    let ilik = argsort(&neg(&likelihood));
    let mut res = RmseResults { logpo2: logpo2.clone(), likelihood: likelihood.clone(), ..Default::default() };

    if lik_only {
        let (anam, atitle) = if pprior {
            ("rmse_prior_post", "RMSE prior: Log Posterior Kernel")
        } else {
            ("rmse_mc_post", "RMSE MC: Log Posterior Kernel")
        };
        let mk = |n: &str, t: &str, b: &str, nb: &str| McfSpec {
            pvalue_ks: o.alpha_rmse,
            pvalue_corr: o.alpha2_rmse,
            alpha2: 0.0,
            param_names: names,
            amcf_name: n.into(),
            amcf_title: t.into(),
            beha_title: b.into(),
            nobeha_title: nb.into(),
            title: t.into(),
            xparam1: None,
            noprint: false,
        };
        analysis(
            ctx,
            &x,
            &ipost[..nfilt],
            &ipost[nfilt..],
            &mk(anam, atitle, "better posterior kernel", "worse posterior kernel"),
        );
        let (anam, atitle) = if pprior {
            ("rmse_prior_lik", "RMSE prior: Log Likelihood Kernel")
        } else {
            ("rmse_mc_lik", "RMSE MC: Log Likelihood Kernel")
        };
        analysis(ctx, &x, &ilik[..nfilt], &ilik[nfilt..], &mk(anam, atitle, "better likelihood", "worse likelihood"));
        return Ok(res);
    }

    let nv = vars.len();
    let colv = |m: &[Vec<f64>], i: usize| m.iter().map(|r| r[i]).collect::<Vec<f64>>();
    if !o.nograph {
        // scatter_analysis(r2_MC, x, ...)
        println!();
        let (t, n) = if ppost {
            ("R2 Scatter plot: Posterior", "r2_post")
        } else if pprior {
            ("R2 Scatter plot: Prior", "r2_prior")
        } else {
            ("R2 Scatter plot: MC", "r2_mc")
        };
        ctx.save(&super::mcf::scatter_plots(&r2_mc, &vnames, n, t));
    }
    let ixx: Vec<Vec<usize>> = (0..nv).map(|i| argsort(&colv(&rmse_mc, i))).collect();
    // Smirnov tests, best nfilt vs the rest, per parameter and series
    let mut pp = vec![vec![1.0; nv]; npar];
    let mut ss = vec![vec![0.0; nv]; npar];
    let mut ppv = vec![vec![vec![1.0; npar]; nv]; nv]; // ppv[i][l][j]
    for j in 0..npar {
        for i in 0..nv {
            let rest: Vec<f64> = ixx[i][nfilt..].iter().map(|&k| x[k][j]).collect();
            let best: Vec<f64> = ixx[i][..nfilt].iter().map(|&k| x[k][j]).collect();
            let (_, p, _) = smirnov_test(&rest, &best, o.alpha_rmse, 0);
            let (h1, _, _) = smirnov_test(&rest, &best, o.alpha_rmse, 1);
            let (h2, _, _) = smirnov_test(&rest, &best, o.alpha_rmse, -1);
            ss[j][i] = if !h1 && !h2 {
                1.0
            } else if !h1 {
                -1.0
            } else {
                0.0
            };
            pp[j][i] = p;
        }
        for i in 0..nv {
            for l in 0..nv {
                if l != i && pp[j][i] < o.alpha_rmse && pp[j][l] < o.alpha_rmse {
                    let a: Vec<f64> = ixx[i][..nfilt].iter().map(|&k| x[k][j]).collect();
                    let b: Vec<f64> = ixx[l][..nfilt].iter().map(|&k| x[k][j]).collect();
                    ppv[i][l][j] = smirnov_test(&a, &b, o.alpha_rmse, 0).1;
                } else if l == i {
                    ppv[i][l][j] = pp[j][i];
                }
            }
        }
    }
    if !o.nograph {
        lnlik_figures(ctx, tag, &vnames, &ixx, nfilt, &lnprior, &likelihood, &logpo2);
    }
    let pf = num2str(o.pfilt_rmse * 100.0);
    let minmax = |v: &[f64]| {
        (
            v.iter().copied().filter(|a| !a.is_nan()).fold(f64::INFINITY, f64::min),
            v.iter().copied().filter(|a| !a.is_nan()).fold(f64::NEG_INFINITY, f64::max),
        )
    };
    println!();
    let mm: Vec<Vec<f64>> = (0..nv)
        .map(|i| {
            let (a, b) = minmax(&colv(&rmse_mc, i));
            vec![a, b]
        })
        .collect();
    print!(
        "{}",
        dyntable("RMSE over the MC sample:", &["Variable", "min yr RMSE", "max yr RMSE"], &vnames, &mm, 0, 15, 5)
    );
    let cv: Vec<f64> = (0..nv)
        .map(|i| {
            let c = colv(&rmse_mc, i);
            std(&c) / mean(&c)
        })
        .collect();
    let invar: Vec<usize> = (0..nv).filter(|&i| cv[i] <= 0.0001).collect();
    if !invar.is_empty() {
        println!("\nRMSE is not varying significantly over the MC sample for the following variables:");
        for &i in &invar {
            println!("{}", vnames[i]);
        }
        println!("These variables are excluded from SA");
        println!("[Unless you treat these series as exogenous, there is something wrong in your estimation !]");
    }
    let ivar: Vec<usize> = (0..nv).filter(|&i| cv[i] > 0.0001).collect();
    let vnames_f: Vec<String> = ivar.iter().map(|&i| vnames[i].clone()).collect();
    println!("\nSample filtered the {pf}% best RMSE's for each observed series ...\n");
    let last_header = if pprior { "posterior mode" } else { "posterior mean" };
    let ranges = |m: &[Vec<f64>], txt: &[f64]| -> Vec<Vec<f64>> {
        // Dynare indexes ixx and rmse_txt by the position in the filtered list
        (0..ivar.len())
            .map(|j| {
                let c = colv(m, ivar[j]);
                let (a, b) = minmax(&ixx[j][..nfilt].iter().map(|&k| c[k]).collect::<Vec<_>>());
                let (cc, d) = minmax(&ixx[j][nfilt..].iter().map(|&k| c[k]).collect::<Vec<_>>());
                vec![a, b, cc, d, txt[j]]
            })
            .collect()
    };
    let optional = |title: &str, rows: &[Vec<f64>]| -> String {
        let headers = ["Variable", "min", "max", "min", "max", last_header];
        let lw = vnames_f.iter().map(|s| s.len()).chain(std::iter::once(8)).max().unwrap_or(8) + 2;
        let fin: Vec<f64> = rows.iter().flatten().copied().filter(|v| v.is_finite()).collect();
        let mut vl = if fin.is_empty() {
            4
        } else {
            (fin.iter().map(|v| v.abs().log10()).fold(f64::NEG_INFINITY, f64::max).ceil().max(1.0) as usize) + 5 + 1
        };
        if rows.iter().flatten().any(|&v| v < 0.0) {
            vl += 1;
        }
        let hl = headers[1..].iter().map(|h| h.len()).max().unwrap();
        let vw = (hl.max(vl) + 2).max(15);
        let (b, r) = if pprior {
            (format!("best {pf}% filtered"), "remaining 90%".to_string())
        } else {
            ("best  filtered".to_string(), "remaining".to_string())
        };
        let opt_header = format!("{:<lw$}{:>vw$}{:>vw$}{:>vw$}{:>vw$}", "", "", b, "", r);
        let t = dyntable(title, &headers, &vnames_f, rows, 0, vw, 5);
        // dyntable prints the optional header between the title and the column headers
        let (head, body) = t.split_at(t.find("Variable").unwrap());
        format!("{head}{opt_header}\n{body}")
    };
    let rr = ranges(&rmse_mc, &rmse_txt.0);
    print!("{}", optional("RMSE ranges after filtering:", &rr));
    println!();
    let mm2: Vec<Vec<f64>> = (0..nv)
        .map(|i| {
            let (a, b) = minmax(&colv(&r2_mc, i));
            vec![a, b]
        })
        .collect();
    print!("{}", dyntable("R2 over the MC sample:", &["Variable", "min yr R2", "max yr R2"], &vnames, &mm2, 0, 15, 5));
    println!("\nSample filtered the {pf}% best R2's for each observed series ...\n");
    println!("R2 ranges after filtering:");
    let rr2 = ranges(&r2_mc, &rmse_txt.1);
    print!("{}", optional("R2 ranges after filtering:", &rr2));
    // SP: significant at alpha for series j (filtered list position, as Dynare)
    let mut sp = vec![vec![0.0; nv]; npar];
    for j in 0..ivar.len() {
        for k in 0..npar {
            if pp[k][j] < o.alpha_rmse {
                sp[k][j] = 1.0;
            }
            ss[k][j] *= sp[k][j];
        }
    }
    let nsp: Vec<usize> = (0..npar).map(|k| sp[k].iter().filter(|&&v| v != 0.0).count()).collect();
    let pick =
        |f: &dyn Fn(usize) -> bool| (0..npar).filter(|&k| f(nsp[k])).map(|k| names[k].clone()).collect::<Vec<_>>();
    let snam0 = pick(&|n| n == 0);
    let snam1 = pick(&|n| n == 1);
    let snam2 = pick(&|n| n > 1);
    println!("\nThese parameters do not affect significantly the fit of ANY observed series:");
    print!("{}", char_block(&snam0));
    println!("\nThese parameters affect ONE single observed series:");
    print!("{}", char_block(&snam1));
    println!("\nThese parameters affect MORE THAN ONE observed series: trade off exists!");
    print!("{}", char_block(&snam2));
    if !o.nograph {
        let (temp_name, atitle, asname) = if ppost {
            ("RMSE Posterior Tradeoffs:", "RMSE Posterior Map:", "rmse_post")
        } else if pprior {
            ("RMSE Prior Tradeoffs:", "RMSE Prior Map:", "rmse_prior")
        } else {
            ("RMSE MC Tradeoffs:", "RMSE MC Map:", "rmse_mc")
        };
        for (iy, v) in vnames_f.iter().enumerate() {
            let spec = McfSpec {
                pvalue_ks: o.alpha_rmse,
                pvalue_corr: o.alpha2_rmse,
                alpha2: 0.0,
                param_names: names,
                amcf_name: format!("{asname}_{v}_map"),
                amcf_title: format!("{atitle} {v}"),
                beha_title: format!("better fit of {v}"),
                nobeha_title: format!("worse fit of {v}"),
                title: format!("the fit of {v}"),
                xparam1: None,
                noprint: false,
            };
            analysis(ctx, &x, &ixx[iy][..nfilt], &ixx[iy][nfilt..], &spec);
        }
        // trade-off figures: CDFs of each parameter in the best sets of the series
        for (iy, v) in vnames_f.iter().enumerate() {
            let ipar: Vec<usize> = (0..npar).filter(|&j| (0..nv).any(|l| ppv[iy][l][j] < o.alpha_rmse)).collect();
            for (ix, chunk) in ipar.chunks(5).enumerate() {
                let panels = chunk
                    .iter()
                    .map(|&j| {
                        let mut s =
                            vec![cdf_series(&colv(&x, j), Color::Black, LineStyle::Solid, 1.0, Some("base".into()))];
                        for i in 0..vnames_f.len() {
                            if ppv[iy][i][j] < o.alpha_rmse || i == iy {
                                let b: Vec<f64> = ixx[i][..nfilt].iter().map(|&k| x[k][j]).collect();
                                s.push(cdf_series(
                                    &b,
                                    Color::Palette(i),
                                    LineStyle::Solid,
                                    2.0,
                                    Some(vnames_f[i].clone()),
                                ));
                            }
                        }
                        Panel { title: names[j].clone(), series: s, legend: true, ..Default::default() }
                    })
                    .collect();
                ctx.save(&Figure {
                    name: format!("{asname}_{v}_{}", ix + 1),
                    title: format!("{temp_name} observed variable {v}"),
                    rows: 2,
                    cols: 3,
                    panels,
                });
            }
        }
        let nsnam: Vec<usize> = (0..npar).filter(|&k| nsp[k] > 1).collect();
        for (ix, chunk) in nsnam.chunks(5).enumerate() {
            let panels = chunk
                .iter()
                .map(|&j| {
                    let mut s =
                        vec![cdf_series(&colv(&x, j), Color::Black, LineStyle::Solid, 1.0, Some("base".into()))];
                    for i in 0..vnames_f.len() {
                        if sp[j][i] != 0.0 {
                            let b: Vec<f64> = ixx[i][..nfilt].iter().map(|&k| x[k][j]).collect();
                            s.push(cdf_series(&b, Color::Palette(i), LineStyle::Solid, 2.0, Some(vnames_f[i].clone())));
                        }
                    }
                    Panel { title: names[j].clone(), series: s, legend: true, ..Default::default() }
                })
                .collect();
            ctx.save(&Figure {
                name: format!("{asname}_params_{}", ix + 1),
                title: format!("{temp_name} estimated params and shocks {}", ix + 1),
                rows: 2,
                cols: 3,
                panels,
            });
        }
    }
    res.rmse_mc = rmse_mc;
    res.r2_mc = r2_mc;
    Ok(res)
}

/// What the RMSE mapping saves (`x`, `logpo2`, `likelihood`, `rmse_MC`, `r2_MC`).
#[derive(Default, Clone, Debug)]
pub struct RmseResults {
    /// Log posterior kernel at each stable point (`logpo2`).
    pub logpo2: Vec<f64>,
    /// `logpo2 - log prior`.
    pub likelihood: Vec<f64>,
    /// `nstable x nvar` (empty with lik_only).
    pub rmse_mc: Vec<Vec<f64>>,
    pub r2_mc: Vec<Vec<f64>>,
}

/// A saved RMSE analysis (`<fname>_<prior|mc|post>.mat` of Dynare, or dynare-rust's `_rmse.mat`).
struct Saved {
    x: Vec<Vec<f64>>,
    logpo2: Vec<f64>,
    likelihood: Vec<f64>,
    rmse_mc: Vec<Vec<f64>>,
    r2_mc: Vec<Vec<f64>>,
    mode: Option<(Vec<f64>, Vec<f64>)>,
    pmean: Option<(Vec<f64>, Vec<f64>)>,
}

fn file_of(ctx: &Ctx, tag: &str, rust: bool) -> std::path::PathBuf {
    // ppost: Dynare's file is `<fname>_post.mat` itself; otherwise Dynare appends to `_prior`/`_mc`
    if rust && tag != "post" {
        ctx.out_dir.join(format!("{}_{}_rmse.mat", ctx.fname, tag))
    } else {
        ctx.out_dir.join(format!("{}_{}.mat", ctx.fname, tag))
    }
}

/// `load_rmse = 1`: reads the saved `x, logpo2, likelihood, rmse_MC, r2_MC, rmse_mode, rmse_pmean`
/// (monte_carlo_filtering.m); `None` (-> recompute) when the file or the variable list differs.
fn load_saved(ctx: &Ctx, tag: &str, vnames: &[String], lik_only: bool) -> Option<Saved> {
    for rust in [true, false] {
        let path = file_of(ctx, tag, rust);
        if !path.exists() {
            continue;
        }
        let Ok(m) = crate::matfile::read(&path) else { continue };
        let rows = |k: &str| {
            m.get(k)
                .and_then(|v| v.as_matrix())
                .map(|a| (0..a.nrows()).map(|i| a.row(i).iter().copied().collect::<Vec<f64>>()).collect::<Vec<_>>())
        };
        let vec = |k: &str| m.get(k).and_then(|v| v.as_vec());
        let (Some(x), Some(logpo2), Some(likelihood)) = (rows("x"), vec("logpo2"), vec("likelihood")) else { continue };
        let (mut rmse_mc, mut r2_mc) = (Vec::new(), Vec::new());
        if !lik_only {
            let (Some(a), Some(b)) = (rows("rmse_MC"), rows("r2_MC")) else { continue };
            // the list of variables must be the one requested (monte_carlo_filtering.m)
            let same = match m.get("vvarvecm").and_then(|v| v.as_strings()) {
                Some(l) => l == vnames,
                None => a.first().map_or(0, Vec::len) == vnames.len(),
            };
            if !same {
                println!(
                    "WARNING: cannot load results since the list of variables in the mat file differs from the one requested."
                );
                return None;
            }
            rmse_mc = a;
            r2_mc = b;
        }
        let pair = |a: &str, b: &str| Some((vec(a)?, vec(b)?));
        return Some(Saved {
            x,
            logpo2,
            likelihood,
            rmse_mc,
            r2_mc,
            mode: pair("rmse_mode", "r2_mode"),
            pmean: pair("rmse_pmean", "r2_pmean"),
        });
    }
    println!("WARNING: load_rmse: no saved RMSE results found in {}; computing them.", ctx.out_dir.display());
    None
}

/// The row of mode / posterior-mean fits monte_carlo_filtering.m prints next to the ranges.
fn pick_txt(
    ppost: bool,
    pprior: bool,
    pmean: Option<(Vec<f64>, Vec<f64>)>,
    mode: Option<(Vec<f64>, Vec<f64>)>,
    nv: usize,
) -> (Vec<f64>, Vec<f64>) {
    let nan = (vec![f64::NAN; nv], vec![f64::NAN; nv]);
    if ppost {
        pmean.unwrap_or(nan)
    } else if pprior || pmean.is_none() {
        mode.unwrap_or(nan)
    } else {
        pmean.unwrap_or(nan)
    }
}

/// `xparam1` of `<fname>/Output/<fname>_mean.mat` (written by Dynare's estimation); with `ppost`
/// and no such file, the mean of the posterior draws.
fn read_mean(dir: &Path, fname: &str, o: &GsaOptions, ppost: bool, x: &[Vec<f64>]) -> Option<Vec<f64>> {
    let p = dir.join(fname).join("Output").join(format!("{fname}_mean.mat"));
    let from_file = crate::matfile::read(&p).ok().and_then(|m| m.get("xparam1").and_then(|v| v.as_vec()));
    if from_file.is_some() && (ppost || o.mode_file.is_some()) {
        return from_file;
    }
    if ppost && !x.is_empty() {
        let n = x[0].len();
        return Some((0..n).map(|j| x.iter().map(|r| r[j]).sum::<f64>() / x.len() as f64).collect());
    }
    None
}

fn write_results(
    ctx: &Ctx,
    tag: &str,
    vnames: &[String],
    x: &[Vec<f64>],
    r: &RmseResults,
    mode: Option<&(Vec<f64>, Vec<f64>)>,
    pmean: Option<&(Vec<f64>, Vec<f64>)>,
) -> Result<()> {
    use crate::matfile::{MatOut, write};
    let rows = |m: &[Vec<f64>]| {
        let nc = m.first().map_or(0, Vec::len);
        Mat::from_fn(m.len(), nc, |i, j| m[i][j])
    };
    let (xm, rm, r2) = (rows(x), rows(&r.rmse_mc), rows(&r.r2_mc));
    let path = file_of(ctx, tag, true);
    let mut vars = vec![
        ("x", MatOut::Matrix(&xm)),
        ("logpo2", MatOut::Vector(&r.logpo2)),
        ("likelihood", MatOut::Vector(&r.likelihood)),
    ];
    if !r.rmse_mc.is_empty() {
        vars.push(("rmse_MC", MatOut::Matrix(&rm)));
        vars.push(("r2_MC", MatOut::Matrix(&r2)));
        vars.push(("vvarvecm", MatOut::Strings(vnames)));
    }
    if let Some((a, b)) = mode {
        vars.push(("rmse_mode", MatOut::Vector(a)));
        vars.push(("r2_mode", MatOut::Vector(b)));
    }
    if let Some((a, b)) = pmean {
        vars.push(("rmse_pmean", MatOut::Vector(a)));
        vars.push(("r2_pmean", MatOut::Vector(b)));
    }
    write(&path, &vars)
}

fn read_xparam1(dir: &Path, f: &str) -> Result<Vec<f64>> {
    let p = dir.join(if f.ends_with(".mat") { f.to_string() } else { format!("{f}.mat") });
    let vars = crate::matfile::read(&p)?;
    vars.get("xparam1").and_then(|v| v.as_vec()).ok_or_else(|| anyhow!("{}: no xparam1", p.display()))
}

/// The `Log Prior` / `Log Likelihood` / `Log Posterior` CDF figures of monte_carlo_filtering.m.
#[allow(clippy::too_many_arguments)]
fn lnlik_figures(
    ctx: &Ctx,
    tag: &str,
    vnames: &[String],
    ixx: &[Vec<usize>],
    nfilt: usize,
    lnprior: &[f64],
    lik: &[f64],
    logpo: &[f64],
) {
    let tag = match tag {
        "post" => ("Posterior", "post"),
        "prior" => ("Prior", "prior"),
        _ => ("MC", "mc"),
    };
    for (what, stem, v) in
        [("Log Prior", "lnprior", lnprior), ("Log Likelihood", "lnlik", lik), ("Log Posterior", "lnpost", logpo)]
    {
        for (ifig, chunk) in (0..vnames.len()).collect::<Vec<_>>().chunks(9).enumerate() {
            let panels = chunk
                .iter()
                .map(|&i| {
                    let best: Vec<f64> = ixx[i][..nfilt].iter().map(|&k| v[k]).collect();
                    let rest: Vec<f64> = ixx[i][nfilt..].iter().map(|&k| v[k]).collect();
                    Panel {
                        title: vnames[i].clone(),
                        series: vec![
                            cdf_series(
                                &best,
                                Color::Blue,
                                LineStyle::Solid,
                                2.0,
                                Some(format!("{what} for BETTER R2")),
                            ),
                            cdf_series(v, Color::Black, LineStyle::Solid, 1.0, None),
                            cdf_series(&rest, Color::Red, LineStyle::Solid, 2.0, Some(format!("{what} for WORSE R2"))),
                        ],
                        ..Default::default()
                    }
                })
                .collect();
            ctx.save(&Figure {
                name: format!("rmse_{}_{stem}{}", tag.1, ifig + 1),
                title: format!("RMSE {}: {what} {}", tag.0, ifig + 1),
                rows: 3,
                cols: 3,
                panels,
            });
        }
    }
}
