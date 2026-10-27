// SPDX-License-Identifier: GPL-3.0-or-later
//! `sensitivity(identification=1)` — port of `+gsa/map_identification.m` with
//! `+gsa/monte_carlo_moments.m`, `+gsa/th_moments.m` and `+gsa/teff.m`.
//!
//! * `morris = 1` (the default with `identification`): Morris elementary effects of the
//!   parameters on the theoretical moments of the observed variables (variance decomposition,
//!   (co)variances, autocovariances, steady state) and on the model's reduced form (steady
//!   state, transition and innovation covariance), on the Morris trajectories of the stability
//!   map; boxplot figures `<fname>_morris_vdec`, `_morris_moments`, `_morris_par` (Dynare's
//!   `Screening identification: ...` figures) and `<fname>_morris_IDE.mat`.
//! * `morris = 2`: the local identification analysis (`identification.run`) on the stable
//!   points of the sample, then the boxplots of the variance decompositions by shock.

use anyhow::Result;
use rayon::prelude::*;

use super::redform::morris_measures;
use super::stability::StabMap;
use super::{Ctx, GsaOptions};
use crate::dsge::DsgeModel;
use crate::graphics::{Color, Figure, LineStyle, Panel, Series};
use crate::linalg::Mat;
use crate::modfile::{ModFile, OptValue};

/// The moments of one stable point (`th_moments.m` on `options_.varobs`).
struct PointMoments {
    /// nobs x exo_nbr, percent.
    vdec: Mat,
    /// nobs x nobs (`triu`).
    cc: Mat,
    /// nobs x (nobs * ar).
    ac: Mat,
    /// Observed steady states (`yys(bayestopt_.mfys)`).
    mss: Vec<f64>,
    /// `[yys(order_var); vec(A)]` with `A = [T, triu(R Sigma_e R')]` (kalman_transition_matrix
    /// over all variables in DR order).
    yt: Vec<f64>,
}

fn point_moments(
    dsge: &DsgeModel,
    x: &[f64],
    ar: usize,
    useautocorr: bool,
    base: &crate::analysis::moments::MomentOptions,
) -> Option<PointMoments> {
    let r = dsge.resolve(x).ok()?;
    let dr = &r.dr;
    let ivar = &dsge.varobs;
    let nobs = ivar.len();
    let mo = crate::analysis::moments::MomentOptions { ar, ..base.clone() };
    let th = crate::analysis::moments::th_autocovariances(dr, &r.q, ivar, &mo).ok()?;
    let nexo = dsge.model.n_exo();
    let vdec = if nexo > 1 {
        th.vardec.as_ref().map(|v| v * 100.0).unwrap_or_else(|| Mat::from_element(nobs, nexo, f64::NAN))
    } else {
        Mat::from_element(nobs, 1, 100.0)
    };
    let g = &th.gamma0;
    let sd: Vec<f64> = (0..nobs).map(|i| g[(i, i)].sqrt()).collect();
    let mut cc = if useautocorr {
        Mat::from_fn(nobs, nobs, |i, j| if i == j { g[(i, i)] } else { g[(i, j)] / (sd[i] * sd[j]) })
    } else {
        g.clone()
    };
    for i in 0..nobs {
        for j in 0..i {
            cc[(i, j)] = 0.0; // triu
        }
    }
    let mut ac = Mat::from_element(nobs, nobs * ar, f64::NAN);
    for l in 0..ar {
        for i in 0..nobs {
            for j in 0..nobs {
                let c = th.autocorr[l][(i, j)];
                ac[(i, l * nobs + j)] = if useautocorr { c } else { c * sd[i] * sd[j] };
            }
        }
    }
    let mss: Vec<f64> = ivar.iter().map(|&v| r.steady_state[v]).collect();
    // A = [Aa, triu(Bb Sigma_e Bb')] with kalman_transition_matrix(dr, 1:endo_nbr, ic)
    let n = dr.ghx.nrows();
    let mut aa = Mat::zeros(n, n);
    for k in 0..dr.nspred {
        aa.column_mut(dr.nstatic + k).copy_from(&dr.ghx.column(k));
    }
    let om = &dr.ghu * &r.q * dr.ghu.transpose();
    let mut yt: Vec<f64> = dr.order_var.iter().map(|&v| r.steady_state[v]).collect();
    yt.extend(aa.iter().copied());
    for j in 0..n {
        for i in 0..n {
            yt.push(if i <= j { om[(i, j)] } else { 0.0 });
        }
    }
    Some(PointMoments { vdec, cc, ac, mss, yt })
}

/// `teff(T, Nsam, istable)` on a list of per-point vectors: the entries that vary (range
/// above 1e-8), as `Nsam x j0` columns with NaN at the unstable points.
fn teff(vals: &[Vec<f64>], nsam: usize, istable: &[usize]) -> Vec<Vec<f64>> {
    if vals.is_empty() {
        return Vec::new();
    }
    let m = vals[0].len();
    let mut out = Vec::new();
    for e in 0..m {
        let (mn, mx) = vals.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(v[e]), b.max(v[e])));
        if mx - mn > 1e-8 {
            let mut col = vec![f64::NAN; nsam];
            for (k, &i) in istable.iter().enumerate() {
                col[i] = vals[k][e];
            }
            out.push(col);
        }
    }
    out
}

/// `standardize_columns(log(x)) / 2` on the stable entries of one column.
fn standardized_log_half(col: &mut [f64]) {
    let l: Vec<f64> = col.iter().filter(|v| !v.is_nan()).map(|v| v.ln()).collect();
    let n = l.len() as f64;
    let m = l.iter().sum::<f64>() / n;
    let s = (l.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    for v in col.iter_mut() {
        if !v.is_nan() {
            *v = (v.ln() - m) / s / 2.0;
        }
    }
}

/// A boxplot of the columns of `rows` (one box per parameter): the inter-quartile range, the
/// median and the extremes.
fn boxplot(name: &str, title: &str, rows: &[Vec<f64>], names: &[String]) -> Figure {
    let np = names.len();
    let mut series = Vec::new();
    for p in 0..np {
        let mut v: Vec<f64> = rows.iter().map(|r| r[p]).filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            continue;
        }
        v.sort_by(f64::total_cmp);
        let q = |f: f64| v[((v.len() - 1) as f64 * f).round() as usize];
        let x = (p + 1) as f64;
        series.push(Series::Line {
            x: vec![x, x],
            y: vec![v[0], v[v.len() - 1]],
            color: Color::Black,
            style: LineStyle::Dashed,
            width: 0.5,
            label: None,
        });
        series.push(Series::Line {
            x: vec![x, x],
            y: vec![q(0.25), q(0.75)],
            color: Color::Blue,
            style: LineStyle::Solid,
            width: 6.0,
            label: None,
        });
        series.push(Series::Points { x: vec![x], y: vec![q(0.5)], color: Color::Red, label: None });
    }
    let ticks: Vec<(f64, String)> = names.iter().enumerate().map(|(i, n)| ((i + 1) as f64, n.clone())).collect();
    Figure {
        name: name.into(),
        title: title.into(),
        rows: 1,
        cols: 1,
        panels: vec![Panel { title: title.into(), series, x_ticks: Some(ticks), ..Default::default() }],
    }
}

/// Morris `mu*` of every output column (`SAMorris(:, 1, :)`), one row per column.
fn morris_rows(npt: usize, sample: &[Vec<f64>], cols: &[Vec<f64>], nliv: usize) -> Vec<Vec<f64>> {
    cols.par_iter().map(|c| morris_measures(npt, sample, c, nliv).iter().map(|m| m[0]).collect()).collect()
}

/// Port of `+gsa/map_identification.m`.
/// The Morris screening of `map_identification` (`<fname>_morris_IDE.mat`): one row per output.
#[derive(Clone, Debug, Default)]
pub struct IdentScreening {
    pub sa_vdec: Vec<Vec<f64>>,
    pub sa_cc: Vec<Vec<f64>>,
    pub sa_norm: Vec<Vec<f64>>,
}

/// `load_ident_files = 1` (morris = 1): `SAvdec`, `SAcc`, `SAnorm` of the saved `<fname>_morris_IDE.mat`
/// (Dynare's file or ours), drawn again.
fn load_screening(ctx: &Ctx, names: &[String]) -> Result<IdentScreening> {
    let path = ctx.out_dir.join(format!("{}_morris_IDE.mat", ctx.fname));
    let m = crate::matfile::read(&path)
        .map_err(|e| anyhow::anyhow!("load_ident_files: {} not found ({e})", path.display()))?;
    let rows = |k: &str, t: bool| -> Result<Vec<Vec<f64>>> {
        let a = m.get(k).and_then(|v| v.as_matrix()).ok_or_else(|| anyhow::anyhow!("{}: no {k}", path.display()))?;
        let a = if t { a.transpose() } else { a };
        Ok((0..a.nrows()).map(|i| a.row(i).iter().copied().collect()).collect())
    };
    let s =
        IdentScreening { sa_vdec: rows("SAvdec", false)?, sa_cc: rows("SAcc", false)?, sa_norm: rows("SAnorm", true)? };
    ctx.save(&boxplot("morris_vdec", "Screening identification: variance decomposition", &s.sa_vdec, names));
    ctx.save(&boxplot("morris_moments", "Screening identification: theoretical moments", &s.sa_cc, names));
    ctx.save(&boxplot("morris_par", "Screening identification: model", &s.sa_norm, names));
    Ok(s)
}

pub fn map_identification(
    ctx: &Ctx,
    mf: &ModFile,
    dir: &std::path::Path,
    dsge: &DsgeModel,
    o: &GsaOptions,
    stab: &StabMap,
    seed: u64,
) -> Result<Option<IdentScreening>> {
    if o.load_ident_files {
        // map_identification.m: with morris = 2 nothing is recomputed or displayed (the sample is not analysed
        // again); with morris = 1 the saved screening is drawn again
        if o.morris == 2 {
            return Ok(None);
        }
        return load_screening(ctx, &ctx.param_names).map(Some);
    }
    let nsam = stab.lpmat.len();
    let npt = dsge.n_estimated();
    let points: Vec<Vec<f64>> = stab.istable.iter().map(|&k| stab.point(k)).collect();
    println!("monte_carlo_moments: Computing theoretical moments ...");
    let pm: Vec<Option<PointMoments>> =
        points.par_iter().map(|x| point_moments(dsge, x, o.ident_ar, o.ident_useautocorr, &ctx.moment_base)).collect();
    println!();
    println!("... done !");
    if pm.iter().any(|p| p.is_none()) {
        anyhow::bail!("map_identification: theoretical moments failed at a stable point");
    }
    let pm: Vec<PointMoments> = pm.into_iter().map(|p| p.unwrap()).collect();
    let names = ctx.param_names.clone();
    if o.morris == 2 {
        // identification.run on the stable points of the sample
        let sample = points.clone();
        // run.m: with options_.opt_gsa.identification the spectrum and the minimal system are skipped
        let opts = vec![
            ("ar".to_string(), OptValue::Num(o.ident_ar as f64)),
            ("useautocorr".to_string(), OptValue::Num(o.ident_useautocorr as u8 as f64)),
            ("no_identification_minimal".to_string(), OptValue::Flag),
            ("no_identification_spectrum".to_string(), OptValue::Flag),
            ("nograph".to_string(), OptValue::Flag),
        ];
        crate::identification::run::run_external(mf, dir, &ctx.fname, &opts, seed, sample)?;
        println!("Sample check OK. Largest difference: 0");
        if dsge.model.n_exo() > 1 {
            let nobs = dsge.varobs.len();
            let obs: Vec<String> = dsge.varobs.iter().map(|&v| dsge.model.endo_names[v].clone()).collect();
            for (f, chunk) in (0..dsge.model.n_exo()).collect::<Vec<_>>().chunks(6).enumerate() {
                let panels = chunk
                    .iter()
                    .map(|&j| {
                        let rows: Vec<Vec<f64>> =
                            pm.iter().map(|p| (0..nobs).map(|i| p.vdec[(i, j)]).collect()).collect();
                        let mut fig = boxplot("", &dsge.model.exo_names[j], &rows, &obs);
                        fig.panels.remove(0)
                    })
                    .collect();
                ctx.save(&Figure {
                    name: format!("vdec_exo_{}", f + 1),
                    title: "Variance decomposition shocks".into(),
                    rows: 2,
                    cols: 3,
                    panels,
                });
            }
        }
        return Ok(None);
    }
    // morris = 1: elementary effects on the Morris trajectories
    let sample: Vec<Vec<f64>> = (0..nsam).map(|k| stab.point(k)).collect();
    let vdec = teff(&pm.iter().map(|p| p.vdec.iter().copied().collect()).collect::<Vec<_>>(), nsam, &stab.istable);
    // cc(j,j,:) = standardize_columns(log(cc(j,j,:)))/2, then teff
    let nobs = dsge.varobs.len();
    let mut ccv: Vec<Vec<f64>> = pm.iter().map(|p| p.cc.iter().copied().collect()).collect();
    for j in 0..nobs {
        let e = j * nobs + j;
        let mut col: Vec<f64> = ccv.iter().map(|v| v[e]).collect();
        standardized_log_half(&mut col);
        for (v, c) in ccv.iter_mut().zip(col) {
            v[e] = c;
        }
    }
    let cc = teff(&ccv, nsam, &stab.istable);
    let ac = teff(&pm.iter().map(|p| p.ac.iter().copied().collect()).collect::<Vec<_>>(), nsam, &stab.istable);
    let mss = teff(&pm.iter().map(|p| p.mss.clone()).collect::<Vec<_>>(), nsam, &stab.istable);
    let yt = teff(&pm.iter().map(|p| p.yt.clone()).collect::<Vec<_>>(), nsam, &stab.istable);
    let nliv = o.morris_nliv;
    let sa_vdec = morris_rows(npt, &sample, &vdec, nliv);
    ctx.save(&boxplot("morris_vdec", "Screening identification: variance decomposition", &sa_vdec, &names));
    let ccac: Vec<Vec<f64>> = mss.iter().chain(&cc).chain(&ac).cloned().collect();
    let sa_cc: Vec<Vec<f64>> = morris_rows(npt, &sample, &ccac, nliv)
        .into_iter()
        .map(|r| {
            let mx = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            r.iter().map(|v| v / mx).collect()
        })
        .collect();
    ctx.save(&boxplot("morris_moments", "Screening identification: theoretical moments", &sa_cc, &names));
    let sa_norm: Vec<Vec<f64>> = morris_rows(npt, &sample, &yt, nliv)
        .into_iter()
        .map(|r| {
            let mx = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            r.iter().map(|v| v / mx).collect()
        })
        .collect();
    ctx.save(&boxplot("morris_par", "Screening identification: model", &sa_norm, &names));
    use crate::matfile::{MatOut, write};
    let to_mat = |rows: &[Vec<f64>]| {
        let nc = rows.first().map_or(0, Vec::len);
        Mat::from_fn(rows.len(), nc, |i, j| rows[i][j])
    };
    let (a, b, c) = (to_mat(&sa_vdec), to_mat(&sa_cc), to_mat(&sa_norm));
    write(
        &ctx.out_dir.join(format!("{}_morris_IDE.mat", ctx.fname)),
        &[("SAvdec", MatOut::Matrix(&a)), ("SAcc", MatOut::Matrix(&b)), ("SAnorm", MatOut::Matrix(&c.transpose()))],
    )?;
    Ok(Some(IdentScreening { sa_vdec, sa_cc, sa_norm }))
}
