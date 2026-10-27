// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `+gsa/stability_mapping.m`: solve the model at every point of the sample (in
//! parallel), classify each point as unique stable solution / explosive / indeterminate /
//! no solution found, and run the Monte Carlo filtering of each class against the rest.

use anyhow::Result;
use rayon::prelude::*;

use super::mcf::{McfSpec, analysis};
use super::sampling::{Sample, SamplingSpec, draw_sample};
use super::stats::fmt_f;
use super::{Ctx, GsaOptions};
use crate::dsge::DsgeModel;
use crate::linalg::Mat;
use crate::solver::DrStructure;

/// The saved state of a stability map (Dynare's `<fname>_prior.mat` / `_mc.mat`), 0-based indices.
#[derive(Clone, Debug, Default)]
pub struct StabMap {
    /// Shock parameters (stderr / corr), `nsam x nshock`.
    pub lpmat0: Sample,
    /// Structural parameters, `nsam x np`.
    pub lpmat: Sample,
    /// `infox`: Dynare's info code at each point (0 = unique stable solution).
    pub infox: Vec<i32>,
    pub istable: Vec<usize>,
    /// Everything that is not stable (`iunstable`).
    pub iunstable: Vec<usize>,
    pub iindeterm: Vec<usize>,
    pub iwrong: Vec<usize>,
    /// Explosive (`ixun` = iunstable minus iindeterm, iwrong and inorestriction).
    pub ixun: Vec<usize>,
    // ---- calibration restrictions ----
    /// Stable points that satisfy the `irf_calibration` / `moment_calibration` restrictions
    /// (`irestriction`; = istable without restrictions).
    pub irestriction: Vec<usize>,
    /// Stable points that violate them (`inorestriction`, `infox` = 49; also in iunstable).
    pub inorestriction: Vec<usize>,
    // ---- end calibration restrictions ----
    /// `[ghx ghu]` (DR order) at each stable point, when the reduced form is needed (`T`).
    pub t: Vec<Mat>,
    /// Steady state at each stable point.
    pub ys: Vec<Vec<f64>>,
    pub nspred: usize,
    pub nboth: usize,
    pub nfwrd: usize,
    /// The point returned by `stability_mapping` (`x0`, xparam1 order), if any.
    pub x0: Option<Vec<f64>>,
}

impl StabMap {
    /// `[lpmat0(k,:) lpmat(k,:)]`, the full xparam1 vector of point k.
    pub fn point(&self, k: usize) -> Vec<f64> {
        self.lpmat0[k].iter().chain(&self.lpmat[k]).copied().collect()
    }
}

/// The info codes stability_mapping.m explains when Dynare could not find a solution.
const WRONG_MESSAGES: &[(i32, &str)] = &[
    (1, "The model doesn't determine the current variables uniquely."),
    (2, "MJDGGES returned an error code."),
    (6, "The Jacobian evaluated at the deterministic steady state is complex."),
    (19, "The steadystate routine has thrown an exception (inconsistent deep parameters)."),
    (20, "Cannot find the steady state."),
    (21, "The steady state is complex."),
    (22, "The steady has NaNs."),
    (23, "M_.params has been updated in the steadystate routine and has complex valued scalars."),
    (24, "M_.params has been updated in the steadystate routine and has some NaNs."),
    (30, "Ergodic variance can't be computed."),
];

/// Classification of one solved point (stability_mapping.m's `egg` test, which Dynare's
/// info codes determine: 4 = more stable roots than predetermined variables -> indeterminacy,
/// 3/5 -> explosive, anything without eigenvalues -> no solution).
fn classify(info: i32) -> u8 {
    match info {
        0 => 0,
        3 | 5 => 1,
        4 => 2,
        _ => 3,
    }
}

/// Solves the model on a given sample (`lpmat0`, `lpmat`) and classifies the points.
pub fn classify_sample(dsge: &DsgeModel, lpmat0: Sample, lpmat: Sample, keep_t: bool) -> StabMap {
    classify_sample_restricted(dsge, lpmat0, lpmat, keep_t, None)
}

/// [`classify_sample`] with `options_.endogenous_prior_restrictions`: stability_mapping.m checks
/// them at every stable point (`infox` = 49 -> `inorestriction`, else `irestriction`).
pub fn classify_sample_restricted(
    dsge: &DsgeModel,
    lpmat0: Sample,
    lpmat: Sample,
    keep_t: bool,
    rs: Option<&crate::gsa::calibration::Restrictions>,
) -> StabMap {
    let nsam = lpmat.len().max(lpmat0.len());
    let solved: Vec<(i32, Option<(Mat, Vec<f64>)>)> = (0..nsam)
        .into_par_iter()
        .map(|k| {
            let x: Vec<f64> = lpmat0[k].iter().chain(&lpmat[k]).copied().collect();
            match dsge.resolve(&x) {
                Ok(res) => {
                    let t = if keep_t {
                        let (n, a, b) = (res.dr.ghx.nrows(), res.dr.ghx.ncols(), res.dr.ghu.ncols());
                        let mut t = Mat::zeros(n, a + b);
                        t.columns_mut(0, a).copy_from(&res.dr.ghx);
                        t.columns_mut(a, b).copy_from(&res.dr.ghu);
                        t
                    } else {
                        Mat::zeros(0, 0)
                    };
                    // ---- calibration restrictions ----
                    let info =
                        rs.map_or(0, |rs| crate::gsa::calibration::restrictions_info(dsge, &res.dr, &res.q, rs).0);
                    (info, Some((t, res.steady_state)))
                }
                Err(f) => (f.info, None),
            }
        })
        .collect();
    let s = DrStructure::new(&dsge.model.lead_lag_incidence, dsge.model.n_exo());
    let mut m = StabMap { nspred: s.nspred, nboth: s.nboth, nfwrd: s.nfwrd, ..Default::default() };
    for (k, (info, sol)) in solved.into_iter().enumerate() {
        m.infox.push(info);
        if let Some((t, ys)) = sol {
            m.istable.push(k);
            if keep_t {
                m.t.push(t);
            }
            m.ys.push(ys);
            if info == 0 {
                m.irestriction.push(k);
            } else {
                m.iunstable.push(k);
                m.inorestriction.push(k);
            }
            continue;
        }
        match classify(info) {
            0 => unreachable!("info 0 always comes with a solution"),
            c => {
                m.iunstable.push(k);
                match c {
                    1 => m.ixun.push(k),
                    2 => m.iindeterm.push(k),
                    _ => m.iwrong.push(k),
                }
            }
        }
    }
    m.lpmat0 = lpmat0;
    m.lpmat = lpmat;
    m
}

/// `stability_mapping.m`: sample, solve, classify, print and filter.
pub fn stability_mapping(
    ctx: &Ctx,
    dsge: &DsgeModel,
    o: &GsaOptions,
    rng: &mut super::rng::MatlabRng,
    rs: Option<&crate::gsa::calibration::Restrictions>,
) -> Result<StabMap> {
    let nshock = ctx.nshock;
    let np = dsge.n_estimated() - nshock;
    let spec = SamplingSpec {
        nsam: o.nsam,
        nshock,
        np,
        ilptau: o.ilptau,
        morris: o.morris == 1,
        morris_nliv: o.morris_nliv,
        morris_ntra: o.morris_ntra,
        pprior: o.pprior,
        prior_range: o.prior_range,
        bounds: &ctx.bounds,
        priors: &dsge.priors,
        neighborhood: (o.neighborhood_width > 0.0).then_some((o.neighborhood_width, ctx.xparam_calib.as_slice())),
        posterior: ctx.posterior_mode.as_ref().map(|(x, hh)| (x.as_slice(), hh)),
    };
    let (lpmat0, lpmat) = draw_sample(&spec, rng)?;
    let keep_t = o.redform || o.identification;
    let mut m = classify_sample_restricted(dsge, lpmat0, lpmat, keep_t, rs);
    report(ctx, o, &mut m);
    Ok(m)
}

/// The printed part of stability_mapping.m and its Monte Carlo filtering.
pub fn report(ctx: &Ctx, o: &GsaOptions, m: &mut StabMap) {
    let nsam = m.infox.len();
    let nshock = ctx.nshock;
    let pct = |k: usize| k as f64 / nsam as f64 * 100.0;
    let (prefix, title_prefix) =
        if o.pprior { ("prior", "Prior StabMap") } else { ("mc", "MC (around posterior mode) StabMap") };
    let mut s = String::from("\nSensitivity Analysis: Stability mapping:\n");
    let names: Vec<String> = ctx.param_names[nshock..].to_vec();
    if !m.iunstable.is_empty() || !m.iwrong.is_empty() {
        s += &format!(
            "{}% of the prior support gives unique saddle-path solution.\n",
            fmt_f(pct(m.istable.len()), 4, 1)
        );
        s += &format!("{}% of the prior support gives explosive dynamics.\n", fmt_f(pct(m.ixun.len()), 4, 1));
        if !m.iindeterm.is_empty() {
            s += &format!("{}% of the prior support gives indeterminacy.\n", fmt_f(pct(m.iindeterm.len()), 4, 1));
        }
        // ---- calibration restrictions: inorestriction ----
        if !m.iwrong.is_empty() || !m.inorestriction.is_empty() {
            s.push('\n');
            if m.infox.contains(&49) {
                s += &format!(
                    "{}% of the prior support violates prior restrictions.\n",
                    fmt_f(pct(m.inorestriction.len()), 4, 1)
                );
            }
            if !m.iwrong.is_empty() {
                s.push('\n');
                s += &format!(
                    "For {}% of the prior support Dynare could not find a solution.\n\n",
                    fmt_f(pct(m.iwrong.len()), 4, 1).trim()
                );
            }
            for (c, msg) in WRONG_MESSAGES {
                let n = m.infox.iter().filter(|&&i| i == *c).count();
                if n > 0 {
                    s += &format!("    For {}% {msg}\n", fmt_f(pct(n), 4, 1).trim());
                }
            }
        }
        s.push('\n');
        print!("{s}");
        if m.iunstable.len() < nsam || m.istable.len() > 1 {
            let itot: Vec<usize> = (0..nsam).collect();
            let isolve: Vec<usize> = itot.iter().copied().filter(|k| !m.iwrong.contains(k)).collect();
            let xparam1 = (o.neighborhood_width > 0.0).then(|| ctx.xparam_calib[nshock..].to_vec());
            let mk = |name: &str, title: &str, beha: &str, nobeha: &str, t: &str| McfSpec {
                pvalue_ks: o.pvalue_ks,
                pvalue_corr: o.pvalue_corr,
                alpha2: o.alpha2_stab,
                param_names: &names,
                amcf_name: format!("{prefix}_{name}"),
                amcf_title: format!("{title_prefix}: {title}"),
                beha_title: beha.into(),
                nobeha_title: nobeha.into(),
                title: t.into(),
                xparam1: xparam1.clone(),
                noprint: true,
            };
            let not_in =
                |set: &[usize], ex: &[usize]| set.iter().copied().filter(|k| !ex.contains(k)).collect::<Vec<_>>();
            let itmp = not_in(&itot, &m.istable);
            analysis(
                ctx,
                &m.lpmat,
                &m.istable,
                &itmp,
                &mk(
                    "stable",
                    "Parameter driving non-existence of unique stable solution (Unacceptable)",
                    "unique Stable Saddle-Path",
                    "NO unique Stable Saddle-Path",
                    "unique solution",
                ),
            );
            if !m.iindeterm.is_empty() {
                let itmp = not_in(&isolve, &m.iindeterm);
                analysis(
                    ctx,
                    &m.lpmat,
                    &itmp,
                    &m.iindeterm,
                    &mk("indeterm", "Indeterminacy", "NO indeterminacy", "indeterminacy", "indeterminacy"),
                );
            }
            if !m.ixun.is_empty() {
                let itmp = not_in(&isolve, &m.ixun);
                analysis(
                    ctx,
                    &m.lpmat,
                    &itmp,
                    &m.ixun,
                    &mk(
                        "unstable",
                        "explosiveness of solution",
                        "NO explosive solution",
                        "explosive solution",
                        "instability",
                    ),
                );
            }
            if !m.iwrong.is_empty() {
                let itmp = not_in(&itot, &m.iwrong);
                analysis(
                    ctx,
                    &m.lpmat,
                    &itmp,
                    &m.iwrong,
                    &mk(
                        "wrong",
                        "inability to find solution",
                        "NO inability to find a solution",
                        "inability to find a solution",
                        "inability to find a solution",
                    ),
                );
            }
            // ---- calibration restrictions: the 'prior_calib' filtering (without restrictions
            // irestriction = istable, inorestriction is empty and it prints nothing) ----
            if !m.irestriction.is_empty() {
                let x: Sample = (0..nsam).map(|k| m.point(k)).collect();
                let spec = McfSpec {
                    pvalue_ks: o.pvalue_ks,
                    pvalue_corr: o.pvalue_corr,
                    alpha2: o.alpha2_stab,
                    param_names: &ctx.param_names,
                    amcf_name: format!("{prefix}_calib"),
                    amcf_title: format!("{title_prefix}: IRF/moment restrictions"),
                    beha_title: "prior IRF/moment calibration".into(),
                    nobeha_title: "NO prior IRF/moment calibration".into(),
                    title: "prior restrictions".into(),
                    xparam1: (o.neighborhood_width > 0.0).then(|| ctx.xparam_calib.clone()),
                    noprint: true,
                };
                if !m.inorestriction.is_empty() {
                    analysis(ctx, &x, &m.irestriction, &m.inorestriction, &spec);
                }
                m.x0 = Some(m.point(m.irestriction[0]));
            } else {
                let mut x0: Vec<f64> =
                    (0..nshock).map(|j| 0.5 * (ctx.bounds.ub[j] - ctx.bounds.lb[j]) + ctx.bounds.lb[j]).collect();
                x0.extend(&m.lpmat[m.istable[0]]);
                m.x0 = Some(x0);
            }
        } else {
            println!("All parameter values in the specified ranges are not acceptable!");
            m.x0 = None;
        }
    } else {
        s += "All parameter values in the specified ranges give unique saddle-path solution,\n";
        s += "and match prior IRF/moment restriction(s) if any!\n";
        print!("{s}");
        let mut x0: Vec<f64> =
            (0..nshock).map(|j| 0.5 * (ctx.bounds.ub[j] - ctx.bounds.lb[j]) + ctx.bounds.lb[j]).collect();
        x0.extend(&m.lpmat[m.istable[0]]);
        m.x0 = Some(x0);
    }
    println!();
}

fn to_mat(rows: &Sample, ncols: usize) -> Mat {
    Mat::from_fn(rows.len(), ncols, |i, j| rows[i][j])
}

fn one_based(v: &[usize]) -> Vec<f64> {
    v.iter().map(|&k| (k + 1) as f64).collect()
}

fn file_name(ctx: &Ctx, o: &GsaOptions) -> std::path::PathBuf {
    ctx.out_dir.join(format!("{}_{}.mat", ctx.fname, if o.pprior { "prior" } else { "mc" }))
}

/// Writes `<fname>_prior.mat` (or `_mc.mat`) with Dynare's variable names (1-based indices;
/// `T` is not written: a 3-D array, recomputed on load as Dynare does when it is missing).
pub fn save(ctx: &Ctx, o: &GsaOptions, m: &StabMap) -> Result<()> {
    use crate::matfile::{MatOut, write};
    let path = file_name(ctx, o);
    let nshock = m.lpmat0.first().map_or(0, Vec::len);
    let np = m.lpmat.first().map_or(0, Vec::len);
    let (l0, l) = (to_mat(&m.lpmat0, nshock), to_mat(&m.lpmat, np));
    let infox: Vec<f64> = m.infox.iter().map(|&i| i as f64).collect();
    let (is, iu, ii, iw, ix) = (
        one_based(&m.istable),
        one_based(&m.iunstable),
        one_based(&m.iindeterm),
        one_based(&m.iwrong),
        one_based(&m.ixun),
    );
    write(
        &path,
        &[
            ("lpmat", MatOut::Matrix(&l)),
            ("lpmat0", MatOut::Matrix(&l0)),
            ("istable", MatOut::Vector(&is)),
            ("irestriction", MatOut::Vector(&one_based(&m.irestriction))),
            ("iunstable", MatOut::Vector(&iu)),
            ("iindeterm", MatOut::Vector(&ii)),
            ("iwrong", MatOut::Vector(&iw)),
            ("ixun", MatOut::Vector(&ix)),
            ("infox", MatOut::Vector(&infox)),
            ("nspred", MatOut::Scalar(m.nspred as f64)),
            ("nboth", MatOut::Scalar(m.nboth as f64)),
            ("nfwrd", MatOut::Scalar(m.nfwrd as f64)),
        ],
    )
}

/// `load_stab=1`: the sample of a previous run (dynare-rust's or Dynare's file), re-solved.
pub fn load(
    ctx: &Ctx,
    dsge: &DsgeModel,
    o: &GsaOptions,
    rs: Option<&crate::gsa::calibration::Restrictions>,
) -> Result<StabMap> {
    let path = file_name(ctx, o);
    let vars = crate::matfile::read(&path).map_err(|e| {
        anyhow::anyhow!("{} not found! ({e})\nYou asked to load a non existent analysis", path.display())
    })?;
    let get = |k: &str| vars.get(k).and_then(|v| v.as_matrix());
    let l = get("lpmat").ok_or_else(|| anyhow::anyhow!("{}: no lpmat", path.display()))?;
    let rows = |m: &Mat| -> Sample { (0..m.nrows()).map(|i| m.row(i).iter().copied().collect()).collect() };
    let lpmat = rows(&l);
    let lpmat0 = match get("lpmat0") {
        Some(m) if m.nrows() == l.nrows() => rows(&m),
        _ => vec![Vec::new(); l.nrows()],
    };
    let mut m = classify_sample_restricted(dsge, lpmat0, lpmat, o.redform || o.identification, rs);
    report(ctx, o, &mut m);
    Ok(m)
}
