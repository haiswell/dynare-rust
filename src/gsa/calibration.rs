// SPDX-License-Identifier: GPL-3.0-or-later
//! `irf_calibration` / `moment_calibration` (`options_.endogenous_prior_restrictions`)
//! and their GSA mapping: ports of `endogenous_prior_restrictions.m` and
//! `+gsa/map_calibration.m`.
//!
//! The parser keeps each block entry as raw text ([`crate::modfile::BLOCK_BODY_KEY`]); the
//! entries follow the manual's syntax: `VAR(periods), SHOCK, [lb, ub]` (or `+` / `-`) for IRF
//! restrictions, `VAR1, VAR2(lags), [lb, ub]` for moment restrictions.

use anyhow::{Result, anyhow, bail};
use rayon::prelude::*;

use super::mcf::{self, McfSpec};
use super::stability::StabMap;
use super::{Ctx, GsaOptions};
use crate::dsge::DsgeModel;
use crate::graphics::{Color, Figure, LineStyle, Panel, Series};
use crate::linalg::Mat;
use crate::modfile::{BLOCK_BODY_KEY, Command, ModFile, OptValue, opt};
use crate::solver::DecisionRule;

/// One row of `options_.endogenous_prior_restrictions.irf` / `.moment`: `{var1, var2,
/// periods, [lb ub]}`.
#[derive(Clone, Debug, PartialEq)]
pub struct Restriction {
    pub v1: String,
    pub v2: String,
    pub periods: Vec<i64>,
    pub bounds: [f64; 2],
}

/// `options_.endogenous_prior_restrictions` (+ `relative_irf` of `irf_calibration(...)`).
#[derive(Clone, Debug, Default)]
pub struct Restrictions {
    pub irf: Vec<Restriction>,
    pub moment: Vec<Restriction>,
    pub relative_irf: bool,
    /// The `options_` that `th_autocovariances` sees (`hp_filter`, `bandpass`, ... left by earlier
    /// `stoch_simul` / `estimation` commands), see [`persistent_moment_options`].
    pub moment_options: crate::analysis::moments::MomentOptions,
}

impl Restrictions {
    pub fn is_empty(&self) -> bool {
        self.irf.is_empty() && self.moment.is_empty()
    }
}

fn split_top(s: &str) -> Vec<String> {
    let mut out = vec![String::new()];
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            _ => {}
        }
        if c == ',' && depth == 0 {
            out.push(String::new());
        } else {
            out.last_mut().unwrap().push(c);
        }
    }
    out.into_iter().map(|p| p.trim().to_string()).collect()
}

/// `1`, `-1`, `1:4`, `-(1:4)`, `(-2:2)`, `a:s:b`.
fn periods(e: &str) -> Result<Vec<i64>> {
    let e: String = e.chars().filter(|c| !c.is_whitespace()).collect();
    if let Some(inner) = e.strip_prefix("-(").and_then(|x| x.strip_suffix(')')) {
        return Ok(periods(inner)?.into_iter().map(|v| -v).collect());
    }
    if let Some(inner) = e.strip_prefix("+(").and_then(|x| x.strip_suffix(')')) {
        return periods(inner);
    }
    if let Some(inner) = e.strip_prefix('(').and_then(|x| x.strip_suffix(')')) {
        return periods(inner);
    }
    let parts: Vec<i64> = e
        .split(':')
        .map(|p| p.parse::<i64>().map_err(|_| anyhow!("calibration: bad period `{e}`")))
        .collect::<Result<_>>()?;
    Ok(match parts.as_slice() {
        [a] => vec![*a],
        [a, b] => (*a..=*b).collect(),
        [a, s, b] if *s != 0 => {
            let mut v = Vec::new();
            let mut x = *a;
            while (*s > 0 && x <= *b) || (*s < 0 && x >= *b) {
                v.push(x);
                x += s;
            }
            v
        }
        _ => bail!("calibration: bad period `{e}`"),
    })
}

/// `name` or `name(periods)`.
fn name_and_periods(s: &str, default: Option<Vec<i64>>) -> Result<(String, Vec<i64>)> {
    match s.find('(') {
        Some(k) if s.ends_with(')') => Ok((s[..k].trim().to_string(), periods(&s[k + 1..s.len() - 1])?)),
        _ => Ok((s.trim().to_string(), default.ok_or_else(|| anyhow!("calibration: `{s}` needs periods"))?)),
    }
}

fn value(s: &str) -> Result<f64> {
    let t: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    match t.to_ascii_lowercase().as_str() {
        "inf" | "+inf" => Ok(f64::INFINITY),
        "-inf" => Ok(f64::NEG_INFINITY),
        _ => t.parse::<f64>().map_err(|_| anyhow!("calibration: bad bound `{s}`")),
    }
}

fn bounds(s: &str) -> Result<[f64; 2]> {
    let t = s.trim();
    match t {
        "+" => Ok([0.0, f64::INFINITY]),
        "-" => Ok([f64::NEG_INFINITY, 0.0]),
        _ => {
            let inner = t
                .strip_prefix('[')
                .and_then(|x| x.strip_suffix(']'))
                .ok_or_else(|| anyhow!("calibration: bad bounds `{t}`"))?;
            let p = split_top(inner);
            if p.len() != 2 {
                bail!("calibration: bounds `{t}` need two values");
            }
            Ok([value(&p[0])?, value(&p[1])?])
        }
    }
}

/// One block entry; `irf` = irf_calibration syntax.
pub fn parse_entry(s: &str, irf: bool) -> Result<Restriction> {
    let p = split_top(s);
    if p.len() != 3 {
        bail!("calibration: entry `{s}` must have three comma-separated fields");
    }
    let b = bounds(&p[2])?;
    if irf {
        let (v1, per) = name_and_periods(&p[0], None)?;
        Ok(Restriction { v1, v2: p[1].clone(), periods: per, bounds: b })
    } else {
        let (v2, per) = name_and_periods(&p[1], Some(vec![0]))?;
        Ok(Restriction { v1: p[0].clone(), v2, periods: per, bounds: b })
    }
}

/// The restrictions in force at command `upto` (the last block of each kind before it).
/// The global `options_` as the commands before `upto` leave them, for `th_autocovariances`
/// (`endogenous_prior_restrictions.m`, `+gsa/th_moments.m`): the options of every earlier
/// `stoch_simul` and `estimation` (`hp_filter`, `bandpass_filter`, `filtered_theoretical_moments_grid`,
/// ...). `qz_criterium` keeps the moment default (GSA's own criterium is not carried here).
pub fn persistent_moment_options(mf: &ModFile, upto: usize) -> crate::analysis::moments::MomentOptions {
    let mut so = crate::analysis::StochSimulOptions::default();
    for c in mf.commands.iter().take(upto) {
        match c {
            Command::StochSimul(o, _) | Command::Estimation(o, _) => {
                so.apply(o);
            }
            _ => {}
        }
    }
    crate::analysis::moments::MomentOptions {
        qz_criterium: crate::analysis::moments::MomentOptions::default().qz_criterium,
        ..so.moment_options()
    }
}

pub fn restrictions_before(mf: &ModFile, upto: usize) -> Result<Restrictions> {
    let mut r = Restrictions { moment_options: persistent_moment_options(mf, upto), ..Default::default() };
    for c in mf.commands.iter().take(upto) {
        if let Command::Other(name, o) = c {
            let irf = name == "irf_calibration";
            if !irf && name != "moment_calibration" {
                continue;
            }
            let rows: Vec<Restriction> = o
                .iter()
                .filter(|(k, _)| k == BLOCK_BODY_KEY)
                .filter_map(|(_, v)| if let OptValue::Str(s) = v { Some(s.as_str()) } else { None })
                .map(|s| parse_entry(s, irf))
                .collect::<Result<_>>()?;
            if irf {
                r.irf = rows;
                r.relative_irf = opt(o, "relative_irf").is_some();
            } else {
                r.moment = rows;
            }
        }
    }
    Ok(r)
}

/// `[info, info_irf, info_moment, data_irf, data_moment]` of `endogenous_prior_restrictions.m`.
#[derive(Clone, Debug, Default)]
pub struct RestrictionValues {
    /// `info = [49, penalty]` when restrictions are violated, `[0, 0]` otherwise.
    pub info: (i32, f64),
    /// Per restriction `[code, penalty]`: products over its periods, 0 when one period holds.
    pub info_irf: Vec<[f64; 2]>,
    pub info_moment: Vec<[f64; 2]>,
    /// Per restriction, `(t, value)` in increasing `t`.
    pub data_irf: Vec<Vec<(i64, f64)>>,
    pub data_moment: Vec<Vec<(i64, f64)>>,
}

/// MATLAB `if any(info)` on the `n x 2` matrix of per-restriction `[code, penalty]`: with one
/// restriction `any` of the 1x2 row is a scalar (code or penalty non-zero); with several, `any`
/// is taken per column and `if` needs both columns to hold a non-zero entry.
/// `any` ignores NaN entries.
fn matlab_any(info: &[[f64; 2]]) -> bool {
    let nz = |v: f64| v != 0.0 && !v.is_nan();
    if info.len() == 1 {
        nz(info[0][0]) || nz(info[0][1])
    } else {
        info.iter().any(|i| nz(i[0])) && info.iter().any(|i| nz(i[1]))
    }
}

fn check(v: f64, b: [f64; 2], info: &mut [f64; 2]) {
    if v > b[0] && v < b[1] {
        *info = [0.0, 0.0];
    } else {
        let d = if v < b[0] { (v - b[0]).powi(2) } else { (v - b[1]).powi(2) };
        *info = [info[0] * 49.0, info[1] * d];
    }
}

/// Port of `endogenous_prior_restrictions.m` (order 1) at a solved point.
pub fn endogenous_prior_restrictions(
    dsge: &DsgeModel,
    dr: &DecisionRule,
    sigma_e: &Mat,
    r: &Restrictions,
) -> Result<RestrictionValues> {
    let m = &dsge.model;
    let mut out = RestrictionValues::default();
    let mut penalty = 0.0;
    let mut violated = false;
    if !r.irf.is_empty() {
        let nt = r.irf.iter().flat_map(|x| x.periods.iter().copied()).max().unwrap_or(1).max(1) as usize;
        out.info_irf = vec![[1.0, 1.0]; r.irf.len()];
        out.data_irf = vec![Vec::new(); r.irf.len()];
        // RR = T^(t-1) R diag(sqrt(diag(Sigma_e))) (relative_irf: R * 100), one shock at a time
        let mut cache: Vec<Option<Mat>> = vec![None; m.n_exo()];
        for (j, x) in r.irf.iter().enumerate() {
            let iendo = m.endo_index(&x.v1).ok_or_else(|| anyhow!("{} is not an endogenous variable", x.v1))?;
            let iexo = m.exo_index(&x.v2).ok_or_else(|| anyhow!("{} is not an exogenous variable", x.v2))?;
            if cache[iexo].is_none() {
                let mut e1 = vec![0.0; m.n_exo()];
                e1[iexo] = if r.relative_irf { 100.0 } else { sigma_e[(iexo, iexo)].sqrt() };
                cache[iexo] = Some(crate::analysis::irf::irf(dr, &e1, nt));
            }
            let y = cache[iexo].as_ref().unwrap();
            for t in 1..=nt as i64 {
                if !x.periods.contains(&t) {
                    continue;
                }
                let v = y[(iendo, (t - 1) as usize)];
                out.data_irf[j].push((t, v));
                check(v, x.bounds, &mut out.info_irf[j]);
            }
        }
        if matlab_any(&out.info_irf) {
            violated = true;
            penalty += out.info_irf.iter().map(|i| i[1]).sum::<f64>();
        }
    }
    if !r.moment.is_empty() {
        let mut names: Vec<String> = vec![r.moment[0].v1.clone()];
        for x in &r.moment {
            for v in [&x.v1, &x.v2] {
                if !names.contains(v) {
                    names.push(v.clone());
                }
            }
        }
        let ivar: Vec<usize> = names
            .iter()
            .map(|n| m.endo_index(n).ok_or_else(|| anyhow!("One of the variable specified does not exist")))
            .collect::<Result<_>>()?;
        let ntmax = r.moment.iter().flat_map(|x| x.periods.iter().copied()).max().unwrap_or(0).max(0);
        let ntmin = r.moment.iter().flat_map(|x| x.periods.iter().copied()).min().unwrap_or(0).min(0);
        let mo = crate::analysis::moments::MomentOptions {
            ar: ntmin.unsigned_abs().max(ntmax as u64) as usize,
            nodecomposition: true,
            ..r.moment_options.clone()
        };
        let th = crate::analysis::moments::th_autocovariances(dr, sigma_e, &ivar, &mo)?;
        out.info_moment = vec![[1.0, 1.0]; r.moment.len()];
        out.data_moment = vec![Vec::new(); r.moment.len()];
        let nv = ivar.len();
        for t in ntmin..=ntmax {
            let rr =
                if t == 0 {
                    let g = &th.gamma0;
                    Mat::from_fn(nv, nv, |a, b| {
                        if a == b { g[(a, a)] } else { g[(a, b)] / (g[(a, a)].sqrt() * g[(b, b)].sqrt()) }
                    })
                } else {
                    th.autocorr[t.unsigned_abs() as usize - 1].clone()
                };
            for (j, x) in r.moment.iter().enumerate() {
                if !x.periods.contains(&t) {
                    continue;
                }
                let mut i1 = names.iter().position(|n| *n == x.v1).unwrap();
                let mut i2 = names.iter().position(|n| *n == x.v2).unwrap();
                if t > 0 {
                    std::mem::swap(&mut i1, &mut i2);
                }
                let v = rr[(i1, i2)];
                out.data_moment[j].push((t, v));
                check(v, x.bounds, &mut out.info_moment[j]);
            }
        }
        if matlab_any(&out.info_moment) {
            violated = true;
            penalty += out.info_moment.iter().map(|i| i[1]).sum::<f64>();
        }
    }
    out.info = if violated { (49, penalty) } else { (0, 0.0) };
    Ok(out)
}

// ---- calibration restrictions: the consumers outside the GSA mapping ----

/// The restrictions in force at the command whose option list is `o` (one of
/// `mf.commands`, e.g. an `estimation(...)`), names checked; `None` without
/// `irf_calibration` / `moment_calibration` blocks (the callers then pay nothing).
pub fn restrictions_at_options(mf: &ModFile, o: &crate::modfile::Options) -> Result<Option<Restrictions>> {
    let ci = mf
        .commands
        .iter()
        .position(|c| match c {
            Command::Estimation(oo, _) | Command::Other(_, oo) => std::ptr::eq(oo, o),
            _ => false,
        })
        .unwrap_or(mf.commands.len());
    restrictions_checked(mf, ci)
}

/// [`restrictions_before`] with the variable names checked once (the errors
/// `endogenous_prior_restrictions.m` raises on its first call); `None` when empty.
pub fn restrictions_checked(mf: &ModFile, upto: usize) -> Result<Option<Restrictions>> {
    let r = restrictions_before(mf, upto)?;
    if r.is_empty() {
        return Ok(None);
    }
    let m = &mf.model;
    for x in &r.irf {
        if m.endo_index(&x.v1).is_none() {
            bail!("irf_calibration: {} is not an endogenous variable", x.v1);
        }
        if m.exo_index(&x.v2).is_none() {
            bail!("irf_calibration: {} is not an exogenous variable", x.v2);
        }
    }
    if r.moment.iter().any(|x| m.endo_index(&x.v1).is_none() || m.endo_index(&x.v2).is_none()) {
        bail!("One of the variable specified does not exist");
    }
    Ok(Some(r))
}

/// `info(1)` of `endogenous_prior_restrictions` at a solved point: 49 when a restriction is
/// violated, 0 otherwise (prior_sampler.m, prior.m, stability_mapping.m).
pub fn restrictions_info(dsge: &DsgeModel, dr: &DecisionRule, sigma_e: &Mat, r: &Restrictions) -> (i32, f64) {
    match endogenous_prior_restrictions(dsge, dr, sigma_e, r) {
        Ok(v) => v.info,
        Err(_) => (49, f64::NAN),
    }
}

/// The check of dsge_likelihood.m / dsge_conditional_likelihood_1.m after `dynare_resolve`:
/// `info = endogenous_prior_restrictions(...)`; on a violation `fval = Inf`, `info(1) = 49` and
/// `info(4) = info(2)` (the summed squared distances to the violated bounds; 0.1 when not
/// finite), which `penalty_objective_function` adds during mode finding. A pure function of
/// the solved point.
pub fn likelihood_check(
    dsge: &DsgeModel,
    res: &crate::dsge::Resolved,
    r: &Restrictions,
) -> std::result::Result<(), crate::dsge::EvalFailure> {
    match restrictions_info(dsge, &res.dr, &res.q, r) {
        (0, _) => Ok(()),
        (info, p) => Err(crate::dsge::EvalFailure { info, penalty: if p.is_finite() { p } else { 0.1 } }),
    }
}

/// `method_of_moments`: default_option_mom_values.m drops the restrictions, with a warning
/// when both blocks are present (its test is `~isempty(irf) && ~isempty(moment)`).
pub fn mom_warning(mf: &ModFile, upto: usize) {
    if let Ok(r) = restrictions_before(mf, upto) {
        if !r.irf.is_empty() && !r.moment.is_empty() {
            println!(
                "Warning: method_of_moments: Endogenous prior restrictions are not supported yet and will be skipped."
            );
        }
    }
}

// ---- end calibration restrictions ----

/// MATLAB `%4.1f` (Inf / NaN as MATLAB prints them).
fn f41(v: f64) -> String {
    if v.is_infinite() {
        format!("{:>4}", if v > 0.0 { "Inf" } else { "-Inf" })
    } else if v.is_nan() {
        format!("{:>4}", "NaN")
    } else {
        format!("{v:4.1}")
    }
}

fn num2str(x: f64) -> String {
    super::stats::num2str(x)
}

/// What map_calibration computed (for tests): the kept sample rows (`irestrictions`, 0-based),
/// the indicators (`indx_irf`, `indx_moment`, all rows) and the restriction values.
#[derive(Clone, Debug, Default)]
pub struct CalibrationMap {
    pub irestrictions: Vec<usize>,
    pub indx_irf: Vec<Vec<f64>>,
    pub indx_moment: Vec<Vec<f64>>,
    pub mat_irf: Vec<Vec<Vec<f64>>>,
    pub mat_moment: Vec<Vec<Vec<f64>>>,
}

/// The couples of `map_calibration.m`: `unique(cellstr([char(v1) char(v2)]))`.
fn couples(rs: &[Restriction]) -> (Vec<String>, Vec<usize>) {
    let w1 = rs.iter().map(|r| r.v1.len()).max().unwrap_or(0);
    let w2 = rs.iter().map(|r| r.v2.len()).max().unwrap_or(0);
    let keys: Vec<String> = rs.iter().map(|r| format!("{:<w1$}{:<w2$}", r.v1, r.v2).trim_end().to_string()).collect();
    let mut u = keys.clone();
    u.sort();
    u.dedup();
    let idx = keys.iter().map(|k| u.iter().position(|x| x == k).unwrap()).collect();
    (u, idx)
}

/// Port of `+gsa/map_calibration.m` on the stability sample (`type` = `prior` / `mc`).
pub fn map_calibration(
    ctx: &Ctx,
    dsge: &DsgeModel,
    o: &GsaOptions,
    stab: &StabMap,
    r: &Restrictions,
) -> Result<CalibrationMap> {
    let ty = if o.ppost {
        "post"
    } else if o.pprior {
        "prior"
    } else {
        "mc"
    };
    let nsam = stab.lpmat.len();
    let nshock = ctx.nshock;
    println!();
    println!("Sensitivity analysis for calibration criteria");
    let points: Vec<Vec<f64>> = (0..nsam).map(|k| stab.point(k)).collect();
    let vals: Vec<Option<RestrictionValues>> = points
        .par_iter()
        .map(|x| dsge.resolve(x).ok().and_then(|res| endogenous_prior_restrictions(dsge, &res.dr, &res.q, r).ok()))
        .collect();
    let mut cm = CalibrationMap::default();
    let nirf = r.irf.len();
    let nmom = r.moment.len();
    cm.mat_irf = r.irf.iter().map(|x| vec![vec![f64::NAN; x.periods.len()]; nsam]).collect();
    cm.mat_moment = r.moment.iter().map(|x| vec![vec![f64::NAN; x.periods.len()]; nsam]).collect();
    cm.indx_irf = vec![vec![0.0; nirf]; nsam];
    cm.indx_moment = vec![vec![0.0; nmom]; nsam];
    for (j, v) in vals.iter().enumerate() {
        if let Some(v) = v {
            cm.irestrictions.push(j);
            for ij in 0..nirf {
                for (k, (_, val)) in v.data_irf[ij].iter().enumerate() {
                    cm.mat_irf[ij][j][k] = *val;
                }
                cm.indx_irf[j][ij] = v.info_irf[ij][0];
            }
            for ij in 0..nmom {
                for (k, (_, val)) in v.data_moment[ij].iter().enumerate() {
                    cm.mat_moment[ij][j][k] = *val;
                }
                cm.indx_moment[j][ij] = v.info_moment[ij][0];
            }
        }
    }
    println!();
    let xmat: Vec<Vec<f64>> = cm.irestrictions.iter().map(|&j| points[j].clone()).collect();
    let xstruct: Vec<Vec<f64>> = xmat.iter().map(|x| x[nshock..].to_vec()).collect();
    let nkeep = cm.irestrictions.len().max(1) as f64;
    let pnames: Vec<String> = ctx.param_names[nshock..].to_vec();
    let bnames: Vec<String> = dsge.estimated_names.clone(); // bayestopt_.name
    fn spec<'a>(
        o: &GsaOptions,
        name: String,
        title: String,
        beha: &str,
        t0: String,
        names: &'a [String],
    ) -> McfSpec<'a> {
        McfSpec {
            pvalue_ks: o.pvalue_ks,
            pvalue_corr: o.pvalue_corr,
            alpha2: o.alpha2_stab,
            param_names: names,
            amcf_name: name,
            amcf_title: title,
            beha_title: format!("{beha} restriction"),
            nobeha_title: format!("NO {beha} restriction"),
            title: t0,
            xparam1: None,
            noprint: true,
        }
    }
    let mut figures = Vec::new();
    for (kind, rs, indx, mat, xsam, names) in [
        ("IRF", &r.irf, &cm.indx_irf, &cm.mat_irf, &xstruct, &pnames),
        ("MOMENT", &r.moment, &cm.indx_moment, &cm.mat_moment, &xmat, &bnames),
    ] {
        if rs.is_empty() {
            continue;
        }
        let low = if kind == "IRF" { "irf" } else { "moment" };
        let beha = if kind == "IRF" { "IRF" } else { "moment" };
        println!();
        println!("Deleting old {kind} calibration plots ...");
        println!("done !");
        println!();
        let (cpl, plot_indx) = couples(rs);
        let mut time_matrix: Vec<Vec<i64>> = vec![Vec::new(); cpl.len()];
        let per: Vec<Vec<i64>> = rs
            .iter()
            .map(|x| {
                let mut p = x.periods.clone();
                if kind == "MOMENT" {
                    p.sort_unstable();
                }
                p
            })
            .collect();
        for (ij, p) in per.iter().enumerate() {
            time_matrix[plot_indx[ij]].extend(p.iter().copied());
        }
        let mut iplot = vec![1usize; rs.len()];
        let keep = &cm.irestrictions;
        let mut sum_matrix = vec![vec![0.0; cpl.len()]; keep.len()];
        let mut panels: Vec<Panel> = (0..cpl.len()).map(|_| Panel::default()).collect();
        for (ij, x) in rs.iter().enumerate() {
            for (a, &j) in keep.iter().enumerate() {
                sum_matrix[a][plot_indx[ij]] += indx[j][ij];
            }
            let mut leg = num2str(per[ij][0] as f64);
            let mut aleg = leg.clone();
            if per[ij].len() > 1 {
                leg = format!("{leg}:{}", num2str(*per[ij].last().unwrap() as f64));
                aleg = format!(
                    "{aleg}{}{}",
                    if kind == "IRF" { "-" } else { "_" },
                    num2str(*per[ij].last().unwrap() as f64)
                );
                iplot[ij] = 0;
            }
            if time_matrix[plot_indx[ij]].len() == 1 {
                let v: Vec<f64> = keep.iter().map(|&j| *mat[ij][j].last().unwrap_or(&f64::NAN)).collect();
                let lo = v.iter().copied().filter(|z| z.is_finite()).fold(f64::INFINITY, f64::min);
                let hi = v.iter().copied().filter(|z| z.is_finite()).fold(f64::NEG_INFINITY, f64::max);
                let (x1, x2) = (x.bounds[0].max(lo), x.bounds[1].min(hi));
                panels[plot_indx[ij]] = Panel {
                    title: format!("{} vs {}({leg})", x.v1, x.v2),
                    series: vec![
                        Series::Band {
                            x: vec![x1, x2],
                            lower: vec![0.0, 0.0],
                            upper: vec![1.0, 1.0],
                            color: Color::Rgb(179, 204, 255),
                            alpha: 1.0,
                            label: None,
                        },
                        mcf::cdf_series(&v, Color::Black, LineStyle::Solid, 2.0, None),
                    ],
                    ..Default::default()
                };
            }
            let i1: Vec<usize> = (0..keep.len()).filter(|&a| indx[keep[a]][ij] == 0.0).collect();
            let i2: Vec<usize> = (0..keep.len()).filter(|&a| indx[keep[a]][ij] != 0.0).collect();
            let t0 = format!("{} vs {}({leg})", x.v1, x.v2);
            println!(
                "{}% of the {ty} support matches {kind} {t0} inside [{}, {}]",
                f41(i1.len() as f64 / nkeep * 100.0),
                f41(x.bounds[0]),
                f41(x.bounds[1])
            );
            if !i1.is_empty() && !i2.is_empty() {
                let s = spec(
                    o,
                    format!("{ty}_{low}_calib_{}_vs_{}_{aleg}", x.v1, x.v2),
                    format!("{ty} {kind} Calib: Parameter(s) driving {} vs {}({leg})", x.v1, x.v2),
                    beha,
                    t0,
                    names,
                );
                mcf::analysis(ctx, xsam, &i1, &i2, &s);
            }
        }
        for (ic, tm) in time_matrix.iter().enumerate() {
            if tm.len() <= 1 {
                continue;
            }
            let itmp: Vec<usize> = (0..rs.len()).filter(|&k| plot_indx[k] == ic).collect();
            // band of the restriction, min / max / median over the kept sample
            let mut tsort = tm.clone();
            if kind == "MOMENT" {
                tsort.sort_unstable();
            }
            let mut cols: Vec<Vec<f64>> = Vec::new();
            for &k in &itmp {
                for c in 0..per[k].len() {
                    cols.push(keep.iter().map(|&j| mat[k][j][c]).collect());
                }
            }
            let tx: Vec<f64> = tm.iter().map(|&t| t as f64).collect();
            let mx: Vec<f64> = cols.iter().map(|c| c.iter().copied().fold(f64::NEG_INFINITY, f64::max)).collect();
            let mn: Vec<f64> = cols.iter().map(|c| c.iter().copied().fold(f64::INFINITY, f64::min)).collect();
            let md: Vec<f64> = cols
                .iter()
                .map(|c| {
                    let mut s: Vec<f64> = c.iter().copied().filter(|v| !v.is_nan()).collect();
                    s.sort_by(f64::total_cmp);
                    if s.is_empty() {
                        f64::NAN
                    } else if s.len() % 2 == 1 {
                        s[s.len() / 2]
                    } else {
                        0.5 * (s[s.len() / 2 - 1] + s[s.len() / 2])
                    }
                })
                .collect();
            let lb: Vec<f64> = tm
                .iter()
                .map(|t| itmp.iter().find(|&&k| per[k].contains(t)).map(|&k| rs[k].bounds[0]).unwrap_or(f64::NAN))
                .map(|v| if v.is_finite() { v } else { mn.iter().copied().fold(f64::INFINITY, f64::min) })
                .collect();
            let ub: Vec<f64> = tm
                .iter()
                .map(|t| itmp.iter().find(|&&k| per[k].contains(t)).map(|&k| rs[k].bounds[1]).unwrap_or(f64::NAN))
                .map(|v| if v.is_finite() { v } else { mx.iter().copied().fold(f64::NEG_INFINITY, f64::max) })
                .collect();
            let first = itmp[0];
            panels[ic] = Panel {
                title: format!("{} vs {}", rs[first].v1, rs[first].v2),
                series: vec![
                    Series::Band {
                        x: tx.clone(),
                        lower: lb,
                        upper: ub,
                        color: Color::Rgb(179, 204, 255),
                        alpha: 1.0,
                        label: None,
                    },
                    Series::Line {
                        x: tx.clone(),
                        y: mx,
                        color: Color::Black,
                        style: LineStyle::Dashed,
                        width: 2.0,
                        label: None,
                    },
                    Series::Line {
                        x: tx.clone(),
                        y: mn,
                        color: Color::Black,
                        style: LineStyle::Dashed,
                        width: 2.0,
                        label: None,
                    },
                    Series::Line {
                        x: tx.clone(),
                        y: md,
                        color: Color::Black,
                        style: LineStyle::Solid,
                        width: 2.0,
                        label: None,
                    },
                    Series::HLine { y: 0.0, color: Color::Red, style: LineStyle::Solid },
                ],
                ..Default::default()
            };
            if itmp.iter().any(|&k| iplot[k] == 1) {
                let i1: Vec<usize> = (0..keep.len()).filter(|&a| sum_matrix[a][ic] == 0.0).collect();
                let i2: Vec<usize> = (0..keep.len()).filter(|&a| sum_matrix[a][ic] != 0.0).collect();
                let leg = format!("{}...{}", num2str(tm[0] as f64), num2str(*tm.last().unwrap() as f64));
                let t0 = format!("{} vs {}({leg})", rs[first].v1, rs[first].v2);
                println!(
                    "{}% of the {ty} support matches {kind} restrictions {t0}",
                    f41(i1.len() as f64 / nkeep * 100.0)
                );
                if !i1.is_empty() && !i2.is_empty() {
                    let s = spec(
                        o,
                        format!("{ty}_{low}_calib_{}_vs_{}_ALL", rs[first].v1, rs[first].v2),
                        format!("{ty} {kind} Calib: Parameter(s) driving {} vs {}({leg})", rs[first].v1, rs[first].v2),
                        beha,
                        t0,
                        names,
                    );
                    mcf::analysis(ctx, xsam, &i1, &i2, &s);
                }
            }
        }
        let nrow = (cpl.len() as f64).sqrt().ceil() as usize;
        let ncol = if nrow * (nrow.saturating_sub(1)) > cpl.len() { nrow - 1 } else { nrow };
        figures.push(Figure {
            name: format!("{ty}_{low}_restrictions"),
            title: format!("{ty} evaluation of {low} restrictions"),
            rows: nrow,
            cols: ncol,
            panels,
        });
        println!();
    }
    for f in &figures {
        ctx.save(f);
    }
    save(ctx, ty, &cm, &xmat)?;
    Ok(cm)
}

/// `<fname>_<type>_restrictions.mat`: `xmat`, `irestrictions` (1-based), `indx_irf`, `indx_moment`.
fn save(ctx: &Ctx, ty: &str, cm: &CalibrationMap, xmat: &[Vec<f64>]) -> Result<()> {
    use crate::matfile::{MatOut, write};
    let to_mat = |rows: &[Vec<f64>]| {
        let nc = rows.first().map_or(0, Vec::len);
        Mat::from_fn(rows.len(), nc, |i, j| rows[i][j])
    };
    let ir: Vec<f64> = cm.irestrictions.iter().map(|&j| (j + 1) as f64).collect();
    let (x, i1, i2) = (to_mat(xmat), to_mat(&cm.indx_irf), to_mat(&cm.indx_moment));
    let p = ctx.out_dir.join(format!("{}_{ty}_restrictions.mat", ctx.fname));
    write(
        &p,
        &[
            ("xmat", MatOut::Matrix(&x)),
            ("irestrictions", MatOut::Vector(&ir)),
            ("indx_irf", MatOut::Matrix(&i1)),
            ("indx_moment", MatOut::Matrix(&i2)),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matlab_any_quirks() {
        // one restriction: `any` of the 1x2 row (a violation exactly on the bound, penalty 0)
        assert!(matlab_any(&[[49.0, 0.0]]));
        // several: both columns need a non-zero entry, NaN ignored
        assert!(!matlab_any(&[[49.0, 0.0], [0.0, 0.0]]));
        assert!(matlab_any(&[[49.0, 0.0], [0.0, 1e-3]]));
        assert!(!matlab_any(&[[f64::NAN, f64::NAN]]));
        // a restriction over several periods: codes and squared distances multiply, and one
        // period inside the bounds clears it
        let mut i = [1.0, 1.0];
        check(2.0, [0.0, 1.0], &mut i);
        check(3.0, [0.0, 1.0], &mut i);
        assert_eq!(i, [49.0 * 49.0, 1.0 * 4.0]);
        check(0.5, [0.0, 1.0], &mut i);
        check(3.0, [0.0, 1.0], &mut i);
        assert_eq!(i, [0.0, 0.0]);
    }

    #[test]
    fn entries() {
        let r = parse_entry("y ( 1 : 4 ) , e_g , [ 0 , 3 ]", true).unwrap();
        assert_eq!(r, Restriction { v1: "y".into(), v2: "e_g".into(), periods: vec![1, 2, 3, 4], bounds: [0.0, 3.0] });
        let r = parse_entry("pi ( 1 ) , e_u , +", true).unwrap();
        assert_eq!(r.bounds, [0.0, f64::INFINITY]);
        let r = parse_entry("y , pi ( - ( 1 : 2 ) ) , [ 0 , 1 ]", false).unwrap();
        assert_eq!(r.periods, vec![-1, -2]);
        let r = parse_entry("y , y , [ 0.5 , 3 ]", false).unwrap();
        assert_eq!(r.periods, vec![0]);
        let r = parse_entry("y , pi ( - 2 ) , -", false).unwrap();
        assert_eq!((r.periods.clone(), r.bounds[0]), (vec![-2], f64::NEG_INFINITY));
    }
}
