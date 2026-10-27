// SPDX-License-Identifier: GPL-3.0-or-later
//! Stream C decompositions (Dynare 7.2 `shock_decomposition/`):
//!
//! - `realtime_shock_decomposition.m`: the smoother re-run on each vintage of the sample
//!   (`nobs = presample+1 .. nobs`, optionally `fast_realtime`), giving the rolling ("pool")
//!   realtime decomposition, the 1-step-ahead conditional decomposition, the `forecast`-step
//!   forecast decompositions and the vintages listed in `save_realtime`;
//! - `initial_condition_decomposition.m`: contribution of each variable's smoothed initial
//!   value;
//! - `shock_decomposition.m`'s `forecast_type` branch (decomposition extended over an
//!   unconditional or conditional forecast);
//! - `plot_shock_decomposition.m` (`realtime`, `vintage`, `type=qoq|yoy|aoa`, `fig_name`,
//!   `use_shock_groups`, `flip`, `diff`, `detail_plot`, `plot_init_date`/`plot_end_date`) with
//!   `graph_decomp.m` / `graph_decomp_detail.m` figures;
//! - `squeeze_shock_decomposition.m`.
//!
//! Decompositions are [`Z3`] arrays laid out like MATLAB's `z` (endo x components x periods,
//! column-major): components `0..nshocks` are the shocks (or initial values), then the
//! "initial values" residual, then the smoothed deviation from the steady state.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use rayon::prelude::*;

use crate::analysis::shock_decomposition::{self as sdmod, ShockGroups, parameter_set_name};
use crate::dsge::{DsgeModel, LikelihoodData};
use crate::forecast::commands::{
    self as fc, EstimationState, default_parameter_set, find_shock_groups, graph_options, parameter_vector,
    smoother_inputs, var_indices,
};
use crate::forecast::conditional::{ConditionalForecast, ConstrainedPaths};
use crate::forecast::{Forecast, dsge_smoother, extend_dates, save_figures};
use crate::graphics::{Color, Figure, GraphOptions, LineStyle, Panel, Series};
use crate::linalg::Mat;
use crate::modfile::{Command, ModFile, OptValue, Options, opt};
use crate::solver::DecisionRule;

// ---------------------------------------------------------------------------
// 3-D arrays and periods
// ---------------------------------------------------------------------------

/// n x c x t array, MATLAB layout (`data[(i*c + k)*n + v]`).
#[derive(Clone, Debug, PartialEq)]
pub struct Z3 {
    pub n: usize,
    pub c: usize,
    pub t: usize,
    pub data: Vec<f64>,
}

impl Z3 {
    pub fn zeros(n: usize, c: usize, t: usize) -> Z3 {
        Z3 { n, c, t, data: vec![0.0; n * c * t] }
    }
    #[inline]
    pub fn at(&self, v: usize, k: usize, i: usize) -> f64 {
        self.data[(i * self.c + k) * self.n + v]
    }
    #[inline]
    pub fn at_mut(&mut self, v: usize, k: usize, i: usize) -> &mut f64 {
        &mut self.data[(i * self.c + k) * self.n + v]
    }
    /// Periods `a..b`.
    pub fn periods(&self, a: usize, b: usize) -> Z3 {
        let s = self.n * self.c;
        Z3 { n: self.n, c: self.c, t: b - a, data: self.data[a * s..b * s].to_vec() }
    }
    /// Copies period `src_i` of `src` into period `i`.
    pub fn set_period(&mut self, i: usize, src: &Z3, src_i: usize) {
        let s = self.n * self.c;
        self.data[i * s..(i + 1) * s].copy_from_slice(&src.data[src_i * s..(src_i + 1) * s]);
    }
    /// Rows `rows` only.
    pub fn rows(&self, rows: &[usize]) -> Z3 {
        let mut z = Z3::zeros(rows.len(), self.c, self.t);
        for i in 0..self.t {
            for k in 0..self.c {
                for (a, &v) in rows.iter().enumerate() {
                    *z.at_mut(a, k, i) = self.at(v, k, i);
                }
            }
        }
        z
    }
    /// One row as a c x t matrix (`squeeze(z(v,:,:))`).
    pub fn row_mat(&self, v: usize) -> Mat {
        Mat::from_fn(self.c, self.t, |k, i| self.at(v, k, i))
    }
    pub fn from_row_mats(m: &[Mat]) -> Z3 {
        let (c, t) = m.first().map_or((0, 0), |x| (x.nrows(), x.ncols()));
        let mut z = Z3::zeros(m.len(), c, t);
        for (v, x) in m.iter().enumerate() {
            for i in 0..t {
                for k in 0..c {
                    *z.at_mut(v, k, i) = x[(k, i)];
                }
            }
        }
        z
    }
}

/// A Dynare `dates` value of the frequencies used here (daily dates are what Dynare makes of
/// ISO `YYYY-MM-DD` labels in a csv datafile).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Period {
    Annual(i64),
    /// y*4 + q-1
    Quarter(i64),
    /// y*12 + m-1
    Month(i64),
    /// days since 1970-01-01
    Day(i64),
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

impl Period {
    /// `2004Q1`, `2004M3`, `2004Y`, `2004`, `2004-03-31`.
    pub fn parse(s: &str) -> Option<Period> {
        let s = s.trim();
        let p: Vec<&str> = s.split('-').collect();
        if p.len() == 3 {
            let (y, m, d) = (p[0].parse().ok()?, p[1].parse().ok()?, p[2].parse().ok()?);
            return Some(Period::Day(days_from_civil(y, m, d)));
        }
        for (sep, per) in [('Q', 4), ('q', 4), ('M', 12), ('m', 12)] {
            if let Some((a, b)) = s.split_once(sep) {
                let (y, k): (i64, i64) = (a.parse().ok()?, b.parse().ok()?);
                return Some(if per == 4 { Period::Quarter(y * 4 + k - 1) } else { Period::Month(y * 12 + k - 1) });
            }
        }
        s.trim_end_matches(['Y', 'y']).parse().ok().map(Period::Annual)
    }
    pub fn add(self, k: i64) -> Period {
        match self {
            Period::Annual(x) => Period::Annual(x + k),
            Period::Quarter(x) => Period::Quarter(x + k),
            Period::Month(x) => Period::Month(x + k),
            Period::Day(x) => Period::Day(x + k),
        }
    }
    pub fn year(self) -> i64 {
        match self {
            Period::Annual(x) => x,
            Period::Quarter(x) => x.div_euclid(4),
            Period::Month(x) => x.div_euclid(12),
            Period::Day(x) => civil_from_days(x).0,
        }
    }
    /// `subperiod` (1-based).
    pub fn subperiod(self) -> i64 {
        match self {
            Period::Annual(_) => 1,
            Period::Quarter(x) => x.rem_euclid(4) + 1,
            Period::Month(x) => x.rem_euclid(12) + 1,
            Period::Day(x) => {
                let (y, _, _) = civil_from_days(x);
                x - days_from_civil(y, 1, 1) + 1
            }
        }
    }
    /// Periods from `self` to `o`.
    fn diff(self, o: Period) -> Option<i64> {
        match (self, o) {
            (Period::Annual(a), Period::Annual(b))
            | (Period::Quarter(a), Period::Quarter(b))
            | (Period::Month(a), Period::Month(b))
            | (Period::Day(a), Period::Day(b)) => Some(b - a),
            _ => None,
        }
    }
    /// `char(d)`.
    pub fn label(self) -> String {
        match self {
            Period::Annual(x) => format!("{x}Y"),
            Period::Quarter(x) => format!("{}Q{}", x.div_euclid(4), x.rem_euclid(4) + 1),
            Period::Month(x) => format!("{}M{}", x.div_euclid(12), x.rem_euclid(12) + 1),
            Period::Day(x) => {
                let (y, m, d) = civil_from_days(x);
                format!("{y:04}-{m:02}-{d:02}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// recursions
// ---------------------------------------------------------------------------

/// ghx / ghu with declaration-order rows, and `i_state` (declaration indices of the states).
struct Reduced {
    ghx: Mat,
    ghu: Mat,
    i_state: Vec<usize>,
}

impl Reduced {
    fn new(dr: &DecisionRule) -> Reduced {
        let n = dr.ghx.nrows();
        Reduced {
            ghx: Mat::from_fn(n, dr.ghx.ncols(), |i, j| dr.ghx[(dr.inv_order_var[i], j)]),
            ghu: Mat::from_fn(n, dr.ghu.ncols(), |i, j| dr.ghu[(dr.inv_order_var[i], j)]),
            i_state: (dr.nstatic..dr.nstatic + dr.nspred).map(|k| dr.order_var[k]).collect(),
        }
    }
    /// `ghx(inv_order_var,:) * x(i_state, :)`
    fn propagate(&self, x: &Mat) -> Mat {
        let s = Mat::from_fn(self.i_state.len(), x.ncols(), |a, k| x[(self.i_state[a], k)]);
        &self.ghx * s
    }
}

/// The z recursion of realtime_shock_decomposition.m for one vintage: `dev` n x gend smoothed
/// deviations, `eps` nshocks x gend smoothed shocks; periods after `gend` (`forecast` of
/// them) propagate the total and the shock contributions with ghx. No shock contribution
/// in the first period.
fn realtime_recursion(r: &Reduced, dev: &Mat, eps: &Mat, forecast: usize) -> Z3 {
    let n = dev.nrows();
    let gend = dev.ncols();
    let ns = eps.nrows();
    let tt = gend + forecast;
    let mut z = Z3::zeros(n, ns + 2, tt);
    let mut w = Mat::zeros(n, ns);
    let mut tot = Mat::from_fn(n, 1, |v, _| dev[(v, 0)]);
    for i in 0..tt {
        if i > 0 {
            if i >= gend {
                tot = r.propagate(&tot);
            } else {
                tot = Mat::from_fn(n, 1, |v, _| dev[(v, i)]);
            }
            w = r.propagate(&w);
            if i < gend {
                for k in 0..ns {
                    let e = eps[(k, i)];
                    for v in 0..n {
                        w[(v, k)] += r.ghu[(v, k)] * e;
                    }
                }
            }
        }
        for v in 0..n {
            let mut s = 0.0;
            for k in 0..ns {
                *z.at_mut(v, k, i) = w[(v, k)];
                s += w[(v, k)];
            }
            *z.at_mut(v, ns + 1, i) = tot[(v, 0)];
            *z.at_mut(v, ns, i) = tot[(v, 0)] - s;
        }
    }
    z
}

/// ztmp of realtime_shock_decomposition.m's epilogue branches: the first `h` smoothed periods
/// as (zeros, dev, dev) followed by the periods of `zn`.
fn with_history(dev: &Mat, h: usize, zn: &Z3) -> Z3 {
    let (n, c) = (zn.n, zn.c);
    let mut z = Z3::zeros(n, c, h + zn.t);
    for i in 0..h {
        for v in 0..n {
            *z.at_mut(v, c - 2, i) = dev[(v, i)];
            *z.at_mut(v, c - 1, i) = dev[(v, i)];
        }
    }
    for i in 0..zn.t {
        z.set_period(h + i, zn, i);
    }
    z
}

/// One smoother vintage: smoothed deviations (n x gend, declaration order) and shocks
/// (nshocks x gend).
#[derive(Clone)]
pub struct Vintage {
    pub dev: Mat,
    pub eps: Mat,
}

impl Vintage {
    fn truncate(&self, gend: usize) -> Vintage {
        Vintage { dev: self.dev.columns(0, gend).into_owned(), eps: self.eps.columns(0, gend).into_owned() }
    }
}

/// Everything realtime_shock_decomposition.m stores in `oo_` (keys: `time_<j>`).
#[derive(Clone, Debug)]
pub struct Realtime {
    /// `oo_.realtime_shock_decomposition.pool` (nobs + forecast periods).
    pub pool: Z3,
    /// `oo_.conditional_shock_decomposition.pool` (1-step ahead, nobs periods).
    pub conditional_pool: Z3,
    /// `oo_.realtime_forecast_shock_decomposition.pool` (nobs+1 periods), if forecast > 0.
    pub forecast_pool: Option<Z3>,
    /// `oo_.realtime_shock_decomposition.time_<j>` (gend + forecast periods).
    pub vintages: BTreeMap<usize, Z3>,
    /// `oo_.realtime_forecast_shock_decomposition.time_<j>` (forecast+1 periods from j).
    pub forecast_vintages: BTreeMap<usize, Z3>,
    /// `oo_.conditional_shock_decomposition.time_<j>` (forecast-step conditional, from j).
    pub conditional_vintages: BTreeMap<usize, Z3>,
    /// `oo_.realtime_conditional_shock_decomposition.time_<j>`.
    pub realtime_conditional_vintages: BTreeMap<usize, Z3>,
    pub forecast: usize,
    pub nshocks: usize,
}

/// `with_epilogue`: number of epilogue variables and `epilogue_shock_decomposition` on an
/// n x c x T array (rows appended).
pub type Epilogue<'a> = (usize, &'a (dyn Fn(&Z3) -> Result<Z3> + Sync));

/// Port of realtime_shock_decomposition.m's loop (order 1, `with_epilogue` through `epi`, no
/// `forecast_params`). `smooth(j)` runs the smoother on the first j observations.
pub fn realtime_decomposition(
    dr: &DecisionRule,
    nobs: usize,
    presample: usize,
    forecast: usize,
    save_realtime: &[usize],
    fast_realtime: Option<&[usize]>,
    smooth: &(dyn Fn(usize) -> Result<Vintage> + Sync),
    epi: Option<&Epilogue>,
) -> Result<Realtime> {
    let r = Reduced::new(dr);
    let n0 = dr.ghx.nrows();
    let n = n0 + epi.map_or(0, |e| e.0);
    let ns = dr.ghu.ncols();
    let c = ns + 2;
    if presample >= nobs {
        bail!("realtime_shock_decomposition: presample ({presample}) must be smaller than nobs ({nobs})");
    }
    // prefetch every vintage in parallel when each needs its own smoother run
    let mut cache: HashMap<usize, Vintage> = HashMap::new();
    if fast_realtime.is_none() {
        let v: Vec<(usize, Result<Vintage>)> = (presample + 1..=nobs).into_par_iter().map(|j| (j, smooth(j))).collect();
        for (j, s) in v {
            cache.insert(j, s?);
        }
    }
    let mut get = |j: usize| -> Result<Vintage> {
        if let Some(v) = cache.get(&j) {
            return Ok(v.clone());
        }
        let v = smooth(j)?;
        cache.insert(j, v.clone());
        Ok(v)
    };
    let saved = |j: usize| save_realtime.contains(&j);
    let mut out = Realtime {
        pool: Z3::zeros(n, c, nobs + forecast),
        conditional_pool: Z3::zeros(n, c, nobs),
        forecast_pool: (forecast > 0).then(|| Z3::zeros(n, c, nobs + 1)),
        vintages: BTreeMap::new(),
        forecast_vintages: BTreeMap::new(),
        conditional_vintages: BTreeMap::new(),
        realtime_conditional_vintages: BTreeMap::new(),
        forecast,
        nshocks: ns,
    };
    let mut rf: HashMap<usize, Z3> = HashMap::new(); // ootmp.realtime_forecast_shock_decomposition
    let mut gend0 = 0usize;
    let mut oo0: Option<Vintage> = None;
    let mut init = true;
    for j in presample + 1..=nobs {
        let (gend, local) = match fast_realtime {
            None => (j, get(j)?),
            Some(fast) => {
                let fmin = *fast.iter().min().unwrap();
                let fmax = *fast.iter().max().unwrap();
                if j < fmin && gend0 < j {
                    oo0 = Some(get(fmin)?);
                    gend0 = fmin;
                }
                if fast.contains(&j) && gend0 < j {
                    let v = get(j)?;
                    gend0 = j;
                    oo0 = Some(v.clone());
                    (j, v)
                } else {
                    if j > gend0 {
                        let use_n = if j > fmax { nobs } else { *fast.iter().filter(|&&f| f > j).min().unwrap() };
                        oo0 = Some(get(use_n)?);
                        gend0 = use_n;
                    }
                    (j, oo0.as_ref().unwrap().truncate(j))
                }
            }
        };
        let mut z = realtime_recursion(&r, &local.dev, &local.eps, forecast);
        if let Some((_, f)) = epi {
            z = f(&z)?;
        }
        // conditional shock decomp 1 step ahead
        let mut z1 = Z3::zeros(n0, c, 1);
        for v in 0..n0 {
            let mut s = 0.0;
            for k in 0..ns {
                let x = r.ghu[(v, k)] * local.eps[(k, gend - 1)];
                *z1.at_mut(v, k, 0) = x;
                s += x;
            }
            let tot = local.dev[(v, gend - 1)];
            *z1.at_mut(v, ns + 1, 0) = tot;
            *z1.at_mut(v, ns, 0) = tot - s;
        }
        if let Some((_, f)) = epi {
            z1 = f(&with_history(&local.dev, gend - 1, &z1))?.periods(gend - 1, gend);
        }
        // conditional shock decomp k step ahead
        if forecast > 0 && forecast < j && saved(j - forecast) {
            let a = gend - forecast - 1;
            let w = Vintage {
                dev: local.dev.columns(a, forecast + 1).into_owned(),
                eps: local.eps.columns(a, forecast + 1).into_owned(),
            };
            let mut zn = realtime_recursion(&r, &w.dev, &w.eps, 0);
            if let Some((_, f)) = epi {
                zn = f(&with_history(&local.dev, a, &zn))?.periods(a, a + forecast + 1);
            }
            out.conditional_vintages.insert(j - forecast, zn);
        }
        if init {
            for i in 0..j {
                out.pool.set_period(i, &z, i);
            }
        } else if j < nobs {
            out.pool.set_period(j - 1, &z, gend - 1);
        } else {
            for (k, i) in (gend - 1..z.t).enumerate() {
                out.pool.set_period(j - 1 + k, &z, i);
            }
        }
        out.conditional_pool.set_period(j - 1, &z1, 0);
        if saved(j) {
            out.vintages.insert(j, z.clone());
        }
        if forecast > 0 {
            out.forecast_pool.as_mut().unwrap().set_period(j, &z, gend);
            let f = z.periods(gend - 1, z.t);
            if saved(j) {
                out.forecast_vintages.insert(j, f.clone());
            }
            rf.insert(j, f);
            if j > forecast + presample {
                let mut todo = vec![forecast];
                if j == nobs {
                    todo.extend((1..forecast).rev());
                }
                for mf in todo {
                    let k = j - mf;
                    if !saved(k) {
                        continue;
                    }
                    let fk = &rf[&k];
                    let mut rc = out.pool.periods(k - 1, j);
                    for i in 0..=mf {
                        for v in 0..n {
                            for kk in 0..c {
                                *rc.at_mut(v, kk, i) -= fk.at(v, kk, i);
                            }
                            *rc.at_mut(v, ns, i) += fk.at(v, ns + 1, i);
                            *rc.at_mut(v, ns + 1, i) = out.pool.at(v, ns + 1, k - 1 + i);
                        }
                    }
                    out.realtime_conditional_vintages.insert(k, rc);
                }
            }
        }
        init = false;
    }
    Ok(out)
}

/// Port of initial_condition_decomposition.m's recursion: z is n x (n+2) x gend; component
/// k < n is the contribution of variable k's smoothed value in the first period, component n
/// what is left (the shocks), n+1 the smoothed deviation.
pub fn initial_condition_decomposition(dr: &DecisionRule, dev: &Mat) -> Z3 {
    let r = Reduced::new(dr);
    let n = dev.nrows();
    let gend = dev.ncols();
    let mut z = Z3::zeros(n, n + 2, gend);
    let mut w = Mat::from_fn(n, n, |a, b| if a == b { dev[(a, 0)] } else { 0.0 });
    for i in 0..gend {
        if i > 0 {
            w = r.propagate(&w);
        }
        for v in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                *z.at_mut(v, k, i) = w[(v, k)];
                s += w[(v, k)];
            }
            *z.at_mut(v, n + 1, i) = dev[(v, i)];
            *z.at_mut(v, n, i) = dev[(v, i)] - s;
        }
    }
    z
}

// ---------------------------------------------------------------------------
// plot_shock_decomposition
// ---------------------------------------------------------------------------

/// A stored decomposition (`oo_.shock_decomposition`, `oo_.initval_decomposition`,
/// `oo_.forecast_shock_decomposition.<type>`).
#[derive(Clone, Debug)]
pub struct Stored {
    pub z: Z3,
    pub parameter_set: String,
    /// Number of leading "shock" components (exo_nbr, or endo_nbr for initval).
    pub nshocks: usize,
    /// Number of forecast periods at the end (forecast_type decompositions).
    pub forecast_length: Option<usize>,
}

/// `options_.plot_shock_decomp` as set by a `plot_shock_decomposition` command (after
/// `set_default_plot_shock_decomposition_options`).
#[derive(Clone, Debug, Default)]
pub struct PlotSpec {
    pub realtime: u8,
    pub vintage: usize,
    pub kind: String,
    pub fig_name: String,
    pub detail_plot: bool,
    pub flip: bool,
    pub diff: bool,
    pub groups: Option<ShockGroups>,
    pub plot_init_date: Option<Period>,
    pub plot_end_date: Option<Period>,
    pub max_nrows: usize,
    /// initial_condition_decomposition's private flag (and screen_shocks).
    pub init_cond: bool,
    /// shock_decomposition's forecast_type (the source is oo_.forecast_shock_decomposition).
    pub forecast_type: Option<String>,
    /// options_.initial_date (empty for a standalone plot command, the data's first date when
    /// shock_decomposition plots).
    pub initial_date: Option<Period>,
    /// init2shocks, screen_shocks, steadystate, write_xls, colormap, interactive.
    pub q4: crate::q4::decomp::PlotExtra,
}

impl PlotSpec {
    pub fn from_options(mf: &ModFile, o: &Options) -> Result<PlotSpec> {
        let n = |k: &str| match opt(o, k) {
            Some(OptValue::Num(v)) => Some(*v),
            Some(OptValue::Flag) => Some(1.0),
            _ => None,
        };
        let s = |k: &str| match opt(o, k) {
            Some(OptValue::Str(v)) => Some(v.clone()),
            Some(OptValue::Num(v)) => Some(v.to_string()),
            _ => None,
        };
        let date = |k: &str| -> Result<Option<Period>> {
            match s(k) {
                None => Ok(None),
                Some(d) => Period::parse(&d).map(Some).ok_or_else(|| anyhow!("{k}: cannot read the date {d}")),
            }
        };
        let groups = match opt(o, "use_shock_groups") {
            Some(OptValue::Str(g)) => Some(find_shock_groups(mf, g)?),
            Some(OptValue::Flag) => Some(find_shock_groups(mf, "default")?),
            _ => None,
        };
        let kind = s("type").unwrap_or_default();
        if !["", "qoq", "yoy", "aoa"].contains(&kind.as_str()) {
            bail!("plot_shock_decomposition:: Wrong type");
        }
        Ok(PlotSpec {
            realtime: n("realtime").unwrap_or(0.0) as u8,
            vintage: n("vintage").unwrap_or(0.0) as usize,
            kind,
            fig_name: s("fig_name").unwrap_or_default(),
            detail_plot: n("detail_plot").is_some_and(|v| v != 0.0),
            flip: n("flip").is_some_and(|v| v != 0.0),
            diff: n("diff").is_some_and(|v| v != 0.0),
            groups,
            plot_init_date: date("plot_init_date")?,
            plot_end_date: date("plot_end_date")?,
            max_nrows: n("max_nrows").map_or(6, |v| v as usize),
            q4: crate::q4::decomp::PlotExtra::from_options(mf, o)?,
            ..Default::default()
        })
    }
}

/// The decomposition as plot_shock_decomposition hands it to graph_decomp (after `get_decomp_only`'s
/// point, before the date window), plus what the figures need.
#[derive(Clone, Debug)]
pub struct Plotted {
    /// nvar x (ncomp+1) x T
    pub z: Z3,
    pub names: Vec<String>,
    /// Component labels (without the total).
    pub labels: Vec<String>,
    pub initial_date: Period,
    /// For each period of `z`, its position in the source's time base (for real dates).
    pub base: Vec<i64>,
    pub fig_name_long: String,
    pub parameter_set: String,
    pub forecast_length: Option<usize>,
    pub plot_init_date: Option<Period>,
    pub plot_end_date: Option<Period>,
    /// `options_.plot_shock_decomp.steady_state` of the plotted rows (`steadystate`).
    pub steady_state: Option<Vec<f64>>,
}

/// What plot_shock_decomposition needs from `oo_`.
pub struct Sources<'a> {
    pub shock_decomposition: Option<&'a Stored>,
    pub initval: Option<&'a Stored>,
    pub forecast_sd: &'a BTreeMap<String, Stored>,
    pub realtime: Option<&'a (Realtime, String)>,
    /// `options_.shock_decomp.forecast`
    pub sd_forecast: usize,
    /// `options_.nobs`
    pub nobs: usize,
}

/// Port of plot_shock_decomposition.m up to its `get_decomp_only` return (z for the requested
/// rows), with the figure names. `rows`: rows of the stored decomposition to plot, `names`
/// their names, `shock_names` the labels of the shock components.
pub fn transform(
    spec: &PlotSpec,
    src: &Sources,
    rows: &[usize],
    names: &[String],
    shock_names: &[String],
) -> Result<Plotted> {
    let mut fig_name = if spec.fig_name.is_empty() { String::new() } else { format!(" {}", spec.fig_name) };
    let mut forecast = src.sd_forecast;
    if spec.vintage > 0 {
        forecast = forecast.min(src.nobs.saturating_sub(spec.vintage));
    }
    let mut initial_date = spec.initial_date.unwrap_or(if spec.kind.is_empty() {
        Period::Annual(1)
    } else {
        Period::Quarter(4) /* 1Q1 */
    });
    let mut base0: i64 = 0;
    let missing =
        |what: &str| anyhow!("plot_shock_decomposition: {what} is not available (run the corresponding command first)");
    let rt = || src.realtime.ok_or_else(|| missing("oo_.realtime_shock_decomposition"));
    let vint = |m: &'_ BTreeMap<usize, Z3>, what: &str| -> Result<Z3> {
        m.get(&spec.vintage).cloned().ok_or_else(|| {
            anyhow!("plot_shock_decomposition: {what}.time_{} was not saved (save_realtime)", spec.vintage)
        })
    };
    let (z, parameter_set, nshocks, forecast_length, fig_name1) = match spec.realtime {
        0 => match &spec.forecast_type {
            None => {
                let s = if spec.init_cond { src.initval } else { src.shock_decomposition }.ok_or_else(|| {
                    missing(if spec.init_cond { "oo_.initval_decomposition" } else { "oo_.shock_decomposition" })
                })?;
                (s.z.clone(), s.parameter_set.clone(), s.nshocks, None, fig_name.clone())
            }
            Some(t) => {
                let s =
                    src.forecast_sd.get(t).ok_or_else(|| missing(&format!("oo_.forecast_shock_decomposition.{t}")))?;
                (s.z.clone(), s.parameter_set.clone(), s.nshocks, s.forecast_length, format!("{fig_name} {t} forecast"))
            }
        },
        1 => {
            let (r, set) = rt()?;
            if spec.vintage > 0 {
                let name =
                    format!("{fig_name} realtime (vintage {})", initial_date.add(spec.vintage as i64 - 1).label());
                (vint(&r.vintages, "oo_.realtime_shock_decomposition")?, set.clone(), r.nshocks, None, name)
            } else {
                (r.pool.clone(), set.clone(), r.nshocks, None, format!("{fig_name} realtime (rolling)"))
            }
        }
        2 => {
            let (r, set) = rt()?;
            if spec.vintage > 0 {
                initial_date = initial_date.add(spec.vintage as i64 - 1);
                base0 = spec.vintage as i64 - 1;
                let name =
                    format!("{fig_name} {forecast}-step ahead conditional forecast (given {})", initial_date.label());
                (
                    vint(&r.realtime_conditional_vintages, "oo_.realtime_conditional_shock_decomposition")?,
                    set.clone(),
                    r.nshocks,
                    None,
                    name,
                )
            } else {
                let name = format!("{fig_name} 1-step ahead conditional forecast (rolling)");
                (r.conditional_pool.clone(), set.clone(), r.nshocks, None, name)
            }
        }
        3 => {
            let (r, set) = rt()?;
            if spec.vintage > 0 {
                initial_date = initial_date.add(spec.vintage as i64 - 1);
                base0 = spec.vintage as i64 - 1;
                let name = format!("{fig_name} {forecast}-step ahead forecast (given {})", initial_date.label());
                (
                    vint(&r.forecast_vintages, "oo_.realtime_forecast_shock_decomposition")?,
                    set.clone(),
                    r.nshocks,
                    None,
                    name,
                )
            } else {
                let fp = r
                    .forecast_pool
                    .clone()
                    .ok_or_else(|| missing("oo_.realtime_forecast_shock_decomposition (forecast = 0)"))?;
                (fp, set.clone(), r.nshocks, None, format!("{fig_name} 1-step ahead forecast (rolling)"))
            }
        }
        x => bail!("plot_shock_decomposition: realtime = {x} (0..3 expected)"),
    };
    fig_name = fig_name1;
    let mut z = z.rows(rows);
    let mut labels: Vec<String> = shock_names.to_vec();
    labels.push(if spec.init_cond { "All shocks".into() } else { "Initial values".into() });
    // ---- init2shocks: initial conditions attributed to shocks (realtime = 0)
    let i2s = spec.q4.init2shocks.as_ref().filter(|_| spec.realtime == 0);
    if let Some(pairs) = i2s {
        let iv = src.initval.ok_or_else(|| missing("oo_.initval_decomposition"))?;
        let mut mats = Vec::new();
        for (i, &r) in rows.iter().enumerate() {
            let mut m = z.row_mat(i);
            let names_init =
                crate::q4::decomp::apply_init2shocks(&mut m, &iv.z.row_mat(r), pairs, &spec.q4.endo_names, shock_names);
            labels[..names_init.len()].clone_from_slice(&names_init);
            mats.push(m);
        }
        z = Z3::from_row_mats(&mats);
    }
    let mut steady_state: Option<Vec<f64>> =
        spec.q4.steady_state.as_ref().map(|ss| rows.iter().map(|&r| ss.get(r).copied().unwrap_or(0.0)).collect());
    if let Some(g) = &spec.groups {
        fig_name += &format!(" group {}", g.name);
        let mut mats = Vec::new();
        let mut l = Vec::new();
        for v in 0..z.n {
            let (zz, ll) = sdmod::group(&z.row_mat(v), nshocks, g);
            mats.push(zz);
            l = ll;
        }
        // Dynare labels the last component "Others + Initial Values" if any variable has
        // ungrouped shocks (zothers over all rows).
        z = Z3::from_row_mats(&mats);
        labels = l;
    }
    if spec.flip {
        fig_name += " flip";
    }
    if spec.diff {
        fig_name += " diff";
    }
    if i2s.is_some() {
        fig_name += " init2shocks";
    }
    let mut base: Vec<i64> = (0..z.t as i64).map(|i| base0 + i).collect();
    if spec.diff {
        for i in (1..z.t).rev() {
            for k in 0..z.c {
                for v in 0..z.n {
                    *z.at_mut(v, k, i) -= z.at(v, k, i - 1);
                }
            }
        }
        for k in 0..z.c {
            for v in 0..z.n {
                *z.at_mut(v, k, 0) = f64::NAN;
            }
        }
        steady_state.iter_mut().flatten().for_each(|s| *s = 0.0);
    }
    let mut plot_init_date = spec.plot_init_date;
    let mut plot_end_date = spec.plot_end_date;
    match spec.kind.as_str() {
        "" | "qoq" => {}
        "yoy" => {
            let t = z.t.saturating_sub(3);
            let mut y = Z3::zeros(z.n, z.c, t);
            for i in 0..t {
                for k in 0..z.c {
                    for v in 0..z.n {
                        *y.at_mut(v, k, i) = z.at(v, k, i) + z.at(v, k, i + 1) + z.at(v, k, i + 2) + z.at(v, k, i + 3);
                    }
                }
            }
            z = y;
            initial_date = initial_date.add(3);
            base = base[3.min(base.len())..].to_vec();
            steady_state.iter_mut().flatten().for_each(|s| *s *= 4.0);
        }
        "aoa" => {
            if matches!(initial_date, Period::Day(_)) {
                // Dynare 7.2's dates/subperiod error message
                bail!("dates::subperiod: Method not available for daily frequency!");
            }
            let sub = initial_date.subperiod();
            let mut t0 = if sub == 1 { 1 } else { 4 - sub + 2 };
            if spec.realtime == 0 {
                t0 += 3;
            }
            if plot_init_date.is_none() && spec.realtime == 0 {
                plot_init_date = Some(initial_date.add(t0));
            }
            let t0 = 4 - sub + 1; // no q2a: Q4 values
            if t0 < 1 {
                bail!("plot_shock_decomposition: type = aoa needs quarterly dates");
            }
            let idx: Vec<usize> = (t0 as usize - 1..z.t).step_by(4).collect();
            let mut y = Z3::zeros(z.n, z.c, idx.len());
            for (a, &i) in idx.iter().enumerate() {
                y.set_period(a, &z, i);
            }
            z = y;
            base = idx.iter().map(|&i| base[i]).collect();
            initial_date = Period::Annual(initial_date.year());
            plot_init_date = plot_init_date.map(|d| Period::Annual(d.year()));
            plot_end_date = plot_end_date.map(|d| Period::Annual(d.year()));
        }
        _ => bail!("plot_shock_decomposition:: Wrong type"),
    }
    if spec.flip {
        z.data.iter_mut().for_each(|x| *x = -*x);
        steady_state.iter_mut().flatten().for_each(|s| *s = -*s);
    }
    Ok(Plotted {
        z,
        names: names.to_vec(),
        labels,
        initial_date,
        base,
        fig_name_long: fig_name,
        parameter_set,
        forecast_length,
        plot_init_date,
        plot_end_date,
        steady_state: steady_state.filter(|_| spec.q4.steadystate),
    })
}

impl Plotted {
    /// plot_init_date / plot_end_date window: (first period, end, first date), None (with
    /// Dynare's warning) if plot_init_date is outside the sample.
    pub fn window(&self) -> Option<(usize, usize, Period)> {
        let t = self.z.t;
        let (mut a, mut b, mut first) = (0usize, t, self.initial_date);
        if let Some(p) = self.plot_init_date {
            match self.initial_date.diff(p).filter(|&k| k >= 0 && (k as usize) < t) {
                Some(k) => {
                    a = k as usize;
                    first = p;
                }
                None => {
                    println!(
                        "WARNING: You set plot_init_date larger than the last observation ({}); nothing to plot.",
                        self.initial_date.add(t as i64).label()
                    );
                    return None;
                }
            }
        }
        if let Some(p) = self.plot_end_date {
            match self.initial_date.diff(p) {
                Some(k) if k >= 0 && (k as usize) < t => b = k as usize + 1,
                _ => println!("WARNING: You set plot_end_date larger than the last observation; plotting to the end."),
            }
        }
        (a < b).then_some((a, b, first))
    }

    pub(crate) fn sanitized(&self) -> String {
        let s = self.fig_name_long.replace("(given ", "").replace("(vintage ", "").replace(' ', "_");
        let s: String = s.chars().filter(|c| !matches!(c, '.' | '-' | ')' | '(')).collect();
        format!("{s}_{}", self.parameter_set)
    }

    pub(crate) fn fig_mode1(kind: &str) -> String {
        if kind.is_empty() { String::new() } else { format!("_{kind}") }
    }

    /// graph_decomp.m / graph_decomp_detail.m figures for every variable.
    pub fn figures(&self, spec: &PlotSpec, realtime: u8, vintage: usize) -> Vec<Figure> {
        let Some((a, b, first)) = self.window() else { return Vec::new() };
        let comp = self.z.c - 1;
        let (preamble_txt, preamble) = if spec.init_cond {
            ("Initial condition decomposition", "initval_decomposition_")
        } else if vintage > 0 && realtime > 1 {
            ("Shock decomposition", "shock_decomposition_")
        } else {
            ("Historical shock decomposition", "shock_decomposition_")
        };
        let fm1 = Self::fig_mode1(&spec.kind);
        // screen_shocks (initial_condition_decomposition sets it): graph_decomp
        // appends " SCREEN" to the figure name when it applies
        let screen = (spec.init_cond || spec.q4.screen_shocks) && spec.groups.is_none() && comp > 18;
        let mut this = self.clone();
        if screen {
            this.fig_name_long += " SCREEN";
        }
        let fig_name = this.sanitized();
        let pss = parameter_set_name(&self.parameter_set);
        let t = b - a;
        let x: Vec<f64> = (1..=t).map(|i| i as f64).collect();
        let thin = (t / 8).max(1);
        let ticks: Vec<(f64, String)> =
            (0..t).step_by(thin).map(|i| ((i + 1) as f64, first.add(i as i64).label())).collect();
        let mut figs = Vec::new();
        for (v, var) in self.names.iter().enumerate() {
            let mut z1 = Mat::from_fn(self.z.c, t, |k, i| self.z.at(v, k, a + i));
            let mut labels = self.labels.clone();
            if screen {
                // 16 largest contributions (mean |z|), Others, then the last component + total
                let m = comp - 1;
                let mut isort: Vec<usize> = (0..m).collect();
                let score = |k: usize| (0..t).map(|i| z1[(k, i)].abs()).sum::<f64>() / t as f64;
                isort.sort_by(|&p, &q| score(q).total_cmp(&score(p)));
                let mut nz = Mat::zeros(19, t);
                for (r, &k) in isort.iter().take(16).enumerate() {
                    nz.row_mut(r).copy_from(&z1.row(k));
                }
                for &k in isort.iter().skip(16) {
                    for i in 0..t {
                        nz[(16, i)] += z1[(k, i)];
                    }
                }
                nz.row_mut(17).copy_from(&z1.row(comp - 1));
                nz.row_mut(18).copy_from(&z1.row(comp));
                let mut l: Vec<String> = isort.iter().take(16).map(|&k| self.labels[k].clone()).collect();
                l.push("Others".into());
                l.push(self.labels[comp - 1].clone());
                z1 = nz;
                labels = l;
            }
            // graph_decomp_detail.m: a fig_name ending in 'initval' (initval
            // decomposition plotted through plot_shock_decomposition) labels the last component
            // "All shocks"
            let fig_initval = spec.fig_name.len() + 1 >= 8 && spec.fig_name.ends_with("initval");
            if spec.detail_plot
                && fig_initval
                && let Some(l) = labels.last_mut()
            {
                *l = "All shocks".into();
            }
            let nc = z1.nrows() - 1;
            let (mut ymax, mut ymin) = (f64::NEG_INFINITY, f64::INFINITY);
            for i in 0..t {
                let (p, q) = (0..nc)
                    .map(|k| z1[(k, i)])
                    .filter(|x| x.is_finite())
                    .fold((0.0f64, 0.0f64), |(p, q), x| (p + x.max(0.0), q + x.min(0.0)));
                ymax = ymax.max(p * 1.1);
                ymin = ymin.min(q * 1.1);
            }
            if !(ymax - ymin >= 1e-6) {
                continue;
            }
            let row = |k: usize| z1.row(k).iter().copied().collect::<Vec<f64>>();
            let total = Series::Line {
                x: x.clone(),
                y: row(nc),
                color: Color::Black,
                style: LineStyle::Solid,
                width: 2.0,
                label: None,
            };
            let title_mid = format!("{preamble_txt}{}{}", this.fig_name_long, fm1.replace('_', " "));
            if spec.detail_plot {
                let ncol = 3;
                let ntot = nc.div_ceil(ncol);
                let nrow = ntot.min(spec.max_nrows.max(1));
                let nfigs = if screen { 1 } else { ntot.div_ceil(nrow) };
                for jf in 0..nfigs {
                    let mut panels = Vec::new();
                    for ic in nrow * ncol * jf..(nrow * ncol * (jf + 1)).min(nc) {
                        let zz = row(ic);
                        let rest: Vec<f64> = (0..t).map(|i| z1[(nc, i)] - zz[i]).collect();
                        panels.push(Panel {
                            title: labels[ic].clone(),
                            series: vec![
                                Series::StackedBars {
                                    x: x.clone(),
                                    components: vec![zz, rest],
                                    labels: vec![labels[ic].clone(), "Rest".into()],
                                },
                                total.clone(),
                            ],
                            x_ticks: Some(ticks.clone()),
                            ..Default::default()
                        });
                    }
                    let suffix = if nfigs > 1 { format!("_detail_{}", jf + 1) } else { "_detail".into() };
                    figs.push(Figure {
                        name: format!("{preamble}{var}{fm1}{fig_name}{suffix}"),
                        title: format!("{title_mid} ({pss}): {var} (detail)."),
                        rows: nrow,
                        cols: ncol,
                        panels,
                    });
                }
            } else {
                let mut series = vec![
                    Series::StackedBars {
                        x: x.clone(),
                        components: (0..nc).map(row).collect(),
                        labels: labels.clone(),
                    },
                    total,
                ];
                if let Some(fl) = self.forecast_length {
                    series.push(Series::VLine {
                        x: (self.z.t - fl) as f64 + 0.5 - a as f64,
                        color: Color::Red,
                        style: LineStyle::Solid,
                    });
                }
                // steadystate: the zero line of the levels (y = -steady state)
                if let Some(ss) = self.steady_state.as_ref().and_then(|s| s.get(v)) {
                    series.push(Series::HLine { y: 0.0, color: Color::Gray, style: LineStyle::Dashed });
                    if ymin + ss < 0.0 && ymax + ss > 0.0 {
                        series.push(Series::HLine { y: -ss, color: Color::Black, style: LineStyle::Dashed });
                    }
                }
                figs.push(Figure {
                    name: format!("{preamble}{var}{fm1}{fig_name}"),
                    title: format!("{title_mid}: {var} ({pss})."),
                    rows: 1,
                    cols: 1,
                    panels: vec![Panel {
                        title: var.clone(),
                        series,
                        x_ticks: Some(ticks.clone()),
                        legend: true,
                        ..Default::default()
                    }],
                });
            }
        }
        figs
    }

    /// `<fname>_<shock|initval>_decomposition<_type><fig_name>.csv`: `period,date,variable,
    /// component,value` over the plotted window (`period` = Dynare's dates, `date` = the
    /// sample date the column refers to, when known).
    pub fn write_csv(
        &self,
        spec: &PlotSpec,
        out_dir: &Path,
        fname: &str,
        base_dates: &dyn Fn(i64) -> String,
    ) -> Result<std::path::PathBuf> {
        let (a, b, first) = self.window().unwrap_or((0, self.z.t, self.initial_date));
        let stem = if spec.init_cond { "initval_decomposition" } else { "shock_decomposition" };
        let p = out_dir.join(format!("{fname}_{stem}{}{}.csv", Self::fig_mode1(&spec.kind), self.sanitized()));
        std::fs::create_dir_all(out_dir)?;
        let mut w = String::from("period,date,variable,component,value\n");
        for (v, var) in self.names.iter().enumerate() {
            for i in a..b {
                let per = first.add((i - a) as i64).label();
                let date = base_dates(self.base[i]);
                for k in 0..self.z.c {
                    let c = self.labels.get(k).map_or("Smoothed", String::as_str);
                    w += &format!("{per},{date},{var},{c},{}\n", self.z.at(v, k, i));
                }
            }
        }
        std::fs::write(&p, w).with_context(|| format!("writing {}", p.display()))?;
        Ok(p)
    }
}

// ---------------------------------------------------------------------------
// driver state and commands
// ---------------------------------------------------------------------------

/// The parts of `oo_` / `options_` the stream C commands share between commands.
#[derive(Default)]
pub struct DecompositionState {
    /// `constrained_vars_` / `constrained_paths_`.
    pub cond_paths: Option<ConstrainedPaths>,
    /// `oo_.conditional_forecast`.
    pub conditional_forecast: Option<ConditionalForecast>,
    /// `oo_.forecast` of the last `forecast` command.
    pub forecast: Option<Forecast>,
    pub shock_decomposition: Option<Stored>,
    pub forecast_sd: BTreeMap<String, Stored>,
    /// `oo_.realtime_*` and their parameter set.
    pub realtime: Option<(Realtime, String)>,
    pub initval: Option<Stored>,
    /// `oo_.shock_decomposition_info.i_var` after squeeze_shock_decomposition (declaration
    /// indices of the rows kept).
    pub squeezed: Option<Vec<usize>>,
    /// `oo_.plot_shock_decomposition_info.i_var`.
    pub plotted: BTreeSet<usize>,
    /// `options_.parameter_set` (persists between commands, as in Dynare).
    pub parameter_set: Option<String>,
    /// `options_.shock_decomp.forecast`.
    pub sd_forecast: usize,
    /// epilogue steady state, `oo_.dr.ys`.
    pub q4: crate::q4::decomp::DecompExtra,
}

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

fn nums(o: &Options, k: &str) -> Option<Vec<usize>> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(vec![*v as usize]),
        Some(OptValue::Vec(v)) => Some(v.iter().map(|&x| x as usize).collect()),
        Some(OptValue::List(l)) => {
            Some(l.iter().filter_map(|x| if let OptValue::Num(v) = x { Some(*v as usize) } else { None }).collect())
        }
        _ => None,
    }
}

impl DecompositionState {
    fn check_not_squeezed(&self, who: &str) -> Result<()> {
        if self.squeezed.is_some()
            && (self.shock_decomposition.is_some()
                || self.realtime.is_some()
                || self.initval.is_some()
                || !self.forecast_sd.is_empty())
        {
            bail!("{who}::squeezed shock decompositions are already stored in oo_");
        }
        Ok(())
    }

    /// Records `options_.parameter_set` from a command's options; returns the set to use.
    /// `who` names the command in Dynare's error when there is no default.
    fn parameter_set(&mut self, st: Option<&EstimationState>, o: &Options, who: &str) -> Result<String> {
        if let Some(OptValue::Str(s)) = opt(o, "parameter_set") {
            self.parameter_set = Some(s.clone());
        }
        match &self.parameter_set {
            Some(s) => Ok(s.clone()),
            None => Ok(default_parameter_set(st, who)?.to_string()),
        }
    }

    /// Row of the stored decompositions for declaration index `v` (None if squeezed out).
    fn row(&self, v: usize) -> Option<usize> {
        match &self.squeezed {
            None => Some(v),
            Some(iv) => iv.iter().position(|&w| w == v),
        }
    }
}

/// Parameters for evaluate_smoother(parameter_set) (`calibration` = current `M_.params`).
pub(crate) fn smoother_point(st: Option<&EstimationState>, dsge: &DsgeModel, set: &str) -> Result<Vec<f64>> {
    parameter_vector(st, dsge, set)
}

/// Sample dates (or 1..) extended past the sample.
fn dates_fn(dates: Option<Vec<String>>, nobs: usize, first_obs: usize) -> impl Fn(i64) -> String {
    let base: Vec<String> =
        dates.clone().unwrap_or_else(|| (first_obs..first_obs + nobs).map(|t| t.to_string()).collect());
    let ext = extend_dates(dates.as_deref(), first_obs - 1 + nobs, 400);
    move |i: i64| {
        if i < 0 {
            String::new()
        } else if (i as usize) < base.len() {
            base[i as usize].clone()
        } else {
            ext.get(i as usize - base.len()).cloned().unwrap_or_default()
        }
    }
}

fn graphs_for(o: &Options, st: Option<&EstimationState>, dir: &Path, fname: &str) -> GraphOptions {
    let mut g = graph_options(o, dir, fname);
    if let Some(s) = st {
        g.enabled &= s.graphs.enabled || opt(o, "nograph").is_none();
        if opt(o, "graph_format").is_none() {
            g.formats = s.graphs.formats.clone();
        }
    }
    g
}

fn write_z3_csv(
    path: &Path,
    z: &Z3,
    rows: &[(usize, String)],
    labels: &[String],
    date: &dyn Fn(usize) -> String,
) -> Result<()> {
    let sd_rows: Vec<(String, Mat, Vec<String>)> =
        rows.iter().map(|(r, n)| (n.clone(), z.row_mat(*r), labels.to_vec())).collect();
    let dates: Vec<String> = (0..z.t).map(date).collect();
    sdmod::write_csv(path, &sd_rows, &dates)
}

/// `shock_decomposition(...) varlist;`: stream F's command, or the `forecast_type` branch;
/// either way the result is kept for plot_shock_decomposition.
#[allow(clippy::too_many_arguments)]
pub fn shock_decomposition_command(
    mf: &ModFile,
    st: Option<&EstimationState>,
    o: &Options,
    vars: &[String],
    state: &mut DecompositionState,
    dir: &Path,
    fname: &str,
    out_dir: &Path,
) -> Result<()> {
    state.check_not_squeezed("shock_decomposition")?;
    let set = state.parameter_set(st, o, "shock_decomposition")?;
    if let Some(OptValue::Str(t)) = opt(o, "forecast_type") {
        return forecast_shock_decomposition(mf, st, o, vars, state, &set, t, dir, fname, out_dir);
    }
    let mut o2 = o.clone();
    if opt(o, "parameter_set").is_none() {
        o2.push(("parameter_set".into(), OptValue::Str(set)));
    }
    let (sd, set, _ys) = fc::shock_decomposition_command(mf, st, &o2, vars, dir, fname, out_dir)?;
    // ---- with_epilogue: epilogue_shock_decomposition.m
    let zrows = crate::q4::decomp::after_shock_decomposition(
        mf,
        st,
        o,
        &sd.z,
        &_ys,
        &mut state.q4,
        out_dir,
        fname,
        "shock_decomposition",
    )?;
    let z = Z3::from_row_mats(&zrows);
    state.shock_decomposition = Some(Stored { z, parameter_set: set, nshocks: sd.nshocks, forecast_length: None });
    Ok(())
}

/// `oo_.forecast` as the estimation's `forecast=H` option leaves it (forecast at the mode from
/// the last smoothed state, the estimation's var_list), when no `forecast` command ran.
fn estimation_forecast(mf: &ModFile, st: &EstimationState) -> Result<Option<Forecast>> {
    let Some(h) = mf.commands.iter().rev().find_map(|c| match c {
        Command::Estimation(o, _) => Some(fc::forecast_horizon(o)).filter(|&h| h > 0),
        _ => None,
    }) else {
        return Ok(None);
    };
    let conf = mf.commands.iter().rev().find_map(|c| match c {
        Command::Estimation(o, _) => Some(num(o, "conf_sig").unwrap_or(0.9)),
        _ => None,
    });
    Ok(Some(fc::smoother_forecast(mf, st, &st.mode, h, conf.unwrap_or(0.9))?)) // forecasts.run(..., 'smoother')
}

/// shock_decomposition.m's `forecast_type` branch: the smoothed decomposition extended over
/// the unconditional (`oo_.forecast` / `oo_.MeanForecast`) or conditional
/// (`oo_.conditional_forecast`) forecast of `vars`.
#[allow(clippy::too_many_arguments)]
fn forecast_shock_decomposition(
    mf: &ModFile,
    st: Option<&EstimationState>,
    o: &Options,
    vars: &[String],
    state: &mut DecompositionState,
    set: &str,
    ftype: &str,
    dir: &Path,
    fname: &str,
    out_dir: &Path,
) -> Result<()> {
    let mut own = None;
    let (dsge, data, dates) = smoother_inputs(mf, st, o, dir, &mut own)?;
    let x = smoother_point(st, dsge, set)?;
    let sm = dsge_smoother(dsge, data, &x, false)?;
    let (i_var, names) = var_indices(mf, vars)?;
    let dev = sm.smoothed_deviation();
    let ys = sm.ys().to_vec();
    let n = dev.nrows();
    let ns = sm.out.etahat.nrows();
    let gs = dev.ncols();
    // forecast levels per i_var: (fl, closure t -> value)
    let (fl, fvals): (usize, Vec<Vec<f64>>) = match ftype {
        "unconditional" => {
            // shock_decomposition.m: calibration / mle_mode use oo_.forecast, every Bayesian
            // parameter set oo_.MeanForecast (the posterior mean forecast)
            let uf = if matches!(set, "calibration" | "mle_mode") {
                match &state.forecast {
                    Some(f) => f.clone(),
                    None => st.map(|s| estimation_forecast(mf, s)).transpose()?.flatten().ok_or_else(|| {
                        anyhow!(if set == "mle_mode" {
                            "Can't find unconditional forecasts in oo_.forecasts. Did you run ML estimation with the forecast option?"
                        } else {
                            "Can't find unconditional forecasts in oo_.forecasts. Did you run the forecast command?"
                        })
                    })?,
                }
            } else {
                st.and_then(|s| s.mean_forecast.clone()).ok_or_else(|| {
                    anyhow!(
                        "Can't find unconditional forecasts in oo_.MeanForecast. Did you run Bayesian estimation with the forecast option?"
                    )
                })?
            };
            let h = uf.mean.nrows();
            let cols = i_var
                .iter()
                .zip(&names)
                .map(|(v, nm)| {
                    let j = uf
                        .vars
                        .iter()
                        .position(|w| w == v)
                        .ok_or_else(|| anyhow!("oo_.forecast has no forecast of {nm}"))?;
                    Ok(uf.mean.column(j).iter().copied().collect())
                })
                .collect::<Result<Vec<Vec<f64>>>>()?;
            (h, cols)
        }
        "conditional" => {
            let cf = state.conditional_forecast.as_ref().ok_or_else(|| anyhow!("Can't find conditional forecasts"))?;
            let cols = i_var
                .iter()
                .zip(&names)
                .map(|(&v, nm)| {
                    let j = cf.column(v).ok_or_else(|| anyhow!("oo_.conditional_forecast has no forecast of {nm}"))?;
                    Ok((1..=cf.periods).map(|t| cf.cond.mean[(t, j)]).collect())
                })
                .collect::<Result<Vec<Vec<f64>>>>()?;
            (cf.periods, cols)
        }
        other => bail!("shock_decomposition: forecast_type must be unconditional or conditional (got {other})"),
    };
    let gend = gs + fl;
    let mut dev_ext = Mat::zeros(n, gend);
    dev_ext.columns_mut(0, gs).copy_from(&dev);
    for (k, &v) in i_var.iter().enumerate() {
        for h in 0..fl {
            // Dynare subtracts dr.ys even when the forecast is in logs (loglinear) or around
            // the sample mean (prefilter); reproduced as is
            dev_ext[(v, gs + h)] = fvals[k][h] - ys[v];
        }
    }
    let mut eps = Mat::zeros(ns, gend);
    eps.columns_mut(0, gs).copy_from(&sm.out.etahat);
    if ftype == "conditional" {
        // shock_decomposition.m reads cf.controlled_exo_variables.Mean.(M_.exo_names{i}) of
        // every shock: MATLAB errors on the first shock that is not controlled
        let cf = state.conditional_forecast.as_ref().unwrap();
        if let Some(k) = (0..ns).find(|k| !cf.instruments.contains(k)) {
            bail!("Reference to non-existent field '{}'.", mf.model.exo_names[k]);
        }
        for (j, &k) in cf.instruments.iter().enumerate() {
            for t in 0..cf.exo.mean.nrows() {
                eps[(k, gs + t)] = cf.exo.mean[(t, j)];
            }
        }
    }
    let init_state = num(o, "init_state").unwrap_or(0.0) as usize;
    let sd = sdmod::shock_decomposition(sm.dr(), &dev_ext, &eps, init_state);
    let zrows = crate::q4::decomp::after_shock_decomposition(
        mf,
        st,
        o,
        &sd.z,
        &ys,
        &mut state.q4,
        out_dir,
        fname,
        &format!("shock_decomposition_{ftype}_forecast"),
    )?;
    let z = Z3::from_row_mats(&zrows);
    let stored = Stored { z, parameter_set: set.to_string(), nshocks: ns, forecast_length: Some(fl) };
    let nobs = data.y.ncols();
    let first_obs = st.map_or(1, |s| s.first_obs);
    let date = dates_fn(dates.clone(), nobs, first_obs);
    let mut labels = mf.model.exo_names.clone();
    labels.push("Initial values".into());
    let rows: Vec<(usize, String)> = i_var.iter().copied().zip(names.iter().cloned()).collect();
    let path = out_dir.join(format!("{fname}_shock_decomposition_{ftype}_forecast.csv"));
    write_z3_csv(&path, &stored.z, &rows, &labels, &|i| date(i as i64))?;
    state.forecast_sd.insert(ftype.to_string(), stored);
    println!(
        "Shock decomposition over the {ftype} forecast ({}, {} variables, {fl} forecast periods) written to {}",
        parameter_set_name(set),
        rows.len(),
        path.display()
    );
    // shock_decomposition plots unless nograph (with the data's first date)
    let g = graphs_for(o, st, dir, fname);
    if g.enabled && opt(o, "nograph").is_none() {
        let mut spec = PlotSpec::from_options(mf, &Vec::new())?;
        if let Some(OptValue::Str(gname)) = opt(o, "use_shock_groups") {
            spec.groups = Some(find_shock_groups(mf, gname)?);
        }
        spec.forecast_type = Some(ftype.to_string());
        // options_.initial_date = dataset_.init (Dynare's dates: ISO labels in a csv
        // make a daily dataset, so this is the file's first date + first_obs - 1 days)
        spec.initial_date = crate::dataset::last_sample_init()
            .and_then(|d| Period::parse(&d.label()))
            .or_else(|| dates.as_ref().and_then(|d| d.first()).and_then(|d| Period::parse(d)));
        plot_rows(mf, st, &spec, state, &i_var, &names, &g, fname, None, &date)?;
    }
    Ok(())
}

/// Builds and saves the figures (and optionally the CSV) for `rows`.
#[allow(clippy::too_many_arguments)]
fn plot_rows(
    mf: &ModFile,
    st: Option<&EstimationState>,
    spec: &PlotSpec,
    state: &DecompositionState,
    vars: &[usize],
    names: &[String],
    g: &GraphOptions,
    fname: &str,
    csv_dir: Option<&Path>,
    date: &dyn Fn(i64) -> String,
) -> Result<Option<Plotted>> {
    let rows = vars
        .iter()
        .zip(names)
        .map(|(&v, n)| {
            state.row(v).ok_or_else(|| {
                anyhow!("shock decomposition results for {n} were not stored (squeeze_shock_decomposition)")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    // initval components are the initial values of every variable (squeezing keeps them)
    let shock_names: Vec<String> =
        if spec.init_cond { mf.model.endo_names.clone() } else { mf.model.exo_names.clone() };
    let src = Sources {
        shock_decomposition: state.shock_decomposition.as_ref(),
        initval: state.initval.as_ref(),
        forecast_sd: &state.forecast_sd,
        realtime: state.realtime.as_ref(),
        sd_forecast: state.sd_forecast,
        nobs: st.map_or(0, |s| s.data.y.ncols()),
    };
    let p = transform(spec, &src, &rows, names, &shock_names)?;
    if let Some(d) = csv_dir {
        let path = p.write_csv(spec, d, fname, date)?;
        println!("plot_shock_decomposition:{} written to {}", p.fig_name_long, path.display());
        if spec.q4.write_xls {
            let path = crate::q4::decomp::write_xls(&p, spec, mf, d, fname)?;
            println!("plot_shock_decomposition: write_xls: {}", path.display());
        }
    }
    let figs = p.figures(spec, spec.realtime, spec.vintage);
    crate::q4::decomp::with_colormap(spec, p.z.c - 1, || save_figures(&figs, fname, g)); // colormap
    Ok(Some(p))
}

/// `realtime_shock_decomposition(parameter_set=, forecast=, save_realtime=, fast_realtime=,
/// presample=, ...) varlist;` Writes `<fname>_realtime_shock_decomposition_pool.csv`,
/// `..._conditional_shock_decomposition_pool.csv`, `..._realtime_forecast_shock_decomposition_pool.csv`
/// and one CSV per saved vintage (`_time_<j>`), for `varlist` (all declared variables if empty).
#[allow(clippy::too_many_arguments)]
pub fn realtime_shock_decomposition_command(
    mf: &ModFile,
    st: Option<&EstimationState>,
    o: &Options,
    vars: &[String],
    state: &mut DecompositionState,
    dir: &Path,
    fname: &str,
    out_dir: &Path,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    state.check_not_squeezed("realtime_shock_decomposition")?;
    let set = state.parameter_set(st, o, "realtime_shock_decomposition")?;
    let mut own = None;
    let (dsge, data, dates) = smoother_inputs(mf, st, o, dir, &mut own)?;
    let x = smoother_point(st, dsge, &set)?;
    let nobs = num(o, "nobs").map_or(data.y.ncols(), |v| (v as usize).min(data.y.ncols()));
    // presample = min(max(1, options_.presample-1), max(1, options_.shock_decomp.presample)),
    // the latter NaN by default (max(1, NaN) = 1): vintages start at 2 unless set
    let presample =
        (data.kalman.presample.saturating_sub(1).max(1)).min(num(o, "presample").map_or(1, |p| (p as usize).max(1)));
    if let Some(f) = num(o, "forecast") {
        state.sd_forecast = f as usize;
    }
    let forecast = state.sd_forecast;
    let save = nums(o, "save_realtime").unwrap_or_default();
    let fast = nums(o, "fast_realtime").filter(|v| !v.is_empty());
    let resolved =
        dsge.resolve(&x).map_err(|e| anyhow!("realtime_shock_decomposition: no solution (info {})", e.info))?;
    let smooth = |j: usize| -> Result<Vintage> {
        let sample = data.y.columns(0, j).transpose();
        let mut d = LikelihoodData::new(&sample, data.kalman.clone());
        if let Some(m) = &data.prefilter_mean {
            // evaluate_smoother -> makedataset with options_.nobs = j: the vintage's own mean
            for (i, &mi) in m.iter().enumerate() {
                d.y.row_mut(i).add_scalar_mut(mi);
            }
            d.prefilter();
        }
        let sm = dsge_smoother(dsge, &d, &x, false)?;
        Ok(Vintage { dev: sm.smoothed_deviation(), eps: sm.out.etahat.clone() })
    };
    // with_epilogue (options_.shock_decomp.with_epilogue)
    let with_epi = crate::q4::decomp::with_epilogue(mf, o)?;
    let params = dsge.set_all_parameters(&x).0;
    let ys = resolved.steady_state.clone();
    let extra = std::cell::RefCell::new(std::mem::take(&mut state.q4));
    let epi_fn = |z: &Z3| -> Result<Z3> {
        let mats: Vec<Mat> = (0..z.n).map(|v| z.row_mat(v)).collect();
        let all = crate::q4::decomp::append_epilogue(mf, &params, &ys, &mats, &mut extra.borrow_mut())?;
        Ok(Z3::from_row_mats(&all))
    };
    // the closure needs to be Sync: keep the RefCell out of it by running it serially
    let rt = {
        let epi_sync = SyncFn(&epi_fn);
        let g = move |z: &Z3| epi_sync.call(z);
        let ep: Option<Epilogue> =
            with_epi.then_some((mf.q4.epilogue.len(), &g as &(dyn Fn(&Z3) -> Result<Z3> + Sync)));
        realtime_decomposition(&resolved.dr, nobs, presample, forecast, &save, fast.as_deref(), &smooth, ep.as_ref())?
    };
    state.q4 = extra.into_inner();
    state.q4.ys = Some(ys.clone());
    let (idx, names) = crate::q4::decomp::var_indices(mf, vars)?;
    let rows: Vec<(usize, String)> = idx.iter().copied().zip(names.iter().cloned()).collect();
    let mut labels = mf.model.exo_names.clone();
    labels.push("Initial values".into());
    let date = dates_fn(dates, nobs, st.map_or(1, |s| s.first_obs));
    let w = |file: String, z: &Z3, start: usize| -> Result<()> {
        write_z3_csv(&out_dir.join(format!("{fname}_{file}.csv")), z, &rows, &labels, &|i| date((start + i) as i64))
    };
    w("realtime_shock_decomposition_pool".into(), &rt.pool, 0)?;
    w("conditional_shock_decomposition_pool".into(), &rt.conditional_pool, 0)?;
    if let Some(fp) = &rt.forecast_pool {
        w("realtime_forecast_shock_decomposition_pool".into(), fp, 0)?;
    }
    for (j, z) in &rt.vintages {
        w(format!("realtime_shock_decomposition_time_{j}"), z, 0)?;
    }
    for (j, z) in &rt.forecast_vintages {
        w(format!("realtime_forecast_shock_decomposition_time_{j}"), z, j - 1)?;
    }
    for (j, z) in &rt.conditional_vintages {
        w(format!("conditional_shock_decomposition_time_{j}"), z, j - 1)?;
    }
    for (j, z) in &rt.realtime_conditional_vintages {
        w(format!("realtime_conditional_shock_decomposition_time_{j}"), z, j - 1)?;
    }
    println!(
        "{}ealtime shock decomposition ({}, vintages {}..{nobs}, forecast = {forecast}, {:.2}s) written to {}/{fname}_*shock_decomposition_*.csv",
        if fast.is_some() { "Fast r" } else { "R" },
        parameter_set_name(&set),
        presample + 1,
        t0.elapsed().as_secs_f64(),
        out_dir.display()
    );
    state.realtime = Some((rt, set));
    Ok(())
}

/// The epilogue closure is only ever called from the (serial) recursion loop.
struct SyncFn<'a>(&'a dyn Fn(&Z3) -> Result<Z3>);
unsafe impl Sync for SyncFn<'_> {}
impl SyncFn<'_> {
    fn call(&self, z: &Z3) -> Result<Z3> {
        (self.0)(z)
    }
}

/// `plot_shock_decomposition(options) varlist;`: figures through graphics, and the plotted
/// decomposition as `<fname>_shock_decomposition<_type><fig_name>.csv`.
#[allow(clippy::too_many_arguments)]
pub fn plot_shock_decomposition_command(
    mf: &ModFile,
    st: Option<&EstimationState>,
    o: &Options,
    vars: &[String],
    state: &mut DecompositionState,
    dir: &Path,
    fname: &str,
    out_dir: &Path,
) -> Result<()> {
    let mut spec = PlotSpec::from_options(mf, o)?;
    let (idx, names) = crate::q4::decomp::var_indices(mf, vars)?; // + epilogue variables
    crate::q4::decomp::prepare_plot(mf, st, o, &mut spec, state, &idx, dir)?;
    if state.squeezed.is_none() && !vars.is_empty() {
        state.plotted.extend(idx.iter().copied());
    }
    let g = graphs_for(o, st, dir, fname);
    let nobs = st.map_or(0, |s| s.data.y.ncols());
    let date = dates_fn(st.and_then(|s| s.dates.clone()), nobs.max(1), st.map_or(1, |s| s.first_obs));
    plot_rows(mf, st, &spec, state, &idx, &names, &g, fname, Some(out_dir), &date)?;
    Ok(())
}

/// `initial_condition_decomposition(options) varlist;`
#[allow(clippy::too_many_arguments)]
pub fn initial_condition_decomposition_command(
    mf: &ModFile,
    st: Option<&EstimationState>,
    o: &Options,
    vars: &[String],
    state: &mut DecompositionState,
    dir: &Path,
    fname: &str,
    out_dir: &Path,
) -> Result<()> {
    let set = state.parameter_set(st, o, "shock_decomposition")?;
    let mut own = None;
    let (dsge, data, dates) = smoother_inputs(mf, st, o, dir, &mut own)?;
    if state.initval.is_none() {
        state.check_not_squeezed("initval_decomposition")?;
        let x = smoother_point(st, dsge, &set)?;
        let sm = dsge_smoother(dsge, data, &x, false)?;
        let z = initial_condition_decomposition(sm.dr(), &sm.smoothed_deviation());
        let n = z.n;
        // with_epilogue (options_.initial_condition_decomp.with_epilogue)
        let z = crate::q4::decomp::initval_with_epilogue(mf, st, o, z, sm.ys(), &mut state.q4)?;
        state.initval = Some(Stored { z, parameter_set: set.clone(), nshocks: n, forecast_length: None });
    }
    let (idx, names) = crate::q4::decomp::var_indices(mf, vars)?;
    let nobs = data.y.ncols();
    let date = dates_fn(dates, nobs, st.map_or(1, |s| s.first_obs));
    let iv = state.initval.as_ref().unwrap();
    let mut labels: Vec<String> = mf.model.endo_names.clone();
    labels.push("All shocks".into());
    let rows = idx
        .iter()
        .zip(&names)
        .map(|(&v, n)| {
            state.row(v).map(|r| (r, n.clone())).ok_or_else(|| anyhow!("initval decomposition of {n} was not stored"))
        })
        .collect::<Result<Vec<_>>>()?;
    let path = out_dir.join(format!("{fname}_initval_decomposition.csv"));
    write_z3_csv(&path, &iv.z, &rows, &labels, &|i| date(i as i64))?;
    println!(
        "Initial condition decomposition ({}) written to {}",
        parameter_set_name(&iv.parameter_set),
        path.display()
    );
    let mut spec = PlotSpec::from_options(mf, o)?;
    spec.realtime = 0;
    spec.groups = None;
    spec.init_cond = true;
    let g = graphs_for(o, st, dir, fname);
    plot_rows(mf, st, &spec, state, &idx, &names, &g, fname, None, &date)?;
    Ok(())
}

/// `squeeze_shock_decomposition varlist;`: keeps only the plotted variables and `varlist` in
/// the stored decompositions.
pub fn squeeze_shock_decomposition(mf: &ModFile, vars: &[String], state: &mut DecompositionState) -> Result<()> {
    let mut keep: Vec<usize> = Vec::new();
    for v in vars {
        keep.push(mf.model.endo_index(v).ok_or_else(|| anyhow!("squeeze_shock_decomposition: unknown variable {v}"))?);
    }
    keep.extend(state.plotted.iter().copied());
    let mut seen = BTreeSet::new();
    keep.retain(|v| seen.insert(*v));
    if keep.is_empty() {
        println!("Nothing has been squeezed: there is no list of variables for it!");
        return Ok(());
    }
    let rows = keep
        .iter()
        .map(|&v| state.row(v).ok_or_else(|| anyhow!("squeeze: {} was already squeezed out", mf.model.endo_names[v])))
        .collect::<Result<Vec<_>>>()?;
    let sq = |z: &mut Z3| *z = z.rows(&rows);
    if let Some(s) = state.shock_decomposition.as_mut() {
        sq(&mut s.z);
    }
    for s in state.forecast_sd.values_mut() {
        sq(&mut s.z);
    }
    if let Some(s) = state.initval.as_mut() {
        // the components (initial values of each variable) are kept, only rows go
        sq(&mut s.z);
    }
    if let Some((r, _)) = state.realtime.as_mut() {
        sq(&mut r.pool);
        sq(&mut r.conditional_pool);
        if let Some(f) = r.forecast_pool.as_mut() {
            sq(f);
        }
        for m in [
            &mut r.vintages,
            &mut r.forecast_vintages,
            &mut r.conditional_vintages,
            &mut r.realtime_conditional_vintages,
        ] {
            m.values_mut().for_each(sq);
        }
    }
    state.squeezed = Some(keep.clone());
    state.plotted = keep.iter().copied().collect();
    println!("squeeze_shock_decomposition: decompositions kept for {} variable(s)", keep.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periods() {
        let q = Period::parse("2010Q1").unwrap();
        assert_eq!(q.add(5).label(), "2011Q2");
        assert_eq!(q.subperiod(), 1);
        assert_eq!(Period::parse("2004-03-31").unwrap().add(1).label(), "2004-04-01");
        assert_eq!(Period::parse("2004-03-31").unwrap().year(), 2004);
        assert_eq!(Period::Annual(1).add(85).label(), "86Y");
        assert_eq!(Period::Quarter(4).label(), "1Q1");
        assert_eq!(Period::parse("1999Y"), Some(Period::Annual(1999)));
    }

    #[test]
    fn z3_layout() {
        let mut z = Z3::zeros(2, 3, 4);
        *z.at_mut(1, 2, 3) = 5.0;
        assert_eq!(z.data[(3 * 3 + 2) * 2 + 1], 5.0);
        assert_eq!(z.periods(2, 4).at(1, 2, 1), 5.0);
        assert_eq!(z.rows(&[1]).at(0, 2, 3), 5.0);
        assert_eq!(Z3::from_row_mats(&[z.row_mat(0), z.row_mat(1)]), z);
    }
}
