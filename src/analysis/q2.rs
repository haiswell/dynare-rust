// SPDX-License-Identifier: GPL-3.0-or-later
//! the `stoch_simul` analysis options that stream S left out.
//!
//! * filters of the simulated moments (`get_filtered_time_series.m`): `one_sided_hp_filter`
//!   (`one_sided_hp_filter.m`, a Kalman filter on the local-linear-trend model) and
//!   `bandpass_filter` (dseries' `baxter_king_filter_.m`); the theoretical bandpass gain of
//!   `th_autocovariances.m` lives in `moments.rs`, its frequency grid ([`matlab_colon`]) here;
//! * `spectral_density` (`UnivariateSpectralDensity.m`, with its figures);
//! * labels of `loglinear` tables (`get_labels_transformed_vars.m`);
//! * `tex` tables of `stoch_simul` (`dyn_latex_table.m` calls of `stoch_simul.m`,
//!   `disp_th_moments.m`, `disp_moments.m`, `display_*_variance_decomposition.m`).

use anyhow::{Result, bail};
use nalgebra::{Complex, DMatrix};

use super::{OutputContext, StochSimulOptions};
use crate::graphics::{self, Color, Figure, LineStyle, Panel, Series};
use crate::linalg::Mat;
use crate::model::Model;
use crate::solver::DecisionRule;

/// The filter `get_filtered_time_series.m` applies (at most one may be set).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Filter {
    /// Demeaning only.
    None,
    /// `hp_filter = lambda` (`sample_hp_filter.m`).
    Hp(f64),
    /// `one_sided_hp_filter = lambda`.
    OneSidedHp(f64),
    /// `bandpass_filter = [high low]` (`options_.bandpass.passband`), `K` leads/lags.
    Bandpass { high: f64, low: f64, k: usize },
}

impl Filter {
    /// `get_filtered_time_series.m`'s choice; Dynare errors out when several filters are set.
    pub fn from_options(o: &StochSimulOptions) -> Result<Filter> {
        let n = [o.hp_filter != 0.0, o.one_sided_hp_filter != 0.0, o.bandpass.is_some()].iter().filter(|&&b| b).count();
        if n > 1 {
            bail!("disp_moments:: You cannot use more than one filter at the same time");
        }
        Ok(if o.hp_filter != 0.0 {
            Filter::Hp(o.hp_filter)
        } else if o.one_sided_hp_filter != 0.0 {
            Filter::OneSidedHp(o.one_sided_hp_filter)
        } else if let Some([h, l]) = o.bandpass {
            Filter::Bandpass { high: h, low: l, k: o.bandpass_k }
        } else {
            Filter::None
        })
    }

    /// Observations `disp_moments.m` drops at each end after filtering (`bandpass.K`).
    pub fn trim(&self) -> usize {
        if let Filter::Bandpass { k, .. } = self { *k } else { 0 }
    }
}

/// `add_filter_subtitle.m`.
pub fn add_filter_subtitle(title: &str, o: &StochSimulOptions) -> String {
    use super::table::num2str;
    let hp = o.hp_filter != 0.0;
    let os = o.one_sided_hp_filter != 0.0;
    match (hp, os, o.bandpass) {
        (false, false, None) => title.to_string(),
        (false, false, Some([h, l])) => format!("{title} (Bandpass filter, ({} {}))", num2str(h), num2str(l)),
        (true, false, None) => format!("{title} (HP filter, lambda = {})", num2str(o.hp_filter)),
        (false, true, None) => format!("{title} (One-sided HP filter, lambda = {})", num2str(o.one_sided_hp_filter)),
        _ => title.to_string(),
    }
}

/// Port of `one_sided_hp_filter.m` (no user initial state, no discard): `(trend, cycle)` of
/// each column of `y` (T x n).
pub fn one_sided_hp_filter(y: &Mat, lambda: f64) -> (Mat, Mat) {
    let (t, n) = y.shape();
    let q = 1.0 / lambda;
    let mut trend = Mat::from_element(t, n, f64::NAN);
    for k in 0..n {
        if t < 2 {
            break;
        }
        // x = [2 y(1) - y(2); 3 y(1) - 2 y(2)], P = 1e5 I
        let mut x = [2.0 * y[(0, k)] - y[(1, k)], 3.0 * y[(0, k)] - 2.0 * y[(1, k)]];
        let mut p = [[1e5, 0.0], [0.0, 1e5]];
        for j in 0..t {
            // kalman_update(F = [2 -1; 1 0], H = [1 0], Q = [q 0; 0 0], R = 1)
            let s = p[0][0] + 1.0;
            // K = F P H' / S
            let kk = [(2.0 * p[0][0] - p[1][0]) / s, p[0][0] / s];
            let innov = y[(j, k)] - x[0];
            let fx = [2.0 * x[0] - x[1], x[0]];
            x = [fx[0] + kk[0] * innov, fx[1] + kk[1] * innov];
            // Temp = F - K H
            let tm = [[2.0 - kk[0], -1.0], [1.0 - kk[1], 0.0]];
            // P = Temp P Temp'
            let mut tp = [[0.0; 2]; 2];
            for a in 0..2 {
                for b in 0..2 {
                    tp[a][b] = tm[a][0] * p[0][b] + tm[a][1] * p[1][b];
                }
            }
            let mut np = [[0.0; 2]; 2];
            for a in 0..2 {
                for b in 0..2 {
                    np[a][b] = tp[a][0] * tm[b][0] + tp[a][1] * tm[b][1];
                }
            }
            // P = P + Q + K R K'
            np[0][0] += q;
            for a in 0..2 {
                for b in 0..2 {
                    np[a][b] += kk[a] * kk[b];
                }
            }
            p = np;
            trend[(j, k)] = x[1];
        }
    }
    let cycle = y - &trend;
    (trend, cycle)
}

/// Port of dseries' `baxter_king_filter_.m`: the filtered observations `K+1..T-K`
/// ((T - 2K) x n). `high`/`low` are the periodicities of `options_.bandpass.passband`.
pub fn baxter_king_filter(y: &Mat, high: f64, low: f64, k: usize) -> Result<Mat> {
    // baxter_king_filter_.m checks `high >= 2` and `high <= low` only inside `if nargin<4 ||
    // isempty(K)`; get_filtered_time_series.m always passes `options_.bandpass.K`
    let (t, n) = y.shape();
    if t < 2 * k + 1 {
        bail!("dseries::baxter_king_filter: the sample is too short ({t} observations, K = {k})");
    }
    let hf = 2.0 * std::f64::consts::PI / high;
    let lf = 2.0 * std::f64::consts::PI / low;
    let pi = std::f64::consts::PI;
    let mut w = vec![0.0; k + 1];
    w[0] = (hf - lf) / pi;
    for (l, wl) in w.iter_mut().enumerate().skip(1) {
        let lp = l as f64;
        *wl = ((lp * hf).sin() - (lp * lf).sin()) / (lp * pi);
    }
    let constraint = if low > 1000.0 { 1.0 } else { 0.0 };
    let sum: f64 = w[0] + 2.0 * w[1..].iter().sum::<f64>();
    for wl in w.iter_mut() {
        *wl += (constraint - sum) / (2 * k + 1) as f64;
    }
    // weights = [flipud(w(2:K+1)); w] : lags K..1, then 0..K
    let full: Vec<f64> = w[1..].iter().rev().chain(w.iter()).copied().collect();
    let mut out = Mat::zeros(t - 2 * k, n);
    for c in 0..n {
        for tt in k..t - k {
            let mut s = 0.0;
            for (j, wj) in full.iter().enumerate() {
                s += wj * y[(tt - k + j, c)];
            }
            out[(tt - k, c)] = s;
        }
    }
    Ok(out)
}

/// `get_filtered_time_series(y, m, options_)` followed by `disp_moments.m`'s removal of the
/// `bandpass.K` NaN rows at each end (`y` is T x n, `m` the column means).
pub fn filtered_series(y: &Mat, m: &[f64], f: &Filter) -> Result<Mat> {
    Ok(match *f {
        Filter::None => {
            let mut out = y.clone();
            for (c, mc) in m.iter().enumerate() {
                out.column_mut(c).add_scalar_mut(-mc);
            }
            out
        }
        Filter::Hp(l) => super::simulate::sample_hp_filter(y, l),
        Filter::OneSidedHp(l) => one_sided_hp_filter(y, l).1,
        Filter::Bandpass { high, low, k } => baxter_king_filter(y, high, low, k)?,
    })
}

/// MATLAB's colon operator `a:d:b` (Cleve Moler's `colonop.m`: symmetric about the midpoint),
/// as used for the frequency grids of `th_autocovariances.m` / `UnivariateSpectralDensity.m`.
pub fn matlab_colon(a: f64, d: f64, b: f64) -> Vec<f64> {
    let tol = 2.0 * f64::EPSILON * a.abs().max(b.abs());
    let sig = d.signum();
    let n: i64 = if a == a.floor() && d == 1.0 {
        (b.floor() - a) as i64
    } else if a == a.floor() && d == d.floor() {
        ((b - a) / d).floor() as i64
    } else {
        let mut n = ((b - a) / d).round() as i64;
        if sig * (a + n as f64 * d - b) > tol {
            n -= 1;
        }
        n
    };
    if n < 0 {
        return Vec::new();
    }
    let mut c = a + n as f64 * d;
    if sig * (c - b) > -tol {
        c = b;
    }
    let nu = n as usize;
    let mut v = vec![0.0; nu + 1];
    for k in 0..=nu / 2 {
        v[k] = a + k as f64 * d;
        v[nu - k] = c - k as f64 * d;
    }
    if nu.is_multiple_of(2) {
        v[nu / 2] = (a + c) / 2.0;
    }
    v
}

/// The bandpass `filter_gain` of `th_autocovariances.m` / `UnivariateSpectralDensity.m` on
/// `freqs` (`passband = [highest_periodicity lowest_periodicity]`).
pub fn bandpass_gain(freqs: &[f64], high: f64, low: f64) -> Vec<f64> {
    let two_pi = 2.0 * std::f64::consts::PI;
    let high = high.max(2.0);
    freqs
        .iter()
        .map(|&w| {
            let a = w >= two_pi / low && w <= two_pi / high;
            let b = w <= -two_pi / low + two_pi && w >= -two_pi / high + two_pi;
            if a || b { 1.0 } else { 0.0 }
        })
        .collect()
}

/// The HP `filter_gain`: `4 lambda (1 - cos w)^2 / (1 + 4 lambda (1 - cos w)^2)`.
pub fn hp_gain(freqs: &[f64], lambda: f64) -> Vec<f64> {
    freqs
        .iter()
        .map(|&w| {
            let c1 = 1.0 - w.cos();
            4.0 * lambda * c1 * c1 / (1.0 + 4.0 * lambda * c1 * c1)
        })
        .collect()
}

/// `get_labels_transformed_vars.m`: `log(x)` (`\log(x)` for TeX) under `loglinear`.
pub fn transformed_labels(names: &[String], loglinear: bool, tex: bool) -> Vec<String> {
    names
        .iter()
        .map(|n| match (loglinear, tex) {
            (true, true) => format!("\\log({n})"),
            (true, false) => format!("log({n})"),
            _ => n.clone(),
        })
        .collect()
}

/// `log_variable(1:endo_nbr, ys, M_)`: logs of every variable but the auxiliary variables of
/// lagged / leaded exogenous variables (types 2 and 3).
pub fn log_variable(model: &Model, ys: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = ys.iter().map(|v| v.ln()).collect();
    for a in &model.aux_vars {
        if a.ty == 2 || a.ty == 3 {
            out[a.endo_index] = ys[a.endo_index];
        }
    }
    out
}

/// Variables `resol.m` checks for a positive steady state under `loglinear`
/// (`get_all_variables_but_lagged_leaded_exogenous`); `Err(info(2))` = sum of squares of the
/// offending values (info 26).
pub fn loglinear_check(model: &Model, ys: &[f64]) -> Result<(), f64> {
    let mut skip = vec![false; ys.len()];
    for a in &model.aux_vars {
        if a.ty == 2 || a.ty == 3 {
            skip[a.endo_index] = true;
        }
    }
    let bad: Vec<f64> = ys.iter().zip(&skip).filter(|&(&v, &s)| !s && v < 1e-16).map(|(v, _)| *v).collect();
    if bad.is_empty() { Ok(()) } else { Err(ys.iter().filter(|&&v| v < 1e-16).map(|v| v * v).sum()) }
}

/// stochastic_solvers.m, `options_.loglinear` at order 1 (`indices_lagged_leaded_exogenous_variables`):
/// `ghx`/`ghu` in logs, the aux variables of lagged/leaded exogenous variables in levels;
/// `dr.ys` is left in levels (stoch_simul logs it afterwards).
pub fn loglinear_decision_rule(model: &Model, dr: &mut DecisionRule, ys: &[f64]) {
    let mut lev = vec![false; ys.len()];
    for a in &model.aux_vars {
        if a.ty == 2 || a.ty == 3 {
            lev[a.endo_index] = true;
        }
    }
    for i in 0..dr.ghx.nrows() {
        let vi = dr.order_var[i];
        let row = if lev[vi] { 1.0 } else { 1.0 / ys[vi] };
        for j in 0..dr.ghx.ncols() {
            let vj = dr.state_var[j];
            let col = if lev[vj] { 1.0 } else { ys[vj] };
            dr.ghx[(i, j)] *= row * col;
        }
        for k in 0..dr.ghu.ncols() {
            dr.ghu[(i, k)] *= row;
        }
    }
}

/// `oo_.SpectralDensity`.
#[derive(Clone, Debug)]
pub struct SpectralDensity {
    /// `freqs` (ngrid points on [0, pi]).
    pub freqs: Vec<f64>,
    /// `density`: nvar x ngrid.
    pub density: Mat,
    /// Declaration indices of the variables (rows of `density`).
    pub ivar: Vec<usize>,
}

/// Port of `UnivariateSpectralDensity.m` (order 1).
pub fn univariate_spectral_density(
    dr: &DecisionRule,
    sigma_e: &Mat,
    ivar: &[usize],
    o: &StochSimulOptions,
) -> Result<SpectralDensity> {
    if o.one_sided_hp_filter != 0.0 {
        bail!("UnivariateSpectralDensity:: spectral density estimate not available with one-sided HP filter");
    }
    let nvar = ivar.len();
    let (nstatic, nspred) = (dr.nstatic, dr.nspred);
    let a = dr.ghx.rows(nstatic, nspred).into_owned();
    let ghu1 = dr.ghu.rows(nstatic, nspred).into_owned();
    let lyap = super::moments::LyapunovSchur::new(&a, o.qz_criterium)?;
    let mut ivar_sel = ivar.to_vec();
    if lyap.u_unit.ncols() > 0 {
        // iky = iky(find(any(abs(ghx(iky,:)*u) < schur_vec_tol, 2)))  (sic: any)
        let x = crate::linalg::mul(&dr.ghx, false, &lyap.u_unit, false);
        ivar_sel.retain(|&v| {
            let r = dr.inv_order_var[v];
            (0..x.ncols()).any(|j| x[(r, j)].abs() < o.schur_vec_tol)
        });
        if ivar_sel.len() != nvar {
            bail!(
                "UnivariateSpectralDensity: Index exceeds the number of array elements (nonstationary variables in the list)"
            );
        }
    }
    let rows: Vec<usize> = ivar_sel.iter().map(|&v| dr.inv_order_var[v]).collect();
    let ngrid = o.filtered_theoretical_moments_grid;
    let pi = std::f64::consts::PI;
    let freqs = matlab_colon(0.0, pi / (ngrid as f64 - 1.0), pi);
    let gain = if o.hp_filter == 0.0 && o.bandpass.is_none() {
        vec![1.0; freqs.len()]
    } else if let Some([h, l]) = o.bandpass {
        bandpass_gain(&freqs, h, l)
    } else {
        hp_gain(&freqs, o.hp_filter)
    };
    type C = Complex<f64>;
    let cz = |m: &Mat| DMatrix::<C>::from_fn(m.nrows(), m.ncols(), |i, j| C::new(m[(i, j)], 0.0));
    let aa = cz(&dr.ghx.select_rows(&rows));
    let bb = cz(&dr.ghu.select_rows(&rows));
    let (ac, gc, sc) = (cz(&a), cz(&ghu1), cz(sigma_e));
    let mut density = Mat::zeros(nvar, freqs.len());
    for (ig, &w) in freqs.iter().enumerate() {
        let tneg = C::new(w.cos(), -w.sin());
        let g = if nspred > 0 {
            let m = DMatrix::<C>::identity(nspred, nspred) - &ac * tneg;
            let sol = m
                .lu()
                .solve(&gc)
                .ok_or_else(|| anyhow::anyhow!("UnivariateSpectralDensity: singular I - A e^(-iw)"))?;
            &aa * sol * tneg + &bb
        } else {
            bb.clone()
        };
        let f = &g * &sc * g.adjoint();
        let scale = gain[ig] * gain[ig] / (2.0 * pi);
        for i in 0..nvar {
            density[(i, ig)] = f[(i, i)].re * scale;
        }
    }
    Ok(SpectralDensity { freqs, density, ivar: ivar_sel })
}

/// The figures of `UnivariateSpectralDensity.m` (`<fname>/graphs/SpectralDensity_<var>`).
pub fn spectral_density_figures(
    model: &Model,
    sd: &SpectralDensity,
    o: &StochSimulOptions,
    ctx: &OutputContext,
) -> Result<()> {
    if o.nograph {
        return Ok(());
    }
    let gopts = o.graph_options(&ctx.dir, &ctx.fname);
    for (i, &v) in sd.ivar.iter().enumerate() {
        let name = &model.endo_names[v];
        let fig = Figure {
            name: format!("SpectralDensity_{name}"),
            title: format!("Spectral Density of {name}."),
            rows: 1,
            cols: 1,
            panels: vec![Panel {
                title: String::new(),
                series: vec![Series::Line {
                    x: sd.freqs.clone(),
                    y: sd.density.row(i).iter().copied().collect(),
                    color: Color::Black,
                    style: LineStyle::Solid,
                    width: 2.0,
                    label: None,
                }],
                x_label: Some("0 \u{2264} \u{03c9} \u{2264} \u{03c0}".into()),
                y_label: Some("f(\u{03c9})".into()),
                ..Default::default()
            }],
        };
        graphics::save_as(&fig, &format!("SpectralDensity_{name}"), &gopts)?;
    }
    Ok(())
}

/// `dyn_latex_table(M_, options_, title, latex_title, headers, labels, values, lh, w, p)` when
/// `options_.TeX` (labels / headers as LaTeX strings).
#[allow(clippy::too_many_arguments)]
pub fn latex_table(
    o: &StochSimulOptions,
    ctx: &OutputContext,
    created_by: &str,
    title: &str,
    latex_title: &str,
    headers: &[String],
    labels: &[String],
    values: &Mat,
    lh: usize,
    w: usize,
    p: usize,
) -> Result<()> {
    if !o.tex || o.noprint {
        return Ok(());
    }
    let rows: Vec<Vec<f64>> = (0..values.nrows()).map(|i| values.row(i).iter().copied().collect()).collect();
    crate::latex::dyn_latex_table(
        &ctx.dir,
        &ctx.fname,
        title,
        latex_title,
        created_by,
        headers,
        labels,
        &rows,
        lh,
        w,
        p,
    )?;
    Ok(())
}

/// TeX names of the endogenous variables (`M_.endo_names_tex`).
pub fn endo_tex(mf_labels: Option<&crate::modfile::ModFile>, model: &Model, ivar: &[usize]) -> Vec<String> {
    ivar.iter()
        .map(|&i| match mf_labels {
            Some(mf) if i < model.orig_endo_nbr => crate::latex::endo_tex(mf, i),
            _ => model.endo_names[i].replace('_', "\\_"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colon_grid() {
        let v = matlab_colon(0.0, 1.0, 4.0);
        assert_eq!(v, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let pi = std::f64::consts::PI;
        let g = matlab_colon(0.0, 2.0 * pi / 8.0, 2.0 * pi * (1.0 - 0.5 / 8.0));
        assert_eq!(g.len(), 8);
        let s = matlab_colon(0.0, pi / 511.0, pi);
        assert_eq!(s.len(), 512);
        assert_eq!(*s.last().unwrap(), pi);
    }

    #[test]
    fn baxter_king_removes_constants() {
        let y = Mat::from_fn(60, 1, |t, _| 3.0 + 0.0 * t as f64);
        let f = baxter_king_filter(&y, 6.0, 32.0, 12).unwrap();
        assert_eq!(f.nrows(), 36);
        assert!(f.iter().all(|v| v.abs() < 1e-12));
    }
}
