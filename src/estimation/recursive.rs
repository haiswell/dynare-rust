// SPDX-License-Identifier: GPL-3.0-or-later
//! Recursive and rolling-window estimation (port of the sample loop of
//! `estimation/dynare_estimation.m`).
//!
//! `nobs = [a b ...]` (or `[a:b]`) estimates the model once per sample size, `first_obs = [...]`
//! once per first observation. Sample `k` runs as an ordinary estimation in the folder
//! `<dirname>_<k>` (`M_.dname`), so its results (`oo_recursive_{k}`) are that folder's
//! `Output/<fname>_results.json`, forecasts and MCMC files. After the first sample the mode
//! file of the first sample is the starting point of the others (`options_.mode_file`, unless
//! `mode_compute = 0`). With `forecast` and several `nobs`, the forecasts of every sample are
//! gathered in `oo_.RecursiveForecast` (`<fname>_recursive_forecast.json` in the last sample's
//! `Output` folder, where Dynare saves `oo_`) and plotted (`<fname>_RecursiveForecasts_<k>`).

use anyhow::{Result, bail};
use serde_json::{Map, Value, json};

use crate::forecast::{Forecast, save_figures};
use crate::graphics::{Color, Figure, GraphOptions, LineStyle, Panel, Series, pltorg_full};
use crate::linalg::Mat;
use crate::modfile::{OptValue, Options, opt};

/// The samples of a recursive (`nobs`) or rolling-window (`first_obs`) estimation.
pub struct Recursive {
    /// Sorted sample sizes or first observations.
    pub keys: Vec<usize>,
    /// `nobs` vector (true) or `first_obs` vector.
    pub by_nobs: bool,
    /// `options_.dirname` (the .mod file name by default).
    dname: String,
    /// `options_.mode_compute ~= 0`.
    mode_compute: bool,
    /// `options_.recursive_estimation_restart`.
    restart: usize,
}

fn int_vector(o: &Options, k: &str) -> Vec<usize> {
    let mut v: Vec<usize> = match opt(o, k) {
        Some(OptValue::Vec(v)) => v.iter().map(|&x| x as usize).collect(),
        Some(OptValue::Num(x)) => vec![*x as usize],
        _ => vec![],
    };
    v.sort_unstable();
    v
}

impl Recursive {
    /// `None` for an estimation on a single sample.
    pub fn new(o: &Options, fname: &str) -> Result<Option<Recursive>> {
        let nobs = int_vector(o, "nobs");
        let first_obs = int_vector(o, "first_obs");
        // options_.nobs / options_.first_obs default to NaN (one element)
        let (nn, nf) = (nobs.len().max(1), first_obs.len().max(1));
        if nn != 1 && nf != 1 {
            bail!("You cannot simultaneously do rolling window and recursive estimation");
        }
        if nn == 1 && nf == 1 {
            return Ok(None);
        }
        let dname = match opt(o, "dirname") {
            Some(OptValue::Str(d)) if !d.is_empty() => d.clone(),
            _ => fname.to_string(),
        };
        let num = |k: &str| match opt(o, k) {
            Some(OptValue::Num(v)) => Some(*v),
            _ => None,
        };
        Ok(Some(Recursive {
            by_nobs: nn > 1,
            keys: if nn > 1 { nobs } else { first_obs },
            dname,
            mode_compute: num("mode_compute").unwrap_or(5.0) != 0.0,
            restart: num("recursive_estimation_restart").unwrap_or(0.0) as usize,
        }))
    }

    /// `M_.dname` of sample `i`.
    pub fn dname(&self, i: usize) -> String {
        format!("{}_{}", self.dname, self.keys[i])
    }

    /// The estimation runs of sample `i`: the estimation, then `recursive_estimation_restart`
    /// reruns. Each is the command's options with this sample's `nobs` / `first_obs` and
    /// `dirname`, and from the first sample's estimation on the first sample's mode file.
    pub fn runs(&self, o: &Options, i: usize, fname: &str) -> Vec<Options> {
        let key = if self.by_nobs { "nobs" } else { "first_obs" };
        let mut base: Options = o.iter().filter(|(k, _)| k != key && k != "dirname").cloned().collect();
        base.push((key.into(), OptValue::Num(self.keys[i] as f64)));
        base.push(("dirname".into(), OptValue::Str(self.dname(i))));
        let with_mode = |mut b: Options| {
            if self.mode_compute {
                b.retain(|(k, _)| k != "mode_file");
                b.push(("mode_file".into(), OptValue::Str(format!("{}/Output/{fname}_mode", self.dname(0)))));
            }
            b
        };
        let mut v = vec![if i == 0 { base.clone() } else { with_mode(base.clone()) }];
        for _ in 0..self.restart {
            v.push(with_mode(base.clone()));
        }
        v
    }
}

/// What one sample leaves for `oo_.RecursiveForecast`.
pub struct SampleForecast {
    /// `oo_recursive_{k}.MeanForecast` after an MCMC, `oo_recursive_{k}.forecast` otherwise.
    pub mean: Forecast,
    /// `oo_recursive_{k}.PointForecast` (after an MCMC).
    pub point: Option<Forecast>,
}

/// `oo_.RecursiveForecast` and its figures (`nobs` vector and `forecast > 0`).
/// `var_list` is the estimation's (all endogenous variables when empty), `varobs` the observed
/// variables and `rawdata` the last sample's data (`dataset_info.rawdata`, nobs x nvarobs).
#[allow(clippy::too_many_arguments)]
pub fn recursive_forecast(
    rec: &Recursive,
    samples: &[SampleForecast],
    var_list: &[String],
    endo_names: &[String],
    varobs: &[String],
    rawdata: &Mat,
    horizon: usize,
    out_dir: &std::path::Path,
    fname: &str,
    graphs: &GraphOptions,
) -> Result<()> {
    let vars: Vec<String> = if var_list.is_empty() { endo_names.to_vec() } else { var_list.to_vec() };
    let nobs = &rec.keys;
    let mcmc = samples.iter().all(|s| s.point.is_some());
    // oo_.RecursiveForecast.<field>.<var>: one row per sample
    let fields: &[&str] =
        if mcmc { &["Mean", "HPDinf", "HPDsup", "HPDTotalinf", "HPDTotalsup"] } else { &["Mean", "HPDinf", "HPDsup"] };
    let row = |f: &Forecast, m: &Mat, v: &str| -> Result<Vec<f64>> {
        match f.names.iter().position(|n| n == v) {
            Some(j) => Ok(m.column(j).iter().copied().collect()),
            None => bail!("RecursiveForecast: no forecast of {v}"),
        }
    };
    let mut out: Vec<Map<String, Value>> = vec![Map::new(); fields.len()];
    let mut rows: Vec<Vec<Vec<Vec<f64>>>> = Vec::new(); // var -> field -> sample -> horizon
    for v in &vars {
        let mut per_field = vec![Vec::new(); fields.len()];
        for s in samples {
            per_field[0].push(row(&s.mean, &s.mean.mean, v)?);
            per_field[1].push(row(&s.mean, &s.mean.lower, v)?);
            per_field[2].push(row(&s.mean, &s.mean.upper, v)?);
            if let Some(p) = s.point.as_ref().filter(|_| mcmc) {
                per_field[3].push(row(p, &p.lower, v)?);
                per_field[4].push(row(p, &p.upper, v)?);
            }
        }
        for (k, m) in out.iter_mut().enumerate() {
            m.insert(v.clone(), json!(per_field[k]));
        }
        rows.push(per_field);
    }
    let mut rf = Map::new();
    for (f, m) in fields.iter().zip(out) {
        rf.insert((*f).into(), Value::Object(m));
    }
    std::fs::create_dir_all(out_dir)?;
    let path = out_dir.join(format!("{fname}_recursive_forecast.json"));
    let dirs: Vec<String> = (0..nobs.len()).map(|i| rec.dname(i)).collect();
    std::fs::write(
        &path,
        serde_json::to_string_pretty(
            &json!({"nobs": nobs, "dname": dirs, "horizon": horizon, "RecursiveForecast": rf}),
        )?,
    )?;
    crate::progress::note(format!("Recursive forecasts (oo_.RecursiveForecast) written to {}", path.display()));

    // figures "Out of sample forecasts (k)"
    if !graphs.enabled {
        return Ok(());
    }
    let gend = rawdata.nrows();
    let time_offset = 3.min(gend.saturating_sub(1));
    let (n1, nend) = (nobs[0], nobs[nobs.len() - 1]);
    let k = time_offset + (nend - n1 + horizon).min(gend.saturating_sub(n1));
    let (_, nr, nc, _, _, nstar) = pltorg_full(vars.len());
    let line = |x: Vec<f64>, y: Vec<f64>, color: Color, style: LineStyle, width: f64| Series::Line {
        x,
        y,
        color,
        style,
        width,
        label: None,
    };
    let mut figs: Vec<Figure> = Vec::new();
    for (i, v) in vars.iter().enumerate() {
        if i % nstar == 0 {
            let p = figs.len() + 1;
            figs.push(Figure {
                name: format!("RecursiveForecasts_{p}"),
                title: format!("Out of sample forecasts ({p})"),
                rows: nr,
                cols: nc,
                panels: vec![],
            });
        }
        let mut series = Vec::new();
        if let Some(k2) = varobs.iter().position(|o| o == v).filter(|_| k <= gend) {
            // plot(nobs(1)-offsetx+1:nobs(1)-offsetx+k, data2(end-k+1:end,k2))
            let x: Vec<f64> = (0..k).map(|t| (n1 + t + 1) as f64 - 3.0).collect();
            let y: Vec<f64> = (gend - k..gend).map(|t| rawdata[(t, k2)]).collect();
            series.push(line(x, y, Color::Black, LineStyle::Solid, 2.0));
        }
        for (j, &nj) in nobs.iter().enumerate() {
            let x: Vec<f64> = (1..=horizon).map(|h| (nj + h) as f64).collect();
            let r = &rows[i];
            series.push(line(x.clone(), r[0][j].clone(), Color::Blue, LineStyle::Solid, 2.0));
            series.push(line(x.clone(), r[1][j].clone(), Color::Green, LineStyle::Dashed, 1.5));
            series.push(line(x.clone(), r[2][j].clone(), Color::Green, LineStyle::Dashed, 1.5));
            if mcmc {
                series.push(line(x.clone(), r[3][j].clone(), Color::Red, LineStyle::Dashed, 1.5));
                series.push(line(x, r[4][j].clone(), Color::Red, LineStyle::Dashed, 1.5));
            }
        }
        figs.last_mut().unwrap().panels.push(Panel { title: v.clone(), series, ..Default::default() });
    }
    save_figures(&figs, fname, graphs);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_and_runs() {
        let o: Options = vec![
            ("first_obs".into(), OptValue::Num(8.0)),
            ("nobs".into(), OptValue::Vec(vec![79.0, 76.0])),
            ("recursive_estimation_restart".into(), OptValue::Num(1.0)),
        ];
        let r = Recursive::new(&o, "m").unwrap().unwrap();
        assert_eq!(r.keys, [76, 79]);
        let runs = r.runs(&o, 1, "m");
        assert_eq!(runs.len(), 2);
        assert_eq!(opt(&runs[0], "nobs"), Some(&OptValue::Num(79.0)));
        assert_eq!(opt(&runs[0], "dirname"), Some(&OptValue::Str("m_79".into())));
        assert_eq!(opt(&runs[0], "mode_file"), Some(&OptValue::Str("m_76/Output/m_mode".into())));
        assert!(opt(&r.runs(&o, 0, "m")[0], "mode_file").is_none());
        let both: Options =
            vec![("first_obs".into(), OptValue::Vec(vec![1.0, 2.0])), ("nobs".into(), OptValue::Vec(vec![3.0, 4.0]))];
        assert!(Recursive::new(&both, "m").is_err());
        assert!(Recursive::new(&o[..1].to_vec(), "m").unwrap().is_none());
    }
}
