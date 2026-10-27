// SPDX-License-Identifier: GPL-3.0-or-later
//! The `bvar_density`, `bvar_forecast` and `bvar_irf` commands: option handling (Dynare's
//! global `options_`), data loading, console output, result files and figures.
//!
//! Result files (`<dir>/<fname>/Output/`):
//! - `<fname>_bvar_density.json`: `{"log_marginal_data_density": [..per lag..], "posterior":
//!   [{"df", "S", "XXi", "PhiHat"}..], "prior": [..]}` (`oo_.bvar`), matrices as row lists;
//! - `<fname>_bvar_forecast_<nlags>.json`: `oo_.bvar.forecast` (`with_shocks`/`no_shock` ->
//!   `Mean`/`Median`/`Var`/`HPDsup`/`HPDinf` -> variable -> horizon values, `rmse`);
//! - `<fname>_bvar_irf_<nlags>.json`: `oo_.bvar.irf` (`Mean`/`Median`/`Var`/`Upper_bound`/
//!   `Lower_bound` -> varobs{a} -> varobs{b} -> horizon values).

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};

use super::{BvarOptions, ChaChaNormals, Identification, density, forecast, irf, toolbox};
use crate::data::{DataOptions, Table};
use crate::graphics::{self, Color, Figure, GraphFormat, GraphOptions, LineStyle, Panel, Series};
use crate::linalg::Mat;
use crate::modfile::{Command, ModFile, OptValue, Options, RAW_ARGS_KEY};

/// The BVAR commands handled here.
pub fn handles(name: &str) -> bool {
    matches!(name, "bvar_density" | "bvar_forecast" | "bvar_irf")
}

/// State carried between commands: `bvar.toolbox` sets `options_.nobs` permanently the first
/// time it runs without one (`set_default_option`), so later commands keep that value.
#[derive(Clone, Debug, Default)]
pub struct Session {
    /// (index of the command that set it, value)
    sticky_nobs: Option<(usize, usize)>,
}

pub fn command_parts(c: &Command) -> (&str, &Options) {
    match c {
        Command::Steady(o) => ("steady", o),
        Command::Check(o) => ("check", o),
        Command::StochSimul(o, _) => ("stoch_simul", o),
        Command::Estimation(o, _) => ("estimation", o),
        Command::Other(n, o) => (n.as_str(), o),
        // commands of other streams: their options also land in Dynare's global options_
        Command::Forecast(o, _) => ("forecast", o),
        Command::ShockDecomposition(o, _) => ("shock_decomposition", o),
        Command::CalibSmoother(o, _) => ("calib_smoother", o),
        Command::Deterministic(n, o, _) | Command::Policy(n, o, _) => (n.as_str(), o),
        // options_.occbin is separate from the options BVAR reads; no options otherwise
        Command::Occbin(_) => ("occbin", &NO_OPTIONS),
        Command::ShockGroups(..) => ("shock_groups", &NO_OPTIONS),
        Command::Endval(_) => ("endval", &NO_OPTIONS),
        Command::Mom(crate::mom::MomCommand::MethodOfMoments(o)) => ("method_of_moments", o),
        Command::Mom(_) => ("matched_moments", &NO_OPTIONS),
        Command::ConditionalForecastPaths(_) => ("conditional_forecast_paths", &NO_OPTIONS),
        Command::ConditionalForecast(o) => ("conditional_forecast", o),
        Command::PlotConditionalForecast(o, _) => ("plot_conditional_forecast", o),
        Command::RealtimeShockDecomposition(o, _) => ("realtime_shock_decomposition", o),
        Command::PlotShockDecomposition(o, _) => ("plot_shock_decomposition", o),
        Command::InitialConditionDecomposition(o, _) => ("initial_condition_decomposition", o),
        Command::SqueezeShockDecomposition(_) => ("squeeze_shock_decomposition", &NO_OPTIONS),
    }
}

static NO_OPTIONS: Options = Vec::new();

fn as_num(v: &OptValue) -> Option<f64> {
    match v {
        OptValue::Num(x) => Some(*x),
        OptValue::Flag => Some(1.0),
        OptValue::Str(s) => s.parse().ok(),
        _ => None,
    }
}

fn as_flag(v: &OptValue) -> bool {
    as_num(v).is_some_and(|x| x != 0.0)
}

impl BvarOptions {
    /// The options in force at command `upto` (inclusive): every option of every earlier
    /// command, as Dynare's global `options_` would hold them.
    pub fn from_commands(commands: &[Command], upto: usize) -> (BvarOptions, Option<usize>) {
        let mut o = BvarOptions::default();
        let mut nobs_at = None;
        for (ci, c) in commands[..=upto].iter().enumerate() {
            let (name, opts) = command_parts(c);
            let bvar_cmd = name.starts_with("bvar_");
            for (k, v) in opts {
                let n = as_num(v);
                match k.as_str() {
                    "datafile" => {
                        if let OptValue::Str(s) = v {
                            o.datafile = Some(s.clone())
                        }
                    }
                    "first_obs" => {
                        if let Some(x) = n {
                            o.first_obs = Some(x as usize)
                        }
                    }
                    "xls_sheet" => {
                        o.xls.0 = match v {
                            OptValue::Num(k) => Some(format!("{k}")),
                            OptValue::Str(s) => Some(s.clone()),
                            _ => None,
                        }
                    }
                    "xls_range" => {
                        if let OptValue::Str(s) = v {
                            o.xls.1 = Some(s.clone())
                        }
                    }
                    "nobs" => {
                        if let OptValue::Num(x) = v {
                            o.nobs = Some(*x as usize);
                            nobs_at = Some(ci);
                        }
                    }
                    "presample" => o.presample = n.unwrap_or(0.0) as usize,
                    "prefilter" => o.prefilter = as_flag(v),
                    "noconstant" => o.noconstant = as_flag(v),
                    "loglinear" => o.loglinear = as_flag(v),
                    "logdata" => o.logdata = as_flag(v),
                    "bvar_prior_tau" => o.tau = n.unwrap_or(o.tau),
                    "bvar_prior_decay" => o.decay = n.unwrap_or(o.decay),
                    "bvar_prior_lambda" => o.lambda = n.unwrap_or(o.lambda),
                    "bvar_prior_mu" => o.mu = n.unwrap_or(o.mu),
                    "bvar_prior_omega" => o.omega = n.unwrap_or(o.omega),
                    "bvar_prior_flat" => o.flat = as_flag(v),
                    "bvar_prior_train" => o.train = n.unwrap_or(0.0) as usize,
                    "forecast" => o.forecast = n.unwrap_or(0.0) as usize,
                    "bvar_replic" => o.replic = n.unwrap_or(o.replic as f64) as usize,
                    "conf_sig" if bvar_cmd => o.conf_sig = n.unwrap_or(o.conf_sig),
                    "irf" => o.irf = n.unwrap_or(o.irf as f64) as usize,
                    _ => {}
                }
            }
            // dynare_estimation_init.m: `if isnan(options_.first_obs) options_.first_obs=1; end`,
            // kept in the global options_ by estimation and calib_smoother (evaluate_smoother)
            if matches!(name, "estimation" | "calib_smoother") && o.first_obs.is_none() {
                o.first_obs = Some(1);
            }
        }
        (o, nobs_at)
    }
}

/// Raw trailing tokens of a command (`bvar_density 4;` -> ["4"]).
pub(crate) fn raw_args(o: &Options) -> Vec<String> {
    match o.iter().find(|(k, _)| k == RAW_ARGS_KEY) {
        Some((_, OptValue::List(v))) => v
            .iter()
            .filter_map(|x| match x {
                OptValue::Str(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn graph_options(commands: &[Command], upto: usize, dir: &Path, fname: &str) -> GraphOptions {
    let mut enabled = true;
    let mut formats = vec![GraphFormat::Pdf];
    for c in &commands[..=upto] {
        let (_, opts) = command_parts(c);
        for (k, v) in opts {
            match (k.as_str(), v) {
                ("nograph", _) => enabled = false,
                ("graph", _) => enabled = true,
                ("graph_format", OptValue::Str(s)) => formats = GraphFormat::parse(s).into_iter().collect(),
                ("graph_format", OptValue::List(l)) => {
                    formats = l
                        .iter()
                        .filter_map(|x| match x {
                            OptValue::Str(s) => GraphFormat::parse(s),
                            _ => None,
                        })
                        .collect()
                }
                _ => {}
            }
        }
    }
    GraphOptions { enabled, formats, dir: graphics::graph_dir(dir, fname) }
}

/// Port of `read_variables.m` (also used by `sbvar` and the `ms_*` commands): the datafile
/// `f`, relative to the folder of the .mod file, is looked for with the extensions `.m`, `.mat`,
/// `.xls`, `.xlsx` then `.csv` when it has none (`options_.xls_sheet`/`xls_range` apply to
/// spreadsheets). Returns every observation of the variables `names`, in that order.
pub fn read_variables(dir: &Path, f: &str, names: &[String], xls: &DataOptions) -> Result<Mat> {
    let given = dir.join(f);
    let path = match given.extension() {
        Some(_) => given,
        None => {
            let base = given.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            ["m", "mat", "xls", "xlsx", "csv"]
                .iter()
                .map(|e| given.with_file_name(format!("{base}.{e}")))
                .find(|p| p.is_file())
                .ok_or_else(|| anyhow!("Can't find datafile: {base}.{{m,mat,xls,xlsx,csv}}"))?
        }
    };
    let fullname = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| f.to_string());
    if !path.is_file() {
        bail!("Can't find datafile: {fullname}");
    }
    let ext = path.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
    if !matches!(ext.as_str(), "m" | "mat" | "xls" | "xlsx" | "csv") {
        bail!("Unsupported extension for datafile: .{ext}");
    }
    let table = Table::read(&path, xls)?;
    let cols = names
        .iter()
        .map(|v| table.column(v.trim()).ok_or_else(|| anyhow!("{} not found in {fullname}", v.trim())))
        .collect::<Result<Vec<usize>>>()?;
    let data = Mat::from_fn(table.values.nrows(), cols.len(), |i, j| table.values[(i, cols[j])]);
    println!("Loading {} observations from {fullname}", data.nrows());
    Ok(data)
}

/// Reads the whole dataset (all rows, columns in `varobs` order).
pub fn load_dataset(dir: &Path, o: &BvarOptions, varobs: &[String]) -> Result<Mat> {
    let f = o.datafile.as_deref().ok_or_else(|| anyhow!("bvar: the datafile option is missing"))?;
    if varobs.is_empty() {
        bail!("bvar: varobs must be declared");
    }
    let xls = DataOptions { xls_sheet: o.xls.0.clone(), xls_range: o.xls.1.clone() };
    read_variables(dir, f, varobs, &xls)
}

fn rows_json(m: &Mat) -> Value {
    json!((0..m.nrows()).map(|i| m.row(i).iter().copied().collect::<Vec<f64>>()).collect::<Vec<_>>())
}

fn write_json(out_dir: &Path, name: &str, v: &Value) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir)?;
    let p = out_dir.join(name);
    std::fs::write(&p, serde_json::to_string_pretty(v)?)?;
    Ok(p)
}

/// Runs one BVAR command `c` of `mf` (driver entry point).
pub fn run_command(mf: &ModFile, c: &Command, dir: &Path, fname: &str, seed: u64, session: &mut Session) -> Result<()> {
    let k = mf
        .commands
        .iter()
        .position(|x| std::ptr::eq(x, c))
        .ok_or_else(|| anyhow!("bvar: command not found in the mod file"))?;
    let (name, copts) = command_parts(c);
    let (mut o, nobs_at) = BvarOptions::from_commands(&mf.commands, k);
    if let Some((j, v)) = session.sticky_nobs
        && nobs_at.is_none_or(|i| i < j)
    {
        o.nobs = Some(v);
    }
    let args = raw_args(copts);
    let out_dir = dir.join(fname).join("Output");
    let gopts = graph_options(&mf.commands, k, dir, fname);
    let dataset = load_dataset(dir, &o, &mf.varobs)?;
    if let (None, Some(first_obs)) = (o.nobs, o.first_obs) {
        session.sticky_nobs = Some((k, (dataset.nrows() + 1).saturating_sub(first_obs)));
    }
    let varobs = &mf.varobs;
    let parse_lags = |s: Option<&String>| -> Result<usize> {
        let s = s.ok_or_else(|| anyhow!("{name}: the number of lags is missing"))?;
        s.trim().parse::<f64>().map(|x| x as usize).map_err(|_| anyhow!("{name}: bad number of lags '{s}'"))
    };
    match name {
        "bvar_density" => {
            let maxnlags = parse_lags(args.iter().find(|a| a.parse::<f64>().is_ok()))?;
            let res = density(&dataset, maxnlags, &o)?;
            let mut lmdd = Vec::new();
            for (p, (_, d)) in res.iter().enumerate() {
                println!("\nThe marginal log density of the BVAR({}) model is equal to {:10.4}\n", p + 1, d);
                lmdd.push(*d);
            }
            let niw = |n: &super::NiW| json!({"df": n.df, "S": rows_json(&n.s), "XXi": rows_json(&n.xxi), "PhiHat": rows_json(&n.phi_hat)});
            let v = json!({
                "varobs": varobs,
                "log_marginal_data_density": lmdd,
                "posterior": res.iter().map(|(t, _)| niw(&t.posterior)).collect::<Vec<_>>(),
                "prior": res.iter().map(|(t, _)| niw(&t.prior)).collect::<Vec<_>>(),
            });
            write_json(&out_dir, &format!("{fname}_bvar_density.json"), &v)?;
        }
        "bvar_forecast" => {
            let nlags = parse_lags(args.iter().find(|a| a.parse::<f64>().is_ok()))?;
            let tb = toolbox(&dataset, nlags, &o)?;
            let mut src = ChaChaNormals::new(seed);
            let r = forecast(&tb, nlags, &o, &mut src)?;
            if r.explosive > 0 {
                println!("\nSome of the VAR models sampled from the posterior distribution");
                println!("were found to be explosive ({} percent).\n", r.explosive as f64 / o.replic as f64);
            }
            if let Some(rmse) = &r.rmse {
                println!("RMSE of BVAR({nlags}):");
                for (v, e) in varobs.iter().zip(rmse) {
                    println!("{v}: {e:10.4}");
                }
            }
            let stats = |s: &super::ForecastStats| {
                let per = |m: &Mat| {
                    let mut o = serde_json::Map::new();
                    for (j, v) in varobs.iter().enumerate() {
                        o.insert(v.clone(), json!(m.column(j).iter().copied().collect::<Vec<f64>>()));
                    }
                    Value::Object(o)
                };
                json!({"Mean": per(&s.mean), "Median": per(&s.median), "Var": per(&s.var), "HPDsup": per(&s.hpd_sup), "HPDinf": per(&s.hpd_inf)})
            };
            let mut v = json!({"nlags": nlags, "with_shocks": stats(&r.with_shocks), "no_shock": stats(&r.no_shock), "explosive_draws": r.explosive});
            if let Some(rmse) = &r.rmse {
                v["rmse"] = json!(
                    varobs
                        .iter()
                        .cloned()
                        .zip(rmse.iter().map(|&e| json!(e)))
                        .collect::<serde_json::Map<String, Value>>()
                );
            }
            write_json(&out_dir, &format!("{fname}_bvar_forecast_{nlags}.json"), &v)?;
            for fig in forecast_figures(&r, varobs) {
                graphics::save(&fig, fname, &gopts)?;
            }
        }
        "bvar_irf" => {
            // bvar_irf(NLAGS[, 'Cholesky'|'SquareRoot']): the arguments are parsed as option keys
            let mut items: Vec<String> =
                copts.iter().filter(|(k, _)| k != RAW_ARGS_KEY).map(|(k, _)| k.clone()).collect();
            items.extend(args);
            let nlags = parse_lags(items.iter().find(|a| a.parse::<f64>().is_ok()))?;
            let ident = match items.iter().find(|a| a.parse::<f64>().is_err()) {
                Some(s) => Identification::parse(s.trim_matches(|c| c == '\'' || c == '"'))?,
                None => Identification::Cholesky,
            };
            let tb = toolbox(&dataset, nlags, &o)?;
            let mut src = ChaChaNormals::new(seed);
            let r = irf(&tb, nlags, ident, &o, &mut src)?;
            if r.explosive > 0 {
                println!("\nSome of the VAR models sampled from the posterior distribution");
                println!("were found to be explosive ({} samples).\n", r.explosive);
            }
            let ny = varobs.len();
            let stat = |s: &[Mat]| {
                let mut out = serde_json::Map::new();
                for a in 0..ny {
                    let mut inner = serde_json::Map::new();
                    for b in 0..ny {
                        inner.insert(varobs[b].clone(), json!(s.iter().map(|m| m[(a, b)]).collect::<Vec<f64>>()));
                    }
                    out.insert(varobs[a].clone(), Value::Object(inner));
                }
                Value::Object(out)
            };
            let v = json!({
                "nlags": nlags, "identification": format!("{ident:?}"),
                "Mean": stat(&r.mean), "Median": stat(&r.median), "Var": stat(&r.var),
                "Upper_bound": stat(&r.upper), "Lower_bound": stat(&r.lower), "explosive_draws": r.explosive,
            });
            write_json(&out_dir, &format!("{fname}_bvar_irf_{nlags}.json"), &v)?;
            for fig in irf_figures(&r, varobs) {
                graphics::save(&fig, fname, &gopts)?;
            }
        }
        _ => bail!("bvar: unknown command {name}"),
    }
    Ok(())
}

/// `bvar.graph_init` grid.
fn graph_grid(nplot: usize) -> (usize, usize) {
    match nplot {
        1 => (1, 1),
        2 => (2, 1),
        3 => (3, 1),
        4 => (2, 2),
        5 | 6 => (2, 3),
        7 | 8 => (2, 4),
        n => (n.div_ceil(3).min(3), n.min(3)),
    }
}

/// `BVAR_forecast_<nlags>`: per variable the median and the bands without shocks (blue, green)
/// and the bands with shocks (red), as `bvar.forecast`.
pub fn forecast_figures(r: &super::ForecastResult, varobs: &[String]) -> Vec<Figure> {
    let ny = varobs.len();
    let (nr, nc) = graph_grid(ny);
    let per = nr * nc;
    let h = r.no_shock.mean.nrows();
    let x: Vec<f64> = (1..=h).map(|t| t as f64).collect();
    let line = |m: &Mat, j: usize, color| Series::Line {
        x: x.clone(),
        y: m.column(j).iter().copied().collect(),
        color,
        style: LineStyle::Solid,
        width: 1.0,
        label: None,
    };
    let panels: Vec<Panel> = (0..ny)
        .map(|j| Panel {
            title: varobs[j].clone(),
            series: vec![
                line(&r.no_shock.median, j, Color::Blue),
                line(&r.no_shock.hpd_sup, j, Color::Green),
                line(&r.no_shock.hpd_inf, j, Color::Green),
                line(&r.with_shocks.hpd_sup, j, Color::Red),
                line(&r.with_shocks.hpd_inf, j, Color::Red),
            ],
            ..Default::default()
        })
        .collect();
    let nfig = ny.div_ceil(per);
    panels
        .chunks(per)
        .enumerate()
        .map(|(i, p)| Figure {
            name: if nfig == 1 {
                format!("BVAR_forecast_{}", r.nlags)
            } else {
                format!("BVAR_forecast_{}_{}", r.nlags, i + 1)
            },
            title: format!("BVAR forecasts (nlags = {})", r.nlags),
            rows: nr,
            cols: nc,
            panels: p.to_vec(),
        })
        .collect()
}

/// One figure per shock (`bvar.irf`): HPD band in gray, median in black.
pub fn irf_figures(r: &super::IrfResult, varobs: &[String]) -> Vec<Figure> {
    let ny = varobs.len();
    let nc = (ny as f64).sqrt().floor().max(1.0) as usize;
    let nr = ny.div_ceil(nc);
    let h = r.mean.len();
    let x: Vec<f64> = (1..=h).map(|t| t as f64).collect();
    (0..ny)
        .map(|shock| Figure {
            name: format!("BVAR_irf_{}_shock{}", r.nlags, shock + 1),
            title: format!("Posterior BVAR Impulse Responses (shock in equation {}).", shock + 1),
            rows: nr,
            cols: nc,
            panels: (0..ny)
                .map(|v| Panel {
                    title: varobs[v].clone(),
                    series: vec![
                        Series::Band {
                            x: x.clone(),
                            lower: r.lower.iter().map(|m| m[(shock, v)]).collect(),
                            upper: r.upper.iter().map(|m| m[(shock, v)]).collect(),
                            color: Color::Gray,
                            alpha: 0.5,
                            label: None,
                        },
                        Series::Line {
                            x: x.clone(),
                            y: r.median.iter().map(|m| m[(shock, v)]).collect(),
                            color: Color::Black,
                            style: LineStyle::Solid,
                            width: 2.0,
                            label: None,
                        },
                    ],
                    ..Default::default()
                })
                .collect(),
        })
        .collect()
}
