// SPDX-License-Identifier: GPL-3.0-or-later
//! Data, dates, epilogue and output commands:
//!
//! - `data(file=..., first_obs=..., last_obs=..., nobs=...)` and `set_time(DATE)`: Dynare's
//!   `options_.dataset` / `options_.initial_period`, read by every later data-reading command
//!   ([`crate::dataset`]);
//! - the `epilogue` block ([`epilogue`]): `<fname>.epilogue_dynamic` / `epilogue_static`,
//!   evaluated on the smoothed variables after an estimation with `smoother`, and in the shock
//!   decompositions (`with_epilogue`);
//! - `dsample` for `rplot` (`options_.smpl`).

use std::path::Path;

use anyhow::Result;

use crate::modfile::{ModFile, Options};

pub mod decomp;
pub mod epilogue;
pub mod json;
pub mod xlsx;

/// Declarations read by the parser (`ModFile::q4`).
#[derive(Clone, Debug, Default)]
pub struct Q4Decl {
    /// `epilogue` block, in order (`M_.epilogue_names`).
    pub epilogue: Vec<epilogue::EpilogueEq>,
    /// `init2shocks(name=NAME)` blocks: (NAME, [(endogenous, exogenous)]).
    pub init2shocks: Vec<(String, Vec<(String, String)>)>,
    /// `model_local_variable` declarations: (name, TeX name, long name).
    pub local_labels: Vec<(String, String, String)>,
}

/// What keeps between commands.
#[derive(Debug, Default)]
pub struct Session {}

impl Session {
    /// Start of a run: clears Dynare's global `options_.dataset` / `options_.initial_period`
    /// and installs the default global random stream (`set_dynare_seed('default')`).
    pub fn start() -> Session {
        crate::dataset::reset();
        crate::gsa::rng::reset_global(0);
        Session {}
    }
}

/// After `estimation(smoother)`: the epilogue variables over the smoothed variables
/// (`smoothed`: T x orig_endo_nbr levels), `<fname>_epilogue_smoothed_variables.csv`.
pub fn smoothed_epilogue(
    mf: &ModFile,
    params: &[f64],
    smoothed: &crate::linalg::Mat,
    dates: Option<&[String]>,
    out_dir: &Path,
    fname: &str,
) -> Result<()> {
    if mf.q4.epilogue.is_empty() {
        return Ok(());
    }
    let n = mf.model.n_endo();
    let paths = crate::linalg::Mat::from_fn(n, smoothed.nrows(), |j, t| {
        if j < smoothed.ncols() { smoothed[(t, j)] } else { f64::NAN }
    });
    let labels: Vec<String> = match dates {
        Some(d) => d.to_vec(),
        None => (1..=smoothed.nrows()).map(|t| t.to_string()).collect(),
    };
    epilogue::write_csv(mf, params, &paths, &labels, out_dir, fname, "smoothed_variables")
}

/// After `perfect_foresight_solver` (and `simul`): the epilogue over `oo_.endo_simul`,
/// `<fname>_epilogue_endo_simul.csv` (periods labelled as `<fname>_endo_simul.csv`).
pub fn after_deterministic(
    mf: &ModFile,
    name: &str,
    det: &crate::deterministic::DetState,
    out_dir: &Path,
    fname: &str,
) -> Result<()> {
    if mf.q4.epilogue.is_empty() || !matches!(name, "perfect_foresight_solver" | "simul") {
        return Ok(());
    }
    let Some(e) = &det.endo_simul else { return Ok(()) };
    let ml = det.maximum_lag as i64;
    let labels: Vec<String> = (0..e.ncols()).map(|c| (c as i64 + 1 - ml).to_string()).collect();
    epilogue::write_csv(mf, &det.params, e, &labels, out_dir, fname, "endo_simul")
}

/// `rplot` after `dsample a b` (`options_.smpl`): rplot.m plots columns
/// `smpl(1)+maximum_lag : smpl(2)+maximum_lag` of `oo_.endo_simul` (x = `smpl(1)..smpl(2)`).
/// Returns false (nothing done) without `dsample` or for another command.
#[allow(clippy::too_many_arguments)]
pub fn rplot_dsample(
    mf: &ModFile,
    name: &str,
    o: &Options,
    vars: &[String],
    det: &crate::deterministic::DetState,
    dir: &Path,
    fname: &str,
    smpl: Option<(usize, usize)>,
) -> Result<bool> {
    use crate::graphics::{self, Color, Figure, GraphFormat, GraphOptions, LineStyle, Panel, Series};
    use crate::modfile::{OptValue, opt};
    let (Some((s1, s2)), "rplot") = (smpl, name) else { return Ok(false) };
    let e = det.endo_simul.as_ref().ok_or_else(|| anyhow::anyhow!("rplot: oo_.endo_simul is empty."))?;
    let ml = det.maximum_lag;
    if s2 + ml > e.ncols() || s1 == 0 || s1 > s2 {
        anyhow::bail!(
            "rplot: dsample {s1} {s2} is outside the simulation (Index exceeds the number of array elements)"
        );
    }
    let cols: Vec<usize> = (s1 + ml - 1..s2 + ml).collect(); // 0-based
    let xs: Vec<f64> = cols.iter().map(|&c| (c as i64 + 1 - ml as i64) as f64).collect();
    let model = &mf.model;
    let mut panel = Panel {
        title: format!("Plot of {}", vars.join(" ")),
        x_label: Some("Periods".into()),
        legend: vars.len() > 1,
        ..Default::default()
    };
    for (k, v) in vars.iter().enumerate() {
        let y: Vec<f64> = if let Some(j) = model.endo_index(v) {
            cols.iter().map(|&c| e[(j, c)]).collect()
        } else if let (Some(kx), Some(x)) = (model.exo_index(v), &det.exo_simul) {
            cols.iter().map(|&c| x[(c, kx)]).collect()
        } else {
            anyhow::bail!("rplot: One of the variables specified does not exist");
        };
        panel.series.push(Series::Line {
            x: xs.clone(),
            y,
            color: Color::Palette(k),
            style: LineStyle::Solid,
            width: 1.0,
            label: Some(v.clone()),
        });
    }
    let fig = Figure {
        name: format!("SimulatedTrajectory_{}", vars.first().map(String::as_str).unwrap_or("")),
        title: "Simulated Trajectory".into(),
        rows: 1,
        cols: 1,
        panels: vec![panel],
    };
    let formats = match opt(o, "graph_format") {
        Some(OptValue::Str(s)) => GraphFormat::parse(s).into_iter().collect(),
        Some(OptValue::List(l)) => {
            l.iter().filter_map(|v| if let OptValue::Str(s) = v { GraphFormat::parse(s) } else { None }).collect()
        }
        _ => vec![GraphFormat::Pdf],
    };
    let gopts = GraphOptions { enabled: opt(o, "nograph").is_none(), formats, dir: graphics::graph_dir(dir, fname) };
    graphics::save(&fig, fname, &gopts)?;
    println!("rplot: periods {s1} to {s2} (dsample)");
    Ok(true)
}

thread_local! {
    /// `@#echomacrovars(save)`: `options_.macrovars_line_<n>` (line, variables, functions).
    static MACROVARS: std::cell::RefCell<Vec<(usize, serde_json::Value)>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn macro_json(v: &crate::modfile::macro_proc::Value) -> serde_json::Value {
    use crate::modfile::macro_proc::Value as V;
    match v {
        V::Num(x) if x.fract() == 0.0 && x.abs() < 1e15 => serde_json::json!(*x as i64),
        V::Num(x) => serde_json::json!(x),
        V::Str(s) => serde_json::json!(s),
        V::Bool(b) => serde_json::json!(b),
        V::Array(a) | V::Tuple(a) => serde_json::Value::Array(a.iter().map(macro_json).collect()),
    }
}

/// Called by the macro processor for `@#echomacrovars(save)` at `line`.
pub fn record_macrovars(line: usize, vars: &[(String, crate::modfile::macro_proc::Value)], funcs: &[(String, String)]) {
    let mut m = serde_json::Map::new();
    for (k, v) in vars {
        m.insert(k.clone(), macro_json(v));
    }
    if !funcs.is_empty() {
        let f: serde_json::Map<String, serde_json::Value> =
            funcs.iter().map(|(k, s)| (k.clone(), serde_json::json!(s))).collect();
        m.insert("function".into(), serde_json::Value::Object(f));
    }
    MACROVARS.with(|v| v.borrow_mut().push((line, serde_json::Value::Object(m))));
}

/// `options_.macrovars_line_<n>` of the last preprocessing, written to
/// `<fname>/Output/<fname>_macrovars.json` (dynare-rust has no `options_` to hold them).
fn write_macrovars(out_dir: &Path, fname: &str) -> Result<()> {
    let saved = MACROVARS.with(|v| std::mem::take(&mut *v.borrow_mut()));
    if saved.is_empty() {
        return Ok(());
    }
    let mut m = serde_json::Map::new();
    for (line, v) in saved {
        m.insert(format!("macrovars_line_{line}"), v);
    }
    std::fs::create_dir_all(out_dir)?;
    let p = out_dir.join(format!("{fname}_macrovars.json"));
    std::fs::write(&p, serde_json::to_string_pretty(&serde_json::Value::Object(m))?)?;
    println!("@#echomacrovars(save): macro variables written to {}", p.display());
    Ok(())
}

/// Right after preprocessing: `json=...` output; returns true when the run stops
/// (`onlyjson`). Notes for the command-line options without an effect here.
pub fn after_parse(mf: &ModFile, dir: &Path, fname: &str, ro: &crate::driver::RunOptions) -> Result<bool> {
    let c = &ro.cli;
    write_macrovars(&dir.join(fname).join("Output"), fname)?;
    if let Some(d) = c.params_derivs_order {
        println!(
            "NOTE: params_derivs_order={d}: dynare-rust differentiates with respect to the parameters only where a command needs it."
        );
    }
    let Some(stage) = &c.json else {
        if c.onlyjson || c.jsonstdout || c.jsonderivsimple {
            println!("WARNING: onlyjson/jsonstdout/jsonderivsimple need the json=... option; ignored.");
        }
        return Ok(false);
    };
    let cap = crate::model::dtree::json_take().unwrap_or_default();
    json::write(
        mf,
        &cap,
        dir,
        fname,
        &json::JsonOptions { stage: stage.clone(), stdout: c.jsonstdout, derivsimple: c.jsonderivsimple },
    )?;
    Ok(c.onlyjson)
}

/// Commands run by [`command`].
pub fn handles(name: &str) -> bool {
    matches!(name, "data" | "set_time" | "epilogue" | "init2shocks")
}

/// The driver's one call for the commands of [`handles`].
pub fn command(
    mf: &ModFile,
    name: &str,
    o: &Options,
    dir: &Path,
    fname: &str,
    out_dir: &Path,
    s: &mut Session,
) -> Result<()> {
    let _ = (mf, dir, fname, out_dir, s);
    match name {
        "data" => crate::dataset::data_command(o),
        "set_time" => crate::dataset::set_time(o),
        // declarations, read where they are used (ModFile::q4)
        "epilogue" | "init2shocks" => Ok(()),
        _ => unreachable!(),
    }
}
