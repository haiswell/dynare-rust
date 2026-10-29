// SPDX-License-Identifier: GPL-3.0-or-later
//! Shock-decomposition extras: `with_epilogue` (epilogue rows appended to the
//! stored decompositions, `oo_.shock_decomposition_info.epilogue_steady_state`), and the
//! `plot_shock_decomposition` options `init2shocks`, `steadystate`, `write_xls`, `colormap`,
//! `screen_shocks`, `interactive` (ports of the corresponding parts of
//! `plot_shock_decomposition.m`, `graph_decomp.m` and `WriteShockDecomp2Excel.m`).

use std::path::Path;

use anyhow::{Result, anyhow, bail};

use crate::forecast::commands::EstimationState;
use crate::linalg::Mat;
use crate::modfile::{ModFile, OptValue, Options, opt};

/// What keeps in `oo_` next to the decompositions (`DecompositionState::q4`).
#[derive(Clone, Debug, Default)]
pub struct DecompExtra {
    /// `oo_.shock_decomposition_info.epilogue_steady_state` (`[dr.ys; epilogue ss]`).
    pub epilogue_steady_state: Option<Vec<f64>>,
    /// `oo_.dr.ys` of the last decomposition (the steady state of the `steadystate` option).
    pub ys: Option<Vec<f64>>,
}

/// `M_.params` after the last estimation (posterior mean or mode), else the calibration.
pub fn current_params(mf: &ModFile, st: Option<&EstimationState>) -> Vec<f64> {
    match st {
        Some(s) => s.dsge.set_all_parameters(s.current()).0,
        None => mf.model.params.clone(),
    }
}

/// Whether the command asks for the epilogue (`with_epilogue`, or `options_.shock_decomp.with_epilogue`).
pub fn with_epilogue(mf: &ModFile, o: &Options) -> Result<bool> {
    let on = matches!(opt(o, "with_epilogue"), Some(OptValue::Flag))
        || matches!(opt(o, "with_epilogue"), Some(OptValue::Num(v)) if *v != 0.0);
    if on && mf.q4.epilogue.is_empty() {
        bail!("the 'with_epilogue' option cannot be specified when there is no 'epilogue' block");
    }
    Ok(on)
}

/// `epilogue_shock_decomposition(z, M_, oo_)` on the per-variable matrices `z` (declaration
/// order, (ncomp x T) each): returns `z` with the epilogue rows appended, and records the
/// epilogue steady state.
pub fn append_epilogue(
    mf: &ModFile,
    params: &[f64],
    ys: &[f64],
    z: &[Mat],
    extra: &mut DecompExtra,
) -> Result<Vec<Mat>> {
    let (rows, yss) = super::epilogue::shock_decomposition(mf, params, ys, z)?;
    if extra.epilogue_steady_state.is_none() {
        extra.epilogue_steady_state = Some(ys.iter().copied().chain(yss).collect());
    }
    Ok(z.iter().cloned().chain(rows).collect())
}

/// `var_list` of a plot command as rows of the stored decompositions: endogenous variables
/// (declaration index), then epilogue variables (`n_endo + k`).
pub fn var_indices(mf: &ModFile, vars: &[String]) -> Result<(Vec<usize>, Vec<String>)> {
    if vars.is_empty() {
        let n = mf.model.orig_endo_nbr;
        return Ok(((0..n).collect(), mf.model.endo_names[..n].to_vec()));
    }
    let n = mf.model.n_endo();
    let idx = vars
        .iter()
        .map(|v| {
            mf.model
                .endo_index(v)
                .or_else(|| super::epilogue::index(mf, v).map(|k| n + k))
                .ok_or_else(|| anyhow!("{v} isn't an endogenous variable"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((idx, vars.to_vec()))
}

/// `M_.init2shocks.(name)`: (endogenous index, exogenous index) pairs.
pub fn init2shocks(mf: &ModFile, name: &str) -> Result<Vec<(usize, usize)>> {
    let (_, pairs) = mf
        .q4
        .init2shocks
        .iter()
        .find(|(n, _)| n == name)
        .ok_or_else(|| anyhow!("plot_shock_decomposition: no init2shocks block named {name}"))?;
    pairs
        .iter()
        .map(|(e, x)| {
            Ok((
                mf.model.endo_index(e).ok_or_else(|| anyhow!("init2shocks: {e}"))?,
                mf.model.exo_index(x).ok_or_else(|| anyhow!("init2shocks: {x}"))?,
            ))
        })
        .collect()
}

/// The `init2shocks` step of plot_shock_decomposition.m on one variable's (ncomp x T)
/// decomposition `z` (shocks first, then initial values, then the total) with that variable's
/// initial-condition decomposition `iv` ((n_endo+2) x T'): the contribution of each listed
/// variable's initial condition moves from "Initial values" to its shock. Returns the new
/// shock labels (`M_.exo_names_init`).
pub fn apply_init2shocks(
    z: &mut Mat,
    iv: &Mat,
    pairs: &[(usize, usize)],
    endo_names: &[String],
    exo_names: &[String],
) -> Vec<String> {
    let mut names = exo_names.to_vec();
    let (c, t) = (z.nrows(), z.ncols());
    for &(j, jj) in pairs {
        names[jj] = format!("{} + {}", names[jj], endo_names[j]);
        for s in 0..t.min(iv.ncols()) {
            z[(jj, s)] += iv[(j, s)];
            z[(c - 2, s)] -= iv[(j, s)];
        }
    }
    names
}

/// MATLAB colormaps by name (`colormap = VARIABLE_NAME` is a MATLAB variable holding a
/// colormap; the usual ones are the built-in functions): `n` RGB colors for the components,
/// as `fill(..., k)` maps component k of n onto the colormap (CDataMapping 'scaled').
pub fn colormap(name: &str, n: usize) -> Option<Vec<(f64, f64, f64)>> {
    let anchors: Vec<(f64, f64, f64)> = match name.to_ascii_lowercase().as_str() {
        "jet" => {
            vec![(0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
        }
        "parula" => vec![
            (0.2422, 0.1504, 0.6603),
            (0.2810, 0.3228, 0.9579),
            (0.1786, 0.5289, 0.9682),
            (0.0689, 0.6948, 0.8394),
            (0.2161, 0.7843, 0.5923),
            (0.6720, 0.7793, 0.2227),
            (0.9970, 0.7659, 0.2199),
            (0.9769, 0.9839, 0.0805),
        ],
        "turbo" => vec![
            (0.19, 0.07, 0.23),
            (0.27, 0.51, 0.98),
            (0.11, 0.90, 0.71),
            (0.64, 0.99, 0.24),
            (0.98, 0.73, 0.22),
            (0.85, 0.22, 0.03),
            (0.48, 0.02, 0.01),
        ],
        "hsv" => vec![
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        ],
        "hot" => vec![(0.0416, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)],
        "cool" => vec![(0.0, 1.0, 1.0), (1.0, 0.0, 1.0)],
        "spring" => vec![(1.0, 0.0, 1.0), (1.0, 1.0, 0.0)],
        "summer" => vec![(0.0, 0.5, 0.4), (1.0, 1.0, 0.4)],
        "autumn" => vec![(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
        "winter" => vec![(0.0, 0.0, 1.0), (0.0, 1.0, 0.5)],
        "gray" | "grey" => vec![(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
        "bone" => vec![(0.0, 0.0, 0.0), (0.32, 0.32, 0.44), (0.65, 0.78, 0.78), (1.0, 1.0, 1.0)],
        "copper" => vec![(0.0, 0.0, 0.0), (1.0, 0.7812, 0.4975)],
        "pink" => vec![(0.1178, 0.0, 0.0), (0.7, 0.53, 0.53), (0.85, 0.85, 0.62), (1.0, 1.0, 1.0)],
        "lines" => {
            let l = [
                (0.0, 0.447, 0.741),
                (0.85, 0.325, 0.098),
                (0.929, 0.694, 0.125),
                (0.494, 0.184, 0.556),
                (0.466, 0.674, 0.188),
                (0.301, 0.745, 0.933),
                (0.635, 0.078, 0.184),
            ];
            return Some((0..n).map(|k| l[k % l.len()]).collect());
        }
        _ => return None,
    };
    let m = anchors.len() - 1;
    Some(
        (0..n)
            .map(|k| {
                let u = if n <= 1 { 0.0 } else { k as f64 / (n - 1) as f64 } * m as f64;
                let i = (u.floor() as usize).min(m.saturating_sub(1));
                let f = u - i as f64;
                let (a, b) = (anchors[i], anchors[(i + 1).min(m)]);
                (a.0 + f * (b.0 - a.0), a.1 + f * (b.1 - a.1), a.2 + f * (b.2 - a.2))
            })
            .collect(),
    )
}

/// One sheet of `WriteShockDecomp2Excel.m`: header `Decomposition | labels.. | Smoot Var |
/// Steady State`, one row per period (`x` = the period as `double(dates)`), the steady state
/// in the first row of the last column, and the group legend when shock groups are used.
pub struct XlsSheet {
    pub name: String,
    pub rows: Vec<Vec<XlsCell>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum XlsCell {
    Empty,
    Num(f64),
    Text(String),
}

/// `double(d)` of a dates object (`year + (subperiod-1)/freq`; daily: the datenum).
pub fn date_double(d: crate::dates::Date) -> f64 {
    use crate::dates::Freq;
    match d.freq {
        Freq::Daily => d.time as f64,
        Freq::Annual => d.time as f64,
        f => {
            let n = f.number() as f64;
            d.year() as f64 + (d.subperiod().expect("non-daily") as f64 - 1.0) / n
        }
    }
}

/// Builds the sheet for one variable. `z1`: (comp_nbr+1) x T (components then total),
/// `labels`: comp_nbr component labels (with "Initial values"), `x`: the period values.
pub fn xls_sheet(
    name: &str,
    z1: &Mat,
    labels: &[String],
    x: &[f64],
    steady_state: f64,
    legend: Option<&[(String, Vec<String>)]>,
) -> XlsSheet {
    let comp = z1.nrows() - 1;
    let mut rows = Vec::new();
    let mut head = vec![XlsCell::Text("Decomposition".into())];
    head.extend(labels.iter().take(comp).map(|l| XlsCell::Text(l.clone())));
    head.push(XlsCell::Text("Smoot Var".into()));
    head.push(XlsCell::Text("Steady State".into()));
    rows.push(head);
    for (t, &xv) in x.iter().enumerate() {
        let mut r = vec![XlsCell::Num(xv)];
        r.extend((0..=comp).map(|k| XlsCell::Num(z1[(k, t)])));
        r.push(if t == 0 { XlsCell::Num(steady_state) } else { XlsCell::Empty });
        rows.push(r);
    }
    if let Some(groups) = legend {
        rows.push(Vec::new());
        rows.push(vec![XlsCell::Text("Legend.".into()), XlsCell::Text("Shocks include:".into())]);
        for (label, members) in groups {
            let mut r = vec![XlsCell::Text(label.clone())];
            r.extend(members.iter().map(|m| XlsCell::Text(m.clone())));
            rows.push(r);
        }
    }
    XlsSheet { name: name.to_string(), rows }
}

/// Writes the sheets to an .xlsx workbook.
pub fn write_xlsx(path: &Path, sheets: &[XlsSheet]) -> Result<()> {
    super::xlsx::write(path, sheets)
}

/// After `shock_decomposition`'s recursion (stream F / C): with `with_epilogue`, appends the
/// epilogue rows (and writes `<fname>_<what>_epilogue.csv`); records `oo_.dr.ys`. Returns the
/// rows to store.
#[allow(clippy::too_many_arguments)]
pub fn after_shock_decomposition(
    mf: &ModFile,
    st: Option<&EstimationState>,
    o: &Options,
    z: &[Mat],
    ys: &[f64],
    extra: &mut DecompExtra,
    out_dir: &Path,
    fname: &str,
    what: &str,
) -> Result<Vec<Mat>> {
    extra.ys = Some(ys.to_vec());
    if !with_epilogue(mf, o)? {
        return Ok(z.to_vec());
    }
    let params = current_params(mf, st);
    let all = append_epilogue(mf, &params, ys, z, extra)?;
    let n = z.len();
    let mut labels = mf.model.exo_names.clone();
    labels.push("Initial values".into());
    let rows: Vec<(String, Mat, Vec<String>)> =
        mf.q4.epilogue.iter().enumerate().map(|(k, e)| (e.name.clone(), all[n + k].clone(), labels.clone())).collect();
    let t = z.first().map_or(0, |m| m.ncols());
    let dates: Vec<String> = match st.and_then(|s| s.dates.clone()) {
        Some(d) if d.len() >= t => d[..t].to_vec(),
        Some(d) => crate::forecast::extend_dates(Some(&d), st.map_or(1, |s| s.first_obs) - 1 + d.len(), t - d.len())
            .into_iter()
            .fold(d.clone(), |mut a, x| {
                a.push(x);
                a
            }),
        None => (1..=t).map(|k| k.to_string()).collect(),
    };
    let path = out_dir.join(format!("{fname}_{what}_epilogue.csv"));
    crate::analysis::shock_decomposition::write_csv(&path, &rows, &dates)?;
    println!("with_epilogue: decomposition of {} epilogue variable(s) written to {}", rows.len(), path.display());
    Ok(all)
}

/// `initial_condition_decomposition` with `with_epilogue`.
pub fn initval_with_epilogue(
    mf: &ModFile,
    st: Option<&EstimationState>,
    o: &Options,
    z: crate::analysis::realtime_decomposition::Z3,
    ys: &[f64],
    extra: &mut DecompExtra,
) -> Result<crate::analysis::realtime_decomposition::Z3> {
    use crate::analysis::realtime_decomposition::Z3;
    extra.ys.get_or_insert_with(|| ys.to_vec());
    if !with_epilogue(mf, o)? {
        return Ok(z);
    }
    let mats: Vec<Mat> = (0..z.n).map(|v| z.row_mat(v)).collect();
    let all = append_epilogue(mf, &current_params(mf, st), ys, &mats, extra)?;
    Ok(Z3::from_row_mats(&all))
}

/// The parts of plot_shock_decomposition.m before its `switch realtime_` that /// adds: the steady state of the plotted rows (`oo_.dr.ys`, or `epilogue_steady_state` when a
/// plotted variable is an epilogue variable) and, for `init2shocks`, `oo_.initval_decomposition`
/// (computed here if absent, as Dynare does).
pub fn prepare_plot(
    mf: &ModFile,
    st: Option<&EstimationState>,
    o: &Options,
    spec: &mut crate::analysis::realtime_decomposition::PlotSpec,
    state: &mut crate::analysis::realtime_decomposition::DecompositionState,
    idx: &[usize],
    dir: &Path,
) -> Result<()> {
    let n = mf.model.n_endo();
    let epilogue_decomp = idx.iter().any(|&v| v >= n);
    spec.q4.steady_state = if epilogue_decomp {
        Some(state.q4.epilogue_steady_state.clone().ok_or_else(|| {
            anyhow!("plot_shock_decomposition: epilogue variables need shock_decomposition(with_epilogue) first")
        })?)
    } else {
        state.q4.ys.clone()
    };
    if spec.q4.init2shocks.is_some() && spec.realtime == 0 && state.initval.is_none() {
        println!("computing initval shock decomposition ...");
        let set = state.parameter_set.clone().map_or_else(
            || crate::forecast::commands::default_parameter_set(st, "shock_decomposition").map(str::to_string),
            Ok,
        )?;
        let mut own = None;
        let (dsge, data, _) = crate::forecast::commands::smoother_inputs(mf, st, o, dir, &mut own)?;
        let x = crate::analysis::realtime_decomposition::smoother_point(st, dsge, &set)?;
        let sm = crate::forecast::dsge_smoother(dsge, data, &x, false)?;
        let z =
            crate::analysis::realtime_decomposition::initial_condition_decomposition(sm.dr(), &sm.smoothed_deviation());
        let nn = z.n;
        // set_default_initial_condition_decomposition_options: with_epilogue as shock_decomp's
        let z = if state.q4.epilogue_steady_state.is_some() {
            let mats: Vec<Mat> = (0..z.n).map(|v| z.row_mat(v)).collect();
            crate::analysis::realtime_decomposition::Z3::from_row_mats(&append_epilogue(
                mf,
                &current_params(mf, st),
                sm.ys(),
                &mats,
                &mut state.q4,
            )?)
        } else {
            z
        };
        state.initval = Some(crate::analysis::realtime_decomposition::Stored {
            z,
            parameter_set: set,
            nshocks: nn,
            forecast_length: None,
        });
    }
    Ok(())
}

/// `WriteShockDecomp2Excel.m`: `Output/<fname>_shock_decomposition<_type><fig_name>.xlsx`, one
/// sheet per plotted variable (Dynare writes `.xls` through MATLAB's writetable/xlswrite).
pub fn write_xls(
    p: &crate::analysis::realtime_decomposition::Plotted,
    spec: &crate::analysis::realtime_decomposition::PlotSpec,
    mf: &ModFile,
    out_dir: &Path,
    fname: &str,
) -> Result<std::path::PathBuf> {
    let _ = mf;
    let (a, b, first) = p.window().unwrap_or((0, p.z.t, p.initial_date));
    let comp = p.z.c - 1;
    // WriteShockDecomp2Excel: screen_shocks only with more than 18 components and no groups
    let screen = (spec.q4.screen_shocks || spec.init_cond) && spec.groups.is_none() && comp > 18;
    let mut fig_name1 = p.fig_name_long.clone();
    if screen {
        fig_name1 += "_screen";
    }
    let fig_name1: String = fig_name1.replace(' ', "_").replace('.', "");
    // (options_.plot_shock_decomp.fig_mode is never set: the type is not in the file name)
    let path = out_dir.join(format!("{fname}_shock_decomposition{fig_name1}.xlsx"));
    let first_d = crate::dates::Date::parse(&first.label());
    let freq = first_d.map_or(1.0, |d| d.freq.number() as f64);
    let x0 = match first_d {
        Some(d) if d.freq != crate::dates::Freq::Daily && d.freq != crate::dates::Freq::Weekly => date_double(d),
        Some(_) => bail!("write_xls: This method is not implemented for daily frequency."),
        None => 1.0,
    };
    let t = b - a;
    let x: Vec<f64> = (0..t).map(|i| x0 + i as f64 / freq).collect();
    let legend: Option<Vec<(String, Vec<String>)>> = spec.groups.as_ref().map(|g| {
        let mut l: Vec<(String, Vec<String>)> = g
            .groups
            .iter()
            .map(|(label, idx)| (label.clone(), idx.iter().map(|&k| mf.model.exo_names[k].clone()).collect()))
            .collect();
        let used: Vec<usize> = g.groups.iter().flat_map(|(_, i)| i.iter().copied()).collect();
        let others: Vec<String> =
            (0..mf.model.n_exo()).filter(|k| !used.contains(k)).map(|k| mf.model.exo_names[k].clone()).collect();
        l.push(("Others".into(), others));
        l.truncate(comp);
        l
    });
    let mut sheets = Vec::new();
    for (v, var) in p.names.iter().enumerate() {
        let mut z1 = Mat::from_fn(p.z.c, t, |k, i| p.z.at(v, k, a + i));
        let mut labels = p.labels.clone();
        if screen {
            let m = comp - 1;
            let mut isort: Vec<usize> = (0..m).collect();
            let score = |k: usize| (0..t).map(|i| z1[(k, i)].abs()).sum::<f64>() / t as f64;
            isort.sort_by(|&p1, &q| score(q).total_cmp(&score(p1)));
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
            let mut l: Vec<String> = isort.iter().take(16).map(|&k| p.labels[k].clone()).collect();
            l.push("Others".into());
            l.push("Initial values".into());
            z1 = nz;
            labels = l;
        }
        let ss = p.steady_state.as_ref().and_then(|s| s.get(v).copied()).unwrap_or(0.0);
        sheets.push(xls_sheet(var, &z1, &labels, &x, ss, legend.as_deref()));
    }
    write_xlsx(&path, &sheets)?;
    Ok(path)
}

/// `colormap(options_.plot_shock_decomp.colormap)` for the figures saved by `f`.
pub fn with_colormap<R>(
    spec: &crate::analysis::realtime_decomposition::PlotSpec,
    ncomp: usize,
    f: impl FnOnce() -> R,
) -> R {
    let pal = spec.q4.colormap.as_deref().and_then(|name| {
        let p = colormap(name, ncomp.max(1));
        if p.is_none() {
            println!("WARNING: colormap = {name}: unknown MATLAB colormap (dynare-rust knows jet, parula, turbo, hsv, hot, cool, spring, summer, autumn, winter, gray, bone, copper, pink, lines); default colors used.");
        }
        p
    });
    crate::graphics::with_palette(pal, f)
}

/// The options of `plot_shock_decomposition` (`options_.plot_shock_decomp`), kept in
/// `PlotSpec::q4`.
#[derive(Clone, Debug, Default)]
pub struct PlotExtra {
    /// `init2shocks` / `init2shocks = NAME`: the block's (endogenous, exogenous) pairs.
    pub init2shocks: Option<Vec<(usize, usize)>>,
    pub screen_shocks: bool,
    pub steadystate: bool,
    pub write_xls: bool,
    pub colormap: Option<String>,
    pub interactive: bool,
    /// Steady state by stored row (`oo_.dr.ys`, or `epilogue_steady_state`), set by the caller.
    pub steady_state: Option<Vec<f64>>,
    pub endo_names: Vec<String>,
}

impl PlotExtra {
    pub fn from_options(mf: &ModFile, o: &Options) -> Result<PlotExtra> {
        let flag = |k: &str| match opt(o, k) {
            Some(OptValue::Flag) => true,
            Some(OptValue::Num(v)) => *v != 0.0,
            _ => false,
        };
        let init2shocks = match opt(o, "init2shocks") {
            None => None,
            Some(OptValue::Str(s)) => Some(init2shocks(mf, s)?),
            Some(_) => Some(init2shocks(mf, "default")?),
        };
        let colormap = match opt(o, "colormap") {
            Some(OptValue::Str(s)) => Some(s.clone()),
            _ => None,
        };
        if flag("interactive") {
            println!(
                "NOTE: plot_shock_decomposition(interactive): the MATLAB uimenus of the figures are not reproduced (static figures)."
            );
        }
        Ok(PlotExtra {
            init2shocks,
            screen_shocks: flag("screen_shocks"),
            steadystate: flag("steadystate"),
            write_xls: flag("write_xls"),
            colormap,
            interactive: flag("interactive"),
            steady_state: None,
            endo_names: mf.model.endo_names.clone(),
        })
    }
}
