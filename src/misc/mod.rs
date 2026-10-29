// SPDX-License-Identifier: GPL-3.0-or-later
//! [stream X] Small commands and MATLAB-level statements of the .mod file:
//!
//! - parameter assignments after a command (`M_.params` changes at that point of the driver),
//!   `set_param_value`, `options_.X = V` (the parser also applies them as defaults to later
//!   commands), other `M_.`/`oo_.` statements (warned about);
//! - `dynare_version`, `set_dynare_seed`, `dsample`, `model_info` (`model_info.m`, non-block
//!   part), `resid` (`display_static_residuals.m`), `save_params_and_steady_state`
//!   (`save_params_and_steady_state.m`), `prior` (`cli/prior.m`: `table`, `plot`, `simulate`,
//!   `moments`), `smoother2histval` (`smoother2histval.m`), `trace_plot`
//!   (`trace_plot.m`);
//! - commands that only make sense in MATLAB (bytecode printing, workspace transfers,
//!   compilation setup...): accepted with a note.

use std::path::Path;

use anyhow::{Result, anyhow, bail};

use crate::deterministic::fmt_g;
use crate::estimation::plots::McmcPlotContext;
use crate::forecast::commands::EstimationState;
use crate::modfile::{
    Command, MATLAB_STATEMENT, ModFile, OPTIONS_ASSIGNMENT, OptValue, Options, RAW_ARGS_KEY, SET_PARAM, opt,
};

pub mod cli;
pub mod homotopy;
pub mod model_info; // model_info: block decomposition display
pub mod prior_cmd;
pub mod prior_dist; // prior moments(distribution), irfs(distribution), optimize

/// Commands run by [`Session::run`].
pub fn handles(name: &str) -> bool {
    matches!(
        name,
        SET_PARAM
            | crate::modfile::SET_SHOCK_STDERR
            | OPTIONS_ASSIGNMENT
            | MATLAB_STATEMENT
            | "dynare_version"
            | "set_dynare_seed"
            | "dsample"
            | "model_info"
            | "resid"
            | "save_params_and_steady_state"
            | "prior"
            | "smoother2histval"
            | "trace_plot"
    ) || NOTE_ONLY.iter().any(|(n, _)| *n == name)
}

/// Commands accepted without effect, with the reason printed.
const NOTE_ONLY: &[(&str, &str)] = &[
    ("print_bytecode_dynamic_model", "needs the bytecode representation of the model, which dynare-rust does not use"),
    ("print_bytecode_static_model", "needs the bytecode representation of the model, which dynare-rust does not use"),
    ("compilation_setup", "dynare-rust does not compile the model with an external compiler"),
    ("model_options", "applied by the parser exactly as the options of the model block"),
    (
        "send_endogenous_variables_to_workspace",
        "there is no MATLAB workspace; simulated paths are in the Output/*.csv files",
    ),
    (
        "send_exogenous_variables_to_workspace",
        "there is no MATLAB workspace; simulated paths are in the Output/*.csv files",
    ),
    ("send_irfs_to_workspace", "there is no MATLAB workspace; IRFs are in Output/<fname>_IRF_<shock>.csv"),
    ("set_dynare_threads", "use the --threads command-line option"),
    ("clean_current_folder", "dynare-rust does not delete files"),
    ("internals", "MATLAB unit-test runner"),
    ("search", "MATLAB-only helper"),
    ("dplot", "MATLAB-only helper (dseries plots)"),
    ("dynatype", "MATLAB-only helper (writes oo_.endo_simul as text); see Output/<fname>_endo_simul.csv"),
    ("dynasave", "MATLAB-only helper (saves oo_.endo_simul); see Output/<fname>_endo_simul.csv"),
];

/// What the driver lends to [`Session::run`].
pub struct Ctx<'a> {
    pub dir: &'a Path,
    pub fname: &'a str,
    pub out_dir: &'a Path,
    /// Index of the command in `mf.commands`.
    pub ci: usize,
    pub est_state: Option<&'a EstimationState>,
    pub mcmc_plots: Option<&'a McmcPlotContext>,
    /// oo_.SmoothedVariables / oo_.SmoothedShocks of the last `calib_smoother` (when it ran
    /// after the last estimation): read by `smoother2histval`.
    pub calib_smoothed: Option<&'a SmoothedRecord>,
    /// `oo_.steady_state` and `oo_.exo_steady_state` of the driver's deterministic state: an
    /// `endval` block overwrites them (`resid`, `save_params_and_steady_state` read them).
    pub det_steady: Option<(&'a [f64], &'a [f64])>,
}

/// Smoothed values stored by `calib_smoother` (oo_.SmoothedVariables, oo_.SmoothedShocks).
#[derive(Debug, Clone)]
pub struct SmoothedRecord {
    /// The smoother output (levels, logs under loglinear) at the calibration.
    pub series: crate::driver::SmoothedSeries,
    /// Steady state of all endogenous variables (auxiliaries included).
    pub steady_state: Vec<f64>,
    pub loglinear: bool,
}

/// State kept across commands (`options_.smpl`, the seed of the last `set_dynare_seed`).
#[derive(Debug, Default)]
pub struct Session {
    /// `dsample`: `options_.smpl` (1-based first and last period).
    pub dsample: Option<(usize, usize)>,
    last_seed: Option<u64>,
}

fn raw_args(o: &Options) -> Vec<String> {
    match opt(o, RAW_ARGS_KEY) {
        Some(OptValue::List(l)) => l
            .iter()
            .filter_map(|v| match v {
                OptValue::Str(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Positional arguments of `name(a, 'b', c)`: the option keys the parser made of them, unquoted.
fn positional(o: &Options) -> Vec<String> {
    o.iter()
        .filter(|(k, _)| k != RAW_ARGS_KEY)
        .map(|(k, v)| match v {
            OptValue::Flag => k.trim_matches(['\'', '"']).to_string(),
            OptValue::Str(s) => format!("{k}={s}"),
            OptValue::Num(x) => format!("{k}={x}"),
            _ => k.clone(),
        })
        .collect()
}

impl Session {
    /// Runs one command; `seed` is the run's random seed (`set_dynare_seed` changes it).
    pub fn run(&mut self, mf: &mut ModFile, name: &str, o: &Options, seed: &mut u64, ctx: &Ctx) -> Result<()> {
        match name {
            SET_PARAM => {
                for (p, v) in o {
                    let OptValue::Num(v) = v else { continue };
                    let k = mf
                        .model
                        .param_names
                        .iter()
                        .position(|n| n == p)
                        .ok_or_else(|| anyhow!("unknown parameter {p}"))?;
                    mf.model.params[k] = *v;
                }
            }
            // ---- set_shock_stderr_value after a command (M_.Sigma_e(i,i) = v^2)
            crate::modfile::SET_SHOCK_STDERR => {
                for (e, v) in o {
                    let OptValue::Num(v) = v else { continue };
                    let i = mf.model.exo_index(e).ok_or_else(|| anyhow!("Shock name {e} doesn't exist"))?;
                    mf.sigma_e[(i, i)] = v * v;
                }
            }
            OPTIONS_ASSIGNMENT => {
                for (k, _) in o {
                    // options_.steady.maxit: GlobalOptions::write
                    if k.contains('.') && k != "steady.maxit" {
                        println!("NOTE: `options_.{k} = ...` is not used by dynare-rust.");
                    }
                }
            }
            MATLAB_STATEMENT => {
                let text = o.first().map(|(k, _)| k.as_str()).unwrap_or("");
                println!("WARNING: MATLAB statement `{text};` is not supported by dynare-rust; skipped.");
            }
            "dynare_version" => println!("{}", version_string()),
            "set_dynare_seed" => self.set_dynare_seed(o, seed)?,
            "dsample" => {
                let a: Vec<usize> = raw_args(o).iter().filter_map(|s| s.parse().ok()).collect();
                self.dsample = match a.as_slice() {
                    [] => None,
                    [n] => Some((1, *n)),
                    [a, b, ..] => Some((*a, *b)),
                };
                // options_.smpl is read by rplot (q4::rplot_dsample)
            }
            "model_info" => print!("{}", model_info_text(mf, o, ctx.fname)?),
            "resid" => print!("{}", resid_text_det(mf, o, ctx.ci, ctx.det_steady)?),
            "save_params_and_steady_state" => save_params_and_steady_state(mf, o, ctx)?,
            "prior" => prior_cmd::run(mf, &raw_args(o), ctx, *seed)?,
            "smoother2histval" => smoother2histval(mf, o, ctx)?,
            "trace_plot" => trace_plot(o, ctx)?,
            _ => {
                if let Some((_, why)) = NOTE_ONLY.iter().find(|(n, _)| *n == name) {
                    println!("NOTE: `{name}` is accepted but has no effect in dynare-rust ({why}).");
                }
            }
        }
        Ok(())
    }

    /// `set_dynare_seed(INTEGER)`, `('default')`, `('clock')`, `('reset')`, `('ALGORITHM', INTEGER)`.
    /// dynare-rust's streams are ChaCha8 whatever the algorithm; the integer seeds them.
    fn set_dynare_seed(&mut self, o: &Options, seed: &mut u64) -> Result<()> {
        let args = positional(o);
        let num = |s: &str| s.trim().parse::<f64>().ok().filter(|v| *v >= 0.0).map(|v| v as u64);
        match args.as_slice() {
            [a] if num(a).is_some() => *seed = num(a).unwrap(),
            [a] if a.eq_ignore_ascii_case("default") => *seed = 0,
            [a] if a.eq_ignore_ascii_case("clock") => {
                *seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0)
            }
            [a] if a.eq_ignore_ascii_case("reset") => *seed = self.last_seed.unwrap_or(0),
            [alg, n] if num(n).is_some() => {
                println!(
                    "NOTE: set_dynare_seed: dynare-rust's random streams are ChaCha8; the algorithm `{alg}` is ignored."
                );
                *seed = num(n).unwrap();
            }
            _ => bail!("set_dynare_seed: expected an integer, 'default', 'clock', 'reset' or ('ALGORITHM', INTEGER)"),
        }
        if !args.first().is_some_and(|a| a.eq_ignore_ascii_case("reset")) {
            self.last_seed = Some(*seed);
        }
        // MATLAB's global stream (mt19937ar), which csminwel draws from
        crate::gsa::rng::reset_global(*seed as u32);
        Ok(())
    }
}

/// `dynare_version`.
pub fn version_string() -> String {
    format!("7.2 (dynare-rust {})", env!("CARGO_PKG_VERSION"))
}

/// `subst_auxvar.m`: how an endogenous variable is printed with a lead/lag.
fn subst_auxvar(m: &crate::model::Model, v: usize, lead_lag: Option<i32>) -> String {
    let ll = |k: i32| format!("({k})");
    let opt_ll = |l: Option<i32>| l.map(ll).unwrap_or_else(|| "(".into());
    if v < m.orig_endo_nbr {
        return format!("{}{}", m.endo_names[v], opt_ll(lead_lag));
    }
    let Some(a) = m.aux_vars.iter().find(|a| a.endo_index == v) else { return m.endo_names[v].clone() };
    match a.ty {
        0 => format!("{}{}", m.endo_names[v], opt_ll(lead_lag)),
        1 => {
            let orig = a.orig_index.map(|i| m.endo_names[i].clone()).unwrap_or_default();
            format!("{orig}{}", opt_ll(lead_lag.map(|l| a.orig_lead_lag.unwrap_or(0) + l)))
        }
        2 => format!("{}(+1)", a.orig_expr),
        3 => {
            let orig = a.orig_index.map(|i| m.exo_names[i].clone()).unwrap_or_default();
            let base = a.orig_lead_lag.unwrap_or(0);
            let shift = if base == 0 { -1 } else { 0 };
            format!("{orig}{}", opt_ll(lead_lag.map(|l| base + l + shift)))
        }
        _ => a.orig_expr.clone(),
    }
}

/// `model_info.m` (non-block part): states, forward-looking and purely static variables.
pub fn model_info_text(mf: &ModFile, o: &Options, fname: &str) -> Result<String> {
    use std::fmt::Write;
    let m = &mf.model;
    let mut out = String::new();
    let (dynamic, stat, incidence) =
        (opt(o, "block_dynamic").is_some(), opt(o, "block_static").is_some(), opt(o, "incidence").is_some());
    if dynamic && stat {
        bail!("model_info: block_dynamic and block_static cannot be used simultaneously.");
    }
    if incidence && !(dynamic || stat) {
        bail!("model_info: the incidence option requires either block_dynamic or block_static to be specified.");
    }
    writeln!(out, "\nInformation about {fname} ({} model)", if stat { "static" } else { "dynamic" })?;
    // ---- model_info: the block decomposition (M_.block_structure / block_structure_stat)
    if dynamic || stat {
        let bs = if stat { model_info::static_block_structure(mf) } else { model_info::dynamic_block_structure(mf) };
        let Some(bs) = bs else {
            writeln!(out, "\nmodel_info: block information not present; skipping display.")?;
            return Ok(out);
        };
        let header_len =
            format!("\nInformation about {fname} ({} model)\n", if stat { "static" } else { "dynamic" }).len();
        out += &model_info::block_text(mf, &bs, stat, incidence, header_len);
        return Ok(out);
    }
    // ---- end model_info
    let line = |v: usize, ll: i32| -> String {
        if v < m.orig_endo_nbr {
            if ll == 0 { format!("{}\n", m.endo_names[v]) } else { format!("{}({ll})\n", m.endo_names[v]) }
        } else {
            let s = subst_auxvar(m, v, if ll == 0 { None } else { Some(ll) });
            let a = m.aux_vars.iter().find(|a| a.endo_index == v);
            match a.map(|a| a.orig_expr.as_str()).filter(|e| !e.is_empty()) {
                Some(e) => format!("{s} (original expression {e}) \n"),
                None => format!("{s}\n"),
            }
        }
    };
    let n = m.n_endo();
    let lag: Vec<usize> = (0..n).filter(|&i| m.lead_lag_incidence[i][0] != 0).collect();
    let lead: Vec<usize> = (0..n).filter(|&i| m.lead_lag_incidence[i][2] != 0).collect();
    if !lag.is_empty() {
        writeln!(out, "\nThe following variables appear with a lag and are therefore states:")?;
        lag.iter().for_each(|&v| out += &line(v, -1));
    }
    if !lead.is_empty() {
        writeln!(out, "\nThe following variables appear with a lead and are therefore forward-looking variables:")?;
        lead.iter().for_each(|&v| out += &line(v, 1));
    }
    let stat_vars: Vec<usize> = (0..n).filter(|i| !lag.contains(i) && !lead.contains(i)).collect();
    if !stat_vars.is_empty() {
        writeln!(
            out,
            "\nThe following variables do not appear with a lead or lag and are therefore purely static variables:"
        )?;
        stat_vars.iter().for_each(|&v| out += &line(v, 0));
    }
    writeln!(out)?;
    Ok(out)
}

/// `oo_.steady_state` / `oo_.exo_steady_state` as a command at position `ci` sees them: the
/// steady state if a `steady` command ran before, else the `initval` values (aux variables set).
pub fn current_steady_state(mf: &ModFile, ci: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    current_steady_state_det(mf, ci, None)
}

/// [`current_steady_state`] with the deterministic state's `oo_.steady_state` /
/// `oo_.exo_steady_state` (`det`), which an earlier `endval` block has overwritten.
pub fn current_steady_state_det(
    mf: &ModFile,
    ci: usize,
    det: Option<(&[f64], &[f64])>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let after_endval = mf.commands[..ci.min(mf.commands.len())].iter().any(|c| matches!(c, Command::Endval(_)));
    if let (Some((y, x)), true) = (det, after_endval) {
        return Ok((y.to_vec(), x.to_vec()));
    }
    let m = &mf.model;
    let mut x = vec![0.0; m.n_exo()];
    for &(i, v) in &mf.initval_exo {
        x[i] = v;
    }
    let mut y = vec![0.0; m.n_endo()];
    for &(i, v) in &mf.initval {
        y[i] = v;
    }
    m.set_auxiliary_variables(&mut y, &x, &m.params);
    let steady_ran = mf.commands[..ci.min(mf.commands.len())].iter().any(|c| matches!(c, Command::Steady(_)));
    if steady_ran {
        let ssm = crate::deterministic::SteadyStateModel::from_modfile(mf)?;
        let p = m.params.clone();
        y = crate::dsge::solve_steady(m, ssm.as_ref(), &p, &y, &crate::dsge::steady_options(mf))
            .map_err(|e| anyhow!("steady state failed (info {})", e.info))?;
    }
    Ok((y, x))
}

/// `resid` (`display_static_residuals.m`): static residuals at the `initval` values (with a
/// `steady_state_model`, at its values, as `options_.steadystate_flag` makes Dynare do).
pub fn resid_text(mf: &ModFile, o: &Options, ci: usize) -> Result<String> {
    resid_text_det(mf, o, ci, None)
}

/// [`resid_text`] with the deterministic state's steady state (`det`), see [`Ctx::det_steady`].
pub fn resid_text_det(mf: &ModFile, o: &Options, ci: usize, det: Option<(&[f64], &[f64])>) -> Result<String> {
    let m = &mf.model;
    let (mut y, x) = current_steady_state_det(mf, ci, det)?;
    let mut p = m.params.clone();
    if let Some(ssm) = crate::deterministic::SteadyStateModel::from_modfile(mf)? {
        ssm.eval(m, &mut y, &x, &mut p);
        m.set_auxiliary_variables(&mut y, &x, &p);
    }
    let mut z = vec![0.0; m.n_equations()];
    m.static_residuals(&y, &x, &p, &mut z);
    // options_.steady.non_zero persists once set
    let non_zero = mf.commands[..=ci.min(mf.commands.len() - 1)]
        .iter()
        .any(|c| matches!(c, Command::Other(n, o) if n == "resid" && opt(o, "non_zero").is_some()))
        || opt(o, "non_zero").is_some();
    let so = crate::dsge::steady_options(mf);
    let disp = mcp_residuals(m, &y, &p, so.solve_algo, &mut z);
    Ok(residuals_text_disp(m, &z, &equation_names(mf), so.tolf, non_zero, disp))
}

/// `display_static_residuals.m`'s complementarity conditions: with `solve_algo` 10 or 11 the
/// residuals of the equations whose variable sits at a bound are set to 0. Returns the
/// `disp_string` of the header.
pub fn mcp_residuals(m: &crate::model::Model, ys: &[f64], p: &[f64], solve_algo: u8, z: &mut [f64]) -> &'static str {
    let Ok(cc) = crate::deterministic::mcp::Complementarity::new(m, p) else {
        return "";
    };
    if matches!(solve_algo, 10 | 11) {
        for (j, (&lb, &ub)) in cc.lb.iter().zip(&cc.ub).enumerate() {
            if (lb.is_finite() || ub.is_finite()) && (ys[j] <= lb + f64::EPSILON || ys[j] >= ub - f64::EPSILON) {
                z[cc.eq_index[j]] = 0.0;
            }
        }
        " (accounting for complementarity conditions)"
    } else if cc.lb.iter().zip(&cc.ub).any(|(l, u)| l.is_finite() || u.is_finite()) {
        " (ignoring complementarity conditions)"
    } else {
        ""
    }
}

/// The body of `display_static_residuals.m` (with `nargout == 0`): the "Residuals of the static
/// equations" listing; `names` are the equation names (tags), `tolf` is `options_.solve_tolf`
/// and `non_zero` is `options_.steady.non_zero`. Also printed when `steady` fails.
pub fn residuals_text(
    m: &crate::model::Model,
    z: &[f64],
    names: &[Option<String>],
    tolf: f64,
    non_zero: bool,
) -> String {
    residuals_text_disp(m, z, names, tolf, non_zero, "")
}

/// [`residuals_text`] with the header's `disp_string` (see [`mcp_residuals`]).
pub fn residuals_text_disp(
    m: &crate::model::Model,
    z: &[f64],
    names: &[Option<String>],
    tolf: f64,
    non_zero: bool,
    disp: &str,
) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "\n\nResiduals of the static equations{disp}:");
    let last = m.orig_endo_nbr.min(z.len());
    // size(strvcat(tags(:,3)),2)+1: the longest tag value of the model
    let wtag = m
        .equation_tags
        .iter()
        .flatten()
        .map(|(_, v)| v.len())
        .chain(names.iter().flatten().map(String::len))
        .max()
        .unwrap_or(0)
        + 1;
    let wnum = m.n_equations().to_string().len();
    let mut any = false;
    for (i, &zi) in z.iter().enumerate().take(last) {
        let tmp = if zi.abs() < tolf / 100.0 { 0.0 } else { zi };
        any |= tmp != 0.0;
        if non_zero && tmp == 0.0 {
            continue;
        }
        match names.get(i).and_then(|n| n.as_ref()) {
            Some(tag) => {
                let _ = writeln!(s, "Equation number {:>wnum$}: {:<wtag$}: {:14.6}", i + 1, tag, tmp);
            }
            None => {
                let _ = writeln!(s, "Equation number {}: {}", i + 1, fmt_g(tmp));
            }
        }
    }
    if non_zero && !any {
        let _ = writeln!(s, "All residuals are zero");
    }
    let _ = writeln!(s, "\n");
    s
}

/// The `name` tag of each equation of the model block; as Dynare's preprocessor does, an
/// equation without one is named after its left-hand side when that is a single endogenous
/// variable.
pub fn equation_names(mf: &ModFile) -> Vec<Option<String>> {
    let m = &mf.model;
    // model_info: Ramsey models put the planner's FOCs before the original equations
    let ramsey = m.planner().is_some_and(|p| p.ramsey.is_some());
    let offset = if ramsey { m.orig_eq_nbr.saturating_sub(mf.original_model.equations.len()) } else { 0 };
    (0..m.n_equations())
        .map(|i| {
            if let Some((_, v)) = m.equation_tags.get(i).and_then(|t| t.iter().find(|(k, _)| k == "name")) {
                return Some(v.clone());
            }
            if i < offset || (ramsey && i >= m.orig_eq_nbr) {
                return None;
            }
            match mf.original_model.equations.get(i - offset) {
                Some((crate::modfile::Ast::Var { name, .. }, _, _))
                    if m.endo_names[..m.orig_endo_nbr].contains(name) =>
                {
                    Some(name.clone())
                }
                _ => None,
            }
        })
        .collect()
}

/// `save_params_and_steady_state(FILENAME)`: `name value` lines (`%.16g`) for the parameters,
/// the endogenous variables (steady state, or `initval` values if no `steady` ran) and the
/// exogenous variables.
fn save_params_and_steady_state(mf: &ModFile, o: &Options, ctx: &Ctx) -> Result<()> {
    let mut args = raw_args(o);
    if args.is_empty() {
        args = positional(o);
    }
    let file: String = args.concat().trim_matches(['(', ')']).trim_matches(['\'', '"']).to_string();
    if file.is_empty() {
        bail!("save_params_and_steady_state: a file name is required");
    }
    let path = if Path::new(&file).is_relative() { ctx.dir.join(&file) } else { file.clone().into() };
    let m = &mf.model;
    let (y, x) = current_steady_state_det(mf, ctx.ci, ctx.det_steady)?;
    let mut s = String::new();
    let np = m.n_user_params(); // the varexo_det values come last
    for (n, v) in m.param_names[..np].iter().zip(&m.params) {
        s += &format!("{n} {}\n", fmt_g16(*v));
    }
    for (n, v) in m.endo_names.iter().zip(&y) {
        s += &format!("{n} {}\n", fmt_g16(*v));
    }
    for (n, v) in m.exo_names.iter().zip(&x) {
        s += &format!("{n} {}\n", fmt_g16(*v));
    }
    for (n, v) in m.exo_det_names.iter().zip(m.exo_det_values(&m.params)) {
        s += &format!("{n} {}\n", fmt_g16(v));
    }
    std::fs::write(&path, s)
        .map_err(|e| anyhow!("SAVE_PARAMS_AND_STEADY_STATE: Can't open {}: {e}", path.display()))?;
    println!("save_params_and_steady_state: wrote {}.", path.display());
    Ok(())
}

/// C's `%.16g`.
pub fn fmt_g16(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Inf".into() } else { "-Inf".into() };
    }
    if v == 0.0 {
        return "0".into();
    }
    let s = format!("{:.15e}", v);
    let (mant, exp) = s.split_once('e').unwrap();
    let e: i32 = exp.parse().unwrap();
    let trim = |s: &str| -> String {
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s.to_string() }
    };
    if !(-4..16).contains(&e) {
        format!("{}e{}{:02}", trim(mant), if e < 0 { '-' } else { '+' }, e.abs())
    } else {
        trim(&format!("{:.*}", (15 - e).max(0) as usize, v))
    }
}

/// `smoother2histval(period=, invars=, outvars=, outfile=)`: initial conditions from the
/// smoothed variables of the last estimation (or `calib_smoother`) at its point. Without
/// `outfile` they become the `histval` of later commands; with `outfile` a `.mat` file in
/// dseries layout (`INIT__`, `NAMES__`, one column per variable) is written.
fn smoother2histval(mf: &mut ModFile, o: &Options, ctx: &Ctx) -> Result<()> {
    // oo_.SmoothedVariables of the last calib_smoother, else of the last estimation
    let (sm, ys, loglinear) = match (ctx.calib_smoothed, ctx.est_state) {
        (Some(r), _) => (r.series.clone(), r.steady_state.clone(), r.loglinear),
        (None, Some(st)) => {
            let x = st.current().to_vec();
            let mut sm = crate::driver::run_smoother(&st.dsge, &st.data, &x, false)?;
            // after an MCMC smoother2histval.m takes oo_.SmoothedVariables.Mean (post_metropolis),
            // the posterior mean of the smoothed draws, not the smoother at the posterior mean
            if let Some((v, s)) = &st.smoothed_mean {
                sm.smoothed = v.clone();
                sm.shocks = s.clone();
            }
            let ys = st
                .dsge
                .resolve(&x)
                .map(|r| r.steady_state)
                .map_err(|e| anyhow!("smoother2histval: no steady state (info {})", e.info))?;
            (sm, ys, st.dsge.loglinear)
        }
        (None, None) => bail!(
            "Could not find smoothed variables, did you compute them? (either by the \"smoother\" option of the \"estimation\" command, or the \"calib_smoother\" command"
        ),
    };
    let m = &mf.model;
    let n = sm.smoothed.nrows();
    let exo_ss = mf.exo_steady_state();
    let max_lag = m.orig_maximum_lag.max(1);
    if n < max_lag {
        bail!("Not enough observations to create initial conditions");
    }
    let endo_names: Vec<String> = m.endo_names[..m.orig_endo_nbr].to_vec();
    let list = |k: &str| -> Option<Vec<String>> {
        match opt(o, k) {
            Some(OptValue::List(l)) => {
                Some(l.iter().filter_map(|v| if let OptValue::Str(s) = v { Some(s.clone()) } else { None }).collect())
            }
            Some(OptValue::Str(s)) => Some(s.split_whitespace().map(str::to_string).collect()),
            _ => None,
        }
    };
    let period = match opt(o, "period") {
        Some(OptValue::Num(p)) => {
            let p = *p as usize;
            if p > n {
                bail!("The period that you indicated is beyond the data sample");
            }
            if p < max_lag {
                bail!("The period that you indicated is too small to construct initial conditions");
            }
            p
        }
        _ => n,
    };
    // lag auxiliary (AUX_ENDO_LAG: endogenous, AUX_EXO_LAG: exogenous): (endogenous?, orig index, lag)
    let lag_aux = |j: usize| -> Option<(bool, usize, i32)> {
        let a = m.aux_vars.iter().find(|a| a.endo_index == j)?;
        match (a.ty, a.orig_index, a.orig_lead_lag) {
            (1, Some(o), Some(l)) if l < 0 => Some((true, o, l)),
            (3, Some(o), Some(l)) if l <= 0 => Some((false, o, l)), // AUX_EXO_LAG_k_0 = u (l = 0)
            _ => None,
        }
    };
    // default: fieldnames(oo_.SmoothedVariables) (the lag auxiliaries included), then the shocks
    let invars = list("invars").unwrap_or_else(|| {
        (0..m.n_endo())
            .filter(|&j| j < m.orig_endo_nbr || lag_aux(j).is_some())
            .map(|j| m.endo_names[j].clone())
            .chain(m.exo_names.iter().cloned())
            .collect()
    });
    let outvars = list("outvars").unwrap_or_else(|| invars.clone());
    if invars.len() != outvars.len() {
        bail!("The number of input and output variables is not the same");
    }
    let series = |name: &str| -> Result<Vec<f64>> {
        if let Some(c) = endo_names.iter().position(|v| v == name) {
            return Ok(sm.smoothed.column(c).iter().map(|&v| if loglinear { v.exp() } else { v }).collect());
        }
        if let Some(c) = m.exo_names.iter().position(|v| v == name) {
            return Ok(sm.shocks.column(c).iter().copied().collect());
        }
        // a lag auxiliary: its smoothed value is the original variable's, shifted
        if let Some(j) = m.endo_names.iter().position(|v| v == name)
            && let Some((endo, o, l)) = lag_aux(j)
        {
            let src = if endo { sm.smoothed.column(o) } else { sm.shocks.column(o) };
            return Ok((0..n as i64)
                .map(|t| {
                    let s = t + l as i64;
                    if s < 0 {
                        ys[j] // the smoothed initial state is not kept: its steady state
                    } else if endo && loglinear {
                        src[s as usize].exp()
                    } else {
                        src[s as usize]
                    }
                })
                .collect());
        }
        bail!("smoother2histval: unknown input variable") // (sic: no name, as smoother2histval.m)
    };
    if let Some(OptValue::Str(file)) = opt(o, "outfile") {
        let mut cols = Vec::new();
        for v in &invars {
            let s = series(v)?;
            cols.push(s[period - max_lag..period].to_vec());
        }
        let path = std::path::PathBuf::from(file);
        let path = if path.extension().is_none() { path.with_extension("mat") } else { path };
        if path.extension().is_some_and(|e| e != "mat") {
            bail!("smoother2hisvtval: if outfile has an extension, it must be .mat");
        }
        let path = if path.is_relative() { ctx.dir.join(path) } else { path };
        let init = vec!["1Y".to_string()];
        let mut vars: Vec<(&str, crate::matfile::MatOut)> = vec![("INIT__", crate::matfile::MatOut::Strings(&init))];
        let names = invars.clone(); // o = dseries(data, '1Y', invars)
        vars.push(("NAMES__", crate::matfile::MatOut::Strings(&names)));
        for (k, v) in invars.iter().enumerate() {
            vars.push((v.as_str(), crate::matfile::MatOut::Vector(&cols[k])));
        }
        crate::matfile::write(&path, &vars)?;
        println!("smoother2histval: initial conditions written to {}.", path.display());
        return Ok(());
    }
    // M_.endo_histval = repmat(oo_.steady_state) and (Dynare 7.2) M_.exo_histval =
    // repmat(oo_.exo_steady_state) replace any earlier histval (the steady state over all
    // orig_maximum_lag lags makes the lag auxiliaries steady too), then the smoothed values
    // of the last M_.maximum_lag periods (1 once the lags are substituted)
    let ml = m.orig_maximum_lag.min(1);
    let mut hist = Vec::new();
    for l in 0..max_lag {
        for (j, &v) in ys[..m.orig_endo_nbr].iter().enumerate() {
            hist.push((true, j, -(l as i32), v));
        }
        for k in 0..m.n_exo() {
            hist.push((false, k, -(l as i32), exo_ss[k])); // repmat(oo_.exo_steady_state)
        }
    }
    for (vin, vout) in invars.iter().zip(&outvars) {
        let s = series(vin)?;
        // (endogenous?, index, lag of the histval entry of the date-0 value)
        let (is_endo, idx, shift) = if let Some(i) = m.endo_names.iter().position(|n| n == vout) {
            match lag_aux(i) {
                _ if i < m.orig_endo_nbr => (true, i, 0),
                Some((true, o, l)) => (true, o, l),
                // AUX_EXO_LAG: its own row of M_.endo_histval (kept apart from the exogenous histval)
                Some((false, _, _)) => (true, i, 0),
                None => continue, // other auxiliaries: no histval entry of their own
            }
        } else if let Some(i) = m.exo_names.iter().position(|n| n == vout) {
            (false, i, 0)
        } else {
            bail!("smoother2histval: output variable {vout} does not exist.");
        };
        for l in 0..ml {
            hist.push((is_endo, idx, shift - l as i32, s[period - 1 - l]));
        }
    }
    mf.histval = hist; // later entries win (the smoothed values over the steady state)
    println!("smoother2histval: histval set from the smoothed values at period {period} ({} variables).", invars.len());
    Ok(())
}

/// `trace_plot(options_, M_, estim_params_, TYPE, BLOCK, NAME1[, NAME2])` after an MCMC.
fn trace_plot(o: &Options, ctx: &Ctx) -> Result<()> {
    let Some(pc) = ctx.mcmc_plots else {
        println!("WARNING: trace_plot: no MCMC draws in memory (run an estimation with mh_replic > 0 first); skipped.");
        return Ok(());
    };
    let args: Vec<String> =
        positional(o).into_iter().filter(|a| !matches!(a.as_str(), "options_" | "m_" | "estim_params_")).collect();
    let [ty, blck, rest @ ..] = args.as_slice() else {
        bail!("trace_plot: expected (options_, M_, estim_params_, TYPE, BLOCK, NAME1[, NAME2])")
    };
    let b: usize = blck.parse().map_err(|_| anyhow!("trace_plot: bad block number {blck}"))?;
    let name = match (ty.to_ascii_lowercase().as_str(), rest) {
        ("posteriordensity", _) => None,
        ("deepparameter", [n, ..]) => Some(n.clone()),
        ("structuralshock" | "measurementerror", [n]) => Some(format!("stderr {n}")),
        ("structuralshock" | "measurementerror", [a, c, ..]) => Some(format!("corr {a}, {c}")),
        _ => bail!("trace_plot: unknown type {ty}"),
    };
    let j = match &name {
        None => None,
        Some(n) => Some(
            pc.names
                .iter()
                .position(|v| v == n)
                .ok_or_else(|| anyhow!("trace_plot: {n} is not an estimated parameter"))?,
        ),
    };
    let nblck = pc.chains.len();
    if b == 0 || b > nblck {
        bail!("trace_plot: block number {b} out of range (1..{nblck})");
    }
    let c = &pc.chains[b - 1];
    let draws: Vec<f64> =
        (c.first_stored..c.total_draws).map(|t| j.map_or(c.logpost_at(t), |j| c.draw(t)[j])).collect();
    let (what, plot_name) = match &name {
        None => ("the posterior density".to_string(), "Posterior".to_string()),
        Some(n) => (format!("parameter {n}"), crate::estimation::plots::dynare_param_name(n)),
    };
    let stem = format!("TracePlot_{plot_name}_blck_{b}");
    let fig = crate::estimation::plots::trace_plot_figure(&draws, &what, b, nblck, pc.trace_plot_ma, &stem);
    crate::graphics::save_as(&fig, &stem, &pc.gopts)?;
    Ok(())
}

/// Port of `datatomfile.m`: writes the simulated `oo_.endo_simul` of the variables `vars`
/// (default: the declared endogenous variables), renamed `names`, to the MATLAB script
/// `<file>.m` (`%15.8g`, one value per line), which an estimation can read as `datafile`.
pub fn datatomfile(
    mf: &crate::modfile::ModFile,
    endo_simul: &crate::linalg::Mat,
    o: &Options,
    dir: &std::path::Path,
    fname: &str,
) -> Result<()> {
    let strs = |k: &str| -> Option<Vec<String>> {
        match opt(o, k) {
            Some(OptValue::List(l)) => {
                Some(l.iter().filter_map(|v| if let OptValue::Str(s) = v { Some(s.clone()) } else { None }).collect())
            }
            _ => None,
        }
    };
    let Some(OptValue::Str(file)) = opt(o, "file") else { bail!("datatomfile: missing file name") };
    let sm = match std::path::Path::new(file).extension().and_then(|e| e.to_str()) {
        None => format!("{file}.m"),
        Some("m") => file.clone(),
        Some(_) => bail!("You did not provide a .m file as name of the datafile."),
    };
    let model = &mf.model;
    let vars =
        strs("vars").filter(|v| !v.is_empty()).unwrap_or_else(|| model.endo_names[..model.orig_endo_nbr].to_vec());
    let names = strs("names").unwrap_or_else(|| vars.clone());
    if names.len() != vars.len() {
        bail!("datatomfile:: Second and third arguments must have the same number of rows (variables)!");
    }
    let mut w = format!("% Dataset generated by {fname}.mod\n%\n");
    for (v, n) in vars.iter().zip(&names) {
        let i = model
            .endo_names
            .iter()
            .position(|e| e == v)
            .ok_or_else(|| anyhow!("One of the specified variables does not exist"))?;
        w += &format!("{n} = [\n\n");
        for t in 0..endo_simul.ncols() {
            w += &format!("{}\n", crate::analysis::table::fmt_g(endo_simul[(i, t)], 15, 8));
        }
        w += "];\n\n";
    }
    std::fs::write(dir.join(&sm), w)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g16() {
        assert_eq!(fmt_g16(0.33), "0.33");
        assert_eq!(fmt_g16(1.0 / 3.0), "0.3333333333333333");
        assert_eq!(fmt_g16(1e-20), "9.999999999999999e-21");
        assert_eq!(fmt_g16(0.1 + 0.2), "0.3");
        assert_eq!(fmt_g16(0.5e-4), "5e-05");
        assert_eq!(fmt_g16(123456.0), "123456");
        assert_eq!(fmt_g16(-2.5e17), "-2.5e+17");
    }
}
