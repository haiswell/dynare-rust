// SPDX-License-Identifier: GPL-3.0-or-later
//! LaTeX output, as Dynare writes it into `<fname>/latex/`:
//!
//! - `write_latex_original_model`, `write_latex_dynamic_model`, `write_latex_static_model`
//!   (the preprocessor's `writeLatexFile`): `original.tex` + `original_content.tex`, ...;
//! - `write_latex_definitions`, `write_latex_parameter_table`, `write_latex_prior_table`
//!   (`latex/*.m`);
//! - the `TeX` option's estimation tables (`display_estimation_results_table.m`,
//!   `GetPosteriorParametersStatistics.m`: [`estimation_tables`]) and `dyn_latex_table.m`
//!   (used by `model_comparison`);
//! - `collect_latex_files` (`<fname>_TeX_binder.tex` next to the .mod file).
//!
//! The equations are printed from dynare-rust's own expression trees, so the terms of a sum
//! or a product may come in another order than in Dynare's files (its DAG reorders
//! commutative operands); everything else (file names, layout, names, tables) is the same.

pub mod estimation_tables;
pub mod expr;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use crate::estimation::{Prior, PriorShape};
use crate::modfile::{Command, EstimatedKind, EstimatedParam, ModFile, opt};
use expr::{Ctx, Orderer, ast_latex_ordered, dag_latex};

/// The LaTeX commands handled here.
pub fn handles(name: &str) -> bool {
    matches!(
        name,
        "write_latex_original_model"
            | "write_latex_dynamic_model"
            | "write_latex_static_model"
            | "write_latex_steady_state_model"
            | "write_latex_definitions"
            | "write_latex_parameter_table"
            | "write_latex_prior_table"
            | "collect_latex_files"
    )
}

/// `<dir>/<fname>/latex`, created.
pub fn latex_dir(dir: &Path, fname: &str) -> Result<PathBuf> {
    let d = dir.join(fname).join("latex");
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

/// Driver entry point for the commands of [`handles`].
pub fn run_command(mf: &ModFile, c: &Command, dir: &Path, fname: &str) -> Result<()> {
    let Command::Other(name, o) = c else { bail!("latex: unexpected command") };
    let tags = opt(o, "write_equation_tags").is_some();
    let p = match name.as_str() {
        "write_latex_original_model" => write_model(mf, dir, fname, ModelKind::Original, tags)?,
        "write_latex_dynamic_model" => write_model(mf, dir, fname, ModelKind::Dynamic, tags)?,
        "write_latex_static_model" => write_model(mf, dir, fname, ModelKind::Static, tags)?,
        "write_latex_steady_state_model" => write_steady_state_model(mf, dir, fname)?,
        "write_latex_definitions" => write_latex_definitions(mf, dir, fname)?,
        "write_latex_parameter_table" => write_latex_parameter_table(mf, &mf.model.params, dir, fname)?,
        "write_latex_prior_table" => match write_latex_prior_table(
            mf,
            dir,
            fname,
            mf.options.prior_trunc.unwrap_or(0.0), // options_.prior_trunc as earlier commands left it
            mf.options.prior_interval.unwrap_or(0.9),
        )? {
            Some(p) => p,
            None => return Ok(()),
        },
        "collect_latex_files" => collect_latex_files(dir, fname)?,
        _ => bail!("latex: unknown command {name}"),
    };
    println!("LaTeX output written to {}", p.display());
    Ok(())
}

// ---- helpers ----------------------------------------------------------------------------

/// MATLAB's `datestr(now, 0)`: `21-Oct-2026 00:22:11` (UTC here).
pub fn datestr_now() -> String {
    let secs =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let days = secs.div_euclid(86400);
    let tod = secs.rem_euclid(86400);
    // civil_from_days (H. Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    const MON: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    format!("{:02}-{}-{} {:02}:{:02}:{:02}", d, MON[(m - 1) as usize], y, tod / 3600, tod % 3600 / 60, tod % 60)
}

/// MATLAB `sprintf('%W.Pf', v)` (Inf/NaN printed as `Inf`/`NaN`, right-aligned).
pub fn fmt_f(v: f64, width: usize, prec: usize) -> String {
    if v.is_finite() {
        format!("{v:>width$.prec$}")
    } else {
        let s = if v.is_nan() {
            "NaN"
        } else if v > 0.0 {
            "Inf"
        } else {
            "-Inf"
        };
        format!("{s:>width$}")
    }
}

/// `regexprep(s, '_', '\\_')`.
pub fn escape_underscore(s: &str) -> String {
    s.replace('_', "\\_")
}

fn get(v: &[String], i: usize, fallback: &str) -> String {
    v.get(i).cloned().unwrap_or_else(|| escape_underscore(fallback))
}

/// `M_.endo_names_tex` for all endogenous variables (aux variables: escaped name).
pub fn endo_tex(mf: &ModFile, j: usize) -> String {
    get(&mf.labels.endo_tex, j, &mf.model.endo_names[j])
}
pub fn exo_tex(mf: &ModFile, k: usize) -> String {
    get(&mf.labels.exo_tex, k, &mf.model.exo_names[k])
}
pub fn param_tex(mf: &ModFile, i: usize) -> String {
    get(&mf.labels.param_tex, i, &mf.model.param_names[i])
}
/// the `varexo_det` variable a parameter of the model stands for.
pub fn exo_det_of(mf: &ModFile, i: usize) -> Option<usize> {
    mf.model.exo_det_params.iter().position(|&p| p == i)
}
/// `M_.exo_det_names_tex`.
pub fn exo_det_tex(mf: &ModFile, k: usize) -> String {
    get(&mf.labels.exo_det_tex, k, &mf.model.exo_det_names[k])
}
fn long_or_name(v: &[String], i: usize, name: &str) -> String {
    v.get(i).cloned().unwrap_or_else(|| name.to_string())
}

// ---- model equations ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    Original,
    Dynamic,
    Static,
}

impl ModelKind {
    fn base(self) -> &'static str {
        match self {
            ModelKind::Original => "original",
            ModelKind::Dynamic => "dynamic",
            ModelKind::Static => "static",
        }
    }
}

fn tags_line(tags: &[(String, String)]) -> Option<String> {
    if tags.is_empty() {
        return None;
    }
    let mut t: Vec<&(String, String)> = tags.iter().collect();
    t.sort_by(|a, b| a.0.cmp(&b.0));
    let body: Vec<String> = t.iter().map(|(k, v)| format!("{k}= `{v}'")).collect();
    Some(format!("\\noindent[{}]\n", body.join(", ")))
}

/// TeX name of a model-local variable: its `model_local_variable` `$tex$`, else
/// the name with escaped underscores (as the preprocessor).
pub(crate) fn local_tex(mf: &ModFile, name: &str) -> String {
    mf.q4.local_labels.iter().find(|(n, _, _)| n == name).map_or_else(|| escape_underscore(name), |(_, t, _)| t.clone())
}

/// The `<kind>_content.tex` body: model-local variables (`dmath*`), then one `dmath` per
/// equation (with its tags if `write_equation_tags`).
pub fn model_content(mf: &ModFile, kind: ModelKind, write_equation_tags: bool) -> String {
    let ctx = Ctx::new(mf, kind == ModelKind::Static);
    let mut ord = Orderer::new();
    let mut s = String::new();
    for (name, ast) in &mf.original_model.locals {
        let _ = write!(
            s,
            "\\begin{{dmath*}}\n{} = {}\n\\end{{dmath*}}\n",
            local_tex(mf, name),
            ast_latex_ordered(ast, &ctx, &mut ord)
        );
    }
    match kind {
        ModelKind::Original => {
            for (k, (l, r, tags)) in mf.original_model.equations.iter().enumerate() {
                let _ = writeln!(s, "% Equation {}", k + 1);
                if write_equation_tags && let Some(t) = tags_line(tags) {
                    s.push_str(&t);
                }
                let l = ast_latex_ordered(l, &ctx, &mut ord);
                let r = ast_latex_ordered(r, &ctx, &mut ord);
                let _ = write!(s, "\\begin{{dmath}}\n{l}={r}\n\\end{{dmath}}\n");
            }
        }
        ModelKind::Dynamic | ModelKind::Static => {
            let m = &mf.model;
            let mut dag = m.dag().clone();
            let mut memo = std::collections::HashMap::new();
            for (k, &(l, r)) in m.equations().iter().enumerate() {
                let (l, r) = if kind == ModelKind::Static {
                    (dag.to_static(l, &mut memo), dag.to_static(r, &mut memo))
                } else {
                    (l, r)
                };
                let _ = writeln!(s, "% Equation {}", k + 1);
                if write_equation_tags && let Some(t) = m.equation_tags.get(k).and_then(|t| tags_line(t)) {
                    s.push_str(&t);
                }
                let _ = write!(
                    s,
                    "\\begin{{dmath}}\n{}={}\n\\end{{dmath}}\n",
                    dag_latex(&dag, l, &ctx),
                    dag_latex(&dag, r, &ctx)
                );
            }
        }
    }
    s
}

/// The preprocessor's `writeLatexFile`: `<kind>.tex` (document) and `<kind>_content.tex`.
pub fn write_model(
    mf: &ModFile,
    dir: &Path,
    fname: &str,
    kind: ModelKind,
    write_equation_tags: bool,
) -> Result<PathBuf> {
    let d = latex_dir(dir, fname)?;
    let base = kind.base();
    let doc = format!(
        "\\documentclass[10pt,a4paper]{{article}}\n\\usepackage[landscape]{{geometry}}\n\\usepackage{{fullpage}}\n\\usepackage{{amsfonts}}\n\\usepackage{{breqn}}\n\\begin{{document}}\n\\footnotesize\n\\include{{{base}_content}}\n\\end{{document}}\n"
    );
    std::fs::write(d.join(format!("{base}.tex")), doc)?;
    let p = d.join(format!("{base}_content.tex"));
    std::fs::write(&p, model_content(mf, kind, write_equation_tags))?;
    Ok(p)
}

/// The preprocessor's `SteadyStateModel::writeLatexSteadyStateFile`: `steady_state.tex`
/// (document) and `steady_state_content.tex`, one `dmath` per assignment, `lhs = rhs` with the
/// LHS's LaTeX name unbraced and the RHS as a static expression (one operand-order tree for
/// the whole block, as Dynare's `SteadyStateModel` DataTree).
pub fn write_steady_state_model(mf: &ModFile, dir: &Path, fname: &str) -> Result<PathBuf> {
    let m = &mf.model;
    let ctx = Ctx::new(mf, true);
    let mut ord = Orderer::new();
    let mut s = String::new();
    for (lhs, rhs) in &mf.steady_state_model {
        let l = match m.symbol(lhs) {
            Some(sym) if sym.kind == crate::model::SymbolKind::Endo => endo_tex(mf, sym.index),
            Some(sym) if sym.kind == crate::model::SymbolKind::Param => param_tex(mf, sym.index),
            _ => escape_underscore(lhs), // a temporary of the block
        };
        let ast = crate::modfile::parse_expression(rhs)?;
        let _ = write!(s, "\\begin{{dmath}}\n{l} = {}\n\\end{{dmath}}\n", ast_latex_ordered(&ast, &ctx, &mut ord));
    }
    let d = latex_dir(dir, fname)?;
    let doc = "\\documentclass[10pt,a4paper]{article}\n\\usepackage[landscape]{geometry}\n\\usepackage{fullpage}\n\\usepackage{amsfonts}\n\\usepackage{breqn}\n\\begin{document}\n\\footnotesize\n\\include{steady_state_content.tex}\n\\end{document}\n";
    std::fs::write(d.join("steady_state.tex"), doc)?;
    let p = d.join("steady_state_content.tex");
    std::fs::write(&p, s)?;
    Ok(p)
}

// ---- tables -------------------------------------------------------------------------------

const DEF_HEAD: &str = "\\hline%\n\\multicolumn{1}{c}{\\textbf{Variable}} &\n\\multicolumn{1}{c}{\\textbf{\\LaTeX}} &\n\\multicolumn{1}{c}{\\textbf{Description}}\\\\%\n\\hline\\hline%\n";

/// Port of `write_latex_definitions.m`: `<fname>_latex_definitions.tex`.
pub fn write_latex_definitions(mf: &ModFile, dir: &Path, fname: &str) -> Result<PathBuf> {
    let m = &mf.model;
    let endo: Vec<(String, String, String)> = (0..m.n_endo())
        .map(|j| (m.endo_names[j].clone(), endo_tex(mf, j), long_or_name(&mf.labels.endo_long, j, &m.endo_names[j])))
        .collect();
    let exo: Vec<_> = (0..m.n_exo())
        .map(|k| (m.exo_names[k].clone(), exo_tex(mf, k), long_or_name(&mf.labels.exo_long, k, &m.exo_names[k])))
        .collect();
    let par: Vec<_> = (0..m.n_user_params())
        .map(|i| {
            (m.param_names[i].clone(), param_tex(mf, i), long_or_name(&mf.labels.param_long, i, &m.param_names[i]))
        })
        .collect();
    // the `Exogenous Deterministic` table when there are `varexo_det` variables
    let det: Vec<_> = (0..m.exo_det_names.len())
        .map(|k| {
            (
                m.exo_det_names[k].clone(),
                exo_det_tex(mf, k),
                long_or_name(&mf.labels.exo_det_long, k, &m.exo_det_names[k]),
            )
        })
        .collect();
    let mut tables = vec![("Endogenous", &endo), ("Exogenous", &exo)];
    if !det.is_empty() {
        tables.push(("Exogenous Deterministic", &det));
    }
    tables.push(("Parameters", &par));
    let mut s = String::new();
    for (title, rows) in tables {
        s.push_str("\\begin{center}\n\\begin{longtable}{ccc}\n");
        let _ = writeln!(s, "\\caption{{{title}}}\\\\%");
        s.push_str(DEF_HEAD);
        s.push_str("\\endfirsthead\n\\multicolumn{3}{c}{{\\tablename} \\thetable{} -- Continued}\\\\%\n");
        s.push_str(DEF_HEAD);
        s.push_str("\\endhead\n");
        for (n, t, l) in rows.iter() {
            let _ = writeln!(s, "\\texttt{{{}}} & ${}$ & {}\\\\", escape_underscore(n), t, escape_underscore(l));
        }
        s.push_str("\\hline%\n\\end{longtable}\n\\end{center}\n");
    }
    let p = latex_dir(dir, fname)?.join(format!("{fname}_latex_definitions.tex"));
    std::fs::write(&p, s)?;
    Ok(p)
}

/// Port of `write_latex_parameter_table.m`: `<fname>_latex_parameters.tex` with the values
/// `params` (the calibration, or the values after an estimation).
pub fn write_latex_parameter_table(mf: &ModFile, params: &[f64], dir: &Path, fname: &str) -> Result<PathBuf> {
    let m = &mf.model;
    let np = m.n_user_params(); // not the varexo_det values
    let long: Vec<String> = (0..np).map(|i| long_or_name(&mf.labels.param_long, i, &m.param_names[i])).collect();
    let present = long != m.param_names[..np];
    let mut s = String::from("\\begin{center}\n");
    s.push_str(if present { "\\begin{longtable}{ccc}\n" } else { "\\begin{longtable}{cc}\n" });
    s.push_str("\\caption{Parameter Values}\\\\%\n\\toprule%\n\\multicolumn{1}{c}{\\textbf{Parameter}} &\n\\multicolumn{1}{c}{\\textbf{Value}} ");
    s.push_str(if present { "&\n \\multicolumn{1}{c}{\\textbf{Description}}\\\\%\n" } else { " \\\\%\n" });
    s.push_str("\\midrule%\n\\endfirsthead\n");
    s.push_str(if present {
        "\\multicolumn{3}{c}{{\\tablename} \\thetable{} -- Continued}\\\\%\n"
    } else {
        "\\multicolumn{2}{c}{{\\tablename} \\thetable{} -- Continued}\\\\%\n"
    });
    s.push_str("\\midrule%\n\\multicolumn{1}{c}{\\textbf{Parameter}} &\n\\multicolumn{1}{c}{\\textbf{Value}} ");
    s.push_str(if present { "&\n  \\multicolumn{1}{c}{\\textbf{Description}}\\\\%\n" } else { "\\\\%\n" });
    s.push_str("\\midrule%\n\\endhead\n");
    for i in 0..np {
        let v = fmt_f(params[i], 4, 3);
        if present {
            let l = escape_underscore(&long[i]).replace('%', "\\%");
            let _ = writeln!(s, "${}$ \t & \t {} \t & \t {}\\\\", param_tex(mf, i), v, l);
        } else {
            let _ = writeln!(s, "${}$ \t & \t {} \\\\", param_tex(mf, i), v);
        }
    }
    s.push_str("\\bottomrule%\n\\end{longtable}\n\\end{center}\n");
    let p = latex_dir(dir, fname)?.join(format!("{fname}_latex_parameters.tex"));
    std::fs::write(&p, s)?;
    Ok(p)
}

/// `get_the_name(k, TeX=1, ...)`'s TeX name: `$ \sigma_{e} $`, `$ \rho_{e1,e2} $`, `$ p $`.
pub fn get_the_tex_name(mf: &ModFile, e: &EstimatedParam) -> String {
    match e.kind {
        EstimatedKind::StdErr(k) => format!("$ \\sigma_{{{}}} $", exo_tex(mf, k)),
        EstimatedKind::Corr(a, b) => format!("$ \\rho_{{{},{}}} $", exo_tex(mf, a), exo_tex(mf, b)),
        EstimatedKind::Param(i) => format!("$ {} $", param_tex(mf, i)),
        // measurement errors (stream M): get_the_name.m's `SE_{y}` / `CC_{y1,y2}` (not checked against MATLAB)
        EstimatedKind::MeStdErr(k) => format!("$ SE_{{{}}} $", endo_tex(mf, k)),
        EstimatedKind::MeCorr(a, b) => format!("$ CC_{{{},{}}} $", endo_tex(mf, a), endo_tex(mf, b)),
        EstimatedKind::Skew(k) => format!("$ \\gamma_{{{}}} $", exo_tex(mf, k)),
        EstimatedKind::InitState(k) => format!("$ init {} $", endo_tex(mf, k)), // ---- estimation completion
    }
}

/// The estimated parameters' priors in `xparam1` order (`set_prior.m`); `None` for ML.
pub fn priors_of(mf: &ModFile) -> Result<Option<(Vec<EstimatedParam>, Vec<Prior>)>> {
    let est = mf.estimated_params_dynare_order();
    if est.is_empty() || est.iter().any(|e| e.prior.is_none()) {
        return Ok(None);
    }
    let priors = est
        .iter()
        .map(|e| {
            let shape = PriorShape::from_code(e.prior.unwrap().dynare_code()).unwrap();
            Prior::new(shape, e.mean, e.sd, e.p3, e.p4)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some((est, priors)))
}

/// Port of `write_latex_prior_table.m`: `<fname>_priors_table.tex`.
pub fn write_latex_prior_table(
    mf: &ModFile,
    dir: &Path,
    fname: &str,
    prior_trunc: f64,
    prior_interval: f64,
) -> Result<Option<PathBuf>> {
    let Some((est, priors)) = priors_of(mf)? else {
        println!("\nwrite_latex_prior_table:: No prior distributions detected. Skipping table creation.");
        return Ok(None);
    };
    const NAMES: [&str; 8] = ["Beta", "Gamma", "Gaussian", "Inv. Gamma", "Uniform", "Inv. Gamma -- 2", "", "Weibull"];
    let bounds_head = if prior_trunc == 0.0 { "Bounds" } else { "Bounds*" };
    let head = format!(
        "\\toprule%\n  &  &  &  &  & \\multicolumn{{2}}{{c}}{{{bounds_head}}} & \\multicolumn{{2}}{{c}}{{90\\% HPDI}} \\\\ \n  \\cmidrule(r{{.75em}}){{6-7}} \\cmidrule(r{{.75em}}){{8-9}}\n  & Distribution & Mean & Mode & Std.dev. & Lower & Upper & Lower & Upper  \\\\ \n"
    );
    let mut s = String::new();
    s.push_str("% TeX-table generated by Dynare write_latex_prior_table.m.\n% Prior Information\n");
    let _ = write!(s, "% {} \n \n", datestr_now());
    s.push_str("\\begin{center}\n\\begin{longtable}{lcccccccc} \n\\caption{Prior information (parameters)}\\\\\n \\label{Table:Prior}\\\\\n");
    s.push_str(&head);
    s.push_str("\\midrule\n\\endfirsthead\n\\caption{(continued)}\\\\\n ");
    s.push_str(&head);
    s.push_str("\\midrule\n\\endhead\n");
    if prior_trunc != 0.0 {
        let _ = writeln!(
            s,
            "\\midrule\n\\caption*{{*Displayed bounds are after applying a prior truncation of options_.trunc={}}}\\\\",
            fmt_f(prior_trunc, 4, 3)
        );
    }
    s.push_str("\\midrule\n\\multicolumn{9}{r}{(Continued on next page)} \\\\ \n\\bottomrule\n\\endfoot\n");
    if prior_trunc != 0.0 {
        let _ = writeln!(
            s,
            "\\midrule\n\\caption*{{\\emph{{Note:}} Displayed bounds are after applying a prior truncation of options\\_.prior\\_trunc={:.2e}}}\\\\",
            prior_trunc
        );
    }
    s.push_str("\\bottomrule\n\\endlastfoot\n");
    let interval = (1.0 - prior_interval) / 2.0; // options_.prior_interval
    let num = |v: f64| fmt_f(v, 6, 4);
    for (e, p) in est.iter().zip(&priors) {
        let (lb, ub) = p.bounds(prior_trunc);
        let (lo, hi): (String, String) = match p.shape {
            PriorShape::Beta | PriorShape::Uniform => {
                let lo = if lb.is_finite() { p.p3.max(lb) } else { p.p3 };
                let hi = if ub.is_finite() { p.p4.min(ub) } else { p.p4 };
                (num(lo), num(hi))
            }
            PriorShape::Gamma | PriorShape::InvGamma1 | PriorShape::InvGamma2 | PriorShape::Weibull => {
                let lo = if lb.is_finite() { p.p3.max(lb) } else { p.p3 };
                (num(lo), if ub.is_finite() { num(ub) } else { "$\\infty$".into() })
            }
            PriorShape::Normal => {
                let lo = if p.p3.is_infinite() && lb.is_infinite() {
                    "$-\\infty$".into()
                } else {
                    num(if lb.is_finite() { p.p3.max(lb) } else { p.p3 })
                };
                let hi = if p.p4.is_infinite() && ub.is_infinite() {
                    "$\\infty$".into()
                } else {
                    num(if ub.is_finite() { p.p4.min(ub) } else { p.p4 })
                };
                (lo, hi)
            }
        };
        let mode = if p.p5.is_nan() { "NaN".to_string() } else { num(p.p5) };
        let (ilo, ihi) = p.bounds(interval);
        let _ = writeln!(
            s,
            "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ ",
            get_the_tex_name(mf, e),
            NAMES[p.shape.code() as usize - 1],
            num(p.mean),
            mode,
            num(p.sd),
            lo,
            hi,
            num(ilo),
            num(ihi)
        );
    }
    s.push_str("\\end{longtable}\n \\end{center}\n% End of TeX file.\n");
    let path = latex_dir(dir, fname)?.join(format!("{fname}_priors_table.tex"));
    std::fs::write(&path, s)?;
    Ok(Some(path))
}

fn max_len<'a>(v: impl Iterator<Item = &'a String>) -> usize {
    v.map(|s| s.chars().count()).max().unwrap_or(0)
}

/// Port of `dyn_latex_table.m`: `<fname>/latex/<fname>_<latex_title>.tex`. `headers` and
/// `labels` are LaTeX strings (already escaped); `values` is labels x (headers - 1).
#[allow(clippy::too_many_arguments)]
pub fn dyn_latex_table(
    dir: &Path,
    fname: &str,
    title: &str,
    latex_title: &str,
    created_by: &str,
    headers: &[String],
    labels: &[String],
    values: &[Vec<f64>],
    label_width: usize,
    val_width: usize,
    val_precis: usize,
) -> Result<PathBuf> {
    if headers.len() < 2 {
        bail!("headers length must be >= 2");
    }
    let strip = |v: &[String]| -> Vec<String> {
        if v.iter().all(|x| x.starts_with('$') && x.ends_with('$') && x.len() >= 2) {
            v.iter().map(|x| x[1..x.len() - 1].to_string()).collect()
        } else {
            v.to_vec()
        }
    };
    let labels = strip(labels);
    let headers = strip(headers);
    let lw = (max_len(std::iter::once(&headers[0]).chain(labels.iter())) + 2).max(label_width);
    let finite: Vec<f64> = values.iter().flatten().copied().filter(|v| v.is_finite()).collect();
    let mut values_length = if finite.is_empty() {
        4
    } else {
        let m = finite.iter().map(|v| v.abs().log10()).fold(f64::NEG_INFINITY, f64::max);
        (m.ceil() as i64).max(1) as usize + val_precis + 1
    };
    if values.iter().flatten().any(|&v| v < 0.0) {
        values_length += 1;
    }
    let headers_length = max_len(headers[1..].iter());
    let vw = (headers_length.max(values_length) + 4).max(val_width);
    // MATLAB pads the headers with their backslashes doubled, then prints them as a format.
    let pad_left = |s: &str, w: usize| {
        let d = s.replace('\\', "\\\\");
        let n = d.chars().count();
        format!("{}{}", " ".repeat(w.saturating_sub(n)), s)
    };
    let pad_right = |s: &str, w: usize, double: bool| {
        let n = if double { s.replace('\\', "\\\\").chars().count() } else { s.chars().count() };
        format!("{}{}", s, " ".repeat(w.saturating_sub(n)))
    };
    let mut header = format!("${}$", pad_right(&headers[0], lw, true));
    for h in &headers[1..] {
        let _ = write!(header, "\t & \t ${}$", pad_left(h, vw));
    }
    header.push_str("\\\\\n");
    let mut s = String::new();
    let _ = write!(s, "% {}, created by {} \n \n", datestr_now(), created_by);
    s.push_str("\\begin{center}\n");
    let _ = writeln!(s, "\\begin{{longtable}}{{l{}}} ", "c".repeat(headers.len() - 1));
    let _ = write!(s, "\\caption{{{title}}}\\\\\n \\label{{Table:{latex_title}}}\\\\\n\\toprule \n");
    s.push_str(&header);
    s.push_str("\\midrule \\endfirsthead \n\\caption{(continued)}\\\\\n \\toprule \\\\ \n");
    s.push_str(&header);
    s.push_str("\\midrule \\endhead \n");
    let _ = writeln!(
        s,
        "\\midrule \\multicolumn{{{}}}{{r}}{{(Continued on next page)}} \\\\ \\bottomrule \\endfoot ",
        headers.len()
    );
    s.push_str("\\bottomrule \\endlastfoot \n");
    for (l, row) in labels.iter().zip(values) {
        let _ = write!(s, "${}$", pad_right(l, lw, false));
        for v in row {
            let _ = write!(s, "\t & \t{}", fmt_f(*v, vw, val_precis));
        }
        s.push_str(" \\\\ \n");
    }
    s.push_str("\\end{longtable}\n \\end{center}\n% End of TeX file.\n");
    let p = latex_dir(dir, fname)?.join(format!("{fname}_{latex_title}.tex"));
    std::fs::write(&p, s)?;
    Ok(p)
}

/// Port of `collect_latex_files.m`: `<dir>/<fname>_TeX_binder.tex` including every `.tex`
/// file of `<fname>/latex` (except the four document wrappers) and the `<fname>*.tex` files of
/// `<fname>/Output`, `<fname>/graphs`, `<fname>/identification[/Output]`, `<fname>/gsa[/Output]`.
pub fn collect_latex_files(dir: &Path, fname: &str) -> Result<PathBuf> {
    let binder = format!("{fname}_TeX_binder.tex");
    let mut s = String::new();
    for l in [
        "\\documentclass[12pt]{article}",
        "\\usepackage[margin=2cm]{geometry}",
        "\\usepackage{psfrag}",
        "\\usepackage{graphicx}",
        "\\usepackage{epstopdf}",
        "\\usepackage{longtable,booktabs}",
        "\\usepackage{amsmath,amsfonts,amssymb}",
        "\\usepackage{breqn}",
        "\\usepackage{float,morefloats,caption}",
        "\\begin{document}",
    ] {
        let _ = writeln!(s, "{l} ");
    }
    let stems = |sub: &str, prefix: Option<&str>| -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir.join(fname).join(sub))
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter_map(|e| e.file_name().into_string().ok())
                    .filter(|n| n.ends_with(".tex") && prefix.is_none_or(|p| n.starts_with(p)) && *n != binder)
                    .map(|n| n.trim_end_matches(".tex").to_string())
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    };
    for f in stems("latex", None) {
        if !matches!(f.as_str(), "dynamic" | "static" | "original" | "steady_state") {
            let _ = writeln!(s, "\\include{{{fname}/latex/{f}}} ");
        }
    }
    for sub in ["Output", "graphs", "identification", "identification/Output", "gsa", "gsa/Output"] {
        for f in stems(sub, Some(fname)) {
            let _ = writeln!(s, "\\include{{{fname}/{sub}/{f}}} ");
        }
    }
    s.push_str("\\end{document} \n");
    let p = dir.join(&binder);
    std::fs::write(&p, s)?;
    Ok(p)
}
