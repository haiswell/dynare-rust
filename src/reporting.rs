// SPDX-License-Identifier: GPL-3.0-or-later
//! Dynare's reporting package (`report()` objects of `matlab/reporting/`, dseries 7.2): the
//! statements `r = report(...)`, `r = r.addPage(...)`, `addSection`, `addGraph`, `addTable`,
//! `addSeries`, `addVspace`, `r.write()`, `r.compile()`, and the `dseries` built from the IRFs
//! that `send_irfs_to_workspace` exposes (`dseries(y_e, 2014q3, 'y')`, `[ds1 ds2]`, `ds.y`).
//!
//! The parser (`modfile/parser_report.rs`) turns each statement into a [`Stmt`] stored in a
//! [`REPORT_STATEMENT`] command; the driver runs them in order with a [`Session`]. Names that
//! are not report/dseries values resolve to the last `stoch_simul` IRFs (`<var>_<shock>` in
//! `<fname>_stoch_simul.json`). `write()` writes `report.tex` and `tmpRepDir/*.tex` (the layout
//! of Dynare's reporting classes: pgfplots graphs, booktabs tables); `compile()` runs pdflatex.
//! Not ported: the `report_param`/`report_data` table types, graph shading/legends options, and
//! most formatting options (accepted and ignored).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};

use crate::dates::Date;

/// The `Command::Other` name of a reporting statement (its option key is the JSON [`Stmt`]).
pub const REPORT_STATEMENT: &str = "__report_statement__";

/// A reporting / dseries expression.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Expr {
    Num(f64),
    Str(String),
    /// A date literal (`2014q3`), as its label.
    Date(String),
    /// `d1:d2`
    Range(Box<Expr>, Box<Expr>),
    /// A name (workspace value, IRF, `true`/`false`).
    Var(String),
    /// `x.name`
    Field(Box<Expr>, String),
    /// `{a, b}`
    Cell(Vec<Expr>),
    /// `[a b]`
    Concat(Vec<Expr>),
    /// `f(args)`
    Call(String, Vec<Expr>),
    /// `x.method(args)`
    Method(Box<Expr>, String, Vec<Expr>),
}

/// `lhs = rhs;` or `rhs;` (with the source text for messages).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stmt {
    pub lhs: Option<String>,
    pub rhs: Expr,
    pub text: String,
}

#[derive(Clone, Debug, Default)]
pub struct Dseries {
    pub init: Option<Date>,
    pub names: Vec<String>,
    /// One vector per variable, all of the same length.
    pub data: Vec<Vec<f64>>,
}

impl Dseries {
    fn nobs(&self) -> usize {
        self.data.first().map_or(0, Vec::len)
    }
    /// Value of variable `k` at date `d` (NaN outside the sample).
    fn at(&self, k: usize, d: Date) -> f64 {
        let Some(init) = self.init else { return f64::NAN };
        match init.diff(d) {
            Some(t) if t >= 0 && (t as usize) < self.nobs() => self.data[k][t as usize],
            _ => f64::NAN,
        }
    }
    /// `[a b]`: union of the variables over the union of the dates.
    fn merge(&self, o: &Dseries) -> Result<Dseries> {
        if self.names.is_empty() {
            return Ok(o.clone());
        }
        if o.names.is_empty() {
            return Ok(self.clone());
        }
        let (a, b) = (self.init.unwrap(), o.init.unwrap());
        if a.freq != b.freq {
            bail!("dseries: the frequencies of the dseries objects must be the same");
        }
        let first = if a.time <= b.time { a } else { b };
        let last = std::cmp::max(a.time + self.nobs() as i64, b.time + o.nobs() as i64);
        let n = (last - first.time) as usize;
        let mut out = Dseries { init: Some(first), ..Default::default() };
        for src in [self, o] {
            for (k, name) in src.names.iter().enumerate() {
                if out.names.contains(name) {
                    bail!("dseries: variable {name} appears twice in the concatenation");
                }
                out.names.push(name.clone());
                out.data.push((0..n).map(|t| src.at(k, first.add(t as i64))).collect());
            }
        }
        Ok(out)
    }
}

type Opts = Vec<(String, Val)>;

#[derive(Clone, Debug)]
enum Elem {
    Graph { opts: Opts, series: Vec<(Dseries, Opts)> },
    Table { opts: Opts, series: Vec<(Dseries, Opts)> },
    Vspace(usize),
}

#[derive(Clone, Debug, Default)]
struct Section {
    cols: usize,
    elems: Vec<Elem>,
}

#[derive(Clone, Debug, Default)]
struct Page {
    opts: Opts,
    sections: Vec<Section>,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    opts: Opts,
    pages: Vec<Page>,
}

#[derive(Clone, Debug)]
enum Val {
    Num(f64),
    Str(String),
    Bool(bool),
    Date(Date),
    Range(Vec<Date>),
    Cell(Vec<Val>),
    Ds(Dseries),
    Report(Box<Report>),
}

fn opt<'a>(o: &'a Opts, k: &str) -> Option<&'a Val> {
    o.iter().rev().find(|(n, _)| n.eq_ignore_ascii_case(k)).map(|(_, v)| v)
}

fn opt_str(o: &Opts, k: &str) -> Option<String> {
    match opt(o, k)? {
        Val::Str(s) => Some(s.clone()),
        _ => None,
    }
}

fn opt_num(o: &Opts, k: &str) -> Option<f64> {
    match opt(o, k)? {
        Val::Num(x) => Some(*x),
        _ => None,
    }
}

fn opt_bool(o: &Opts, k: &str) -> Option<bool> {
    match opt(o, k)? {
        Val::Bool(b) => Some(*b),
        Val::Num(x) => Some(*x != 0.0),
        _ => None,
    }
}

/// A string or a cell of strings.
fn opt_strs(o: &Opts, k: &str) -> Vec<String> {
    match opt(o, k) {
        Some(Val::Str(s)) => vec![s.clone()],
        Some(Val::Cell(v)) => {
            v.iter().filter_map(|x| if let Val::Str(s) = x { Some(s.clone()) } else { None }).collect()
        }
        _ => vec![],
    }
}

/// The reporting state of a run: workspace values and the IRFs of the last `stoch_simul`.
#[derive(Default)]
pub struct Session {
    vars: HashMap<String, Val>,
}

impl Session {
    /// Runs one [`REPORT_STATEMENT`] command.
    pub fn run(&mut self, o: &crate::modfile::Options, dir: &Path, fname: &str, out_dir: &Path) -> Result<()> {
        let Some((json, _)) = o.first() else { return Ok(()) };
        let st: Stmt = serde_json::from_str(json)?;
        let cx = Cx { dir, fname, out_dir };
        let v = self.eval(&st.rhs, &cx).map_err(|e| anyhow!("{}: {e}", st.text))?;
        if let (Some(n), Some(v)) = (&st.lhs, v) {
            self.vars.insert(n.clone(), v);
        }
        Ok(())
    }

    fn eval(&mut self, e: &Expr, cx: &Cx) -> Result<Option<Val>> {
        Ok(Some(match e {
            Expr::Num(x) => Val::Num(*x),
            Expr::Str(s) => Val::Str(s.clone()),
            Expr::Date(s) => Val::Date(Date::parse(s).ok_or_else(|| anyhow!("'{s}' is not a date"))?),
            Expr::Range(a, b) => match (self.value(a, cx)?, self.value(b, cx)?) {
                (Val::Date(a), Val::Date(b)) => Val::Range(Date::range(a, b)?),
                (Val::Num(a), Val::Num(b)) => Val::Cell((a as i64..=b as i64).map(|k| Val::Num(k as f64)).collect()),
                _ => bail!("':' expects two dates"),
            },
            Expr::Var(n) => match n.as_str() {
                "true" => Val::Bool(true),
                "false" => Val::Bool(false),
                _ => match self.vars.get(n) {
                    Some(v) => v.clone(),
                    None => Val::Cell(cx.irf(n)?.into_iter().map(Val::Num).collect()),
                },
            },
            Expr::Field(x, f) => match self.value(x, cx)? {
                Val::Ds(d) => {
                    let k =
                        d.names.iter().position(|n| n == f).ok_or_else(|| anyhow!("dseries: unknown variable {f}"))?;
                    Val::Ds(Dseries { init: d.init, names: vec![f.clone()], data: vec![d.data[k].clone()] })
                }
                _ => bail!("cannot access field {f}"),
            },
            Expr::Cell(v) => Val::Cell(v.iter().map(|x| self.value(x, cx)).collect::<Result<_>>()?),
            Expr::Concat(v) => {
                let mut acc: Option<Val> = None;
                for x in v {
                    let x = self.value(x, cx)?;
                    acc = Some(match (acc, x) {
                        (None, x) => x,
                        (Some(Val::Ds(a)), Val::Ds(b)) => Val::Ds(a.merge(&b)?),
                        (Some(Val::Cell(mut a)), Val::Cell(b)) => {
                            a.extend(b);
                            Val::Cell(a)
                        }
                        _ => bail!("unsupported concatenation"),
                    });
                }
                acc.unwrap_or(Val::Cell(vec![]))
            }
            Expr::Call(f, args) => {
                let args: Vec<Val> = args.iter().map(|x| self.value(x, cx)).collect::<Result<_>>()?;
                match f.as_str() {
                    "dseries" => match args.as_slice() {
                        // dseries(file): a csv/xls(x)/mat/m file
                        [Val::Str(file)] => {
                            let (init, names, v) = crate::deterministic::workspace::load_dseries_file(file, cx.dir)?;
                            let data = (0..v.ncols()).map(|j| v.column(j).iter().copied().collect()).collect();
                            Val::Ds(Dseries { init: Some(init), names, data })
                        }
                        _ => Val::Ds(dseries(&args)?),
                    },
                    "report" => Val::Report(Box::new(Report { opts: pairs(&args)?, pages: vec![] })),
                    "dates" => match args.as_slice() {
                        [Val::Str(s)] => Val::Date(Date::parse(s).ok_or_else(|| anyhow!("'{s}' is not a date"))?),
                        _ => bail!("dates: expected a string"),
                    },
                    _ => bail!("unknown function {f}"),
                }
            }
            Expr::Method(x, m, args) => {
                let args: Vec<Val> = args.iter().map(|x| self.value(x, cx)).collect::<Result<_>>()?;
                let Val::Report(mut r) = self.value(x, cx)? else { bail!("{m}: not a report object") };
                match m.as_str() {
                    "write" => {
                        r.write(cx.dir)?;
                        return Ok(None);
                    }
                    "compile" => {
                        r.compile(cx.dir)?;
                        return Ok(None);
                    }
                    _ => r.add(m, pairs(&args)?)?,
                }
                Val::Report(r)
            }
        }))
    }

    fn value(&mut self, e: &Expr, cx: &Cx) -> Result<Val> {
        self.eval(e, cx)?.ok_or_else(|| anyhow!("expression without a value"))
    }
}

struct Cx<'a> {
    dir: &'a Path,
    fname: &'a str,
    out_dir: &'a Path,
}

impl Cx<'_> {
    /// `<var>_<shock>` of the last stoch_simul (`send_irfs_to_workspace`).
    fn irf(&self, n: &str) -> Result<Vec<f64>> {
        let p = self.out_dir.join(format!("{}_stoch_simul.json", self.fname));
        let j: serde_json::Value = std::fs::read_to_string(&p)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .ok_or_else(|| anyhow!("unknown variable '{n}' (no stoch_simul IRFs in {})", p.display()))?;
        let v = j["irfs"][n].as_array().ok_or_else(|| anyhow!("unknown variable '{n}'"))?;
        let _ = self.dir;
        Ok(v.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect())
    }
}

/// `dseries()`, `dseries(data, init, names)`.
fn dseries(args: &[Val]) -> Result<Dseries> {
    let data: Vec<f64> = match args.first() {
        None => return Ok(Dseries::default()),
        Some(Val::Cell(v)) => v.iter().map(|x| if let Val::Num(x) = x { *x } else { f64::NAN }).collect(),
        Some(Val::Num(x)) => vec![*x],
        Some(_) => bail!("dseries: unsupported first argument"),
    };
    let init = match args.get(1) {
        Some(Val::Date(d)) => *d,
        Some(Val::Str(s)) => Date::parse(s).ok_or_else(|| anyhow!("dseries: '{s}' is not a date"))?,
        None => Date::parse("1Y").unwrap(),
        Some(_) => bail!("dseries: the second argument must be a date"),
    };
    let name = match args.get(2) {
        Some(Val::Str(s)) => s.clone(),
        Some(Val::Cell(v)) => match v.first() {
            Some(Val::Str(s)) => s.clone(),
            _ => "Variable_1".into(),
        },
        _ => "Variable_1".into(),
    };
    Ok(Dseries { init: Some(init), names: vec![name], data: vec![data] })
}

/// Name/value pairs.
fn pairs(args: &[Val]) -> Result<Opts> {
    if !args.len().is_multiple_of(2) {
        bail!("options must come in name/value pairs");
    }
    args.chunks(2)
        .map(|c| match &c[0] {
            Val::Str(k) => Ok((k.clone(), c[1].clone())),
            _ => bail!("option names must be strings"),
        })
        .collect()
}

fn page<'r>(r: &'r mut Report, m: &str) -> Result<&'r mut Page> {
    r.pages.last_mut().ok_or_else(|| anyhow!("{m}: add a page first (addPage)"))
}

impl Report {
    fn add(&mut self, m: &str, o: Opts) -> Result<()> {
        match m {
            "addPage" => {
                println!("Adding Page: {}", self.pages.len() + 1);
                self.pages.push(Page { opts: o, sections: vec![] });
            }
            "addSection" => {
                let cols = opt_num(&o, "cols").unwrap_or(1.0) as usize;
                page(self, m)?.sections.push(Section { cols: cols.max(1), elems: vec![] });
            }
            "addGraph" | "addTable" | "addVspace" => {
                let p = page(self, m)?;
                if p.sections.is_empty() {
                    p.sections.push(Section { cols: 1, elems: vec![] });
                }
                let s = p.sections.last_mut().unwrap();
                let e = match m {
                    "addVspace" => Elem::Vspace(opt_num(&o, "number").unwrap_or(1.0) as usize),
                    _ => {
                        let mut series = vec![];
                        if let Some(Val::Ds(d)) = opt(&o, "data") {
                            series.extend(split(d).into_iter().map(|d| (d, vec![])));
                        }
                        if m == "addGraph" { Elem::Graph { opts: o, series } } else { Elem::Table { opts: o, series } }
                    }
                };
                s.elems.push(e);
            }
            "addSeries" => {
                let Some(Val::Ds(d)) = opt(&o, "data") else { bail!("addSeries: the 'data' option must be a dseries") };
                let d = d.clone();
                let p = page(self, m)?;
                let el = p
                    .sections
                    .last_mut()
                    .and_then(|s| s.elems.iter_mut().rev().find(|e| !matches!(e, Elem::Vspace(_))));
                match el {
                    Some(Elem::Graph { series, .. } | Elem::Table { series, .. }) => {
                        series.extend(split(&d).into_iter().map(|d| (d, o.clone())))
                    }
                    _ => bail!("addSeries: add a graph or a table first"),
                }
            }
            _ => bail!("report: method {m} is not supported by dynare-rust"),
        }
        Ok(())
    }

    /// `report.write`: `report.tex` and `tmpRepDir/`.
    pub fn write(&self, dir: &Path) -> Result<()> {
        let fname = opt_str(&self.opts, "filename").unwrap_or_else(|| "report.tex".into());
        let rep = opt_str(&self.opts, "directory").unwrap_or_else(|| "tmpRepDir".into());
        let rdir = dir.join(&rep);
        std::fs::create_dir_all(&rdir)?;
        let orient_l = opt_str(&self.opts, "orientation").is_some_and(|s| s == "landscape");
        let paper = opt_str(&self.opts, "paper").unwrap_or_else(|| "a4".into());
        let margin = opt_num(&self.opts, "margin").unwrap_or(2.5);
        let unit = opt_str(&self.opts, "marginUnit").unwrap_or_else(|| "cm".into());
        let mut pre =
            String::from("% Report Object written by dynare-rust\n\\documentclass[11pt,notitlepage]{article}\n");
        let _ = writeln!(
            pre,
            "\\usepackage[{paper}paper,{}margin={margin:.6}{unit}]{{geometry}}",
            if orient_l { "landscape," } else { "" }
        );
        pre += "\\usepackage{pdflscape, booktabs, pgfplots, colortbl, adjustbox, multicol}\n\\pgfplotsset{compat=1.5.1}\\makeatletter\n\
\\def\\blfootnote{\\gdef\\@thefnmark{}\\@footnotetext}\n\\makeatother\n\\usepgfplotslibrary{fillbetween}\n\
\\definecolor{LightCyan}{rgb}{0.88,1,1}\n\\definecolor{Gray}{gray}{0.9}\n\\usepackage{fancyhdr, datetime}\n\
\\newdateformat{reportdate}{\\THEDAY\\ \\shortmonthname\\ \\THEYEAR}\n\\pagestyle{fancy}\n\\renewcommand{\\headrulewidth}{0pt}\n\
\\renewcommand{\\footrulewidth}{0.5pt}\n\\rfoot{\\scriptsize\\reportdate\\today\\ -- \\currenttime}\n\
\\renewcommand{\\textfraction}{0.05}\n\\renewcommand{\\topfraction}{0.8}\n\\renewcommand{\\bottomfraction}{0.8}\n\
\\setlength{\\parindent}{0in}\n\\setlength{\\tabcolsep}{1em}\n\\newlength\\sectionheight\n\\begin{document}\n";
        std::fs::write(rdir.join("report_preamble.tex"), pre)?;
        let mut doc = String::from("\\pgfplotsset{tick scale binop={\\times},\ntrim axis left}\n\\centering\n");
        for (pi, p) in self.pages.iter().enumerate() {
            let pg = pi + 1;
            println!("Writing Page: {pg}");
            let _ = writeln!(doc, "\\input{{{rep}/report_pg{pg}.tex}}");
            let mut s = format!("% Page Number {pg} written by dynare-rust\n\n");
            let titles = opt_strs(&p.opts, "title");
            let fmts = opt_strs(&p.opts, "titleFormat");
            if let Some(t) = titles.first() {
                let _ = writeln!(s, "\\addcontentsline{{toc}}{{subsection}}{{{t}}}");
            }
            s += "\\begin{tabular}[t]{c}\n";
            for (k, t) in titles.iter().enumerate() {
                let f = fmts.get(k).or(fmts.last()).cloned().unwrap_or_else(|| "\\large\\bfseries".into());
                let _ = writeln!(s, "\\multicolumn{{1}}{{c}}{{{f} {t}}}\\\\");
            }
            for (si, sec) in p.sections.iter().enumerate() {
                let sn = si + 1;
                s += "% Section Object written by dynare-rust\n\\maxsizebox{\\linewidth}{!}{%\n";
                let _ = writeln!(s, "\\begin{{tabular}}[t]{{{}}}", "l".repeat(sec.cols));
                let (mut row, mut col) = (1usize, 0usize);
                let mut line: Vec<String> = vec![];
                let mut vspace = String::new();
                for e in &sec.elems {
                    let (kind, body) = match e {
                        Elem::Vspace(n) => {
                            vspace += &" \\par \\medskip ".repeat(*n);
                            continue;
                        }
                        Elem::Graph { opts, series } => ("graph", graph_tex(opts, series)),
                        Elem::Table { opts, series } => ("table", table_tex(opts, series)?),
                    };
                    col += 1;
                    let f = format!("{kind}_pg{pg}_sec{sn}_row{row}_col{col}.tex");
                    std::fs::write(rdir.join(&f), body)?;
                    line.push(format!("\\input{{{rep}/{f}}}"));
                    if col == sec.cols {
                        let _ = writeln!(s, "{}\\\\", line.join(" & "));
                        line.clear();
                        col = 0;
                        row += 1;
                    }
                }
                if !line.is_empty() {
                    let _ = writeln!(s, "{}\\\\", line.join(" & "));
                }
                if !vspace.is_empty() {
                    let _ = writeln!(s, "{vspace}\\\\");
                }
                s += "\\end{tabular}}\\\\\n% End Section Object\n\n";
            }
            s += "\\end{tabular}\n\\clearpage\n% End Page Object\n";
            std::fs::write(rdir.join(format!("report_pg{pg}.tex")), s)?;
        }
        std::fs::write(rdir.join("report_document.tex"), doc)?;
        std::fs::write(
            dir.join(&fname),
            format!("\\input{{{rep}/report_preamble.tex}}\n\\input{{{rep}/report_document.tex}}\n\\end{{document}}\n"),
        )?;
        println!("Finished Writing Report!");
        Ok(())
    }

    /// `report.compile`: pdflatex on `report.tex` (twice, as Dynare for the contents). A failed
    /// compilation (no LaTeX, missing packages) is reported and the run goes on.
    pub fn compile(&self, dir: &Path) -> Result<()> {
        let fname = opt_str(&self.opts, "filename").unwrap_or_else(|| "report.tex".into());
        if !dir.join(&fname).is_file() {
            self.write(dir)?;
        }
        let Some(compiler) = opt_str(&self.opts, "compiler").map(PathBuf::from).or_else(find_pdflatex) else {
            println!(
                "WARNING: report.compile: no LaTeX compiler found (pdflatex); {fname} was written but not compiled."
            );
            return Ok(());
        };
        println!("Using compiler: {}", compiler.display());
        for _ in 0..2 {
            let out = std::process::Command::new(&compiler)
                .args(["-interaction=nonstopmode", "-halt-on-error", &fname])
                .current_dir(if dir.as_os_str().is_empty() { Path::new(".") } else { dir })
                .stdin(std::process::Stdio::null())
                .output();
            match out {
                Ok(o) if o.status.success() => {}
                Ok(o) => {
                    let log = String::from_utf8_lossy(&o.stdout);
                    let err = log.lines().find(|l| l.starts_with('!')).unwrap_or("").to_string();
                    println!(
                        "WARNING: report.compile: {} failed on {fname} (status {}) {err}; see the .log file.",
                        compiler.display(),
                        o.status.code().unwrap_or(-1)
                    );
                    return Ok(());
                }
                Err(e) => {
                    println!("WARNING: report.compile: cannot run {}: {e}", compiler.display());
                    return Ok(());
                }
            }
        }
        println!("Done compiling {}.", fname.replace(".tex", ".pdf"));
        Ok(())
    }
}

fn find_pdflatex() -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> =
        std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect()).unwrap_or_default();
    dirs.push("/Library/TeX/texbin".into());
    dirs.push("/usr/texbin".into());
    dirs.into_iter().map(|d| d.join("pdflatex")).find(|p| p.is_file())
}

/// One dseries per variable.
fn split(d: &Dseries) -> Vec<Dseries> {
    d.names
        .iter()
        .zip(&d.data)
        .map(|(n, x)| Dseries { init: d.init, names: vec![n.clone()], data: vec![x.clone()] })
        .collect()
}

/// The dates of a graph or table: the `xrange`/`range` option, else the union of the series.
fn span(opts: &Opts, key: &str, series: &[(Dseries, Opts)]) -> Vec<Date> {
    if let Some(Val::Range(r)) = opt(opts, key) {
        return r.clone();
    }
    let mut lo: Option<Date> = None;
    let mut hi: Option<Date> = None;
    for (d, _) in series {
        if let Some(i) = d.init {
            let e = i.add(d.nobs() as i64 - 1);
            lo = Some(lo.map_or(i, |l| if i.time < l.time { i } else { l }));
            hi = Some(hi.map_or(e, |h| if e.time > h.time { e } else { h }));
        }
    }
    match (lo, hi) {
        (Some(a), Some(b)) => Date::range(a, b).unwrap_or_default(),
        _ => vec![],
    }
}

fn graph_tex(opts: &Opts, series: &[(Dseries, Opts)]) -> String {
    let dates = span(opts, "xrange", series);
    let n = dates.len();
    let prec = opt_num(opts, "yTickLabelPrecision").unwrap_or(1.0) as usize;
    let mut s = String::from(
        "% Graph Object written by dynare-rust\n\\begin{tikzpicture}[baseline]\\begin{axis}[%\nset layers,\n",
    );
    if opt_bool(opts, "showGrid").unwrap_or(true) {
        s += "xmajorgrids=true,\nymajorgrids=true,\n";
    }
    s += "xminorticks=true,\nyminorticks=true,\n";
    if let (Some(a), Some(b)) = (dates.first(), dates.last()) {
        let _ = writeln!(
            s,
            "xticklabels={{{},{},}},\nxtick={{1,{n}}},",
            a.label().to_lowercase(),
            b.label().to_lowercase()
        );
    }
    let _ = writeln!(s, "y tick label style={{\n/pgf/number format/.cd,\nfixed,\nprecision={prec},\n/tikz/.cd\n}},");
    let w = opt_num(opts, "width").unwrap_or(6.0);
    let h = opt_num(opts, "height").unwrap_or(4.5);
    let _ = writeln!(
        s,
        "x tick label style={{rotate=0.000000}},\nwidth={w:.6}in,\nheight={h:.6}in,\nscale only axis,\nunbounded coords=jump,\naxis lines=box,\ntitle style={{align=center}},"
    );
    if let Some(t) = opt_strs(opts, "title").first() {
        let _ = writeln!(s, "title={t},");
    }
    let _ = writeln!(
        s,
        "enlarge x limits=false,\nenlarge y limits=true,\nxmin = 1,\nxmax = {n},\ntick label style={{font=\\normalsize}},\n]"
    );
    s += "\\pgfplotsset{every axis title/.append style={}}=[font=\\normalsize]\n";
    if opt_bool(opts, "showZeroLine").unwrap_or(false) {
        let c = opt_str(opts, "zeroLineColor").unwrap_or_else(|| "black".into());
        let _ = writeln!(s, "%zeroline\n\\addplot[{c},line width=.5,forget plot] coordinates {{(1,0)({n},0)}};");
    }
    for (k, (d, so)) in series.iter().enumerate() {
        let color = opt_str(so, "graphLineColor").unwrap_or_else(|| "black".into());
        let style = opt_str(so, "graphLineStyle").unwrap_or_else(|| "solid".into());
        let lw = opt_num(so, "graphLineWidth").unwrap_or(0.5);
        let _ = writeln!(
            s,
            "%series {}\n\\addplot[color={color},{style},line width={lw:.6}pt,line join=round,name path={}]\ntable[row sep=crcr]{{\nx y\\\\",
            d.names[0],
            k + 1
        );
        for (t, dt) in dates.iter().enumerate() {
            let v = d.at(0, *dt);
            if v.is_finite() {
                let _ = writeln!(s, "{} {v:.6}\\\\", t + 1);
            } else {
                let _ = writeln!(s, "{} nan\\\\", t + 1);
            }
        }
        s += "};\n";
    }
    s += "\\end{axis}\n\\end{tikzpicture}%";
    s
}

fn table_tex(opts: &Opts, series: &[(Dseries, Opts)]) -> Result<String> {
    let dates = span(opts, "range", series);
    let prec = opt_num(opts, "precision").unwrap_or(1.0) as usize;
    let mut s = String::from(
        "% Report_Table Object written by dynare-rust\n\\begin{tabular}[t]{l}\n\\setlength{\\parindent}{6pt}\n\\setlength{\\tabcolsep}{4pt}\n",
    );
    let _ = writeln!(s, "\\begin{{tabular}}{{@{{}}l{}@{{}}}}%\n\\toprule%", "r".repeat(dates.len()));
    if let Some(t) = opt_strs(opts, "title").first() {
        let _ = writeln!(s, "\\multicolumn{{{}}}{{c}}{{{t}}}\\\\%", dates.len() + 1);
    }
    let annual = dates.first().is_none_or(|d| d.freq == crate::dates::Freq::Annual);
    if annual {
        s += " & ";
        s += &dates.iter().map(|d| format!("\\multicolumn{{1}}{{c}}{{{}}}", d.label())).collect::<Vec<_>>().join(" & ");
        s += "\\\\[-2pt]%\n";
    } else {
        // years over sub-periods
        let mut groups: Vec<(i64, usize)> = vec![];
        for d in &dates {
            match groups.last_mut() {
                Some((y, c)) if *y == d.year() => *c += 1,
                _ => groups.push((d.year(), 1)),
            }
        }
        s += " & ";
        s += &groups.iter().map(|(y, c)| format!("\\multicolumn{{{c}}}{{c}}{{{y}}}")).collect::<Vec<_>>().join(" & ");
        s += "\\\\\n & ";
        let letter = dates[0].freq.letter();
        s += &dates
            .iter()
            .map(|d| Ok(format!("\\multicolumn{{1}}{{c}}{{{letter}{}}}", d.subperiod()?)))
            .collect::<Result<Vec<_>>>()?
            .join(" & ");
        s += "\\\\[-2pt]%\n";
    }
    s += "\\hline%\n%\n";
    for (d, so) in series {
        let name = opt_str(so, "tableRowTitle").unwrap_or_else(|| d.names[0].clone());
        let _ = write!(s, "% Table Row (report_series)\n\\rowcolor{{white}}\\noindent {name}");
        for dt in &dates {
            let v = d.at(0, *dt);
            if v.is_finite() {
                let _ = write!(s, "&{v:.prec$}");
            } else {
                s += "&";
            }
        }
        s += "\\\\%\n";
    }
    s += "\\bottomrule\n\\end{tabular}\n\\end{tabular}\n\\setlength{\\parindent}{0pt}\n \\par \\medskip\n\n% End Report_Table Object\n";
    Ok(s)
}
