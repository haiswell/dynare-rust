// SPDX-License-Identifier: GPL-3.0-or-later
//! [stream X] Dynare's command-line options (`dynare model.mod OPTIONS`, manual section
//! "Running Dynare"), passed to `dynare-rs` after the .mod file:
//!
//! - `savemacro[=FILE]`: writes the macro-expanded file (`<fname>_macroexp.mod`, comments
//!   removed); `linemacro` adds `@#line "file" n` markers; `onlymacro` stops after it;
//! - `-I<path>`: `@#include` search path; `-D` is handled by the CLI itself;
//! - `nograph`: as `options_.nograph = 1` before the first command; `onlymodel`: preprocess
//!   only; `nointeractive`, `console`, `nolog`, `notime`, `nowarn`, `noclearall`,
//!   `onlyclearglobals`, `debug`, `notmpterms`, `fast`, `minimal_workspace`, `use_dll`,
//!   `nostrict`, `nopathchange`, `nopreprocessoroutput`, `mexext=`, `matlabroot=`,
//!   `parallel...`, `conffile=`, `compute_xrefs`, `transform_unary_ops`, `nocommutativity`,
//!   `params_derivs_order=`, `output=`, `language=`, `warn_uninit`: accepted, no effect.
//! - `stochastic` (the model is treated as stochastic: exogenous leads become
//!   aux variables of the whole nonlinear term), `nostrict` (undeclared symbols of the model
//!   block become exogenous), `exclude_eqs=` / `include_eqs=` (equations removed by tag, their
//!   variable removed or turned exogenous), `json=parse|check|transform|compute`,
//!   `jsonstdout`, `onlyjson`, `jsonderivsimple` (`crate::q4::json`), `fast` (no effect: the
//!   preprocessor writes no files to skip), `minimal_workspace` (no MATLAB workspace),
//!   `params_derivs_order=0|1|2` (parameter derivatives are computed where needed);
//! - options on the first line of the .mod file (`// --+ options: json=compute, nostrict +--`)
//!   are read as Dynare does ([`CliOptions::merge_first_line`]).

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use crate::driver::RunOptions;

#[derive(Clone, Debug, Default)]
pub struct CliOptions {
    /// `savemacro` (`Some(None)`) or `savemacro=FILE`.
    pub savemacro: Option<Option<String>>,
    pub onlymacro: bool,
    pub linemacro: bool,
    pub nograph: bool,
    pub onlymodel: bool,
    /// `-I<path>`.
    pub include_paths: Vec<PathBuf>,
    pub stochastic: bool,
    pub nostrict: bool,
    pub exclude_eqs: Option<String>,
    pub include_eqs: Option<String>,
    /// `json=parse|check|transform|compute`.
    pub json: Option<String>,
    pub jsonstdout: bool,
    pub onlyjson: bool,
    pub jsonderivsimple: bool,
    pub params_derivs_order: Option<u8>,
}

const NO_EFFECT: &[&str] = &[
    "nointeractive",
    "console",
    "nolog",
    "notime",
    "nowarn",
    "noclearall",
    "onlyclearglobals",
    "debug",
    "notmpterms",
    "fast",
    "minimal_workspace",
    "use_dll",
    "nopathchange",
    "nopreprocessoroutput",
    "mexext",
    "matlabroot",
    "parallel",
    "parallel_follower_open_mode",
    "parallel_test",
    "parallel_use_psexec",
    "conffile",
    "compute_xrefs",
    "transform_unary_ops",
    "nocommutativity",
    "output",
    "language",
    "warn_uninit",
    "nolinemacro",
    "cygwin",
    "msvc",
    "mingw",
];

impl CliOptions {
    /// Reads Dynare's bare-word options; returns an error for an unknown or unsupported one.
    pub fn parse(words: &[String]) -> Result<CliOptions> {
        let mut o = CliOptions::default();
        for w in words {
            let (k, v) = match w.split_once('=') {
                Some((k, v)) => (k, Some(v.to_string())),
                None => (w.as_str(), None),
            };
            if let Some(p) = k.strip_prefix("-I") {
                o.include_paths.push(PathBuf::from(if p.is_empty() {
                    v.clone().unwrap_or_default()
                } else {
                    p.to_string()
                }));
                continue;
            }
            match k {
                "savemacro" => o.savemacro = Some(v),
                "onlymacro" => o.onlymacro = true,
                "linemacro" => o.linemacro = true,
                "nograph" => o.nograph = true,
                "onlymodel" => o.onlymodel = true,
                "stochastic" => o.stochastic = true,
                "nostrict" => o.nostrict = true,
                "exclude_eqs" | "include_eqs" => {
                    let v =
                        v.clone().filter(|s| !s.is_empty()).ok_or_else(|| anyhow::anyhow!("{k}: expected {k}=TAGS"))?;
                    if k == "exclude_eqs" { o.exclude_eqs = Some(v) } else { o.include_eqs = Some(v) }
                }
                "json" => match v.as_deref() {
                    Some(s @ ("parse" | "check" | "transform" | "compute")) => o.json = Some(s.to_string()),
                    _ => bail!("Incorrect syntax for json option: expected json=parse|check|transform|compute"),
                },
                "jsonstdout" => o.jsonstdout = true,
                "onlyjson" => o.onlyjson = true,
                "jsonderivsimple" => o.jsonderivsimple = true,
                "params_derivs_order" => match v.as_deref() {
                    Some(s @ ("0" | "1" | "2")) => o.params_derivs_order = Some(s.parse().unwrap()),
                    _ => bail!("Incorrect syntax for params_derivs_order option: expected 0, 1 or 2"),
                },
                _ if NO_EFFECT.contains(&k) => {}
                _ => bail!("unknown option `{w}` (Dynare command-line options go after the .mod file)"),
            }
        }
        Ok(o)
    }
}

impl CliOptions {
    /// Dynare reads options from the first line of the .mod file too:
    /// `// --+ options: OPT1 OPT2,OPT3 +--` (whitespace or comma separated; `-D` defines go to
    /// `defines`). The command line wins for flags given there.
    pub fn merge_first_line(&mut self, path: &Path, defines: &mut Vec<(String, String)>) -> Result<()> {
        let Ok(text) = std::fs::read_to_string(path) else { return Ok(()) };
        let Some(first) = text.lines().next() else { return Ok(()) };
        let Some(rest) = first.trim_start().strip_prefix("//").map(str::trim_start).and_then(|s| s.strip_prefix("--+"))
        else {
            return Ok(());
        };
        let Some(body) = rest.trim_start().strip_prefix("options:") else { return Ok(()) };
        let body = body.split("+--").next().unwrap_or("");
        let mut words = Vec::new();
        for w in body.split([' ', '\t', ',']).filter(|w| !w.is_empty()) {
            if let Some(d) = w.strip_prefix("-D") {
                match d.split_once('=') {
                    Some((a, b)) if !defines.iter().any(|(k, _)| k == a) => {
                        defines.push((a.to_string(), b.to_string()))
                    }
                    None if !defines.iter().any(|(k, _)| k == d) => defines.push((d.to_string(), String::new())),
                    _ => {}
                }
            } else {
                words.push(w.to_string());
            }
        }
        let f = CliOptions::parse(&words)?;
        self.savemacro = self.savemacro.take().or(f.savemacro);
        self.onlymacro |= f.onlymacro;
        self.linemacro |= f.linemacro;
        self.nograph |= f.nograph;
        self.onlymodel |= f.onlymodel;
        self.include_paths.extend(f.include_paths);
        self.stochastic |= f.stochastic;
        self.nostrict |= f.nostrict;
        self.exclude_eqs = self.exclude_eqs.take().or(f.exclude_eqs);
        self.include_eqs = self.include_eqs.take().or(f.include_eqs);
        self.json = self.json.take().or(f.json);
        self.jsonstdout |= f.jsonstdout;
        self.onlyjson |= f.onlyjson;
        self.jsonderivsimple |= f.jsonderivsimple;
        self.params_derivs_order = self.params_derivs_order.or(f.params_derivs_order);
        Ok(())
    }
}

/// The parser's settings for a run.
pub fn parse_options(ro: &RunOptions) -> crate::modfile::ParseOptions {
    crate::modfile::ParseOptions {
        defines: ro.defines.clone(),
        include_paths: ro.cli.include_paths.clone(),
        prelude: if ro.cli.nograph { vec!["options_.nograph = 1;".to_string()] } else { Vec::new() },
        stochastic: ro.cli.stochastic,
        nostrict: ro.cli.nostrict,
        exclude_eqs: ro.cli.exclude_eqs.clone(),
        include_eqs: ro.cli.include_eqs.clone(),
        cwd: std::env::current_dir().ok(),
    }
}

/// `savemacro` / `onlymacro`; returns true when the run stops here.
pub fn before_parse(path: &Path, ro: &RunOptions) -> Result<bool> {
    let c = &ro.cli;
    if c.savemacro.is_none() && !c.onlymacro {
        return Ok(false);
    }
    let lines = crate::modfile::macro_proc::preprocess_with(path, &ro.defines, &c.include_paths)?;
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let out = match &c.savemacro {
        Some(Some(f)) => dir.join(f),
        _ => dir.join(format!("{stem}_macroexp.mod")),
    };
    if c.savemacro.is_some() {
        let mut s = String::new();
        let mut last: Option<(&str, usize)> = None;
        // the preprocessor's output has no empty lines (whitespace-only ones stay)
        for l in lines.iter().filter(|l| !l.text.is_empty()) {
            if c.linemacro && last.is_none_or(|(f, n)| f != &*l.file || n + 1 != l.line) {
                s += &format!("@#line \"{}\" {}\n", l.file, l.line);
            }
            last = Some((&l.file, l.line));
            s += &l.text;
            s.push('\n');
        }
        std::fs::write(&out, s)?;
        println!("Macro-expanded file written to {}.", out.display());
    }
    Ok(c.onlymacro)
}
