// SPDX-License-Identifier: GPL-3.0-or-later
//! The few MATLAB idioms of Dynare's own test files that dynare-rust interprets at parse time
//! (anything else MATLAB stays a skipped statement with a warning):
//!
//! * `if isoctave ... elseif ... else ... end`: `isoctave` is false; a condition made of
//!   `isoctave`, `~`, `&&`, `||` and parentheses selects the branch whose statements are parsed,
//!   the others are skipped. A condition that also needs another MATLAB value
//!   (`~isoctave && user_has_matlab_license(...)`) keeps the old behaviour (the `if` line is
//!   skipped with a warning and every branch runs).
//! * `load 'X.mat'` / `load('X.mat')`: the MAT-file variables become workspace variables;
//! * `NAMES = fieldnames(S.a.b);`
//! * `for i=1:numel(NAMES) P = NAMES{i}; set_param_value(P, S.a.b.(P)); end` (the heterogeneity
//!   tests' calibration from a steady-state file).

use super::lexer::{Tok, Token};
use super::{BLOCK_KEYWORDS, Parser, raw_join, x::raw_join_spaced};
use crate::matfile::MatValue;
use crate::model::SymbolKind;
use crate::modfile::{Command, MATLAB_STATEMENT, OptValue};
use anyhow::{Result, bail};

/// MATLAB statements opening a block closed by `end`.
const MATLAB_OPENERS: &[&str] = &["if", "for", "parfor", "while", "switch", "try", "function"];

/// Dynare blocks closed by `end;` besides [`BLOCK_KEYWORDS`] (a skipped branch may hold one).
const MORE_DYNARE_BLOCKS: &[&str] = &[
    "estimated_params",
    "estimated_params_init",
    "estimated_params_bounds",
    "estimated_params_remove",
    "shock_groups",
    "moment_calibration",
    "irf_calibration",
    "homotopy_setup",
    "occbin_constraints",
    "svar_identification",
    "ramsey_constraints",
    "filter_initial_state",
    "generate_irfs",
    "epilogue",
    "model_replace",
    "pac_target_info",
    "perfect_foresight_controlled_paths",
    "shock_paths",
];

/// State of the MATLAB control flow met so far.
#[derive(Default)]
pub(super) struct MatlabState {
    /// Open MATLAB blocks: `true` for an `if` whose branch was selected here (its remaining
    /// branches are skipped), `false` for one kept as skipped statements (`end` warns).
    pub frames: Vec<bool>,
    /// Workspace variables (`load`, `fieldnames`).
    pub ws: std::collections::HashMap<String, MatValue>,
}

/// Three-valued MATLAB condition: `Some(b)` when `isoctave` decides it, `None` otherwise.
struct Cond<'t> {
    t: &'t [Token],
    k: usize,
}

impl Cond<'_> {
    fn at(&self, p: &str) -> bool {
        self.t.get(self.k).is_some_and(|t| t.is(p))
    }
    fn or(&mut self) -> Option<Option<bool>> {
        let mut a = self.and()?;
        while self.at("||") {
            self.k += 1;
            let b = self.and()?;
            a = match (a, b) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            };
        }
        Some(a)
    }
    fn and(&mut self) -> Option<Option<bool>> {
        let mut a = self.not()?;
        while self.at("&&") {
            self.k += 1;
            let b = self.not()?;
            a = match (a, b) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            };
        }
        Some(a)
    }
    fn not(&mut self) -> Option<Option<bool>> {
        if self.at("~") || self.at("!") {
            self.k += 1;
            return self.not().map(|v| v.map(|b| !b));
        }
        if self.at("(") {
            self.k += 1;
            let v = self.or()?;
            if !self.at(")") {
                return None;
            }
            self.k += 1;
            return Some(v);
        }
        if self.t.get(self.k).is_some_and(|t| t.is_ident("isoctave")) {
            self.k += 1;
            if self.at("(") && self.t.get(self.k + 1).is_some_and(|t| t.is(")")) {
                self.k += 2;
            }
            if self.at_operand_end() {
                return Some(Some(false));
            }
        }
        // any other MATLAB operand: unknown, up to the next `&&` / `||` / unmatched `)`
        let st = self.k;
        let mut depth = 0i32;
        while let Some(t) = self.t.get(self.k) {
            if depth == 0 && (t.is("&&") || t.is("||") || t.is(")")) {
                break;
            }
            if t.is("(") || t.is("[") || t.is("{") {
                depth += 1;
            } else if t.is(")") || t.is("]") || t.is("}") {
                depth -= 1;
            }
            self.k += 1;
        }
        (self.k > st).then_some(None)
    }
    fn at_operand_end(&self) -> bool {
        self.t.get(self.k).is_none_or(|t| t.is("&&") || t.is("||") || t.is(")"))
    }
}

/// The value of an `if`/`elseif` condition when `isoctave` (false) decides it.
fn octave_condition(t: &[Token]) -> Option<bool> {
    if !t.iter().any(|t| t.is_ident("isoctave")) {
        return None;
    }
    let mut c = Cond { t, k: 0 };
    let v = c.or()?;
    if c.k == t.len() { v } else { None }
}

impl<'a> Parser<'a> {
    /// Handles the statement starting with `kw` when it is one of the idioms above; `false`
    /// leaves it to the usual parsing.
    pub(super) fn ml_statement(&mut self, kw: &str) -> Result<bool> {
        match kw {
            "if" => self.ml_if(),
            "elseif" | "else" if self.ml.frames.last() == Some(&true) => {
                // the branch that ran ends here: skip the others
                self.ml.frames.pop();
                let end = self.ml_skip_to(false)?;
                self.c.i = end;
                self.ml_eat_keyword_line();
                Ok(true)
            }
            "end" => {
                let decided = self.ml.frames.pop() == Some(true);
                if decided {
                    self.ml_eat_keyword_line();
                }
                Ok(decided)
            }
            "for" => {
                if self.ml_for()? {
                    return Ok(true);
                }
                self.ml.frames.push(false);
                Ok(false)
            }
            "parfor" | "while" | "switch" | "try" => {
                self.ml.frames.push(false);
                Ok(false)
            }
            "load" => self.ml_load(),
            _ if self.c.peek_at(1).is_some_and(|t| t.is("="))
                && self.c.peek_at(2).is_some_and(|t| t.is_ident("fieldnames")) =>
            {
                self.ml_fieldnames()
            }
            _ => Ok(false),
        }
    }

    /// Consumes `end` / `else` and a following `;` or `,`.
    fn ml_eat_keyword_line(&mut self) {
        self.c.i += 1;
        if self.c.at(";") || self.c.at(",") {
            self.c.i += 1;
        }
    }

    /// The tokens of the MATLAB line starting at the cursor (ended by `;` or the end of line),
    /// consumed.
    fn ml_line(&mut self) -> Result<&'a [Token]> {
        self.c.native_end = self.c.native_line_end();
        self.c.statement()
    }

    /// `if COND` (the cursor is on `if`).
    fn ml_if(&mut self) -> Result<bool> {
        let st = self.c.i;
        let toks = self.ml_line()?;
        let Some(mut take) = octave_condition(&toks[1..]) else {
            // not decided here: the `if` line is skipped with a warning, the branches all run
            self.c.i = st;
            self.ml.frames.push(false);
            return Ok(false);
        };
        loop {
            if take {
                self.ml.frames.push(true);
                return Ok(true);
            }
            self.c.i = self.ml_skip_to(true)?;
            let t = &self.c.t[self.c.i];
            if t.is_ident("end") {
                self.ml_eat_keyword_line();
                return Ok(true);
            }
            if t.is_ident("else") {
                self.ml_eat_keyword_line();
                take = true;
                continue;
            }
            // elseif COND
            let toks = self.ml_line()?;
            match octave_condition(&toks[1..]) {
                Some(v) => take = v,
                None => {
                    // as an undecided `if`: warn, the remaining branches run
                    self.commands
                        .push(Command::Other(MATLAB_STATEMENT.into(), vec![(raw_join_spaced(toks), OptValue::Flag)]));
                    self.ml.frames.push(false);
                    return Ok(true);
                }
            }
        }
    }

    /// Index of the `end` closing the MATLAB block whose body starts at the cursor (with
    /// `branch`, of the first `else` / `elseif` / `end` of the current `if` level).
    fn ml_skip_to(&self, branch: bool) -> Result<usize> {
        let t = self.c.t;
        let (mut depth, mut brackets) = (0usize, 0i32);
        let mut k = self.c.i;
        while k < t.len() {
            let tok = &t[k];
            let start =
                brackets <= 0 && (k == self.c.i || t[k - 1].is(";") || t[k - 1].is(",") || t[k - 1].loc != tok.loc);
            if start && let Some(id) = tok.ident() {
                let dynare_block = (BLOCK_KEYWORDS.contains(&id) && id != "verbatim"
                    || MORE_DYNARE_BLOCKS.contains(&id))
                    && t.get(k + 1).is_some_and(|n| n.is(";") || n.is("("));
                if MATLAB_OPENERS.contains(&id) || dynare_block {
                    depth += 1;
                } else if id == "end" {
                    if depth == 0 {
                        return Ok(k);
                    }
                    depth -= 1;
                } else if branch && depth == 0 && (id == "else" || id == "elseif") {
                    return Ok(k);
                }
            }
            if tok.is("(") || tok.is("[") || tok.is("{") {
                brackets += 1;
            } else if tok.is(")") || tok.is("]") || tok.is("}") {
                brackets -= 1;
            }
            k += 1;
        }
        let at = t.get(self.c.i).map(|t| t.loc.to_string()).unwrap_or_default();
        bail!("{at}: MATLAB block not closed by 'end'")
    }

    /// `load 'X.mat'`, `load('X.mat')`, `load X.mat`: the file's variables join the workspace.
    /// Anything else, or a file that is not there yet (written by the run), stays a skipped
    /// statement.
    fn ml_load(&mut self) -> Result<bool> {
        let st = self.c.i;
        let toks = self.ml_line()?;
        let args = &toks[1..];
        let args = match args {
            [o, inner @ .., c] if o.is("(") && c.is(")") => inner,
            a => a,
        };
        let file = match args {
            [t] if matches!(t.tok, Tok::Str(_)) => match &t.tok {
                Tok::Str(s) => s.clone(),
                _ => unreachable!(),
            },
            a if !a.is_empty() && a.iter().all(|t| t.ident().is_some() || t.is(".")) => raw_join(a),
            _ => {
                self.c.i = st;
                return Ok(false);
            }
        };
        let mut path = std::path::PathBuf::from(&file);
        if path.extension().is_none() {
            path.set_extension("mat");
        }
        if path.is_relative()
            && let Some(dir) = std::path::Path::new(&*toks[0].loc.file).parent()
            && dir.join(&path).exists()
        {
            path = dir.join(path);
        }
        match crate::matfile::read(&path) {
            Ok(vars) => {
                self.ml.ws.extend(vars);
                Ok(true)
            }
            Err(_) => {
                self.c.i = st;
                Ok(false)
            }
        }
    }

    /// The workspace value at `S.a.b` (`toks` = `S . a . b`).
    fn ml_path(&self, toks: &[Token]) -> Option<&MatValue> {
        let mut v = self.ml.ws.get(toks.first()?.ident()?)?;
        let mut k = 1;
        while k < toks.len() {
            if !toks[k].is(".") {
                return None;
            }
            v = v.field(toks.get(k + 1)?.ident()?)?;
            k += 2;
        }
        Some(v)
    }

    /// `NAMES = fieldnames(S.a.b);` of a workspace struct.
    fn ml_fieldnames(&mut self) -> Result<bool> {
        let st = self.c.i;
        let toks = self.ml_line()?;
        let fields = match &toks[3..] {
            [o, inner @ .., c] if o.is("(") && c.is(")") => match self.ml_path(inner) {
                Some(MatValue::Struct { fields, .. }) => Some(fields.clone()),
                _ => None,
            },
            _ => None,
        };
        let Some(fields) = fields else {
            self.c.i = st;
            return Ok(false);
        };
        let n = fields.len();
        let items = fields.into_iter().map(MatValue::Char).collect();
        self.ml.ws.insert(toks[0].raw.clone(), MatValue::Cell { dims: vec![n, 1], items });
        Ok(true)
    }

    /// `for i=1:numel(NAMES)` (or `length`) over a workspace cell of names, whose body is only
    /// `P = NAMES{i};` and `set_param_value(P, S.a.b.(P));` statements: run here.
    fn ml_for(&mut self) -> Result<bool> {
        let st = self.c.i;
        let head = self.ml_line()?;
        let body_start = self.c.i;
        let names = match &head[1..] {
            [i, eq, one, colon, f, o, cell, c]
                if eq.is("=")
                    && matches!(one.tok, Tok::Num(x) if x == 1.0)
                    && colon.is(":")
                    && (f.is_ident("numel") || f.is_ident("length"))
                    && o.is("(")
                    && c.is(")") =>
            {
                match (i.ident(), cell.ident().and_then(|n| self.ml.ws.get(n)).and_then(|v| v.as_strings())) {
                    (Some(i), Some(names)) => Some((i.to_string(), cell.raw.clone(), names)),
                    _ => None,
                }
            }
            _ => None,
        };
        let Some((index, cell, names)) = names else {
            self.c.i = st;
            return Ok(false);
        };
        let end = self.ml_skip_to(false)?;
        // the body statements (split at ';' and line ends)
        let t = self.c.t;
        let mut stmts: Vec<&[Token]> = Vec::new();
        let mut a = body_start;
        for k in body_start..=end {
            if k == end || t[k].is(";") || (k > a && t[k].loc != t[k - 1].loc) {
                if k > a {
                    stmts.push(&t[a..k]);
                }
                a = if k < end && t[k].is(";") { k + 1 } else { k };
            }
        }
        enum Step<'t> {
            Name(&'t str),
            SetParam(&'t str, &'t [Token], &'t Token),
        }
        let mut steps = Vec::new();
        for s in &stmts {
            let step = match s {
                [v, eq, c, o, i, cl]
                    if eq.is("=") && c.raw == cell && o.is("{") && i.is_ident(&index) && cl.is("}") =>
                {
                    v.ident().map(Step::Name)
                }
                [f, o, p, comma, rest @ .., dot, o2, p2, c2, c]
                    if f.is_ident("set_param_value")
                        && o.is("(")
                        && comma.is(",")
                        && dot.is(".")
                        && o2.is("(")
                        && c2.is(")")
                        && c.is(")")
                        && p.ident().is_some()
                        && p2.raw == p.raw =>
                {
                    Some(Step::SetParam(p.ident().unwrap(), rest, f))
                }
                _ => None,
            };
            match step {
                Some(s) => steps.push(s),
                None => {
                    self.c.i = st;
                    return Ok(false);
                }
            }
        }
        for name in &names {
            let mut vars: Vec<(&str, &str)> = Vec::new();
            for s in &steps {
                match s {
                    Step::Name(v) => vars.push((v, name)),
                    Step::SetParam(p, path, at) => {
                        let Some(&(_, pname)) = vars.iter().rev().find(|(v, _)| v == p) else {
                            bail!("{}: Unrecognized function or variable '{p}'.", at.loc)
                        };
                        let value = self.ml_path(path).and_then(|s| s.field(pname)).and_then(|v| v.as_scalar());
                        let Some(value) = value else {
                            bail!("{}: set_param_value: {}.{pname} is not a scalar", at.loc, raw_join(path))
                        };
                        match self.b.symbol(pname) {
                            Some(s) if s.kind == SymbolKind::Param => {
                                let k = s.index;
                                self.x_param_assignment(k, pname, value);
                                self.b.params[k] = value;
                            }
                            _ => bail!("{}: Parameter name {pname} doesn't exist", at.loc),
                        }
                    }
                }
            }
        }
        self.c.i = end;
        self.ml_eat_keyword_line();
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::octave_condition;
    use crate::modfile::lexer::tokenize;
    use crate::modfile::macro_proc::SrcLine;
    use std::sync::Arc;

    fn cond(s: &str) -> Option<bool> {
        let lines = vec![SrcLine { file: Arc::from("t.mod"), line: 1, text: s.to_string() }];
        octave_condition(&tokenize(&lines).unwrap())
    }

    #[test]
    fn isoctave_conditions() {
        assert_eq!(cond("isoctave"), Some(false));
        assert_eq!(cond("isoctave()"), Some(false));
        assert_eq!(cond("~isoctave % comment"), Some(true));
        assert_eq!(cond("~isoctave()"), Some(true));
        assert_eq!(cond("isoctave && options_.solve_algo==0"), Some(false));
        assert_eq!(cond("~isoctave && matlab.internal.parallel.isPCTInstalled"), None);
        assert_eq!(cond("~isoctave || x"), Some(true));
        assert_eq!(cond("isoctave || x"), None);
        assert_eq!(
            cond(
                "(isoctave && user_has_octave_package('optim', '1.6')) || (~isoctave && user_has_matlab_license('optimization_toolbox'))"
            ),
            None
        );
        assert_eq!(cond("(isoctave && f(1)) || ~(isoctave)"), Some(true));
        assert_eq!(cond("isoctave == 0"), None);
        assert_eq!(cond("status"), None);
    }
}
