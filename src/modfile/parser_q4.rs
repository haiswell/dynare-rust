// SPDX-License-Identifier: GPL-3.0-or-later
//! Blocks parsed for `epilogue` (`EpilogueStatement`: `NAME = EXPR;`
//! lines, endogenous variables, earlier epilogue variables and parameters, with leads and
//! lags), `init2shocks(name=...)` (`ENDO [,] EXO;` lines, `M_.init2shocks.NAME`) and the TeX
//! / long names of `model_local_variable` declarations.

use super::{Ast, Parser, slice_expr, split_commas};
use crate::model::SymbolKind;
use crate::modfile::lexer::Tok;
use anyhow::{Result, bail};

impl<'a> Parser<'a> {
    pub(super) fn q4_handles(&self, kw: &str) -> bool {
        matches!(kw, "epilogue" | "init2shocks" | "external_function")
    }

    pub(super) fn q4_statement(&mut self, kw: &str) -> Result<()> {
        match kw {
            "epilogue" => self.q4_epilogue(),
            "init2shocks" => self.q4_init2shocks(),
            "external_function" => self.q4_external_function(),
            _ => unreachable!(),
        }
    }

    /// `external_function(name = NAME, nargs = N, first_deriv_provided [= F1],
    /// second_deriv_provided [= F2]);`: binds NAME to a Rust-registered function or to
    /// `NAME.m` next to the .mod file (`model::external`).
    fn q4_external_function(&mut self) -> Result<()> {
        use crate::model::external::{DerivOpt, declare};
        use crate::modfile::{OptValue, opt};
        let loc = self.c.next()?.loc.clone();
        let opts = self.header_options()?;
        for (k, _) in &opts {
            if !matches!(k.as_str(), "name" | "nargs" | "first_deriv_provided" | "second_deriv_provided") {
                bail!("{loc}: external_function: unknown option '{k}'");
            }
        }
        let name = match opt(&opts, "name") {
            Some(OptValue::Str(s)) => s.clone(),
            _ => bail!("{loc}: external_function: the 'name' option is required"),
        };
        if self.b.symbol(&name).is_some() || crate::model::expr::UnaryOp::from_name(&name).is_some() {
            bail!("{loc}: external_function: '{name}' is already declared");
        }
        let nargs = match opt(&opts, "nargs") {
            None => 1,
            Some(OptValue::Num(v)) if *v >= 1.0 && v.fract() == 0.0 => *v as usize,
            Some(_) => bail!("{loc}: external_function: nargs must be a positive integer"),
        };
        let deriv = |key: &str| -> Result<DerivOpt> {
            Ok(match opt(&opts, key) {
                None => DerivOpt::Numerical,
                Some(OptValue::Flag) => DerivOpt::SameFunction,
                Some(OptValue::Str(s)) if s == &name => DerivOpt::SameFunction,
                Some(OptValue::Str(s)) => DerivOpt::Named(s.clone()),
                Some(_) => bail!("{loc}: external_function: {key} takes a function name"),
            })
        };
        let (first, second) = (deriv("first_deriv_provided")?, deriv("second_deriv_provided")?);
        if first == DerivOpt::Numerical && second != DerivOpt::Numerical {
            bail!("{loc}: external_function: second_deriv_provided requires first_deriv_provided (as Dynare)");
        }
        let dir = std::path::Path::new(&*loc.file)
            .parent()
            .map(|d| if d.as_os_str().is_empty() { std::path::Path::new(".") } else { d });
        declare(&name, nargs, first, second, dir).map_err(|e| anyhow::anyhow!("{loc}: {e:#}"))?;
        Ok(())
    }

    /// `epilogue; NAME = EXPR; ... end;`
    fn q4_epilogue(&mut self) -> Result<()> {
        let loc = self.c.next()?.loc.clone();
        self.header_options()?;
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                break;
            }
            if self.c.peek().is_none() {
                bail!("{loc}: block 'epilogue' not closed by 'end;'");
            }
            let toks = self.c.statement()?;
            let (name, rest) = match toks {
                [n, eq, rest @ ..] if eq.is("=") && n.ident().is_some() => (n.ident().unwrap().to_string(), rest),
                _ => bail!(
                    "{}: epilogue: expected NAME = EXPRESSION;",
                    toks.first().map(|t| t.loc.to_string()).unwrap_or_default()
                ),
            };
            if self.b.symbol(&name).is_some() {
                bail!("{}: epilogue: '{name}' is already declared", toks[0].loc);
            }
            let ast = slice_expr(rest)?;
            let known: Vec<String> = self.q4.epilogue.iter().map(|e| e.name.clone()).collect();
            check_epilogue_ast(&ast, &|n| {
                if known.iter().any(|k| k == n) || n == name {
                    return Ok(());
                }
                match self.b.symbol(n).map(|s| s.kind) {
                    Some(SymbolKind::Endo) | Some(SymbolKind::Param) => Ok(()),
                    // varexo_det
                    Some(SymbolKind::Exo) if self.b.symbol(n).is_some_and(|s| self.exo_det.contains(&s.index)) => {
                        bail!(
                            "Symbol '{n}' cannot be used inside the epilogue block, because it is an exogenous deterministic variable."
                        )
                    }
                    Some(SymbolKind::Exo) => {
                        bail!(
                            "Symbol '{n}' cannot be used inside the epilogue block, because it is an exogenous variable."
                        )
                    }
                    Some(k) => bail!("Symbol '{n}' ({k:?}) cannot be used inside the epilogue block"),
                    None => bail!("epilogue: unknown symbol '{n}'"),
                }
            })?;
            self.q4.epilogue.push(crate::q4::epilogue::EpilogueEq { name, ast });
        }
        Ok(())
    }

    /// `init2shocks(name=NAME); ENDO [,] EXO; ... end;` (`M_.init2shocks.NAME`).
    fn q4_init2shocks(&mut self) -> Result<()> {
        let loc = self.c.next()?.loc.clone();
        let opts = self.header_options()?;
        let name = match crate::modfile::opt(&opts, "name") {
            Some(crate::modfile::OptValue::Str(s)) => s.clone(),
            _ => "default".to_string(),
        };
        let mut pairs = Vec::new();
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                break;
            }
            if self.c.peek().is_none() {
                bail!("{loc}: block 'init2shocks' not closed by 'end;'");
            }
            let toks = self.c.statement()?;
            let ids: Vec<&str> = toks.iter().filter_map(|t| t.ident()).collect();
            if ids.len() != 2 || toks.iter().any(|t| !(t.ident().is_some() || t.is(","))) {
                bail!(
                    "{}: init2shocks: expected ENDOGENOUS [,] EXOGENOUS;",
                    toks.first().map(|t| t.loc.to_string()).unwrap_or_default()
                );
            }
            match self.b.symbol(ids[0]).map(|s| s.kind) {
                Some(SymbolKind::Endo) => {}
                _ => bail!("{}: init2shocks: {} is not an endogenous variable", toks[0].loc, ids[0]),
            }
            match self.b.symbol(ids[1]).map(|s| s.kind) {
                Some(SymbolKind::Exo) => {}
                _ => bail!("{}: init2shocks: {} is not an exogenous variable", toks[0].loc, ids[1]),
            }
            pairs.push((ids[0].to_string(), ids[1].to_string()));
        }
        self.q4.init2shocks.retain(|(n, _)| n != &name);
        self.q4.init2shocks.push((name, pairs));
        Ok(())
    }

    /// `model_local_variable x $\tex$ (long_name='...') y ...;` -> TeX / long names.
    pub(super) fn q4_model_local_variable_labels(&mut self, toks: &[crate::modfile::lexer::Token]) {
        let mut k = 0;
        while k < toks.len() {
            if let Tok::Ident(n) = &toks[k].tok {
                let mut tex = n.replace('_', "\\_");
                let mut long = n.clone();
                let mut j = k + 1;
                while j < toks.len() {
                    if let Tok::Tex(s) = &toks[j].tok {
                        tex = s.clone();
                        j += 1;
                    } else if toks[j].is("(") {
                        let st = j;
                        let mut depth = 0;
                        while j < toks.len() {
                            if toks[j].is("(") {
                                depth += 1;
                            } else if toks[j].is(")") {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            j += 1;
                        }
                        for item in split_commas(&toks[st + 1..j.min(toks.len())]) {
                            if let [kk, eq, v] = item
                                && kk.is_ident("long_name")
                                && eq.is("=")
                                && let Tok::Str(s) = &v.tok
                            {
                                long = s.clone();
                            }
                        }
                        j += 1;
                    } else {
                        break;
                    }
                }
                self.q4.local_labels.retain(|(m, _, _)| m != n);
                self.q4.local_labels.push((n.clone(), tex, long));
                k = j;
                continue;
            }
            k += 1;
        }
    }
}

// ---- command-line flags: stochastic, nostrict, exclude_eqs / include_eqs ----------------------

/// An equation selection: alternatives, each a (tag, value) pair.
type Selection = Vec<(String, String)>;

#[derive(Clone, Debug, Default)]
pub(crate) struct Q4Flags {
    pub stochastic: bool,
    pub nostrict: bool,
    exclude: Option<Selection>,
    include: Option<Selection>,
    /// Selections met so far (exclude or include).
    found: Vec<bool>,
}

impl Q4Flags {
    /// `exclude_eqs=` or `include_eqs=` is given.
    pub fn selects_equations(&self) -> bool {
        self.exclude.is_some() || self.include.is_some()
    }
}

/// Port of the preprocessor's parsing of `exclude_eqs=` / `include_eqs=` (a file, or a list
/// of `name` tags, or `[tagname=value]` / `[tagname=(v1, 'v2')]`).
fn parse_selection(arg: &str, cwd: Option<&std::path::Path>) -> Result<Selection> {
    let file = cwd.map_or_else(|| std::path::PathBuf::from(arg), |d| d.join(arg));
    if file.is_file() {
        let text = std::fs::read_to_string(&file)?;
        let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
        let first = lines.next().unwrap_or("");
        return Ok(if let Some(tag) = first.strip_suffix('=') {
            lines.map(|l| (tag.trim().to_string(), l.to_string())).collect()
        } else {
            std::iter::once(first).chain(lines).map(|l| ("name".to_string(), l.to_string())).collect()
        });
    }
    let unquote = |s: &str| s.trim().trim_matches('\'').trim_matches('"').to_string();
    // split at commas outside quotes
    let split = |s: &str| -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let mut q = false;
        for c in s.chars() {
            match c {
                '\'' => {
                    q = !q;
                    cur.push(c)
                }
                ',' if !q => out.push(std::mem::take(&mut cur)),
                _ => cur.push(c),
            }
        }
        out.push(cur);
        out.into_iter().map(|x| unquote(&x)).filter(|x| !x.is_empty()).collect()
    };
    let inner = match arg.trim().strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        Some(i) => i.trim(),
        None => return Ok(vec![("name".to_string(), unquote(arg))]),
    };
    match inner.split_once('=') {
        Some((tag, rest)) if !tag.contains(',') && !tag.contains('\'') => {
            let rest = rest.trim();
            let vals = match rest.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                Some(l) => split(l),
                None => vec![unquote(rest)],
            };
            Ok(vals.into_iter().map(|v| (tag.trim().to_string(), v)).collect())
        }
        _ => Ok(split(inner).into_iter().map(|v| ("name".to_string(), v)).collect()),
    }
}

/// The parser's flags from [`crate::modfile::ParseOptions`].
pub(crate) fn q4_flags(o: &crate::modfile::ParseOptions) -> Result<Q4Flags> {
    let cwd = o.cwd.as_deref();
    let exclude = o.exclude_eqs.as_deref().map(|a| parse_selection(a, cwd)).transpose()?;
    let include = o.include_eqs.as_deref().map(|a| parse_selection(a, cwd)).transpose()?;
    if exclude.is_some() && include.is_some() {
        bail!("You may only pass one of `include_eqs` and `exclude_eqs`");
    }
    let n = exclude.as_ref().or(include.as_ref()).map_or(0, Vec::len);
    Ok(Q4Flags { stochastic: o.stochastic, nostrict: o.nostrict, exclude, include, found: vec![false; n] })
}

impl<'a> Parser<'a> {
    /// `nostrict`: a symbol of the model block that is not declared becomes exogenous.
    pub(super) fn q4_declare_unknown(&mut self, ast: &Ast) -> Result<()> {
        if !self.q4_flags.nostrict {
            return Ok(());
        }
        fn walk(a: &Ast, out: &mut Vec<String>) {
            match a {
                Ast::Num(_) => {}
                Ast::Var { name, .. } => out.push(name.clone()),
                Ast::Un(_, x) => walk(x, out),
                Ast::Bin(_, x, y) => {
                    walk(x, out);
                    walk(y, out);
                }
                Ast::Op(_, args) => args.iter().for_each(|x| walk(x, out)),
            }
        }
        let mut names = Vec::new();
        walk(ast, &mut names);
        for n in names {
            if self.b.symbol(&n).is_none() && super::builtin_const(&n).is_none() {
                println!("WARNING: Unknown symbol '{n}' in the model block: declared exogenous (nostrict).");
                self.b.declare(&n, SymbolKind::Exo)?;
                self.labels.push(SymbolKind::Exo, n.replace('_', "\\_"), n.clone());
            }
        }
        Ok(())
    }

    /// End of a model block: `exclude_eqs` / `include_eqs` (`DynamicModel::includeExcludeEquations`).
    pub(super) fn q4_after_model_block(&mut self) -> Result<()> {
        let (sel, include) = match (&self.q4_flags.exclude, &self.q4_flags.include) {
            (Some(s), None) => (s.clone(), false),
            (None, Some(s)) => (s.clone(), true),
            _ => return Ok(()),
        };
        let tags: Vec<Vec<(String, String)>> = self.b.equation_tags().to_vec();
        let has = |t: &[(String, String)], (k, v): &(String, String)| t.iter().any(|(a, b)| a == k && b == v);
        for (k, s) in sel.iter().enumerate() {
            if tags.iter().any(|t| has(t, s)) {
                self.q4_flags.found[k] = true;
            }
        }
        if !sel.iter().any(|s| tags.iter().any(|t| has(t, s))) {
            // no equation selected: nothing removed (ModelTree::includeExcludeEquations)
            return Ok(());
        }
        let removal: Vec<Vec<(String, String)>> = if include {
            let mut r = Vec::new();
            for t in &tags {
                if !t.is_empty() && !sel.iter().any(|s| has(t, s)) {
                    r.push(t.clone());
                }
            }
            if tags.iter().any(Vec::is_empty) {
                // the equations without tags are removed too: they get a tag to be removed by
                let tag = ("#untagged".to_string(), String::new());
                self.b.tag_untagged_equations(&tag);
                for (_, _, t) in self.original.equations.iter_mut().filter(|(_, _, t)| t.is_empty()) {
                    t.push(tag.clone());
                }
                r.push(vec![tag]);
            }
            r
        } else {
            sel.iter().filter(|s| tags.iter().any(|t| has(t, s))).map(|s| vec![s.clone()]).collect()
        };
        if removal.is_empty() {
            return Ok(());
        }
        let n_before = self.b.equation_tags().len();
        let excluded = self.b.remove_equations(&removal, true)?;
        let n_removed = n_before - self.b.equation_tags().len();
        if excluded.len() != n_removed {
            bail!(
                "{}_eqs: every excluded equation must have an 'endogenous' tag or a left-hand side with a single endogenous variable",
                if include { "include" } else { "exclude" }
            );
        }
        self.remove_original_equations(&removal);
        for v in excluded {
            if self.b.is_used(&v) {
                self.q4_endo_to_exo(&v)?;
            } else {
                self.remove_symbol(&v)?;
            }
        }
        Ok(())
    }

    /// After the whole file: every `exclude_eqs` / `include_eqs` tag must have been met.
    pub(super) fn q4_check_exclusions(&self) -> Result<()> {
        let sel = self.q4_flags.exclude.as_ref().or(self.q4_flags.include.as_ref());
        if let Some(sel) = sel {
            for (s, f) in sel.iter().zip(&self.q4_flags.found) {
                if !f {
                    bail!("Could not find tag: {} = {}", s.0, s.1);
                }
            }
        }
        Ok(())
    }

    /// An endogenous variable becomes exogenous (excluded by `exclude_eqs` but still used): the
    /// indices held by the statements parsed so far follow the model's.
    pub(super) fn q4_endo_to_exo(&mut self, name: &str) -> Result<()> {
        let in_shocks_or_est = !self.shocks.is_empty() || !self.est.is_empty() || !self.est_init.is_empty();
        if in_shocks_or_est {
            bail!(
                "exclude_eqs/include_eqs: '{name}' becomes exogenous after a shocks or estimated_params block; move the model block first"
            );
        }
        let (j, p) = self.b.endo_to_exo(name)?;
        let fix_endo = |v: &mut usize| -> bool {
            if *v == j {
                return false;
            }
            if *v > j {
                *v -= 1;
            }
            true
        };
        let fix_exo = |v: &mut usize| {
            if *v >= p {
                *v += 1;
            }
        };
        self.initval.retain_mut(|(k, _)| fix_endo(k));
        self.endval.retain_mut(|(k, _)| fix_endo(k));
        // histval entries are (endogenous?, index, lag, value); the variable's own entries stay,
        // on the exogenous variable it becomes (Dynare's histval keeps the symbol id)
        self.histval.iter_mut().for_each(|(endo, k, _, _)| {
            if !*endo {
                fix_exo(k);
            } else if *k == j {
                (*endo, *k) = (false, p);
            } else if *k > j {
                *k -= 1;
            }
        });
        self.predetermined.retain_mut(fix_endo);
        self.varobs.retain(|v| v != name);
        self.initval_exo.iter_mut().for_each(|(k, _)| fix_exo(k));
        self.exo_det.iter_mut().for_each(fix_exo);
        self.det_shocks.iter_mut().for_each(|d| fix_exo(&mut d.exo));
        let tex = self.labels.endo_tex.remove(j);
        let long = self.labels.endo_long.remove(j);
        self.labels.exo_tex.insert(p, tex);
        self.labels.exo_long.insert(p, long);
        println!("NOTE: {name} is now an exogenous variable (its equation was excluded).");
        Ok(())
    }
}

fn check_epilogue_ast(a: &Ast, ok: &dyn Fn(&str) -> Result<()>) -> Result<()> {
    match a {
        Ast::Num(_) => Ok(()),
        Ast::Var { name, loc, .. } => ok(name).map_err(|e| anyhow::anyhow!("{loc}: {e}")),
        Ast::Un(_, x) => check_epilogue_ast(x, ok),
        Ast::Bin(_, x, y) => {
            check_epilogue_ast(x, ok)?;
            check_epilogue_ast(y, ok)
        }
        Ast::Op(op, args) => {
            if matches!(op, super::SpecialOp::SteadyState) {
                bail!("The STEADY_STATE() operator is forbidden in epilogue block");
            }
            if matches!(op, super::SpecialOp::Expectation(_) | super::SpecialOp::Diff) {
                bail!("The {}() operator is not supported in the epilogue block", op.name());
            }
            args.iter().try_for_each(|x| check_epilogue_ast(x, ok))
        }
    }
}
