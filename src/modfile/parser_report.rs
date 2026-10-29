// SPDX-License-Identifier: GPL-3.0-or-later
//! Reporting statements (`crate::reporting`): `r = report(...)`, `r = r.addPage(...)`,
//! `r.write();`, `ds = dseries(...)`, `ds = [ds dseries(y_e, 2014q3, 'y')]`, parsed into a
//! [`Stmt`] kept as a `REPORT_STATEMENT` command and run by the driver.

use anyhow::{Result, anyhow, bail};

use super::Parser;
use super::shk::date_at;
use crate::modfile::lexer::{Tok, Token};
use crate::modfile::{Command, OptValue};
use crate::reporting::{Expr, REPORT_STATEMENT, Stmt};

/// Methods of the report classes.
const METHODS: &[&str] = &[
    "write",
    "compile",
    "addPage",
    "addSection",
    "addGraph",
    "addTable",
    "addSeries",
    "addVspace",
    "addData",
    "addParagraph",
];

/// Whether the right-hand side is a reporting / dseries expression the workspace evaluator
/// cannot run at parse time (it needs the IRFs of a stoch_simul, or builds a report).
pub(super) fn is_report_rhs(toks: &[Token]) -> bool {
    toks.iter().enumerate().any(|(k, t)| {
        t.is_ident("report") && toks.get(k + 1).is_some_and(|n| n.is("("))
            || t.is_ident("dseries")
            || (t.is(".")
                && toks.get(k + 1).and_then(|n| n.ident()).is_some_and(|m| METHODS.contains(&m) && m != "write"))
    })
}

impl<'a> Parser<'a> {
    /// `r.write();` / `r.compile();` / `r.addX(...);` (a method call statement).
    pub(super) fn rep_handles(&self, kw: &str) -> bool {
        let t = |k| self.c.peek_at(k);
        !matches!(kw, "options_" | "M_" | "oo_" | "estim_params_" | "bayestopt_" | "estimation_info")
            && t(1).is_some_and(|x| x.is("."))
            && t(2).and_then(|x| x.ident()).is_some_and(|m| METHODS.contains(&m))
            && t(3).is_some_and(|x| x.is("("))
    }

    pub(super) fn rep_statement(&mut self) -> Result<()> {
        let toks = self.c.statement()?;
        self.rep_push(None, toks)
    }

    /// Stores `lhs = toks` (or `toks;`) as a reporting command.
    pub(super) fn rep_push(&mut self, lhs: Option<&str>, toks: &[Token]) -> Result<()> {
        let text: Vec<&str> = toks.iter().map(|t| t.raw.as_str()).collect();
        let text = match lhs {
            Some(n) => format!("{n} = {}", text.join(" ")),
            None => text.join(" "),
        };
        let loc = toks.first().map(|t| t.loc.to_string()).unwrap_or_default();
        // the functions the reporting driver evaluates (`surgibbs(dseries(...), ...)` is not one)
        if let Some(k) = (0..toks.len()).find(|&k| {
            toks[k].ident().is_some_and(|n| !matches!(n, "dseries" | "report" | "dates"))
                && toks.get(k + 1).is_some_and(|n| n.is("("))
                && !(k > 0 && toks[k - 1].is("."))
        }) {
            bail!("{loc}: function '{}' is not supported", toks[k].raw);
        }
        let mut p = RepParser { t: toks, i: 0 };
        let rhs = p.expr().map_err(|e| anyhow!("{loc}: cannot parse `{text}` ({e})"))?;
        if p.i < toks.len() {
            bail!("{loc}: cannot parse `{text}` (at '{}')", toks[p.i].raw);
        }
        let st = Stmt { lhs: lhs.map(str::to_string), rhs, text: format!("{loc}: {text}") };
        self.commands
            .push(Command::Other(REPORT_STATEMENT.into(), vec![(serde_json::to_string(&st)?, OptValue::Flag)]));
        Ok(())
    }
}

struct RepParser<'t> {
    t: &'t [Token],
    i: usize,
}

impl RepParser<'_> {
    fn peek_is(&self, p: &str) -> bool {
        self.t.get(self.i).is_some_and(|t| t.is(p))
    }

    /// MATLAB line continuations (`...`) reach the parser as three dots.
    fn skip_continuations(&mut self) {
        while self.peek_is(".")
            && self.t.get(self.i + 1).is_some_and(|t| t.is("."))
            && self.t.get(self.i + 2).is_some_and(|t| t.is("."))
        {
            self.i += 3;
        }
    }

    fn expect(&mut self, p: &str) -> Result<()> {
        self.skip_continuations();
        if !self.peek_is(p) {
            let at = self.t.get(self.i).map_or("end of statement", |t| t.raw.as_str());
            bail!("expected '{p}', found '{at}'");
        }
        self.i += 1;
        Ok(())
    }

    fn expr(&mut self) -> Result<Expr> {
        let a = self.primary()?;
        if self.peek_is(":") {
            self.i += 1;
            let b = self.primary()?;
            return Ok(Expr::Range(Box::new(a), Box::new(b)));
        }
        Ok(a)
    }

    /// Comma- (or blank-) separated elements up to `close`.
    fn list(&mut self, close: &str) -> Result<Vec<Expr>> {
        let mut v = vec![];
        loop {
            self.skip_continuations();
            if self.peek_is(close) {
                self.i += 1;
                return Ok(v);
            }
            if self.peek_is(",") {
                self.i += 1;
                continue;
            }
            if self.i >= self.t.len() {
                bail!("missing '{close}'");
            }
            v.push(self.expr()?);
        }
    }

    fn primary(&mut self) -> Result<Expr> {
        self.skip_continuations();
        if let Some((d, n)) = date_at(self.t, self.i) {
            self.i += n;
            return Ok(Expr::Date(d.label()));
        }
        let Some(t) = self.t.get(self.i) else { bail!("unexpected end of statement") };
        self.i += 1;
        let mut e = match &t.tok {
            Tok::Num(x) => return Ok(Expr::Num(*x)),
            Tok::Punct("-") => match self.primary()? {
                Expr::Num(x) => return Ok(Expr::Num(-x)),
                _ => bail!("unsupported '-'"),
            },
            Tok::Str(s) => return Ok(Expr::Str(s.clone())),
            Tok::Punct("{") => return Ok(Expr::Cell(self.list("}")?)),
            Tok::Punct("[") => return Ok(Expr::Concat(self.list("]")?)),
            Tok::Punct("(") => {
                let e = self.expr()?;
                self.expect(")")?;
                e
            }
            Tok::Ident(n) => {
                if self.peek_is("(") {
                    self.i += 1;
                    Expr::Call(n.clone(), self.list(")")?)
                } else {
                    Expr::Var(n.clone())
                }
            }
            _ => bail!("unexpected '{}'", t.raw),
        };
        // .field / .method(args)
        while self.peek_is(".") && self.t.get(self.i + 1).and_then(|t| t.ident()).is_some() {
            let name = self.t[self.i + 1].ident().unwrap().to_string();
            self.i += 2;
            if self.peek_is("(") {
                self.i += 1;
                e = Expr::Method(Box::new(e), name, self.list(")")?);
            } else {
                e = Expr::Field(Box::new(e), name);
            }
        }
        Ok(e)
    }
}
