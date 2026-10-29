// SPDX-License-Identifier: GPL-3.0-or-later
//! ---- pac ----: `pac_model`, `pac_target_info`, the `pac_expectation(m)` and
//! `pac_target_nonstationary(m)` operators and the `pac.*` statements.
//!
//! As the preprocessor does, the PAC equation (the one that contains `pac_expectation`) is
//! decomposed into its error correction term `ec*(target(-1) - lhs(-1))` and autoregressive terms
//! `ar_k*diff(lhs(-k))`; the operators are replaced by auxiliary endogenous variables: the PAC
//! expectation is a linear combination of the companion state of the auxiliary model
//! (parameters `h_<pac>_*`), or, without auxiliary model, the recursion of the MCE case
//! (`mce_Z1_<pac>`). The parser declares the parameters and describes the models
//! ([`PacSpec`]); the variables and their equations are created in the preprocessor's transform
//! pass (`model::dtree::pac`), after the auxiliary variables of the diff operators.

use super::{Ast, Cursor, Parser, slice_expr_nums, split_commas};
use crate::model::SymbolKind;
use crate::model::dtree::pac::{CompSpec, PacKind, PacSpec};
use crate::model::expr::{BinaryOp, UnaryOp};
use crate::modfile::lexer::{Loc, Tok, Token};
use crate::modfile::{Command, OptValue, SpecialOp};
use crate::varmodels::pac::{PacBgp, PacBgpTerm, PacComponent, PacModel};
use crate::varmodels::{self as vm, Discount};
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;

pub use crate::model::dtree::pac::{PE, PNS};

struct PacDecl {
    name: String,
    aux: Option<String>,
    discount: Discount,
    /// The expression and its rank among the preamble expressions of the model's tree.
    growth: Option<(Ast, usize)>,
    kind: String,
    auxname: Option<String>,
    loc: String,
}

struct CompDecl {
    expr: Ast,
    pre: usize,
    kind: String,
    auxname: Option<String>,
    growth: Option<(Ast, usize)>,
}

#[derive(Default)]
struct InfoDecl {
    target: Option<(Ast, usize)>,
    auxname_ns: Option<String>,
    comps: Vec<CompDecl>,
}

/// Parser state of the PAC models.
#[derive(Default)]
pub(super) struct PacParse {
    decls: Vec<PacDecl>,
    infos: HashMap<String, InfoDecl>,
    /// Every equation of the model blocks seen in the prepasses (tag, lhs, rhs), and the
    /// text of its numerical constants.
    eqs: Vec<(Option<String>, Ast, Ast)>,
    eq_nums: Vec<Vec<String>>,
    /// Replacement of the operators: placeholder name -> expression.
    subst: HashMap<String, Ast>,
    /// `eparams.name = value;` structures of the estimation statements, in assignment order.
    structs: HashMap<String, Vec<(String, f64)>>,
    /// `x = dseries(...)` / `x = dates(...)...;` right-hand sides (data and range variables).
    vars: HashMap<String, Vec<Token>>,
    /// `edata.ez = dseries(NaN, ...);`: series of a dseries variable set to NaN (the residual).
    nan_fields: HashMap<String, Vec<String>>,
    models: Vec<PacModel>,
    /// dseries computed when the commands run (`X = simul_backward_model(...)`, `Y = X;`).
    runtime: std::collections::HashSet<String>,
}

fn num(v: f64) -> Ast {
    Ast::Num(v)
}

fn var(n: &str, lag: i32, loc: &Loc) -> Ast {
    Ast::Var { name: n.to_string(), lag, loc: loc.clone() }
}

fn bin(op: BinaryOp, a: Ast, b: Ast) -> Ast {
    Ast::Bin(op, Box::new(a), Box::new(b))
}

fn add(a: Option<Ast>, b: Ast) -> Option<Ast> {
    Some(match a {
        None => b,
        Some(x) => bin(BinaryOp::Add, x, b),
    })
}

fn no_loc() -> Loc {
    Loc { file: std::sync::Arc::from("<pac>"), line: 0 }
}

/// Placeholder for `pac_expectation(m)` / `pac_target_nonstationary(m)` in the expression parser.
pub(super) fn placeholder(c: &mut Cursor, loc: &Loc, prefix: &str) -> Result<Ast> {
    c.expect("(")?;
    if c.at_ident("model_name") && c.peek_at(1).is_some_and(|t| t.is("=")) {
        c.i += 2;
    }
    let t = c.next()?;
    let n = match &t.tok {
        Tok::Ident(s) | Tok::Str(s) => s.clone(),
        _ => bail!("{}: expected the name of a PAC model", t.loc),
    };
    // pac_expectation(model_name = m, ...) : ignore further arguments
    while c.at(",") {
        c.i += 1;
        while c.peek().is_some_and(|t| !t.is(",") && !t.is(")")) {
            c.i += 1;
        }
    }
    c.expect(")")?;
    Ok(Ast::Var { name: format!("{prefix}{n}"), lag: 0, loc: loc.clone() })
}

// ---------------------------------------------------------------- equation analysis

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum Atom {
    Level(String, i32),
    /// A unary function of a variable (`log(x(-1))`): its key, lag and expression at `t`.
    Expr(String, i32, Ast),
    Diff(String, i32),
    Pe(String),
    Pns(String),
}

type KindFn<'a> = &'a dyn Fn(&str) -> Option<SymbolKind>;

fn is_const(a: &Ast, kind: KindFn) -> bool {
    match a {
        Ast::Num(_) => true,
        Ast::Var { name, .. } => {
            !(name.starts_with(PE) || name.starts_with(PNS))
                && !matches!(kind(name), Some(SymbolKind::Endo) | Some(SymbolKind::Exo))
        }
        Ast::Un(_, x) => is_const(x, kind),
        Ast::Bin(_, x, y) => is_const(x, kind) && is_const(y, kind),
        Ast::Op(_, v) => v.iter().all(|x| is_const(x, kind)),
    }
}

fn lin(a: &Ast, coef: &Ast, kind: KindFn, out: &mut Vec<(Ast, Atom)>) {
    if is_const(a, kind) {
        return;
    }
    let mulc = |c: &Ast, x: &Ast| bin(BinaryOp::Mul, c.clone(), x.clone());
    match a {
        Ast::Var { name, lag, .. } => {
            if let Some(m) = name.strip_prefix(PE) {
                out.push((coef.clone(), Atom::Pe(m.to_string())));
            } else if let Some(m) = name.strip_prefix(PNS) {
                out.push((coef.clone(), Atom::Pns(m.to_string())));
            } else {
                out.push((coef.clone(), Atom::Level(name.clone(), *lag)));
            }
        }
        Ast::Un(UnaryOp::Neg, x) => lin(x, &mulc(coef, &num(-1.0)), kind, out),
        Ast::Un(..) => {
            if let Ok(Some((e, key, lag))) = vm::key_of(a, kind) {
                out.push((coef.clone(), Atom::Expr(key, -(lag as i32), e)));
            }
        }
        Ast::Bin(BinaryOp::Add, x, y) => {
            lin(x, coef, kind, out);
            lin(y, coef, kind, out);
        }
        Ast::Bin(BinaryOp::Sub, x, y) => {
            lin(x, coef, kind, out);
            lin(y, &mulc(coef, &num(-1.0)), kind, out);
        }
        Ast::Bin(BinaryOp::Mul, x, y) => {
            if is_const(x, kind) {
                lin(y, &mulc(coef, x), kind, out);
            } else if is_const(y, kind) {
                lin(x, &mulc(coef, y), kind, out);
            }
        }
        Ast::Bin(BinaryOp::Div, x, y) if is_const(y, kind) => {
            lin(x, &bin(BinaryOp::Div, coef.clone(), (**y).clone()), kind, out)
        }
        Ast::Op(SpecialOp::Diff, args) => match args.first() {
            Some(Ast::Var { name, lag, .. }) => out.push((coef.clone(), Atom::Diff(name.clone(), *lag))),
            // diff(log(x(-1))): the auxiliary variable of the unary function
            Some(x @ Ast::Un(..)) => {
                if let Ok(Some((_, key, lag))) = vm::key_of(x, kind) {
                    out.push((coef.clone(), Atom::Diff(key, -(lag as i32))));
                }
            }
            _ => {}
        },
        _ => {}
    }
}

/// A coefficient of the form `±param` (times numbers): `(parameter, factor)`.
fn param_sign(c: &Ast, kind: KindFn) -> Option<(Option<String>, f64)> {
    match c {
        Ast::Num(v) => Some((None, *v)),
        Ast::Var { name, .. } if kind(name) == Some(SymbolKind::Param) => Some((Some(name.clone()), 1.0)),
        Ast::Un(UnaryOp::Neg, x) => param_sign(x, kind).map(|(p, v)| (p, -v)),
        Ast::Bin(BinaryOp::Mul, x, y) => {
            let ((p, v), (q, w)) = (param_sign(x, kind)?, param_sign(y, kind)?);
            if p.is_some() && q.is_some() { None } else { Some((p.or(q), v * w)) }
        }
        Ast::Bin(BinaryOp::Div, x, y) => match (param_sign(x, kind)?, param_sign(y, kind)?) {
            ((p, v), (None, w)) => Some((p, v / w)),
            _ => None,
        },
        _ => None,
    }
}

fn contains_var(a: &Ast, pred: &dyn Fn(&str) -> bool) -> bool {
    match a {
        Ast::Num(_) => false,
        Ast::Var { name, .. } => pred(name),
        Ast::Un(_, x) => contains_var(x, pred),
        Ast::Bin(_, x, y) => contains_var(x, pred) || contains_var(y, pred),
        Ast::Op(_, v) => v.iter().any(|x| contains_var(x, pred)),
    }
}

/// `decomposeAdditiveTerms`: the additive terms of `a` with their signs.
fn additive_terms(a: &Ast, sign: f64, out: &mut Vec<(Ast, f64)>) {
    match a {
        Ast::Bin(BinaryOp::Add, x, y) => {
            additive_terms(x, sign, out);
            additive_terms(y, sign, out);
        }
        Ast::Bin(BinaryOp::Sub, x, y) => {
            additive_terms(x, sign, out);
            additive_terms(y, -sign, out);
        }
        Ast::Un(UnaryOp::Neg, x) => additive_terms(x, -sign, out),
        _ => out.push((a.clone(), sign)),
    }
}

/// Decomposition of a PAC equation with rule of thumb agents (the preprocessor's
/// `getPacOptimizingShareAndExprNodes`): `rhs = share*(optim) + (1-share)*(non_optim) + ...`.
struct ShareSplit {
    share: String,
    optim: Ast,
    non_optim: Ast,
}

fn split_share(rhs: &Ast, kind: KindFn) -> Option<ShareSplit> {
    let mut terms = Vec::new();
    additive_terms(rhs, 1.0, &mut terms);
    let is_pe = |n: &str| n.starts_with(PE);
    // pac_expectation is additive in the expression: no optimizing share
    if terms.iter().any(|(t, _)| matches!(t, Ast::Var { name, .. } if is_pe(name))) {
        return None;
    }
    let param = |a: &Ast| match a {
        Ast::Var { name, .. } if kind(name) == Some(SymbolKind::Param) => Some(name.clone()),
        _ => None,
    };
    let (i, share, optim) = terms.iter().enumerate().find_map(|(i, (t, _))| match t {
        Ast::Bin(BinaryOp::Mul, x, y) => {
            if let Some(p) = param(x)
                && contains_var(y, &is_pe)
            {
                Some((i, p, (**y).clone()))
            } else if let Some(p) = param(y)
                && contains_var(x, &is_pe)
            {
                Some((i, p, (**x).clone()))
            } else {
                None
            }
        }
        _ => None,
    })?;
    terms.remove(i);
    let one_minus = |a: &Ast| matches!(a, Ast::Bin(BinaryOp::Sub, o, p) if matches!(**o, Ast::Num(v) if v == 1.0) && param(p).as_deref() == Some(share.as_str()));
    // the remaining terms are the additive part
    let non_optim = terms.iter().find_map(|(t, _)| match t {
        Ast::Bin(BinaryOp::Mul, x, y) if one_minus(x) => Some((**y).clone()),
        Ast::Bin(BinaryOp::Mul, x, y) if one_minus(y) => Some((**x).clone()),
        _ => None,
    })?;
    Some(ShareSplit { share, optim, non_optim })
}

/// A term `scaling_factor*param*variable` (`param` optional): `(param, variable, factor)`.
fn scaled_term(a: &Ast, kind: KindFn) -> Option<(Option<String>, String, f64)> {
    fn walk(a: &Ast, kind: KindFn, p: &mut Option<String>, v: &mut Option<String>, s: &mut f64) -> bool {
        match a {
            Ast::Num(x) => {
                *s *= x;
                true
            }
            Ast::Var { name, .. } if kind(name) == Some(SymbolKind::Param) => p.replace(name.clone()).is_none(),
            Ast::Var { name, .. } => {
                !name.starts_with(PE) && !name.starts_with(PNS) && v.replace(name.clone()).is_none()
            }
            Ast::Un(UnaryOp::Neg, x) => {
                *s = -*s;
                walk(x, kind, p, v, s)
            }
            Ast::Bin(BinaryOp::Mul, x, y) => walk(x, kind, p, v, s) && walk(y, kind, p, v, s),
            Ast::Bin(BinaryOp::Div, x, y) => match **y {
                Ast::Num(d) => {
                    *s /= d;
                    walk(x, kind, p, v, s)
                }
                _ => false,
            },
            _ => false,
        }
    }
    let (mut p, mut v, mut s) = (None, None, 1.0);
    if walk(a, kind, &mut p, &mut v, &mut s) { v.map(|v| (p, v, s)) } else { None }
}

/// The exogenous terms of a PAC equation (`M_.pac.<m>.optim_additive` in the optimizing
/// part `share*(...)`, `non_optimizing_behaviour` in `(1-share)*(...)`, `additive` outside),
/// with the default `bgp` (false) that `pac.initialize` gives them.
fn bgp_terms(rhs: &Ast, split: Option<&ShareSplit>, skip: &[&String], kind: KindFn) -> Vec<PacBgpTerm> {
    let mut out = Vec::new();
    let mut push = |group: &'static str, a: &Ast| {
        let mut terms = Vec::new();
        additive_terms(a, 1.0, &mut terms);
        for (t, sign) in terms {
            if contains_var(&t, &|n| skip.iter().any(|s| *s == n)) {
                continue;
            }
            if let Some((param, var, s)) = scaled_term(&t, kind) {
                out.push(PacBgpTerm { group, var, param, scaling: sign * s, bgp: PacBgp::Bool(false) });
            }
        }
    };
    match split {
        Some(sp) => {
            push("optim_additive", &sp.optim);
            push("non_optimizing_behaviour", &sp.non_optim);
            let mut terms = Vec::new();
            additive_terms(rhs, 1.0, &mut terms);
            let is_share = |a: &Ast| matches!(a, Ast::Var { name, .. } if *name == sp.share);
            let one_minus = |a: &Ast| matches!(a, Ast::Bin(BinaryOp::Sub, o, p) if matches!(**o, Ast::Num(v) if v == 1.0) && is_share(p));
            for (t, sign) in terms {
                let part = matches!(&t, Ast::Bin(BinaryOp::Mul, x, y) if is_share(x) || is_share(y) || one_minus(x) || one_minus(y));
                if !part {
                    let t = if sign < 0.0 { Ast::Un(UnaryOp::Neg, Box::new(t)) } else { t };
                    push("additive", &t);
                }
            }
        }
        None => push("additive", rhs),
    }
    out
}

fn tex(n: &str) -> String {
    n.replace('_', "\\_")
}

impl<'a> Parser<'a> {
    pub(super) fn pac_handles(&self, kw: &str) -> bool {
        let next_is = |p: &str| self.c.peek_at(1).is_some_and(|t| t.is(p));
        matches!(kw, "pac_model" | "pac_target_info") || (kw == "pac" && next_is("."))
    }

    /// `s.f = number;`, `x = dseries(...)`, `x = dates(...)...` or `clear x`: the MATLAB
    /// statements that set up `pac.estimate.*`.
    pub(super) fn pac_assign_ahead(&self) -> bool {
        let t = |k| self.c.peek_at(k);
        let id = t(0).and_then(|x| x.ident());
        if id == Some("clear") && t(1).is_some_and(|x| x.ident().is_some()) && !self.b.symbol("clear").is_some() {
            return true;
        }
        let Some(id) = id else { return false };
        // `s.f = v;` is a PAC estimation structure only for a file with PAC models (or a
        // structure already used as such); `options_.X = V;`, `M_.`/`oo_.` assignments go
        // to the generic MATLAB-statement handling.
        let field_assign = t(1).is_some_and(|x| x.is("."))
            && t(2).is_some_and(|x| x.ident().is_some())
            && t(3).is_some_and(|x| x.is("="))
            && !matches!(
                id,
                "options_" | "M_" | "oo_" | "estim_params_" | "bayestopt_" | "estimation_info" | "oo_recursive_"
            )
            && (!self.pac.decls.is_empty()
                || !self.pac.models.is_empty()
                || self.pac.structs.contains_key(id)
                || self.pac.runtime.contains(id));
        if !t(1).is_some_and(|x| x.is("=")) {
            return field_assign;
        }
        // `x = dates(...)...`, or `x = 2005Q1:2010Q4;` (a date-literal range, which the
        // preprocessor turns into `dates('2005Q1'):dates('2010Q4')`)
        field_assign || t(2).is_some_and(|x| x.is_ident("dates")) || {
            let mut v = Vec::new();
            let mut k = 2;
            while let Some(tk) = t(k).filter(|x| !x.is(";")) {
                v.push(tk.clone());
                k += 1;
            }
            matches!(v.first().map(|x| &x.tok), Some(Tok::Num(_))) && parse_date_range(&v).is_ok()
        }
    }

    /// `x = dseries(...)`: remembered for `pac.estimate.*`, the statement itself is left to the
    /// generic dseries assignment handling.
    pub(super) fn pac_note_dseries(&mut self) {
        let t = |k| self.c.peek_at(k);
        if let (Some(x), true, true) =
            (t(0).and_then(|x| x.ident()), t(1).is_some_and(|x| x.is("=")), t(2).is_some_and(|x| x.is_ident("dseries")))
        {
            let mut v = Vec::new();
            let mut k = 2;
            while let Some(tk) = t(k) {
                if tk.is(";") {
                    break;
                }
                v.push(tk.clone());
                k += 1;
            }
            let x = x.to_string();
            self.pac.vars.insert(x, v);
        }
    }

    /// `X = simul_backward_model(init, n[, innovations]);` or `X = Y;` with `Y` such a dseries.
    pub(super) fn bw_assign_ahead(&self) -> bool {
        let t = |k| self.c.peek_at(k);
        t(0).and_then(|x| x.ident()).is_some()
            && t(1).is_some_and(|x| x.is("="))
            && (t(2).is_some_and(|x| x.is_ident(vm::SIMUL_BACKWARD))
                || (t(2).and_then(|x| x.ident()).is_some_and(|n| self.pac.runtime.contains(n))
                    && t(3).is_some_and(|x| x.is(";"))))
    }

    /// Commands of [`Parser::bw_assign_ahead`] (run by [`crate::varmodels::run`]).
    pub(super) fn bw_assign(&mut self) -> Result<()> {
        let (out, loc) = self.c.ident()?;
        let (out, loc) = (out.to_string(), loc.clone());
        self.c.expect("=")?;
        let toks = self.c.statement()?;
        self.pac.vars.remove(&out);
        self.pac.nan_fields.remove(&out);
        if toks.len() == 1 {
            let src = toks[0].ident().unwrap().to_string();
            self.pac.runtime.insert(out.clone());
            self.commands.push(Command::Other(
                vm::DS_COPY.into(),
                vec![("out".into(), OptValue::Str(out)), ("src".into(), OptValue::Str(src))],
            ));
            return Ok(());
        }
        if !(toks.len() >= 3 && toks[1].is("(") && toks[toks.len() - 1].is(")")) {
            bail!("{loc}: {}: expected (initialconditions, samplesize[, innovations])", vm::SIMUL_BACKWARD);
        }
        let args = split_commas(&toks[2..toks.len() - 1]);
        if !(2..=3).contains(&args.len()) {
            bail!("{loc}: {}: expected 2 or 3 arguments", vm::SIMUL_BACKWARD);
        }
        let mut o: Vec<(String, OptValue)> = vec![("out".into(), OptValue::Str(out.clone()))];
        // a dseries argument: a runtime dseries, or a workspace value (stored under a name)
        let series = |p: &mut Self, a: &[Token], key: &str, o: &mut Vec<(String, OptValue)>| -> Result<()> {
            if matches!(a, [l, r] if l.is("[") && r.is("]")) {
                return Ok(()); // []
            }
            if let [t] = a
                && let Some(n) = t.ident()
                && (p.pac.runtime.contains(n) || p.q1.workspace.vars.contains_key(n))
            {
                o.push((key.into(), OptValue::Str(n.to_string())));
                return Ok(());
            }
            let v = p.ws_value(a).map_err(|e| anyhow!("{loc}: {}: {e}", vm::SIMUL_BACKWARD))?;
            let name = format!("__{}_{out}_{key}", vm::SIMUL_BACKWARD);
            p.q1.workspace.vars.insert(name.clone(), v);
            o.push((key.into(), OptValue::Str(name)));
            Ok(())
        };
        series(self, args[0], "init", &mut o)?;
        let n =
            self.eval_tokens(args[1]).map_err(|e| anyhow!("{loc}: {}: the sample size: {e}", vm::SIMUL_BACKWARD))?;
        if n < 1.0 || n.fract() != 0.0 {
            bail!("{loc}: {}: the sample size must be a positive integer", vm::SIMUL_BACKWARD);
        }
        o.push(("periods".into(), OptValue::Num(n)));
        if let Some(a) = args.get(2) {
            series(self, a, "innovations", &mut o)?;
        }
        self.pac.runtime.insert(out);
        self.commands.push(Command::Other(vm::SIMUL_BACKWARD.into(), o));
        Ok(())
    }

    /// `X.f = dseries(V, DATE)` on a runtime dseries (`V` a number or NaN; `toks` after `dseries(`).
    fn runtime_field_assign(&mut self, var: &str, field: &str, toks: &[Token]) -> Result<()> {
        let loc = toks.first().map(|t| t.loc.to_string()).unwrap_or_default();
        let inner = &toks[..toks.len().saturating_sub(1)];
        let args = split_commas(inner);
        let v = match args.first() {
            Some([t]) if t.ident().is_some_and(|n| n.eq_ignore_ascii_case("nan")) => f64::NAN,
            // NaN(n, 1): a column of NaN
            Some([t, p, ..]) if t.ident().is_some_and(|n| n.eq_ignore_ascii_case("nan")) && p.is("(") => f64::NAN,
            Some(a) => self.eval_tokens(a)?,
            None => bail!("{loc}: {var}.{field} = dseries(...): expected a value"),
        };
        let mut o = vec![
            ("var".into(), OptValue::Str(var.into())),
            ("field".into(), OptValue::Str(field.into())),
            ("value".into(), OptValue::Num(v)),
        ];
        if let Some(d) = args.get(1) {
            let (d, _) = super::shk::date_at(d, 0)
                .filter(|(_, n)| *n == d.len())
                .ok_or_else(|| anyhow!("{loc}: {var}.{field} = dseries(...): the second argument must be a date"))?;
            o.push(("date".into(), OptValue::Str(d.label())));
        }
        self.commands.push(Command::Other(vm::DS_SET.into(), o));
        Ok(())
    }

    pub(super) fn pac_assign(&mut self) -> Result<()> {
        let first = self.c.peek_at(0).and_then(|x| x.ident()).unwrap_or("").to_string();
        let line = self.c.peek_at(0).map(|x| (x.loc.file.clone(), x.loc.line));
        self.c.i += 1;
        if first == "clear" {
            // a MATLAB command: it ends with the line when there is no `;`
            while let Some(t) = self.c.peek_at(0).filter(|t| Some((t.loc.file.clone(), t.loc.line)) == line) {
                self.c.i += 1;
                if t.is(";") {
                    break;
                }
                if let Some(n) = t.ident() {
                    self.pac.structs.remove(n);
                    self.pac.vars.remove(n);
                    self.pac.nan_fields.remove(n);
                }
            }
            return Ok(());
        }
        let toks = self.c.statement()?;
        if toks.first().is_some_and(|t| t.is(".")) {
            let field = toks[1].ident().unwrap_or("").to_string();
            let rest = &toks[3..];
            if self.pac.runtime.contains(&first)
                && rest.first().is_some_and(|t| t.is_ident("dseries"))
                && rest.get(1).is_some_and(|t| t.is("("))
            {
                // the runtime dseries itself (`X.f = dseries(V, DATE)`)
                self.runtime_field_assign(&first, &field, &rest[2..])?;
            }
            let (neg, rest) = match rest.first() {
                Some(t) if t.is("-") => (true, &rest[1..]),
                Some(t) if t.is("+") => (false, &rest[1..]),
                _ => (false, rest),
            };
            let Some(Tok::Num(v)) = rest.first().map(|t| &t.tok) else {
                // edata.ez = dseries(NaN, ...): the residual of the equation, not observed
                if rest.first().is_some_and(|t| t.is_ident("dseries"))
                    && rest.get(1).is_some_and(|t| t.is("("))
                    && rest.get(2).is_some_and(|t| t.ident().is_some_and(|n| n.eq_ignore_ascii_case("nan")))
                {
                    let e = self.pac.nan_fields.entry(first).or_default();
                    if !e.contains(&field) {
                        e.push(field);
                    }
                }
                return Ok(()); // not a parameter value
            };
            if rest.len() != 1 {
                bail!("{}: only a numerical constant can be assigned to {first}.{field}", toks[0].loc);
            }
            let v = if neg { -v } else { *v };
            let e = self.pac.structs.entry(first).or_default();
            match e.iter_mut().find(|(n, _)| *n == field) {
                Some(x) => x.1 = v,
                None => e.push((field, v)),
            }
        } else {
            self.pac.vars.insert(first, toks[1..].to_vec());
        }
        Ok(())
    }

    /// `pac.estimate.nls(eq, params, data, range[, optimizer, ...])` /
    /// `pac.estimate.iterative_ols(eq, params, data, range)`, as options of the command.
    fn pac_estimate_command(&mut self, name: String, toks: &[Token]) -> Result<()> {
        let loc = self.c.loc();
        if !(toks.len() >= 2 && toks[0].is("(") && toks[toks.len() - 1].is(")")) {
            bail!("{loc}: {name}: expected (equation, params, data, range)");
        }
        let args = split_commas(&toks[1..toks.len() - 1]);
        if args.len() < 4 {
            bail!("{loc}: {name}: expected at least 4 arguments (equation, params, data, range)");
        }
        let resolve = |a: &[Token]| -> Vec<Token> {
            match a {
                [t] if t.ident().is_some_and(|n| self.pac.vars.contains_key(n)) => {
                    self.pac.vars[t.ident().unwrap()].clone()
                }
                _ => a.to_vec(),
            }
        };
        let strs = |a: &[Token]| -> Vec<(String, usize)> {
            a.iter()
                .enumerate()
                .filter_map(|(k, t)| if let Tok::Str(s) = &t.tok { Some((s.clone(), k)) } else { None })
                .collect()
        };
        let mut o: Vec<(String, OptValue)> = Vec::new();
        let eq = strs(args[0])
            .first()
            .map(|x| x.0.clone())
            .ok_or_else(|| anyhow!("{loc}: {name}: the equation name must be a string"))?;
        o.push(("eq".into(), OptValue::Str(eq)));
        let sname = args[1]
            .first()
            .and_then(|t| t.ident())
            .ok_or_else(|| anyhow!("{loc}: {name}: the second argument must be a structure"))?;
        let st = self.pac.structs.get(sname).ok_or_else(|| {
            anyhow!("{loc}: {name}: structure {sname} has no field (assign `{sname}.param = value;` first)")
        })?;
        for (n, v) in st {
            o.push((format!("param:{n}"), OptValue::Num(*v)));
        }
        if let Some(n) = args[2].first().and_then(|t| t.ident())
            && args[2].len() == 1
        {
            for f in self.pac.nan_fields.get(n).into_iter().flatten() {
                o.push(("nan".into(), OptValue::Str(f.clone())));
            }
        }
        let data = resolve(args[2]);
        match strs(&data).first() {
            _ if matches!(args[2], [t] if t.ident().is_some_and(|n| self.pac.runtime.contains(n))) => {
                o.push(("data_var".into(), OptValue::Str(args[2][0].ident().unwrap().to_string())))
            }
            Some((f, _)) => o.push(("file".into(), OptValue::Str(f.clone()))),
            None if data.first().is_some_and(|t| t.is_ident("data")) => {}
            None => bail!(
                "{loc}: {name}: the data must be `dseries('file')`, a variable assigned so, or a file name string"
            ),
        }
        let rg = resolve(args[3]);
        let ((first, first_off), (last, last_off)) =
            parse_date_range(&rg).map_err(|e| anyhow!("{loc}: {name}: {e}"))?;
        o.push(("first".into(), OptValue::Str(first)));
        o.push(("first_offset".into(), OptValue::Num(first_off as f64)));
        o.push(("last".into(), OptValue::Str(last)));
        o.push(("last_offset".into(), OptValue::Num(last_off as f64)));
        if let Some(a) = args.get(4) {
            let Some((s, _)) = strs(a).into_iter().next() else {
                bail!("{loc}: {name}: the optimizer must be a string")
            };
            o.push(("optimizer".into(), OptValue::Str(s)));
            // the optional arguments of the optimizer (e.g. 'Algorithm', 'levenberg-marquardt')
            for a in &args[5..] {
                if let Some((s, _)) = strs(a).first() {
                    o.push(("optimizer_arg".into(), OptValue::Str(s.clone())));
                }
            }
        }
        self.commands.push(Command::Other(name, o));
        Ok(())
    }

    pub(super) fn pac_pending(&self) -> bool {
        !self.pac.decls.is_empty()
    }

    pub(super) fn pac_collect(&mut self, tag: Option<String>, lhs: &Ast, rhs: &Ast, nums: Vec<String>) {
        if self.pac_pending() {
            self.pac.eqs.push((tag, lhs.clone(), rhs.clone()));
            self.pac.eq_nums.push(nums);
        }
    }

    pub(super) fn pac_statement(&mut self, kw: &str) -> Result<()> {
        let loc = self.c.loc();
        self.c.i += 1;
        match kw {
            "pac" => {
                let mut name = String::from("pac");
                while self.c.eat(".") {
                    let (id, _) = self.c.ident()?;
                    name.push('.');
                    name.push_str(id);
                }
                let name = if name == "pac.update.equation" { "pac.update.expectation".to_string() } else { name };
                let toks = self.c.statement()?;
                if vm::PAC_ESTIMATE.contains(&name.as_str()) {
                    return self.pac_estimate_command(name, toks);
                }
                if name == vm::PAC_PRINT {
                    // pac.print(pacname, eqname[, withcalibration])
                    let args = if toks.len() >= 2 && toks[0].is("(") && toks[toks.len() - 1].is(")") {
                        split_commas(&toks[1..toks.len() - 1])
                    } else {
                        Vec::new()
                    };
                    let word = |a: &[Token]| match a {
                        [t] => match &t.tok {
                            Tok::Str(s) => Some(s.clone()),
                            _ => None,
                        },
                        _ => None,
                    };
                    let (Some(pac), Some(eq)) = (args.first().and_then(|a| word(a)), args.get(1).and_then(|a| word(a)))
                    else {
                        bail!("{loc}: pac.print: expected ('PAC model name', 'equation name'[, withcalibration])")
                    };
                    let mut o = vec![(pac, OptValue::Flag), ("eq".into(), OptValue::Str(eq))];
                    if let Some(a) = args.get(2) {
                        let v = match a {
                            [t] if t.is_ident("true") => 1.0,
                            [t] if t.is_ident("false") => 0.0,
                            _ => self.eval_tokens(a)?,
                        };
                        o.push(("withcalibration".into(), OptValue::Num(v)));
                    }
                    self.commands.push(Command::Other(name, o));
                    return Ok(());
                }
                if name == vm::PAC_BGP_SET {
                    // pac.bgp.set(pacmodel, paceq, variable, nonzeromean)
                    let args = if toks.len() >= 2 && toks[0].is("(") && toks[toks.len() - 1].is(")") {
                        split_commas(&toks[1..toks.len() - 1])
                    } else {
                        Vec::new()
                    };
                    let word = |a: &[Token]| match a {
                        [t] => match &t.tok {
                            Tok::Str(s) => Some(s.clone()),
                            _ => None,
                        },
                        _ => None,
                    };
                    let (Some(pac), Some(eq), Some(var), Some(val), 4) = (
                        args.first().and_then(|a| word(a)),
                        args.get(1).and_then(|a| word(a)),
                        args.get(2).and_then(|a| word(a)),
                        args.get(3),
                        args.len(),
                    ) else {
                        bail!("{loc}: pac.bgp.set: expected ('PAC model name', 'equation name', 'variable', value)")
                    };
                    // MATLAB's true/false are logical, any other value is numeric
                    let v = match &val[..] {
                        [t] if t.is_ident("true") => ("logical".to_string(), OptValue::Num(1.0)),
                        [t] if t.is_ident("false") => ("logical".to_string(), OptValue::Num(0.0)),
                        a => ("value".to_string(), OptValue::Num(self.eval_tokens(a)?)),
                    };
                    let o = vec![
                        (pac, OptValue::Flag),
                        ("eq".into(), OptValue::Str(eq)),
                        ("var".into(), OptValue::Str(var)),
                        v,
                    ];
                    self.commands.push(Command::Other(name, o));
                    return Ok(());
                }
                if !matches!(name.as_str(), vm::PAC_INITIALIZE | vm::PAC_UPDATE | vm::PAC_MCE) {
                    bail!("{loc}: {name} is not supported");
                }
                let arg = toks
                    .iter()
                    .skip(1)
                    .find_map(|t| match &t.tok {
                        Tok::Ident(s) | Tok::Str(s) => Some(s.clone()),
                        _ => None,
                    })
                    .ok_or_else(|| anyhow!("{loc}: {name}: expected the name of a PAC model"))?;
                self.commands.push(Command::Other(name, vec![(arg, OptValue::Flag)]));
                Ok(())
            }
            "pac_model" => {
                let toks = self.c.statement()?;
                if !(toks.len() >= 2 && toks[0].is("(") && toks[toks.len() - 1].is(")")) {
                    bail!("{loc}: pac_model: expected (OPTIONS)");
                }
                let mut opts: Vec<(String, &[Token])> = Vec::new();
                for f in split_commas(&toks[1..toks.len() - 1]) {
                    let Some(k) = f.first().and_then(|t| t.ident()) else { bail!("{loc}: pac_model: bad option") };
                    let v = if f.len() >= 2 && f[1].is("=") { &f[2..] } else { &f[1..] };
                    opts.push((k.to_ascii_lowercase(), v));
                }
                let get = |k: &str| opts.iter().find(|(n, _)| n == k).map(|(_, v)| *v);
                let word = |t: &[Token]| -> Result<String> {
                    match t {
                        [x] => match &x.tok {
                            Tok::Ident(s) | Tok::Str(s) => Ok(s.clone()),
                            _ => bail!("{loc}: pac_model: expected a name"),
                        },
                        _ => bail!("{loc}: pac_model: expected a name"),
                    }
                };
                let name = word(get("model_name").ok_or_else(|| anyhow!("{loc}: pac_model: model_name is required"))?)?;
                let aux = match get("auxiliary_model_name").or_else(|| get("auxiliary_model")) {
                    Some(t) => Some(word(t)?),
                    None => None,
                };
                let discount = match get("discount") {
                    None => bail!("{loc}: pac_model {name}: discount is required"),
                    Some([t]) if matches!(&t.tok, Tok::Ident(_)) => Discount::Param(t.raw.clone()),
                    Some(t) => Discount::Value(self.eval_tokens(t)?),
                };
                let growth = match get("growth") {
                    Some(t) => {
                        let (a, nums) = slice_expr_nums(t)?;
                        let pre = self.preamble(a.clone(), nums);
                        Some((a, pre))
                    }
                    None => None,
                };
                let kind = match get("kind") {
                    Some(t) => word(t)?.to_ascii_lowercase(),
                    None => String::new(),
                };
                if !matches!(kind.as_str(), "" | "dd" | "dl" | "ll") {
                    bail!("{loc}: pac_model {name}: kind must be dd or dl");
                }
                let auxname = match get("auxname") {
                    Some(t) => Some(word(t)?),
                    None => None,
                };
                if self.pac.decls.iter().any(|d| d.name == name) {
                    bail!("{loc}: PAC model {name} declared twice");
                }
                self.pac.decls.push(PacDecl { name, aux, discount, growth, kind, auxname, loc });
                Ok(())
            }
            _ => {
                // pac_target_info(NAME); target T; auxname_target_nonstationary N; component C; ... end;
                let toks = self.c.statement()?;
                let name = toks
                    .iter()
                    .skip(1)
                    .find_map(|t| match &t.tok {
                        Tok::Ident(s) | Tok::Str(s) => Some(s.clone()),
                        _ => None,
                    })
                    .ok_or_else(|| anyhow!("{loc}: pac_target_info: expected the name of a PAC model"))?;
                let mut info = InfoDecl::default();
                while !self.c.at_end() {
                    let st = self.c.statement()?;
                    let Some(k) = st.first().and_then(|t| t.ident()) else { continue };
                    let rest = &st[1..];
                    let word = |t: &[Token]| -> Result<String> {
                        match t {
                            [x] => match &x.tok {
                                Tok::Ident(s) | Tok::Str(s) => Ok(s.clone()),
                                _ => bail!("{loc}: pac_target_info: expected a name"),
                            },
                            _ => bail!("{loc}: pac_target_info: expected a name"),
                        }
                    };
                    // the expressions are parsed into the model's tree at their place
                    let expr = |p: &mut Self| -> Result<(Ast, usize)> {
                        let (a, nums) = slice_expr_nums(rest)?;
                        let pre = p.preamble(a.clone(), nums);
                        Ok((a, pre))
                    };
                    match k.to_ascii_lowercase().as_str() {
                        "target" => info.target = Some(expr(self)?),
                        "auxname_target_nonstationary" => info.auxname_ns = Some(word(rest)?),
                        "component" => {
                            let (a, pre) = expr(self)?;
                            info.comps.push(CompDecl { expr: a, pre, kind: String::new(), auxname: None, growth: None })
                        }
                        "kind" | "auxname" | "growth" => {
                            if info.comps.is_empty() {
                                bail!("{loc}: pac_target_info: '{k}' before the first component");
                            }
                            let g = if k.eq_ignore_ascii_case("growth") { Some(expr(self)?) } else { None };
                            let c = info.comps.last_mut().unwrap();
                            match k.to_ascii_lowercase().as_str() {
                                "kind" => c.kind = word(rest)?.to_ascii_lowercase(),
                                "auxname" => c.auxname = Some(word(rest)?),
                                _ => c.growth = g,
                            }
                        }
                        other => bail!("{loc}: pac_target_info: unknown statement '{other}'"),
                    }
                }
                self.c.i += 2; // end ;
                if info.target.is_none() {
                    bail!("{loc}: the block 'pac_target_info({name})' is missing the 'target' statement");
                }
                self.pac.infos.insert(name, info);
                Ok(())
            }
        }
    }

    /// The parameters the preprocessor creates in its transform pass (their symbol ids come
    /// after the auxiliary variables of the diff operators).
    fn pac_declare_params(&mut self, names: &[String], loc: &str) -> Result<()> {
        for n in names {
            self.b.declare_late_param(n).map_err(|e| anyhow!("{loc}: {e}"))?;
            self.labels.push(SymbolKind::Param, tex(n), n.clone());
        }
        Ok(())
    }

    /// Resolves the declared PAC models whose equation is in the model block that follows.
    pub(super) fn pac_resolve(&mut self) -> Result<()> {
        let decls = std::mem::take(&mut self.pac.decls);
        let mut keep = Vec::new();
        for d in decls {
            let key = format!("{PE}{}", d.name);
            let found = self.pac.eqs.iter().find(|(_, l, r)| {
                let p = |n: &str| n == key;
                contains_var(l, &p) || contains_var(r, &p)
            });
            let Some((tag, lhs, rhs)) = found.cloned() else {
                keep.push(d);
                continue;
            };
            self.pac_resolve_one(d, tag, lhs, rhs)?;
        }
        self.pac.decls = keep;
        Ok(())
    }

    fn pac_resolve_one(&mut self, d: PacDecl, tag: Option<String>, lhs: Ast, rhs: Ast) -> Result<()> {
        let loc = d.loc.clone();
        let l0 = no_loc();
        let name = d.name.clone();
        let sk = |n: &str| self.b.symbol(n).map(|s| s.kind);
        // ---- decomposition of the PAC equation
        // diff(x), or diff(f(x)) of a unary function: the auxiliary variable that stands for it
        // (`substituteUnaryOps` runs on the PAC equation before its analysis)
        let (lhs_var, lhs_expr) = match &lhs {
            Ast::Op(SpecialOp::Diff, a) => match a.first() {
                Some(Ast::Var { name, lag: 0, .. }) => (name.clone(), None),
                Some(x @ Ast::Un(..)) => match vm::key_of(x, &sk) {
                    Ok(Some((e, key, 0))) => (key, Some(e)),
                    _ => bail!("{loc}: PAC equation of {name}: the left hand side must be diff(variable)"),
                },
                _ => bail!("{loc}: PAC equation of {name}: the left hand side must be diff(variable)"),
            },
            _ => bail!("{loc}: PAC equation of {name}: the left hand side must be diff(variable)"),
        };
        // rule of thumb agents: the error correction and autoregressive terms are in the
        // optimizing part
        let split = split_share(&rhs, &sk);
        let mut atoms = Vec::new();
        lin(split.as_ref().map_or(&rhs, |s| &s.optim), &num(1.0), &sk, &mut atoms);
        let mut ec: Option<(String, String, Option<Ast>)> = None; // (param, target, its expression)
        {
            let zp = atoms.iter().find_map(|(c, a)| match a {
                Atom::Level(n, -1) | Atom::Expr(n, -1, _) if *n == lhs_var => param_sign(c, &sk),
                _ => None,
            });
            let Some((Some(p), s)) = zp else {
                bail!(
                    "{loc}: PAC equation of {name}: cannot find the error correction term (param*(target(-1)-{lhs_var}(-1)))"
                )
            };
            if s != -1.0 {
                bail!(
                    "{loc}: PAC equation of {name}: the error correction term must be param*(target(-1)-{lhs_var}(-1))"
                );
            }
            for (c, a) in &atoms {
                let (t, e) = match a {
                    Atom::Level(n, -1) if *n != lhs_var => (n.clone(), None),
                    // a unary function of a variable: the auxiliary variable that stands for it
                    Atom::Expr(k, -1, e) if *k != lhs_var => (k.clone(), Some(e.clone())),
                    Atom::Pns(m) if *m == name => (format!("{PNS}{m}"), None),
                    _ => continue,
                };
                if let Some((Some(q), 1.0)) = param_sign(c, &sk)
                    && q == p
                {
                    ec = Some((p.clone(), t, e));
                    break;
                }
            }
        }
        let Some((ec_param, mut target, target_expr)) = ec else {
            bail!("{loc}: PAC equation of {name}: cannot find the target in the error correction term")
        };
        if target_expr.is_some() && d.aux.is_none() {
            bail!(
                "{loc}: PAC model {name}: a target that is a function of a variable is not supported with model consistent expectations"
            );
        }
        let mut ar: Vec<Option<String>> = Vec::new();
        for (c, a) in &atoms {
            if let Atom::Diff(n, lag) = a
                && *n == lhs_var
                && *lag < 0
                && let Some((Some(p), 1.0)) = param_sign(c, &sk)
            {
                let k = (-*lag) as usize;
                if ar.len() < k {
                    ar.resize(k, None);
                }
                ar[k - 1] = Some(p);
            }
        }
        // no autoregressive term is accepted, as by the preprocessor (a2alpha then fails at
        // evaluation, a2alpha.m reads past the end of its argument)
        let ar_params: Vec<String> = ar
            .into_iter()
            .enumerate()
            .map(|(k, p)| {
                p.ok_or_else(|| {
                    anyhow!("{loc}: PAC equation of {name}: the lag {} of diff({lhs_var}) is missing", k + 1)
                })
            })
            .collect::<Result<_>>()?;
        let bgp = {
            let mut skip: Vec<&String> = vec![&ec_param];
            skip.extend(&ar_params);
            bgp_terms(&rhs, split.as_ref(), &skip, &sk)
        };
        // ---- auxiliary model: its companion variables (keys and expressions at t)
        struct Aux {
            kind: &'static str,
            vars: Vec<(String, Ast)>,
            lags: usize,
        }
        let aux = match &d.aux {
            None => None,
            Some(a) => {
                let vms = &self.vm.models;
                match vms.auxiliary_kind(a) {
                    Some("var") => {
                        let m = vms.var(a).unwrap();
                        Some(Aux { kind: "var", vars: m.companion_exprs(), lags: m.max_lag() })
                    }
                    Some(_) => {
                        let m = vms.trend(a).unwrap();
                        Some(Aux { kind: "trend_component", vars: m.companion_exprs(), lags: m.max_lag() + 1 })
                    }
                    None => bail!("{loc}: PAC model {name}: unknown auxiliary model '{a}'"),
                }
            }
        };
        let aux_keys: Vec<usize> = match &aux {
            Some(a) => a.vars.iter().map(|(k, e)| self.vm_key(k, e)).collect(),
            None => Vec::new(),
        };
        if let Some(e) = &target_expr {
            self.vm_key(&target, e);
        }
        if let Some(e) = &lhs_expr {
            self.vm_key(&lhs_var, e);
        }
        let info = self.pac.infos.remove(&name);
        let uses_pns = contains_var(&rhs, &|n| n == format!("{PNS}{name}"));
        if uses_pns && info.is_none() {
            bail!("{loc}: PAC model {name}: pac_target_nonstationary requires a pac_target_info block");
        }
        // what the h parameters multiply: the constant (None) or (expression at t, key, lag)
        type Lay = Vec<Option<(Ast, String, usize)>>;
        let layout_of = |aux: &Aux| -> Lay {
            let mut v = Vec::new();
            if aux.kind == "var" {
                v.push(None);
            }
            for k in 1..=aux.lags {
                for (key, e) in &aux.vars {
                    v.push(Some((e.clone(), key.clone(), k)));
                }
            }
            v
        };
        // named after the keys, renamed after the auxiliary variables once the model is built
        let hnames = |prefix: &str, lay: &Lay| -> Vec<String> {
            lay.iter()
                .map(|e| match e {
                    None => format!("{prefix}_constant"),
                    Some((_, key, k)) => format!("{prefix}_var_{key}_lag_{k}"),
                })
                .collect()
        };
        let expectation = |hp: &[String], lay: &Lay, growth: Option<(&Ast, &str)>| -> Ast {
            let mut acc: Option<Ast> = None;
            if let Some((g, gp)) = growth {
                acc = add(acc, bin(BinaryOp::Mul, g.clone(), var(gp, 0, &l0)));
            }
            for (p, e) in hp.iter().zip(lay) {
                let pv = var(p, 0, &l0);
                let term = match e {
                    None => pv,
                    Some((x, _, k)) => bin(BinaryOp::Mul, pv, vm::lagged(x, *k)),
                };
                acc = add(acc, term);
            }
            acc.unwrap_or(num(0.0))
        };
        // `rewrite_aux_variable`: the variable and its transformations, innermost first
        fn rewrite(a: &Ast) -> (String, Vec<String>) {
            match a {
                Ast::Op(SpecialOp::Diff, v) => {
                    let (n, mut t) = rewrite(&v[0]);
                    t.push("diff".into());
                    (n, t)
                }
                Ast::Un(op, x) => {
                    let (n, mut t) = rewrite(x);
                    t.push(op.name().into());
                    (n, t)
                }
                Ast::Var { name, .. } => (name.clone(), Vec::new()),
                _ => (String::new(), Vec::new()),
            }
        }
        let lay = aux.as_ref().map(layout_of).unwrap_or_default();
        let mut model = PacModel {
            name: name.clone(),
            eq_name: tag.clone().unwrap_or_default(),
            aux_model: d.aux.clone(),
            aux_type: aux.as_ref().map(|a| a.kind.to_string()).unwrap_or_default(),
            discount: d.discount.clone(),
            growth: d.growth.as_ref().map(|g| g.0.clone()),
            kind: d.kind.clone(),
            lhs_var: lhs_var.clone(),
            lhs_expr,
            ec_param,
            ar_params,
            share: split.as_ref().map(|s| s.share.clone()),
            non_optimizing: split.as_ref().map(|s| s.non_optim.clone()),
            target: target.clone(),
            target_expr: target_expr.clone(),
            target_var: String::new(),
            components: Vec::new(),
            aux_var: String::new(),
            h_params: Vec::new(),
            h_layout: lay
                .iter()
                .map(|e| {
                    e.as_ref().map(|(x, _, k)| {
                        let (n, t) = rewrite(x);
                        (n, t, *k)
                    })
                })
                .collect(),
            h_keys: lay.iter().map(|e| e.as_ref().map(|(_, key, _)| key.clone())).collect(),
            growth_param: None,
            mce_alpha: Vec::new(),
            mce_z: String::new(),
            vns: String::new(),
            eq_lhs: num(0.0),
            eq_rhs: num(0.0),
            defs: Vec::new(),
            bgp,
            initialized: false,
        };
        let gname = format!("{name}_pac_growth_neutrality_correction");
        let spec_growth = d.growth.as_ref().map(|g| (g.1, gname.clone()));
        let mut pe_subst: Ast;
        let kind = match (&aux, &info) {
            // ---- model consistent expectations
            (None, _) => {
                if info.is_some() {
                    bail!(
                        "{loc}: PAC model {name}: a composite target is not supported with model consistent expectations"
                    );
                }
                let m = model.ar_params.len() + 1;
                let alpha: Vec<String> = (1..=m).map(|i| format!("mce_alpha_{name}_{i}")).collect();
                let zname = d.auxname.clone().unwrap_or_else(|| format!("mce_Z1_{name}"));
                let mut newp = alpha.clone();
                if d.growth.is_some() {
                    newp.insert(0, gname.clone());
                }
                self.pac_declare_params(&newp, &loc)?;
                let beta = match &d.discount {
                    Discount::Param(p) => var(p, 0, &l0),
                    Discount::Value(v) => num(*v),
                };
                let powb = |i: usize| bin(BinaryOp::Pow, beta.clone(), num(i as f64));
                let al = |i: usize| var(&alpha[i - 1], 0, &l0);
                let dx = |j: i32| Ast::Op(SpecialOp::Diff, vec![var(&target, j, &l0)]);
                let mut sum_a = num(1.0);
                for i in 1..=m {
                    sum_a = bin(BinaryOp::Add, sum_a, al(i));
                }
                // dx_t - sum_j c_j dx_{t+j}, c_j = sum_{i>j} alpha_i beta^i
                let mut inner = dx(0);
                for j in 1..m {
                    let mut c: Option<Ast> = None;
                    for i in j + 1..=m {
                        c = add(c, bin(BinaryOp::Mul, al(i), powb(i)));
                    }
                    inner = bin(BinaryOp::Sub, inner, bin(BinaryOp::Mul, c.unwrap(), dx(j as i32)));
                }
                let mut tail: Option<Ast> = None;
                for j in 1..=m {
                    tail =
                        add(tail, bin(BinaryOp::Mul, bin(BinaryOp::Mul, al(j), powb(j)), var(&zname, j as i32, &l0)));
                }
                let z_rhs = bin(BinaryOp::Sub, bin(BinaryOp::Mul, sum_a, inner), tail.unwrap());
                model.defs.push((zname.clone(), z_rhs));
                model.mce_alpha = alpha.clone();
                model.mce_z = zname.clone();
                pe_subst = var(&zname, 0, &l0);
                if let Some((g, _)) = &d.growth {
                    model.growth_param = Some(gname.clone());
                    pe_subst = bin(BinaryOp::Add, bin(BinaryOp::Mul, g.clone(), var(&gname, 0, &l0)), pe_subst);
                }
                let discount = match &d.discount {
                    Discount::Param(p) => p.clone(),
                    Discount::Value(v) => format!("{v}"),
                };
                PacKind::Mce { target: target.clone(), max_lag: m - 1, alpha, discount, auxname: zname }
            }
            // ---- simple target
            (Some(a), None) => {
                let hp = hnames(&format!("h_{name}"), &lay);
                let auxv = d.auxname.clone().unwrap_or_else(|| format!("pac_expectation_{name}"));
                let mut newp = hp.clone();
                if d.growth.is_some() {
                    newp.insert(0, gname.clone());
                }
                self.pac_declare_params(&newp, &loc)?;
                let def = expectation(&hp, &lay, d.growth.as_ref().map(|g| (&g.0, gname.as_str())));
                model.defs.push((auxv.clone(), def));
                model.h_params = hp.clone();
                model.aux_var = auxv.clone();
                if d.growth.is_some() {
                    model.growth_param = Some(gname.clone());
                }
                pe_subst = var(&auxv, 0, &l0);
                PacKind::Backward {
                    is_var: a.kind == "var",
                    lhs: aux_keys.clone(),
                    max_lag: a.lags,
                    h: hp,
                    auxname: auxv,
                }
            }
            // ---- composite target
            (Some(a), Some(info)) => {
                if a.kind != "var" {
                    bail!("{loc}: PAC model {name}: a composite target is not allowed with a trend component model");
                }
                if d.growth.is_some() {
                    bail!(
                        "{loc}: for PAC model '{name}', it is not possible to declare a 'growth' option in the 'pac_model' command when there is also a 'pac_target_info' block"
                    );
                }
                if info.comps.is_empty() {
                    bail!("{loc}: pac_target_info({name}): no component");
                }
                let vns = info.auxname_ns.clone().ok_or_else(|| {
                    anyhow!("{loc}: the block 'pac_target_info({name})' is missing the 'auxname_target_nonstationary' statement")
                })?;
                let sk2 = |n: &str| self.b.symbol(n).map(|s| s.kind);
                let (target_ast, target_pre) = info.target.clone().unwrap();
                let tkey = vm::key_of(&target_ast, &sk2).ok().flatten().map(|x| x.1).unwrap_or_default();
                // the equation of the target
                let tpos = self
                    .pac
                    .eqs
                    .iter()
                    .position(|(_, l, _)| vm::key_of(l, &sk2).ok().flatten().is_some_and(|x| x.1 == tkey && x.2 == 0))
                    .ok_or_else(|| {
                        anyhow!(
                            "{loc}: there is no equation whose LHS is equal to the 'target' of 'pac_target_info({name})'"
                        )
                    })?;
                let teq = self.pac.eqs[tpos].clone();
                // the numbers as written (`coeff_str` prints `.5`)
                let mut num_texts = Vec::new();
                fn leaves(a: &Ast, out: &mut Vec<f64>) {
                    match a {
                        Ast::Num(v) => out.push(*v),
                        Ast::Var { .. } => {}
                        Ast::Un(_, x) => leaves(x, out),
                        Ast::Bin(_, x, y) => {
                            leaves(x, out);
                            leaves(y, out);
                        }
                        Ast::Op(_, v) => v.iter().for_each(|x| leaves(x, out)),
                    }
                }
                let mut vals = Vec::new();
                leaves(&teq.1, &mut vals);
                leaves(&teq.2, &mut vals);
                if let Some(texts) = self.pac.eq_nums.get(tpos).filter(|t| t.len() == vals.len()) {
                    num_texts = vals.into_iter().zip(texts.iter().cloned()).collect();
                }
                let spec = vm::eq_spec("target", &teq.1, &teq.2, &sk2, true).map_err(|_| {
                    anyhow!(
                        "{loc}: the model equation defining the 'target' of 'pac_target_info({name})' is not of the right form (should be a linear combination of endogenous variables)"
                    )
                })?;
                let ckeys: Vec<(String, Ast)> = info
                    .comps
                    .iter()
                    .map(|c| {
                        vm::key_of(&c.expr, &sk2)
                            .ok()
                            .flatten()
                            .filter(|x| x.2 == 0)
                            .map(|x| (x.1, x.0))
                            .ok_or_else(|| anyhow!("{loc}: pac_target_info({name}): a component must be a variable"))
                    })
                    .collect::<Result<_>>()?;
                if let Some(t) = spec.terms.iter().find(|t| t.lag != 0 || !ckeys.iter().any(|(k, _)| *k == t.var)) {
                    bail!(
                        "{loc}: the model equation defining the 'target' of 'pac_target_info({name})' contains a variable ({}) that is not declared as a 'component'",
                        t.var
                    );
                }
                let mut comps = Vec::new();
                let mut cspecs = Vec::new();
                let mut pe_sum: Option<Ast> = None;
                let mut ns_sum: Option<Ast> = None;
                let mut newp = Vec::new();
                for (i, c) in info.comps.iter().enumerate() {
                    let (ckey, cexpr) = &ckeys[i];
                    let mut coef: Option<Ast> = None;
                    for t in spec.terms.iter().filter(|t| t.var == *ckey) {
                        coef = add(coef, t.coef.clone());
                    }
                    let coef = coef.ok_or_else(|| {
                        anyhow!(
                            "{loc}: a 'component' of 'pac_target_info({name})' does not appear in the model equation defining the 'target'"
                        )
                    })?;
                    if c.auxname.is_none() {
                        bail!(
                            "{loc}: the block 'pac_target_info({name})' is missing the 'auxname' statement in some 'component'"
                        );
                    }
                    if !matches!(c.kind.as_str(), "ll" | "dd" | "dl") {
                        bail!(
                            "{loc}: the block 'pac_target_info({name})' is missing the 'kind' statement in some 'component'"
                        );
                    }
                    if c.kind == "ll" && c.growth.is_some() {
                        bail!(
                            "{loc}: in the block 'pac_target_info({name})', a component of 'kind ll' (i.e. stationary) has a 'growth' option. This is not permitted."
                        );
                    }
                    self.vm_key(ckey, cexpr);
                    let hp = hnames(&format!("h_{name}_component{}", i + 1), &lay);
                    let auxv = c.auxname.clone().unwrap();
                    let gname = format!("{name}_component{}_pac_growth_neutrality_correction", i + 1);
                    newp.extend(hp.iter().cloned());
                    if c.growth.is_some() {
                        newp.push(gname.clone());
                    }
                    let def = expectation(&hp, &lay, c.growth.as_ref().map(|g| (&g.0, gname.as_str())));
                    model.defs.push((auxv.clone(), def));
                    pe_sum = add(pe_sum, bin(BinaryOp::Mul, coef.clone(), var(&auxv, 0, &l0)));
                    if c.kind != "ll" {
                        ns_sum = add(ns_sum, bin(BinaryOp::Mul, coef.clone(), cexpr.clone()));
                    }
                    cspecs.push(CompSpec {
                        expr: c.pre,
                        ll: c.kind == "ll",
                        h: hp.clone(),
                        growth: c.growth.as_ref().map(|g| (g.1, gname.clone())),
                        auxname: auxv.clone(),
                    });
                    comps.push(PacComponent {
                        endo: ckey.clone(),
                        endo_key: ckey.clone(),
                        num_texts: num_texts.clone(),
                        kind: c.kind.clone(),
                        coef,
                        growth: c.growth.as_ref().map(|g| g.0.clone()),
                        auxname: auxv,
                        h_params: hp,
                        growth_param: c.growth.as_ref().map(|_| gname),
                    });
                }
                if info.comps.iter().all(|c| c.kind == "ll") {
                    bail!(
                        "{loc}: the block 'pac_target_info({name})' must contain at least one nonstationary component (i.e. of 'kind' equal to either 'dd' or 'dl')."
                    );
                }
                model.defs.push((vns.clone(), ns_sum.unwrap_or(num(0.0))));
                model.vns = vns.clone();
                if uses_pns {
                    self.pac.subst.insert(format!("{PNS}{name}"), var(&vns, -1, &l0));
                    target = vns.clone();
                }
                self.pac_declare_params(&newp, &loc)?;
                model.target = target.clone();
                model.target_var = tkey;
                model.components = comps;
                pe_subst = pe_sum.unwrap();
                PacKind::Composite {
                    is_var: true,
                    lhs: aux_keys.clone(),
                    max_lag: a.lags,
                    target: target_pre,
                    vns,
                    comps: cspecs,
                }
            }
        };
        self.b.pac.models.push(PacSpec { name: name.clone(), growth: spec_growth, kind });
        if model.vns.is_empty() {
            self.pac.subst.remove(&format!("{PNS}{name}"));
        }
        self.pac.subst.insert(format!("{PE}{name}"), std::mem::replace(&mut pe_subst, num(0.0)));
        // the equation, with the auxiliary variables
        model.eq_lhs = self.pac_expand(lhs)?;
        model.eq_rhs = self.pac_expand(rhs)?;
        self.pac.models.push(model);
        Ok(())
    }

    /// Replaces the operators by their auxiliary variables.
    pub(super) fn pac_expand(&self, a: Ast) -> Result<Ast> {
        Ok(match a {
            Ast::Var { name, lag, loc } if name.starts_with(PE) || name.starts_with(PNS) => {
                if lag != 0 {
                    bail!("{loc}: PAC operators cannot have a lead/lag");
                }
                self.pac.subst.get(&name).cloned().ok_or_else(|| {
                    anyhow!(
                        "{loc}: unknown PAC model '{}' (or the pac_model statement does not precede the model block)",
                        &name[name.find(':').map_or(0, |i| i + 1)..]
                    )
                })?
            }
            Ast::Un(o, x) => Ast::Un(o, Box::new(self.pac_expand(*x)?)),
            Ast::Bin(o, x, y) => Ast::Bin(o, Box::new(self.pac_expand(*x)?), Box::new(self.pac_expand(*y)?)),
            Ast::Op(o, v) => Ast::Op(o, v.into_iter().map(|x| self.pac_expand(x)).collect::<Result<_>>()?),
            other => other,
        })
    }

    pub(super) fn pac_finish(&mut self) -> Result<Vec<PacModel>> {
        if let Some(d) = self.pac.decls.first() {
            bail!(
                "{}: PAC model {}: no equation of the model block contains pac_expectation({})",
                d.loc,
                d.name,
                d.name
            );
        }
        Ok(std::mem::take(&mut self.pac.models))
    }
}

/// One date of a `pac.estimate.*` range with its integer offsets: `dates('2005Q1')`,
/// Dynare's literal `2005Q1` (lexed as the number 2005 and the identifier `Q1`; also `2005M3`,
/// `2005S1`, `2005W12`, `2005Y`), followed by any `+ n` / `- n`. Returns the date, the offset
/// and the next token index.
fn parse_range_date(t: &[Token], mut i: usize) -> Result<(String, i64, usize)> {
    let date = if t.get(i).is_some_and(|x| x.is_ident("dates")) {
        match (t.get(i + 1), t.get(i + 2).map(|x| &x.tok), t.get(i + 3)) {
            (Some(p), Some(Tok::Str(s)), Some(q)) if p.is("(") && q.is(")") => {
                i += 4;
                s.trim().to_string()
            }
            _ => bail!("expected dates('...') in the estimation range"),
        }
    } else if let (Some(Tok::Num(y)), Some(Tok::Ident(f))) = (t.get(i).map(|x| &x.tok), t.get(i + 1).map(|x| &x.tok)) {
        let ok = y.fract() == 0.0
            && f.chars().next().is_some_and(|c| "QqMmSsWwHhYy".contains(c))
            && f[1..].chars().all(|c| c.is_ascii_digit())
            && (f.len() > 1 || f.eq_ignore_ascii_case("y"));
        if !ok {
            bail!("{y}{f} is not a date");
        }
        i += 2;
        format!("{}{}", *y as i64, f.to_ascii_uppercase())
    } else {
        bail!("the estimation range must be dates('...'):dates('...') or a date literal range such as 2005Q1:2010Q4");
    };
    let mut off = 0i64;
    while let (Some(p), Some(Tok::Num(n))) = (t.get(i), t.get(i + 1).map(|x| &x.tok)) {
        if !(p.is("+") || p.is("-")) || n.fract() != 0.0 {
            break;
        }
        off += if p.is("-") { -(*n as i64) } else { *n as i64 };
        i += 2;
    }
    Ok((date, off, i))
}

/// `first:last` range of `pac.estimate.*`: ((first date, offset), (last date, offset)).
pub(super) fn parse_date_range(t: &[Token]) -> Result<((String, i64), (String, i64))> {
    let (f, fo, i) = parse_range_date(t, 0)?;
    if !t.get(i).is_some_and(|x| x.is(":")) {
        bail!("expected `:` in the estimation range");
    }
    let (l, lo, j) = parse_range_date(t, i + 1)?;
    if j != t.len() {
        bail!("unexpected `{}` in the estimation range", t[j].raw);
    }
    Ok(((f, fo), (l, lo)))
}
