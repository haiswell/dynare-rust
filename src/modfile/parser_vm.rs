// SPDX-License-Identifier: GPL-3.0-or-later
//! ---- var models ----: `var_model`, `trend_component_model`, `var_expectation_model`, the
//! `var_expectation(m)` operator and the `var_expectation.initialize/update` statements.
//!
//! The equations of the auxiliary models come after their declaration, and the number of
//! `var_expectation_model_<m>_*` parameters (which the operator expands into) depends on their
//! lags: at the start of each `model` block the equations are parsed once to find them
//! ([`Parser::vm_prepass`]), then read again for real with the operator expanded.

use super::{Ast, Cursor, Parser, slice_expr, split_commas};
use crate::model::SymbolKind;
use crate::model::expr::BinaryOp;
use crate::modfile::lexer::{Loc, Tok, Token};
use crate::modfile::{Command, OptValue};
use crate::varmodels::{self as vm, Discount, Horizon, TrendComponentModel, VarExpectationModel, VarModel, VarModels};
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;

struct VeDecl {
    name: String,
    aux: String,
    expr: Ast,
    horizon: Horizon,
    time_shift: i32,
    discount: Discount,
    loc: String,
}

/// Parser state of the VAR family.
#[derive(Default)]
pub(super) struct VmParse {
    var_decl: Vec<(String, Vec<String>, bool, String)>,
    tc_decl: Vec<(String, Vec<String>, Vec<String>, String)>,
    ve_decl: Vec<VeDecl>,
    /// Tagged equations seen in the prepasses.
    eqs: Vec<(String, Ast, Ast)>,
    /// Per expectation model: the (variable, lag) each parameter multiplies (`None`: the constant).
    layout: HashMap<String, Vec<Option<(String, usize)>>>,
    pub models: vm::VarModels,
}

fn str_list(toks: &[Token]) -> Vec<String> {
    toks.iter()
        .filter_map(|t| match &t.tok {
            Tok::Str(s) => Some(s.clone()),
            Tok::Ident(s) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

fn name_of(toks: &[Token], what: &str) -> Result<String> {
    match toks {
        [t] => match &t.tok {
            Tok::Ident(s) | Tok::Str(s) => Ok(s.clone()),
            _ => bail!("{}: {what}: expected a name", t.loc),
        },
        _ => bail!("{what}: expected a name"),
    }
}

impl<'a> Parser<'a> {
    pub(super) fn vm_handles(&self, kw: &str) -> bool {
        let next_is = |p: &str| self.c.peek_at(1).is_some_and(|t| t.is(p));
        matches!(kw, "var_model" | "trend_component_model" | "var_expectation_model")
            || (kw == "var_expectation" && next_is("."))
            || (kw == "get_companion_matrix" && next_is("("))
    }

    pub(super) fn vm_statement(&mut self, kw: &str) -> Result<()> {
        let loc = self.c.loc();
        self.c.i += 1;
        if kw == "var_expectation" {
            self.c.expect(".")?;
            let (f, _) = self.c.ident()?;
            let f = f.to_string();
            if f != "initialize" && f != "update" {
                bail!("{loc}: var_expectation.{f} is not supported (only initialize and update)");
            }
            let toks = self.c.statement()?;
            let arg = self.vm_call_arg(toks, &loc)?;
            self.commands.push(Command::Other(format!("var_expectation.{f}"), vec![(arg, OptValue::Flag)]));
            return Ok(());
        }
        let toks = self.c.statement()?;
        if kw == "get_companion_matrix" {
            let arg = self.vm_call_arg(toks, &loc)?;
            self.commands.push(Command::Other("get_companion_matrix".into(), vec![(arg, OptValue::Flag)]));
            return Ok(());
        }
        if !(toks.len() >= 2 && toks[0].is("(") && toks[toks.len() - 1].is(")")) {
            bail!("{loc}: {kw}: expected (OPTIONS)");
        }
        // option -> value tokens
        let mut opts: Vec<(String, &[Token])> = Vec::new();
        for f in split_commas(&toks[1..toks.len() - 1]) {
            let Some(k) = f.first().and_then(|t| t.ident()) else { bail!("{loc}: {kw}: bad option") };
            let v = if f.len() >= 2 && f[1].is("=") { &f[2..] } else { &f[1..] };
            opts.push((k.to_ascii_lowercase(), v));
        }
        let get = |k: &str| opts.iter().find(|(n, _)| n == k).map(|(_, v)| *v);
        let model_name = name_of(get("model_name").ok_or_else(|| anyhow!("{loc}: {kw}: model_name is required"))?, kw)?;
        let flat = |t: &[Token]| -> Vec<String> {
            str_list(&t.iter().filter(|x| !x.is("[") && !x.is("]") && !x.is(",")).cloned().collect::<Vec<_>>())
        };
        match kw {
            "var_model" => {
                let tags = flat(get("eqtags").ok_or_else(|| anyhow!("{loc}: var_model: eqtags is required"))?);
                let structural = get("structural").is_some();
                self.vm.var_decl.push((model_name, tags, structural, loc));
            }
            "trend_component_model" => {
                let tags =
                    flat(get("eqtags").ok_or_else(|| anyhow!("{loc}: trend_component_model: eqtags is required"))?);
                let targets =
                    flat(get("targets").ok_or_else(|| anyhow!("{loc}: trend_component_model: targets is required"))?);
                self.vm.tc_decl.push((model_name, tags, targets, loc));
            }
            _ => {
                let aux = get("auxiliary_model_name")
                    .or_else(|| get("auxiliary_model"))
                    .ok_or_else(|| anyhow!("{loc}: var_expectation_model: auxiliary_model_name is required"))?;
                let aux = name_of(aux, kw)?;
                let expr = slice_expr(
                    get("expression").ok_or_else(|| anyhow!("{loc}: var_expectation_model: expression is required"))?,
                )?;
                let h = get("horizon").ok_or_else(|| anyhow!("{loc}: var_expectation_model: horizon is required"))?;
                let horizon = vm_horizon(h, &loc)?;
                let time_shift = match get("time_shift") {
                    None => 0,
                    Some(t) => self.eval_tokens(t)? as i32,
                };
                let discount = match get("discount") {
                    None => bail!("{loc}: var_expectation_model: discount is required"),
                    Some([t]) if matches!(&t.tok, Tok::Ident(_)) => Discount::Param(t.raw.clone()),
                    Some(t) => Discount::Value(self.eval_tokens(t)?),
                };
                self.vm.ve_decl.push(VeDecl { name: model_name, aux, expr, horizon, time_shift, discount, loc });
            }
        }
        Ok(())
    }

    fn vm_call_arg(&self, toks: &[Token], loc: &str) -> Result<String> {
        if toks.len() >= 3 && toks[0].is("(") && toks[toks.len() - 1].is(")") {
            if let Some(t) = toks[1..toks.len() - 1].iter().find(|t| !t.is("'")) {
                return name_of(std::slice::from_ref(t), "call");
            }
        }
        bail!("{loc}: expected a model name in parentheses")
    }

    fn vm_pending(&self) -> bool {
        !(self.vm.var_decl.is_empty() && self.vm.tc_decl.is_empty() && self.vm.ve_decl.is_empty()) || self.pac_pending() // ---- pac ----
    }

    /// Reads the equations of the model block that follows, without keeping them, to learn
    /// the structure of the auxiliary models.
    pub(super) fn vm_prepass(&mut self) -> Result<()> {
        if !self.vm_pending() {
            return Ok(());
        }
        let save = self.c.i;
        loop {
            if self.c.at_end() || self.c.peek().is_none() {
                break;
            }
            let mut tag = None;
            if self.c.eat("[") {
                loop {
                    if self.c.eat("]") {
                        break;
                    }
                    if self.c.eat(",") {
                        continue;
                    }
                    let key = self.c.next()?.raw.to_ascii_lowercase();
                    let mut val = String::new();
                    if self.c.eat("=") {
                        let v = self.c.next()?;
                        val = match &v.tok {
                            Tok::Str(s) => s.clone(),
                            _ => v.raw.clone(),
                        };
                    }
                    if key == "name" {
                        tag = Some(val);
                    }
                }
            }
            if self.c.eat("#") {
                self.c.statement()?;
                continue;
            }
            let lhs = self.c.expr()?;
            let rhs = if self.c.eat("=") { self.c.expr()? } else { Ast::Num(0.0) };
            while self.c.peek().is_some_and(|t| !t.is(";")) {
                self.c.i += 1; // complementarity condition
            }
            self.c.expect(";")?;
            self.pac_collect(tag.clone(), &lhs, &rhs); // ---- pac ----
            if let Some(t) = tag {
                self.vm.eqs.retain(|(n, _, _)| *n != t);
                self.vm.eqs.push((t, lhs, rhs));
            }
        }
        self.c.i = save;
        self.c.nums.clear();
        self.vm_resolve()?;
        self.pac_resolve() // ---- pac ----
    }

    fn vm_resolve(&mut self) -> Result<()> {
        let kind = |n: &str| -> Option<SymbolKind> {
            if n.starts_with(vm::VE_PLACEHOLDER) {
                return Some(SymbolKind::Local);
            }
            self.b.symbol(n).map(|s| s.kind)
        };
        let find = |tag: &str, eqs: &[(String, Ast, Ast)]| eqs.iter().find(|(n, _, _)| n == tag).cloned();
        let eqs = std::mem::take(&mut self.vm.eqs);
        // ---- var_model
        let mut keep = Vec::new();
        let mut built = Vec::new();
        for d in std::mem::take(&mut self.vm.var_decl) {
            let (name, tags, structural, loc) = &d;
            if !tags.iter().all(|t| find(t, &eqs).is_some()) {
                keep.push(d);
                continue;
            }
            let mut specs = Vec::new();
            for t in tags {
                let (_, l, r) = find(t, &eqs).unwrap();
                let e = vm::eq_spec(t, &l, &r, &kind, true).map_err(|e| anyhow!("{loc}: var_model {name}: {e}"))?;
                if !structural && e.terms.iter().any(|x| x.lag == 0) {
                    bail!(
                        "{loc}: var_model {name}: equation '{t}' has a contemporaneous variable other than its LHS; use the structural option"
                    );
                }
                specs.push(e);
            }
            built.push(VarModel { name: name.clone(), structural: *structural, eqtags: tags.clone(), eqs: specs });
        }
        self.vm.var_decl = keep;
        // ---- trend_component_model
        let mut keep = Vec::new();
        let mut built_tc = Vec::new();
        for d in std::mem::take(&mut self.vm.tc_decl) {
            let (name, tags, targets, loc) = &d;
            if !tags.iter().all(|t| find(t, &eqs).is_some()) {
                keep.push(d);
                continue;
            }
            if let Some(t) = targets.iter().find(|t| !tags.contains(t)) {
                bail!("{loc}: trend_component_model {name}: target '{t}' is not among the eqtags");
            }
            let mut specs = Vec::new();
            for t in tags {
                let (_, l, r) = find(t, &eqs).unwrap();
                specs.push(
                    vm::eq_spec(t, &l, &r, &kind, true)
                        .map_err(|e| anyhow!("{loc}: trend_component_model {name}: {e}"))?,
                );
            }
            built_tc.push(TrendComponentModel {
                name: name.clone(),
                eqtags: tags.clone(),
                targets: tags.iter().map(|t| targets.contains(t)).collect(),
                eqs: specs,
            });
        }
        self.vm.tc_decl = keep;
        self.vm.eqs = eqs;
        self.vm.models.vars.extend(built);
        self.vm.models.trends.extend(built_tc);
        // ---- var_expectation_model
        let mut keep = Vec::new();
        for d in std::mem::take(&mut self.vm.ve_decl) {
            let (aux_vars, aux_p, kind_name): (Vec<String>, usize, &str) = if let Some(m) = self.vm.models.var(&d.aux) {
                (m.lhs(), m.max_lag(), "var")
            } else if let Some(m) = self.vm.models.trend(&d.aux) {
                (m.lhs(), m.max_lag() + 1, "trend_component")
            } else {
                keep.push(d);
                continue;
            };
            let kind = |n: &str| self.b.symbol(n).map(|s| s.kind);
            let expr = vm::expr_terms(&d.expr, &kind).map_err(|e| anyhow!("{}: {e}", d.loc))?;
            if let Discount::Param(p) = &d.discount
                && self.b.symbol(p).is_none_or(|s| s.kind != SymbolKind::Param)
            {
                bail!("{}: var_expectation_model {}: discount '{p}' is not a parameter", d.loc, d.name);
            }
            let mut layout = Vec::new();
            let mut names = Vec::new();
            let pre = format!("var_expectation_model_{}", d.name);
            if kind_name == "var" {
                names.push(format!("{pre}_constant"));
                layout.push(None);
            }
            for k in 0..aux_p {
                for v in &aux_vars {
                    names.push(format!("{pre}_{v}_{k}"));
                    layout.push(Some((v.clone(), k + (-d.time_shift).max(0) as usize))); // information set of t-tau
                }
            }
            for n in &names {
                self.b.declare(n, SymbolKind::Param).map_err(|e| anyhow!("{}: {e}", d.loc))?;
                self.labels.push(SymbolKind::Param, n.replace('_', "\\_"), n.clone());
            }
            self.vm.layout.insert(d.name.clone(), layout);
            self.vm.models.expectations.push(VarExpectationModel {
                name: d.name,
                auxiliary_model_name: d.aux,
                auxiliary_model_type: kind_name.to_string(),
                horizon: d.horizon,
                time_shift: d.time_shift,
                discount: d.discount,
                expr,
                param_names: names,
            });
        }
        self.vm.ve_decl = keep;
        Ok(())
    }

    /// Replaces `var_expectation(m)` by `const + sum param_i * var_i(-lag_i)`.
    pub(super) fn vm_expand(&mut self, a: Ast) -> Result<Ast> {
        Ok(match a {
            Ast::Var { name, lag, loc } if name.starts_with(vm::VE_PLACEHOLDER) => {
                if lag != 0 {
                    bail!("{loc}: var_expectation cannot have a lead/lag");
                }
                let m = &name[vm::VE_PLACEHOLDER.len()..];
                let (Some(e), Some(lay)) = (self.vm.models.expectation(m), self.vm.layout.get(m)) else {
                    bail!(
                        "{loc}: var_expectation({m}): unknown var_expectation_model (or its auxiliary model is not defined by the equations of the model block)"
                    )
                };
                let mut acc: Option<Ast> = None;
                for (pn, l) in e.param_names.iter().zip(lay) {
                    let p = Ast::Var { name: pn.clone(), lag: 0, loc: loc.clone() };
                    let term = match l {
                        None => p,
                        Some((v, k)) => Ast::Bin(
                            BinaryOp::Mul,
                            Box::new(p),
                            Box::new(Ast::Var { name: v.clone(), lag: -(*k as i32), loc: loc.clone() }),
                        ),
                    };
                    acc = Some(match acc {
                        None => term,
                        Some(s) => Ast::Bin(BinaryOp::Add, Box::new(s), Box::new(term)),
                    });
                }
                acc.ok_or_else(|| anyhow!("{loc}: var_expectation({m}): empty model"))?
            }
            Ast::Un(o, x) => Ast::Un(o, Box::new(self.vm_expand(*x)?)),
            Ast::Bin(o, x, y) => Ast::Bin(o, Box::new(self.vm_expand(*x)?), Box::new(self.vm_expand(*y)?)),
            Ast::Op(o, v) => Ast::Op(o, v.into_iter().map(|x| self.vm_expand(x)).collect::<Result<_>>()?),
            other => other,
        })
    }

    /// End of the file: every declaration must have found its equations.
    pub(super) fn vm_finish(&mut self) -> Result<vm::VarModels> {
        if let Some((n, tags, _, loc)) = self.vm.var_decl.first() {
            let missing =
                tags.iter().find(|t| !self.vm.eqs.iter().any(|(x, _, _)| x == *t)).cloned().unwrap_or_default();
            bail!("{loc}: var_model {n}: equation tag '{missing}' not found in the model block");
        }
        if let Some((n, tags, _, loc)) = self.vm.tc_decl.first() {
            let missing =
                tags.iter().find(|t| !self.vm.eqs.iter().any(|(x, _, _)| x == *t)).cloned().unwrap_or_default();
            bail!("{loc}: trend_component_model {n}: equation tag '{missing}' not found in the model block");
        }
        if let Some(d) = self.vm.ve_decl.first() {
            bail!("{}: var_expectation_model {}: unknown auxiliary model '{}'", d.loc, d.name, d.aux);
        }
        Ok(std::mem::take(&mut self.vm.models))
    }
}

/// `horizon = 3`, `1:5`, `0:Inf`, `[1:5]`.
fn vm_horizon(toks: &[Token], loc: &str) -> Result<Horizon> {
    let t: Vec<&Token> = toks.iter().filter(|t| !t.is("[") && !t.is("]")).collect();
    let int = |t: &Token| -> Result<usize> {
        match t.tok {
            Tok::Num(v) if v >= 0.0 && v.fract() == 0.0 => Ok(v as usize),
            _ => bail!("{loc}: horizon must be an integer or a range of integers"),
        }
    };
    match t.as_slice() {
        [a] => Ok(Horizon::Single(int(a)?)),
        [a, c, b] if c.is(":") => {
            let hi = match &b.tok {
                Tok::Ident(s) if s.eq_ignore_ascii_case("inf") => f64::INFINITY,
                _ => int(b)? as f64,
            };
            Ok(Horizon::Range(int(a)?, hi))
        }
        _ => bail!("{loc}: horizon must be an integer or a range of integers"),
    }
}

/// Placeholder for `var_expectation(m)` in the expression parser.
pub(super) fn ve_placeholder(c: &mut Cursor, loc: &Loc) -> Result<Ast> {
    c.expect("(")?;
    if c.at_ident("model_name") && c.peek_at(1).is_some_and(|t| t.is("=")) {
        c.i += 2;
    }
    let t = c.next()?;
    let n = match &t.tok {
        Tok::Ident(s) | Tok::Str(s) => s.clone(),
        _ => bail!("{}: var_expectation(NAME): expected a model name", t.loc),
    };
    c.expect(")")?;
    Ok(Ast::Var { name: format!("{}{}", vm::VE_PLACEHOLDER, n), lag: 0, loc: loc.clone() })
}

/// In a `var_model` whose LHS is `diff(z)` the VAR variable is the auxiliary variable
/// `AUX_DIFF_<n>` of the preprocessor, which names the `var_expectation_model_<m>_*`
/// parameters (`var_expectation_model_ve_AUX_DIFF_10_0`): renamed once the model is built.
pub(super) fn rename_aux_diff(models: &mut VarModels, model: &mut crate::model::Model) {
    // The preprocessor numbers the nodes of its DataTree in creation order, and the
    // `expression` of a var_expectation_model (declared before the model block) creates its
    // variable nodes first: AUX_DIFF_<n> is shifted by the nodes created before `z`.
    let mut created: Vec<String> = Vec::new();
    for e in &models.expectations {
        for t in &e.expr {
            if !created.contains(&t.var) {
                created.push(t.var.clone());
            }
        }
    }
    for e in &mut models.expectations {
        let Some(vmod) = models.vars.iter().find(|m| m.name == e.auxiliary_model_name) else { continue };
        for eq in vmod.eqs.iter().filter(|q| q.lhs_diff) {
            let Some(zi) = model.endo_index(&eq.lhs) else { continue };
            let Some(aux) = model
                .aux_vars
                .iter()
                .find(|a| a.ty == 8 && a.orig_index == Some(zi) && a.orig_lead_lag == Some(0))
                .map(|a| model.endo_names[a.endo_index].clone())
            else {
                continue;
            };
            let n0 = aux.strip_prefix("AUX_DIFF_").unwrap_or(&aux);
            let shift = created.iter().position(|v| *v == eq.lhs).unwrap_or(created.len());
            let n = n0.parse::<usize>().map_or_else(|_| n0.to_string(), |k| (k + shift).to_string());
            for p in &mut e.param_names {
                let old = format!("var_expectation_model_{}_{}_", e.name, eq.lhs);
                if let Some(k) = p.strip_prefix(&old) {
                    let new = format!("var_expectation_model_{}_AUX_DIFF_{n}_{k}", e.name);
                    model.rename_param(p, &new);
                    *p = new;
                }
            }
        }
    }
}
