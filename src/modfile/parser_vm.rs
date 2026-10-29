// SPDX-License-Identifier: GPL-3.0-or-later
//! ---- var models ----: `var_model`, `trend_component_model`, `var_expectation_model`, the
//! `var_expectation(m)` operator and the `var_expectation.initialize/update` statements.
//!
//! The equations of the auxiliary models come after their declaration, and the number of
//! `var_expectation_model_<m>_*` parameters (which the operator stands for) depends on their
//! lags: at the start of each `model` block the equations are parsed once to find them
//! ([`Parser::vm_prepass`]). The operator itself is replaced in the preprocessor's transform
//! pass (`model::dtree::pac`), after the auxiliary variables of the diff operators.

use super::{Ast, Cursor, Parser, slice_expr_nums, split_commas};
use crate::model::SymbolKind;
use crate::model::dtree::ModelItem;
use crate::model::dtree::pac::VeSpec;
use crate::modfile::lexer::{Loc, Tok, Token};
use crate::modfile::{Command, OptValue};
use crate::varmodels::{self as vm, Discount, Horizon, TrendComponentModel, VarExpectationModel, VarModel, VarModels};
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;

struct VeDecl {
    name: String,
    aux: String,
    expr: Ast,
    /// Rank of `expr` among the preamble expressions of the model's tree.
    pre: usize,
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
    /// Keys of the expressions handed to the transform pass (`ModelBuilder::pac.keys`).
    keys: Vec<String>,
    /// Token ranges of the declarations that follow a model block, read before it
    /// ([`Parser::vm_hoist`]) and skipped where they stand.
    hoisted: Vec<(usize, usize)>,
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
        if kw == "var_expectation" || kw == "get_companion_matrix" {
            // native MATLAB statements for the preprocessor: they end with their line
            self.c.native_end = self.c.native_line_end();
        }
        self.c.i += 1;
        if kw == "var_expectation" {
            self.c.expect(".")?;
            let (f, _) = self.c.ident()?;
            let f = f.to_string();
            if f != "initialize" && f != "update" && f != "print" {
                bail!("{loc}: var_expectation.{f} is not supported (only initialize, update and print)");
            }
            let toks = self.c.statement()?;
            let mut o = Vec::new();
            if f == "print" && toks.len() >= 2 && toks[0].is("(") && toks[toks.len() - 1].is(")") {
                // var_expectation.print(name[, withcalibration])
                let args = split_commas(&toks[1..toks.len() - 1]);
                if let Some(a) = args.get(1) {
                    let v = match a {
                        [t] if t.is_ident("true") => 1.0,
                        [t] if t.is_ident("false") => 0.0,
                        _ => self.eval_tokens(a)?,
                    };
                    o.push(("withcalibration".to_string(), OptValue::Num(v)));
                }
                let arg = self.vm_call_arg(&[&toks[..1], args[0], &toks[toks.len() - 1..]].concat(), &loc)?;
                o.insert(0, (arg, OptValue::Flag));
            } else {
                o.push((self.vm_call_arg(toks, &loc)?, OptValue::Flag));
            }
            self.commands.push(Command::Other(format!("var_expectation.{f}"), o));
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
                let (expr, nums) = slice_expr_nums(
                    get("expression").ok_or_else(|| anyhow!("{loc}: var_expectation_model: expression is required"))?,
                )?;
                let h = get("horizon").ok_or_else(|| anyhow!("{loc}: var_expectation_model: horizon is required"))?;
                let horizon = vm_horizon(h, &loc)?;
                let time_shift = match get("time_shift") {
                    None => 0,
                    Some(t) => {
                        let v = self.eval_tokens(t)?;
                        if v.fract() != 0.0 {
                            bail!("{loc}: syntax error, the 'time_shift' option expects an integer");
                        }
                        // VarExpectationModelStatement (checkPass)
                        if v > 0.0 {
                            bail!("{loc}: The 'time_shift' option must be a non-positive integer");
                        }
                        v as i32
                    }
                };
                let discount = match get("discount") {
                    None => bail!("{loc}: var_expectation_model: discount is required"),
                    Some([t]) if matches!(&t.tok, Tok::Ident(_)) => Discount::Param(t.raw.clone()),
                    Some(t) => Discount::Value(self.eval_tokens(t)?),
                };
                let pre = self.preamble(expr.clone(), nums);
                self.vm.ve_decl.push(VeDecl { name: model_name, aux, expr, pre, horizon, time_shift, discount, loc });
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

    /// An expression parsed into the model's tree before the model block
    /// ([`ModelItem::Preamble`]); returns its rank.
    pub(super) fn preamble(&mut self, ast: Ast, nums: Vec<String>) -> usize {
        let k = self.b.items.iter().filter(|i| matches!(i, ModelItem::Preamble { .. })).count();
        self.b.items.push(ModelItem::Preamble { ast, nums });
        k
    }

    /// The index of an expression at `t` of an auxiliary model among the keys that the
    /// transform pass resolves.
    pub(super) fn vm_key(&mut self, key: &str, expr: &Ast) -> usize {
        if let Some(k) = self.vm.keys.iter().position(|x| x == key) {
            return k;
        }
        self.vm.keys.push(key.to_string());
        self.b.pac.keys.push(expr.clone());
        self.vm.keys.len() - 1
    }

    /// The preprocessor builds the auxiliary and PAC models in its transform pass, once the whole
    /// file is read: their declarations may follow the model block (tcm6.mod). Those that come
    /// later in the file are read now, before the prepass of the model block.
    pub(super) fn vm_hoist(&mut self) -> Result<()> {
        let save = self.c.i;
        let t = self.c.t;
        for k in save + 1..t.len() {
            let Some(kw) = t[k].ident() else { continue };
            let starts = t[k - 1].is(";") || t[k - 1].loc.line < t[k].loc.line;
            if !(starts
                && matches!(kw, "var_model" | "trend_component_model" | "var_expectation_model" | "pac_model")
                && t.get(k + 1).is_some_and(|n| n.is("("))
                && !self.vm.hoisted.iter().any(|(a, _)| *a == k))
            {
                continue;
            }
            self.c.i = k;
            let r = if kw == "pac_model" { self.pac_statement(kw) } else { self.vm_statement(kw) };
            let end = self.c.i;
            self.c.i = save;
            r?;
            self.vm.hoisted.push((k, end));
        }
        Ok(())
    }

    /// End of the hoisted declaration that starts here, if any.
    pub(super) fn vm_hoisted_end(&self) -> Option<usize> {
        self.vm.hoisted.iter().find(|(a, _)| *a == self.c.i).map(|(_, e)| *e)
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
            let n0 = self.c.nums.len();
            let lhs = self.c.expr()?;
            let rhs = if self.c.eat("=") { self.c.expr()? } else { Ast::Num(0.0) };
            let nums = self.c.nums[n0..].to_vec();
            while self.c.peek().is_some_and(|t| !t.is(";")) {
                self.c.i += 1; // complementarity condition
            }
            self.c.expect(";")?;
            self.pac_collect(tag.clone(), &lhs, &rhs, nums); // ---- pac ----
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
                target_eqtags: targets.clone(),
                eqs: specs,
            });
        }
        self.vm.tc_decl = keep;
        self.vm.eqs = eqs;
        // the equations whose unary operators get auxiliary variables, the companion variables
        let comp: Vec<(String, Ast)> = built
            .iter()
            .flat_map(|m| m.companion_exprs())
            .chain(built_tc.iter().flat_map(|m| m.companion_exprs()))
            .collect();
        for (k, e) in comp {
            self.vm_key(&k, &e);
        }
        let tags = built.iter().flat_map(|m| m.eqtags.iter()).chain(built_tc.iter().flat_map(|m| m.eqtags.iter()));
        self.b.pac.vm_tags.extend(tags.cloned());
        self.vm.models.vars.extend(built);
        self.vm.models.trends.extend(built_tc);
        // ---- var_expectation_model
        let mut keep = Vec::new();
        for d in std::mem::take(&mut self.vm.ve_decl) {
            let (aux_vars, aux_p, kind_name) = if let Some(m) = self.vm.models.var(&d.aux) {
                (m.companion_exprs(), m.max_lag(), "var")
            } else if let Some(m) = self.vm.models.trend(&d.aux) {
                (m.companion_exprs(), m.max_lag() + 1, "trend_component")
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
            // the parameters are named after the variables of the companion VAR (renamed after
            // the auxiliary variables that stand for them once the model is built)
            let mut names = Vec::new();
            let pre = format!("var_expectation_model_{}", d.name);
            if kind_name == "var" {
                names.push(format!("{pre}_constant"));
            }
            for k in 0..aux_p {
                for (v, _) in &aux_vars {
                    names.push(format!("{pre}_{v}_{k}"));
                }
            }
            for n in &names {
                self.b.declare_late_param(n).map_err(|e| anyhow!("{}: {e}", d.loc))?;
                self.labels.push(SymbolKind::Param, n.replace('_', "\\_"), n.clone());
            }
            let lhs: Vec<usize> = aux_vars.iter().map(|(k, e)| self.vm_key(k, e)).collect();
            self.b.pac.ve.push(VeSpec {
                name: d.name.clone(),
                expr: d.pre,
                is_var: kind_name == "var",
                lhs,
                max_lag: aux_p,
                params: names.clone(),
                time_shift: d.time_shift,
            });
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
        self.vm.models.keys = std::mem::take(&mut self.vm.keys);
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

/// Once the model is built, the variables of the auxiliary models (`diff(z)`, `log(x)`, ...)
/// get the names of the auxiliary variables that stand for them (`AUX_DIFF_<n>`,
/// `AUX_UOP_<n>`, from the transform pass), and so do the `var_expectation_model_<m>_*` and
/// PAC `h` parameters named after them (`var_expectation_model_ve_AUX_DIFF_10_0`,
/// `h_pacman_var_AUX_DIFF_11_lag_1`): renamed in the model, the models and the PAC equations.
pub(super) fn resolve_names(models: &mut VarModels, model: &mut crate::model::Model) {
    if models.keys.is_empty() || models.keys.len() != model.pac_key_names.len() {
        return;
    }
    models.aux = model
        .aux_vars
        .iter()
        .filter(|a| matches!(a.ty, 8 | 10))
        .map(|a| {
            let orig = a.orig_index.map(|i| model.endo_names[i].clone()).unwrap_or_default();
            (model.endo_names[a.endo_index].clone(), a.ty, orig)
        })
        .collect();
    let names: HashMap<String, String> = models.keys.iter().cloned().zip(model.pac_key_names.iter().cloned()).collect();
    let get = |k: &str| names.get(k).cloned().unwrap_or_else(|| k.to_string());
    for e in models
        .vars
        .iter_mut()
        .flat_map(|m| m.eqs.iter_mut())
        .chain(models.trends.iter_mut().flat_map(|m| m.eqs.iter_mut()))
    {
        e.lhs_name = get(&e.lhs);
        if let Some((k, _)) = &e.undiff {
            e.undiff_name = get(k);
        }
    }
    let mut map: HashMap<String, String> = HashMap::new();
    for e in &mut models.expectations {
        for t in &mut e.expr {
            t.var = get(&t.var);
        }
        let comp: Vec<String> = match (
            models.vars.iter().find(|m| m.name == e.auxiliary_model_name),
            models.trends.iter().find(|m| m.name == e.auxiliary_model_name),
        ) {
            (Some(m), _) => m.companion_exprs().into_iter().map(|x| x.0).collect(),
            (None, Some(m)) => m.companion_exprs().into_iter().map(|x| x.0).collect(),
            _ => continue,
        };
        let pre = format!("var_expectation_model_{}_", e.name);
        for p in &mut e.param_names {
            // `<pre><key>_<lag>`
            for k in &comp {
                if let Some(lag) = p.strip_prefix(&format!("{pre}{k}_")).filter(|l| l.parse::<usize>().is_ok()) {
                    let new = format!("{pre}{}_{lag}", get(k));
                    if new != *p {
                        map.insert(p.clone(), new.clone());
                        *p = new;
                    }
                    break;
                }
            }
        }
    }
    for p in &mut models.pacs {
        let lay: Vec<Option<(String, usize)>> =
            p.h_layout.iter().zip(&p.h_keys).map(|(l, k)| Some((k.clone()?, l.as_ref()?.2))).collect();
        // `<prefix>_var_<key>_lag_<lag>`
        let mut ren = |hp: &mut Vec<String>| {
            for (h, l) in hp.iter_mut().zip(&lay) {
                let Some((k, lag)) = l else { continue };
                if let Some(prefix) = h.strip_suffix(&format!("_var_{k}_lag_{lag}")) {
                    let new = format!("{prefix}_var_{}_lag_{lag}", get(k));
                    if new != *h {
                        map.insert(h.clone(), new.clone());
                        *h = new;
                    }
                }
            }
        };
        ren(&mut p.h_params);
        p.target = get(&p.target);
        p.lhs_var = get(&p.lhs_var);
        for c in &mut p.components {
            ren(&mut c.h_params);
            c.endo = get(&c.endo_key);
        }
    }
    if map.is_empty() {
        return;
    }
    for (old, new) in &map {
        model.rename_param(old, new);
    }
    fn rename(a: &mut Ast, map: &HashMap<String, String>) {
        match a {
            Ast::Var { name, .. } => {
                if let Some(n) = map.get(name.as_str()) {
                    *name = n.clone();
                }
            }
            Ast::Num(_) => {}
            Ast::Un(_, x) => rename(x, map),
            Ast::Bin(_, x, y) => {
                rename(x, map);
                rename(y, map);
            }
            Ast::Op(_, v) => v.iter_mut().for_each(|x| rename(x, map)),
        }
    }
    for p in &mut models.pacs {
        rename(&mut p.eq_lhs, &map);
        rename(&mut p.eq_rhs, &map);
        for (_, d) in &mut p.defs {
            rename(d, &map);
        }
    }
}
