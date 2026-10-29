// SPDX-License-Identifier: GPL-3.0-or-later
//! The VAR-model / PAC part of `ModFile::transformPass` (stream PP):
//! `DynamicModel::substituteUnaryOps` (on the equations of the `var_model` /
//! `trend_component_model` and on the PAC equations), the PAC growth expressions in
//! `substituteDiff`, `VarExpectationModelTable::transformPass` (the `var_expectation`
//! operator) and `PacModelTable::transformPass` (the `pac_expectation` /
//! `pac_target_nonstationary` auxiliary variables, created after the diff auxiliaries:
//! `computePacBackwardExpectationSubstitution[WithComponents]`,
//! `computePacModelConsistentExpectationSubstitution`).
//!
//! The parser analyses the PAC equations and the auxiliary models on its syntax trees and
//! declares the parameters; what it hands over here ([`PacInput`]) is what fixes the nodes and
//! symbols the preprocessor creates: the expressions of `var_expectation_model(expression=...)`,
//! `pac_model(growth=...)` and of the `pac_target_info` blocks (parsed into the model's tree
//! before the model block), the companion variables of the auxiliary models as expressions at
//! `t` ([`PacInput::keys`], resolved into the auxiliary variables that stand for them), and the
//! order in which the parameters and variables are created.

use super::{BOp, Id, LagEquiv, N, Sym, SymType, Tree, UOp};
use crate::modfile::Ast;
use anyhow::{Result, anyhow, bail};
use std::collections::{BTreeSet, HashMap};

/// Placeholder variable name of `pac_expectation(m)` in the parser's trees.
pub const PE: &str = "\u{1}pac_expectation:";
/// Placeholder variable name of `pac_target_nonstationary(m)`.
pub const PNS: &str = "\u{1}pac_target_nonstationary:";
/// Placeholder variable name of `var_expectation(m)`.
pub const VE: &str = "\u{1}var_expectation:";
/// Symbol ids of the placeholders (`PH_BASE + k`, `k` the rank in [`Tree::placeholders`]):
/// outside the symbol table.
pub(super) const PH_BASE: u32 = 1 << 30;

/// One `var_expectation_model`, in declaration order.
#[derive(Clone, Debug)]
pub struct VeSpec {
    pub name: String,
    /// The preamble expression of the `expression` option.
    pub expr: usize,
    pub is_var: bool,
    /// The companion variables (indices in [`PacInput::keys`]) and the number of lags.
    pub lhs: Vec<usize>,
    pub max_lag: usize,
    /// The parameters, constant first for a VAR, then by lag and variable.
    pub params: Vec<String>,
    pub time_shift: i32,
}

/// One `pac_model`, in declaration order.
#[derive(Clone, Debug)]
pub struct PacSpec {
    pub name: String,
    /// `pac_model(growth = ...)`: the preamble expression and the growth neutrality parameter.
    pub growth: Option<(usize, String)>,
    pub kind: PacKind,
}

#[derive(Clone, Debug)]
pub enum PacKind {
    /// Model consistent expectations: the target variable, the number of autoregressive terms
    /// of the PAC equation, the `mce_alpha` parameters, the discount parameter, `mce_Z1_<m>`.
    Mce { target: String, max_lag: usize, alpha: Vec<String>, discount: String, auxname: String },
    /// Backward expectation from a VAR / trend component model: the companion variables
    /// (indices in [`PacInput::keys`]), the lags, the `h` parameters (constant first for a VAR,
    /// then by lag and variable) and the auxiliary variable.
    Backward { is_var: bool, lhs: Vec<usize>, max_lag: usize, h: Vec<String>, auxname: String },
    /// `pac_target_info`: the preamble expression of the target, `auxname_target_nonstationary`
    /// and the components.
    Composite { is_var: bool, lhs: Vec<usize>, max_lag: usize, target: usize, vns: String, comps: Vec<CompSpec> },
}

#[derive(Clone, Debug)]
pub struct CompSpec {
    /// Preamble expression of the component.
    pub expr: usize,
    /// `kind ll` (stationary).
    pub ll: bool,
    pub h: Vec<String>,
    pub growth: Option<(usize, String)>,
    pub auxname: String,
}

/// What the parser hands over (see the module documentation).
#[derive(Clone, Debug, Default)]
pub struct PacInput {
    /// `eqtags` of the `var_model` / `trend_component_model` equations.
    pub vm_tags: Vec<String>,
    /// Expressions at `t` (`x`, `diff(x)`, `diff(log(x))`, ...) to resolve into the variables
    /// that stand for them after the substitutions.
    pub keys: Vec<Ast>,
    pub ve: Vec<VeSpec>,
    pub models: Vec<PacSpec>,
    /// The parameters created by the PAC transformation: name -> parameter index.
    pub late_params: Vec<(String, usize)>,
}

impl PacInput {
    pub fn is_empty(&self) -> bool {
        self.vm_tags.is_empty() && self.keys.is_empty() && self.models.is_empty() && self.ve.is_empty()
    }

    /// The placeholder names, in the order of their symbols.
    pub(super) fn placeholders(&self) -> Vec<String> {
        let mut v: Vec<String> = self.ve.iter().map(|m| format!("{VE}{}", m.name)).collect();
        for m in &self.models {
            v.push(format!("{PE}{}", m.name));
            v.push(format!("{PNS}{}", m.name));
        }
        v
    }
}

/// What the PAC transformation returns: the names of [`PacInput::keys`] and the symbol ids of
/// the late parameters.
#[derive(Clone, Debug, Default)]
pub struct PacOutput {
    pub key_names: Vec<String>,
    pub param_symb: Vec<(String, u32)>,
}

impl Tree {
    pub(super) fn sym_name(&self, s: u32) -> String {
        if s < PH_BASE {
            return self.names[s as usize].clone();
        }
        let p = self.placeholders.get((s - PH_BASE) as usize).map_or("", |x| x.as_str());
        for (pre, op) in [(PE, "pac_expectation"), (PNS, "pac_target_nonstationary"), (VE, "var_expectation")] {
            if let Some(m) = p.strip_prefix(pre) {
                return format!("{op}(model_name = {m})");
            }
        }
        p.to_string()
    }

    /// The symbol of a placeholder name of the parser.
    pub(super) fn placeholder_symbol(&self, name: &str) -> Option<Result<u32>> {
        if !(name.starts_with(PE) || name.starts_with(PNS) || name.starts_with(VE)) {
            return None;
        }
        Some(self.placeholders.iter().position(|n| n == name).map(|k| PH_BASE + k as u32).ok_or_else(|| {
            let what = if name.starts_with(VE) { "var_expectation_model" } else { "PAC model" };
            anyhow!("unknown {what} '{}'", &name[name.find(':').map_or(0, |i| i + 1)..])
        }))
    }

    fn placeholder_of(&self, name: &str) -> u32 {
        PH_BASE + self.placeholders.iter().position(|n| n == name).unwrap() as u32
    }

    fn contains_placeholder(&self, id: Id, pred: &dyn Fn(u32) -> bool, seen: &mut BTreeSet<Id>) -> bool {
        if !seen.insert(id) {
            return false;
        }
        match self.node(id) {
            N::Num(_) => false,
            N::Var(s, _) => pred(s),
            N::Un(_, a, _) => self.contains_placeholder(a, pred, seen),
            N::Bin(_, a, b) => self.contains_placeholder(a, pred, seen) || self.contains_placeholder(b, pred, seen),
            N::Tri(_, a, b, c) => {
                self.contains_placeholder(a, pred, seen)
                    || self.contains_placeholder(b, pred, seen)
                    || self.contains_placeholder(c, pred, seen)
            }
            N::Ext(_, l) => self.ext_args(l).into_iter().any(|a| self.contains_placeholder(a, pred, seen)),
        }
    }

    // ---- unary operators -------------------------------------------------------------------

    /// `findUnaryOpNodesForAuxVarCreation`.
    fn find_uop_nodes(&mut self, id: Id, nodes: &mut LagEquiv, seen: &mut BTreeSet<Id>) {
        if !seen.insert(id) {
            return;
        }
        match self.node(id) {
            N::Num(_) => {}
            N::Var(s, lag) => {
                if self.sym_ty(s) == SymType::Local {
                    let v = self.get_local(s, lag);
                    self.find_uop_nodes(v, nodes, seen);
                }
            }
            N::Un(op, a, _) => {
                self.find_uop_nodes(a, nodes, seen);
                if let UOp::F(_) = op {
                    let (repr, index) = self.lag_equiv_class(id);
                    nodes.entry(repr).or_default().insert(index, id);
                }
            }
            N::Bin(_, a, b) => {
                self.find_uop_nodes(a, nodes, seen);
                self.find_uop_nodes(b, nodes, seen);
            }
            N::Tri(_, a, b, c) => {
                self.find_uop_nodes(a, nodes, seen);
                self.find_uop_nodes(b, nodes, seen);
                self.find_uop_nodes(c, nodes, seen);
            }
            N::Ext(_, l) => {
                for a in self.ext_args(l) {
                    self.find_uop_nodes(a, nodes, seen);
                }
            }
        }
    }

    /// `substituteUnaryOpNodes`: `AUX_UOP_<idx>` for the (lead/lag classes of the) marked
    /// unary operators.
    pub(super) fn subst_uop(&mut self, id: Id, neweqs: &mut Vec<Id>) -> Result<Id> {
        match self.node(id) {
            N::Num(_) => Ok(id),
            N::Var(s, lag) => {
                if self.sym_ty(s) == SymType::Local {
                    let v = self.get_local(s, lag);
                    self.subst_uop(v, neweqs)
                } else {
                    Ok(id)
                }
            }
            N::Bin(op, a, b) => {
                let a2 = self.subst_uop(a, neweqs)?;
                let b2 = self.subst_uop(b, neweqs)?;
                self.build_bin(op, a2, b2)
            }
            N::Tri(op, a, b, c) => {
                let a2 = self.subst_uop(a, neweqs)?;
                let b2 = self.subst_uop(b, neweqs)?;
                let c2 = self.subst_uop(c, neweqs)?;
                Ok(self.build_tri(op, a2, b2, c2))
            }
            N::Ext(f, l) => {
                let mut v = Vec::new();
                for a in self.ext_args(l) {
                    v.push(self.subst_uop(a, neweqs)?);
                }
                Ok(self.add_ext(f, v))
            }
            N::Un(op, a, info) => {
                if let Some(&r) = self.uop_table.get(&id) {
                    return Ok(r);
                }
                let (repr, index) = self.lag_equiv_class(id);
                let class = self.uop_nodes.get(&repr).cloned();
                let argsubst = self.subst_uop(a, neweqs)?;
                let Some(class) = class else { return Ok(self.build_un(op, info, argsubst)) };
                let base_index = *class.keys().next_back().unwrap();
                let mut aux_var = None;
                for (k, (&idx, &node)) in class.iter().rev().enumerate() {
                    if k == 0 {
                        if self.max_lead(node) > 0 {
                            bail!("Cannot substitute unary operations that contain leads");
                        }
                        let shifted = self.decrease(argsubst, index - base_index);
                        let def = self.build_un(op, info, shifted);
                        let (os, ol) = match self.node(shifted) {
                            N::Var(s, l) => (Some(s), Some(l)),
                            _ => (None, None),
                        };
                        let symb = self.add_aux(format!("AUX_UOP_{id}"), 10, os, ol, def);
                        let v = self.add_var(symb, 0);
                        neweqs.push(self.add_equal(v, def));
                        self.uop_table.insert(node, v);
                        aux_var = Some(v);
                    } else {
                        let v = self.decrease(aux_var.unwrap(), base_index - idx);
                        self.uop_table.insert(node, v);
                    }
                }
                self.uop_table.get(&id).copied().ok_or_else(|| anyhow!("internal error in the unary op substitution"))
            }
        }
    }

    /// `DynamicModel::substituteUnaryOps(eqnumbers, ...)`: the equations of the VAR / trend
    /// component models and the PAC equations, then the PAC growth expressions.
    pub(super) fn substitute_unary_ops(&mut self, inp: &PacInput) -> Result<()> {
        let mut eqs = BTreeSet::new();
        for tag in &inp.vm_tags {
            let found: Vec<usize> = (0..self.equations.len())
                .filter(|&k| self.equations[k].tags.iter().any(|(t, v)| t == "name" && v == tag))
                .collect();
            if found.is_empty() {
                bail!("looking for equation tag {tag} failed.");
            }
            eqs.extend(found);
        }
        for k in 0..self.equations.len() {
            let e = self.equations[k].eq;
            let pe: Vec<u32> = inp.models.iter().map(|m| self.placeholder_of(&format!("{PE}{}", m.name))).collect();
            if self.contains_placeholder(e, &|s| pe.contains(&s), &mut BTreeSet::new()) {
                eqs.insert(k);
            }
        }
        let mut nodes = LagEquiv::new();
        let mut seen = BTreeSet::new();
        for &k in &eqs {
            let e = self.equations[k].eq;
            self.find_uop_nodes(e, &mut nodes, &mut seen);
        }
        self.uop_nodes = nodes;
        let mut neweqs = Vec::new();
        for &k in &eqs {
            let e = self.equations[k].eq;
            self.equations[k].eq = self.subst_uop(e, &mut neweqs)?;
        }
        for m in &inp.ve {
            self.pre[m.expr] = self.subst_uop(self.pre[m.expr], &mut neweqs)?; // substituteUnaryOpsInExpression
        }
        for g in self.growth_order(inp) {
            self.pre[g] = self.subst_uop(self.pre[g], &mut neweqs)?;
        }
        self.push_neweqs(neweqs);
        Ok(())
    }

    /// The growth expressions in the order of `PacModelTable` (`growth` then `target_info`,
    /// both maps ordered by PAC model name).
    pub(super) fn growth_order(&self, inp: &PacInput) -> Vec<usize> {
        let mut models: Vec<&PacSpec> = inp.models.iter().collect();
        models.sort_by(|a, b| a.name.cmp(&b.name));
        let mut out: Vec<usize> = models.iter().filter_map(|m| m.growth.as_ref().map(|g| g.0)).collect();
        for m in &models {
            if let PacKind::Composite { comps, .. } = &m.kind {
                out.extend(comps.iter().filter_map(|c| c.growth.as_ref().map(|g| g.0)));
            }
        }
        out
    }

    /// Unary op then diff substitution of an expression of the model that must not create
    /// any auxiliary variable (`what` names it in the error).
    fn subst_existing(&mut self, id: Id, what: &str) -> Result<Id> {
        let mut neweqs = Vec::new();
        let a = self.subst_uop(id, &mut neweqs)?;
        if !neweqs.is_empty() {
            bail!("{what} contains a variable with a unary operator that is not present in the model");
        }
        let nodes = std::mem::take(&mut self.diff_nodes);
        let mut table = std::mem::take(&mut self.diff_table);
        let r = self.subst_diff(a, &nodes, &mut table, &mut neweqs);
        self.diff_nodes = nodes;
        self.diff_table = table;
        let r = r?;
        if !neweqs.is_empty() {
            bail!("{what} contains a diff'd variable that is not present in the model");
        }
        Ok(r)
    }

    /// The variable that stands for `ast` (an expression at `t`) after the substitutions.
    fn resolve_key(&mut self, ast: &Ast, lookup: &dyn Fn(&str) -> Option<u32>) -> Result<u32> {
        let mut it = [].iter();
        let id = self.replay(ast, &mut it, lookup)?;
        let r = self.subst_existing(id, "a variable of an auxiliary model")?;
        match self.node(r) {
            N::Var(s, 0) => Ok(s),
            _ => bail!("internal error: the variable of an auxiliary model is not a variable after the substitutions"),
        }
    }

    fn create_param(&mut self, name: &str, late: &HashMap<&str, usize>, out: &mut PacOutput) -> Result<u32> {
        let index = *late.get(name).ok_or_else(|| anyhow!("internal error: undeclared PAC parameter {name}"))?;
        let s = self.syms.len() as u32;
        self.syms.push(Sym { ty: SymType::Param, index });
        self.names.push(name.to_string());
        out.param_symb.push((name.to_string(), s));
        Ok(s)
    }

    /// `PacModelTable::transformPass` (after `substituteDiff`) and the resolution of the keys.
    pub(super) fn pac_transform(&mut self, inp: &PacInput, lookup: &dyn Fn(&str) -> Option<u32>) -> Result<PacOutput> {
        let mut out = PacOutput::default();
        let mut keys = Vec::with_capacity(inp.keys.len());
        for k in &inp.keys {
            let s = self.resolve_key(k, lookup)?;
            keys.push(s);
            out.key_names.push(self.names[s as usize].clone());
        }
        let late: HashMap<&str, usize> = inp.late_params.iter().map(|(n, i)| (n.as_str(), *i)).collect();
        // ---- VarExpectationModelTable::transformPass
        let mut vsubst = Vec::new();
        for m in &inp.ve {
            let mut pi = m.params.iter();
            let mut next = |t: &mut Tree, out: &mut PacOutput| -> Result<Id> {
                let p = pi.next().ok_or_else(|| anyhow!("internal error: var_expectation_model parameters"))?;
                let s = t.create_param(p, &late, out)?;
                Ok(t.add_var(s, 0))
            };
            let mut sub = self.zero;
            if m.is_var {
                let pv = next(self, &mut out)?;
                sub = self.add_plus(sub, pv);
            }
            for lag in 0..m.max_lag {
                for &i in &m.lhs {
                    let pv = next(self, &mut out)?;
                    let x = self.add_var(keys[i], -(lag as i32) + m.time_shift);
                    let t = self.add_times(pv, x);
                    sub = self.add_plus(sub, t);
                }
            }
            vsubst.push((self.placeholder_of(&format!("{VE}{}", m.name)), sub));
        }
        for (ph, e) in vsubst {
            self.replace_placeholder(ph, e)?;
        }
        let mut subst: Vec<(usize, Id)> = Vec::new();
        for (k, m) in inp.models.iter().enumerate() {
            // ---- pac_target_info
            let mut comps: Vec<(Id, Id)> = Vec::new(); // (component, coefficient)
            if let PacKind::Composite { target, vns, comps: cs, .. } = &m.kind {
                let what = format!("the 'target' expression of 'pac_target_info({})'", m.name);
                let t = self.subst_existing(self.pre[*target], &what)?;
                let mut cnodes = Vec::new();
                for c in cs {
                    let what = format!("a 'component' expression of 'pac_target_info({})'", m.name);
                    cnodes.push(self.subst_existing(self.pre[c.expr], &what)?);
                }
                let rhs = (0..self.equations.len())
                    .map(|q| self.eq_args(self.equations[q].eq))
                    .find(|(l, _)| *l == t)
                    .map(|(_, r)| r)
                    .ok_or_else(|| {
                        anyhow!(
                            "there is no equation whose LHS is equal to the 'target' of 'pac_target_info({})'",
                            m.name
                        )
                    })?;
                let what = format!("the equation defining the target of 'pac_target_info({})'", m.name);
                let rhs = self.subst_existing(rhs, &what)?;
                let (terms, constant) = self.match_linear_endo_with_constant(rhs).ok_or_else(|| {
                    anyhow!(
                        "the model equation defining the 'target' of 'pac_target_info({})' is not of the right form (should be a linear combination of endogenous variables)",
                        m.name
                    )
                })?;
                let mut coeffs: Vec<Option<Id>> = vec![None; cnodes.len()];
                for (v, c) in terms {
                    let vn = self.add_var(v, 0);
                    let Some(j) = cnodes.iter().position(|&x| x == vn) else {
                        bail!(
                            "the model equation defining the 'target' of 'pac_target_info({})' contains a variable ({}) that is not declared as a 'component'",
                            m.name,
                            self.names[v as usize]
                        )
                    };
                    coeffs[j] = Some(c);
                }
                let mut yns = constant;
                for (j, c) in cs.iter().enumerate() {
                    let coeff = coeffs[j].ok_or_else(|| {
                        anyhow!(
                            "a 'component' of 'pac_target_info({})' does not appear in the model equation defining the 'target'",
                            m.name
                        )
                    })?;
                    comps.push((cnodes[j], coeff));
                    if !c.ll {
                        let p = self.add_times(coeff, cnodes[j]);
                        yns = self.add_plus(yns, p);
                    }
                }
                let s = self.add_aux(vns.clone(), 13, None, None, yns);
                let v = self.add_var(s, 0);
                let e = self.add_equal(v, yns);
                self.add_equation(e, Vec::new());
                let lagged = self.add_var(s, -1);
                self.replace_placeholder(self.placeholder_of(&format!("{PNS}{}", m.name)), lagged)?;
            }
            // ---- growth neutrality parameter
            let mut growth_term = self.zero;
            if let Some((g, p)) = &m.growth {
                let ps = self.create_param(p, &late, &mut out)?;
                let pv = self.add_var(ps, 0);
                growth_term = self.add_times(self.pre[*g], pv);
            }
            let e = match &m.kind {
                PacKind::Mce { target, max_lag, alpha, discount, auxname } => {
                    self.pac_mce(target, *max_lag, alpha, discount, auxname, growth_term, lookup, &late, &mut out)?
                }
                PacKind::Backward { is_var, lhs, max_lag, h, auxname } => {
                    let lhs: Vec<u32> = lhs.iter().map(|&i| keys[i]).collect();
                    let mut hi = h.iter();
                    let mut sub = self.zero;
                    if *is_var {
                        let p = hi.next().ok_or_else(|| anyhow!("internal error: PAC h parameters"))?;
                        let ps = self.create_param(p, &late, &mut out)?;
                        let pv = self.add_var(ps, 0);
                        sub = self.add_plus(sub, pv);
                    }
                    for i in 1..=*max_lag {
                        for &v in &lhs {
                            let p = hi.next().ok_or_else(|| anyhow!("internal error: PAC h parameters"))?;
                            let ps = self.create_param(p, &late, &mut out)?;
                            let pv = self.add_var(ps, 0);
                            let x = self.add_var(v, -(i as i32));
                            let t = self.add_times(pv, x);
                            sub = self.add_plus(sub, t);
                        }
                    }
                    sub = self.add_plus(sub, growth_term);
                    let s = self.add_aux(auxname.clone(), 12, None, None, sub);
                    let v = self.add_var(s, 0);
                    let e = self.add_equal(v, sub);
                    self.add_equation(e, Vec::new());
                    v
                }
                PacKind::Composite { is_var, lhs, max_lag, comps: cs, .. } => {
                    let lhs: Vec<u32> = lhs.iter().map(|&i| keys[i]).collect();
                    let mut sub = self.zero;
                    for (c, &(_, coeff)) in cs.iter().zip(&comps) {
                        let mut hi = c.h.iter();
                        let mut def = self.zero;
                        if *is_var {
                            let p = hi.next().ok_or_else(|| anyhow!("internal error: PAC h parameters"))?;
                            let ps = self.create_param(p, &late, &mut out)?;
                            let pv = self.add_var(ps, 0);
                            def = self.add_plus(def, pv);
                        }
                        for i in 1..=*max_lag {
                            for &v in &lhs {
                                let p = hi.next().ok_or_else(|| anyhow!("internal error: PAC h parameters"))?;
                                let ps = self.create_param(p, &late, &mut out)?;
                                let pv = self.add_var(ps, 0);
                                let x = self.add_var(v, -(i as i32));
                                let t = self.add_times(pv, x);
                                def = self.add_plus(def, t);
                            }
                        }
                        if let Some((g, p)) = &c.growth {
                            let ps = self.create_param(p, &late, &mut out)?;
                            let pv = self.add_var(ps, 0);
                            let t = self.add_times(self.pre[*g], pv);
                            def = self.add_plus(def, t);
                        }
                        let s = self.add_aux(c.auxname.clone(), 12, None, None, def);
                        let v = self.add_var(s, 0);
                        let e = self.add_equal(v, def);
                        self.add_equation(e, Vec::new());
                        let t = self.add_times(coeff, v);
                        sub = self.add_plus(sub, t);
                    }
                    sub
                }
            };
            subst.push((k, e));
        }
        // substitutePacExpectation (map ordered by model name)
        subst.sort_by(|a, b| inp.models[a.0].name.cmp(&inp.models[b.0].name));
        for (k, e) in subst {
            self.replace_placeholder(self.placeholder_of(&format!("{PE}{}", inp.models[k].name)), e)?;
        }
        for q in 0..self.equations.len() {
            let e = self.equations[q].eq;
            if self.contains_placeholder(e, &|s| s >= PH_BASE, &mut BTreeSet::new()) {
                bail!("there remains a pac_expectation or pac_target_nonstationary operator in the model");
            }
        }
        Ok(out)
    }

    /// `substitutePacExpectation` / `substitutePacTargetNonstationary`.
    fn replace_placeholder(&mut self, ph: u32, e: Id) -> Result<()> {
        let mut leaf = |_t: &mut Tree, _id: Id, s: u32, _lag: i32| -> Result<Option<Id>> {
            Ok(if s == ph { Some(e) } else { None })
        };
        let mut memo = HashMap::new();
        for q in 0..self.equations.len() {
            let eq = self.equations[q].eq;
            self.equations[q].eq = self.recurse(eq, &mut memo, &mut leaf)?;
        }
        Ok(())
    }

    /// `isConstant`: no endogenous / exogenous variable.
    fn is_constant(&mut self, id: Id) -> bool {
        let mut out = BTreeSet::new();
        self.collect_vars(id, SymType::Endo, &mut out, &mut BTreeSet::new());
        self.collect_vars(id, SymType::Exo, &mut out, &mut BTreeSet::new());
        out.is_empty()
    }

    fn additive_terms(&self, id: Id, sign: i32, out: &mut Vec<(Id, i32)>) {
        match self.node(id) {
            N::Un(UOp::Uminus, a, _) => self.additive_terms(a, -sign, out),
            N::Bin(BOp::Plus, a, b) => {
                self.additive_terms(a, sign, out);
                self.additive_terms(b, sign, out);
            }
            N::Bin(BOp::Minus, a, b) => {
                self.additive_terms(a, sign, out);
                self.additive_terms(b, -sign, out);
            }
            _ => out.push((id, sign)),
        }
    }

    /// `matchLinearCombinationOfEndogenousWithConstant`: (endogenous, coefficient) terms and
    /// the intercept.
    fn match_linear_endo_with_constant(&mut self, id: Id) -> Option<(Vec<(u32, Id)>, Id)> {
        let mut terms = Vec::new();
        self.additive_terms(id, 1, &mut terms);
        let mut endo = Vec::new();
        let mut intercept = self.zero;
        for (t, sign) in terms {
            if self.is_constant(t) {
                intercept = if sign == -1 { self.add_minus(intercept, t) } else { self.add_plus(intercept, t) };
                continue;
            }
            let (v, mut c) = match self.node(t) {
                N::Var(s, _) if self.sym_ty(s) == SymType::Endo => (s, self.one),
                N::Bin(BOp::Times, a, b) => match (self.node(a), self.node(b)) {
                    (N::Var(s, _), _) if self.sym_ty(s) == SymType::Endo && self.is_constant(b) => (s, b),
                    (_, N::Var(s, _)) if self.sym_ty(s) == SymType::Endo && self.is_constant(a) => (s, a),
                    _ => return None,
                },
                _ => return None,
            };
            if sign == -1 {
                c = self.add_uminus(c);
            }
            endo.push((v, c));
        }
        Some((endo, intercept))
    }

    /// `computePacModelConsistentExpectationSubstitution`: `mce_Z1_<m>` and its recursive
    /// definition, the `AUX_DIFF_LEAD` chain of the target; returns the substitution of
    /// `pac_expectation`.
    #[allow(clippy::too_many_arguments)]
    fn pac_mce(
        &mut self,
        target: &str,
        max_lag: usize,
        alpha: &[String],
        discount: &str,
        auxname: &str,
        growth_term: Id,
        lookup: &dyn Fn(&str) -> Option<u32>,
        late: &HashMap<&str, usize>,
        out: &mut PacOutput,
    ) -> Result<Id> {
        let ts = lookup(target).ok_or_else(|| anyhow!("Can't identify target for PAC model: unknown {target}"))?;
        // (the preprocessor's binary evaluates the arguments of the Add* calls left to right)
        let z1 = self.add_aux(auxname.to_string(), 12, None, None, 0);
        let mut a = self.one;
        let mut fp = self.zero;
        let beta = match lookup(discount) {
            Some(s) => self.add_var(s, 0),
            None => {
                let v: f64 = discount.parse().map_err(|_| anyhow!("unknown discount {discount}"))?;
                self.add_possibly_negative(v)
            }
        };
        let mut alpha_s = Vec::with_capacity(alpha.len());
        for (i, p) in alpha.iter().enumerate().take(max_lag + 1) {
            let s = self.create_param(p, late, out)?;
            alpha_s.push(s);
            let av = self.add_var(s, 0);
            a = self.add_plus(a, av);
            // AddTimes(AddTimes(alpha, AddPower(beta, i)), Z1(+i))
            let c = self.add_possibly_negative((i + 1) as f64);
            let pw = self.add_power(beta, c);
            let t1 = self.add_times(av, pw);
            let zl = self.add_var(z1, i as i32 + 1);
            let t2 = self.add_times(t1, zl);
            fp = self.add_plus(fp, t2);
        }
        // the diff of the target and its leads
        let t0 = self.add_var(ts, 0);
        let d0 = self.add_diff(t0);
        let base = match self.diff_table.get(&d0) {
            Some(&v) => v,
            None => {
                let s = self.add_aux(format!("AUX_DIFF_{d0}"), 8, None, None, 0);
                let v = self.add_var(s, 0);
                let t0 = self.add_var(ts, 0);
                let t1 = self.add_var(ts, -1);
                let m = self.add_minus(t0, t1);
                self.aux.last_mut().unwrap().rhs = m;
                let e = self.add_equal(v, m);
                self.add_equation(e, Vec::new());
                v
            }
        };
        let mut leads: Vec<Id> = Vec::new();
        let mut last = base;
        for i in 1..=max_lag {
            let ti = self.add_var(ts, i as i32);
            let d = self.add_diff(ti);
            let N::Var(ls, _) = self.node(last) else { unreachable!() };
            let s = self.add_aux(format!("AUX_DIFF_LEAD_{d}"), 11, Some(ls), Some(1), 0);
            self.aux.last_mut().unwrap().shown = Some(d);
            let v = self.add_var(s, 0);
            let l1 = self.add_var(ls, 1);
            self.aux.last_mut().unwrap().rhs = l1;
            let e = self.add_equal(v, l1);
            self.add_equation(e, Vec::new());
            last = v;
            leads.push(v);
        }
        let mut fs = self.zero;
        for k in 1..=max_lag {
            let mut ssum = self.zero;
            for j in k + 1..=max_lag + 1 {
                let av = self.add_var(alpha_s[j - 1], 0);
                let c = self.add_possibly_negative(j as f64);
                let pw = self.add_power(beta, c);
                let t = self.add_times(av, pw);
                ssum = self.add_plus(ssum, t);
            }
            let t = self.add_times(ssum, leads[k - 1]);
            fs = self.add_plus(fs, t);
        }
        let z1v = self.add_var(z1, 0);
        // recursive: not an auxiliary equation (its definition for `set_auxiliary_variables`
        // is the variable itself)
        self.aux.iter_mut().find(|r| r.symb == z1).unwrap().rhs = z1v;
        let d = self.add_minus(base, fs);
        let t = self.add_times(a, d);
        let rhs = self.add_minus(t, fp);
        let e = self.add_equal(z1v, rhs);
        self.add_equation(e, Vec::new());
        Ok(self.add_plus(z1v, growth_term))
    }
}
