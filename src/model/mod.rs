// SPDX-License-Identifier: GPL-3.0-or-later
//! The symbolic model, Dynare's `M_` (see DESIGN.md, section `model`).
//!
//! A [`ModelBuilder`] collects declarations and equations while the .mod file is
//! parsed; [`ModelBuilder::build`] then performs Dynare's transformations
//! (aux variables for leads/lags >= 2, see [`aux_subst`]), computes the
//! lead/lag incidence, takes the symbolic first derivatives of the static and
//! dynamic models and compiles everything into flat [`tape::Tape`]s.
//!
//! A [`Model`] is immutable, `Send + Sync`, and every evaluation takes `&self`
//! plus a caller-provided [`Workspace`] (the convenience methods without one use a
//! thread-local workspace).

mod aux_subst;
pub mod dtree; // stream PP: the preprocessor's DataTree and transform pass
pub mod expr;
pub mod external; // external_function: registry of functions and the .m loader
// stream K: 2nd/3rd derivatives of the dynamic model
pub mod higher;
pub mod planner; // [stream P] optimal policy
pub mod tape;
// ---- model_info: the preprocessor's block decomposition
pub mod blocks;
mod blocks_graph;

use crate::linalg::Mat;
use anyhow::{Result, bail};
use expr::{Dag, DiffVar, Node, NodeId};
use std::cell::RefCell;
use std::collections::HashMap;
use tape::Tape;

/// One entry of `M_.aux_vars`.
#[derive(Clone, Debug, PartialEq)]
pub struct AuxVar {
    /// 0-based index of the aux variable among the endogenous variables.
    pub endo_index: usize,
    /// Dynare's `AuxVarType`: 0 endo lead, 1 endo lag, 2 exo lead, 3 exo lag.
    pub ty: u8,
    /// 0-based type-specific index of the original variable (lag aux vars only).
    pub orig_index: Option<usize>,
    /// Lead/lag of the original variable that this aux var stands for (lag aux vars only).
    pub orig_lead_lag: Option<i32>,
    /// The right-hand side of the aux equation `AUX = orig_expr` (as `M_.aux_vars(k).orig_expr`).
    pub orig_expr: String,
}

/// Kind of a declared symbol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolKind {
    Endo,
    Exo,
    Param,
    /// Model-local variable (`# name = expr;`), inlined in the equations.
    Local,
    /// `trend_var` / `log_trend_var` (removed by the detrending; index among them).
    Trend,
}

/// `symb_id` of a parameter declared with [`ModelBuilder::declare_late_param`] until the
/// transform pass gives it one.
const LATE_SYMB_ID: usize = usize::MAX;

/// Entry of the symbol table. `symb_id` is Dynare's global symbol id (declaration order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Symbol {
    pub kind: SymbolKind,
    pub index: usize,
    pub symb_id: usize,
}

/// Collects declarations and equations during parsing (Dynare's `SymbolTable` + `DynamicModel`
/// before `ModFile::transformPass`).
#[derive(Clone, Debug, Default)]
pub struct ModelBuilder {
    pub dag: Dag,
    pub endo_names: Vec<String>,
    pub exo_names: Vec<String>,
    pub param_names: Vec<String>,
    /// Current parameter values (NaN = not calibrated).
    pub params: Vec<f64>,
    symbols: HashMap<String, Symbol>,
    endo_symb_id: Vec<usize>,
    exo_symb_id: Vec<usize>,
    next_symb_id: usize,
    locals: Vec<(String, NodeId)>,
    locals_used: Vec<bool>,
    /// stream PP: declared by `model_local_variable`, not defined yet.
    locals_pending: Vec<bool>,
    /// stream PP: symbols removed by `var_remove` / `model_remove` (name, symbol id): the
    /// removed equations are still replayed by `dtree` (Dynare's `excludedVariable`).
    excluded: Vec<(String, usize)>,
    /// stream PP: `differentiate_forward_vars` (None: off; empty: all variables).
    pub differentiate_forward_vars: Option<Vec<String>>,
    equations: Vec<(NodeId, NodeId)>,
    equation_tags: Vec<Vec<(String, String)>>,
    pub linear: bool,
    /// OccBin: static-model residual replacing equation i's (`static_only_equations`).
    occbin_static: Vec<Option<NodeId>>,
    /// [stream P] optimal-policy commands (`planner_objective`, `ramsey_model`, ...).
    pub planner: Option<planner::PlannerSpec>,
    // ---- stream PP
    /// The model block as parsed, replayed by [`dtree::transform`].
    pub items: Vec<dtree::ModelItem>,
    /// `predetermined_variables` (endogenous indices).
    pub predetermined: Vec<usize>,
    /// The file runs a BVAR / MS-SBVAR command (`bvar_density`, `sbvar`, ...): the
    /// preprocessor's `bvar_present`, the only case where the model may have no equation.
    pub bvar_present: bool,
    /// Why the model needs the [`dtree`] transform pass (EXPECTATION, `[static]` equations),
    /// if it does.
    pub needs_dtree: Option<String>,
    /// `trend_var` / `log_trend_var` names (`SymbolKind::Trend` index) and whether log.
    pub trend_names: Vec<(String, bool)>,
    /// `var(log)` variables (endogenous indices).
    pub log_vars: Vec<usize>,
    /// Balanced-growth test context (initval values, `balanced_growth_test_tol`).
    pub bg: dtree::BgContext,
    /// `varexo_det` variables of a stochastic model (name, value of `oo_.exo_det_steady_state`):
    /// they enter the model as parameters appended to `M_.params` (Dynare keeps them out of
    /// `M_.exo_names`; their paths only matter for forecasts, through `dr.ghud`).
    pub q1_exo_det: Vec<(String, f64)>,
    /// VAR models / PAC: what the transform pass needs (`dtree::pac`); the parameters declared
    /// with [`ModelBuilder::declare_late_param`] get their symbol id there.
    pub pac: dtree::pac::PacInput,
}

impl ModelBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// heterogeneity: Dynare's symbol ids also count the heterogeneous symbols, which are not
    /// part of this builder (they matter for the names of lag auxiliary variables).
    pub fn reserve_symbol_id(&mut self) {
        self.next_symb_id += 1;
    }

    /// Declares a symbol; fails on duplicates.
    pub fn declare(&mut self, name: &str, kind: SymbolKind) -> Result<usize> {
        if let Some(s) = self.symbols.get(name) {
            bail!("symbol '{name}' declared twice (already a {:?})", s.kind);
        }
        let index = match kind {
            SymbolKind::Endo => {
                self.endo_names.push(name.to_string());
                self.endo_symb_id.push(self.next_symb_id);
                self.endo_names.len() - 1
            }
            SymbolKind::Exo => {
                self.exo_names.push(name.to_string());
                self.exo_symb_id.push(self.next_symb_id);
                self.exo_names.len() - 1
            }
            SymbolKind::Param => {
                self.param_names.push(name.to_string());
                self.params.push(f64::NAN);
                self.param_names.len() - 1
            }
            SymbolKind::Local => bail!("use add_local for model-local variables"),
            SymbolKind::Trend => {
                // whether it is a log trend: ModelBuilder::trend_names.1
                self.trend_names.push((name.to_string(), false));
                self.trend_names.len() - 1
            }
        };
        self.symbols.insert(name.to_string(), Symbol { kind, index, symb_id: self.next_symb_id });
        self.next_symb_id += 1;
        Ok(index)
    }

    /// A parameter that the preprocessor creates in its transform pass (PAC `h`, growth
    /// neutrality and `mce_alpha` parameters): its symbol id comes after the auxiliary
    /// variables created before it (see `dtree::pac`).
    pub fn declare_late_param(&mut self, name: &str) -> Result<usize> {
        if let Some(s) = self.symbols.get(name) {
            bail!("symbol '{name}' declared twice (already a {:?})", s.kind);
        }
        self.param_names.push(name.to_string());
        self.params.push(f64::NAN);
        let index = self.param_names.len() - 1;
        self.symbols.insert(name.to_string(), Symbol { kind: SymbolKind::Param, index, symb_id: LATE_SYMB_ID });
        Ok(index)
    }

    pub fn symbol(&self, name: &str) -> Option<Symbol> {
        self.symbols.get(name).copied()
    }

    /// Adds a model-local variable `# name = value;`.
    pub fn add_local(&mut self, name: &str, value: NodeId) -> Result<()> {
        // stream PP: declared beforehand by `model_local_variable`
        if let Some(s) = self.symbols.get(name).copied()
            && s.kind == SymbolKind::Local
            && self.locals_pending.get(s.index).copied().unwrap_or(false)
        {
            self.locals[s.index].1 = value;
            self.locals_pending[s.index] = false;
            return Ok(());
        }
        if self.symbols.contains_key(name) {
            bail!("model-local variable '{name}' clashes with an existing symbol");
        }
        let index = self.locals.len();
        self.locals.push((name.to_string(), value));
        self.locals_used.push(false);
        self.symbols.insert(name.to_string(), Symbol { kind: SymbolKind::Local, index, symb_id: self.next_symb_id });
        self.next_symb_id += 1;
        Ok(())
    }

    /// The (inlined) definition of local variable `index`; marks it as used.
    pub fn use_local(&mut self, index: usize) -> NodeId {
        self.locals_used[index] = true;
        self.locals[index].1
    }

    // ---- stream PP: model_local_variable, model_remove / model_replace, var_remove ----------

    /// `model_local_variable name;`: declares the symbol (its id counts in Dynare's symbol
    /// table) before its `# name = ...` definition.
    pub fn declare_local(&mut self, name: &str) -> Result<()> {
        if let Some(s) = self.symbols.get(name) {
            if s.kind == SymbolKind::Local {
                return Ok(());
            }
            bail!("symbol '{name}' declared twice (already a {:?})", s.kind);
        }
        let index = self.locals.len();
        self.locals.push((name.to_string(), Dag::ZERO));
        self.locals_used.push(false);
        self.locals_pending.resize(index, false);
        self.locals_pending.push(true);
        self.symbols.insert(name.to_string(), Symbol { kind: SymbolKind::Local, index, symb_id: self.next_symb_id });
        self.next_symb_id += 1;
        Ok(())
    }

    /// Port of `DynamicModel::removeEquations` (`model_remove`, `model_replace`): drops the
    /// equations whose tags include one of `selection`'s tag sets. With `exclude_vars`
    /// (`model_remove`), returns the endogenous variables to exclude (`endogenous` tag, or the
    /// only endogenous variable of the left-hand side).
    pub fn remove_equations(&mut self, selection: &[Vec<(String, String)>], exclude_vars: bool) -> Result<Vec<String>> {
        let matches = |tags: &[(String, String)], sel: &[(String, String)]| {
            sel.iter().all(|(k, v)| tags.iter().any(|(a, b)| a == k && b == v))
        };
        // the [static] equations are in `items` only
        let item_tags: Vec<&[(String, String)]> = self
            .items
            .iter()
            .filter_map(|it| match it {
                dtree::ModelItem::Equation { tags, .. } => Some(tags.as_slice()),
                _ => None,
            })
            .collect();
        for sel in selection {
            let found = self.equation_tags.iter().any(|t| matches(t, sel)) || item_tags.iter().any(|t| matches(t, sel));
            if !found {
                let d: Vec<String> = sel.iter().map(|(k, v)| format!("{k}={v}")).collect();
                bail!("model_remove/model_replace: the equations specified by [{}] were not found", d.join(", "));
            }
        }
        let mut excluded = Vec::new();
        let mut keep_eq = Vec::new();
        for (i, t) in self.equation_tags.iter().enumerate() {
            let del = selection.iter().any(|sel| matches(t, sel));
            if del && exclude_vars {
                if let Some((_, v)) = t.iter().find(|(k, _)| k == "endogenous") {
                    excluded.push(v.clone());
                } else {
                    let mut leaves = Vec::new();
                    self.dag.collect_leaves(self.equations[i].0, &mut leaves);
                    let mut endo: Vec<u32> =
                        leaves.iter().filter_map(|l| if let Node::Endo(j, _) = l { Some(*j) } else { None }).collect();
                    endo.sort_unstable();
                    endo.dedup();
                    if let [j] = endo[..] {
                        excluded.push(self.endo_names[j as usize].clone());
                    }
                }
            }
            keep_eq.push(!del);
        }
        let mut k = 0;
        self.equations.retain(|_| {
            k += 1;
            keep_eq[k - 1]
        });
        let mut k = 0;
        self.equation_tags.retain(|_| {
            k += 1;
            keep_eq[k - 1]
        });
        if !self.occbin_static.is_empty() {
            self.occbin_static.resize(keep_eq.len(), None);
            let mut k = 0;
            self.occbin_static.retain(|_| {
                k += 1;
                keep_eq[k - 1]
            });
        }
        self.items.push(dtree::ModelItem::Remove { selection: selection.to_vec() });
        Ok(excluded)
    }

    /// `var_remove` (and the variables excluded by `model_remove`): removes a declared
    /// endogenous/exogenous variable or parameter. Later symbols of the same kind move down
    /// one index (Dynare's type-specific ids skip excluded variables); `remap(kind, old)` is
    /// how the caller updates the indices it holds. Fails if the symbol is still used.
    pub fn remove_symbol(&mut self, name: &str) -> Result<(SymbolKind, usize)> {
        let Some(sym) = self.symbols.get(name).copied() else { bail!("var_remove: unknown symbol '{name}'") };
        let j = sym.index as u32;
        let mut used = false;
        let mut memo = HashMap::new();
        let mut eqs = std::mem::take(&mut self.equations);
        let mut locs = std::mem::take(&mut self.locals);
        let mut statics = std::mem::take(&mut self.occbin_static);
        {
            let mut f = |d: &mut Dag, n: Node| -> Option<NodeId> {
                let shift = |k: u32| if k > j { Some(k - 1) } else { None };
                match (sym.kind, n) {
                    (SymbolKind::Endo, Node::Endo(k, l)) => {
                        used |= k == j;
                        shift(k).map(|k| d.endo(k as usize, l))
                    }
                    (SymbolKind::Endo, Node::SteadyState(k, l)) => {
                        used |= k == j;
                        shift(k).map(|k| {
                            let e = d.endo(k as usize, l);
                            d.steady_state(e)
                        })
                    }
                    (SymbolKind::Exo, Node::Exo(k, l)) => {
                        used |= k == j;
                        shift(k).map(|k| d.exo(k as usize, l))
                    }
                    (SymbolKind::Param, Node::Param(k)) => {
                        used |= k == j;
                        shift(k).map(|k| d.param(k as usize))
                    }
                    _ => None,
                }
            };
            for e in eqs.iter_mut() {
                e.0 = self.dag.map_vars(e.0, &mut memo, &mut f);
                e.1 = self.dag.map_vars(e.1, &mut memo, &mut f);
            }
            for l in locs.iter_mut() {
                l.1 = self.dag.map_vars(l.1, &mut memo, &mut f);
            }
            for v in statics.iter_mut().flatten() {
                *v = self.dag.map_vars(*v, &mut memo, &mut f);
            }
        }
        self.equations = eqs;
        self.locals = locs;
        self.occbin_static = statics;
        if used {
            bail!("'{name}' cannot be removed: it is still used in the model");
        }
        let idx = sym.index;
        match sym.kind {
            SymbolKind::Endo => {
                self.endo_names.remove(idx);
                self.endo_symb_id.remove(idx);
                self.predetermined.retain(|&k| k != idx);
                for k in self.predetermined.iter_mut().filter(|k| **k > idx) {
                    *k -= 1;
                }
            }
            SymbolKind::Exo => {
                self.exo_names.remove(idx);
                self.exo_symb_id.remove(idx);
            }
            SymbolKind::Param => {
                self.param_names.remove(idx);
                self.params.remove(idx);
            }
            SymbolKind::Local => bail!("var_remove: '{name}' is a model-local variable"),
            SymbolKind::Trend => bail!("var_remove: '{name}' is a trend variable"),
        }
        self.symbols.remove(name);
        self.excluded.push((name.to_string(), sym.symb_id));
        for s in self.symbols.values_mut() {
            if s.kind == sym.kind && s.index > idx {
                s.index -= 1;
            }
        }
        Ok((sym.kind, idx))
    }
    // ---- end stream PP ------------------------------------------------------------------------

    // ---- exclude_eqs / include_eqs (DynamicModel::includeExcludeEquations) --------
    /// The tags of the equations parsed so far (model-block order).
    pub fn equation_tags(&self) -> &[Vec<(String, String)>] {
        &self.equation_tags
    }

    /// Gives every untagged equation (and its model item) the tag `tag`, so that `include_eqs`
    /// can remove them by tag.
    pub fn tag_untagged_equations(&mut self, tag: &(String, String)) {
        for t in self.equation_tags.iter_mut().filter(|t| t.is_empty()) {
            t.push(tag.clone());
        }
        for it in &mut self.items {
            if let dtree::ModelItem::Equation { tags, .. } = it
                && tags.is_empty()
            {
                tags.push(tag.clone());
            }
        }
    }

    /// Whether a symbol appears in the equations or model-local variables parsed so far.
    pub fn is_used(&self, name: &str) -> bool {
        let Some(sym) = self.symbols.get(name).copied() else { return false };
        let j = sym.index as u32;
        let mut leaves = Vec::new();
        for (l, r) in &self.equations {
            self.dag.collect_leaves(*l, &mut leaves);
            self.dag.collect_leaves(*r, &mut leaves);
        }
        for (_, v) in &self.locals {
            self.dag.collect_leaves(*v, &mut leaves);
        }
        leaves.iter().any(|n| match (sym.kind, n) {
            (SymbolKind::Endo, Node::Endo(k, _) | Node::SteadyState(k, _)) => *k == j,
            (SymbolKind::Exo, Node::Exo(k, _)) => *k == j,
            (SymbolKind::Param, Node::Param(k)) => *k == j,
            _ => false,
        })
    }

    /// The smallest lag (<= 0) of an endogenous or exogenous variable in the equations parsed so
    /// far (the preprocessor's `DynamicModel::minLagForSymbol`).
    pub fn min_lag(&self, name: &str) -> i32 {
        let Some(sym) = self.symbols.get(name).copied() else { return 0 };
        let j = sym.index as u32;
        let mut leaves = Vec::new();
        for (l, r) in &self.equations {
            self.dag.collect_leaves(*l, &mut leaves);
            self.dag.collect_leaves(*r, &mut leaves);
        }
        for (_, v) in &self.locals {
            self.dag.collect_leaves(*v, &mut leaves);
        }
        leaves
            .iter()
            .filter_map(|n| match (sym.kind, n) {
                (SymbolKind::Endo, Node::Endo(k, l)) | (SymbolKind::Exo, Node::Exo(k, l)) if *k == j => Some(*l),
                _ => None,
            })
            .fold(0, i32::min)
    }

    /// `varexo_det`: moves the exogenous variable `from` to index `to <= from` (the ones in
    /// between shift up), in the names, the symbol table and the equations parsed so far, so
    /// that the `varexo_det` variables stay after the ordinary ones (Dynare numbers the two
    /// kinds separately).
    pub fn move_exo(&mut self, from: usize, to: usize) {
        if from == to {
            return;
        }
        let (f, t) = (from as u32, to as u32);
        let mut memo = HashMap::new();
        let mut eqs = std::mem::take(&mut self.equations);
        let mut locs = std::mem::take(&mut self.locals);
        let mut statics = std::mem::take(&mut self.occbin_static);
        {
            let mut map = |d: &mut Dag, n: Node| -> Option<NodeId> {
                match n {
                    Node::Exo(k, l) if k == f => Some(d.exo(to, l)),
                    Node::Exo(k, l) if k >= t && k < f => Some(d.exo(k as usize + 1, l)),
                    _ => None,
                }
            };
            for e in eqs.iter_mut() {
                e.0 = self.dag.map_vars(e.0, &mut memo, &mut map);
                e.1 = self.dag.map_vars(e.1, &mut memo, &mut map);
            }
            for l in locs.iter_mut() {
                l.1 = self.dag.map_vars(l.1, &mut memo, &mut map);
            }
            for v in statics.iter_mut().flatten() {
                *v = self.dag.map_vars(*v, &mut memo, &mut map);
            }
        }
        self.equations = eqs;
        self.locals = locs;
        self.occbin_static = statics;
        let name = self.exo_names.remove(from);
        let id = self.exo_symb_id.remove(from);
        self.exo_names.insert(to, name);
        self.exo_symb_id.insert(to, id);
        for s in self.symbols.values_mut() {
            if s.kind == SymbolKind::Exo {
                if s.index == from {
                    s.index = to;
                } else if s.index >= to && s.index < from {
                    s.index += 1;
                }
            }
        }
    }

    /// `symbol_table.changeType(symb_id, exogenous)`: an endogenous variable becomes exogenous.
    /// Type-specific ids follow the symbol ids, so it takes the place among the exogenous
    /// variables that its symbol id gives it. Returns (old endogenous index, new exogenous index).
    pub fn endo_to_exo(&mut self, name: &str) -> Result<(usize, usize)> {
        let Some(sym) = self.symbols.get(name).copied() else { bail!("unknown symbol '{name}'") };
        if sym.kind != SymbolKind::Endo {
            bail!("'{name}' is not an endogenous variable");
        }
        let j = sym.index;
        let p = self.exo_symb_id.iter().position(|&s| s > sym.symb_id).unwrap_or(self.exo_symb_id.len());
        let (ju, pu) = (j as u32, p as u32);
        let mut memo = HashMap::new();
        let mut eqs = std::mem::take(&mut self.equations);
        let mut locs = std::mem::take(&mut self.locals);
        let mut statics = std::mem::take(&mut self.occbin_static);
        let mut bad = false;
        {
            let mut f = |d: &mut Dag, n: Node| -> Option<NodeId> {
                match n {
                    Node::Endo(k, l) if k == ju => Some(d.exo(p, l)),
                    Node::Endo(k, l) if k > ju => Some(d.endo(k as usize - 1, l)),
                    Node::SteadyState(k, _) if k == ju => {
                        bad = true;
                        None
                    }
                    Node::SteadyState(k, l) if k > ju => {
                        let e = d.endo(k as usize - 1, l);
                        Some(d.steady_state(e))
                    }
                    Node::Exo(k, l) if k >= pu => Some(d.exo(k as usize + 1, l)),
                    _ => None,
                }
            };
            for e in eqs.iter_mut() {
                e.0 = self.dag.map_vars(e.0, &mut memo, &mut f);
                e.1 = self.dag.map_vars(e.1, &mut memo, &mut f);
            }
            for l in locs.iter_mut() {
                l.1 = self.dag.map_vars(l.1, &mut memo, &mut f);
            }
            for v in statics.iter_mut().flatten() {
                *v = self.dag.map_vars(*v, &mut memo, &mut f);
            }
        }
        self.equations = eqs;
        self.locals = locs;
        self.occbin_static = statics;
        if bad {
            bail!("'{name}' is used in STEADY_STATE() and cannot become exogenous");
        }
        self.endo_names.remove(j);
        self.endo_symb_id.remove(j);
        self.exo_names.insert(p, name.to_string());
        self.exo_symb_id.insert(p, sym.symb_id);
        self.predetermined.retain(|&k| k != j);
        for k in self.predetermined.iter_mut().filter(|k| **k > j) {
            *k -= 1;
        }
        for s in self.symbols.values_mut() {
            if s.kind == SymbolKind::Endo && s.index > j {
                s.index -= 1;
            } else if s.kind == SymbolKind::Exo && s.index >= p {
                s.index += 1;
            }
        }
        self.symbols.insert(name.to_string(), Symbol { kind: SymbolKind::Exo, index: p, symb_id: sym.symb_id });
        Ok((j, p))
    }

    pub fn add_equation(&mut self, lhs: NodeId, rhs: NodeId, tags: Vec<(String, String)>) {
        self.equations.push((lhs, rhs));
        self.equation_tags.push(tags);
    }

    pub fn n_equations(&self) -> usize {
        self.equations.len()
    }

    // ---- OccBin (stream: occbin) -------------------------------------------------------
    /// Port of the preprocessor's `DynamicModel::addOccbinEquation`: an equation tagged
    /// `[name='..', bind='c1[,c2]', relax='..']`. Its residual `lhs - rhs` is multiplied by
    /// `occbin_<c>_bind` for each `bind` constraint and by `1 - occbin_<c>_bind` for each
    /// `relax` one (the parameters are declared on first use, value 0) and added to the
    /// equation of the same name (created at the first snippet, without bind/relax tags).
    /// Snippets without `bind` also form the static model's equation (the relaxed regime).
    pub fn add_occbin_equation(&mut self, lhs: NodeId, rhs: NodeId, mut tags: Vec<(String, String)>) -> Result<()> {
        let get = |k: &str| tags.iter().find(|(t, _)| t == k).map(|(_, v)| v.clone());
        let split = |v: Option<String>| -> Vec<String> {
            v.map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
                .unwrap_or_default()
        };
        let (bind, relax) = (split(get("bind")), split(get("relax")));
        let Some(name) = get("name") else { bail!("an equation with a bind/relax tag needs a 'name' tag") };
        let basic = self.dag.sub(lhs, rhs);
        let mut term = basic;
        for (c, is_bind) in bind.iter().map(|c| (c, true)).chain(relax.iter().map(|c| (c, false))) {
            let pname = format!("occbin_{c}_bind");
            let k = match self.symbols.get(&pname) {
                Some(s) if s.kind == SymbolKind::Param => s.index,
                Some(_) => bail!("'{pname}' is already declared and is not a parameter"),
                None => {
                    let k = self.declare(&pname, SymbolKind::Param)?;
                    self.params[k] = 0.0;
                    k
                }
            };
            let p = self.dag.param(k);
            let f = if is_bind { p } else { self.dag.sub(Dag::ONE, p) };
            term = self.dag.mul(term, f);
        }
        let found = self.equation_tags.iter().position(|t| t.iter().any(|(k, v)| k == "name" && *v == name));
        let eq = match found {
            Some(k) => {
                let (l, r) = self.equations[k];
                let d = self.dag.sub(l, r);
                self.equations[k] = (self.dag.add(d, term), Dag::ZERO);
                k
            }
            None => {
                tags.retain(|(k, _)| k != "bind" && k != "relax");
                self.equations.push((term, Dag::ZERO));
                self.equation_tags.push(tags);
                self.equations.len() - 1
            }
        };
        if bind.is_empty() {
            self.occbin_static.resize(self.equations.len(), None);
            self.occbin_static[eq] = Some(match self.occbin_static[eq] {
                Some(prev) => self.dag.add(prev, basic),
                None => basic,
            });
        }
        Ok(())
    }
    // ---- end OccBin ----------------------------------------------------------------------

    /// Runs Dynare's transform pass and compiles the model. `stochastic` selects the
    /// stochastic-model substitution rules (true when the file has `stoch_simul`,
    /// `estimation`, ...), as in `ModFile::transformPass`.
    pub fn build(self, stochastic: bool) -> Result<Model> {
        let ModelBuilder {
            items,
            excluded,
            differentiate_forward_vars,
            predetermined,
            needs_dtree,
            bvar_present,
            trend_names,
            log_vars,
            mut bg,
            q1_exo_det,
            mut pac,
            mut dag,
            mut endo_names,
            exo_names,
            mut param_names,
            mut params,
            mut symbols,
            mut endo_symb_id,
            exo_symb_id,
            mut next_symb_id,
            locals,
            locals_used,
            mut equations,
            mut equation_tags,
            linear,
            occbin_static,
            planner: planner_spec,
            ..
        } = self;
        let orig_endo_nbr = endo_names.len();
        // ---- [stream P] optimal policy: Ramsey FOCs, multipliers, discount parameter ----
        let n_eq_written = equations.len();
        let mut planner_pre = planner::prepare(
            planner::Builder {
                dag: &mut dag,
                endo_names: &mut endo_names,
                next_symb_id: &mut next_symb_id,
                param_names: &mut param_names,
                params: &mut params,
                symbols: &mut symbols,
                equations: &mut equations,
            },
            planner_spec,
        )?;
        let eq_deficit = planner_pre.as_ref().map_or(0, |p| p.eq_deficit);
        // ---- [end stream P] ----
        // ---- varexo_det of a stochastic model -> parameters (see `q1_exo_det`)
        let mut exo_det_params = Vec::with_capacity(q1_exo_det.len());
        for (name, v) in &q1_exo_det {
            let k = param_names.len();
            param_names.push(name.clone());
            params.push(*v);
            if let Some(s) = symbols.get_mut(name) {
                s.kind = SymbolKind::Param;
                s.index = k;
            }
            exo_det_params.push(k);
        }
        let mut aux_vars = Vec::new();
        let mut pac_key_names = Vec::new();
        let mut aux_rhs = Vec::new();
        let (orig_max_lead, orig_max_lag, orig_max_lag_with_diffs, orig_eq_nbr, static_override);
        let static_eqs: Vec<Option<(NodeId, NodeId)>>; // model_info
        // ---- stream PP: the preprocessor's transform pass (exact aux names and equations);
        // Ramsey models: the FOCs are built inside the transform pass (dtree_ramsey.rs).
        if !items.is_empty() {
            let mut syms = vec![dtree::Sym { ty: dtree::SymType::Param, index: 0 }; next_symb_id];
            let mut names = vec![String::new(); next_symb_id];
            pac.late_params =
                symbols.iter().filter(|(_, s)| s.symb_id == LATE_SYMB_ID).map(|(n, s)| (n.clone(), s.index)).collect();
            for (name, sy) in symbols.iter().filter(|(_, s)| s.symb_id != LATE_SYMB_ID) {
                let ty = match sy.kind {
                    SymbolKind::Endo => dtree::SymType::Endo,
                    SymbolKind::Exo => dtree::SymType::Exo,
                    SymbolKind::Param => dtree::SymType::Param,
                    SymbolKind::Local => dtree::SymType::Local,
                    SymbolKind::Trend if trend_names[sy.index].1 => dtree::SymType::LogTrend,
                    SymbolKind::Trend => dtree::SymType::Trend,
                };
                syms[sy.symb_id] = dtree::Sym { ty, index: sy.index };
                names[sy.symb_id] = name.clone();
            }
            for (name, id) in &excluded {
                syms[*id] = dtree::Sym { ty: dtree::SymType::Excluded, index: 0 };
                names[*id] = name.clone();
            }
            let lookup = |name: &str| {
                symbols
                    .get(name)
                    .filter(|s| s.symb_id != LATE_SYMB_ID)
                    .map(|s| s.symb_id as u32)
                    .or_else(|| excluded.iter().rev().find(|(n, _)| n == name).map(|x| x.1 as u32))
            };
            let n_endo_before_aux = endo_names.len();
            let tr = dtree::transform(
                dtree::Input {
                    items: &items,
                    syms,
                    names,
                    n_endo: endo_names.len(),
                    lookup: &lookup,
                    predetermined: predetermined.iter().map(|&j| endo_symb_id[j] as u32).collect(),
                    stochastic,
                    differentiate_forward_vars: differentiate_forward_vars.clone(),
                    log_vars: log_vars.iter().map(|&j| endo_symb_id[j] as u32).collect(),
                    bg: {
                        bg.params = params.clone();
                        if bg.tol == 0.0 {
                            bg.tol = 1e-6;
                        }
                        bg.clone()
                    },
                    pac: &pac,
                    planner: planner_pre.as_ref().and_then(|p| {
                        p.tree_input(&endo_symb_id, symbols.get("optimal_policy_discount_factor")?.symb_id)
                    }),
                },
                &mut dag,
            )?;
            if let Some(p) = planner_pre.as_mut() {
                let mut memo = HashMap::new();
                p.set_static_focs(tr.ramsey_focs.iter().map(|&f| dag.to_static(f, &mut memo)).collect());
            }
            equations = tr.equations;
            equation_tags = tr.tags;
            static_override = tr.static_override;
            static_eqs = tr.static_eqs; // model_info
            for (name, s) in &tr.pac.param_symb {
                if let Some(sy) = symbols.get_mut(name) {
                    sy.symb_id = *s as usize;
                }
            }
            pac_key_names = tr.pac.key_names.clone();
            // the Ramsey multipliers (aux type 6) are endogenous variables without aux equation,
            // kept in the planner information rather than in `aux_vars`
            let mut mult_index = Vec::new();
            let mut shown = Vec::with_capacity(tr.shown.len());
            for ((name, a, rhs), sh) in tr.aux.iter().cloned().zip(tr.shown.iter().cloned()) {
                endo_symb_id.push(next_symb_id);
                if a.ty == 6 {
                    let j = endo_names.len();
                    symbols.insert(name.clone(), Symbol { kind: SymbolKind::Endo, index: j, symb_id: next_symb_id });
                    mult_index.push(j);
                } else {
                    aux_vars.push(a);
                    aux_rhs.push(rhs);
                    shown.push(sh);
                }
                next_symb_id += 1;
                endo_names.push(name);
            }
            let n_mult = mult_index.len();
            orig_eq_nbr = equations.len() - (tr.aux.len() - n_mult);
            if let (Some(p), Some(r)) = (planner_pre.as_mut(), tr.ramsey.as_ref()) {
                p.set_ramsey(r.orig_endo, r.orig_eq, mult_index);
            }
            aux_subst::name_exprs(&dag, &endo_names, &exo_names, Some(&param_names), &mut aux_vars, &aux_rhs);
            let first = aux_vars.len() - shown.len();
            for (k, sh) in shown.iter().enumerate() {
                if let Some((f, arg)) = sh {
                    let (pre, post) = (format!("{f}("), ")");
                    let mut one = [aux_vars[first + k].clone()];
                    aux_subst::name_exprs(&dag, &endo_names, &exo_names, Some(&param_names), &mut one, &[*arg]);
                    aux_vars[first + k].orig_expr = format!("{pre}{}{post}", one[0].orig_expr);
                }
            }
            orig_max_lead = tr.orig_max_lead.max(0);
            orig_max_lag = tr.orig_max_lag.max(0);
            orig_max_lag_with_diffs = tr.orig_max_lag_with_diffs.max(0);
            if orig_eq_nbr + eq_deficit != n_endo_before_aux + n_mult {
                bail!("the model has {orig_eq_nbr} equations for {orig_endo_nbr} endogenous variables");
            }
        } else {
            if !trend_names.is_empty() || !log_vars.is_empty() {
                bail!("var(log), deflators and trend variables are not supported together with optimal policy");
            }
            if let Some(why) = &needs_dtree {
                bail!("{why} is not supported together with optimal policy");
            }
            if planner_pre.as_ref().is_some_and(|p| p.is_ramsey()) {
                bail!("ramsey_model needs a model block");
            }
            orig_eq_nbr = equations.len();
            if n_eq_written == 0 && !endo_names.is_empty() && !bvar_present {
                // ModFile::checkPass: zero equations only for BVAR files
                bail!("At least one model equation must be declared!");
            }
            if n_eq_written > 0 && orig_eq_nbr + eq_deficit != endo_names.len() {
                bail!("the model has {n_eq_written} equations for {orig_endo_nbr} endogenous variables");
            }
            static_eqs = occbin_static.iter().map(|o| o.map(|r| (r, Dag::ZERO))).collect(); // model_info
            static_override = occbin_static;
            let (mut olead, mut olag) = (0, 0);
            for &(l, r) in &equations {
                for n in [l, r] {
                    olead = olead.max(dag.max_endo_lead(n)).max(dag.max_exo_lead(n));
                    olag = olag.max(dag.max_endo_lag(n)).max(dag.max_exo_lag(n));
                }
            }

            let used_locals: Vec<NodeId> =
                locals.iter().zip(&locals_used).filter(|(_, u)| **u).map(|((_, n), _)| *n).collect();
            {
                let mut s = aux_subst::Subst {
                    dag: &mut dag,
                    endo_names: &mut endo_names,
                    exo_names: &exo_names,
                    endo_symb_id: &endo_symb_id,
                    exo_symb_id: &exo_symb_id,
                    aux_vars: &mut aux_vars,
                    aux_rhs: &mut aux_rhs,
                    deterministic: !stochastic,
                };
                // Order of ModFile::transformPass. Local variables are already inlined, so their
                // substituted values only prime the substitution table (same visit order as Dynare).
                let passes: &[aux_subst::Pass] = if stochastic {
                    &[
                        aux_subst::Pass::EndoLead,
                        aux_subst::Pass::ExoLead,
                        aux_subst::Pass::EndoLag,
                        aux_subst::Pass::ExoLag,
                    ]
                } else {
                    // Dynare 7 also substitutes exogenous leads/lags in deterministic models (checked
                    // against the 7.1 preprocessor), only leaving endogenous leads/lags of 1 (stream D).
                    &[
                        aux_subst::Pass::EndoLead,
                        aux_subst::Pass::ExoLead,
                        aux_subst::Pass::EndoLag,
                        aux_subst::Pass::ExoLag,
                    ]
                };
                for &p in passes {
                    s.run(p, &used_locals, &mut equations);
                }
                s.name_exprs();
            }
            orig_max_lead = olead;
            orig_max_lag = olag;
            orig_max_lag_with_diffs = olag;
        }
        equation_tags.resize(equations.len(), Vec::new());
        // ---- complementarity conditions (`eq ⟂ x > lb;`, or the obsolete `mcp` tag)
        let mut mcp = Vec::new();
        for (i, tags) in equation_tags.iter_mut().enumerate() {
            if let Some(k) = tags.iter().position(|(t, _)| t == MCP_TAG) {
                mcp.push((i, tags.remove(k).1));
            } else if let Some((_, v)) = tags.iter().find(|(t, _)| t == "mcp") {
                mcp.push((i, v.clone()));
            }
        }
        let n = endo_names.len();
        let n_exo = exo_names.len();
        if orig_eq_nbr > 0 && equations.len() + eq_deficit != n {
            bail!("internal error: {} equations for {} endogenous after aux substitution", equations.len(), n);
        }

        // residuals lhs - rhs, and their static versions
        let resid: Vec<NodeId> = equations.iter().map(|&(l, r)| dag.sub(l, r)).collect();
        let mut static_memo = HashMap::new();
        let static_resid: Vec<NodeId> = resid
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                // [static] equations / OccBin relaxed regime (DynamicModel's static_only_equations)
                let r = static_override.get(i).copied().flatten().unwrap_or(r);
                dag.to_static(r, &mut static_memo)
            })
            .collect();

        // incidence and dynamic Jacobian pattern
        let mut present = vec![[false; 3]; n];
        let mut diff_memo: HashMap<DiffVar, HashMap<NodeId, NodeId>> = HashMap::new();
        let mut dyn_entries: Vec<(usize, usize, NodeId)> = Vec::new();
        let mut leaves = Vec::new();
        for (i, &r) in resid.iter().enumerate() {
            leaves.clear();
            dag.collect_leaves(r, &mut leaves);
            for &leaf in &leaves {
                let (v, col) = match leaf {
                    Node::Endo(j, l) => {
                        if !(-1..=1).contains(&l) {
                            bail!("internal error: endogenous lead/lag {l} left after substitution");
                        }
                        present[j as usize][(l + 1) as usize] = true;
                        (DiffVar::Endo(j, l), (l + 1) as usize * n + j as usize)
                    }
                    Node::Exo(k, l) => {
                        if l != 0 {
                            bail!(
                                "exogenous variable '{}' with lead/lag {l} is not supported in a deterministic model",
                                exo_names[k as usize]
                            );
                        }
                        (DiffVar::Exo(k, 0), 3 * n + k as usize)
                    }
                    // stream PP: variables under STEADY_STATE() count in the incidence (Dynare's
                    // deriv ids), their derivatives are zero
                    Node::SteadyState(j, l) => {
                        if (-1..=1).contains(&l) {
                            present[j as usize][(l + 1) as usize] = true;
                        }
                        continue;
                    }
                    _ => continue,
                };
                let d = dag.diff(r, v, diff_memo.entry(v).or_default());
                if d != Dag::ZERO {
                    dyn_entries.push((i, col, d));
                }
            }
        }
        dyn_entries.sort_by_key(|&(i, c, _)| (c, i));
        let mut lead_lag_incidence = vec![[0usize; 3]; n];
        let mut counter = 0;
        for t in 0..3 {
            for j in 0..n {
                if present[j][t] {
                    counter += 1;
                    lead_lag_incidence[j][t] = counter;
                }
            }
        }

        // static Jacobian
        let mut static_entries: Vec<(usize, usize, NodeId)> = Vec::new();
        for (i, &r) in static_resid.iter().enumerate() {
            leaves.clear();
            dag.collect_leaves(r, &mut leaves);
            for &leaf in &leaves {
                if let Node::Endo(j, _) = leaf {
                    let v = DiffVar::Endo(j, 0);
                    let d = dag.diff(r, v, diff_memo.entry(v).or_default());
                    if d != Dag::ZERO {
                        static_entries.push((i, j as usize, d));
                    }
                }
            }
        }
        static_entries.sort_by_key(|&(i, c, _)| (c, i));

        // set_auxiliary_variables: static aux definitions with earlier aux vars expanded
        let mut expanded: HashMap<u32, NodeId> = HashMap::new();
        let mut aux_defs = Vec::with_capacity(aux_rhs.len());
        for (a, &rhs) in aux_vars.iter().zip(&aux_rhs) {
            let mut memo = HashMap::new();
            let st = dag.to_static(rhs, &mut memo);
            let mut memo = HashMap::new();
            let e = dag.map_vars(st, &mut memo, &mut |_, nd| match nd {
                Node::Endo(j, _) => expanded.get(&j).copied(),
                _ => None,
            });
            expanded.insert(a.endo_index as u32, e);
            aux_defs.push(e);
        }

        // stream PP: STEADY_STATE(y_j) is read from the extended parameters [params; ys]
        dag.ss_offset = param_names.len() as u32;
        dag.ss_n_endo = n as u32;
        dag.ss_param_nbr = (param_names.len() - exo_det_params.len()) as u32;
        let has_ss = resid.iter().any(|&r| dag.has_steady_state(r));
        let dyn_y = move |j: u32, l: i32| (l + 1) as usize * n + j as usize;
        let x0 = |k: u32, _l: i32| k as usize;
        let st_y = |j: u32, _l: i32| j as usize;
        let dyn_resid_tape = Tape::compile(&dag, &resid, &dyn_y, &x0);
        let dyn_jac_nodes: Vec<NodeId> = dyn_entries.iter().map(|e| e.2).collect();
        let dyn_jac_tape = Tape::compile(&dag, &dyn_jac_nodes, &dyn_y, &x0);
        let static_resid_tape = Tape::compile(&dag, &static_resid, &st_y, &x0);
        let st_jac_nodes: Vec<NodeId> = static_entries.iter().map(|e| e.2).collect();
        let static_jac_tape = Tape::compile(&dag, &st_jac_nodes, &st_y, &x0);
        let aux_tape = Tape::compile(&dag, &aux_defs, &st_y, &x0);

        let scratch_len = [&dyn_resid_tape, &dyn_jac_tape, &static_resid_tape, &static_jac_tape, &aux_tape]
            .iter()
            .map(|t| t.scratch_len())
            .max()
            .unwrap_or(0);

        let planner = planner_pre.map(|p| p.finish(&mut dag)); // [stream P]
        Ok(Model {
            endo_names,
            orig_endo_nbr,
            pac_key_names,
            exo_names,
            param_names,
            params,
            aux_vars,
            lead_lag_incidence,
            equation_tags,
            orig_eq_nbr,
            orig_maximum_lead: orig_max_lead as usize,
            orig_maximum_lag: orig_max_lag as usize,
            orig_maximum_lag_with_diffs_expanded: orig_max_lag_with_diffs as usize,
            linear,
            n_exo,
            symbols,
            dag,
            equations,
            resid,
            dyn_pattern: dyn_entries.iter().map(|e| (e.0, e.1)).collect(),
            static_pattern: static_entries.iter().map(|e| (e.0, e.1)).collect(),
            dyn_resid_tape,
            dyn_jac_tape,
            static_resid_tape,
            static_jac_tape,
            aux_tape,
            scratch_len,
            has_ss,
            planner,
            mcp,
            exo_det_names: q1_exo_det.into_iter().map(|x| x.0).collect(),
            exo_det_params,
            static_eqs, // model_info
            block_solve: None,
            bytecode: false,
        })
    }
}

/// Dynare's `M_` plus compiled residual/Jacobian evaluators.
#[derive(Clone, Debug)]
pub struct Model {
    /// Declared endogenous variables, then aux variables (as `M_.endo_names`).
    pub endo_names: Vec<String>,
    pub orig_endo_nbr: usize,
    pub exo_names: Vec<String>,
    pub param_names: Vec<String>,
    /// Calibration (`M_.params`); NaN if never set.
    pub params: Vec<f64>,
    pub aux_vars: Vec<AuxVar>,
    /// VAR models / PAC: the variables that stand for `ModelBuilder::pac.keys`.
    pub pac_key_names: Vec<String>,
    /// Per endogenous variable `[lag, current, lead]`: 1-based column in the compressed dynamic
    /// Jacobian, 0 = absent (`M_.lead_lag_incidence'`).
    pub lead_lag_incidence: Vec<[usize; 3]>,
    /// Equation tags (`[name='...']`), one list per equation (empty for aux equations).
    pub equation_tags: Vec<Vec<(String, String)>>,
    /// Number of equations written in the model block (aux equations follow them).
    pub orig_eq_nbr: usize,
    /// `M_.orig_maximum_lead` / `M_.orig_maximum_lag` (before aux substitution).
    pub orig_maximum_lead: usize,
    pub orig_maximum_lag: usize,
    /// `M_.orig_maximum_lag_with_diffs_expanded` (each `diff` adds a lag).
    pub orig_maximum_lag_with_diffs_expanded: usize,
    /// `model(linear)`.
    pub linear: bool,
    n_exo: usize,
    symbols: HashMap<String, Symbol>,
    dag: Dag,
    equations: Vec<(NodeId, NodeId)>,
    resid: Vec<NodeId>,
    dyn_pattern: Vec<(usize, usize)>,
    static_pattern: Vec<(usize, usize)>,
    dyn_resid_tape: Tape,
    dyn_jac_tape: Tape,
    static_resid_tape: Tape,
    static_jac_tape: Tape,
    aux_tape: Tape,
    scratch_len: usize,
    /// stream PP: the dynamic model has `STEADY_STATE()` terms (tapes read `[params; ys]`).
    has_ss: bool,
    /// [stream P] optimal-policy structure, see [`planner`].
    planner: Option<planner::PlannerInfo>,
    /// complementarity conditions (equation index, condition such as `i > 0`), see
    /// `deterministic::mcp`.
    pub mcp: Vec<(usize, String)>,
    /// `M_.exo_det_names` of a stochastic model, and the parameters standing for
    /// them (see `ModelBuilder::q1_exo_det`).
    pub exo_det_names: Vec<String>,
    pub exo_det_params: Vec<usize>,
    // ---- model_info ----
    /// Equations (lhs, rhs) replacing the static versions of some equations (`[static]` equations, OccBin
    /// relaxed regime), as the preprocessor's static model (see [`blocks`]).
    pub static_eqs: Vec<Option<(NodeId, NodeId)>>,
    // ---- end model_info ----
    /// `model(block)`: the block decompositions the solvers use (set after parsing).
    pub block_solve: Option<std::sync::Arc<blocks::BlockSolve>>,
    /// `model(bytecode)`: the steady state with `solve_algo = 5` and the perfect foresight
    /// solver with `stack_solve_algo = 5` (the `bytecode` MEX's Newton) are allowed.
    pub bytecode: bool,
}

/// hidden equation tag carrying a `⟂` complementarity condition from the parser.
pub const MCP_TAG: &str = "__mcp";

/// Scratch memory for allocation-free evaluation (one per thread).
#[derive(Clone, Debug, Default)]
pub struct Workspace {
    s: Vec<f64>,
    y3: Vec<f64>,
    /// stream PP: extended parameters `[params; ys]` for `STEADY_STATE()`.
    pext: Vec<f64>,
}

thread_local! {
    static TLS_WS: RefCell<Workspace> = RefCell::new(Workspace::default());
}

/// A compiled standalone expression over parameters (and optionally steady-state values of
/// endogenous/exogenous variables); see [`Model::compile_expr`].
#[derive(Clone, Debug)]
pub struct CompiledExpr {
    tape: Tape,
}

impl CompiledExpr {
    /// Evaluates the expression. `y` (endogenous, any lead/lag reads `y[j]`) and `x` may be empty
    /// if the expression does not reference them.
    pub fn eval(&self, y: &[f64], x: &[f64], p: &[f64]) -> f64 {
        let mut s = vec![0.0; self.tape.scratch_len()];
        self.tape.eval(y, x, p, &mut s);
        self.tape.output(0, &s)
    }

    /// Allocation-free variant; `s.len() >= self.scratch_len()`.
    pub fn eval_with(&self, y: &[f64], x: &[f64], p: &[f64], s: &mut [f64]) -> f64 {
        self.tape.eval(y, x, p, s);
        self.tape.output(0, s)
    }

    pub fn scratch_len(&self) -> usize {
        self.tape.scratch_len()
    }
}

/// Commands that set the preprocessor's `bvar_present` (a model without equations is allowed).
pub fn is_bvar_command(name: &str) -> bool {
    matches!(name, "bvar_density" | "bvar_forecast" | "bvar_irf") || crate::msbvar::handles(name)
}

impl Model {
    /// The preprocessor refuses derivatives of order 3 or more of an `external_function` (only
    /// first and second derivatives can be provided): checked before `what` differentiates.
    pub fn check_external_order(&self, order: usize, what: &str) -> Result<()> {
        if order >= 3 && self.dag.has_external() {
            bail!(
                "{what}: the model calls external functions, whose derivatives of order 3 or more are not available (as in Dynare's preprocessor); use order <= 2"
            );
        }
        Ok(())
    }

    pub fn n_endo(&self) -> usize {
        self.endo_names.len()
    }
    pub fn n_exo(&self) -> usize {
        self.n_exo
    }
    pub fn n_params(&self) -> usize {
        self.param_names.len()
    }
    /// `M_.param_nbr` — the declared parameters, without the ones standing for
    /// the `varexo_det` variables (appended after them, see `ModelBuilder::q1_exo_det`).
    pub fn n_user_params(&self) -> usize {
        self.param_names.len() - self.exo_det_params.len()
    }
    /// `oo_.exo_det_steady_state` at the parameter vector `params`.
    pub fn exo_det_values(&self, params: &[f64]) -> Vec<f64> {
        self.exo_det_params.iter().map(|&k| params[k]).collect()
    }
    pub fn n_equations(&self) -> usize {
        self.equations.len()
    }

    // ---- stream PP: STEADY_STATE() in model equations --------------------------------------
    /// True when the dynamic model has `STEADY_STATE()` terms. Its dynamic evaluators then take
    /// the steady state through the parameter slice: pass `params_with_steady_state(p, ys)`
    /// (`[params; ys]`). Methods that receive `ys` themselves (`dynamic_jacobian*`,
    /// `DynamicDerivatives::eval`) extend plain parameters on their own; the `y3` evaluators
    /// given plain parameters read the steady state from `y_t` (exact at the steady state).
    pub fn uses_steady_state(&self) -> bool {
        self.has_ss
    }

    /// `[params; ys]` when the model uses `STEADY_STATE()`, else `params` (Dynare passes
    /// `steady_state` to every dynamic file).
    pub fn params_with_steady_state(&self, p: &[f64], ys: &[f64]) -> Vec<f64> {
        if self.has_ss {
            let np = self.n_params();
            let mut v = Vec::with_capacity(np + ys.len());
            v.extend_from_slice(&p[..np]);
            v.extend_from_slice(&ys[..self.n_endo()]);
            v
        } else {
            p.to_vec()
        }
    }

    /// The parameter slice of the tapes of parameter derivatives of the dynamic model:
    /// `[params; ys; dys(:); d2ys(:)]` when the model uses `STEADY_STATE()`, else `params`.
    /// Dynare passes `steady_state`, `ss_param_deriv = dys` (`n_endo x n_params`) and
    /// `ss_param_2nd_deriv = d2ys` to `dynamic_params_derivs.m`
    /// (get_perturbation_params_derivs.m); the derivatives of `STEADY_STATE(y)` w.r.t. the
    /// parameters ([`expr::Node::SteadyStateParamDeriv`]) read them from here. `d2ys`
    /// (`d2ys[q2]` is the `n_endo x n_params` slice `d2ys(:,:,q2)`) is only read by second
    /// parameter derivatives; `None` fills it with zeros, as Dynare's initialization.
    pub fn params_with_steady_state_derivs(&self, p: &[f64], ys: &[f64], dys: &Mat, d2ys: Option<&[Mat]>) -> Vec<f64> {
        if !self.has_ss {
            return p.to_vec();
        }
        let (np, n) = (self.n_params(), self.n_endo());
        let mut v = Vec::with_capacity(np + n + n * np * (1 + np));
        v.extend_from_slice(&p[..np]);
        v.extend_from_slice(&ys[..n]);
        for q in 0..np {
            v.extend((0..n).map(|j| dys[(j, q)]));
        }
        match d2ys {
            Some(d2) => {
                for m in &d2[..np] {
                    for q1 in 0..np {
                        v.extend((0..n).map(|j| m[(j, q1)]));
                    }
                }
            }
            None => v.resize(np + n + n * np * (1 + np), 0.0),
        }
        v
    }

    /// The parameter slice for a dynamic tape: `p` itself, or `[p; ys]` built in `buf`.
    #[inline]
    fn ext_params<'a>(&self, p: &'a [f64], ys: &[f64], buf: &'a mut Vec<f64>) -> &'a [f64] {
        let np = self.n_params();
        if !self.has_ss || p.len() >= np + self.n_endo() {
            return p;
        }
        buf.clear();
        buf.extend_from_slice(&p[..np]);
        buf.extend_from_slice(&ys[..self.n_endo()]);
        buf
    }
    // ---- end stream PP ----------------------------------------------------------------------

    // ---- stream B (LaTeX): read-only access to the transformed equations
    /// The expression DAG holding the equations.
    pub fn dag(&self) -> &Dag {
        &self.dag
    }
    /// The transformed equations `(lhs, rhs)` (model-local variables inlined, aux equations
    /// appended), as Dynare's dynamic model.
    pub fn equations(&self) -> &[(NodeId, NodeId)] {
        &self.equations
    }

    pub fn symbol(&self, name: &str) -> Option<Symbol> {
        self.symbols.get(name).copied()
    }
    pub fn endo_index(&self, name: &str) -> Option<usize> {
        self.endo_names.iter().position(|n| n == name)
    }
    pub fn exo_index(&self, name: &str) -> Option<usize> {
        self.exo_names.iter().position(|n| n == name)
    }
    /// Renames a parameter (symbol table and `param_names`; the tapes refer to indices).
    pub fn rename_param(&mut self, old: &str, new: &str) {
        if let Some(s) = self.symbols.remove(old) {
            self.symbols.insert(new.to_string(), s);
            if let Some(n) = self.param_names.get_mut(s.index) {
                *n = new.to_string();
            }
        }
    }

    pub fn param_index(&self, name: &str) -> Option<usize> {
        self.param_names.iter().position(|n| n == name)
    }

    /// A fresh workspace sized for this model.
    pub fn workspace(&self) -> Workspace {
        Workspace { s: vec![0.0; self.scratch_len], y3: vec![0.0; 3 * self.n_endo()], pext: Vec::new() }
    }

    fn with_tls<R>(&self, f: impl FnOnce(&mut Workspace) -> R) -> R {
        TLS_WS.with(|w| {
            let mut w = w.borrow_mut();
            if w.s.len() < self.scratch_len {
                w.s.resize(self.scratch_len, 0.0);
            }
            if w.y3.len() < 3 * self.n_endo() {
                w.y3.resize(3 * self.n_endo(), 0.0);
            }
            f(&mut w)
        })
    }

    /// Static residuals (`<fname>.sparse.static_resid`), `out.len() == n_equations()`.
    pub fn static_residuals(&self, y: &[f64], x: &[f64], p: &[f64], out: &mut [f64]) {
        self.with_tls(|w| self.static_residuals_with(y, x, p, w, out))
    }

    pub fn static_residuals_with(&self, y: &[f64], x: &[f64], p: &[f64], ws: &mut Workspace, out: &mut [f64]) {
        self.static_resid_tape.eval_into(y, x, p, &mut ws.s, out);
    }

    /// Static Jacobian, `n_equations() x n_endo()`.
    pub fn static_jacobian(&self, y: &[f64], x: &[f64], p: &[f64]) -> Mat {
        let mut m = Mat::zeros(self.n_equations(), self.n_endo());
        self.with_tls(|w| self.static_jacobian_with(y, x, p, w, &mut m));
        m
    }

    /// Fills `out` (must be `n_equations() x n_endo()`; entries outside the pattern are zeroed).
    pub fn static_jacobian_with(&self, y: &[f64], x: &[f64], p: &[f64], ws: &mut Workspace, out: &mut Mat) {
        assert_eq!(out.shape(), (self.n_equations(), self.n_endo()));
        self.static_jac_tape.eval(y, x, p, &mut ws.s);
        let nr = out.nrows();
        let o = out.as_mut_slice();
        o.fill(0.0);
        for (k, &(i, j)) in self.static_pattern.iter().enumerate() {
            o[i + j * nr] = self.static_jac_tape.output(k, &ws.s);
        }
    }

    /// Dynamic residuals at `y3 = [y_{t-1}; y_t; y_{t+1}]` (each `n_endo()` long) and `x` (exo at t).
    pub fn dynamic_residuals(&self, y3: &[f64], x: &[f64], p: &[f64], out: &mut [f64]) {
        self.with_tls(|w| self.dynamic_resid_tape_eval(y3, x, p, w, out))
    }

    fn dynamic_resid_tape_eval(&self, y3: &[f64], x: &[f64], p: &[f64], ws: &mut Workspace, out: &mut [f64]) {
        assert_eq!(y3.len(), 3 * self.n_endo());
        let n = self.n_endo();
        let p = self.ext_params(p, &y3[n..2 * n], &mut ws.pext);
        self.dyn_resid_tape.eval_into(y3, x, p, &mut ws.s, out);
    }

    /// Dynare 7's (sparse) dynamic g1 at the steady state (y_{t-1}=y_t=y_{t+1}=ys, x=x_ss),
    /// returned dense: `n_equations() x (3*n_endo + n_exo)`, columns `[y_{t-1}, y_t, y_{t+1}, x]`
    /// (== `full(<fname>.sparse.dynamic_g1)`).
    pub fn dynamic_jacobian(&self, ys: &[f64], x: &[f64], p: &[f64]) -> Mat {
        let mut m = Mat::zeros(self.n_equations(), 3 * self.n_endo() + self.n_exo);
        self.with_tls(|w| self.dynamic_jacobian_with(ys, x, p, w, &mut m));
        m
    }

    /// Allocation-free [`Model::dynamic_jacobian`]: fills `out` (zeroing it first).
    pub fn dynamic_jacobian_with(&self, ys: &[f64], x: &[f64], p: &[f64], ws: &mut Workspace, out: &mut Mat) {
        let n = self.n_endo();
        assert_eq!(ys.len(), n);
        let mut y3 = std::mem::take(&mut ws.y3);
        y3.resize(3 * n, 0.0);
        for t in 0..3 {
            y3[t * n..(t + 1) * n].copy_from_slice(ys);
        }
        let mut pext = std::mem::take(&mut ws.pext);
        let pp = self.ext_params(p, ys, &mut pext);
        self.dynamic_jacobian_y3_with(&y3, x, pp, ws, out);
        ws.pext = pext;
        ws.y3 = y3;
    }

    /// Dynamic Jacobian at an arbitrary `y3 = [y_{t-1}; y_t; y_{t+1}]`.
    pub fn dynamic_jacobian_y3_with(&self, y3: &[f64], x: &[f64], p: &[f64], ws: &mut Workspace, out: &mut Mat) {
        assert_eq!(out.shape(), (self.n_equations(), 3 * self.n_endo() + self.n_exo));
        let n = self.n_endo();
        let p = self.ext_params(p, &y3[n..2 * n], &mut ws.pext);
        self.dyn_jac_tape.eval(y3, x, p, &mut ws.s);
        let nr = out.nrows();
        let o = out.as_mut_slice();
        o.fill(0.0);
        for (k, &(i, j)) in self.dyn_pattern.iter().enumerate() {
            o[i + j * nr] = self.dyn_jac_tape.output(k, &ws.s);
        }
    }

    /// Structural non-zeros `(row, col)` of the dynamic Jacobian, column-major order.
    pub fn dynamic_jacobian_pattern(&self) -> &[(usize, usize)] {
        &self.dyn_pattern
    }

    /// Values of the dynamic Jacobian non-zeros (same order as [`Model::dynamic_jacobian_pattern`])
    /// at `y3`; `out.len() == pattern.len()`.
    pub fn dynamic_jacobian_values_with(&self, y3: &[f64], x: &[f64], p: &[f64], ws: &mut Workspace, out: &mut [f64]) {
        let n = self.n_endo();
        let p = self.ext_params(p, &y3[n..2 * n], &mut ws.pext);
        self.dyn_jac_tape.eval_into(y3, x, p, &mut ws.s, out);
    }

    /// Port of `<fname>.set_auxiliary_variables`: sets every aux variable of `y` from the
    /// original variables (static model).
    pub fn set_auxiliary_variables(&self, y: &mut [f64], x: &[f64], p: &[f64]) {
        if self.aux_vars.is_empty() {
            return;
        }
        self.with_tls(|w| {
            self.aux_tape.eval(y, x, p, &mut w.s);
            for (k, a) in self.aux_vars.iter().enumerate() {
                y[a.endo_index] = self.aux_tape.output(k, &w.s);
            }
        })
    }

    /// Compiles an expression in .mod syntax (e.g. `"(1-rho)*ss_x"`) referencing parameters
    /// and, optionally, endogenous/exogenous variables (read at their steady state).
    pub fn compile_expr(&self, src: &str) -> Result<CompiledExpr> {
        let ast = crate::modfile::parse_expression(src)?;
        let mut dag = Dag::new();
        let id = crate::modfile::ast_to_node(&ast, &mut dag, &mut |d, name, lag, loc| match self.symbol(name) {
            Some(s) => match s.kind {
                SymbolKind::Endo => Ok(d.endo(s.index, lag)),
                SymbolKind::Exo => Ok(d.exo(s.index, lag)),
                SymbolKind::Param => Ok(d.param(s.index)),
                SymbolKind::Local | SymbolKind::Trend => {
                    bail!("{loc}: model-local or trend variable '{name}' cannot be used here")
                }
            },
            None => match name {
                "inf" | "Inf" => Ok(d.num(f64::INFINITY)),
                "nan" | "NaN" => Ok(d.num(f64::NAN)),
                "pi" => Ok(d.num(std::f64::consts::PI)),
                _ => bail!("{loc}: unknown symbol '{name}'"),
            },
        })?;
        let tape = Tape::compile(&dag, &[id], &|j, _| j as usize, &|k, _| k as usize);
        Ok(CompiledExpr { tape })
    }

    /// Equation `i` in .mod syntax (after aux substitution), for diagnostics.
    pub fn equation_string(&self, i: usize) -> String {
        let namef = |n: Node| match n {
            Node::Endo(j, _) => self.endo_names[j as usize].clone(),
            Node::Exo(k, _) => self.exo_names[k as usize].clone(),
            Node::Param(q) => self.param_names[q as usize].clone(),
            _ => String::new(),
        };
        let (l, r) = self.equations[i];
        format!("{} = {}", self.dag.to_string(l, &namef), self.dag.to_string(r, &namef))
    }

    /// Residual of equation `i` (lhs - rhs) as a DAG node, with the model's expression arena.
    pub fn residual_node(&self, i: usize) -> (&Dag, NodeId) {
        (&self.dag, self.resid[i])
    }
}
