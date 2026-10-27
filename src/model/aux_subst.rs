// SPDX-License-Identifier: GPL-3.0-or-later
//! Auxiliary-variable substitution for leads >= 2 and lags >= 2, a port of the
//! preprocessor's `DynamicModel::substituteEndoLeadGreaterThanTwo`,
//! `substituteExoLead`, `substituteEndoLagGreaterThanTwo` and `substituteExoLag`
//! (`ExprNode.cc`: `createEndoLeadAuxiliaryVarForMyself`, `VariableNode::substitute*`).
//!
//! The traversal order (local variables first, then equations in order, left
//! operand before right operand), the shared substitution table, the rules for
//! products/quotients and the naming of lag auxiliaries (`AUX_ENDO_LAG_<symb_id>_<k>`)
//! all follow Dynare, so the resulting `endo_names`, `M_.aux_vars` and equation
//! order are identical. Lead auxiliaries are named `AUX_ENDO_LEAD_<node id>`, where
//! Dynare uses its own internal expression index.

use super::AuxVar;
use super::expr::{BinaryOp, Dag, Node, NodeId, UnaryOp};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pass {
    EndoLead,
    ExoLead,
    EndoLag,
    ExoLag,
}

/// Mutable view of the model being transformed.
pub(crate) struct Subst<'a> {
    pub dag: &'a mut Dag,
    pub endo_names: &'a mut Vec<String>,
    pub exo_names: &'a [String],
    /// Global symbol id (declaration order across all symbol types) of each endo / exo.
    pub endo_symb_id: &'a [usize],
    pub exo_symb_id: &'a [usize],
    pub aux_vars: &'a mut Vec<AuxVar>,
    /// Right-hand side of each aux equation `AUX = rhs`, parallel to `aux_vars`.
    pub aux_rhs: &'a mut Vec<NodeId>,
    pub deterministic: bool,
}

impl Subst<'_> {
    /// Runs one substitution pass over the used local variables then all equations
    /// (`DynamicModel::substituteLeadLagInternal`), appending the new aux equations.
    pub fn run(&mut self, pass: Pass, locals: &[NodeId], equations: &mut Vec<(NodeId, NodeId)>) {
        let mut st = PassState { pass, table: HashMap::new(), memo: HashMap::new(), new_eqs: Vec::new() };
        for &l in locals {
            self.subst(&mut st, l);
        }
        for eq in equations.iter_mut() {
            let lhs = self.subst(&mut st, eq.0);
            let rhs = self.subst(&mut st, eq.1);
            *eq = (lhs, rhs);
        }
        equations.extend(st.new_eqs);
    }

    fn above(&self, pass: Pass, id: NodeId) -> bool {
        match pass {
            Pass::EndoLead => self.dag.max_endo_lead(id) >= 2,
            Pass::ExoLead => self.dag.max_exo_lead(id) >= 1,
            Pass::EndoLag => self.dag.max_endo_lag(id) >= 2,
            Pass::ExoLag => self.dag.max_exo_lag(id) >= 1,
        }
    }

    fn subst(&mut self, st: &mut PassState, id: NodeId) -> NodeId {
        if !self.above(st.pass, id) {
            return id;
        }
        if let Some(&r) = st.memo.get(&id) {
            return r;
        }
        let r = match st.pass {
            Pass::EndoLead | Pass::ExoLead => self.subst_lead(st, id),
            Pass::EndoLag | Pass::ExoLag => self.subst_lag(st, id),
        };
        st.memo.insert(id, r);
        r
    }

    /// `substituteEndoLeadGreaterThanTwo` / `substituteExoLead` for one node.
    fn subst_lead(&mut self, st: &mut PassState, id: NodeId) -> NodeId {
        let endo = st.pass == Pass::EndoLead;
        match self.dag.node(id) {
            Node::Endo(..) | Node::Exo(..) => self.create_lead_aux(st, id),
            Node::Num(_)
            | Node::Param(_)
            | Node::SteadyState(..)
            | Node::SteadyStateParamDeriv(..)
            | Node::SteadyStateParam2ndDeriv(..) => id,
            // `AbstractExternalFunctionNode::substitute*`: recurse into the arguments
            Node::External(c) => {
                let args = self.dag.ext_call(c).args.clone();
                let a2: Vec<NodeId> = args.iter().map(|&a| self.subst(st, a)).collect();
                self.dag.rebuild_external(id, c, a2)
            }
            Node::Unary(op, a) => {
                if op == UnaryOp::Neg || self.deterministic {
                    let a2 = self.subst(st, a);
                    self.dag.unary(op, a2)
                } else {
                    self.create_lead_aux(st, id)
                }
            }
            Node::Binary(op, a, b) => {
                if self.deterministic || matches!(op, BinaryOp::Add | BinaryOp::Sub) {
                    let a2 = self.subst(st, a);
                    let b2 = self.subst(st, b);
                    return self.dag.binary(op, a2, b2);
                }
                if matches!(op, BinaryOp::Mul | BinaryOp::Div) {
                    // (own lead, other kind's lead) of each operand
                    let leads = |d: &Dag, n: NodeId| {
                        if endo {
                            (d.max_endo_lead(n), d.max_exo_lead(n))
                        } else {
                            (d.max_exo_lead(n), d.max_endo_lead(n))
                        }
                    };
                    let thr = if endo { 2 } else { 1 };
                    let (own1, other1) = leads(self.dag, a);
                    let (own2, other2) = leads(self.dag, b);
                    if own1 >= thr && own2 == 0 && other2 == 0 {
                        let a2 = self.subst(st, a);
                        return self.dag.binary(op, a2, b);
                    }
                    if own1 == 0 && other1 == 0 && own2 >= thr && op == BinaryOp::Mul {
                        let b2 = self.subst(st, b);
                        return self.dag.binary(op, a, b2);
                    }
                }
                self.create_lead_aux(st, id)
            }
        }
    }

    /// `ExprNode::createEndoLeadAuxiliaryVarForMyself` / `createExoLeadAuxiliaryVarForMyself`.
    fn create_lead_aux(&mut self, st: &mut PassState, id: NodeId) -> NodeId {
        if let Some(&r) = st.table.get(&id) {
            return r;
        }
        let endo = st.pass == Pass::EndoLead;
        let (mut substexpr, mut lag) = if endo {
            let n = self.dag.max_endo_lead(id);
            (self.dag.shift(id, n - 1), n - 2)
        } else {
            let n = self.dag.max_exo_lead(id);
            (self.dag.shift(id, n), n - 1)
        };
        while lag >= 0 {
            let orig = self.dag.shift(id, lag);
            if let Some(&r) = st.table.get(&orig) {
                substexpr = r;
            } else {
                let prefix = if endo { "AUX_ENDO_LEAD_" } else { "AUX_EXO_LEAD_" };
                let aux = self.new_aux(format!("{prefix}{orig}"), if endo { 0 } else { 2 }, None, None, substexpr);
                let lhs = self.dag.endo(aux, 0);
                st.new_eqs.push((lhs, substexpr));
                substexpr = self.dag.endo(aux, 1);
                st.table.insert(orig, substexpr);
            }
            lag -= 1;
        }
        substexpr
    }

    /// `VariableNode::substituteEndoLagGreaterThanTwo` / `substituteExoLag`; other nodes recurse.
    fn subst_lag(&mut self, st: &mut PassState, id: NodeId) -> NodeId {
        match self.dag.node(id) {
            Node::Endo(j, lag) if st.pass == Pass::EndoLag => self.create_lag_aux(st, id, j, lag, true),
            Node::Exo(k, lag) if st.pass == Pass::ExoLag => self.create_lag_aux(st, id, k, lag, false),
            Node::Unary(op, a) => {
                let a2 = self.subst(st, a);
                self.dag.unary(op, a2)
            }
            Node::Binary(op, a, b) => {
                let a2 = self.subst(st, a);
                let b2 = self.subst(st, b);
                self.dag.binary(op, a2, b2)
            }
            Node::External(c) => {
                let args = self.dag.ext_call(c).args.clone();
                let a2: Vec<NodeId> = args.iter().map(|&a| self.subst(st, a)).collect();
                self.dag.rebuild_external(id, c, a2)
            }
            _ => id,
        }
    }

    fn create_lag_aux(&mut self, st: &mut PassState, id: NodeId, v: u32, lag: i32, endo: bool) -> NodeId {
        if let Some(&r) = st.table.get(&id) {
            return r;
        }
        let v = v as usize;
        // substexpr stands for v(cur+1): v(-1) for endogenous (AUX_ENDO_LAG_k_1 = x(-1)), v(0) for exogenous
        let mut substexpr = if endo { self.dag.endo(v, -1) } else { self.dag.exo(v, 0) };
        let mut cur = if endo { -2 } else { -1 };
        while cur >= lag {
            let orig = if endo { self.dag.endo(v, cur) } else { self.dag.exo(v, cur) };
            if let Some(&r) = st.table.get(&orig) {
                substexpr = r;
            } else {
                let rhs = substexpr;
                let (name, ty) = if endo {
                    (format!("AUX_ENDO_LAG_{}_{}", self.endo_symb_id[v], -(cur + 1)), 1)
                } else {
                    (format!("AUX_EXO_LAG_{}_{}", self.exo_symb_id[v], -(cur + 1)), 3)
                };
                let aux = self.new_aux(name, ty, Some(v), Some(cur + 1), rhs);
                let lhs = self.dag.endo(aux, 0);
                st.new_eqs.push((lhs, rhs));
                substexpr = self.dag.endo(aux, -1);
                st.table.insert(orig, substexpr);
            }
            cur -= 1;
        }
        substexpr
    }

    fn new_aux(
        &mut self,
        name: String,
        ty: u8,
        orig_index: Option<usize>,
        orig_lead_lag: Option<i32>,
        rhs: NodeId,
    ) -> usize {
        let idx = self.endo_names.len();
        self.endo_names.push(name);
        self.aux_vars.push(AuxVar { endo_index: idx, ty, orig_index, orig_lead_lag, orig_expr: String::new() });
        self.aux_rhs.push(rhs);
        idx
    }

    /// Fills `orig_expr` of every aux var (printed right-hand side of its equation).
    pub fn name_exprs(&mut self) {
        name_exprs(self.dag, self.endo_names, self.exo_names, None, self.aux_vars, self.aux_rhs);
    }
}

/// Fills `orig_expr` of every aux var (printed right-hand side of its equation).
/// `params`: parameter names (as the driver prints them; ), else `params(i)`.
pub(crate) fn name_exprs(
    dag: &Dag,
    endo_names: &[String],
    exo: &[String],
    params: Option<&[String]>,
    aux_vars: &mut [AuxVar],
    aux_rhs: &[NodeId],
) {
    let namef = |n: Node| match n {
        Node::Endo(j, _) => endo_names[j as usize].clone(),
        Node::Exo(k, _) => exo[k as usize].clone(),
        Node::Param(i) => match params {
            Some(p) => p[i as usize].clone(),
            None => format!("params({})", i + 1),
        },
        Node::Num(_)
        | Node::SteadyState(..)
        | Node::SteadyStateParamDeriv(..)
        | Node::SteadyStateParam2ndDeriv(..)
        | Node::Unary(..)
        | Node::Binary(..)
        | Node::External(_) => String::new(),
    };
    for (a, &rhs) in aux_vars.iter_mut().zip(aux_rhs.iter()) {
        a.orig_expr = dag.to_string(rhs, &namef);
    }
}

struct PassState {
    pass: Pass,
    /// Dynare's `subst_table`: original expression -> replacement.
    table: HashMap<NodeId, NodeId>,
    /// Per-pass memo of already rewritten nodes (DAG sharing; same visit order as a tree walk).
    memo: HashMap<NodeId, NodeId>,
    new_eqs: Vec<(NodeId, NodeId)>,
}
