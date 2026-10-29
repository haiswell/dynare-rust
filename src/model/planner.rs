// SPDX-License-Identifier: GPL-3.0-or-later
//! Optimal-policy transformations of the preprocessor (stream P).
//!
//! Dynare's preprocessor (`ModFile::transformPass` + `DynamicModel::computeRamseyPolicyFOCs`)
//! does three things for the optimal-policy commands, mirrored here:
//!
//! * `ramsey_model` / `ramsey_policy`: one Lagrange multiplier `MULT_i` per model equation
//!   (endogenous, appended after the declared variables, `M_.aux_vars` type 6), and the
//!   first-order conditions of the Lagrangian
//!   `sum_t beta^t [U_t + sum_i MULT_i,t f_i,t]` with respect to every declared variable,
//!   placed *before* the original equations (`Ramsey FOC w.r.t. <var>`):
//!   `dU/dy_j + sum_i sum_l beta^(-l) MULT_i(-l) * (df_i/dy_j(l))(-l) = 0`.
//!   The FOCs are built before the lead/lag auxiliary-variable substitution, which then runs
//!   on the whole system (so `MULT_1(-2)` gets an `AUX_ENDO_LAG_..` variable, as in Dynare).
//! * `discretionary_policy`: the model keeps fewer equations than variables (one missing per
//!   instrument).
//! * all of them (and `osr`): a parameter `optimal_policy_discount_factor` appended to the
//!   parameters, set to `planner_discount` (default 1).
//!
//! The planner objective (`planner_objective`) is compiled as Dynare's
//! `<fname>.objective.static_resid/static_g1/static_g2` (value, gradient, Hessian w.r.t. the
//! endogenous variables of the static model), and the Jacobian of the static FOCs with respect
//! to the multipliers as `<fname>.ramsey_multipliers_static_g1` (computed before the aux
//! substitution, as Dynare does).

use super::Model;
use super::expr::{Dag, DiffVar, Node, NodeId};
use super::tape::Tape;
use crate::linalg::Mat;
use anyhow::{Result, bail};
use std::collections::HashMap;

/// What the .mod file asks for (collected by the parser into the `ModelBuilder`).
#[derive(Clone, Debug, Default)]
pub struct PlannerSpec {
    /// `planner_objective` (static expression in the model DAG, current-period variables only).
    pub objective: Option<NodeId>,
    /// The objective as parsed, with the text of its numerical constants (replayed by `dtree`).
    pub objective_ast: Option<(crate::modfile::Ast, Vec<String>)>,
    pub ramsey: bool,
    pub discretionary: bool,
    pub osr: bool,
    /// `planner_discount` (evaluated with the calibration when the command was read).
    pub discount: Option<f64>,
    /// `instruments=(...)`, names as written.
    pub instruments: Vec<String>,
}

impl PlannerSpec {
    fn needs_discount_param(&self) -> bool {
        self.ramsey || self.discretionary || self.osr
    }
}

/// Ramsey structure (`M_.ramsey_orig_endo_nbr`, `M_.ramsey_orig_eq_nbr`, multipliers).
#[derive(Clone, Debug)]
pub struct RamseyInfo {
    /// Declared endogenous variables and the `diff` / `EXPECTATION` aux variables
    /// (`M_.ramsey_orig_endo_nbr`); the first that many equations are the FOCs.
    pub orig_endo_nbr: usize,
    /// Original model equations and the `diff` / `EXPECTATION` aux equations
    /// (`M_.ramsey_orig_eq_nbr`), equations `orig_endo_nbr..`.
    pub orig_eq_nbr: usize,
    /// Endogenous index of `MULT_{i+1}` (multiplier of original equation `i`).
    pub mult_index: Vec<usize>,
    mult_jac: Tape,
    /// (FOC row, multiplier column) of each `mult_jac` output.
    mult_pattern: Vec<(usize, usize)>,
}

/// Compiled planner objective: `U`, `Uy` and `Uyy` of the static model.
#[derive(Clone, Debug)]
struct ObjectiveTape {
    tape: Tape,
    /// endo index of each gradient output (outputs 1..=grad.len())
    grad: Vec<usize>,
    /// (j, k) of each Hessian output (both triangles listed)
    hess: Vec<(usize, usize)>,
    /// Third derivatives (`<fname>.objective.static_g3`, order >= 3 welfare): a separate tape
    /// with one output per sorted triple `j <= k <= l` of `third`.
    tape3: Tape,
    third: Vec<(usize, usize, usize)>,
    /// The static objective (for the derivatives of any order, `objective_derivatives_k`).
    node: NodeId,
}

/// Optimal-policy information attached to a [`Model`].
#[derive(Clone, Debug)]
pub struct PlannerInfo {
    /// Index of `optimal_policy_discount_factor` among the parameters (if added).
    pub discount_param: Option<usize>,
    /// Instruments as endogenous indices (declared order of `instruments=(...)`).
    pub instruments: Vec<usize>,
    pub ramsey: Option<RamseyInfo>,
    pub discretionary: bool,
    pub osr: bool,
    objective: Option<ObjectiveTape>,
}

/// Output of [`prepare`], completed by [`Prepared::finish`] once the model is compiled.
pub(super) struct Prepared {
    spec: PlannerSpec,
    discount_param: Option<usize>,
    instruments: Vec<usize>,
    /// Number of equations missing relative to the endogenous variables (discretionary policy).
    pub eq_deficit: usize,
    ramsey: Option<(usize, usize, Vec<usize>, Vec<NodeId>)>,
}

/// Mutable view of the model under construction.
pub(super) struct Builder<'a> {
    pub dag: &'a mut Dag,
    pub endo_names: &'a mut Vec<String>,
    pub next_symb_id: &'a mut usize,
    pub param_names: &'a mut Vec<String>,
    pub params: &'a mut Vec<f64>,
    pub symbols: &'a mut HashMap<String, super::Symbol>,
    pub equations: &'a mut Vec<(NodeId, NodeId)>,
}

/// Applies the optimal-policy transformations (before the aux-variable passes).
/// Returns `None` for models without optimal-policy commands.
pub(super) fn prepare(b: Builder<'_>, spec: Option<PlannerSpec>) -> Result<Option<Prepared>> {
    let Some(spec) = spec else { return Ok(None) };
    if !spec.needs_discount_param() && spec.objective.is_none() {
        return Ok(None);
    }
    let orig_endo_nbr = b.endo_names.len();
    let mut instruments = Vec::new();
    for name in &spec.instruments {
        match b.endo_names.iter().position(|n| n == name) {
            Some(j) => instruments.push(j),
            None => bail!("instrument '{name}' is not an endogenous variable"),
        }
    }
    let mut discount_param = None;
    if spec.needs_discount_param() {
        let name = "optimal_policy_discount_factor";
        if b.symbols.contains_key(name) {
            bail!("'{name}' is reserved for the planner discount factor");
        }
        b.param_names.push(name.to_string());
        b.params.push(spec.discount.unwrap_or(1.0));
        let idx = b.param_names.len() - 1;
        b.symbols.insert(
            name.to_string(),
            super::Symbol { kind: super::SymbolKind::Param, index: idx, symb_id: *b.next_symb_id },
        );
        *b.next_symb_id += 1;
        discount_param = Some(idx);
    }
    let n_eq = b.equations.len();
    let mut eq_deficit = 0;
    let mut ramsey = None;
    if spec.ramsey || spec.discretionary {
        if n_eq >= orig_endo_nbr {
            bail!(
                "optimal policy: the model has {n_eq} equations for {orig_endo_nbr} endogenous variables; \
                 it needs one equation less per instrument"
            );
        }
        // a different number of instruments only warns (model_diagnostics.m, ramsey::check_instruments)
    }
    if spec.discretionary {
        eq_deficit = orig_endo_nbr - n_eq;
        if spec.objective.is_none() {
            bail!("discretionary_policy needs a planner_objective");
        }
    }
    if spec.ramsey {
        let Some(obj) = spec.objective else { bail!("ramsey_model needs a planner_objective") };
        if b.dag.max_endo_lead(obj) > 0 || b.dag.max_endo_lag(obj) > 0 {
            bail!("leads and lags on variables are forbidden in 'planner_objective'");
        }
        // The multipliers (one per equation, after the declared variables and the diff /
        // EXPECTATION aux variables) and the first-order conditions are created by `dtree` (the
        // preprocessor's DataTree), see `Prepared::set_ramsey`; the static FOCs are filled in
        // by `ModelBuilder::build` (`Prepared::ramsey.3`).
        let _ = obj;
        ramsey = Some((orig_endo_nbr, n_eq, Vec::new(), Vec::new()));
    }
    Ok(Some(Prepared { spec, discount_param, instruments, eq_deficit, ramsey }))
}

impl Prepared {
    /// Input of the Ramsey step of the `dtree` transform pass.
    pub(super) fn tree_input(&self, endo_symb_id: &[usize], discount_symb: usize) -> Option<super::dtree::PlannerIn> {
        let (orig_endo, _, _, _) = self.ramsey.as_ref()?;
        let (objective, nums) = self.spec.objective_ast.clone()?;
        Some(super::dtree::PlannerIn {
            objective,
            nums,
            discount: discount_symb as u32,
            endo_symb: endo_symb_id[..*orig_endo].iter().map(|&x| x as u32).collect(),
        })
    }

    /// Sizes and multipliers created by the transform pass: `M_.ramsey_orig_endo_nbr`,
    /// `M_.ramsey_orig_eq_nbr` and the endogenous index of each `MULT_i`.
    pub(super) fn set_ramsey(&mut self, orig_endo: usize, orig_eq: usize, mult_index: Vec<usize>) {
        if let Some(r) = self.ramsey.as_mut() {
            (r.0, r.1, r.2) = (orig_endo, orig_eq, mult_index);
        }
    }

    /// Whether the model has `ramsey_model` / `ramsey_policy`.
    pub(super) fn is_ramsey(&self) -> bool {
        self.ramsey.is_some()
    }

    /// Static versions of the first-order conditions (before the aux substitution).
    pub(super) fn set_static_focs(&mut self, focs: Vec<NodeId>) {
        if let Some(r) = self.ramsey.as_mut() {
            r.3 = focs;
        }
    }

    /// Compiles the objective and the multiplier Jacobian (static model, `y` = all endogenous).
    pub(super) fn finish(self, dag: &mut Dag) -> PlannerInfo {
        let st_y = |j: u32, _l: i32| j as usize;
        let x0 = |k: u32, _l: i32| k as usize;
        let objective = self.spec.objective.map(|obj| {
            let mut smemo = HashMap::new();
            let u = dag.to_static(obj, &mut smemo);
            let mut leaves = Vec::new();
            dag.collect_leaves(u, &mut leaves);
            let mut vars: Vec<usize> =
                leaves.iter().filter_map(|l| if let Node::Endo(j, _) = l { Some(*j as usize) } else { None }).collect();
            vars.sort_unstable();
            vars.dedup();
            let mut roots = vec![u];
            let mut grad = Vec::new();
            let mut gnodes = Vec::new();
            for &j in &vars {
                let mut memo = HashMap::new();
                let d = dag.diff(u, DiffVar::Endo(j as u32, 0), &mut memo);
                if d != Dag::ZERO {
                    grad.push(j);
                    gnodes.push(d);
                }
            }
            roots.extend(&gnodes);
            let mut hess = Vec::new();
            let mut roots3 = Vec::new();
            let mut third = Vec::new();
            for (&j, &d) in grad.iter().zip(&gnodes) {
                for &k in &vars {
                    let mut memo = HashMap::new();
                    let h = dag.diff(d, DiffVar::Endo(k as u32, 0), &mut memo);
                    if h != Dag::ZERO {
                        hess.push((j, k));
                        roots.push(h);
                        if j <= k {
                            for &l in vars.iter().filter(|&&l| l >= k) {
                                let mut memo = HashMap::new();
                                let t = dag.diff(h, DiffVar::Endo(l as u32, 0), &mut memo);
                                if t != Dag::ZERO {
                                    third.push((j, k, l));
                                    roots3.push(t);
                                }
                            }
                        }
                    }
                }
            }
            ObjectiveTape {
                tape: Tape::compile(dag, &roots, &st_y, &x0),
                grad,
                hess,
                tape3: Tape::compile(dag, &roots3, &st_y, &x0),
                third,
                node: u,
            }
        });
        let ramsey = self.ramsey.map(|(orig_endo_nbr, orig_eq_nbr, mult_index, foc_static)| {
            let mut roots = Vec::new();
            let mut mult_pattern = Vec::new();
            for (c, &m) in mult_index.iter().enumerate() {
                let mut memo = HashMap::new();
                for (r, &f) in foc_static.iter().enumerate() {
                    let d = dag.diff(f, DiffVar::Endo(m as u32, 0), &mut memo);
                    if d != Dag::ZERO {
                        mult_pattern.push((r, c));
                        roots.push(d);
                    }
                }
            }
            RamseyInfo {
                orig_endo_nbr,
                orig_eq_nbr,
                mult_index,
                mult_jac: Tape::compile(dag, &roots, &st_y, &x0),
                mult_pattern,
            }
        });
        PlannerInfo {
            discount_param: self.discount_param,
            instruments: self.instruments,
            ramsey,
            discretionary: self.spec.discretionary,
            osr: self.spec.osr,
            objective,
        }
    }
}

/// Planner objective at a point of the static model: `U`, `Uy` (`n_endo`) and `Uyy`
/// (`n_endo x n_endo`), as `<fname>.objective.static_resid/static_g1/static_g2`.
#[derive(Clone, Debug)]
pub struct ObjectiveValue {
    pub u: f64,
    pub uy: Vec<f64>,
    pub uyy: Mat,
}

impl Model {
    /// Optimal-policy information (`None` without `ramsey_model`/`discretionary_policy`/`osr`/
    /// `planner_objective`).
    pub fn planner(&self) -> Option<&PlannerInfo> {
        self.planner.as_ref()
    }

    /// The planner discount factor (`get_optimal_policy_discount_factor.m`) for parameters `p`.
    pub fn planner_discount(&self, p: &[f64]) -> Option<f64> {
        self.planner.as_ref()?.discount_param.map(|i| p[i])
    }

    /// Planner objective, gradient and Hessian at the static point `y` (all endogenous).
    pub fn planner_objective(&self, y: &[f64], x: &[f64], p: &[f64]) -> Option<ObjectiveValue> {
        let o = self.planner.as_ref()?.objective.as_ref()?;
        let n = self.n_endo();
        let mut s = vec![0.0; o.tape.scratch_len()];
        o.tape.eval(y, x, p, &mut s);
        let mut uy = vec![0.0; n];
        for (k, &j) in o.grad.iter().enumerate() {
            uy[j] = o.tape.output(1 + k, &s);
        }
        let mut uyy = Mat::zeros(n, n);
        let off = 1 + o.grad.len();
        for (k, &(i, j)) in o.hess.iter().enumerate() {
            uyy[(i, j)] = o.tape.output(off + k, &s);
        }
        Some(ObjectiveValue { u: o.tape.output(0, &s), uy, uyy })
    }

    /// Third derivatives of the planner objective at the static point `y`
    /// (`<fname>.objective.static_g3`): `(j, k, l, value)` for the sorted triples `j <= k <= l`
    /// with a nonzero derivative.
    pub fn planner_objective_g3(&self, y: &[f64], x: &[f64], p: &[f64]) -> Option<Vec<(usize, usize, usize, f64)>> {
        let o = self.planner.as_ref()?.objective.as_ref()?;
        let mut s = vec![0.0; o.tape3.scratch_len()];
        o.tape3.eval(y, x, p, &mut s);
        Some(o.third.iter().enumerate().map(|(k, &(i, j, l))| (i, j, l, o.tape3.output(k, &s))).collect())
    }

    /// Derivatives of orders `1..=order` of the planner objective at the static point `y`
    /// (`k_order_welfare` at order >= 4): `out[m - 1]` lists `([j1 <= ... <= jm], value)` over
    /// the endogenous variables (declaration order), structurally non-zero entries only.
    pub fn planner_objective_derivatives_k(
        &self,
        order: usize,
        y: &[f64],
        x: &[f64],
        p: &[f64],
    ) -> Option<Vec<Vec<(Vec<usize>, f64)>>> {
        let o = self.planner.as_ref()?.objective.as_ref()?;
        let mut dag = self.dag.clone();
        let mut found: Vec<Vec<(Vec<usize>, NodeId)>> = vec![Vec::new(); order];
        let mut stack = vec![(Vec::<usize>::new(), o.node)];
        let mut leaves = Vec::new();
        while let Some((js, f)) = stack.pop() {
            if js.len() == order {
                continue;
            }
            leaves.clear();
            dag.collect_leaves(f, &mut leaves);
            let lo = js.last().copied().unwrap_or(0);
            let mut vars: Vec<usize> = leaves
                .iter()
                .filter_map(|l| if let Node::Endo(j, _) = l { Some(*j as usize) } else { None })
                .filter(|&j| j >= lo)
                .collect();
            vars.sort_unstable();
            vars.dedup();
            for j in vars {
                let mut memo = HashMap::new();
                let d = dag.diff(f, DiffVar::Endo(j as u32, 0), &mut memo);
                if d != Dag::ZERO {
                    let mut js2 = js.clone();
                    js2.push(j);
                    found[js2.len() - 1].push((js2.clone(), d));
                    stack.push((js2, d));
                }
            }
        }
        let st_y = |j: u32, _l: i32| j as usize;
        let x0 = |k: u32, _l: i32| k as usize;
        Some(
            found
                .into_iter()
                .map(|f| {
                    let tape = Tape::compile(&dag, &f.iter().map(|e| e.1).collect::<Vec<_>>(), &st_y, &x0);
                    let mut s = vec![0.0; tape.scratch_len()];
                    tape.eval(y, x, p, &mut s);
                    f.into_iter().enumerate().map(|(k, (js, _))| (js, tape.output(k, &s))).collect()
                })
                .collect(),
        )
    }

    /// `<fname>.ramsey_multipliers_static_g1`: derivative of the static FOCs (rows,
    /// `ramsey_orig_endo_nbr`) with respect to the multipliers (columns), before aux substitution.
    pub fn ramsey_multipliers_static_g1(&self, y: &[f64], x: &[f64], p: &[f64]) -> Option<Mat> {
        let r = self.planner.as_ref()?.ramsey.as_ref()?;
        let mut s = vec![0.0; r.mult_jac.scratch_len()];
        r.mult_jac.eval(y, x, p, &mut s);
        let mut a = Mat::zeros(r.orig_endo_nbr, r.mult_index.len());
        for (k, &(i, j)) in r.mult_pattern.iter().enumerate() {
            a[(i, j)] = r.mult_jac.output(k, &s);
        }
        Some(a)
    }
}
