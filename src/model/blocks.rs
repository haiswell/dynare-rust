// SPDX-License-Identifier: GPL-3.0-or-later
//! The preprocessor's block decomposition (`ModelTree::computingPassBlock`, Dynare 7.2), static
//! and dynamic variants: `M_.block_structure` / `M_.block_structure_stat`.
//!
//! Port of `ModelTree::{computeNonSingularNormalization, computeNormalization,
//! evaluateAndReduceJacobian, computeLeftHandSideSymbolicJacobian, computePrologueAndEpilogue,
//! equationTypeDetermination, computeBlockDecomposition, getVariableLeadLagByBlock,
//! computeDynamicStructureOfBlock, computeSimulationTypeOfBlock, reduceBlockDecomposition,
//! determineLinearBlocks}`, `DynamicModel::{determineBlockDerivativesType,
//! computeChainRuleJacobian}` (for `is_linear`), `BinaryOpNode::normalizeEquation` and the
//! driver output (`{Dynamic,Static}Model::writeBlockDriverOutput`). The graph algorithms (Boost
//! matching, SCCs, feedback vertex set) are in [`super::blocks_graph`].

use super::Model;
use super::blocks_graph::{DepGraph, edmonds_matching};
use super::expr::{BinaryOp, Dag, DiffVar, Node, NodeId, UnaryOp};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// `BlockSimulationType` (values as written to `Simulation_Type`).
pub mod sim {
    pub const EVALUATE_FORWARD: u8 = 1;
    pub const EVALUATE_BACKWARD: u8 = 2;
    pub const SOLVE_FORWARD_SIMPLE: u8 = 3;
    pub const SOLVE_BACKWARD_SIMPLE: u8 = 4;
    pub const SOLVE_TWO_BOUNDARIES_SIMPLE: u8 = 5;
    pub const SOLVE_FORWARD_COMPLETE: u8 = 6;
    pub const SOLVE_BACKWARD_COMPLETE: u8 = 7;
    pub const SOLVE_TWO_BOUNDARIES_COMPLETE: u8 = 8;
}

/// One block of `M_.block_structure(_stat).block(i)` (indices 0-based).
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    /// `Simulation_Type` (see [`sim`]).
    pub simulation_type: u8,
    /// `mfs`: size of the minimal feedback set (the last `mfs` equations/variables of the block).
    pub mfs: usize,
    /// `equation` / `variable`: recursive ones first, then the feedback ones.
    pub equations: Vec<usize>,
    pub variables: Vec<usize>,
    /// `is_linear` (dynamic model).
    pub linear: bool,
    /// Max lag/lead of the block's own variables inside the block.
    pub max_lag: i32,
    pub max_lead: i32,
}

/// `M_.block_structure` (dynamic) or `M_.block_structure_stat` (static), 0-based.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockStructure {
    pub blocks: Vec<Block>,
    pub variable_reordered: Vec<usize>,
    pub equation_reordered: Vec<usize>,
    /// `M_.block_structure.time_recursive` (dynamic model).
    pub time_recursive: bool,
    /// `incidence(k)`: (lead_lag, sorted (equation, variable) pairs of the endogenous Jacobian
    /// non-zeros at that lead/lag); the static model has one entry with lead_lag 0.
    pub incidence: Vec<(i32, Vec<(usize, usize)>)>,
}

/// The decompositions used by the block-wise solvers of `model(block)`
/// (`M_.block_structure_stat` / `M_.block_structure`).
#[derive(Clone, Debug, PartialEq)]
pub struct BlockSolve {
    pub statik: Option<BlockStructure>,
    pub dynamic: Option<BlockStructure>,
}

/// Settings of one `computingPassBlock` call.
#[derive(Clone, Copy, Debug)]
pub struct BlockSettings {
    pub dynamic: bool,
    /// `mfs` (dynamic model) or `static_mfs` (static model).
    pub mfs: i32,
    pub cutoff: f64,
    pub time_recursive: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EqType {
    Evaluate,
    EvaluateRenormalized,
    Solve,
}

/// The model as `ModelTree` sees it (static or dynamic version).
struct Side {
    dag: Dag,
    n: usize,
    lhs: Vec<NodeId>,
    rhs: Vec<NodeId>,
    /// `collectEndogenous` of each equation.
    endos: Vec<BTreeSet<(usize, i32)>>,
    /// `derivatives[1]` w.r.t. endogenous variables: (eq, var, lag) -> derivative.
    derivs: BTreeMap<(usize, usize, i32), NodeId>,
    hand_written: Vec<bool>,
    fake: usize,
}

/// In the static model, `STEADY_STATE(y_j)` is the variable `j + fake` (`fake` = number of
/// endogenous variables): it is `y_j` but a distinct node, as the preprocessor keeps the
/// `steady_state` operator (not simplified, derivative 1). `NO_FAKE` for the dynamic model.
const NO_FAKE: usize = usize::MAX / 4;

fn collect_endos(dag: &Dag, ids: &[NodeId], fake: usize, out: &mut BTreeSet<(usize, i32)>) {
    let mut leaves = Vec::new();
    for &id in ids {
        dag.collect_leaves(id, &mut leaves);
    }
    for l in leaves {
        if let Node::Endo(j, lag) | Node::SteadyState(j, lag) = l {
            let j = j as usize;
            out.insert((if j >= fake { j - fake } else { j }, lag));
        }
    }
}

fn contains_endo(dag: &Dag, id: NodeId, j: usize, lag: i32, fake: usize) -> bool {
    let mut s = BTreeSet::new();
    collect_endos(dag, &[id], fake, &mut s);
    s.contains(&(j, lag))
}

impl Side {
    fn new(m: &Model, dynamic: bool) -> Side {
        let mut dag = m.dag().clone();
        let n = m.n_equations();
        let ne = m.n_endo();
        let fake = if dynamic { NO_FAKE } else { ne };
        let (mut lhs, mut rhs) = (Vec::with_capacity(n), Vec::with_capacity(n));
        // static model: leads/lags dropped (DataTree simplifications apply, e.g. x - x(-1) = 0),
        // STEADY_STATE(y_j) kept as the distinct variable j + ne
        let mut memo = HashMap::new();
        let mut statik = |dag: &mut Dag, id: NodeId| to_static_dt(dag, id, ne, &mut memo);
        // Ramsey policy: the planner writes the model equations and FOCs as `lhs - rhs = 0` (the
        // auxiliary equations created afterwards keep their form)
        let ramsey = m.planner().is_some_and(|p| p.ramsey.is_some());
        for i in 0..n {
            if ramsey && i < m.orig_eq_nbr {
                let (l, r) = m.static_eqs.get(i).copied().flatten().filter(|_| !dynamic).unwrap_or(m.equations()[i]);
                let e = dt_binary(&mut dag, BinaryOp::Sub, l, r);
                let e = if dynamic { e } else { statik(&mut dag, e) };
                lhs.push(e);
                rhs.push(Dag::ZERO);
            } else if dynamic {
                let (l, r) = m.equations()[i];
                lhs.push(l);
                rhs.push(r);
            } else {
                let (l, r) = m.static_eqs.get(i).copied().flatten().unwrap_or(m.equations()[i]);
                let (l, r) = (statik(&mut dag, l), statik(&mut dag, r));
                lhs.push(l);
                rhs.push(r);
            }
        }
        let mut endos = Vec::with_capacity(n);
        for i in 0..n {
            let mut s = BTreeSet::new();
            collect_endos(&dag, &[lhs[i], rhs[i]], fake, &mut s);
            endos.push(s);
        }
        let no_rec = HashMap::new();
        let mut derivs = BTreeMap::new();
        let mut dmemo: HashMap<DiffVar, HashMap<NodeId, NodeId>> = HashMap::new();
        for i in 0..n {
            for &(j, lag) in &endos[i] {
                let v = DiffVar::Endo(j as u32, lag);
                let memo = dmemo.entry(v).or_default();
                let dl = dt_diff(&mut dag, lhs[i], j as u32, lag, fake, &no_rec, memo);
                let dr = dt_diff(&mut dag, rhs[i], j as u32, lag, fake, &no_rec, memo);
                let d = dt_binary(&mut dag, BinaryOp::Sub, dl, dr);
                if d != Dag::ZERO {
                    derivs.insert((i, j, lag), d);
                }
            }
        }
        let hand_written = (0..n).map(|i| i < m.orig_eq_nbr).collect();
        Side { dag, n, lhs, rhs, endos, derivs, hand_written, fake }
    }
}

/// `DataTree::Add*` for the static model (`ExprNode::toStatic` rebuilds every node in the
/// static DataTree): the simplification rules of the preprocessor, commutative operands in a
/// canonical order (DataTree sorts them by node index), so that e.g. `c/c(+1)` becomes 1.
fn dt_binary(dag: &mut Dag, op: BinaryOp, a: NodeId, b: NodeId) -> NodeId {
    let node = |dag: &Dag, x: NodeId| dag.node(x);
    let is_num = |dag: &Dag, x: NodeId, v: f64| dag.as_num(x) == Some(v);
    match op {
        BinaryOp::Add => {
            if b == Dag::ZERO {
                return a;
            }
            if a == Dag::ZERO {
                return b;
            }
            if let Node::Unary(UnaryOp::Neg, c) = node(dag, b) {
                return dt_binary(dag, BinaryOp::Sub, a, c);
            }
            if let Node::Unary(UnaryOp::Neg, c) = node(dag, a) {
                return dt_binary(dag, BinaryOp::Sub, b, c);
            }
            if let Node::Binary(BinaryOp::Sub, x, y) = node(dag, a)
                && y == b
            {
                return x;
            }
            if let Node::Binary(BinaryOp::Sub, x, y) = node(dag, b)
                && y == a
            {
                return x;
            }
            let (a, b) = if a > b { (b, a) } else { (a, b) };
            dag.binary(op, a, b)
        }
        BinaryOp::Sub => {
            if b == Dag::ZERO {
                return a;
            }
            if a == Dag::ZERO {
                return dag.neg(b);
            }
            if a == b {
                return Dag::ZERO;
            }
            if let Node::Unary(UnaryOp::Neg, c) = node(dag, b) {
                return dt_binary(dag, BinaryOp::Add, a, c);
            }
            if let Node::Binary(BinaryOp::Add, x, y) = node(dag, a) {
                if y == b {
                    return x;
                }
                if x == b {
                    return y;
                }
            }
            dag.binary(op, a, b)
        }
        BinaryOp::Mul => {
            if a == Dag::ZERO || b == Dag::ZERO {
                return Dag::ZERO;
            }
            if a == Dag::ONE {
                return b;
            }
            if b == Dag::ONE {
                return a;
            }
            if is_num(dag, a, -1.0) {
                return dag.neg(b);
            }
            if is_num(dag, b, -1.0) {
                return dag.neg(a);
            }
            if let Node::Binary(BinaryOp::Div, x, y) = node(dag, a)
                && y == b
            {
                return x;
            }
            if let Node::Binary(BinaryOp::Div, x, y) = node(dag, b)
                && y == a
            {
                return x;
            }
            let (a, b) = if a > b { (b, a) } else { (a, b) };
            dag.binary(op, a, b)
        }
        BinaryOp::Div => {
            if b == Dag::ONE {
                return a;
            }
            if a == Dag::ZERO {
                return Dag::ZERO;
            }
            if a == b {
                return Dag::ONE;
            }
            if let Node::Binary(BinaryOp::Div, x, y) = node(dag, b)
                && x == Dag::ONE
            {
                return dt_binary(dag, BinaryOp::Mul, a, y);
            }
            if let Node::Binary(BinaryOp::Mul, x, y) = node(dag, a) {
                if y == b {
                    return x;
                }
                if x == b {
                    return y;
                }
            }
            dag.binary(op, a, b)
        }
        BinaryOp::Pow => {
            if b == Dag::ZERO {
                return Dag::ONE;
            }
            if a == Dag::ZERO {
                return Dag::ZERO;
            }
            if a == Dag::ONE {
                return Dag::ONE;
            }
            dag.binary(op, a, b)
        }
        _ => dag.binary(op, a, b),
    }
}

/// `ExprNode::getDerivative` w.r.t. the endogenous `(j, lag)` with the preprocessor's
/// `composeDerivatives` formulas and [`dt_binary`] simplifications (so that the symbolic zeros,
/// hence the Jacobian incidence, are the preprocessor's). In the static model the variable
/// `j + fake` (`STEADY_STATE(y_j)`) has derivative 1 too.
fn dt_diff(
    dag: &mut Dag,
    id: NodeId,
    j: u32,
    lag: i32,
    fake: usize,
    rec: &HashMap<u32, NodeId>,
    memo: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Some(&r) = memo.get(&id) {
        return r;
    }
    let r = match dag.node(id) {
        Node::Endo(k, l) => {
            if (k == j && l == lag) || (fake != NO_FAKE && k as usize == j as usize + fake && l == lag) {
                Dag::ONE
            } else if l == 0
                && let Some(&r) = rec.get(&k)
            {
                // chain rule through a recursive variable (`getChainRuleDerivative`)
                dt_diff(dag, r, j, lag, fake, rec, memo)
            } else {
                Dag::ZERO
            }
        }
        Node::Num(_)
        | Node::Exo(..)
        | Node::Param(_)
        | Node::SteadyState(..)
        | Node::SteadyStateParamDeriv(..)
        | Node::SteadyStateParam2ndDeriv(..) => Dag::ZERO,
        Node::External(c) => {
            let args = dag.ext_call(c).args.clone();
            let dargs: Vec<NodeId> = args.iter().map(|&a| dt_diff(dag, a, j, lag, fake, rec, memo)).collect();
            dag.diff_external(c, &dargs, &mut |d, x, y| dt_binary(d, BinaryOp::Mul, x, y), &mut |d, x, y| {
                dt_binary(d, BinaryOp::Add, x, y)
            })
        }
        Node::Unary(op, a) => {
            let da = dt_diff(dag, a, j, lag, fake, rec, memo);
            if da == Dag::ZERO && op != UnaryOp::Neg {
                Dag::ZERO
            } else {
                let mul = |dag: &mut Dag, x, y| dt_binary(dag, BinaryOp::Mul, x, y);
                let div = |dag: &mut Dag, x, y| dt_binary(dag, BinaryOp::Div, x, y);
                match op {
                    UnaryOp::Neg => {
                        if da == Dag::ZERO {
                            Dag::ZERO
                        } else {
                            dag.neg(da)
                        }
                    }
                    UnaryOp::Exp => mul(dag, da, id),
                    UnaryOp::Log => div(dag, da, a),
                    UnaryOp::Log10 => {
                        let c = dag.num(std::f64::consts::LOG10_E);
                        let t = div(dag, da, a);
                        mul(dag, c, t)
                    }
                    UnaryOp::Cos => {
                        let s = dag.unary(UnaryOp::Sin, a);
                        let s = dag.neg(s);
                        mul(dag, da, s)
                    }
                    UnaryOp::Sin => {
                        let c = dag.unary(UnaryOp::Cos, a);
                        mul(dag, da, c)
                    }
                    UnaryOp::Tan => {
                        let t = mul(dag, id, id);
                        let t = dt_binary(dag, BinaryOp::Add, t, Dag::ONE);
                        mul(dag, da, t)
                    }
                    UnaryOp::Acos => {
                        let s = dag.unary(UnaryOp::Sin, id);
                        let t = div(dag, da, s);
                        dag.neg(t)
                    }
                    UnaryOp::Asin => {
                        let c = dag.unary(UnaryOp::Cos, id);
                        div(dag, da, c)
                    }
                    UnaryOp::Atan => {
                        let t = mul(dag, a, a);
                        let t = dt_binary(dag, BinaryOp::Add, Dag::ONE, t);
                        div(dag, da, t)
                    }
                    UnaryOp::Cosh => {
                        let s = dag.unary(UnaryOp::Sinh, a);
                        mul(dag, da, s)
                    }
                    UnaryOp::Sinh => {
                        let c = dag.unary(UnaryOp::Cosh, a);
                        mul(dag, da, c)
                    }
                    UnaryOp::Tanh => {
                        let t = mul(dag, id, id);
                        let t = dt_binary(dag, BinaryOp::Sub, Dag::ONE, t);
                        mul(dag, da, t)
                    }
                    UnaryOp::Acosh => {
                        let s = dag.unary(UnaryOp::Sinh, id);
                        div(dag, da, s)
                    }
                    UnaryOp::Asinh => {
                        let c = dag.unary(UnaryOp::Cosh, id);
                        div(dag, da, c)
                    }
                    UnaryOp::Atanh => {
                        let t = mul(dag, a, a);
                        let t = dt_binary(dag, BinaryOp::Sub, Dag::ONE, t);
                        div(dag, da, t)
                    }
                    UnaryOp::Sqrt => {
                        let t = dt_binary(dag, BinaryOp::Add, id, id);
                        div(dag, da, t)
                    }
                    UnaryOp::Cbrt => {
                        let e = dag.num(2.0 / 3.0);
                        let t = dt_binary(dag, BinaryOp::Pow, a, e);
                        let three = dag.num(3.0);
                        let t = mul(dag, three, t);
                        div(dag, da, t)
                    }
                    UnaryOp::Abs => {
                        let s = dag.unary(UnaryOp::Sign, a);
                        mul(dag, s, da)
                    }
                    UnaryOp::Sign => Dag::ZERO,
                    UnaryOp::Erf | UnaryOp::Erfc => {
                        let two = dag.num(2.0);
                        let t = dt_binary(dag, BinaryOp::Pow, a, two);
                        let t = dag.unary(UnaryOp::Exp, t);
                        let sp = dag.num(std::f64::consts::PI.sqrt());
                        let t = mul(dag, sp, t);
                        let t = div(dag, two, t);
                        let t = mul(dag, t, da);
                        if op == UnaryOp::Erfc { dag.neg(t) } else { t }
                    }
                }
            }
        }
        Node::Binary(op, a, b) => {
            let da = dt_diff(dag, a, j, lag, fake, rec, memo);
            let db = dt_diff(dag, b, j, lag, fake, rec, memo);
            if da == Dag::ZERO && db == Dag::ZERO {
                Dag::ZERO
            } else {
                let mul = |dag: &mut Dag, x, y| dt_binary(dag, BinaryOp::Mul, x, y);
                match op {
                    BinaryOp::Add => dt_binary(dag, BinaryOp::Add, da, db),
                    BinaryOp::Sub => dt_binary(dag, BinaryOp::Sub, da, db),
                    BinaryOp::Mul => {
                        let t11 = mul(dag, da, b);
                        let t12 = mul(dag, db, a);
                        dt_binary(dag, BinaryOp::Add, t11, t12)
                    }
                    BinaryOp::Div => {
                        if db != Dag::ZERO {
                            let t11 = mul(dag, da, b);
                            let t12 = mul(dag, db, a);
                            let t13 = dt_binary(dag, BinaryOp::Sub, t11, t12);
                            let t14 = mul(dag, b, b);
                            dt_binary(dag, BinaryOp::Div, t13, t14)
                        } else {
                            dt_binary(dag, BinaryOp::Div, da, b)
                        }
                    }
                    BinaryOp::Pow => {
                        if db == Dag::ZERO && dag.is_num_const(b) {
                            // NumConst exponent: b * a^(b-1)
                            let t11 = dt_binary(dag, BinaryOp::Sub, b, Dag::ONE);
                            let t12 = dt_binary(dag, BinaryOp::Pow, a, t11);
                            let t13 = mul(dag, b, t12);
                            mul(dag, da, t13)
                        } else if db == Dag::ZERO {
                            // other exponents: a PowerDeriv node (getPowerDeriv, exact 0 at a = 0
                            // for an integer exponent below the order)
                            let t = dag.binary(BinaryOp::PowerDeriv(1), a, b);
                            mul(dag, da, t)
                        } else {
                            let t11 = dag.unary(UnaryOp::Log, a);
                            let t12 = mul(dag, db, t11);
                            let t13 = mul(dag, da, b);
                            let t14 = dt_binary(dag, BinaryOp::Div, t13, a);
                            let t15 = dt_binary(dag, BinaryOp::Add, t12, t14);
                            mul(dag, t15, id)
                        }
                    }
                    BinaryOp::Max | BinaryOp::Min => {
                        let t11 = if op == BinaryOp::Max {
                            dag.binary(BinaryOp::Gt, a, b)
                        } else {
                            dag.binary(BinaryOp::Gt, b, a)
                        };
                        let t12 = mul(dag, t11, da);
                        let t13 = dt_binary(dag, BinaryOp::Sub, Dag::ONE, t11);
                        let t14 = mul(dag, t13, db);
                        dt_binary(dag, BinaryOp::Add, t14, t12)
                    }
                    _ => Dag::ZERO,
                }
            }
        }
    };
    memo.insert(id, r);
    r
}

/// `toStatic` of an expression into the static DataTree (see [`dt_binary`]); `STEADY_STATE(y_j)`
/// becomes the variable `j + ne` (see [`collect_endos`]).
fn to_static_dt(dag: &mut Dag, id: NodeId, ne: usize, memo: &mut HashMap<NodeId, NodeId>) -> NodeId {
    if let Some(&r) = memo.get(&id) {
        return r;
    }
    let r = match dag.node(id) {
        Node::Endo(j, _) => dag.endo(j as usize, 0),
        Node::SteadyState(j, _) => dag.endo(j as usize + ne, 0),
        Node::Exo(k, _) => dag.exo(k as usize, 0),
        Node::Num(_) | Node::Param(_) | Node::SteadyStateParamDeriv(..) | Node::SteadyStateParam2ndDeriv(..) => id,
        Node::External(c) => {
            let args = dag.ext_call(c).args.clone();
            let a2: Vec<NodeId> = args.iter().map(|&a| to_static_dt(dag, a, ne, memo)).collect();
            dag.rebuild_external(id, c, a2)
        }
        Node::Unary(op, a) => {
            let a = to_static_dt(dag, a, ne, memo);
            match op {
                UnaryOp::Exp if a == Dag::ZERO => Dag::ONE,
                UnaryOp::Log if a == Dag::ONE => Dag::ZERO,
                _ => dag.unary(op, a),
            }
        }
        Node::Binary(op, a, b) => {
            let a = to_static_dt(dag, a, ne, memo);
            let b = to_static_dt(dag, b, ne, memo);
            dt_binary(dag, op, a, b)
        }
    };
    memo.insert(id, r);
    r
}

/// Values of the preprocessor's `global_eval_context` (endogenous, exogenous, parameters).
#[derive(Clone, Debug, Default)]
pub struct EvalContext {
    pub y: Vec<f64>,
    pub x: Vec<f64>,
    pub p: Vec<f64>,
}

fn eval_memo(dag: &Dag, id: NodeId, c: &EvalContext, memo: &mut HashMap<NodeId, f64>) -> f64 {
    if let Some(&v) = memo.get(&id) {
        return v;
    }
    let v = match dag.node(id) {
        Node::Num(b) => f64::from_bits(b),
        Node::Endo(j, _) | Node::SteadyState(j, _) => c.y[j as usize % c.y.len().max(1)],
        Node::Exo(k, _) => c.x[k as usize],
        Node::Param(i) => c.p.get(i as usize).copied().unwrap_or(0.0),
        // parameter derivatives of STEADY_STATE never appear in the model equations
        Node::SteadyStateParamDeriv(..) | Node::SteadyStateParam2ndDeriv(..) => f64::NAN,
        Node::Unary(op, a) => op.apply(eval_memo(dag, a, c, memo)),
        Node::External(k) => {
            let e = dag.ext_call(k).clone();
            let v: Vec<f64> = e.args.iter().map(|&a| eval_memo(dag, a, c, memo)).collect();
            crate::model::external::get(e.fid).eval(e.d1, e.d2, &v)
        }
        Node::Binary(op, a, b) => {
            let (x, y) = (eval_memo(dag, a, c, memo), eval_memo(dag, b, c, memo));
            op.apply(x, y)
        }
    };
    memo.insert(id, v);
    v
}

/// `std::max` on doubles (`a < b ? b : a`).
fn cmax(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

type Jac = BTreeMap<(usize, usize), f64>;

/// `computeNormalization`: endo2eq, or `None` if some endogenous is unmatched.
fn normalization(n: usize, jac: &Jac) -> Option<Vec<usize>> {
    let edges: Vec<(usize, usize)> = jac.keys().map(|&(eq, endo)| (eq + n, endo)).collect();
    let mate = edmonds_matching(2 * n, &edges);
    let mut endo2eq = Vec::with_capacity(n);
    for m in &mate[..n] {
        endo2eq.push((*m)? - n);
    }
    Some(endo2eq)
}

fn non_singular_normalization(s: &mut Side, st: &BlockSettings, ctx: &EvalContext) -> Option<Vec<usize>> {
    let n = s.n;
    if st.time_recursive {
        // computeLeftHandSideSymbolicJacobian
        let mut jac = Jac::new();
        let mut ok = true;
        for eq in 0..n {
            if s.hand_written[eq] {
                let mut e = BTreeSet::new();
                collect_endos(&s.dag, &[s.lhs[eq]], s.fake, &mut e);
                e.retain(|&(_, l)| l == 0);
                if e.len() == 1 {
                    jac.insert((eq, e.iter().next().unwrap().0), 1.0);
                } else {
                    ok = false;
                    break;
                }
            } else {
                for &(endo, l) in &s.endos[eq] {
                    if l == 0 {
                        jac.insert((eq, endo), 1.0);
                    }
                }
            }
        }
        if ok && let Some(e) = normalization(n, &jac) {
            return Some(e);
        }
    }
    // evaluateAndReduceJacobian
    let mut cj = Jac::new();
    let mut memo = HashMap::new();
    for (&(eq, var, lag), &d) in &s.derivs {
        if lag != 0 {
            continue;
        }
        let val = eval_memo(&s.dag, d, ctx, &mut memo);
        if val.is_nan() || val.abs() >= st.cutoff {
            cj.insert((eq, var), val);
        }
    }
    let mut max_val = vec![0.0f64; n];
    for (&(eq, _), &v) in &cj {
        max_val[eq] = cmax(max_val[eq], v.abs());
    }
    let mut norm = cj.clone();
    for (&(eq, _), v) in norm.iter_mut() {
        *v /= max_val[eq];
    }
    let mut current = 0.99999999f64;
    let lower = 1e-19;
    let mut last_suppressed = 0usize;
    while current > lower {
        let mut above = Jac::new();
        let mut suppressed = 0usize;
        for (&k, &v) in &norm {
            if v.abs() > current.max(st.cutoff) {
                above.insert(k, v);
            } else {
                suppressed += 1;
            }
        }
        if suppressed != last_suppressed
            && let Some(e) = normalization(n, &above)
        {
            return Some(e);
        }
        last_suppressed = suppressed;
        current /= 2.0;
    }
    if let Some(e) = normalization(n, &norm) {
        return Some(e);
    }
    // symbolic normalization
    let mut sj = Jac::new();
    for eq in 0..n {
        for &(endo, l) in &s.endos[eq] {
            if l == 0 {
                sj.entry((eq, endo)).or_insert(1.0);
            }
        }
    }
    normalization(n, &sj)
}

/// `BinaryOpNode::normalizeEquation(var, 0)`: `Some(rhs)` of `var = rhs` if the equation can be
/// solved for the contemporaneous `var` by inverting the operators on its path.
fn normalize_equation(dag: &mut Dag, lhs: NodeId, rhs: NodeId, var: usize, fake: usize) -> Option<NodeId> {
    let mut cmemo: HashMap<NodeId, bool> = HashMap::new();
    fn contains(dag: &Dag, id: NodeId, var: u32, fake: usize, memo: &mut HashMap<NodeId, bool>) -> bool {
        if let Some(&b) = memo.get(&id) {
            return b;
        }
        let b = match dag.node(id) {
            Node::Endo(j, 0) => j == var || j as usize == var as usize + fake,
            Node::SteadyState(j, 0) => j == var,
            Node::Unary(_, a) => contains(dag, a, var, fake, memo),
            Node::External(c) => {
                let args = dag.ext_call(c).args.clone();
                args.iter().fold(false, |acc, &a| contains(dag, a, var, fake, memo) || acc)
            }
            Node::Binary(_, a, b) => {
                let x = contains(dag, a, var, fake, memo);
                contains(dag, b, var, fake, memo) || x
            }
            _ => false,
        };
        memo.insert(id, b);
        b
    }
    let v = var as u32;
    let (cl, cr) = (contains(dag, lhs, v, fake, &mut cmemo), contains(dag, rhs, v, fake, &mut cmemo));
    if cl == cr {
        return None;
    }
    let (mut node, mut r) = if cl { (lhs, rhs) } else { (rhs, lhs) };
    loop {
        match dag.node(node) {
            Node::Endo(j, 0) if j == v => return Some(r),
            Node::Unary(op, a) => {
                r = match op {
                    UnaryOp::Neg => dag.neg(r),
                    UnaryOp::Exp => {
                        if r == Dag::ZERO {
                            return None; // AddLog(0): the preprocessor aborts
                        }
                        dag.unary(UnaryOp::Log, r)
                    }
                    UnaryOp::Log => dag.unary(UnaryOp::Exp, r),
                    UnaryOp::Log10 => {
                        let ten = dag.num(10.0);
                        dt_binary(dag, BinaryOp::Pow, ten, r)
                    }
                    UnaryOp::Acos => dag.unary(UnaryOp::Cos, r),
                    UnaryOp::Asin => dag.unary(UnaryOp::Sin, r),
                    UnaryOp::Atan => dag.unary(UnaryOp::Tan, r),
                    UnaryOp::Sinh => dag.unary(UnaryOp::Asinh, r),
                    UnaryOp::Tanh => dag.unary(UnaryOp::Atanh, r),
                    UnaryOp::Acosh => dag.unary(UnaryOp::Cosh, r),
                    UnaryOp::Asinh => dag.unary(UnaryOp::Sinh, r),
                    UnaryOp::Atanh => dag.unary(UnaryOp::Tanh, r),
                    UnaryOp::Sqrt => {
                        let two = dag.num(2.0);
                        dt_binary(dag, BinaryOp::Pow, r, two)
                    }
                    UnaryOp::Cbrt => {
                        let three = dag.num(3.0);
                        dt_binary(dag, BinaryOp::Pow, r, three)
                    }
                    _ => return None,
                };
                node = a;
            }
            Node::Binary(op, a, b) => {
                let (ca, cb) = (contains(dag, a, v, fake, &mut cmemo), contains(dag, b, v, fake, &mut cmemo));
                if ca && cb {
                    return None;
                }
                let other = if ca { b } else { a };
                r = match op {
                    BinaryOp::Add => dt_binary(dag, BinaryOp::Sub, r, other),
                    BinaryOp::Sub => {
                        if ca {
                            dt_binary(dag, BinaryOp::Add, r, b)
                        } else {
                            dt_binary(dag, BinaryOp::Sub, a, r)
                        }
                    }
                    BinaryOp::Mul => {
                        if other == Dag::ZERO {
                            return None;
                        }
                        dt_binary(dag, BinaryOp::Div, r, other)
                    }
                    BinaryOp::Div => {
                        if ca {
                            dt_binary(dag, BinaryOp::Mul, r, b)
                        } else {
                            if r == Dag::ZERO {
                                return None;
                            }
                            dt_binary(dag, BinaryOp::Div, a, r)
                        }
                    }
                    BinaryOp::Pow => {
                        if ca || a == Dag::ONE {
                            return None;
                        }
                        if r == Dag::ZERO {
                            return None;
                        }
                        let lr = dag.unary(UnaryOp::Log, r);
                        let la = dag.unary(UnaryOp::Log, a);
                        dt_binary(dag, BinaryOp::Div, lr, la)
                    }
                    _ => return None,
                };
                node = if ca { a } else { b };
            }
            _ => return None,
        }
    }
}

struct BlockInfo {
    simulation_type: u8,
    first_equation: usize,
    size: usize,
    mfs_size: usize,
    linear: bool,
    max_endo_lag: i32,
    max_endo_lead: i32,
}

struct Decomp<'a> {
    s: &'a mut Side,
    st: BlockSettings,
    endo2eq: Vec<usize>,
    eq_b2o: Vec<usize>,
    endo_b2o: Vec<usize>,
    eq_o2b: Vec<usize>,
    endo_o2b: Vec<usize>,
    eq_type: Vec<EqType>,
    normalized: Vec<Option<NodeId>>,
    blocks: Vec<BlockInfo>,
    endo2block: Vec<usize>,
    eq2block: Vec<usize>,
}

impl Decomp<'_> {
    fn update_reverse(&mut self) {
        let n = self.s.n;
        self.eq_o2b = vec![0; n];
        self.endo_o2b = vec![0; n];
        for i in 0..n {
            self.endo_o2b[self.endo_b2o[i]] = i;
            self.eq_o2b[self.eq_b2o[i]] = i;
        }
    }

    fn prologue_epilogue(&mut self) -> (usize, usize) {
        let n = self.s.n;
        self.eq_b2o = (0..n).collect();
        self.endo_b2o = vec![0; n];
        for i in 0..n {
            self.endo_b2o[self.endo2eq[i]] = i;
        }
        let mut im = vec![false; n * n];
        for i in 0..n {
            for &(endo, lag) in &self.s.endos[i] {
                if !self.st.time_recursive || lag == 0 {
                    im[i * n + self.endo2eq[endo]] = true;
                }
            }
        }
        let mut prologue = 0;
        loop {
            let mut done = false;
            let mut newp = prologue;
            for i in prologue..n {
                let (mut nze, mut k) = (0, 0);
                for j in newp..n {
                    if im[i * n + j] {
                        nze += 1;
                        k = j;
                    }
                }
                if nze == 1 {
                    for j in 0..n {
                        im.swap(newp * n + j, i * n + j);
                    }
                    self.eq_b2o.swap(newp, i);
                    for j in 0..n {
                        im.swap(j * n + newp, j * n + k);
                    }
                    self.endo_b2o.swap(newp, k);
                    newp += 1;
                    done = true;
                }
            }
            prologue = newp;
            if !done {
                break;
            }
        }
        let mut epilogue = 0;
        loop {
            let mut done = false;
            let mut newe = epilogue;
            for i in prologue..n - epilogue {
                let (mut nze, mut k) = (0, 0);
                for j in prologue..n - newe {
                    if im[j * n + i] {
                        nze += 1;
                        k = j;
                    }
                }
                if nze == 1 {
                    let t = n - 1 - newe;
                    for j in 0..n {
                        im.swap(t * n + j, k * n + j);
                    }
                    self.eq_b2o.swap(t, k);
                    for j in 0..n {
                        im.swap(j * n + t, j * n + i);
                    }
                    self.endo_b2o.swap(t, i);
                    newe += 1;
                    done = true;
                }
            }
            epilogue = newe;
            if !done {
                break;
            }
        }
        self.update_reverse();
        (prologue, epilogue)
    }

    fn equation_types(&mut self) {
        let n = self.s.n;
        self.eq_type = vec![EqType::Solve; n];
        self.normalized = vec![None; n];
        for i in 0..n {
            let (eq, var) = (self.eq_b2o[i], self.endo_b2o[i]);
            let mut ty = EqType::Solve;
            let mut normalized = None;
            if let Some(&d) = self.s.derivs.get(&(eq, var, 0)) {
                let lhs_is_var = self.s.dag.node(self.s.lhs[eq]) == Node::Endo(var as u32, 0);
                if lhs_is_var && d == Dag::ONE {
                    ty = EqType::Evaluate;
                } else {
                    let not_in_deriv = !contains_endo(&self.s.dag, d, var, 0, self.s.fake);
                    let (l, r) = (self.s.lhs[eq], self.s.rhs[eq]);
                    if let Some(nr) = normalize_equation(&mut self.s.dag, l, r, var, self.s.fake) {
                        normalized = Some(nr);
                        if (self.st.mfs == 2 && not_in_deriv) || self.st.mfs == 3 {
                            ty = EqType::EvaluateRenormalized;
                        }
                    }
                }
            }
            self.eq_type[eq] = ty;
            self.normalized[eq] = normalized;
        }
    }

    /// `getVariableLeadLagByBlock`: (equation_lag_lead, variable_lag_lead).
    fn lead_lag_by_block(&self) -> (Vec<(i32, i32)>, Vec<(i32, i32)>) {
        let n = self.s.n;
        let (mut var_ll, mut eq_ll) = (vec![(0, 0); n], vec![(0, 0); n]);
        for eq in 0..n {
            for &(endo, lag) in &self.s.endos[eq] {
                if self.endo2block[endo] == self.eq2block[eq] {
                    var_ll[endo].0 = var_ll[endo].0.max(-lag);
                    var_ll[endo].1 = var_ll[endo].1.max(lag);
                    eq_ll[eq].0 = eq_ll[eq].0.max(-lag);
                    eq_ll[eq].1 = eq_ll[eq].1.max(lag);
                }
            }
        }
        (eq_ll, var_ll)
    }

    fn dynamic_structure(&mut self, blk: usize) {
        let (first, size) = (self.blocks[blk].first_equation, self.blocks[blk].size);
        let (mut ml, mut md) = (0, 0);
        for eq in 0..size {
            for &(endo, lag) in &self.s.endos[self.eq_b2o[first + eq]] {
                if self.endo2block[endo] == blk {
                    ml = ml.max(-lag);
                    md = md.max(lag);
                }
            }
        }
        self.blocks[blk].max_endo_lag = ml;
        self.blocks[blk].max_endo_lead = md;
    }

    fn simulation_type(&mut self, blk: usize) {
        let b = &self.blocks[blk];
        let ty = if b.max_endo_lag > 0 && b.max_endo_lead > 0 {
            if b.size == 1 { sim::SOLVE_TWO_BOUNDARIES_SIMPLE } else { sim::SOLVE_TWO_BOUNDARIES_COMPLETE }
        } else if b.size > 1 {
            if b.max_endo_lead > 0 { sim::SOLVE_BACKWARD_COMPLETE } else { sim::SOLVE_FORWARD_COMPLETE }
        } else {
            let t = self.eq_type[self.eq_b2o[b.first_equation]];
            let can_eval = t == EqType::Evaluate || t == EqType::EvaluateRenormalized;
            if b.max_endo_lead > 0 {
                if can_eval { sim::EVALUATE_BACKWARD } else { sim::SOLVE_BACKWARD_SIMPLE }
            } else if can_eval {
                sim::EVALUATE_FORWARD
            } else {
                sim::SOLVE_FORWARD_SIMPLE
            }
        };
        self.blocks[blk].simulation_type = ty;
    }

    fn block_decomposition(&mut self, prologue: usize, epilogue: usize) {
        let nb_var = self.s.n;
        let nb_simvars = nb_var - prologue - epilogue;
        let mut g = DepGraph::new(nb_simvars);
        // computeSymbolicJacobian(time_recursive)
        let mut sym: BTreeSet<(usize, usize)> = BTreeSet::new();
        for eq in 0..nb_var {
            for &(endo, lag) in &self.s.endos[eq] {
                if !self.st.time_recursive || lag == 0 {
                    sym.insert((eq, endo));
                }
            }
        }
        let inside = |p: usize| p >= prologue && p < nb_var - epilogue;
        for &(eq, endo) in &sym {
            if inside(self.eq_o2b[eq]) && inside(self.endo_o2b[endo]) && eq != self.endo2eq[endo] {
                g.add_edge_pos(self.eq_o2b[self.endo2eq[endo]] - prologue, self.eq_o2b[eq] - prologue);
            }
        }
        let (num_simblocks, simvar2simblock) = g.sorted_sccs();
        let num_blocks = prologue + num_simblocks + epilogue;
        self.blocks = (0..num_blocks)
            .map(|_| BlockInfo {
                simulation_type: 0,
                first_equation: 0,
                size: 0,
                mfs_size: 0,
                linear: true,
                max_endo_lag: 0,
                max_endo_lead: 0,
            })
            .collect();
        self.endo2block = vec![0; nb_var];
        self.eq2block = vec![0; nb_var];
        for blk in 0..num_blocks {
            if blk < prologue || blk >= num_blocks - epilogue {
                let var_eq = if blk < prologue { blk } else { blk - num_simblocks + nb_simvars };
                self.blocks[blk].size = 1;
                self.blocks[blk].mfs_size = 1;
                self.blocks[blk].first_equation = var_eq;
                self.endo2block[self.endo_b2o[var_eq]] = blk;
                self.eq2block[self.eq_b2o[var_eq]] = blk;
            }
        }
        let mut simblock2simvars = vec![Vec::new(); num_simblocks];
        for (i, &sb) in simvar2simblock.iter().enumerate() {
            simblock2simvars[sb].push(i);
            let blk = prologue + sb;
            self.blocks[blk].size += 1;
            self.endo2block[self.endo_b2o[prologue + i]] = blk;
            self.eq2block[self.eq_b2o[prologue + i]] = blk;
        }
        let (eq_ll, var_ll) = self.lead_lag_by_block();
        for i in 0..nb_simvars {
            let (e, v) = (self.eq_b2o[i + prologue], self.endo_b2o[i + prologue]);
            if self.eq_type[e] == EqType::Solve
                || (!self.st.time_recursive && (var_ll[v].0 > 0 || var_ll[v].1 > 0 || eq_ll[e].0 > 0 || eq_ll[e].1 > 0))
                || self.st.mfs == 0
            {
                g.add_edge_pos(i, i);
            }
        }
        let (old_eq, old_endo) = (self.eq_b2o.clone(), self.endo_b2o.clone());
        let mut ordidx = prologue;
        for blk in prologue..prologue + num_simblocks {
            self.blocks[blk].first_equation =
                if blk == 0 { 0 } else { self.blocks[blk - 1].first_equation + self.blocks[blk - 1].size };
            let sub = g.extract_subgraph(&simblock2simvars[blk - prologue]);
            let fb = sub.minimal_feedback_set();
            self.blocks[blk].mfs_size = fb.len();
            let rec = sub.reorder_recursive(&fb);
            for &vtx in &rec {
                let simvar = sub.index1[vtx];
                self.eq_b2o[ordidx] = old_eq[simvar + prologue];
                self.endo_b2o[ordidx] = old_endo[simvar + prologue];
                ordidx += 1;
            }
            for ll in [(0, 0), (1, 0), (1, 1), (0, 1)] {
                for &vtx in &fb {
                    let simvar = sub.index1[vtx];
                    if var_ll[old_endo[simvar + prologue]] == ll {
                        self.eq_b2o[ordidx] = old_eq[simvar + prologue];
                        self.endo_b2o[ordidx] = old_endo[simvar + prologue];
                        ordidx += 1;
                    }
                }
            }
        }
        self.update_reverse();
        for blk in 0..self.blocks.len() {
            self.dynamic_structure(blk);
            self.simulation_type(blk);
        }
    }

    fn reduce(&mut self) {
        let mut blk = 1;
        while blk < self.blocks.len() {
            if self.blocks[blk].size == 1 {
                let eq = self.eq_b2o[self.blocks[blk].first_equation];
                let endos = &self.s.endos[eq];
                let prev = &self.blocks[blk - 1];
                let (mut is_lag, mut is_lead) = (false, false);
                for var in 0..prev.size {
                    let v = self.endo_b2o[prev.first_equation + var];
                    is_lag = is_lag || endos.contains(&(v, -1));
                    is_lead = is_lead || endos.contains(&(v, 1));
                }
                let (tp, tc) = (prev.simulation_type, self.blocks[blk].simulation_type);
                if (tp == sim::EVALUATE_FORWARD && tc == sim::EVALUATE_FORWARD && !is_lead)
                    || (tp == sim::EVALUATE_BACKWARD && tc == sim::EVALUATE_BACKWARD && !is_lag)
                {
                    self.blocks[blk - 1].size += 1;
                    self.blocks[blk - 1].mfs_size = self.blocks[blk - 1].size;
                    self.dynamic_structure(blk - 1);
                    self.blocks.remove(blk);
                    for b in self.endo2block.iter_mut().chain(self.eq2block.iter_mut()) {
                        if *b >= blk {
                            *b -= 1;
                        }
                    }
                    continue;
                }
            }
            blk += 1;
        }
    }

    /// `DynamicModel::computeChainRuleJacobian` + `determineLinearBlocks`.
    fn linear_blocks(&mut self) {
        for blk in 0..self.blocks.len() {
            let ty = self.blocks[blk].simulation_type;
            let one_boundary = matches!(
                ty,
                sim::SOLVE_BACKWARD_SIMPLE
                    | sim::SOLVE_BACKWARD_COMPLETE
                    | sim::SOLVE_FORWARD_SIMPLE
                    | sim::SOLVE_FORWARD_COMPLETE
            );
            let two_boundaries = matches!(ty, sim::SOLVE_TWO_BOUNDARIES_SIMPLE | sim::SOLVE_TWO_BOUNDARIES_COMPLETE);
            if !one_boundary && !two_boundaries {
                continue;
            }
            let b = &self.blocks[blk];
            let (first, size, nb_rec) = (b.first_equation, b.size, b.size - b.mfs_size);
            let (lo, hi) = if self.st.time_recursive { (0, 0) } else { (-b.max_endo_lag, b.max_endo_lead) };
            let eqs: Vec<usize> = self.eq_b2o[first..first + size].to_vec();
            let vars: Vec<usize> = self.endo_b2o[first..first + size].to_vec();
            let eq_id = |k: usize| eqs[k];
            let var_id = |k: usize| vars[k];
            // determineBlockDerivativesType: 0 standard, 1 chain rule, 2 normalized chain rule
            let mut dtype: BTreeMap<(i32, usize, usize), u8> = BTreeMap::new();
            for lag in lo..=hi {
                for eq in 0..size {
                    let endos = &self.s.endos[eq_id(eq)];
                    for var in 0..size {
                        if endos.contains(&(var_id(var), lag)) {
                            if self.eq_type[eq_id(eq)] == EqType::EvaluateRenormalized && eq < nb_rec {
                                dtype.insert((lag, eq, var), 2);
                            } else {
                                dtype.entry((lag, eq, var)).or_insert(0);
                            }
                            if var < nb_rec {
                                for fv in nb_rec..size {
                                    if dtype.contains_key(&(lag, var, fv)) {
                                        dtype.insert((lag, eq, fv), 1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let mut rec: HashMap<u32, NodeId> = HashMap::new();
            for i in 0..nb_rec {
                let e = eq_id(i);
                let r = if self.eq_type[e] == EqType::EvaluateRenormalized {
                    self.normalized[e].unwrap_or(self.s.rhs[e])
                } else {
                    self.s.rhs[e]
                };
                rec.insert(var_id(i) as u32, r);
            }
            let mut linear = true;
            let mut memos: HashMap<DiffVar, HashMap<NodeId, NodeId>> = HashMap::new();
            'outer: for (&(lag, eq, var), &t) in &dtype {
                // only the feedback equations' derivatives are kept (blocks_derivatives), and
                // determineLinearBlocks reads only the contemporaneous ones of one-boundary blocks
                // (normalized chain-rule derivatives belong to recursive equations: never kept)
                if eq < nb_rec || (one_boundary && lag != 0) {
                    continue;
                }
                let (eo, vo) = (eq_id(eq), var_id(var));
                let dv = DiffVar::Endo(vo as u32, lag);
                let d = if t == 0 {
                    self.s.derivs.get(&(eo, vo, lag)).copied().unwrap_or(Dag::ZERO)
                } else {
                    let (l, r, fake) = (self.s.lhs[eo], self.s.rhs[eo], self.s.fake);
                    let memo = memos.entry(dv).or_default();
                    let dl = dt_diff(&mut self.s.dag, l, vo as u32, lag, fake, &rec, memo);
                    let dr = dt_diff(&mut self.s.dag, r, vo as u32, lag, fake, &rec, memo);
                    dt_binary(&mut self.s.dag, BinaryOp::Sub, dl, dr)
                };
                if d == Dag::ZERO {
                    continue;
                }
                let mut e = BTreeSet::new();
                collect_endos(&self.s.dag, &[d], self.s.fake, &mut e);
                for &bv in &vars {
                    if e.contains(&(bv, lag)) {
                        linear = false;
                        break 'outer;
                    }
                }
            }
            self.blocks[blk].linear = linear;
        }
    }
}

/// The symbolic non-zeros (equation, variable) of the static model's Jacobian, with the
/// preprocessor's simplifications (so `(1/(1+b))*(x + b*x)` w.r.t. `x` is 1 and `x - x` is 0):
/// `static_g1_sparse_*` of the JSON output.
pub fn static_jacobian_incidence(m: &Model) -> Vec<(usize, usize)> {
    let side = Side::new(m, false);
    let mut pat: Vec<(usize, usize)> = side.derivs.keys().map(|&(i, j, _)| (i, j)).collect();
    pat.dedup();
    pat
}

/// `computingPassBlock` on the dynamic (`settings.dynamic`) or static model. `None` when the
/// model cannot be normalized (no block structure is written then).
pub fn compute(m: &Model, settings: &BlockSettings, ctx: &EvalContext) -> Option<BlockStructure> {
    let mut side = Side::new(m, settings.dynamic);
    let n = side.n;
    if n == 0 || n != m.n_endo() {
        return None;
    }
    let endo2eq = non_singular_normalization(&mut side, settings, ctx)?;
    let mut d = Decomp {
        s: &mut side,
        st: *settings,
        endo2eq,
        eq_b2o: Vec::new(),
        endo_b2o: Vec::new(),
        eq_o2b: Vec::new(),
        endo_o2b: Vec::new(),
        eq_type: Vec::new(),
        normalized: Vec::new(),
        blocks: Vec::new(),
        endo2block: Vec::new(),
        eq2block: Vec::new(),
    };
    let (prologue, epilogue) = d.prologue_epilogue();
    d.equation_types();
    d.block_decomposition(prologue, epilogue);
    d.reduce();
    if settings.dynamic {
        d.linear_blocks();
    }
    let blocks = d
        .blocks
        .iter()
        .map(|b| Block {
            simulation_type: b.simulation_type,
            mfs: b.mfs_size,
            equations: (0..b.size).map(|k| d.eq_b2o[b.first_equation + k]).collect(),
            variables: (0..b.size).map(|k| d.endo_b2o[b.first_equation + k]).collect(),
            linear: b.linear,
            max_lag: b.max_endo_lag,
            max_lead: b.max_endo_lead,
        })
        .collect();
    let mut by_lag: BTreeMap<i32, BTreeSet<(usize, usize)>> = BTreeMap::new();
    for &(eq, var, lag) in d.s.derivs.keys() {
        by_lag.entry(if settings.dynamic { lag } else { 0 }).or_default().insert((eq, var));
    }
    Some(BlockStructure {
        blocks,
        variable_reordered: d.endo_b2o.clone(),
        equation_reordered: d.eq_b2o.clone(),
        time_recursive: settings.time_recursive,
        incidence: by_lag.into_iter().map(|(l, s)| (l, s.into_iter().collect())).collect(),
    })
}

impl BlockStructure {
    /// The block containing endogenous variable `v`.
    pub fn block_of_variable(&self, v: usize) -> Option<usize> {
        self.blocks.iter().position(|b| b.variables.contains(&v))
    }

    /// JSON in the layout of the MATLAB structure (1-based indices), for dumps and tests.
    pub fn to_json(&self, dynamic: bool) -> serde_json::Value {
        use serde_json::json;
        let one = |v: &[usize]| v.iter().map(|x| x + 1).collect::<Vec<_>>();
        let blocks: Vec<_> = self
            .blocks
            .iter()
            .map(|b| {
                let mut o = json!({
                    "Simulation_Type": b.simulation_type,
                    "endo_nbr": b.equations.len(),
                    "mfs": b.mfs,
                    "equation": one(&b.equations),
                    "variable": one(&b.variables),
                });
                if dynamic {
                    o["is_linear"] = json!(b.linear);
                }
                o
            })
            .collect();
        let incidence: Vec<_> = self
            .incidence
            .iter()
            .map(|(l, s)| json!({"lead_lag": l, "sparse_IM": s.iter().map(|&(e, v)| [e + 1, v + 1]).collect::<Vec<_>>()}))
            .collect();
        let mut o = json!({
            "block": blocks,
            "variable_reordered": one(&self.variable_reordered),
            "equation_reordered": one(&self.equation_reordered),
            "incidence": incidence,
        });
        if dynamic {
            o["time_recursive"] = json!(self.time_recursive);
        }
        o
    }
}

/// The `model(...)` / `model_options(...)` options that drive the block decomposition.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelOptions {
    pub block: bool,
    pub no_static: bool,
    pub mfs: Option<i32>,
    pub static_mfs: Option<i32>,
    pub cutoff: Option<f64>,
    /// Parameters whose last assignment the preprocessor cannot evaluate (it uses a mod-file
    /// local variable or an unknown parameter): 0 in its eval context.
    pub unknown_params: BTreeSet<usize>,
    /// Parameter values after every assignment of the file (`M_.params` is the value when the
    /// first command runs).
    pub final_params: Vec<f64>,
}
