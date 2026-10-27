// SPDX-License-Identifier: GPL-3.0-or-later
//! Emulation of the preprocessor's `DataTree` and of the part of `ModFile::transformPass` that
//! creates auxiliary variables (stream PP).
//!
//! Dynare names some auxiliaries after the internal index of an expression node
//! (`AUX_ENDO_LEAD_<idx>`, `AUX_EXO_LEAD_<idx>`, `AUX_DIFF_<idx>`, `AUX_EXPECT_LAG_1_<idx>`, ...),
//! so reproducing `M_.endo_names` requires creating exactly the nodes the preprocessor creates,
//! in the same order. This module is a port of that machinery:
//!
//! * `DataTree::Add*` (hash-consing, the simplification rules, constant folding through
//!   `eval` and `AddPossiblyNegativeConstant`, numerical constants interned by their text,
//!   commutative operands ordered by node index),
//! * the parser's node creation order (`DynareBison.yy` reduces post-order, left to right; a
//!   lead/lag `x(-2)` first builds the expression `-2`),
//! * `ParsingDriver::add_model_equal` / `DynamicModel::addOccbinEquation` / `[static]`
//!   equations, then `DynamicModel::{transformPredeterminedVariables, simplifyEquations,
//!   substituteModelLocalVariables, substituteDiff, substituteExpectation,
//!   substituteEndoLeadGreaterThanTwo, substituteExoLead, substituteEndoLagGreaterThanTwo,
//!   substituteExoLag}` (`ExprNode.cc`).
//!
//! The transformed equations are then converted to the crate's [`Dag`], so the model that is
//! differentiated and compiled is the preprocessor's (same equations, same aux equations in the
//! same order, same names). Ramsey models too: their FOCs are built here (`dtree_ramsey.rs`).

use super::AuxVar;
use super::expr::{BinaryOp, Dag, NodeId, UnaryOp};
use crate::modfile::{Ast, SpecialOp, special_to_node};
use anyhow::{Result, anyhow, bail};
use std::collections::{BTreeMap, BTreeSet, HashMap};

type Id = u32;

// ---- balanced growth and var(log) (dtree_q1.rs)
#[path = "dtree_q1.rs"]
mod q1;
#[path = "dtree_ramsey.rs"]
mod ramsey;
// ---- heterogeneity: the model(heterogeneity=...) block (dtree_het.rs)
#[path = "dtree_het.rs"]
mod het;
pub(crate) use het::{HetAuxOut, HetInput, het_transform};
pub(crate) use ramsey::PlannerIn;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum UOp {
    Uminus,
    F(UnaryOp),
    SteadyState,
    Expectation,
    Diff,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum BOp {
    Plus,
    Minus,
    Times,
    Divide,
    Power,
    Equal,
    Max,
    Min,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    EqualEqual,
    Different,
    /// `powerDeriv` (`getPowerDeriv(x, p, k)`), created by the derivatives of the Ramsey FOCs.
    PowerDeriv(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TOp {
    Normcdf,
    Normpdf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum N {
    Num(u32),
    Var(u32, i32),
    /// op, arg, expectation information set
    Un(UOp, Id, i32),
    Bin(BOp, Id, Id),
    Tri(TOp, Id, Id, Id),
    /// `ExternalFunctionNode`: function id (`model::external`), argument list ([`Tree::ext_args`]).
    Ext(u32, u32),
}

/// Symbol types the emulation distinguishes (`SymbolType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SymType {
    Endo,
    Exo,
    Param,
    Local,
    /// Removed by `var_remove` / `model_remove` (`excludedVariable`).
    Excluded,
    // ---- `trend_var` / `log_trend_var` (index: position among the trend variables)
    Trend,
    LogTrend,
    // ---- heterogeneity: aggregate variables seen from a model(heterogeneity=...) block
    AggEndo,
    AggExo,
}

#[derive(Clone, Debug)]
pub(crate) struct Sym {
    pub ty: SymType,
    /// Type-specific index (endogenous / exogenous / parameter index; locals: unused).
    pub index: usize,
}

/// One statement of a `model` block, as parsed (input of the replay).
#[derive(Clone, Debug)]
pub enum ModelItem {
    /// `# name = expr;` (`symb_id` of the local).
    Local { symb_id: usize, ast: Ast, nums: Vec<String> },
    /// `[tags] lhs = rhs;` (`rhs` None: `lhs;`). `nums` is the text of every numerical
    /// constant of lhs then rhs, in source order.
    Equation { lhs: Ast, rhs: Option<Ast>, nums: Vec<String>, tags: Vec<(String, String)>, line: usize },
    /// `model_remove(...)` / the removal part of `model_replace(...)`: tag sets.
    Remove { selection: Vec<Vec<(String, String)>> },
    // ---- balanced-growth declarations, parsed into the dynamic model's tree at
    // their place (`begin_model()` / `begin_trend()` in DynareBison.yy)
    /// `trend_var(growth_factor = ast) vars;` (`log`: `log_trend_var(log_growth_factor = ...)`).
    Trend { vars: Vec<String>, ast: Ast, nums: Vec<String>, log: bool },
    /// `var(deflator = ast) vars;` (`log`: `log_deflator`).
    Deflator { vars: Vec<String>, ast: Ast, nums: Vec<String>, log: bool },
}

/// What the transform pass produces.
pub(crate) struct Transformed {
    pub equations: Vec<(NodeId, NodeId)>,
    pub tags: Vec<Vec<(String, String)>>,
    /// Static-model residual source (dynamic form) replacing equation i (`[dynamic]` equations
    /// get the `[static]` ones in order).
    pub static_override: Vec<Option<NodeId>>,
    /// model_info: the same `[static]` equations as (lhs, rhs).
    pub static_eqs: Vec<Option<(NodeId, NodeId)>>,
    /// New endogenous names (aux variables, in creation order) and their `M_.aux_vars` entries
    /// with the aux equation's right-hand side.
    pub aux: Vec<(String, AuxVar, NodeId)>,
    /// For `diff` / `EXPECTATION` aux variables: (function prefix, argument) printed as
    /// `orig_expr` (`diff(x)`, `EXPECTATION(-1)(x(1))`).
    pub shown: Vec<Option<(String, NodeId)>>,
    pub orig_max_lead: i32,
    pub orig_max_lag: i32,
    /// `M_.orig_maximum_lag_with_diffs_expanded` (`DynamicModel::setLeadsLagsOrig`).
    pub orig_max_lag_with_diffs: i32,
    /// Ramsey first-order conditions (before the aux substitution), as lhs expressions.
    pub ramsey_focs: Vec<NodeId>,
}

struct AuxRec {
    symb: u32,
    name: String,
    ty: u8,
    orig_symb: Option<u32>,
    orig_lead_lag: Option<i32>,
    /// Right-hand side of the aux equation at creation (`aux_equations`).
    rhs: Id,
    /// `M_.aux_vars(k).orig_expr` when it is not `rhs`: the `diff` / `EXPECTATION` node.
    shown: Option<Id>,
}

#[derive(Clone)]
struct Equation {
    eq: Id,
    tags: Vec<(String, String)>,
    /// source line (`equations_lineno`), 0 for auxiliary equations.
    line: usize,
}

type SubstTable = HashMap<Id, Id>;
/// Leaf rule of [`Tree::recurse`]: `(tree, node, symbol, lag) -> replacement`.
type LeafFn<'a> = dyn FnMut(&mut Tree, Id, u32, i32) -> Result<Option<Id>> + 'a;
/// `lag_equivalence_table_t`: class representative -> (index -> node).
type LagEquiv = HashMap<Id, BTreeMap<i32, Id>>;

/// `%.16g`, as `AddPossiblyNegativeConstant` (`ostream << setprecision(16)`).
fn fmt_g16(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    let s = format!("{:.15e}", v);
    let (mant, exp) = s.split_once('e').unwrap();
    let x: i32 = exp.parse().unwrap();
    if (-4..16).contains(&x) {
        let dec = (15 - x).max(0) as usize;
        let mut f = format!("{:.*}", dec, v);
        if f.contains('.') {
            while f.ends_with('0') {
                f.pop();
            }
            if f.ends_with('.') {
                f.pop();
            }
        }
        f
    } else {
        let mut m = mant.to_string();
        if m.contains('.') {
            while m.ends_with('0') {
                m.pop();
            }
            if m.ends_with('.') {
                m.pop();
            }
        }
        format!("{m}e{}{:02}", if x < 0 { '-' } else { '+' }, x.abs())
    }
}

fn eval_uop(op: UOp, v: f64) -> Option<f64> {
    match op {
        UOp::Uminus => Some(-v),
        UOp::F(UnaryOp::Sign) => Some(if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }),
        UOp::F(f) => Some(f.apply(v)),
        UOp::SteadyState => Some(v),
        UOp::Expectation | UOp::Diff => None,
    }
}

fn eval_bop(op: BOp, a: f64, b: f64) -> Option<f64> {
    let t = |c: bool| if c { 1.0 } else { 0.0 };
    Some(match op {
        BOp::Plus => a + b,
        BOp::Minus => a - b,
        BOp::Times => a * b,
        BOp::Divide => a / b,
        BOp::Power => a.powf(b),
        BOp::Max => {
            if a < b {
                b
            } else {
                a
            }
        }
        BOp::Min => {
            if a > b {
                b
            } else {
                a
            }
        }
        BOp::Less => t(a < b),
        BOp::Greater => t(a > b),
        BOp::LessEqual => t(a <= b),
        BOp::GreaterEqual => t(a >= b),
        BOp::EqualEqual => t(a == b),
        BOp::Different => t(a != b),
        BOp::PowerDeriv(k) => crate::model::expr::get_power_deriv(a, b, k),
        BOp::Equal => return None,
    })
}

fn eval_top(op: TOp, a: f64, b: f64, c: f64) -> f64 {
    match op {
        TOp::Normcdf => 0.5 * (1.0 + UnaryOp::Erf.apply((a - b) / c / std::f64::consts::SQRT_2)),
        TOp::Normpdf => 1.0 / (c * (2.0 * std::f64::consts::PI).sqrt() * (((a - b) / c).powi(2) / 2.0).exp()),
    }
}

/// Which `max*Lead/Lag` quantity.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Mx {
    EndoLead,
    ExoLead,
    EndoLag,
    ExoLag,
    Lead,
    /// `maxLagWithDiffsExpanded()`: the maximum lag with each `diff` counting one more lag.
    LagDiffs,
}

struct Tree {
    nodes: Vec<N>,
    map: HashMap<N, Id>,
    consts: Vec<(String, f64)>,
    const_ids: HashMap<String, u32>,
    zero: Id,
    one: Id,
    two: Id,
    three: Id,
    minus_one: Id,
    nan: Id,
    inf: Id,
    minus_inf: Id,
    pi: Id,
    syms: Vec<Sym>,
    names: Vec<String>,
    n_endo: usize,
    /// `local_variables_table` (ordered by symbol id, as the C++ `std::map`).
    locals: BTreeMap<u32, Id>,
    predetermined: BTreeSet<u32>,
    equations: Vec<Equation>,
    static_only: Vec<Equation>,
    aux: Vec<AuxRec>,
    mx_cache: HashMap<(Mx, Id), i32>,
    /// Argument lists of the `N::Ext` nodes.
    ext_lists: Vec<Vec<Id>>,
    ext_list_ids: HashMap<Vec<Id>, u32>,
    /// `trend_symbols_map`: trend variable -> growth factor.
    trend_map: BTreeMap<u32, Id>,
    /// `nonstationary_symbols_map`: variable -> (log deflator, deflator).
    nonstationary: BTreeMap<u32, (bool, Id)>,
    /// Converting the trend model for the balanced-growth test: trend variable k -> parameter
    /// `offset + k`.
    trend_param_offset: Option<usize>,
    /// heterogeneity: offsets of the aggregate endogenous / exogenous variables in the
    /// converted [`Dag`] (after the heterogeneous ones).
    agg_offsets: (usize, usize),
}

impl Tree {
    fn new(syms: Vec<Sym>, names: Vec<String>, n_endo: usize) -> Self {
        let mut t = Tree {
            nodes: Vec::new(),
            map: HashMap::new(),
            consts: Vec::new(),
            const_ids: HashMap::new(),
            zero: 0,
            one: 0,
            two: 0,
            three: 0,
            minus_one: 0,
            nan: 0,
            inf: 0,
            minus_inf: 0,
            pi: 0,
            syms,
            names,
            n_endo,
            locals: BTreeMap::new(),
            predetermined: BTreeSet::new(),
            equations: Vec::new(),
            static_only: Vec::new(),
            aux: Vec::new(),
            mx_cache: HashMap::new(),
            ext_lists: Vec::new(),
            ext_list_ids: HashMap::new(),
            trend_map: BTreeMap::new(),
            nonstationary: BTreeMap::new(),
            trend_param_offset: None,
            agg_offsets: (0, 0),
        };
        // DataTree::initConstants
        t.zero = t.add_num("0");
        t.one = t.add_num("1");
        t.two = t.add_num("2");
        t.three = t.add_num("3");
        t.minus_one = t.add_uminus(t.one);
        t.nan = t.add_num("NaN");
        t.inf = t.add_num("Inf");
        t.minus_inf = t.add_uminus(t.inf);
        t.pi = t.add_num("3.141592653589793");
        t
    }

    fn node(&self, id: Id) -> N {
        self.nodes[id as usize]
    }

    fn intern(&mut self, n: N) -> Id {
        if let Some(&id) = self.map.get(&n) {
            return id;
        }
        let id = self.nodes.len() as Id;
        self.nodes.push(n);
        self.map.insert(n, id);
        id
    }

    fn sym_ty(&self, s: u32) -> SymType {
        self.syms[s as usize].ty
    }

    // ---- DataTree::Add* -------------------------------------------------------------------
    fn add_num(&mut self, text: &str) -> Id {
        let cid = match self.const_ids.get(text) {
            Some(&c) => c,
            None => {
                let v = match text {
                    "NaN" => f64::NAN,
                    "Inf" => f64::INFINITY,
                    _ => text.parse::<f64>().unwrap_or(f64::NAN),
                };
                let c = self.consts.len() as u32;
                self.consts.push((text.to_string(), v));
                self.const_ids.insert(text.to_string(), c);
                c
            }
        };
        self.intern(N::Num(cid))
    }

    fn add_possibly_negative(&mut self, v: f64) -> Id {
        if v.is_nan() {
            return self.nan;
        }
        if v.is_infinite() {
            return if v < 0.0 { self.minus_inf } else { self.inf };
        }
        let c = self.add_num(&fmt_g16(v.abs()));
        if v < 0.0 { self.add_uminus(c) } else { c }
    }

    fn add_var(&mut self, s: u32, lag: i32) -> Id {
        self.intern(N::Var(s, lag))
    }

    /// `DataTree::AddExternalFunction`.
    fn add_ext(&mut self, fid: u32, args: Vec<Id>) -> Id {
        let l = match self.ext_list_ids.get(&args) {
            Some(&l) => l,
            None => {
                let l = self.ext_lists.len() as u32;
                self.ext_lists.push(args.clone());
                self.ext_list_ids.insert(args, l);
                l
            }
        };
        self.intern(N::Ext(fid, l))
    }

    fn ext_args(&self, l: u32) -> Vec<Id> {
        self.ext_lists[l as usize].clone()
    }

    fn add_unary_op(&mut self, op: UOp, arg: Id, info: i32) -> Id {
        let key = N::Un(op, arg, info);
        if let Some(&id) = self.map.get(&key) {
            return id;
        }
        let is_num = matches!(self.node(arg), N::Num(_));
        if (op != UOp::Uminus || !is_num)
            && let Some(v) = self.eval(arg)
            && let Some(r) = eval_uop(op, v)
        {
            return self.add_possibly_negative(r);
        }
        self.intern(key)
    }

    fn add_binary_op(&mut self, a: Id, op: BOp, b: Id) -> Id {
        let key = N::Bin(op, a, b);
        if let Some(&id) = self.map.get(&key) {
            return id;
        }
        if let Some(x) = self.eval(a)
            && let Some(y) = self.eval(b)
            && let Some(r) = eval_bop(op, x, y)
        {
            return self.add_possibly_negative(r);
        }
        self.intern(key)
    }

    fn add_trinary_op(&mut self, a: Id, op: TOp, b: Id, c: Id) -> Id {
        let key = N::Tri(op, a, b, c);
        if let Some(&id) = self.map.get(&key) {
            return id;
        }
        if let Some(x) = self.eval(a)
            && let Some(y) = self.eval(b)
            && let Some(z) = self.eval(c)
        {
            let r = eval_top(op, x, y, z);
            return self.add_possibly_negative(r);
        }
        self.intern(key)
    }

    fn uminus_arg(&self, id: Id) -> Option<Id> {
        match self.node(id) {
            N::Un(UOp::Uminus, a, _) => Some(a),
            _ => None,
        }
    }
    fn bin(&self, id: Id, op: BOp) -> Option<(Id, Id)> {
        match self.node(id) {
            N::Bin(o, a, b) if o == op => Some((a, b)),
            _ => None,
        }
    }

    fn add_plus(&mut self, mut a: Id, mut b: Id) -> Id {
        if b == self.zero {
            return a;
        }
        if a == self.zero {
            return b;
        }
        if let Some(y) = self.uminus_arg(b) {
            return self.add_minus(a, y);
        }
        if let Some(x) = self.uminus_arg(a) {
            return self.add_minus(b, x);
        }
        if let Some((x, y)) = self.bin(a, BOp::Minus)
            && y == b
        {
            return x;
        }
        if let Some((x, y)) = self.bin(b, BOp::Minus)
            && y == a
        {
            return x;
        }
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        self.add_binary_op(a, BOp::Plus, b)
    }

    fn add_minus(&mut self, a: Id, b: Id) -> Id {
        if b == self.zero {
            return a;
        }
        if a == self.zero {
            return self.add_uminus(b);
        }
        if a == b {
            return self.zero;
        }
        if let Some(y) = self.uminus_arg(b) {
            return self.add_plus(a, y);
        }
        if let Some((x, y)) = self.bin(a, BOp::Plus) {
            if y == b {
                return x;
            }
            if x == b {
                return y;
            }
        }
        self.add_binary_op(a, BOp::Minus, b)
    }

    fn add_uminus(&mut self, a: Id) -> Id {
        if a == self.zero {
            return self.zero;
        }
        if let Some(x) = self.uminus_arg(a) {
            return x;
        }
        self.add_unary_op(UOp::Uminus, a, 0)
    }

    fn add_times(&mut self, mut a: Id, mut b: Id) -> Id {
        if a == self.zero || b == self.zero {
            return self.zero;
        }
        if a == self.one {
            return b;
        }
        if b == self.one {
            return a;
        }
        if a == self.minus_one {
            return self.add_uminus(b);
        }
        if b == self.minus_one {
            return self.add_uminus(a);
        }
        if let Some((x, y)) = self.bin(a, BOp::Divide)
            && y == b
        {
            return x;
        }
        if let Some((x, y)) = self.bin(b, BOp::Divide)
            && y == a
        {
            return x;
        }
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        self.add_binary_op(a, BOp::Times, b)
    }

    fn add_divide(&mut self, a: Id, b: Id) -> Result<Id> {
        if b == self.one {
            return Ok(a);
        }
        if b == self.zero {
            bail!("division by zero (denominator simplified to 0)");
        }
        if a == self.zero {
            return Ok(self.zero);
        }
        if a == b {
            return Ok(self.one);
        }
        if let Some((x, y)) = self.bin(b, BOp::Divide)
            && x == self.one
        {
            return Ok(self.add_times(a, y));
        }
        if let Some((x, y)) = self.bin(a, BOp::Times) {
            if y == b {
                return Ok(x);
            }
            if x == b {
                return Ok(y);
            }
        }
        Ok(self.add_binary_op(a, BOp::Divide, b))
    }

    fn add_power(&mut self, a: Id, b: Id) -> Id {
        if b == self.zero {
            return self.one;
        }
        if a == self.zero {
            return self.zero;
        }
        if a == self.one {
            return self.one;
        }
        if b == self.one {
            return a;
        }
        self.add_binary_op(a, BOp::Power, b)
    }

    fn add_diff(&mut self, a: Id) -> Id {
        if self.max_lead(a) > 0 {
            // preprocessor#21: diffs with leads are expanded at once
            let l = self.decrease(a, 1);
            return self.add_minus(a, l);
        }
        self.add_unary_op(UOp::Diff, a, 0)
    }

    fn add_fn(&mut self, f: UnaryOp, a: Id) -> Id {
        use UnaryOp as U;
        let (z, o) = (self.zero, self.one);
        let r = match f {
            U::Exp | U::Cos | U::Cosh | U::Erfc if a == z => Some(o),
            U::Log | U::Log10 | U::Acos | U::Acosh if a == o => Some(z),
            U::Sin | U::Tan | U::Asin | U::Atan | U::Sinh | U::Tanh | U::Asinh | U::Atanh | U::Erf if a == z => Some(z),
            U::Sqrt | U::Cbrt | U::Abs | U::Sign if a == z => Some(z),
            U::Sqrt | U::Cbrt | U::Abs | U::Sign if a == o => Some(o),
            _ => None,
        };
        if let Some(r) = r {
            return r;
        }
        if matches!(f, U::Log | U::Log10)
            && let Some((x, y)) = self.bin(a, BOp::Divide)
            && x == self.one
        {
            // log(1/x) -> -log(x)
            let l = self.add_fn(f, y);
            return self.add_uminus(l);
        }
        self.add_unary_op(UOp::F(f), a, 0)
    }

    fn add_equal(&mut self, a: Id, b: Id) -> Id {
        self.add_binary_op(a, BOp::Equal, b)
    }

    /// `buildSimilarUnaryOpNode`.
    fn build_un(&mut self, op: UOp, info: i32, a: Id) -> Id {
        match op {
            UOp::Uminus => self.add_uminus(a),
            UOp::F(f) => self.add_fn(f, a),
            UOp::SteadyState => self.add_unary_op(UOp::SteadyState, a, 0),
            UOp::Expectation => self.add_unary_op(UOp::Expectation, a, info),
            UOp::Diff => self.add_diff(a),
        }
    }

    /// `buildSimilarBinaryOpNode`.
    fn build_bin(&mut self, op: BOp, a: Id, b: Id) -> Result<Id> {
        Ok(match op {
            BOp::Plus => self.add_plus(a, b),
            BOp::Minus => self.add_minus(a, b),
            BOp::Times => self.add_times(a, b),
            BOp::Divide => self.add_divide(a, b)?,
            BOp::Power => self.add_power(a, b),
            BOp::Equal => self.add_equal(a, b),
            o => self.add_binary_op(a, o, b),
        })
    }

    fn build_tri(&mut self, op: TOp, a: Id, b: Id, c: Id) -> Id {
        self.add_trinary_op(a, op, b, c)
    }

    // ---- evaluation and lead/lag queries --------------------------------------------------
    /// `ExprNode::eval({})`: None where the C++ throws `EvalException`.
    fn eval(&mut self, id: Id) -> Option<f64> {
        match self.node(id) {
            N::Num(c) => Some(self.consts[c as usize].1),
            N::Var(s, lag) => {
                if self.sym_ty(s) == SymType::Local {
                    let v = self.get_local(s, lag);
                    self.eval(v)
                } else {
                    None
                }
            }
            N::Un(op, a, _) => {
                let v = self.eval(a)?;
                eval_uop(op, v)
            }
            N::Bin(op, a, b) => {
                let x = self.eval(a)?;
                let y = self.eval(b)?;
                eval_bop(op, x, y)
            }
            N::Tri(op, a, b, c) => {
                let x = self.eval(a)?;
                let y = self.eval(b)?;
                let z = self.eval(c)?;
                Some(eval_top(op, x, y, z))
            }
            // the preprocessor's eval throws on external functions
            N::Ext(..) => None,
        }
    }

    /// `DataTree::getLocalVariable(symb_id, lead_lag)`.
    fn get_local(&mut self, s: u32, lag: i32) -> Id {
        let v = self.locals[&s];
        if lag == 0 { v } else { self.decrease(v, -lag) }
    }

    fn set_local(&mut self, s: u32, v: Id) {
        self.locals.insert(s, v);
        self.mx_cache.clear();
    }

    fn mx(&mut self, k: Mx, id: Id) -> i32 {
        if let Some(&v) = self.mx_cache.get(&(k, id)) {
            return v;
        }
        let v = match self.node(id) {
            N::Num(_) => {
                if k == Mx::Lead {
                    i32::MIN
                } else {
                    0
                }
            }
            N::Var(s, lag) => match (self.sym_ty(s), k) {
                (SymType::Endo, Mx::EndoLead) | (SymType::Exo, Mx::ExoLead) => lag.max(0),
                (SymType::Endo, Mx::EndoLag) | (SymType::Exo, Mx::ExoLag) => (-lag).max(0),
                (SymType::Endo | SymType::Exo, Mx::Lead) => lag,
                (SymType::Endo | SymType::Exo, Mx::LagDiffs) => -lag,
                (SymType::Local, _) => {
                    let v = self.get_local(s, lag);
                    self.mx(k, v)
                }
                _ => 0,
            },
            N::Un(op, a, _) => self.mx(k, a) + (k == Mx::LagDiffs && op == UOp::Diff) as i32,
            N::Bin(_, a, b) => {
                let x = self.mx(k, a);
                x.max(self.mx(k, b))
            }
            N::Tri(_, a, b, c) => {
                let x = self.mx(k, a);
                let y = self.mx(k, b);
                x.max(y).max(self.mx(k, c))
            }
            N::Ext(_, l) => {
                let mut m = if k == Mx::Lead { i32::MIN } else { 0 };
                for a in self.ext_args(l) {
                    m = m.max(self.mx(k, a));
                }
                m
            }
        };
        self.mx_cache.insert((k, id), v);
        v
    }
    fn max_endo_lead(&mut self, id: Id) -> i32 {
        self.mx(Mx::EndoLead, id)
    }
    fn max_exo_lead(&mut self, id: Id) -> i32 {
        self.mx(Mx::ExoLead, id)
    }
    fn max_endo_lag(&mut self, id: Id) -> i32 {
        self.mx(Mx::EndoLag, id)
    }
    fn max_exo_lag(&mut self, id: Id) -> i32 {
        self.mx(Mx::ExoLag, id)
    }
    fn max_lead(&mut self, id: Id) -> i32 {
        self.mx(Mx::Lead, id)
    }

    /// `decreaseLeadsLags(n)`.
    fn decrease(&mut self, id: Id, n: i32) -> Id {
        let mut memo = HashMap::new();
        self.decrease_m(id, n, &mut memo)
    }
    fn decrease_m(&mut self, id: Id, n: i32, memo: &mut HashMap<Id, Id>) -> Id {
        if let Some(&r) = memo.get(&id) {
            return r;
        }
        let r = match self.node(id) {
            N::Num(_) => id,
            N::Var(s, lag) => match self.sym_ty(s) {
                SymType::Endo
                | SymType::Exo
                | SymType::Trend
                | SymType::LogTrend
                | SymType::AggEndo
                | SymType::AggExo => self.add_var(s, lag - n),
                SymType::Local => {
                    let v = self.get_local(s, lag);
                    self.decrease_m(v, n, memo)
                }
                SymType::Param | SymType::Excluded => id,
            },
            N::Un(op, a, info) => {
                let a2 = self.decrease_m(a, n, memo);
                self.build_un(op, info, a2)
            }
            N::Bin(op, a, b) => {
                let a2 = self.decrease_m(a, n, memo);
                let b2 = self.decrease_m(b, n, memo);
                self.build_bin(op, a2, b2).unwrap_or_else(|_| self.add_binary_op(a2, op, b2))
            }
            N::Tri(op, a, b, c) => {
                let a2 = self.decrease_m(a, n, memo);
                let b2 = self.decrease_m(b, n, memo);
                let c2 = self.decrease_m(c, n, memo);
                self.build_tri(op, a2, b2, c2)
            }
            N::Ext(f, l) => {
                let v: Vec<Id> = self.ext_args(l).into_iter().map(|a| self.decrease_m(a, n, memo)).collect();
                self.add_ext(f, v)
            }
        };
        memo.insert(id, r);
        r
    }

    /// Generic `recurseTransform` with a leaf rule (`None`: keep the leaf).
    fn recurse(&mut self, id: Id, memo: &mut HashMap<Id, Id>, leaf: &mut LeafFn<'_>) -> Result<Id> {
        if let Some(&r) = memo.get(&id) {
            return Ok(r);
        }
        let r = match self.node(id) {
            N::Num(_) => id,
            N::Var(s, lag) => leaf(self, id, s, lag)?.unwrap_or(id),
            N::Un(op, a, info) => {
                let a2 = self.recurse(a, memo, leaf)?;
                self.build_un(op, info, a2)
            }
            N::Bin(op, a, b) => {
                let a2 = self.recurse(a, memo, leaf)?;
                let b2 = self.recurse(b, memo, leaf)?;
                self.build_bin(op, a2, b2)?
            }
            N::Tri(op, a, b, c) => {
                let a2 = self.recurse(a, memo, leaf)?;
                let b2 = self.recurse(b, memo, leaf)?;
                let c2 = self.recurse(c, memo, leaf)?;
                self.build_tri(op, a2, b2, c2)
            }
            N::Ext(f, l) => {
                let mut v = Vec::new();
                for a in self.ext_args(l) {
                    v.push(self.recurse(a, memo, leaf)?);
                }
                self.add_ext(f, v)
            }
        };
        memo.insert(id, r);
        Ok(r)
    }

    /// `collectDynamicVariables` of one type (into `out`), recursing into model-local variables.
    fn collect_vars(&mut self, id: Id, ty: SymType, out: &mut BTreeSet<(u32, i32)>, seen: &mut BTreeSet<Id>) {
        if !seen.insert(id) {
            return;
        }
        match self.node(id) {
            N::Num(_) => {}
            N::Var(s, lag) => {
                let t = self.sym_ty(s);
                if t == ty {
                    out.insert((s, lag));
                }
                if t == SymType::Local {
                    let v = self.get_local(s, lag);
                    self.collect_vars(v, ty, out, seen);
                }
            }
            N::Un(_, a, _) => self.collect_vars(a, ty, out, seen),
            N::Bin(_, a, b) => {
                self.collect_vars(a, ty, out, seen);
                self.collect_vars(b, ty, out, seen);
            }
            N::Tri(_, a, b, c) => {
                self.collect_vars(a, ty, out, seen);
                self.collect_vars(b, ty, out, seen);
                self.collect_vars(c, ty, out, seen);
            }
            N::Ext(_, l) => {
                for a in self.ext_args(l) {
                    self.collect_vars(a, ty, out, seen);
                }
            }
        }
    }

    // ---- symbol table -----------------------------------------------------------------------
    fn add_aux(&mut self, name: String, ty: u8, orig_symb: Option<u32>, orig_lead_lag: Option<i32>, rhs: Id) -> u32 {
        let symb = self.syms.len() as u32;
        self.syms.push(Sym { ty: SymType::Endo, index: self.n_endo });
        self.names.push(name.clone());
        self.n_endo += 1;
        self.aux.push(AuxRec { symb, name, ty, orig_symb, orig_lead_lag, rhs, shown: None });
        symb
    }

    fn eq_args(&self, eq: Id) -> (Id, Id) {
        match self.node(eq) {
            N::Bin(BOp::Equal, a, b) => (a, b),
            _ => unreachable!("equation node is not an equality"),
        }
    }

    fn add_equation(&mut self, eq: Id, tags: Vec<(String, String)>) {
        self.equations.push(Equation { eq, tags, line: 0 });
    }

    // ---- parser replay ------------------------------------------------------------------------
    fn replay(
        &mut self,
        ast: &Ast,
        nums: &mut std::slice::Iter<'_, String>,
        lookup: &dyn Fn(&str) -> Option<u32>,
    ) -> Result<Id> {
        Ok(match ast {
            Ast::Num(v) => match nums.next() {
                Some(t) => self.add_num(t),
                None => self.add_num(&fmt_g16(*v)),
            },
            Ast::Var { name, lag, loc } => {
                let s = match lookup(name) {
                    Some(s) => s,
                    None => match name.as_str() {
                        "inf" | "Inf" => return Ok(self.inf),
                        "nan" | "NaN" => return Ok(self.nan),
                        "pi" => return Ok(self.pi),
                        _ => bail!("{loc}: unknown symbol '{name}'"),
                    },
                };
                if *lag != 0 {
                    // the lead/lag is parsed as an expression first
                    let c = self.add_num(&lag.abs().to_string());
                    if *lag < 0 {
                        self.add_uminus(c);
                    }
                }
                self.add_var(s, *lag)
            }
            Ast::Un(UnaryOp::Neg, a) => {
                let a = self.replay(a, nums, lookup)?;
                self.add_uminus(a)
            }
            Ast::Un(op, a) => {
                let a = self.replay(a, nums, lookup)?;
                self.add_fn(*op, a)
            }
            Ast::Bin(op, a, b) => {
                let x = self.replay(a, nums, lookup)?;
                let y = self.replay(b, nums, lookup)?;
                let o = match op {
                    BinaryOp::Add => BOp::Plus,
                    BinaryOp::Sub => BOp::Minus,
                    BinaryOp::Mul => BOp::Times,
                    BinaryOp::Div => BOp::Divide,
                    BinaryOp::Pow => BOp::Power,
                    BinaryOp::Max => BOp::Max,
                    BinaryOp::Min => BOp::Min,
                    BinaryOp::Lt => BOp::Less,
                    BinaryOp::Gt => BOp::Greater,
                    BinaryOp::Le => BOp::LessEqual,
                    BinaryOp::Ge => BOp::GreaterEqual,
                    BinaryOp::Eq => BOp::EqualEqual,
                    BinaryOp::Ne => BOp::Different,
                    BinaryOp::PowerDeriv(_) => unreachable!("getPowerDeriv only arises in derivatives"),
                };
                self.build_bin(o, x, y)?
            }
            Ast::Op(op, args) => {
                let mut v = Vec::with_capacity(args.len());
                for a in args {
                    v.push(self.replay(a, nums, lookup)?);
                }
                match op {
                    SpecialOp::SteadyState => {
                        let mut out = BTreeSet::new();
                        self.collect_vars(v[0], SymType::Exo, &mut out, &mut BTreeSet::new());
                        if !out.is_empty() {
                            bail!("Exogenous variables are not allowed in the context of the STEADY_STATE() operator.");
                        }
                        self.add_unary_op(UOp::SteadyState, v[0], 0)
                    }
                    SpecialOp::Expectation(k) => self.add_unary_op(UOp::Expectation, v[0], *k),
                    SpecialOp::Diff => self.add_diff(v[0]),
                    SpecialOp::Normcdf | SpecialOp::Normpdf => {
                        let t = if *op == SpecialOp::Normcdf { TOp::Normcdf } else { TOp::Normpdf };
                        let (b, c) = if v.len() == 3 { (v[1], v[2]) } else { (self.zero, self.one) };
                        self.build_tri(t, v[0], b, c)
                    }
                    SpecialOp::External(f) => self.add_ext(*f, v),
                }
            }
        })
    }

    /// `ParsingDriver::add_model_equal` (+ `DynamicModel::addOccbinEquation`).
    fn add_model_equal(
        &mut self,
        lhs: Id,
        rhs: Id,
        tags: Vec<(String, String)>,
        lookup: &dyn Fn(&str) -> Option<u32>,
    ) -> Result<()> {
        let id = self.add_equal(lhs, rhs);
        let has = |k: &str| tags.iter().any(|(t, _)| t == k);
        if has("static") {
            self.static_only.push(Equation { eq: id, tags, line: 0 });
            return Ok(());
        }
        if has("bind") || has("relax") {
            let get = |k: &str| tags.iter().find(|(t, _)| t == k).map(|(_, v)| v.clone()).unwrap_or_default();
            let split = |v: String| -> Vec<String> {
                v.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()
            };
            let (bind, relax) = (split(get("bind")), split(get("relax")));
            let name = get("name");
            let mut tags = tags.clone();
            tags.retain(|(k, _)| k != "bind" && k != "relax");
            let (l, r) = self.eq_args(id);
            let basic = self.add_minus(l, r);
            let mut term = basic;
            for c in &bind {
                let p =
                    lookup(&format!("occbin_{c}_bind")).ok_or_else(|| anyhow!("unknown OccBin constraint '{c}'"))?;
                let pv = self.add_var(p, 0);
                term = self.add_times(term, pv);
            }
            for c in &relax {
                let p =
                    lookup(&format!("occbin_{c}_bind")).ok_or_else(|| anyhow!("unknown OccBin constraint '{c}'"))?;
                let pv = self.add_var(p, 0);
                let one = self.one;
                let f = self.add_minus(one, pv);
                term = self.add_times(term, f);
            }
            let find =
                |eqs: &[Equation]| eqs.iter().position(|e| e.tags.iter().any(|(k, v)| k == "name" && *v == name));
            if let Some(k) = find(&self.equations) {
                let (ol, or) = self.eq_args(self.equations[k].eq);
                let d = self.add_minus(ol, or);
                let s = self.add_plus(d, term);
                let z = self.zero;
                self.equations[k].eq = self.add_equal(s, z);
            } else {
                let mut td = tags.clone();
                td.push(("dynamic".into(), String::new()));
                let z = self.zero;
                let e = self.add_equal(term, z);
                self.add_equation(e, td);
            }
            if bind.is_empty() {
                if let Some(k) = find(&self.static_only) {
                    let (ol, or) = self.eq_args(self.static_only[k].eq);
                    let d = self.add_minus(ol, or);
                    let s = self.add_plus(d, basic);
                    let z = self.zero;
                    self.static_only[k].eq = self.add_equal(s, z);
                } else {
                    let mut ts = tags;
                    ts.push(("static".into(), String::new()));
                    let z = self.zero;
                    let e = self.add_equal(basic, z);
                    self.static_only.push(Equation { eq: e, tags: ts, line: 0 });
                }
            }
            return Ok(());
        }
        self.add_equation(id, tags);
        Ok(())
    }

    // ---- transform passes -----------------------------------------------------------------------
    fn for_all_eqs(&mut self, statics: bool, f: &mut dyn FnMut(&mut Tree, Id) -> Result<Id>) -> Result<()> {
        for k in 0..self.equations.len() {
            let e = self.equations[k].eq;
            self.equations[k].eq = f(self, e)?;
        }
        if statics {
            for k in 0..self.static_only.len() {
                let e = self.static_only[k].eq;
                self.static_only[k].eq = f(self, e)?;
            }
        }
        Ok(())
    }

    fn transform_predetermined(&mut self) -> Result<()> {
        if self.predetermined.is_empty() {
            return Ok(());
        }
        let pre = self.predetermined.clone();
        let mut leaf = |t: &mut Tree, _id: Id, s: u32, lag: i32| -> Result<Option<Id>> {
            Ok(if pre.contains(&s) { Some(t.add_var(s, lag - 1)) } else { None })
        };
        let keys: Vec<u32> = self.locals.keys().copied().collect();
        for s in keys {
            let v = self.locals[&s];
            let mut memo = HashMap::new();
            let r = self.recurse(v, &mut memo, &mut leaf)?;
            self.set_local(s, r);
        }
        for k in 0..self.equations.len() {
            let mut memo = HashMap::new();
            let e = self.equations[k].eq;
            self.equations[k].eq = self.recurse(e, &mut memo, &mut leaf)?;
        }
        Ok(())
    }

    /// `DynamicModel::simplifyEquations`: variables set to a constant by an equation `x = c`
    /// (contemporaneous `x`, non-negative constant `c`) are replaced by `c` everywhere.
    fn simplify_equations(&mut self) -> Result<()> {
        // findConstantEquations (keyed by the variable node; replacement looks the symbol up)
        let mut table: Vec<(Id, u32, Id)> = Vec::new();
        // findConstantEquationsWithoutMcpTag
        let mcp =
            |e: &Equation| e.tags.iter().any(|(k, _)| k == crate::model::MCP_TAG || k == "mcp" || k == "__het_agg"); // heterogeneity: SUM_x = 0 placeholders
        for e in self.equations.iter().filter(|e| !mcp(e)) {
            let (a, b) = self.eq_args(e.eq);
            match (self.node(a), self.node(b)) {
                (N::Var(s, 0), N::Num(_)) => table.push((a, s, b)),
                (N::Num(_), N::Var(s, 0)) => table.push((b, s, a)),
                _ => {}
            }
        }
        if table.is_empty() {
            return Ok(());
        }
        // std::map<VariableNode*, NumConstNode*>: one entry per variable node, iterated by address
        // (node creation order here)
        table.sort_by_key(|x| x.0);
        table.dedup_by_key(|x| x.0);
        let lookup = |s: u32| table.iter().find(|x| x.1 == s).map(|x| x.2);
        let keys: Vec<u32> = self.locals.keys().copied().collect();
        for s in keys {
            let v = self.locals[&s];
            let r = self.replace_vars(v, &table, &lookup)?;
            self.set_local(s, r);
        }
        for statics in [false, true] {
            let n = if statics { self.static_only.len() } else { self.equations.len() };
            for k in 0..n {
                let e = if statics { self.static_only[k].eq } else { self.equations[k].eq };
                let r = self.replace_vars(e, &table, &lookup)?;
                if statics {
                    self.static_only[k].eq = r;
                } else {
                    self.equations[k].eq = r;
                }
            }
        }
        Ok(())
    }

    fn replace_vars(&mut self, id: Id, table: &[(Id, u32, Id)], lookup: &dyn Fn(u32) -> Option<Id>) -> Result<Id> {
        match self.node(id) {
            N::Num(_) => Ok(id),
            N::Var(s, _) => Ok(lookup(s).unwrap_or(id)),
            N::Bin(op, a, b) => {
                if op == BOp::Equal && table.iter().any(|&(v, _, c)| (v == a && c == b) || (v == b && c == a)) {
                    return Ok(id);
                }
                let a2 = self.replace_vars(a, table, lookup)?;
                let b2 = self.replace_vars(b, table, lookup)?;
                self.build_bin(op, a2, b2)
            }
            N::Un(op, a, info) => {
                let a2 = self.replace_vars(a, table, lookup)?;
                Ok(self.build_un(op, info, a2))
            }
            N::Tri(op, a, b, c) => {
                let a2 = self.replace_vars(a, table, lookup)?;
                let b2 = self.replace_vars(b, table, lookup)?;
                let c2 = self.replace_vars(c, table, lookup)?;
                Ok(self.build_tri(op, a2, b2, c2))
            }
            N::Ext(f, l) => {
                let mut v = Vec::new();
                for a in self.ext_args(l) {
                    v.push(self.replace_vars(a, table, lookup)?);
                }
                Ok(self.add_ext(f, v))
            }
        }
    }

    /// `DynamicModel::expandEqTags`: equations without a `name` tag get the name of their
    /// left-hand side variable, else their number.
    fn expand_eq_tags(&mut self) -> Result<()> {
        let has = |eqs: &[Equation], v: &str| eqs.iter().any(|e| e.tags.iter().any(|(k, x)| k == "name" && x == v));
        let existing: Vec<bool> = self.equations.iter().map(|e| e.tags.iter().any(|(k, _)| k == "name")).collect();
        for (eq, &named) in existing.iter().enumerate() {
            if named {
                continue;
            }
            let (a, _) = self.eq_args(self.equations[eq].eq);
            let lhs = match self.node(a) {
                N::Var(s, _) => Some(self.names[s as usize].clone()),
                _ => None,
            };
            let name = match lhs {
                Some(n) if !has(&self.equations, &n) => n,
                _ if !has(&self.equations, &(eq + 1).to_string()) => (eq + 1).to_string(),
                _ => bail!(
                    "Error creating default equation tag: cannot assign default tag to equation number {} because it is already in use",
                    eq + 1
                ),
            };
            self.equations[eq].tags.push(("name".into(), name));
        }
        Ok(())
    }

    fn substitute_model_local_variables(&mut self) -> Result<()> {
        let mut leaf = |t: &mut Tree, _id: Id, s: u32, lag: i32| -> Result<Option<Id>> {
            Ok(if t.sym_ty(s) == SymType::Local { Some(t.get_local(s, lag)) } else { None })
        };
        let mut memo = HashMap::new();
        self.for_all_eqs(true, &mut |t, e| t.recurse(e, &mut memo, &mut leaf))
    }

    fn used_locals(&mut self) -> Vec<u32> {
        let mut out = BTreeSet::new();
        let mut seen = BTreeSet::new();
        for k in 0..self.equations.len() {
            let e = self.equations[k].eq;
            self.collect_vars(e, SymType::Local, &mut out, &mut seen);
        }
        let s: BTreeSet<u32> = out.into_iter().map(|x| x.0).collect();
        s.into_iter().collect()
    }

    fn push_neweqs(&mut self, neweqs: Vec<Id>) {
        for e in neweqs {
            self.add_equation(e, Vec::new());
        }
    }

    // -- diff
    fn lag_equiv_class(&mut self, id: Id) -> (Id, i32) {
        let mut index = self.max_lead(id);
        if index == i32::MIN {
            index = 0;
        }
        (self.decrease(id, index), index)
    }

    fn find_diff_nodes(&mut self, id: Id, nodes: &mut LagEquiv, seen: &mut BTreeSet<Id>) {
        if !seen.insert(id) {
            return;
        }
        match self.node(id) {
            N::Num(_) => {}
            N::Var(s, lag) => {
                if self.sym_ty(s) == SymType::Local {
                    let v = self.get_local(s, lag);
                    self.find_diff_nodes(v, nodes, seen);
                }
            }
            N::Un(op, a, _) => {
                self.find_diff_nodes(a, nodes, seen);
                if op == UOp::Diff {
                    let (repr, index) = self.lag_equiv_class(id);
                    nodes.entry(repr).or_default().insert(index, id);
                }
            }
            N::Bin(_, a, b) => {
                self.find_diff_nodes(a, nodes, seen);
                self.find_diff_nodes(b, nodes, seen);
            }
            N::Tri(_, a, b, c) => {
                self.find_diff_nodes(a, nodes, seen);
                self.find_diff_nodes(b, nodes, seen);
                self.find_diff_nodes(c, nodes, seen);
            }
            N::Ext(_, l) => {
                for a in self.ext_args(l) {
                    self.find_diff_nodes(a, nodes, seen);
                }
            }
        }
    }

    fn subst_diff(&mut self, id: Id, nodes: &LagEquiv, table: &mut SubstTable, neweqs: &mut Vec<Id>) -> Result<Id> {
        match self.node(id) {
            N::Num(_) => Ok(id),
            N::Var(s, lag) => {
                if self.sym_ty(s) == SymType::Local {
                    let v = self.get_local(s, lag);
                    self.subst_diff(v, nodes, table, neweqs)
                } else {
                    Ok(id)
                }
            }
            N::Bin(op, a, b) => {
                let a2 = self.subst_diff(a, nodes, table, neweqs)?;
                let b2 = self.subst_diff(b, nodes, table, neweqs)?;
                self.build_bin(op, a2, b2)
            }
            N::Tri(op, a, b, c) => {
                let a2 = self.subst_diff(a, nodes, table, neweqs)?;
                let b2 = self.subst_diff(b, nodes, table, neweqs)?;
                let c2 = self.subst_diff(c, nodes, table, neweqs)?;
                Ok(self.build_tri(op, a2, b2, c2))
            }
            N::Ext(f, l) => {
                let mut v = Vec::new();
                for a in self.ext_args(l) {
                    v.push(self.subst_diff(a, nodes, table, neweqs)?);
                }
                Ok(self.add_ext(f, v))
            }
            N::Un(op, a, info) => {
                let argsubst = self.subst_diff(a, nodes, table, neweqs)?;
                if op != UOp::Diff {
                    return Ok(self.build_un(op, info, argsubst));
                }
                if let Some(&r) = table.get(&id) {
                    return Ok(r);
                }
                let (repr, index) = self.lag_equiv_class(id);
                let class = nodes.get(&repr).filter(|m| m.get(&index) == Some(&id)).cloned();
                let Some(class) = class else {
                    let symb = self.add_aux(format!("AUX_DIFF_{argsubst}"), 8, None, None, 0);
                    self.aux.last_mut().unwrap().shown = Some(id);
                    let aux_var = self.add_var(symb, 0);
                    let l = self.decrease(argsubst, 1);
                    let m = self.add_minus(argsubst, l);
                    self.aux.last_mut().unwrap().rhs = m;
                    neweqs.push(self.add_equal(aux_var, m));
                    table.insert(id, aux_var);
                    return Ok(aux_var);
                };
                let mut last_index = 0;
                let mut last_aux_var: Option<(u32, Id)> = None;
                for (k, (&idx, &node)) in class.iter().rev().enumerate() {
                    let N::Un(_, narg, _) = self.node(node) else { unreachable!() };
                    let argsubst = self.subst_diff(narg, nodes, table, neweqs)?;
                    if k == 0 {
                        let (os, ol) = match self.node(argsubst) {
                            N::Var(s, l) => (Some(s), Some(l)),
                            _ => (None, None),
                        };
                        let symb = self.add_aux(format!("AUX_DIFF_{argsubst}"), 8, os, ol, 0);
                        self.aux.last_mut().unwrap().shown = Some(node);
                        last_index = idx;
                        let v = self.add_var(symb, 0);
                        last_aux_var = Some((symb, v));
                        let l = self.decrease(argsubst, 1);
                        let m = self.add_minus(argsubst, l);
                        self.aux.last_mut().unwrap().rhs = m;
                        neweqs.push(self.add_equal(v, m));
                        table.insert(node, v);
                    } else {
                        let mut new_aux: Option<(u32, Id)> = None;
                        let mut i = last_index;
                        while i > idx {
                            let (ls, lv) = last_aux_var.unwrap();
                            let name_idx = if i == last_index { argsubst } else { new_aux.unwrap().1 };
                            let symb = self.add_aux(format!("AUX_DIFF_LAG_{name_idx}"), 9, Some(ls), Some(-1), 0);
                            self.aux.last_mut().unwrap().shown = Some(node);
                            let v = self.add_var(symb, 0);
                            let l = self.decrease(lv, 1);
                            self.aux.last_mut().unwrap().rhs = l;
                            neweqs.push(self.add_equal(v, l));
                            new_aux = Some((symb, v));
                            last_aux_var = new_aux;
                            i -= 1;
                        }
                        let v = new_aux.map(|x| x.1).ok_or_else(|| anyhow!("internal error in diff substitution"))?;
                        table.insert(node, v);
                        last_index = idx;
                    }
                }
                table.get(&id).copied().ok_or_else(|| anyhow!("internal error in diff substitution"))
            }
        }
    }

    fn substitute_diff(&mut self) -> Result<()> {
        let mut nodes = LagEquiv::new();
        let used = self.used_locals();
        let keys: Vec<u32> = self.locals.keys().copied().collect();
        let mut seen = BTreeSet::new();
        for &s in &keys {
            if used.contains(&s) {
                let v = self.locals[&s];
                self.find_diff_nodes(v, &mut nodes, &mut seen);
            }
        }
        for k in 0..self.equations.len() {
            let e = self.equations[k].eq;
            self.find_diff_nodes(e, &mut nodes, &mut seen);
        }
        let mut table = SubstTable::new();
        let mut neweqs = Vec::new();
        for &s in &keys {
            let v = self.locals[&s];
            let r = self.subst_diff(v, &nodes, &mut table, &mut neweqs)?;
            self.set_local(s, r);
        }
        for k in 0..self.equations.len() {
            let e = self.equations[k].eq;
            self.equations[k].eq = self.subst_diff(e, &nodes, &mut table, &mut neweqs)?;
        }
        self.push_neweqs(neweqs);
        Ok(())
    }

    // -- expectation
    fn subst_expectation(&mut self, id: Id, table: &mut SubstTable, neweqs: &mut Vec<Id>) -> Result<Id> {
        match self.node(id) {
            N::Num(_) => Ok(id),
            N::Var(s, lag) => {
                if self.sym_ty(s) == SymType::Local {
                    let v = self.get_local(s, lag);
                    self.subst_expectation(v, table, neweqs)
                } else {
                    Ok(id)
                }
            }
            N::Un(UOp::Expectation, a, info) => {
                if let Some(&r) = table.get(&id) {
                    return Ok(r);
                }
                let name = format!("AUX_EXPECT_{}_{}_{a}", if info < 0 { "LAG" } else { "LEAD" }, info.abs());
                let symb = self.add_aux(name, 4, None, None, 0);
                self.aux.last_mut().unwrap().shown = Some(id);
                let aux0 = self.add_var(symb, 0);
                let inner = self.subst_expectation(a, table, neweqs)?;
                let substexpr = self.decrease(inner, info);
                self.aux.last_mut().unwrap().rhs = substexpr;
                neweqs.push(self.add_equal(aux0, substexpr));
                let r = self.add_var(symb, info);
                table.insert(id, r);
                Ok(r)
            }
            N::Un(op, a, info) => {
                let a2 = self.subst_expectation(a, table, neweqs)?;
                Ok(self.build_un(op, info, a2))
            }
            N::Bin(op, a, b) => {
                let a2 = self.subst_expectation(a, table, neweqs)?;
                let b2 = self.subst_expectation(b, table, neweqs)?;
                self.build_bin(op, a2, b2)
            }
            N::Tri(op, a, b, c) => {
                let a2 = self.subst_expectation(a, table, neweqs)?;
                let b2 = self.subst_expectation(b, table, neweqs)?;
                let c2 = self.subst_expectation(c, table, neweqs)?;
                Ok(self.build_tri(op, a2, b2, c2))
            }
            N::Ext(f, l) => {
                let mut v = Vec::new();
                for a in self.ext_args(l) {
                    v.push(self.subst_expectation(a, table, neweqs)?);
                }
                Ok(self.add_ext(f, v))
            }
        }
    }

    fn substitute_expectation(&mut self) -> Result<()> {
        let mut table = SubstTable::new();
        let mut neweqs = Vec::new();
        let keys: Vec<u32> = self.locals.keys().copied().collect();
        for s in keys {
            let v = self.locals[&s];
            let r = self.subst_expectation(v, &mut table, &mut neweqs)?;
            self.set_local(s, r);
        }
        for k in 0..self.equations.len() {
            let e = self.equations[k].eq;
            self.equations[k].eq = self.subst_expectation(e, &mut table, &mut neweqs)?;
        }
        self.push_neweqs(neweqs);
        Ok(())
    }

    // -- leads and lags
    fn create_lead_aux(&mut self, id: Id, endo: bool, table: &mut SubstTable, neweqs: &mut Vec<Id>) -> Id {
        if let Some(&r) = table.get(&id) {
            return r;
        }
        let (mut substexpr, mut lag) = if endo {
            let n = self.max_endo_lead(id);
            (self.decrease(id, n - 1), n - 2)
        } else {
            let n = self.max_exo_lead(id);
            (self.decrease(id, n), n - 1)
        };
        while lag >= 0 {
            let orig = self.decrease(id, lag);
            if let Some(&r) = table.get(&orig) {
                substexpr = r;
            } else {
                let name = format!("{}{orig}", if endo { "AUX_ENDO_LEAD_" } else { "AUX_EXO_LEAD_" });
                let symb = self.add_aux(name, if endo { 0 } else { 2 }, None, None, substexpr);
                let v0 = self.add_var(symb, 0);
                neweqs.push(self.add_equal(v0, substexpr));
                substexpr = self.add_var(symb, 1);
                table.insert(orig, substexpr);
            }
            lag -= 1;
        }
        substexpr
    }

    fn subst_lead(
        &mut self,
        id: Id,
        endo: bool,
        det: bool,
        table: &mut SubstTable,
        neweqs: &mut Vec<Id>,
    ) -> Result<Id> {
        let thr = if endo { 2 } else { 1 };
        let own = |t: &mut Tree, n: Id| if endo { t.max_endo_lead(n) } else { t.max_exo_lead(n) };
        let other = |t: &mut Tree, n: Id| if endo { t.max_exo_lead(n) } else { t.max_endo_lead(n) };
        match self.node(id) {
            N::Num(_) => Ok(id),
            N::Var(s, lag) => match self.sym_ty(s) {
                SymType::Endo if endo => Ok(if lag <= 1 { id } else { self.create_lead_aux(id, endo, table, neweqs) }),
                SymType::Exo if !endo => Ok(if lag <= 0 { id } else { self.create_lead_aux(id, endo, table, neweqs) }),
                SymType::Local => {
                    let v = self.get_local(s, lag);
                    if own(self, v) < thr { Ok(id) } else { self.subst_lead(v, endo, det, table, neweqs) }
                }
                _ => Ok(id),
            },
            N::Un(op, a, info) => {
                if op == UOp::Uminus || det {
                    let a2 = self.subst_lead(a, endo, det, table, neweqs)?;
                    Ok(self.build_un(op, info, a2))
                } else if own(self, id) >= thr {
                    Ok(self.create_lead_aux(id, endo, table, neweqs))
                } else {
                    Ok(id)
                }
            }
            N::Bin(op, a, b) => {
                let (m1, m2) = (own(self, a), own(self, b));
                if m1 < thr && m2 < thr {
                    return Ok(id);
                }
                let both = |t: &mut Tree, table: &mut SubstTable, neweqs: &mut Vec<Id>| -> Result<Id> {
                    let a2 = if m1 >= thr { t.subst_lead(a, endo, det, table, neweqs)? } else { a };
                    let b2 = if m2 >= thr { t.subst_lead(b, endo, det, table, neweqs)? } else { b };
                    t.build_bin(op, a2, b2)
                };
                if det {
                    return both(self, table, neweqs);
                }
                match op {
                    BOp::Plus | BOp::Minus | BOp::Equal => both(self, table, neweqs),
                    BOp::Times | BOp::Divide => {
                        if m1 >= thr && m2 == 0 && other(self, b) == 0 {
                            let a2 = self.subst_lead(a, endo, det, table, neweqs)?;
                            return self.build_bin(op, a2, b);
                        }
                        if m1 == 0 && other(self, a) == 0 && m2 >= thr && op == BOp::Times {
                            let b2 = self.subst_lead(b, endo, det, table, neweqs)?;
                            return self.build_bin(op, a, b2);
                        }
                        Ok(self.create_lead_aux(id, endo, table, neweqs))
                    }
                    _ => Ok(self.create_lead_aux(id, endo, table, neweqs)),
                }
            }
            N::Tri(op, a, b, c) => {
                if own(self, id) < thr {
                    Ok(id)
                } else if det {
                    let a2 = self.subst_lead(a, endo, det, table, neweqs)?;
                    let b2 = self.subst_lead(b, endo, det, table, neweqs)?;
                    let c2 = self.subst_lead(c, endo, det, table, neweqs)?;
                    Ok(self.build_tri(op, a2, b2, c2))
                } else {
                    Ok(self.create_lead_aux(id, endo, table, neweqs))
                }
            }
            // `AbstractExternalFunctionNode::substitute*Lead*`: the arguments are substituted
            N::Ext(f, l) => {
                let mut v = Vec::new();
                for a in self.ext_args(l) {
                    v.push(self.subst_lead(a, endo, det, table, neweqs)?);
                }
                Ok(self.add_ext(f, v))
            }
        }
    }

    fn subst_lag(&mut self, id: Id, endo: bool, table: &mut SubstTable, neweqs: &mut Vec<Id>) -> Result<Id> {
        match self.node(id) {
            N::Num(_) => Ok(id),
            N::Var(s, lag) => match self.sym_ty(s) {
                SymType::Endo | SymType::Exo if (self.sym_ty(s) == SymType::Endo) == endo => {
                    let min = if endo { -1 } else { 0 };
                    if lag >= min {
                        return Ok(id);
                    }
                    if let Some(&r) = table.get(&id) {
                        return Ok(r);
                    }
                    let mut substexpr = self.add_var(s, min);
                    let mut cur = min - 1;
                    while cur >= lag {
                        let orig = self.add_var(s, cur);
                        if let Some(&r) = table.get(&orig) {
                            substexpr = r;
                        } else {
                            let name =
                                format!("{}{s}_{}", if endo { "AUX_ENDO_LAG_" } else { "AUX_EXO_LAG_" }, -(cur + 1));
                            let symb = self.add_aux(name, if endo { 1 } else { 3 }, Some(s), Some(cur + 1), substexpr);
                            let v0 = self.add_var(symb, 0);
                            neweqs.push(self.add_equal(v0, substexpr));
                            substexpr = self.add_var(symb, -1);
                            table.insert(orig, substexpr);
                        }
                        cur -= 1;
                    }
                    Ok(substexpr)
                }
                SymType::Local => {
                    let v = self.get_local(s, lag);
                    let m = if endo { self.max_endo_lag(v) } else { self.max_exo_lag(v) };
                    let lim = if endo { 1 } else { 0 };
                    if m <= lim { Ok(id) } else { self.subst_lag(v, endo, table, neweqs) }
                }
                _ => Ok(id),
            },
            N::Un(op, a, info) => {
                let a2 = self.subst_lag(a, endo, table, neweqs)?;
                Ok(self.build_un(op, info, a2))
            }
            N::Bin(op, a, b) => {
                let a2 = self.subst_lag(a, endo, table, neweqs)?;
                let b2 = self.subst_lag(b, endo, table, neweqs)?;
                self.build_bin(op, a2, b2)
            }
            N::Tri(op, a, b, c) => {
                let a2 = self.subst_lag(a, endo, table, neweqs)?;
                let b2 = self.subst_lag(b, endo, table, neweqs)?;
                let c2 = self.subst_lag(c, endo, table, neweqs)?;
                Ok(self.build_tri(op, a2, b2, c2))
            }
            N::Ext(f, l) => {
                let mut v = Vec::new();
                for a in self.ext_args(l) {
                    v.push(self.subst_lag(a, endo, table, neweqs)?);
                }
                Ok(self.add_ext(f, v))
            }
        }
    }

    /// `DynamicModel::substituteLeadLagInternal`. `kind`: 0 endo lead, 1 endo lag, 2 exo lead, 3 exo lag.
    fn substitute_lead_lag(&mut self, kind: u8, det: bool) -> Result<()> {
        let mut table = SubstTable::new();
        let mut neweqs = Vec::new();
        let run = |t: &mut Tree, id: Id, table: &mut SubstTable, neweqs: &mut Vec<Id>| -> Result<Id> {
            match kind {
                0 => t.subst_lead(id, true, det, table, neweqs),
                2 => t.subst_lead(id, false, det, table, neweqs),
                1 => t.subst_lag(id, true, table, neweqs),
                _ => t.subst_lag(id, false, table, neweqs),
            }
        };
        for s in self.used_locals() {
            let v = self.locals[&s];
            let r = run(self, v, &mut table, &mut neweqs)?;
            self.set_local(s, r);
        }
        for k in 0..self.equations.len() {
            let e = self.equations[k].eq;
            self.equations[k].eq = run(self, e, &mut table, &mut neweqs)?;
        }
        self.push_neweqs(neweqs);
        Ok(())
    }

    /// `differentiateForwardVars`: `x(+1)` -> `x + AUX_DIFF_FWRD_k(+1)`, `AUX_DIFF_FWRD_k = x - x(-1)`.
    fn diff_forward(&mut self, id: Id, subset: &[String], table: &mut SubstTable, neweqs: &mut Vec<Id>) -> Result<Id> {
        match self.node(id) {
            N::Num(_) => Ok(id),
            N::Var(s, lag) => match self.sym_ty(s) {
                SymType::Endo => {
                    if lag <= 0 || (!subset.is_empty() && !subset.iter().any(|n| *n == self.names[s as usize])) {
                        return Ok(id);
                    }
                    let diffvar = match table.get(&id) {
                        Some(&d) => d,
                        None => {
                            let x0 = self.add_var(s, 0);
                            let x1 = self.add_var(s, -1);
                            let substexpr = self.add_minus(x0, x1);
                            let aux = self.add_aux(format!("AUX_DIFF_FWRD_{}", s + 1), 5, Some(s), Some(0), substexpr);
                            let a0 = self.add_var(aux, 0);
                            neweqs.push(self.add_equal(a0, substexpr));
                            let d = self.add_var(aux, 1);
                            table.insert(id, d);
                            d
                        }
                    };
                    let x0 = self.add_var(s, 0);
                    Ok(self.add_plus(x0, diffvar))
                }
                SymType::Local => {
                    let v = self.get_local(s, lag);
                    if self.max_endo_lead(v) <= 0 { Ok(id) } else { self.diff_forward(v, subset, table, neweqs) }
                }
                _ => Ok(id),
            },
            N::Un(op, a, info) => {
                let a2 = self.diff_forward(a, subset, table, neweqs)?;
                Ok(self.build_un(op, info, a2))
            }
            N::Bin(op, a, b) => {
                let a2 = self.diff_forward(a, subset, table, neweqs)?;
                let b2 = self.diff_forward(b, subset, table, neweqs)?;
                self.build_bin(op, a2, b2)
            }
            N::Tri(op, a, b, c) => {
                let a2 = self.diff_forward(a, subset, table, neweqs)?;
                let b2 = self.diff_forward(b, subset, table, neweqs)?;
                let c2 = self.diff_forward(c, subset, table, neweqs)?;
                Ok(self.build_tri(op, a2, b2, c2))
            }
            N::Ext(f, l) => {
                let mut v = Vec::new();
                for a in self.ext_args(l) {
                    v.push(self.diff_forward(a, subset, table, neweqs)?);
                }
                Ok(self.add_ext(f, v))
            }
        }
    }

    fn differentiate_forward_vars(&mut self, subset: &[String]) -> Result<()> {
        let mut table = SubstTable::new();
        let mut neweqs = Vec::new();
        for s in self.used_locals() {
            let v = self.locals[&s];
            let r = self.diff_forward(v, subset, &mut table, &mut neweqs)?;
            self.set_local(s, r);
        }
        for k in 0..self.equations.len() {
            let e = self.equations[k].eq;
            self.equations[k].eq = self.diff_forward(e, subset, &mut table, &mut neweqs)?;
        }
        self.push_neweqs(neweqs);
        Ok(())
    }

    // ---- conversion to the crate's DAG ------------------------------------------------------
    fn convert(&mut self, id: Id, dag: &mut Dag, memo: &mut HashMap<Id, NodeId>) -> Result<NodeId> {
        if let Some(&r) = memo.get(&id) {
            return Ok(r);
        }
        let r = match self.node(id) {
            N::Num(c) => dag.num(self.consts[c as usize].1),
            N::Var(s, lag) => {
                let sym = self.syms[s as usize].clone();
                match sym.ty {
                    SymType::Endo => dag.endo(sym.index, lag),
                    SymType::Exo => dag.exo(sym.index, lag),
                    SymType::Param => dag.param(sym.index),
                    SymType::AggEndo => dag.endo(self.agg_offsets.0 + sym.index, lag),
                    SymType::AggExo => dag.exo(self.agg_offsets.1 + sym.index, lag),
                    SymType::Local => {
                        let v = self.get_local(s, lag);
                        self.convert(v, dag, memo)?
                    }
                    SymType::Excluded => {
                        bail!("'{}' was removed by var_remove/model_remove but is still used", self.names[s as usize])
                    }
                    // only in the balanced-growth test (trend variables as parameters)
                    SymType::Trend | SymType::LogTrend => match self.trend_param_offset {
                        Some(off) => dag.param(off + sym.index),
                        None => {
                            bail!("internal error: trend variable '{}' left after detrending", self.names[s as usize])
                        }
                    },
                }
            }
            N::Un(op, a, _) => {
                let a = self.convert(a, dag, memo)?;
                match op {
                    UOp::Uminus => dag.neg(a),
                    UOp::F(f) => dag.unary(f, a),
                    UOp::SteadyState => dag.steady_state(a),
                    UOp::Expectation => bail!("internal error: EXPECTATION left after its substitution"),
                    UOp::Diff => bail!("internal error: diff() left after its substitution"),
                }
            }
            N::Bin(op, a, b) => {
                let a = self.convert(a, dag, memo)?;
                let b = self.convert(b, dag, memo)?;
                let o = match op {
                    BOp::Plus => BinaryOp::Add,
                    BOp::Minus => BinaryOp::Sub,
                    BOp::Times => BinaryOp::Mul,
                    BOp::Divide => BinaryOp::Div,
                    BOp::Power => BinaryOp::Pow,
                    BOp::Max => BinaryOp::Max,
                    BOp::Min => BinaryOp::Min,
                    BOp::Less => BinaryOp::Lt,
                    BOp::Greater => BinaryOp::Gt,
                    BOp::LessEqual => BinaryOp::Le,
                    BOp::GreaterEqual => BinaryOp::Ge,
                    BOp::EqualEqual => BinaryOp::Eq,
                    BOp::Different => BinaryOp::Ne,
                    BOp::PowerDeriv(k) => BinaryOp::PowerDeriv(k),
                    BOp::Equal => bail!("internal error: nested equation"),
                };
                dag.binary(o, a, b)
            }
            N::Tri(op, a, b, c) => {
                let v = [self.convert(a, dag, memo)?, self.convert(b, dag, memo)?, self.convert(c, dag, memo)?];
                special_to_node(dag, if op == TOp::Normcdf { SpecialOp::Normcdf } else { SpecialOp::Normpdf }, &v)?
            }
            N::Ext(f, l) => {
                let mut v = Vec::new();
                for a in self.ext_args(l) {
                    v.push(self.convert(a, dag, memo)?);
                }
                dag.external(f, v)
            }
        };
        memo.insert(id, r);
        Ok(r)
    }
}

/// Inputs of [`transform`] taken from the `ModelBuilder`.
pub(crate) struct Input<'a> {
    pub items: &'a [ModelItem],
    /// By symbol id: type and type-specific index; with the names.
    pub syms: Vec<Sym>,
    pub names: Vec<String>,
    pub n_endo: usize,
    pub lookup: &'a dyn Fn(&str) -> Option<u32>,
    pub predetermined: Vec<u32>,
    pub stochastic: bool,
    /// `model(differentiate_forward_vars[=(...)])`: None off, empty = all variables.
    pub differentiate_forward_vars: Option<Vec<String>>,
    /// Symbol ids of the `var(log)` variables.
    pub log_vars: Vec<u32>,
    /// Context of the balanced-growth test (`global_eval_context`).
    pub bg: BgContext,
    /// Ramsey step ([`PlannerIn`]), when the model has `ramsey_model`.
    pub planner: Option<PlannerIn>,
}

/// what `DynamicModel::testTrendDerivativesEqualToZero` evaluates at: the
/// calibration and the `initval` values (0 when unset), trend variables at 2.
#[derive(Clone, Debug, Default)]
pub struct BgContext {
    pub params: Vec<f64>,
    pub endo: Vec<f64>,
    pub exo: Vec<f64>,
    /// `model(balanced_growth_test_tol = ...)` (1e-6).
    pub tol: f64,
}

/// Replays the model block and runs the transform pass; converts the result into `dag`.
pub(crate) fn transform(inp: Input<'_>, dag: &mut Dag) -> Result<Transformed> {
    let mut t = Tree::new(inp.syms, inp.names, inp.n_endo);
    t.predetermined = inp.predetermined.into_iter().collect();
    let lookup = inp.lookup;
    let mut q4_lines: Vec<Option<usize>> = Vec::new(); // JSON "line"
    for item in inp.items {
        match item {
            ModelItem::Local { symb_id, ast, nums } => {
                let mut it = nums.iter();
                let v = t.replay(ast, &mut it, lookup)?;
                t.set_local(*symb_id as u32, v);
            }
            ModelItem::Equation { lhs, rhs, nums, tags, line } => {
                let mut it = nums.iter();
                let l = t.replay(lhs, &mut it, lookup)?;
                let r = match rhs {
                    Some(r) => t.replay(r, &mut it, lookup)?,
                    None => t.zero,
                };
                let before = t.equations.len();
                t.add_model_equal(l, r, tags.clone(), lookup)?;
                if t.equations.len() > before {
                    q4_lines.push(first_line(lhs).or_else(|| rhs.as_ref().and_then(first_line)));
                    t.equations.last_mut().unwrap().line = *line;
                }
            }
            ModelItem::Trend { vars, ast, nums, log } => {
                let mut it = nums.iter();
                let g = t.replay(ast, &mut it, lookup)?;
                for v in vars {
                    let s = lookup(v).ok_or_else(|| anyhow!("unknown trend variable '{v}'"))?;
                    t.trend_map.insert(s, g);
                    let _ = log;
                }
            }
            ModelItem::Deflator { vars, ast, nums, log } => {
                let mut it = nums.iter();
                let d = t.replay(ast, &mut it, lookup)?;
                let mut endo = BTreeSet::new();
                t.collect_vars(d, SymType::Endo, &mut endo, &mut BTreeSet::new());
                if !endo.is_empty() {
                    bail!("The deflator contains an endogenous variable");
                }
                for v in vars {
                    let s = lookup(v).ok_or_else(|| anyhow!("unknown variable '{v}'"))?;
                    t.nonstationary.insert(s, (*log, d));
                }
            }
            ModelItem::Remove { selection } => {
                let m = |e: &Equation| {
                    selection.iter().any(|sel| sel.iter().all(|(k, v)| e.tags.iter().any(|(a, b)| a == k && b == v)))
                };
                let keep: Vec<bool> = t.equations.iter().map(|e| !m(e)).collect();
                let mut k = 0;
                q4_lines.retain(|_| {
                    k += 1;
                    keep.get(k - 1).copied().unwrap_or(true)
                });
                t.equations.retain(|e| !m(e));
                t.static_only.retain(|e| !m(e));
            }
        }
    }
    if json_on() {
        let parsed = t.json_equations(&q4_lines, true); // json=parse
        JSON_CAPTURE.with(|c| c.borrow_mut().as_mut().unwrap().parsed = parsed);
    }
    // ModFile::transformPass
    t.transform_predetermined()?;
    t.simplify_equations()?;
    if json_on() {
        let orig = t.json_equations(&q4_lines, true); // modfile-original.json
        JSON_CAPTURE.with(|c| c.borrow_mut().as_mut().unwrap().original = orig);
    }
    let q4_n_orig = t.equations.len();
    t.expand_eq_tags()?;
    t.substitute_model_local_variables()?;
    let mut ramsey_focs = Vec::new();
    if let Some(p) = &inp.planner {
        let n = t.ramsey_focs(p, lookup)?;
        q4_lines.splice(0..0, std::iter::repeat_n(None, n));
        let mut memo = HashMap::new();
        for k in 0..n {
            let (f, _) = t.eq_args(t.equations[k].eq);
            ramsey_focs.push(t.convert(f, dag, &mut memo)?);
        }
    }
    let (mut max_lead, mut max_lag) = (0, 0);
    {
        let mut vars = BTreeSet::new();
        for k in 0..t.equations.len() {
            let e = t.equations[k].eq;
            let mut seen = BTreeSet::new();
            t.collect_vars(e, SymType::Endo, &mut vars, &mut seen);
            let mut seen = BTreeSet::new();
            t.collect_vars(e, SymType::Exo, &mut vars, &mut seen);
        }
        for (_, l) in vars {
            max_lead = max_lead.max(l);
            max_lag = max_lag.max(-l);
        }
    }
    let mut max_lag_with_diffs = 0;
    for k in 0..t.equations.len() {
        let e = t.equations[k].eq;
        max_lag_with_diffs = max_lag_with_diffs.max(t.mx(Mx::LagDiffs, e));
    }
    t.substitute_diff()?;
    t.substitute_expectation()?;
    if json_on() {
        let map = t.json_mapping(q4_n_orig); // variable_mapping
        JSON_CAPTURE.with(|c| c.borrow_mut().as_mut().unwrap().mapping = map);
    }
    // ---- balanced growth (detrendEquations, runTrendTest, removeTrendVariableFromEquations)
    // and var(log) (substituteLogTransform)
    if !t.nonstationary.is_empty() || !t.trend_map.is_empty() {
        t.detrend_equations()?;
        t.trend_test(&inp.bg)?;
        t.remove_trend_variables()?;
    }
    t.substitute_log_transform(&inp.log_vars)?;
    let det = !inp.stochastic;
    t.substitute_lead_lag(0, det)?;
    t.substitute_lead_lag(2, det)?;
    t.substitute_lead_lag(1, det)?;
    t.substitute_lead_lag(3, det)?;
    if let Some(subset) = &inp.differentiate_forward_vars {
        t.differentiate_forward_vars(subset)?;
    }
    if json_on() {
        // modfile.json: the transformed model, aux equations last
        let tr = t.json_equations(&q4_lines, true);
        let orig: Vec<String> = t.aux.iter().map(|a| t.json_str(a.shown.unwrap_or(a.rhs))).collect();
        JSON_CAPTURE.with(|c| {
            let mut b = c.borrow_mut();
            let cap = b.as_mut().unwrap();
            cap.transformed = tr;
            cap.aux_orig_expr = orig;
        });
    }

    // conversion
    let mut memo = HashMap::new();
    let mut equations = Vec::with_capacity(t.equations.len());
    let mut tags = Vec::with_capacity(t.equations.len());
    let eqs = t.equations.clone();
    for e in &eqs {
        let (a, b) = t.eq_args(e.eq);
        equations.push((t.convert(a, dag, &mut memo)?, t.convert(b, dag, &mut memo)?));
        tags.push(e.tags.clone());
    }
    let mut static_override = vec![None; eqs.len()];
    let mut static_eqs = vec![None; eqs.len()]; // model_info
    let statics = t.static_only.clone();
    let dynamic: Vec<usize> =
        (0..eqs.len()).filter(|&i| eqs[i].tags.iter().any(|(k, v)| k == "dynamic" && v.is_empty())).collect();
    if dynamic.len() != statics.len() {
        bail!(
            "the model has {} equation(s) tagged [dynamic] but {} tagged [static]; their numbers must match",
            dynamic.len(),
            statics.len()
        );
    }
    for (&i, s) in dynamic.iter().zip(&statics) {
        let (a, b) = t.eq_args(s.eq);
        let (a, b) = (t.convert(a, dag, &mut memo)?, t.convert(b, dag, &mut memo)?);
        static_override[i] = Some(dag.sub(a, b));
        static_eqs[i] = Some((a, b)); // model_info
    }
    let mut aux = Vec::with_capacity(t.aux.len());
    let recs: Vec<_> =
        t.aux.iter().map(|a| (a.symb, a.name.clone(), a.ty, a.orig_symb, a.orig_lead_lag, a.rhs, a.shown)).collect();
    let mut shown = Vec::with_capacity(recs.len());
    for (symb, name, ty, os, ol, rhs, sh) in recs {
        shown.push(match sh.map(|id| t.node(id)) {
            Some(N::Un(UOp::Diff, a, _)) => t.convert(a, dag, &mut memo).ok().map(|a| ("diff".to_string(), a)),
            Some(N::Un(UOp::Expectation, a, k)) => {
                t.convert(a, dag, &mut memo).ok().map(|a| (format!("EXPECTATION({k})"), a))
            }
            _ => None,
        });
        let rhs = t.convert(rhs, dag, &mut memo)?;
        let orig_index = os.map(|s| t.syms[s as usize].index);
        let (orig_index, orig_lead_lag) = match ty {
            1 | 3 | 5 | 7 | 8 | 9 => (orig_index, ol),
            _ => (None, None),
        };
        aux.push((
            name,
            AuxVar { endo_index: t.syms[symb as usize].index, ty, orig_index, orig_lead_lag, orig_expr: String::new() },
            rhs,
        ));
    }
    Ok(Transformed {
        equations,
        tags,
        static_override,
        static_eqs,
        aux,
        shown,
        orig_max_lead: max_lead,
        orig_max_lag: max_lag,
        orig_max_lag_with_diffs: max_lag_with_diffs,
        ramsey_focs,
    })
}

// ---- JSON output of the model (ExprNode::writeJsonOutput) ------------------------

/// An equation as the preprocessor writes it in `modfile*.json`.
#[derive(Clone, Debug)]
pub struct JsonEquation {
    pub lhs: String,
    pub rhs: String,
    pub line: Option<usize>,
    pub tags: Vec<(String, String)>,
    /// `abstract_syntax_tree` entry (original model only).
    pub ast: Option<serde_json::Value>,
}

/// What [`transform`] records when [`json_capture`] is on: the model after parsing
/// (`modfile-original.json`) and after the transform pass (`modfile.json`, aux equations last).
#[derive(Clone, Debug, Default)]
pub struct JsonCapture {
    /// As parsed (`json=parse|check`).
    pub parsed: Vec<JsonEquation>,
    /// `modfile-original.json`: after the predetermined-variable and constant-equation steps
    /// of the transform pass, before the default equation names and the substitutions.
    pub original: Vec<JsonEquation>,
    /// `variable_mapping`: endogenous / exogenous variables (symbol-id order, incl. the
    /// diff / expectation aux variables) and the names of the original equations using them,
    /// before the lead/lag substitutions.
    pub mapping: Vec<(String, Vec<String>)>,
    pub transformed: Vec<JsonEquation>,
    /// `M_.aux_vars(k).orig_expr` as printed by the preprocessor.
    pub aux_orig_expr: Vec<String>,
}

thread_local! {
    static JSON_CAPTURE: std::cell::RefCell<Option<JsonCapture>> = const { std::cell::RefCell::new(None) };
}

/// Turns the capture on (`json=...`) for the next [`transform`] on this thread.
pub fn json_capture() {
    JSON_CAPTURE.with(|c| *c.borrow_mut() = Some(JsonCapture::default()));
}

/// The capture of the last [`transform`] (and turns it off).
pub fn json_take() -> Option<JsonCapture> {
    JSON_CAPTURE.with(|c| c.borrow_mut().take())
}

fn json_on() -> bool {
    JSON_CAPTURE.with(|c| c.borrow().is_some())
}

impl Tree {
    /// `precedenceJson`.
    fn json_prec(&self, id: Id) -> i32 {
        match self.node(id) {
            N::Num(_) | N::Var(..) | N::Tri(..) | N::Ext(..) => 100,
            N::Un(UOp::Uminus, ..) => 6,
            N::Un(..) => 100,
            N::Bin(op, ..) => match op {
                BOp::Equal => 0,
                BOp::EqualEqual | BOp::Different => 1,
                BOp::Less | BOp::Greater | BOp::LessEqual | BOp::GreaterEqual => 2,
                BOp::Plus | BOp::Minus => 3,
                BOp::Times | BOp::Divide => 4,
                BOp::Power => 5,
                BOp::Max | BOp::Min | BOp::PowerDeriv(_) => 100,
            },
        }
    }

    fn bop_str(op: BOp) -> &'static str {
        match op {
            BOp::Plus => "+",
            BOp::Minus => "-",
            BOp::Times => "*",
            BOp::Divide => "/",
            BOp::Power => "^",
            BOp::Equal => "=",
            BOp::Max => "max",
            BOp::Min => "min",
            BOp::Less => "<",
            BOp::Greater => ">",
            BOp::LessEqual => "<=",
            BOp::GreaterEqual => ">=",
            BOp::EqualEqual => "==",
            BOp::Different => "!=",
            BOp::PowerDeriv(_) => "get_power_deriv",
        }
    }

    /// `ExprNode::writeJsonOutput` (infix, minimal parentheses).
    fn json_write(&self, id: Id, out: &mut String) {
        match self.node(id) {
            N::Num(c) => out.push_str(&self.consts[c as usize].0),
            N::Var(s, lag) => {
                out.push_str(&self.names[s as usize]);
                if lag != 0 {
                    out.push_str(&format!("({lag})"));
                }
            }
            N::Un(op, a, k) => match op {
                UOp::Uminus => {
                    // always parenthesized; the argument too if it binds less tightly
                    out.push_str("(-");
                    let p = self.json_prec(a) < 6;
                    if p {
                        out.push('(');
                    }
                    self.json_write(a, out);
                    if p {
                        out.push(')');
                    }
                    out.push(')');
                }
                _ => {
                    out.push_str(&match op {
                        UOp::F(f) => f.name().to_string(),
                        UOp::SteadyState => "STEADY_STATE".into(),
                        UOp::Expectation => format!("EXPECTATION({k})"),
                        UOp::Diff => "diff".into(),
                        UOp::Uminus => unreachable!(),
                    });
                    out.push('(');
                    self.json_write(a, out);
                    out.push(')');
                }
            },
            N::Bin(op @ (BOp::Max | BOp::Min), a, b) => {
                out.push_str(Self::bop_str(op));
                out.push('(');
                self.json_write(a, out);
                out.push(',');
                self.json_write(b, out);
                out.push(')');
            }
            N::Bin(BOp::PowerDeriv(k), a, b) => {
                out.push_str("get_power_deriv(");
                self.json_write(a, out);
                out.push(',');
                self.json_write(b, out);
                out.push_str(&format!(",{k})"));
            }
            N::Bin(op, a, b) => {
                let prec = self.json_prec(id);
                let pa = self.json_prec(a);
                let p1 = pa < prec || (op == BOp::Power && pa == prec);
                if p1 {
                    out.push('(');
                }
                self.json_write(a, out);
                if p1 {
                    out.push(')');
                }
                out.push_str(Self::bop_str(op));
                let pb = self.json_prec(b);
                let b_pow = matches!(self.node(b), N::Bin(BOp::Power, ..));
                let p2 = pb < prec
                    || (op == BOp::Power && b_pow)
                    || (op == BOp::Minus && pb == prec)
                    || (op == BOp::Divide && pb == prec);
                if p2 {
                    out.push('(');
                }
                self.json_write(b, out);
                if p2 {
                    out.push(')');
                }
            }
            N::Tri(op, a, b, c) => {
                out.push_str(if op == TOp::Normcdf { "normcdf(" } else { "normpdf(" });
                self.json_write(a, out);
                out.push(',');
                self.json_write(b, out);
                out.push(',');
                self.json_write(c, out);
                out.push(')');
            }
            N::Ext(f, l) => {
                out.push_str(crate::model::external::name(f));
                out.push('(');
                for (k, a) in self.ext_args(l).into_iter().enumerate() {
                    if k > 0 {
                        out.push_str(", ");
                    }
                    self.json_write(a, out);
                }
                out.push(')');
            }
        }
    }

    fn json_str(&self, id: Id) -> String {
        let mut s = String::new();
        self.json_write(id, &mut s);
        s
    }

    /// `writeJsonAST`.
    fn json_ast(&self, id: Id) -> serde_json::Value {
        use serde_json::json;
        match self.node(id) {
            N::Num(c) => {
                let (t, v) = &self.consts[c as usize];
                let num = if v.fract() == 0.0 && v.abs() < 1e15 {
                    json!(*v as i64)
                } else {
                    serde_json::Number::from_f64(*v).map(serde_json::Value::Number).unwrap_or_else(|| json!(t))
                };
                json!({"node_type": "NumConstNode", "value": num})
            }
            N::Var(s, lag) => {
                let ty = match self.sym_ty(s) {
                    SymType::Endo => "endogenous",
                    SymType::Exo => "exogenous",
                    SymType::Param => "parameter",
                    SymType::Local => "modelLocalVariable",
                    SymType::Excluded => "excludedVariable",
                    SymType::Trend => "trendVariable", // preprocessor's writeJsonAST
                    SymType::LogTrend => "logTrendVariable",
                    SymType::AggEndo => "endogenous",
                    SymType::AggExo => "exogenous",
                };
                json!({"node_type": "VariableNode", "name": self.names[s as usize], "type": ty, "lag": lag})
            }
            N::Un(op, a, k) => {
                let _ = k;
                let name = match op {
                    UOp::Uminus => "uminus".to_string(),
                    UOp::F(f) => f.name().to_string(),
                    UOp::SteadyState => "steady_state".into(),
                    UOp::Expectation => "expectation".into(),
                    UOp::Diff => "diff".into(),
                };
                json!({"node_type": "UnaryOpNode", "op": name, "arg": self.json_ast(a)})
            }
            N::Bin(op, a, b) => {
                json!({"node_type": "BinaryOpNode", "op": Self::bop_str(op), "arg1": self.json_ast(a), "arg2": self.json_ast(b)})
            }
            N::Tri(op, a, b, c) => json!({
                "node_type": "TrinaryOpNode",
                "op": if op == TOp::Normcdf { "normcdf" } else { "normpdf" },
                "arg1": self.json_ast(a), "arg2": self.json_ast(b), "arg3": self.json_ast(c)
            }),
            N::Ext(f, l) => json!({
                "node_type": "ExternalFunctionNode",
                "name": crate::model::external::name(f),
                "args": self.ext_args(l).into_iter().map(|a| self.json_ast(a)).collect::<Vec<_>>()
            }),
        }
    }

    fn json_mapping(&mut self, n_orig: usize) -> Vec<(String, Vec<String>)> {
        let mut per_eq = Vec::new();
        for k in 0..n_orig.min(self.equations.len()) {
            let e = self.equations[k].eq;
            let mut vars = BTreeSet::new();
            let mut seen = BTreeSet::new();
            self.collect_vars(e, SymType::Endo, &mut vars, &mut seen);
            let mut seen = BTreeSet::new();
            self.collect_vars(e, SymType::Exo, &mut vars, &mut seen);
            let name = self.equations[k]
                .tags
                .iter()
                .find(|(t, _)| t == "name")
                .map_or((k + 1).to_string(), |(_, v)| v.clone());
            per_eq.push((name, vars.into_iter().map(|(s, _)| s).collect::<BTreeSet<u32>>()));
        }
        (0..self.syms.len() as u32)
            .filter(|&s| matches!(self.sym_ty(s), SymType::Endo | SymType::Exo))
            .map(|s| {
                (
                    self.names[s as usize].clone(),
                    per_eq.iter().filter(|(_, v)| v.contains(&s)).map(|(n, _)| n.clone()).collect(),
                )
            })
            .collect()
    }

    fn json_equations(&self, lines: &[Option<usize>], with_ast: bool) -> Vec<JsonEquation> {
        self.equations
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let (a, b) = self.eq_args(e.eq);
                JsonEquation {
                    lhs: self.json_str(a),
                    rhs: self.json_str(b),
                    line: lines.get(i).copied().flatten(),
                    tags: e.tags.clone(),
                    ast: with_ast.then(|| self.json_ast(e.eq)),
                }
            })
            .collect()
    }
}

/// Line of the first variable of an equation (the preprocessor's equation line).
fn first_line(a: &Ast) -> Option<usize> {
    match a {
        Ast::Num(_) => None,
        Ast::Var { loc, .. } => Some(loc.line),
        Ast::Un(_, x) => first_line(x),
        Ast::Bin(_, x, y) => first_line(x).or_else(|| first_line(y)),
        Ast::Op(_, args) => args.iter().find_map(first_line),
    }
}

#[cfg(test)]
mod tests {
    use super::fmt_g16;

    #[test]
    fn g16_formatting() {
        assert_eq!(fmt_g16(0.5), "0.5");
        assert_eq!(fmt_g16(6.0), "6");
        assert_eq!(fmt_g16(1.0 / 3.0), "0.3333333333333333");
        assert_eq!(fmt_g16(1e-5), "1e-05");
        assert_eq!(fmt_g16(1.5e20), "1.5e+20");
        assert_eq!(fmt_g16(123456.0), "123456");
        assert_eq!(fmt_g16(0.0001), "0.0001");
    }
}
