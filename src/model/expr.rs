// SPDX-License-Identifier: GPL-3.0-or-later
//! Hash-consed expression DAG, the counterpart of Dynare's `DataTree` / `ExprNode`.
//!
//! Every node is interned once (structurally identical expressions share one
//! [`NodeId`]), and the constructors apply the same light simplifications as
//! Dynare's `DataTree::Add*` (x+0, x*1, x*0, x-x, -(-x), ...) plus constant folding.
//! Symbolic derivatives ([`Dag::diff`]) follow `ExprNode::computeDerivative`.

use std::collections::HashMap;

/// Index of a node in a [`Dag`].
pub type NodeId = u32;

/// Unary operators (Dynare's `UnaryOpcode` subset).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Exp,
    Log,
    Log10,
    Sqrt,
    Abs,
    Sign,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    // ---- stream PP: remaining Dynare unary operators
    Asinh,
    Acosh,
    Atanh,
    Cbrt,
    Erf,
    Erfc,
}

// erf/erfc from the system libm (MATLAB's erf/erfc agree to the last bits on Apple's libm).
unsafe extern "C" {
    fn erf(x: f64) -> f64;
    fn erfc(x: f64) -> f64;
}

impl UnaryOp {
    /// Maps a .mod function name to the operator (`ln` is an alias of `log`).
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "exp" => Self::Exp,
            "log" | "ln" => Self::Log,
            "log10" => Self::Log10,
            "sqrt" => Self::Sqrt,
            "abs" => Self::Abs,
            "sign" => Self::Sign,
            "sin" => Self::Sin,
            "cos" => Self::Cos,
            "tan" => Self::Tan,
            "asin" => Self::Asin,
            "acos" => Self::Acos,
            "atan" => Self::Atan,
            "sinh" => Self::Sinh,
            "cosh" => Self::Cosh,
            "tanh" => Self::Tanh,
            "asinh" => Self::Asinh,
            "acosh" => Self::Acosh,
            "atanh" => Self::Atanh,
            "cbrt" => Self::Cbrt,
            "erf" => Self::Erf,
            "erfc" => Self::Erfc,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Neg => "-",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Log10 => "log10",
            Self::Sqrt => "sqrt",
            Self::Abs => "abs",
            Self::Sign => "sign",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Asinh => "asinh",
            Self::Acosh => "acosh",
            Self::Atanh => "atanh",
            Self::Cbrt => "cbrt",
            Self::Erf => "erf",
            Self::Erfc => "erfc",
        }
    }

    #[inline]
    pub fn apply(self, a: f64) -> f64 {
        match self {
            Self::Neg => -a,
            Self::Exp => a.exp(),
            Self::Log => a.ln(),
            Self::Log10 => a.log10(),
            Self::Sqrt => a.sqrt(),
            Self::Abs => a.abs(),
            // MATLAB's sign: 0 at 0, NaN at NaN.
            Self::Sign => {
                if a > 0.0 {
                    1.0
                } else if a < 0.0 {
                    -1.0
                } else {
                    a * 0.0
                }
            }
            Self::Sin => a.sin(),
            Self::Cos => a.cos(),
            Self::Tan => a.tan(),
            Self::Asin => a.asin(),
            Self::Acos => a.acos(),
            Self::Atan => a.atan(),
            Self::Sinh => a.sinh(),
            Self::Cosh => a.cosh(),
            Self::Tanh => a.tanh(),
            Self::Asinh => a.asinh(),
            Self::Acosh => a.acosh(),
            Self::Atanh => a.atanh(),
            Self::Cbrt => a.cbrt(),
            // SAFETY: plain libm functions on a double.
            Self::Erf => unsafe { erf(a) },
            Self::Erfc => unsafe { erfc(a) },
        }
    }
}

/// Binary operators (Dynare's `BinaryOpcode` subset). Comparisons return 1.0 / 0.0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    /// `getPowerDeriv(x, p, k)`: the `k`-th derivative of `x^p` w.r.t. `x`, the preprocessor's
    /// `powerDeriv` node (derivatives of a power whose exponent is not a numeric constant).
    PowerDeriv(u8),
}

/// Port of `getPowerDeriv.m`: `d^k (x^p) / dx^k`, exactly 0 at `x = 0` for an integer
/// `p >= 0` below the order (where `x^(p-k) * p * (p-1) * ...` would give `Inf * 0`).
pub fn get_power_deriv(x: f64, p: f64, k: u8) -> f64 {
    let kf = f64::from(k);
    if x.abs() < 1e-12 && p >= 0.0 && kf > p && (p - p.round()).abs() < 1e-12 {
        0.0
    } else {
        let mut dxp = x.powf(p - kf);
        let mut p = p;
        for _ in 0..k {
            dxp *= p;
            p -= 1.0;
        }
        dxp
    }
}

impl BinaryOp {
    #[inline]
    pub fn apply(self, a: f64, b: f64) -> f64 {
        let t = |c: bool| if c { 1.0 } else { 0.0 };
        match self {
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
            Self::Div => a / b,
            Self::Pow => a.powf(b),
            Self::Max => {
                if a >= b || b.is_nan() {
                    a
                } else {
                    b
                }
            }
            Self::Min => {
                if a <= b || b.is_nan() {
                    a
                } else {
                    b
                }
            }
            Self::Lt => t(a < b),
            Self::Gt => t(a > b),
            Self::Le => t(a <= b),
            Self::Ge => t(a >= b),
            Self::Eq => t(a == b),
            Self::Ne => t(a != b),
            Self::PowerDeriv(k) => get_power_deriv(a, b, k),
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Pow => "^",
            Self::Max => "max",
            Self::Min => "min",
            Self::Lt => "<",
            Self::Gt => ">",
            Self::Le => "<=",
            Self::Ge => ">=",
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::PowerDeriv(_) => "getPowerDeriv",
        }
    }

    /// Binding strength used by the printer (higher binds tighter).
    fn precedence(self) -> u8 {
        match self {
            Self::Lt | Self::Gt | Self::Le | Self::Ge | Self::Eq | Self::Ne => 1,
            Self::Add | Self::Sub => 2,
            Self::Mul | Self::Div => 3,
            Self::Pow => 5,
            Self::Max | Self::Min | Self::PowerDeriv(_) => 6,
        }
    }
}

/// A DAG node. `Num` stores the bits of an `f64` so that nodes can be hashed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    Num(u64),
    /// Endogenous variable (type-specific index) at a lead (>0) or lag (<0).
    Endo(u32, i32),
    /// Exogenous variable (type-specific index) at a lead or lag.
    Exo(u32, i32),
    Param(u32),
    /// `STEADY_STATE(x(lag))` of endogenous variable `x` (stream PP; the lag only counts for the
    /// lead/lag incidence, as Dynare's deriv ids): the steady-state value of
    /// the variable, a constant for the dynamic model (derivative 0 w.r.t. every dynamic
    /// variable), equal to `x` in the static model. Compiled tapes read it from the extended
    /// parameter vector `[params; ys]` at `ss_offset + j` (see [`Dag::ss_offset`]).
    SteadyState(u32, i32),
    /// `d STEADY_STATE(x) / d p` of endogenous variable `x` w.r.t. parameter `p`, the
    /// preprocessor's `steadyStateParamDeriv` node (`ss_param_deriv(x, p)` in
    /// `dynamic_params_derivs.m`): the `dys` of get_perturbation_params_derivs.m. Compiled
    /// tapes read it from the extended parameter vector `[params; ys; dys(:); d2ys(:)]`
    /// (see [`Dag::ss_param_deriv_slot`]).
    SteadyStateParamDeriv(u32, u32),
    /// `d2 STEADY_STATE(x) / d p1 d p2`, the preprocessor's `steadyStateParam2ndDeriv` node
    /// (`ss_param_2nd_deriv(x, p1, p2)`): the `d2ys` of get_perturbation_params_derivs.m.
    SteadyStateParam2ndDeriv(u32, u32, u32),
    Unary(UnaryOp, NodeId),
    Binary(BinaryOp, NodeId, NodeId),
    /// `external_function` call (or one of its derivatives): index into [`Dag::ext_call`].
    External(u32),
}

/// An external function call: function id in [`super::external`], arguments and derivative
/// indices (`d1 = d2 = 0`: the value; `d1 = i + 1, d2 = 0`: derivative w.r.t. argument `i`;
/// `d1 = i + 1, d2 = j + 1` with `i <= j`: second derivative), as the preprocessor's
/// `ExternalFunctionNode` / `FirstDerivExternalFunctionNode` / `SecondDerivExternalFunctionNode`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExtCall {
    pub fid: u32,
    pub args: Vec<NodeId>,
    pub d1: u16,
    pub d2: u16,
}

/// The variable a derivative is taken with respect to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DiffVar {
    Endo(u32, i32),
    Exo(u32, i32),
    Param(u32),
}

/// Per-node lead/lag summary, as `ExprNode::maxEndoLead` & co. (all >= 0).
#[derive(Clone, Copy, Debug, Default)]
struct Info {
    max_endo_lead: i32,
    max_endo_lag: i32,
    max_exo_lead: i32,
    max_exo_lag: i32,
}

/// Hash-consed expression arena.
#[derive(Clone, Debug)]
pub struct Dag {
    nodes: Vec<Node>,
    info: Vec<Info>,
    map: HashMap<Node, NodeId>,
    /// `external_function` calls referenced by `Node::External`.
    ext_calls: Vec<ExtCall>,
    ext_map: HashMap<ExtCall, u32>,
    /// Position of `STEADY_STATE(y_0)` in the parameter slice given to compiled tapes
    /// (the number of model parameters; set by `ModelBuilder::build`).
    pub ss_offset: u32,
    /// Number of endogenous variables, the length of the `ys` block of the extended parameter
    /// vector (set by `ModelBuilder::build`; locates the `dys` / `d2ys` blocks).
    pub ss_n_endo: u32,
    /// `M_.param_nbr`: `STEADY_STATE(y)` has a derivative w.r.t. `Param(q)` for
    /// `q < ss_param_nbr` only. The `varexo_det` stand-ins appended after the parameters (and
    /// the trend variables of the transform passes) are variables for the preprocessor, for
    /// which `STEADY_STATE(y)` is a constant (set by `ModelBuilder::build`; 0 before).
    pub ss_param_nbr: u32,
}

impl Default for Dag {
    fn default() -> Self {
        Self::new()
    }
}

impl Dag {
    pub const ZERO: NodeId = 0;
    pub const ONE: NodeId = 1;

    pub fn new() -> Self {
        let mut d = Dag {
            nodes: Vec::new(),
            info: Vec::new(),
            map: HashMap::new(),
            ext_calls: Vec::new(),
            ext_map: HashMap::new(),
            ss_offset: 0,
            ss_n_endo: 0,
            ss_param_nbr: 0,
        };
        d.intern(Node::Num(0f64.to_bits()));
        d.intern(Node::Num(1f64.to_bits()));
        d
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[inline]
    pub fn node(&self, id: NodeId) -> Node {
        self.nodes[id as usize]
    }

    /// Whether `id` is a `NumConstNode` of the preprocessor's DataTree: a constant that is not
    /// negative (the DataTree writes `-c` as `uminus(c)`, folded here into one constant).
    pub fn is_num_const(&self, id: NodeId) -> bool {
        self.as_num(id).is_some_and(|v| !(v < 0.0))
    }

    pub fn as_num(&self, id: NodeId) -> Option<f64> {
        match self.node(id) {
            Node::Num(b) => Some(f64::from_bits(b)),
            _ => None,
        }
    }

    fn intern(&mut self, n: Node) -> NodeId {
        if let Some(&id) = self.map.get(&n) {
            return id;
        }
        let info = match n {
            Node::Num(_)
            | Node::Param(_)
            | Node::SteadyState(..)
            | Node::SteadyStateParamDeriv(..)
            | Node::SteadyStateParam2ndDeriv(..) => Info::default(),
            Node::Endo(_, l) => Info { max_endo_lead: l.max(0), max_endo_lag: (-l).max(0), ..Info::default() },
            Node::Exo(_, l) => Info { max_exo_lead: l.max(0), max_exo_lag: (-l).max(0), ..Info::default() },
            Node::Unary(_, a) => self.info[a as usize],
            Node::Binary(_, a, b) => {
                let (x, y) = (self.info[a as usize], self.info[b as usize]);
                Info {
                    max_endo_lead: x.max_endo_lead.max(y.max_endo_lead),
                    max_endo_lag: x.max_endo_lag.max(y.max_endo_lag),
                    max_exo_lead: x.max_exo_lead.max(y.max_exo_lead),
                    max_exo_lag: x.max_exo_lag.max(y.max_exo_lag),
                }
            }
            Node::External(c) => self.ext_calls[c as usize].args.iter().fold(Info::default(), |x, &a| {
                let y = self.info[a as usize];
                Info {
                    max_endo_lead: x.max_endo_lead.max(y.max_endo_lead),
                    max_endo_lag: x.max_endo_lag.max(y.max_endo_lag),
                    max_exo_lead: x.max_exo_lead.max(y.max_exo_lead),
                    max_exo_lag: x.max_exo_lag.max(y.max_exo_lag),
                }
            }),
        };
        let id = self.nodes.len() as NodeId;
        self.nodes.push(n);
        self.info.push(info);
        self.map.insert(n, id);
        id
    }

    pub fn num(&mut self, v: f64) -> NodeId {
        let v = if v == 0.0 { 0.0 } else { v }; // fold -0.0
        self.intern(Node::Num(v.to_bits()))
    }
    pub fn endo(&mut self, j: usize, lag: i32) -> NodeId {
        self.intern(Node::Endo(j as u32, lag))
    }
    pub fn exo(&mut self, k: usize, lag: i32) -> NodeId {
        self.intern(Node::Exo(k as u32, lag))
    }
    pub fn param(&mut self, i: usize) -> NodeId {
        self.intern(Node::Param(i as u32))
    }

    /// `external_function` call `fid(args)` (no constant folding: the preprocessor's `eval`
    /// throws on external functions).
    pub fn external(&mut self, fid: u32, args: Vec<NodeId>) -> NodeId {
        self.external_deriv(fid, args, 0, 0)
    }

    /// A derivative node of an external function (see [`ExtCall`]).
    pub fn external_deriv(&mut self, fid: u32, args: Vec<NodeId>, d1: u16, d2: u16) -> NodeId {
        let (d1, d2) = if d2 != 0 && d2 < d1 { (d2, d1) } else { (d1, d2) };
        let call = ExtCall { fid, args, d1, d2 };
        let c = match self.ext_map.get(&call) {
            Some(&c) => c,
            None => {
                let c = self.ext_calls.len() as u32;
                self.ext_calls.push(call.clone());
                self.ext_map.insert(call, c);
                c
            }
        };
        self.intern(Node::External(c))
    }

    /// The external call of a `Node::External(c)`.
    pub fn ext_call(&self, c: u32) -> &ExtCall {
        &self.ext_calls[c as usize]
    }

    /// Whether the DAG holds `external_function` calls.
    pub fn has_external(&self) -> bool {
        !self.ext_calls.is_empty()
    }

    /// Same call with other arguments (`id` itself when no argument changed).
    pub fn rebuild_external(&mut self, id: NodeId, c: u32, args: Vec<NodeId>) -> NodeId {
        let e = self.ext_call(c);
        if e.args == args {
            return id;
        }
        let (fid, d1, d2) = (e.fid, e.d1, e.d2);
        self.external_deriv(fid, args, d1, d2)
    }

    /// Chain rule through an external call: `sum_i d f / d arg_i * dargs[i]`, the partial
    /// derivatives being `FirstDeriv`/`SecondDeriv` nodes. `mul`/`add` build the products and
    /// sums (plain or with the preprocessor's static-model simplifications).
    pub fn diff_external(
        &mut self,
        c: u32,
        dargs: &[NodeId],
        mul: &mut dyn FnMut(&mut Dag, NodeId, NodeId) -> NodeId,
        add: &mut dyn FnMut(&mut Dag, NodeId, NodeId) -> NodeId,
    ) -> NodeId {
        let e = self.ext_call(c).clone();
        if e.d2 != 0 && dargs.iter().any(|&d| d != Self::ZERO) {
            // the callers check `has_external` first (`Model::check_external_order`); this
            // unwinds to `external::catch_external_errors` if one does not
            super::external::fail(format!(
                "external function {}: third-order derivatives of external functions are not available \
                 (as in Dynare's preprocessor); use order <= 2",
                super::external::name(e.fid)
            ));
        }
        let mut acc = Self::ZERO;
        for (i, &da) in dargs.iter().enumerate() {
            if da == Self::ZERO {
                continue;
            }
            let p = if e.d1 == 0 {
                self.external_deriv(e.fid, e.args.clone(), i as u16 + 1, 0)
            } else {
                self.external_deriv(e.fid, e.args.clone(), e.d1, i as u16 + 1)
            };
            let t = mul(self, p, da);
            acc = if acc == Self::ZERO { t } else { add(self, acc, t) };
        }
        acc
    }

    /// `STEADY_STATE(e)` (stream PP): every endogenous variable of `e` (at any lead/lag)
    /// replaced by its steady-state value, as the preprocessor writes `steady_state(j)` for the
    /// variables under the operator. Exogenous variables are rejected by the parser (as Dynare).
    pub fn steady_state(&mut self, e: NodeId) -> NodeId {
        let mut memo = HashMap::new();
        self.map_vars(e, &mut memo, &mut |d, n| match n {
            Node::Endo(j, l) => Some(d.intern(Node::SteadyState(j, l))),
            _ => None,
        })
    }

    /// True if `id` contains a `STEADY_STATE` leaf.
    pub fn has_steady_state(&self, id: NodeId) -> bool {
        let mut leaves = Vec::new();
        self.collect_leaves(id, &mut leaves);
        leaves.iter().any(|l| matches!(l, Node::SteadyState(..)))
    }

    /// The parameters w.r.t. which `id` may have a non-zero derivative, sorted: its parameter
    /// leaves, or every parameter (`0..ss_param_nbr`) when it holds a `STEADY_STATE` node (`prepareForDerivation`
    /// adds all parameter deriv ids for `steadyState`, `steadyStateParamDeriv` and
    /// `steadyStateParam2ndDeriv`, `DataTree::addAllParamDerivId`).
    pub fn param_deps(&self, id: NodeId) -> Vec<u32> {
        let mut leaves = Vec::new();
        self.collect_leaves(id, &mut leaves);
        if leaves.iter().any(|l| {
            matches!(l, Node::SteadyState(..) | Node::SteadyStateParamDeriv(..) | Node::SteadyStateParam2ndDeriv(..))
        }) {
            return (0..self.ss_param_nbr).collect();
        }
        let mut ps: Vec<u32> =
            leaves.iter().filter_map(|l| if let Node::Param(q) = l { Some(*q) } else { None }).collect();
        ps.sort_unstable();
        ps
    }

    /// Position of `ss_param_deriv(x_j, p_q)` (`dys(j, q)`, column-major `n_endo x n_params`)
    /// in the extended parameter vector `[params; ys; dys(:); d2ys(:)]` of compiled tapes.
    pub fn ss_param_deriv_slot(&self, j: u32, q: u32) -> u32 {
        let (np, n) = (self.ss_offset, self.ss_n_endo);
        np + n + q * n + j
    }

    /// Position of `ss_param_2nd_deriv(x_j, p_q1, p_q2)` (`d2ys(j, q1, q2)`, column-major
    /// `n_endo x n_params x n_params`) in the extended parameter vector.
    pub fn ss_param_2nd_deriv_slot(&self, j: u32, q1: u32, q2: u32) -> u32 {
        let (np, n) = (self.ss_offset, self.ss_n_endo);
        np + n + n * np + j + n * q1 + n * np * q2
    }

    pub fn max_endo_lead(&self, id: NodeId) -> i32 {
        self.info[id as usize].max_endo_lead
    }
    pub fn max_endo_lag(&self, id: NodeId) -> i32 {
        self.info[id as usize].max_endo_lag
    }
    pub fn max_exo_lead(&self, id: NodeId) -> i32 {
        self.info[id as usize].max_exo_lead
    }
    pub fn max_exo_lag(&self, id: NodeId) -> i32 {
        self.info[id as usize].max_exo_lag
    }

    pub fn neg(&mut self, a: NodeId) -> NodeId {
        self.unary(UnaryOp::Neg, a)
    }
    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.binary(BinaryOp::Add, a, b)
    }
    pub fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.binary(BinaryOp::Sub, a, b)
    }
    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.binary(BinaryOp::Mul, a, b)
    }
    pub fn div(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.binary(BinaryOp::Div, a, b)
    }
    pub fn pow(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.binary(BinaryOp::Pow, a, b)
    }

    /// Unary node with constant folding and `-(-x) = x`.
    pub fn unary(&mut self, op: UnaryOp, a: NodeId) -> NodeId {
        if let Some(v) = self.as_num(a) {
            return self.num(op.apply(v));
        }
        if op == UnaryOp::Neg
            && let Node::Unary(UnaryOp::Neg, inner) = self.node(a)
        {
            return inner;
        }
        self.intern(Node::Unary(op, a))
    }

    /// Binary node with constant folding and Dynare's `DataTree::Add*` simplifications.
    pub fn binary(&mut self, op: BinaryOp, a: NodeId, b: NodeId) -> NodeId {
        let (na, nb) = (self.as_num(a), self.as_num(b));
        if let (Some(x), Some(y)) = (na, nb) {
            return self.num(op.apply(x, y));
        }
        let is = |v: Option<f64>, c: f64| v == Some(c);
        match op {
            BinaryOp::Add => {
                if is(na, 0.0) {
                    return b;
                }
                if is(nb, 0.0) {
                    return a;
                }
                if let Node::Unary(UnaryOp::Neg, c) = self.node(b) {
                    return self.sub(a, c);
                }
                if let Node::Unary(UnaryOp::Neg, c) = self.node(a) {
                    return self.sub(b, c);
                }
            }
            BinaryOp::Sub => {
                if is(nb, 0.0) {
                    return a;
                }
                if is(na, 0.0) {
                    return self.neg(b);
                }
                if a == b {
                    return Self::ZERO;
                }
                if let Node::Unary(UnaryOp::Neg, c) = self.node(b) {
                    return self.add(a, c);
                }
            }
            BinaryOp::Mul => {
                if is(na, 0.0) || is(nb, 0.0) {
                    return Self::ZERO;
                }
                if is(na, 1.0) {
                    return b;
                }
                if is(nb, 1.0) {
                    return a;
                }
                if is(na, -1.0) {
                    return self.neg(b);
                }
                if is(nb, -1.0) {
                    return self.neg(a);
                }
            }
            BinaryOp::Div => {
                if is(na, 0.0) {
                    return Self::ZERO;
                }
                if is(nb, 1.0) {
                    return a;
                }
                if a == b {
                    return Self::ONE;
                }
            }
            BinaryOp::Pow => {
                if is(nb, 0.0) {
                    return Self::ONE;
                }
                if is(nb, 1.0) {
                    return a;
                }
            }
            _ => {}
        }
        self.intern(Node::Binary(op, a, b))
    }

    /// `ExprNode::decreaseLeadsLags(k)`: every endogenous/exogenous lead/lag shifted by `-k`.
    pub fn shift(&mut self, id: NodeId, k: i32) -> NodeId {
        if k == 0 {
            return id;
        }
        let mut memo = HashMap::new();
        self.map_vars(id, &mut memo, &mut |d, n| match n {
            Node::Endo(j, l) => Some(d.intern(Node::Endo(j, l - k))),
            Node::Exo(j, l) => Some(d.intern(Node::Exo(j, l - k))),
            Node::SteadyState(j, l) => Some(d.intern(Node::SteadyState(j, l - k))),
            _ => None,
        })
    }

    /// Static version of an expression (`ExprNode::toStatic`): all leads/lags set to 0.
    pub fn to_static(&mut self, id: NodeId, memo: &mut HashMap<NodeId, NodeId>) -> NodeId {
        self.map_vars(id, memo, &mut |d, n| match n {
            Node::Endo(j, _) | Node::SteadyState(j, _) => Some(d.intern(Node::Endo(j, 0))),
            Node::Exo(j, _) => Some(d.intern(Node::Exo(j, 0))),
            _ => None,
        })
    }

    /// Rebuilds `id` replacing leaves for which `f` returns `Some`.
    pub fn map_vars(
        &mut self,
        id: NodeId,
        memo: &mut HashMap<NodeId, NodeId>,
        f: &mut dyn FnMut(&mut Dag, Node) -> Option<NodeId>,
    ) -> NodeId {
        if let Some(&r) = memo.get(&id) {
            return r;
        }
        let r = match self.node(id) {
            n @ (Node::Num(_)
            | Node::Endo(..)
            | Node::Exo(..)
            | Node::Param(_)
            | Node::SteadyState(..)
            | Node::SteadyStateParamDeriv(..)
            | Node::SteadyStateParam2ndDeriv(..)) => f(self, n).unwrap_or(id),
            Node::Unary(op, a) => {
                let a2 = self.map_vars(a, memo, f);
                if a2 == a { id } else { self.unary(op, a2) }
            }
            Node::Binary(op, a, b) => {
                let a2 = self.map_vars(a, memo, f);
                let b2 = self.map_vars(b, memo, f);
                if a2 == a && b2 == b { id } else { self.binary(op, a2, b2) }
            }
            Node::External(c) => {
                let args = self.ext_call(c).args.clone();
                let a2: Vec<NodeId> = args.iter().map(|&a| self.map_vars(a, memo, f)).collect();
                self.rebuild_external(id, c, a2)
            }
        };
        memo.insert(id, r);
        r
    }

    /// Collects the distinct leaves (variables and parameters) of `id`, in DFS order.
    pub fn collect_leaves(&self, id: NodeId, out: &mut Vec<Node>) {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![id];
        // explicit DFS (left first) to keep the order deterministic
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            match self.node(n) {
                Node::Num(_) => {}
                l @ (Node::Endo(..)
                | Node::Exo(..)
                | Node::Param(_)
                | Node::SteadyState(..)
                | Node::SteadyStateParamDeriv(..)
                | Node::SteadyStateParam2ndDeriv(..)) => {
                    if !out.contains(&l) {
                        out.push(l)
                    }
                }
                Node::Unary(_, a) => stack.push(a),
                Node::Binary(_, a, b) => {
                    stack.push(b);
                    stack.push(a);
                }
                Node::External(c) => stack.extend(self.ext_call(c).args.iter().rev()),
            }
        }
    }

    /// Symbolic derivative of `id` with respect to `v` (port of `computeDerivative`).
    /// `memo` must be specific to `v`.
    pub fn diff(&mut self, id: NodeId, v: DiffVar, memo: &mut HashMap<NodeId, NodeId>) -> NodeId {
        if let Some(&r) = memo.get(&id) {
            return r;
        }
        let r = match self.node(id) {
            Node::Num(_) => Self::ZERO,
            // UnaryOpNode::composeDerivatives (dynamic model): STEADY_STATE(x) is a constant for
            // the dynamic variables; w.r.t. a parameter it is `AddSteadyStateParamDeriv`, whose
            // own parameter derivative is `AddSteadyStateParam2ndDeriv`.
            Node::SteadyState(j, _) => match v {
                DiffVar::Param(q) if q < self.ss_param_nbr => self.intern(Node::SteadyStateParamDeriv(j, q)),
                _ => Self::ZERO,
            },
            Node::SteadyStateParamDeriv(j, q1) => match v {
                DiffVar::Param(q2) if q2 < self.ss_param_nbr => self.intern(Node::SteadyStateParam2ndDeriv(j, q1, q2)),
                _ => Self::ZERO,
            },
            Node::SteadyStateParam2ndDeriv(..) => match v {
                DiffVar::Param(q) if q < self.ss_param_nbr => {
                    panic!("3rd derivative of STEADY_STATE node w.r.t. three parameters not implemented")
                }
                _ => Self::ZERO,
            },
            Node::Endo(j, l) => {
                if v == DiffVar::Endo(j, l) {
                    Self::ONE
                } else {
                    Self::ZERO
                }
            }
            Node::Exo(j, l) => {
                if v == DiffVar::Exo(j, l) {
                    Self::ONE
                } else {
                    Self::ZERO
                }
            }
            Node::Param(i) => {
                if v == DiffVar::Param(i) {
                    Self::ONE
                } else {
                    Self::ZERO
                }
            }
            Node::Unary(op, a) => {
                let da = self.diff(a, v, memo);
                if da == Self::ZERO { Self::ZERO } else { self.diff_unary(op, a, id, da) }
            }
            Node::Binary(op, a, b) => {
                let da = self.diff(a, v, memo);
                let db = self.diff(b, v, memo);
                if da == Self::ZERO && db == Self::ZERO { Self::ZERO } else { self.diff_binary(op, a, b, id, da, db) }
            }
            Node::External(c) => {
                let args = self.ext_call(c).args.clone();
                let dargs: Vec<NodeId> = args.iter().map(|&a| self.diff(a, v, memo)).collect();
                self.diff_external(c, &dargs, &mut |d, x, y| d.mul(x, y), &mut |d, x, y| d.add(x, y))
            }
        };
        memo.insert(id, r);
        r
    }

    fn diff_unary(&mut self, op: UnaryOp, a: NodeId, this: NodeId, da: NodeId) -> NodeId {
        match op {
            UnaryOp::Neg => self.neg(da),
            UnaryOp::Exp => self.mul(da, this),
            UnaryOp::Log => self.div(da, a),
            UnaryOp::Log10 => {
                let ten = self.num(10.0);
                let ln10 = self.unary(UnaryOp::Log, ten);
                let t = self.mul(a, ln10);
                self.div(da, t)
            }
            UnaryOp::Sqrt => {
                let two = self.num(2.0);
                let t = self.mul(two, this);
                self.div(da, t)
            }
            UnaryOp::Abs => {
                let s = self.unary(UnaryOp::Sign, a);
                self.mul(da, s)
            }
            UnaryOp::Sign => Self::ZERO,
            UnaryOp::Sin => {
                let c = self.unary(UnaryOp::Cos, a);
                self.mul(da, c)
            }
            UnaryOp::Cos => {
                let s = self.unary(UnaryOp::Sin, a);
                let t = self.mul(da, s);
                self.neg(t)
            }
            UnaryOp::Tan => {
                let t = self.mul(this, this);
                let t = self.add(Self::ONE, t);
                self.mul(da, t)
            }
            UnaryOp::Asin | UnaryOp::Acos => {
                let t = self.mul(a, a);
                let t = self.sub(Self::ONE, t);
                let t = self.unary(UnaryOp::Sqrt, t);
                let r = self.div(da, t);
                if op == UnaryOp::Acos { self.neg(r) } else { r }
            }
            UnaryOp::Atan => {
                let t = self.mul(a, a);
                let t = self.add(Self::ONE, t);
                self.div(da, t)
            }
            UnaryOp::Sinh => {
                let c = self.unary(UnaryOp::Cosh, a);
                self.mul(da, c)
            }
            UnaryOp::Cosh => {
                let s = self.unary(UnaryOp::Sinh, a);
                self.mul(da, s)
            }
            UnaryOp::Tanh => {
                let t = self.mul(this, this);
                let t = self.sub(Self::ONE, t);
                self.mul(da, t)
            }
            // ---- stream PP (UnaryOpNode::composeDerivatives)
            UnaryOp::Asinh => {
                let c = self.unary(UnaryOp::Cosh, this);
                self.div(da, c)
            }
            UnaryOp::Acosh => {
                let s = self.unary(UnaryOp::Sinh, this);
                self.div(da, s)
            }
            UnaryOp::Atanh => {
                let t = self.mul(a, a);
                let t = self.sub(Self::ONE, t);
                self.div(da, t)
            }
            UnaryOp::Cbrt => {
                // da / (3 * a^(2/3))
                let two_thirds = self.num(2.0 / 3.0);
                let t = self.pow(a, two_thirds);
                let three = self.num(3.0);
                let t = self.mul(three, t);
                self.div(da, t)
            }
            UnaryOp::Erf | UnaryOp::Erfc => {
                // 2 / (sqrt(pi) * exp(a^2)) * da, negated for erfc
                let two = self.num(2.0);
                let a2 = self.pow(a, two);
                let e = self.unary(UnaryOp::Exp, a2);
                let sp = self.num(std::f64::consts::PI.sqrt());
                let t = self.mul(sp, e);
                let t = self.div(two, t);
                let t = self.mul(t, da);
                if op == UnaryOp::Erfc { self.neg(t) } else { t }
            }
        }
    }

    fn diff_binary(&mut self, op: BinaryOp, a: NodeId, b: NodeId, this: NodeId, da: NodeId, db: NodeId) -> NodeId {
        match op {
            BinaryOp::Add => self.add(da, db),
            BinaryOp::Sub => self.sub(da, db),
            BinaryOp::Mul => {
                let t1 = self.mul(da, b);
                let t2 = self.mul(a, db);
                self.add(t1, t2)
            }
            BinaryOp::Div => {
                if db == Self::ZERO {
                    self.div(da, b)
                } else {
                    let t11 = self.mul(da, b);
                    let t12 = self.mul(db, a);
                    let t13 = self.sub(t11, t12);
                    let t14 = self.mul(b, b);
                    self.div(t13, t14)
                }
            }
            BinaryOp::Pow => {
                if db == Self::ZERO && self.is_num_const(b) {
                    // numeric exponent: d(a^b) = da * b * a^(b-1)
                    let bm1 = self.sub(b, Self::ONE);
                    let t12 = self.pow(a, bm1);
                    let t13 = self.mul(b, t12);
                    self.mul(da, t13)
                } else if db == Self::ZERO {
                    // other exponents: da * getPowerDeriv(a, b, 1) (`AddPowerDeriv`)
                    let t = self.binary(BinaryOp::PowerDeriv(1), a, b);
                    self.mul(da, t)
                } else {
                    // d(a^b) = (db*log(a) + da*b/a) * a^b
                    let t11 = self.unary(UnaryOp::Log, a);
                    let t12 = self.mul(db, t11);
                    let t13 = self.mul(da, b);
                    let t14 = self.div(t13, a);
                    let t15 = self.add(t12, t14);
                    self.mul(t15, this)
                }
            }
            BinaryOp::PowerDeriv(k) => {
                if db == Self::ZERO {
                    let t = self.binary(BinaryOp::PowerDeriv(k + 1), a, b);
                    self.mul(da, t)
                } else {
                    // BinaryOpNode::composeDerivatives, `powerDeriv` with a varying exponent:
                    // f = a^(b-k), first = f * (db*log(a) + da*(b-k)/a) * prod_{i<k} (b-i),
                    // plus f * db * sum_i prod_{j!=i} (b-j)
                    let t11 = self.unary(UnaryOp::Log, a);
                    let t11 = self.mul(db, t11);
                    let kn = self.num(f64::from(k));
                    let t12 = self.sub(b, kn);
                    let t13 = self.mul(da, t12);
                    let t14 = self.div(t13, a);
                    let t15 = self.add(t11, t14);
                    let f = self.pow(a, t12);
                    let mut first = self.mul(f, t15);
                    for i in 0..k {
                        let ci = self.num(f64::from(i));
                        let t = self.sub(b, ci);
                        first = self.mul(first, t);
                    }
                    let mut t13 = Self::ZERO;
                    for i in 0..k {
                        let mut t11 = Self::ONE;
                        for j in 0..k {
                            if i != j {
                                let cj = self.num(f64::from(j));
                                let t12 = self.sub(b, cj);
                                t11 = self.mul(t11, t12);
                            }
                        }
                        t13 = self.add(t13, t11);
                    }
                    let t13 = self.mul(db, t13);
                    let t14 = self.mul(f, t13);
                    self.add(first, t14)
                }
            }
            BinaryOp::Max | BinaryOp::Min => {
                // Dynare: max(a,b)' = (1-(a>b))*db + (a>b)*da, min(a,b)' = (1-(b>a))*db + (b>a)*da
                // (at a kink a == b, the derivative of the second argument)
                let ca =
                    if op == BinaryOp::Max { self.binary(BinaryOp::Gt, a, b) } else { self.binary(BinaryOp::Gt, b, a) };
                let cb = self.sub(Self::ONE, ca);
                let t1 = self.mul(ca, da);
                let t2 = self.mul(cb, db);
                self.add(t2, t1)
            }
            _ => Self::ZERO,
        }
    }

    /// Interpretive evaluation (used for one-off expressions; hot paths use [`super::tape::Tape`]).
    pub fn eval(&self, id: NodeId, leaf: &dyn Fn(Node) -> f64) -> f64 {
        match self.node(id) {
            Node::Num(b) => f64::from_bits(b),
            n @ (Node::Endo(..)
            | Node::Exo(..)
            | Node::Param(_)
            | Node::SteadyState(..)
            | Node::SteadyStateParamDeriv(..)
            | Node::SteadyStateParam2ndDeriv(..)) => leaf(n),
            Node::Unary(op, a) => op.apply(self.eval(a, leaf)),
            Node::Binary(op, a, b) => op.apply(self.eval(a, leaf), self.eval(b, leaf)),
            Node::External(c) => {
                let e = self.ext_call(c);
                let v: Vec<f64> = e.args.iter().map(|&a| self.eval(a, leaf)).collect();
                super::external::get(e.fid).eval(e.d1, e.d2, &v)
            }
        }
    }

    /// Prints an expression in .mod syntax, as Dynare writes `M_.aux_vars(k).orig_expr`
    /// (`x(1)`, `x(-1)`, ...). `name` maps a leaf to its name.
    pub fn to_string(&self, id: NodeId, name: &dyn Fn(Node) -> String) -> String {
        let mut s = String::new();
        self.write(id, name, 0, &mut s);
        s
    }

    fn write(&self, id: NodeId, name: &dyn Fn(Node) -> String, parent_prec: u8, s: &mut String) {
        match self.node(id) {
            Node::Num(b) => {
                let v = f64::from_bits(b);
                if v < 0.0 && parent_prec > 0 {
                    s.push_str(&format!("({v})"));
                } else {
                    s.push_str(&format!("{v}"));
                }
            }
            n @ (Node::Endo(_, l) | Node::Exo(_, l)) => {
                s.push_str(&name(n));
                if l != 0 {
                    s.push_str(&format!("({l})"));
                }
            }
            n @ Node::Param(_) => s.push_str(&name(n)),
            Node::SteadyState(j, _) => {
                s.push_str("STEADY_STATE(");
                s.push_str(&name(Node::Endo(j, 0)));
                s.push(')');
            }
            Node::SteadyStateParamDeriv(j, q) => {
                s.push_str("ss_param_deriv(");
                s.push_str(&name(Node::Endo(j, 0)));
                s.push(',');
                s.push_str(&name(Node::Param(q)));
                s.push(')');
            }
            Node::SteadyStateParam2ndDeriv(j, q1, q2) => {
                s.push_str("ss_param_2nd_deriv(");
                s.push_str(&name(Node::Endo(j, 0)));
                for q in [q1, q2] {
                    s.push(',');
                    s.push_str(&name(Node::Param(q)));
                }
                s.push(')');
            }
            Node::Unary(UnaryOp::Neg, a) => {
                let paren = parent_prec >= 3;
                if paren {
                    s.push('(');
                }
                s.push('-');
                self.write(a, name, 4, s);
                if paren {
                    s.push(')');
                }
            }
            Node::Unary(op, a) => {
                s.push_str(op.name());
                s.push('(');
                self.write(a, name, 0, s);
                s.push(')');
            }
            Node::Binary(op @ (BinaryOp::Max | BinaryOp::Min), a, b) => {
                s.push_str(op.symbol());
                s.push('(');
                self.write(a, name, 0, s);
                s.push(','); // as the preprocessor (`max(a,b)`)
                self.write(b, name, 0, s);
                s.push(')');
            }
            Node::Binary(BinaryOp::PowerDeriv(k), a, b) => {
                s.push_str("getPowerDeriv(");
                self.write(a, name, 0, s);
                s.push(',');
                self.write(b, name, 0, s);
                s.push_str(&format!(",{k})"));
            }
            Node::Binary(op, a, b) => {
                let p = op.precedence();
                let paren = p < parent_prec;
                if paren {
                    s.push('(');
                }
                self.write(a, name, p, s);
                s.push_str(op.symbol());
                // right operand of a non-associative op binds one level tighter
                let rp = if matches!(op, BinaryOp::Sub | BinaryOp::Div | BinaryOp::Pow) { p + 1 } else { p };
                self.write(b, name, rp, s);
                if paren {
                    s.push(')');
                }
            }
            Node::External(c) => {
                // value: `f(a,b)`; derivatives as the preprocessor's MATLAB output
                // (`jacob_element('f',1,{a,b})`, `hess_element('f',1,2,{a,b})`)
                let e = self.ext_call(c);
                let f = super::external::name(e.fid);
                let (open, close) = match (e.d1, e.d2) {
                    (0, _) => (format!("{f}("), ")".to_string()),
                    (i, 0) => (format!("jacob_element('{f}',{i},{{"), "})".to_string()),
                    (i, j) => (format!("hess_element('{f}',{i},{j},{{"), "})".to_string()),
                };
                s.push_str(&open);
                for (k, &a) in e.args.iter().enumerate() {
                    if k > 0 {
                        s.push(',');
                    }
                    self.write(a, name, 0, s);
                }
                s.push_str(&close);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_consing_and_simplification() {
        let mut d = Dag::new();
        let x = d.endo(0, 0);
        let y = d.endo(1, -1);
        let a = d.add(x, y);
        let b = d.add(x, y);
        assert_eq!(a, b);
        let z = d.num(0.0);
        assert_eq!(d.add(x, z), x);
        assert_eq!(d.mul(x, z), Dag::ZERO);
        let one = d.num(1.0);
        assert_eq!(d.mul(one, y), y);
        assert_eq!(d.sub(x, x), Dag::ZERO);
        let nx = d.neg(x);
        assert_eq!(d.neg(nx), x);
        let two = d.num(2.0);
        let three = d.num(3.0);
        let s = d.add(two, three);
        assert_eq!(d.as_num(s), Some(5.0));
        assert_eq!(d.max_endo_lag(a), 1);
    }

    #[test]
    fn derivatives() {
        let mut d = Dag::new();
        let x = d.endo(0, 0);
        let p = d.param(0);
        // f = p*x^3 + exp(x)/x
        let three = d.num(3.0);
        let x3 = d.pow(x, three);
        let t1 = d.mul(p, x3);
        let ex = d.unary(UnaryOp::Exp, x);
        let t2 = d.div(ex, x);
        let f = d.add(t1, t2);
        let mut memo = HashMap::new();
        let df = d.diff(f, DiffVar::Endo(0, 0), &mut memo);
        let (xv, pv) = (1.3f64, 0.7f64);
        let leaf = |n: Node| match n {
            Node::Endo(..) => xv,
            Node::Param(_) => pv,
            _ => unreachable!(),
        };
        let expect = 3.0 * pv * xv * xv + (xv.exp() * xv - xv.exp()) / (xv * xv);
        assert!((d.eval(df, &leaf) - expect).abs() < 1e-12);
        // derivative w.r.t. an absent variable is the zero node
        let mut memo = HashMap::new();
        assert_eq!(d.diff(f, DiffVar::Endo(0, 1), &mut memo), Dag::ZERO);
        // d/dp
        let mut memo = HashMap::new();
        let dp = d.diff(f, DiffVar::Param(0), &mut memo);
        assert!((d.eval(dp, &leaf) - xv.powi(3)).abs() < 1e-12);
    }

    #[test]
    fn shift_and_print() {
        let mut d = Dag::new();
        let x = d.endo(0, 4);
        let e = d.exo(0, 0);
        let s = d.add(x, e);
        let s1 = d.shift(s, 3);
        let name = |n: Node| match n {
            Node::Endo(..) => "x".to_string(),
            Node::Exo(..) => "e".to_string(),
            _ => "p".to_string(),
        };
        assert_eq!(d.to_string(s1, &name), "x(1)+e(-3)");
        let p = d.param(0);
        let one = d.num(1.0);
        let t = d.sub(one, p);
        let t = d.mul(t, x);
        assert_eq!(d.to_string(t, &name), "(1-p)*x(4)");
    }
}
