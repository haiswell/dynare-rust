// SPDX-License-Identifier: GPL-3.0-or-later
//! Statement and expression parser (the subset of Dynare's `DynareBison.yy` used by
//! semi-structural models). Unsupported commands are kept as [`Command::Other`].

use super::lexer::{Loc, Tok, Token};
use super::macro_proc::SrcLine;
use super::{
    Command, DetShock, DetShockKind, EstimatedKind, EstimatedParam, EstimatedParamInit, ModFile, OptValue, Options,
    OriginalModel, PathValue, PolicyDecl, PriorShape, RAW_ARGS_KEY, ShockEntry, StateBlock, SymbolLabels, lexer, opt,
};
use crate::linalg::Mat;
use crate::model::expr::{BinaryOp, Dag, NodeId, UnaryOp};
use crate::model::{ModelBuilder, SymbolKind};
use anyhow::{Result, anyhow, bail};
use std::sync::Arc;

// ---- var(log), deflators, trend variables (parser_q1.rs) ----
#[path = "parser_q1.rs"]
mod q1;
pub use q1::SET_SHOCK_STDERR;
// ---- shocks/shock_paths: dates, workspace vectors and databases (parser_shocks.rs) ----
#[path = "parser_shocks.rs"]
mod shk;
// ---- stream X: misc statements (parser_x.rs) ----
#[path = "parser_matlab.rs"]
mod ml;
#[path = "parser_x.rs"]
mod x;
pub use x::{MATLAB_STATEMENT, OPTIONS_ASSIGNMENT, SET_PARAM};
// ---- epilogue, init2shocks, model_local_variable labels (parser_q4.rs) ----
#[path = "parser_q4.rs"]
mod q4;
pub(super) use q4::q4_flags;
// ---- heteroskedastic_shocks, filter_initial_state, skew, dsge_prior_weight ----
#[path = "parser_q3.rs"]
mod q3;
pub use q3::{HeteroShock, Q3Decl};

// ---- var models ----
#[path = "parser_vm.rs"]
mod vm;
// ---- pac ----
#[path = "parser_pac.rs"]
mod pac;
// ---- end pac ----

// ---- heterogeneity ----
#[path = "parser_het.rs"]
mod het;
// ---- reporting: report() objects and IRF dseries (parser_report.rs) ----
#[path = "parser_report.rs"]
mod rep;
// ---- end heterogeneity ----

/// Expression syntax tree (before symbol resolution).
#[derive(Clone, Debug, PartialEq)]
pub enum Ast {
    Num(f64),
    /// A name with its lead/lag (`x(-1)` -> lag -1).
    Var {
        name: String,
        lag: i32,
        loc: Loc,
    },
    Un(UnaryOp, Box<Ast>),
    Bin(BinaryOp, Box<Ast>, Box<Ast>),
    /// stream PP: Dynare operators that are not plain functions of their arguments.
    Op(SpecialOp, Vec<Ast>),
}

/// stream PP: `STEADY_STATE(e)`, `EXPECTATION(k)(e)`, `diff(e)`, `normcdf(x[, mu, sigma])`,
/// `normpdf(x[, mu, sigma])` (one or three arguments).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpecialOp {
    SteadyState,
    Expectation(i32),
    Diff,
    Normcdf,
    Normpdf,
    /// `external_function` call (function id in `model::external`).
    External(u32),
}

impl SpecialOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::SteadyState => "STEADY_STATE",
            Self::Expectation(_) => "EXPECTATION",
            Self::Diff => "diff",
            Self::Normcdf => "normcdf",
            Self::Normpdf => "normpdf",
            Self::External(f) => crate::model::external::name(f),
        }
    }
}

impl Ast {
    /// True if the expression contains an operator in `pred`.
    pub fn any_op(&self, pred: &dyn Fn(SpecialOp) -> bool) -> bool {
        match self {
            Ast::Num(_) | Ast::Var { .. } => false,
            Ast::Un(_, a) => a.any_op(pred),
            Ast::Bin(_, a, b) => a.any_op(pred) || b.any_op(pred),
            Ast::Op(op, args) => pred(*op) || args.iter().any(|a| a.any_op(pred)),
        }
    }
}

/// Named constants accepted where no symbol of that name exists.
fn builtin_const(name: &str) -> Option<f64> {
    match name {
        "inf" | "Inf" => Some(f64::INFINITY),
        "nan" | "NaN" => Some(f64::NAN),
        "pi" => Some(std::f64::consts::PI),
        _ => None,
    }
}

/// Converts an AST into DAG nodes; `resolve(dag, name, lag, loc)` maps names to nodes.
pub fn ast_to_node(
    ast: &Ast,
    dag: &mut Dag,
    resolve: &mut dyn FnMut(&mut Dag, &str, i32, &Loc) -> Result<NodeId>,
) -> Result<NodeId> {
    Ok(match ast {
        Ast::Num(v) => dag.num(*v),
        Ast::Var { name, lag, loc } => resolve(dag, name, *lag, loc)?,
        Ast::Un(op, a) => {
            let a = ast_to_node(a, dag, resolve)?;
            dag.unary(*op, a)
        }
        Ast::Bin(op, a, b) => {
            let a = ast_to_node(a, dag, resolve)?;
            let b = ast_to_node(b, dag, resolve)?;
            dag.binary(*op, a, b)
        }
        Ast::Op(op, args) => {
            let mut v = Vec::with_capacity(args.len());
            for a in args {
                v.push(ast_to_node(a, dag, resolve)?);
            }
            special_to_node(dag, *op, &v)?
        }
    })
}

/// stream PP: the DAG form of a special operator. `STEADY_STATE` maps the variables of its
/// argument to their steady state; `diff(e)` is `e - e(-1)` (the aux-variable form is built by
/// the transform pass, `model::dtree`); `normcdf`/`normpdf` are written with `erfc`/`exp`
/// (Dynare's `TrinaryOpNode::eval_opcode`). `EXPECTATION` cannot be expressed without the
/// transform pass: its argument is returned and the caller must not rely on the result.
pub fn special_to_node(dag: &mut Dag, op: SpecialOp, v: &[NodeId]) -> Result<NodeId> {
    let (mu, sigma) = if v.len() == 3 { (v[1], v[2]) } else { (Dag::ZERO, Dag::ONE) };
    Ok(match op {
        SpecialOp::SteadyState => dag.steady_state(v[0]),
        SpecialOp::Expectation(_) => v[0],
        SpecialOp::Diff => {
            let l = dag.shift(v[0], 1);
            dag.sub(v[0], l)
        }
        SpecialOp::Normcdf => {
            // 0.5 * erfc(-(x - mu) / (sigma * sqrt(2)))
            let d = dag.sub(v[0], mu);
            let s2 = dag.num(std::f64::consts::SQRT_2);
            let den = dag.mul(sigma, s2);
            let z = dag.div(d, den);
            let z = dag.neg(z);
            let e = dag.unary(UnaryOp::Erfc, z);
            let h = dag.num(0.5);
            dag.mul(h, e)
        }
        SpecialOp::Normpdf => {
            // exp(-((x - mu) / sigma)^2 / 2) / (sigma * sqrt(2 pi))
            let d = dag.sub(v[0], mu);
            let z = dag.div(d, sigma);
            let z2 = dag.mul(z, z);
            let h = dag.num(-0.5);
            let t = dag.mul(h, z2);
            let e = dag.unary(UnaryOp::Exp, t);
            let c = dag.num((2.0 * std::f64::consts::PI).sqrt());
            let den = dag.mul(sigma, c);
            dag.div(e, den)
        }
        SpecialOp::External(f) => dag.external(f, v.to_vec()),
    })
}

/// Evaluates an AST numerically; `lookup(name, lag, loc)` gives the value of a name.
pub fn eval_ast(ast: &Ast, lookup: &dyn Fn(&str, i32, &Loc) -> Result<f64>) -> Result<f64> {
    Ok(match ast {
        Ast::Num(v) => *v,
        Ast::Var { name, lag, loc } => lookup(name, *lag, loc)?,
        Ast::Un(op, a) => op.apply(eval_ast(a, lookup)?),
        Ast::Bin(op, a, b) => op.apply(eval_ast(a, lookup)?, eval_ast(b, lookup)?),
        Ast::Op(op, args) => {
            let v: Vec<f64> = args.iter().map(|a| eval_ast(a, lookup)).collect::<Result<_>>()?;
            let (mu, sigma) = if v.len() == 3 { (v[1], v[2]) } else { (0.0, 1.0) };
            match op {
                SpecialOp::SteadyState | SpecialOp::Expectation(_) => v[0],
                SpecialOp::Diff => bail!("diff() cannot be evaluated outside the model block"),
                SpecialOp::Normcdf => 0.5 * UnaryOp::Erfc.apply(-(v[0] - mu) / (sigma * std::f64::consts::SQRT_2)),
                SpecialOp::Normpdf => {
                    let z = (v[0] - mu) / sigma;
                    (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
                }
                SpecialOp::External(f) => crate::model::external::get(*f).eval(0, 0, &v),
            }
        }
    })
}

/// Parses a standalone expression such as `"(1-rho)*ss_x"`.
pub fn parse_expression(src: &str) -> Result<Ast> {
    let lines = [SrcLine { file: Arc::from("<expr>"), line: 1, text: src.to_string() }];
    let toks = lexer::tokenize(&lines)?;
    let mut c = Cursor::new(&toks);
    let a = c.expr()?;
    if let Some(t) = c.peek() {
        bail!("{}: unexpected '{}' after expression", t.loc, t.raw);
    }
    Ok(a)
}

// ---------------------------------------------------------------------------------------

struct Cursor<'a> {
    t: &'a [Token],
    i: usize,
    /// stream PP: source text of every `Ast::Num` built, in order (Dynare interns numerical
    /// constants by their text, see `model::dtree`).
    nums: Vec<String>,
    // ---- heterogeneity ----
    /// A `heterogeneity_dimension` was declared: `SUM(x)` is an operator.
    het_dim: bool,
    /// Heterogeneous variables found inside `SUM()` while parsing the current equation.
    sums: Vec<String>,
    /// Current nesting depth of the expression being parsed (parentheses, unary operators,
    /// `^` chains, function arguments), bounded by [`MAX_EXPR_NESTING`].
    depth: usize,
    /// inside a `model` block: `pow` / `power` are not Dynare functions there
    in_model_block: bool,
    /// End (exclusive) of the native MATLAB statement being parsed when it ends with its line
    /// instead of a ';': the next [`Cursor::statement`] stops there.
    native_end: Option<usize>,
}

/// Deepest expression nesting accepted (the later passes over the expression tree are
/// recursive; the driver runs them on a large stack).
const MAX_EXPR_NESTING: usize = 10_000;

impl<'a> Cursor<'a> {
    fn new(t: &'a [Token]) -> Self {
        Cursor {
            t,
            i: 0,
            nums: Vec::new(),
            het_dim: false,
            sums: Vec::new(),
            depth: 0,
            in_model_block: false,
            native_end: None,
        }
    }
    fn peek(&self) -> Option<&'a Token> {
        self.t.get(self.i)
    }
    fn peek_at(&self, k: usize) -> Option<&'a Token> {
        self.t.get(self.i + k)
    }
    fn loc(&self) -> String {
        match self.t.get(self.i).or(self.t.last()) {
            Some(t) => t.loc.to_string(),
            None => "<input>".to_string(),
        }
    }
    fn next(&mut self) -> Result<&'a Token> {
        let t = self.t.get(self.i).ok_or_else(|| anyhow!("{}: unexpected end of file", self.loc()))?;
        self.i += 1;
        Ok(t)
    }
    fn at(&self, p: &str) -> bool {
        self.peek().is_some_and(|t| t.is(p))
    }
    fn at_ident(&self, s: &str) -> bool {
        self.peek().is_some_and(|t| t.is_ident(s))
    }
    fn eat(&mut self, p: &str) -> bool {
        if self.at(p) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, p: &str) -> Result<()> {
        match self.peek() {
            Some(t) if t.is(p) => {
                self.i += 1;
                Ok(())
            }
            Some(t) => bail!("{}: expected '{p}', found '{}'", t.loc, t.raw),
            None => bail!("{}: expected '{p}', found end of file", self.loc()),
        }
    }
    fn ident(&mut self) -> Result<(&'a str, &'a Loc)> {
        let t = self.next()?;
        match &t.tok {
            Tok::Ident(s) => Ok((s, &t.loc)),
            _ => bail!("{}: expected a name, found '{}'", t.loc, t.raw),
        }
    }
    /// At `end ;`?
    fn at_end(&self) -> bool {
        self.at_ident("end") && self.peek_at(1).is_some_and(|t| t.is(";"))
    }
    /// Tokens up to (excluding) the next `;` at bracket depth 0; consumes the `;`.
    fn statement(&mut self) -> Result<&'a [Token]> {
        let st = self.i;
        if let Some(e) = self.native_end.take() {
            self.i = e;
            return Ok(&self.t[st..e]);
        }
        let mut depth = 0i32;
        while let Some(t) = self.peek() {
            if t.is("(") || t.is("[") || t.is("{") {
                depth += 1;
            } else if t.is(")") || t.is("]") || t.is("}") {
                depth -= 1;
            } else if t.is(";") && depth <= 0 {
                let s = &self.t[st..self.i];
                self.i += 1;
                return Ok(s);
            }
            self.i += 1;
        }
        let at = self.t.get(st).map(|t| t.loc.to_string()).unwrap_or_default();
        bail!("{at}: statement not terminated by ';'")
    }
    /// Where a native MATLAB statement starting here ends when its line has no ';' (Dynare's
    /// lexer ends a native statement with its line): the index of the first token of the next
    /// line, brackets and `...` continuations followed. `None` when a ';' ends it first.
    fn native_line_end(&self) -> Option<usize> {
        let mut depth = 0i32;
        let mut k = self.i;
        while let Some(t) = self.t.get(k) {
            if k > self.i && depth <= 0 && t.loc != self.t[k - 1].loc {
                let continued = k >= self.i + 3 && self.t[k - 3..k].iter().all(|p| p.is("."));
                if !continued {
                    return Some(k);
                }
            }
            if t.is("(") || t.is("[") || t.is("{") {
                depth += 1;
            } else if t.is(")") || t.is("]") || t.is("}") {
                depth -= 1;
            } else if t.is(";") && depth <= 0 {
                return None;
            }
            k += 1;
        }
        Some(k)
    }
    /// Skips a balanced `( ... )` group if present.
    fn skip_parens(&mut self) -> Result<()> {
        if !self.at("(") {
            return Ok(());
        }
        let mut depth = 0;
        loop {
            let t = self.next()?;
            if t.is("(") {
                depth += 1;
            } else if t.is(")") {
                depth -= 1;
                if depth == 0 {
                    return Ok(());
                }
            }
        }
    }

    // ---- expressions: precedence as DynareBison.yy (comparison < +- < */ < unary < ^)
    /// `==` and `!=` bind looser than `<`, `>`, `<=`, `>=` (DynareBison.yy precedence).
    fn expr(&mut self) -> Result<Ast> {
        let mut a = self.relational()?;
        loop {
            let op = match self.peek().map(|t| &t.tok) {
                Some(Tok::Punct("==")) => BinaryOp::Eq,
                Some(Tok::Punct("!=")) => BinaryOp::Ne,
                _ => return Ok(a),
            };
            self.i += 1;
            let b = self.relational()?;
            a = Ast::Bin(op, Box::new(a), Box::new(b));
        }
    }
    fn relational(&mut self) -> Result<Ast> {
        let mut a = self.additive()?;
        // (left-associative, so that `lb < x < ub` of a heterogeneity ⟂ condition parses)
        loop {
            let op = match self.peek().map(|t| &t.tok) {
                Some(Tok::Punct("<")) => BinaryOp::Lt,
                Some(Tok::Punct(">")) => BinaryOp::Gt,
                Some(Tok::Punct("<=")) => BinaryOp::Le,
                Some(Tok::Punct(">=")) => BinaryOp::Ge,
                _ => return Ok(a),
            };
            self.i += 1;
            let b = self.additive()?;
            a = Ast::Bin(op, Box::new(a), Box::new(b));
        }
    }
    fn additive(&mut self) -> Result<Ast> {
        let mut a = self.multiplicative()?;
        loop {
            let op = if self.eat("+") {
                BinaryOp::Add
            } else if self.eat("-") {
                BinaryOp::Sub
            } else {
                return Ok(a);
            };
            let b = self.multiplicative()?;
            a = Ast::Bin(op, Box::new(a), Box::new(b));
        }
    }
    fn multiplicative(&mut self) -> Result<Ast> {
        let mut a = self.unary()?;
        loop {
            let op = if self.eat("*") {
                BinaryOp::Mul
            } else if self.eat("/") {
                BinaryOp::Div
            } else {
                return Ok(a);
            };
            let b = self.unary()?;
            a = Ast::Bin(op, Box::new(a), Box::new(b));
        }
    }
    fn unary(&mut self) -> Result<Ast> {
        if self.depth >= MAX_EXPR_NESTING {
            bail!(
                "{}: expression nested too deeply (more than {MAX_EXPR_NESTING} levels of parentheses, unary operators or powers)",
                self.loc()
            );
        }
        self.depth += 1;
        let r = self.unary_inner();
        self.depth -= 1;
        r
    }
    fn unary_inner(&mut self) -> Result<Ast> {
        if self.eat("-") {
            return Ok(Ast::Un(UnaryOp::Neg, Box::new(self.unary()?)));
        }
        if self.eat("+") {
            return self.unary();
        }
        let a = self.primary()?;
        if self.eat("^") {
            let b = self.unary()?;
            return Ok(Ast::Bin(BinaryOp::Pow, Box::new(a), Box::new(b)));
        }
        Ok(a)
    }
    fn primary(&mut self) -> Result<Ast> {
        let t = self.next()?;
        match &t.tok {
            Tok::Num(v) => {
                self.nums.push(t.raw.clone());
                Ok(Ast::Num(*v))
            }
            Tok::Punct("(") => {
                let a = self.expr()?;
                self.expect(")")?;
                Ok(a)
            }
            Tok::Ident(name) => {
                if !self.at("(") {
                    return Ok(Ast::Var { name: name.clone(), lag: 0, loc: t.loc.clone() });
                }
                // external_function: `f(args)` calls the declared function
                if let Some((fid, nargs)) = crate::model::external::declared(name) {
                    self.i += 1;
                    let mut args = Vec::new();
                    if !self.eat(")") {
                        args.push(self.expr()?);
                        while self.eat(",") {
                            args.push(self.expr()?);
                        }
                        self.expect(")")?;
                    }
                    if args.len() != nargs {
                        bail!(
                            "{}: external function {name} called with {} argument(s), declared with nargs = {nargs}",
                            t.loc,
                            args.len()
                        );
                    }
                    return Ok(Ast::Op(SpecialOp::External(fid), args));
                }
                // Dynare's lexer is case-insensitive for function names
                let lname = name.to_ascii_lowercase();
                if let Some(op) = UnaryOp::from_name(&lname) {
                    self.i += 1;
                    let a = self.expr()?;
                    self.expect(")")?;
                    return Ok(Ast::Un(op, Box::new(a)));
                }
                // ---- heterogeneity ----
                if lname == "sum" && self.het_dim {
                    self.i += 1;
                    let (x, xloc) = self.ident()?;
                    let x = x.to_string();
                    if self.at("(") {
                        bail!("{xloc}: SUM(): the argument must be a heterogeneous variable without lead or lag");
                    }
                    self.expect(")")?;
                    self.sums.push(x.clone());
                    return Ok(Ast::Var { name: format!("SUM_{x}"), lag: 0, loc: xloc.clone() });
                }
                // ---- end heterogeneity ----
                // ---- var models ----
                if lname == "var_expectation" {
                    return vm::ve_placeholder(self, &t.loc);
                }
                // ---- pac ----
                if lname == "pac_expectation" || lname == "pac_target_nonstationary" {
                    let prefix = if lname == "pac_expectation" { pac::PE } else { pac::PNS };
                    return pac::placeholder(self, &t.loc, prefix);
                }
                // ---- stream PP: STEADY_STATE, EXPECTATION, diff, normcdf, normpdf
                let special = match lname.as_str() {
                    "steady_state" => Some(SpecialOp::SteadyState),
                    "diff" => Some(SpecialOp::Diff),
                    "normcdf" => Some(SpecialOp::Normcdf),
                    "normpdf" => Some(SpecialOp::Normpdf),
                    "expectation" => {
                        // EXPECTATION(k)(expr)
                        self.i += 1;
                        let neg = self.eat("-");
                        if !neg {
                            self.eat("+");
                        }
                        let k = self.next()?;
                        let k = match k.tok {
                            Tok::Num(v) if v.fract() == 0.0 => v as i32,
                            _ => bail!("{}: EXPECTATION(k): k must be an integer", k.loc),
                        };
                        self.expect(")")?;
                        if !self.at("(") {
                            bail!("{}: expected EXPECTATION(k)(expression)", t.loc);
                        }
                        Some(SpecialOp::Expectation(if neg { -k } else { k }))
                    }
                    _ => None,
                };
                if let Some(op) = special {
                    self.i += 1;
                    let mut args = vec![self.expr()?];
                    while self.eat(",") {
                        args.push(self.expr()?);
                    }
                    self.expect(")")?;
                    let ok = match op {
                        SpecialOp::Normcdf | SpecialOp::Normpdf => args.len() == 1 || args.len() == 3,
                        _ => args.len() == 1,
                    };
                    if !ok {
                        bail!("{}: wrong number of arguments to {}()", t.loc, op.name());
                    }
                    return Ok(Ast::Op(op, args));
                }
                let name = &lname_or(name, &lname);
                if self.in_model_block && (name == "pow" || name == "power") {
                    bail!(
                        "{}: To use an external function ({name}) within the model block, you must first declare it via the external_function() statement.",
                        t.loc
                    );
                }
                if name == "max" || name == "min" || name == "pow" || name == "power" {
                    self.i += 1;
                    let a = self.expr()?;
                    self.expect(",")?;
                    let b = self.expr()?;
                    self.expect(")")?;
                    let op = match name.as_str() {
                        "max" => BinaryOp::Max,
                        "min" => BinaryOp::Min,
                        _ => BinaryOp::Pow,
                    };
                    return Ok(Ast::Bin(op, Box::new(a), Box::new(b)));
                }
                // lead/lag: name(+k), name(-k), name(k)
                self.i += 1;
                let sign = if self.eat("-") {
                    -1
                } else {
                    self.eat("+");
                    1
                };
                let k = self.next()?;
                let lag = match k.tok {
                    Tok::Num(v) if v.fract() == 0.0 && v.abs() < 1e6 => sign * v as i32,
                    _ => bail!(
                        "{}: '{name}(' is neither a known function nor a lead/lag; calls to MATLAB functions (e.g. a helper .m file using fsolve/fzero) cannot be run by dynare-rust: compute the value in the .mod file instead",
                        t.loc
                    ),
                };
                self.expect(")")?;
                Ok(Ast::Var { name: name.clone(), lag, loc: t.loc.clone() })
            }
            _ => bail!("{}: unexpected '{}' in expression", t.loc, t.raw),
        }
    }
}

/// stream PP: `tag_pair_list_for_selection` of `model_remove(...)` / `model_replace(...)`:
/// `'name1', tag='v', [tag1='v1', tag2='v2'], ...` (a bare string selects `name='...'`).
fn tag_selection(toks: &[Token]) -> Result<Vec<Vec<(String, String)>>> {
    let inner = match toks {
        [o, inner @ .., c] if o.is("(") && c.is(")") => inner,
        _ => bail!("{}: expected (tag list)", toks.first().map(|t| t.loc.to_string()).unwrap_or_default()),
    };
    let mut out = Vec::new();
    for item in split_commas(inner) {
        match item {
            [t] if matches!(t.tok, Tok::Str(_)) => {
                let Tok::Str(s) = &t.tok else { unreachable!() };
                out.push(vec![("name".to_string(), s.clone())]);
            }
            // `tag = 'value'`
            [k, eq, v] if eq.is("=") && k.ident().is_some() && matches!(v.tok, Tok::Str(_)) => {
                let Tok::Str(s) = &v.tok else { unreachable!() };
                out.push(vec![(k.raw.to_ascii_lowercase(), s.clone())]);
            }
            [o, pairs @ .., c] if o.is("[") && c.is("]") => {
                let mut sel = Vec::new();
                for p in split_commas(pairs) {
                    match p {
                        [k, eq, v] if eq.is("=") => {
                            let v = match &v.tok {
                                Tok::Str(s) => s.clone(),
                                _ => v.raw.clone(),
                            };
                            sel.push((k.raw.to_ascii_lowercase(), v));
                        }
                        [k] => sel.push((k.raw.to_ascii_lowercase(), String::new())),
                        _ => bail!(
                            "{}: malformed tag in selection",
                            p.first().map(|t| t.loc.to_string()).unwrap_or_default()
                        ),
                    }
                }
                out.push(sel);
            }
            [] => {}
            other => bail!("{}: malformed equation selection", other[0].loc),
        }
    }
    Ok(out)
}

/// Lower-cased name for the built-in binary functions (`MAX(a,b)`), the name itself otherwise.
fn lname_or(name: &str, lname: &str) -> String {
    if matches!(lname, "max" | "min") { lname.to_string() } else { name.to_string() }
}

/// Parses a token slice as one complete expression.
fn slice_expr(toks: &[Token]) -> Result<Ast> {
    let mut c = Cursor::new(toks);
    let a = c.expr()?;
    if let Some(t) = c.peek() {
        bail!("{}: unexpected '{}' in expression", t.loc, t.raw);
    }
    Ok(a)
}

/// The expression with 0 instead of the operators of the VAR / PAC models (placeholder
/// variables), which only the transform pass replaces.
fn strip_placeholders(a: &Ast) -> Ast {
    match a {
        Ast::Var { name, .. } if name.starts_with('\u{1}') => Ast::Num(0.0),
        Ast::Un(o, x) => Ast::Un(*o, Box::new(strip_placeholders(x))),
        Ast::Bin(o, x, y) => Ast::Bin(*o, Box::new(strip_placeholders(x)), Box::new(strip_placeholders(y))),
        Ast::Op(o, v) => Ast::Op(*o, v.iter().map(strip_placeholders).collect()),
        _ => a.clone(),
    }
}

/// [`slice_expr`] with the text of its numerical constants (stream PP, see `Cursor::nums`).
fn slice_expr_nums(toks: &[Token]) -> Result<(Ast, Vec<String>)> {
    let mut c = Cursor::new(toks);
    let a = c.expr()?;
    if let Some(t) = c.peek() {
        bail!("{}: unexpected '{}' in expression", t.loc, t.raw);
    }
    Ok((a, c.nums))
}

/// Splits a token slice at top-level commas (empty fields are kept).
fn split_commas(toks: &[Token]) -> Vec<&[Token]> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut st = 0;
    for (k, t) in toks.iter().enumerate() {
        if t.is("(") || t.is("[") || t.is("{") {
            depth += 1;
        } else if t.is(")") || t.is("]") || t.is("}") {
            depth -= 1;
        } else if t.is(",") && depth == 0 {
            out.push(&toks[st..k]);
            st = k + 1;
        }
    }
    out.push(&toks[st..]);
    out
}

fn raw_join(toks: &[Token]) -> String {
    toks.iter().map(|t| t.raw.as_str()).collect()
}

/// Commands whose body runs until `end;` (kept as `Command::Other`).
const OTHER_BLOCKS: &[&str] = &[
    "observation_trends",
    "optim_weights",
    "homotopy_setup",
    "conditional_forecast_paths",
    "irf_calibration",
    "moment_calibration",
    "estimated_params_bounds",
    "osr_params_bounds",
    "ramsey_constraints",
    "filter_initial_state",
    "deterministic_trends",
    "occbin_constraints",
    "matched_moments",
    "generate_irfs",
    "perfect_foresight_controlled_paths",
    "pac_target_info",
    "epilogue",
    "heteroskedastic_shocks",
    "learnt_in",
    "endval_learnt_in",
    "svar_identification",
];

/// Commands whose trailing arguments are kept as raw tokens under [`RAW_ARGS_KEY`] (stream B).
const RAW_ARG_COMMANDS: &[&str] = &[
    "bvar_density",
    "bvar_forecast",
    "bvar_irf",
    "model_comparison",
    "prior",
    "dsample",
    "save_params_and_steady_state",
]; // + [stream X]

/// Commands that make Dynare treat the model as stochastic (`ModFile::transformPass`).
const STOCHASTIC_COMMANDS: &[&str] = &[
    "stoch_simul",
    "estimation",
    "osr",
    "ramsey_model",
    "ramsey_policy",
    "discretionary_policy",
    "calib_smoother",
    "identification",
    "dynare_sensitivity",
    "method_of_moments",
];

/// A `shocks` entry target: an exogenous shock, or an observed endogenous variable
/// (measurement error, `M_.H`).
#[derive(Clone, Copy, PartialEq)]
enum Tgt {
    Exo(usize),
    Endo(usize),
}

#[derive(Clone)]
enum ShockStmt {
    Var(Tgt, PathValue),
    Std(Tgt, PathValue),
    Cov(Tgt, Tgt, PathValue),
    Corr(Tgt, Tgt, PathValue),
}

/// `file: line L, cols a-b` (Dynare's `Location`, printed by `WarningConsolidation`) of the
/// tokens `a` ..= `b`.
pub(super) fn span_text(a: &Token, b: &Token) -> String {
    let file = std::path::Path::new(&*a.loc.file)
        .file_name()
        .map_or_else(|| a.loc.file.to_string(), |f| f.to_string_lossy().into_owned());
    let (c1, c2) = (a.col.0, b.col.1);
    if a.loc.line == b.loc.line {
        if c1 == c2 {
            format!("{file}: line {}, col {c1}", a.loc.line)
        } else {
            format!("{file}: line {}, cols {c1}-{c2}", a.loc.line)
        }
    } else {
        format!("{file}: line {}, col {c1} - line {}, col {c2}", a.loc.line, b.loc.line)
    }
}

/// The warnings of Dynare's preprocessor (`ModFile::checkPass` and the parser actions), in its
/// order: those of the parsing pass, then the `osr_params` repeat, the `bytecode` option, the
/// unsuitable operators of a stochastic model and the unused parameters.
fn pp_warnings(pp: &PpNotes, toks: &[Token], params: &[String]) -> Vec<String> {
    let mut w = pp.warn.clone();
    if pp.osr_params_stmts > 1 {
        w.push("WARNING: You have more than one osr_params statement in the .mod file.".into());
    }
    if pp.bytecode {
        w.push(
            "WARNING: the 'bytecode' option is deprecated and will be removed in a future release of Dynare.".into(),
        );
    }
    let model: Vec<&[Token]> = pp
        .model_rng
        .iter()
        .map(|&(a, b)| (a, b))
        .chain(pp.eqs.iter().map(|&(_, a, b)| (a, b)))
        .filter_map(|(a, b)| toks.get(a..b.min(toks.len())))
        .collect();
    if pp.stoch {
        let bad = model.iter().any(|r| {
            r.iter().enumerate().any(|(k, t)| match &t.tok {
                Tok::Ident(n) => {
                    matches!(n.as_str(), "max" | "min" | "abs" | "sign") && r.get(k + 1).is_some_and(|x| x.is("("))
                }
                Tok::Punct(p) => matches!(*p, "<" | ">" | "<=" | ">=" | "==" | "!="),
                _ => false,
            })
        });
        if bad {
            w.push("WARNING: you are using a function (max, min, abs, sign) or an operator (<, >, <=, >=, ==, !=) which is unsuitable for a stochastic context; see the reference manual, section about \"Expressions\", for more details.".into());
        }
    }
    // used: the identifiers of the model (see [`model_idents`]) and every identifier of the
    // steady_state_model blocks
    let mut used = model_idents(pp, toks);
    for &(a, b) in &pp.ssm_rng {
        used.extend(toks.get(a..b.min(toks.len())).unwrap_or(&[]).iter().filter_map(|t| t.ident()));
    }
    // only the parameters of `parameters` statements (not those the preprocessor adds: PAC,
    // var_expectation_model, occbin, ...)
    let mut declared: Vec<&str> = Vec::new();
    for (k, t) in toks.iter().enumerate() {
        if t.is_ident("parameters") && (k == 0 || toks[k - 1].is(";") || toks[k - 1].is_ident("end")) {
            let mut depth = 0;
            for t in &toks[k + 1..] {
                if t.is(";") {
                    break;
                }
                if t.is("(") {
                    depth += 1;
                } else if t.is(")") {
                    depth -= 1;
                } else if let (0, Some(n)) = (depth, t.ident()) {
                    declared.push(n);
                }
            }
        }
    }
    let mut unused: Vec<&str> = declared.into_iter().filter(|p| !used.contains(p)).collect();
    if !unused.is_empty() && params.iter().any(|p| p == "optimal_policy_discount_factor") {
        unused.push("optimal_policy_discount_factor");
    }
    if !unused.is_empty() {
        w.push(format!("WARNING: Parameter(s) {}  not used in the model", unused.join(" ")));
    }
    w
}

/// The identifiers of the model blocks' equations (`model`, `ramsey_constraints`, heterogeneous
/// `model`) and those of the model-local variables they use (transitively).
fn model_idents<'t>(pp: &PpNotes, toks: &'t [Token]) -> std::collections::HashSet<&'t str> {
    use std::collections::{HashMap, HashSet};
    let model: Vec<&[Token]> = pp
        .model_rng
        .iter()
        .map(|&(a, b)| (a, b))
        .chain(pp.eqs.iter().map(|&(_, a, b)| (a, b)))
        .filter_map(|(a, b)| toks.get(a..b.min(toks.len())))
        .collect();
    let mut used: HashSet<&str> = HashSet::new();
    let mut locals: HashMap<&str, Vec<&str>> = HashMap::new();
    for r in &model {
        for st in r.split(|t| t.is(";")) {
            let mut st = st;
            if st.first().is_some_and(|t| t.is("[")) {
                let close = st.iter().position(|t| t.is("]")).map_or(st.len(), |k| k + 1);
                st = &st[close..];
            }
            if st.first().is_some_and(|t| t.is("#")) {
                if let Some(name) = st.get(1).and_then(|t| t.ident()) {
                    locals.insert(name, idents(st.get(3..).unwrap_or(&[])));
                }
            } else {
                used.extend(idents(st));
            }
        }
    }
    loop {
        let before = used.len();
        let add: Vec<&str> =
            locals.iter().filter(|(k, _)| used.contains(*k)).flat_map(|(_, v)| v.iter().copied()).collect();
        used.extend(add);
        if used.len() == before {
            break;
        }
    }
    used
}

/// `ModFile::checkPass` (Dynare issue #841): an exogenous variable of a `varexo` statement that
/// no model equation uses is an error, a warning with the `nostrict` option (which removes it in
/// Dynare; dynare-rs keeps it).
fn unused_exo_check(pp: &PpNotes, toks: &[Token], exo: &[String], nostrict: bool) -> Result<Option<String>> {
    let mut used = model_idents(pp, toks);
    // `DynamicModel::findUnusedExogenous` skips the observed exogenous variables (`varexobs`), and
    // the check runs after `pac_expectation` is substituted, which brings in the `growth` of the
    // `pac_model` statements
    for (k, t) in toks.iter().enumerate() {
        let start = k == 0 || toks[k - 1].is(";") || toks[k - 1].is_ident("end");
        if !start {
            continue;
        }
        let stmt = toks[k + 1..].split(|t| t.is(";")).next().unwrap_or(&[]);
        if t.is_ident("varexobs") {
            used.extend(stmt.iter().filter_map(|t| t.ident()));
        } else if t.is_ident("pac_model") {
            let mut depth = 0;
            let mut growth = false;
            for (j, t) in stmt.iter().enumerate() {
                if t.is("(") {
                    depth += 1;
                } else if t.is(")") {
                    depth -= 1;
                } else if depth == 1 && t.is(",") {
                    growth = false;
                } else if depth == 1 && t.is_ident("growth") && stmt.get(j + 1).is_some_and(|x| x.is("=")) {
                    growth = true;
                } else if growth {
                    used.extend(t.ident());
                }
            }
        }
    }
    let mut unused: Vec<&str> = Vec::new();
    for (k, t) in toks.iter().enumerate() {
        let start = k == 0 || toks[k - 1].is(";") || toks[k - 1].is_ident("end");
        // `varexo(heterogeneity_dimension = ...)` declares heterogeneous exogenous variables
        if !(start && t.is_ident("varexo")) || toks.get(k + 1).is_some_and(|x| x.is("(")) {
            continue;
        }
        let mut depth = 0;
        for t in &toks[k + 1..] {
            if t.is(";") {
                break;
            }
            if t.is("(") {
                depth += 1;
            } else if t.is(")") {
                depth -= 1;
            } else if let (0, Some(n)) = (depth, t.ident()) {
                // `var_remove` drops a symbol
                if !used.contains(n) && !unused.contains(&n) && exo.iter().any(|e| e == n) {
                    unused.push(n);
                }
            }
        }
    }
    if unused.is_empty() {
        return Ok(None);
    }
    let names: String = unused.iter().map(|n| format!("{n} ")).collect();
    if nostrict {
        return Ok(Some(format!("WARNING: {names}not used in model block, removed by nostrict command-line option")));
    }
    bail!(
        "{names}not used in model block. To bypass this error, use the `nostrict` option. This may lead to crashes or unexpected behavior."
    )
}

/// What the preprocessor's warnings are computed from (see [`pp_warnings`]).
#[derive(Default)]
struct PpNotes {
    /// Warnings of the parsing pass (deprecated statements and options), in file order.
    warn: Vec<String>,
    /// Token ranges of the `ramsey_constraints` and heterogeneous `model` blocks.
    model_rng: Vec<(usize, usize)>,
    /// Token range of each equation of the `model` blocks (tags included, `;` last) with its
    /// tags, and of each model-local variable definition (`# x = ...;`); equations removed by
    /// `model_remove` / `model_replace` leave the list.
    eqs: Vec<(Vec<(String, String)>, usize, usize)>,
    /// Token ranges of the `steady_state_model` blocks.
    ssm_rng: Vec<(usize, usize)>,
    bytecode: bool,
    osr_params_stmts: usize,
    /// A command that triggers the stochastic-context warning was seen.
    stoch: bool,
    /// Parameters of the stochastic `shocks` values (indices).
    shock_params: Vec<usize>,
}

struct Parser<'a> {
    pp: PpNotes,
    c: Cursor<'a>,
    b: ModelBuilder,
    initval: Vec<(usize, f64)>,
    endval: Vec<(usize, f64)>,
    varobs: Vec<String>,
    shocks: Vec<ShockStmt>,
    est: Vec<EstimatedParam>,
    est_history: Vec<(usize, Vec<EstimatedParam>)>,
    est_init: Vec<EstimatedParamInit>,
    use_calibration: bool,
    commands: Vec<Command>,
    stochastic: bool,
    model_seen: bool,
    occbin_constraints: Vec<crate::occbin::OccbinConstraint>,
    // stream D (deterministic simulations)
    initval_exo: Vec<(usize, f64)>,
    /// A plain `endval` block was parsed (InitValStatement::checkPass rejects a later `initval`).
    endval_seen: bool,
    det_shocks: Vec<DetShock>,
    histval: Vec<(bool, usize, i32, f64)>,
    steady_state_model: Vec<(String, String)>,
    predetermined: Vec<usize>,
    exo_det: Vec<usize>,
    labels: SymbolLabels,
    original: OriginalModel,
    policy: PolicyDecl,
    /// [stream X]
    x: x::XParse,
    /// `if isoctave`, `load`, `fieldnames`, `for ... set_param_value` (see `parser_matlab`)
    ml: ml::MatlabState,
    // ---- stream EP: shocks(learnt_in=...), endval(learnt_in=...)
    ep: crate::ep::EpDecl,
    /// Number of commands before each `ep.controlled_paths` entry ([`ModFile::controlled_paths_pos`]).
    controlled_paths_pos: Vec<usize>,
    /// stream PP: mod-file local variables (`phi = 0.1;` without a declaration), usable in
    /// later expressions outside the model block (Dynare's `modFileLocalVariable`).
    mf_locals: std::collections::HashMap<String, f64>,
    /// `verbatim` blocks: the MATLAB workspace they build (see `super::verbatim`).
    verbatim_ws: super::verbatim::Workspace,

    q4: crate::q4::Q4Decl,
    q4_flags: q4::Q4Flags,

    q1: q1::Q1Parse,

    q3: q3::Q3Decl,
    /// model_info: block-decomposition options of `model(...)`
    block_opts: crate::model::blocks::ModelOptions,
    // ---- var models ----
    vm: vm::VmParse,
    // ---- pac ----
    pac: pac::PacParse,
    // ---- heterogeneity ----
    het: het::HetParse,
    /// `shocks`/`mshocks`/`initval`/`histval` blocks at their position (see
    /// [`ModFile::state_blocks`]), with the shock targets resolved at the end of the parse.
    state_blocks: Vec<(usize, RawStateBlock)>,
}

/// A [`StateBlock`] before `varobs` is complete (measurement errors are indexed by their
/// position in `varobs`).
enum RawStateBlock {
    Shocks { overwrite: bool, stmts: Vec<ShockStmt>, reset_det: bool, det: Vec<DetShock> },
    Initval { endo: Vec<(usize, f64)>, exo: Vec<(usize, f64)>, sources: Vec<(String, String)> },
    Histval(Vec<(bool, usize, i32, f64)>),
}

thread_local! {
    static ERROR_WARNINGS: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn set_error_warnings(w: Vec<String>) {
    ERROR_WARNINGS.with(|c| *c.borrow_mut() = w);
}

/// The preprocessor warnings collected before the last parse failed (taken, not copied).
pub(super) fn take_error_warnings() -> Vec<String> {
    ERROR_WARNINGS.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

pub(super) fn parse(toks: Vec<Token>) -> Result<ModFile> {
    parse_with(toks, q4::Q4Flags::default())
}

/// [`parse`] with the command-line flags that change the preprocessing
/// (`stochastic`, `nostrict`, `exclude_eqs`, `include_eqs`).
fn idents(s: &[Token]) -> Vec<&str> {
    s.iter().filter_map(|t| t.ident()).collect()
}

pub(super) fn parse_with(toks: Vec<Token>, q4_flags: q4::Q4Flags) -> Result<ModFile> {
    crate::model::external::clear_declared();
    set_error_warnings(Vec::new());
    let mut p = Parser {
        pp: PpNotes::default(),
        c: Cursor::new(&toks),
        b: ModelBuilder::new(),
        initval: Vec::new(),
        endval: Vec::new(),
        varobs: Vec::new(),
        shocks: Vec::new(),
        est: Vec::new(),
        est_history: Vec::new(),
        est_init: Vec::new(),
        use_calibration: false,
        commands: Vec::new(),
        stochastic: false,
        model_seen: false,
        occbin_constraints: Vec::new(),
        initval_exo: Vec::new(),
        endval_seen: false,
        det_shocks: Vec::new(),
        histval: Vec::new(),
        steady_state_model: Vec::new(),
        predetermined: Vec::new(),
        exo_det: Vec::new(),
        labels: SymbolLabels::default(),
        original: OriginalModel::default(),
        policy: PolicyDecl::default(),
        x: x::XParse::default(),
        ml: ml::MatlabState::default(),
        ep: Default::default(),
        controlled_paths_pos: Vec::new(),
        mf_locals: Default::default(),
        verbatim_ws: Default::default(),
        q4: Default::default(),
        q4_flags,
        q1: Default::default(),
        q3: Default::default(),
        block_opts: Default::default(), // model_info
        vm: Default::default(),         // ---- var models ----
        pac: Default::default(),        // ---- pac ----
        het: Default::default(),        // ---- heterogeneity ----
        state_blocks: Vec::new(),
    };
    while let Some(t) = p.c.peek() {
        if let Err(e) = p.statement(t) {
            set_error_warnings(p.pp.warn.clone()); // the preprocessor printed these before the error
            return Err(e);
        }
    }
    // these checkPass errors come before its warnings (unused parameters, ...)
    set_error_warnings(p.pp.warn.clone());
    // (a model without equations fails later with its own error; include_eqs / exclude_eqs
    // drop the equations of the excluded exogenous variables without the check)
    let flags = &p.q4_flags;
    let unused_exo_warning = if p.b.n_equations() == 0 || flags.selects_equations() {
        None
    } else {
        unused_exo_check(&p.pp, &toks, &p.b.exo_names, flags.nostrict)?
    };
    // ModFile::checkPass (#634): an estimated parameter may not enter the shocks' covariance
    let mut in_shocks: Vec<usize> = p
        .est
        .iter()
        .filter_map(|e| match e.kind {
            EstimatedKind::Param(i) if p.pp.shock_params.contains(&i) => Some(i),
            _ => None,
        })
        .collect();
    if !in_shocks.is_empty() {
        in_shocks.sort_unstable();
        in_shocks.dedup();
        let names: Vec<&str> = in_shocks.iter().map(|&i| p.b.param_names[i].as_str()).collect();
        bail!(
            "some estimated parameters ({}) also appear in the expressions defining the variance/covariance matrix of shocks; this is not allowed.",
            names.join(", ")
        );
    }
    // checkPass warnings (unused parameters, ...) precede the errors of the later passes
    set_error_warnings(pp_warnings(&p.pp, &toks, &p.b.param_names));
    p.het_finish_equations()?; // ---- heterogeneity ----
    p.q1_exo_det_view()?;
    let mut varmodels = p.vm_finish()?; // ---- var models ----
    varmodels.pacs = p.pac_finish()?; // ---- pac ----
    let n_exo = p.b.exo_names.len();
    let nvobs = p.varobs.len();
    // q1_exo_det_view moved the varexo_det entries out of the exogenous ones
    let n_exo_final = n_exo;
    // M_.H is indexed by the position in varobs.
    let obs_pos = |k: usize| -> Result<usize> {
        let name = &p.b.endo_names[k];
        p.varobs.iter().position(|v| v == name).ok_or_else(|| {
            anyhow!("measurement error on {name}: the variable has to be declared as observable (varobs)")
        })
    };
    let mut sigma_e = Mat::zeros(n_exo, n_exo);
    let mut h = Mat::zeros(nvobs, nvobs);
    let mut corr_e = Mat::identity(n_exo, n_exo);
    let mut corr_h = Mat::identity(nvobs, nvobs);
    for &(a, b, r) in &p.q1.shock_corrs {
        match (a, b) {
            (Tgt::Exo(i), Tgt::Exo(j)) => {
                corr_e[(i, j)] = r;
                corr_e[(j, i)] = r;
            }
            (Tgt::Endo(i), Tgt::Endo(j)) => {
                let (i, j) = (obs_pos(i)?, obs_pos(j)?);
                corr_h[(i, j)] = r;
                corr_h[(j, i)] = r;
            }
            _ => {}
        }
    }
    let mut corrs = Vec::new();
    let mut corrs_me = Vec::new();
    for s in &p.shocks {
        let (a, b) = match *s {
            ShockStmt::Var(t, _) | ShockStmt::Std(t, _) => (t, t),
            ShockStmt::Cov(a, b, _) | ShockStmt::Corr(a, b, _) => (a, b),
        };
        let (mat, i, j) = match (a, b) {
            (Tgt::Exo(i), Tgt::Exo(j)) => (&mut sigma_e, i, j),
            (Tgt::Endo(i), Tgt::Endo(j)) => (&mut h, obs_pos(i)?, obs_pos(j)?),
            _ => bail!("shocks: cannot mix an exogenous shock and a measurement error in one covariance entry"),
        };
        match s {
            ShockStmt::Var(_, v) => mat[(i, i)] = v.num(),
            ShockStmt::Std(_, v) => mat[(i, i)] = v.num() * v.num(),
            ShockStmt::Cov(_, _, v) => {
                mat[(i, j)] = v.num();
                mat[(j, i)] = v.num();
            }
            ShockStmt::Corr(_, _, v) => {
                let v = v.num();
                if matches!(a, Tgt::Exo(_)) { corrs.push((i, j, v)) } else { corrs_me.push((i, j, v)) }
            }
        }
    }
    for (list, mat) in [(corrs, &mut sigma_e), (corrs_me, &mut h)] {
        for (a, b, v) in list {
            let c = v * (mat[(a, a)] * mat[(b, b)]).sqrt();
            mat[(a, b)] = c;
            mat[(b, a)] = c;
        }
    }
    for kind in p.est.iter().map(|e| e.kind).chain(p.est_init.iter().map(|e| e.kind)) {
        match kind {
            EstimatedKind::MeStdErr(k) => {
                obs_pos(k)?;
            }
            EstimatedKind::MeCorr(a, b) => {
                obs_pos(a)?;
                obs_pos(b)?;
            }
            _ => {}
        }
    }
    // the blocks at their position, with M_.H indexed by the position in varobs
    let mut state_blocks = Vec::with_capacity(p.state_blocks.len());
    for (k, b) in std::mem::take(&mut p.state_blocks) {
        state_blocks.push((
            k,
            match b {
                RawStateBlock::Shocks { overwrite, stmts, reset_det, det } => {
                    let mut entries = Vec::new();
                    let mut cross = Vec::new(); // covariances and correlations come after the variances
                    for s in stmts {
                        let (a, b) = match s {
                            ShockStmt::Var(t, _) | ShockStmt::Std(t, _) => (t, t),
                            ShockStmt::Cov(a, b, _) | ShockStmt::Corr(a, b, _) => (a, b),
                        };
                        let (me, i, j) = match (a, b) {
                            (Tgt::Exo(i), Tgt::Exo(j)) => (false, i, j),
                            (Tgt::Endo(i), Tgt::Endo(j)) => (true, obs_pos(i)?, obs_pos(j)?),
                            _ => bail!(
                                "shocks: cannot mix an exogenous shock and a measurement error in one covariance entry"
                            ),
                        };
                        match s {
                            ShockStmt::Var(_, v) => entries.push(ShockEntry::Var { me, i, v }),
                            ShockStmt::Std(_, v) => entries.push(ShockEntry::Var {
                                me,
                                i,
                                v: match v {
                                    PathValue::Num(x) => PathValue::Num(x * x),
                                    PathValue::Steady(e) => {
                                        PathValue::Steady(Ast::Bin(BinaryOp::Pow, Box::new(e), Box::new(Ast::Num(2.0))))
                                    }
                                },
                            }),
                            ShockStmt::Cov(_, _, v) => cross.push(ShockEntry::Cov { me, i, j, v }),
                            ShockStmt::Corr(_, _, r) => cross.push(ShockEntry::Corr { me, i, j, r }),
                        }
                    }
                    entries.extend(cross);
                    let det = det.into_iter().filter(|d| d.exo < n_exo_final).collect();
                    StateBlock::Shocks { overwrite, entries, reset_det, det }
                }
                RawStateBlock::Initval { endo, exo, sources } => StateBlock::Initval {
                    endo,
                    exo: exo.into_iter().filter(|&(k, _)| k < n_exo_final).collect(),
                    sources,
                },
                RawStateBlock::Histval(h) => {
                    StateBlock::Histval(h.into_iter().filter(|&(endo, k, _, _)| endo || k < n_exo_final).collect())
                }
            },
        ));
    }
    p.b.predetermined = p.predetermined.clone(); // stream PP (model::dtree)
    p.q1_bg_context();
    p.q4_check_exclusions()?; // exclude_eqs / include_eqs tags all found
    p.block_opts.final_params = p.b.params.clone(); // model_info: every assignment of the file
    p.b.bvar_present = p.commands.iter().any(|c| matches!(c, Command::Other(n, _) if crate::model::is_bvar_command(n)));
    let mut model = p.b.build(p.stochastic || p.q4_flags.stochastic)?;
    vm::resolve_names(&mut varmodels, &mut model); // the auxiliary variables of the VAR-model variables
    let het = p.het.het_build(&mut model)?; // ---- heterogeneity ----
    // [stream X] parameter assignments after a command run at their place (SET_PARAM).
    if let Some(snap) = &p.x.params_snapshot {
        model.params[..snap.len()].copy_from_slice(snap);
    }
    let mut warnings = pp_warnings(&p.pp, &toks, &model.param_names);
    warnings.extend(unused_exo_warning);
    let bytecode = p.pp.bytecode;
    let mut mf = ModFile {
        model,
        estimated_params_history: p.est_history,
        initval: p.initval,
        endval: p.endval,
        varobs: p.varobs,
        sigma_e,
        h,
        corr_e,
        corr_h,
        estimated_params: p.est,
        estimated_params_init: p.est_init,
        estimated_params_init_use_calibration: p.use_calibration,
        commands: p.commands,
        occbin_constraints: p.occbin_constraints,
        initval_exo: p.initval_exo,
        det_shocks: p.det_shocks,
        histval: p.histval,
        steady_state_model: p.steady_state_model,
        predetermined_variables: p.predetermined,
        exo_det: p.exo_det,
        labels: p.labels,
        original_model: p.original,
        policy: p.policy,
        observation_trends: p.x.observation_trends,
        homotopy_setup: p.x.homotopy_setup,
        homotopy_from_initval_to_endval: p.x.homotopy_from_initval_to_endval,
        homotopy_block: p.x.homotopy_block,
        ep: p.ep,
        controlled_paths_pos: p.controlled_paths_pos,
        q4: p.q4,
        q1: crate::deterministic::shock_paths::Q1Decl {
            shock_paths: p.q1.shock_paths,
            exo_det_shocks: p.q1.exo_det_shocks,
            exo_det_steady_state: p.q1.exo_det_steady_state,
            databases: p.q1.databases,
            workspace: p.q1.workspace,
        },
        q3: p.q3,
        block_opts: p.block_opts, // model_info
        varmodels,                // ---- var models ----
        het,                      // ---- heterogeneity ----
        verbatim_workspace: p.verbatim_ws.vars.iter().map(|(k, v)| (k.clone(), v.to_mat())).collect(),
        state_blocks,
        warnings,
        options: Default::default(),
    };
    // ramsey_constraints: the preprocessor writes them as complementarity conditions of the
    // Ramsey FOCs (`dynamic_complementarity_conditions`, the FOC w.r.t. the j-th variable being
    // equation j: identity `M_.dynamic_mcp_equations_reordering`), read by the perfect foresight
    // solver with solve_algo = 10 (dyn_ramsey_static.m ignores them)
    if !mf.policy.ramsey_constraints.is_empty() && mf.model.planner().is_some_and(|p| p.ramsey.is_some()) {
        let cs = crate::policy::ramsey::parse_constraints(&mf.model, &mf.policy.ramsey_constraints)?;
        for (c, l) in cs.iter().zip(&mf.policy.ramsey_constraints) {
            mf.model.mcp.push((c.var, l.trim().trim_end_matches(';').trim().to_string()));
        }
    }
    mf.model.bytecode = bytecode;
    if mf.block_opts.block {
        // model(block): the block-wise solvers use the decompositions
        let statik = crate::misc::model_info::static_block_structure(&mf);
        let dynamic = crate::misc::model_info::dynamic_block_structure(&mf);
        mf.model.block_solve = Some(std::sync::Arc::new(crate::model::blocks::BlockSolve { statik, dynamic }));
    }
    Ok(mf)
}

/// MATLAB graphics/display functions called as native statements in the official examples
/// (`figure(h)`, `subplot(3,1,1)`, `title('Prior')`): like any native MATLAB statement for
/// Dynare's lexer, they end at the end of their line even without ';'.
const NATIVE_LINE_FUNCTIONS: &[&str] = &[
    "figure", "subplot", "plot", "title", "xlabel", "ylabel", "legend", "hold", "grid", "axis", "drawnow", "close",
    "clf", "disp", "fprintf", "box", "clc", "format",
];

/// MATLAB keywords and functions called with the command syntax (`save f.mat a b`) starting a
/// native statement: one line each (the body of an `if`, `for`, ... is parsed line by line, as
/// by Dynare's lexer).
const MATLAB_KEYWORDS: &[&str] = &[
    "if",
    "elseif",
    "else",
    "end",
    "for",
    "parfor",
    "while",
    "switch",
    "case",
    "otherwise",
    "try",
    "catch",
    "function",
    "return",
    "break",
    "continue",
    "global",
    "persistent",
    "save",
    "load",
    "delete",
    "clc",
    "format",
    "addpath",
    "rmpath",
    "cd",
    "diary",
    "more",
    "tic",
    "toc",
    "keyboard",
    "dbstop",
    "pause",
];

/// Dynare statements that may continue on the next line after a first line of the form `kw`
/// or `kw(options)` (a variable or symbol list follows): never native MATLAB statements.
const MULTILINE_KEYWORDS: &[&str] = &[
    "var",
    "varexo",
    "varexo_det",
    "parameters",
    "predetermined_variables",
    "trend_var",
    "log_trend_var",
    "varobs",
    "varexobs",
    "stoch_simul",
    "estimation",
    "forecast",
    "shock_decomposition",
    "realtime_shock_decomposition",
    "plot_shock_decomposition",
    "initial_condition_decomposition",
    "squeeze_shock_decomposition",
    "calib_smoother",
    "osr",
    "osr_params",
    "ramsey_model",
    "ramsey_policy",
    "discretionary_policy",
    "evaluate_planner_objective",
    "planner_objective",
    "identification",
    "dynare_sensitivity",
    "method_of_moments",
    "conditional_forecast",
    "plot_conditional_forecast",
    "simul",
    "extended_path",
    "perfect_foresight_setup",
    "perfect_foresight_solver",
    "smoother2histval",
    "sbvar",
    "bvar_density",
    "bvar_forecast",
    "bvar_irf",
    "model_comparison",
    "ms_estimation",
    "ms_simulation",
    "ms_compute_mdd",
    "ms_compute_probabilities",
    "ms_irf",
    "ms_forecast",
    "ms_variance_decomposition",
    "steady",
    "check",
    "model_diagnostics",
    "write_latex_dynamic_model",
    "write_latex_static_model",
    "write_latex_original_model",
    "write_latex_steady_state_model",
    "dsample",
    "save_params_and_steady_state",
    "load_params_and_steady_state",
    "irf_calibration",
    "moment_calibration",
    "histval_file",
    "initval_file",
    "model_info",
    "generate_trace_plots",
    "prior",
    "occbin_setup",
    "occbin_solver",
    "occbin_graph",
    "occbin_write_regimes",
    "estimated_params",
    "estimated_params_init",
    "estimated_params_bounds",
];

/// MATLAB functions that dynare-rust runs when written as native statements (in Dynare they
/// are MATLAB functions, not statements of the .mod language).
const NATIVE_RUN_FUNCTIONS: &[&str] = &["datatomfile", "set_dynare_seed", "set_param_value"];

/// Block statements, with [`MULTILINE_KEYWORDS`] the keywords [`statement_keyword`] knows.
const BLOCK_KEYWORDS: &[&str] = &[
    "model",
    "initval",
    "endval",
    "histval",
    "shocks",
    "mshocks",
    "steady_state_model",
    "observation_trends",
    "optim_weights",
    "conditional_forecast_paths",
    "heteroskedastic_shocks",
    "matched_moments",
    "verbatim",
];

/// The lower-case keyword of a statement written in another case (`Var`, `MODEL`): Dynare's
/// lexer matches its keywords case-insensitively.
fn statement_keyword(kw: &str) -> &str {
    if !kw.bytes().any(|b| b.is_ascii_uppercase()) {
        return kw;
    }
    MULTILINE_KEYWORDS.iter().chain(BLOCK_KEYWORDS).find(|k| k.eq_ignore_ascii_case(kw)).copied().unwrap_or(kw)
}

impl<'a> Parser<'a> {
    /// A native MATLAB statement ending at ';' or at the end of its line (outside brackets),
    /// kept as a [`MATLAB_STATEMENT`] for the run-time warning.
    fn native_line_statement(&mut self) -> Result<()> {
        self.c.native_end = self.c.native_line_end();
        let toks = self.c.statement()?;
        self.commands.push(Command::Other(MATLAB_STATEMENT.into(), vec![(x::raw_join_spaced(toks), OptValue::Flag)]));
        Ok(())
    }

    /// Whether the statement starting with `kw` is a native MATLAB statement ending with its
    /// line: a MATLAB keyword, or a first line `kw`, `kw(...)` without ';' that no Dynare
    /// statement continues (`Simulated_time_series(2000Q3:2050Q2)`, `cmdout`).
    fn native_line_ahead(&self, kw: &str) -> bool {
        if MATLAB_KEYWORDS.contains(&kw) {
            return true;
        }
        if MULTILINE_KEYWORDS.iter().chain(BLOCK_KEYWORDS).any(|k| k.eq_ignore_ascii_case(kw))
            || self.b.symbol(kw).is_some()
        {
            return false;
        }
        let Some(end) = self.c.native_line_end() else { return false };
        // the line is `kw` or `kw(...)`
        let rest = &self.c.t[self.c.i + 1..end];
        rest.is_empty()
            || (rest[0].is("(") && rest[rest.len() - 1].is(")") && {
                let mut depth = 0;
                rest.iter().enumerate().all(|(k, t)| {
                    if t.is("(") {
                        depth += 1;
                    } else if t.is(")") {
                        depth -= 1;
                    }
                    depth > 0 || k == rest.len() - 1
                })
            })
    }

    fn statement(&mut self, t: &'a Token) -> Result<()> {
        if t.is(";") {
            self.c.i += 1;
            return Ok(());
        }
        if t.is("[") {
            // a MATLAB multiple assignment (`[a, b] = f(...);`), passed through by the
            // preprocessor: kept for the warning of an unsupported MATLAB statement
            let toks = self.c.statement()?;
            self.commands
                .push(Command::Other(MATLAB_STATEMENT.into(), vec![(x::raw_join_spaced(toks), OptValue::Flag)]));
            return Ok(());
        }
        if let Some(e) = self.vm_hoisted_end() {
            self.c.i = e; // ---- var models: read before the model block
            return Ok(());
        }
        let Some(kw) = t.ident() else { bail!("{}: unexpected '{}' at the start of a statement", t.loc, t.raw) };
        self.pac_note_dseries();
        if self.bw_assign_ahead() {
            return self.bw_assign(); // ---- simul_backward_model and its dseries
        }
        if self.pac_assign_ahead() {
            return self.pac_assign(); // ---- pac.estimate setup
        }
        if self.ml_statement(kw)? {
            return Ok(()); // `if isoctave`, `load`, `fieldnames`, `for`
        }
        if self.c.peek_at(1).is_some_and(|n| n.is("=")) {
            return self.calibration();
        }
        if self.ws_indexed_assignment_ahead(kw) {
            return self.ws_indexed_assignment();
        }
        if NATIVE_LINE_FUNCTIONS.contains(&kw) {
            return self.native_line_statement();
        }
        if self.native_line_ahead(kw) {
            if !NATIVE_RUN_FUNCTIONS.contains(&kw) {
                return self.native_line_statement();
            }
            // `datatomfile('f', {'y'})` without ';': its arguments end with the line
            self.c.native_end = self.c.native_line_end();
        }
        // Dynare's keywords are case-insensitive (`Var`, `VAREXO`, `Stoch_Simul`)
        let kw = statement_keyword(kw);
        match kw {
            // ---- heterogeneity ----
            "heterogeneity_dimension" => self.het_dimension(),
            "var" if self.het_header().is_some() => self.het_declaration(het::HetKind::Endo),
            "varexo" if self.het_header().is_some() => self.het_declaration(het::HetKind::Exo),
            "parameters" if self.het_header().is_some() => self.het_declaration(het::HetKind::Param),
            "model" if self.het_header().is_some() => {
                let st = self.c.i;
                let r = self.het_model_block();
                self.pp.model_rng.push((st, self.c.i));
                r
            }
            // ---- end heterogeneity ----
            "var" => self.declaration(SymbolKind::Endo),
            "trend_var" => self.q1_trend_var(false),
            "shock_paths" => self.q1_shock_paths_block(),
            "database" => self.shocks_database(), // ---- shocks/shock_paths
            "set_shock_stderr_value" => self.q1_set_shock_stderr_value(),
            "log_trend_var" => self.q1_trend_var(true),
            "varexo" => {
                let before = self.b.exo_names.len();
                self.declaration(SymbolKind::Exo)?;
                self.q1_keep_exo_det_last(before) // varexo after varexo_det
            }
            "parameters" => self.declaration(SymbolKind::Param),
            "varexo_det" => {
                let before = self.b.exo_names.len();
                self.declaration(SymbolKind::Exo)?;
                self.exo_det.extend(before..self.b.exo_names.len());
                Ok(())
            }
            "predetermined_variables" => {
                self.c.i += 1;
                for t in self.c.statement()? {
                    match &t.tok {
                        Tok::Ident(n) => match self.b.symbol(n) {
                            Some(s) if s.kind == SymbolKind::Endo => self.predetermined.push(s.index),
                            _ => bail!("{}: predetermined_variables: '{n}' is not an endogenous variable", t.loc),
                        },
                        Tok::Punct(",") => {}
                        _ => bail!("{}: unexpected '{}' in predetermined_variables", t.loc, t.raw),
                    }
                }
                Ok(())
            }
            // ---- stream PP: model-editing statements
            "datatomfile" => {
                // datatomfile('file', {'v1', ...}[, {'n1', ...}]): string arguments only
                self.c.i += 1;
                let toks = self.c.statement()?;
                let (mut file, mut groups, mut cur, mut in_cell) = (None, Vec::new(), Vec::new(), false);
                for t in toks {
                    match &t.tok {
                        Tok::Str(s) if in_cell => cur.push(OptValue::Str(s.clone())),
                        Tok::Str(s) if file.is_none() => file = Some(s.clone()),
                        // `[]`: all the variables (an empty list); `{'y'; 'c'}`: a column cell
                        Tok::Punct("{" | "[") => in_cell = true,
                        Tok::Punct("}" | "]") => {
                            in_cell = false;
                            groups.push(OptValue::List(std::mem::take(&mut cur)));
                        }
                        Tok::Punct(";") if in_cell => {}
                        Tok::Punct("(" | ")" | ",") => {}
                        _ => bail!("{}: datatomfile: only string and cell-of-string arguments are supported", t.loc),
                    }
                }
                let file = file.ok_or_else(|| anyhow!("{}: datatomfile: missing file name", t.loc))?;
                let mut o: Options = vec![("file".into(), OptValue::Str(file))];
                for (k, g) in ["vars", "names"].into_iter().zip(groups) {
                    o.push((k.into(), g));
                }
                self.commands.push(Command::Other("datatomfile".into(), o));
                Ok(())
            }
            "model_options" => {
                self.c.i += 1;
                let opts = self.header_options()?;
                self.apply_model_options(&opts);
                Ok(())
            }
            "model_local_variable" => {
                self.c.i += 1;
                let toks = self.c.statement()?;
                self.q4_model_local_variable_labels(toks); // TeX / long names
                let mut k = 0;
                while k < toks.len() {
                    let t = &toks[k];
                    if t.is("(") {
                        // (long_name='...')
                        let mut depth = 0;
                        while k < toks.len() {
                            if toks[k].is("(") {
                                depth += 1;
                            } else if toks[k].is(")") {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            k += 1;
                        }
                    } else if let Tok::Ident(n) = &t.tok {
                        self.b.declare_local(n).map_err(|e| anyhow!("{}: {e}", t.loc))?;
                    }
                    k += 1;
                }
                Ok(())
            }
            "model_remove" => {
                let loc = self.c.next()?.loc.clone();
                let toks = self.c.statement()?;
                let sel = tag_selection(toks)?;
                let excluded = self.b.remove_equations(&sel, true).map_err(|e| anyhow!("{loc}: {e}"))?;
                self.remove_original_equations(&sel);
                for v in excluded {
                    if self.b.is_used(&v) {
                        // still used elsewhere: turned into an exogenous variable (manual)
                        self.q4_endo_to_exo(&v).map_err(|e| anyhow!("{loc}: model_remove: {e}"))?;
                    } else {
                        self.remove_symbol(&v).map_err(|e| anyhow!("{loc}: model_remove: {e}"))?;
                    }
                }
                Ok(())
            }
            "model_replace" => {
                let loc = self.c.next()?.loc.clone();
                let toks = self.c.statement()?;
                let sel = tag_selection(toks)?;
                self.b.remove_equations(&sel, false).map_err(|e| anyhow!("{loc}: {e}"))?;
                self.remove_original_equations(&sel);
                self.model_body()
            }
            "var_remove" => {
                let loc = self.c.next()?.loc.clone();
                let toks = self.c.statement()?;
                for t in toks {
                    if let Tok::Ident(n) = &t.tok {
                        self.remove_symbol(n).map_err(|e| anyhow!("{loc}: {e}"))?;
                    }
                }
                Ok(())
            }
            // ---- end stream PP
            "histval" => self.histval_block(),
            "steady_state_model" => {
                let st = self.c.i;
                let r = self.steady_state_model_block();
                self.pp.ssm_rng.push((st, self.c.i));
                r
            }
            "verbatim" => self.verbatim_block(),
            "mshocks" => self.shocks_block_kind(true),
            // ---- stream EP
            "perfect_foresight_controlled_paths" => self.controlled_paths_block(),
            "model" => self.model_block(),
            "initval" => self.value_block(true),
            "endval" => self.value_block(false),
            "shocks" => self.shocks_block(),
            "estimated_params" => self.estimated_params(),
            "estimated_params_init" => self.estimated_params_init(),
            "occbin_constraints" => self.occbin_constraints_block(),
            // [stream P] optimal policy
            "planner_objective" | "optim_weights" | "osr_params_bounds" | "ramsey_constraints" => {
                let st = self.c.i;
                let r = self.policy_statement(kw);
                if kw == "ramsey_constraints" {
                    self.pp.model_rng.push((st, self.c.i));
                }
                r
            }
            _ if self.vm_handles(kw) => self.vm_statement(kw), // ---- var models ----
            _ if self.pac_handles(kw) => self.pac_statement(kw), // ---- pac ----
            _ if self.rep_handles(kw) => self.rep_statement(), // reporting: r.write(); ...
            _ if self.x_handles(kw) => self.x_statement(kw),   // [stream X]
            _ if self.q4_handles(kw) => self.q4_statement(kw),
            _ if self.q3_handles(kw) => self.q3_statement(kw),
            // ---- MoM ----
            "matched_moments" => self.matched_moments_block(),
            "matched_irfs" => self.matched_irfs_block(),
            "matched_irfs_weights" => self.matched_irfs_weights_block(),
            // ---- end MoM ----
            "varobs" => {
                self.c.i += 1;
                for t in self.c.statement()? {
                    match &t.tok {
                        Tok::Ident(n) => {
                            if self.b.symbol(n).is_none_or(|s| s.kind != SymbolKind::Endo) {
                                bail!("{}: varobs: '{n}' is not an endogenous variable", t.loc);
                            }
                            self.varobs.push(n.clone())
                        }
                        Tok::Punct(",") => {}
                        _ => bail!("{}: unexpected '{}' in varobs", t.loc, t.raw),
                    }
                }
                Ok(())
            }
            _ => self.command(kw),
        }
    }

    /// `var a b $tex$ (long_name='..') c;`
    fn declaration(&mut self, kind: SymbolKind) -> Result<()> {
        self.c.i += 1;
        // var(log), var(deflator = ...), var(log_deflator = ...)
        let q1_opts = if kind == SymbolKind::Endo && self.c.at("(") { Some(self.q1_var_options()?) } else { None };
        self.c.skip_parens()?;
        let mut q1_names = Vec::new();
        loop {
            if self.c.eat(";") {
                if let Some(o) = q1_opts {
                    self.q1_apply_var_options(o, q1_names);
                }
                return Ok(());
            }
            if self.c.eat(",") {
                continue;
            }
            let (name, loc) = self.c.ident()?;
            self.b.declare(name, kind).map_err(|e| anyhow!("{loc}: {e}"))?;
            q1_names.push(name.to_string());
            // ---- stream B (LaTeX): TeX and long names, with Dynare's defaults
            let mut tex = name.replace('_', "\\_");
            let mut long = name.to_string();
            while let Some(t) = self.c.peek() {
                if let Tok::Tex(s) = &t.tok {
                    tex = s.clone();
                    self.c.i += 1;
                } else if t.is("(") {
                    let st = self.c.i;
                    self.c.skip_parens()?;
                    for item in split_commas(&self.c.t[st + 1..self.c.i - 1]) {
                        if let [k, eq, v] = item
                            && k.is_ident("long_name")
                            && eq.is("=")
                            && let Tok::Str(s) = &v.tok
                        {
                            long = s.clone();
                        }
                    }
                } else {
                    break;
                }
            }
            self.labels.push(kind, tex, long);
        }
    }

    /// Value of a name outside the model block: calibrated parameters and constants.
    fn lookup_param(&self, name: &str, lag: i32, loc: &Loc) -> Result<f64> {
        match self.b.symbol(name) {
            Some(s) if s.kind == SymbolKind::Param => {
                if lag != 0 {
                    bail!("{loc}: parameter '{name}' cannot have a lead/lag");
                }
                Ok(self.b.params[s.index])
            }
            Some(s) => bail!("{loc}: {:?} variable '{name}' cannot be used in this expression", s.kind),
            None => self
                .mf_locals
                .get(name)
                .copied()
                .or_else(|| builtin_const(name))
                .ok_or_else(|| anyhow!("{loc}: unknown symbol '{name}'")),
        }
    }

    /// `ast` with the parameters and mod-file locals replaced by their values; the endogenous
    /// and exogenous variables stay (the preprocessor compiles them to `oo_.steady_state(i)` /
    /// `oo_.exo_steady_state(k)`, read when the statement runs). `found` tells whether there was one.
    fn bind_steady(&self, ast: &Ast, found: &mut bool) -> Result<Ast> {
        Ok(match ast {
            Ast::Num(_) => ast.clone(),
            Ast::Var { name, lag, loc } => match self.b.symbol(name) {
                Some(s) if matches!(s.kind, SymbolKind::Endo | SymbolKind::Exo) => {
                    if *lag != 0 {
                        bail!("{loc}: variable '{name}' cannot have a lead/lag in this expression");
                    }
                    *found = true;
                    ast.clone()
                }
                _ => Ast::Num(self.lookup_param(name, *lag, loc)?),
            },
            Ast::Un(op, a) => Ast::Un(*op, Box::new(self.bind_steady(a, found)?)),
            Ast::Bin(op, a, b) => {
                Ast::Bin(*op, Box::new(self.bind_steady(a, found)?), Box::new(self.bind_steady(b, found)?))
            }
            Ast::Op(op, args) => Ast::Op(*op, args.iter().map(|a| self.bind_steady(a, found)).collect::<Result<_>>()?),
        })
    }

    /// The expression of a value item that mentions an endogenous variable, bound by
    /// [`Self::bind_steady`]; `None` when it does not.
    fn steady_expr(&self, toks: &[Token]) -> Result<Option<Ast>> {
        let endo = toks.iter().any(|t| {
            t.ident()
                .and_then(|n| self.b.symbol(n))
                .is_some_and(|s| matches!(s.kind, SymbolKind::Endo | SymbolKind::Exo))
        });
        if !endo {
            return Ok(None);
        }
        let mut found = false;
        let ast = self.bind_steady(&slice_expr(toks)?, &mut found)?;
        Ok(found.then_some(ast))
    }

    /// A value of a `shocks` / `endval(learnt_in)` statement: a number, or an expression with
    /// endogenous / exogenous variables kept for the run.
    fn shock_value(&self, toks: &[Token]) -> Result<PathValue> {
        Ok(match self.steady_expr(toks)? {
            Some(e) => PathValue::Steady(e),
            None => PathValue::Num(self.eval_tokens(toks)?),
        })
    }

    fn eval_tokens(&self, toks: &[Token]) -> Result<f64> {
        let ast = slice_expr(toks)?;
        eval_ast(&ast, &|n, l, loc| self.lookup_param(n, l, loc))
    }

    /// [`src_join`] with the mod-file local variables (`betabar = ...;` of an undeclared name, a
    /// MATLAB variable for Dynare) replaced by their values: the source is compiled again at
    /// run time against the model's symbols only.
    fn src_join_locals(&self, toks: &[Token]) -> String {
        let mut out = Vec::with_capacity(toks.len());
        for (k, t) in toks.iter().enumerate() {
            let local = t.ident().filter(|n| self.b.symbol(n).is_none()).and_then(|n| self.mf_locals.get(n));
            match local {
                Some(v) if !toks.get(k + 1).is_some_and(|n| n.is("(")) => out.push(format!("({v:?})")),
                _ => out.push(src_join(std::slice::from_ref(t))),
            }
        }
        out.join(" ")
    }

    fn eval_opt(&self, toks: &[Token]) -> Result<Option<f64>> {
        if toks.is_empty() { Ok(None) } else { self.eval_tokens(toks).map(Some) }
    }

    /// `param = expr;`
    fn calibration(&mut self) -> Result<()> {
        // a name that is not a Dynare symbol starts a native MATLAB statement, which ends with
        // its line when no ';' ends it first (`aaa = 8343553 / 3737 * 90`)
        let native_end = match self.c.peek().and_then(|t| t.ident()) {
            Some(n) if self.b.symbol(n).is_none() => self.c.native_line_end(),
            _ => None,
        };
        let (name, loc) = self.c.ident()?;
        self.c.expect("=")?;
        self.c.native_end = native_end;
        let toks = self.c.statement()?;
        // ---- shocks/shock_paths: MATLAB workspace values (vectors, dseries, tables)
        let v = match self.eval_tokens(toks) {
            Ok(v) => v,
            Err(_) if self.b.symbol(name).is_none() => return self.shocks_workspace_assignment(name, toks),
            Err(e) => return Err(e),
        };
        match self.b.symbol(name) {
            Some(s) if s.kind == SymbolKind::Param => {
                let k = s.index;
                // ---- model_info: the preprocessor's eval context cannot evaluate an assignment
                // that uses a mod-file local variable or a parameter it does not know
                let unknown = toks.iter().filter_map(|t| t.ident()).any(|id| {
                    self.mf_locals.contains_key(id)
                        || self.b.symbol(id).is_some_and(|s| {
                            s.kind == SymbolKind::Param
                                && (self.b.params[s.index].is_nan()
                                    || self.block_opts.unknown_params.contains(&s.index))
                        })
                });
                if unknown {
                    self.block_opts.unknown_params.insert(k);
                } else {
                    self.block_opts.unknown_params.remove(&k);
                }
                // ---- end model_info
                self.x_param_assignment(k, name, v); // [stream X]
                self.b.params[k] = v;
                Ok(())
            }
            Some(s) => bail!("{loc}: cannot assign {:?} variable '{name}' outside initval/endval", s.kind),
            // stream PP: a mod-file local variable (it does not shift the symbol ids that aux
            // names use: checked against the 7.1 preprocessor, reference/pp/pp_mflocal.mod)
            None => {
                self.mf_locals.insert(name.to_string(), v);
                Ok(())
            }
        }
    }

    fn header_options(&mut self) -> Result<Options> {
        let toks = self.c.statement()?;
        if toks.is_empty() {
            return Ok(Vec::new());
        }
        if !(toks[0].is("(") && toks.last().is_some_and(|t| t.is(")"))) {
            bail!("{}: unexpected '{}'", toks[0].loc, toks[0].raw);
        }
        Ok(self.parse_options(&toks[1..toks.len() - 1]))
    }

    fn model_block(&mut self) -> Result<()> {
        let _loc = self.c.next()?.loc.clone();
        // several model blocks are concatenated, as in Dynare (stream PP)
        self.model_seen = true;
        let opts = self.header_options()?;
        self.apply_model_options(&opts);
        self.c.in_model_block = true;
        let r = self.model_body();
        self.c.in_model_block = false;
        r?;
        self.q4_after_model_block() // exclude_eqs / include_eqs
    }

    /// stream PP: `model(...)` / `model_options(...)` options. `linear` and
    /// `differentiate_forward_vars` change the model; `block`, `bytecode`, `use_dll`, `mfs`,
    /// `cutoff`, `no_static`, ... only select Dynare's solving/compilation strategies, which map to
    /// dynare-rust's own (compiled tapes, sparse solvers): they are accepted.
    fn apply_model_options(&mut self, opts: &Options) {
        for (k, v) in opts {
            match k.as_str() {
                "linear" => self.b.linear = true,
                "differentiate_forward_vars" => {
                    let list = match v {
                        OptValue::List(l) => l
                            .iter()
                            .filter_map(|x| if let OptValue::Str(s) = x { Some(s.clone()) } else { None })
                            .collect(),
                        OptValue::Str(s) => s.split([',', ' ']).filter(|x| !x.is_empty()).map(str::to_string).collect(),
                        _ => Vec::new(),
                    };
                    self.b.differentiate_forward_vars = Some(list);
                }
                "block"
                | "bytecode"
                | "use_dll"
                | "no_static"
                | "mfs"
                | "static_mfs"
                | "cutoff"
                | "parallel_local_files"
                | "linear_decomposition" => {
                    // ---- model_info: options of the block decomposition
                    if k == "bytecode" {
                        self.pp.bytecode = true;
                    }
                    let num = || if let OptValue::Num(x) = v { Some(*x) } else { None };
                    match k.as_str() {
                        "block" => self.block_opts.block = true,
                        "no_static" => self.block_opts.no_static = true,
                        "mfs" => self.block_opts.mfs = num().map(|x| x as i32),
                        "static_mfs" => self.block_opts.static_mfs = num().map(|x| x as i32),
                        "cutoff" => self.block_opts.cutoff = num(),
                        _ => {}
                    }
                }

                "balanced_growth_test_tol" => {
                    if let OptValue::Num(x) = v {
                        self.b.bg.tol = *x;
                    }
                }
                _ => eprintln!("WARNING: model option '{k}' is not supported by dynare-rust and is ignored"),
            }
        }
    }

    /// On-the-fly declarations of a model block (`c|e`, `e|x`, `beta|p`, `[endogenous='c']`):
    /// the preprocessor creates the symbols of the equations in order of first appearance and
    /// gives them their type afterwards (an undeclared name typed further down keeps its rank;
    /// with `nostrict`, a name never typed is exogenous). Nothing is done in a block without
    /// such declarations.
    fn on_the_fly_declarations(&mut self) -> Result<()> {
        use std::collections::{HashMap, HashSet};
        let t = self.c.t;
        let mut end = self.c.i;
        while end + 1 < t.len() && !(t[end].is_ident("end") && t[end + 1].is(";")) {
            end += 1;
        }
        let blk = &t[self.c.i..end.min(t.len())];
        let mut kinds: HashMap<&str, SymbolKind> = HashMap::new();
        let mut tagged = false;
        for (k, tok) in blk.iter().enumerate() {
            if let Some(n) = tok.ident() {
                let kind = match tok.raw.get(n.len()..) {
                    Some("|e" | "|E") => Some(SymbolKind::Endo),
                    Some("|x" | "|X") => Some(SymbolKind::Exo),
                    Some("|p" | "|P") => Some(SymbolKind::Param),
                    _ => None,
                };
                if let Some(kind) = kind {
                    kinds.entry(n).or_insert(kind);
                }
                // `[endogenous='c']`
                let tag = match n.to_ascii_lowercase().as_str() {
                    "endogenous" => Some(SymbolKind::Endo),
                    "exogenous" => Some(SymbolKind::Exo),
                    "parameter" => Some(SymbolKind::Param),
                    _ => None,
                };
                if let (Some(kind), Some(eq), Some(v)) = (tag, blk.get(k + 1), blk.get(k + 2))
                    && eq.is("=")
                    && let Tok::Str(s) = &v.tok
                {
                    tagged = true;
                    kinds.entry(s.as_str()).or_insert(kind);
                }
            }
        }
        if kinds.is_empty() && !tagged {
            return Ok(());
        }
        let skip_after = ["pac_expectation", "var_expectation", "pac_target_nonstationary"];
        let mut locals: HashSet<&str> = HashSet::new();
        let mut depth_tag = 0;
        let mut stmt_start = true;
        for (k, tok) in blk.iter().enumerate() {
            if stmt_start && tok.is("[") {
                depth_tag += 1;
            }
            if depth_tag > 0 {
                if tok.is("]") {
                    depth_tag -= 1;
                }
                continue;
            }
            if tok.is(";") {
                stmt_start = true;
                continue;
            }
            let first = stmt_start;
            stmt_start = false;
            if first && tok.is("#") {
                if let Some(n) = blk.get(k + 1).and_then(|x| x.ident()) {
                    locals.insert(n);
                }
                continue;
            }
            let Some(n) = tok.ident() else { continue };
            if locals.contains(n)
                || self.b.symbol(n).is_some()
                || builtin_const(n).is_some()
                || (k >= 2 && blk[k - 1].is("(") && blk[k - 2].ident().is_some_and(|f| skip_after.contains(&f)))
            {
                continue;
            }
            let kind = match kinds.get(n) {
                Some(&kind) => kind,
                None => {
                    // a function call unless `name(lead/lag)`
                    if blk.get(k + 1).is_some_and(|x| x.is("(")) {
                        let lag = match (blk.get(k + 2), blk.get(k + 3), blk.get(k + 4)) {
                            (Some(a), Some(b), _) if matches!(a.tok, Tok::Num(_)) && b.is(")") => true,
                            (Some(s), Some(a), Some(b)) => {
                                (s.is("+") || s.is("-")) && matches!(a.tok, Tok::Num(_)) && b.is(")")
                            }
                            _ => false,
                        };
                        if !lag {
                            continue;
                        }
                    }
                    if !self.q4_flags.nostrict {
                        continue; // "Unknown symbol" when the equation is read
                    }
                    println!("WARNING: Unknown symbol '{n}' in the model block: declared exogenous (nostrict).");
                    SymbolKind::Exo
                }
            };
            self.b.declare(n, kind).map_err(|e| anyhow!("{}: {e}", tok.loc))?;
            self.labels.push(kind, n.replace('_', "\\_"), n.to_string());
        }
        Ok(())
    }

    /// Model equations up to `end;` (model block, model_replace).
    fn model_body(&mut self) -> Result<()> {
        self.vm_hoist()?; // ---- var models ----
        self.vm_prepass()?; // ---- var models ----
        self.on_the_fly_declarations()?;
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                return Ok(());
            }
            let mut tags = Vec::new();
            let tag_start = self.c.i;
            let stmt_start = self.c.i;
            if self.c.eat("[") {
                loop {
                    if self.c.eat("]") {
                        break;
                    }
                    if self.c.eat(",") {
                        continue;
                    }
                    let key = self.c.next()?.raw.to_ascii_lowercase(); // tag_pair: str_tolower (stream PP)
                    let mut val = String::new();
                    if self.c.eat("=") {
                        let v = self.c.next()?;
                        val = match &v.tok {
                            Tok::Str(s) => s.clone(),
                            _ => v.raw.clone(),
                        };
                    }
                    // on-the-fly declaration: `[endogenous='c']` declares `c` if needed
                    let kind = match key.as_str() {
                        "endogenous" => Some(SymbolKind::Endo),
                        "exogenous" => Some(SymbolKind::Exo),
                        "parameter" => Some(SymbolKind::Param),
                        _ => None,
                    };
                    if let Some(kind) = kind
                        && self.b.symbol(&val).is_none()
                    {
                        let loc = &self.c.t[self.c.i - 1].loc;
                        self.b.declare(&val, kind).map_err(|e| anyhow!("{loc}: {e}"))?;
                        self.labels.push(kind, val.replace('_', "\\_"), val.clone());
                    }
                    tags.push((key, val));
                }
            }
            self.c.nums.clear();
            if self.c.eat("#") {
                let (name, nloc) = self.c.ident()?;
                self.c.expect("=")?;
                let ast = self.c.expr()?;
                self.c.expect(";")?;
                self.pp.eqs.push((Vec::new(), stmt_start, self.c.i));
                let node = self.resolve_ast(&ast)?;
                self.b.add_local(name, node).map_err(|e| anyhow!("{nloc}: {e}"))?;
                // stream PP: replayed by model::dtree
                let symb_id = self.b.symbol(name).map(|s| s.symb_id).unwrap_or(0);
                let nums = std::mem::take(&mut self.c.nums);
                self.note_special(&ast, &[]);
                self.b.items.push(crate::model::dtree::ModelItem::Local { symb_id, ast: ast.clone(), nums });
                self.original.locals.push((name.to_string(), ast)); // stream B
                continue;
            }
            self.q1.eq_line = self.c.peek().map_or(0, |t| t.loc.line);
            // ---- var models / pac: the var_expectation, pac_expectation and
            // pac_target_nonstationary operators are replaced in the transform pass
            // (model::dtree::pac); the direct model gets 0 in their place
            let lhs = self.c.expr()?;
            let has_rhs = self.c.eat("=");
            let rhs = if has_rhs { self.c.expr()? } else { Ast::Num(0.0) };
            self.het_register_sums()?; // ---- heterogeneity ----
            // ---- `eq ⟂ x > lb;` / `eq _|_ x < ub;` (complementarity condition)
            if self.c.eat("⟂") {
                let st = self.c.i;
                while self.c.peek().is_some_and(|t| !t.is(";")) {
                    self.c.i += 1;
                }
                tags.push((crate::model::MCP_TAG.to_string(), src_join(&self.c.t[st..self.c.i])));
            }
            self.c.expect(";")?;
            self.pp.eqs.push((tags.clone(), stmt_start, self.c.i));
            if tags.iter().any(|(k, _)| k == "mcp") && tag_start < self.c.i {
                let (a, b) = (&self.c.t[tag_start], &self.c.t[self.c.i - 1]);
                self.pp.warn.push(format!(
                    "WARNING: {}: Specifying complementarity conditions with the 'mcp' tag is obsolete. Please consider switching to the new syntax using the perpendicular symbol.",
                    span_text(a, b)
                ));
            }
            // ---- stream PP: replayed by model::dtree
            {
                let nums = std::mem::take(&mut self.c.nums);
                self.note_special(&lhs, &tags);
                self.note_special(&rhs, &tags);
                let item = crate::model::dtree::ModelItem::Equation {
                    lhs: lhs.clone(),
                    rhs: has_rhs.then(|| rhs.clone()),
                    nums,
                    tags: tags.clone(),
                    line: self.q1.eq_line,
                };
                self.b.items.push(item);
            }
            let (lhs, rhs) = (strip_placeholders(&lhs), strip_placeholders(&rhs));
            if tags.iter().any(|(k, _)| k == "static") {
                // [static] equations only enter the static model (model::dtree)
                self.original.equations.push((lhs, rhs, tags.clone())); // stream B
                continue;
            }
            let (l, r) = (self.resolve_ast(&lhs)?, self.resolve_ast(&rhs)?);
            self.original.equations.push((lhs, rhs, tags.clone())); // stream B
            if tags.iter().any(|(k, _)| k == "bind" || k == "relax") {
                let loc = self.c.loc();
                self.b.add_occbin_equation(l, r, tags).map_err(|e| anyhow!("{loc}: {e}"))?;
                continue;
            }
            self.b.add_equation(l, r, tags);
        }
    }

    /// stream PP: drops the parsed copies (stream B's original model) of removed equations.
    fn remove_original_equations(&mut self, sel: &[Vec<(String, String)>]) {
        let m = |tags: &[(String, String)]| {
            sel.iter().any(|s| s.iter().all(|(k, v)| tags.iter().any(|(a, b)| a == k && b == v)))
        };
        self.original.equations.retain(|(_, _, t)| !m(t));
        self.pp.eqs.retain(|(t, _, _)| t.is_empty() || !m(t));
    }

    /// stream PP: `var_remove` / variables excluded by `model_remove`. Indices held by the
    /// statements parsed so far move down with the model's.
    fn remove_symbol(&mut self, name: &str) -> Result<()> {
        let (kind, idx) = self.b.remove_symbol(name)?;
        let fix = |v: &mut usize| -> bool {
            if *v == idx {
                return false;
            }
            if *v > idx {
                *v -= 1;
            }
            true
        };
        match kind {
            SymbolKind::Endo => {
                self.initval.retain_mut(|(j, _)| fix(j));
                self.endval.retain_mut(|(j, _)| fix(j));
                self.histval.retain_mut(|(endo, j, _, _)| !*endo || fix(j));
                self.predetermined.retain_mut(fix);
                self.varobs.retain(|v| v != name);
                self.labels.endo_tex.remove(idx);
                self.labels.endo_long.remove(idx);
            }
            SymbolKind::Exo => {
                self.initval_exo.retain_mut(|(j, _)| fix(j));
                self.histval.retain_mut(|(endo, j, _, _)| *endo || fix(j));
                self.exo_det.retain_mut(fix);
                self.det_shocks.retain_mut(|d| fix(&mut d.exo));
                self.labels.exo_tex.remove(idx);
                self.labels.exo_long.remove(idx);
            }
            SymbolKind::Param => {
                self.labels.param_tex.remove(idx);
                self.labels.param_long.remove(idx);
            }
            SymbolKind::Local | SymbolKind::Trend => {}
        }
        for (_, b) in &mut self.state_blocks {
            match (kind, b) {
                (SymbolKind::Endo, RawStateBlock::Initval { endo, .. }) => endo.retain_mut(|(j, _)| fix(j)),
                (SymbolKind::Exo, RawStateBlock::Initval { exo, .. }) => exo.retain_mut(|(j, _)| fix(j)),
                (SymbolKind::Endo, RawStateBlock::Histval(h)) => h.retain_mut(|(exo, j, _, _)| *exo || fix(j)),
                (SymbolKind::Exo, RawStateBlock::Histval(h)) => h.retain_mut(|(exo, j, _, _)| !*exo || fix(j)),
                (SymbolKind::Exo, RawStateBlock::Shocks { det, .. }) => det.retain_mut(|d| fix(&mut d.exo)),
                _ => {}
            }
        }
        let tgt_used = |t: &Tgt| matches!((kind, t), (SymbolKind::Endo, Tgt::Endo(_)) | (SymbolKind::Exo, Tgt::Exo(_)));
        let in_shocks = self.shocks.iter().any(|s| match s {
            ShockStmt::Var(t, _) | ShockStmt::Std(t, _) => tgt_used(t),
            ShockStmt::Cov(a, b, _) | ShockStmt::Corr(a, b, _) => tgt_used(a) || tgt_used(b),
        });
        if in_shocks || !self.est.is_empty() || !self.est_init.is_empty() {
            bail!(
                "var_remove/model_remove of '{name}' after a shocks or estimated_params block is not supported; move it before"
            );
        }
        Ok(())
    }

    /// stream PP: checks the operators only the transform pass (`model::dtree`) handles.
    fn note_special(&mut self, ast: &Ast, tags: &[(String, String)]) {
        if ast.any_op(&|o| matches!(o, SpecialOp::Expectation(_))) {
            self.b.needs_dtree.get_or_insert_with(|| "EXPECTATION()".to_string());
        }
        if tags.iter().any(|(k, _)| k == "static") {
            self.b.needs_dtree.get_or_insert_with(|| "a [static] equation".to_string());
        }
    }

    /// Resolves an equation AST into the model DAG.
    fn resolve_ast(&mut self, ast: &Ast) -> Result<NodeId> {
        self.q4_declare_unknown(ast)?; // nostrict
        let mut dag = std::mem::take(&mut self.b.dag);
        let pre = &self.predetermined;
        let b = &mut self.b;
        let r = ast_to_node(ast, &mut dag, &mut |d, name, lag, loc| match b.symbol(name) {
            Some(s) => match s.kind {
                SymbolKind::Endo => Ok(d.endo(s.index, lag - pre.contains(&s.index) as i32)),
                SymbolKind::Exo => Ok(d.exo(s.index, lag)),
                SymbolKind::Param if lag != 0 => {
                    bail!("{loc}: '{name}' cannot have a lead/lag")
                }
                SymbolKind::Param => Ok(d.param(s.index)),
                // a model-local variable with a lead/lag shifts its definition (stream PP)
                SymbolKind::Local => {
                    let v = b.use_local(s.index);
                    Ok(if lag == 0 { v } else { d.shift(v, -lag) })
                }
                // trend variables only exist in the dtree replay (detrending);
                // this older DAG is only used by optimal policy, which rejects them
                SymbolKind::Trend => Ok(d.num(1.0)),
            },
            None => match builtin_const(name) {
                Some(v) => Ok(d.num(v)),
                None => bail!("{loc}: unknown symbol '{name}'"),
            },
        });
        self.b.dag = dag;
        r
    }

    /// Evaluates `toks` with parameters plus the endogenous/exogenous values set so far by
    /// `initval` (0 otherwise), as Dynare's driver evaluates initval expressions in order.
    fn eval_with_values(&self, toks: &[Token], endo: &[(usize, f64)], exo: &[(usize, f64)]) -> Result<f64> {
        let ast = slice_expr(toks)?;
        eval_ast(&ast, &|n, l, loc| match self.b.symbol(n) {
            Some(s) if s.kind == SymbolKind::Endo => {
                Ok(endo.iter().rev().find(|(j, _)| *j == s.index).map_or(0.0, |x| x.1))
            }
            Some(s) if s.kind == SymbolKind::Exo => {
                Ok(exo.iter().rev().find(|(j, _)| *j == s.index).map_or(0.0, |x| x.1))
            }
            _ => self.lookup_param(n, l, loc),
        })
    }

    /// `initval; x = expr; ... end;` (also `endval`).
    fn value_block(&mut self, initval: bool) -> Result<()> {
        self.c.i += 1;
        let opts = self.header_options()?;
        // ---- stream EP: endval(learnt_in=p), p >= 2 (p = 1: a regular endval block); dates
        if !initval {
            let li = shk::learnt_in_option(&opts, "endval")?;
            if li.1.is_some() || li.0 > 1 {
                return self.learnt_endval_block(li);
            }
        }
        if initval && self.endval_seen {
            bail!("an 'initval' block cannot appear after an 'endval' block");
        }
        let mut sources = Vec::new();
        let mut endo_vals = self.initval.clone();
        let mut exo_vals = self.initval_exo.clone();
        let (endo_st, exo_st) = (endo_vals.len(), exo_vals.len());
        let avr = opt(&opts, "all_values_required").is_some();
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                if avr {
                    let name = if initval { "initval" } else { "endval" };
                    let endo: Vec<usize> = endo_vals[endo_st..].iter().map(|x| x.0).collect();
                    let exo: Vec<usize> = exo_vals[exo_st..].iter().map(|x| x.0).collect();
                    self.check_all_values(name, name, &endo, &exo)?;
                }
                if !initval {
                    self.endval_seen = true;
                    self.commands.push(Command::Endval(sources));
                } else {
                    let block = RawStateBlock::Initval {
                        endo: endo_vals[endo_st..].to_vec(),
                        exo: exo_vals[exo_st..].to_vec(),
                        sources,
                    };
                    self.state_blocks.push((self.commands.len(), block));
                }
                return Ok(());
            }
            let (name, loc) = self.c.ident()?;
            // EndValStatement::checkPass
            if !initval
                && self.b.symbol(name).is_some_and(|s| s.kind == SymbolKind::Exo && self.exo_det.contains(&s.index))
            {
                bail!("{loc}: {name} is an exogenous deterministic.");
            }
            // `x += v` / `x *= v` need endval(learnt_in>1 or date) (Dynare 7.2 wording; 7.1 lacked "or date")
            if !initval && self.c.peek_at(1).is_some_and(|t| t.is("=")) {
                if let Some(op) = ["+", "*"].into_iter().find(|op| self.c.peek().is_some_and(|t| t.is(op))) {
                    bail!(
                        "{loc}: endval: '{name} {op}= ...' line not allowed unless 'learnt_in' option with value >1 or date is passed"
                    );
                }
            }
            self.c.expect("=")?;
            let toks = self.c.statement()?;
            sources.push((name.to_string(), self.src_join_locals(toks)));
            let v = self.eval_with_values(toks, &endo_vals, &exo_vals)?;
            match self.b.symbol(name) {
                Some(s) if s.kind == SymbolKind::Endo => endo_vals.push((s.index, v)),
                Some(s) if s.kind == SymbolKind::Exo => {
                    exo_vals.push((s.index, v));
                    if initval {
                        self.initval_exo.retain(|(j, _)| *j != s.index);
                        self.initval_exo.push((s.index, v));
                    }
                }
                _ => {}
            }
            match self.b.symbol(name) {
                Some(s) if s.kind == SymbolKind::Endo => {
                    let list = if initval { &mut self.initval } else { &mut self.endval };
                    list.retain(|(j, _)| *j != s.index);
                    list.push((s.index, v));
                }
                Some(s) if s.kind == SymbolKind::Exo => {} // exogenous steady state: always 0 here
                _ => bail!("{loc}: '{name}' is not a variable"),
            }
        }
    }

    // ---- stream EP ----
    /// Body of `shocks(learnt_in=p[, overwrite]); var e; periods ...; values|add|multiply ...; end;`
    /// (`mshocks(learnt_in=p)`: `values` multiply the anticipated terminal steady state).
    fn learnt_shocks_block(
        &mut self,
        li: (usize, Option<crate::dates::Date>),
        multiplicative: bool,
        overwrite: bool,
        relative_to_initval: bool,
    ) -> Result<()> {
        use crate::ep::decl::{LearntKind, LearntShock};
        let learnt_in = shk::learnt_in_label(li);
        if overwrite {
            self.ep.learnt_shocks.retain(|s| s.learnt_in_date != li.1 || (li.1.is_none() && s.learnt_in != li.0));
        }
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                return Ok(());
            }
            if !self.c.at_ident("var") {
                let t = self.c.next()?;
                bail!("{}: shocks(learnt_in=...): unexpected '{}'", t.loc, t.raw);
            }
            self.c.i += 1;
            let (a, la) = self.c.ident()?;
            let la = la.clone();
            let exo = self.exo_index(a, &la)?;
            self.c.expect(";")?;
            let mut periods: Vec<shk::PGroup> = Vec::new();
            let mut values: Vec<Vec<PathValue>> = Vec::new();
            let mut kind = match (multiplicative, relative_to_initval) {
                (false, _) => LearntKind::Level,
                (true, false) => LearntKind::MultiplySteadyState,
                (true, true) => LearntKind::MultiplyInitialSteadyState, // ---- shocks/shock_paths
            };
            while self.c.at_ident("periods")
                || self.c.at_ident("values")
                || self.c.at_ident("add")
                || self.c.at_ident("multiply")
            {
                let kw = self.c.next()?;
                let (kwr, loc) = (kw.raw.clone(), kw.loc.clone());
                let toks = self.c.statement()?;
                if kwr == "periods" {
                    periods = shk::sim_period_list(toks).map_err(|e| anyhow!("{loc}: {e}"))?;
                } else {
                    if multiplicative && kwr != "values" {
                        bail!("{loc}: mshocks(learnt_in=...): only 'values' is allowed");
                    }
                    if !multiplicative {
                        kind = match kwr.as_str() {
                            "add" => LearntKind::Add,
                            "multiply" => LearntKind::Multiply,
                            _ => LearntKind::Level,
                        };
                    }
                    values = self.shocks_values_x(toks)?;
                }
            }
            for (g, v) in self.shocks_pair(periods, values, &format!("{la}: shocks(learnt_in={learnt_in}) on '{a}'"))? {
                let (p, dates) = g.split();
                if li.1.is_none() && p.iter().any(|&q| (q as usize) < li.0) {
                    bail!(
                        "{la}: shocks(learnt_in={learnt_in}): the shock on '{a}' is set for a period before {learnt_in}"
                    );
                }
                // (the 7.2 preprocessor checks integer periods only)
                let (value, expr) = match v {
                    PathValue::Num(x) => (x, None),
                    PathValue::Steady(e) => (0.0, Some(e)),
                };
                self.ep.learnt_shocks.push(LearntShock {
                    learnt_in: li.0,
                    exo,
                    periods: p,
                    kind,
                    value,
                    expr,
                    pos: self.commands.len(),
                    learnt_in_date: li.1,
                    dates,
                });
            }
        }
    }

    /// Body of `endval(learnt_in=p); x = v; y += v; z *= v; end;` (exogenous variables).
    fn learnt_endval_block(&mut self, li: (usize, Option<crate::dates::Date>)) -> Result<()> {
        use crate::ep::decl::{LearntEndval, LearntKind};
        let learnt_in = shk::learnt_in_label(li);
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                return Ok(());
            }
            let (name, loc) = self.c.ident()?;
            let loc = loc.clone();
            let kind = if self.c.eat("+") {
                LearntKind::Add
            } else if self.c.eat("*") {
                LearntKind::Multiply
            } else {
                LearntKind::Level
            };
            self.c.expect("=")?;
            let toks = self.c.statement()?;
            let (value, expr) = match self.shock_value(toks)? {
                PathValue::Num(x) => (x, None),
                PathValue::Steady(e) => (0.0, Some(e)),
            };
            match self.b.symbol(name) {
                Some(s) if s.kind == SymbolKind::Exo => self.ep.learnt_endval.push(LearntEndval {
                    learnt_in: li.0,
                    exo: s.index,
                    kind,
                    value,
                    expr,
                    pos: self.commands.len(),
                    learnt_in_date: li.1,
                }),
                Some(s) if s.kind == SymbolKind::Endo => {
                    println!(
                        "WARNING: endval(learnt_in={learnt_in}): '{name}' is endogenous; the terminal condition is a steady state, its value is ignored"
                    );
                }
                _ => bail!("{loc}: '{name}' is not a variable"),
            }
        }
    }
    /// `perfect_foresight_controlled_paths[(learnt_in=p)]; exogenize y; periods ...; values ...;
    /// endogenize e; ... end;`
    fn controlled_paths_block(&mut self) -> Result<()> {
        use crate::ep::controlled::ControlledPath;
        self.c.i += 1;
        let opts = self.header_options()?;
        let li = shk::learnt_in_option(&opts, "perfect_foresight_controlled_paths")?;
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                return Ok(());
            }
            if !self.c.at_ident("exogenize") {
                let t = self.c.next()?;
                bail!("{}: perfect_foresight_controlled_paths: expected 'exogenize', found '{}'", t.loc, t.raw);
            }
            self.c.i += 1;
            let (y, ly) = self.c.ident()?;
            let ly = ly.clone();
            let exogenize = match self.b.symbol(y) {
                Some(s) if s.kind == SymbolKind::Endo => s.index,
                _ => bail!("{ly}: '{y}' is not an endogenous variable"),
            };
            self.c.expect(";")?;
            let mut periods: Vec<shk::PGroup> = Vec::new();
            let mut values: Vec<Vec<f64>> = Vec::new();
            let mut endogenize = None;
            while self.c.at_ident("periods") || self.c.at_ident("values") || self.c.at_ident("endogenize") {
                let kw = self.c.next()?;
                let (kwr, loc) = (kw.raw.clone(), kw.loc.clone());
                if kwr == "endogenize" {
                    let (e, le) = self.c.ident()?;
                    let le = le.clone();
                    endogenize = Some(self.exo_index(e, &le)?);
                    self.c.expect(";")?;
                    break;
                }
                let toks = self.c.statement()?;
                if kwr == "periods" {
                    periods = shk::sim_period_list(toks).map_err(|e| anyhow!("{loc}: {e}"))?;
                } else {
                    values = self.shocks_values(toks)?;
                }
            }
            let Some(endogenize) = endogenize else {
                bail!("{ly}: perfect_foresight_controlled_paths: missing 'endogenize' for '{y}'")
            };
            for (g, value) in
                self.shocks_pair(periods, values, &format!("{ly}: perfect_foresight_controlled_paths on '{y}'"))?
            {
                let (p, dates) = g.split();
                self.controlled_paths_pos.push(self.commands.len());
                self.ep.controlled_paths.push(ControlledPath {
                    learnt_in: li.0,
                    exogenize,
                    endogenize,
                    periods: p,
                    value,
                    learnt_in_date: li.1,
                    dates,
                });
            }
        }
    }
    // ---- end stream EP ----

    /// `histval; x(0) = 1; x(-1) = expr; end;` (stream D).
    /// `all_values_required` (the preprocessor's checkPass of Init/End/HistValStatement): every
    /// endogenous variable and every non-deterministic exogenous one must be set. `exo_label`
    /// is the block name of the exogenous message (the 7.2 histval check says "endval").
    fn check_all_values(&self, endo_label: &str, exo_label: &str, endo: &[usize], exo: &[usize]) -> Result<()> {
        let miss_endo: Vec<&str> = (0..self.b.endo_names.len())
            .filter(|j| !endo.contains(j) && !self.b.endo_names[*j].starts_with("AUX_"))
            .map(|j| self.b.endo_names[j].as_str())
            .collect();
        let miss_exo: Vec<&str> = (0..self.b.exo_names.len())
            .filter(|j| !exo.contains(j) && !self.exo_det.contains(j))
            .map(|j| self.b.exo_names[j].as_str())
            .collect();
        let mut msg = Vec::new();
        if !miss_endo.is_empty() {
            msg.push(format!(
                "You have not set the following endogenous variables in {endo_label}: {}",
                miss_endo.join(" ")
            ));
        }
        if !miss_exo.is_empty() {
            msg.push(format!(
                "You have not set the following exogenous variables in {exo_label}: {}",
                miss_exo.join(" ")
            ));
        }
        if msg.is_empty() { Ok(()) } else { bail!("{}", msg.join("\nERROR: ")) }
    }

    fn histval_block(&mut self) -> Result<()> {
        self.c.i += 1;
        let hopts = self.header_options()?;
        let avr = opt(&hopts, "all_values_required").is_some();
        // HistValStatement: each block rebuilds M_.endo_histval / M_.exo_histval from zeros
        self.histval.clear();
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                if avr {
                    let endo: Vec<usize> = self.histval.iter().filter(|h| h.0).map(|h| h.1).collect();
                    let exo: Vec<usize> = self.histval.iter().filter(|h| !h.0).map(|h| h.1).collect();
                    self.check_all_values("histval", "endval", &endo, &exo)?;
                }
                self.state_blocks.push((self.commands.len(), RawStateBlock::Histval(self.histval.clone())));
                return Ok(());
            }
            let (name, loc) = self.c.ident()?;
            let loc = loc.clone();
            let mut lag = 0i32;
            if self.c.eat("(") {
                let neg = self.c.eat("-");
                if !neg {
                    self.c.eat("+");
                }
                let t = self.c.next()?;
                lag = match t.tok {
                    Tok::Num(v) if v.fract() == 0.0 => v as i32 * if neg { -1 } else { 1 },
                    _ => bail!("{}: histval: expected an integer lag", t.loc),
                };
                self.c.expect(")")?;
            }
            if lag > 0 {
                bail!("{loc}: histval: '{name}({lag})': only lags (<= 0) can be set");
            }
            self.c.expect("=")?;
            let toks = self.c.statement()?;
            let v = self.eval_tokens(toks)?;
            match self.b.symbol(name) {
                Some(s) if matches!(s.kind, SymbolKind::Endo | SymbolKind::Exo) => {
                    let endo = s.kind == SymbolKind::Endo;
                    // ParsingDriver::hist_val (Dynare 7.2 wording; 7.1: `hist_val: (NAME, LAG) declared twice`)
                    if self.histval.iter().any(|&(e, j, l, _)| e == endo && j == s.index && l == lag) {
                        bail!("{loc}: histval: {name}({lag}) declared twice");
                    }
                    self.histval.push((endo, s.index, lag, v));
                }
                _ => bail!("{loc}: histval: '{name}' is not a variable"),
            }
        }
    }

    /// `steady_state_model; x = expr; ... end;` (stream D): kept as source, compiled against
    /// the built model (`<fname>.steadystate`).
    /// `verbatim; ... end;` (the lexer's `verbatim`, source string, `;`): runs the MATLAB code
    /// with the parameters assigned so far as variables. Numeric scalars it defines become
    /// usable in later parameter assignments (as in Dynare's driver). Code outside the
    /// interpreted subset only warns: its variables stay undefined.
    fn verbatim_block(&mut self) -> Result<()> {
        let loc = self.c.next()?.loc.clone();
        let src = match self.c.next()? {
            Token { tok: Tok::Str(s), .. } => s.clone(),
            t => bail!("{}: verbatim: unexpected '{}'", t.loc, t.raw),
        };
        self.c.expect(";")?;
        for (k, n) in self.b.param_names.iter().enumerate() {
            if !self.b.params[k].is_nan() {
                self.verbatim_ws.set(n, super::verbatim::V::scalar(self.b.params[k]));
            }
        }
        if let Err(e) = self.verbatim_ws.run(&src) {
            eprintln!(
                "WARNING: {loc}: verbatim block not run by dynare-rust ({e:#}); the variables it defines are unavailable"
            );
        }
        for n in &self.verbatim_ws.order {
            if self.b.symbol(n).is_none()
                && let Some(super::verbatim::V::Num { data, .. }) = self.verbatim_ws.get(n)
                && data.len() == 1
            {
                self.mf_locals.insert(n.clone(), data[0]);
            }
        }
        Ok(())
    }

    fn steady_state_model_block(&mut self) -> Result<()> {
        self.c.i += 1;
        self.header_options()?;
        let mut ss_locals = std::collections::HashMap::new();
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                return Ok(());
            }
            if self.c.at("[") {
                bail!(
                    "{}: steady_state_model: multiple-output assignments ([a, b] = f(...)) call external MATLAB functions, which dynare-rust cannot execute; inline the function body as one assignment per output (a = ...; b = ...;)",
                    self.c.loc()
                );
            }
            let (name, loc) = self.c.ident()?;
            let name = name.to_string();
            self.c.expect("=")?;
            let toks = self.c.statement()?;
            slice_expr(toks)?; // syntax check now; symbols are resolved against the model
            // undeclared names are temporary variables of the block (the
            // preprocessor's `SteadyStateModel` local variables); they are inlined.
            let src = q1::inline_ss_locals(toks, &ss_locals);
            match self.b.symbol(&name) {
                Some(s) if matches!(s.kind, SymbolKind::Endo | SymbolKind::Param) => {
                    self.steady_state_model.push((name, src))
                }
                Some(_) => bail!("{loc}: steady_state_model: '{name}' is not an endogenous variable or a parameter"),
                None => {
                    ss_locals.insert(name, src);
                }
            }
        }
    }

    /// An exogenous variable (OccBin surprise shocks).
    fn exo_index(&self, name: &str, loc: &Loc) -> Result<usize> {
        match self.b.symbol(name) {
            Some(s) if s.kind == SymbolKind::Exo => Ok(s.index),
            _ => bail!("{loc}: '{name}' is not an exogenous variable"),
        }
    }

    /// An exogenous shock, or an endogenous variable (measurement error).
    fn shock_target(&self, name: &str, loc: &Loc) -> Result<Tgt> {
        match self.b.symbol(name) {
            Some(s) if s.kind == SymbolKind::Exo => Ok(Tgt::Exo(s.index)),
            Some(s) if s.kind == SymbolKind::Endo => Ok(Tgt::Endo(s.index)),
            _ => bail!("{loc}: '{name}' is not an exogenous or endogenous variable"),
        }
    }

    fn shocks_block(&mut self) -> Result<()> {
        self.shocks_block_kind(false)
    }

    /// The parameters of a variance / covariance / correlation expression of a `shocks` block
    /// (`ShocksStatement::checkPass`: `parameters_within_shocks_values`).
    fn note_shock_params(&mut self, toks: &[Token]) {
        for n in toks.iter().filter_map(|t| t.ident()) {
            if let Some(s) = self.b.symbol(n)
                && s.kind == SymbolKind::Param
                && !self.pp.shock_params.contains(&s.index)
            {
                self.pp.shock_params.push(s.index);
            }
        }
    }

    /// `shocks` (or `mshocks` when `multiplicative`).
    fn shocks_block_kind(&mut self, multiplicative: bool) -> Result<()> {
        self.c.i += 1;
        let opts = self.shocks_header_options()?; // ---- shocks/shock_paths: space-separated options
        if opts.iter().any(|(k, _)| k == "surprise") {
            return self.surprise_shocks_block(opts.iter().any(|(k, _)| k == "overwrite"));
        }
        // ---- heterogeneity ----
        if let Some(v) = opt(&opts, "heterogeneity") {
            let dim = match v {
                OptValue::Str(s) => s.clone(),
                _ => String::new(),
            };
            let overwrite = opts.iter().any(|(k, _)| k == "overwrite");
            return self.het_shocks_block(&dim, overwrite);
        }
        // ---- end heterogeneity ----
        // ---- stream EP: shocks(learnt_in=p) / mshocks(learnt_in=p), p >= 2 (p = 1: a regular
        // block); a date-valued learnt_in is always learnt-in information
        let li = shk::learnt_in_option(&opts, if multiplicative { "mshocks" } else { "shocks" })?;
        if li.1.is_some() || li.0 > 1 {
            let flag = |name: &str| opts.iter().any(|(k, _)| k == name);
            return self.learnt_shocks_block(li, multiplicative, flag("overwrite"), flag("relative_to_initval"));
        }
        // ---- shocks/shock_paths: shocks(overwrite) resets det_shocks, Sigma_e, H (not mshocks')
        let overwrite = opts.iter().any(|(k, _)| k == "overwrite");
        if multiplicative && overwrite {
            self.det_shocks.clear();
        }
        self.shocks_block_start(overwrite && !multiplicative);
        let (st, det_st) = (self.shocks.len(), self.det_shocks.len());
        let kind = if !multiplicative {
            DetShockKind::Level
        } else if opts.iter().any(|(k, _)| k == "relative_to_initval") {
            DetShockKind::MultiplyInitialSteadyState
        } else {
            DetShockKind::MultiplySteadyState
        };
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                let block = RawStateBlock::Shocks {
                    overwrite: overwrite && !multiplicative,
                    stmts: self.shocks[st..].to_vec(),
                    reset_det: overwrite,
                    det: self.det_shocks[det_st..].to_vec(),
                };
                self.state_blocks.push((self.commands.len(), block));
                self.shocks_block_end(); // ---- shocks/shock_paths
                return Ok(());
            }
            if self.c.at_ident("var") {
                self.c.i += 1;
                let (a, la) = self.c.ident()?;
                let aname = a;
                let a = self.shock_target(a, la)?;
                let b = if self.c.eat(",") {
                    let (b, lb) = self.c.ident()?;
                    Some(self.shock_target(b, lb)?)
                } else {
                    None
                };
                if b == Some(a) {
                    bail!(
                        "{la}: shocks: covariance of '{aname}' with itself (use `var {aname} = ...;` for its variance)"
                    );
                }
                if self.c.eat("=") {
                    self.q1_check_exo_det_moment(if b.is_some() { "covariance" } else { "variance" }, a, b)?;
                    let toks = self.c.statement()?;
                    self.note_shock_params(toks);
                    let v = self.shock_value(toks)?;
                    self.shocks.push(match b {
                        None => ShockStmt::Var(a, v),
                        Some(b) => ShockStmt::Cov(a, b, v),
                    });
                    continue;
                }
                self.c.expect(";")?;
                if self.c.at_ident("stderr") {
                    self.c.i += 1;
                    self.q1_check_exo_det_moment("standard error", a, None)?;
                    let toks = self.c.statement()?;
                    self.note_shock_params(toks);
                    let v = self.shock_value(toks)?;
                    self.shocks.push(ShockStmt::Std(a, v));
                } else {
                    // deterministic shocks: `periods ...; values ...;`
                    let Tgt::Exo(a) = a else {
                        bail!("{la}: deterministic shocks (periods/values) must be on an exogenous variable")
                    };
                    let mut periods: Vec<shk::PGroup> = Vec::new();
                    let mut values: Vec<Vec<PathValue>> = Vec::new();
                    while self.c.at_ident("periods") || self.c.at_ident("values") {
                        let is_periods = self.c.at_ident("periods");
                        let loc = self.c.next()?.loc.clone();
                        let toks = self.c.statement()?;
                        if is_periods {
                            periods = shk::sim_period_list(toks).map_err(|e| anyhow!("{loc}: {e}"))?;
                        } else {
                            values = self.shocks_values_x(toks)?;
                        }
                    }
                    let what = format!("{la}: shocks on '{}'", self.b.exo_names[a]);
                    for (g, v) in self.shocks_pair(periods, values, &what)? {
                        let (p, dates) = g.split();
                        let (value, expr) = match v {
                            PathValue::Num(x) => (x, None),
                            PathValue::Steady(e) => (0.0, Some(e)),
                        };
                        self.det_shocks.push(DetShock { exo: a, periods: p, value, expr, kind, dates });
                    }
                }
            } else if self.c.at_ident("corr") {
                self.c.i += 1;
                let (a, la) = self.c.ident()?;
                let a = self.shock_target(a, la)?;
                self.c.expect(",")?;
                let (b, lb) = self.c.ident()?;
                let bname = b.to_string();
                let b = self.shock_target(b, lb)?;
                if a == b {
                    bail!("{la}: shocks: correlation of '{bname}' with itself");
                }
                self.q1_check_exo_det_moment("correlation", a, Some(b))?;
                self.c.expect("=")?;
                let toks = self.c.statement()?;
                self.note_shock_params(toks);
                let v = self.shock_value(toks)?;
                self.shocks.push(ShockStmt::Corr(a, b, v));
            } else if self.c.at_ident("skew") {
                self.c.i += 1;
                self.q3_shocks_skew()?;
            } else {
                let t = self.c.next()?;
                bail!("{}: syntax error, unexpected '{}', expecting corr or skew or end or var", t.loc, t.raw);
            }
        }
    }

    /// Parses the `stderr x` / `corr x, y` / `name` head of an estimation entry; returns the
    /// kind and the remaining fields.
    fn estimated_head<'f>(&self, fields: &'f [&'a [Token]]) -> Result<(EstimatedKind, &'f [&'a [Token]])> {
        let f0 = fields[0];
        let Some(first) = f0.first() else { bail!("{}: empty estimated_params entry", self.c.loc()) };
        let single = |f: &[Token]| -> Result<(String, Loc)> {
            match f {
                [t] if t.ident().is_some() => Ok((t.raw.clone(), t.loc.clone())),
                _ => bail!("{}: expected a name", f.first().map(|t| t.loc.to_string()).unwrap_or_default()),
            }
        };
        // EstimatedParamsStatement::checkPass on a varexo_det variable
        let not_det = |n: &str, l: &Loc| -> Result<()> {
            match self.b.symbol(n) {
                Some(s) if s.kind == SymbolKind::Exo && self.exo_det.contains(&s.index) => {
                    bail!("{l}: {n} must be an endogenous or an exogenous variable")
                }
                _ => Ok(()),
            }
        };
        if (first.is_ident("stderr") || first.is_ident("corr")) && f0.len() == 2 {
            let (n, l) = single(&f0[1..])?;
            not_det(&n, &l)?;
            if first.is_ident("corr")
                && let Some(f1) = fields.get(1)
                && let Ok((n, l)) = single(f1)
            {
                not_det(&n, &l)?;
            }
        }
        if first.is_ident("stderr") && f0.len() == 2 {
            let (n, l) = single(&f0[1..])?;
            let kind = match self.shock_target(&n, &l)? {
                Tgt::Exo(k) => EstimatedKind::StdErr(k),
                Tgt::Endo(k) => EstimatedKind::MeStdErr(k),
            };
            return Ok((kind, &fields[1..]));
        }
        // `skew e`
        if first.is_ident("skew") && f0.len() == 2 {
            let (n, l) = single(&f0[1..])?;
            return match self.shock_target(&n, &l)? {
                Tgt::Exo(k) => Ok((EstimatedKind::Skew(k), &fields[1..])),
                Tgt::Endo(_) => bail!("{l}: skew: '{n}' is not an exogenous variable"),
            };
        }
        if first.is_ident("corr") && f0.len() == 2 {
            let (a, la) = single(&f0[1..])?;
            let (b, lb) = single(fields.get(1).ok_or_else(|| anyhow!("{}: corr needs two names", first.loc))?)?;
            let kind = match (self.shock_target(&a, &la)?, self.shock_target(&b, &lb)?) {
                (Tgt::Exo(i), Tgt::Exo(j)) => EstimatedKind::Corr(i, j),
                (Tgt::Endo(i), Tgt::Endo(j)) => EstimatedKind::MeCorr(i, j),
                _ => bail!("{la}: corr between an exogenous shock and a measurement error"),
            };
            if matches!(kind, EstimatedKind::Corr(i, j) | EstimatedKind::MeCorr(i, j) if i == j) {
                bail!("{la}: estimated_params: correlation of '{a}' with itself");
            }
            return Ok((kind, &fields[2..]));
        }
        let (n, l) = single(f0)?;
        match self.b.symbol(&n) {
            Some(s) if s.kind == SymbolKind::Param => Ok((EstimatedKind::Param(s.index), &fields[1..])),
            _ => bail!("{l}: '{n}' is not a parameter (dsge_prior_weight and other forms are not supported)"),
        }
    }

    /// Records `estim_params_` as it stands after an estimated_params(_bounds/_remove) block.
    pub(super) fn est_snapshot(&mut self) {
        let n = self.commands.len();
        match self.est_history.last_mut() {
            Some((k, e)) if *k == n => *e = self.est.clone(),
            _ => self.est_history.push((n, self.est.clone())),
        }
    }

    fn estimated_params(&mut self) -> Result<()> {
        self.c.i += 1;
        if self.header_options()?.iter().any(|(k, _)| k == "overwrite") {
            self.est.clear(); // [stream X] estimated_params(overwrite)
        }
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                self.est_snapshot();
                return Ok(());
            }
            let toks = self.c.statement()?;
            let loc = toks.first().map(|t| t.loc.clone());
            let fields = split_commas(toks);
            if matches!(fields.first(), Some([t]) if t.is_ident("dsge_prior_weight")) {
                self.q3_dsge_prior_weight()?;
            }
            let (kind, rest) = self.estimated_head(&fields)?;
            let prior_pos =
                rest.iter().position(|f| matches!(f, [t] if t.ident().and_then(PriorShape::from_keyword).is_some()));
            let mut e = EstimatedParam {
                kind,
                init: None,
                lb: None,
                ub: None,
                prior: None,
                mean: f64::NAN,
                sd: f64::NAN,
                p3: None,
                p4: None,
                jscale: None,
            };
            let (before, after): (&[&[Token]], &[&[Token]]) = match prior_pos {
                Some(k) => {
                    e.prior = rest[k][0].ident().and_then(PriorShape::from_keyword);
                    (&rest[..k], &rest[k + 1..])
                }
                None => (rest, &[]),
            };
            match before.len() {
                0 => {}
                1 => e.init = self.eval_opt(before[0])?,
                3 => {
                    e.init = self.eval_opt(before[0])?;
                    e.lb = self.eval_opt(before[1])?;
                    e.ub = self.eval_opt(before[2])?;
                }
                n => bail!(
                    "{}: estimated_params entry has {n} fields before the prior (expected INIT or INIT, LB, UB)",
                    loc.map(|l| l.to_string()).unwrap_or_default()
                ),
            }
            if after.len() > 5 || (e.prior.is_some() && after.len() < 2) {
                bail!(
                    "{}: estimated_params entry needs MEAN, SD [, P3, P4, JSCALE] after the prior shape",
                    loc.map(|l| l.to_string()).unwrap_or_default()
                );
            }
            let vals: Vec<Option<f64>> = after.iter().map(|f| self.eval_opt(f)).collect::<Result<_>>()?;
            let get = |k: usize| vals.get(k).copied().flatten();
            e.mean = get(0).unwrap_or(f64::NAN);
            e.sd = get(1).unwrap_or(f64::NAN);
            e.p3 = get(2);
            e.p4 = get(3);
            e.jscale = get(4);
            // [stream X] as the preprocessor's generated check for concatenated blocks
            if self.est.iter().any(|o| o.kind == e.kind) {
                bail!(
                    "{}: this parameter has been specified twice in two concatenated 'estimated_params' blocks. Depending on your intention, you may want to use the 'overwrite' option or an 'estimated_params_remove' block.",
                    loc.map(|l| l.to_string()).unwrap_or_default()
                );
            }
            self.est.push(e);
        }
    }

    fn estimated_params_init(&mut self) -> Result<()> {
        self.c.i += 1;
        let opts = self.header_options()?;
        if opts.iter().any(|(k, _)| k == "use_calibration") {
            self.use_calibration = true;
        }
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                return Ok(());
            }
            let toks = self.c.statement()?;
            let fields = split_commas(toks);
            let (kind, rest) = self.estimated_head(&fields)?;
            let [v] = rest else { bail!("{}: expected NAME, VALUE", toks[0].loc) };
            let value = self.eval_tokens(v)?;
            self.est_init.push(EstimatedParamInit { kind, value });
        }
    }

    /// Generic `name(options) varlist;` command.
    /// The option checks of the preprocessor's parser actions and `checkPass` of the
    /// `extended_path` and `estimation` statements, with its error texts.
    fn statement_checks(&self, name: &str, toks: &[Token], opts: &Options) -> Result<()> {
        let has = |k: &str| opts.iter().any(|(o, _)| o == k);
        let word = |k: &str| match opt(opts, k) {
            Some(OptValue::Str(s)) => s.to_lowercase(),
            _ => String::new(),
        };
        match name {
            "extended_path" => {
                if let Some(OptValue::Num(x)) = opt(opts, "hybrid")
                    && x % 2.0 != 0.0
                {
                    let k = toks.iter().position(|t| t.is_ident("hybrid")).unwrap_or(0);
                    let end = toks.get(k + 2).unwrap_or(&toks[k]);
                    bail!(
                        "{}: The 'hybrid' option in stochastic extended path must be an even integer",
                        span_text(&toks[k], end)
                    );
                }
            }
            "estimation" => {
                if has("mh_tune_jscale") && has("mh_jscale") {
                    bail!("The mh_tune_jscale and mh_jscale options of the estimation statement are incompatible.");
                }
                if !has("mh_tune_jscale") && has("mh_tune_guess") {
                    bail!(
                        "The option mh_tune_guess in estimation statement cannot be used without option mh_tune_jscale."
                    );
                }
                if word("filter_algorithm") == "gmf" {
                    if word("proposal_approximation") == "montecarlo" {
                        bail!(
                            "The filter_algorithm=gmf option is incompatible with proposal_approximation=montecarlo in the estimation statement."
                        );
                    }
                    if word("distribution_approximation") == "montecarlo" {
                        bail!(
                            "The filter_algorithm=gmf option is incompatible with distribution_approximation=montecarlo in the estimation statement."
                        );
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn command(&mut self, name: &'a str) -> Result<()> {
        let loc = self.c.loc();
        let first = self.c.i;
        self.c.i += 1;
        let toks = self.c.statement()?;
        match name {
            "simul" => self.pp.warn.push(format!(
                "WARNING: {}: The 'simul' statement is deprecated. Please use 'perfect_foresight_setup' and 'perfect_foresight_solver' instead.",
                span_text(&self.c.t[first], &self.c.t[self.c.i - 1])
            )),
            "ramsey_policy" => self.pp.warn.push(format!(
                "WARNING: {}: The 'ramsey_policy' statement is deprecated. Please use 'ramsey_model', 'stoch_simul', and 'evaluate_planner_objective' instead.",
                span_text(&self.c.t[first], &self.c.t[self.c.i - 1])
            )),
            "osr_params" => self.pp.osr_params_stmts += 1,
            _ => {}
        }
        if matches!(
            name,
            "stoch_simul"
                | "estimation"
                | "osr"
                | "identification"
                | "calib_smoother"
                | "method_of_moments"
                | "discretionary_policy"
                | "ramsey_policy"
                | "extended_path"
        ) {
            self.pp.stoch = true;
        }
        if toks.first().is_some_and(|t| t.is("(")) {
            for t in toks.iter().filter(|t| t.is_ident("aim_solver")) {
                self.pp.warn.push(format!(
                    "WARNING: {}: The 'aim_solver' option is deprecated. It has been superseded by the 'dr=aim' option.",
                    span_text(t, t)
                ));
            }
        }
        let (opts, rest) = if toks.first().is_some_and(|t| t.is("(")) {
            let mut depth = 0;
            let mut close = toks.len();
            for (k, t) in toks.iter().enumerate() {
                if t.is("(") {
                    depth += 1;
                } else if t.is(")") {
                    depth -= 1;
                    if depth == 0 {
                        close = k;
                        break;
                    }
                }
            }
            if name == "histval_file" {
                // Dynare 7.2 grammar: `histval_file_option` no longer has `nobs` and
                // `last_simulation_period` (DynareBison.yy), so they are syntax errors.
                let inner = &toks[1..close.min(toks.len())];
                for (k, t) in inner.iter().enumerate() {
                    if let Some(id) = t.ident()
                        && matches!(id, "nobs" | "last_simulation_period")
                        && inner.get(k + 1).is_some_and(|n| n.is("="))
                    {
                        bail!("{}: syntax error, unexpected {}", t.loc, id.to_uppercase());
                    }
                }
            }
            (self.parse_options(&toks[1..close.min(toks.len())]), toks.get(close + 1..).unwrap_or(&[]))
        } else {
            (Vec::new(), toks)
        };
        let vars: Vec<String> = rest.iter().filter_map(|t| t.ident().map(str::to_string)).collect();
        // ---- stream B: commands whose arguments are not a variable list
        // (`bvar_density 4;`, `model_comparison m1(0.7) m2(0.3);`) keep their raw tokens.
        let mut opts = opts;
        self.x_global_options(name, &mut opts); // [stream X] options_.X = V defaults
        super::option_check::check_options(name, &opts).map_err(|e| anyhow!("{loc}: {name}: {e}"))?;
        self.statement_checks(name, toks, &opts)?;
        if RAW_ARG_COMMANDS.contains(&name) {
            opts.push((
                RAW_ARGS_KEY.to_string(),
                OptValue::List(rest.iter().map(|t| OptValue::Str(t.raw.clone())).collect()),
            ));
        }
        if STOCHASTIC_COMMANDS.contains(&name) {
            self.stochastic = true;
        }
        let cmd = match name {
            "steady" => Command::Steady(opts),
            "check" => Command::Check(opts),
            "stoch_simul" => Command::StochSimul(opts, vars),
            "estimation" => Command::Estimation(opts, vars),
            // [stream P] optimal policy
            "ramsey_model"
            | "ramsey_policy"
            | "discretionary_policy"
            | "osr"
            | "osr_params"
            | "evaluate_planner_objective" => self.policy_command(name, opts, vars)?,
            // ---- stream F ----
            "forecast" => Command::Forecast(opts, vars),
            "shock_decomposition" => Command::ShockDecomposition(opts, vars),
            "calib_smoother" => Command::CalibSmoother(opts, vars),
            "shock_groups" => {
                let name = match opt(&opts, "name") {
                    Some(OptValue::Str(s)) => s.clone(),
                    _ => "default".to_string(),
                };
                let mut groups = Vec::new();
                while !self.c.at_end() {
                    if self.c.peek().is_none() {
                        bail!("{}: block 'shock_groups' not closed by 'end;'", self.c.loc());
                    }
                    let st = self.c.statement()?;
                    let Some(eq) = st.iter().position(|t| t.is("=")) else {
                        bail!(
                            "{}: shock_groups: expected 'label' = shocks",
                            st.first().map(|t| t.loc.to_string()).unwrap_or_default()
                        )
                    };
                    let label: String = st[..eq]
                        .iter()
                        .map(|t| match &t.tok {
                            Tok::Str(s) | Tok::Ident(s) => s.clone(),
                            _ => t.raw.clone(),
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    let shocks: Vec<String> =
                        st[eq + 1..].iter().filter_map(|t| t.ident().map(str::to_string)).collect();
                    groups.push((label, shocks));
                }
                self.c.i += 2;
                Command::ShockGroups(name, groups)
            }
            // ---- end stream F ----
            // ---- stream C ----
            "conditional_forecast_paths" => Command::ConditionalForecastPaths(self.conditional_forecast_paths()?),
            "conditional_forecast" => Command::ConditionalForecast(opts),
            "plot_conditional_forecast" => Command::PlotConditionalForecast(opts, vars),
            "realtime_shock_decomposition" => Command::RealtimeShockDecomposition(opts, vars),
            "plot_shock_decomposition" => Command::PlotShockDecomposition(opts, vars),
            "initial_condition_decomposition" => Command::InitialConditionDecomposition(opts, vars),
            "squeeze_shock_decomposition" => Command::SqueezeShockDecomposition(vars),
            // ---- end stream C ----
            "method_of_moments" => Command::Mom(crate::mom::MomCommand::MethodOfMoments(opts)), // MoM
            _ if crate::occbin::is_occbin_command(name) => {
                Command::Occbin(crate::occbin::OccbinCommand::new(name, opts, vars))
            }
            "perfect_foresight_setup" | "perfect_foresight_solver" | "simul" | "rplot" | "model_diagnostics" => {
                Command::Deterministic(name.to_string(), opts, vars)
            }
            _ => {
                let mut opts = opts;
                if OTHER_BLOCKS.contains(&name) {
                    while !self.c.at_end() {
                        if self.c.peek().is_none() {
                            bail!("{}: block '{name}' not closed by 'end;'", self.c.loc());
                        }
                        let st = self.c.statement()?;
                        // ---- the entries of irf_calibration / moment_calibration ----
                        if matches!(name, "irf_calibration" | "moment_calibration" | "svar_identification")
                            && !st.is_empty()
                        {
                            let txt: Vec<&str> = st.iter().map(|t| t.raw.as_str()).collect();
                            opts.push((super::BLOCK_BODY_KEY.to_string(), OptValue::Str(txt.join(" "))));
                        }
                    }
                    self.c.i += 2;
                }
                Command::Other(name.to_string(), opts)
            }
        };
        self.commands.push(cmd);
        Ok(())
    }

    // ---- stream C ----
    /// Body of `conditional_forecast_paths; var v; periods 1:2, 3; values a, b; ... end;`
    /// (the preprocessor's ConditionalForecastPathsStatement): one value expression per period
    /// group, groups and values separated by commas or blanks.
    #[allow(clippy::type_complexity)]
    fn conditional_forecast_paths(&mut self) -> Result<Vec<(String, Vec<(usize, usize, PathValue)>)>> {
        let mut out: Vec<(String, Vec<(usize, usize, PathValue)>)> = Vec::new();
        let mut periods: Vec<(usize, usize)> = Vec::new();
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                break;
            }
            if self.c.peek().is_none() {
                bail!("{}: block 'conditional_forecast_paths' not closed by 'end;'", self.c.loc());
            }
            let st = self.c.statement()?;
            let Some(first) = st.first() else { continue };
            match first.ident() {
                Some("var") => {
                    let [_, v] = st else { bail!("{}: conditional_forecast_paths: expected 'var NAME;'", first.loc) };
                    let name = v.ident().ok_or_else(|| anyhow!("{}: expected a variable name", v.loc))?;
                    if self.b.symbol(name).is_none_or(|s| s.kind != SymbolKind::Endo) {
                        bail!("{}: conditional_forecast_paths: '{name}' is not an endogenous variable", v.loc);
                    }
                    if out.iter().any(|(n, _)| n == name) {
                        bail!("{}: conditional_forecast_paths: '{name}' is constrained twice", v.loc);
                    }
                    out.push((name.to_string(), Vec::new()));
                    periods.clear();
                }
                Some("periods") => {
                    periods.clear();
                    let mut k = 1;
                    let num = |t: &Token| -> Result<usize> {
                        match t.tok {
                            Tok::Num(x) if x >= 1.0 && x.fract() == 0.0 => Ok(x as usize),
                            _ => bail!("{}: periods: expected a positive integer, found '{}'", t.loc, t.raw),
                        }
                    };
                    while k < st.len() {
                        if st[k].is(",") {
                            k += 1;
                            continue;
                        }
                        let a = num(&st[k])?;
                        let mut b = a;
                        if st.get(k + 1).is_some_and(|t| t.is(":")) {
                            b = num(st
                                .get(k + 2)
                                .ok_or_else(|| anyhow!("{}: periods: incomplete range", st[k].loc))?)?;
                            k += 2;
                        }
                        if b < a {
                            bail!("{}: periods: empty range {a}:{b}", st[k].loc);
                        }
                        periods.push((a, b));
                        k += 1;
                    }
                }
                Some("values") => {
                    let Some((_, ranges)) = out.last_mut() else {
                        bail!("{}: conditional_forecast_paths: 'values' before 'var'", first.loc)
                    };
                    // signed numbers and parenthesized expressions (`values 0.01 -0.02 0;` has
                    // three values, as in Dynare's grammar); `(xx)` a workspace vector
                    // an endogenous variable in a value is its steady state (`gy_obs+0.01`)
                    let vals = self.shocks_values_x(&st[1..])?;
                    if vals.len() != periods.len() {
                        bail!(
                            "{}: conditional_forecast_paths: {} period group(s) but {} value(s)",
                            first.loc,
                            periods.len(),
                            vals.len()
                        );
                    }
                    for (&(a, b), v) in periods.iter().zip(&vals) {
                        match v.as_slice() {
                            [v] => ranges.push((a, b, v.clone())),
                            // constrained_paths_(i, a:b) = xx
                            v if v.len() == b - a + 1 => ranges.extend((a..=b).zip(v).map(|(t, x)| (t, t, x.clone()))),
                            v => bail!(
                                "{}: conditional_forecast_paths: a vector of {} values for the {} periods {a}:{b}",
                                first.loc,
                                v.len(),
                                b - a + 1
                            ),
                        }
                    }
                }
                _ => bail!("{}: conditional_forecast_paths: expected var, periods or values", first.loc),
            }
        }
        if out.is_empty() {
            bail!("{}: empty conditional_forecast_paths block", self.c.loc());
        }
        Ok(out)
    }
    // ---- end stream C ----

    fn parse_options(&self, toks: &[Token]) -> Options {
        let mut out = Vec::new();
        if toks.is_empty() {
            return out;
        }
        for item in split_commas(toks) {
            match item {
                [] => {}
                [k] if k.ident().is_some() => out.push((k.raw.to_lowercase(), OptValue::Flag)),
                [k, eq, v @ ..] if k.ident().is_some() && eq.is("=") => {
                    out.push((k.raw.to_lowercase(), self.parse_value(v)))
                }
                _ => out.push((raw_join(item), OptValue::Flag)),
            }
        }
        out
    }

    fn parse_value(&self, v: &[Token]) -> OptValue {
        match v {
            [] => return OptValue::Str(String::new()),
            [t] => match &t.tok {
                Tok::Num(x) => return OptValue::Num(*x),
                Tok::Str(s) | Tok::Ident(s) => return OptValue::Str(s.clone()),
                _ => {}
            },
            [m, t] if m.is("-") => {
                if let Tok::Num(x) = t.tok {
                    return OptValue::Num(-x);
                }
            }
            _ => {}
        }
        let wrapped = |o: &str, c: &str| v.len() >= 2 && v[0].is(o) && v[v.len() - 1].is(c);
        if wrapped("(", ")") {
            let inner = &v[1..v.len() - 1];
            let items = split_commas(inner);
            let list = if items.len() == 1 && inner.len() > 1 && !inner.iter().any(|t| t.is("(") || t.is("[")) {
                inner.iter().map(|t| self.parse_value(std::slice::from_ref(t))).collect()
            } else {
                items.into_iter().filter(|i| !i.is_empty()).map(|i| self.parse_value(i)).collect()
            };
            return OptValue::List(list);
        }
        if wrapped("[", "]") {
            if let Some(x) = numeric_vector(&v[1..v.len() - 1]) {
                return OptValue::Vec(x);
            }
            let items = split_commas(&v[1..v.len() - 1]);
            return OptValue::List(items.into_iter().filter(|i| !i.is_empty()).map(|i| self.parse_value(i)).collect());
        }
        match self.eval_tokens(v) {
            Ok(x) => OptValue::Num(x),
            Err(_) => OptValue::Str(raw_join(v)),
        }
    }
}

// ---- OccBin (stream: occbin) -----------------------------------------------------------
impl<'a> Parser<'a> {
    /// `occbin_constraints; name 'C'; bind EXPR; [relax EXPR;] [error_bind EXPR;]
    /// [error_relax EXPR;] ... end;`
    fn occbin_constraints_block(&mut self) -> Result<()> {
        self.c.i += 1;
        self.header_options()?;
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                return Ok(());
            }
            let toks = self.c.statement()?;
            let Some((kw, rest)) = toks.split_first() else { continue };
            let key = kw.ident().unwrap_or("");
            if key == "name" {
                let name = match rest {
                    [t] => match &t.tok {
                        Tok::Str(s) | Tok::Ident(s) => s.clone(),
                        _ => bail!("{}: expected a constraint name", t.loc),
                    },
                    _ => bail!("{}: expected name 'STRING';", kw.loc),
                };
                self.occbin_constraints.push(crate::occbin::OccbinConstraint::new(name));
                continue;
            }
            let Some(c) = self.occbin_constraints.last_mut() else {
                bail!("{}: occbin_constraints: '{key}' before any 'name'", kw.loc)
            };
            // the preprocessor's OccbinConstraintsDataTree: an exogenous (or undeclared, which
            // it declares exogenous) variable, or a variable with a lead or lag, is an error
            for (k, t) in rest.iter().enumerate() {
                let Some(n) = t.ident() else { continue };
                let sym = self.b.symbol(n).map(|s| s.kind);
                if rest.get(k + 1).is_some_and(|x| x.is("(")) {
                    if matches!(sym, Some(SymbolKind::Endo | SymbolKind::Exo)) {
                        let close = rest[k..].iter().position(|x| x.is(")")).map_or(k, |j| k + j);
                        bail!(
                            "{}: Leads and lags on variables are forbidden in 'occbin_constraints'. Note that you can achieve the same effect by introducing an auxiliary variable in the model.",
                            span_text(t, &rest[close])
                        );
                    }
                } else if matches!(sym, Some(SymbolKind::Exo)) || (sym.is_none() && !self.mf_locals.contains_key(n)) {
                    bail!("{}: Exogenous variable {n} cannot be used in 'occbin_constraints'.", span_text(t, t));
                }
            }
            let e = slice_expr(rest)?;
            match key {
                "bind" => c.bind = Some(e),
                "relax" => c.relax = Some(e),
                "error_bind" => c.error_bind = Some(e),
                "error_relax" => c.error_relax = Some(e),
                _ => bail!("{}: occbin_constraints: unexpected '{}'", kw.loc, kw.raw),
            }
        }
    }

    /// `shocks(surprise[,overwrite]); var e; periods 1:9, 10; values -0.01, (2*x); ... end;`
    fn surprise_shocks_block(&mut self, overwrite: bool) -> Result<()> {
        let mut shocks = Vec::new();
        let mut cur: Option<usize> = None;
        let mut periods: Vec<Vec<usize>> = Vec::new();
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                break;
            }
            let toks = self.c.statement()?;
            let Some((kw, rest)) = toks.split_first() else { continue };
            match kw.ident().unwrap_or("") {
                "var" => {
                    let [t] = rest else { bail!("{}: expected var NAME;", kw.loc) };
                    cur = Some(self.exo_index(&t.raw, &t.loc)?);
                }
                "periods" => {
                    periods.clear();
                    let mut k = 0;
                    while k < rest.len() {
                        if rest[k].is(",") {
                            k += 1;
                            continue;
                        }
                        let a = match rest[k].tok {
                            Tok::Num(v) if v >= 1.0 && v.fract() == 0.0 => v as usize,
                            _ => bail!("{}: periods: expected a positive integer", rest[k].loc),
                        };
                        k += 1;
                        let mut b = a;
                        if rest.get(k).is_some_and(|t| t.is(":")) {
                            b = match rest.get(k + 1).map(|t| &t.tok) {
                                Some(Tok::Num(v)) if *v >= a as f64 && v.fract() == 0.0 => *v as usize,
                                _ => bail!("{}: periods: bad range", rest[k].loc),
                            };
                            k += 2;
                        }
                        periods.push((a..=b).collect());
                    }
                }
                "values" => {
                    let Some(exo) = cur else { bail!("{}: values before var", kw.loc) };
                    let vals = self.surprise_values(rest)?;
                    if vals.len() != periods.len() && vals.len() != 1 {
                        bail!(
                            "{}: shocks(surprise): {} values for {} period groups",
                            kw.loc,
                            vals.len(),
                            periods.len()
                        );
                    }
                    for (g, p) in periods.iter().enumerate() {
                        let value = if vals.len() == 1 { vals[0] } else { vals[g] };
                        shocks.push(crate::occbin::SurpriseShock { exo, periods: p.clone(), value });
                    }
                }
                _ => bail!("{}: shocks(surprise): unexpected '{}'", kw.loc, kw.raw),
            }
        }
        self.commands.push(Command::Occbin(crate::occbin::OccbinCommand::SurpriseShocks { overwrite, shocks }));
        Ok(())
    }
}
impl<'a> Parser<'a> {
    /// Shock values: signed numbers, names or `(expression)`, separated by optional commas.
    fn surprise_values(&self, t: &[Token]) -> Result<Vec<f64>> {
        let mut out = Vec::new();
        let mut k = 0;
        while k < t.len() {
            if t[k].is(",") {
                k += 1;
                continue;
            }
            let st = k;
            if t[k].is("(") {
                let mut depth = 0;
                while k < t.len() {
                    if t[k].is("(") {
                        depth += 1;
                    } else if t[k].is(")") {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    k += 1;
                }
                if k == t.len() {
                    bail!("{}: unbalanced parenthesis in values", t[st].loc);
                }
                k += 1;
            } else {
                if t[k].is("-") || t[k].is("+") {
                    k += 1;
                }
                k += 1;
            }
            out.push(self.eval_tokens(&t[st..k.min(t.len())])?);
        }
        Ok(out)
    }
}
// ---- end OccBin -------------------------------------------------------------------------
/// Tokens joined with spaces (re-lexable .mod source).
fn src_join(toks: &[Token]) -> String {
    toks.iter()
        .map(|t| match &t.tok {
            Tok::Str(s) => format!("'{s}'"),
            _ => t.raw.clone(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// `values 0.1 -0.2, (a*2) b` -> one token slice per value.
fn split_value_items(toks: &[Token]) -> Vec<&[Token]> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if toks[i].is(",") {
            i += 1;
            continue;
        }
        let st = i;
        if toks[i].is("-") || toks[i].is("+") {
            i += 1;
        }
        if toks.get(i).is_some_and(|t| t.is("(")) {
            let mut depth = 0;
            while i < toks.len() {
                if toks[i].is("(") {
                    depth += 1;
                } else if toks[i].is(")") {
                    depth -= 1;
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                i += 1;
            }
        } else {
            i += 1;
        }
        out.push(&toks[st..i.min(toks.len())]);
    }
    out
}

// ---- MoM (stream: method of moments) ------------------------------------------------------
impl<'a> Parser<'a> {
    fn endo_index_mom(&self, t: &Token) -> Result<usize> {
        match self.b.symbol(&t.raw) {
            Some(s) if s.kind == SymbolKind::Endo => Ok(s.index),
            _ => bail!("{}: '{}' is not an endogenous variable", t.loc, t.raw),
        }
    }

    /// `matched_moments; y; c*c(-1); y(1)^2*c(-4)^3; end;` -> `M_.matched_moments` rows
    /// (declaration indices, leads/lags, powers; one entry per factor as Dynare's preprocessor).
    fn matched_moments_block(&mut self) -> Result<()> {
        self.c.i += 1;
        self.header_options()?;
        let mut rows = Vec::new();
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                break;
            }
            let toks = self.c.statement()?;
            if toks.is_empty() {
                continue;
            }
            let mut m = crate::mom::MatchedMoment::default();
            let mut k = 0;
            loop {
                let Some(t) = toks.get(k) else { bail!("{}: matched_moments: incomplete product", toks[0].loc) };
                let v = self.endo_index_mom(t)?;
                k += 1;
                let mut lag = 0i32;
                if toks.get(k).is_some_and(|t| t.is("(")) {
                    let close = toks[k..].iter().position(|t| t.is(")")).map(|p| p + k);
                    let Some(close) = close else { bail!("{}: matched_moments: missing ')'", t.loc) };
                    let x = self.eval_tokens(&toks[k + 1..close])?;
                    if x.fract() != 0.0 {
                        bail!("{}: matched_moments: leads/lags must be integers", t.loc);
                    }
                    lag = x as i32;
                    k = close + 1;
                }
                let mut pow = 1u32;
                if toks.get(k).is_some_and(|t| t.is("^")) {
                    let x = match toks.get(k + 1).map(|t| &t.tok) {
                        Some(Tok::Num(x)) if *x >= 0.0 && x.fract() == 0.0 => *x,
                        _ => bail!("{}: matched_moments: the power must be a non-negative integer", t.loc),
                    };
                    pow = x as u32;
                    k += 2;
                }
                // a factor to the power 0 is dropped (`YGR^1*INFL^0` is `YGR`, as in Dynare)
                if pow > 0 {
                    m.vars.push(v);
                    m.lags.push(lag);
                    m.powers.push(pow);
                }
                match toks.get(k) {
                    None => break,
                    Some(t) if t.is("*") => k += 1,
                    Some(t) => bail!("{}: matched_moments: unexpected '{}'", t.loc, t.raw),
                }
            }
            if m.vars.is_empty() {
                bail!("{}: matched_moments: every factor has the power 0", toks[0].loc);
            }
            rows.push(m);
        }
        self.commands.push(Command::Mom(crate::mom::MomCommand::MatchedMoments(rows)));
        Ok(())
    }

    /// Items of `periods 1, 3:5 7` / `values 0.2, (expr), ([1 2 3])`: top-level groups separated
    /// by commas or blanks; `a:b` ranges and `(...)` / `[...]` groups are single items.
    fn mom_items(t: &[Token]) -> Vec<&[Token]> {
        let mut out = Vec::new();
        let mut k = 0;
        while k < t.len() {
            if t[k].is(",") {
                k += 1;
                continue;
            }
            let st = k;
            if t[k].is("(") || t[k].is("[") {
                let mut depth = 0;
                while k < t.len() {
                    if t[k].is("(") || t[k].is("[") {
                        depth += 1;
                    } else if t[k].is(")") || t[k].is("]") {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    k += 1;
                }
                k += 1;
            } else {
                if t[k].is("-") || t[k].is("+") {
                    k += 1;
                }
                k += 1;
                while t.get(k).is_some_and(|x| x.is(":")) {
                    k += 2;
                }
            }
            out.push(&t[st..k.min(t.len())]);
        }
        out
    }

    /// A value item: a number, `(expression)` or `([v1 v2 ...])` / `[v1 v2 ...]`.
    fn mom_values(&self, item: &[Token]) -> Result<Vec<f64>> {
        let inner = if item.len() >= 2 && item[0].is("(") && item[item.len() - 1].is(")") {
            &item[1..item.len() - 1]
        } else {
            item
        };
        if inner.len() >= 2 && inner[0].is("[") && inner[inner.len() - 1].is("]") {
            return numeric_vector(&inner[1..inner.len() - 1])
                .ok_or_else(|| anyhow!("{}: expected a numeric vector", item[0].loc));
        }
        if let Some(v) = numeric_vector(inner).filter(|v| v.len() > 1) {
            return Ok(v);
        }
        // a MATLAB workspace vector: `values (xx)` / `weights (ww)`
        if let [name] = inner
            && let Some(n) = name.ident()
            && self.b.symbol(n).is_none()
            && let Some(v) = self.q1.workspace.vector(n)
        {
            return Ok(v);
        }
        self.eval_tokens(inner).map(|x| vec![x]).map_err(|e| {
            anyhow!(
                "{e} (dynare-rust evaluates `values`/`weights` as constant expressions of numbers and parameters; MATLAB workspace variables are not available)"
            )
        })
    }

    /// `matched_irfs[(overwrite)]; var y; varexo e; periods 1:4, 6; values ...; weights ...; end;`
    fn matched_irfs_block(&mut self) -> Result<()> {
        self.c.i += 1;
        let opts = self.header_options()?;
        let overwrite = opts.iter().any(|(k, _)| k == "overwrite");
        let mut entries: Vec<crate::mom::MatchedIrf> = Vec::new();
        let mut cur = crate::mom::MatchedIrf::default();
        let flush = |cur: &mut crate::mom::MatchedIrf, entries: &mut Vec<crate::mom::MatchedIrf>| {
            if cur.endo.is_some() || cur.exo.is_some() || !cur.periods.is_empty() {
                entries.push(std::mem::take(cur));
            }
        };
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                break;
            }
            let toks = self.c.statement()?;
            let Some((kw, rest)) = toks.split_first() else { continue };
            match kw.ident().unwrap_or("") {
                "var" | "varexo" => {
                    let is_var = kw.is_ident("var");
                    let [t] = rest else { bail!("{}: expected {} NAME;", kw.loc, kw.raw) };
                    // a new entry starts with var/varexo once the current one has data or the name
                    if !cur.periods.is_empty() || (is_var && cur.endo.is_some()) || (!is_var && cur.exo.is_some()) {
                        flush(&mut cur, &mut entries);
                    }
                    if is_var {
                        cur.endo = Some(self.endo_index_mom(t)?);
                    } else {
                        cur.exo = Some(self.exo_index(&t.raw, &t.loc)?);
                    }
                }
                "periods" => {
                    for item in Self::mom_items(rest) {
                        let v = numeric_vector(item)
                            .filter(|v| !v.is_empty() && v.iter().all(|x| *x >= 1.0 && x.fract() == 0.0))
                            .ok_or_else(|| anyhow!("{}: periods: expected positive integers or ranges", kw.loc))?;
                        cur.periods.push(v.iter().map(|&x| x as usize).collect());
                    }
                }
                "values" => {
                    for item in Self::mom_items(rest) {
                        cur.values.push(self.mom_values(item)?);
                    }
                }
                "weights" => {
                    for item in Self::mom_items(rest) {
                        cur.weights.push(self.mom_values(item)?);
                    }
                }
                _ => bail!("{}: matched_irfs: unexpected '{}'", kw.loc, kw.raw),
            }
        }
        flush(&mut cur, &mut entries);
        self.commands.push(Command::Mom(crate::mom::MomCommand::MatchedIrfs { overwrite, entries }));
        Ok(())
    }

    /// `matched_irfs_weights[(overwrite)]; y(1), e, c(2:3), e, 20; end;`
    fn matched_irfs_weights_block(&mut self) -> Result<()> {
        self.c.i += 1;
        let opts = self.header_options()?;
        let overwrite = opts.iter().any(|(k, _)| k == "overwrite");
        let mut entries = Vec::new();
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                break;
            }
            let toks = self.c.statement()?;
            if toks.is_empty() {
                continue;
            }
            let f = split_commas(toks);
            if f.len() != 5 {
                bail!("{}: matched_irfs_weights: expected VAR(H), SHOCK, VAR(H), SHOCK, WEIGHT;", toks[0].loc);
            }
            let var_h = |f: &[Token]| -> Result<(usize, Vec<usize>)> {
                let Some(t) = f.first() else { bail!("{}: matched_irfs_weights: empty field", toks[0].loc) };
                let v = self.endo_index_mom(t)?;
                if f.len() < 4 || !f[1].is("(") || !f[f.len() - 1].is(")") {
                    bail!("{}: matched_irfs_weights: expected VAR(HORIZON)", t.loc);
                }
                let h = numeric_vector(&f[2..f.len() - 1])
                    .filter(|v| !v.is_empty() && v.iter().all(|x| *x >= 1.0 && x.fract() == 0.0))
                    .ok_or_else(|| anyhow!("{}: matched_irfs_weights: bad horizon", t.loc))?;
                Ok((v, h.iter().map(|&x| x as usize).collect()))
            };
            let shock = |f: &[Token]| -> Result<usize> {
                match f {
                    [t] => self.exo_index(&t.raw, &t.loc),
                    _ => bail!("{}: matched_irfs_weights: expected a shock name", toks[0].loc),
                }
            };
            let (endo1, periods1) = var_h(f[0])?;
            let exo1 = shock(f[1])?;
            let (endo2, periods2) = var_h(f[2])?;
            let exo2 = shock(f[3])?;
            let weights = self.mom_values(f[4])?;
            entries.push(crate::mom::IrfWeight { endo1, exo1, periods1, endo2, exo2, periods2, weights });
        }
        self.commands.push(Command::Mom(crate::mom::MomCommand::MatchedIrfsWeights { overwrite, entries }));
        Ok(())
    }
}
// ---- end MoM ------------------------------------------------------------------------------

/// `1:12`, `1 2 3`, `1, -2`, `0:0.5:2` -> numbers; `None` if not purely numeric.
fn numeric_vector(toks: &[Token]) -> Option<Vec<f64>> {
    let mut out = Vec::new();
    let mut i = 0;
    let num = |i: &mut usize| -> Option<f64> {
        let neg = toks.get(*i)?.is("-");
        if neg {
            *i += 1;
        }
        let Tok::Num(x) = toks.get(*i)?.tok else {
            return None;
        };
        *i += 1;
        Some(if neg { -x } else { x })
    };
    while i < toks.len() {
        if toks[i].is(",") {
            i += 1;
            continue;
        }
        let a = num(&mut i)?;
        if toks.get(i).is_some_and(|t| t.is(":")) {
            i += 1;
            let b = num(&mut i)?;
            let (step, hi) = if toks.get(i).is_some_and(|t| t.is(":")) {
                i += 1;
                (b, num(&mut i)?)
            } else {
                (1.0, b)
            };
            if step == 0.0 {
                return None;
            }
            let mut x = a;
            while (step > 0.0 && x <= hi + 1e-12) || (step < 0.0 && x >= hi - 1e-12) {
                out.push(x);
                x += step;
            }
        } else {
            out.push(a);
        }
    }
    Some(out)
}

// ==== [stream P] optimal-policy statements ===============================================
impl<'a> Parser<'a> {
    fn planner_spec(&mut self) -> &mut crate::model::planner::PlannerSpec {
        self.b.planner.get_or_insert_with(Default::default)
    }

    /// `planner_objective EXPR;` and the blocks `optim_weights`, `osr_params_bounds`,
    /// `ramsey_constraints`.
    fn policy_statement(&mut self, kw: &str) -> Result<()> {
        let loc = self.c.next()?.loc.clone();
        if kw == "planner_objective" {
            self.c.nums.clear();
            let ast = self.c.expr()?;
            self.c.expect(";")?;
            let nums = std::mem::take(&mut self.c.nums);
            let node = self.resolve_ast(&ast)?;
            let d = &self.b.dag;
            if d.max_endo_lead(node) > 0
                || d.max_endo_lag(node) > 0
                || d.max_exo_lead(node) > 0
                || d.max_exo_lag(node) > 0
            {
                bail!("{loc}: leads and lags on variables are forbidden in 'planner_objective'");
            }
            self.planner_spec().objective = Some(node);
            self.planner_spec().objective_ast = Some((ast, nums));
            return Ok(());
        }
        self.header_options()?;
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                return Ok(());
            }
            if self.c.peek().is_none() {
                bail!("{loc}: block '{kw}' not closed by 'end;'");
            }
            let toks = self.c.statement()?;
            let tloc = toks.first().map(|t| t.loc.to_string()).unwrap_or_default();
            match kw {
                "optim_weights" => {
                    let endo = |t: &Token| -> Result<usize> {
                        match self.b.symbol(&t.raw) {
                            Some(s) if s.kind == SymbolKind::Endo => Ok(s.index),
                            _ => bail!("{}: optim_weights: '{}' is not an endogenous variable", t.loc, t.raw),
                        }
                    };
                    let (i, j, rest) = if toks.len() >= 3 && toks[1].is(",") {
                        (endo(&toks[0])?, endo(&toks[2])?, &toks[3..])
                    } else if !toks.is_empty() {
                        let i = endo(&toks[0])?;
                        (i, i, &toks[1..])
                    } else {
                        continue;
                    };
                    let w = self.eval_tokens(rest)?;
                    self.policy.optim_weights.push((i, j, w));
                }
                "osr_params_bounds" => {
                    let f = split_commas(toks);
                    let [name, lb, ub] = f[..] else { bail!("{tloc}: expected NAME, LOWER, UPPER") };
                    let p = match name {
                        [t] => match self.b.symbol(&t.raw) {
                            Some(s) if s.kind == SymbolKind::Param => s.index,
                            _ => bail!("{tloc}: '{}' is not a parameter", t.raw),
                        },
                        _ => bail!("{tloc}: expected a parameter name"),
                    };
                    let (lb, ub) = (self.eval_tokens(lb)?, self.eval_tokens(ub)?);
                    self.policy.osr_bounds.push((p, lb, ub));
                }
                _ => self.policy.ramsey_constraints.push(raw_join(toks)),
            }
        }
    }

    /// `ramsey_model`, `ramsey_policy`, `discretionary_policy`, `osr`, `osr_params`,
    /// `evaluate_planner_objective`.
    fn policy_command(&mut self, name: &str, opts: Options, vars: Vec<String>) -> Result<Command> {
        let loc = self.c.loc();
        if name == "osr_params" {
            for v in &vars {
                match self.b.symbol(v) {
                    Some(s) if s.kind == SymbolKind::Param => self.policy.osr_params.push(s.index),
                    _ => bail!("{loc}: osr_params: '{v}' is not a parameter"),
                }
            }
            return Ok(Command::Policy(name.to_string(), opts, vars));
        }
        let discount = match super::opt(&opts, "planner_discount") {
            None => None,
            Some(OptValue::Num(v)) => Some(*v),
            Some(OptValue::Str(s)) => {
                let ast = parse_expression(s)?;
                Some(eval_ast(&ast, &|n, l, lc| self.lookup_param(n, l, lc))?)
            }
            Some(v) => bail!("{loc}: planner_discount: unexpected value {v:?}"),
        };
        let instruments: Vec<String> = match super::opt(&opts, "instruments") {
            None => Vec::new(),
            Some(OptValue::Str(s)) => vec![s.clone()],
            Some(OptValue::List(l)) => l
                .iter()
                .map(|v| match v {
                    OptValue::Str(s) => Ok(s.clone()),
                    _ => Err(anyhow!("{loc}: instruments: expected variable names")),
                })
                .collect::<Result<_>>()?,
            Some(v) => bail!("{loc}: instruments: unexpected value {v:?}"),
        };
        let spec = self.planner_spec();
        match name {
            "ramsey_model" | "ramsey_policy" => spec.ramsey = true,
            "discretionary_policy" => spec.discretionary = true,
            "osr" => spec.osr = true,
            _ => {}
        }
        if discount.is_some() {
            spec.discount = discount;
        }
        if !instruments.is_empty() {
            spec.instruments = instruments;
        }
        Ok(Command::Policy(name.to_string(), opts, vars))
    }
}
// ==== [end stream P] =======================================================================

#[cfg(test)]
mod tests {
    use super::super::{opt, parse_mod_str};
    use super::*;

    #[test]
    fn expression_precedence() {
        let a = parse_expression("-2^2 + 3*4/2 - 1").unwrap();
        let v = eval_ast(&a, &|_, _, _| unreachable!()).unwrap();
        assert_eq!(v, -4.0 + 6.0 - 1.0);
        let a = parse_expression("2^3^2").unwrap();
        assert_eq!(eval_ast(&a, &|_, _, _| unreachable!()).unwrap(), 512.0);
        let a = parse_expression("x(-1) + y(+4) * exp(0)").unwrap();
        match a {
            Ast::Bin(BinaryOp::Add, l, r) => {
                assert!(matches!(*l, Ast::Var { ref name, lag: -1, .. } if name == "x"));
                assert!(matches!(*r, Ast::Bin(BinaryOp::Mul, ..)));
            }
            _ => panic!("{a:?}"),
        }
        assert!(parse_expression("foo(bar)").is_err());
    }

    const SMALL: &str = r#"
var y pi i;
varexo e_y e_pi;
parameters rho beta kappa phi ss;
rho = 0.9; beta = 0.99; kappa = 0.1;
phi = 1.5*rho;  // references an earlier parameter
ss = 2;
model(linear);
#r = i - pi(+1);
[name='IS']
y = y(+1) - 0.5*r + e_y;
pi = beta*pi(+2) + kappa*y + e_pi;
i = phi*pi(-3) + ss;
end;
initval;
pi = ss;
i = ss*phi + ss;
end;
shocks;
var e_y; stderr 0.5;
var e_pi = 0.04;
corr e_y, e_pi = 0.5;
end;
varobs y, pi;
estimated_params;
rho, 0.8, beta_pdf, 0.9, 0.05;
stderr e_y, 0.4, 0.01, 5, inv_gamma_pdf, 0.5, 2;
kappa, normal_pdf, 0.1, 0.05;
beta, 0.9, , , uniform_pdf, , , 0, 1;
phi, 1.2, 1, 3, gamma_pdf, 1.5, 0.25, 0, 10, 0.3;
corr e_y, e_pi, 0.1, normal_pdf, 0, 0.3;
end;
estimated_params_init;
rho, 0.85;
stderr e_y, 0.45;
end;
steady(solve_algo=1, maxit=1000);
estimation(datafile='d.csv', graph_format=(pdf,eps), mode_file=out/m_mode, nodisplay,
           filter_step_ahead=[1:3], conf_sig=0.7, mh_jscale=-0.5) y pi;
generate_trace_plots([1:4]);
homotopy_setup;
rho, 0.5, 0.9;
end;
"#;

    #[test]
    fn parses_small_model() {
        let m = parse_mod_str(SMALL, &[]).unwrap();
        let model = &m.model;
        assert_eq!(model.param_names, ["rho", "beta", "kappa", "phi", "ss"]);
        assert!((model.params[3] - 1.35).abs() < 1e-15);
        assert!(model.linear);
        assert_eq!(model.equation_tags[0], vec![("name".to_string(), "IS".to_string())]);
        // pi(+2) -> one lead aux; pi(-3) -> two lag aux
        assert_eq!(
            model.endo_names,
            ["y", "pi", "i", model.endo_names[3].as_str(), "AUX_ENDO_LAG_1_1", "AUX_ENDO_LAG_1_2"]
        );
        assert!(model.endo_names[3].starts_with("AUX_ENDO_LEAD_"));
        assert_eq!(model.aux_vars[0].orig_expr, "pi(1)");
        assert_eq!(model.aux_vars[1].orig_expr, "pi(-1)");
        assert_eq!(model.aux_vars[2].orig_expr, "AUX_ENDO_LAG_1_1(-1)");
        assert_eq!(model.aux_vars[2].orig_lead_lag, Some(-2));
        assert_eq!(m.initval, vec![(1, 2.0), (2, 2.0 * 1.35 + 2.0)]);
        assert_eq!(m.varobs, ["y", "pi"]);
        assert_eq!(m.sigma_e[(0, 0)], 0.25);
        assert_eq!(m.sigma_e[(1, 1)], 0.04);
        assert!((m.sigma_e[(0, 1)] - 0.5 * 0.5 * 0.2).abs() < 1e-15);
        let e = &m.estimated_params;
        assert_eq!(e.len(), 6);
        assert_eq!(e[0].prior, Some(PriorShape::Beta));
        assert_eq!(e[0].init, Some(0.8));
        assert_eq!((e[1].kind, e[1].lb, e[1].ub), (EstimatedKind::StdErr(0), Some(0.01), Some(5.0)));
        assert_eq!((e[2].init, e[2].mean, e[2].sd), (None, 0.1, 0.05));
        assert!(e[3].mean.is_nan() && e[3].lb.is_none());
        assert_eq!((e[3].p3, e[3].p4), (Some(0.0), Some(1.0)));
        assert_eq!((e[4].p3, e[4].p4, e[4].jscale), (Some(0.0), Some(10.0), Some(0.3)));
        assert_eq!(e[5].kind, EstimatedKind::Corr(0, 1));
        let order: Vec<String> = m.estimated_params_dynare_order().iter().map(|e| m.estimated_param_name(e)).collect();
        assert_eq!(order, ["stderr e_y", "corr e_y, e_pi", "rho", "kappa", "beta", "phi"]);
        assert_eq!(m.estimated_params_init.len(), 2);
        assert_eq!(m.commands.len(), 3);
        let Command::Estimation(o, vars) = &m.commands[1] else { panic!() };
        assert_eq!(vars, &["y", "pi"]);
        assert_eq!(opt(o, "datafile"), Some(&OptValue::Str("d.csv".into())));
        assert_eq!(
            opt(o, "graph_format"),
            Some(&OptValue::List(vec![OptValue::Str("pdf".into()), OptValue::Str("eps".into())]))
        );
        assert_eq!(opt(o, "mode_file"), Some(&OptValue::Str("out/m_mode".into())));
        assert_eq!(opt(o, "nodisplay"), Some(&OptValue::Flag));
        assert_eq!(opt(o, "filter_step_ahead"), Some(&OptValue::Vec(vec![1.0, 2.0, 3.0])));
        assert_eq!(opt(o, "mh_jscale"), Some(&OptValue::Num(-0.5)));
        assert!(matches!(&m.commands[2], Command::Other(n, _) if n == "generate_trace_plots"));
        assert_eq!(m.homotopy_setup, vec![("rho".to_string(), Some(0.5), 0.9)]); // [stream X]
    }

    #[test]
    fn small_model_derivatives() {
        let m = parse_mod_str(SMALL, &[]).unwrap();
        let model = &m.model;
        let n = model.n_endo();
        let ys = vec![0.3, 2.0, 0.7, 2.0, 2.0, 2.0];
        let p = &model.params;
        let g1 = model.dynamic_jacobian(&ys, &[0.0, 0.0], p);
        assert_eq!(g1.shape(), (6, 3 * n + 2));
        // eq 0: y - y(+1) + 0.5*(i - pi(+1)) - e_y
        assert_eq!(g1[(0, n)], 1.0);
        assert_eq!(g1[(0, 2 * n)], -1.0);
        assert_eq!(g1[(0, 2 * n + 1)], -0.5);
        assert_eq!(g1[(0, n + 2)], 0.5);
        assert_eq!(g1[(0, 3 * n)], -1.0);
        // eq 1: pi - beta*AUX(+1) - kappa*y - e_pi
        assert_eq!(g1[(1, 2 * n + 3)], -0.99);
        // eq 2: i - phi*AUX_LAG_2(-1) - ss
        assert!((g1[(2, 5)] + 1.35).abs() < 1e-15);
        // aux eqs
        assert_eq!((g1[(3, n + 3)], g1[(3, 2 * n + 1)]), (1.0, -1.0));
        assert_eq!((g1[(4, n + 4)], g1[(4, 1)]), (1.0, -1.0));
        assert_eq!((g1[(5, n + 5)], g1[(5, 4)]), (1.0, -1.0));
        // finite differences of the dynamic residuals
        let mut y3: Vec<f64> = (0..3 * n).map(|k| 0.1 * k as f64).collect();
        let x = [0.2, -0.1];
        let mut ws = model.workspace();
        let mut g = Mat::zeros(6, 3 * n + 2);
        model.dynamic_jacobian_y3_with(&y3, &x, p, &mut ws, &mut g);
        let mut r0 = vec![0.0; 6];
        let mut r1 = vec![0.0; 6];
        model.dynamic_residuals(&y3, &x, p, &mut r0);
        for c in 0..3 * n {
            y3[c] += 1e-6;
            model.dynamic_residuals(&y3, &x, p, &mut r1);
            y3[c] -= 1e-6;
            for i in 0..6 {
                assert!(((r1[i] - r0[i]) / 1e-6 - g[(i, c)]).abs() < 1e-6);
            }
        }
        // static
        let gs = model.static_jacobian(&ys, &[0.0, 0.0], p);
        assert!((gs[(1, 1)] - 1.0).abs() < 1e-15 && (gs[(1, 3)] + 0.99).abs() < 1e-15);
        let mut y = vec![0.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        model.set_auxiliary_variables(&mut y, &[0.0, 0.0], p);
        assert_eq!(&y[3..], &[2.0, 2.0, 2.0]);
        // lead_lag_incidence: lags pi? no (pi(-3) is substituted); AUX_LAG_1_1, AUX_LAG_1_2 and pi lagged
        assert_eq!(model.lead_lag_incidence[1][0], 1); // pi(-1) in AUX_ENDO_LAG_1_1 = pi(-1)
        let ce = model.compile_expr("phi*(1-rho) + ss").unwrap();
        assert!((ce.eval(&[], &[], p) - (1.35 * 0.1 + 2.0)).abs() < 1e-15);
    }

    #[test]
    fn errors_point_to_lines() {
        let e = parse_mod_str("var x;\nparameters a;\nmodel;\nx = a*z;\nend;\n", &[]).unwrap_err();
        assert!(format!("{e:#}").contains("<string>:4") && format!("{e:#}").contains("'z'"), "{e:#}");
        let e = parse_mod_str("var x;\nmodel;\nx = 1\nend;\n", &[]).unwrap_err();
        assert!(format!("{e:#}").contains("<string>:4"), "{e:#}");
    }

    #[test]
    fn product_of_leads_creates_expression_aux() {
        // x(+2)*y(+1): no factor is lead-free, so Dynare substitutes the whole product.
        let src = "var x y;\nvarexo e;\nmodel;\nx = x(+2)*y(+1) + e;\ny = 0.5*y(-1) + e;\nend;\nstoch_simul;\n";
        let m = parse_mod_str(src, &[]).unwrap();
        assert_eq!(m.model.aux_vars.len(), 1);
        assert_eq!(m.model.aux_vars[0].orig_expr, "y*x(1)"); // Dynare 7.2: operands ordered by node index
        let src = "var x;\nvarexo e;\nmodel;\nx = 0.5*x(-1) + e(-2) + e(+1);\nend;\nstoch_simul;\n";
        let m = parse_mod_str(src, &[]).unwrap();
        let names: Vec<&str> = m.model.endo_names.iter().map(String::as_str).collect();
        assert_eq!(names[2..], ["AUX_EXO_LAG_1_0", "AUX_EXO_LAG_1_1"]);
        assert!(names[1].starts_with("AUX_EXO_LEAD_"));
        assert_eq!(m.model.aux_vars[0].ty, 2);
        assert_eq!(m.model.aux_vars[0].orig_expr, "e");
        assert_eq!(m.model.aux_vars[2].orig_expr, "AUX_EXO_LAG_1_0(-1)");
    }
}
