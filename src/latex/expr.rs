// SPDX-License-Identifier: GPL-3.0-or-later
//! LaTeX printing of expressions, following the preprocessor's `ExprNode::writeOutput` in
//! LaTeX mode: `{x}_{t-1}` (static: `{x}`), `\, ` for products, `\frac{a}{b}`, `{a}^{b}`,
//! `\left( \right)` parentheses, `\exp\left(..\right)`, `\log\left(..\right)`, `\sqrt{..}`.

use crate::model::SymbolKind;
use crate::model::expr::{BinaryOp, Dag, Node, NodeId, UnaryOp};
use crate::modfile::{Ast, ModFile};

/// Names and time convention for printing.
pub struct Ctx<'a> {
    pub mf: &'a ModFile,
    /// Static model: no time subscripts.
    pub is_static: bool,
}

impl<'a> Ctx<'a> {
    pub fn new(mf: &'a ModFile, is_static: bool) -> Self {
        Ctx { mf, is_static }
    }
    fn var(&self, tex: &str, lag: i32) -> String {
        if self.is_static {
            format!("{{{tex}}}")
        } else if lag == 0 {
            format!("{{{tex}}}_{{t}}")
        } else if lag > 0 {
            format!("{{{tex}}}_{{t+{lag}}}")
        } else {
            format!("{{{tex}}}_{{t{lag}}}")
        }
    }
}

/// A printable expression (common form of parser ASTs and DAG nodes).
enum E {
    Num(f64),
    Leaf(String),
    Un(UnaryOp, Box<E>),
    Bin(BinaryOp, Box<E>, Box<E>),
}

fn prec(e: &E) -> u8 {
    match e {
        E::Num(v) if *v < 0.0 => 4,
        E::Num(_) | E::Leaf(_) => 10,
        E::Un(UnaryOp::Neg, _) => 4,
        E::Un(..) => 10,
        E::Bin(op, ..) => match op {
            BinaryOp::Lt | BinaryOp::Gt | BinaryOp::Le | BinaryOp::Ge | BinaryOp::Eq | BinaryOp::Ne => 1,
            BinaryOp::Add | BinaryOp::Sub => 2,
            BinaryOp::Mul => 3,
            // \frac{}{} and {}^{} are self-delimiting, functions too
            BinaryOp::Div | BinaryOp::Max | BinaryOp::Min | BinaryOp::PowerDeriv(_) => 10,
            BinaryOp::Pow => 5,
        },
    }
}

fn paren(s: String) -> String {
    format!("\\left({s}\\right)")
}

fn fmt_num(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 { format!("{}", v as i64) } else { format!("{v}") }
}

fn print(e: &E) -> String {
    match e {
        E::Num(v) => fmt_num(*v),
        E::Leaf(s) => s.clone(),
        E::Un(UnaryOp::Neg, a) => {
            let s = print(a);
            if prec(a) <= 2 { format!("-{}", paren(s)) } else { format!("-{s}") }
        }
        E::Un(op, a) => {
            let s = print(a);
            match op {
                UnaryOp::Sqrt => format!("\\sqrt{{{s}}}"),
                UnaryOp::Abs => format!("\\left|{s}\\right|"),
                UnaryOp::Log10 => format!("\\log_{{10}}{}", paren(s)),
                UnaryOp::Sign => format!("\\mathrm{{sign}}{}", paren(s)),
                // [stream X]
                UnaryOp::Cbrt => format!("\\sqrt[3]{{{s}}}"),
                UnaryOp::Erf | UnaryOp::Erfc | UnaryOp::Asinh | UnaryOp::Acosh | UnaryOp::Atanh => {
                    format!("\\mathrm{{{}}}{}", op.name(), paren(s))
                }
                op => format!("\\{}{}", op.name(), paren(s)),
            }
        }
        E::Bin(op, a, b) => {
            let (sa, sb) = (print(a), print(b));
            let p = prec(e);
            let wrap = |s: String, sub: &E, min: u8| if prec(sub) < min { paren(s) } else { s };
            match op {
                // as BinaryOpNode::writeOutput: a right operand of + needs parentheses only below
                // its precedence (the transformed model may nest sums on the right, stream PP)
                BinaryOp::Add => format!("{}+{}", wrap(sa, a, p), wrap(sb, b, 2)),
                BinaryOp::Sub => format!("{}-{}", wrap(sa, a, p), wrap(sb, b, 3)),
                BinaryOp::Mul => format!("{}\\, {}", wrap(sa, a, p), wrap(sb, b, p)),
                BinaryOp::Div => format!("\\frac{{{sa}}}{{{sb}}}"),
                // as the preprocessor: the base is parenthesised when its precedence is below
                // the power's (sums, products, \frac, negations) or when it is a power itself;
                // only the exponent is braced
                BinaryOp::Pow => {
                    let base_paren = prec(a) < 5 || matches!(**a, E::Bin(BinaryOp::Div | BinaryOp::Pow, ..));
                    format!("{}^{{{sb}}}", if base_paren { paren(sa) } else { sa })
                }
                BinaryOp::Max | BinaryOp::Min => format!("\\{}{}", op.symbol(), paren(format!("{sa}, {sb}"))),
                cmp => {
                    let sym = match cmp {
                        BinaryOp::Le => "\\leq ",
                        BinaryOp::Ge => "\\geq ",
                        BinaryOp::Ne => "\\neq ",
                        BinaryOp::Eq => "==",
                        o => o.symbol(),
                    };
                    format!("{}{}{}", wrap(sa, a, 2), sym, wrap(sb, b, 2))
                }
            }
        }
    }
}

fn from_ast(a: &Ast, ctx: &Ctx) -> E {
    match a {
        Ast::Num(v) => E::Num(*v),
        Ast::Var { name, lag, .. } => {
            let mf = ctx.mf;
            let leaf = match mf.model.symbol(name) {
                Some(s) => match s.kind {
                    SymbolKind::Endo => ctx.var(&super::endo_tex(mf, s.index), *lag),
                    SymbolKind::Exo => ctx.var(&super::exo_tex(mf, s.index), *lag),
                    SymbolKind::Param => match super::exo_det_of(mf, s.index) {
                        Some(k) => ctx.var(&super::exo_det_tex(mf, k), *lag), // varexo_det
                        None => format!("{{{}}}", super::param_tex(mf, s.index)),
                    },
                    SymbolKind::Local => format!("{{{}}}", super::local_tex(mf, name)), // model_local_variable
                    SymbolKind::Trend => format!("{{{}}}", super::escape_underscore(name)),
                },
                None => format!("{{{}}}", super::escape_underscore(name)),
            };
            E::Leaf(leaf)
        }
        Ast::Un(op, x) => E::Un(*op, Box::new(from_ast(x, ctx))),
        Ast::Bin(op, x, y) => E::Bin(*op, Box::new(from_ast(x, ctx)), Box::new(from_ast(y, ctx))),
        // ---- stream PP: STEADY_STATE (`(\bar{x})`, as the preprocessor), other operators as functions
        Ast::Op(op, args) => {
            let inner: Vec<String> = args
                .iter()
                .map(|a| {
                    if *op == crate::modfile::SpecialOp::SteadyState {
                        print(&steady_state_e(from_ast(a, ctx)))
                    } else {
                        print(&from_ast(a, ctx))
                    }
                })
                .collect();
            E::Leaf(match op {
                crate::modfile::SpecialOp::SteadyState => format!("({})", inner.join(",")),
                crate::modfile::SpecialOp::Expectation(k) => {
                    format!(
                        "\\mathbb{{E}}_{{t{}}}\\left[{}\\right]",
                        if *k < 0 { format!("{k}") } else { format!("+{k}") },
                        inner.join(",")
                    )
                }
                o => format!("\\mathrm{{{}}}\\left({}\\right)", o.name(), inner.join(",")),
            })
        }
    }
}

/// Leaves `{x}_{t..}` of an expression under `STEADY_STATE` printed as `\bar{x}`.
fn steady_state_e(e: E) -> E {
    match e {
        E::Leaf(s) => {
            let base = s.split("}_{").next().unwrap_or(&s).trim_start_matches('{').to_string();
            if s.contains("}_{") || s.starts_with('{') {
                E::Leaf(format!("\\bar{{{}}}", base.trim_end_matches('}')))
            } else {
                E::Leaf(s)
            }
        }
        E::Num(v) => E::Num(v),
        E::Un(op, a) => E::Un(op, Box::new(steady_state_e(*a))),
        E::Bin(op, a, b) => E::Bin(op, Box::new(steady_state_e(*a)), Box::new(steady_state_e(*b))),
    }
}

fn from_dag(d: &Dag, id: NodeId, ctx: &Ctx) -> E {
    let mf = ctx.mf;
    match d.node(id) {
        Node::Num(b) => E::Num(f64::from_bits(b)),
        Node::Endo(j, l) => E::Leaf(ctx.var(&super::endo_tex(mf, j as usize), l)),
        Node::Exo(k, l) => E::Leaf(ctx.var(&super::exo_tex(mf, k as usize), l)),
        Node::Param(i) => E::Leaf(match super::exo_det_of(mf, i as usize) {
            Some(k) => ctx.var(&super::exo_det_tex(mf, k), 0), // varexo_det
            None => format!("{{{}}}", super::param_tex(mf, i as usize)),
        }),
        // stream PP: STEADY_STATE(x), as the preprocessor's LaTeX output (`\bar{x}`)
        Node::SteadyState(j, _) => E::Leaf(format!("\\bar{{{}}}", super::endo_tex(mf, j as usize))),
        Node::SteadyStateParamDeriv(..) | Node::SteadyStateParam2ndDeriv(..) => {
            unreachable!("parameter derivatives of STEADY_STATE are not part of the model equations")
        }
        Node::Unary(op, a) => E::Un(op, Box::new(from_dag(d, a, ctx))),
        Node::Binary(op, a, b) => E::Bin(op, Box::new(from_dag(d, a, ctx)), Box::new(from_dag(d, b, ctx))),
        Node::External(c) => {
            let e = d.ext_call(c);
            let inner: Vec<String> = e.args.iter().map(|&a| print(&from_dag(d, a, ctx))).collect();
            let f = crate::model::external::name(e.fid);
            E::Leaf(match (e.d1, e.d2) {
                (0, _) => format!("\\mathrm{{{f}}}\\left({}\\right)", inner.join(",")),
                (i, 0) => format!("\\mathrm{{{f}}}_{{{i}}}\\left({}\\right)", inner.join(",")),
                (i, j) => format!("\\mathrm{{{f}}}_{{{i},{j}}}\\left({}\\right)", inner.join(",")),
            })
        }
    }
}

/// LaTeX of a parsed (original-model) expression; model-local variables stay symbols.
pub fn ast_latex(a: &Ast, ctx: &Ctx) -> String {
    print(&from_ast(a, ctx))
}

/// Mimics the operand order of the preprocessor's `DataTree`: nodes are hash-consed and
/// numbered in creation order (post-order, left to right, starting after the constants 0, 1,
/// 2, 3), and `AddPlus`/`AddTimes` put the operand with the lower number on the left.
/// One `Orderer` spans a whole model block (locals then equations), like Dynare's tree.
#[derive(Default)]
pub struct Orderer {
    ids: std::collections::HashMap<String, usize>,
}

impl Orderer {
    pub fn new() -> Self {
        let mut o = Orderer::default();
        for c in ["0", "1", "2", "3"] {
            o.id(format!("N:{c}"));
        }
        o
    }
    fn id(&mut self, key: String) -> usize {
        let n = self.ids.len();
        *self.ids.entry(key).or_insert(n)
    }
    fn canon(&mut self, e: E) -> (E, usize) {
        match e {
            E::Num(v) => {
                let k = format!("N:{}", fmt_num(v));
                (E::Num(v), self.id(k))
            }
            E::Leaf(s) => {
                let k = format!("L:{s}");
                (E::Leaf(s), self.id(k))
            }
            E::Un(op, a) => {
                let (a, ia) = self.canon(*a);
                let k = format!("U:{op:?}:{ia}");
                (E::Un(op, Box::new(a)), self.id(k))
            }
            E::Bin(op, a, b) => {
                let (a, ia) = self.canon(*a);
                let (b, ib) = self.canon(*b);
                let (a, ia, b, ib) = if matches!(op, BinaryOp::Add | BinaryOp::Mul) && ia > ib {
                    (b, ib, a, ia)
                } else {
                    (a, ia, b, ib)
                };
                let k = format!("B:{op:?}:{ia}:{ib}");
                (E::Bin(op, Box::new(a), Box::new(b)), self.id(k))
            }
        }
    }
}

/// [`ast_latex`] with Dynare's operand order (see [`Orderer`]).
pub fn ast_latex_ordered(a: &Ast, ctx: &Ctx, ord: &mut Orderer) -> String {
    let (e, _) = ord.canon(from_ast(a, ctx));
    print(&e)
}

/// LaTeX of a DAG expression (transformed model).
pub fn dag_latex(d: &Dag, id: NodeId, ctx: &Ctx) -> String {
    print(&from_dag(d, id, ctx))
}
