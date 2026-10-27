// SPDX-License-Identifier: GPL-3.0-or-later
//! Flat instruction tape: the compiled form of a set of DAG roots.
//!
//! Plays the role of Dynare's generated `dynamic_g1.m` / `static_resid.m`: every
//! DAG node reachable from the roots becomes one instruction writing one slot of a
//! caller-provided scratch buffer (shared subexpressions are computed once, like
//! Dynare's temporary terms). Evaluation takes `&self` and allocates nothing.

use super::expr::{BinaryOp, Dag, Node, NodeId, UnaryOp};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
enum Ins {
    Const(f64),
    Y(u32),
    X(u32),
    P(u32),
    Neg(u32),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Div(u32, u32),
    Un(UnaryOp, u32),
    Bin(BinaryOp, u32, u32),
    /// `external_function` call: index into [`Tape::ext`].
    Ext(u32),
}

/// A compiled external call: the function, the derivative indices and the argument slots.
#[derive(Clone, Debug)]
struct ExtIns {
    f: std::sync::Arc<super::external::ExternalFunction>,
    d1: u16,
    d2: u16,
    args: Vec<u32>,
}

/// A compiled list of expressions.
#[derive(Clone, Debug, Default)]
pub struct Tape {
    ins: Vec<Ins>,
    outputs: Vec<u32>,
    ext: Vec<ExtIns>,
}

impl Tape {
    /// Compiles `roots`. `y_index(j, lag)` / `x_index(k, lag)` map variables to
    /// positions in the `y` / `x` input slices.
    pub fn compile(
        dag: &Dag,
        roots: &[NodeId],
        y_index: &dyn Fn(u32, i32) -> usize,
        x_index: &dyn Fn(u32, i32) -> usize,
    ) -> Tape {
        let mut slot: HashMap<NodeId, u32> = HashMap::new();
        let mut ins = Vec::new();
        let mut outputs = Vec::with_capacity(roots.len());
        let mut ext = Vec::new();
        for &r in roots {
            outputs.push(Self::emit(dag, r, &mut slot, &mut ins, &mut ext, y_index, x_index));
        }
        Tape { ins, outputs, ext }
    }

    fn emit(
        dag: &Dag,
        root: NodeId,
        slot: &mut HashMap<NodeId, u32>,
        ins: &mut Vec<Ins>,
        ext: &mut Vec<ExtIns>,
        y_index: &dyn Fn(u32, i32) -> usize,
        x_index: &dyn Fn(u32, i32) -> usize,
    ) -> u32 {
        // iterative post-order DFS
        let mut stack = vec![(root, false)];
        while let Some((id, expanded)) = stack.pop() {
            if slot.contains_key(&id) {
                continue;
            }
            let node = dag.node(id);
            if !expanded {
                stack.push((id, true));
                match node {
                    Node::Unary(_, a) => stack.push((a, false)),
                    Node::Binary(_, a, b) => {
                        stack.push((b, false));
                        stack.push((a, false));
                    }
                    Node::External(c) => stack.extend(dag.ext_call(c).args.iter().rev().map(|&a| (a, false))),
                    _ => {}
                }
                continue;
            }
            let s = |n: NodeId| slot[&n];
            let i = match node {
                Node::Num(b) => Ins::Const(f64::from_bits(b)),
                Node::Endo(j, l) => Ins::Y(y_index(j, l) as u32),
                Node::Exo(k, l) => Ins::X(x_index(k, l) as u32),
                Node::Param(p) => Ins::P(p),
                // stream PP: STEADY_STATE(y_j) reads the extended parameter vector [params; ys]
                Node::SteadyState(j, _) => Ins::P(dag.ss_offset + j),
                // ss_param_deriv / ss_param_2nd_deriv: the dys / d2ys blocks that follow
                Node::SteadyStateParamDeriv(j, q) => Ins::P(dag.ss_param_deriv_slot(j, q)),
                Node::SteadyStateParam2ndDeriv(j, q1, q2) => Ins::P(dag.ss_param_2nd_deriv_slot(j, q1, q2)),
                Node::Unary(UnaryOp::Neg, a) => Ins::Neg(s(a)),
                Node::Unary(op, a) => Ins::Un(op, s(a)),
                Node::Binary(BinaryOp::Add, a, b) => Ins::Add(s(a), s(b)),
                Node::Binary(BinaryOp::Sub, a, b) => Ins::Sub(s(a), s(b)),
                Node::Binary(BinaryOp::Mul, a, b) => Ins::Mul(s(a), s(b)),
                Node::Binary(BinaryOp::Div, a, b) => Ins::Div(s(a), s(b)),
                Node::Binary(op, a, b) => Ins::Bin(op, s(a), s(b)),
                Node::External(c) => {
                    let e = dag.ext_call(c);
                    ext.push(ExtIns {
                        f: super::external::get(e.fid),
                        d1: e.d1,
                        d2: e.d2,
                        args: e.args.iter().map(|&a| s(a)).collect(),
                    });
                    Ins::Ext(ext.len() as u32 - 1)
                }
            };
            slot.insert(id, ins.len() as u32);
            ins.push(i);
        }
        slot[&root]
    }

    /// Scratch slots needed by [`Tape::eval`].
    pub fn scratch_len(&self) -> usize {
        self.ins.len()
    }

    pub fn n_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Runs the tape; afterwards output `k` is `s[self.outputs[k]]` (see [`Tape::output`]).
    #[inline]
    pub fn eval(&self, y: &[f64], x: &[f64], p: &[f64], s: &mut [f64]) {
        let s = &mut s[..self.ins.len()];
        for (i, ins) in self.ins.iter().enumerate() {
            let v = match *ins {
                Ins::Const(c) => c,
                Ins::Y(j) => y[j as usize],
                Ins::X(k) => x[k as usize],
                Ins::P(q) => p[q as usize],
                Ins::Neg(a) => -s[a as usize],
                Ins::Add(a, b) => s[a as usize] + s[b as usize],
                Ins::Sub(a, b) => s[a as usize] - s[b as usize],
                Ins::Mul(a, b) => s[a as usize] * s[b as usize],
                Ins::Div(a, b) => s[a as usize] / s[b as usize],
                Ins::Un(op, a) => op.apply(s[a as usize]),
                Ins::Bin(op, a, b) => op.apply(s[a as usize], s[b as usize]),
                Ins::Ext(k) => {
                    let e = &self.ext[k as usize];
                    let mut buf = [0.0f64; 16];
                    if e.args.len() <= buf.len() {
                        for (b, &a) in buf.iter_mut().zip(&e.args) {
                            *b = s[a as usize];
                        }
                        e.f.eval(e.d1, e.d2, &buf[..e.args.len()])
                    } else {
                        let v: Vec<f64> = e.args.iter().map(|&a| s[a as usize]).collect();
                        e.f.eval(e.d1, e.d2, &v)
                    }
                }
            };
            s[i] = v;
        }
    }

    #[inline]
    pub fn output(&self, k: usize, s: &[f64]) -> f64 {
        s[self.outputs[k] as usize]
    }

    /// Evaluates and copies all outputs into `out`.
    pub fn eval_into(&self, y: &[f64], x: &[f64], p: &[f64], s: &mut [f64], out: &mut [f64]) {
        self.eval(y, x, p, s);
        for (o, &k) in out.iter_mut().zip(&self.outputs) {
            *o = s[k as usize];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tape_matches_interpreter() {
        let mut d = Dag::new();
        let x = d.endo(0, -1);
        let y = d.endo(1, 1);
        let e = d.exo(0, 0);
        let p = d.param(0);
        let t = d.mul(p, x);
        let t2 = d.unary(UnaryOp::Exp, y);
        let r1 = d.add(t, t2);
        let r2 = d.binary(BinaryOp::Pow, r1, e);
        let r3 = d.neg(r1);
        let yi = |j: u32, l: i32| ((l + 1) * 2) as usize + j as usize;
        let tape = Tape::compile(&d, &[r1, r2, r3], &yi, &|k, _| k as usize);
        let yv = [0.5, 0.0, 0.0, 0.0, 0.0, 0.3];
        let mut s = vec![0.0; tape.scratch_len()];
        let mut out = [0.0; 3];
        tape.eval_into(&yv, &[2.0], &[1.5], &mut s, &mut out);
        let r1v = 1.5 * 0.5 + 0.3f64.exp();
        assert!((out[0] - r1v).abs() < 1e-15);
        assert!((out[1] - r1v.powf(2.0)).abs() < 1e-14);
        assert_eq!(out[2], -r1v);
    }
}
