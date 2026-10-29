// SPDX-License-Identifier: GPL-3.0-or-later
//! Second and third derivatives of the dynamic model (Dynare's `dynamic_g2` / `dynamic_g3`),
//! for higher-order perturbation (stream K, `solver::korder`).
//!
//! Built on demand from a [`Model`] (the first-order path is untouched): the residual DAG is
//! cloned, differentiated symbolically up to order 3 and compiled to tapes like the Jacobian.
//! Variables are the columns of the dynamic Jacobian, `[y_{t-1} (n), y_t (n), y_{t+1} (n), x (n_exo)]`
//! (0-based, `Model::dynamic_jacobian` layout). Only unique entries are stored, with sorted
//! variable indices (`v1 <= v2 <= v3`), as in `M_.dynamic_g2_sparse_indices` /
//! `M_.dynamic_g3_sparse_indices` (which are 1-based).

use super::Model;
use super::expr::{Dag, DiffVar, Node, NodeId};
use super::tape::Tape;
use std::collections::HashMap;

/// Compiled 2nd (and optionally 3rd) derivatives of the dynamic residuals.
#[derive(Clone, Debug)]
pub struct DynamicDerivatives {
    /// Number of variables (`3 n + n_exo`).
    pub nvar: usize,
    /// Highest derivative order compiled (2 or 3).
    pub order: usize,
    /// `(eq, v1, v2)`, `v1 <= v2`, one per structurally non-zero unique second derivative.
    pub g2_index: Vec<(usize, usize, usize)>,
    /// `(eq, v1, v2, v3)`, `v1 <= v2 <= v3`.
    pub g3_index: Vec<(usize, usize, usize, usize)>,
    n_endo: usize,
    /// stream PP: `STEADY_STATE()` support (tapes read `[params; ys]`).
    n_params: usize,
    has_ss: bool,
    tape2: Tape,
    tape3: Tape,
}

/// Values at a point, in the order of `g2_index` / `g3_index`.
#[derive(Clone, Debug, Default)]
pub struct DerivativeValues {
    pub g2: Vec<f64>,
    pub g3: Vec<f64>,
}

fn col_var(n: usize, col: usize) -> DiffVar {
    if col < 3 * n {
        DiffVar::Endo((col % n) as u32, (col / n) as i32 - 1)
    } else {
        DiffVar::Exo((col - 3 * n) as u32, 0)
    }
}

fn leaf_cols(dag: &Dag, id: NodeId, n: usize, buf: &mut Vec<Node>) -> Vec<usize> {
    buf.clear();
    dag.collect_leaves(id, buf);
    let mut cols: Vec<usize> = buf
        .iter()
        .filter_map(|l| match *l {
            Node::Endo(j, l) if (-1..=1).contains(&l) => Some((l + 1) as usize * n + j as usize),
            Node::Exo(k, 0) => Some(3 * n + k as usize),
            _ => None,
        })
        .collect();
    cols.sort_unstable();
    cols.dedup();
    cols
}

impl Model {
    /// Symbolic 2nd (`order >= 2`) and 3rd (`order >= 3`) derivatives of the dynamic model.
    pub fn dynamic_derivatives(&self, order: usize) -> DynamicDerivatives {
        let n = self.n_endo();
        let nvar = 3 * n + self.n_exo;
        let mut dag = self.dag.clone();
        let mut memo: HashMap<DiffVar, HashMap<NodeId, NodeId>> = HashMap::new();
        let mut buf = Vec::new();
        let mut d2: Vec<(usize, usize, usize, NodeId)> = Vec::new();
        let mut d3: Vec<(usize, usize, usize, usize, NodeId)> = Vec::new();
        for (eq, &r) in self.resid.iter().enumerate() {
            let cols = leaf_cols(&dag, r, n, &mut buf);
            for &c1 in &cols {
                let v1 = col_var(n, c1);
                let f1 = dag.diff(r, v1, memo.entry(v1).or_default());
                if f1 == Dag::ZERO {
                    continue;
                }
                let cols1 = leaf_cols(&dag, f1, n, &mut buf);
                for &c2 in cols1.iter().filter(|&&c| c >= c1) {
                    let v2 = col_var(n, c2);
                    let f2 = dag.diff(f1, v2, memo.entry(v2).or_default());
                    if f2 == Dag::ZERO {
                        continue;
                    }
                    d2.push((eq, c1, c2, f2));
                    if order >= 3 {
                        let cols2 = leaf_cols(&dag, f2, n, &mut buf);
                        for &c3 in cols2.iter().filter(|&&c| c >= c2) {
                            let v3 = col_var(n, c3);
                            let f3 = dag.diff(f2, v3, memo.entry(v3).or_default());
                            if f3 != Dag::ZERO {
                                d3.push((eq, c1, c2, c3, f3));
                            }
                        }
                    }
                }
            }
        }
        let dyn_y = move |j: u32, l: i32| (l + 1) as usize * n + j as usize;
        let x0 = |k: u32, _l: i32| k as usize;
        let n2: Vec<NodeId> = d2.iter().map(|e| e.3).collect();
        let n3: Vec<NodeId> = d3.iter().map(|e| e.4).collect();
        DynamicDerivatives {
            nvar,
            order: order.clamp(2, 3),
            g2_index: d2.iter().map(|e| (e.0, e.1, e.2)).collect(),
            g3_index: d3.iter().map(|e| (e.0, e.1, e.2, e.3)).collect(),
            n_endo: n,
            n_params: self.n_params(),
            has_ss: self.uses_steady_state(),
            tape2: Tape::compile(&dag, &n2, &dyn_y, &x0),
            tape3: Tape::compile(&dag, &n3, &dyn_y, &x0),
        }
    }
}

impl DynamicDerivatives {
    /// Evaluates at `y3 = [y_{t-1}; y_t; y_{t+1}]`, `x` (exogenous at t) and parameters `p`.
    /// With `STEADY_STATE()` in the model, `p` is `[params; ys]` (`Model::params_with_steady_state`);
    /// plain `params` read the steady state from `y_t`.
    pub fn eval_y3(&self, y3: &[f64], x: &[f64], p: &[f64]) -> DerivativeValues {
        assert_eq!(y3.len(), 3 * self.n_endo);
        let pext;
        let p = if self.has_ss && p.len() < self.n_params + self.n_endo {
            pext = [&p[..self.n_params], &y3[self.n_endo..2 * self.n_endo]].concat();
            &pext[..]
        } else {
            p
        };
        let mut s = vec![0.0; self.tape2.scratch_len().max(self.tape3.scratch_len())];
        let mut g2 = vec![0.0; self.g2_index.len()];
        self.tape2.eval_into(y3, x, p, &mut s, &mut g2);
        let mut g3 = vec![0.0; self.g3_index.len()];
        if !g3.is_empty() {
            self.tape3.eval_into(y3, x, p, &mut s, &mut g3);
        }
        DerivativeValues { g2, g3 }
    }

    /// Evaluates at the steady state `ys` (all leads/lags at `ys`).
    pub fn eval(&self, ys: &[f64], x: &[f64], p: &[f64]) -> DerivativeValues {
        let y3: Vec<f64> = ys.iter().chain(ys).chain(ys).copied().collect();
        self.eval_y3(&y3, x, p)
    }

    /// True when every second derivative is structurally zero (`M_.hessian_eq_zero`).
    pub fn hessian_eq_zero(&self) -> bool {
        self.g2_index.is_empty()
    }
}

/// Symbolic derivatives of the dynamic model of every order `2..=order` (any `order`), for the
/// general k-order solver (`order >= 4`). Same variables and sorted-index convention as
/// [`DynamicDerivatives`]: `index[m - 2]` lists `(eq, [v1 <= ... <= vm])`.
#[derive(Clone, Debug)]
pub struct DynamicDerivativesK {
    pub order: usize,
    pub index: Vec<Vec<(usize, Vec<usize>)>>,
    n_endo: usize,
    n_params: usize,
    has_ss: bool,
    tapes: Vec<Tape>,
}

impl Model {
    /// Symbolic derivatives of orders `2..=order` of the dynamic model (see [`DynamicDerivativesK`]).
    pub fn dynamic_derivatives_k(&self, order: usize) -> DynamicDerivativesK {
        let n = self.n_endo();
        let mut dag = self.dag.clone();
        let mut memo: HashMap<DiffVar, HashMap<NodeId, NodeId>> = HashMap::new();
        let mut buf = Vec::new();
        let mut found: Vec<Vec<(usize, Vec<usize>, NodeId)>> = vec![Vec::new(); order.saturating_sub(1)];
        // depth-first over sorted variable tuples
        let mut stack: Vec<(usize, Vec<usize>, NodeId)> = Vec::new();
        for (eq, &r) in self.resid.iter().enumerate() {
            stack.push((eq, Vec::new(), r));
            while let Some((eq, cols, f)) = stack.pop() {
                if cols.len() == order {
                    continue;
                }
                let lo = cols.last().copied().unwrap_or(0);
                for c in leaf_cols(&dag, f, n, &mut buf).into_iter().filter(|&c| c >= lo) {
                    let v = col_var(n, c);
                    let d = dag.diff(f, v, memo.entry(v).or_default());
                    if d == Dag::ZERO {
                        continue;
                    }
                    let mut cs = cols.clone();
                    cs.push(c);
                    if cs.len() >= 2 {
                        found[cs.len() - 2].push((eq, cs.clone(), d));
                    }
                    stack.push((eq, cs, d));
                }
            }
        }
        let dyn_y = move |j: u32, l: i32| (l + 1) as usize * n + j as usize;
        let x0 = |k: u32, _l: i32| k as usize;
        let tapes = found
            .iter()
            .map(|f| Tape::compile(&dag, &f.iter().map(|e| e.2).collect::<Vec<_>>(), &dyn_y, &x0))
            .collect();
        DynamicDerivativesK {
            order,
            index: found.into_iter().map(|f| f.into_iter().map(|(eq, cs, _)| (eq, cs)).collect()).collect(),
            n_endo: n,
            n_params: self.n_params(),
            has_ss: self.uses_steady_state(),
            tapes,
        }
    }
}

impl DynamicDerivativesK {
    /// Values at the steady state `ys` (all leads/lags at `ys`), `vals[m - 2]` in the order of
    /// `index[m - 2]` (see [`DynamicDerivatives::eval`] for `p`).
    pub fn eval(&self, ys: &[f64], x: &[f64], p: &[f64]) -> Vec<Vec<f64>> {
        let y3: Vec<f64> = ys.iter().chain(ys).chain(ys).copied().collect();
        let pext;
        let p = if self.has_ss && p.len() < self.n_params + self.n_endo {
            pext = [&p[..self.n_params], ys].concat();
            &pext[..]
        } else {
            p
        };
        let mut s = vec![0.0; self.tapes.iter().map(|t| t.scratch_len()).max().unwrap_or(0)];
        self.tapes
            .iter()
            .zip(&self.index)
            .map(|(t, idx)| {
                let mut out = vec![0.0; idx.len()];
                if !out.is_empty() {
                    t.eval_into(&y3, x, p, &mut s, &mut out);
                }
                out
            })
            .collect()
    }
}
