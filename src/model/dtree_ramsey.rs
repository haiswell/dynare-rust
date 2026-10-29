// SPDX-License-Identifier: GPL-3.0-or-later
//! Ramsey first-order conditions in the preprocessor's DataTree (`DynamicModel::
//! computeRamseyPolicyFOCs`): symbolic differentiation of the Lagrangian with the preprocessor's
//! `composeDerivatives` rules, so that the node indices (which name `AUX_ENDO_LEAD_<idx>`) match.

use super::{BOp, Id, N, SymType, Tree, UOp};
use crate::model::expr::UnaryOp;
use anyhow::{Result, bail};
use std::collections::{BTreeSet, HashMap};

/// What the Ramsey step needs from the model builder.
pub(crate) struct PlannerIn {
    pub objective: crate::modfile::Ast,
    pub nums: Vec<String>,
    /// Symbol id of `optimal_policy_discount_factor`.
    pub discount: u32,
    /// Symbol ids of the declared endogenous variables, by endogenous index.
    pub endo_symb: Vec<u32>,
}

/// What the Ramsey step adds to the model (`M_.ramsey_orig_endo_nbr`, `M_.ramsey_orig_eq_nbr`).
#[derive(Clone, Debug, Default)]
pub(crate) struct RamseyOut {
    /// Number of FOCs: the declared variables plus the `diff` / `EXPECTATION` aux variables.
    pub orig_endo: usize,
    /// Equations the multipliers are attached to (model equations, then the `diff` /
    /// `EXPECTATION` aux equations).
    pub orig_eq: usize,
}

impl Tree {
    fn diff_node(&mut self, id: Id, v: (u32, i32), memo: &mut HashMap<Id, Id>) -> Result<Id> {
        if let Some(&r) = memo.get(&id) {
            return Ok(r);
        }
        let z = self.zero;
        let r = match self.node(id) {
            N::Num(_) => z,
            N::Var(s, l) => match self.sym_ty(s) {
                SymType::Local => bail!("internal error: model-local variable in a Ramsey derivative"),
                _ => {
                    if (s, l) == v {
                        self.one
                    } else {
                        z
                    }
                }
            },
            N::Un(op, a, _) => {
                let da = self.diff_node(a, v, memo)?;
                if da == z { z } else { self.diff_un(op, a, id, da)? }
            }
            N::Bin(op, a, b) => {
                let da = self.diff_node(a, v, memo)?;
                let db = self.diff_node(b, v, memo)?;
                if da == z && db == z { z } else { self.diff_bin(op, a, b, id, da, db)? }
            }
            N::Tri(..) => bail!("normcdf/normpdf are not supported together with ramsey_model"),
            N::Ext(..) => bail!("external functions are not supported together with ramsey_model"),
        };
        memo.insert(id, r);
        Ok(r)
    }

    fn diff_un(&mut self, op: UOp, a: Id, this: Id, da: Id) -> Result<Id> {
        let (one, two) = (self.one, self.two);
        let f = match op {
            UOp::Uminus => return Ok(self.add_uminus(da)),
            UOp::SteadyState => return Ok(self.zero),
            UOp::Expectation | UOp::Diff => bail!("diff/EXPECTATION are not supported together with ramsey_model"),
            UOp::F(f) => f,
        };
        use UnaryOp as U;
        Ok(match f {
            U::Neg => self.add_uminus(da),
            U::Exp => self.add_times(da, this),
            U::Log => self.add_divide(da, a)?,
            U::Log10 => {
                let ten = self.add_possibly_negative(10.0);
                let ln10 = self.add_fn(U::Log, ten);
                let t = self.add_times(a, ln10);
                self.add_divide(da, t)?
            }
            U::Sqrt => {
                let t = self.add_times(two, this);
                self.add_divide(da, t)?
            }
            U::Abs => {
                let s = self.add_fn(U::Sign, a);
                self.add_times(da, s)
            }
            U::Sign => self.zero,
            U::Sin => {
                let c = self.add_fn(U::Cos, a);
                self.add_times(da, c)
            }
            U::Cos => {
                let s = self.add_fn(U::Sin, a);
                let t = self.add_times(da, s);
                self.add_uminus(t)
            }
            U::Tan => {
                let t = self.add_times(this, this);
                let t = self.add_plus(one, t);
                self.add_times(da, t)
            }
            U::Asin | U::Acos => {
                let t = self.add_times(a, a);
                let t = self.add_minus(one, t);
                let t = self.add_fn(U::Sqrt, t);
                let r = self.add_divide(da, t)?;
                if f == U::Acos { self.add_uminus(r) } else { r }
            }
            U::Atan => {
                let t = self.add_times(a, a);
                let t = self.add_plus(one, t);
                self.add_divide(da, t)?
            }
            U::Sinh => {
                let c = self.add_fn(U::Cosh, a);
                self.add_times(da, c)
            }
            U::Cosh => {
                let s = self.add_fn(U::Sinh, a);
                self.add_times(da, s)
            }
            U::Tanh => {
                let t = self.add_times(this, this);
                let t = self.add_minus(one, t);
                self.add_times(da, t)
            }
            U::Asinh => {
                let c = self.add_fn(U::Cosh, this);
                self.add_divide(da, c)?
            }
            U::Acosh => {
                let s = self.add_fn(U::Sinh, this);
                self.add_divide(da, s)?
            }
            U::Atanh => {
                let t = self.add_times(a, a);
                let t = self.add_minus(one, t);
                self.add_divide(da, t)?
            }
            U::Cbrt => {
                let tt = self.add_possibly_negative(2.0 / 3.0);
                let t = self.add_power(a, tt);
                let three = self.three;
                let t = self.add_times(three, t);
                self.add_divide(da, t)?
            }
            U::Erf | U::Erfc => {
                let a2 = self.add_power(a, two);
                let e = self.add_fn(U::Exp, a2);
                let sp = self.add_possibly_negative(std::f64::consts::PI.sqrt());
                let t = self.add_times(sp, e);
                let t = self.add_divide(two, t)?;
                let t = self.add_times(t, da);
                if f == U::Erfc { self.add_uminus(t) } else { t }
            }
        })
    }

    fn diff_bin(&mut self, op: BOp, a: Id, b: Id, this: Id, da: Id, db: Id) -> Result<Id> {
        let (z, one) = (self.zero, self.one);
        Ok(match op {
            BOp::Plus => self.add_plus(da, db),
            BOp::Minus => self.add_minus(da, db),
            BOp::Times => {
                let t1 = self.add_times(da, b);
                let t2 = self.add_times(a, db);
                self.add_plus(t1, t2)
            }
            BOp::Divide => {
                if db == z {
                    self.add_divide(da, b)?
                } else {
                    let t11 = self.add_times(da, b);
                    let t12 = self.add_times(db, a);
                    let t13 = self.add_minus(t11, t12);
                    let t14 = self.add_times(b, b);
                    self.add_divide(t13, t14)?
                }
            }
            BOp::Power => {
                if db == z && matches!(self.node(b), N::Num(_)) {
                    let bm1 = self.add_minus(b, one);
                    let t12 = self.add_power(a, bm1);
                    let t13 = self.add_times(b, t12);
                    self.add_times(da, t13)
                } else if db == z {
                    let t = self.add_binary_op(a, BOp::PowerDeriv(1), b);
                    self.add_times(da, t)
                } else {
                    let t11 = self.add_fn(UnaryOp::Log, a);
                    let t12 = self.add_times(db, t11);
                    let t13 = self.add_times(da, b);
                    let t14 = self.add_divide(t13, a)?;
                    let t15 = self.add_plus(t12, t14);
                    self.add_times(t15, this)
                }
            }
            BOp::Max | BOp::Min => {
                let ca = if op == BOp::Max {
                    self.add_binary_op(a, BOp::Greater, b)
                } else {
                    self.add_binary_op(b, BOp::Greater, a)
                };
                let cb = self.add_minus(one, ca);
                let t1 = self.add_times(ca, da);
                let t2 = self.add_times(cb, db);
                self.add_plus(t2, t1)
            }
            _ => z,
        })
    }

    /// `computeRamseyPolicyFOCs`: creates the multipliers `MULT_1..` (aux variables of type 6,
    /// one per equation) and puts the first-order conditions (w.r.t. every endogenous variable
    /// existing at this point: the declared ones and the `diff` / `EXPECTATION` aux variables,
    /// which are substituted before) in front of the equations, which become `lhs - rhs = 0`.
    ///
    /// The Lagrangian is `U + sum_i sum_s beta^s MULT_i(s) f_i(s)` where `f_i = lhs_i - rhs_i`
    /// and `f_i(s)` is `f_i` shifted by `s` periods, for the shifts `s = -l` of the leads/lags
    /// `l` that occur in equation `i`; the FOC of `y_j` is its derivative with respect to
    /// `y_j(0)`. The node creation order (power, multiplier, shifted equation; equation by
    /// equation, shifts in increasing order) reproduces the preprocessor's node indices.
    pub(super) fn ramsey_focs(&mut self, p: &PlannerIn, lookup: &dyn Fn(&str) -> Option<u32>) -> Result<RamseyOut> {
        let mut endo_symb = p.endo_symb.clone();
        endo_symb.extend(self.aux.iter().map(|a| a.symb));
        let mut it = p.nums.iter();
        let u = self.replay(&p.objective, &mut it, lookup)?;
        let pbeta = self.add_var(p.discount, 0);
        let n_eq = self.equations.len();
        let mut mults = Vec::with_capacity(n_eq);
        for i in 0..n_eq {
            let name = format!("MULT_{}", i + 1);
            if lookup(&name).is_some() {
                bail!("'{name}' is reserved for the Ramsey multipliers");
            }
            let z = self.zero;
            mults.push(self.add_aux(name, 6, None, None, z));
        }
        let mut resid = Vec::with_capacity(n_eq);
        for k in 0..n_eq {
            let (l, r) = self.eq_args(self.equations[k].eq);
            resid.push(self.add_minus(l, r));
        }
        let mut lsum = self.zero;
        for (i, &f) in resid.iter().enumerate() {
            let mut set = BTreeSet::new();
            self.collect_vars(f, SymType::Endo, &mut set, &mut BTreeSet::new());
            let shifts: BTreeSet<i32> = set.iter().map(|x| -x.1).collect();
            let mut esum = self.zero;
            for s in shifts {
                let c = if s != 0 {
                    let e = self.add_possibly_negative(s as f64);
                    self.add_power(pbeta, e)
                } else {
                    self.one
                };
                let m = self.add_var(mults[i], s);
                let shifted = if s == 0 { f } else { self.decrease(f, -s) };
                let term = self.add_times(m, shifted);
                let term = if s != 0 { self.add_times(c, term) } else { term };
                esum = self.add_plus(esum, term);
            }
            lsum = self.add_plus(lsum, esum);
        }
        let lagr = self.add_plus(u, lsum);
        let zero = self.zero;
        let mut eqs = Vec::with_capacity(endo_symb.len());
        for (j, &sj) in endo_symb.iter().enumerate() {
            let f = self.diff_node(lagr, (sj, 0), &mut HashMap::new())?;
            let e = self.add_equal(f, zero);
            let name = format!("Ramsey FOC w.r.t. {}", self.names[endo_symb[j] as usize]);
            eqs.push(super::Equation { eq: e, tags: vec![("name".to_string(), name)], line: 0 });
        }
        let n = eqs.len();
        for k in 0..resid.len() {
            self.equations[k].eq = self.add_equal(resid[k], zero);
        }
        eqs.append(&mut self.equations);
        self.equations = eqs;
        Ok(RamseyOut { orig_endo: n, orig_eq: n_eq })
    }
}
