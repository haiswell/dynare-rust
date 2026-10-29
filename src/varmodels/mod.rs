// SPDX-License-Identifier: GPL-3.0-or-later
//! VAR / trend-component auxiliary models and VAR-based expectations (Dynare's
//! `var_model`, `trend_component_model`, `var_expectation_model`, the `var_expectation(...)`
//! operator and `var_expectation.initialize/update`).
//!
//! The parser (`modfile/parser_vm.rs`) extracts the AR structure of the tagged equations
//! ([`EqSpec`]), as the preprocessor does in `varmatrices.m` / `trend_component_ar_a0.m`, and
//! replaces `var_expectation(m)` by the linear form in the `var_expectation_model_<m>_*`
//! parameters. This module holds those structures (Dynare's `M_.var`, `M_.trend_component`,
//! `M_.var_expectation`), the numerical part of `get_companion_matrix.m` (`oo_.var.<m>.
//! CompanionMatrix`, `oo_.trend_component.<m>.CompanionMatrix`) and of
//! `+var_expectation/update_parameters.m`. A PAC model can sit on top: it needs
//! [`VarModels::companion`] and [`VarModels::auxiliary_kind`].

pub mod backward;
pub mod pac;
pub mod pac_estimate;
mod pac_print;

use crate::linalg::{Mat, mul, solve};
use crate::model::SymbolKind;
use crate::model::expr::{BinaryOp, UnaryOp};
use crate::modfile::lexer::Loc;
use crate::modfile::{Ast, OptValue, Options, SpecialOp, eval_ast};
use anyhow::{Result, anyhow, bail};
use std::collections::BTreeMap;

/// Placeholder variable name the parser gives to `var_expectation(m)` (replaced when the
/// equation is read).
pub const VE_PLACEHOLDER: &str = "\u{1}var_expectation:";

/// One variable of a tagged equation's right-hand side: an endogenous variable or an
/// expression of one (`diff(x)`, `log(x)`, `diff(diff(x))`, ...) that the preprocessor replaces
/// by an auxiliary variable, at lag `lag` ([`key_of`]).
#[derive(Clone, Debug)]
pub struct Term {
    /// The key of the expression at `t` (`x`, `diff(x)`, ...).
    pub var: String,
    /// `x(-lag)`; 0 is a contemporaneous variable (structural VAR).
    pub lag: usize,
    /// Coefficient: an expression of the parameters, evaluated with the current calibration.
    pub coef: Ast,
}

/// Linear structure of one tagged equation (`lhs = sum coef*x(-k) + constants + shocks`).
#[derive(Clone, Debug)]
pub struct EqSpec {
    pub tag: String,
    /// Key of the left-hand side ([`key_of`]) and the expression at `t`.
    pub lhs: String,
    pub lhs_expr: Ast,
    /// The left-hand side has a `diff` (`M_.var.<m>.diff`, `nonstationary`).
    pub lhs_diff: bool,
    /// `undiff` of a left-hand side `diff(e)`: the key of `e` and `e`.
    pub undiff: Option<(String, Ast)>,
    /// The variables that stand for the left-hand side and for its `undiff` once the model is
    /// built (the key until then).
    pub lhs_name: String,
    pub undiff_name: String,
    pub terms: Vec<Term>,
    /// Terms without variables (the VAR constant `c`).
    pub consts: Vec<Ast>,
}

/// `var_model` (`M_.var.<name>`).
#[derive(Clone, Debug)]
pub struct VarModel {
    pub name: String,
    pub structural: bool,
    pub eqtags: Vec<String>,
    pub eqs: Vec<EqSpec>,
}

/// `trend_component_model` (`M_.trend_component.<name>`).
#[derive(Clone, Debug)]
pub struct TrendComponentModel {
    pub name: String,
    pub eqtags: Vec<String>,
    /// Per equation: is it a trend (target) equation.
    pub targets: Vec<bool>,
    /// The `targets` option in its order (`target_eqtags`, the columns of `A0star`).
    pub target_eqtags: Vec<String>,
    pub eqs: Vec<EqSpec>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Horizon {
    Single(usize),
    /// `a:b`, `b` may be infinite.
    Range(usize, f64),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Discount {
    Param(String),
    Value(f64),
}

/// One term `constant * [param *] var` of the `expression` option.
#[derive(Clone, Debug)]
pub struct ExprTerm {
    pub var: String,
    pub param: Option<String>,
    pub constant: f64,
}

/// `var_expectation_model` (`M_.var_expectation.<name>`).
#[derive(Clone, Debug)]
pub struct VarExpectationModel {
    pub name: String,
    pub auxiliary_model_name: String,
    /// `"var"` / `"trend_component"` once [`VarModels::initialize`] has run.
    pub auxiliary_model_type: String,
    pub horizon: Horizon,
    pub time_shift: i32,
    pub discount: Discount,
    pub expr: Vec<ExprTerm>,
    /// Names of the parameters filled by `update` (`param_indices`), in order.
    pub param_names: Vec<String>,
}

/// `oo_.var.<name>` / `oo_.trend_component.<name>`: the companion matrix.
#[derive(Clone, Debug)]
pub struct Companion {
    pub matrix: Mat,
    /// `M_.<type>.<name>.list_of_variables_in_companion_var`.
    pub variables: Vec<String>,
    /// Companion state starts with a 1 (VAR with a constant).
    pub is_constant: bool,
}

/// Reduced-form matrices of a VAR (`varmatrices.m`).
#[derive(Clone, Debug)]
pub struct VarMatrices {
    /// `ar(:,:,k)`, k = 1..p.
    pub ar: Vec<Mat>,
    pub a0: Mat,
    pub constants: Vec<f64>,
}

/// `AR`, `A0`, `A0star` of a trend component model (`trend_component_ar_a0.m`).
#[derive(Clone, Debug)]
pub struct TcMatrices {
    pub ar: Vec<Mat>,
    pub a0: Mat,
    pub a0star: Mat,
}

/// All the VAR-family declarations of a .mod file.
#[derive(Clone, Debug, Default)]
pub struct VarModels {
    /// PAC models (`M_.pac`), see [`pac`].
    pub pacs: Vec<pac::PacModel>,
    pub vars: Vec<VarModel>,
    pub trends: Vec<TrendComponentModel>,
    pub expectations: Vec<VarExpectationModel>,
    /// `oo_.var.<name>.CompanionMatrix`, `oo_.trend_component.<name>.CompanionMatrix`
    /// (key `("var"|"trend_component", name)`).
    pub companions: BTreeMap<(String, String), Companion>,
    /// dseries computed by the commands (`X = simul_backward_model(...)`, `Y = X;`).
    pub series: BTreeMap<String, backward::Series>,
    /// Keys of the variables of the auxiliary models ([`key_of`]) resolved by the transform
    /// pass (`Model::pac_key_names`, same order).
    pub keys: Vec<String>,
    /// The `diff` (type 8) and unary operator (type 10) auxiliary variables of the model:
    /// (name, type, original variable).
    pub aux: Vec<(String, u8, String)>,
}

// ---------------------------------------------------------------- numerics

fn mpow(a: &Mat, k: usize) -> Mat {
    let mut r = Mat::identity(a.nrows(), a.ncols());
    for _ in 0..k {
        r = mul(&r, false, a, false);
    }
    r
}

fn rowmul(v: &Mat, m: &Mat) -> Mat {
    mul(v, false, m, false)
}

/// `v / M` (MATLAB mrdivide, row vector `v`): solves `x M = v`.
fn rdiv(v: &Mat, m: &Mat) -> Result<Mat> {
    solve(&m.transpose(), &v.transpose()).map(|x| x.transpose()).ok_or_else(|| anyhow!("singular matrix"))
}

impl VarModel {
    /// `list_of_variables_in_companion_var`.
    pub fn lhs(&self) -> Vec<String> {
        self.eqs.iter().map(|e| e.lhs_name.clone()).collect()
    }

    /// The companion variables as expressions at `t` (what the `h` / expectation parameters
    /// multiply).
    pub fn companion_exprs(&self) -> Vec<(String, Ast)> {
        self.eqs.iter().map(|e| (e.lhs.clone(), e.lhs_expr.clone())).collect()
    }

    /// The equation whose LHS is the variable of the RHS term `t`. As in the preprocessor, the
    /// LHS `diff(x)` is the auxiliary variable `AUX_DIFF_x`: it matches the terms `diff(x(-k))`
    /// only, and the LHS `x` the level terms only. Terms that match no LHS are left out of the
    /// VAR (`varmatrices.m` has no entry for them).
    fn lhs_index(&self, t: &Term) -> Option<usize> {
        self.eqs.iter().position(|e| e.lhs == t.var)
    }

    /// Lag order `p` (at least one): the largest lag of the terms that enter the VAR.
    pub fn max_lag(&self) -> usize {
        self.eqs
            .iter()
            .flat_map(|e| e.terms.iter().filter(|t| self.lhs_index(t).is_some()).map(|t| t.lag))
            .max()
            .unwrap_or(1)
            .max(1)
    }

    /// `varmatrices(name, params, reducedform)`.
    pub fn matrices(&self, ev: &dyn Fn(&Ast) -> Result<f64>, reducedform: bool) -> Result<VarMatrices> {
        let n = self.eqs.len();
        let p = self.max_lag();
        let mut ar = vec![Mat::zeros(n, n); p];
        let mut a0 = Mat::identity(n, n);
        let mut constants = vec![0.0; n];
        for (i, e) in self.eqs.iter().enumerate() {
            for t in &e.terms {
                let Some(j) = self.lhs_index(t) else { continue };
                let v = ev(&t.coef)?;
                if t.lag == 0 {
                    a0[(i, j)] -= v;
                } else {
                    ar[t.lag - 1][(i, j)] += v;
                }
            }
            for k in &e.consts {
                constants[i] += ev(k)?;
            }
        }
        if reducedform {
            for a in ar.iter_mut() {
                *a = solve(&a0, a).ok_or_else(|| anyhow!("var_model {}: A0 is singular", self.name))?;
            }
            let cm = Mat::from_column_slice(n, 1, &constants);
            let r = solve(&a0, &cm).ok_or_else(|| anyhow!("var_model {}: A0 is singular", self.name))?;
            constants = r.as_slice().to_vec();
            a0 = Mat::identity(n, n);
        }
        Ok(VarMatrices { ar, a0, constants })
    }
}

impl TrendComponentModel {
    /// `list_of_variables_in_companion_var`: the `undiff` of the LHS of the error correction
    /// equations (`getUndiffLHSForPac`), the LHS of the target equations.
    pub fn lhs(&self) -> Vec<String> {
        self.eqs
            .iter()
            .enumerate()
            .map(
                |(i, e)| if self.targets[i] || e.undiff.is_none() { e.lhs_name.clone() } else { e.undiff_name.clone() },
            )
            .collect()
    }

    /// The companion variables as expressions at `t`.
    pub fn companion_exprs(&self) -> Vec<(String, Ast)> {
        self.eqs
            .iter()
            .enumerate()
            .map(|(i, e)| match &e.undiff {
                Some(u) if !self.targets[i] => u.clone(),
                _ => (e.lhs.clone(), e.lhs_expr.clone()),
            })
            .collect()
    }

    pub fn ecm_eqs(&self) -> Vec<usize> {
        (0..self.eqs.len()).filter(|&i| !self.targets[i]).collect()
    }

    /// The target equations in the order of the `targets` option (`target_eqn`).
    pub fn target_eqs(&self) -> Vec<usize> {
        self.target_eqtags.iter().filter_map(|t| self.eqtags.iter().position(|x| x == t)).collect()
    }

    /// `getMaxLag`: the largest lag of the right-hand side terms that are left-hand sides of
    /// the model (`VarMaxLag`).
    pub fn max_lag(&self) -> usize {
        self.eqs
            .iter()
            .flat_map(|e| e.terms.iter().filter(|t| self.eqs.iter().any(|q| q.lhs == t.var)).map(|t| t.lag))
            .max()
            .unwrap_or(1)
            .max(1)
    }

    /// `trend_component_ar_a0(name, params)`.
    pub fn matrices(&self, ev: &dyn Fn(&Ast) -> Result<f64>) -> Result<TcMatrices> {
        let ecm = self.ecm_eqs();
        let tgt = self.target_eqs();
        let (m, q, p) = (ecm.len(), tgt.len(), self.max_lag());
        let dnames: Vec<&str> = ecm.iter().map(|&i| self.eqs[i].lhs.as_str()).collect();
        let names: Vec<&str> = ecm.iter().map(|&i| self.eqs[i].undiff.as_ref().map_or("", |u| u.0.as_str())).collect();
        let tnames: Vec<&str> = tgt.iter().map(|&i| self.eqs[i].lhs.as_str()).collect();
        let mut ar = vec![Mat::zeros(m, m); p];
        let mut a0 = Mat::zeros(m, m);
        let mut a0star = Mat::zeros(m, q);
        for (r, &i) in ecm.iter().enumerate() {
            let e = &self.eqs[i];
            if e.undiff.is_none() {
                bail!("{} is not a trend component model. The LHS variables should be in differences", self.name);
            }
            for t in &e.terms {
                let v = ev(&t.coef)?;
                if let Some(c) = dnames.iter().position(|n| *n == t.var) {
                    if t.lag == 0 {
                        bail!(
                            "trend_component_model {}: contemporaneous diff({}) on the RHS of '{}'",
                            self.name,
                            t.var,
                            e.tag
                        );
                    }
                    ar[t.lag - 1][(r, c)] += v;
                } else if let Some(c) = names.iter().position(|n| *n == t.var) {
                    a0[(r, c)] += v;
                } else if let Some(c) = tnames.iter().position(|n| *n == t.var) {
                    a0star[(r, c)] -= v;
                }
                // other terms are skipped, as by `fillAutoregressiveRow` / `fillErrorCorrectionRow`
            }
        }
        Ok(TcMatrices { ar, a0, a0star })
    }
}

impl VarModels {
    pub fn is_empty(&self) -> bool {
        self.vars.is_empty() && self.trends.is_empty() && self.expectations.is_empty() && self.pacs.is_empty()
    }

    pub fn var(&self, name: &str) -> Option<&VarModel> {
        self.vars.iter().find(|m| m.name == name)
    }

    pub fn trend(&self, name: &str) -> Option<&TrendComponentModel> {
        self.trends.iter().find(|m| m.name == name)
    }

    pub fn expectation(&self, name: &str) -> Option<&VarExpectationModel> {
        self.expectations.iter().find(|m| m.name == name)
    }

    /// `"var"` / `"trend_component"` for an auxiliary model name.
    pub fn auxiliary_kind(&self, name: &str) -> Option<&'static str> {
        if self.var(name).is_some() {
            Some("var")
        } else if self.trend(name).is_some() {
            Some("trend_component")
        } else {
            None
        }
    }

    /// `oo_.<kind>.<name>.CompanionMatrix`, once computed.
    pub fn companion(&self, name: &str) -> Option<&Companion> {
        let k = self.auxiliary_kind(name)?;
        self.companions.get(&(k.to_string(), name.to_string()))
    }

    /// `get_companion_matrix(name)`: computes and stores the companion matrix of an auxiliary
    /// model for the calibration `params` (`names` are the parameter names).
    pub fn get_companion_matrix(&mut self, name: &str, names: &[String], params: &[f64]) -> Result<&Companion> {
        let kind = self.auxiliary_kind(name).ok_or_else(|| anyhow!("Unknown type of auxiliary model."))?;
        let ev = |a: &Ast| eval_coef(a, names, params);
        let comp = if kind == "var" {
            let m = self.var(name).unwrap();
            let vm = m.matrices(&ev, m.structural)?;
            let is_constant = vm.constants.iter().any(|c| c.abs() > 0.0);
            let (n, p, k) = (m.eqs.len(), vm.ar.len(), is_constant as usize);
            let mut c = Mat::zeros(n * p + k, n * p + k);
            for i in 0..p {
                for r in 0..n {
                    for s in 0..n {
                        c[(k + r, k + i * n + s)] = vm.ar[i][(r, s)];
                    }
                    if i > 0 {
                        c[(k + i * n + r, k + (i - 1) * n + r)] = 1.0;
                    }
                }
            }
            if is_constant {
                c[(0, 0)] = 1.0;
                for i in 0..n {
                    c[(1 + i, 0)] = vm.constants[i];
                }
            }
            Companion { matrix: c, variables: m.lhs(), is_constant }
        } else {
            let m = self.trend(name).unwrap();
            let tm = m.matrices(&ev)?;
            let (n, p) = (m.eqs.len(), tm.ar.len());
            let ecm = m.ecm_eqs();
            let tgt = m.target_eqs();
            let mut b = vec![Mat::zeros(n, n); p + 1];
            for (r, &i) in ecm.iter().enumerate() {
                for (c, &j) in ecm.iter().enumerate() {
                    b[0][(i, j)] = tm.a0[(r, c)] + tm.ar[0][(r, c)] + if r == c { 1.0 } else { 0.0 };
                    for k in 1..p {
                        b[k][(i, j)] = tm.ar[k][(r, c)] - tm.ar[k - 1][(r, c)];
                    }
                    b[p][(i, j)] = -tm.ar[p - 1][(r, c)];
                }
                for (c, &j) in tgt.iter().enumerate() {
                    b[0][(i, j)] = -tm.a0star[(r, c)];
                }
            }
            for &i in &tgt {
                b[0][(i, i)] = 1.0;
            }
            let mut c = Mat::zeros(n * (p + 1), n * (p + 1));
            for i in 0..p {
                for r in 0..n {
                    for s in 0..n {
                        c[(r, i * n + s)] = b[i][(r, s)];
                    }
                    c[((i + 1) * n + r, i * n + r)] = 1.0;
                }
            }
            for r in 0..n {
                for s in 0..n {
                    c[(r, p * n + s)] = b[p][(r, s)];
                }
            }
            Companion { matrix: c, variables: m.lhs(), is_constant: false }
        };
        let key = (kind.to_string(), name.to_string());
        self.companions.insert(key.clone(), comp);
        Ok(&self.companions[&key])
    }

    /// `var_expectation.initialize(name)`: resolves the type of the auxiliary model and
    /// computes its companion matrix.
    pub fn initialize(&mut self, ve: &str, names: &[String], params: &[f64]) -> Result<()> {
        let aux = self
            .expectation(ve)
            .ok_or_else(|| anyhow!("VAR_EXPECTATION_MODEL {ve} is not defined!"))?
            .auxiliary_model_name
            .clone();
        let kind = self.auxiliary_kind(&aux).ok_or_else(|| anyhow!("Unknown type of auxiliary model."))?;
        self.expectations.iter_mut().find(|m| m.name == ve).unwrap().auxiliary_model_type = kind.to_string();
        self.get_companion_matrix(&aux, names, params)?;
        Ok(())
    }

    /// `var_expectation.update_parameters`: the new values of the `param_names` of `ve`.
    pub fn update_values(&self, ve: &str, names: &[String], params: &[f64]) -> Result<Vec<(String, f64)>> {
        let m = self.expectation(ve).ok_or_else(|| anyhow!("VAR_EXPECTATION_MODEL {ve} is not defined!"))?;
        let kind = self
            .auxiliary_kind(&m.auxiliary_model_name)
            .ok_or_else(|| anyhow!("Unknown VAR ({}) in VAR_EXPECTATION_MODEL ({ve})!", m.auxiliary_model_name))?;
        let comp =
            self.companion(&m.auxiliary_model_name).filter(|c| !c.matrix.iter().any(|x| x.is_nan())).ok_or_else(
                || anyhow!("Auxiliary model {} has to be estimated or calibrated first!", m.auxiliary_model_name),
            )?;
        let pidx = |n: &str| names.iter().position(|x| x == n).ok_or_else(|| anyhow!("unknown parameter {n}"));
        let beta = match &m.discount {
            Discount::Value(v) => *v,
            Discount::Param(n) => params[pidx(n)?],
        };
        if beta <= 0.0 {
            bail!("The discount factor must be positive.");
        }
        if beta > 1.0 {
            bail!("The discount cannot be greater than one.");
        }
        let ts = m.time_shift;
        if ts > 0 {
            bail!("Option time_shift of the var_expectation command cannot be positive.");
        }
        let c = &comp.matrix;
        let n = c.nrows();
        let mut ids = Vec::new();
        for t in &m.expr {
            let j = comp
                .variables
                .iter()
                .position(|v| *v == t.var)
                .ok_or_else(|| anyhow!("Cannot find variable {} in the companion VAR", t.var))?;
            ids.push(j + comp.is_constant as usize);
        }
        let mut alpha = Mat::zeros(1, n);
        for (t, &j) in m.expr.iter().zip(&ids) {
            alpha[(0, j)] = t.constant;
        }
        // Dynare 7.2's update_parameters.m assigns `expr.params` -- the (1-based) *index* of the
        // parameter -- and not its value to alpha; reproduced as is.
        for (t, &j) in m.expr.iter().zip(&ids) {
            if let Some(pn) = &t.param {
                alpha[(0, j)] = (pidx(pn)? + 1) as f64;
            }
        }
        let shift = |v: Mat| if ts < 0 { rowmul(&v, &mpow(c, (-ts) as usize)) } else { v };
        let dc = c * beta;
        let parameters = match m.horizon {
            Horizon::Single(h) => {
                if h == 0 {
                    bail!("horizon must be an integer scalar or an integer vector with two elements.");
                }
                shift(rowmul(&alpha, &mpow(&dc, h)))
            }
            Horizon::Range(a, b) => {
                let id = Mat::identity(n, n);
                if (b.is_finite() && (b.fract() != 0.0 || b as usize <= a)) || b < 0.0 {
                    bail!("horizon must be an integer scalar or an integer vector with two elements.");
                }
                if a == 0 && b.is_infinite() {
                    shift(rdiv(&alpha, &(&id - &dc))?)
                } else if b.is_infinite() {
                    let mut tmp1 = Mat::zeros(n, n);
                    for h in 0..a {
                        tmp1 += mpow(&dc, h);
                    }
                    let tmp1 = shift(rowmul(&alpha, &tmp1));
                    let tmp2 = shift(rdiv(&alpha, &(&id - &dc))?);
                    tmp2 - tmp1
                } else {
                    let mut tmp = Mat::zeros(n, n);
                    for h in a..=(b as usize) {
                        tmp += mpow(&dc, h);
                    }
                    shift(rowmul(&alpha, &tmp))
                }
            }
        };
        let vals: Vec<f64> = parameters.iter().copied().collect();
        let mut out = Vec::new();
        if kind == "var" && !comp.is_constant {
            out.push((m.param_names[0].clone(), 0.0));
            for (k, v) in vals.iter().enumerate() {
                out.push((m.param_names[k + 1].clone(), *v));
            }
        } else {
            if vals.len() != m.param_names.len() {
                bail!(
                    "var_expectation_model {ve}: {} parameters expected, {} computed",
                    m.param_names.len(),
                    vals.len()
                );
            }
            for (k, v) in vals.iter().enumerate() {
                out.push((m.param_names[k].clone(), *v));
            }
        }
        Ok(out)
    }
}

/// Evaluates a coefficient expression with the calibration `params`.
pub fn eval_coef(a: &Ast, names: &[String], params: &[f64]) -> Result<f64> {
    eval_ast(a, &|n: &str, _l: i32, _loc: &Loc| {
        if let Some(i) = names.iter().position(|x| x == n) {
            return Ok(params[i]);
        }
        match n {
            "pi" => Ok(std::f64::consts::PI),
            _ => Err(anyhow!("unknown symbol '{n}'")),
        }
    })
}

// ---------------------------------------------------------------- equation analysis

type Kind<'a> = &'a dyn Fn(&str) -> Option<SymbolKind>;

fn mulc(coef: &Ast, a: &Ast) -> Ast {
    Ast::Bin(BinaryOp::Mul, Box::new(coef.clone()), Box::new(a.clone()))
}

fn is_const(a: &Ast, kind: Kind) -> Result<bool> {
    Ok(match a {
        Ast::Num(_) => true,
        Ast::Var { name, .. } => match kind(name) {
            Some(SymbolKind::Param) | None => true,
            Some(SymbolKind::Endo) | Some(SymbolKind::Exo) => false,
            Some(_) => bail!("'{name}' cannot appear in an equation of a VAR model"),
        },
        Ast::Un(_, x) => is_const(x, kind)?,
        Ast::Bin(_, x, y) => is_const(x, kind)? && is_const(y, kind)?,
        Ast::Op(_, v) => {
            for x in v {
                if !is_const(x, kind)? {
                    return Ok(false);
                }
            }
            true
        }
    })
}

/// The variables of the auxiliary models: an endogenous variable, or `diff` / a unary function
/// (which the preprocessor replaces by an auxiliary variable) of one. Returns the expression at
/// `t`, its key (`diff(log(x))`) and the lag; None for any other expression.
pub fn key_of(a: &Ast, kind: Kind) -> Result<Option<(Ast, String, usize)>> {
    fn walk(a: &Ast, kind: Kind) -> Option<(Ast, String, i32)> {
        match a {
            Ast::Var { name, lag, loc } if kind(name) == Some(SymbolKind::Endo) => {
                Some((Ast::Var { name: name.clone(), lag: 0, loc: loc.clone() }, name.clone(), *lag))
            }
            Ast::Op(SpecialOp::Diff, v) if v.len() == 1 => {
                let (e, k, l) = walk(&v[0], kind)?;
                Some((Ast::Op(SpecialOp::Diff, vec![e]), format!("diff({k})"), l))
            }
            Ast::Un(op, x) if *op != UnaryOp::Neg => {
                let (e, k, l) = walk(x, kind)?;
                Some((Ast::Un(*op, Box::new(e)), format!("{}({k})", op.name()), l))
            }
            _ => None,
        }
    }
    match walk(a, kind) {
        Some((_, k, l)) if l > 0 => bail!("leads are not allowed in a VAR model ({k}(+{l}))"),
        Some((e, k, l)) => Ok(Some((e, k, (-l) as usize))),
        None => Ok(None),
    }
}

/// `a` shifted by `-lag` (an expression of [`key_of`] at `t`).
pub fn lagged(a: &Ast, lag: usize) -> Ast {
    match a {
        Ast::Var { name, lag: l, loc } => Ast::Var { name: name.clone(), lag: l - lag as i32, loc: loc.clone() },
        Ast::Op(o, v) => Ast::Op(*o, v.iter().map(|x| lagged(x, lag)).collect()),
        Ast::Un(o, x) => Ast::Un(*o, Box::new(lagged(x, lag))),
        Ast::Bin(o, x, y) => Ast::Bin(*o, Box::new(lagged(x, lag)), Box::new(lagged(y, lag))),
        Ast::Num(_) => a.clone(),
    }
}

fn lin(a: &Ast, coef: &Ast, kind: Kind, tag: &str, out: &mut EqSpec) -> Result<()> {
    if is_const(a, kind)? {
        out.consts.push(mulc(coef, a));
        return Ok(());
    }
    if let Some((_, key, lag)) = key_of(a, kind).map_err(|e| anyhow!("equation '{tag}': {e}"))? {
        out.terms.push(Term { var: key, lag, coef: coef.clone() });
        return Ok(());
    }
    match a {
        // exogenous variables are the innovations of the model
        Ast::Var { .. } => {}
        Ast::Un(UnaryOp::Neg, x) => lin(x, &mulc(coef, &Ast::Num(-1.0)), kind, tag, out)?,
        Ast::Bin(BinaryOp::Add, x, y) => {
            lin(x, coef, kind, tag, out)?;
            lin(y, coef, kind, tag, out)?;
        }
        Ast::Bin(BinaryOp::Sub, x, y) => {
            lin(x, coef, kind, tag, out)?;
            lin(y, &mulc(coef, &Ast::Num(-1.0)), kind, tag, out)?;
        }
        Ast::Bin(BinaryOp::Mul, x, y) => {
            if is_const(x, kind)? {
                lin(y, &mulc(coef, x), kind, tag, out)?;
            } else if is_const(y, kind)? {
                lin(x, &mulc(coef, y), kind, tag, out)?;
            } else {
                bail!("equation '{tag}' is not linear in the variables");
            }
        }
        Ast::Bin(BinaryOp::Div, x, y) if is_const(y, kind)? => {
            lin(x, &Ast::Bin(BinaryOp::Div, Box::new(coef.clone()), y.clone()), kind, tag, out)?
        }
        Ast::Op(SpecialOp::Diff, _) => bail!("equation '{tag}': only diff(x(-k)) of a variable is supported"),
        _ => bail!("equation '{tag}' is not linear in the variables"),
    }
    Ok(())
}

/// Linear structure of the equation `lhs = rhs` tagged `tag`. The LHS is a variable or an
/// expression of one ([`key_of`]) at `t`; with `allow_expr_lhs` false, a variable.
pub fn eq_spec(tag: &str, lhs: &Ast, rhs: &Ast, kind: Kind, allow_expr_lhs: bool) -> Result<EqSpec> {
    let bad = || anyhow!("equation '{tag}': the LHS must be a single endogenous variable at time t");
    let (lhs_expr, key, lag) = key_of(lhs, kind).map_err(|e| anyhow!("equation '{tag}': {e}"))?.ok_or_else(bad)?;
    if lag != 0 || (!allow_expr_lhs && !matches!(lhs_expr, Ast::Var { .. })) {
        return Err(bad());
    }
    fn count_diffs(a: &Ast) -> usize {
        match a {
            Ast::Op(SpecialOp::Diff, v) => 1 + count_diffs(&v[0]),
            Ast::Op(_, v) => v.iter().map(count_diffs).sum(),
            Ast::Un(_, x) => count_diffs(x),
            _ => 0,
        }
    }
    // `undiff`: the argument of the outermost diff
    fn undiff(a: &Ast) -> Option<Ast> {
        match a {
            Ast::Op(SpecialOp::Diff, v) => Some(v[0].clone()),
            Ast::Un(_, x) => undiff(x),
            _ => None,
        }
    }
    let undiff = undiff(&lhs_expr).map(|u| {
        let k = key_of(&u, kind).ok().flatten().map(|x| x.1).unwrap_or_default();
        (k, u)
    });
    let mut e = EqSpec {
        tag: tag.to_string(),
        lhs: key.clone(),
        lhs_diff: count_diffs(&lhs_expr) > 0,
        lhs_expr,
        undiff_name: undiff.as_ref().map(|u| u.0.clone()).unwrap_or_default(),
        undiff,
        lhs_name: key,
        terms: Vec::new(),
        consts: Vec::new(),
    };
    lin(rhs, &Ast::Num(1.0), kind, tag, &mut e)?;
    Ok(e)
}

/// `expression` option of `var_expectation_model`.
pub fn expr_terms(a: &Ast, kind: Kind) -> Result<Vec<ExprTerm>> {
    // scalar factor: (numeric value, parameter)
    fn scalar(a: &Ast, kind: Kind) -> Option<(f64, Option<String>)> {
        match a {
            Ast::Num(v) => Some((*v, None)),
            Ast::Var { name, .. } if kind(name) == Some(SymbolKind::Param) => Some((1.0, Some(name.clone()))),
            Ast::Un(UnaryOp::Neg, x) => scalar(x, kind).map(|(v, p)| (-v, p)),
            Ast::Bin(BinaryOp::Mul, x, y) => {
                let ((v, p), (w, q)) = (scalar(x, kind)?, scalar(y, kind)?);
                if p.is_some() && q.is_some() { None } else { Some((v * w, p.or(q))) }
            }
            Ast::Bin(BinaryOp::Div, x, y) => match (scalar(x, kind)?, scalar(y, kind)?) {
                ((v, p), (w, None)) => Some((v / w, p)),
                _ => None,
            },
            _ => None,
        }
    }
    fn walk(a: &Ast, mult: f64, param: Option<String>, kind: Kind, out: &mut Vec<ExprTerm>) -> Result<()> {
        let bad =
            || anyhow!("var_expectation_model: the expression must be a linear combination of variables at time t");
        match a {
            Ast::Var { name, lag: 0, .. } if kind(name) == Some(SymbolKind::Endo) => {
                out.push(ExprTerm { var: name.clone(), param, constant: mult });
            }
            Ast::Un(UnaryOp::Neg, x) => walk(x, -mult, param, kind, out)?,
            // `diff(x)`, `diff(log(x))`, ...: the auxiliary variable the preprocessor substitutes
            // for it (`substituteDiff` / `substituteUnaryOpsInExpression`), renamed once the model
            // is built (`resolve_names`)
            Ast::Op(SpecialOp::Diff, _) | Ast::Un(..) => match key_of(a, kind).ok().flatten() {
                Some((_, key, 0)) => out.push(ExprTerm { var: key, param, constant: mult }),
                _ => return Err(bad()),
            },
            Ast::Bin(BinaryOp::Add, x, y) => {
                walk(x, mult, param.clone(), kind, out)?;
                walk(y, mult, param, kind, out)?;
            }
            Ast::Bin(BinaryOp::Sub, x, y) => {
                walk(x, mult, param.clone(), kind, out)?;
                walk(y, -mult, param, kind, out)?;
            }
            Ast::Bin(BinaryOp::Mul, x, y) => {
                let (s, other) = match (scalar(x, kind), scalar(y, kind)) {
                    (Some(s), _) => (s, y),
                    (None, Some(s)) => (s, x),
                    _ => return Err(bad()),
                };
                if s.1.is_some() && param.is_some() {
                    bail!("var_expectation_model: an expression term can have at most one parameter");
                }
                walk(other, mult * s.0, param.or(s.1), kind, out)?;
            }
            Ast::Bin(BinaryOp::Div, x, y) => match scalar(y, kind) {
                Some((w, None)) => walk(x, mult / w, param, kind, out)?,
                _ => return Err(bad()),
            },
            _ => return Err(bad()),
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(a, 1.0, None, kind, &mut out)?;
    Ok(out)
}

// ---------------------------------------------------------------- commands

pub const INITIALIZE: &str = "var_expectation.initialize";
pub const UPDATE: &str = "var_expectation.update";
pub const PRINT: &str = "var_expectation.print";
pub const GET_COMPANION: &str = "get_companion_matrix";
// ---- pac ----
pub const PAC_INITIALIZE: &str = "pac.initialize";
pub const PAC_UPDATE: &str = "pac.update.expectation";
pub const PAC_MCE: &str = "pac.mce.parameters";
pub const PAC_PRINT: &str = "pac.print";
pub const PAC_BGP_SET: &str = "pac.bgp.set";
/// PAC estimation needs MATLAB workspace objects (dseries, structures): not runnable from a
/// .mod file, see [`pac_estimate`].
pub const PAC_ESTIMATE: &[&str] = &["pac.estimate.nls", "pac.estimate.iterative_ols"];
/// `X = simul_backward_model(init, n[, innovations]);` (options `out`, `init`, `periods`,
/// `innovations`: names of runtime or workspace dseries).
pub const SIMUL_BACKWARD: &str = "simul_backward_model";
/// `Y = X;` with `X` a runtime dseries (options `out`, `src`).
pub const DS_COPY: &str = "dseries.copy";
/// `X.f = dseries(v[, date]);` on a runtime dseries (options `var`, `field`, `value`, `date`).
pub const DS_SET: &str = "dseries.set_field";

/// A runtime dseries, or a dseries of the MATLAB workspace evaluated by the parser.
fn series_named(mf: &crate::modfile::ModFile, n: &str) -> Result<backward::Series> {
    if let Some(s) = mf.varmodels.series.get(n) {
        return Ok(s.clone());
    }
    match mf.q1.workspace.vars.get(n) {
        Some(crate::deterministic::workspace::WsValue::Dseries { init, names, data }) => {
            Ok(backward::Series { init: *init, names: names.clone(), data: data.clone() })
        }
        Some(_) => bail!("{n} is not a dseries object"),
        None => bail!("unknown dseries {n} (dynare-rust: it must be assigned in the .mod file)"),
    }
}

fn run_series(mf: &mut crate::modfile::ModFile, name: &str, o: &Options) -> Result<()> {
    use crate::modfile::opt;
    let s = |k: &str| match opt(o, k) {
        Some(OptValue::Str(v)) => Some(v.clone()),
        _ => None,
    };
    let out_dir = s("out_dir").map(std::path::PathBuf::from);
    match name {
        SIMUL_BACKWARD => {
            let out = s("out").ok_or_else(|| anyhow!("{name}: no output"))?;
            let init = s("init").map(|n| series_named(mf, &n)).transpose()?;
            let innov = s("innovations").map(|n| series_named(mf, &n)).transpose()?;
            let n = match opt(o, "periods") {
                Some(OptValue::Num(v)) => *v as usize,
                _ => bail!("{name}: no sample size"),
            };
            let sim = backward::simul_backward_model(
                &mf.model,
                &mf.model.params,
                &mf.sigma_e,
                init.as_ref(),
                n,
                innov.as_ref(),
                &mf.histval,
                // options_.solve_algo as the commands so far left it (default 4)
                mf.options.solve_algo.map_or(4, |v| v as u8),
            )?;
            println!(
                "{name}: {n} periods simulated ({}, {} to {}).",
                out,
                sim.init.label(),
                sim.init.add(sim.data.nrows() as i64 - 1).label()
            );
            if let Some(d) = out_dir {
                std::fs::create_dir_all(&d)?;
                std::fs::write(d.join(format!("{}_{out}.csv", mf_fname(o))), sim.to_csv())?;
            }
            mf.varmodels.series.insert(out, sim);
        }
        DS_COPY => {
            let src = s("src").unwrap_or_default();
            let v = series_named(mf, &src)?;
            mf.varmodels.series.insert(s("out").unwrap_or_default(), v);
        }
        _ => {
            let var = s("var").unwrap_or_default();
            let field = s("field").unwrap_or_default();
            let v = match opt(o, "value") {
                Some(OptValue::Num(v)) => *v,
                _ => f64::NAN,
            };
            let date = s("date")
                .map(|d| crate::dates::Date::parse(&d).ok_or_else(|| anyhow!("{d} is not a date")))
                .transpose()?;
            let mut x = series_named(mf, &var)?;
            x.set_field(&field, v, date)?;
            mf.varmodels.series.insert(var, x);
        }
    }
    Ok(())
}

fn mf_fname(o: &Options) -> String {
    match crate::modfile::opt(o, "fname") {
        Some(OptValue::Str(v)) => v.clone(),
        _ => "dynare".into(),
    }
}

/// `dyn_table.m`: the table printed by `pac.estimate.nls` (columns 16 wide, `%16.5g` cells).
fn dyn_table(
    title: &str,
    preamble: &[String],
    afterward: &[String],
    rows: &[String],
    cols: &[&str],
    indent: usize,
    data: &[Vec<f64>],
) -> String {
    use crate::analysis::table::fmt_g;
    let maxrow = rows.iter().map(|r| r.chars().count()).max().unwrap_or(0);
    let mut colrow = " ".repeat(2 * indent + maxrow);
    let mut colrow2 = colrow.clone();
    for (i, c) in cols.iter().enumerate() {
        let n = c.chars().count();
        let fw = 16usize;
        if n < fw {
            let l = (fw - n % fw) / 2;
            let r = (fw - n % fw).div_ceil(2);
            colrow += &format!("{}{c}{}", " ".repeat(l), " ".repeat(r));
            colrow2 += &"_".repeat(fw);
        } else {
            colrow += c;
            colrow2 += &"_".repeat(n);
        }
        if i + 1 != cols.len() {
            colrow += &" ".repeat(indent);
            colrow2 += &" ".repeat(indent);
        }
    }
    let sp = " ".repeat(indent);
    let mut out = String::from("\n\n");
    if title.chars().count() >= colrow.chars().count() {
        out += &format!("{title}\n\n");
    } else {
        out += &format!("{}{title}\n\n", " ".repeat((colrow.chars().count() + indent - title.chars().count()) / 2));
    }
    for p in preamble {
        out += &format!("{sp}{p}\n");
    }
    out += &format!("{colrow}\n{colrow2}\n\n");
    for (r, row) in rows.iter().zip(data) {
        out += &format!("    {r:<maxrow$}");
        for &v in row {
            out += &format!("{sp}{}", fmt_g(v, 16, 5));
        }
        out.push('\n');
    }
    out.push('\n');
    for a in afterward {
        out += &format!("{sp}{a}\n");
    }
    out += &format!("{}\n\n", "_".repeat(colrow2.chars().count()));
    out
}

/// `pac.estimate.nls` / `pac.estimate.iterative_ols` (options built by the parser, plus `dir`, the
/// folder of the .mod file, added by the driver): loads the dseries file, estimates and writes the
/// estimates and the `h` parameters into the model parameters.
fn run_estimate(mf: &mut crate::modfile::ModFile, name: &str, o: &Options) -> Result<()> {
    use crate::modfile::opt;
    let s = |k: &str| match opt(o, k) {
        Some(OptValue::Str(v)) => Some(v.clone()),
        _ => None,
    };
    let eq = s("eq").ok_or_else(|| anyhow!("{name}: no equation"))?;
    let guess: Vec<(String, f64)> = o
        .iter()
        .filter_map(|(k, v)| match (k.strip_prefix("param:"), v) {
            (Some(n), OptValue::Num(x)) => Some((n.to_string(), *x)),
            _ => None,
        })
        .collect();
    // the data: a dseries computed by an earlier command, or a file
    let (file, dates, tnames, values): (String, Option<Vec<String>>, Vec<String>, Mat) = match s("data_var") {
        Some(v) => {
            let x = mf.varmodels.series.get(&v).ok_or_else(|| anyhow!("{name}: unknown dseries {v}"))?;
            (
                v.clone(),
                Some((0..x.data.nrows()).map(|t| x.init.add(t as i64).label()).collect()),
                x.names.clone(),
                x.data.clone(),
            )
        }
        None => {
            let file = s("file").ok_or_else(|| {
                anyhow!("{name}: the data must come from a dseries file, e.g. `edata = dseries('data.csv');`")
            })?;
            let dir = std::path::PathBuf::from(s("dir").unwrap_or_else(|| ".".into()));
            let path = crate::driver::resolve_datafile(&dir, &file)?;
            let table = crate::data::Table::read(&path, &Default::default())?;
            (file, table.dates, table.names, table.values)
        }
    };
    let nobs = values.nrows();
    let row = |d: &str| -> Result<usize> {
        let ds = dates.as_ref().ok_or_else(|| anyhow!("{name}: {file} has no dates column, cannot locate {d}"))?;
        let want = crate::dates::Date::parse(d.trim());
        ds.iter()
            .position(|x| {
                x.trim().eq_ignore_ascii_case(d.trim())
                    || (want.is_some() && crate::dates::Date::parse(x.trim()) == want)
            })
            .ok_or_else(|| anyhow!("{name}: date {d} is not in {file}"))
    };
    let offset = |k: &str| match opt(o, k) {
        Some(OptValue::Num(v)) => *v as i64,
        _ => 0,
    };
    let dated = |d: Option<String>, k: &str, default: usize| -> Result<usize> {
        let Some(d) = d else { return Ok(default) };
        let r = row(&d)? as i64 + offset(k);
        if r < 0 || r >= nobs as i64 {
            bail!("{name}: {d}{:+} is outside the sample of {file}", offset(k));
        }
        Ok(r as usize)
    };
    let first = dated(s("first"), "first_offset", 0)?;
    let last = dated(s("last"), "last_offset", nobs.saturating_sub(1))?;
    if last < first {
        bail!("{name}: the estimation range is empty");
    }
    // every series of the data, observed exogenous regressors included; the series set to NaN
    // (`edata.ez = dseries(NaN, ...)`) are the residual of the equation
    let nan: Vec<&String> = o
        .iter()
        .filter_map(|(k, v)| if let ("nan", OptValue::Str(f)) = (k.as_str(), v) { Some(f) } else { None })
        .collect();
    let mut names: Vec<String> = tnames.clone();
    let mut cols: Vec<Vec<f64>> =
        (0..tnames.len())
            .map(|j| {
                if nan.contains(&&tnames[j]) {
                    vec![f64::NAN; nobs]
                } else {
                    (0..nobs).map(|i| values[(i, j)]).collect()
                }
            })
            .collect();
    for f in nan {
        if !names.contains(f) {
            names.push(f.clone());
            cols.push(vec![f64::NAN; nobs]);
        }
    }
    let data = pac_estimate::PacData::new(names, cols).with_exo(mf.model.exo_names.clone());
    let opt_args: Vec<String> = o
        .iter()
        .filter_map(
            |(k, v)| if let ("optimizer_arg", OptValue::Str(f)) = (k.as_str(), v) { Some(f.clone()) } else { None },
        )
        .collect();
    let mut params = mf.model.params.clone();
    let names = mf.model.param_names.clone();
    let est = if name == "pac.estimate.nls" {
        let opt_name = s("optimizer");
        pac_estimate::estimate_nls_with(
            &mf.varmodels,
            &eq,
            &names,
            &mut params,
            &guess,
            &data,
            (first, last),
            opt_name.as_deref(),
            &opt_args,
        )?
    } else {
        pac_estimate::estimate_iterative_ols(&mf.varmodels, &eq, &names, &mut params, &guess, &data, (first, last))?
    };
    if name == "pac.estimate.nls" && !matches!(opt(o, "noprint"), Some(OptValue::Num(v)) if *v != 0.0) {
        // nls.m: dyn_table('NLS Estimation of equation ...', ...)
        let lhs_var = mf.varmodels.pacs.iter().find(|p| p.eq_name == eq).map(|p| p.lhs_var.clone()).unwrap_or_default();
        let label = |r: usize| match &dates {
            Some(ds) => crate::dates::Date::parse(ds[r].trim()).map_or_else(|| ds[r].trim().to_string(), |d| d.label()),
            None => (r + 1).to_string(),
        };
        let sd: Vec<f64> = match &est.covariance {
            Some(c) => (0..c.nrows()).map(|i| c[(i, i)].sqrt()).collect(),
            None => vec![f64::NAN; est.estimator.len()],
        };
        print!(
            "{}",
            dyn_table(
                &format!("NLS Estimation of equation '{eq}'"),
                &[
                    format!("Dependent Variable: diff({lhs_var})"),
                    format!("Observations: {} from {} to {}\n", last - first + 1, label(first), label(last)),
                ],
                &[format!("R^2: {:.6}", est.r2), format!("s^2: {:.6}", est.s2)],
                &est.parnames,
                &["Estimates", "t-statistic", "Std. Error"],
                4,
                &est.estimator
                    .iter()
                    .zip(&est.student)
                    .zip(&sd)
                    .map(|((a, b), c)| vec![*a, *b, *c])
                    .collect::<Vec<_>>(),
            )
        );
    } else if name != "pac.estimate.nls" {
        println!("{name}({eq}): SSR = {:.8}, R2 = {:.6}", est.ssr, est.r2);
        for (n, v) in est.parnames.iter().zip(&est.estimator) {
            println!("  {n} = {v:.8}");
        }
    }
    // oo_.pac.<model>
    if let Some(OptValue::Str(d)) = opt(o, "out_dir") {
        let d = std::path::PathBuf::from(d);
        let pm = mf.varmodels.pacs.iter().find(|p| p.eq_name == eq).map(|p| p.name.clone()).unwrap_or_default();
        let cov: Option<Vec<Vec<f64>>> =
            est.covariance.as_ref().map(|c| (0..c.nrows()).map(|i| c.row(i).iter().copied().collect()).collect());
        let j = serde_json::json!({
            "equation": eq, "parnames": est.parnames, "estimator": est.estimator, "student": est.student,
            "covariance": cov, "ssr": est.ssr, "s2": est.s2, "R2": est.r2, "iterations": est.iterations,
        });
        std::fs::create_dir_all(&d)?;
        std::fs::write(d.join(format!("{}_pac_{pm}.json", mf_fname(o))), serde_json::to_string_pretty(&j)?)?;
    }
    mf.model.params = params;
    Ok(())
}

/// Commands run by [`run`].
pub fn handles(name: &str) -> bool {
    matches!(name, INITIALIZE | UPDATE | PRINT | GET_COMPANION | PAC_INITIALIZE | PAC_UPDATE | PAC_MCE | PAC_PRINT | PAC_BGP_SET) // ---- pac ----
        || PAC_ESTIMATE.contains(&name)
        || matches!(name, SIMUL_BACKWARD | DS_COPY | DS_SET)
}

/// `var_expectation.initialize('m')`, `var_expectation.update('m')`, `get_companion_matrix('m')`.
pub fn run(mf: &mut crate::modfile::ModFile, name: &str, o: &Options) -> Result<()> {
    // ---- pac ----
    if PAC_ESTIMATE.contains(&name) {
        return run_estimate(mf, name, o);
    }
    if matches!(name, SIMUL_BACKWARD | DS_COPY | DS_SET) {
        return run_series(mf, name, o);
    }
    let Some((arg, OptValue::Flag)) = o.first() else { bail!("{name}: expects the name of a model") };
    let names = mf.model.param_names.clone();
    let params = mf.model.params.clone();
    match name {
        INITIALIZE => mf.varmodels.initialize(arg, &names, &params),
        // ---- pac ----
        PAC_INITIALIZE => mf.varmodels.pac_initialize(arg, &names, &params),
        PAC_BGP_SET => {
            use crate::modfile::opt;
            let s = |k| match opt(o, k) {
                Some(OptValue::Str(s)) => s.clone(),
                _ => String::new(),
            };
            let (eq, var) = (s("eq"), s("var"));
            let value = match (opt(o, "logical"), opt(o, "value")) {
                (Some(OptValue::Num(v)), _) => pac::PacBgp::Bool(*v != 0.0),
                (_, Some(OptValue::Num(v))) => pac::PacBgp::Num(*v),
                _ => bail!("{name}: expects a value"),
            };
            let pm = mf.varmodels.pacs.iter_mut().find(|p| p.name == *arg);
            let pm = pm.ok_or_else(|| anyhow!("Reference to non-existent field '{arg}'."))?;
            if !mf.model.endo_names.contains(&var) && !mf.model.exo_names.contains(&var) {
                bail!("Endogenous/Exogenous variable {var} is unknown.");
            }
            pm.bgp_set(&eq, &var, value);
            Ok(())
        }
        PAC_UPDATE | PAC_MCE => {
            let vals = mf.varmodels.pac_update_values(arg, &names, &params)?;
            for (n, v) in vals {
                let k = names.iter().position(|x| *x == n).ok_or_else(|| anyhow!("unknown parameter {n}"))?;
                mf.model.params[k] = v;
            }
            Ok(())
        }
        PAC_PRINT | PRINT => {
            use crate::modfile::opt;
            let withcal = !matches!(opt(o, "withcalibration"), Some(OptValue::Num(v)) if *v == 0.0);
            let fname = mf_fname(o);
            let (files, console) = if name == PRINT {
                mf.varmodels.print_var_expectation(arg, &names, &params, &fname, withcal)?
            } else {
                let p = mf.varmodels.pac(arg).ok_or_else(|| anyhow!("PAC_EXPECTATION_MODEL {arg} is not defined."))?;
                p.print(&mf.varmodels, &names, &params, &fname, withcal)?
            };
            let dir = match opt(o, "dir") {
                Some(OptValue::Str(d)) => std::path::PathBuf::from(d),
                _ => std::path::PathBuf::from("."),
            };
            for (f, text) in files {
                let path = dir.join(&f);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(path, text)?;
            }
            print!("{console}");
            Ok(())
        }
        GET_COMPANION => mf.varmodels.get_companion_matrix(arg, &names, &params).map(|_| ()),
        _ => {
            let vals = mf.varmodels.update_values(arg, &names, &params)?;
            for (n, v) in vals {
                let k = names.iter().position(|x| *x == n).ok_or_else(|| anyhow!("unknown parameter {n}"))?;
                mf.model.params[k] = v;
            }
            Ok(())
        }
    }
}
