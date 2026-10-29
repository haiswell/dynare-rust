// SPDX-License-Identifier: GPL-3.0-or-later
//! Estimation of a PAC equation: `pac.estimate.nls` (Gauss-Newton on the sum of squared
//! residuals, `gauss_newton.m`) and `pac.estimate.iterative_ols` (Brayton et al., 2000).
//!
//! In Dynare these commands take a `dseries` and a structure of initial values from the MATLAB
//! workspace; here the data is a [`PacData`] (named columns of equal length, the rows are the
//! periods) and `range` the first and last rows (inclusive) of the estimation sample. The
//! equation is the one decomposed by the parser (`PacModel::eq_lhs/eq_rhs`, with the
//! auxiliary variables of the PAC expectation substituted by their definitions), so the data
//! only needs the variables of the model (not the auxiliary ones); the residual of the
//! equation (the exogenous variable that is not in the data, or has only NaN over the sample,
//! `pac.estimate.init`) is zero. Every other variable of the equation, observed exogenous
//! regressors included, is read from the data; a missing one is an error.
//!
//! Every optimizer name of `pac.estimate.nls` is mapped to the Gauss-Newton iteration of
//! `GaussNewton` (projected on the bounds that `nls.m` gives the optimizer); an unknown name
//! is an error, as in `nls.m`.

use super::VarModels;
use super::pac::PacModel;
use crate::linalg::Mat;
use crate::modfile::{Ast, SpecialOp};
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;

/// Named series (same length), the period of row `t` is the same in all of them.
#[derive(Clone, Debug, Default)]
pub struct PacData {
    pub names: Vec<String>,
    pub cols: Vec<Vec<f64>>,
    /// `M_.exo_names`: the candidates for the residual of the equation. Empty: every variable
    /// of the equation that is not in the data (or only NaN) is a candidate.
    pub exo: Vec<String>,
}

impl PacData {
    pub fn new(names: Vec<String>, cols: Vec<Vec<f64>>) -> Self {
        PacData { names, cols, exo: Vec::new() }
    }

    /// [`PacData::new`] with the exogenous variables of the model (`M_.exo_names`).
    pub fn with_exo(mut self, exo: Vec<String>) -> Self {
        self.exo = exo;
        self
    }

    fn col(&self, n: &str) -> Option<&Vec<f64>> {
        self.names.iter().position(|x| x == n).map(|i| &self.cols[i])
    }
}

/// `oo_.pac.<name>` after an estimation.
#[derive(Clone, Debug)]
pub struct PacEstimate {
    pub parnames: Vec<String>,
    pub estimator: Vec<f64>,
    /// NLS only.
    pub covariance: Option<Mat>,
    pub student: Vec<f64>,
    pub ssr: f64,
    pub s2: f64,
    pub r2: f64,
    pub residual: Vec<f64>,
    pub lhs: Vec<f64>,
    /// Number of iterations (iterative OLS).
    pub iterations: usize,
}

struct Ev<'a> {
    names: &'a [String],
    params: &'a [f64],
    data: &'a PacData,
    defs: HashMap<&'a str, &'a Ast>,
    /// The residual of the equation (zero).
    residual: Option<&'a str>,
}

impl Ev<'_> {
    fn ev(&self, a: &Ast, t: isize) -> Result<f64> {
        Ok(match a {
            Ast::Num(v) => *v,
            Ast::Var { name, lag, .. } => {
                if let Some(i) = self.names.iter().position(|x| x == name) {
                    return Ok(self.params[i]);
                }
                if let Some(d) = self.defs.get(name.as_str()) {
                    return self.ev(d, t + *lag as isize);
                }
                if self.residual == Some(name.as_str()) {
                    return Ok(0.0);
                }
                match self.data.col(name) {
                    Some(c) => {
                        let k = t + *lag as isize;
                        if k < 0 || k as usize >= c.len() {
                            bail!(
                                "the sample does not contain enough observations before the first period ({name}({lag}))"
                            );
                        }
                        let v = c[k as usize];
                        if v.is_nan() {
                            bail!("Some variable values are missing in the database ({name}).");
                        }
                        v
                    }
                    None => bail!("Variable {name} is not in the database."),
                }
            }
            Ast::Un(op, x) => op.apply(self.ev(x, t)?),
            Ast::Bin(op, x, y) => op.apply(self.ev(x, t)?, self.ev(y, t)?),
            Ast::Op(SpecialOp::Diff, v) => self.ev(&v[0], t)? - self.ev(&v[0], t - 1)?,
            Ast::Op(o, _) => bail!("{} is not supported in a PAC equation", o.name()),
        })
    }
}

/// Variables of the PAC equation (not the parameters, nor the auxiliary variables of `defs`).
fn eq_vars(pac: &PacModel, names: &[String]) -> Vec<String> {
    fn walk(a: &Ast, out: &mut Vec<String>) {
        match a {
            Ast::Var { name, .. } => {
                if !out.contains(name) {
                    out.push(name.clone())
                }
            }
            Ast::Un(_, x) => walk(x, out),
            Ast::Bin(_, x, y) => {
                walk(x, out);
                walk(y, out)
            }
            Ast::Op(_, v) => v.iter().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(&pac.eq_lhs, &mut out);
    walk(&pac.eq_rhs, &mut out);
    for (_, d) in &pac.defs {
        walk(d, &mut out);
    }
    out.retain(|n| !names.contains(n) && !pac.defs.iter().any(|(d, _)| d == n));
    out
}

/// `pac.estimate.init`: the residual of the equation, the exogenous variable that is not
/// observed (not in the data, or only NaN over `range`). `required`: estimation, where there
/// must be exactly one.
fn residual_name(
    pac: &PacModel,
    names: &[String],
    data: &PacData,
    range: (usize, usize),
    required: bool,
) -> Result<Option<String>> {
    let vars = eq_vars(pac, names);
    let unobserved: Vec<&String> = vars
        .iter()
        .filter(|n| data.exo.is_empty() || data.exo.contains(n))
        .filter(|n| match data.col(n) {
            None => true,
            Some(c) => {
                c.get(range.0..=range.1.min(c.len().saturating_sub(1))).is_none_or(|s| s.iter().all(|v| v.is_nan()))
            }
        })
        .collect();
    match unobserved.len() {
        0 if required => bail!("There is no residual in this equation, all the exogenous variables are observed."),
        0 => Ok(None),
        1 => Ok(Some(unobserved[0].clone())),
        _ => bail!(
            "It is not allowed to have more than one residual in a PAC equation ({} are not observed).",
            unobserved.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
        ),
    }
}

fn pac_by_eq<'a>(vms: &'a VarModels, eqname: &str) -> Result<&'a PacModel> {
    let p = vms
        .pacs
        .iter()
        .find(|p| p.eq_name == eqname)
        .ok_or_else(|| anyhow!("This is not a PAC equation ({eqname})."))?;
    Ok(p)
}

/// Residuals `lhs - rhs` of the PAC equation over `range` (rows, inclusive) and the left hand
/// side, for the parameters `params` (the `h` parameters are *not* updated here).
fn residuals(
    pac: &PacModel,
    names: &[String],
    params: &[f64],
    data: &PacData,
    range: (usize, usize),
    residual: Option<&str>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let e = Ev { names, params, data, defs: pac.defs.iter().map(|(n, a)| (n.as_str(), a)).collect(), residual };
    let mut r = Vec::new();
    let mut l = Vec::new();
    for t in range.0..=range.1 {
        let lhs = e.ev(&pac.eq_lhs, t as isize)?;
        let rhs = e.ev(&pac.eq_rhs, t as isize)?;
        l.push(lhs);
        r.push(lhs - rhs);
    }
    Ok((r, l))
}

/// Residuals `lhs - rhs` of the PAC equation `eqname` over `range` (rows, inclusive) for the
/// parameters `params` (which must contain the current `h` parameters). The residual of the
/// equation, if not observed (not in `data` or only NaN), counts as zero.
pub fn pac_residuals(
    vms: &VarModels,
    eqname: &str,
    names: &[String],
    params: &[f64],
    data: &PacData,
    range: (usize, usize),
) -> Result<Vec<f64>> {
    let pac = pac_by_eq(vms, eqname)?;
    let rname = residual_name(pac, names, data, range, false)?;
    Ok(residuals(pac, names, params, data, range, rname.as_deref())?.0)
}

fn set_params(
    vms: &VarModels,
    pac: &PacModel,
    names: &[String],
    params: &mut [f64],
    est: &[usize],
    values: &[f64],
) -> Result<()> {
    for (&i, &v) in est.iter().zip(values) {
        params[i] = v;
    }
    let upd = if pac.is_mce() { pac.mce_values(names, params)? } else { pac.update_values(vms, names, params)? };
    for (n, v) in upd {
        let k = names.iter().position(|x| *x == n).ok_or_else(|| anyhow!("unknown parameter {n}"))?;
        params[k] = v;
    }
    Ok(())
}

fn indices(names: &[String], guess: &[(String, f64)]) -> Result<Vec<usize>> {
    guess
        .iter()
        .map(|(n, _)| names.iter().position(|x| x == n).ok_or_else(|| anyhow!("{n} is not a parameter of the model")))
        .collect()
}

fn var_of(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0)
}

/// `pac.estimate.nls(eqname, guess, data, range, algo)`. `names`/`params` are `M_.param_names`
/// / `M_.params` (updated: the estimates and the `h` parameters).
pub fn estimate_nls(
    vms: &VarModels,
    eqname: &str,
    names: &[String],
    params: &mut [f64],
    guess: &[(String, f64)],
    data: &PacData,
    range: (usize, usize),
) -> Result<PacEstimate> {
    estimate_nls_with(vms, eqname, names, params, guess, data, range, None, &[])
}

/// Optimizers known to `nls.m` (in the order of its error message, then `particleswarm`).
pub const NLS_OPTIMIZERS: [&str; 9] = [
    "fmincon",
    "fminunc",
    "csminwel",
    "fminsearch",
    "simplex",
    "annealing",
    "lsqnonlin",
    "GaussNewton",
    "particleswarm",
];

/// Bounds that `nls.m` puts on the error correction parameter for `optimizer` (with the
/// optional arguments `args` that follow it), `None` if unbounded; an unknown name is an error.
/// `lsqnonlin`, `fmincon` and `annealing`: [0, 1], except `lsqnonlin` with
/// `'levenberg-marquardt'` (which does not handle bounds); `particleswarm`: nls.m assigns the
/// lower bound twice (0, then 1), so the bounds are [1, Inf].
pub fn nls_bounds(optimizer: Option<&str>, args: &[String]) -> Result<Option<(f64, f64)>> {
    let Some(o) = optimizer else { return Ok(None) }; // csminwel
    Ok(match o {
        "lsqnonlin" if args.iter().any(|a| a == "levenberg-marquardt") => None,
        "lsqnonlin" | "fmincon" | "annealing" => Some((0.0, 1.0)),
        "particleswarm" => Some((1.0, f64::INFINITY)),
        "GaussNewton" | "fminunc" | "csminwel" | "fminsearch" | "simplex" => None,
        _ => {
            let mut msg = format!("{o} is not an implemented optimization routine.\nAvailable algorithms are:\n");
            for a in &NLS_OPTIMIZERS[..8] {
                msg.push_str(&format!(" - {a}\n"));
            }
            bail!("{msg}")
        }
    })
}

/// [`estimate_nls`] with the optimizer name of `nls.m` and its optional arguments: the error
/// correction parameter is bounded as [`nls_bounds`] says (projected Gauss-Newton here, not
/// the MATLAB algorithms); every known name runs Gauss-Newton.
#[allow(clippy::too_many_arguments)]
pub fn estimate_nls_with(
    vms: &VarModels,
    eqname: &str,
    names: &[String],
    params: &mut [f64],
    guess: &[(String, f64)],
    data: &PacData,
    range: (usize, usize),
    optimizer: Option<&str>,
    opt_args: &[String],
) -> Result<PacEstimate> {
    let pac = pac_by_eq(vms, eqname)?;
    let est = indices(names, guess)?;
    let ec_bounds = nls_bounds(optimizer, opt_args)?;
    if range.0 == 0 || range.1 < range.0 || range.1 >= data.cols.first().map_or(0, |c| c.len()) {
        bail!("invalid estimation range");
    }
    let rname = residual_name(pac, names, data, range, true)?;
    let rname = rname.as_deref();
    let base = params.to_vec();
    let resfun = |p: &[f64]| -> Result<Vec<f64>> {
        let mut q = base.clone();
        set_params(vms, pac, names, &mut q, &est, p)?;
        Ok(residuals(pac, names, &q, data, range, rname)?.0)
    };
    let x0: Vec<f64> = guess.iter().map(|(_, v)| *v).collect();
    let r0 = resfun(&x0)?;
    let s0: f64 = r0.iter().map(|v| v * v).sum();
    if s0.is_nan() || s0.is_infinite() {
        bail!("Cannot evaluate the Sum of Squared Residuals on the initial guess.");
    }
    let bounds: Vec<(f64, f64)> = if let Some(b) = ec_bounds {
        guess.iter().map(|(n, _)| if *n == pac.ec_param { b } else { (f64::NEG_INFINITY, f64::INFINITY) }).collect()
    } else {
        Vec::new()
    };
    let (x1, ssr) = gauss_newton(&resfun, x0, &bounds)?;
    // covariance (Newey-West)
    let (r, jac) = jacobian(&resfun, &x1, 1e-6)?;
    let t = r.len();
    let p = x1.len();
    let tf = t as f64;
    let a = (jac.transpose() * &jac) * (2.0 / tf);
    let mut jr = jac.clone();
    for i in 0..t {
        for j in 0..p {
            jr[(i, j)] *= r[i];
        }
    }
    let mut b = jr.transpose() * &jr;
    let l = tf.powf(0.25).round() as usize;
    for tau in 1..=l {
        let w = 1.0 - tau as f64 / (l as f64 + 1.0);
        let up = jr.rows(tau, t - tau).into_owned();
        let lo = jr.rows(0, t - tau).into_owned();
        b += (up.transpose() * &lo + lo.transpose() * &up) * w;
    }
    let b = b * (4.0 / tf);
    let ainv = a.clone().try_inverse().ok_or_else(|| anyhow!("singular matrix in the NLS covariance"))?;
    let c = (&ainv * b * &ainv) / tf;
    let student: Vec<f64> = (0..p).map(|i| x1[i] / c[(i, i)].sqrt()).collect();
    let (rr, lhs) = residuals_at(vms, pac, names, params, &est, &x1, data, range, rname)?;
    set_params(vms, pac, names, params, &est, &x1)?;
    Ok(PacEstimate {
        parnames: guess.iter().map(|(n, _)| n.clone()).collect(),
        estimator: x1,
        covariance: Some(c),
        student,
        ssr,
        s2: ssr / tf,
        r2: 1.0 - var_of(&rr) / var_of(&lhs),
        residual: rr,
        lhs,
        iterations: 0,
    })
}

#[allow(clippy::too_many_arguments)]
fn residuals_at(
    vms: &VarModels,
    pac: &PacModel,
    names: &[String],
    params: &[f64],
    est: &[usize],
    x: &[f64],
    data: &PacData,
    range: (usize, usize),
    rname: Option<&str>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut q = params.to_vec();
    set_params(vms, pac, names, &mut q, est, x)?;
    residuals(pac, names, &q, data, range, rname)
}

/// `jacobian.m`: forward differences of `r` (note the sign: `(r(x) - r(x+h))/h`).
fn jacobian(f: &dyn Fn(&[f64]) -> Result<Vec<f64>>, x: &[f64], h: f64) -> Result<(Vec<f64>, Mat)> {
    let r = f(x)?;
    let mut j = Mat::zeros(r.len(), x.len());
    for i in 0..x.len() {
        let mut z = x.to_vec();
        z[i] += h;
        let ri = f(&z)?;
        for k in 0..r.len() {
            j[(k, i)] = (r[k] - ri[k]) / h;
        }
    }
    Ok((r, j))
}

/// `optimization/gauss_newton.m`.
fn gauss_newton(
    f: &dyn Fn(&[f64]) -> Result<Vec<f64>>,
    x0: Vec<f64>,
    bounds: &[(f64, f64)],
) -> Result<(Vec<f64>, f64)> {
    let clip = |mut x: Vec<f64>| -> Vec<f64> {
        for (v, b) in x.iter_mut().zip(bounds) {
            *v = v.clamp(b.0, b.1);
        }
        x
    };
    let x0 = clip(x0);
    let max_iter = 100;
    let h = f64::EPSILON.sqrt();
    let xtol = 1e-6;
    let phi = 0.5 * (5f64.sqrt() + 1.0);
    let ssr = |x: &[f64]| -> Result<f64> { Ok(f(x)?.iter().map(|v| v * v).sum()) };
    let mut x0 = x0;
    let mut counter = 0;
    let mut s1;
    loop {
        let (r0, jac) = jacobian(f, &x0, h)?;
        let pinv = jac
            .clone()
            .svd(true, true)
            .pseudo_inverse(f64::EPSILON * jac.nrows().max(jac.ncols()) as f64 * jac.norm())
            .map_err(|e| anyhow!("{e}"))?;
        let d = pinv * Mat::from_column_slice(r0.len(), 1, &r0);
        let d: Vec<f64> = d.iter().copied().collect();
        let x1: Vec<f64> = clip(x0.iter().zip(&d).map(|(a, b)| a + b).collect());
        let r1 = f(&x1)?;
        let s0: f64 = r0.iter().map(|v| v * v).sum();
        s1 = r1.iter().map(|v| v * v).sum();
        let x1 = if s1 > s0 {
            let (mut l1, mut l2) = (0.0, 1.0);
            let mut ll1 = l2 - (l2 - l1) / phi;
            let mut ll2 = l1 + (l2 - l1) / phi;
            while (ll1 - ll2).abs() > 1e-6 {
                let a: Vec<f64> = clip(x0.iter().zip(&d).map(|(a, b)| a + ll1 * b).collect());
                let b: Vec<f64> = clip(x0.iter().zip(&d).map(|(a, b)| a + ll2 * b).collect());
                if ssr(&a)? < ssr(&b)? {
                    l2 = ll2;
                } else {
                    l1 = ll1;
                }
                ll1 = l2 - (l2 - l1) / phi;
                ll2 = l1 + (l2 - l1) / phi;
            }
            let scale = 0.5 * (l1 + l2);
            clip(x0.iter().zip(&d).map(|(a, b)| a + scale * b).collect())
        } else {
            x1
        };
        let noconv = x1.iter().zip(&x0).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) > xtol;
        counter += 1;
        x0 = x1;
        if !noconv || counter > max_iter {
            break;
        }
    }
    Ok((x0, s1))
}

/// `decomposeAdditiveTerms`: the additive terms of `a` with their signs.
fn additive_terms(a: &Ast, sign: f64, out: &mut Vec<(Ast, f64)>) {
    use crate::model::expr::{BinaryOp, UnaryOp};
    match a {
        Ast::Bin(BinaryOp::Add, x, y) => {
            additive_terms(x, sign, out);
            additive_terms(y, sign, out);
        }
        Ast::Bin(BinaryOp::Sub, x, y) => {
            additive_terms(x, sign, out);
            additive_terms(y, -sign, out);
        }
        Ast::Un(UnaryOp::Neg, x) => additive_terms(x, -sign, out),
        _ => out.push((a.clone(), sign)),
    }
}

fn has_name(a: &Ast, n: &str) -> bool {
    match a {
        Ast::Var { name, .. } => name == n,
        Ast::Num(_) => false,
        Ast::Un(_, x) => has_name(x, n),
        Ast::Bin(_, x, y) => has_name(x, n) || has_name(y, n),
        Ast::Op(_, v) => v.iter().any(|x| has_name(x, n)),
    }
}

/// `a` with the parameter `n` replaced by one.
fn with_unit(a: &Ast, n: &str) -> Ast {
    match a {
        Ast::Var { name, .. } if name == n => Ast::Num(1.0),
        Ast::Un(o, x) => Ast::Un(*o, Box::new(with_unit(x, n))),
        Ast::Bin(o, x, y) => Ast::Bin(*o, Box::new(with_unit(x, n)), Box::new(with_unit(y, n))),
        Ast::Op(o, v) => Ast::Op(*o, v.iter().map(|x| with_unit(x, n)).collect()),
        _ => a.clone(),
    }
}

/// The terms `scaling_factor*param*variable` of the PAC equation other than the error
/// correction, autoregressive and PAC expectation terms: `M_.pac.<m>.optim_additive` (in the
/// optimizing part `share*(...)`) and `M_.pac.<m>.additive` (the others), as (terms, sign) per
/// parameter `n`.
fn extra_terms(pac: &PacModel, n: &str) -> (Vec<(Ast, f64)>, Vec<(Ast, f64)>) {
    use crate::model::expr::BinaryOp;
    let mut top = Vec::new();
    additive_terms(&pac.eq_rhs, 1.0, &mut top);
    let is_share = |a: &Ast| matches!(a, Ast::Var { name, .. } if Some(name) == pac.share.as_ref());
    let (mut optim, mut additive) = (Vec::new(), Vec::new());
    for (t, s) in top {
        let inner = match &t {
            Ast::Bin(BinaryOp::Mul, x, y) if is_share(x) => Some(&**y),
            Ast::Bin(BinaryOp::Mul, x, y) if is_share(y) => Some(&**x),
            _ => None,
        };
        match inner {
            Some(i) => {
                let mut v = Vec::new();
                additive_terms(i, s, &mut v);
                optim.extend(v.into_iter().filter(|(a, _)| has_name(a, n)));
            }
            None if has_name(&t, n) => additive.push((t, s)),
            None => {}
        }
    }
    (optim, additive)
}

/// `[~, ia] = setdiff(a, b)`: the positions in `a` of the values not in `b`, sorted by value.
fn setdiff_ia(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut ia: Vec<usize> = (0..a.len()).filter(|&i| !b.contains(&a[i])).collect();
    ia.sort_by_key(|&i| a[i]);
    ia.dedup_by_key(|i| a[*i]);
    ia
}

/// `pac.estimate.iterative_ols(eqname, guess, data, range)` (`+pac/+estimate/iterative_ols.m`):
/// all the error correction and autoregressive parameters of the equation must be in `guess`;
/// the other estimated parameters are those of the share of optimizing agents and of the
/// additive terms (`optim_additive` inside the optimizing part, `additive` outside). The
/// regressors are ordered as in MATLAB: error correction, autoregressive, optimizing additive,
/// additive terms.
pub fn estimate_iterative_ols(
    vms: &VarModels,
    eqname: &str,
    names: &[String],
    params: &mut Vec<f64>,
    guess: &[(String, f64)],
    data: &PacData,
    range: (usize, usize),
) -> Result<PacEstimate> {
    let pac = pac_by_eq(vms, eqname)?;
    if pac.is_mce() {
        bail!("pac.estimate.iterative_ols: model consistent expectations are not supported");
    }
    let mut order = vec![pac.ec_param.clone()];
    order.extend(pac.ar_params.iter().cloned());
    for n in &order {
        if !guess.iter().any(|(g, _)| g == n) {
            bail!("pac.estimate.iterative_ols: the parameter {n} must be in the initial values");
        }
    }
    // ipnames_ and params0
    let ip = indices(names, guess)?;
    let params0: Vec<f64> = guess.iter().map(|(_, v)| *v).collect();
    // share of optimizing agents: estimated (in the initial values) or calibrated
    let share_k = match &pac.share {
        Some(n) => Some(indices(names, &[(n.clone(), 0.0)])?[0]),
        None => None,
    };
    let estimate_share = share_k.is_some_and(|k| ip.contains(&k));
    if let (Some(k), false) = (share_k, estimate_share)
        && !(0.0..=1.0).contains(&params[k])
    {
        bail!("The share of optimizing agents shoud be in (0,1).");
    }
    // the estimated parameters of the additive terms
    let (mut optim, mut additive) = (Vec::new(), Vec::new());
    for (i, (n, _)) in guess.iter().enumerate() {
        if order.contains(n) || Some(ip[i]) == share_k {
            continue;
        }
        let (o, a) = extra_terms(pac, n);
        if !o.is_empty() {
            optim.push((ip[i], o));
        } else if !a.is_empty() {
            additive.push((ip[i], a));
        } else if pac.non_optimizing.as_ref().is_some_and(|e| has_name(e, n)) {
            bail!("The estimation of non optimizing behaviour parameters is not yet allowed.");
        } else {
            bail!(
                "pac.estimate.iterative_ols: only the parameters of the error correction, autoregressive and additive terms and the share of optimizing agents can be estimated ({n})"
            );
        }
    }
    if range.0 == 0 || range.1 < range.0 || range.1 >= data.cols.first().map_or(0, |c| c.len()) {
        bail!("invalid estimation range");
    }
    let rname = residual_name(pac, names, data, range, true)?;
    let rname = rname.as_deref();
    // regressors: error correction, autoregressive, optimizing additive and additive terms
    let defs: HashMap<&str, &Ast> = pac.defs.iter().map(|(n, a)| (n.as_str(), a)).collect();
    let t_n = range.1 - range.0 + 1;
    let n1 = 1 + pac.ar_params.len();
    let ncol = n1 + optim.len() + additive.len();
    let mut xdata = Mat::zeros(t_n, ncol);
    // the parameter of each column
    let mut colpar = indices(names, &order.iter().map(|n| (n.clone(), 0.0)).collect::<Vec<_>>())?;
    colpar.extend(optim.iter().chain(&additive).map(|(k, _)| *k));
    {
        let e = Ev { names, params, data, defs: defs.clone(), residual: rname };
        let lag = |n: &str, k: i32| Ast::Var {
            name: n.to_string(),
            lag: k,
            loc: crate::modfile::lexer::Loc { file: std::sync::Arc::from("<pac>"), line: 0 },
        };
        // (the auxiliary variable of a target `log(x)` is computed from its definition, as
        // `dynamic_set_auxiliary_series` adds it to the data)
        let target = match &pac.target_expr {
            Some(e) => crate::varmodels::lagged(e, 1),
            None => lag(&pac.target, -1),
        };
        let lhs = |k: i32| match &pac.lhs_expr {
            Some(e) => crate::varmodels::lagged(e, k as usize),
            None => lag(&pac.lhs_var, -k),
        };
        let ect = Ast::Bin(crate::model::expr::BinaryOp::Sub, Box::new(target), Box::new(lhs(1)));
        let extra: Vec<(&str, &Vec<(Ast, f64)>)> =
            optim.iter().chain(&additive).map(|(k, v)| (names[*k].as_str(), v)).collect();
        for (i, t) in (range.0..=range.1).enumerate() {
            xdata[(i, 0)] = e.ev(&ect, t as isize)?;
            for k in 1..n1 {
                xdata[(i, k)] = e.ev(&Ast::Op(SpecialOp::Diff, vec![lhs(k as i32)]), t as isize)?;
            }
            for (c, (n, terms)) in extra.iter().enumerate() {
                let mut v = 0.0;
                for (a, s) in terms.iter() {
                    v += s * e.ev(&with_unit(a, n), t as isize)?;
                }
                xdata[(i, n1 + c)] = v;
            }
        }
    }
    // params_id_1 (error correction and autoregressive), params_id_5 (and optimizing additive),
    // params_id_4 = [params_id_1; params_id_3] (the order of params0_)
    let mut excl: Vec<usize> = share_k.into_iter().collect();
    excl.extend(optim.iter().chain(&additive).map(|(k, _)| *k));
    let id1 = setdiff_ia(&ip, &excl);
    let mut excl5: Vec<usize> = share_k.into_iter().collect();
    excl5.extend(additive.iter().map(|(k, _)| *k));
    let n5 = setdiff_ia(&ip, &excl5).len();
    let id0: Vec<usize> = (0..ip.len()).filter(|&i| Some(ip[i]) == share_k).collect();
    let id3: Vec<usize> = (0..ip.len()).filter(|i| !id1.contains(i) && !id0.contains(i)).collect();
    let id4: Vec<usize> = id1.iter().chain(&id3).copied().collect();
    if id4.len() != ncol {
        bail!("pac.estimate.iterative_ols: {} parameters for {ncol} regressors", id4.len());
    }
    let mut x0: Vec<f64> = id4.iter().map(|&i| params0[i]).collect();
    let mut gamma = match share_k {
        Some(_) if estimate_share => params0[id0[0]],
        Some(k) => params[k],
        None => 1.0,
    };
    if !(0.0..=1.0).contains(&gamma) {
        bail!("Initial value for the share of optimizing agents shoud be in (0,1).");
    }
    // M_.params(ipnames_) = params0
    set_params(vms, pac, names, params, &ip, &params0)?;
    let col = |c: usize| xdata.column(c).iter().copied().collect::<Vec<f64>>();
    let at_share = |p: &[f64], v: f64| -> Result<Vec<f64>> {
        let mut p = p.to_vec();
        p[share_k.unwrap()] = v;
        Ok(residuals(pac, names, &p, data, range, rname)?.0)
    };
    let mut counter = 0;
    let (mut ssr, mut resid, mut lhs);
    loop {
        counter += 1;
        // YDATA = lhs - share*PacExpectations - (1-share)*non optimizing - calibrated additive
        // terms - share*calibrated optimizing additive terms: the residual plus the estimated
        // terms at the current M_.params
        let (r0, l) = residuals(pac, names, params, data, range, rname)?;
        let mut y = r0;
        for (c, &k) in colpar.iter().enumerate() {
            let w = if c < n1 + optim.len() { gamma } else { 1.0 } * params[k];
            for (i, v) in col(c).iter().enumerate() {
                y[i] += w * v;
            }
        }
        let ym = Mat::from_column_slice(t_n, 1, &y);
        let p1 = xdata.clone().svd(true, true).solve(&ym, 1e-14).map_err(|e| anyhow!("{e}"))?;
        let mut p1: Vec<f64> = p1.iter().copied().collect();
        for v in p1.iter_mut().take(n5) {
            *v /= gamma;
        }
        let fit = xdata.columns(0, n5) * Mat::from_column_slice(n5, 1, &p1[..n5]);
        resid = (0..t_n).map(|i| y[i] - fit[(i, 0)] * gamma).collect::<Vec<f64>>();
        if !optim.is_empty() {
            let rest = xdata.columns(n5, ncol - n5) * Mat::from_column_slice(ncol - n5, 1, &p1[n5..]);
            for (i, r) in resid.iter_mut().enumerate() {
                *r -= rest[(i, 0)];
            }
        }
        ssr = resid.iter().map(|v| v * v).sum();
        lhs = l;
        let crit = p1.iter().zip(&x0).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        x0 = p1;
        if estimate_share {
            // M_.params(ipnames_(params_id_4)) = params0_ (and the h parameters)
            let est: Vec<usize> = id4.iter().map(|&i| ip[i]).collect();
            set_params(vms, pac, names, params, &est, &x0)?;
            // YDATA = lhs - non optimizing - calibrated additive - XDATA(:,n1:end)*params0_(n1:end)
            // (from the last autoregressive column, as in MATLAB); ZDATA = XDATA(:,1:n5)*
            // params0_(1:n5) + PacExpectations - non optimizing + calibrated optimizing additive
            let k = share_k.unwrap();
            let yv = at_share(params, 0.0)?;
            let r1 = at_share(params, 1.0)?;
            let (mut zy, mut zz) = (0.0, 0.0);
            for i in 0..t_n {
                let mut yi = yv[i];
                let mut zi = yv[i] - r1[i];
                for (c, &kc) in colpar.iter().enumerate() {
                    let x = xdata[(i, c)];
                    if c >= n1 + optim.len() {
                        yi += x * params[kc];
                    }
                    if !additive.is_empty() && c + 1 >= n1 {
                        yi -= x * x0[c];
                    }
                    if c < n5 {
                        zi += x * (x0[c] - params[kc]);
                    }
                }
                zy += zi * yi;
                zz += zi * zi;
            }
            gamma = (zy / zz).clamp(0.0, 1.0);
            if gamma < f64::EPSILON {
                bail!(
                    "On iteration {counter} the share of optimizing agents is found to be zero. Please increase the default lower bound for this parameter."
                );
            }
            params[k] = gamma;
        } else {
            // M_.params(ipnames_) = params0_
            set_params(vms, pac, names, params, &ip, &x0)?;
        }
        if crit <= 1e-6 || counter > 1000 {
            break;
        }
    }
    let tf = t_n as f64;
    Ok(PacEstimate {
        // fieldnames(params): the share of optimizing agents is not in the estimator
        parnames: guess.iter().map(|(n, _)| n.clone()).collect(),
        estimator: x0,
        covariance: None,
        student: vec![f64::NAN; ncol],
        ssr,
        s2: ssr / tf,
        r2: 1.0 - var_of(&resid) / var_of(&lhs),
        residual: resid,
        lhs,
        iterations: counter,
    })
}
