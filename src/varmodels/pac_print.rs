// SPDX-License-Identifier: GPL-3.0-or-later
//! `pac.print(pacname, eqname[, withcalibration])`: Dynare's `print_expectations.m` (kind
//! `pac`) and `write_expectations.m`. Writes `<fname>/model/pac-expectations/<pac>-parameters.inc`
//! (declaration and calibration of the `h` parameters), `<pac>-expression.inc` (the unrolled
//! expectation), `<pac>-growth-neutrality-correction.inc` (with a growth neutrality correction)
//! and `+<fname>/+pac_expectations/+<pac>/evaluate.m` (the expectation as a dseries expression).

use super::VarModels;
use super::pac::PacModel;
use crate::model::expr::{BinaryOp, UnaryOp};
use crate::modfile::{Ast, SpecialOp};
use anyhow::{Result, anyhow, bail};

/// `%1.16f`.
fn f16(v: f64) -> String {
    format!("{v:.16}")
}

/// The expression as the preprocessor prints it (`coeff_str`, `growth_str`).
fn show(a: &Ast) -> String {
    show_with(a, &[])
}

/// [`show`] with the numbers written as in the .mod file (`texts`: value, text).
fn show_with(a: &Ast, texts: &[(f64, String)]) -> String {
    fn prec(a: &Ast) -> u8 {
        match a {
            Ast::Bin(BinaryOp::Add | BinaryOp::Sub, ..) => 1,
            Ast::Bin(BinaryOp::Mul | BinaryOp::Div, ..) => 2,
            Ast::Un(UnaryOp::Neg, _) => 3,
            Ast::Bin(BinaryOp::Pow, ..) => 4,
            _ => 5,
        }
    }
    fn go(a: &Ast, texts: &[(f64, String)]) -> String {
        let go = |x: &Ast| go(x, texts);
        let wrap = |x: &Ast, p: u8| if prec(x) < p { format!("({})", go(x)) } else { go(x) };
        match a {
            Ast::Num(v) => texts.iter().find(|(w, _)| w == v).map_or_else(|| format!("{v}"), |(_, t)| t.clone()),
            Ast::Var { name, lag: 0, .. } => name.clone(),
            Ast::Var { name, lag, .. } => format!("{name}({lag})"),
            Ast::Un(UnaryOp::Neg, x) => format!("-{}", wrap(x, 3)),
            Ast::Un(op, x) => format!("{}({})", op.name(), go(x)),
            Ast::Bin(op, x, y) => {
                let (s, p) = match op {
                    BinaryOp::Add => ("+", 1),
                    BinaryOp::Sub => ("-", 1),
                    BinaryOp::Mul => ("*", 2),
                    BinaryOp::Div => ("/", 2),
                    BinaryOp::Pow => ("^", 4),
                    _ => return format!("{op:?}({},{})", go(x), go(y)),
                };
                // right operand of - and /: parenthesized at equal precedence
                let rp = if matches!(op, BinaryOp::Sub | BinaryOp::Div | BinaryOp::Pow) { p + 1 } else { p };
                format!("{}{s}{}", wrap(x, p), wrap(y, rp))
            }
            Ast::Op(SpecialOp::Diff, v) => format!("diff({})", go(&v[0])),
            Ast::Op(op, v) => format!("{op:?}({})", v.iter().map(go).collect::<Vec<_>>().join(",")),
        }
    }
    go(&simplify(a), texts)
}

/// The coefficients built by the equation analysis: `1*x` -> `x`, `x*-1` -> `-x`.
fn simplify(a: &Ast) -> Ast {
    match a {
        Ast::Bin(BinaryOp::Mul, x, y) => {
            let (x, y) = (simplify(x), simplify(y));
            match (&x, &y) {
                (Ast::Num(v), _) if *v == 1.0 => y,
                (_, Ast::Num(v)) if *v == 1.0 => x,
                (Ast::Num(v), _) if *v == -1.0 => Ast::Un(UnaryOp::Neg, Box::new(y)),
                (_, Ast::Num(v)) if *v == -1.0 => Ast::Un(UnaryOp::Neg, Box::new(x)),
                _ => Ast::Bin(BinaryOp::Mul, Box::new(x), Box::new(y)),
            }
        }
        Ast::Bin(op, x, y) => Ast::Bin(*op, Box::new(simplify(x)), Box::new(simplify(y))),
        Ast::Un(op, x) => Ast::Un(*op, Box::new(simplify(x))),
        other => other.clone(),
    }
}

/// A term of `growth_linear_comb`: constant, parameter, (variable, lag, diff).
struct GrowthTerm {
    constant: f64,
    param: Option<String>,
    var: Option<(String, i32, bool)>,
}

/// A growth term that the preprocessor replaces by an auxiliary variable (`diff(log(x(-2)))`
/// is `AUX_DIFF_LAG_<n>`, a chain of `diff` lag / `diff` / unary operator auxiliary variables)
/// as `print_expectations.m` writes it after `rewrite_aux_variable`: `x.log().lag(2).diff()`.
fn rewrite_growth(a: &Ast) -> Result<String> {
    let bad = || anyhow!("pac.print: the growth must be a linear combination of variables");
    // the auxiliary variable types from the outermost: 9 (diff lag), 8 (diff), 10 (unary op)
    let mut chain: Vec<(u8, &str)> = Vec::new();
    let mut cur = a;
    let lag = loop {
        match cur {
            Ast::Op(SpecialOp::Diff, v) => {
                chain.push((8, "diff"));
                cur = &v[0];
            }
            Ast::Un(op, x) if *op != UnaryOp::Neg => {
                chain.push((10, op.name()));
                cur = x;
            }
            Ast::Var { lag, .. } => break *lag,
            _ => return Err(bad()),
        }
    };
    let Ast::Var { name, .. } = cur else { return Err(bad()) };
    if chain.first().is_some_and(|c| c.0 == 8) {
        for _ in 0..(-lag).max(0) {
            chain.insert(0, (9, ""));
        }
    }
    // rewrite_aux_variable
    let mut tr: Vec<(String, Option<i32>)> = Vec::new();
    let (mut op, mut dl) = (0usize, 0);
    for (ty, f) in chain {
        op += 1;
        if tr.len() < op {
            tr.push((String::new(), None));
        }
        match ty {
            9 => {
                dl += 1;
                tr[op - 1] = ("lag".into(), Some(dl));
                op -= 1;
            }
            _ => tr[op - 1].0 = f.to_string(),
        }
    }
    let mut v = name.clone();
    for (t, l) in tr.iter().rev() {
        v = match (t.as_str(), l) {
            ("lag", Some(l)) => format!("{v}.lag({})", -l),
            ("diff", None) => format!("{v}.diff()"),
            ("diff", Some(l)) => format!("{v}.lag({l}).diff()"),
            (f, _) => format!("{v}.{f}()"),
        };
    }
    Ok(v)
}

fn growth_terms(g: &Ast, names: &[String]) -> Result<Vec<GrowthTerm>> {
    fn additive(a: &Ast, sign: f64, out: &mut Vec<(Ast, f64)>) {
        match a {
            Ast::Bin(BinaryOp::Add, x, y) => {
                additive(x, sign, out);
                additive(y, sign, out);
            }
            Ast::Bin(BinaryOp::Sub, x, y) => {
                additive(x, sign, out);
                additive(y, -sign, out);
            }
            Ast::Un(UnaryOp::Neg, x) => additive(x, -sign, out),
            _ => out.push((a.clone(), sign)),
        }
    }
    fn factors(a: &Ast, t: &mut GrowthTerm, names: &[String]) -> Result<()> {
        match a {
            Ast::Num(v) => t.constant *= v,
            Ast::Bin(BinaryOp::Mul, x, y) => {
                factors(x, t, names)?;
                factors(y, t, names)?;
            }
            Ast::Un(UnaryOp::Neg, x) => {
                t.constant = -t.constant;
                factors(x, t, names)?;
            }
            Ast::Var { name, .. } if names.contains(name) && t.param.is_none() => t.param = Some(name.clone()),
            Ast::Var { name, lag, .. } if t.var.is_none() => t.var = Some((name.clone(), *lag, false)),
            Ast::Op(SpecialOp::Diff, v) if t.var.is_none() => match &v[0] {
                Ast::Var { name, lag, .. } => t.var = Some((name.clone(), *lag, true)),
                _ => t.var = Some((rewrite_growth(a)?, 0, false)),
            },
            Ast::Un(_, _) if t.var.is_none() => t.var = Some((rewrite_growth(a)?, 0, false)),
            _ => bail!("pac.print: the growth must be a linear combination of variables"),
        }
        Ok(())
    }
    let mut terms = Vec::new();
    additive(g, 1.0, &mut terms);
    terms
        .into_iter()
        .map(|(a, sign)| {
            let mut t = GrowthTerm { constant: sign, param: None, var: None };
            factors(&a, &mut t, names)?;
            Ok(t)
        })
        .collect()
}

impl PacModel {
    /// `print_expectations(eqname, pacname, 'pac', withcalibration)`: the files (paths relative
    /// to the directory of the .mod file) and the console output.
    pub fn print(
        &self,
        vms: &VarModels,
        names: &[String],
        params: &[f64],
        fname: &str,
        withcal: bool,
    ) -> Result<(Vec<(String, String)>, String)> {
        let Some(aux) = &self.aux_model else { bail!("print method does not work in PAC/MCE.") };
        if vms.auxiliary_kind(aux).is_none() {
            bail!("Unknown VAR/TREND_COMPONENT model ({aux}) in PAC_EXPECTATION_MODEL ({})!", self.name);
        }
        let val = |n: &str| -> Result<f64> {
            names.iter().position(|x| x == n).map(|i| params[i]).ok_or_else(|| anyhow!("unknown parameter {n}"))
        };
        let lookup = |n: &str, _: i32, _: &crate::modfile::lexer::Loc| val(n);
        let simple = self.components.is_empty();
        let is_var = self.aux_type == "var";
        let date = crate::latex::datestr_now();
        let name = &self.name;
        let dir = format!("{fname}/model/pac-expectations");
        let mut files = Vec::new();
        let mut out = String::new();
        // pac_target_coefficients
        let tc: Vec<f64> =
            self.components.iter().map(|c| crate::modfile::eval_ast(&c.coef, &lookup)).collect::<Result<_>>()?;
        let coeff_str: Vec<String> = self.components.iter().map(|c| show_with(&c.coef, &c.num_texts)).collect();
        let hp = |k: usize, id: usize| -> &String { &self.components[k].h_params[id] };

        // ---- parameters
        let mut s = format!("// This file has been generated by Dynare ({date}).\n\n");
        let all: Vec<&String> = if simple {
            self.h_params.iter().collect()
        } else {
            self.components.iter().flat_map(|c| &c.h_params).collect()
        };
        s += "parameters";
        for p in &all {
            s += &format!(" {p}");
        }
        s += ";\n\n";
        if withcal {
            for p in &all {
                s += &format!("{p} = {};\n", f16(val(p)?));
            }
        }
        let mut growth_correction = false;
        let gps: Vec<&String> = if simple {
            self.growth_param.iter().collect()
        } else {
            self.components.iter().filter_map(|c| c.growth_param.as_ref()).collect()
        };
        for g in gps {
            s += &format!("\nparameters {g};\n\n");
            if withcal {
                s += &format!("{g} = {};\n", f16(val(g)?));
            }
            growth_correction = true;
        }
        let f = format!("{dir}/{name}-parameters.inc");
        out += &format!("\nParameters declarations and calibrations are saved in {f}.\n");
        files.push((f, s));

        // ---- expression (write_expectations, iscrlf)
        let mut expr = String::new();
        let mut id = 0;
        let agg_names = |id: usize| -> String {
            let mut e = String::new();
            for k in 0..self.components.len() {
                if k > 0 {
                    e += "+";
                }
                if coeff_str[k] == "1" {
                    e += hp(k, id);
                } else {
                    e += &format!("{}*{}", coeff_str[k], hp(k, id));
                }
            }
            e
        };
        if is_var {
            id += 1;
            expr = if simple { self.h_params[0].clone() } else { agg_names(0) };
        }
        for e in self.h_layout.iter().skip(id) {
            let Some((v, tr, lag)) = e else { continue };
            id += 1;
            let p = if simple { self.h_params[id - 1].clone() } else { format!("({})", agg_names(id - 1)) };
            let v = tr.iter().fold(format!("{v}(-{lag})"), |v, t| format!("{t}({v})"));
            expr = if id == 1 { format!("{p}*{v}\n") } else { format!("{expr} + {p}*{v}\n") };
        }
        let f = format!("{dir}/{name}-expression.inc");
        files.push((f.clone(), format!("// This file has been generated by Dynare ({date}).\n{expr}")));
        out += &format!("Expectation unrolled expression is saved in {f}.\n");

        // ---- growth neutrality correction
        if growth_correction {
            let one = |g: &Ast| growth_terms(g, names).map(|t| t.len() == 1);
            let mut gnc = String::new();
            if simple {
                let g = self.growth.as_ref().unwrap();
                let gp = self.growth_param.as_ref().unwrap();
                gnc = if one(g)? { format!("{gp}*{}", show(g)) } else { format!("{gp}*({})", show(g)) };
            } else {
                for (k, c) in self.components.iter().enumerate() {
                    let (Some(g), Some(gp)) = (&c.growth, &c.growth_param) else { continue };
                    if c.kind == "ll" {
                        continue;
                    }
                    let gs = if one(g)? { show(g) } else { format!("({})", show(g)) };
                    let term =
                        if coeff_str[k] == "1" { format!("{gp}*{gs}") } else { format!("{}*{gp}*{gs}", coeff_str[k]) };
                    gnc = if gnc.is_empty() { term } else { format!("{gnc}+{term}") };
                }
            }
            let f = format!("{dir}/{name}-growth-neutrality-correction.inc");
            files.push((f.clone(), format!("// This file has been generated by Dynare ({date}).\n{gnc}")));
            out += &format!("Growth neutrality correction is saved in {f}.\n");
        }

        // ---- evaluate.m
        let mut m = String::from("function ds = evaluate(dbase)\n\n");
        m += &format!("% Evaluates pac_expectations term ({name}).\n");
        m += "%\n% INPUTS\n";
        m += "% - dbase     [dseries]  databse containing all the variables appearing in the auxiliary model for the expectation.\n";
        m += "%\n% OUTPUTS\n% - ds        [dseries]  the expectation term .\n%\n% REMARKS\n";
        m += "% The name of the appended variable in dbase is the declared name for the (PAC/VAR) expectation model.\n\n";
        m += &format!("% This file has been generated by Dynare ({date}).\n\n");
        m += "ds = dseries();\n\n";
        let mut expression: Option<String> = None;
        let mut expressions: Vec<String> = Vec::new();
        let mut id = 0;
        if is_var {
            id += 1;
            if simple {
                expression = Some(f16(val(&self.h_params[0])?));
            } else {
                for k in 0..self.components.len() {
                    expressions.push(f16(val(hp(k, 0))?));
                }
            }
        }
        let signed = |e: &str, p: f64, v: &str| {
            if p >= 0.0 {
                format!("{e}+{}*{v}", f16(p))
            } else if p < 0.0 {
                format!("{e}-{}*{v}", f16(-p))
            } else {
                e.to_string()
            }
        };
        for e in self.h_layout.iter().skip(id) {
            let Some((v, tr, lag)) = e else { continue };
            id += 1;
            let p = if simple {
                val(&self.h_params[id - 1])?
            } else {
                let mut p = 0.0;
                for k in 0..self.components.len() {
                    p += tc[k] * val(hp(k, id - 1))?;
                }
                p
            };
            let v = tr.iter().fold(format!("dbase.{v}(-{lag})"), |v, t| format!("{v}.{t}()"));
            expression = match expression.take() {
                Some(e) => Some(signed(&e, p, &v)),
                None if p >= 0.0 => Some(format!("{}*{v}", f16(p))),
                None if p < 0.0 => Some(format!("-{}*{v}", f16(-p))),
                None => None,
            };
            for (k, x) in expressions.iter_mut().enumerate() {
                let q = val(hp(k, id - 1))?;
                *x = if q >= 0.0 { format!("{x}+{}*{v}", f16(q)) } else { format!("{x}-{}*{v}", f16(-q)) };
            }
        }
        let mut expression = expression.unwrap_or_default();
        if growth_correction {
            let lin = |g: &Ast| -> Result<String> {
                let mut lc = String::new();
                for (i, t) in growth_terms(g, names)?.iter().enumerate() {
                    let mut vg = String::new();
                    if let Some((v, lag, df)) = &t.var {
                        let v = match (*df, *lag) {
                            (false, 0) => v.clone(),
                            (false, -1) => format!("{v}(-1)"),
                            (false, l) if l < -1 => format!("{v}.lag({})", -l),
                            (true, l) if l < 0 => format!("{v}.lag({}).diff()", -l),
                            _ => bail!("pac.print: the growth term of {v} is not supported"),
                        };
                        vg = format!("dbase.{v}");
                    }
                    if let Some(p) = &t.param {
                        vg = if vg.is_empty() { f16(val(p)?) } else { format!("{}*{vg}", f16(val(p)?)) };
                    }
                    if t.constant.abs() != 1.0 {
                        vg = if vg.is_empty() { f16(t.constant) } else { format!("{}*{vg}", f16(t.constant)) };
                    }
                    lc = if i == 0 {
                        vg
                    } else if t.constant > 0.0 {
                        format!("{lc}+{vg}")
                    } else {
                        format!("{lc}-{vg}")
                    };
                }
                Ok(lc)
            };
            let gc = if simple {
                format!("{}*({})", f16(val(self.growth_param.as_ref().unwrap())?), lin(self.growth.as_ref().unwrap())?)
            } else {
                let mut gc: Option<String> = None;
                for (k, c) in self.components.iter().enumerate() {
                    let (Some(g), Some(gp)) = (&c.growth, &c.growth_param) else { continue };
                    if c.kind == "ll" {
                        continue;
                    }
                    let pg = tc[k] * val(gp)?;
                    let l = lin(g)?;
                    gc = Some(match gc {
                        None => format!("{}*({l})", f16(pg)),
                        Some(x) if pg > 0.0 => format!("{x}+{}*({l})", f16(pg)),
                        Some(x) if pg < 0.0 => format!("{x}-{}*({l})", f16(-pg)),
                        Some(x) => x,
                    });
                }
                gc.unwrap_or_default()
            };
            expression = format!("{expression}+{gc}");
        }
        m += &format!("ds.{name} = {expression};\n");
        for (k, x) in expressions.iter().enumerate() {
            m += &format!("ds.{} = {x};\n", self.components[k].auxname);
        }
        let f = format!("+{fname}/+pac_expectations/+{name}/evaluate.m");
        out += &format!("Expectation dseries expression is saved in {f}.\n\n");
        files.push((f, m));
        Ok((files, out))
    }
}

/// `rewrite_aux_variable`: the variable of a companion expression and its transformations,
/// innermost first (`diff(log(x))` is `x` with `log`, `diff`).
fn rewrite(a: &Ast) -> (String, Vec<String>) {
    match a {
        Ast::Op(SpecialOp::Diff, v) => {
            let (n, mut t) = rewrite(&v[0]);
            t.push("diff".into());
            (n, t)
        }
        Ast::Un(op, x) => {
            let (n, mut t) = rewrite(x);
            t.push(op.name().into());
            (n, t)
        }
        Ast::Var { name, .. } => (name.clone(), Vec::new()),
        _ => (String::new(), Vec::new()),
    }
}

impl VarModels {
    /// `var_expectation.print(name[, withcalibration])`: `print_expectations('fake', name,
    /// 'var', withcalibration)` and `write_expectations.m`. Writes
    /// `<fname>/model/var-expectations/<name>-parameters.inc`, `<name>-expression.inc` and
    /// `+<fname>/+var_expectations/+<name>/evaluate.m`; returns the files (paths relative to
    /// the directory of the .mod file) and the console output.
    pub fn print_var_expectation(
        &self,
        name: &str,
        names: &[String],
        params: &[f64],
        fname: &str,
        withcal: bool,
    ) -> Result<(Vec<(String, String)>, String)> {
        let m = self.expectation(name).ok_or_else(|| anyhow!("VAR_EXPECTATION_MODEL {name} is not defined."))?;
        let aux = &m.auxiliary_model_name;
        let comp: Vec<(String, Ast)> = match (self.var(aux), self.trend(aux)) {
            (Some(v), _) => v.companion_exprs(),
            (None, Some(t)) => t.companion_exprs(),
            _ => bail!("Unknown VAR/TREND_COMPONENT model ({aux}) in VAR_EXPECTATION_MODEL ({name})!"),
        };
        let is_var = self.var(aux).is_some();
        let val = |n: &str| -> Result<f64> {
            names.iter().position(|x| x == n).map(|i| params[i]).ok_or_else(|| anyhow!("unknown parameter {n}"))
        };
        let date = crate::latex::datestr_now();
        let dir = format!("{fname}/model/var-expectations");
        let mut files = Vec::new();
        let mut out = String::new();

        // ---- parameters
        let mut s = format!("// This file has been generated by Dynare ({date}).\n\n");
        s += "parameters";
        for p in &m.param_names {
            s += &format!(" {p}");
        }
        s += ";\n\n";
        if withcal {
            for p in &m.param_names {
                s += &format!("{p} = {};\n", f16(val(p)?));
            }
        }
        let f = format!("{dir}/{name}-parameters.inc");
        out += &format!("\nParameters declarations and calibrations are saved in {f}.\n");
        files.push((f, s));

        // the companion variables by lag: (parameter, variable, transformations, time index)
        let k0 = is_var as usize;
        let n = comp.len();
        let ts = m.time_shift.unsigned_abs() as usize;
        let terms: Vec<(&String, String, Vec<String>, usize)> = m.param_names[k0..]
            .iter()
            .enumerate()
            .map(|(id, p)| {
                let (v, tr) = rewrite(&comp[id % n].1);
                (p, v, tr, id / n + ts)
            })
            .collect();

        // ---- expression (write_expectations, iscrlf)
        let mut expr = if is_var { m.param_names[0].clone() } else { String::new() };
        for (id, (p, v, tr, l)) in terms.iter().enumerate() {
            let v = if *l > 0 { format!("{v}(-{l})") } else { v.clone() };
            let v = tr.iter().fold(v, |v, t| format!("{t}({v})"));
            expr = if id + k0 == 0 { format!("{p}*{v}\n") } else { format!("{expr} + {p}*{v}\n") };
        }
        let f = format!("{dir}/{name}-expression.inc");
        files.push((f.clone(), format!("// This file has been generated by Dynare ({date}).\n{expr}")));
        out += &format!("Expectation unrolled expression is saved in {f}.\n");

        // ---- evaluate.m
        let mut e = String::from("function ds = evaluate(dbase)\n\n");
        e += &format!("% Evaluates var_expectations term ({name}).\n");
        e += "%\n% INPUTS\n";
        e += "% - dbase     [dseries]  databse containing all the variables appearing in the auxiliary model for the expectation.\n";
        e += "%\n% OUTPUTS\n% - ds        [dseries]  the expectation term .\n%\n% REMARKS\n";
        e += "% The name of the appended variable in dbase is the declared name for the (PAC/VAR) expectation model.\n\n";
        e += &format!("% This file has been generated by Dynare ({date}).\n\n");
        e += "ds = dseries();\n\n";
        let mut expression: Option<String> = if is_var { Some(f16(val(&m.param_names[0])?)) } else { None };
        for (p, v, tr, l) in &terms {
            let p = val(p)?;
            let v = if *l > 0 { format!("dbase.{v}(-{l})") } else { format!("dbase.{v}") };
            let v = tr.iter().fold(v, |v, t| format!("{v}.{t}()"));
            expression = match expression.take() {
                Some(x) if p >= 0.0 => Some(format!("{x}+{}*{v}", f16(p))),
                Some(x) if p < 0.0 => Some(format!("{x}-{}*{v}", f16(-p))),
                Some(x) => Some(x),
                None if p >= 0.0 => Some(format!("{}*{v}", f16(p))),
                None if p < 0.0 => Some(format!("-{}*{v}", f16(-p))),
                None => None,
            };
        }
        e += &format!("ds.{name} = {};\n", expression.unwrap_or_default());
        let f = format!("+{fname}/+var_expectations/+{name}/evaluate.m");
        out += &format!("Expectation dseries expression is saved in {f}.\n\n");
        files.push((f, e));
        Ok((files, out))
    }
}
