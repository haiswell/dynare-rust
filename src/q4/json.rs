// SPDX-License-Identifier: GPL-3.0-or-later
//! `json=parse|check|transform|compute` (+ `jsonstdout`, `onlyjson`,
//! `jsonderivsimple`): the preprocessor's JSON output, `<fname>/model/json/`.
//!
//! - `parse`, `check`: `modfile.json` with the model as parsed (the preprocessor's DataTree
//!   after parsing: constants folded, commutative operands by node index);
//! - `transform`: `modfile-original.json` (the same) and `modfile.json` with the transformed
//!   model (aux variables and equations, default equation names);
//! - `compute`: also `static.json` and `dynamic.json` (residuals, first derivatives with
//!   their sparse layout) and `model_info` in `modfile.json`.
//!
//! The equations are printed from `model::dtree` (the port of the preprocessor's
//! DataTree) with `ExprNode::writeJsonOutput`'s precedence rules, so `model`, `aux_vars`,
//! `abstract_syntax_tree`, the declarations, `variable_mapping`, the Jacobian sparsity and
//! `model_info` are those of the 7.1 preprocessor (`tests/q4.rs` compares them). Not
//! reproduced: the derivative expressions (dynare-rust's own symbolic derivatives, printed
//! without the preprocessor's temporary terms `T<k>`), the static model's operand order (it
//! is printed from the dynamic one with the lags dropped), statements other than
//! `param_init`, `shocks` and the option lists of commands, `steady_state_model` contents.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use serde_json::{Map, Value, json};

use crate::model::dtree::{JsonCapture, JsonEquation};
use crate::model::expr::{BinaryOp, Dag, DiffVar, Node, NodeId, UnaryOp};
use crate::modfile::{Command, ModFile, OptValue};

/// The JSON options of the command line.
#[derive(Clone, Debug, Default)]
pub struct JsonOptions {
    pub stage: String,
    pub stdout: bool,
    pub derivsimple: bool,
}

fn tags_obj(tags: &[(String, String)]) -> Value {
    let mut m = Map::new();
    for (k, v) in tags {
        m.insert(k.clone(), json!(v));
    }
    Value::Object(m)
}

fn model_entries(eqs: &[JsonEquation], with_tags: bool) -> Value {
    Value::Array(
        eqs.iter()
            .map(|e| {
                let mut m = Map::new();
                m.insert("lhs".into(), json!(e.lhs));
                m.insert("rhs".into(), json!(e.rhs));
                if let Some(l) = e.line {
                    m.insert("line".into(), json!(l));
                }
                if with_tags && !e.tags.is_empty() {
                    m.insert("tags".into(), tags_obj(&e.tags));
                }
                Value::Object(m)
            })
            .collect(),
    )
}

/// the `varexo_det` variables are parameters of dynare-rust's model; the
/// preprocessor's AST types them `exogenousDet`.
fn exo_det_types(v: &mut Value, det: &[String]) {
    match v {
        Value::Object(m) => {
            if m.get("node_type").and_then(Value::as_str) == Some("VariableNode")
                && m.get("name").and_then(Value::as_str).is_some_and(|n| det.iter().any(|d| d == n))
            {
                m.insert("type".into(), json!("exogenousDet"));
            }
            m.values_mut().for_each(|x| exo_det_types(x, det));
        }
        Value::Array(a) => a.iter_mut().for_each(|x| exo_det_types(x, det)),
        _ => {}
    }
}

fn ast_entries(eqs: &[JsonEquation], n_orig: usize, det: &[String]) -> Value {
    let mut v = ast_entries0(eqs, n_orig);
    if !det.is_empty() {
        exo_det_types(&mut v, det);
    }
    v
}

fn ast_entries0(eqs: &[JsonEquation], n_orig: usize) -> Value {
    Value::Array(
        eqs.iter()
            .enumerate()
            .filter_map(|(i, e)| {
                let ast = e.ast.clone()?;
                let mut m = Map::new();
                m.insert("number".into(), json!(i));
                if let Some(l) = e.line {
                    m.insert("line".into(), json!(l));
                }
                if i < n_orig {
                    m.insert("tags".into(), tags_obj(&e.tags)); // aux equations have none
                }
                m.insert("AST".into(), ast);
                Some(Value::Object(m))
            })
            .collect(),
    )
}

fn symbols(names: &[String], tex: &[String], long: &[String]) -> Value {
    Value::Array(
        names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                json!({
                    "name": n,
                    "texName": tex.get(i).cloned().unwrap_or_else(|| n.replace('_', "\\_")),
                    "longName": long.get(i).cloned().unwrap_or_else(|| n.clone()),
                })
            })
            .collect(),
    )
}

/// MATLAB/preprocessor `%.16g` for numbers we print ourselves.
fn g16(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Inf".into() } else { "(-Inf)".into() };
    }
    let t = format!("{v:.15e}");
    let (m, e) = t.split_once('e').unwrap();
    let exp: i32 = e.parse().unwrap();
    let r = if (-5..16).contains(&exp) {
        let d = (15 - exp).max(0) as usize;
        let x = format!("{v:.d$}");
        if x.contains('.') { x.trim_end_matches('0').trim_end_matches('.').to_string() } else { x }
    } else {
        let m = m.trim_end_matches('0').trim_end_matches('.');
        format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    };
    if v < 0.0 { format!("({r})") } else { r }
}

/// Infix printer of the crate's DAG with the preprocessor's JSON precedence rules (for the
/// derivatives and the static model).
fn dag_str(d: &Dag, id: NodeId, names: &dyn Fn(Node) -> String, out: &mut String) {
    let prec = |n: NodeId| -> i32 {
        match d.node(n) {
            Node::Unary(UnaryOp::Neg, _) => 6,
            Node::Binary(op, ..) => match op {
                BinaryOp::Eq | BinaryOp::Ne => 1,
                BinaryOp::Lt | BinaryOp::Gt | BinaryOp::Le | BinaryOp::Ge => 2,
                BinaryOp::Add | BinaryOp::Sub => 3,
                BinaryOp::Mul | BinaryOp::Div => 4,
                BinaryOp::Pow => 5,
                _ => 100,
            },
            Node::Num(b) if f64::from_bits(b) < 0.0 => 6,
            _ => 100,
        }
    };
    match d.node(id) {
        Node::Num(b) => out.push_str(&g16(f64::from_bits(b))),
        n @ (Node::Endo(..) | Node::Exo(..) | Node::Param(_) | Node::SteadyState(..)) => out.push_str(&names(n)),
        Node::SteadyStateParamDeriv(..) | Node::SteadyStateParam2ndDeriv(..) => {
            unreachable!("parameter derivatives of STEADY_STATE are not part of the model equations")
        }
        Node::External(c) => {
            // value `f(a, b)`; derivatives as the preprocessor's MATLAB output
            let e = d.ext_call(c);
            let f = crate::model::external::name(e.fid);
            match (e.d1, e.d2) {
                (0, _) => out.push_str(&format!("{f}(")),
                (i, 0) => out.push_str(&format!("jacob_element('{f}',{i},{{")),
                (i, j) => out.push_str(&format!("hess_element('{f}',{i},{j},{{")),
            }
            for (k, &a) in e.args.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                dag_str(d, a, names, out);
            }
            out.push_str(if e.d1 == 0 { ")" } else { "})" });
        }
        Node::Unary(UnaryOp::Neg, a) => {
            out.push_str("(-");
            let p = prec(a) < 6;
            if p {
                out.push('(');
            }
            dag_str(d, a, names, out);
            if p {
                out.push(')');
            }
            out.push(')');
        }
        Node::Unary(op, a) => {
            out.push_str(op.name());
            out.push('(');
            dag_str(d, a, names, out);
            out.push(')');
        }
        Node::Binary(op @ (BinaryOp::Max | BinaryOp::Min), a, b) => {
            out.push_str(if op == BinaryOp::Max { "max(" } else { "min(" });
            dag_str(d, a, names, out);
            out.push(',');
            dag_str(d, b, names, out);
            out.push(')');
        }
        Node::Binary(BinaryOp::PowerDeriv(k), a, b) => {
            out.push_str("get_power_deriv(");
            dag_str(d, a, names, out);
            out.push(',');
            dag_str(d, b, names, out);
            out.push_str(&format!(",{k})"));
        }
        Node::Binary(op, a, b) => {
            let p0 = prec(id);
            let (pa, pb) = (prec(a), prec(b));
            let p1 = pa < p0 || (op == BinaryOp::Pow && pa == p0);
            if p1 {
                out.push('(');
            }
            dag_str(d, a, names, out);
            if p1 {
                out.push(')');
            }
            out.push_str(op.symbol());
            let b_pow = matches!(d.node(b), Node::Binary(BinaryOp::Pow, ..));
            let p2 = pb < p0
                || (op == BinaryOp::Pow && b_pow)
                || (op == BinaryOp::Sub && pb == p0)
                || (op == BinaryOp::Div && pb == p0);
            if p2 {
                out.push('(');
            }
            dag_str(d, b, names, out);
            if p2 {
                out.push(')');
            }
        }
    }
}

fn statements(mf: &ModFile) -> Value {
    let mut out = Vec::new();
    let m = &mf.model;
    // parameter initializations (the values as computed; the preprocessor prints the
    // expression's DataTree form, which for a constant is the same)
    for (i, n) in m.param_names.iter().enumerate() {
        if n.starts_with("occbin_") && n.ends_with("_bind") {
            continue;
        }
        if let Some(v) = m.params.get(i).filter(|v| !v.is_nan()) {
            out.push(json!({"statementName": "param_init", "name": n, "value": g16(*v)}));
        }
    }
    if mf.sigma_e.nrows() > 0 {
        let mut stderr = Vec::new();
        let mut corr = Vec::new();
        for i in 0..mf.sigma_e.nrows() {
            let v = mf.sigma_e[(i, i)];
            if v != 0.0 {
                stderr.push(json!({"name": m.exo_names[i], "stderr": g16(v.sqrt())}));
            }
            for j in 0..i {
                let c = mf.sigma_e[(i, j)];
                if c != 0.0 {
                    corr.push(json!({"name": m.exo_names[i], "name2": m.exo_names[j], "correlation": g16(c / (mf.sigma_e[(i, i)] * mf.sigma_e[(j, j)]).sqrt())}));
                }
            }
        }
        if !stderr.is_empty() || !corr.is_empty() {
            out.push(json!({"statementName": "shocks", "overwrite": false, "variance": [], "stderr": stderr, "covariance": [], "correlation": corr, "skewness": []}));
        }
    }
    let opt_json = |o: &crate::modfile::Options| -> Value {
        fn v(x: &OptValue) -> Value {
            match x {
                OptValue::Flag => json!(true),
                OptValue::Num(n) => serde_json::Number::from_f64(*n).map_or(json!(n.to_string()), |n| {
                    if n.as_f64().is_some_and(|f| f.fract() == 0.0 && f.abs() < 1e15) {
                        json!(n.as_f64().unwrap() as i64)
                    } else {
                        Value::Number(n)
                    }
                }),
                OptValue::Str(s) => json!(s),
                OptValue::List(l) => Value::Array(l.iter().map(v).collect()),
                OptValue::Vec(l) => Value::Array(l.iter().map(|f| json!(f)).collect()),
            }
        }
        let mut keys: Vec<&(String, OptValue)> = o.iter().filter(|(k, _)| !k.starts_with("__")).collect();
        keys.sort_by(|a, b| a.0.cmp(&b.0));
        let mut mm = Map::new();
        for (k, x) in keys {
            mm.insert(k.clone(), v(x));
        }
        Value::Object(mm)
    };
    for c in &mf.commands {
        let (name, o, vars): (&str, Option<&crate::modfile::Options>, &[String]) = match c {
            Command::Steady(o) => ("steady", Some(o), &[]),
            Command::Check(o) => ("check", Some(o), &[]),
            Command::StochSimul(o, v) => ("stoch_simul", Some(o), v),
            Command::Estimation(o, v) => ("estimation", Some(o), v),
            Command::Forecast(o, v) => ("forecast", Some(o), v),
            Command::ShockDecomposition(o, v) => ("shock_decomposition", Some(o), v),
            Command::CalibSmoother(o, v) => ("calib_smoother", Some(o), v),
            Command::Deterministic(n, o, v) => (n.as_str(), Some(o), v),
            Command::RealtimeShockDecomposition(o, v) => ("realtime_shock_decomposition", Some(o), v),
            Command::PlotShockDecomposition(o, v) => ("plot_shock_decomposition", Some(o), v),
            Command::InitialConditionDecomposition(o, v) => ("initial_condition_decomposition", Some(o), v),
            Command::Other(n, o) if !n.starts_with("__") => (n.as_str(), Some(o), &[]),
            _ => continue,
        };
        let mut s = Map::new();
        s.insert("statementName".into(), json!(name));
        if let Some(o) = o.filter(|o| !o.is_empty()) {
            s.insert("options".into(), opt_json(o));
        }
        if !vars.is_empty() {
            s.insert("symbol_list".into(), json!(vars));
        }
        out.push(Value::Object(s));
    }
    Value::Array(out)
}

fn declarations(mf: &ModFile, transformed: bool) -> Map<String, Value> {
    let m = &mf.model;
    let l = &mf.labels;
    let n_endo = if transformed { m.n_endo() } else { m.orig_endo_nbr };
    let is_det = |k: usize| mf.exo_det.contains(&k);
    let exo: Vec<usize> = (0..m.n_exo()).filter(|&k| !is_det(k)).collect();
    let det: Vec<usize> = (0..m.n_exo()).filter(|&k| is_det(k)).collect();
    let pick = |idx: &[usize], v: &[String]| -> Vec<String> { idx.iter().filter_map(|&k| v.get(k).cloned()).collect() };
    let mut o = Map::new();
    o.insert("endogenous".into(), symbols(&m.endo_names[..n_endo], &l.endo_tex, &l.endo_long));
    o.insert("exogenous".into(), symbols(&pick(&exo, &m.exo_names), &pick(&exo, &l.exo_tex), &pick(&exo, &l.exo_long)));
    if m.exo_det_names.is_empty() {
        o.insert(
            "exogenous_deterministic".into(),
            symbols(&pick(&det, &m.exo_names), &pick(&det, &l.exo_tex), &pick(&det, &l.exo_long)),
        );
    } else {
        // varexo_det (parameters of the model, see ModelBuilder::q1_exo_det)
        o.insert("exogenous_deterministic".into(), symbols(&m.exo_det_names, &l.exo_det_tex, &l.exo_det_long));
    }
    let np = m.n_user_params();
    o.insert("parameters".into(), symbols(&m.param_names[..np], &l.param_tex, &l.param_long));
    o.insert("orig_endo_nbr".into(), json!(m.orig_endo_nbr));
    o
}

fn aux_vars(mf: &ModFile, cap: &JsonCapture, transformed: bool) -> Value {
    if !transformed {
        return json!([]);
    }
    let entry = |a: &crate::model::AuxVar, orig_expr: Option<String>| {
        let mut m = Map::new();
        m.insert("endo_index".into(), json!(a.endo_index + 1));
        m.insert("type".into(), json!(a.ty));
        if let Some(i) = a.orig_index {
            m.insert("orig_index".into(), json!(i + 1));
        }
        if let Some(l) = a.orig_lead_lag {
            m.insert("orig_lead_lag".into(), json!(l));
        }
        if let Some(e) = orig_expr {
            m.insert("orig_expr".into(), json!(e));
        }
        Value::Object(m)
    };
    // The aux vars the preprocessor keeps in M_.aux_vars but Rust tracks elsewhere: the
    // heterogeneity aggregation variables (type 14, listed first, `HetDecl::aggregate_aux`)
    // and the Ramsey multipliers (type 6 with `eq_nbr`, planner `mult_index`; created before
    // the lead/lag auxiliaries of the FOCs).
    let mut out: Vec<Value> = mf.het.aggregate_aux.iter().map(|a| entry(a, Some(a.orig_expr.clone()))).collect();
    let mut mults: Vec<(usize, usize)> = mf
        .model
        .planner()
        .and_then(|p| p.ramsey.as_ref())
        .map(|r| r.mult_index.iter().enumerate().map(|(i, &j)| (j, i + 1)).collect())
        .unwrap_or_default();
    for (k, a) in mf.model.aux_vars.iter().enumerate() {
        while mults.first().is_some_and(|&(j, _)| j < a.endo_index) {
            let (j, eq) = mults.remove(0);
            out.push(json!({"endo_index": j + 1, "type": 6, "eq_nbr": eq}));
        }
        let e = cap.aux_orig_expr.get(k).cloned().unwrap_or_else(|| a.orig_expr.clone());
        out.push(entry(a, Some(e)));
    }
    for (j, eq) in mults {
        out.push(json!({"endo_index": j + 1, "type": 6, "eq_nbr": eq}));
    }
    Value::Array(out)
}

/// Variables of an `abstract_syntax_tree` entry: (name, type, lag).
fn ast_vars(v: &Value, out: &mut Vec<(String, String, i64)>) {
    match v {
        Value::Object(m) => {
            if m.get("node_type").and_then(Value::as_str) == Some("VariableNode") {
                let s = |k: &str| m.get(k).and_then(Value::as_str).unwrap_or("").to_string();
                out.push((s("name"), s("type"), m.get("lag").and_then(Value::as_i64).unwrap_or(0)));
            }
            m.values().for_each(|x| ast_vars(x, out));
        }
        Value::Array(a) => a.iter().for_each(|x| ast_vars(x, out)),
        _ => {}
    }
}

/// `variable_mapping` (recorded by `dtree`).
fn variable_mapping(_mf: &ModFile, cap: &JsonCapture) -> Value {
    Value::Array(cap.mapping.iter().map(|(n, e)| json!({"name": n, "equations": e})).collect())
}

#[allow(dead_code)]
fn variable_mapping_from_ast(mf: &ModFile, cap: &JsonCapture) -> Value {
    let m = &mf.model;
    let name_of = |i: usize| -> String {
        m.equation_tags
            .get(i)
            .and_then(|t| t.iter().find(|(k, _)| k == "name"))
            .map_or((i + 1).to_string(), |(_, v)| v.clone())
    };
    let mut endo: Vec<Vec<usize>> = vec![Vec::new(); m.orig_endo_nbr];
    let mut exo: Vec<Vec<usize>> = vec![Vec::new(); m.n_exo()];
    for (i, e) in cap.original.iter().enumerate() {
        let mut vars = Vec::new();
        if let Some(a) = &e.ast {
            ast_vars(a, &mut vars);
        }
        for (n, ty, _) in vars {
            match ty.as_str() {
                "endogenous" => {
                    if let Some(j) = m.endo_index(&n).filter(|&j| j < m.orig_endo_nbr) {
                        endo[j].push(i)
                    }
                }
                "exogenous" => {
                    if let Some(k) = m.exo_index(&n) {
                        exo[k].push(i)
                    }
                }
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    for (names, lists) in [(&m.endo_names[..m.orig_endo_nbr], &mut endo), (&m.exo_names[..], &mut exo)] {
        for (n, l) in names.iter().zip(lists.iter_mut()) {
            l.sort_unstable();
            l.dedup();
            out.push(json!({"name": n, "equations": l.iter().map(|&i| name_of(i)).collect::<Vec<_>>()}));
        }
    }
    Value::Array(out)
}

/// The static Jacobian's sparse layout, 1-based CSC (`static_g1_sparse_*`): the structural
/// non-zeros of the static residuals' derivatives, with the preprocessor's simplifications.
fn static_sparse(mf: &ModFile) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = mf.model.n_endo();
    let mut pat = crate::model::blocks::static_jacobian_incidence(&mf.model);
    pat.sort_by_key(|&(r, c)| (c, r));
    let rowval = pat.iter().map(|&(r, _)| r + 1).collect();
    let colval = pat.iter().map(|&(_, c)| c + 1).collect();
    let mut colptr = vec![1usize; n + 1];
    for c in 0..n {
        colptr[c + 1] = colptr[c] + pat.iter().filter(|&&(_, cc)| cc == c).count();
    }
    (rowval, colval, colptr)
}

/// `model_info` (compute stage).
fn model_info(mf: &ModFile, cap: &JsonCapture) -> Value {
    let m = &mf.model;
    // M_.lead_lag_incidence': only the columns of the lags / leads the model has
    let has_lag = m.lead_lag_incidence.iter().any(|r| r[0] > 0);
    let has_lead = m.lead_lag_incidence.iter().any(|r| r[2] > 0);
    let lli: Vec<Vec<usize>> = m
        .lead_lag_incidence
        .iter()
        .map(|r| {
            [(has_lag, r[0]), (true, r[1]), (has_lead, r[2])].iter().filter(|(k, _)| *k).map(|(_, v)| *v).collect()
        })
        .collect();
    let (mut ns, mut nf, mut np, mut nb) = (0, 0, 0, 0);
    for r in &m.lead_lag_incidence {
        match (r[0] > 0, r[2] > 0) {
            (false, false) => ns += 1,
            (false, true) => nf += 1,
            (true, false) => np += 1,
            (true, true) => nb += 1,
        }
    }
    let mx = |f: &dyn Fn(&Dag, NodeId) -> i32| -> i32 {
        m.equations().iter().map(|(l, r)| f(m.dag(), *l).max(f(m.dag(), *r))).max().unwrap_or(0)
    };
    let (el, eld, xl, xld) = (
        mx(&|d, n| d.max_endo_lag(n)),
        mx(&|d, n| d.max_endo_lead(n)),
        mx(&|d, n| d.max_exo_lag(n)),
        mx(&|d, n| d.max_exo_lead(n)),
    );
    // orig_maximum_*: over the model as parsed
    let mut orig = (0i64, 0i64, 0i64, 0i64, 0i64, 0i64);
    for e in &cap.original {
        let mut vars = Vec::new();
        if let Some(a) = &e.ast {
            ast_vars(a, &mut vars);
        }
        for (n, ty, lag) in vars {
            let det = m.exo_index(&n).is_some_and(|k| mf.exo_det.contains(&k));
            let (lg, ld) = match (ty.as_str(), det) {
                ("endogenous", _) => (&mut orig.0, &mut orig.1),
                ("exogenous", false) => (&mut orig.2, &mut orig.3),
                ("exogenous", true) => (&mut orig.4, &mut orig.5),
                _ => continue,
            };
            *lg = (*lg).max(-lag);
            *ld = (*ld).max(lag);
        }
    }
    json!({
        "lead_lag_incidence": lli,
        "nstatic": ns, "nfwrd": nf, "npred": np, "nboth": nb, "nsfwrd": nf + nb, "nspred": np + nb, "ndynamic": m.n_endo() - ns,
        "maximum_endo_lag": el, "maximum_endo_lead": eld, "maximum_exo_lag": xl, "maximum_exo_lead": xld,
        "maximum_exo_det_lag": 0, "maximum_exo_det_lead": 0,
        "maximum_lag": el.max(xl), "maximum_lead": eld.max(xld),
        "orig_maximum_endo_lag": orig.0, "orig_maximum_endo_lead": orig.1,
        "orig_maximum_exo_lag": orig.2, "orig_maximum_exo_lead": orig.3,
        "orig_maximum_exo_det_lag": orig.4, "orig_maximum_exo_det_lead": orig.5,
        "orig_maximum_lag": m.orig_maximum_lag, "orig_maximum_lead": m.orig_maximum_lead,
        "orig_maximum_lag_with_diffs_expanded": m.orig_maximum_lag,
    })
}

/// The dynamic Jacobian's sparse layout, 1-based CSC (`dynamic_g1_sparse_*`).
fn dynamic_sparse(mf: &ModFile) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let m = &mf.model;
    let ncols = 3 * m.n_endo() + m.n_exo() + m.exo_det_names.len();
    let mut pat: Vec<(usize, usize)> = dynamic_pattern(mf);
    pat.sort_by_key(|&(r, c)| (c, r));
    let rowval = pat.iter().map(|&(r, _)| r + 1).collect();
    let colval = pat.iter().map(|&(_, c)| c + 1).collect();
    let mut colptr = vec![1usize; ncols + 1];
    for c in 0..ncols {
        colptr[c + 1] = colptr[c] + pat.iter().filter(|&&(_, cc)| cc == c).count();
    }
    (rowval, colval, colptr)
}

/// The dynamic Jacobian's structural non-zeros (row, column), with the `varexo_det` columns
///  after the exogenous ones.
fn dynamic_pattern(mf: &ModFile) -> Vec<(usize, usize)> {
    let m = &mf.model;
    let mut pat: Vec<(usize, usize)> = m.dynamic_jacobian_pattern().to_vec();
    if !m.exo_det_params.is_empty() {
        let c0 = 3 * m.n_endo() + m.n_exo();
        let mut dag = m.dag().clone();
        for (r, &(l, rr)) in m.equations().iter().enumerate() {
            let res = dag.sub(l, rr);
            for (k, &p) in m.exo_det_params.iter().enumerate() {
                let mut dm = HashMap::new();
                let d = dag.diff(res, DiffVar::Param(p as u32), &mut dm);
                if dag.as_num(d) != Some(0.0) {
                    pat.push((r, c0 + k));
                }
            }
        }
    }
    pat
}

fn names_fn<'a>(mf: &'a ModFile, statik: bool) -> impl Fn(Node) -> String + 'a {
    move |n| {
        let m = &mf.model;
        let lag = |l: i32| if l == 0 || statik { String::new() } else { format!("({l})") };
        match n {
            Node::Endo(j, l) => format!("{}{}", m.endo_names[j as usize], lag(l)),
            Node::Exo(k, l) => format!("{}{}", m.exo_names[k as usize], lag(l)),
            Node::Param(i) => m.param_names.get(i as usize).cloned().unwrap_or_else(|| format!("steady_state({})", i)),
            Node::SteadyState(j, _) => format!("STEADY_STATE({})", m.endo_names[j as usize]),
            _ => String::new(),
        }
    }
}

/// `dynamic.json` / `static.json`: residuals and first derivatives.
fn derivatives(mf: &ModFile, cap: &JsonCapture, statik: bool, simple: bool) -> Value {
    let m = &mf.model;
    let n = m.n_endo();
    let mut dag = m.dag().clone();
    let mut memo = HashMap::new();
    let eqs = m.equations().to_vec();
    let names = names_fn(mf, statik);
    let residuals: Vec<Value> = if statik {
        eqs.iter()
            .map(|&(l, r)| {
                let (l, r) = (dag.to_static(l, &mut memo), dag.to_static(r, &mut memo));
                let (mut a, mut b) = (String::new(), String::new());
                dag_str(&dag, l, &names, &mut a);
                dag_str(&dag, r, &names, &mut b);
                json!({"residual": {"lhs": a, "rhs": b}})
            })
            .collect()
    } else {
        cap.transformed.iter().map(|e| json!({"residual": {"lhs": e.lhs, "rhs": e.rhs}})).collect()
    };
    let mut entries = Vec::new();
    let ncols = if statik { n } else { 3 * n + m.n_exo() + m.exo_det_names.len() };
    if statik {
        let mut memo2 = HashMap::new();
        for (i, &(l, r)) in eqs.iter().enumerate() {
            let res = dag.sub(l, r);
            let res = dag.to_static(res, &mut memo2);
            for j in 0..n {
                let v = dag.endo(j, 0);
                let mut dm = HashMap::new();
                let de = dag.diff(res, DiffVar::Endo(j as u32, 0), &mut dm);
                let _ = v;
                if dag.as_num(de) == Some(0.0) {
                    continue;
                }
                let mut s = String::new();
                dag_str(&dag, de, &names, &mut s);
                let mut e = Map::new();
                e.insert("indices".into(), json!([i + 1, j + 1]));
                if !simple {
                    e.insert("eq".into(), json!(i + 1));
                    e.insert("var".into(), json!(m.endo_names[j]));
                }
                e.insert("val".into(), json!(s));
                entries.push(Value::Object(e));
            }
        }
    } else {
        let mut pat: Vec<(usize, usize)> = dynamic_pattern(mf);
        pat.sort_by_key(|&(r, c)| (r, c));
        let mut cache: HashMap<usize, NodeId> = HashMap::new();
        for (r, c) in pat {
            let res = *cache.entry(r).or_insert_with(|| dag.sub(eqs[r].0, eqs[r].1));
            let (var, shift, dv) = if c < 3 * n {
                let (j, s) = (c % n, (c / n) as i32 - 1);
                (m.endo_names[j].clone(), s, DiffVar::Endo(j as u32, s))
            } else if c < 3 * n + m.n_exo() {
                (m.exo_names[c - 3 * n].clone(), 0, DiffVar::Exo((c - 3 * n) as u32, 0))
            } else {
                let k = c - 3 * n - m.n_exo(); // varexo_det
                (m.exo_det_names[k].clone(), 0, DiffVar::Param(m.exo_det_params[k] as u32))
            };
            let mut dm = HashMap::new();
            let de = dag.diff(res, dv, &mut dm);
            let mut s = String::new();
            dag_str(&dag, de, &names, &mut s);
            let mut e = Map::new();
            e.insert("indices".into(), json!([r + 1, c + 1]));
            if !simple {
                e.insert("eq".into(), json!(r + 1));
                e.insert("var".into(), json!(var));
                e.insert("shift".into(), json!(shift));
            }
            e.insert("val".into(), json!(s));
            entries.push(Value::Object(e));
        }
    }
    let key = if statik { "static_model" } else { "dynamic_model" };
    json!({ key: {
        "model_local_variables": [],
        "residuals": residuals,
        "jacobian": {"dims": [eqs.len(), ncols], "entries": entries},
        "note": "dynare-rust: derivatives printed from dynare-rust's symbolic derivatives (no temporary terms)",
    }})
}

/// Writes the JSON files of `stage` (or prints them with `jsonstdout`).
pub fn write(mf: &ModFile, cap: &JsonCapture, dir: &Path, fname: &str, o: &JsonOptions) -> Result<()> {
    let out_dir = dir.join(fname).join("model").join("json");
    let transformed = matches!(o.stage.as_str(), "transform" | "compute");
    let mut files: Vec<(String, Value)> = Vec::new();
    let build = |eqs: &[JsonEquation], transformed: bool| -> Value {
        let mut m = declarations(mf, transformed);
        m.insert("aux_vars".into(), aux_vars(mf, cap, transformed));
        m.insert("model_local_variables".into(), json!([]));
        m.insert("model".into(), model_entries(eqs, transformed));
        m.insert(
            "xrefs".into(),
            json!({"parameters": [], "endogenous": [], "exogenous": [], "exogenous_deterministic": []}),
        );
        m.insert("abstract_syntax_tree".into(), ast_entries(eqs, mf.model.orig_eq_nbr, &mf.model.exo_det_names));
        m.insert("variable_mapping".into(), variable_mapping(mf, cap));
        if transformed {
            let (rv, cv, cp) = dynamic_sparse(mf);
            m.insert("dynamic_g1_sparse_rowval".into(), json!(rv));
            m.insert("dynamic_g1_sparse_colval".into(), json!(cv));
            m.insert("dynamic_g1_sparse_colptr".into(), json!(cp));
            let (rv, cv, cp) = static_sparse(mf);
            m.insert("static_g1_sparse_rowval".into(), json!(rv));
            m.insert("static_g1_sparse_colval".into(), json!(cv));
            m.insert("static_g1_sparse_colptr".into(), json!(cp));
        }
        if !mf.varobs.is_empty() {
            m.insert("varobs".into(), json!(mf.varobs));
            let ids: Vec<usize> = mf.varobs.iter().filter_map(|v| mf.model.endo_index(v)).map(|j| j + 1).collect();
            m.insert("varobs_ids".into(), json!(ids));
        }
        m.insert("statements".into(), statements(mf));
        if o.stage == "compute" {
            m.insert("model_info".into(), model_info(mf, cap));
        }
        m.insert("steady_state_model".into(), json!(!mf.steady_state_model.is_empty()));
        Value::Object(m)
    };
    if transformed {
        let mut orig = Map::new();
        orig.insert("model".into(), model_entries(&cap.original, true));
        orig.insert("abstract_syntax_tree".into(), ast_entries(&cap.original, usize::MAX, &mf.model.exo_det_names));
        orig.insert("statements".into(), statements(mf));
        files.push(("modfile-original.json".into(), Value::Object(orig)));
        files.push(("modfile.json".into(), build(&cap.transformed, true)));
    } else {
        files.push(("modfile.json".into(), build(&cap.parsed, false)));
    }
    if o.stage == "compute" {
        files.push(("static.json".into(), derivatives(mf, cap, true, o.derivsimple)));
        files.push(("dynamic.json".into(), derivatives(mf, cap, false, o.derivsimple)));
        if !mf.steady_state_model.is_empty() {
            let ss: Vec<Value> = mf.steady_state_model.iter().map(|(l, r)| json!({"lhs": l, "rhs": r})).collect();
            files.push(("steady_state_model.json".into(), json!({"steady_state_model": ss})));
        }
    }
    if !o.stdout {
        std::fs::create_dir_all(&out_dir)?;
    }
    for (name, v) in files {
        let text = serde_json::to_string_pretty(&v)?;
        if o.stdout {
            println!("//-- BEGIN JSON --// {name}\n{text}\n//-- END JSON --//");
        } else {
            std::fs::write(out_dir.join(&name), text)?;
        }
    }
    if !o.stdout {
        println!("JSON written after {} step to {}.", o.stage, out_dir.display());
    }
    Ok(())
}
