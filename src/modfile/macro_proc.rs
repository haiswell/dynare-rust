// SPDX-License-Identifier: GPL-3.0-or-later
//! Dynare's macro processor (the subset used by the models here).
//!
//! Supports `@#define`, `@#if/@#ifdef/@#ifndef/@#elseif/@#else/@#endif`, `@#include`,
//! `@#includepath`, `@#for ... in ... / @#endfor`, `@#echo`, `@#error`, `@{expr}`
//! interpolation and `-D` command-line defines (initial values: a later `@#define` of the
//! same name in the file replaces them). As in Dynare, comments are left to the .mod lexer
//! (see [`super::lexer`]): a directive inside `/* */` is executed and `@{}` inside a comment
//! is expanded; only a `//` comment ending a directive line is dropped here. The output
//! keeps, for each line, its original file and line number so that parse errors can point
//! at the source.

use anyhow::{Context, Result, anyhow, bail};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One line of macro-expanded output with its origin.
#[derive(Clone, Debug)]
pub struct SrcLine {
    pub file: Arc<str>,
    pub line: usize,
    pub text: String,
}

/// A macro-language value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Num(f64),
    Str(String),
    Bool(bool),
    Array(Vec<Value>),
    /// [stream X] `(a, b, c)`.
    Tuple(Vec<Value>),
}

/// [stream X] C++ `setprecision(15)` output of a real (the macro processor's `Real::to_string`).
fn fmt_real(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let s = format!("{:.14e}", v);
    let (mant, exp) = s.split_once('e').unwrap();
    let e: i32 = exp.parse().unwrap();
    let trim = |t: &str| {
        if t.contains('.') { t.trim_end_matches('0').trim_end_matches('.').to_string() } else { t.to_string() }
    };
    // %g: scientific notation when the exponent is < -4 or >= the precision
    if !(-4..15).contains(&e) {
        format!("{}e{}{:02}", trim(mant), if e < 0 { '-' } else { '+' }, e.abs())
    } else {
        trim(&format!("{:.*}", (14 - e).max(0) as usize, v))
    }
}

/// The macro processor's `to_string` (`@{}`, `@#echo`, `(string)` casts): strings are written
/// without quotes, also inside arrays and tuples.
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Num(v) => write!(f, "{}", fmt_real(*v)),
            Value::Str(s) => write!(f, "{s}"),
            Value::Bool(b) => write!(f, "{}", if *b { "true" } else { "false" }),
            Value::Array(a) | Value::Tuple(a) => {
                let (open, close) = if matches!(self, Value::Array(_)) { ("[", "]") } else { ("(", ")") };
                write!(f, "{open}")?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "{close}")
            }
        }
    }
}

impl Value {
    fn truthy(&self) -> Result<bool> {
        match self {
            Value::Bool(b) => Ok(*b),
            Value::Num(v) => Ok(*v != 0.0),
            _ => bail!("macro value {self} is not a boolean"),
        }
    }
    /// A real; booleans are a different type (`true + 1` is an error in Dynare).
    fn num(&self) -> Result<f64> {
        match self {
            Value::Num(v) => Ok(*v),
            _ => bail!("macro value {self} is not a real (operator does not exist for this type)"),
        }
    }
    /// The macro processor's `print` (`@#echomacrovars`): strings are quoted.
    fn print(&self) -> String {
        match self {
            Value::Str(s) => format!("\"{s}\""),
            Value::Array(a) | Value::Tuple(a) => {
                let (open, close) = if matches!(self, Value::Array(_)) { ("[", "]") } else { ("(", ")") };
                let items: Vec<String> = a.iter().map(Value::print).collect();
                format!("{open}{}{close}", items.join(", "))
            }
            v => v.to_string(),
        }
    }
}

/// Runs the macro processor on `path`.
pub fn preprocess(path: &Path, defines: &[(String, String)]) -> Result<Vec<SrcLine>> {
    preprocess_with(path, defines, &[])
}

/// [stream X] [`preprocess`] with extra `@#include` search paths (the `-I` command-line option),
/// searched after the .mod file's directory.
pub fn preprocess_with(path: &Path, defines: &[(String, String)], include_paths: &[PathBuf]) -> Result<Vec<SrcLine>> {
    let base = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut env = Env {
        vars: HashMap::new(),
        funcs: HashMap::new(),
        include_paths: std::iter::once(base.clone()).chain(include_paths.iter().cloned()).collect(),
        depth: 0,
    };
    for (k, v) in defines {
        let val =
            if v.is_empty() { Value::Bool(true) } else { eval_str(v, &env).unwrap_or_else(|_| Value::Str(v.clone())) };
        env.vars.insert(k.clone(), val);
    }
    let lines = read_lines(path)?;
    let mut out = Vec::new();
    process(&lines, &mut env, &mut out)?;
    Ok(out)
}

/// Same as [`preprocess`] on an in-memory source (used by tests).
pub fn preprocess_str(src: &str, file: &str, defines: &[(String, String)]) -> Result<Vec<SrcLine>> {
    let mut env =
        Env { vars: HashMap::new(), funcs: HashMap::new(), include_paths: vec![PathBuf::from(".")], depth: 0 };
    for (k, v) in defines {
        let val = eval_str(v, &env).unwrap_or_else(|_| Value::Str(v.clone()));
        env.vars.insert(k.clone(), val);
    }
    let lines = split_lines(src, file);
    let mut out = Vec::new();
    process(&lines, &mut env, &mut out)?;
    Ok(out)
}

struct Env {
    vars: HashMap<String, Value>,
    /// [stream X] `@#define f(x, y) = EXPR`: parameter names and body.
    funcs: HashMap<String, (Vec<String>, String)>,
    include_paths: Vec<PathBuf>,
    depth: usize,
}

fn read_lines(path: &Path) -> Result<Vec<SrcLine>> {
    let src = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    Ok(split_lines(&src, &path.display().to_string()))
}

fn split_lines(src: &str, file: &str) -> Vec<SrcLine> {
    let file: Arc<str> = Arc::from(file);
    src.lines().enumerate().map(|(i, t)| SrcLine { file: file.clone(), line: i + 1, text: t.to_string() }).collect()
}

/// A directive line without its trailing `//` comment (outside double-quoted strings; Dynare's
/// macro lexer knows no other comment on a directive line).
fn strip_directive_comment(d: &str) -> &str {
    let mut in_str = false;
    for (k, c) in d.char_indices() {
        match c {
            '"' => in_str = !in_str,
            '/' if !in_str && d[k + 1..].starts_with('/') => return &d[..k],
            _ => {}
        }
    }
    d
}

struct IfFrame {
    parent_active: bool,
    taken: bool,
    active: bool,
    loc: String,
}

fn loc(l: &SrcLine) -> String {
    format!("{}:{}", l.file, l.line)
}

fn process(lines: &[SrcLine], env: &mut Env, out: &mut Vec<SrcLine>) -> Result<()> {
    let mut stack: Vec<IfFrame> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let l = &lines[i];
        let active = stack.last().is_none_or(|f| f.active);
        let t = l.text.trim_start();
        let Some(d) = t.strip_prefix("@#") else {
            if active {
                let text = interpolate(&l.text, env).with_context(|| loc(l))?;
                out.push(SrcLine { file: l.file.clone(), line: l.line, text });
            }
            i += 1;
            continue;
        };
        // a directive continues on the next line after a trailing `\\`
        let mut joined = strip_directive_comment(d).trim_start().to_string();
        while let Some(head) = joined.trim_end().strip_suffix("\\\\")
            && let Some(next) = lines.get(i + 1)
        {
            joined = format!("{head} {}", strip_directive_comment(&next.text));
            i += 1;
        }
        let d = joined.as_str();
        let kw_end = d.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(d.len());
        let (kw, rest) = (&d[..kw_end], d[kw_end..].trim());
        let ctx = || format!("{}: in @#{kw}", loc(l));
        match kw {
            "if" | "ifdef" | "ifndef" => {
                let cond = if active {
                    match kw {
                        "if" => eval_str(rest, env).and_then(|v| v.truthy()).with_context(ctx)?,
                        "ifdef" => env.vars.contains_key(rest),
                        _ => !env.vars.contains_key(rest),
                    }
                } else {
                    false
                };
                stack.push(IfFrame { parent_active: active, taken: cond, active: cond, loc: loc(l) });
            }
            "elseif" => {
                let f = stack.last_mut().ok_or_else(|| anyhow!("{}: @#elseif without @#if", loc(l)))?;
                if f.parent_active && !f.taken {
                    let c = eval_str(rest, env).and_then(|v| v.truthy()).with_context(ctx)?;
                    f.active = c;
                    f.taken = c;
                } else {
                    f.active = false;
                }
            }
            "else" => {
                let f = stack.last_mut().ok_or_else(|| anyhow!("{}: @#else without @#if", loc(l)))?;
                f.active = f.parent_active && !f.taken;
                f.taken = true;
            }
            "endif" => {
                stack.pop().ok_or_else(|| anyhow!("{}: @#endif without @#if", loc(l)))?;
            }
            "for" => {
                // find the matching @#endfor
                let mut depth = 0;
                let mut j = i + 1;
                loop {
                    let Some(lj) = lines.get(j) else { bail!("{}: @#for without @#endfor", loc(l)) };
                    let tj = lj.text.trim_start();
                    if let Some(dj) = tj.strip_prefix("@#") {
                        let dj = dj.trim_start();
                        if dj.starts_with("for") && !dj.starts_with("for_") {
                            depth += 1;
                        } else if dj.starts_with("endfor") {
                            if depth == 0 {
                                break;
                            }
                            depth -= 1;
                        }
                    }
                    j += 1;
                }
                if j == i + 1 {
                    // the macro grammar needs at least one statement (a blank line will do)
                    bail!("{}: syntax error, unexpected ENDFOR", loc(&lines[j]));
                }
                if active {
                    // [stream X] tuples (`@#for (i, j) in ...`) and `when` filters
                    let (var, list) =
                        rest.split_once(" in ").ok_or_else(|| anyhow!("{}: expected '@#for VAR in EXPR'", loc(l)))?;
                    let pat = Pattern::parse(var).with_context(ctx)?;
                    let (list, cond) = match list.split_once(" when ") {
                        Some((a, c)) => (a, Some(c.trim())),
                        None => (list, None),
                    };
                    let Value::Array(items) = eval_str(list.trim(), env).with_context(ctx)? else {
                        bail!("{}: in @#for: the index must loop through an array", loc(l));
                    };
                    // Dynare filters the array with the `when` condition first (binding the loop
                    // variables to every element, so a loop without any element kept still
                    // leaves them at the last one), then loops over the elements kept; the loop
                    // variables keep their last values after the loop
                    let mut kept = Vec::new();
                    for v in items {
                        for (n, x) in pat.bind(&v).with_context(ctx)? {
                            env.vars.insert(n, x);
                        }
                        if let Some(c) = cond
                            && !eval_str(c, env).and_then(|v| v.truthy()).with_context(ctx)?
                        {
                            continue;
                        }
                        kept.push(v);
                    }
                    for v in kept {
                        for (n, x) in pat.bind(&v).with_context(ctx)? {
                            env.vars.insert(n, x);
                        }
                        process(&lines[i + 1..j], env, out)?;
                    }
                }
                i = j;
            }
            "endfor" => bail!("{}: @#endfor without @#for", loc(l)),
            _ if !active => {}
            "define" => {
                // [stream X] `@#define NAME` (true) and macro functions `@#define f(x) = EXPR`
                let (lhs, rhs) = match rest.split_once('=') {
                    Some((a, b)) if !b.starts_with('=') => (a, b),
                    _ => (rest, "true"),
                };
                let name = lhs.trim();
                if let Some((fname, args)) = name.split_once('(') {
                    let args = args
                        .trim_end()
                        .strip_suffix(')')
                        .ok_or_else(|| anyhow!("{}: bad macro function head '{name}'", loc(l)))?;
                    let params: Vec<String> =
                        args.split(',').map(|a| a.trim().to_string()).filter(|a| !a.is_empty()).collect();
                    env.funcs.insert(fname.trim().to_string(), (params, rhs.trim().to_string()));
                    i += 1;
                    continue;
                }
                if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    bail!("{}: bad @#define target '{name}'", loc(l));
                }
                let v = eval_str(rhs.trim(), env).with_context(ctx)?;
                env.vars.insert(name.to_string(), v);
            }
            "include" => {
                let v = eval_str(rest, env).with_context(ctx)?;
                let Value::Str(f) = v else { bail!("{}: @#include needs a string", loc(l)) };
                let p = env
                    .include_paths
                    .iter()
                    .map(|d| d.join(&f))
                    .find(|p| p.exists())
                    .ok_or_else(|| anyhow!("{}: cannot find include file '{f}'", loc(l)))?;
                if env.depth > 50 {
                    bail!("{}: @#include nested too deeply", loc(l));
                }
                env.depth += 1;
                let inc = read_lines(&p)?;
                process(&inc, env, out)?;
                env.depth -= 1;
            }
            "includepath" => {
                let v = eval_str(rest, env).with_context(ctx)?;
                let Value::Str(f) = v else { bail!("{}: @#includepath needs a string", loc(l)) };
                let p = PathBuf::from(f);
                let p = if p.is_absolute() { p } else { env.include_paths[0].join(p) };
                env.include_paths.push(p);
            }
            "echo" => eprintln!("{}: @#echo {}", loc(l), eval_str(rest, env).with_context(ctx)?),
            "error" => bail!("{}: @#error {}", loc(l), eval_str(rest, env).with_context(ctx)?),
            "echomacrovars" => {
                // [stream X] display (the `save` option has no options_ to save to here)
                let save = rest.trim_start().starts_with("(save)");
                let list = rest.trim_start().trim_start_matches("(save)").trim();
                let wanted: Vec<&str> = list.split([' ', ',']).filter(|w| !w.is_empty()).collect();
                let mut names: Vec<&String> =
                    env.vars.keys().filter(|k| wanted.is_empty() || wanted.contains(&k.as_str())).collect();
                names.sort();
                if save {
                    // options_.macrovars_line_<n> (written to Output/<fname>_macrovars.json)
                    let vars: Vec<(String, Value)> =
                        names.iter().map(|k| ((*k).clone(), env.vars[*k].clone())).collect();
                    let mut fnames: Vec<&String> =
                        env.funcs.keys().filter(|k| wanted.is_empty() || wanted.contains(&k.as_str())).collect();
                    fnames.sort();
                    let funcs: Vec<(String, String)> = fnames
                        .iter()
                        .map(|k| ((*k).clone(), format!("{k}({}) = {}", env.funcs[*k].0.join(", "), env.funcs[*k].1)))
                        .collect();
                    crate::q4::record_macrovars(l.line, &vars, &funcs);
                    i += 1;
                    continue;
                }
                println!("Macro Variables (at {}):", loc(l));
                for k in names {
                    println!("  {k} = {}", env.vars[k].print());
                }
                let mut fnames: Vec<&String> =
                    env.funcs.keys().filter(|k| wanted.is_empty() || wanted.contains(&k.as_str())).collect();
                fnames.sort();
                for k in fnames {
                    let (a, b) = &env.funcs[k];
                    println!("  {k}({}) = {b}", a.join(", "));
                }
            }
            _ => bail!("{}: unknown macro directive '@#{kw}'", loc(l)),
        }
        i += 1;
    }
    if let Some(f) = stack.last() {
        bail!("{}: @#if not closed by @#endif", f.loc);
    }
    Ok(())
}

/// Replaces every `@{expr}` in `text` by the value of `expr`.
fn interpolate(text: &str, env: &Env) -> Result<String> {
    if !text.contains("@{") {
        return Ok(text.to_string());
    }
    let mut out = String::new();
    let mut rest = text;
    while let Some(p) = rest.find("@{") {
        out.push_str(&rest[..p]);
        let after = &rest[p + 2..];
        let mut depth = 0;
        let mut end = None;
        for (k, c) in after.char_indices() {
            match c {
                '{' => depth += 1,
                '}' if depth == 0 => {
                    end = Some(k);
                    break;
                }
                '}' => depth -= 1,
                _ => {}
            }
        }
        let end = end.ok_or_else(|| anyhow!("unterminated @{{"))?;
        out.push_str(&eval_str(&after[..end], env)?.to_string());
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

// ---------------------------------------------------------------------------------------
// Macro expressions ([stream X]: tuples, comprehensions, set operations, slices, casts,
// user functions, the full function list of the manual)

#[derive(Clone, Debug, PartialEq)]
enum MTok {
    Num(f64),
    Str(String),
    Ident(String),
    Op(&'static str),
}

fn mlex(s: &str) -> Result<Vec<MTok>> {
    const OPS: [&str; 24] = [
        "==", "!=", "<=", ">=", "&&", "||", "+", "-", "*", "/", "^", "(", ")", "[", "]", ",", ":", "<", ">", "!", "=",
        ";", "|", "&",
    ];
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    'outer: while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && b.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let st = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut k = i + 1;
                if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
                    k += 1;
                }
                if k < b.len() && b[k].is_ascii_digit() {
                    i = k;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            out.push(MTok::Num(s[st..i].parse().map_err(|_| anyhow!("bad number '{}'", &s[st..i]))?));
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let st = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(MTok::Ident(s[st..i].to_string()));
            continue;
        }
        // strings are double-quoted only (`'x'` is a syntax error in Dynare's macro language)
        if c == '"' {
            let end = s[i + 1..].find(c).ok_or_else(|| anyhow!("unterminated string in macro expression"))?;
            out.push(MTok::Str(s[i + 1..i + 1 + end].to_string()));
            i += end + 2;
            continue;
        }
        for op in OPS {
            if s[i..].starts_with(op) {
                out.push(MTok::Op(op));
                i += op.len();
                continue 'outer;
            }
        }
        bail!("unexpected character '{c}' in macro expression '{s}'");
    }
    Ok(out)
}

fn eval_str(s: &str, env: &Env) -> Result<Value> {
    let toks = mlex(s)?;
    let mut p = MParser { t: &toks, i: 0, env, locals: Vec::new(), depth: 0 };
    let v = p.expr()?;
    if p.i != toks.len() {
        bail!("trailing tokens in macro expression '{s}'");
    }
    Ok(v)
}

/// A loop or comprehension pattern: a name or a tuple of names.
#[derive(Clone, Debug)]
enum Pattern {
    Name(String),
    Tuple(Vec<String>),
}

impl Pattern {
    fn parse(s: &str) -> Result<Pattern> {
        let s = s.trim();
        if let Some(inner) = s.strip_prefix('(').and_then(|r| r.strip_suffix(')')) {
            let names: Vec<String> = inner.split(',').map(|n| n.trim().to_string()).collect();
            if names.iter().any(|n| n.is_empty() || !n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')) {
                bail!("bad tuple pattern '{s}'");
            }
            return Ok(Pattern::Tuple(names));
        }
        if s.is_empty() || !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            bail!("bad loop variable '{s}'");
        }
        Ok(Pattern::Name(s.to_string()))
    }

    /// The (name, value) bindings of one element.
    fn bind(&self, v: &Value) -> Result<Vec<(String, Value)>> {
        match self {
            Pattern::Name(n) => Ok(vec![(n.clone(), v.clone())]),
            Pattern::Tuple(names) => {
                let (Value::Tuple(items) | Value::Array(items)) = v else { bail!("cannot unpack {v} into a tuple") };
                if items.len() != names.len() {
                    bail!("cannot unpack {v} into {} variables", names.len());
                }
                Ok(names.iter().cloned().zip(items.iter().cloned()).collect())
            }
        }
    }
}

/// The elements an `in` / `for` iterates over.
fn elements(v: Value) -> Result<Vec<Value>> {
    match v {
        Value::Array(a) | Value::Tuple(a) => Ok(a),
        Value::Str(s) => Ok(s.chars().map(|c| Value::Str(c.to_string())).collect()),
        other => bail!("cannot iterate over {other}"),
    }
}

struct MParser<'a> {
    t: &'a [MTok],
    i: usize,
    env: &'a Env,
    /// Comprehension / function-argument bindings, innermost last.
    locals: Vec<(String, Value)>,
    /// Nesting of macro-function calls, bounded by [`MAX_MACRO_CALL_DEPTH`].
    depth: usize,
}

/// Deepest nesting of macro-function calls (catches `@#define f(a) = f(a)`).
const MAX_MACRO_CALL_DEPTH: usize = 100;

impl MParser<'_> {
    fn peek_op(&self, op: &str) -> bool {
        matches!(self.t.get(self.i), Some(MTok::Op(o)) if *o == op)
    }
    fn peek_ident(&self, id: &str) -> bool {
        matches!(self.t.get(self.i), Some(MTok::Ident(s)) if s == id)
    }
    fn eat_op(&mut self, op: &str) -> bool {
        if self.peek_op(op) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect_op(&mut self, op: &str) -> Result<()> {
        if self.eat_op(op) { Ok(()) } else { bail!("expected '{op}' in macro expression") }
    }
    fn lookup(&self, id: &str) -> Option<Value> {
        self.locals
            .iter()
            .rev()
            .find(|(n, _)| n == id)
            .map(|(_, v)| v.clone())
            .or_else(|| self.env.vars.get(id).cloned())
    }

    // precedence (Dynare's macro Parser.yy): || < && < == != < relations < in < : < | < & < + - < * / < unary < ^
    fn expr(&mut self) -> Result<Value> {
        self.or()
    }

    fn or(&mut self) -> Result<Value> {
        let mut a = self.and()?;
        while self.eat_op("||") {
            let b = self.and()?;
            a = Value::Bool(a.truthy()? || b.truthy()?);
        }
        Ok(a)
    }

    fn and(&mut self) -> Result<Value> {
        let mut a = self.equality()?;
        while self.eat_op("&&") {
            let b = self.equality()?;
            a = Value::Bool(a.truthy()? && b.truthy()?);
        }
        Ok(a)
    }

    fn equality(&mut self) -> Result<Value> {
        let mut a = self.relation()?;
        loop {
            if self.eat_op("==") {
                a = Value::Bool(values_eq(&a, &self.relation()?));
            } else if self.eat_op("!=") {
                a = Value::Bool(!values_eq(&a, &self.relation()?));
            } else {
                return Ok(a);
            }
        }
    }

    fn relation(&mut self) -> Result<Value> {
        let mut a = self.membership()?;
        loop {
            let Some(op) = ["<=", ">=", "<", ">"].into_iter().find(|o| self.peek_op(o)) else { return Ok(a) };
            self.i += 1;
            let b = self.membership()?;
            let o = match (&a, &b) {
                (Value::Str(x), Value::Str(y)) => x.partial_cmp(y),
                _ => a.num()?.partial_cmp(&b.num()?),
            };
            // a comparison with NaN is false (C++ doubles)
            a = Value::Bool(o.is_some_and(|o| match op {
                "<" => o.is_lt(),
                ">" => o.is_gt(),
                "<=" => o.is_le(),
                _ => o.is_ge(),
            }));
        }
    }

    fn membership(&mut self) -> Result<Value> {
        let a = self.range()?;
        if self.peek_ident("in") {
            self.i += 1;
            let Value::Array(items) = self.range()? else { bail!("second argument of `in` operator must be an array") };
            return Ok(Value::Bool(items.iter().any(|x| values_eq(x, &a))));
        }
        Ok(a)
    }

    /// `a:b` or `a:s:b`.
    fn range(&mut self) -> Result<Value> {
        let a = self.union()?;
        if !self.eat_op(":") {
            return Ok(a);
        }
        let b = self.union()?;
        let (lo, step, hi) =
            if self.eat_op(":") { (a.num()?, b.num()?, self.union()?.num()?) } else { (a.num()?, 1.0, b.num()?) };
        if !(lo.is_finite() && step.is_finite() && hi.is_finite()) {
            bail!("non-finite bound or step in macro range ({lo}:{step}:{hi})");
        }
        if step != 0.0 && (hi - lo) / step > 1e8 {
            bail!("macro range {lo}:{step}:{hi} has too many elements");
        }
        // Range::eval: repeated addition of the step while within the bound (1:0.1:1.3 is
        // [1, 1.1, 1.2]); a zero step or a step away from the bound gives []
        let mut v = Vec::new();
        let within = |x: f64| if step > 0.0 { x <= hi } else { x >= hi };
        if (step > 0.0 && lo <= hi) || (lo >= hi && step < 0.0) {
            let mut x = lo;
            while within(x) {
                v.push(Value::Num(x));
                if x + step == x {
                    bail!("macro range {lo}:{step}:{hi}: the step is below the precision of the bounds");
                }
                x += step;
            }
        }
        Ok(Value::Array(v))
    }

    fn union(&mut self) -> Result<Value> {
        let mut a = self.intersection()?;
        while self.eat_op("|") {
            let b = self.intersection()?;
            let (Value::Array(mut x), Value::Array(y)) = (a, b) else { bail!("| needs two arrays") };
            for v in y {
                if !x.iter().any(|w| values_eq(w, &v)) {
                    x.push(v);
                }
            }
            a = Value::Array(x);
        }
        Ok(a)
    }

    fn intersection(&mut self) -> Result<Value> {
        let mut a = self.add()?;
        while self.eat_op("&") {
            let b = self.add()?;
            let (Value::Array(x), Value::Array(y)) = (a, b) else { bail!("& needs two arrays") };
            let mut out: Vec<Value> = Vec::new();
            for v in x {
                if y.iter().any(|w| values_eq(w, &v)) && !out.iter().any(|w| values_eq(w, &v)) {
                    out.push(v);
                }
            }
            a = Value::Array(out);
        }
        Ok(a)
    }

    fn add(&mut self) -> Result<Value> {
        let mut a = self.mul()?;
        loop {
            if self.eat_op("+") {
                let b = self.mul()?;
                a = match (a, b) {
                    (Value::Str(x), Value::Str(y)) => Value::Str(x + &y),
                    (Value::Array(mut x), Value::Array(y)) => {
                        x.extend(y);
                        Value::Array(x)
                    }
                    (x, y) => Value::Num(x.num()? + y.num()?),
                };
            } else if self.eat_op("-") {
                let b = self.mul()?;
                a = match (a, b) {
                    // set difference
                    (Value::Array(x), Value::Array(y)) => {
                        Value::Array(x.into_iter().filter(|v| !y.iter().any(|w| values_eq(w, v))).collect())
                    }
                    (x, y) => Value::Num(x.num()? - y.num()?),
                };
            } else {
                return Ok(a);
            }
        }
    }

    fn mul(&mut self) -> Result<Value> {
        let mut a = self.unary()?;
        loop {
            if self.eat_op("*") {
                let b = self.unary()?;
                a = match (a, b) {
                    (Value::Array(x), Value::Array(y)) => Value::Array(cartesian(&x, &y)),
                    (x, y) => Value::Num(x.num()? * y.num()?),
                };
            } else if self.eat_op("/") {
                // IEEE division: 1/0 is inf, as in Dynare
                let b = self.unary()?.num()?;
                a = Value::Num(a.num()? / b);
            } else {
                return Ok(a);
            }
        }
    }

    fn unary(&mut self) -> Result<Value> {
        if self.eat_op("!") {
            return Ok(Value::Bool(!self.unary()?.truthy()?));
        }
        if self.eat_op("-") {
            return Ok(Value::Num(-self.unary()?.num()?));
        }
        if self.eat_op("+") {
            return self.unary();
        }
        self.power()
    }

    fn power(&mut self) -> Result<Value> {
        let a = self.primary()?;
        if self.eat_op("^") {
            let b = if self.eat_op("-") { Value::Num(-self.primary()?.num()?) } else { self.primary()? };
            return match a {
                // cartesian power of an array
                Value::Array(x) => {
                    let n = b.num()? as usize;
                    if n == 0 {
                        return Ok(Value::Array(vec![]));
                    }
                    let mut acc: Vec<Value> = x.iter().map(|v| Value::Tuple(vec![v.clone()])).collect();
                    for _ in 1..n {
                        acc = acc
                            .iter()
                            .flat_map(|t| {
                                x.iter().map(move |v| {
                                    let Value::Tuple(mut items) = t.clone() else { unreachable!() };
                                    items.push(v.clone());
                                    Value::Tuple(items)
                                })
                            })
                            .collect();
                    }
                    if n == 1 {
                        acc = x.clone();
                    }
                    Ok(Value::Array(acc))
                }
                a => Ok(Value::Num(a.num()?.powf(b.num()?))),
            };
        }
        Ok(a)
    }

    /// `NAME[k]` after the `[`: the grammar only indexes a variable, once (`[1, 2][1]`,
    /// `v[1][2]` and `f(x)[1]` are syntax errors in Dynare).
    fn index(&mut self, a: Value) -> Result<Value> {
        // `a[i]`, `a[i:j]`, `a[1, 8, 17:19]` (a comma-separated list of indices and ranges)
        let mut keys = vec![self.expr()?];
        while self.eat_op(",") {
            keys.push(self.expr()?);
        }
        self.expect_op("]")?;
        let mut idx: Vec<usize> = Vec::new();
        for k in &keys {
            match k {
                Value::Array(v) => {
                    for x in v {
                        idx.push(x.num()? as usize);
                    }
                }
                x => idx.push(x.num()? as usize),
            }
        }
        let single = keys.len() == 1 && !matches!(keys[0], Value::Array(_));
        Ok(match &a {
            Value::Array(items) | Value::Tuple(items) => {
                let picked: Vec<Value> = idx
                    .iter()
                    .map(|&k| {
                        items.get(k.wrapping_sub(1)).cloned().ok_or_else(|| anyhow!("macro index {k} out of bounds"))
                    })
                    .collect::<Result<_>>()?;
                if single { picked.into_iter().next().unwrap() } else { Value::Array(picked) }
            }
            Value::Str(s) => {
                let chars: Vec<char> = s.chars().collect();
                let picked: String = idx
                    .iter()
                    .map(|&k| {
                        chars.get(k.wrapping_sub(1)).copied().ok_or_else(|| anyhow!("macro index {k} out of bounds"))
                    })
                    .collect::<Result<_>>()?;
                Value::Str(picked)
            }
            _ => bail!("indexing a non-array macro value"),
        })
    }

    /// The tokens up to the matching closing bracket (not consumed).
    fn find_top(&self, stop: &[&str], words: &[&str]) -> Option<usize> {
        let mut depth = 0i32;
        for (k, t) in self.t[self.i..].iter().enumerate() {
            match t {
                MTok::Op("(") | MTok::Op("[") => depth += 1,
                MTok::Op(")") | MTok::Op("]") => {
                    if depth == 0 {
                        return None;
                    }
                    depth -= 1
                }
                MTok::Op(o) if depth == 0 && stop.contains(o) => return Some(self.i + k),
                MTok::Ident(w) if depth == 0 && words.contains(&w.as_str()) => return Some(self.i + k),
                _ => {}
            }
        }
        None
    }

    /// `[ ... ]` after the `[`: an array literal or a comprehension.
    fn bracket(&mut self) -> Result<Value> {
        if self.eat_op("]") {
            return Ok(Value::Array(vec![]));
        }
        // comprehension: [expr for PAT in A (when C)] or [PAT in A when C]
        if let Some(k) = self.find_top(&[], &["for"]) {
            let body = (self.i, k);
            self.i = k + 1;
            let pat = self.pattern()?;
            if !self.peek_ident("in") {
                bail!("expected 'in' in comprehension");
            }
            self.i += 1;
            let src = elements(self.union_until_when()?)?;
            let cond = self.when_clause()?;
            self.expect_op("]")?;
            let end = self.i;
            let mut out = Vec::new();
            for v in src {
                let n = self.locals.len();
                self.locals.extend(pat.bind(&v)?);
                let keep = match cond {
                    Some((a, b)) => self.sub_eval(a, b)?.truthy()?,
                    None => true,
                };
                if keep {
                    out.push(self.sub_eval(body.0, body.1)?);
                }
                self.locals.truncate(n);
            }
            self.i = end;
            return Ok(Value::Array(out));
        }
        if self.find_top(&[], &["when"]).is_some() {
            let pat = self.pattern()?;
            if !self.peek_ident("in") {
                bail!("expected 'in' in comprehension");
            }
            self.i += 1;
            let src = elements(self.union_until_when()?)?;
            let cond = self.when_clause()?.ok_or_else(|| anyhow!("expected 'when' in comprehension"))?;
            self.expect_op("]")?;
            let end = self.i;
            let mut out = Vec::new();
            for v in src {
                let n = self.locals.len();
                self.locals.extend(pat.bind(&v)?);
                if self.sub_eval(cond.0, cond.1)?.truthy()? {
                    out.push(v);
                }
                self.locals.truncate(n);
            }
            self.i = end;
            return Ok(Value::Array(out));
        }
        // a range inside brackets is an element: [1:3] == [[1, 2, 3]]
        let mut items = Vec::new();
        loop {
            items.push(self.expr()?);
            if self.eat_op("]") {
                break;
            }
            self.expect_op(",")?;
        }
        Ok(Value::Array(items))
    }

    fn pattern(&mut self) -> Result<Pattern> {
        if self.eat_op("(") {
            let mut names = Vec::new();
            loop {
                let Some(MTok::Ident(n)) = self.t.get(self.i).cloned() else { bail!("bad tuple pattern") };
                self.i += 1;
                names.push(n);
                if self.eat_op(")") {
                    break;
                }
                self.expect_op(",")?;
            }
            return Ok(Pattern::Tuple(names));
        }
        let Some(MTok::Ident(n)) = self.t.get(self.i).cloned() else { bail!("expected a loop variable") };
        self.i += 1;
        Ok(Pattern::Name(n))
    }

    /// The source of a comprehension (stops before `when`).
    fn union_until_when(&mut self) -> Result<Value> {
        let stop = self.find_top(&[], &["when"]).unwrap_or(usize::MAX);
        let close = {
            let mut depth = 0i32;
            let mut end = self.t.len();
            for (k, t) in self.t[self.i..].iter().enumerate() {
                match t {
                    MTok::Op("(") | MTok::Op("[") => depth += 1,
                    MTok::Op(")") | MTok::Op("]") => {
                        if depth == 0 {
                            end = self.i + k;
                            break;
                        }
                        depth -= 1
                    }
                    _ => {}
                }
            }
            end
        };
        let end = stop.min(close);
        let v = self.sub_eval(self.i, end)?;
        self.i = end;
        Ok(v)
    }

    fn when_clause(&mut self) -> Result<Option<(usize, usize)>> {
        if !self.peek_ident("when") {
            return Ok(None);
        }
        self.i += 1;
        let st = self.i;
        let mut depth = 0i32;
        while let Some(t) = self.t.get(self.i) {
            match t {
                MTok::Op("(") | MTok::Op("[") => depth += 1,
                MTok::Op(")") | MTok::Op("]") => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1
                }
                _ => {}
            }
            self.i += 1;
        }
        Ok(Some((st, self.i)))
    }

    /// Evaluates the tokens `[a, b)` with the current bindings.
    fn sub_eval(&mut self, a: usize, b: usize) -> Result<Value> {
        let mut p = MParser { t: &self.t[a..b], i: 0, env: self.env, locals: self.locals.clone(), depth: self.depth };
        let v = p.expr()?;
        if p.i != b - a {
            bail!("trailing tokens in macro expression");
        }
        Ok(v)
    }

    fn primary(&mut self) -> Result<Value> {
        let t = self.t.get(self.i).cloned().ok_or_else(|| anyhow!("unexpected end of macro expression"))?;
        self.i += 1;
        match t {
            MTok::Num(v) => Ok(Value::Num(v)),
            MTok::Str(s) => Ok(Value::Str(s)),
            MTok::Op("(") => {
                // casts: (bool) x, (real) x, (string) x, (tuple) x, (array) x
                if let (Some(MTok::Ident(ty)), Some(MTok::Op(")"))) = (self.t.get(self.i), self.t.get(self.i + 1))
                    && matches!(ty.as_str(), "bool" | "real" | "string" | "tuple" | "array")
                {
                    let ty = ty.clone();
                    self.i += 2;
                    let v = self.unary()?;
                    return cast(&ty, v);
                }
                let v = self.expr()?;
                if self.eat_op(",") {
                    let mut items = vec![v];
                    if !self.peek_op(")") {
                        loop {
                            items.push(self.expr()?);
                            if !self.eat_op(",") {
                                break;
                            }
                        }
                    }
                    self.expect_op(")")?;
                    return Ok(Value::Tuple(items));
                }
                self.expect_op(")")?;
                Ok(v)
            }
            MTok::Op("[") => self.bracket(),
            MTok::Ident(id) => {
                if self.eat_op("(") {
                    if id == "defined" {
                        let Some(MTok::Ident(n)) = self.t.get(self.i).cloned() else { bail!("defined() needs a name") };
                        self.i += 1;
                        self.expect_op(")")?;
                        return Ok(Value::Bool(self.lookup(&n).is_some()));
                    }
                    let mut args = Vec::new();
                    if !self.eat_op(")") {
                        loop {
                            args.push(self.expr()?);
                            if self.eat_op(")") {
                                break;
                            }
                            self.expect_op(",")?;
                        }
                    }
                    if let Some((params, body)) = self.env.funcs.get(&id) {
                        if params.len() != args.len() {
                            bail!("macro function {id} takes {} argument(s)", params.len());
                        }
                        if self.depth >= MAX_MACRO_CALL_DEPTH {
                            bail!(
                                "macro function {id}: more than {MAX_MACRO_CALL_DEPTH} nested calls (infinite recursion?)"
                            );
                        }
                        // the body sees its own arguments and the global variables, not the
                        // caller's arguments (tests/macro_processor/example1_macro.mod: in
                        // `g(a) = f(1) + a`, `f` reads the global `a`)
                        let locals = params.iter().cloned().zip(args).collect();
                        let mut p = MParser { t: &[], i: 0, env: self.env, locals, depth: self.depth + 1 };
                        let toks = mlex(body)?;
                        p.t = &toks;
                        let v = p.expr()?;
                        if p.i != toks.len() {
                            bail!("trailing tokens in macro function {id}");
                        }
                        return Ok(v);
                    }
                    return call(&id, &args);
                }
                match id.as_str() {
                    "true" => Ok(Value::Bool(true)),
                    "false" => Ok(Value::Bool(false)),
                    // number literals of the macro lexer
                    "inf" => Ok(Value::Num(f64::INFINITY)),
                    "nan" => Ok(Value::Num(f64::NAN)),
                    _ => {
                        let v = self.lookup(&id).ok_or_else(|| anyhow!("unknown macro variable '{id}'"))?;
                        if self.eat_op("[") { self.index(v) } else { Ok(v) }
                    }
                }
            }
            MTok::Op(o) => bail!("unexpected '{o}' in macro expression"),
        }
    }
}

fn cartesian(x: &[Value], y: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for a in x {
        for b in y {
            let mut t = match a {
                Value::Tuple(items) => items.clone(),
                v => vec![v.clone()],
            };
            t.push(b.clone());
            out.push(Value::Tuple(t));
        }
    }
    out
}

fn cast(ty: &str, v: Value) -> Result<Value> {
    Ok(match ty {
        "bool" => match v {
            // String::toBool: "true" / "false" in any case, else a non-zero number
            Value::Str(s) if s.trim().eq_ignore_ascii_case("true") => Value::Bool(true),
            Value::Str(s) if s.trim().eq_ignore_ascii_case("false") => Value::Bool(false),
            Value::Str(s) => {
                Value::Bool(s.trim().parse::<f64>().map_err(|_| anyhow!("cannot cast \"{s}\" to bool"))? != 0.0)
            }
            v => Value::Bool(v.truthy()?),
        },
        "real" => match v {
            Value::Str(s) => Value::Num(s.trim().parse().map_err(|_| anyhow!("cannot cast \"{s}\" to real"))?),
            Value::Array(a) | Value::Tuple(a) if a.len() == 1 => Value::Num(a[0].num()?),
            Value::Bool(b) => Value::Num(if b { 1.0 } else { 0.0 }),
            v => Value::Num(v.num()?),
        },
        "string" => Value::Str(v.to_string()),
        "tuple" => match v {
            Value::Array(a) | Value::Tuple(a) => Value::Tuple(a),
            v => Value::Tuple(vec![v]),
        },
        _ => match v {
            Value::Array(a) | Value::Tuple(a) => Value::Array(a),
            v => Value::Array(vec![v]),
        },
    })
}

/// `==` of the macro language: values of different types (a real and a boolean included) differ.
fn values_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Array(x), Value::Array(y)) | (Value::Tuple(x), Value::Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| values_eq(p, q))
        }
        _ => a == b,
    }
}

// erf/erfc from the system libm, as the preprocessor's std::erf/std::erfc
unsafe extern "C" {
    #[link_name = "erf"]
    fn libm_erf(x: f64) -> f64;
    #[link_name = "erfc"]
    fn libm_erfc(x: f64) -> f64;
}

fn erf(x: f64) -> f64 {
    unsafe { libm_erf(x) }
}

fn call(f: &str, args: &[Value]) -> Result<Value> {
    let one = || -> Result<f64> {
        if args.len() != 1 {
            bail!("{f}() takes one argument");
        }
        args[0].num()
    };
    let n = |v: f64| Value::Num(v);
    Ok(match f {
        "length" => match args {
            [Value::Array(a) | Value::Tuple(a)] => n(a.len() as f64),
            [Value::Str(s)] => n(s.chars().count() as f64),
            _ => bail!("length() needs an array, tuple or string"),
        },
        "isempty" | "empty" => match args {
            [Value::Array(a) | Value::Tuple(a)] => Value::Bool(a.is_empty()),
            [Value::Str(s)] => Value::Bool(s.is_empty()),
            _ => bail!("{f}() needs an array, tuple or string"),
        },
        "sum" => match args {
            [Value::Array(a) | Value::Tuple(a)] => n(a.iter().map(Value::num).sum::<Result<f64>>()?),
            _ => bail!("sum() needs an array"),
        },
        "isboolean" => Value::Bool(matches!(args, [Value::Bool(_)])),
        "isreal" => Value::Bool(matches!(args, [Value::Num(_)])),
        "isstring" => Value::Bool(matches!(args, [Value::Str(_)])),
        "istuple" => Value::Bool(matches!(args, [Value::Tuple(_)])),
        "isarray" => Value::Bool(matches!(args, [Value::Array(_)])),
        "floor" => n(one()?.floor()),
        "ceil" => n(one()?.ceil()),
        "trunc" => n(one()?.trunc()),
        "round" => n(one()?.round()),
        "abs" => n(one()?.abs()),
        "sign" => n({
            let x = one()?;
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }),
        "sqrt" => n(one()?.sqrt()),
        "cbrt" => n(one()?.cbrt()),
        "exp" => n(one()?.exp()),
        "log" | "ln" => n(one()?.ln()),
        "log10" => n(one()?.log10()),
        "sin" => n(one()?.sin()),
        "cos" => n(one()?.cos()),
        "tan" => n(one()?.tan()),
        "asin" => n(one()?.asin()),
        "acos" => n(one()?.acos()),
        "atan" => n(one()?.atan()),
        "erf" => n(erf(one()?)),
        "erfc" => n(unsafe { libm_erfc(one()?) }),
        "gamma" => n(statrs::function::gamma::gamma(one()?)),
        "lgamma" => n(statrs::function::gamma::ln_gamma(one()?)),
        "mod" => match args {
            [a, b] => {
                // std::fmod: the sign of the dividend, mod(-7, 3) = -1, mod(x, 0) = nan
                n(a.num()? % b.num()?)
            }
            _ => bail!("mod() takes two arguments"),
        },
        "normpdf" => match args {
            [x] => n((-0.5 * x.num()?.powi(2)).exp() / (2.0 * std::f64::consts::PI).sqrt()),
            [x, m, s] => {
                let (x, m, s) = (x.num()?, m.num()?, s.num()?);
                n((-0.5 * ((x - m) / s).powi(2)).exp() / (s * (2.0 * std::f64::consts::PI).sqrt()))
            }
            _ => bail!("normpdf() takes 1 or 3 arguments"),
        },
        "normcdf" => match args {
            [x] => n(0.5 * (1.0 + erf(x.num()? / std::f64::consts::SQRT_2))),
            [x, m, s] => n(0.5 * (1.0 + erf((x.num()? - m.num()?) / s.num()? / std::f64::consts::SQRT_2))),
            _ => bail!("normcdf() takes 1 or 3 arguments"),
        },
        // binary operators on reals in Dynare's grammar (std::max / std::min)
        "max" | "min" => match args {
            [Value::Num(a), Value::Num(b)] => n(match f {
                "max" if a < b => *b,
                "min" if b < a => *b,
                _ => *a,
            }),
            [_, _] => bail!("operator `{f}` does not exist for this type"),
            _ => bail!("{f}() takes two arguments"),
        },
        _ => bail!("unsupported macro function '{f}'"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str, defs: &[(&str, &str)]) -> Vec<String> {
        let defs: Vec<(String, String)> = defs.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
        preprocess_str(src, "t.mod", &defs)
            .unwrap()
            .into_iter()
            .map(|l| l.text)
            .filter(|t| !t.trim().is_empty())
            .collect()
    }

    #[test]
    fn defines_and_conditionals() {
        let src = "@#ifndef STAGE\n@#define STAGE = 3\n@#endif\n@#if STAGE == 1\none;\n@#elseif STAGE == 2\ntwo;\n@#else\nother @{STAGE};\n@#endif\n";
        assert_eq!(run(src, &[]), vec!["other 3;"]);
        assert_eq!(run(src, &[("STAGE", "2")]), vec!["two;"]);
        assert_eq!(run(src, &[("STAGE", "1")]), vec!["one;"]);
        assert_eq!(run(src, &[("STAGE", "9")]), vec!["other 9;"]);
    }

    #[test]
    fn command_line_define_is_an_initial_value() {
        // dynare-preprocessor -DX=5: `c0 = 5; c1 = 1;`
        let src = "c0 = @{X};
@#define X = 1
c1 = @{X};
";
        assert_eq!(run(src, &[("X", "5")]), vec!["c0 = 5;", "c1 = 1;"]);
    }

    #[test]
    fn nested_if_and_logic() {
        let src = "@#define A = true\n@#define B = 2\n@#if A && (B > 1 || B < -1)\n@#ifdef A\nyes;\n@#endif\n@#if !A\nno;\n@#endif\n@#endif\n";
        assert_eq!(run(src, &[]), vec!["yes;"]);
        // undefined variables in an inactive branch are not evaluated
        let src = "@#if false\n@#if UNDEFINED == 1\nx;\n@#endif\n@#endif\nok;";
        assert_eq!(run(src, &[]), vec!["ok;"]);
    }

    #[test]
    fn for_loops_strings_and_arrays() {
        let src = "@#define names = [\"a\", \"b\"]\n@#for n in names\nvar @{n};\n@#endfor\n@#for i in 1:3\nx@{i};\n@#endfor\n@#define s = \"p\" + \"q\"\n@{s} @{length(names)}\n";
        assert_eq!(run(src, &[]), vec!["var a;", "var b;", "x1;", "x2;", "x3;", "pq 2"]);
    }

    #[test]
    fn comments_are_left_to_the_lexer() {
        // the directive inside /* */ is executed, @{} is expanded inside comments, and only a
        // `//` comment ends a directive line (dynare-preprocessor savemacro)
        let src = "/*
@#define X = 99 // c
*/
x = 1; // c @{X}
@#define Y = \"a//b\" // c
y = @{Y}; % @{Y}
";
        assert_eq!(run(src, &[]), vec!["/*", "*/", "x = 1; // c 99", "y = a//b; % a//b"]);
    }

    #[test]
    fn errors_have_locations() {
        let e = preprocess_str("\n@#if 1\nx\n", "m.mod", &[]).unwrap_err();
        assert!(format!("{e:#}").contains("m.mod:2"), "{e:#}");
        let e = preprocess_str("@#if Y == 1\n@#endif\n", "m.mod", &[]).unwrap_err();
        assert!(format!("{e:#}").contains("m.mod:1"), "{e:#}");
    }
}
