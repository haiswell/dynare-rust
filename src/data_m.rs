// SPDX-License-Identifier: GPL-3.0-or-later
//! [stream X] `.m` datafiles (`load_m_file_data_legacy.m`): Dynare `eval`s the script and
//! keeps the variables named in `varobs`, which must be vectors of equal length. dynare-rust
//! interprets the subset of MATLAB that such scripts use: assignments of numeric matrices
//! (`x = [1 2; 3 4];`, rows on separate lines), arithmetic (`+ - * / .* ./ .^ ^`, transposes),
//! ranges, indexing (`A(:,2)`, `x(2:end)`, `x(k) = v`), and the functions `log`, `exp`,
//! `sqrt`, `abs`, `diff`, `cumsum`, `cumprod`, `sum`, `mean`, `zeros`, `ones`, `nan`/`NaN`,
//! `log10`, `round`, `floor`, `ceil`, `length`, `numel`, `size`, `transpose`, `vec`,
//! `min`/`max` (element-wise with two arguments), `repmat`, `reshape(A,m,n)`, `isnan`. `clear`, `clc`, `close`,
//! `disp` and `format` statements are ignored.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

/// A real matrix, column-major.
#[derive(Clone, Debug, PartialEq)]
pub struct M {
    pub r: usize,
    pub c: usize,
    pub v: Vec<f64>,
}

impl M {
    fn scalar(x: f64) -> M {
        M { r: 1, c: 1, v: vec![x] }
    }
    fn filled(r: usize, c: usize, x: f64) -> M {
        M { r, c, v: vec![x; r * c] }
    }
    fn is_scalar(&self) -> bool {
        self.v.len() == 1
    }
    fn is_vector(&self) -> bool {
        self.r == 1 || self.c == 1
    }
    fn get(&self, i: usize, j: usize) -> f64 {
        self.v[j * self.r + i]
    }
    fn t(&self) -> M {
        let mut v = Vec::with_capacity(self.v.len());
        for i in 0..self.r {
            for j in 0..self.c {
                v.push(self.get(i, j));
            }
        }
        M { r: self.c, c: self.r, v }
    }
    fn map(&self, f: impl Fn(f64) -> f64) -> M {
        M { r: self.r, c: self.c, v: self.v.iter().map(|&x| f(x)).collect() }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum T {
    Num(f64),
    Id(String),
    Str(String),
    Op(&'static str),
    Nl,
}

#[derive(Clone, Debug)]
struct Tok {
    t: T,
    space_before: bool,
    line: usize,
}

const OPS: [&str; 22] = [
    ".*", "./", ".^", ".'", "==", "~=", "<=", ">=", "+", "-", "*", "/", "^", "'", "(", ")", "[", "]", ",", ";", ":",
    "=",
];

fn lex(src: &str) -> Result<Vec<Tok>> {
    let mut out: Vec<Tok> = Vec::new();
    let mut in_block_comment = false;
    for (ln, raw_line) in src.lines().enumerate() {
        let line_no = ln + 1;
        let trimmed = raw_line.trim();
        if in_block_comment {
            if trimmed == "%}" {
                in_block_comment = false;
            }
            continue;
        }
        if trimmed == "%{" {
            in_block_comment = true;
            continue;
        }
        let b: Vec<char> = raw_line.chars().collect();
        let mut i = 0;
        let mut space = true;
        let mut continued = false;
        while i < b.len() {
            let ch = b[i];
            if ch == '%' || ch == '#' {
                break;
            }
            if ch.is_whitespace() {
                space = true;
                i += 1;
                continue;
            }
            if ch == '.' && b.get(i + 1) == Some(&'.') && b.get(i + 2) == Some(&'.') {
                continued = true;
                break;
            }
            if ch.is_ascii_digit() || (ch == '.' && b.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
                let st = i;
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.') {
                    // `1./x`: the dot belongs to the operator
                    if b[i] == '.' && matches!(b.get(i + 1), Some('*' | '/' | '^' | '\'')) {
                        break;
                    }
                    i += 1;
                }
                if i < b.len() && (b[i] == 'e' || b[i] == 'E' || b[i] == 'd' || b[i] == 'D') {
                    let mut k = i + 1;
                    if k < b.len() && (b[k] == '+' || b[k] == '-') {
                        k += 1;
                    }
                    if k < b.len() && b[k].is_ascii_digit() {
                        i = k;
                        while i < b.len() && b[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
                let s: String = b[st..i].iter().collect::<String>().replace(['d', 'D'], "e");
                let v: f64 = s.parse().map_err(|_| anyhow!("line {line_no}: invalid number {s}"))?;
                out.push(Tok { t: T::Num(v), space_before: space, line: line_no });
                space = false;
                continue;
            }
            if ch.is_alphabetic() || ch == '_' {
                let st = i;
                while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_') {
                    i += 1;
                }
                out.push(Tok { t: T::Id(b[st..i].iter().collect()), space_before: space, line: line_no });
                space = false;
                continue;
            }
            // a quote is a transpose right after a value, else a string
            if ch == '\'' || ch == '"' {
                let after_value = !space
                    && out.last().is_some_and(|t| {
                        matches!(t.t, T::Num(_) | T::Id(_) | T::Op(")") | T::Op("]") | T::Op("'") | T::Op(".'"))
                            && t.line == line_no
                    });
                if ch == '\'' && after_value {
                    out.push(Tok { t: T::Op("'"), space_before: space, line: line_no });
                    i += 1;
                    space = false;
                    continue;
                }
                let st = i + 1;
                let mut k = st;
                while k < b.len() && b[k] != ch {
                    k += 1;
                }
                out.push(Tok { t: T::Str(b[st..k.min(b.len())].iter().collect()), space_before: space, line: line_no });
                i = k + 1;
                space = false;
                continue;
            }
            let rest: String = b[i..b.len().min(i + 2)].iter().collect();
            let Some(op) = OPS.iter().find(|o| rest.starts_with(**o)) else {
                bail!("line {line_no}: unexpected character '{ch}'")
            };
            out.push(Tok { t: T::Op(op), space_before: space, line: line_no });
            i += op.len();
            space = false;
        }
        if !continued {
            out.push(Tok { t: T::Nl, space_before: true, line: line_no });
        }
    }
    Ok(out)
}

struct Interp<'a> {
    t: &'a [Tok],
    i: usize,
    vars: HashMap<String, M>,
    /// Inside `[ ]`: blanks separate elements and newlines separate rows.
    in_matrix: usize,
    /// Size of the indexed dimension, for `end` inside an index.
    end_value: Vec<f64>,
}

fn broadcast(a: &M, b: &M, f: impl Fn(f64, f64) -> f64, what: &str) -> Result<M> {
    if a.is_scalar() {
        return Ok(b.map(|y| f(a.v[0], y)));
    }
    if b.is_scalar() {
        return Ok(a.map(|x| f(x, b.v[0])));
    }
    if a.r != b.r || a.c != b.c {
        bail!("{what}: matrix dimensions must agree ({}x{} and {}x{})", a.r, a.c, b.r, b.c);
    }
    Ok(M { r: a.r, c: a.c, v: a.v.iter().zip(&b.v).map(|(&x, &y)| f(x, y)).collect() })
}

fn matmul(a: &M, b: &M) -> Result<M> {
    if a.is_scalar() || b.is_scalar() {
        return broadcast(a, b, |x, y| x * y, "*");
    }
    if a.c != b.r {
        bail!("*: inner matrix dimensions must agree");
    }
    let mut out = M::filled(a.r, b.c, 0.0);
    for j in 0..b.c {
        for k in 0..a.c {
            let bkj = b.get(k, j);
            for i in 0..a.r {
                out.v[j * a.r + i] += a.get(i, k) * bkj;
            }
        }
    }
    Ok(out)
}

impl<'a> Interp<'a> {
    fn peek(&self) -> Option<&'a Tok> {
        self.t.get(self.i)
    }
    fn at(&self, op: &str) -> bool {
        self.peek().is_some_and(|t| matches!(&t.t, T::Op(o) if *o == op))
    }
    fn eat(&mut self, op: &str) -> bool {
        if self.at(op) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn line(&self) -> usize {
        self.peek().or(self.t.last()).map_or(0, |t| t.line)
    }
    fn expect(&mut self, op: &str) -> Result<()> {
        if !self.eat(op) {
            bail!("line {}: expected '{op}'", self.line());
        }
        Ok(())
    }

    fn run(&mut self) -> Result<()> {
        while self.i < self.t.len() {
            if matches!(self.peek().map(|t| &t.t), Some(T::Nl) | Some(T::Op(";")) | Some(T::Op(","))) {
                self.i += 1;
                continue;
            }
            self.statement()?;
        }
        Ok(())
    }

    fn skip_statement(&mut self) {
        while let Some(t) = self.peek() {
            if matches!(t.t, T::Nl | T::Op(";") | T::Op(",")) {
                break;
            }
            self.i += 1;
        }
    }

    fn statement(&mut self) -> Result<()> {
        let line = self.line();
        let Some(T::Id(name)) = self.peek().map(|t| t.t.clone()) else {
            bail!("line {line}: expected an assignment");
        };
        if matches!(name.as_str(), "clear" | "clc" | "close" | "disp" | "format" | "warning" | "fprintf") {
            self.skip_statement();
            return Ok(());
        }
        self.i += 1;
        // name(idx) = value
        let mut index: Option<Vec<IndexArg>> = None;
        if self.at("(") {
            self.i += 1;
            let cur = self.vars.get(&name).cloned();
            index = Some(self.index_args(cur.as_ref())?);
        }
        if !self.eat("=") {
            bail!("line {line}: only assignments are supported in .m datafiles");
        }
        let v = self.expr()?;
        match index {
            None => {
                self.vars.insert(name, v);
            }
            Some(args) => {
                let mut cur = self.vars.remove(&name).unwrap_or(M { r: 0, c: 0, v: vec![] });
                assign_index(&mut cur, &args, &v).with_context(|| format!("line {line}: {name}(...) = ..."))?;
                self.vars.insert(name, cur);
            }
        }
        Ok(())
    }

    fn index_args(&mut self, target: Option<&M>) -> Result<Vec<IndexArg>> {
        let mut args = Vec::new();
        let saved = self.in_matrix;
        self.in_matrix = 0;
        let mut k = 0;
        let nargs = self.count_args_ahead();
        loop {
            if self.eat(")") {
                break;
            }
            if self.at(":") && self.t.get(self.i + 1).is_some_and(|t| matches!(t.t, T::Op(",") | T::Op(")"))) {
                self.i += 1;
                args.push(IndexArg::All);
            } else {
                // `end` = size along this dimension (the number of elements with one index)
                let endv = target.map_or(0.0, |m| match (k, nargs) {
                    (_, 1) => m.v.len() as f64,
                    (0, _) => m.r as f64,
                    _ => m.c as f64,
                });
                self.end_value.push(endv);
                let e = self.expr();
                self.end_value.pop();
                args.push(IndexArg::Idx(e?));
            }
            k += 1;
            if !self.eat(",") {
                self.expect(")")?;
                break;
            }
        }
        self.in_matrix = saved;
        Ok(args)
    }

    /// Number of top-level arguments of the index list starting at the current token.
    fn count_args_ahead(&self) -> usize {
        let mut depth = 0i32;
        let mut n = 1;
        for t in &self.t[self.i..] {
            match &t.t {
                T::Op("(") | T::Op("[") => depth += 1,
                T::Op(")") | T::Op("]") => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1
                }
                T::Op(",") if depth == 0 => n += 1,
                T::Nl => break,
                _ => {}
            }
        }
        n
    }

    // precedence: comparison < range < +- < */ < unary < ^ < postfix
    fn expr(&mut self) -> Result<M> {
        let a = self.range()?;
        for (op, f) in [("==", 0), ("~=", 1), ("<=", 2), (">=", 3)] {
            if self.at(op) && !self.matrix_break() {
                self.i += 1;
                let b = self.range()?;
                return broadcast(
                    &a,
                    &b,
                    |x, y| {
                        let r = match f {
                            0 => x == y,
                            1 => x != y,
                            2 => x <= y,
                            _ => x >= y,
                        };
                        r as u8 as f64
                    },
                    op,
                );
            }
        }
        Ok(a)
    }

    fn range(&mut self) -> Result<M> {
        let a = self.additive()?;
        if self.at(":") {
            self.i += 1;
            let b = self.additive()?;
            let (start, mut step, mut stop) = (scalar(&a)?, 1.0, scalar(&b)?);
            if self.at(":") {
                self.i += 1;
                let c = self.additive()?;
                step = stop;
                stop = scalar(&c)?;
            }
            let mut v = Vec::new();
            if step != 0.0 {
                let n = ((stop - start) / step + 1e-10).floor();
                for k in 0..=(n.max(-1.0) as i64) {
                    v.push(start + k as f64 * step);
                }
            }
            return Ok(M { r: 1, c: v.len(), v });
        }
        Ok(a)
    }

    /// In a matrix, `[a -b]` is two elements: a `+`/`-` after a blank and before a non-blank.
    fn matrix_break(&self) -> bool {
        if self.in_matrix == 0 {
            return false;
        }
        let Some(t) = self.peek() else { return false };
        let next_space = self.t.get(self.i + 1).is_none_or(|n| n.space_before);
        t.space_before && !next_space
    }

    fn additive(&mut self) -> Result<M> {
        let mut a = self.multiplicative()?;
        loop {
            let op = if self.at("+") {
                "+"
            } else if self.at("-") {
                "-"
            } else {
                break;
            };
            if self.matrix_break() {
                break;
            }
            self.i += 1;
            let b = self.multiplicative()?;
            a = if op == "+" { broadcast(&a, &b, |x, y| x + y, op)? } else { broadcast(&a, &b, |x, y| x - y, op)? };
        }
        Ok(a)
    }

    fn multiplicative(&mut self) -> Result<M> {
        let mut a = self.unary()?;
        loop {
            let op = ["*", "/", ".*", "./"].into_iter().find(|o| self.at(o));
            let Some(op) = op else { break };
            self.i += 1;
            let b = self.unary()?;
            a = match op {
                "*" => matmul(&a, &b)?,
                ".*" => broadcast(&a, &b, |x, y| x * y, op)?,
                "./" => broadcast(&a, &b, |x, y| x / y, op)?,
                _ => {
                    if !b.is_scalar() {
                        bail!("line {}: matrix right division is not supported", self.line());
                    }
                    broadcast(&a, &b, |x, y| x / y, op)?
                }
            };
        }
        Ok(a)
    }

    fn unary(&mut self) -> Result<M> {
        if self.eat("-") {
            return Ok(self.unary()?.map(|x| -x));
        }
        if self.eat("+") {
            return self.unary();
        }
        self.power()
    }

    fn power(&mut self) -> Result<M> {
        let a = self.postfix()?;
        if self.at("^") || self.at(".^") {
            let elementwise = self.at(".^");
            self.i += 1;
            // MATLAB: 2^-1 is allowed (unary minus binds to the exponent)
            let b = if self.eat("-") { self.postfix()?.map(|x| -x) } else { self.postfix()? };
            if !elementwise && !(a.is_scalar() && b.is_scalar()) {
                bail!("line {}: matrix power is not supported", self.line());
            }
            return broadcast(&a, &b, f64::powf, "^");
        }
        Ok(a)
    }

    fn postfix(&mut self) -> Result<M> {
        let mut a = self.primary()?;
        while self.peek().is_some_and(|t| !t.space_before || self.in_matrix == 0) && (self.at("'") || self.at(".'")) {
            self.i += 1;
            a = a.t();
        }
        Ok(a)
    }

    fn primary(&mut self) -> Result<M> {
        let line = self.line();
        let Some(t) = self.peek().cloned() else { bail!("line {line}: unexpected end of file") };
        self.i += 1;
        match t.t {
            T::Num(x) => Ok(M::scalar(x)),
            T::Op("(") => {
                let saved = self.in_matrix;
                self.in_matrix = 0;
                let v = self.expr();
                self.in_matrix = saved;
                let v = v?;
                self.expect(")")?;
                Ok(v)
            }
            T::Op("[") => self.matrix(),
            T::Id(name) => {
                if name == "end" {
                    return self
                        .end_value
                        .last()
                        .map(|&v| M::scalar(v))
                        .ok_or_else(|| anyhow!("line {line}: 'end' outside an index"));
                }
                let call = self.at("(") && !(self.in_matrix > 0 && self.peek().is_some_and(|t| t.space_before));
                if let Some(v) = self.vars.get(&name).cloned() {
                    if call {
                        self.i += 1;
                        let args = self.index_args(Some(&v))?;
                        return index(&v, &args).with_context(|| format!("line {line}: indexing {name}"));
                    }
                    return Ok(v);
                }
                let args = if call {
                    self.i += 1;
                    let mut args = Vec::new();
                    let saved = self.in_matrix;
                    self.in_matrix = 0;
                    if !self.eat(")") {
                        loop {
                            args.push(self.expr()?);
                            if !self.eat(",") {
                                self.expect(")")?;
                                break;
                            }
                        }
                    }
                    self.in_matrix = saved;
                    args
                } else {
                    Vec::new()
                };
                call_function(&name, &args).with_context(|| format!("line {line}"))
            }
            T::Str(_) => bail!("line {line}: strings are not supported in .m datafiles"),
            other => bail!("line {line}: unexpected {other:?}"),
        }
    }

    /// `[a b; c d]` with rows on separate lines too.
    fn matrix(&mut self) -> Result<M> {
        self.in_matrix += 1;
        let mut rows: Vec<Vec<M>> = vec![Vec::new()];
        loop {
            let Some(t) = self.peek() else { bail!("unterminated [") };
            match t.t {
                T::Op("]") => {
                    self.i += 1;
                    break;
                }
                T::Op(";") | T::Nl => {
                    self.i += 1;
                    if !rows.last().unwrap().is_empty() {
                        rows.push(Vec::new());
                    }
                }
                T::Op(",") => self.i += 1,
                _ => {
                    let e = self.expr()?;
                    rows.last_mut().unwrap().push(e);
                }
            }
        }
        self.in_matrix -= 1;
        rows.retain(|r| !r.is_empty());
        // horizontal then vertical concatenation
        let mut blocks = Vec::new();
        for row in rows {
            let row: Vec<M> = row.into_iter().filter(|m| !m.v.is_empty()).collect();
            if row.is_empty() {
                continue;
            }
            let r = row[0].r;
            if row.iter().any(|m| m.r != r) {
                bail!("line {}: dimensions of arrays being concatenated are not consistent", self.line());
            }
            let mut v = Vec::new();
            let mut c = 0;
            for m in &row {
                v.extend_from_slice(&m.v);
                c += m.c;
            }
            blocks.push(M { r, c, v });
        }
        if blocks.is_empty() {
            return Ok(M { r: 0, c: 0, v: vec![] });
        }
        let c = blocks[0].c;
        if blocks.iter().any(|b| b.c != c) {
            bail!("line {}: dimensions of arrays being concatenated are not consistent", self.line());
        }
        let r: usize = blocks.iter().map(|b| b.r).sum();
        let mut out = M::filled(r, c, 0.0);
        let mut r0 = 0;
        for b in &blocks {
            for i in 0..b.r {
                for j in 0..c {
                    out.v[j * r + r0 + i] = b.get(i, j);
                }
            }
            r0 += b.r;
        }
        Ok(out)
    }
}

fn scalar(m: &M) -> Result<f64> {
    if !m.is_scalar() {
        bail!("expected a scalar");
    }
    Ok(m.v[0])
}

#[derive(Debug)]
enum IndexArg {
    All,
    Idx(M),
}

fn positions(arg: &IndexArg, n: usize) -> Result<Vec<usize>> {
    match arg {
        IndexArg::All => Ok((0..n).collect()),
        IndexArg::Idx(m) => {
            // logical indexing: 0/1 vector of the full length
            m.v.iter()
                .map(|&x| {
                    if x < 1.0 || x.fract() != 0.0 {
                        bail!("index must be a positive integer ({x})");
                    }
                    Ok(x as usize - 1)
                })
                .collect()
        }
    }
}

fn index(m: &M, args: &[IndexArg]) -> Result<M> {
    match args {
        [a] => {
            let p = positions(a, m.v.len())?;
            if let Some(&k) = p.iter().find(|&&k| k >= m.v.len()) {
                bail!("index {} exceeds the number of elements ({})", k + 1, m.v.len());
            }
            let v: Vec<f64> = p.iter().map(|&k| m.v[k]).collect();
            // A(:) is a column; a vector keeps its orientation; a matrix takes the index's
            let col = match a {
                IndexArg::All => true,
                IndexArg::Idx(ix) => {
                    if m.is_vector() && !m.is_scalar() {
                        m.c == 1
                    } else {
                        ix.c == 1 && ix.r > 1
                    }
                }
            };
            Ok(if col { M { r: v.len(), c: 1, v } } else { M { r: 1, c: v.len(), v } })
        }
        [a, b] => {
            let (pi, pj) = (positions(a, m.r)?, positions(b, m.c)?);
            if pi.iter().any(|&i| i >= m.r) || pj.iter().any(|&j| j >= m.c) {
                bail!("index exceeds matrix dimensions");
            }
            let mut out = M::filled(pi.len(), pj.len(), 0.0);
            for (jj, &j) in pj.iter().enumerate() {
                for (ii, &i) in pi.iter().enumerate() {
                    out.v[jj * pi.len() + ii] = m.get(i, j);
                }
            }
            Ok(out)
        }
        _ => bail!("only one or two indices are supported"),
    }
}

fn assign_index(m: &mut M, args: &[IndexArg], v: &M) -> Result<()> {
    match args {
        [a] => {
            let p = positions(a, m.v.len())?;
            let need = p.iter().map(|&k| k + 1).max().unwrap_or(0);
            if need > m.v.len() {
                // grow: a column stays a column, otherwise a row
                if m.c == 1 && m.r > 1 {
                    m.v.resize(need, 0.0);
                    m.r = need;
                } else {
                    m.v.resize(need, 0.0);
                    m.r = 1;
                    m.c = need;
                }
            }
            for (n, &k) in p.iter().enumerate() {
                m.v[k] =
                    if v.is_scalar() { v.v[0] } else { *v.v.get(n).ok_or_else(|| anyhow!("dimension mismatch"))? };
            }
            Ok(())
        }
        [a, b] => {
            let (pi, pj) = (positions(a, m.r)?, positions(b, m.c)?);
            let (nr, nc) = (
                pi.iter().map(|i| i + 1).max().unwrap_or(0).max(m.r),
                pj.iter().map(|j| j + 1).max().unwrap_or(0).max(m.c),
            );
            if (nr, nc) != (m.r, m.c) {
                let mut g = M::filled(nr, nc, 0.0);
                for j in 0..m.c {
                    for i in 0..m.r {
                        g.v[j * nr + i] = m.get(i, j);
                    }
                }
                *m = g;
            }
            let mut n = 0;
            for &j in &pj {
                for &i in &pi {
                    m.v[j * m.r + i] = if v.is_scalar() {
                        v.v[0]
                    } else {
                        *v.v.get(n).ok_or_else(|| anyhow!("dimension mismatch"))?
                    };
                    n += 1;
                }
            }
            Ok(())
        }
        _ => bail!("only one or two indices are supported"),
    }
}

/// Column-wise reduction of a matrix (a vector is reduced as a whole), as MATLAB's `sum`/`mean`.
fn columnwise(m: &M, f: impl Fn(&[f64]) -> f64) -> M {
    if m.is_vector() {
        return M::scalar(f(&m.v));
    }
    M { r: 1, c: m.c, v: (0..m.c).map(|j| f(&m.v[j * m.r..(j + 1) * m.r])).collect() }
}

/// Column-wise transformation (a row vector is treated as one series).
fn along(m: &M, f: impl Fn(&[f64]) -> Vec<f64>) -> M {
    if m.r == 1 {
        let v = f(&m.v);
        return M { r: 1, c: v.len(), v };
    }
    let cols: Vec<Vec<f64>> = (0..m.c).map(|j| f(&m.v[j * m.r..(j + 1) * m.r])).collect();
    let r = cols.first().map_or(0, Vec::len);
    M { r, c: m.c, v: cols.concat() }
}

fn dims(args: &[M]) -> Result<(usize, usize)> {
    Ok(match args {
        [] => (1, 1),
        [n] if n.is_scalar() => (n.v[0] as usize, n.v[0] as usize),
        [n] if n.v.len() == 2 => (n.v[0] as usize, n.v[1] as usize),
        [a, b] => (scalar(a)? as usize, scalar(b)? as usize),
        _ => bail!("bad size arguments"),
    })
}

fn call_function(name: &str, args: &[M]) -> Result<M> {
    let one = || -> Result<&M> { args.first().ok_or_else(|| anyhow!("{name}: missing argument")) };
    Ok(match name {
        "NaN" | "nan" => {
            let (r, c) = dims(args)?;
            M::filled(r, c, f64::NAN)
        }
        "Inf" | "inf" => {
            let (r, c) = dims(args)?;
            M::filled(r, c, f64::INFINITY)
        }
        "pi" => M::scalar(std::f64::consts::PI),
        "zeros" => {
            let (r, c) = dims(args)?;
            M::filled(r, c, 0.0)
        }
        "ones" => {
            let (r, c) = dims(args)?;
            M::filled(r, c, 1.0)
        }
        "log" => one()?.map(f64::ln),
        "log10" => one()?.map(f64::log10),
        "exp" => one()?.map(f64::exp),
        "sqrt" => one()?.map(f64::sqrt),
        "abs" => one()?.map(f64::abs),
        "round" => one()?.map(f64::round),
        "floor" => one()?.map(f64::floor),
        "ceil" => one()?.map(f64::ceil),
        "isnan" => one()?.map(|x| x.is_nan() as u8 as f64),
        "transpose" => one()?.t(),
        "vec" => {
            let m = one()?;
            M { r: m.v.len(), c: 1, v: m.v.clone() }
        }
        "diff" => along(one()?, |s| s.windows(2).map(|w| w[1] - w[0]).collect()),
        "cumsum" => along(one()?, |s| {
            s.iter()
                .scan(0.0, |acc, &x| {
                    *acc += x;
                    Some(*acc)
                })
                .collect()
        }),
        "cumprod" => along(one()?, |s| {
            s.iter()
                .scan(1.0, |acc, &x| {
                    *acc *= x;
                    Some(*acc)
                })
                .collect()
        }),
        "sum" => columnwise(one()?, |s| s.iter().sum()),
        "mean" => columnwise(one()?, |s| s.iter().sum::<f64>() / s.len() as f64),
        "length" => {
            let m = one()?;
            M::scalar(if m.v.is_empty() { 0.0 } else { m.r.max(m.c) as f64 })
        }
        "numel" => M::scalar(one()?.v.len() as f64),
        "size" => match args {
            [m] => M { r: 1, c: 2, v: vec![m.r as f64, m.c as f64] },
            [m, d] => M::scalar(if scalar(d)? == 1.0 { m.r as f64 } else { m.c as f64 }),
            _ => bail!("size: bad arguments"),
        },
        "min" | "max" => match args {
            [a, b] => broadcast(a, b, if name == "min" { f64::min } else { f64::max }, name)?,
            [a] => columnwise(a, |s| {
                s.iter().copied().fold(
                    if name == "min" { f64::INFINITY } else { f64::NEG_INFINITY },
                    if name == "min" { f64::min } else { f64::max },
                )
            }),
            _ => bail!("{name}: bad arguments"),
        },
        "repmat" => match args {
            [m, a, b] => {
                let (ra, cb) = (scalar(a)? as usize, scalar(b)? as usize);
                let (r, c) = (m.r * ra, m.c * cb);
                let mut out = M::filled(r, c, 0.0);
                for j in 0..c {
                    for i in 0..r {
                        out.v[j * r + i] = m.get(i % m.r, j % m.c);
                    }
                }
                out
            }
            _ => bail!("repmat: expected repmat(A, m, n)"),
        },
        "reshape" => match args {
            [m, a, b] => {
                let (r, c) = (scalar(a)? as usize, scalar(b)? as usize);
                if r * c != m.v.len() {
                    bail!("reshape: To RESHAPE the number of elements must not change.");
                }
                M { r, c, v: m.v.clone() }
            }
            _ => bail!("reshape: expected reshape(A, m, n)"),
        },
        _ => bail!("unknown variable or function '{name}'"),
    })
}

/// Runs a `.m` datafile; returns its variables.
pub fn run_script(src: &str) -> Result<HashMap<String, M>> {
    let toks = lex(src)?;
    let mut it = Interp { t: &toks, i: 0, vars: HashMap::new(), in_matrix: 0, end_value: Vec::new() };
    it.run()?;
    Ok(it.vars)
}

/// `load_m_file_data_legacy.m`: every vector variable of the script becomes a series
/// (`vec()` of it), in name order; shorter series are padded with NaN.
pub fn read_m_file(path: &Path) -> Result<crate::data::Table> {
    let src = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let vars = run_script(&src).with_context(|| format!("{}", path.display()))?;
    let mut names: Vec<&String> = vars.keys().filter(|k| vars[*k].is_vector() && !vars[*k].v.is_empty()).collect();
    names.sort();
    if names.is_empty() {
        bail!("{}: no vectors defined in the file", path.display());
    }
    let n = names.iter().map(|k| vars[*k].v.len()).max().unwrap();
    let values =
        crate::linalg::Mat::from_fn(n, names.len(), |i, j| vars[names[j]].v.get(i).copied().unwrap_or(f64::NAN));
    Ok(crate::data::Table { names: names.into_iter().cloned().collect(), dates: None, values })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrices_and_arithmetic() {
        let v = run_script(
            "% comment\nA = [1 2 3\n4 5 6];\nx = A(:,2)';\ny = [1 -2 3]; z = [1 - 2, 3];\nw = diff(log([1;2;4]));\nq = 2^-1 + 1e-1;\nr = 1:3; r(5) = 7;\ns = A(end, end-1:end); t = -2^2; u = [1,2]' .* [3;4];\nv = 1./[2 4]; c = [1 2 ...\n 3];\n",
        )
        .unwrap();
        assert_eq!(v["A"], M { r: 2, c: 3, v: vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0] });
        assert_eq!(v["x"], M { r: 1, c: 2, v: vec![2.0, 5.0] });
        assert_eq!(v["y"].v, vec![1.0, -2.0, 3.0]);
        assert_eq!(v["z"].v, vec![-1.0, 3.0]);
        assert!((v["w"].v[1] - 2f64.ln()).abs() < 1e-15 && v["w"].r == 2);
        assert_eq!(v["q"].v, vec![0.6]);
        assert_eq!(v["r"].v, vec![1.0, 2.0, 3.0, 0.0, 7.0]);
        assert_eq!(v["s"].v, vec![5.0, 6.0]);
        assert_eq!(v["t"].v, vec![-4.0]);
        assert_eq!(v["u"].v, vec![3.0, 8.0]);
        assert_eq!(v["v"].v, vec![0.5, 0.25]);
        assert_eq!(v["c"].v, vec![1.0, 2.0, 3.0]);
        let v = run_script("d = [1 2 3 4 5 6 ...\n7 8 9 10];\nd = reshape(d,5,2)';\ny = d(:,2);\n").unwrap();
        assert_eq!(v["d"], M { r: 2, c: 5, v: vec![1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 5.0, 10.0] });
        assert_eq!(v["y"].v, vec![2.0, 7.0]);
    }
}
