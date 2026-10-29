// SPDX-License-Identifier: GPL-3.0-or-later
//! `shocks` / `mshocks` / `shock_paths` syntax shared by the blocks (stream shocks/shock_paths):
//! `periods` lists of integers, integer ranges, dates and date ranges (`2023Q1`,
//! `2023Q2:2023Q4`), `values` items that are MATLAB workspace vectors (`values (xx);`),
//! `learnt_in = INTEGER | DATE`, the `database` command, the MATLAB workspace assignments
//! that feed them (`xx = [1.2; 1.3; 1];`, `db = dseries(...)`, `tb = table(...)`), the
//! `heterogeneity=` option (heterogeneous-agent models, not ported) and the per-block
//! evaluation of `corr` entries (the preprocessor writes `M_.Sigma_e(i,j) = r *
//! sqrt(M_.Sigma_e(i,i) * M_.Sigma_e(j,j))` at the end of each block, with the variances known
//! at that point).

use std::path::Path;

use anyhow::{Result, anyhow, bail};

use super::{Parser, PathValue, ShockStmt, Tgt, Tok, Token, split_commas};
use crate::dates::Date;
use crate::deterministic::workspace::WsValue;
use crate::linalg::Mat;
use crate::model::SymbolKind;
use crate::modfile::{OptValue, Options, opt};

/// One entry of a `periods` statement: integer periods or dates.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum PGroup {
    Ints(Vec<i64>),
    Dates(Vec<Date>),
}

impl PGroup {
    fn len(&self) -> usize {
        match self {
            PGroup::Ints(v) => v.len(),
            PGroup::Dates(v) => v.len(),
        }
    }
    /// (integer periods, dates) as stored in `DetShock` / `LearntShock` / `ControlledPath`.
    pub(super) fn split(&self) -> (Vec<i64>, Vec<Date>) {
        match self {
            PGroup::Ints(v) => (v.clone(), Vec::new()),
            PGroup::Dates(v) => (Vec::new(), v.clone()),
        }
    }
    fn nth(&self, k: usize) -> PGroup {
        match self {
            PGroup::Ints(v) => PGroup::Ints(vec![v[k]]),
            PGroup::Dates(v) => PGroup::Dates(vec![v[k]]),
        }
    }
}

/// A date literal at `toks[i..]` (`2023Q1`, `-5Q2`, `2004Y`, `2004M12`): the lexer splits it
/// into a number and an identifier. Returns the date and the number of tokens used.
pub(super) fn date_at(toks: &[Token], i: usize) -> Option<(Date, usize)> {
    let (neg, k) = if toks.get(i).is_some_and(|t| t.is("-")) { (true, i + 1) } else { (false, i) };
    let Tok::Num(y) = toks.get(k)?.tok else { return None };
    let suffix = toks.get(k + 1)?.ident()?;
    if y.fract() != 0.0 || !suffix.starts_with(|c: char| "YyAaSsHhQqMmWw".contains(c)) {
        return None;
    }
    let s = format!("{}{}{}", if neg { "-" } else { "" }, y as i64, suffix);
    Date::parse(&s).map(|d| (d, k + 2 - i))
}

/// A bound of a period entry: an integer or a date (with the number of tokens used).
pub(super) fn point_at(toks: &[Token], i: usize) -> Result<(Result<i64, Date>, usize)> {
    if let Some((d, n)) = date_at(toks, i) {
        return Ok((Err(d), n));
    }
    match toks.get(i).map(|t| &t.tok) {
        Some(Tok::Num(v)) if v.fract() == 0.0 => Ok((Ok(*v as i64), 1)),
        _ => bail!(
            "expected an integer period or a date{}",
            toks.get(i).map(|t| format!(", found '{}'", t.raw)).unwrap_or_default()
        ),
    }
}

/// `periods 1:3 5, 2023Q1 2023Q2:2023Q4;` -> one group per entry (as `M_.det_shocks`).
pub(super) fn sim_period_list(toks: &[Token]) -> Result<Vec<PGroup>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if toks[i].is(",") {
            i += 1;
            continue;
        }
        let (a, n) = point_at(toks, i)?;
        i += n;
        let b = if toks.get(i).is_some_and(|t| t.is(":")) {
            let (b, n) = point_at(toks, i + 1)?;
            i += 1 + n;
            if toks.get(i).is_some_and(|t| t.is(":")) {
                // the 7.2 grammar has `first:last` only (`1:2:9` is a syntax error there)
                bail!(
                    "periods: a range is `first:last`; a stepped range such as `1:2:9` is not valid Dynare syntax (list the periods instead, e.g. `1 3 5 7 9`)"
                );
            }
            b
        } else {
            a
        };
        out.push(match (a, b) {
            (Ok(a), Ok(b)) => PGroup::Ints((a..=b).collect()),
            (Err(a), Err(b)) => PGroup::Dates(Date::range(a, b)?),
            _ => bail!("a period range cannot mix an integer and a date"),
        });
    }
    Ok(out)
}

/// A `learnt_in = INTEGER | DATE` option: `(integer, None)` or `(0, Some(date))`; `(1, None)`
/// when absent.
pub(super) fn learnt_in_option(opts: &Options, what: &str) -> Result<(usize, Option<Date>)> {
    match opt(opts, "learnt_in") {
        None => Ok((1, None)),
        Some(OptValue::Num(v)) if *v >= 1.0 && v.fract() == 0.0 => Ok((*v as usize, None)),
        Some(OptValue::Str(s)) => match Date::parse_mod(s) {
            Some(d) => Ok((0, Some(d))),
            None => bail!("{what}: learnt_in = {s}: expected a positive integer or a date"),
        },
        Some(_) => bail!("{what}: learnt_in must be a positive integer or a date"),
    }
}

/// The label of a `learnt_in` option in messages.
pub(super) fn learnt_in_label(li: (usize, Option<Date>)) -> String {
    match li.1 {
        Some(d) => d.label(),
        None => li.0.to_string(),
    }
}

/// `shock_paths(overwrite learnt_in = 2000Q4)`, `mshocks(learnt_in = 2 relative_to_initval)`:
/// the 7.2 grammar separates the options of `shock_paths` and `mshocks` by spaces (a comma is a
/// syntax error there); dynare-rust accepts both.
pub(super) fn split_space_separated_options(inner: &[Token]) -> Vec<Token> {
    const KEYS: [&str; 5] = ["overwrite", "learnt_in", "relative_to_initval", "surprise", "heterogeneity"];
    let mut out: Vec<Token> = Vec::with_capacity(inner.len() + 2);
    let mut k = 0;
    while k < inner.len() {
        let t = &inner[k];
        let starts_option = KEYS.iter().any(|kw| t.is_ident(kw)) && k > 0 && !inner[k - 1].is(",");
        if starts_option {
            out.push(Token { tok: Tok::Punct(","), raw: ",".into(), loc: t.loc.clone(), col: t.col });
        }
        out.push(t.clone());
        k += 1;
    }
    out
}

impl<'a> Parser<'a> {
    /// The `( ... )` options of a `shocks` / `mshocks` / `shock_paths` block, separated by
    /// commas or spaces (see [`split_space_separated_options`]).
    pub(super) fn shocks_header_options(&mut self) -> Result<Options> {
        let toks = self.c.statement()?;
        if toks.is_empty() {
            return Ok(Vec::new());
        }
        if !(toks[0].is("(") && toks.last().is_some_and(|t| t.is(")"))) {
            bail!("{}: unexpected '{}'", toks[0].loc, toks[0].raw);
        }
        let inner = split_space_separated_options(&toks[1..toks.len() - 1]);
        Ok(self.parse_options(&inner))
    }

    /// The values of a `values`/`add`/`multiply` statement: one vector per item (a scalar, or
    /// the elements of a workspace vector `(xx)`).
    pub(super) fn shocks_values(&self, toks: &[Token]) -> Result<Vec<Vec<f64>>> {
        let mut out = Vec::new();
        for item in super::split_value_items(toks) {
            if let [l, name, r] = item
                && l.is("(")
                && r.is(")")
                && let Some(n) = name.ident()
                && self.b.symbol(n).is_none()
                && let Some(v) = self.q1.workspace.vector(n)
            {
                out.push(v);
                continue;
            }
            out.push(vec![self.eval_tokens(item)?]);
        }
        Ok(out)
    }

    /// [`Self::shocks_values`] where an item may be an expression with endogenous variables.
    pub(super) fn shocks_values_x(&self, toks: &[Token]) -> Result<Vec<Vec<PathValue>>> {
        let mut out = Vec::new();
        for item in super::split_value_items(toks) {
            if let Some(e) = self.steady_expr(item)? {
                out.push(vec![PathValue::Steady(e)]);
            } else {
                out.extend(self.shocks_values(item)?.into_iter().map(|v| v.into_iter().map(PathValue::Num).collect()));
            }
        }
        Ok(out)
    }

    /// Pairs the `periods` groups with the values (a vector value gives one entry per period
    /// of its range, as the MATLAB assignment `oo_.exo_simul(k, ivar) = v`).
    pub(super) fn shocks_pair<T: Clone>(
        &self,
        periods: Vec<PGroup>,
        values: Vec<Vec<T>>,
        what: &str,
    ) -> Result<Vec<(PGroup, T)>> {
        if periods.len() != values.len() {
            bail!("{what}: {} period ranges but {} values", periods.len(), values.len());
        }
        let mut out = Vec::new();
        for (g, v) in periods.into_iter().zip(values) {
            if v.len() == 1 {
                out.push((g, v[0].clone()));
            } else if v.len() == g.len() {
                for (k, x) in v.into_iter().enumerate() {
                    out.push((g.nth(k), x));
                }
            } else {
                bail!("{what}: a vector of {} values for a range of {} periods", v.len(), g.len());
            }
        }
        Ok(out)
    }

    /// Start of a `shocks` block: `shocks(overwrite)` resets what the preprocessor resets
    /// (`M_.det_shocks`, `M_.Sigma_e`, `M_.Correlation_matrix`, `M_.H`,
    /// `M_.Correlation_matrix_ME`; not `M_.Skew_e`).
    pub(super) fn shocks_block_start(&mut self, overwrite: bool) {
        if overwrite {
            self.det_shocks.clear();
            self.shocks.clear();
            // M_.Correlation_matrix = eye, M_.Correlation_matrix_ME = eye
            self.q1.shock_corrs.clear();
        }
        self.q1.shocks_block_start = self.shocks.len();
    }

    /// End of a `shocks` block: its `corr` entries become covariances computed with the
    /// variances set so far (the generated `M_.Sigma_e(i,j) = r*sqrt(...)` lines).
    pub(super) fn shocks_block_end(&mut self) {
        let st = self.q1.shocks_block_start.min(self.shocks.len());
        let var_of = |stmts: &[ShockStmt], t: Tgt| -> f64 {
            let mut v = 0.0;
            for s in stmts {
                match *s {
                    ShockStmt::Var(a, ref x) if a == t => v = x.num(),
                    ShockStmt::Std(a, ref x) if a == t => v = x.num() * x.num(),
                    _ => {}
                }
            }
            v
        };
        for k in st..self.shocks.len() {
            if let ShockStmt::Corr(a, b, r) = self.shocks[k].clone() {
                let r = r.num();
                self.q1.shock_corrs.push((a, b, r));
                let c = r * (var_of(&self.shocks, a) * var_of(&self.shocks, b)).sqrt();
                self.shocks[k] = ShockStmt::Cov(a, b, PathValue::Num(c));
            }
        }
        self.q1.shocks_block_start = self.shocks.len();
    }

    /// `database db1, db2;` (`M_.database`).
    pub(super) fn shocks_database(&mut self) -> Result<()> {
        self.c.i += 1;
        let toks = self.c.statement()?;
        for item in split_commas(toks) {
            for t in item {
                match t.ident() {
                    Some(n) => {
                        if !self.q1.databases.iter().any(|d| d == n) {
                            self.q1.databases.push(n.to_string());
                        }
                    }
                    None => bail!("{}: database: expected a name, found '{}'", t.loc, t.raw),
                }
            }
        }
        Ok(())
    }

    /// `NAME = <MATLAB expression>;` for a name that is not a Dynare symbol and whose value
    /// is not a constant expression: evaluated into the workspace when dynare-rust
    /// understands it (see `deterministic::workspace`), otherwise skipped with a warning.
    /// Value of a MATLAB workspace expression (`dseries('file.csv')`, a workspace variable, ...).
    pub(super) fn ws_value(&self, toks: &[Token]) -> Result<WsValue> {
        let dir = toks
            .first()
            .map(|t| Path::new(&*t.loc.file).parent().map(Path::to_path_buf).unwrap_or_default())
            .unwrap_or_default();
        WsEval { p: self, dir: &dir }.expr_all(toks)
    }

    pub(super) fn shocks_workspace_assignment(&mut self, name: &str, toks: &[Token]) -> Result<()> {
        let dir = toks
            .first()
            .map(|t| Path::new(&*t.loc.file).parent().map(Path::to_path_buf).unwrap_or_default())
            .unwrap_or_default();
        let ev = WsEval { p: self, dir: &dir };
        match ev.expr_all(toks) {
            Ok(v) => {
                if let WsValue::Num(m) = &v
                    && m.len() == 1
                {
                    self.mf_locals.insert(name.to_string(), m[0]);
                }
                self.q1.workspace.vars.insert(name.to_string(), v);
            }
            // reporting: dseries of the stoch_simul IRFs, report objects (run by the driver)
            Err(e) if super::rep::is_report_rhs(toks) => {
                // a statement that is neither a workspace value nor a report one is skipped
                let mark = self.commands.len();
                if self.rep_push(Some(name), toks).is_err() {
                    self.commands.truncate(mark);
                    Self::ws_skip_warning(name, toks, &e);
                }
            }
            Err(e) => Self::ws_skip_warning(name, toks, &e),
        }
        Ok(())
    }

    /// The warning of a skipped `name = <MATLAB expression>;`.
    fn ws_skip_warning(name: &str, toks: &[Token], e: &anyhow::Error) {
        let text: Vec<&str> = toks.iter().map(|t| t.raw.as_str()).collect();
        let loc = toks.first().map(|t| t.loc.to_string()).unwrap_or_default();
        println!(
            "WARNING: {loc}: MATLAB statement `{name} = {};` is not supported by dynare-rust ({e}); skipped.",
            text.join(" ")
        );
    }
}

impl Parser<'_> {
    /// `x(i, j) = v;` / `x(k) = v;` on a numeric workspace array (`init(:,[2,4]) = ones(10,2);`
    /// before a `dseries(init, ...)`): `:`, scalar or vector indices, a scalar or an array of
    /// the indexed shape. Whether the statement is one: the name is a numeric workspace array
    /// followed by `(...) =`.
    pub(super) fn ws_indexed_assignment_ahead(&self, name: &str) -> bool {
        if !matches!(self.q1.workspace.vars.get(name), Some(WsValue::Num(_)))
            || !self.c.peek_at(1).is_some_and(|t| t.is("("))
        {
            return false;
        }
        let mut depth = 0;
        let mut k = 1;
        while let Some(t) = self.c.peek_at(k) {
            if t.is("(") {
                depth += 1;
            } else if t.is(")") {
                depth -= 1;
                if depth == 0 {
                    return self.c.peek_at(k + 1).is_some_and(|t| t.is("="));
                }
            } else if t.is(";") {
                return false;
            }
            k += 1;
        }
        false
    }

    pub(super) fn ws_indexed_assignment(&mut self) -> Result<()> {
        let (name, loc) = self.c.ident()?;
        let (name, loc) = (name.to_string(), loc.clone());
        let toks = self.c.statement()?;
        let mut depth = 0;
        let close = toks
            .iter()
            .position(|t| {
                depth += t.is("(") as i32 - t.is(")") as i32;
                depth == 0
            })
            .ok_or_else(|| anyhow!("{loc}: unbalanced parentheses"))?;
        let dir = Path::new(&*loc.file).parent().map(Path::to_path_buf).unwrap_or_default();
        let ev = WsEval { p: self, dir: &dir };
        let rhs = ev.expr_all(&toks[close + 2..]).map_err(|e| anyhow!("{loc}: {name}(...) = ...: {e}"))?;
        let rhs = as_mat(&rhs)?.clone();
        let Some(WsValue::Num(m)) = self.q1.workspace.vars.get(&name) else { unreachable!() };
        let (nr, nc) = (m.nrows(), m.ncols());
        let args = super::split_commas(&toks[1..close]);
        // 0-based indices of one subscript (`:` is all of `n`)
        let idx = |a: &[Token], n: usize| -> Result<Vec<usize>> {
            if matches!(a, [t] if t.is(":")) {
                return Ok((0..n).collect());
            }
            let ev = WsEval { p: self, dir: &dir };
            let v = ev.expr_all(a)?;
            as_mat(&v)?
                .iter()
                .map(|&x| {
                    if x >= 1.0 && x.fract() == 0.0 && (x as usize) <= n {
                        Ok(x as usize - 1)
                    } else {
                        bail!("index {x} out of bounds (1..{n})")
                    }
                })
                .collect()
        };
        let cells: Vec<(usize, usize)> = match args.as_slice() {
            [i, j] => {
                let (ri, ci) = (idx(i, nr)?, idx(j, nc)?);
                ci.iter().flat_map(|&c| ri.iter().map(move |&r| (r, c))).collect()
            }
            [k] => idx(k, nr * nc)?.into_iter().map(|l| (l % nr, l / nr)).collect(),
            _ => bail!("{loc}: {name}(...) = ...: one or two subscripts expected"),
        };
        if rhs.len() != 1 && rhs.len() != cells.len() {
            bail!("{loc}: {name}(...) = ...: the dimensions do not agree");
        }
        let Some(WsValue::Num(m)) = self.q1.workspace.vars.get_mut(&name) else { unreachable!() };
        for (k, &(r, c)) in cells.iter().enumerate() {
            m[(r, c)] = if rhs.len() == 1 { rhs[0] } else { rhs[k] };
        }
        Ok(())
    }
}

/// Evaluator of the MATLAB expressions of workspace assignments.
struct WsEval<'p, 'a> {
    p: &'p Parser<'a>,
    dir: &'p Path,
}

fn num(m: Mat) -> WsValue {
    WsValue::Num(m)
}

fn scalar(v: f64) -> WsValue {
    WsValue::Num(Mat::from_element(1, 1, v))
}

fn as_mat(v: &WsValue) -> Result<&Mat> {
    match v {
        WsValue::Num(m) => Ok(m),
        _ => bail!("expected a numeric value"),
    }
}

fn as_scalar(v: &WsValue) -> Result<f64> {
    let m = as_mat(v)?;
    if m.len() != 1 {
        bail!("expected a scalar");
    }
    Ok(m[0])
}

fn as_strings(v: &WsValue) -> Result<Vec<String>> {
    match v {
        WsValue::Str(s) => Ok(vec![s.clone()]),
        WsValue::Cell(c) => Ok(c.clone()),
        _ => bail!("expected a string or a cell array of strings"),
    }
}

fn as_date(v: &WsValue) -> Result<Date> {
    match v {
        WsValue::Str(s) => Date::parse(s).ok_or_else(|| anyhow!("'{s}' is not a date")),
        _ => bail!("expected a date"),
    }
}

/// Elementwise binary operation with scalar broadcasting.
fn elementwise(a: &Mat, b: &Mat, f: impl Fn(f64, f64) -> f64) -> Result<Mat> {
    if a.len() == 1 {
        return Ok(b.map(|y| f(a[0], y)));
    }
    if b.len() == 1 {
        return Ok(a.map(|x| f(x, b[0])));
    }
    if a.shape() != b.shape() {
        bail!("matrix dimensions must agree ({}x{} and {}x{})", a.nrows(), a.ncols(), b.nrows(), b.ncols());
    }
    Ok(a.zip_map(b, f))
}

/// `m(k)` (linear) or `m(rows, cols)` with 1-based indices.
fn index_mat(m: &Mat, args: &[WsValue]) -> Result<Mat> {
    let ix = |v: &WsValue, n: usize| -> Result<Vec<usize>> {
        as_mat(v)?
            .iter()
            .map(|&k| {
                if k >= 1.0 && k as usize <= n && k.fract() == 0.0 {
                    Ok(k as usize - 1)
                } else {
                    Err(anyhow!("index {k} out of bounds"))
                }
            })
            .collect()
    };
    match args {
        [k] => {
            let k = ix(k, m.len())?;
            let v: Vec<f64> = k.iter().map(|&j| m[j]).collect();
            Ok(if m.ncols() == 1 {
                Mat::from_column_slice(v.len(), 1, &v)
            } else {
                Mat::from_row_slice(1, v.len(), &v)
            })
        }
        [r, c] => {
            let (r, c) = (ix(r, m.nrows())?, ix(c, m.ncols())?);
            Ok(Mat::from_fn(r.len(), c.len(), |a, b| m[(r[a], c[b])]))
        }
        _ => bail!("only one or two indices are supported"),
    }
}

fn hcat(parts: &[Mat]) -> Result<Mat> {
    let parts: Vec<&Mat> = parts.iter().filter(|m| !m.is_empty()).collect();
    let Some(r) = parts.first().map(|m| m.nrows()) else { return Ok(Mat::zeros(0, 0)) };
    if parts.iter().any(|m| m.nrows() != r) {
        bail!("dimensions of arrays being concatenated are not consistent");
    }
    let c: usize = parts.iter().map(|m| m.ncols()).sum();
    let mut out = Mat::zeros(r, c);
    let mut j = 0;
    for m in parts {
        out.columns_mut(j, m.ncols()).copy_from(m);
        j += m.ncols();
    }
    Ok(out)
}

fn vcat(parts: &[Mat]) -> Result<Mat> {
    let t: Vec<Mat> = parts.iter().map(|m| m.transpose()).collect();
    Ok(hcat(&t)?.transpose())
}

impl WsEval<'_, '_> {
    fn expr_all(&self, toks: &[Token]) -> Result<WsValue> {
        let mut i = 0;
        let v = self.expr(toks, &mut i)?;
        if i != toks.len() {
            bail!("unexpected '{}'", toks[i].raw);
        }
        Ok(v)
    }

    /// `a:b` / `a:s:b` (lowest precedence), then additive expressions.
    fn expr(&self, t: &[Token], i: &mut usize) -> Result<WsValue> {
        let a = self.additive(t, i)?;
        if !t.get(*i).is_some_and(|x| x.is(":")) {
            return Ok(a);
        }
        *i += 1;
        let b = self.additive(t, i)?;
        let (lo, step, hi) = if t.get(*i).is_some_and(|x| x.is(":")) {
            *i += 1;
            let c = self.additive(t, i)?;
            (as_scalar(&a)?, as_scalar(&b)?, as_scalar(&c)?)
        } else {
            (as_scalar(&a)?, 1.0, as_scalar(&b)?)
        };
        if step == 0.0 {
            return Ok(num(Mat::zeros(1, 0)));
        }
        let n = ((hi - lo) / step + 1e-10).floor();
        let n = if n < 0.0 { 0 } else { n as usize + 1 };
        Ok(num(Mat::from_fn(1, n, |_, k| lo + k as f64 * step)))
    }

    fn additive(&self, t: &[Token], i: &mut usize) -> Result<WsValue> {
        let mut a = self.term(t, i)?;
        while let Some(op) = t.get(*i).filter(|x| x.is("+") || x.is("-")) {
            *i += 1;
            let b = self.term(t, i)?;
            let plus = op.is("+");
            a = num(elementwise(as_mat(&a)?, as_mat(&b)?, |x, y| if plus { x + y } else { x - y })?);
        }
        Ok(a)
    }

    fn term(&self, t: &[Token], i: &mut usize) -> Result<WsValue> {
        let mut a = self.unary(t, i)?;
        loop {
            let mut dot = t.get(*i).is_some_and(|x| x.is(".")) && t.get(*i + 1).is_some_and(|x| x.is("*") || x.is("/"));
            let k = *i + dot as usize;
            let Some(op) = t.get(k).filter(|x| x.is("*") || x.is("/")) else { break };
            // `1./x`: the lexer reads the number `1.` followed by `/`
            dot |= !dot && k > 0 && matches!(t[k - 1].tok, Tok::Num(_)) && t[k - 1].raw.ends_with('.');
            *i = k + 1;
            let b = self.unary(t, i)?;
            let (ma, mb) = (as_mat(&a)?, as_mat(&b)?);
            a = num(if op.is("*") && !dot && ma.len() != 1 && mb.len() != 1 {
                if ma.ncols() != mb.nrows() {
                    bail!("inner matrix dimensions must agree");
                }
                ma * mb
            } else if op.is("*") {
                elementwise(ma, mb, |x, y| x * y)?
            } else {
                if !dot && mb.len() != 1 {
                    bail!("matrix right division is not supported");
                }
                elementwise(ma, mb, |x, y| x / y)?
            });
        }
        Ok(a)
    }

    fn unary(&self, t: &[Token], i: &mut usize) -> Result<WsValue> {
        if t.get(*i).is_some_and(|x| x.is("-")) {
            *i += 1;
            let a = self.unary(t, i)?;
            return Ok(num(as_mat(&a)?.map(|x| -x)));
        }
        if t.get(*i).is_some_and(|x| x.is("+")) {
            *i += 1;
            return self.unary(t, i);
        }
        self.power(t, i)
    }

    fn power(&self, t: &[Token], i: &mut usize) -> Result<WsValue> {
        let a = self.primary(t, i)?;
        let dot = t.get(*i).is_some_and(|x| x.is(".")) && t.get(*i + 1).is_some_and(|x| x.is("^"));
        let k = *i + dot as usize;
        if t.get(k).is_some_and(|x| x.is("^")) {
            *i = k + 1;
            let b = self.unary(t, i)?;
            let (ma, mb) = (as_mat(&a)?, as_mat(&b)?);
            if !dot && (ma.len() != 1 || mb.len() != 1) {
                bail!("matrix power is not supported (use .^)");
            }
            return Ok(num(elementwise(ma, mb, f64::powf)?));
        }
        Ok(a)
    }

    /// Comma-separated arguments up to the matching `)` (the cursor is after `(`).
    fn args(&self, t: &[Token], i: &mut usize) -> Result<Vec<WsValue>> {
        let mut out = Vec::new();
        if t.get(*i).is_some_and(|x| x.is(")")) {
            *i += 1;
            return Ok(out);
        }
        loop {
            out.push(self.expr(t, i)?);
            match t.get(*i) {
                Some(x) if x.is(",") => *i += 1,
                Some(x) if x.is(")") => {
                    *i += 1;
                    return Ok(out);
                }
                _ => bail!("expected ',' or ')'"),
            }
        }
    }

    fn primary(&self, t: &[Token], i: &mut usize) -> Result<WsValue> {
        if let Some((d, n)) = date_at(t, *i).filter(|_| !t[*i].is("-")) {
            *i += n;
            return Ok(WsValue::Str(d.label()));
        }
        let Some(tok) = t.get(*i) else { bail!("unexpected end of expression") };
        *i += 1;
        match &tok.tok {
            Tok::Num(v) => Ok(scalar(*v)),
            Tok::Str(s) => Ok(WsValue::Str(s.clone())),
            Tok::Punct("(") => {
                let v = self.expr(t, i)?;
                if !t.get(*i).is_some_and(|x| x.is(")")) {
                    bail!("expected ')'");
                }
                *i += 1;
                Ok(v)
            }
            Tok::Punct("[") => self.matrix(t, i),
            Tok::Punct("{") => {
                let mut c = Vec::new();
                while let Some(x) = t.get(*i) {
                    *i += 1;
                    match &x.tok {
                        Tok::Punct("}") => return Ok(WsValue::Cell(c)),
                        Tok::Punct(",") | Tok::Punct(";") => {}
                        Tok::Str(s) => c.push(s.clone()),
                        _ => bail!("only cell arrays of strings are supported"),
                    }
                }
                bail!("unterminated cell array")
            }
            // `M_.endo_nbr`, `M_.endo_names`, ...: the declared symbols (the auxiliary variables
            // the preprocessor adds later are not known yet)
            Tok::Ident(name)
                if name == "M_"
                    && t.get(*i).is_some_and(|x| x.is("."))
                    && t.get(*i + 1).and_then(|x| x.ident()).is_some() =>
            {
                let field = t[*i + 1].ident().unwrap();
                *i += 2;
                let b = &self.p.b;
                Ok(match field {
                    "endo_nbr" => scalar(b.endo_names.len() as f64),
                    "exo_nbr" => scalar(b.exo_names.len() as f64),
                    "param_nbr" => scalar(b.param_names.len() as f64),
                    "endo_names" => WsValue::Cell(b.endo_names.clone()),
                    "exo_names" => WsValue::Cell(b.exo_names.clone()),
                    "param_names" => WsValue::Cell(b.param_names.clone()),
                    _ => bail!("M_.{field} is not supported"),
                })
            }
            Tok::Ident(name)
                if t.get(*i).is_some_and(|x| x.is("."))
                    && t.get(*i + 1).and_then(|x| x.ident()).is_some()
                    && self.p.q1.workspace.vars.contains_key(name.as_str()) =>
            {
                // a field of a workspace struct: `s.data`, `s.data(2:80, 1)` (importdata)
                let field = t[*i + 1].ident().unwrap();
                *i += 2;
                let v = match (self.p.q1.workspace.vars.get(name.as_str()).unwrap(), field) {
                    (WsValue::Table { data, .. }, "data") => num(data.clone()),
                    (WsValue::Table { names, .. }, "colheaders") => WsValue::Cell(names.clone()),
                    _ => bail!("field '{field}' of '{name}' is not supported"),
                };
                if t.get(*i).is_some_and(|x| x.is("(")) {
                    *i += 1;
                    let args = self.args(t, i)?;
                    return index_mat(as_mat(&v)?, &args).map(num);
                }
                Ok(v)
            }
            Tok::Ident(name) => {
                let call = t.get(*i).is_some_and(|x| x.is("("));
                let first_arg_name = if call {
                    t.get(*i + 1)
                        .and_then(|x| x.ident())
                        .filter(|_| t.get(*i + 2).is_some_and(|y| y.is(",") || y.is(")")))
                } else {
                    None
                };
                let args = if call {
                    *i += 1;
                    Some(self.args(t, i)?)
                } else {
                    None
                };
                self.call(name, args, first_arg_name, t, *i)
            }
            _ => bail!("unexpected '{}'", tok.raw),
        }
    }

    /// `[a b; c d]` (the cursor is after `[`): rows separated by `;`, elements by `,` or by
    /// juxtaposition (`[1 -2]` is read as `[1-2]`: the lexer drops the spaces).
    fn matrix(&self, t: &[Token], i: &mut usize) -> Result<WsValue> {
        let mut rows: Vec<Mat> = Vec::new();
        let mut row: Vec<Mat> = Vec::new();
        loop {
            match t.get(*i) {
                None => bail!("unterminated matrix"),
                Some(x) if x.is("]") => {
                    *i += 1;
                    rows.push(hcat(&row)?);
                    return Ok(num(vcat(&rows)?));
                }
                Some(x) if x.is(";") => {
                    *i += 1;
                    rows.push(hcat(&std::mem::take(&mut row))?);
                }
                Some(x) if x.is(",") => *i += 1,
                _ => {
                    // MATLAB: in brackets `[0.01 -0.02]` is two elements (a sign after a blank
                    // and before its operand starts a new element), `[0.01 - 0.02]` one
                    let mut depth = 0i32;
                    let mut end = *i;
                    while let Some(x) = t.get(end) {
                        if x.is("(") || x.is("[") || x.is("{") {
                            depth += 1;
                        } else if x.is(")") || x.is("]") || x.is("}") {
                            if depth == 0 {
                                break;
                            }
                            depth -= 1;
                        } else if depth == 0 && (x.is(",") || x.is(";")) {
                            break;
                        } else if depth == 0 && end > *i && (x.is("-") || x.is("+")) {
                            let (p, n) = (&t[end - 1], t.get(end + 1));
                            let after_blank = p.loc == x.loc && x.col.0 > p.col.1 + 1;
                            let glued = n.is_some_and(|n| n.loc == x.loc && n.col.0 == x.col.1 + 1);
                            let p_operand = !matches!(&p.tok, Tok::Punct(o) if !matches!(*o, ")" | "]" | "}"));
                            if after_blank && glued && p_operand {
                                break;
                            }
                        }
                        end += 1;
                    }
                    let v = self.expr(&t[..end], i)?;
                    row.push(as_mat(&v)?.clone());
                }
            }
        }
    }

    fn call(
        &self,
        name: &str,
        args: Option<Vec<WsValue>>,
        first_arg_name: Option<&str>,
        _t: &[Token],
        _i: usize,
    ) -> Result<WsValue> {
        let p = self.p;
        let Some(args) = args else {
            if let Some(v) = p.q1.workspace.vars.get(name) {
                return Ok(v.clone());
            }
            if let Some(v) = p.mf_locals.get(name) {
                return Ok(scalar(*v));
            }
            // a calibrated parameter: Dynare's driver also assigns it to a MATLAB variable
            // (`rho = M_.params(3);`)
            if let Some(s) = p.b.symbol(name)
                && s.kind == SymbolKind::Param
                && !p.b.params[s.index].is_nan()
            {
                return Ok(scalar(p.b.params[s.index]));
            }
            return match name {
                "pi" => Ok(scalar(std::f64::consts::PI)),
                "Inf" | "inf" => Ok(scalar(f64::INFINITY)),
                "NaN" | "nan" => Ok(scalar(f64::NAN)),
                "true" => Ok(scalar(1.0)),
                "false" => Ok(scalar(0.0)),
                _ => bail!("unknown variable '{name}'"),
            };
        };
        // indexing a workspace variable: x(k), x(a:b)
        if let Some(WsValue::Num(m)) = p.q1.workspace.vars.get(name) {
            let idx: Vec<usize> = match args.as_slice() {
                [k] => as_mat(k)?.iter().map(|&v| v as usize).collect(),
                _ => bail!("{name}: only linear indexing is supported"),
            };
            let vals: Result<Vec<f64>> = idx
                .iter()
                .map(|&k| {
                    if k >= 1 && k <= m.len() { Ok(m[k - 1]) } else { Err(anyhow!("{name}: index {k} out of bounds")) }
                })
                .collect();
            let vals = vals?;
            return Ok(num(if m.ncols() == 1 {
                Mat::from_column_slice(vals.len(), 1, &vals)
            } else {
                Mat::from_row_slice(1, vals.len(), &vals)
            }));
        }
        let unary = |f: fn(f64) -> f64| -> Result<WsValue> {
            match args.as_slice() {
                [a] => Ok(num(as_mat(a)?.map(f))),
                _ => bail!("{name}: expected one argument"),
            }
        };
        let dims = |a: &[WsValue]| -> Result<(usize, usize)> {
            Ok(match a {
                [n] => {
                    let n = as_scalar(n)? as usize;
                    (n, n)
                }
                [m, n] => (as_scalar(m)? as usize, as_scalar(n)? as usize),
                _ => bail!("{name}: expected one or two dimensions"),
            })
        };
        match name {
            "exp" => unary(f64::exp),
            "log" => unary(f64::ln),
            "sqrt" => unary(f64::sqrt),
            "abs" => unary(f64::abs),
            "floor" => unary(f64::floor),
            "ceil" => unary(f64::ceil),
            "round" => unary(f64::round),
            // max/min of a vector, or elementwise of two arrays (a scalar broadcasts); NaN
            // is ignored as in MATLAB
            "max" | "min" => {
                let pick = |a: f64, b: f64| match (a.is_nan(), b.is_nan()) {
                    (true, _) => b,
                    (_, true) => a,
                    _ if (name == "max") == (b > a) => b,
                    _ => a,
                };
                match args.as_slice() {
                    [v] => {
                        let v = as_mat(v)?;
                        if v.nrows() != 1 && v.ncols() != 1 {
                            bail!("{name}: only vectors are supported");
                        }
                        Ok(scalar(v.iter().copied().reduce(pick).unwrap_or(f64::NAN)))
                    }
                    [a, b] => {
                        let (a, b) = (as_mat(a)?, as_mat(b)?);
                        let (r, c) = if a.len() == 1 { b.shape() } else { a.shape() };
                        if !(a.len() == 1 || b.len() == 1 || a.shape() == b.shape()) {
                            bail!("{name}: arrays of different sizes");
                        }
                        let at = |m: &Mat, i: usize, j: usize| if m.len() == 1 { m[0] } else { m[(i, j)] };
                        Ok(num(Mat::from_fn(r, c, |i, j| pick(at(a, i, j), at(b, i, j)))))
                    }
                    _ => bail!("{name}: expected one or two arguments"),
                }
            }
            "transpose" => match args.as_slice() {
                [a] => Ok(num(as_mat(a)?.transpose())),
                _ => bail!("transpose: expected one argument"),
            },
            "linspace" => match args.as_slice() {
                [a, b, n] => {
                    let (a, b, n) = (as_scalar(a)?, as_scalar(b)?, as_scalar(n)?.floor() as usize);
                    Ok(num(Mat::from_fn(1, n, |_, k| {
                        if n == 1 {
                            b
                        } else if k + 1 == n {
                            b
                        } else {
                            a + k as f64 * (b - a) / (n - 1) as f64
                        }
                    })))
                }
                [a, b] => {
                    let (a, b) = (as_scalar(a)?, as_scalar(b)?);
                    Ok(num(Mat::from_fn(1, 100, |_, k| if k == 99 { b } else { a + k as f64 * (b - a) / 99.0 })))
                }
                _ => bail!("linspace: expected two or three arguments"),
            },
            "ones" => {
                let (m, n) = dims(&args)?;
                Ok(num(Mat::from_element(m, n, 1.0)))
            }
            "zeros" => {
                let (m, n) = dims(&args)?;
                Ok(num(Mat::zeros(m, n)))
            }
            "repmat" => match args.as_slice() {
                [a, m, n] => {
                    let (a, m, n) = (as_mat(a)?, as_scalar(m)? as usize, as_scalar(n)? as usize);
                    Ok(num(Mat::from_fn(a.nrows() * m, a.ncols() * n, |r, c| a[(r % a.nrows(), c % a.ncols())])))
                }
                _ => bail!("repmat: expected three arguments"),
            },
            "dates" => match args.as_slice() {
                [WsValue::Str(s)] => {
                    Ok(WsValue::Str(Date::parse(s).ok_or_else(|| anyhow!("dates('{s}'): not a date"))?.label()))
                }
                _ => bail!("dates: only dates('DATE') is supported"),
            },
            "importdata" => match args.as_slice() {
                // importdata('file.csv'): the numeric block under one header line, as a struct
                // with fields `data` and `colheaders`
                [WsValue::Str(f)] => {
                    let path = if Path::new(f).is_absolute() { Path::new(f).to_path_buf() } else { self.dir.join(f) };
                    let t = crate::data::Table::read_csv(&path).map_err(|e| anyhow!("importdata: {e}"))?;
                    let data = t.values.clone();
                    Ok(WsValue::Table { names: t.names.clone(), data })
                }
                _ => bail!("importdata: only importdata('FILE.csv') is supported"),
            },
            "vertcat" | "horzcat" if args.iter().all(|a| matches!(a, WsValue::Cell(_) | WsValue::Str(_))) => {
                Ok(WsValue::Cell(args.iter().map(as_strings).collect::<Result<Vec<_>>>()?.concat()))
            }
            "vertcat" | "horzcat" => {
                let m: Vec<Mat> = args.iter().map(|a| as_mat(a).cloned()).collect::<Result<_>>()?;
                Ok(num(if name == "vertcat" { vcat(&m)? } else { hcat(&m)? }))
            }
            "dseries" => self.dseries(&args),
            "table" => self.table(&args, first_arg_name),
            "array2table" => match args.as_slice() {
                [m, k, names] if matches!(k, WsValue::Str(s) if s == "VariableNames") => {
                    let m = as_mat(m)?.clone();
                    let names = as_strings(names)?;
                    if names.len() != m.ncols() {
                        bail!("array2table: {} names for {} columns", names.len(), m.ncols());
                    }
                    Ok(WsValue::Table { names, data: m })
                }
                [m] => {
                    let m = as_mat(m)?.clone();
                    let names = (1..=m.ncols()).map(|k| format!("Var{k}")).collect();
                    Ok(WsValue::Table { names, data: m })
                }
                _ => bail!("array2table: expected (MATRIX[, 'VariableNames', NAMES])"),
            },
            _ => bail!("function '{name}' is not supported"),
        }
    }

    /// `dseries(file)`, `dseries(data)`, `dseries(data, init)`, `dseries(data, init, names)`.
    fn dseries(&self, args: &[WsValue]) -> Result<WsValue> {
        if let [WsValue::Str(f)] = args
            && Date::parse(f).is_none()
        {
            let (init, names, data) = crate::deterministic::workspace::load_dseries_file(f, self.dir)?;
            return Ok(WsValue::Dseries { init, names, data });
        }
        let data = as_mat(args.first().ok_or_else(|| anyhow!("dseries: expected arguments"))?)?.clone();
        let init = match args.get(1) {
            Some(v) => as_date(v)?,
            None => Date::new(crate::dates::Freq::Annual, 1),
        };
        let names = match args.get(2) {
            Some(v) => as_strings(v)?,
            None => (1..=data.ncols()).map(|k| format!("Variable_{k}")).collect(),
        };
        if names.len() != data.ncols() {
            bail!("dseries: {} names for {} columns", names.len(), data.ncols());
        }
        Ok(WsValue::Dseries { init, names, data })
    }

    /// `table(col1, col2, ..., 'VariableNames', {...})` (default names: the argument names).
    fn table(&self, args: &[WsValue], first_arg_name: Option<&str>) -> Result<WsValue> {
        let k = args.iter().position(|a| matches!(a, WsValue::Str(s) if s == "VariableNames")).unwrap_or(args.len());
        let cols: Vec<Mat> = args[..k].iter().map(|a| as_mat(a).cloned()).collect::<Result<_>>()?;
        let data = hcat(&cols)?;
        let names = match args.get(k + 1) {
            Some(v) => as_strings(v)?,
            None if cols.len() == 1 && data.ncols() == 1 => {
                vec![first_arg_name.ok_or_else(|| anyhow!("table: give 'VariableNames'"))?.to_string()]
            }
            None => bail!("table: give 'VariableNames'"),
        };
        if names.len() != data.ncols() {
            bail!("table: {} names for {} columns", names.len(), data.ncols());
        }
        Ok(WsValue::Table { names, data })
    }
}
