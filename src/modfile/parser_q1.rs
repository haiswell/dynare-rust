// SPDX-License-Identifier: GPL-3.0-or-later
//! Declarations handled outside the main parser: `var(log)`,
//! `var(deflator = ...)`, `var(log_deflator = ...)`, `trend_var(growth_factor = ...)`,
//! `log_trend_var(log_growth_factor = ...)` (the expressions are replayed into the dynamic
//! model's tree by `model::dtree`, as the preprocessor parses them there), and
//! `model(balanced_growth_test_tol = ...)`.

use super::{Cursor, Parser, split_commas};
use crate::model::SymbolKind;
use crate::model::dtree::ModelItem;
use anyhow::{Result, anyhow, bail};

/// Parser state of .
#[derive(Default)]
pub(super) struct Q1Parse {
    /// Line of the equation being parsed (`equations_lineno`).
    pub eq_line: usize,
    /// `shock_paths` blocks.
    pub shock_paths: Vec<crate::deterministic::shock_paths::ShockPathsBlock>,
    /// `varexo_det` of a stochastic model: their deterministic shocks and steady state.
    pub exo_det_shocks: Vec<crate::modfile::DetShock>,
    pub exo_det_steady_state: Vec<f64>,
    // ---- shocks/shock_paths ----
    /// `M_.database`.
    pub databases: Vec<String>,
    /// MATLAB workspace variables of the .mod file.
    pub workspace: crate::deterministic::workspace::Workspace,
    /// Index in `Parser::shocks` where the current `shocks` block starts.
    pub shocks_block_start: usize,
    /// `corr` entries of the `shocks` blocks (`M_.Correlation_matrix`, `M_.Correlation_matrix_ME`).
    pub shock_corrs: Vec<(super::Tgt, super::Tgt, f64)>,
}

/// Options of a `var(...)` declaration.
#[derive(Default)]
pub(super) struct VarOptions {
    pub log: bool,
    /// (deflator expression, its numerical constants, log deflator)
    pub deflator: Option<(crate::modfile::Ast, Vec<String>, bool)>,
}

impl<'a> Parser<'a> {
    /// Parses the `( ... )` after `var` (the cursor is on `(`).
    pub(super) fn q1_var_options(&mut self) -> Result<VarOptions> {
        let st = self.c.i;
        self.c.skip_parens()?;
        let inner = &self.c.t[st + 1..self.c.i - 1];
        let mut o = VarOptions::default();
        for item in split_commas(inner) {
            match item {
                [] => {}
                [k] if k.is_ident("log") => o.log = true,
                [k, eq, rest @ ..] if eq.is("=") && (k.is_ident("deflator") || k.is_ident("log_deflator")) => {
                    let mut c = Cursor::new(rest);
                    let ast = c.expr()?;
                    if let Some(t) = c.peek() {
                        bail!("{}: unexpected '{}' in the deflator", t.loc, t.raw);
                    }
                    o.deflator = Some((ast, std::mem::take(&mut c.nums), k.is_ident("log_deflator")));
                }
                [k, ..] => bail!("{}: unknown option '{}' of var", k.loc, k.raw),
            }
        }
        Ok(o)
    }

    /// Records the declared variables' options (after `declaration` declared `names`).
    pub(super) fn q1_apply_var_options(&mut self, o: VarOptions, names: Vec<String>) {
        if o.log {
            for n in &names {
                if let Some(s) = self.b.symbol(n) {
                    self.b.log_vars.push(s.index);
                }
            }
        }
        if let Some((ast, nums, log)) = o.deflator {
            self.b.items.push(ModelItem::Deflator { vars: names, ast, nums, log });
        }
    }

    /// `trend_var(growth_factor = EXPR) A B;` / `log_trend_var(log_growth_factor = EXPR) ...;`
    pub(super) fn q1_trend_var(&mut self, log: bool) -> Result<()> {
        let kw = self.c.next()?;
        if !self.c.at("(") {
            bail!("{}: {} requires the {}growth_factor option", kw.loc, kw.raw, if log { "log_" } else { "" });
        }
        let st = self.c.i;
        self.c.skip_parens()?;
        let inner = &self.c.t[st + 1..self.c.i - 1];
        let key = if log { "log_growth_factor" } else { "growth_factor" };
        let (ast, nums) = match inner {
            [k, eq, rest @ ..] if eq.is("=") && k.is_ident(key) => {
                let mut c = Cursor::new(rest);
                let ast = c.expr()?;
                if let Some(t) = c.peek() {
                    bail!("{}: unexpected '{}' in {key}", t.loc, t.raw);
                }
                (ast, std::mem::take(&mut c.nums))
            }
            _ => bail!("{}: {} requires the {key} option", kw.loc, kw.raw),
        };
        let mut vars = Vec::new();
        for t in self.c.statement()? {
            if let Some(n) = t.ident() {
                let k = self.b.declare(n, SymbolKind::Trend).map_err(|e| anyhow!("{}: {e}", t.loc))?;
                self.b.trend_names[k].1 = log;
                vars.push(n.to_string());
            } else if !t.is(",") && !matches!(t.tok, super::Tok::Tex(_)) {
                bail!("{}: unexpected '{}' in {}", t.loc, t.raw, kw.raw);
            }
        }
        self.b.items.push(ModelItem::Trend { vars, ast, nums, log });
        Ok(())
    }

    /// `varexo` after `varexo_det`: the variables declared since index `before` move in front
    /// of the `varexo_det` block, which stays at the end of the exogenous list until
    /// [`Self::q1_exo_det_view`] takes it out (Dynare numbers the two kinds separately).
    pub(super) fn q1_keep_exo_det_last(&mut self, before: usize) -> Result<()> {
        let Some(&first) = self.exo_det.iter().min() else { return Ok(()) };
        let added = self.b.exo_names.len() - before;
        if added == 0 {
            return Ok(());
        }
        let refs = !self.shocks.is_empty()
            || !self.est.is_empty()
            || !self.est_init.is_empty()
            || !self.histval.is_empty()
            || !self.det_shocks.is_empty();
        if refs {
            bail!(
                "varexo after varexo_det: declare the exogenous variables before the shocks, histval and estimated_params blocks"
            );
        }
        for k in 0..added {
            self.b.move_exo(before + k, first + k);
            let t = self.labels.exo_tex.remove(before + k);
            let l = self.labels.exo_long.remove(before + k);
            self.labels.exo_tex.insert(first + k, t);
            self.labels.exo_long.insert(first + k, l);
        }
        for i in self.exo_det.iter_mut() {
            *i += added;
        }
        for (i, _) in self.initval_exo.iter_mut() {
            if *i >= first {
                *i = if *i >= before { *i - before + first } else { *i + added };
            }
        }
        Ok(())
    }

    /// `shocks`: variances, standard errors, covariances and correlations cannot be set on a
    /// `varexo_det` variable (`ShocksStatement::checkPass` of the 7.2 preprocessor).
    pub(super) fn q1_check_exo_det_moment(&self, what: &str, a: super::Tgt, b: Option<super::Tgt>) -> Result<()> {
        let det = |t: super::Tgt| matches!(t, super::Tgt::Exo(i) if self.exo_det.contains(&i));
        let name = |t: super::Tgt| match t {
            super::Tgt::Exo(i) => self.b.exo_names[i].clone(),
            super::Tgt::Endo(i) => self.b.endo_names[i].clone(),
        };
        match b {
            None if det(a) => bail!(
                "shocks: setting a {what} on '{}' is not allowed, because it is neither an exogenous variable nor an observed endogenous variable",
                name(a)
            ),
            Some(b) if det(a) || det(b) => {
                let plural = if what == "covariance" { "covariances" } else { "correlations" };
                bail!(
                    "shocks: setting a {what} between '{}' and '{}'is not allowed; {plural} can only be specified for exogenous or observed endogenous variables of same type",
                    name(a),
                    name(b)
                )
            }
            _ => Ok(()),
        }
    }

    /// `varexo_det`: the deterministic exogenous variables leave the exogenous list
    /// (`M_.exo_det_names`, not in `M_.exo_names` nor `M_.Sigma_e`) and enter the model as
    /// parameters at their `initval` value (`oo_.exo_det_steady_state`); their `shocks` paths
    /// are kept for forecasts (`ModFile::q1.exo_det_shocks`, `M_.det_shocks` with `exo_det`).
    /// Raises the 7.2 preprocessor's errors: a lead or lag on a `varexo_det` variable, one in
    /// a variable list or `irf_shocks`, a perfect-foresight solver in the file (and the mix of
    /// perfect-foresight and stochastic commands).
    pub(super) fn q1_exo_det_view(&mut self) -> Result<()> {
        // ModFile::checkPass: perfect foresight and stochastic contexts
        let pf = self.commands.iter().any(|c| match c {
            super::Command::Deterministic(n, ..) => n == "perfect_foresight_solver" || n == "simul",
            super::Command::Other(n, _) => n == "perfect_foresight_with_expectation_errors_solver",
            _ => false,
        });
        let stoch = self.commands.iter().any(|c| match c {
            super::Command::StochSimul(..) | super::Command::Estimation(..) => true,
            super::Command::Policy(n, ..) => matches!(n.as_str(), "osr" | "ramsey_policy" | "discretionary_policy"),
            _ => false,
        });
        if pf && stoch {
            bail!(
                "A .mod file cannot contain both one of {{perfect_foresight_solver, simul, perfect_foresight_with_expectation_errors_solver}} and one of {{stoch_simul, estimation, osr, ramsey_policy, discretionary_policy}}. This is not possible: one cannot mix perfect foresight context with stochastic context in the same file."
            );
        }
        if self.exo_det.is_empty() {
            return Ok(());
        }
        let det_names: Vec<String> = self.exo_det.iter().map(|&k| self.b.exo_names[k].clone()).collect();
        // a lead or a lag (rejected by the preprocessor's parser)
        fn walk(a: &crate::modfile::Ast, det: &[String]) -> Result<()> {
            use crate::modfile::Ast;
            match a {
                Ast::Var { name, lag, loc } if *lag != 0 && det.contains(name) => {
                    bail!("{loc}: Exogenous deterministic variable {name} cannot be given a lead or a lag.")
                }
                Ast::Num(_) | Ast::Var { .. } => Ok(()),
                Ast::Un(_, x) => walk(x, det),
                Ast::Bin(_, x, y) => walk(x, det).and_then(|_| walk(y, det)),
                Ast::Op(_, v) => v.iter().try_for_each(|x| walk(x, det)),
            }
        }
        for it in &self.b.items {
            match it {
                ModelItem::Equation { lhs, rhs, .. } => {
                    walk(lhs, &det_names)?;
                    if let Some(r) = rhs {
                        walk(r, &det_names)?;
                    }
                }
                ModelItem::Local { ast, .. }
                | ModelItem::Trend { ast, .. }
                | ModelItem::Deflator { ast, .. }
                | ModelItem::Preamble { ast, .. } => walk(ast, &det_names)?,
                ModelItem::Remove { .. } => {}
            }
        }
        if pf {
            bail!(
                "A .mod file cannot contain both one of {{perfect_foresight_solver, simul, perfect_foresight_with_expectation_errors_solver}} and varexo_det declaration (all exogenous variables are deterministic in this case)"
            );
        }
        // variable lists and irf_shocks
        fn strs(v: &crate::modfile::OptValue, out: &mut Vec<String>) {
            match v {
                crate::modfile::OptValue::Str(s) => out.push(s.clone()),
                crate::modfile::OptValue::List(l) => l.iter().for_each(|x| strs(x, out)),
                _ => {}
            }
        }
        for c in &self.commands {
            let (cmd, o, vars) = match c {
                super::Command::StochSimul(o, v) => ("stoch_simul", o, v),
                super::Command::Forecast(o, v) => ("forecast", o, v),
                _ => continue,
            };
            if let Some(v) = vars.iter().find(|v| det_names.contains(v)) {
                bail!("{cmd}: Variable {v} is not one of {{endogenous}}");
            }
            if let Some(v) = crate::modfile::opt(o, "irf_shocks") {
                let mut names = Vec::new();
                strs(v, &mut names);
                if let Some(n) = names.iter().find(|n| det_names.contains(n)) {
                    bail!("Variables passed to irf_shocks must be exogenous. Caused by: {n}");
                }
            }
        }
        if self.b.planner.as_ref().is_some_and(|p| p.ramsey) {
            bail!("ramsey_model and ramsey_policy are incompatible with deterministic exogenous variables");
        }
        if self.b.planner.is_some() {
            println!(
                "NOTE: varexo_det in an optimal-policy model: treated as ordinary exogenous variables by dynare-rust."
            );
            return Ok(());
        }
        let nx = self.b.exo_names.len();
        let nd = self.exo_det.len();
        let mut det = self.exo_det.clone();
        det.sort_unstable();
        if !det.iter().enumerate().all(|(k, &i)| i == nx - nd + k) {
            // only after exclude_eqs turned an endogenous variable into an exogenous one
            println!(
                "NOTE: varexo_det variables declared before other exogenous variables are treated as ordinary exogenous variables by dynare-rust."
            );
            return Ok(());
        }
        let ns = nx - nd;
        let mut values = vec![0.0; nd];
        for &(k, v) in &self.initval_exo {
            if k >= ns {
                values[k - ns] = v;
            }
        }
        self.initval_exo.retain(|&(k, _)| k < ns);
        // M_.exo_det_histval: never read by the stochastic commands (simult_varexo_det.m starts
        // at period 1)
        self.histval.retain(|&(endo, k, _, _)| endo || k < ns);
        let (keep, moved): (Vec<_>, Vec<_>) = self.det_shocks.drain(..).partition(|d| d.exo < ns);
        self.det_shocks = keep;
        self.q1.exo_det_shocks = moved
            .into_iter()
            .map(|mut d| {
                d.exo -= ns;
                d
            })
            .collect();
        let names: Vec<String> = self.b.exo_names.drain(ns..).collect();
        self.labels.exo_det_tex = self.labels.exo_tex.split_off(ns);
        self.labels.exo_det_long = self.labels.exo_long.split_off(ns);
        self.q1.exo_det_steady_state = values.clone();
        self.b.q1_exo_det = names.into_iter().zip(values).collect();
        self.exo_det.clear();
        Ok(())
    }

    /// The balanced-growth test context, set before the model is built: the calibration is
    /// filled in by `ModelBuilder::build`, the `initval` values here (`global_eval_context`).
    pub(super) fn q1_bg_context(&mut self) {
        let mut endo = vec![0.0; self.b.endo_names.len()];
        for &(j, v) in &self.initval {
            if j < endo.len() {
                endo[j] = v;
            }
        }
        let mut exo = vec![0.0; self.b.exo_names.len()];
        for &(k, v) in &self.initval_exo {
            if k < exo.len() {
                exo[k] = v;
            }
        }
        self.b.bg.endo = endo;
        self.b.bg.exo = exo;
    }
}

impl<'a> Parser<'a> {
    /// `shock_paths(learnt_in=... overwrite); var x; periods ...; values ...; exogenize y;
    /// periods ...; values ...; endogenize e; end;` (stored in `ModFile::q1`, a
    /// `shock_paths` deterministic command marks the block's position).
    pub(super) fn q1_shock_paths_block(&mut self) -> Result<()> {
        use crate::deterministic::shock_paths::{ControlledStanza, ShockPathsBlock, ShockStanza};
        use crate::modfile::{Command, OptValue};
        let loc0 = self.c.next()?.loc.clone();
        // ---- shocks/shock_paths: options separated by spaces (7.2 grammar) or commas
        let opts = self.shocks_header_options()?;
        let li = super::shk::learnt_in_option(&opts, "shock_paths").map_err(|e| anyhow!("{loc0}: {e}"))?;
        let mut b = ShockPathsBlock {
            learnt_in: li.0,
            learnt_in_date: li.1,
            overwrite: opts.iter().any(|(k, _)| k == "overwrite"),
            active: true,
            ..Default::default()
        };
        let mut scope_dates = false;
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                break;
            }
            let kw = self.c.next()?;
            let (kwr, kloc) = (kw.raw.clone(), kw.loc.clone());
            let (name, nloc) = self.c.ident()?;
            let (name, nloc) = (name.to_string(), nloc.clone());
            self.c.expect(";")?;
            let mut periods: Option<Vec<super::Token>> = None;
            let mut values: Option<Vec<super::Token>> = None;
            for _ in 0..2 {
                let k = self.c.next()?;
                let kr = k.raw.clone();
                let toks = self.c.statement()?.to_vec();
                match kr.as_str() {
                    "periods" => periods = Some(toks),
                    "values" => values = Some(toks),
                    _ => bail!("{}: shock_paths: expected 'periods' or 'values', found '{kr}'", k.loc),
                }
            }
            let (Some(pt), Some(vt)) = (periods, values) else {
                bail!("{kloc}: shock_paths: a stanza needs periods and values")
            };
            let ranges = q1_period_ranges(&pt)?;
            let mut vals: Vec<crate::modfile::Ast> = Vec::new();
            for t in split_commas(&vt) {
                let (ast, has_date) = self.q1_scoped_expr(t)?;
                scope_dates |= has_date;
                vals.push(ast);
            }
            if ranges.len() != vals.len() {
                bail!(
                    "{kloc}: shock_paths: variable {name}: number of periods is different from number of shock values"
                );
            }
            match kwr.as_str() {
                "var" => {
                    let exo = match self.b.symbol(&name) {
                        Some(s) if s.kind == SymbolKind::Exo => s.index,
                        _ => bail!("{nloc}: shock_paths: '{name}' is not an exogenous variable"),
                    };
                    b.stanzas.push(ShockStanza { exo, items: ranges.into_iter().zip(vals).collect() });
                }
                "exogenize" => {
                    let exogenize = match self.b.symbol(&name) {
                        Some(s) if s.kind == SymbolKind::Endo => s.index,
                        _ => bail!("{nloc}: shock_paths: '{name}' is not an endogenous variable"),
                    };
                    if !self.c.at_ident("endogenize") {
                        bail!("{nloc}: shock_paths: exogenize {name}: missing 'endogenize'");
                    }
                    self.c.i += 1;
                    let (e, eloc) = self.c.ident()?;
                    let endogenize = match self.b.symbol(e) {
                        Some(s) if s.kind == SymbolKind::Exo => s.index,
                        _ => bail!("{eloc}: shock_paths: '{e}' is not an exogenous variable"),
                    };
                    self.c.expect(";")?;
                    let mut items = Vec::new();
                    for (r, v) in ranges.into_iter().zip(vals) {
                        use crate::deterministic::shock_paths::PeriodPoint as P;
                        match (r.from, r.to) {
                            (Some(P::Int(a)), Some(P::Int(z))) => {
                                items.push(((a as i64..=z as i64).collect(), Vec::new(), v))
                            }
                            (Some(P::Date(a)), Some(P::Date(z))) => {
                                items.push((Vec::new(), crate::dates::Date::range(a, z)?, v))
                            }
                            (Some(_), Some(_)) => {
                                bail!("{kloc}: shock_paths: a period range cannot mix an integer and a date")
                            }
                            _ => bail!("{kloc}: shock_paths: 'end' is not allowed in an exogenize stanza"),
                        }
                    }
                    b.controlled.push(ControlledStanza { exogenize, endogenize, items });
                }
                _ => bail!("{kloc}: shock_paths: expected 'var' or 'exogenize', found '{kwr}'"),
            }
        }
        b.contains_date = b.learnt_in_date.is_some() || scope_dates || b.periods_have_dates();
        if b.overwrite {
            for x in self.q1.shock_paths.iter_mut() {
                if x.same_learnt_in(li.0, li.1) {
                    x.active = false;
                }
            }
            // (the generated filter also applies to M_.perfect_foresight_controlled_paths: the
            // driver applies it when it reaches the block)
        }
        let k = self.q1.shock_paths.len();
        self.q1.shock_paths.push(b);
        self.commands.push(Command::Deterministic(
            "shock_paths".into(),
            vec![("__index".into(), OptValue::Num(k as f64))],
            Vec::new(),
        ));
        Ok(())
    }

    /// A `values` expression with its scopes (`initval.x`, `init.x`, `self.x`, `prev.x`,
    /// `learnt_in(INTEGER | DATE).x`, `DATABASE.x`) rewritten into the scoped names of
    /// `deterministic::shock_paths`; also whether a `learnt_in(DATE)` scope appears.
    fn q1_scoped_expr(&self, toks: &[super::Token]) -> Result<(crate::modfile::Ast, bool)> {
        use crate::deterministic::shock_paths::{Scope, scoped_name};
        let mut out: Vec<super::Token> = Vec::with_capacity(toks.len());
        let mut has_date = false;
        let mut k = 0;
        while k < toks.len() {
            let t = &toks[k];
            // learnt_in(INTEGER | DATE).x
            if t.is_ident("learnt_in") && toks.get(k + 1).is_some_and(|x| x.is("(")) {
                let (scope, n) = match super::shk::point_at(toks, k + 2) {
                    Ok((Ok(p), n)) if p >= 1 => (Scope::LearntIn(p as usize, None), n),
                    Ok((Err(d), n)) => {
                        has_date = true;
                        (Scope::LearntIn(0, Some(d)), n)
                    }
                    _ => bail!("{}: shock_paths: learnt_in(...) expects a positive integer or a date", t.loc),
                };
                let j = k + 2 + n;
                if !(toks.get(j).is_some_and(|x| x.is(")")) && toks.get(j + 1).is_some_and(|x| x.is("."))) {
                    bail!("{}: shock_paths: expected learnt_in(PERIOD).VARIABLE", t.loc);
                }
                let Some(vn) = toks.get(j + 2).and_then(|x| x.ident()) else {
                    bail!("{}: shock_paths: expected learnt_in(PERIOD).VARIABLE", t.loc)
                };
                let name = scoped_name(scope, vn);
                out.push(super::Token {
                    tok: super::Tok::Ident(name.clone()),
                    raw: name,
                    loc: t.loc.clone(),
                    col: t.col,
                });
                k = j + 3;
                continue;
            }
            if let (Some(sc), Some(dot), Some(v)) = (t.ident(), toks.get(k + 1), toks.get(k + 2))
                && dot.is(".")
                && let Some(vn) = v.ident()
            {
                let scope = match sc {
                    "initval" | "init" => Scope::Init,
                    "self" => Scope::SelfPath,
                    "prev" => Scope::Prev,
                    _ => match self.q1.databases.iter().position(|d| d == sc) {
                        Some(i) => Scope::Database(i),
                        None => bail!(
                            "{}: shock_paths: unknown scope '{sc}' (not initval, init, self, prev, learnt_in(...) or a database declared with the database command)",
                            t.loc
                        ),
                    },
                };
                let name = scoped_name(scope, vn);
                out.push(super::Token {
                    tok: super::Tok::Ident(name.clone()),
                    raw: name,
                    loc: t.loc.clone(),
                    col: t.col,
                });
                k += 3;
                continue;
            }
            out.push(t.clone());
            k += 1;
        }
        let mut c = Cursor::new(&out);
        let ast = c.expr()?;
        if let Some(t) = c.peek() {
            bail!("{}: shock_paths: unexpected '{}' in a value", t.loc, t.raw);
        }
        Ok((ast, has_date))
    }
}

/// Command name of a `set_shock_stderr_value` after the first command (options
/// `[(exogenous name, Num(stderr))]`, run by `misc::Session`).
pub const SET_SHOCK_STDERR: &str = "__set_shock_stderr";

impl<'a> Parser<'a> {
    /// `set_shock_stderr_value('e', VALUE);` (MATLAB's `M_.Sigma_e(i,i) = value^2`): applied to
    /// the `shocks` declarations before the first command, else at its place (a command).
    pub(super) fn q1_set_shock_stderr_value(&mut self) -> Result<()> {
        use crate::modfile::{Command, OptValue};
        self.c.i += 1;
        let toks = self.c.statement()?;
        let loc = toks.first().map(|t| t.loc.to_string()).unwrap_or_default();
        if !(toks.len() >= 2 && toks[0].is("(") && toks[toks.len() - 1].is(")")) {
            bail!("{loc}: set_shock_stderr_value: expected ('NAME', VALUE)");
        }
        let fields = split_commas(&toks[1..toks.len() - 1]);
        let [[name_tok], value] = fields.as_slice() else {
            bail!("{loc}: set_shock_stderr_value: expected ('NAME', VALUE)")
        };
        let name = match &name_tok.tok {
            super::Tok::Str(s) | super::Tok::Ident(s) => s.clone(),
            _ => bail!("{loc}: set_shock_stderr_value: expected a shock name"),
        };
        let v = self.eval_tokens(value)?;
        let i = match self.b.symbol(&name) {
            Some(s) if s.kind == SymbolKind::Exo => s.index,
            _ => bail!("{loc}: Shock name {name} doesn't exist"),
        };
        if self.commands.is_empty() {
            let stmt = super::ShockStmt::Var(super::Tgt::Exo(i), super::PathValue::Num(v * v));
            self.shocks.push(stmt.clone());
            let block =
                super::RawStateBlock::Shocks { overwrite: false, stmts: vec![stmt], reset_det: false, det: Vec::new() };
            self.state_blocks.push((0, block));
        } else {
            self.commands.push(Command::Other(SET_SHOCK_STDERR.into(), vec![(name, OptValue::Num(v))]));
        }
        Ok(())
    }
}

/// `periods 1, 2:5, 6:end, end, 2000Q1:2000Q3, 2001Q2:end` of a `shock_paths` stanza.
fn q1_period_ranges(toks: &[super::Token]) -> Result<Vec<crate::deterministic::shock_paths::PeriodRange>> {
    use crate::deterministic::shock_paths::{PeriodPoint, PeriodRange};
    let point = |item: &[super::Token], i: usize| -> Result<(PeriodPoint, usize)> {
        match super::shk::point_at(item, i) {
            Ok((Ok(p), n)) if p >= 1 => Ok((PeriodPoint::Int(p as usize), n)),
            Ok((Err(d), n)) => Ok((PeriodPoint::Date(d), n)),
            _ => bail!(
                "{}: shock_paths: expected a period (positive integer, date or 'end')",
                item[i.min(item.len() - 1)].loc
            ),
        }
    };
    let mut out = Vec::new();
    for item in split_commas(toks) {
        if item.is_empty() {
            continue;
        }
        if let [e] = item
            && e.is_ident("end")
        {
            out.push(PeriodRange { from: None, to: None });
            continue;
        }
        let (a, n) = point(item, 0)?;
        let r = match &item[n..] {
            [] => PeriodRange { from: Some(a), to: Some(a) },
            [c, e] if c.is(":") && e.is_ident("end") => PeriodRange { from: Some(a), to: None },
            [c, ..] if c.is(":") => {
                let (z, m) = point(item, n + 1)?;
                if n + 1 + m != item.len() {
                    bail!("{}: shock_paths: invalid period entry", item[0].loc);
                }
                PeriodRange { from: Some(a), to: Some(z) }
            }
            _ => bail!("{}: shock_paths: invalid period entry", item[0].loc),
        };
        out.push(r);
    }
    Ok(out)
}

/// `steady_state_model` temporaries: the statement's source with every temporary replaced by
/// its (already inlined) definition in parentheses.
pub(super) fn inline_ss_locals(toks: &[super::Token], locals: &std::collections::HashMap<String, String>) -> String {
    let mut out = Vec::with_capacity(toks.len());
    for (k, t) in toks.iter().enumerate() {
        let fn_call = toks.get(k + 1).is_some_and(|n| n.is("("));
        match (&t.tok, t.ident()) {
            (_, Some(n)) if !fn_call && locals.contains_key(n) => out.push(format!("({})", locals[n])),
            (super::Tok::Str(s), _) => out.push(format!("'{s}'")),
            _ => out.push(t.raw.clone()),
        }
    }
    out.join(" ")
}
