// SPDX-License-Identifier: GPL-3.0-or-later
//! [stream X] Statements of the .mod file handled outside the main parser: parameter
//! assignments after the first command (Dynare runs them in file order), `options_.X = V;`,
//! `set_param_value`, `estimated_params(overwrite)`, `estimated_params_bounds`,
//! `estimated_params_remove`, `observation_trends`, `homotopy_setup`,
//! `load_params_and_steady_state` (read at preprocessing time, as Dynare's preprocessor
//! does) and other MATLAB-level statements (`M_.x = ...`), kept for a warning.

use super::{Parser, raw_join, slice_expr, split_commas};
use crate::model::SymbolKind;
use crate::modfile::lexer::{Tok, Token};
use crate::modfile::{Command, OptValue, Options};
use anyhow::{Context, Result, anyhow, bail};

/// Command name of a parameter assignment that follows a command: `M_.params(i) = v` at that
/// point of the driver (options: `[(name, Num(v))]`).
pub const SET_PARAM: &str = "__set_param";
/// Command name of an `options_.X = V;` statement (options: `[(path, value)]`, `path` lower
/// case, nested fields joined by `.`).
pub const OPTIONS_ASSIGNMENT: &str = "__options";
/// Command name of another MATLAB statement (`M_.x = ...`, `oo_...`): option `[(text, Flag)]`.
pub const MATLAB_STATEMENT: &str = "__matlab";

/// Options set by `options_.X = V` that are boolean flags in the command options (a command
/// reads them by presence): `options_.X = 0` removes them instead of setting them.
const BOOLEAN_FLAGS: &[&str] = &[
    "nograph",
    "nodisplay",
    "noprint",
    "nomoments",
    "nocorr",
    "nofunctions",
    "nodecomposition",
    "nomodelsummary",
    "smoother",
    "filtered_vars",
    "loglinear",
    "logdata",
    "pruning",
    "diffuse_filter",
    "bayesian_irf",
    "moments_varendo",
    "mh_tune_jscale",
    "endogenous_prior",
    "tex",
    "nodiagnostic",
    "smoothed_state_uncertainty",
    "relative_irf",
    "contemporaneous_correlation",
    "use_univariate_filters_if_singularity_is_detected",
    "console",
    "nointeractive",
    "noconstant",
    "print",
    "graph",
    "mode_check",
    "silent_optimizer",
    "conditional_likelihood",
];

/// Commands that do not take `options_` defaults (blocks parsed through `command`;
/// `set_dynare_seed`, whose arguments are positional).
const NO_GLOBAL_OPTIONS: &[&str] =
    &["shock_groups", "conditional_forecast_paths", "model_comparison", "set_dynare_seed"];

/// Parser state of stream X.
#[derive(Default)]
pub(super) struct XParse {
    /// `M_.params` when the first parameter assignment after a command was met: the
    /// calibration the commands before it see (`None`: no assignment after a command).
    pub params_snapshot: Option<Vec<f64>>,
    /// `options_.X = V` assignments so far (Dynare's global `options_`), appended as
    /// defaults to the options of every later command.
    pub global_opts: Options,
    pub observation_trends: Vec<(String, String)>,
    pub homotopy_setup: Vec<(String, Option<f64>, f64)>,
    /// `homotopy_setup(from_initval_to_endval)`: `options_.homotopy_from_initval_to_endval`.
    pub homotopy_from_initval_to_endval: bool,
    pub homotopy_block: bool,
}

impl<'a> Parser<'a> {
    /// Whether [`Parser::x_statement`] parses the statement starting with `kw`.
    pub(super) fn x_handles(&self, kw: &str) -> bool {
        let next_is = |p: &str| self.c.peek_at(1).is_some_and(|t| t.is(p));
        matches!(
            kw,
            "set_param_value"
                | "estimated_params_bounds"
                | "estimated_params_remove"
                | "observation_trends"
                | "homotopy_setup"
                | "load_params_and_steady_state"
        ) || (matches!(kw, "options_" | "M_" | "oo_" | "estim_params_" | "bayestopt_" | "estimation_info")
            && (next_is(".") || next_is("(")))
    }

    pub(super) fn x_statement(&mut self, kw: &str) -> Result<()> {
        match kw {
            "set_param_value" => self.x_set_param_value(),
            "estimated_params_bounds" => self.x_estimated_params_bounds(),
            "estimated_params_remove" => self.x_estimated_params_remove(),
            "observation_trends" => self.x_observation_trends(),
            "homotopy_setup" => self.x_homotopy_setup(),
            "load_params_and_steady_state" => self.x_load_params_and_steady_state(),
            "options_" => self.x_options_assignment(),
            _ => {
                let toks = self.c.statement()?;
                self.commands
                    .push(Command::Other(MATLAB_STATEMENT.into(), vec![(raw_join_spaced(toks), OptValue::Flag)]));
                Ok(())
            }
        }
    }

    /// Called by `calibration()` before `param = value` is stored: after the first command,
    /// the assignment also becomes a [`SET_PARAM`] command (Dynare's driver sets `M_.params`
    /// at that point), and the calibration seen by the earlier commands is kept.
    pub(super) fn x_param_assignment(&mut self, index: usize, name: &str, value: f64) {
        if self.commands.is_empty() {
            return;
        }
        if self.x.params_snapshot.is_none() {
            self.x.params_snapshot = Some(self.b.params.clone());
        }
        let _ = index;
        self.commands.push(Command::Other(SET_PARAM.into(), vec![(name.to_string(), OptValue::Num(value))]));
    }

    /// Appends the `options_` defaults to a command's options (the command's own win).
    pub(super) fn x_global_options(&self, name: &str, opts: &mut Options) {
        if NO_GLOBAL_OPTIONS.contains(&name) {
            return;
        }
        for (k, v) in &self.x.global_opts {
            // read by rplot only
            if k == "rplottype" && name != "rplot" {
                continue;
            }
            if !opts.iter().any(|(kk, _)| kk == k) {
                opts.push((k.clone(), v.clone()));
            }
        }
    }

    /// `set_param_value('name', expr);`
    fn x_set_param_value(&mut self) -> Result<()> {
        self.c.i += 1;
        let toks = self.c.statement()?;
        let loc = toks.first().map(|t| t.loc.to_string()).unwrap_or_default();
        if !(toks.len() >= 2 && toks[0].is("(") && toks[toks.len() - 1].is(")")) {
            bail!("{loc}: set_param_value: expected ('NAME', VALUE)");
        }
        let fields = split_commas(&toks[1..toks.len() - 1]);
        let [[name_tok], value] = fields.as_slice() else { bail!("{loc}: set_param_value: expected ('NAME', VALUE)") };
        let name = match &name_tok.tok {
            Tok::Str(s) | Tok::Ident(s) => s.clone(),
            _ => bail!("{loc}: set_param_value: expected a parameter name"),
        };
        // a MATLAB variable holding the name, or a MATLAB value
        // (`set_param_value(param, steady_state.params.(param))` in a for loop): kept for the
        // warning of an unsupported MATLAB statement
        let matlab_name =
            matches!(name_tok.tok, Tok::Ident(_)) && !self.b.symbol(&name).is_some_and(|s| s.kind == SymbolKind::Param);
        let v = match self.eval_tokens(value) {
            Ok(v) if !matlab_name => v,
            Err(e) if !matlab_name && !value.iter().any(|t| t.is(".") || t.is("{")) => return Err(e),
            _ => {
                self.commands.push(Command::Other(
                    MATLAB_STATEMENT.into(),
                    vec![(format!("set_param_value {}", raw_join_spaced(toks)), OptValue::Flag)],
                ));
                return Ok(());
            }
        };
        match self.b.symbol(&name) {
            Some(s) if s.kind == SymbolKind::Param => {
                let k = s.index;
                self.x_param_assignment(k, &name, v);
                self.b.params[k] = v;
                Ok(())
            }
            _ => bail!("{loc}: Parameter name {name} doesn't exist"),
        }
    }

    /// `options_.X = V;` / `options_.a.b = V;`
    fn x_options_assignment(&mut self) -> Result<()> {
        // a native MATLAB statement: ends with its line when no ';' ends it
        self.c.native_end = self.c.native_line_end();
        self.c.i += 1;
        let toks = self.c.statement()?;
        let loc = toks.first().map(|t| t.loc.to_string()).unwrap_or_default();
        let Some(eq) = toks.iter().position(|t| t.is("=")) else {
            bail!("{loc}: options_: expected options_.NAME = VALUE")
        };
        let mut path = Vec::new();
        for t in &toks[..eq] {
            match &t.tok {
                Tok::Punct(".") => {}
                Tok::Ident(s) => path.push(s.to_lowercase()),
                _ => {
                    // options_.x(2) = ... and the like: kept for a warning only
                    self.commands
                        .push(Command::Other(MATLAB_STATEMENT.into(), vec![(raw_join_spaced(toks), OptValue::Flag)]));
                    return Ok(());
                }
            }
        }
        let key = path.join(".");
        let value = match &toks[eq + 1..] {
            [t] if t.is_ident("true") => OptValue::Num(1.0),
            [t] if t.is_ident("false") => OptValue::Num(0.0),
            // options_.initial_date = dates('1989Q4')
            [d, o, t, c] if d.is_ident("dates") && o.is("(") && c.is(")") && matches!(t.tok, Tok::Str(_)) => {
                self.parse_value(std::slice::from_ref(t))
            }
            v => self.parse_value(v),
        };
        let value = match value {
            // MATLAB cell arrays {'a', 'b'}: a list of strings
            OptValue::Str(s) if s.starts_with('{') && s.ends_with('}') => {
                let inner = &toks[eq + 2..toks.len() - 1];
                OptValue::List(
                    split_commas(inner).into_iter().filter(|f| !f.is_empty()).map(|f| self.parse_value(f)).collect(),
                )
            }
            v => v,
        };
        self.x.global_opts.retain(|(k, _)| k != &key);
        let is_off = matches!(value, OptValue::Num(v) if v == 0.0) && BOOLEAN_FLAGS.contains(&key.as_str());
        if !key.contains('.') && !is_off {
            self.x.global_opts.push((key.clone(), value.clone()));
        }
        self.commands.push(Command::Other(OPTIONS_ASSIGNMENT.into(), vec![(key, value)]));
        Ok(())
    }

    /// `estimated_params_bounds; NAME, LB, UB; ... end;`
    fn x_estimated_params_bounds(&mut self) -> Result<()> {
        self.c.i += 1;
        self.header_options()?;
        while !self.c.at_end() {
            if self.c.peek().is_none() {
                bail!("{}: block 'estimated_params_bounds' not closed by 'end;'", self.c.loc());
            }
            let toks = self.c.statement()?;
            let loc = toks.first().map(|t| t.loc.to_string()).unwrap_or_default();
            let fields = split_commas(toks);
            let (kind, rest) = self.estimated_head(&fields)?;
            let [lb, ub] = rest else {
                bail!("{loc}: estimated_params_bounds: expected NAME, LOWER_BOUND, UPPER_BOUND")
            };
            let (lb, ub) = (self.eval_opt(lb)?, self.eval_opt(ub)?);
            let e =
                self.est.iter_mut().find(|e| e.kind == kind).ok_or_else(|| {
                    anyhow!("{loc}: estimated_params_bounds: this parameter is not in estimated_params")
                })?;
            e.lb = lb;
            e.ub = ub;
        }
        self.c.i += 2;
        self.est_snapshot();
        Ok(())
    }

    /// `estimated_params_remove; NAME; ... end;`
    fn x_estimated_params_remove(&mut self) -> Result<()> {
        self.c.i += 1;
        self.header_options()?;
        while !self.c.at_end() {
            if self.c.peek().is_none() {
                bail!("{}: block 'estimated_params_remove' not closed by 'end;'", self.c.loc());
            }
            let toks = self.c.statement()?;
            let loc = toks.first().map(|t| t.loc.to_string()).unwrap_or_default();
            let fields = split_commas(toks);
            let (kind, rest) = self.estimated_head(&fields)?;
            if !rest.is_empty() {
                bail!("{loc}: estimated_params_remove: expected only the parameter name");
            }
            let before = self.est.len();
            self.est.retain(|e| e.kind != kind);
            if self.est.len() == before {
                bail!("{loc}: estimated_params_remove: this parameter is not in estimated_params");
            }
            self.est_init.retain(|e| e.kind != kind);
        }
        self.c.i += 2;
        self.est_snapshot();
        Ok(())
    }

    /// `observation_trends; VAR (EXPR); ... end;`
    fn x_observation_trends(&mut self) -> Result<()> {
        self.c.i += 1;
        self.header_options()?;
        while !self.c.at_end() {
            if self.c.peek().is_none() {
                bail!("{}: block 'observation_trends' not closed by 'end;'", self.c.loc());
            }
            let toks = self.c.statement()?;
            let loc = toks.first().map(|t| t.loc.to_string()).unwrap_or_default();
            let ok = toks.len() >= 3 && toks[0].ident().is_some() && toks[1].is("(") && toks[toks.len() - 1].is(")");
            if !ok {
                bail!("{loc}: observation_trends: expected VARIABLE_NAME(EXPRESSION);");
            }
            let name = toks[0].raw.clone();
            if self.b.symbol(&name).is_none_or(|s| s.kind != SymbolKind::Endo) {
                bail!("{loc}: observation_trends: '{name}' is not an endogenous variable");
            }
            let expr = &toks[2..toks.len() - 1];
            slice_expr(expr)?;
            self.x.observation_trends.push((name, raw_join_spaced(expr)));
        }
        self.c.i += 2;
        Ok(())
    }

    /// `homotopy_setup; NAME, END_VALUE; NAME, START_VALUE, END_VALUE; ... end;`
    fn x_homotopy_setup(&mut self) -> Result<()> {
        self.c.i += 1;
        let ho = self.header_options()?;
        self.x.homotopy_block = true;
        // options_.homotopy_from_initval_to_endval, written by the preprocessor at the block
        self.x.homotopy_from_initval_to_endval =
            ho.iter().any(|(k, v)| k == "from_initval_to_endval" && !matches!(v, OptValue::Num(x) if *x == 0.0));
        while !self.c.at_end() {
            if self.c.peek().is_none() {
                bail!("{}: block 'homotopy_setup' not closed by 'end;'", self.c.loc());
            }
            let toks = self.c.statement()?;
            let loc = toks.first().map(|t| t.loc.to_string()).unwrap_or_default();
            let fields = split_commas(toks);
            let name = match fields.first() {
                Some([t]) if t.ident().is_some() => t.raw.clone(),
                _ => bail!("{loc}: homotopy_setup: expected NAME, [START_VALUE,] END_VALUE;"),
            };
            if self.b.symbol(&name).is_none_or(|s| !matches!(s.kind, SymbolKind::Param | SymbolKind::Exo)) {
                bail!("{loc}: homotopy_setup: '{name}' is not a parameter or an exogenous variable");
            }
            let (start, end) = match &fields[1..] {
                [e] => (None, self.eval_tokens(e)?),
                [s, e] => (Some(self.eval_tokens(s)?), self.eval_tokens(e)?),
                _ => bail!("{loc}: homotopy_setup: expected NAME, [START_VALUE,] END_VALUE;"),
            };
            self.x.homotopy_setup.push((name, start, end));
        }
        self.c.i += 2;
        Ok(())
    }

    /// `load_params_and_steady_state(FILENAME);`: as Dynare's preprocessor, reads the file
    /// now; parameters are calibrated, endogenous and exogenous values go to `initval`.
    fn x_load_params_and_steady_state(&mut self) -> Result<()> {
        let at = self.c.peek().map(|t| t.loc.clone());
        self.c.i += 1;
        let toks = self.c.statement()?;
        let inner = if toks.first().is_some_and(|t| t.is("(")) && toks.last().is_some_and(|t| t.is(")")) {
            &toks[1..toks.len() - 1]
        } else {
            toks
        };
        let file = match inner {
            [t] => match &t.tok {
                Tok::Str(s) => s.clone(),
                _ => t.raw.clone(),
            },
            _ => raw_join(inner),
        };
        let mut path = std::path::PathBuf::from(&file);
        if path.is_relative()
            && let Some(dir) =
                at.as_ref().and_then(|l| std::path::Path::new(&*l.file).parent().map(|p| p.to_path_buf()))
            && dir.join(&path).exists()
        {
            path = dir.join(path);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("load_params_and_steady_state: cannot read {}", path.display()))?;
        let (mut endo, mut exo) = (Vec::new(), Vec::new());
        for (ln, line) in text.lines().enumerate() {
            let mut it = line.split_whitespace();
            let (Some(name), Some(v)) = (it.next(), it.next()) else { continue };
            let v: f64 = v
                .parse()
                .map_err(|_| anyhow!("{}:{}: load_params_and_steady_state: bad value '{v}'", path.display(), ln + 1))?;
            match self.b.symbol(name).map(|s| (s.kind, s.index)) {
                Some((SymbolKind::Param, k)) => {
                    self.x_param_assignment(k, name, v);
                    self.b.params[k] = v;
                }
                Some((SymbolKind::Endo, k)) => {
                    self.initval.retain(|(i, _)| *i != k);
                    self.initval.push((k, v));
                    endo.push((k, v));
                }
                Some((SymbolKind::Exo, k)) => {
                    self.initval_exo.retain(|(i, _)| *i != k);
                    self.initval_exo.push((k, v));
                    exo.push((k, v));
                }
                _ => {} // unknown names are skipped, as by the preprocessor
            }
        }
        // oo_.steady_state / oo_.exo_steady_state are set where the call stands
        self.state_blocks.push((self.commands.len(), super::RawStateBlock::Initval { endo, exo, sources: Vec::new() }));
        Ok(())
    }
}

/// Tokens joined by blanks (keeps `a - -b` and names apart when re-parsed).
pub(super) fn raw_join_spaced(toks: &[Token]) -> String {
    toks.iter().map(|t| t.raw.as_str()).collect::<Vec<_>>().join(" ")
}
