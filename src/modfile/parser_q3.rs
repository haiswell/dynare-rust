// SPDX-License-Identifier: GPL-3.0-or-later
//! Estimation-related declarations: the `heteroskedastic_shocks` block
//! (`M_.heteroskedastic_shocks.Qvalue_orig` / `Qscale_orig`), the `filter_initial_state`
//! block (`M_.filter_initial_state`), `skew` entries of `shocks` blocks (`M_.Skew_e`) and
//! `dsge_prior_weight` in `estimated_params` (declared as a parameter by the preprocessor).

use super::{Parser, split_commas, split_value_items};
use crate::model::SymbolKind;
use crate::modfile::lexer::Tok;
use anyhow::{Result, anyhow, bail};

/// One `var e; periods ...; values|scales ...;` group of `heteroskedastic_shocks`.
#[derive(Clone, Debug, PartialEq)]
pub struct HeteroShock {
    pub exo: usize,
    /// Integer periods of the original dataset (1-based), one entry per value.
    pub periods: Vec<i64>,
    /// Or dates (`periods 2000Y:2020Y;`), placed on the estimation sample's dates.
    pub dates: Vec<crate::dates::Date>,
    /// `values` (standard deviations) when true, `scales` otherwise.
    pub is_value: bool,
    pub value: f64,
}

/// Parsed declarations (`ModFile::q3`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Q3Decl {
    /// `heteroskedastic_shocks` entries, file order (`overwrite` clears the earlier ones).
    pub heteroskedastic_shocks: Vec<HeteroShock>,
    /// `filter_initial_state`: (endogenous index, lag <= 0, value at the calibration, expression
    /// as written: Dynare evaluates it with the current parameters at every filter run).
    pub filter_initial_state: Vec<(usize, i32, f64, String)>,
    /// `M_.Skew_e`: (i, j, k, value), exogenous indices sorted.
    pub skew_e: Vec<(usize, usize, usize, f64)>,
}

impl<'a> Parser<'a> {
    pub(super) fn q3_handles(&self, kw: &str) -> bool {
        matches!(kw, "heteroskedastic_shocks" | "filter_initial_state" | "set_shock_skew_value")
    }

    pub(super) fn q3_statement(&mut self, kw: &str) -> Result<()> {
        match kw {
            "heteroskedastic_shocks" => self.q3_heteroskedastic_shocks(),
            "filter_initial_state" => self.q3_filter_initial_state(),
            "set_shock_skew_value" => self.q3_set_shock_skew_value(),
            _ => unreachable!(),
        }
    }

    fn q3_exo(&self, name: &str, loc: &crate::modfile::lexer::Loc) -> Result<usize> {
        match self.b.symbol(name) {
            Some(s) if s.kind == SymbolKind::Exo => Ok(s.index),
            _ => bail!("{loc}: '{name}' is not an exogenous variable"),
        }
    }

    /// `heteroskedastic_shocks[(overwrite)]; var e; periods 86:87, 89:97; scales 0.5, 0; ... end;`
    fn q3_heteroskedastic_shocks(&mut self) -> Result<()> {
        self.c.i += 1;
        let opts = self.header_options()?;
        if opts.iter().any(|(k, _)| k == "overwrite") {
            self.q3.heteroskedastic_shocks.clear();
        }
        let mut exo: Option<usize> = None;
        let mut periods: Vec<(Vec<i64>, Vec<crate::dates::Date>)> = Vec::new();
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                return Ok(());
            }
            if self.c.peek().is_none() {
                bail!("{}: block 'heteroskedastic_shocks' not closed by 'end;'", self.c.loc());
            }
            let t = self.c.next()?;
            let loc = t.loc.clone();
            let kw = t.ident().unwrap_or("").to_string();
            let toks = self.c.statement()?;
            match kw.as_str() {
                "var" => {
                    let name =
                        toks.first().and_then(|t| t.ident()).ok_or_else(|| anyhow!("{loc}: var: expected a shock"))?;
                    exo = Some(self.q3_exo(name, &loc)?);
                    periods.clear();
                }
                "periods" => {
                    periods = super::shk::sim_period_list(toks)
                        .map_err(|e| anyhow!("{loc}: heteroskedastic_shocks: {e}"))?
                        .iter()
                        .map(|g| g.split())
                        .collect();
                }
                "values" | "scales" => {
                    let e = exo.ok_or_else(|| anyhow!("{loc}: '{kw}' before 'var'"))?;
                    let vals: Vec<f64> =
                        split_value_items(toks).into_iter().map(|it| self.eval_tokens(it)).collect::<Result<_>>()?;
                    if vals.len() != periods.len() {
                        bail!("{loc}: heteroskedastic_shocks: {} period groups but {} {kw}", periods.len(), vals.len());
                    }
                    for ((p, d), v) in periods.iter().zip(vals) {
                        self.q3.heteroskedastic_shocks.push(HeteroShock {
                            exo: e,
                            periods: p.clone(),
                            dates: d.clone(),
                            is_value: kw == "values",
                            value: v,
                        });
                    }
                }
                _ => bail!("{loc}: heteroskedastic_shocks: unexpected '{kw}'"),
            }
        }
    }

    /// `filter_initial_state; x(0) = 1; y(-1) = 0.5; end;` (`M_.filter_initial_state`).
    fn q3_filter_initial_state(&mut self) -> Result<()> {
        self.c.i += 1;
        self.header_options()?;
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                return Ok(());
            }
            if self.c.peek().is_none() {
                bail!("{}: block 'filter_initial_state' not closed by 'end;'", self.c.loc());
            }
            let toks = self.c.statement()?;
            let loc = toks.first().map(|t| t.loc.to_string()).unwrap_or_default();
            let eq = toks
                .iter()
                .position(|t| t.is("="))
                .ok_or_else(|| anyhow!("{loc}: filter_initial_state: expected VAR(LAG) = VALUE"))?;
            let name = toks[0].ident().ok_or_else(|| anyhow!("{loc}: expected a variable"))?;
            let lag = match &toks[1..eq] {
                [] => 0,
                [o, n, c] if o.is("(") && c.is(")") => match n.tok {
                    Tok::Num(v) => v as i32,
                    _ => bail!("{loc}: expected an integer lag"),
                },
                [o, m, n, c] if o.is("(") && m.is("-") && c.is(")") => match n.tok {
                    Tok::Num(v) => -(v as i32),
                    _ => bail!("{loc}: expected an integer lag"),
                },
                _ => bail!("{loc}: filter_initial_state: expected VAR(LAG) = VALUE"),
            };
            if lag > 0 {
                bail!("{loc}: filter_initial_state: leads are not allowed");
            }
            let s = self.b.symbol(name).ok_or_else(|| anyhow!("{loc}: unknown variable '{name}'"))?;
            if s.kind != SymbolKind::Endo {
                bail!("{loc}: filter_initial_state: '{name}' is not an endogenous variable");
            }
            // ParsingDriver::filter_initial_state_element: x(lag) is the state of x(lag-1)
            if self.b.min_lag(name) > lag - 1 {
                bail!(
                    "{}: filter_initial_state: variable {name} does not appear in the model with the lag {} (see the reference manual for the timing convention in 'filter_initial_state')",
                    super::span_text(&toks[0], &self.c.t[self.c.i - 1]),
                    lag - 1
                );
            }
            let idx = s.index;
            let v = self.eval_tokens(&toks[eq + 1..])?;
            let src = toks[eq + 1..].iter().map(|t| t.raw.as_str()).collect::<String>();
            self.q3.filter_initial_state.push((idx, lag, v, src));
        }
    }

    /// `set_shock_skew_value('e', v);` / `set_shock_skew_value('u', 'v', 'w', c);`
    /// (accessors/set_shock_skew_value.m), applied when it is parsed (as stream X does for
    /// `set_param_value`).
    fn q3_set_shock_skew_value(&mut self) -> Result<()> {
        self.c.i += 1;
        let toks = self.c.statement()?;
        let Some(first) = toks.first() else { bail!("{}: set_shock_skew_value: expected arguments", self.c.loc()) };
        let loc = first.loc.clone();
        if !(toks.len() >= 2 && toks[0].is("(") && toks[toks.len() - 1].is(")")) {
            bail!("{loc}: set_shock_skew_value: expected ('NAME', VALUE) or ('N1', 'N2', 'N3', VALUE)");
        }
        let fields = split_commas(&toks[1..toks.len() - 1]);
        let Some((value, names)) = fields.split_last() else {
            bail!("{loc}: set_shock_skew_value: expected arguments")
        };
        let mut idx = Vec::new();
        for f in names {
            let [t] = f else { bail!("{loc}: set_shock_skew_value: expected shock names") };
            let n = match &t.tok {
                Tok::Str(s) | Tok::Ident(s) => s.clone(),
                _ => bail!("{loc}: set_shock_skew_value: expected a shock name"),
            };
            idx.push(self.q3_exo(&n, &loc)?);
        }
        let v = self.eval_tokens(value)?;
        let (i, j, k) = match idx[..] {
            [i] => (i, i, i),
            [a, b, c] => {
                let mut s = [a, b, c];
                s.sort_unstable();
                (s[0], s[1], s[2])
            }
            _ => bail!("{loc}: set_shock_skew_value: expected one or three shocks"),
        };
        self.q3.skew_e.retain(|e| (e.0, e.1, e.2) != (i, j, k));
        if v != 0.0 {
            self.q3.skew_e.push((i, j, k, v));
        }
        Ok(())
    }

    /// `skew e = v;` / `skew u, v, w = c;` inside a `shocks` block (cursor after `skew`).
    pub(super) fn q3_shocks_skew(&mut self) -> Result<()> {
        let toks = self.c.statement()?;
        let loc = toks.first().map(|t| t.loc.clone());
        let eq = toks
            .iter()
            .position(|t| t.is("="))
            .ok_or_else(|| anyhow!("{}: skew: expected '='", loc.clone().map(|l| l.to_string()).unwrap_or_default()))?;
        let mut idx = Vec::new();
        for t in &toks[..eq] {
            if let Some(n) = t.ident() {
                let k = self.q3_exo(n, &t.loc)?;
                if self.exo_det.contains(&k) {
                    // ShocksStatement::checkPass
                    bail!(
                        "shocks: setting skewness for '{n}' is not allowed; skewness can only be specified for exogenous variables"
                    );
                }
                idx.push(k);
            }
        }
        let v = self.eval_tokens(&toks[eq + 1..])?;
        let (i, j, k) = match idx[..] {
            [i] => (i, i, i),
            [a, b, c] => {
                let mut s = [a, b, c];
                s.sort_unstable();
                (s[0], s[1], s[2])
            }
            _ => bail!("{}: skew: expected one or three shocks", loc.map(|l| l.to_string()).unwrap_or_default()),
        };
        self.q3.skew_e.retain(|e| (e.0, e.1, e.2) != (i, j, k));
        if v != 0.0 {
            self.q3.skew_e.push((i, j, k, v));
        }
        Ok(())
    }

    /// `dsge_prior_weight` in `estimated_params`: declared as a parameter when it is not yet
    /// (the preprocessor's `declare_dsge_prior_weight`). Returns its parameter index.
    pub(super) fn q3_dsge_prior_weight(&mut self) -> Result<usize> {
        if let Some(s) = self.b.symbol("dsge_prior_weight") {
            if s.kind != SymbolKind::Param {
                bail!("dsge_prior_weight may only be used as a parameter");
            }
            return Ok(s.index);
        }
        self.b.declare("dsge_prior_weight", SymbolKind::Param).map_err(|e| anyhow!("dsge_prior_weight: {e}"))?;
        self.labels.push(SymbolKind::Param, "dsge\\_prior\\_weight".into(), "dsge_prior_weight".into());
        Ok(self.b.symbol("dsge_prior_weight").map(|s| s.index).unwrap())
    }
}
