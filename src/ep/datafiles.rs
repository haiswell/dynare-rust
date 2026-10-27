// SPDX-License-Identifier: GPL-3.0-or-later
//! `initval_file` and `histval_file`: port of `histvalf_initvalf.m` and `histvalf.m` for
//! `.csv`, `.xls`/`.xlsx` and `.mat` datafiles (`data::Table`), with integer `first_obs`,
//! `last_obs`, `nobs`, `first_simulation_period` and `last_simulation_period`. The series
//! of the auxiliary variables are built as `<fname>.dynamic_set_auxiliary_series` does
//! (shifted copies of the original series, NaN outside the sample).
//!
//! `initval_file` sets `oo_.initval_series` (`DetState::initval_series`, used by
//! `perfect_foresight_setup` as `make_y_.m` / `make_ex_.m` do) and `options_.periods`;
//! `histval_file` sets `M_.endo_histval` / `M_.exo_histval` (`DetState::histval_file`).

use std::path::Path;

use anyhow::{Result, anyhow, bail};

use crate::data::{DataOptions, Table};
use crate::deterministic::DetState;
use crate::linalg::Mat;
use crate::model::Model;
use crate::modfile::{ModFile, OptValue, Options, opt};

/// A data set over all endogenous (incl. auxiliary) and exogenous variables.
#[derive(Clone, Debug)]
pub struct Series {
    /// n_endo x T.
    pub endo: Mat,
    /// T x n_exo.
    pub exo: Mat,
}

fn num(o: &Options, k: &str) -> Result<Option<i64>> {
    match opt(o, k) {
        None => Ok(None),
        Some(OptValue::Num(v)) => Ok(Some(*v as i64)),
        Some(_) => bail!("{k}: expected an integer"),
    }
}

// ---- DATE values of first_obs, last_obs, first/last_simulation_period ----
/// An observation number (`Ok(Some((k, false)))`) or a date, as its 1-based position in the
/// series whose first date is `init` (`Ok(Some((k, true)))`).
fn obs(o: &Options, k: &str, init: crate::dates::Date) -> Result<Option<(i64, bool)>> {
    Ok(match crate::dates::option(o, k)? {
        None => None,
        Some(crate::dates::DateOrIndex::Index(i)) => Some((i, false)),
        Some(crate::dates::DateOrIndex::Date(d)) => {
            let p =
                init.diff(d).ok_or_else(|| anyhow!("{k} = {d}: the frequency differs from the datafile's ({init})"))?;
            Some((p + 1, true))
        }
    })
}

/// The series of the auxiliary variables (`dynamic_set_auxiliary_series`).
fn set_auxiliary_series(model: &Model, endo: &mut Mat, exo: &Mat) {
    let t = endo.ncols();
    let shift = |src: &dyn Fn(i64) -> f64, l: i64| -> Vec<f64> { (0..t as i64).map(|s| src(s + l)).collect() };
    for a in &model.aux_vars {
        let vals: Vec<f64> = match (a.ty, a.orig_index, a.orig_lead_lag) {
            (1, Some(o), Some(l)) => {
                let e = endo.clone();
                shift(&|s| if s >= 0 && (s as usize) < t { e[(o, s as usize)] } else { f64::NAN }, l as i64)
            }
            (2 | 3, Some(o), Some(l)) => {
                shift(&|s| if s >= 0 && (s as usize) < t { exo[(s as usize, o)] } else { f64::NAN }, l as i64)
            }
            _ => {
                // lead aux variables: orig_expr is `name(k)` (or `name` for an exogenous lead)
                let e = a.orig_expr.trim();
                let parsed = if e.chars().all(|ch| ch.is_alphanumeric() || ch == '_') {
                    Some((e.to_string(), 0))
                } else {
                    e.strip_suffix(')').and_then(|s| s.split_once('(')).and_then(|(n, k)| {
                        let k: i64 = k.trim().parse().ok()?;
                        Some((n.trim().to_string(), k))
                    })
                };
                match parsed {
                    Some((n, k)) if model.endo_index(&n).is_some() => {
                        let j = model.endo_index(&n).unwrap();
                        let e = endo.clone();
                        shift(&|s| if s >= 0 && (s as usize) < t { e[(j, s as usize)] } else { f64::NAN }, k)
                    }
                    Some((n, k)) if model.exo_index(&n).is_some() => {
                        let j = model.exo_index(&n).unwrap();
                        shift(&|s| if s >= 0 && (s as usize) < t { exo[(s as usize, j)] } else { f64::NAN }, k)
                    }
                    _ => {
                        println!(
                            "WARNING: cannot build the series of auxiliary variable {}",
                            model.endo_names[a.endo_index]
                        );
                        vec![f64::NAN; t]
                    }
                }
            }
        };
        for (s, v) in vals.into_iter().enumerate() {
            endo[(a.endo_index, s)] = v;
        }
    }
}

/// MATLAB's `error('<msg> (%d).', caller, n)` with one `%d` and two arguments: the format is
/// recycled, the char array `caller` giving one copy per character code, then `n` (the
/// messages of Dynare 7.2's `histval_file` range errors, reproduced verbatim).
fn matlab_recycled_format(msg: &str, caller: &str, n: i64) -> String {
    caller.bytes().map(i64::from).chain([n]).map(|v| format!("{msg} ({v}).")).collect()
}

/// Port of `histvalf_initvalf.m`: the (sliced) series and the number of simulation periods `p`.
pub fn histvalf_initvalf(caller: &str, mf: &ModFile, o: &Options, dir: &Path) -> Result<(Series, i64)> {
    let model = &mf.model;
    if opt(o, "series").is_some() {
        bail!(
            "{caller}_FILE: option series (a dseries in the MATLAB workspace) is not supported by dynare-rust; use datafile"
        );
    }
    let file = match (opt(o, "datafile"), opt(o, "filename")) {
        (Some(_), Some(_)) => {
            bail!("{caller}_FILE: you can't use option DATAFILE and option FILENAME at the same time")
        }
        (Some(OptValue::Str(s)), None) => s.clone(),
        (None, Some(OptValue::Str(s))) => {
            println!("Warning: {caller}_FILE: option FILENAME is deprecated, please use option DATAFILE");
            s.clone()
        }
        _ => bail!("{caller}_FILE: the datafile option is required"),
    };
    let mut path = if Path::new(&file).is_absolute() { Path::new(&file).to_path_buf() } else { dir.join(&file) };
    if path.extension().is_none() {
        let found = ["m", "mat", "xls", "xlsx", "csv"].iter().map(|e| path.with_extension(e)).find(|p| p.is_file());
        path = found.ok_or_else(|| anyhow!("{caller}_FILE: Can't find datafile: {file}.{{m,mat,xls,xlsx}}"))?;
    }
    let table = Table::read(&path, &DataOptions::default())?;
    let mut missing = false;
    for name in model.endo_names[..model.orig_endo_nbr].iter().chain(&model.exo_names) {
        if table.column(name).is_none() {
            println!("{caller}_FILE: variable {name} is missing");
            missing = true;
        }
    }
    if missing {
        bail!("{caller}_FILE: some variables are missing");
    }
    let t = table.values.nrows();
    let (n, nx) = (model.n_endo(), model.n_exo());
    let mut endo = Mat::from_element(n, t, f64::NAN);
    for j in 0..model.orig_endo_nbr {
        let c = table.column(&model.endo_names[j]).unwrap();
        for s in 0..t {
            endo[(j, s)] = table.values[(s, c)];
        }
    }
    let exo = Mat::from_fn(t, nx, |s, k| table.values[(s, table.column(&model.exo_names[k]).unwrap())]);
    set_auxiliary_series(model, &mut endo, &exo);
    // sample selection (1-based observation numbers)
    let (oml, omld) = (model.orig_maximum_lag as i64, model.orig_maximum_lead as i64);
    let nobs0 = t as i64;
    let nobs = num(o, "nobs")?.unwrap_or(0);
    // dates: positions in series.dates (dseries init = the file's first date)
    let init =
        crate::dataset::table_init(&table).unwrap_or(crate::dates::Date::from_parts(crate::dates::Freq::Annual, 1, 1));
    let fsp_d = obs(o, "first_simulation_period", init)?;
    let fsp = fsp_d.map(|(k, _)| k);
    let mut first_obs_present = false;
    let mut first_obs = 1;
    if let Some((f, true)) = obs(o, "first_obs", init)? {
        first_obs = f;
        if let Some(fs) = fsp
            && first_obs != fs - oml
        {
            bail!(
                "{caller}_FILE: first_obs = {} and first_simulation_period = {} have values inconsistent with a maximum lag of {oml} periods",
                init.add(f - 1),
                init.add(fs - 1)
            );
        }
        first_obs_present = true;
    } else if let Some(f) = num(o, "first_obs")? {
        if f < 1 {
            bail!("{caller}_FILE: first_obs must be a positive number");
        } else if f > nobs0 {
            bail!(
                "{caller}_FILE: first_obs = {f} is larger than the number of observations in the data file ({nobs0})"
            );
        }
        first_obs = f;
        if let Some(fs) = fsp
            && first_obs != fs - oml
        {
            bail!(
                "{caller}_FILE: first_obs = {f} and first_simulation_period = {fs} have values inconsistent with a maximum lag of {oml} periods"
            );
        }
        first_obs_present = true;
    }
    if !first_obs_present && let Some((fs, true)) = fsp_d {
        first_obs = fs - oml; // a date: no range check (histvalf_initvalf.m)
        first_obs_present = true;
    } else if !first_obs_present && let Some(fs) = fsp {
        if fs < oml {
            bail!("{caller}_FILE: first_simulation_period = {fs} must be larger than the maximum lag ({oml})");
        } else if fs > nobs0 {
            bail!(
                "{caller}_FILE: first_simulations_period = {fs} is larger than the number of observations in the data file ({nobs0})"
            );
        }
        first_obs = fs - oml;
        first_obs_present = true;
    }
    // Dynare 7.2: `assert(strcmp(caller, 'INITVAL'))` guards the NOBS branches, but the
    // preprocessor calls this function with 'INITVALF', so an initval_file with NOBS (other
    // than a consistent FIRST_OBS/LAST_OBS/NOBS triple) fails the assertion.
    let assert_initval = || -> Result<()> {
        if caller != "INITVAL" {
            bail!("Assertion failed.");
        }
        Ok(())
    };
    let histval = caller == "HISTVAL";
    let last_given = match obs(o, "last_obs", init)? {
        // a date
        Some((l, true)) => {
            if l > nobs0 {
                bail!(
                    "{caller}_FILE: last_obs = {} is larger than the number of observations in the dataset ({})",
                    init.add(l - 1),
                    init.add(nobs0 - 1)
                );
            }
            Some(l)
        }
        Some((l, false)) => {
            if l > nobs0 {
                bail!(
                    "{caller}_FILE: last_obs = {l} is larger than the number of observations in the dataset ({nobs0})"
                );
            }
            Some(l)
        }
        None => None,
    };
    let mut last_obs;
    if let Some(l) = last_given {
        last_obs = l;
        if first_obs_present {
            if nobs > 0 && last_obs != first_obs + nobs - 1 {
                assert_initval()?;
                bail!(
                    "{caller}_FILE: FIRST_OBS, LAST_OBS and NOBS contain inconsistent information. Use only two of these options."
                );
            } else if histval && last_obs != first_obs + oml - 1 {
                bail!(
                    "{caller}_FILE: the number of observations given by FIRST_OBS and LAST_OBS differs from the maximum number of lags in the model ({oml})."
                );
            }
        } else if nobs > 0 {
            assert_initval()?;
            first_obs = last_obs - nobs + 1;
        } else if histval {
            first_obs = last_obs - oml + 1;
            if first_obs < 1 {
                bail!(matlab_recycled_format(
                    "HISTVAL_FILE: the number of observations in the file before LAST_OBS is less than the maximum number of lags in the model",
                    caller,
                    oml
                ));
            }
        }
    } else if nobs > 0 {
        assert_initval()?;
        last_obs = first_obs + nobs - 1;
    } else if histval {
        last_obs = first_obs + oml - 1;
        if last_obs > nobs0 {
            bail!(matlab_recycled_format(
                "HISTVAL_FILE: the number of observations in the file after FIRST_OBS is less than the maximum number of lags in the model",
                caller,
                oml
            ));
        }
    } else {
        last_obs = nobs0;
    }
    let p = if let Some((lsp, _)) = obs(o, "last_simulation_period", init)? {
        if histval {
            bail!("Assertion failed."); // unreachable: a syntax error since Dynare 7.2
        }
        if lsp <= last_obs - omld {
            last_obs = lsp + omld;
        } else {
            bail!("{caller}_FILE: LAST_SIMULATION_PERIOD is too large compared to the available data.");
        }
        lsp - (first_obs + oml) + 1
    } else {
        (last_obs - omld) - (first_obs + oml) + 1
    };
    if first_obs < 1 || last_obs > nobs0 || last_obs < first_obs {
        bail!("{caller}_FILE: empty or invalid sample (observations {first_obs} to {last_obs} of {nobs0})");
    }
    let (a, b) = ((first_obs - 1) as usize, last_obs as usize);
    Ok((Series { endo: endo.columns(a, b - a).clone_owned(), exo: exo.rows(a, b - a).clone_owned() }, p))
}

/// The `initval_file` command.
pub fn initval_file(mf: &ModFile, det: &mut DetState, o: &Options, dir: &Path) -> Result<()> {
    let (s, p) = histvalf_initvalf("INITVALF", mf, o, dir)?; // the preprocessor's caller name
    det.initval_series = Some((s.endo, s.exo));
    det.simul.set_periods(p.max(0) as usize); // options_.periods
    Ok(())
}

/// The `histval_file` command (`histvalf.m`).
pub fn histval_file(mf: &ModFile, det: &mut DetState, o: &Options, dir: &Path) -> Result<()> {
    let model = &mf.model;
    let (s, _) = histvalf_initvalf("HISTVAL", mf, o, dir)?;
    let oml = model.orig_maximum_lag;
    if s.endo.ncols() != oml {
        bail!("histval_file: number of observations does not correspond to the number of lags in the model");
    }
    let ml = det.maximum_lag;
    let k = oml - ml; // 0-based first row kept
    let endo = s.endo.columns(k, ml).clone_owned();
    let exo = s.exo.rows(k, ml).transpose();
    det.histval_file = Some((endo, exo));
    Ok(())
}
