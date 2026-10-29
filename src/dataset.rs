// SPDX-License-Identifier: GPL-3.0-or-later
//! The estimation sample: port of `utilities/dataset/makedataset.m` (datafile
//! or `data` command, `first_obs` / `last_obs` / `nobs`, `set_time`) over a dseries-like
//! [`Table`] whose first column may hold Dynare dates.
//!
//! As `dseries(datafile)`, only the first date of the file matters: the dataset's dates are
//! `init, init+1, ...` (`load_csv_file_data.m` / `load_xls_file_data.m` read `init` from the
//! first cell). A csv with ISO dates (`2003-03-31, 2003-06-30, ...`) is therefore a *daily*
//! dataset in Dynare, whose observations are labelled 2003-03-31, 2003-04-01, ...:
//! [`Dataset::init`] is that (Dynare's) date of the first sample row, used wherever Dynare
//! shows dates (`options_.initial_date`, decomposition `period` columns, figure titles),
//! while the `date` columns of dynare-rust's CSV outputs keep the file's own labels
//! ([`Dataset::file_dates`]).
//!
//! `data(file=..., first_obs=..., last_obs=..., nobs=...)` and `set_time(DATE)` set
//! `options_.dataset` / `options_.initial_period`, which are global in Dynare: the driver
//! records them with [`data_command`] / [`set_time`] as it meets them (per thread, i.e. per
//! run), and every later command that reads data sees them.

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};

use crate::data::{DataOptions, Table};
use crate::dates::{Date, DateOrIndex, Freq};
use crate::linalg::Mat;
use crate::modfile::{ModFile, OptValue, Options, opt};

/// `options_.dataset` (the `data` command).
#[derive(Clone, Debug, Default)]
pub struct DataCommand {
    pub file: Option<String>,
    pub series: Option<String>,
    pub first_obs: Option<DateOrIndex>,
    pub last_obs: Option<DateOrIndex>,
    pub nobs: Option<usize>,
    pub xls_sheet: Option<String>,
    pub xls_range: Option<String>,
}

#[derive(Default)]
struct Globals {
    dataset: Option<DataCommand>,
    /// `options_.initial_period` (`set_time`).
    initial_period: Option<Date>,
    /// `dataset_.init` of the last dataset built (Dynare's global `dataset_`).
    last_init: Option<Date>,
}

thread_local! {
    static GLOBALS: RefCell<Globals> = RefCell::new(Globals::default());
}

/// Clears `options_.dataset` and `options_.initial_period` (start of a run).
pub fn reset() {
    GLOBALS.with(|g| *g.borrow_mut() = Globals::default());
}

/// The `data(...)` command: sets `options_.dataset`.
pub fn data_command(o: &Options) -> Result<()> {
    let s = |k: &str| match opt(o, k) {
        Some(OptValue::Str(v)) => Some(v.clone()),
        Some(OptValue::Num(v)) => Some(format!("{v}")),
        _ => None,
    };
    let d = DataCommand {
        file: s("file"),
        series: s("series"),
        first_obs: crate::dates::option(o, "first_obs")?,
        last_obs: crate::dates::option(o, "last_obs")?,
        nobs: match opt(o, "nobs") {
            Some(OptValue::Num(v)) => Some(*v as usize),
            Some(OptValue::Vec(v)) if v.len() == 1 => Some(v[0] as usize),
            None => None,
            Some(_) => bail!("data: nobs = [INTEGER1:INTEGER2] (recursive estimation) is not supported by dynare-rust"),
        },
        xls_sheet: s("xls_sheet"),
        xls_range: s("xls_range"),
    };
    if d.file.is_none() && d.series.is_none() {
        bail!("data: one of the options file or series is required");
    }
    if d.series.is_some() {
        bail!("data(series=...): dynare-rust has no MATLAB workspace holding dseries objects; use data(file=...)");
    }
    GLOBALS.with(|g| g.borrow_mut().dataset = Some(d));
    Ok(())
}

/// `set_time(DATE)`: `options_.initial_period`.
pub fn set_time(o: &Options) -> Result<()> {
    // the parser keeps `set_time(1990Q1)` as an option key without a value
    let raw = o.iter().find_map(|(k, v)| match v {
        OptValue::Flag => Some(k.clone()),
        OptValue::Str(s) => Some(s.clone()),
        _ => None,
    });
    let d = raw
        .as_deref()
        .and_then(|r| Date::parse_mod(&r.to_uppercase()))
        .ok_or_else(|| anyhow!("set_time: expected a date, e.g. set_time(1990Q1)"))?;
    GLOBALS.with(|g| g.borrow_mut().initial_period = Some(d));
    Ok(())
}

/// The current `options_.dataset`.
pub fn current_data_command() -> Option<DataCommand> {
    GLOBALS.with(|g| g.borrow().dataset.clone())
}

/// `dataset_.init` of the last sample selected by [`Dataset::load`] (Dynare's global
/// `dataset_`, from which `evaluate_smoother` returns `initial_date`).
pub fn last_sample_init() -> Option<Date> {
    GLOBALS.with(|g| g.borrow().last_init)
}

/// This thread's dataset globals (`options_.dataset`, `options_.initial_period`,
/// `dataset_.init`), to hand them over to another thread (see `driver::run`).
pub(crate) type GlobalsSnapshot = (Option<DataCommand>, Option<Date>, Option<Date>);

pub(crate) fn export_globals() -> GlobalsSnapshot {
    GLOBALS.with(|g| {
        let g = g.borrow();
        (g.dataset.clone(), g.initial_period, g.last_init)
    })
}

/// Installs globals copied by [`export_globals`] on this thread.
pub(crate) fn import_globals((dataset, initial_period, last_init): GlobalsSnapshot) {
    GLOBALS.with(|g| *g.borrow_mut() = Globals { dataset, initial_period, last_init });
}

/// The current `options_.initial_period`.
pub fn initial_period() -> Option<Date> {
    GLOBALS.with(|g| g.borrow().initial_period)
}

/// `dseries(datafile).init`: the first date label of the file (`isdate`; an integer is a
/// year), or None when the file has no dates (Dynare: `dates(1,1)` = `1Y`).
pub fn table_init(t: &Table) -> Option<Date> {
    let first = t.dates.as_ref()?.first()?.trim().to_string();
    if let Ok(y) = first.parse::<i64>() {
        return Some(Date::from_parts(Freq::Annual, y, 1));
    }
    Date::parse(&first)
}

/// What makedataset.m selects.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub path: PathBuf,
    pub table: Table,
    /// 1-based first row of the table in the sample (`options_.first_obs` as an observation number).
    pub first_obs: usize,
    pub nobs: usize,
    /// Dynare's date of the first sample row (`dataset_.init` after the selection).
    pub init: Date,
    /// True when the file had no dates (Dynare's default `1Y` time base).
    pub default_dates: bool,
}

impl Dataset {
    /// Port of makedataset.m (without the log / prefilter steps, done by the callers):
    /// the `datafile` option of the command or, without it, the `data` command.
    pub fn load(mf: &ModFile, dir: &Path, o: &Options) -> Result<Dataset> {
        let _ = mf;
        let datafile = match opt(o, "datafile") {
            Some(OptValue::Str(s)) => Some(s.clone()),
            _ => None,
        };
        let dc = current_data_command();
        let (path, new_interface, dopts) = match (&datafile, &dc) {
            (Some(_), Some(_)) => {
                bail!(
                    "makedataset: You cannot simultaneously use the data command and the datafile option (in the estimation command)!"
                )
            }
            (Some(f), None) => (crate::driver::resolve_datafile(dir, f)?, false, xls_options(o)),
            (None, Some(d)) => {
                let f = d.file.as_deref().ok_or_else(|| anyhow!("makedataset: You have to specify the datafile!"))?;
                (
                    crate::driver::resolve_datafile(dir, f)?,
                    true,
                    DataOptions { xls_sheet: d.xls_sheet.clone(), xls_range: d.xls_range.clone() },
                )
            }
            (None, None) => bail!("makedataset: datafile option is missing!"),
        };
        let table = Table::read(&path, &dopts)?;
        if (1..table.names.len()).any(|k| table.names[..k].contains(&table.names[k])) {
            bail!(
                "makedataset: the data set must not contain two variables with the same name and must not contain empty/non-named columns."
            );
        }
        let n0 = table.values.nrows() as i64;
        // dataset_.init and options_.initial_period
        let file_init = table_init(&table);
        let default_dates = file_init.is_none();
        let mut init = file_init.unwrap_or(Date::from_parts(Freq::Annual, 1, 1));
        if let Some(ip) = initial_period() {
            if default_dates {
                init = ip;
            } else if ip.partial_cmp(&init).is_none_or(|c| c.is_lt()) {
                bail!(
                    "makedataset: The date as defined by the set_time command is not consistent with the initial period in the database!"
                );
            }
        }
        let last = init.add(n0 - 1);
        // position (0-based) of a date in the dataset
        let pos = |d: Date, what: &str| -> Result<i64> {
            init.diff(d)
                .ok_or_else(|| anyhow!("makedataset: {what} ({d}) and the dataset ({init}) have different frequencies"))
        };
        let (first, nobs): (i64, i64) = if new_interface {
            let d = dc.unwrap();
            let firstobs = match d.first_obs {
                None => 0,
                Some(DateOrIndex::Date(x)) => pos(x, "first_obs")?,
                Some(DateOrIndex::Index(k)) => k - 1,
            };
            let lastobs_opt = match d.last_obs {
                None => None,
                Some(DateOrIndex::Date(x)) => Some(pos(x, "last_obs")?),
                Some(DateOrIndex::Index(k)) => Some(k - 1),
            };
            let (lastobs, nobs) = match (d.nobs, lastobs_opt) {
                (None, None) => (n0 - 1, n0),
                (None, Some(l)) => (l, l - firstobs + 1),
                (Some(n), None) => (firstobs + n as i64 - 1, n as i64),
                (Some(n), Some(l)) => {
                    if l != firstobs + n as i64 - 1 {
                        bail!(
                            "makedataset: Options last_obs ({}), first_obs ({}) and nobs ({n}) are not consistent!",
                            init.add(l),
                            init.add(firstobs)
                        );
                    }
                    (l, n as i64)
                }
            };
            if firstobs < 0 {
                bail!(
                    "makedataset: first_obs ({}) cannot be less than the first date in the dataset ({init})!",
                    init.add(firstobs)
                );
            }
            if lastobs > n0 - 1 {
                bail!(
                    "makedataset: last_obs ({}) cannot be greater than the last date in the dataset ({last})!",
                    init.add(lastobs)
                );
            }
            (firstobs, nobs)
        } else {
            let firstobs = match crate::dates::option(o, "first_obs")? {
                None => 0,
                Some(DateOrIndex::Index(k)) => {
                    if k < 1 || k > n0 {
                        bail!("makedataset: first_obs = {k} is outside the data (1..={n0})");
                    }
                    k - 1
                }
                Some(DateOrIndex::Date(x)) => {
                    bail!("estimation: first_obs must be an integer (got the date {x}); use the data command for dates")
                }
            };
            let nobs = match opt(o, "nobs") {
                None => n0 - firstobs,
                Some(OptValue::Num(v)) => *v as i64,
                // nobs = [76]: one sample size is an ordinary estimation
                Some(OptValue::Vec(v)) if v.len() == 1 => v[0] as i64,
                Some(_) => bail!("nobs = [INTEGER1:INTEGER2] (recursive estimation) is not supported by dynare-rust"),
            };
            if firstobs > 0 {
                if firstobs + nobs > n0 {
                    bail!(
                        "makedataset: Given first_obs={} and {n0} total observations in the dataset, the current nobs of {nobs} must not be greater than {}!",
                        firstobs + 1,
                        n0 - firstobs
                    );
                }
            } else if nobs > n0 {
                bail!("makedataset: nobs ({nobs}) cannot be greater than the last date in the dataset ({n0})!");
            }
            (firstobs, nobs)
        };
        if nobs < 1 {
            bail!("makedataset: empty sample (first_obs {}, nobs {nobs})", first + 1);
        }
        let init = init.add(first);
        GLOBALS.with(|g| g.borrow_mut().last_init = Some(init));
        Ok(Dataset { path, table, first_obs: first as usize + 1, nobs: nobs as usize, init, default_dates })
    }

    /// nobs x nvobs, columns in `varobs` order (`dataset_{options_.varobs{:}}`).
    pub fn sample(&self, varobs: &[String]) -> Result<Mat> {
        self.table.sample(varobs, self.first_obs, Some(self.nobs))
    }

    /// The file's own labels of the sample rows.
    pub fn file_dates(&self) -> Option<Vec<String>> {
        self.table.dates.as_ref().map(|d| d[self.first_obs - 1..self.first_obs - 1 + self.nobs].to_vec())
    }

    /// Dynare's dates of the sample rows.
    pub fn dynare_dates(&self) -> Vec<Date> {
        crate::dates::dates_from(self.init, self.nobs)
    }
}

fn xls_options(o: &Options) -> DataOptions {
    DataOptions {
        xls_sheet: opt(o, "xls_sheet").map(|v| match v {
            OptValue::Num(k) => format!("{k}"),
            OptValue::Str(s) => s.clone(),
            _ => String::new(),
        }),
        xls_range: match opt(o, "xls_range") {
            Some(OptValue::Str(s)) => Some(s.clone()),
            _ => None,
        },
    }
}

/// Dynare's date of the first observation of a sample that starts at row `first_obs`
/// (1-based) of a table with file labels `dates` (None: no dates, Dynare's `1Y` base,
/// shifted by `set_time`).
pub fn sample_init(table_first_label: Option<&str>, first_obs: usize) -> Date {
    let base = table_first_label
        .and_then(|l| {
            l.trim().parse::<i64>().ok().map(|y| Date::from_parts(Freq::Annual, y, 1)).or_else(|| Date::parse(l))
        })
        .or_else(initial_period)
        .unwrap_or(Date::from_parts(Freq::Annual, 1, 1));
    base.add(first_obs as i64 - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daily_iso_dataset() {
        let t = Table::parse_csv(",a\n2003-03-31,1\n2003-06-30,2\n2003-09-30,3\n2003-12-31,4\n2004-03-31,5\n").unwrap();
        let init = table_init(&t).unwrap();
        assert_eq!(init.freq, Freq::Daily);
        // first_obs = 5: Dynare's date is init + 4 days, not the file's 2004-03-31
        assert_eq!(init.add(4).label(), "2003-04-04");
        assert_eq!(sample_init(Some("2003-03-31"), 5).label(), "2003-04-04");
        assert_eq!(sample_init(Some("1990Q1"), 5).label(), "1991Q1");
        assert_eq!(sample_init(None, 3).label(), "3Y");
    }
}
