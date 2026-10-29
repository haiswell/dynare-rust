// SPDX-License-Identifier: GPL-3.0-or-later
//! Dynare's `dates` class (port of `dseries/src/@dates` and the string helpers in
//! `dseries/src/utilities/{is,convert,misc}`).
//!
//! A date is a frequency and an integer `time`, as in MATLAB:
//! - annual (`1`): `time` = year (`2004Y`, `2004A`);
//! - bi-annual (`2`), quarterly (`4`), monthly (`12`): `time` = `year*freq + subperiod`
//!   (`2004S1`/`2004H1`, `2004Q1`, `2004M3`);
//! - weekly (`52`, ISO weeks): `time` = `weeksince(year, week)` (`2004W3`);
//! - daily (`365`): `time` = MATLAB's `datenum` (`2004-03-31`).
//!
//! `char(d)` is [`Date::label`]; `d + k`, `d - k`, `d1 - d2`, comparisons and `d1:d2` are
//! [`Date::add`], [`Date::diff`], `Ord` (same frequency) and [`Date::range`].
//!
//! In a .mod file Dynare accepts the annual, bi-annual, quarterly and monthly literals,
//! optionally shifted (`2023Q1+6`); [`Date::parse_mod`] reads those (and also the other
//! forms, which the MATLAB `dates()` constructor accepts).

use std::cmp::Ordering;
use std::fmt;

use anyhow::{Result, anyhow, bail};

/// A frequency of Dynare's `dates` (`freq` member).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Freq {
    Annual,
    Biannual,
    Quarterly,
    Monthly,
    Weekly,
    Daily,
}

impl Freq {
    /// The integer `freq` (1, 2, 4, 12, 52, 365).
    pub fn number(self) -> i64 {
        match self {
            Freq::Annual => 1,
            Freq::Biannual => 2,
            Freq::Quarterly => 4,
            Freq::Monthly => 12,
            Freq::Weekly => 52,
            Freq::Daily => 365,
        }
    }
    pub fn from_number(f: i64) -> Option<Freq> {
        Some(match f {
            1 => Freq::Annual,
            2 => Freq::Biannual,
            4 => Freq::Quarterly,
            12 => Freq::Monthly,
            52 => Freq::Weekly,
            365 => Freq::Daily,
            _ => return None,
        })
    }
    /// `string2freq.m` (`'Y'`/`'A'`, `'H'`, `'Q'`, `'M'`, `'D'`, case-insensitive; also `'S'`, `'W'`).
    pub fn from_letter(s: &str) -> Option<Freq> {
        Some(match s.to_ascii_uppercase().as_str() {
            "Y" | "A" => Freq::Annual,
            "H" | "S" => Freq::Biannual,
            "Q" => Freq::Quarterly,
            "M" => Freq::Monthly,
            "W" => Freq::Weekly,
            "D" => Freq::Daily,
            _ => return None,
        })
    }
    /// `freq2string.m`.
    pub fn letter(self) -> &'static str {
        match self {
            Freq::Annual => "Y",
            Freq::Biannual => "S",
            Freq::Quarterly => "Q",
            Freq::Monthly => "M",
            Freq::Weekly => "W",
            Freq::Daily => "D",
        }
    }
}

/// One element of a Dynare `dates` object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Date {
    pub freq: Freq,
    pub time: i64,
}

impl PartialOrd for Date {
    /// Dates of different frequencies are not comparable (MATLAB errors).
    fn partial_cmp(&self, o: &Date) -> Option<Ordering> {
        (self.freq == o.freq).then(|| self.time.cmp(&o.time))
    }
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label())
    }
}

/// `islongyear.m`: ISO years with 53 weeks.
pub fn islongyear(y: i64) -> bool {
    let fix = |a: i64, b: i64| a / b; // MATLAB fix (truncation toward zero)
    (y + fix(y, 4) - fix(y, 100) + fix(y, 400)).rem_euclid(7) == 4
        || (y - 1 + fix(y - 1, 4) - fix(y - 1, 100) + fix(y - 1, 400)).rem_euclid(7) == 3
}

/// Weeks in one 400-year Gregorian cycle (the ISO calendar repeats every 400 years) and the
/// cumulated weeks before each year of the cycle (`isoyearweek.m`'s `weeksincycle`).
fn weeks_in_cycle() -> &'static [i64; 401] {
    static T: std::sync::OnceLock<[i64; 401]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0i64; 401];
        for y in 0..400 {
            t[y + 1] = t[y] + 52 + islongyear(y as i64) as i64;
        }
        t
    })
}

/// `longyearsince(y)` (dseries 7.2, base year 0): the number of long ISO years in `0..=y`
/// (0 for `y < 0`).
pub fn longyearsince(y: i64) -> i64 {
    if y < 0 {
        return 0;
    }
    let t = weeks_in_cycle();
    let per_cycle = t[400] - 400 * 52; // 71 long years per cycle
    let (q, r) = (y.div_euclid(400), y.rem_euclid(400) as usize);
    per_cycle * q + (t[r + 1] - 52 * (r as i64 + 1))
}

/// `weeksince.m` (base year 0): `52*y + #long years in 0..=y-1 + w`.
fn weeksince(y: i64, w: i64) -> i64 {
    52 * y + longyearsince(y - 1) + w
}

/// `isoyearweek.m` (dseries 7.2): the ISO (year, week) of the weekly `time` `n`. MATLAB
/// errors for `n < 1` (no weekly date before `0W1`); here the 400-year cycle is extended
/// backwards instead.
pub fn isoyearweek(n: i64) -> (i64, i64) {
    let t = weeks_in_cycle();
    let b = (n - 1).div_euclid(t[400]);
    let m = n - b * t[400];
    let j = t.iter().filter(|&&c| c < m).count() as i64 - 1;
    (400 * b + j, m - t[j as usize])
}

/// Days since 1970-01-01 of a proleptic Gregorian date.
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Inverse of [`days_from_civil`].
pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// MATLAB `datenum(1970,1,1)`.
const DATENUM_1970: i64 = 719529;

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn month_days(y: i64, m: i64) -> i64 {
    match m {
        2 if is_leap(y) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// `^-?[0-9]+<letters><digits>$` -> (year, subperiod); `sub_digits = false`: no subperiod.
fn split_date(s: &str, letters: &[char], sub: Option<&dyn Fn(&str) -> bool>) -> Option<(i64, i64)> {
    let (neg, body) = match s.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, s),
    };
    let k = body.find(|c: char| !c.is_ascii_digit())?;
    if k == 0 {
        return None;
    }
    let (ys, rest) = body.split_at(k);
    let mut ch = rest.chars();
    let l = ch.next()?;
    if !letters.contains(&l) {
        return None;
    }
    let tail = ch.as_str();
    let y: i64 = ys.parse().ok()?;
    let y = if neg { -y } else { y };
    match sub {
        None => tail.is_empty().then_some((y, 1)),
        Some(ok) => (ok(tail) && tail.chars().all(|c| c.is_ascii_digit())).then(|| (y, tail.parse().unwrap())),
    }
}

impl Date {
    pub fn new(freq: Freq, time: i64) -> Date {
        Date { freq, time }
    }
    /// `dates(FREQ, YEAR, SUBPERIOD)` (not daily).
    pub fn from_parts(freq: Freq, year: i64, sub: i64) -> Date {
        let time = match freq {
            Freq::Annual => year,
            Freq::Weekly => weeksince(year, sub),
            Freq::Daily => panic!("daily dates need year, month and day"),
            f => year * f.number() + sub,
        };
        Date { freq, time }
    }
    /// `dates('W', y, w)` with the checks of `@dates/dates.m` (dseries 7.2).
    pub fn weekly(y: i64, w: i64) -> Result<Date> {
        if y < 0 {
            bail!(
                "dates: Second input must contain nonnegative years (the weekly frequency does not extend to the years preceding the year zero)."
            );
        }
        if w < 1 {
            bail!("dates: Third input must contain strictly positive integers.");
        }
        if w > 52 && !islongyear(y) {
            bail!("dates: Third input specifies non existing ISO weeks (argument cannot be greater than 52).");
        }
        if w > 53 {
            bail!("dates: Third input specifies non existing ISO weeks (argument cannot be greater than 53).");
        }
        Ok(Date { freq: Freq::Weekly, time: weeksince(y, w) })
    }
    /// `dates('D', y, m, d)`.
    pub fn daily(y: i64, m: i64, d: i64) -> Date {
        Date { freq: Freq::Daily, time: days_from_civil(y, m, d) + DATENUM_1970 }
    }

    /// `string2date.m` on a string for which `isdate` holds (`isyearly`, `isbiannual`,
    /// `isquarterly`, `ismonthly`, `isweekly`, `isdaily`, tested in that order); None otherwise.
    pub fn parse(s: &str) -> Option<Date> {
        let s = s.trim();
        if let Some((y, _)) = split_date(s, &['Y', 'y', 'A', 'a'], None) {
            return Some(Date::from_parts(Freq::Annual, y, 1));
        }
        if let Some((y, p)) = split_date(s, &['H', 'h', 'S', 's'], Some(&|t| t == "1" || t == "2")) {
            return Some(Date::from_parts(Freq::Biannual, y, p));
        }
        if let Some((y, p)) = split_date(s, &['Q', 'q'], Some(&|t| matches!(t, "1" | "2" | "3" | "4"))) {
            return Some(Date::from_parts(Freq::Quarterly, y, p));
        }
        // ismonthly: [1-9] | 1[0-2] (also 0[1-9])
        let month = |t: &str| {
            matches!(t.len(), 1 | 2)
                && t.parse::<i64>().is_ok_and(|m| (1..=12).contains(&m))
                && !(t.len() == 2 && t.starts_with('0') && t == "00")
        };
        if let Some((y, p)) = split_date(s, &['M', 'm'], Some(&month)) {
            return Some(Date::from_parts(Freq::Monthly, y, p));
        }
        let week = |t: &str| {
            !t.starts_with('0') && matches!(t.len(), 1 | 2) && t.parse::<i64>().is_ok_and(|w| (1..=53).contains(&w))
        };
        // isweekly.m (7.2): `^[0-9]+[Ww]...`, no weekly date before the year zero
        if let Some((y, w)) = split_date(s, &['W', 'w'], Some(&week))
            && y >= 0
            && w <= 52 + islongyear(y) as i64
        {
            return Some(Date::from_parts(Freq::Weekly, y, w));
        }
        Self::parse_daily(s)
    }

    /// `isdaily.m`: `-?yyyy-mm-dd` with two-digit month and day, valid day of month.
    fn parse_daily(s: &str) -> Option<Date> {
        let (neg, body) = match s.strip_prefix('-') {
            Some(b) => (true, b),
            None => (false, s),
        };
        let p: Vec<&str> = body.split('-').collect();
        if p.len() != 3 || p[0].is_empty() || !p.iter().all(|x| !x.is_empty() && x.chars().all(|c| c.is_ascii_digit()))
        {
            return None;
        }
        if p[1].len() != 2 || p[2].len() != 2 {
            return None;
        }
        let y: i64 = p[0].parse().ok()?;
        let y = if neg { -y } else { y };
        let (m, d): (i64, i64) = (p[1].parse().ok()?, p[2].parse().ok()?);
        if !(1..=12).contains(&m) || d < 1 || d > month_days(y, m) {
            return None;
        }
        Some(Date::daily(y, m, d))
    }

    /// A date as written in a .mod file: a literal, optionally followed by `+k` / `-k` shifts
    /// (`2023Q1+6`); unary `+`/`-` prefixes shift by one period (`+1950Q1` = `1950Q2`).
    pub fn parse_mod(s: &str) -> Option<Date> {
        let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        let mut body = s.as_str();
        let mut shift = 0i64;
        while let Some(c) = body.chars().next() {
            if c == '+' {
                shift += 1;
            } else if c == '-' && !body[1..].starts_with(|c: char| c.is_ascii_digit()) {
                shift -= 1;
            } else {
                break;
            }
            body = &body[1..];
        }
        if let Some(d) = Date::parse(body) {
            return Some(d.add(shift));
        }
        // split trailing +k / -k terms (after the literal's letter)
        let lit_end = body.find(|c: char| c.is_ascii_alphabetic()).map(|k| {
            let rest = &body[k + 1..];
            k + 1 + rest.find(['+', '-']).unwrap_or(rest.len())
        })?;
        let mut d = Date::parse(&body[..lit_end])?;
        let mut rest = &body[lit_end..];
        while !rest.is_empty() {
            let sign = if rest.starts_with('+') {
                1
            } else if rest.starts_with('-') {
                -1
            } else {
                return None;
            };
            rest = &rest[1..];
            let k = rest.find(['+', '-']).unwrap_or(rest.len());
            let n: i64 = rest[..k].parse().ok()?;
            shift += sign * n;
            rest = &rest[k..];
        }
        d.time += shift;
        Some(d)
    }

    /// `char(d)` (`date2string.m`).
    pub fn label(&self) -> String {
        let (f, t) = (self.freq.number(), self.time);
        match self.freq {
            Freq::Daily => {
                let (y, m, d) = civil_from_days(t - DATENUM_1970);
                if y < 0 { format!("-{:04}-{m:02}-{d:02}", -y) } else { format!("{y:04}-{m:02}-{d:02}") }
            }
            Freq::Annual => format!("{t}Y"),
            Freq::Weekly => {
                let (y, w) = self.year_week();
                format!("{y}W{w}")
            }
            _ => {
                let year = (t - 1).div_euclid(f);
                format!("{year}{}{}", self.freq.letter(), t - year * f)
            }
        }
    }

    /// ISO (year, week) of a weekly date (`isoyearweek.m`).
    fn year_week(&self) -> (i64, i64) {
        isoyearweek(self.time)
    }

    pub fn add(self, k: i64) -> Date {
        Date { freq: self.freq, time: self.time + k }
    }
    /// `o - self` in periods (None if the frequencies differ).
    pub fn diff(self, o: Date) -> Option<i64> {
        (self.freq == o.freq).then_some(o.time - self.time)
    }
    /// `year(d)` (`@dates/year.m`; weekly: the ISO year, dseries 7.2).
    pub fn year(self) -> i64 {
        match self.freq {
            Freq::Weekly => isoyearweek(self.time).0,
            Freq::Daily => civil_from_days(self.time - DATENUM_1970).0,
            Freq::Annual => self.time,
            f => (self.time - 1).div_euclid(f.number()),
        }
    }
    /// `subperiod(d)` (`@dates/subperiod.m`; weekly: the ISO week, dseries 7.2). Daily dates
    /// error as in Dynare 7.2.
    pub fn subperiod(self) -> Result<i64> {
        match self.freq {
            Freq::Daily => bail!("dates::subperiod: Method not available for daily frequency!"),
            Freq::Annual => Ok(1),
            Freq::Weekly => Ok(isoyearweek(self.time).1),
            f => Ok(self.time - (self.time - 1).div_euclid(f.number()) * f.number()),
        }
    }
    /// `a:b` (`colon.m`, step 1).
    pub fn range(a: Date, b: Date) -> Result<Vec<Date>> {
        if a.freq != b.freq {
            bail!("dates:colon: {a} and {b} must have common frequency");
        }
        if a.time > b.time {
            bail!("dates:colon: First date must preceed the second one.");
        }
        Ok((a.time..=b.time).map(|t| Date { freq: a.freq, time: t }).collect())
    }
}

/// An option value that may be a date or an observation number (`first_obs = {INTEGER | DATE}`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DateOrIndex {
    Index(i64),
    Date(Date),
}

/// Reads `key` of `o` as an integer or a date (`OptValue::Num` / `OptValue::Str`).
pub fn option(o: &crate::modfile::Options, key: &str) -> Result<Option<DateOrIndex>> {
    use crate::modfile::OptValue;
    Ok(match crate::modfile::opt(o, key) {
        None => None,
        Some(OptValue::Num(v)) => Some(DateOrIndex::Index(*v as i64)),
        // first_obs = [8]: a one-element vector is the scalar
        Some(OptValue::Vec(v)) if v.len() == 1 => Some(DateOrIndex::Index(v[0] as i64)),
        Some(OptValue::Str(s)) => Some(DateOrIndex::Date(
            Date::parse_mod(s).ok_or_else(|| anyhow!("option {key} = {s}: not an integer or a date"))?,
        )),
        Some(OptValue::List(l)) if l.len() == 1 => match &l[0] {
            OptValue::Num(v) => Some(DateOrIndex::Index(*v as i64)),
            OptValue::Str(s) => Some(DateOrIndex::Date(
                Date::parse_mod(s).ok_or_else(|| anyhow!("option {key} = {s}: not an integer or a date"))?,
            )),
            _ => bail!("option {key}: expected an integer or a date"),
        },
        Some(_) => bail!("option {key}: expected an integer or a date"),
    })
}

/// `sameperiodpreviousyear.m` (dseries 7.2) on the dates `init`, `init+1`, ..., `init+n-1`:
/// the 0-based index of the same period of the previous year, `None` when it is before
/// `init` or does not exist (the 53rd ISO week after a 52-week year, the 29th of February).
pub fn same_period_previous_year(init: Date, n: usize) -> Vec<Option<usize>> {
    (0..n)
        .map(|t| {
            let d = init.add(t as i64);
            let lag = match d.freq {
                Freq::Weekly => {
                    let (y, w) = isoyearweek(d.time);
                    let p = 52 + islongyear(y - 1) as i64;
                    if w > p {
                        return None;
                    }
                    p
                }
                Freq::Daily => {
                    let (y, m, dd) = civil_from_days(d.time - DATENUM_1970);
                    if m == 2 && dd == 29 {
                        return None;
                    }
                    d.time - DATENUM_1970 - days_from_civil(y - 1, m, dd)
                }
                f => f.number(),
            };
            (t as i64 - lag >= 0).then(|| (t as i64 - lag) as usize)
        })
        .collect()
}

/// `ydiff_` (dseries 7.2): `x(t) - x(same period of the previous year)`, NaN when undefined.
pub fn ydiff(init: Date, x: &[f64]) -> Vec<f64> {
    same_period_previous_year(init, x.len())
        .iter()
        .enumerate()
        .map(|(t, id)| id.map_or(f64::NAN, |j| x[t] - x[j]))
        .collect()
}

/// `ygrowth_` (dseries 7.2): `x(t) / x(same period of the previous year) - 1`, NaN when undefined.
pub fn ygrowth(init: Date, x: &[f64]) -> Vec<f64> {
    same_period_previous_year(init, x.len())
        .iter()
        .enumerate()
        .map(|(t, id)| id.map_or(f64::NAN, |j| x[t] / x[j] - 1.0))
        .collect()
}

/// The dates of a dseries: `init`, `init+1`, ... (`dseries.dates`).
pub fn dates_from(init: Date, n: usize) -> Vec<Date> {
    (0..n as i64).map(|k| init.add(k)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_label() {
        for s in ["2004Q1", "2004M3", "2004M12", "2004Y", "2004S2", "2004W1", "2020W53", "2004-03-31", "-5Q2", "0Q4"] {
            let d = Date::parse(s).unwrap_or_else(|| panic!("{s}"));
            assert_eq!(d.label(), s.replace('H', "S"), "{s}");
        }
        assert_eq!(Date::parse("2004A").unwrap().label(), "2004Y");
        assert_eq!(Date::parse("2004H1").unwrap().label(), "2004S1");
        assert_eq!(Date::parse("2004q1").unwrap().label(), "2004Q1");
        assert_eq!(Date::parse("2004Q1").unwrap().time, 2004 * 4 + 1);
        assert_eq!(Date::parse("2003-03-31").unwrap().time, 731671); // datenum(2003,3,31)
        for bad in ["2004Q5", "2004M13", "2004M0", "2004-3-31", "2004-02-30", "2003W53", "Q1", "2004", "2004QQ1"] {
            assert!(Date::parse(bad).is_none(), "{bad}");
        }
        assert_eq!(Date::parse("2003-03-31").unwrap().add(4).label(), "2003-04-04");
        assert_eq!(Date::parse("2004Q4").unwrap().add(1).label(), "2005Q1");
        assert_eq!(Date::parse("2004W52").unwrap().add(1).label(), "2004W53");
        assert_eq!(Date::parse("2004W53").unwrap().add(1).label(), "2005W1");
    }

    #[test]
    fn mod_literals() {
        assert_eq!(Date::parse_mod("2023Q1+6").unwrap().label(), "2024Q3");
        assert_eq!(Date::parse_mod("+1950Q1").unwrap().label(), "1950Q2");
        assert_eq!(Date::parse_mod("1951Q2-2").unwrap().label(), "1950Q4");
        assert_eq!(Date::parse_mod("2004M3").unwrap().label(), "2004M3");
        assert!(Date::parse_mod("abc").is_none());
        let a = Date::parse("1950Q1").unwrap();
        assert_eq!(a.diff(Date::parse("1951Q2").unwrap()), Some(5));
        assert_eq!(Date::range(a, Date::parse("1951Q1").unwrap()).unwrap().len(), 5);
    }
}
