// SPDX-License-Identifier: GPL-3.0-or-later
//! Parse-time checks of the values of the commands' numerical options (Dynare's
//! preprocessor and `check_*` routines reject them before anything runs): a known option
//! with an unusable value is an error, not a silently ignored option.

use super::{OptValue, Options};
use anyhow::{Result, bail};

/// Largest horizon / number of periods accepted (`irf`, `forecast`, `periods`, ...): the
/// arrays sized by them must fit in memory.
pub const MAX_HORIZON: f64 = 1e7;

enum Rule {
    /// Integer in `lo..=hi`.
    Int(f64, f64),
    /// `lo <= x < hi`.
    HalfOpen(f64, f64),
    /// `lo < x < hi`.
    Open(f64, f64),
}

fn rule(cmd: &str, key: &str) -> Option<Rule> {
    Some(match key {
        // options_.ep.stochastic.order: 0 is the deterministic extended path (the default)
        "order" if cmd == "extended_path" => Rule::Int(0.0, 100.0),
        "irf"
        | "forecast"
        | "periods"
        | "ar"
        | "drop"
        | "replic"
        | "simul_replic"
        | "mh_replic"
        | "sub_draws"
        | "presample"
        | "filtered_theoretical_moments_grid" => Rule::Int(0.0, MAX_HORIZON),
        "order" => Rule::Int(1.0, 100.0),
        "lik_init" => Rule::Int(1.0, 5.0),
        "kalman_algo" => Rule::Int(0.0, 5.0),
        "mh_nblocks" => Rule::Int(1.0, 1e6),
        "mh_drop" => Rule::HalfOpen(0.0, 1.0),
        "mh_conf_sig" => Rule::Open(0.0, 1.0),
        _ => return None,
    })
}

/// Checks the options of a command; the error reads "invalid value for option X ...".
pub fn check_options(cmd: &str, opts: &Options) -> Result<()> {
    for (k, v) in opts {
        let Some(r) = rule(cmd, k) else { continue };
        let x = match v {
            OptValue::Num(x) => *x,
            OptValue::Vec(x) if x.len() == 1 => x[0],
            _ => bail!("invalid value for option {k}: expected a number"),
        };
        let what = match r {
            Rule::Int(lo, hi) if !(x.is_finite() && x.fract() == 0.0 && x >= lo && x <= hi) => {
                if hi >= MAX_HORIZON {
                    format!("expected an integer between {lo} and {hi:e}")
                } else {
                    format!("expected an integer between {lo} and {hi}")
                }
            }
            Rule::HalfOpen(lo, hi) if !(x >= lo && x < hi) => format!("expected {lo} <= {k} < {hi}"),
            Rule::Open(lo, hi) if !(x > lo && x < hi) => format!("expected {lo} < {k} < {hi}"),
            _ => continue,
        };
        bail!("invalid value for option {k} ({x}): {what}");
    }
    Ok(())
}
