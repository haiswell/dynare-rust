// SPDX-License-Identifier: GPL-3.0-or-later
//! Perfect foresight with expectation errors: port of
//! `perfect_foresight_with_expectation_errors_setup.m` (information sets from `shocks`,
//! `mshocks`, `shocks(learnt_in=...)`, `mshocks(learnt_in=...)`, `endval(learnt_in=...)` or
//! the `datafile` CSV) and `perfect_foresight_with_expectation_errors_solver.m` (one
//! perfect-foresight simulation per informational period, by stream D's `DetState::solve`,
//! with the terminal steady state recomputed as in `endval_steady`).

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

use super::decl::LearntKind;
use crate::deterministic::DetState;
use crate::linalg::Mat;
use crate::modfile::{DetShockKind, ModFile, OptValue, Options, opt};

/// `oo_.pfwee`.
#[derive(Clone, Debug)]
pub struct PfweeInfo {
    pub periods: usize,
    /// `terminal_info`: n_exo x periods (column = informational period).
    pub terminal: Mat,
    /// `shocks_info(:, :, p)`: n_exo x periods (real time), as anticipated in period `p + 1`.
    pub shocks: Vec<Mat>,
    /// `controlled_paths_by_period(:, p)` (real time), as anticipated in period `p + 1`;
    /// empty without `perfect_foresight_controlled_paths`.
    pub controlled: Vec<Vec<super::controlled::ControlledPeriod>>,
}

fn flag(o: &Options, k: &str) -> bool {
    matches!(opt(o, k), Some(OptValue::Flag)) || matches!(opt(o, k), Some(OptValue::Num(v)) if *v != 0.0)
}

/// Options of perfect_foresight_solver understood by `SimulOptions::update`.
fn simul_options(o: &Options, skip: &[&str]) -> Options {
    o.iter().filter(|(k, _)| !skip.contains(&k.as_str())).cloned().collect()
}

/// `perfect_foresight_with_expectation_errors_setup`.
pub fn setup(mf: &ModFile, det: &mut DetState, o: &Options, dir: &Path) -> Result<()> {
    let m = &mf.model;
    let (n, nx) = (m.n_endo(), m.n_exo());
    if det.initval_series.is_some() {
        bail!("perfect_foresight_with_expectation_errors_setup: cannot be used in conjunction with initval_file");
    }
    let datafile = match opt(o, "datafile") {
        Some(OptValue::Str(s)) => Some(s.clone()),
        _ => None,
    };
    det.update_options(&simul_options(o, &["datafile"]));
    // ---- shocks/shock_paths: get_simulation_periods.m
    let (periods, fsp, _) = det.simulation_periods()?;
    let mut terminal = Mat::from_element(nx, periods, f64::NAN);
    let mut shocks = vec![Mat::from_element(nx, periods, f64::NAN); periods];
    if let Some(df) = datafile {
        if !mf.det_shocks.is_empty()
            || !mf.ep.learnt_shocks.is_empty()
            || det.initial_steady_state.is_some()
            || !mf.ep.learnt_endval.is_empty()
        {
            println!(
                "Warning: perfect_foresight_with_expectation_errors_setup: since you passed the datafile option, the contents of shocks and endval blocks will be ignored"
            );
        }
        let path = if Path::new(&df).is_absolute() { Path::new(&df).to_path_buf() } else { dir.join(&df) };
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header: Vec<String> = lines
            .next()
            .ok_or_else(|| anyhow!("{}: empty file", path.display()))?
            .split(',')
            .map(|s| s.trim().trim_matches('"').to_string())
            .collect();
        let names = &header[1..];
        let raw: Vec<Vec<f64>> =
            lines.map(|l| l.split(',').skip(1).map(|s| s.trim().parse::<f64>().unwrap_or(0.0)).collect()).collect();
        if raw.len() != periods + 2 {
            bail!(
                "perfect_foresight_with_expectation_errors_setup: the number of rows in {df} does not match the periods setting"
            );
        }
        if raw.iter().any(|r| r.len() != names.len()) {
            bail!("perfect_foresight_with_expectation_errors_setup: first line malformed in {df}");
        }
        for (i, name) in names.iter().enumerate() {
            let exo = m.exo_index(name).ok_or_else(|| {
                anyhow!("perfect_foresight_with_expectation_errors_setup: '{name}' is not an exogenous variable")
            })?;
            let pid = raw[0][i] as usize;
            if pid < 1 || pid > periods {
                bail!(
                    "perfect_foresight_with_expectation_errors_setup: informational period {pid} out of range in {df}"
                );
            }
            for t in pid..=periods {
                shocks[pid - 1][(exo, t - 1)] = raw[t][i];
            }
            terminal[(exo, pid - 1)] = raw[periods + 1][i];
        }
    } else {
        // ---- shocks/shock_paths: shock_paths / controlled paths with learnt_in, dates
        use crate::deterministic::periods::{learnt_in_period, to_sim_periods};
        let sp: Vec<&crate::deterministic::shock_paths::ShockPathsBlock> =
            mf.q1.shock_paths.iter().filter(|b| b.active).collect();
        let at_one = |n: usize, d: Option<crate::dates::Date>| d.is_none() && n == 1;
        let cps_one =
            mf.ep.controlled_paths.iter().chain(&det.q1_controlled).all(|c| at_one(c.learnt_in, c.learnt_in_date));
        if mf.ep.learnt_shocks.is_empty()
            && mf.ep.learnt_endval.is_empty()
            && sp.iter().all(|b| at_one(b.learnt_in, b.learnt_in_date))
            && cps_one
        {
            println!(
                "Warning: perfect_foresight_with_expectation_errors_setup: there is no shocks(learnt_in=...), endval(learnt_in=...), shock_paths(learnt_in=...) or perfect_foresight_controlled_paths(learnt_in=...) block, and you did not pass the datafile option, so there is no point in using this command"
            );
        }
        let any_date = mf.det_shocks.iter().any(|s| !s.dates.is_empty())
            || mf.ep.learnt_shocks.iter().any(|s| !s.dates.is_empty() || s.learnt_in_date.is_some())
            || mf.ep.learnt_endval.iter().any(|e| e.learnt_in_date.is_some())
            || sp.iter().any(|b| b.contains_date);
        if any_date && fsp.is_none() {
            bail!(
                "perfect_foresight_with_expectations_error_setup: at least one periods statement, or learnt_in option, or learnt_in() namespace in shock_paths block is specified using a date but neither first_simulation_period nor last_simulation_period option was passed"
            );
        }
        if !sp.is_empty() {
            crate::deterministic::shock_paths::check_databases(mf, "perfect_foresight_with_expectation_errors_setup")?;
            let paths = crate::deterministic::shock_paths::information_paths(
                mf,
                &det.params,
                &det.steady_state,
                &det.exo_steady_state,
                periods,
                fsp,
            )?;
            for (ip, m) in paths.iter().enumerate() {
                shocks[ip] = m.columns(0, periods).into_owned();
                terminal.set_column(ip, &m.column(periods));
            }
        }
        let xs = &det.exo_steady_state;
        for k in 0..nx {
            if !sp.is_empty() {
                break;
            }
            terminal[(k, 0)] = xs[k];
            for t in 0..periods {
                shocks[0][(k, t)] = xs[k];
            }
        }
        let check = |p: i64| -> Result<usize> {
            if p < 1 || p as usize > periods {
                bail!(
                    "perfect_foresight_with_expectation_errors_setup: a shock is set for period {p}, outside 1..{periods}"
                );
            }
            Ok(p as usize - 1)
        };
        for s in mf.det_shocks.iter().filter(|_| sp.is_empty()) {
            let v = match s.kind {
                DetShockKind::Level => s.value,
                DetShockKind::MultiplySteadyState => xs[s.exo] * s.value,
                DetShockKind::MultiplyInitialSteadyState => {
                    let ix = det.initial_exo_steady_state.as_ref().ok_or_else(|| {
                        anyhow!("Option relative_to_initval of mshocks block cannot be used without an endval block")
                    })?;
                    ix[s.exo] * s.value
                }
            };
            for p in s.sim_periods(fsp)? {
                shocks[0][(s.exo, check(p)?)] = v;
            }
        }
        for p in (1..=periods).filter(|_| sp.is_empty()) {
            let ip = p - 1;
            if p > 1 {
                let (prev_t, prev_s) = (terminal.column(ip - 1).clone_owned(), shocks[ip - 1].clone());
                terminal.set_column(ip, &prev_t);
                shocks[ip] = prev_s;
            }
            let learnt_now =
                |n: usize, d: Option<crate::dates::Date>| learnt_in_period(n, d, fsp).is_ok_and(|q| q == p);
            for e in mf.ep.learnt_endval.iter().filter(|e| learnt_now(e.learnt_in, e.learnt_in_date)) {
                if ip == 0 && e.kind != LearntKind::Level {
                    bail!(
                        "perfect_foresight_with_expectation_errors_setup: endval(learnt_in=...) with += or *= cannot be learnt in the first period"
                    );
                }
                let v = match e.kind {
                    LearntKind::Level => e.value,
                    LearntKind::Add => terminal[(e.exo, ip - 1)] + e.value,
                    LearntKind::Multiply => terminal[(e.exo, ip - 1)] * e.value,
                    LearntKind::MultiplySteadyState | LearntKind::MultiplyInitialSteadyState => {
                        bail!("Unknown type in M_.learnt_endval")
                    }
                };
                terminal[(e.exo, ip)] = v;
                for t in ip..periods {
                    shocks[ip][(e.exo, t)] = v;
                }
            }
            for s in mf.ep.learnt_shocks.iter().filter(|s| learnt_now(s.learnt_in, s.learnt_in_date)) {
                if ip == 0 && matches!(s.kind, LearntKind::Add | LearntKind::Multiply) {
                    bail!(
                        "perfect_foresight_with_expectation_errors_setup: shocks(learnt_in=...) with add or multiply cannot be learnt in the first period"
                    );
                }
                for q in to_sim_periods(&s.periods, &s.dates, fsp)? {
                    let t = check(q)?;
                    let v = match s.kind {
                        LearntKind::Level => s.value,
                        LearntKind::Add => shocks[ip - 1][(s.exo, t)] + s.value,
                        LearntKind::Multiply => shocks[ip - 1][(s.exo, t)] * s.value,
                        LearntKind::MultiplySteadyState => terminal[(s.exo, ip)] * s.value,
                        LearntKind::MultiplyInitialSteadyState => bail!("Unknown type in M_.learnt_shocks"),
                    };
                    shocks[ip][(s.exo, t)] = v;
                }
            }
        }
    }
    // initial guess and conditions (oo_.endo_simul / oo_.exo_simul)
    let (ml, mlead) = (det.maximum_lag, det.maximum_lead);
    let ncols = ml + periods + mlead;
    let ss = det.steady_state.clone();
    let mut endo = Mat::from_fn(n, ncols, |j, _| ss[j]);
    if mf.histval.is_empty() && det.histval_file.is_none() {
        if let Some(is) = &det.initial_steady_state {
            for c in 0..ml {
                for j in 0..n {
                    endo[(j, c)] = is[j];
                }
            }
        }
    } else {
        if det.initial_steady_state.is_some() {
            bail!("histval and endval cannot be used simultaneously");
        }
        let eh = match &det.histval_file {
            Some((e, _)) => e.clone(),
            None => det.histval(mf).0,
        };
        for c in 0..ml {
            for j in 0..n {
                endo[(j, c)] = eh[(j, c)];
            }
        }
    }
    let xs = det.exo_steady_state.clone();
    let ix = det.initial_exo_steady_state.clone();
    let exo = Mat::from_fn(ncols, nx, |r, k| match &ix {
        Some(ix) if r < ml => ix[k],
        _ => xs[k],
    });
    det.endo_simul = Some(endo);
    det.exo_simul = Some(exo);
    let mut controlled = Vec::new();
    if !mf.ep.controlled_paths.is_empty() || !det.q1_controlled.is_empty() {
        for p in 1..=periods {
            controlled.push(
                super::controlled::by_period_with(mf, &det.q1_controlled, periods, Some(p), fsp)?.unwrap_or_default(),
            );
        }
    }
    det.pfwee = Some(PfweeInfo { periods, terminal, shocks, controlled });
    Ok(())
}

/// `perfect_foresight_with_expectation_errors_solver`.
pub fn solver(mf: &ModFile, det: &mut DetState, o: &Options, dir: &Path, fname: &str) -> Result<()> {
    let model = &mf.model;
    let (n, nx) = (model.n_endo(), model.n_exo());
    let info = det
        .pfwee
        .clone()
        .ok_or_else(|| anyhow!("perfect_foresight_with_expectation_errors_solver: run perfect_foresight_with_expectation_errors_setup first"))?;
    // the command's options are global (the preprocessor writes them to options_ first);
    // the changes below are local to the MATLAB function
    det.update_options(&simul_options(o, &["constant_simulation_length"]));
    let saved = det.simul.clone();
    let constant_length = flag(o, "constant_simulation_length");
    if constant_length && det.simul.last_simulation_period.is_some() {
        bail!("Options constant_simulation_length and last_simulation_period cannot be used together");
    }
    let has_controlled = info.controlled.iter().any(|c| c.iter().any(|p| !p.endogenize.is_empty()));
    if constant_length && has_controlled {
        bail!(
            "Options constant_simulation_length is incompatible with perfect_foresight_controlled_paths or shock_paths+exogenize block"
        );
    }
    let periods = info.periods;
    let (ml, mlead) = (det.maximum_lag, det.maximum_lead);
    let mut endo = det.endo_simul.clone().ok_or_else(|| anyhow!("oo_.endo_simul is empty"))?;
    let mut exo = det.exo_simul.clone().ok_or_else(|| anyhow!("oo_.exo_simul is empty"))?;
    if mlead > 0 {
        det.simul.endval_steady = true;
    }
    if det.initial_steady_state.is_none() {
        det.initial_steady_state = Some(det.steady_state.clone());
        det.initial_exo_steady_state = Some(det.exo_steady_state.clone());
    }
    let mut info_period = 1usize;
    let mut increment = 0usize;
    let mut completion_share = 1.0;
    let mlf = det.simul.q1.homotopy_marginal_linearization_fallback;
    let result = (|| -> Result<()> {
        while info_period <= periods {
            let ip = info_period - 1;
            if !det.simul.noprint {
                println!(
                    "perfect_foresight_with_expectations_errors_solver: computing solution for information available at period {info_period}"
                );
            }
            det.exo_steady_state = info.terminal.column(ip).iter().copied().collect();
            if constant_length && increment > 0 {
                let ss = det.steady_state.clone();
                let (c0, r0) = (endo.ncols(), exo.nrows());
                endo = endo.clone().resize_horizontally(c0 + increment, 0.0);
                for c in c0..c0 + increment {
                    for j in 0..n {
                        endo[(j, c)] = ss[j];
                    }
                }
                exo = exo.clone().resize_vertically(r0 + increment, f64::NAN);
            }
            let sim_length = if constant_length { periods } else { periods - info_period + 1 };
            let e = endo.columns(ip, endo.ncols() - ip).clone_owned();
            let mut x = exo.rows(ip, exo.nrows() - ip).clone_owned();
            let known = periods - info_period + 1;
            for t in 0..known {
                for k in 0..nx {
                    x[(ml + t, k)] = info.shocks[ip][(k, ip + t)];
                }
            }
            for r in ml + known..x.nrows() {
                for k in 0..nx {
                    x[(r, k)] = det.exo_steady_state[k];
                }
            }
            det.simul.set_periods(sim_length);
            // reset so as to avoid an inconsistency with periods
            det.simul.first_simulation_period = None;
            det.simul.last_simulation_period = None;
            // marginal_linearization_previous_raw_sims: the raw simulations of the previous
            // informational period, from its first period on
            let prev = match (&det.sim1, &det.sim2) {
                (Some(s1), Some(s2)) if info_period > 1 && completion_share < 1.0 && mlf > 0.0 => {
                    let tail = |r: &crate::deterministic::RawSim| crate::deterministic::RawSim {
                        endo_simul: r.endo_simul.columns(increment, r.endo_simul.ncols() - increment).into_owned(),
                        exo_simul: r.exo_simul.rows(increment, r.exo_simul.nrows() - increment).into_owned(),
                        homotopy_completion_share: r.homotopy_completion_share,
                    };
                    Some(crate::deterministic::MarginalRawSims { sim1: tail(s1), sim2: tail(s2) })
                }
                _ => None,
            };
            det.endo_simul = Some(e);
            det.exo_simul = Some(x);
            det.controlled = info
                .controlled
                .get(ip)
                .map(|c| c[ip..].to_vec())
                .filter(|c| c.iter().any(|p| !p.endogenize.is_empty()));
            let r = det.solve_with(model, prev.as_ref());
            det.controlled = None;
            r?;
            if !det.last.status {
                bail!(
                    "perfect_foresight_with_expectation_errors_solver: failed to compute solution for information available at period {info_period}"
                );
            }
            if info_period == 1 {
                completion_share = det.last.homotopy_completion_share;
                det.simul.homotopy_max_completion_share = completion_share;
            } else if det.last.homotopy_completion_share != completion_share {
                bail!(
                    "perfect_foresight_solver_with_expectation_errors: could not find a solution for information available at period {info_period} with the same homotopy completion share as period 1; a possible solution is to retry the simulation with homotopy_max_completion_share={}",
                    (det.last.homotopy_completion_share * 1000.0).floor() / 1000.0
                );
            }
            let es = det.endo_simul.as_ref().unwrap();
            endo.columns_mut(ip, es.ncols()).copy_from(es);
            let xsim = det.exo_simul.as_ref().unwrap();
            exo.rows_mut(ip, xsim.nrows()).copy_from(xsim);
            increment = 1;
            while info_period + increment <= periods && {
                let (a, b) = (ip, ip + increment);
                (0..nx).all(|k| info.terminal[(k, a)] == info.terminal[(k, b)])
                    && (b..periods).all(|t| (0..nx).all(|k| info.shocks[a][(k, t)] == info.shocks[b][(k, t)]))
                    && (info.controlled.is_empty() || info.controlled[a][b..] == info.controlled[b][b..])
            } {
                increment += 1;
            }
            info_period += increment;
        }
        Ok(())
    })();
    // options_ is local to the MATLAB function
    det.simul = saved;
    result?;
    det.endo_simul = Some(endo);
    det.exo_simul = Some(exo);
    det.write_results(model, dir, fname)
}
