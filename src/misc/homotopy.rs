// SPDX-License-Identifier: GPL-3.0-or-later
//! [stream X] `steady(homotopy_mode = 1|2|3, homotopy_steps = N, homotopy_force_continue)`
//! with a `homotopy_setup` block: port of `homotopy1/2/3` in `steady.m`. The homotopy moves
//! the parameters / exogenous steady-state values of the block from their start to their end
//! values, solving the steady state at each step from the previous one; the `steady` command
//! then solves at the end values as usual.

use anyhow::{Result, bail};

use crate::deterministic::{DetState, SteadyStateModel, evaluate_steady_state};
use crate::model::Model;
use crate::modfile::{ModFile, OptValue, Options, opt};
use crate::solver::SteadyOptions;

/// Whether a `steady` command asks for a homotopy.
pub fn requested(o: &Options) -> bool {
    matches!(opt(o, "homotopy_mode"), Some(OptValue::Num(v)) if *v != 0.0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Target {
    Param(usize),
    Exo(usize),
}

/// The state the homotopy works on (`M_.params`, `oo_.steady_state`, `oo_.exo_steady_state`).
pub struct State {
    pub ys: Vec<f64>,
    pub xs: Vec<f64>,
    pub params: Vec<f64>,
}

struct Solver<'a> {
    model: &'a Model,
    ssm: Option<&'a SteadyStateModel>,
    opts: SteadyOptions,
    check: bool,
}

impl Solver<'_> {
    /// `evaluate_steady_state` from `st.ys`; on success updates `ys` and `params`.
    fn solve(&self, st: &mut State) -> bool {
        match evaluate_steady_state(self.model, self.ssm, &st.ys, &st.xs, &st.params, &self.opts, self.check) {
            Ok((ys, p)) => {
                st.ys = ys;
                st.params = p;
                true
            }
            Err(_) => false,
        }
    }
}

fn set(st: &mut State, t: Target, v: f64) {
    match t {
        Target::Param(k) => st.params[k] = v,
        Target::Exo(k) => st.xs[k] = v,
    }
}

fn get(st: &State, t: Target) -> f64 {
    match t {
        Target::Param(k) => st.params[k],
        Target::Exo(k) => st.xs[k],
    }
}

/// MATLAB's `a:step:b` for the homotopy grids (`a + k*step`, the last point set to `b`).
fn colon(a: f64, b: f64, n: usize) -> Vec<f64> {
    let step = (b - a) / n as f64;
    (0..=n).map(|k| if k == n { b } else { a + k as f64 * step }).collect()
}

/// Runs `homotopy<mode>`; returns Dynare's `errorcode` (0 ok, 1 step not completed, 2 failure
/// at the starting values).
#[allow(clippy::too_many_arguments)]
fn homotopy(
    mode: u8,
    values: &[(Target, f64, f64)],
    steps: usize,
    s: &Solver,
    st: &mut State,
    names: &dyn Fn(Target) -> String,
) -> Result<i32> {
    let mut last: Option<(Vec<f64>, Vec<f64>)> = None;
    let restore = |st: &mut State, last: &Option<(Vec<f64>, Vec<f64>)>| {
        if let Some((p, x)) = last {
            st.params = p.clone();
            st.xs = x.clone();
        }
    };
    match mode {
        1 => {
            let grids: Vec<Vec<f64>> =
                values.iter().map(|&(_, a, b)| if a != b { colon(a, b, steps) } else { vec![b; steps + 1] }).collect();
            for i in 0..=steps {
                println!("HOMOTOPY mode 1: computing step {i}/{steps}...");
                for (k, &(t, _, _)) in values.iter().enumerate() {
                    set(st, t, grids[k][i]);
                }
                if !s.solve(st) {
                    if i == 0 {
                        return Ok(2);
                    }
                    restore(st, &last);
                    return Ok(1);
                }
                last = Some((st.params.clone(), st.xs.clone()));
            }
            Ok(0)
        }
        2 => {
            for &(t, a, _) in values {
                set(st, t, a);
            }
            if values.iter().any(|&(_, a, b)| a == b) {
                bail!("HOMOTOPY mode 2: initial and final values should be different");
            }
            for (i, &(t, a, b)) in values.iter().enumerate() {
                for (k, v) in colon(a, b, steps).into_iter().enumerate() {
                    set(st, t, v);
                    println!(
                        "HOMOTOPY mode 2: lauching solver with {} = {} ...",
                        names(t),
                        crate::deterministic::fmt_g(v)
                    );
                    if !s.solve(st) {
                        if i == 0 && k == 0 {
                            return Ok(2);
                        }
                        restore(st, &last);
                        return Ok(1);
                    }
                    last = Some((st.params.clone(), st.xs.clone()));
                }
            }
            Ok(0)
        }
        3 => {
            let tol = 1e-8;
            let old: Vec<f64> = values.iter().map(|v| v.1).collect();
            let target: Vec<f64> = values.iter().map(|v| v.2).collect();
            let nv = values.len();
            let mut cur = old.clone();
            let mut inc: Vec<f64> = (0..nv).map(|i| (target[i] - old[i]) / 2.0).collect();
            let mut last_values: Vec<f64> = Vec::new();
            let mut reached = 0usize;
            println!("HOMOTOPY mode 3: launching solver at initial point...");
            for iter in 1..=steps {
                for (k, &(t, _, _)) in values.iter().enumerate() {
                    set(st, t, cur[k]);
                }
                let old_ss = st.ys.clone();
                if s.solve(st) {
                    if reached == nv {
                        return Ok(0);
                    }
                    println!(
                        "{}",
                        if iter == 1 {
                            "HOMOTOPY mode 3: successful step, now jumping to final point..."
                        } else {
                            "HOMOTOPY mode 3: successful step, now multiplying increment by 2..."
                        }
                    );
                    last_values = cur.clone();
                    last = Some((st.params.clone(), st.xs.clone()));
                    inc.iter_mut().for_each(|v| *v *= 2.0);
                } else if iter == 1 {
                    return Ok(2);
                } else {
                    println!("HOMOTOPY mode 3: failed step, now dividing increment by 2...");
                    inc.iter_mut().for_each(|v| *v /= 2.0);
                    st.ys = old_ss;
                }
                reached = 0;
                for i in 0..nv {
                    cur[i] = last_values[i] + inc[i];
                    let up = target[i] > old[i] && cur[i] >= target[i];
                    let down = target[i] < old[i] && cur[i] <= target[i];
                    if up || down {
                        cur[i] = target[i];
                    }
                    reached += usize::from(up || down);
                }
                if inc.iter().fold(0.0f64, |m, v| m.max(v.abs())) < tol {
                    println!("HOMOTOPY mode 3: failed, increment has become too small");
                    restore(st, &last);
                    return Ok(1);
                }
            }
            println!(
                "HOMOTOPY mode 3: failed, maximum iterations reached; you may want to increase the homotopy_steps option"
            );
            restore(st, &last);
            Ok(1)
        }
        m => bail!("STEADY: invalid value for homotopy_mode option ({m})"),
    }
}

/// `steady.m` before `evaluate_steady_state`: the homotopy of the `homotopy_setup` block, on
/// the deterministic state when there is one (it holds `oo_`), else on the `initval` values.
/// `M_.params` (both copies), `oo_.steady_state` and `oo_.exo_steady_state` are updated.
pub fn before_steady(mf: &mut ModFile, det: Option<&mut DetState>, o: &Options) -> Result<()> {
    if mf.homotopy_setup.is_empty() {
        bail!("STEADY: a homotopy_setup block must be present when the homotopy_mode option is specified");
    }
    let num = |k: &str| match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    };
    let mode = num("homotopy_mode").unwrap_or(0.0) as u8;
    let steps = num("homotopy_steps").unwrap_or(10.0) as usize;
    let force = opt(o, "homotopy_force_continue").is_some_and(|v| !matches!(v, OptValue::Num(x) if *x == 0.0));
    let m = &mf.model;
    let ssm_owned = SteadyStateModel::from_modfile(mf)?;
    let mut st = match &det {
        Some(d) => State { ys: d.steady_state.clone(), xs: d.exo_steady_state.clone(), params: d.params.clone() },
        None => {
            let mut xs = vec![0.0; m.n_exo()];
            for &(k, v) in &mf.initval_exo {
                xs[k] = v;
            }
            let mut ys = vec![0.0; m.n_endo()];
            for &(i, v) in &mf.initval {
                ys[i] = v;
            }
            m.set_auxiliary_variables(&mut ys, &xs, &m.params);
            State { ys, xs, params: m.params.clone() }
        }
    };
    if ssm_owned.is_some() {
        bail!("STEADY: Can't use homotopy when providing a steady state external file");
    }
    let mut values = Vec::new();
    for (name, start, end) in &mf.homotopy_setup {
        let t = if let Some(k) = m.param_names.iter().position(|n| n == name) {
            Target::Param(k)
        } else if let Some(k) = m.exo_names.iter().position(|n| n == name) {
            Target::Exo(k)
        } else {
            bail!("HOMOTOPY_SETUP: unknown symbol {name}");
        };
        values.push((t, start.unwrap_or_else(|| get(&st, t)), *end));
    }
    // steady.m: homotopy_setup(from_initval_to_endval) adds the path from
    // oo_.initial_exo_steady_state to oo_.exo_steady_state for every exogenous variable the
    // block does not list (not overwriting the user's entries)
    if mf.homotopy_from_initval_to_endval {
        let Some(d) = det.as_ref().filter(|d| d.initial_exo_steady_state.is_some()) else {
            bail!("HOMOTOPY_SETUP: the from_initval_to_endval option cannot be used without an endval block");
        };
        let initial = d.initial_exo_steady_state.as_ref().unwrap();
        for k in 0..m.n_exo() {
            if !values.iter().any(|&(t, _, _)| t == Target::Exo(k)) {
                values.push((Target::Exo(k), initial[k], d.exo_steady_state[k]));
            }
        }
    }
    let mut opts = match &det {
        Some(d) => d.steady_opts,
        None => crate::dsge::steady_options(mf),
    };
    if let Some(v) = num("solve_algo") {
        opts.solve_algo = v as u8;
    }
    if let Some(v) = num("maxit") {
        opts.maxit = v as usize;
    }
    let solver = Solver { model: m, ssm: None, opts, check: opt(o, "nocheck").is_none() };
    let names = |t: Target| match t {
        Target::Param(k) => m.param_names[k].clone(),
        Target::Exo(k) => m.exo_names[k].clone(),
    };
    let code = homotopy(mode, &values, steps, &solver, &mut st, &names)?;
    if code != 0 {
        if code == 2 {
            println!("WARNING: homotopy failed at the first iteration (for the starting values)");
        } else {
            println!("\nWARNING: homotopy step was not completed");
            println!("The last values for which a solution was found are:");
            for &(t, _, _) in &values {
                println!("{:>12} {:12.6}", names(t), get(&st, t));
            }
        }
        if force {
            println!("Option homotopy_continue is set, so I continue ...");
        } else {
            bail!("Homotopy step failed");
        }
    }
    // write the state back
    let n_orig = mf.model.orig_endo_nbr;
    mf.model.params = st.params.clone();
    match det {
        Some(d) => {
            d.steady_state = st.ys;
            d.exo_steady_state = st.xs;
            d.params = st.params;
        }
        None => {
            mf.initval = (0..n_orig).map(|i| (i, st.ys[i])).collect();
            mf.initval_exo = st.xs.iter().enumerate().filter(|(_, v)| **v != 0.0).map(|(k, v)| (k, *v)).collect();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn colon_grid() {
        assert_eq!(super::colon(0.0, 1.0, 4), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }
}
