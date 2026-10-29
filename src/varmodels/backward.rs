// SPDX-License-Identifier: GPL-3.0-or-later
//! Simulation of a backward-looking model: port of `simul_backward_model.m` (nonlinear
//! path, `simul_backward_nonlinear_model.m`, `+backward_model/initialize.m`,
//! `+backward_model/simul_nonlinear_model.m`), and the dseries values that the commands of a
//! .mod file compute (the `X` of `X = simul_backward_model(...)`).
//!
//! - The auxiliary variables of the initial conditions are recomputed from the original ones
//!   (`<fname>.dynamic_set_auxiliary_series`): each auxiliary equation is solved for its
//!   variable, period by period (NaN where a lag is missing, as the dseries lag operators);
//! - each period is solved by Newton on the dynamic residuals with `y_{t-1}` as the initial
//!   guess (Dynare: `dynare_solve`, the solution is the same up to the solver tolerance);
//! - without innovations the shocks are drawn from N(0, `M_.Sigma_e`) with dynare-rust's
//!   generator (ChaCha8 with a fixed seed: `options_.bnlms.set_dynare_seed_to_default`), not
//!   MATLAB's; pass the innovations (third argument) to replay given shocks.

use crate::dates::Date;
use crate::linalg::Mat;
use crate::model::Model;
use anyhow::{Result, anyhow, bail};

/// A dseries: `data` is nobs x nvars, row `t` is the period `init + t`.
#[derive(Clone, Debug)]
pub struct Series {
    pub init: Date,
    pub names: Vec<String>,
    pub data: Mat,
}

impl Series {
    pub fn col(&self, n: &str) -> Option<usize> {
        self.names.iter().position(|x| x == n)
    }

    /// `X.f = dseries(v, date)` (`@dseries/subsasgn` + `merge`): column `f` becomes NaN except
    /// `v` at `date` (the whole column without a date); the range grows to contain `date`.
    pub fn set_field(&mut self, field: &str, v: f64, date: Option<Date>) -> Result<()> {
        if let Some(d) = date {
            let k = self.init.diff(d).ok_or_else(|| anyhow!("dseries: the frequencies differ"))?;
            let (n, m) = (self.data.nrows() as i64, self.data.ncols());
            let (lo, hi) = (k.min(0), (k + 1).max(n));
            if lo < 0 || hi > n {
                let mut data = Mat::from_element((hi - lo) as usize, m, f64::NAN);
                data.rows_mut((-lo) as usize, n as usize).copy_from(&self.data);
                self.data = data;
                self.init = self.init.add(lo);
            }
        }
        let j = match self.col(field) {
            Some(j) => j,
            None => {
                let (n, m) = self.data.shape();
                let mut data = Mat::from_element(n, m + 1, f64::NAN);
                data.columns_mut(0, m).copy_from(&self.data);
                self.data = data;
                self.names.push(field.to_string());
                m
            }
        };
        match date {
            Some(d) => {
                self.data.column_mut(j).fill(f64::NAN);
                let k = self.init.diff(d).unwrap() as usize;
                self.data[(k, j)] = v;
            }
            None => self.data.column_mut(j).fill(v),
        }
        Ok(())
    }

    /// CSV in the dseries layout (first column: the periods).
    pub fn to_csv(&self) -> String {
        let mut s = String::new();
        for n in &self.names {
            s.push(',');
            s.push_str(n);
        }
        s.push('\n');
        for t in 0..self.data.nrows() {
            s.push_str(&self.init.add(t as i64).label());
            for j in 0..self.data.ncols() {
                s.push_str(&format!(",{}", self.data[(t, j)]));
            }
            s.push('\n');
        }
        s
    }
}

/// Row of the dynamic model that defines each auxiliary variable (the `M_.aux_vars`): a
/// matching of the variables with the defining equations in which they appear at `t`.
fn aux_rows(model: &Model) -> Result<Vec<(usize, usize)>> {
    let n = model.n_endo();
    let pat = model.dynamic_jacobian_pattern();
    let first = model.orig_eq_nbr;
    let vars: Vec<usize> = model.aux_vars.iter().map(|a| a.endo_index).collect();
    let edges: Vec<Vec<usize>> = vars
        .iter()
        .map(|&v| pat.iter().filter(|&&(r, c)| c == n + v && r >= first).map(|&(r, _)| r).collect())
        .collect();
    // augmenting paths (Kuhn)
    fn augment(k: usize, edges: &[Vec<usize>], seen: &mut [bool], owner: &mut [Option<usize>]) -> bool {
        for &r in &edges[k] {
            if !seen[r] {
                seen[r] = true;
                if owner[r].is_none_or(|o| augment(o, edges, seen, owner)) {
                    owner[r] = Some(k);
                    return true;
                }
            }
        }
        false
    }
    let mut owner = vec![None; model.n_equations()];
    for (k, &v) in vars.iter().enumerate() {
        let mut seen = vec![false; model.n_equations()];
        if !augment(k, &edges, &mut seen, &mut owner) {
            bail!("no equation defines the auxiliary variable {}", model.endo_names[v]);
        }
    }
    let mut out: Vec<(usize, usize)> = owner.iter().enumerate().filter_map(|(r, o)| o.map(|k| (vars[k], r))).collect();
    out.sort_by_key(|&(_, r)| r);
    Ok(out)
}

/// Fills the auxiliary variables of the periods `0..nobs` of `y` (n x T, column = period) from
/// the other variables (`dynamic_set_auxiliary_series`).
fn set_auxiliary_series(model: &Model, params: &[f64], y: &mut Mat, x: &Mat, nobs: usize) -> Result<()> {
    let n = model.n_endo();
    let rows = aux_rows(model)?;
    if rows.is_empty() {
        return Ok(());
    }
    let mut ws = model.workspace();
    let mut jac = Mat::zeros(model.n_equations(), 3 * n + model.n_exo());
    let mut res = vec![0.0; model.n_equations()];
    let mut y3 = vec![0.0; 3 * n];
    for t in 0..nobs {
        for &(e, _) in &rows {
            y[(e, t)] = f64::NAN;
        }
        let xt: Vec<f64> = x.row(t).iter().copied().collect();
        // Gauss-Seidel sweeps: an auxiliary variable may depend on others at t
        for _ in 0..rows.len() + 1 {
            for &(e, r) in &rows {
                for i in 0..n {
                    y3[i] = if t == 0 { f64::NAN } else { y[(i, t - 1)] };
                    y3[n + i] = y[(i, t)];
                }
                y3[n + e] = 0.0;
                model.dynamic_residuals(&y3, &xt, params, &mut res);
                model.dynamic_jacobian_y3_with(&y3, &xt, params, &mut ws, &mut jac);
                let d = jac[(r, n + e)];
                let mut v = if d != 0.0 && d.is_finite() { -res[r] / d } else { f64::NAN };
                // one more Newton step for a nonlinear definition
                if v.is_finite() {
                    y3[n + e] = v;
                    model.dynamic_residuals(&y3, &xt, params, &mut res);
                    model.dynamic_jacobian_y3_with(&y3, &xt, params, &mut ws, &mut jac);
                    let d = jac[(r, n + e)];
                    if d != 0.0 && d.is_finite() {
                        v -= res[r] / d;
                    }
                }
                y[(e, t)] = v;
            }
        }
    }
    Ok(())
}

/// Upper Cholesky factor of a symmetric positive definite matrix (`chol`).
fn chol_upper(a: &Mat) -> Result<Mat> {
    let c =
        a.clone().cholesky().ok_or_else(|| anyhow!("the covariance matrix of the shocks is not positive definite"))?;
    Ok(c.l().transpose())
}

/// The values of the innovations (third argument) for the `samplesize` simulated periods: the
/// first `samplesize` observations, zero for the exogenous variables it lacks.
fn innovations_matrix(model: &Model, s: &Series, samplesize: usize) -> Result<Mat> {
    if s.data.nrows() < samplesize {
        bail!(
            "Time span in third argument is too short (should not be less than {samplesize}, the value of the second argument)"
        );
    }
    let mut shocks = Mat::zeros(samplesize, model.n_exo());
    for (k, e) in model.exo_names.iter().enumerate() {
        match s.col(e) {
            Some(j) => shocks.column_mut(k).copy_from(&s.data.view((0, j), (samplesize, 1))),
            None => println!("Exogenous variable {e} is not available in third argument, default value is zero."),
        }
    }
    Ok(shocks)
}

/// The initial conditions of an empty first argument (`+backward_model/initialize.m`, the
/// `isempty(initialconditions)` branch): a dseries of `M_.orig_maximum_lag` periods from 1Y.
///
/// `M_.endo_histval` (zero without `histval`) goes into the first column of `yinitdata`, the
/// `AUX_ENDO_LAG` rows of `M_.endo_histval` into the column of their lag, and the columns are
/// flipped when `M_.orig_maximum_endo_lag > 1` (here: when the model has an `AUX_ENDO_LAG`
/// variable). The exogenous variables are zero except the `AUX_EXO_LAG` entries, which Dynare
/// reads at the linear index `M_.orig_endo_nbr+i` of `M_.exo_histval` (exo_nbr x
/// `M_.maximum_lag`, i.e. one value per exogenous variable in a backward model), and fails
/// when that index is out of bounds.
fn histval_initial_conditions(model: &Model, origs: &[usize], histval: &[(bool, usize, i32, f64)]) -> Result<Series> {
    let nx = model.n_exo();
    let maxlag = model.orig_maximum_lag;
    let endo_histval =
        crate::analysis::stochastic_histval_of(model, histval).unwrap_or_else(|| vec![0.0; model.n_endo()]);
    let exo_histval =
        |k: usize| -> f64 { histval.iter().rev().find(|&&(e, j, l, _)| !e && j == k && l == 0).map_or(0.0, |h| h.3) };
    let mut yinitdata = Mat::zeros(origs.len(), maxlag);
    for (r, &i) in origs.iter().enumerate() {
        yinitdata[(r, 0)] = endo_histval[i];
    }
    let mut xinitdata = Mat::zeros(nx, maxlag);
    if model.aux_vars.iter().any(|a| a.ty == 1) {
        for a in model.aux_vars.iter().filter(|a| a.ty == 1) {
            if let (Some(o), Some(l)) = (a.orig_index, a.orig_lead_lag)
                && let Some(r) = origs.iter().position(|&i| i == o)
            {
                yinitdata[(r, l.unsigned_abs() as usize)] = endo_histval[a.endo_index];
            }
        }
        yinitdata = Mat::from_fn(yinitdata.nrows(), maxlag, |r, c| yinitdata[(r, maxlag - 1 - c)]);
    }
    if model.aux_vars.iter().any(|a| a.ty == 3) {
        for (i, a) in model.aux_vars.iter().enumerate().filter(|(_, a)| a.ty == 3) {
            // M_.exo_histval(M_.orig_endo_nbr+i), 1-based linear index
            let idx = origs.len() + i + 1;
            if idx > nx {
                bail!("Index exceeds the number of array elements. Index must not exceed {nx}.");
            }
            if let (Some(o), Some(l)) = (a.orig_index, a.orig_lead_lag) {
                xinitdata[(o, l.unsigned_abs() as usize)] = exo_histval(idx - 1);
            }
        }
        xinitdata = Mat::from_fn(nx, maxlag, |r, c| xinitdata[(r, maxlag - 1 - c)]);
    }
    let mut names: Vec<String> = origs.iter().map(|&i| model.endo_names[i].clone()).collect();
    names.extend(model.exo_names.iter().cloned());
    let data = Mat::from_fn(maxlag, origs.len() + nx, |t, j| {
        if j < origs.len() { yinitdata[(j, t)] } else { xinitdata[(j - origs.len(), t)] }
    });
    Ok(Series { init: Date::parse("1Y").unwrap(), names, data })
}

/// Port of `+backward_model/simul_static_model.m` with the innovations given (third argument):
/// each period solves the static model from zero (`dynare_solve`, `options_.solve_algo`,
/// `options_.simul.maxit`, `options_.dynatol.{f,x}`); a period where the solver fails is NaN.
/// The dseries starts at the first date of the innovations.
fn simul_static_model(
    model: &Model,
    params: &[f64],
    samplesize: usize,
    innovations: &Series,
    origs: &[usize],
    solve_algo: u8,
) -> Result<Series> {
    let n = model.n_endo();
    let nx = model.n_exo();
    let shocks = innovations_matrix(model, innovations, samplesize)?;
    let opts = crate::solver::SteadyOptions { solve_algo, maxit: 50, tolf: 1e-5, tolx: 1e-5, ..Default::default() };
    let mut y = Mat::from_element(n, samplesize, f64::NAN);
    for t in 0..samplesize {
        let xt: Vec<f64> = shocks.row(t).iter().copied().collect();
        let f = |z: &[f64]| -> (Vec<f64>, Mat) {
            let mut res = vec![0.0; model.n_equations()];
            model.static_residuals(z, &xt, params, &mut res);
            (res, model.static_jacobian(z, &xt, params))
        };
        let out = crate::solver::dynare_solve(&f, &vec![0.0; n], &opts);
        if out.errorflag {
            println!(
                "simul_static_mode: Nonlinear solver failed with errorcode={} in period {}.",
                out.errorcode,
                t + 1
            );
        } else {
            for i in 0..n {
                y[(i, t)] = out.x[i];
            }
        }
    }
    let mut names: Vec<String> = origs.iter().map(|&i| model.endo_names[i].clone()).collect();
    names.extend(model.exo_names.iter().cloned());
    let orig = origs.len();
    let data =
        Mat::from_fn(samplesize, orig + nx, |t, j| if j < orig { y[(origs[j], t)] } else { shocks[(t, j - orig)] });
    Ok(Series { init: innovations.init, names, data })
}

/// `simul_backward_model(initialconditions, samplesize[, innovations])`. `init` None is the
/// empty first argument (`[]`): the initial conditions then come from `M_.endo_histval` and
/// `M_.exo_histval`, i.e. the `histval` entries in force (`histval`, see
/// [`histval_initial_conditions`]), and the innovations need not follow them in time.
#[allow(clippy::too_many_arguments)]
pub fn simul_backward_model(
    model: &Model,
    params: &[f64],
    sigma_e: &Mat,
    init: Option<&Series>,
    samplesize: usize,
    innovations: Option<&Series>,
    histval: &[(bool, usize, i32, f64)],
    solve_algo: u8,
) -> Result<Series> {
    let n = model.n_endo();
    let nx = model.n_exo();
    if model.lead_lag_incidence.iter().any(|l| l[2] != 0) {
        bail!("simul_backward_model: the model is not backward or static.");
    }
    let origs: Vec<usize> = (0..model.orig_endo_nbr).collect();
    if model.lead_lag_incidence.iter().all(|l| l[0] == 0) {
        println!("The model is static. Use simul_static_model instead.");
        // Dynare passes its third argument on, undefined when it is not given
        let s = innovations.ok_or_else(|| {
            anyhow!("simul_backward_model: the model is static and needs the innovations (third argument)")
        })?;
        return simul_static_model(model, params, samplesize, s, &origs, solve_algo);
    }
    let orig = origs.len();
    let orig_names: Vec<String> = origs.iter().map(|&i| model.endo_names[i].clone()).collect();
    let histval_init;
    let init_given = init.is_some();
    let init = match init {
        Some(i) => i,
        None => {
            histval_init = histval_initial_conditions(model, &origs, histval)?;
            &histval_init
        }
    };
    let nobs = init.data.nrows();
    // the lagged variables need a value in the last period of the initial conditions
    for (i, l) in model.lead_lag_incidence.iter().enumerate().filter(|(i, _)| origs.contains(i)) {
        if l[0] != 0 {
            let nm = &model.endo_names[i];
            let j = init
                .col(nm)
                .ok_or_else(|| anyhow!("simul_backward_model: you have to initialize the endogenous variable {nm}"))?;
            if init.data[(nobs - 1, j)].is_nan() {
                bail!(
                    "simul_backward_model: variable {nm} should not have a NaN value in period {}.",
                    init.init.add(nobs as i64 - 1).label()
                );
            }
        }
    }
    // innovations
    let mut shocks = Mat::zeros(samplesize, nx);
    match innovations {
        Some(s) => {
            // the dates are compared only when the first argument is a dseries
            if init_given && s.init.diff(init.init.add(nobs as i64 - 1)) != Some(-1) {
                bail!("Time spans in first and third arguments should be contiguous!");
            }
            shocks = innovations_matrix(model, s, samplesize)?;
        }
        None => {
            use rand::SeedableRng;
            use rand_distr::{Distribution, StandardNormal};
            let pos: Vec<usize> = (0..nx).filter(|&k| sigma_e[(k, k)] > 0.0).collect();
            let cov = Mat::from_fn(pos.len(), pos.len(), |a, b| sigma_e[(pos[a], pos[b])]);
            let u = chol_upper(&cov)?;
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x424e_4c4d_535f_5345);
            let z = Mat::from_fn(samplesize, pos.len(), |_, _| StandardNormal.sample(&mut rng));
            let e = z * u;
            for (a, &k) in pos.iter().enumerate() {
                shocks.column_mut(k).copy_from(&e.column(a));
            }
        }
    }
    let total = nobs + samplesize;
    let mut y = Mat::from_element(n, total, f64::NAN);
    let mut x = Mat::from_element(total, nx, f64::NAN);
    for (i, nm) in model.endo_names.iter().enumerate() {
        if let Some(j) = init.col(nm) {
            for t in 0..nobs {
                y[(i, t)] = init.data[(t, j)];
            }
        }
    }
    for (k, nm) in model.exo_names.iter().enumerate() {
        if let Some(j) = init.col(nm) {
            for t in 0..nobs {
                x[(t, k)] = init.data[(t, j)];
            }
        }
    }
    x.rows_mut(nobs, samplesize).copy_from(&shocks);
    set_auxiliary_series(model, params, &mut y, &x, nobs)?;
    // period by period: dynare_solve on the dynamic residuals (`backward_model.dynamic_model`),
    // with options_.solve_algo, options_.simul.maxit and options_.dynatol.{f,x}
    let opts = crate::solver::SteadyOptions { solve_algo, maxit: 50, tolf: 1e-5, tolx: 1e-5, ..Default::default() };
    let ws = std::cell::RefCell::new(model.workspace());
    for it in nobs..total {
        let xt: Vec<f64> = x.row(it).iter().copied().collect();
        let ylag: Vec<f64> = (0..n).map(|i| y[(i, it - 1)]).collect();
        let f = |z: &[f64]| -> (Vec<f64>, Mat) {
            let mut y3 = ylag.clone();
            y3.extend_from_slice(z);
            y3.extend(std::iter::repeat_n(0.0, n));
            let mut res = vec![0.0; model.n_equations()];
            model.dynamic_residuals(&y3, &xt, params, &mut res);
            let mut jac = Mat::zeros(model.n_equations(), 3 * n + nx);
            model.dynamic_jacobian_y3_with(&y3, &xt, params, &mut ws.borrow_mut(), &mut jac);
            (res, jac.columns(n, n).into_owned())
        };
        let out = crate::solver::dynare_solve(&f, &ylag, &opts);
        if out.errorflag {
            bail!(
                "simul_backward_model: nonlinear solver routine failed with errorcode={} in period {}.",
                out.errorcode,
                init.init.add(it as i64).label()
            );
        }
        for i in 0..n {
            y[(i, it)] = out.x[i];
        }
    }
    let mut names = orig_names;
    names.extend(model.exo_names.iter().cloned());
    let mut data = Mat::zeros(total, orig + nx);
    for t in 0..total {
        for i in 0..orig {
            data[(t, i)] = y[(origs[i], t)];
        }
        for k in 0..nx {
            data[(t, orig + k)] = x[(t, k)];
        }
    }
    Ok(Series { init: init.init, names, data })
}
