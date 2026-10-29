// SPDX-License-Identifier: GPL-3.0-or-later
//! Conditional forecasts (stream C): port of `+conditional_forecasts/run.m`,
//! `get_shock_paths.m` and `plot.m` (Dynare 7.2), driven by the `conditional_forecast_paths`
//! block and the `conditional_forecast` / `plot_conditional_forecast` commands.
//!
//! Given paths for some endogenous variables over the first `cL` periods, the same number of
//! controlled shocks is solved for, period by period, so that the constrained variables hit
//! their paths (`get_shock_paths`); the other shocks are drawn from N(0, Sigma_e) `replic`
//! times. The unconditional forecast uses the same draws scheme with the controlled shocks set
//! to zero. Results: mean and `conf_sig` bands (order statistics, as Dynare) per variable and
//! period, plus the paths without free shocks (`*_path`, the limit of the Monte Carlo mean),
//! which are deterministic and match Dynare to rounding.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as _, StandardNormal};

use super::commands::{EstimationState, calibration_vector, graph_options, parameter_vector};
use super::{dsge_smoother, extend_dates, save_figures, simult_constant};
use crate::dsge::DsgeModel;
use crate::graphics::{Color, Figure, LineStyle, Panel, Series};
use crate::linalg::{Mat, chol_upper};
use crate::modfile::PathValue;
use crate::modfile::{ModFile, OptValue, Options, opt};
use crate::solver::DecisionRule;

/// `constrained_vars_` / `constrained_paths_` written by the preprocessor for a
/// `conditional_forecast_paths` block.
#[derive(Clone, Debug)]
pub struct ConstrainedPaths {
    /// Declaration-order indices of the constrained variables.
    pub vars: Vec<usize>,
    pub names: Vec<String>,
    /// nvars x cL levels, NaN where a variable is not constrained.
    pub paths: Mat,
}

impl ConstrainedPaths {
    /// `steady_state` is `oo_.steady_state` at the time of the command, for the values that
    /// mention endogenous variables (`values (gy_obs+0.01)`).
    pub fn from_block(
        mf: &ModFile,
        block: &[(String, Vec<(usize, usize, PathValue)>)],
        steady_state: &dyn Fn() -> Result<(Vec<f64>, Vec<f64>)>,
    ) -> Result<ConstrainedPaths> {
        let mut ss: Option<(Vec<f64>, Vec<f64>)> = None;
        let cl = block.iter().flat_map(|(_, r)| r.iter().map(|&(_, b, _)| b)).max().unwrap_or(0);
        let mut paths = Mat::from_element(block.len(), cl, f64::NAN);
        let mut vars = Vec::new();
        for (i, (name, ranges)) in block.iter().enumerate() {
            vars.push(
                mf.model
                    .endo_index(name)
                    .ok_or_else(|| anyhow!("conditional_forecast_paths: unknown variable {name}"))?,
            );
            for (a, b, v) in ranges {
                if ss.is_none() && matches!(v, PathValue::Steady(_)) {
                    ss = Some(steady_state()?);
                }
                let (y, x) = ss.as_ref().map_or((&[][..], &[][..]), |(y, x)| (y, x));
                let v = v.eval(&mf.model, y, x)?;
                for t in *a..=*b {
                    paths[(i, t - 1)] = v;
                }
            }
        }
        Ok(ConstrainedPaths { vars, names: block.iter().map(|(n, _)| n.clone()).collect(), paths })
    }
}

/// Mean, `conf_sig` band and the path without free shocks of one forecast object:
/// periods x variables.
#[derive(Clone, Debug)]
pub struct Distribution {
    pub mean: Mat,
    pub lower: Mat,
    pub upper: Mat,
    /// The same simulation with all free shocks at zero.
    pub path: Mat,
}

/// `oo_.conditional_forecast`.
#[derive(Clone, Debug)]
pub struct ConditionalForecast {
    /// `graph.title` ("Posterior Mode", "Calibration", ...).
    pub title: String,
    pub parameter_set: String,
    /// Declaration-order indices / names of the variables (all declared endogenous variables).
    pub vars: Vec<usize>,
    pub names: Vec<String>,
    /// (periods+1) x nvars, row 0 = initial state (last smoothed value or steady state).
    pub cond: Distribution,
    pub uncond: Distribution,
    /// Controlled shocks (`instruments`), indices and names.
    pub instruments: Vec<usize>,
    pub instrument_names: Vec<String>,
    /// cL x n_instruments values of the controlled shocks.
    pub exo: Distribution,
    pub constrained: ConstrainedPaths,
    pub periods: usize,
    pub replic: usize,
    pub conf_sig: f64,
    /// Dates of rows 0..=periods (row 0 = last sample date).
    pub dates: Vec<String>,
    /// Steady state, declaration order.
    pub ys: Vec<f64>,
}

impl ConditionalForecast {
    pub fn column(&self, var: usize) -> Option<usize> {
        self.vars.iter().position(|&v| v == var)
    }
}

/// Settings of one run of [`conditional_forecasts`].
pub struct CondSettings<'a> {
    pub dr: &'a DecisionRule,
    /// Levels added to the deviations (run.m's `trend`), declaration order: the steady state,
    /// its log under `loglinear`.
    pub ys: &'a [f64],
    /// `observation_trends` of the estimated model (run.m's `trend_addition(:, gend:end)` added
    /// to the constant of the observed variables): n_endo x (periods+1), declaration order,
    /// column 0 the last sample period; None without trends.
    pub trend: Option<&'a Mat>,
    /// Initial state, DR order, deviation from `ys` (`InitState`).
    pub init: &'a [f64],
    pub sigma_e: &'a Mat,
    pub constrained: &'a ConstrainedPaths,
    pub instruments: &'a [usize],
    pub periods: usize,
    pub replic: usize,
    pub conf_sig: f64,
    pub seed: u64,
    /// Variables reported (declaration order).
    pub vars: &'a [usize],
}

/// `sQ`: `sqrt(Sigma_e)` if diagonal, else `chol(Sigma_e, 'lower')`.
fn shock_factor(sigma_e: &Mat) -> Result<Mat> {
    let n = sigma_e.nrows();
    let diag = (0..n).all(|i| (0..n).all(|j| i == j || sigma_e[(i, j)] == 0.0));
    if diag {
        return Ok(Mat::from_fn(n, n, |i, j| if i == j { sigma_e[(i, i)].sqrt() } else { 0.0 }));
    }
    println!(
        "WARNING: conditional_forecast: the innovations are correlated; the results depend on the ordering of the \
         innovations (Cholesky factorization of Sigma_e)."
    );
    Ok(chol_upper(sigma_e)
        .ok_or_else(|| anyhow!("conditional_forecast: Sigma_e is not positive definite"))?
        .transpose())
}

/// Port of `conditional_forecasts.get_shock_paths` for one replication, in deviations, DR
/// order. `forcs` holds the initial state in column 0 and receives columns 1..=H; `shocks`
/// is nexo x H with the controlled rows at zero; returns the n1 x cL controlled shocks.
struct Paths<'a> {
    dr: &'a DecisionRule,
    k2: Vec<usize>,
    /// constrained rows (DR order)
    cv: Vec<usize>,
    instruments: &'a [usize],
    /// per constrained period: (finite rows k, inverse of R(cv_k, instr_k))
    blocks: Vec<(Vec<usize>, Mat)>,
    /// constrained values, deviations from the steady state, n1 x cL
    target: Mat,
}

impl Paths<'_> {
    fn tx(&self, x: &[f64], out: &mut [f64]) {
        let ghx = &self.dr.ghx;
        let s: Vec<f64> = self.k2.iter().map(|&k| x[k]).collect();
        for (i, o) in out.iter_mut().enumerate() {
            *o = (0..s.len()).map(|j| ghx[(i, j)] * s[j]).sum();
        }
    }
    fn add_ru(&self, e: &[f64], out: &mut [f64]) {
        let ghu = &self.dr.ghu;
        for (k, &ek) in e.iter().enumerate() {
            if ek != 0.0 {
                for (i, o) in out.iter_mut().enumerate() {
                    *o += ghu[(i, k)] * ek;
                }
            }
        }
    }
    /// One replication; `forcs`: (H+1) columns of length n (column-major Vec), `shocks`: H x nexo
    /// rows of draws. Returns the controlled shocks (cL x n1, row-major) when `cl > 0`.
    fn run(&self, cl: usize, forcs: &mut [Vec<f64>], shocks: &[Vec<f64>], e_out: &mut [f64]) {
        let n = self.dr.ghx.nrows();
        let n1 = self.cv.len();
        let h = forcs.len() - 1;
        let mut tf = vec![0.0; n];
        for t in 0..h {
            self.tx(&forcs[t], &mut tf);
            let mut e = shocks[t].clone();
            if t < cl {
                let (k, inv) = &self.blocks[t];
                if !k.is_empty() {
                    // rhs = mcValue(k,t) - T(cv_k,:) f_t - R(cv_k,:) shocks_t
                    let mut ru = vec![0.0; n];
                    self.add_ru(&shocks[t], &mut ru);
                    let rhs: Vec<f64> =
                        k.iter().map(|&i| self.target[(i, t)] - tf[self.cv[i]] - ru[self.cv[i]]).collect();
                    for (a, &i) in k.iter().enumerate() {
                        let v: f64 = (0..k.len()).map(|b| inv[(a, b)] * rhs[b]).sum();
                        e[self.instruments[i]] += v;
                        e_out[t * n1 + i] = v;
                    }
                }
            }
            let mut next = tf.clone();
            self.add_ru(&e, &mut next);
            forcs[t + 1] = next;
        }
    }
}

/// Port of `conditional_forecasts.run` from the point where `InitState`, `T`, `R` and `trend`
/// are known (both the estimated-model and the calibration branches).
pub fn conditional_forecasts(s: &CondSettings) -> Result<(Distribution, Distribution, Distribution)> {
    let dr = s.dr;
    let n = dr.ghx.nrows();
    let nexo = dr.ghu.ncols();
    let c = s.constrained;
    let n1 = c.vars.len();
    if n1 != s.instruments.len() {
        bail!(
            "conditional_forecast: The number of constrained variables doesn't match the number of controlled shocks"
        );
    }
    let cl = c.paths.ncols();
    if cl > s.periods {
        bail!(
            "conditional_forecast: the constrained paths ({cl} periods) are longer than the forecast (periods = {})",
            s.periods
        );
    }
    let cv: Vec<usize> = c.vars.iter().map(|&v| dr.inv_order_var[v]).collect();
    // trend(v, t): the level of variable v in forecast period t (column 0 = InitState's period)
    let level = |v: usize, t: usize| s.ys[v] + s.trend.map_or(0.0, |tr| tr[(v, t)]);
    // constrained paths in deviations; "trend includes last actual data point and therefore we
    // need to start in period 2"
    let target = Mat::from_fn(n1, cl, |i, t| c.paths[(i, t)] - level(c.vars[i], t + 1));
    let mut blocks = Vec::with_capacity(cl);
    for t in 0..cl {
        let k: Vec<usize> = (0..n1).filter(|&i| c.paths[(i, t)].is_finite()).collect();
        let m = Mat::from_fn(k.len(), k.len(), |a, b| dr.ghu[(cv[k[a]], s.instruments[k[b]])]);
        // get_shock_paths.m: MATLAB's inv() of a singular matrix only warns and returns Inf; the
        // run then ends with 'controlling of variables was not successful'
        let inv = m.try_inverse().unwrap_or_else(|| Mat::from_element(k.len(), k.len(), f64::INFINITY));
        blocks.push((k, inv));
    }
    let k2: Vec<usize> = dr.state_var.iter().map(|&v| dr.inv_order_var[v]).collect();
    let p = Paths { dr, k2, cv: cv.clone(), instruments: s.instruments, blocks, target };
    let sq = shock_factor(s.sigma_e)?;
    let h = s.periods;
    let rows: Vec<usize> = s.vars.iter().map(|&v| dr.inv_order_var[v]).collect();
    let nv = s.vars.len();

    // one simulation -> (periods+1) x nv levels, and the controlled shocks
    let simulate = |cl_used: usize, shocks: &[Vec<f64>], e: &mut [f64]| -> Vec<f64> {
        let mut forcs = vec![vec![0.0; n]; h + 1];
        forcs[0] = s.init.to_vec();
        p.run(cl_used, &mut forcs, shocks, e);
        let mut out = vec![0.0; (h + 1) * nv];
        for t in 0..=h {
            for (j, &r) in rows.iter().enumerate() {
                out[t * nv + j] = forcs[t][r] + level(s.vars[j], t);
            }
        }
        out
    };
    let zero = vec![vec![0.0; nexo]; h];
    let mut e0 = vec![0.0; cl * n1];
    let cond_path = simulate(cl, &zero, &mut e0);
    let mut scratch = vec![0.0; cl * n1];
    let uncond_path = simulate(0, &zero, &mut scratch);

    let mut rng = ChaCha8Rng::seed_from_u64(s.seed);
    let draw = |rng: &mut ChaCha8Rng| -> Vec<Vec<f64>> {
        // shocks = sQ*randn(ExoSize, periods); shocks(controlled_varexo,:) = 0
        (0..h)
            .map(|_| {
                let z: Vec<f64> = (0..nexo).map(|_| StandardNormal.sample(rng)).collect();
                let mut col: Vec<f64> = (0..nexo).map(|i| (0..nexo).map(|j| sq[(i, j)] * z[j]).sum()).collect();
                for &k in s.instruments {
                    col[k] = 0.0;
                }
                col
            })
            .collect()
    };
    let replic = s.replic.max(1);
    let mut cond_all = Vec::with_capacity(replic);
    let mut exo_all = Vec::with_capacity(replic);
    let mut worst: f64 = 0.0;
    for _ in 0..replic {
        let shocks = draw(&mut rng);
        let mut e = vec![0.0; cl * n1];
        let y = simulate(cl, &shocks, &mut e);
        // run.m compares FORCS1(:, 1+t) with trend(:, t) + constrained_paths(:, t) (the paths
        // in deviations from trend(:, 1+t)): with a time-varying trend the check is off by one
        // trend increment, as in Dynare
        for (i, &v) in c.vars.iter().enumerate() {
            if let Some(j) = s.vars.iter().position(|&w| w == v) {
                for t in 0..cl {
                    if c.paths[(i, t)].is_finite() {
                        let want = level(v, t) + p.target[(i, t)];
                        worst = worst.max((y[(t + 1) * nv + j] - want).abs());
                    }
                }
            }
        }
        cond_all.push(y);
        exo_all.push(e);
    }
    if worst > 1e-4 {
        println!("\nconditional_forecasts: controlling of variables was not successful.");
        println!("This can be due to numerical imprecision (e.g. explosive simulations)");
        println!("or because the instrument(s) do not allow controlling the variable(s).");
    }
    let mut uncond_all = Vec::with_capacity(replic);
    for _ in 0..replic {
        let shocks = draw(&mut rng);
        uncond_all.push(simulate(0, &shocks, &mut scratch));
    }
    let tt = (1.0 - s.conf_sig) / 2.0;
    let t1 = ((replic as f64 * tt).round() as usize).max(1);
    let t2 = ((replic as f64 * (1.0 - tt)).round() as usize).min(replic);
    let summarize = |all: &[Vec<f64>], path: &[f64], nr: usize, nc: usize| -> Distribution {
        let mut d = Distribution {
            mean: Mat::zeros(nr, nc),
            lower: Mat::zeros(nr, nc),
            upper: Mat::zeros(nr, nc),
            path: Mat::from_fn(nr, nc, |t, j| path[t * nc + j]),
        };
        let mut buf = vec![0.0; all.len()];
        for t in 0..nr {
            for j in 0..nc {
                for (b, y) in all.iter().enumerate() {
                    buf[b] = y[t * nc + j];
                }
                d.mean[(t, j)] = buf.iter().sum::<f64>() / buf.len() as f64;
                buf.sort_by(|a, b| a.total_cmp(b));
                d.lower[(t, j)] = buf[t1 - 1];
                d.upper[(t, j)] = buf[t2 - 1];
            }
        }
        d
    };
    let cond = summarize(&cond_all, &cond_path, h + 1, nv);
    let uncond = summarize(&uncond_all, &uncond_path, h + 1, nv);
    let exo = summarize(&exo_all, &e0, cl, n1);
    Ok((cond, uncond, exo))
}

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

fn names_of(v: Option<&OptValue>) -> Vec<String> {
    match v {
        Some(OptValue::Str(s)) => vec![s.clone()],
        Some(OptValue::List(l)) => l.iter().flat_map(|x| names_of(Some(x))).collect(),
        _ => Vec::new(),
    }
}

/// Graph title of a parameter set (`conditional_forecasts.run`).
fn graph_title(set: &str) -> Result<&'static str> {
    Ok(match set {
        "posterior_mode" => "Posterior Mode",
        "posterior_mean" => "Posterior Mean",
        "posterior_median" => "Posterior Median",
        "mle_mode" => "ML Mode",
        "prior_mode" => "Prior Mode",
        "prior_mean" => "Prior Mean",
        "calibration" => "Calibration",
        _ => bail!(
            "conditional_forecast: parameter_set must be calibration, posterior_mode, posterior_mean, posterior_median, \
             prior_mode or prior_mean (got {set})"
        ),
    })
}

/// `conditional_forecast(parameter_set=, controlled_varexo=(...), periods=, replic=, conf_sig=);`
/// Writes `<fname>_conditional_forecast.csv` (all declared endogenous variables) and
/// `<fname>_conditional_forecast_shocks.csv` (controlled shocks); returns `oo_.conditional_forecast`.
#[allow(clippy::too_many_arguments)]
pub fn conditional_forecast_command(
    mf: &ModFile,
    st: Option<&EstimationState>,
    o: &Options,
    paths: Option<&ConstrainedPaths>,
    seed: u64,
    out_dir: &Path,
    fname: &str,
) -> Result<ConditionalForecast> {
    let t0 = std::time::Instant::now();
    let paths =
        paths.ok_or_else(|| anyhow!("conditional_forecast: no conditional_forecast_paths block before the command"))?;
    let set = match opt(o, "parameter_set") {
        Some(OptValue::Str(s)) => s.clone(),
        // run.m: posterior_mode if oo_.posterior_mode exists, else mle_mode
        _ if st.is_some_and(|s| s.dsge.ml) => "mle_mode".to_string(),
        _ if st.is_some() => "posterior_mode".to_string(),
        _ => bail!(
            "conditional_forecast: No valid parameter set found (set parameter_set=calibration without an estimation)"
        ),
    };
    let title = graph_title(&set)?;
    let periods = num(o, "periods").map_or(40, |v| v as usize);
    let replic = num(o, "replic").map_or(5000, |v| v as usize);
    let conf_sig = num(o, "conf_sig").unwrap_or(0.8);
    let instrument_names = names_of(opt(o, "controlled_varexo"));
    let instruments = instrument_names
        .iter()
        .map(|s| {
            mf.model.exo_index(s).ok_or_else(|| {
                anyhow!("conditional_forecast: Declared controlled shock {s} is not a shock in the model.")
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let own;
    // `trend` of run.m (declaration order): the levels added to the deviations
    let (resolved, init, dates, trend, trend_addition) = if set == "calibration" {
        // estimated_model = 0: current M_.params (the estimation's point if any), start from
        // M_.endo_histval (e.g. set by smoother2histval), else from the steady state.
        let (dsge, x) = match st {
            Some(s) => (&s.dsge, s.current().to_vec()),
            None => {
                own = DsgeModel::new(mf, None)?;
                let x = calibration_vector(&own)?;
                (&own, x)
            }
        };
        let r = dsge
            .resolve(&x)
            .map_err(|e| anyhow!("conditional_forecast: no solution at the calibration (info {})", e.info))?;
        let n = r.dr.ghx.nrows();
        // ys = log_variable(1:endo_nbr, dr.ys, M_) under loglinear (logged_steady_state)
        let trend = simult_constant(dsge, &r.steady_state);
        // InitState = y0(order_var) - ys(order_var), y0 = M_.endo_histval (levels); under
        // loglinear only the nonzero entries of M_.endo_histval are logged, the others stay 0
        let init = match crate::analysis::stochastic_histval_of(&mf.model, &mf.histval) {
            Some(h) => {
                let y0: Vec<f64> = if dsge.loglinear {
                    let lv = crate::analysis::q2::log_variable(&mf.model, &h);
                    h.iter().zip(&lv).map(|(&v, &l)| if v != 0.0 { l } else { 0.0 }).collect()
                } else {
                    h
                };
                (0..n).map(|k| y0[r.dr.order_var[k]] - trend[r.dr.order_var[k]]).collect()
            }
            None => vec![0.0; n],
        };
        (r, init, (0..=periods).map(|h| h.to_string()).collect::<Vec<_>>(), trend, None)
    } else {
        let s = st.ok_or_else(|| anyhow!("conditional_forecast: parameter_set = {set} needs a previous estimation"))?;
        if s.data.prefilter_mean.is_some() {
            bail!("conditional_forecast: Conditional forecasting does not support the prefiltering option");
        }
        let x = parameter_vector(st, &s.dsge, &set)?;
        let sm = dsge_smoother(&s.dsge, &s.data, &x, false)?;
        let init = sm.last_state();
        let nobs = s.data.y.ncols();
        let mut dates = vec![
            s.dates.as_ref().and_then(|d| d.last().cloned()).unwrap_or_else(|| (s.first_obs - 1 + nobs).to_string()),
        ];
        dates.extend(extend_dates(s.dates.as_deref(), s.first_obs - 1 + nobs, periods));
        // constant = log(ys) under loglinear: every variable logged, exogenous aux variables
        // included (unlike log_variable in the calibration branch)
        let trend: Vec<f64> = sm.ys().iter().map(|&y| if s.dsge.loglinear { y.ln() } else { y }).collect();
        // trend part: trend_addition over gend+periods observations, kept from period gend on
        // (the last data point), added to the constant of the observed variables
        let trend_addition =
            s.dsge.trend_addition(&sm.resolved.params, s.data.first_obs, nobs + periods, false).map(|tr| {
                let mut m = Mat::zeros(trend.len(), periods + 1);
                for (i, &v) in s.dsge.varobs.iter().enumerate() {
                    for t in 0..=periods {
                        m[(v, t)] += tr[(i, nobs - 1 + t)];
                    }
                }
                m
            });
        (sm.resolved, init, dates, trend, trend_addition)
    };
    let n_orig = mf.model.orig_endo_nbr;
    let vars: Vec<usize> = (0..n_orig).collect();
    let settings = CondSettings {
        dr: &resolved.dr,
        ys: &trend,
        trend: trend_addition.as_ref(),
        init: &init,
        sigma_e: &resolved.q,
        constrained: paths,
        instruments: &instruments,
        periods,
        replic,
        conf_sig,
        seed,
        vars: &vars,
    };
    let (cond, uncond, exo) = conditional_forecasts(&settings)?;
    let cf = ConditionalForecast {
        title: title.to_string(),
        parameter_set: set,
        vars,
        names: mf.model.endo_names[..n_orig].to_vec(),
        cond,
        uncond,
        instruments,
        instrument_names,
        exo,
        constrained: paths.clone(),
        periods,
        replic,
        conf_sig,
        dates,
        ys: resolved.steady_state.clone(),
    };
    write_csv(&cf, out_dir, fname)?;
    print_summary(&cf);
    println!(
        "Conditional forecast ({}, {} replications, {:.0}% bands, {:.2}s) written to {}/{fname}_conditional_forecast*.csv",
        cf.title,
        replic,
        100.0 * conf_sig,
        t0.elapsed().as_secs_f64(),
        out_dir.display()
    );
    Ok(cf)
}

/// dynare-rust's summary table (Dynare prints nothing here): a progress note.
fn print_summary(cf: &ConditionalForecast) {
    use std::fmt::Write;
    let mut o = format!(
        "\nCONDITIONAL FORECAST ({}): mean of the constrained variables and of the controlled shocks\n{:<12}",
        cf.title, ""
    );
    let show = |n: &str| if n.len() > 14 { n[..14].to_string() } else { n.to_string() };
    for n in cf.constrained.names.iter().chain(&cf.instrument_names) {
        let _ = write!(o, " {:>14}", show(n));
    }
    for t in 1..=cf.constrained.paths.ncols() {
        let _ = write!(o, "\n{:<12}", cf.dates[t]);
        for &v in &cf.constrained.vars {
            let _ = write!(o, " {:>14.4}", cf.cond.mean[(t, cf.column(v).unwrap())]);
        }
        for j in 0..cf.instruments.len() {
            let _ = write!(o, " {:>14.4}", cf.exo.mean[(t - 1, j)]);
        }
    }
    crate::progress::note(o);
}

/// `<fname>_conditional_forecast.csv`: `date,horizon,variable,cond_mean,cond_lower,cond_upper,
/// uncond_mean,uncond_lower,uncond_upper,cond_path,uncond_path` (horizon 0 = initial state);
/// `<fname>_conditional_forecast_shocks.csv`: `date,horizon,shock,mean,lower,upper,path`.
pub fn write_csv(cf: &ConditionalForecast, out_dir: &Path, fname: &str) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let mut w = String::from(
        "date,horizon,variable,cond_mean,cond_lower,cond_upper,uncond_mean,uncond_lower,uncond_upper,cond_path,uncond_path\n",
    );
    for (j, name) in cf.names.iter().enumerate() {
        for t in 0..=cf.periods {
            let (c, u) = (&cf.cond, &cf.uncond);
            w += &format!(
                "{},{t},{name},{},{},{},{},{},{},{},{}\n",
                cf.dates[t],
                c.mean[(t, j)],
                c.lower[(t, j)],
                c.upper[(t, j)],
                u.mean[(t, j)],
                u.lower[(t, j)],
                u.upper[(t, j)],
                c.path[(t, j)],
                u.path[(t, j)]
            );
        }
    }
    let p = out_dir.join(format!("{fname}_conditional_forecast.csv"));
    std::fs::write(&p, w).with_context(|| format!("writing {}", p.display()))?;
    let mut w = String::from("date,horizon,shock,mean,lower,upper,path\n");
    for (j, name) in cf.instrument_names.iter().enumerate() {
        for t in 0..cf.exo.mean.nrows() {
            let e = &cf.exo;
            w += &format!(
                "{},{},{name},{},{},{},{}\n",
                cf.dates[t + 1],
                t + 1,
                e.mean[(t, j)],
                e.lower[(t, j)],
                e.upper[(t, j)],
                e.path[(t, j)]
            );
        }
    }
    let p = out_dir.join(format!("{fname}_conditional_forecast_shocks.csv"));
    std::fs::write(&p, w).with_context(|| format!("writing {}", p.display()))
}

/// Port of `conditional_forecasts.plot`'s `build_figure`: gray band of the conditional
/// forecast, conditional mean (solid), unconditional mean (dashed) and its band (thin dashed).
/// File `Conditional_forecast_<title>_<var>`.
pub fn conditional_forecast_figure(cf: &ConditionalForecast, var: usize, periods: usize) -> Option<Figure> {
    let j = cf.column(var)?;
    let h = periods.min(cf.periods + 1);
    let x: Vec<f64> = (1..=h).map(|t| t as f64).collect();
    let col = |m: &Mat| (0..h).map(|t| m[(t, j)]).collect::<Vec<f64>>();
    let line =
        |y: Vec<f64>, style, width| Series::Line { x: x.clone(), y, color: Color::Black, style, width, label: None };
    let name = &cf.names[j];
    let ticks: Vec<(f64, String)> = (0..h).map(|t| ((t + 1) as f64, cf.dates[t].clone())).collect();
    Some(Figure {
        name: format!("Conditional_forecast_{}_{name}", cf.title.trim().replace(' ', "_")),
        title: format!("Conditional forecast ({}): {name}.", cf.title),
        rows: 1,
        cols: 1,
        panels: vec![Panel {
            title: name.clone(),
            series: vec![
                Series::Band {
                    x: x.clone(),
                    lower: col(&cf.cond.lower),
                    upper: col(&cf.cond.upper),
                    color: Color::Gray,
                    alpha: 0.4,
                    label: None,
                },
                line(col(&cf.cond.mean), LineStyle::Solid, 3.0),
                line(col(&cf.uncond.mean), LineStyle::Dashed, 3.0),
                line(col(&cf.uncond.lower), LineStyle::Dashed, 1.0),
                line(col(&cf.uncond.upper), LineStyle::Dashed, 1.0),
            ],
            x_ticks: Some(ticks),
            ..Default::default()
        }],
    })
}

/// `plot_conditional_forecast(periods=) varlist;` (`conditional_forecasts.plot`).
pub fn plot_conditional_forecast_command(
    mf: &ModFile,
    st: Option<&EstimationState>,
    cf: Option<&ConditionalForecast>,
    o: &Options,
    vars: &[String],
    dir: &Path,
    fname: &str,
) -> Result<()> {
    let cf = cf.ok_or_else(|| anyhow!("Can't find conditional forecasts"))?;
    let mut periods = num(o, "periods").map_or(cf.periods + 1, |p| p as usize + 1);
    if periods > cf.periods + 1 {
        println!(
            "\nconditional_forecasts.plot:: Number of periods for plotting exceeds forecast horizon. Setting periods to {}.",
            cf.periods
        );
        periods = cf.periods + 1;
    }
    let list: Vec<String> = if vars.is_empty() { cf.constrained.names.clone() } else { vars.to_vec() };
    // conditional_forecasts.plot always saves its figures (no nograph option), in the
    // estimation's graph_format unless the command sets one
    let mut g = graph_options(o, dir, fname);
    if let Some(s) = st.filter(|_| opt(o, "graph_format").is_none()) {
        g.formats = s.graphs.formats.clone();
    }
    let mut figs = Vec::new();
    for v in &list {
        let idx = mf
            .model
            .endo_index(v)
            .ok_or_else(|| anyhow!("plot_conditional_forecast: {v} isn't an endogenous variable"))?;
        figs.push(
            conditional_forecast_figure(cf, idx, periods)
                .ok_or_else(|| anyhow!("plot_conditional_forecast: no forecast for {v}"))?,
        );
    }
    save_figures(&figs, fname, &g);
    println!("plot_conditional_forecast: {} figure(s) (Conditional_forecast_*)", figs.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shock_factor_diag_and_chol() {
        let d = Mat::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 9.0]);
        assert_eq!(shock_factor(&d).unwrap(), Mat::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]));
        let c = Mat::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 5.0]);
        let l = shock_factor(&c).unwrap();
        assert!((&l * l.transpose() - &c).abs().max() < 1e-14);
        assert_eq!(l[(0, 1)], 0.0);
    }
}
