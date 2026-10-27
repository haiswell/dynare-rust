// SPDX-License-Identifier: GPL-3.0-or-later
//! The OccBin smoother: ports of `+occbin/DSGE_smoother.m` and of the OccBin branches of
//! `kalman/DsgeSmoother.m` / `kalman/missing_DiffuseKalmanSmootherH3_Z.m` (Dynare 7.2,
//! `smoother_redux = false`, `lik_init` 1, 2, 4 or 5, no diffuse periods).
//!
//! 1. The realtime smoother: the univariate filter with OccBin updates
//!    (`kalman_update_engine` -> `kalman_update_algo_3`, regimes found period by period), then the
//!    univariate fixed-interval smoother on the resulting time-varying state space.
//! 2. The piecewise-linear model is simulated with the smoothed shocks from the smoothed initial
//!    state; if the simulated regimes differ from the filter's, a conditional smoother (initial
//!    state known) is tried, then the smoother is iterated on the time-varying state space of the
//!    simulated regimes until the regimes are a fixed point (or loop: "periodic").
//!
//! The state space is the full one in decision-rule order (`bayestopt_.smoother_var_list`
//! with all endogenous variables).

use anyhow::{Result, anyhow, bail};
use nalgebra::DVector;

use super::estimation::{OccbinEstimation, kalman_starting_values};
use super::pkf::{self, PkfOptions, PointModel, Univariate, UpdateInput};
use super::regimes::{self, Regimes};
use super::solver::{SimulOptions, SolveOutput};
use crate::dsge::{DsgeModel, LikelihoodData};
use crate::kalman::lik_init::{HARVEY_SCALE_FACTOR, initial_pstar};
use crate::kalman::lyapunov::lyapunov_solver_opts;
use crate::linalg::Mat;
use crate::solver::{StateSpaceIndex, kalman_transition_matrix};

type Vector = DVector<f64>;

/// One pass of `missing_DiffuseKalmanSmootherH3_Z` (univariate, d = 0).
struct Pass {
    /// Smoothed states (DR order, deviations), mm x nobs.
    alphahat: Mat,
    /// Smoothed shocks, n_exo x nobs.
    etahat: Mat,
    /// Updated states a_{t|t}, mm x nobs.
    ahat: Mat,
    /// One-step-ahead predictions a_{t+1|t} (`aK(1,:,t+1)`), mm x nobs.
    filtered: Mat,
    alphahat0: Vector,
    /// Filter regimes (realtime passes), periods 1..nobs+1.
    regimes: Vec<Regimes>,
}

/// The time-varying state space of a regime path (`ss.T(order_var, order_var, :)`...).
#[derive(Clone)]
struct TimeVarying {
    tt: Vec<Mat>,
    rr: Vec<Mat>,
    cc: Vec<Vector>,
}

struct Ctx<'a, 'b> {
    pm: PointModel<'a>,
    y: &'b Mat,
    data_index: &'b [Vec<usize>],
    mf: Vec<usize>,
    q: Mat,
    h: Mat,
    kalman_tol: f64,
    pkf: PkfOptions,
}

/// `missing_DiffuseKalmanSmootherH3_Z` with OccBin: `fixed = None` runs the OccBin updates from
/// period 1 (`first_period_occbin_update = 1`; `regimes_init` = `opts_regime.regime_history`,
/// empty for the base regime), `Some` the plain univariate smoother on given time-varying
/// matrices (`first_period_occbin_update = Inf`). `a_init`, `pstar`: a_{0|0} and P_{0|0}.
fn h3z(
    c: &mut Ctx,
    a_init: &Vector,
    pstar: &Mat,
    regimes_init: &[Regimes],
    fixed: Option<&TimeVarying>,
) -> Result<Pass, i32> {
    let (pp, smpl) = (c.y.nrows(), c.y.ncols());
    let mm = a_init.len();
    let nc = c.pm.nc;
    let tol = c.kalman_tol;
    let q = c.q.clone();
    let (mut tt, mut rr, mut cc) = match fixed {
        Some(f) => (f.tt.clone(), f.rr.clone(), f.cc.clone()),
        None => (vec![c.pm.t.clone(); smpl + 1], vec![c.pm.r.clone(); smpl + 1], vec![Vector::zeros(mm); smpl + 1]),
    };
    // check_regimes in regime-only mode: [rh(1) rh] or the base regime
    let mut regimes_: Vec<Regimes> = if regimes_init.is_empty() {
        vec![regimes::base(nc); smpl + 1]
    } else {
        let mut v = vec![regimes_init[0].clone()];
        v.extend(regimes_init.iter().cloned());
        v
    };
    let mut a = vec![Vector::zeros(mm); smpl];
    let mut a1 = vec![Vector::zeros(mm); smpl + 1];
    let mut p = vec![Mat::zeros(mm, mm); smpl + 1];
    let mut p1 = vec![Mat::zeros(mm, mm); smpl + 1];
    let mut v = vec![Vector::zeros(pp); smpl];
    let mut fi = vec![Vector::zeros(pp); smpl];
    let mut ki = vec![Mat::zeros(mm, pp); smpl];
    a[0] = a_init.clone();
    a1[0] = a_init.clone();
    let ainit = a_init.clone();
    let pinit = pstar.clone();
    p[0] = pstar.clone();
    let empty: Vec<usize> = Vec::new();
    let zero_v = Vector::zeros(pp);
    let zero_k = Mat::zeros(mm, pp);
    let zero_c = Vector::zeros(mm);
    let (base_t, base_r) = (c.pm.t.clone(), c.pm.r.clone());
    for t in 1..=smpl {
        a[t - 1] = a1[t - 1].clone();
        p1[t - 1] = p[t - 1].clone();
        let di = &c.data_index[t - 1];
        if fixed.is_none() {
            let yt = Vector::from_fn(pp, |i, _| c.y[(i, t - 1)]);
            let x = if t == 1 {
                let inp = UpdateInput {
                    a0: &a[0],
                    a1_0: &a[0],
                    p0: &p[0],
                    p1_0: &p1[0],
                    data_index: [&empty, &c.data_index[0]],
                    mf: &c.mf,
                    v0: &zero_v,
                    y: &yt,
                    h: &c.h,
                    q: &q,
                    tt: [&base_t, &tt[0]],
                    rr: [&base_r, &rr[0]],
                    cc: [&zero_c, &cc[0]],
                    uni: Some(Univariate { fi0: &zero_v, ki0: &zero_k, kalman_tol: tol }),
                };
                pkf::kalman_update_engine(&mut c.pm, &inp, t, &regimes_[0..2], &c.pkf)
            } else {
                let inp = UpdateInput {
                    a0: &a[t - 2],
                    a1_0: &a1[t - 2],
                    p0: &p[t - 2],
                    p1_0: &p1[t - 2],
                    data_index: [&c.data_index[t - 2], &c.data_index[t - 1]],
                    mf: &c.mf,
                    v0: &v[t - 2],
                    y: &yt,
                    h: &c.h,
                    q: &q,
                    tt: [&tt[t - 2], &tt[t - 1]],
                    rr: [&rr[t - 2], &rr[t - 1]],
                    cc: [&cc[t - 2], &cc[t - 1]],
                    uni: Some(Univariate { fi0: &fi[t - 2], ki0: &ki[t - 2], kalman_tol: tol }),
                };
                pkf::kalman_update_engine(&mut c.pm, &inp, t, &regimes_[t - 1..t + 1], &c.pkf)
            };
            if x.info != 0 {
                return Err(x.info);
            }
            if regimes_.len() < t + 2 {
                regimes_.resize(t + 2, regimes::base(nc));
            }
            for (k, r) in x.regimes.iter().enumerate().take(3) {
                regimes_[t - 1 + k] = r.clone();
            }
            a[t - 1] = x.a[0].clone();
            a1[t - 1] = x.a1[1].clone();
            a1[t] = x.a[1].clone();
            for &i in di {
                v[t - 1][i] = x.v[1][i];
                fi[t - 1][i] = x.fi[i];
                ki[t - 1].set_column(i, &x.ki.column(i));
            }
            tt[t - 1] = x.t[0].clone();
            tt[t] = x.t[1].clone();
            rr[t - 1] = x.r[0].clone();
            rr[t] = x.r[1].clone();
            cc[t - 1] = x.c[0].clone();
            cc[t] = x.c[1].clone();
            p[t - 1] = x.p[0].clone();
            p1[t - 1] = x.p1[1].clone();
            p[t] = x.p[1].clone();
        } else {
            if t == 1 {
                let qq = &rr[0] * &q * rr[0].transpose();
                a1[0] = &tt[0] * &a[0] + &cc[0];
                a[0] = a1[0].clone();
                p[0] = &tt[0] * &p[0] * tt[0].transpose() + qq;
                p1[0] = p[0].clone();
            }
            let mut at = a[t - 1].clone();
            let mut pt = p[t - 1].clone();
            for &i in di {
                let zi = c.mf[i];
                v[t - 1][i] = c.y[(i, t - 1)] - at[zi];
                fi[t - 1][i] = pt[(zi, zi)] + c.h[(i, i)];
                let k = pt.column(zi).into_owned();
                ki[t - 1].set_column(i, &k);
                if fi[t - 1][i] > tol {
                    at += &k * (v[t - 1][i] / fi[t - 1][i]);
                    pt -= &k * k.transpose() / fi[t - 1][i];
                    pt = (&pt + pt.transpose()) * 0.5; // Dynare 7.2
                }
            }
            a[t - 1] = at;
            p[t - 1] = pt;
            let qq = &rr[t] * &q * rr[t].transpose();
            a1[t] = &tt[t] * &a[t - 1] + &cc[t];
            p[t] = &tt[t] * &p[t - 1] * tt[t].transpose() + qq;
            p[t] = (&p[t] + p[t].transpose()) * 0.5; // Dynare 7.2
        }
    }
    p1[smpl] = p[smpl].clone();
    // backward pass
    let n_exo = q.nrows();
    let mut alphahat = Mat::zeros(mm, smpl);
    let mut etahat = Mat::zeros(n_exo, smpl);
    let mut ri = Vector::zeros(mm);
    for t in (1..=smpl).rev() {
        for &i in c.data_index[t - 1].iter().rev() {
            let f = fi[t - 1][i];
            if f > tol {
                let kr: f64 = (0..mm).map(|j| ki[t - 1][(j, i)] * ri[j]).sum();
                ri[c.mf[i]] += (v[t - 1][i] - kr) / f;
            }
        }
        alphahat.set_column(t - 1, &(&a1[t - 1] + &p1[t - 1] * &ri));
        etahat.set_column(t - 1, &(&q * rr[t - 1].transpose() * &ri));
        ri = tt[t - 1].transpose() * &ri;
    }
    let alphahat0 = &ainit + &pinit * &ri;
    let mut ahat = Mat::zeros(mm, smpl);
    let mut filtered = Mat::zeros(mm, smpl);
    for t in 0..smpl {
        ahat.set_column(t, &a[t]);
        filtered.set_column(t, &a1[t + 1]);
    }
    regimes_.truncate(smpl + 1);
    Ok(Pass { alphahat, etahat, ahat, filtered, alphahat0, regimes: regimes_ })
}

/// Result of [`occbin_smoother`] (`oo_.occbin.smoother`; states in DR order, deviations).
#[derive(Clone, Debug)]
pub struct OccbinSmoothed {
    pub alphahat: Mat,
    pub etahat: Mat,
    /// Updated states a_{t|t} of the pass that produced `alphahat`.
    pub ahat: Mat,
    /// One-step-ahead predictions of the realtime filter (`aK(1,:,t+1)`, DR order).
    pub filtered: Mat,
    pub alphahat0: Vector,
    /// Smoothed regimes, periods 1..nobs+1.
    pub regime_history: Vec<Regimes>,
    /// The realtime (filter) regimes.
    pub realtime_regime_history: Vec<Regimes>,
    /// `error_flag`, `warning_flag` (320 converged to a regime different from the realtime one,
    /// 321 periodic, 322 realtime results used, 323 fast).
    pub error_flag: i32,
    pub warning_flag: i32,
    /// Number of smoother passes on the time-varying state space (`size(regime_history_iter, 1)`).
    pub iterations: usize,
    /// The linear smoother (`oo_.occbin.linear_smoother`).
    pub linear_alphahat: Mat,
    pub linear_etahat: Mat,
    /// Steady state (declaration order) and the DR ordering, for levels.
    pub steady_state: Vec<f64>,
    /// observation_trends of the observables (`Trend`, varobs x nobs) and the trend one period
    /// ahead (`Trend + trend_coeff`, for the filtered variables), as in `store_smoother_results.m`.
    pub trend: Option<(Mat, Mat)>,
    pub order_var: Vec<usize>,
}

fn smoother_pkf_options(occ: &OccbinEstimation) -> PkfOptions {
    let mut o = occ.pkf.clone();
    o.simul = occ.smoother_simul.clone();
    o
}

/// The time-varying state space of a simulation (DR order, all periods).
fn time_varying(out: &SolveOutput, order_var: &[usize]) -> TimeVarying {
    let n = order_var.len();
    let mut tv = TimeVarying { tt: Vec::new(), rr: Vec::new(), cc: Vec::new() };
    for s in &out.ss {
        tv.tt.push(Mat::from_fn(n, n, |i, j| s.t[(order_var[i], order_var[j])]));
        tv.rr.push(Mat::from_fn(n, s.r.ncols(), |i, j| s.r[(order_var[i], j)]));
        tv.cc.push(Vector::from_fn(n, |i, _| s.c[order_var[i]]));
    }
    tv
}

/// Port of `occbin.DSGE_smoother` (with its `DsgeSmoother` calls) at parameter vector `x`.
pub fn occbin_smoother(
    dsge: &DsgeModel,
    data: &LikelihoodData,
    occ: &OccbinEstimation,
    x: &[f64],
) -> Result<OccbinSmoothed> {
    let opts = &data.kalman;
    if opts.lik_init == 3 {
        bail!("OccBin smoother: lik_init = 3 (diffuse_filter) is not ported");
    }
    // linear smoother (oo_.occbin.linear_smoother)
    let lin = crate::forecast::dsge_smoother(dsge, data, x, false)?;
    let res = lin.resolved.clone();
    let dr = &res.dr;
    let n = dr.order_var.len();
    let full = StateSpaceIndex::full(dr);
    let (t, r) = kalman_transition_matrix(dr, &full);
    let mf: Vec<usize> = dsge.varobs.iter().map(|&v| dr.inv_order_var[v]).collect();
    // data1 = Y - trend (constant + observation_trends)
    let mut y = data.y.clone();
    if !opts.noconstant {
        // (DsgeSmoother.m:155: the constant is zero with `noconstant`)
        for (i, &v) in dsge.varobs.iter().enumerate() {
            let c = res.steady_state[v];
            for j in 0..y.ncols() {
                y[(i, j)] -= c;
            }
        }
    }
    let trend = dsge.trend_addition(&res.params, data.first_obs, y.ncols(), false).map(|tr| {
        let ahead = dsge.trend_addition(&res.params, data.first_obs + 1, y.ncols(), false).unwrap();
        (tr, ahead)
    });
    if let Some((tr, _)) = &trend {
        y -= tr;
    }
    let h = res.h.clone();
    if (0..h.nrows()).any(|i| (0..h.ncols()).any(|j| i != j && h[(i, j)].abs() >= 1e-14)) {
        bail!("OccBin smoother: correlated measurement errors are not supported");
    }
    // DsgeSmoother.m: Pstar of the lik_init case (Pinf = [])
    let pstar = match opts.lik_init {
        1 => lyapunov_solver_opts(&t, &r, &res.q, opts),
        2 => Ok(Mat::identity(n, n) * HARVEY_SCALE_FACTOR),
        k => initial_pstar(k, &t, &r, &res.q, Some(&h), &mf, opts.qz_criterium, opts.lyapunov_complex_threshold, true),
    }
    .map_err(|e| anyhow!("OccBin smoother: {e}"))?;
    let order_var = dr.order_var.clone();
    let all: Vec<usize> = (0..n).collect();
    let pm = PointModel::new(
        &dsge.model,
        &occ.constraints,
        &occ.pswitch,
        res.params.clone(),
        res.steady_state.clone(),
        dr,
        &all,
        t.clone(),
        r.clone(),
    );
    let nc = pm.nc;
    let n_exo = res.q.nrows();
    let nobs = y.ncols();
    let mut c = Ctx {
        pm,
        y: &y,
        data_index: &data.missing.aindex,
        mf,
        q: res.q.clone(),
        h,
        kalman_tol: opts.kalman_tol,
        pkf: smoother_pkf_options(occ),
    };
    // a_initial = set_Kalman_smoother_starting_values(...) (filter_initial_state), not
    // predicted with the OccBin smoother
    let a_initial = kalman_starting_values(&res, &full, &data.filter_initial_state);
    let decl = |alpha: &Vector| -> Vec<f64> {
        let mut e = vec![0.0; n];
        for k in 0..n {
            e[order_var[k]] = alpha[k];
        }
        e
    };
    let shocks_of = |eta: &Mat| -> Mat {
        let mut s = Mat::zeros(nobs + 1, n_exo);
        for tt in 0..nobs {
            for k in 0..n_exo {
                s[(tt, k)] = eta[(k, tt)];
            }
        }
        s
    };
    // 1. realtime smoother
    let realtime = h3z(&mut c, &a_initial, &pstar, &[], None);
    let realtime_ok = realtime.is_ok();
    let filtered = realtime.as_ref().map_or_else(|_| lin.out.filtered.clone(), |p| p.filtered.clone());
    let (mut alphahat, mut etahat, mut ahat, mut alphahat0, mut regime_history) = match &realtime {
        Ok(p) => (p.alphahat.clone(), p.etahat.clone(), p.ahat.clone(), p.alphahat0.clone(), p.regimes.clone()),
        Err(_) => (
            Mat::zeros(n, nobs),
            lin.out.etahat.clone(),
            Mat::zeros(n, nobs),
            lin.out.alphahat0.clone(),
            vec![regimes::base(nc); nobs + 1],
        ),
    };
    let realtime_regime_history = regime_history.clone();
    let mut regime_history0: Vec<Vec<Regimes>> = vec![regime_history.clone()];
    let mut so: SimulOptions = occ.smoother_simul.clone();
    so.shocks = shocks_of(&etahat);
    so.exo_pos = (0..n_exo).collect();
    so.endo_init = decl(&alphahat0);
    so.init_regime = Some(regime_history.clone());
    so.periods = nobs + 1;
    let mut error_flag = 0;
    let mut warning_flag = 0;
    let mut last_sim_ok = true;
    // when the simulation fails, Dynare keeps the regimes of the last smoother run and the
    // partial (NaN) matrices of the solver, which no later step uses: error_flag = 321 rules
    // out the conditional smoother's convergence and the iterations below
    let mut tv = match c.pm.solve(&so) {
        Ok(out) => {
            regime_history = out.regime_history.clone();
            time_varying(&out, &order_var)
        }
        Err(_) => {
            last_sim_ok = false;
            error_flag = 321;
            TimeVarying { tt: Vec::new(), rr: Vec::new(), cc: Vec::new() }
        }
    };
    let mut is_changed = regime_history0[0] != regime_history;
    let mut iter = 1usize;
    let mut is_periodic = false;
    let mut is_changed_start = false;
    let maxiter = occ.smoother.max_number_of_iterations;
    let fast = occ.smoother.fast;
    let mut sto_tv = tv.clone();
    let mut cond_converged = false;
    let realtime_converged = !(is_changed || error_flag != 0);
    let sto_realtime = realtime_ok.then(|| {
        (alphahat.clone(), etahat.clone(), ahat.clone(), alphahat0.clone(), regime_history.clone(), so.clone())
    });
    // conditional smoother (initial state known: lik_init = 2, Harvey_scale_factor = 0); the
    // initial state M_.endo_initial_state.values(dr.state_var) covers all nspred states
    if !realtime_converged && !(opts.lik_init == 2 && HARVEY_SCALE_FACTOR == 0.0) {
        let mut a_init = Vector::zeros(n);
        for k in dr.nstatic..dr.nstatic + dr.nspred {
            a_init[k] = (alphahat0[k] + res.steady_state[order_var[k]]) - res.steady_state[order_var[k]];
        }
        let pzero = Mat::zeros(n, n);
        match h3z(&mut c, &a_init, &pzero, &regime_history, None) {
            Err(_) => error_flag = 322,
            Ok(cp) => {
                let rh = cp.regimes.clone();
                so.shocks = shocks_of(&cp.etahat);
                so.endo_init = decl(&alphahat0);
                error_flag = 0;
                let ctv;
                let sim_rh;
                match c.pm.solve(&so) {
                    Ok(o) => {
                        ctv = time_varying(&o, &order_var);
                        sim_rh = o.regime_history.clone();
                    }
                    Err(_) => {
                        error_flag = 321;
                        ctv = tv.clone();
                        sim_rh = rh.clone();
                    }
                }
                regime_history = sim_rh;
                is_changed = rh != regime_history;
                if !(is_changed || error_flag != 0) {
                    cond_converged = true;
                    tv = ctv;
                    alphahat = cp.alphahat;
                    etahat = cp.etahat;
                    ahat = cp.ahat;
                    alphahat0 = cp.alphahat0;
                }
            }
        }
    }
    // iterations on the time-varying state space of the simulated regimes
    while is_changed && maxiter > iter && !is_periodic && last_sim_ok {
        if iter == 1 {
            if let Some(s) = &sto_realtime {
                regime_history = s.4.clone();
            }
            error_flag = 0;
        }
        iter += 1;
        let pass = h3z(&mut c, &a_initial, &pstar, &[], Some(&tv))
            .map_err(|e| anyhow!("OccBin smoother: time-varying smoother failed ({e})"))?;
        alphahat = pass.alphahat;
        etahat = pass.etahat;
        ahat = pass.ahat;
        alphahat0 = pass.alphahat0;
        regime_history0.push(regime_history.clone());
        sto_tv = tv.clone();
        so.shocks = shocks_of(&etahat);
        so.endo_init = decl(&alphahat0);
        let out = match c.pm.solve(&so) {
            Ok(o) => o,
            Err(_) => {
                error_flag = 321;
                last_sim_ok = false;
                break;
            }
        };
        regime_history = out.regime_history.clone();
        tv = time_varying(&out, &order_var);
        let prev = &regime_history0[iter - 1];
        is_changed = *prev != regime_history;
        let mut changed_regime = false;
        is_changed_start = false;
        for (k, g) in prev.iter().enumerate() {
            for (j, gj) in g.iter().enumerate() {
                if let Some(h) = regime_history.get(k).and_then(|h| h.get(j)) {
                    changed_regime |= gj.regime != h.regime;
                    is_changed_start |= gj.start != h.start;
                }
            }
        }
        if fast {
            is_changed = changed_regime;
        }
        let mut is_tmp = false;
        for kiter in 0..iter - 1 {
            is_tmp = regime_history0[kiter] == regime_history;
            if is_tmp {
                break;
            }
        }
        is_periodic = is_changed && is_tmp;
    }
    regime_history0.push(regime_history.clone());
    let iterations = regime_history0.len();
    let mut final_regimes = regime_history.clone();
    let mut smoother_converged = false;
    if ((maxiter == iter || !last_sim_ok) && is_changed) || is_periodic || error_flag != 0 {
        if is_periodic {
            smoother_converged = true;
            error_flag = 0;
            warning_flag = 321;
        } else if !realtime_ok {
            error_flag = 322;
        }
    } else {
        smoother_converged = true;
        error_flag = 0;
        if !cond_converged && !realtime_converged {
            warning_flag = 320;
        } else if fast && is_changed_start {
            warning_flag = 323;
        }
    }
    if let Some(s) = &sto_realtime
        && !cond_converged
        && !realtime_converged
        && !smoother_converged
    {
        warning_flag = 322;
        error_flag = 0;
        final_regimes = realtime_regime_history.clone();
        is_changed = false;
        alphahat = s.0.clone();
        etahat = s.1.clone();
        ahat = s.2.clone();
        alphahat0 = s.3.clone();
    }
    if is_periodic && is_changed && error_flag == 0 {
        // periodic: the regimes of the last smoother pass (sto_TT)
        let _ = &sto_tv;
        final_regimes = regime_history0[regime_history0.len() - 2].clone();
    }
    Ok(OccbinSmoothed {
        alphahat,
        etahat,
        ahat,
        filtered,
        alphahat0,
        regime_history: final_regimes,
        realtime_regime_history,
        error_flag,
        warning_flag,
        iterations,
        linear_alphahat: lin.out.alphahat.clone(),
        linear_etahat: lin.out.etahat.clone(),
        steady_state: res.steady_state.clone(),
        trend,
        order_var,
    })
}
