//! OccBin and Kalman parity with Dynare 7.2 on paths without a MATLAB reference. The expected
//! values come from Dynare's own formulas or from the linear paths validated against Dynare:
//!
//! - an OccBin model whose constraint never binds: the piecewise Kalman filter is the linear
//!   filter, so `dsge_likelihood.m` with OccBin equals the linear likelihood (observation_trends,
//!   endogenous_prior, `lyapunov = fixed_point`) or the linear Kalman filter started from the
//!   state the OccBin branch starts from (`filter_initial_state`, lik_init 2 and 4: the OccBin
//!   update treats `a_0_given_tm1` and `Pstar` as period-0 updates and predicts them once more,
//!   `occbin_kalman_update0` in kalman_update_algo_1.m);
//! - the OccBin smoother (`occbin.DSGE_smoother`) on that model equals the linear smoother
//!   (`DsgeSmoother.m`: filter_initial_state, observation_trends);
//! - the univariate smoother with correlated measurement errors honours filter_initial_state;
//! - `DsgeSmoother.m` errors when the Riccati equation of lik_init = 4 cannot be solved.
mod common;

use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
use dynare::kalman::{KalmanOptions, KalmanWorkspace, MissingData, SmootherRequest, lyapunov_solver};
use dynare::linalg::Mat;
use dynare::modfile::{Command, ModFile, parse_mod_file};
use dynare::occbin::estimation::{OccbinEstimation, OccbinPosterior, smoother_series};
use dynare::occbin::{OccbinCommand, Session};

/// A ZLB New Keynesian model with habits (`y` has a lag and a lead: nboth = 1). `ILB` is the
/// lower bound (-1: the constraint never binds on the data below); `TREND` adds
/// observation_trends, `FIS` filter_initial_state, `SETUP` an occbin_setup line.
fn mod_text(ilb: &str, trend: bool, fis: bool, setup: &str) -> String {
    let mut s = format!(
        "var y pi i inot r_nat u_s;
varexo eps_d eps_s eps_m;
parameters beta sigma kappa phi_pi phi_y rho_i rho_d rho_s istar ilb hab;
beta = 0.99; sigma = 1; kappa = 0.1; phi_pi = 1.5; phi_y = 0.125; rho_i = 0.6;
rho_d = 0.85; rho_s = 0.5; istar = 1/beta - 1; ilb = {ilb}; hab = 0.6;
model;
[name='IS']
y = (hab*y(-1) + y(+1))/(1+hab) - 1/sigma*(i - pi(+1) - istar - r_nat);
[name='PC']
pi = beta*pi(+1) + kappa*y + u_s;
r_nat = rho_d*r_nat(-1) + eps_d;
u_s = rho_s*u_s(-1) + eps_s;
[name='notional rate']
inot = rho_i*i(-1) + (1-rho_i)*(istar + phi_pi*pi + phi_y*y) + eps_m;
[name='mp', relax='zlb']
i = inot;
[name='mp', bind='zlb']
i = ilb;
end;
occbin_constraints;
name 'zlb'; bind inot < ilb; relax inot > ilb;
end;
initval;
y = 0; pi = 0; i = istar; inot = istar; r_nat = 0; u_s = 0;
end;
steady;
shocks;
var eps_d; stderr 0.01;
var eps_s; stderr 0.002;
var eps_m; stderr 0.002;
end;
varobs y pi i;
shocks;
var i; stderr 0.0005;
end;
estimated_params;
kappa, gamma_pdf, 0.1, 0.03;
rho_d, beta_pdf, 0.85, 0.05;
stderr eps_d, inv_gamma_pdf, 0.01, inf;
end;
{setup}
"
    );
    if trend {
        s += "observation_trends;\ny (0.003);\nend;\n";
    }
    if fis {
        s += "filter_initial_state;\nr_nat(0) = 0.012;\nu_s(0) = -0.003;\ny(0) = 0.008;\ni(0) = 0.004;\nend;\n";
    }
    s
}

/// Kalman options with `riccati_tol = 0`: the linear filter never switches to the steady-state
/// gain (the OccBin filter has no such switch), so both run the same recursion.
fn kopts(lik_init: u8) -> KalmanOptions {
    KalmanOptions { lik_init, riccati_tol: 0.0, ..Default::default() }
}

/// Calibrated values of the estimated parameters (kappa, rho_d, stderr eps_d).
const X: [f64; 3] = [0.1, 0.85, 0.01];

/// Deterministic data (nobs x 3: y, pi, i); `low` pushes the policy rate to the bound.
fn data(nobs: usize, low: bool) -> Mat {
    let istar = 1.0 / 0.99 - 1.0;
    Mat::from_fn(nobs, 3, |t, j| {
        let t = t as f64;
        match j {
            0 => 0.01 * (0.7 * t).sin() + 0.004 * (1.9 * t).cos(),
            1 => 0.003 * (1.3 * t + 1.0).sin() + 0.001 * (0.4 * t).cos(),
            _ => {
                if low {
                    (istar + 0.018 * (0.25 * t + 0.5).cos() + 0.002 * (1.7 * t).sin()).max(0.0)
                } else {
                    istar + 0.004 * (0.5 * t).sin() + 0.001 * (2.3 * t).cos()
                }
            }
        }
    })
}

struct Setup {
    mf: ModFile,
    dsge: DsgeModel,
    data: LikelihoodData,
    occ: OccbinEstimation,
}

fn setup(tag: &str, text: &str, sample: &Mat, kalman: KalmanOptions) -> Setup {
    let dir = common::temp_dir(tag);
    let path = dir.join(format!("{tag}.mod"));
    std::fs::write(&path, text).unwrap();
    let mf = parse_mod_file(&path, &[]).unwrap();
    let mut s = Session::new(&mf, tag);
    for c in &mf.commands {
        if let Command::Occbin(OccbinCommand::Setup(o)) = c {
            s.setup(&mf, o).unwrap();
        }
    }
    let dsge = DsgeModel::new(&mf, Some(&X)).unwrap();
    let mut data = LikelihoodData::new(sample, kalman);
    data.filter_initial_state = dynare::estimation::q3::filter_initial_state(&mf).unwrap();
    let occ = OccbinEstimation::new(&mf, &s, &dsge, &data).unwrap().unwrap();
    Setup { mf, dsge, data, occ }
}

impl Setup {
    fn occbin_fval(&self) -> f64 {
        let post = OccbinPosterior { dsge: &self.dsge, data: &self.data, bounds: &self.dsge.bounds, occ: &self.occ };
        post.evaluate(&X).unwrap_or_else(|f| panic!("OccBin likelihood failed (info {})", f.info)).fval
    }

    fn linear_fval(&self) -> f64 {
        let post = DsgePosterior { dsge: &self.dsge, data: &self.data, bounds: &self.dsge.bounds };
        post.evaluate(&X).unwrap_or_else(|f| panic!("linear likelihood failed (info {})", f.info)).fval
    }

    /// The linear Kalman likelihood (minus the log prior) started from the prediction
    /// `a_pred`, `p_pred` of period 1.
    fn kalman_fval(&self, a_pred: Vec<f64>, p_pred: &Mat) -> f64 {
        self.kalman_fval_const(a_pred, p_pred, true)
    }

    /// As [`Self::kalman_fval`]; `demean = false` leaves the data as they are (the constant of
    /// the measurement equation is zero, `options_.noconstant`).
    fn kalman_fval_const(&self, a_pred: Vec<f64>, p_pred: &Mat, demean: bool) -> f64 {
        let res = self.dsge.resolve(&X).unwrap();
        let idx = &self.dsge.ss_index;
        let mut y = self.data.y.clone();
        for (i, &v) in idx.mfys.iter().enumerate() {
            for j in 0..y.ncols() {
                if demean {
                    y[(i, j)] -= res.steady_state[v];
                }
            }
        }
        let opts = KalmanOptions { lik_init: 1, ..self.data.kalman.clone() };
        let mut ws = KalmanWorkspace::new(res.t.nrows(), y.nrows(), res.q.nrows());
        ws.set_initial_state(Some(a_pred));
        let pinf = Mat::zeros(res.t.nrows(), res.t.nrows());
        let lik =
            ws.likelihood(&y, &self.data.missing, &idx.mf, &res.t, &res.r, &res.q, Some(&res.h), p_pred, &pinf, &opts);
        assert_eq!(lik.info, 0);
        lik.minus_loglik - self.dsge.log_prior(&X)
    }
}

fn rel(a: f64, b: f64) -> f64 {
    ((a - b) / b).abs()
}

fn max_abs_diff(a: &Mat, b: &Mat) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// filter_initial_state in the piecewise Kalman filter (dsge_likelihood.m:
/// `a_0_given_tm1 = T*a`, then the OccBin update predicts `T*a_0_given_tm1`).
#[test]
fn pkf_filter_initial_state() {
    let s = setup("pko_fis", &mod_text("-1", false, true, ""), &data(40, false), kopts(1));
    let res = s.dsge.resolve(&X).unwrap();
    let a = dynare::estimation::q3::kalman_a0(&s.dsge, &res, &s.dsge.ss_index, &s.data.filter_initial_state);
    assert!(a.iter().any(|&v| v != 0.0));
    let t = &res.t;
    let a_pred: Vec<f64> = (t * Mat::from_column_slice(a.len(), 1, &a)).iter().copied().collect();
    let pstar = lyapunov_solver(t, &res.r, &res.q, 1.000001, 1e-15).unwrap();
    let (got, want) = (s.occbin_fval(), s.kalman_fval(a_pred, &pstar));
    println!("PKF with filter_initial_state: {got:.12} vs {want:.12}");
    assert!(rel(got, want) < 1e-10, "{got} vs {want}");
    // the initial state matters
    assert!(rel(got, s.kalman_fval(vec![0.0; a.len()], &pstar)) > 1e-6);
}

/// observation_trends: the PKF likelihood demeans with constant + trend_addition.
#[test]
fn pkf_observation_trends() {
    let mut y = data(40, false);
    for t in 0..40 {
        y[(t, 0)] += 0.003 * (t + 1) as f64;
    }
    let s = setup("pko_trend", &mod_text("-1", true, false, ""), &y, kopts(1));
    let (got, want) = (s.occbin_fval(), s.linear_fval());
    println!("PKF with observation_trends: {got:.12} vs linear {want:.12}");
    assert!(rel(got, want) < 1e-10, "{got} vs {want}");
}

/// lyapunov = fixed_point (a loose tolerance, so that it differs from the Schur solution).
#[test]
fn pkf_lyapunov_fixed_point() {
    let mut k = kopts(1);
    k.lyapunov = dynare::kalman::lyapunov::LyapunovOptions { method: 1, fixed_point_tol: 1e-3, doubling_tol: 1e-16 };
    let s = setup("pko_lyap", &mod_text("-1", false, false, ""), &data(40, false), k.clone());
    let res = s.dsge.resolve(&X).unwrap();
    let (t, r, q) = (&res.t, &res.r, &res.q);
    let mm = t.nrows();
    // Pstar = lyapunov_solver(T,R,Q,options_), predicted once more by the OccBin update
    let p_fp = dynare::kalman::lyapunov::lyapunov_solver_opts(t, r, q, &k).unwrap();
    let p_pred = t * &p_fp * t.transpose() + r * q * r.transpose();
    let p_schur = lyapunov_solver(t, r, q, 1.000001, 1e-15).unwrap();
    let (got, want) = (s.occbin_fval(), s.kalman_fval(vec![0.0; mm], &p_pred));
    let schur = s.kalman_fval(vec![0.0; mm], &p_schur);
    println!("PKF with lyapunov=fixed_point: {got:.12} vs {want:.12} (Schur {schur:.12})");
    assert!(rel(want, schur) > 1e-8);
    assert!(rel(got, want) < 1e-10, "{got} vs {want}");
}

/// endogenous_prior: fval = likelihood - lnprior - lnpriormom with OccBin too.
#[test]
fn pkf_endogenous_prior() {
    let mut s = setup("pko_endo", &mod_text("-1", false, false, ""), &data(40, false), kopts(1));
    let plain = s.occbin_fval();
    s.data.endogenous_prior = true;
    let (got, want) = (s.occbin_fval(), s.linear_fval());
    println!("PKF with endogenous_prior: {got:.12} vs linear {want:.12} (without: {plain:.12})");
    assert!(rel(got, plain) > 1e-6);
    assert!(rel(got, want) < 1e-10, "{got} vs {want}");
}

/// lik_init = 2 (`a_0_given_tm1 = a`, `Pstar = 10*I` as the period-0 covariance with
/// first_period_occbin_update = 1) and lik_init = 4 (Riccati steady state, `T*a`).
#[test]
fn pkf_lik_init_2_and_4() {
    for lik_init in [2u8, 4] {
        let k = kopts(lik_init);
        let s = setup(&format!("pko_li{lik_init}"), &mod_text("-1", false, true, ""), &data(40, false), k);
        let res = s.dsge.resolve(&X).unwrap();
        let idx = &s.dsge.ss_index;
        let (t, r, q) = (&res.t, &res.r, &res.q);
        let mm = t.nrows();
        let a = dynare::estimation::q3::kalman_a0(&s.dsge, &res, idx, &s.data.filter_initial_state); // T*a
        let a0 = Mat::from_column_slice(mm, 1, &a);
        let (a_given_0, p0) = if lik_init == 2 {
            // a itself: undo nothing, rebuild from the starting values
            let mut a_un = Mat::zeros(mm, 1);
            for &pos in &idx.restrict_columns {
                let endo = res.dr.order_var[idx.restrict_var_list[pos]];
                if let Some(&(_, v)) = s.data.filter_initial_state.iter().find(|(e, _)| *e == endo) {
                    a_un[pos] = v - res.steady_state[endo];
                }
            }
            (a_un, Mat::identity(mm, mm) * 10.0)
        } else {
            let p = dynare::kalman::lik_init::initial_pstar(4, t, r, q, Some(&res.h), &idx.mf, 1.000001, 1e-15, false)
                .unwrap();
            (a0, p)
        };
        let a_pred: Vec<f64> = (t * &a_given_0).iter().copied().collect();
        let p_pred = t * &p0 * t.transpose() + r * q * r.transpose();
        let (got, want) = (s.occbin_fval(), s.kalman_fval(a_pred, &p_pred));
        println!("PKF with lik_init = {lik_init}: {got:.12} vs {want:.12}");
        assert!(rel(got, want) < 1e-10, "lik_init {lik_init}: {got} vs {want}");
    }
}

/// The OccBin smoother with filter_initial_state and observation_trends equals the linear
/// smoother (`DsgeSmoother.m` honours both) when the constraint never binds.
#[test]
fn occbin_smoother_filter_initial_state_and_trends() {
    for (tag, trend, fis) in [("pks_fis", false, true), ("pks_trend", true, false), ("pks_both", true, true)] {
        let mut y = data(40, false);
        if trend {
            for t in 0..40 {
                y[(t, 0)] += 0.003 * (t + 1) as f64;
            }
        }
        let s = setup(tag, &mod_text("-1", trend, fis, ""), &y, kopts(1));
        let dir = common::temp_dir(&format!("{tag}_out"));
        let (occ, rh) = smoother_series(&s.dsge, &s.data, &s.occ, &X, &dir, tag).unwrap();
        assert!(rh.iter().all(|g| g.iter().all(|r| r.regime.iter().all(|&b| !b))));
        let lin = dynare::driver::run_smoother(&s.dsge, &s.data, &X, false).unwrap();
        let (ds, de) = (max_abs_diff(&occ.smoothed, &lin.smoothed), max_abs_diff(&occ.shocks, &lin.shocks));
        println!("{tag}: |d smoothed| {ds:.1e}, |d shocks| {de:.1e}");
        assert!(ds < 1e-10 && de < 1e-10, "{tag}");
        if !trend {
            let df = max_abs_diff(&occ.filtered, &lin.filtered);
            assert!(df < 1e-10, "{tag}: filtered {df:.1e}");
        }
        let me = max_abs_diff(occ.measurement_errors.as_ref().unwrap(), lin.measurement_errors.as_ref().unwrap());
        assert!(me < 1e-10, "{tag}: measurement errors {me:.1e}");
        let _ = &s.mf;
    }
}

/// The univariate smoother with a non-diagonal H (state vector augmented with the
/// measurement errors) starts from filter_initial_state like the multivariate one.
#[test]
fn univariate_smoother_correlated_me_initial_state() {
    let t = Mat::from_row_slice(3, 3, &[0.9, 0.1, 0.0, 0.0, 0.5, 0.2, 0.0, 0.0, 0.3]);
    let r = Mat::from_row_slice(3, 2, &[1.0, 0.0, 0.3, 1.0, 0.0, 0.5]);
    let q = Mat::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.02]);
    let h = Mat::from_row_slice(2, 2, &[0.01, 0.004, 0.004, 0.02]);
    let mf = [0usize, 2];
    let nobs = 25;
    let y = Mat::from_fn(2, nobs, |i, s| 0.1 * ((s as f64) * (0.6 + 0.5 * i as f64)).sin());
    let missing = MissingData::from_data(&y);
    let pstar = lyapunov_solver(&t, &r, &q, 1.000001, 1e-15).unwrap();
    let req = SmootherRequest { a_initial: Some(vec![0.5, -0.3, 0.2]), ..Default::default() };
    let run = |kalman_algo: u8| {
        let o = KalmanOptions { kalman_algo, ..kopts(1) };
        dynare::kalman::smoother_request(
            &y,
            &missing,
            &mf,
            &t,
            &r,
            &q,
            Some(&h),
            &pstar,
            None,
            &o,
            false,
            None,
            Some(&req),
        )
        .unwrap()
    };
    let (multi, uni) = (run(1), run(2));
    assert!(uni.univariate);
    let d = max_abs_diff(&multi.alphahat, &uni.alphahat).max(max_abs_diff(&multi.etahat, &uni.etahat));
    println!("univariate vs multivariate smoother with filter_initial_state: {d:.1e}");
    assert!(d < 1e-10, "{d:.1e}");
}

/// `DsgeSmoother.m` has no fallback when the Riccati equation of lik_init = 4 has no
/// solution (an unobserved unit root); `dsge_likelihood.m` switches to lik_init = 1.
#[test]
fn lik_init_4_smoother_has_no_fallback() {
    let t = Mat::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.5]);
    let r = Mat::identity(2, 2);
    let q = Mat::identity(2, 2) * 0.01;
    let lik = dynare::kalman::lik_init::initial_pstar(4, &t, &r, &q, None, &[1], 1.000001, 1e-15, false);
    let smo = dynare::kalman::lik_init::initial_pstar(4, &t, &r, &q, None, &[1], 1.000001, 1e-15, true);
    println!("likelihood: {:?}; smoother: {:?}", lik.as_ref().map(|_| ()), smo.as_ref().map(|_| ()));
    assert!(lik.is_ok());
    assert!(smo.is_err());
}

/// `noconstant`: dsge_likelihood.m:204 sets the constant of the measurement equation to zero,
/// so the PKF likelihood does not demean the data by the steady state of the observables.
#[test]
fn pkf_noconstant() {
    let k = KalmanOptions { noconstant: true, ..kopts(1) };
    let s = setup("pko_nocst", &mod_text("-1", false, false, ""), &data(40, false), k);
    let res = s.dsge.resolve(&X).unwrap();
    let pstar = lyapunov_solver(&res.t, &res.r, &res.q, 1.000001, 1e-15).unwrap();
    let zero = vec![0.0; res.t.nrows()];
    let (got, want) = (s.occbin_fval(), s.kalman_fval_const(zero.clone(), &pstar, false));
    println!("PKF with noconstant: {got:.12} vs {want:.12}");
    assert!(rel(got, want) < 1e-10, "{got} vs {want}");
    // the steady state of i is istar = 1/beta - 1: demeaning matters
    assert!(rel(got, s.kalman_fval_const(zero, &pstar, true)) > 1e-6);
}
