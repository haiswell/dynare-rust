//! hetero-policy parity fixes: optimal policy commands (`src/policy`) and heterogeneity
//! (`src/heterogeneity`). Expected values come from independent computations on the decision
//! rules (the welfare of the linear-quadratic problem as a discounted sum of moments), from the
//! Dynare-validated references of reference/policy, and from the Dynare 7.2 sources named in
//! each test.
mod common;

use std::path::{Path, PathBuf};

use dynare::linalg::Mat;
use dynare::modfile::{ModFile, parse_mod_file};
use dynare::policy::{Session, handles};
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/policy")
}

fn refjson(name: &str) -> Value {
    common::json_ref(root().join(format!("ref_{name}.json")))
}

fn nums(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(nums).collect(),
        Value::Number(x) => vec![x.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

/// Parses `text` (written next to the reference models, so `@#include` works) and runs its
/// policy commands in order; the result of the last failing command is returned as an error.
fn run_text(tag: &str, text: &str) -> (ModFile, Session, anyhow::Result<()>) {
    let dir = common::temp_dir(tag);
    let f = dir.join(format!("{tag}.mod"));
    std::fs::write(&f, text).unwrap();
    let mf = parse_mod_file(&f, &[]).unwrap();
    let mut s = Session::new(true);
    let mut r = Ok(());
    for c in &mf.commands {
        if handles(&mf, c) {
            r = s.run(&mf, c, None);
            if r.is_err() {
                break;
            }
        }
    }
    (mf, s, r)
}

fn model_text(name: &str) -> String {
    std::fs::read_to_string(root().join(format!("{name}.mod"))).unwrap()
}

/// `gali_ramsey.mod` without its commands.
fn gali_ramsey_model() -> String {
    let t = model_text("gali_ramsey");
    t[..t.find("planner_objective").unwrap()].to_string()
}

const GALI_OBJ: &str =
    "planner_objective -(pi^2 + lambda*y_gap^2)/2;\nramsey_model(instruments=(i), planner_discount=beta);\n";

// ---------------------------------------------------------------------------------------
// the linear-quadratic welfare, computed independently from the decision rules
// ---------------------------------------------------------------------------------------

struct Lq {
    gy: Mat,  // n x ns (endogenous order)
    gu: Mat,  // n x nx
    gxs: Mat, // ns x ns
    gus: Mat, // ns x nx
    states: Vec<usize>,
    sigma: Mat,
    uyy: Mat, // n x n: Hessian of the planner objective
    beta: f64,
}

impl Lq {
    fn new(mf: &ModFile, s: &Session) -> Lq {
        let dr = s.dr.as_ref().unwrap();
        let m = &mf.model;
        let (n, nx, ns) = (m.n_endo(), m.n_exo(), dr.nspred);
        let mut gy = Mat::zeros(n, ns);
        let mut gu = Mat::zeros(n, nx);
        for (k, &v) in dr.order_var.iter().enumerate() {
            gy.set_row(v, &dr.ghx.row(k));
            gu.set_row(v, &dr.ghu.row(k));
        }
        let gxs = dr.ghx.rows(dr.nstatic, ns).into_owned();
        let gus = dr.ghu.rows(dr.nstatic, ns).into_owned();
        let states = (0..ns).map(|k| dr.order_var[dr.nstatic + k]).collect();
        let p = &m.params;
        let par = |n: &str| p[m.param_names.iter().position(|x| x == n).unwrap()];
        // U = -(pi^2 + lambda*y_gap^2)/2
        let mut uyy = Mat::zeros(n, n);
        uyy[(m.endo_index("pi").unwrap(), m.endo_index("pi").unwrap())] = -1.0;
        uyy[(m.endo_index("y_gap").unwrap(), m.endo_index("y_gap").unwrap())] = -par("lambda");
        Lq { gy, gu, gxs, gus, states, sigma: mf.sigma_e.clone(), uyy, beta: par("beta") }
    }

    /// Variance of the endogenous variables around the mean when the state variance is `vx`.
    fn quad_loss(&self, vx: &Mat) -> f64 {
        let vy = &self.gy * vx * self.gy.transpose() + &self.gu * &self.sigma * self.gu.transpose();
        0.5 * (&self.uyy * vy).trace()
    }

    /// `sum_k gxs^k q gxs'^k` by iteration.
    fn stationary(&self) -> Mat {
        let q = &self.gus * &self.sigma * self.gus.transpose();
        let mut v = q.clone();
        for _ in 0..4000 {
            v = &self.gxs * &v * self.gxs.transpose() + &q;
        }
        v
    }

    /// E sum_t beta^t U, started from the stationary distribution: U(ys)=0 here.
    fn unconditional(&self) -> f64 {
        self.quad_loss(&self.stationary()) / (1.0 - self.beta)
    }

    /// `Wyy = gy' Uyy gy + beta Gy' Wyy Gy` (Dynare's Lyapunov equation) by iteration, and the
    /// coefficients `c[e, s]` of `xhat_s u_e` in the welfare, `gu' Uyy gy + beta gus' Wyy gxs`.
    fn cross_coefficients(&self) -> Mat {
        let a = self.gy.transpose() * &self.uyy * &self.gy;
        let mut w = a.clone();
        for _ in 0..6000 {
            w = &a + self.beta * self.gxs.transpose() * &w * &self.gxs;
        }
        self.gu.transpose() * &self.uyy * &self.gy + self.beta * self.gus.transpose() * &w * &self.gxs
    }

    /// The same cross term as `A_times_B_kronecker_C(Wyu, yhat, u)` evaluates it in
    /// `evaluate_planner_objective.m`: `Wyu` is the row vector of `c` (exo-major, `nx*ns`
    /// entries) that is multiplied by `kron(yhat, u)` (state-major), so the entry
    /// `s*nx + e` of `c` flattened as `e'*ns + s'` pairs with `yhat_s u_e`.
    fn cross_term(&self, xhat: &[f64], u0: &[f64], matlab: bool) -> f64 {
        let c = self.cross_coefficients();
        let (ns, nx) = (xhat.len(), u0.len());
        let mut t = 0.0;
        for s in 0..ns {
            for e in 0..nx {
                let coef = if matlab {
                    let idx = s * nx + e;
                    c[(idx / ns, idx % ns)]
                } else {
                    c[(e, s)]
                };
                t += coef * xhat[s] * u0[e];
            }
        }
        t
    }

    /// Welfare at t = 0 given the previous state `xhat` (deviations of the state variables) and the
    /// first shock `u0`: the mean path plus the variance created by the shocks of t >= 1.
    fn conditional(&self, xhat: &[f64], u0: &[f64]) -> f64 {
        // the exact conditional welfare, with the cross term as MATLAB evaluates it
        self.conditional_exact(xhat, u0) + self.cross_term(xhat, u0, true) - self.cross_term(xhat, u0, false)
    }

    fn conditional_exact(&self, xhat: &[f64], u0: &[f64]) -> f64 {
        let ns = xhat.len();
        let mut x = Mat::from_column_slice(ns, 1, xhat);
        let u = Mat::from_column_slice(u0.len(), 1, u0);
        let q = &self.gus * &self.sigma * self.gus.transpose();
        let mut vx = Mat::zeros(ns, ns); // variance of x_t: zero at t = 1 (x_1 is known)
        let mut w = 0.0;
        // t = 0: the mean of y_0 = gy xhat + gu u0, no variance (the shock is known)
        let y0 = &self.gy * &x + &self.gu * &u;
        w += 0.5 * (y0.transpose() * &self.uyy * &y0)[(0, 0)];
        x = &self.gxs * &x + &self.gus * &u;
        let mut bt = self.beta;
        for _ in 1..6000 {
            let ym = &self.gy * &x;
            w += bt * (0.5 * (ym.transpose() * &self.uyy * &ym)[(0, 0)] + self.quad_loss(&vx));
            x = &self.gxs * &x;
            vx = &self.gxs * &vx * self.gxs.transpose() + &q;
            bt *= self.beta;
        }
        w
    }
}

/// hetero-policy-2: `stoch_simul` forces order 1 for a model with a zero Hessian only locally
/// (`options_ = options_old`), so `evaluate_planner_objective` takes the linear-quadratic branch
/// at `order = 2`: EW = (U + 0.5 Uyy.(gu Sigma gu' + gy Var gy')) / (1 - beta).
#[test]
fn lq_welfare_at_order_two_with_zero_hessian() {
    let text = format!(
        "{}{GALI_OBJ}stoch_simul(order=2, irf=0) pi y_gap i;\nevaluate_planner_objective;\n",
        gali_ramsey_model()
    );
    let (mf, s, r) = run_text("php_lq2", &text);
    r.unwrap();
    assert_eq!(s.order, 2, "options_.order is restored after stoch_simul");
    let lq = Lq::new(&mf, &s);
    let v = s.planner_objective_value.as_ref().unwrap();
    let want = lq.unconditional();
    assert!(want < -1e-9, "the loss is negative: {want}");
    close(v.unconditional, want, 1e-7, "unconditional welfare");
    let ns = s.dr.as_ref().unwrap().nspred;
    let cond = lq.conditional(&vec![0.0; ns], &[0.0, 0.0]);
    close(v.conditional_zero_initial_multiplier, cond, 1e-7, "conditional welfare (zero multipliers)");
    close(v.conditional_steady_initial_multiplier, cond, 1e-7, "conditional welfare (steady multipliers)");
}

/// hetero-policy-3: `get_initial_state` starts the conditional welfare from `M_.endo_histval`
/// (yhat = histval - ys for the state variables) and from the period-1 `M_.det_shocks`. With
/// histval alone or shocks alone the welfare is the exact discounted sum; with both,
/// `A_times_B_kronecker_C(Wyu, yhat, u)` pairs the entries of `Wyu` as MATLAB does (see
/// [`Lq::cross_term`]).
fn histval_case(tag: &str, hist: bool, shocks: bool) {
    let text = format!(
        "{}{GALI_OBJ}{}{}stoch_simul(order=2, irf=0) pi y_gap i;
evaluate_planner_objective;
",
        gali_ramsey_model(),
        if hist {
            "histval;
u(0) = 0.01;
a(0) = 0.02;
end;
"
        } else {
            ""
        },
        if shocks {
            "shocks;
var eps_u;
periods 1;
values 0.005;
end;
"
        } else {
            ""
        },
    );
    let (mf, s, r) = run_text(tag, &text);
    r.unwrap();
    let lq = Lq::new(&mf, &s);
    let m = &mf.model;
    let dr = s.dr.as_ref().unwrap();
    let mut h = vec![0.0; m.n_endo()];
    if hist {
        h[m.endo_index("u").unwrap()] = 0.01;
        h[m.endo_index("a").unwrap()] = 0.02;
    }
    let mults: Vec<usize> = (0..m.n_endo()).filter(|&j| m.endo_names[j].starts_with("mult_")).collect();
    // multipliers start at their steady state (0 here) or at zero: the same
    let xhat: Vec<f64> = lq.states.iter().map(|&j| if mults.contains(&j) { 0.0 } else { h[j] - dr.ys[j] }).collect();
    let mut u0 = vec![0.0; m.n_exo()];
    if shocks {
        u0[m.exo_names.iter().position(|e| e == "eps_u").unwrap()] = 0.005;
    }
    let want = lq.conditional(&xhat, &u0);
    let zero = lq.conditional(&vec![0.0; xhat.len()], &vec![0.0; u0.len()]);
    assert!((want - zero).abs() > 1e-9, "histval and shocks must matter");
    let v = s.planner_objective_value.as_ref().unwrap();
    close(v.conditional_zero_initial_multiplier, want, 1e-7, "conditional welfare");
    close(v.conditional_steady_initial_multiplier, want, 1e-7, "conditional welfare (steady multipliers)");
    // the unconditional welfare does not depend on the initial state
    close(v.unconditional, lq.unconditional(), 1e-7, "unconditional welfare");
}

#[test]
fn conditional_welfare_from_histval() {
    histval_case("php_hist", true, false);
}

#[test]
fn conditional_welfare_from_deterministic_shocks() {
    histval_case("php_detshock", false, true);
}

#[test]
fn conditional_welfare_from_histval_and_deterministic_shocks() {
    histval_case("php_hist_shock", true, true);
}

// ---------------------------------------------------------------------------------------
// OSR
// ---------------------------------------------------------------------------------------

fn gali_osr_text(extra: &str) -> String {
    model_text("gali_osr").replace(
        "osr(order=1, irf=0, opt_algo=@{OPT_ALGO}) pi y_gap i;",
        &format!("osr(order=1, irf=0{extra}) pi y_gap i;"),
    )
}

/// hetero-policy-1 (through the driver): after `osr`, `M_.params` holds the optimal osr
/// parameters, so the following `stoch_simul` solves the model at the optimum: its decision rule
/// is the reference one at the osr parameters (ref_gali_osr.json: ghx/ghu at the optimum).
#[test]
fn osr_parameters_reach_the_following_stoch_simul() {
    let dir = common::temp_dir("php_osr_driver");
    let text = format!("{}stoch_simul(order=1, irf=0, nomoments) i;\n", gali_osr_text(""));
    std::fs::write(dir.join("php_osr_driver.mod"), text).unwrap();
    dynare::driver::run(
        &dir.join("php_osr_driver.mod"),
        &dynare::driver::RunOptions { quiet: true, ..Default::default() },
    )
    .unwrap();
    let out: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("php_osr_driver/Output/php_osr_driver_stoch_simul.json")).unwrap(),
    )
    .unwrap();
    let r = refjson("gali_osr");
    let (a, b) = (nums(&out["dr"]["ghx"]), nums(&r["ghx"]));
    assert_eq!(a.len(), b.len());
    for (k, (x, y)) in a.iter().zip(&b).enumerate() {
        close(*x, *y, 1e-6, &format!("ghx[{k}]"));
    }
    let (a, b) = (nums(&out["dr"]["ghu"]), nums(&r["ghu"]));
    for (k, (x, y)) in a.iter().zip(&b).enumerate() {
        close(*x, *y, 1e-6, &format!("ghu[{k}]"));
    }
}

/// hetero-policy-4: `maxit` and `tolf` are appended to `optim_opt` as `MaxIter` and `TolFun`
/// (`+osr/run.m:104-122`), i.e. they equal the `optim` option; they stop csminwel early, and are an
/// error with any other `opt_algo`.
#[test]
fn osr_maxit_and_tolf_are_optimizer_options() {
    let run_osr = |tag: &str, extra: &str| {
        let (_mf, s, r) = run_text(tag, &gali_osr_text(extra));
        (s, r)
    };
    let (full, r) = run_osr("php_osr_full", "");
    r.unwrap();
    let full = full.osr.unwrap();
    let (a, r) = run_osr("php_osr_maxit", ", maxit=0");
    r.unwrap();
    let a = a.osr.unwrap();
    let (b, r) = run_osr("php_osr_optim", ", optim=('MaxIter', 0)");
    r.unwrap();
    let b = b.osr.unwrap();
    assert!(a.objective_function > full.objective_function * 1.0001, "stopped before convergence");
    assert_eq!(a.iterations, b.iterations);
    close(a.objective_function, b.objective_function, 1e-12, "maxit = optim MaxIter");
    // tolf = optim TolFun
    let (c, r) = run_osr("php_osr_tolf", ", tolf=1");
    r.unwrap();
    let (d, r) = run_osr("php_osr_tolf2", ", optim=('TolFun', 1)");
    r.unwrap();
    let (c, d) = (c.osr.unwrap(), d.osr.unwrap());
    assert_eq!(c.func_count, d.func_count);
    close(c.objective_function, d.objective_function, 1e-12, "tolf = optim TolFun");
    assert!(c.objective_function > full.objective_function, "a loose TolFun stops earlier");
    // not with another algorithm
    let (_s, r) = run_osr("php_osr_algo", ", maxit=1, opt_algo=7");
    let e = format!("{:#}", r.unwrap_err());
    assert!(e.contains("not supported when not using the default opt_algo=4"), "{e}");
}

// ---------------------------------------------------------------------------------------
// solver options, korder options, discretion
// ---------------------------------------------------------------------------------------

/// hetero-policy-5 and -13: the persistent `stoch_simul` options reach the first-order solver of
/// the Ramsey `stoch_simul`: with `qz_criterium = 0.7` the root 0.9 of the model counts as
/// explosive and Blanchard-Kahn fails, with the default criterium the same model solves.
#[test]
fn ramsey_stoch_simul_uses_the_persistent_qz_criterium() {
    let ok = format!("{}{GALI_OBJ}stoch_simul(order=1, irf=0) pi y_gap i;\n", gali_ramsey_model());
    let (_mf, _s, r) = run_text("php_qz_ok", &ok);
    r.unwrap();
    let bad = format!("{}{GALI_OBJ}stoch_simul(order=1, irf=0, qz_criterium=0.7) pi y_gap i;\n", gali_ramsey_model());
    let (_mf, _s, r) = run_text("php_qz_bad", &bad);
    assert!(r.is_err(), "qz_criterium = 0.7 must change the Blanchard-Kahn classification");
    // options set earlier (a `ramsey_model` option) persist into the later `stoch_simul`
    let early = format!(
        "{}planner_objective -(pi^2 + lambda*y_gap^2)/2;\nramsey_model(instruments=(i), planner_discount=beta, qz_criterium=0.7);\n\
stoch_simul(order=1, irf=0) pi y_gap i;\n",
        gali_ramsey_model()
    );
    let (_mf, _s, r) = run_text("php_qz_early", &early);
    assert!(r.is_err());
}

/// hetero-policy-6: the k-order options of a Ramsey/OSR `stoch_simul` come from the persistent
/// options like the driver's `stoch_simul` (pruning, replic, q2 options).
#[test]
fn ramsey_korder_options_keep_pruning_and_replic() {
    let text = format!("{}stoch_simul(order=2, pruning, replic=7, irf=0, periods=50) pi_ann log_h R_ann;\n", {
        let t = model_text("ramsey_cmr_o2");
        t[..t
            .find(
                "
ramsey_policy(",
            )
            .unwrap()
            + 1]
            .to_string()
            + "ramsey_model(instruments=(R), planner_discount=0.98);\n"
    });
    let (_mf, s, r) = run_text("php_korder", &text);
    r.unwrap();
    assert!(s.kdr.as_ref().expect("second-order rule").ghxx.nrows() > 0);
    assert!(s.ss_opts.pruning);
    let o = s.korder_options(&_mf, 2);
    assert!(o.pruning, "pruning must reach the k-order stoch_simul");
    assert_eq!(o.replic, 7);
}

/// hetero-policy-7: `discretionary_policy` evaluates `steady_state_model` (ys and the parameters)
/// and linearizes at that point. The model below is Gali's NK model with the inflation target
/// `pibar` in levels: `pidev = pi - pibar` is the loss variable and the instrument `i` has the
/// steady state `pibar`; the decision rules equal those of reference/policy/gali_discretion.
#[test]
fn discretion_with_steady_state_model() {
    let text = "var pi y_gap i r_nat a u pidev;\nvarexo eps_a eps_u;\n\
parameters beta sigma varphi alpha epsilon theta rho_a rho_u kappa psi lambda pibar;\n\
beta = 0.99; sigma = 1; varphi = 5; alpha = 0.25; epsilon = 9; theta = 0.75; rho_a = 0.9; rho_u = 0.5;\n\
kappa = (1-theta)*(1-beta*theta)/theta*(1-alpha)/(1-alpha+alpha*epsilon)*(sigma+(varphi+alpha)/(1-alpha));\n\
psi = (1+varphi)/(sigma*(1-alpha)+varphi+alpha);\nlambda = kappa/epsilon;\npibar = 0.02;\n\
model(linear);\n[name='NKPC']\npidev = beta*pidev(+1) + kappa*y_gap + u;\n[name='DIS']\n\
y_gap = y_gap(+1) - 1/sigma*(i - pi(+1) - r_nat);\n[name='natural rate']\nr_nat = -sigma*psi*(1-rho_a)*a;\n\
a = rho_a*a(-1) + eps_a;\nu = rho_u*u(-1) + eps_u;\n[name='def']\npidev = pi - pibar;\nend;\n\
steady_state_model;\npi = pibar;\ny_gap = 0;\ni = pibar;\nr_nat = 0;\na = 0;\nu = 0;\npidev = 0;\nend;\n\
shocks;\nvar eps_a; stderr 0.01;\nvar eps_u; stderr 0.005;\nend;\n\
planner_objective pidev^2 + lambda*y_gap^2;\n\
discretionary_policy(instruments=(i), irf=0, planner_discount=beta) pi y_gap i;\n";
    let (mf, s, r) = run_text("php_dp_ssm", text);
    r.unwrap();
    let dr = s.dr.as_ref().unwrap();
    let m = &mf.model;
    for (name, v) in [("pi", 0.02), ("i", 0.02), ("y_gap", 0.0), ("pidev", 0.0)] {
        close(dr.ys[m.endo_index(name).unwrap()], v, 1e-12, &format!("ys[{name}]"));
    }
    // compare with the reference decision rule of the model in deviations
    let r = refjson("gali_discretion");
    let names: Vec<String> =
        r["endo_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let rorder: Vec<usize> = nums(&r["order_var"]).iter().map(|&x| x as usize - 1).collect();
    let (nstat, ns_ref) = (r["nstatic"].as_u64().unwrap() as usize, r["nspred"].as_u64().unwrap() as usize);
    let rstate: Vec<usize> = rorder[nstat..nstat + ns_ref].to_vec();
    let rghx = nums(&r["ghx"]); // row-major as jsonencode gives rows, nrows = n
    let rghu = nums(&r["ghu"]);
    let nx = m.n_exo();
    for (k, &v) in rorder.iter().enumerate() {
        let name = &names[v];
        let row = dr.inv_order_var[m.endo_index(name).unwrap()];
        for (c, &sv) in rstate.iter().enumerate() {
            let col = (0..dr.nspred).position(|k| m.endo_names[dr.order_var[dr.nstatic + k]] == names[sv]).unwrap();
            close(dr.ghx[(row, col)], rghx[k * ns_ref + c], 1e-8, &format!("ghx[{name}, {}]", names[sv]));
        }
        for e in 0..nx {
            close(dr.ghu[(row, e)], rghu[k * nx + e], 1e-8, &format!("ghu[{name}, {e}]"));
        }
    }
}

/// hetero-policy-14: `discretionary_policy_1` starts the Dennis iteration from the persistent
/// `Hold` of the previous call: from the converged H a single iteration is enough, from zero it is
/// not.
#[test]
fn discretion_restarts_from_the_previous_solution() {
    use dynare::policy::discretion::{DiscretionOptions, discretionary_policy_1};
    let t = model_text("gali_discretion");
    let t = t[..t.find("discretionary_policy(").unwrap()].to_string()
        + "discretionary_policy(instruments=(i), irf=0, planner_discount=beta) pi y_gap i;\n\
discretionary_policy(instruments=(i), irf=0, planner_discount=beta) pi y_gap i;\n";
    let (mf, s, r) = run_text("php_hold", &t);
    r.unwrap();
    let h = s.hold.clone().expect("Hold is kept after the command");
    let p = s.params.clone().unwrap();
    let one = DiscretionOptions { maxit: 1, ..Default::default() };
    assert!(
        discretionary_policy_1(&mf.model, &p, &one, None, 1e-6, None).is_err(),
        "from H = 0 one step is not enough"
    );
    assert!(discretionary_policy_1(&mf.model, &p, &one, None, 1e-6, Some(&h)).is_ok(), "from the converged H it is");
}

/// hetero-policy-10: `discretionary_policy` with `loglinear` is an error
/// (`discretionary_policy_initialization.m:28-30`).
#[test]
fn discretion_rejects_loglinear() {
    let t = model_text("gali_discretion");
    let t = t.replace("irf=0, planner_discount=beta)", "irf=0, loglinear, planner_discount=beta)");
    let (_mf, _s, r) = run_text("php_dp_log", &t);
    let e = format!("{:#}", r.unwrap_err());
    assert!(e.contains("not compatible with `loglinear`"), "{e}");
}

/// hetero-policy-11: after `stoch_simul` at order 2 the initial shock of `get_initial_state` is
/// `oo_.exo_simul(1,:)` (the exogenous steady state), not nothing.
#[test]
fn exo_simul_first_row_is_set_at_order_two() {
    let (_mf, s, r) = run_text("php_exo", &model_text("ramsey_cmr_o2"));
    r.unwrap();
    assert_eq!(s.exo_simul_first, Some(vec![0.0]));
}

/// hetero-policy-8: `ramsey_constraints` are ignored (Dynare 7.2 never reads them in
/// `dyn_ramsey_static.m`); see also `ramsey_constraints_in_the_static_problem` in tests/policy.rs.
#[test]
fn ramsey_constraints_do_not_change_the_steady_state() {
    let src = model_text("ramsey_cmr_ssm");
    let (_m0, s0, r) = run_text("php_rc0", &src);
    r.unwrap();
    let text = src.replace("ramsey_model(", "ramsey_constraints;\nR > 0.02;\nend;\nramsey_model(");
    let (_m1, s1, r) = run_text("php_rc1", &text);
    r.unwrap();
    let (a, b) = (&s0.dr.as_ref().unwrap().ys, &s1.dr.as_ref().unwrap().ys);
    for (x, y) in a.iter().zip(b) {
        close(*x, *y, 1e-12, "steady state");
    }
}

// ---------------------------------------------------------------------------------------
// heterogeneity
// ---------------------------------------------------------------------------------------

/// The Krusell-Smith example of Dynare 7.2 up to its steady-state step; `tail` is appended.
fn ks_run(tag: &str, tail: &str) -> anyhow::Result<()> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/examples72/krusell_smith_1998");
    let dir = common::temp_dir(tag);
    std::fs::copy(src.join("krusell_smith_1998.mat"), dir.join("krusell_smith_1998.mat")).unwrap();
    let t = std::fs::read_to_string(src.join("krusell_smith_1998.mod")).unwrap();
    let t = &t[..t.find("heterogeneity_load_steady_state").unwrap()];
    let f = dir.join(format!("{tag}.mod"));
    std::fs::write(&f, format!("{t}{tail}")).unwrap();
    dynare::driver::run(&f, &dynare::driver::RunOptions { quiet: true, ..Default::default() })
}

/// hetero-policy-12: `options_.heterogeneity` persists. A second `heterogeneity_solve` without
/// `truncation_horizon` keeps the first value (the preprocessor writes only given options), so an
/// IRF horizon beyond it fails; steady-state options (`filename`) persist as well.
#[test]
fn heterogeneity_options_persist_between_commands() {
    let load = "heterogeneity_load_steady_state(filename = krusell_smith_1998);\n";
    let r = ks_run(
        "php_het_t30",
        &format!(
            "{load}heterogeneity_solve(truncation_horizon = 30);\nheterogeneity_solve;\nheterogeneity_simulate(irf = 50, nograph);\n"
        ),
    );
    let e = format!("{:#}", r.unwrap_err());
    assert!(e.contains("exceeds Jacobian truncation horizon 30"), "{e}");
    // the default horizon (300) is used when none was ever given
    ks_run("php_het_t300", &format!("{load}heterogeneity_solve;\nheterogeneity_simulate(irf = 50, nograph);\n"))
        .unwrap();
    // the second load reuses `filename`
    ks_run(
        "php_het_file",
        &format!("{load}heterogeneity_load_steady_state;\nheterogeneity_solve(truncation_horizon = 60);\n"),
    )
    .unwrap();
}

/// hetero-policy-9: a news-shock simulation starts from the steady state and refuses `histval`,
/// `endval` and `initval_series` (`+heterogeneity/simulate_news_shocks.m:64-75`).
#[test]
fn news_shock_simulation_refuses_histval_and_endval() {
    let solve = "heterogeneity_load_steady_state(filename = krusell_smith_1998);\nheterogeneity_solve(truncation_horizon = 60);\n";
    let news = "shocks;\nvar eps_Z; periods 2:6; values 0.01;\nend;\n";
    let sim = "heterogeneity_simulate(nograph);\n";
    ks_run("php_news_ok", &format!("{news}{solve}{sim}")).unwrap();
    let e = format!(
        "{:#}",
        ks_run("php_news_hist", &format!("{news}histval;\nK(0) = 30;\nend;\n{solve}{sim}")).unwrap_err()
    );
    assert!(e.contains("histval not supported for news shock simulation"), "{e}");
    let e = format!(
        "{:#}",
        ks_run(
            "php_news_end",
            &format!("{news}initval;\neps_Z = 0;\nend;\nendval;\neps_Z = 0.001;\nend;\n{solve}{sim}")
        )
        .unwrap_err()
    );
    assert!(e.contains("endval not supported for news shock simulation"), "{e}");
    // histval is fine for a stochastic (non-news) simulation
    ks_run(
        "php_stoch_hist",
        &format!("histval;\nK(0) = 30;\nend;\n{solve}heterogeneity_simulate(irf = 10, nograph);\n"),
    )
    .unwrap();
}

/// The Ramsey steady state and the first-order solution are computed at `oo_.exo_steady_state`
/// (`initval; e = 1; end;`): the AR(1) `a` has the steady state e/(1-rho) = 10, which a zero
/// exogenous steady state would give as 0.
#[test]
fn ramsey_uses_the_exogenous_steady_state() {
    let text = "var pi y i a;\nvarexo e;\nparameters beta kappa rho;\nbeta = 0.99; kappa = 0.1; rho = 0.9;\n\
model;\npi = beta*pi(+1) + kappa*y + a;\ny = y(+1) - (i - pi(+1));\na = rho*a(-1) + e;\nend;\n\
initval;\ne = 1;\nend;\nshocks;\nvar e; stderr 0.01;\nend;\n\
planner_objective -(pi^2 + 0.5*y^2)/2;\nramsey_model(instruments=(i), planner_discount=beta);\n\
stoch_simul(order=1, irf=0) pi y i;\n";
    let (mf, s, r) = run_text("php_exo_ss", text);
    r.unwrap();
    let dr = s.dr.as_ref().unwrap();
    close(dr.ys[mf.model.endo_index("a").unwrap()], 10.0, 1e-6, "steady state of a");
    assert_eq!(s.exo_simul_first, Some(vec![1.0]));
}
