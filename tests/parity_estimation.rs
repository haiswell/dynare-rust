//! Parity fixes in the estimation group: model_comparison without a marginal density
//! estimate, the TaRB block optimizer's `optim`, analytic gradients in csminwel
//! (`has_gradient`, method of moments with `analytic_jacobian`), `numgrad3_`/`numgrad5_`,
//! `InitialInverseHessian`, the slice sampler's convergence diagnostics, the DSGE-VAR
//! likelihood's Lyapunov options.
mod common;

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use dynare::driver::{RunOptions, run};
use dynare::estimation::optim::{MinimizeContext, OptimResult, minimize};
use dynare::estimation::{Bounds, Objective};
use dynare::modfile::OptValue;

fn s(v: &str) -> OptValue {
    OptValue::Str(format!("'{v}'"))
}

fn optim(pairs: &[(&str, OptValue)]) -> OptValue {
    OptValue::List(pairs.iter().flat_map(|(k, v)| [s(k), v.clone()]).collect())
}

fn run_minimize(obj: &dyn Objective, x0: &[f64], o: Option<&OptValue>) -> anyhow::Result<OptimResult> {
    let n = x0.len();
    let bounds = Bounds::unbounded(n);
    let names: Vec<String> = (0..n).map(|i| format!("p{i}")).collect();
    let sd = vec![f64::NAN; n];
    let ctx =
        MinimizeContext { bounds: &bounds, names: &names, prior_sd: &sd, optim: o, verbose: false, seed: 1, stream: 0 };
    minimize(4, obj, x0, &ctx)
}

/// Counts evaluations and records the points evaluated.
struct Counting<F: Fn(&[f64]) -> f64, G: Fn(&[f64]) -> Option<Vec<f64>>> {
    f: F,
    g: G,
    analytic: bool,
    evals: AtomicUsize,
    grads: AtomicUsize,
    points: Mutex<Vec<Vec<f64>>>,
}

impl<F: Fn(&[f64]) -> f64 + Sync, G: Fn(&[f64]) -> Option<Vec<f64>> + Sync> Objective for Counting<F, G> {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        self.evals.fetch_add(1, Ordering::SeqCst);
        self.points.lock().unwrap().push(x.to_vec());
        ((self.f)(x), 0)
    }
    fn gradient(&self, x: &[f64]) -> Option<Vec<f64>> {
        self.grads.fetch_add(1, Ordering::SeqCst);
        (self.g)(x)
    }
    fn has_gradient(&self) -> bool {
        self.analytic
    }
}

fn counting<F: Fn(&[f64]) -> f64, G: Fn(&[f64]) -> Option<Vec<f64>>>(f: F, g: G, analytic: bool) -> Counting<F, G> {
    Counting { f, g, analytic, evals: 0.into(), grads: 0.into(), points: Mutex::new(Vec::new()) }
}

// ---- estimation-2: model_comparison with a missing estimate ----------------------------------

fn write_results(dir: &std::path::Path, name: &str, body: &str) {
    let out = dir.join(name).join("Output");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join(format!("{name}_results.json")), body).unwrap();
}

/// model_comparison.m reads `oo_.MarginalDensity.<estimator>` and, when it is missing, prints
/// "I cant't find ..." and returns: no fallback to another density, no output.
#[test]
fn model_comparison_missing_estimate_returns_without_output() {
    let dir = common::temp_dir("pe_mc_missing");
    write_results(&dir, "m1", r#"{"log_data_density_laplace": -100.0}"#);
    // an SMC-style results file: only MarginalDensity.hssmc
    write_results(&dir, "m2", r#"{"MarginalDensity": {"hssmc": -99.0}}"#);
    let p = dir.join("cmp.mod");
    std::fs::write(&p, "var y; varexo e; model; y = 0.5*y(-1) + e; end;\nmodel_comparison m1 m2;\n").unwrap();
    run(&p, &RunOptions { quiet: true, ..Default::default() }).unwrap();
    assert!(!dir.join("cmp/Output/cmp_model_comparison.json").exists());
}

/// With both estimates present: posterior probabilities `exp(l_i) p_i / sum_j exp(l_j) p_j`.
#[test]
fn model_comparison_posterior_probabilities() {
    let dir = common::temp_dir("pe_mc_ok");
    write_results(&dir, "m1", r#"{"log_data_density_laplace": -100.0}"#);
    write_results(&dir, "m2", r#"{"log_data_density_laplace": -101.0}"#);
    let p = dir.join("cmp.mod");
    std::fs::write(&p, "var y; varexo e; model; y = 0.5*y(-1) + e; end;\nmodel_comparison m1(0.5) m2(0.5);\n").unwrap();
    run(&p, &RunOptions { quiet: true, ..Default::default() }).unwrap();
    let r: serde_json::Value = common::json_ref(dir.join("cmp/Output/cmp_model_comparison.json"));
    let p1 = r["models"]["m1"]["Posterior_Model_Probability"].as_f64().unwrap();
    let want = 1.0 / (1.0 + (-1.0f64).exp());
    assert!((p1 - want).abs() < 1e-12, "{p1} vs {want}");
}

// ---- estimation-3: TaRB's optim defaults to the estimation's ---------------------------------

#[test]
fn tarb_block_optimizer_uses_the_estimation_optim_unless_its_own() {
    use dynare::estimation::samplers::SamplerOptions;
    let est = optim(&[("MaxIter", OptValue::Num(7.0))]);
    let own = optim(&[("MaxIter", OptValue::Num(3.0))]);
    let a = SamplerOptions::default().with_estimation_optim(Some(&est));
    assert_eq!(format!("{:?}", a.tarb.optim), format!("{:?}", Some(&est)));
    let mut b = SamplerOptions::default();
    b.tarb.optim = Some(own.clone());
    assert_eq!(format!("{:?}", b.with_estimation_optim(Some(&est)).tarb.optim), format!("{:?}", Some(&own)));
    assert!(SamplerOptions::default().with_estimation_optim(None).tarb.optim.is_none());
}

// ---- estimation-4, 10: analytic gradient in csminwel -----------------------------------------

/// csminwel1.m with an analytic gradient takes `g` from `penalty_objective_function` (no
/// numerical gradient evaluations). At a point where the gradient is zero csminit1 returns
/// retcode 1 and the "stuck" branch recomputes the gradient once (`fcount + 1`): two
/// evaluations in total, whatever the number of parameters.
#[test]
fn csminwel_with_objective_gradient_makes_no_numerical_gradient_evaluations() {
    let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
    let g = |x: &[f64]| Some(vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)]);
    let obj = counting(f, g, true);
    let r = run_minimize(&obj, &[1.0, 2.0], None).unwrap();
    assert_eq!(obj.evals.load(Ordering::SeqCst), 2, "f(x0) and the gradient recomputed at xh");
    assert_eq!(r.func_count, 2);
    assert_eq!(r.x, vec![1.0, 2.0]);
    assert!(obj.grads.load(Ordering::SeqCst) >= 1);
}

/// Converges from a distant start with the analytic gradient.
#[test]
fn csminwel_with_objective_gradient_converges() {
    let f = |x: &[f64]| (x[0] - 1.0).powi(2) + 3.0 * (x[1] + 2.0).powi(2);
    let g = |x: &[f64]| Some(vec![2.0 * (x[0] - 1.0), 6.0 * (x[1] + 2.0)]);
    let obj = counting(f, g, true);
    let r = run_minimize(&obj, &[4.0, 3.0], None).unwrap();
    assert!((r.x[0] - 1.0).abs() < 1e-4 && (r.x[1] + 2.0).abs() < 1e-4, "{:?}", r.x);
}

// ---- estimation-5: numgrad3_ / numgrad5_ (NumgradAlgorithm 13 / 15) ---------------------------

/// numgrad3_.m sets `g0 = 0` where `f0 < f1 && f0 < f2` (a kink at the minimum), numgrad3.m
/// returns `(f1 - f2)/(2h)`. With `g = 0` csminit1 returns retcode 1 at once and csminwel1's
/// "stuck" branch recomputes the gradient: `f(x0)` plus two gradients of `2n` evaluations.
#[test]
fn numgrad3_zeroes_the_gradient_at_a_kink_minimum() {
    let kink = |x: &[f64]| x.iter().map(|&v| if v > 1.0 { 2.0 * (v - 1.0) } else { 1.0 - v }).sum::<f64>();
    let obj = counting(kink, |_| None, false);
    let o = optim(&[("NumgradAlgorithm", OptValue::Num(13.0))]);
    let r = run_minimize(&obj, &[1.0, 1.0], Some(&o)).unwrap();
    assert_eq!(obj.evals.load(Ordering::SeqCst), 1 + 2 * 2 + 2 * 2);
    assert_eq!(r.x, vec![1.0, 1.0]);
    // numgrad3.m: g = (2h - h)/(2h) = 0.5, a step is tried
    let obj = counting(kink, |_| None, false);
    let o = optim(&[("NumgradAlgorithm", OptValue::Num(3.0))]);
    run_minimize(&obj, &[1.0, 1.0], Some(&o)).unwrap();
    assert!(obj.evals.load(Ordering::SeqCst) > 9);
}

/// numgrad5_.m: zero when `f0 < f1 < f3` and `f0 < f2 < f4` (both sides increasing): `f(x0)`
/// plus two gradients of `4n` evaluations.
#[test]
fn numgrad5_zeroes_the_gradient_at_a_kink_minimum() {
    let kink = |x: &[f64]| x.iter().map(|&v| if v > 1.0 { 2.0 * (v - 1.0) } else { 1.0 - v }).sum::<f64>();
    let obj = counting(kink, |_| None, false);
    let o = optim(&[("NumgradAlgorithm", OptValue::Num(15.0))]);
    // n = 2: numgrad5_ never restores x(i), so with x(1) left at 1+2h... the second coordinate
    // is perturbed around (x1 - 2h, x2); the kink at the first coordinate still has f0 < f1 < f3
    // on both sides, and the second one sees the first shifted by -2h (a constant offset)
    let r = run_minimize(&obj, &[1.0, 1.0], Some(&o)).unwrap();
    assert_eq!(obj.evals.load(Ordering::SeqCst), 1 + 4 * 2 + 4 * 2);
    assert_eq!(r.x, vec![1.0, 1.0]);
}

// ---- estimation-12: InitialInverseHessian ----------------------------------------------------

/// The first line-search trial is `x0 - H0*g` (csminit1.m with lambda = 1).
#[test]
fn initial_inverse_hessian_is_the_evaluated_expression() {
    let f = |x: &[f64]| 0.5 * (x[0] * x[0] + x[1] * x[1]);
    let g = |x: &[f64]| Some(vec![x[0], x[1]]);
    for (expr, scale) in [("0.25*eye(2)", 0.25), ("eye(n_params)*0.5", 0.5), ("0.125", 0.125)] {
        let obj = counting(f, g, true);
        let v = if expr == "0.125" { OptValue::Num(0.125) } else { s(expr) };
        let o = optim(&[("InitialInverseHessian", v)]);
        run_minimize(&obj, &[4.0, 2.0], Some(&o)).unwrap();
        let pts = obj.points.lock().unwrap();
        let want = [4.0 * (1.0 - scale), 2.0 * (1.0 - scale)];
        assert!(
            (pts[1][0] - want[0]).abs() < 1e-12 && (pts[1][1] - want[1]).abs() < 1e-12,
            "{expr}: first trial {:?} vs {want:?}",
            pts[1]
        );
    }
    // wrong size and expressions that are not A*eye(n) are errors, not silent defaults
    for bad in ["eye(3)", "diag([1 2])"] {
        let obj = counting(f, g, true);
        let o = optim(&[("InitialInverseHessian", s(bad))]);
        assert!(run_minimize(&obj, &[4.0, 2.0], Some(&o)).is_err(), "{bad}");
    }
}

// ---- estimation-7: slice sampler convergence diagnostics -------------------------------------

fn chains(n: usize, ndraws: usize) -> Vec<dynare::estimation::mcmc::Chain> {
    // deterministic pseudo-random normals (Box-Muller on an LCG)
    let mut state = 12345u64;
    let mut u = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let mut c = dynare::estimation::mcmc::Chain::empty(2);
            for _ in 0..ndraws {
                let (a, b) = (u(), u());
                let r = (-2.0 * a.ln()).sqrt();
                let x = [r * (std::f64::consts::TAU * b).cos(), r * (std::f64::consts::TAU * b).sin()];
                c.push(&x, -0.5 * (x[0] * x[0] + x[1] * x[1]), false);
            }
            c.total_draws = ndraws;
            c
        })
        .collect()
}

/// mcmc_diagnostics.m:218-223: slice with fewer than 2000 draws uses `Origin = StepSize = 1`;
/// other samplers `Origin = 1000`, `StepSize = ceil((N-Origin)/100)`.
#[test]
fn brooks_gelman_origin_and_step_for_slice() {
    use dynare::estimation::diagnostics::brooks_gelman_multivariate;
    use dynare::estimation::posterior::brooks_gelman;
    let c = chains(2, 1500);
    let bg = brooks_gelman(&c, 0.5, true).unwrap();
    assert_eq!(bg.iterations, (1..=1500).collect::<Vec<_>>());
    let bg = brooks_gelman(&c, 0.5, false).unwrap();
    assert_eq!(bg.iterations, (1000..=1500).step_by(5).collect::<Vec<_>>());
    let (it, _) = brooks_gelman_multivariate(&c, 0.5, true).unwrap();
    assert_eq!(it.len(), 1500);
    let (it, _) = brooks_gelman_multivariate(&c, 0.5, false).unwrap();
    assert_eq!(it.len(), 101);
    // fewer than 1000 draws: slice still computes, the others report "too small"
    let c = chains(2, 600);
    assert_eq!(brooks_gelman(&c, 0.5, true).unwrap().iterations.len(), 600);
    assert!(brooks_gelman(&c, 0.5, false).is_none());
    assert!(brooks_gelman_multivariate(&c, 0.5, false).is_none());
}

#[test]
fn convergence_options_detect_the_slice_sampler() {
    use dynare::estimation::convergence::ConvergenceOptions;
    let o = vec![("posterior_sampling_method".to_string(), OptValue::Str("'slice'".into()))];
    assert!(ConvergenceOptions::from_options(&o).slice);
    let o = vec![("posterior_sampling_method".to_string(), OptValue::Str("'random_walk_metropolis_hastings'".into()))];
    assert!(!ConvergenceOptions::from_options(&o).slice);
    assert!(!ConvergenceOptions::from_options(&vec![]).slice);
}

// ---- estimation-9: dsge_var_likelihood passes qz_criterium to lyapunov_symm -----------------

/// `lyapunov_symm(T, R*Q*R', ..., options_.qz_criterium, options_.lyapunov_complex_threshold)`:
/// an eigenvalue above `2-qz_criterium` is a unit root for the Lyapunov solution, so a
/// `qz_criterium` of 1.9 (threshold 0.1) changes the likelihood of a stable model, while the
/// default (`1+1e-6`) is what the reference values of tests/estimation_q3.rs check.
#[test]
fn dsge_var_likelihood_honours_qz_criterium() {
    use dynare::dsge::{DsgeModel, LikelihoodData};
    use dynare::estimation::dsge_var::DsgeVarPosterior;
    use dynare::estimation::q3::Extras;
    use dynare::kalman::KalmanOptions;
    use dynare::modfile::{Command, parse_mod_file};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/q3");
    let mf = parse_mod_file(&root.join("q3_nk.mod"), &[("STAGE".to_string(), "1".to_string())]).unwrap();
    let est = mf
        .commands
        .iter()
        .find_map(|c| if let Command::Estimation(o, _) = c { Some(o.clone()) } else { None })
        .unwrap();
    let x0 = [0.8, 0.5, 1.0, 0.6, 0.15, 1.4, 0.5];
    let dsge = DsgeModel::new(&mf, Some(&x0)).unwrap();
    let table = dynare::data::Table::read(&root.join("nk_data.csv"), &Default::default()).unwrap();
    let init = dynare::estimation::q3::initial_conditions(&mf, &est).unwrap();
    let first_obs = match dynare::modfile::opt(&est, "first_obs") {
        Some(OptValue::Num(v)) => *v as usize,
        _ => 1,
    } - init;
    let sample = table.sample(&mf.varobs, first_obs, None).unwrap();
    let mut k = KalmanOptions::default();
    dynare::estimation::q3::kalman_options(&est, &mut k);
    k.qz_criterium = 1.9;
    k.lyapunov_complex_threshold = 1e-14;
    let data = LikelihoodData::new(&sample, k);
    let extras = Extras::new(&mf, &est, &dsge, &data).unwrap();
    assert_eq!(extras.lyap, (1.9, 1e-14), "the options reach the DSGE-VAR objective");
    let (m, w) = extras.dsge_var.as_ref().unwrap();
    let eval = |qz: f64| {
        DsgeVarPosterior {
            dsge: &dsge,
            moments: m,
            weight: *w,
            bounds: &dsge.bounds,
            qz_criterium: qz,
            lyapunov_complex_threshold: 1e-15,
        }
        .eval(&x0)
        .0
    };
    let (default, strict) = (eval(1.0 + 1e-6), eval(1.9));
    assert!(default.is_finite());
    assert!(strict.is_nan() || strict.is_infinite() || (strict - default).abs() > 1e-6 * default.abs());
}
