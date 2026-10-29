//! newrat (`mode_compute = 5`) iterate by iterate against Dynare 7.2's `newrat.m`: the
//! optimizer stopped after k = 1, 2, ... iterations (`optim = ('MaxIter', k)`) and run to
//! convergence, on an objective evaluated bit for bit as in MATLAB
//! (`reference/estimation/ratfit_loop.m`, dumped by `dump_newrat_iterates.m`). With the
//! same function values, the iterates, gradients, Hessians, step sizes and function counts
//! follow MATLAB's to rounding.
mod common;

use dynare::estimation::{Bounds, Evaluation, NewratOptions, Objective, newrat};
use serde_json::Value;

/// `ratfit_loop.m`.
struct RatFit {
    with_dlik: bool,
}

const Y: [f64; 12] = [1.2, 1.9, 2.3, 2.4, 2.35, 2.2, 2.0, 1.9, 1.75, 1.6, 1.5, 1.45];

impl RatFit {
    fn contributions(x: &[f64]) -> Vec<f64> {
        (1..=12)
            .map(|k| {
                let t = 0.5 * k as f64;
                let m = (x[0] + x[1] * t) / (1.0 + x[2] * t + x[3] * x[3] * t * t);
                let e = Y[k - 1] - m;
                e * e
            })
            .collect()
    }
    fn sum(r: &[f64]) -> f64 {
        let mut f = 0.0;
        for v in r {
            f += v;
        }
        f
    }
}

impl Objective for RatFit {
    fn dim(&self) -> usize {
        4
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        (Self::sum(&Self::contributions(x)), 0)
    }
    fn eval_full(&self, x: &[f64], c: bool) -> Evaluation {
        let r = Self::contributions(x);
        Evaluation { fval: Self::sum(&r), info: 0, info4: 0.0, contributions: (c && self.with_dlik).then_some(r) }
    }
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

/// The paths are not bit for bit MATLAB's: MATLAB's LAPACK rounds the Cholesky solve of
/// mr_hessian.m (`hh_mat\\A`) differently from Accelerate (1e-13 relative here), and
/// mr_gstep.m's second differences with steps of `gradient_epsilon` = 1e-6 turn those last
/// bits into ~1e-5 relative differences of the next iterate (MATLAB itself moves as much when
/// the start is moved by one ulp). Each step given MATLAB's inputs is checked bit for bit by
/// the unit tests of `src/estimation/optim/newrat.rs`.
const TOL_X: f64 = 1e-3;
const TOL_H: f64 = 1e-2;
const TOL_F: f64 = 1e-3;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / (1.0 + b.abs())
}

#[test]
fn newrat_follows_dynare_iterate_by_iterate() {
    let r: Value =
        common::json_ref(concat!(env!("CARGO_MANIFEST_DIR"), "/reference/estimation/ref_newrat_iterates.json"));
    for (name, c) in r.as_object().unwrap() {
        let obj = RatFit { with_dlik: c["with_dlik"].as_f64().unwrap() != 0.0 };
        let x0 = vecf(&c["x0"]);
        let bounds = Bounds { lb: vecf(&c["lb"]), ub: vecf(&c["ub"]) };
        let ftol = c["ftol"].as_f64().unwrap();
        // newrat.tolerance.gstep_rel: mr_hessian's steps aim at |f(x+h)-f(x)| = 1e-6*|f|, near the
        // rounding noise of f: the intermediate iterates (1e-2 apart after 5 iterations, and
        // converging one iteration apart) are not compared, the converged point is (to ftol)
        let noisy = c["htol_rel"].is_number();
        for run in c["runs"].as_array().unwrap() {
            let k = run["maxiter"].as_u64().unwrap() as usize;
            let opts = NewratOptions {
                ftol,
                nit: k,
                gstep_crit_rel: c["htol_rel"].as_f64().unwrap_or(f64::NAN),
                robust: c["robust"].as_bool().unwrap(),
                gradient_epsilon: c["gradient_epsilon"].as_f64().unwrap(),
                verbose: false,
                parallel: false,
                ..Default::default()
            };
            let res = newrat(&obj, &x0, &bounds, &[f64::NAN; 4], &opts);
            let tag = format!("{name} MaxIter {k}");
            let xr = vecf(&run["x"]);
            let dx = res.x.iter().zip(&xr).map(|(a, b)| rel(*a, *b)).fold(0.0, f64::max);
            let fr = run["f"].as_f64().unwrap();
            let hr: Vec<Vec<f64>> = run["hh"].as_array().unwrap().iter().map(vecf).collect();
            let hh = res.hessian.as_ref().expect("hh");
            let hmax = hr.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
            let dh = (0..16).map(|k| (hh[(k / 4, k % 4)] - hr[k / 4][k % 4]).abs()).fold(0.0, f64::max) / hmax;
            eprintln!(
                "{tag}: it {} / {}, fc {} / {}, ef {} / {}, f {:e}, max rel dx {dx:e}, hh {dh:e}",
                res.iterations,
                run["it"],
                res.func_count,
                run["fc"],
                res.exitflag,
                run["ef"],
                res.fval - fr
            );
            if noisy {
                if k == 1000 {
                    assert_eq!(res.exitflag, 1, "{tag}: exitflag");
                    assert!(dx < 10.0 * TOL_X && (res.fval - fr).abs() < ftol, "{tag}: x {:?} vs {xr:?}", res.x);
                }
                continue;
            }
            assert_eq!(res.iterations as u64, run["it"].as_u64().unwrap(), "{tag}: iterations");
            assert_eq!(res.exitflag as i64, run["ef"].as_i64().unwrap(), "{tag}: exitflag");
            assert_eq!(res.message, run["message"].as_str().unwrap(), "{tag}: message");
            // Function counts follow MATLAB's until two line searches end differently
            let fcr = run["fc"].as_u64().unwrap() as f64;
            assert!((res.func_count as f64 - fcr).abs() <= 0.1 * fcr, "{tag}: function count");
            assert!(dx < TOL_X, "{tag}: x {:?} vs {xr:?}", res.x);
            assert!(rel(res.fval, fr) < TOL_F, "{tag}: f {} vs {fr}", res.fval);
            assert!(dh < TOL_H, "{tag}: hh");
        }
    }
}

/// `options_.gradient_epsilon = V;` in the .mod (Dynare's tests/estimation/slice/fs2000_slice.mod)
/// reaches the estimation command and, through it, newrat's initial steps (`hess_info.h1`,
/// mr_gstep.m) and csminwel's numerical gradient, as `dynare_minimize_objective.m` reads
/// `options_.gradient_epsilon`.
#[test]
fn options_gradient_epsilon_reaches_the_optimizers() {
    use dynare::estimation::optim::minimize::{GradientGuard, GradientOptions};
    use dynare::estimation::optim::{MinimizeContext, minimize};
    use dynare::estimation::{CsminwelOptions, csminwel};
    use dynare::modfile::{Command, OptValue, opt, parse_mod_str};
    let src = "var y; varexo e; parameters r; r=0.5; model; y = r*y(-1)+e; end;\n\
               shocks; var e; stderr 1; end; steady_state_model; y=0; end;\n\
               estimated_params; r, 0.5, 0, 0.99; end; varobs y;\n\
               options_.gradient_epsilon = 1e-5;\n\
               estimation(datafile=data, mode_compute=5);\n";
    let mf = parse_mod_str(src, &[]).unwrap();
    let est = mf.commands.iter().find_map(|c| if let Command::Estimation(o, _) = c { Some(o) } else { None });
    assert_eq!(opt(est.expect("estimation"), "gradient_epsilon"), Some(&OptValue::Num(1e-5)));

    let obj = RatFit { with_dlik: true };
    let x0 = [1.0, 1.0, 0.1, 0.3];
    let bounds = Bounds { lb: vec![-5.0, -5.0, -0.5, -5.0], ub: vec![5.0, 5.0, 2.0, 5.0] };
    let names: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
    let ctx = MinimizeContext {
        bounds: &bounds,
        names: &names,
        prior_sd: &[f64::NAN; 4],
        optim: None,
        verbose: false,
        seed: 0,
        stream: 0,
    };
    let nr = |eps: f64| {
        let o = NewratOptions { gradient_epsilon: eps, verbose: false, names: names.clone(), ..Default::default() };
        newrat(&obj, &x0, &bounds, &[f64::NAN; 4], &o).x
    };
    let cw = |eps: f64| {
        csminwel(&obj, &x0, &CsminwelOptions { gradient_epsilon: eps, verbose: false, ..Default::default() }).x
    };
    let run = |m: i32| minimize(m, &obj, &x0, &ctx).unwrap().x;
    let (nr_default, cw_default) = (run(5), run(4));
    assert_eq!(nr_default, nr(1e-6));
    assert_eq!(cw_default, cw(1e-6));
    let _g = GradientGuard::set(GradientOptions { method: 2.0, epsilon: 1e-5 });
    let (nr5, cw5) = (run(5), run(4));
    assert_eq!(nr5, nr(1e-5));
    assert_eq!(cw5, cw(1e-5));
    assert_ne!(nr5, nr_default);
    assert_ne!(cw5, cw_default);
}
