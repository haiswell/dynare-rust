//! Optimizers and numerical Hessian against Dynare 7.2 (`reference/estimation/ml_ref.json`,
//! written by `reference/estimation/run_ref.m`), plus analytic checks.
mod common;

use dynare::estimation::{
    Bounds, CsminwelOptions, Evaluation, FminsearchOptions, NewratOptions, Objective, csminwel, fminsearch, hessian,
    newrat,
};
use dynare::linalg::Mat;
use serde_json::Value;

fn load() -> Option<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/estimation/ml_ref.json");
    Some(common::json_ref(path))
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![],
    }
}

/// Rosenbrock as a sum of squared residuals (`rosen5.m`), with contributions.
struct Rosen {
    n: usize,
    a: f64,
}

impl Rosen {
    fn contributions(&self, x: &[f64]) -> Vec<f64> {
        let mut r = Vec::with_capacity(2 * (self.n - 1));
        for i in 0..self.n - 1 {
            r.push((self.a * (x[i + 1] - x[i] * x[i])).powi(2));
            r.push((1.0 - x[i]).powi(2));
        }
        r
    }
}

impl Objective for Rosen {
    fn dim(&self) -> usize {
        self.n
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        (self.contributions(x).iter().sum(), 0)
    }
    fn eval_full(&self, x: &[f64], c: bool) -> Evaluation {
        let v = self.contributions(x);
        Evaluation { fval: v.iter().sum(), info: 0, info4: 0.0, contributions: c.then_some(v) }
    }
}

/// `quad5.m`: f = sum_k 0.5 (A x - b)_k^2.
struct Quad {
    a: Mat,
    b: Vec<f64>,
}

impl Quad {
    fn contributions(&self, x: &[f64]) -> Vec<f64> {
        (0..self.a.nrows())
            .map(|k| {
                let r: f64 = (0..x.len()).map(|j| self.a[(k, j)] * x[j]).sum::<f64>() - self.b[k];
                0.5 * r * r
            })
            .collect()
    }
}

impl Objective for Quad {
    fn dim(&self) -> usize {
        self.a.ncols()
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        (self.contributions(x).iter().sum(), 0)
    }
    fn eval_full(&self, x: &[f64], c: bool) -> Evaluation {
        let v = self.contributions(x);
        Evaluation { fval: v.iter().sum(), info: 0, info4: 0.0, contributions: c.then_some(v) }
    }
}

fn quad() -> Quad {
    Quad { a: Mat::from_row_slice(4, 3, &[2., 1., 0., 0., 1., 1., 1., 0., 3., 1., 1., 1.]), b: vec![1., 2., 3., 4.] }
}

fn quiet_cw() -> CsminwelOptions {
    CsminwelOptions { verbose: false, ..Default::default() }
}

fn quiet_nr() -> NewratOptions {
    NewratOptions { verbose: false, ..Default::default() }
}

#[test]
fn csminwel_matches_dynare_on_rosenbrock() {
    let r = load().expect("reference/setup unavailable");
    for (key, x0) in [("cw2", vec![-1.2, 1.0]), ("cw4", vec![-1.2, 1.0, -1.2, 1.0])] {
        let obj = Rosen { n: x0.len(), a: 10.0 };
        let res = csminwel(&obj, &x0, &quiet_cw());
        let d = &r[key];
        let xr = vecf(&d["x"]);
        eprintln!(
            "{key}: rust f={:e} it={} fc={} rc={} | dynare f={} it={} fc={} rc={}",
            res.fval, res.iterations, res.func_count, res.exitflag, d["f"], d["it"], d["fc"], d["rc"]
        );
        let dx = res.x.iter().zip(&xr).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        eprintln!(
            "{key}: max |x - x_dynare| = {dx:e}, |f - f_dynare| = {:e}",
            (res.fval - d["f"].as_f64().unwrap()).abs()
        );
        if key == "cw2" {
            // bit-for-bit the same path as csminwel1.m
            assert_eq!(res.iterations as u64, d["it"].as_u64().unwrap(), "{key} iterations");
            assert_eq!(res.func_count as u64, d["fc"].as_u64().unwrap(), "{key} function count");
            assert_eq!(res.exitflag as i64, d["rc"].as_i64().unwrap(), "{key} retcode");
            assert!(dx < 1e-12, "{key}: {dx}");
        } else {
            // the 4-d path drifts at rounding level from iteration 4 on (BLAS
            // summation order in H*g and the BFGS update), amplified by the
            // finite-difference gradients: same algorithm, nearby end point.
            assert!(res.fval < 1e-6 && dx < 1e-3, "{key}: {dx}");
        }
    }
}

#[test]
fn newrat_matches_dynare_on_quadratic() {
    let r = load().expect("reference/setup unavailable");
    let obj = quad();
    let bounds = Bounds { lb: vec![-10.0; 3], ub: vec![10.0; 3] };
    let res = newrat(&obj, &[0.0, 0.0, 0.0], &bounds, &[f64::NAN; 3], &quiet_nr());
    let d = &r["nrq"];
    let xr = vecf(&d["x"]);
    let dx = res.x.iter().zip(&xr).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!(
        "nrq: rust f={} it={} fc={} ef={} | dynare f={} it={} fc={} ef={}; max dx {dx:e}",
        res.fval, res.iterations, res.func_count, res.exitflag, d["f"], d["it"], d["fc"], d["ef"]
    );
    // MATLAB evaluates A*x with BLAS (FMA), so the objective differs in the last
    // bits; newrat's univariate steps use second differences with h = 1e-6,
    // which turns that into ~1e-4 relative differences in the step: compare
    // the end point at the precision allowed by ftol = 1e-5.
    assert_eq!(res.exitflag as i64, d["ef"].as_i64().unwrap());
    assert!(dx < 5e-3, "{dx}");
    assert!((res.fval - d["f"].as_f64().unwrap()).abs() < 1e-5);
    let hh = res.hessian.unwrap();
    // outer-product-gradient Hessian (row-major nested array in the JSON)
    let hr: Vec<Vec<f64>> = d["hh"].as_array().unwrap().iter().map(vecf).collect();
    for i in 0..3 {
        for j in 0..3 {
            let t = hr[i][j];
            assert!((hh[(i, j)] - t).abs() < 1e-2 * (1.0 + t.abs()), "hh {i},{j}: {} vs {t}", hh[(i, j)]);
        }
    }
    // the exact minimizer of the least-squares problem
    let a = &obj.a;
    let ata = a.transpose() * a;
    let atb = a.transpose() * Mat::from_column_slice(4, 1, &obj.b);
    let xs = ata.lu().solve(&atb).unwrap();
    for i in 0..3 {
        assert!((res.x[i] - xs[i]).abs() < 1e-2, "{:?} vs {}", res.x, xs);
    }
}

#[test]
fn newrat_on_rosenbrock_tracks_dynare() {
    // Ill-conditioned (two contributions for two parameters, singular outer
    // product at the optimum): compare the end point only loosely.
    let r = load().expect("reference/setup unavailable");
    let obj = Rosen { n: 2, a: 10.0 };
    let bounds = Bounds { lb: vec![-5.0; 2], ub: vec![5.0; 2] };
    let res = newrat(&obj, &[-1.2, 1.0], &bounds, &[f64::NAN; 2], &quiet_nr());
    let d = &r["nr2"];
    eprintln!(
        "nr2: rust f={} x={:?} it={} fc={} ef={} | dynare f={} x={} it={} fc={} ef={}",
        res.fval, res.x, res.iterations, res.func_count, res.exitflag, d["f"], d["x"], d["it"], d["fc"], d["ef"]
    );
    assert!(res.fval < 0.05);
}

#[test]
fn hessian_matches_dynare_and_analytic() {
    let obj = Rosen { n: 2, a: 10.0 };
    let x = [0.5, 0.7];
    let h = hessian(&obj, &x, [1e-2, 1.0]);
    // analytic: f = 100 (x2 - x1^2)^2 + (1 - x1)^2
    let truth = [1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0, -400.0 * x[0], -400.0 * x[0], 200.0];
    for k in 0..4 {
        assert!((h[(k % 2, k / 2)] - truth[k]).abs() < 1e-3, "{k}");
    }
    if let Some(r) = load() {
        let hr = vecf(&r["hess"]);
        for k in 0..4 {
            assert!(
                (h[(k % 2, k / 2)] - hr[k]).abs() < 1e-9 * (1.0 + hr[k].abs()),
                "{k}: {} vs {}",
                h[(k % 2, k / 2)],
                hr[k]
            );
        }
    }
}

#[test]
fn csminwel_quadratic_inverse_hessian() {
    // f = 0.5 x'Ax - b'x: BFGS H converges towards inv(A)
    let a = Mat::from_row_slice(3, 3, &[4., 1., 0.5, 1., 3., 0.2, 0.5, 0.2, 2.]);
    let obj = dynare::estimation::FnObjective {
        n: 3,
        f: move |x: &[f64]| {
            let mut s = 0.0;
            for i in 0..3 {
                for j in 0..3 {
                    s += 0.5 * x[i] * a[(i, j)] * x[j];
                }
                s -= x[i] * (i as f64 + 1.0);
            }
            s
        },
    };
    let res = csminwel(&obj, &[3.0, -2.0, 1.0], &quiet_cw());
    let a = Mat::from_row_slice(3, 3, &[4., 1., 0.5, 1., 3., 0.2, 0.5, 0.2, 2.]);
    let xs = a.clone().lu().solve(&Mat::from_column_slice(3, 1, &[1., 2., 3.])).unwrap();
    for i in 0..3 {
        assert!((res.x[i] - xs[i]).abs() < 1e-4, "{:?}", res.x);
    }
    let hh = hessian(&obj, &res.x, [1e-2, 1.0]);
    assert!((hh - &a).amax() < 1e-4);
}

#[test]
fn fminsearch_rosenbrock() {
    let obj = Rosen { n: 2, a: 10.0 };
    let res = fminsearch(&obj, &[-1.2, 1.0], &FminsearchOptions::default());
    assert_eq!(res.exitflag, 1);
    assert!((res.x[0] - 1.0).abs() < 1e-5 && (res.x[1] - 1.0).abs() < 1e-5, "{:?}", res.x);
}

#[test]
fn penalty_on_failure() {
    // failures inside the line search are replaced by base_penalty + info(4)
    struct Walled;
    impl Objective for Walled {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> (f64, i32) {
            if x[0] > 0.5 { (f64::INFINITY, 41) } else { ((x[0] - 1.0).powi(2) + x[1] * x[1], 0) }
        }
    }
    let res = csminwel(&Walled, &[-1.0, 1.0], &quiet_cw());
    assert!(res.x[0] <= 0.5 && res.fval.is_finite());
    assert!((res.x[0] - 0.5).abs() < 1e-2 && res.x[1].abs() < 0.1, "{:?}", res.x);
}

/// `optim=('Hessian',1,'robust',true)` (Dynare 7.2 tests/optimizers/fs2000_5.mod): MATLAB's
/// logicals true/false are numeric option values 1/0.
#[test]
fn optim_logical_values_are_numbers() {
    use dynare::estimation::optim::minimize::kv_num;
    use dynare::modfile::OptValue;
    assert_eq!(kv_num("robust", &OptValue::Str("true".into())).unwrap(), 1.0);
    assert_eq!(kv_num("robust", &OptValue::Str("false".into())).unwrap(), 0.0);
    assert_eq!(kv_num("TolFun", &OptValue::Str("1e-5".into())).unwrap(), 1e-5);
    assert!(kv_num("robust", &OptValue::Str("yes".into())).is_err());
}
