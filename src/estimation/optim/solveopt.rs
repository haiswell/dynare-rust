// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `optimization/solvopt.m` (Kuntsevich & Kappel's SolvOpt, Shor's
//! r-algorithm) and `optimization/apprgrdn.m` (its finite-difference gradient),
//! as Dynare calls it for `mode_compute = 101`:
//! `solvopt(x0, objective_function, [], [], [], options_.solveopt)` — no
//! analytic gradient, no constraints, on the raw objective (`+inf` on failure,
//! which the 1-D search treats as "step too long").
//!
//! The coordinates of `apprgrdn` are independent of each other, so with
//! `parallel` they are evaluated on the rayon pool; the result is bit for bit
//! the sequential one.
//!
//! To follow MATLAB's iterates bit for bit, vector norms go through Accelerate's
//! `dnrm2` (what MATLAB's `norm` calls on Apple silicon) and the small
//! vector/matrix products use the summation orders of MATLAB's kernels
//! ([`mdot`] for `row*column` and `row*matrix`, sequential for `matrix*column`
//! and `(g1/d1)*B'`). With these, `tests/estimation_solveopt.rs` reproduces
//! MATLAB's whole sequence of objective evaluations on Rosenbrock (n = 2, 4) and a
//! quadratic. Residual sources of last-bit differences: MATLAB's scalar `^` and
//! `log10` (its own libm, not correctly rounded; they only enter the step-size
//! control `gamma`), and MATLAB's different kernels for products longer than 8.

use anyhow::Result;
use rayon::prelude::*;

use super::OptimResult;
use super::minimize::kv_num;
use crate::estimation::Objective;
use crate::modfile::OptValue;

/// `options_.solveopt` (`default_option_values.m`).
#[derive(Clone, Debug)]
pub struct SolveoptOptions {
    /// `minimizer_indicator` (-1: minimize; |H| != 1 is the initial step size).
    pub minimizer_indicator: f64,
    pub tol_x: f64,
    pub tol_fun: f64,
    pub max_iter: usize,
    /// `verbosity`: > 0 prints every `verbosity`-th iteration and warnings,
    /// 0 only warnings, -1 nothing (`silent_optimizer`).
    pub verbosity: f64,
    pub tol_x_constraint: f64,
    pub space_dilation: f64,
    pub lb_gradient_step: f64,
    /// Evaluate the gradient's coordinates in parallel (same result).
    pub parallel: bool,
    /// ---- estimation completion ---- `analytic_derivation` or the method of moments'
    /// `analytic_jacobian`: solvopt's `grad = 1` (exact gradients from
    /// [`crate::estimation::analytic::eval_with_gradient`], `app = 0`).
    pub analytic: bool,
}

impl Default for SolveoptOptions {
    fn default() -> Self {
        SolveoptOptions {
            minimizer_indicator: -1.0,
            tol_x: 1e-6,
            tol_fun: 1e-6,
            max_iter: 15000,
            verbosity: 1.0,
            tol_x_constraint: 1e-8,
            space_dilation: 2.5,
            lb_gradient_step: 1e-11,
            parallel: true,
            analytic: false,
        }
    }
}

impl SolveoptOptions {
    /// The `optim` keys of `dynare_minimize_objective.m`, case 101.
    pub fn apply_optim(&mut self, kv: &[(String, OptValue)]) -> Result<()> {
        for (k, v) in kv {
            match k.as_str() {
                "TolX" => self.tol_x = kv_num(k, v)?,
                "TolFun" => self.tol_fun = kv_num(k, v)?,
                "MaxIter" => self.max_iter = kv_num(k, v)? as usize,
                "verbosity" => self.verbosity = kv_num(k, v)?,
                "SpaceDilation" => self.space_dilation = kv_num(k, v)?,
                "LBGradientStep" => self.lb_gradient_step = kv_num(k, v)?,
                other => println!("WARNING: solveopt: Unknown option ({other})!"),
            }
        }
        Ok(())
    }
}

/// MATLAB `sign` (NaN stays NaN).
fn msign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else if v == 0.0 {
        0.0
    } else {
        f64::NAN
    }
}

#[cfg_attr(target_os = "macos", link(name = "Accelerate", kind = "framework"))]
unsafe extern "C" {
    fn cblas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64;
}

/// MATLAB's `norm` of a vector: Accelerate's `dnrm2` (MATLAB R2025b on Apple
/// silicon links Accelerate; bit-identical on 2100 random vectors, n = 2..73).
fn norm(v: &[f64]) -> f64 {
    // SAFETY: v is a valid slice of v.len() doubles.
    unsafe { cblas_dnrm2(v.len() as i32, v.as_ptr(), 1) }
}

/// MATLAB's row-times-column product `a*b'` (and each entry of row-times-matrix):
/// the leading element alone when the length is odd, then the sums of adjacent
/// pairs, accumulated in order: `((p0 + p1) + (p2 + p3)) + ...`. Bit-identical to
/// MATLAB R2025b (Apple silicon) for n <= 8 on random data; for longer vectors
/// MATLAB's kernel is different (last-bit differences only).
fn mdot(a: &[f64], b: &[f64]) -> f64 {
    mdot_by(a.len(), |i| a[i] * b[i])
}

fn mdot_by(n: usize, p: impl Fn(usize) -> f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let (mut s, mut i) = if n % 2 == 1 { (p(0), 1) } else { (p(0) + p(1), 2) };
    while i < n {
        s += p(i) + p(i + 1);
        i += 2;
    }
    s
}

fn eval(obj: &dyn Objective, x: &[f64]) -> f64 {
    obj.eval(x).0
}

/// Port of `apprgrdn.m` with `obj = 1` (objective, not constraint).
/// Returns `(g, fcount)`.
pub(crate) fn apprgrdn(obj: &dyn Objective, x: &[f64], f: f64, deltax: &[f64], parallel: bool) -> (Vec<f64>, usize) {
    let n = x.len();
    let mut di: Vec<f64> = x.iter().map(|v| if v.abs() < 5e-15 { 5e-15 } else { v.abs() }).collect();
    for (d, dx) in di.iter_mut().zip(deltax) {
        *d *= dx;
    }
    let central: Vec<bool> = di.iter().map(|d| d.abs() < 2e-10).collect();
    for (d, &c) in di.iter_mut().zip(&central) {
        if c {
            *d = 2e-10 * msign(*d);
        }
    }
    let coord = |i: usize| -> (f64, usize) {
        let mut y = x.to_vec();
        let mut d = di[i];
        y[i] = x[i] + d;
        let mut fi = eval(obj, &y);
        let mut cnt = 1;
        if fi == f {
            for _ in 0..3 {
                d *= 10.0;
                y[i] = x[i] + d;
                fi = eval(obj, &y);
                cnt += 1;
                if fi != f {
                    break;
                }
            }
        }
        let mut g = (fi - f) / d;
        if central[i] {
            y[i] = x[i] - d;
            let fi = eval(obj, &y);
            cnt += 1;
            g = 0.5 * (g + (f - fi) / d);
        }
        (g, cnt)
    };
    let r: Vec<(f64, usize)> =
        if parallel { (0..n).into_par_iter().map(coord).collect() } else { (0..n).map(coord).collect() };
    let fcount = r.iter().map(|p| p.1).sum();
    (r.into_iter().map(|p| p.0).collect(), fcount)
}

/// MATLAB `%13.5e`.
fn fmt_e(v: f64) -> String {
    let s = format!("{v:.5e}");
    let s = match s.split_once('e') {
        Some((m, e)) => {
            let (sg, d) = if let Some(d) = e.strip_prefix('-') { ('-', d) } else { ('+', e) };
            format!("{m}e{sg}{d:0>2}")
        }
        None => s,
    };
    format!("{s:>13}")
}

const ERRMES: &str = "SolvOpt error:";
const WRNMES: &str = "SolvOpt warning:";
const ERROR2: &str = "Argument X has to be a row or column vector of dimension > 1.";
const ERROR31: &str = "Function value does not exist (NaN is returned).";
const ERROR32: &str = "Function equals infinity at the point.";
const ERROR41: &str = "Gradient does not exist (NaN is returned by <grad>).";
const ERROR42: &str = "Gradient equals infinity at the starting point.";
const ERROR43: &str = "Gradient equals zero at the starting point.";
const ERROR5: &str = "Function is unbounded.";
const ERROR6: &str = "Choose another starting point.";
const WARN1: &str = "Gradient is zero at the point, but stopping criteria are not fulfilled.";
const WARN20: &str = "Normal re-setting of a transformation matrix.";
const WARN4: &str = "Iterations limit exceeded.";
const WARN31: &str = "The function is flat in certain directions.";
const WARN32: &str = "Trying to recover by shifting insensitive variables.";
const WARN09: &str = "Re-run from recorded point.";
const WARN08: &str = "Ravine with a flat bottom is detected.";
const TERMWARN0: &str = "SolvOpt: Normal termination.";
const TERMWARN1: &str = "SolvOpt: Termination warning:";
const APPWARN: &str = "The above warning may be reasoned by inaccurate gradient approximation.";
const ENDWARN: [&str; 4] = [
    "Premature stop is possible. Try to re-run the routine from the obtained point.",
    "Result may not provide the optimum. The function apparently has many extremum points.",
    "Result may be inaccurate in the coordinates. The function is flat at the optimum.",
    "Result may be inaccurate in a function value. The function is extremely steep at the optimum.",
];

/// Port of `solvopt.m` (unconstrained, numerical gradient), as called by
/// `dynare_minimize_objective.m` for `mode_compute = 101`.
///
/// `iterations` is solvopt's `k`, `func_count` is `n_f_evals + n_grad_evals`
/// (all evaluations, the gradient's included), `exitflag` is 1 for a normal
/// termination (solvopt returns the iteration count) and solvopt's negative
/// code otherwise (-3 function NaN/inf, -4 gradient, -7 unbounded, -8 zero
/// gradient, -9 iteration limit, -11..-14 termination warnings).
pub fn solveopt(obj: &dyn Objective, x0: &[f64], opts: &SolveoptOptions) -> Result<OptimResult> {
    let n = x0.len();
    let par = opts.parallel;
    let mut n_f_evals: usize = 0;
    let mut n_grad_evals: usize = 0; // ---- estimation completion ---- (analytic gradients)
    let mut k: i64 = 0;
    macro_rules! ret {
        ($x:expr, $f:expr, $flag:expr, $msg:expr) => {{
            let flag: i64 = $flag;
            return Ok(OptimResult {
                x: $x.to_vec(),
                fval: $f,
                hessian: None,
                inv_hessian: None,
                iterations: k.max(0) as usize,
                func_count: (n_f_evals + n_grad_evals).saturating_sub(opts.analytic as usize),
                exitflag: if flag > 0 { 1 } else { flag as i32 },
                message: $msg,
            });
        }};
    }
    // Display control
    let (dispdata, dispwarn) =
        if opts.verbosity <= 0.0 { (0i64, opts.verbosity != -1.0) } else { (opts.verbosity.round() as i64, true) };
    let warn = |lines: &[&str]| {
        if dispwarn {
            for l in lines {
                println!("{l}");
            }
        }
    };
    if n <= 1 {
        warn(&[ERRMES, ERROR2]);
        ret!(x0, f64::NAN, -2, format!("{ERRMES} {ERROR2}"));
    }
    // Check the values
    let mut tol_x = opts.tol_x.max(1e-12);
    let mut tol_fun = opts.tol_fun.max(1e-12);
    tol_x = (opts.lb_gradient_step * 1e2).max(tol_x);
    tol_x = tol_x.min(1.0);
    tol_fun = tol_fun.min(1.0);
    let space_dilation = opts.space_dilation.max(1.5);
    let lb_gradient_step = opts.lb_gradient_step.max(1e-11);

    let nf = n as f64;
    let epsnorm = 1e-15;
    let epsnorm2 = 1e-30;
    let h1 = msign(opts.minimizer_indicator);
    let wdef = 1.0 / space_dilation - 1.0;
    let ajb = 1.0 + 0.1 / (nf * nf);
    let mut ajp: f64 = 20.0;
    let ajpp = ajp;
    let ajs: f64 = 1.15;
    let mut knorms = 0usize;
    let mut gnorms = [0.0f64; 10];
    let mut ld = dispdata;
    let dq = 5.1;
    let (du20, du10, du03) = (2.0, 1.5, 1.05);
    let kstore = 3usize;
    let mut nsteps = [0.0f64; 3];
    let app = !opts.analytic; // ---- estimation completion ----
    // exact gradient (`[~, g] = feval(fun, x)`) or apprgrdn
    let grad = |x: &[f64], f: f64, deltax: &[f64]| -> (Vec<f64>, usize, usize) {
        if app {
            let (g, c) = apprgrdn(obj, x, f, deltax, par);
            (g, c, 0)
        } else {
            let g = crate::estimation::analytic::eval_with_gradient(
                obj,
                x,
                crate::estimation::analytic::AnalyticHessian::None,
            )
            .map(|e| e.grad)
            .unwrap_or_else(|| vec![f64::NAN; x.len()]);
            (g, 0, 1)
        }
    };
    let appwarn = if app { format!(" {APPWARN}") } else { String::new() };
    let des = if app { 6.3 } else { 3.3 };
    let mxtc = 3;
    let mut termx = 0;
    let limxterm = 50;
    let ddx = 1e-11f64.max(lb_gradient_step);
    let low_bound = -1.0 + 1e-4;
    let zero_grad = nf * 1e-16;
    let mut nzero = 0;
    let lowxbound = tol_x.max(1e-3);
    let lowfbound = tol_fun * tol_fun;
    let mut krerun = 0;
    let detfr = tol_fun * 100.0;
    let detxr = tol_x * 10.0;
    let mut warnno: i64 = 0;
    let mut kflat = 0;
    let mut stepvanish = 0;
    let mut stopf = false;

    // First function value
    let mut x = x0.to_vec();
    let mut f = eval(obj, &x);
    n_f_evals += 1;
    if f.is_nan() {
        warn(&[ERRMES, ERROR31, ERROR6]);
        ret!(x, f, -3, format!("{ERRMES} {ERROR31} {ERROR6}"));
    } else if f.is_infinite() {
        warn(&[ERRMES, ERROR32, ERROR6]);
        ret!(x, f, -3, format!("{ERRMES} {ERROR32} {ERROR6}"));
    }
    let mut xrec = x.clone();
    let mut frec = f;
    // First gradient
    let deltax = vec![h1 * ddx; n];
    let (mut g, c, cg) = grad(&x, f, &deltax);
    n_f_evals += c;
    n_grad_evals += cg;
    let mut ng = norm(&g);
    if ng.is_nan() {
        warn(&[ERRMES, ERROR41, ERROR6]);
        ret!(x, f, -4, format!("{ERRMES} {ERROR41} {ERROR6}"));
    } else if ng == f64::INFINITY {
        warn(&[ERRMES, ERROR42, ERROR6]);
        ret!(x, f, -4, format!("{ERRMES} {ERROR42} {ERROR6}"));
    } else if ng < zero_grad {
        warn(&[ERRMES, ERROR43, ERROR6]);
        ret!(x, f, -4, format!("{ERRMES} {ERROR43} {ERROR6}"));
    }
    let mut grec = g.clone();
    let mut nng = ng;
    // Initial step size
    let xmax = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut h = h1 * tol_x.sqrt() * xmax;
    if opts.minimizer_indicator.abs() != 1.0 {
        h = h1 * opts.minimizer_indicator.abs().max(h.abs());
    } else {
        h = h1 * (1.0 / (ng + 1.1).ln()).max(h.abs());
    }

    let mut kd: i64 = 4;
    let mut xx = x.clone();
    let mut b = vec![0.0f64; n * n]; // column-major
    let mut gt = vec![0.0f64; n];
    let mut g0 = vec![0.0f64; n];
    let mut z = vec![0.0f64; n];
    let mut bz = vec![0.0f64; n];
    // Resetting loop
    loop {
        let mut kcheck: i64 = 0;
        let mut kg = 0usize;
        let mut kj: i64 = 0;
        b.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..n {
            b[i + i * n] = 1.0;
        }
        let fst = f;
        let mut g1 = g.clone();
        let mut dx = 0.0f64;
        // Main iterations
        'main: loop {
            k += 1;
            kcheck += 1;
            let laststep = dx;
            // Adjust gamma
            let mut gamma = 1.0 + ajb.powf((ajp - kcheck as f64) * nf).max(2.0 * tol_fun);
            gamma = gamma.min(ajs.powf(1.0f64.max((nng + 1.0).log10())));
            // gt = g*B
            for j in 0..n {
                let col = &b[j * n..(j + 1) * n];
                gt[j] = mdot(&g, col);
            }
            let mut w = wdef;
            // Jumping over a ravine
            let ngt = norm(&gt);
            let ng1 = norm(&g1);
            let cosv = mdot_by(n, |i| (gt[i] / ngt) * (g1[i] / ng1));
            if cosv < low_bound {
                if kj == 2 {
                    xx = x.clone();
                }
                if kj == 0 {
                    kd = 4;
                }
                kj += 1;
                w = -0.9;
                h *= 2.0;
                if kj > 2 * kd {
                    kd += 1;
                    warnno = 1;
                    if x.iter().zip(&xx).any(|(a, c)| (a - c).abs() < epsnorm * a.abs()) {
                        warn(&[WRNMES, WARN08]);
                    }
                }
            } else {
                kj = 0;
            }
            // Dilation
            for i in 0..n {
                z[i] = gt[i] - g1[i];
            }
            let mut nrmz = norm(&z);
            if nrmz > epsnorm * ngt {
                for v in z.iter_mut() {
                    *v /= nrmz;
                }
                let zg = mdot(&z, &gt);
                let wz = w * zg;
                for i in 0..n {
                    g1[i] = gt[i] + wz * z[i];
                }
                for i in 0..n {
                    let mut s = 0.0;
                    for j in 0..n {
                        s += b[i + j * n] * z[j];
                    }
                    bz[i] = w * s;
                }
                for j in 0..n {
                    for i in 0..n {
                        b[i + j * n] += bz[i] * z[j];
                    }
                }
            } else {
                z.iter_mut().for_each(|v| *v = 0.0);
                nrmz = 0.0;
                g1.copy_from_slice(&gt);
            }
            let d1 = norm(&g1);
            // g0 = (g1/d1)*B'
            for j in 0..n {
                // B*u' order (sequential axpy over the columns of B)
                let mut s = 0.0;
                for i in 0..n {
                    s += (g1[i] / d1) * b[j + i * n];
                }
                g0[j] = s;
            }
            // Resetting
            if kcheck > 1 {
                let idx: Vec<usize> = (0..n).filter(|&i| g[i].abs() > zero_grad).collect();
                let numelem = idx.len();
                if numelem > 0 {
                    let grbnd = epsnorm * (numelem * numelem) as f64;
                    if idx.iter().all(|&i| g1[i].abs() <= g[i].abs() * grbnd) || nrmz == 0.0 {
                        warn(&[WRNMES, WARN20]);
                        if (fst - f).abs() < f.abs() * 0.01 {
                            ajp -= 10.0 * nf;
                        } else {
                            ajp = ajpp;
                        }
                        h = h1 * dx / 3.0;
                        k -= 1;
                        break 'main;
                    }
                }
            }
            // Store the current values, 1-D search
            let xopt = x.clone();
            let fopt = f;
            let (mut k1, mut k2, mut ksm, mut kc, mut knan) = (0, 0, false, 0, false);
            let mut hp = h;
            let mut x1 = x.clone();
            loop {
                x1.copy_from_slice(&x);
                let f1 = f;
                for i in 0..n {
                    x[i] += hp * g0[i];
                }
                f = eval(obj, &x);
                n_f_evals += 1;
                if h1 * f == f64::INFINITY {
                    warn(&[ERRMES, ERROR5]);
                    ret!(x, f, -7, format!("{ERRMES} {ERROR5}"));
                }
                if f.is_infinite() || f.is_nan() {
                    let e = if f.is_nan() { ERROR31 } else { ERROR32 };
                    warn(&[WRNMES, e]);
                    if ksm || kc >= mxtc {
                        // don't return with NaN or Inf despite the error code
                        ret!(x1, f1, -3, format!("{WRNMES} {e}"));
                    } else {
                        k2 += 1;
                        k1 = 0;
                        hp /= dq;
                        x.copy_from_slice(&x1);
                        f = f1;
                        knan = true;
                    }
                } else if x.iter().zip(&x1).all(|(a, c)| (a - c).abs() < a.abs() * epsnorm) {
                    stepvanish += 1;
                    if stepvanish >= 5 {
                        warn(&[TERMWARN1, ENDWARN[3]]);
                        ret!(x, f, -14, format!("{TERMWARN1} {}", ENDWARN[3]));
                    } else {
                        x.copy_from_slice(&x1);
                        f = f1;
                        hp *= 10.0;
                        ksm = true;
                    }
                } else if h1 * f < h1 * gamma.powf(msign(f1)) * f1 {
                    if ksm {
                        break;
                    }
                    k2 += 1;
                    k1 = 0;
                    hp /= dq;
                    x.copy_from_slice(&x1);
                    f = f1;
                    if kc >= mxtc {
                        break;
                    }
                } else {
                    if h1 * f <= h1 * f1 {
                        break;
                    }
                    k1 += 1;
                    if k2 > 0 {
                        kc += 1;
                    }
                    k2 = 0;
                    if k1 >= 20 {
                        hp *= du20;
                    } else if k1 >= 10 {
                        hp *= du10;
                    } else if k1 >= 3 {
                        hp *= du03;
                    }
                }
            }
            // Adjust the trial step size
            dx = norm(&xopt.iter().zip(&x).map(|(a, c)| a - c).collect::<Vec<_>>());
            if kg < kstore {
                kg += 1;
            }
            if kg >= 2 {
                for i in (1..kg).rev() {
                    nsteps[i] = nsteps[i - 1];
                }
            }
            nsteps[0] = dx / (h.abs() * norm(&g0));
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..kg {
                num += nsteps[i] * (kg - i) as f64;
                den += (kg - i) as f64;
            }
            let kk = num / den;
            if kk > des {
                if kg == 1 {
                    h *= kk - des + 1.0;
                } else {
                    h *= (kk - des + 1.0).sqrt();
                }
            } else if kk < des {
                h *= (kk / des).sqrt();
            }
            stepvanish += ksm as i32;
            // Gradient
            let deltax: Vec<f64> =
                g0.iter().map(|&v| if msign(v) == 0.0 { 1.0 } else { msign(v) }).map(|s| h1 * ddx * s).collect();
            let (gn, c, cg) = grad(&x, f, &deltax);
            g = gn;
            n_f_evals += c;
            n_grad_evals += cg;
            ng = norm(&g);
            if ng.is_nan() {
                warn(&[ERRMES, ERROR41]);
                ret!(x, f, -4, format!("{ERRMES} {ERROR41}"));
            } else if ng == f64::INFINITY {
                warn(&[ERRMES, ERROR42]);
                ret!(x, f, -4, format!("{ERRMES} {ERROR42}"));
            } else if ng < zero_grad {
                warn(&[WRNMES, WARN1]);
                ng = zero_grad;
            }
            if h1 * f > h1 * frec {
                frec = f;
                xrec.copy_from_slice(&x);
                grec.copy_from_slice(&g);
            }
            if ng > zero_grad {
                if knorms < 10 {
                    knorms += 1;
                }
                if knorms >= 2 {
                    for i in (1..knorms).rev() {
                        gnorms[i] = gnorms[i - 1];
                    }
                }
                gnorms[0] = ng;
                nng = gnorms[..knorms].iter().product::<f64>().powf(1.0 / knorms as f64);
            }
            // Display
            if k == ld {
                println!("Iter.# ..... Function ... Step Value ... Gradient Norm ");
                println!("{k:5}   {}   {}     {}", fmt_e(f), fmt_e(dx), fmt_e(ng));
                ld = k + dispdata;
            }
            // Stopping criteria
            let termflag = kcheck > 5 && !knan && kc < mxtc;
            if termflag {
                let idx: Vec<usize> = (0..n).filter(|&i| x[i].abs() >= lowxbound).collect();
                if idx.is_empty() || idx.iter().all(|&i| (xopt[i] - x[i]).abs() <= tol_x * x[i].abs()) {
                    termx += 1;
                    if (f - frec).abs() > detfr * f.abs() && (f - fopt).abs() <= tol_fun * f.abs() && krerun <= 3 {
                        if idx.iter().any(|&i| (xrec[i] - x[i]).abs() > detxr * x[i].abs()) {
                            warn(&[WRNMES, WARN09]);
                            x.copy_from_slice(&xrec);
                            f = frec;
                            g.copy_from_slice(&grec);
                            ng = norm(&g);
                            let _ = ng;
                            krerun += 1;
                            h = h1 * dx.max(detxr * norm(&x)) / krerun as f64;
                            warnno = 2;
                            break 'main;
                        } else {
                            h *= 10.0;
                        }
                    } else if (f - fopt).abs() <= tol_fun * f.abs()
                        || f.abs() <= lowfbound
                        || ((f - fopt).abs() <= tol_fun && termx >= limxterm)
                    {
                        if stopf {
                            if dx <= laststep {
                                if warnno == 1 && ng < tol_fun.sqrt() {
                                    warnno = 0;
                                }
                                if !app && g.iter().any(|v| v.abs() <= epsnorm2) {
                                    warnno = 3;
                                }
                                if warnno != 0 {
                                    let w = ENDWARN[(warnno - 1) as usize];
                                    if app {
                                        warn(&[TERMWARN1, w, APPWARN]);
                                    } else {
                                        warn(&[TERMWARN1, w]);
                                    }
                                    ret!(x, f, -warnno - 10, format!("{TERMWARN1} {w}{appwarn}"));
                                } else {
                                    warn(&[TERMWARN0]);
                                    ret!(x, f, k, TERMWARN0.to_string());
                                }
                            }
                        } else {
                            stopf = true;
                        }
                    } else if dx < 1e-12 * norm(&x).max(1.0) && termx >= limxterm {
                        if app {
                            warn(&[TERMWARN1, ENDWARN[3], APPWARN]);
                        } else {
                            warn(&[TERMWARN1, ENDWARN[3]]);
                        }
                        ret!(xrec, frec, -14, format!("{TERMWARN1} {}{appwarn}", ENDWARN[3]));
                    } else {
                        stopf = false;
                    }
                }
            }
            // Iterations limit
            if k == opts.max_iter as i64 {
                warn(&[WRNMES, WARN4]);
                ret!(x, f, -9, format!("{WRNMES} {WARN4}"));
            }
            // Zero gradient
            if ng <= zero_grad {
                nzero += 1;
                warn(&[WRNMES, WARN1]);
                if nzero >= 3 {
                    ret!(x, f, -8, format!("{WRNMES} {WARN1}"));
                }
                for v in g0.iter_mut() {
                    *v = -h * *v / 2.0;
                }
                for _ in 0..10 {
                    for i in 0..n {
                        x[i] += g0[i];
                    }
                    f = eval(obj, &x);
                    n_f_evals += 1;
                    if f.is_infinite() {
                        warn(&[ERRMES, ERROR32]);
                        ret!(x, f, -3, format!("{ERRMES} {ERROR32}"));
                    } else if f.is_nan() {
                        warn(&[ERRMES, ERROR31]);
                        ret!(x, f, -3, format!("{ERRMES} {ERROR31}"));
                    }
                    let deltax: Vec<f64> = g0
                        .iter()
                        .map(|&v| if msign(v) == 0.0 { 1.0 } else { msign(v) })
                        .map(|s| h1 * ddx * s)
                        .collect();
                    let (gn, c, cg) = grad(&x, f, &deltax);
                    g = gn;
                    n_f_evals += c;
                    n_grad_evals += cg;
                    ng = norm(&g);
                    if ng == f64::INFINITY {
                        warn(&[ERRMES, ERROR42]);
                        ret!(x, f, -4, format!("{ERRMES} {ERROR42}"));
                    } else if ng.is_nan() {
                        warn(&[ERRMES, ERROR41]);
                        ret!(x, f, -4, format!("{ERRMES} {ERROR41}"));
                    }
                    if ng > zero_grad {
                        break;
                    }
                }
                if ng <= zero_grad {
                    warn(&[TERMWARN1, WARN1]);
                    ret!(x, f, -8, format!("{TERMWARN1} {WARN1}"));
                }
                h = h1 * dx;
                break 'main;
            }
            // Function is flat at the point
            if (f - fopt).abs() < fopt.abs() * tol_fun && kcheck > 5 && ng < 1.0 {
                let idx: Vec<usize> = (0..n).filter(|&i| g[i].abs() <= epsnorm2).collect();
                let ni = idx.len();
                if ni >= 1 && (ni as f64) <= nf / 2.0 && kflat <= 3 {
                    kflat += 1;
                    warn(&[WRNMES, WARN31]);
                    warnno = 1;
                    let mut x1 = x.clone();
                    let mut fm = f;
                    for &j in &idx {
                        let mut y = x[j];
                        let mut f2 = fm;
                        x1[j] = if y == 0.0 {
                            1.0
                        } else if y.abs() < 1.0 {
                            msign(y)
                        } else {
                            y
                        };
                        for _ in 0..20 {
                            x1[j] /= 1.15;
                            let f1 = eval(obj, &x1);
                            n_f_evals += 1;
                            if f1.is_finite() {
                                if h1 * f1 > h1 * fm {
                                    y = x1[j];
                                    fm = f1;
                                } else if h1 * f2 > h1 * f1 {
                                    break;
                                } else if f2 == f1 {
                                    x1[j] /= 1.5;
                                }
                                f2 = f1;
                            }
                        }
                        x1[j] = y;
                    }
                    if h1 * fm > h1 * f {
                        let deltax = vec![h1 * ddx; n];
                        let (gt2, c, cg) = grad(&x1, fm, &deltax);
                        n_f_evals += c;
                        n_grad_evals += cg;
                        let ngt2 = norm(&gt2);
                        if !ngt2.is_nan() && ngt2 > epsnorm2 {
                            warn(&[WARN32]);
                            tol_fun /= 5.0;
                            x = x1;
                            g = gt2;
                            ng = ngt2;
                            let _ = ng;
                            f = fm;
                            h = h1 * dx / 3.0;
                            break 'main;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::FnObjective;

    #[test]
    fn quadratic_minimum() {
        let obj = FnObjective {
            n: 3,
            f: |x: &[f64]| (x[0] - 1.0).powi(2) + 2.0 * (x[1] + 0.5).powi(2) + 0.5 * x[2].powi(2) + 3.0,
        };
        let r = solveopt(&obj, &[0.3, 0.2, 1.0], &SolveoptOptions { verbosity: -1.0, ..Default::default() }).unwrap();
        assert!((r.fval - 3.0).abs() < 1e-8, "{r:?}");
        assert!((r.x[0] - 1.0).abs() < 1e-3 && (r.x[1] + 0.5).abs() < 1e-3);
    }

    #[test]
    fn parallel_gradient_is_bitwise_sequential() {
        let obj = FnObjective {
            n: 5,
            f: |x: &[f64]| x.iter().enumerate().map(|(i, v)| (i as f64 + 1.0) * v.powi(4) + v.sin()).sum::<f64>(),
        };
        let x = [0.3, -1e-16, 2.0, -0.7, 0.0];
        let f = obj.eval(&x).0;
        let d = [-1e-11; 5];
        assert_eq!(apprgrdn(&obj, &x, f, &d, true), apprgrdn(&obj, &x, f, &d, false));
    }
}
