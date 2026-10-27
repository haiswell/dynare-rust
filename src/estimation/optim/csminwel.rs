// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `optimization/csminwel1.m` (Sims' csminwel, `mode_compute = 4`).

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::{GradientMethod, OptimResult, bfgsi1, csminit1, numgrad};
use crate::estimation::{Objective, penalty_objective};
use crate::linalg::Mat;

/// Options of `mode_compute = 4`, defaults from `default_option_values.m`
/// and `dynare_minimize_objective.m`.
#[derive(Clone, Debug)]
pub struct CsminwelOptions {
    /// `options_.csminwel.tolerance.f` (1e-7).
    pub crit: f64,
    /// `options_.csminwel.maxiter` (1000).
    pub nit: usize,
    /// `options_.gradient_method` (2 = forward differences, `numgrad2.m`).
    pub gradient_method: GradientMethod,
    /// `options_.gradient_epsilon` (1e-6).
    pub gradient_epsilon: f64,
    /// Initial inverse Hessian; `None` = `1e-4*eye(n)`.
    pub h0: Option<Mat>,
    /// A scalar initial inverse Hessian (`InitialInverseHessian` given as a number), which
    /// overrides `h0`. MATLAB keeps `H` a scalar until the first BFGS update or the cliff
    /// perturbation `H+diag(diag(H).*rand(nx,1))`, which add it to every element.
    pub h0_scalar: Option<f64>,
    pub verbose: bool,
    /// Evaluate gradient coordinates in parallel.
    pub parallel: bool,
    /// Seed of the `rand` used to perturb the search direction on cliffs.
    pub seed: u64,
    /// ---- estimation completion ---- `analytic_derivation` or the method of moments'
    /// `analytic_jacobian` (`analytic_grad = 1`): the gradient is the objective's analytic one
    /// (`penalty_objective_function`'s third output).
    pub analytic: bool,
}

impl Default for CsminwelOptions {
    fn default() -> Self {
        CsminwelOptions {
            crit: 1e-7,
            nit: 1000,
            gradient_method: GradientMethod::Forward,
            gradient_epsilon: 1e-6,
            h0: None,
            h0_scalar: None,
            verbose: true,
            parallel: true,
            seed: 0,
            analytic: false,
        }
    }
}

fn message(rc: i32) -> &'static str {
    match rc {
        0 => "normal step",
        1 => "zero gradient",
        6 => "smallest step still improving too slow, reversed gradient",
        5 => "largest step still improving too fast",
        2 | 4 => "back and forth on step length never finished",
        3 => "smallest step still improving too slow",
        7 => "warning: possible inaccuracy in H matrix",
        8 => "warning: iteration count termination",
        _ => "Unaccounted Case, please contact the developers",
    }
}

/// Port of `csminwel1.m` with numerical gradients, as called by
/// `dynare_minimize_objective.m` for `mode_compute = 4`. The returned
/// `hessian` is `inv(H)` (as Dynare's `hessian_mat`), `inv_hessian` is `H`.
pub fn csminwel(obj: &dyn Objective, x0: &[f64], opts: &CsminwelOptions) -> OptimResult {
    let nx = x0.len();
    let verbose = opts.verbose;
    let mut rng = ChaCha8Rng::seed_from_u64(opts.seed);
    let crit = opts.crit;
    let grad = |f0: f64, x: &[f64], penalty: f64| {
        numgrad(obj, opts.gradient_method, f0, x, penalty, opts.gradient_epsilon, opts.parallel)
    };
    let mut penalty = 1e8;
    // ---- estimation completion ---- analytic gradient: `[~, cost_flag, g] =
    // penalty_objective_function(x, ...)` (csminwel1.m evaluates it at x1 for g2, g3 and gh).
    let agrad = |x: &[f64], penalty: f64| -> (Vec<f64>, bool, usize) {
        let (_, ok, g, _) = crate::estimation::analytic::penalty_objective_grad(
            obj,
            x,
            penalty,
            crate::estimation::analytic::AnalyticHessian::None,
        );
        (g, !ok, 1)
    };
    let (f0, cost_flag, g_an) = if opts.analytic {
        let (f, ok, g, _) = crate::estimation::analytic::penalty_objective_grad(
            obj,
            x0,
            penalty,
            crate::estimation::analytic::AnalyticHessian::None,
        );
        (f, ok, Some(g))
    } else {
        let (f, ok, _) = penalty_objective(obj, x0, penalty, false);
        (f, ok, None)
    };
    let mut fcount = 1usize;
    if !cost_flag {
        let msg = "Bad initial parameter.";
        if verbose {
            println!("{msg}");
        }
        return OptimResult {
            x: x0.to_vec(),
            fval: f0,
            hessian: None,
            inv_hessian: None,
            iterations: 0,
            func_count: fcount,
            exitflag: -1,
            message: msg.into(),
        };
    }
    let (mut g, mut badg, fcg) = match g_an {
        Some(g) => (g, false, 0),
        None => grad(f0, x0, penalty),
    };
    fcount += fcg;
    let mut x = x0.to_vec();
    let mut f = f0;
    // H; while it is still MATLAB's scalar H0, `h` is `h_scalar*eye(nx)` (same `-H*g`)
    let mut h_scalar = opts.h0_scalar;
    let mut h = match h_scalar {
        Some(s) => Mat::identity(nx, nx) * s,
        None => opts.h0.clone().unwrap_or_else(|| Mat::identity(nx, nx) * 1e-4),
    };
    let mut itct = 0usize;
    let mut gh: Option<Vec<f64>> = None;
    let mut retcodeh;
    let mut msg;
    loop {
        penalty = f;
        if verbose {
            println!("-----------------");
            println!("f at the beginning of new iteration, {f:20.10}");
        }
        itct += 1;
        crate::progress::detail(format!("csminwel iteration {itct}, f = {f:.6}"));
        let (f1, x1, fc, retcode1) = csminit1(obj, &x, penalty, f, &g, badg, &h, verbose);
        fcount += fc;
        let (mut g1, mut g2, mut g3) = (None, None, None);
        let (mut badg1, mut badg2, mut badg3) = (None::<bool>, None::<bool>, None::<bool>);
        let (f2, x2, retcode2, f3, x3, retcode3);
        let mut f1 = f1;
        if retcode1 != 1 {
            let wall1 = if retcode1 == 2 || retcode1 == 4 {
                badg1 = Some(true);
                true
            } else {
                let (gg, b, fcg) = if opts.analytic { agrad(&x1, penalty) } else { grad(f1, &x1, penalty) };
                fcount += fcg;
                g1 = Some(gg);
                badg1 = Some(b);
                b
            };
            if wall1 {
                // Hcliff = H + diag(diag(H).*rand(nx,1)); a scalar H is added to every element
                let mut hcliff = match h_scalar {
                    Some(s) => Mat::from_element(nx, nx, s),
                    None => h.clone(),
                };
                for i in 0..nx {
                    let r: f64 = rng.r#gen();
                    hcliff[(i, i)] += h[(i, i)] * r;
                }
                if verbose {
                    println!("Cliff.  Perturbing search direction.");
                }
                let (ff2, xx2, fc, rc2) = csminit1(obj, &x, penalty, f, &g, badg, &hcliff, verbose);
                fcount += fc;
                f2 = ff2;
                x2 = xx2;
                retcode2 = rc2;
                if f2 < f {
                    let wall2 = if retcode2 == 2 || retcode2 == 4 {
                        badg2 = Some(true);
                        true
                    } else {
                        let (gg, b, fcg) = if opts.analytic { agrad(&x1, penalty) } else { grad(f2, &x2, penalty) };
                        fcount += fcg;
                        g2 = Some(gg);
                        badg2 = Some(b);
                        if verbose {
                            println!("Value of bad gradient 2: {}", b as u8);
                        }
                        b
                    };
                    if wall2 {
                        if verbose {
                            println!("Cliff again.  Try traversing");
                        }
                        let d: Vec<f64> = x2.iter().zip(&x1).map(|(a, b)| a - b).collect();
                        let dn = super::norm(&d);
                        if dn < 1e-13 {
                            f3 = f;
                            x3 = x.clone();
                            badg3 = Some(true);
                            retcode3 = 101;
                        } else {
                            let c = (f2 - f1) / (dn * dn);
                            let gcliff: Vec<f64> = d.iter().map(|v| c * v).collect();
                            let (ff3, xx3, fc, rc3) =
                                csminit1(obj, &x, penalty, f, &gcliff, false, &Mat::identity(nx, nx), verbose);
                            fcount += fc;
                            f3 = ff3;
                            x3 = xx3;
                            retcode3 = rc3;
                            if retcode3 == 2 || retcode3 == 4 {
                                badg3 = Some(true);
                            } else {
                                let (gg, b, fcg) =
                                    if opts.analytic { agrad(&x1, penalty) } else { grad(f3, &x3, penalty) };
                                fcount += fcg;
                                g3 = Some(gg);
                                badg3 = Some(b);
                            }
                        }
                    } else {
                        f3 = f;
                        x3 = x.clone();
                        badg3 = Some(true);
                        retcode3 = 101;
                    }
                } else {
                    f3 = f;
                    x3 = x.clone();
                    badg3 = Some(true);
                    retcode3 = 101;
                }
            } else {
                f2 = f;
                f3 = f;
                x2 = x.clone();
                x3 = x.clone();
                badg2 = Some(true);
                badg3 = Some(true);
                retcode2 = 101;
                retcode3 = 101;
            }
        } else {
            f2 = f;
            f3 = f;
            f1 = f;
            x2 = x.clone();
            x3 = x.clone();
            retcode2 = retcode1;
            retcode3 = retcode1;
        }
        let x1c = if opts.analytic { x1.clone() } else { Vec::new() }; // ---- estimation completion ----
        let good = |b: Option<bool>| b == Some(false);
        let (fh, xh, badgh);
        if f3 < f - crit && good(badg3) && f3 < f2 && f3 < f1 {
            fh = f3;
            xh = x3;
            gh = g3;
            badgh = badg3.unwrap();
            retcodeh = retcode3;
        } else if f2 < f - crit && good(badg2) && f2 < f1 {
            fh = f2;
            xh = x2;
            gh = g2;
            badgh = badg2.unwrap();
            retcodeh = retcode2;
        } else if f1 < f - crit && good(badg1) {
            fh = f1;
            xh = x1;
            gh = g1;
            badgh = badg1.unwrap();
            retcodeh = retcode1;
        } else {
            // stuck or bad gradient: [fh, ih] = min([f1, f2, f3])
            let fs = [f1, f2, f3];
            let mut ih = 0;
            for k in 1..3 {
                if fs[k] < fs[ih] || (fs[ih].is_nan() && !fs[k].is_nan()) {
                    ih = k;
                }
            }
            fh = fs[ih];
            xh = match ih {
                0 => x1,
                1 => x2,
                _ => x3,
            };
            retcodeh = [retcode1, retcode2, retcode3][ih];
            let mut b = true;
            if gh.is_none() {
                let (gg, bb, fcg) = if opts.analytic { agrad(&x1c, penalty) } else { grad(fh, &xh, penalty) };
                fcount += fcg;
                gh = Some(gg);
                b = bb;
            }
            badgh = b;
        }
        let stuck = (fh - f).abs() < crit;
        let ghv = gh.clone().expect("gradient");
        if !badg && !badgh && !stuck {
            let dg: Vec<f64> = ghv.iter().zip(&g).map(|(a, b)| a - b).collect();
            let dx: Vec<f64> = xh.iter().zip(&x).map(|(a, b)| a - b).collect();
            h = bfgsi1(&h, &dg, &dx, verbose);
            // bfgsi1.m with a scalar H0: `H0 + matrix` adds it to every element (no update,
            // and H stays a scalar, when |dg'*dx| <= 1e-12)
            if let Some(s) = h_scalar
                && super::dot(&dg, &dx).abs() > 1e-12
            {
                for j in 0..nx {
                    for i in 0..nx {
                        if i != j {
                            h[(i, j)] += s;
                        }
                    }
                }
                h_scalar = None;
            }
        }
        if verbose {
            println!("----");
            println!("Improvement on iteration {itct} = {:18.9}", f - fh);
        }
        let mut done = false;
        if itct > opts.nit {
            if verbose {
                println!("iteration count termination");
            }
            done = true;
            retcodeh = 8;
        } else if stuck {
            if verbose {
                println!("improvement < crit termination");
            }
            done = true;
        }
        msg = message(retcodeh);
        if verbose {
            println!("{msg}");
        }
        f = fh;
        x = xh;
        g = ghv;
        badg = badgh;
        if done {
            break;
        }
    }
    let hessian = crate::linalg::solve(&h, &Mat::identity(nx, nx));
    OptimResult {
        x,
        fval: f,
        hessian,
        inv_hessian: Some(h),
        iterations: itct,
        func_count: fcount,
        exitflag: retcodeh,
        message: msg.into(),
    }
}
