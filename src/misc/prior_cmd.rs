// SPDX-License-Identifier: GPL-3.0-or-later
//! [stream X] The `prior` command (`matlab/cli/prior.m`): `table` (`print_table_prior.m`),
//! `plot` (`plot_priors.m`), `simulate` (`prior_sampler.m`: effective prior mass and the
//! shares of draws without a determinate solution) and `moments` (theoretical moments at the
//! prior mode, `disp_th_moments.m`); `moments(distribution)` / `irfs(distribution)` and
//! `optimize` are (`prior_dist.rs`).

use anyhow::{Result, bail};
use rand::SeedableRng;

use super::Ctx;
use crate::dsge::DsgeModel;
use crate::estimation::prior::{PriorShape, prior_bounds, prior_draw};
use crate::modfile::ModFile;

const HELP: &str = "
Possible options are:
 + irfs(distribution)    Saves the implied prior for the impulse response function of all endogenous variables.
 + moments               Computes and displays moments of the endogenous variables at the prior mode.
 + moments(distribution) Print tables describing the implied prior for the first and second order unconditional
                         moments of all the endogenous variables.
 + optimize              Optimizes the prior density (starting from a random initial guess).
 + plot                  Plots the marginal prior densities.
 + simulate              Computes the effective prior mass (using a Monte-Carlo).
 + table                 Prints a table describing the priors.
";

/// `prior OPTIONS;` with the raw option tokens (`table`, `moments`, `(`, `distribution`, `)`...).
pub fn run(mf: &ModFile, raw: &[String], ctx: &Ctx, seed: u64) -> Result<()> {
    // re-join `moments ( distribution )` into one word
    let mut args: Vec<String> = Vec::new();
    for t in raw {
        match (t.as_str(), args.last_mut()) {
            ("(" | ")" | "distribution", Some(last)) => last.push_str(t),
            (",", _) => {}
            _ => args.push(t.clone()),
        }
    }
    if args.is_empty() || args == ["help"] {
        println!("{HELP}");
        return Ok(());
    }
    if mf.estimated_params.iter().all(|e| e.prior.is_none()) {
        println!("WARNING: No prior detected!");
        return Ok(());
    }
    let dsge = DsgeModel::new(mf, None)?;
    let names: Vec<String> =
        dsge.estimated_names.iter().map(|n| crate::estimation::plots::dynare_param_name(n)).collect();
    for a in &args {
        match a.as_str() {
            "table" => print!("{}", table_prior_text(&dsge, &names)),
            "plot" => {
                let gopts = crate::graphics::GraphOptions::from_options(&Vec::new(), ctx.dir, ctx.fname);
                for fig in crate::estimation::plots::priors_figures(&dsge.priors, &names) {
                    crate::graphics::save(&fig, ctx.fname, &gopts)?;
                }
                println!("prior plot: figures written to {}.", gopts.dir.display());
            }
            "simulate" => simulate(
                &dsge,
                prior_mc(mf),
                seed,
                crate::gsa::calibration::restrictions_checked(mf, ctx.ci)?.as_ref(),
            )?,
            "moments" => moments_at_prior_mode(mf, &dsge, ctx)?,
            "moments(distribution)" | "irfs(distribution)" => {
                let m = args.iter().any(|x| x == "moments(distribution)");
                let i = args.iter().any(|x| x == "irfs(distribution)");
                // prior.m handles both in one pass, after the other options
                if args.iter().rev().find(|x| x.ends_with("(distribution)")) == Some(a) {
                    super::prior_dist::distribution(mf, &dsge, ctx, seed, m, i)?;
                }
            }
            "optimize" => super::prior_dist::optimize(mf, &dsge, ctx, seed)?,
            other => bail!("prior: unknown option {other}"),
        }
    }
    Ok(())
}

/// `options_.prior_mc` (20000) as set by an earlier command.
fn prior_mc(mf: &ModFile) -> usize {
    if mf.options.active {
        // the estimations run so far (options_ is global, later commands have not run)
        return match crate::modfile::opt(&mf.options.estimation, "prior_mc") {
            Some(crate::modfile::OptValue::Num(v)) => *v as usize,
            _ => 20000,
        };
    }
    for c in mf.commands.iter().rev() {
        if let crate::modfile::Command::Other(_, o) | crate::modfile::Command::Estimation(o, _) = c
            && let Some(crate::modfile::OptValue::Num(v)) = crate::modfile::opt(o, "prior_mc")
        {
            return *v as usize;
        }
    }
    20000
}

/// MATLAB `%6.4f` (Inf/NaN right-aligned in 6 characters).
fn f64_6_4(v: f64) -> String {
    if v.is_nan() {
        "   NaN".into()
    } else if v.is_infinite() {
        if v > 0.0 { "   Inf".into() } else { "  -Inf".into() }
    } else {
        format!("{v:6.4}")
    }
}

/// `print_table_prior.m`.
pub fn table_prior_text(dsge: &DsgeModel, names: &[String]) -> String {
    let mut out = String::new();
    const SHAPES: [&str; 8] =
        ["Beta", "Gamma", "Gaussian", "Inverted Gamma", "Uniform", "Inverted Gamma -- 2", "Dirichlet", "Weibull"];
    let width_names = names.iter().map(String::len).max().unwrap_or(0);
    let shape_w = SHAPES.iter().map(|s| s.len()).max().unwrap();
    // options_.prior_interval = 0.9
    let intervals = prior_bounds(&dsge.priors, (1.0 - 0.9) / 2.0);
    let mut t2 = Vec::new();
    for (i, p) in dsge.priors.iter().enumerate() {
        let e = &dsge.estimated[i];
        let (lb, ub) = (e.lb.unwrap_or(f64::NEG_INFINITY), e.ub.unwrap_or(f64::INFINITY));
        let (lo, hi) = match p.shape {
            PriorShape::Beta | PriorShape::Uniform => {
                let mut lo = p.p3;
                let mut hi = p.p4;
                if lb.is_finite() {
                    lo = lo.max(lb);
                }
                if ub.is_finite() {
                    hi = hi.min(ub);
                }
                (lo, hi)
            }
            PriorShape::Normal => {
                let lo = if p.p3.is_infinite() && lb.is_infinite() {
                    f64::NEG_INFINITY
                } else if lb.is_finite() {
                    p.p3.max(lb)
                } else {
                    p.p3
                };
                let hi = if p.p4.is_infinite() && ub.is_infinite() {
                    f64::INFINITY
                } else if ub.is_finite() {
                    p.p4.min(ub)
                } else {
                    p.p4
                };
                (lo, hi)
            }
            _ => {
                let lo = if lb.is_finite() { p.p3.max(lb) } else { p.p3 };
                (lo, if ub.is_finite() { ub } else { f64::INFINITY })
            }
        };
        let shape = SHAPES[(p.shape.code() - 1) as usize];
        t2.push(format!(
            "{shape:<shape_w$} \t {} \t {} \t {} \t {} \t {} \t {} \t {}",
            f64_6_4(p.mean),
            if p.p5.is_nan() { "NaN".to_string() } else { f64_6_4(p.p5) },
            f64_6_4(p.sd),
            f64_6_4(lo),
            f64_6_4(hi),
            f64_6_4(intervals.lb[i]),
            f64_6_4(intervals.ub[i])
        ));
    }
    let header2 = "Prior shape      \t Prior mean \t Prior mode \t Prior std. \t Prior lb \t Prior ub \t Prior HPD lb \t Prior HPH ub";
    let w2 = t2.iter().map(String::len).chain([133, header2.len()]).max().unwrap();
    let w1 = width_names.max(10);
    let resize = width_names > 10;
    let l1 = "-".repeat(w1);
    let l2 = "-".repeat(133);
    let mut col1 = vec![l1.clone(), "PARAMETER ".to_string(), l1.clone()];
    col1.extend(names.iter().cloned());
    col1.push(l1);
    let mut col2 = vec![l2.clone(), header2.to_string(), l2.clone()];
    col2.extend(t2);
    col2.push(l2);
    out += "\n\n";
    for (k, (a, b)) in col1.iter().zip(&col2).enumerate() {
        let sep = if resize { if k == 0 || k == 2 || k == col1.len() - 1 { "--" } else { "  " } } else { "" };
        out += &format!("{a:<w1$}{sep}{b:<w2$}\n");
    }
    out += "\n\n";
    out
}

/// `prior_sampler.m` without saving the draws: shares of `prior_mc` draws by `info` code.
fn simulate(dsge: &DsgeModel, n: usize, seed: u64, rs: Option<&crate::gsa::calibration::Restrictions>) -> Result<()> {
    let bounds = prior_bounds(&dsge.priors, 0.0);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed ^ 0x5052_494f_525f_4d43);
    let draws: Vec<Vec<f64>> = (0..n).map(|_| prior_draw(&dsge.priors, &bounds, &mut rng)).collect();
    use rayon::prelude::*;
    let infos: Vec<i32> = draws
        .par_iter()
        .map(|x| match dsge.resolve(x) {
            Err(e) => e.info,
            // ---- calibration restrictions: prior_sampler.m, info 49 ----
            Ok(r) => rs.map_or(0, |rs| crate::gsa::calibration::restrictions_info(dsge, &r.dr, &r.q, rs).0),
        })
        .collect();
    let share = |codes: &[i32]| infos.iter().filter(|i| codes.contains(i)).count() as f64 / n as f64;
    let ok = share(&[0]);
    println!("\n");
    println!("Prior mass = {}", num2str(ok));
    println!("BK indeterminacy share                = {}", num2str(share(&[4])));
    println!("BK unstability share                  = {}", num2str(share(&[3])));
    println!("BK singularity share                  = {}", num2str(share(&[5])));
    println!("Complex Jacobian share                = {}", num2str(share(&[6])));
    println!("mjdgges crash share                   = {}", num2str(share(&[2])));
    println!("Steady state problem share            = {}", num2str(share(&[20])));
    println!("Complex steady state share            = {}", num2str(share(&[21])));
    println!("Endogenous prior violation share      = {}", num2str(share(&[49])));
    if dsge.loglinear {
        println!("Nonpositive steady state share        = {}", num2str(share(&[26])));
    }
    println!("Analytical steady state problem share = {}", num2str(share(&[19])));
    println!("\n");
    println!("(dynare-rust: {n} draws from its own random stream; the shares agree with Dynare's statistically.)");
    Ok(())
}

/// MATLAB `num2str` of a scalar (`%.4g` for non-integers... `max(log10(|x|)+5, 5)` significant digits).
fn num2str(x: f64) -> String {
    if x == x.round() {
        return format!("{}", x as i64);
    }
    let digits = ((x.abs().log10().floor() as i32) + 5).max(5) as usize;
    let s = format!("{:.*e}", digits - 1, x);
    let v: f64 = s.parse().unwrap();
    let mut out = format!("{v}");
    if out.len() > 12 {
        out = format!("{v:.4}");
    }
    out
}

/// `prior moments`: theoretical moments of the endogenous variables at the prior mode (prior
/// mean where the mode is undefined), as `stoch_simul(nofunctions, irf=0, nomodelsummary)`.
fn moments_at_prior_mode(mf: &ModFile, dsge: &DsgeModel, ctx: &Ctx) -> Result<()> {
    let x: Vec<f64> = dsge.priors.iter().map(|p| if p.p5.is_nan() { p.mean } else { p.p5 }).collect();
    let info = match dsge.resolve(&x) {
        Err(e) => e.info,
        // ---- calibration restrictions: prior.m checks them at the prior mode ----
        Ok(r) => match crate::gsa::calibration::restrictions_checked(mf, ctx.ci)? {
            // `if info` on [49, penalty]: false when the penalty is exactly 0 (MATLAB's `if` on a vector)
            Some(rs) => match crate::gsa::calibration::restrictions_info(dsge, &r.dr, &r.q, &rs) {
                (49, p) if p != 0.0 => 49,
                _ => 0,
            },
            None => 0,
        },
    };
    if info == 49 {
        println!(
            "\nCannot solve the model on the prior mode (info = 49, The model violates one (many) endogenous prior restriction(s))\n"
        );
        return Ok(());
    } else if info != 0 {
        println!("\nCannot solve the model on the prior mode (info = {info})\n");
        return Ok(());
    }
    let o = crate::analysis::StochSimulOptions {
        order: 1,
        irf: 0,
        nofunctions: true,
        nomodelsummary: true,
        ..Default::default()
    };
    let octx = crate::analysis::OutputContext {
        dir: ctx.dir.to_path_buf(),
        fname: format!("{}_prior", ctx.fname),
        out_dir: ctx.out_dir.to_path_buf(),
        seed: 0,
    };
    crate::analysis::run_stoch_simul(mf, &o, &[], &octx, Some(&x))?;
    Ok(())
}
