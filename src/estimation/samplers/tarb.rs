// SPDX-License-Identifier: GPL-3.0-or-later
//! `tailored_random_block_metropolis_hastings` branch of
//! `posterior_sampler_iteration.m` (Chib and Ramamurthy 2010) with
//! `TaRB_optimizer_wrapper.m`: each iteration splits the parameters into
//! random blocks (`randperm`, a new block with probability
//! `new_block_probability`), and for each block maximizes the conditional
//! posterior (`tarb.mode_compute`, csminwel by default), takes the inverse
//! `hessian.m` Hessian there as proposal covariance (`chol`, else `chol_SE`)
//! scaled by `jscale`, and accepts the draw with the Metropolis–Hastings ratio
//! corrected for the (state-dependent) proposal densities.

use anyhow::{Result, bail};
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use super::{ProposalDist, SamplerOptions, Stepper, chol_se, randperm, strictly_inside};
use crate::estimation::optim::minimize::{initial_inverse_hessian, key_values, kv_num, numgrad_algorithm};
use crate::estimation::{Bounds, CsminwelOptions, Evaluation, NewratOptions, Objective, csminwel, hessian, newrat};
use crate::linalg::{Mat, chol_upper, solve};

/// `TaRB_optimizer_wrapper.m`: the objective as a function of one block.
struct BlockObjective<'a> {
    obj: &'a dyn Objective,
    base: Vec<f64>,
    idx: Vec<usize>,
}

impl BlockObjective<'_> {
    fn full(&self, x: &[f64]) -> Vec<f64> {
        let mut p = self.base.clone();
        for (k, &i) in self.idx.iter().enumerate() {
            p[i] = x[k];
        }
        p
    }
}

impl Objective for BlockObjective<'_> {
    fn dim(&self) -> usize {
        self.idx.len()
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        self.obj.eval(&self.full(x))
    }
    fn eval_full(&self, x: &[f64], contributions: bool) -> Evaluation {
        self.obj.eval_full(&self.full(x), contributions)
    }
}

/// `dynare_minimize_objective(@TaRB_optimizer_wrapper, ..., tarb.mode_compute, ...)`
/// with `silent_optimizer = 1` and `optim_opt = tarb.optim` (the estimation's `optim` unless
/// the sampler has its own).
fn block_minimize(
    obj: &dyn Objective,
    x0: &[f64],
    bounds: &Bounds,
    prior_sd: &[f64],
    so: &SamplerOptions,
    seed: u64,
) -> Result<Vec<f64>> {
    let kv = key_values(so.tarb.optim.as_ref(), "tarb optim")?;
    match so.tarb.mode_compute {
        4 => {
            let mut o = CsminwelOptions { verbose: false, parallel: true, seed, ..Default::default() };
            for (k, v) in &kv {
                match k.as_str() {
                    "MaxIter" => o.nit = kv_num(k, v)? as usize,
                    "TolFun" => o.crit = kv_num(k, v)?,
                    "NumgradAlgorithm" => {
                        if let Some(m) = numgrad_algorithm(k, v, o.analytic)? {
                            o.gradient_method = m;
                        }
                    }
                    "NumgradEpsilon" => o.gradient_epsilon = kv_num(k, v)?,
                    "InitialInverseHessian" => initial_inverse_hessian(&mut o, v, x0.len())?,
                    "verbosity" | "SaveFiles" => {}
                    other => println!("WARNING: csminwel: Unknown option ({other})!"),
                }
            }
            Ok(csminwel(obj, x0, &o).x)
        }
        5 => {
            let mut o = NewratOptions { verbose: false, parallel: true, ..Default::default() };
            for (k, v) in &kv {
                match k.as_str() {
                    "MaxIter" => o.nit = kv_num(k, v)? as usize,
                    "TolFun" => o.ftol = kv_num(k, v)?,
                    "robust" => o.robust = kv_num(k, v)? != 0.0,
                    "TolGstep" => o.gstep_crit = kv_num(k, v)?,
                    "TolGstepRel" => o.gstep_crit_rel = kv_num(k, v)?,
                    "Hessian" | "verbosity" | "SaveFiles" => {}
                    other => println!("WARNING: newrat: Unknown option ({other})!"),
                }
            }
            Ok(newrat(obj, x0, bounds, prior_sd, &o).x)
        }
        6 => bail!("tailored_random_block_metropolis_hastings: tarb mode_compute = 6 is not supported"),
        m => {
            // Every other optimizer through dynare_minimize_objective (silent_optimizer = 1).
            let names: Vec<String> = (0..x0.len()).map(|i| format!("block_param_{}", i + 1)).collect();
            let ctx = crate::estimation::optim::MinimizeContext {
                bounds,
                names: &names,
                prior_sd,
                optim: so.tarb.optim.as_ref(),
                verbose: false,
                seed,
                stream: 0,
            };
            Ok(crate::estimation::optim::minimize(m, obj, x0, &ctx)?.x)
        }
    }
}

pub(crate) struct Tarb<'a> {
    obj: &'a dyn Objective,
    bounds: &'a Bounds,
    so: &'a SamplerOptions,
    jscale: &'a [f64],
    rng: ChaCha8Rng,
    last: Vec<f64>,
    last_lp: f64,
    chain: usize,
    iter: u64,
}

impl<'a> Tarb<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        obj: &'a dyn Objective,
        bounds: &'a Bounds,
        so: &'a SamplerOptions,
        jscale: &'a [f64],
        rng: ChaCha8Rng,
        x: &[f64],
        lp: f64,
        chain: usize,
    ) -> Self {
        Tarb { obj, bounds, so, jscale, rng, last: x.to_vec(), last_lp: lp, chain, iter: 0 }
    }

    fn iterate(&mut self) -> Result<f64> {
        let npar = self.last.len();
        let indices = randperm(&mut self.rng, npar);
        let p = self.so.tarb.new_block_probability;
        let mut blocks = vec![0usize; npar];
        for k in 1..npar {
            blocks[k] = blocks[k - 1] + (self.rng.r#gen::<f64>() > 1.0 - p) as usize;
        }
        let nblocks = blocks[npar - 1] + 1;
        let mut current = self.last.clone();
        let mut accepted = 0usize;
        for blk in 0..nblocks {
            let idx: Vec<usize> = (0..npar).filter(|&k| blocks[k] == blk).map(|k| indices[k]).collect();
            let nx = idx.len();
            let start: Vec<f64> = idx.iter().map(|&i| current[i]).collect();
            let bo = BlockObjective { obj: self.obj, base: current.clone(), idx: idx.clone() };
            let bb = Bounds {
                lb: idx.iter().map(|&i| self.bounds.lb[i]).collect(),
                ub: idx.iter().map(|&i| self.bounds.ub[i]).collect(),
            };
            let seed = (self.chain as u64) << 48 ^ self.iter << 16 ^ blk as u64;
            let sd: Vec<f64> = match &self.so.priors {
                Some(p) => idx.iter().map(|&i| p[i].sd).collect(),
                None => vec![f64::NAN; nx],
            };
            let xopt = block_minimize(&bo, &start, &bb, &sd, self.so, seed)?;
            let h = hessian(&bo, &xopt, [1e-2, 1.0]);
            let small = || Mat::identity(nx, nx) * 1e-4;
            let inv = if h.iter().any(|v| !v.is_finite()) {
                small()
            } else {
                match solve(&h, &Mat::identity(nx, nx)) {
                    Some(m) if m.iter().all(|v| v.is_finite()) => m,
                    _ => small(),
                }
            };
            let mut u = match chol_upper(&inv) {
                Some(u) => u,
                None => chol_se(&((&inv + inv.transpose()) * 0.5))?.0,
            };
            for (j, &i) in idx.iter().enumerate() {
                for r in 0..nx {
                    u[(r, j)] *= self.jscale[i];
                }
            }
            let prop = ProposalDist { kind: self.so.proposal, df: self.so.student_df, u };
            let mut proposed = vec![0.0; nx];
            prop.draw(&mut self.rng, &xopt, &mut proposed);
            let lp = if strictly_inside(&proposed, &bb) { -bo.eval(&proposed).0 } else { f64::NEG_INFINITY };
            if lp > f64::NEG_INFINITY {
                // Formula (6) of Chib and Ramamurthy
                let accprob = lp - self.last_lp + prop.logpdf(&start, &xopt) - prop.logpdf(&proposed, &xopt);
                if self.rng.r#gen::<f64>().ln() < accprob {
                    for (k, &i) in idx.iter().enumerate() {
                        current[i] = proposed[k];
                    }
                    self.last_lp = lp;
                    accepted += 1;
                }
            }
        }
        self.last = current;
        self.iter += 1;
        Ok(accepted as f64 / nblocks as f64)
    }
}

impl Stepper for Tarb<'_> {
    fn rng(&self) -> &ChaCha8Rng {
        &self.rng
    }

    fn advance(&mut self, _max_steps: usize, on_step: &mut dyn FnMut(&[f64], f64, f64, usize)) -> Result<usize> {
        let acc = self.iterate()?;
        on_step(&self.last, self.last_lp, acc, 1);
        Ok(1)
    }
}
