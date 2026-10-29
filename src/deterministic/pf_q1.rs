// SPDX-License-Identifier: GPL-3.0-or-later
//! the parts of `perfect_foresight_solver_core.m` beyond the direct stacked Newton
//! (`sim1.m` with `stack_solve_algo = 0`, stream D):
//!
//! * `stack_solve_algo` 2 / 3 (`lin_solve.m`: GMRES / BiCGStab preconditioned by the LU of the
//!   first iteration's Jacobian, a block-diagonal LU or an incomplete LU), 8 / 9 (ParU, Pardiso:
//!   direct sparse solves, here the sparse LU) and 10 (Pardiso then CGS preconditioned by the
//!   first LU), with `iter_tol`, `iter_maxit`, `gmres_restart`, `block_diagonal_lu_*`;
//! * `robust_lin_solve` (`lin_solve_robust`), `allow_nonfinite_values`,
//!   `check_jacobian_singularity` (`check_Jacobian_for_singularity`);
//! * `stack_solve_algo` 1 / 6 (`sim1_lbj.m`, Laffargue–Boucekkine–Juillard);
//! * `stack_solve_algo` 7 (`solve_stacked_problem.m`: `dynare_solve` on the stacked system with
//!   `options_.solve_algo`, or `lmmcp` on the mixed complementarity problem for the `lmmcp`
//!   option / `solve_algo = 10`, `perfect_foresight_mcp_problem.m`);
//! * purely backward / forward / static models (`sim1_purely_{backward,forward,static}.m`:
//!   `dynare_solve` period by period).
//!
//! `incomplete_lu` (MATLAB's `ilu`, `ilutp` with `droptol` 1e-12) is replaced by the complete
//! sparse LU (what an ILUTP with that drop tolerance computes up to the dropped entries).

use super::iterative::{IterResult, bicgstab, cgs, gmres};
use super::lmmcp::{LmmcpOptions, lmmcp};
use super::mcp::Complementarity;
use super::sparse_lu::{Csc, LuWork, SparseLu};
use super::spmat::{Csr, norm_inf};
use super::{SimulOptions, compute_maxerror, fmt_g};
use crate::linalg::Mat;
use crate::model::Model;
use crate::modfile::{OptValue, Options, opt};
use crate::solver::{SteadyOptions, dynare_solve};
use std::time::Instant;

/// `options_.simul` fields of the iterative / robust solvers, `options_.lmmcp.status`.
#[derive(Clone, Debug)]
pub struct PfQ1Options {
    /// `first_iter_lu` (default), `block_diagonal_lu`, `incomplete_lu`.
    pub preconditioner: String,
    pub iter_tol: Option<f64>,
    pub iter_maxit: usize,
    pub gmres_restart: usize,
    pub block_diagonal_lu_maxlu: usize,
    pub block_diagonal_lu_nperiods: usize,
    pub block_diagonal_lu_nlu: i64,
    pub block_diagonal_lu_relu: f64,
    pub robust_lin_solve: bool,
    pub allow_nonfinite_values: bool,
    pub check_jacobian_singularity: bool,
    /// `homotopy_marginal_linearization_fallback` (0: off; the flag alone is 0.01, as the preprocessor writes).
    pub homotopy_marginal_linearization_fallback: f64,
    /// `minimal_solving_periods` (stored, unused by Dynare 7.2).
    pub minimal_solving_periods: usize,
    /// `lmmcp` (`options_.lmmcp.status`): stack_solve_algo 7 with solve_algo 10.
    pub lmmcp: bool,
    /// `options_.jacobian_tolerance` (rank tolerance of `check_jacobian_singularity`).
    pub jacobian_tolerance: Option<f64>,
}

impl Default for PfQ1Options {
    fn default() -> Self {
        PfQ1Options {
            preconditioner: "first_iter_lu".into(),
            iter_tol: None,
            iter_maxit: 500,
            gmres_restart: 100,
            block_diagonal_lu_maxlu: 20000,
            block_diagonal_lu_nperiods: 0,
            block_diagonal_lu_nlu: 0,
            block_diagonal_lu_relu: 0.5,
            robust_lin_solve: false,
            allow_nonfinite_values: false,
            check_jacobian_singularity: false,
            homotopy_marginal_linearization_fallback: 0.0,
            minimal_solving_periods: 1,
            lmmcp: false,
            jacobian_tolerance: None,
        }
    }
}

/// Options this module reads (the perfect-foresight commands do not warn about them).
pub const PF_Q1_OPTIONS: &[&str] = &[
    "preconditioner",
    "iter_tol",
    "iter_maxit",
    "gmres_restart",
    "block_diagonal_lu_maxlu",
    "block_diagonal_lu_nperiods",
    "block_diagonal_lu_nlu",
    "block_diagonal_lu_relu",
    "robust_lin_solve",
    "allow_nonfinite_values",
    "check_jacobian_singularity",
    "homotopy_marginal_linearization_fallback",
    "minimal_solving_periods",
    "lmmcp",
    "jacobian_tolerance",
];

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

fn flag(o: &Options, k: &str) -> Option<bool> {
    match opt(o, k) {
        Some(OptValue::Flag) => Some(true),
        Some(OptValue::Num(v)) => Some(*v != 0.0),
        Some(OptValue::Str(s)) => Some(s != "false" && s != "0"),
        _ => None,
    }
}

impl PfQ1Options {
    pub fn update(&mut self, o: &Options) {
        if let Some(OptValue::Str(s)) = opt(o, "preconditioner") {
            self.preconditioner = s.clone();
        }
        if let Some(v) = num(o, "iter_tol") {
            self.iter_tol = Some(v);
        }
        if let Some(v) = num(o, "iter_maxit") {
            self.iter_maxit = v as usize;
        }
        if let Some(v) = num(o, "gmres_restart") {
            self.gmres_restart = v as usize;
        }
        if let Some(v) = num(o, "block_diagonal_lu_maxlu") {
            self.block_diagonal_lu_maxlu = v as usize;
        }
        if let Some(v) = num(o, "block_diagonal_lu_nperiods") {
            self.block_diagonal_lu_nperiods = v as usize;
        }
        if let Some(v) = num(o, "block_diagonal_lu_nlu") {
            self.block_diagonal_lu_nlu = v as i64;
        }
        if let Some(v) = num(o, "block_diagonal_lu_relu") {
            self.block_diagonal_lu_relu = v;
        }
        if let Some(b) = flag(o, "robust_lin_solve") {
            self.robust_lin_solve = b;
        }
        if let Some(b) = flag(o, "allow_nonfinite_values") {
            self.allow_nonfinite_values = b;
        }
        if let Some(b) = flag(o, "check_jacobian_singularity") {
            self.check_jacobian_singularity = b;
        }
        match opt(o, "homotopy_marginal_linearization_fallback") {
            Some(OptValue::Flag) => self.homotopy_marginal_linearization_fallback = 0.01,
            Some(OptValue::Num(v)) => self.homotopy_marginal_linearization_fallback = *v,
            _ => {}
        }
        if let Some(v) = num(o, "minimal_solving_periods") {
            self.minimal_solving_periods = v as usize;
        }
        if let Some(b) = flag(o, "lmmcp") {
            self.lmmcp = b;
        }
        if let Some(v) = num(o, "jacobian_tolerance") {
            self.jacobian_tolerance = Some(v);
        }
    }
}

/// MATLAB `num2str` of a scalar (`%.<k>g`, k = max(digits before the point, 1) + 4).
pub fn num2str(v: f64) -> String {
    if !v.is_finite() {
        return fmt_g(v);
    }
    if v == v.trunc() && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    let d = if v == 0.0 { 1 } else { (v.abs().log10().floor() as i32 + 1).max(1) } + 4;
    let s = format!("{:.*e}", (d - 1) as usize, v);
    let (mant, exp) = s.split_once('e').unwrap();
    let e: i32 = exp.parse().unwrap();
    let trim = |m: &str| -> String {
        if m.contains('.') { m.trim_end_matches('0').trim_end_matches('.').to_string() } else { m.to_string() }
    };
    if e < -5 || e >= d {
        format!("{}e{}{:02}", trim(mant), if e < 0 { '-' } else { '+' }, e.abs())
    } else {
        let dec = (d - 1 - e).max(0) as usize;
        trim(&format!("{:.*}", dec, v))
    }
}

// ---- the stacked problem (`perfect_foresight_problem`) with a compressed-row Jacobian ----------

/// Residuals and sparse Jacobian of the stacked system over `T` periods (`y` period-major).
pub struct Stacked<'a> {
    pub model: &'a Model,
    pub n: usize,
    ws: crate::model::Workspace,
    vals: Vec<f64>,
}

impl<'a> Stacked<'a> {
    pub fn new(model: &'a Model) -> Self {
        Stacked {
            model,
            n: model.n_endo(),
            ws: model.workspace(),
            vals: vec![0.0; model.dynamic_jacobian_pattern().len()],
        }
    }

    /// `exo(t)`: exogenous row of simulation period t (0-based).
    pub fn eval(
        &mut self,
        y: &[f64],
        y0: &[f64],
        yt: &[f64],
        exo: &dyn Fn(usize) -> Vec<f64>,
        params: &[f64],
        jac: bool,
    ) -> (Vec<f64>, Option<Csr>) {
        let n = self.n;
        let tt = y.len() / n;
        let mut res = vec![0.0; n * tt];
        let mut trip = Vec::new();
        let mut y3 = vec![0.0; 3 * n];
        let pat: Vec<(usize, usize)> = self.model.dynamic_jacobian_pattern().to_vec();
        for t in 0..tt {
            let prev = if t == 0 { y0 } else { &y[(t - 1) * n..t * n] };
            let next = if t + 1 == tt { yt } else { &y[(t + 1) * n..(t + 2) * n] };
            y3[..n].copy_from_slice(prev);
            y3[n..2 * n].copy_from_slice(&y[t * n..(t + 1) * n]);
            y3[2 * n..].copy_from_slice(next);
            let x = exo(t);
            self.model.dynamic_residuals(&y3, &x, params, &mut res[t * n..(t + 1) * n]);
            if jac {
                self.model.dynamic_jacobian_values_with(&y3, &x, params, &mut self.ws, &mut self.vals);
                for (k, &(i, c)) in pat.iter().enumerate() {
                    if c >= 3 * n {
                        continue;
                    }
                    let (blk, j) = (c / n, c % n);
                    // blk 0: lag (period t-1), 1: current, 2: lead (t+1)
                    let q = t as i64 + blk as i64 - 1;
                    if q < 0 || q >= tt as i64 {
                        continue;
                    }
                    trip.push((t * n + i, q as usize * n + j, self.vals[k]));
                }
            }
        }
        (res, jac.then(|| Csr::from_triplets(n * tt, n * tt, trip)))
    }
}

fn exo_row(exo: &Mat, r: usize) -> Vec<f64> {
    exo.row(r).iter().copied().collect()
}

/// `y0`, `yT` and the stacked unknowns of `endo` (`n x (ml + T + mlead)`).
fn conditions(endo: &Mat, ml: usize, periods: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = endo.nrows();
    let y0: Vec<f64> = if ml > 0 { endo.column(ml - 1).iter().copied().collect() } else { vec![f64::NAN; n] };
    let yt: Vec<f64> = if endo.ncols() > ml + periods {
        endo.column(ml + periods).iter().copied().collect()
    } else {
        vec![f64::NAN; n]
    };
    let mut y = Vec::with_capacity(n * periods);
    for t in 0..periods {
        y.extend(endo.column(ml + t).iter());
    }
    (y0, yt, y)
}

// ---- lin_solve.m ----------------------------------------------------------------------------

/// Factorization kept across Newton iterations (`first_iter_lu`).
pub struct FirstLu(SparseLu);

fn factor(a: &Csr) -> Option<SparseLu> {
    SparseLu::factor(&a.to_csc(), None, 1.0, &mut LuWork::default())
}

fn sub_csr(a: &Csr, idx: std::ops::Range<usize>) -> Csr {
    let mut t = Vec::new();
    for i in idx.clone() {
        for p in a.rowptr[i]..a.rowptr[i + 1] {
            let j = a.colind[p];
            if idx.contains(&j) {
                t.push((i - idx.start, j - idx.start, a.vals[p]));
            }
        }
    }
    Csr::from_triplets(idx.len(), idx.len(), t)
}

/// `block_diagonal_lu_preconditioner`: LU of one diagonal block (taken at relative position
/// `relu`) applied to every block, plus the remainder block.
fn block_diagonal_lu(a: &Csr, o: &PfQ1Options, periods: usize) -> Result<(usize, SparseLu, Option<SparseLu>), String> {
    let nn = a.nrows;
    let ny = nn / periods;
    let lusize = if o.block_diagonal_lu_nperiods > 0 {
        if o.block_diagonal_lu_nperiods > periods {
            return Err("block_diagonal_lu preconditioner: block_diagonal_lu_nperiods option is greater than the periods options".into());
        }
        ny * o.block_diagonal_lu_nperiods
    } else if o.block_diagonal_lu_nlu != 0 {
        if o.block_diagonal_lu_nlu < 0 {
            return Err(
                "block_diagonal_lu preconditioner: negative value of block_diagonal_lu_nlu option is not supported"
                    .into(),
            );
        }
        let l = (nn / o.block_diagonal_lu_nlu as usize) / ny * ny;
        if l == 0 {
            return Err("block_diagonal_lu preconditioner: block_diagonal_lu_nlu option is too large".into());
        }
        l
    } else {
        if nn < o.block_diagonal_lu_maxlu {
            return Err("block_diagonal_lu preconditioner: too small problem; either decrease block_diagonal_lu_maxlu option, or use block_diagonal_lu_nperiods or block_diagonal_lu_nlu options".into());
        }
        if ny > o.block_diagonal_lu_maxlu {
            return Err("block_diagonal_lu preconditioner: too large problem; either increase block_diagonal_lu_maxlu option, or use block_diagonal_lu_nperiods or block_diagonal_lu_nlu options".into());
        }
        o.block_diagonal_lu_maxlu / ny * ny
    };
    if !(0.0..=1.0).contains(&o.block_diagonal_lu_relu) {
        return Err("block_diagonal_lu preconditioner: block_diagonal_lu_relu option must be between 0 and 1".into());
    }
    let start = (((nn / lusize) as f64 - 1.0) * o.block_diagonal_lu_relu).floor() as usize * lusize;
    let main = factor(&sub_csr(a, start..start + lusize)).ok_or("block_diagonal_lu preconditioner: singular block")?;
    let r = nn % lusize;
    let rem = if r > 0 {
        Some(factor(&sub_csr(a, nn - r..nn)).ok_or("block_diagonal_lu preconditioner: singular block")?)
    } else {
        None
    };
    Ok((lusize, main, rem))
}

fn iter_error(flag: i32) -> String {
    match flag {
        1 => "Maximum number of iterations exceeded in GMRES/BiCGStab. You may want to increase the iter_maxit option or the gmres_restart option (if using stack_solve_algo=2 for the latter), or decrease iter_tol.",
        2 => "The preconditioner matrix is ill conditioned in GMRES/BiCGStab. You should try another preconditioner.",
        3 => "No progress between two successive iterations in GMRES/BiCGStab. You may want to increase the iter_tol option.",
        _ => "One of the scalar quantities calculated by GMRES/BiCGStab became too small or too large. Try another preconditioner or algorithm.",
    }
    .into()
}

/// Port of `lin_solve.m` (A x = b). `Err` carries the MATLAB error message.
pub fn lin_solve(
    a: &Csr,
    b: &[f64],
    algo: u8,
    o: &PfQ1Options,
    tolf: f64,
    periods: usize,
    first: &mut Option<FirstLu>,
) -> Result<Vec<f64>, String> {
    let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if nb < f64::EPSILON.sqrt() {
        return Ok(vec![0.0; b.len()]);
    }
    let nn = b.len();
    let iter_tol = o.iter_tol.unwrap_or(tolf / norm_inf(b) / 10.0);
    let mv = |x: &[f64]| a.mul_vec(x);
    let mut x = match algo {
        2 | 3 => {
            let r: IterResult = match o.preconditioner.as_str() {
                "first_iter_lu" if first.is_none() => {
                    let f = factor(a).ok_or("Matrix is singular to working precision.")?;
                    let mut x = b.to_vec();
                    f.solve(&mut x);
                    *first = Some(FirstLu(f));
                    IterResult { x, flag: 0, relres: 0.0, iter: 0 }
                }
                pre => {
                    let maxit = o.iter_maxit.min(nn);
                    let lu_cur;
                    let bd;
                    let prec: Box<dyn Fn(&[f64]) -> Vec<f64> + '_> = match pre {
                        "first_iter_lu" => {
                            let f = &first.as_ref().unwrap().0;
                            Box::new(move |v: &[f64]| {
                                let mut z = v.to_vec();
                                f.solve(&mut z);
                                z
                            })
                        }
                        "block_diagonal_lu" => {
                            bd = block_diagonal_lu(a, o, periods)?;
                            let (ls, main, rem) = (&bd.0, &bd.1, &bd.2);
                            Box::new(move |v: &[f64]| {
                                let mut z = v.to_vec();
                                let nfull = nn / ls * ls;
                                for k in 0..nn / ls {
                                    main.solve(&mut z[k * ls..(k + 1) * ls]);
                                }
                                if let Some(r) = rem {
                                    r.solve(&mut z[nfull..]);
                                }
                                z
                            })
                        }
                        "incomplete_lu" => {
                            lu_cur = factor(a).ok_or("ilu: singular matrix")?;
                            let f = &lu_cur;
                            Box::new(move |v: &[f64]| {
                                let mut z = v.to_vec();
                                f.solve(&mut z);
                                z
                            })
                        }
                        p => return Err(format!("lin_solve: unknown preconditioner '{p}'")),
                    };
                    if algo == 2 {
                        gmres(&mv, Some(&*prec), b, o.gmres_restart.min(nn), iter_tol, maxit, None)
                    } else {
                        bicgstab(&mv, Some(&*prec), b, iter_tol, maxit, None)
                    }
                }
            };
            if r.flag != 0 {
                return Err(iter_error(r.flag));
            }
            r.x
        }
        10 => match first {
            None => {
                let f = factor(a).ok_or("Matrix is singular to working precision.")?;
                let mut x = b.to_vec();
                f.solve(&mut x);
                *first = Some(FirstLu(f));
                x
            }
            Some(FirstLu(f)) => {
                let prec = |v: &[f64]| {
                    let mut z = v.to_vec();
                    f.solve(&mut z);
                    z
                };
                cgs(&mv, Some(&prec), b, iter_tol, 150).x
            }
        },
        // 0, 8 (ParU), 9 (Pardiso): direct sparse solves
        _ => {
            let f = factor(a).ok_or("Matrix is singular to working precision.")?;
            let mut x = b.to_vec();
            f.solve(&mut x);
            x
        }
    };
    if !o.allow_nonfinite_values {
        x.iter_mut().filter(|v| !v.is_finite()).for_each(|v| *v = 0.0);
    }
    Ok(x)
}

/// `lin_solve_robust` of `sim1.m`: `A\b`, then BiCGStab (from it, then from 0), GMRES (from
/// the best, then from 0), then the pseudo-inverse.
pub fn lin_solve_robust(a: &Csr, b: &[f64], o: &PfQ1Options) -> Vec<f64> {
    let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if nb < f64::EPSILON.sqrt() {
        return vec![0.0; b.len()];
    }
    let nn = b.len();
    let fix = |x: &mut Vec<f64>| {
        if !o.allow_nonfinite_values {
            x.iter_mut().filter(|v| !v.is_finite()).for_each(|v| *v = 0.0);
        }
    };
    let mv = |x: &[f64]| a.mul_vec(x);
    let mut x = match factor(a) {
        Some(f) => {
            let mut x = b.to_vec();
            f.solve(&mut x);
            x
        }
        None => vec![f64::INFINITY; nn],
    };
    fix(&mut x);
    let (bmaxit, gmaxit) = (nn.min(20), nn.min(10));
    let r = bicgstab(&mv, None, b, 1e-6, bmaxit, Some(&x));
    if r.flag == 0 {
        return r.x;
    }
    let (old_x, old_relres) = (r.x, r.relres);
    let r = bicgstab(&mv, None, b, 1e-6, bmaxit, None);
    if r.flag == 0 {
        return r.x;
    }
    let mut x = r.x;
    fix(&mut x);
    if old_relres < r.relres {
        x = old_x;
    }
    let r = gmres(&mv, None, b, nn, 1e-6, gmaxit, Some(&x));
    if r.flag == 0 {
        return r.x;
    }
    let mut x = r.x;
    fix(&mut x);
    let (old_x, old_relres) = (x, r.relres);
    let r = gmres(&mv, None, b, nn, 1e-6, gmaxit, None);
    if r.flag == 0 {
        return r.x;
    }
    let mut x = r.x;
    fix(&mut x);
    let (mut x, mut relres) = if old_relres < r.relres { (old_x, old_relres) } else { (x, r.relres) };
    let (old_x, old_relres) = (x.clone(), relres);
    // pinv(full(A)) * b
    let ad = a.to_dense();
    let svd = ad.clone().svd(true, true);
    let ptol = f64::EPSILON * nn as f64 * svd.singular_values.max();
    if let Ok(p) = svd.pseudo_inverse(ptol) {
        x = (p * crate::linalg::Vector::from_column_slice(b)).iter().copied().collect();
        fix(&mut x);
        let rr: Vec<f64> = b.iter().zip(a.mul_vec(&x)).map(|(u, v)| u - v).collect();
        relres = rr.iter().map(|v| v * v).sum::<f64>().sqrt() / nb;
        if old_relres < relres {
            x = old_x;
        }
    } else {
        x = old_x;
        relres = old_relres;
    }
    let _ = relres;
    x
}

/// `check_Jacobian_for_singularity` (`sim1.m`): rank of the stacked Jacobian, collinear
/// variables and equations.
pub fn check_jacobian_for_singularity(a: &Csr, endo_names: &[String], tol: Option<f64>) {
    let j = a.to_dense();
    let (m, n) = (j.nrows(), j.ncols());
    let svd = j.clone().svd(true, true);
    let s = &svd.singular_values;
    let smax = s.max();
    let tol = tol.unwrap_or(m.max(n) as f64 * eps_of(smax));
    let rank = s.iter().filter(|&&v| v > tol).count();
    if rank >= m {
        println!("sim1:  The Jacobian of the dynamic model has full rank.");
        return;
    }
    println!("sim1:  The Jacobian of the dynamic model is singular");
    println!("sim1:  there is {} collinear relationships between the variables and the equations", n - rank);
    let v_t = svd.v_t.as_ref().unwrap();
    let u = svd.u.as_ref().unwrap();
    // null(J): right singular vectors of the zero singular values (sorted decreasingly)
    let mut order: Vec<usize> = (0..s.len()).collect();
    order.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap());
    let ny = endo_names.len();
    let ncol: Vec<Vec<f64>> = order[rank..].iter().map(|&k| v_t.row(k).iter().copied().collect()).collect();
    let n_rel = ncol.len();
    for (i, c) in ncol.iter().enumerate() {
        if n_rel > 1 {
            println!("Relation {}", i + 1);
        }
        println!("Collinear variables:");
        let mut k = Vec::new();
        for jj in 1..=10 {
            k = (0..n).filter(|&r| c[r].abs() > 10f64.powi(-jj)).collect();
            let worst = (0..m).map(|r| k.iter().map(|&q| j[(r, q)] * c[q]).sum::<f64>().abs()).fold(0.0, f64::max);
            if worst < 1e-6 {
                break;
            }
        }
        for q in k {
            println!("{}", endo_names[q % ny]);
        }
    }
    let neq: Vec<Vec<f64>> =
        order[rank..].iter().filter(|&&k| k < u.ncols()).map(|&k| u.column(k).iter().copied().collect()).collect();
    let n_rel = neq.len();
    for (i, c) in neq.iter().enumerate() {
        if n_rel > 1 {
            println!("Relation {}", i + 1);
        }
        println!("Collinear equations");
        let mut k = Vec::new();
        for jj in 1..=10 {
            k = (0..m).filter(|&r| c[r].abs() > 10f64.powi(-jj)).collect();
            let worst = (0..n).map(|q| k.iter().map(|&r| j[(r, q)] * c[r]).sum::<f64>().abs()).fold(0.0, f64::max);
            if worst < 1e-6 {
                break;
            }
        }
        for r in k {
            println!("Equation {:5}, period {:5}", r % ny + 1, r / ny + 1);
        }
    }
}

fn eps_of(x: f64) -> f64 {
    if x == 0.0 { f64::MIN_POSITIVE } else { 2f64.powi(x.abs().log2().floor() as i32 - 52) }
}

// ---- sim1.m with the lin_solve variants ------------------------------------------------------

/// `sim1.m` for `stack_solve_algo` 2, 3, 8, 9, 10, `robust_lin_solve`, `check_jacobian_singularity`
/// (with the horizon truncation of `endogenous_terminal_period`).
pub fn sim1(
    model: &Model,
    o: &SimulOptions,
    endo: &mut Mat,
    exo: &Mat,
    params: &[f64],
    ml: usize,
) -> (bool, f64, usize) {
    let q = &o.q1;
    let verbose = o.verbosity > 0 && !o.noprint;
    let n = model.n_endo();
    let mut periods = o.periods;
    let (y0, mut yt, mut y) = conditions(endo, ml, periods);
    if verbose {
        println!();
        println!("{}", "-".repeat(56));
        println!("MODEL SIMULATION:");
        println!();
    }
    let h1 = Instant::now();
    let mut iter = 1;
    let mut converged = false;
    let mut first = None;
    let mut st = Stacked::new(model);
    let mut err = f64::NAN;
    let mut failed = false;
    while !(converged || iter > o.maxit) {
        let h2 = Instant::now();
        let (mut res, a) = st.eval(&y, &y0, &yt, &|t| exo_row(exo, t + ml), params, true);
        let mut a = a.unwrap();
        if iter == 1 && q.check_jacobian_singularity {
            check_jacobian_for_singularity(&a, &model.endo_names, q.jacobian_tolerance);
        }
        if o.endogenous_terminal_period && iter > 1 {
            for it in 1..=periods {
                if norm_inf(&res[(it - 1) * n..it * n]) < o.tolf / 1e7 {
                    if it < periods {
                        res.truncate(it * n);
                        a = sub_csr(&a, 0..it * n);
                        yt = y[it * n..(it + 1) * n].to_vec();
                        for t in it..periods {
                            for j in 0..n {
                                endo[(j, ml + t)] = y[t * n + j];
                            }
                        }
                        y.truncate(it * n);
                        periods = it;
                    }
                    break;
                }
            }
        }
        err = norm_inf(&res);
        if err < o.tolf {
            converged = true;
        } else {
            let dy = if q.robust_lin_solve {
                lin_solve_robust(&a, &res, q)
            } else {
                match lin_solve(&a, &res, o.stack_solve_algo, q, o.tolf, periods, &mut first) {
                    Ok(d) => d,
                    Err(e) => {
                        println!("ERROR: {e}");
                        failed = true;
                        break;
                    }
                }
            };
            for (v, d) in y.iter_mut().zip(&dy) {
                *v -= d;
            }
        }
        if verbose {
            println!("Iter: {},\t err. = {},\t time = {}", iter, fmt_g(err), fmt_g(h2.elapsed().as_secs_f64()));
        }
        iter += 1;
    }
    for t in 0..periods {
        for j in 0..n {
            endo[(j, ml + t)] = y[t * n + j];
        }
    }
    if o.endogenous_terminal_period {
        err = compute_maxerror(model, endo, exo, params, ml, o.periods);
    }
    let success = converged && !failed && y.iter().all(|v| v.is_finite());
    if verbose {
        println!();
        println!("Total time of simulation: {}.", fmt_g(h1.elapsed().as_secs_f64()));
        if converged && !success {
            println!("Simulation terminated with NaN or Inf in the residuals or endogenous variables.");
            println!(
                "There is most likely something wrong with your model. Try model_diagnostics or another simulation method."
            );
            println!("{}", "-".repeat(105));
        } else if success {
            println!("{}", "-".repeat(56));
        } else {
            println!("Maximum number of iterations is reached (modify option maxit).");
            println!("{}", "-".repeat(62));
        }
        println!();
    }
    (success, err, iter)
}

// ---- sim1_lbj.m -----------------------------------------------------------------------------

/// Port of `sim1_lbj.m` (`stack_solve_algo` 1 and 6): Laffargue–Boucekkine–Juillard forward
/// elimination over the compressed dynamic Jacobian, then `back_subst_lbj`.
pub fn sim1_lbj(
    model: &Model,
    o: &SimulOptions,
    endo: &mut Mat,
    exo: &Mat,
    params: &[f64],
    ml: usize,
) -> (bool, f64, usize) {
    sim1_lbj_controlled(model, o, endo, &mut exo.clone(), params, ml, None)
}

/// `sim1_lbj.m` with a `controlled_paths_by_period` (`perfect_foresight_controlled_paths` or
/// `shock_paths` + `exogenize`): the exogenized endogenous variables are fixed to their values,
/// their Jacobian columns are replaced by those of the endogenized exogenous variables (and
/// zeroed in the neighbouring periods), and the exogenous variables absorb the Newton step.
/// Convergence is on the step size (`dynatol.x`), as in `sim1_lbj.m`.
#[allow(clippy::too_many_arguments)]
pub fn sim1_lbj_controlled(
    model: &Model,
    o: &SimulOptions,
    endo: &mut Mat,
    exo: &mut Mat,
    params: &[f64],
    ml: usize,
    cp: Option<&[crate::ep::controlled::ControlledPeriod]>,
) -> (bool, f64, usize) {
    let lli = &model.lead_lag_incidence;
    let ny = model.n_endo();
    let iyp: Vec<usize> = (0..ny).filter(|&j| lli[j][0] > 0).collect();
    let iyf: Vec<usize> = (0..ny).filter(|&j| lli[j][2] > 0).collect();
    let (nyp, nyf) = (iyp.len(), iyf.len());
    let nrc = nyf + 1;
    let periods = o.periods;
    let verbose = o.verbosity > 0;
    // compressed columns: [lags (lli order), current (all), leads (lli order)]
    let mut lag_cols: Vec<(usize, usize)> = iyp.iter().map(|&j| (lli[j][0], j)).collect();
    lag_cols.sort();
    let mut lead_cols: Vec<(usize, usize)> = iyf.iter().map(|&j| (lli[j][2], j)).collect();
    lead_cols.sort();
    let lag_vars: Vec<usize> = lag_cols.iter().map(|x| x.1).collect();
    let lead_vars: Vec<usize> = lead_cols.iter().map(|x| x.1).collect();
    // icp of sim1_lbj: rows of c for the lagged variables (iyp, in declaration order) — the lag
    // columns of the Jacobian are in lli order, which is declaration order for the lag block
    let pos_in_lag: Vec<usize> = lag_vars.clone();
    if verbose {
        println!("{}", "-".repeat(56));
        println!("MODEL SIMULATION :");
    }
    let h1 = Instant::now();
    let mut success = false;
    let mut iter_done = 0;
    let nx = model.n_exo();
    if let Some(cp) = cp {
        for (p, c) in cp.iter().enumerate().take(periods) {
            for (&j, &v) in c.exogenize.iter().zip(&c.values) {
                endo[(j, p + ml)] = v;
            }
        }
    }
    let mut g1 = Mat::zeros(model.n_equations(), 3 * ny + nx);
    let mut ws = model.workspace();
    let mut d1 = vec![0.0; ny];
    let mut y3 = vec![0.0; 3 * ny];
    for iter in 1..=o.maxit {
        iter_done = iter;
        let h2 = Instant::now();
        let mut c = Mat::zeros(ny * periods, nrc);
        for p in 0..periods {
            let it = ml + p;
            for b in 0..3 {
                for j in 0..ny {
                    y3[b * ny + j] = endo[(j, it + b - 1)];
                }
            }
            let x = exo_row(exo, it);
            model.dynamic_residuals(&y3, &x, params, &mut d1);
            model.dynamic_jacobian_y3_with(&y3, &x, params, &mut ws, &mut g1);
            if let Some(cp) = cp {
                // dynamicmodel() of sim1_lbj.m
                if p > 0 {
                    for &j in &cp[p - 1].exogenize {
                        g1.column_mut(j).fill(0.0);
                    }
                }
                for (&j, &k) in cp[p].exogenize.iter().zip(&cp[p].endogenize) {
                    let col = g1.column(3 * ny + k).into_owned();
                    g1.set_column(ny + j, &col);
                }
                if p + 1 < periods {
                    for &j in &cp[p + 1].exogenize {
                        g1.column_mut(2 * ny + j).fill(0.0);
                    }
                }
            }
            // jacobian = [g1(:, iz), -d1]: lag part (ny x nyp), current (ny x ny), rhs block [lead, -d1]
            let mut cur = Mat::from_fn(ny, ny, |i, j| g1[(i, ny + j)]);
            let mut rhs = Mat::zeros(ny, nrc);
            for (k, &j) in lead_vars.iter().enumerate() {
                for i in 0..ny {
                    rhs[(i, k)] = g1[(i, 2 * ny + j)];
                }
            }
            for i in 0..ny {
                rhs[(i, nyf)] = -d1[i];
            }
            if p > 0 {
                // jacobian(:, [isf nrs]) -= jacobian(:, isp) * c(icp, :)
                let lagj = Mat::from_fn(ny, nyp, |i, k| g1[(i, lag_vars[k])]);
                let cprev = Mat::from_fn(nyp, nrc, |k, col| c[((p - 1) * ny + pos_in_lag[k], col)]);
                let prod = &lagj * &cprev;
                // isf = current-block columns of the forward-looking variables, nrs = rhs column
                for (k, &j) in lead_vars.iter().enumerate() {
                    for i in 0..ny {
                        cur[(i, j)] -= prod[(i, k)];
                    }
                }
                for i in 0..ny {
                    rhs[(i, nyf)] -= prod[(i, nyf)];
                }
            }
            let sol = match crate::linalg::solve(&cur, &rhs) {
                Some(s) => s,
                None => Mat::from_element(ny, nrc, f64::NAN),
            };
            for i in 0..ny {
                for col in 0..nrc {
                    c[(p * ny + i, col)] = sol[(i, col)];
                }
            }
        }
        // back_subst_lbj
        for p in (0..periods.saturating_sub(1)).rev() {
            for i in 0..ny {
                let mut s = c[(p * ny + i, nyf)];
                for (k, &j) in iyf.iter().enumerate() {
                    s -= c[(p * ny + i, k)] * c[((p + 1) * ny + j, nyf)];
                }
                c[(p * ny + i, nyf)] = s;
            }
        }
        let mut err_x = 0.0f64;
        for p in 0..periods {
            for j in 0..ny {
                let d = c[(p * ny + j, nyf)];
                endo[(j, ml + p)] += d;
                err_x = if d.is_nan() || err_x.is_nan() { f64::NAN } else { err_x.max(d.abs()) };
            }
        }
        if let Some(cp) = cp {
            for (p, cpp) in cp.iter().enumerate().take(periods) {
                if cpp.endogenize.is_empty() {
                    continue;
                }
                for ((&j, &k), &v) in cpp.exogenize.iter().zip(&cpp.endogenize).zip(&cpp.values) {
                    endo[(j, ml + p)] = v;
                    exo[(ml + p, k)] += c[(p * ny + j, nyf)];
                }
            }
        }
        if verbose {
            println!("Iter: {},\t err. = {}, \t time = {}", iter, num2str(err_x), num2str(h2.elapsed().as_secs_f64()));
        }
        if err_x < o.tolx {
            success = true;
            break;
        }
    }
    let err_f = compute_maxerror(model, endo, exo, params, ml, periods);
    if verbose {
        println!("\nTotal time of simulation: {}", num2str(h1.elapsed().as_secs_f64()));
        if success {
            println!("{}", "-".repeat(56));
        } else {
            println!("Maximum number of iterations is reached (modify option maxit).");
            println!("{}", "-".repeat(62));
        }
        println!();
    }
    (success, err_f, iter_done)
}

// ---- solve_stacked_problem.m (stack_solve_algo = 7) -----------------------------------------

/// `solve_stacked_problem.m`: `dynare_solve` on the stacked system (`solve_algo`), or `lmmcp` on
/// the mixed complementarity problem (`solve_algo = 10`, the `lmmcp` option).
#[allow(clippy::too_many_arguments)]
pub fn solve_stacked_problem(
    model: &Model,
    o: &SimulOptions,
    solve_algo: u8,
    endo: &mut Mat,
    exo: &Mat,
    params: &[f64],
    model_params: &[f64],
    ml: usize,
) -> (bool, f64, usize) {
    let n = model.n_endo();
    let periods = o.periods;
    let (y0, yt, z) = conditions(endo, ml, periods);
    let mut st = Stacked::new(model);
    let exo_t = |t: usize| exo_row(exo, t + ml);
    if solve_algo == 10 || solve_algo == 11 {
        if solve_algo == 11 {
            println!("ERROR: PATH can't be provided with Dynare (solve_algo=11); using LMMCP (solve_algo=10) instead.");
        }
        let cc = match Complementarity::new(model, model_params) {
            Ok(c) => c,
            Err(e) => {
                println!("ERROR: {e}");
                return (false, f64::NAN, 0);
            }
        };
        let lb: Vec<f64> = (0..periods).flat_map(|_| cc.lb.iter().copied()).collect();
        let ub: Vec<f64> = (0..periods).flat_map(|_| cc.ub.iter().copied()).collect();
        let eqi: Vec<usize> = (0..periods).flat_map(|t| cc.eq_index.iter().map(move |&k| t * n + k)).collect();
        let mut f = |y: &[f64], jac: bool| {
            let (r, a) = st.eval(y, &y0, &yt, &exo_t, params, jac);
            let r2: Vec<f64> = eqi.iter().map(|&k| r[k]).collect();
            let a2 = a.map(|a| {
                let mut t = Vec::with_capacity(a.vals.len());
                for (row, &k) in eqi.iter().enumerate() {
                    for p in a.rowptr[k]..a.rowptr[k + 1] {
                        t.push((row, a.colind[p], a.vals[p]));
                    }
                }
                Csr::from_triplets(a.nrows, a.ncols, t)
            });
            (r2, a2)
        };
        let lo = LmmcpOptions::default();
        let r = lmmcp(&mut f, &z, &lb, &ub, &lo);
        let y = r.x;
        for t in 0..periods {
            for j in 0..n {
                endo[(j, ml + t)] = y[t * n + j];
            }
        }
        // residuals (mcp order), those of bounded variables at a bound set to 0 (all periods)
        let mut res = f(&y, false).0;
        for t in 0..periods {
            for j in 0..n {
                if (cc.lb[j].is_finite() || cc.ub[j].is_finite())
                    && (y[t * n + j] <= cc.lb[j] + f64::EPSILON || y[t * n + j] >= cc.ub[j] - f64::EPSILON)
                {
                    res[t * n + j] = 0.0;
                }
            }
        }
        return (r.exitflag == 1, norm_inf(&res), r.iterations);
    }
    let so = SteadyOptions { solve_algo, maxit: o.maxit, tolf: o.tolf, tolx: o.tolx, ..Default::default() };
    let st = std::cell::RefCell::new(st);
    let f = |y: &[f64]| {
        let (r, a) = st.borrow_mut().eval(y, &y0, &yt, &exo_t, params, true);
        (r, a.unwrap().to_dense())
    };
    let out = dynare_solve(&f, &z, &so);
    let y = out.x;
    for t in 0..periods {
        for j in 0..n {
            endo[(j, ml + t)] = y[t * n + j];
        }
    }
    let res = f(&y).0;
    (!out.errorflag, norm_inf(&res), 0)
}

// ---- sim1_purely_{backward,forward,static}.m --------------------------------------------------

/// Purely backward (`kind` 0), forward (1) or static (2) model: `dynare_solve` period by period
/// on the current-period variables (`options_.solve_algo`, `simul.maxit`, `dynatol`).
#[allow(clippy::too_many_arguments)]
pub fn sim1_purely(
    model: &Model,
    o: &SimulOptions,
    solve_algo: u8,
    kind: u8,
    endo: &mut Mat,
    exo: &Mat,
    params: &[f64],
    ml: usize,
) -> (bool, f64, usize) {
    let n = model.n_endo();
    let periods = o.periods;
    let so = SteadyOptions { solve_algo, maxit: o.maxit, tolf: o.tolf, tolx: o.tolx, ..Default::default() };
    let mut success = true;
    let nx = model.n_exo();
    let order: Vec<usize> = if kind == 1 { (0..periods).rev().collect() } else { (0..periods).collect() };
    let mut ystat: Vec<f64> = endo.column(0).iter().copied().collect();
    for p in order {
        let it = ml + p;
        let (other, guess): (Vec<f64>, Vec<f64>) = match kind {
            0 => {
                let v: Vec<f64> = endo.column(it - 1).iter().copied().collect();
                (v.clone(), v)
            }
            1 => {
                let v: Vec<f64> = endo.column(it + 1).iter().copied().collect();
                (v.clone(), v)
            }
            _ => (vec![0.0; n], ystat.clone()),
        };
        let x = exo_row(exo, it);
        let f = |z: &[f64]| {
            let mut y3 = vec![0.0; 3 * n];
            match kind {
                0 => y3[..n].copy_from_slice(&other),
                1 => y3[2 * n..].copy_from_slice(&other),
                _ => {}
            }
            y3[n..2 * n].copy_from_slice(z);
            let mut r = vec![0.0; n];
            model.dynamic_residuals(&y3, &x, params, &mut r);
            let mut g = Mat::zeros(model.n_equations(), 3 * n + nx);
            model.dynamic_jacobian_y3_with(&y3, &x, params, &mut model.workspace(), &mut g);
            (r, Mat::from_fn(n, n, |i, j| g[(i, n + j)]))
        };
        let out = dynare_solve(&f, &guess, &so);
        if out.errorflag {
            success = false;
            if kind != 2 {
                let what = if kind == 0 { "backward" } else { "forward" };
                println!(
                    "sim1_purely_{what}: Nonlinear solver routine failed with errorcode={} in period {}",
                    out.errorcode,
                    it + 1
                );
                break;
            }
        }
        for j in 0..n {
            endo[(j, it)] = out.x[j];
        }
        if kind == 2 {
            ystat = out.x;
        }
    }
    let err = compute_maxerror(model, endo, exo, params, ml, periods);
    (success, err, 0)
}

/// The dispatcher called by `DetState::core`: `None` keeps stream D's direct stacked Newton
/// (`stack_solve_algo = 0`) and `sim1_linear`.
#[allow(clippy::too_many_arguments)]
pub fn core(
    model: &Model,
    o: &SimulOptions,
    global_solve_algo: u8,
    endo: &mut Mat,
    exo: &mut Mat,
    params: &[f64],
    model_params: &[f64],
    ml: usize,
    mlead: usize,
    linear: bool,
) -> Option<(bool, f64, usize)> {
    let q = &o.q1;
    let solve_algo = if q.lmmcp { 10 } else { global_solve_algo };
    let stack = if q.lmmcp { 7 } else { o.stack_solve_algo };
    if !q.lmmcp {
        let kind = match (ml > 0, mlead > 0) {
            (true, false) => Some(0),
            (false, true) => Some(1),
            (false, false) => Some(2),
            _ => None,
        };
        if let Some(k) = kind {
            return Some(sim1_purely(model, o, solve_algo, k, endo, exo, params, ml));
        }
    }
    match stack {
        0 | 2 | 3 | 8 | 9 | 10 => {
            if linear {
                return None;
            }
            if stack == 0 && !q.robust_lin_solve && !q.check_jacobian_singularity && !q.allow_nonfinite_values {
                return None;
            }
            Some(sim1(model, o, endo, exo, params, ml))
        }
        1 | 6 => Some(sim1_lbj(model, o, endo, exo, params, ml)),
        7 => {
            if linear && solve_algo != 10 {
                return None; // solve_stacked_linear_problem: the linear system, solved directly
            }
            Some(solve_stacked_problem(model, o, solve_algo, endo, exo, params, model_params, ml))
        }
        _ => {
            println!("ERROR: Invalid value of stack_solve_algo option!");
            Some((false, f64::NAN, 0))
        }
    }
}

/// `extended_path(lmmcp)`: `dynare_solve` with `solve_algo = 10` on an extended-path problem
/// (`extended_path_initialization.m` / `solve_stochastic_perfect_foresight_model_*.m`: the
/// residuals of every block of `ny` in `M_.dynamic_mcp_equations_reordering` order, the bounds
/// repeated per block). Returns (errorflag, iterations).
pub fn ep_lmmcp(
    sys: &mut dyn crate::ep::trust_region::System,
    jac: &mut crate::ep::sparse::SparseJac,
    x: &mut [f64],
    cc: &Complementarity,
) -> (bool, usize) {
    let ny = cc.lb.len();
    let nn = x.len();
    let nb = nn / ny;
    let lb: Vec<f64> = (0..nb).flat_map(|_| cc.lb.iter().copied()).collect();
    let ub: Vec<f64> = (0..nb).flat_map(|_| cc.ub.iter().copied()).collect();
    let eqi: Vec<usize> = (0..nb).flat_map(|b| cc.eq_index.iter().map(move |&k| b * ny + k)).collect();
    let mut pos = vec![0; nn];
    for (row, &k) in eqi.iter().enumerate() {
        pos[k] = row;
    }
    let mut f = |y: &[f64], want: bool| {
        let mut r = vec![0.0; nn];
        sys.eval(y, &mut r, if want { Some(&mut *jac) } else { None });
        let r2: Vec<f64> = eqi.iter().map(|&k| r[k]).collect();
        let j = want.then(|| {
            let c = &jac.csc;
            let mut t = Vec::with_capacity(c.vals.len());
            for col in 0..c.n {
                for p in c.colptr[col]..c.colptr[col + 1] {
                    t.push((pos[c.rowind[p]], col, c.vals[p]));
                }
            }
            Csr::from_triplets(nn, nn, t)
        });
        (r2, j)
    };
    let r = lmmcp(&mut f, x, &lb, &ub, &LmmcpOptions::default());
    x.copy_from_slice(&r.x);
    (r.exitflag != 1, r.iterations)
}

/// `Csc` re-export for tests.
pub fn csc_of(a: &Csr) -> Csc {
    a.to_csc()
}

#[cfg(test)]
mod tests {
    use super::num2str;

    #[test]
    fn num2str_matches_matlab() {
        assert_eq!(num2str(std::f64::consts::PI), "3.1416");
        assert_eq!(num2str(1e-10), "1e-10");
        assert_eq!(num2str(0.5), "0.5");
        assert_eq!(num2str(123456.789), "123456.789");
        assert_eq!(num2str(3.0), "3");
        assert_eq!(num2str(0.000123456), "0.00012346");
    }
}
