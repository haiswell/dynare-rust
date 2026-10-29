// SPDX-License-Identifier: GPL-3.0-or-later
//! `stoch_simul(partial_information)`: port of Dynare's `matlab/partial_information/`
//! (Pearlman, Currie and Levine 1986): `PCL_resol.m` / `dr1_PI.m` (the solution),
//! `PI_gensys.m` + `PI_gensys_singularC.m` + `PI_qzdiv.m` (the generalized Sims solver; the QZ
//! of `(I, G1pi)` is the complex Schur form of `G1pi`, its stable invariant subspace is taken
//! from the ordered real Schur form, which gives the same `nmat`), `disc_riccati_fast.m`,
//! `PCL_Part_info_irf.m` and `PCL_Part_info_moments.m`, and the `partial_information` branches
//! of `stoch_simul.m` (order forced to 1, `SOLUTION UNDER PARTIAL INFORMATION` block, no
//! decision rule display and no simulation, `options_.partial_information` reset afterwards).
//!
//! Agents observe the `varobs` variables (all endogenous variables when `varobs` is absent:
//! full information); IRFs are the responses to one-standard-deviation shocks
//! (`stderr = diag(Sigma_e^0.5)`), moments the unconditional ones of the filtered system.

use anyhow::{Result, anyhow, bail};
use serde_json::json;

use super::table::{dyntable, max_len};
use super::{OutputContext, StochSimulOptions, StochSimulResults, TexNames, irf, q2};
use crate::linalg::schur::schur;
use crate::linalg::{Mat, lu, mul, norm1, rcond_from_lu, solve, sym_eig};
use crate::modfile::ModFile;
use crate::solver::{DecisionRule, FirstOrderSolver, SolveError};

/// `dr.PI_*` of `dr1_PI.m`.
#[derive(Clone, Debug)]
pub struct PiSolution {
    /// `PI_ghx` (= `G1pi`), `PI_ghu` (`impact`).
    pub ghx: Mat,
    pub ghu: Mat,
    pub tt1: Mat,
    pub tt2: Mat,
    pub nmat: Mat,
    pub fl_rank: usize,
    pub nx: usize,
}

fn inv(a: &Mat) -> Mat {
    solve(a, &Mat::identity(a.nrows(), a.nrows()))
        .unwrap_or_else(|| Mat::from_element(a.nrows(), a.ncols(), f64::INFINITY))
}

fn rcond(a: &Mat) -> f64 {
    if a.nrows() == 0 {
        return f64::INFINITY;
    }
    let (f, _, info) = lu(a);
    if info > 0 { 0.0 } else { rcond_from_lu(&f, norm1(a)) }
}

/// `x / a` (MATLAB mrdivide, square `a`).
fn mrdiv(x: &Mat, a: &Mat) -> Mat {
    match solve(&a.transpose(), &x.transpose()) {
        Some(s) => s.transpose(),
        None => Mat::from_element(x.nrows(), x.ncols(), f64::INFINITY),
    }
}

fn eps_of(x: f64) -> f64 {
    if x == 0.0 {
        return f64::MIN_POSITIVE;
    }
    let e = x.abs().log2().floor() as i32;
    2f64.powi(e - 52)
}

/// MATLAB `null(A)`: the right singular vectors beyond the numerical rank.
fn null_space(a: &Mat) -> Mat {
    let n = a.ncols();
    let (s, v) = crate::identification::la::svd_v_full(a);
    let tol = a.nrows().max(n) as f64 * eps_of(s.first().copied().unwrap_or(0.0));
    let r = s.iter().filter(|&&x| x > tol).count();
    v.columns(r, n - r).into_owned()
}

fn rank(a: &Mat) -> usize {
    crate::identification::la::rank(a, None)
}

/// `A^0.5` for a symmetric positive semi-definite `A` (`M_.Sigma_e^0.5`).
pub fn sqrtm_sym(a: &Mat) -> Mat {
    let (w, v) = sym_eig(a);
    let d = Mat::from_diagonal(&w.map(|x| x.max(0.0).sqrt()));
    &v * d * v.transpose()
}

/// Block matrix from rows of blocks (all blocks of a row with the same height).
fn blocks(rows: &[Vec<&Mat>]) -> Result<Mat> {
    let widths: Vec<usize> = rows[0].iter().map(|b| b.ncols()).collect();
    let nc: usize = widths.iter().sum();
    let nr: usize = rows.iter().map(|r| r[0].nrows()).sum();
    let mut out = Mat::zeros(nr, nc);
    let mut r0 = 0;
    for row in rows {
        let h = row[0].nrows();
        let mut c0 = 0;
        for (k, b) in row.iter().enumerate() {
            if b.nrows() != h || b.ncols() != widths[k] {
                bail!("Dimensions of arrays being concatenated are not consistent.");
            }
            out.view_mut((r0, c0), (h, b.ncols())).copy_from(*b);
            c0 += b.ncols();
        }
        r0 += h;
    }
    Ok(out)
}

struct Singular {
    c1: Mat,
    c3: Mat,
    c4: Mat,
    c5: Mat,
    f1: Mat,
    f2: Mat,
    f3: Mat,
    f4: Mat,
    f5: Mat,
    uavinv: Mat,
    fl_rank: usize,
    v01: Mat,
    v02: Mat,
}

/// Port of `PI_gensys_singularC.m` (recursive reduction when `U02'*a1*V02` is singular).
#[allow(clippy::too_many_arguments)]
fn pi_gensys_singular_c(
    c1in: &Mat,
    c2in: &Mat,
    c3in: &Mat,
    c4in: &Mat,
    c5in: &Mat,
    f1: &Mat,
    f2: &Mat,
    f3: &Mat,
    f4: &Mat,
    f5: &Mat,
    v01: &Mat,
    v02: &Mat,
    level: usize,
) -> Result<Singular> {
    let level = level + 1;
    println!("level =\n\n     {level}\n");
    if level > 100 {
        bail!(" PI_gensys_singularC recurssion exceeeded its maximum of 100 iterations! ");
    }
    let (j0, k0, _) = crate::identification::la::svd_econ(c2in);
    let n = c2in.nrows();
    let k_rank = rank(&Mat::from_diagonal(&nalgebra::DVector::from_vec(k0)));
    let j2 = j0.columns(k_rank, n - k_rank).into_owned();
    let j2c1 = j2.transpose() * c1in;
    let m = null_space(&j2c1).transpose();
    let mj = blocks(&[vec![&m], vec![&j2c1]])?;
    let mjcinv = inv(&mj);
    let (sm1, sm2) = (m.ncols(), m.nrows());
    let m1 = mjcinv.view((0, 0), (sm1, sm2)).into_owned();
    let m2 = mjcinv.view((0, sm2), (sm1, mjcinv.ncols() - sm2)).into_owned();
    let fl_rank = rank(&m);
    let c5 = blocks(&[vec![c5in], vec![&(&j2c1 * f5)]])?;
    let c4 = blocks(&[vec![c4in, &(c3in * &m2)], vec![&(&j2c1 * f4), &(&j2c1 * f3 * &m2)]])?;
    let c3 = blocks(&[vec![&(c3in * &m1)], vec![&(&j2c1 * f3 * &m1)]])?;
    let c2 = blocks(&[
        vec![c2in, &(c1in * &m2)],
        vec![&(j2.transpose() * (c4in + c1in * f2)), &(j2.transpose() * (c3in + c1in * f1) * &m2)],
    ])?;
    let c1 = blocks(&[vec![&(c1in * &m1)], vec![&(j2.transpose() * (c3in + c1in * f1) * &m1)]])?;
    let nf5 = &m * f5;
    let nf4 = blocks(&[vec![&(&m * f4), &(&m * f3 * &m2)]])?;
    let nf3 = &m * f3 * &m1;
    let nf2 = blocks(&[vec![&(&m * f2), &(&m * f1 * &m2)]])?;
    let nf1 = &m * f1 * &m1;
    let nv02 = blocks(&[vec![v02, &(v01 * &m2)]])?;
    let nv01 = v01 * &m1;
    let singular = rcond(&c2) < 1e-8;
    let uavinv = if singular { Mat::zeros(0, 0) } else { inv(&c2) };
    if singular || uavinv.iter().any(|x| x.is_infinite()) {
        return pi_gensys_singular_c(&c1, &c2, &c3, &c4, &c5, &nf1, &nf2, &nf3, &nf4, &nf5, &nv01, &nv02, level);
    }
    Ok(Singular { c1, c3, c4, c5, f1: nf1, f2: nf2, f3: nf3, f4: nf4, f5: nf5, uavinv, fl_rank, v01: nv01, v02: nv02 })
}

/// Port of `PI_gensys(a0, a1, a2, c, PSI, NX)` as `dr1_PI.m` calls it (`a0` the lead Jacobian,
/// `a1` the current one, `a2` minus the lag one). `Ok(Err(eu))` when `eu ~= [1;1]`.
pub fn pi_gensys(a0: &Mat, a1: &Mat, a2: &Mat, psi: &Mat, nx: usize) -> Result<Result<PiSolution, i32>> {
    let realsmall = 1e-6;
    let n = a0.nrows();
    let (u0, s0, v0) = crate::identification::la::svd_econ(a0);
    let mut fl_rank = rank(&Mat::from_diagonal(&nalgebra::DVector::from_vec(s0.clone())));
    let u01 = u0.columns(0, fl_rank).into_owned();
    let u02 = u0.columns(fl_rank, n - fl_rank).into_owned();
    let mut v01 = v0.columns(0, fl_rank).into_owned();
    let mut v02 = v0.columns(fl_rank, n - fl_rank).into_owned();
    let mut c1 = u02.transpose() * a1 * &v01;
    let c2 = u02.transpose() * a1 * &v02;
    let mut c3 = u02.transpose() * a2 * &v01;
    let mut c4 = u02.transpose() * a2 * &v02;
    let mut c5 = u02.transpose() * psi;
    let sinv = Mat::from_diagonal(&nalgebra::DVector::from_iterator(fl_rank, s0[..fl_rank].iter().map(|s| 1.0 / s)));
    let su = &sinv * u01.transpose();
    let mut f1 = &su * a1 * &v01;
    let mut f2 = &su * a1 * &v02;
    let mut f3 = &su * a2 * &v01;
    let mut f4 = &su * a2 * &v02;
    let mut f5 = &su * psi;
    let singular = rcond(&c2) < 1e-8;
    let mut uavinv = if singular { Mat::zeros(0, 0) } else { inv(&c2) };
    if singular || uavinv.iter().any(|x| x.is_infinite()) {
        let s = pi_gensys_singular_c(&c1, &c2, &c3, &c4, &c5, &f1, &f2, &f3, &f4, &f5, &v01, &v02, 0)?;
        (c1, c3, c4, c5, f1, f2, f3, f4, f5, uavinv, fl_rank, v01, v02) =
            (s.c1, s.c3, s.c4, s.c5, s.f1, s.f2, s.f3, s.f4, s.f5, s.uavinv, s.fl_rank, s.v01, s.v02);
    }
    if uavinv.iter().any(|x| !x.is_finite()) {
        println!("Warning: PI_gensys: Evading inversion of zero matrix UAVinv=inv(U02'*a1*V02)!");
        return Ok(Err(0));
    }
    let tt1 = v02;
    let tt2 = v01;
    let ff = &f2;
    let g11 = &uavinv * &c4;
    let g12 = &uavinv * &c3;
    let g13 = -(&uavinv * &c1);
    let g31 = -(ff * &g11) + &f4;
    let g32 = -(ff * &g12) + &f3;
    let g33 = -(ff * &g13) - &f1;
    let p1 = &uavinv * &c5;
    let p3 = -(ff * &p1) + &f5;
    let nmf = g11.ncols();
    let z = |r: usize, c: usize| Mat::zeros(r, c);
    let (ze11, ze12, ze134, ze31) = (z(nx, nx), z(nx, nmf), z(nx, fl_rank), z(fl_rank, nx));
    let (g21, g22, g23) = (z(fl_rank, nmf), z(fl_rank, fl_rank), Mat::identity(fl_rank, fl_rank));
    let g1pi = blocks(&[
        vec![&ze11, &ze12, &ze134, &ze134],
        vec![&p1, &g11, &g12, &g13],
        vec![&ze31, &g21, &g22, &g23],
        vec![&p3, &g31, &g32, &g33],
    ])?;
    let ss = g1pi.nrows();
    let mut impact = Mat::zeros(ss, nx);
    for k in 0..nx {
        impact[(k, k)] = 1.0;
    }
    if g1pi.iter().any(|x| !x.is_finite()) {
        println!("PI_Gensys: Input to QZ must not contain NaN or Inf.");
        return Ok(Err(-2));
    }
    // qz(I, G1pi): generalized eigenvalues |b_ii / a_ii| = |eig(G1pi)|
    let mut sch = schur(&g1pi)?;
    let roots = sch.eig_abs();
    let mut div = 1.01;
    let mut nunstab = 0;
    for &d in &roots {
        if 1.0 + realsmall < d && d <= div {
            div = 0.5 * (1.0 + d);
        }
        if d > div {
            nunstab += 1;
        }
    }
    if fl_rank != nunstab {
        println!("Number of unstable variables {nunstab}");
        println!("does not match number of expectational equations {fl_rank}");
        return Ok(Err(-2));
    }
    let stable: Vec<bool> = roots.iter().map(|&d| d <= div).collect();
    crate::linalg::schur::ordschur(&mut sch, &stable)?;
    let k = ss - fl_rank;
    let z2 = sch.u.columns(k, fl_rank).transpose();
    let z12 = z2.columns(0, k).into_owned();
    let z22 = z2.columns(k, fl_rank).into_owned();
    let nmat = inv(&z22) * z12;
    Ok(Ok(PiSolution { ghx: g1pi, ghu: impact, tt1, tt2, nmat, fl_rank, nx }))
}

/// Port of `disc_riccati_fast(F, D, R, H)` (doubling algorithm).
pub fn disc_riccati_fast(f: &Mat, d: &Mat, r: &Mat, h: &Mat) -> Result<Mat> {
    let tol = 1e-10;
    let mut p0 = h.clone();
    let mut x0 = f.clone();
    let mut y0 = if r.iter().all(|&x| x == 0.0) {
        println!("Warning: Dangerously evading inversion of zero matrix!");
        Mat::zeros(r.nrows(), r.ncols())
    } else {
        mrdiv(&d.transpose(), r) * d
    };
    let n = p0.nrows();
    let id = Mat::identity(n, n);
    let mut matd = 1.0;
    let mut count = 0;
    while matd > tol && count < 100 {
        let invpy = inv(&(&id + &p0 * &y0));
        let p1 = &x0 * &invpy * &p0 * x0.transpose() + &p0;
        let y1 = x0.transpose() * &y0 * &invpy * &x0 + &y0;
        let x1 = &x0 * &invpy * &x0;
        matd = (&p1 - &p0).iter().map(|x| x.abs()).sum();
        p0 = p1;
        x0 = x1;
        y0 = y1;
        count += 1;
    }
    if count == 100 {
        println!("matd =\n\n   {matd:.4e}\n");
        bail!("Riccati not converged fast enough!");
    }
    Ok((&p0 + p0.transpose()) * 0.5)
}

/// The filtering quantities shared by `PCL_Part_info_irf.m` and `PCL_Part_info_moments.m`.
struct Pcl {
    pd: usize,
    aa: Mat,
    cccc: Mat,
    pp: Mat,
    pdidpdrd: Mat,
}

/// The common part of the two PCL routines; `h` is `M_.H` (zero: no measurement errors).
fn pcl_core(pi: &PiSolution, sigma_e: &Mat, obs: &[usize], ny: usize, h: Option<&Mat>, mrdivide: bool) -> Result<Pcl> {
    let g1 = &pi.ghx;
    let (nx, fl) = (pi.nx, pi.fl_rank);
    let ss = g1.nrows();
    let pd = ss - pi.nmat.nrows();
    let mut ll = Mat::zeros(ny, ny);
    for (i, &o) in obs.iter().enumerate() {
        ll[(i, o)] = 1.0;
    }
    let sdx = sqrtm_sym(sigma_e);
    let mm = &pi.ghu * &sdx;
    let l1 = &ll * &pi.tt1;
    let l2 = &ll * &pi.tt2;
    let mm1 = mm.rows(0, ss - fl).into_owned();
    let u11 = &mm1 * mm1.transpose();
    let g12 = g1.rows(nx, ss - 2 * fl - nx).into_owned();
    let kk1 = &l1 * &g12;
    let k1 = kk1.columns(0, ss - fl).into_owned();
    let k2 = kk1.columns(ss - fl, fl).into_owned() + &l2;
    let a11 = g1.view((0, 0), (pd, pd)).into_owned();
    let a22 = g1.view((pd, pd), (ss - pd, ss - pd)).into_owned();
    let a12 = g1.view((0, pd), (pd, ss - pd)).into_owned();
    let a21 = g1.view((pd, 0), (ss - pd, pd)).into_owned();
    let (bb, ff) = if mrdivide { (mrdiv(&a12, &a22), mrdiv(&k2, &a22)) } else { (&a12 * inv(&a22), &k2 * inv(&a22)) };
    let qq = u11; // BB*U22*BB' + U11 with U22 = 0
    let aa = &a11 - &bb * &a21;
    let cccc = &a11 - &a12 * &pi.nmat;
    let dd = &k1 - &ff * &a21;
    // RR = FF*UFT + VV with UFT = U22*FF' = 0
    let mut rr = Mat::zeros(ny, ny);
    if let Some(h) = h {
        if h.iter().any(|&x| x != 0.0) {
            if h.shape() != (ny, ny) {
                bail!("Arrays have incompatible sizes for this operation.");
            }
            rr += h;
        }
    }
    if rr.iter().all(|&x| x == 0.0) {
        rr = Mat::identity(ny, ny) * 1.0e-6;
    }
    let aqds = &aa * &qq * dd.transpose(); // + SS = BB*UFT = 0
    let dqdr = &dd * &qq * dd.transpose() + &rr;
    let aqdq = if mrdivide { mrdiv(&aqds, &dqdr) } else { &aqds * inv(&dqdr) };
    let fff = &aa - &aqdq * &dd;
    let hh = &aa * &qq * aa.transpose() - &aqdq * aqds.transpose();
    let zsig0 = disc_riccati_fast(&fff, &dd, &dqdr, &hh)?;
    let pp = zsig0 + &qq;
    let dpdr = &dd * &pp * dd.transpose() + &rr;
    let pdidpdrd =
        if mrdivide { mrdiv(&(&pp * dd.transpose()), &dpdr) * &dd } else { &pp * dd.transpose() * inv(&dpdr) * &dd };
    Ok(Pcl { pd, aa, cccc, pp, pdidpdrd })
}

/// Port of `PCL_Part_info_irf(0, varobs, ivar, M_, dr, irfpers, ii)`: n_endo x `irfpers`
/// responses (declaration order) to a one-standard-deviation shock `ii`, and the
/// `irfmat` / `irfst` the routine saves.
pub fn pcl_part_info_irf(
    pi: &PiSolution,
    sigma_e: &Mat,
    obs: &[usize],
    ny: usize,
    irfpers: usize,
    ii: usize,
) -> Result<(Mat, Mat, Mat)> {
    let c = pcl_core(pi, sigma_e, obs, ny, None, false)?;
    let (nx, pd) = (pi.nx, c.pd);
    let ipd = Mat::identity(pd, pd) - &c.pdidpdrd;
    let gg = blocks(&[vec![&c.cccc, &((&c.aa - &c.cccc) * &ipd)], vec![&Mat::zeros(pd, pd), &(&c.aa * &ipd)]])?;
    let vv = blocks(&[vec![&pi.tt1, &pi.tt2]])?;
    let stderr = sqrtm_sym(sigma_e).diagonal();
    let mut irfmat = Mat::zeros(pi.tt1.nrows(), irfpers + 1);
    let mut irfst = Mat::zeros(gg.nrows(), irfpers + 1);
    for r in 0..pd {
        irfst[(r, 0)] = stderr[ii] * pi.ghu[(r, ii)];
        irfst[(pd + r, 0)] = stderr[ii] * pi.ghu[(r, ii)];
    }
    for jj in 1..=irfpers {
        let next = &gg * irfst.column(jj - 1);
        irfst.set_column(jj, &next);
        let x = next.rows(nx, pd - nx).into_owned();
        irfmat.set_column(jj - 1, &(&vv * x));
    }
    let y = irfmat.columns(0, irfpers).into_owned();
    Ok((y, irfmat, irfst))
}

/// What `PCL_Part_info_moments.m` computes.
#[derive(Clone, Debug)]
pub struct PclMoments {
    /// `COV_YR0` (all endogenous variables, declaration order).
    pub cov_yr0: Mat,
    /// `COR_Y` (None with `nocorr`).
    pub cor_y: Option<Mat>,
    /// `AutoCOR_YRk` (n_endo x ar).
    pub autocor: Mat,
    /// `oo_.autocorr{k}` (ivar x ivar).
    pub autocorr: Vec<Mat>,
    /// `oo_.contemporaneous_correlation`.
    pub contemporaneous_correlation: Option<Mat>,
    pub text: String,
}

/// Port of `PCL_Part_info_moments(M_, oo_, options_, varobs, dr, ivar)` (prints its tables).
#[allow(clippy::too_many_arguments)]
pub fn pcl_part_info_moments(
    pi: &PiSolution,
    sigma_e: &Mat,
    h: &Mat,
    obs: &[usize],
    endo_names: &[String],
    ivar: &[usize],
    o: &StochSimulOptions,
    tex: Option<&TexNames>,
) -> Result<PclMoments> {
    let ny = endo_names.len();
    let c = pcl_core(pi, sigma_e, obs, ny, Some(h), true)?;
    let pd = c.pd;
    let ipd = Mat::identity(pd, pd) - &c.pdidpdrd;
    // MSIG = lyapunov_solver(CCCC, CCCC, PDIDPDRD*PP, options_) (lyapunov_symm)
    let rhs = &c.cccc * (&c.pdidpdrd * &c.pp) * c.cccc.transpose();
    let msig = super::moments::LyapunovSchur::new(&c.cccc, o.qz_criterium)?.solve(&rhs, o.lyapunov_complex_threshold);
    let mut cov_p = blocks(&[vec![&c.pp, &c.pp], vec![&c.pp, &(&c.pp + &msig)]])?;
    cov_p = (&cov_p + cov_p.transpose()) * 0.5;
    let gam = blocks(&[vec![&(&c.aa * &ipd), &Mat::zeros(pd, pd)], vec![&((&c.aa - &c.cccc) * &ipd), &c.cccc]])?;
    let vv = blocks(&[vec![&pi.tt1, &pi.tt2]])?;
    let nn = vv.nrows();
    let np = cov_p.nrows();
    let omega = |p: &Mat| p.view((np - nn, np - nn), (nn, nn)).into_owned();
    let cov_yr0 = mul(&mul(&vv, false, &omega(&cov_p), false), false, &vv, true);
    let mut dg: Vec<f64> = cov_yr0.diagonal().iter().copied().collect();
    for d in dg.iter_mut() {
        if d.abs() < 1e-10 {
            *d = 0.0;
        }
    }
    let lab = |idx: &[usize], t: bool| -> Vec<String> {
        let names: Vec<String> = match (t, tex) {
            (true, Some(tn)) => idx.iter().map(|&i| tn.endo[i].clone()).collect(),
            _ => idx.iter().map(|&i| endo_names[i].clone()).collect(),
        };
        q2::transformed_labels(&names, o.loglinear, t)
    };
    let mut text = String::new();
    let pr = !o.noprint;
    if !o.nomoments && pr {
        let labels = lab(ivar, false);
        let z = Mat::from_fn(ivar.len(), 2, |i, j| if j == 0 { dg[ivar[i]].sqrt() } else { dg[ivar[i]] });
        let h = vec!["VARIABLE".to_string(), "STD. DEV.".into(), "VARIANCE".into()];
        text += &dyntable("THEORETICAL MOMENTS", &h, &labels, &z, max_len(&labels) + 2, 16, 10);
    }
    let mut delta: Option<Vec<f64>> = None;
    let mut cor_y = None;
    let mut contemporaneous_correlation = None;
    let mut i1: Vec<usize> = Vec::new();
    if !o.nocorr {
        let sd: Vec<f64> = dg.iter().map(|d| d.sqrt()).collect();
        let nz: Vec<usize> = (0..ny).filter(|&i| sd[i] > 1e-10).collect();
        let dl: Vec<f64> = (0..ny).map(|i| if nz.contains(&i) { 1.0 / sd[i] } else { 0.0 }).collect();
        let cy = Mat::from_fn(ny, ny, |i, j| dl[i] * cov_yr0[(i, j)] * dl[j]);
        i1 = nz.iter().copied().filter(|i| ivar.contains(i)).collect();
        if o.contemporaneous_correlation {
            contemporaneous_correlation = Some(Mat::from_fn(ivar.len(), ivar.len(), |a, b| cy[(ivar[a], ivar[b])]));
        }
        if pr {
            text.push('\n');
            let labels = lab(&i1, false);
            let mut h = vec!["Variables".to_string()];
            h.extend(labels.iter().cloned());
            let sub = Mat::from_fn(i1.len(), i1.len(), |a, b| cy[(i1[a], i1[b])]);
            text += &dyntable("MATRIX OF CORRELATIONS", &h, &labels, &sub, max_len(&labels) + 2, 8, 4);
            if let Some(t) = tex {
                let tl = lab(&i1, true);
                let mut th = vec!["Variables".to_string()];
                th.extend(tl.iter().cloned());
                q2::latex_table(
                    o,
                    &t.ctx,
                    "PCL_Part_info_moments.m",
                    "MATRIX OF CORRELATIONS",
                    "th_corr_matrix",
                    &th,
                    &tl,
                    &sub,
                    max_len(&tl) + 2,
                    8,
                    4,
                )?;
            }
        }
        delta = Some(dl);
        cor_y = Some(cy);
    }
    let ar = o.ar;
    let mut autocor = Mat::zeros(0, 0);
    let mut autocorr = Vec::new();
    if ar > 0 {
        let Some(dl) = delta.as_ref() else {
            // nocorr leaves DELTA undefined in PCL_Part_info_moments.m: MATLAB stops here,
            // after the moments table
            print!("{text}");
            bail!("Unrecognized function or variable 'DELTA'.");
        };
        autocor = Mat::zeros(nn, ar);
        let mut cp = cov_p.clone();
        for k in 0..ar {
            cp = &gam * &cp;
            let cyk = mul(&mul(&vv, false, &omega(&cp), false), false, &vv, true);
            autocorr
                .push(Mat::from_fn(ivar.len(), ivar.len(), |a, b| dl[ivar[a]] * cyk[(ivar[a], ivar[b])] * dl[ivar[b]]));
            for i in 0..nn {
                autocor[(i, k)] = cyk[(i, i)] / dg[i];
            }
        }
        if pr {
            let labels = lab(&i1, false);
            let mut h = vec!["Order ".to_string()];
            h.extend((1..=ar).map(|k| k.to_string()));
            let sub = Mat::from_fn(i1.len(), ar, |a, k| autocor[(i1[a], k)]);
            text += &dyntable("COEFFICIENTS OF AUTOCORRELATION", &h, &labels, &sub, max_len(&labels) + 2, 8, 4);
            if let Some(t) = tex {
                let tl = lab(&i1, true);
                let mut th = vec!["Order ".to_string()];
                th.extend((1..=ar).map(|k| k.to_string()));
                q2::latex_table(
                    o,
                    &t.ctx,
                    "PCL_Part_info_moments.m",
                    "COEFFICIENTS OF AUTOCORRELATION",
                    "th_autocorr_matrix",
                    &th,
                    &tl,
                    &sub,
                    max_len(&tl) + 2,
                    8,
                    4,
                )?;
            }
        }
    }
    Ok(PclMoments { cov_yr0, cor_y, autocor, autocorr, contemporaneous_correlation, text })
}

/// Port of `PCL_resol.m` / `dr1_PI.m` at a steady state: the partial-information solution.
/// `Ok(Err(info))` when Dynare's `info` is nonzero (4: the Blanchard-Kahn-type count failed).
pub fn dr1_pi(
    model: &crate::model::Model,
    ys: &[f64],
    params: &[f64],
    o: &StochSimulOptions,
) -> Result<Result<PiSolution, SolveError>> {
    if o.aim_solver {
        bail!("Anderson and Moore AIM solver is not compatible with Partial Information models");
    }
    let n = model.n_endo();
    let nx = model.n_exo();
    let g1 = model.dynamic_jacobian(ys, &vec![0.0; nx], params);
    let aa2 = g1.columns(0, n).into_owned();
    let aa1 = g1.columns(n, n).into_owned();
    let aa0 = g1.columns(2 * n, n).into_owned();
    let psi = -g1.columns(3 * n, nx).into_owned();
    match pi_gensys(&aa0, &aa1, &(-aa2), &psi, nx)? {
        Ok(s) => Ok(Ok(s)),
        Err(eu) => {
            if eu == 0 {
                bail!(
                    "Output argument \"nmat\" (and possibly others) not assigned a value in the execution with \"PI_gensys\" function."
                );
            }
            Ok(Err(SolveError::new((2 * eu).abs(), 1.0e8)))
        }
    }
}

/// The `partial_information` branch of `stoch_simul.m` (any requested order: forced to 1).
/// Prints like Dynare, writes the IRF CSV files / figures, Dynare's `.mat` files
/// (`<fname>_PCL_PtInfoIRFs_<k>_<shock>.mat`, `Output/<fname>_PCL_moments.mat`) and
/// `Output/<fname>_stoch_simul.json`.
pub fn stoch_simul_pi(
    mf: &ModFile,
    o: &StochSimulOptions,
    var_list: &[String],
    ctx: &OutputContext,
    point: Option<&[f64]>,
) -> Result<Option<StochSimulResults>> {
    let model = &mf.model;
    stoch_simul_pi_prologue(model, o)?;
    // PCL_resol: the steady state, then dr1_PI
    let (ys, params, sigma_e) = match point {
        None => (super::calibrated_steady_state(mf, &model.params)?, model.params.clone(), mf.sigma_e.clone()),
        Some(x) => {
            let dsge = crate::dsge::DsgeModel::new(mf, Some(x))?;
            match dsge.resolve(x) {
                Ok(r) => (r.steady_state, r.params, r.q),
                Err(e) => {
                    super::print_info(&SolveError::new(e.info, e.penalty));
                    return Ok(None);
                }
            }
        }
    };
    stoch_simul_pi_at(mf, o, var_list, ctx, ys, params, sigma_e)
}

/// The checks of `stoch_simul.m` that come before `PCL_resol.m` (order forced to 1).
pub fn stoch_simul_pi_prologue(model: &crate::model::Model, o: &StochSimulOptions) -> Result<()> {
    if model.n_exo() == 0 {
        bail!(
            "stoch_simul:: does not support having no varexo in the model. As a workaround you could define a dummy exogenous variable."
        );
    }
    if o.order != 1 && !model.linear {
        if model.dynamic_derivatives(2).hessian_eq_zero() {
            println!("Warning: stoch_simul: using order = 1 because Hessian is equal to zero");
        } else {
            println!("Warning: stoch_simul:: forcing order=1 since you are using partial_information");
        }
    }
    Ok(())
}

/// [`stoch_simul_pi`] after `PCL_resol.m`'s `evaluate_steady_state`, at the steady state `ys`
/// and parameters `params` it returned (a Ramsey model's come from `dyn_ramsey_static.m`).
pub fn stoch_simul_pi_at(
    mf: &ModFile,
    o: &StochSimulOptions,
    var_list: &[String],
    ctx: &OutputContext,
    ys: Vec<f64>,
    params: Vec<f64>,
    sigma_e: Mat,
) -> Result<Option<StochSimulResults>> {
    let model = &mf.model;
    let ivar = super::var_indices(model, var_list)?;
    let pi = match dr1_pi(model, &ys, &params, o)? {
        Ok(p) => p,
        Err(e) => {
            super::print_info(&e);
            return Ok(None);
        }
    };
    let solver = FirstOrderSolver::new(&model.lead_lag_incidence, model.n_exo(), Default::default());
    let s = &solver.structure;
    let summary_dr = DecisionRule {
        ys: ys.clone(),
        ghx: Mat::zeros(model.n_endo(), 0),
        ghu: Mat::zeros(model.n_endo(), 0),
        order_var: s.order_var.clone(),
        inv_order_var: s.inv_order_var.clone(),
        state_var: s.state_var.clone(),
        nstatic: s.nstatic,
        nfwrd: s.nfwrd,
        npred: s.npred,
        nboth: s.nboth,
        nspred: s.nspred,
        nsfwrd: s.nsfwrd,
        eigval: Vec::new(),
        eigval_abs: Vec::new(),
        sdim: 0,
    };
    let sol = super::LinearSolution { dr: summary_dr, sigma_e: sigma_e.clone(), params: params.clone() };
    let tex = o.tex.then(|| TexNames::from_modfile(mf, ctx));
    // PCL_varobs: options_.varobs, or every endogenous variable (only set when printing)
    let varobs: Option<Vec<String>> =
        (!o.noprint).then(|| if mf.varobs.is_empty() { model.endo_names.clone() } else { mf.varobs.clone() });
    if !o.noprint {
        let mut out = String::new();
        if !o.nomodelsummary {
            out += &super::model_summary(model, &sol);
            if let Some(t) = &tex {
                let mut h = vec!["Variables".to_string()];
                h.extend(t.exo.iter().cloned());
                q2::latex_table(
                    o,
                    &t.ctx,
                    "stoch_simul.m",
                    "MATRIX OF COVARIANCE OF EXOGENOUS SHOCKS",
                    "covar_ex_shocks",
                    &h,
                    &t.exo,
                    &sigma_e,
                    max_len(&t.exo) + 2,
                    10,
                    6,
                )?;
            }
        }
        out += "\nSOLUTION UNDER PARTIAL INFORMATION\n\n";
        if mf.varobs.is_empty() {
            out += " VAROBS LIST NOT SPECIFIED\n ASSUMED OBSERVED VARIABLES\n";
        } else {
            out += "OBSERVED VARIABLES\n";
        }
        for v in varobs.as_ref().unwrap() {
            out += &format!("    {v}\n");
        }
        out.push('\n');
        print!("{out}");
    }
    let need_varobs = || varobs.as_ref().ok_or_else(|| anyhow!("Unrecognized function or variable 'PCL_varobs'."));
    let obs_of = |vo: &[String]| -> Vec<usize> { vo.iter().filter_map(|v| model.endo_index(v)).collect() };
    let mut res = StochSimulResults { ivar: ivar.clone(), ..Default::default() };
    let mut moments = None;
    if !o.nomoments {
        let obs = obs_of(need_varobs()?);
        std::fs::create_dir_all(&ctx.out_dir)?;
        let m = pcl_part_info_moments(&pi, &sigma_e, &mf.h, &obs, &model.endo_names, &ivar, o, tex.as_ref())?;
        print!("{}", m.text);
        use crate::matfile::{MatOut, write};
        let empty = Mat::zeros(0, 0);
        write(
            &ctx.out_dir.join(format!("{}_PCL_moments.mat", ctx.fname)),
            &[
                ("COV_YR0", MatOut::Matrix(&m.cov_yr0)),
                ("AutoCOR_YRk", MatOut::Matrix(&m.autocor)),
                ("COR_Y", MatOut::Matrix(m.cor_y.as_ref().unwrap_or(&empty))),
            ],
        )?;
        res.contemporaneous_correlation = m.contemporaneous_correlation.clone();
        moments = Some(m);
    }
    if o.irf > 0 {
        let cs = super::vardec::lower_cholesky(&sigma_e, o.add_tiny_number_to_cholesky);
        let shocks = irf::irf_shock_indices(&model.exo_names, o.irf_shocks.as_deref())?;
        let labels: Vec<String> = ivar.iter().map(|&i| model.endo_names[i].clone()).collect();
        let gopts = o.graph_options(&ctx.dir, &ctx.fname);
        for &i in shocks.iter().filter(|&&i| cs[(i, i)] > 5e-7) {
            let obs = obs_of(need_varobs()?);
            let (mut y, irfmat, irfst) = pcl_part_info_irf(&pi, &sigma_e, &obs, model.n_endo(), o.irf, i)?;
            use crate::matfile::{MatOut, write};
            write(
                &ctx.dir.join(format!("{}_PCL_PtInfoIRFs_{}_{}.mat", ctx.fname, i + 1, model.exo_names[i])),
                &[("irfmat", MatOut::Matrix(&irfmat)), ("irfst", MatOut::Matrix(&irfst))],
            )?;
            if o.relative_irf {
                y *= 100.0 / cs[(i, i)];
            }
            let yv = y.select_rows(&ivar);
            let shock = &model.exo_names[i];
            irf::write_irf_csv(&ctx.out_dir, &ctx.fname, shock, &labels, &yv)?;
            if !o.nograph {
                for fig in irf::irf_figures(shock, &labels, &yv, o.relative_irf, o.irf_plot_threshold) {
                    crate::graphics::save(&fig, &ctx.fname, &gopts)?;
                }
            }
            res.irfs.push(irf::ShockIrf { shock: i, y: yv });
        }
    }
    write_json(model, &pi, &res, moments.as_ref(), ctx)?;
    if o.spectral_density {
        // UnivariateSpectralDensity.m reads oo_.dr.order_var, which dr1_PI does not set
        bail!("Unrecognized field name \"order_var\".");
    }
    Ok(Some(res))
}

/// `<out_dir>/<fname>_stoch_simul.json` of a partial-information run.
fn write_json(
    model: &crate::model::Model,
    pi: &PiSolution,
    r: &StochSimulResults,
    m: Option<&PclMoments>,
    ctx: &OutputContext,
) -> Result<()> {
    let names: Vec<&String> = r.ivar.iter().map(|&i| &model.endo_names[i]).collect();
    let m2v = |m: &Mat| -> Vec<Vec<f64>> { (0..m.nrows()).map(|i| m.row(i).iter().copied().collect()).collect() };
    // dr1_PI.m: oo_.dr.ghx = G1pi, oo_.dr.ghu = impact (the PCL state space, not Dynare's dr)
    let dr = json!({ "ghx": m2v(&pi.ghx), "ghu": m2v(&pi.ghu) });
    let mut j = json!({ "var_list": names, "exo_names": model.exo_names, "partial_information": true, "dr": dr });
    if !r.irfs.is_empty() {
        let mut irfs = serde_json::Map::new();
        for s in &r.irfs {
            for (k, n) in names.iter().enumerate() {
                irfs.insert(
                    format!("{}_{}", n, model.exo_names[s.shock]),
                    json!(s.y.row(k).iter().copied().collect::<Vec<_>>()),
                );
            }
        }
        j["irfs"] = serde_json::Value::Object(irfs);
    }
    if let Some(m) = m {
        j["COV_YR0"] = json!(m2v(&m.cov_yr0));
        if !m.autocorr.is_empty() {
            j["autocorr"] = json!(m.autocorr.iter().map(m2v).collect::<Vec<_>>());
        }
        if let Some(c) = &m.contemporaneous_correlation {
            j["contemporaneous_correlation"] = json!(m2v(c));
        }
    }
    std::fs::create_dir_all(&ctx.out_dir)?;
    std::fs::write(ctx.out_dir.join(format!("{}_stoch_simul.json", ctx.fname)), serde_json::to_string_pretty(&j)?)?;
    Ok(())
}
