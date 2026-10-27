// SPDX-License-Identifier: GPL-3.0-or-later
//! Theoretical moments of the pruned state space at order 2 and 3: port of
//! `+pruned_SS/pruned_state_space_system.m` (Andreasen, Fernández-Villaverde & Rubio-Ramírez
//! 2018; Mutschler 2018), without parameter derivatives, and of the second-order mean
//! correction of `moments/th_autocovariances.m` (Kim, Kim, Schaumburg & Sims 2008) used by
//! `disp_th_moments.m` at `order = 2` without pruning.
//!
//! The shocks are Gaussian: the 4th and 6th product moments (`prodmom.m` with
//! `quadruplication` / `Q6_plication`) come from Isserlis' theorem.

use super::KOrderDr;
use crate::linalg::Mat;
use anyhow::{Result, bail};

/// `z = c + A z(-1) + B inov`, `y = ys + d + C z(-1) + D inov` and its moments
/// (fields named as in `pruned_state_space_system.m`).
#[derive(Debug, Clone)]
pub struct PrunedStateSpace {
    pub a: Mat,
    pub b: Mat,
    pub c: Vec<f64>,
    pub cmat: Mat,
    pub dmat: Mat,
    pub d: Vec<f64>,
    pub var_inov: Mat,
    /// Unconditional mean of the selected variables.
    pub e_y: Vec<f64>,
    pub var_y: Mat,
    /// Autocovariances, `var_yi[i-1]` = lag `i`.
    pub var_yi: Vec<Mat>,
    pub corr_y: Mat,
    pub corr_yi: Vec<Mat>,
}

fn kron(a: &Mat, b: &Mat) -> Mat {
    a.kronecker(b)
}

fn vec_of(a: &Mat) -> Mat {
    Mat::from_column_slice(a.len(), 1, a.as_slice())
}

fn reshape(a: &Mat, r: usize, c: usize) -> Mat {
    assert_eq!(a.len(), r * c);
    Mat::from_column_slice(r, c, a.as_slice())
}

/// `pruned_SS.commutation(n, m)`: `K vec(X) = vec(X')` for `X` n x m.
pub fn commutation(n: usize, m: usize) -> Mat {
    let mut k = Mat::zeros(n * m, n * m);
    for i in 0..n {
        for j in 0..m {
            k[(j + m * i, i + n * j)] = 1.0;
        }
    }
    k
}

/// Gaussian product moments `E[u ⊗ ... ⊗ u]` (`k` = 4 or 6 factors, kron order), Isserlis.
fn gaussian_moments(s: &Mat, k: usize) -> Mat {
    let nu = s.nrows();
    let len = nu.pow(k as u32);
    let mut out = Mat::zeros(len, 1);
    let mut idx = vec![0usize; k];
    for c in 0..len {
        super::tensor::tuple_of(c, nu, k, &mut idx);
        out[(c, 0)] = isserlis(s, &idx);
    }
    out
}

fn isserlis(s: &Mat, idx: &[usize]) -> f64 {
    if idx.is_empty() {
        return 1.0;
    }
    if idx.len() % 2 == 1 {
        return 0.0;
    }
    let first = idx[0];
    let mut total = 0.0;
    for j in 1..idx.len() {
        let sv = s[(first, idx[j])];
        if sv == 0.0 {
            continue;
        }
        let rest: Vec<usize> = idx[1..].iter().enumerate().filter(|&(p, _)| p + 1 != j).map(|(_, &v)| v).collect();
        total += sv * isserlis(s, &rest);
    }
    total
}

/// Port of `disclyap_fast.m`: `X = G X G' + V` by doubling (tolerance `tol`, 2000 iterations).
pub fn disclyap_fast(g: &Mat, v: &Mat, tol: f64) -> Result<Mat> {
    let mut p0 = v.clone();
    let mut a0 = g.clone();
    let mut matd = 1.0;
    let mut iter = 1;
    let max_iter = 2000;
    while matd > tol && iter < max_iter {
        let p1 = &p0 + &a0 * &p0 * a0.transpose();
        let a1 = &a0 * &a0;
        matd = (&p1 - &p0).abs().max();
        p0 = p1;
        a0 = a1;
        iter += 1;
        if matd.is_nan() {
            break;
        }
    }
    if iter == max_iter || matd.is_nan() {
        bail!("disclyap_fast: no convergence");
    }
    Ok((&p0 + p0.transpose()) * 0.5)
}

fn rows(m: &Mat, idx: &[usize]) -> Mat {
    Mat::from_fn(idx.len(), m.ncols(), |i, j| m[(idx[i], j)])
}

/// Sets block `(r0.., c0..)` of `m` to `b`.
fn set(m: &mut Mat, r0: usize, c0: usize, b: &Mat) {
    m.view_mut((r0, c0), b.shape()).copy_from(b);
}

fn zero_block(m: &mut Mat, r0: usize, nr: usize, c0: usize, nc: usize) {
    m.view_mut((r0, c0), (nr, nc)).fill(0.0);
}

/// [`pruned_state_space_system_qz`] with `options_.qz_criterium` empty (Dynare's default
/// `1+1e-6`).
pub fn pruned_state_space_system(
    dr: &KOrderDr,
    sigma_e: &Mat,
    indy: &[usize],
    nlags: usize,
) -> Result<PrunedStateSpace> {
    pruned_state_space_system_qz(dr, sigma_e, indy, nlags, 1.0 + 1e-6)
}

/// Port of `pruned_state_space_system(M_, options_, dr, indy, nlags, useautocorr, 0)` at
/// `order = dr.order` (2 or 3) and `options_.qz_criterium = qz_criterium`. `indy`: selected
/// variables as DR positions (0-based).
pub fn pruned_state_space_system_qz(
    dr: &KOrderDr,
    sigma_e: &Mat,
    indy: &[usize],
    nlags: usize,
    qz_criterium: f64,
) -> Result<PrunedStateSpace> {
    let order = dr.order;
    let f = &dr.first;
    let x_nbr = f.nspred;
    let u_nbr = dr.n_exo;
    let indx: Vec<usize> = (f.nstatic..f.nstatic + x_nbr).collect();
    let y_nbr = indy.len();
    let yss: Vec<f64> = f.order_var.iter().map(|&j| f.ys[j]).collect();
    let hx = rows(&f.ghx, &indx);
    let gx = rows(&f.ghx, indy);
    let hu = rows(&f.ghu, &indx);
    let gu = rows(&f.ghu, indy);
    let e_uu = sigma_e.clone();
    let vec_euu = vec_of(&e_uu);
    // lyapunov_symm(A = hx, ...) of the first-order system: unit roots are the eigenvalues
    // with |e| > 2-qz_criterium (none for a single state, solved as b/(1-a^2)); a non-empty
    // Schur_u is an error at order > 1
    if x_nbr > 1 && crate::linalg::eig_general(&hx).iter().any(|z| z.norm() > 2.0 - qz_criterium) {
        bail!(
            "pruned_state_space_system: the pruned state space currently does not support unit roots at higher order."
        );
    }
    let lyap_tol = 1e-16; // options_.lyapunov_doubling_tol
    let e_xfxf = disclyap_fast(&hx, &(&hu * &e_uu * hu.transpose()), lyap_tol)?;

    let (hxx, gxx, hxu, gxu, huu, guu) = (
        rows(&dr.ghxx, &indx),
        rows(&dr.ghxx, indy),
        rows(&dr.ghxu, &indx),
        rows(&dr.ghxu, indy),
        rows(&dr.ghuu, &indx),
        rows(&dr.ghuu, indy),
    );
    let (hss, gss) = (rows(&dr.ghs2, &indx), rows(&dr.ghs2, indy));
    let x2 = x_nbr * x_nbr;
    let u2 = u_nbr * u_nbr;
    // index offsets
    let z1 = 0;
    let z2 = x_nbr;
    let z3 = 2 * x_nbr;
    let i1 = 0;
    let i2 = u_nbr;
    let i3 = u_nbr + u2;

    let hx_hx = kron(&hx, &hx);
    let hx_hu = kron(&hx, &hu);
    let hu_hu = kron(&hu, &hu);
    let i_xx = Mat::identity(x2, x2);
    let k_x_x = commutation(x_nbr, x_nbr);
    let inv_ix_hx = (Mat::identity(x_nbr, x_nbr) - &hx)
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("pruned_state_space_system: I - hx is singular"))?;
    let qpu_e4 = gaussian_moments(&e_uu, 4);
    let e_xfxf_uu = kron(&e_xfxf, &e_uu.transpose());

    let c_xs = &hss * 0.5 + &huu * &vec_euu * 0.5;
    let c_xfxf = &hu_hu * &vec_euu;
    let e_xs = &inv_ix_hx * (&hxx * vec_of(&e_xfxf) * 0.5 + &c_xs);
    let d_vec = &gss * 0.5 + &guu * &vec_euu * 0.5;
    let var_inov2 = reshape(&qpu_e4, u2, u2) - &vec_euu * vec_euu.transpose();

    let (a, b, c, cm, dm, var_inov, var_z, e_inovzlag1, e_z, hxi_data);
    if order == 2 {
        let z_nbr = 2 * x_nbr + x2;
        let inov_nbr = u_nbr + u2 + x_nbr * u_nbr;
        let mut am = Mat::zeros(z_nbr, z_nbr);
        set(&mut am, z1, z1, &hx);
        set(&mut am, z2, z2, &hx);
        set(&mut am, z2, z3, &(&hxx * 0.5));
        set(&mut am, z3, z3, &hx_hx);
        let mut bm = Mat::zeros(z_nbr, inov_nbr);
        set(&mut bm, z1, i1, &hu);
        set(&mut bm, z2, i2, &(&huu * 0.5));
        set(&mut bm, z2, i3, &hxu);
        set(&mut bm, z3, i2, &hu_hu);
        set(&mut bm, z3, i3, &((&i_xx + &k_x_x) * &hx_hu));
        let mut cv = Mat::zeros(z_nbr, 1);
        set(&mut cv, z2, 0, &c_xs);
        set(&mut cv, z3, 0, &c_xfxf);
        let mut cmm = Mat::zeros(y_nbr, z_nbr);
        set(&mut cmm, 0, z1, &gx);
        set(&mut cmm, 0, z2, &gx);
        set(&mut cmm, 0, z3, &(&gxx * 0.5));
        let mut dmm = Mat::zeros(y_nbr, inov_nbr);
        set(&mut dmm, 0, i1, &gu);
        set(&mut dmm, 0, i2, &(&guu * 0.5));
        set(&mut dmm, 0, i3, &gxu);
        let mut vi = Mat::zeros(inov_nbr, inov_nbr);
        set(&mut vi, i1, i1, &e_uu);
        set(&mut vi, i2, i2, &var_inov2);
        set(&mut vi, i3, i3, &e_xfxf_uu);
        let om = &bm * &vi * bm.transpose();
        let mut vz = disclyap_fast(&am, &om, lyap_tol)?;
        zero_block(&mut vz, z1, x_nbr, z2, x_nbr);
        zero_block(&mut vz, z1, x_nbr, z3, x2);
        zero_block(&mut vz, z2, x_nbr, z1, x_nbr);
        zero_block(&mut vz, z3, x2, z1, x_nbr);
        let mut ez = Mat::zeros(z_nbr, 1);
        set(&mut ez, z2, 0, &e_xs);
        set(&mut ez, z3, 0, &vec_of(&e_xfxf));
        a = am;
        b = bm;
        c = cv;
        cm = cmm;
        dm = dmm;
        var_inov = vi;
        var_z = vz;
        e_inovzlag1 = Mat::zeros(inov_nbr, z_nbr);
        e_z = ez;
        hxi_data = None;
    } else if order == 3 {
        let x3 = x2 * x_nbr;
        let u3 = u2 * u_nbr;
        let z4 = z3 + x2;
        let z5 = z4 + x_nbr;
        let z6 = z5 + x2;
        let i4 = i3 + x_nbr * u_nbr;
        let i5 = i4 + x_nbr * u_nbr;
        let i6 = i5 + x2 * u_nbr;
        let i7 = i6 + x_nbr * u2;
        let (hxxx, gxxx, hxxu, gxxu, hxuu, gxuu, huuu, guuu, hxss, gxss, huss, guss) = (
            rows(&dr.ghxxx, &indx),
            rows(&dr.ghxxx, indy),
            rows(&dr.ghxxu, &indx),
            rows(&dr.ghxxu, indy),
            rows(&dr.ghxuu, &indx),
            rows(&dr.ghxuu, indy),
            rows(&dr.ghuuu, &indx),
            rows(&dr.ghuuu, indy),
            rows(&dr.ghxss, &indx),
            rows(&dr.ghxss, indy),
            rows(&dr.ghuss, &indx),
            rows(&dr.ghuss, indy),
        );
        // the order-2 system first (its Var_z is needed below)
        let z_nbr2 = 2 * x_nbr + x2;
        let inov_nbr2 = u_nbr + u2 + x_nbr * u_nbr;
        let mut a2 = Mat::zeros(z_nbr2, z_nbr2);
        set(&mut a2, z1, z1, &hx);
        set(&mut a2, z2, z2, &hx);
        set(&mut a2, z2, z3, &(&hxx * 0.5));
        set(&mut a2, z3, z3, &hx_hx);
        let mut b2 = Mat::zeros(z_nbr2, inov_nbr2);
        set(&mut b2, z1, i1, &hu);
        set(&mut b2, z2, i2, &(&huu * 0.5));
        set(&mut b2, z2, i3, &hxu);
        set(&mut b2, z3, i2, &hu_hu);
        set(&mut b2, z3, i3, &((&i_xx + &k_x_x) * &hx_hu));
        let mut vi2 = Mat::zeros(inov_nbr2, inov_nbr2);
        set(&mut vi2, i1, i1, &e_uu);
        set(&mut vi2, i2, i2, &var_inov2);
        set(&mut vi2, i3, i3, &e_xfxf_uu);
        let mut vz2 = disclyap_fast(&a2, &(&b2 * &vi2 * b2.transpose()), lyap_tol)?;
        zero_block(&mut vz2, z1, x_nbr, z2, x_nbr);
        zero_block(&mut vz2, z1, x_nbr, z3, x2);
        zero_block(&mut vz2, z2, x_nbr, z1, x_nbr);
        zero_block(&mut vz2, z3, x2, z1, x_nbr);

        let k_u_xx = commutation(u_nbr, x2);
        let k_u_ux = commutation(u_nbr, u_nbr * x_nbr);
        let k_xx_x = commutation(x2, x_nbr);
        let hx_hss2 = kron(&hx, &(&hss * 0.5));
        let hu_hss2 = kron(&hu, &(&hss * 0.5));
        let hx_hxx2 = kron(&hx, &(&hxx * 0.5));
        let hxx2_hu = kron(&(&hxx * 0.5), &hu);
        let hx_hxu = kron(&hx, &hxu);
        let hxu_hu = kron(&hxu, &hu);
        let hx_huu2 = kron(&hx, &(&huu * 0.5));
        let hu_huu2 = kron(&hu, &(&huu * 0.5));
        let hx_hx_hx = kron(&hx, &hx_hx);
        let hx_hx_hu = kron(&hx_hx, &hu);
        let hu_hx_hx = kron(&hu, &hx_hx);
        let hu_hu_hu = kron(&hu_hu, &hu);
        let hx_hu_hu = kron(&hx, &hu_hu);
        let hu_hx_hu = kron(&hu, &hx_hu);
        let inv_ixx_hx_hx = (Mat::identity(x2, x2) - &hx_hx)
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("pruned_state_space_system: I - hx⊗hx is singular"))?;
        let vblk = |r0: usize, nr: usize, c0: usize, nc: usize| vz2.view((r0, c0), (nr, nc)).into_owned();
        let vec_exfxf = vec_of(&e_xfxf);
        let e_xsxs = vblk(z2, x_nbr, z2, x_nbr) + &e_xs * e_xs.transpose();
        let e_xsxf_xf = vblk(z2, x_nbr, z3, x2) + &e_xs * vec_exfxf.transpose();
        let e_xf_xfxs = vblk(z3, x2, z2, x_nbr) + &vec_exfxf * e_xs.transpose();
        let e_xf_xfxf_xf = vblk(z3, x2, z3, x2) + &vec_exfxf * vec_exfxf.transpose();
        let hxt = hx.transpose();
        let hut = hu.transpose();
        let inner = &hxx * reshape(&(&k_xx_x * vec_of(&e_xf_xfxs)), x2, x_nbr) * &hxt
            + &hxu * kron(&e_xs, &e_uu) * &hut
            + &hxxx * reshape(&e_xf_xfxf_xf, x3, x_nbr) * &hxt / 6.0
            + &huuu * reshape(&qpu_e4, u3, u_nbr) * &hut / 6.0
            + &hxxu * kron(&vec_exfxf, &e_uu) * &hut * 0.5
            + &hxuu * kron(&e_xfxf, &vec_euu) * &hxt * 0.5
            + &hxss * &e_xfxf * &hxt * 0.5
            + &huss * &e_uu * &hut * 0.5;
        let e_xrdxf = reshape(&(&inv_ixx_hx_hx * vec_of(&inner)), x_nbr, x_nbr);
        let q6 = gaussian_moments(&e_uu, 6);

        let z_nbr = x_nbr + x_nbr + x2 + x_nbr + x2 + x3;
        let inov_nbr = u_nbr + u2 + x_nbr * u_nbr + x_nbr * u_nbr + x2 * u_nbr + x_nbr * u2 + u3;
        let mut am = Mat::zeros(z_nbr, z_nbr);
        set(&mut am, z1, z1, &hx);
        set(&mut am, z2, z2, &hx);
        set(&mut am, z2, z3, &(&hxx * 0.5));
        set(&mut am, z3, z3, &hx_hx);
        set(&mut am, z4, z1, &(&hxss * 0.5));
        set(&mut am, z4, z4, &hx);
        set(&mut am, z4, z5, &hxx);
        set(&mut am, z4, z6, &(&hxxx / 6.0));
        set(&mut am, z5, z1, &hx_hss2);
        set(&mut am, z5, z5, &hx_hx);
        set(&mut am, z5, z6, &hx_hxx2);
        set(&mut am, z6, z6, &hx_hx_hx);
        let mut bm = Mat::zeros(z_nbr, inov_nbr);
        set(&mut bm, z1, i1, &hu);
        set(&mut bm, z2, i2, &(&huu * 0.5));
        set(&mut bm, z2, i3, &hxu);
        set(&mut bm, z3, i2, &hu_hu);
        set(&mut bm, z3, i3, &((&i_xx + &k_x_x) * &hx_hu));
        set(&mut bm, z4, i1, &(&huss * 0.5));
        set(&mut bm, z4, i4, &hxu);
        set(&mut bm, z4, i5, &(&hxxu * 0.5));
        set(&mut bm, z4, i6, &(&hxuu * 0.5));
        set(&mut bm, z4, i7, &(&huuu / 6.0));
        set(&mut bm, z5, i1, &hu_hss2);
        set(&mut bm, z5, i4, &(&k_x_x * &hx_hu));
        set(&mut bm, z5, i5, &(&hx_hxu + &k_x_x * &hxx2_hu));
        set(&mut bm, z5, i6, &(&hx_huu2 + &k_x_x * &hxu_hu));
        set(&mut bm, z5, i7, &hu_huu2);
        set(&mut bm, z6, i5, &(&hx_hx_hu + kron(&hx, &(&k_x_x * &hx_hu)) + &hu_hx_hx * &k_u_xx));
        set(&mut bm, z6, i6, &(&hx_hu_hu + &hu_hx_hu * &k_u_ux + kron(&hu, &(&k_x_x * &hx_hu)) * &k_u_ux));
        set(&mut bm, z6, i7, &hu_hu_hu);
        let mut cv = Mat::zeros(z_nbr, 1);
        set(&mut cv, z2, 0, &c_xs);
        set(&mut cv, z3, 0, &c_xfxf);
        let mut cmm = Mat::zeros(y_nbr, z_nbr);
        set(&mut cmm, 0, z1, &(&gx + &gxss * 0.5));
        set(&mut cmm, 0, z2, &gx);
        set(&mut cmm, 0, z3, &(&gxx * 0.5));
        set(&mut cmm, 0, z4, &gx);
        set(&mut cmm, 0, z5, &gxx);
        set(&mut cmm, 0, z6, &(&gxxx / 6.0));
        let mut dmm = Mat::zeros(y_nbr, inov_nbr);
        set(&mut dmm, 0, i1, &(&gu + &guss * 0.5));
        set(&mut dmm, 0, i2, &(&guu * 0.5));
        set(&mut dmm, 0, i3, &gxu);
        set(&mut dmm, 0, i4, &gxu);
        set(&mut dmm, 0, i5, &(&gxxu * 0.5));
        set(&mut dmm, 0, i6, &(&gxuu * 0.5));
        set(&mut dmm, 0, i7, &(&guuu / 6.0));
        let m4_1_3 = reshape(&qpu_e4, u_nbr, u3);
        let m4_3_1 = reshape(&qpu_e4, u3, u_nbr);
        let mut vi = Mat::zeros(inov_nbr, inov_nbr);
        set(&mut vi, i1, i1, &e_uu);
        set(&mut vi, i1, i4, &kron(&e_xs.transpose(), &e_uu));
        set(&mut vi, i1, i5, &kron(&vec_exfxf.transpose(), &e_uu));
        set(&mut vi, i1, i7, &m4_1_3);
        set(&mut vi, i2, i2, &var_inov2);
        set(&mut vi, i3, i3, &e_xfxf_uu);
        set(&mut vi, i4, i1, &kron(&e_xs, &e_uu));
        set(&mut vi, i4, i4, &kron(&e_xsxs, &e_uu));
        set(&mut vi, i4, i5, &kron(&e_xsxf_xf, &e_uu));
        set(&mut vi, i4, i7, &kron(&e_xs, &m4_1_3));
        set(&mut vi, i5, i1, &kron(&vec_exfxf, &e_uu));
        set(&mut vi, i5, i4, &kron(&e_xf_xfxs, &e_uu));
        set(&mut vi, i5, i5, &kron(&e_xf_xfxf_xf, &e_uu));
        set(&mut vi, i5, i7, &kron(&vec_exfxf, &m4_1_3));
        set(&mut vi, i6, i6, &kron(&e_xfxf, &reshape(&qpu_e4, u2, u2)));
        set(&mut vi, i7, i1, &m4_3_1);
        set(&mut vi, i7, i4, &kron(&e_xs.transpose(), &m4_3_1));
        set(&mut vi, i7, i5, &kron(&vec_exfxf.transpose(), &m4_3_1));
        set(&mut vi, i7, i7, &reshape(&q6, u3, u3));
        let k_xx_x_e_xsxf_xf = reshape(&(&k_xx_x * vec_of(&e_xsxf_xf)), x_nbr, x2);
        let (lb_exfxf, lb_vec_euu, lb_xrd, lb_e3) =
            (e_xfxf.clone(), vec_euu.clone(), e_xrdxf.transpose(), reshape(&e_xf_xfxf_xf, x_nbr, x3));
        let lb_k = k_xx_x_e_xsxf_xf.clone();
        let lag_blocks = move |hxi: &Mat| -> Mat {
            let mut e = Mat::zeros(inov_nbr, z_nbr);
            set(&mut e, i6, z1, &kron(&(hxi * &lb_exfxf), &lb_vec_euu));
            set(&mut e, i6, z4, &kron(&(hxi * &lb_xrd), &lb_vec_euu));
            set(&mut e, i6, z5, &kron(&(hxi * &lb_k), &lb_vec_euu));
            set(&mut e, i6, z6, &kron(&(hxi * &lb_e3), &lb_vec_euu));
            e
        };
        let eiz1 = lag_blocks(&Mat::identity(x_nbr, x_nbr));
        let binov_a = &bm * &eiz1 * am.transpose();
        let om = &bm * &vi * bm.transpose() + &binov_a + binov_a.transpose();
        let mut vz = disclyap_fast(&am, &om, lyap_tol)?;
        for &(r0, nr, c0, nc) in &[
            (z1, x_nbr, z2, x_nbr),
            (z1, x_nbr, z3, x2),
            (z2, x_nbr, z1, x_nbr),
            (z2, x_nbr, z4, x_nbr),
            (z2, x_nbr, z5, x2),
            (z2, x_nbr, z6, x3),
            (z3, x2, z1, x_nbr),
            (z3, x2, z4, x_nbr),
            (z3, x2, z5, x2),
            (z3, x2, z6, x3),
            (z4, x_nbr, z2, x_nbr),
            (z4, x_nbr, z3, x2),
            (z5, x2, z2, x_nbr),
            (z5, x2, z3, x2),
            (z6, x3, z2, x_nbr),
            (z6, x3, z3, x2),
        ] {
            zero_block(&mut vz, r0, nr, c0, nc);
        }
        let mut ez = Mat::zeros(z_nbr, 1);
        set(&mut ez, z2, 0, &e_xs);
        set(&mut ez, z3, 0, &vec_exfxf);
        a = am;
        b = bm;
        c = cv;
        cm = cmm;
        dm = dmm;
        var_inov = vi;
        var_z = vz;
        e_inovzlag1 = eiz1;
        e_z = ez;
        hxi_data = Some(Box::new(lag_blocks) as Box<dyn Fn(&Mat) -> Mat>);
    } else {
        bail!("pruned_state_space_system: order must be 2 or 3");
    }

    let mut var_y = if order < 3 {
        &cm * &var_z * cm.transpose() + &dm * &var_inov * dm.transpose()
    } else {
        &cm * &var_z * cm.transpose()
            + &dm * &e_inovzlag1 * cm.transpose()
            + &cm * e_inovzlag1.transpose() * dm.transpose()
            + &dm * &var_inov * dm.transpose()
    };
    var_y.apply(|v| {
        if v.abs() < 1e-12 {
            *v = 0.0
        }
    });
    let sdy: Vec<f64> = (0..y_nbr).map(|i| var_y[(i, i)].sqrt()).collect();
    let sy = Mat::from_fn(y_nbr, y_nbr, |i, j| sdy[i] * sdy[j]);
    let corr_y = var_y.component_div(&sy);
    let z_nbr = a.nrows();
    let mut ai = Mat::identity(z_nbr, z_nbr);
    let mut hxi = Mat::identity(x_nbr, x_nbr);
    let mut var_zi = var_z.clone();
    let mut e_inovzlagi = e_inovzlag1.clone();
    let tmp = if order <= 2 {
        &a * &var_z * cm.transpose() + &b * &var_inov * dm.transpose()
    } else {
        &a * e_inovzlag1.transpose() * dm.transpose() + &b * &var_inov * dm.transpose()
    };
    let mut var_yi = Vec::with_capacity(nlags);
    let mut corr_yi = Vec::with_capacity(nlags);
    for _ in 0..nlags {
        let v = if order <= 2 {
            &cm * &ai * &tmp
        } else {
            var_zi = &a * &var_zi + &b * &e_inovzlagi;
            hxi = &hx * &hxi;
            e_inovzlagi = (hxi_data.as_ref().unwrap())(&hxi);
            &cm * &var_zi * cm.transpose() + &cm * &ai * &tmp + &dm * &e_inovzlagi * cm.transpose()
        };
        corr_yi.push(v.component_div(&sy));
        var_yi.push(v);
        ai = &ai * &a;
    }
    let ce = &cm * &e_z;
    let e_y: Vec<f64> = (0..y_nbr).map(|i| yss[indy[i]] + ce[(i, 0)] + d_vec[(i, 0)]).collect();
    Ok(PrunedStateSpace {
        a,
        b,
        c: c.as_slice().to_vec(),
        cmat: cm,
        dmat: dm,
        d: d_vec.as_slice().to_vec(),
        var_inov,
        e_y,
        var_y,
        var_yi,
        corr_y,
        corr_yi,
    })
}

/// Second-order mean of the variables (`th_autocovariances.m`, `order = 2`, no filter):
/// `ys + ghx Ex + (ghs2 + ghxx vec(Var_x) + ghuu vec(Sigma_e)) / 2` with
/// `Ex = (I - ghx_x)^{-1} (ghs2_x + ghxx_x vec(Var_x) + ghuu_x vec(Sigma_e)) / 2`; returned in
/// declaration order (all variables).
pub fn second_order_mean(dr: &KOrderDr, sigma_e: &Mat) -> Result<Vec<f64>> {
    let f = &dr.first;
    let ns = f.nspred;
    let n = f.ghx.nrows();
    let indx: Vec<usize> = (f.nstatic..f.nstatic + ns).collect();
    let hx = rows(&f.ghx, &indx);
    let hu = rows(&f.ghu, &indx);
    let var_x = disclyap_fast(&hx, &(&hu * sigma_e * hu.transpose()), 1e-16)?;
    let vx = vec_of(&var_x);
    let vs = vec_of(sigma_e);
    let corr = &dr.ghs2 + &dr.ghxx * &vx + &dr.ghuu * &vs; // n x 1, DR order
    let ex_rhs = rows(&corr, &indx) * 0.5;
    let ex = (Mat::identity(ns, ns) - &hx).lu().solve(&ex_rhs).ok_or_else(|| anyhow::anyhow!("I - ghx singular"))?;
    let m = &f.ghx * &ex + &corr * 0.5;
    let mut out = vec![0.0; n];
    for (k, &j) in f.order_var.iter().enumerate() {
        out[j] = f.ys[j] + m[(k, 0)];
    }
    Ok(out)
}
