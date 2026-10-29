// SPDX-License-Identifier: GPL-3.0-or-later
//! Dense linear algebra: nalgebra matrices (column-major) plus safe wrappers
//! over Accelerate's BLAS/LAPACK for the hot paths.
//!
//! Ownership: `mod.rs`/`lapack.rs` are shared; `qz.rs` belongs to the solver
//! work, `schur.rs` to the Kalman work. Add wrappers rather than calling the
//! FFI directly from other modules.

pub mod lapack;
pub mod qz;
pub mod schur;

use std::os::raw::{c_char, c_int};

pub type Mat = nalgebra::DMatrix<f64>;
pub type Vector = nalgebra::DVector<f64>;

#[inline]
fn ch(b: bool) -> c_char {
    (if b { b'T' } else { b'N' }) as c_char
}

// ---- BLAS backend for gemm/gemv ---------------------------------------------
//
// The level-2/3 products of the wrappers below (`gemm`, `mul`, `gemm_raw`,
// `gemv_raw`) can run either through Accelerate (default) or through the
// pure-Rust `matrixmultiply` kernel (gemm) and plain loops (gemv). LAPACK
// routines always use Accelerate. Choose with the environment variable
// `DYNARE_BLAS=accelerate|rust` (read once) or [`set_blas_backend`].

use std::sync::atomic::{AtomicU8, Ordering};

/// Which implementation the gemm/gemv wrappers use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlasBackend {
    /// Apple Accelerate `dgemm_`/`dgemv_` (AMX/SME-backed on Apple silicon).
    Accelerate,
    /// Pure Rust: `matrixmultiply::dgemm` (NEON) and straightforward gemv loops,
    /// running entirely on the calling core.
    Rust,
}

static BACKEND: AtomicU8 = AtomicU8::new(0); // 0 = not chosen yet, 1 = Accelerate, 2 = Rust

/// The current gemm/gemv backend (first call reads `DYNARE_BLAS`).
#[inline]
pub fn blas_backend() -> BlasBackend {
    match BACKEND.load(Ordering::Relaxed) {
        1 => BlasBackend::Accelerate,
        2 => BlasBackend::Rust,
        _ => {
            let b = match std::env::var("DYNARE_BLAS").map(|s| s.to_ascii_lowercase()) {
                Ok(s) if s == "rust" => BlasBackend::Rust,
                _ => BlasBackend::Accelerate,
            };
            set_blas_backend(b);
            b
        }
    }
}

/// Selects the gemm/gemv backend for the whole process.
pub fn set_blas_backend(b: BlasBackend) {
    BACKEND.store(if b == BlasBackend::Accelerate { 1 } else { 2 }, Ordering::Relaxed);
}

/// Calls OpenBLAS's `openblas_set_num_threads(n)` when the process is linked against OpenBLAS
/// (looked up at run time, so other BLAS libraries are unaffected); returns whether it was
/// found. OpenBLAS commits buffers for each of its threads (about 2 GB with 28), and on the
/// small matrices of these models its threads only add overhead: dynare-rs parallelizes over
/// chains and evaluations itself.
pub fn set_openblas_threads(n: usize) -> bool {
    let Some(f) = openblas_set_num_threads_fn() else { return false };
    // SAFETY: the symbol is OpenBLAS's `void openblas_set_num_threads(int)`.
    unsafe { f(n.min(i32::MAX as usize) as std::ffi::c_int) };
    true
}

type SetThreadsFn = unsafe extern "C" fn(std::ffi::c_int);

#[cfg(windows)]
fn openblas_set_num_threads_fn() -> Option<SetThreadsFn> {
    unsafe extern "system" {
        fn GetModuleHandleA(name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
        fn LoadLibraryA(name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
        fn GetProcAddress(module: *mut std::ffi::c_void, name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    }
    for dll in [c"libopenblas.dll", c"openblas.dll"] {
        // SAFETY: NUL-terminated names. libopenblas.dll is delay-loaded (build.rs), so it may
        // not be loaded yet: load it then (it would be at the first BLAS call anyway).
        let p = unsafe {
            let mut h = GetModuleHandleA(dll.as_ptr());
            if h.is_null() && dll == c"libopenblas.dll" {
                h = LoadLibraryA(dll.as_ptr());
            }
            if h.is_null() {
                continue;
            }
            GetProcAddress(h, c"openblas_set_num_threads".as_ptr())
        };
        if !p.is_null() {
            // SAFETY: a function pointer of the documented signature.
            return Some(unsafe { std::mem::transmute::<*mut std::ffi::c_void, SetThreadsFn>(p) });
        }
    }
    None
}

#[cfg(all(unix, not(target_os = "macos")))]
fn openblas_set_num_threads_fn() -> Option<SetThreadsFn> {
    unsafe extern "C" {
        fn dlsym(handle: *mut std::ffi::c_void, symbol: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    }
    // SAFETY: RTLD_DEFAULT (a null handle on glibc) searches the loaded libraries.
    let p = unsafe { dlsym(std::ptr::null_mut(), c"openblas_set_num_threads".as_ptr()) };
    // SAFETY: a function pointer of the documented signature.
    (!p.is_null()).then(|| unsafe { std::mem::transmute::<*mut std::ffi::c_void, SetThreadsFn>(p) })
}

#[cfg(any(target_os = "macos", not(any(windows, unix))))]
fn openblas_set_num_threads_fn() -> Option<SetThreadsFn> {
    None // Accelerate
}

/// C(m x n) <- alpha op(A) op(B) + beta C with matrixmultiply (column-major,
/// leading dimensions as BLAS). `beta == 0` does not read C, as in BLAS.
#[allow(clippy::too_many_arguments)]
fn gemm_rust(
    ta: bool,
    tb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    // op(A)(i, p) = A[i + p*lda] (no transpose) or A[p + i*lda] (transpose)
    let (rsa, csa) = if ta { (lda as isize, 1) } else { (1, lda as isize) };
    let (rsb, csb) = if tb { (ldb as isize, 1) } else { (1, ldb as isize) };
    unsafe { matrixmultiply::dgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, 1, ldc as isize) };
}

/// y <- alpha op(A) x + beta y in plain Rust (unit strides).
#[allow(clippy::too_many_arguments)]
fn gemv_rust(ta: bool, m: usize, n: usize, alpha: f64, a: &[f64], lda: usize, x: &[f64], beta: f64, y: &mut [f64]) {
    if ta {
        // y(j) = alpha * A(:, j)' x + beta y(j)
        for j in 0..n {
            let col = &a[j * lda..j * lda + m];
            // four partial sums (vectorisable)
            let mut s = [0.0f64; 4];
            for (i, (c, v)) in col.iter().zip(&x[..m]).enumerate() {
                s[i & 3] += c * v;
            }
            let t = (s[0] + s[1]) + (s[2] + s[3]);
            y[j] = if beta == 0.0 { alpha * t } else { alpha * t + beta * y[j] };
        }
    } else {
        let y = &mut y[..m];
        if beta == 0.0 {
            y.iter_mut().for_each(|v| *v = 0.0);
        } else if beta != 1.0 {
            y.iter_mut().for_each(|v| *v *= beta);
        }
        for j in 0..n {
            let s = alpha * x[j];
            if s == 0.0 {
                continue;
            }
            for (v, c) in y.iter_mut().zip(&a[j * lda..j * lda + m]) {
                *v += s * c;
            }
        }
    }
}

/// C <- alpha * op(A) * op(B) + beta * C, via Accelerate dgemm (or the Rust backend).
pub fn gemm(alpha: f64, a: &Mat, ta: bool, b: &Mat, tb: bool, beta: f64, c: &mut Mat) {
    let (m, k) = if ta { (a.ncols(), a.nrows()) } else { (a.nrows(), a.ncols()) };
    let (kb, n) = if tb { (b.ncols(), b.nrows()) } else { (b.nrows(), b.ncols()) };
    assert_eq!(k, kb, "gemm: inner dimensions");
    assert_eq!((c.nrows(), c.ncols()), (m, n), "gemm: output shape");
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        *c *= beta;
        return;
    }
    if blas_backend() == BlasBackend::Rust {
        let (lda, ldb, ldc) = (a.nrows().max(1), b.nrows().max(1), c.nrows().max(1));
        gemm_rust(ta, tb, m, n, k, alpha, a.as_ptr(), lda, b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc);
        return;
    }
    let (m, n, k) = (m as c_int, n as c_int, k as c_int);
    let lda = a.nrows().max(1) as c_int;
    let ldb = b.nrows().max(1) as c_int;
    let ldc = c.nrows().max(1) as c_int;
    unsafe {
        lapack::dgemm_(
            &ch(ta),
            &ch(tb),
            &m,
            &n,
            &k,
            &alpha,
            a.as_ptr(),
            &lda,
            b.as_ptr(),
            &ldb,
            &beta,
            c.as_mut_ptr(),
            &ldc,
        );
    }
}

/// op(A) * op(B) as a new matrix.
pub fn mul(a: &Mat, ta: bool, b: &Mat, tb: bool) -> Mat {
    let m = if ta { a.ncols() } else { a.nrows() };
    let n = if tb { b.nrows() } else { b.ncols() };
    let mut c = Mat::zeros(m, n);
    gemm(1.0, a, ta, b, tb, 0.0, &mut c);
    c
}

/// LU factorisation (dgetrf). Returns the factors, the pivots and LAPACK's info
/// (> 0: exactly singular U).
pub fn lu(a: &Mat) -> (Mat, Vec<c_int>, i32) {
    assert!(a.is_square());
    let n = a.nrows() as c_int;
    let mut f = a.clone();
    let mut ipiv = vec![0 as c_int; a.nrows()];
    let mut info: c_int = 0;
    unsafe { lapack::dgetrf_(&n, &n, f.as_mut_ptr(), &n.max(1), ipiv.as_mut_ptr(), &mut info) };
    (f, ipiv, info)
}

/// Solve A X = B (or A' X = B) from an LU factorisation, in place on B.
pub fn lu_solve(f: &Mat, ipiv: &[c_int], trans: bool, b: &mut Mat) {
    let n = f.nrows() as c_int;
    let nrhs = b.ncols() as c_int;
    let mut info: c_int = 0;
    unsafe {
        lapack::dgetrs_(
            &ch(trans),
            &n,
            &nrhs,
            f.as_ptr(),
            &n.max(1),
            ipiv.as_ptr(),
            b.as_mut_ptr(),
            &(b.nrows().max(1) as c_int),
            &mut info,
        )
    };
}

/// MATLAB `A\B` for square A via LU. Returns None if A is exactly singular.
pub fn solve(a: &Mat, b: &Mat) -> Option<Mat> {
    let (f, ipiv, info) = lu(a);
    if info > 0 {
        return None;
    }
    let mut x = b.clone();
    lu_solve(&f, &ipiv, false, &mut x);
    Some(x)
}

/// Reciprocal 1-norm condition number estimate from an LU factorisation (MATLAB `rcond`).
pub fn rcond_from_lu(f: &Mat, anorm1: f64) -> f64 {
    let n = f.nrows() as c_int;
    let mut rc = 0.0;
    let mut work = vec![0.0; 4 * f.nrows().max(1)];
    let mut iwork = vec![0 as c_int; f.nrows().max(1)];
    let mut info: c_int = 0;
    unsafe {
        lapack::dgecon_(
            &(b'1' as c_char),
            &n,
            f.as_ptr(),
            &n.max(1),
            &anorm1,
            &mut rc,
            work.as_mut_ptr(),
            iwork.as_mut_ptr(),
            &mut info,
        )
    };
    rc
}

/// 1-norm (max absolute column sum).
pub fn norm1(a: &Mat) -> f64 {
    a.column_iter().map(|c| c.iter().map(|x| x.abs()).sum::<f64>()).fold(0.0, f64::max)
}

/// Cholesky factor (upper, MATLAB `chol`), None if not positive definite.
pub fn chol_upper(a: &Mat) -> Option<Mat> {
    let n = a.nrows() as c_int;
    let mut f = a.clone();
    let mut info: c_int = 0;
    unsafe { lapack::dpotrf_(&(b'U' as c_char), &n, f.as_mut_ptr(), &n.max(1), &mut info) };
    if info != 0 {
        return None;
    }
    for j in 0..f.ncols() {
        for i in (j + 1)..f.nrows() {
            f[(i, j)] = 0.0;
        }
    }
    Some(f)
}

/// MATLAB `A'*A`: MATLAB recognises the pattern and calls dsyrk (one triangle, mirrored),
/// so the result is exactly symmetric and rounds as MATLAB's.
pub fn syrk_t(a: &Mat) -> Mat {
    let (k, n) = (a.nrows(), a.ncols());
    let mut c = Mat::zeros(n, n);
    if n == 0 || k == 0 {
        return c;
    }
    unsafe {
        lapack::dsyrk_(
            &(b'U' as c_char),
            &(b'T' as c_char),
            &(n as c_int),
            &(k as c_int),
            &1.0,
            a.as_ptr(),
            &(k as c_int),
            &0.0,
            c.as_mut_ptr(),
            &(n as c_int),
        )
    };
    for j in 0..n {
        for i in (j + 1)..n {
            c[(i, j)] = c[(j, i)];
        }
    }
    c
}

/// MATLAB `A\B` for a square, full `A`: Cholesky (dpotrf/dpotrs) when `A` is exactly
/// symmetric with a positive diagonal and the factorisation succeeds, LU otherwise.
/// None if `A` is exactly singular.
pub fn mldivide_sq(a: &Mat, b: &Mat) -> Option<Mat> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "mldivide_sq: square matrices only");
    let sym = (0..n).all(|j| a[(j, j)] > 0.0 && (j + 1..n).all(|i| a[(i, j)] == a[(j, i)]));
    if sym && n > 0 {
        let mut f = a.clone();
        let mut info: c_int = 0;
        let ni = n as c_int;
        unsafe { lapack::dpotrf_(&(b'U' as c_char), &ni, f.as_mut_ptr(), &ni, &mut info) };
        if info == 0 {
            let mut x = b.clone();
            unsafe {
                lapack::dpotrs_(
                    &(b'U' as c_char),
                    &ni,
                    &(b.ncols() as c_int),
                    f.as_ptr(),
                    &ni,
                    x.as_mut_ptr(),
                    &(b.nrows().max(1) as c_int),
                    &mut info,
                )
            };
            return Some(x);
        }
    }
    solve(a, b)
}

/// MATLAB `inv(A)` (dgetrf + dgetri). None if `A` is exactly singular.
pub fn inv_lu(a: &Mat) -> Option<Mat> {
    let n = a.nrows();
    if n == 0 {
        return Some(a.clone());
    }
    let (mut f, ipiv, info) = lu(a);
    if info > 0 {
        return None;
    }
    let ni = n as c_int;
    let lwork = (64 * n) as c_int;
    let mut work = vec![0.0; lwork as usize];
    let mut info: c_int = 0;
    unsafe { lapack::dgetri_(&ni, f.as_mut_ptr(), &ni, ipiv.as_ptr(), work.as_mut_ptr(), &lwork, &mut info) };
    Some(f)
}

/// Symmetric eigen-decomposition (ascending eigenvalues, eigenvectors in columns).
pub fn sym_eig(a: &Mat) -> (Vector, Mat) {
    let n = a.nrows() as c_int;
    let mut v = a.clone();
    let mut w = Vector::zeros(a.nrows());
    let mut info: c_int = 0;
    let lwork: c_int = (3 * a.nrows()).max(1) as c_int * 8;
    let mut work = vec![0.0; lwork as usize];
    unsafe {
        lapack::dsyev_(
            &(b'V' as c_char),
            &(b'U' as c_char),
            &n,
            v.as_mut_ptr(),
            &n.max(1),
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        )
    };
    (w, v)
}

/// Singular values (descending).
pub fn singular_values(a: &Mat) -> Vector {
    let (m, n) = (a.nrows() as c_int, a.ncols() as c_int);
    let mut f = a.clone();
    let k = a.nrows().min(a.ncols());
    let mut s = Vector::zeros(k);
    let lwork: c_int = (5 * (a.nrows() + a.ncols())).max(1) as c_int * 4;
    let mut work = vec![0.0; lwork as usize];
    let mut info: c_int = 0;
    let one: c_int = 1;
    unsafe {
        lapack::dgesvd_(
            &(b'N' as c_char),
            &(b'N' as c_char),
            &m,
            &n,
            f.as_mut_ptr(),
            &m.max(1),
            s.as_mut_ptr(),
            std::ptr::null_mut(),
            &one,
            std::ptr::null_mut(),
            &one,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        )
    };
    s
}

/// Householder QR of `a` (m x k, dgeqrf) applied to `c` (m x n): `c <- Q' c`
/// (MATLAB `[Q, ~] = qr(a); Q' * c` with the full m x m `Q`).
pub fn qr_qt_mul(a: &Mat, c: &mut Mat) {
    let (m, k) = (a.nrows(), a.ncols());
    assert_eq!(c.nrows(), m, "qr_qt_mul: row mismatch");
    let kk = m.min(k);
    if m == 0 || kk == 0 || c.ncols() == 0 {
        return;
    }
    let mut f = a.clone();
    let mut tau = vec![0.0; kk];
    let (mi, ki, ni) = (m as c_int, k as c_int, c.ncols() as c_int);
    let mut info: c_int = 0;
    let lwork = (64 * (k.max(c.ncols()) + 1)) as c_int;
    let mut work = vec![0.0; lwork as usize];
    unsafe {
        lapack::dgeqrf_(&mi, &ki, f.as_mut_ptr(), &mi, tau.as_mut_ptr(), work.as_mut_ptr(), &lwork, &mut info);
        lapack::dormqr_(
            &(b'L' as c_char),
            &(b'T' as c_char),
            &mi,
            &ni,
            &(kk as c_int),
            f.as_ptr(),
            &mi,
            tau.as_ptr(),
            c.as_mut_ptr(),
            &mi,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
}

// ---- Slice-level wrappers (no allocation) for hot loops ----------------------
//
// Column-major buffers with explicit leading dimensions, so preallocated
// workspaces can hold sub-matrices of varying size. Added for the Kalman filters.

/// C(m x n) <- alpha * op(A) * op(B) + beta * C on raw column-major slices (dgemm).
#[allow(clippy::too_many_arguments)]
pub fn gemm_raw(
    ta: bool,
    tb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(c.len() >= ldc * (n - 1) + m, "gemm_raw: C too short");
    if k == 0 {
        for j in 0..n {
            for x in &mut c[j * ldc..j * ldc + m] {
                *x *= beta;
            }
        }
        return;
    }
    let (ar, ac) = if ta { (k, m) } else { (m, k) };
    let (br, bc) = if tb { (n, k) } else { (k, n) };
    assert!(lda >= ar && a.len() >= lda * (ac - 1) + ar, "gemm_raw: A too short");
    assert!(ldb >= br && b.len() >= ldb * (bc - 1) + br, "gemm_raw: B too short");
    if blas_backend() == BlasBackend::Rust {
        gemm_rust(ta, tb, m, n, k, alpha, a.as_ptr(), lda, b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc);
        return;
    }
    let (mi, ni, ki) = (m as c_int, n as c_int, k as c_int);
    let (lda, ldb, ldc) = (lda as c_int, ldb as c_int, ldc as c_int);
    unsafe {
        lapack::dgemm_(
            &ch(ta),
            &ch(tb),
            &mi,
            &ni,
            &ki,
            &alpha,
            a.as_ptr(),
            &lda,
            b.as_ptr(),
            &ldb,
            &beta,
            c.as_mut_ptr(),
            &ldc,
        );
    }
}

/// Triangular solve (dtrsm, left side): `b <- alpha * op(A)^{-1} b` with `A`
/// upper (`upper = true`) or lower triangular, non-unit diagonal.
pub fn trsm_left(upper: bool, trans: bool, alpha: f64, a: &Mat, b: &mut Mat) {
    assert!(a.is_square() && a.nrows() == b.nrows(), "trsm_left: shapes");
    if b.nrows() == 0 || b.ncols() == 0 {
        return;
    }
    let (m, n) = (b.nrows() as c_int, b.ncols() as c_int);
    unsafe {
        lapack::dtrsm_(
            &(b'L' as c_char),
            &((if upper { b'U' } else { b'L' }) as c_char),
            &ch(trans),
            &(b'N' as c_char),
            &m,
            &n,
            &alpha,
            a.as_ptr(),
            &m,
            b.as_mut_ptr(),
            &m,
        );
    }
}

/// Eigenvalues of a general real square matrix (dgeev, no vectors).
pub fn eig_general(a: &Mat) -> Vec<nalgebra::Complex<f64>> {
    assert!(a.is_square());
    let n = a.nrows();
    if n == 0 {
        return Vec::new();
    }
    let ni = n as c_int;
    let mut f = a.clone();
    let (mut wr, mut wi) = (vec![0.0; n], vec![0.0; n]);
    let lwork = (8 * n + 64) as c_int;
    let mut work = vec![0.0; lwork as usize];
    let mut info: c_int = 0;
    let one: c_int = 1;
    let mut dummy = [0.0f64; 1];
    let mut dummy2 = [0.0f64; 1];
    unsafe {
        lapack::dgeev_(
            &(b'N' as c_char),
            &(b'N' as c_char),
            &ni,
            f.as_mut_ptr(),
            &ni,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            dummy.as_mut_ptr(),
            &one,
            dummy2.as_mut_ptr(),
            &one,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    wr.into_iter().zip(wi).map(|(r, i)| nalgebra::Complex::new(r, i)).collect()
}

// ---- stoch_simul ----
/// MATLAB `[V, D] = eig(A)` for a general real matrix (dgeev with right eigenvectors):
/// the eigenvalues and LAPACK's packed real eigenvector matrix `vr` (for a complex pair
/// `j, j+1`, the vectors are `vr(:,j) ± i vr(:,j+1)`; unit 2-norm, as MATLAB returns them).
pub fn eig_general_vectors(a: &Mat) -> (Vec<nalgebra::Complex<f64>>, Mat) {
    assert!(a.is_square());
    let n = a.nrows();
    if n == 0 {
        return (Vec::new(), Mat::zeros(0, 0));
    }
    let ni = n as c_int;
    let mut f = a.clone();
    let (mut wr, mut wi) = (vec![0.0; n], vec![0.0; n]);
    let mut vr = Mat::zeros(n, n);
    let lwork = (8 * n + 64) as c_int;
    let mut work = vec![0.0; lwork as usize];
    let mut info: c_int = 0;
    let one: c_int = 1;
    let mut dummy = [0.0f64; 1];
    unsafe {
        lapack::dgeev_(
            &(b'N' as c_char),
            &(b'V' as c_char),
            &ni,
            f.as_mut_ptr(),
            &ni,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            dummy.as_mut_ptr(),
            &one,
            vr.as_mut_ptr(),
            &ni,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    (wr.into_iter().zip(wi).map(|(r, i)| nalgebra::Complex::new(r, i)).collect(), vr)
}
// ---- end stoch_simul ----

/// y <- alpha * op(A) * x + beta * y with A (m x n, leading dimension lda), via dgemv.
#[allow(clippy::too_many_arguments)]
pub fn gemv_raw(ta: bool, m: usize, n: usize, alpha: f64, a: &[f64], lda: usize, x: &[f64], beta: f64, y: &mut [f64]) {
    let (xl, yl) = if ta { (m, n) } else { (n, m) };
    assert!(x.len() >= xl && y.len() >= yl, "gemv_raw: vector too short");
    if m == 0 || n == 0 {
        for v in &mut y[..yl] {
            *v *= beta;
        }
        return;
    }
    assert!(lda >= m && a.len() >= lda * (n - 1) + m, "gemv_raw: A too short");
    if blas_backend() == BlasBackend::Rust {
        gemv_rust(ta, m, n, alpha, a, lda, x, beta, y);
        return;
    }
    let (mi, ni, lda) = (m as c_int, n as c_int, lda as c_int);
    let one: c_int = 1;
    unsafe {
        lapack::dgemv_(&ch(ta), &mi, &ni, &alpha, a.as_ptr(), &lda, x.as_ptr(), &one, &beta, y.as_mut_ptr(), &one)
    };
}

/// In-place LU (dgetrf) of the n x n matrix in `a` (leading dimension `lda`). Returns LAPACK's info.
pub fn getrf_raw(n: usize, a: &mut [f64], lda: usize, ipiv: &mut [c_int]) -> i32 {
    if n == 0 {
        return 0;
    }
    assert!(lda >= n && a.len() >= lda * (n - 1) + n && ipiv.len() >= n, "getrf_raw: buffers too short");
    let ni = n as c_int;
    let mut info: c_int = 0;
    unsafe { lapack::dgetrf_(&ni, &ni, a.as_mut_ptr(), &(lda as c_int), ipiv.as_mut_ptr(), &mut info) };
    info
}

/// Solve op(A) X = B from `getrf_raw` factors, in place on B (n x nrhs, leading dimension ldb).
#[allow(clippy::too_many_arguments)]
pub fn getrs_raw(trans: bool, n: usize, nrhs: usize, a: &[f64], lda: usize, ipiv: &[c_int], b: &mut [f64], ldb: usize) {
    if n == 0 || nrhs == 0 {
        return;
    }
    assert!(lda >= n && a.len() >= lda * (n - 1) + n && ipiv.len() >= n, "getrs_raw: factors too short");
    assert!(ldb >= n && b.len() >= ldb * (nrhs - 1) + n, "getrs_raw: B too short");
    let mut info: c_int = 0;
    unsafe {
        lapack::dgetrs_(
            &ch(trans),
            &(n as c_int),
            &(nrhs as c_int),
            a.as_ptr(),
            &(lda as c_int),
            ipiv.as_ptr(),
            b.as_mut_ptr(),
            &(ldb as c_int),
            &mut info,
        )
    };
}

/// MATLAB `rcond` from `getrf_raw` factors and the 1-norm of the original matrix (dgecon).
/// `work` needs 4n entries, `iwork` n.
pub fn gecon_raw(n: usize, a: &[f64], lda: usize, anorm1: f64, work: &mut [f64], iwork: &mut [c_int]) -> f64 {
    if n == 0 {
        return f64::INFINITY;
    }
    assert!(lda >= n && a.len() >= lda * (n - 1) + n, "gecon_raw: factors too short");
    assert!(work.len() >= 4 * n && iwork.len() >= n, "gecon_raw: work too short");
    let mut rc = 0.0;
    let mut info: c_int = 0;
    unsafe {
        lapack::dgecon_(
            &(b'1' as c_char),
            &(n as c_int),
            a.as_ptr(),
            &(lda as c_int),
            &anorm1,
            &mut rc,
            work.as_mut_ptr(),
            iwork.as_mut_ptr(),
            &mut info,
        )
    };
    rc
}

/// Singular values (descending) of the m x n matrix in a raw buffer, which is overwritten.
/// `s` needs min(m, n) entries; `work` should hold at least 5*(m+n) entries.
pub fn singular_values_raw(m: usize, n: usize, a: &mut [f64], lda: usize, s: &mut [f64], work: &mut [f64]) {
    if m == 0 || n == 0 {
        return;
    }
    let (mi, ni) = (m as c_int, n as c_int);
    let lwork = work.len() as c_int;
    let mut info: c_int = 0;
    let one: c_int = 1;
    unsafe {
        lapack::dgesvd_(
            &(b'N' as c_char),
            &(b'N' as c_char),
            &mi,
            &ni,
            a.as_mut_ptr(),
            &(lda.max(1) as c_int),
            s.as_mut_ptr(),
            std::ptr::null_mut(),
            &one,
            std::ptr::null_mut(),
            &one,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_and_solve() {
        let a = Mat::from_row_slice(2, 2, &[4.0, 1.0, 2.0, 3.0]);
        let b = Mat::from_row_slice(2, 1, &[1.0, 2.0]);
        let x = solve(&a, &b).unwrap();
        let r = mul(&a, false, &x, false) - &b;
        assert!(r.amax() < 1e-14);
        let at_a = mul(&a, true, &a, false);
        assert!((at_a - a.transpose() * &a).amax() < 1e-14);
        let c = chol_upper(&(a.transpose() * &a)).unwrap();
        assert!((c.transpose() * &c - a.transpose() * &a).amax() < 1e-12);
    }

    /// The Rust gemm/gemv kernels agree with Accelerate (transposes, leading
    /// dimensions, beta = 0 ignoring a NaN-filled C, beta != 0).
    #[test]
    fn rust_backend_matches_accelerate() {
        let (m, n, k, ld) = (7, 5, 6, 9);
        let val = |i: usize| ((i * 7919) % 103) as f64 / 103.0 - 0.4;
        let a: Vec<f64> = (0..ld * ld).map(val).collect();
        let b: Vec<f64> = (0..ld * ld).map(|i| val(i + 11)).collect();
        for ta in [false, true] {
            for tb in [false, true] {
                for beta in [0.0, 1.5] {
                    let c0: Vec<f64> = (0..ld * n).map(|i| if beta == 0.0 { f64::NAN } else { val(i + 5) }).collect();
                    let (mut c1, mut c2) = (c0.clone(), c0.clone());
                    unsafe {
                        lapack::dgemm_(
                            &ch(ta),
                            &ch(tb),
                            &(m as c_int),
                            &(n as c_int),
                            &(k as c_int),
                            &0.7,
                            a.as_ptr(),
                            &(ld as c_int),
                            b.as_ptr(),
                            &(ld as c_int),
                            &beta,
                            c1.as_mut_ptr(),
                            &(ld as c_int),
                        )
                    };
                    gemm_rust(ta, tb, m, n, k, 0.7, a.as_ptr(), ld, b.as_ptr(), ld, beta, c2.as_mut_ptr(), ld);
                    for j in 0..n {
                        for i in 0..m {
                            let (x, y) = (c1[j * ld + i], c2[j * ld + i]);
                            assert!((x - y).abs() < 1e-14, "gemm ta={ta} tb={tb} beta={beta}: {x} vs {y}");
                        }
                    }
                }
            }
            for beta in [0.0, 1.5] {
                let (xl, yl) = if ta { (m, n) } else { (n, m) };
                let x: Vec<f64> = (0..xl).map(|i| val(i + 3)).collect();
                let y0: Vec<f64> = (0..yl).map(|i| if beta == 0.0 { f64::NAN } else { val(i + 1) }).collect();
                let (mut y1, mut y2) = (y0.clone(), y0.clone());
                let one: c_int = 1;
                unsafe {
                    lapack::dgemv_(
                        &ch(ta),
                        &(m as c_int),
                        &(n as c_int),
                        &0.7,
                        a.as_ptr(),
                        &(ld as c_int),
                        x.as_ptr(),
                        &one,
                        &beta,
                        y1.as_mut_ptr(),
                        &one,
                    )
                };
                gemv_rust(ta, m, n, 0.7, &a, ld, &x, beta, &mut y2);
                for (p, q) in y1.iter().zip(&y2) {
                    assert!((p - q).abs() < 1e-14, "gemv ta={ta} beta={beta}: {p} vs {q}");
                }
            }
        }
    }
}
