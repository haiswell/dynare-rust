// SPDX-License-Identifier: GPL-3.0-or-later
//! Raw FFI to the reference LAPACK / BLAS symbols exported by Accelerate.
//!
//! All matrices are column-major (same layout as nalgebra's `DMatrix` and
//! MATLAB), integers are 32-bit (`__CLPK_integer`). Wrappers live in
//! `linalg::mod`; add new declarations here when a module needs one.
#![allow(non_snake_case, clippy::too_many_arguments)]

use std::os::raw::{c_char, c_int};

/// Selection callback for dgges (`SELCTG`): (alphar, alphai, beta) -> keep in the top-left block.
pub type SelectFn3 = extern "C" fn(*const f64, *const f64, *const f64) -> c_int;
/// Selection callback for dgees (`SELECT`): (wr, wi) -> keep in the top-left block.
pub type SelectFn2 = extern "C" fn(*const f64, *const f64) -> c_int;

#[cfg_attr(target_os = "macos", link(name = "Accelerate", kind = "framework"))]
unsafe extern "C" {
    // ---- BLAS ---------------------------------------------------------------
    pub fn dgemm_(
        transa: *const c_char,
        transb: *const c_char,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const f64,
        a: *const f64,
        lda: *const c_int,
        b: *const f64,
        ldb: *const c_int,
        beta: *const f64,
        c: *mut f64,
        ldc: *const c_int,
    );
    pub fn dsymm_(
        side: *const c_char,
        uplo: *const c_char,
        m: *const c_int,
        n: *const c_int,
        alpha: *const f64,
        a: *const f64,
        lda: *const c_int,
        b: *const f64,
        ldb: *const c_int,
        beta: *const f64,
        c: *mut f64,
        ldc: *const c_int,
    );
    pub fn dsyrk_(
        uplo: *const c_char,
        trans: *const c_char,
        n: *const c_int,
        k: *const c_int,
        alpha: *const f64,
        a: *const f64,
        lda: *const c_int,
        beta: *const f64,
        c: *mut f64,
        ldc: *const c_int,
    );
    pub fn dgemv_(
        trans: *const c_char,
        m: *const c_int,
        n: *const c_int,
        alpha: *const f64,
        a: *const f64,
        lda: *const c_int,
        x: *const f64,
        incx: *const c_int,
        beta: *const f64,
        y: *mut f64,
        incy: *const c_int,
    );
    pub fn dtrsm_(
        side: *const c_char,
        uplo: *const c_char,
        transa: *const c_char,
        diag: *const c_char,
        m: *const c_int,
        n: *const c_int,
        alpha: *const f64,
        a: *const f64,
        lda: *const c_int,
        b: *mut f64,
        ldb: *const c_int,
    );

    // ---- LAPACK: linear systems ----------------------------------------------
    pub fn dgetrf_(
        m: *const c_int,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        ipiv: *mut c_int,
        info: *mut c_int,
    );
    pub fn dgetrs_(
        trans: *const c_char,
        n: *const c_int,
        nrhs: *const c_int,
        a: *const f64,
        lda: *const c_int,
        ipiv: *const c_int,
        b: *mut f64,
        ldb: *const c_int,
        info: *mut c_int,
    );
    pub fn dgetri_(
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        ipiv: *const c_int,
        work: *mut f64,
        lwork: *const c_int,
        info: *mut c_int,
    );
    pub fn dgecon_(
        norm: *const c_char,
        n: *const c_int,
        a: *const f64,
        lda: *const c_int,
        anorm: *const f64,
        rcond: *mut f64,
        work: *mut f64,
        iwork: *mut c_int,
        info: *mut c_int,
    );
    pub fn dpotrf_(uplo: *const c_char, n: *const c_int, a: *mut f64, lda: *const c_int, info: *mut c_int);
    pub fn dpotrs_(
        uplo: *const c_char,
        n: *const c_int,
        nrhs: *const c_int,
        a: *const f64,
        lda: *const c_int,
        b: *mut f64,
        ldb: *const c_int,
        info: *mut c_int,
    );
    pub fn dgels_(
        trans: *const c_char,
        m: *const c_int,
        n: *const c_int,
        nrhs: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        b: *mut f64,
        ldb: *const c_int,
        work: *mut f64,
        lwork: *const c_int,
        info: *mut c_int,
    );
    pub fn dgelsd_(
        m: *const c_int,
        n: *const c_int,
        nrhs: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        b: *mut f64,
        ldb: *const c_int,
        s: *mut f64,
        rcond: *const f64,
        rank: *mut c_int,
        work: *mut f64,
        lwork: *const c_int,
        iwork: *mut c_int,
        info: *mut c_int,
    );

    // ---- LAPACK: eigen / Schur / QZ ----------------------------------------
    pub fn dsyev_(
        jobz: *const c_char,
        uplo: *const c_char,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        w: *mut f64,
        work: *mut f64,
        lwork: *const c_int,
        info: *mut c_int,
    );
    pub fn dgeev_(
        jobvl: *const c_char,
        jobvr: *const c_char,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        wr: *mut f64,
        wi: *mut f64,
        vl: *mut f64,
        ldvl: *const c_int,
        vr: *mut f64,
        ldvr: *const c_int,
        work: *mut f64,
        lwork: *const c_int,
        info: *mut c_int,
    );
    pub fn dgees_(
        jobvs: *const c_char,
        sort: *const c_char,
        select: Option<SelectFn2>,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        sdim: *mut c_int,
        wr: *mut f64,
        wi: *mut f64,
        vs: *mut f64,
        ldvs: *const c_int,
        work: *mut f64,
        lwork: *const c_int,
        bwork: *mut c_int,
        info: *mut c_int,
    );
    /// Reorder a real Schur factorisation (MATLAB `ordschur`). `select` is a LOGICAL array.
    pub fn dtrsen_(
        job: *const c_char,
        compq: *const c_char,
        select: *const c_int,
        n: *const c_int,
        t: *mut f64,
        ldt: *const c_int,
        q: *mut f64,
        ldq: *const c_int,
        wr: *mut f64,
        wi: *mut f64,
        m: *mut c_int,
        s: *mut f64,
        sep: *mut f64,
        work: *mut f64,
        lwork: *const c_int,
        iwork: *mut c_int,
        liwork: *const c_int,
        info: *mut c_int,
    );
    pub fn dgges_(
        jobvsl: *const c_char,
        jobvsr: *const c_char,
        sort: *const c_char,
        selctg: Option<SelectFn3>,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        b: *mut f64,
        ldb: *const c_int,
        sdim: *mut c_int,
        alphar: *mut f64,
        alphai: *mut f64,
        beta: *mut f64,
        vsl: *mut f64,
        ldvsl: *const c_int,
        vsr: *mut f64,
        ldvsr: *const c_int,
        work: *mut f64,
        lwork: *const c_int,
        bwork: *mut c_int,
        info: *mut c_int,
    );
    pub fn dtrsyl_(
        trana: *const c_char,
        tranb: *const c_char,
        isgn: *const c_int,
        m: *const c_int,
        n: *const c_int,
        a: *const f64,
        lda: *const c_int,
        b: *const f64,
        ldb: *const c_int,
        c: *mut f64,
        ldc: *const c_int,
        scale: *mut f64,
        info: *mut c_int,
    );
    pub fn dgesvd_(
        jobu: *const c_char,
        jobvt: *const c_char,
        m: *const c_int,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        s: *mut f64,
        u: *mut f64,
        ldu: *const c_int,
        vt: *mut f64,
        ldvt: *const c_int,
        work: *mut f64,
        lwork: *const c_int,
        info: *mut c_int,
    );
    pub fn dgeqp3_(
        m: *const c_int,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        jpvt: *mut c_int,
        tau: *mut f64,
        work: *mut f64,
        lwork: *const c_int,
        info: *mut c_int,
    );
    pub fn dgeqrf_(
        m: *const c_int,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        tau: *mut f64,
        work: *mut f64,
        lwork: *const c_int,
        info: *mut c_int,
    );
    pub fn dormqr_(
        side: *const c_char,
        trans: *const c_char,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        a: *const f64,
        lda: *const c_int,
        tau: *const f64,
        c: *mut f64,
        ldc: *const c_int,
        work: *mut f64,
        lwork: *const c_int,
        info: *mut c_int,
    );
    pub fn dtrcon_(
        norm: *const c_char,
        uplo: *const c_char,
        diag: *const c_char,
        n: *const c_int,
        a: *const f64,
        lda: *const c_int,
        rcond: *mut f64,
        work: *mut f64,
        iwork: *mut c_int,
        info: *mut c_int,
    );
}
