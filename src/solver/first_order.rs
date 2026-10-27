// SPDX-License-Identifier: GPL-3.0-or-later
//! First-order perturbation solution.
//!
//! Port of `stochastic_solver/dyn_first_order_solver.m` (QZ path, task = 0)
//! plus the Jacobian checks and the purely backward branch of
//! `stochastic_solver/stochastic_solvers.m`.
//!
//! [`FirstOrderSolver::solve`] runs a structure-exploiting version of the QZ
//! step (`super::block_schur`, ~2x faster on SW07, more on models with long
//! aux chains) with Dynare's exact path ([`FirstOrderSolver::solve_full_qz`]) as
//! fallback; both give the same decision rule up to rounding (~1e-13 relative
//! on SW07, tests/solver_sw07.rs).

use super::block_schur;
use super::structure::DrStructure;
use super::{SolveError, SolverOptions};
use crate::linalg::{Mat, gemm, lu, lu_solve, mul, norm1, qr_qt_mul, qz::mjdgges, rcond_from_lu, trsm_left};
use nalgebra::Complex;

/// First-order decision rule `y_t - ys = ghx (y^s_{t-1} - ys^s) + ghu u_t`
/// (Dynare's `oo_.dr`). Rows of `ghx`/`ghu` are in DR order (`order_var`),
/// the columns of `ghx` are the state variables `state_var`.
#[derive(Debug, Clone)]
pub struct DecisionRule {
    pub ys: Vec<f64>,
    /// n_endo x nspred, rows in DR order.
    pub ghx: Mat,
    /// n_endo x n_exo, rows in DR order.
    pub ghu: Mat,
    pub order_var: Vec<usize>,
    pub inv_order_var: Vec<usize>,
    pub state_var: Vec<usize>,
    pub nstatic: usize,
    pub nfwrd: usize,
    pub npred: usize,
    pub nboth: usize,
    pub nspred: usize,
    pub nsfwrd: usize,
    /// Generalized eigenvalues as returned by mjdgges (`dr.eigval`).
    pub eigval: Vec<Complex<f64>>,
    /// `abs(dr.eigval)`.
    pub eigval_abs: Vec<f64>,
    /// Number of stable eigenvalues (`dr.sdim`).
    pub sdim: usize,
}

/// What `check.m` reports from the pencil ([`FirstOrderSolver::check_eigenvalues`]).
#[derive(Debug, Clone)]
pub struct CheckEigen {
    /// `dr.eigval` in mjdgges order.
    pub eigval: Vec<Complex<f64>>,
    pub sdim: usize,
    /// `dr.edim`: number of eigenvalues with modulus >= qz_criterium.
    pub edim: usize,
    pub full_rank: bool,
}

/// Structure-only part of `dyn_first_order_solver.m` (its `persistent`
/// index sets), built once; [`FirstOrderSolver::solve`] is then a pure
/// function of the Jacobian, safe to call from several threads.
#[derive(Debug, Clone)]
pub struct FirstOrderSolver {
    pub structure: DrStructure,
    pub opts: SolverOptions,
    /// `reorder_jacobian_columns` (0-based columns of the dense g1).
    jac_cols: Vec<usize>,
    /// DR positions of the variables present at t (`cols_b`).
    cols_b: Vec<usize>,
    index_s: Vec<usize>,
    index_d: Vec<usize>,
    index_d1: Vec<usize>,
    index_e: Vec<usize>,
    index_e1: Vec<usize>,
}

/// Info code of [`FirstOrderSolver::solve_structured_only`] when the structured
/// path declines (never returned by [`FirstOrderSolver::solve`]).
#[doc(hidden)]
pub const STRUCTURED_DECLINED: i32 = -1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Path {
    /// Structured path, full QZ as fallback.
    Auto,
    /// Dynare's dense path only.
    Full,
    /// Structured path, no fallback (tests).
    StructuredOnly,
}

fn range(start: usize, len: usize) -> Vec<usize> {
    (start..start + len).collect()
}

impl FirstOrderSolver {
    /// `lli[i] = [lag, current, lead]` per endogenous variable (nonzero = present),
    /// see [`DrStructure::new`].
    pub fn new(lli: &[[usize; 3]], n_exo: usize, opts: SolverOptions) -> Self {
        let s = DrStructure::new(lli, n_exo);
        let n = s.n_endo;
        let (nstatic, npred, nspred, nsfwrd) = (s.nstatic, s.npred, s.nspred, s.nsfwrd);
        let ov = &s.order_var;
        let cur = |k: usize| s.current[ov[k]];
        let cols_b: Vec<usize> = (0..n).filter(|&k| cur(k)).collect();
        let n_current = cols_b.len();
        debug_assert!((0..nstatic).all(cur), "static variables must appear at t");

        let mut jac_cols: Vec<usize> = (0..nspred).map(|k| ov[nstatic + k]).collect();
        jac_cols.extend(cols_b.iter().map(|&k| n + ov[k]));
        jac_cols.extend((0..nsfwrd).map(|k| 2 * n + ov[nstatic + npred + k]));
        jac_cols.extend((0..n_exo).map(|j| 3 * n + j));

        // positions in the reordered Jacobian `jacobia`
        let index_m = range(0, nspred);
        let index_p = range(nspred + n_current, nsfwrd);
        let index_s = range(nspred, nstatic);
        let npred0 = (0..npred).filter(|&k| cur(nstatic + k)).count();
        let nfwrd0 = (0..nsfwrd).filter(|&k| cur(nstatic + npred + k)).count();
        let index_0m = range(nspred + nstatic, npred0);
        let index_0p = range(nspred + nstatic + npred0, nfwrd0);
        let index_d: Vec<usize> = index_0m.iter().chain(&index_p).copied().collect();
        let mut index_d1: Vec<usize> = (0..npred).filter(|&k| cur(nstatic + k)).collect();
        index_d1.extend(range(nspred, nsfwrd));
        let index_e: Vec<usize> = index_m.iter().chain(&index_0p).copied().collect();
        let mut index_e1 = range(0, nspred);
        index_e1.extend((0..nsfwrd).filter(|&k| cur(nstatic + npred + k)).map(|k| nspred + k));

        FirstOrderSolver { structure: s, opts, jac_cols, cols_b, index_s, index_d, index_d1, index_e, index_e1 }
    }

    fn rule(&self, ys: &[f64], ghx: Mat, ghu: Mat, eigval: Vec<Complex<f64>>, sdim: usize) -> DecisionRule {
        let s = &self.structure;
        DecisionRule {
            ys: ys.to_vec(),
            ghx,
            ghu,
            order_var: s.order_var.clone(),
            inv_order_var: s.inv_order_var.clone(),
            state_var: s.state_var.clone(),
            nstatic: s.nstatic,
            nfwrd: s.nfwrd,
            npred: s.npred,
            nboth: s.nboth,
            nspred: s.nspred,
            nsfwrd: s.nsfwrd,
            eigval_abs: eigval.iter().map(|z| z.norm()).collect(),
            eigval,
            sdim,
        }
    }

    // ---- stream S (check command) -------------------------------------------------
    /// The `task = 1` part of `dyn_first_order_solver.m` that `check.m` reports: the
    /// eigenvalues of the full pencil from one `mjdgges` (Dynare's exact path: dense QR static
    /// elimination), `dr.sdim`, `dr.edim` and `dr.full_rank`
    /// (`rcond(w(npred+nboth+1:end, npred+nboth+1:end)) >= 1e-9`). Purely backward models
    /// (stochastic_solvers.m): the eigenvalues of the state transition, `edim` their number
    /// above `qz_criterium` (explosive roots are not an error here, `info = 3` is not printed
    /// by check.m) and `full_rank = 1`.
    /// `Err` only for mjdgges failures (info 2 / 7) and NaN/Inf Jacobians.
    pub fn check_eigenvalues(&self, g1: &Mat) -> Result<CheckEigen, SolveError> {
        let s = &self.structure;
        if g1.as_slice().iter().any(|x| !x.is_finite()) {
            let inf = g1.as_slice().iter().any(|x| x.is_infinite());
            return Err(SolveError::new(if inf { 10 } else { 8 }, 0.0));
        }
        if s.maximum_endo_lead == 0 {
            return self.check_backward(g1);
        }
        let (nstatic, npred, nboth, nspred, ndynamic) = (s.nstatic, s.npred, s.nboth, s.nspred, s.ndynamic);
        let nd = ndynamic + nboth;
        let mut aa = g1.select_columns(&self.jac_cols);
        if nstatic > 0 {
            let a_s = aa.select_columns(&self.index_s);
            qr_qt_mul(&a_s, &mut aa);
        }
        let mut d = Mat::zeros(nd, nd);
        let mut e = Mat::zeros(nd, nd);
        for (&src, &dst) in self.index_d.iter().zip(&self.index_d1) {
            for r in 0..ndynamic {
                d[(r, dst)] = aa[(nstatic + r, src)];
            }
        }
        for (&src, &dst) in self.index_e.iter().zip(&self.index_e1) {
            for r in 0..ndynamic {
                e[(r, dst)] = -aa[(nstatic + r, src)];
            }
        }
        for k in 0..nboth {
            d[(ndynamic + k, npred + k)] = 1.0;
            e[(ndynamic + k, nspred + k)] = 1.0;
        }
        let qz = mjdgges(&e, &d, self.opts.qz_criterium, self.opts.qz_zero_threshold);
        if qz.info != 0 {
            return Err(if qz.info == -30 { SolveError::new(7, 0.0) } else { SolveError::new(2, qz.info as f64) });
        }
        let k0 = npred + nboth;
        let w22 = qz.z.view((k0, k0), (nd - k0, nd - k0)).clone_owned();
        let full_rank = if w22.nrows() == 0 {
            true
        } else {
            let (f, _, _) = lu(&w22);
            let rc = rcond_from_lu(&f, norm1(&w22));
            !(rc.is_nan() || rc < 1e-9)
        };
        let edim = qz.eigval.len() - qz.sdim;
        Ok(CheckEigen { eigval: qz.eigval, sdim: qz.sdim, edim, full_rank })
    }
    // ---- end stream S ---------------------------------------------------------------

    /// Solve for the decision rule given the dynamic Jacobian `g1` at the
    /// steady state `ys` (layout of `Model::dynamic_jacobian`:
    /// n_endo x (3 n_endo + n_exo), columns `[y_{t-1}, y_t, y_{t+1}, x]`).
    ///
    /// Mathematically the same computation as Dynare, organised around the
    /// sparsity of the problem: a sparse static elimination
    /// ([`Self::eliminate_static_sparse`]), the stable deflating subspace from
    /// `block_schur` (a small unordered QZ of the core plus Sylvester solves
    /// instead of an ordered QZ of the whole pencil), and block-triangular solves
    /// for the static rows and `ghu`. It falls back to Dynare's path
    /// ([`Self::solve_full_qz`]) whenever a structural or numerical safeguard
    /// trips: structurally singular pencil, 0/0 roots (info 7 then comes from
    /// mjdgges), roots within 1e-9 (relative) of `qz_criterium`, nearly common
    /// roots in a Sylvester equation, or a rank condition anywhere near
    /// Dynare's `rcond(Z22) < 1e-9` (info 5). Blanchard–Kahn order failures
    /// (info 3/4) and their `info(2)` penalty are computed from the same
    /// eigenvalues as mjdgges' (up to rounding).
    pub fn solve(&self, g1: &Mat, ys: &[f64]) -> Result<DecisionRule, SolveError> {
        self.solve_impl(g1, ys, Path::Auto)
    }

    /// [`Self::solve`] through Dynare's exact path: dense QR static elimination
    /// and one `mjdgges` on the full pencil (kept as the reference and fallback).
    pub fn solve_full_qz(&self, g1: &Mat, ys: &[f64]) -> Result<DecisionRule, SolveError> {
        self.solve_impl(g1, ys, Path::Full)
    }

    /// Test hook: [`Self::solve`] without the fallback; `Err` with info
    /// [`STRUCTURED_DECLINED`] where `solve` would fall back to the full QZ.
    #[doc(hidden)]
    pub fn solve_structured_only(&self, g1: &Mat, ys: &[f64]) -> Result<DecisionRule, SolveError> {
        self.solve_impl(g1, ys, Path::StructuredOnly)
    }

    fn solve_impl(&self, g1: &Mat, ys: &[f64], path: Path) -> Result<DecisionRule, SolveError> {
        // cycle / logarithmic reduction run on Dynare's dense QR elimination
        let reduction = !matches!(self.opts.method, super::FirstOrderMethod::Qz);
        let use_blocks = path != Path::Full && !reduction;
        let s = &self.structure;
        let n = s.n_endo;
        assert_eq!(g1.nrows(), n, "first_order: g1 must have n_endo rows");
        assert_eq!(g1.ncols(), 3 * n + s.n_exo, "first_order: g1 must have 3 n_endo + n_exo columns");
        // stochastic_solvers.m: Inf (info 10) then NaN (info 8) in the Jacobian
        if g1.as_slice().iter().any(|x| !x.is_finite()) {
            if g1.as_slice().iter().any(|x| x.is_infinite()) {
                return Err(SolveError::new(10, 0.0));
            }
            return Err(SolveError::new(8, 0.0));
        }
        if s.maximum_endo_lead == 0 {
            return self.solve_backward(g1, ys);
        }

        let (nstatic, npred, nboth, nspred, nsfwrd, ndynamic) =
            (s.nstatic, s.npred, s.nboth, s.nspred, s.nsfwrd, s.ndynamic);
        let nd = ndynamic + nboth;

        // jacobia = g1(:, reorder_jacobian_columns); aa = Q' * jacobia
        let mut aa = g1.select_columns(&self.jac_cols);
        if nstatic > 0 {
            let sparse = use_blocks && self.eliminate_static_sparse(&mut aa);
            if !sparse {
                let a_s = aa.select_columns(&self.index_s);
                qr_qt_mul(&a_s, &mut aa);
            }
        }

        // ---- dr=cycle_reduction | logarithmic_reduction ----
        if reduction {
            let (gx, hx) = self.reduction_gx_hx(&aa)?;
            return self.finish_rule(&aa, ys, gx, hx, Vec::new(), 0, false);
        }

        // D/E pencil
        let mut d = Mat::zeros(nd, nd);
        let mut e = Mat::zeros(nd, nd);
        for (&src, &dst) in self.index_d.iter().zip(&self.index_d1) {
            for r in 0..ndynamic {
                d[(r, dst)] = aa[(nstatic + r, src)];
            }
        }
        for (&src, &dst) in self.index_e.iter().zip(&self.index_e1) {
            for r in 0..ndynamic {
                e[(r, dst)] = -aa[(nstatic + r, src)];
            }
        }
        for k in 0..nboth {
            d[(ndynamic + k, npred + k)] = 1.0;
            e[(ndynamic + k, nspred + k)] = 1.0;
        }

        // Fast path: block-triangularised pencil (see `block_schur`); any doubt -> full QZ.
        let mut fast = None;
        if use_blocks {
            if let Ok(ss) = block_schur::stable_subspace(&e, &d, self.opts.qz_criterium, self.opts.qz_zero_threshold) {
                let nba = nd - ss.sdim;
                if nba != nsfwrd {
                    return Err(self.bk_error(&ss.eigval, nba));
                }
                if let Some((gx, hx)) = rule_from_subspace(&ss, nspred) {
                    fast = Some((gx, hx, ss.eigval, ss.sdim));
                }
            }
        }
        let (gx, hx, eigval, sdim) = match fast {
            Some(x) => x,
            None if path == Path::StructuredOnly => return Err(SolveError::new(STRUCTURED_DECLINED, 0.0)),
            None => self.full_qz(&e, &d)?,
        };

        self.finish_rule(&aa, ys, gx, hx, eigval, sdim, use_blocks)
    }

    /// Whether `dr=cycle_reduction|logarithmic_reduction` applies: Dynare requires every
    /// variable to appear at `t` (`n_current == endo_nbr`), else it errors out.
    pub fn reduction_applicable(&self) -> bool {
        self.cols_b.len() == self.structure.n_endo
    }

    /// The `dr_cycle_reduction || dr_logarithmic_reduction` branch of
    /// `dyn_first_order_solver.m`: `gx`, `hx` from the minimal solvent of
    /// `A1 + B1 X + C1 X^2 = 0` on the dynamic rows of `aa`.
    fn reduction_gx_hx(&self, aa: &Mat) -> Result<(Mat, Mat), SolveError> {
        let s = &self.structure;
        let (nstatic, npred, nspred, nsfwrd, nd) = (s.nstatic, s.npred, s.nspred, s.nsfwrd, s.ndynamic);
        let n_current = self.cols_b.len();
        if n_current < s.n_endo {
            return Err(SolveError::new(1, 0.0));
        }
        let a1 = Mat::from_fn(nd, nd, |r, c| if c < nspred { aa[(nstatic + r, c)] } else { 0.0 });
        let b1 = Mat::from_fn(nd, nd, |r, c| aa[(nstatic + r, nspred + nstatic + c)]);
        let c1 = Mat::from_fn(
            nd,
            nd,
            |r, c| if c < npred { 0.0 } else { aa[(nstatic + r, nspred + n_current + c - npred)] },
        );
        let x = match self.opts.method {
            super::FirstOrderMethod::CycleReduction { tol, maxiter } => {
                super::reduction::cycle_reduction(&a1, &b1, &c1, tol, maxiter)
            }
            super::FirstOrderMethod::LogarithmicReduction { tol, maxiter } => {
                super::reduction::logarithmic_reduction(&c1, &b1, &a1, tol, maxiter)
            }
            super::FirstOrderMethod::Qz => unreachable!(),
        }
        .map_err(|(i, v)| SolveError::new(i, v))?;
        let hx = x.view((0, 0), (nspred, nspred)).into_owned();
        let gx = x.view((npred, 0), (nsfwrd, nspred)).into_owned();
        Ok((gx, hx))
    }

    /// The end of `dyn_first_order_solver.m` once `gx`/`hx` are known: static rows of `ghx`,
    /// then `ghu`.
    #[allow(clippy::too_many_arguments)]
    fn finish_rule(
        &self,
        aa: &Mat,
        ys: &[f64],
        gx: Mat,
        hx: Mat,
        eigval: Vec<Complex<f64>>,
        sdim: usize,
        use_blocks: bool,
    ) -> Result<DecisionRule, SolveError> {
        let s = &self.structure;
        let n = s.n_endo;
        let (nstatic, nboth, nspred, nsfwrd) = (s.nstatic, s.nboth, s.nspred, s.nsfwrd);
        let n_current = self.cols_b.len();
        // ghx (dynamic part) = [hx; gx(nboth+1:end, :)]
        let nfwrd = nsfwrd - nboth;
        let mut ghx = Mat::zeros(n, nspred);
        ghx.view_mut((nstatic, 0), (nspred, nspred)).copy_from(&hx);
        ghx.view_mut((nstatic + nspred, 0), (nfwrd, nspred)).copy_from(&gx.view((nboth, 0), (nfwrd, nspred)));

        if use_blocks && nstatic > 0 && s.n_exo > 0 {
            if let Some(ghu) = self.static_and_shocks(aa, &gx, &hx, &mut ghx) {
                return Ok(self.rule(ys, ghx, ghu, eigval, sdim));
            }
        }

        // C = aa(:, index_p); B(:, cols_b) = aa(:, index_c)
        let c = aa.columns(nspred + n_current, nsfwrd).clone_owned();
        let cgx = mul(&c, false, &gx, false); // n x nspred
        let mut b = Mat::zeros(n, n);
        for (j, &k) in self.cols_b.iter().enumerate() {
            b.column_mut(k).copy_from(&aa.column(nspred + j));
        }

        if nstatic > 0 {
            // temp = -C(1:nstatic,:)*gx*hx - A(1:nstatic,:) - b11*ghx_dyn; temp = b10 \ temp
            let cgx_s = cgx.rows(0, nstatic).clone_owned();
            let mut temp = Mat::zeros(nstatic, nspred);
            gemm(-1.0, &cgx_s, false, &hx, false, 0.0, &mut temp);
            temp -= aa.view((0, 0), (nstatic, nspred));
            let b11 = b.view((0, nstatic), (nstatic, n - nstatic)).clone_owned();
            let ghx_dyn = ghx.rows(nstatic, n - nstatic).clone_owned();
            gemm(-1.0, &b11, false, &ghx_dyn, false, 1.0, &mut temp);
            let b10 = b.view((0, 0), (nstatic, nstatic)).clone_owned();
            let (f10, piv10, _) = lu(&b10);
            lu_solve(&f10, &piv10, false, &mut temp);
            ghx.rows_mut(0, nstatic).copy_from(&temp);
        }

        // A_ = [B_static, C*gx + B_pred, B_fyd]; ghu = -A_ \ fu
        let ghu = if s.n_exo > 0 {
            let mut a_ = b;
            {
                let mut mid = a_.columns_mut(nstatic, nspred);
                mid += &cgx;
            }
            let mut fu = aa.columns(nspred + n_current + nsfwrd, s.n_exo).clone_owned();
            let (fa, pa, _) = lu(&a_);
            lu_solve(&fa, &pa, false, &mut fu);
            fu.neg_mut();
            fu
        } else {
            Mat::zeros(n, 0)
        };

        Ok(self.rule(ys, ghx, ghu, eigval, sdim))
    }

    /// The static rows of `ghx` and `ghu` (`dyn_first_order_solver.m`, after
    /// `gx`/`hx`), exploiting that the static columns of `aa` vanish on the
    /// dynamic rows (that is what the static elimination does; the dense QR
    /// leaves rounding residuals of order eps there, which are dropped).
    /// `A_ = [B_static, C*gx + B_pred, B_fyd]` is then block upper triangular,
    /// so `ghu = -A_ \ fu` is one LU of the dynamic block plus a solve with
    /// `b10 = B(1:nstatic, 1:nstatic)`, whose LU also gives the static rows of
    /// `ghx`. Writes `ghx(1:nstatic, :)`, returns `ghu`; `None` if a block is
    /// exactly singular (the caller then runs Dynare's dense formulas).
    fn static_and_shocks(&self, aa: &Mat, gx: &Mat, hx: &Mat, ghx: &mut Mat) -> Option<Mat> {
        use crate::linalg::{gemm_raw, getrf_raw, getrs_raw};
        let s = &self.structure;
        let (n, nstatic, nspred, nsfwrd, n_exo) = (s.n_endo, s.nstatic, s.nspred, s.nsfwrd, s.n_exo);
        let n_current = self.cols_b.len();
        let n2 = n - nstatic;
        let a = aa.as_slice();
        let col = |j: usize| j * n;
        // C*gx (n x nspred)
        let mut cgx = Mat::zeros(n, nspred);
        gemm_raw(
            false,
            false,
            n,
            nspred,
            nsfwrd,
            1.0,
            &a[col(nspred + n_current)..],
            n,
            gx.as_slice(),
            nsfwrd,
            0.0,
            cgx.as_mut_slice(),
            n,
        );
        // b10 = B(1:nstatic, 1:nstatic) = aa(1:nstatic, nspred + (1:nstatic))
        let mut b10 = Mat::from_fn(nstatic, nstatic, |i, j| a[i + col(nspred + j)]);
        let mut piv10 = vec![0; nstatic];
        if getrf_raw(nstatic, b10.as_mut_slice(), nstatic, &mut piv10) != 0 {
            return None;
        }
        // temp = -C(1:nstatic,:)*gx*hx - A(1:nstatic,:) - b11*ghx_dyn ; ghx_s = b10 \ temp
        let mut temp = Mat::from_fn(nstatic, nspred, |i, j| -a[i + col(j)]);
        gemm_raw(
            false,
            false,
            nstatic,
            nspred,
            nspred,
            -1.0,
            cgx.as_slice(),
            n,
            hx.as_slice(),
            nspred,
            1.0,
            temp.as_mut_slice(),
            nstatic,
        );
        let ncd = n_current - nstatic; // dynamic variables present at t
        if ncd > 0 {
            let g = Mat::from_fn(ncd, nspred, |i, j| ghx[(self.cols_b[nstatic + i], j)]);
            gemm_raw(
                false,
                false,
                nstatic,
                nspred,
                ncd,
                -1.0,
                &a[col(nspred + nstatic)..],
                n,
                g.as_slice(),
                ncd,
                1.0,
                temp.as_mut_slice(),
                nstatic,
            );
        }
        getrs_raw(false, nstatic, nspred, b10.as_slice(), nstatic, &piv10, temp.as_mut_slice(), nstatic);
        ghx.rows_mut(0, nstatic).copy_from(&temp);

        // A_(:, nstatic+1:end) = [C*gx + B_pred, B_fyd]
        let mut a_dyn = Mat::zeros(n, n2);
        for j in nstatic..n_current {
            let k = self.cols_b[j] - nstatic;
            a_dyn.as_mut_slice()[k * n..(k + 1) * n].copy_from_slice(&a[col(nspred + j)..col(nspred + j + 1)]);
        }
        for (x, y) in a_dyn.as_mut_slice()[..nspred * n].iter_mut().zip(cgx.as_slice()) {
            *x += y;
        }
        // dynamic block: A22 u2 = -fu2 ; static block: b10 u1 = -fu1 - A12 u2
        let fu0 = col(nspred + n_current + nsfwrd);
        let mut ghu = Mat::from_fn(n, n_exo, |i, j| -a[fu0 + i + j * n]);
        let mut u2 = ghu.rows(nstatic, n2).clone_owned();
        let mut piv2 = vec![0; n2];
        if getrf_raw(n2, &mut a_dyn.as_mut_slice()[nstatic..], n, &mut piv2) != 0 {
            return None;
        }
        getrs_raw(false, n2, n_exo, &a_dyn.as_slice()[nstatic..], n, &piv2, u2.as_mut_slice(), n2);
        // (rows 1:nstatic of a_dyn are untouched by the in-place LU of rows nstatic+1:end)
        gemm_raw(
            false,
            false,
            nstatic,
            n_exo,
            n2,
            -1.0,
            a_dyn.as_slice(),
            n,
            u2.as_slice(),
            n2,
            1.0,
            ghu.as_mut_slice(),
            n,
        );
        let mut u1 = ghu.rows(0, nstatic).clone_owned();
        getrs_raw(false, nstatic, n_exo, b10.as_slice(), nstatic, &piv10, u1.as_mut_slice(), nstatic);
        ghu.rows_mut(0, nstatic).copy_from(&u1);
        ghu.rows_mut(nstatic, n2).copy_from(&u2);
        Some(ghu)
    }

    /// Sparse version of `dyn_first_order_solver.m`'s static elimination
    /// `[Q, ~] = qr(jacobia(:, index_s)); aa = Q' * jacobia`, in place on `aa`.
    ///
    /// Any orthonormal basis of the left null space of `A_s = jacobia(:, index_s)`
    /// gives a pencil that differs from Dynare's by a left orthogonal factor (same
    /// roots, same deflating subspaces), and any orthogonal `Q` keeps `ghx`/`ghu`
    /// unchanged. Dynare's dense `Q` mixes every equation that touches a static
    /// variable into the dynamic rows, which couples otherwise independent parts
    /// of the pencil (e.g. the long lead chains of multi-period averages). Here:
    /// 1. static columns with a single nonzero among the remaining rows are
    ///    peeled with their row (that equation just defines the static variable;
    ///    the peeled rows x pivots are triangular, hence nonsingular);
    /// 2. a Householder QR (as Dynare's) of the rows still touching the remaining
    ///    static columns, applied to those rows only;
    /// 3. `aa = [peeled rows; R rows; residual rows of 2.; untouched rows]`.
    ///
    /// Returns false (and leaves `aa` alone) when 2. has fewer rows than columns
    /// (structurally rank deficient `A_s`); the caller then runs the dense QR.
    fn eliminate_static_sparse(&self, aa: &mut Mat) -> bool {
        let n = aa.nrows();
        let cols = &self.index_s;
        let nz = |r: usize, c: usize| aa[(r, c)] != 0.0;
        let mut peeled_row = vec![false; n];
        let mut col_done = vec![false; cols.len()];
        let mut peeled = Vec::new();
        loop {
            let mut progress = false;
            for (k, &c) in cols.iter().enumerate() {
                if col_done[k] {
                    continue;
                }
                let mut only = None;
                let mut count = 0;
                for r in 0..n {
                    if !peeled_row[r] && nz(r, c) {
                        count += 1;
                        only = Some(r);
                        if count > 1 {
                            break;
                        }
                    }
                }
                if count == 1 {
                    let r = only.unwrap();
                    peeled_row[r] = true;
                    col_done[k] = true;
                    peeled.push(r);
                    progress = true;
                }
            }
            if !progress {
                break;
            }
        }
        let rest_cols: Vec<usize> = cols.iter().enumerate().filter(|(k, _)| !col_done[*k]).map(|(_, &c)| c).collect();
        let rest_rows: Vec<usize> = (0..n).filter(|&r| !peeled_row[r] && rest_cols.iter().any(|&c| nz(r, c))).collect();
        let k = rest_cols.len();
        if rest_rows.len() < k {
            return false;
        }
        let mut sub = aa.select_rows(&rest_rows);
        if k > 0 {
            let a_s = sub.select_columns(&rest_cols);
            qr_qt_mul(&a_s, &mut sub);
        }
        let in_rest = {
            let mut v = vec![false; n];
            for &r in &rest_rows {
                v[r] = true;
            }
            v
        };
        let mut out = Mat::zeros(n, aa.ncols());
        let mut row = 0;
        for &r in &peeled {
            out.row_mut(row).copy_from(&aa.row(r));
            row += 1;
        }
        for i in 0..sub.nrows() {
            out.row_mut(row).copy_from(&sub.row(i));
            row += 1;
        }
        for r in 0..n {
            if !peeled_row[r] && !in_rest[r] {
                out.row_mut(row).copy_from(&aa.row(r));
                row += 1;
            }
        }
        debug_assert_eq!(row, n);
        *aa = out;
        true
    }

    /// `dyn_first_order_solver.m`'s Blanchard–Kahn order condition failure:
    /// info 3 (too many unstable roots) or 4 (too few), `info(2)` = the sum of
    /// `(|lambda| - 1 - qz_criterium)^2` over the misclassified roots.
    fn bk_error(&self, eigval: &[Complex<f64>], nba: usize) -> SolveError {
        let s = &self.structure;
        let (nd, nsfwrd) = (s.nspred + s.nsfwrd, s.nsfwrd);
        let mut temp: Vec<f64> = eigval.iter().map(|z| z.norm()).collect();
        temp.sort_by(f64::total_cmp); // MATLAB sort: NaN last
        let (info, sel) =
            if nba > nsfwrd { (3, &temp[nd - nba..nd - nsfwrd]) } else { (4, &temp[nd - nsfwrd..nd - nba]) };
        let pen = sel.iter().map(|x| {
            let t = x - 1.0 - self.opts.qz_criterium;
            t * t
        });
        SolveError::new(info, pen.sum())
    }

    /// The QZ part of `dyn_first_order_solver.m` exactly as Dynare runs it: one
    /// `mjdgges` on the full pencil, then `gx = -Z22 \ Z21` (with the rank check
    /// `rcond(Z22) < 1e-9` -> info 5) and `hx = Z11' * (tt11 \ ss11) / Z11'`.
    #[allow(clippy::type_complexity)]
    fn full_qz(&self, e: &Mat, d: &Mat) -> Result<(Mat, Mat, Vec<Complex<f64>>, usize), SolveError> {
        let s = &self.structure;
        let (nspred, nsfwrd) = (s.nspred, s.nsfwrd);
        let nd = nspred + nsfwrd;
        let qz = mjdgges(e, d, self.opts.qz_criterium, self.opts.qz_zero_threshold);
        if qz.info != 0 {
            return Err(if qz.info == -30 { SolveError::new(7, 0.0) } else { SolveError::new(2, qz.info as f64) });
        }
        let sdim = qz.sdim;
        let nba = nd - sdim;
        if nba != nsfwrd {
            return Err(self.bk_error(&qz.eigval, nba));
        }

        // Z = w'
        let w = &qz.z;
        let z11 = w.view((0, 0), (nspred, nspred)).transpose();
        let z21 = w.view((0, nspred), (nspred, nsfwrd)).transpose();
        let z22 = w.view((nspred, nspred), (nsfwrd, nsfwrd)).transpose();
        // [minus_gx, rc] = linsolve(Z22, Z21)
        let (f22, piv22, _) = lu(&z22);
        let rc = rcond_from_lu(&f22, norm1(&z22));
        let mut gx = z21;
        lu_solve(&f22, &piv22, false, &mut gx);
        if rc.is_nan() || rc < 1e-9 {
            return Err(SolveError::new(5, -rc.ln()));
        }
        gx.neg_mut();
        // hx = Z11' * tt11^{-1} * ss11 * Z11'^{-1}
        let mut x1 = z11.clone();
        trsm_left(true, true, 1.0, &qz.t.view((0, 0), (nspred, nspred)).clone_owned(), &mut x1);
        let hx1 = x1.transpose();
        let mut x2 = qz.s.view((0, 0), (nspred, nspred)).transpose();
        let (f11, piv11, _) = lu(&z11);
        lu_solve(&f11, &piv11, false, &mut x2);
        let hx2 = x2.transpose();
        let hx = mul(&hx1, false, &hx2, false);
        Ok((gx, hx, qz.eigval, sdim))
    }

    /// The eigenvalues and `edim` of a purely backward model (`stochastic_solvers.m`, order 1),
    /// as [`Self::solve_backward`] computes them but without failing on explosive roots.
    fn check_backward(&self, g1: &Mat) -> Result<CheckEigen, SolveError> {
        let s = &self.structure;
        let n = s.n_endo;
        if self.cols_b.len() != n {
            return Err(SolveError::new(1, 0.0));
        }
        let bcols: Vec<usize> = self.cols_b.iter().map(|&k| n + s.order_var[k]).collect();
        let (fb, pb, _) = lu(&g1.select_columns(&bcols));
        let xcols: Vec<usize> = (0..s.nspred).map(|k| s.order_var[s.nstatic + k]).collect();
        let mut ghx = g1.select_columns(&xcols);
        lu_solve(&fb, &pb, false, &mut ghx);
        ghx.neg_mut();
        let eigval = crate::linalg::eig_general(&ghx.rows(s.nstatic, s.nspred).clone_owned());
        let edim = eigval.iter().filter(|z| z.norm() > self.opts.qz_criterium).count();
        Ok(CheckEigen { sdim: s.nspred + s.nsfwrd - edim, eigval, edim, full_rank: true })
    }

    /// Purely backward models (`M_.maximum_endo_lead == 0`), `stochastic_solvers.m`.
    fn solve_backward(&self, g1: &Mat, ys: &[f64]) -> Result<DecisionRule, SolveError> {
        let s = &self.structure;
        let n = s.n_endo;
        if self.cols_b.len() != n {
            // MATLAB would take a least-squares solution of a non-square system.
            return Err(SolveError::new(1, 0.0));
        }
        let bcols: Vec<usize> = self.cols_b.iter().map(|&k| n + s.order_var[k]).collect();
        let b = g1.select_columns(&bcols);
        let (fb, pb, _) = lu(&b);
        let xcols: Vec<usize> = (0..s.nspred).map(|k| s.order_var[s.nstatic + k]).collect();
        let mut ghx = g1.select_columns(&xcols);
        lu_solve(&fb, &pb, false, &mut ghx);
        ghx.neg_mut();
        let ucols: Vec<usize> = (3 * n..3 * n + s.n_exo).collect();
        let mut ghu = g1.select_columns(&ucols);
        lu_solve(&fb, &pb, false, &mut ghu);
        ghu.neg_mut();
        let a = ghx.rows(s.nstatic, s.nspred).clone_owned();
        let eigval = crate::linalg::eig_general(&a);
        let nd = s.nspred + s.nsfwrd;
        let mut abs: Vec<f64> = eigval.iter().map(|z| z.norm()).collect();
        let edim = abs.iter().filter(|&&x| x > self.opts.qz_criterium).count();
        let sdim = nd - edim;
        if edim > 0 {
            abs.sort_by(f64::total_cmp);
            let pen = abs[sdim..nd].iter().map(|x| (x - 1.0 - self.opts.qz_criterium).powi(2)).sum();
            return Err(SolveError::new(3, pen));
        }
        Ok(self.rule(ys, ghx, ghu, eigval, sdim))
    }
}

/// Rank guard of the fast path. Dynare declares a rank failure (info 5) when
/// `rcond(Z22) < 1e-9`, `Z` being the orthogonal Schur vectors of the full QZ.
/// With `W = V R^{-1}` an orthonormal basis of the stable subspace (`R` from a
/// thin QR of `V`), `W1 = V1 R^{-1}` and Dynare's `Z22` are complementary blocks
/// of an orthogonal matrix and share their smallest singular value (CS
/// decomposition), both having norm <= 1, so
/// `cond2(Z22) <= 1 / sigma_min(W1) <= ||R||_2 ||V1^{-1}||_2 <= ||V||_F sqrt(nspred) ||V1^{-1}||_1`.
/// The fast path is only kept when `||V||_F * est(||V1^{-1}||_1) <= KAPPA_GUARD`;
/// with nspred, nsfwrd ~ 100 that bounds Dynare's `1 / rcond(Z22)` (1-norm,
/// <= nsfwrd * cond2) by ~1e8 even if dgecon underestimates by 10x, well
/// clear of 1e9. Otherwise the full QZ runs and applies Dynare's own check.
const KAPPA_GUARD: f64 = 1e4;

/// `gx`, `hx` from a stable deflating subspace basis `V = [V1; V2]`
/// (`E V = D V M`, `M = T^{-1} S`): `gx = V2 V1^{-1}`, `hx = V1 M V1^{-1}`.
/// These are `dyn_first_order_solver.m`'s `-Z22 \ Z21` and
/// `Z11' (tt11 \ ss11) / Z11'` (with `Z = w'`, `[w11; w21]` is one basis of
/// the stable subspace, and both formulas are invariant under a change of
/// basis). `None` if `V1` fails the rank guard (see [`KAPPA_GUARD`]).
fn rule_from_subspace(ss: &block_schur::StableSubspace, nspred: usize) -> Option<(Mat, Mat)> {
    let nd = ss.v.nrows();
    let nsfwrd = nd - nspred;
    let v1 = ss.v.rows(0, nspred).clone_owned();
    let (f1, piv1, info) = lu(&v1);
    if info != 0 {
        return None;
    }
    let n1 = norm1(&v1);
    let inv_norm = 1.0 / (rcond_from_lu(&f1, n1) * n1);
    if !(ss.v.norm() * inv_norm <= KAPPA_GUARD) {
        return None;
    }
    // gx' = V1'^{-1} V2'
    let mut gxt = ss.v.rows(nspred, nsfwrd).transpose();
    lu_solve(&f1, &piv1, true, &mut gxt);
    // hx' = V1'^{-1} (V1 M)'
    let mut m = ss.s.clone();
    trsm_left(true, false, 1.0, &ss.t, &mut m);
    let mut hxt = mul(&m, true, &v1, true);
    lu_solve(&f1, &piv1, true, &mut hxt);
    Some((gxt.transpose(), hxt.transpose()))
}

/// One-shot version of [`FirstOrderSolver`] (DESIGN.md contract). Prefer
/// building a `FirstOrderSolver` once when solving repeatedly.
pub fn first_order(
    g1: &Mat,
    lli: &[[usize; 3]],
    n_exo: usize,
    ys: &[f64],
    opts: &SolverOptions,
) -> Result<DecisionRule, SolveError> {
    FirstOrderSolver::new(lli, n_exo, *opts).solve(g1, ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// y_t = rho y_{t-1} + b E_t z_{t+1} + e_t ; z_t = phi E_t z_{t+1} + y_t (forward)
    /// Static: s_t = y_t + z_t.
    fn toy(rho: f64, phi: f64) -> (Mat, Vec<[usize; 3]>) {
        // variables: 0 y (lag+current), 1 z (current+lead), 2 s (static)
        let n = 3;
        let lli = vec![[1, 2, 0], [0, 3, 5], [0, 4, 0]];
        let mut g1 = Mat::zeros(3, 3 * n + 1);
        // eq0: y - rho y(-1) - e = 0
        g1[(0, n)] = 1.0;
        g1[(0, 0)] = -rho;
        g1[(0, 3 * n)] = -1.0;
        // eq1: z - phi z(+1) - y = 0
        g1[(1, n + 1)] = 1.0;
        g1[(1, 2 * n + 1)] = -phi;
        g1[(1, n)] = -1.0;
        // eq2: s - y - z = 0
        g1[(2, n + 2)] = 1.0;
        g1[(2, n)] = -1.0;
        g1[(2, n + 1)] = -1.0;
        (g1, lli)
    }

    #[test]
    fn solver_is_send_sync() {
        fn check<T: Send + Sync>() {}
        check::<FirstOrderSolver>();
        check::<DecisionRule>();
        // concurrent solves (the QZ selection criterion is thread-local)
        let (g1, lli) = toy(0.9, 0.5);
        let solver = FirstOrderSolver::new(&lli, 1, SolverOptions::default());
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| assert!(solver.solve(&g1, &[0.0; 3]).is_ok()));
            }
        });
    }

    #[test]
    fn toy_model_closed_form() {
        let (rho, phi) = (0.9, 0.5);
        let (g1, lli) = toy(rho, phi);
        let dr = first_order(&g1, &lli, 1, &[0.0; 3], &SolverOptions::default()).unwrap();
        // order_var = [s, y, z]; z = y / (1 - phi rho) * ... : z_t = c y_t, c = 1/(1 - phi rho)
        assert_eq!(dr.order_var, vec![2, 0, 1]);
        let c = 1.0 / (1.0 - phi * rho);
        let gy = [(1.0 + c) * rho, rho, c * rho];
        let gu = [1.0 + c, 1.0, c];
        for k in 0..3 {
            assert!((dr.ghx[(k, 0)] - gy[k]).abs() < 1e-12, "ghx {k}: {}", dr.ghx[(k, 0)]);
            assert!((dr.ghu[(k, 0)] - gu[k]).abs() < 1e-12, "ghu {k}: {}", dr.ghu[(k, 0)]);
        }
    }

    #[test]
    fn bk_failures() {
        // explosive backward root: no stable equilibrium (info 3)
        let (g1, lli) = toy(1.2, 0.5);
        let err = first_order(&g1, &lli, 1, &[0.0; 3], &SolverOptions::default()).unwrap_err();
        assert_eq!(err.info, 3);
        // root 1.2 counted: (1.2 - 1 - 1.000001)^2
        assert!((err.info2 - (1.2f64 - 2.000001).powi(2)).abs() < 1e-10, "{}", err.info2);
        // forward root inside the unit circle (|1/phi| < 1): indeterminacy (info 4)
        let (g1, lli) = toy(0.5, 2.0);
        let err = first_order(&g1, &lli, 1, &[0.0; 3], &SolverOptions::default()).unwrap_err();
        assert_eq!(err.info, 4);
        assert!((err.info2 - (0.5f64 - 2.000001).powi(2)).abs() < 1e-10, "{}", err.info2);
        // NaN / Inf in the Jacobian
        let (mut g1, lli) = toy(0.5, 0.5);
        g1[(0, 0)] = f64::NAN;
        assert_eq!(first_order(&g1, &lli, 1, &[0.0; 3], &SolverOptions::default()).unwrap_err().info, 8);
        g1[(0, 0)] = f64::INFINITY;
        assert_eq!(first_order(&g1, &lli, 1, &[0.0; 3], &SolverOptions::default()).unwrap_err().info, 10);
    }

    #[test]
    fn purely_backward() {
        // y = 0.5 y(-1) + e ; s = 2 y
        let n = 2;
        let lli = vec![[1, 2, 0], [0, 3, 0]];
        let mut g1 = Mat::zeros(2, 3 * n + 1);
        g1[(0, n)] = 1.0;
        g1[(0, 0)] = -0.5;
        g1[(0, 3 * n)] = -1.0;
        g1[(1, n + 1)] = 1.0;
        g1[(1, n)] = -2.0;
        let dr = first_order(&g1, &lli, 1, &[0.0; 2], &SolverOptions::default()).unwrap();
        assert_eq!(dr.order_var, vec![1, 0]);
        assert!((dr.ghx[(0, 0)] - 1.0).abs() < 1e-14 && (dr.ghx[(1, 0)] - 0.5).abs() < 1e-14);
        assert!((dr.ghu[(0, 0)] - 2.0).abs() < 1e-14 && (dr.ghu[(1, 0)] - 1.0).abs() < 1e-14);
        g1[(0, 0)] = -1.5;
        let err = first_order(&g1, &lli, 1, &[0.0; 2], &SolverOptions::default()).unwrap_err();
        assert_eq!(err.info, 3);
    }
}
