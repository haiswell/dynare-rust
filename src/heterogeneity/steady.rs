// SPDX-License-Identifier: GPL-3.0-or-later
// ---- heterogeneity ----
//! Port of `+heterogeneity/load_steady_state.m` and of the MEX helpers it calls
//! (`find_bracket_linear_weight`, `compute_Phi_tilde_e`): turns a checked steady state into the
//! matrices `oo_.heterogeneity.mat` that the steady-state computation and the sequence-space
//! solution work with.
//!
//! Storage conventions: arrays are column-major like MATLAB; policy and distribution points are
//! ordered with the shocks first (first shock fastest) and then the states (first state
//! fastest). Indices are 0-based except the explicitly marked MATLAB-style ones.

use super::HetDecl;
use super::check::{
    CheckOptions, Indices, NdArray, Sizes, SteadyStateData, SteadyStateInput, check_steady_state_input,
};
use super::model::HetModel;
use crate::linalg::Mat;
use crate::model::Model;
use anyhow::{Result, bail};

/// A sparse matrix in compressed-column form (MATLAB `sparse`).
#[derive(Clone, Debug, PartialEq)]
pub struct Sparse {
    pub nrows: usize,
    pub ncols: usize,
    pub colptr: Vec<usize>,
    pub rowind: Vec<usize>,
    pub vals: Vec<f64>,
}

impl Sparse {
    /// `sparse(i, j, v, m, n)`: duplicates are summed, zeros dropped.
    pub fn from_triplets(nrows: usize, ncols: usize, mut t: Vec<(usize, usize, f64)>) -> Sparse {
        t.sort_by_key(|&(i, j, _)| (j, i));
        let mut colptr = vec![0usize; ncols + 1];
        let (mut rowind, mut vals): (Vec<usize>, Vec<f64>) = (Vec::new(), Vec::new());
        let mut last: Option<(usize, usize)> = None;
        for (i, j, v) in t {
            if last == Some((i, j)) {
                *vals.last_mut().unwrap() += v;
            } else {
                rowind.push(i);
                vals.push(v);
                colptr[j + 1] += 1;
                last = Some((i, j));
            }
        }
        for j in 0..ncols {
            colptr[j + 1] += colptr[j];
        }
        // drop explicit zeros
        let mut s = Sparse { nrows, ncols, colptr, rowind, vals };
        s.drop_zeros();
        s
    }

    fn drop_zeros(&mut self) {
        let mut colptr = vec![0usize; self.ncols + 1];
        let (mut r, mut v) = (Vec::new(), Vec::new());
        for j in 0..self.ncols {
            for p in self.colptr[j]..self.colptr[j + 1] {
                if self.vals[p] != 0.0 {
                    r.push(self.rowind[p]);
                    v.push(self.vals[p]);
                }
            }
            colptr[j + 1] = r.len();
        }
        self.colptr = colptr;
        self.rowind = r;
        self.vals = v;
    }

    pub fn identity(n: usize) -> Sparse {
        Sparse { nrows: n, ncols: n, colptr: (0..=n).collect(), rowind: (0..n).collect(), vals: vec![1.0; n] }
    }

    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    pub fn to_dense(&self) -> Mat {
        let mut m = Mat::zeros(self.nrows, self.ncols);
        for j in 0..self.ncols {
            for p in self.colptr[j]..self.colptr[j + 1] {
                m[(self.rowind[p], j)] = self.vals[p];
            }
        }
        m
    }

    /// `kron(a, b)`.
    pub fn kron(a: &Sparse, b: &Sparse) -> Sparse {
        let mut t = Vec::with_capacity(a.nnz() * b.nnz());
        for ja in 0..a.ncols {
            for pa in a.colptr[ja]..a.colptr[ja + 1] {
                for jb in 0..b.ncols {
                    for pb in b.colptr[jb]..b.colptr[jb + 1] {
                        t.push((a.rowind[pa] * b.nrows + b.rowind[pb], ja * b.ncols + jb, a.vals[pa] * b.vals[pb]));
                    }
                }
            }
        }
        Sparse::from_triplets(a.nrows * b.nrows, a.ncols * b.ncols, t)
    }

    /// `dense * self`.
    pub fn left_mul_dense(&self, d: &Mat) -> Mat {
        let mut out = Mat::zeros(d.nrows(), self.ncols);
        for j in 0..self.ncols {
            for p in self.colptr[j]..self.colptr[j + 1] {
                let (r, v) = (self.rowind[p], self.vals[p]);
                for i in 0..d.nrows() {
                    out[(i, j)] += d[(i, r)] * v;
                }
            }
        }
        out
    }

    /// `self * x`.
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.nrows];
        for j in 0..self.ncols {
            for p in self.colptr[j]..self.colptr[j + 1] {
                out[self.rowind[p]] += self.vals[p] * x[j];
            }
        }
        out
    }
}

/// `[ind, w] = find_bracket_linear_weight(grid, xq)`: 1-based lower bracket index (`ind` and
/// `ind + 1` bracket the query, `ind <= n - 1`) and the weight of the lower node.
pub fn bracket_linear_weight(x: &[f64], xq: f64) -> (usize, f64) {
    let n = x.len();
    if xq < x[0] {
        (1, 1.0)
    } else if xq >= x[n - 1] {
        (n - 1, 0.0)
    } else {
        let (mut lo, mut hi) = (1usize, n);
        while hi - lo > 1 {
            let mid = (hi + lo) / 2;
            if xq >= x[mid - 1] {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo, (x[lo] - xq) / (x[lo] - x[lo - 1]))
    }
}

/// `mat.pol` / `mat.d` of the MATLAB code (the parts shared by both).
#[derive(Clone, Debug, Default)]
pub struct GridMat {
    /// 1-based brackets of the (next-period) states: one column per state, `n_points x n_a`
    /// column-major.
    pub ind: Vec<i32>,
    pub w: Vec<f64>,
    pub inv_h: Vec<f64>,
    pub n_points: usize,
    /// Grid sizes of the states.
    pub dims: Vec<usize>,
    /// The grids of the states (`grids_array`).
    pub grids_array: Vec<Vec<f64>>,
}

/// `oo_.heterogeneity.mat`.
#[derive(Clone, Debug, Default)]
pub struct SteadyMat {
    pub pol: GridMat,
    pub d: GridMat,
    /// `mat.d.Phi` (`N_sp x N_om`): linear interpolation from the policy grids to the
    /// distribution grids.
    pub d_phi: Option<Sparse>,
    /// `mat.pol.Phi` is the identity (`N_sp`), so `x_bar_dash == x_bar` and `L = U = I`,
    /// `P = Q = 1:N_sp` of the MATLAB LU factorisation.
    pub pol_phi_e: Option<Sparse>,
    /// `mat.Mu` (`N_e x N_e`): the Kronecker product of the shock transition matrices.
    pub mu: Mat,
    /// `mat.pol.sm`: `(n_e + n_a) x N_sp`, shocks then states.
    pub sm: Mat,
    /// `mat.pol.x_bar` (`H.endo_nbr x N_sp`, rows of the auxiliary variables last).
    pub x_bar: Mat,
    /// `mat.pol.x_bar_dash` (equal to `x_bar`).
    pub x_bar_dash: Mat,
    /// `mat.d.x_bar` (`H.endo_nbr x N_om`), final load only.
    pub d_x_bar: Option<Mat>,
    /// `mat.d.hist(:)`, final load only.
    pub d_hist: Vec<f64>,
    /// `mat.y` (`3 M_.endo_nbr`) and `mat.x`.
    pub y: Vec<f64>,
    pub x: Vec<f64>,
    /// Aggregate equations' residuals `mat.G` and Jacobian `mat.dG` at the steady state.
    pub g: Vec<f64>,
    pub dg: Option<Mat>,
    /// `mat.F` (`H.endo_nbr x N_sp`) and `mat.dF` (`H.endo_nbr x ncols x N_sp`, column-major).
    pub f: Option<Mat>,
    pub df: Vec<f64>,
    /// `H.endo_nbr` x `ncols` of `dF`.
    pub df_dims: (usize, usize, usize),
    /// `yh` (`3 H.endo_nbr x N_sp`) and `xh` at the steady state.
    pub yh: Mat,
    pub xh: Mat,
    /// Aggregates `Ix = x_bar_dash * Phi * hist` (final load only).
    pub ix: Vec<f64>,
    /// Calibration data (set by `heterogeneity_compute_steady_state`).
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
}

/// `indices` beyond the states/shocks: `Y` (aggregate variables entering the heterogeneous
/// Jacobian), `Ix` (aggregation), `mult`.
#[derive(Clone, Debug, Default)]
pub struct ModelIndices {
    pub base: Indices,
    /// `indices.Y.ind_in_dG.{all,exo,lead,current,lag}` (1-based as in MATLAB).
    pub y_all: Vec<usize>,
    pub y_exo: Vec<usize>,
    pub y_lead: Vec<usize>,
    pub y_current: Vec<usize>,
    pub y_lag: Vec<usize>,
    /// `indices.Y.ind_in_dF`.
    pub y_ind_in_df: Vec<usize>,
    /// `indices.Ix.in_agg`: 0-based position in `M_.endo_names` of the aggregate `k`.
    pub ix_in_agg: Vec<usize>,
    /// `indices.Ix.in_het`: 0-based position in `H_.endo_names`.
    pub ix_in_het: Vec<usize>,
    /// `indices.mult.in_het` (0-based).
    pub mult_in_het: Vec<usize>,
    /// `indices.free_parameters` and `indices.target_equations` (compute only).
    pub free_parameters_ind: Vec<usize>,
    pub free_parameters_names: Vec<String>,
    pub target_equations: Vec<usize>,
}

/// `oo_.heterogeneity` after `heterogeneity_load_steady_state` / `..._compute_steady_state`.
#[derive(Clone, Debug, Default)]
pub struct LoadedSteadyState {
    pub steady_state: SteadyStateData,
    pub sizes: Sizes,
    pub indices: ModelIndices,
    pub mat: SteadyMat,
}

fn compute_model_indices(model: &Model, het: &HetDecl, h: &HetModel, base: Indices) -> ModelIndices {
    let n = 3 * h.endo_nbr() + h.exo_nbr();
    let ncols_y = 3 * model.n_endo() + model.n_exo();
    let mut all: Vec<usize> =
        h.dynamic_g1_sparse_colval.iter().map(|&c| c as usize).filter(|&c| c > n && c <= n + ncols_y).collect();
    all.sort_unstable();
    all.dedup();
    let ind_in_df = all.clone();
    let all: Vec<usize> = all.iter().map(|c| c - n).collect();
    let ne = model.n_endo();
    let mut ix = ModelIndices {
        base,
        y_exo: all.iter().filter(|&&c| c > 3 * ne).map(|c| c - 3 * ne).collect(),
        y_lead: all.iter().filter(|&&c| c > 2 * ne && c <= 3 * ne).map(|c| c - 2 * ne).collect(),
        y_current: all.iter().filter(|&&c| c > ne && c <= 2 * ne).map(|c| c - ne).collect(),
        y_lag: all.iter().filter(|&&c| c <= ne).copied().collect(),
        y_all: all,
        y_ind_in_df: ind_in_df,
        ..Default::default()
    };
    for (k, a) in het.aggregate_aux.iter().enumerate() {
        if het.aggregates[k].op == "sum" {
            ix.ix_in_agg.push(a.endo_index);
            ix.ix_in_het.push(het.aggregates[k].het_endo);
        }
    }
    for a in &h.aux_vars {
        if a.ty == 15 {
            ix.mult_in_het.push(a.endo_index);
        }
    }
    ix
}

/// Everything `load_steady_state` needs from the model file.
pub struct LoadContext<'a> {
    pub model: &'a Model,
    pub het: &'a HetDecl,
    pub params: &'a [f64],
    pub check: &'a CheckOptions,
}

/// Port of `heterogeneity.load_steady_state(M_, options_het, oo_het, steady_state,
/// flag_initial_guess)`. With `flag_initial_guess` the distribution and the residual checks
/// are skipped.
pub fn load_steady_state(
    cx: &LoadContext<'_>,
    input: SteadyStateInput,
    flag_initial_guess: bool,
    verbose: bool,
) -> Result<LoadedSteadyState> {
    let (model, het, params) = (cx.model, cx.het, cx.params);
    let Some(h) = het.model.as_ref() else { bail!("heterogeneity: the model has no heterogeneity dimension") };
    let agg_symbs: Vec<String> = model.endo_names[..model.orig_endo_nbr].to_vec();
    let (mut data, sizes, base) = check_steady_state_input(h, &agg_symbs, cx.check, input, flag_initial_guess)?;
    let indices = compute_model_indices(model, het, h, base);
    let n_a = sizes.n_a;
    let n_e = sizes.n_e;
    let (n_sp, n_om) = (sizes.n_sp, sizes.n_om);
    let n_e_tot = sizes.n_e_total;
    let nh = h.endo_nbr();
    let states = &indices.base.states;
    let mut mat = SteadyMat::default();
    // ---- brackets of the distribution grid in the policy grid; policy of the states
    let pol_sizes: Vec<usize> = states.iter().map(|s| sizes.pol_state(s)).collect();
    let d_sizes: Vec<usize> = states.iter().map(|s| sizes.d_state(s)).collect();
    mat.pol.n_points = n_sp;
    mat.pol.ind = vec![0; n_sp * n_a];
    mat.pol.w = vec![0.0; n_sp * n_a];
    mat.pol.inv_h = vec![0.0; n_sp * n_a];
    mat.d.n_points = sizes.d_n_a;
    mat.d.ind = vec![0; sizes.d_n_a * n_a];
    mat.d.w = vec![0.0; sizes.d_n_a * n_a];
    let mut phi = Sparse::identity(n_e_tot);
    let mut stride_d = 1usize;
    for (i, st) in states.iter().enumerate() {
        let grid = data.pol_grid(st).to_vec();
        let dgrid = data.d_grid(st).to_vec();
        let br: Vec<(usize, f64)> = dgrid.iter().map(|&q| bracket_linear_weight(&grid, q)).collect();
        for p in 0..sizes.d_n_a {
            let (ind, w) = br[(p / stride_d) % d_sizes[i]];
            mat.d.ind[p + i * sizes.d_n_a] = ind as i32;
            mat.d.w[p + i * sizes.d_n_a] = w;
        }
        stride_d *= d_sizes[i];
        // basis matrix: m = pol size, n = d size
        let mut t = Vec::new();
        for (c, &(ind, w)) in br.iter().enumerate() {
            t.push((ind - 1, c, w));
            if w < 1.0 {
                t.push((ind, c, 1.0 - w));
            }
        }
        let basis = Sparse::from_triplets(pol_sizes[i], dgrid.len(), t);
        phi = Sparse::kron(&basis, &phi);
        let pv = data.pol_value(st).ok_or_else(|| anyhow::anyhow!("missing policy for state {st}"))?;
        for p in 0..n_sp {
            let (ind, w) = bracket_linear_weight(&grid, pv.data[p]);
            mat.pol.ind[p + i * n_sp] = ind as i32;
            mat.pol.w[p + i * n_sp] = w;
            mat.pol.inv_h[p + i * n_sp] = 1.0 / (grid[ind] - grid[ind - 1]);
        }
    }
    mat.pol.dims = pol_sizes.clone();
    mat.d.dims = d_sizes.clone();
    mat.d_phi = Some(phi);
    // ---- Mu
    let mut mu = Mat::from_element(1, 1, 1.0);
    for s in h.exo_names.iter().rev() {
        let pi = data.shock_pi(s);
        let mut k = Mat::zeros(mu.nrows() * pi.nrows(), mu.ncols() * pi.ncols());
        for a in 0..mu.nrows() {
            for b in 0..mu.ncols() {
                for c in 0..pi.nrows() {
                    for d in 0..pi.ncols() {
                        k[(a * pi.nrows() + c, b * pi.ncols() + d)] = mu[(a, b)] * pi[(c, d)];
                    }
                }
            }
        }
        mu = k;
    }
    mat.mu = mu;
    // ---- state matrix
    let shock_sizes: Vec<usize> = h.exo_names.iter().map(|s| sizes.shock(s)).collect();
    let mut sm = Mat::zeros(n_e + n_a, n_sp);
    {
        let mut stride = 1usize;
        for (j, s) in h.exo_names.iter().enumerate() {
            let g = data.shock_grid(s);
            for p in 0..n_sp {
                sm[(j, p)] = g[(p / stride) % shock_sizes[j]];
            }
            stride *= shock_sizes[j];
        }
        for (j, s) in states.iter().enumerate() {
            let g = data.pol_grid(s);
            for p in 0..n_sp {
                sm[(n_e + j, p)] = g[(p / stride) % pol_sizes[j]];
            }
            stride *= pol_sizes[j];
        }
    }
    mat.sm = sm;
    mat.pol_phi_e = Some(compute_phi_tilde_e(&mat.pol.ind, &mat.pol.w, &pol_sizes, &mat.mu, n_sp));
    // ---- policies
    let n_orig = h.orig_endo_nbr;
    let mut x_bar = Mat::from_element(n_orig, n_sp, f64::NAN);
    for i in 0..n_orig {
        if let Some(a) = data.pol_value(&h.endo_names[i]) {
            for p in 0..n_sp {
                x_bar[(i, p)] = a.data[p];
            }
        }
    }
    // ---- aggregates
    let mut y = vec![f64::NAN; model.n_endo()];
    for (i, nm) in agg_symbs.iter().enumerate() {
        y[i] = data.agg(nm).unwrap();
    }
    mat.x = vec![0.0; model.n_exo()];
    let n_ix = het.aggregate_aux.len();
    let mut yagg = vec![f64::NAN; n_ix];
    het.set_auxiliary_variables(model, &mut y, &mat.x, params, &yagg);
    let mut y3 = Vec::with_capacity(3 * y.len());
    for _ in 0..3 {
        y3.extend_from_slice(&y);
    }
    mat.y = y3;
    let mut yh = Mat::from_element(3 * nh, n_sp, f64::NAN);
    let mut xh = Mat::from_element(h.exo_nbr(), n_sp, f64::NAN);
    for (k, &sv) in h.state_var.iter().enumerate() {
        for p in 0..n_sp {
            yh[(sv, p)] = mat.sm[(n_e + k, p)];
        }
    }
    for i in 0..n_orig {
        for p in 0..n_sp {
            yh[(nh + i, p)] = x_bar[(i, p)];
        }
    }
    let phi_e = mat.pol_phi_e.as_ref().unwrap();
    let nxt = phi_e.left_mul_dense(&x_bar);
    for i in 0..n_orig {
        for p in 0..n_sp {
            yh[(2 * nh + i, p)] = nxt[(i, p)];
        }
    }
    for j in 0..h.exo_nbr() {
        for p in 0..n_sp {
            xh[(j, p)] = mat.sm[(j, p)];
        }
    }
    let mut x_bar_full = x_bar;
    if h.set_auxiliary_variables {
        let pol_dims = data.pol_value(&h.endo_names[0]).unwrap().dims.clone();
        let mut ws = h.workspace(params);
        ws.y_agg_mut().copy_from_slice(&mat.y);
        ws.xf[h.exo_nbr()..].copy_from_slice(&mat.x);
        for (lv, vars) in h.het_aux_levels.iter().enumerate() {
            for p in 0..n_sp {
                for r in 0..3 * nh {
                    ws.yf[r] = yh[(r, p)];
                }
                for j in 0..h.exo_nbr() {
                    ws.xf[j] = xh[(j, p)];
                }
                h.set_auxiliary_variables(&mut ws, lv);
                for r in 0..3 * nh {
                    yh[(r, p)] = ws.yf[r];
                }
            }
            for &idx in vars {
                let vals: Vec<f64> = (0..n_sp).map(|p| yh[(nh + idx, p)]).collect();
                data.set_pol_value(&h.endo_names[idx], NdArray::new(pol_dims.clone(), vals));
            }
            // x_bar_dash = x_bar for the identity basis: the next-period value of the level
            let rows = Mat::from_fn(vars.len(), n_sp, |r, p| yh[(nh + vars[r], p)]);
            let nx = phi_e.left_mul_dense(&rows);
            for (r, &idx) in vars.iter().enumerate() {
                for p in 0..n_sp {
                    yh[(2 * nh + idx, p)] = nx[(r, p)];
                }
            }
        }
        let n_aux = nh - n_orig;
        let mut aux_rows = Mat::from_element(n_aux, n_sp, f64::NAN);
        for r in 0..n_aux {
            if let Some(a) = data.pol_value(&h.endo_names[n_orig + r]) {
                for p in 0..n_sp {
                    aux_rows[(r, p)] = a.data[p];
                }
            }
        }
        x_bar_full =
            Mat::from_fn(nh, n_sp, |r, p| if r < n_orig { x_bar_full[(r, p)] } else { aux_rows[(r - n_orig, p)] });
    }
    mat.x_bar_dash = x_bar_full.clone();
    mat.x_bar = x_bar_full;
    mat.pol.grids_array = states.iter().map(|s| data.pol_grid(s).to_vec()).collect();
    mat.d.grids_array = states.iter().map(|s| data.d_grid(s).to_vec()).collect();
    mat.yh = yh;
    mat.xh = xh;
    let (lo, hi) = h.complementarity_bounds(params);
    mat.lower_bounds = lo;
    mat.upper_bounds = hi;
    if flag_initial_guess {
        return Ok(LoadedSteadyState { steady_state: data, sizes, indices, mat });
    }
    // ---- distribution
    let phi = mat.d_phi.as_ref().unwrap();
    let d_x_bar = phi.left_mul_dense(&mat.x_bar_dash);
    mat.d.n_points = n_om;
    mat.d.ind = vec![0; n_om * n_a];
    mat.d.w = vec![0.0; n_om * n_a];
    mat.d.inv_h = vec![0.0; n_om * n_a];
    for (i, st) in states.iter().enumerate() {
        let grid = data.d_grid(st);
        for p in 0..n_om {
            let (ind, w) = bracket_linear_weight(grid, d_x_bar[(h.state_var[i], p)]);
            mat.d.ind[p + i * n_om] = ind as i32;
            mat.d.w[p + i * n_om] = w;
            mat.d.inv_h[p + i * n_om] = 1.0 / (grid[ind] - grid[ind - 1]);
        }
    }
    mat.d_hist = data.d_hist.as_ref().unwrap().data.clone();
    mat.ix = phi.mul_vec_dense_rows(&mat.x_bar_dash, &mat.d_hist);
    mat.d_x_bar = Some(d_x_bar);
    for k in 0..n_ix {
        yagg[k] = mat.ix[indices.ix_in_het[k]];
    }
    // y with the aggregates
    let mid = y3_middle(&mat.y, model.n_endo());
    let mut y = mid;
    het.set_auxiliary_variables(model, &mut y, &mat.x, params, &yagg);
    let mut y3 = Vec::with_capacity(3 * y.len());
    for _ in 0..3 {
        y3.extend_from_slice(&y);
    }
    mat.y = y3;
    if model.orig_endo_nbr > 0 {
        let mut g = vec![0.0; model.n_equations()];
        het.agg_residuals(model, &mat.y, &mat.x, params, &yagg, &mut g);
        mat.g = g;
        mat.dg = Some(het.agg_jacobian(model, &y, &mat.x, params));
    }
    // heterogeneous residuals and Jacobian at each grid point
    let ncols = h.jacobian_ncols();
    let mut f = Mat::zeros(nh, n_sp);
    mat.df = vec![0.0; nh * ncols * n_sp];
    mat.df_dims = (nh, ncols, n_sp);
    let mut ws = h.workspace(params);
    ws.y_agg_mut().copy_from_slice(&mat.y);
    ws.xf[h.exo_nbr()..].copy_from_slice(&mat.x);
    let mut r = vec![0.0; nh];
    for p in 0..n_sp {
        for k in 0..3 * nh {
            ws.yf[k] = mat.yh[(k, p)];
        }
        for j in 0..h.exo_nbr() {
            ws.xf[j] = mat.xh[(j, p)];
        }
        h.resid(&mut ws, &mut r);
        for i in 0..nh {
            f[(i, p)] = r[i];
        }
        let d = h.jacobian_dense(&mut ws);
        for c in 0..ncols {
            for i in 0..nh {
                mat.df[i + nh * c + nh * ncols * p] = d[(i, c)];
            }
        }
    }
    mat.f = Some(f);
    let out = LoadedSteadyState { steady_state: data, sizes, indices, mat };
    if verbose {
        display_steady_state(model, h, &out);
        display_residuals(model, h, &out);
    }
    if model.orig_endo_nbr > 0 {
        let norm = out.mat.g.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        if norm > cx.check.tolf {
            bail!(
                "heterogeneity_load_steady_state: residuals too large on aggregate equations. Consider increasing the value of \"tolf\" option if you nevertheless want to use the provided values."
            );
        }
    }
    Ok(out)
}

fn y3_middle(y3: &[f64], n: usize) -> Vec<f64> {
    y3[n..2 * n].to_vec()
}

impl Sparse {
    /// `x_bar_dash * Phi * hist` with `self = Phi`: the aggregates of every row of `x_bar_dash`.
    pub fn mul_vec_dense_rows(&self, x_bar_dash: &Mat, hist: &[f64]) -> Vec<f64> {
        let v = self.mul_vec(hist);
        (0..x_bar_dash.nrows()).map(|i| (0..x_bar_dash.ncols()).map(|k| x_bar_dash[(i, k)] * v[k]).sum()).collect()
    }
}

/// `compute_Phi_tilde_e`: the sparse `N_sp x N_sp` matrix such that column `j` holds the
/// multilinear interpolation weights, over the corners of the bracket of the next-period state
/// chosen at point `j` (rows `(corner - 1) * N_e + e'`), times the transition probability
/// `Mu(e_j, e')`.
fn compute_phi_tilde_e(ind: &[i32], w: &[f64], dims: &[usize], mu: &Mat, n_sp: usize) -> Sparse {
    let n = dims.len();
    let n_e = mu.nrows();
    let mut strides = vec![1usize; n];
    for k in 1..n {
        strides[k] = strides[k - 1] * dims[k - 1];
    }
    let mut t = Vec::new();
    for j in 0..n_sp {
        let e = j % n_e;
        let mut base = 0usize;
        for k in 0..n {
            base += strides[k] * (ind[j + k * n_sp] as usize - 1);
        }
        for corner in 0..(1usize << n) {
            let mut wt = 1.0;
            let mut lin = base;
            for k in 0..n {
                let wk = w[j + k * n_sp];
                if corner >> k & 1 == 1 {
                    wt *= 1.0 - wk;
                    lin += strides[k];
                } else {
                    wt *= wk;
                }
            }
            if wt == 0.0 {
                continue;
            }
            for k in 0..n_e {
                t.push((lin * n_e + k, j, wt * mu[(e, k)]));
            }
        }
    }
    Sparse::from_triplets(n_sp, n_sp, t)
}

/// `display_steady_state` (aggregate values and the moments of the states).
pub fn display_steady_state(model: &Model, h: &HetModel, s: &LoadedSteadyState) {
    print!("{}", steady_state_text(model, h, s));
}

/// The text `display_steady_state` prints.
pub fn steady_state_text(model: &Model, h: &HetModel, s: &LoadedSteadyState) -> String {
    use std::fmt::Write as _;
    let mut o = String::new();
    writeln!(o, "\nSTEADY STATE").unwrap();
    if model.orig_endo_nbr > 0 {
        writeln!(o, "\nAggregate variables:").unwrap();
        // char(M_.endo_names) pads every name to the longest one (auxiliary variables included)
        let w = model.endo_names.iter().map(|n| n.chars().count()).max().unwrap_or(0);
        for i in 0..model.orig_endo_nbr {
            writeln!(
                o,
                "{:<w$} \t\t {}",
                model.endo_names[i],
                fmt_g(s.steady_state.agg(&model.endo_names[i]).unwrap())
            )
            .unwrap();
        }
    }
    let mut var_names = Vec::new();
    let mut moments: Vec<(f64, f64)> = Vec::new();
    if let Some(hist) = &s.steady_state.d_hist {
        let order = &s.steady_state.d_order;
        for (i, name) in order.iter().enumerate() {
            if !h.endo_names.contains(name) {
                continue;
            }
            let n = hist.dims.get(i).copied().unwrap_or(1);
            let mut marg = vec![0.0; n];
            let stride: usize = hist.dims[..i].iter().product();
            for (p, v) in hist.data.iter().enumerate() {
                marg[(p / stride) % n] += v;
            }
            let g = s.steady_state.d_grid(name);
            let mean: f64 = marg.iter().zip(g).map(|(m, x)| m * x).sum();
            let sd = marg.iter().zip(g).map(|(m, x)| m * (x - mean).powi(2)).sum::<f64>().sqrt();
            var_names.push(name.clone());
            moments.push((mean, sd));
        }
    }
    let vals = crate::linalg::Mat::from_fn(moments.len(), 2, |i, j| if j == 0 { moments[i].0 } else { moments[i].1 });
    let headers = ["".to_string(), "Mean".to_string(), "Std. dev.".to_string()];
    o += &crate::analysis::table::dyntable(
        "Heterogeneous state variables (unconditional moments):",
        &headers,
        &var_names,
        &vals,
        0,
        0,
        4,
    );
    o
}

/// `display_residuals`.
pub fn display_residuals(model: &Model, h: &HetModel, s: &LoadedSteadyState) {
    print!("{}", residuals_text(model, h, s));
}

/// The text `display_residuals` prints.
pub fn residuals_text(model: &Model, h: &HetModel, s: &LoadedSteadyState) -> String {
    use std::fmt::Write as _;
    let mut o = String::new();
    writeln!(o, "\n\nRESIDUALS (at the steady state)\n").unwrap();
    let tag = |tags: &[Vec<(String, String)>], i: usize| {
        tags.get(i)
            .and_then(|t| t.iter().find(|(k, _)| k == "name"))
            .map(|(_, v)| format!(" ({v})"))
            .unwrap_or_default()
    };
    // `char(headers)` pads the headers to the longest one
    let headers = |n: usize, tags: &[Vec<(String, String)>]| -> Vec<String> {
        let hs: Vec<String> = (0..n).map(|i| format!("Equation {}{}", i + 1, tag(tags, i))).collect();
        let w = hs.iter().map(|x| x.chars().count()).max().unwrap_or(0);
        hs.into_iter().map(|x| format!("{x:<w$}")).collect()
    };
    if model.orig_endo_nbr > 0 {
        writeln!(o, "Aggregate equations:").unwrap();
        for (i, hd) in headers(model.orig_endo_nbr, &model.equation_tags).iter().enumerate() {
            writeln!(o, "{hd}\t{}", fmt_g(s.mat.g[i])).unwrap();
        }
        writeln!(o).unwrap();
    }
    writeln!(o, "Heterogeneous equations (infinity norm over the grid):").unwrap();
    if let Some(f) = &s.mat.f {
        for (i, hd) in headers(h.orig_endo_nbr, &h.equations_tags).iter().enumerate() {
            let m = (0..f.ncols()).fold(0.0f64, |a, p| a.max(f[(i, p)].abs()));
            writeln!(o, "{hd}\t{}", fmt_g(m)).unwrap();
        }
    }
    o
}

/// C `%g` (6 significant digits).
pub fn fmt_g(x: f64) -> String {
    crate::analysis::table::fmt_g(x, 0, 6)
}
