// SPDX-License-Identifier: GPL-3.0-or-later
//! Bayesian VARs with Sims' Minnesota-type prior: port of Dynare's `+bvar` package
//! (`bvar.toolbox`, `bvar.density`, `bvar.forecast`, `bvar.irf`; Sims' `varprior.m`,
//! `rfvar3.m`, `matrictint.m`) and of `rand_inverse_wishart.m` / `rand_matrix_normal.m`.
//!
//! The .mod commands are `bvar_density(OPTIONS) MAXNLAGS;`, `bvar_forecast(OPTIONS) NLAGS;`
//! and `bvar_irf(NLAGS[, 'Cholesky'|'SquareRoot']);`. As in Dynare, the options are those of
//! the global `options_`: every option given to any earlier command (`datafile`, `first_obs`,
//! `nobs`, `presample`, `prefilter`, `noconstant`, `loglinear`, `bvar_prior_*`, `forecast`,
//! `bvar_replic`, `irf`, ...) stays in force ([`BvarOptions::from_commands`]).
//!
//! Random draws come from a [`NormalSource`]; [`ReplaySource`] replays a recorded stream of
//! MATLAB `randn` values, which reproduces Dynare's forecasts and IRFs to rounding
//! (`tests/bvar.rs`, `reference/bvar/dump_bvar.m`).

pub mod commands;

use anyhow::{Result, anyhow, bail};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::estimation::special::gammaln;
use crate::linalg::{Mat, chol_upper, eig_general, sym_eig};

pub use commands::{Session, handles, run_command};

/// The part of Dynare's `options_` used by the BVAR routines (defaults of
/// `default_option_values.m`).
#[derive(Clone, Debug, PartialEq)]
pub struct BvarOptions {
    pub datafile: Option<String>,
    /// `options_.xls_sheet` / `options_.xls_range` (spreadsheet datafiles).
    pub xls: (Option<String>, Option<String>),
    /// 1-based. `None` is Dynare's default NaN: no command has set it yet (an `estimation` or
    /// `calib_smoother` without `first_obs` sets it to 1), and the toolbox fails.
    pub first_obs: Option<usize>,
    /// `None` = up to the end of the data (`set_default_option(options_,'nobs',...)`).
    pub nobs: Option<usize>,
    pub presample: usize,
    pub prefilter: bool,
    pub noconstant: bool,
    pub loglinear: bool,
    pub logdata: bool,
    pub tau: f64,
    pub decay: f64,
    pub lambda: f64,
    pub mu: f64,
    /// `bvar_prior_omega` (MATLAB loops `for i = 1:w`, i.e. floor for positive values).
    pub omega: f64,
    pub flat: bool,
    pub train: usize,
    pub forecast: usize,
    pub replic: usize,
    /// `options_.bvar.conf_sig`.
    pub conf_sig: f64,
    pub irf: usize,
}

impl Default for BvarOptions {
    fn default() -> Self {
        BvarOptions {
            datafile: None,
            xls: (None, None),
            first_obs: None,
            nobs: None,
            presample: 0,
            prefilter: false,
            noconstant: false,
            loglinear: false,
            logdata: false,
            tau: 3.0,
            decay: 0.5,
            lambda: 5.0,
            mu: 2.0,
            omega: 1.0,
            flat: false,
            train: 0,
            forecast: 0,
            replic: 2000,
            conf_sig: 0.6,
            irf: 40,
        }
    }
}

/// Normal-inverse-Wishart distribution of a BVAR (`posterior` / `prior` of `bvar.toolbox`).
#[derive(Clone, Debug)]
pub struct NiW {
    /// Degrees of freedom of the inverse-Wishart part.
    pub df: f64,
    /// Scale matrix of the inverse-Wishart part (ny x ny).
    pub s: Mat,
    /// First factor of the covariance of the matrix-normal part (k x k).
    pub xxi: Mat,
    /// Mean of the matrix-normal part (k x ny): y_t' = [y_{t-1}' ... y_{t-p}' 1] PhiHat.
    pub phi_hat: Mat,
}

/// `forecast_data` of `bvar.toolbox`.
#[derive(Clone, Debug)]
pub struct ForecastData {
    /// The last `nlags` observations of the sample (nlags x ny, oldest first).
    pub initval: Mat,
    /// Observations after the sample, if any (for the RMSE).
    pub realized_val: Option<Mat>,
}

#[derive(Clone, Debug)]
pub struct Toolbox {
    pub ny: usize,
    /// 1 with a constant term, 0 with `prefilter` or `noconstant`.
    pub nx: usize,
    pub posterior: NiW,
    pub prior: NiW,
    pub forecast_data: ForecastData,
}

/// Port of `bvar.toolbox(nlags)`. `dataset` is the whole datafile (rows = periods, columns in
/// `varobs` order), before any transformation.
pub fn toolbox(dataset: &Mat, nlags: usize, o: &BvarOptions) -> Result<Toolbox> {
    let (nrows, ny) = (dataset.nrows(), dataset.ncols());
    // with first_obs = NaN, Dynare's idx = NaN:NaN fails when it indexes the dataset
    let first_obs = o.first_obs.ok_or_else(|| {
        anyhow!("bvar: first_obs is not set (NaN): Array indices must be positive integers or logical values.")
    })?;
    if first_obs == 0 || first_obs > nrows {
        bail!("bvar: first_obs = {first_obs} is outside the data (1..={nrows})");
    }
    let nobs = o.nobs.unwrap_or(nrows + 1 - first_obs);
    if first_obs + nobs - 1 > nrows {
        bail!(
            "Incorrect or missing specification of the number of observations. nobs can be at most {}",
            nrows + 1 - first_obs
        );
    }
    if first_obs + o.presample <= nlags {
        bail!("first_obs+presample should be > nlags (for initializing the VAR)");
    }
    let train = o.train;
    if first_obs + o.presample <= nlags + train {
        bail!("first_obs+presample-train should be > nlags (for initializating the VAR)");
    }
    // 0-based rows of idx = first_obs+presample-train-nlags : first_obs+nobs-1
    let i0 = first_obs + o.presample - train - nlags - 1;
    let i1 = first_obs + nobs - 1; // exclusive
    let mut data = dataset.clone();
    if o.loglinear && !o.logdata {
        data.apply(|x| *x = x.ln());
    }
    if o.prefilter {
        for j in 0..ny {
            let m = (i0..i1).map(|i| data[(i, j)]).sum::<f64>() / (i1 - i0) as f64;
            for i in i0..i1 {
                data[(i, j)] -= m;
            }
        }
    }
    // Variance prior from the initialization lags only:
    // std(dataset(first_obs+presample-nlags : first_obs+presample, :))
    let s0 = first_obs + o.presample - nlags - 1;
    let s1 = first_obs + o.presample; // exclusive
    let sig: Vec<f64> = (0..ny).map(|j| std_dev((s0..s1).map(|i| data[(i, j)]))).collect();
    let nx = if o.prefilter || o.noconstant { 0 } else { 1 };
    let (ydum, xdum, pbreaks) = varprior(ny, nx, nlags, o.tau, o.decay, &sig, o.omega);

    let ydata = data.rows(i0, i1 - i0).into_owned();
    let t = ydata.nrows();
    let stack = |y: &Mat, n: usize| -> (Mat, Mat) {
        let yy = vstack(&y.rows(0, n).into_owned(), &ydum);
        let xx = vstack(&Mat::from_element(n, nx, 1.0), &xdum);
        (yy, xx)
    };
    let breaks_for = |n: usize| -> Vec<usize> {
        let mut b = vec![n];
        b.extend(pbreaks.iter().map(|&p| n + p));
        b
    };
    let flat = if o.flat { (ny + 1) as f64 } else { 0.0 };
    let k = (ny * nlags + nx) as f64;

    let (yy, xx) = stack(&ydata, t);
    let var = rfvar3(&yy, nlags, &xx, &breaks_for(t), o.lambda, o.mu);
    let posterior =
        NiW { df: var.u.nrows() as f64 - k - flat, s: var.u.transpose() * &var.u, xxi: var.xxi, phi_hat: var.b };
    let tp = train + nlags;
    let (yy, xx) = stack(&ydata, tp);
    let varp = rfvar3(&yy, nlags, &xx, &breaks_for(tp), o.lambda, o.mu);
    let prior =
        NiW { df: varp.u.nrows() as f64 - k - flat, s: varp.u.transpose() * &varp.u, xxi: varp.xxi, phi_hat: varp.b };
    if prior.df < ny as f64 {
        bail!(
            "Too few degrees of freedom in the inverse-Wishart part of prior distribution. You should increase training sample size."
        );
    }
    let realized_val =
        (first_obs + nobs <= nrows).then(|| data.rows(first_obs + nobs - 1, nrows + 1 - first_obs - nobs).into_owned());
    let forecast_data = ForecastData { initval: ydata.rows(t - nlags, nlags).into_owned(), realized_val };
    Ok(Toolbox { ny, nx, posterior, prior, forecast_data })
}

fn std_dev(it: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = it.clone().count() as f64;
    let m = it.clone().sum::<f64>() / n;
    (it.map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)).sqrt()
}

fn vstack(a: &Mat, b: &Mat) -> Mat {
    let mut m = Mat::zeros(a.nrows() + b.nrows(), a.ncols().max(b.ncols()));
    m.view_mut((0, 0), (a.nrows(), a.ncols())).copy_from(a);
    m.view_mut((a.nrows(), 0), (b.nrows(), b.ncols())).copy_from(b);
    m
}

/// Port of Sims' `varprior.m` (as in `bvar.toolbox`): dummy observations implementing the
/// Minnesota prior (`tau`, `decay`) and the prior on the covariance (`omega` dummies scaled by
/// `sig`). Returns (ydum, xdum, breaks) with breaks relative to the start of the dummies.
pub fn varprior(
    nv: usize,
    nx: usize,
    lags: usize,
    tight: f64,
    decay: f64,
    sig: &[f64],
    w: f64,
) -> (Mat, Mat, Vec<usize>) {
    let h = lags + 1;
    let mut blocks: Vec<Mat> = Vec::new();
    // Minnesota part: dummy (il, b) for b = 1..nv (slow), il = 1..lags (fast); before the
    // flip, row 1 holds diag(sig) for il = 1 and row il+1 holds il^decay*diag(sig).
    for b in 0..nv {
        for il in 1..=lags {
            let mut d = Mat::zeros(h, nv);
            d[(il, b)] = (il as f64).powf(decay) * sig[b];
            if il == 1 {
                d[(0, b)] = sig[b];
            }
            // flipdim(.., 1) and the overall tightness
            let mut f = Mat::zeros(h, nv);
            for r in 0..h {
                for c in 0..nv {
                    f[(h - 1 - r, c)] = tight * d[(r, c)];
                }
            }
            blocks.push(f);
        }
    }
    let mut breaks: Vec<usize> = (1..=nv * lags).map(|i| h * i).collect();
    let mut lbreak = breaks.last().copied().unwrap_or(0);
    if w > 0.0 {
        let reps = w.floor() as usize;
        for _ in 0..reps {
            for b in 0..nv {
                let mut d = Mat::zeros(h, nv);
                d[(h - 1, b)] = sig[b];
                blocks.push(d);
            }
            breaks.extend((1..=nv).map(|i| h * i + lbreak));
            lbreak = *breaks.last().unwrap();
        }
    }
    let nd = blocks.len();
    let mut ydum = Mat::zeros(h * nd, nv);
    for (d, blk) in blocks.iter().enumerate() {
        ydum.view_mut((d * h, 0), (h, nv)).copy_from(blk);
    }
    let xdum = Mat::zeros(h * nd, nx);
    breaks.pop();
    (ydum, xdum, breaks)
}

/// Output of [`rfvar3`].
pub struct RfVar {
    pub b: Mat,
    pub u: Mat,
    pub xxi: Mat,
}

/// Port of Sims' `rfvar3.m`: OLS of a VAR with breaks (dummy observations) and the
/// co-persistence (`lambda`) and own-persistence (`mu`) dummies, through an SVD of X.
pub fn rfvar3(ydata: &Mat, lags: usize, xdata: &Mat, breaks: &[usize], lambda: f64, mu: f64) -> RfVar {
    let (t, nvar) = (ydata.nrows(), ydata.ncols());
    let nx = xdata.ncols();
    let mut br = vec![0usize];
    br.extend_from_slice(breaks);
    br.push(t);
    // 0-based sample rows
    let mut smpl = Vec::new();
    for nb in 0..br.len() - 1 {
        for r in br[nb] + lags..br[nb + 1] {
            smpl.push(r);
        }
    }
    let ncol = nvar * lags + nx;
    let mut xr: Vec<Vec<f64>> = Vec::with_capacity(smpl.len() + 1 + nvar);
    let mut yr: Vec<Vec<f64>> = Vec::with_capacity(smpl.len() + 1 + nvar);
    for &s in &smpl {
        let mut row = Vec::with_capacity(ncol);
        for l in 1..=lags {
            for v in 0..nvar {
                row.push(ydata[(s - l, v)]);
            }
        }
        for j in 0..nx {
            row.push(xdata[(s, j)]);
        }
        xr.push(row);
        yr.push((0..nvar).map(|v| ydata[(s, v)]).collect());
    }
    if lambda != 0.0 || mu > 0.0 {
        let ybar: Vec<f64> = (0..nvar).map(|v| (0..lags).map(|i| ydata[(i, v)]).sum::<f64>() / lags as f64).collect();
        let xbar: Vec<f64> = (0..nx).map(|j| (0..lags).map(|i| xdata[(i, j)]).sum::<f64>() / lags as f64).collect();
        if lambda != 0.0 {
            let lam = lambda.abs();
            let mut row = Vec::with_capacity(ncol);
            for _ in 0..lags {
                row.extend(ybar.iter().map(|y| lam * y));
            }
            row.extend(xbar.iter().map(|x| if lambda > 0.0 { lam * x } else { 0.0 }));
            xr.push(row);
            yr.push(ybar.iter().map(|y| lam * y).collect());
        }
        if mu > 0.0 {
            for v in 0..nvar {
                let mut row = vec![0.0; ncol];
                for l in 0..lags {
                    row[l * nvar + v] = ybar[v] * mu;
                }
                xr.push(row);
                let mut yrow = vec![0.0; nvar];
                yrow[v] = mu * ybar[v];
                yr.push(yrow);
            }
        }
    }
    let n = xr.len();
    let x = Mat::from_fn(n, ncol, |i, j| xr[i][j]);
    let y = Mat::from_fn(n, nvar, |i, j| yr[i][j]);
    let svd = x.clone().svd(true, true);
    let (vl, vrt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let d = &svd.singular_values;
    // vr .* repmat(di', ...)
    let mut vrd = vrt.transpose();
    for j in 0..vrd.ncols() {
        let di = 1.0 / d[j];
        vrd.column_mut(j).scale_mut(di);
    }
    let b = &vrd * (vl.transpose() * &y);
    let u = &y - &x * &b;
    let xxi = &vrd * vrd.transpose();
    RfVar { b, u, xxi }
}

/// Port of `matrictint.m`: log of the integral of the normal-inverse-Wishart kernel.
pub fn matrictint(s: &Mat, df: f64, xxi: &Mat) -> Result<f64> {
    let k = xxi.nrows() as f64;
    let ny = s.nrows();
    let cx = chol_upper(xxi).ok_or_else(|| anyhow!("singular XXi"))?;
    let cs = chol_upper(s).ok_or_else(|| anyhow!("singular S"))?;
    let eps100 = 100.0 * f64::EPSILON;
    if cx.diagonal().iter().any(|&v| v < eps100) {
        bail!("singular XXi");
    }
    if cs.diagonal().iter().any(|&v| v < eps100) {
        bail!("singular S");
    }
    let nyf = ny as f64;
    let w1 =
        0.5 * k * nyf * (2.0 * std::f64::consts::PI).ln() + nyf * cx.diagonal().iter().map(|v| v.ln()).sum::<f64>();
    if df <= nyf - 1.0 {
        bail!("too few df in ggammaln");
    }
    let lgg: f64 = (0..ny).map(|i| gammaln(0.5 * (df - i as f64))).sum();
    let w2 = -df * cs.diagonal().iter().map(|v| v.ln()).sum::<f64>()
        + 0.5 * df * nyf * 2f64.ln()
        + nyf * (nyf - 1.0) * 0.25 * std::f64::consts::PI.ln()
        + lgg;
    Ok(w1 + w2)
}

/// Log marginal data density of a BVAR (the loop body of `bvar.density`).
pub fn log_marginal_density(tb: &Toolbox) -> Result<f64> {
    let post = matrictint(&tb.posterior.s, tb.posterior.df, &tb.posterior.xxi)?;
    let prior = matrictint(&tb.prior.s, tb.prior.df, &tb.prior.xxi)?;
    let lik_nobs = tb.posterior.df - tb.prior.df;
    Ok(post - prior - 0.5 * tb.ny as f64 * lik_nobs * (2.0 * std::f64::consts::PI).ln())
}

/// Port of `bvar.density(maxnlags)`: the log marginal density for nlags = 1..=maxnlags.
pub fn density(dataset: &Mat, maxnlags: usize, o: &BvarOptions) -> Result<Vec<(Toolbox, f64)>> {
    (1..=maxnlags)
        .map(|p| {
            let tb = toolbox(dataset, p, o)?;
            let d = log_marginal_density(&tb)?;
            Ok((tb, d))
        })
        .collect()
}

// ---- random draws --------------------------------------------------------------------

/// A stream of standard normal draws (MATLAB's `randn`, consumed in column-major order).
pub trait NormalSource {
    fn next_normal(&mut self) -> f64;
}

/// ChaCha8-based standard normals (dynare-rust's default; not MATLAB's stream).
pub struct ChaChaNormals(pub ChaCha8Rng);

impl ChaChaNormals {
    pub fn new(seed: u64) -> Self {
        ChaChaNormals(ChaCha8Rng::seed_from_u64(seed))
    }
}

impl NormalSource for ChaChaNormals {
    fn next_normal(&mut self) -> f64 {
        StandardNormal.sample(&mut self.0)
    }
}

/// Replays a recorded stream (e.g. MATLAB `randn` values); panics when exhausted.
pub struct ReplaySource<'a> {
    pub values: &'a [f64],
    pub pos: usize,
}

impl NormalSource for ReplaySource<'_> {
    fn next_normal(&mut self) -> f64 {
        let v = self.values[self.pos];
        self.pos += 1;
        v
    }
}

fn randn(src: &mut dyn NormalSource, r: usize, c: usize) -> Mat {
    let mut m = Mat::zeros(r, c);
    for j in 0..c {
        for i in 0..r {
            m[(i, j)] = src.next_normal();
        }
    }
    m
}

/// Port of `rand_inverse_wishart(m, v, H_inv_upper_chol)`: X = randn(v, m) * H_inv_upper_chol,
/// G = inv(X'X) through the SVD of X.
pub fn rand_inverse_wishart(m: usize, v: usize, h_inv_upper_chol: &Mat, src: &mut dyn NormalSource) -> Mat {
    let x = randn(src, v, m) * h_inv_upper_chol;
    let svd = x.svd(false, true);
    let vv = svd.v_t.unwrap().transpose();
    let mut vs = vv.clone();
    for j in 0..vs.ncols() {
        let s = svd.singular_values[j];
        vs.column_mut(j).scale_mut(1.0 / (s * s));
    }
    vs * vv.transpose()
}

/// Port of `rand_matrix_normal(n, p, M, Omega_lower_chol, Sigma_lower_chol)`:
/// M + Sigma_lower_chol * reshape(randn(n*p,1), n, p) * Omega_lower_chol'
/// (= reshape(kron(Omega_lc, Sigma_lc) * randn(n*p, 1), n, p)).
pub fn rand_matrix_normal(
    n: usize,
    p: usize,
    m: &Mat,
    omega_lower_chol: &Mat,
    sigma_lower_chol: &Mat,
    src: &mut dyn NormalSource,
) -> Mat {
    let z = randn(src, n, p);
    m + sigma_lower_chol * z * omega_lower_chol.transpose()
}

/// One posterior draw (Sigma, its lower Cholesky factor, Phi) as in `bvar.forecast` / `bvar.irf`.
struct PosteriorSampler {
    s_inv_upper_chol: Mat,
    xxi_lower_chol: Mat,
    df: usize,
    k: usize,
    ny: usize,
}

impl PosteriorSampler {
    fn new(tb: &Toolbox, nlags: usize) -> Result<Self> {
        let s_inv = tb.posterior.s.clone().try_inverse().ok_or_else(|| anyhow!("bvar: posterior S is singular"))?;
        let s_inv_upper_chol = chol_upper(&s_inv).ok_or_else(|| anyhow!("bvar: inv(S) is not positive definite"))?;
        let xxi_lower_chol =
            chol_upper(&tb.posterior.xxi).ok_or_else(|| anyhow!("bvar: XXi is not positive definite"))?.transpose();
        let df = tb.posterior.df;
        if df < tb.ny as f64 || df.fract() != 0.0 {
            bail!("bvar: invalid posterior degrees of freedom {df}");
        }
        Ok(PosteriorSampler { s_inv_upper_chol, xxi_lower_chol, df: df as usize, k: tb.ny * nlags + tb.nx, ny: tb.ny })
    }
    fn draw(&self, tb: &Toolbox, src: &mut dyn NormalSource) -> Result<(Mat, Mat, Mat)> {
        let sigma = rand_inverse_wishart(self.ny, self.df, &self.s_inv_upper_chol, src);
        let sigma_lc =
            chol_upper(&sigma).ok_or_else(|| anyhow!("bvar: drawn Sigma is not positive definite"))?.transpose();
        let phi = rand_matrix_normal(self.k, self.ny, &tb.posterior.phi_hat, &sigma_lc, &self.xxi_lower_chol, src);
        Ok((sigma, sigma_lc, phi))
    }
}

fn is_explosive(phi: &Mat, ny: usize, nlags: usize) -> bool {
    let n = ny * nlags;
    let mut c = Mat::zeros(n, n);
    for i in ny..n {
        c[(i, i - ny)] = 1.0;
    }
    for i in 0..ny {
        for j in 0..n {
            c[(i, j)] = phi[(j, i)];
        }
    }
    eig_general(&c).iter().any(|z| z.norm() > 1.0000000000001)
}

/// One-step VAR prediction from `lags` (nlags x ny, oldest first): [y_{t-1}' ... y_{t-p}' x] Phi.
fn predict(lags: &Mat, phi: &Mat, nx: usize) -> Vec<f64> {
    let (p, ny) = (lags.nrows(), lags.ncols());
    (0..ny)
        .map(|j| {
            let mut s = 0.0;
            for l in 0..p {
                for v in 0..ny {
                    s += lags[(p - 1 - l, v)] * phi[(l * ny + v, j)];
                }
            }
            for c in 0..nx {
                s += phi[(p * ny + c, j)];
            }
            s
        })
        .collect()
}

fn push_lag(lags: &mut Mat, y: &[f64]) {
    let p = lags.nrows();
    for r in 0..p - 1 {
        for v in 0..lags.ncols() {
            lags[(r, v)] = lags[(r + 1, v)];
        }
    }
    for (v, &x) in y.iter().enumerate() {
        lags[(p - 1, v)] = x;
    }
}

/// Summary statistics of one forecast type (`oo_.bvar.forecast.with_shocks` / `no_shock`),
/// each `forecast x ny`.
#[derive(Clone, Debug)]
pub struct ForecastStats {
    pub mean: Mat,
    pub median: Mat,
    pub var: Mat,
    pub hpd_sup: Mat,
    pub hpd_inf: Mat,
}

#[derive(Clone, Debug)]
pub struct ForecastResult {
    pub nlags: usize,
    pub with_shocks: ForecastStats,
    pub no_shock: ForecastStats,
    /// RMSE of the posterior-mean forecast over the realized values after the sample.
    pub rmse: Option<Vec<f64>>,
    /// Number of explosive draws (printed by Dynare).
    pub explosive: usize,
    /// All simulations: `[draw][t][variable]` (Dynare's `simulations.mat`).
    pub sims_no_shock: Vec<Mat>,
    pub sims_with_shocks: Vec<Mat>,
}

fn quantile_index(conf_sig: f64, replic: usize) -> [usize; 3] {
    // round((0.5 + [-c, c, 0]/2) * replic), 1-based
    [-conf_sig, conf_sig, 0.0].map(|c| (((0.5 + c / 2.0) * replic as f64).round() as usize).max(1))
}

fn sample_var(v: &[f64], ddof: f64) -> f64 {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - ddof)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// Port of `bvar.forecast(nlags)`. Like Dynare, draws `bvar_replic + 1` posterior samples
/// (`while d <= options_.bvar_replic`) and takes the bands at the ranks of `bvar_replic`.
pub fn forecast(tb: &Toolbox, nlags: usize, o: &BvarOptions, src: &mut dyn NormalSource) -> Result<ForecastResult> {
    if o.forecast == 0 {
        bail!("bvar.forecast: you must specify \"forecast\" option");
    }
    let (ny, nx, h) = (tb.ny, tb.nx, o.forecast);
    let sampler = PosteriorSampler::new(tb, nlags)?;
    let ndraws = o.replic + 1;
    let mut sims_no_shock = Vec::with_capacity(ndraws);
    let mut sims_with_shocks = Vec::with_capacity(ndraws);
    let mut explosive = 0;
    for _ in 0..ndraws {
        let (_sigma, sigma_lc, phi) = sampler.draw(tb, src)?;
        if is_explosive(&phi, ny, nlags) {
            explosive += 1;
        }
        let mut lags = tb.forecast_data.initval.clone();
        let mut ns = Mat::zeros(h, ny);
        for t in 0..h {
            let y = predict(&lags, &phi, nx);
            push_lag(&mut lags, &y);
            for v in 0..ny {
                ns[(t, v)] = y[v];
            }
        }
        let mut lags = tb.forecast_data.initval.clone();
        let mut ws = Mat::zeros(h, ny);
        for t in 0..h {
            let z = randn(src, ny, 1);
            let shock = &sigma_lc * z;
            let mut y = predict(&lags, &phi, nx);
            for v in 0..ny {
                y[v] += shock[v];
            }
            push_lag(&mut lags, &y);
            for v in 0..ny {
                ws[(t, v)] = y[v];
            }
        }
        sims_no_shock.push(ns);
        sims_with_shocks.push(ws);
    }
    let idx = quantile_index(o.conf_sig, o.replic);
    let stats = |sims: &[Mat], median_by_rank: bool| -> ForecastStats {
        let mut st = ForecastStats {
            mean: Mat::zeros(h, ny),
            median: Mat::zeros(h, ny),
            var: Mat::zeros(h, ny),
            hpd_sup: Mat::zeros(h, ny),
            hpd_inf: Mat::zeros(h, ny),
        };
        for t in 0..h {
            for v in 0..ny {
                let mut x: Vec<f64> = sims.iter().map(|m| m[(t, v)]).collect();
                st.mean[(t, v)] = x.iter().sum::<f64>() / x.len() as f64;
                st.var[(t, v)] = sample_var(&x, 1.0);
                let med = median(&mut x); // sorts x
                st.median[(t, v)] = if median_by_rank { x[idx[2] - 1] } else { med };
                st.hpd_inf[(t, v)] = x[idx[0] - 1];
                st.hpd_sup[(t, v)] = x[idx[1] - 1];
            }
        }
        st
    };
    let with_shocks = stats(&sims_with_shocks, false);
    let no_shock = stats(&sims_no_shock, true);
    let rmse = tb.forecast_data.realized_val.as_ref().map(|real| {
        let mut sq = vec![0.0; ny];
        let mut lags = tb.forecast_data.initval.clone();
        for t in 0..real.nrows() {
            let y = predict(&lags, &tb.posterior.phi_hat, nx);
            push_lag(&mut lags, &y);
            for v in 0..ny {
                sq[v] += (y[v] - real[(t, v)]).powi(2);
            }
        }
        sq.iter().map(|s| (s / real.nrows() as f64).sqrt()).collect()
    });
    Ok(ForecastResult { nlags, with_shocks, no_shock, rmse, explosive, sims_no_shock, sims_with_shocks })
}

/// Identification of the structural shocks in `bvar_irf`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Identification {
    Cholesky,
    SquareRoot,
}

impl Identification {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cholesky" => Ok(Self::Cholesky),
            "squareroot" => Ok(Self::SquareRoot),
            _ => bail!("Unknown identification option. Valid choices are 'Cholesky' and 'SquareRoot'."),
        }
    }
}

/// IRF statistics, `stat[t][(a, b)]` with Dynare's indexing of `sampled_irfs(a, b, t, draw)`:
/// `oo_.bvar.irf.<Stat>.(varobs{a}).(varobs{b})` is `stat[.][(a, b)]`.
#[derive(Clone, Debug)]
pub struct IrfResult {
    pub nlags: usize,
    pub identification: Identification,
    pub mean: Vec<Mat>,
    pub median: Vec<Mat>,
    /// Population variance (`var(.., 1, 4)`).
    pub var: Vec<Mat>,
    pub upper: Vec<Mat>,
    pub lower: Vec<Mat>,
    pub explosive: usize,
}

/// Symmetric matrix square root (`sqrtm` of a symmetric positive definite matrix).
fn sqrtm_sym(a: &Mat) -> Mat {
    let (w, v) = sym_eig(a);
    let mut vs = v.clone();
    for j in 0..vs.ncols() {
        vs.column_mut(j).scale_mut(w[j].max(0.0).sqrt());
    }
    vs * v.transpose()
}

/// Port of `bvar.irf(nlags, identification)`.
pub fn irf(
    tb: &Toolbox,
    nlags: usize,
    ident: Identification,
    o: &BvarOptions,
    src: &mut dyn NormalSource,
) -> Result<IrfResult> {
    let ny = tb.ny;
    let h = o.irf;
    if h == 0 {
        bail!("bvar_irf: irf = 0");
    }
    let sampler = PosteriorSampler::new(tb, nlags)?;
    let nd = o.replic;
    // samples[t][draw] = ny x ny
    let mut samples: Vec<Vec<Mat>> = vec![Vec::with_capacity(nd); h];
    let mut explosive = 0;
    for _ in 0..nd {
        let (sigma, sigma_lc, phi) = sampler.draw(tb, src)?;
        if is_explosive(&phi, ny, nlags) {
            explosive += 1;
        }
        let structural = match ident {
            Identification::Cholesky => sigma_lc,
            Identification::SquareRoot => sqrtm_sym(&sigma),
        };
        let phi_y = phi.rows(0, ny * nlags).into_owned();
        let mut lags = Mat::zeros(ny, ny * nlags);
        lags.columns_mut(0, ny).copy_from(&structural);
        samples[0].push(structural);
        for t in 1..h {
            let r = &lags * &phi_y;
            if nlags > 1 {
                let old = lags.columns(0, ny * (nlags - 1)).into_owned();
                lags.columns_mut(ny, ny * (nlags - 1)).copy_from(&old);
            }
            lags.columns_mut(0, ny).copy_from(&r);
            samples[t].push(r);
        }
    }
    let idx = quantile_index(o.conf_sig, nd);
    let mut res = IrfResult {
        nlags,
        identification: ident,
        mean: Vec::new(),
        median: Vec::new(),
        var: Vec::new(),
        upper: Vec::new(),
        lower: Vec::new(),
        explosive,
    };
    for st in &samples {
        let (mut mean, mut med, mut var, mut up, mut lo) =
            (Mat::zeros(ny, ny), Mat::zeros(ny, ny), Mat::zeros(ny, ny), Mat::zeros(ny, ny), Mat::zeros(ny, ny));
        for a in 0..ny {
            for b in 0..ny {
                let mut x: Vec<f64> = st.iter().map(|m| m[(a, b)]).collect();
                mean[(a, b)] = x.iter().sum::<f64>() / x.len() as f64;
                var[(a, b)] = sample_var(&x, 0.0);
                x.sort_by(|p, q| p.partial_cmp(q).unwrap());
                lo[(a, b)] = x[idx[0] - 1];
                up[(a, b)] = x[idx[1] - 1];
                med[(a, b)] = x[idx[2] - 1];
            }
        }
        res.mean.push(mean);
        res.median.push(med);
        res.var.push(var);
        res.upper.push(up);
        res.lower.push(lo);
    }
    Ok(res)
}
