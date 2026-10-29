// SPDX-License-Identifier: GPL-3.0-or-later
//! Priors: hyperparameters (`set_prior.m` and the `*_specification.m`
//! helpers), log densities (`lpdf*.m`, `priordens.m`), bounds
//! (`prior_bounds.m`, `dynare_estimation_init.m`, `set_mcmc_prior_bounds.m`)
//! and draws (`prior_draw.m`).

use anyhow::{Result, bail};
use rand::Rng;
use rand_distr::{Beta as BetaDist, Distribution, Gamma as GammaDist, StandardNormal};

use super::special::{betainv, betaln, gaminv, gammaln, norminv, wblinv};

/// Prior shapes, with Dynare's `bayestopt_.pshape` codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PriorShape {
    Beta,
    Gamma,
    Normal,
    InvGamma1,
    Uniform,
    InvGamma2,
    Weibull,
}

impl PriorShape {
    /// Dynare's numeric code (1, 2, 3, 4, 5, 6, 8).
    pub fn code(self) -> u8 {
        match self {
            PriorShape::Beta => 1,
            PriorShape::Gamma => 2,
            PriorShape::Normal => 3,
            PriorShape::InvGamma1 => 4,
            PriorShape::Uniform => 5,
            PriorShape::InvGamma2 => 6,
            PriorShape::Weibull => 8,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            1 => PriorShape::Beta,
            2 => PriorShape::Gamma,
            3 => PriorShape::Normal,
            4 => PriorShape::InvGamma1,
            5 => PriorShape::Uniform,
            6 => PriorShape::InvGamma2,
            8 => PriorShape::Weibull,
            _ => return None,
        })
    }

    /// Order in which `priordens.m` accumulates the shapes.
    const ORDER: [PriorShape; 7] = [
        PriorShape::Beta,
        PriorShape::Gamma,
        PriorShape::Normal,
        PriorShape::InvGamma1,
        PriorShape::Uniform,
        PriorShape::InvGamma2,
        PriorShape::Weibull,
    ];
}

/// One prior, with the fields of `bayestopt_` (`p1`..`p7`).
#[derive(Clone, Debug, PartialEq)]
pub struct Prior {
    pub shape: PriorShape,
    /// p1: prior mean.
    pub mean: f64,
    /// p2: prior standard deviation.
    pub sd: f64,
    /// p3: lower bound of the (generalized) support.
    pub p3: f64,
    /// p4: upper bound of the (generalized) support.
    pub p4: f64,
    /// p5: prior mode (NaN for the uniform).
    pub p5: f64,
    /// p6: first hyperparameter (a for beta/gamma, s for inverse gammas, mean for
    /// the normal, lower bound for the uniform, shape for the Weibull).
    pub p6: f64,
    /// p7: second hyperparameter (b for beta/gamma, nu for inverse gammas,
    /// sd for the normal, upper bound for the uniform, scale for the Weibull).
    pub p7: f64,
}

/// Parameter bounds (`bounds.lb`, `bounds.ub`).
#[derive(Clone, Debug, PartialEq)]
pub struct Bounds {
    pub lb: Vec<f64>,
    pub ub: Vec<f64>,
}

impl Bounds {
    pub fn unbounded(n: usize) -> Self {
        Bounds { lb: vec![f64::NEG_INFINITY; n], ub: vec![f64::INFINITY; n] }
    }
    pub fn len(&self) -> usize {
        self.lb.len()
    }
    pub fn is_empty(&self) -> bool {
        self.lb.is_empty()
    }
}

/// Port of `distributions/beta_specification.m`.
pub fn beta_specification(mu: f64, sigma2: f64, lb: f64, ub: f64) -> Result<(f64, f64)> {
    if mu < lb {
        bail!("The prior expectation ({mu}) cannot be smaller than the lower bound of the Beta distribution ({lb})!");
    }
    if mu > ub {
        bail!("The prior expectation ({mu}) cannot be greater than the upper bound of the Beta distribution ({ub})!");
    }
    let len = ub - lb;
    let mu = (mu - lb) / len;
    let sigma2 = sigma2 / (len * len);
    if sigma2 > (1.0 - mu) * mu {
        bail!(
            "Beta prior: given the prior expectation and bounds, the prior std. has to be smaller than {}.",
            ((1.0 - mu) * mu).sqrt()
        );
    }
    let a = (1.0 - mu) * mu * mu / sigma2 - mu;
    let b = a * (1.0 / mu - 1.0);
    Ok((a, b))
}

/// Port of `distributions/gamma_specification.m`.
pub fn gamma_specification(mu: f64, sigma2: f64, lb: f64) -> Result<(f64, f64)> {
    if mu < lb {
        bail!("The prior expectation ({mu}) cannot be smaller than the lower bound of the Gamma distribution ({lb})!");
    }
    if sigma2.is_infinite() {
        bail!("The variance of the Gamma distribution has to be finite!");
    }
    let mu = mu - lb;
    let b = sigma2 / mu;
    let a = mu / b;
    Ok((a, b))
}

/// Port of `distributions/ig1fun.m`.
fn ig1fun(nu: f64, mu2: f64, sigma2: f64) -> f64 {
    (2.0 * mu2).ln() - ((sigma2 + mu2) * (nu - 2.0)).ln() + 2.0 * (gammaln(nu / 2.0) - gammaln((nu - 1.0) / 2.0))
}

/// Port of `distributions/inverse_gamma_specification.m` (`use_fzero_flag = false`,
/// i.e. Dynare's default bisection for type 1). Returns `(s, nu)`.
pub fn inverse_gamma_specification(mu: f64, sigma2: f64, lb: f64, ig_type: u8) -> Result<(f64, f64)> {
    if sigma2.is_nan() || sigma2 <= 0.0 {
        bail!("Second input argument must be a real positive scalar!");
    }
    if mu <= lb {
        bail!(
            "The prior mean ({mu}) must be above the lower bound ({lb}) of the Inverse Gamma (type {ig_type}) prior distribution!"
        );
    }
    let sigma = sigma2.sqrt();
    let mu2 = (mu - lb) * (mu - lb);
    match ig_type {
        2 => {
            let nu = 2.0 * (2.0 + mu2 / sigma2);
            let s = 2.0 * mu * (1.0 + mu2 / sigma2);
            Ok((s, nu))
        }
        1 => {
            if sigma2 < f64::INFINITY {
                let mut nu = (2.0 * (2.0 + mu2 / sigma2)).sqrt();
                let mut nu2 = 2.0 * nu;
                let mut nu1 = 2.0;
                let mut err = ig1fun(nu, mu2, sigma2);
                let mut err2 = ig1fun(nu2, mu2, sigma2);
                if err2 > 0.0 {
                    while nu2 < 1e12 {
                        nu1 = nu2;
                        nu2 *= 2.0;
                        err2 = ig1fun(nu2, mu2, sigma2);
                        if err2 < 0.0 {
                            break;
                        }
                    }
                    if err2 > 0.0 {
                        bail!(
                            "inverse_gamma_specification:: Failed in finding an interval containing a sign change! You should check that the prior variance is not too small compared to the prior mean..."
                        );
                    }
                }
                while (nu2 / nu1 - 1.0).abs() > 1e-14 {
                    if err > 0.0 {
                        nu1 = nu;
                        // Dynare: `if nu < nu2, nu = nu2; else nu = 2*nu; nu2 = nu; end`
                        // (the value of nu is overwritten right below).
                        if nu >= nu2 {
                            nu2 = 2.0 * nu;
                        }
                    } else {
                        nu2 = nu;
                    }
                    nu = (nu1 + nu2) / 2.0;
                    err = ig1fun(nu, mu2, sigma2);
                }
                let s = (sigma2 + mu2) * (nu - 2.0);
                if ((mu - lb).ln() - (s / 2.0).sqrt().ln() - gammaln((nu - 1.0) / 2.0) + gammaln(nu / 2.0)).abs() > 1e-7
                    || (sigma - (s / (nu - 2.0) - (mu - lb) * (mu - lb)).sqrt()).abs() > 1e-7
                {
                    bail!("inverse_gamma_specification:: Failed in solving for the hyperparameters!");
                }
                Ok((s, nu))
            } else {
                Ok((2.0 * mu2 / std::f64::consts::PI, 2.0))
            }
        }
        _ => bail!("inverse_gamma_specification: unknown type"),
    }
}

/// Port of `distributions/weibull_specification.m`. Returns `(shape, scale)`.
///
/// Dynare minimizes the squared moment equation with `fminbnd` on
/// `[1e-9, 100]` and polishes with `fzero`; the equation
/// `gammaln(1+2/k) - 2 gammaln(1+1/k) = log(1 + sigma2/mu^2)` is monotone
/// decreasing in `k`, so a bisection on the same interval finds the same
/// root to machine precision.
pub fn weibull_specification(mu: f64, sigma2: f64, lb: f64) -> Result<(f64, f64)> {
    if mu < lb {
        bail!(
            "The prior expectation ({mu}) cannot be smaller than the lower bound of the Weibull distribution ({lb})!"
        );
    }
    if sigma2.is_infinite() {
        bail!("The variance of the Weibull distribution has to be finite!");
    }
    let mu = mu - lb;
    let target = (1.0 + sigma2 / (mu * mu)).ln();
    let eqn = |k: f64| gammaln(1.0 + 2.0 / k) - 2.0 * gammaln(1.0 + 1.0 / k) - target;
    let (mut lo, mut hi) = (1e-9_f64, 100.0_f64);
    if eqn(lo) < 0.0 || eqn(hi) > 0.0 {
        bail!("weibull_specification: no shape parameter in [1e-9, 100] matches the prior moments");
    }
    for _ in 0..400 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if eqn(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let shape = if eqn(lo).abs() < eqn(hi).abs() { lo } else { hi };
    let scale = mu / gammaln(1.0 + 1.0 / shape).exp();
    Ok((shape, scale))
}

/// Port of `distributions/compute_prior_mode.m` (first mode if bimodal).
fn compute_prior_mode(shape: PriorShape, p6: f64, p7: f64, p3: f64, p4: f64) -> f64 {
    let eps = f64::EPSILON;
    match shape {
        PriorShape::Beta => {
            let (a, b) = (p6, p7);
            let m = if a > 1.0 && b > 1.0 {
                (a - 1.0) / (a + b - 2.0)
            } else if (a < 1.0 && b < 1.0) || (a < 1.0 && b > 1.0 - eps) || ((a - 1.0).abs() < 2.0 * eps && b > 1.0) {
                // a, b < 1: two modes [0; 1], set_prior.m keeps the first
                0.0
            } else if (a > 1.0 && b < 1.0 + eps) || ((a - 1.0).abs() < 2.0 * eps && b < 1.0) {
                1.0
            } else if (a - 1.0).abs() < 2.0 * eps && (b - 1.0).abs() < 2.0 * eps {
                0.5
            } else {
                f64::NAN
            };
            m * (p4 - p3) + p3
        }
        PriorShape::Gamma => (if p6 < 1.0 { 0.0 } else { (p6 - 1.0) * p7 }) + p3,
        PriorShape::Normal => p6,
        PriorShape::InvGamma1 => 1.0 / ((p7 + 1.0) / p6).sqrt() + p3,
        PriorShape::Uniform => f64::NAN,
        PriorShape::InvGamma2 => p6 / (p7 + 2.0) + p3,
        PriorShape::Weibull => (if p6 <= 1.0 { 0.0 } else { p7 * ((p6 - 1.0) / p6).powf(1.0 / p6) }) + p3,
    }
}

impl Prior {
    /// Port of the per-shape blocks of `estimation/set_prior.m`: fills the
    /// default support (p3/p4), the hyperparameters p6/p7 and the mode p5.
    /// `p3`/`p4` are the optional 3rd/4th prior parameters of
    /// `estimated_params` (NaN in Dynare when absent). For the uniform prior,
    /// `mean`/`sd` may be NaN when `p3`/`p4` are given (`uniform_specification.m`).
    pub fn new(shape: PriorShape, mean: f64, sd: f64, p3: Option<f64>, p4: Option<f64>) -> Result<Prior> {
        if sd == 0.0 {
            bail!(
                "you cannot use a point prior in estimation. Either increase the prior standard deviation or fix the parameter completely."
            );
        }
        let (mut mean, mut sd) = (mean, sd);
        let (p3, p4, p6, p7) = match shape {
            PriorShape::Beta => {
                let (p3, p4) = (p3.unwrap_or(0.0), p4.unwrap_or(1.0));
                let (a, b) = beta_specification(mean, sd * sd, p3, p4)?;
                (p3, p4, a, b)
            }
            PriorShape::Gamma => {
                let (p3, p4) = (p3.unwrap_or(0.0), p4.unwrap_or(f64::INFINITY));
                let (a, b) = gamma_specification(mean, sd * sd, p3)?;
                (p3, p4, a, b)
            }
            PriorShape::Normal => (p3.unwrap_or(f64::NEG_INFINITY), p4.unwrap_or(f64::INFINITY), mean, sd),
            PriorShape::InvGamma1 | PriorShape::InvGamma2 => {
                let (p3, p4) = (p3.unwrap_or(0.0), p4.unwrap_or(f64::INFINITY));
                let t = if shape == PriorShape::InvGamma1 { 1 } else { 2 };
                let (s, nu) = inverse_gamma_specification(mean, sd * sd, p3, t)?;
                (p3, p4, s, nu)
            }
            PriorShape::Uniform => {
                // uniform_specification.m
                match (p3.filter(|v| !v.is_nan()), p4.filter(|v| !v.is_nan())) {
                    (Some(a), Some(b)) => {
                        if !mean.is_nan() || !sd.is_nan() {
                            bail!(
                                "uniform_specification: You defined lower and upper bounds. In this case, you need to leave mean and standard deviation empty."
                            );
                        }
                        mean = (a + b) / 2.0;
                        sd = (b - a) / 12f64.sqrt();
                        (a, b, a, b)
                    }
                    _ => {
                        if !mean.is_finite() || !sd.is_finite() {
                            bail!("uniform_specification: Mean or standard deviation is not finite.");
                        }
                        let a = mean - sd * 3f64.sqrt();
                        let b = mean + sd * 3f64.sqrt();
                        (a, b, a, b)
                    }
                }
            }
            PriorShape::Weibull => {
                let (p3, p4) = (p3.unwrap_or(0.0), p4.unwrap_or(f64::INFINITY));
                let (k, l) = weibull_specification(mean, sd * sd, p3)?;
                (p3, p4, k, l)
            }
        };
        let p5 = compute_prior_mode(shape, p6, p7, p3, p4);
        Ok(Prior { shape, mean, sd, p3, p4, p5, p6, p7 })
    }

    /// Log density at `x` (the `lpdf*.m` function `priordens.m` uses for this shape).
    pub fn logpdf(&self, x: f64) -> f64 {
        match self.shape {
            PriorShape::Beta => lpdfgbeta(x, self.p6, self.p7, self.p3, self.p4),
            PriorShape::Gamma => lpdfgam(x - self.p3, self.p6, self.p7),
            PriorShape::Normal => lpdfnorm(x, self.p6, self.p7),
            PriorShape::InvGamma1 => lpdfig1(x - self.p3, self.p6, self.p7),
            PriorShape::Uniform => {
                if x - self.p3 < 0.0 || x - self.p4 > 0.0 {
                    f64::NEG_INFINITY
                } else {
                    (1.0 / (self.p4 - self.p3)).ln()
                }
            }
            PriorShape::InvGamma2 => lpdfig2(x - self.p3, self.p6, self.p7),
            PriorShape::Weibull => lpdfgweibull(x, self.p6, self.p7, self.p3),
        }
    }

    /// Port of `estimation/prior_bounds.m` for this prior.
    pub fn bounds(&self, prior_trunc: f64) -> (f64, f64) {
        assert!((0.0..=1.0).contains(&prior_trunc), "prior_trunc must be in [0, 1]");
        let (p3, p4, p6, p7) = (self.p3, self.p4, self.p6, self.p7);
        let t = prior_trunc;
        match self.shape {
            PriorShape::Beta => {
                if t == 0.0 {
                    (p3, p4)
                } else {
                    (betainv(t, p6, p7) * (p4 - p3) + p3, betainv(1.0 - t, p6, p7) * (p4 - p3) + p3)
                }
            }
            PriorShape::Gamma => {
                if t == 0.0 {
                    (p3, f64::INFINITY)
                } else {
                    (gaminv(t, p6, p7) + p3, gaminv(1.0 - t, p6, p7) + p3)
                }
            }
            PriorShape::Normal => {
                if t == 0.0 {
                    (p3.max(f64::NEG_INFINITY), p4.min(f64::INFINITY))
                } else {
                    (norminv(t, p6, p7).max(p3), norminv(1.0 - t, p6, p7).min(p4))
                }
            }
            PriorShape::InvGamma1 => {
                if t == 0.0 {
                    (p3, f64::INFINITY)
                } else {
                    (
                        1.0 / gaminv(1.0 - t, p7 / 2.0, 2.0 / p6).sqrt() + p3,
                        1.0 / gaminv(t, p7 / 2.0, 2.0 / p6).sqrt() + p3,
                    )
                }
            }
            PriorShape::Uniform => {
                if t == 0.0 {
                    (p6, p7)
                } else {
                    (p6 + (p7 - p6) * t, p7 - (p7 - p6) * t)
                }
            }
            PriorShape::InvGamma2 => {
                if t == 0.0 {
                    (p3, f64::INFINITY)
                } else {
                    (1.0 / gaminv(1.0 - t, p7 / 2.0, 2.0 / p6) + p3, 1.0 / gaminv(t, p7 / 2.0, 2.0 / p6) + p3)
                }
            }
            PriorShape::Weibull => {
                if t == 0.0 {
                    (p3, f64::INFINITY)
                } else {
                    (p3 + wblinv(t, p7, p6), p3 + wblinv(1.0 - t, p7, p6))
                }
            }
        }
    }

    /// One draw from the (untruncated) prior, as the per-shape lines of
    /// `estimation/prior_draw.m` (`gamrnd`, `betarnd`, `randn`, `wblrnd`).
    pub fn draw<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let (p3, p4, p6, p7) = (self.p3, self.p4, self.p6, self.p7);
        match self.shape {
            PriorShape::Uniform => rng.r#gen::<f64>() * (p4 - p3) + p3,
            PriorShape::Normal => {
                let z: f64 = StandardNormal.sample(rng);
                z * p7 + p6
            }
            PriorShape::Gamma => gamrnd(rng, p6, p7) + p3,
            PriorShape::Beta => (p4 - p3) * BetaDist::new(p6, p7).expect("beta prior").sample(rng) + p3,
            PriorShape::InvGamma1 => (1.0 / gamrnd(rng, p7 / 2.0, 2.0 / p6)).sqrt() + p3,
            PriorShape::InvGamma2 => 1.0 / gamrnd(rng, p7 / 2.0, 2.0 / p6) + p3,
            PriorShape::Weibull => {
                // wblrnd(scale = p7, shape = p6)
                let u: f64 = rng.r#gen();
                p7 * (-(1.0 - u).ln()).powf(1.0 / p6) + p3
            }
        }
    }
}

fn gamrnd<R: Rng + ?Sized>(rng: &mut R, a: f64, b: f64) -> f64 {
    GammaDist::new(a, b).expect("gamma prior").sample(rng)
}

/// Port of `estimation/lpdfgbeta.m` (generalized beta on `[aa, bb]`).
pub fn lpdfgbeta(x: f64, a: f64, b: f64, aa: f64, bb: f64) -> f64 {
    if x - aa > 0.0 && x - bb < 0.0 {
        -betaln(a, b) + (a - 1.0) * (x - aa).ln() + (b - 1.0) * (bb - x).ln() - (a + b - 1.0) * (bb - aa).ln()
    } else {
        f64::NEG_INFINITY
    }
}

/// Port of `estimation/lpdfgam.m` (shape `a`, scale `b`).
pub fn lpdfgam(x: f64, a: f64, b: f64) -> f64 {
    if x > 0.0 { -gammaln(a) - a * b.ln() + (a - 1.0) * x.ln() - x / b } else { f64::NEG_INFINITY }
}

/// Port of `estimation/lpdfnorm.m`.
pub fn lpdfnorm(x: f64, a: f64, b: f64) -> f64 {
    let z = (x - a) / b;
    -b.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() - 0.5 * z * z
}

/// Port of `estimation/lpdfig1.m`.
pub fn lpdfig1(x: f64, s: f64, nu: f64) -> f64 {
    if x > 0.0 {
        2f64.ln() - gammaln(0.5 * nu) - 0.5 * nu * (2f64.ln() - s.ln()) - (nu + 1.0) * x.ln() - 0.5 * s / (x * x)
    } else {
        f64::NEG_INFINITY
    }
}

/// Port of `estimation/lpdfig2.m`.
pub fn lpdfig2(x: f64, s: f64, nu: f64) -> f64 {
    if x > 0.0 {
        -gammaln(0.5 * nu) - (0.5 * nu) * (2f64.ln() - s.ln()) - 0.5 * (nu + 2.0) * x.ln() - 0.5 * s / x
    } else {
        f64::NEG_INFINITY
    }
}

/// Port of `distributions/lpdfgweibull.m` (shape `a`, scale `b`, location `c`),
/// including its convention of returning 1 at the origin of an exponential.
pub fn lpdfgweibull(x: f64, a: f64, b: f64, c: f64) -> f64 {
    if x - c < 0.0 {
        return f64::NEG_INFINITY;
    }
    if (a - 1.0).abs() < 1e-12 && x >= c && (x - c) < 1e-12 {
        return 1.0;
    }
    let x0 = x - c;
    a.ln() - a * b.ln() + (a - 1.0) * x0.ln() - (x0 / b).powf(a)
}

/// Port of `estimation/priordens.m`: log prior density, accumulated by shape
/// in Dynare's order (beta, gamma, normal, IG1, uniform, IG2, Weibull) and
/// returning -inf as soon as a shape group is outside its support.
pub fn log_prior_density(x: &[f64], priors: &[Prior]) -> f64 {
    assert_eq!(x.len(), priors.len());
    let mut total = 0.0;
    for shape in PriorShape::ORDER {
        let mut group = 0.0;
        let mut any = false;
        for (xi, p) in x.iter().zip(priors) {
            if p.shape != shape {
                continue;
            }
            any = true;
            if shape == PriorShape::Uniform && (xi - p.p3 < 0.0 || xi - p.p4 > 0.0) {
                return f64::NEG_INFINITY;
            }
            group += p.logpdf(*xi);
        }
        if !any {
            continue;
        }
        total += group;
        if shape != PriorShape::Normal && total.is_infinite() {
            return total;
        }
    }
    total
}

/// Port of `estimation/prior_bounds.m` for all estimated parameters.
pub fn prior_bounds(priors: &[Prior], prior_trunc: f64) -> Bounds {
    let (lb, ub) = priors.iter().map(|p| p.bounds(prior_trunc)).unzip();
    Bounds { lb, ub }
}

/// Bounds used during mode finding, as in `dynare_estimation_init.m`:
/// `prior_bounds(bayestopt_, prior_trunc)` intersected with the explicit
/// bounds of `estimated_params` (`None` = -inf / +inf, Dynare's default).
/// For correlations, pass explicit bounds already clamped to [-1, 1] as
/// `set_prior.m` does.
pub fn estimation_bounds(priors: &[Prior], explicit: &[(Option<f64>, Option<f64>)], prior_trunc: f64) -> Bounds {
    assert_eq!(priors.len(), explicit.len());
    let mut b = prior_bounds(priors, prior_trunc);
    for (i, (lb, ub)) in explicit.iter().enumerate() {
        b.lb[i] = b.lb[i].max(lb.unwrap_or(f64::NEG_INFINITY));
        b.ub[i] = b.ub[i].min(ub.unwrap_or(f64::INFINITY));
    }
    b
}

/// Port of `utilities/estimation/set_mcmc_prior_bounds.m`: the MCMC uses the
/// prior bounds only (explicit `estimated_params` bounds are operational
/// during mode finding only). Errors if the mode is outside them.
pub fn mcmc_bounds(xparam: &[f64], priors: &[Prior], prior_trunc: f64) -> Result<Bounds> {
    let b = prior_bounds(priors, prior_trunc);
    let out: Vec<usize> = (0..xparam.len()).filter(|&i| xparam[i] < b.lb[i] || xparam[i] > b.ub[i]).collect();
    if !out.is_empty() {
        if prior_trunc > 0.0 {
            bail!(
                "Mode value(s) of parameters {out:?} are outside parameter bounds. Potentially, you should set prior_trunc=0!"
            );
        }
        bail!("Mode value(s) of parameters {out:?} are outside parameter bounds!");
    }
    Ok(b)
}

/// Port of `estimation/prior_draw.m`: a joint draw from the priors, each
/// component redrawn until it lies in `bounds` (normally
/// `prior_bounds(priors, prior_trunc)`).
pub fn prior_draw<R: Rng + ?Sized>(priors: &[Prior], bounds: &Bounds, rng: &mut R) -> Vec<f64> {
    let mut out = vec![f64::NAN; priors.len()];
    // Same shape ordering as prior_draw.m (uniform, gaussian, gamma, beta, IG1, IG2, Weibull).
    let order = [
        PriorShape::Uniform,
        PriorShape::Normal,
        PriorShape::Gamma,
        PriorShape::Beta,
        PriorShape::InvGamma1,
        PriorShape::InvGamma2,
        PriorShape::Weibull,
    ];
    for shape in order {
        for (i, p) in priors.iter().enumerate() {
            if p.shape != shape {
                continue;
            }
            loop {
                let v = p.draw(rng);
                if !(v > bounds.ub[i] || v < bounds.lb[i]) {
                    out[i] = v;
                    break;
                }
            }
        }
    }
    out
}

/// Port of `check_prior_stderr_corr_skew.m`: the warnings (without the "Warning: " prefix)
/// when a prior allows negative standard deviations, correlations beyond +-1 or skewness
/// beyond the bound of the skew normal distribution. `kinds` and `priors` are in Dynare's
/// order (shocks, measurement errors, correlations, ...).
pub fn check_prior_stderr_corr_skew(kinds: &[crate::modfile::EstimatedKind], priors: &[Prior]) -> Vec<String> {
    use crate::modfile::EstimatedKind as K;
    let sel = |f: &dyn Fn(&K) -> bool| -> Vec<&Prior> {
        kinds.iter().zip(priors).filter(|(k, _)| f(k)).map(|(_, p)| p).collect()
    };
    let neg_sd = |what: &str| {
        format!(
            "Your prior allows for negative standard deviations for {what}. Due to working with variances, Dynare will be able to continue, but it is recommended to change your prior!"
        )
    };
    let corr = |what: &str| {
        format!(
            "Your prior allows for correlations between {what} larger than +-1 and will not integrate to 1 due to truncation. Please change your prior!"
        )
    };
    let mut out = Vec::new();
    if sel(&|k| matches!(k, K::StdErr(_))).iter().any(|p| p.p3 < 0.0) {
        out.push(neg_sd("structural shocks"));
    }
    if sel(&|k| matches!(k, K::MeStdErr(_))).iter().any(|p| p.p3 < 0.0) {
        out.push(neg_sd("measurement error"));
    }
    if sel(&|k| matches!(k, K::Corr(..))).iter().any(|p| p.p3 < -1.0 || p.p4 > 1.0) {
        out.push(corr("structural shocks"));
    }
    if sel(&|k| matches!(k, K::MeCorr(..))).iter().any(|p| p.p3 < -1.0 || p.p4 > 1.0) {
        out.push(corr("measurement errors"));
    }
    let bound = ((2f64.sqrt() * (std::f64::consts::PI - 4.0)) / (std::f64::consts::PI - 2.0).powf(1.5)).abs();
    if sel(&|k| matches!(k, K::Skew(_))).iter().any(|p| p.p3 < -bound || p.p4 > bound) {
        out.push(format!(
            "Your prior allows for skewness parameters larger than +-{bound:.6}, which is outside the theoretical bound of the skew normal distribution, and thus will not integrate to 1 due to truncation. Please change your prior!"
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_match_dynare_unit_tests() {
        let (a, b) = beta_specification(0.5, 0.05, 0.0, 1.0).unwrap();
        assert!((0.5 - a / (a + b)).abs() < 1e-12);
        assert!((0.05 - a * b / ((a + b).powi(2) * (a + b + 1.0))).abs() < 1e-12);
        let (s, nu) = inverse_gamma_specification(0.5, 0.05, 0.0, 1).unwrap();
        assert!((0.5 - (0.5 * s).sqrt() * (gammaln(0.5 * (nu - 1.0)) - gammaln(0.5 * nu)).exp()).abs() < 1e-12);
        assert!((0.05 - s / (nu - 2.0) + 0.25).abs() < 1e-12);
        let (s, nu) = inverse_gamma_specification(0.5, 0.05, 0.0, 2).unwrap();
        assert!((0.5 - s / (nu - 2.0)).abs() < 1e-12);
        assert!((0.05 - 2.0 * 0.25 / (nu - 4.0)).abs() < 1e-12);
        let (k, l) = weibull_specification(2.0, 0.5, 0.0).unwrap();
        let m = l * gammaln(1.0 + 1.0 / k).exp();
        let v = l * l * (gammaln(1.0 + 2.0 / k).exp() - gammaln(1.0 + 1.0 / k).exp().powi(2));
        assert!((m - 2.0).abs() < 1e-10 && (v - 0.5).abs() < 1e-10);
    }

    #[test]
    fn densities_integrate_to_one() {
        let priors = [
            Prior::new(PriorShape::Beta, 0.4, 0.2, None, None).unwrap(),
            Prior::new(PriorShape::Gamma, 0.4, 0.2, None, None).unwrap(),
            Prior::new(PriorShape::InvGamma1, 0.4, 0.2, None, None).unwrap(),
            Prior::new(PriorShape::InvGamma2, 0.4, 0.2, None, None).unwrap(),
            Prior::new(PriorShape::Weibull, 0.4, 0.2, None, None).unwrap(),
            Prior::new(PriorShape::Beta, 1.0, 0.05, Some(0.0), Some(1.1)).unwrap(),
        ];
        for p in &priors {
            let (lo, hi) = (p.p3.max(-50.0), p.p4.min(60.0));
            let n = 400_000;
            let h = (hi - lo) / n as f64;
            let s: f64 = (0..n).map(|k| p.logpdf(lo + (k as f64 + 0.5) * h).exp()).sum::<f64>() * h;
            assert!((s - 1.0).abs() < 2e-3, "{:?} integrates to {s}", p.shape);
        }
    }
}
