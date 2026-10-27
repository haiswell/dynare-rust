// SPDX-License-Identifier: GPL-3.0-or-later
//! PAC models (`pac_model`, `pac_expectation`, `pac_target_info`, `pac.initialize`,
//! `pac.update.expectation`, `pac.mce.parameters`): the numerical part of Dynare's `+pac/`
//! and `pac-tools/` (`hVectors.m`, `buildGmatrixWithAlphaAndBeta.m`, `a2alpha.m`,
//! `+pac/+update/parameters.m`, `+pac/+mce/parameters.m`).
//!
//! The parser (`modfile/parser_pac.rs`) does what the preprocessor does for a PAC equation: it
//! decomposes the equation (error correction term, autoregressive terms), declares the
//! parameters `h_<pac>_*`, `<pac>_pac_growth_neutrality_correction`, `mce_alpha_<pac>_*` and
//! the auxiliary variables (`pac_expectation_<pac>`, the component and target
//! variables of `pac_target_info`, `mce_Z1_<pac>`) with their defining equations, and keeps the
//! result in a [`PacModel`] (Dynare's `M_.pac.<name>`).

use super::{Companion, Discount, VarModels};
use crate::linalg::{Mat, solve};
use crate::modfile::Ast;
use anyhow::{Result, anyhow, bail};

/// One component of a composite target (`M_.pac.<name>.components(i)`).
#[derive(Clone, Debug)]
pub struct PacComponent {
    /// The endogenous variable of the auxiliary model.
    pub endo: String,
    /// `ll`, `dd` or `dl`.
    pub kind: String,
    /// Weight in the equation of the target (`coeff_str`).
    pub coef: Ast,
    /// `growth` of a non-stationary component.
    pub growth: Option<Ast>,
    /// The auxiliary variable of its PAC expectation (`auxname`).
    pub auxname: String,
    pub h_params: Vec<String>,
    pub growth_param: Option<String>,
}

/// `M_.pac.<name>`.
#[derive(Clone, Debug)]
pub struct PacModel {
    pub name: String,
    /// Tag (`name=`) of the PAC equation, empty if untagged.
    pub eq_name: String,
    /// Auxiliary model (None: model consistent expectations).
    pub aux_model: Option<String>,
    /// `"var"` / `"trend_component"`.
    pub aux_type: String,
    pub discount: Discount,
    pub growth: Option<Ast>,
    /// `kind` of `pac_model` (empty: inferred).
    pub kind: String,
    /// The variable of the left hand side `diff(lhs_var)`.
    pub lhs_var: String,
    pub ec_param: String,
    /// Parameters of `diff(lhs_var(-k))`, k = 1..max_lag.
    pub ar_params: Vec<String>,
    /// The target of the error correction term (the nonstationary target variable of a
    /// composite target).
    pub target: String,
    /// Composite target: the variable of `pac_target_info`'s `target`.
    pub target_var: String,
    pub components: Vec<PacComponent>,
    /// Auxiliary variable of the expectation (no components).
    pub aux_var: String,
    pub h_params: Vec<String>,
    pub growth_param: Option<String>,
    pub mce_alpha: Vec<String>,
    pub mce_z: String,
    /// `auxname_target_nonstationary`.
    pub vns: String,
    /// The PAC equation with the auxiliary variables instead of the operators.
    pub eq_lhs: Ast,
    pub eq_rhs: Ast,
    /// Definitions of the auxiliary variables: `(name, expression)`.
    pub defs: Vec<(String, Ast)>,
    /// `pac.initialize` was run (`M_.pac.<name>.model_consistent_expectations` is set).
    pub initialized: bool,
}

impl PacModel {
    pub fn max_lag(&self) -> usize {
        self.ar_params.len()
    }

    pub fn is_mce(&self) -> bool {
        self.aux_model.is_none()
    }
}

// ---------------------------------------------------------------- numerics

/// `a2alpha.m`. With a single element (a PAC equation without autoregressive terms),
/// `alpha(1) = a(1)-a(2)-1` reads past the end of `a` and MATLAB fails.
fn a2alpha(a: &[f64]) -> Result<Vec<f64>> {
    let m = a.len();
    if m < 2 {
        bail!("Index exceeds the number of array elements. Index must not exceed {m}.");
    }
    let mut al = vec![0.0; m];
    al[m - 1] = a[m - 1];
    for i in 1..m - 1 {
        al[i] = a[i] - a[i + 1];
    }
    al[0] = a[0] - a[1] - 1.0;
    Ok(al)
}

/// `buildGmatrixWithAlphaAndBeta.m` for `params = [a_1..a_m, beta]`: `(G, alpha, beta)` with
/// `alpha` flipped as in the MATLAB function.
fn g_matrix(params: &[f64]) -> Result<(Mat, Vec<f64>, f64)> {
    if params.len() < 2 {
        bail!("Input argument has to be a vector with at least two elements. The last element is the discount factor.");
    }
    let m = params.len() - 1;
    let mut alpha = a2alpha(&params[..m])?;
    alpha.reverse();
    let beta = params[m];
    if beta < f64::EPSILON || beta > 1.0 - f64::EPSILON {
        bail!("beta has to be a discount factor!");
    }
    let mut g = Mat::zeros(m, m);
    for i in 0..m.saturating_sub(1) {
        g[(i, i + 1)] = 1.0;
    }
    for j in 0..m {
        // flip(cumprod(beta*ones(1,m)))(j) = beta^(m-j)
        g[(m - 1, j)] = -alpha[j] * beta.powi((m - j) as i32);
    }
    Ok((g, alpha, beta))
}

/// `polyval([alpha; 1], x)`.
fn polyval1(alpha: &[f64], x: f64) -> f64 {
    alpha.iter().fold(0.0, |acc, &a| acc * x + a) * x + 1.0
}

fn kron(a: &Mat, b: &Mat) -> Mat {
    let mut k = Mat::zeros(a.nrows() * b.nrows(), a.ncols() * b.ncols());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            k.view_mut((i * b.nrows(), j * b.ncols()), (b.nrows(), b.ncols())).copy_from(&(b * a[(i, j)]));
        }
    }
    k
}

fn unit(n: usize, idx: usize) -> Mat {
    let mut e = Mat::zeros(n, 1);
    e[(idx, 0)] = 1.0;
    e
}

/// `hVectors.m`: `params = [ec, ar_1..ar_{p}, beta]`, `h_comp` the companion matrix, `id` the
/// (1-based) position of the target in the companion state. Returns `h` and the long run
/// coefficient of the growth neutrality correction (NaN for `ll`).
pub fn h_vectors(params: &[f64], h_comp: &Mat, kind: &str, id: usize) -> Result<(Vec<f64>, f64)> {
    let (g, alpha, beta) = g_matrix(params)?;
    let a1 = polyval1(&alpha, 1.0);
    let ab = polyval1(&alpha, beta);
    let m = alpha.len();
    let n = h_comp.nrows();
    let ht = h_comp.transpose();
    let tmp = Mat::identity(n * m, n * m) - kron(&g, &ht);
    let rhs = kron(&unit(m, m - 1), &unit(n, id - 1));
    let y = solve(&tmp, &rhs).ok_or_else(|| anyhow!("PAC: singular matrix in hVectors"))?;
    let em = unit(m, m - 1);
    let img = Mat::identity(m, m) - &g;
    let h = match kind {
        "ll" => (kron(&em, h_comp).transpose() * &y) * (a1 * ab),
        "dd" | "dl" => {
            // iota(m,m)' * inv(I-G): the row vector (inv(I-G)' e_m)'.
            let r =
                solve(&img.transpose(), &em).ok_or_else(|| anyhow!("PAC: singular matrix in hVectors"))?.transpose();
            let w = if kind == "dd" { ht.clone() } else { &ht - Mat::identity(n, n) };
            (kron(&r, &w) * &y) * (a1 * ab)
        }
        _ => bail!("Unknown kind value in PAC model."),
    };
    let lrcp = if kind == "ll" {
        f64::NAN
    } else {
        let s1 = solve(&img, &em).ok_or_else(|| anyhow!("PAC: singular matrix in hVectors"))?;
        let s2 = solve(&img, &s1).ok_or_else(|| anyhow!("PAC: singular matrix in hVectors"))?;
        let d = a1 * ab * (em.transpose() * s2)[(0, 0)];
        let sum: f64 = params[1..params.len() - 1].iter().sum();
        1.0 - sum - d
    };
    Ok((h.iter().copied().collect(), lrcp))
}

fn pidx(names: &[String], n: &str) -> Result<usize> {
    names.iter().position(|x| x == n).ok_or_else(|| anyhow!("unknown parameter {n}"))
}

impl PacModel {
    fn pac_values(&self, names: &[String], params: &[f64]) -> Result<Vec<f64>> {
        let mut v = vec![params[pidx(names, &self.ec_param)?]];
        for a in &self.ar_params {
            v.push(params[pidx(names, a)?]);
        }
        Ok(v)
    }

    fn beta(&self, names: &[String], params: &[f64]) -> Result<f64> {
        Ok(match &self.discount {
            Discount::Value(v) => *v,
            Discount::Param(n) => params[pidx(names, n)?],
        })
    }

    /// `pac.update.parameters`: new values of the `h` parameters and of the growth
    /// neutrality correction parameters.
    pub fn update_values(&self, vms: &VarModels, names: &[String], params: &[f64]) -> Result<Vec<(String, f64)>> {
        let Some(aux) = &self.aux_model else {
            bail!("This function cannot be used with Model Consistent Expectation. Try pac.mce.parameters instead.")
        };
        let comp: &Companion = vms
            .companion(aux)
            .filter(|c| !c.matrix.iter().any(|x| x.is_nan()))
            .ok_or_else(|| anyhow!("Auxiliary model {aux} has to be estimated first."))?;
        let is_var = self.aux_type == "var";
        if !self.components.is_empty() && !is_var {
            bail!("Composite PAC target not allowed with trend component model.");
        }
        let mut pv = self.pac_values(names, params)?;
        pv.push(self.beta(names, params)?);
        let off = (is_var && comp.is_constant) as usize;
        let find = |v: &str| -> Result<usize> {
            comp.variables.iter().position(|x| x == v).map(|j| j + 1 + off).ok_or_else(|| {
                anyhow!("Cannot find the trend variable {v} in the auxiliary VAR / Trend component model.")
            })
        };
        // (id, kind, h param names, growth param)
        let mut items: Vec<(usize, String, &Vec<String>, Option<&String>)> = Vec::new();
        if self.components.is_empty() {
            let id = find(&self.target)?;
            let kind = if !self.kind.is_empty() {
                self.kind.clone()
            } else if is_var {
                let j = id - 1 - off;
                if vms.var(aux).is_some_and(|m| m.eqs[j].lhs_diff) { "dd".into() } else { "ll".into() }
            } else {
                "dd".into()
            };
            items.push((id, kind, &self.h_params, self.growth_param.as_ref()));
        } else {
            for c in &self.components {
                if c.kind.is_empty() {
                    bail!("kind declaration is mandatory for each component in pac_target_info.");
                }
                items.push((find(&c.endo)?, c.kind.clone(), &c.h_params, c.growth_param.as_ref()));
            }
        }
        let mut out = Vec::new();
        for (id, kind, hp, gp) in items {
            let growth_flag = self.growth.is_some() || self.components.iter().any(|c| c.growth.is_some());
            let (h, lrcp) = h_vectors(&pv, &comp.matrix, &kind, id)?;
            if is_var && !comp.is_constant {
                out.push((hp[0].clone(), 0.0));
                if h.len() + 1 != hp.len() {
                    bail!("PAC model {}: {} h parameters expected, {} computed", self.name, hp.len(), h.len() + 1);
                }
                for (k, v) in h.iter().enumerate() {
                    out.push((hp[k + 1].clone(), *v));
                }
            } else {
                if h.len() != hp.len() {
                    bail!("PAC model {}: {} h parameters expected, {} computed", self.name, hp.len(), h.len());
                }
                for (k, v) in h.iter().enumerate() {
                    out.push((hp[k].clone(), *v));
                }
            }
            if growth_flag && !lrcp.is_nan() {
                // gamma (share of optimizing agents) is one: no rule of thumb agents
                let gg = -(lrcp - 1.0);
                let cc = 1.0 - gg;
                if let Some(gp) = gp {
                    out.push((gp.clone(), cc));
                }
            }
        }
        Ok(out)
    }

    /// `pac.mce.parameters`: the `mce_alpha` parameters and the growth neutrality parameter.
    pub fn mce_values(&self, names: &[String], params: &[f64]) -> Result<Vec<(String, f64)>> {
        if !self.is_mce() {
            bail!("This function is to be used only for PAC Model Consistent Expectations.");
        }
        let mut pv = self.pac_values(names, params)?;
        pv.push(self.beta(names, params)?);
        let (g, alpha, beta) = g_matrix(&pv)?;
        let mut out = Vec::new();
        // M_.params(mce.alpha) = flip(alpha)
        let mut fa = alpha.clone();
        fa.reverse();
        for (n, v) in self.mce_alpha.iter().zip(&fa) {
            out.push((n.clone(), *v));
        }
        if let Some(gp) = &self.growth_param {
            let m = alpha.len();
            let a1 = polyval1(&alpha, 1.0);
            let ab = polyval1(&alpha, beta);
            let img = Mat::identity(m, m) - &g;
            let em = unit(m, m - 1);
            let s1 = solve(&img, &em).ok_or_else(|| anyhow!("PAC: singular matrix"))?;
            let s2 = solve(&img, &s1).ok_or_else(|| anyhow!("PAC: singular matrix"))?;
            let d = a1 * ab * (em.transpose() * s2)[(0, 0)];
            let sum: f64 = pv[1..pv.len() - 1].iter().sum();
            out.push((gp.clone(), 1.0 - (sum + d)));
        }
        Ok(out)
    }
}

impl VarModels {
    pub fn pac(&self, name: &str) -> Option<&PacModel> {
        self.pacs.iter().find(|p| p.name == name)
    }

    /// `pac.initialize(name)`: builds the companion matrix of the auxiliary model.
    pub fn pac_initialize(&mut self, name: &str, names: &[String], params: &[f64]) -> Result<()> {
        let p = self.pac(name).ok_or_else(|| anyhow!("PAC model {name} is not defined in the model block!"))?;
        if let Some(aux) = p.aux_model.clone() {
            self.get_companion_matrix(&aux, names, params)?;
        }
        self.pacs.iter_mut().find(|p| p.name == name).unwrap().initialized = true;
        Ok(())
    }

    /// `pac.update.expectation(name)` (or `pac.mce.parameters(name)` for MCE): the new values.
    pub fn pac_update_values(&self, name: &str, names: &[String], params: &[f64]) -> Result<Vec<(String, f64)>> {
        let p = self.pac(name).ok_or_else(|| anyhow!("PAC model {name} is not defined in the model block!"))?;
        if p.is_mce() { p.mce_values(names, params) } else { p.update_values(self, names, params) }
    }
}

#[cfg(test)]
mod tests {
    use super::a2alpha;

    /// `a2alpha.m`: alpha(m) = a(m); alpha(2:m-1) = a(2:m-1)-a(3:m); alpha(1) = a(1)-a(2)-1, and an
    /// error for m = 1 (`a(2)` is out of range; the preprocessor accepts a PAC equation without
    /// autoregressive terms, so `pac.update.expectation` fails in MATLAB).
    #[test]
    fn a2alpha_follows_the_matlab_function() {
        let al = a2alpha(&[0.5, 0.3, 0.1]).unwrap();
        assert_eq!(al, vec![0.5 - 0.3 - 1.0, 0.3 - 0.1, 0.1]);
        assert_eq!(a2alpha(&[0.4, 0.1]).unwrap(), vec![0.4 - 0.1 - 1.0, 0.1]);
        assert!(a2alpha(&[0.5]).is_err());
    }
}
