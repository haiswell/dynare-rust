// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `estimation/model_comparison.m`: posterior model probabilities from the log
//! marginal data densities of earlier estimation runs.
//!
//! Inputs: for every model `NAME`, the results file that dynare-rust's `estimation` writes,
//! `<dir of the .mod>/NAME/Output/NAME_results.json` (Dynare reads `NAME_results.mat`). The
//! log densities are read from (first match wins):
//!
//! | estimator | JSON keys |
//! |---|---|
//! | Laplace (`marginal_density = laplace`, default) | `MarginalDensity.LaplaceApproximation`, `log_data_density_laplace` |
//! | Modified harmonic mean (`= modifiedharmonicmean`) | `MarginalDensity.ModifiedHarmonicMean`, `mcmc.log_data_density_mhm` |
//!
//! A model without the requested estimate (an SMC run only has `MarginalDensity.dsmh` or
//! `MarginalDensity.hssmc`) stops the comparison with Dynare's "I can't find" message and no
//! output, as `model_comparison.m`'s `catch` branch does.
//!
//! Output: `oo_.Model_Comparison` as `<fname>/Output/<fname>_model_comparison.json`
//! (`{"estimator": .., "models": {NAME: {"Prior", "Log_Marginal_Density", "Bayes_Ratio",
//! "Posterior_Model_Probability"}}}`) and, with `TeX`, the `dyn_latex_table` file
//! `<fname>/latex/<fname>_model_comparison<Estimator>.tex`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::modfile::{Command, ModFile, OptValue};

/// Which marginal density estimator to compare.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimator {
    LaplaceApproximation,
    ModifiedHarmonicMean,
}

impl Estimator {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "laplace" | "Laplace" => Ok(Self::LaplaceApproximation),
            "modifiedharmonicmean" | "ModifiedHarmonicMean" => Ok(Self::ModifiedHarmonicMean),
            _ => bail!("model_comparison: unknown marginal_density '{s}' (laplace or modifiedharmonicmean)"),
        }
    }
    pub fn key(self) -> &'static str {
        match self {
            Self::LaplaceApproximation => "LaplaceApproximation",
            Self::ModifiedHarmonicMean => "ModifiedHarmonicMean",
        }
    }
    pub fn title(self) -> &'static str {
        match self {
            Self::LaplaceApproximation => "Model Comparison (based on Laplace approximation)",
            Self::ModifiedHarmonicMean => "Model Comparison (based on Modified Harmonic Mean Estimator)",
        }
    }
}

/// `oo_.Model_Comparison`, one entry per model in the given order.
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    pub names: Vec<String>,
    /// Normalized prior model probabilities.
    pub prior: Vec<f64>,
    pub log_marginal_density: Vec<f64>,
    /// Posterior odds relative to the first model, exp(lmpd - lmpd(1)) (Dynare's `Bayes_Ratio`
    /// when priors are given, which the preprocessor always does).
    pub bayes_ratio: Vec<f64>,
    pub posterior_probability: Vec<f64>,
}

/// The computation of `model_comparison.m` (with `ModelPriors` given; priors that do not sum
/// to one are rescaled, with a message unless they are all 1).
pub fn compare(names: &[String], priors: &[f64], log_density: &[f64]) -> Comparison {
    let sum: f64 = priors.iter().sum();
    let prior: Vec<f64> = if (sum - 1.0).abs() > 1e-6 {
        if !priors.iter().all(|&p| p == 1.0) {
            println!("model_comparison:: The user supplied prior distribution over models is improper...");
            println!("model_comparison:: The distribution is automatically rescaled!");
        }
        priors.iter().map(|p| p / sum).collect()
    } else {
        priors.to_vec()
    };
    let lmpd: Vec<f64> = prior.iter().zip(log_density).map(|(p, d)| p.ln() + d).collect();
    let maxval = lmpd.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let elmpd: Vec<f64> = lmpd.iter().map(|v| (v - maxval).exp()).collect();
    let tot: f64 = elmpd.iter().sum();
    Comparison {
        names: names.to_vec(),
        prior,
        log_marginal_density: log_density.to_vec(),
        bayes_ratio: lmpd.iter().map(|v| (v - lmpd[0]).exp()).collect(),
        posterior_probability: elmpd.iter().map(|e| e / tot).collect(),
    }
}

/// `get_short_names`: strips the directory and a `.mod`/`.dyn` extension.
pub fn short_name(name: &str) -> String {
    let n = name.strip_suffix(".mod").or_else(|| name.strip_suffix(".dyn")).unwrap_or(name);
    n.rsplit(['/', '\\']).next().unwrap_or(n).to_string()
}

/// Path of a model's results file.
pub fn results_path(dir: &Path, name: &str) -> PathBuf {
    let base = name.strip_suffix(".mod").or_else(|| name.strip_suffix(".dyn")).unwrap_or(name);
    let short = short_name(name);
    dir.join(base).join("Output").join(format!("{short}_results.json"))
}

/// Reads a log marginal density from a results JSON (see the module docs).
pub fn read_log_density(results: &Value, est: Estimator) -> Option<f64> {
    if let Some(v) = results.pointer(&format!("/MarginalDensity/{}", est.key())).and_then(Value::as_f64) {
        return Some(v);
    }
    let v = match est {
        Estimator::LaplaceApproximation => results.get("log_data_density_laplace").and_then(Value::as_f64),
        Estimator::ModifiedHarmonicMean => results.pointer("/mcmc/log_data_density_mhm").and_then(Value::as_f64),
    };
    v.filter(|x| x.is_finite())
}

/// Splits the raw tokens of `model_comparison m1(0.7) dir/m2.mod(0.3)` into (name, prior).
pub fn parse_models(tokens: &[String]) -> Result<Vec<(String, f64)>> {
    let mut out: Vec<(String, f64)> = Vec::new();
    let mut cur = String::new();
    let mut prev_word = false;
    let mut i = 0;
    let is_word = |t: &str| t.chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_');
    while i < tokens.len() {
        let t = tokens[i].as_str();
        if t == "(" {
            let v: f64 = tokens
                .get(i + 1)
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| anyhow!("model_comparison: bad prior after {cur}"))?;
            if tokens.get(i + 2).map(String::as_str) != Some(")") {
                bail!("model_comparison: expected ')' after the prior of {cur}");
            }
            if cur.is_empty() {
                bail!("model_comparison: prior without a model name");
            }
            out.push((std::mem::take(&mut cur), v));
            prev_word = false;
            i += 3;
            continue;
        }
        if t == "," {
            if !cur.is_empty() {
                out.push((std::mem::take(&mut cur), 1.0));
            }
            prev_word = false;
        } else if is_word(t) && prev_word && !cur.is_empty() {
            out.push((std::mem::replace(&mut cur, t.to_string()), 1.0));
        } else {
            cur.push_str(t.trim_matches('\''));
            prev_word = is_word(t);
        }
        i += 1;
    }
    if !cur.is_empty() {
        out.push((cur, 1.0));
    }
    if out.is_empty() {
        bail!("model_comparison: no model given");
    }
    Ok(out)
}

/// Driver entry point: `model_comparison[(marginal_density = ...)] NAME[(PRIOR)] ...;`.
/// `None` when a model has no estimate of the requested marginal density (Dynare prints a
/// message and returns).
pub fn run_command(mf: &ModFile, c: &Command, dir: &Path, fname: &str) -> Result<Option<Comparison>> {
    let Command::Other(_, o) = c else { bail!("model_comparison: unexpected command") };
    let k = mf.commands.iter().position(|x| std::ptr::eq(x, c)).unwrap_or(mf.commands.len() - 1);
    // options_ is global: `marginal_density` and `TeX` persist from earlier commands
    let (mut est, mut tex) = (Estimator::LaplaceApproximation, false);
    for c in &mf.commands[..=k] {
        let (_, opts) = crate::bvar::commands::command_parts(c);
        for (key, v) in opts {
            match (key.as_str(), v) {
                ("marginal_density", OptValue::Str(s)) => est = Estimator::parse(s)?,
                ("tex", _) => tex = true,
                _ => {}
            }
        }
    }
    let models = parse_models(&crate::bvar::commands::raw_args(o))?;
    let names: Vec<String> = models.iter().map(|(n, _)| short_name(n)).collect();
    let priors: Vec<f64> = models.iter().map(|(_, p)| *p).collect();
    let mut dens = Vec::new();
    for (n, _) in &models {
        let p = results_path(dir, n);
        let txt = std::fs::read_to_string(&p).with_context(|| format!("model_comparison: reading {}", p.display()))?;
        let v: Value =
            serde_json::from_str(&txt).with_context(|| format!("model_comparison: parsing {}", p.display()))?;
        // try MarginalLogDensity(i) = mstruct.oo_.MarginalDensity.(type); catch: disp, return
        let Some(d) = read_log_density(&v, est) else {
            println!(
                "MODEL_COMPARISON: I cant't find the {} associated to model {n}",
                match est {
                    Estimator::LaplaceApproximation => "Laplace approximation",
                    Estimator::ModifiedHarmonicMean => "modified harmonic mean estimate",
                }
            );
            return Ok(None);
        };
        dens.push(d);
    }
    let r = compare(&names, &priors, &dens);
    print_table(&r, est);
    let mut models_json = serde_json::Map::new();
    for i in 0..names.len() {
        models_json.insert(
            names[i].clone(),
            json!({"Prior": r.prior[i], "Log_Marginal_Density": r.log_marginal_density[i],
                   "Bayes_Ratio": r.bayes_ratio[i], "Posterior_Model_Probability": r.posterior_probability[i]}),
        );
    }
    let out_dir = dir.join(fname).join("Output");
    std::fs::create_dir_all(&out_dir)?;
    std::fs::write(
        out_dir.join(format!("{fname}_model_comparison.json")),
        serde_json::to_string_pretty(&json!({"estimator": est.key(), "models": Value::Object(models_json)}))?,
    )?;
    if tex {
        let mut headers = vec!["Model".to_string()];
        headers.extend(names.iter().map(|n| n.replace('_', "\\_")));
        let labels: Vec<String> = ["Priors", "Log Marginal Density", "Bayes Ratio", "Posterior Model Probability"]
            .iter()
            .map(|l| l.replace(' ', "\\ "))
            .collect();
        let values = vec![
            r.prior.clone(),
            r.log_marginal_density.clone(),
            r.bayes_ratio.clone(),
            r.posterior_probability.clone(),
        ];
        crate::latex::dyn_latex_table(
            dir,
            fname,
            est.title(),
            &format!("model_comparison{}", est.key()),
            "model_comparison.m",
            &headers,
            &labels,
            &values,
            0,
            16,
            6,
        )?;
    }
    Ok(Some(r))
}

/// `dyntable(options_, title, headers, labels, values, 0, 15, 6)`.
fn print_table(r: &Comparison, est: Estimator) {
    let labels = ["Priors", "Log Marginal Density", "Bayes Ratio", "Posterior Model Probability"];
    let lw = labels.iter().map(|l| l.len()).max().unwrap_or(5).max(5) + 2;
    let vw = r.names.iter().map(|n| n.len()).max().unwrap_or(0).max(10) + 4;
    let vw = vw.max(15);
    println!("\n\n{}\n", est.title());
    let mut h = format!("{:<lw$}", "Model");
    for n in &r.names {
        h.push_str(&format!("{n:>vw$}"));
    }
    println!("{h}");
    for (l, row) in labels.iter().zip([&r.prior, &r.log_marginal_density, &r.bayes_ratio, &r.posterior_probability]) {
        let mut s = format!("{l:<lw$}");
        for v in row.iter() {
            s.push_str(&format!("{v:>vw$.6}"));
        }
        println!("{s}");
    }
    println!();
}
