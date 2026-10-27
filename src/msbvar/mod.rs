// SPDX-License-Identifier: GPL-3.0-or-later
//! Markov-switching SBVAR toolbox, part 1: the declarations (`svar_identification`,
//! `markov_switching`, `svar`) as `options_.ms.*` and the constant-parameter `sbvar` command.
//! The `ms_*` commands (estimation, simulation, mdd, probabilities, irf, forecast, variance
//! decomposition) build on [`MsDecl`] and [`identification::SvarIdentification`] and are not
//! ported yet.
//!
//! Field mapping (preprocessor-generated driver, Dynare 7.2):
//! * `svar_identification`: `options_.ms.{upper_cholesky,lower_cholesky,constants_exclusion}`,
//!   `options_.ms.Qi{eq}(row,var)` / `options_.ms.Ri{eq}(row,(lag-1)*n+var)` (see
//!   [`identification::SvarIdentification::qi_ri`]), `options_.ms.nlags` = largest lag.
//! * `markov_switching`: `options_.ms.ms_chain(c).regime(r).duration`,
//!   `.restrictions{k} = [from, to, prob]`, `.number_of_lags`, `.parameters` ([`MsChain`]).
//! * `svar`: `options_.ms.ms_chain(c).svar_variances.equations` /
//!   `svar_coefficients.equations` (`'ALL'` or a vector; [`SvarSpec`]).

pub mod hamilton;
pub mod identification;
pub mod mdd;
pub mod ms_commands;
pub mod ms_estimation;
pub mod ms_files;
pub mod ms_graphs;
pub mod ms_optim;
pub mod ms_sim;
pub mod sbvar;
pub mod sbvar_extra;

use std::path::Path;

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};

use crate::linalg::Mat;
use crate::matfile::{self, MatOut};
use crate::modfile::{BLOCK_BODY_KEY, Command, ModFile, OptValue, Options, opt};
use identification::{RestrictionKind, SvarIdentification};

/// Which equations a chain controls (`svar(..., equations=...)`; `ALL` when omitted).
#[derive(Clone, Debug, PartialEq)]
pub enum Equations {
    All,
    List(Vec<usize>),
}

/// `svar(variances|coefficients, equations=..., chain=N)`.
#[derive(Clone, Debug, PartialEq)]
pub struct SvarSpec {
    pub equations: Equations,
}

/// One Markov chain (`markov_switching`), with the `svar` statements that attach to it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MsChain {
    pub number: usize,
    pub number_of_regimes: usize,
    /// Average duration of each regime (a scalar `duration` is replicated).
    pub durations: Vec<f64>,
    /// `[from regime, to regime, probability]`, 1-based regimes.
    pub restrictions: Vec<[f64; 3]>,
    /// MS-DSGE only: switching parameters and `number_of_lags`.
    pub parameters: Vec<String>,
    pub number_of_lags: Option<usize>,
    pub svar_variances: Option<SvarSpec>,
    pub svar_coefficients: Option<SvarSpec>,
}

/// All MS-SBVAR declarations of a .mod file, in `options_.ms` terms.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MsDecl {
    pub identification: Option<SvarIdentification>,
    pub chains: Vec<MsChain>,
}

fn num(v: &OptValue) -> Option<f64> {
    match v {
        OptValue::Num(x) => Some(*x),
        OptValue::Str(s) => s.parse().ok(),
        _ => None,
    }
}

fn int(v: &OptValue, what: &str) -> Result<usize> {
    match num(v) {
        Some(x) if x >= 0.0 && x.fract() == 0.0 => Ok(x as usize),
        _ => bail!("{what} must be a non-negative integer"),
    }
}

fn nums(v: &OptValue) -> Option<Vec<f64>> {
    match v {
        OptValue::Vec(x) => Some(x.clone()),
        OptValue::Num(x) => Some(vec![*x]),
        OptValue::List(l) => l.iter().map(num).collect(),
        _ => None,
    }
}

impl MsDecl {
    /// Collects `svar_identification`, `markov_switching` and `svar` from the commands of `mf`
    /// and applies the preprocessor's consistency checks.
    pub fn from_modfile(mf: &ModFile) -> Result<MsDecl> {
        let mut d = MsDecl::default();
        let mut svars: Vec<(bool, usize, SvarSpec)> = Vec::new(); // (is_variances, chain, spec)
        for c in &mf.commands {
            let Command::Other(name, o) = c else { continue };
            match name.as_str() {
                "svar_identification" => {
                    if d.identification.is_some() {
                        bail!("You may only have one svar_identification block in your .mod file.");
                    }
                    let stmts: Vec<String> = o
                        .iter()
                        .filter(|(k, _)| k == BLOCK_BODY_KEY)
                        .filter_map(|(_, v)| if let OptValue::Str(s) = v { Some(s.clone()) } else { None })
                        .collect();
                    let pv = |n: &str| {
                        mf.model
                            .param_names
                            .iter()
                            .position(|p| p == n)
                            .and_then(|i| mf.model.params.get(i).copied())
                            .filter(|v| v.is_finite())
                    };
                    d.identification = Some(SvarIdentification::parse_with_params(&stmts, &mf.model.endo_names, &pv)?);
                }
                "markov_switching" => d.add_chain(o)?,
                "svar" => svars.push(parse_svar(o)?),
                _ => {}
            }
        }
        for (variances, chain, spec) in svars {
            let Some(ch) = d.chains.iter_mut().find(|c| c.number == chain) else {
                bail!("svar: chain {chain} has not been declared with markov_switching");
            };
            let slot = if variances { &mut ch.svar_variances } else { &mut ch.svar_coefficients };
            if slot.is_some() {
                bail!("svar: chain {chain} is already controlling these parameters");
            }
            *slot = Some(spec);
        }
        Ok(d)
    }

    fn add_chain(&mut self, o: &Options) -> Result<()> {
        let mut ch = MsChain::default();
        ch.number = int(
            opt(o, "chain").ok_or_else(|| anyhow!("markov_switching: the chain option is required"))?,
            "markov_switching: chain",
        )?;
        if ch.number != self.chains.len() + 1 {
            bail!("The markov_switching chain option takes consecutive integers beginning at 1.");
        }
        let dur = opt(o, "duration").ok_or_else(|| anyhow!("markov_switching: the duration option is required"))?;
        let dur = nums(dur).ok_or_else(|| anyhow!("markov_switching: duration must be a number or a vector"))?;
        ch.number_of_regimes = match opt(o, "number_of_regimes") {
            Some(v) => int(v, "markov_switching: number_of_regimes")?,
            None if dur.len() > 1 => dur.len(),
            None => bail!("markov_switching: the number_of_regimes option is required"),
        };
        let n = ch.number_of_regimes;
        if n < 1 {
            bail!("markov_switching: number_of_regimes must be positive");
        }
        ch.durations = match dur.len() {
            1 => vec![dur[0]; n],
            l if l == n => dur,
            _ => bail!("markov_switching: duration must be a scalar or have number_of_regimes entries"),
        };
        if let Some(r) = opt(o, "restrictions") {
            let rows: Vec<Vec<f64>> = match r {
                OptValue::List(l) => l.iter().map(|x| nums(x).unwrap_or_default()).collect(),
                OptValue::Vec(v) => vec![v.clone()],
                _ => bail!("markov_switching: restrictions must be [[from,to,prob],...]"),
            };
            for row in rows {
                if row.len() != 3 {
                    bail!(
                        "restrictions in the markov_switching statement must be specified in the form [from, to, probability]"
                    );
                }
                let (f, t, p) = (row[0], row[1], row[2]);
                if f.fract() != 0.0 || t.fract() != 0.0 || f < 1.0 || t < 1.0 {
                    bail!("The first two arguments for a restriction must be integers");
                }
                if f as usize > n || t as usize > n {
                    bail!("the regimes specified in the restrictions option must be <= number_of_regimes");
                }
                if !(0.0..=1.0).contains(&p) {
                    bail!("the transition probability of a restriction must be in [0,1]");
                }
                if ch.restrictions.iter().any(|r| r[0] == f && r[1] == t) {
                    bail!("two restrictions were given for: {f}, {t}");
                }
                ch.restrictions.push([f, t, p]);
            }
            for from in 1..=n {
                let sel: Vec<f64> = ch.restrictions.iter().filter(|r| r[0] as usize == from).map(|r| r[2]).collect();
                let sum: f64 = sel.iter().sum();
                if sel.len() == n && (sum - 1.0).abs() > 1e-10 {
                    bail!("When transition probabilities are specified for every regime, their sum must be 1");
                }
                if !sel.is_empty() && sel.len() < n && sum >= 1.0 {
                    bail!("When transition probabilites are not specified for every regime, their sum must be < 1");
                }
            }
        }
        if let Some(OptValue::List(l)) = opt(o, "parameters") {
            ch.parameters =
                l.iter().filter_map(|v| if let OptValue::Str(s) = v { Some(s.clone()) } else { None }).collect();
        } else if let Some(OptValue::Str(s)) = opt(o, "parameters") {
            ch.parameters = vec![s.clone()];
        }
        if let Some(v) = opt(o, "number_of_lags") {
            ch.number_of_lags = Some(int(v, "markov_switching: number_of_lags")?);
        }
        self.chains.push(ch);
        Ok(())
    }
}

fn parse_svar(o: &Options) -> Result<(bool, usize, SvarSpec)> {
    let has = |k: &str| opt(o, k).is_some();
    let variances = match (has("variances"), has("coefficients")) {
        (true, false) => true,
        (false, true) => false,
        _ => bail!("svar: exactly one of `coefficients` or `variances` must be given"),
    };
    let chain = int(opt(o, "chain").ok_or_else(|| anyhow!("svar: the chain option is required"))?, "svar: chain")?;
    let equations = match opt(o, "equations") {
        None => Equations::All,
        Some(v) => {
            let l = nums(v).ok_or_else(|| anyhow!("svar: equations must be an integer or a vector"))?;
            if let Some(x) = l.iter().find(|x| !(**x >= 1.0 && x.fract() == 0.0)) {
                bail!("svar: equation {x} is not a valid equation number (1 to the number of observed variables)");
            }
            Equations::List(l.into_iter().map(|x| x as usize).collect())
        }
    };
    Ok((variances, chain, SvarSpec { equations }))
}

// ---------------------------------------------------------------------------------------
// sbvar
// ---------------------------------------------------------------------------------------

/// The commands handled here.
pub fn handles(name: &str) -> bool {
    matches!(name, "sbvar" | "markov_switching" | "svar" | "svar_identification") || ms_commands::handles(name)
}

/// The options read by `sbvar` and the `ms_*` commands: `options_.ms.*` (defaults of
/// `initialize_ms_sbvar_options.m`), `options_.datafile`, `options_.xls_sheet`/`xls_range` and
/// `options_.forecast`.
#[derive(Clone, Debug)]
pub struct SbvarOptions {
    pub datafile: Option<String>,
    /// `options_.xls_sheet` / `options_.xls_range` (spreadsheet datafiles).
    pub xls: crate::data::DataOptions,
    pub freq: usize,
    pub initial_year: Option<usize>,
    pub initial_subperiod: usize,
    pub final_year: Option<usize>,
    pub final_subperiod: Option<usize>,
    pub nlags: usize,
    pub mu: [f64; 6],
    pub bayesian_prior: bool,
    pub contemp_reduced_form: bool,
    pub cross_restrictions: bool,
    /// `options_.forecast` (years; `sbvar` uses 4 when it is 0).
    pub forecast: usize,
    pub restriction_fname: Option<String>,
    /// `vlistlog`: observed variables in logs (growth-rate files and graphs).
    pub vlistlog: Vec<String>,
}

impl Default for SbvarOptions {
    fn default() -> Self {
        let mut s = SbvarOptions {
            datafile: None,
            xls: Default::default(),
            freq: 4,
            initial_year: None,
            initial_subperiod: 1,
            final_year: None,
            final_subperiod: None,
            nlags: 1,
            mu: [1.0, 1.0, 0.1, 1.2, 1.0, 1.0],
            bayesian_prior: true,
            contemp_reduced_form: false,
            cross_restrictions: false,
            forecast: 0,
            restriction_fname: None,
            vlistlog: Vec::new(),
        };
        s.initialize_ms_sbvar_options();
        s
    }
}

impl SbvarOptions {
    /// The fields that `initialize_ms_sbvar_options.m` resets (the others keep their value).
    fn initialize_ms_sbvar_options(&mut self) {
        self.mu = [1.0, 1.0, 0.1, 1.2, 1.0, 1.0];
        self.freq = 4;
        self.initial_subperiod = 1;
        self.final_subperiod = None;
        self.nlags = 1;
        self.cross_restrictions = false;
        self.contemp_reduced_form = false;
        self.bayesian_prior = true;
        self.initial_year = None;
        self.final_year = None;
    }

    /// The options of the command `o` alone, over the defaults.
    pub fn from_options(o: &Options) -> Result<SbvarOptions> {
        let mut s = SbvarOptions::default();
        s.apply("sbvar", o)?;
        s.check()?;
        Ok(s)
    }

    /// The options in force when the command `o` (an element of `mf.commands`) runs: Dynare's
    /// global `options_` as the preprocessed driver leaves it. In file order,
    /// `svar_identification` sets `options_.ms.nlags` to the largest lag of its restrictions
    /// (`ident_nlags`), every `ms_*` command first calls `initialize_ms_sbvar_options` (and
    /// `ms_estimation` clears `options_.datafile`), the options of `sbvar` and of the `ms_*`
    /// commands set `options_.ms.*`, and those of any command set `options_.datafile`,
    /// `options_.xls_sheet`/`xls_range` and `options_.forecast`. No
    /// `initialize_ms_sbvar_options` precedes `sbvar`, so it sees what the earlier commands
    /// left. When `o` is not one of the commands, the options are those of an `sbvar` command
    /// with options `o` right after the declarations.
    pub fn in_force(mf: &ModFile, o: &Options, ident_nlags: Option<usize>) -> Result<SbvarOptions> {
        use crate::bvar::commands::command_parts;
        let none = Options::new();
        let cmds: Vec<(&str, &Options)> = match mf.commands.iter().position(|c| std::ptr::eq(command_parts(c).1, o)) {
            Some(k) => mf.commands[..=k].iter().map(command_parts).collect(),
            None => vec![("svar_identification", &none), ("sbvar", o)],
        };
        let mut s = SbvarOptions::default();
        for (name, opts) in cmds {
            if name == "svar_identification" {
                if let Some(l) = ident_nlags {
                    s.nlags = l;
                }
            } else if ms_commands::handles(name) {
                s.initialize_ms_sbvar_options();
                if name == "ms_estimation" {
                    s.datafile = None;
                }
            }
            s.apply(name, opts)?;
        }
        s.check()?;
        Ok(s)
    }

    /// The assignments to `options_` made by the options `o` of the command `name`.
    fn apply(&mut self, name: &str, o: &Options) -> Result<()> {
        let ms = name == "sbvar" || ms_commands::handles(name);
        let flag = |v: &OptValue| !matches!(num(v), Some(0.0));
        for (k, v) in o {
            match k.as_str() {
                // method_of_moments sets options_mom_.datafile
                "datafile" if name != "method_of_moments" => {
                    self.datafile = Some(match v {
                        OptValue::Str(x) => x.clone(),
                        _ => bail!("{name}: datafile must be a file name"),
                    })
                }
                "xls_sheet" => {
                    self.xls.xls_sheet = match v {
                        OptValue::Num(k) => Some(format!("{k}")),
                        OptValue::Str(x) => Some(x.clone()),
                        _ => None,
                    }
                }
                "xls_range" => {
                    if let OptValue::Str(x) = v {
                        self.xls.xls_range = Some(x.clone())
                    }
                }
                "forecast" if ms => self.forecast = int(v, "sbvar: forecast")?,
                "forecast" => {
                    if let Ok(x) = int(v, "forecast") {
                        self.forecast = x
                    }
                }
                _ if !ms => {}
                "freq" => {
                    self.freq = match v {
                        OptValue::Str(x) if x.eq_ignore_ascii_case("monthly") => 12,
                        OptValue::Str(x) if x.eq_ignore_ascii_case("quarterly") => 4,
                        OptValue::Str(x) if x.eq_ignore_ascii_case("yearly") => 1,
                        _ => int(v, "sbvar: freq")?,
                    }
                }
                "initial_year" => self.initial_year = Some(int(v, "sbvar: initial_year")?),
                "initial_subperiod" => self.initial_subperiod = int(v, "sbvar: initial_subperiod")?,
                "final_year" => self.final_year = Some(int(v, "sbvar: final_year")?),
                "final_subperiod" => self.final_subperiod = Some(int(v, "sbvar: final_subperiod")?),
                "nlags" => self.nlags = int(v, "sbvar: nlags")?,
                "no_bayesian_prior" => self.bayesian_prior = !flag(v),
                "contemp_reduced_form" => self.contemp_reduced_form = flag(v),
                "cross_restrictions" => self.cross_restrictions = flag(v),
                "vlistlog" => {
                    fn names(v: &OptValue, out: &mut Vec<String>) {
                        match v {
                            OptValue::Str(x) => out.push(x.clone()),
                            OptValue::List(l) => l.iter().for_each(|e| names(e, out)),
                            _ => {}
                        }
                    }
                    self.vlistlog.clear();
                    names(v, &mut self.vlistlog);
                }
                "restriction_fname" => {
                    if let OptValue::Str(x) = v {
                        self.restriction_fname = Some(x.trim_matches('\'').to_string())
                    }
                }
                "coefficients_prior_hyperparameters" => {
                    let x = nums(v)
                        .filter(|x| x.len() == 6)
                        .ok_or_else(|| anyhow!("sbvar: coefficients_prior_hyperparameters needs 6 values"))?;
                    self.mu.copy_from_slice(&x);
                }
                _ => {} // data / vlist / forecast-band / conditional-forecast options of the TZ code: unused here
            }
        }
        Ok(())
    }

    fn check(&self) -> Result<()> {
        if self.freq < 1 {
            bail!("sbvar: freq must be a positive integer");
        }
        if self.nlags == 0 {
            bail!("sbvar: nlags must be a positive integer");
        }
        Ok(())
    }
}

/// Runs `sbvar` (command `ci`), writing `outm.mat` next to the .mod file (as MATLAB writes it
/// into the working directory) and `<fname>/Output/sbvar_results.json`.
pub fn run_command(mf: &ModFile, c: &Command, dir: &Path, fname: &str, seed: u64) -> Result<()> {
    let Command::Other(cname, o) = c else { bail!("sbvar: unexpected command") };
    if ms_commands::handles(cname) {
        return ms_commands::run_ms_command(mf, c, dir, fname, seed);
    }
    if cname != "sbvar" {
        // declarations: validated here (as the preprocessor does), consumed by sbvar / ms_*
        MsDecl::from_modfile(mf)?;
        return Ok(());
    }
    let p = prepare_sample(mf, o, dir)?;
    let res = sbvar::sbvar(&p.z, &p.settings, &p.restr)?;
    let opts = p.opts.clone();
    let nvar = res.nvar;
    println!("\nSBVAR: {} variables, {} lags, {} effective observations", nvar, opts.nlags, res.fss);
    println!("Posterior mode objective (fhat): {:.8}", res.fhat);
    println!("A0 (posterior mode):");
    for i in 0..nvar {
        println!("  {}", (0..nvar).map(|j| format!("{:12.6}", res.a0[(i, j)])).collect::<Vec<_>>().join(" "));
    }
    let out = dir.join(fname).join("Output");
    std::fs::create_dir_all(&out)?;
    matfile::write(
        &dir.join("outm.mat"),
        &[
            ("xhat", MatOut::Vector(&res.xhat)),
            ("A0hat", MatOut::Matrix(&res.a0)),
            ("A0hatinv", MatOut::Matrix(&res.a0_inv)),
            ("grad", MatOut::Vector(&res.grad)),
            ("fhat", MatOut::Scalar(res.fhat)),
            ("itct", MatOut::Scalar(res.iterations as f64)),
            ("fcount", MatOut::Scalar(res.fcount as f64)),
            ("retcodehat", MatOut::Scalar(0.0)),
        ],
    )?;
    let mut json = result_json(&res);
    // growth-rate files, graphs and the Gibbs-sampled marginal data density
    let vlistlog: Vec<usize> = opts
        .vlistlog
        .iter()
        .map(|n| {
            mf.varobs
                .iter()
                .position(|v| v == n)
                .ok_or_else(|| anyhow!("sbvar: vlistlog: {n} is not an observed variable"))
        })
        .collect::<Result<_>>()?;
    let y0 = opts.initial_year.unwrap_or(1) as i64;
    let s0 = opts.initial_subperiod as i64;
    sbvar_extra::write_prn(dir, &p.z, y0, s0, opts.freq, &vlistlog)?;
    let g = crate::graphics::GraphOptions::from_options(o, dir, fname);
    let files = sbvar_extra::graphs(
        &res,
        &mf.varobs,
        &p.z,
        y0,
        s0,
        opts.freq,
        &vlistlog,
        p.settings.forecast_steps,
        fname,
        &g,
    )?;
    if !files.is_empty() {
        println!("sbvar: {} figure file(s) written to {}", files.len(), g.dir.display());
    }
    if res.vlog_a0p.is_some() {
        let m = sbvar_extra::gibbs_mdd(&res, seed, 1000, 10000)?;
        println!("Log marginal data density: {:.4}", m.log_mdd);
        json["vlog_a0_Yao"] = json!(m.vlog_a0_yao);
        json["log_mdd"] = json!(m.log_mdd);
    }
    std::fs::write(out.join("sbvar_results.json"), serde_json::to_string_pretty(&json)?)?;
    Ok(())
}

fn rows(m: &Mat) -> Value {
    json!((0..m.nrows()).map(|i| m.row(i).iter().copied().collect::<Vec<f64>>()).collect::<Vec<_>>())
}

/// JSON of the results (matrices as arrays of rows).
pub fn result_json(r: &sbvar::SbvarResult) -> Value {
    json!({
        "nvar": r.nvar, "ncoef": r.ncoef, "fss": r.fss,
        "A0hat": rows(&r.a0), "A0hatinv": rows(&r.a0_inv), "Fhat": rows(&r.aplus), "Bhat": rows(&r.bhat),
        "fhat": r.fhat, "xhat": r.xhat, "imfhat": rows(&r.imf), "yforehat": rows(&r.forecast),
        "vlog_a0p": r.vlog_a0p, "vlog_Y_a": r.vlog_y_a, "logMarLHres": r.log_mar_lh_res, "vlog_ap_Ya0": r.vlog_ap_ya0,
    })
}

/// Computes `sbvar` for the option list `o` of the command (everything but the output).
pub fn run_sbvar(mf: &ModFile, o: &Options, dir: &Path) -> Result<(sbvar::SbvarResult, SbvarOptions)> {
    let p = prepare_sample(mf, o, dir)?;
    let res = sbvar::sbvar(&p.z, &p.settings, &p.restr)?;
    Ok((res, p.opts))
}

/// The sample and restrictions shared by `sbvar` and the `ms_*` commands (`ms_sbvar_setup.m`).
pub struct Prepared {
    pub z: Mat,
    pub opts: SbvarOptions,
    pub decl: MsDecl,
    pub restr: identification::Restrictions,
    pub settings: sbvar::SbvarSettings,
}

/// `o`: the options of the `sbvar` or `ms_estimation` command, an element of `mf.commands`
/// (see [`SbvarOptions::in_force`]).
pub fn prepare_sample(mf: &ModFile, o: &Options, dir: &Path) -> Result<Prepared> {
    let decl = MsDecl::from_modfile(mf)?;
    let ident = decl.identification.clone().unwrap_or_default();
    let opts = SbvarOptions::in_force(mf, o, ident.max_lag())?;
    let f = opts.datafile.clone().ok_or_else(|| anyhow!("Must set datafile option"))?;
    let iy = opts.initial_year.ok_or_else(|| anyhow!("Must set initial_year option"))?;
    let data = crate::bvar::commands::read_variables(dir, &f, &mf.varobs, &opts.xls)?;
    let nobs = data.nrows();
    let freq = opts.freq;
    // check_datafile_years_assigned
    let (fy, fsub) = match (opts.final_year, opts.final_subperiod) {
        (None, _) => ((iy + (nobs - 1) / freq), (opts.initial_subperiod + nobs - 2) % freq + 1),
        (Some(y), None) => (y, freq),
        (Some(y), Some(s)) => (y, s),
    };
    let ndata = (fy - iy) * freq + fsub + 1 - opts.initial_subperiod;
    if nobs < ndata {
        bail!("The declared sample is longer than the available data");
    }
    let z = data.rows(0, ndata).into_owned();
    let kind = if ident.upper_cholesky {
        RestrictionKind::UpperCholesky
    } else if ident.lower_cholesky {
        RestrictionKind::LowerCholesky
    } else if !ident.rows.is_empty() {
        RestrictionKind::Exclusions
    } else {
        match opts.restriction_fname.as_deref() {
            Some("upper_cholesky") => RestrictionKind::UpperCholesky,
            Some("lower_cholesky") => RestrictionKind::LowerCholesky,
            Some("exclusions") => RestrictionKind::Exclusions,
            Some(x) => bail!("sbvar: unsupported restriction_fname {x}"),
            None => bail!("sbvar: no identification: add a svar_identification block"),
        }
    };
    let nlags = opts.nlags;
    let nvar = mf.varobs.len();
    let restr = ident.restrictions_for_cross(kind, nvar, nlags, 1, opts.cross_restrictions)?;
    if !restr.cross.is_empty() && opts.contemp_reduced_form {
        bail!("sbvar: cross A0-A+ restrictions cannot be combined with contemp_reduced_form");
    }
    let forecast_years = if opts.forecast == 0 { 4 } else { opts.forecast };
    let settings = sbvar::SbvarSettings {
        freq,
        nlags,
        mu: opts.mu,
        bayesian_prior: opts.bayesian_prior,
        contemp_reduced_form: opts.contemp_reduced_form,
        irf_steps: 4 * freq,
        forecast_steps: forecast_years * freq,
    };
    Ok(Prepared { z, opts, decl, restr, settings })
}
