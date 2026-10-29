// SPDX-License-Identifier: GPL-3.0-or-later
//! Drivers of `ms_estimation`, `ms_compute_probabilities` and `ms_compute_mdd`.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde_json::json;

use super::mdd::{self, MddOptions};
use super::ms_estimation::{self, MsEstimate, MsOptions, MsSetup, estimate_json, mat_rows, write_estimation_files};
use super::{SbvarOptions, prepare_sample};
use super::{ms_files, ms_optim};
use crate::linalg::Mat;
use crate::modfile::{Command, ModFile, OptValue, Options};

pub fn handles(name: &str) -> bool {
    matches!(
        name,
        "ms_estimation"
            | "ms_compute_probabilities"
            | "ms_compute_mdd"
            | "ms_simulation"
            | "ms_irf"
            | "ms_forecast"
            | "ms_variance_decomposition"
    )
}

fn onum(o: &Options, key: &str) -> Option<f64> {
    o.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        OptValue::Num(x) => Some(*x),
        OptValue::Str(s) => s.parse().ok(),
        _ => None,
    })
}

fn oflag(o: &Options, key: &str) -> bool {
    o.iter().any(|(k, v)| k == key && !matches!(v, OptValue::Num(x) if *x == 0.0))
}

fn ovec(o: &Options, key: &str) -> Option<Vec<f64>> {
    o.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        OptValue::Vec(x) => Some(x.clone()),
        OptValue::Num(x) => Some(vec![*x]),
        OptValue::List(l) => {
            Some(l.iter().filter_map(|e| if let OptValue::Num(x) = e { Some(*x) } else { None }).collect())
        }
        _ => None,
    })
}

fn ostr(o: &Options, key: &str) -> Option<String> {
    o.iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| if let OptValue::Str(s) = v { Some(s.trim_matches('\'').to_string()) } else { None })
}

/// The options of `ms_compute_mdd` (`options_.ms.proposal_*`, `use_mean_center`).
///
/// The reference manual names two of them `mdd_proposal_draws` and `mdd_use_mean_center`,
/// spellings Dynare 7.2's preprocessor rejects (it only knows `proposal_draws` and
/// `use_mean_center`); both spellings are accepted here. `use_mean_center` has no effect, as in
/// Dynare: ms_compute_mdd.m passes `-use_mean` to the MEX, whose command-line parser only knows
/// `-mean` (checked with Dynare 7.2: the same seed with and without `-use_mean` gives
/// byte-identical mdd files, `-mean` changes them).
pub fn mdd_options(o: &Options) -> MddOptions {
    let mut mo = MddOptions::default();
    if let Some(x) = onum(o, "proposal_type") {
        mo.proposal_type = x as usize;
    }
    if let Some(x) = onum(o, "proposal_draws").or_else(|| onum(o, "mdd_proposal_draws")) {
        mo.proposal_draws = x as usize;
    }
    if let Some(x) = onum(o, "proposal_lower_bound") {
        mo.lower = x;
    }
    if let Some(x) = onum(o, "proposal_upper_bound") {
        mo.upper = x;
    }
    mo.use_mean_center = oflag(o, "use_mean_center") || oflag(o, "mdd_use_mean_center");
    if mo.use_mean_center {
        println!(
            "Note: ms_compute_mdd: use_mean_center has no effect (as in Dynare 7.2, whose MEX ignores the -use_mean flag): the proposal stays centred on the best posterior draw."
        );
    }
    mo
}

pub fn run_ms_command(mf: &ModFile, c: &Command, dir: &Path, fname: &str, seed: u64) -> Result<()> {
    let Command::Other(name, o) = c else { bail!("ms command: unexpected command") };
    let out = dir.join(fname).join("Output");
    std::fs::create_dir_all(&out)?;
    let ftag = ostr(o, "file_tag").unwrap_or_else(|| fname.to_string());
    let tag = ostr(o, "output_file_tag").unwrap_or_else(|| ftag.clone());
    match name.as_str() {
        "ms_estimation" => {
            println!("MS-SBVAR Estimation");
            if oflag(o, "cross_restrictions") {
                bail!(
                    "ms_estimation: cross_restrictions is not supported (Dynare 7.2 fails as well: ms_sbvar_setup.m uses the undefined Fhat)"
                );
            }
            let mo = MsOptions::from_options(o)?;
            let init = dir.join(format!("init_{ftag}.dat"));
            let (setup, start) = if mo.no_create_init {
                if !init.is_file() {
                    bail!(
                        "You must run ms_estimation without no_create_init as you are missing an initialization file it needs ({}).",
                        init.display()
                    );
                }
                ms_files::read_init(&init)?
            } else {
                let p = prepare_sample(mf, o, dir)?;
                let SbvarOptions { freq, initial_year, initial_subperiod, .. } = &p.opts;
                let setup = MsSetup::build(
                    &p.z,
                    &p.decl,
                    &p.restr,
                    &p.settings,
                    &mo,
                    (*freq, initial_year.unwrap_or(1), *initial_subperiod),
                )?;
                let start = setup.start_value(&p.z, &p.settings, &p.restr, &p.decl)?;
                ms_files::write_init(&init, &setup, &start)?;
                (setup, start)
            };
            let setup = Arc::new(setup);
            let mut trace = ms_optim::Trace::default();
            let (mut theta, _) = ms_optim::maximise(&setup, &start, &mo.optim, seed, &mut trace);
            setup.normalise_signs(&mut theta);
            let (ll, lp) = (setup.log_likelihood(&theta), setup.log_prior(&theta));
            let est = MsEstimate {
                setup,
                theta,
                log_posterior: ll + lp,
                log_likelihood: ll,
                log_prior: lp,
                iterations: trace.find_mode_calls,
            };
            println!("Number of free parameters: {}", est.theta.len());
            println!(
                "Posterior mode: log posterior {:.6}, log likelihood {:.6}, log prior {:.6} ({} mode searches)",
                est.log_posterior, est.log_likelihood, est.log_prior, est.iterations
            );
            write_estimation_files(dir, &tag, &est)?;
            ms_files::write_est_final(
                &dir.join(format!("est_final_{tag}.out")),
                &format!("init_{ftag}.dat"),
                &est,
                &start,
                &trace,
            )?;
            std::fs::write(out.join("ms_estimation.json"), serde_json::to_string_pretty(&estimate_json(&est))?)?;
        }
        "ms_compute_probabilities" => {
            println!("MS-SBVAR Compute Regime Probabilities");
            let est = ms_files::load_estimate(dir, &ftag)?;
            let (r, f) = est.filtered();
            if !oflag(o, "filtered_probabilities")
                && (oflag(o, "real_time_smoothed") || oflag(o, "real_time_smoothed_probabilities"))
            {
                // Dynare writes real_time_smoothed_<tag>.out (no plot): row k = P(s_k | Y_c), c = k..T
                let m = super::hamilton::real_time_smoothed(&f, &r.q);
                let mut s = String::new();
                for k in 0..m.nrows() {
                    for j in 0..m.ncols() {
                        if m[(k, j)].is_nan() {
                            s.push_str(", ");
                        } else {
                            s.push_str(&format!("{:.6}, ", m[(k, j)]));
                        }
                    }
                    s.push('\n');
                }
                std::fs::write(dir.join(format!("real_time_smoothed_{tag}.out")), s)?;
                return Ok(());
            }
            let (label, m) = if oflag(o, "filtered_probabilities") {
                ("filtered", f.filtered.clone())
            } else {
                ("smoothed", super::hamilton::smooth(&f, &r.q))
            };
            write_probabilities(&dir.join(format!("{label}_{tag}.out")), &m)?;
            let sizes: Vec<usize> = est.setup.chains.iter().map(|c| c.n).collect();
            super::ms_graphs::probabilities(&m, &sizes, o, dir, fname, &tag)?;
            std::fs::write(
                out.join(format!("ms_{label}_probabilities.json")),
                serde_json::to_string_pretty(
                    &json!({ "probabilities": mat_rows(&m), "note": "rows t = 0..T, columns = regimes (chain 1 slowest)" }),
                )?,
            )?;
        }
        "ms_compute_mdd" => {
            println!("MS-SBVAR Marginal Data Density");
            // set_ms_estimation_file(file_tag), set_ms_simulation_file: both files must exist; the C code
            // then runs with -ft simulation_file_tag, i.e. reads est_final_<simulation_file_tag>.out
            let est = ms_files::load_estimate(dir, &ftag)?;
            let sim_tag = ostr(o, "simulation_file_tag").unwrap_or_else(|| ftag.clone());
            let sim = dir.join(format!("simulation_{sim_tag}.out"));
            if !sim.is_file() {
                bail!("ERROR: Could not find Metropolis Hastings file: simulation_{sim_tag}.out");
            }
            let est = if sim_tag == ftag { est } else { ms_files::load_estimate(dir, &sim_tag)? };
            let mo = mdd_options(o);
            println!("Using the posterior draws of {}", sim.display());
            let (draws, lps) = read_simulation_lp(&sim, est.theta.len())?;
            let res = mdd::mdd(&est.setup, &est.theta, &draws, Some(&lps), &mo, seed)?;
            println!("Muller log(mdd) = {:e}\nBridge log(mdd) = {:e}", res.muller, res.bridge);
            let text = format!(
                "tag: {tag}\n{}\nnproposal = {}  nposterior = {}\n\nWaggoner-Zha\nlog(mdd) = {:e}   log(cut) = minus infinity in_proposal = {}   in_posterior = {}\n\nMuller log(mdd) = {:e}\nnproposal = {}  nposterior = {}   in_P1 = {}   in_P2 = {}\n\nBridge log(mdd) = {:e}\n",
                res.proposal_info,
                res.n_proposal,
                res.n_posterior,
                res.wz,
                res.n_proposal,
                res.n_posterior,
                res.muller,
                res.n_proposal,
                res.n_posterior,
                res.in_p.0,
                res.in_p.1,
                res.bridge
            );
            std::fs::write(dir.join(format!("mdd_t{}_{tag}.out", mo.proposal_type)), text)?;
            std::fs::write(
                out.join("ms_mdd.json"),
                serde_json::to_string_pretty(
                    &json!({ "mueller_log_mdd": res.muller, "bridged_log_mdd": res.bridge, "proposal_type": mo.proposal_type }),
                )?,
            )?;
        }
        "ms_simulation" => run_simulation(o, dir, &ftag, &tag, seed)?,
        "ms_irf" | "ms_forecast" | "ms_variance_decomposition" => {
            run_inference(name, o, &mf.varobs, dir, fname, &ftag, &tag, seed)?
        }
        _ => bail!("unexpected ms command {name}"),
    }
    Ok(())
}

/// `filtered_*.out` / `smoothed_*.out`: one row per `t = 0..T`, 6 decimals.
pub fn write_probabilities(path: &Path, m: &Mat) -> Result<()> {
    let mut s = String::new();
    for t in 0..m.nrows() {
        for j in 0..m.ncols() {
            s.push_str(&format!("{:.6} ", m[(t, j)]));
        }
        s.push('\n');
    }
    std::fs::write(path, s).with_context(|| format!("writing {}", path.display()))
}

/// `simulation_*.out`: rows `log posterior, log likelihood, free parameters`.
pub fn read_simulation(path: &Path, nfree: usize) -> Result<Mat> {
    Ok(read_simulation_lp(path, nfree)?.0)
}

/// `simulation_*.out`: the free parameters and the log posterior (column 0).
pub fn read_simulation_lp(path: &Path, nfree: usize) -> Result<(Mat, Vec<f64>)> {
    let text = std::fs::read_to_string(path)?;
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect())
        .collect();
    if rows.is_empty() || rows[0].len() != nfree + 2 {
        bail!(
            "{}: expected {} columns (log posterior, log likelihood, {} free parameters)",
            path.display(),
            nfree + 2,
            nfree
        );
    }
    Ok((Mat::from_fn(rows.len(), nfree, |i, j| rows[i][j + 2]), rows.iter().map(|r| r[0]).collect()))
}

pub fn mode_of(e: &MsEstimate) -> &[f64] {
    &e.theta
}

/// `ms_simulation`: random-walk Metropolis draws of the posterior -> `simulation_<tag>.out`.
fn run_simulation(o: &Options, dir: &Path, ftag: &str, tag: &str, seed: u64) -> Result<()> {
    println!("MS-SBVAR Simulation");
    let est = ms_files::load_estimate(dir, ftag)?;
    let replic = onum(o, "mh_replic").map(|x| x as usize).unwrap_or(10000);
    let thin = onum(o, "thinning_factor").map(|x| x as usize).unwrap_or(1).max(1);
    let drop = onum(o, "drop").map(|x| x as usize).unwrap_or(replic * thin / 10);
    let adapt = onum(o, "adaptive_mh_draws").map(|x| x as usize).unwrap_or(30000);
    let (draws, lps) = mdd::rwmh(&est.setup, &est.theta, replic, drop, thin, adapt, seed);
    let mut s = String::new();
    for i in 0..draws.nrows() {
        let th: Vec<f64> = (0..draws.ncols()).map(|j| draws[(i, j)]).collect();
        let ll = est.setup.log_likelihood(&th);
        s.push_str(&format!("{:.7e} {:.7e}", lps[i], ll));
        for x in th {
            s.push_str(&format!(" {x:.7e}"));
        }
        s.push('\n');
    }
    std::fs::write(dir.join(format!("simulation_{tag}.out")), s)?;
    if oflag(o, "save_draws") {
        // draws_<tag>.out: one line per draw with Q, A0, A+, Zeta of every regime (layout of est_flat_*.out)
        let (mut head, mut body) = (Vec::new(), String::new());
        for i in 0..draws.nrows() {
            let th: Vec<f64> = (0..draws.ncols()).map(|j| draws[(i, j)]).collect();
            if let Some((h, v)) = ms_estimation::flat_row(&est.setup, &th) {
                head = h;
                body.push_str(&v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(" "));
                body.push_str(" \n");
            }
        }
        std::fs::write(dir.join(format!("draws_header_{tag}.out")), head.join(" ") + " \n")?;
        std::fs::write(dir.join(format!("draws_{tag}.out")), body)?;
    }
    Ok(())
}

fn write_blocks(path: &Path, m: &Mat, rows_per_block: usize) -> Result<()> {
    let mut s = String::new();
    for t in 0..m.nrows() {
        for j in 0..m.ncols() {
            s.push_str(&format!("{:.6} ", m[(t, j)]));
        }
        s.push('\n');
        if (t + 1) % rows_per_block == 0 {
            s.push('\n');
        }
    }
    std::fs::write(path, s).with_context(|| format!("writing {}", path.display()))
}

/// `ms_irf`, `ms_variance_decomposition`, `ms_forecast` of the latest `ms_estimation` of the run.
fn run_inference(
    name: &str,
    o: &Options,
    names: &[String],
    dir: &Path,
    fname: &str,
    ftag: &str,
    tag: &str,
    seed: u64,
) -> Result<()> {
    use super::ms_sim::{self, Kind, Mode};
    let (title, sub) = match name {
        "ms_irf" => ("Impulse Response Function", "IRF"),
        "ms_forecast" => ("Forecasts", "Forecast"),
        _ => ("Variance Decomposition", "Variance_Decomposition"),
    };
    println!("MS-SBVAR {title}");
    let est = ms_files::load_estimate(dir, ftag)?;
    let set = &est.setup;
    let odir = dir.join(tag).join(sub);
    std::fs::create_dir_all(&odir)?;
    let horizon = onum(o, "horizon").map(|x| x as usize).unwrap_or(12);
    let thin = onum(o, "thinning_factor").map(|x| x as usize).unwrap_or(1).max(1);
    let regimes = oflag(o, "regimes");
    let regime = onum(o, "regime").map(|x| x as usize).filter(|&x| x > 0);
    let filtered = oflag(o, "filtered_probabilities");
    if regimes as usize + regime.is_some() as usize + filtered as usize > 1 {
        bail!("{name}: only one of filtered_probabilities, regime and regimes may be passed");
    }
    let band = ovec(o, "error_band_percentiles");
    if let Some(x) = band.iter().flatten().find(|x| !(**x > 0.0 && **x < 1.0)) {
        bail!("{name}: error_band_percentiles must lie in (0,1), got {x}");
    }
    let pcts = if oflag(o, "median") { vec![0.5] } else { band.clone().unwrap_or_else(|| vec![0.16, 0.5, 0.84]) };
    let nreg = set.nstates;
    if let Some(k) = regime {
        if k > nreg {
            bail!("{name}: regime {k} exceeds the number of regimes {nreg}");
        }
    }
    // parameter vectors: the estimate, `free_parameters`, or the posterior draws of ms_simulation
    let thetas: Vec<Vec<f64>> = if oflag(o, "parameter_uncertainty") {
        let stag = ostr(o, "simulation_file_tag").unwrap_or_else(|| ftag.to_string());
        let f = dir.join(format!("simulation_{stag}.out"));
        if !f.is_file() {
            bail!("{name}: parameter_uncertainty requires ms_simulation (no {})", f.display());
        }
        let d = read_simulation(&f, est.theta.len())?;
        (0..d.nrows()).step_by(thin).map(|i| d.row(i).iter().cloned().collect()).collect()
    } else if let Some(v) = ovec(o, "free_parameters") {
        if v.len() != est.theta.len() {
            bail!("{name}: free_parameters has {} elements, the model has {}", v.len(), est.theta.len());
        }
        vec![v]
    } else {
        vec![est.theta.clone()]
    };
    let pu = oflag(o, "parameter_uncertainty");
    let modes: Vec<(String, Mode, Option<usize>)> = if regimes {
        (0..nreg).map(|k| (format!("regime_{k}"), Mode::Regime(k), Some(k))).collect()
    } else if let Some(k) = regime {
        vec![(format!("regime_{}", k - 1), Mode::Regime(k - 1), Some(k - 1))]
    } else if filtered {
        vec![("filtered".to_string(), Mode::Filtered, None)]
    } else {
        vec![("ergodic".to_string(), Mode::Ergodic, None)]
    };
    for (label, mode, start) in modes {
        match name {
            "ms_irf" => {
                let m = ms_sim::compute(set, &thetas, &mode, Kind::Irf, horizon, &pcts)?;
                write_blocks(&odir.join(format!("ir_percentiles_{label}_{tag}.out")), &m, horizon)?;
                super::ms_graphs::blocks("IRF", &label, &m, set.nvar, horizon, names, o, dir, fname, odir.clone())?;
            }
            "ms_variance_decomposition" => {
                let bands = !oflag(o, "no_error_bands");
                let pc = band.clone().unwrap_or_else(|| vec![0.16, 0.5, 0.84]);
                let (kind, stem, rows) = if bands {
                    (Kind::Vd, "var_decomp_percentiles", horizon)
                } else {
                    (Kind::VdMean, "var_decomp_mean", horizon)
                };
                let m = ms_sim::compute(set, &thetas, &mode, kind, horizon, &pc)?;
                // the regime-specific files are named var_decomp_*_regime_<k>_<tag>.out; the others _ergodic_/_filtered_
                write_blocks(&odir.join(format!("{stem}_{label}_{tag}.out")), &m, rows)?;
                super::ms_graphs::blocks(
                    "VarianceDecomposition",
                    &label,
                    &m,
                    set.nvar,
                    horizon,
                    names,
                    o,
                    dir,
                    fname,
                    odir.clone(),
                )?;
            }
            _ => {
                let paths = if pu {
                    onum(o, "shocks_per_parameter").map(|x| x as usize).unwrap_or(10)
                } else {
                    onum(o, "shock_draws").map(|x| x as usize).unwrap_or(10000)
                };
                let m = ms_sim::forecast(set, &thetas, start, horizon, paths.max(1), &pcts, start.is_none(), seed)?;
                let dobs = onum(o, "data_obs_nbr").map(|x| x as usize).unwrap_or(0).min(set.y.nrows());
                let n = set.nvar;
                let mut full = Mat::zeros(pcts.len() * (dobs + horizon), n);
                for k in 0..pcts.len() {
                    for r in 0..dobs {
                        for v in 0..n {
                            full[(k * (dobs + horizon) + r, v)] = set.y[(set.y.nrows() - dobs + r, v)];
                        }
                    }
                    for r in 0..horizon {
                        for v in 0..n {
                            full[(k * (dobs + horizon) + dobs + r, v)] = m[(k * horizon + r, v)];
                        }
                    }
                }
                let lbl = if start.is_some() {
                    format!("regime_{label}", label = label.trim_start_matches("regime_"))
                } else {
                    String::new()
                };
                let file = if lbl.is_empty() {
                    format!("forecasts_percentiles_{tag}.out")
                } else {
                    format!("forecasts_percentiles_{lbl}_{tag}.out")
                };
                write_blocks(&odir.join(file), &full, dobs + horizon)?;
                super::ms_graphs::forecast(
                    if label.is_empty() { "ergodic" } else { &label },
                    &full,
                    dobs + horizon,
                    dobs,
                    names,
                    o,
                    dir,
                    fname,
                    odir.clone(),
                )?;
            }
        }
    }
    Ok(())
}
