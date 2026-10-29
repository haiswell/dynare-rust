// SPDX-License-Identifier: GPL-3.0-or-later
//! One MCMC session with Dynare's files : `posterior_sampler.m` with the
//! bookkeeping of `posterior_sampler_initialization.m` (fresh run, `load_mh_file`,
//! `mh_recover`, `mh_initialize_from_previous_mcmc`), `CutSample.m`,
//! `compute_posterior_covariance_matrix.m` (`use_mh_covariance_matrix`,
//! `mh_posterior_mode_estimation`) and the options that go with them. Used by the linear
//! (Kalman) and the nonlinear (particle filter) estimations alike.
//!
//! Every run writes `<dname>/metropolis/<fname>_mh<k>_blck<b>.mat` and the history files
//! ([`super::mh_files`]); a fresh run keeps its draws in memory for the posterior
//! statistics, a continued, recovered or loaded run reads the kept draws of all sessions
//! back from the files (as Dynare does).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};

use super::mcmc::{MhOptions, MhProgress, initial_values, rng_stream};
use super::mh_files::{
    self, ChainStart, ChainWriter, MhPaths, MhRecord, RngState, SessionIo, WriterStart, cut_sample, max_nruns,
};
use super::samplers::{SamplerOptions, SamplingMethod, posterior_sampler_full, prior_initial_values};
use super::{Bounds, Chain, Objective, Prior, prior_draw};
use crate::linalg::{Mat, chol_upper};
use crate::modfile::{OptValue, Options, opt};

/// `options_.posterior_max_subsample_draws` (default 1200), global like Dynare's `options_`.
static POSTERIOR_MAX_SUBSAMPLE_DRAWS: AtomicUsize = AtomicUsize::new(1200);

/// `options_.posterior_max_subsample_draws` (`set_number_of_subdraws.m`).
pub fn posterior_max_subsample_draws() -> usize {
    POSTERIOR_MAX_SUBSAMPLE_DRAWS.load(Ordering::Relaxed)
}

pub fn set_posterior_max_subsample_draws(n: usize) {
    POSTERIOR_MAX_SUBSAMPLE_DRAWS.store(n, Ordering::Relaxed);
}

/// `mh_initialize_from_previous_mcmc` and its `_directory`, `_record`, `_prior`.
#[derive(Clone, Debug, Default)]
pub struct PreviousMcmc {
    pub directory: Option<PathBuf>,
    pub record: Option<PathBuf>,
    pub prior: Option<PathBuf>,
}

/// The estimation options of the MCMC files.
#[derive(Clone, Debug)]
pub struct MhFileOptions {
    pub load_mh_file: bool,
    pub mh_recover: bool,
    pub load_results_after_load_mh: bool,
    pub use_mh_covariance_matrix: bool,
    pub previous: Option<PreviousMcmc>,
    /// `MaxNumberOfBytes` (1e8): size of an mh file.
    pub max_number_of_bytes: f64,
    pub mh_posterior_mode_estimation: bool,
    pub nodiagnostic: bool,
}

/// The option names handled here (for the drivers' lists of supported options).
pub const OPTIONS: &[&str] = &[
    "load_mh_file",
    "mh_recover",
    "load_results_after_load_mh",
    "use_mh_covariance_matrix",
    "mh_initialize_from_previous_mcmc",
    "mh_initialize_from_previous_mcmc_directory",
    "mh_initialize_from_previous_mcmc_record",
    "mh_initialize_from_previous_mcmc_prior",
    "posterior_max_subsample_draws",
    "mh_posterior_mode_estimation",
    "nodiagnostic",
    "maxnumberofbytes",
    "max_number_of_bytes",
];

fn flag(o: &Options, k: &str) -> bool {
    match opt(o, k) {
        Some(OptValue::Flag) => true,
        Some(OptValue::Num(v)) => *v != 0.0,
        _ => false,
    }
}

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

fn string(o: &Options, k: &str) -> Option<String> {
    let s = match opt(o, k)? {
        OptValue::Str(s) => s.clone(),
        OptValue::List(v) if v.len() == 1 => match &v[0] {
            OptValue::Str(s) => s.clone(),
            _ => return None,
        },
        _ => return None,
    };
    Some(s.trim_matches(|c| c == '\'' || c == '"').to_string())
}

impl MhFileOptions {
    /// From the estimation options (paths relative to `dir`, the .mod's folder);
    /// `use_mh_covariance_matrix` may also come from `posterior_sampler_options`.
    /// Sets `posterior_max_subsample_draws` when given.
    pub fn from_options(o: &Options, so: &SamplerOptions, dir: &Path) -> Result<Self> {
        if let Some(v) = num(o, "posterior_max_subsample_draws") {
            if v < 1.0 {
                bail!("posterior_max_subsample_draws must be a positive integer");
            }
            set_posterior_max_subsample_draws(v as usize);
        }
        let path = |k: &str| string(o, k).map(|s| dir.join(s));
        let previous = flag(o, "mh_initialize_from_previous_mcmc").then(|| PreviousMcmc {
            directory: path("mh_initialize_from_previous_mcmc_directory"),
            record: path("mh_initialize_from_previous_mcmc_record"),
            prior: path("mh_initialize_from_previous_mcmc_prior"),
        });
        if let Some(p) = &previous
            && p.directory.is_none()
            && p.record.is_none()
        {
            bail!("mh_initialize_from_previous_mcmc needs mh_initialize_from_previous_mcmc_directory or _record");
        }
        let mh_recover = flag(o, "mh_recover");
        Ok(MhFileOptions {
            load_mh_file: flag(o, "load_mh_file") || mh_recover,
            mh_recover,
            load_results_after_load_mh: flag(o, "load_results_after_load_mh"),
            use_mh_covariance_matrix: flag(o, "use_mh_covariance_matrix") || so.use_mh_covariance_matrix,
            previous,
            max_number_of_bytes: num(o, "maxnumberofbytes")
                .or(num(o, "max_number_of_bytes"))
                .unwrap_or(mh_files::MAX_NUMBER_OF_BYTES),
            mh_posterior_mode_estimation: flag(o, "mh_posterior_mode_estimation"),
            nodiagnostic: flag(o, "nodiagnostic"),
        })
    }
}

/// `options_.posterior_sampler_options.posterior_sampling_method` (`record.Sampler`).
pub fn sampler_name(so: &SamplerOptions) -> &'static str {
    match so.method {
        SamplingMethod::RandomWalk => "random_walk_metropolis_hastings",
        SamplingMethod::Independent => "independent_metropolis_hastings",
        SamplingMethod::Slice => "slice",
        SamplingMethod::TaRB => "tailored_random_block_metropolis_hastings",
    }
}

/// Inputs of [`run`].
pub struct SessionInputs<'a> {
    pub obj: &'a dyn Objective,
    /// Mode (or start) and proposal covariance (`invhess`).
    pub xparam1: &'a [f64],
    pub invhess: &'a Mat,
    /// MCMC bounds (`set_mcmc_prior_bounds.m`).
    pub bounds: &'a Bounds,
    /// `replic` (0 = only load), `nblocks`, `jscale`, `seed`, `prefetch`, ...
    pub mh: MhOptions,
    pub sampler: &'a SamplerOptions,
    pub names: &'a [String],
    pub priors: &'a [Prior],
    /// `<dir>/<fname>` (`M_.dname`).
    pub dname: &'a Path,
    pub fname: &'a str,
    pub mh_drop: f64,
    /// Keep the burn-in in memory too (Brooks–Gelman, Geweke, trace plots).
    pub keep_all: bool,
    pub progress: Option<&'a (dyn Fn(&MhProgress) + Sync)>,
    /// Prefix of the console messages (`Estimation::mcmc`, `method_of_moments::mcmc`).
    pub disp: &'a str,
}

/// Result of [`run`].
pub struct SessionOutput {
    /// Draws of all sessions: from `first_stored` (0 with `keep_all`, else the first kept
    /// draw) to the end, `total_draws` per chain.
    pub chains: Vec<Chain>,
    pub record: MhRecord,
    /// Acceptance ratio per chain (this session; over all sessions when continued).
    pub acceptance: Vec<f64>,
    /// `record.FunctionEvalPerIteration`.
    pub evals_per_draw: Vec<f64>,
    /// Pre-fetching width (1 when nothing was run).
    pub prefetch: usize,
    pub init_scale_factor: f64,
    /// Draws per chain run in this session.
    pub new_draws: usize,
    /// The proposal's `jscale` used (from the previous record when continuing).
    pub jscale: Vec<f64>,
    pub seconds: f64,
}

fn log_line(paths: &MhPaths, s: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(paths.log()) {
        let _ = f.write_all(s.as_bytes());
    }
}

/// `use_mh_covariance_matrix` with `load_mh_file` (`check_posterior_sampler_options.m` ->
/// `compute_posterior_covariance_matrix.m`): the covariance of the kept draws of the previous
/// MCMC, also saved with its mode as `<out_dir>/<fname>_mh_mode.mat`.
pub fn previous_mh_covariance(dname: &Path, fname: &str, out_dir: &Path, names: &[String]) -> Result<(Mat, Vec<f64>)> {
    let paths = MhPaths::new(dname, fname);
    let (rec, _) = mh_files::load_last_record(&paths)?;
    let (_mean, cov, mode, kernel) = mh_files::compute_mh_covariance_matrix(&paths, &rec)?;
    let p = mh_files::write_mh_mode(out_dir, fname, &mode, &cov, kernel, names)?;
    println!("use_mh_covariance_matrix: proposal covariance from the previous MCMC ({} saved).", p.display());
    Ok((cov, mode))
}

/// `mh_posterior_mode_estimation` (`compute_posterior_covariance_matrix.m` + `fill_mh_mode.m`):
/// the draw with the highest posterior kernel and the posterior covariance of the kept draws,
/// saved as `<out_dir>/<fname>_mh_mode.mat`. Returns (mode, sd, kernel).
pub fn posterior_mode_from_mcmc(
    dname: &Path,
    fname: &str,
    out_dir: &Path,
    names: &[String],
) -> Result<(Vec<f64>, Vec<f64>, f64)> {
    let paths = MhPaths::new(dname, fname);
    let (rec, _) = mh_files::load_last_record(&paths)?;
    let (_mean, cov, mode, kernel) = mh_files::compute_mh_covariance_matrix(&paths, &rec)?;
    mh_files::write_mh_mode(out_dir, fname, &mode, &cov, kernel, names)?;
    let sd = (0..mode.len()).map(|i| cov[(i, i)].max(0.0).sqrt()).collect();
    Ok((mode, sd, kernel))
}

/// The previous run's `<fname>_results.json` (`load_results_after_load_mh`), if any.
pub fn load_previous_results(out_dir: &Path, fname: &str) -> Option<serde_json::Value> {
    let p = out_dir.join(format!("{fname}_results.json"));
    let s = std::fs::read_to_string(p).ok()?;
    serde_json::from_str(&s).ok()
}

/// Runs (or loads) the MCMC as `posterior_sampler.m` and `CutSample.m` do.
pub fn run(inp: SessionInputs, fo: &MhFileOptions) -> Result<SessionOutput> {
    let paths = Arc::new(MhPaths::new(inp.dname, inp.fname));
    std::fs::create_dir_all(&paths.folder).with_context(|| format!("creating {}", paths.folder.display()))?;
    let npar = inp.xparam1.len();
    let mut mh = inp.mh.clone();
    let mx = max_nruns(npar, fo.max_number_of_bytes);
    mh_files::write_prior_definition(inp.dname, inp.names, inp.priors, &mh.jscale)?;
    let disp = inp.disp;
    let uses_hessian =
        inp.sampler.method != SamplingMethod::Slice || inp.sampler.slice.initialize_with_mode || fo.load_mh_file;
    let t0 = Instant::now();

    // ---- posterior_sampler_initialization.m -------------------------------------------------
    // (record, starts, writers, proposal chol, accepted draws before this session, continued?)
    let mut prior_acc: Vec<f64>;
    let continued: bool;
    let mut record: MhRecord;
    let starts: Vec<ChainStart>;
    let mut writer_starts: Vec<Option<WriterStart>>;
    let mut proposal_chol: Option<Mat> = None;
    let mut factor = mh.init_scale_factor;
    let mut eval_secs = 0.0;
    if mh.replic == 0 {
        if !fo.load_mh_file {
            bail!("mh_replic = 0: nothing to sample (use load_mh_file to load a previous MCMC)");
        }
        let (r, _) = mh_files::load_last_record(&paths)?;
        return finish(&paths, r, None, inp, fo, &mh, vec![], 1, factor, true, t0);
    } else if fo.mh_recover {
        println!("{disp}: Recover mode!");
        let (r, _) = mh_files::load_last_record(&paths)?;
        record = r;
        let nb = record.nblck;
        mh.nblocks = nb;
        let last = *record.mh_draws.last().ok_or_else(|| anyhow!("mh_recover: empty mh history"))?;
        if last[0] as usize != mh.replic {
            println!("\n{disp}: You cannot specify a different mh_replic than in the chain you are trying to recover");
            println!("{disp}: I am resetting mh_replic to {}", last[0]);
            mh.replic = last[0] as usize;
        }
        let mx = match record.max_nruns.last() {
            Some(&m) if m.is_finite() && m as usize != mx => {
                println!("{disp}: I am resetting MAX_nruns to {m}");
                m as usize
            }
            _ => mx,
        };
        let k = record.mh_draws.len();
        let old = k > 1;
        let prev_total: usize = record.mh_draws[..k - 1].iter().map(|r| r[0] as usize).sum();
        let prev_files: usize = record.mh_draws[..k - 1].iter().map(|r| r[1] as usize).sum();
        let prev_lines = if old { record.mh_draws[k - 2][2] as usize } else { 0 };
        set_proposal_to_previous(&mut record, fo, inp.sampler, &mut mh, &mut proposal_chol, disp)?;
        // posterior_sampler_initialization.m: as many mh files as expected means nothing crashed
        let expected_per_block: usize = record.mh_draws.iter().map(|r| r[1] as usize).sum();
        let existing: usize = (1..=nb).map(|b| paths.mh_files_of_block(b).len()).sum();
        if expected_per_block > prev_files && existing == expected_per_block * nb {
            println!("{disp}: It appears that you don't need to use the mh_recover option!");
            println!("                  You have to edit the mod file and remove the mh_recover option");
            println!("                  in the estimation command.");
            bail!("{disp}: mh_recover option not required!");
        }
        if uses_hessian {
            // posterior_sampler_core.m: record.ProposalCovariance = d, ProposalScaleVec = jscale
            let d = proposal_chol.clone().or_else(|| chol_upper(inp.invhess));
            record.proposal_scale_vec = d.as_ref().map(|_| mh.jscale.clone());
            record.proposal_covariance = d;
        }
        let mut st = Vec::with_capacity(nb);
        writer_starts = Vec::with_capacity(nb);
        prior_acc = vec![0.0; nb];
        for b in 0..nb {
            // Draws of this chain on disk (all sessions), in its contiguous files.
            let files = paths.mh_files_of_block(b + 1);
            let mut rows = 0usize;
            let mut last_file = 0usize;
            for &f in &files {
                if f != last_file + 1 {
                    break;
                }
                rows += mh_files::MhFile::read_parts(&paths.mh_file(f, b + 1), false)?.rows();
                last_file = f;
            }
            let done = rows.saturating_sub(prev_total);
            if done > 0 {
                let f = mh_files::MhFile::read(&paths.mh_file(last_file, b + 1))?;
                let seed = f.seed(last_file).cloned().ok_or_else(|| {
                    anyhow!("{}: no LastSeeds.file{last_file}", paths.mh_file(last_file, b + 1).display())
                })?;
                let n = f.rows();
                prior_acc[b] = f.accepted_draws_this_chain;
                let nruns = mh.replic.saturating_sub(done);
                if nruns == 0 {
                    println!("{disp}: Chain {} is complete!", b + 1);
                    writer_starts.push(None);
                } else {
                    println!("{disp}: Chain {} is not complete!", b + 1);
                    writer_starts.push(Some(WriterStart { new_file: last_file + 1, first_line: 1, nruns }));
                }
                st.push(ChainStart { x: f.row(n - 1).to_vec(), lp: f.logpo2[n - 1], rng: seed.to_chacha(b), nruns });
            } else {
                println!("{disp}: Chain {} is not complete!", b + 1);
                let (x, lp) = if old {
                    (record.last_parameters.row(b).iter().copied().collect(), record.last_log_post[b])
                } else {
                    (record.initial_parameters.row(b).iter().copied().collect(), record.initial_log_post[b])
                };
                let seed = record
                    .initial_seeds
                    .get(b)
                    .cloned()
                    .unwrap_or_else(|| RngState::from_chacha(&rng_stream(mh.seed, b as u64 + 1)));
                let ws = if old && prev_lines < mx {
                    WriterStart { new_file: prev_files, first_line: prev_lines + 1, nruns: mh.replic }
                } else {
                    WriterStart { new_file: prev_files + 1, first_line: 1, nruns: mh.replic }
                };
                writer_starts.push(Some(ws));
                st.push(ChainStart { x, lp, rng: seed.to_chacha(b), nruns: mh.replic });
            }
        }
        starts = st;
        continued = true;
        let mut wr = Vec::with_capacity(nb);
        for (b, ws) in writer_starts.iter().enumerate() {
            wr.push(match ws {
                Some(s) => Some(ChainWriter::new(paths.clone(), b + 1, npar, mx, *s)?),
                None => None,
            });
        }
        return sample(
            paths,
            record,
            starts,
            wr,
            proposal_chol,
            prior_acc,
            inp,
            fo,
            mh,
            factor,
            continued,
            t0,
            eval_secs,
        );
    } else if fo.load_mh_file {
        println!("{disp}: I am loading past Metropolis-Hastings simulations...");
        let (r, _) = mh_files::load_last_record(&paths)?;
        record = r;
        if record.mcmc_concluded_successfully == 0.0 {
            bail!("{disp}: You are trying to load an MCMC that did not finish successfully. Please use mh_recover.");
        }
        record.mcmc_concluded_successfully = 0.0;
        if paths.mh_files_of_block(1).is_empty() {
            bail!("{disp}: I cannot find any MH file to load here!");
        }
        if record.nblck != mh.nblocks {
            println!("{disp}: The specified number of blocks doesn't match with the previous number of blocks!");
            println!(
                "{disp}: You declared {} blocks, but the previous number of blocks was {}.",
                mh.nblocks, record.nblck
            );
            println!("{disp}: I will run the Metropolis-Hastings with {} blocks.", record.nblck);
            mh.nblocks = record.nblck;
        }
        log_line(
            &paths,
            &format!(
                "\n% Session {}.\n \n  Number of blocks...............: {}\n  Number of simulations per block: {}\n \n",
                record.mh_draws.len() + 1,
                mh.nblocks,
                mh.replic
            ),
        );
        let (lfn, lln) = (record.last_file_number, record.last_line_number);
        let (new_file, first_line) = if lln < mx { (lfn, lln + 1) } else { (lfn + 1, 1) };
        if record.last_seeds.len() < record.nblck {
            println!("{disp}: no random-generator state in the mh history; the chains continue on new streams.");
        }
        starts = (0..record.nblck)
            .map(|b| {
                let seed = record.last_seeds.get(b).cloned();
                ChainStart {
                    x: record.last_parameters.row(b).iter().copied().collect(),
                    lp: record.last_log_post[b],
                    rng: seed.map_or_else(|| rng_stream(mh.seed ^ 0x5eed, b as u64 + 1), |s| s.to_chacha(b)),
                    nruns: mh.replic,
                }
            })
            .collect();
        set_proposal_to_previous(&mut record, fo, inp.sampler, &mut mh, &mut proposal_chol, disp)?;
        if uses_hessian {
            // posterior_sampler_core.m: record.ProposalCovariance = d, ProposalScaleVec = jscale
            let d = proposal_chol.clone().or_else(|| chol_upper(inp.invhess));
            record.proposal_scale_vec = d.as_ref().map(|_| mh.jscale.clone());
            record.proposal_covariance = d;
        }
        let previous = record.total_draws();
        print!("{disp}: I am writing a new mh-history file... ");
        // new history row (posterior_sampler_initialization.m)
        let in_last = mx as f64 - lln as f64;
        let to_save = mh.replic as f64 - in_last;
        let nfiles = (to_save / mx as f64).ceil();
        let lines = to_save - (nfiles - 1.0) * mx as f64;
        record.last_file_number = (lfn as f64 + nfiles) as usize;
        record.last_line_number = lines as usize;
        record.mh_draws.push([mh.replic as f64, nfiles, lines]);
        record.max_nruns.push(mx as f64);
        record.initial_seeds = record.last_seeds.clone();
        mh_files::write_new_record(&paths, &record)?;
        println!("Done.");
        if !fo.use_mh_covariance_matrix {
            println!("{disp}: Ok. I have loaded {previous} simulations.\n");
        }
        writer_starts = vec![Some(WriterStart { new_file, first_line, nruns: mh.replic }); record.nblck];
        prior_acc = vec![0.0; record.nblck];
        continued = true;
        eval_secs = time_eval(inp.obj, &starts[0].x);
    } else {
        // fresh run (possibly initialized from a previous MCMC)
        println!("{disp}: {}", if mh.nblocks > 1 { "Multiple chains mode." } else { "One Chain mode." });
        let deleted = paths.delete_all()?;
        if deleted > 0 {
            println!("{disp}: Old mh-files successfully erased!");
        }
        if paths.log().exists() {
            println!("{disp}: Old metropolis.log file successfully erased!");
            println!("{disp}: Creation of a new metropolis.log file.");
        }
        let _ = std::fs::remove_file(paths.log());
        log_line(
            &paths,
            &format!(
                "% MH log file (Dynare-rust).\n \n\n% Session 1.\n \n  Number of blocks...............: {}\n  Number of simulations per block: {}\n \n",
                mh.nblocks, mh.replic
            ),
        );
        let d = if uses_hessian {
            Some(
                chol_upper(inp.invhess)
                    .ok_or_else(|| anyhow!("posterior_sampler: the proposal covariance is not positive definite"))?,
            )
        } else {
            None
        };
        let t = Instant::now();
        if mh.nblocks > 1 || fo.previous.is_some() {
            println!("{disp}: Searching for initial values...");
        }
        let (ix2, ilp, seeds): (Vec<Vec<f64>>, Vec<f64>, Vec<RngState>) = if let Some(prev) = &fo.previous {
            let (x, l, s, nb) = initialize_from_previous(prev, &inp, &mh, d.as_ref(), &mut factor, disp)?;
            mh.nblocks = nb;
            (x, l, s)
        } else {
            let (x, l) = match &d {
                Some(d) => {
                    let (x, l, f) = initial_values(inp.obj, inp.xparam1, d, inp.bounds, &mh)?;
                    factor = f;
                    (x, l)
                }
                None => prior_initial_values(inp.obj, inp.xparam1, inp.sampler.priors.as_deref(), inp.bounds, &mh)?,
            };
            let s = (0..mh.nblocks).map(|b| RngState::from_chacha(&rng_stream(mh.seed, b as u64 + 1))).collect();
            (x, l, s)
        };
        eval_secs = t.elapsed().as_secs_f64() / ix2.len().max(1) as f64;
        if mh.nblocks > 1 || fo.previous.is_some() {
            println!("{disp}: Initial values found!\n");
        } else {
            println!(
                "{disp}: Initialization at the {} mode.\n",
                if inp.sampler.method == SamplingMethod::Slice { "prior" } else { "posterior" }
            );
        }
        let nb = ix2.len();
        let nfiles = mh.replic.div_ceil(mx);
        let lines = mh.replic - (nfiles - 1) * mx;
        record = MhRecord {
            sampler: sampler_name(inp.sampler).to_string(),
            nblck: nb,
            mh_draws: vec![[mh.replic as f64, nfiles as f64, lines as f64]],
            max_nruns: vec![mx as f64],
            acceptance_ratio: vec![0.0; nb],
            function_eval_per_iteration: vec![f64::NAN; nb],
            initial_seeds: seeds.clone(),
            last_seeds: vec![],
            initial_parameters: Mat::from_fn(nb, npar, |i, j| ix2[i][j]),
            initial_log_post: ilp.clone(),
            last_parameters: Mat::zeros(nb, npar),
            last_log_post: vec![0.0; nb],
            last_file_number: nfiles,
            last_line_number: lines,
            mcmc_concluded_successfully: 0.0,
            proposal_scale_vec: d.as_ref().map(|_| mh.jscale.clone()),
            proposal_covariance: d.clone(),
            keeped_draws: None,
        };
        let id = mh_files::write_new_record(&paths, &record)?;
        println!("{disp}: Write details about the MCMC... Ok!");
        println!("{disp}: Details about the MCMC are available in {}\n", paths.history(id).display());
        starts = (0..nb)
            .map(|b| ChainStart { x: ix2[b].clone(), lp: ilp[b], rng: seeds[b].to_chacha(b), nruns: mh.replic })
            .collect();
        writer_starts = vec![Some(WriterStart { new_file: 1, first_line: 1, nruns: mh.replic }); nb];
        prior_acc = vec![0.0; nb];
        proposal_chol = d;
        continued = false;
    }
    let mut wr = Vec::with_capacity(writer_starts.len());
    for (b, ws) in writer_starts.iter().enumerate() {
        wr.push(match ws {
            Some(s) => Some(ChainWriter::new(paths.clone(), b + 1, npar, mx, *s)?),
            None => None,
        });
    }
    sample(paths, record, starts, wr, proposal_chol, prior_acc, inp, fo, mh, factor, continued, t0, eval_secs)
}

fn time_eval(obj: &dyn Objective, x: &[f64]) -> f64 {
    let t = Instant::now();
    let _ = obj.eval(x);
    t.elapsed().as_secs_f64()
}

/// `set_proposal_density_to_previous_value`: the previous run's `d` and `jscale`, unless
/// `use_mh_covariance_matrix`.
fn set_proposal_to_previous(
    record: &mut MhRecord,
    fo: &MhFileOptions,
    so: &SamplerOptions,
    mh: &mut MhOptions,
    chol: &mut Option<Mat>,
    disp: &str,
) -> Result<()> {
    if !fo.use_mh_covariance_matrix {
        if let (Some(d), Some(js)) = (&record.proposal_covariance, &record.proposal_scale_vec) {
            println!("{disp}: Recovering the previous proposal density.");
            if d.nrows() != mh.jscale.len() || js.len() != mh.jscale.len() {
                bail!(
                    "{disp}: the previous proposal density has {} parameters, the estimation {}",
                    d.nrows(),
                    mh.jscale.len()
                );
            }
            *chol = Some(d.clone());
            mh.jscale = js.clone();
        } else if so.method != SamplingMethod::Slice {
            println!("{disp}: No stored previous proposal density found, continuing with the one implied by the mode.");
        }
    } else {
        println!(
            "{disp}: 'use_mh_covariance_matrix' specified, continuing with proposal density implied by the previous MCMC run."
        );
    }
    let name = sampler_name(so);
    if record.sampler != name {
        println!(
            "WARNING: {disp}: The posterior_sampling_method {name} selected differs from the {} of the original chain. This may create problems with the convergence diagnostics.",
            record.sampler
        );
        record.sampler = name.to_string();
    }
    Ok(())
}

/// `mh_initialize_from_previous_mcmc`: the last draws of a previous MCMC (matched by
/// parameter name); parameters new to this estimation, or outside the bounds, are drawn as
/// fresh initial values. The chains continue the previous random streams.
fn initialize_from_previous(
    prev: &PreviousMcmc,
    inp: &SessionInputs,
    mh: &MhOptions,
    d: Option<&Mat>,
    factor: &mut f64,
    disp: &str,
) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<RngState>, usize)> {
    let (record0, folder0): (MhRecord, PathBuf) = match (&prev.record, &prev.directory) {
        (Some(rf), _) => {
            let vars =
                crate::matfile::read(rf).with_context(|| format!("{disp}: Path to record file is not found!"))?;
            let r = MhRecord::from_mat(vars.get("record").ok_or_else(|| anyhow!("{}: no record", rf.display()))?)?;
            let mfolder = rf.parent().map(Path::to_path_buf).unwrap_or_default();
            (r, mfolder.parent().map(Path::to_path_buf).unwrap_or_default())
        }
        (None, Some(dir)) => {
            let model0 = dir.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
            let p0 = MhPaths { folder: dir.join("metropolis"), model: model0 };
            (mh_files::load_last_record(&p0)?.0, dir.clone())
        }
        _ => bail!("mh_initialize_from_previous_mcmc: no directory or record given"),
    };
    if record0.mcmc_concluded_successfully == 0.0 {
        bail!("{disp}: You are trying to load an MCMC that did not finish successfully. Please use 'mh_recover'!");
    }
    println!("{disp}: Initializing from past Metropolis-Hastings simulations...");
    println!("{disp}: Past MH path {}", folder0.join("metropolis").display());
    println!("{disp}: Past model name {}", folder0.file_name().map(|f| f.to_string_lossy()).unwrap_or_default());
    let prior_file = prev.prior.clone().unwrap_or_else(|| folder0.join("prior").join("definition.mat"));
    let names0 =
        mh_files::read_prior_names(&prior_file).with_context(|| format!("{disp}: Path to prior file is not found!"))?;
    let nb = record0.nblck;
    if nb != mh.nblocks {
        println!("{disp}: The specified number of blocks doesn't match with the previous number of blocks!");
        println!("{disp}: You declared {} blocks, but the previous number of blocks was {nb}.", mh.nblocks);
        println!("{disp}: I will run the Metropolis-Hastings with {nb} block.");
    }
    let npar = inp.names.len();
    let mut newp: Vec<bool> = inp.names.iter().map(|n| !names0.contains(n)).collect();
    let mut ix2 = vec![vec![0.0; npar]; nb];
    for (i, n) in inp.names.iter().enumerate() {
        if let Some(j) = names0.iter().position(|m| m == n) {
            for b in 0..nb {
                ix2[b][i] = record0.last_parameters[(b, j)];
            }
        }
    }
    for x in &ix2 {
        for i in 0..npar {
            if x[i] < inp.bounds.lb[i] || x[i] > inp.bounds.ub[i] {
                newp[i] = true;
            }
        }
    }
    let mut ilp = record0.last_log_post.clone();
    let mut rng = rng_stream(mh.seed, 0);
    let (mut z, mut cand) = (vec![0.0; npar], vec![0.0; npar]);
    let inside = |c: &[f64]| (0..npar).all(|i| c[i] >= inp.bounds.lb[i] && c[i] <= inp.bounds.ub[i]);
    for j in 0..nb {
        let (mut validate, mut init_iter, mut trial) = (false, 0, 1);
        while !validate && trial <= 10 {
            match d {
                Some(d) => {
                    let u = d * (*factor * mh.mh_jscale);
                    super::mcmc::rand_mvn(&mut rng, inp.xparam1, &u, &mut z, &mut cand);
                }
                None => cand = prior_draw(inp.priors, inp.bounds, &mut rng),
            }
            if inside(&cand) {
                for i in 0..npar {
                    if newp[i] {
                        ix2[j][i] = cand[i];
                    }
                }
                ilp[j] = -inp.obj.eval(&ix2[j]).0;
                validate = ilp[j].is_finite();
            }
            init_iter += 1;
            if !validate && init_iter > 100 {
                *factor *= 0.9;
                trial += 1;
            }
        }
        if !validate {
            bail!("{disp}: I'm unable to find a starting value for block {}", j + 1);
        }
    }
    let seeds = (0..nb)
        .map(|b| {
            record0
                .last_seeds
                .get(b)
                .cloned()
                .unwrap_or_else(|| RngState::from_chacha(&rng_stream(mh.seed, b as u64 + 1)))
        })
        .collect();
    Ok((ix2, ilp, seeds, nb))
}

#[allow(clippy::too_many_arguments)]
fn sample(
    paths: Arc<MhPaths>,
    mut record: MhRecord,
    starts: Vec<ChainStart>,
    writers: Vec<Option<ChainWriter>>,
    proposal_chol: Option<Mat>,
    prior_acc: Vec<f64>,
    inp: SessionInputs,
    fo: &MhFileOptions,
    mut mh: MhOptions,
    factor: f64,
    continued: bool,
    t0: Instant,
    eval_secs: f64,
) -> Result<SessionOutput> {
    let nb = starts.len();
    mh.nblocks = nb;
    mh.init_scale_factor = factor;
    // A continued run reads all draws back from the files: keep none in memory.
    if continued {
        mh.store_from = usize::MAX;
    }
    let io = Arc::new(SessionIo::new(Some(starts.clone()), proposal_chol, writers, eval_secs));
    mh.session = Some(io.clone());
    let res = posterior_sampler_full(inp.obj, inp.xparam1, inp.invhess, inp.bounds, &mh, inp.sampler, inp.progress)?;
    let ends = io.ends();
    record.last_seeds.resize(nb, RngState { unifor: vec![], normal: vec![], algo: String::new() });
    if record.last_parameters.nrows() != nb {
        record.last_parameters = Mat::zeros(nb, inp.xparam1.len());
        record.last_log_post = vec![0.0; nb];
    }
    record.acceptance_ratio.resize(nb, 0.0);
    record.function_eval_per_iteration.resize(nb, f64::NAN);
    for (b, e) in ends.iter().enumerate() {
        let e = e.as_ref().ok_or_else(|| anyhow!("MCMC chain {} did not finish", b + 1))?;
        for j in 0..inp.xparam1.len() {
            record.last_parameters[(b, j)] = e.x[j];
        }
        record.last_log_post[b] = e.lp;
        record.last_seeds[b] = e.seed.clone();
        let replic = record.mh_draws.last().map_or(mh.replic as f64, |r| r[0]);
        record.acceptance_ratio[b] = (prior_acc[b] + e.accepted) / replic;
        if e.draws > 0 {
            record.function_eval_per_iteration[b] = e.fevals / e.draws as f64;
        }
    }
    // a chain whose mh file could not be written is an error (the record stays unfinished)
    for b in 0..nb {
        let p = paths.mh_file(record.last_file_number.max(1), b + 1);
        if starts[b].nruns > 0 && !p.exists() {
            bail!(
                "MCMC: the mh file {} was not written (see the errors above); use mh_recover once the problem is fixed",
                p.display()
            );
        }
    }
    record.mcmc_concluded_successfully = 1.0;
    mh_files::update_last_record(&paths, &record)?;
    let new_draws = starts.iter().map(|s| s.nruns).max().unwrap_or(0);
    let mem = if continued { None } else { Some(res.mh.chains) };
    let prefetch = res.mh.prefetch;
    finish(&paths, record, mem, inp, fo, &mh, res.evals_per_draw, prefetch, factor, continued, t0).map(|mut o| {
        o.new_draws = new_draws;
        o
    })
}

/// `CutSample.m` and the draws for the posterior analysis.
#[allow(clippy::too_many_arguments)]
fn finish(
    paths: &MhPaths,
    mut record: MhRecord,
    mem: Option<Vec<Chain>>,
    inp: SessionInputs,
    fo: &MhFileOptions,
    mh: &MhOptions,
    evals: Vec<f64>,
    prefetch: usize,
    factor: f64,
    from_files: bool,
    t0: Instant,
) -> Result<SessionOutput> {
    let disp = inp.disp;
    let kd = cut_sample(&record, inp.mh_drop, fo.max_number_of_bytes);
    let total = record.total_draws();
    let first_draw = 1usize.max((inp.mh_drop * total as f64).floor() as usize + 1);
    let acceptance = if from_files || mem.is_none() {
        mh_files::overall_acceptance_ratio(paths).unwrap_or_else(|_| record.acceptance_ratio.clone())
    } else {
        record.acceptance_ratio.clone()
    };
    // posterior_sampler.m: the diagnostic output (no sampler run with mh_replic = 0)
    if mh.replic > 0 {
        println!();
        println!("{disp}: Number of mh files: {} per block.", record.total_files());
        println!("{disp}: Total number of generated files: {}.", record.total_files() * record.nblck);
        println!("{disp}: Total number of iterations: {total}.");
        println!("{disp}: Current acceptance ratio per chain:");
        let chain_line = |i: usize, v: String| {
            println!("{:55}Chain {}{}: {v}", "", if i < 10 { " " } else { "" }, i);
        };
        for (i, a) in acceptance.iter().enumerate() {
            chain_line(i + 1, format!("{}%", crate::gsa::stats::num2str(100.0 * a)));
        }
        let evals_it: &[f64] = if evals.is_empty() { &record.function_eval_per_iteration } else { &evals };
        if evals_it.iter().copied().fold(f64::NEG_INFINITY, f64::max) > 1.0 {
            println!("{disp}: Current function evaluations per iteration:");
            for (i, e) in evals_it.iter().enumerate() {
                chain_line(i + 1, crate::gsa::stats::num2str(*e));
            }
        }
    }
    println!("{disp}: Total number of MH draws per chain: {total}.");
    println!("{disp}: Total number of generated MH files: {}.", record.total_files());
    println!("{disp}: I'll use mh-files {} to {}.", kd.first_mh_file, record.total_files());
    println!("{disp}: In MH-file number {} I'll start at line {}.", kd.first_mh_file, kd.first_line);
    println!("{disp}: Finally I keep {} draws per chain.\n", total - first_draw + 1);
    record.keeped_draws = Some(kd);
    mh_files::update_last_record(paths, &record)?;
    let chains = match mem {
        Some(c) => c,
        None => {
            let from = if inp.keep_all && crate::estimation::plots::keep_all_draws(record.npar(), total, record.nblck) {
                0
            } else {
                first_draw - 1
            };
            mh_files::read_chains(paths, &record, from, &acceptance)?
        }
    };
    let n = chains.len();
    let jscale = record.proposal_scale_vec.clone().unwrap_or_else(|| mh.jscale.clone());
    Ok(SessionOutput {
        chains,
        acceptance,
        evals_per_draw: if evals.is_empty() { record.function_eval_per_iteration.clone() } else { evals },
        record,
        prefetch,
        init_scale_factor: factor,
        new_draws: 0,
        jscale: if jscale.len() == inp.xparam1.len() { jscale } else { vec![f64::NAN; inp.xparam1.len()] },
        seconds: t0.elapsed().as_secs_f64(),
    })
    .inspect(|o| {
        debug_assert_eq!(o.chains.len(), n);
    })
}
