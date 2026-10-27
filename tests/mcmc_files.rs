//! : MCMC persistence (`<fname>/metropolis/` in Dynare's layout) and continuation on
//! `reference/mcmc/nk_mcmc.mod` (7 parameters, 278 draws per mh file):
//! - `load_mh_file` continues every chain exactly: 3000 + 2000 draws are bit for bit the
//!   draws of one 5000-draw run (generator state saved in the files), with the same
//!   posterior statistics;
//! - `mh_recover` after a crash (missing mh files, first or second session) reproduces the
//!   uninterrupted run;
//! - `load_mh_file` with `mh_replic = 0`, `mh_initialize_from_previous_mcmc`,
//!   `use_mh_covariance_matrix`, `mh_posterior_mode_estimation`;
//! - Dynare 7.1's own files (`reference/mcmc/dynare_run/`, MATLAB-compressed, with MATLAB's
//!   RandStream objects) are read and continued, and the posterior statistics agree with
//!   Dynare's (`reference/mcmc/ref_mcmc_phase1.json`, `dump_mcmc.m`; the reverse direction,
//!   Dynare reading dynare-rust's files, is recorded there too).

use std::path::{Path, PathBuf};

use dynare::estimation::Chain;
use dynare::estimation::mh_files::{self, MhPaths};
use serde_json::Value;

fn reference() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/mcmc")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("dynare_rust_mcmc_files_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    for f in ["nk_mcmc.mod", "nk_model.inc", "nk_data.csv"] {
        std::fs::copy(reference().join(f), d.join(f)).unwrap();
    }
    d
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let p = e.path();
        if p.is_dir() {
            copy_dir(&p, &to.join(e.file_name()));
        } else {
            std::fs::copy(&p, to.join(e.file_name())).unwrap();
        }
    }
}

fn run(dir: &Path, stage: u32, replic: usize, seed: u64) {
    let ro = dynare::driver::RunOptions {
        defines: vec![("STAGE".into(), stage.to_string()), ("REPLIC".into(), replic.to_string())],
        seed,
        quiet: true,
        ..Default::default()
    };
    dynare::driver::run(&dir.join("nk_mcmc.mod"), &ro).unwrap();
}

fn paths(dir: &Path) -> MhPaths {
    MhPaths::new(&dir.join("nk_mcmc"), "nk_mcmc")
}

fn chains(dir: &Path) -> Vec<Chain> {
    let p = paths(dir);
    let (rec, _) = mh_files::load_last_record(&p).unwrap();
    mh_files::read_chains(&p, &rec, 0, &[]).unwrap()
}

fn results(dir: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("nk_mcmc/Output/nk_mcmc_results.json")).unwrap()).unwrap()
}

fn vecf(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn same_draws(a: &[Chain], b: &[Chain], from: usize) {
    assert_eq!(a.len(), b.len());
    for (ca, cb) in a.iter().zip(b) {
        assert_eq!(ca.total_draws, cb.total_draws);
        for t in from..ca.total_draws {
            assert_eq!(ca.draw(t), cb.draw(t), "draw {t}");
            assert_eq!(ca.logpost[t - ca.first_stored], cb.logpost[t - cb.first_stored], "logpost {t}");
        }
    }
}

/// Simulates a crash: removes the mh files `files` of chain `blck` and marks the last session unfinished.
fn crash(dir: &Path, blck: usize, files: std::ops::RangeInclusive<usize>) {
    let p = paths(dir);
    for k in files {
        std::fs::remove_file(p.mh_file(k, blck)).unwrap();
    }
    let (mut rec, i) = mh_files::load_last_record(&p).unwrap();
    rec.mcmc_concluded_successfully = 0.0;
    // the record as the crashed session left it: the end of the previous session
    if i > 0 {
        let prev = mh_files::load_record(&p, i - 1).unwrap();
        rec.last_parameters = prev.last_parameters;
        rec.last_log_post = prev.last_log_post;
        rec.last_seeds = prev.last_seeds;
    } else {
        rec.last_parameters.fill(0.0);
        rec.last_log_post.fill(0.0);
        rec.last_seeds.clear();
    }
    mh_files::update_last_record(&p, &rec).unwrap();
}

#[test]
fn continuation_and_recovery_reproduce_one_long_run() {
    // one run of 5000 draws per chain
    let full = scratch("full");
    run(&full, 1, 5000, 1);
    let want = chains(&full);
    assert_eq!(want[0].total_draws, 5000);
    let rec = mh_files::load_last_record(&paths(&full)).unwrap().0;
    assert_eq!(rec.mh_draws, vec![[5000.0, 18.0, 5000.0 - 17.0 * 278.0]]);
    assert_eq!(rec.mcmc_concluded_successfully, 1.0);
    let r_full = results(&full);

    // 3000 + 2000 (load_mh_file)
    let cont = scratch("cont");
    run(&cont, 1, 3000, 1);
    run(&cont, 2, 2000, 1);
    same_draws(&chains(&cont), &want, 0);
    let rec = mh_files::load_last_record(&paths(&cont)).unwrap().0;
    assert_eq!(rec.mh_draws, vec![[3000.0, 11.0, 220.0], [2000.0, 7.0, 274.0]]);
    let r_cont = results(&cont);
    for k in ["posterior_mean", "posterior_median", "hpd_lower", "hpd_upper", "posterior_sd"] {
        assert_eq!(vecf(&r_cont["mcmc"][k]), vecf(&r_full["mcmc"][k]), "{k}");
    }
    assert_eq!(r_cont["mcmc"]["log_data_density_mhm"], r_full["mcmc"]["log_data_density_mhm"]);

    // load_mh_file, mh_replic = 0: the same statistics from the files
    run(&cont, 3, 0, 99);
    let r_load = results(&cont);
    for k in ["posterior_mean", "hpd_lower", "hpd_upper"] {
        assert_eq!(vecf(&r_load["mcmc"][k]), vecf(&r_full["mcmc"][k]), "{k}");
    }

    // crash in the first session: chain 2 lost its last 4 files, chain 1 its last one
    let rc1 = scratch("recover1");
    run(&rc1, 1, 5000, 1);
    crash(&rc1, 2, 15..=18);
    crash(&rc1, 1, 18..=18);
    run(&rc1, 4, 5000, 1);
    same_draws(&chains(&rc1), &want, 0);
    let rec = mh_files::load_last_record(&paths(&rc1)).unwrap().0;
    assert_eq!(rec.mcmc_concluded_successfully, 1.0);
    let want_rec = mh_files::load_last_record(&paths(&full)).unwrap().0;
    assert_eq!(rec.last_parameters, want_rec.last_parameters);
    assert_eq!(rec.last_seeds, want_rec.last_seeds);

    // crash in the second session, chain 1 with no new file at all
    let rc2 = scratch("recover2");
    run(&rc2, 1, 3000, 1);
    run(&rc2, 2, 2000, 1);
    crash(&rc2, 2, 13..=18);
    crash(&rc2, 1, 12..=18);
    // chain 1: restore its file 11 as the first session left it (220 draws)
    let p = paths(&rc2);
    let f11 = mh_files::MhFile::read(&p.mh_file(11, 1)).unwrap();
    let mut first = f11.clone();
    first.x2.truncate(220 * 7);
    first.logpo2.truncate(220);
    first.write(&p.mh_file(11, 1)).unwrap();
    run(&rc2, 4, 2000, 1);
    same_draws(&chains(&rc2), &want, 0);

    // mh_initialize_from_previous_mcmc: a new MCMC from where the 3000-draw run stopped
    // continues its chains (no new parameter, same proposal)
    let ini = scratch("init_prev");
    run(&ini, 1, 3000, 1);
    copy_dir(&ini.join("nk_mcmc"), &ini.join("prev/nk_mcmc"));
    run(&ini, 7, 2000, 1);
    let new = chains(&ini);
    assert_eq!(new[0].total_draws, 2000);
    for (c, w) in new.iter().zip(&want) {
        for t in 0..2000 {
            assert_eq!(c.draw(t), w.draw(3000 + t));
        }
    }

    for d in [full, cont, rc1, rc2, ini] {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[test]
fn proposal_from_previous_mcmc_and_posterior_mode_estimation() {
    let d = scratch("usecov");
    run(&d, 1, 3000, 2);
    run(&d, 5, 1000, 2);
    // use_mh_covariance_matrix: the proposal is chol of the kept draws' covariance (saved as hh^-1)
    let mhm = dynare::matfile::read(&d.join("nk_mcmc/Output/nk_mcmc_mh_mode.mat")).unwrap();
    let cov = mhm["hh"].as_matrix().unwrap().try_inverse().unwrap();
    let rec = mh_files::load_last_record(&paths(&d)).unwrap().0;
    let dch = rec.proposal_covariance.unwrap();
    let back = dch.transpose() * &dch;
    assert!((&back - &cov).amax() < 1e-10 * cov.amax(), "{}", (&back - &cov).amax());
    assert_eq!(rec.mh_draws.len(), 2);
    let _ = std::fs::remove_dir_all(&d);

    // mh_posterior_mode_estimation: MCMC from the prior mode, the posterior mode from the draws
    let d = scratch("modeest");
    run(&d, 6, 6000, 3);
    let r = results(&d);
    let m = vecf(&r["mh_posterior_mode"]["mode"]);
    let k = r["mh_posterior_mode"]["log_posterior_kernel"].as_f64().unwrap();
    assert!(k.is_finite() && m.len() == 7);
    // close to the posterior mode found by csminwel (stage 1: rho_i 0.618, e_i 0.258)
    assert!((m[6] - 0.618).abs() < 0.05 && (m[2] - 0.258).abs() < 0.05, "{m:?}");
    assert!(d.join("nk_mcmc/Output/nk_mcmc_mh_mode.mat").exists());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn reads_and_continues_dynare_files() {
    let d = scratch("dynare");
    copy_dir(&reference().join("dynare_run"), &d);
    let refj: Value =
        serde_json::from_str(&std::fs::read_to_string(reference().join("ref_mcmc_phase1.json")).unwrap()).unwrap();
    let m = &refj["m2_load"];
    run(&d, 3, 0, 1);
    let r = results(&d)["mcmc"].clone();
    for (k, rk) in [
        ("mean", "posterior_mean"),
        ("median", "posterior_median"),
        ("hpdinf", "hpd_lower"),
        ("hpdsup", "hpd_upper"),
        ("std", "posterior_sd"),
    ] {
        let e = vecf(&m[k]).iter().zip(vecf(&r[rk])).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(e < 1e-12, "{k}: {e:e}");
    }
    let e = (m["mhm"].as_f64().unwrap() - r["log_data_density_mhm"].as_f64().unwrap()).abs();
    assert!(e < 1e-9, "MHM {e:e}");
    // continue Dynare's chains (MATLAB's generator state -> a reproducible ChaCha8 stream)
    run(&d, 2, 500, 1);
    let c = chains(&d);
    assert_eq!(c[0].total_draws, 3500);
    let rec = mh_files::load_last_record(&paths(&d)).unwrap().0;
    assert_eq!(rec.mh_draws[1], [500.0, 2.0, 164.0]);
    let _ = std::fs::remove_dir_all(&d);
}

/// hssmc and dime_mcmc on the linear NK model: posterior means within 3 posterior sd of the
/// RW Metropolis–Hastings ones, the hssmc log marginal density close to the MHM one, and
/// model_comparison finds no Laplace/MHM estimate in the SMC results file (as in Dynare).
#[test]
fn smc_samplers_on_the_linear_model() {
    let d = scratch("smc");
    run(&d, 1, 20000, 4);
    let rw = results(&d)["mcmc"].clone();
    let (m0, s0) = (vecf(&rw["posterior_mean"]), vecf(&rw["posterior_sd"]));
    let mhm = rw["log_data_density_mhm"].as_f64().unwrap();
    for (stage, key, replic) in [(8, "hssmc", 4000), (9, "dime", 1500)] {
        run(&d, stage, replic, 4);
        let r = results(&d);
        let means: Vec<f64> =
            r[key]["posterior"].as_array().unwrap().iter().map(|p| p["mean"].as_f64().unwrap()).collect();
        for i in 0..7 {
            assert!((means[i] - m0[i]).abs() < 3.0 * s0[i], "{key} {i}: {} vs {}", means[i], m0[i]);
        }
        if key == "hssmc" {
            let lmd = r["MarginalDensity"]["hssmc"].as_f64().unwrap();
            assert!((lmd - mhm).abs() < 1.0, "hssmc {lmd} vs MHM {mhm}");
            assert!(r.get("log_data_density_laplace").is_none());
            let est = dynare::estimation::model_comparison::Estimator::LaplaceApproximation;
            // model_comparison.m only reads MarginalDensity.<estimator>: an SMC run has none of them
            assert_eq!(dynare::estimation::model_comparison::read_log_density(&r, est), None);
        }
    }
    let _ = std::fs::remove_dir_all(&d);
}
