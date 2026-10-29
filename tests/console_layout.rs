//! The console output of `estimation` against the console output of Dynare 7.2 (MATLAB).
//!
//! `reference/console/<name>.matlab.txt` is the part of Dynare's own `<name>.log` that starts
//! at the estimation (`Initial value of the log posterior`) with MATLAB-only text removed
//! (warning blocks, parallel pool messages, the optimizer's per-step lines, which depend on
//! rounding). The test runs `dynare-rs` on
//! the same `.mod` file and compares the two outputs line by line with the numbers masked:
//! every line must have the same text, the same number of numbers, and each number must keep
//! the left edge (left-aligned `num2str`/`disp` output) or the right edge (right-aligned
//! `%10.4f` columns) of the MATLAB line, so column widths and spacing are checked as well.
//! Numbers differ legitimately (optimizer stopping digits, MCMC draws, timing).

mod common;
use std::path::{Path, PathBuf};
use std::process::Command;

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/console")
}

/// Lines that depend on the optimizer's path (not on the layout).
fn is_noise(l: &str) -> bool {
    const PREFIXES: [&str; 33] = [
        "Cliff",
        "Value of bad gradient",
        "Iteration",
        "Done for param",
        "Sequence of univ",
        "Actual dxnorm",
        "FVAL",
        "Ftol",
        "Htol",
        "Gradient",
        "Minimum Hessian",
        "Maximum Hessian",
        "Elapsed time for iter",
        "Try gradient",
        "No further",
        "Note: ",
        "lambda",
        "Predicted",
        "f at the",
        "Norm of dx",
        "Improvement",
        "-----",
        "----",
        "normal step",
        "warning: possible inaccuracy",
        "zero gradient",
        "smallest step",
        "largest step",
        "back and forth",
        "Correct for low angle",
        "Total computing time",
        "Warning:",
        "warning:",
    ];
    PREFIXES.iter().any(|p| l.starts_with(p))
}

/// The console lines from the estimation on, without the optimizer's noise.
fn estimation_lines(text: &str) -> Vec<String> {
    let mut on = false;
    let mom = text.lines().any(|l| l.starts_with("==== Method of Moments"));
    let mut out: Vec<String> = Vec::new();
    let mut skip_blank = false;
    for l in text.lines() {
        if !on
            && (l.starts_with("You did not declare")
                || l.starts_with("Prior distribution")
                || l.starts_with("Initial value")
                || l.starts_with("==== Method of Moments"))
        {
            on = true;
        }
        // method_of_moments: the optimizer prints bare numbers, and MATLAB-only lines may
        // leave a different number of blank lines; Dynare's own warnings are compared
        if on {
            let t = l.trim();
            // csminwel's cliff message prints the step as a bare number between blank lines
            if !t.is_empty() && t.parse::<f64>().is_ok() {
                if out.len() >= 2 && out[out.len() - 1].is_empty() && out[out.len() - 2].is_empty() {
                    out.truncate(out.len() - 2);
                    skip_blank = true;
                }
                continue;
            }
            if skip_blank && t.is_empty() {
                skip_blank = false;
                continue;
            }
            skip_blank = false;
            if mom && t.is_empty() && out.last().is_some_and(|x| x.is_empty()) {
                continue;
            }
        }
        if on && (!is_noise(l) || (mom && l.starts_with("Warning:"))) {
            // the MH progress bar closes with a newline only if it ever refreshed (timing)
            if l.starts_with("Estimation::mcmc: Number of mh files") {
                while out.last().is_some_and(|x| x.is_empty()) {
                    out.pop();
                }
                out.extend([String::new(), String::new()]);
            }
            let mut line = l.trim_end().to_string();
            // hssmc iteration rows: whether the particles were resampled depends on the draws
            let tok: Vec<&str> = line.split_whitespace().collect();
            if tok.len() == 7 && tok[0].parse::<u32>().is_ok() && (tok[5] == "yes" || tok[5] == "no") {
                line = line.replace("yes", "XXX").replace(" no ", "XXX ");
            }
            out.push(line);
        }
    }
    while out.last().is_some_and(|l| l.is_empty()) {
        out.pop();
    }
    {
        // csminwel's "cliff" messages leave a blank line before the termination message
        let keep: Vec<bool> = (0..out.len())
            .map(|i| !(out[i].is_empty() && out.get(i + 1).is_some_and(|n| n.starts_with("improvement <"))))
            .collect();
        let mut k = keep.into_iter();
        out.retain(|_| k.next().unwrap());
    }
    out
}

/// The line with every number replaced by `N`, and the `[start, end)` spans of the numbers.
fn mask(line: &str) -> (String, Vec<(usize, usize)>) {
    let c: Vec<char> = line.chars().collect();
    let (mut s, mut spans, mut i) = (String::new(), Vec::new(), 0);
    while i < c.len() {
        let sign = (c[i] == '-' || c[i] == '+') && i + 1 < c.len() && c[i + 1].is_ascii_digit();
        let prev_word = i > 0 && (c[i - 1].is_alphanumeric() || c[i - 1] == '_');
        if (c[i].is_ascii_digit() || sign) && !(sign && prev_word) {
            let st = i;
            i += usize::from(sign);
            while i < c.len() && c[i].is_ascii_digit() {
                i += 1;
            }
            if i + 1 < c.len() && c[i] == '.' && c[i + 1].is_ascii_digit() {
                i += 1;
                while i < c.len() && c[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i + 1 < c.len() && (c[i] == 'e' || c[i] == 'E') {
                let j = i + 1 + usize::from(c[i + 1] == '-' || c[i + 1] == '+');
                if j < c.len() && c[j].is_ascii_digit() {
                    i = j;
                    while i < c.len() && c[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            s.push('N');
            spans.push((st, i));
        } else {
            s.push(c[i]);
            i += 1;
        }
    }
    (s, spans)
}

/// Compares the two outputs; returns the first differences.
fn diff(expected: &[String], got: &[String]) -> Vec<String> {
    let mut d = Vec::new();
    if expected.len() != got.len() {
        d.push(format!("line count: expected {}, got {}", expected.len(), got.len()));
    }
    for (k, (e, g)) in expected.iter().zip(got).enumerate() {
        let ((me, se), (mg, sg)) = (mask(e), mask(g));
        // the masked text has the numbers collapsed to one character: compare it with the
        // blanks around the numbers, which the edges below check
        let norm = |t: &str| t.split_whitespace().collect::<Vec<_>>().join(" ");
        let ok_edges = se.len() == sg.len() && se.iter().zip(&sg).all(|(a, b)| a.0 == b.0 || a.1 == b.1);
        if norm(&me) != norm(&mg) || !ok_edges {
            d.push(format!("line {}:\n  expected: {e}\n       got: {g}", k + 1));
        }
        if d.len() > 12 {
            break;
        }
    }
    d
}

fn run_and_compare(name: &str, files: &[&str], modfile: &str) {
    // `name` is `<folder>/<stem>` under reference/console, or just the stem of the base folder
    let (dir, stem) = match name.split_once('/') {
        Some((d, s)) => (refdir().join(d), s),
        None => (refdir(), name),
    };
    let work = common::temp_dir(&format!("console_{}", name.replace('/', "_")));
    for f in files {
        std::fs::copy(dir.join(f), work.join(f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    let out =
        Command::new(env!("CARGO_BIN_EXE_dynare-rs")).arg(modfile).arg("nograph").current_dir(&work).output().unwrap();
    assert!(out.status.success(), "{name} failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let got = estimation_lines(&String::from_utf8_lossy(&out.stdout));
    let expected = estimation_lines(&common::read_ref(dir.join(format!("{stem}.matlab.txt"))));
    assert!(
        expected.len() > 3 && got.len() > 3,
        "{name}: no estimation output ({} / {} lines)",
        expected.len(),
        got.len()
    );
    let d = diff(&expected, &got);
    assert!(d.is_empty(), "{name}: console output differs from Dynare's:\n{}", d.join("\n"));
}

/// Bayesian estimation: csminwel messages, mode check, posterior mode table, Laplace
/// approximation, MCMC messages, inefficiency factors, Geweke and Brooks-Gelman diagnostics,
/// marginal density, posterior table, posterior IRFs and moments, smoother and forecasts
/// (`schorfheide_2000.mod` of the Dynare examples, 5000 draws in 2 chains).
#[test]
fn console_bayesian_estimation_matches_dynare() {
    run_and_compare("schorfheide_2000", &["schorfheide_2000.mod", "schorfheide_2000_data.m"], "schorfheide_2000.mod");
}

/// Maximum likelihood with measurement errors and a correlation: the ML table.
#[test]
fn console_maximum_likelihood_matches_dynare() {
    run_and_compare("model_ml", &["model_ml.mod", "model_ml_obs.csv"], "model_ml.mod");
}

/// `mode_compute = 6` (Monte-Carlo based mode finding): the progress blocks and the scale.
#[test]
fn console_gmhmaxlik_matches_dynare() {
    run_and_compare(
        "mc6/schorfheide_2000",
        &["schorfheide_2000.mod", "schorfheide_2000_data.m"],
        "schorfheide_2000.mod",
    );
}

/// DSGE-VAR with a free prior weight: mode finding, MCMC and the DSGE-VAR specific output.
#[test]
fn console_dsge_var_matches_dynare() {
    run_and_compare("dsge_var/dv_est", &["dv_est.mod", "nk_data.csv"], "dv_est.mod");
}

/// OccBin estimation with the piecewise Kalman filter: the likelihood at the initial values
/// and the OccBin smoother messages (`occbin.DSGE_smoother`).
#[test]
fn console_occbin_pkf_matches_dynare() {
    run_and_compare("occbin_est/pkf", &["pkf.mod", "nk_est_data.csv"], "pkf.mod");
}

/// `mh_recover`: the messages of `posterior_sampler_initialization.m` when one chain lost its
/// file (Dynare 7.2 prints no counts after "is not complete!"), and the error when nothing
/// crashed (checked against Dynare 7.2 with the same `.mod`, 600 draws in 2 chains).
#[test]
fn console_mh_recover_matches_dynare() {
    let dir = refdir().join("mh_load");
    let work = common::temp_dir("console_mh_recover");
    std::fs::copy(dir.join("schorfheide_2000_data.m"), work.join("schorfheide_2000_data.m")).unwrap();
    let first = std::fs::read_to_string(dir.join("sch_first.mod")).unwrap().replace("mh_replic=2000", "mh_replic=600");
    let run = |text: &str| {
        std::fs::write(work.join("sch.mod"), text).unwrap();
        Command::new(env!("CARGO_BIN_EXE_dynare-rs")).args(["sch.mod", "nograph"]).current_dir(&work).output().unwrap()
    };
    assert!(run(&first).status.success());
    let recover = first.replace("mode_compute=4,", "mode_compute=0, mode_file='sch/Output/sch_mode', mh_recover,");
    let out = run(&recover);
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(!out.status.success());
    for l in [
        "Estimation::mcmc: Recover mode!",
        "Estimation::mcmc: Recovering the previous proposal density.",
        "Estimation::mcmc: It appears that you don't need to use the mh_recover option!",
        "                  You have to edit the mod file and remove the mh_recover option",
        "                  in the estimation command.",
    ] {
        assert!(text.lines().any(|x| x == l), "missing line {l:?} in:\n{text}");
    }
    assert!(String::from_utf8_lossy(&out.stderr).contains("mh_recover option not required!"));
    std::fs::remove_file(work.join("sch/metropolis/sch_mh1_blck2.mat")).unwrap();
    let out = run(&recover);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let lines: Vec<&str> = text.lines().collect();
    let at = lines.iter().position(|l| *l == "Estimation::mcmc: Recover mode!").expect("recover mode");
    assert_eq!(
        &lines[at..at + 4],
        [
            "Estimation::mcmc: Recover mode!",
            "Estimation::mcmc: Recovering the previous proposal density.",
            "Estimation::mcmc: Chain 1 is complete!",
            "Estimation::mcmc: Chain 2 is not complete!",
        ]
    );
    assert!(lines.contains(&"Estimation::mcmc: Total number of iterations: 600."));
}

/// `method_of_moments`, GMM with `mode_compute = 5` (RBC model, 12 moments, optimal weighting):
/// header, data moments, the stage messages and the estimation table with the J test.
#[test]
fn console_mom_gmm_matches_dynare() {
    run_and_compare("mom/gmm", &["gmm.mod", "rbc_mom_data1.csv", "rbc_mom_data2.csv"], "gmm.mod");
}

/// Penalized GMM with priors (Laplace approximation of non-normal priors, the
/// `check_prior_stderr_corr_skew` warning).
#[test]
fn console_mom_penalized_gmm_matches_dynare() {
    run_and_compare(
        "mom/gmm_penalized",
        &["gmm_penalized.mod", "rbc_mom_data1.csv", "rbc_mom_data2.csv"],
        "gmm_penalized.mod",
    );
}

/// Bayesian IRF matching with a short MCMC (1000 draws, 2 chains): the `method_of_moments::mcmc`
/// messages, the convergence-diagnostics warning, the posterior table.
#[test]
fn console_mom_irf_matching_mcmc_matches_dynare() {
    run_and_compare("mom/nkmc", &["nkmc.mod", "nk_irf_data.inc"], "nkmc.mod");
}

/// The posterior samplers on the Schorfheide model with short chains (`tarb`, `slice`,
/// `hssmc`, `dime_mcmc`): sampler messages, iteration tables, the results table (MATLAB-only
/// parallel pool text removed). Numbers differ (random draws), the layout must not.
#[test]
fn console_tarb_matches_dynare() {
    run_and_compare("samplers/tarb", &["tarb.mod", "schorfheide_2000_data.m"], "tarb.mod");
}

#[test]
fn console_slice_matches_dynare() {
    run_and_compare("samplers/slice", &["slice.mod", "schorfheide_2000_data.m"], "slice.mod");
}

#[test]
fn console_hssmc_matches_dynare() {
    run_and_compare("samplers/hssmc", &["hssmc.mod", "schorfheide_2000_data.m"], "hssmc.mod");
}

#[test]
fn console_dime_matches_dynare() {
    run_and_compare("samplers/dime", &["dime.mod", "schorfheide_2000_data.m"], "dime.mod");
}

/// Continuations of a chain: a first run, then `load_mh_file` with 1000 more draws
/// (the "I am writing a new mh-history file" messages, the mode file's blank line), then
/// `load_mh_file` with `mh_replic = 0` and `load_results_after_load_mh` (no sampler
/// messages, nothing but the `CutSample` lines). All three use the same output folder.
#[test]
fn console_load_mh_file_matches_dynare() {
    let dir = refdir().join("mh_load");
    let work = common::temp_dir("console_mh_load");
    std::fs::copy(dir.join("schorfheide_2000_data.m"), work.join("schorfheide_2000_data.m")).unwrap();
    for (modfile, reference) in
        [("sch_first", None), ("sch_load", Some("sch_load")), ("sch_results", Some("sch_results"))]
    {
        std::fs::copy(dir.join(format!("{modfile}.mod")), work.join("sch.mod")).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_dynare-rs"))
            .args(["sch.mod", "nograph"])
            .current_dir(&work)
            .output()
            .unwrap();
        assert!(out.status.success(), "{modfile} failed:\n{}", String::from_utf8_lossy(&out.stderr));
        if let Some(r) = reference {
            let got = estimation_lines(&String::from_utf8_lossy(&out.stdout));
            let expected = estimation_lines(&common::read_ref(dir.join(format!("{r}.matlab.txt"))));
            let d = diff(&expected, &got);
            assert!(d.is_empty(), "{modfile}: console output differs from Dynare's:\n{}", d.join("\n"));
        }
    }
}

#[test]
fn mask_keeps_the_edges_of_numbers() {
    let (m, s) = mask("alp            0.3560   0.4035 -0.0207 1e-05");
    assert_eq!(m, "alp            N   N N N");
    assert_eq!(s.len(), 4);
    assert_eq!(mask("rho2 and x_1").0, "rhoN and x_N");
    let a = vec!["a      0.356".to_string()];
    assert!(diff(&a, &["a     10.356".to_string()]).is_empty());
    assert!(!diff(&a, &["a    0.356".to_string()]).is_empty());
}
