//! Divergences found by running Dynare 7.2's own test suite (reference/sweep_estfix, numbers from
//! the MATLAB runs of the tests named in ref.json):
//! * `stoch_simul(partial_information)` after `ramsey_model` goes through `PCL_resol.m` /
//!   `dr1_PI.m` at the Ramsey steady state (it ran the full-information solver), and `oo_.dr`
//!   is the PCL state space (`ghx = G1pi`, `ghu = impact`);
//! * `options_.solve_tolf = V;` (and `solve_tolx`, `steady.maxit`, `solve_algo`) reaches the
//!   steady-state solver of the later commands (fs2000.mod's likelihood at its mode).
mod common;

use dynare::driver::{RunOptions, run};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sweep_estfix")
}

fn setup(tag: &str, files: &[&str]) -> PathBuf {
    let dir = common::temp_dir(tag);
    for f in files {
        let src = if refdir().join(f).exists() {
            refdir().join(f)
        } else {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sweep_numfix").join(f)
        };
        std::fs::copy(&src, dir.join(f)).unwrap_or_else(|e| panic!("{f}: {e}"));
    }
    dir
}

fn json(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}

fn quiet() -> RunOptions {
    RunOptions { quiet: true, ..Default::default() }
}

#[test]
fn partial_information_after_ramsey_model() {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    let r = &r["Partial_ramsey"];
    let dir = setup("estfix_partial_ramsey", &["Partial_ramsey.mod"]);
    run(&dir.join("Partial_ramsey.mod"), &quiet()).unwrap();
    let j = json(&dir.join("Partial_ramsey/Output/Partial_ramsey_stoch_simul.json"));
    assert_eq!(j["partial_information"], Value::Bool(true));
    // oo_.dr.ghx = G1pi (42 x 42 here, against 37 x 10 for the full-information rule)
    let size = |m: &Value| [m.as_array().unwrap().len(), m[0].as_array().unwrap().len()];
    let want = |k: &str| [r[k][0].as_u64().unwrap() as usize, r[k][1].as_u64().unwrap() as usize];
    assert_eq!(size(&j["dr"]["ghx"]), want("ghx_size"));
    assert_eq!(size(&j["dr"]["ghu"]), want("ghu_size"));
    // IRFs (PCL_Part_info_irf.m), judged against the largest response: the technology shock's are
    // 1e-8 noise in Dynare (optimal policy offsets it)
    let irfs = r["irfs"].as_object().unwrap();
    let scale =
        irfs.values().flat_map(|v| v.as_array().unwrap()).map(|x| x.as_f64().unwrap().abs()).fold(0.0, f64::max);
    for (k, want) in irfs {
        let got = j["irfs"][k].as_array().unwrap_or_else(|| panic!("irfs.{k}"));
        for (t, w) in want.as_array().unwrap().iter().enumerate() {
            let (a, w) = (got[t].as_f64().unwrap(), w.as_f64().unwrap());
            assert!((a - w).abs() <= 1e-8 * scale, "irfs.{k}[{t}]: rust {a} vs dynare {w}");
        }
    }
}

#[test]
fn options_solve_tolf_reaches_the_estimation_steady_state() {
    let r: Value = common::json_ref(refdir().join("ref.json"));
    let dir = setup("estfix_solve_tolf", &["fs2000_solve_tolf.mod", "fsdat_simul.m"]);
    run(&dir.join("fs2000_solve_tolf.mod"), &quiet()).unwrap();
    let j = json(&dir.join("fs2000_solve_tolf/Output/fs2000_solve_tolf_results.json"));
    let got = -j["minus_log_posterior_at_mode"].as_f64().unwrap();
    let want = r["fs2000_log_posterior_at_mode"].as_f64().unwrap();
    // 3e-4 off with the default solve_tolf (eps^(1/3))
    assert!((got - want).abs() < 1e-8, "log posterior at Dynare's mode: rust {got} vs dynare {want}");
}
