//! `calib_smoother` / `shock_decomposition` with `prefilter=1` and the
//! `use_univariate_*_if_singularity_is_detected` options, against Dynare 7.2
//! (reference/smoother_opts/dump.m on a linear NK model with prefiltered data).
mod common;

use std::path::Path;

use dynare::driver::{RunOptions, run};
use serde_json::Value;

fn csv(p: &Path) -> Vec<Vec<String>> {
    let s = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    s.lines().map(|l| l.split(',').map(str::to_string).collect()).collect()
}

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-8 * (1.0 + b.abs()), "{what}: rust {a} vs dynare {b}");
}

#[test]
fn prefilter_in_calib_smoother_and_shock_decomposition() {
    let refdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/smoother_opts");
    let r: Value = common::json_ref(refdir.join("ref.json"));
    let dir = common::temp_dir("smoother_opts");
    for f in ["nk_sm.mod", "nk_gsa_data.csv"] {
        std::fs::copy(refdir.join(f), dir.join(f)).unwrap();
    }
    run(&dir.join("nk_sm.mod"), &RunOptions { quiet: true, ..Default::default() }).unwrap();
    let out = dir.join("nk_sm/Output");

    let sm = csv(&out.join("nk_sm_calib_smoothed_variables.csv"));
    for v in ["y", "pi", "i", "g", "u"] {
        let j = sm[0].iter().position(|s| s == v).unwrap();
        let want = r["smoothed"][v].as_array().unwrap();
        assert_eq!(want.len(), sm.len() - 1);
        for (t, w) in want.iter().enumerate() {
            close(sm[t + 1][j].parse().unwrap(), w.as_f64().unwrap(), &format!("smoothed {v}[{t}]"));
        }
    }

    // oo_.shock_decomposition(endo, component, t): shocks, initial values, smoothed
    let comps = ["e_g", "e_u", "e_i", "Initial values", "Smoothed"];
    let endo: Vec<&str> = r["endo_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let sd = csv(&out.join("nk_sm_shock_decomposition.csv"));
    let mut n = 0;
    for row in &sd[1..] {
        let (t, v, c): (usize, &str, &str) = (row[0].parse().unwrap(), &row[1], &row[2]);
        let (Some(iv), Some(ic)) = (endo.iter().position(|e| *e == v), comps.iter().position(|e| *e == c)) else {
            continue;
        };
        let want = r["shock_decomposition"][iv][ic][t - 1].as_f64().unwrap();
        close(row[3].parse().unwrap(), want, &format!("shock_decomposition {v} {c} [{t}]"));
        n += 1;
    }
    assert_eq!(n, 3 * 5 * 250);
}
