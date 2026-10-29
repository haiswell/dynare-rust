//! Global sensitivity analysis against Dynare 7.2 (`reference/gsa/dump_gsa.m` on
//! `reference/gsa/nk_gsa.mod`, `reference/gsa/dump_sobol.m`).
//!
//! The samples (Sobol points, MATLAB's Mersenne-Twister `randperm`/`rand` draws, Morris
//! trajectories) and the classification of every point must match exactly; Smirnov and
//! correlation statistics, reduced-form coefficients and RMSEs to rounding.
mod common;

use std::path::{Path, PathBuf};

use dynare::gsa::{self, GsaOutput, rng::MatlabRng, sobol::qmc_sequence, stats::smirnov_test};
use dynare::modfile::{Command, parse_mod_file};
use serde_json::Value;

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/gsa")
}

fn load(name: &str) -> Option<Value> {
    let p = refdir().join(name);
    Some(common::json_ref(&p))
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn vec1(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(num).collect(),
        Value::Null => vec![],
        x => vec![num(x)],
    }
}

fn mat(v: &Value) -> Vec<Vec<f64>> {
    match v {
        Value::Array(a) if a.first().is_some_and(Value::is_array) => a.iter().map(vec1).collect(),
        Value::Array(a) => a.iter().map(|x| vec![num(x)]).collect(),
        _ => vec![],
    }
}

/// 1-based MATLAB index vector -> 0-based.
fn idx(v: &Value) -> Vec<usize> {
    vec1(v).into_iter().map(|x| x as usize - 1).collect()
}

/// MATLAB's jsonencode prints 16 significant digits (not always round-trip exact), so
/// "exact" means agreement to that precision.
const JSON_TOL: f64 = 6e-16;

/// The prior-range sample scales the unit-cube points by `prior_bounds(.., 1e-10)`; for gamma,
/// beta and inverse-gamma priors these are Newton iterates of `missing/stats/gaminv.m` /
/// `betainv.m` stopped at |step| < sqrt(eps) on CDFs computed near 1 - 1e-10, so they carry
/// the ~1e-10 relative difference between MATLAB's `gammainc`/`betainc` and statrs' in the
/// tails. The unit-cube points themselves (and the uniform-prior columns) agree exactly.
/// With glibc's or the MSVC CRT's libm the difference is 1.5e-9; the Newton stopping rule
/// (|step| < sqrt(eps) = 1.5e-8) is the bound that holds whatever the libm.
const SAMPLE_TOL: f64 = if cfg!(target_os = "macos") { 1e-9 } else { 1.5e-8 };

/// Columns of lpmat with uniform priors in nk_gsa.mod (phi_pi, rho_g): exact.
const UNIFORM_COLS: [usize; 2] = [2, 5];

fn assert_close(a: &[Vec<f64>], b: &[Vec<f64>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: rows");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.len(), y.len(), "{what}: cols");
        for (u, v) in x.iter().zip(y) {
            assert!(rel(*u, *v) <= JSON_TOL, "{what}[{k}]: {u} vs {v}");
        }
    }
}

fn rel(a: f64, b: f64) -> f64 {
    if a.is_nan() && b.is_nan() { 0.0 } else { (a - b).abs() / b.abs().max(1.0) }
}

/// Runs `sensitivity` of nk_gsa.mod at `stage` in a private copy of reference/gsa.
fn run_stage(stage: u32) -> GsaOutput {
    run_stage_with(stage, |_, _, _| ())
}

/// Runs the stage, then `f(output, modfile, folder)` before the private folder is removed.
fn run_stage_with(stage: u32, f: impl FnOnce(&GsaOutput, &dynare::modfile::ModFile, &Path)) -> GsaOutput {
    let tmp = std::env::temp_dir().join(format!("dynare_rs_gsa_{}_{stage}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    for f in ["nk_gsa.mod", "nk_gsa_data.csv"] {
        std::fs::copy(refdir().join(f), tmp.join(f)).unwrap();
    }
    let modp = tmp.join("nk_gsa.mod");
    let mf = parse_mod_file(&modp, &[("STAGE".into(), stage.to_string())]).unwrap();
    let (ci, o) = mf
        .commands
        .iter()
        .enumerate()
        .find_map(|(i, c)| match c {
            Command::Other(n, o) if n == "sensitivity" => Some((i, o.clone())),
            _ => None,
        })
        .expect("no sensitivity command");
    let out = gsa::run(&mf, &tmp, "nk_gsa", &o, ci).unwrap().expect("no output");
    f(&out, &mf, &tmp);
    let _ = std::fs::remove_dir_all(&tmp);
    out
}

fn check_sample(out: &GsaOutput, r: &Value, tol: f64) {
    let l = mat(&r["lpmat"]);
    let l0 = mat(&r["lpmat0"]);
    assert_eq!(out.stab.lpmat.len(), l.len(), "sample size");
    let mut worst = (0.0f64, 0, 0, 0.0, 0.0);
    for (which, (ours, theirs)) in [(&out.stab.lpmat, &l), (&out.stab.lpmat0, &l0)].into_iter().enumerate() {
        for (a, b) in ours.iter().zip(theirs) {
            for (j, (x, y)) in a.iter().zip(b).enumerate() {
                if rel(*x, *y) > worst.0 {
                    worst = (rel(*x, *y), which, j, *x, *y);
                }
            }
        }
    }
    for (a, b) in out.stab.lpmat.iter().zip(&l) {
        for &j in &UNIFORM_COLS {
            if a.len() > j {
                assert!(rel(a[j], b[j]) <= JSON_TOL || tol > SAMPLE_TOL, "uniform column {j}: {} vs {}", a[j], b[j]);
            }
        }
    }
    assert!(
        worst.0 <= tol,
        "sample differs from Dynare's by {:e} (lpmat{} column {}: {} vs {})",
        worst.0,
        if worst.1 == 1 { "0" } else { "" },
        worst.2,
        worst.3,
        worst.4
    );
}

fn check_classes(out: &GsaOutput, r: &Value) {
    assert_eq!(out.stab.istable, idx(&r["istable"]), "istable");
    assert_eq!(out.stab.iindeterm, idx(&r["iindeterm"]), "iindeterm");
    assert_eq!(out.stab.ixun, idx(&r["ixun"]), "ixun");
    assert_eq!(out.stab.iwrong, idx(&r["iwrong"]), "iwrong");
    let infox: Vec<i32> = vec1(&r["infox"]).into_iter().map(|x| x as i32).collect();
    assert_eq!(out.stab.infox, infox, "infox");
    assert_eq!(out.stab.nspred, num(&r["nspred"]) as usize);
}

fn check_ks(x: &[Vec<f64>], ib: &[usize], inb: &[usize], r: &Value) {
    let p = vec1(&r["proba"]);
    let d = vec1(&r["dproba"]);
    for j in 0..p.len() {
        let a: Vec<f64> = ib.iter().map(|&k| x[k][j]).collect();
        let b: Vec<f64> = inb.iter().map(|&k| x[k][j]).collect();
        let (_, pj, dj) = smirnov_test(&a, &b, 0.05, 0);
        assert!((dj - d[j]).abs() < 1e-14, "KS d[{j}]: {dj} vs {}", d[j]);
        assert!((pj - p[j]).abs() < 1e-12 * p[j].max(1e-300) + 1e-300, "KS p[{j}]: {pj:e} vs {:e}", p[j]);
    }
}

#[test]
fn sobol_matches_qmc_sequence() {
    let r = load("sobol_check.json").expect("reference/setup unavailable");
    let first = mat(&r["first"]);
    let p = qmc_sequence(40, 1, first.len()).unwrap();
    assert_close(&p, &first, "first");
    let high = mat(&r["high"]);
    let p = qmc_sequence(1111, 1, high.len()).unwrap();
    let p: Vec<Vec<f64>> = p.iter().map(|a| a[1089..].to_vec()).collect();
    assert_close(&p, &high, "high");
    let seed = num(&r["extra_seed"]) as u64;
    let p = qmc_sequence(1111, seed, 1).unwrap();
    assert_close(&p, &[vec1(&r["extra"])], "extra");
    let s = mat(&r["sample_np7_n2048"]);
    assert_close(&qmc_sequence(7, 1, 2048).unwrap(), &s, "np7");
}

#[test]
fn matlab_stream_randperm() {
    let mut r = MatlabRng::new(0);
    assert_eq!(r.rand(), 0.81472368639317894);
}

#[test]
fn stage1_stability_map() {
    let r = load("ref_gsa_stage1.json").expect("reference/setup unavailable");
    let out = run_stage(1);
    check_sample(&out, &r, SAMPLE_TOL);
    check_classes(&out, &r);
    let s = &out.stab;
    let n = s.infox.len();
    let itot: Vec<usize> = (0..n).collect();
    let not_stable: Vec<usize> = itot.iter().copied().filter(|k| !s.istable.contains(k)).collect();
    check_ks(&s.lpmat, &s.istable, &not_stable, &r["ks_stable"]);
    let isolve: Vec<usize> = itot.iter().copied().filter(|k| !s.iwrong.contains(k)).collect();
    let b: Vec<usize> = isolve.iter().copied().filter(|k| !s.iindeterm.contains(k)).collect();
    check_ks(&s.lpmat, &b, &s.iindeterm, &r["ks_indeterm"]);
    let b: Vec<usize> = isolve.iter().copied().filter(|k| !s.ixun.contains(k)).collect();
    check_ks(&s.lpmat, &b, &s.ixun, &r["ks_unstable"]);
    // corrcoef with p-values on the stable subset
    let np = s.lpmat[0].len();
    let cols: Vec<Vec<f64>> = (0..np).map(|j| s.istable.iter().map(|&k| s.lpmat[k][j]).collect()).collect();
    let (c, p) = gsa::stats::corrcoef(&cols);
    let (rc, rp) = (mat(&r["corr_stable"]["c"]), mat(&r["corr_stable"]["p"]));
    for i in 0..np {
        for j in 0..np {
            assert!((c[i][j] - rc[i][j]).abs() < 1e-13, "corr {i},{j}");
            assert!(
                (p[i][j] - rp[i][j]).abs() < 1e-10 * rp[i][j].max(1e-3),
                "corr p {i},{j}: {} vs {}",
                p[i][j],
                rp[i][j]
            );
        }
    }
}

#[test]
fn stage2_prior_quantiles() {
    let r = load("ref_gsa_stage2.json").expect("reference/setup unavailable");
    let out = run_stage(2);
    // inverse prior CDFs (betainv/gaminv Newton iterations): rounding-level agreement
    check_sample(&out, &r, 1e-10);
    check_classes(&out, &r);
}

#[test]
fn stage3_reduced_form() {
    let r = load("ref_gsa_stage3.json").expect("reference/setup unavailable");
    let out = run_stage(3);
    check_sample(&out, &r, SAMPLE_TOL);
    check_classes(&out, &r);
    // [ghx ghu] rows of y and pi at every stable point (DR order)
    let s = &out.stab;
    let st = dynare::solver::DrStructure::new(&out.dsge.model.lead_lag_incidence, out.dsge.model.n_exo());
    for name in ["y", "pi"] {
        let row = st.inv_order_var[out.dsge.model.endo_index(name).unwrap()];
        let want = mat(&r[format!("T_{name}")]);
        assert_eq!(want.len(), s.t.len());
        for (k, w) in want.iter().enumerate() {
            for (c, wv) in w.iter().enumerate() {
                let v = s.t[k][(row, c)];
                // Evaluated at our sample, so it carries the sample's difference (SAMPLE_TOL).
                let tol = if cfg!(target_os = "macos") { 1e-9 } else { 1e-7 };
                assert!((v - wv).abs() < tol * wv.abs().max(1.0), "T_{name}[{k}][{c}]: {v} vs {wv}");
            }
        }
    }
}

#[test]
fn stage4_morris() {
    let r = load("ref_gsa_stage4.json").expect("reference/setup unavailable");
    let out = run_stage(4);
    check_sample(&out, &r, SAMPLE_TOL);
    check_classes(&out, &r);
    let m = &r["morris"]["y_vs_e_i"];
    let y0 = vec1(&m["y0"]);
    let sample: Vec<Vec<f64>> = (0..out.stab.lpmat.len()).map(|k| out.stab.point(k)).collect();
    let meas = gsa::redform::morris_measures(sample[0].len(), &sample, &y0, 6);
    let want = mat(&m["SAMorris"]);
    for (a, b) in meas.iter().zip(&want) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-12 * y.abs().max(1.0), "Morris {x} vs {y}");
        }
    }
}

#[test]
fn stage5_rmse() {
    let r = load("ref_gsa_stage5.json").expect("reference/setup unavailable");
    let cmp = |a: &[f64], b: &[f64], what: &str, tol: f64| {
        assert_eq!(a.len(), b.len(), "{what} length");
        let (k, w) = a
            .iter()
            .zip(b)
            .map(|(x, y)| rel(*x, *y))
            .enumerate()
            .fold((0, 0.0), |m, (k, e)| if e > m.1 { (k, e) } else { m });
        eprintln!("{what}: worst relative difference {w:e} at {k} ({} vs {})", a[k], b[k]);
        let mut e: Vec<(f64, usize)> = a.iter().zip(b).map(|(x, y)| rel(*x, *y)).zip(0..).collect();
        e.sort_by(|p, q| q.0.total_cmp(&p.0));
        eprintln!("  largest: {:?}", &e[..5.min(e.len())]);
        assert!(w < tol, "{what}[{k}]: {} vs {}", a[k], b[k]);
    };
    let flat = |m: &[Vec<f64>]| m.iter().flatten().copied().collect::<Vec<f64>>();
    // Dynare's own sample (lpmat/lpmat0 from the dump), filtered by our code: rounding level.
    let mut exact = None;
    let out = run_stage_with(5, |out, mf, dir| {
        let stab = gsa::stability::classify_sample(&out.dsge, mat(&r["lpmat0"]), mat(&r["lpmat"]), false);
        let res = gsa::rmse::run(&out.ctx, mf, dir, &out.dsge, &out.options, &out.options.raw, &stab).unwrap();
        exact = Some(res);
    });
    let ex = exact.unwrap();
    cmp(&ex.logpo2, &vec1(&r["logpo2"]), "logpo2 (Dynare's sample)", 1e-8);
    cmp(&ex.likelihood, &vec1(&r["likelihood"]), "likelihood (Dynare's sample)", 1e-8);
    cmp(&flat(&ex.rmse_mc), &flat(&mat(&r["rmse_MC"])), "rmse_MC (Dynare's sample)", 1e-6);
    cmp(&flat(&ex.r2_mc), &flat(&mat(&r["r2_MC"])), "r2_MC (Dynare's sample)", 1e-6);
    // Our own sample: its gamma/beta bounds differ by ~1e-10 (see SAMPLE_TOL), which the
    // likelihood amplifies at a few badly fitting points.
    check_sample(&out, &r, SAMPLE_TOL);
    check_classes(&out, &r);
    let res = out.rmse.as_ref().unwrap();
    cmp(&res.logpo2, &vec1(&r["logpo2"]), "logpo2", 1e-7);
    cmp(&res.likelihood, &vec1(&r["likelihood"]), "likelihood", 1e-7);
    cmp(&flat(&res.rmse_mc), &flat(&mat(&r["rmse_MC"])), "rmse_MC", 1e-5);
    cmp(&flat(&res.r2_mc), &flat(&mat(&r["r2_MC"])), "r2_MC", 1e-5);
}

/// `sensitivity(rmse=1, prefilter=1, conf_sig, use_univariate_*_if_singularity_is_detected)`
/// (reference/gsa/ref_gsa_stage8.json): the prefiltered data change the RMSE / likelihood.
#[test]
fn stage8_rmse_prefilter() {
    let r = load("ref_gsa_stage8.json").expect("reference unavailable");
    let flat = |m: &[Vec<f64>]| m.iter().flatten().copied().collect::<Vec<f64>>();
    let mut res = None;
    run_stage_with(8, |out, mf, dir| {
        let stab = gsa::stability::classify_sample(&out.dsge, mat(&r["lpmat0"]), mat(&r["lpmat"]), false);
        res = Some(gsa::rmse::run(&out.ctx, mf, dir, &out.dsge, &out.options, &out.options.raw, &stab).unwrap());
    });
    let ex = res.unwrap();
    for (a, b, what) in [
        (ex.logpo2.clone(), vec1(&r["logpo2"]), "logpo2"),
        (ex.likelihood.clone(), vec1(&r["likelihood"]), "likelihood"),
        (flat(&ex.rmse_mc), flat(&mat(&r["rmse_MC"])), "rmse_MC"),
        (flat(&ex.r2_mc), flat(&mat(&r["r2_MC"])), "r2_MC"),
    ] {
        assert_eq!(a.len(), b.len(), "{what} length");
        for (k, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(rel(*x, *y) < 1e-6, "{what}[{k}]: {x} vs {y}");
        }
    }
}
