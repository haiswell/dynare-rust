//! `identification` against Dynare 7.2 (reference/identification/ref_identification_<k>.json,
//! written by reference/identification/dump_identification.m for nk_ident.mod -DCASE=k).
mod common;

use std::path::Path;

use dynare::identification::run::{IdentOptions, build_context};
use dynare::identification::strength::strength_measures;
use dynare::identification::{Persistent, la};
use dynare::linalg::Mat;
use dynare::modfile::{Command, parse_mod_file};
use rand::SeedableRng;
use serde_json::Value;

fn load(case: usize) -> Option<Value> {
    load_named(&case.to_string())
}

fn load_named(case: &str) -> Option<Value> {
    let p =
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("reference/identification/ref_identification_{case}.json"));
    Some(common::json_ref(p))
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a
            .iter()
            .flat_map(|x| match x {
                Value::Array(b) => b.iter().map(num).collect::<Vec<_>>(),
                x => vec![num(x)],
            })
            .collect(),
        x => vec![num(x)],
    }
}

/// MATLAB matrix (array of rows); a vector becomes a column.
fn mat(v: &Value) -> Mat {
    match v {
        Value::Array(rows) if rows.first().is_some_and(|r| r.is_array()) => {
            let nr = rows.len();
            let nc = rows[0].as_array().unwrap().len();
            Mat::from_fn(nr, nc, |i, j| num(&rows[i][j]))
        }
        _ => {
            let x = vecf(v);
            Mat::from_column_slice(x.len(), 1, &x)
        }
    }
}

fn assert_close(name: &str, a: &Mat, b: &Mat, rtol: f64) {
    assert_eq!(a.shape(), b.shape(), "{name}: shape");
    let scale = b.iter().filter(|v| v.is_finite()).fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    let mut worst = 0.0f64;
    let mut at = (0, 0);
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let (x, y) = (a[(i, j)], b[(i, j)]);
            if y.is_nan() {
                assert!(x.is_nan(), "{name}: expected NaN at ({i},{j}), got {x}");
                continue;
            }
            let d = (x - y).abs() / scale;
            if d > worst {
                worst = d;
                at = (i, j);
            }
        }
    }
    println!("{name}: max rel. difference {worst:.2e} at {at:?} (ours {}, Dynare {})", a[at], b[at]);
    assert!(worst < rtol, "{name}: max relative difference {worst:e} > {rtol:e}");
}

fn bools(v: &Value) -> Vec<bool> {
    vecf(v).iter().map(|&x| x != 0.0).collect()
}

fn ident_options(mf: &dynare::modfile::ModFile) -> IdentOptions {
    let o = mf
        .commands
        .iter()
        .find_map(|c| match c {
            Command::Other(n, o) if n == "identification" => Some(o.clone()),
            _ => None,
        })
        .expect("identification command");
    IdentOptions::from_options(&o).unwrap()
}

fn modfile(case: usize) -> dynare::modfile::ModFile {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/identification/nk_ident.mod");
    parse_mod_file(&p, &[("CASE".into(), case.to_string())]).unwrap()
}

#[test]
fn nk_point_analysis_case1() {
    let r = load(1).expect("reference/setup unavailable");
    let mf = modfile(1);
    let opts = ident_options(&mf);
    assert!(opts.advanced);
    let ctx = build_context(&mf, opts).unwrap();
    let theta = vecf(&r["params_point"]);
    let (t2, ps, _) = ctx.point_vector().unwrap();
    assert_eq!(ps, "prior_mean");
    for (a, b) in t2.iter().zip(&theta) {
        assert!((a - b).abs() < 1e-14 * b.abs().max(1.0), "prior mean {a} vs {b}");
    }
    let mut pers = Persistent::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let an = ctx.analysis(&theta, true, true, &mut pers, &mut rng).unwrap().unwrap();

    // Jacobians
    let m = &r["ide_moments_point"];
    let mo = an.moments.as_ref().unwrap();
    assert_close("dMOMENTS", &mo.jacobian, &mat(&m["dMOMENTS"]), 1e-9);
    let ind: Vec<usize> = vecf(&m["ind_dMOMENTS"]).iter().map(|&x| x as usize - 1).collect();
    assert_eq!(mo.ind, ind);
    assert_close("MOMENTS", &Mat::from_column_slice(mo.values.len(), 1, &mo.values), &mat(&m["MOMENTS"]), 1e-12);
    let rf = an.reducedform.as_ref().unwrap();
    let rr = &r["ide_reducedform_point"];
    assert_close("dREDUCEDFORM", &rf.jacobian, &mat(&rr["dREDUCEDFORM"]), 1e-9);
    let rd = &r["ide_dynamic_point"];
    assert_close("dDYNAMIC", &an.dynamic.jacobian, &mat(&rd["dDYNAMIC"]), 1e-10);
    let sp = an.spectrum.as_ref().unwrap();
    let rs = &r["ide_spectrum_point"];
    assert_close("dSPECTRUM", &sp.jacobian, &mat(&rs["dSPECTRUM"]), 1e-9);
    assert_close("dSPECTRUM_NO_MEAN", an.spectrum_no_mean.as_ref().unwrap(), &mat(&rs["dSPECTRUM_NO_MEAN"]), 1e-9);
    let mi = an.minimal.as_ref().unwrap();
    let rmi = &r["ide_minimal_point"];
    assert_close("dMINIMAL", &mi.jacobian, &mat(&rmi["dMINIMAL"]), 1e-9);

    // checks
    for (label, blk, rb) in [("moments", mo, m), ("reducedform", rf, rr), ("spectrum", sp, rs), ("minimal", mi, rmi)] {
        let c = &blk.check;
        assert_eq!(c.rank as f64, num(&rb["rank"]), "{label}: rank");
        assert_eq!(c.ind0, bools(&rb["ind0"]), "{label}: ind0");
        assert_eq!(c.ino as f64, num(&rb["ino"]), "{label}: ino");
        assert_eq!(c.jweak_pair, bools(&rb["jweak_pair"]), "{label}: jweak_pair");
        assert_eq!(c.jweak, bools(&rb["jweak"]), "{label}: jweak");
        let mco = vecf(&rb["Mco"]);
        for (a, b) in c.mco.iter().zip(&mco) {
            assert!((a - b).abs() < 1e-6, "{label}: Mco {a} vs {b}");
        }
        assert_eq!(c.indno[0], bools(&rb["indno"]), "{label}: indno");
    }
    let dc = &an.dynamic.check;
    assert_eq!(dc.rank as f64, num(&rd["rank"]));
    assert_eq!(dc.jweak_pair, bools(&rd["jweak_pair"]));

    // advanced: brute force collinearity
    let (pars, cosn) = an.bruteforce.as_ref().unwrap();
    assert_close("cosndMOMENTS", cosn, &mat(&m["cosndMOMENTS"]), 1e-8);
    let rpars = m["pars"].as_array().unwrap();
    let np = theta.len();
    for ll in 0..2 {
        for i in 0..np {
            let want: Vec<Option<usize>> = match &rpars[ll * np + i] {
                Value::Array(a) => a.iter().map(|x| x.as_f64().map(|v| v as usize - 1)).collect(),
                x => vec![x.as_f64().map(|v| v as usize - 1)],
            };
            if pars[i][ll] != want {
                // exact ties (alpha and psi are perfectly collinear) are broken by rounding, in
                // MATLAB as here: accept another group with the same cosn
                let cols: Vec<usize> = want.iter().flatten().copied().collect();
                let x = Mat::from_fn(mo.si.nrows(), cols.len(), |a, b| mo.si[(a, cols[b])]);
                let y: Vec<f64> = mo.si.column(i).iter().copied().collect();
                let c = dynare::identification::checks::cosn(&y, &x);
                assert!((c - cosn[(i, ll)]).abs() < 1e-10, "pars{{{i},{ll}}}: {:?} vs {want:?}", pars[i][ll]);
            }
        }
    }
    // singular values of J
    let s_ref = vecf(&m["S"]);
    for (a, b) in an.moments_s.iter().zip(&s_ref) {
        assert!((a - b).abs() < 1e-9 * s_ref[0], "S {a} vs {b}");
    }

    // identification strength on MATLAB's simulated sample
    let pt = ctx.solve(&theta).unwrap();
    let sim = mat(&r["endo_simul"]);
    let data = sim.columns(100, sim.ncols() - 100).into_owned();
    let ahess = ctx.asymptotic_hessian_on(&pt, &data).unwrap();
    let h = &r["ide_hess_point"];
    assert_close("AHess", &ahess, &mat(&h["AHess"]), 1e-9);
    let st = strength_measures(
        ahess,
        true,
        &theta,
        &ctx.prior_sd,
        &mo.si,
        None,
        &rf.si,
        &an.dynamic.si,
        ctx.set.n_shock(),
        None,
        1e-3,
    );
    let idx: Vec<usize> = vecf(&h["identified_parameter_indices"]).iter().map(|&x| x as usize - 1).collect();
    assert_eq!(st.identified, idx);
    for (key, ours) in [
        ("ide_strength_dMOMENTS", &st.ide_strength),
        ("ide_strength_dMOMENTS_prior", &st.ide_strength_prior),
        ("deltaM", &st.delta_m),
        ("deltaM_prior", &st.delta_m_prior),
    ] {
        let want = vecf(&h[key]);
        for (a, b) in ours.iter().zip(&want) {
            if b.is_nan() {
                assert!(a.is_nan(), "{key}: expected NaN, got {a}");
            } else {
                assert!((a - b).abs() < 1e-7 * b.abs(), "{key}: {a} vs {b}");
            }
        }
    }
    assert!(an.strength.is_some());
}

#[test]
fn nk_point_analysis_case2_calibration_autocorr() {
    let r = load(2).expect("reference/setup unavailable");
    let mf = modfile(2);
    let opts = ident_options(&mf);
    assert!(opts.useautocorr && opts.ar == 3 && opts.no_strength);
    let ctx = build_context(&mf, opts).unwrap();
    let theta = vecf(&r["params_point"]);
    let (t2, ps, _) = ctx.point_vector().unwrap();
    assert_eq!(ps, "calibration");
    for (a, b) in t2.iter().zip(&theta) {
        assert!((a - b).abs() < 1e-14 * b.abs().max(1.0));
    }
    let mut pers = Persistent::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let an = ctx.analysis(&theta, false, true, &mut pers, &mut rng).unwrap().unwrap();
    let m = &r["ide_moments_point"];
    let mo = an.moments.as_ref().unwrap();
    assert_close("dMOMENTS (autocorr, ar=3)", &mo.jacobian, &mat(&m["dMOMENTS"]), 1e-9);
    assert_eq!(mo.check.rank as f64, num(&m["rank"]));
    assert_eq!(mo.check.jweak_pair, bools(&m["jweak_pair"]));
    assert_close(
        "dSPECTRUM",
        &an.spectrum.as_ref().unwrap().jacobian,
        &mat(&r["ide_spectrum_point"]["dSPECTRUM"]),
        1e-9,
    );
    assert_close("dMINIMAL", &an.minimal.as_ref().unwrap().jacobian, &mat(&r["ide_minimal_point"]["dMINIMAL"]), 1e-9);
    assert!(an.strength.is_none());
}

#[test]
fn numerical_derivatives_agree_with_analytic() {
    let mf = modfile(1);
    let mut opts = ident_options(&mf);
    let ctx = build_context(&mf, opts.clone()).unwrap();
    let (theta, _, _) = ctx.point_vector().unwrap();
    let pt = ctx.solve(&theta).unwrap();
    let a = ctx.perturbation_derivs(&pt).unwrap();
    opts.analytic_derivation_mode = -1;
    let ctx2 = build_context(&mf, opts).unwrap();
    let b = ctx2.perturbation_derivs(&pt).unwrap();
    for j in 0..theta.len() {
        let d =
            (&a.dghx[j] - &b.dghx[j]).amax().max((&a.dghu[j] - &b.dghu[j]).amax()).max((&a.dom[j] - &b.dom[j]).amax());
        assert!(d < 1e-6, "param {j}: analytic vs numerical {d:e}");
    }
    assert!((&a.d_yss - &b.d_yss).amax() < 1e-6);
    for (x, y) in a.dg1_legacy.iter().zip(&b.dg1_legacy) {
        assert!((x - y).amax() < 1e-6);
    }
    let _ = la::eps_of(1.0);
}

#[test]
fn nk_noprior_all_parameters_and_minreal() {
    let r = load_named("noprior").expect("reference/setup unavailable");
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/identification/nk_noprior.mod");
    let mf = parse_mod_file(&p, &[]).unwrap();
    let opts = ident_options(&mf);
    let ctx = build_context(&mf, opts).unwrap();
    assert!(!ctx.prior_exist);
    assert_eq!(ctx.set.names[0], "SE_eg");
    let theta = vecf(&r["params_point"]);
    let (t2, ps, _) = ctx.point_vector().unwrap();
    assert_eq!(ps, "Current_params");
    for (a, b) in t2.iter().zip(&theta) {
        assert!((a - b).abs() < 1e-14 * b.abs().max(1.0));
    }
    let mut pers = Persistent::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let an = ctx.analysis(&theta, true, true, &mut pers, &mut rng).unwrap().unwrap();
    let mo = an.moments.as_ref().unwrap();
    let rf = an.reducedform.as_ref().unwrap();
    let sp = an.spectrum.as_ref().unwrap();
    assert_close("dMOMENTS", &mo.jacobian, &mat(&r["ide_moments_point"]["dMOMENTS"]), 1e-9);
    assert_close("dREDUCEDFORM", &rf.jacobian, &mat(&r["ide_reducedform_point"]["dREDUCEDFORM"]), 1e-9);
    assert_close("dSPECTRUM", &sp.jacobian, &mat(&r["ide_spectrum_point"]["dSPECTRUM"]), 1e-9);
    let mi = an.minimal.as_ref().unwrap();
    let rmi = &r["ide_minimal_point"];
    // minreal removes z: 2 states, so Deltabar has 12 + 2^2 + 3^2 columns
    assert_eq!(mi.jacobian.shape(), mat(&rmi["dMINIMAL"]).shape());
    let blocks = [
        ("moments", mo, &r["ide_moments_point"]),
        ("reducedform", rf, &r["ide_reducedform_point"]),
        ("spectrum", sp, &r["ide_spectrum_point"]),
        ("minimal", mi, rmi),
    ];
    for (label, blk, rb) in blocks {
        let c = &blk.check;
        assert_eq!(c.rank as f64, num(&rb["rank"]), "{label}: rank");
        assert_eq!(c.ind0, bools(&rb["ind0"]), "{label}: ind0");
        assert_eq!(c.ino as f64, num(&rb["ino"]), "{label}: ino");
        assert_eq!(c.jweak_pair, bools(&rb["jweak_pair"]), "{label}: jweak_pair");
    }
    // without priors, Dynare falls back to the simulated moments information matrix
    assert!(!an.strength.as_ref().unwrap().flag_score);
}

/// reference/sw07/sw07.mod up to its estimation call, with `drop` removed from varobs and
/// `ident` appended (as reference/identification/dump_sw07.m builds sw07_ident.mod; the last
/// observable, `pinflrobs;`, hands its semicolon to the line before).
fn sw07_ident_mod(tag: &str, drop: &[&str], ident: &str) -> std::path::PathBuf {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07/sw07.mod");
    let txt = common::read_ref(src);
    let cut = txt.find("// ESTIMATION CALL").expect("sw07.mod: no ESTIMATION CALL marker");
    let txt = &txt[..cut];
    let vo = txt.find("\nvarobs").expect("sw07.mod: no varobs");
    let (head, tail) = txt.split_at(vo);
    let mut removed = vec![false; drop.len()];
    let mut out = String::from(head);
    for line in tail.split_inclusive('\n') {
        let t = line.trim();
        if let Some(k) = drop.iter().position(|d| t == *d || t.strip_suffix(';') == Some(*d)) {
            if !removed[k] {
                removed[k] = true;
                if t.ends_with(';') {
                    out.truncate(out.trim_end().len());
                    out.push_str(";\n");
                }
                continue;
            }
        }
        out.push_str(line);
    }
    out.push('\n');
    out.push_str(ident);
    out.push('\n');
    // one directory per test: the tests run in parallel and write different files
    let p = common::temp_dir(&format!("ident_sw07_{tag}")).join("sw07_ident.mod");
    std::fs::write(&p, out).unwrap();
    p
}

#[test]
fn sw07_stationary_observables() {
    let r = load_named("sw07").expect("reference/setup unavailable");
    let p = sw07_ident_mod(
        "stationary",
        &["pinfobs", "robs", "labobs", "pinfexp4obs", "pinflrobs"],
        "identification(parameter_set=calibration, diffuse_filter, no_identification_strength);",
    );
    let mf = parse_mod_file(&p, &[]).unwrap();
    let ctx = build_context(&mf, ident_options(&mf)).unwrap();
    let theta = vecf(&r["params_point"]);
    let (t2, _, _) = ctx.point_vector().unwrap();
    for (a, b) in t2.iter().zip(&theta) {
        assert!((a - b).abs() < 1e-12 * b.abs().max(1.0), "calibration {a} vs {b}");
    }
    let t = std::time::Instant::now();
    let mut pers = Persistent::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let an = ctx.analysis(&theta, false, true, &mut pers, &mut rng).unwrap().unwrap();
    println!("SW07 identification: {:.2} s", t.elapsed().as_secs_f64());
    let m = &r["ide_moments_point"];
    let mo = an.moments.as_ref().unwrap();
    let refm = mat(&m["dMOMENTS"]);
    assert_close("SW07 dMOMENTS", &mo.jacobian, &refm, 1e-10); // measured 1e-13
    let ind: Vec<usize> = vecf(&m["ind_dMOMENTS"]).iter().map(|&x| x as usize - 1).collect();
    assert_eq!(mo.ind, ind);
    let c = &mo.check;
    println!(
        "rank {} ino {} ind0=0 {:?}",
        c.rank,
        c.ino,
        (0..c.ind0.len()).filter(|&i| !c.ind0[i]).collect::<Vec<_>>()
    );
    for row in &c.indno {
        println!("indno {:?}", (0..row.len()).filter(|&i| row[i]).collect::<Vec<_>>());
    }
    assert_eq!(c.rank as f64, num(&m["rank"]), "rank");
    assert_eq!(c.ind0, bools(&m["ind0"]), "ind0");
    assert_eq!(c.ino as f64, num(&m["ino"]), "ino");
    assert_eq!(c.indno[0], bools(&m["indno"]), "indno");
    assert_eq!(c.jweak_pair, bools(&m["jweak_pair"]), "jweak_pair");
    // dSPECTRUM is NaN (unit roots at frequency 0) and obs < shocks: both skipped, as in Dynare
    assert!(an.spectrum.is_none() && an.minimal.is_none());
    let rf = an.reducedform.as_ref().unwrap();
    assert_eq!(rf.check.rank as f64, num(&r["ide_reducedform_point"]["rank"]));
}

#[test]
fn sw07_nonstationary_observables_are_rejected_like_dynare() {
    // Dynare 7.2 (dump_sw07.m with only labobs and pinflrobs removed): "There are NaN's in the
    // theoretical moments. Make sure that for non-stationary models stationary transformations
    // of non-stationary observables are used for checking identification." -- pinfobs, robs and
    // pinfexp4obs still load on the pistar random walk.
    let p = sw07_ident_mod(
        "nonstationary",
        &["labobs", "pinflrobs"],
        "identification(parameter_set=calibration, diffuse_filter, no_identification_strength, no_identification_minimal);",
    );
    let mf = parse_mod_file(&p, &[]).unwrap();
    let ctx = build_context(&mf, ident_options(&mf)).unwrap();
    let (theta, _, _) = ctx.point_vector().unwrap();
    let t = std::time::Instant::now();
    let mut pers = Persistent::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let err = ctx.analysis(&theta, false, true, &mut pers, &mut rng).expect_err("NaN moments must be an error");
    println!("SW07: {err} ({:.2} s)", t.elapsed().as_secs_f64());
    assert!(err.to_string().contains("There are NaN's in the theoretical moments. Make sure"));
    // which observables are non-stationary
    let pt = ctx.solve(&theta).unwrap();
    let j = ctx.jacobians(&pt, 1, false, false).unwrap();
    let ns: Vec<&String> = ctx
        .varobs
        .iter()
        .zip(&j.pruned.stationary)
        .filter(|(_, s)| !**s)
        .map(|(v, _)| &mf.model.endo_names[*v])
        .collect();
    println!("non-stationary observables: {ns:?}");
    assert!(!ns.is_empty());
}

/// Maximum likelihood (estimated_params without priors, reference/identification/nk_ml.mod
/// with `parameter_set=prior_mean, prior_mc=10`): Dynare tests the ML starting values
/// ("ML_Starting_value") at one point, and its identification strength takes the simulated
/// moments branch (prior_bounds fails on pshape 0 inside the asymptotic Hessian's `try`).
#[test]
fn nk_ml_starting_values_single_point() {
    let r = load_named("ml").expect("reference/setup unavailable");
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/identification/nk_ml.mod");
    let mf = parse_mod_file(&src, &[]).unwrap();
    let ctx = build_context(&mf, ident_options(&mf)).unwrap();
    assert!(ctx.prior_exist && ctx.dsge.ml);
    let (theta, ps, _) = ctx.point_vector().unwrap();
    assert_eq!(ps, "ML_Starting_value");
    assert_eq!(theta, vecf(&r["params_point"]));
    let mut pers = Persistent::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let an = ctx.analysis(&theta, true, true, &mut pers, &mut rng).unwrap().unwrap();
    let mo = an.moments.as_ref().unwrap();
    assert_close("dMOMENTS", &mo.jacobian, &mat(&r["ide_moments_point"]["dMOMENTS"]), 1e-9);
    let rf = an.reducedform.as_ref().unwrap();
    assert_close("dREDUCEDFORM", &rf.jacobian, &mat(&r["ide_reducedform_point"]["dREDUCEDFORM"]), 1e-9);
    for (label, blk, rb) in [("moments", mo, &r["ide_moments_point"]), ("reducedform", rf, &r["ide_reducedform_point"])]
    {
        assert_eq!(blk.check.rank as f64, num(&rb["rank"]), "{label}: rank");
        assert_eq!(blk.check.ind0, bools(&rb["ind0"]), "{label}: ind0");
        assert_eq!(blk.check.jweak_pair, bools(&rb["jweak_pair"]), "{label}: jweak_pair");
    }
    assert_eq!(an.strength.as_ref().unwrap().flag_score, r["ide_hess_point"]["flag_score"].as_f64() == Some(1.0));
    // the whole command: one point, files named as Dynare's, no Monte Carlo (prior_mc ignored)
    let tmp = std::env::temp_dir().join(format!("dynare_rs_ident_ml_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::copy(&src, tmp.join("nk_ml.mod")).unwrap();
    let mf = parse_mod_file(&tmp.join("nk_ml.mod"), &[]).unwrap();
    let o = mf
        .commands
        .iter()
        .find_map(|c| match c {
            Command::Other(n, o) if n == "identification" => Some(o.clone()),
            _ => None,
        })
        .unwrap();
    dynare::identification::run(&mf, &tmp, "nk_ml", &o, 1).unwrap();
    let mut files: Vec<String> = std::fs::read_dir(tmp.join("nk_ml/identification"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|f| f.ends_with("_identif.json"))
        .collect();
    files.sort();
    let _ = std::fs::remove_dir_all(&tmp);
    assert_eq!(files, ["nk_ml_ML_Starting_value_identif.json", "nk_ml_identif.json"]);
}
