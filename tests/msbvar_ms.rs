//! MS-SBVAR part 2a: `ms_estimation`, `ms_compute_probabilities`, `ms_compute_mdd` against Dynare 7.2
//! (reference/msbvar/ms/: models ms_small / ms_chains / ms_three, ref_ms.json written by gen_ref.py from the
//! files Dynare's `ms_estimation`/`ms_compute_probabilities`/`ms_simulation`/`ms_compute_mdd` produced).
//! Dynare prints the likelihood, prior and probabilities with 6-7 significant digits, which is the precision
//! of the comparisons.

use std::path::Path;
use std::sync::Arc;

use dynare::linalg::Mat;
use dynare::modfile::{Command, parse_mod_file};
use dynare::msbvar::hamilton::smooth;
use dynare::msbvar::mdd::{self, MddOptions};
use dynare::msbvar::ms_estimation::{self, MsOptions, MsSetup};
use dynare::msbvar::prepare_sample;
use serde_json::Value;

fn dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/msbvar/ms")
}

fn refs() -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir().join("ref_ms.json")).unwrap()).unwrap()
}

fn vecf(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn matf(v: &Value) -> Mat {
    let rows: Vec<Vec<f64>> = v.as_array().unwrap().iter().map(vecf).collect();
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

struct Case {
    setup: Arc<MsSetup>,
    start: Vec<f64>,
}

fn case(model: &str) -> Case {
    let mf = parse_mod_file(&dir().join(format!("{model}.mod")), &[]).unwrap();
    let Some(Command::Other(_, o)) =
        mf.commands.iter().find(|c| matches!(c, Command::Other(n, _) if n == "ms_estimation"))
    else {
        panic!("no ms_estimation")
    };
    let p = prepare_sample(&mf, o, &dir()).unwrap();
    let mo = MsOptions::from_options(o).unwrap();
    let setup = MsSetup::build(&p.z, &p.decl, &p.restr, &p.settings, &mo, (p.opts.freq, 1960, 1)).unwrap();
    let start = setup.start_value(&p.z, &p.settings, &p.restr, &p.decl).unwrap();
    Case { setup: Arc::new(setup), start }
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (diff {:e})", (a - b).abs());
}

/// Dynare's free vector in this port's basis of the restricted subspaces (`null()` vs unit vectors).
fn theta_in_our_basis(c: &Case, m: &Value) -> Vec<f64> {
    let mut th = vecf(&m["theta"]);
    let s = &c.setup;
    for j in 0..s.nvar {
        let (u_ml, v_ml) = (matf(&m["U"][j]), matf(&m["V"][j]));
        for &o in &s.layout.b[j] {
            let b = Mat::from_column_slice(s.n0[j], 1, &th[o..o + s.n0[j]]);
            let nb = s.ui[j].transpose() * &u_ml * b;
            th[o..o + s.n0[j]].copy_from_slice(nb.as_slice());
        }
        for &o in &s.layout.g[j] {
            let g = Mat::from_column_slice(s.np[j], 1, &th[o..o + s.np[j]]);
            let ng = s.vi[j].transpose() * &v_ml * g;
            th[o..o + s.np[j]].copy_from_slice(ng.as_slice());
        }
    }
    th
}

/// Likelihood, prior, one-step log densities and smoothed probabilities at Dynare's posterior mode.
fn check_at_mode(model: &str, key: &str) {
    let r = refs();
    let m = &r[key];
    let c = case(model);
    assert_eq!(vecf(&m["theta"]).len(), c.setup.layout.len, "{model}: number of free parameters");
    let th = theta_in_our_basis(&c, m);
    let (reg, f) = c.setup.filter(&th).unwrap();
    close(f.loglik, m["log_likelihood"].as_f64().unwrap(), 2e-4, &format!("{model} log likelihood"));
    close(f.loglik, m["final_lik"].as_f64().unwrap(), 2e-3, &format!("{model} final likelihood"));
    // Dynare's "Final prior Value" carries an unexplained extra constant at the mode (ln 4 for ms_small and
    // ms_three); the prior itself is checked at Dynare's starting values and, through the log posterior of its
    // Metropolis draws, below.
    assert!(c.setup.log_prior(&th).is_finite());
    let lnp = vecf(&m["lnp"]);
    assert_eq!(lnp.len(), f.lnp.len());
    for (t, (a, b)) in f.lnp.iter().zip(&lnp).enumerate() {
        // 5e-6: Dynare prints the mode with 7 significant digits; at the ms_chains mode of the 7.2 run (a
        // transition probability at 5e-10) the differences reach 3.6e-6 (2e-6 sufficed for the 7.1 run)
        close(*a, *b, 5e-6, &format!("{model} ln p(y_{t}|Y_t-1)"));
    }
    let sm = smooth(&f, &reg.q);
    let refsm = matf(&m["smoothed"]);
    assert_eq!(sm.shape(), refsm.shape());
    let d = (&sm - &refsm).abs().max();
    assert!(d < 2e-6, "{model}: smoothed probabilities differ by {d:e}");
    if let Some(fl) = m.get("filtered") {
        let rf = matf(fl);
        let d = (&f.filtered - &rf).abs().max();
        assert!(d < 2e-6, "{model}: filtered probabilities differ by {d:e}");
    }
}

#[test]
fn small_two_regime_variance_switching_at_mode() {
    check_at_mode("ms_small", "small");
}

#[test]
fn two_chains_coefficients_and_variances_at_mode() {
    check_at_mode("ms_chains", "chains");
}

#[test]
fn three_regime_chain_at_mode() {
    check_at_mode("ms_three", "three");
}

/// The starting value is Dynare's: same prior and likelihood as Dynare's `Initial ... Value`.
#[test]
fn starting_values_match_dynare_initial_values() {
    let r = refs();
    for (model, key) in [("ms_small", "small"), ("ms_three", "three"), ("ms_chains", "chains")] {
        let c = case(model);
        let m = &r[key];
        let ll = c.setup.log_likelihood(&c.start);
        let lp = c.setup.log_prior(&c.start);
        println!("{model}: start lik {ll:.4} prior {lp:.5}; Dynare {} {}", m["init_lik"], m["init_prior"]);
        if model == "ms_small" || model == "ms_three" {
            close(ll, m["init_lik"].as_f64().unwrap(), 2e-3, &format!("{model} initial likelihood"));
            close(lp, m["init_prior"].as_f64().unwrap(), 2e-5, &format!("{model} initial prior"));
        }
        if model == "ms_chains" {
            // exclusion restrictions on A+ and two chains (conditional precision V' Hp^-1 V, two-chain scaling)
            close(ll, m["init_lik"].as_f64().unwrap(), 2e-3, "initial likelihood");
            close(lp, m["init_prior"].as_f64().unwrap(), 2e-5, "initial prior");
        }
    }
}

#[test]
fn small_posterior_mode_matches_dynare() {
    let r = refs();
    let m = &r["small"];
    let c = case("ms_small");
    let e = ms_estimation::estimate(c.setup.clone(), c.start.clone()).unwrap();
    let want = c.setup.log_posterior(&theta_in_our_basis(&c, m)); // our kernel at Dynare's mode
    println!("mode: ours {} Dynare's mode under our kernel {want}", e.log_posterior);
    assert!(e.log_posterior >= want - 1e-4, "posterior below Dynare's mode");
    close(e.log_posterior, want, 5e-4, "log posterior at the mode");
    let th = theta_in_our_basis(&c, m);
    for (i, (a, b)) in e.theta.iter().zip(&th).enumerate() {
        close(*a, *b, 5e-3, &format!("mode parameter {i}"));
    }
}

#[test]
fn chains_posterior_mode_is_not_worse_than_dynare() {
    let r = refs();
    let m = &r["chains"];
    let c = case("ms_chains");
    let e = ms_estimation::estimate(c.setup.clone(), c.start.clone()).unwrap();
    let want = c.setup.log_posterior(&theta_in_our_basis(&c, m));
    println!("mode: ours {} Dynare's mode under our kernel {want}", e.log_posterior);
    assert!(e.log_posterior >= want - 5e-3);
}

fn read_gz(name: &str) -> Vec<Vec<f64>> {
    use std::io::Read;
    let f = std::fs::File::open(dir().join("mdd").join(name)).unwrap();
    let mut text = String::new();
    flate2::read::GzDecoder::new(f).read_to_string(&mut text).unwrap();
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect()
}

/// `simulation_msd.out` of Dynare (5000 draws of ms_small, reference/msbvar/ms/mdd/msd.mod + go.m): free
/// parameters and log posterior.
fn dynare_simulation() -> (Mat, Vec<f64>) {
    let rows = read_gz("simulation_msd.out.gz");
    (Mat::from_fn(rows.len(), rows[0].len() - 2, |i, j| rows[i][j + 2]), rows.iter().map(|r| r[0]).collect())
}

/// `(ln h, ln p)` rows of Dynare's `posterior_t<pt>` / `proposal_t<pt>` files (`-print`).
fn dynare_matrix(kind: &str, pt: usize) -> Vec<(f64, f64)> {
    read_gz(&format!("{kind}_t{pt}_msd.out.gz")).iter().map(|r| (r[0], r[1])).collect()
}

/// `mdd_t<pt>_msd.out`: elliptical info (`key=value`), Muller, Bridge, Waggoner-Zha without cut.
fn dynare_mdd(pt: usize) -> (std::collections::HashMap<String, f64>, f64, f64, f64) {
    let text = std::fs::read_to_string(dir().join("mdd").join(format!("mdd_t{pt}_msd.out"))).unwrap();
    let mut info = std::collections::HashMap::new();
    for l in text.lines() {
        if let Some((k, v)) = l.trim().split_once('=') {
            if let Ok(x) = v.trim().parse::<f64>() {
                info.insert(k.trim().to_string(), x);
            }
        }
    }
    let after =
        |key: &str| -> f64 { text.split(key).nth(1).unwrap().split_whitespace().next().unwrap().parse().unwrap() };
    (info, after("Muller log(mdd) = "), after("Bridge log(mdd) = "), after("Waggoner-Zha\nlog(mdd) = "))
}

/// Deterministic check of the proposal of `dw_mdd_switch.c` on Dynare's own draws and matrices: the first 20%
/// of the 5000 draws fit it, the other 4000 are the posterior sample. Bounds / power / step table equal the
/// printed ones, the log proposal density (elliptical x Dirichlet) of every posterior draw equals column 0 of
/// Dynare's `posterior_t<pt>` file, and with Dynare's proposal matrix (its random draws) the Muller, bridge and
/// Waggoner-Zha estimates equal Dynare's to its printed digits, for the six proposal types.
#[test]
fn mdd_proposal_matches_dynare() {
    let c = case("ms_small");
    let (draws, lps) = dynare_simulation();
    assert_eq!(draws.nrows(), 5000);
    for pt in 1..=6 {
        let opts = MddOptions { proposal_type: pt, ..Default::default() };
        let prop = mdd::Proposal::fit(&c.setup, &draws, &lps, 1000, &opts).unwrap();
        assert_eq!(prop.elliptical.dim, 15);
        let (info, want_m, want_b, want_wz) = dynare_mdd(pt);
        let ours = prop.elliptical.info();
        let mut nchecked = 0;
        for l in ours.lines() {
            if let Some((k, v)) = l.trim().split_once('=') {
                if let (Ok(x), Some(w)) = (v.trim().parse::<f64>(), info.get(k.trim())) {
                    close(x, *w, 1e-5 * w.abs().max(1.0), &format!("type {pt}: {k}"));
                    nchecked += 1;
                }
            }
        }
        assert!(nchecked >= 1, "type {pt}: proposal info {ours}");
        if pt == 4 {
            let want = std::fs::read_to_string(dir().join("mdd/mdd_t4_msd.out")).unwrap();
            let tab = |t: &str| -> Vec<f64> {
                t.split("table=")
                    .nth(1)
                    .unwrap()
                    .lines()
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|x| x.parse().unwrap())
                    .collect()
            };
            let (a, b) = (tab(&ours), tab(&want));
            assert_eq!(a.len(), b.len(), "step table size");
            for (x, y) in a.iter().zip(&b) {
                close(*x, *y, 1e-5 * y.abs(), "step table");
            }
        }
        let dyn_post = dynare_matrix("posterior", pt);
        assert_eq!(dyn_post.len(), 4000);
        let ours_post: Vec<(f64, f64)> = (1000..5000)
            .map(|r| {
                let th: Vec<f64> = draws.row(r).iter().copied().collect();
                (prop.ln_density(&th), lps[r])
            })
            .collect();
        let mut worst: f64 = 0.0;
        for (o, d) in ours_post.iter().zip(&dyn_post) {
            if d.0 > -1e299 {
                worst = worst.max((o.0 - d.0).abs() / d.0.abs().max(1.0));
            } else {
                assert!(!o.0.is_finite(), "type {pt}: {} where Dynare has -inf", o.0);
            }
            close(o.1, d.1, 1e-3, "log posterior column");
        }
        let dyn_prop = dynare_matrix("proposal", pt);
        let (m, _, _) = mdd::mueller(&dyn_prop, &ours_post);
        let b = mdd::bridge(&dyn_prop, &ours_post);
        let wz = mdd::waggoner_zha(&dyn_prop, &ours_post);
        println!(
            "type {pt}: muller {m:.4} bridge {b:.4} wz {wz:.4} (Dynare {want_m} {want_b} {want_wz}), worst ln h {worst:e}"
        );
        assert!(
            worst < 1e-5,
            "type {pt}: log proposal density at the posterior draws, worst relative difference {worst:e}"
        );
        close(m, want_m, 2e-4, &format!("type {pt}: Muller on Dynare's matrices"));
        close(b, want_b, 2e-4, &format!("type {pt}: bridge on Dynare's matrices"));
        close(wz, want_wz, 2e-4, &format!("type {pt}: Waggoner-Zha on Dynare's matrices"));
    }
}

/// The whole `ms_compute_mdd` on Dynare's draws with our own proposal draws. A single run of 10000 proposal draws
/// has a Monte Carlo spread of 0.04-0.09 (Dynare's and ours alike), so we compare means over 20 seeds with the
/// means of 20 Dynare runs on the same simulation (`dynare_seeds.txt`, seeds.m): standard error of the
/// difference about 0.02, observed differences up to 0.035. Dynare's estimate mixes a sign-normalised kernel
/// (proposal draws, `+ nvars ln 2`, `-inf` outside the region) with the unnormalised one of the simulation file
/// (posterior draws); `consistent` normalises both sides and gives the true mdd (-325.41).
#[test]
fn mdd_from_dynare_draws() {
    let c = case("ms_small");
    let th = vecf(&refs()["mdd"]["theta"]);
    let (draws, lps) = dynare_simulation();
    let runs: Vec<Vec<f64>> = std::fs::read_to_string(dir().join("mdd/dynare_seeds.txt"))
        .unwrap()
        .lines()
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect();
    const NSEEDS: u64 = 20;
    for pt in 1..=6 {
        let dy: Vec<&Vec<f64>> = runs.iter().filter(|r| r[0] as usize == pt).collect();
        let want_m = dy.iter().map(|r| r[2]).sum::<f64>() / dy.len() as f64;
        let want_b = dy.iter().map(|r| r[3]).sum::<f64>() / dy.len() as f64;
        let (mut m, mut b) = (0.0, 0.0);
        for seed in 0..NSEEDS {
            let opts = MddOptions { proposal_type: pt, proposal_draws: 10000, ..Default::default() };
            let res = mdd::mdd(&c.setup, &th, &draws, Some(&lps), &opts, seed).unwrap();
            assert_eq!((res.n_posterior, res.dim), (4000, 15));
            m += res.muller / NSEEDS as f64;
            b += res.bridge / NSEEDS as f64;
        }
        println!("type {pt}: mean muller {m:.4} bridge {b:.4} (Dynare {want_m:.4} / {want_b:.4})");
        close(b, want_b, 0.07, &format!("mean bridge, proposal type {pt}"));
        close(m, want_m, 0.07, &format!("mean muller, proposal type {pt}"));
    }
    let opts = MddOptions { proposal_type: 3, proposal_draws: 40000, ..Default::default() };
    let res = mdd::mdd(&c.setup, &th, &draws, Some(&lps), &opts, 7).unwrap();
    let cons = mdd::mdd(&c.setup, &th, &draws, Some(&lps), &MddOptions { consistent: true, ..opts }, 7).unwrap();
    println!("consistent: muller {:.4} bridge {:.4}", cons.muller, cons.bridge);
    // the true mdd is -325.41 (our own long chain, mdd_from_own_sampler); on Dynare's 4000 posterior draws the
    // consistent estimate carries the chain's Monte Carlo error: -325.4 on the 7.1 chain, -325.19 on the 7.2 one
    close(cons.bridge, -325.41, 0.3, "consistent bridge");
    close(cons.muller, cons.bridge, 0.1, "consistent muller vs bridge");
    assert!(res.bridge < cons.bridge && res.bridge > cons.bridge - 4f64.ln(), "Dynare's mixture lies in the band");
}

/// Own random-walk Metropolis draws instead of Dynare's.
#[test]
fn mdd_from_own_sampler() {
    let r = refs();
    let m = &r["mdd"];
    let c = case("ms_small");
    let th = vecf(&m["theta"]);
    let (draws, lps) = mdd::rwmh(&c.setup, &th, 6000, 1000, 5, 3000, 11);
    assert_eq!(draws.nrows(), 6000);
    assert!(lps.iter().all(|x| x.is_finite()));
    let opts = MddOptions { proposal_draws: 30000, ..Default::default() };
    let res = mdd::mdd(&c.setup, &th, &draws, Some(&lps), &opts, 3).unwrap();
    let cons = mdd::mdd(&c.setup, &th, &draws, None, &MddOptions { consistent: true, ..opts }, 3).unwrap();
    println!(
        "own sampler: muller {:.3} bridge {:.3}; consistent {:.3} / {:.3}",
        res.muller, res.bridge, cons.muller, cons.bridge
    );
    // Loose on purpose: our own random-walk draws (not Dynare's) feed the estimators, so this
    // compares independent Monte Carlo estimates; the bridge estimate's spread across seeds is
    // a few tenths. The estimators on Dynare's own draws are checked tightly elsewhere.
    close(res.bridge, m["bridge"].as_f64().unwrap(), 0.6, "bridge from own draws");
    close(cons.bridge, -325.41, 0.3, "consistent bridge from own draws");
}

/// The posterior kernel (likelihood x normalised prior) equals the log posterior Dynare writes into its
/// Metropolis draws file, at draws spread over the whole posterior.
#[test]
fn log_posterior_matches_dynare_metropolis_draws() {
    let c = case("ms_small");
    let text = std::fs::read_to_string(dir().join("sim_small.out")).unwrap();
    let mut n = 0;
    for l in text.lines().step_by(20) {
        let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
        close(c.setup.log_posterior(&v[2..]), v[0], 3e-4, "log posterior at a Dynare draw");
        close(c.setup.log_likelihood(&v[2..]), v[1], 3e-4, "log likelihood at a Dynare draw");
        n += 1;
    }
    assert!(n >= 100);
}

/// `ms_compute_mdd` options: Dynare's spellings (`proposal_draws`, `use_mean_center`, what its
/// preprocessor accepts) and the reference manual's (`mdd_proposal_draws`, `mdd_use_mean_center`,
/// which Dynare 7.2's preprocessor rejects with a syntax error).
#[test]
fn ms_compute_mdd_option_spellings() {
    let src = std::fs::read_to_string(dir().join("ms_small.mod")).unwrap();
    let tmp = std::env::temp_dir().join(format!("msbvar_mdd_opts_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    for (opts, draws, mean) in [
        ("proposal_type=2, proposal_draws=1234, use_mean_center", 1234, true),
        ("proposal_type=2, mdd_proposal_draws=4321, mdd_use_mean_center", 4321, true),
        ("proposal_type=2", 100000, false),
    ] {
        let f = tmp.join("m.mod");
        std::fs::write(&f, format!("{src}\nms_compute_mdd({opts});\n")).unwrap();
        let mf = parse_mod_file(&f, &[]).unwrap();
        let o = mf
            .commands
            .iter()
            .find_map(|c| match c {
                Command::Other(n, o) if n == "ms_compute_mdd" => Some(o),
                _ => None,
            })
            .unwrap();
        let mo = dynare::msbvar::ms_commands::mdd_options(o);
        assert_eq!((mo.proposal_type, mo.proposal_draws, mo.use_mean_center), (2, draws, mean), "{opts}");
    }
}
