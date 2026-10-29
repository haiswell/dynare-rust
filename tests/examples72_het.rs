//! Dynare 7.2's official examples (examples/heterogeneity, examples/optimal_policy) against
//! Dynare 7.2: reference/examples72/<name>/ holds the example, its data files and `ref.json`,
//! written by reference/examples72/dump_het.m (`matlab -batch "mdl='<name>'; run('dump_het.m')"`
//! in a private copy of the directory).
//!
//! Heterogeneity: the .mod file's heterogeneity_* commands run in order (as the driver does);
//! steady-state aggregates and policy summaries, the sequence-space Jacobians `dr.G` (first
//! column, first rows), the IRFs, the stochastic paths given MATLAB's draws (first 40 periods)
//! and the news-shock paths are compared.

use std::path::{Path, PathBuf};

use dynare::heterogeneity::commands::{HetSession, handles, run_command};
use dynare::linalg::Mat;
use dynare::matfile::MatValue;
use dynare::modfile::{Command, ModFile, OptValue, parse_mod_file};
use serde_json::Value;

fn dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/examples72").join(name)
}

fn reference(name: &str) -> Value {
    let p = dir(name).join("ref.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}

fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        Value::Number(x) => vec![x.as_f64().unwrap()],
        Value::Null => vec![f64::NAN],
        _ => panic!("not numeric: {v}"),
    }
}

/// Matrix from a jsonencoded MATLAB matrix (array of rows; a row vector is a flat array).
fn rows(v: &Value) -> Vec<Vec<f64>> {
    match v {
        Value::Array(a) if a.first().is_some_and(|x| x.is_array()) => a.iter().map(flat).collect(),
        other => vec![flat(other)],
    }
}

fn assert_close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        if g.is_nan() && w.is_nan() {
            continue;
        }
        assert!((g - w).abs() <= tol * (1.0 + w.abs()), "{what}[{k}]: {g} vs {w} (diff {:e})", (g - w).abs());
    }
}

/// Runs the example's heterogeneity commands (without figures).
fn run_het(name: &str) -> (ModFile, HetSession) {
    let d = dir(name);
    let mut mf = parse_mod_file(&d.join(format!("{name}.mod")), &[]).unwrap();
    let mut st = HetSession::default();
    let cmds = mf.commands.clone();
    for c in &cmds {
        if let Command::Other(n, o) = c
            && handles(n)
        {
            let mut o = o.clone();
            if n == "heterogeneity_simulate" {
                o.push(("nograph".into(), OptValue::Flag));
            }
            run_command(&mut mf, n, &o, &d, name, &mut st).unwrap_or_else(|e| panic!("{name}: {n}: {e:#}"));
        }
    }
    (mf, st)
}

fn check_het(name: &str, tol: f64, pol_tol: f64) {
    let r = reference(name);
    let (mf, st) = run_het(name);
    // parameters (after a calibration) and steady state
    assert_close("params", &mf.model.params, &flat(&r["params"]), tol);
    let ss = st.steady_state.as_ref().unwrap();
    let ragg = r["het"]["agg"].as_object().unwrap();
    assert_eq!(ss.steady_state.agg.len(), ragg.len(), "agg size");
    for (n, v) in &ss.steady_state.agg {
        assert_close(&format!("agg.{n}"), &[*v], &[ragg[n].as_f64().unwrap()], tol);
    }
    for (n, a) in &ss.steady_state.pol_values {
        let rp = &r["het"]["pol"][n];
        assert_eq!(a.dims.iter().map(|&x| x as f64).collect::<Vec<_>>(), flat(&rp["size"]), "pol.{n} size");
        assert_close(&format!("pol.{n} sum"), &[a.data.iter().sum()], &[rp["sum"].as_f64().unwrap()], pol_tol);
        let first = flat(&rp["first"]);
        assert_close(&format!("pol.{n} first"), &a.data[..first.len()], &first, tol);
    }
    if let Some(d) = ss.steady_state.d_hist.as_ref() {
        let rd = &r["het"]["d"];
        assert_close("d sum", &[d.data.iter().sum()], &[rd["sum"].as_f64().unwrap()], tol);
        let first = flat(&rd["first"]);
        assert_close("d first", &d.data[..first.len()], &first, tol);
    }
    // Jacobians
    let dr = st.dr.as_ref().unwrap();
    assert_eq!(dr.truncation_horizon as f64, r["het"]["T"].as_f64().unwrap());
    let mut n = 0;
    for (v, by_shock) in r["het"]["G"].as_object().unwrap() {
        for (s, g0) in by_shock.as_object().unwrap() {
            let want = flat(g0);
            let g = dr.g_block(v, s).unwrap_or_else(|| panic!("G.{v}.{s}"));
            let got: Vec<f64> = (0..want.len()).map(|i| g[(i, 0)]).collect();
            assert_close(&format!("G.{v}.{s}"), &got, &want, tol);
            n += 1;
        }
    }
    assert!(n > 0);
    // simulation
    let sim = st.sim.as_ref().unwrap();
    if let Some(ri) = r["irfs"].as_object() {
        assert_eq!(sim.irfs.len(), ri.len(), "number of IRFs");
        for (k, v) in &sim.irfs {
            assert_close(&format!("irf {k}"), v, &flat(&ri[k]), tol);
        }
    }
    if let Some(np) = r["het"]["NP"].as_u64() {
        // stochastic paths given MATLAB's draws
        let np = np as usize;
        let e = rows(&r["het"]["eps_draws"]);
        let eps = Mat::from_fn(e.len(), e[0].len(), |i, j| e[i][j]);
        let y = &ss.mat.y[mf.model.n_endo()..2 * mf.model.n_endo()];
        let shocks: Vec<usize> = (0..mf.model.exo_names.len()).collect();
        let p = dynare::heterogeneity::simulate::stochastic_paths(&mf.model, dr, y, &eps, np, &shocks).unwrap();
        let want = rows(&r["endo_simul"]);
        assert_eq!(p.nrows(), want.len());
        for (i, w) in want.iter().enumerate() {
            let got: Vec<f64> = (0..np).map(|t| p[(i, t)]).collect();
            assert_close(&format!("endo_simul {}", mf.model.endo_names[i]), &got, w, tol);
        }
        // our own draws: the shape
        let es = sim.endo_simul.as_ref().unwrap();
        assert_eq!(es.nrows(), mf.model.n_endo());
    } else if !mf.det_shocks.is_empty() {
        // news shocks: deterministic paths
        let es = sim.endo_simul.as_ref().unwrap();
        let want = rows(&r["endo_simul"]);
        assert_eq!((es.nrows(), es.ncols()), (want.len(), want[0].len()), "endo_simul size");
        for (i, w) in want.iter().enumerate() {
            let got: Vec<f64> = (0..es.ncols()).map(|t| es[(i, t)]).collect();
            assert_close(&format!("endo_simul {}", mf.model.endo_names[i]), &got, w, tol);
        }
    }
}

#[test]
fn krusell_smith_1998() {
    check_het("krusell_smith_1998", 1e-8, 1e-8);
}

#[test]
fn hank_one_asset() {
    check_het("hank_one_asset", 1e-8, 1e-8);
}

#[test]
fn hank_two_assets() {
    check_het("hank_two_assets", 1e-8, 1e-8);
}

/// `a` (built by dynare-rust) equals `b` (MATLAB's), recursively.
fn same_value(path: &str, a: &MatValue, b: &MatValue) {
    match (a, b) {
        (MatValue::Numeric { dims: da, data: xa }, MatValue::Numeric { dims: db, data: xb }) => {
            assert_eq!(da, db, "{path}: dims");
            assert_close(path, xa, xb, 1e-12); // pow() rounding
        }
        (MatValue::Char(x), MatValue::Char(y)) => assert_eq!(x, y, "{path}"),
        (MatValue::Cell { dims: da, items: ia }, MatValue::Cell { dims: db, items: ib }) => {
            assert_eq!(da, db, "{path}: dims");
            for (k, (x, y)) in ia.iter().zip(ib).enumerate() {
                same_value(&format!("{path}{{{k}}}"), x, y);
            }
        }
        (MatValue::Struct { fields: fa, elems: ea, .. }, MatValue::Struct { fields: fb, elems: eb, .. }) => {
            assert_eq!(fa, fb, "{path}: fields");
            for (f, (x, y)) in fa.iter().zip(ea[0].iter().zip(&eb[0])) {
                same_value(&format!("{path}.{f}"), x, y);
            }
        }
        _ => panic!("{path}: {a:?} vs {b:?}"),
    }
}

/// The `initial_guess` structure built by the example's `verbatim` blocks equals MATLAB's
/// (initial_guess_ref.mat, saved by dump_het.m).
fn check_verbatim_initial_guess(name: &str) {
    let d = dir(name);
    let mf = parse_mod_file(&d.join(format!("{name}.mod")), &[]).unwrap();
    let want = dynare::matfile::read(&d.join("initial_guess_ref.mat")).unwrap();
    same_value("initial_guess", &mf.verbatim_workspace["initial_guess"], &want["initial_guess"]);
}

/// verbatim blocks (logspace, rouwenhorst, structures, cells) building the initial guess, a
/// parameter assignment using a verbatim variable (`Z_ss = (rk / alpha) ^ alpha;`),
/// heterogeneity_compute_steady_state calibrating beta, then solve and IRFs.
#[test]
fn krusell_smith_1998_steady_state() {
    check_verbatim_initial_guess("krusell_smith_1998_steady_state");
    check_het("krusell_smith_1998_steady_state", 1e-6, 1e-6);
}

/// Calibration of beta and vphi on two target equations; stochastic simulation.
#[test]
fn hank_one_asset_steady_state() {
    check_verbatim_initial_guess("hank_one_asset_steady_state");
    check_het("hank_one_asset_steady_state", 1e-6, 1e-6);
}

/// 3-D initial guess (reshape, implicit expansion, max), three calibrated parameters; news shocks.
/// Parameters agree to 1e-13 and everything else to 1e-6, except the policy functions at the
/// massless corner of the grid (largest a and b, lowest e, where c < 0): there Va differs by
/// 6e-8 (5e-4 relative), c by 0.03, so the sums of c, a, b differ by up to 5e-6 relative.
#[test]
fn hank_two_assets_steady_state() {
    check_verbatim_initial_guess("hank_two_assets_steady_state");
    // The massless corner moves with the libm: the sum of c differs by 1.3e-5 on Windows.
    check_het("hank_two_assets_steady_state", 1e-6, if cfg!(target_os = "macos") { 1e-5 } else { 5e-5 });
}

// ---------------------------------------------------------------------------
// optimal_policy
// ---------------------------------------------------------------------------

use dynare::policy::{Session, handles as policy_handles};

fn run_policy(name: &str, defines: &[(String, String)]) -> (ModFile, Session, Value) {
    let d = dir(name);
    let r = reference(name);
    let mf = parse_mod_file(&d.join(format!("{name}.mod")), defines).unwrap();
    let mut s = Session::new(true);
    for c in &mf.commands {
        if policy_handles(&mf, c) {
            s.run(&mf, c, None).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        }
    }
    (mf, s, r)
}

/// `oo_.dr` matrices: rows in `order_var`, as MATLAB's.
fn check_dr_mat(what: &str, got: &Mat, want: &Value, tol: f64) {
    let mut w = rows(want);
    if got.ncols() == 1 && w.len() == 1 && got.nrows() > 1 {
        // a column vector is jsonencoded as a flat array
        w = w[0].iter().map(|&x| vec![x]).collect();
    }
    assert_eq!(got.nrows(), w.len(), "{what} rows");
    for (i, wr) in w.iter().enumerate() {
        let g: Vec<f64> = (0..got.ncols()).map(|j| got[(i, j)]).collect();
        assert_close(&format!("{what} row {i}"), &g, wr, tol);
    }
}

fn check_policy_dr(s: &Session, r: &Value, tol: f64) {
    let dr = s.dr.as_ref().unwrap();
    assert_close("steady state", &dr.ys, &flat(&r["dr"]["ys"]), tol);
    let ov: Vec<f64> = dr.order_var.iter().map(|&v| (v + 1) as f64).collect();
    assert_eq!(ov, flat(&r["dr"]["order_var"]));
    check_dr_mat("ghx", &dr.ghx, &r["dr"]["ghx"], tol);
    check_dr_mat("ghu", &dr.ghu, &r["dr"]["ghu"], tol);
}

/// nk_ramsey_osr.mod with its defaults: ramsey_model + stoch_simul(order=1, periods=500) +
/// evaluate_planner_objective. The conditional welfare uses the first simulated shock
/// (`oo_.exo_simul(1,:)`): checked with MATLAB's draw.
#[test]
fn nk_ramsey_osr_ramsey() {
    check_ramsey_order1("nk_ramsey_osr");
}

/// nk_ramsey_steady_file.mod: the same Ramsey problem with the steady state from
/// nk_ramsey_steady_file_steadystate.m (Dynare's template: run as the equivalent
/// steady_state_model, conditional on the instrument R).
#[test]
fn nk_ramsey_steady_file() {
    check_ramsey_order1("nk_ramsey_steady_file");
}

fn check_ramsey_order1(name: &str) {
    let (mf, mut s, r) = run_policy(name, &[]);
    check_policy_dr(&s, &r, 1e-8);
    let v = s.planner_objective_value.clone().unwrap();
    let pr = &r["planner_objective_value"];
    assert_close("unconditional welfare", &[v.unconditional], &[pr["unconditional"].as_f64().unwrap()], 1e-8);
    // no simulation here (no output context): u = 0, and at order 1 the conditional welfare with
    // steady-state multipliers is the unconditional one
    assert_close("W_L_SS (u = 0)", &[v.conditional_steady_initial_multiplier], &[v.unconditional], 1e-10);
    // with MATLAB's first draw
    s.exo_simul_first = Some(flat(&r["exo_simul_first"]));
    let c = Command::Policy("evaluate_planner_objective".into(), Vec::new(), Vec::new());
    s.run(&mf, &c, None).unwrap();
    let v = s.planner_objective_value.clone().unwrap();
    assert_close(
        "conditional welfare",
        &[v.conditional_zero_initial_multiplier, v.conditional_steady_initial_multiplier],
        &[
            pr["conditional"]["zero_initial_multiplier"].as_f64().unwrap(),
            pr["conditional"]["steady_initial_multiplier"].as_f64().unwrap(),
        ],
        1e-8,
    );
}

/// nk_ramsey_osr.mod with -DRamsey=0: osr(opt_algo=9 (CMA-ES), order=2) of the Taylor-rule
/// coefficient, which ends at its upper bound; then the second-order decision rule.
#[test]
fn nk_ramsey_osr_osr() {
    let (_mf, s, r) = run_policy("nk_ramsey_osr", &[("Ramsey".into(), "0".into())]);
    let r = &r["osr_variant"];
    let o = s.osr.as_ref().unwrap();
    assert_close("OSR loss", &[o.objective_function], &[r["osr"]["objective_function"].as_f64().unwrap()], 1e-8);
    for (n, v) in &o.optim_params {
        assert_close(n, &[*v], &[r["osr"]["optim_params"][n].as_f64().unwrap()], 1e-6);
    }
    check_policy_dr(&s, r, 1e-6);
    let k = s.kdr.as_ref().expect("second-order decision rule");
    for (n, m) in [("ghxx", &k.ghxx), ("ghxu", &k.ghxu), ("ghuu", &k.ghuu), ("ghs2", &k.ghs2)] {
        check_dr_mat(n, m, &r["dr"][n], 1e-6);
    }
}

/// Masks every numeric token (whitespace-separated) of a line so the layout can be compared
/// exactly; the numbers are returned separately.
fn mask_numbers(line: &str) -> (String, Vec<f64>) {
    let mut out = String::new();
    let mut nums = Vec::new();
    let mut tok = String::new();
    let flush = |tok: &mut String, out: &mut String, nums: &mut Vec<f64>| {
        if tok.is_empty() {
            return;
        }
        match tok.parse::<f64>() {
            Ok(v) if tok.chars().next().is_some_and(|c| c.is_ascii_digit() || c == '-' || c == '.') => {
                nums.push(v);
                out.push('#');
            }
            _ => out.push_str(tok),
        }
        tok.clear();
    };
    for c in line.chars() {
        if c == ' ' || c == '\t' {
            flush(&mut tok, &mut out, &mut nums);
            out.push(c);
        } else {
            tok.push(c);
        }
    }
    flush(&mut tok, &mut out, &mut nums);
    (out, nums)
}

/// The console tables of `heterogeneity_load_steady_state` (STEADY STATE, RESIDUALS) have
/// Dynare's layout (padding, headers, number formats) line by line; the numbers agree to 1e-6
/// (residuals at the 1e-10 level are solver noise, so an absolute 1e-9 is allowed there).
fn check_display(name: &str) {
    let (mf, st) = run_het(name);
    let ss = st.steady_state.as_ref().unwrap();
    let h = mf.het.model.as_ref().unwrap();
    let got = format!(
        "{}{}",
        dynare::heterogeneity::steady::steady_state_text(&mf.model, h, ss),
        dynare::heterogeneity::steady::residuals_text(&mf.model, h, ss)
    );
    let want = std::fs::read_to_string(dir(name).join("display.txt")).unwrap();
    let got_l: Vec<&str> = got.lines().skip_while(|l| l.is_empty()).collect();
    let want_l: Vec<&str> = want.lines().collect();
    assert_eq!(got_l.len(), want_l.len(), "{name}: number of lines\n{got}");
    for (g, w) in got_l.iter().zip(&want_l) {
        let (gm, gn) = mask_numbers(g);
        let (wm, wn) = mask_numbers(w);
        assert_eq!(gm, wm, "{name}: layout of {g:?} vs {w:?}");
        for (a, b) in gn.iter().zip(&wn) {
            assert!((a - b).abs() <= 1e-9 + 1e-6 * b.abs(), "{name}: {g:?} vs {w:?}");
        }
    }
}

#[test]
fn display_krusell_smith_1998() {
    check_display("krusell_smith_1998");
}

#[test]
fn display_hank_one_asset() {
    check_display("hank_one_asset");
}

#[test]
fn display_hank_two_assets() {
    check_display("hank_two_assets");
}

/// `fp` is a fingerprint (sizes, sum, absolute sum, i-, j- and ij-weighted sums) of a matrix.
fn fingerprint(a: &Mat) -> Vec<f64> {
    let (mut s, mut abs, mut wi, mut wj, mut wij) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let v = a[(i, j)];
            let (fi, fj) = ((i + 1) as f64, (j + 1) as f64);
            s += v;
            abs += v.abs();
            wi += v * fi;
            wj += v * fj;
            wij += v * fi * fj;
        }
    }
    vec![s, abs, wi, wj, wij]
}

fn check_fp(what: &str, got: &Mat, want: &Value) {
    assert_eq!(vec![got.nrows() as f64, got.ncols() as f64], flat(&want["size"]), "{what} size");
    let w = [&want["sum"], &want["abs"], &want["wi"], &want["wj"], &want["wij"]];
    let g = fingerprint(got);
    let scale = want["abs"].as_f64().unwrap().max(1e-300);
    if scale < 1e-6 {
        // a block that is zero in exact arithmetic: both sides are rounding noise
        assert!(fingerprint(got)[1] < 1e-6, "{what}: expected a (numerically) zero block");
        return;
    }
    for (k, (a, b)) in g.iter().zip(w).enumerate() {
        let b = b.as_f64().unwrap();
        // k = 0: the plain sum; 2..: weighted sums are as large as `abs` times the weights
        let sc =
            if k >= 2 { scale * (got.nrows().max(got.ncols()) as f64).powi(if k == 4 { 2 } else { 1 }) } else { scale };
        assert!((a - b).abs() <= 1e-8 * sc.max(1e-2), "{what} fingerprint[{k}]: {a} vs {b}");
    }
}

fn row_mat(v: &[f64]) -> Mat {
    Mat::from_fn(1, v.len(), |_, j| v[j])
}

/// Every field of the Dynare results at fingerprint level: all Jacobian blocks `dr.G.*.*` in
/// full (not only the first column), `mat.dG`, `mat.G`, `mat.F`, `mat.dF`, the policies and
/// distribution, the grids and the IRFs.
fn check_full(name: &str) {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(dir(name).join("fp.json")).unwrap()).unwrap();
    let (mf, st) = run_het(name);
    let ss = st.steady_state.as_ref().unwrap();
    let dr = st.dr.as_ref().unwrap();
    let mut n = 0;
    for (v, by) in r["G"].as_object().unwrap() {
        for (s, fp) in by.as_object().unwrap() {
            check_fp(&format!("G.{v}.{s}"), dr.g_block(v, s).unwrap(), fp);
            n += 1;
        }
    }
    assert_eq!(n, dr.g.len(), "number of G blocks");
    check_fp("dG", ss.mat.dg.as_ref().unwrap(), &r["dG"]);
    assert_close("mat.G", &ss.mat.g, &flat(&r["mat_G"]), 1e-8);
    check_fp("mat.F", ss.mat.f.as_ref().unwrap(), &r["mat_F"]);
    let (nh, nc, np) = ss.mat.df_dims;
    check_fp("mat.dF", &Mat::from_fn(nh, nc * np, |i, j| ss.mat.df[i + nh * j]), &r["mat_dF"]);
    check_fp("d.hist", &row_mat(&ss.steady_state.d_hist.as_ref().unwrap().data), &r["d_hist"]);
    for (k, a) in &ss.steady_state.pol_values {
        check_fp(&format!("pol.{k}"), &row_mat(&a.data), &r["pol"][k]);
    }
    for (k, g) in r["d_grids"].as_object().unwrap() {
        assert_close(&format!("d.grids.{k}"), ss.steady_state.d_grid(k), &flat(g), 1e-12);
    }
    if let Some(ri) = r["irfs"].as_object() {
        let sim = st.sim.as_ref().unwrap();
        for (k, fp) in ri {
            check_fp(&format!("irf {k}"), &row_mat(&sim.irfs.iter().find(|(n, _)| n == k).unwrap().1), fp);
        }
    }
    let _ = mf;
}

#[test]
fn full_krusell_smith_1998() {
    check_full("krusell_smith_1998");
}

#[test]
fn full_hank_one_asset() {
    check_full("hank_one_asset");
}

#[test]
fn full_hank_two_assets() {
    check_full("hank_two_assets");
}
