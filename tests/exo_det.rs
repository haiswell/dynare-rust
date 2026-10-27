//! `varexo_det` against Dynare 7.2 (reference/exo_det/, written by dump_exo_det.m and
//! dump_pp.sh): order-2 decision rules (`ghud`, `ghxud`, `ghuud`, `ghudud` of
//! `stochastic_solvers.m`), order-2 forecasts (`simult_varexo_det.m`, bands replayed on
//! MATLAB's normals), order-1 forecasts with two `varexo_det` and from `histval`, `periods`
//! simulations and IRFs (no `varexo_det` effect), estimation (smoother, `forecast=H` with
//! `make_ex_.m`'s zeros, shock decomposition, conditional forecast), homotopy and
//! `save_params_and_steady_state`, the JSON output, and every error Dynare raises.

use dynare::deterministic::exo_det;
use dynare::forecast::commands::{ForecastOrder, forecast_command};
use dynare::linalg::Mat;
use dynare::modfile::{ModFile, OptValue, parse_mod_file};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/exo_det")
}

fn load(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(refdir().join(format!("ref_{name}.json"))).unwrap()).unwrap()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        x => vec![num(x)],
    }
}

/// A MATLAB matrix as jsonencode writes it: rows of numbers, or a flat list for a vector.
fn mat(v: &Value, nrows: usize, ncols: usize) -> Mat {
    match v {
        Value::Array(rows) if rows.first().is_some_and(|r| r.is_array()) => {
            Mat::from_fn(nrows, ncols, |i, j| num(&rows[i][j]))
        }
        _ => {
            let x = vecf(v);
            assert_eq!(x.len(), nrows * ncols);
            Mat::from_fn(nrows, ncols, |i, j| x[i * ncols + j])
        }
    }
}

fn assert_mat(what: &str, a: &Mat, b: &Mat, tol: f64) {
    assert_eq!(a.shape(), b.shape(), "{what}: shape");
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let (x, y) = (a[(i, j)], b[(i, j)]);
            assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{what}({i},{j}): {x} vs {y}");
        }
    }
}

fn parse(name: &str) -> ModFile {
    parse_mod_file(&refdir().join(format!("{name}.mod")), &[]).unwrap()
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("dynare_exo_det_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn forecast_opts(periods: usize) -> Vec<(String, OptValue)> {
    vec![("periods".into(), OptValue::Num(periods as f64)), ("nograph".into(), OptValue::Flag)]
}

/// `oo_.forecast.{Mean,HPDinf,HPDsup}` of `vars` against a forecast.
fn check_forecast(what: &str, f: &dynare::forecast::Forecast, want: &Value, bands: bool, tol: f64) {
    let keys: &[(&str, &Mat)] = &[("Mean", &f.mean), ("HPDinf", &f.lower), ("HPDsup", &f.upper)];
    for &(key, m) in if bands { keys } else { &keys[..1] } {
        for (c, name) in f.names.iter().enumerate() {
            let w = vecf(&want[key][name.as_str()]);
            assert_eq!(w.len(), m.nrows(), "{what}: {key}.{name} length");
            for (h, &y) in w.iter().enumerate() {
                let x = m[(h, c)];
                assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{what}: {key}.{name}[{h}]: {x} vs {y}");
            }
        }
    }
}

/// `dr.ghud`, `dr.ghxud`, `dr.ghuud`, `dr.ghudud` at order 2 (`stochastic_solvers.m`), on
/// dynare-rust's order-2 rule.
#[test]
fn order2_decision_rules() {
    for name in ["xdet_o2", "xdet_o2m", "xdet_o2n"] {
        let r = load(name);
        let mf = parse(name);
        let m = &mf.model;
        let (n, ne, nd) = (m.n_endo(), m.n_exo(), m.exo_det_names.len());
        let sol = dynare::analysis::solve_calibration(&mf, 1.0 + 1e-6).unwrap().unwrap();
        let ys = sol.dr.ys.clone();
        assert_mat("ys", &Mat::from_column_slice(n, 1, &ys), &mat(&r["ys"], n, 1), 1e-12);
        let dr2 = dynare::solver::korder::solve_model(m, &ys, &sol.params, &mf.sigma_e, 2, Default::default()).unwrap();
        let nsp = dr2.nspred();
        let len = exo_det::exo_det_length(&mf);
        assert_eq!(len as u64, r["exo_det_length"].as_u64().unwrap(), "{name}: exo_det_length");
        let xd = exo_det::second_order(m, &dr2, &ys, &sol.params, len);
        // cells: a list of matrices (ghudud: the l x l cell in column-major order, j <= i filled;
        // MATLAB keeps ghud{1} etc. when exo_det_length = 0)
        let l = xd.ghud.len();
        assert_eq!(l, len.max(1));
        let cell = |key: &str| r[key].as_array().unwrap().clone();
        let (gu, gx, guu, gdd) = (cell("ghud"), cell("ghxud"), cell("ghuud"), cell("ghudud"));
        assert_eq!((gu.len(), gdd.len()), (l, l * l));
        for k in 0..l {
            let tol = 1e-9;
            assert_mat(&format!("{name} ghud{{{}}}", k + 1), &xd.ghud[k], &mat(&gu[k], n, nd), tol);
            assert_mat(&format!("{name} ghxud{{{}}}", k + 1), &xd.ghxud[k], &mat(&gx[k], n, nsp * nd), tol);
            assert_mat(&format!("{name} ghuud{{{}}}", k + 1), &xd.ghuud[k], &mat(&guu[k], n, ne * nd), tol);
        }
        for i in 0..l {
            for j in 0..=i {
                assert_mat(
                    &format!("{name} ghudud{{{},{}}}", j + 1, i + 1),
                    &xd.ghudud[j][i],
                    &mat(&gdd[j + i * l], n, nd * nd),
                    1e-9,
                );
            }
        }
    }
}

/// Order-2 forecasts (`simult_varexo_det.m`): the mean (no shocks, `ghs2/2` every period)
/// and, with 50 simulations on MATLAB's normals, the bands (`quantile`).
#[test]
fn order2_forecasts() {
    // and, for comparison, forecasts.stochastic_classical at orders 2 and 3 (pruned) without varexo_det
    for (name, order, pruning) in
        [("xdet_o2", 2, false), ("xdet_o2m", 2, false), ("xdet_nox_o2", 2, false), ("xdet_nox_o3", 3, true)]
    {
        let r = load(name);
        let mf = parse(name);
        let vars = mf
            .commands
            .iter()
            .find_map(|c| match c {
                dynare::modfile::Command::Forecast(_, v) => Some(v.clone()),
                _ => None,
            })
            .unwrap();
        let periods = r["replay_periods"].as_u64().unwrap() as usize;
        let tmp = scratch(name);
        let mut fo = ForecastOrder {
            order,
            pruning,
            periods,
            replic: 50,
            seed: 0,
            replay: Some(vecf(&r["replay_normals"])),
            ..ForecastOrder::first()
        };
        let f = forecast_command(&mf, None, &forecast_opts(periods), &vars, &tmp, name, &tmp, &fo).unwrap();
        check_forecast(name, &f, &r["replay_forecast"], true, 1e-10);
        // the run's own forecast (1000 simulations): same mean
        fo.replay = None;
        fo.replic = 1000;
        let f = forecast_command(&mf, None, &forecast_opts(periods), &vars, &tmp, name, &tmp, &fo).unwrap();
        check_forecast(name, &f, &r["forecast"], false, 1e-10);
    }
}

/// Order 1 with two `varexo_det`: forecast, IRFs, and the `periods` simulation on MATLAB's
/// shocks (`simult_.m` ignores the `varexo_det` paths).
#[test]
fn order1_two_exo_det() {
    let name = "xdet_o1m";
    let r = load(name);
    let mf = parse(name);
    let m = &mf.model;
    assert_eq!(m.exo_names, ["e", "u"]);
    assert_eq!(m.exo_det_names, ["g", "tau"]);
    assert_eq!(m.n_user_params(), 4);
    let tmp = scratch(name);
    let f = forecast_command(&mf, None, &forecast_opts(9), &[], &tmp, name, &tmp, &ForecastOrder::first()).unwrap();
    check_forecast(name, &f, &r["forecast"], true, 1e-10);
    let sol = dynare::analysis::solve_calibration(&mf, 1.0 + 1e-6).unwrap().unwrap();
    let n = m.n_endo();
    let ex = mat(&r["exo_simul"], 30, 2);
    let y = dynare::analysis::simulate::simult(&sol.dr, &sol.dr.ys, &ex);
    assert_mat("endo_simul", &y.columns(1, 30).into_owned(), &mat(&r["endo_simul"], n, 30), 1e-10);
    // IRFs (irf=4) of e: ghu column, ghx powers
    let ov = &sol.dr.order_var;
    let mut dev = vec![0.0; n];
    for t in 0..4 {
        let s: Vec<f64> = sol.dr.state_var.iter().map(|&v| dev[sol.dr.inv_order_var[v]]).collect();
        dev = (0..n)
            .map(|i| {
                (0..s.len()).map(|j| sol.dr.ghx[(i, j)] * s[j]).sum::<f64>()
                    + if t == 0 { sol.dr.ghu[(i, 0)] * 0.01 } else { 0.0 }
            })
            .collect();
        for (k, &v) in ov.iter().enumerate() {
            let want = vecf(&r["irfs"][format!("{}_e", m.endo_names[v])])[t];
            assert!((dev[k] - want).abs() < 1e-12, "irf {}_e[{t}]", m.endo_names[v]);
        }
    }
}

/// `varexo_det` declared before `varexo`; `histval` (also on the `varexo_det`, which Dynare
/// keeps in `M_.exo_det_histval` and never reads here), `perfect_foresight_setup` in a
/// stochastic file; the forecast starts from `M_.endo_histval`.
#[test]
fn histval_forecast() {
    let name = "xdet_hv";
    let r = load(name);
    let mf = parse(name);
    assert_eq!(mf.model.exo_names, ["e"]);
    assert_eq!(mf.model.exo_det_names, ["g"]);
    assert_eq!(mf.sigma_e.shape(), (1, 1));
    let tmp = scratch(name);
    let f = forecast_command(&mf, None, &forecast_opts(6), &[], &tmp, name, &tmp, &ForecastOrder::first()).unwrap();
    check_forecast(name, &f, &r["forecast"], true, 1e-10);
}

/// `forecast;` after `stoch_simul(periods=12)`: the horizon is `options_.periods`, and a
/// deterministic shock on an ordinary exogenous variable counts in `dr.exo_det_length`. Run
/// through the binary (the options persist between the commands).
#[test]
fn forecast_horizon_from_options_periods() {
    let name = "xdet_per";
    let r = load(name);
    let (dir, out, err, ok) = run_bin(name, &[]);
    assert!(ok, "{out}\n{err}");
    let fc = csv_rows(&dir.join(name).join("Output").join(format!("{name}_forecast.csv")));
    assert_eq!(fc.len() - 1, 12 * 4);
    for row in &fc[1..] {
        let (h, v) = (row[1].parse::<usize>().unwrap() - 1, row[2].as_str());
        for (k, key) in [(3, "Mean"), (4, "HPDinf"), (5, "HPDsup")] {
            let want = vecf(&r["forecast"][key][v])[h];
            let x: f64 = row[k].parse().unwrap();
            assert!((x - want).abs() < 1e-10 * (1.0 + want.abs()), "forecast {key}.{v}[{h}]: {x} vs {want}");
        }
    }
    assert_eq!(r["exo_det_length"].as_u64(), Some(6));
    assert_eq!(exo_det::exo_det_length(&parse(name)), 6);
}

fn run_bin(name: &str, extra: &[&str]) -> (PathBuf, String, String, bool) {
    let dir = scratch(&format!("bin_{name}"));
    for e in std::fs::read_dir(refdir()).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "mod" || x == "csv") {
            std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
        }
    }
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_dynare-rs"))
        .arg(format!("{name}.mod"))
        .args(extra)
        .current_dir(&dir)
        .output()
        .unwrap();
    (
        dir,
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

fn csv_rows(path: &Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(path).unwrap().lines().map(|l| l.split(',').map(str::to_string).collect()).collect()
}

/// Estimation with a `varexo_det` (a constant of the state space): smoothed variables,
/// `forecast=8` (the deterministic path with the zeros `make_ex_.m` leaves at
/// `options_.periods = 0`), shock decomposition and the conditional forecast's mean.
#[test]
fn estimation() {
    let name = "xdet_est";
    let r = load(name);
    let (dir, out, err, ok) = run_bin(name, &[]);
    assert!(ok, "{out}\n{err}");
    let o = dir.join(name).join("Output");
    // smoothed variables
    let sv = csv_rows(&o.join(format!("{name}_smoothed_variables.csv")));
    for (c, v) in sv[0].iter().enumerate().skip(1) {
        let want = vecf(&r["SmoothedVariables"][v.as_str()]);
        for (t, row) in sv[1..].iter().enumerate() {
            let x: f64 = row[c].parse().unwrap();
            assert!((x - want[t]).abs() < 1e-9 * (1.0 + want[t].abs()), "smoothed {v}[{t}]: {x} vs {}", want[t]);
        }
    }
    // forecast = 8 after the estimation
    let fc = csv_rows(&o.join(format!("{name}_forecast.csv")));
    for row in &fc[1..] {
        let (h, v) = (row[1].parse::<usize>().unwrap() - 1, row[2].as_str());
        for (k, key) in [(3, "Mean"), (4, "HPDinf"), (5, "HPDsup")] {
            let want = vecf(&r["forecast"][key][v])[h];
            let x: f64 = row[k].parse().unwrap();
            assert!((x - want).abs() < 1e-9 * (1.0 + want.abs()), "forecast {key}.{v}[{h}]: {x} vs {want}");
        }
    }
    assert_eq!(vecf(&r["forecast"]["Exogenous"]["g"]), [0.15, 0.15, 0.0, 0.2, 0.1, 0.1, 0.1, 0.1]);
    // shock decomposition of y (oo_.shock_decomposition: endo x (e, initial values, sum) x T)
    let sd = csv_rows(&o.join(format!("{name}_shock_decomposition.csv")));
    let w = r["shock_decomposition"].as_array().unwrap()[0].as_array().unwrap(); // y
    let mut n_checked = 0;
    for row in sd[1..].iter().filter(|row| row[1] == "y") {
        let t: usize = row[0].parse::<usize>().unwrap() - 1;
        let comp = match row[2].as_str() {
            "e" => 0,
            "Initial values" => 1,
            _ => 2,
        };
        let want = num(&w[comp][t]);
        let x: f64 = row[3].parse().unwrap();
        assert!((x - want).abs() < 1e-9, "shock decomposition y/{}[{t}]: {x} vs {want}", row[2]);
        n_checked += 1;
    }
    assert!(n_checked >= 160);
    // conditional forecast from the steady state (parameter_set = calibration): means
    let cfr = csv_rows(&o.join(format!("{name}_conditional_forecast.csv")));
    for (key, col) in [("cond", 3), ("uncond", 6)] {
        let want = vecf(&r["conditional_forecast"][key]["Mean"]["y"]);
        for row in cfr[1..].iter().filter(|row| row[2] == "y") {
            let h: usize = row[1].parse().unwrap();
            let x: f64 = row[col].parse().unwrap();
            assert!((x - want[h]).abs() < 1e-9, "conditional forecast {key} y[{h}]: {x} vs {}", want[h]);
        }
    }
}

/// Homotopy on a `varexo_det` (it moves `oo_.exo_det_steady_state`), then
/// `save_params_and_steady_state` (the `varexo_det` value last) and the forecast.
#[test]
fn homotopy_and_save_params() {
    let name = "xdet_hom";
    let r = load(name);
    let (dir, out, err, ok) = run_bin(name, &[]);
    assert!(ok, "{out}\n{err}");
    let ours = std::fs::read_to_string(dir.join("xdet_hom_ss.txt")).unwrap();
    let theirs = std::fs::read_to_string(refdir().join("xdet_hom_ss.txt")).unwrap();
    let parse = |s: &str| -> Vec<(String, f64)> {
        s.lines()
            .map(|l| {
                let (a, b) = l.split_once(' ').unwrap();
                (a.to_string(), b.parse().unwrap())
            })
            .collect()
    };
    let (a, b) = (parse(&ours), parse(&theirs));
    assert_eq!(a.iter().map(|x| &x.0).collect::<Vec<_>>(), b.iter().map(|x| &x.0).collect::<Vec<_>>());
    for ((n, x), (_, y)) in a.iter().zip(&b) {
        assert!((x - y).abs() < 1e-10 * (1.0 + y.abs()), "{n}: {x} vs {y}");
    }
    let fc = csv_rows(&dir.join(name).join("Output").join(format!("{name}_forecast.csv")));
    for row in &fc[1..] {
        let (h, v) = (row[1].parse::<usize>().unwrap() - 1, row[2].as_str());
        let want = vecf(&r["forecast"]["Mean"][v])[h];
        let x: f64 = row[3].parse().unwrap();
        assert!((x - want).abs() < 1e-9 * (1.0 + want.abs()), "forecast {v}[{h}]: {x} vs {want}");
    }
}

/// The message of Dynare's error: the preprocessor's (`ERROR: <file>: line L, cols a-b: msg`
/// -> `msg`), or MATLAB's `error()`.
fn dynare_error(name: &str) -> String {
    let pp = std::fs::read_to_string(refdir().join(format!("{name}.pp.txt"))).unwrap();
    if let Some(l) = pp.lines().find(|l| l.starts_with("ERROR: ")) {
        let m = &l[7..];
        return match m
            .find(": line ")
            .and_then(|i| m[i..].find(": ").map(|j| (i, j)))
            .and_then(|(i, j)| m[i + j + 2..].find(": ").map(|k| i + j + 2 + k + 2))
        {
            Some(k) if m.starts_with(name) => m[k..].to_string(),
            _ => m.to_string(),
        };
    }
    if let Some(l) = pp.lines().find(|l| l.starts_with("shocks: ")) {
        return l.to_string(); // printed without ERROR:
    }
    load(name)["err"].as_str().unwrap().to_string()
}

/// Every error Dynare raises on a `varexo_det` model, with the lines printed before it.
#[test]
fn errors_match_dynare() {
    let cases = [
        "xdet_e_leadlag",
        "xdet_e_pf",
        "xdet_e_mix",
        "xdet_e_stderr",
        "xdet_e_varlist",
        "xdet_e_irfshocks",
        "xdet_e_endval",
        "xdet_e_estim",
        "xdet_e_ep",
        "xdet_e_fwd",
        "xdet_e_bwd",
        "xdet_e_o3",
        "xdet_e_korder",
        "xdet_e_pruning",
        "xdet_e_dates",
        "xdet_e_epilogue",
    ];
    for name in cases {
        let want = dynare_error(name);
        let (_, out, err, ok) = run_bin(name, &[]);
        assert!(!ok, "{name}: no error\n{out}");
        let got = err
            .lines()
            .chain(out.lines())
            .find(|l| l.starts_with("ERROR: "))
            .unwrap_or_else(|| panic!("{name}: {out}\n{err}"));
        // dynare-rust's location is FILE:LINE (no columns)
        let got = got[7..].to_string();
        let got = match got.split_once(".mod:") {
            Some((_, rest)) => rest.split_once(": ").map_or(rest.to_string(), |x| x.1.to_string()),
            None => got,
        };
        assert_eq!(got, want, "{name}");
        // lines printed by MATLAB just before the error
        let log = std::fs::read_to_string(refdir().join(format!("{name}.log"))).unwrap();
        let before: Vec<&str> = log.lines().take_while(|l| !l.starts_with("ERRMSG: ")).collect();
        for l in
            before.iter().rev().take_while(|l| l.starts_with("STOCHASTIC_SOLVER") || l.starts_with("Index exceeds"))
        {
            assert!(out.lines().any(|o| o == *l), "{name}: missing line {l:?}\n{out}");
        }
    }
}

/// `json=compute`: modfile.json / modfile-original.json against the 7.1 preprocessor's
/// (`exogenous_deterministic`, `exogenousDet` AST nodes, the `varexo_det` Jacobian columns).
#[test]
fn json_matches_preprocessor() {
    let o = dynare::modfile::ParseOptions::default();
    dynare::model::dtree::json_capture();
    let mf = dynare::modfile::parse_mod_file_opts(&refdir().join("xdet_json.mod"), &o).unwrap();
    let cap = dynare::model::dtree::json_take().unwrap();
    let dir = scratch("json");
    let jo = dynare::q4::json::JsonOptions { stage: "compute".into(), stdout: false, derivsimple: false };
    dynare::q4::json::write(&mf, &cap, &dir, "xdet_json", &jo).unwrap();
    let skip =
        ["statements", "dynamic_tmp_nbr", "static_tmp_nbr", "dynamic_g2_sparse_indices", "dynamic_g3_sparse_indices"];
    fn norm(v: &Value) -> Value {
        match v {
            Value::Object(m) => {
                let mut m: serde_json::Map<String, Value> = m.iter().map(|(k, x)| (k.clone(), norm(x))).collect();
                if let Some(a3) = m.remove("arg3") {
                    m.insert("arg2".into(), a3);
                }
                Value::Object(m)
            }
            Value::Array(a) => Value::Array(a.iter().map(norm).collect()),
            x => x.clone(),
        }
    }
    for (ours, theirs) in
        [("modfile.json", "xdet_json_modfile.json"), ("modfile-original.json", "xdet_json_modfile-original.json")]
    {
        let want: Value =
            serde_json::from_str(&std::fs::read_to_string(refdir().join("json").join(theirs)).unwrap()).unwrap();
        let got: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("xdet_json/model/json").join(ours)).unwrap())
                .unwrap();
        for (k, w) in want.as_object().unwrap() {
            if skip.contains(&k.as_str()) {
                continue;
            }
            assert_eq!(got.get(k).map(norm).as_ref(), Some(w), "{theirs}: key {k}");
        }
    }
}

/// LaTeX: the `Exogenous Deterministic` table of `write_latex_definitions.m`, the parameter
/// table without the `varexo_det` values, `{G}_{t}` in the dynamic model.
#[test]
fn latex_matches_dynare() {
    let name = "xdet_latex";
    let (dir, out, err, ok) = run_bin(name, &[]);
    assert!(ok, "{out}\n{err}");
    for f in ["xdet_latex_latex_definitions.tex", "xdet_latex_latex_parameters.tex"] {
        let ours = std::fs::read_to_string(dir.join(name).join("latex").join(f)).unwrap();
        let theirs = std::fs::read_to_string(refdir().join("latex").join(f)).unwrap();
        assert_eq!(ours, theirs, "{f}");
    }
    let ours = std::fs::read_to_string(dir.join(name).join("latex/dynamic_content.tex")).unwrap();
    let theirs = std::fs::read_to_string(refdir().join("latex/dynamic_content.tex")).unwrap();
    let k_eq = |s: &str| s.lines().find(|l| l.starts_with("{k}_{t}=")).unwrap().to_string();
    assert_eq!(k_eq(&ours), k_eq(&theirs));
}
