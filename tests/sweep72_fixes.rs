//! Behaviours of Dynare 7.2 found by running its test suite (second batch):
//! - `steady(solve_algo=10)` and the `resid` that follows (tests/lmmcp/SMOPEC.mod);
//! - `model(bytecode)` with `steady(solve_algo=5)` and `stack_solve_algo=5`
//!   (tests/steady_state_operator/bytecode_test.mod; reference/sweep72/bytecode_test.json);
//! - `var(log, deflator=...)` (tests/log_transform/fs2000_nonstationary.mod;
//!   reference/sweep72/fs2000_nonstationary_log.json);
//! - `dseries('file')` in the MATLAB-like statements (tests/smoother2histval);
//! - v7.3 (HDF5) MAT-files (tests/occbin/filter/dataobsfile.mat);
//! - `homotopy_setup(from_initval_to_endval)` with an empty block
//!   (tests/steady_state/homotopy/homotopy_from_initval_to_endval.mod);
//! - on-the-fly declarations `c|e`, `e|x`, `beta|p` (tests/on-the-fly/ex1.mod, ex2.mod);
//! - a positive `time_shift` of `var_expectation_model` (tests/var-expectations/1-with-time-shift-b).
mod common;

use std::path::{Path, PathBuf};

use dynare::modfile::parse_mod_file;
use serde_json::Value;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sweep72f_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sweep72")
}

fn run_cli(dir: &Path, file: &str) -> (bool, String) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_dynare-rs"))
        .args([file, "nograph", "nointeractive"])
        .current_dir(dir)
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    (out.status.success(), s)
}

fn copy_ref(dir: &Path, file: &str) {
    std::fs::copy(refdir().join(file), dir.join(file)).unwrap();
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(g: f64, w: f64, tol: f64, what: &str) {
    assert!((g - w).abs() <= tol * (1.0 + w.abs()), "{what}: {g} vs Dynare {w}");
}

#[test]
fn lmmcp_steady_state_and_resid() {
    let d = tmp("smopec");
    copy_ref(&d, "SMOPEC.mod");
    let (ok, out) = run_cli(&d, "SMOPEC.mod");
    assert!(ok, "{out}");
    // before `steady`, solve_algo = 4: the conditions are ignored; after it, the residual of
    // the Euler equation (d at its upper bound) is set to 0
    let first = out.find("Residuals of the static equations (ignoring complementarity conditions):").unwrap();
    let second = out.find("Residuals of the static equations (accounting for complementarity conditions):").unwrap();
    assert!(first < second, "{out}");
    assert!(out[first..second].contains("-0.069565"), "{out}");
    assert!(out[second..].contains("Euler equation    :       0.000000"), "{out}");
    assert!(out.contains("d \t\t 2\n"), "{out}");
}

#[test]
fn bytecode_solve_algo_5() {
    let d = tmp("bytecode");
    copy_ref(&d, "bytecode_test.mod");
    let (ok, out) = run_cli(&d, "bytecode_test.mod");
    assert!(ok, "{out}");
    let r: Value = common::json_ref(refdir().join("bytecode_test.json"));
    let want = nums(&r["endo_simul"]);
    let n = r["endo_simul_size"][0].as_u64().unwrap() as usize;
    let csv = std::fs::read_to_string(d.join("bytecode_test/Output/bytecode_test_endo_simul.csv")).unwrap();
    let rows: Vec<Vec<f64>> =
        csv.lines().skip(1).map(|l| l.split(',').skip(1).map(|x| x.parse().unwrap()).collect()).collect();
    assert_eq!(rows.len() * n, want.len());
    for (t, row) in rows.iter().enumerate() {
        for i in 0..n {
            close(row[i], want[t * n + i], 1e-9, &format!("endo_simul({i},{t})"));
        }
    }
}

#[test]
fn log_transform_with_deflator() {
    let d = tmp("logdefl");
    copy_ref(&d, "fs2000_nonstationary_log.mod");
    let (ok, out) = run_cli(&d, "fs2000_nonstationary_log.mod");
    assert!(ok, "{out}");
    let r: Value = common::json_ref(refdir().join("fs2000_nonstationary_log.json"));
    let p = d.join("fs2000_nonstationary_log/Output/fs2000_nonstationary_log_stoch_simul.json");
    let j: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    for (k, (g, w)) in nums(&j["dr"]["ys"]).iter().zip(nums(&r["steady_state"])).enumerate() {
        close(*g, w, 1e-10, &format!("steady_state({k})"));
    }
    let var: Vec<f64> = j["var"].as_array().unwrap().iter().flat_map(nums).collect();
    let want = nums(&r["var"]);
    assert_eq!(var.len(), want.len());
    // oo_.var is symmetric: row-major here, column-major in the reference
    for (k, (g, w)) in var.iter().zip(&want).enumerate() {
        close(*g, *w, 1e-8, &format!("var[{k}]"));
    }
}

#[test]
fn dseries_from_a_file() {
    let d = tmp("dseries_file");
    std::fs::write(d.join("h.csv"), "Date,y\n1Y,1\n2Y,2\n").unwrap();
    let m = "var y; varexo e; parameters a; a = 0.5;\nmodel; y = a*y(-1) + e; end;\n\
             steady;\nhf = dseries('h.csv');\n";
    std::fs::write(d.join("ds.mod"), m).unwrap();
    let (ok, out) = run_cli(&d, "ds.mod");
    assert!(ok, "{out}");
    let (ok, out) = {
        std::fs::write(d.join("ds2.mod"), m.replace("h.csv", "missing.csv")).unwrap();
        run_cli(&d, "ds2.mod")
    };
    assert!(!ok, "{out}");
}

#[test]
fn mat_v73_hdf5() {
    let m = dynare::matfile::read(&refdir().join("dataobsfile_v73.mat")).unwrap();
    assert_eq!(m.keys().cloned().collect::<Vec<_>>(), ["inom", "inom0", "pi", "yg"]);
    let first = |k: &str| match &m[k] {
        dynare::matfile::MatValue::Numeric { dims, data } => {
            assert_eq!(dims, &[120, 1]);
            data[0]
        }
        v => panic!("{k}: {v:?}"),
    };
    assert_eq!(first("inom"), 1.00012600255453);
    assert_eq!(first("pi"), 1.00388898152081);
    assert_eq!(first("yg"), 1.00223961316058);
}

#[test]
fn homotopy_from_initval_to_endval_with_an_empty_block() {
    let d = tmp("homotopy");
    let m = "var c k; varexo x; parameters alph gam delt bet aa;\n\
             alph=0.5; gam=0.5; delt=0.02; aa=0.5; bet=0.05;\nmodel;\n\
             c + k - aa*x*k(-1)^alph - (1-delt)*k(-1);\n\
             c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);\nend;\n\
             initval; x = 1; k = ((delt+bet)/(aa*x*alph))^(1/(alph-1)); c = aa*x*k^alph-delt*k; end;\n\
             steady;\nendval; x = 2; end;\nhomotopy_setup(from_initval_to_endval);\nend;\n\
             steady(homotopy_mode = 3, homotopy_steps = 50);\n";
    std::fs::write(d.join("h.mod"), m).unwrap();
    let (ok, out) = run_cli(&d, "h.mod");
    assert!(ok, "{out}");
    // Dynare 7.2: c = 6.1224487931112, k = 51.020401980200866
    assert!(out.contains("c               \t\t 6.12245\nk               \t\t 51.0204"), "{out}");
    // without any homotopy_setup block, homotopy_mode is an error
    std::fs::write(d.join("h2.mod"), m.replace("homotopy_setup(from_initval_to_endval);\nend;\n", "")).unwrap();
    let (ok, out) = run_cli(&d, "h2.mod");
    assert!(!ok && out.contains("a homotopy_setup block must be present"), "{out}");
}

#[test]
fn on_the_fly_declarations() {
    let d = tmp("on_the_fly");
    // tests/on-the-fly/ex1.mod and ex2.mod (nostrict), without their MATLAB checks; the
    // symbols are ranked by first appearance, as Dynare 7.2's M_.*_names
    let ex1 = "model;\nc*theta|p*h|e^(1+psi|p)=(1-alpha)*y;\nk|e = beta|p*(((exp(b)*c)/(exp(b(+1))*c(+1)))\n\
               *(exp(b(+1))*alpha|p*y(+1)+(1-delta)*k));\ny|e = exp(a)*(k(-1)^alpha)*(h^(1-alpha));\n\
               k = exp(b)*(y-c|e)+(1-delta|p)*k(-1);\na|e = rho|p*a(-1)+tau*b(-1) + e|x;\n\
               b|e = tau|p*a(-1)+rho*b(-1) + u|x;\nend;\n";
    std::fs::write(d.join("ex1.mod"), ex1).unwrap();
    let mf = parse_mod_file(&d.join("ex1.mod"), &[]).unwrap();
    assert_eq!(mf.model.endo_names[..6], ["c", "h", "y", "k", "b", "a"]);
    assert_eq!(mf.model.exo_names, ["e", "u"]);
    assert_eq!(mf.model.param_names, ["theta", "psi", "alpha", "beta", "delta", "rho", "tau"]);
    let ex2 = ex1.replace("c*theta|p*h|e^(1+psi|p)=(1-alpha)*y;\n", "").replace("u|x;", "u|x + .0*theta;");
    std::fs::write(d.join("ex2.mod"), ex2).unwrap();
    let o = dynare::modfile::ParseOptions { nostrict: true, ..Default::default() };
    let mf = dynare::modfile::parse_mod_file_opts(&d.join("ex2.mod"), &o).unwrap();
    assert_eq!(mf.model.endo_names[..5], ["k", "b", "c", "y", "a"]);
    assert_eq!(mf.model.exo_names, ["h", "e", "u", "theta"]);
    assert_eq!(mf.model.param_names, ["beta", "alpha", "delta", "rho", "tau"]);
}

#[test]
fn var_expectation_positive_time_shift() {
    let d = tmp("time_shift");
    let m = |ts: &str| {
        format!(
            "var x z; varexo e_x e_z; parameters a f beta; a = .9; f = .8; beta = 1/1.02;\n\
             var_model(model_name = toto, eqtags = [ 'X' 'Z' ]);\n\
             var_expectation_model(model_name = varexp, expression = x, auxiliary_model_name = toto, \
             horizon = 1, time_shift={ts}, discount = beta);\n\
             model;\n[ name = 'X' ]\nx = a*x(-1) + z(-1) + e_x;\n[ name = 'Z' ]\nz = f*z(-1) + e_z;\nend;\n"
        )
    };
    std::fs::write(d.join("ts1.mod"), m("1")).unwrap();
    let e = format!("{:#}", parse_mod_file(&d.join("ts1.mod"), &[]).unwrap_err());
    assert!(e.contains("The 'time_shift' option must be a non-positive integer"), "{e}");
    std::fs::write(d.join("ts0.mod"), m("-1")).unwrap();
    parse_mod_file(&d.join("ts0.mod"), &[]).unwrap();
}
