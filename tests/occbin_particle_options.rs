//! The OccBin particle-filter options of `occbin_setup` (`particle_filtering`,
//! `filter_init_periods_using_particles`, `filter_particle_*`) against Dynare 7.2
//! (reference/occbin_est/ref_occbin_particle.json, dump_occbin_particle.m): the preprocessor
//! writes them into `options_occbin_.filter`, but `+occbin/setup.m` forwards only its fixed
//! list (`filter.use_relaxation` for the filter), so `options_.occbin.filter` keeps its
//! defaults, the particle path of the PKF (`occbin.ppf.engine`) is never entered and the
//! likelihood is the default PKF one. Checked: the options are accepted, the OccBin filter
//! options stay at Dynare's defaults, and the likelihood matches Dynare's at six points.
mod common;

use std::path::{Path, PathBuf};

use dynare::data::Table;
use dynare::dsge::{DsgeModel, LikelihoodData};
use dynare::kalman::KalmanOptions;
use dynare::modfile::{Command, OptValue, parse_mod_file};
use dynare::occbin::estimation::{OccbinEstimation, OccbinPosterior};
use dynare::occbin::{OccbinCommand, Session};
use serde_json::Value;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/occbin_est")
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        Value::Bool(b) => vec![*b as u8 as f64],
        _ => vec![f64::NAN],
    }
}

const OPTIONS: &[&str] = &[
    "particle_filtering",
    "filter_init_periods_using_particles",
    "filter_particle_number_of_particles",
    "filter_particle_number_of_shocks_per_particle",
    "filter_particle_draw_states_from_empirical_density",
    "filter_particle_state_importance_sampling_pkf_init",
    "filter_particle_state_importance_sampling_logpost_crit_threshold",
    "filter_particle_state_importance_sampling_slice_override_iteration",
    "filter_particle_state_importance_sampling_slice_burnin",
    "filter_particle_initial_state_ergodic_simul",
    "filter_particle_diagnostics",
    "filter_particle_diagnostics_graph_periods",
    "filter_particle_diagnostics_nograph",
    "filter_particle_state_draws",
    "filter_particle_use_pkf_updated_state_threshold",
];

#[test]
fn particle_options_are_not_forwarded() {
    let r: Value = serde_json::from_str(&common::read_ref(dir().join("ref_occbin_particle.json"))).unwrap();
    // the preprocessor did write the options (Dynare's options_occbin_)
    let given = &r["options_occbin_filter"];
    assert_eq!(given["particle"]["number_of_particles"], 50);
    assert_eq!(given["init_periods_using_particles"], true);
    // ... but options_.occbin.filter keeps its defaults
    let fp = &r["options_occbin_filter_particle"];
    assert_eq!(fp["status"], false);
    assert_eq!(fp["number_of_particles"], 127);
    let fs = &r["options_occbin_filter_scalars"];
    assert_eq!(fs["init_periods_using_particles"], false);

    let defs = vec![("PARTICLE".to_string(), "1".to_string())];
    let mf = parse_mod_file(&dir().join("nk_est.mod"), &defs).unwrap();
    let mut s = Session::new(&mf, "nk_est");
    let mut seen = Vec::new();
    for c in &mf.commands {
        if let Command::Occbin(OccbinCommand::Setup(o)) = c {
            for (k, v) in o {
                seen.push(k.clone());
                match k.as_str() {
                    "filter_particle_state_draws" => assert_eq!(v, &OptValue::Vec(vec![1.0, 2.0, 3.0])),
                    "filter_particle_diagnostics_graph_periods" => assert_eq!(v, &OptValue::Vec(vec![2.0, 3.0])),
                    _ => {}
                }
            }
            s.setup(&mf, o).unwrap();
        }
    }
    for o in OPTIONS {
        assert!(seen.iter().any(|k| k == o), "option {o} not parsed");
    }
    assert_eq!(s.filter.use_relaxation, fs["use_relaxation"].as_bool().unwrap());
    assert_eq!(s.filter.use_relaxation_tol_period as f64, fs["use_relaxation_tol_period"].as_f64().unwrap());
    assert_eq!(s.filter.periodic_solution, fs["periodic_solution"].as_bool().unwrap());

    // the PKF likelihood is Dynare's (the default PKF one)
    let x_init = vecf(&r["xparam_init"]);
    let dsge = DsgeModel::new(&mf, Some(&x_init)).unwrap();
    let table = Table::read_csv(&dir().join("nk_est_data.csv")).unwrap();
    let sample = table.sample(&mf.varobs, 1, None).unwrap();
    let data = LikelihoodData::new(&sample, KalmanOptions { lik_init: 1, ..Default::default() });
    let occ = OccbinEstimation::new(&mf, &s, &dsge, &data).unwrap().unwrap();
    assert!(!occ.inversion_filter);
    let post = OccbinPosterior { dsge: &dsge, data: &data, bounds: &dsge.bounds, occ: &occ };
    for (k, p) in r["points"].as_array().unwrap().iter().enumerate() {
        let want = p["fval"].as_f64().unwrap();
        let got = post.evaluate(&vecf(&p["x"])).unwrap_or_else(|f| panic!("point {k}: info {}", f.info)).fval;
        let rel = ((got - want) / want).abs();
        println!("point {k}: fval {got:.10} vs Dynare {want:.10} (rel {rel:.1e})");
        assert!(rel < 1e-8, "point {k}");
    }
}
