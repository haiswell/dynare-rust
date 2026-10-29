//! `heterogeneity_simulate(tex, graph_format = eps)`: the TeX eps-loader files
//! (`<fname>_IRF.tex`, `<fname>_simulation.tex`) equal those of Dynare 7.2 (reference/heterogeneity/tex,
//! written by Dynare 7.2 on ks.mod + `heterogeneity_solve(truncation_horizon = 40);
//! heterogeneity_simulate(irf = 30, periods = 25, tex, graph_format = eps);`), except the date line.

use std::path::Path;

use dynare::heterogeneity::commands::{HetSession, handles, run_command};
use dynare::modfile::{Command, parse_mod_file};

#[test]
fn het_simulate_tex_files() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/heterogeneity");
    let tmp = std::env::temp_dir().join(format!("het_tex_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    for f in ["ks_guess.mat", "krusell_smith_1998.mat"] {
        std::fs::copy(src.join(f), tmp.join(f)).unwrap();
    }
    let mut m = std::fs::read_to_string(src.join("ks.mod")).unwrap();
    m += "\nheterogeneity_solve(truncation_horizon = 40);\nheterogeneity_simulate(irf = 30, periods = 25, tex, graph_format = eps);\n";
    std::fs::write(tmp.join("ks.mod"), m).unwrap();
    let mut mf = parse_mod_file(&tmp.join("ks.mod"), &[]).unwrap();
    let mut st = HetSession::default();
    for c in mf.commands.clone() {
        if let Command::Other(n, o) = &c
            && handles(n)
        {
            run_command(&mut mf, n, o, &tmp, "ks", &mut st).unwrap();
        }
    }
    for f in ["ks_IRF.tex", "ks_simulation.tex"] {
        let drop_date =
            |s: String| s.lines().enumerate().filter(|(i, _)| *i != 1).map(|(_, l)| l.to_string()).collect::<Vec<_>>();
        let got = drop_date(std::fs::read_to_string(tmp.join("ks/graphs").join(f)).unwrap());
        let want = drop_date(std::fs::read_to_string(src.join("tex").join(f)).unwrap());
        assert_eq!(got, want, "{f}");
    }
}
