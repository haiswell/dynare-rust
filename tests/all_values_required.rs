//! `initval`/`endval`/`histval` option `all_values_required`: error texts are those of the
//! Dynare 7.2 preprocessor (run on the same mod files; the 7.2 histval check names "endval"
//! in its exogenous message).
use dynare::modfile::parse_mod_file;

const HEAD: &str = "var y c; varexo e u; parameters a; a=0.5;\nmodel; y = a*y(-1) + e; c = a*c(-1)+u; end;\n";

fn err(tail: &str) -> String {
    let d = std::env::temp_dir().join(format!("avr_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(format!("t{}.mod", tail.len()));
    std::fs::write(&p, format!("{HEAD}{tail}")).unwrap();
    match parse_mod_file(&p, &[]) {
        Ok(_) => String::new(),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn all_values_required() {
    let e = err("initval(all_values_required); y=1; e=0; u=0; end;\n");
    assert!(e.contains("You have not set the following endogenous variables in initval: c"), "{e}");
    let e = err("initval(all_values_required); y=1; c=1; e=0; end;\n");
    assert!(e.contains("You have not set the following exogenous variables in initval: u"), "{e}");
    let e = err("initval(all_values_required); y=1; end;\n");
    assert!(
        e.contains("endogenous variables in initval: c") && e.contains("exogenous variables in initval: e u"),
        "{e}"
    );
    let e = err("initval; y=1; c=1; e=0; u=0; end;\nendval(all_values_required); y=1; e=0; u=0; end;\n");
    assert!(e.contains("endogenous variables in endval: c"), "{e}");
    let e = err("histval(all_values_required); y(0)=1; e(0)=0; u(0)=0; end;\n");
    assert!(e.contains("endogenous variables in histval: c"), "{e}");
    let e = err("histval(all_values_required); y(0)=1; c(0)=1; e(0)=0; end;\n");
    assert!(e.contains("exogenous variables in endval: u"), "{e}");
    // any lag counts; complete blocks pass
    assert_eq!(err("histval(all_values_required); y(0)=1; c(-1)=1; e(0)=0; u(0)=0; end;\n"), "");
    assert_eq!(err("initval(all_values_required); y=1; c=1; e=0; u=0; end;\n"), "");
    // without the option nothing is checked
    assert_eq!(err("initval; y=1; end;\n"), "");
}
