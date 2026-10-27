// SPDX-License-Identifier: GPL-3.0-or-later
//! Figures of the identification analysis, built as [`graphics::Figure`]s: port of the figures
//! of `+identification/plot.m` (identification strength and sensitivity bars, collinearity
//! patterns, identification patterns from the SVD of the information matrix, MC sensitivities).
//!
//! Differences to Dynare: grouped bars are two bar series shifted by -0.15/+0.15; the
//! parameter names of the collinearity heat maps are not rotated.

use anyhow::Result;

use crate::graphics::{self, Figure, GraphOptions, Panel, Series};
use crate::linalg::Mat;

use super::Analysis;
use super::la::svd_econ;

fn ticks(names: &[String], order: &[usize]) -> Vec<(f64, String)> {
    order.iter().enumerate().map(|(k, &i)| ((k + 1) as f64, names[i].clone())).collect()
}

fn bars(x: Vec<f64>, y: Vec<f64>, label: &str) -> Series {
    Series::StackedBars { x, components: vec![y], labels: vec![label.to_string()] }
}

/// MATLAB `[~, is] = sort(v)` (ascending, NaN last, stable).
fn sort_index(v: &[f64]) -> Vec<usize> {
    let mut is: Vec<usize> = (0..v.len()).collect();
    is.sort_by(|&a, &b| {
        let (x, y) = (v[a], v[b]);
        match (x.is_nan(), y.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => x.partial_cmp(&y).unwrap(),
        }
    });
    is
}

fn ln(v: f64) -> f64 {
    v.ln()
}

/// Two grouped bar series (value, prior-normalized) in `order`, log scale.
fn grouped(a: &[f64], b: &[f64], order: &[usize], with_prior: bool) -> Vec<Series> {
    let x1: Vec<f64> = (1..=order.len()).map(|k| k as f64 - if with_prior { 0.15 } else { 0.0 }).collect();
    let mut s = vec![bars(x1, order.iter().map(|&i| ln(a[i])).collect(), "relative to param value")];
    if with_prior {
        let x2: Vec<f64> = (1..=order.len()).map(|k| k as f64 + 0.15).collect();
        s.push(bars(x2, order.iter().map(|&i| ln(b[i])).collect(), "relative to prior std"));
    }
    s
}

fn tittxt1(t: &str) -> String {
    t.replace(' ', "_").replace('.', "")
}

/// Figures of a point analysis (`SampleSize == 1`).
pub fn point_figures(an: &Analysis, names: &[String], tittxt: &str, advanced: bool) -> Vec<Figure> {
    let mut figs = Vec::new();
    let Some(st) = &an.strength else {
        return figs;
    };
    let is = sort_index(&st.ide_strength);
    let with_prior = !st.ide_strength_prior.iter().all(|v| v.is_nan());
    let kind = if st.flag_score { "asymptotic Information matrix" } else { "moments Information matrix" };
    let p1 = Panel {
        title: format!("Identification strength with {kind} (log-scale)"),
        series: grouped(&st.ide_strength, &st.ide_strength_prior, &is, with_prior),
        x_ticks: Some(ticks(names, &is)),
        legend: true,
        ..Default::default()
    };
    let with_prior_d = !st.delta_m_prior.iter().all(|v| v.is_nan());
    let p2 = Panel {
        title: format!("Sensitivity component with {kind} (log-scale)"),
        series: grouped(&st.delta_m, &st.delta_m_prior, &is, with_prior_d),
        x_ticks: Some(ticks(names, &is)),
        legend: true,
        ..Default::default()
    };
    let t1 = tittxt1(tittxt);
    figs.push(Figure {
        name: format!("ident_strength_{t1}"),
        title: format!("{tittxt} - Identification using info from observables"),
        rows: 2,
        cols: 1,
        panels: vec![p1, p2],
    });
    if !advanced {
        return figs;
    }
    // sensitivity bars
    let np = names.len();
    let norm = |v: &[f64]| -> Vec<f64> {
        let m = v.iter().copied().fold(f64::NAN, f64::max);
        v.iter().map(|x| x / m).collect()
    };
    let mom = norm(&st.si_dmoments_norm);
    let tau = norm(&st.si_dreducedform_norm);
    let lre = norm(&st.si_ddynamic_norm);
    if !(mom.iter().chain(&tau).chain(&lre).all(|v| v.is_nan())) {
        let offset = np.saturating_sub(lre.len());
        let lre_full: Vec<f64> = (0..np).map(|i| if i < offset { f64::NAN } else { lre[i - offset] }).collect();
        let col = |v: &[f64], dx: f64, label: &str| {
            let x: Vec<f64> = (1..=np).map(|k| k as f64 + dx).collect();
            bars(x, is.iter().map(|&i| (v.get(i).copied().unwrap_or(f64::NAN) * 100.0).ln()).collect(), label)
        };
        figs.push(Figure {
            name: format!("sensitivity_{t1}"),
            title: format!("{tittxt} - Sensitivity plot"),
            rows: 1,
            cols: 1,
            panels: vec![Panel {
                title: "Sensitivity bars using derivatives (log-scale)".into(),
                series: vec![col(&mom, -0.25, "Moments"), col(&tau, 0.0, "Model"), col(&lre_full, 0.25, "LRE model")],
                x_ticks: Some(ticks(names, &is)),
                legend: true,
                ..Default::default()
            }],
        });
    }
    if let Some((pars, cosn)) = &an.bruteforce {
        // [stream X] imagesc(pax, [0 1]): pax(i, k) = cosn(i, j) for the explanatory parameters k
        for j in 0..cosn.ncols() {
            let mut pax = vec![vec![f64::NAN; np]; np];
            for i in 0..np {
                for k in pars[i][j].iter().flatten() {
                    pax[i][*k] = cosn[(i, j)];
                }
            }
            figs.push(Figure {
                name: format!("ident_collinearity_{t1}_{}", j + 1),
                title: format!("{tittxt} - Collinearity patterns with {} parameter(s)", j + 1),
                rows: 1,
                cols: 1,
                panels: vec![Panel {
                    series: vec![Series::HeatMap { z: pax, vmin: 0.0, vmax: 1.0, labels: names.to_vec() }],
                    x_label: Some(format!("{tittxt} - Collinearity patterns with {} parameter(s)", j + 1)),
                    ..Default::default()
                }],
            });
        }
    }
    // identification patterns: singular vectors of the information matrix
    let (_, s, v) = svd_econ(&st.ahess);
    let kind2 = if st.flag_score { "Information matrix" } else { "moments Information matrix" };
    let order: Vec<usize> = (0..np).collect();
    let pattern = |vcol: usize, sv: f64| Panel {
        title: format!("Singular value {}", super::display::num2str(sv)),
        series: vec![bars((1..=np).map(|k| k as f64).collect(), (0..np).map(|i| v[(i, vcol)].abs()).collect(), "|V|")],
        x_ticks: Some(ticks(names, &order)),
        ..Default::default()
    };
    let nsv = s.len();
    let mut f1 = Vec::new();
    let mut f2 = Vec::new();
    for j in 0..np.min(8) {
        if j < 4 {
            f1.push(pattern(nsv - 1 - j, s[nsv - 1 - j]));
        } else {
            f2.push(pattern(j - 4, s[j - 4]));
        }
    }
    let suffix = if np < 5 { String::new() } else { ": SMALLEST SV".into() };
    let n1 = f1.len();
    figs.push(Figure {
        name: format!("ident_pattern_{t1}_1"),
        title: format!("{tittxt} - Identification patterns ({kind2}){suffix}"),
        rows: n1.max(1),
        cols: 1,
        panels: f1,
    });
    if np > 4 {
        let n2 = f2.len();
        figs.push(Figure {
            name: format!("ident_pattern_{t1}_2"),
            title: format!("{tittxt} - Identification patterns ({kind2}): HIGHEST SV"),
            rows: n2.max(1),
            cols: 1,
            panels: f2,
        });
    }
    figs
}

/// `MC_sensitivity`: MC mean of the normalized sensitivities (sorted by the point strength).
pub fn mc_figures(si_dmoments_norm: &Mat, point_strength: Option<&[f64]>, names: &[String]) -> Vec<Figure> {
    let np = names.len();
    let mean: Vec<f64> = (0..np)
        .map(|j| {
            let col: Vec<f64> = si_dmoments_norm.column(j).iter().copied().filter(|v| !v.is_nan()).collect();
            col.iter().sum::<f64>() / col.len().max(1) as f64
        })
        .collect();
    let mx = mean.iter().copied().fold(f64::NAN, f64::max);
    let is = match point_strength {
        Some(s) => sort_index(s),
        None => (0..np).collect(),
    };
    vec![Figure {
        name: "MC_sensitivity".into(),
        title: "MC sensitivities".into(),
        rows: 1,
        cols: 1,
        panels: vec![Panel {
            title: "MC mean of sensitivity measures".into(),
            series: vec![bars(
                (1..=np).map(|k| k as f64).collect(),
                is.iter().map(|&i| mean[i] / mx).collect(),
                "Moments",
            )],
            x_ticks: Some(ticks(names, &is)),
            ..Default::default()
        }],
    }]
}

/// The TeX eps-loader files of `identification.plot.m` (`options_.TeX` with the `eps` graph
/// format): `<fname>_ident_strength_<t1>.tex`, `<fname>_sensitivity_<t1>.tex` and
/// `<fname>_ident_collinearity_<t1>_<j>.tex`, one per saved figure of that kind.
pub fn write_eps_loaders(figs: &[Figure], fname: &str, go: &GraphOptions, tittxt: &str, tit_tex: &str) -> Result<()> {
    if !go.enabled || !go.formats.contains(&crate::graphics::GraphFormat::Eps) {
        return Ok(());
    }
    let t1 = tittxt1(tittxt);
    let label = tittxt.trim_end();
    let dirname = format!("{fname}/identification");
    for f in figs {
        let (caption, lab) = if f.name == format!("ident_strength_{t1}") {
            (format!("{tit_tex}  - Identification using info from observables."), format!("Fig:ident:{label}"))
        } else if f.name == format!("sensitivity_{t1}") {
            (format!("{tit_tex}  - Sensitivity plot."), format!("Fig:sensitivity:{label}"))
        } else if let Some(j) = f.name.strip_prefix(&format!("ident_collinearity_{t1}_")) {
            (
                format!("{tit_tex}  - Collinearity patterns with {j} parameter(s)."),
                format!("Fig:collinearity:{label}:{j}_pars"),
            )
        } else {
            continue;
        };
        let mut s = String::new();
        s += "% TeX eps-loader file generated by identification.plot.m (Dynare).\n";
        s += &format!("% {}\n\n", crate::latex::datestr_now());
        s += "\\begin{figure}[H]\n\\centering \n";
        s += &format!("\\includegraphics[width=0.8\\textwidth]{{{dirname}/{fname}_{}}}\n", f.name);
        s += &format!("\\caption{{{caption}}}");
        s += &format!("\\label{{{lab}}}\n");
        s += "\\end{figure}\n\n% End Of TeX file. \n";
        std::fs::write(go.dir.join(format!("{fname}_{}.tex", f.name)), s)?;
    }
    Ok(())
}

pub fn save_all(figs: &[Figure], fname: &str, go: &GraphOptions) -> Result<()> {
    for f in figs {
        graphics::save(f, fname, go)?;
    }
    Ok(())
}
