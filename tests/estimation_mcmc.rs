//! RW Metropolis–Hastings, jscale tuning and posterior statistics on Gaussian
//! targets with known moments and normalizing constant.

use std::time::Instant;

use dynare::estimation::{
    Bounds, MhOptions, Objective, PosteriorOptions, TuneOptions, brooks_gelman, calibrate_mh_scale_parameter, hessian,
    laplace_approximation, posterior_stats, rw_metropolis,
};
use dynare::linalg::{Mat, chol_upper};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// f(x) = 0.5 (x-m)' P (x-m) + c, the minus log kernel of N(m, inv(P)).
struct Gauss {
    m: Vec<f64>,
    p: Mat,
    c: f64,
}

impl Objective for Gauss {
    fn dim(&self) -> usize {
        self.m.len()
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        let n = self.m.len();
        let d: Vec<f64> = (0..n).map(|i| x[i] - self.m[i]).collect();
        let mut s = 0.0;
        for j in 0..n {
            let col = &self.p.as_slice()[j * n..(j + 1) * n];
            let mut t = 0.0;
            for i in 0..n {
                t += col[i] * d[i];
            }
            s += d[j] * t;
        }
        (0.5 * s + self.c, 0)
    }
}

impl Gauss {
    /// log of the integral of exp(-f) = -c + n/2 log(2 pi) - 1/2 log det P
    fn log_marginal(&self) -> f64 {
        let n = self.m.len() as f64;
        let u = chol_upper(&self.p).unwrap();
        let logdet: f64 = 2.0 * (0..self.m.len()).map(|i| u[(i, i)].ln()).sum::<f64>();
        -self.c + 0.5 * n * (2.0 * std::f64::consts::PI).ln() - 0.5 * logdet
    }
}

/// A random covariance with heterogeneous scales and correlations.
fn target(n: usize, seed: u64) -> (Gauss, Mat) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let a = Mat::from_fn(n, n, |_, _| rng.r#gen::<f64>() - 0.5);
    let mut sigma = &a * a.transpose() / n as f64 + Mat::identity(n, n) * 0.05;
    let scales: Vec<f64> = (0..n).map(|_| 10f64.powf(2.0 * rng.r#gen::<f64>() - 1.0)).collect();
    for j in 0..n {
        for i in 0..n {
            sigma[(i, j)] *= scales[i] * scales[j];
        }
    }
    let p = sigma.clone().try_inverse().unwrap();
    let m: Vec<f64> = (0..n).map(|i| scales[i] * (rng.r#gen::<f64>() - 0.5)).collect();
    (Gauss { m, p, c: 123.4 }, sigma)
}

#[test]
fn rwmh_recovers_73_dim_gaussian() {
    let n = 73;
    let (obj, sigma) = target(n, 7);
    let mut opts = MhOptions::new(n);
    opts.replic = 200_000;
    opts.nblocks = 6;
    opts.seed = 42;
    opts.progress_every = 0;
    let bounds = Bounds::unbounded(n);
    let t = Instant::now();
    let res = rw_metropolis(&obj, &obj.m, &sigma, &bounds, &opts, None).unwrap();
    eprintln!("{} draws in {:.2?}", opts.replic * opts.nblocks, t.elapsed());
    let stats = posterior_stats(&res.chains, &PosteriorOptions::default()).unwrap();
    eprintln!("acceptance {:?}", stats.acceptance);
    for a in &stats.acceptance {
        assert!(*a > 0.15 && *a < 0.35);
    }
    let mut worst_mean: f64 = 0.0;
    let mut worst_var: f64 = 0.0;
    let mut worst_corr: f64 = 0.0;
    for i in 0..n {
        let sd = sigma[(i, i)].sqrt();
        worst_mean = worst_mean.max((stats.mean[i] - obj.m[i]).abs() / sd);
        worst_var = worst_var.max((stats.covariance[(i, i)] / sigma[(i, i)] - 1.0).abs());
        for j in 0..i {
            let c_hat = stats.covariance[(i, j)] / (stats.covariance[(i, i)] * stats.covariance[(j, j)]).sqrt();
            let c = sigma[(i, j)] / (sigma[(i, i)] * sigma[(j, j)]).sqrt();
            worst_corr = worst_corr.max((c_hat - c).abs());
        }
        // pooled moments agree with the covariance-matrix pass
        assert!((stats.params[i].mean - stats.mean[i]).abs() < 1e-9 * (1.0 + stats.mean[i].abs()));
        let hpd = stats.params[i].hpd;
        // 90% HPD of a normal: mean +- 1.645 sd
        assert!(((hpd.1 - hpd.0) / (2.0 * 1.6449 * sd) - 1.0).abs() < 0.15, "hpd {i}: {hpd:?}");
    }
    eprintln!(
        "max |mean error|/sd = {worst_mean:.4}, max rel var error = {worst_var:.4}, max corr error = {worst_corr:.4}"
    );
    eprintln!(
        "inefficiency factors: min {:.1}, max {:.1}",
        stats.ifac.iter().cloned().fold(f64::INFINITY, f64::min),
        stats.ifac.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    assert!(worst_mean < 0.15);
    assert!(worst_var < 0.2);
    assert!(worst_corr < 0.1);
    // modified harmonic mean vs the true normalizing constant
    let truth = obj.log_marginal();
    eprintln!("log marginal: true {truth:.4}, MHM {:.4} (increase {})", stats.mhm.log_marginal, stats.mhm.increase);
    assert!((stats.mhm.log_marginal - truth).abs() < 0.5);
    // Laplace with the numerical Hessian at the mode is exact for a Gaussian
    let hh = hessian(&obj, &obj.m, [1e-2, 1.0]);
    let lap = laplace_approximation(&hh, obj.eval(&obj.m).0);
    eprintln!("Laplace {lap:.6}");
    assert!((lap - truth).abs() < 1e-3);
}

#[test]
fn mhm_on_small_gaussian_is_accurate() {
    let n = 5;
    let (obj, sigma) = target(n, 3);
    let mut opts = MhOptions::new(n);
    opts.replic = 100_000;
    opts.nblocks = 4;
    opts.seed = 1;
    let res = rw_metropolis(&obj, &obj.m, &sigma, &Bounds::unbounded(n), &opts, None).unwrap();
    let stats = posterior_stats(&res.chains, &PosteriorOptions::default()).unwrap();
    let truth = obj.log_marginal();
    eprintln!("n=5 log marginal: true {truth:.5}, MHM {:.5}", stats.mhm.log_marginal);
    assert!((stats.mhm.log_marginal - truth).abs() < 0.02);
    assert!((laplace_approximation(&obj.p, obj.c) - truth).abs() < 1e-10);
    let bg = brooks_gelman(&res.chains, 0.5, false).unwrap();
    // pooled and within-chain interval lengths agree for converged chains
    let last = bg.udiag[0].last().unwrap();
    assert!((last[0] / last[1] - 1.0).abs() < 0.05, "{last:?}");
    assert!((last[2] / last[3] - 1.0).abs() < 0.05, "{last:?}");
}

#[test]
fn tuned_jscale_hits_target_acceptance() {
    let n = 20;
    let (obj, sigma) = target(n, 11);
    let bounds = Bounds::unbounded(n);
    let tune = TuneOptions { seed: 5, ..Default::default() };
    let scale = calibrate_mh_scale_parameter(&obj, &sigma, &obj.m, &bounds, &tune).unwrap();
    let mut opts = MhOptions::new(n);
    opts.jscale = vec![scale; n];
    opts.mh_jscale = scale;
    opts.replic = 50_000;
    opts.nblocks = 2;
    let res = rw_metropolis(&obj, &obj.m, &sigma, &bounds, &opts, None).unwrap();
    let acc: Vec<f64> = res.chains.iter().map(|c| c.acceptance_ratio()).collect();
    eprintln!("tuned scale {scale:.4}, acceptance {acc:?}");
    for a in acc {
        assert!((a - 0.33).abs() < 0.03);
    }
}

#[test]
fn chains_are_reproducible_and_respect_bounds() {
    let n = 4;
    let (obj, sigma) = target(n, 2);
    // truncate the first parameter at its mean: half the mass is outside
    let mut bounds = Bounds::unbounded(n);
    bounds.lb[0] = obj.m[0];
    let mut opts = MhOptions::new(n);
    opts.replic = 5000;
    opts.nblocks = 3;
    opts.seed = 9;
    opts.store_from = 1000;
    let start: Vec<f64> =
        obj.m.iter().enumerate().map(|(i, v)| if i == 0 { v + sigma[(0, 0)].sqrt() } else { *v }).collect();
    let a = rw_metropolis(&obj, &start, &sigma, &bounds, &opts, None).unwrap();
    let b = rw_metropolis(&obj, &start, &sigma, &bounds, &opts, None).unwrap();
    for (ca, cb) in a.chains.iter().zip(&b.chains) {
        assert_eq!(ca.draws, cb.draws);
        assert_eq!(ca.logpost, cb.logpost);
        assert_eq!(ca.len(), 4000);
        for t in ca.first_stored..ca.total_draws {
            assert!(ca.draw(t)[0] > bounds.lb[0]);
        }
    }
    assert_ne!(a.chains[0].draws, a.chains[1].draws);
    // initial values are drawn around the start and inside the bounds
    for c in &a.chains {
        assert!(c.init[0] >= bounds.lb[0] && c.init_logpost.is_finite());
    }
    // posterior stats with only the kept draws stored
    let stats = posterior_stats(&a.chains, &PosteriorOptions { mh_drop: 0.2, ..Default::default() }).unwrap();
    assert_eq!(stats.kept_per_chain, 4000);
    assert!(posterior_stats(&a.chains, &PosteriorOptions { mh_drop: 0.1, ..Default::default() }).is_err());
}

#[test]
fn bookkeeping_overhead_per_draw() {
    // A trivial objective: measures the sampler's own cost per draw (73 parameters).
    struct Cheap(usize);
    impl Objective for Cheap {
        fn dim(&self) -> usize {
            self.0
        }
        fn eval(&self, x: &[f64]) -> (f64, i32) {
            (0.5 * x.iter().map(|v| v * v).sum::<f64>(), 0)
        }
    }
    let n = 73;
    let obj = Cheap(n);
    let mut opts = MhOptions::new(n);
    opts.replic = 200_000;
    opts.nblocks = 1;
    let t = Instant::now();
    let res = rw_metropolis(&obj, &vec![0.0; n], &Mat::identity(n, n), &Bounds::unbounded(n), &opts, None).unwrap();
    let per = t.elapsed().as_secs_f64() / opts.replic as f64;
    eprintln!("sampler cost per draw (73 params, incl. a trivial objective): {:.2} us", per * 1e6);
    assert_eq!(res.chains[0].len(), opts.replic);
    // should be negligible against a ~1 ms likelihood (debug builds are slower)
    assert!(per < 2e-4);
}

#[test]
fn progress_callback_is_called() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let n = 3;
    let (obj, sigma) = target(n, 4);
    let mut opts = MhOptions::new(n);
    opts.replic = 1000;
    opts.nblocks = 2;
    opts.progress_every = 100;
    let calls = AtomicUsize::new(0);
    let cb = |p: &dynare::estimation::MhProgress| {
        assert!(p.acceptance >= 0.0 && p.acceptance <= 1.0 && p.draws_done.is_multiple_of(100));
        calls.fetch_add(1, Ordering::Relaxed);
    };
    rw_metropolis(&obj, &obj.m, &sigma, &Bounds::unbounded(n), &opts, Some(&cb)).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 20);
}

/// A Gaussian kernel inside a box, with an infeasible region (x0 + x1 > 1.5:
/// +inf, like a failed solution) so that out-of-bounds proposals, failed
/// evaluations, rejections and acceptances all occur.
struct Holed;
impl Objective for Holed {
    fn dim(&self) -> usize {
        5
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        if x[0] + x[1] > 1.5 {
            return (f64::INFINITY, 1);
        }
        (0.5 * x.iter().enumerate().map(|(i, v)| v * v * (1.0 + i as f64)).sum::<f64>(), 0)
    }
}

fn holed_bounds() -> Bounds {
    Bounds { lb: vec![-1.0, -2.0, -3.0, -1.5, -10.0], ub: vec![2.0, 1.0, 3.0, 1.5, 10.0] }
}

/// `rand_multivariate_normal`: mean + randn(1, n) * U (U upper triangular).
fn ref_proposal(rng: &mut ChaCha8Rng, mean: &[f64], u: &Mat) -> Vec<f64> {
    let n = mean.len();
    let z: Vec<f64> = (0..n).map(|_| rng.sample(rand_distr::StandardNormal)).collect();
    (0..n).map(|j| mean[j] + (0..=j).map(|i| z[i] * u[(i, j)]).sum::<f64>()).collect()
}

fn ref_stream(seed: u64, stream: u64) -> ChaCha8Rng {
    let mut r = ChaCha8Rng::seed_from_u64(seed);
    r.set_stream(stream);
    r
}

/// Pre-fetching (several evaluations per round) must give exactly the plain
/// sequential chain of posterior_sampler_iteration.m, here written out directly.
#[test]
fn prefetching_reproduces_the_sequential_chain() {
    let (obj, b) = (Holed, holed_bounds());
    let cov = Mat::identity(5, 5) * 0.3;
    let x0 = vec![0.1; 5];
    let mut opts = MhOptions::new(5);
    opts.replic = 4000;
    opts.nblocks = 1;
    opts.seed = 9;
    opts.progress_every = 0;
    // reference: one chain from the mode, stream 1
    let u = chol_upper(&cov).unwrap() * opts.jscale[0];
    let mut rng = ref_stream(opts.seed, 1);
    let (mut last, mut last_lp) = (x0.clone(), -obj.eval(&x0).0);
    let (mut draws, mut accepted, mut failed) = (Vec::new(), 0, 0);
    for _ in 0..opts.replic {
        let par = ref_proposal(&mut rng, &last, &u);
        let inside = (0..5).all(|i| par[i] > b.lb[i] && par[i] < b.ub[i]);
        let lp = if inside { -obj.eval(&par).0 } else { f64::NEG_INFINITY };
        failed += (inside && !lp.is_finite()) as usize;
        if lp > f64::NEG_INFINITY && rng.r#gen::<f64>().ln() < lp - last_lp {
            accepted += 1;
            last = par;
            last_lp = lp;
        }
        draws.extend_from_slice(&last);
    }
    assert!(failed > 0 && accepted > 100 && accepted < 3900, "{failed} {accepted}");
    for pf in [1, 2, 3, 6, 11] {
        opts.prefetch = pf;
        let r = rw_metropolis(&obj, &x0, &cov, &b, &opts, None).unwrap();
        assert_eq!(r.prefetch, pf);
        assert_eq!(r.chains[0].accepted, accepted, "prefetch {pf}");
        assert!(r.chains[0].draws.iter().zip(&draws).all(|(a, b)| a.to_bits() == b.to_bits()), "prefetch {pf}");
    }
    // several chains: the same draws whatever the width
    opts.nblocks = 3;
    opts.store_from = 1000;
    opts.prefetch = 1;
    let seq = rw_metropolis(&obj, &x0, &cov, &b, &opts, None).unwrap();
    for pf in [2, 4] {
        opts.prefetch = pf;
        let par = rw_metropolis(&obj, &x0, &cov, &b, &opts, None).unwrap();
        for (a, c) in seq.chains.iter().zip(&par.chains) {
            assert_eq!(a.accepted, c.accepted);
            assert!(a.draws.iter().zip(&c.draws).all(|(x, y)| x.to_bits() == y.to_bits()));
            assert!(a.logpost.iter().zip(&c.logpost).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
    }
}

/// Same for calibrate_mh_scale_parameter.m, written out directly.
#[test]
fn prefetching_reproduces_the_sequential_jscale_tuning() {
    let (obj, b) = (Holed, holed_bounds());
    let cov = Mat::identity(5, 5) * 0.3;
    let x0 = vec![0.1; 5];
    let opts = TuneOptions { stepsize: 200, seed: 3, ..Default::default() };
    let dd = chol_upper(&cov).unwrap();
    let mut rng = ref_stream(opts.seed, u64::MAX);
    let mut parameters = x0.clone();
    let mut logpo1 = -obj.eval(&x0).0;
    let (mut correction, mut scale) = (1.0f64, opts.guess.unwrap_or(2.38 / 5f64.sqrt()));
    let (mut j, mut jj, mut isux, mut jsux, mut i) = (1usize, 1usize, 0usize, 0usize, 0usize);
    while j <= opts.maxiter {
        let proposal = ref_proposal(&mut rng, &parameters, &(&dd * scale));
        let inside = (0..5).all(|k| proposal[k] > b.lb[k] && proposal[k] < b.ub[k]);
        let logpo0 = if inside { -obj.eval(&proposal).0 } else { f64::NEG_INFINITY };
        if logpo0 > f64::NEG_INFINITY && rng.r#gen::<f64>().ln() < logpo0 - logpo1 {
            parameters = proposal;
            logpo1 = logpo0;
            isux += 1;
            jsux += 1;
        }
        if j % opts.stepsize == 0 {
            let r1 = jsux as f64 / jj as f64;
            let r2 = isux as f64 / j as f64;
            let c1 = r1 / opts.target;
            correction = if (c1 - 1.0).abs() > 0.05 { correction.powf(opts.rho) * c1.powf(1.0 - opts.rho) } else { c1 };
            scale = if c1 > 0.0 { scale * correction } else { scale / 10.0 };
            jsux = 0;
            jj = 0;
            i = if (r2 - opts.target).abs() < opts.c2 && (r1 - opts.target).abs() < opts.c1 { i + 1 } else { 0 };
            if i > opts.c3 {
                break;
            }
        }
        j += 1;
        jj += 1;
    }
    assert!(j < opts.maxiter, "the reference tuning should converge");
    for pf in [1, 2, 5, 9] {
        let got =
            calibrate_mh_scale_parameter(&obj, &cov, &x0, &b, &TuneOptions { prefetch: pf, ..opts.clone() }).unwrap();
        assert_eq!(got.to_bits(), scale.to_bits(), "prefetch {pf}: {got} vs {scale}");
    }
}
