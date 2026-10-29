// SPDX-License-Identifier: GPL-3.0-or-later
//! Live progress of a run, for the `dynare-progress` viewer (`dynare-rs --window`
//! opens one in a small Terminal window, like Dynare's waitbars).
//!
//! The driver names the phase (optimizer, Hessian, jscale tuning, MCMC, smoother),
//! [`DsgePosterior`](crate::dsge::DsgePosterior) counts posterior evaluations and
//! keeps the best value of the phase, and the MCMC reports each chain's draws.
//! A background thread writes the state as JSON to `<fname>/Output/<fname>_progress.json`
//! twice a second (write + rename, so the viewer never reads half a file).
//! Nothing here touches the numbers: it only observes.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::json;

#[derive(Clone, Copy, Default)]
struct ChainState {
    done: usize,
    total: usize,
    acceptance: f64,
}

struct State {
    path: PathBuf,
    model: String,
    started: Instant,
    phase: String,
    phase_started: Instant,
    detail: String,
    chains: Vec<ChainState>,
    status: &'static str,
    message: String,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static ACTIVE: AtomicBool = AtomicBool::new(false);
static EVALS: AtomicU64 = AtomicU64::new(0);
/// Incremented by each [`start`]: a writer thread exits once a newer run has started.
static GENERATION: AtomicU64 = AtomicU64::new(0);
/// Best (lowest) minus log posterior of the phase, as f64 bits.
static BEST: AtomicU64 = AtomicU64::new(f64::INFINITY.to_bits());

/// Path of the progress file of model `fname` run from `dir`.
pub fn file_path(dir: &Path, fname: &str) -> PathBuf {
    dir.join(fname).join("Output").join(format!("{fname}_progress.json"))
}

/// Starts writing the progress of model `fname` to `path`.
pub fn start(path: PathBuf, fname: &str) {
    let now = Instant::now();
    *STATE.lock().unwrap() = Some(State {
        path,
        model: fname.to_string(),
        started: now,
        phase: "preprocessing".into(),
        phase_started: now,
        detail: String::new(),
        chains: Vec::new(),
        status: "running",
        message: String::new(),
    });
    let generation = GENERATION.fetch_add(1, Ordering::SeqCst) + 1;
    ACTIVE.store(true, Ordering::SeqCst);
    write_now();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(500));
            if !ACTIVE.load(Ordering::SeqCst) || GENERATION.load(Ordering::SeqCst) != generation {
                break;
            }
            write_now();
        }
    });
}

fn with_state(f: impl FnOnce(&mut State)) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        f(s);
    }
}

/// dynare-rust's own information lines (what was written where, timings, MCMC progress).
/// Dynare's console output has none of them, so stdout stays identical to Dynare's; on an
/// interactive terminal (or with `DYNARE_RS_VERBOSE` set, e.g. for the log mode of
/// `dynare-progress`, which reads `chains x` and `chain 1:` lines) they go to stderr, the
/// role of Dynare's waitbars.
pub fn note(text: impl AsRef<str>) {
    use std::io::IsTerminal;
    if std::io::stderr().is_terminal() || std::env::var_os("DYNARE_RS_VERBOSE").is_some() {
        eprintln!("{}", text.as_ref());
    }
}

/// Starts a new phase: resets the detail line, the chains and the evaluation counter.
pub fn phase(name: impl Into<String>) {
    let name = name.into();
    with_state(|s| {
        s.phase = name;
        s.phase_started = Instant::now();
        s.detail.clear();
        s.chains.clear();
    });
    EVALS.store(0, Ordering::Relaxed);
    BEST.store(f64::INFINITY.to_bits(), Ordering::Relaxed);
}

/// Free-text line under the phase (e.g. the optimizer's round).
pub fn detail(text: impl Into<String>) {
    let text = text.into();
    with_state(|s| s.detail = text);
}

/// One posterior evaluation with value `fval` (minus log posterior; +inf = failure).
pub fn eval(fval: f64) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    EVALS.fetch_add(1, Ordering::Relaxed);
    if fval.is_finite() {
        let mut cur = BEST.load(Ordering::Relaxed);
        while fval < f64::from_bits(cur) {
            match BEST.compare_exchange_weak(cur, fval.to_bits(), Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(now) => cur = now,
            }
        }
    }
}

/// `n` chains of `total` draws each are about to start.
pub fn chains(n: usize, total: usize) {
    with_state(|s| s.chains = vec![ChainState { done: 0, total, acceptance: 0.0 }; n]);
}

/// Chain `i` has done `done` draws with acceptance ratio `acceptance`.
pub fn chain(i: usize, done: usize, acceptance: f64) {
    with_state(|s| {
        if let Some(c) = s.chains.get_mut(i) {
            c.done = done;
            c.acceptance = acceptance;
        }
    });
}

/// The run ended (`ok` false: with the error `message`). Writes the final state.
pub fn finish(ok: bool, message: &str) {
    with_state(|s| {
        s.status = if ok { "done" } else { "error" };
        s.message = message.to_string();
    });
    write_now();
    ACTIVE.store(false, Ordering::SeqCst);
    // No write after the run: a lingering writer thread finds no state (it holds the
    // lock while writing, so none is in flight once this returns).
    *STATE.lock().unwrap() = None;
}

fn write_now() {
    let guard = STATE.lock().unwrap();
    let Some(s) = guard.as_ref() else { return };
    let best = f64::from_bits(BEST.load(Ordering::Relaxed));
    let unix = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    let v = json!({
        "model": s.model,
        "pid": std::process::id(),
        "updated": unix,
        "status": s.status,
        "message": s.message,
        "elapsed": s.started.elapsed().as_secs_f64(),
        "phase": s.phase,
        "phase_elapsed": s.phase_started.elapsed().as_secs_f64(),
        "detail": s.detail,
        "evals": EVALS.load(Ordering::Relaxed),
        "best": best.is_finite().then_some(best),
        "chains": s.chains.iter().map(|c| json!({"done": c.done, "total": c.total, "acceptance": c.acceptance})).collect::<Vec<_>>(),
    });
    // Keep the lock while writing, so `finish` returning means no write is in flight.
    let path = s.path.clone();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, v.to_string()).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}
