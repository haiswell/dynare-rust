// SPDX-License-Identifier: GPL-3.0-or-later
//! `dynare-progress [--window] <path>...`: live progress of dynare-rs runs, one
//! panel per path, refreshed every second (Dynare's waitbars, in a terminal).
//!
//! A path is a progress file (`<fname>/Output/<fname>_progress.json`, written by
//! every dynare-rs run), a dynare-rs log (its console output redirected to a file, with
//! `DYNARE_RS_VERBOSE=1` so that the progress notes, which are otherwise only written to a
//! terminal, are in it: only chain 1 and the last optimizer line are shown), or a directory, which shows
//! the most recently updated of those found in it and one level below.
//! `--window` opens a small Terminal window (macOS) or a new console window (Windows)
//! per path running this viewer.
//! `dynare-rs --window` does that for its own run.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: dynare-progress [--window] <progress.json | run.log | dir>...");
        std::process::exit(2);
    }
    if args[0] == "--window" {
        for (k, p) in args[1..].iter().enumerate() {
            if let Err(e) = open_window(Path::new(p), k) {
                eprintln!("ERROR: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    let paths: Vec<PathBuf> = args.iter().map(PathBuf::from).collect();
    enable_ansi();
    if paths.len() == 1 {
        print!("\x1b]0;dynare-rs: {}\x07", label(&paths[0]));
    }
    loop {
        let mut out = String::from("\x1b[H\x1b[2J");
        let mut all_done = true;
        for p in &paths {
            let panel = render(p);
            all_done &= panel.finished;
            out.push_str(&panel.text);
            out.push('\n');
        }
        if all_done {
            out.push_str(" Finished. Close this window when done.\n");
        }
        print!("{out}");
        use std::io::Write;
        let _ = std::io::stdout().flush();
        if all_done {
            loop {
                std::thread::park();
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

/// Windows: opens a new console window running this viewer on `path` (placement is
/// left to the console host). Launched with PowerShell's Start-Process (ShellExecute),
/// which does not pass on inheritable handles: a viewer created with CreateProcess, as
/// `std::process::Command` or `cmd /C start` do, would inherit the caller's (possibly
/// redirected) output handle and keep it open until the window is closed. The paths go
/// through the environment to avoid quoting them for PowerShell.
#[cfg(windows)]
pub fn open_window(path: &Path, _k: usize) -> Result<(), String> {
    use std::process::Stdio;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let path = std::path::absolute(path).map_err(|e| e.to_string())?;
    let st = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command"])
        .arg("Start-Process -FilePath $env:DYNARE_PROGRESS_EXE -ArgumentList ('\"' + $env:DYNARE_PROGRESS_PATH + '\"')")
        .env("DYNARE_PROGRESS_EXE", exe)
        .env("DYNARE_PROGRESS_PATH", path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| e.to_string())?;
    if !st.success() {
        return Err(format!("Start-Process failed ({st})"));
    }
    Ok(())
}

/// Windows consoles interpret the ANSI sequences the viewer prints (clear screen, window
/// title) only with virtual-terminal processing on.
#[cfg(windows)]
fn enable_ansi() {
    unsafe extern "system" {
        fn GetStdHandle(n: u32) -> *mut std::ffi::c_void;
        fn GetConsoleMode(h: *mut std::ffi::c_void, mode: *mut u32) -> i32;
        fn SetConsoleMode(h: *mut std::ffi::c_void, mode: u32) -> i32;
    }
    const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
    // SAFETY: plain kernel32 calls on this process's own stdout handle.
    unsafe {
        let h = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut mode = 0;
        if GetConsoleMode(h, &mut mode) != 0 {
            SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }
}

#[cfg(not(windows))]
fn enable_ansi() {}

/// Opens a small Terminal window running this viewer on `path` (the `k`-th window
/// is placed below the previous one).
#[cfg(not(windows))]
pub fn open_window(path: &Path, k: usize) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let path = std::path::absolute(path).map_err(|e| e.to_string())?;
    let sh = |s: &Path| format!("'{}'", s.display().to_string().replace('\'', r"'\''"));
    let cmd = format!("clear; exec {} {}", sh(&exe), sh(&path));
    let cmd = cmd.replace('\\', r"\\").replace('"', "\\\"");
    let (x, y) = (40 + 30 * k as i64, 60 + 300 * k as i64);
    let script = format!(
        "tell application \"Terminal\"\n\
           set t to do script \"{cmd}\"\n\
           set number of columns of t to 78\n\
           set number of rows of t to 14\n\
           delay 0.5\n\
           set position of window 1 to {{{x}, {y}}}\n\
         end tell"
    );
    let st = std::process::Command::new("osascript").arg("-e").arg(script).status().map_err(|e| e.to_string())?;
    if !st.success() {
        return Err(format!("osascript failed ({st})"));
    }
    Ok(())
}

struct Panel {
    text: String,
    finished: bool,
}

fn label(p: &Path) -> String {
    let name = p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    name.trim_end_matches("_progress.json").to_string()
}

/// Last two components of `p` (the scratch paths are too long for the window).
fn short(p: &Path) -> String {
    let parts: Vec<String> = p.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect();
    parts[parts.len().saturating_sub(2)..].join("/")
}

fn mtime(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// Newest progress file or log in `dir` (and one level below, plus `*/Output`).
fn newest_in(dir: &Path) -> Option<PathBuf> {
    let mut cands = Vec::new();
    let mut scan = |d: &Path| {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let p = e.path();
                let n = p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                if n.ends_with("_progress.json") || n.ends_with(".log") {
                    cands.push(p);
                }
            }
        }
    };
    scan(dir);
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if e.path().is_dir() {
                scan(&e.path());
                scan(&e.path().join("Output"));
            }
        }
    }
    cands.into_iter().filter_map(|p| mtime(&p).map(|t| (t, p))).max_by_key(|(t, _)| *t).map(|(_, p)| p)
}

fn render(p: &Path) -> Panel {
    if p.is_dir() {
        let head = format!(" {}\n", short(p));
        return match newest_in(p) {
            Some(f) => {
                let mut panel = render(&f);
                panel.text = head + &panel.text;
                panel
            }
            None => Panel { text: head + " waiting for the run to start...\n", finished: false },
        };
    }
    let Ok(text) = std::fs::read_to_string(p) else {
        return Panel { text: format!(" waiting for {}...\n", p.display()), finished: false };
    };
    if p.extension().is_some_and(|e| e == "json") {
        match serde_json::from_str::<Value>(&text) {
            Ok(v) => render_json(&v),
            // Renamed into place atomically, so this is only a file from elsewhere.
            Err(e) => Panel { text: format!(" {}: {e}\n", p.display()), finished: false },
        }
    } else {
        render_log(p, &text)
    }
}

fn hms(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    format!("{}h{:02}m{:02}s", s / 3600, (s / 60) % 60, s % 60)
}

fn clock_in(secs: f64) -> String {
    let t = SystemTime::now() + Duration::from_secs_f64(secs.max(0.0));
    let unix = t.duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    // Local time through `date`, so the time zone is the user's.
    std::process::Command::new("date")
        .args(["-r", &unix.to_string(), "+%H:%M"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn bar(frac: f64, width: usize) -> String {
    let f = frac.clamp(0.0, 1.0);
    let full = (f * width as f64).round() as usize;
    format!("{}{}", "█".repeat(full), "░".repeat(width - full))
}

fn eta_line(done: f64, total: f64, elapsed: f64) -> String {
    if done <= 0.0 || done >= total {
        return String::new();
    }
    let left = elapsed * (total - done) / done;
    format!(" ETA {} (about {})\n", hms(left), clock_in(left))
}

fn alive(pid: u64) -> bool {
    std::process::Command::new("kill").args(["-0", &pid.to_string()]).status().is_ok_and(|s| s.success())
}

fn render_json(v: &Value) -> Panel {
    let f = |k: &str| v[k].as_f64().unwrap_or(0.0);
    let s = |k: &str| v[k].as_str().unwrap_or("").to_string();
    let status = s("status");
    let mut t = format!(" {:<40} {:>10}  {}\n", s("model"), status, hms(f("elapsed")));
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    let stale = now - f("updated");
    let finished = status != "running";
    if !finished && stale > 5.0 && !alive(f("pid") as u64) {
        t.push_str(&format!(" the process ended without finishing (last update {} ago)\n", hms(stale)));
        return Panel { text: t, finished: true };
    }
    if status == "error" {
        t.push_str(&format!(" ERROR: {}\n", s("message")));
        return Panel { text: t, finished };
    }
    t.push_str(&format!(" {}  [{}]\n", s("phase"), hms(f("phase_elapsed"))));
    let chains = v["chains"].as_array().cloned().unwrap_or_default();
    if !chains.is_empty() {
        let (mut done, mut total) = (0.0, 0.0);
        for (i, c) in chains.iter().enumerate() {
            let d = c["done"].as_f64().unwrap_or(0.0);
            let n = c["total"].as_f64().unwrap_or(1.0).max(1.0);
            done += d;
            total += n;
            t.push_str(&format!(
                " chain {:<2} {} {:5.1}%  acc {:5.2}%\n",
                i + 1,
                bar(d / n, 34),
                100.0 * d / n,
                100.0 * c["acceptance"].as_f64().unwrap_or(0.0)
            ));
        }
        t.push_str(&eta_line(done, total, f("phase_elapsed")));
    } else {
        let evals = f("evals");
        let rate = evals / f("phase_elapsed").max(1e-9);
        t.push_str(&format!(" {} posterior evaluations ({:.0}/s)", evals as u64, rate));
        if let Some(b) = v["best"].as_f64() {
            t.push_str(&format!(", best minus log posterior {b:.6}"));
        }
        t.push('\n');
    }
    let d = s("detail");
    if !d.is_empty() {
        t.push_str(&format!(" {d}\n"));
    }
    Panel { text: t, finished }
}

/// "[0h21m31s]" or "[23.9s]" to seconds.
fn parse_hms(s: &str) -> Option<f64> {
    let s = s.trim().trim_start_matches('[').trim_end_matches(']');
    let (mut tot, mut num) = (0.0, String::new());
    for ch in s.chars() {
        match ch {
            '0'..='9' | '.' => num.push(ch),
            'h' | 'm' | 's' => {
                let v: f64 = num.parse().ok()?;
                tot += v * match ch {
                    'h' => 3600.0,
                    'm' => 60.0,
                    _ => 1.0,
                };
                num.clear();
            }
            _ => return None,
        }
    }
    Some(tot)
}

fn render_log(p: &Path, text: &str) -> Panel {
    let mut t = format!(" {}\n", short(p));
    let lines: Vec<&str> = text.lines().collect();
    let last = |pat: &str| lines.iter().rposition(|l| l.contains(pat));
    if let Some(i) = last("ERROR") {
        t.push_str(&format!(" {}\n", lines[i].trim()));
        return Panel { text: t, finished: true };
    }
    if let Some(i) = last("Total computing time") {
        t.push_str(&format!(" finished. {}\n", lines[i].trim()));
        return Panel { text: t, finished: true };
    }
    let mcmc = last("chains x");
    let chain1 = last("chain 1:");
    if let (Some(m), Some(c)) = (mcmc, chain1)
        && c > m
    {
        // "  chain 1:    700000/2000000 draws, acceptance 31.23%  [0h21m31s]"
        let l = lines[c];
        let nums: Option<(f64, f64)> = l.split_whitespace().nth(2).and_then(|f| {
            let (a, b) = f.split_once('/')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        });
        let acc = l.split("acceptance").nth(1).and_then(|r| r.split('%').next()).map(str::trim).unwrap_or("?");
        let el = l.rsplit('[').next().and_then(parse_hms);
        t.push_str(&format!(" {}\n", lines[m].trim().trim_start_matches("Estimation::mcmc: ")));
        if let (Some((d, n)), Some(el)) = (nums, el) {
            t.push_str(&format!(" chain 1  {} {:5.1}%  acc {acc}%  [{}]\n", bar(d / n, 34), 100.0 * d / n, hms(el)));
            let age = mtime(p).and_then(|m| m.elapsed().ok()).map_or(0.0, |a| a.as_secs_f64());
            if d > 0.0 && d < n {
                let left = (el * (n - d) / d - age).max(0.0);
                t.push_str(&format!(" ETA {} (about {})\n", hms(left), clock_in(left)));
            }
            t.push_str(&format!(" (the log prints every 5% of the draws; last line {} ago)\n", hms(age)));
        }
        return Panel { text: t, finished: false };
    }
    let interesting = [
        "f at the beginning of new iteration",
        "Current mode =",
        "New value of jscale",
        "Iteration ",
        "mode_compute =",
        "Final value",
        "mh_jscale",
        "Hessian",
        "Smoother",
        "Laplace",
    ];
    if let Some(i) = lines.iter().rposition(|l| interesting.iter().any(|k| l.contains(k))) {
        t.push_str(&format!(" {}\n", lines[i].trim()));
    } else if let Some(l) = lines.iter().rev().find(|l| !l.trim().is_empty()) {
        t.push_str(&format!(" {}\n", l.trim()));
    }
    Panel { text: t, finished: false }
}
