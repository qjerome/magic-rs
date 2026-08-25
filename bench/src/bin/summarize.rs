//! Parses criterion's own machine-readable output
//! (`target/criterion/<group>/<function>/new/estimates.json`) and prints
//! a markdown table comparing `pure_magic` vs `libmagic` per benchmark
//! group, bolding whichever is faster.
//!
//! Run `cargo bench -p magic-bench` first, then:
//! `cargo run -p magic-bench --bin summarize`
//!
//! Accepts an optional path to the criterion output dir (default:
//! `target/criterion`, relative to the current directory).
//!
//! This tool doesn't hardcode a scenario ordering of its own: each
//! `vs_libmagic.rs` benchmark group is named `NN/scenario` (criterion
//! sanitizes the `/` to `_` on disk, same as the `pure_magic/file` /
//! `libmagic/buffer` function names), so the intended display order lives
//! entirely in the benchmark's own output and this tool just sorts by the
//! `NN` prefix it finds there.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn mean_ns(criterion_dir: &Path, group: &str, function: &str) -> Option<f64> {
    let path = criterion_dir
        .join(group)
        .join(function)
        .join("new")
        .join("estimates.json");
    let data = fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;
    json.get("mean")?.get("point_estimate")?.as_f64()
}

fn format_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.1} ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2} \u{b5}s", ns / 1_000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else {
        format!("{:.2} s", ns / 1_000_000_000.0)
    }
}

/// Splits a `NN_scenario` group directory name into its sort key and
/// display name. Groups without a numeric prefix (e.g. hand-run ad hoc
/// benchmarks) sort last, in filesystem order, and display unchanged.
fn order_and_name(group: &str) -> (usize, &str) {
    if let Some((prefix, rest)) = group.split_once('_')
        && !prefix.is_empty()
        && prefix.bytes().all(|b| b.is_ascii_digit())
        && let Ok(n) = prefix.parse::<usize>()
    {
        return (n, rest);
    }
    (usize::MAX, group)
}

/// The API a benchmark was run through. The scenario name carries no
/// `_buffer`/`_file` marker of its own (that would just duplicate what's
/// already in the function names), so this is detected straight from
/// which function subdirectory criterion actually wrote: `pure_magic/file`
/// / `libmagic/buffer` etc. sanitize to `pure_magic_file` / `libmagic_buffer`
/// on disk.
fn detect_api(criterion_dir: &Path, group: &str) -> Option<&'static str> {
    for api in ["file", "buffer"] {
        if criterion_dir
            .join(group)
            .join(format!("pure_magic_{api}"))
            .is_dir()
        {
            return Some(api);
        }
    }
    None
}

fn main() {
    let criterion_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/criterion"));

    let entries = fs::read_dir(&criterion_dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", criterion_dir.display()));

    let mut groups: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|name| name != "report")
        .collect();

    groups.sort_by_key(|g| {
        let (order, name) = order_and_name(g);
        (order, name.to_string())
    });

    println!("| Benchmark | API | pure_magic | libmagic | Speedup |");
    println!("| --- | --- | --- | --- | --- |");

    for group in &groups {
        let (_, name) = order_and_name(group);
        let Some(api) = detect_api(&criterion_dir, group) else {
            eprintln!("skipping {group}: no pure_magic_file/pure_magic_buffer subdirectory");
            continue;
        };
        let pm_function = format!("pure_magic_{api}");
        let lm_function = format!("libmagic_{api}");

        let (Some(pm), Some(lm)) = (
            mean_ns(&criterion_dir, group, &pm_function),
            mean_ns(&criterion_dir, group, &lm_function),
        ) else {
            eprintln!("skipping {group}: missing {pm_function}/{lm_function} data");
            continue;
        };

        let (pm_cell, lm_cell) = if pm <= lm {
            (format!("**{}**", format_ns(pm)), format_ns(lm))
        } else {
            (format_ns(pm), format!("**{}**", format_ns(lm)))
        };

        let speedup = if pm <= lm {
            format!("pure_magic {:.2}x", lm / pm)
        } else {
            format!("libmagic {:.2}x", pm / lm)
        };

        println!("| {name} | {api} | {pm_cell} | {lm_cell} | {speedup} |");
    }
}
