//! Library-level (API call) comparison of `pure-magic` (via the precompiled
//! `magic-db` ruleset) against real `libmagic` (via the `magic` crate's FFI
//! bindings, loaded with its default system database). Unlike the
//! process-level hyperfine benchmarks, this isolates the
//! matching engine itself from process-spawn overhead.
//!
//! The many-small-files scenario runs twice, with and without a matching
//! file extension, since `pure_magic_rs::MagicDb::first_magic_file` uses
//! the extension as an acceleration hint that real `libmagic` doesn't have
//! access to -- see `many_small_files`'s doc comment.
//!
//! Each scenario is benchmarked through both APIs each library exposes:
//! path-based (`first_magic_file` / `Cookie::file`, which touches the
//! filesystem: open, seek, read) and buffer-based (`first_magic_slice` /
//! `Cookie::buffer`, given bytes already in memory). The buffer API is the
//! fairer one for isolating pure matching-engine speed, since it removes
//! filesystem I/O from both sides; the path API is what most real callers
//! actually use.
//!
//! `medium_file` exists specifically to make `pure-magic`'s
//! `LazyCache` hot/warm/cold tier sizing visible: `large_file` and
//! `SMALL_FILE_TEMPLATES` both have their only real content within the
//! first ~200 bytes, so every hot-cache size touches the same bytes and
//! the scenario can't distinguish a bigger cache from a smaller one.
//! `medium_file` has its real content (an ISO 9660 signature) at a fixed
//! offset of 32769 bytes -- see `medium_file`'s doc comment.
//!
//! Run with: `cargo bench -p magic-bench`

use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use magic::Cookie;
use magic::cookie::{DatabasePaths, Flags};

const LARGE_FILE_SIZE: u64 = 64 * 1024 * 1024; // 64 MiB
const MEDIUM_FILE_SIZE: u64 = 512 * 1024; // 512 KiB
const SMALL_FILE_COUNT: usize = 2000;

/// Shared counter that numbers each benchmark group in the order it's
/// actually created at run time -- which, since criterion runs the
/// functions listed in `criterion_group!` at the bottom of this file
/// sequentially, matches their declaration order there. Prefixing group
/// names with it (`NN/scenario`, sanitized to `NN_scenario` on disk like
/// the `pure_magic/file` function names) lets `summarize.rs` sort groups
/// without hardcoding that order a second time.
static GROUP_COUNTER: AtomicUsize = AtomicUsize::new(1);

fn group_id(name: &str) -> String {
    let n = GROUP_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{n:02}/{name}")
}

/// A genuinely valid single-entry deflate zip (local file header +
/// central directory + end-of-central-directory), not just a truncated
/// local file header -- real `libmagic`'s installed ruleset requires a
/// central-directory record to be present within the first 1024 bytes,
/// which a bare local file header doesn't have (confirmed empirically:
/// `file` reports a truncated header as "data").
const ZIP_TEMPLATE: &[u8] = b"PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00!X-;\x08\xaf\x0e\x00\x00\x00\x0c\x00\x00\x00\x05\x00\x00\x00a.txt\xcbH\xcd\xc9\xc9W(\xcf/\xcaI\xe1\x02\x00PK\x01\x02\x14\x03\x14\x00\x00\x00\x08\x00\x00\x00!X-;\x08\xaf\x0e\x00\x00\x00\x0c\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x01\x00\x00\x00\x00a.txtPK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x003\x00\x00\x001\x00\x00\x00\x00\x00";

/// A large file with a real, recognizable header (`ZIP_TEMPLATE`)
/// followed by zero padding. Deliberately not ELF: `libmagic` parses ELF
/// program/section headers via a dedicated C parser, not
/// just magic-rule matching, which `pure-magic` doesn't replicate --
/// using ELF here would benchmark that gap instead of matching-engine speed.
fn large_file(dir: &Path) -> PathBuf {
    let path = dir.join("large_file.bin");

    let mut f = File::create(&path).expect("create large_file fixture");
    f.write_all(ZIP_TEMPLATE).expect("write header");

    let zeros = vec![0u8; 1024 * 1024];
    let mut written = ZIP_TEMPLATE.len() as u64;
    while written < LARGE_FILE_SIZE {
        let n = zeros.len().min((LARGE_FILE_SIZE - written) as usize);
        f.write_all(&zeros[..n]).expect("write padding");
        written += n as u64;
    }
    path
}

/// A 512 KiB file whose only real content is an ISO 9660 "CD001" signature
/// at the format's fixed offset 32769 (`magic-db/src/magdir/filesystems`),
/// everything else zero. Unlike `large_file`/`SMALL_FILE_TEMPLATES` (whose
/// real content sits in the first ~200 bytes), this forces a read well
/// past any small head cache
fn medium_file(dir: &Path) -> PathBuf {
    let path = dir.join("medium_file.bin");
    let mut f = File::create(&path).expect("create medium_file fixture");
    let mut written = 0u64;
    let zeros = vec![0u8; 4096];
    while written < MEDIUM_FILE_SIZE {
        let n = zeros.len().min((MEDIUM_FILE_SIZE - written) as usize);
        f.write_all(&zeros[..n]).expect("write padding");
        written += n as u64;
    }
    // ISO 9660 primary volume descriptor signature, offset 32769.
    use std::io::{Seek, SeekFrom};
    f.seek(SeekFrom::Start(32769))
        .expect("seek to iso9660 offset");
    f.write_all(b"CD001").expect("write iso9660 signature");
    path
}

/// Byte-signature templates for a variety of real, distinct file types,
/// used to build a directory of many small files without depending on
/// whatever happens to be installed on the machine running the benchmark.
/// Each signature (and its extension tag) is checked against the actual
/// `magic-db/src/magdir` rule that recognizes it, so the `with_ext`
/// variant genuinely exercises that rule's `!:ext` acceleration path
/// rather than just carrying a plausible-looking file name.
///
/// Several templates (zip, bmp, jpg) carry enough real header fields to
/// reach 3-4 levels into that rule's continuation chain rather than
/// stopping at the first matching line, so the benchmark exercises more
/// than a single-test fast path. Each one is verified against both real
/// `file` and `wiza` to produce the same, fully-populated description.
///
/// Deliberately no ELF template: `libmagic` parses ELF program/section
/// headers via a dedicated C parser (`readelf.c`), not just magic-rule
/// matching, which `pure-magic` doesn't replicate (see README's
/// "Differences from libmagic") -- including ELF here would benchmark
/// that scope gap instead of matching-engine speed.
const SMALL_FILE_TEMPLATES: &[(&str, &[u8])] = &[
    ("png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"),
    ("zip", ZIP_TEMPLATE),
    ("pdf", b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n"),
    ("sh", b"#!/bin/sh\necho hello world\n"),
    ("gif", b"GIF89a\x01\x00\x01\x00"),
    (
        // SOI + APP0(JFIF) with version/density fields populated, so the
        // `jpeg` rule's nested `>6 string JFIF` -> `>>11/>>12/>>13/>>14`
        // branches (version, aspect ratio, density) all fire, not just
        // the top-level signature match.
        "jpg",
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x02\x01\x00H\x00H\x00\x00",
    ),
    (
        // "BM" + reserved/offBits fields + a DIB header size of 12
        // (OS/2 1.x format) + width/height/bits-per-pixel, reaching the
        // `>>18`/`>>20`/`>>24` branches under `bitmap-bmp`.
        "bmp",
        b"BM\x00\x00\x00\x00\x00\x00\x00\x00\x1a\x00\x00\x00\x0c\x00\x00\x00@\x000\x00\x01\x00\x18\x00",
    ),
    // gzip header with FLG=0 (no FNAME/FCOMMENT).
    ("gz", b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03"),
    ("rar", b"Rar!\x1a\x07\x01\x00"), // RAR v5 signature, !:ext rar
    ("wasm", b"\x00asm\x01\x00\x00\x00"),
];

/// Many small files, cycling through `SMALL_FILE_TEMPLATES` verbatim --
/// no added filler, since several templates are structured binary formats
/// whose fields are read past the signature, and padding would get parsed
/// as more struct fields and corrupt the match.
///
/// `with_ext` names files with their real, matching extension or with
/// none at all, since `pure_magic`'s `first_magic_file` uses the
/// extension as an acceleration hint that real `libmagic` never has.
fn many_small_files(dir: &Path, with_ext: bool) -> Vec<PathBuf> {
    let name_of = |i: usize| {
        let (kind, _) = SMALL_FILE_TEMPLATES[i % SMALL_FILE_TEMPLATES.len()];
        if with_ext {
            format!("{i:05}.{kind}")
        } else {
            format!("{i:05}")
        }
    };

    let paths: Vec<PathBuf> = (0..SMALL_FILE_COUNT)
        .map(|i| dir.join(name_of(i)))
        .collect();
    for (i, path) in paths.iter().enumerate() {
        let (_, header) = SMALL_FILE_TEMPLATES[i % SMALL_FILE_TEMPLATES.len()];
        let mut f = File::create(path).expect("create small file fixture");
        f.write_all(header).expect("write header");
    }
    paths
}

fn pure_magic_db() -> pure_magic::MagicDb {
    magic_db::load().expect("load embedded magic-db ruleset")
}

fn libmagic_cookie() -> Cookie<magic::cookie::Load> {
    Cookie::open(Flags::default())
        .expect("open libmagic cookie")
        .load(&DatabasePaths::default())
        .expect("load libmagic default database")
}

fn bench_single_large_file(c: &mut Criterion) {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let path = large_file(tmp.path());
    let db = pure_magic_db();
    let cookie = libmagic_cookie();

    let mut group = c.benchmark_group(group_id("single_large_file"));
    group.sample_size(20);
    group.bench_function("pure_magic/file", |b| {
        b.iter(|| black_box(db.first_magic_file(&path).unwrap()))
    });
    group.bench_function("libmagic/file", |b| {
        b.iter(|| black_box(cookie.file(&path).unwrap()))
    });
    group.finish();
}

fn bench_medium_file(c: &mut Criterion) {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let path = medium_file(tmp.path());
    let db = pure_magic_db();
    let cookie = libmagic_cookie();

    let mut group = c.benchmark_group(group_id("medium_file"));
    group.bench_function("pure_magic/file", |b| {
        b.iter(|| black_box(db.first_magic_file(&path).unwrap()))
    });
    group.bench_function("libmagic/file", |b| {
        b.iter(|| black_box(cookie.file(&path).unwrap()))
    });
    group.finish();
}

fn bench_medium_file_buffer(c: &mut Criterion) {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let path = medium_file(tmp.path());
    let bytes = fs::read(&path).expect("read medium_file fixture into memory");
    let db = pure_magic_db();
    let cookie = libmagic_cookie();

    let mut group = c.benchmark_group(group_id("medium_file"));
    group.bench_function("pure_magic/buffer", |b| {
        b.iter(|| black_box(db.first_magic_slice(&bytes, None).unwrap()))
    });
    group.bench_function("libmagic/buffer", |b| {
        b.iter(|| black_box(cookie.buffer(&bytes).unwrap()))
    });
    group.finish();
}

fn bench_many_small_files(c: &mut Criterion, with_ext: bool, group_name: &str) {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let files = many_small_files(tmp.path(), with_ext);
    let db = pure_magic_db();
    let cookie = libmagic_cookie();

    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(files.len() as u64));
    group.bench_function("pure_magic/file", |b| {
        b.iter(|| {
            for f in &files {
                black_box(db.first_magic_file(f).unwrap());
            }
        })
    });
    group.bench_function("libmagic/file", |b| {
        b.iter(|| {
            for f in &files {
                black_box(cookie.file(f).unwrap());
            }
        })
    });
    group.finish();
}

fn bench_many_small_files_with_ext(c: &mut Criterion) {
    bench_many_small_files(c, true, &group_id("many_small_files_with_ext"));
}

fn bench_many_small_files_no_ext(c: &mut Criterion) {
    bench_many_small_files(c, false, &group_id("many_small_files_no_ext"));
}

fn bench_single_large_file_buffer(c: &mut Criterion) {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let path = large_file(tmp.path());
    let bytes = fs::read(&path).expect("read large_file fixture into memory");
    let db = pure_magic_db();
    let cookie = libmagic_cookie();

    let mut group = c.benchmark_group(group_id("single_large_file"));
    group.sample_size(20);
    group.bench_function("pure_magic/buffer", |b| {
        b.iter(|| black_box(db.first_magic_slice(&bytes, None).unwrap()))
    });
    group.bench_function("libmagic/buffer", |b| {
        b.iter(|| black_box(cookie.buffer(&bytes).unwrap()))
    });
    group.finish();
}

/// Same file set as `bench_many_small_files`, but read into memory once
/// up front and matched via the buffer APIs (no filesystem access inside
/// the timed loop). `libmagic`'s `Cookie::buffer` has no concept of a
/// filename at all, so it's always a no-extension call; `pure_magic`'s
/// `first_magic_slice` takes an explicit, caller-supplied `extension`
/// hint, which we pass only in the `with_ext` variant to show the
/// acceleration is opt-in on the buffer API too, not just a side effect
/// of using paths.
fn bench_many_small_files_buffer(c: &mut Criterion, with_ext: bool, group_name: &str) {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let files = many_small_files(tmp.path(), with_ext);
    let buffers: Vec<(Vec<u8>, Option<String>)> = files
        .iter()
        .map(|p| {
            let bytes = fs::read(p).expect("read small file fixture into memory");
            let ext = p.extension().and_then(|e| e.to_str()).map(String::from);
            (bytes, ext)
        })
        .collect();
    let db = pure_magic_db();
    let cookie = libmagic_cookie();

    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(buffers.len() as u64));
    group.bench_function("pure_magic/buffer", |b| {
        b.iter(|| {
            for (bytes, ext) in &buffers {
                black_box(db.first_magic_slice(bytes, ext.as_deref()).unwrap());
            }
        })
    });
    group.bench_function("libmagic/buffer", |b| {
        b.iter(|| {
            for (bytes, _) in &buffers {
                black_box(cookie.buffer(bytes).unwrap());
            }
        })
    });
    group.finish();
}

fn bench_many_small_files_buffer_with_ext(c: &mut Criterion) {
    bench_many_small_files_buffer(c, true, &group_id("many_small_files_with_ext"));
}

fn bench_many_small_files_buffer_no_ext(c: &mut Criterion) {
    bench_many_small_files_buffer(c, false, &group_id("many_small_files_no_ext"));
}

criterion_group!(
    benches,
    bench_single_large_file,
    bench_medium_file,
    bench_many_small_files_with_ext,
    bench_many_small_files_no_ext,
    bench_single_large_file_buffer,
    bench_medium_file_buffer,
    bench_many_small_files_buffer_with_ext,
    bench_many_small_files_buffer_no_ext,
);
criterion_main!(benches);
