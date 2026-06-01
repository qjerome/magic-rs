#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use pure_magic::readers::{DataRead, LazyCache};
use std::io::{Cursor, SeekFrom};

#[derive(Arbitrary, Debug)]
struct Input {
    buf: Vec<u8>,
    /// Enable hot cache (pre-reads head/tail into RAM buffers).
    hot_cache: bool,
    /// Enable warm cache (memory-mapped intermediate region).
    warm_cache: bool,
    /// Initial seek before ops.
    initial_pos: u64,
    ops: Vec<Op>,
}

#[derive(Arbitrary, Debug)]
enum Op {
    ReadRange(u64, u64),
    ReadCount(u64),
    ReadWhileOrLimit { stop_at: u8, limit: u64 },
    ReadUntilOrLimit { byte: u8, limit: u64 },
    ReadUntilAnyDelimOrLimit { delims: Vec<u8>, limit: u64 },
    ReadUntilUtf16OrLimit { utf16_char: [u8; 2], limit: u64 },
    SeekStart(u64),
    SeekCurrent(i64),
    SeekEnd(i64),
}

// Hot and warm cache sizes are kept small so all three tiers (hot, warm, cold)
// get exercised even with the tiny inputs the fuzzer produces.
const HOT_CACHE_BYTES: usize = 64;
const WARM_CACHE_BYTES: u64 = 256;

fuzz_target!(|input: Input| {
    let cursor = Cursor::new(input.buf.clone());

    let cache = LazyCache::from_read_seek(cursor);
    let Ok(cache) = cache else { return };

    let cache = if input.hot_cache {
        match cache.with_hot_cache(HOT_CACHE_BYTES) {
            Ok(c) => c,
            Err(_) => return,
        }
    } else {
        cache
    };

    let mut cache = if input.warm_cache {
        cache.with_warm_cache(WARM_CACHE_BYTES)
    } else {
        cache
    };

    let _ = cache.seek(SeekFrom::Start(input.initial_pos));

    for op in input.ops.iter().take(32) {
        match op {
            Op::ReadRange(start, end) => {
                let _ = cache.read_range(*start..*end);
            }
            Op::ReadCount(n) => {
                let _ = cache.read_count(*n);
            }
            Op::ReadWhileOrLimit { stop_at, limit } => {
                let _ = cache.read_while_or_limit(|b| b != *stop_at, *limit);
            }
            Op::ReadUntilOrLimit { byte, limit } => {
                let _ = cache.read_until_or_limit(*byte, *limit);
            }
            Op::ReadUntilAnyDelimOrLimit { delims, limit } => {
                let _ = cache.read_until_any_delim_or_limit(delims.as_slice(), *limit);
            }
            Op::ReadUntilUtf16OrLimit { utf16_char, limit } => {
                let _ = cache.read_until_utf16_or_limit(utf16_char, *limit);
            }
            Op::SeekStart(n) => {
                let _ = cache.seek(SeekFrom::Start(*n));
            }
            Op::SeekCurrent(n) => {
                let _ = cache.seek(SeekFrom::Current(*n));
            }
            Op::SeekEnd(n) => {
                let _ = cache.seek(SeekFrom::End(*n));
            }
        }
    }
});
