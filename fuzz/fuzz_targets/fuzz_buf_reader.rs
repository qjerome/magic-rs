#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use pure_magic::readers::{BufReader, DataRead};
use std::io::SeekFrom;

#[derive(Arbitrary, Debug)]
struct Input {
    buf: Vec<u8>,
    /// Initial seek before running ops, to exercise past-end starting positions.
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

fuzz_target!(|input: Input| {
    let mut r = BufReader::from_slice(input.buf.as_slice());

    // Seed with an arbitrary starting position (may be past end — that's the point).
    let _ = r.seek(SeekFrom::Start(input.initial_pos));

    for op in input.ops.iter().take(32) {
        match op {
            Op::ReadRange(start, end) => {
                let _ = r.read_range(*start..*end);
            }
            Op::ReadCount(n) => {
                let _ = r.read_count(*n);
            }
            Op::ReadWhileOrLimit { stop_at, limit } => {
                let _ = r.read_while_or_limit(|b| b != *stop_at, *limit);
            }
            Op::ReadUntilOrLimit { byte, limit } => {
                let _ = r.read_until_or_limit(*byte, *limit);
            }
            Op::ReadUntilAnyDelimOrLimit { delims, limit } => {
                let _ = r.read_until_any_delim_or_limit(delims.as_slice(), *limit);
            }
            Op::ReadUntilUtf16OrLimit { utf16_char, limit } => {
                let _ = r.read_until_utf16_or_limit(utf16_char, *limit);
            }
            Op::SeekStart(n) => {
                let _ = r.seek(SeekFrom::Start(*n));
            }
            Op::SeekCurrent(n) => {
                let _ = r.seek(SeekFrom::Current(*n));
            }
            Op::SeekEnd(n) => {
                let _ = r.seek(SeekFrom::End(*n));
            }
        }
    }
});
