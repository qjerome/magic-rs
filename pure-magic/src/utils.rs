use jiff::{
    Timestamp,
    civil::{Date, Time},
    tz::TimeZone,
};
use memchr::memchr2;

use crate::TIMESTAMP_FORMAT;

#[inline(always)]
pub(crate) const fn decode_id3(v: u32) -> u32 {
    (v & 0x7f) | (((v >> 8) & 0x7f) << 7) | (((v >> 16) & 0x7f) << 14) | (((v >> 24) & 0x7f) << 21)
}

// test this properly
#[inline]
pub(crate) fn nonmagic(str: &str) -> usize {
    let mut rv = 0;
    let mut chars = str.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                // Escaped anything counts 1
                if chars.peek().is_none() {
                    rv += 1; // Handle trailing backslash
                } else {
                    chars.next(); // Skip the escaped character
                    rv += 1;
                }
            }
            '?' | '*' | '.' | '+' | '^' | '$' => {
                // Magic characters count 0
                continue;
            }
            '[' => {
                // Bracketed expressions count 1
                rv += 1;
                // Skip until closing ']' or end of string
                while let Some(&ch) = chars.peek() {
                    chars.next();
                    if ch == ']' {
                        break;
                    }
                }
            }
            '{' => {
                // Braced expressions count 0
                // Skip until closing '}' or end of string
                while let Some(&ch) = chars.peek() {
                    chars.next();
                    if ch == '}' {
                        break;
                    }
                }
            }
            _ => {
                // Anything else counts 1
                rv += 1;
            }
        }
    }

    if rv == 0 { 1 } else { rv }
}

/// Parses a u16 FAT/DOS date into a `Date`.
// https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-dosdatetimetofiletime?redirectedfrom=MSDN
pub(crate) fn parse_fat_date(fat_date: u16) -> Option<Date> {
    let day = (fat_date & 0x1f) as i8; // Bits 0-4
    let month = ((fat_date >> 5) & 0xf) as i8; // Bits 5-8
    let year = (fat_date >> 9) as i16 + 1980; // Bits 9-15 + 1980
    Date::new(year, month, day).ok()
}

pub(crate) fn parse_fat_time(fat_time: u16) -> Option<Time> {
    // time is encoded in 2 sec
    let sec = ((fat_time & 0x1f) * 2) as i8;
    let min = ((fat_time >> 5) & 0b111111) as i8;
    let hour = ((fat_time >> 11) & 0b11111) as i8;
    Time::new(hour, min, sec, 0).ok()
}

fn windows_time_to_datetime(filetime: i64) -> Option<Timestamp> {
    // FILETIME starts 1601-01-01, which is this many seconds before the Unix epoch
    const WINDOWS_EPOCH_SECS: i64 = -11_644_473_600;
    // convert 100ns units into seconds + nanos
    let secs = filetime / 10_000_000;
    let nanos = ((filetime % 10_000_000) * 100) as i32;
    let elapsed = jiff::SignedDuration::new(secs, nanos);
    Timestamp::from_second(WINDOWS_EPOCH_SECS)
        .ok()?
        .checked_add(elapsed)
        .ok()
}

#[inline(always)]
pub(crate) fn unix_local_time_to_string(timestamp: i64) -> String {
    Timestamp::from_second(timestamp)
        .map(|ts| {
            ts.to_zoned(TimeZone::system())
                .strftime(TIMESTAMP_FORMAT)
                .to_string()
        })
        .unwrap_or_else(|_| "invalid timestamp".into())
}

#[inline(always)]
pub(crate) fn unix_utc_time_to_string(timestamp: i64) -> String {
    Timestamp::from_second(timestamp)
        .map(|ts| ts.strftime(TIMESTAMP_FORMAT).to_string())
        .unwrap_or_else(|_| "invalid timestamp".into())
}

#[inline(always)]
pub(crate) fn windows_filetime_to_string(timestamp: i64) -> String {
    windows_time_to_datetime(timestamp)
        .map(|ts| ts.strftime(TIMESTAMP_FORMAT).to_string())
        .unwrap_or_else(|| "invalid timestamp".into())
}

#[inline(always)]
pub(crate) fn find_json_boundaries<S: AsRef<[u8]>>(buf: S) -> Option<(usize, usize)> {
    let buf = buf.as_ref();
    let mut cnt = 0usize;
    let mut in_string = false;
    let mut prev_char = 0u8;

    let i_open = memchr2(b'[', b'{', buf)?;

    let (opening, closing) = if buf[i_open] == b'[' {
        (b'[', b']')
    } else {
        (b'{', b'}')
    };

    for (i, c) in buf[i_open..].iter().enumerate() {
        if c == &b'"' && prev_char != b'\\' {
            in_string ^= true
        }
        if c == &opening && !in_string {
            cnt += 1
        } else if c == &closing && !in_string {
            cnt = cnt.saturating_sub(1)
        }
        if cnt == 0 {
            return Some((i_open, i + i_open));
        }
        prev_char = *c;
    }

    None
}

pub(crate) fn debug_string_from_vec_u8(data: &[u8]) -> String {
    let mut result = String::new();
    for &byte in data {
        if byte.is_ascii_graphic() || byte == b' ' {
            result.push(byte as char);
        } else {
            result.push_str(&format!("\\x{:02x}", byte));
        }
    }
    result
}

pub(crate) fn debug_string_from_vec_u16(data: &[u16]) -> String {
    let mut result = String::new();
    for &short in data {
        for byte in short.to_be_bytes() {
            if byte.is_ascii_graphic() || byte == b' ' {
                result.push(byte as char);
            } else {
                result.push_str(&format!("\\x{:02x}", byte));
            }
        }
    }
    result
}

/// Copy of rust standard library Utf8Error, to be returned
/// by run_utf8_validation
#[derive(Copy, Eq, PartialEq, Clone, Debug)]
pub(crate) struct Utf8Error {
    pub(crate) valid_up_to: usize,
    pub(crate) error_len: Option<u8>,
}

// https://tools.ietf.org/html/rfc3629
const UTF8_CHAR_WIDTH: &[u8; 256] = &[
    // 1  2  3  4  5  6  7  8  9  A  B  C  D  E  F
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 0
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 1
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 3
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 4
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 5
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 6
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 7
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 8
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 9
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // B
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // C
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // D
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // E
    4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // F
];

#[inline]
const fn utf8_char_width(b: u8) -> usize {
    UTF8_CHAR_WIDTH[b as usize] as usize
}

/// Mirrors libmagic's `text_chars` table (encoding.c): only these ASCII-range
/// bytes are considered text-like. A single other byte in 0x00-0x7F (most
/// notably NUL) makes libmagic treat the whole buffer as binary, even if
/// every byte is otherwise valid ASCII/UTF-8 (looks_ascii/file_looks_utf8).
const ASCII_TEXT_CHAR: [bool; 128] = {
    let mut t = [false; 128];
    let mut i = 0;
    while i < 128 {
        t[i] = matches!(i as u8, 0x07..=0x0d | 0x1a | 0x1b | 0x20..=0x7e);
        i += 1;
    }
    t
};

/// This is a pale, modified and simplified copy of the Rust std method
/// to do UTF8 validation. It has been changed to return a boolean flag
/// if buffer is full of ascii characters, and to reject (as if invalid)
/// any ASCII-range byte outside libmagic's `text_chars` whitelist.
#[inline(always)]
pub(crate) fn run_utf8_validation(v: &[u8]) -> Result<bool, Utf8Error> {
    let mut index = 0;
    let len = v.len();
    let mut ascii_only = len > 0;

    while index < len {
        let old_offset = index;
        macro_rules! err {
            ($error_len: expr) => {
                return Err(Utf8Error {
                    valid_up_to: old_offset,
                    error_len: $error_len,
                })
            };
        }

        macro_rules! next {
            () => {{
                index += 1;
                // we needed data, but there was none: error!
                if index >= len {
                    err!(None)
                }
                v[index]
            }};
        }

        let first = v[index];
        if first >= 128 {
            let w = utf8_char_width(first);
            // 2-byte encoding is for codepoints  \u{0080} to  \u{07ff}
            //        first  C2 80        last DF BF
            // 3-byte encoding is for codepoints  \u{0800} to  \u{ffff}
            //        first  E0 A0 80     last EF BF BF
            //   excluding surrogates codepoints  \u{d800} to  \u{dfff}
            //               ED A0 80 to       ED BF BF
            // 4-byte encoding is for codepoints \u{10000} to \u{10ffff}
            //        first  F0 90 80 80  last F4 8F BF BF
            //
            // Use the UTF-8 syntax from the RFC
            //
            // https://tools.ietf.org/html/rfc3629
            // UTF8-1      = %x00-7F
            // UTF8-2      = %xC2-DF UTF8-tail
            // UTF8-3      = %xE0 %xA0-BF UTF8-tail / %xE1-EC 2( UTF8-tail ) /
            //               %xED %x80-9F UTF8-tail / %xEE-EF 2( UTF8-tail )
            // UTF8-4      = %xF0 %x90-BF 2( UTF8-tail ) / %xF1-F3 3( UTF8-tail ) /
            //               %xF4 %x80-8F 2( UTF8-tail )
            match w {
                2 => {
                    if next!() as i8 >= -64 {
                        err!(Some(1))
                    }
                }
                3 => {
                    match (first, next!()) {
                        (0xE0, 0xA0..=0xBF)
                        | (0xE1..=0xEC, 0x80..=0xBF)
                        | (0xED, 0x80..=0x9F)
                        | (0xEE..=0xEF, 0x80..=0xBF) => {}
                        _ => err!(Some(1)),
                    }
                    if next!() as i8 >= -64 {
                        err!(Some(2))
                    }
                }
                4 => {
                    match (first, next!()) {
                        (0xF0, 0x90..=0xBF) | (0xF1..=0xF3, 0x80..=0xBF) | (0xF4, 0x80..=0x8F) => {}
                        _ => err!(Some(1)),
                    }
                    if next!() as i8 >= -64 {
                        err!(Some(2))
                    }
                    if next!() as i8 >= -64 {
                        err!(Some(3))
                    }
                }
                _ => err!(Some(1)),
            }
            index += 1;
            ascii_only = false;
        } else {
            if !ASCII_TEXT_CHAR[first as usize] {
                err!(None)
            }
            index += 1;
        }
    }

    Ok(ascii_only)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_dates() {
        // (FAT date, expected (year, month, day))
        // test values from: https://github.com/nathanhi/pyfatfs/blob/master/tests/test_DosDateTime.py
        let cases = [
            (0x5490, (2022, 4, 16)),
            (0x21, (1980, 1, 1)),
            (0xff9f, (2107, 12, 31)),
        ];

        for (fat_date, (year, month, day)) in cases {
            let parsed = parse_fat_date(fat_date).unwrap();
            assert_eq!(parsed.year(), year);
            assert_eq!(parsed.month(), month);
            assert_eq!(parsed.day(), day);
        }
    }

    #[test]
    fn test_invalid_dates() {
        assert_eq!(parse_fat_date(0xffa0), None)
    }

    #[test]
    fn test_valid_times() {
        assert_eq!(parse_fat_time(0x0), Time::new(0, 0, 0, 0).ok());

        assert_eq!(parse_fat_time(0xbf7d), Time::new(23, 59, 58, 0).ok());

        assert_eq!(parse_fat_time(18301), Time::new(8, 59, 58, 0).ok());
    }

    #[test]
    fn test_windows_date() {
        assert_eq!(windows_filetime_to_string(0), "1601-01-01 00:00:00");
        assert_eq!(
            windows_filetime_to_string(132723834270000000),
            "2021-08-02 13:10:27"
        );
    }

    #[test]
    fn test_decode_id3() {
        assert_eq!(decode_id3(0x00000000), 0);
        assert_eq!(decode_id3(0x0000007F), 127);
        assert_eq!(decode_id3(0x00007F7F), 16_383);
        assert_eq!(decode_id3(0x0000017E), 254);
        assert_eq!(decode_id3(0x7F7F7F7F), 268_435_455);
    }

    #[test]
    fn test_find_json_boundaries() {
        // Valid JSON objects
        assert_eq!(find_json_boundaries(b"{\"key\": \"value\"}"), Some((0, 15)));
        assert_eq!(find_json_boundaries(b"{\"a\": {\"b\": []}}"), Some((0, 15)));
        assert_eq!(find_json_boundaries(b"{\"a\": [1, 2, 3]}"), Some((0, 15)));

        // Valid JSON arrays
        assert_eq!(find_json_boundaries(b"[1, 2, 3]"), Some((0, 8)));
        assert_eq!(
            find_json_boundaries(b"[{\"a\": 1}, {\"b\": 2}]"),
            Some((0, 19))
        );

        // Nested structures
        assert_eq!(
            find_json_boundaries(b"{\"a\": {\"b\": {\"c\": []}}}"),
            Some((0, 22))
        );
        assert_eq!(find_json_boundaries(b"[[[1, 2], [3, 4]]]"), Some((0, 17)));

        // Strings with escaped quotes
        assert_eq!(
            find_json_boundaries(b"{\"key\": \"va\\\"lue\"}"),
            Some((0, 17))
        );
        assert_eq!(
            find_json_boundaries(b"[{\"a\": \"va\\\"lue\"}]"),
            Some((0, 17))
        );

        // No JSON
        assert_eq!(find_json_boundaries(b"no json here"), None);

        // Incomplete JSON
        assert_eq!(find_json_boundaries(b"{\"key\": \"value\""), None);
        assert_eq!(find_json_boundaries(b"[1, 2, 3"), None);

        // Multiple JSON documents
        assert_eq!(find_json_boundaries(b"{\"a\": 1}{\"b\": 2}"), Some((0, 7)));
        assert_eq!(find_json_boundaries(b"[1, 2][3, 4]"), Some((0, 5)));

        // Edge cases
        assert_eq!(find_json_boundaries(b"{}"), Some((0, 1)));
        assert_eq!(find_json_boundaries(b"[]"), Some((0, 1)));
        assert_eq!(find_json_boundaries(b""), None);
        assert_eq!(find_json_boundaries(b"   {\"a\": 1}   "), Some((3, 10)));
    }
}
