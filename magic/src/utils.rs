use chrono::{DateTime, Local, NaiveDate, NaiveTime, TimeZone, Utc};
use memchr::memchr2;

use crate::TIMESTAMP_FORMAT;

#[inline(always)]
pub(crate) const fn decode_id3(v: u32) -> u32 {
    (v & 0x7f)
        | (((v >> 8) & 0x7f) << 7)
        | (((v >> 16) & 0x7f) << 14)
        | (((v >> 24) & 0x7f) << 21)
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

/// Parses a u16 FAT/DOS date into a `NaiveDate`.
// https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-dosdatetimetofiletime?redirectedfrom=MSDN
pub(crate) fn parse_fat_date(fat_date: u16) -> Option<NaiveDate> {
    let day = (fat_date & 0x1f) as u32; // Bits 0-4
    let month = ((fat_date >> 5) & 0xf) as u32; // Bits 5-8
    let year = (fat_date >> 9) as i32 + 1980; // Bits 9-15 + 1980
    NaiveDate::from_ymd_opt(year, month, day)
}

pub(crate) fn parse_fat_time(fat_time: u16) -> Option<NaiveTime> {
    // time is encoded in 2 sec
    let sec = (fat_time & 0x1f) * 2;
    let min = (fat_time >> 5) & 0b111111;
    let hour = (fat_time >> 11) & 0b11111;
    NaiveTime::from_hms_opt(hour as u32, min as u32, sec as u32)
}

fn windows_time_to_datetime(filetime: i64) -> Option<DateTime<Utc>> {
    // FILETIME starts 1601-01-01
    let windows_epoch = Utc.with_ymd_and_hms(1601, 1, 1, 0, 0, 0).single()?;
    // convert 100ns units into seconds + nanos
    let secs = filetime / 10_000_000;
    let nanos = (filetime % 10_000_000) * 100;
    Some(windows_epoch + chrono::Duration::seconds(secs) + chrono::Duration::nanoseconds(nanos))
}

#[inline(always)]
pub(crate) fn unix_local_time_to_string(timestamp: i64) -> String {
    Local
        .timestamp_opt(timestamp, 0)
        .earliest()
        .map(|ts| ts.naive_local().format(TIMESTAMP_FORMAT).to_string())
        .unwrap_or("invalid timestamp".into())
}

#[inline(always)]
pub(crate) fn unix_utc_time_to_string(timestamp: i64) -> String {
    DateTime::from_timestamp(timestamp, 0)
        .map(|ts| ts.format(TIMESTAMP_FORMAT).to_string())
        .unwrap_or("invalid timestamp".into())
}

#[inline(always)]
pub(crate) fn windows_filetime_to_string(timestamp: i64) -> String {
    windows_time_to_datetime(timestamp)
        .map(|ts| ts.format(TIMESTAMP_FORMAT).to_string())
        .unwrap_or("invalid timestamp".into())
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Datelike;

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
        assert_eq!(parse_fat_time(0x0), NaiveTime::from_hms_opt(0, 0, 0));

        assert_eq!(parse_fat_time(0xbf7d), NaiveTime::from_hms_opt(23, 59, 58));

        assert_eq!(parse_fat_time(18301), NaiveTime::from_hms_opt(8, 59, 58));
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
