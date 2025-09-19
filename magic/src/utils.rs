use chrono::{DateTime, Local, NaiveDate, NaiveTime, TimeZone};

use crate::TIMESTAMP_FORMAT;

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

// test this properly
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

#[inline(always)]
pub(crate) fn unix_local_time_to_string(timestamp: i64) -> String {
    Local
        .timestamp_opt(timestamp, 0)
        .earliest()
        .map(|ts| ts.naive_local().format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or("invalid timestamp".into())
}

#[inline(always)]
pub(crate) fn unix_utc_time_to_string(timestamp: i64) -> String {
    DateTime::from_timestamp(timestamp, 0)
        .map(|ts| ts.format(TIMESTAMP_FORMAT).to_string())
        .unwrap_or("invalid timestamp".into())
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
}
