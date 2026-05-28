use std::{
    io::{self, Read, SeekFrom},
    ops::{Range, Sub},
};

use crate::readers::DataRead;

/// A buffered reader for byte slices that tracks the current read position.
///
/// Wraps any type implementing [`AsRef<[u8]>`] and provides seeking and reading
/// operations while maintaining an internal cursor position.
///
/// See [`BufReader::from_slice`] for construction.
pub struct BufReader<S: AsRef<[u8]>> {
    stream_pos: u64,
    buf: S,
}

impl<S> Read for BufReader<S>
where
    S: AsRef<[u8]>,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let r = self.read_count(buf.len() as u64)?;
        for (i, b) in r.iter().enumerate() {
            buf[i] = *b;
        }
        Ok(r.len())
    }
}

impl<S> DataRead for BufReader<S>
where
    S: AsRef<[u8]>,
{
    #[inline(always)]
    fn stream_position(&self) -> u64 {
        self.stream_pos
    }

    #[inline]
    fn read_range(&mut self, range: Range<u64>) -> io::Result<&[u8]> {
        // we fix range in case we attempt at reading beyond end of file
        let range = if range.end > self.buf.as_ref().len() as u64 {
            range.start..self.buf.as_ref().len() as u64
        } else {
            range
        };

        let range_len = range.end.saturating_sub(range.start);

        self.seek(SeekFrom::Start(range.end))
            .expect("buffer seek should never fail");

        if range.start >= self.buf.as_ref().len() as u64 || range_len == 0 {
            return Ok(&[]);
        }

        Ok(&self.buf.as_ref()[range.start as usize..range.end as usize])
    }

    fn read_until_any_delim_or_limit(
        &mut self,
        delims: &[u8],
        limit: u64,
    ) -> Result<&[u8], io::Error> {
        self._read_while_or_limit(|b| !delims.contains(&b), limit, true)
    }

    fn read_until_or_limit(&mut self, byte: u8, limit: u64) -> Result<&[u8], io::Error> {
        self._read_while_or_limit(|b| b != byte, limit, true)
    }

    fn read_while_or_limit<F>(&mut self, f: F, limit: u64) -> Result<&[u8], io::Error>
    where
        F: Fn(u8) -> bool,
    {
        self._read_while_or_limit(f, limit, false)
    }

    fn read_until_utf16_or_limit(
        &mut self,
        utf16_char: &[u8; 2],
        limit: u64,
    ) -> Result<&[u8], io::Error> {
        let start = self.stream_pos;
        let mut end = 0;

        let Some(buf) = self.buf.as_ref().get(start as usize..) else {
            return Ok(&[]);
        };

        let buf = if buf.len().is_multiple_of(2) {
            buf
        } else if buf.len() > 1 {
            &buf[..buf.len().sub(1)]
        } else {
            return Ok(&[]);
        };

        let even = buf
            .iter()
            .enumerate()
            .filter(|(i, _)| i.is_multiple_of(2))
            .map(|t| t.1);

        let odd = buf
            .iter()
            .enumerate()
            .filter(|(i, _)| !i.is_multiple_of(2))
            .map(|t| t.1);

        for t in even.zip(odd) {
            if limit.saturating_sub(end) == 0 {
                break;
            }

            end += 2;

            // tail check
            if t.0 == &utf16_char[0] && t.1 == &utf16_char[1] {
                // we include char
                break;
            }
        }

        self.read_exact_range(start..start + end)
    }

    fn data_size(&self) -> u64 {
        self.buf.as_ref().len() as u64
    }

    #[inline(always)]
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.stream_pos = self.offset_from_start(pos);
        Ok(self.stream_pos)
    }
}

impl<S> BufReader<S>
where
    S: AsRef<[u8]>,
{
    /// Creates a new `BufReader` wrapping the provided byte slice.
    ///
    /// The reader's position is initialized to `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pure_magic::readers::{BufReader, DataRead};
    ///
    /// let reader = BufReader::from_slice(b"hello world");
    /// assert_eq!(reader.stream_position(), 0);
    /// ```
    pub fn from_slice(s: S) -> Self {
        Self {
            stream_pos: 0,
            buf: s,
        }
    }

    // reads while f returns true or we reach limit
    #[inline(always)]
    fn _read_while_or_limit<F>(
        &mut self,
        f: F,
        limit: u64,
        include_last: bool,
    ) -> Result<&[u8], io::Error>
    where
        F: Fn(u8) -> bool,
    {
        let start = self.stream_pos;
        let mut end = 0;

        let Some(buf) = self.buf.as_ref().get(start as usize..) else {
            return Ok(&[]);
        };

        for b in buf {
            if limit - end == 0 {
                break;
            }

            if !f(*b) {
                if include_last && end < self.data_size() {
                    end += 1;
                }
                break;
            }

            end += 1;
        }

        self.read_exact_range(start..start + end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === from_slice ===

    #[test]
    fn test_from_slice() {
        let buf = b"hello world";
        let r = BufReader::from_slice(buf);
        assert_eq!(r.stream_position(), 0);
        assert_eq!(r.data_size(), buf.len() as u64);
    }

    #[test]
    fn test_from_slice_empty() {
        let r = BufReader::from_slice(b"");
        assert_eq!(r.stream_position(), 0);
        assert_eq!(r.data_size(), 0);
    }

    // === Seek impl ===

    #[test]
    fn test_seek_start() {
        let mut r = BufReader::from_slice(b"hello world");
        assert_eq!(r.seek(SeekFrom::Start(5)).unwrap(), 5);
        assert_eq!(r.stream_position(), 5);
    }

    #[test]
    fn test_seek_start_zero() {
        let mut r = BufReader::from_slice(b"hello");
        r.seek(SeekFrom::Start(3)).unwrap();
        assert_eq!(r.seek(SeekFrom::Start(0)).unwrap(), 0);
        assert_eq!(r.stream_position(), 0);
    }

    #[test]
    fn test_seek_current() {
        let mut r = BufReader::from_slice(b"hello world");
        r.seek(SeekFrom::Start(5)).unwrap();
        assert_eq!(r.seek(SeekFrom::Current(2)).unwrap(), 7);
        assert_eq!(r.stream_position(), 7);
    }

    #[test]
    fn test_seek_current_negative() {
        let mut r = BufReader::from_slice(b"hello world");
        r.seek(SeekFrom::Start(5)).unwrap();
        assert_eq!(r.seek(SeekFrom::Current(-3)).unwrap(), 2);
        assert_eq!(r.stream_position(), 2);
    }

    #[test]
    fn test_seek_end() {
        let mut r = BufReader::from_slice(b"hello world");
        assert_eq!(r.seek(SeekFrom::End(0)).unwrap(), 11);
        assert_eq!(r.stream_position(), 11);
    }

    #[test]
    fn test_seek_end_negative() {
        let mut r = BufReader::from_slice(b"hello world");
        assert_eq!(r.seek(SeekFrom::End(-5)).unwrap(), 6);
        assert_eq!(r.stream_position(), 6);
    }

    // === offset_from_start ===

    #[test]
    fn test_offset_from_start_start() {
        let r = BufReader::from_slice(b"hello");
        assert_eq!(r.offset_from_start(SeekFrom::Start(3)), 3);
    }

    #[test]
    fn test_offset_from_start_current() {
        let mut r = BufReader::from_slice(b"hello");
        r.stream_pos = 5;
        assert_eq!(r.offset_from_start(SeekFrom::Current(3)), 8);
        assert_eq!(r.offset_from_start(SeekFrom::Current(-2)), 3);
    }

    #[test]
    fn test_offset_from_start_end() {
        let r = BufReader::from_slice(b"hello");
        assert_eq!(r.offset_from_start(SeekFrom::End(0)), 5);
        assert_eq!(r.offset_from_start(SeekFrom::End(-2)), 3);
    }

    // === stream_position ===

    #[test]
    fn test_stream_position() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(r.stream_position(), 0);
        r.stream_pos = 3;
        assert_eq!(r.stream_position(), 3);
    }

    // === data_size ===

    #[test]
    fn test_data_size() {
        let r = BufReader::from_slice(b"hello world");
        assert_eq!(r.data_size(), 11);
    }

    #[test]
    fn test_data_size_empty() {
        let r = BufReader::from_slice(b"");
        assert_eq!(r.data_size(), 0);
    }

    // === read_range ===

    #[test]
    fn test_read_range_full() {
        let mut r = BufReader::from_slice(b"hello world");
        assert_eq!(r.read_range(0..11).unwrap(), b"hello world");
        assert_eq!(r.stream_position(), 11);
    }

    #[test]
    fn test_read_range_partial() {
        let mut r = BufReader::from_slice(b"hello world");
        assert_eq!(r.read_range(0..5).unwrap(), b"hello");
        assert_eq!(r.stream_position(), 5);
        assert_eq!(r.read_range(6..11).unwrap(), b"world");
        assert_eq!(r.stream_position(), 11);
    }

    #[test]
    fn test_read_range_beyond_end() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(r.read_range(0..100).unwrap(), b"hello");
        assert_eq!(r.stream_position(), 5);
    }

    #[test]
    fn test_read_range_start_beyond() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(r.read_range(100..200).unwrap(), b"");
        assert_eq!(r.stream_position(), 5);
    }

    #[test]
    fn test_read_range_empty() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(r.read_range(3..3).unwrap(), b"");
        assert_eq!(r.stream_position(), 3);
    }

    #[test]
    fn test_read_range_empty_slice() {
        let mut r = BufReader::from_slice(b"");
        assert_eq!(r.read_range(0..0).unwrap(), b"");
        assert_eq!(r.stream_position(), 0);
    }

    // === read_count ===

    #[test]
    fn test_read_count_all() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(r.read_count(5).unwrap(), b"hello");
        assert_eq!(r.stream_position(), 5);
    }

    #[test]
    fn test_read_count_partial() {
        let mut r = BufReader::from_slice(b"hello world");
        assert_eq!(r.read_count(5).unwrap(), b"hello");
        assert_eq!(r.stream_position(), 5);
        assert_eq!(r.read_count(1).unwrap(), b" ");
        assert_eq!(r.stream_position(), 6);
        assert_eq!(r.read_count(5).unwrap(), b"world");
        assert_eq!(r.stream_position(), 11);
    }

    #[test]
    fn test_read_count_beyond_end() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(r.read_count(100).unwrap(), b"hello");
        assert_eq!(r.stream_position(), 5);
    }

    #[test]
    fn test_read_count_zero() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(r.read_count(0).unwrap(), b"");
        assert_eq!(r.stream_position(), 0);
    }

    #[test]
    fn test_read_count_zero_at_middle() {
        let mut r = BufReader::from_slice(b"hello");
        r.seek(SeekFrom::Start(3)).unwrap();
        assert_eq!(r.read_count(0).unwrap(), b"");
        assert_eq!(r.stream_position(), 3);
    }

    // === read_exact_range ===

    #[test]
    fn test_read_exact_range_success() {
        let mut r = BufReader::from_slice(b"hello world");
        assert_eq!(r.read_exact_range(0..5).unwrap(), b"hello");
        assert_eq!(r.stream_position(), 5);
    }

    #[test]
    fn test_read_exact_range_beyond_end() {
        let mut r = BufReader::from_slice(b"hello");
        assert!(r.read_exact_range(0..100).is_err());
    }

    #[test]
    fn test_read_exact_range_start_beyond() {
        let mut r = BufReader::from_slice(b"hello");
        assert!(r.read_exact_range(10..20).is_err());
    }

    #[test]
    fn test_read_exact_range_zero_length() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(r.read_exact_range(3..3).unwrap(), b"");
        assert_eq!(r.stream_position(), 3);
    }

    // === read_exact_count ===

    #[test]
    fn test_read_exact_count_success() {
        let mut r = BufReader::from_slice(b"hello world");
        assert_eq!(r.read_exact_count(5).unwrap(), b"hello");
        assert_eq!(r.stream_position(), 5);
    }

    #[test]
    fn test_read_exact_count_beyond_end() {
        let mut r = BufReader::from_slice(b"hello");
        assert!(r.read_exact_count(100).is_err());
    }

    #[test]
    fn test_read_exact_count_exact_length() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(r.read_exact_count(5).unwrap(), b"hello");
        assert_eq!(r.stream_position(), 5);
    }

    #[test]
    fn test_read_exact_count_zero() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(r.read_exact_count(0).unwrap(), b"");
        assert_eq!(r.stream_position(), 0);
    }

    // === read_exact_into ===

    #[test]
    fn test_read_exact_into_success() {
        let mut r = BufReader::from_slice(b"hello world");
        let mut buf = [0u8; 5];
        r.read_exact_into(&mut buf).unwrap();
        assert_eq!(&buf, b"hello");
        assert_eq!(r.stream_position(), 5);
    }

    #[test]
    fn test_read_exact_into_full() {
        let mut r = BufReader::from_slice(b"hello");
        let mut buf = [0u8; 5];
        r.read_exact_into(&mut buf).unwrap();
        assert_eq!(&buf, b"hello");
        assert_eq!(r.stream_position(), 5);
    }

    #[test]
    fn test_read_exact_into_error_too_large() {
        let mut r = BufReader::from_slice(b"hello");
        let mut buf = [0u8; 100];
        assert!(r.read_exact_into(&mut buf).is_err());
    }

    #[test]
    fn test_read_exact_into_empty() {
        let mut r = BufReader::from_slice(b"hello");
        let mut buf = [0u8; 0];
        r.read_exact_into(&mut buf).unwrap();
        assert_eq!(r.stream_position(), 0);
    }

    // === read_until_or_limit ===

    #[test]
    fn test_read_until_or_limit_found() {
        let mut r = BufReader::from_slice(b"hello world");
        assert_eq!(r.read_until_or_limit(b' ', 100).unwrap(), b"hello ");
        assert_eq!(r.stream_position(), 6);
    }

    #[test]
    fn test_read_until_or_limit_not_found() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(r.read_until_or_limit(b'x', 100).unwrap(), b"hello");
        assert_eq!(r.stream_position(), 5);
    }

    #[test]
    fn test_read_until_or_limit_with_limit() {
        let mut r = BufReader::from_slice(b"hello world");
        assert_eq!(r.read_until_or_limit(b' ', 3).unwrap(), b"hel");
        assert_eq!(r.stream_position(), 3);
    }

    #[test]
    fn test_read_until_or_limit_limit_zero() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(r.read_until_or_limit(b' ', 0).unwrap(), b"");
        assert_eq!(r.stream_position(), 0);
    }

    #[test]
    fn test_read_until_or_limit_at_start() {
        let mut r = BufReader::from_slice(b" world");
        assert_eq!(r.read_until_or_limit(b' ', 100).unwrap(), b" ");
        assert_eq!(r.stream_position(), 1);
    }

    // === read_until_any_delim_or_limit ===

    #[test]
    fn test_read_until_any_delim_or_limit_found() {
        let mut r = BufReader::from_slice(b"hello,world;test");
        assert_eq!(
            r.read_until_any_delim_or_limit(b",;", 100).unwrap(),
            b"hello,"
        );
        assert_eq!(r.stream_position(), 6);
    }

    #[test]
    fn test_read_until_any_delim_or_limit_not_found() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(
            r.read_until_any_delim_or_limit(b",;", 100).unwrap(),
            b"hello"
        );
        assert_eq!(r.stream_position(), 5);
    }

    #[test]
    fn test_read_until_any_delim_or_limit_multiple_delims() {
        let mut r = BufReader::from_slice(b"hello;world,test");
        assert_eq!(
            r.read_until_any_delim_or_limit(b",;", 100).unwrap(),
            b"hello;"
        );
        assert_eq!(r.stream_position(), 6);
    }

    #[test]
    fn test_read_until_any_delim_or_limit_with_limit() {
        let mut r = BufReader::from_slice(b"hello,world");
        assert_eq!(r.read_until_any_delim_or_limit(b",", 3).unwrap(), b"hel");
        assert_eq!(r.stream_position(), 3);
    }

    #[test]
    fn test_read_until_any_delim_or_limit_empty_delims() {
        let mut r = BufReader::from_slice(b"hello");
        // Empty delims means no delimiter matches, so reads until limit
        assert_eq!(r.read_until_any_delim_or_limit(b"", 100).unwrap(), b"hello");
        assert_eq!(r.stream_position(), 5);
    }

    // === read_while_or_limit ===

    #[test]
    fn test_read_while_or_limit_stream_pos_past_end() {
        // stream_pos > buf.len() previously cause an OOB panic via unchecked slice indexing
        let mut r = BufReader::from_slice(b"hello");
        r.stream_pos = 10; // past end
        assert_eq!(r.read_while_or_limit(|_| true, 100).unwrap(), b"");
    }

    #[test]
    fn test_read_while_or_limit_all_match() {
        let mut r = BufReader::from_slice(b"hello world");
        assert_eq!(r.read_while_or_limit(|b| b != b' ', 100).unwrap(), b"hello");
        assert_eq!(r.stream_position(), 5);
    }

    #[test]
    fn test_read_while_or_limit_stop_at_delim() {
        let mut r = BufReader::from_slice(b"hello world");
        assert_eq!(r.read_while_or_limit(|b| b != b' ', 100).unwrap(), b"hello");
        assert_eq!(r.stream_position(), 5);
    }

    #[test]
    fn test_read_while_or_limit_with_limit() {
        let mut r = BufReader::from_slice(b"hello world");
        assert_eq!(r.read_while_or_limit(|b| b != b' ', 3).unwrap(), b"hel");
        assert_eq!(r.stream_position(), 3);
    }

    #[test]
    fn test_read_while_or_limit_limit_zero() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(r.read_while_or_limit(|b| b != b'x', 0).unwrap(), b"");
        assert_eq!(r.stream_position(), 0);
    }

    #[test]
    fn test_read_while_or_limit_all_match_no_delim() {
        let mut r = BufReader::from_slice(b"hello");
        assert_eq!(r.read_while_or_limit(|b| b != b'x', 100).unwrap(), b"hello");
        assert_eq!(r.stream_position(), 5);
    }

    // === read_until_utf16_or_limit ===

    #[test]
    fn test_read_until_utf16_or_limit_found() {
        // UTF-16LE encoded "ab\0\0" (a, b, null terminator)
        let mut r = BufReader::from_slice(b"\x61\x00\x62\x00\x00\x00");
        assert_eq!(
            r.read_until_utf16_or_limit(b"\x00\x00", 100).unwrap(),
            b"\x61\x00\x62\x00\x00\x00"
        );
        assert_eq!(r.stream_position(), 6);
    }

    #[test]
    fn test_read_until_utf16_or_limit_not_found() {
        let mut r = BufReader::from_slice(b"\x61\x00\x62\x00\x63\x00");
        assert_eq!(
            r.read_until_utf16_or_limit(b"\xff\xff", 100).unwrap(),
            b"\x61\x00\x62\x00\x63\x00"
        );
        assert_eq!(r.stream_position(), 6);
    }

    #[test]
    fn test_read_until_utf16_or_limit_with_limit() {
        let mut r = BufReader::from_slice(b"\x61\x00\x62\x00\x63\x00\x00\x00");
        assert_eq!(
            r.read_until_utf16_or_limit(b"\x00\x00", 1).unwrap(),
            b"\x61\x00"
        );
        assert_eq!(r.stream_position(), 2);
    }

    #[test]
    fn test_read_until_utf16_or_limit_odd_length() {
        let mut r = BufReader::from_slice(b"\x61\x00\x62");
        // Odd length: truncates to even, reads all available pairs
        // Input has 3 bytes, truncated to 2 bytes = one utf16 char
        // Delimiter not found, so reads all 2 bytes
        assert_eq!(
            r.read_until_utf16_or_limit(b"\x00\x00", 100).unwrap(),
            b"\x61\x00"
        );
        assert_eq!(r.stream_position(), 2);
    }

    #[test]
    fn test_read_until_utf16_or_limit_stream_pos_past_end() {
        // unchecked [start..] indexing previously panicked when stream_pos > buf.len()
        let mut r = BufReader::from_slice(b"\x61\x00\x62\x00");
        r.stream_pos = 10;
        assert_eq!(r.read_until_utf16_or_limit(b"\x00\x00", 100).unwrap(), b"");
    }

    #[test]
    fn test_read_until_utf16_or_limit_odd_nonzero_start() {
        // (len - 1) was used as an absolute index; when start > 0 it would be
        // less than start, causing a start > end panic
        let mut r = BufReader::from_slice(b"\x61\x00\x62\x00\x63");
        r.stream_pos = 2; // 3 bytes remaining (\x62\x00\x63) — odd, start != 0
        assert_eq!(
            r.read_until_utf16_or_limit(b"\x00\x00", 100).unwrap(),
            b"\x62\x00"
        );
    }

    #[test]
    fn test_read_until_utf16_or_limit_single_byte() {
        let mut r = BufReader::from_slice(b"\x61");
        assert_eq!(r.read_until_utf16_or_limit(b"\x00\x00", 100).unwrap(), b"");
        assert_eq!(r.stream_position(), 0);
    }

    #[test]
    fn test_read_until_utf16_or_limit_empty() {
        let mut r = BufReader::from_slice(b"");
        assert_eq!(r.read_until_utf16_or_limit(b"\x00\x00", 100).unwrap(), b"");
        assert_eq!(r.stream_position(), 0);
    }
}
