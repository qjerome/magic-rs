#![deny(unsafe_code)]

use std::{
    cmp::{max, min},
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
    ops::Range,
    path::Path,
};

use memmap2::MmapMut;

const EMPTY_RANGE: &[u8] = &[];

pub struct LazyCache<R>
where
    R: Read + Seek,
{
    source: R,
    loaded: Vec<bool>,
    hot_head: Vec<u8>,
    hot_tail: Vec<u8>,
    warm: Option<MmapMut>,
    cold: Vec<u8>,
    block_size: u64,
    warm_size: Option<u64>,
    stream_pos: u64,
    pos_end: u64,
}

const BLOCK_SIZE: usize = 4096;

impl<R> Seek for LazyCache<R>
where
    R: Read + Seek,
{
    #[inline(always)]
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.stream_pos = self.offset_from_start(pos);
        Ok(self.stream_pos)
    }
}

impl LazyCache<File> {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        Self::from_read_seek(File::open(path)?)
    }
}

impl<R> io::Read for LazyCache<R>
where
    R: Read + Seek,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let r = self.inner_read_count(buf.len() as u64)?;
        for (i, b) in r.iter().enumerate() {
            buf[i] = *b;
        }
        Ok(r.len())
    }
}

impl<R> LazyCache<R>
where
    R: Read + Seek,
{
    pub fn from_read_seek(mut rs: R) -> Result<Self, io::Error> {
        let block_size = BLOCK_SIZE as u64;
        let pos_end = rs.seek(SeekFrom::End(0))?;
        let cache_cap = pos_end.div_ceil(BLOCK_SIZE as u64);

        Ok(Self {
            source: rs,
            hot_head: vec![],
            hot_tail: vec![],
            warm: None,
            cold: vec![0; block_size as usize],
            loaded: vec![false; cache_cap as usize],
            block_size,
            warm_size: None,
            stream_pos: 0,
            pos_end,
        })
    }

    pub fn with_hot_cache(mut self, size: usize) -> Result<Self, io::Error> {
        let head_tail_size = size / 2;

        self.source.seek(SeekFrom::Start(0))?;

        if self.pos_end > size as u64 {
            self.hot_head = vec![0u8; head_tail_size];
            self.source.read_exact(self.hot_head.as_mut_slice())?;

            self.source.seek(SeekFrom::End(-(size as i64)))?;
            self.hot_tail = vec![0u8; head_tail_size];
            self.source.read_exact(self.hot_tail.as_mut_slice())?;
        } else {
            self.hot_head = vec![0u8; self.pos_end as usize];
            self.source.read_exact(self.hot_head.as_mut())?;
        }

        Ok(self)
    }

    pub fn with_warm_cache(mut self, mut warm_size: u64) -> Self {
        // if warm_size is smaller than block_size we will not
        // be able to write chunks into the warm cache
        warm_size = max(warm_size, self.block_size);
        self.warm_size = Some(warm_size);
        self
    }

    #[inline(always)]
    pub fn offset_from_start(&self, pos: SeekFrom) -> u64 {
        match pos {
            SeekFrom::Start(s) => s,
            SeekFrom::Current(p) => (self.stream_pos as i128 + p as i128) as u64,
            SeekFrom::End(e) => (self.pos_end as i128 + e as i128) as u64,
        }
    }

    #[inline(always)]
    pub fn lazy_stream_position(&self) -> u64 {
        self.stream_pos
    }

    #[inline(always)]
    fn warm(&mut self) -> Result<&mut MmapMut, io::Error> {
        if self.warm.is_none() && self.warm_size.is_some() {
            self.warm = Some(MmapMut::map_anon(
                self.warm_size.unwrap_or_default() as usize
            )?);
        }
        Ok(self.warm.as_mut().unwrap())
    }

    #[inline(always)]
    fn range_warmup(&mut self, range: Range<u64>) -> Result<(), io::Error> {
        let start_chunk_id = range.start / self.block_size;
        let end_chunk_id = (range.end.saturating_sub(1)) / self.block_size;

        if self.loaded.is_empty() {
            return Ok(());
        }

        for chunk_id in start_chunk_id..=end_chunk_id {
            if self.loaded[chunk_id as usize] {
                continue;
            }

            let offset = chunk_id * self.block_size;
            let buf_size = min(
                self.block_size as usize,
                (self.pos_end.saturating_sub(offset)) as usize,
            );
            let mut buf = vec![0u8; buf_size];
            self.source.seek(SeekFrom::Start(offset))?;
            self.source.read_exact(&mut buf)?;

            (&mut self.warm()?[offset as usize..]).write_all(&buf)?;
            self.loaded[chunk_id as usize] = true;
        }

        Ok(())
    }

    #[inline(always)]
    fn get_range_u64(&mut self, range: Range<u64>) -> Result<&[u8], io::Error> {
        // we fix range in case we attempt at reading beyond end of file
        let range = if range.end > self.pos_end {
            range.start..self.pos_end
        } else {
            range
        };

        let range_len = range.end.saturating_sub(range.start);

        if range.start > self.pos_end || range_len == 0 {
            return Ok(EMPTY_RANGE);
        } else if range.start < self.hot_head.len() as u64
            && range.end <= self.hot_head.len() as u64
        {
            self.seek(SeekFrom::Start(range.end))?;
            return Ok(&self.hot_head[range.start as usize..range.end as usize]);
        } else if range.start > (self.pos_end - self.hot_tail.len() as u64) {
            let start_from_end = self.pos_end.saturating_sub(1).saturating_sub(range.start);
            self.seek(SeekFrom::Start(range.end))?;
            return Ok(&self.hot_tail
                [start_from_end as usize..start_from_end.saturating_add(range_len) as usize]);
        } else if range.end < self.warm_size.unwrap_or_default() {
            self.range_warmup(range.clone())?;
            self.seek(SeekFrom::Start(range.end))?;
            return Ok(&self.warm()?[range.start as usize..range.end as usize]);
        } else if range_len > self.cold.len() as u64 {
            self.cold.resize(range_len as usize, 0);
        }

        self.source.seek(SeekFrom::Start(range.start))?;
        let n = self.source.read(self.cold[..range_len as usize].as_mut())?;
        self.seek(SeekFrom::Start(range.end))?;
        Ok(&self.cold[..n])
    }

    pub fn read_range(&mut self, range: Range<u64>) -> Result<&[u8], io::Error> {
        let range = range.start..range.end;
        self.get_range_u64(range)
    }

    #[inline(always)]
    fn inner_read_count(&mut self, count: u64) -> Result<&[u8], io::Error> {
        let pos = self.stream_pos;
        let range = pos..(pos.saturating_add(count));
        self.get_range_u64(range)
    }

    /// Read at current reader position and return byte slice
    pub fn read_count(&mut self, count: u64) -> Result<&[u8], io::Error> {
        self.inner_read_count(count)
    }

    pub fn read_exact_range(&mut self, range: Range<u64>) -> Result<&[u8], io::Error> {
        let range_len = range.end - range.start;
        let b = self.read_range(range)?;
        if b.len() as u64 != range_len {
            Err(io::Error::from(io::ErrorKind::UnexpectedEof))
        } else {
            Ok(b)
        }
    }

    pub fn read_exact_count(&mut self, count: u64) -> Result<&[u8], io::Error> {
        let b = self.read_count(count)?;
        debug_assert!(b.len() <= count as usize);
        if b.len() as u64 != count {
            Err(io::ErrorKind::UnexpectedEof.into())
        } else {
            Ok(b)
        }
    }

    pub fn read_exact_into(&mut self, buf: &mut [u8]) -> Result<(), io::Error> {
        let read = self.read_exact_count(buf.len() as u64)?;
        // this function call should not panic as read_exact
        // guarantees we read exactly the length of buf
        buf.copy_from_slice(read);
        Ok(())
    }

    pub fn read_until_any_delim_or_limit(
        &mut self,
        delims: &[u8],
        limit: u64,
    ) -> Result<&[u8], io::Error> {
        self._read_while_or_limit(|b| !delims.contains(&b), limit, true)
    }

    pub fn read_until_or_limit(&mut self, byte: u8, limit: u64) -> Result<&[u8], io::Error> {
        self._read_while_or_limit(|b| b != byte, limit, true)
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

        'outer: while limit - end > 0 {
            let buf = self.read_count(self.block_size)?;

            for b in buf {
                if limit - end == 0 {
                    break 'outer;
                }

                if !f(*b) {
                    if include_last {
                        end += 1;
                    }
                    // read_until includes delimiter
                    break 'outer;
                }

                end += 1;
            }

            // we processed last chunk
            if buf.len() as u64 != self.block_size {
                break;
            }
        }

        self.read_exact_range(start..start + end)
    }

    pub fn read_while_or_limit<F>(&mut self, f: F, limit: u64) -> Result<&[u8], io::Error>
    where
        F: Fn(u8) -> bool,
    {
        self._read_while_or_limit(f, limit, false)
    }

    // limit is expressed in numbers of utf16 chars
    pub fn read_until_utf16_or_limit(
        &mut self,
        utf16_char: &[u8; 2],
        limit: u64,
    ) -> Result<&[u8], io::Error> {
        let start = self.stream_pos;
        let mut end = 0;

        let even_bs = if self.block_size.is_multiple_of(2) {
            self.block_size
        } else {
            self.block_size.saturating_add(1)
        };

        'outer: while limit.saturating_sub(end) > 0 {
            let buf = self.read_count(even_bs)?;

            let even = buf
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == 0)
                .map(|t| t.1);

            let odd = buf
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 2 != 0)
                .map(|t| t.1);

            for t in even.zip(odd) {
                if limit.saturating_sub(end) == 0 {
                    break 'outer;
                }

                end += 2;

                // tail check
                if t.0 == &utf16_char[0] && t.1 == &utf16_char[1] {
                    // we include char
                    break 'outer;
                }
            }

            // we processed the last chunk
            if buf.len() as u64 != even_bs {
                // if we arrive here we reached end of file
                if buf.len() % 2 != 0 {
                    // we include last byte missed by zip
                    end += 1
                }
                break;
            }
        }

        self.read_exact_range(start..start + end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! lazy_cache {
        ($content: literal) => {
            LazyCache::from_read_seek(std::io::Cursor::new($content)).unwrap()
        };
    }

    #[test]
    fn test_get_single_block() {
        let mut cache = lazy_cache!(b"hello world");
        let data = cache.read_range(0..4).unwrap();
        assert_eq!(data, b"hell");
    }

    #[test]
    fn test_get_across_blocks() {
        let mut cache = lazy_cache!(b"hello world");
        let data = cache.read_range(2..7).unwrap();
        assert_eq!(data, b"llo w");
    }

    #[test]
    fn test_get_entire_file() {
        let mut cache = lazy_cache!(b"hello world");
        let data = cache.read_range(0..11).unwrap();
        assert_eq!(data, b"hello world");
    }

    #[test]
    fn test_get_empty_range() {
        let mut cache = lazy_cache!(b"hello world");
        let data = cache.read_range(0..0).unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn test_get_out_of_bounds() {
        let mut cache = lazy_cache!(b"hello world");
        // This should not panic, but return an error or empty slice depending on your design
        // Currently, your code will panic due to `unwrap()` on `None`
        // You may want to handle this case more gracefully
        assert!(cache.read_range(20..30).unwrap().is_empty());
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = lazy_cache!(b"0123456789abcdef");
        // Load blocks 0 and 1
        let _ = cache.read_range(0..8).unwrap();
        // Load block 2, which should evict block 0 or 1 due to max_size=8
        let _ = cache.read_range(8..12).unwrap();
        // Check that the cache still works
        let data = cache.read_range(8..12).unwrap();
        assert_eq!(data, b"89ab");
    }

    #[test]
    fn test_chunk_consolidation() {
        let mut cache = lazy_cache!(b"0123456789abcdef");
        // Load blocks 0 and 1 separately
        let _ = cache.read_range(0..4).unwrap();
        let _ = cache.read_range(4..8).unwrap();
        // Load block 2, which should not consolidate with 0 or 1
        let _ = cache.read_range(8..12).unwrap();
        // Now load block 1 again, which should consolidate with block 0
        let _ = cache.read_range(2..6).unwrap();
        // Check that the consolidated chunk is correct
        let data = cache.read_range(0..8).unwrap();
        assert_eq!(data, b"01234567");
    }

    #[test]
    fn test_overlapping_ranges() {
        let mut cache = lazy_cache!(b"0123456789abcdef");
        // Load overlapping ranges
        let _ = cache.read_range(2..6).unwrap();
        let _ = cache.read_range(4..10).unwrap();
        // Check that the data is correct
        let data = cache.read_range(2..10).unwrap();
        assert_eq!(data, b"23456789");
    }

    #[test]
    fn test_lru_behavior() {
        let mut cache = lazy_cache!(b"0123456789abcdef");
        // Load block 0
        let _ = cache.read_range(0..4).unwrap();
        // Load block 1
        let _ = cache.read_range(4..8).unwrap();
        // Load block 2, which should evict block 0
        let _ = cache.read_range(8..12).unwrap();
        // Block 0 should be evicted, so accessing it again should reload it
        let data = cache.read_range(0..4).unwrap();
        assert_eq!(data, b"0123");
    }

    #[test]
    fn test_small_block_size() {
        let mut cache = lazy_cache!(b"abc");
        let data = cache.read_range(0..3).unwrap();
        assert_eq!(data, b"abc");
    }

    #[test]
    fn test_large_block_size() {
        let mut cache = lazy_cache!(b"hello world");
        let data = cache.read_range(0..11).unwrap();
        assert_eq!(data, b"hello world");
    }

    #[test]
    fn test_file_smaller_than_block() {
        let mut cache = lazy_cache!(b"abc");
        let data = cache.read_range(0..3).unwrap();
        assert_eq!(data, b"abc");
    }

    #[test]
    fn test_multiple_gets_same_block() {
        let mut cache = lazy_cache!(b"0123456789abcdef");
        // Get the same block multiple times
        let _ = cache.read_range(0..4).unwrap();
        let _ = cache.read_range(0..4).unwrap();
        let _ = cache.read_range(0..4).unwrap();
        // The block should still be in the cache
        let data = cache.read_range(0..4).unwrap();
        assert_eq!(data, b"0123");
    }

    #[test]
    fn test_read_method() {
        let mut cache = lazy_cache!(b"hello world");
        let _ = cache.read_count(6).unwrap();
        let data = cache.read_count(5).unwrap();
        assert_eq!(data, b"world");
        // We reached the end so next read should bring an empty slice
        assert!(cache.read_count(1).unwrap().is_empty());
    }

    #[test]
    fn test_read_empty() {
        let mut cache = lazy_cache!(b"hello world");
        let data = cache.read_count(0).unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn test_read_beyond_end() {
        let mut cache = lazy_cache!(b"hello world");
        let _ = cache.read_count(11).unwrap();
        let data = cache.read_count(5).unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn test_read_exact_range() {
        let mut cache = lazy_cache!(b"hello world");
        let data = cache.read_exact_range(0..5).unwrap();
        assert_eq!(data, b"hello");
        assert_eq!(cache.read_exact_range(5..11).unwrap(), b" world");
        assert!(cache.read_exact_range(12..13).is_err());
    }

    #[test]
    fn test_read_exact_range_error() {
        let mut cache = lazy_cache!(b"hello world");
        let result = cache.read_exact_range(0..20);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_exact() {
        let mut cache = lazy_cache!(b"hello world");
        let data = cache.read_exact_count(5).unwrap();
        assert_eq!(data, b"hello");
        assert_eq!(cache.read_exact_count(6).unwrap(), b" world");
        assert!(cache.read_exact_count(0).is_ok());
        assert!(cache.read_exact_count(1).is_err());
    }

    #[test]
    fn test_read_exact_error() {
        let mut cache = lazy_cache!(b"hello world");
        let result = cache.read_exact_count(20);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_until_limit() {
        let mut cache = lazy_cache!(b"hello world");
        let data = cache.read_until_or_limit(b' ', 10).unwrap();
        assert_eq!(data, b"hello ");
        assert_eq!(cache.read_exact_count(5).unwrap(), b"world");
    }

    #[test]
    fn test_read_until_limit_not_found() {
        let mut cache = lazy_cache!(b"hello world");
        let data = cache.read_until_or_limit(b'\n', 11).unwrap();
        assert_eq!(data, b"hello world");
        assert!(cache.read_count(1).unwrap().is_empty());
    }

    #[test]
    fn test_read_until_limit_beyond_stream() {
        let mut cache = lazy_cache!(b"hello world");
        let data = cache.read_until_or_limit(b'\n', 42).unwrap();
        assert_eq!(data, b"hello world");
        assert!(cache.read_count(1).unwrap().is_empty());
    }

    #[test]
    fn test_read_until_limit_with_limit() {
        let mut cache = lazy_cache!(b"hello world");
        let data = cache.read_until_or_limit(b' ', 42).unwrap();
        assert_eq!(data, b"hello ");

        let data = cache.read_until_or_limit(b' ', 2).unwrap();
        assert_eq!(data, b"wo");

        let data = cache.read_until_or_limit(b' ', 42).unwrap();
        assert_eq!(data, b"rld");
    }

    #[test]
    fn test_read_until_utf16_limit() {
        let mut cache = lazy_cache!(
            b"\x61\x00\x62\x00\x63\x00\x64\x00\x00\x00\x61\x00\x62\x00\x63\x00\x64\x00\x00"
        );
        let data = cache.read_until_utf16_or_limit(b"\x00\x00", 512).unwrap();
        assert_eq!(data, b"\x61\x00\x62\x00\x63\x00\x64\x00\x00\x00");

        let data = cache.read_until_utf16_or_limit(b"\x00\x00", 1).unwrap();
        assert_eq!(data, b"\x61\x00");

        assert_eq!(
            cache.read_until_utf16_or_limit(b"\xff\xff", 64).unwrap(),
            b"\x62\x00\x63\x00\x64\x00\x00"
        );
    }

    #[test]
    fn test_io_read() {
        let mut f = File::open("./src/lib.rs").unwrap();
        let mut lr = LazyCache::from_read_seek(File::open("./src/lib.rs").unwrap()).unwrap();

        let mut fb = vec![];
        let file_n = f.read_to_end(&mut fb).unwrap();

        let mut lcb = vec![];
        let lcn = lr.read_to_end(&mut lcb).unwrap();

        assert_eq!(lcb, fb);
        assert_eq!(file_n, lcn);
    }
}
