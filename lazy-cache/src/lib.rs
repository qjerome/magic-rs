#![deny(unsafe_code)]

use std::{
    cmp::min,
    collections::HashMap,
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    ops::Range,
    path::Path,
};

use lru_st::collections::LruHashMap;

pub struct LazyCache<R>
where
    R: Read + Seek,
{
    source: R,
    chunks_lru: LruHashMap<u64, Vec<u8>>,
    chunks_map: HashMap<u64, u64>, // maps the chunks id to a larger chunk in LRU
    block_size: u64,
    size: u64,
    max_size: u64,
    stream_pos: u64,
    pos_end: u64,
}

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
    pub fn open<P: AsRef<Path>>(
        path: P,
        block_size: u64,
        max_size: u64,
    ) -> Result<Self, io::Error> {
        Self::from_read_seek(File::open(path)?, block_size, max_size)
    }
}

impl<R> LazyCache<R>
where
    R: Read + Seek,
{
    pub fn from_read_seek(mut rs: R, block_size: u64, max_size: u64) -> Result<Self, io::Error> {
        let cap = max_size / block_size;
        let pos_end = rs.seek(SeekFrom::End(0))?;

        Ok(Self {
            source: rs,
            chunks_lru: LruHashMap::with_max_entries(cap as usize),
            chunks_map: HashMap::with_capacity(cap as usize),
            block_size,
            size: 0,
            max_size,
            stream_pos: 0,
            pos_end,
        })
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
    fn cleanup_lru_item(&mut self, chunk_id: u64, data: Vec<u8>) {
        let end = chunk_id + data.len() as u64 / self.block_size;
        // in case data.len() == 0 the range is empty yet we
        // need to remove the chunk_id from the map
        self.chunks_map.remove(&chunk_id);

        for cid in chunk_id..end {
            self.chunks_map.remove(&cid);
        }
        self.size -= data.len() as u64;
    }

    #[inline(always)]
    fn fix_chunk_id_map(&mut self, chunk_id: u64) {
        let chunk_len = self
            .chunks_lru
            .get(&chunk_id)
            .map(|c| c.len() as u64)
            .unwrap();
        // last chunk isn't guaranteed to be a multiple of block_size
        // so we must ceil the division otherwise we might miss the last chunk
        let end = chunk_id + chunk_len.div_ceil(self.block_size);
        for i in chunk_id..end {
            self.chunks_map.insert(i, chunk_id);
        }
    }

    #[inline(always)]
    fn load_range_if_needed(&mut self, range: Range<u64>) -> Result<(), io::Error> {
        let range_len = range.end.saturating_sub(range.start);
        if range_len > self.max_size {
            self.max_size = range_len + self.block_size;
        }

        if range.is_empty() {
            // we make sure we have a chunk initialized to empty
            let chunk_id = range.start / self.block_size;

            if !self.chunks_map.contains_key(&chunk_id) {
                if let Some((chunk_id, data)) = self.chunks_lru.insert(chunk_id, vec![]) {
                    self.cleanup_lru_item(chunk_id, data);
                }
                self.chunks_map.insert(chunk_id, chunk_id);
            }
            return Ok(());
        }

        let start_chunk_id = range.start / self.block_size;
        let end_chunk_id = (range.end - 1) / self.block_size;

        for chunk_id in start_chunk_id..=end_chunk_id {
            if self.chunks_map.contains_key(&chunk_id) {
                continue;
            }

            let offset = chunk_id * self.block_size;
            // in case we reach the last chunk
            let buf_size = min(
                self.block_size as usize,
                (self.pos_end.saturating_sub(offset)) as usize,
            );
            let mut buf = vec![0u8; buf_size];

            // we make room in the cache if necessary
            while self.size + buf.len() as u64 > self.max_size {
                if let Some((chunk_id, data)) = self.chunks_lru.pop_lru() {
                    self.cleanup_lru_item(chunk_id, data);
                }
            }

            self.source.seek(SeekFrom::Start(offset))?;
            self.source.read_exact(&mut buf)?;

            self.size += buf.len() as u64;

            if chunk_id > 0 {
                if let Some(real_chunk_id) = self.chunks_map.get(&(chunk_id - 1)).copied() {
                    // this cannot panic as implementation guarantees that any chunk_id in
                    // chunk_map is valid in chunks_lru
                    let chunk = self.chunks_lru.get_mut(&real_chunk_id).unwrap();
                    chunk.extend(buf);

                    // we must consolidate with next chunk
                    if let Some(next_chunk_key) = self.chunks_map.remove(&(chunk_id + 1)) {
                        // we remove next_chunk from lru
                        if let Some(next_chunk) = self.chunks_lru.remove(&next_chunk_key) {
                            // we must call get_mut again to satisfy borrow checker
                            // we can safely unwrap as implementation guarantee it exists
                            let chunk = self.chunks_lru.get_mut(&real_chunk_id).unwrap();
                            // we extend previous chunk with it
                            chunk.extend(next_chunk);
                        }
                    }

                    // we must fix chunk_map to point to the new chunk

                    self.fix_chunk_id_map(real_chunk_id);

                    continue;
                }
            }

            if let Some(real_chunk_id) = self.chunks_map.remove(&(chunk_id + 1)) {
                if let Some(next_chunk) = self.chunks_lru.remove(&real_chunk_id) {
                    buf.extend(next_chunk);
                }
                // we trigger a cleanup in case LRU insertion evicted and existing entry
                if let Some((chunk_id, data)) = self.chunks_lru.insert(chunk_id, buf) {
                    self.cleanup_lru_item(chunk_id, data);
                }
                self.fix_chunk_id_map(chunk_id);
            } else {
                // we trigger a cleanup in case LRU insertion evicted and existing entry
                if let Some((chunk_id, data)) = self.chunks_lru.insert(chunk_id, buf) {
                    self.cleanup_lru_item(chunk_id, data);
                }
                self.chunks_map.insert(chunk_id, chunk_id);
            }
        }

        #[cfg(debug_assertions)]
        self.is_valid_or_panic();

        Ok(())
    }

    fn is_valid_or_panic(&mut self) {
        // control size
        let cnt_size: usize = self.chunks_lru.values().map(|v| v.len()).sum();
        assert_eq!(cnt_size as u64, self.size);
        assert!(self.size <= self.max_size);

        // control that every chunk in map is available in lru
        for (first_chunk_id, real_chunk_id) in self.chunks_map.iter() {
            if !self.chunks_lru.contains_key(real_chunk_id) {
                panic!("found inconsistency {first_chunk_id} {real_chunk_id} not in lru");
            }
        }
    }

    #[inline(always)]
    fn get_range_u64(&mut self, range: Range<u64>) -> Result<&[u8], io::Error> {
        let range_len = range.end - range.start;
        // we fix range in case we attempt at reading beyond end of file
        let range = if range.end > self.pos_end {
            range.start..self.pos_end
        } else {
            range
        };

        self.load_range_if_needed(range.clone())?;
        self.seek(SeekFrom::Start(range.end))?;

        let start = range.start;

        let chunk_id = start / self.block_size;
        // chunk_id is guaranteed to be in chunks_map
        let real_chunk_id = self.chunks_map.get(&chunk_id).unwrap();

        // real_chunk_id is guaranteed to be in chunks_lru
        let chunk = self.chunks_lru.get(real_chunk_id).unwrap();
        // compute start relative to the slice
        let slice_start = (start - (real_chunk_id * self.block_size)) as usize;
        let slice_end = min(slice_start + range_len as usize, chunk.len());

        if slice_start >= chunk.len() {
            Ok(&chunk[0..0])
        } else {
            Ok(&chunk[slice_start..slice_end])
        }
    }

    pub fn read_range(&mut self, range: Range<u64>) -> Result<&[u8], io::Error> {
        let range = range.start..range.end;
        self.get_range_u64(range)
    }

    /// Read at current reader position and return byte slice
    pub fn read(&mut self, count: u64) -> Result<&[u8], io::Error> {
        let pos = self.stream_pos;
        let range = pos..(pos + count);
        self.get_range_u64(range)
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

    pub fn read_exact(&mut self, count: u64) -> Result<&[u8], io::Error> {
        let b = self.read(count)?;
        debug_assert!(b.len() <= count as usize);
        if b.len() as u64 != count {
            Err(io::ErrorKind::UnexpectedEof.into())
        } else {
            Ok(b)
        }
    }

    pub fn read_exact_into(&mut self, buf: &mut [u8]) -> Result<(), io::Error> {
        let read = self.read_exact(buf.len() as u64)?;
        // this function call should not panic as read_exact
        // guarantees we read exactly the length of buf
        buf.copy_from_slice(read);
        Ok(())
    }

    pub fn read_until_or_limit(&mut self, byte: u8, limit: u64) -> Result<&[u8], io::Error> {
        let limit = min(self.max_size, limit);
        let start = self.stream_pos;
        let mut end = 0;

        'outer: while limit - end > 0 {
            let buf = self.read(self.block_size)?;

            for b in buf {
                if limit - end == 0 {
                    break 'outer;
                }

                end += 1;

                if *b == byte {
                    // read_until includes delimiter
                    break 'outer;
                }
            }

            // we processed last chunk
            if buf.len() as u64 != self.block_size {
                break;
            }
        }

        self.read_exact_range(start..start + end)
    }

    // limit is expressed in numbers of utf16 chars
    pub fn read_until_utf16_or_limit(
        &mut self,
        utf16_char: &[u8; 2],
        limit: u64,
    ) -> Result<&[u8], io::Error> {
        let limit = min(self.max_size, limit.saturating_mul(2));
        let start = self.stream_pos;
        let mut end = 0;

        let even_bs = if self.block_size % 2 == 0 {
            self.block_size
        } else {
            self.block_size.saturating_add(1)
        };

        'outer: while limit.saturating_sub(end) > 0 {
            let buf = self.read(even_bs)?;

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
    use std::io::{Seek, SeekFrom, Write};
    use tempfile::NamedTempFile;

    // Helper function to create a test file with known content
    fn create_test_file(content: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file
    }

    fn verify<R: Read + Seek>(lf: &LazyCache<R>) {
        let cnt_size: usize = lf.chunks_lru.values().map(|v| v.len()).sum();
        assert_eq!(cnt_size as u64, lf.size);
        assert!(lf.size <= lf.max_size);
    }

    #[test]
    fn test_open_file() {
        let file = create_test_file(b"hello world");
        let cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        assert_eq!(cache.block_size, 4);
        assert_eq!(cache.max_size, 1024);
        verify(&cache);
    }

    #[test]
    fn test_get_single_block() {
        let file = create_test_file(b"hello world");
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let data = cache.read_range(0..4).unwrap();
        assert_eq!(data, b"hell");
        verify(&cache);
    }

    #[test]
    fn test_get_across_blocks() {
        let file = create_test_file(b"hello world");
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let data = cache.read_range(2..7).unwrap();
        assert_eq!(data, b"llo w");
        verify(&cache);
    }

    #[test]
    fn test_get_entire_file() {
        let file = create_test_file(b"hello world");
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let data = cache.read_range(0..11).unwrap();
        assert_eq!(data, b"hello world");
        verify(&cache);
    }

    #[test]
    fn test_get_empty_range() {
        let file = create_test_file(b"hello world");
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let data = cache.read_range(0..0).unwrap();
        assert!(data.is_empty());
        verify(&cache);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let file = create_test_file(b"hello world");
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        // This should not panic, but return an error or empty slice depending on your design
        // Currently, your code will panic due to `unwrap()` on `None`
        // You may want to handle this case more gracefully
        assert!(cache.read_range(20..30).unwrap().is_empty());
        verify(&cache);
    }

    #[test]
    fn test_cache_eviction() {
        let file = create_test_file(b"0123456789abcdef");
        let mut cache = LazyCache::open(file.path(), 4, 8).unwrap();
        // Load blocks 0 and 1
        let _ = cache.read_range(0..8).unwrap();
        // Load block 2, which should evict block 0 or 1 due to max_size=8
        let _ = cache.read_range(8..12).unwrap();
        // Check that the cache still works
        let data = cache.read_range(8..12).unwrap();
        assert_eq!(data, b"89ab");
        verify(&cache);
    }

    #[test]
    fn test_chunk_consolidation() {
        let file = create_test_file(b"0123456789abcdef");
        let mut cache = LazyCache::open(file.path(), 4, 16).unwrap();
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
        verify(&cache);
    }

    #[test]
    fn test_overlapping_ranges() {
        let file = create_test_file(b"0123456789abcdef");
        let mut cache = LazyCache::open(file.path(), 4, 16).unwrap();
        // Load overlapping ranges
        let _ = cache.read_range(2..6).unwrap();
        let _ = cache.read_range(4..10).unwrap();
        // Check that the data is correct
        let data = cache.read_range(2..10).unwrap();
        assert_eq!(data, b"23456789");
        verify(&cache);
    }

    #[test]
    fn test_lru_behavior() {
        let file = create_test_file(b"0123456789abcdef");
        let mut cache = LazyCache::open(file.path(), 4, 8).unwrap();
        // Load block 0
        let _ = cache.read_range(0..4).unwrap();
        // Load block 1
        let _ = cache.read_range(4..8).unwrap();
        // Load block 2, which should evict block 0
        let _ = cache.read_range(8..12).unwrap();
        // Block 0 should be evicted, so accessing it again should reload it
        let data = cache.read_range(0..4).unwrap();
        assert_eq!(data, b"0123");
        verify(&cache);
    }

    #[test]
    fn test_small_block_size() {
        let file = create_test_file(b"abc");
        let mut cache = LazyCache::open(file.path(), 1, 3).unwrap();
        let data = cache.read_range(0..3).unwrap();
        assert_eq!(data, b"abc");
        verify(&cache);
    }

    #[test]
    fn test_large_block_size() {
        let file = create_test_file(b"hello world");
        let mut cache = LazyCache::open(file.path(), 1024, 1024).unwrap();
        let data = cache.read_range(0..11).unwrap();
        assert_eq!(data, b"hello world");
        verify(&cache);
    }

    #[test]
    fn test_file_smaller_than_block() {
        let file = create_test_file(b"abc");
        let mut cache = LazyCache::open(file.path(), 1024, 1024).unwrap();
        let data = cache.read_range(0..3).unwrap();
        assert_eq!(data, b"abc");
        verify(&cache);
    }

    #[test]
    fn test_multiple_gets_same_block() {
        let file = create_test_file(b"0123456789abcdef");
        let mut cache = LazyCache::open(file.path(), 4, 16).unwrap();
        // Get the same block multiple times
        let _ = cache.read_range(0..4).unwrap();
        let _ = cache.read_range(0..4).unwrap();
        let _ = cache.read_range(0..4).unwrap();
        // The block should still be in the cache
        let data = cache.read_range(0..4).unwrap();
        assert_eq!(data, b"0123");
        verify(&cache);
    }

    #[test]
    fn test_read_method() {
        let data = b"hello world";
        let file = create_test_file(data);
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let _ = cache.read(6).unwrap();
        let data = cache.read(5).unwrap();
        assert_eq!(data, b"world");
        // We reached the end so next read should bring an empty slice
        assert!(cache.read(1).unwrap().is_empty());
        verify(&cache);
    }

    #[test]
    fn test_read_empty() {
        let data = b"hello world";
        let file = create_test_file(data);
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let data = cache.read(0).unwrap();
        assert!(data.is_empty());
        verify(&cache);
    }

    #[test]
    fn test_read_beyond_end() {
        let data = b"hello world";
        let file = create_test_file(data);
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let _ = cache.read(11).unwrap();
        let data = cache.read(5).unwrap();
        assert!(data.is_empty());
        verify(&cache);
    }

    #[test]
    fn test_read_exact_range() {
        let file = create_test_file(b"hello world");
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let data = cache.read_exact_range(0..5).unwrap();
        assert_eq!(data, b"hello");
        assert_eq!(cache.read_exact_range(5..11).unwrap(), b" world");
        assert!(cache.read_exact_range(12..13).is_err());
        verify(&cache);
    }

    #[test]
    fn test_read_exact_range_error() {
        let file = create_test_file(b"hello world");
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let result = cache.read_exact_range(0..20);
        assert!(result.is_err());
        verify(&cache);
    }

    #[test]
    fn test_read_exact() {
        let file = create_test_file(b"hello world");
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let data = cache.read_exact(5).unwrap();
        assert_eq!(data, b"hello");
        assert_eq!(cache.read_exact(6).unwrap(), b" world");
        assert!(cache.read_exact(0).is_ok());
        assert!(cache.read_exact(1).is_err());
        verify(&cache);
    }

    #[test]
    fn test_read_exact_error() {
        let file = create_test_file(b"hello world");
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let result = cache.read_exact(20);
        assert!(result.is_err());
        verify(&cache);
    }

    #[test]
    fn test_read_until_limit() {
        let file = create_test_file(b"hello world");
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let data = cache.read_until_or_limit(b' ', 10).unwrap();
        assert_eq!(data, b"hello ");
        assert_eq!(cache.read_exact(5).unwrap(), b"world");
        verify(&cache);
    }

    #[test]
    fn test_read_until_limit_not_found() {
        let file = create_test_file(b"hello world");
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let data = cache.read_until_or_limit(b'\n', 11).unwrap();
        assert_eq!(data, b"hello world");
        assert!(cache.read(1).unwrap().is_empty());
        verify(&cache);
    }

    #[test]
    fn test_read_until_limit_beyond_stream() {
        let file = create_test_file(b"hello world");
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let data = cache.read_until_or_limit(b'\n', 42).unwrap();
        assert_eq!(data, b"hello world");
        assert!(cache.read(1).unwrap().is_empty());
        verify(&cache);
    }

    #[test]
    fn test_read_until_limit_with_limit() {
        let file = create_test_file(b"hello world");
        let mut cache = LazyCache::open(file.path(), 4, 1024).unwrap();
        let data = cache.read_until_or_limit(b' ', 42).unwrap();
        assert_eq!(data, b"hello ");

        let data = cache.read_until_or_limit(b' ', 2).unwrap();
        assert_eq!(data, b"wo");

        let data = cache.read_until_or_limit(b' ', 42).unwrap();
        assert_eq!(data, b"rld");
        verify(&cache);
    }

    #[test]
    fn test_read_until_utf16_limit() {
        let content = io::Cursor::new(
            b"\x61\x00\x62\x00\x63\x00\x64\x00\x00\x00\x61\x00\x62\x00\x63\x00\x64\x00\x00",
        );
        let mut cache = LazyCache::from_read_seek(content, 3, 512).unwrap();
        let data = cache.read_until_utf16_or_limit(b"\x00\x00", 512).unwrap();
        assert_eq!(data, b"\x61\x00\x62\x00\x63\x00\x64\x00\x00\x00");

        let data = cache.read_until_utf16_or_limit(b"\x00\x00", 1).unwrap();
        assert_eq!(data, b"\x61\x00");

        assert_eq!(
            cache.read_until_utf16_or_limit(b"\xff\xff", 64).unwrap(),
            b"\x62\x00\x63\x00\x64\x00\x00"
        );
        verify(&cache);
    }
}
