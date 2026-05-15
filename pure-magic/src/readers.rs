//! Data readers for magic number detection.
//!
//! Provides efficient readers for different data sources with caching and buffering
//! strategies optimized for file format identification.
//!
//! # Types
//!
//! - [`DataReader`] - A generic reader enum supporting slices, vectors, and files.
//! - [`BufReader`] - A buffered reader for in-memory byte slices.
//! - [`LazyCache`] - A lazy-loading cache reader for files with multi-tiered caching.
//!
//! # Traits
//!
//! - [`DataRead`] - Extended read operations for magic number detection.

use std::{
    fs::File,
    io::{self, SeekFrom},
    ops::Range,
};

mod cache;
pub use cache::LazyCache;

mod slice;
pub use slice::BufReader;

use crate::FILE_BYTES_MAX;

/// A trait for reading data with position tracking and range-based access.
///
/// Implementors provide efficient random access to byte data for file magic
/// detection, supporting both in-memory and file-backed storage.
pub trait DataRead {
    /// Returns the current position in the data stream.
    fn stream_position(&self) -> u64;

    /// Computes the absolute byte offset from a [`SeekFrom`] position.
    #[inline]
    fn offset_from_start(&self, pos: SeekFrom) -> u64 {
        match pos {
            SeekFrom::Start(s) => s,
            SeekFrom::Current(p) => {
                (self.stream_position() as i128 + p as i128).clamp(0, u64::MAX as i128) as u64
            }
            SeekFrom::End(e) => {
                (self.data_size() as i128 + e as i128).clamp(0, u64::MAX as i128) as u64
            }
        }
    }

    /// Reads a range of bytes from the data.
    ///
    /// Returns an empty slice if the range is beyond the end of data.
    fn read_range(&mut self, range: Range<u64>) -> Result<&[u8], io::Error>;

    /// Reads up to `count` bytes from the current position.
    ///
    /// Returns fewer bytes if the end of data is reached.
    #[inline]
    fn read_count(&mut self, count: u64) -> Result<&[u8], io::Error> {
        let pos = self.stream_position();
        let range = pos..(pos.saturating_add(count));
        self.read_range(range)
    }

    /// Reads exactly the specified byte range.
    ///
    /// # Errors
    ///
    /// Returns an error if the range extends beyond the available data.
    fn read_exact_range(&mut self, range: Range<u64>) -> Result<&[u8], io::Error> {
        let range_len = range.end - range.start;
        let b = self.read_range(range)?;
        if b.len() as u64 != range_len {
            Err(io::Error::from(io::ErrorKind::UnexpectedEof))
        } else {
            Ok(b)
        }
    }

    /// Reads exactly `count` bytes from the current position.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than `count` bytes are available.
    fn read_exact_count(&mut self, count: u64) -> Result<&[u8], io::Error> {
        let b = self.read_count(count)?;
        debug_assert!(b.len() <= count as usize);
        if b.len() as u64 != count {
            Err(io::ErrorKind::UnexpectedEof.into())
        } else {
            Ok(b)
        }
    }

    /// Reads exactly enough bytes to fill `buf`.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than `buf.len()` bytes are available.
    fn read_exact_into(&mut self, buf: &mut [u8]) -> Result<(), io::Error> {
        let read = self.read_exact_count(buf.len() as u64)?;
        // this function call should not panic as read_exact
        // guarantees we read exactly the length of buf
        buf.copy_from_slice(read);
        Ok(())
    }

    /// Reads bytes until any of the delimiters or `limit` bytes is reached.
    ///
    /// The delimiter byte is included in the returned slice.
    fn read_until_any_delim_or_limit(
        &mut self,
        delims: &[u8],
        limit: u64,
    ) -> Result<&[u8], io::Error>;

    /// Reads bytes until `byte` or `limit` bytes is reached.
    ///
    /// The delimiter byte is included in the returned slice.
    fn read_until_or_limit(&mut self, byte: u8, limit: u64) -> Result<&[u8], io::Error>;

    /// Reads bytes while `f` returns `true` or until `limit` bytes is reached.
    ///
    /// The byte that caused `f` to return `false` is not included.
    fn read_while_or_limit<F>(&mut self, f: F, limit: u64) -> Result<&[u8], io::Error>
    where
        F: Fn(u8) -> bool;

    /// Reads bytes until a UTF-16 character or `limit` bytes is reached.
    ///
    /// The UTF-16 character is included in the returned slice.
    fn read_until_utf16_or_limit(
        &mut self,
        utf16_char: &[u8; 2],
        limit: u64,
    ) -> Result<&[u8], io::Error>;

    /// Returns the total size of the data in bytes.
    fn data_size(&self) -> u64;

    /// Sets the position for future reads.
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64>;
}

/// A generic reader for data backed by different sources.
///
/// Provides a uniform interface for reading from in-memory buffers or files.
pub enum DataReader<'b> {
    /// A reader backed by a borrowed byte slice.
    ///
    /// Useful for zero-copy reads from existing in-memory data.
    Slice(BufReader<&'b [u8]>),
    /// A reader backed by an owned byte vector.
    ///
    /// Useful when the data needs to be owned.
    Vec(BufReader<Vec<u8>>),
    /// A reader backed by a file with lazy caching.
    ///
    /// Uses [`LazyCache`] for efficient disk I/O.
    File(LazyCache<File>),
}

impl DataReader<'_> {
    /// Creates a new `DataReader` backed by a file with lazy caching.
    ///
    /// The file is wrapped in a [`LazyCache`] with:
    /// - A hot cache of 14 MiB (2 × [`FILE_BYTES_MAX`])
    /// - A warm cache of 100 MiB
    ///
    /// This configuration is optimized for file-based magic number detection,
    /// balancing memory usage with I/O efficiency.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if cache initialization fails.
    pub fn from_file(r: File) -> Result<Self, io::Error> {
        let x = LazyCache::<File>::from_read_seek(r)
            .and_then(|lc| lc.with_hot_cache(2 * FILE_BYTES_MAX))
            .map(|lc| lc.with_warm_cache(100 << 20))?;
        Ok(Self::File(x))
    }
}

impl<'b> DataReader<'b> {
    /// Creates a new `DataReader` backed by a borrowed byte slice.
    ///
    /// This is a zero-copy constructor that wraps the slice in a [`BufReader`].
    /// The lifetime of the returned reader is tied to the input slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use pure_magic::readers::{DataReader, DataRead};
    ///
    /// let data = b"hello world";
    /// let reader = DataReader::from_slice(data);
    /// assert_eq!(reader.data_size(), data.len() as u64);
    /// ```
    pub fn from_slice(s: &'b [u8]) -> Self {
        Self::Slice(BufReader::from_slice(s))
    }
}

impl DataReader<'_> {
    /// Creates a new `DataReader` backed by an owned byte vector.
    ///
    /// The vector is wrapped in a [`BufReader`], allowing the data to be owned
    /// independently of any borrow.
    ///
    /// # Examples
    ///
    /// ```
    /// use pure_magic::readers::{DataReader, DataRead};
    ///
    /// let data = vec![1u8, 2, 3, 4, 5];
    /// let reader = DataReader::from_vec(data);
    /// assert_eq!(reader.data_size(), 5);
    /// ```
    pub fn from_vec(v: Vec<u8>) -> Self {
        Self::Vec(BufReader::from_slice(v))
    }
}

impl DataRead for DataReader<'_> {
    fn stream_position(&self) -> u64 {
        match self {
            DataReader::Slice(b) => b.stream_position(),
            DataReader::Vec(v) => v.stream_position(),
            DataReader::File(f) => f.stream_position(),
        }
    }

    fn offset_from_start(&self, pos: SeekFrom) -> u64 {
        match self {
            DataReader::Slice(b) => b.offset_from_start(pos),
            DataReader::Vec(v) => v.offset_from_start(pos),
            DataReader::File(f) => f.offset_from_start(pos),
        }
    }

    fn read_range(&mut self, range: Range<u64>) -> Result<&[u8], io::Error> {
        match self {
            DataReader::Slice(b) => b.read_range(range),
            DataReader::Vec(v) => v.read_range(range),
            DataReader::File(f) => f.read_range(range),
        }
    }

    fn read_count(&mut self, count: u64) -> Result<&[u8], io::Error> {
        match self {
            DataReader::Slice(b) => b.read_count(count),
            DataReader::Vec(v) => v.read_count(count),
            DataReader::File(f) => f.read_count(count),
        }
    }

    fn read_exact_range(&mut self, range: Range<u64>) -> Result<&[u8], io::Error> {
        match self {
            DataReader::Slice(b) => b.read_exact_range(range),
            DataReader::Vec(v) => v.read_exact_range(range),
            DataReader::File(f) => f.read_exact_range(range),
        }
    }

    fn read_exact_count(&mut self, count: u64) -> Result<&[u8], io::Error> {
        match self {
            DataReader::Slice(b) => b.read_exact_count(count),
            DataReader::Vec(v) => v.read_exact_count(count),
            DataReader::File(f) => f.read_exact_count(count),
        }
    }

    fn read_exact_into(&mut self, buf: &mut [u8]) -> Result<(), io::Error> {
        match self {
            DataReader::Slice(b) => b.read_exact_into(buf),
            DataReader::Vec(v) => v.read_exact_into(buf),
            DataReader::File(f) => f.read_exact_into(buf),
        }
    }

    fn read_until_any_delim_or_limit(
        &mut self,
        delims: &[u8],
        limit: u64,
    ) -> Result<&[u8], io::Error> {
        match self {
            DataReader::Slice(b) => b.read_until_any_delim_or_limit(delims, limit),
            DataReader::Vec(v) => v.read_until_any_delim_or_limit(delims, limit),
            DataReader::File(f) => f.read_until_any_delim_or_limit(delims, limit),
        }
    }

    fn read_until_or_limit(&mut self, byte: u8, limit: u64) -> Result<&[u8], io::Error> {
        match self {
            DataReader::Slice(b) => b.read_until_or_limit(byte, limit),
            DataReader::Vec(v) => v.read_until_or_limit(byte, limit),
            DataReader::File(f) => f.read_until_or_limit(byte, limit),
        }
    }

    fn read_while_or_limit<F>(&mut self, f: F, limit: u64) -> Result<&[u8], io::Error>
    where
        F: Fn(u8) -> bool,
    {
        match self {
            DataReader::Slice(b) => b.read_while_or_limit(f, limit),
            DataReader::Vec(v) => v.read_while_or_limit(f, limit),
            DataReader::File(l) => l.read_while_or_limit(f, limit),
        }
    }

    fn read_until_utf16_or_limit(
        &mut self,
        utf16_char: &[u8; 2],
        limit: u64,
    ) -> Result<&[u8], io::Error> {
        match self {
            DataReader::Slice(b) => b.read_until_utf16_or_limit(utf16_char, limit),
            DataReader::Vec(v) => v.read_until_utf16_or_limit(utf16_char, limit),
            DataReader::File(f) => f.read_until_utf16_or_limit(utf16_char, limit),
        }
    }

    fn data_size(&self) -> u64 {
        match self {
            DataReader::Slice(b) => b.data_size(),
            DataReader::Vec(v) => v.data_size(),
            DataReader::File(f) => f.data_size(),
        }
    }

    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match self {
            DataReader::Slice(b) => b.seek(pos),
            DataReader::Vec(v) => v.seek(pos),
            DataReader::File(f) => f.seek(pos),
        }
    }
}
