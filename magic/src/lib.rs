#![forbid(unsafe_code)]

use dyf::{DynDisplay, FormatString, dformat};
use flagset::{FlagSet, flags};
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use lazy_cache::LazyCache;
use memchr::memchr;
use pest::{Span, error::ErrorVariant};
use regex::bytes::{self};
use serde::{Deserialize, Serialize};
use serde_json;
use std::{
    borrow::Cow,
    cmp::max,
    collections::{HashMap, HashSet},
    fmt::{self, Debug, Display},
    io::{self, Read, Seek, SeekFrom},
    ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Sub},
    path::Path,
    rc::Rc,
    str::Utf8Error,
    usize,
};
use tar::Archive;
use thiserror::Error;
use tracing::{Level, debug, enabled, error, trace};

use crate::{
    numeric::{Float, FloatDataType, Scalar, ScalarDataType},
    parser::{FileMagicParser, Rule},
    utils::{decode_id3, find_json_boundaries},
};

mod numeric;
mod parser;
mod utils;

const HARDCODED_MAGIC_STRENGTH: u64 = 2048;
const HARDCODED_SOURCE: &str = "hardcoded";
// corresponds to FILE_INDIR_MAX constant defined in libmagic
const MAX_RECURSION: usize = 50;
// constant found in libmagic. It is used to limit for search tests
pub const FILE_BYTES_MAX: usize = 7 * 1024 * 1024;
// constant found in libmagic. It is used to limit for regex tests
const FILE_REGEX_MAX: usize = 8192;

pub const DEFAULT_BIN_MIMETYPE: &str = "application/octet-stream";
pub const DEFAULT_TEXT_MIMETYPE: &str = "text/plain";

pub(crate) const TIMESTAMP_FORMAT: &'static str = "%Y-%m-%d %H:%M:%S";

macro_rules! debug_panic {
    ($($arg:tt)*) => {
        if cfg!(debug_assertions) {
            panic!($($arg)*);
        }
    };
}

macro_rules! read {
    ($r: expr, $ty: ty) => {{
        let mut a = [0u8; std::mem::size_of::<$ty>()];
        $r.read_exact(&mut a)?;
        a
    }};
}

macro_rules! read_le {
    ($r:expr, $ty: ty ) => {{ <$ty>::from_le_bytes(read!($r, $ty)) }};
}

macro_rules! read_be {
    ($r:expr, $ty: ty ) => {{ <$ty>::from_be_bytes(read!($r, $ty)) }};
}

macro_rules! read_me {
    ($r: expr) => {{ ((read_le!($r, u16) as i32) << 16) | (read_le!($r, u16) as i32) }};
}

#[inline(always)]
fn read_octal_u64<R: Read + Seek>(haystack: &mut LazyCache<R>) -> Option<u64> {
    let s = haystack
        .read_while_or_limit(
            |b| match b {
                b'0'..=b'7' => true,
                _ => false,
            },
            22,
        )
        .map(|buf| str::from_utf8(buf))
        .ok()?
        .ok()?;

    if !s.starts_with("0") {
        return None;
    }

    u64::from_str_radix(s, 8).ok()
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Msg(String),
    #[error("source={0} line={1} error={2}")]
    Localized(String, usize, Box<Error>),
    #[error("unexpected rule: {0}")]
    UnexpectedRule(String),
    #[error("missing rule: {0}")]
    MissingRule(String),
    #[error("maximum recursion reached: {0}")]
    MaximumRecursion(usize),
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("parser error: {0}")]
    Parse(#[from] Box<pest::error::Error<Rule>>),
    #[error("from-utf8: {0}")]
    Utf8(#[from] Utf8Error),
    #[error("formatting: {0}")]
    Format(#[from] dyf::Error),
    #[error("regex: {0}")]
    Regex(#[from] regex::Error),
    #[error("{0}")]
    Serialize(#[from] bincode::error::EncodeError),
    #[error("{0}")]
    Deserialize(#[from] bincode::error::DecodeError),
}

impl Error {
    fn parser<S: AsRef<str>>(msg: S, span: Span<'_>) -> Self {
        Self::Parse(Box::new(pest::error::Error::new_from_span(
            ErrorVariant::CustomError {
                message: msg.as_ref().into(),
            },
            span,
        )))
    }

    fn msg<M: AsRef<str>>(msg: M) -> Self {
        Self::Msg(msg.as_ref().into())
    }

    fn localized<S: AsRef<str>>(source: S, line: usize, err: Error) -> Self {
        Self::Localized(source.as_ref().into(), line, err.into())
    }

    /// Unwraps the localized error
    pub fn unwrap_localized(&self) -> &Self {
        match self {
            Self::Localized(_, _, e) => e,
            _ => self,
        }
    }
}

#[derive(Debug, Error)]
#[error("{0}")]
pub struct ParserError(pest::error::Error<Rule>);

impl From<pest::error::Error<Rule>> for ParserError {
    fn from(value: pest::error::Error<Rule>) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Message {
    String(String),
    Format {
        printf_spec: String,
        fs: FormatString,
    },
}

impl Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(s) => write!(f, "{}", s),
            Self::Format { printf_spec: _, fs } => write!(f, "{}", fs.to_string_lossy()),
        }
    }
}

impl Message {
    fn to_string_lossy(&self) -> Cow<'_, str> {
        match self {
            Message::String(s) => Cow::Borrowed(s),
            Message::Format { printf_spec: _, fs } => fs.to_string_lossy(),
        }
    }

    #[inline(always)]
    fn format_with(&self, mr: Option<&MatchRes>) -> Result<Cow<'_, str>, Error> {
        match self {
            Self::String(s) => Ok(Cow::Borrowed(s.as_str())),
            Self::Format {
                printf_spec: c_spec,
                fs,
            } => {
                if let Some(mr) = mr {
                    match mr {
                        MatchRes::Float(_, _) | MatchRes::Bytes(_, _, _, _) => {
                            Ok(Cow::Owned(dformat!(fs, mr)?))
                        }
                        MatchRes::Scalar(_, scalar) => {
                            // we want to print a byte as char
                            if c_spec.as_str() == "c" {
                                match scalar {
                                    Scalar::byte(b) => {
                                        let b = (*b as u8) as char;
                                        Ok(Cow::Owned(dformat!(fs, b)?))
                                    }
                                    Scalar::ubyte(b) => {
                                        let b = *b as char;
                                        Ok(Cow::Owned(dformat!(fs, b)?))
                                    }
                                    _ => Ok(Cow::Owned(dformat!(fs, mr)?)),
                                }
                            } else {
                                Ok(Cow::Owned(dformat!(fs, mr)?))
                            }
                        }
                    }
                } else {
                    Ok(fs.to_string_lossy())
                }
            }
        }
    }
}

impl ScalarDataType {
    #[inline(always)]
    fn read<R: Read + Seek>(&self, from: &mut R, switch_endianness: bool) -> Result<Scalar, Error> {
        macro_rules! _read_le {
            ($ty: ty) => {{
                if switch_endianness {
                    <$ty>::from_be_bytes(read!(from, $ty))
                } else {
                    <$ty>::from_le_bytes(read!(from, $ty))
                }
            }};
        }

        macro_rules! _read_be {
            ($ty: ty) => {{
                if switch_endianness {
                    <$ty>::from_le_bytes(read!(from, $ty))
                } else {
                    <$ty>::from_be_bytes(read!(from, $ty))
                }
            }};
        }

        macro_rules! _read_ne {
            ($ty: ty) => {{
                if cfg!(target_endian = "big") {
                    _read_be!($ty)
                } else {
                    _read_le!($ty)
                }
            }};
        }

        macro_rules! _read_me {
            () => {
                ((_read_le!(u16) as i32) << 16) | (_read_le!(u16) as i32)
            };
        }

        Ok(match self {
            // signed
            Self::byte => Scalar::byte(read!(from, u8)[0] as i8),
            Self::short => Scalar::short(_read_ne!(i16)),
            Self::long => Scalar::long(_read_ne!(i32)),
            Self::date => Scalar::date(_read_ne!(i32)),
            Self::ldate => Scalar::ldate(_read_ne!(i32)),
            Self::qwdate => Scalar::qwdate(_read_ne!(i64)),
            Self::leshort => Scalar::leshort(_read_le!(i16)),
            Self::lelong => Scalar::lelong(_read_le!(i32)),
            Self::lequad => Scalar::lequad(_read_le!(i64)),
            Self::bequad => Scalar::bequad(_read_be!(i64)),
            Self::belong => Scalar::belong(_read_be!(i32)),
            Self::bedate => Scalar::bedate(_read_be!(i32)),
            Self::beldate => Scalar::beldate(_read_be!(i32)),
            Self::beqdate => Scalar::beqdate(_read_be!(i64)),
            // unsigned
            Self::ubyte => Scalar::ubyte(read!(from, u8)[0]),
            Self::ushort => Scalar::ushort(_read_ne!(u16)),
            Self::uleshort => Scalar::uleshort(_read_le!(u16)),
            Self::ulelong => Scalar::ulelong(_read_le!(u32)),
            Self::uledate => Scalar::uledate(_read_le!(u32)),
            Self::ulequad => Scalar::ulequad(_read_le!(u64)),
            Self::offset => Scalar::offset(from.stream_position()?),
            Self::ubequad => Scalar::ubequad(_read_be!(u64)),
            Self::medate => Scalar::medate(_read_me!()),
            Self::meldate => Scalar::meldate(_read_me!()),
            Self::melong => Scalar::melong(_read_me!()),
            Self::beshort => Scalar::beshort(_read_be!(i16)),
            Self::quad => Scalar::quad(_read_ne!(i64)),
            Self::uquad => Scalar::uquad(_read_ne!(u64)),
            Self::ledate => Scalar::ledate(_read_le!(i32)),
            Self::leldate => Scalar::leldate(_read_le!(i32)),
            Self::leqdate => Scalar::leqdate(_read_le!(i64)),
            Self::leqldate => Scalar::leqldate(_read_le!(i64)),
            Self::leqwdate => Scalar::leqwdate(_read_le!(i64)),
            Self::ubelong => Scalar::ubelong(_read_be!(u32)),
            Self::ulong => Scalar::ulong(_read_ne!(u32)),
            Self::ubeshort => Scalar::ubeshort(_read_be!(u16)),
            Self::ubeqdate => Scalar::ubeqdate(_read_be!(u64)),
            Self::lemsdosdate => Scalar::lemsdosdate(_read_le!(u16)),
            Self::lemsdostime => Scalar::lemsdostime(_read_le!(u16)),
            Self::guid => Scalar::guid(u128::from_be_bytes(read!(from, u128))),
        })
    }
}

impl FloatDataType {
    #[inline(always)]
    fn read<R: Read + Seek>(&self, from: &mut R, switch_endianness: bool) -> Result<Float, Error> {
        macro_rules! _read_le {
            ($ty: ty) => {{
                if switch_endianness {
                    <$ty>::from_be_bytes(read!(from, $ty))
                } else {
                    <$ty>::from_le_bytes(read!(from, $ty))
                }
            }};
        }

        macro_rules! _read_be {
            ($ty: ty) => {{
                if switch_endianness {
                    <$ty>::from_le_bytes(read!(from, $ty))
                } else {
                    <$ty>::from_be_bytes(read!(from, $ty))
                }
            }};
        }

        macro_rules! _read_ne {
            ($ty: ty) => {{
                if cfg!(target_endian = "big") {
                    _read_be!($ty)
                } else {
                    _read_le!($ty)
                }
            }};
        }

        macro_rules! _read_me {
            () => {
                ((_read_le!(u16) as i32) << 16) | (_read_le!(u16) as i32)
            };
        }

        Ok(match self {
            Self::lefloat => Float::lefloat(_read_le!(f32)),
            Self::befloat => Float::befloat(_read_le!(f32)),
            Self::ledouble => Float::ledouble(_read_le!(f64)),
            Self::bedouble => Float::bedouble(_read_be!(f64)),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum Op {
    Mul,
    Add,
    Sub,
    Div,
    Mod,
    And,
    Xor,
    Or,
}

impl Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Mul => write!(f, "*"),
            Op::Add => write!(f, "+"),
            Op::Sub => write!(f, "-"),
            Op::Div => write!(f, "/"),
            Op::Mod => write!(f, "%"),
            Op::And => write!(f, "&"),
            Op::Or => write!(f, "|"),
            Op::Xor => write!(f, "^"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum CmpOp {
    Eq,
    Lt,
    Gt,
    BitAnd,
    Neq, // ! operator
    Xor,
    Not, // ~ operator
}

impl CmpOp {
    #[inline(always)]
    fn is_neq(&self) -> bool {
        matches!(self, Self::Neq)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScalarTransform {
    op: Op,
    num: Scalar,
}

impl ScalarTransform {
    fn apply(&self, s: Scalar) -> Option<Scalar> {
        match self.op {
            Op::Add => s.checked_add(self.num),
            Op::Sub => s.checked_sub(self.num),
            Op::Mul => s.checked_mul(self.num),
            Op::Div => s.checked_div(self.num),
            Op::Mod => s.checked_rem(self.num),
            Op::And => Some(s.bitand(self.num)),
            Op::Xor => Some(s.bitxor(self.num)),
            Op::Or => Some(s.bitor(self.num)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FloatTransform {
    op: Op,
    num: Float,
}

impl FloatTransform {
    fn apply(&self, s: Float) -> Float {
        match self.op {
            Op::Add => s.add(self.num),
            Op::Sub => s.sub(self.num),
            Op::Mul => s.mul(self.num),
            // returns inf when div by 0
            Op::Div => s.div(self.num),
            // returns NaN when rem by 0
            Op::Mod => s.rem(self.num),
            // parser makes sure those operators cannot be used
            Op::And | Op::Xor | Op::Or => {
                debug_panic!("unsupported operation");
                s
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum TestValue<T> {
    Value(T),
    Any,
}

impl<T> TestValue<T> {
    #[inline(always)]
    fn as_ref(&self) -> TestValue<&T> {
        match self {
            Self::Value(v) => TestValue::Value(v),
            Self::Any => TestValue::Any,
        }
    }
}

flags! {
    enum ReMod: u8{
        CaseInsensitive,
        StartOffsetUpdate,
        LineLimit,
        ForceBinary,
        ForceText,
        TrimMatch,
    }
}

fn serialize_regex<S>(re: &bytes::Regex, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    re.as_str().serialize(serializer)
}

fn deserialize_regex<'de, D>(deserializer: D) -> Result<bytes::Regex, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let wrapper = String::deserialize(deserializer)?;
    Ok(bytes::Regex::new(&wrapper).map_err(|e| serde::de::Error::custom(e))?)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegexTest {
    #[serde(
        serialize_with = "serialize_regex",
        deserialize_with = "deserialize_regex"
    )]
    re: bytes::Regex,
    length: Option<usize>,
    mods: FlagSet<ReMod>,
    str_mods: FlagSet<StringMod>,
    non_magic_len: usize,
    binary: bool,
    cmp_op: CmpOp,
}

impl RegexTest {
    #[inline(always)]
    fn is_binary(&self) -> bool {
        self.binary
            || self.mods.contains(ReMod::ForceBin)
            || self.str_mods.contains(StringMod::ForceBin)
    }

    fn match_buf<'buf>(
        &self,
        off_buf: u64, // absolute buffer offset in content
        stream_kind: StreamKind,
        buf: &'buf [u8],
    ) -> Option<MatchRes<'buf>> {
        let mr = match stream_kind {
            StreamKind::Text(_) => {
                let mut off_txt = off_buf;

                let mut line_limit = self.length.unwrap_or(usize::MAX);

                for line in buf.split(|c| c == &b'\n') {
                    // we don't need to break on offset
                    // limit as buf contains the good amount
                    // of bytes to match against
                    if line_limit == 0 {
                        break;
                    }

                    if let Some(re_match) = self.re.find(&line) {
                        // the offset of the string is computed from the start of the buffer
                        let start_offset = off_txt + re_match.start() as u64;

                        // if we matched until EOL we need to add one to include the delimiter removed from the split
                        let stop_offset = if re_match.end() == line.len() {
                            Some(start_offset + re_match.as_bytes().len() as u64 + 1)
                        } else {
                            None
                        };

                        return Some(MatchRes::Bytes(
                            start_offset,
                            stop_offset,
                            re_match.as_bytes(),
                            Encoding::Utf8,
                        ));
                    }

                    off_txt += line.len() as u64;
                    // we have to add one because lines do not contain splitting character
                    off_txt += 1;
                    line_limit = line_limit.saturating_sub(1)
                }
                None
            }

            StreamKind::Binary => {
                if let Some(re_match) = self.re.find(&buf) {
                    Some(MatchRes::Bytes(
                        // the offset of the string is computed from the start of the buffer
                        off_buf + re_match.start() as u64,
                        None,
                        re_match.as_bytes(),
                        Encoding::Utf8,
                    ))
                } else {
                    None
                }
            }
        };

        // handle the case where we want the regex not to match
        if self.cmp_op.is_neq() && mr.is_none() {
            return Some(MatchRes::Bytes(off_buf, None, buf, Encoding::Utf8));
        }

        mr
    }
}

impl From<RegexTest> for Test {
    fn from(value: RegexTest) -> Self {
        Self::Regex(value)
    }
}

flags! {
    enum StringMod: u8{
        ForceBin,
        UpperInsensitive,
        LowerInsensitive,
        FullWordMatch,
        Trim,
        ForceText,
        CompactWhitespace,
        OptBlank,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StringTest {
    test_val: TestValue<Vec<u8>>,
    cmp_op: CmpOp,
    length: Option<usize>,
    mods: FlagSet<StringMod>,
    binary: bool,
}

impl From<StringTest> for Test {
    fn from(value: StringTest) -> Self {
        Self::String(value)
    }
}

#[inline(always)]
fn string_match<'str>(str: &'str [u8], mods: FlagSet<StringMod>, buf: &[u8]) -> (bool, usize) {
    let mut consumed = 0;
    // we can do a simple string comparison
    if mods.is_disjoint(
        StringMod::UpperInsensitive
            | StringMod::LowerInsensitive
            | StringMod::FullWordMatch
            | StringMod::CompactWhitespace
            | StringMod::OptBlank,
    ) {
        // we check if target contains
        if buf.starts_with(str) {
            (true, str.len())
        } else {
            (false, consumed)
        }
    } else {
        let mut i_src = 0;
        let mut iter = buf.iter().peekable();

        macro_rules! consume_target {
            () => {{
                iter.next();
                consumed += 1;
            }};
        }

        macro_rules! continue_next_iteration {
            () => {{
                consume_target!();
                i_src += 1;
                continue;
            }};
        }

        while let Some(&&b) = iter.peek() {
            let Some(&ref_byte) = str.get(i_src) else {
                break;
            };

            if mods.contains(StringMod::OptBlank) && (b == b' ' || ref_byte == b' ') {
                if b == b' ' {
                    // we ignore whitespace in target
                    consume_target!();
                }

                if ref_byte == b' ' {
                    // we ignore whitespace in test
                    i_src += 1;
                }

                continue;
            }

            if mods.contains(StringMod::UpperInsensitive) {
                //upper case characters in the magic match both lower and upper case characters in the target
                if ref_byte.is_ascii_uppercase() && ref_byte == b.to_ascii_uppercase()
                    || ref_byte == b
                {
                    continue_next_iteration!()
                }
            }

            if mods.contains(StringMod::LowerInsensitive) {
                if ref_byte.is_ascii_lowercase() && ref_byte == b.to_ascii_lowercase()
                    || ref_byte == b
                {
                    continue_next_iteration!()
                }
            }

            if mods.contains(StringMod::CompactWhitespace) && ref_byte == b' ' {
                let mut src_blk = 0;
                while let Some(b' ') = str.get(i_src) {
                    src_blk += 1;
                    i_src += 1;
                }

                let mut tgt_blk = 0;
                while let Some(b' ') = iter.peek() {
                    tgt_blk += 1;
                    consume_target!();
                }

                if src_blk > tgt_blk {
                    return (false, consumed);
                }

                continue;
            }

            if ref_byte == b {
                continue_next_iteration!()
            } else {
                return (false, consumed);
            }
        }

        if mods.contains(StringMod::FullWordMatch) {
            if let Some(b) = iter.peek() {
                if !b.is_ascii_whitespace() {
                    return (false, consumed);
                }
            }
        }

        (consumed > 0 && consumed < buf.len(), consumed)
    }
}

impl StringTest {
    fn has_length_mod(&self) -> bool {
        !self.mods.is_disjoint(
            StringMod::UpperInsensitive
                | StringMod::LowerInsensitive
                | StringMod::FullWordMatch
                | StringMod::CompactWhitespace
                | StringMod::OptBlank,
        )
    }

    #[inline(always)]
    fn test_value_len(&self) -> usize {
        match self.test_val.as_ref() {
            TestValue::Value(s) => s.len(),
            TestValue::Any => 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchTest {
    str: Vec<u8>,
    n_pos: Option<usize>,
    str_mods: FlagSet<StringMod>,
    re_mods: FlagSet<ReMod>,
    binary: bool,
    cmp_op: CmpOp,
}

impl From<SearchTest> for Test {
    fn from(value: SearchTest) -> Self {
        Self::Search(value)
    }
}

impl SearchTest {
    #[inline]
    fn matches<'buf>(&self, buf: &'buf [u8]) -> Option<(u64, &'buf [u8])> {
        let mut i = 0;

        let Some(needle) = self.str.get(0) else {
            return None;
        };

        while i < buf.len() {
            // we cannot match if the first character isn't the same
            // so we accelerate the search by finding potential matches
            i += memchr(*needle, &buf[i..])?;

            // if we want a full word match
            if self.str_mods.contains(StringMod::FullWordMatch) {
                let prev_is_whitespace = buf
                    .get(i.saturating_sub(1))
                    .map(|c| c.is_ascii_whitespace())
                    .unwrap_or_default();

                // if it is not the first character
                // and its previous character isn't
                // a whitespace. It cannot be a
                // fullword match
                if i > 0 && !prev_is_whitespace {
                    i += 1;
                    continue;
                }
            }

            if let Some(npos) = self.n_pos {
                if i > npos {
                    return None;
                }
            }

            let pos = i;
            let (ok, consumed) = string_match(&self.str, self.str_mods, &buf[i..]);

            if ok {
                return Some((pos as u64, &buf[i..i + consumed]));
            } else {
                i += max(consumed, 1)
            }
        }

        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScalarTest {
    ty: ScalarDataType,
    transform: Option<ScalarTransform>,
    cmp_op: CmpOp,
    test_val: TestValue<Scalar>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FloatTest {
    ty: FloatDataType,
    transform: Option<FloatTransform>,
    cmp_op: CmpOp,
    test_val: TestValue<Float>,
}

// the value read from the haystack we want to match against
// 'buf is the lifetime of the buffer we are scanning
#[derive(Debug, PartialEq)]
enum ReadValue<'buf> {
    Float(u64, Float),
    Scalar(u64, Scalar),
    Bytes(u64, &'buf [u8]),
}

impl DynDisplay for ReadValue<'_> {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        match self {
            Self::Float(_, s) => DynDisplay::dyn_fmt(s, f),
            Self::Scalar(_, s) => DynDisplay::dyn_fmt(s, f),
            Self::Bytes(_, b) => Ok(format!("{:?}", b)),
        }
    }
}

impl DynDisplay for &ReadValue<'_> {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        // Dereference self to get the TestValue and call its fmt method
        DynDisplay::dyn_fmt(*self, f)
    }
}

impl Display for ReadValue<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float(_, v) => write!(f, "{v}"),
            Self::Scalar(_, s) => write!(f, "{s}"),
            Self::Bytes(_, b) => write!(f, "{b:?}"),
        }
    }
}

enum Encoding {
    Utf16(String16Encoding),
    Utf8,
}

// Carry the offset of the start of the data in the stream
// and the data itself
enum MatchRes<'buf> {
    // Bytes.0: offset of the match
    // Bytes.1: optional end of match (to address the need of EOL adjustment in string regex)
    // Bytes.2: the bytes matching
    // Bytes.3: encoding of the buffer
    Bytes(u64, Option<u64>, &'buf [u8], Encoding),
    Scalar(u64, Scalar),
    Float(u64, Float),
}

impl DynDisplay for &MatchRes<'_> {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        (*self).dyn_fmt(f)
    }
}

impl DynDisplay for MatchRes<'_> {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        match self {
            Self::Scalar(_, v) => v.dyn_fmt(f),
            Self::Float(_, v) => v.dyn_fmt(f),
            Self::Bytes(_, _, v, enc) => match enc {
                Encoding::Utf8 => String::from_utf8_lossy(v).to_string().dyn_fmt(f),
                Encoding::Utf16(enc) => {
                    let utf16: Vec<u16> = slice_to_utf16_iter(v, *enc).collect();
                    String::from_utf16_lossy(&utf16).dyn_fmt(f)
                }
            },
        }
    }
}

impl MatchRes<'_> {
    // start offset of the match
    #[inline]
    fn start_offset(&self) -> u64 {
        match self {
            MatchRes::Bytes(o, _, _, _) => *o,
            MatchRes::Scalar(o, _) => *o,
            MatchRes::Float(o, _) => *o,
        }
    }

    // start offset of the match
    #[inline]
    fn end_offset(&self) -> u64 {
        match self {
            MatchRes::Bytes(start, end, buf, _) => match end {
                Some(end) => *end,
                None => start.saturating_add(buf.len() as u64),
            },
            MatchRes::Scalar(o, sc) => o.add(sc.size_of() as u64),
            MatchRes::Float(o, f) => o.add(f.size_of() as u64),
        }
    }
}

fn slice_to_utf16_iter(read: &[u8], encoding: String16Encoding) -> impl Iterator<Item = u16> {
    let even = read
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .map(|t| t.1);

    let odd = read
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 2 != 0)
        .map(|t| t.1);

    even.zip(odd).map(move |(e, o)| match encoding {
        String16Encoding::Le => u16::from_le_bytes([*e, *o]),
        String16Encoding::Be => u16::from_be_bytes([*e, *o]),
    })
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum String16Encoding {
    Le,
    Be,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct String16Test {
    orig: String,
    test_val: TestValue<Vec<u16>>,
    encoding: String16Encoding,
}

impl String16Test {
    /// if the test value is a specific value this method returns
    /// the number of utf16 characters. To obtain the length in
    /// bytes the return value needs to be multiplied by two.
    #[inline(always)]
    fn test_value_len(&self) -> usize {
        match self.test_val.as_ref() {
            TestValue::Value(str16) => str16.len(),
            TestValue::Any => 0,
        }
    }
}

flags! {
    enum IndirectMod: u8{
        Relative,
    }
}

type IndirectMods = FlagSet<IndirectMod>;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum PStringLen {
    Byte,    // B
    ShortBe, // H
    ShortLe, // h
    LongBe,  // L
    LongLe,  // l
}

impl PStringLen {
    #[inline(always)]
    const fn size_of_len(&self) -> usize {
        match self {
            PStringLen::Byte => 1,
            PStringLen::ShortBe => 2,
            PStringLen::ShortLe => 2,
            PStringLen::LongBe => 4,
            PStringLen::LongLe => 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PStringTest {
    len: PStringLen,
    test_val: TestValue<Vec<u8>>,
    include_len: bool,
}

impl PStringTest {
    #[inline]
    fn read<'cache, R: Read + Seek>(
        &self,
        haystack: &'cache mut LazyCache<R>,
    ) -> Result<Option<&'cache [u8]>, Error> {
        let mut len = match self.len {
            PStringLen::Byte => read_le!(haystack, u8) as u32,
            PStringLen::ShortBe => read_be!(haystack, u16) as u32,
            PStringLen::ShortLe => read_le!(haystack, u16) as u32,
            PStringLen::LongBe => read_be!(haystack, u32),
            PStringLen::LongLe => read_le!(haystack, u32),
        } as usize;

        if self.include_len {
            len = len.saturating_sub(self.len.size_of_len())
        }

        if let TestValue::Value(s) = self.test_val.as_ref() {
            if len != s.len() {
                return Ok(None);
            }
        }

        let read = haystack.read_exact_count(len as u64)?;

        Ok(Some(read))
    }

    #[inline(always)]
    fn test_value_len(&self) -> usize {
        match self.test_val.as_ref() {
            TestValue::Value(s) => s.len(),
            TestValue::Any => 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Test {
    Name(String),
    Use(bool, String),
    Scalar(ScalarTest),
    Float(FloatTest),
    String(StringTest),
    Search(SearchTest),
    PString(PStringTest),
    Regex(RegexTest),
    Indirect(FlagSet<IndirectMod>),
    String16(String16Test),
    // FIXME: placeholder for strength computation
    #[allow(dead_code)]
    Der,
    Clear,
    Default,
}

impl Test {
    // read the value to test from the haystack
    #[inline]
    fn read_test_value<'haystack, R: Read + Seek>(
        &self,
        haystack: &'haystack mut LazyCache<R>,
        switch_endianness: bool,
    ) -> Result<Option<ReadValue<'haystack>>, Error> {
        let test_value_offset = haystack.lazy_stream_position();

        match self {
            Self::Scalar(t) => {
                t.ty.read(haystack, switch_endianness)
                    .map(|s| Some(ReadValue::Scalar(test_value_offset, s)))
            }

            Self::Float(t) => {
                t.ty.read(haystack, switch_endianness)
                    .map(|f| Some(ReadValue::Float(test_value_offset, f)))
            }
            Self::String(t) => {
                match t.test_val.as_ref() {
                    TestValue::Value(str) => {
                        let buf = if let Some(length) = t.length {
                            // if there is a length specified
                            let read = haystack.read_exact_count(length as u64)?;
                            read
                        } else {
                            // no length specified we read until end of string
                            let read = match t.cmp_op {
                                CmpOp::Eq | CmpOp::Neq => {
                                    if !t.has_length_mod() {
                                        haystack.read_exact_count(str.len() as u64)?
                                    } else {
                                        haystack.read_count(FILE_BYTES_MAX as u64)?
                                    }
                                }
                                CmpOp::Lt | CmpOp::Gt => {
                                    let read =
                                        haystack.read_until_any_delim_or_limit(b"\n\0", 8092)?;

                                    if read.ends_with(b"\0") || read.ends_with(b"\n") {
                                        &read[..read.len() - 1]
                                    } else {
                                        read
                                    }
                                }
                                _ => {
                                    return Err(Error::Msg(format!(
                                        "string test does not support {:?} operator",
                                        t.cmp_op
                                    )));
                                }
                            };
                            read
                        };

                        Ok(Some(ReadValue::Bytes(test_value_offset, buf)))
                    }
                    TestValue::Any => {
                        let read = haystack.read_until_any_delim_or_limit(b"\0\n", 8192)?;
                        // we don't take last byte if it matches end of string
                        let bytes = if read.ends_with(b"\0") || read.ends_with(b"\n") {
                            &read[..read.len() - 1]
                        } else {
                            read
                        };

                        Ok(Some(ReadValue::Bytes(test_value_offset, bytes)))
                    }
                }
            }

            Self::String16(t) => {
                match t.test_val.as_ref() {
                    TestValue::Value(str16) => {
                        let read = haystack.read_exact_count((str16.len() * 2) as u64)?;

                        Ok(Some(ReadValue::Bytes(test_value_offset, read)))
                    }
                    TestValue::Any => {
                        let read = haystack.read_until_utf16_or_limit(b"\x00\x00", 8192)?;

                        // we make sure we have an even number of elements
                        let end = if read.len() % 2 == 0 {
                            read.len()
                        } else {
                            // we decide to read anyway even though
                            // length isn't even
                            read.len().saturating_sub(1)
                        };

                        Ok(Some(ReadValue::Bytes(test_value_offset, &read[..end])))
                    }
                }
            }

            Self::PString(t) => {
                let Some(read) = t.read(haystack)? else {
                    return Ok(None);
                };
                Ok(Some(ReadValue::Bytes(test_value_offset, read)))
            }

            Self::Search(_) => {
                let buf = haystack.read_count(FILE_BYTES_MAX as u64)?;
                Ok(Some(ReadValue::Bytes(test_value_offset, buf)))
            }

            Self::Regex(r) => {
                let length = {
                    match r.length {
                        Some(len) => {
                            if r.mods.contains(ReMod::LineLimit) {
                                len * 80
                            } else {
                                len
                            }
                        }

                        None => FILE_REGEX_MAX,
                    }
                };

                let read = haystack.read_count(length as u64)?;
                Ok(Some(ReadValue::Bytes(test_value_offset, read)))
            }

            Self::Name(_)
            | Self::Use(_, _)
            | Self::Indirect(_)
            | Self::Clear
            | Self::Default
            | Self::Der => Err(Error::msg("no value to read for this test")),
        }
    }

    #[inline(always)]
    fn match_value<'s>(
        &'s self,
        tv: &ReadValue<'s>,
        stream_kind: StreamKind,
    ) -> Option<MatchRes<'s>> {
        match (self, tv) {
            (Self::Scalar(t), ReadValue::Scalar(o, ts)) => {
                let read_value: Scalar = match t.transform.as_ref() {
                    Some(t) => t.apply(*ts)?,
                    None => *ts,
                };

                match t.test_val {
                    TestValue::Value(test_value) => {
                        let ok = match t.cmp_op {
                            // NOTE: this should not happen in practice because
                            // we convert it into Eq equivalent at parsing time
                            CmpOp::Not => read_value == !test_value,
                            CmpOp::Eq => read_value == test_value,
                            CmpOp::Lt => read_value < test_value,
                            CmpOp::Gt => read_value > test_value,
                            CmpOp::Neq => read_value != test_value,
                            CmpOp::BitAnd => read_value & test_value == test_value,
                            CmpOp::Xor => (read_value & test_value).is_zero(),
                        };

                        if ok {
                            Some(MatchRes::Scalar(*o, read_value))
                        } else {
                            None
                        }
                    }

                    TestValue::Any => Some(MatchRes::Scalar(*o, read_value)),
                }
            }

            (Self::Float(t), ReadValue::Float(o, f)) => {
                let read_value: Float = t.transform.as_ref().map(|t| t.apply(*f)).unwrap_or(*f);

                match t.test_val {
                    TestValue::Value(tf) => {
                        let ok = match t.cmp_op {
                            CmpOp::Eq => read_value == tf,
                            CmpOp::Lt => read_value < tf,
                            CmpOp::Gt => read_value > tf,
                            CmpOp::Neq => read_value != tf,
                            _ => {
                                // this should never be reached as we validate
                                // operator in parser
                                debug_panic!("unsupported float comparison");
                                debug!("unsupported float comparison");
                                false
                            }
                        };

                        if ok {
                            Some(MatchRes::Float(*o, read_value))
                        } else {
                            None
                        }
                    }
                    TestValue::Any => Some(MatchRes::Float(*o, read_value)),
                }
            }

            (Self::String(st), ReadValue::Bytes(o, buf)) => {
                macro_rules! trim_buf {
                    ($buf: expr) => {{
                        if st.mods.contains(StringMod::Trim) {
                            $buf.trim_ascii()
                        } else {
                            $buf
                        }
                    }};
                }

                match st.test_val.as_ref() {
                    TestValue::Value(str) => {
                        match st.cmp_op {
                            CmpOp::Eq => {
                                if let (true, _) = string_match(str, st.mods, buf) {
                                    Some(MatchRes::Bytes(*o, None, trim_buf!(str), Encoding::Utf8))
                                } else {
                                    None
                                }
                            }
                            CmpOp::Neq => {
                                if let (false, _) = string_match(str, st.mods, buf) {
                                    Some(MatchRes::Bytes(*o, None, trim_buf!(str), Encoding::Utf8))
                                } else {
                                    None
                                }
                            }
                            CmpOp::Gt => {
                                if buf.len() > str.len() {
                                    Some(MatchRes::Bytes(*o, None, trim_buf!(buf), Encoding::Utf8))
                                } else {
                                    None
                                }
                            }
                            CmpOp::Lt => {
                                if buf.len() < str.len() {
                                    Some(MatchRes::Bytes(*o, None, trim_buf!(buf), Encoding::Utf8))
                                } else {
                                    None
                                }
                            }

                            // unsupported for strings
                            _ => {
                                // this should never be reached as we validate
                                // operator in parser
                                debug_panic!("unsupported string comparison");
                                debug!("unsupported string comparison");
                                None
                            }
                        }
                    }
                    TestValue::Any => {
                        Some(MatchRes::Bytes(*o, None, trim_buf!(buf), Encoding::Utf8))
                    }
                }
            }

            (Self::PString(m), ReadValue::Bytes(o, buf)) => match m.test_val.as_ref() {
                TestValue::Value(psv) => {
                    if buf == psv {
                        Some(MatchRes::Bytes(*o, None, buf, Encoding::Utf8))
                    } else {
                        None
                    }
                }
                TestValue::Any => Some(MatchRes::Bytes(*o, None, buf, Encoding::Utf8)),
            },

            (Self::String16(t), ReadValue::Bytes(o, buf)) => {
                match t.test_val.as_ref() {
                    TestValue::Value(str16) => {
                        // strings cannot be equal
                        if str16.len() * 2 != buf.len() {
                            return None;
                        }

                        // we check string equality
                        for (i, utf16_char) in slice_to_utf16_iter(buf, t.encoding).enumerate() {
                            if str16[i] != utf16_char {
                                return None;
                            }
                        }

                        Some(MatchRes::Bytes(
                            *o,
                            None,
                            &t.orig.as_bytes(),
                            Encoding::Utf16(t.encoding),
                        ))
                    }

                    TestValue::Any => {
                        Some(MatchRes::Bytes(*o, None, buf, Encoding::Utf16(t.encoding)))
                    }
                }
            }

            (Self::Regex(r), ReadValue::Bytes(o, buf)) => {
                match stream_kind {
                    StreamKind::Text(_) => {
                        let mut offset = *o;

                        let mut line_limit = r.length.unwrap_or(usize::MAX);

                        for line in buf.split(|c| c == &b'\n') {
                            // we don't need to break on offset
                            // limit as buf contains the good amount
                            // of bytes to match against
                            if line_limit == 0 {
                                break;
                            }

                            if let Some(re_match) = r.re.find(&line) {
                                // the offset of the string is computed from the start of the buffer
                                let start_offset = offset + re_match.start() as u64;

                                // if we matched until EOL we need to add one to include the delimiter removed from the split
                                let stop_offset = if re_match.end() == line.len() {
                                    Some(start_offset + re_match.as_bytes().len() as u64 + 1)
                                } else {
                                    None
                                };

                                return Some(MatchRes::Bytes(
                                    start_offset,
                                    stop_offset,
                                    re_match.as_bytes(),
                                    Encoding::Utf8,
                                ));
                            }

                            offset += line.len() as u64;
                            // we have to add one because lines do not contain splitting character
                            offset += 1;
                            line_limit = line_limit.saturating_sub(1)
                        }
                        None
                    }

                    StreamKind::Binary => {
                        if let Some(re_match) = r.re.find(&buf) {
                            Some(MatchRes::Bytes(
                                // the offset of the string is computed from the start of the buffer
                                o + re_match.start() as u64,
                                None,
                                re_match.as_bytes(),
                                Encoding::Utf8,
                            ))
                        } else {
                            None
                        }
                    }
                }
            }

            (Self::Search(t), ReadValue::Bytes(o, buf)) => {
                // the offset of the string is computed from the start of the buffer
                t.matches(&buf)
                    .map(|(p, m)| MatchRes::Bytes(o + p, None, m, Encoding::Utf8))
            }

            _ => None,
        }
    }

    #[inline(always)]
    fn strength(&self) -> u64 {
        const MULT: usize = 10;

        let mut out = 2 * MULT;

        // FIXME: octal is missing but it is not used in practice ...
        match self {
            Test::Default => return 0,

            Test::Scalar(s) => match s.ty {
                _ => {
                    out += s.ty.type_size() * MULT;
                }
            },

            Test::Float(t) => match t.ty {
                _ => {
                    out += t.ty.type_size() * MULT;
                }
            },

            Test::String(t) => out += t.test_value_len().saturating_add(MULT),

            Test::PString(t) => out += t.test_value_len().saturating_add(MULT),

            Test::Search(s) => {
                // NOTE: this is how it is implemented in libmagic
                // but it seems odd to have strength decreasing as
                // search str len increase, as smaller search tend
                // to be less reliable.
                out += s.str.len() * max(MULT.checked_div(s.str.len()).unwrap_or_default(), 1)
            }

            Test::Regex(r) => {
                let v = r.non_magic_len;
                // NOTE: this is how it is implemented in libmagic
                // but it seems odd to have strength decreasing as
                // regex str len increase, as smaller regex tend
                // to be less reliable.
                out += v * max(MULT.checked_div(v).unwrap_or_default(), 1);
            }

            Test::String16(t) => {
                // NOTE: in libmagic the result is div by 2
                // but I GUESS it is because the len is expressed
                // in number bytes. In our case length is expressed
                // in number of u16 so we shouldn't divide.
                out += t.test_value_len().saturating_mul(MULT);
            }

            Test::Der => out += MULT,

            _ => {}
        }

        // matching any output gets penalty
        if self.is_match_any() {
            out = 0
        }

        if let Some(op) = self.cmp_op() {
            match op {
                // matching almost any gets penalty
                CmpOp::Neq => out = 0,
                CmpOp::Eq | CmpOp::Not => out += MULT,
                CmpOp::Lt | CmpOp::Gt => out -= 2 * MULT,
                CmpOp::Xor | CmpOp::BitAnd => out -= MULT,
            }
        }

        out as u64
    }

    #[inline(always)]
    fn cmp_op(&self) -> Option<CmpOp> {
        match self {
            Self::String(t) => Some(t.cmp_op),
            Self::Scalar(s) => Some(s.cmp_op),
            Self::Float(t) => Some(t.cmp_op),
            Self::Name(_)
            | Self::Use(_, _)
            | Self::Search(_)
            | Self::PString(_)
            | Self::Regex(_)
            | Self::Clear
            | Self::Default
            | Self::Indirect(_)
            | Self::String16(_)
            | Self::Der => None,
        }
    }

    #[inline(always)]
    fn is_match_any(&self) -> bool {
        match self {
            Test::Name(_) => false,
            Test::Use(_, _) => false,
            Test::Scalar(scalar_test) => matches!(scalar_test.test_val, TestValue::Any),
            Test::Float(float_test) => matches!(float_test.test_val, TestValue::Any),
            Test::String(string_test) => matches!(string_test.test_val, TestValue::Any),
            Test::Search(_) => false,
            Test::PString(pstring_test) => matches!(pstring_test.test_val, TestValue::Any),
            Test::Regex(_) => false,
            Test::Indirect(_) => false,
            Test::String16(string16_test) => matches!(string16_test.test_val, TestValue::Any),
            Test::Der => false,
            Test::Clear => false,
            Test::Default => false,
        }
    }

    #[inline(always)]
    fn is_binary(&self) -> bool {
        match self {
            Self::Name(_) => true,
            Self::Use(_, _) => true,
            Self::Scalar(_) => true,
            Self::Float(_) => true,
            Self::String(_) => true,
            Self::Search(_) => true,
            Self::PString(_) => true,
            Self::Regex(_) => true,
            Self::Clear => true,
            Self::Default => true,
            Self::Indirect(_) => true,
            Self::String16(_) => true,
            Self::Der => true,
        }
    }

    #[inline(always)]
    fn is_text(&self) -> bool {
        match self {
            Self::Name(_) => true,
            Self::Use(_, _) => true,
            Self::Indirect(_) => true,
            Self::String(_) => true,
            Self::Search(_) => true,
            Self::Regex(_) => true,
            Self::Clear => true,
            Self::Default => true,
            _ => !self.is_binary(),
        }
    }

    #[inline(always)]
    fn is_only_text(&self) -> bool {
        self.is_text() && !self.is_binary()
    }

    #[inline(always)]
    fn is_only_binary(&self) -> bool {
        self.is_binary() && !self.is_text()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum OffsetType {
    Byte,
    DoubleLe,
    DoubleBe,
    ShortLe,
    ShortBe,
    Id3Le,
    Id3Be,
    LongLe,
    LongBe,
    Middle,
    Octal,
    QuadBe,
    QuadLe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum Shift {
    Direct(u64),
    Indirect(i64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct IndOffset {
    // where to find the offset
    off_addr: DirOffset,
    // signed or unsigned
    signed: bool,
    // type of the offset
    ty: OffsetType,
    op: Option<Op>,
    shift: Option<Shift>,
}

impl IndOffset {
    // if we overflow we must not return an offset
    fn read_offset<R: Read + Seek>(
        &self,
        haystack: &mut LazyCache<R>,
        rule_base_offset: Option<u64>,
        last_upper_match_offset: Option<u64>,
    ) -> Result<Option<u64>, io::Error> {
        let offset_address = match self.off_addr {
            DirOffset::Start(s) => {
                let Some(o) = s.checked_add(rule_base_offset.unwrap_or_default()) else {
                    return Ok(None);
                };

                haystack.seek(SeekFrom::Start(o))?
            }
            DirOffset::LastUpper(c) => haystack.seek(SeekFrom::Start(
                (last_upper_match_offset.unwrap_or_default() as i64 + c) as u64,
            ))?,
            DirOffset::End(e) => haystack.seek(SeekFrom::End(e as i64))?,
        };

        macro_rules! read_value {
            () => {
                match self.ty {
                    OffsetType::Byte => {
                        if self.signed {
                            read_le!(haystack, u8) as u64
                        } else {
                            read_le!(haystack, i8) as u64
                        }
                    }
                    OffsetType::DoubleLe => read_le!(haystack, f64) as u64,
                    OffsetType::DoubleBe => read_be!(haystack, f64) as u64,
                    OffsetType::ShortLe => {
                        if self.signed {
                            read_le!(haystack, i16) as u64
                        } else {
                            read_le!(haystack, u16) as u64
                        }
                    }
                    OffsetType::ShortBe => {
                        if self.signed {
                            read_be!(haystack, i16) as u64
                        } else {
                            read_be!(haystack, u16) as u64
                        }
                    }
                    OffsetType::Id3Le => decode_id3(read_le!(haystack, u32)) as u64,
                    OffsetType::Id3Be => decode_id3(read_be!(haystack, u32)) as u64,
                    OffsetType::LongLe => {
                        if self.signed {
                            read_le!(haystack, i32) as u64
                        } else {
                            read_le!(haystack, u32) as u64
                        }
                    }
                    OffsetType::LongBe => {
                        if self.signed {
                            read_be!(haystack, i32) as u64
                        } else {
                            read_be!(haystack, u32) as u64
                        }
                    }
                    OffsetType::Middle => read_me!(haystack) as u64,
                    OffsetType::Octal => {
                        if let Some(o) = read_octal_u64(haystack) {
                            o
                        } else {
                            debug!("failed to read octal offset @ {offset_address}");
                            return Ok(None);
                        }
                    }
                    OffsetType::QuadLe => {
                        if self.signed {
                            read_le!(haystack, i64) as u64
                        } else {
                            read_le!(haystack, u64)
                        }
                    }
                    OffsetType::QuadBe => {
                        if self.signed {
                            read_be!(haystack, i64) as u64
                        } else {
                            read_be!(haystack, u64)
                        }
                    }
                }
            };
        }

        // in theory every offset read should end up in something seekable from start, so we can use u64 to store the result
        let o = read_value!();

        trace!(
            "offset read @ {offset_address} value={o} op={:?} shift={:?}",
            self.op, self.shift
        );

        // apply transformation
        if let (Some(op), Some(shift)) = (self.op, self.shift) {
            let shift = match shift {
                Shift::Direct(i) => i,
                Shift::Indirect(i) => {
                    let tmp = offset_address as i128 + i as i128;
                    if tmp.is_negative() {
                        return Ok(None);
                    } else {
                        haystack.seek(SeekFrom::Start(tmp as u64))?;
                    };
                    // NOTE: here we assume that the shift has the same
                    // type as the main offset !
                    read_value!()
                }
            };

            match op {
                Op::Add => return Ok(o.checked_add(shift)),
                Op::Mul => return Ok(o.checked_mul(shift)),
                Op::Sub => return Ok(o.checked_sub(shift)),
                Op::Div => return Ok(o.checked_div(shift)),
                Op::Mod => return Ok(o.checked_rem(shift)),
                Op::And => return Ok(Some(o & shift)),
                Op::Or => return Ok(Some(o | shift)),
                Op::Xor => return Ok(Some(o ^ shift)),
            }
        }

        Ok(Some(o))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum DirOffset {
    Start(u64),
    // relative to the last up-level field
    LastUpper(i64),
    End(i64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum Offset {
    Direct(DirOffset),
    Indirect(IndOffset),
}

impl From<DirOffset> for Offset {
    fn from(value: DirOffset) -> Self {
        Self::Direct(value)
    }
}

impl From<IndOffset> for Offset {
    fn from(value: IndOffset) -> Self {
        Self::Indirect(value)
    }
}

impl Display for DirOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DirOffset::Start(i) => write!(f, "{i}"),
            DirOffset::LastUpper(c) => write!(f, "&{c}"),
            DirOffset::End(e) => write!(f, "-{e}"),
        }
    }
}

impl Default for DirOffset {
    fn default() -> Self {
        Self::LastUpper(0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Match {
    line: usize,
    depth: u8,
    offset: Offset,
    test: Test,
    test_strength: u64,
    message: Option<Message>,
}

impl From<Use> for Match {
    fn from(value: Use) -> Self {
        let test = Test::Use(value.switch_endianness, value.rule_name);
        let test_strength = test.strength();
        Self {
            line: value.line,
            depth: value.depth,
            offset: value.start_offset,
            test,
            test_strength,
            message: value.message,
        }
    }
}

impl From<Name> for Match {
    fn from(value: Name) -> Self {
        let test = Test::Name(value.name);
        let test_strength = test.strength();
        Self {
            line: value.line,
            depth: 0,
            offset: Offset::Direct(DirOffset::Start(0)),
            test,
            test_strength,
            message: value.message,
        }
    }
}

impl Match {
    /// Turns the `Match`'s offset into an absolute offset from the start of the stream
    #[inline(always)]
    fn offset_from_start<'a, R: Read + Seek>(
        &self,
        haystack: &mut LazyCache<R>,
        rule_base_offset: Option<u64>,
        last_level_offset: Option<u64>,
    ) -> Result<Option<u64>, io::Error> {
        match self.offset {
            Offset::Direct(dir_offset) => match dir_offset {
                DirOffset::Start(s) => Ok(Some(s)),
                DirOffset::LastUpper(shift) => {
                    let o = last_level_offset.unwrap_or_default() as i64 + shift;

                    if o.is_positive() {
                        Ok(Some(o as u64))
                    } else {
                        Ok(None)
                    }
                }
                DirOffset::End(e) => Ok(Some(haystack.offset_from_start(SeekFrom::End(e)))),
            },
            Offset::Indirect(ind_offset) => {
                let Some(o) =
                    ind_offset.read_offset(haystack, rule_base_offset, last_level_offset)?
                else {
                    return Ok(None);
                };

                Ok(Some(o))
            }
        }
    }

    /// this method emulates the buffer based matching
    /// logic implemented in libmagic. It needs some aweful
    /// and weird offset convertions to turn buffer
    /// relative offsets (libmagic is based on) into
    /// absolute offset in the file.
    ///
    /// this method shoud bubble up only critical errors
    /// all the other errors should make the match result
    /// false and be logged via debug!
    ///
    /// the function returns an error if the maximum recursion
    /// has been reached or if a dependency rule is missing.
    #[inline]
    fn matches<'a: 'h, 'h, R: Read + Seek>(
        &'a self,
        source: Option<&str>,
        magic: &mut Magic<'a>,
        stream_kind: StreamKind,
        state: &mut MatchState,
        buf_base_offset: Option<u64>,
        rule_base_offset: Option<u64>,
        last_level_offset: Option<u64>,
        haystack: &'h mut LazyCache<R>,
        switch_endianness: bool,
        db: &'a MagicDb,
        depth: usize,
    ) -> Result<(bool, Option<MatchRes<'h>>), Error> {
        let source = source.unwrap_or("unknown");
        let line = self.line;

        if depth >= MAX_RECURSION {
            return Err(Error::localized(
                source,
                line,
                Error::MaximumRecursion(MAX_RECURSION),
            ));
        }

        if self.test.is_only_binary() && stream_kind.is_text() {
            trace!("skip binary test source={source} line={line} stream_kind={stream_kind:?}",);
            return Ok((false, None));
        }

        if self.test.is_only_text() && !stream_kind.is_text() {
            trace!("skip text test source={source} line={line} stream_kind={stream_kind:?}",);
            return Ok((false, None));
        }

        let Ok(Some(mut offset)) = self
            .offset_from_start(haystack, rule_base_offset, last_level_offset)
            .inspect_err(|e| debug!("source={source} line={line} failed at computing offset: {e}"))
        else {
            return Ok((false, None));
        };

        offset = match self.offset {
            Offset::Indirect(_) => {
                // the result we get for an indirect offset
                // is relative to the start of the libmagic
                // buffer so we need to add base to make it
                // absolute.
                buf_base_offset.unwrap_or_default().saturating_add(offset)
            }
            // offset from start are computed from rule base
            Offset::Direct(DirOffset::Start(_)) => {
                rule_base_offset.unwrap_or_default().saturating_add(offset)
            }
            _ => offset,
        };

        match &self.test {
            Test::Clear => {
                trace!("source={source} line={line} clear");
                state.clear_continuation_level(&self.continuation_level());
                Ok((true, None))
            }

            Test::Name(name) => {
                trace!(
                    "source={source} line={line} running rule {name} switch_endianness={switch_endianness}",
                );
                Ok((true, None))
            }

            Test::Use(flip_endianness, rule_name) => {
                trace!(
                    "source={source} line={line} use {rule_name} switch_endianness={flip_endianness}",
                );

                // switch_endianness must propagate down the rule call stack
                let switch_endianness = switch_endianness ^ flip_endianness;

                let dr: &DependencyRule = db.dependencies.get(rule_name).ok_or(
                    Error::localized(source, line, Error::MissingRule(rule_name.clone())),
                )?;

                // we push the message here otherwise we push message in depth first
                if let Some(msg) = self.message.as_ref() {
                    magic.push_message(msg.to_string_lossy());
                }

                dr.rule.magic(
                    magic,
                    stream_kind,
                    buf_base_offset,
                    Some(offset),
                    haystack,
                    db,
                    switch_endianness,
                    depth.saturating_add(1),
                )?;

                // we return false not to push message again
                Ok((false, None))
            }

            Test::Indirect(m) => {
                trace!(
                    "source={source} line={line} indirect mods={:?} offset={offset:#x}",
                    m
                );

                let new_buf_base_off = if m.contains(IndirectMod::Relative) {
                    Some(offset)
                } else {
                    None
                };

                // we push the message here otherwise we push message in depth first
                if let Some(msg) = self.message.as_ref() {
                    magic.push_message(msg.to_string_lossy());
                }

                for (_, r) in db.rules.iter() {
                    let messages_cnt = magic.message.len();

                    r.magic(
                        magic,
                        stream_kind,
                        new_buf_base_off,
                        Some(offset),
                        haystack,
                        db,
                        false,
                        depth.saturating_add(1),
                    )?;

                    // this means we matched a rule
                    if magic.message.len() != messages_cnt {
                        break;
                    }
                }

                // we return false not to push message again
                Ok((false, None))
            }

            Test::Default => {
                // default matches if nothing else at the continuation level matched
                let ok = !state.get_continuation_level(&self.continuation_level());

                trace!("source={source} line={line} default match={ok}");
                if ok {
                    state.set_continuation_level(self.continuation_level());
                }

                Ok((ok, None))
            }

            _ => {
                if let Err(e) = haystack.seek(SeekFrom::Start(offset)) {
                    debug!("source={source} line={line} failed to seek in haystack: {e}");
                    return Ok((false, None));
                }

                let mut trace_msg = None;

                if enabled!(Level::DEBUG) {
                    trace_msg = Some(vec![format!(
                        "source={source} line={line} stream_offset={:#x}",
                        haystack.lazy_stream_position()
                    )])
                }

                // NOTE: we may have a way to optimize here. In case we do a Any
                // test and we don't use the value to format the message, we don't
                // need to read the value.
                if let Ok(opt_test_value) = self
                    .test
                    .read_test_value(haystack, switch_endianness)
                    .inspect_err(|e| {
                        debug!("source={source} line={line} error while reading test value @{offset}: {e}",)
                    })
                {
                    trace_msg
                        .as_mut()
                        .map(|v| v.push(format!("test={:?}", self.test)));

                    let match_res =
                        opt_test_value.and_then(|tv| self.test.match_value(&tv, stream_kind));

                    trace_msg.as_mut().map(|v| {
                        v.push(format!(
                            "message=\"{}\" match={}",
                            self.message
                                .as_ref()
                                .map(|fs| fs.to_string_lossy())
                                .unwrap_or_default(),
                            match_res.is_some()
                        ))
                    });

                    // trace message
                    if enabled!(Level::DEBUG) && !enabled!(Level::TRACE) && match_res.is_some() {
                        trace_msg.map(|m| debug!("{}", m.join(" ")));
                    } else if enabled!(Level::TRACE) {
                        trace_msg.map(|m| trace!("{}", m.join(" ")));
                    }

                    if let Some(mr) = match_res {
                        state.set_continuation_level(self.continuation_level());
                        return Ok((true, Some(mr)));
                    }
                }

                Ok((false, None))
            }
        }
    }

    #[inline(always)]
    fn continuation_level(&self) -> ContinuationLevel {
        ContinuationLevel(self.depth)
    }
}

#[derive(Debug, Clone)]
pub struct Use {
    line: usize,
    depth: u8,
    start_offset: Offset,
    rule_name: String,
    switch_endianness: bool,
    message: Option<Message>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StrengthMod {
    op: Op,
    by: u8,
}

impl StrengthMod {
    #[inline(always)]
    fn apply(&self, strength: u64) -> u64 {
        let by = self.by as u64;
        debug!("applying strength modifier: {strength} {} {}", self.op, by);
        match self.op {
            Op::Mul => strength.saturating_mul(by),
            Op::Add => strength.saturating_add(by),
            Op::Sub => strength.saturating_sub(by),
            Op::Div => {
                if by > 0 {
                    strength.saturating_div(by)
                } else {
                    strength
                }
            }
            Op::Mod => strength % by,
            Op::And => strength & by,
            // this should never happen as strength operators
            // are enforced by our parser
            Op::Xor | Op::Or => {
                debug_panic!("unsupported strength operator");
                strength
            }
        }
    }
}

#[derive(Debug, Clone)]
enum Flag {
    Mime(String),
    Ext(HashSet<String>),
    Strength(StrengthMod),
    Apple(String),
}

#[derive(Debug, Clone)]
struct Name {
    line: usize,
    name: String,
    message: Option<Message>,
}

#[derive(Debug, Clone)]
enum Entry<'span> {
    Match(Span<'span>, Match),
    Flag(Span<'span>, Flag),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntryNode {
    root: bool,
    entry: Match,
    children: Vec<EntryNode>,
    mimetype: Option<String>,
    apple: Option<String>,
    strength_mod: Option<StrengthMod>,
    exts: HashSet<String>,
    // if root exts contains all children extensions
    children_exts: HashSet<String>,
}

impl EntryNode {
    fn children_exts_recursive(&self, exts: &mut HashSet<String>) {
        for c in self.children.iter() {
            for ext in c.exts.iter() {
                if !exts.contains(ext) {
                    exts.insert(ext.clone());
                }
            }
            c.children_exts_recursive(exts);
        }
    }

    #[inline]
    fn matches<'r, R: Read + Seek>(
        &'r self,
        opt_source: Option<&str>,
        magic: &mut Magic<'r>,
        state: &mut MatchState,
        stream_kind: StreamKind,
        buf_base_offset: Option<u64>,
        rule_base_offset: Option<u64>,
        last_level_offset: Option<u64>,
        haystack: &mut LazyCache<R>,
        db: &'r MagicDb,
        switch_endianness: bool,
        depth: usize,
    ) -> Result<(), Error> {
        let (ok, opt_match_res) = self.entry.matches(
            opt_source,
            magic,
            stream_kind,
            state,
            buf_base_offset,
            rule_base_offset,
            last_level_offset,
            haystack,
            switch_endianness,
            db,
            depth,
        )?;

        let source = opt_source.unwrap_or("unknown");
        let line = self.entry.line;

        if ok {
            // update magic with message if match is successful
            if let Some(msg) = self.entry.message.as_ref() {
                if let Ok(msg) = msg.format_with(opt_match_res.as_ref()).inspect_err(|e| {
                    debug!("source={source} line={line} failed to format message: {e}")
                }) {
                    magic.push_message(msg);
                }
            }

            // we need to adjust stream offset in case of regex/search tests
            if let Some(mr) = opt_match_res {
                match &self.entry.test {
                    Test::String(t) => {
                        if t.has_length_mod() {
                            let o = mr.end_offset();
                            haystack.seek(SeekFrom::Start(o))?;
                        }
                    }
                    Test::Search(t) => {
                        if t.re_mods.contains(ReMod::StartOffsetUpdate) {
                            let o = mr.start_offset();
                            haystack.seek(SeekFrom::Start(o))?;
                        } else {
                            let o = mr.end_offset();
                            haystack.seek(SeekFrom::Start(o))?;
                        }
                    }

                    Test::Regex(t) => {
                        if t.mods.contains(ReMod::StartOffsetUpdate) {
                            let o = mr.start_offset();
                            haystack.seek(SeekFrom::Start(o))?;
                        } else {
                            let o = mr.end_offset();
                            haystack.seek(SeekFrom::Start(o))?;
                        }
                    }
                    // other types do not need offset adjustement
                    _ => {}
                }
            }

            if let Some(mimetype) = self.mimetype.as_ref() {
                magic.set_mime_type(Cow::Borrowed(mimetype));
            }

            if let Some(apple_ty) = self.apple.as_ref() {
                magic.set_creator_code(Cow::Borrowed(apple_ty));
            }

            if !self.exts.is_empty() {
                magic.insert_extensions(self.exts.iter().map(|s| s.as_str()));
            }

            // NOTE: here we try to implement a similar logic as in file_magic_strength.
            // Sticking to the exact same strength computation logic is complicated due
            // to implementation differences. Let's wait and see if that is a real issue.
            let mut strength = self.entry.test_strength;

            let continuation_level = self.entry.continuation_level().0 as u64;
            if self.entry.message.is_none() && continuation_level < 3 {
                strength = strength.saturating_add(continuation_level);
            }

            if let Some(sm) = self.strength_mod.as_ref() {
                strength = sm.apply(strength);
            }

            // entries with no message get a bonus
            if self.entry.message.is_none() {
                strength += 1
            }

            magic.update_strength(strength);

            let end_upper_level = haystack.lazy_stream_position();

            // we have to fix rule_base_offset if
            // the rule_base_starts from end otherwise it
            // breaks some offset computation in match
            // see test_offset_bug_1 and test_offset_bug_2
            // they implement the same test logic yet indirect
            // offsets have to be different so that it works
            // in libmagic/file
            let rule_base_offset = if self.root {
                match self.entry.offset {
                    Offset::Direct(DirOffset::End(o)) => {
                        Some(haystack.offset_from_start(SeekFrom::End(o)))
                    }
                    _ => rule_base_offset,
                }
            } else {
                rule_base_offset
            };

            for e in self.children.iter() {
                e.matches(
                    opt_source,
                    magic,
                    state,
                    stream_kind,
                    buf_base_offset,
                    rule_base_offset,
                    Some(end_upper_level),
                    haystack,
                    db,
                    switch_endianness,
                    depth,
                )?
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagicRule {
    source: Option<String>,
    entries: EntryNode,
}

impl MagicRule {
    #[inline]
    fn magic_entrypoint<'r, R: Read + Seek>(
        &'r self,
        magic: &mut Magic<'r>,
        stream_kind: StreamKind,
        haystack: &mut LazyCache<R>,
        db: &'r MagicDb,
        switch_endianness: bool,
        depth: usize,
    ) -> Result<(), Error> {
        self.entries.matches(
            self.source.as_ref().map(|s| s.as_str()),
            magic,
            &mut MatchState::empty(),
            stream_kind,
            None,
            None,
            None,
            haystack,
            db,
            switch_endianness,
            depth,
        )
    }

    #[inline]
    fn magic<'r, R: Read + Seek>(
        &'r self,
        magic: &mut Magic<'r>,
        stream_kind: StreamKind,
        buf_base_offset: Option<u64>,
        rule_base_offset: Option<u64>,
        haystack: &mut LazyCache<R>,
        db: &'r MagicDb,
        switch_endianness: bool,
        depth: usize,
    ) -> Result<(), Error> {
        self.entries.matches(
            self.source.as_ref().map(|s| s.as_str()),
            magic,
            &mut MatchState::empty(),
            stream_kind,
            buf_base_offset,
            rule_base_offset,
            None,
            haystack,
            db,
            switch_endianness,
            depth,
        )
    }

    fn is_text(&self) -> bool {
        self.entries.entry.test.is_text()
            && self.entries.children.iter().all(|e| e.entry.test.is_text())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyRule {
    name: String,
    rule: MagicRule,
}

impl DependencyRule {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn rule(&self) -> &MagicRule {
        &self.rule
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagicFile {
    rules: Vec<MagicRule>,
    dependencies: HashMap<String, DependencyRule>,
}

impl MagicFile {
    pub fn open<P: AsRef<Path>>(p: P) -> Result<Self, Error> {
        FileMagicParser::parse_file(p)
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
struct ContinuationLevel(u8);

// FIXME: magic handles many more text encodings
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum TextEncoding {
    Ascii,
    Utf8,
}

impl TextEncoding {
    const fn as_magic_str(&self) -> &'static str {
        match self {
            TextEncoding::Ascii => "ASCII",
            TextEncoding::Utf8 => "UTF-8",
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum StreamKind {
    Binary,
    Text(TextEncoding),
}

impl StreamKind {
    const fn is_text(&self) -> bool {
        matches!(self, StreamKind::Text(_))
    }
}

#[derive(Debug)]
struct MatchState {
    continuation_levels: [bool; 256],
}

impl MatchState {
    #[inline(always)]
    fn empty() -> Self {
        MatchState {
            continuation_levels: [false; 256],
        }
    }

    #[inline(always)]
    fn get_continuation_level(&mut self, level: &ContinuationLevel) -> bool {
        self.continuation_levels
            .get(level.0 as usize)
            .cloned()
            .unwrap_or_default()
    }

    #[inline(always)]
    fn set_continuation_level(&mut self, level: ContinuationLevel) {
        self.continuation_levels
            .get_mut(level.0 as usize)
            .map(|b| *b = true);
    }

    #[inline(always)]
    fn clear_continuation_level(&mut self, level: &ContinuationLevel) {
        self.continuation_levels
            .get_mut(level.0 as usize)
            .map(|b| *b = false);
    }
}

#[derive(Debug, Default)]
pub struct Magic<'m> {
    stream_kind: Option<StreamKind>,
    source: Option<Cow<'m, str>>,
    message: Vec<Cow<'m, str>>,
    mime_type: Option<Cow<'m, str>>,
    creator_code: Option<Cow<'m, str>>,
    strength: Option<u64>,
    exts: HashSet<Cow<'m, str>>,
    is_default: bool,
}

impl<'m> Magic<'m> {
    #[inline(always)]
    fn set_source(&mut self, source: Option<&'m str>) {
        self.source = source.map(|s| Cow::Borrowed(s));
    }

    #[inline(always)]
    fn set_stream_kind(&mut self, stream_kind: StreamKind) {
        self.stream_kind = Some(stream_kind)
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.stream_kind = None;
        self.source = None;
        self.message.clear();
        self.mime_type = None;
        self.creator_code = None;
        self.strength = None;
        self.exts.clear();
        self.is_default = false;
    }

    #[inline]
    pub fn into_owned<'owned>(self) -> Magic<'owned> {
        Magic {
            stream_kind: self.stream_kind,
            source: self.source.map(|s| Cow::Owned(s.into_owned())),
            message: self
                .message
                .into_iter()
                .map(Cow::into_owned)
                .map(Cow::Owned)
                .collect(),
            mime_type: self.mime_type.map(|m| Cow::Owned(m.into_owned())),
            creator_code: self.creator_code.map(|m| Cow::Owned(m.into_owned())),
            strength: self.strength,
            exts: self
                .exts
                .into_iter()
                .map(|e| Cow::Owned(e.into_owned()))
                .collect(),
            is_default: self.is_default,
        }
    }

    #[inline(always)]
    pub fn message(&self) -> String {
        let mut out = String::new();
        for (i, m) in self.message.iter().enumerate() {
            if let Some(s) = m.strip_prefix(r#"\b"#) {
                out.push_str(s);
            } else {
                // don't put space on first string
                if i > 0 {
                    out.push(' ');
                }
                out.push_str(m);
            }
        }
        out
    }

    #[inline(always)]
    pub fn update_strength(&mut self, value: u64) {
        match self.strength.as_mut() {
            Some(s) => *s = (*s).saturating_add(value),
            None => self.strength = Some(value),
        }
        debug!("updated strength = {:?}", self.strength)
    }

    #[inline(always)]
    pub fn mime_type(&self) -> &str {
        self.mime_type.as_deref().unwrap_or(match self.stream_kind {
            Some(StreamKind::Text(_)) => DEFAULT_TEXT_MIMETYPE,
            Some(StreamKind::Binary) | None => DEFAULT_BIN_MIMETYPE,
        })
    }

    #[inline(always)]
    fn push_message<'a: 'm>(&mut self, msg: Cow<'a, str>) {
        if !msg.is_empty() {
            debug!("pushing message: msg={msg} len={}", msg.len());
            self.message.push(msg);
        }
    }

    #[inline(always)]
    fn set_mime_type<'a: 'm>(&mut self, mime: Cow<'a, str>) {
        if self.mime_type.is_none() {
            debug!("insert mime: {:?}", mime);
            self.mime_type = Some(mime)
        }
    }

    #[inline(always)]
    fn set_creator_code<'a: 'm>(&mut self, apple_ty: Cow<'a, str>) {
        if self.creator_code.is_none() {
            debug!("insert apple type: {apple_ty:?}");
            self.creator_code = Some(apple_ty)
        }
    }

    #[inline(always)]
    fn insert_extensions<'a: 'm, I: Iterator<Item = &'a str>>(&mut self, exts: I) {
        if self.exts.is_empty() {
            self.exts.extend(exts.map(|e| Cow::Borrowed(e)))
        }
    }

    #[inline(always)]
    pub fn is_default(&self) -> bool {
        self.is_default
    }

    #[inline(always)]
    pub fn strength(&self) -> Option<u64> {
        self.strength
    }

    #[inline(always)]
    pub fn source(&self) -> Option<&Cow<'m, str>> {
        self.source.as_ref()
    }

    #[inline(always)]
    pub fn creator_code(&self) -> Option<&Cow<'m, str>> {
        self.creator_code.as_ref()
    }

    #[inline(always)]
    pub fn exts(&self) -> &HashSet<Cow<'m, str>> {
        &self.exts
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct SerializedDb {
    rules: Vec<(usize, MagicRule)>,
    dependencies: HashMap<String, DependencyRule>,
}

impl From<MagicDb> for SerializedDb {
    #[inline]
    fn from(mut value: MagicDb) -> Self {
        // we decrease the Rc so that we don't panic on try_unwrap
        value.by_ext.clear();

        Self {
            rules: value
                .rules
                .into_iter()
                .map(|(i, r)| (i, Rc::try_unwrap(r).unwrap()))
                .collect(),
            dependencies: value.dependencies,
        }
    }
}

impl From<SerializedDb> for MagicDb {
    #[inline]
    fn from(value: SerializedDb) -> Self {
        let rules: Vec<(usize, Rc<MagicRule>)> = value
            .rules
            .into_iter()
            .map(|(i, r)| (i, Rc::new(r)))
            .collect();

        let mut by_ext = HashMap::new();
        for (i, r) in rules.iter() {
            for ext in r.entries.exts.iter() {
                by_ext
                    .entry(ext.to_lowercase())
                    .and_modify(|v: &mut Vec<(usize, Rc<MagicRule>)>| v.push((*i, Rc::clone(r))))
                    .or_insert_with(|| vec![(*i, Rc::clone(r))]);
            }
        }

        Self {
            rules,
            dependencies: value.dependencies,
            by_ext,
            ..Default::default()
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct MagicDb {
    rule_id: usize,
    rules: Vec<(usize, Rc<MagicRule>)>,
    by_ext: HashMap<String, Vec<(usize, Rc<MagicRule>)>>,
    dependencies: HashMap<String, DependencyRule>,
}

#[inline(always)]
fn guess_stream_kind<S: AsRef<[u8]>>(stream: S) -> StreamKind {
    let Ok(s) = str::from_utf8(stream.as_ref()) else {
        return StreamKind::Binary;
    };

    let count = s.chars().count();
    let mut is_ascii = true;

    for c in s.chars().take(count.saturating_sub(1)) {
        is_ascii &= c.is_ascii()
    }

    if is_ascii {
        StreamKind::Text(TextEncoding::Ascii)
    } else {
        StreamKind::Text(TextEncoding::Utf8)
    }
}

impl MagicDb {
    fn open_reader<R: Read + Seek>(f: R) -> Result<LazyCache<R>, Error> {
        Ok(LazyCache::<R>::from_read_seek(f)
            .and_then(|lc| lc.with_hot_cache(2 * FILE_BYTES_MAX))?)
        .map(|lc| lc.with_warm_cache(100 << 20))
    }

    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    fn next_rule_id(&mut self) -> usize {
        let t = self.rule_id;
        self.rule_id += 1;
        t
    }

    #[inline(always)]
    fn try_json<R: Read + Seek>(
        haystack: &mut LazyCache<R>,
        stream_kind: StreamKind,
        magic: &mut Magic,
    ) -> Result<bool, Error> {
        // cannot be json if content is binary
        if matches!(stream_kind, StreamKind::Binary) {
            return Ok(false);
        }

        let buf = haystack.read_range(0..FILE_BYTES_MAX as u64)?.trim_ascii();

        let Some((start, end)) = find_json_boundaries(buf) else {
            return Ok(false);
        };

        let mut is_ndjson = false;

        trace!("maybe a json document");
        let ok = serde_json::from_slice::<serde_json::Value>(&buf[start..=end]).is_ok();
        if !ok {
            return Ok(false);
        }

        // we are sure it is json now we must look if we are ndjson
        if end + 1 < buf.len() {
            // after first json
            let buf = &buf[end + 1..];
            if let Some((second_start, second_end)) = find_json_boundaries(buf) {
                // there is a new line between the two json docs
                if memchr(b'\n', &buf[..second_start]).is_some() {
                    trace!("might be ndjson");
                    is_ndjson = serde_json::from_slice::<serde_json::Value>(
                        &buf[second_start..=second_end],
                    )
                    .is_ok();
                }
            }
        }

        if is_ndjson {
            magic.push_message(Cow::Borrowed("New Line Delimited"));
            magic.set_mime_type(Cow::Borrowed("application/x-ndjson"));
            magic.insert_extensions(["ndjson", "jsonl"].into_iter());
        } else {
            magic.set_mime_type(Cow::Borrowed("application/json"));
            magic.insert_extensions(["json"].into_iter());
        }

        magic.push_message(Cow::Borrowed("JSON text data"));
        magic.set_source(Some(HARDCODED_SOURCE));
        magic.update_strength(HARDCODED_MAGIC_STRENGTH);
        return Ok(true);
    }

    #[inline(always)]
    fn try_csv<R: Read + Seek>(
        haystack: &mut LazyCache<R>,
        stream_kind: StreamKind,
        magic: &mut Magic,
    ) -> Result<bool, Error> {
        // cannot be csv if content is binary
        let StreamKind::Text(enc) = stream_kind else {
            return Ok(false);
        };

        let buf = haystack.read_range(0..FILE_BYTES_MAX as u64)?;
        let mut reader = csv::Reader::from_reader(io::Cursor::new(buf));
        let mut records = reader.records();

        let Some(Ok(first)) = records.next() else {
            return Ok(false);
        };

        // very not likely a CSV otherwise all programming
        // languages having ; line terminator would be
        // considered as CSV
        if first.len() <= 1 {
            return Ok(false);
        }

        // we already parsed first line
        let mut n = 1;
        for i in records.take(9) {
            if let Ok(rec) = i {
                if first.len() != rec.len() {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
            n += 1;
        }

        // we need at least 10 lines
        if n != 10 {
            return Ok(false);
        }

        magic.set_mime_type(Cow::Borrowed("text/csv"));
        magic.push_message(Cow::Borrowed("CSV"));
        magic.push_message(Cow::Borrowed(enc.as_magic_str()));
        magic.push_message(Cow::Borrowed("text"));
        magic.insert_extensions(["csv"].into_iter());
        magic.set_source(Some(HARDCODED_SOURCE));
        magic.update_strength(HARDCODED_MAGIC_STRENGTH);
        Ok(true)
    }

    #[inline(always)]
    fn try_tar<R: Read + Seek>(
        haystack: &mut LazyCache<R>,
        stream_kind: StreamKind,
        magic: &mut Magic,
    ) -> Result<bool, Error> {
        // cannot be json if content is not binary
        if !matches!(stream_kind, StreamKind::Binary) {
            return Ok(false);
        }

        let buf = haystack.read_range(0..FILE_BYTES_MAX as u64)?;
        let mut ar = Archive::new(io::Cursor::new(buf));

        let Ok(mut entries) = ar.entries() else {
            return Ok(false);
        };

        let Some(Ok(first)) = entries.next() else {
            return Ok(false);
        };

        let header = first.header();

        if header.as_ustar().is_some() {
            magic.push_message(Cow::Borrowed("POSIX tar archive"));
        } else if header.as_gnu().is_some() {
            magic.push_message(Cow::Borrowed("POSIX tar archive (GNU)"));
        } else {
            magic.push_message(Cow::Borrowed("tar archive"));
        }

        magic.set_mime_type(Cow::Borrowed("application/x-tar"));
        magic.set_source(Some(HARDCODED_SOURCE));
        magic.update_strength(HARDCODED_MAGIC_STRENGTH);
        magic.insert_extensions(["tar"].into_iter());
        Ok(true)
    }

    #[inline(always)]
    fn try_hard_magic<R: Read + Seek>(
        haystack: &mut LazyCache<R>,
        stream_kind: StreamKind,
        magic: &mut Magic,
    ) -> Result<bool, Error> {
        if Self::try_json(haystack, stream_kind, magic)? {
            return Ok(true);
        } else if Self::try_csv(haystack, stream_kind, magic)? {
            return Ok(true);
        } else if Self::try_tar(haystack, stream_kind, magic)? {
            return Ok(true);
        }

        Ok(false)
    }

    #[inline(always)]
    fn magic_default<'m, R: Read + Seek>(
        haystack: &mut LazyCache<R>,
        stream_kind: StreamKind,
        magic: &mut Magic<'m>,
    ) -> Result<(), Error> {
        let buf = haystack.read_range(0..FILE_BYTES_MAX as u64)?;

        magic.set_source(Some(HARDCODED_SOURCE));
        magic.set_stream_kind(stream_kind);
        magic.is_default = true;

        if buf.len() == 0 {
            magic.push_message(Cow::Borrowed("empty"));
            magic.set_mime_type(Cow::Borrowed(DEFAULT_BIN_MIMETYPE));
            return Ok(());
        }

        match stream_kind {
            StreamKind::Binary => {
                magic.push_message(Cow::Borrowed("data"));
            }
            StreamKind::Text(e) => {
                magic.push_message(Cow::Borrowed(e.as_magic_str()));
                magic.push_message(Cow::Borrowed("text"));
            }
        }

        Ok(())
    }

    #[inline(always)]
    fn update_by_ext(&mut self, ext: &str, rule_id: usize, rule: &Rc<MagicRule>) {
        self.by_ext
            // we normalize extension to lowercase
            .entry(ext.to_lowercase())
            .and_modify(|v| {
                v.push((rule_id, Rc::clone(rule)));
                v.sort_by_key(|(id, r)| (r.is_text(), *id));
            })
            .or_insert_with(|| vec![(rule_id, Rc::clone(rule))]);
    }

    pub fn load(&mut self, mf: MagicFile) -> Result<&mut Self, Error> {
        // it seems rules are evaluated in their reverse definition order
        for rule in mf.rules.into_iter() {
            let (id, rule) = (self.next_rule_id(), Rc::new(rule));

            let all_exts: HashSet<Cow<'_, str>> = rule
                .entries
                .exts
                .union(&rule.entries.children_exts)
                .map(|s| Cow::Borrowed(s.as_str()))
                .collect();

            if all_exts.is_empty() {
                self.update_by_ext("", id, &rule);
            } else {
                for ext in all_exts {
                    // acceptable to clone here
                    self.update_by_ext(&ext, id, &rule);
                }
            }
            self.rules.push((id, rule));
        }

        // put text rules at the end
        self.rules.sort_by_key(|(id, r)| (r.is_text(), *id));
        self.dependencies.extend(mf.dependencies);
        Ok(self)
    }

    #[inline]
    fn magic_first_with_stream_kind<'m, R: Read + Seek>(
        &self,
        haystack: &mut LazyCache<R>,
        stream_kind: StreamKind,
        extension: Option<&str>,
    ) -> Result<Magic<'_>, Error> {
        // re-using magic makes this function faster
        let mut magic = Magic::default();

        if Self::try_hard_magic(haystack, stream_kind, &mut magic)? {
            return Ok(magic);
        }

        let mut marked = vec![false; self.rules.len()];

        macro_rules! do_magic {
            ($rule: expr) => {{
                $rule.magic_entrypoint(&mut magic, stream_kind, haystack, &self, false, 0)?;

                if !magic.message.is_empty() {
                    magic.set_stream_kind(stream_kind);
                    magic.set_source($rule.source.as_deref());
                    return Ok(magic);
                }

                magic.reset();
            }};
        }

        if let Some(by_extensions) = extension.and_then(|ext| self.by_ext.get(&ext.to_lowercase()))
        {
            for (id, rule) in by_extensions.iter() {
                do_magic!(rule);
                if let Some(f) = marked.get_mut(*id) {
                    *f = true
                }
            }
        }

        for (_, rule) in self
            .rules
            .iter()
            // we don't run again rules run by extension
            .filter(|(id, _)| !*marked.get(*id).unwrap_or(&false))
        {
            do_magic!(rule)
        }

        Self::magic_default(haystack, stream_kind, &mut magic)?;

        Ok(magic)
    }

    /// an empty extension must be `Some("")`, if extension acceleration is not desired specify `None`
    pub fn magic_first<R: Read + Seek>(
        &self,
        r: &mut R,
        extension: Option<&str>,
    ) -> Result<Magic<'_>, Error> {
        let mut haystack = Self::open_reader(r)?;
        let stream_kind = guess_stream_kind(haystack.read_range(0..FILE_BYTES_MAX as u64)?);
        self.magic_first_with_stream_kind(&mut haystack, stream_kind, extension)
    }

    #[inline(always)]
    fn magic_all_with_stream_kind<R: Read + Seek>(
        &self,
        haystack: &mut LazyCache<R>,
        stream_kind: StreamKind,
    ) -> Result<Vec<(u64, Magic<'_>)>, Error> {
        let mut out = Vec::new();

        let mut magic = Magic::default();

        if Self::try_hard_magic(haystack, stream_kind, &mut magic)? {
            out.push((magic.strength.unwrap_or_default(), magic));
            magic = Magic::default();
        }

        for (_, rule) in self.rules.iter() {
            rule.magic_entrypoint(&mut magic, stream_kind, haystack, &self, false, 0)?;

            // it is possible we have a strength with no message
            if !magic.message.is_empty() {
                magic.set_stream_kind(stream_kind);
                magic.set_source(rule.source.as_deref());
                out.push((magic.strength.unwrap_or_default(), magic));
                magic = Magic::default();
            }

            magic.reset();
        }

        Self::magic_default(haystack, stream_kind, &mut magic)?;
        out.push((magic.strength.unwrap_or_default(), magic));

        Ok(out)
    }

    pub fn magic_all<R: Read + Seek>(&self, r: &mut R) -> Result<Vec<(u64, Magic<'_>)>, Error> {
        let mut haystack = Self::open_reader(r)?;
        let stream_kind = guess_stream_kind(haystack.read_range(0..FILE_BYTES_MAX as u64)?);
        self.magic_all_with_stream_kind(&mut haystack, stream_kind)
    }

    #[inline(always)]
    fn magic_best_with_stream_kind<R: Read + Seek>(
        &self,
        haystack: &mut LazyCache<R>,
        stream_kind: StreamKind,
    ) -> Result<Option<Magic<'_>>, Error> {
        let mut magics = self.magic_all_with_stream_kind(haystack, stream_kind)?;
        magics.sort_by(|a, b| b.0.cmp(&a.0));
        return Ok(magics.into_iter().map(|(_, m)| m).next());
    }

    pub fn magic_best<R: Read + Seek>(&self, r: &mut R) -> Result<Option<Magic<'_>>, Error> {
        let mut haystack = Self::open_reader(r)?;
        let stream_kind = guess_stream_kind(haystack.read_range(0..FILE_BYTES_MAX as u64)?);
        self.magic_best_with_stream_kind(&mut haystack, stream_kind)
    }

    pub fn serialize(self) -> Result<Vec<u8>, bincode::error::EncodeError> {
        let sdb = SerializedDb::from(self);
        let mut encoder = GzEncoder::new(vec![], Compression::best());

        bincode::serde::encode_into_std_write(&sdb, &mut encoder, bincode::config::standard())?;
        Ok(encoder.finish().unwrap())
    }

    pub fn deserialize_slice<S: AsRef<[u8]>>(r: S) -> Result<Self, Error> {
        Self::deserialize_reader(&mut r.as_ref())
    }

    pub fn deserialize_reader<R: Read>(r: &mut R) -> Result<Self, Error> {
        let mut buf = vec![];
        let mut gz = GzDecoder::new(r);
        gz.read_to_end(&mut buf).map_err(|e| {
            bincode::error::DecodeError::OtherString(format!("failed to read: {e}"))
        })?;
        let (sdb, _): (SerializedDb, usize) =
            bincode::serde::decode_from_slice(&buf, bincode::config::standard())?;
        Ok(sdb.into())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use regex::bytes::Regex;

    use crate::utils::unix_local_time_to_string;

    use super::*;

    macro_rules! lazy_cache {
        ($l: literal) => {
            LazyCache::from_read_seek(Cursor::new($l)).unwrap()
        };
    }

    fn first_magic(
        rule: &str,
        content: &[u8],
        stream_kind: StreamKind,
    ) -> Result<Option<Magic<'static>>, Error> {
        let mut md = MagicDb::new();
        md.load(
            FileMagicParser::parse_str(rule, None)
                .inspect_err(|e| eprintln!("{e}"))
                .unwrap(),
        )
        .unwrap();
        let mut reader = LazyCache::from_read_seek(Cursor::new(content)).unwrap();
        let v = md.magic_best_with_stream_kind(&mut reader, stream_kind)?;
        Ok(v.map(|m| m.into_owned()))
    }

    /// helper macro to debug tests
    #[allow(unused_macros)]
    macro_rules! enable_trace {
        () => {
            tracing_subscriber::fmt()
                .with_max_level(tracing_subscriber::filter::LevelFilter::TRACE)
                .try_init();
        };
    }

    macro_rules! parse_assert {
        ($rule:literal) => {
            FileMagicParser::parse_str($rule, None)
                .inspect_err(|e| eprintln!("{e}"))
                .unwrap();
        };
    }

    macro_rules! assert_magic_match_bin {
        ($rule: literal, $content:literal) => {{
            first_magic($rule, $content, StreamKind::Binary)
                .unwrap()
                .unwrap();
        }};
        ($rule: literal, $content:literal, $message:expr) => {{
            assert_eq!(
                first_magic($rule, $content, StreamKind::Binary)
                    .unwrap()
                    .unwrap()
                    .message(),
                $message
            );
        }};
    }

    macro_rules! assert_magic_match_text {
        ($rule: literal, $content:literal) => {{
            first_magic($rule, $content, StreamKind::Text(TextEncoding::Utf8))
                .unwrap()
                .unwrap();
        }};
        ($rule: literal, $content:literal, $message:expr) => {{
            assert_eq!(
                first_magic($rule, $content, StreamKind::Text(TextEncoding::Utf8))
                    .unwrap()
                    .unwrap()
                    .message(),
                $message
            );
        }};
    }

    macro_rules! assert_magic_not_match_text {
        ($rule: literal, $content:literal) => {{
            assert!(
                first_magic($rule, $content, StreamKind::Text(TextEncoding::Utf8))
                    .unwrap()
                    .unwrap()
                    .is_default()
            );
        }};
    }

    macro_rules! assert_magic_not_match_bin {
        ($rule: literal, $content:literal) => {{
            assert!(
                first_magic($rule, $content, StreamKind::Binary)
                    .unwrap()
                    .unwrap()
                    .is_default()
            );
        }};
    }

    #[test]
    fn test_regex() {
        assert_magic_match_text!(
            r#"
0	regex/1024 \^#![[:space:]]*/usr/bin/env[[:space:]]+
!:mime	text/x-shellscript
>&0  regex/64 .*($|\\b) %s shell script text executable
    "#,
            br#"#!/usr/bin/env bash
        echo hello world"#,
            // the magic generated
            "bash shell script text executable"
        );

        let re = Regex::new(r"(?-u)\x42\x82").unwrap();
        assert!(re.is_match(b"\x42\x82"));

        assert_magic_match_bin!(
            r#"0 regex \x42\x82 binary regex match"#,
            b"\x00\x00\x00\x00\x00\x00\x42\x82"
        );

        // test regex continuation after match
        assert_magic_match_bin!(
            r#"
            0 regex \x42\x82
            >&0 string \xde\xad\xbe\xef it works
            "#,
            b"\x00\x00\x00\x00\x00\x00\x42\x82\xde\xad\xbe\xef"
        );

        assert_magic_match_bin!(
            r#"
            0 regex/s \x42\x82
            >&0 string \x42\x82\xde\xad\xbe\xef it works
            "#,
            b"\x00\x00\x00\x00\x00\x00\x42\x82\xde\xad\xbe\xef"
        );

        // ^ must match stat of line when matching text
        assert_magic_match_text!(
            r#"
0	regex/1024 \^HelloWorld$ HelloWorld String"#,
            br#"
// this is a comment after an empty line
HelloWorld
            "#
        );
    }

    #[test]
    fn test_string_with_mods() {
        assert_magic_match_text!(
            r#"0	string/w	#!\ \ \ /usr/bin/env\ bash	BASH
        "#,
            b"#! /usr/bin/env bash i
        echo hello world"
        );

        // test uppercase insensitive
        assert_magic_match_text!(
            r#"0	string/C	HelloWorld	it works
        "#,
            b"helloworld"
        );

        assert_magic_not_match_text!(
            r#"0	string/C	HelloWorld	it works
        "#,
            b"hELLOwORLD"
        );

        // test lowercase insensitive
        assert_magic_match_text!(
            r#"0	string/c	HelloWorld	it works
        "#,
            b"HELLOWORLD"
        );

        assert_magic_not_match_text!(
            r#"0	string/c	HelloWorld	it works
        "#,
            b"helloworld"
        );

        // test full word match
        assert_magic_match_text!(
            r#"0	string/f	#!/usr/bin/env\ bash	BASH
        "#,
            b"#!/usr/bin/env bash"
        );

        assert_magic_not_match_text!(
            r#"0	string/f	#!/usr/bin/python PYTHON"#,
            b"#!/usr/bin/pythonic"
        );

        // testing whitespace compacting
        assert_magic_match_text!(
            r#"0	string/W	#!/usr/bin/env\ python  PYTHON"#,
            b"#!/usr/bin/env    python"
        );

        assert_magic_not_match_text!(
            r#"0	string/W	#!/usr/bin/env\ \ python  PYTHON"#,
            b"#!/usr/bin/env python"
        );
    }

    #[test]
    fn test_search_with_mods() {
        assert_magic_match_text!(
            r#"0	search/1/fwt	#!\ /usr/bin/luatex	LuaTex script text executable"#,
            b"#!          /usr/bin/luatex "
        );

        // test matching from the beginning
        assert_magic_match_text!(
            r#"
            0	search/s	/usr/bin/env
            >&0 string /usr/bin/env it works
            "#,
            b"#!/usr/bin/env    python"
        );

        assert_magic_not_match_text!(
            r#"
            0	search	/usr/bin/env
            >&0 string /usr/bin/env it works
            "#,
            b"#!/usr/bin/env    python"
        );
    }

    #[test]
    fn test_pstring() {
        assert_magic_match_bin!(r#"0 pstring Toast it works"#, b"\x05Toast");

        assert_magic_match_bin!(r#"0 pstring Toast %s"#, b"\x05Toast", "Toast");

        assert_magic_not_match_bin!(r#"0 pstring Toast Doesn't work"#, b"\x07Toaster");

        // testing with modifiers
        assert_magic_match_bin!(r#"0 pstring/H Toast it works"#, b"\x00\x05Toast");

        assert_magic_match_bin!(r#"0 pstring/HJ Toast it works"#, b"\x00\x07Toast");

        assert_magic_match_bin!(r#"0 pstring/HJ Toast %s"#, b"\x00\x07Toast", "Toast");

        assert_magic_match_bin!(r#"0 pstring/h Toast it works"#, b"\x05\x00Toast");

        assert_magic_match_bin!(r#"0 pstring/hJ Toast it works"#, b"\x07\x00Toast");

        assert_magic_match_bin!(r#"0 pstring/L Toast it works"#, b"\x00\x00\x00\x05Toast");

        assert_magic_match_bin!(r#"0 pstring/LJ Toast it works"#, b"\x00\x00\x00\x09Toast");

        assert_magic_match_bin!(r#"0 pstring/l Toast it works"#, b"\x05\x00\x00\x00Toast");

        assert_magic_match_bin!(r#"0 pstring/lJ Toast it works"#, b"\x09\x00\x00\x00Toast");
    }

    #[test]
    fn test_max_recursion() {
        let res = first_magic(
            r#"0	indirect x"#,
            b"#!          /usr/bin/luatex ",
            StreamKind::Binary,
        );
        assert!(res.is_err());
        let _ = res.inspect_err(|e| {
            assert!(matches!(
                e.unwrap_localized(),
                Error::MaximumRecursion(MAX_RECURSION)
            ))
        });
    }

    #[test]
    fn test_string_ops() {
        assert_magic_match_text!("0	string/b MZ MZ File", b"MZ\0");
        assert_magic_match_text!("0	string !MZ Not MZ File", b"AZ\0");
        assert_magic_match_text!("0	string >\0 Any String", b"A\0");
        assert_magic_match_text!("0	string >Test Any String", b"Test 1\0");
        assert_magic_match_text!("0	string <Test Any String", b"\0");
        assert_magic_not_match_text!("0	string >Test Any String", b"\0");
    }

    #[test]
    fn test_lestring16() {
        assert_magic_match_bin!(
            "0 lestring16 abcd Little-endian UTF-16 string",
            b"\x61\x00\x62\x00\x63\x00\x64\x00"
        );
        assert_magic_match_bin!(
            "0 lestring16 x %s",
            b"\x61\x00\x62\x00\x63\x00\x64\x00\x00",
            "abcd"
        );
        assert_magic_not_match_bin!(
            "0 lestring16 abcd Little-endian UTF-16 string",
            b"\x00\x61\x00\x62\x00\x63\x00\x64"
        );
        assert_magic_match_bin!(
            "4 lestring16 abcd Little-endian UTF-16 string",
            b"\x00\x00\x00\x00\x61\x00\x62\x00\x63\x00\x64\x00"
        );
    }

    #[test]
    fn test_bestring16() {
        assert_magic_match_bin!(
            "0 bestring16 abcd Big-endian UTF-16 string",
            b"\x00\x61\x00\x62\x00\x63\x00\x64"
        );
        assert_magic_match_bin!(
            "0 bestring16 x %s",
            b"\x00\x61\x00\x62\x00\x63\x00\x64",
            "abcd"
        );
        assert_magic_not_match_bin!(
            "0 bestring16 abcd Big-endian UTF-16 string",
            b"\x61\x00\x62\x00\x63\x00\x64\x00"
        );
        assert_magic_match_bin!(
            "4 bestring16 abcd Big-endian UTF-16 string",
            b"\x00\x00\x00\x00\x00\x61\x00\x62\x00\x63\x00\x64"
        );
    }

    #[test]
    fn test_offset_from_end() {
        assert_magic_match_bin!("-1 ubyte 0x42 last byte ok", b"\x00\x00\x42");
        assert_magic_match_bin!("-2 ubyte 0x41 last byte ok", b"\x00\x41\x00");
    }

    #[test]
    fn test_relative_offset() {
        assert_magic_match_bin!(
            "
            0 ubyte 0x42
            >&0 ubyte 0x00
            >>&0 ubyte 0x41 third byte ok
            ",
            b"\x42\x00\x41\x00"
        );
    }

    #[test]
    fn test_indirect_offset() {
        assert_magic_match_bin!("(0.l) ubyte 0x42 it works", b"\x04\x00\x00\x00\x42");
        // adding fixed value to offset
        assert_magic_match_bin!("(0.l+3) ubyte 0x42 it works", b"\x01\x00\x00\x00\x42");
        // testing offset pair
        assert_magic_match_bin!(
            "(0.l+(4)) ubyte 0x42 it works",
            b"\x04\x00\x00\x00\x04\x00\x00\x00\x42"
        );
    }

    #[test]
    fn test_use_with_message() {
        assert_magic_match_bin!(
            r#"
0 string MZ
>0 use mz first match

0 name mz then second match
>0 string MZ
"#,
            b"MZ\0",
            "first match then second match"
        );
    }

    #[test]
    fn test_scalar_transform() {
        assert_magic_match_bin!("0 ubyte+1 0x1 add works", b"\x00");
        assert_magic_match_bin!("0 ubyte-1 0xfe sub works", b"\xff");
        assert_magic_match_bin!("0 ubyte%2 0 mod works", b"\x0a");
        assert_magic_match_bin!("0 ubyte&0x0f 0x0f bitand works", b"\xff");
        assert_magic_match_bin!("0 ubyte|0x0f 0xff bitor works", b"\xf0");
        assert_magic_match_bin!("0 ubyte^0x0f 0xf0 bitxor works", b"\xff");

        FileMagicParser::parse_str("0 ubyte%0 mod by zero", None)
            .expect_err("expect div by zero error");
        FileMagicParser::parse_str("0 ubyte/0 div by zero", None)
            .expect_err("expect div by zero error");
    }

    #[test]
    fn test_belong() {
        // Test that a file with a four-byte value at offset 0 that matches the given value in big-endian byte order
        assert_magic_match_bin!("0 belong 0x12345678 Big-endian long", b"\x12\x34\x56\x78");
        // Test that a file with a four-byte value at offset 0 that does not match the given value in big-endian byte order
        assert_magic_not_match_bin!("0 belong 0x12345678 Big-endian long", b"\x78\x56\x34\x12");
        // Test that a file with a four-byte value at a non-zero offset that matches the given value in big-endian byte order
        assert_magic_match_bin!(
            "4 belong 0x12345678 Big-endian long",
            b"\x00\x00\x00\x00\x12\x34\x56\x78"
        );
        // Test < operator
        assert_magic_match_bin!("0 belong <0x12345678 Big-endian long", b"\x12\x34\x56\x77");
        assert_magic_not_match_bin!("0 belong <0x12345678 Big-endian long", b"\x12\x34\x56\x78");

        // Test > operator
        assert_magic_match_bin!("0 belong >0x12345678 Big-endian long", b"\x12\x34\x56\x79");
        assert_magic_not_match_bin!("0 belong >0x12345678 Big-endian long", b"\x12\x34\x56\x78");

        // Test & operator
        assert_magic_match_bin!("0 belong &0x5678 Big-endian long", b"\x00\x00\x56\x78");
        assert_magic_not_match_bin!("0 belong &0x0000FFFF Big-endian long", b"\x12\x34\x56\x78");

        // Test ^ operator (bitwise AND with complement)
        assert_magic_match_bin!("0 belong ^0xFFFF0000 Big-endian long", b"\x00\x00\x56\x78");
        assert_magic_not_match_bin!("0 belong ^0xFFFF0000 Big-endian long", b"\x00\x01\x56\x78");

        // Test ~ operator
        assert_magic_match_bin!("0 belong ~0x12345678 Big-endian long", b"\xed\xcb\xa9\x87");
        assert_magic_not_match_bin!("0 belong ~0x12345678 Big-endian long", b"\x12\x34\x56\x78");

        // Test x operator
        assert_magic_match_bin!("0 belong x Big-endian long", b"\x12\x34\x56\x78");
        assert_magic_match_bin!("0 belong x Big-endian long", b"\x78\x56\x34\x12");
    }

    #[test]
    fn test_parse_search() {
        parse_assert!("0 search test");
        parse_assert!("0 search/24/s test");
        parse_assert!("0 search/s/24 test");
    }

    #[test]
    fn test_bedate() {
        assert_magic_match_bin!(
            "0 bedate 946684800 Unix date (Jan 1, 2000)",
            b"\x38\x6D\x43\x80"
        );
        assert_magic_not_match_bin!(
            "0 bedate 946684800 Unix date (Jan 1, 2000)",
            b"\x00\x00\x00\x00"
        );
        assert_magic_match_bin!(
            "4 bedate 946684800 %s",
            b"\x00\x00\x00\x00\x38\x6D\x43\x80",
            "2000-01-01 00:00:00"
        );
    }
    #[test]
    fn test_beldate() {
        assert_magic_match_bin!(
            "0 beldate 946684800 Local date (Jan 1, 2000)",
            b"\x38\x6D\x43\x80"
        );
        assert_magic_not_match_bin!(
            "0 beldate 946684800 Local date (Jan 1, 2000)",
            b"\x00\x00\x00\x00"
        );

        assert_magic_match_bin!(
            "4 beldate 946684800 {}",
            b"\x00\x00\x00\x00\x38\x6D\x43\x80",
            unix_local_time_to_string(946684800)
        );
    }

    #[test]
    fn test_beqdate() {
        assert_magic_match_bin!(
            "0 beqdate 946684800 Unix date (Jan 1, 2000)",
            b"\x00\x00\x00\x00\x38\x6D\x43\x80"
        );

        assert_magic_not_match_bin!(
            "0 beqdate 946684800 Unix date (Jan 1, 2000)",
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        );

        assert_magic_match_bin!(
            "0 beqdate 946684800 %s",
            b"\x00\x00\x00\x00\x38\x6D\x43\x80",
            "2000-01-01 00:00:00"
        );
    }

    #[test]
    fn test_medate() {
        assert_magic_match_bin!(
            "0 medate 946684800 Unix date (Jan 1, 2000)",
            b"\x6D\x38\x80\x43"
        );

        assert_magic_not_match_bin!(
            "0 medate 946684800 Unix date (Jan 1, 2000)",
            b"\x00\x00\x00\x00"
        );

        assert_magic_match_bin!(
            "4 medate 946684800 %s",
            b"\x00\x00\x00\x00\x6D\x38\x80\x43",
            "2000-01-01 00:00:00"
        );
    }

    #[test]
    fn test_meldate() {
        assert_magic_match_bin!(
            "0 meldate 946684800 Local date (Jan 1, 2000)",
            b"\x6D\x38\x80\x43"
        );
        assert_magic_not_match_bin!(
            "0 meldate 946684800 Local date (Jan 1, 2000)",
            b"\x00\x00\x00\x00"
        );

        assert_magic_match_bin!(
            "4 meldate 946684800 %s",
            b"\x00\x00\x00\x00\x6D\x38\x80\x43",
            unix_local_time_to_string(946684800)
        );
    }

    #[test]
    fn test_date() {
        assert_magic_match_bin!(
            "0 date 946684800 Local date (Jan 1, 2000)",
            b"\x80\x43\x6D\x38"
        );
        assert_magic_not_match_bin!(
            "0 date 946684800 Local date (Jan 1, 2000)",
            b"\x00\x00\x00\x00"
        );
        assert_magic_match_bin!(
            "4 date 946684800 {}",
            b"\x00\x00\x00\x00\x80\x43\x6D\x38",
            "2000-01-01 00:00:00"
        );
    }

    #[test]
    fn test_leldate() {
        assert_magic_match_bin!(
            "0 leldate 946684800 Local date (Jan 1, 2000)",
            b"\x80\x43\x6D\x38"
        );
        assert_magic_not_match_bin!(
            "0 leldate 946684800 Local date (Jan 1, 2000)",
            b"\x00\x00\x00\x00"
        );
        assert_magic_match_bin!(
            "4 leldate 946684800 {}",
            b"\x00\x00\x00\x00\x80\x43\x6D\x38",
            unix_local_time_to_string(946684800)
        );
    }

    #[test]
    fn test_leqdate() {
        assert_magic_match_bin!(
            "0 leqdate 1577836800 Unix date (Jan 1, 2020)",
            b"\x00\xe1\x0b\x5E\x00\x00\x00\x00"
        );

        assert_magic_not_match_bin!(
            "0 leqdate 1577836800 Unix date (Jan 1, 2020)",
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        );
        assert_magic_match_bin!(
            "8 leqdate 1577836800 %s",
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\xE1\x0B\x5E\x00\x00\x00\x00",
            "2020-01-01 00:00:00"
        );
    }

    #[test]
    fn test_leqldate() {
        assert_magic_match_bin!(
            "0 leqldate 1577836800 Unix date (Jan 1, 2020)",
            b"\x00\xe1\x0b\x5E\x00\x00\x00\x00"
        );

        assert_magic_not_match_bin!(
            "0 leqldate 1577836800 Unix date (Jan 1, 2020)",
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        );
        assert_magic_match_bin!(
            "8 leqldate 1577836800 %s",
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\xE1\x0B\x5E\x00\x00\x00\x00",
            unix_local_time_to_string(1577836800)
        );
    }

    #[test]
    fn test_melong() {
        // Test = operator
        assert_magic_match_bin!(
            "0 melong =0x12345678 Middle-endian long",
            b"\x34\x12\x78\x56"
        );
        assert_magic_not_match_bin!(
            "0 melong =0x12345678 Middle-endian long",
            b"\x00\x00\x00\x00"
        );

        // Test < operator
        assert_magic_match_bin!(
            "0 melong <0x12345678 Middle-endian long",
            b"\x34\x12\x78\x55"
        ); // 0x12345677 in middle-endian
        assert_magic_not_match_bin!(
            "0 melong <0x12345678 Middle-endian long",
            b"\x34\x12\x78\x56"
        ); // 0x12345678 in middle-endian

        // Test > operator
        assert_magic_match_bin!(
            "0 melong >0x12345678 Middle-endian long",
            b"\x34\x12\x78\x57"
        ); // 0x12345679 in middle-endian
        assert_magic_not_match_bin!(
            "0 melong >0x12345678 Middle-endian long",
            b"\x34\x12\x78\x56"
        ); // 0x12345678 in middle-endian

        // Test & operator
        assert_magic_match_bin!("0 melong &0x5678 Middle-endian long", b"\xab\xcd\x78\x56"); // 0x00007856 in middle-endian
        assert_magic_not_match_bin!(
            "0 melong &0x0000FFFF Middle-endian long",
            b"\x34\x12\x78\x56"
        ); // 0x12347856 in middle-endian

        // Test ^ operator (bitwise AND with complement)
        assert_magic_match_bin!(
            "0 melong ^0xFFFF0000 Middle-endian long",
            b"\x00\x00\x78\x56"
        ); // 0x00007856 in middle-endian
        assert_magic_not_match_bin!(
            "0 melong ^0xFFFF0000 Middle-endian long",
            b"\x00\x01\x78\x56"
        ); // 0x00017856 in middle-endian

        // Test ~ operator
        assert_magic_match_bin!(
            "0 melong ~0x12345678 Middle-endian long",
            b"\xCB\xED\x87\xA9"
        );
        assert_magic_not_match_bin!(
            "0 melong ~0x12345678 Middle-endian long",
            b"\x34\x12\x78\x56"
        ); // The original value

        // Test x operator
        assert_magic_match_bin!("0 melong x Middle-endian long", b"\x34\x12\x78\x56");
        assert_magic_match_bin!("0 melong x Middle-endian long", b"\x00\x00\x00\x00");
    }

    #[test]
    fn test_uquad() {
        // Test = operator
        assert_magic_match_bin!(
            "0 uquad =0x123456789ABCDEF0 Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        );
        assert_magic_not_match_bin!(
            "0 uquad =0x123456789ABCDEF0 Unsigned quad",
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        );

        // Test < operator
        assert_magic_match_bin!(
            "0 uquad <0x123456789ABCDEF0 Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x11"
        );
        assert_magic_not_match_bin!(
            "0 uquad <0x123456789ABCDEF0 Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        );

        // Test > operator
        assert_magic_match_bin!(
            "0 uquad >0x123456789ABCDEF0 Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x13"
        );
        assert_magic_not_match_bin!(
            "0 uquad >0x123456789ABCDEF0 Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        );

        // Test & operator
        assert_magic_match_bin!(
            "0 uquad &0xF0 Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        );
        assert_magic_not_match_bin!(
            "0 uquad &0xFF Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        );

        // Test ^ operator (bitwise AND with complement)
        assert_magic_match_bin!(
            "0 uquad ^0xFFFFFFFFFFFFFFFF Unsigned quad",
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        ); // All bits clear
        assert_magic_not_match_bin!(
            "0 uquad ^0xFFFFFFFFFFFFFFFF Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        ); // Some bits set

        // Test ~ operator
        assert_magic_match_bin!(
            "0 uquad ~0x123456789ABCDEF0 Unsigned quad",
            b"\x0F\x21\x43\x65\x87\xA9\xCB\xED"
        );
        assert_magic_not_match_bin!(
            "0 uquad ~0x123456789ABCDEF0 Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        ); // The original value

        // Test x operator
        assert_magic_match_bin!(
            "0 uquad x {:#x}",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12",
            "0x123456789abcdef0"
        );
        assert_magic_match_bin!(
            "0 uquad x Unsigned quad",
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        );
    }

    #[test]
    fn test_guid() {
        assert_magic_match_bin!(
            "0 guid EC959539-6786-2D4E-8FDB-98814CE76C1E It works",
            b"\xEC\x95\x95\x39\x67\x86\x2D\x4E\x8F\xDB\x98\x81\x4C\xE7\x6C\x1E"
        );

        assert_magic_not_match_bin!(
            "0 guid 399595EC-8667-4E2D-8FDB-98814CE76C1E It works",
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F"
        );

        assert_magic_match_bin!(
            "0 guid x %s",
            b"\xEC\x95\x95\x39\x67\x86\x2D\x4E\x8F\xDB\x98\x81\x4C\xE7\x6C\x1E",
            "EC959539-6786-2D4E-8FDB-98814CE76C1E"
        );
    }

    #[test]
    fn test_ubeqdate() {
        assert_magic_match_bin!(
            "0 ubeqdate 1633046400 It works",
            b"\x00\x00\x00\x00\x61\x56\x4f\x80"
        );

        assert_magic_match_bin!(
            "0 ubeqdate x %s",
            b"\x00\x00\x00\x00\x61\x56\x4f\x80",
            "2021-10-01 00:00:00"
        );

        assert_magic_not_match_bin!(
            "0 ubeqdate 1633046400 It should not work",
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        );
    }

    #[test]
    fn test_ldate() {
        assert_magic_match_bin!("0 ldate 1640551520 It works", b"\x60\xd4\xC8\x61");

        assert_magic_not_match_bin!("0 ldate 1633046400 It should not work", b"\x00\x00\x00\x00");

        assert_magic_match_bin!(
            "0 ldate x %s",
            b"\x60\xd4\xC8\x61",
            unix_local_time_to_string(1640551520)
        );
    }

    #[test]
    fn test_scalar_with_transform() {
        assert_magic_match_bin!("0 ubyte/10 2 {}", b"\x14", "2");
        assert_magic_match_bin!("0 ubyte/10 x {}", b"\x14", "2");
        assert_magic_match_bin!("0 ubyte%10 x {}", b"\x14", "0");
    }

    #[test]
    fn test_float_with_transform() {
        assert_magic_match_bin!("0 lefloat/10 2 {}", b"\x00\x00\xa0\x41", "2");
        assert_magic_match_bin!("0 lefloat/10 x {}", b"\x00\x00\xa0\x41", "2");
        assert_magic_match_bin!("0 lefloat%10 x {}", b"\x00\x00\xa0\x41", "0");
    }

    #[test]
    fn test_read_octal() {
        // Basic cases
        assert_eq!(read_octal_u64(&mut lazy_cache!("0")), Some(0));
        assert_eq!(read_octal_u64(&mut lazy_cache!("00")), Some(0));
        assert_eq!(read_octal_u64(&mut lazy_cache!("01")), Some(1));
        assert_eq!(read_octal_u64(&mut lazy_cache!("07")), Some(7));
        assert_eq!(read_octal_u64(&mut lazy_cache!("010")), Some(8));
        assert_eq!(read_octal_u64(&mut lazy_cache!("0123")), Some(83));
        assert_eq!(read_octal_u64(&mut lazy_cache!("0755")), Some(493));

        // With trailing non-octal characters
        assert_eq!(read_octal_u64(&mut lazy_cache!("0ABC")), Some(0));
        assert_eq!(read_octal_u64(&mut lazy_cache!("01ABC")), Some(1));
        assert_eq!(read_octal_u64(&mut lazy_cache!("0755ABC")), Some(493));
        assert_eq!(read_octal_u64(&mut lazy_cache!("0123ABC")), Some(83));

        // Invalid octal digits
        assert_eq!(read_octal_u64(&mut lazy_cache!("08")), Some(0)); // stops at '8'
        assert_eq!(read_octal_u64(&mut lazy_cache!("01238")), Some(83)); // stops at '8'

        // No leading '0'
        assert_eq!(read_octal_u64(&mut lazy_cache!("123")), None);
        assert_eq!(read_octal_u64(&mut lazy_cache!("755")), None);

        // Empty string
        assert_eq!(read_octal_u64(&mut lazy_cache!("")), None);

        // Only non-octal characters
        assert_eq!(read_octal_u64(&mut lazy_cache!("ABC")), None);
        assert_eq!(read_octal_u64(&mut lazy_cache!("8ABC")), None); // first char is not '0'

        // Longer valid octal (but within u64 range)
        assert_eq!(
            read_octal_u64(&mut lazy_cache!("01777777777")),
            Some(268435455)
        );
    }

    #[test]
    fn test_offset_bug_1() {
        // this tests the exact behaviour
        // expected by libmagic/file
        assert_magic_match_bin!(
            r"
1	string		TEST Bread is
# offset computation is relative to
# rule start
>(5.b)	use toasted

0 name toasted
>0	string twice Toasted
>>0  use toasted_twice 

0 name toasted_twice
>(6.b) string x %s
        ",
            b"\x00TEST\x06twice\x00\x06",
            "Bread is Toasted twice"
        );
    }

    // this test implement the exact same logic as
    // test_offset_bug_1 except that the rule starts
    // matching from end. Surprisingly we need to
    // adjust indirect offsets so that it works in
    // libmagic/file
    #[test]
    fn test_offset_bug_2() {
        // this tests the exact behaviour
        // expected by libmagic/file
        assert_magic_match_bin!(
            r"
-12	string		TEST Bread is
>(4.b)	use toasted

0 name toasted
>0	string twice Toasted
>>0  use toasted_twice

0 name toasted_twice
>(6.b) string x %
        ",
            b"\x00TEST\x06twice\x00\x06",
            "Bread is Toasted twice"
        )
    }

    #[test]
    fn test_offset_bug_3() {
        // this tests the exact behaviour
        // expected by libmagic/file
        assert_magic_match_bin!(
            r"
1	string		TEST Bread is
>(5.b) indirect/r x

0	string twice Toasted
>0  use toasted_twice

0 name toasted_twice
>0 string x %s
        ",
            b"\x00TEST\x06twice\x00\x08",
            "Bread is Toasted twice"
        )
    }

    #[test]
    fn test_offset_bug_4() {
        // this tests the exact behaviour
        // expected by libmagic/file
        assert_magic_match_bin!(
            r"
1	string		Bread %s
>(6.b) indirect/r x

# this one uses a based offset
# computed at indirection
1	string is\ Toasted %s
>(11.b)  use toasted_twice

# this one is using a new base
# offset being previous base 
# offset + offset of use
0 name toasted_twice
>0 string x %s
            ",
            b"\x00Bread\x06is Toasted\x0ctwice\x00",
            "Bread is Toasted twice"
        )
    }

    #[test]
    fn test_offset_bug_5() {
        assert_magic_match_bin!(
            r"
1	string		TEST Bread is
>(5.b) indirect/r x

0	string twice Toasted
>0  use toasted_twice

0 name toasted_twice
>0 string twice
>>&1 byte 0x08 twice
            ",
            b"\x00TEST\x06twice\x00\x08",
            "Bread is Toasted twice"
        )
    }
}
