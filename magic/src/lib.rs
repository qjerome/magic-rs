#![deny(unsafe_code)]

use dyf::{DynDisplay, FormatString, dformat};
use flagset::{FlagSet, flags};
use lazy_cache::LazyCache;
use pest::{Span, error::ErrorVariant};
use regex::bytes::{self};
use std::{
    borrow::Cow,
    char::REPLACEMENT_CHARACTER,
    cmp::max,
    collections::{HashMap, HashSet},
    fmt::{self, Debug, Display},
    io::{self, Read, Seek, SeekFrom},
    iter::Peekable,
    ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Sub},
    path::Path,
    str::Utf8Error,
};
use thiserror::Error;
use tracing::{Level, debug, enabled, error, trace};

use crate::{
    numeric::{Float, FloatDataType},
    parser::{FileMagicParser, Rule},
    utils::nonmagic,
};

mod numeric;
mod parser;
mod utils;

use numeric::{Scalar, ScalarDataType};

// corresponds to FILE_INDIR_MAX constant defined in libmagic
const MAX_RECURSION: usize = 50;
// constant found in libmagic. It is used to limit for search tests
pub const FILE_BYTES_MAX: usize = 7 * 1024 * 1024;
// constant found in libmagic. It is used to limit for regex tests
const FILE_REGEX_MAX: usize = 8192;

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
        $r.read_exact_into(&mut a)?;
        a
    }};
}

macro_rules! read_le {
    ($r:expr, $ty: ty ) => {{ <$ty>::from_le_bytes(read!($r, $ty)) }};
}

macro_rules! read_be {
    ($r:expr, $ty: ty ) => {{ <$ty>::from_be_bytes(read!($r, $ty)) }};
}

#[derive(Debug, Clone)]
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
    #[inline(always)]
    fn format_with(&self, mr: Option<&MatchRes>) -> Cow<'_, str> {
        match self {
            Self::String(s) => Cow::Borrowed(s.as_str()),
            Self::Format {
                printf_spec: c_spec,
                fs,
            } => {
                if let Some(mr) = mr {
                    match mr {
                        MatchRes::OwnedString(_, _) => {
                            // FIXME: fix unwrap
                            Cow::Owned(dformat!(fs, mr).unwrap())
                        }
                        MatchRes::String(_, _) => {
                            // FIXME: fix unwrap
                            Cow::Owned(dformat!(fs, mr).unwrap())
                        }
                        MatchRes::Bytes(_, _) => {
                            // FIX: unwrap
                            Cow::Owned(dformat!(fs, mr).unwrap())
                        }
                        MatchRes::Float(_, _) => {
                            // FIX: unwrap
                            Cow::Owned(dformat!(fs, mr).unwrap())
                        }
                        MatchRes::Scalar(_, scalar) => {
                            // we want to print a byte as char
                            if c_spec.as_str() == "c" {
                                match scalar {
                                    Scalar::byte(b) => {
                                        let b = (*b as u8) as char;
                                        // FIXME: fix unwrap
                                        Cow::Owned(dformat!(fs, b).unwrap())
                                    }
                                    Scalar::ubyte(b) => {
                                        let b = *b as char;
                                        // FIXME: fix unwrap
                                        Cow::Owned(dformat!(fs, b).unwrap())
                                    }
                                    // FIXME: fix unwrap
                                    _ => Cow::Owned(dformat!(fs, mr).unwrap()),
                                }
                            } else {
                                // FIXME: fix unwrap
                                Cow::Owned(dformat!(fs, mr).unwrap())
                            }
                        }
                    }
                } else {
                    fs.to_string_lossy()
                }
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("unexpected rule: {0}")]
    UnexpectedRule(String),
    #[error("missing rule: {0}")]
    MissingRule(String),
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("parser error: {0}")]
    Parse(#[from] Box<pest::error::Error<Rule>>),
    #[error("from-utf8: {0}")]
    Utf8(#[from] Utf8Error),
    #[error("maximum recursion reached: {0}")]
    MaximumRecursion(usize),
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
}

#[derive(Debug, Error)]
#[error("{0}")]
pub struct ParserError(pest::error::Error<Rule>);

impl From<pest::error::Error<Rule>> for ParserError {
    fn from(value: pest::error::Error<Rule>) -> Self {
        Self(value)
    }
}

impl ScalarDataType {
    #[inline(always)]
    fn read<R: Read + Seek>(
        &self,
        from: &mut LazyCache<R>,
        switch_endianness: bool,
    ) -> Result<Scalar, Error> {
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
    fn read<R: Read + Seek>(
        &self,
        from: &mut LazyCache<R>,
        switch_endianness: bool,
    ) -> Result<Float, Error> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy)]
enum CmpOp {
    Eq,
    Lt,
    Gt,
    BitAnd,
    Neq, // ! operator
    Xor,
    // FIXME: this operator might be useless
    // it could be turned into Eq and transforming
    // the test value
    Not, // ~ operator
}

#[derive(Debug, Clone)]
struct ScalarTransform {
    op: Op,
    num: Scalar,
}

impl ScalarTransform {
    fn apply(&self, s: Scalar) -> Scalar {
        match self.op {
            // FIXME: impl checked_ fn
            Op::Add => s.add(self.num),
            // FIXME: impl checked_ fn
            Op::Sub => s.sub(self.num),
            // FIXME: impl checked_ fn
            Op::Mul => s.mul(self.num),
            // FIXME: impl checked_ fn
            Op::Div => s.div(self.num),
            // FIXME: impl checked_ fn
            Op::Mod => s.rem(self.num),
            Op::And => s.bitand(self.num),
            Op::Xor => s.bitxor(self.num),
            Op::Or => s.bitor(self.num),
        }
    }
}

#[derive(Debug, Clone)]
struct FloatTransform {
    op: Op,
    num: Float,
}

impl FloatTransform {
    fn apply(&self, s: Float) -> Float {
        match self.op {
            // FIXME: impl checked_ fn
            Op::Add => s.add(self.num),
            // FIXME: impl checked_ fn
            Op::Sub => s.sub(self.num),
            // FIXME: impl checked_ fn
            Op::Mul => s.mul(self.num),
            // FIXME: impl checked_ fn
            Op::Div => s.div(self.num),
            // FIXME: impl checked_ fn
            Op::Mod => s.rem(self.num),
            // parser makes sure those operators cannot be used
            Op::And | Op::Xor | Op::Or => {
                debug_panic!("unsupported operation");
                s
            }
        }
    }
}

// Any Magic Data type
// FIXME: Any must carry StringTest so that we know the string mods / length
#[derive(Debug, Clone)]
enum Any {
    String,
    String16(Encoding),
    PString,
    Scalar(ScalarDataType),
    Float(FloatDataType),
}

impl Any {
    fn from_rule(r: Rule) -> Self {
        match r {
            Rule::string => Any::String,
            Rule::pstring => Any::PString,
            _ => unimplemented!(),
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

#[derive(Debug, Clone)]
struct RegexTest {
    re: bytes::Regex,
    length: Option<usize>,
    n_pos: Option<usize>,
    mods: FlagSet<ReMod>,
    str_mods: FlagSet<StringMod>,
    // this is actually a search test
    // converted into a regex
    search: bool,
    binary: bool,
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

// FIXME: implement string operators
#[derive(Debug, Clone)]
struct StringTest {
    str: Vec<u8>,
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
            // FIXME: this is wrong
            (false, 1)
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

        (consumed > 0, consumed)
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

    fn matches(&self, buf: &[u8]) -> Option<&[u8]> {
        if let (true, _) = string_match(&self.str, self.mods, buf) {
            Some(&self.str)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct SearchTest {
    str: Vec<u8>,
    n_pos: Option<usize>,
    // FIXME: handle all string mods
    str_mods: FlagSet<StringMod>,
    // FIXME: handle all re mods
    re_mods: FlagSet<ReMod>,
    binary: bool,
}

impl From<SearchTest> for Test {
    fn from(value: SearchTest) -> Self {
        Self::Search(value)
    }
}

impl SearchTest {
    fn matches<'buf>(&self, buf: &'buf [u8]) -> Option<(u64, &'buf [u8])> {
        let mut i = 0;
        while i < buf.len() {
            // we cannot match if the first character isn't the same
            // so we accelerate the search by finding potential matches
            while i < buf.len() && self.str.get(0) != buf.get(i) {
                i += 1
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

#[derive(Debug, Clone)]
struct ScalarTest {
    ty: ScalarDataType,
    transform: Option<ScalarTransform>,
    cmp_op: CmpOp,
    value: Scalar,
}

#[derive(Debug, Clone)]
struct FloatTest {
    ty: FloatDataType,
    transform: Option<FloatTransform>,
    cmp_op: CmpOp,
    value: Float,
}

// the value read from the haystack we want to
// match against
// 'buf is the lifetime of the buffer we are scanning
#[derive(Debug, PartialEq)]
enum TestValue<'buf> {
    Float(u64, Float),
    Scalar(u64, Scalar),
    Bytes(u64, &'buf [u8]),
}

impl DynDisplay for TestValue<'_> {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        match self {
            Self::Float(_, s) => DynDisplay::dyn_fmt(s, f),
            Self::Scalar(_, s) => DynDisplay::dyn_fmt(s, f),
            Self::Bytes(_, b) => Ok(format!("{:?}", b)),
        }
    }
}

impl DynDisplay for &TestValue<'_> {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        // Dereference self to get the TestValue and call its fmt method
        DynDisplay::dyn_fmt(*self, f)
    }
}

impl Display for TestValue<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float(_, v) => write!(f, "{v}"),
            Self::Scalar(_, s) => write!(f, "{s}"),
            Self::Bytes(_, b) => write!(f, "{b:?}"),
        }
    }
}

// Carry the offset of the start of the data in the stream
// and the data itself
enum MatchRes<'buf> {
    // FIXME: maybe we can optimize here by having it as Bytes
    // and managing encoding at display time.
    OwnedString(u64, String),
    String(u64, &'buf str),
    Bytes(u64, &'buf [u8]),
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
            Self::String(_, v) => v.dyn_fmt(f),
            Self::OwnedString(_, v) => v.dyn_fmt(f),
            Self::Bytes(_, v) => Ok(String::from_utf8_lossy(v).to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Encoding {
    Little,
    Big,
}

#[derive(Debug, Clone)]
struct String16Test {
    orig: String,
    str16: Vec<u16>,
    encoding: Encoding,
}

fn slice_to_utf16_iter(read: &[u8], encoding: Encoding) -> impl Iterator<Item = u16> {
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
        Encoding::Little => u16::from_le_bytes([*e, *o]),
        Encoding::Big => u16::from_be_bytes([*e, *o]),
    })
}

flags! {
    enum IndirectMod: u8{
        Relative,
    }
}

type IndirectMods = FlagSet<IndirectMod>;

#[derive(Debug, Clone)]
enum Test {
    /// This corresponds to a DATATYPE x test
    Any(Any),
    Name(String),
    Use(bool, String),
    Scalar(ScalarTest),
    Float(FloatTest),
    String(StringTest),
    Search(SearchTest),
    PString(Vec<u8>),
    Regex(RegexTest),
    Clear,
    Default,
    Indirect(FlagSet<IndirectMod>),
    String16(String16Test),
    // FIXME: placeholders for strength computation
    Der,
}

impl Test {
    // read the value to test from the haystack
    fn read_test_value<'haystack, R: Read + Seek>(
        &self,
        haystack: &'haystack mut LazyCache<R>,
        switch_endianness: bool,
    ) -> Result<TestValue<'haystack>, Error> {
        let test_value_offset = haystack.lazy_stream_position();

        match self {
            Self::Scalar(t) => {
                t.ty.read(haystack, switch_endianness)
                    .map(|s| TestValue::Scalar(test_value_offset, s))
            }
            Self::Float(t) => {
                t.ty.read(haystack, switch_endianness)
                    .map(|f| TestValue::Float(test_value_offset, f))
            }
            Self::String(t) => {
                let buf = if let Some(length) = t.length {
                    // if there is a length specified
                    let read = haystack.read_exact(length as u64)?;
                    read
                } else {
                    // no length specified we read until end of string
                    let read = match t.cmp_op {
                        CmpOp::Eq => {
                            if !t.has_length_mod() {
                                haystack.read_exact(t.str.len() as u64)?
                            } else {
                                haystack.read(FILE_BYTES_MAX as u64)?
                            }
                        }
                        CmpOp::Lt | CmpOp::Gt => {
                            let read = haystack.read_until_any_delim_or_limit(b"\n\0", 8092)?;

                            if read.ends_with(b"\0") || read.ends_with(b"\n") {
                                &read[..read.len() - 1]
                            } else {
                                read
                            }
                        }
                        _ => unimplemented!(),
                    };
                    read
                };

                Ok(TestValue::Bytes(test_value_offset, buf))
            }

            Self::String16(t) => {
                let read = haystack.read_exact((t.str16.len() * 2) as u64)?;
                Ok(TestValue::Bytes(test_value_offset, read))
            }

            Self::PString(buf) => {
                // FIXME: maybe we could optimize here by reading testing on size
                // this is the size of the pstring
                // FIXME: adjust the size function of pstring mods
                let _ = read_le!(haystack, u8);
                let read = haystack.read_exact(buf.len() as u64)?;
                Ok(TestValue::Bytes(test_value_offset, read))
            }

            Self::Search(_) => {
                let buf = haystack.read(FILE_BYTES_MAX as u64)?;
                Ok(TestValue::Bytes(test_value_offset, buf))
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

                        None => {
                            // search tests are made of FILE_BYTES_MAX
                            if r.search {
                                FILE_BYTES_MAX
                            } else {
                                FILE_REGEX_MAX
                            }
                        }
                    }
                };

                let read = haystack.read(length as u64)?;
                Ok(TestValue::Bytes(test_value_offset, read))
            }

            Self::Any(t) => match t {
                Any::String => {
                    // FIXME: Any must carry  StringTest information so we must read accordingly
                    let read = haystack.read_until_any_delim_or_limit(b"\0\n", 8192)?;
                    // we don't take last byte if it matches end of string
                    let bytes = if read.ends_with(b"\0") || read.ends_with(b"\n") {
                        &read[..read.len() - 1]
                    } else {
                        read
                    };

                    Ok(TestValue::Bytes(test_value_offset, bytes))
                }
                Any::PString => {
                    let slen = read_le!(haystack, u8) as usize;
                    let read = haystack.read_exact(slen as u64)?;
                    Ok(TestValue::Bytes(test_value_offset, read))
                }
                Any::String16(_) => {
                    let read = haystack.read_until_utf16_or_limit(b"\x00\x00", 8192)?;

                    // we make sure we have an even number of elements
                    let end = if read.len() % 2 == 0 {
                        read.len()
                    } else {
                        // we decide to read anyway even though
                        // length isn't even
                        read.len().saturating_sub(1)
                    };

                    Ok(TestValue::Bytes(test_value_offset, &read[..end]))
                }
                Any::Scalar(d) => d
                    .read(haystack, switch_endianness)
                    .map(|s| TestValue::Scalar(test_value_offset, s)),
                Any::Float(ty) => ty
                    .read(haystack, switch_endianness)
                    .map(|f| TestValue::Float(test_value_offset, f)),
            },

            // FIXME: all other tests should have been handled
            // before -> make this cleaner
            _ => unimplemented!(),
        }
    }

    #[inline(always)]
    fn match_value<'s>(&'s self, tv: &TestValue<'s>) -> Option<MatchRes<'s>> {
        // always true when we want to read value
        if let Self::Any(v) = self {
            match tv {
                TestValue::Bytes(o, buf) => match v {
                    Any::String | Any::PString => {
                        if let Ok(s) = str::from_utf8(buf) {
                            return Some(MatchRes::String(*o, s));
                        } else {
                            return Some(MatchRes::Bytes(*o, buf));
                        }
                    }

                    Any::String16(enc) => {
                        let utf16_vec: Vec<u16> = slice_to_utf16_iter(buf, *enc).collect();
                        if let Ok(s) = String::from_utf16(&utf16_vec) {
                            return Some(MatchRes::OwnedString(*o, s));
                        } else {
                            return Some(MatchRes::Bytes(*o, buf));
                        }
                    }

                    _ => unimplemented!(),
                },

                TestValue::Scalar(o, s) => {
                    if matches!(v, Any::Scalar(_)) {
                        return Some(MatchRes::Scalar(*o, *s));
                    }
                }
                _ => panic!("not good"),
            }

            // FIXME: remove this
            panic!("any test not properly handled")
        }

        match tv {
            TestValue::Scalar(o, ts) => {
                if let Self::Scalar(t) = self {
                    let read_value: Scalar =
                        t.transform.as_ref().map(|t| t.apply(*ts)).unwrap_or(*ts);

                    let ok = match t.cmp_op {
                        CmpOp::Not => read_value == !t.value,
                        CmpOp::Eq => read_value == t.value,
                        CmpOp::Lt => read_value < t.value,
                        CmpOp::Gt => read_value > t.value,
                        CmpOp::Neq => read_value != t.value,
                        CmpOp::BitAnd => read_value & t.value == t.value,
                        CmpOp::Xor => (read_value & t.value).is_zero(),
                    };

                    if ok {
                        return Some(MatchRes::Scalar(*o, read_value));
                    }
                }
            }
            TestValue::Float(o, f) => {
                if let Self::Float(t) = self {
                    let read_value: Float = t.transform.as_ref().map(|t| t.apply(*f)).unwrap_or(*f);

                    let ok = match t.cmp_op {
                        CmpOp::Eq => read_value == t.value,
                        CmpOp::Lt => read_value < t.value,
                        CmpOp::Gt => read_value > t.value,
                        CmpOp::Neq => read_value != t.value,
                        _ => {
                            debug_panic!("unsupported float comparison");
                            false
                        }
                    };

                    if ok {
                        return Some(MatchRes::Float(*o, read_value));
                    }
                }
            }
            TestValue::Bytes(o, buf) => {
                match self {
                    Self::String(st) => {
                        match st.cmp_op {
                            CmpOp::Eq => {
                                if let Some(b) = st.matches(buf) {
                                    return Some(MatchRes::Bytes(*o, b));
                                }
                            }
                            CmpOp::Gt => {
                                if buf.len() > st.str.len() {
                                    return Some(MatchRes::Bytes(*o, buf));
                                }
                            }
                            CmpOp::Lt => {
                                if buf.len() < st.str.len() {
                                    return Some(MatchRes::Bytes(*o, buf));
                                }
                            }
                            // unsupported for strings
                            _ => {
                                debug_panic!("unsupported cmp operator for string")
                            }
                        }
                    }

                    Self::PString(m) => {
                        if buf == m {
                            return Some(MatchRes::Bytes(*o, buf));
                        }
                    }

                    Self::String16(t) => {
                        // strings cannot be equal
                        if t.str16.len() * 2 != buf.len() {
                            return None;
                        }

                        // we check string equality
                        for (i, utf16_char) in slice_to_utf16_iter(buf, t.encoding).enumerate() {
                            if t.str16[i] != utf16_char {
                                return None;
                            }
                        }

                        return Some(MatchRes::String(*o, &t.orig));
                    }

                    Self::Regex(r) => {
                        if let Some(re_match) = r.re.find(&buf) {
                            if let Some(n_pos) = r.n_pos {
                                // we check for positinal match inherited from search conversion
                                if re_match.start() >= n_pos {
                                    return None;
                                }
                            }

                            return Some(MatchRes::Bytes(
                                // the offset of the string is computed from the start of the buffer
                                o + re_match.start() as u64,
                                re_match.as_bytes(),
                            ));
                        }
                    }

                    Self::Search(t) => {
                        // the offset of the string is computed from the start of the buffer
                        return t.matches(&buf).map(|(p, m)| MatchRes::Bytes(o + p, m));
                    }

                    _ => unimplemented!(),
                }
            }
        }

        None
    }

    //FIXME: complete with all possible operators
    #[inline(always)]
    fn cmp_op(&self) -> Option<CmpOp> {
        match self {
            Self::Scalar(s) => Some(s.cmp_op),
            _ => None,
        }
    }

    #[inline(always)]
    fn is_binary(&self) -> bool {
        match self {
            Self::Any(any) => match any {
                Any::PString => true,
                Any::String => true,
                Any::String16(_) => true,
                Any::Scalar(_) => true,
                Any::Float(_) => true,
            },
            Self::Name(_) => true,
            Self::Use(_, _) => true,
            Self::Scalar(_) => true,
            Self::Float(_) => true,
            Self::String(t) => t.binary || t.mods.contains(StringMod::ForceBin),
            Self::Search(search_test) => {
                search_test.str_mods.contains(StringMod::ForceBin)
                    || search_test.re_mods.contains(ReMod::ForceBinary)
                    || search_test.binary
            }
            Self::PString(_) => true,
            Self::Regex(regex_test) => {
                regex_test.str_mods.contains(StringMod::ForceBin)
                    || regex_test.mods.contains(ReMod::ForceBinary)
                    || regex_test.binary
            }
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
            Self::Any(Any::String) => true,
            Self::Name(_) => true,
            Self::Use(_, _) => true,
            Self::String(t) => !t.binary && t.mods.contains(StringMod::ForceBin),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Shift {
    Direct(u64),
    Indirect(i64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
        opt_start: Option<u64>,
        last_upper_match_offset: Option<u64>,
    ) -> Result<Option<u64>, io::Error> {
        let main_offset_offset = match self.off_addr {
            DirOffset::Start(s) => {
                let Some(o) = s.checked_add(opt_start.unwrap_or_default()) else {
                    return Ok(None);
                };

                haystack.seek(SeekFrom::Start(o))?
            }
            DirOffset::LastUpper(c) => haystack.seek(SeekFrom::Start(
                (last_upper_match_offset.unwrap_or_default() as i64 + c) as u64,
            ))?,
            DirOffset::End(e) => haystack.seek(SeekFrom::End(e as i64))?,
        };

        let offset_pos = haystack.lazy_stream_position();

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
                    OffsetType::Id3Le => unimplemented!(),
                    OffsetType::Id3Be => unimplemented!(),
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
                    OffsetType::Middle => unimplemented!(),
                    OffsetType::Octal => unimplemented!(),
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
            "offset read @ {offset_pos} value={o} op={:?} shift={:?}",
            self.op, self.shift
        );

        // apply transformation
        if let (Some(op), Some(shift)) = (self.op, self.shift) {
            let shift = match shift {
                Shift::Direct(i) => i,
                Shift::Indirect(i) => {
                    let tmp = main_offset_offset as i128 + i as i128;
                    if tmp.is_negative() {
                        return Ok(None);
                    } else {
                        haystack.seek(SeekFrom::Start(tmp as u64))?;
                    };
                    // FIXME: here we assume that the shift has the same
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DirOffset {
    Start(u64),
    // relative to the last up-level field
    LastUpper(i64),
    End(i64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone)]
pub struct Match {
    // FIXME: add file name as &str
    line: usize,
    depth: u8,
    offset: Offset,
    test: Test,
    message: Option<Message>,
}

impl From<Use> for Match {
    fn from(value: Use) -> Self {
        Self {
            line: value.line,
            depth: value.depth,
            offset: value.start_offset,
            test: Test::Use(value.switch_endianness, value.rule_name),
            message: value.message,
        }
    }
}

impl From<Name> for Match {
    fn from(value: Name) -> Self {
        Self {
            line: value.line,
            depth: 0,
            offset: Offset::Direct(DirOffset::Start(0)),
            test: Test::Name(value.name),
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
        opt_start: Option<u64>,
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
                let Some(o) = ind_offset.read_offset(haystack, opt_start, last_level_offset)?
                else {
                    return Ok(None);
                };

                Ok(Some(o))
            }
        }
    }

    // FIXME: handle push_message only once based on the success
    // or not of the test.
    #[inline]
    fn matches<'a, R: Read + Seek>(
        &'a self,
        source: Option<&str>,
        magic: &mut Magic<'a>,
        stream_kind: Option<StreamKind>,
        state: &mut MatchState,
        base_offset: Option<u64>,
        start_offset: Option<u64>,
        last_level_offset: Option<u64>,
        haystack: &mut LazyCache<R>,
        switch_endianness: bool,
        db: &'a MagicDb,
        depth: usize,
    ) -> Result<bool, Error> {
        let source = source.unwrap_or("unknown");

        if let Some(stream_kind) = stream_kind {
            if self.test.is_only_binary() && stream_kind.is_text() {
                trace!(
                    "skip binary test source={source} line={} stream_kind={stream_kind:?}",
                    self.line
                );
                return Ok(false);
            }

            if self.test.is_only_text() && !stream_kind.is_text() {
                trace!(
                    "skip text test source={source} line={} stream_kind={stream_kind:?}",
                    self.line
                );
                return Ok(false);
            }
        }

        if depth >= MAX_RECURSION {
            return Err(Error::MaximumRecursion(MAX_RECURSION));
        }

        let Some(mut offset) = self.offset_from_start(haystack, start_offset, last_level_offset)?
        else {
            return Ok(false);
        };

        offset = match self.offset {
            Offset::Indirect(_) => {
                // offset has been read from stream so we don't want
                // to alter it, unless we decided to re-base
                // the stream after a relative indirect test
                offset.saturating_add(base_offset.unwrap_or_default())
            }
            _ => offset.saturating_add(start_offset.unwrap_or_default()),
        };

        match &self.test {
            Test::Clear => {
                // handle clear and default tests
                trace!("source={source} line={} clear", self.line);
                state.clear_continuation_level(&self.continuation_level());
                Ok(true)
            }
            Test::Name(name) => {
                trace!(
                    "source={source} line={} running rule {name} switch_endianness={switch_endianness}",
                    self.line
                );
                if let Some(msg) = self.message.as_ref() {
                    magic.push_message(msg.format_with(None));
                }
                Ok(true)
            }

            Test::Use(flip_endianness, rule_name) => {
                trace!(
                    "source={source} line={} use {rule_name} switch_endianness={flip_endianness}",
                    self.line
                );

                // switch_endianness must propagate down the rule call stack
                let switch_endianness = switch_endianness ^ flip_endianness;

                let dr: &DependencyRule = db
                    .dependencies
                    .get(rule_name)
                    .ok_or(Error::MissingRule(rule_name.clone()))?;

                if let Some(msg) = self.message.as_ref() {
                    magic.push_message(msg.format_with(None));
                }

                dr.rule.magic(
                    magic,
                    stream_kind,
                    base_offset,
                    Some(offset),
                    haystack,
                    db,
                    switch_endianness,
                    depth.saturating_add(1),
                )?;

                Ok(false)
            }

            Test::Indirect(m) => {
                trace!(
                    "source={source} line={} indirect mods={:?} offset={offset:#x}",
                    self.line, m
                );

                let base_offset = if m.contains(IndirectMod::Relative) {
                    Some(offset)
                } else {
                    None
                };

                for r in db.binary_rules.iter().chain(db.text_rules.iter()) {
                    let messages_cnt = magic.message.len();

                    r.magic(
                        magic,
                        stream_kind,
                        base_offset,
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

                Ok(false)
            }

            Test::Default => {
                // default matches if nothing else at the continuation level matched
                let ok = !state.get_continuation_level(&self.continuation_level());

                trace!("source={source} line={} default match={ok}", self.line);
                if ok {
                    if let Some(msg) = self.message.as_ref() {
                        magic.push_message(msg.format_with(None));
                    }
                    state.set_continuation_level(self.continuation_level());
                }

                Ok(ok)
            }

            _ => {
                haystack.seek(SeekFrom::Start(offset))?;
                let mut trace_msg = None;

                if enabled!(Level::DEBUG) {
                    trace_msg = Some(vec![format!(
                        "source={source} line={} stream_offset={:#x} ",
                        self.line,
                        haystack.stream_position().unwrap_or_default()
                    )])
                }

                // FIXME: we may have a way to optimize here. In case we do a Any
                // test and we don't use the value to format the message, we don't
                // need to read the value.
                if let Ok(tv) = self
                    .test
                    .read_test_value(haystack, switch_endianness)
                    .inspect_err(|e| {
                        trace!(
                            "source={source} line={} error while reading test value: {e}",
                            self.line
                        )
                    })
                {
                    // we need to adjust stream offset if this is a regex test since we read beyond the match
                    let adjust_stream_offset = matches!(&self.test, Test::Regex(_));

                    trace_msg
                        .as_mut()
                        .map(|v| v.push(format!("test={:?}", self.test)));

                    let match_res = self.test.match_value(&tv);

                    trace_msg.as_mut().map(|v| {
                        v.push(format!(
                            "message=\"{}\" match={}",
                            self.message
                                .as_ref()
                                .map(|fs| fs.to_string())
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
                        if let Some(s) = self.message.as_ref() {
                            magic.push_message(s.format_with(Some(&mr)));
                        }
                        // we re-ajust the stream offset only if we have a match
                        if adjust_stream_offset {
                            // we need to compute offset before modifying haystack as
                            // match_res holds a reference to the haystack, not satisfying
                            // borrow checking rules
                            let opt_adjusted_offset = if let MatchRes::Bytes(o, s) = mr {
                                Some(o + s.len() as u64)
                            } else {
                                None
                            };

                            if let Some(o) = opt_adjusted_offset {
                                haystack.seek(SeekFrom::Start(o))?;
                            }
                        } else if let (Test::Search(_), MatchRes::Bytes(o, buf)) = (&self.test, mr)
                        {
                            let opt_adjusted_offset = o + buf.len() as u64;
                            haystack.seek(SeekFrom::Start(opt_adjusted_offset))?;
                        }

                        state.set_continuation_level(self.continuation_level());
                        return Ok(true);
                    }
                }

                Ok(false)
            }
        }
    }

    #[inline(always)]
    fn continuation_level(&self) -> ContinuationLevel {
        //ContinuationLevel(self.depth, self.offset)
        ContinuationLevel(self.depth)
    }

    #[inline(always)]
    fn strength(&self) -> u64 {
        const MULT: usize = 10;

        let mut out = 2 * MULT;

        // FIXME: octal is missing but it is not used in practice ...
        match &self.test {
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

            Test::String(t) => out += t.str.len().saturating_add(MULT),

            Test::PString(t) => out += t.len().saturating_add(MULT),

            Test::Search(s) => out += s.str.len() * max(MULT / s.str.len(), 1),

            Test::Regex(r) => {
                let v = nonmagic(r.re.as_str());
                out += v * max(MULT / v, 1);
            }

            Test::String16(t) => {
                // FIXME: in libmagic the result is div by 2
                // but I GUESS it is because the len is expressed
                // in number bytes. In our case length is expressed
                // in number of u16 so we shouldn't divide.
                out += t.str16.len().saturating_mul(MULT);
            }

            Test::Der => out += MULT,

            // matching any output gets penalty
            Test::Any(_) => out = 0,

            _ => {}
        }

        if let Some(op) = self.test.cmp_op() {
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

#[derive(Debug, Clone)]
struct StrengthMod {
    op: Op,
    by: u8,
}

impl StrengthMod {
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
            Op::Xor | Op::Or => unimplemented!(),
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
    offset: Offset,
    name: String,
    message: Option<Message>,
}

#[derive(Debug, Clone)]
enum Entry {
    Match(Match),
    Flag(Flag),
}

#[derive(Debug, Clone)]
struct EntryNode {
    entry: Match,
    children: Vec<EntryNode>,
    mimetype: Option<String>,
    strength_mod: Option<StrengthMod>,
}

impl EntryNode {
    fn from_entries(entries: Vec<Entry>) -> Self {
        Self::from_peekable(&mut entries.into_iter().peekable())
    }

    fn from_peekable(entries: &mut Peekable<impl Iterator<Item = Entry>>) -> Self {
        let root = match entries.next().unwrap() {
            Entry::Match(m) => m,

            // FIXME: rm dumb panic msg
            Entry::Flag(_) => panic!("should never happen"),
        };

        let mut children = vec![];
        let mut mimetype = None;
        let mut strength_mod = None;

        while let Some(e) = entries.peek() {
            match e {
                Entry::Match(m) => {
                    if m.depth <= root.depth {
                        break;
                    } else if m.depth == root.depth + 1 {
                        children.push(EntryNode::from_peekable(entries))
                    } else {
                        panic!(
                            "unexpected continuation level: line={} level={}",
                            m.line, m.depth
                        )
                    }
                }

                Entry::Flag(_) => {
                    // it cannot be otherwise
                    if let Some(Entry::Flag(f)) = entries.next() {
                        match f {
                            Flag::Mime(m) => mimetype = Some(m),

                            Flag::Strength(s) => strength_mod = Some(s),
                            _ => {
                                // FIXME: implement other flags
                            }
                        }
                    }
                }
            }
        }

        Self {
            entry: root,
            children,
            mimetype,
            strength_mod,
        }
    }

    fn matches<'r, R: Read + Seek>(
        &'r self,
        source: Option<&str>,
        magic: &mut Magic<'r>,
        state: &mut MatchState,
        stream_kind: Option<StreamKind>,
        base_offset: Option<u64>,
        opt_start_offset: Option<u64>,
        last_level_offset: Option<u64>,
        haystack: &mut LazyCache<R>,
        db: &'r MagicDb,
        switch_endianness: bool,
        depth: usize,
    ) -> Result<(), Error> {
        let ok = self.entry.matches(
            source,
            magic,
            stream_kind,
            state,
            base_offset,
            opt_start_offset,
            last_level_offset,
            haystack,
            switch_endianness,
            db,
            depth,
        )?;

        if ok {
            if let Some(mimetype) = self.mimetype.as_ref() {
                magic.insert_mimetype(Cow::Borrowed(mimetype));
            }

            // FIXME: probably strength modifier applies on the magic's
            // strength directly.
            let strength = match self.strength_mod.as_ref() {
                Some(sm) => sm.apply(self.entry.strength()),
                None => self.entry.strength(),
            };

            magic.update_strength(strength);

            let end_upper_level = haystack.lazy_stream_position();

            for e in self.children.iter() {
                e.matches(
                    source,
                    magic,
                    state,
                    stream_kind,
                    base_offset,
                    opt_start_offset,
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

#[derive(Debug, Clone)]
pub struct MagicRule {
    source: Option<String>,
    entries: EntryNode,
}

impl MagicRule {
    fn magic<'r, R: Read + Seek>(
        &'r self,
        magic: &mut Magic<'r>,
        stream_kind: Option<StreamKind>,
        base_offset: Option<u64>,
        opt_start_offset: Option<u64>,
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
            base_offset,
            opt_start_offset,
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

    fn is_binary(&self) -> bool {
        !self.is_text()
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct MagicFile {
    rules: Vec<MagicRule>,
    dependencies: HashMap<String, DependencyRule>,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
struct ContinuationLevel(u8);

// FIXME: magic handles many more text encodings
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum TextEncoding {
    Ascii,
    Utf8,
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
    source: Option<Cow<'m, str>>,
    message: Vec<Cow<'m, str>>,
    mimetype: Option<Cow<'m, str>>,
    strength: Option<u64>,
}

impl<'m> Magic<'m> {
    fn with_source(source: Option<&'m str>) -> Self {
        Self {
            source: source.map(|s| Cow::Borrowed(s)),
            ..Default::default()
        }
    }

    pub fn into_owned<'owned>(self) -> Magic<'owned> {
        Magic {
            source: self.source.map(|s| Cow::Owned(s.into_owned())),
            message: self
                .message
                .into_iter()
                .map(Cow::into_owned)
                .map(Cow::Owned)
                .collect(),
            mimetype: self.mimetype.map(|m| Cow::Owned(m.into_owned())),
            strength: self.strength,
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
    pub fn mimetype(&self) -> &str {
        self.mimetype
            .as_deref()
            .unwrap_or("application/octet-stream")
    }

    #[inline(always)]
    fn push_message<'a: 'm>(&mut self, msg: Cow<'a, str>) {
        if !msg.is_empty() {
            debug!("pushing message: msg={msg} len={}", msg.len());
            self.message.push(msg);
        }
    }

    fn insert_mimetype<'a: 'm>(&mut self, mime: Cow<'a, str>) {
        if self.mimetype.is_none() {
            debug!("insert mime: {:?}", mime);
            self.mimetype = Some(mime)
        }
    }

    pub fn is_empty(&self) -> bool {
        self.message.is_empty() && self.mimetype.is_none() && self.strength.is_none()
    }

    pub fn strength(&self) -> Option<u64> {
        self.strength
    }

    pub fn source(&self) -> Option<&Cow<'m, str>> {
        self.source.as_ref()
    }
}

impl MagicFile {
    pub fn open<P: AsRef<Path>>(p: P) -> Result<Self, Error> {
        FileMagicParser::parse_file(p)
    }
}

#[derive(Debug, Default, Clone)]
pub struct MagicDb {
    binary_rules: Vec<MagicRule>,
    text_rules: Vec<MagicRule>,
    dependencies: HashMap<String, DependencyRule>,
}

#[inline(always)]
fn guess_stream_kind<S: AsRef<[u8]>>(stream: S) -> StreamKind {
    let s = String::from_utf8_lossy(stream.as_ref());
    let count = s.chars().count();
    let mut is_ascii = true;
    for c in s.chars().take(count.saturating_sub(1)) {
        if c == REPLACEMENT_CHARACTER {
            return StreamKind::Binary;
        }
        is_ascii &= c.is_ascii()
    }
    if is_ascii {
        StreamKind::Text(TextEncoding::Ascii)
    } else {
        StreamKind::Text(TextEncoding::Utf8)
    }
}

impl MagicDb {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load(&mut self, mf: MagicFile) -> Result<&mut Self, Error> {
        // it seems rules are evaluated in their reverse definition order
        for rule in mf.rules.into_iter() {
            if rule.is_binary() {
                self.binary_rules.push(rule);
            } else {
                self.text_rules.push(rule)
            }
        }
        self.dependencies.extend(mf.dependencies);
        Ok(self)
    }

    #[inline]
    fn magic_first_with_opt_stream_kind<R: Read + Seek>(
        &self,
        haystack: &mut LazyCache<R>,
        stream_kind: Option<StreamKind>,
    ) -> Result<Option<Magic<'_>>, Error> {
        for rule in self.binary_rules.iter().chain(self.text_rules.iter()) {
            let mut magic = Magic::with_source(rule.source.as_ref().map(|s| s.as_str()));

            rule.magic(
                &mut magic,
                stream_kind,
                None,
                None,
                haystack,
                &self,
                false,
                0,
            )?;

            if !magic.mimetype.is_none() {
                return Ok(Some(magic));
            }
        }

        Ok(None)
    }

    pub fn magic_first<R: Read + Seek>(
        &self,
        haystack: &mut LazyCache<R>,
    ) -> Result<Option<Magic<'_>>, Error> {
        let stream_kind = guess_stream_kind(haystack.read_range(0..4096)?);
        self.magic_first_with_opt_stream_kind(haystack, Some(stream_kind))
    }

    #[inline(always)]
    fn magic_all_with_opt_stream_kind<R: Read + Seek>(
        &self,
        haystack: &mut LazyCache<R>,
        stream_kind: Option<StreamKind>,
    ) -> Result<Vec<(u64, Magic<'_>)>, Error> {
        let mut out = Vec::new();

        for rule in self.binary_rules.iter().chain(self.text_rules.iter()) {
            let mut magic = Magic::with_source(rule.source.as_ref().map(|s| s.as_str()));

            rule.magic(
                &mut magic,
                stream_kind,
                None,
                None,
                haystack,
                &self,
                false,
                0,
            )?;

            // it is possible we have a strength with no message
            if !magic.message.is_empty() {
                out.push((magic.strength.unwrap_or_default(), magic));
            }
        }

        Ok(out)
    }

    pub fn magic_all<R: Read + Seek>(
        &self,
        haystack: &mut LazyCache<R>,
    ) -> Result<Vec<(u64, Magic<'_>)>, Error> {
        let stream_kind = guess_stream_kind(haystack.read_range(0..4096)?);
        self.magic_all_with_opt_stream_kind(haystack, Some(stream_kind))
    }

    #[inline(always)]
    fn magic_best_with_opt_stream_kind<R: Read + Seek>(
        &self,
        haystack: &mut LazyCache<R>,
        stream_kind: Option<StreamKind>,
    ) -> Result<Option<Magic<'_>>, Error> {
        let mut magics = self.magic_all_with_opt_stream_kind(haystack, stream_kind)?;
        magics.sort_by(|a, b| b.0.cmp(&a.0));
        return Ok(magics.into_iter().map(|(_, m)| m).next());
    }

    pub fn magic_best<R: Read + Seek>(
        &self,
        haystack: &mut LazyCache<R>,
    ) -> Result<Option<Magic<'_>>, Error> {
        let stream_kind = guess_stream_kind(haystack.read_range(0..4096)?);
        self.magic_best_with_opt_stream_kind(haystack, Some(stream_kind))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use regex::bytes::Regex;

    use crate::utils::unix_local_time_to_string;

    use super::*;

    fn first_magic(rule: &str, content: &[u8]) -> Result<Option<Magic<'static>>, Error> {
        let mut md = MagicDb::new();
        md.load(
            FileMagicParser::parse_str(rule, None)
                .inspect_err(|e| eprintln!("{e}"))
                .unwrap(),
        )
        .unwrap();
        let mut reader = LazyCache::from_read_seek(Cursor::new(content), 4096, 4 << 20).unwrap();
        let v = md.magic_best_with_opt_stream_kind(&mut reader, None)?;
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

    macro_rules! assert_magic_match {
        ($rule: literal, $content:literal) => {{
            first_magic($rule, $content).unwrap().unwrap();
        }};
        ($rule: literal, $content:literal, $message:expr) => {{
            assert_eq!(
                first_magic($rule, $content).unwrap().unwrap().message(),
                $message
            );
        }};
    }

    macro_rules! assert_magic_not_match {
        ($rule: literal, $content:literal) => {{
            assert!(first_magic($rule, $content).unwrap().is_none());
        }};
    }

    #[test]
    fn test_regex() {
        assert_magic_match!(
            r#"
0	regex/1024 #![[:space:]]*/usr/bin/env[[:space:]]+
!:mime	text/x-shellscript
>&0  regex/64 .*($|\\b) %s shell script text executable
    "#,
            br#"#!/usr/bin/env bash
        echo hello world"#
        );

        let re = Regex::new(r"(?-u)\x42\x82").unwrap();
        assert!(re.is_match(b"\x42\x82"));

        assert_magic_match!(
            r#"0 regex \x42\x82 binary regex match"#,
            b"\x00\x00\x00\x00\x00\x00\x42\x82"
        );

        assert_magic_match!(
            r#"0 search \040\x42\x82 binary regex match"#,
            b"\x00\x00\x00\x00\x00\x20\x42\x82"
        );
    }

    #[test]
    fn test_string_with_mods() {
        assert_magic_match!(
            r#"0	string/w	#!\ \ \ /usr/bin/env\ bash	BASH
        "#,
            b"#! /usr/bin/env bash i
        echo hello world"
        );

        // test uppercase insensitive
        assert_magic_match!(
            r#"0	string/C	HelloWorld	it works
        "#,
            b"helloworld"
        );

        assert_magic_not_match!(
            r#"0	string/C	HelloWorld	it works
        "#,
            b"hELLOwORLD"
        );

        // test lowercase insensitive
        assert_magic_match!(
            r#"0	string/c	HelloWorld	it works
        "#,
            b"HELLOWORLD"
        );

        assert_magic_not_match!(
            r#"0	string/c	HelloWorld	it works
        "#,
            b"helloworld"
        );

        // test full word match
        assert_magic_match!(
            r#"0	string/f	#!/usr/bin/env\ bash	BASH
        "#,
            b"#!/usr/bin/env bash"
        );

        assert_magic_not_match!(
            r#"0	string/f	#!/usr/bin/python PYTHON"#,
            b"#!/usr/bin/pythonic"
        );

        // testing whitespace compacting
        assert_magic_match!(
            r#"0	string/W	#!/usr/bin/env\ python  PYTHON"#,
            b"#!/usr/bin/env    python"
        );

        assert_magic_not_match!(
            r#"0	string/W	#!/usr/bin/env\ \ python  PYTHON"#,
            b"#!/usr/bin/env python"
        )
    }

    #[test]
    fn test_search_with_mods() {
        assert_magic_match!(
            r#"0	search/1/fwt	#!\ /usr/bin/luatex	LuaTex script text executable"#,
            b"#!          /usr/bin/luatex "
        );
    }

    #[test]
    fn test_max_recursion() {
        let res = first_magic(r#"0	indirect x"#, b"#!          /usr/bin/luatex ");
        assert!(matches!(res, Err(Error::MaximumRecursion(MAX_RECURSION))));
    }

    #[test]
    fn test_string_ops() {
        assert_magic_match!("0	string/b MZ MZ File", b"MZ\0");
        assert_magic_match!("0	string >\0 Any String", b"A\0");
        assert_magic_match!("0	string >Test Any String", b"Test 1\0");
        assert_magic_match!("0	string <Test Any String", b"\0");
        assert_magic_not_match!("0	string >Test Any String", b"\0");
    }

    #[test]
    fn test_lestring16() {
        assert_magic_match!(
            "0 lestring16 abcd Little-endian UTF-16 string",
            b"\x61\x00\x62\x00\x63\x00\x64\x00"
        );
        assert_magic_match!(
            "0 lestring16 x %s",
            b"\x61\x00\x62\x00\x63\x00\x64\x00\x00",
            "abcd"
        );
        assert_magic_not_match!(
            "0 lestring16 abcd Little-endian UTF-16 string",
            b"\x00\x61\x00\x62\x00\x63\x00\x64"
        );
        assert_magic_match!(
            "4 lestring16 abcd Little-endian UTF-16 string",
            b"\x00\x00\x00\x00\x61\x00\x62\x00\x63\x00\x64\x00"
        );
    }

    #[test]
    fn test_bestring16() {
        assert_magic_match!(
            "0 bestring16 abcd Big-endian UTF-16 string",
            b"\x00\x61\x00\x62\x00\x63\x00\x64"
        );
        assert_magic_match!(
            "0 bestring16 x %s",
            b"\x00\x61\x00\x62\x00\x63\x00\x64",
            "abcd"
        );
        assert_magic_not_match!(
            "0 bestring16 abcd Big-endian UTF-16 string",
            b"\x61\x00\x62\x00\x63\x00\x64\x00"
        );
        assert_magic_match!(
            "4 bestring16 abcd Big-endian UTF-16 string",
            b"\x00\x00\x00\x00\x00\x61\x00\x62\x00\x63\x00\x64"
        );
    }

    #[test]
    fn test_offset_from_end() {
        assert_magic_match!("-1 ubyte 0x42 last byte ok", b"\x00\x00\x42");
        assert_magic_match!("-2 ubyte 0x41 last byte ok", b"\x00\x41\x00");
    }

    #[test]
    fn test_relative_offset() {
        assert_magic_match!(
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
        assert_magic_match!("(0.l) ubyte 0x42 it works", b"\x04\x00\x00\x00\x42");
        // adding fixed value to offset
        assert_magic_match!("(0.l+3) ubyte 0x42 it works", b"\x01\x00\x00\x00\x42");
        // testing offset pair
        assert_magic_match!(
            "(0.l+(4)) ubyte 0x42 it works",
            b"\x04\x00\x00\x00\x04\x00\x00\x00\x42"
        );
    }

    #[test]
    fn test_use_with_message() {
        assert_magic_match!(
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
        assert_magic_match!("0 ubyte+1 0x1 add works", b"\x00");
        assert_magic_match!("0 ubyte-1 0xfe sub works", b"\xff");
        assert_magic_match!("0 ubyte%2 0 mod works", b"\x0a");
        assert_magic_match!("0 ubyte&0x0f 0x0f bitand works", b"\xff");
        assert_magic_match!("0 ubyte|0x0f 0xff bitor works", b"\xf0");
        assert_magic_match!("0 ubyte^0x0f 0xf0 bitxor works", b"\xff");

        FileMagicParser::parse_str("0 ubyte%0 mod by zero", None)
            .expect_err("expect div by zero error");
        FileMagicParser::parse_str("0 ubyte/0 div by zero", None)
            .expect_err("expect div by zero error");
    }

    #[test]
    fn test_belong() {
        // Test that a file with a four-byte value at offset 0 that matches the given value in big-endian byte order
        assert_magic_match!("0 belong 0x12345678 Big-endian long", b"\x12\x34\x56\x78");
        // Test that a file with a four-byte value at offset 0 that does not match the given value in big-endian byte order
        assert_magic_not_match!("0 belong 0x12345678 Big-endian long", b"\x78\x56\x34\x12");
        // Test that a file with a four-byte value at a non-zero offset that matches the given value in big-endian byte order
        assert_magic_match!(
            "4 belong 0x12345678 Big-endian long",
            b"\x00\x00\x00\x00\x12\x34\x56\x78"
        );
        // Test < operator
        assert_magic_match!("0 belong <0x12345678 Big-endian long", b"\x12\x34\x56\x77");
        assert_magic_not_match!("0 belong <0x12345678 Big-endian long", b"\x12\x34\x56\x78");

        // Test > operator
        assert_magic_match!("0 belong >0x12345678 Big-endian long", b"\x12\x34\x56\x79");
        assert_magic_not_match!("0 belong >0x12345678 Big-endian long", b"\x12\x34\x56\x78");

        // Test & operator
        assert_magic_match!("0 belong &0x5678 Big-endian long", b"\x00\x00\x56\x78");
        assert_magic_not_match!("0 belong &0x0000FFFF Big-endian long", b"\x12\x34\x56\x78");

        // Test ^ operator (bitwise AND with complement)
        assert_magic_match!("0 belong ^0xFFFF0000 Big-endian long", b"\x00\x00\x56\x78");
        assert_magic_not_match!("0 belong ^0xFFFF0000 Big-endian long", b"\x00\x01\x56\x78");

        // Test ~ operator
        assert_magic_match!("0 belong ~0x12345678 Big-endian long", b"\xed\xcb\xa9\x87");
        assert_magic_not_match!("0 belong ~0x12345678 Big-endian long", b"\x12\x34\x56\x78");

        // Test x operator
        assert_magic_match!("0 belong x Big-endian long", b"\x12\x34\x56\x78");
        assert_magic_match!("0 belong x Big-endian long", b"\x78\x56\x34\x12");
    }

    #[test]
    fn test_parse_search() {
        parse_assert!("0 search test");
        parse_assert!("0 search/24/s test");
        parse_assert!("0 search/s/24 test");
    }

    #[test]
    fn test_bedate() {
        assert_magic_match!(
            "0 bedate 946684800 Unix date (Jan 1, 2000)",
            b"\x38\x6D\x43\x80"
        );
        assert_magic_not_match!(
            "0 bedate 946684800 Unix date (Jan 1, 2000)",
            b"\x00\x00\x00\x00"
        );
        assert_magic_match!(
            "4 bedate 946684800 %s",
            b"\x00\x00\x00\x00\x38\x6D\x43\x80",
            "2000-01-01 00:00:00"
        );
    }
    #[test]
    fn test_beldate() {
        assert_magic_match!(
            "0 beldate 946684800 Local date (Jan 1, 2000)",
            b"\x38\x6D\x43\x80"
        );
        assert_magic_not_match!(
            "0 beldate 946684800 Local date (Jan 1, 2000)",
            b"\x00\x00\x00\x00"
        );

        assert_magic_match!(
            "4 beldate 946684800 {}",
            b"\x00\x00\x00\x00\x38\x6D\x43\x80",
            unix_local_time_to_string(946684800)
        );
    }

    #[test]
    fn test_beqdate() {
        assert_magic_match!(
            "0 beqdate 946684800 Unix date (Jan 1, 2000)",
            b"\x00\x00\x00\x00\x38\x6D\x43\x80"
        );

        assert_magic_not_match!(
            "0 beqdate 946684800 Unix date (Jan 1, 2000)",
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        );

        assert_magic_match!(
            "0 beqdate 946684800 %s",
            b"\x00\x00\x00\x00\x38\x6D\x43\x80",
            "2000-01-01 00:00:00"
        );
    }

    #[test]
    fn test_medate() {
        assert_magic_match!(
            "0 medate 946684800 Unix date (Jan 1, 2000)",
            b"\x6D\x38\x80\x43"
        );

        assert_magic_not_match!(
            "0 medate 946684800 Unix date (Jan 1, 2000)",
            b"\x00\x00\x00\x00"
        );

        assert_magic_match!(
            "4 medate 946684800 %s",
            b"\x00\x00\x00\x00\x6D\x38\x80\x43",
            "2000-01-01 00:00:00"
        );
    }

    #[test]
    fn test_meldate() {
        assert_magic_match!(
            "0 meldate 946684800 Local date (Jan 1, 2000)",
            b"\x6D\x38\x80\x43"
        );
        assert_magic_not_match!(
            "0 meldate 946684800 Local date (Jan 1, 2000)",
            b"\x00\x00\x00\x00"
        );

        assert_magic_match!(
            "4 meldate 946684800 %s",
            b"\x00\x00\x00\x00\x6D\x38\x80\x43",
            unix_local_time_to_string(946684800)
        );
    }

    #[test]
    fn test_date() {
        assert_magic_match!(
            "0 date 946684800 Local date (Jan 1, 2000)",
            b"\x80\x43\x6D\x38"
        );
        assert_magic_not_match!(
            "0 date 946684800 Local date (Jan 1, 2000)",
            b"\x00\x00\x00\x00"
        );
        assert_magic_match!(
            "4 date 946684800 {}",
            b"\x00\x00\x00\x00\x80\x43\x6D\x38",
            "2000-01-01 00:00:00"
        );
    }

    #[test]
    fn test_leldate() {
        assert_magic_match!(
            "0 leldate 946684800 Local date (Jan 1, 2000)",
            b"\x80\x43\x6D\x38"
        );
        assert_magic_not_match!(
            "0 leldate 946684800 Local date (Jan 1, 2000)",
            b"\x00\x00\x00\x00"
        );
        assert_magic_match!(
            "4 leldate 946684800 {}",
            b"\x00\x00\x00\x00\x80\x43\x6D\x38",
            unix_local_time_to_string(946684800)
        );
    }

    #[test]
    fn test_leqdate() {
        assert_magic_match!(
            "0 leqdate 1577836800 Unix date (Jan 1, 2020)",
            b"\x00\xe1\x0b\x5E\x00\x00\x00\x00"
        );

        assert_magic_not_match!(
            "0 leqdate 1577836800 Unix date (Jan 1, 2020)",
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        );
        assert_magic_match!(
            "8 leqdate 1577836800 %s",
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\xE1\x0B\x5E\x00\x00\x00\x00",
            "2020-01-01 00:00:00"
        );
    }

    #[test]
    fn test_leqldate() {
        assert_magic_match!(
            "0 leqldate 1577836800 Unix date (Jan 1, 2020)",
            b"\x00\xe1\x0b\x5E\x00\x00\x00\x00"
        );

        assert_magic_not_match!(
            "0 leqldate 1577836800 Unix date (Jan 1, 2020)",
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        );
        assert_magic_match!(
            "8 leqldate 1577836800 %s",
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\xE1\x0B\x5E\x00\x00\x00\x00",
            unix_local_time_to_string(1577836800)
        );
    }

    #[test]
    fn test_melong() {
        // Test = operator
        assert_magic_match!(
            "0 melong =0x12345678 Middle-endian long",
            b"\x34\x12\x78\x56"
        );
        assert_magic_not_match!(
            "0 melong =0x12345678 Middle-endian long",
            b"\x00\x00\x00\x00"
        );

        // Test < operator
        assert_magic_match!(
            "0 melong <0x12345678 Middle-endian long",
            b"\x34\x12\x78\x55"
        ); // 0x12345677 in middle-endian
        assert_magic_not_match!(
            "0 melong <0x12345678 Middle-endian long",
            b"\x34\x12\x78\x56"
        ); // 0x12345678 in middle-endian

        // Test > operator
        assert_magic_match!(
            "0 melong >0x12345678 Middle-endian long",
            b"\x34\x12\x78\x57"
        ); // 0x12345679 in middle-endian
        assert_magic_not_match!(
            "0 melong >0x12345678 Middle-endian long",
            b"\x34\x12\x78\x56"
        ); // 0x12345678 in middle-endian

        // Test & operator
        assert_magic_match!("0 melong &0x5678 Middle-endian long", b"\xab\xcd\x78\x56"); // 0x00007856 in middle-endian
        assert_magic_not_match!(
            "0 melong &0x0000FFFF Middle-endian long",
            b"\x34\x12\x78\x56"
        ); // 0x12347856 in middle-endian

        // Test ^ operator (bitwise AND with complement)
        assert_magic_match!(
            "0 melong ^0xFFFF0000 Middle-endian long",
            b"\x00\x00\x78\x56"
        ); // 0x00007856 in middle-endian
        assert_magic_not_match!(
            "0 melong ^0xFFFF0000 Middle-endian long",
            b"\x00\x01\x78\x56"
        ); // 0x00017856 in middle-endian

        // Test ~ operator
        assert_magic_match!(
            "0 melong ~0x12345678 Middle-endian long",
            b"\xCB\xED\x87\xA9"
        );
        assert_magic_not_match!(
            "0 melong ~0x12345678 Middle-endian long",
            b"\x34\x12\x78\x56"
        ); // The original value

        // Test x operator
        assert_magic_match!("0 melong x Middle-endian long", b"\x34\x12\x78\x56");
        assert_magic_match!("0 melong x Middle-endian long", b"\x00\x00\x00\x00");
    }

    #[test]
    fn test_uquad() {
        // Test = operator
        assert_magic_match!(
            "0 uquad =0x123456789ABCDEF0 Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        );
        assert_magic_not_match!(
            "0 uquad =0x123456789ABCDEF0 Unsigned quad",
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        );

        // Test < operator
        assert_magic_match!(
            "0 uquad <0x123456789ABCDEF0 Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x11"
        );
        assert_magic_not_match!(
            "0 uquad <0x123456789ABCDEF0 Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        );

        // Test > operator
        assert_magic_match!(
            "0 uquad >0x123456789ABCDEF0 Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x13"
        );
        assert_magic_not_match!(
            "0 uquad >0x123456789ABCDEF0 Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        );

        // Test & operator
        assert_magic_match!(
            "0 uquad &0xF0 Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        );
        assert_magic_not_match!(
            "0 uquad &0xFF Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        );

        // Test ^ operator (bitwise AND with complement)
        assert_magic_match!(
            "0 uquad ^0xFFFFFFFFFFFFFFFF Unsigned quad",
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        ); // All bits clear
        assert_magic_not_match!(
            "0 uquad ^0xFFFFFFFFFFFFFFFF Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        ); // Some bits set

        // Test ~ operator
        assert_magic_match!(
            "0 uquad ~0x123456789ABCDEF0 Unsigned quad",
            b"\x0F\x21\x43\x65\x87\xA9\xCB\xED"
        );
        assert_magic_not_match!(
            "0 uquad ~0x123456789ABCDEF0 Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        ); // The original value

        // Test x operator
        assert_magic_match!(
            "0 uquad x {:#x}",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12",
            "0x123456789abcdef0"
        );
        assert_magic_match!(
            "0 uquad x Unsigned quad",
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        );
    }

    #[test]
    fn test_guid() {
        assert_magic_match!(
            "0 guid EC959539-6786-2D4E-8FDB-98814CE76C1E It works",
            b"\xEC\x95\x95\x39\x67\x86\x2D\x4E\x8F\xDB\x98\x81\x4C\xE7\x6C\x1E"
        );

        assert_magic_not_match!(
            "0 guid 399595EC-8667-4E2D-8FDB-98814CE76C1E It works",
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F"
        );

        assert_magic_match!(
            "0 guid x %s",
            b"\xEC\x95\x95\x39\x67\x86\x2D\x4E\x8F\xDB\x98\x81\x4C\xE7\x6C\x1E",
            "EC959539-6786-2D4E-8FDB-98814CE76C1E"
        );
    }

    #[test]
    fn test_ubeqdate() {
        assert_magic_match!(
            "0 ubeqdate 1633046400 It works",
            b"\x00\x00\x00\x00\x61\x56\x4f\x80"
        );

        assert_magic_match!(
            "0 ubeqdate x %s",
            b"\x00\x00\x00\x00\x61\x56\x4f\x80",
            "2021-10-01 00:00:00"
        );

        assert_magic_not_match!(
            "0 ubeqdate 1633046400 It should not work",
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        );
    }

    #[test]
    fn test_ldate() {
        assert_magic_match!("0 ldate 1640551520 It works", b"\x60\xd4\xC8\x61");

        assert_magic_not_match!("0 ldate 1633046400 It should not work", b"\x00\x00\x00\x00");

        assert_magic_match!(
            "0 ldate x %s",
            b"\x60\xd4\xC8\x61",
            unix_local_time_to_string(1640551520)
        );
    }
}
