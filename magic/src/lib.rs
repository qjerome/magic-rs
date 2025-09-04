#![deny(unsafe_code)]

use chrono::{DateTime, Local, TimeZone};
use dyf::{DynDisplay, FormatString, dformat};
use flagset::{FlagSet, flags};
use lazy_cache::LazyCache;
use pest::{Parser, Span, error::ErrorVariant, iterators::Pair};
use pest_derive::Parser;
use regex::bytes::{self, Regex};
use std::{
    borrow::Cow,
    cmp::max,
    collections::{HashMap, HashSet},
    fmt::{self, Debug, Display},
    fs,
    io::{self, Read, Seek, SeekFrom},
    iter::Peekable,
    ops::{Add, BitAnd, BitXor, Div, Mul, Not, Rem, Sub},
    path::Path,
    str::Utf8Error,
};
use thiserror::Error;
use tracing::{Level, debug, enabled, error, trace};

use crate::utils::nonmagic;

mod numeric;
mod utils;

use numeric::{Scalar, ScalarDataType};

// corresponds to FILE_INDIR_MAX constant defined in libmagic
const MAX_RECURSION: usize = 50;
// constant found in libmagic. It is used to limit for search tests
const FILE_BYTES_MAX: usize = 7 * 1024 * 1024;
// constant found in libmagic. It is used to limit for regex tests
const FILE_REGEX_MAX: usize = 8192;

const TIMESTAMP_FORMAT: &'static str = "%Y-%m-%d %H:%M:%S";

macro_rules! debug_panic {
    ($($arg:tt)*) => {
        if cfg!(debug_assertions) {
            panic!($($arg)*);
        }
    };
}

#[derive(Parser)]
#[grammar = "grammar.pest"]
struct FileMagicParser;

fn unescape_string(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        // string termination
        if c == '\0' {
            return result;
        }

        if c == '\\' {
            if let Some(next_char) = chars.peek() {
                match next_char {
                    // string termination
                    'n' => {
                        result.push('\n');
                        chars.next(); // Skip the 'n'
                    }
                    't' => {
                        result.push('\t');
                        chars.next(); // Skip the 't'
                    }
                    'r' => {
                        result.push('\r');
                        chars.next(); // Skip the 'r'
                    }
                    '\\' => {
                        result.push('\\');
                        chars.next(); // Skip the '\\'
                    }
                    'x' => {
                        // Handle hex escape sequences (e.g., \x7F)
                        chars.next(); // Skip the 'x'

                        let mut hex_str = String::new();
                        for _ in 0..2 {
                            if chars
                                .peek()
                                .map(|c| c.is_ascii_hexdigit())
                                .unwrap_or_default()
                            {
                                hex_str.push(chars.next().unwrap());
                                continue;
                            }
                            break;
                        }

                        if let Ok(hex) = u8::from_str_radix(&hex_str, 16) {
                            result.push(hex as char);
                        } else {
                            result.push(c); // Push the backslash if the hex sequence is invalid
                        }
                    }
                    // Handle octal escape sequences (e.g., \1 \23 \177)
                    '0'..='7' => {
                        let mut octal_str = String::new();
                        for _ in 0..3 {
                            if chars
                                .peek()
                                .map(|c| matches!(c, '0'..='7'))
                                .unwrap_or_default()
                            {
                                octal_str.push(chars.next().unwrap());
                                continue;
                            }
                            break;
                        }
                        //let octal_str: String = chars.by_ref().take(1).collect();
                        if let Ok(octal) = u8::from_str_radix(&octal_str, 8) {
                            result.push(octal as char);
                        } else {
                            result.push(c); // Push the backslash if the octal sequence is invalid
                        }
                    }
                    _ => {
                        // we skip the backslash
                    }
                }
            } else {
                result.push(c); // Push the backslash if no character follows
            }
        } else {
            result.push(c);
        }
    }

    result
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
    fn convert_printf_to_rust_format(c_format: &str) -> (String, Option<String>) {
        let mut rust_format = String::new();
        let mut chars = c_format.chars().peekable();
        let mut printf_spec = None;

        while let Some(c) = chars.next() {
            if c == '%' {
                // Handle format specifier
                let mut specifier = String::new();
                let mut hash_flag = false;

                // Check for flags like #
                if let Some(&'#') = chars.peek() {
                    chars.next();
                    hash_flag = true;
                }

                // Read the rest of the specifier
                while let Some(&next_char) = chars.peek() {
                    if next_char.is_alphabetic() {
                        specifier.push(chars.next().unwrap());
                        break;
                    } else {
                        specifier.push(chars.next().unwrap());
                    }
                }

                // Convert C format specifier to Rust format specifier
                let rust_specifier = match specifier.as_str() {
                    "d" | "i" => "{}",
                    "x" => {
                        if hash_flag {
                            "0x{:x}"
                        } else {
                            "{:x}"
                        }
                    }
                    "f" => "{}",
                    "s" => "{}",
                    "o" => "{:o}",
                    // Add more C format specifier conversions here if needed
                    _ => "{}", // Default case
                };

                printf_spec = Some(specifier);

                // Append the converted specifier
                rust_format.push_str(rust_specifier);
            } else {
                rust_format.push(c);
            }
        }

        (rust_format, printf_spec)
    }

    fn from_pair(pair: Pair<'_, Rule>) -> Self {
        assert_eq!(pair.as_rule(), Rule::message);
        Self::from_str(pair.as_str())
    }

    #[inline]
    fn from_str<S: AsRef<str>>(s: S) -> Message {
        let (s, printf_spec) = Self::convert_printf_to_rust_format(s.as_ref());

        // FIXME: remove unwrap
        let fs = FormatString::from_string(s.to_string()).unwrap();
        if fs.contains_format() {
            Message::Format {
                printf_spec: printf_spec.unwrap_or_default(),
                fs,
            }
        } else {
            Message::String(fs.into_string())
        }
    }

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
                        MatchRes::String(_, _) => {
                            // FIXME: fix unwrap
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

impl FileMagicParser {
    fn parse_str<S: AsRef<str>>(s: S) -> Result<MagicFile, Error> {
        let pairs = FileMagicParser::parse(Rule::file, s.as_ref()).map_err(Box::new)?;

        let mut rules = vec![];
        let mut dependencies = HashMap::new();
        for file in pairs {
            for rule in file.into_inner() {
                match rule.as_rule() {
                    Rule::rule => {
                        rules.push(MagicRule::from_pair(rule)?);
                    }
                    Rule::rule_dependency => {
                        let d = DependencyRule::from_pair(rule)?;
                        dependencies.insert(d.name.clone(), d);
                    }
                    Rule::EOI => {}
                    _ => return Err(Error::parser("unexpected rule", rule.as_span())),
                }
            }
        }

        Ok(MagicFile {
            rules,
            dependencies,
        })
    }

    fn parse_file<P: AsRef<Path>>(p: P) -> Result<MagicFile, Error> {
        let s = fs::read_to_string(p)?;
        Self::parse_str(s)
    }
}

impl DynDisplay for Scalar {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        match self {
            Scalar::quad(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::belong(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::bequad(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::beshort(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::bedate(value) => Ok(DateTime::from_timestamp(*value as i64, 0)
                .map(|ts| ts.format(TIMESTAMP_FORMAT).to_string())
                .unwrap_or("invalid timestamp".into())),
            Scalar::beqdate(value) => Ok(DateTime::from_timestamp(*value, 0)
                .map(|ts| ts.format(TIMESTAMP_FORMAT).to_string())
                .unwrap_or("invalid timestamp".into())),
            Scalar::byte(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::ledate(value) => Ok(DateTime::from_timestamp(*value as i64, 0)
                .map(|ts| ts.format(TIMESTAMP_FORMAT).to_string())
                .unwrap_or("invalid timestamp".into())),
            Scalar::lelong(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::leshort(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::lequad(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::long(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::short(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::ushort(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::ulong(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::uquad(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::ubelong(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::ubequad(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::ubeshort(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::ubyte(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::ulelong(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::ulequad(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::uleshort(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::uledate(value) => Ok(DateTime::from_timestamp(*value as i64, 0)
                .map(|ts| ts.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or("invalid timestamp".into())),
            Self::offset(value) => DynDisplay::dyn_fmt(value, f),
            Self::lemsdosdate(value) => Ok(format!("mdosdate({})", value)),
            Self::lemsdostime(value) => Ok(format!("mdostime({})", value)),
            Scalar::medate(value) => Ok(DateTime::from_timestamp(*value as i64, 0)
                .map(|ts| ts.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or("invalid timestamp".into())),
            Scalar::meldate(value) => Ok(Local
                .timestamp_opt(*value as i64, 0)
                .earliest()
                .map(|ts| ts.naive_local().format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or("invalid timestamp".into())),
            Scalar::melong(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::leqdate(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::offset(_) => todo!(),
            Scalar::lemsdosdate(_) => todo!(),
            Scalar::lemsdostime(_) => todo!(),
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Scalar::quad(value) => write!(f, "{}", value),
            Scalar::uquad(value) => write!(f, "{}", value),
            Scalar::belong(value) => write!(f, "{}", value),
            Scalar::bequad(value) => write!(f, "{}", value),
            Scalar::beshort(value) => write!(f, "{}", value),
            Scalar::bedate(value) => write!(f, "bedate({})", value),
            Scalar::beqdate(value) => write!(f, "beqdate({})", value),
            Scalar::byte(value) => write!(f, "{}", value),
            Scalar::ledate(value) => write!(f, "ledate({})", value),
            Scalar::lelong(value) => write!(f, "{}", value),
            Scalar::leshort(value) => write!(f, "{}", value),
            Scalar::lequad(value) => write!(f, "{}", value),
            Scalar::long(value) => write!(f, "{}", value),
            Scalar::short(value) => write!(f, "{}", value),
            Scalar::ushort(value) => write!(f, "{}", value),
            Scalar::ulong(value) => write!(f, "{}", value),
            Scalar::ubelong(value) => write!(f, "{}", value),
            Scalar::ubequad(value) => write!(f, "{}", value),
            Scalar::ubeshort(value) => write!(f, "{}", value),
            Scalar::ubyte(value) => write!(f, "{}", value),
            Scalar::ulelong(value) => write!(f, "{}", value),
            Scalar::ulequad(value) => write!(f, "{}", value),
            Scalar::uleshort(value) => write!(f, "{}", value),
            Scalar::uledate(value) => write!(f, "uledate({})", value),
            Scalar::offset(value) => write!(f, "{:p}", value),
            Scalar::lemsdosdate(value) => write!(f, "lemsdosdate({})", value),
            Scalar::lemsdostime(value) => write!(f, "lemsdostime({})", value),
            Scalar::medate(value) => write!(f, "medate({})", value),
            Scalar::melong(value) => write!(f, "{}", value),
            Scalar::meldate(value) => write!(f, "meldate({})", value),
            Scalar::leqdate(value) => write!(f, "{}", value),
        }
    }
}

impl ScalarDataType {
    fn from_pair(pair: Pair<'_, Rule>) -> Result<Self, Error> {
        let dt = pair.into_inner().next().expect("data type expected");
        match dt.as_rule() {
            Rule::belong => Ok(Self::belong),
            Rule::bequad => Ok(Self::bequad),
            Rule::beshort => Ok(Self::beshort),
            Rule::bedate => Ok(Self::bedate),
            Rule::beqdate => Ok(Self::beqdate),
            Rule::byte => Ok(Self::byte),
            Rule::quad => Ok(Self::quad),
            Rule::uquad => Ok(Self::uquad),
            Rule::lelong => Ok(Self::lelong),
            Rule::ledate => Ok(Self::ledate),
            Rule::leqdate => Ok(Self::leqdate),
            Rule::leshort => Ok(Self::leshort),
            Rule::long => Ok(Self::long),
            Rule::short => Ok(Self::short),
            Rule::ushort => Ok(Self::ushort),
            Rule::ulong => Ok(Self::ulong),
            Rule::ubelong => Ok(Self::ubelong),
            Rule::ubequad => Ok(Self::ubequad),
            Rule::ubeshort => Ok(Self::ubeshort),
            Rule::ubyte => Ok(Self::ubyte),
            Rule::ulelong => Ok(Self::ulelong),
            Rule::ulequad => Ok(Self::ulequad),
            Rule::uleshort => Ok(Self::uleshort),
            Rule::lequad => Ok(Self::lequad),
            Rule::uledate => Ok(Self::uledate),
            Rule::offset_ty => Ok(Self::offset),
            Rule::lemsdosdate => Ok(Self::lemsdosdate),
            Rule::lemsdostime => Ok(Self::lemsdostime),
            Rule::medate => Ok(Self::medate),
            Rule::meldate => Ok(Self::meldate),
            Rule::melong => Ok(Self::melong),
            _ => Err(Error::parser("unimplemented data type", dt.as_span())),
        }
    }

    #[inline(always)]
    fn read<R: Read + Seek>(
        &self,
        from: &mut LazyCache<R>,
        switch_endianness: bool,
    ) -> Result<Scalar, Error> {
        macro_rules! read {
            ($ty: ty) => {{
                let mut a = [0u8; std::mem::size_of::<$ty>()];
                // it is accepted to copy bytes here as we
                // handle only primitive types
                from.read_exact_into(&mut a)?;
                a
            }};
        }

        macro_rules! read_le {
            ($ty: ty) => {{
                if switch_endianness {
                    <$ty>::from_be_bytes(read!($ty))
                } else {
                    <$ty>::from_le_bytes(read!($ty))
                }
            }};
        }

        macro_rules! read_be {
            ($ty: ty) => {{
                if switch_endianness {
                    <$ty>::from_le_bytes(read!($ty))
                } else {
                    <$ty>::from_be_bytes(read!($ty))
                }
            }};
        }

        macro_rules! read_ne {
            ($ty: ty) => {{
                if cfg!(target_endian = "big") {
                    read_be!($ty)
                } else {
                    read_le!($ty)
                }
            }};
        }

        macro_rules! read_me {
            () => {
                ((read_le!(u16) as i32) << 16) | (read_le!(u16) as i32)
            };
        }

        Ok(match self {
            // signed
            Self::byte => Scalar::byte(read!(u8)[0] as i8),
            Self::short => Scalar::short(read_ne!(i16)),
            Self::long => Scalar::long(read_ne!(i32)),
            Self::leshort => Scalar::leshort(read_le!(i16)),
            Self::lelong => Scalar::lelong(read_le!(i32)),
            Self::lequad => Scalar::lequad(read_le!(i64)),
            Self::bequad => Scalar::bequad(read_be!(i64)),
            Self::belong => Scalar::belong(read_be!(i32)),
            Self::bedate => Scalar::bedate(read_be!(i32)),
            Self::beqdate => Scalar::beqdate(read_be!(i64)),
            // unsigned
            Self::ubyte => Scalar::ubyte(read!(u8)[0]),
            Self::ushort => Scalar::ushort(read_ne!(u16)),
            Self::uleshort => Scalar::uleshort(read_le!(u16)),
            Self::ulelong => Scalar::ulelong(read_le!(u32)),
            Self::uledate => Scalar::uledate(read_le!(u32)),
            Self::ulequad => Scalar::ulequad(read_le!(u64)),
            Self::offset => Scalar::offset(from.stream_position()?),
            Self::ubequad => Scalar::ubequad(read_be!(u64)),
            Self::medate => Scalar::medate(read_me!()),
            Self::meldate => Scalar::meldate(read_me!()),
            Self::melong => Scalar::melong(read_me!()),
            Self::beshort => Scalar::beshort(read_be!(i16)),
            Self::quad => Scalar::quad(read_ne!(i64)),
            Self::uquad => Scalar::uquad(read_ne!(u64)),
            Self::ledate => Scalar::ledate(read_le!(i32)),
            Self::leqdate => Scalar::leqdate(read_le!(i64)),
            Self::ubelong => Scalar::ubelong(read_be!(u32)),
            Self::ulong => Scalar::ulong(read_ne!(u32)),
            Self::ubeshort => Scalar::ubeshort(read_be!(u16)),
            Self::lemsdosdate => Scalar::lemsdosdate(read_le!(u16)),
            Self::lemsdostime => Scalar::lemsdostime(read_le!(u16)),
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
    Neg, // ! operator
    Xor,
    // FIXME: this operator might be useless
    // it could be turned into Eq and transforming
    // the test value
    Not, // ~ operator
}

impl CmpOp {
    fn from_pair(value: Pair<'_, Rule>) -> Result<Self, Error> {
        match value.as_rule() {
            Rule::op_lt => Ok(Self::Lt),
            Rule::op_gt => Ok(Self::Gt),
            Rule::op_and => Ok(Self::BitAnd),
            Rule::op_negate => Ok(Self::Neg),
            Rule::op_eq => Ok(Self::Eq),
            Rule::op_xor => Ok(Self::Xor),
            Rule::op_not => Ok(Self::Not),
            _ => Err(Error::parser("unimplemented cmp operator", value.as_span())),
        }
    }
}

impl Op {
    fn from_pair(value: Pair<'_, Rule>) -> Result<Self, Error> {
        match value.as_rule() {
            Rule::op_mul => Ok(Self::Mul),
            Rule::op_add => Ok(Self::Add),
            Rule::op_sub => Ok(Self::Sub),
            Rule::op_div => Ok(Self::Div),
            Rule::op_mod => Ok(Self::Mod),
            Rule::op_and => Ok(Self::And),
            Rule::op_xor => Ok(Self::Xor),
            _ => Err(Error::parser("unimplemented operator", value.as_span())),
        }
    }
}

#[derive(Debug, Clone)]
struct Transform {
    op: Op,
    num: Scalar,
}

impl Transform {
    fn apply(&self, s: Scalar) -> Scalar {
        match self.op {
            Op::Add => s.add(self.num),
            Op::Sub => s.sub(self.num),
            Op::Mul => s.mul(self.num),
            Op::Div => s.div(self.num),
            Op::Mod => s.rem(self.num),
            Op::And => s.bitand(self.num),
            Op::Xor => s.bitxor(self.num),
        }
    }
}

// Any Magic Data type
// FIXME: Any must carry StringTest so that we know the string mods / length
#[derive(Debug, Clone)]
enum Any {
    String,
    PString,
    Scalar(ScalarDataType),
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
        LineLimit
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
}

impl From<RegexTest> for Test {
    fn from(value: RegexTest) -> Self {
        Self::Regex(value)
    }
}

impl RegexTest {
    fn from_pair_with_re(pair: Pair<'_, Rule>, re: &str) -> Result<Self, Error> {
        let mut length = None;
        let mut mods = FlagSet::empty();
        let mut str_mods = FlagSet::empty();
        for p in pair.into_inner() {
            match p.as_rule() {
                Rule::pos_number => length = Some(parse_pos_number(p) as usize),
                Rule::regex_mod => {
                    for m in p.as_str().chars() {
                        match m {
                            'c' => {
                                mods |= ReMod::CaseInsensitive;
                            }
                            's' => mods |= ReMod::StartOffsetUpdate,
                            'l' => mods |= ReMod::LineLimit,
                            _ => {}
                        }
                    }
                }
                Rule::string_mod => str_mods |= StringMod::from_pair(p)?,
                // this should never happen
                _ => unimplemented!(),
            }
        }

        Ok(Self {
            //FIXME: remove unwrap
            re: bytes::Regex::new(re).unwrap(),
            length,
            n_pos: None,
            mods,
            str_mods,
            search: false,
        })
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

impl StringMod {
    fn from_pair(pair: Pair<'_, Rule>) -> Result<StringMod, Error> {
        if !matches!(pair.as_rule(), Rule::string_mod) {
            return Err(Error::parser("unknown string mod", pair.as_span()));
        }

        // this shouldn't panic as our parser guarantee there
        // is one element in the pair
        let c = pair.as_str().chars().next().unwrap();

        match c {
            'b' => Ok(StringMod::ForceBin),
            'C' => Ok(StringMod::UpperInsensitive),
            'c' => Ok(StringMod::LowerInsensitive),
            'f' => Ok(StringMod::FullWordMatch),
            'T' => Ok(StringMod::Trim),
            't' => Ok(StringMod::ForceText),
            'W' => Ok(StringMod::CompactWhitespace),
            'w' => Ok(StringMod::OptBlank),
            _ => Err(Error::parser("unknown string mod", pair.as_span())),
        }
    }
}

// FIXME: implement string operators
#[derive(Debug, Clone)]
struct StringTest {
    str: String,
    cmp_op: CmpOp,
    length: Option<usize>,
    mods: FlagSet<StringMod>,
}

impl From<StringTest> for Test {
    fn from(value: StringTest) -> Self {
        Self::String(value)
    }
}

impl StringTest {
    fn from_pair_with_str(pair: Pair<'_, Rule>, str: &str, cmp_op: CmpOp) -> Result<Self, Error> {
        let mut length = None;
        let mut mods = FlagSet::empty();
        for p in pair.into_inner() {
            match p.as_rule() {
                Rule::pos_number => length = Some(parse_pos_number(p) as usize),
                Rule::string_mod => mods |= StringMod::from_pair(p)?,
                // this should never happen
                _ => unimplemented!(),
            }
        }
        Ok(Self {
            str: str.to_string(),
            cmp_op,
            length,
            mods,
        })
    }
}

#[derive(Debug, Clone)]
struct SearchTest {
    str: String,
    n_pos: Option<usize>,
    // FIXME: handle all string mods
    str_mods: FlagSet<StringMod>,
    // FIXME: handle all re mods
    re_mods: FlagSet<ReMod>,
}

impl From<SearchTest> for Test {
    fn from(value: SearchTest) -> Self {
        Self::Search(value)
    }
}

impl SearchTest {
    fn from_pair_with_str(pair: Pair<'_, Rule>, str: &str) -> Self {
        let mut length = None;
        let mut str_mods: FlagSet<StringMod> = FlagSet::empty();
        let mut re_mods: FlagSet<ReMod> = FlagSet::empty();
        for p in pair.into_inner() {
            match p.as_rule() {
                Rule::pos_number => length = Some(parse_pos_number(p) as usize),
                Rule::string_mod => {
                    for m in p.as_str().chars() {
                        match m {
                            'b' => str_mods |= StringMod::ForceBin,
                            'C' => str_mods |= StringMod::UpperInsensitive,
                            'c' => str_mods |= StringMod::LowerInsensitive,
                            'f' => str_mods |= StringMod::FullWordMatch,
                            'T' => str_mods |= StringMod::Trim,
                            't' => str_mods |= StringMod::ForceText,
                            'W' => str_mods |= StringMod::CompactWhitespace,
                            'w' => str_mods |= StringMod::OptBlank,
                            _ => {}
                        }
                    }
                }
                Rule::regex_mod => {
                    for m in p.as_str().chars() {
                        match m {
                            'c' => {
                                re_mods |= ReMod::CaseInsensitive;
                            }
                            's' => re_mods |= ReMod::StartOffsetUpdate,
                            'l' => re_mods |= ReMod::LineLimit,
                            _ => {}
                        }
                    }
                }
                // this should never happen
                _ => unimplemented!(),
            }
        }
        Self {
            str: str.to_string(),
            n_pos: length,
            str_mods,
            re_mods,
        }
    }
}

#[derive(Debug, Clone)]
struct ScalarTest {
    ty: ScalarDataType,
    transform: Option<Transform>,
    cmp_op: CmpOp,
    value: Scalar,
}

// the value read from the haystack we want to
// match against
// 'buf is the lifetime of the buffer we are scanning
#[derive(Debug, PartialEq, Eq)]
enum TestValue<'buf> {
    Scalar(u64, Scalar),
    String(u64, &'buf str),
    PString(u64, &'buf str),
    Bytes(u64, &'buf [u8]),
}

impl DynDisplay for TestValue<'_> {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        match self {
            Self::Scalar(_, s) => DynDisplay::dyn_fmt(s, f),
            Self::String(_, s) => DynDisplay::dyn_fmt(s, f),
            Self::PString(_, s) => DynDisplay::dyn_fmt(s, f),
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
            Self::Scalar(_, s) => write!(f, "{}", s),
            Self::String(_, s) => write!(f, "{}", s),
            Self::PString(_, s) => write!(f, "{}", s),
            Self::Bytes(_, b) => write!(f, "{:?}", b),
        }
    }
}

// Carry the offset of the start of the data in the stream
// and the data itself
enum MatchRes<'buf> {
    String(u64, &'buf str),
    Scalar(u64, Scalar),
}

impl DynDisplay for &MatchRes<'_> {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        (*self).dyn_fmt(f)
    }
}

impl DynDisplay for MatchRes<'_> {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        match self {
            Self::Scalar(_, s) => s.dyn_fmt(f),
            Self::String(_, s) => s.dyn_fmt(f),
        }
    }
}

#[derive(Debug, Clone)]
enum Test {
    /// This corresponds to a DATATYPE x test
    Any(Any),
    Name(String),
    Use(bool, String),
    Scalar(ScalarTest),
    String(StringTest),
    Search(SearchTest),
    PString(String),
    Regex(RegexTest),
    Clear,
    Default,
    Indirect,
    // FIXME: placeholders for strength computation
    LeString16,
    Der,
}

impl Test {
    fn from_pair(pair: Pair<'_, Rule>) -> Result<Self, Error> {
        let t = match pair.as_rule() {
            Rule::scalar_test => {
                let pairs = pair.into_inner();

                let mut ty = None;
                let mut ty_span = None;
                let mut transform = None;
                let mut condition = CmpOp::Eq;
                let mut scalar = None;
                for pair in pairs {
                    match pair.as_rule() {
                        Rule::scalar_type => {
                            ty_span = Some(pair.as_span());
                            ty = Some(ScalarDataType::from_pair(pair)?);
                        }

                        Rule::scalar_transform => {
                            let mut transform_pairs = pair.into_inner();
                            let op_pair = transform_pairs.next().expect("expect operator pair");
                            let op = Op::from_pair(op_pair)?;

                            let number = transform_pairs.next().expect("expect number pair");
                            let span = number.as_span();
                            transform = Some(Transform {
                                op,
                                // ty is guaranteed to be some by
                                // parser implementation
                                num: ty.unwrap().scalar_from_number(parse_pos_number(number)),
                            });
                        }
                        Rule::scalar_condition => {
                            condition = CmpOp::from_pair(
                                pair.into_inner().next().expect("expecting cmp operator"),
                            )?;
                        }
                        Rule::scalar_value => {
                            let number_pair =
                                pair.into_inner().next().expect("number pair expected");

                            scalar = Some(
                                ty.unwrap()
                                    .scalar_from_number(parse_number_pair(number_pair)),
                            );
                        }
                        Rule::any_value => return Ok(Self::Any(Any::Scalar(ty.unwrap()))),
                        _ => {}
                    }
                }

                Self::Scalar(ScalarTest {
                    // no panic guarantee by parser
                    ty: ty.unwrap(),
                    transform,
                    cmp_op: condition,
                    // no panic guarantee by parser
                    value: scalar.unwrap(),
                })
            }
            Rule::search_test => {
                let mut search_test = pair.into_inner();

                let test_type = search_test.next().expect("expecting a string type");

                let test_value = search_test.next().expect("expecting a string value");
                assert_eq!(test_value.as_rule(), Rule::string_value);

                match test_type.as_rule() {
                    Rule::search => SearchTest::from_pair_with_str(
                        test_type,
                        &unescape_string(test_value.as_str()),
                    )
                    .into(),

                    Rule::regex => RegexTest::from_pair_with_re(
                        test_type,
                        &unescape_string(test_value.as_str()),
                    )?
                    .into(),
                    _ => unimplemented!(),
                }
            }
            Rule::string_test => {
                let mut string_test = pair.into_inner();

                let test_type = string_test.next().expect("expecting a string type");

                let pair = string_test
                    .next()
                    .expect("expecting operator or string value");

                let cmp_op = match pair.as_rule() {
                    Rule::op_eq => Some(CmpOp::Eq),
                    Rule::op_lt => Some(CmpOp::Lt),
                    Rule::op_gt => Some(CmpOp::Gt),
                    _ => None,
                };

                // if there was an operator we need to iterate
                let test_value = if cmp_op.is_some() {
                    string_test.next().expect("expecting a string value")
                } else {
                    pair
                };

                match test_value.as_rule() {
                    Rule::string_value => match test_type.as_rule() {
                        Rule::string => StringTest::from_pair_with_str(
                            test_type,
                            &unescape_string(test_value.as_str()),
                            cmp_op.unwrap_or(CmpOp::Eq),
                        )?
                        .into(),

                        Rule::pstring => Self::PString(unescape_string(test_value.as_str())),

                        _ => unimplemented!(),
                    },
                    Rule::any_value => Self::Any(Any::from_rule(test_type.as_rule())),
                    _ => unimplemented!(),
                }
            }
            Rule::clear_test => Self::Clear,
            Rule::default_test => Self::Default,
            Rule::indirect_test => Self::Indirect,
            _ => unimplemented!(),
        };

        Ok(t.transform())
    }

    // we convert a string with mods into a regexp pattern
    fn string_to_re_pattern(src: &str, mods: FlagSet<StringMod>, match_start: bool) -> String {
        // we escape all regex related characters
        let mut out = regex::escape(src);

        if match_start {
            // we insert start of expression test
            out.insert(0, '^');
        }

        // any blank character is optional
        let mut tmp = String::new();
        let mut chars = out.chars().peekable();

        while let Some(c) = chars.next() {
            if c == ' ' && mods.contains(StringMod::OptBlank) {
                tmp.push(c);
                tmp.push('*');
            } else if c == ' ' && mods.contains(StringMod::CompactWhitespace) {
                tmp.push(c);
                let mut space_rep = 1;
                for c in chars.by_ref() {
                    if c == ' ' {
                        space_rep += 1;
                        continue;
                    }
                    tmp.push_str(&format!("{{{space_rep},}}"));
                    tmp.push(c);
                    break;
                }
            } else if c.is_uppercase() && mods.contains(StringMod::UpperInsensitive)
                || (c.is_lowercase() && mods.contains(StringMod::LowerInsensitive))
            {
                tmp.push_str("(?i:");
                tmp.push(c);

                while let Some(c) = chars.by_ref().peek() {
                    if c.is_uppercase() && mods.contains(StringMod::UpperInsensitive)
                        || (c.is_lowercase() && mods.contains(StringMod::LowerInsensitive))
                    {
                        tmp.push(*c);
                        chars.by_ref().next();
                    } else {
                        break;
                    }
                }

                tmp.push_str(")");
            } else {
                tmp.push(c);
            }
        }
        out = tmp;

        // we insert word boundary check in regex
        if mods.contains(StringMod::FullWordMatch) {
            out.push_str(r"\b");
        }

        out
    }

    fn transform(self) -> Self {
        match self {
            Self::Search(s) => {
                // if we search only at one position it means we must match
                // start of string
                let mut pattern =
                    Self::string_to_re_pattern(&s.str, s.str_mods, matches!(s.n_pos, Some(1)));

                // we handle cases where we wanna match more positions
                if let Some(n_pos) = s.n_pos {
                    if n_pos > 1 {
                        pattern.insert_str(0, ".*?");
                    }
                }

                RegexTest {
                    // FIXME: remove unwrap
                    re: Regex::new(&pattern).unwrap(),
                    length: None,
                    n_pos: s.n_pos,
                    mods: s.re_mods,
                    str_mods: s.str_mods,
                    search: true,
                }
                .into()
            }
            Self::String(st) => {
                // test if  we need to turn the string test into a regex
                if st.mods.contains(StringMod::CompactWhitespace)
                    || st.mods.contains(StringMod::FullWordMatch)
                    || st.mods.contains(StringMod::LowerInsensitive)
                    || st.mods.contains(StringMod::UpperInsensitive)
                    || st.mods.contains(StringMod::OptBlank)
                {
                    RegexTest {
                        // FIXME: remove unwrap
                        re: Regex::new(&Self::string_to_re_pattern(&st.str, st.mods, true))
                            .unwrap(),
                        length: st.length,
                        n_pos: None,
                        mods: FlagSet::empty(),
                        str_mods: st.mods,
                        search: false,
                    }
                    .into()
                } else {
                    st.into()
                }
            }
            _ => self,
        }
    }

    // read the value to test from the haystack
    fn read_test_value<'haystack, R: Read + Seek>(
        &self,
        haystack: &'haystack mut LazyCache<R>,
        switch_endianness: bool,
    ) -> Result<TestValue<'haystack>, Error> {
        macro_rules! read {
            ($ty: ty) => {{
                let mut a = [0u8; std::mem::size_of::<$ty>()];
                haystack.read_exact_into(&mut a)?;
                a
            }};
        }

        macro_rules! read_le {
            ($ty: ty) => {
                <$ty>::from_le_bytes(read!($ty))
            };
        }

        let test_value_offset = haystack.lazy_stream_position();

        match self {
            Self::Scalar(t) => {
                t.ty.read(haystack, switch_endianness)
                    .map(|s| TestValue::Scalar(test_value_offset, s))
            }
            Self::String(t) => {
                let buf = if let Some(length) = t.length {
                    // if there is a length specified
                    let read = haystack.read_exact(length as u64)?;
                    read
                } else {
                    // no length specified we read until end of string
                    let read = match t.cmp_op {
                        CmpOp::Eq => haystack.read_exact(t.str.len() as u64)?,
                        CmpOp::Lt | CmpOp::Gt => {
                            let read = haystack.read_until_limit(b'\0', 8092)?;
                            if read.len() > 0 {
                                &read[..read.len() - 1]
                            } else {
                                read
                            }
                        }
                        _ => unimplemented!(),
                    };
                    read
                };
                str::from_utf8(buf)
                    .map(|s| TestValue::String(test_value_offset, s))
                    .map_err(Error::from)
            }

            Self::PString(s) => {
                // FIXME: maybe we could optimize here by reading testing on size
                // this is the size of the pstring
                // FIXME: adjust the size function of pstring mods
                let _ = read_le!(u8);
                let read = haystack.read_exact(s.len() as u64)?;

                str::from_utf8(read)
                    .map(|s| TestValue::PString(test_value_offset, s))
                    .map_err(Error::from)
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
                    let read = haystack.read_until_limit(b'\0', 8192)?;

                    str::from_utf8(read)
                        .map(|s| TestValue::String(test_value_offset, s))
                        .map_err(Error::from)
                        .inspect_err(|e| println!("{}", e))
                }
                Any::PString => {
                    let slen = read_le!(u8) as usize;
                    let read = haystack.read_exact(slen as u64)?;

                    str::from_utf8(read)
                        .map(|s| TestValue::PString(test_value_offset, s))
                        .map_err(Error::from)
                        .inspect_err(|e| println!("{}", e))
                }
                Any::Scalar(d) => d
                    .read(haystack, switch_endianness)
                    .map(|s| TestValue::Scalar(test_value_offset, s)),
            },

            _ => unimplemented!(),
        }
    }

    #[inline(always)]
    fn match_value<'s>(&self, tv: &TestValue<'s>) -> Option<MatchRes<'s>> {
        // always true when we want to read value
        if let Self::Any(v) = self {
            match tv {
                TestValue::PString(o, ps) => {
                    if matches!(v, Any::PString) {
                        return Some(MatchRes::String(*o, ps));
                    }
                }
                TestValue::String(o, s) => {
                    if matches!(v, Any::String) {
                        return Some(MatchRes::String(*o, s));
                    }
                }
                TestValue::Scalar(o, s) => {
                    if matches!(v, Any::Scalar(_)) {
                        return Some(MatchRes::Scalar(*o, *s));
                    }
                }
                _ => panic!("not good"),
            }

            // FIXME: remove this
            panic!("not good")
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
                        CmpOp::Neg => read_value != t.value,
                        CmpOp::BitAnd => read_value & t.value == read_value,
                        CmpOp::Xor => (read_value & t.value).is_zero(),
                    };

                    if ok {
                        return Some(MatchRes::Scalar(*o, read_value));
                    }
                }
            }
            TestValue::String(o, tv) => {
                if let Self::String(st) = self {
                    match st.cmp_op {
                        CmpOp::Eq => {
                            if *tv == st.str {
                                return Some(MatchRes::String(*o, tv));
                            }
                        }
                        CmpOp::Gt => {
                            if tv.len() > st.str.len() {
                                return Some(MatchRes::String(*o, tv));
                            }
                        }
                        CmpOp::Lt => {
                            if tv.len() < st.str.len() {
                                return Some(MatchRes::String(*o, tv));
                            }
                        }
                        // unsupported for strings
                        _ => {
                            debug_panic!("unsupported cmp operator for string")
                        }
                    }
                }
            }
            TestValue::PString(o, tv) => {
                if let Self::PString(m) = self {
                    if tv == m {
                        return Some(MatchRes::String(*o, tv));
                    }
                }
            }
            TestValue::Bytes(o, buf) => {
                if let Self::Regex(r) = self {
                    if let Some(re_match) = r.re.find(&buf) {
                        if let Some(n_pos) = r.n_pos {
                            // we check for positinal match inherited from search conversion
                            if re_match.start() >= n_pos {
                                return None;
                            }
                        }

                        return Some(MatchRes::String(
                            // the offset of the string is computed from the start of the buffer
                            o + re_match.start() as u64,
                            // FIXME: we shouldn't unwrap here it may panic because
                            // we cannot guarantee this is valid UTF8
                            std::str::from_utf8(re_match.as_bytes()).unwrap().into(),
                        ));
                    }
                }
            }
        }

        None
    }

    //FIXME: complete with all possible operators
    fn cmp_op(&self) -> Option<CmpOp> {
        match self {
            Self::Scalar(s) => Some(s.cmp_op),
            _ => None,
        }
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

impl Shift {
    fn from_pair(pair: Pair<'_, Rule>) -> Self {
        assert_eq!(pair.as_rule(), Rule::shift);
        let shift_variant = pair.into_inner().next().expect("shift cannot be empty");
        match shift_variant.as_rule() {
            Rule::ind_shift => Self::Indirect(parse_number_pair(
                shift_variant
                    .into_inner()
                    .next()
                    .expect("indirect shift must contain number"),
            )),
            Rule::dir_shift => Self::Direct(parse_number_pair(
                shift_variant
                    .into_inner()
                    .next()
                    .expect("direct shift must contain number"),
            ) as u64),
            _ => {
                panic!("unknown shift pair")
            }
        }
    }
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
    fn from_pair(pair: Pair<'_, Rule>) -> Result<Self, Error> {
        let mut off_addr = None;
        let mut signed = false;
        // default type according to magic documentation
        let mut offset_type = OffsetType::LongLe;
        let mut op = None;
        let mut shift = None;

        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::abs_offset | Rule::rel_offset => off_addr = Some(DirOffset::from_pair(pair)),
                Rule::ind_offset_sign => match pair.as_str() {
                    "," => signed = true,
                    "." => signed = false,
                    _ => {}
                },
                Rule::ind_offset_type => match pair.as_str() {
                    "b" | "c" | "B" | "C" => offset_type = OffsetType::Byte,
                    "e" | "f" | "g" => offset_type = OffsetType::DoubleLe,
                    "E" | "F" | "G" => offset_type = OffsetType::DoubleBe,
                    "h" | "s" => offset_type = OffsetType::ShortLe,
                    "H" | "S" => offset_type = OffsetType::ShortBe,
                    "i" => offset_type = OffsetType::Id3Le,
                    "I" => offset_type = OffsetType::Id3Be,
                    "l" => offset_type = OffsetType::LongLe,
                    "L" => offset_type = OffsetType::LongBe,
                    "m" => offset_type = OffsetType::Middle,
                    "o" => offset_type = OffsetType::Octal,
                    "q" => offset_type = OffsetType::QuadLe,
                    "Q" => offset_type = OffsetType::QuadBe,
                    _ => {}
                },
                Rule::op_add
                | Rule::op_sub
                | Rule::op_mul
                | Rule::op_div
                | Rule::op_mod
                | Rule::op_and
                | Rule::op_or
                | Rule::op_xor => op = Some(Op::from_pair(pair)?),

                Rule::shift => {
                    shift = Some(Shift::from_pair(pair));
                }
                _ => {}
            }
        }

        Ok(Self {
            off_addr: off_addr.unwrap(),
            signed,
            ty: offset_type,
            op,
            shift,
        })
    }

    // if we overflow we must not return an offset
    fn get_offset<R: Read + Seek>(
        &self,
        haystack: &mut LazyCache<R>,
        last_upper_match_offset: Option<u64>,
    ) -> Result<Option<u64>, io::Error> {
        let main_offset_offset = match self.off_addr {
            DirOffset::Start(s) => haystack.seek(SeekFrom::Start(s as u64))?,
            DirOffset::LastUpper(c) => haystack.seek(SeekFrom::Start(
                (last_upper_match_offset.unwrap_or_default() as i64 + c) as u64,
            ))?,
            DirOffset::End(e) => haystack.seek(SeekFrom::End(e as i64))?,
        };

        macro_rules! read_buf {
            ($ty: ty) => {{
                let mut a = [0u8; std::mem::size_of::<$ty>()];
                haystack.read_exact_into(&mut a)?;
                a
            }};
        }

        macro_rules! read_le {
            ($ty: ty ) => {{ <$ty>::from_le_bytes(read_buf!($ty)) }};
        }

        macro_rules! read_be {
            ($ty: ty ) => {{ <$ty>::from_be_bytes(read_buf!($ty)) }};
        }

        macro_rules! read_value {
            () => {
                match self.ty {
                    OffsetType::Byte => {
                        if self.signed {
                            read_le!(u8) as u64
                        } else {
                            read_le!(i8) as u64
                        }
                    }
                    OffsetType::DoubleLe => read_le!(f64) as u64,
                    OffsetType::DoubleBe => read_be!(f64) as u64,
                    OffsetType::ShortLe => {
                        if self.signed {
                            read_le!(i16) as u64
                        } else {
                            read_le!(u16) as u64
                        }
                    }
                    OffsetType::ShortBe => {
                        if self.signed {
                            read_be!(i16) as u64
                        } else {
                            read_be!(u16) as u64
                        }
                    }
                    OffsetType::Id3Le => unimplemented!(),
                    OffsetType::Id3Be => unimplemented!(),
                    OffsetType::LongLe => {
                        if self.signed {
                            read_le!(i32) as u64
                        } else {
                            read_le!(u32) as u64
                        }
                    }
                    OffsetType::LongBe => {
                        if self.signed {
                            read_be!(i32) as u64
                        } else {
                            read_be!(u32) as u64
                        }
                    }
                    OffsetType::Middle => unimplemented!(),
                    OffsetType::Octal => unimplemented!(),
                    OffsetType::QuadLe => {
                        if self.signed {
                            read_le!(i64) as u64
                        } else {
                            read_le!(u64)
                        }
                    }
                    OffsetType::QuadBe => {
                        if self.signed {
                            read_be!(i64) as u64
                        } else {
                            read_be!(u64)
                        }
                    }
                }
            };
        }

        // in theory every offset read should end up in something seekable from start, so we can use u64 to store the result
        let o = read_value!();

        trace!(
            "computing offset base={o} op={:?} shift={:?}",
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

impl DirOffset {
    fn from_pair(pair: Pair<'_, Rule>) -> Self {
        match pair.as_rule() {
            Rule::abs_offset => {
                let number_pair = pair.into_inner().next().expect("number pair expected");

                let offset = parse_number_pair(number_pair);

                if offset.is_negative() {
                    DirOffset::End(offset)
                } else {
                    DirOffset::Start(offset as u64)
                }
            }
            Rule::rel_offset => {
                let number_pair = pair.into_inner().next().expect("number pair expected");

                let offset = parse_number_pair(number_pair);

                DirOffset::LastUpper(offset)
            }
            _ => panic!("unexpected offset pair"),
        }
    }
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

impl Offset {
    fn from_pair(pair: Pair<'_, Rule>) -> Self {
        let mut pairs = pair.into_inner();
        let pair = pairs.next().expect("offset must have token");

        match pair.as_rule() {
            Rule::abs_offset => {
                let number_pair = pair.into_inner().next().expect("number pair expected");

                let offset = parse_number_pair(number_pair);

                if offset.is_negative() {
                    Self::Direct(DirOffset::End(offset))
                } else {
                    Self::Direct(DirOffset::Start(offset as u64))
                }
            }
            Rule::rel_offset => {
                let number_pair = pair.into_inner().next().expect("number pair expected");

                let offset = parse_number_pair(number_pair);

                Self::Direct(DirOffset::LastUpper(offset))
            }

            // FIXME: remove unwrap, this function must return Result
            Rule::indirect_offset => Self::Indirect(IndOffset::from_pair(pair).unwrap()),

            _ => panic!("unexpected token"),
        }
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

#[inline]
fn parse_pos_number(pair: Pair<'_, Rule>) -> i64 {
    let number_token = pair.into_inner().next().expect("expect number kind pair");
    match number_token.as_rule() {
        Rule::b10_number => number_token.as_str().parse::<i64>().unwrap(),
        Rule::b16_number => {
            u64::from_str_radix(number_token.as_str().strip_prefix("0x").unwrap(), 16).unwrap()
                as i64
        }
        _ => panic!("unexpected number"),
    }
}

#[inline]
fn parse_number_pair(pair: Pair<'_, Rule>) -> i64 {
    assert_eq!(pair.as_rule(), Rule::number);
    let inner = pair
        .into_inner()
        .next()
        .expect("positive or negative number expected");

    match inner.as_rule() {
        Rule::pos_number => parse_pos_number(inner),
        Rule::neg_number => -parse_pos_number(
            inner
                .into_inner()
                .next()
                .expect("expecting positive number"),
        ),
        _ => panic!("unexpected number inner pair"),
    }
}

#[derive(Debug, Clone)]
pub struct Match {
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
    fn from_pair(pair: Pair<'_, Rule>) -> Result<Self, Error> {
        let (line, _) = pair.line_col();
        let mut pairs = pair.into_inner();

        // first token might be a depth or an offset
        let token = pairs.next().expect("expecting a depth or offset");

        let (depth, offset) = if matches!(token.as_rule(), Rule::depth) {
            (
                token.as_str().len() as u8,
                pairs.next().expect("offset token"),
            )
        } else {
            (0, token)
        };

        assert_eq!(offset.as_rule(), Rule::offset, "expected offset rule");
        let offset = Offset::from_pair(offset);

        let test_pair = pairs.next().expect("test pair expected");
        assert_eq!(test_pair.as_rule(), Rule::test, "wrong token");
        let test = Test::from_pair(test_pair.into_inner().next().unwrap())?;

        // parsing the message
        let message = match pairs.next() {
            Some(msg_pair) => {
                if !msg_pair.as_str().is_empty() {
                    Some(Message::from_pair(msg_pair))
                } else {
                    None
                }
            }
            None => None,
        };

        Ok(Self {
            line,
            depth,
            offset,
            test,
            message,
        })
    }

    // FIXME: handle push_message only once based on the success
    // or not of the test.
    #[inline]
    fn matches<'a, R: Read + Seek>(
        &'a self,
        magic: &mut Magic<'a>,
        state: &mut MatchState,
        last_level_offset: Option<u64>,
        opt_start_offset: Option<Offset>,
        haystack: &mut LazyCache<R>,
        switch_endianness: bool,
        db: &'a MagicDb,
        depth: usize,
    ) -> Result<bool, Error> {
        if depth >= MAX_RECURSION {
            return Err(Error::MaximumRecursion(MAX_RECURSION));
        }

        // handle clear and default tests
        if matches!(self.test, Test::Clear) {
            trace!("line={} clear", self.line);
            state.clear_continuation_level(&self.continuation_level());
            return Ok(true);
        }

        if let Test::Name(name) = &self.test {
            trace!("line={} running rule {name}", self.line);
            if let Some(msg) = self.message.as_ref() {
                magic.push_message(msg.format_with(None));
            }
            return Ok(true);
        }

        if let Test::Use(switch_endianness, rule_name) = &self.test {
            trace!(
                "line={} use {rule_name} switch_endianness={switch_endianness}",
                self.line
            );
            let dr: &DependencyRule = db
                .dependencies
                .get(rule_name)
                .ok_or(Error::MissingRule(rule_name.clone()))?;

            let offset = match self.offset {
                Offset::Direct(dir_offset) => match dir_offset {
                    DirOffset::Start(s) => Offset::from(DirOffset::Start(s)),
                    DirOffset::LastUpper(shift) => Offset::from(DirOffset::Start(
                        (last_level_offset.unwrap_or_default() as i64 + shift) as u64,
                    )),
                    DirOffset::End(e) => Offset::from(DirOffset::End(e)),
                },
                Offset::Indirect(ind_offset) => {
                    let Some(o) = ind_offset.get_offset(haystack, last_level_offset)? else {
                        return Ok(false);
                    };
                    Offset::from(DirOffset::Start(o))
                }
            };

            if let Some(msg) = self.message.as_ref() {
                magic.push_message(msg.format_with(None));
            }

            dr.rule.magic(
                magic,
                Some(offset),
                haystack,
                db,
                *switch_endianness,
                depth.wrapping_add(1),
            )?;

            return Ok(false);
        }

        if matches!(self.test, Test::Default) {
            // default matches if nothing else at the continuation level matched
            let ok = !state.get_continuation_level(&self.continuation_level());

            trace!("line={} default match={ok}", self.line);
            if ok {
                if let Some(msg) = self.message.as_ref() {
                    magic.push_message(msg.format_with(None));
                }
                state.set_continuation_level(self.continuation_level());
            }

            return Ok(ok);
        }

        if matches!(self.test, Test::Indirect) {
            for r in db.rules.iter() {
                r.magic(magic, None, haystack, db, false, depth.wrapping_add(1))?;
            }
        }

        let i = opt_start_offset
            .map(|so| match so {
                Offset::Direct(DirOffset::Start(s)) => s as i64,
                // FIXME: this is relative to previous match so we need to carry up this information
                Offset::Direct(DirOffset::LastUpper(_)) => {
                    // offset from last upper is resolved upstream in use test
                    panic!("this should never happen")
                }
                Offset::Direct(DirOffset::End(e)) => e,
                Offset::Indirect(_) => {
                    // indirect offset is resolved upstream when use test
                    panic!("this should never happen")
                }
            })
            .unwrap_or_default();

        match self.offset {
            Offset::Direct(DirOffset::Start(s)) => haystack.seek(SeekFrom::Start(s + i as u64))?,
            // FIXME: this is relative to previous match so we need to carry up this information
            Offset::Direct(DirOffset::LastUpper(c)) => haystack.seek(SeekFrom::Start(
                (i + c + last_level_offset.unwrap_or_default() as i64) as u64,
            ))?,
            Offset::Direct(DirOffset::End(e)) => haystack.seek(SeekFrom::End(e + i))?,
            Offset::Indirect(io) => {
                let Some(o) = io.get_offset(haystack, last_level_offset)? else {
                    return Ok(false);
                };

                haystack.seek(SeekFrom::Start(o + i as u64))?
            }
        };

        let mut trace_msg = None;

        if enabled!(Level::DEBUG) {
            trace_msg = Some(vec![format!(
                "line={} stream_offset={} ",
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
            .inspect_err(|e| trace!("line={} error while reading test value: {e}", self.line))
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
                    let opt_adjusted_offset = if let MatchRes::String(o, s) = mr {
                        Some(o + s.len() as u64)
                    } else {
                        None
                    };

                    if let Some(o) = opt_adjusted_offset {
                        haystack.seek(SeekFrom::Start(o))?;
                    }
                }
                state.set_continuation_level(self.continuation_level());
                return Ok(true);
            }
        }

        Ok(false)
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

        // NB: octal is missing but it is not used in practice ...
        match &self.test {
            Test::Default => return 0,
            Test::Scalar(s) => match s.ty {
                ScalarDataType::lemsdostime => {
                    out += ScalarDataType::lemsdostime.type_size() * MULT;
                }

                _ => {}
            },
            Test::Search(s) => out += s.str.len() * max(MULT / s.str.len(), 1),

            Test::Regex(r) => {
                let v = nonmagic(r.re.as_str());
                out += v * max(MULT / v, 1);
            }

            Test::LeString16 | Test::Der => {
                unimplemented!()
            }

            // matching any output gets penalty
            Test::Any(_) => out = 0,

            _ => {}
        }

        if let Some(op) = self.test.cmp_op() {
            match op {
                // matching almost any gets penalty
                CmpOp::Neg => out = 0,
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

impl Use {
    fn from_pair(pair: Pair<'_, Rule>) -> Self {
        assert_eq!(pair.as_rule(), Rule::r#use);
        let (line, _) = pair.line_col();
        let mut pairs = pair.into_inner();

        // first token might be a depth or an offset
        let token = pairs.next().expect("expecting depth");

        let (depth, offset) = if matches!(token.as_rule(), Rule::depth) {
            (
                token.as_str().len() as u8,
                pairs.next().expect("offset token"),
            )
        } else {
            (0, token)
        };

        assert_eq!(offset.as_rule(), Rule::offset, "expected offset rule");
        let offset = Offset::from_pair(offset);

        // here token can be both endianness_switch or rule_name
        let token = pairs
            .next()
            .expect("expecting a rule name or an endianness switch");

        let endianness_switch = matches!(token.as_rule(), Rule::endianness_switch);

        let rule_name_token = if endianness_switch {
            pairs.next().expect("expecting a rule name")
        } else {
            token
        };

        assert_eq!(
            rule_name_token.as_rule(),
            Rule::rule_name,
            "wrong parsing rule"
        );

        let message = pairs.next().map(|m| Message::from_pair(m));

        Self {
            line,
            depth,
            start_offset: offset,
            rule_name: rule_name_token.as_str().to_string(),
            switch_endianness: endianness_switch,
            message,
        }
    }
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
            Op::Xor => unimplemented!(),
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

impl Flag {
    fn from_pair(pair: Pair<'_, Rule>) -> Result<Self, Error> {
        assert_eq!(pair.as_rule(), Rule::flag);
        let mut pairs = pair.into_inner();
        let flag = pairs.next().expect("expecting a valid flag");

        match flag.as_rule() {
            Rule::mime_flag => Ok(Self::Mime(
                flag.into_inner()
                    .next()
                    .expect("expecting mime type")
                    .as_str()
                    .into(),
            )),
            Rule::strength_flag => {
                let mut pairs = flag.into_inner();
                // parsing operator
                let op_pair = pairs.next().expect("strength entry must have operator");
                let op = Op::from_pair(op_pair)?;

                // parsing value
                let number_pair = pairs.next().expect("strength entry must have a value");

                let span = number_pair.as_span();
                let by: u8 = parse_pos_number(number_pair)
                    .try_into()
                    .map_err(|_| Error::parser("value must be u8", span))?;

                Ok(Self::Strength(StrengthMod { op, by }))
            }
            Rule::ext_flag => {
                let exts = flag.into_inner().next().expect("expecting extension list");
                assert_eq!(exts.as_rule(), Rule::exts);
                Ok(Self::Ext(
                    exts.as_str().split('/').map(|s| s.into()).collect(),
                ))
            }
            Rule::apple_flag => {
                let creatype = flag.into_inner().next().expect("expecting a creatype");
                assert_eq!(creatype.as_rule(), Rule::printable_no_ws);
                Ok(Self::Apple(creatype.as_str().to_string()))
            }
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Name {
    line: usize,
    offset: Offset,
    name: String,
    message: Option<Message>,
}

#[derive(Debug, Clone)]
pub enum Entry {
    Match(Match),
    Flag(Flag),
}

#[derive(Debug, Clone)]
pub struct EntryNode {
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
                    }
                    if m.depth == root.depth + 1 {
                        children.push(EntryNode::from_peekable(entries))
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
        magic: &mut Magic<'r>,
        state: &mut MatchState,
        last_level_offset: Option<u64>,
        opt_start_offset: Option<Offset>,
        haystack: &mut LazyCache<R>,
        db: &'r MagicDb,
        switch_endianness: bool,
        depth: usize,
    ) -> Result<(), Error> {
        let ok = self.entry.matches(
            magic,
            state,
            last_level_offset,
            opt_start_offset,
            haystack,
            switch_endianness,
            db,
            depth,
        )?;

        if ok {
            if let Some(mimetype) = self.mimetype.as_ref() {
                magic.insert_mimetype(Cow::Borrowed(mimetype));
            }

            let strength = match self.strength_mod.as_ref() {
                Some(sm) => sm.apply(self.entry.strength()),
                None => self.entry.strength(),
            };

            magic.update_strength(strength);

            let end_upper_level = haystack.lazy_stream_position();

            for e in self.children.iter() {
                e.matches(
                    magic,
                    state,
                    Some(end_upper_level),
                    opt_start_offset,
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
    entries: EntryNode,
}

#[derive(Debug, Clone)]
pub struct DependencyRule {
    name: String,
    rule: MagicRule,
}

impl DependencyRule {
    fn from_pair(pair: Pair<'_, Rule>) -> Result<Self, Error> {
        let mut pairs = pair.clone().into_inner();

        let name_entry = pairs.next().expect("expecting name entry");
        assert_eq!(name_entry.as_rule(), Rule::name_entry);

        let name = name_entry
            .into_inner()
            .find(|p| p.as_rule() == Rule::rule_name)
            .expect("missing rule name")
            .as_str()
            .to_string();

        Ok(Self {
            name,
            rule: MagicRule::from_pair(pair)?,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn rule(&self) -> &MagicRule {
        &self.rule
    }
}

impl MagicRule {
    fn from_pair(pair: Pair<'_, Rule>) -> Result<Self, Error> {
        let mut items = vec![];
        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::name_entry => {
                    let (line, _) = pair.line_col();
                    let mut pairs = pair.into_inner();

                    pairs.next().expect("name entry must have offset");

                    let name = pairs.next().expect("rule must have a name");
                    assert_eq!(Rule::rule_name, name.as_rule());

                    let mut message = None;
                    if let Some(msg) = pairs.next() {
                        message = Some(Message::from_pair(msg))
                    }

                    items.push(Entry::Match(
                        Name {
                            line,
                            offset: Offset::Direct(DirOffset::Start(0)),
                            name: name.as_str().into(),
                            message,
                        }
                        .into(),
                    ))
                }
                Rule::r#match_depth | Rule::r#match_no_depth => {
                    items.push(Entry::Match(Match::from_pair(pair)?));
                }
                Rule::r#use => {
                    items.push(Entry::Match(Use::from_pair(pair).into()));
                }
                Rule::flag => items.push(Entry::Flag(Flag::from_pair(pair)?)),
                Rule::EOI => {}
                _ => panic!("unexpected parsing rule"),
            }
        }

        Ok(Self {
            entries: EntryNode::from_entries(items),
        })
    }

    fn magic<'r, R: Read + Seek>(
        &'r self,
        magic: &mut Magic<'r>,
        opt_start_offset: Option<Offset>,
        haystack: &mut LazyCache<R>,
        db: &'r MagicDb,
        switch_endianness: bool,
        depth: usize,
    ) -> Result<(), Error> {
        self.entries
            .matches(
                magic,
                &mut MatchState::empty(),
                None,
                opt_start_offset,
                haystack,
                db,
                switch_endianness,
                depth,
            )
            .map(|_| ())
    }
}

#[derive(Debug, Clone)]
pub struct MagicFile {
    rules: Vec<MagicRule>,
    dependencies: HashMap<String, DependencyRule>,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
struct ContinuationLevel(u8);

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
    message: Vec<Cow<'m, str>>,
    mime: Option<Cow<'m, str>>,
    strength: Option<u64>,
}

impl<'m> Magic<'m> {
    fn into_static(self) -> Magic<'static> {
        Magic {
            message: self
                .message
                .into_iter()
                .map(Cow::into_owned)
                .map(Cow::Owned)
                .collect(),
            mime: self.mime.map(|cow| Cow::Owned(cow.into_owned())),
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
        self.mime.as_deref().unwrap_or("application/octet-stream")
    }

    #[inline(always)]
    fn push_message<'a: 'm>(&mut self, msg: Cow<'a, str>) {
        if !msg.is_empty() {
            debug!("pushing message: msg={msg} len={}", msg.len());
            self.message.push(msg);
        }
    }

    fn insert_mimetype<'a: 'm>(&mut self, mime: Cow<'a, str>) {
        debug!("insert mime: {:?}", mime);
        self.mime = Some(mime)
    }

    pub fn is_empty(&self) -> bool {
        self.message.is_empty() && self.mime.is_none() && self.strength.is_none()
    }
}

impl MagicFile {
    pub fn open<P: AsRef<Path>>(p: P) -> Result<Self, Error> {
        FileMagicParser::parse_file(p)
    }
}

#[derive(Debug, Default, Clone)]
pub struct MagicDb {
    rules: Vec<MagicRule>,
    dependencies: HashMap<String, DependencyRule>,
}

impl MagicDb {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load(&mut self, mf: MagicFile) -> Result<&mut Self, Error> {
        self.rules.extend(mf.rules);
        self.dependencies.extend(mf.dependencies);
        Ok(self)
    }

    pub fn magic<R: Read + Seek>(
        &self,
        haystack: &mut LazyCache<R>,
    ) -> Result<Vec<(u64, Magic<'_>)>, Error> {
        let mut out = Vec::new();
        // using a BufReader to gain speed
        for r in self.rules.iter() {
            let mut magic = Magic::default();
            r.magic(&mut magic, None, haystack, &self, false, 0)?;

            // it is possible we have a strength with no message
            if !magic.message.is_empty() {
                out.push((magic.strength.unwrap_or_default(), magic));
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn first_magic(rule: &str, content: &[u8]) -> Result<Option<Magic<'static>>, Error> {
        let mut md = MagicDb::new();
        md.load(
            FileMagicParser::parse_str(rule)
                .inspect_err(|e| eprintln!("{e}"))
                .unwrap(),
        )
        .unwrap();
        let mut reader = LazyCache::from_read_seek(Cursor::new(content), 4096, 4 << 20).unwrap();
        let v = md.magic(&mut reader)?;
        Ok(v.into_iter().next().map(|(_, m)| m.into_static()))
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
            FileMagicParser::parse_str($rule)
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
    fn test_unescape() {
        assert_eq!(unescape_string(r#"\ hello"#), " hello");
        assert_eq!(unescape_string(r#"hello\ world"#), "hello world");
        assert_eq!(unescape_string(r#"\^[[:space:]]"#), "^[[:space:]]");
        assert_eq!(unescape_string(r#"\x41"#), "A");
        assert_eq!(unescape_string(r#"\xF\xA"#), "\u{f}\n");
        assert_eq!(unescape_string(r#"\101"#), "A");
        println!("{:?}", unescape_string(r#"\1\0\0\0\0\0\0\300\0\2\0\0"#));
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
        )
    }

    #[test]
    fn test_string_with_mods() {
        assert_magic_match!(
            r#"0	string/fwt	#!\ \ \ /usr/bin/env\ bash	Bourne-Again shell script text executable
!:mime	text/x-shellscript
"#,
            b"#!/usr/bin/env bash i 
        echo hello world"
        );
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
        assert_magic_match!("0 belong &0x0000FFFF Big-endian long", b"\x00\x00\x56\x78");
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

        let local = Local
            .timestamp_opt(946684800, 0)
            .earliest()
            .map(|ts| ts.format(TIMESTAMP_FORMAT).to_string())
            .unwrap();

        assert_magic_match!(
            "4 meldate 946684800 %s",
            b"\x00\x00\x00\x00\x6D\x38\x80\x43",
            local
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
        assert_magic_match!(
            "0 melong &0x0000FFFF Middle-endian long",
            b"\x00\x00\x78\x56"
        ); // 0x00007856 in middle-endian
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
        // The bitwise NOT of 0x12345678 is 0xEDCBA987, which in middle-endian would be 0xCB\xED\x87\xA9
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

        // Test & operation (bitwise AND with 0x0000FFFF)
        // Check if the bitwise AND operation is correctly applied
        assert_magic_match!(
            "0 melong &0x0000FFFF Middle-endian long",
            b"\x00\x00\x12\x34"
        );
        assert_magic_not_match!(
            "0 melong &0x0000FFFF Middle-endian long",
            b"\x56\x78\x12\x34"
        );
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
            "0 uquad &0xFFFFFFFFFFFFFFFF Unsigned quad",
            b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        );
        assert_magic_not_match!(
            "0 uquad &0x0000000000000000 Unsigned quad",
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
}
