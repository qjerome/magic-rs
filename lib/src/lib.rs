use chrono::DateTime;
use dyf::{DynDisplay, FormatString, dformat};
use flagset::{FlagSet, flags};
use pest::{
    Parser, Span,
    error::ErrorVariant,
    iterators::{Pair, Pairs},
};
use pest_derive::Parser;
use regex::bytes::{self, Regex};
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fmt::{self, Debug, Display},
    fs,
    io::{self, BufRead, BufReader, Read, Seek, SeekFrom},
    ops::{Add, BitAnd, Div, Mul, Rem, Sub},
    path::Path,
    string::FromUtf8Error,
};
use thiserror::Error;
use tracing::{Level, debug, enabled, error, trace};

#[derive(Parser)]
#[grammar = "grammar.pest"]
struct FileMagicParser;

fn unescape_string(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(next_char) = chars.peek() {
                match next_char {
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

#[derive(Debug)]
enum Message {
    String(String),
    Format(FormatString),
}

impl Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(s) => write!(f, "{}", s),
            Self::Format(fs) => write!(f, "{}", fs.to_string_lossy()),
        }
    }
}

impl Message {
    fn convert_printf_to_rust_format(c_format: &str) -> (String, bool) {
        let mut rust_format = String::new();
        let mut chars = c_format.chars().peekable();
        let mut replacement_happened = false;

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

                // Append the converted specifier
                rust_format.push_str(rust_specifier);
                replacement_happened = true;
            } else {
                rust_format.push(c);
            }
        }

        (rust_format, replacement_happened)
    }

    #[inline]
    fn from_str<S: AsRef<str>>(s: S) -> Message {
        let (s, _) = Self::convert_printf_to_rust_format(s.as_ref());

        // FIXME: remove unwrap
        let fs = FormatString::from_string(s.to_string()).unwrap();
        if fs.contains_format() {
            Message::Format(fs)
        } else {
            Message::String(fs.into_string())
        }
    }

    #[inline]
    fn format_with(&self, mr: &MatchRes) -> Cow<'_, str> {
        match self {
            Self::String(s) => Cow::Borrowed(s.as_str()),
            Self::Format(s) => Cow::Owned(dformat!(s, mr).unwrap()),
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
    FromUtf8(#[from] FromUtf8Error),
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

impl ParserError {
    fn custom<S: AsRef<str>>(msg: S, span: Span<'_>) -> Self {
        pest::error::Error::new_from_span(
            ErrorVariant::CustomError {
                message: msg.as_ref().into(),
            },
            span,
        )
        .into()
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

#[derive(Debug)]
#[allow(non_camel_case_types)]
enum ScalarDataType {
    belong,
    bequad,
    beshort,
    byte,
    lelong,
    leqdate,
    leshort,
    long,
    short,
    ubelong,
    ubequad,
    ubeshort,
    ubyte,
    ulelong,
    ulequad,
    uleshort,
    uledate,
    lequad,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[allow(non_camel_case_types)]
enum Scalar {
    // value read such as
    // >18	default		x
    // >>18	leshort		x		*unknown arch %#x*
    read,
    belong(i32),
    bequad(i64),
    beshort(i16),
    byte(i8),
    lelong(i32),
    leshort(i16),
    lequad(i64),
    long(i32),
    short(i16),
    ubelong(u32),
    ubequad(u64),
    ubeshort(u16),
    ubyte(u8),
    ulelong(u32),
    ulequad(u64),
    uleshort(u16),
    // FIXME: guessed
    uledate(u32),
}

impl DynDisplay for Scalar {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        match self {
            Scalar::read => unimplemented!(),
            Scalar::belong(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::bequad(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::beshort(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::byte(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::lelong(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::leshort(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::lequad(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::long(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::short(value) => DynDisplay::dyn_fmt(value, f),
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
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Scalar::read => unimplemented!(),
            Scalar::belong(value) => write!(f, "{}", value),
            Scalar::bequad(value) => write!(f, "{}", value),
            Scalar::beshort(value) => write!(f, "{}", value),
            Scalar::byte(value) => write!(f, "{}", value),
            Scalar::lelong(value) => write!(f, "{}", value),
            Scalar::leshort(value) => write!(f, "{}", value),
            Scalar::lequad(value) => write!(f, "{}", value),
            Scalar::long(value) => write!(f, "{}", value),
            Scalar::short(value) => write!(f, "{}", value),
            Scalar::ubelong(value) => write!(f, "{}", value),
            Scalar::ubequad(value) => write!(f, "{}", value),
            Scalar::ubeshort(value) => write!(f, "{}", value),
            Scalar::ubyte(value) => write!(f, "{}", value),
            Scalar::ulelong(value) => write!(f, "{}", value),
            Scalar::ulequad(value) => write!(f, "{}", value),
            Scalar::uleshort(value) => write!(f, "{}", value),
            Scalar::uledate(value) => write!(f, "date({})", value),
        }
    }
}

macro_rules! impl_op {
    ($trait:ident, $method:ident) => {
        impl $trait for Scalar {
            type Output = Self;

            fn $method(self, other: Self) -> Self {
                match (self, other) {
                    (Scalar::belong(a), Scalar::belong(b)) => Scalar::belong(a.$method(b)),
                    (Scalar::bequad(a), Scalar::bequad(b)) => Scalar::bequad(a.$method(b)),
                    (Scalar::beshort(a), Scalar::beshort(b)) => Scalar::beshort(a.$method(b)),
                    (Scalar::byte(a), Scalar::byte(b)) => Scalar::byte(a.$method(b)),
                    (Scalar::lelong(a), Scalar::lelong(b)) => Scalar::lelong(a.$method(b)),
                    (Scalar::leshort(a), Scalar::leshort(b)) => Scalar::leshort(a.$method(b)),
                    (Scalar::lequad(a), Scalar::lequad(b)) => Scalar::lequad(a.$method(b)),
                    (Scalar::long(a), Scalar::long(b)) => Scalar::long(a.$method(b)),
                    (Scalar::short(a), Scalar::short(b)) => Scalar::short(a.$method(b)),
                    (Scalar::ubelong(a), Scalar::ubelong(b)) => Scalar::ubelong(a.$method(b)),
                    (Scalar::ubequad(a), Scalar::ubequad(b)) => Scalar::ubequad(a.$method(b)),
                    (Scalar::ubeshort(a), Scalar::ubeshort(b)) => Scalar::ubeshort(a.$method(b)),
                    (Scalar::ubyte(a), Scalar::ubyte(b)) => Scalar::ubyte(a.$method(b)),
                    (Scalar::ulelong(a), Scalar::ulelong(b)) => Scalar::ulelong(a.$method(b)),
                    (Scalar::ulequad(a), Scalar::ulequad(b)) => Scalar::ulequad(a.$method(b)),
                    (Scalar::uleshort(a), Scalar::uleshort(b)) => Scalar::uleshort(a.$method(b)),
                    _ => panic!("Operation not supported between different Scalar variants"),
                }
            }
        }
    };
}

impl_op!(Add, add);
impl_op!(Sub, sub);
impl_op!(Mul, mul);
impl_op!(Div, div);
impl_op!(BitAnd, bitand);
impl_op!(Rem, rem);

impl ScalarDataType {
    fn from_pair(pair: Pair<'_, Rule>) -> Result<Self, Error> {
        let dt = pair.into_inner().next().expect("datay type expected");
        match dt.as_rule() {
            Rule::belong => Ok(Self::belong),
            Rule::bequad => Ok(Self::bequad),
            Rule::beshort => Ok(Self::beshort),
            Rule::byte => Ok(Self::byte),
            Rule::lelong => Ok(Self::lelong),
            Rule::leqdate => Ok(Self::leqdate),
            Rule::leshort => Ok(Self::leshort),
            Rule::long => Ok(Self::long),
            Rule::short => Ok(Self::short),
            Rule::ubelong => Ok(Self::ubelong),
            Rule::ubequad => Ok(Self::ubequad),
            Rule::ubeshort => Ok(Self::ubeshort),
            Rule::ubyte => Ok(Self::ubyte),
            Rule::ulelong => Ok(Self::ulelong),
            Rule::ulequad => Ok(Self::ulequad),
            Rule::uleshort => Ok(Self::uleshort),
            Rule::lequad => Ok(Self::lequad),
            Rule::uledate => Ok(Self::uledate),
            _ => Err(Error::parser("unimplemented data type", dt.as_span())),
        }
    }

    fn scalar_from_number(&self, i: i64) -> Result<Scalar, ()> {
        match self {
            Self::byte => Ok(Scalar::byte(i as i8)),
            Self::ubyte => Ok(Scalar::ubyte(i as u8)),
            Self::short => Ok(Scalar::short(i as i16)),
            Self::leshort => Ok(Scalar::leshort(i as i16)),
            Self::lelong => Ok(Scalar::lelong(i as i32)),
            Self::belong => Ok(Scalar::belong(i as i32)),
            Self::uleshort => Ok(Scalar::uleshort(i as u16)),
            Self::ulelong => Ok(Scalar::ulelong(i as u32)),
            Self::bequad => Ok(Scalar::bequad(i)),
            Self::lequad => Ok(Scalar::lequad(i)),
            _ => {
                // unimplemented
                Err(())
            }
        }
    }

    #[inline(always)]
    fn read<R: Read>(&self, from: &mut R, switch_endianness: bool) -> Result<Scalar, Error> {
        macro_rules! read {
            ($ty: ty) => {{
                let mut a = [0u8; std::mem::size_of::<$ty>()];
                from.read_exact(&mut a)?;
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

        Ok(match self {
            // signed
            Self::byte => Scalar::byte(read!(u8)[0] as i8),
            Self::short => Scalar::short(i16::from_ne_bytes(read!(i16))),
            Self::leshort => Scalar::leshort(read_le!(i16)),
            Self::lelong => Scalar::lelong(read_le!(i32)),
            Self::lequad => Scalar::lequad(read_le!(i64)),
            Self::bequad => Scalar::bequad(read_be!(i64)),
            Self::belong => Scalar::belong(read_be!(i32)),
            // unsigned
            Self::ubyte => Scalar::ubyte(read!(u8)[0]),
            Self::uleshort => Scalar::uleshort(read_le!(u16)),
            Self::ulelong => Scalar::ulelong(read_le!(u32)),
            Self::uledate => Scalar::uledate(read_le!(u32)),
            _ => unimplemented!("{:?}", self),
        })
    }
}

#[derive(Debug)]
enum Op {
    Mul,
    Add,
    Sub,
    Div,
    Mod,
    And,
}

#[derive(Debug)]
enum CmpOp {
    Eq,
    Lt,
    Gt,
    BitAnd,
    Neg,
}

impl CmpOp {
    fn from_pair(value: Pair<'_, Rule>) -> Result<Self, Error> {
        match value.as_rule() {
            Rule::op_lt => Ok(Self::Lt),
            Rule::op_gt => Ok(Self::Gt),
            Rule::op_and => Ok(Self::BitAnd),
            Rule::op_negate => Ok(Self::Neg),
            Rule::op_eq => Ok(Self::Eq),
            _ => Err(Error::parser("unimplemented operator", value.as_span())),
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
            _ => Err(Error::parser("unimplemented operator", value.as_span())),
        }
    }
}

#[derive(Debug)]
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
            _ => panic!("unknown transform"),
        }
    }
}

// Any Magic Data type
#[derive(Debug)]
enum MagicDataType {
    String,
    PString,
}

impl MagicDataType {
    fn from_rule(r: Rule) -> Self {
        match r {
            Rule::string => MagicDataType::String,
            Rule::pstring => MagicDataType::PString,
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

#[derive(Debug)]
struct RegexTest {
    re: bytes::Regex,
    length: Option<usize>,
    mods: FlagSet<ReMod>,
    str_mods: Option<FlagSet<StringMod>>,
}

impl From<RegexTest> for Test {
    fn from(value: RegexTest) -> Self {
        Self::Regex(value)
    }
}

impl RegexTest {
    fn from_pair_with_re(pair: Pair<'_, Rule>, re: &str) -> Self {
        let mut length = None;
        let mut mods = FlagSet::empty();
        for p in pair.into_inner() {
            match p.as_rule() {
                Rule::number => length = Some(parse_number(p) as usize),
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
                // this should never happen
                _ => unimplemented!(),
            }
        }

        Self {
            //FIXME: remove unwrap
            re: bytes::Regex::new(re).unwrap(),
            length,
            mods,
            str_mods: None,
        }
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

#[derive(Debug)]
struct StringTest {
    str: String,
    length: Option<usize>,
    mods: FlagSet<StringMod>,
}

impl From<StringTest> for Test {
    fn from(value: StringTest) -> Self {
        Self::String(value)
    }
}

impl StringTest {
    fn from_pair_with_str(pair: Pair<'_, Rule>, str: &str) -> Self {
        let mut length = None;
        let mut mods = FlagSet::empty();
        for p in pair.into_inner() {
            match p.as_rule() {
                Rule::number => length = Some(parse_number(p) as usize),
                Rule::string_mod => {
                    for m in p.as_str().chars() {
                        match m {
                            'b' => mods |= StringMod::ForceBin,
                            'C' => mods |= StringMod::UpperInsensitive,
                            'c' => mods |= StringMod::LowerInsensitive,
                            'f' => mods |= StringMod::FullWordMatch,
                            'T' => mods |= StringMod::Trim,
                            't' => mods |= StringMod::ForceText,
                            'W' => mods |= StringMod::CompactWhitespace,
                            'w' => mods |= StringMod::OptBlank,
                            _ => {}
                        }
                    }
                }
                // this should never happen
                _ => unimplemented!(),
            }
        }
        Self {
            //FIXME: remove unwrap
            str: str.to_string(),
            length,
            mods,
        }
    }
}

#[derive(Debug)]
struct SearchTest {
    str: String,
    n_pos: Option<usize>,
    mods: FlagSet<StringMod>,
}

impl From<SearchTest> for Test {
    fn from(value: SearchTest) -> Self {
        Self::Search(value)
    }
}

impl SearchTest {
    fn from_pair_with_str(pair: Pair<'_, Rule>, str: &str) -> Self {
        let mut length = None;
        let mut mods = FlagSet::empty();
        for p in pair.into_inner() {
            match p.as_rule() {
                Rule::number => length = Some(parse_number(p) as usize),
                Rule::string_mod => {
                    for m in p.as_str().chars() {
                        match m {
                            'b' => mods |= StringMod::ForceBin,
                            'C' => mods |= StringMod::UpperInsensitive,
                            'c' => mods |= StringMod::LowerInsensitive,
                            'f' => mods |= StringMod::FullWordMatch,
                            'T' => mods |= StringMod::Trim,
                            't' => mods |= StringMod::ForceText,
                            'W' => mods |= StringMod::CompactWhitespace,
                            'w' => mods |= StringMod::OptBlank,
                            _ => {}
                        }
                    }
                }
                // this should never happen
                _ => unimplemented!(),
            }
        }
        Self {
            //FIXME: remove unwrap
            str: str.to_string(),
            n_pos: length,
            mods,
        }
    }
}

#[derive(Debug)]
struct ScalarTest {
    ty: ScalarDataType,
    transform: Option<Transform>,
    cmp_op: CmpOp,
    value: Scalar,
}

#[derive(Debug)]
enum Test {
    Scalar(ScalarTest),
    Read(MagicDataType),
    String(StringTest),
    Search(SearchTest),
    PString(String),
    Regex(RegexTest),
    Clear,
    Default,
}

// the value read from the haystack we want to
// match against
#[derive(Debug, PartialEq, Eq)]
enum TestValue {
    Scalar(Scalar),
    String(String),
    PString(String),
    Bytes(Vec<u8>),
}

impl DynDisplay for TestValue {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        match self {
            Self::Scalar(s) => DynDisplay::dyn_fmt(s, f),
            Self::String(s) => DynDisplay::dyn_fmt(s, f),
            Self::PString(s) => DynDisplay::dyn_fmt(s, f),
            Self::Bytes(b) => Ok(format!("{:?}", b)),
        }
    }
}

impl DynDisplay for &TestValue {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        // Dereference self to get the TestValue and call its fmt method
        DynDisplay::dyn_fmt(*self, f)
    }
}

impl Display for TestValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scalar(s) => write!(f, "{}", s),
            Self::String(s) => write!(f, "{}", s),
            Self::PString(s) => write!(f, "{}", s),
            Self::Bytes(b) => write!(f, "{:?}", b),
        }
    }
}

enum MatchRes {
    String(String),
    Scalar(Scalar),
}

impl DynDisplay for &MatchRes {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        (*self).dyn_fmt(f)
    }
}

impl DynDisplay for MatchRes {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        match self {
            Self::Scalar(s) => s.dyn_fmt(f),
            Self::String(s) => s.dyn_fmt(f),
        }
    }
}

impl Test {
    fn from_pair(pair: Pair<'_, Rule>) -> Result<Self, Error> {
        let t = match pair.as_rule() {
            Rule::scalar_test => {
                let mut pairs = pair.into_inner();

                let ty_pair = pairs.next().expect("data type pair expected");
                let ty_span = ty_pair.as_span();
                let ty = ScalarDataType::from_pair(ty_pair)?;

                let mut transform = None;

                let mut next = pairs.next().expect("expect token pair");
                if matches!(next.as_rule(), Rule::scalar_transform) {
                    let mut transform_pairs = next.into_inner();
                    let op_pair = transform_pairs.next().expect("expect operator pair");
                    let op = Op::from_pair(op_pair)?;

                    let number = transform_pairs.next().expect("expect number pair");
                    let span = number.as_span();
                    transform = Some(Transform {
                        op,
                        num: ty
                            .scalar_from_number(parse_number(number))
                            .map_err(|_| Error::parser("unimplemented scalar", span))?,
                    });
                    next = pairs.next().expect("expect token pair");
                }

                let mut condition = CmpOp::Eq;
                if matches!(next.as_rule(), Rule::scalar_condition) {
                    condition = CmpOp::from_pair(
                        next.into_inner().next().expect("expecting cmp operator"),
                    )?;
                    next = pairs.next().expect("expect token pair");
                }

                let value_pair = next;

                let value = match value_pair.as_rule() {
                    Rule::value_read => Scalar::read,
                    Rule::scalar_value => {
                        let number_pair = value_pair
                            .into_inner()
                            .next()
                            .expect("number pair expected");

                        ty.scalar_from_number(parse_number(number_pair))
                            .map_err(|_| Error::parser("unimplemented scalar", ty_span))?
                    }
                    _ => unimplemented!(),
                };

                Self::Scalar(ScalarTest {
                    ty,
                    transform,
                    cmp_op: condition,
                    value,
                })
            }
            Rule::string_test => {
                let mut string_test = pair.into_inner();

                let test_type = string_test.next().expect("expecting a string type");

                let test_value = string_test.next().expect("expecting a string value");

                match test_value.as_rule() {
                    Rule::string_value => match test_type.as_rule() {
                        Rule::string => StringTest::from_pair_with_str(
                            test_type,
                            &unescape_string(test_value.as_str()),
                        )
                        .into(),
                        Rule::search => SearchTest::from_pair_with_str(
                            test_type,
                            &unescape_string(test_value.as_str()),
                        )
                        .into(),
                        Rule::pstring => Self::PString(unescape_string(test_value.as_str())),
                        Rule::regex => RegexTest::from_pair_with_re(
                            test_type,
                            &unescape_string(test_value.as_str()),
                        )
                        .into(),
                        _ => unimplemented!(),
                    },
                    Rule::value_read => Self::Read(MagicDataType::from_rule(test_type.as_rule())),
                    _ => unimplemented!(),
                }
            }
            Rule::clear_test => Self::Clear,
            Rule::default_test => Self::Default,
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
                    Self::string_to_re_pattern(&s.str, s.mods, matches!(s.n_pos, Some(1)));

                // we handle cases where we wanna match more positions
                if let Some(n_pos) = s.n_pos {
                    if n_pos > 1 {
                        pattern.insert_str(0, &format!("^.{{0,{}}}?", n_pos - 1));
                    }
                }

                RegexTest {
                    // FIXME: remove unwrap
                    re: Regex::new(&pattern).unwrap(),
                    // if there is a length we add string size so that we can match full string until the last search token
                    length: None,
                    mods: FlagSet::empty(),
                    str_mods: Some(s.mods),
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
                        mods: FlagSet::empty(),
                        str_mods: Some(st.mods),
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
    fn read_test_value<R: Read>(
        &self,
        haystack: &mut R,
        switch_endianness: bool,
    ) -> Result<TestValue, Error> {
        macro_rules! read {
            ($ty: ty) => {{
                let mut a = [0u8; std::mem::size_of::<$ty>()];
                haystack.read_exact(&mut a)?;
                a
            }};
        }

        macro_rules! read_le {
            ($ty: ty) => {
                <$ty>::from_le_bytes(read!($ty))
            };
        }

        match self {
            Self::Scalar(t) => {
                t.ty.read(haystack, switch_endianness)
                    .map(TestValue::Scalar)
            }
            Self::String(s) => {
                let mut buf = vec![0u8; s.length.unwrap_or(s.str.len())];
                haystack.read_exact(buf.as_mut_slice())?;
                String::from_utf8(buf)
                    .map(TestValue::String)
                    .map_err(Error::from)
            }
            Self::PString(s) => {
                // FIXME: maybe we could optimize here by reading testing on size
                let pstring_len = 1;
                let _ = read_le!(u8);
                let mut buf = vec![0u8; s.len()];
                haystack.read_exact(buf.as_mut_slice())?;

                String::from_utf8(buf)
                    .map(TestValue::PString)
                    .map_err(Error::from)
            }
            Self::Regex(r) => {
                let mut buf = vec![0u8; r.length.unwrap_or(8192)];
                let n = haystack.read(buf.as_mut_slice())?;
                buf.truncate(n);
                Ok(TestValue::Bytes(buf))
            }
            Self::Read(t) => match t {
                MagicDataType::String => {
                    let mut r = BufReader::new(haystack);
                    let mut buf = vec![];
                    r.read_until(b'\0', &mut buf)?;
                    String::from_utf8(buf)
                        .map(TestValue::String)
                        .map_err(Error::from)
                        .inspect_err(|e| println!("{}", e))
                }
                MagicDataType::PString => {
                    let slen = read_le!(u8) as usize;
                    let mut buf = vec![0u8; slen];
                    haystack.read_exact(&mut buf)?;
                    String::from_utf8(buf)
                        .map(TestValue::PString)
                        .map_err(Error::from)
                        .inspect_err(|e| println!("{}", e))
                }
                _ => unimplemented!(),
            },

            _ => unimplemented!(),
        }
    }

    #[inline(always)]
    fn match_value(&self, tv: TestValue) -> Option<MatchRes> {
        // always true when we want to read value
        if let Self::Read(v) = self {
            match tv {
                TestValue::PString(ps) => {
                    if matches!(v, MagicDataType::PString) {
                        return Some(MatchRes::String(ps));
                    }
                }
                TestValue::String(s) => {
                    if matches!(v, MagicDataType::String) {
                        return Some(MatchRes::String(s));
                    }
                }
                _ => panic!("not good"),
            }

            // FIXME: remove this
            panic!("not good")
        }

        match tv {
            TestValue::Scalar(ts) => {
                if let Self::Scalar(t) = self {
                    if matches!(t.value, Scalar::read) {
                        return Some(MatchRes::Scalar(ts));
                    }

                    let tv = t.transform.as_ref().map(|t| t.apply(ts)).unwrap_or(ts);

                    let ok = match t.cmp_op {
                        CmpOp::Eq => tv == t.value,
                        CmpOp::Lt => tv < t.value,
                        CmpOp::Gt => tv > t.value,
                        CmpOp::Neg => tv != t.value,
                        CmpOp::BitAnd => tv & t.value == t.value,
                    };

                    if ok {
                        return Some(MatchRes::Scalar(tv));
                    }
                }
            }
            TestValue::String(tv) => {
                if let Self::String(m) = self {
                    if tv == m.str {
                        return Some(MatchRes::String(tv));
                    }
                }
            }
            TestValue::PString(tv) => {
                if let Self::PString(m) = self {
                    if &tv == m {
                        return Some(MatchRes::String(tv));
                    }
                }
            }
            TestValue::Bytes(buf) => {
                if let Self::Regex(r) = self {
                    if let Some(re_match) = r.re.find(&buf) {
                        return Some(MatchRes::String(
                            std::str::from_utf8(re_match.as_bytes()).unwrap().into(),
                        ));
                    }
                }
            }
        }

        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Offset {
    Start(i64),
    Current(i64),
    End(i64),
}

impl Default for Offset {
    fn default() -> Self {
        Self::Current(0)
    }
}

fn parse_number(number_pair: Pair<'_, Rule>) -> i64 {
    let number_token = number_pair
        .into_inner()
        .next()
        .expect("expect number kind pair");
    match number_token.as_rule() {
        Rule::b10_number => number_token.as_str().parse::<i64>().unwrap(),
        Rule::b16_number => {
            u64::from_str_radix(number_token.as_str().strip_prefix("0x").unwrap(), 16).unwrap()
                as i64
        }
        _ => panic!("unexpected number"),
    }
}

impl Offset {
    fn from_pair(pair: Pair<'_, Rule>) -> Self {
        let mut pairs = pair.into_inner();
        let pair = pairs.next().expect("offset must have token");

        match pair.as_rule() {
            Rule::abs_offset => {
                let number_pairs = pair.into_inner().next().expect("number pair expected");

                match number_pairs.as_rule() {
                    Rule::neg_number => {
                        let number_pair = number_pairs
                            .into_inner()
                            .next()
                            .expect("number pair expected");
                        Self::End(-parse_number(number_pair))
                    }
                    Rule::number => Self::Start(parse_number(number_pairs)),
                    _ => unimplemented!(),
                }
            }
            Rule::rel_offset => {
                let number_pairs = pair.into_inner().next().expect("number pair expected");
                match number_pairs.as_rule() {
                    Rule::neg_number => {
                        let number_pair = number_pairs
                            .into_inner()
                            .next()
                            .expect("number pair expected");
                        Self::Current(-parse_number(number_pair))
                    }
                    Rule::number => Self::Current(parse_number(number_pairs)),
                    _ => unimplemented!(),
                }
            }
            _ => panic!("unexpected token"),
        }
    }
}

#[derive(Debug)]
pub struct Match {
    depth: u8,
    offset: Offset,
    test: Test,
    message: Option<Message>,
}

impl Match {
    fn from_pairs(pairs: Pairs<'_, Rule>) -> Result<Self, Error> {
        let mut pairs = pairs;

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

        let message = pairs.next().map(|p| Message::from_str(p.as_str()));

        Ok(Self {
            depth,
            offset,
            test,
            message,
        })
    }

    #[inline]
    fn matches<'a, R: Read + Seek>(
        &'a self,
        magic: &mut Magic<'a>,
        opt_start_offset: Option<Offset>,
        haystack: &mut R,
        switch_endianness: bool,
    ) -> Result<bool, Error> {
        // FIXME: handle better
        let current_offset = haystack.stream_position()?;

        let i = opt_start_offset
            .map(|so| match so {
                Offset::Start(s) => s,
                // FIXME: this is relative to previous match so we need to carry up this information
                Offset::Current(_) => unimplemented!(),
                Offset::End(e) => e,
            })
            .unwrap_or_default();

        match self.offset {
            Offset::Start(s) => haystack.seek(SeekFrom::Start((s + i) as u64))?,
            // FIXME: this is relative to previous match so we need to carry up this information
            Offset::Current(c) => haystack.seek(SeekFrom::Current(c))?,
            Offset::End(e) => haystack.seek(SeekFrom::End(e + i))?,
        };

        // handle clear and default tests
        match &self.test {
            Test::Clear => {
                magic.clear_continuation_level(&self.continuation_level());
                return Ok(true);
            }
            Test::Default => {
                magic.set_continuation_level(self.continuation_level());
                return Ok(true);
            }
            _ => {}
        }

        let mut trace_msg = None;

        if enabled!(Level::DEBUG) {
            trace_msg = Some(vec![format!(
                "stream offset={} ",
                haystack.stream_position().unwrap_or_default()
            )])
        }

        if let Ok(tv) = self.test.read_test_value(haystack, switch_endianness) {
            // we need to adjust stream offset if this is a regex test since we read beyond the match
            let adjust_stream_offset = matches!(&self.test, Test::Regex(_));

            trace_msg
                .as_mut()
                //.map(|v| v.push(format!("test={:?} value={:?}", self.test, &tv)));
                .map(|v| v.push(format!("test={:?}", self.test)));

            let match_res = self.test.match_value(tv);

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
            trace_msg.map(|m| trace!("{}", m.join(" ")));

            if let Some(mr) = match_res {
                if let Some(s) = self.message.as_ref() {
                    magic.push_message(s.format_with(&mr))
                }
                // we re-ajust the stream offset only if we have a match
                if adjust_stream_offset {
                    if let MatchRes::String(s) = mr {
                        haystack.seek(SeekFrom::Start(current_offset + s.len() as u64))?;
                    }
                }
                magic.set_continuation_level(self.continuation_level());
                return Ok(true);
            }
        }

        Ok(false)
    }

    #[inline(always)]
    fn continuation_level(&self) -> ContinuationLevel {
        ContinuationLevel(self.depth, self.offset)
    }
}

#[derive(Debug)]
pub struct Use {
    depth: u8,
    start_offset: Offset,
    rule_name: String,
    switch_endianness: bool,
}

impl Use {
    fn from_pairs(pairs: Pairs<'_, Rule>) -> Self {
        let mut pairs = pairs;

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

        Self {
            depth,
            start_offset: offset,
            rule_name: rule_name_token.as_str().to_string(),
            switch_endianness: endianness_switch,
        }
    }

    fn matches<'a, R: Read + Seek>(
        &'a self,
        magic: &mut Magic<'a>,
        haystack: &mut R,
        deps: &'a HashMap<String, DependencyRule>,
    ) -> Result<(), Error> {
        debug!(
            "matching use {} endianness={:?}",
            self.rule_name, self.switch_endianness
        );

        // FIXME: this sucks as we are copying a String
        // probably a better way can be found
        let dr: &DependencyRule = deps
            .get(&self.rule_name)
            .ok_or(Error::MissingRule(self.rule_name.clone()))?;

        let m = dr.rule.magic(
            magic,
            Some(self.start_offset),
            haystack,
            deps,
            false,
            self.switch_endianness,
        );

        m
    }

    #[inline(always)]
    fn continuation_level(&self) -> ContinuationLevel {
        ContinuationLevel(self.depth, self.start_offset)
    }
}

#[derive(Debug)]
pub enum Flag {
    Mime(String),
    Ext(HashSet<String>),
    Strength { op: Op, by: u8 },
}

impl Flag {
    fn from_pairs(mut pairs: Pairs<'_, Rule>) -> Self {
        let flag = pairs.next().expect("expecting a valid flag");

        match flag.as_rule() {
            Rule::mime_flag => Self::Mime(
                flag.into_inner()
                    .next()
                    .expect("expecting mime type")
                    .as_str()
                    .into(),
            ),
            Rule::strength_flag => Self::Strength { op: Op::Mul, by: 0 },
            Rule::ext_flag => {
                let exts = flag.into_inner().next().expect("expecting extension list");
                assert_eq!(exts.as_rule(), Rule::exts);
                Self::Ext(exts.as_str().split('/').map(|s| s.into()).collect())
            }
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug)]
pub enum Entry {
    Match(Match),
    Flag(Flag),
    Use(Use),
}

#[derive(Debug)]
pub struct MagicRule {
    entries: Vec<Entry>,
}

impl MagicRule {
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }
}

#[derive(Debug)]
pub struct DependencyRule {
    name: String,
    rule: MagicRule,
}

impl DependencyRule {
    fn from_pair(pair: Pair<'_, Rule>) -> Result<Self, Error> {
        let mut pairs = pair.into_inner();
        let name = pairs
            .next()
            .map(|p| p.as_str().to_string())
            .expect("dependency rule must have a name");

        let rule_pair = pairs.next().expect("dependency contain entries");

        Ok(Self {
            name,
            rule: MagicRule::from_pair(rule_pair)?,
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
                Rule::r#match_depth | Rule::r#match_no_depth => {
                    items.push(Entry::Match(Match::from_pairs(pair.into_inner())?));
                }
                Rule::r#use => {
                    items.push(Entry::Use(Use::from_pairs(pair.into_inner())));
                }
                Rule::flag => items.push(Entry::Flag(Flag::from_pairs(pair.into_inner()))),
                _ => panic!("unexpected parsing rule"),
            }
        }

        Ok(Self { entries: items })
    }

    fn magic<'r, R: Read + Seek>(
        &'r self,
        magic: &mut Magic<'r>,
        opt_start_offset: Option<Offset>,
        haystack: &mut R,
        deps: &'r HashMap<String, DependencyRule>,
        abort_first_no_match: bool,
        switch_endianness: bool,
    ) -> Result<(), Error> {
        let mut prev_item_match = true;
        let mut last_continuation_level = ContinuationLevel::default();

        for (i, item) in self.entries.iter().enumerate() {
            match item {
                Entry::Flag(f) => {
                    if prev_item_match {
                        match f {
                            Flag::Mime(mime) => magic.insert_mimetype(mime.into()),
                            _ => error!("flag not implemented {f:?}"),
                        }
                    }
                }
                Entry::Match(m) => {
                    let cont_level = m.continuation_level();

                    if magic.get_continuation_level(&cont_level)
                        || (m.depth > last_continuation_level.0 && !prev_item_match)
                    {
                        trace!("skip: {m:?}");
                        prev_item_match = false;
                        continue;
                    }

                    // we are matching stuff
                    prev_item_match =
                        m.matches(magic, opt_start_offset, haystack, switch_endianness)?;

                    // we abort if first item doesn't match
                    if abort_first_no_match && i == 0 && !prev_item_match {
                        return Ok(());
                    }

                    last_continuation_level = cont_level;
                }

                Entry::Use(u) => {
                    if u.depth > last_continuation_level.0 && !prev_item_match {
                        prev_item_match = false;
                        continue;
                    }

                    let mut tmp = Magic::default();
                    u.matches(&mut tmp, haystack, deps)?;
                    magic.merge(tmp);

                    prev_item_match = false;
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct MagicFile {
    rules: Vec<MagicRule>,
    dependencies: HashMap<String, DependencyRule>,
}

#[derive(Debug, Default, Hash, PartialEq, Eq, Clone, Copy)]
struct ContinuationLevel(u8, Offset);

#[derive(Debug, Default)]
pub struct Magic<'m> {
    message: Vec<Cow<'m, str>>,
    mime: Option<Cow<'m, str>>,
    strength: u64,
    continuation_levels: HashSet<ContinuationLevel>,
}

impl<'m> Magic<'m> {
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
    pub fn mimetype(&self) -> &str {
        self.mime.as_deref().unwrap_or("application/octet-stream")
    }

    #[inline(always)]
    fn push_message<'a: 'm>(&mut self, msg: Cow<'a, str>) {
        self.message.push(msg);
    }

    #[inline(always)]
    fn get_continuation_level(&mut self, level: &ContinuationLevel) -> bool {
        self.continuation_levels.contains(level)
    }

    #[inline(always)]
    fn set_continuation_level(&mut self, level: ContinuationLevel) {
        self.continuation_levels.insert(level);
    }

    #[inline(always)]
    fn clear_continuation_level(&mut self, level: &ContinuationLevel) {
        self.continuation_levels.remove(level);
    }

    fn insert_mimetype<'a: 'm>(&mut self, mime: Cow<'a, str>) {
        debug!("insert mime: {:?}", mime);
        self.mime = Some(mime)
    }

    fn merge(&mut self, other: Self) {
        self.message.extend(other.message);
        self.mime = other.mime;
        // FIXME: correctly handle strength
    }
}

impl MagicFile {
    pub fn open<P: AsRef<Path>>(p: P) -> Result<Self, Error> {
        FileMagicParser::parse_file(p)
    }

    pub fn magic<R: Read + Seek>(&self, haystack: &mut R) -> Magic<'_> {
        let mut magic = Magic::default();
        // using a BufReader to gain speed
        let mut br = io::BufReader::new(haystack);
        for r in self.rules.iter() {
            // FIXME: this is bad
            r.magic(&mut magic, None, &mut br, &self.dependencies, true, false);
        }

        magic
    }

    pub fn rules(&self) -> &[MagicRule] {
        &self.rules
    }

    pub fn dep_rules(&self) -> impl Iterator<Item = &DependencyRule> {
        self.dependencies.values().map(|v| v)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Cursor, process::Command};

    use super::*;

    fn parse_rule(rule: &str) -> MagicFile {
        let pairs = FileMagicParser::parse(Rule::file, rule).unwrap_or_else(|e| panic!("{}", e));

        for rules in pairs {
            for pair in rules.into_inner() {
                // A pair is a combination of the rule which matched and a span of input
                println!("Rule:    {:?}", pair.as_rule());
                println!("Span:    {:?}", pair.as_span());
                println!("Text:    {}", pair.as_str());
                println!();
                for pair in pair.into_inner() {
                    println!("\tRule:    {:?}", pair.as_rule());
                    println!("\tSpan:    {:?}", pair.as_span());
                    println!("\tText:    {}", pair.as_str());
                    println!();
                }
            }
        }

        let me = FileMagicParser::parse_str(rule).unwrap();
        println!("{:#?}", me);
        me
    }

    #[test]
    fn one_liners() {
        let tests = [
            "0 string GIF89a GIF image data",
            "0 string %PDF PDF document",
            "0 byte 0x00 NULL byte found",
            r#"0   string   \x89PNG    PNG image data"#,
        ];

        for t in tests {
            let pairs = FileMagicParser::parse(Rule::rule, t).unwrap_or_else(|e| panic!("{}", e));

            for pair in pairs {
                // A pair is a combination of the rule which matched and a span of input
                println!("Rule:    {:?}", pair.as_rule());
                println!("Span:    {:?}", pair.as_span());
                println!("Text:    {}", pair.as_str());
                println!()
            }
        }
    }

    #[test]
    fn full_rule() {
        let rule = r#"0	lelong		0x4F153D1D
>4	lelong		0x00010112	PS4 Signed ELF file
>8	byte		1		\b, SELF/SPRX signed-elf/prx
>8	byte		2		\b, SRVK signed-revoke-list
>8	byte		3		\b, SPKG signed-package
>8	byte		4		\b, SSPP signed-security-policy-profile
>8	byte		5		\b, SDIFF signed-diff
>8	byte		6		\b, SPSFO signed-param-sfo
>42	leshort		x		\b, header size %d
>>0	use		elf-le
>9	byte&0xf0	x		\b, version %#x	
>9	byte&0x0f	4		\b, game
>9	byte&0x0f	5		\b, module
>9	byte&0x0f	6		\b, video app
>9	byte&0x0f	8		\b, System/EX application
>9	byte&0x0f	9		\b, System/EX module/dll
!:strength +20
!:strength - 15
!:strength / 2
!:strength *3
!:mime model/e57
!:mime	image/x-canon-crw
!:mime	application/vnd.stardivision.writer
!:mime		application/x-blender
!:mime	application/x-freemind
!:mime	audio/x-musepack
!:mime	application/x-mobipocket-ebook
!:mime	application/x-fzip
!:mime application/x-epoc-data
!:mime  application/x-freeplane
!:ext hdt
!:ext	aml
!:ext	doc/dot/
!:ext	ico
!:ext		cel
!:ext	rds
!:ext	dmg/iso
!:ext	xla
!:ext	ini/inf
!:ext	wc
!:apple	EMAxTEXT
!:apple	????XLS9
!:apple							????iCal
!:apple	????JIFf
!:apple	????XLS5
!:apple	????L123
!:apple	????amrw
!:apple	ALB3ALD3
!:apple	LZIVZIVU
!:apple	ALD5ALB5
#>12	leshort		x		\b, header size %d
#>14	leshort		x		\b, signature size %d
#>16	lelong		x		\b, file size %d
#>18	leshort		x		\b, number of segments %d
#>20	leshort		22"#;

        parse_rule(rule);
    }

    #[test]
    fn test_parse_dep_rule() {
        let dep = r#"0	name		elf-mips
>0	lelong&0xf0000000	0x00000000	MIPS-I
>0	lelong&0xf0000000	0x10000000	MIPS-II
>0	lelong&0xf0000000	0x20000000	MIPS-III
>0	lelong&0xf0000000	0x30000000	MIPS-IV
>0	lelong&0xf0000000	0x40000000	MIPS-V
# this is a comment
>0	lelong&0xf0000000	0x50000000	MIPS32
>0	lelong&0xf0000000	0x60000000	MIPS64
>0	lelong&0xf0000000	0x70000000	MIPS32 rel2
>0	lelong&0xf0000000	0x80000000	MIPS64 rel2
>0	lelong&0xf0000000	0x90000000	MIPS32 rel6
>0	lelong&0xf0000000	0xa0000000	MIPS64 rel6
"#;
        parse_rule(dep);
    }

    #[test]
    fn test_parse_full_file() {
        let file = r#"0	name		elf-mips
>0	lelong&0xf0000000	0x00000000	MIPS-I
>0	lelong&0xf0000000	0x10000000	MIPS-II
>0	lelong&0xf0000000	0x20000000	MIPS-III
>0	lelong&0xf0000000	0x30000000	MIPS-IV
>0	lelong&0xf0000000	0x40000000	MIPS-V
>0	lelong&0xf0000000	0x50000000	MIPS32
>0	lelong&0xf0000000	0x60000000	MIPS64
>0	lelong&0xf0000000	0x70000000	MIPS32 rel2
>0	lelong&0xf0000000	0x80000000	MIPS64 rel2
>0	lelong&0xf0000000	0x90000000	MIPS32 rel6
>0	lelong&0xf0000000	0xa0000000	MIPS64 rel6

0	name		elf-sparc
>0	lelong&0x00ffff00	0x00000100	V8+ Required,
>0	lelong&0x00ffff00	0x00000200	Sun UltraSPARC1 Extensions Required,
>0	lelong&0x00ffff00	0x00000400	HaL R1 Extensions Required,
>0	lelong&0x00ffff00	0x00000800	Sun UltraSPARC3 Extensions Required,
>0	lelong&0x3		0		total store ordering,
>0	lelong&0x3		1		partial store ordering,
>0	lelong&0x3		2		relaxed memory ordering,

0	name		elf-pa-risc
>2	leshort		0x020b		1.0
>2	leshort		0x0210		1.1
>2	leshort		0x0214		2.0
>0	leshort		&0x0008		(LP64)

0	name		elf-riscv
>0	lelong&0x00000001	0x00000001	RVC,
>0	lelong&0x00000008	0x00000008	RVE,
>0	lelong&0x00000006	0x00000000	soft-float ABI,
>0	lelong&0x00000006	0x00000002	single-float ABI,
>0	lelong&0x00000006	0x00000004	double-float ABI,
>0	lelong&0x00000006	0x00000006	quad-float ABI,

0	name		elf-le
>16	leshort		0		no file type,
!:mime	application/octet-stream
>16	leshort		1		relocatable,
!:mime	application/x-object
>16	leshort		2		executable,
!:mime	application/x-executable
>16	leshort		3		pie executable,
>16	leshort		!3		shared object,
>16	leshort		4		core file,
!:mime	application/x-coredump
# OS-specific
>7	byte		202
>>16	leshort		0xFE01		executable,
!:mime	application/x-executable
# Core file detection is not reliable.
#>>>(0x38+0xcc) string	>\0		of '%s'
#>>>(0x38+0x10) lelong	>0		(signal %d),
>16	leshort		&0xff00
>>18	leshort		!8		processor-specific,
>>18	leshort		8
>>>16	leshort		0xFF80		PlayStation 2 IOP module,
!:mime	application/x-sharedlib
>>>16	leshort		!0xFF80		processor-specific,
>18	clear		x
>18	leshort		0		no machine,
>18	leshort		1		AT&T WE32100,
>18	leshort		2		SPARC,
>18	leshort		3		Intel i386,
>18	leshort		4		Motorola m68k,
>>4	byte		1
>>>36	lelong		&0x01000000	68000,
>>>36	lelong		&0x00810000	CPU32,
>>>36	lelong		0		68020,
>18	leshort		5		Motorola m88k,
>18	leshort		6		Intel i486,
>18	leshort		7		Intel i860,
# The official e_machine number for MIPS is now #8, regardless of endianness.
# The second number (#10) will be deprecated later. For now, we still
# say something if #10 is encountered, but only gory details for #8.
>18	leshort		8		MIPS,
>>4	byte		1
>>>36	lelong		&0x20		N32
>18	leshort		10		MIPS,
>>4	byte		1
>>>36	lelong		&0x20		N32
>18	leshort		8
# only for 32-bit
>>4	byte		1
>>>36	use		elf-mips
# only for 64-bit
>>4	byte		2
>>>48	use		elf-mips
>18	leshort		9		Amdahl,
>18	leshort		10		MIPS (deprecated),
>18	leshort		11		RS6000,
>18	leshort		15		PA-RISC,
# only for 32-bit
>>4	byte		1
>>>36	use		elf-pa-risc
# only for 64-bit
>>4	byte		2
>>>48	use		elf-pa-risc
>18	leshort		16		nCUBE,
>18	leshort		17		Fujitsu VPP500,
>18	leshort		18		SPARC32PLUS,
# only for 32-bit
>>4	byte		1
>>>36	use		elf-sparc
>18	leshort		19		Intel 80960,
>18	leshort		20		PowerPC or cisco 4500,
>18	leshort		21		64-bit PowerPC or cisco 7500,
>>48	lelong		0		Unspecified or Power ELF V1 ABI,
>>48	lelong		1		Power ELF V1 ABI,
>>48	lelong		2		OpenPOWER ELF V2 ABI,
>18	leshort		22		IBM S/390,
>18	leshort		23		Cell SPU,
>18	leshort		24		cisco SVIP,
>18	leshort		25		cisco 7200,
>18	leshort		36		NEC V800 or cisco 12000,
>18	leshort		37		Fujitsu FR20,
>18	leshort		38		TRW RH-32,
>18	leshort		39		Motorola RCE,
>18	leshort		40		ARM,
>>4	byte		1
>>>36	lelong&0xff000000	0x04000000	EABI4
>>>36	lelong&0xff000000	0x05000000	EABI5
>>>36	lelong		&0x00800000	BE8
>>>36	lelong		&0x00400000	LE8
>18	leshort		41		Alpha,
>18	leshort		42		Renesas SH,
>18	leshort		43		SPARC V9,
>>4	byte		2
>>>48	use		elf-sparc
>18	leshort		44		Siemens Tricore Embedded Processor,
>18	leshort		45		Argonaut RISC Core, Argonaut Technologies Inc.,
>18	leshort		46		Renesas H8/300,
>18	leshort		47		Renesas H8/300H,
>18	leshort		48		Renesas H8S,
>18	leshort		49		Renesas H8/500,
>18	leshort		50		IA-64,
>18	leshort		51		Stanford MIPS-X,
>18	leshort		52		Motorola Coldfire,
>18	leshort		53		Motorola M68HC12,
>18	leshort		54		Fujitsu MMA,
>18	leshort		55		Siemens PCP,
>18	leshort		56		Sony nCPU,
>18	leshort		57		Denso NDR1,
>18	leshort		58		Start*Core,
>18	leshort		59		Toyota ME16,
>18	leshort		60		ST100,
>18	leshort		61		Tinyj emb.,
>18	leshort		62		x86-64,
>18	leshort		63		Sony DSP,
>18	leshort		64		DEC PDP-10,
>18	leshort		65		DEC PDP-11,
>18	leshort		66		FX66,
>18	leshort		67		ST9+ 8/16 bit,
>18	leshort		68		ST7 8 bit,
>18	leshort		69		MC68HC16,
>18	leshort		70		MC68HC11,
>18	leshort		71		MC68HC08,
>18	leshort		72		MC68HC05,
>18	leshort		73		SGI SVx or Cray NV1,
>18	leshort		74		ST19 8 bit,
>18	leshort		75		Digital VAX,
>18	leshort		76		Axis cris,
>18	leshort		77		Infineon 32-bit embedded,
>18	leshort		78		Element 14 64-bit DSP,
>18	leshort		79		LSI Logic 16-bit DSP,
>18	leshort		80		MMIX,
>18	leshort		81		Harvard machine-independent,
>18	leshort		82		SiTera Prism,
>18	leshort		83		Atmel AVR 8-bit,
>18	leshort		84		Fujitsu FR30,
>18	leshort		85		Mitsubishi D10V,
>18	leshort		86		Mitsubishi D30V,
>18	leshort		87		NEC v850,
>18	leshort		88		Renesas M32R,
>18	leshort		89		Matsushita MN10300,
>18	leshort		90		Matsushita MN10200,
>18	leshort		91		picoJava,
>18	leshort		92		OpenRISC,
>18	leshort		93		Synopsys ARCompact ARC700 cores,
>18	leshort		94		Tensilica Xtensa,
>18	leshort		95		Alphamosaic VideoCore,
>18	leshort		96		Thompson Multimedia,
>18	leshort		97		NatSemi 32k,
>18	leshort		98		Tenor Network TPC,
>18	leshort		99		Trebia SNP 1000,
>18	leshort		100		STMicroelectronics ST200,
>18	leshort		101		Ubicom IP2022,
>18	leshort		102		MAX Processor,
>18	leshort		103		NatSemi CompactRISC,
>18	leshort		104		Fujitsu F2MC16,
>18	leshort		105		TI msp430,
>18	leshort		106		Analog Devices Blackfin,
>18	leshort		107		S1C33 Family of Seiko Epson,
>18	leshort		108		Sharp embedded,
>18	leshort		109		Arca RISC,
>18	leshort		110		PKU-Unity Ltd.,
>18	leshort		111		eXcess: 16/32/64-bit,
>18	leshort		112		Icera Deep Execution Processor,
>18	leshort		113		Altera Nios II,
>18	leshort		114		NatSemi CRX,
>18	leshort		115		Motorola XGATE,
>18	leshort		116		Infineon C16x/XC16x,
>18	leshort		117		Renesas M16C series,
>18	leshort		118		Microchip dsPIC30F,
>18	leshort		119		Freescale RISC core,
>18	leshort		120		Renesas M32C series,
>18	leshort		131		Altium TSK3000 core,
>18	leshort		132		Freescale RS08,
>18	leshort		134		Cyan Technology eCOG2,
>18	leshort		135		Sunplus S+core7 RISC,
>18	leshort		136		New Japan Radio (NJR) 24-bit DSP,
>18	leshort		137		Broadcom VideoCore III,
>18	leshort		138		LatticeMico32,
>18	leshort		139		Seiko Epson C17 family,
>18	leshort		140		TI TMS320C6000 DSP family,
>18	leshort		141		TI TMS320C2000 DSP family,
>18	leshort		142		TI TMS320C55x DSP family,
>18	leshort		144		TI Programmable Realtime Unit
>18	leshort		160		STMicroelectronics 64bit VLIW DSP,
>18	leshort		161		Cypress M8C,
>18	leshort		162		Renesas R32C series,
>18	leshort		163		NXP TriMedia family,
>18	leshort		164		QUALCOMM DSP6,
>18	leshort		165		Intel 8051 and variants,
>18	leshort		166		STMicroelectronics STxP7x family,
>18	leshort		167		Andes embedded RISC,
>18	leshort		168		Cyan eCOG1X family,
>18	leshort		169		Dallas MAXQ30,
>18	leshort		170		New Japan Radio (NJR) 16-bit DSP,
>18	leshort		171		M2000 Reconfigurable RISC,
>18	leshort		172		Cray NV2 vector architecture,
>18	leshort		173		Renesas RX family,
>18	leshort		174		META,
>18	leshort		175		MCST Elbrus,
>18	leshort		176		Cyan Technology eCOG16 family,
>18	leshort		177		NatSemi CompactRISC,
>18	leshort		178		Freescale Extended Time Processing Unit,
>18	leshort		179		Infineon SLE9X,
>18	leshort		180		Intel L1OM,
>18	leshort		181		Intel K1OM,
>18	leshort		183		ARM aarch64,
>18	leshort		185		Atmel 32-bit family,
>18	leshort		186		STMicroeletronics STM8 8-bit,
>18	leshort		187		Tilera TILE64,
>18	leshort		188		Tilera TILEPro,
>18	leshort		189		Xilinx MicroBlaze 32-bit RISC,
>18	leshort		190		NVIDIA CUDA architecture,
>18	leshort		191		Tilera TILE-Gx,
>18	leshort		195		Synopsys ARCv2/HS3x/HS4x cores,
>18	leshort		197		Renesas RL78 family,
>18	leshort		199		Renesas 78K0R,
>18	leshort		200		Freescale 56800EX,
>18	leshort		201		Beyond BA1,
>18	leshort		202		Beyond BA2,
>18	leshort		203		XMOS xCORE,
>18	leshort		204		Microchip 8-bit PIC(r),
>18	leshort		210		KM211 KM32,
>18	leshort		211		KM211 KMX32,
>18	leshort		212		KM211 KMX16,
>18	leshort		213		KM211 KMX8,
>18	leshort		214		KM211 KVARC,
>18	leshort		215		Paneve CDP,
>18	leshort		216		Cognitive Smart Memory,
>18	leshort		217		iCelero CoolEngine,
>18	leshort		218		Nanoradio Optimized RISC,
>18	leshort		219		CSR Kalimba architecture family
>18	leshort		220		Zilog Z80
>18	leshort		221		Controls and Data Services VISIUMcore processor
>18	leshort		222		FTDI Chip FT32 high performance 32-bit RISC architecture
>18	leshort		223		Moxie processor family
>18	leshort		224		AMD GPU architecture
>18	leshort		243		UCB RISC-V,
# only for 32-bit
>>4	byte		1
>>>36	use		elf-riscv
# only for 64-bit
>>4	byte		2
>>>48	use		elf-riscv
>18	leshort		244		Lanai 32-bit processor,
>18	leshort		245		CEVA Processor Architecture Family,
>18	leshort		246		CEVA X2 Processor Family,
>18	leshort		247		eBPF,
>18	leshort		248		Graphcore Intelligent Processing Unit,
>18	leshort		249		Imagination Technologies,
>18	leshort		250		Netronome Flow Processor,
>18	leshort		251             NEC Vector Engine,
>18	leshort		252		C-SKY processor family,
>18	leshort		253		Synopsys ARCv3 64-bit ISA/HS6x cores,
>18	leshort		254		MOS Technology MCS 6502 processor,
>18	leshort		255		Synopsys ARCv3 32-bit,
>18	leshort		256		Kalray VLIW core of the MPPA family,
>18	leshort		257		WDC 65816/65C816,
>18	leshort		258		LoongArch,
>18	leshort		259		ChipON KungFu32,
>18	leshort		0x1057		AVR (unofficial),
>18	leshort		0x1059		MSP430 (unofficial),
>18	leshort		0x1223		Adapteva Epiphany (unofficial),
>18	leshort		0x2530		Morpho MT (unofficial),
>18	leshort		0x3330		FR30 (unofficial),
>18	leshort		0x3426		OpenRISC (obsolete),
>18	leshort		0x4688		Infineon C166 (unofficial),
>18	leshort		0x5441		Cygnus FRV (unofficial),
>18	leshort		0x5aa5		DLX (unofficial),
>18	leshort		0x7650		Cygnus D10V (unofficial),
>18	leshort		0x7676		Cygnus D30V (unofficial),
>18	leshort		0x8217		Ubicom IP2xxx (unofficial),
>18	leshort		0x8472		OpenRISC (obsolete),
>18	leshort		0x9025		Cygnus PowerPC (unofficial),
>18	leshort		0x9026		Alpha (unofficial),
>18	leshort		0x9041		Cygnus M32R (unofficial),
>18	leshort		0x9080		Cygnus V850 (unofficial),
>18	leshort		0xa390		IBM S/390 (obsolete),
>18	leshort		0xabc7		Old Xtensa (unofficial),
>18	leshort		0xad45		xstormy16 (unofficial),
>18	leshort		0xbaab		Old MicroBlaze (unofficial),,
>18	leshort		0xbeef		Cygnus MN10300 (unofficial),
>18	leshort		0xdead		Cygnus MN10200 (unofficial),
>18	leshort		0xf00d		Toshiba MeP (unofficial),
>18	leshort		0xfeb0		Renesas M32C (unofficial),
>18	leshort		0xfeba		Vitesse IQ2000 (unofficial),
>18	leshort		0xfebb		NIOS (unofficial),
>18	leshort		0xfeed		Moxie (unofficial),
>18	default		x
>>18	leshort		x		*unknown arch %#x*
>20	lelong		0		invalid version
>20	lelong		1		version 1

0	string		\177ELF		ELF
!:strength *2
>4	byte		0		invalid class
>4	byte		1		32-bit
>4	byte		2		64-bit
>5	byte		0		invalid byte order
>5	byte		1		LSB
>>0	use		elf-le
>5	byte		2		MSB
>>0	use		\^elf-le
>7	byte		0		(SYSV)
>7	byte		1		(HP-UX)
>7	byte		2		(NetBSD)
>7	byte		3		(GNU/Linux)
>7	byte		4		(GNU/Hurd)
>7	byte		5		(86Open)
>7	byte		6		(Solaris)
>7	byte		7		(Monterey)
>7	byte		8		(IRIX)
>7	byte		9		(FreeBSD)
>7	byte		10		(Tru64)
>7	byte		11		(Novell Modesto)
>7	byte		12		(OpenBSD)
>7	byte		13		(OpenVMS)
>7	byte		14		(HP NonStop Kernel)
>7	byte		15		(AROS Research Operating System)
>7	byte		16		(FenixOS)
>7	byte		17		(Nuxi CloudABI)
>7	byte		97		(ARM)
>7	byte		102		(Cell LV2)
>7	byte		202		(Cafe OS)
>7	byte		255		(embedded)

# SELF Signed ELF used on the playstation
# https://www.psdevwiki.com/ps4/SELF_File_Format#make_fself_by_flatz
# https://www.psdevwiki.com/ps3/SELF_-_SPRX
0	lelong		0x4F153D1D
>4	lelong		0x00010112	PS4 Signed ELF file
>8	byte		1		\b, SELF/SPRX signed-elf/prx
>8	byte		2		\b, SRVK signed-revoke-list
>8	byte		3		\b, SPKG signed-package
>8	byte		4		\b, SSPP signed-security-policy-profile
>8	byte		5		\b, SDIFF signed-diff
>8	byte		6		\b, SPSFO signed-param-sfo
>9	byte&0xf0	x		\b, version %#x	
>9	byte&0x0f	4		\b, game
>9	byte&0x0f	5		\b, module
>9	byte&0x0f	6		\b, video app
>9	byte&0x0f	8		\b, System/EX application
>9	byte&0x0f	9		\b, System/EX module/dll
#>&-9	byte&0x0f	9		\b, System/EX module/dll
#>12	leshort		x		\b, header size %d
#>14	leshort		x		\b, signature size %d
#>16	lelong		x		\b, file size %d
#>18	leshort		x		\b, number of segments %d
#>20	leshort		22
        "#;
        let me = parse_rule(&file);
        let mut cursor = Cursor::new(r#"\177ELF\x00"#);
        println!("magic: {}", me.magic(&mut cursor).message());
        println!("finished")
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
    fn test_elf_match() {
        let me = FileMagicParser::parse_file("./magdir/elf").unwrap();
        let realpath = fs::canonicalize("/proc/self/exe").unwrap();
        let mut rs = File::open(&realpath).unwrap();
        let mazic = me.magic(&mut rs);
        println!("magic: {}", mazic.message());
        println!("mimetype: {}", mazic.mimetype());
        println!(
            "magic file command: {}",
            String::from_utf8(
                Command::new("file")
                    .arg(realpath.to_string_lossy().to_string())
                    .output()
                    .unwrap()
                    .stdout
            )
            .unwrap()
        );
        println!(
            "magic file command: {}",
            String::from_utf8(
                Command::new("file")
                    .arg("--mime-type")
                    .arg(realpath.to_string_lossy().to_string())
                    .output()
                    .unwrap()
                    .stdout
            )
            .unwrap()
        );
        println!("finished")
    }

    #[test]
    fn test_rust_match() {
        let test_file = "./target/debug/deps/libpest_meta-9f140d66dcb06b7c.rmeta";
        let me = FileMagicParser::parse_file("./magdir/rust").unwrap();
        let realpath = fs::canonicalize(test_file).unwrap();
        let mut rs = File::open(&realpath).unwrap();
        let mazic = me.magic(&mut rs);
        println!("magic: {}", mazic.message());
        println!("mimetype: {}", mazic.mimetype());
        println!(
            "magic file command: {}",
            String::from_utf8(
                Command::new("file")
                    .arg(realpath.to_string_lossy().to_string())
                    .output()
                    .unwrap()
                    .stdout
            )
            .unwrap()
        );
        println!(
            "magic file command: {}",
            String::from_utf8(
                Command::new("file")
                    .arg("--mime-type")
                    .arg(realpath.to_string_lossy().to_string())
                    .output()
                    .unwrap()
                    .stdout
            )
            .unwrap()
        );
        println!("finished")
    }

    #[test]
    fn parse_regex() {
        let me = FileMagicParser::parse_str(
            r#"
0	regex/1024 #![[:space:]]*/usr/bin/env[[:space:]]+
!:mime	text/x-shellscript
>&0  regex/64 .*($|\\b) %s shell script text executable
    "#,
        )
        .inspect_err(|e| println!("{e}"))
        .unwrap();
        let mut bash = Cursor::new(
            r#"#!/usr/bin/env bash
        echo hello world"#,
        );
        let mazic = me.magic(&mut bash);
        println!("magic: {}", mazic.message());
        println!("mimetype: {}", mazic.mimetype());
    }

    #[test]
    fn test_string_with_mods() {
        let me = FileMagicParser::parse_str(
            r#"0	string/fwt	#!\ \ \ /usr/bin/env\ bash	Bourne-Again shell script text executable
!:mime	text/x-shellscript
"#,
        )
        .inspect_err(|e| println!("{e}"))
        .unwrap();
        let mut bash = Cursor::new(
            r#"#!/usr/bin/env bash i 
        echo hello world"#,
        );
        let mazic = me.magic(&mut bash);
        println!("magic: {}", mazic.message());
        println!("mimetype: {}", mazic.mimetype());
    }

    #[test]
    fn test_search_with_mods() {
        let me = FileMagicParser::parse_str(
            r#"0	search/1/fwt	#!\ /usr/bin/luatex	LuaTex script text executable
!:mime	text/x-luatex
"#,
        )
        .inspect_err(|e| println!("{e}"))
        .unwrap();
        let mut bash = Cursor::new(r#"#!          /usr/bin/luatex "#);
        let mazic = me.magic(&mut bash);
        println!("magic: {}", mazic.message());
        println!("mimetype: {}", mazic.mimetype());
    }
}
