use std::{collections::HashMap, fs, path::Path};

use dyf::FormatString;
use flagset::FlagSet;
use pest::{Parser, iterators::Pair};
use pest_derive::Parser;
use regex::bytes;
use uuid::Uuid;

use crate::{
    Any, CmpOp, DependencyRule, DirOffset, Entry, EntryNode, Error, Flag, FloatTest,
    FloatTransform, IndOffset, IndirectMod, IndirectMods, MagicFile, MagicRule, Match, Message,
    Name, Offset, OffsetType, Op, PStringLen, PStringTest, ReMod, RegexTest, ScalarTest,
    ScalarTransform, SearchTest, Shift, StrengthMod, String16Encoding, String16Test, StringMod,
    StringTest, Test, Use,
    numeric::{FloatDataType, Scalar, ScalarDataType},
};

pub(crate) fn prepare_bytes_re(s: &[u8], escape: bool) -> String {
    let mut out = String::new();
    for b in s {
        if b.is_ascii() {
            if escape {
                out.push_str(&regex::escape(&format!("{}", *b as char)));
            } else {
                out.push(*b as char);
            }
        } else {
            out.push_str(&format!("\\x{b:02x}"));
        }
    }
    out
}

pub(crate) fn unescape_string_to_string(s: &str) -> String {
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
                                hex_str.push(chars.next().unwrap() as char);
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
                                octal_str.push(chars.next().unwrap() as char);
                                continue;
                            }
                            break;
                        }
                        //let octal_str: String = chars.by_ref().take(1).collect();
                        if let Ok(octal) = u8::from_str_radix(&octal_str, 8) {
                            result.push(octal as char);
                        } else {
                            result.push(c as char); // Push the backslash if the octal sequence is invalid
                        }
                    }
                    _ => {
                        // we skip the backslash
                    }
                }
            } else {
                result.push(c as char); // Push the backslash if no character follows
            }
        } else {
            result.push(c);
        }
    }

    result
}

pub(crate) fn unescape_string_to_vec(s: &str) -> (bool, Vec<u8>) {
    let mut result = Vec::new();
    let mut chars = s.bytes().peekable();
    // this flags wether we replaced some binary values encoded in the
    // pattern. It seems libmagic doesn't care if the encoded value is
    // actually a valid ASCII character.
    let mut binary = false;
    while let Some(c) = chars.next() {
        // string termination
        if c == b'\0' {
            return (binary, result);
        }

        if c == b'\\' {
            if let Some(next_char) = chars.peek() {
                match next_char {
                    // string termination
                    b'n' => {
                        result.push(b'\n');
                        chars.next(); // Skip the 'n'
                    }
                    b't' => {
                        result.push(b'\t');
                        chars.next(); // Skip the 't'
                    }
                    b'r' => {
                        result.push(b'\r');
                        chars.next(); // Skip the 'r'
                    }
                    b'\\' => {
                        result.push(b'\\');
                        chars.next(); // Skip the '\\'
                    }
                    b'x' => {
                        // Handle hex escape sequences (e.g., \x7F)
                        chars.next(); // Skip the 'x'

                        let mut hex_str = String::new();
                        for _ in 0..2 {
                            if chars
                                .peek()
                                .map(|c| c.is_ascii_hexdigit())
                                .unwrap_or_default()
                            {
                                hex_str.push(chars.next().unwrap() as char);
                                continue;
                            }
                            break;
                        }

                        if let Ok(hex) = u8::from_str_radix(&hex_str, 16) {
                            binary = true;
                            result.push(hex);
                        } else {
                            result.push(c as u8); // Push the backslash if the hex sequence is invalid
                        }
                    }
                    // Handle octal escape sequences (e.g., \1 \23 \177)
                    b'0'..=b'7' => {
                        let mut octal_str = String::new();
                        for _ in 0..3 {
                            if chars
                                .peek()
                                .map(|c| matches!(c, b'0'..=b'7'))
                                .unwrap_or_default()
                            {
                                octal_str.push(chars.next().unwrap() as char);
                                continue;
                            }
                            break;
                        }
                        //let octal_str: String = chars.by_ref().take(1).collect();
                        if let Ok(octal) = u8::from_str_radix(&octal_str, 8) {
                            binary = true;
                            result.push(octal);
                        } else {
                            result.push(c as u8); // Push the backslash if the octal sequence is invalid
                        }
                    }
                    _ => {
                        // we skip the backslash
                    }
                }
            } else {
                result.push(c as u8); // Push the backslash if no character follows
            }
        } else {
            result.push(c);
        }
    }

    (binary, result)
}

#[inline]
fn parse_pos_number(pair: Pair<'_, Rule>) -> i64 {
    let number_token = pair.into_inner().next().expect("expect number kind pair");
    match number_token.as_rule() {
        Rule::b10_number => number_token.as_str().parse::<i64>().unwrap(),
        Rule::b16_number => u64::from_str_radix(
            number_token
                .as_str()
                .to_lowercase()
                .strip_prefix("0x")
                .unwrap(),
            16,
        )
        .unwrap() as i64,
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

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub(crate) struct FileMagicParser;

impl CmpOp {
    fn from_pair(value: Pair<'_, Rule>) -> Result<Self, Error> {
        match value.as_rule() {
            Rule::op_lt => Ok(Self::Lt),
            Rule::op_gt => Ok(Self::Gt),
            Rule::op_and => Ok(Self::BitAnd),
            Rule::op_neq => Ok(Self::Neq),
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
            Rule::op_or => Ok(Self::Or),
            Rule::op_xor => Ok(Self::Xor),
            _ => Err(Error::parser("unimplemented operator", value.as_span())),
        }
    }
}

impl ScalarDataType {
    fn from_pair(pair: Pair<'_, Rule>) -> Result<Self, Error> {
        let dt = pair.into_inner().next().expect("data type expected");

        macro_rules! handle_types {
            ($($ty: ident),*) => {
                match dt.as_rule() {
                    $(
                        Rule::$ty => Ok(Self::$ty),
                    )*
                    _ => Err(Error::parser("unimplemented data type", dt.as_span())),
                }
            };
        }

        handle_types!(
            belong,
            bequad,
            beshort,
            bedate,
            beldate,
            beqdate,
            byte,
            date,
            ldate,
            quad,
            qwdate,
            uquad,
            lelong,
            ledate,
            leldate,
            leqdate,
            leqldate,
            leshort,
            long,
            short,
            ushort,
            ulong,
            ubelong,
            ubequad,
            ubeshort,
            ubyte,
            ulelong,
            ulequad,
            uleshort,
            ubeqdate,
            lequad,
            uledate,
            offset,
            lemsdosdate,
            lemsdostime,
            leqwdate,
            medate,
            meldate,
            melong
        )
    }
}

impl FloatDataType {
    fn from_pair(pair: Pair<'_, Rule>) -> Result<Self, Error> {
        let dt = pair.into_inner().next().expect("data type expected");

        macro_rules! handle_types {
            ($($ty: ident),*) => {
                match dt.as_rule() {
                    $(
                        Rule::$ty => Ok(Self::$ty),
                    )*
                    _ => Err(Error::parser("unimplemented data type", dt.as_span())),
                }
            };
        }

        handle_types!(bedouble, ledouble, befloat, lefloat)
    }
}

impl FileMagicParser {
    pub(crate) fn parse_str<S: AsRef<str>>(
        s: S,
        source: Option<String>,
    ) -> Result<MagicFile, Error> {
        let pairs = FileMagicParser::parse(Rule::file, s.as_ref()).map_err(Box::new)?;

        let mut rules = vec![];
        let mut dependencies = HashMap::new();
        for file in pairs {
            for rule in file.into_inner() {
                match rule.as_rule() {
                    Rule::rule => {
                        rules.push(MagicRule::from_pair(rule, source.clone())?);
                    }
                    Rule::rule_dependency => {
                        let d = DependencyRule::from_pair(rule, source.clone())?;
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

    pub(crate) fn parse_file<P: AsRef<Path>>(p: P) -> Result<MagicFile, Error> {
        let s = fs::read_to_string(&p)?;
        Self::parse_str(
            s,
            p.as_ref()
                .file_name()
                .map(|os| os.to_string_lossy().to_string()),
        )
    }
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
                assert_eq!(creatype.as_rule(), Rule::apple_ty);
                Ok(Self::Apple(creatype.as_str().to_string()))
            }

            // parser should guarantee this branch is never reached
            _ => unimplemented!(),
        }
    }
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

        assert_eq!(
            offset.as_rule(),
            Rule::stream_offset,
            "expected offset rule"
        );
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

impl StringTest {
    fn from_pair_with_slice(
        pair: Pair<'_, Rule>,
        str: &[u8],
        binary: bool,
        cmp_op: CmpOp,
    ) -> Result<Self, Error> {
        let mut length = None;
        let mut mods = FlagSet::empty();
        for p in pair.into_inner() {
            match p.as_rule() {
                Rule::pos_number => length = Some(parse_pos_number(p) as usize),
                Rule::string_mod => mods |= StringMod::from_pair(p)?,
                // parser should guarantee this branch is never reached
                _ => unimplemented!(),
            }
        }
        Ok(Self {
            str: str.to_vec(),
            cmp_op,
            length,
            mods,
            binary,
        })
    }
}

impl PStringTest {
    fn from_pair_with_value(pair: Pair<'_, Rule>, value: Vec<u8>) -> Self {
        debug_assert_eq!(pair.as_rule(), Rule::pstring);

        let mut len = PStringLen::Byte;
        let mut include_len = false;

        for r in pair.into_inner() {
            match r.as_rule() {
                Rule::pstring_mod => match r.as_str() {
                    "B" => len = PStringLen::Byte,
                    "H" => len = PStringLen::ShortBe,
                    "h" => len = PStringLen::ShortLe,
                    "L" => len = PStringLen::LongBe,
                    "l" => len = PStringLen::LongLe,
                    "J" => include_len = true,
                    // parser should guarantee this branch is never reached
                    _ => unimplemented!(),
                },
                // parser should guarantee this branch is never reached
                _ => unimplemented!(),
            }
        }

        PStringTest {
            val: value,
            len,
            include_len,
        }
    }
}

impl SearchTest {
    fn from_pair_with_slice(pair: Pair<'_, Rule>, str: &[u8], binary: bool) -> Self {
        let mut length = None;
        let mut str_mods: FlagSet<StringMod> = FlagSet::empty();
        let mut re_mods: FlagSet<ReMod> = FlagSet::empty();
        for p in pair.into_inner() {
            match p.as_rule() {
                Rule::pos_number => length = Some(parse_pos_number(p) as usize),
                Rule::search_mod => {
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
                            's' => re_mods |= ReMod::StartOffsetUpdate,
                            _ => {}
                        }
                    }
                }
                // parser should guarantee this branch is never reached
                _ => unimplemented!(),
            }
        }
        Self {
            str: str.to_vec(),
            n_pos: length,
            str_mods,
            re_mods,
            binary,
        }
    }
}

impl RegexTest {
    fn from_pair_with_re(pair: Pair<'_, Rule>, re: &str, binary: bool) -> Result<Self, Error> {
        let mut length = None;
        let mut mods = FlagSet::empty();
        let str_mods = FlagSet::empty();
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
                            'b' => mods |= ReMod::ForceBinary,
                            't' => mods |= ReMod::ForceText,
                            'T' => mods |= ReMod::TrimMatch,
                            _ => {}
                        }
                    }
                }
                // parser should guarantee this branch is never reached
                _ => unimplemented!(),
            }
        }
        let re = format!("(?-u){re}");

        Ok(Self {
            //FIXME: remove unwrap
            re: bytes::Regex::new(&re).unwrap(),
            length,
            mods,
            str_mods,
            binary,
        })
    }
}

impl Test {
    fn from_pair(pair: Pair<'_, Rule>) -> Result<Self, Error> {
        let t = match pair.as_rule() {
            Rule::scalar_test => {
                let pairs = pair.into_inner();

                let mut ty = None;
                let mut transform = None;
                let mut cmp_op = CmpOp::Eq;
                let mut scalar = None;
                for pair in pairs {
                    match pair.as_rule() {
                        Rule::scalar_type_transform => {
                            for pair in pair.into_inner() {
                                match pair.as_rule() {
                                    Rule::scalar_type => {
                                        ty = Some(ScalarDataType::from_pair(pair)?);
                                    }

                                    Rule::scalar_transform => {
                                        let mut transform_pairs = pair.into_inner();
                                        let op_pair =
                                            transform_pairs.next().expect("expect operator pair");
                                        let op = Op::from_pair(op_pair)?;

                                        let number =
                                            transform_pairs.next().expect("expect number pair");
                                        let span = number.as_span();
                                        // ty is guaranteed to be some by
                                        // parser implementation
                                        let num = ty
                                            .unwrap()
                                            .scalar_from_number(parse_pos_number(number));

                                        // we check if we try to divide by zero
                                        if num.is_zero() {
                                            match op {
                                                Op::Div | Op::Mod => {
                                                    return Err(Error::parser(
                                                        "divide by zero error",
                                                        span,
                                                    ));
                                                }
                                                _ => {}
                                            }
                                        }

                                        transform = Some(ScalarTransform { op, num });
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Rule::scalar_condition => {
                            cmp_op = CmpOp::from_pair(
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

                let mut scalar = scalar.expect("scalar must be known");

                // we handle Not (~) operator only once
                if matches!(cmp_op, CmpOp::Not) {
                    scalar = !scalar;
                    cmp_op = CmpOp::Eq;
                }

                Self::Scalar(ScalarTest {
                    // no panic guarantee by parser
                    ty: ty.unwrap(),
                    transform,
                    cmp_op,
                    // no panic guarantee by parser
                    value: scalar,
                })
            }
            Rule::search_test => {
                let mut search_test = pair.into_inner();

                let test_type = search_test.next().expect("expecting a string type");

                let test_value = search_test.next().expect("expecting a string value");
                assert_eq!(test_value.as_rule(), Rule::string_value);

                match test_type.as_rule() {
                    Rule::search => {
                        let (bin, v) = unescape_string_to_vec(&test_value.as_str());
                        SearchTest::from_pair_with_slice(test_type, &v, bin).into()
                    }

                    Rule::regex => {
                        let (bin, s) = unescape_string_to_vec(test_value.as_str());
                        RegexTest::from_pair_with_re(test_type, &prepare_bytes_re(&s, false), bin)?
                            .into()
                    }
                    // parser should guarantee this branch is never reached
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
                        Rule::string => {
                            let (bin, s) = unescape_string_to_vec(test_value.as_str());
                            StringTest::from_pair_with_slice(
                                test_type,
                                &s,
                                bin,
                                cmp_op.unwrap_or(CmpOp::Eq),
                            )?
                            .into()
                        }

                        Rule::pstring => {
                            let val = unescape_string_to_vec(test_value.as_str()).1;

                            Self::PString(PStringTest::from_pair_with_value(test_type, val))
                        }

                        // parser should guarantee this branch is never reached
                        _ => unimplemented!(),
                    },

                    Rule::any_value => match test_type.as_rule() {
                        Rule::string => Self::Any(Any::String),
                        Rule::pstring => Self::Any(Any::PString(
                            PStringTest::from_pair_with_value(test_type, vec![]),
                        )),
                        // parser should guarantee this branch is never reached
                        _ => unimplemented!(),
                    },
                    // parser should guarantee this branch is never reached
                    _ => unimplemented!(),
                }
            }
            Rule::string16_test => {
                let mut encoding = None;
                let mut orig = None;
                let mut str = None;
                for p in pair.into_inner() {
                    match p.as_rule() {
                        Rule::lestring16 => encoding = Some(String16Encoding::Le),
                        Rule::bestring16 => encoding = Some(String16Encoding::Be),
                        Rule::string_value => {
                            orig = Some(unescape_string_to_string(p.as_str()));
                            str = Some(
                                unescape_string_to_vec(p.as_str())
                                    .1
                                    .iter()
                                    .map(|b| *b as u16)
                                    .collect(),
                            )
                        }
                        Rule::any_value => {
                            return Ok(Self::Any(Any::String16(
                                encoding.expect("encoding must be known"),
                            )));
                        }
                        _ => {}
                    }
                }
                Self::String16(String16Test {
                    orig: orig.expect("test value must be known"),
                    str16: str.expect("test value must be known"),
                    encoding: encoding.expect("encoding must be known"),
                })
            }
            Rule::guid_test => {
                let mut guid = None;
                for p in pair.into_inner() {
                    match p.as_rule() {
                        Rule::any_value => {
                            return Ok(Self::Any(Any::Scalar(ScalarDataType::guid)));
                        }
                        Rule::guid => {
                            guid = Some(
                                Uuid::parse_str(p.as_str())
                                    .expect("valid uuid is guaranteed by grammar"),
                            )
                        }
                        _ => {}
                    }
                }

                Self::Scalar(ScalarTest {
                    ty: ScalarDataType::guid,
                    transform: None,
                    cmp_op: CmpOp::Eq,
                    value: Scalar::guid(guid.unwrap().as_u128()),
                })
            }

            Rule::float_test => {
                let mut float = None;
                let mut ty = None;
                let mut cmp_op = CmpOp::Eq;
                let mut transform = None;
                for p in pair.into_inner() {
                    let span = p.as_span();
                    match p.as_rule() {
                        Rule::any_value => {
                            return Ok(Self::Any(Any::Float(ty.expect("type must be known"))));
                        }
                        Rule::float_type_transform => {
                            for p in p.into_inner() {
                                match p.as_rule() {
                                    Rule::float_type => ty = Some(FloatDataType::from_pair(p)?),
                                    Rule::float_transform => {
                                        let mut pairs = p.into_inner();
                                        let op = Op::from_pair(pairs.next().unwrap())?;

                                        let float_pair = pairs.next().unwrap();
                                        let value: f64 =
                                            float_pair.as_str().parse().map_err(|_| {
                                                Error::parser(
                                                    "cannot parse str to float",
                                                    float_pair.as_span(),
                                                )
                                            })?;

                                        let ty = ty.expect("type must be known");

                                        let num = ty.float_from_f64(value);
                                        transform = Some(FloatTransform { op, num })
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Rule::float_number => {
                            let f: f64 = p
                                .as_str()
                                .parse()
                                .map_err(|_| Error::parser("cannot parse str to float", span))?;

                            let ty = ty.expect("type must be known");
                            float = Some(ty.float_from_f64(f));
                        }
                        Rule::float_condition => {
                            cmp_op = CmpOp::from_pair(p.into_inner().next().unwrap())?
                        }
                        _ => {}
                    }
                }

                Self::Float(FloatTest {
                    value: float.expect("float value must be known"),
                    ty: ty.expect("type must be known"),
                    cmp_op,
                    transform,
                })
            }

            Rule::clear_test => Self::Clear,
            Rule::default_test => Self::Default,
            Rule::indirect_test => {
                let mut ind_mods = IndirectMods::empty();
                for p in pair.into_inner() {
                    match p.as_rule() {
                        Rule::indirect => {
                            for p in p.into_inner() {
                                match p.as_rule() {
                                    Rule::indirect_mod => match p.as_str() {
                                        "r" => ind_mods |= IndirectMod::Relative,
                                        _ => {
                                            return Err(Error::parser(
                                                "unsupported modifier",
                                                p.as_span(),
                                            ));
                                        }
                                    },
                                    _ => {
                                        return Err(Error::parser(
                                            "parsing rule not handled",
                                            p.as_span(),
                                        ));
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Self::Indirect(ind_mods)
            }

            // parser should guarantee this branch is never reached
            _ => unimplemented!(),
        };

        Ok(t)
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

        assert_eq!(
            offset.as_rule(),
            Rule::stream_offset,
            "expected offset rule"
        );
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
}

impl DependencyRule {
    fn from_pair(pair: Pair<'_, Rule>, source: Option<String>) -> Result<Self, Error> {
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
            rule: MagicRule::from_pair(pair, source)?,
        })
    }
}

impl MagicRule {
    pub(crate) fn from_pair(pair: Pair<'_, Rule>, source: Option<String>) -> Result<Self, Error> {
        let mut items = vec![];
        let span = pair.as_span();

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
            source,
            entries: EntryNode::from_entries(items)
                .map_err(|e| Error::parser(e.to_string(), span))?,
        })
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_unescape() {
        assert_eq!(
            str::from_utf8(&unescape_string_to_vec(r#"\ hello"#).1).unwrap(),
            " hello"
        );
        assert_eq!(unescape_string_to_vec(r#"hello\ world"#).1, b"hello world");
        assert_eq!(
            unescape_string_to_vec(r#"\^[[:space:]]"#).1,
            b"^[[:space:]]"
        );
        assert_eq!(unescape_string_to_vec(r#"\x41"#).1, b"A");
        assert_eq!(unescape_string_to_vec(r#"\xF\xA"#).1, b"\x0f\n");
        assert_eq!(unescape_string_to_vec(r#"\101"#).1, b"A");
        println!(
            "{:?}",
            unescape_string_to_vec(r#"\1\0\0\0\0\0\0\300\0\2\0\0"#)
        );
    }
}
