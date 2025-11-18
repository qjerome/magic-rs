use std::fmt;

use dyf::DynDisplay;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::utils::{
    parse_fat_date, parse_fat_time, unix_local_time_to_string, unix_utc_time_to_string,
    windows_filetime_to_string,
};

macro_rules! impl_numeric_types {
    ($($name: tt($ty: ty)),* $(,)?) => {
        #[allow(non_camel_case_types)]
        #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Serialize, Deserialize)]

        pub(crate) enum Scalar {
            $($name($ty),)*
        }

        impl Scalar {
            pub(crate) fn is_zero(&self) -> bool {
                match self {
                    $(Self::$name(x) => *x == 0,)*
                }
            }

            pub(crate) fn checked_add(&self, other: Self) -> Option<Self> {
                match (self, other) {
                    $(
                        (Self::$name(a), Self::$name(b)) => Some(Self::$name(a.checked_add(b)?)),
                    )*
                    _=> panic!("operation not supported between different numeric variants")
                }
            }

            pub(crate) fn checked_sub(&self, other: Self) -> Option<Self> {
                match (self, other) {
                    $(
                        (Self::$name(a), Self::$name(b)) => Some(Self::$name(a.checked_sub(b)?)),
                    )*
                    _=> panic!("operation not supported between different numeric variants")
                }
            }

            pub(crate) fn checked_mul(&self, other: Self) -> Option<Self> {
                match (self, other) {
                    $(
                        (Self::$name(a), Self::$name(b)) => Some(Self::$name(a.checked_mul(b)?)),
                    )*
                    _=> panic!("operation not supported between different numeric variants")
                }
            }

            pub(crate) fn checked_div(&self, other: Self) -> Option<Self> {
                match (self, other) {
                    $(
                        (Self::$name(a), Self::$name(b)) => Some(Self::$name(a.checked_div(b)?)),
                    )*
                    _=> panic!("operation not supported between different numeric variants")
                }
            }

            pub(crate) fn checked_rem(&self, other: Self) -> Option<Self> {
                match (self, other) {
                    $(
                        (Self::$name(a), Self::$name(b)) => Some(Self::$name(a.checked_rem(b)?)),
                    )*
                    _=> panic!("operation not supported between different numeric variants")
                }
            }

            pub(crate) fn size_of(&self) -> usize{
                match self {
                    $(Self::$name(_) => core::mem::size_of::<$ty>(),)*
                }
            }
        }

        impl std::ops::Not for Scalar {
            type Output = Self;
            fn not(self) -> Self::Output {
                match self {
                    $(
                        Self::$name(value) => Self::$name(!value),
                    )*
                }
            }
        }


        impl std::ops::BitAnd for Scalar {
            type Output = Self;

            fn bitand(self, other: Self) -> Self::Output {
                match (self, other) {
                    $(
                        (Self::$name(a), Self::$name(b)) => Self::$name(a.bitand(b)),
                    )*
                    _=> panic!("operation not supported between different numeric variants")
                }
            }
        }

        impl std::ops::BitOr for Scalar {
            type Output = Self;

            fn bitor(self, other: Self) -> Self::Output {
                match (self, other) {
                    $(
                        (Self::$name(a), Self::$name(b)) => Self::$name(a.bitor(b)),
                    )*
                    _=> panic!("operation not supported between different numeric variants")
                }
            }
        }

        impl std::ops::BitXor for Scalar {
            type Output = Self;

            fn bitxor(self, other: Self) -> Self::Output {
                match (self, other) {
                    $(
                        (Self::$name(a), Self::$name(b)) => Self::$name(a.bitxor(b)),
                    )*
                    _=> panic!("operation not supported between different numeric variants")
                }
            }
        }



        #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
        #[allow(non_camel_case_types)]
        pub(crate) enum ScalarDataType {
            $($name,)*
        }

        impl ScalarDataType {
            pub(crate) const fn type_size(&self) -> usize {
                match self {
                    $(Self::$name => core::mem::size_of::<$ty>(),)*
                }
            }

            pub(crate) fn scalar_from_number(&self, i: i64) -> Scalar {
                match self {
                    $(Self::$name => Scalar::$name(i as $ty),)*
                }
            }
        }
    };
}

impl_numeric_types!(
    byte(i8),
    long(i32),
    date(i32),
    ldate(i32),
    short(i16),
    quad(i64),
    qwdate(i64),
    belong(i32),
    bequad(i64),
    beshort(i16),
    bedate(i32),
    beldate(i32),
    beqdate(i64),
    ledate(i32),
    lelong(i32),
    leshort(i16),
    lequad(i64),
    leldate(i32),
    leqdate(i64),
    leqwdate(i64),
    leqldate(i64),
    ushort(u16),
    ulong(u32),
    ubelong(u32),
    ubequad(u64),
    ubeshort(u16),
    ubyte(u8),
    uquad(u64),
    ulelong(u32),
    ulequad(u64),
    uleshort(u16),
    // FIXME: guessed
    uledate(u32),
    ubeqdate(u64),
    offset(u64),
    lemsdosdate(u16),
    lemsdostime(u16),
    medate(i32),
    melong(i32),
    meldate(i32),
    guid(u128),
);

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Scalar::date(value) => write!(f, "date({value})"),
            Scalar::ldate(value) => write!(f, "ldate({value})"),
            Scalar::quad(value) => write!(f, "{value}"),
            Scalar::qwdate(value) => write!(f, "qwdate{value}"),
            Scalar::uquad(value) => write!(f, "{value}"),
            Scalar::belong(value) => write!(f, "{value}"),
            Scalar::bequad(value) => write!(f, "{value}"),
            Scalar::beshort(value) => write!(f, "{value}"),
            Scalar::bedate(value) => write!(f, "bedate({value})"),
            Scalar::beldate(value) => write!(f, "beldate({value})"),
            Scalar::beqdate(value) => write!(f, "beqdate({value})"),
            Scalar::byte(value) => write!(f, "{value}"),
            Scalar::ledate(value) => write!(f, "ledate({value})"),
            Scalar::lelong(value) => write!(f, "{value}"),
            Scalar::leshort(value) => write!(f, "{value}"),
            Scalar::lequad(value) => write!(f, "{value}"),
            Scalar::long(value) => write!(f, "{value}"),
            Scalar::short(value) => write!(f, "{value}"),
            Scalar::ushort(value) => write!(f, "{value}"),
            Scalar::ulong(value) => write!(f, "{value}"),
            Scalar::ubelong(value) => write!(f, "{value}"),
            Scalar::ubequad(value) => write!(f, "{value}"),
            Scalar::ubeshort(value) => write!(f, "{value}"),
            Scalar::ubyte(value) => write!(f, "{value}"),
            Scalar::ulelong(value) => write!(f, "{value}"),
            Scalar::ulequad(value) => write!(f, "{value}"),
            Scalar::uleshort(value) => write!(f, "{value}"),
            Scalar::uledate(value) => write!(f, "uledate({value})"),
            Scalar::ubeqdate(value) => write!(f, "ubeqdate({value})"),
            Scalar::offset(value) => write!(f, "{value:p}"),
            Scalar::lemsdosdate(value) => write!(f, "lemsdosdate({value})"),
            Scalar::lemsdostime(value) => write!(f, "lemsdostime({value})"),
            Scalar::medate(value) => write!(f, "medate({value})"),
            Scalar::melong(value) => write!(f, "{value}"),
            Scalar::meldate(value) => write!(f, "meldate({value})"),
            Scalar::leldate(value) => write!(f, "leldate({value})"),
            Scalar::leqdate(value) => write!(f, "leqdate({value})"),
            Scalar::leqldate(value) => write!(f, "leqldate({value})"),
            Scalar::leqwdate(value) => write!(f, "leqwdate({value})"),
            Scalar::guid(value) => {
                write!(
                    f,
                    "{}",
                    Uuid::from_u128(*value)
                        .hyphenated()
                        .to_string()
                        .to_uppercase()
                )
            }
        }
    }
}

impl DynDisplay for Scalar {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        match self {
            Scalar::date(value) => Ok(unix_utc_time_to_string(*value as i64)),
            Scalar::ldate(value) => Ok(unix_local_time_to_string(*value as i64)),
            Scalar::quad(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::qwdate(value) => Ok(windows_filetime_to_string(*value)),
            Scalar::belong(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::bequad(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::beshort(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::bedate(value) => Ok(unix_utc_time_to_string(*value as i64)),
            Scalar::beldate(value) => Ok(unix_local_time_to_string(*value as i64)),
            Scalar::beqdate(value) => Ok(unix_utc_time_to_string(*value)),
            Scalar::byte(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::ledate(value) => Ok(unix_utc_time_to_string(*value as i64)),
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
            Scalar::uledate(value) => Ok(unix_utc_time_to_string(*value as i64)),
            Scalar::ubeqdate(value) => Ok(unix_utc_time_to_string(*value as i64)),
            Scalar::medate(value) => Ok(unix_utc_time_to_string(*value as i64)),
            Scalar::meldate(value) => Ok(unix_local_time_to_string(*value as i64)),
            Scalar::melong(value) => DynDisplay::dyn_fmt(value, f),
            Scalar::leqdate(value) => Ok(unix_utc_time_to_string(*value)),
            Scalar::leldate(value) => Ok(unix_local_time_to_string(*value as i64)),
            Scalar::leqldate(value) => Ok(unix_local_time_to_string(*value)),
            Scalar::leqwdate(value) => Ok(windows_filetime_to_string(*value)),
            Scalar::guid(value) => Ok(Uuid::from_u128(*value)
                .hyphenated()
                .to_string()
                .to_uppercase()),
            Scalar::offset(v) => Ok(format!("{v:#x}")),
            Scalar::lemsdosdate(v) => Ok(parse_fat_date(*v)
                .map(|fd| fd.to_string())
                .unwrap_or("invalid msdos date".into())),
            Scalar::lemsdostime(v) => Ok(parse_fat_time(*v)
                .map(|ft| ft.to_string())
                .unwrap_or("invalid msdos time".into())),
        }
    }
}

macro_rules! impl_float_type {
    ($($name: tt($ty: ty)),* $(,)?) => {
        #[allow(non_camel_case_types)]
        #[derive(Debug, PartialEq, PartialOrd, Clone, Copy, Serialize, Deserialize)]
        pub(crate) enum Float {
            $($name($ty),)*
        }

        impl Float {
            pub(crate) fn size_of(&self) -> usize{
                match self {
                    $(Self::$name(_) => core::mem::size_of::<$ty>(),)*
                }
            }
        }

        impl std::ops::Add for Float {
            type Output = Self;

            fn add(self, other: Self) -> Self::Output {
                match (self, other) {
                    $(
                        (Self::$name(a), Self::$name(b)) => Self::$name(a.add(b)),
                    )*
                    _=> panic!("operation not supported between different numeric variants")
                }
            }
        }

        impl std::ops::Sub for Float {
            type Output = Self;

            fn sub(self, other: Self) -> Self::Output {
                match (self, other) {
                    $(
                        (Self::$name(a), Self::$name(b)) => Self::$name(a.sub(b)),
                    )*
                    _=> panic!("operation not supported between different numeric variants")
                }
            }
        }

        impl std::ops::Mul for Float {
            type Output = Self;

            fn mul(self, other: Self) -> Self::Output {
                match (self, other) {
                    $(
                        (Self::$name(a), Self::$name(b)) => Self::$name(a.mul(b)),
                    )*
                    _=> panic!("operation not supported between different numeric variants")
                }
            }
        }

        impl std::ops::Div for Float {
            type Output = Self;

            fn div(self, other: Self) -> Self::Output {
                match (self, other) {
                    $(
                        (Self::$name(a), Self::$name(b)) => Self::$name(a.div(b)),
                    )*
                    _=> panic!("operation not supported between different numeric variants")
                }
            }
        }

        impl std::ops::Rem for Float {
            type Output = Self;

            fn rem(self, other: Self) -> Self::Output {
                match (self, other) {
                    $(
                        (Self::$name(a), Self::$name(b)) => Self::$name(a.rem(b)),
                    )*
                    _=> panic!("operation not supported between different numeric variants")
                }
            }
        }

        #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
        #[allow(non_camel_case_types)]
        pub(crate) enum FloatDataType {
            $($name,)*
        }

        impl FloatDataType {
            pub(crate) const fn type_size(&self) -> usize {
                match self {
                    $(Self::$name => core::mem::size_of::<$ty>(),)*
                }
            }

            pub(crate) fn float_from_f64(&self, i: f64) -> Float {
                match self {
                    $(Self::$name => Float::$name(i as $ty),)*
                }
            }
        }
    };
}

impl_float_type!(bedouble(f64), ledouble(f64), lefloat(f32), befloat(f32));

impl fmt::Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Float::bedouble(v) => write!(f, "{v}"),
            Float::ledouble(v) => write!(f, "{v}"),
            Float::lefloat(v) => write!(f, "{v}"),
            Float::befloat(v) => write!(f, "{v}"),
        }
    }
}

impl DynDisplay for Float {
    fn dyn_fmt(&self, f: &dyf::FormatSpec) -> Result<String, dyf::Error> {
        match self {
            Float::bedouble(v) => DynDisplay::dyn_fmt(v, f),
            Float::ledouble(v) => DynDisplay::dyn_fmt(v, f),
            Float::lefloat(v) => DynDisplay::dyn_fmt(v, f),
            Float::befloat(v) => DynDisplay::dyn_fmt(v, f),
        }
    }
}
