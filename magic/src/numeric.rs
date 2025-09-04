macro_rules! impl_numeric_types {
    ($($name: tt($ty: ty)),* $(,)?) => {
        #[allow(non_camel_case_types)]
        #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]

        pub(crate) enum Scalar {
            $($name($ty),)*
        }

        impl Scalar {
            pub(crate) fn is_zero(&self) -> bool {
                match self {
                    $(Self::$name(x) => *x == 0,)*
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

        impl std::ops::Add for Scalar {
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

        impl std::ops::Sub for Scalar {
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

        impl std::ops::Mul for Scalar {
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

        impl std::ops::Div for Scalar {
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

        impl std::ops::Rem for Scalar {
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

        #[derive(Debug, Clone, Copy)]
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
    short(i16),
    quad(i64),
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
    offset(u64),
    lemsdosdate(u16),
    lemsdostime(u16),
    medate(i32),
    melong(i32),
    meldate(i32),
);
