#[cfg(not(feature = "sync"))]
pub(crate) use default::*;

#[cfg(feature = "sync")]
pub(crate) use sync::*;

#[cfg(not(feature = "sync"))]
mod default {
    use std::cell::{OnceCell, RefCell};

    use serde::{Deserialize, Serialize};

    use crate::EntryNode;

    /// Deserialize-on-Use wrapper for an `EntryNode`.
    ///
    /// This struct implements the "Deserialize on Use" pattern, where the `EntryNode` is
    /// deserialized only when first accessed via `get_or_de()`. The serialized form is stored
    /// in a `RefCell<Option<Vec<u8>>>`, allowing for interior mutability, while the actual
    /// `EntryNode` is stored in a `OnceCell`, ensuring it is deserialized at most once.
    ///
    /// The serialized data is consumed during deserialization, ensuring it is only used once.
    #[derive(Debug, Deserialize)]
    pub(crate) struct DouEntryNode {
        ser: RefCell<Option<Vec<u8>>>,
        #[serde(skip)]
        entry: OnceCell<EntryNode>,
    }

    impl From<EntryNode> for DouEntryNode {
        fn from(value: EntryNode) -> Self {
            let cell = OnceCell::new();
            cell.set(value).unwrap();
            Self {
                ser: RefCell::new(None),
                entry: cell,
            }
        }
    }

    impl Serialize for DouEntryNode {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            #[derive(Debug, Serialize)]
            struct Tmp {
                ser: RefCell<Option<Vec<u8>>>,
            }

            let ser = self.ser.borrow_mut();
            let tmp = if ser.is_none() {
                Tmp {
                    ser: RefCell::new(Some(
                        bincode::serde::encode_to_vec(
                            self.entry.get().unwrap(),
                            bincode::config::standard(),
                        )
                        .unwrap(),
                    )),
                }
            } else {
                Tmp {
                    ser: self.ser.clone(),
                }
            };

            tmp.serialize(serializer)
        }
    }

    impl DouEntryNode {
        /// Returns a reference to the entry node.
        pub(crate) fn get_or_de(&self) -> &EntryNode {
            self.entry.get_or_init(|| {
                let ser = self.ser.borrow_mut().take().unwrap();
                let (e, _) =
                    bincode::serde::decode_from_slice(&ser, bincode::config::standard()).unwrap();
                e
            })
        }
    }
}

#[cfg(feature = "sync")]
mod sync {
    use std::sync::{OnceLock, RwLock};

    use serde::{Deserialize, Serialize};

    use crate::EntryNode;

    /// Thread-safe Deserialize-on-Use wrapper for an `EntryNode`.
    ///
    /// This struct implements the "Deserialize on Use" pattern in a thread-safe manner.
    /// The `EntryNode` is deserialized only when first accessed via `get_or_de()`.
    /// The serialized form is stored in an `RwLock<Option<Vec<u8>>>`, allowing for concurrent
    /// read access or exclusive write access. The actual `EntryNode` is stored in a `OnceLock`,
    /// ensuring thread-safe, one-time deserialization.
    ///
    /// The serialized data is consumed during deserialization, ensuring it is only used once.
    #[derive(Debug, Deserialize)]
    pub(crate) struct DouEntryNode {
        ser: RwLock<Option<Vec<u8>>>,
        #[serde(skip)]
        entry: OnceLock<EntryNode>,
    }

    impl From<EntryNode> for DouEntryNode {
        fn from(value: EntryNode) -> Self {
            let cell = OnceLock::new();
            cell.set(value).unwrap();
            Self {
                ser: RwLock::new(None),
                entry: cell,
            }
        }
    }

    impl Serialize for DouEntryNode {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            #[derive(Debug, Serialize)]
            struct Tmp {
                ser: RwLock<Option<Vec<u8>>>,
            }

            let ser = self.ser.read().unwrap();
            let tmp = if ser.is_none() {
                Tmp {
                    ser: RwLock::new(Some(
                        bincode::serde::encode_to_vec(
                            self.entry.get().unwrap(),
                            bincode::config::standard(),
                        )
                        .unwrap(),
                    )),
                }
            } else {
                Tmp {
                    ser: RwLock::new(ser.clone()),
                }
            };

            tmp.serialize(serializer)
        }
    }

    impl DouEntryNode {
        /// Returns a reference to the entry node.
        pub(crate) fn get_or_de(&self) -> &EntryNode {
            self.entry.get_or_init(|| {
                let ser = self.ser.write().unwrap().take().unwrap();
                let (e, _) =
                    bincode::serde::decode_from_slice(&ser, bincode::config::standard()).unwrap();
                e
            })
        }
    }
}
