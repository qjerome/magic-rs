#![forbid(unsafe_code)]
#![deny(unused_imports)]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
//! # `magic-db`: Precompiled Magic Rules Database
//!
//! A precompiled database of file type detection rules based on the original `libmagic` project,
//! optimized and adapted for use with [`pure-magic`](https://crates.io/crates/pure-magic).
//! This crate provides ready-to-use file type detection capabilities without requiring external rule files.
//!
//! ## Features
//!
//! - **Precompiled Rules**: Optimized database embedded directly in your binary
//! - **No External Dependencies**: All rules are included in the compiled crate
//! - **Enhanced Rules**: Improved and extended versions of the original `libmagic` rules
//! - **Easy Integration**: Simple one-line access to the compiled database
//!
//! ### Optional Cargo Features
//!
//! - **global**: Enables `magic_db::global()`, a lazily-initialized, process-wide `MagicDb`.
//!   This provides a convenient singleton but is optional. If you need explicit lifetime
//!   control or multiple independent instances, use `CompiledDb::open()` instead.
//!
//! ## Installation
//!
//! Add `magic-db` to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! magic-db = "0.1"  # Replace with the latest version
//! pure-magic = "0.1"  # Required peer dependency
//! ```
//!
//! ## Usage
//!
//! ### Manual lifecycle (default)
//!
//! ```rust
//! use std::fs::File;
//! use std::env::current_exe;
//!
//! fn main() -> Result<(), pure_magic::Error> {
//!     // Open the precompiled database
//!     let db = magic_db::load()?;
//!
//!     // Use it to detect file types
//!     let mut file = File::open(current_exe()?)?;
//!     let magic = db.first_magic(&mut file, None)?;
//!     assert!(!magic.is_default());
//!
//!     println!("File type: {}", magic.message());
//!     println!("MIME type: {}", magic.mime_type());
//!     Ok(())
//! }
//! ```
//!
//! ### Global singleton (optional)
//!
//! The crate provides a **convenience global database** via the
//! `global` feature. This is process-wide, lazily initialized, and
//! kept alive until program termination.
//!
//! Enable it in `Cargo.toml`:
//!
//! ```toml
//! magic_db = { version = "0.1", features = ["global"] }
//! ```
//!
//! Then use it like this:
//!
//! ```rust
//! use magic_db::global;
//!
//! let db = global().unwrap();
//! ```
//!
//! **Note:** Use the global feature only if you want a single, shared
//! database. For multiple independent instances or explicit lifetime
//! management, use `CompiledDb::open()`.
//!
//! ## About the Rules
//!
//! This database contains slightly modified versions of the original `libmagic` rules that are available
//! in the [`src/magdir`](https://github.com/qjerome/magic-rs/tree/main/magic-db/src/magdir) directory of this repository.
//!
//! Some of the rules have been:
//! - **Adapted**: Modified to work with the [`pure-magic`](https://crates.io/crates/pure-magic) parser
//! - **Optimized**: Performance improvements for common file types
//! - **Extended**: Additional rules were created
//! - **Fixed**: Corrections to inaccurate or problematic original rules
//!
//! ## Rule Exclusions
//!
//! The database intentionally excludes the `der` rules (ASN.1/DER encoding rules) because:
//! - The [`pure-magic`](https://crates.io/crates/pure-magic) parser doesn't support (yet) the specific DER test types
//!   implemented in the original `libmagic`
//!
//! ## Source Rules
//!
//! The source magic rules are available in the repository at:
//! [`src/magdir`](https://github.com/qjerome/magic-rs/tree/main/magic-db/src/magdir)
//!
//! You can:
//! 1. Browse the rules to understand how file types are detected
//! 2. Suggest improvements by opening issues or pull requests
//! 3. Use these rules as a reference for creating your own custom rules
//!
//! ## License
//!
//! This project is licensed under the **GPL-3.0 License**.
//!
//! ## See Also
//!
//! - [`pure-magic`](https://crates.io/crates/pure-magic): The core file type detection library
//! - [`magic-embed`](https://crates.io/crates/magic-embed): The macro used to create this database
//! - [`magic`](https://www.man7.org/linux/man-pages/man4/magic.4.html): Expected magic rule format

use magic_embed::magic_embed;
use pure_magic::{Error, MagicDb};

#[cfg(feature = "global")]
use std::sync::OnceLock;

#[cfg(feature = "global")]
static DB: OnceLock<MagicDb> = OnceLock::new();

#[magic_embed(include=["magdir"], exclude=["magdir/der"])]
struct CompiledDb;

#[cfg(feature = "global")]
#[cfg_attr(docsrs, doc(cfg(feature = "global")))]
/// Returns a process-wide read-only [`MagicDb`] initialized on first use.
///
/// This function is provided as a convenience for applications that
/// want a shared database without managing its lifetime explicitly.
/// The database is kept alive until program termination.
///
/// If you need explicit control over the database lifetime or want
/// multiple independent instances, use [`load`] instead.
pub fn global() -> Result<&'static MagicDb, Error> {
    match DB.get() {
        Some(db) => Ok(db),
        None => {
            let _ = DB.set(CompiledDb::open()?);
            Ok(DB.wait())
        }
    }
}

#[inline(always)]
/// Loads a [`MagicDb`] from the embedded, precompiled database.
///
/// This function constructs an owned [`MagicDb`] from data embedded
/// at compile time. No file system access or runtime dependencies
/// are involved.
///
/// Each call returns a new, independent instance.
pub fn load() -> Result<MagicDb, Error> {
    CompiledDb::open()
}

#[cfg(test)]
mod test {
    use crate as magic_db;
    use std::{env, fs::File};

    #[test]
    fn test_compiled_db() {
        let db = magic_db::load().unwrap();
        let mut exe = File::open(env::current_exe().unwrap()).unwrap();
        let magic = db.first_magic(&mut exe, None).unwrap();
        println!("{}", magic.message());
        assert!(!magic.is_default())
    }

    #[test]
    #[cfg(feature = "global")]
    fn test_compiled_db_static() {
        let db = crate::global().unwrap();
        let mut exe = File::open(env::current_exe().unwrap()).unwrap();
        let magic = db.first_magic(&mut exe, None).unwrap();
        println!("{}", magic.message());
        assert!(!magic.is_default())
    }
}
