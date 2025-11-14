//! # `magic-db`: Precompiled Magic Rules Database
//!
//! A precompiled database of file type detection rules based on the original `libmagic` project,
//! optimized and adapted for use with [`magic-rs`](https://crates.io/crates/magic-rs).
//! This crate provides ready-to-use file type detection capabilities without requiring external rule files.
//!
//! ## Features
//!
//! - **Precompiled Rules**: Optimized database embedded directly in your binary
//! - **No External Dependencies**: All rules are included in the compiled crate
//! - **Enhanced Rules**: Improved and extended versions of the original `libmagic` rules
//! - **Easy Integration**: Simple one-line access to the compiled database
//!
//! ## Installation
//!
//! Add `magic-db` to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! magic-db = "0.1"  # Replace with the latest version
//! magic-rs = "0.1"  # Required peer dependency
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use magic_db::CompiledDb;
//! use std::fs::File;
//! use std::env::current_exe;
//!
//! fn main() -> Result<(), magic_rs::Error> {
//!     // Open the precompiled database
//!     let db = CompiledDb::open()?;
//!
//!     // Use it to detect file types
//!     let mut file = File::open(current_exe()?)?;
//!     let magic = db.magic_first(&mut file, None)?;
//!     assert!(!magic.is_default());
//!
//!     println!("File type: {}", magic.message());
//!     println!("MIME type: {}", magic.mime_type());
//!     Ok(())
//! }
//! ```
//!
//! ## About the Rules
//!
//! This database contains slightly modified versions of the original `libmagic` rules that are available
//! in the [`src/magdir`](https://github.com/qjerome/magic-rs/tree/main/magic-db/src/magdir) directory of this repository.
//!
//! Some of the rules have been:
//! - **Adapted**: Modified to work with the [`magic-rs`](https://crates.io/crates/magic-rs) parser
//! - **Optimized**: Performance improvements for common file types
//! - **Extended**: Additional rules were created
//! - **Fixed**: Corrections to inaccurate or problematic original rules
//!
//! ## Rule Exclusions
//!
//! The database intentionally excludes the `der` rules (ASN.1/DER encoding rules) because:
//! - The [`magic-rs`](https://crates.io/crates/magic-rs) parser doesn't support (yet) the specific DER test types
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
//! - [`magic-rs`](https://crates.io/crates/magic-rs): The core file type detection library
//! - [`magic-embed`](https://crates.io/crates/magic-embed): The macro used to create this database
//! - [`magic`](https://www.man7.org/linux/man-pages/man4/magic.4.html): Expected magic rule format

use magic_embed::magic_embed;

#[magic_embed(include=["magdir"], exclude=["magdir/der"])]
pub struct CompiledDb;

#[cfg(test)]
mod test {
    use crate::CompiledDb;
    use std::{env, fs::File};

    #[test]
    fn test_compiled_db() {
        let db = CompiledDb::open().unwrap();
        let mut exe = File::open(env::current_exe().unwrap()).unwrap();
        let magic = db.magic_first(&mut exe, None).unwrap();
        println!("{}", magic.message());
        assert!(!magic.is_default())
    }
}
