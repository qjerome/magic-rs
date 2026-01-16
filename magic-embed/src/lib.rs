#![forbid(unsafe_code)]
#![deny(unused_imports)]
#![deny(missing_docs)]
//! # `magic-embed`: Compile-time Magic Database Embedding
//!
//! A procedural macro crate for embedding compiled [`pure_magic`](https://crates.io/crates/pure-magic) databases directly into your Rust binary.
//! This crate provides a convenient way to bundle file type detection rules with your application,
//! eliminating the need for external rule files at runtime.
//!
//! ## Features
//!
//! * **Compile-time Embedding**: Magic rule files are compiled and embedded during build
//! * **Zero Runtime Dependencies**: No need to distribute separate rule files
//! * **Flexible Configuration**: Include/exclude specific rule files or directories
//! * **Seamless Integration**: Works with the [`pure_magic`](https://crates.io/crates/pure-magic)
//!
//! ## Installation
//!
//! Add `magic-embed` to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! magic-embed = "0.1"  # Replace with the latest version
//! pure-magic = "0.1"     # Required peer dependency
//! ```
//!
//! ## Usage
//!
//! Apply the `#[magic_embed]` attribute to a struct to embed a compiled magic database:
//!
//! ```rust
//! use magic_embed::magic_embed;
//! use pure_magic::MagicDb;
//!
//! #[magic_embed(include=["../../magic-db/src/magdir"], exclude=["../../magic-db/src/magdir/der"])]
//! struct MyMagicDb;
//!
//! fn main() -> Result<(), pure_magic::Error> {
//!     let db = MyMagicDb::open()?;
//!     // Use the database as you would with pure_magic
//!     Ok(())
//! }
//! ```
//!
//! ## Attributes
//!
//! | Attribute | Type       | Required | Description |
//! |-----------|------------|----------|-------------|
//! | `include` | String[]   | Yes      | Paths to include in the database (files or directories) |
//! | `exclude` | String[]   | No       | Paths to exclude from the database |
//!
//! ## Complete Example
//!
//! ```rust
//! use magic_embed::magic_embed;
//! use pure_magic::MagicDb;
//! use std::fs::File;
//! use std::env::current_exe;
//!
//! #[magic_embed(
//!     include=["../../magic-db/src/magdir"],
//!     exclude=["../../magic-db/src/magdir/der"]
//! )]
//! struct AppMagicDb;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Open the embedded database
//!     let db = AppMagicDb::open()?;
//!
//!     // Use it to detect file types
//!     let mut file = File::open(current_exe()?)?;
//!     let magic = db.first_magic(&mut file, None)?;
//!
//!     println!("Detected: {} (MIME: {})", magic.message(), magic.mime_type());
//!     Ok(())
//! }
//! ```
//!
//! ## Build Configuration
//!
//! To ensure your database is rebuilt when rule files change, create a `build.rs` file:
//!
//! ```rust,ignore
//! // build.rs
//! fn main() {
//!     println!("cargo:rerun-if-changed=magic/rules/");
//! }
//! ```
//!
//! Replace `magic/rules/` with the path to your actual rule files.
//!
//! ## How It Works
//!
//! 1. **Compile Time**: The macro compiles all specified magic rule files into a binary database
//! 2. **Embedding**: The compiled database is embedded in your binary as a byte array
//! 3. **Runtime**: The `open()` method deserializes the embedded database
//!
//! ## Performance Considerations
//!
//! - The database is compiled only when source files change
//! - Embedded databases increase binary size but eliminate runtime file I/O
//! - Database deserialization happens once at runtime when `open()` is called
//!
//! ## License
//!
//! This project is licensed under the **GPL-3.0 License**.

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use proc_macro::TokenStream;
use pure_magic::{MagicDb, MagicSource};
use quote::quote;
use syn::{
    Expr, ExprArray, ItemStruct, Meta, MetaNameValue, Token, parse::Parser, punctuated::Punctuated,
    spanned::Spanned,
};

/// Parser for procedural macro attributes
///
/// Processes comma-separated key-value attributes for the `magic_embed` macro.
struct MetaParser {
    attr: proc_macro2::TokenStream,
    metas: HashMap<String, Meta>,
}

impl MetaParser {
    /// Creates a new [`MetaParser`] from a token stream
    ///
    /// # Arguments
    ///
    /// * `attr` - [`proc_macro2::TokenStream`] - Attribute token stream to parse
    ///
    /// # Returns
    ///
    /// * `Result<Self, syn::Error>` - Parsed metadata or syntax error
    fn parse_meta(attr: proc_macro2::TokenStream) -> Result<Self, syn::Error> {
        let mut out = HashMap::new();

        // parser for a comma-separated list of Meta entries
        let parser = Punctuated::<Meta, Token![,]>::parse_terminated;

        let metas = match parser.parse2(attr.clone()) {
            Ok(m) => m,
            Err(e) => return Err(syn::Error::new_spanned(attr, e.to_string())),
        };

        for meta in metas {
            out.insert(
                meta.path()
                    .get_ident()
                    .ok_or(syn::Error::new_spanned(
                        meta.clone(),
                        "failed to process meta",
                    ))?
                    .to_string(),
                meta,
            );
        }
        Ok(Self {
            attr: attr.clone(),
            metas: out,
        })
    }

    /// Retrieves a key-value attribute by name
    ///
    /// # Arguments
    ///
    /// * `key` - `&str` - Name of the attribute to retrieve
    ///
    /// # Returns
    ///
    /// * `Result<Option<&MetaNameValue>, syn::Error>` - Found attribute or error
    fn get_key_value(&self, key: &str) -> Result<Option<&MetaNameValue>, syn::Error> {
        if let Some(meta) = self.metas.get(key) {
            match meta {
                Meta::NameValue(m) => return Ok(Some(m)),
                _ => {
                    return Err(syn::Error::new_spanned(
                        &self.attr,
                        format!("expecting a key value attribute: {key}"),
                    ));
                }
            }
        }
        Ok(None)
    }
}

/// Converts a [`MetaNameValue`] array expression to a vector of strings
///
/// # Arguments
///
/// * `nv` - Name-value attribute containing array
///
/// # Returns
///
/// * `Result<Vec<(proc_macro2::Span, String)>, syn::Error>` - Vector of (span, string) tuples
fn meta_name_value_to_string_vec(
    nv: &MetaNameValue,
) -> Result<Vec<(proc_macro2::Span, String)>, syn::Error> {
    if let Expr::Array(ExprArray { elems, .. }) = &nv.value {
        Ok(elems
            .into_iter()
            .filter_map(|e| match e {
                Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(lit_str),
                    ..
                }) => Some((lit_str.span(), lit_str.value())),
                _ => None,
            })
            .collect::<Vec<_>>())
    } else {
        Err(syn::Error::new_spanned(
            &nv.value,
            "expected an array literal like [\"foo\", \"bar\"]",
        ))
    }
}

fn impl_magic_embed(attr: TokenStream, item: TokenStream) -> Result<TokenStream, syn::Error> {
    // Parse the input function
    let input_struct: ItemStruct = syn::parse2(item.into())?;
    let struct_name = &input_struct.ident;
    let cs = proc_macro::Span::call_site();

    let Some(source_file) = cs.local_file() else {
        return Ok(quote! {}.into());
    };

    let source_dir = source_file.parent().unwrap();

    // convert to proc-macro2 TokenStream for syn helpers
    let ts2: proc_macro2::TokenStream = attr.into();

    let struct_vis = input_struct.vis;

    let metas = MetaParser::parse_meta(ts2)?;

    let exclude = if let Some(exclude) = metas.get_key_value("exclude")? {
        meta_name_value_to_string_vec(exclude)?
            .into_iter()
            .map(|(s, p)| (s, source_dir.join(p)))
            .collect()
    } else {
        vec![]
    };

    let include_nv = metas.get_key_value("include")?.ok_or(syn::Error::new(
        struct_name.span(),
        "expected  a list of files or directory to include: \"include\" = [\"magdir\"]",
    ))?;

    let include: Vec<(proc_macro2::Span, PathBuf)> = meta_name_value_to_string_vec(include_nv)?
        .into_iter()
        .map(|(s, p)| (s, source_dir.join(p)))
        .collect();

    // we don't walk rules recursively
    let mut wo = fs_walk::WalkOptions::new();
    wo.files().max_depth(0).sort(true);

    let mut db = MagicDb::new();

    let exclude_set: HashSet<PathBuf> = exclude.into_iter().map(|(_, p)| p).collect();

    macro_rules! load_file {
        ($span: expr, $path: expr) => {
            MagicSource::open($path).map_err(|e| {
                syn::Error::new(
                    $span.clone(),
                    format!(
                        "failed to parse magic file={}: {e}",
                        $path.to_string_lossy()
                    ),
                )
            })?
        };
    }

    let mut rules = vec![];
    for (s, p) in include.iter() {
        if p.is_dir() {
            for rule_file in wo.walk(p) {
                let rule_file = rule_file
                    .map_err(|e| syn::Error::new(*s, format!("failed to list rule file: {e}")))?;

                if exclude_set.contains(&rule_file) {
                    continue;
                }

                rules.push(load_file!(s, &rule_file));
            }
        } else if p.is_file() {
            rules.push(load_file!(s, p));
        }
    }

    db.load_bulk(rules.into_iter());
    db.verify().map_err(|e| {
        syn::Error::new(include_nv.span(), format!("inconsistent database: {e}"))
    })?;

    // Serialize and save database
    let mut ser = vec![];
    db.serialize(&mut ser).map_err(|e| {
        syn::Error::new(
            struct_name.span(),
            format!("failed to serialize database: {e}"),
        )
    })?;

    // Generate the output: the original function + a print statement
    let output = quote! {
        /// This structure exposes an embedded compiled magic database.
        #struct_vis struct #struct_name;

        impl #struct_name {
            const DB: &[u8] = &[ #( #ser ),* ];

            /// Opens the embedded magic database and returns a [`pure_magic::MagicDb`]
            #struct_vis fn open() -> Result<pure_magic::MagicDb, pure_magic::Error> {
                pure_magic::MagicDb::deserialize(&mut Self::DB.as_ref())
            }
        }
    };

    Ok(output.into())
}

/// Procedural macro to embed a compiled [`pure_magic::MagicDb`]
///
/// This attribute macro compiles magic rule files at program
/// compile time and embeds them in the binary. The database
/// will not be automatically rebuilt when rule files change
/// (c.f. see Note section below).
///
/// # Attributes
///
/// * `include` - Array of paths to include in the database (required)
/// * `exclude` - Array of paths to exclude from the database (optional)
///
/// # Examples
///
/// ```
/// use magic_embed::magic_embed;
/// use pure_magic::MagicDb;
///
/// #[magic_embed(include=["../../magic-db/src/magdir"], exclude=["../../magic-db/src/magdir/der"])]
/// struct EmbeddedMagicDb;
///
/// let db: MagicDb = EmbeddedMagicDb::open().unwrap();
/// ```
///
/// # Errors
///
/// This macro will emit a compile-time error if:
/// - The `include` attribute is missing
/// - Specified paths don't exist
/// - Database compilation fails
/// - File I/O operations fail
///
/// # Note
///
/// If you want Cargo to track changes to your rule files (e.g., `magdir/`),
/// you **must** create a build script in your project. The proc-macro cannot
/// track these files directly because it embeds only the compiled database,
/// not the rule files themselves. Add a `build.rs` file like this:
///
/// ```ignore
/// // build.rs
/// fn main() {
///     println!("cargo::rerun-if-changed=magdir/");
/// }
/// ```
///
/// Replace `magdir/` with the path to your rule files.
#[proc_macro_attribute]
pub fn magic_embed(attr: TokenStream, item: TokenStream) -> TokenStream {
    match impl_magic_embed(attr, item) {
        Ok(ts) => ts,
        Err(e) => e.to_compile_error().into(),
    }
}
