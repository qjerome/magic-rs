use std::{
    collections::{HashMap, HashSet},
    fs,
    path::PathBuf,
};

use magic_rs::{MagicDb, MagicSource};
use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Expr, ExprArray, ItemStruct, Meta, MetaNameValue, Token, parse::Parser, punctuated::Punctuated,
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

    // convert to proc-macro2 TokenStream for syn helpers
    let ts2: proc_macro2::TokenStream = attr.into();

    let struct_vis = input_struct.vis;

    let metas = MetaParser::parse_meta(ts2)?;

    let exclude = if let Some(exclude) = metas.get_key_value("exclude")? {
        meta_name_value_to_string_vec(exclude)?
            .into_iter()
            .map(|(s, p)| (s, PathBuf::from(p)))
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
        .map(|(s, p)| (s, PathBuf::from(p)))
        .collect();

    let database_dir = {
        let p = PathBuf::from("target").join("magic-db");
        fs::create_dir_all(&p).map_err(|e| {
            syn::Error::new(
                struct_name.span(),
                format!("failed to create directory: {e}"),
            )
        })?;

        p.canonicalize().map_err(|e| {
            syn::Error::new(
                struct_name.span(),
                format!("failed to canonicalize path: {e}"),
            )
        })?
    };

    let database_path = database_dir.join("db.bin");

    let database_path_mod = if database_path.exists() {
        Some(
            database_path
                .metadata()
                .and_then(|m| m.modified())
                .map_err(|e| {
                    syn::Error::new(
                        struct_name.span(),
                        format!("failed to get database file metadata: {e}"),
                    )
                })?,
        )
    } else {
        None
    };

    // we don't walk rules recursively
    let mut wo = fs_walk::WalkOptions::new();
    wo.files().max_depth(0).sort(true);

    let mut db = MagicDb::new();

    for (s, p) in include.iter().chain(exclude.iter()) {
        if !p.exists() {
            return Err(syn::Error::new(
                *s,
                format!("no such file or directory: {}", p.to_string_lossy()),
            ));
        }
    }

    let mut must_compile_db = false;
    for (_, p) in include.iter() {
        if p.is_dir() {
            for f in wo.walk(p).flatten() {
                let metadata = f.metadata().and_then(|m| m.modified()).map_err(|e| {
                    syn::Error::new(
                        struct_name.span(),
                        format!("failed to get database file metadata: {e}"),
                    )
                })?;

                if Some(metadata) > database_path_mod {
                    must_compile_db = true;
                    break;
                }
            }
        } else if p.is_file() {
        }
    }

    if must_compile_db {
        let exclude_set: HashSet<PathBuf> = exclude.into_iter().map(|(_, p)| p).collect();

        macro_rules! load_file {
            ($span: expr, $path: expr) => {
                let f = MagicSource::open($path).map_err(|e| {
                    syn::Error::new(
                        $span.clone(),
                        format!(
                            "failed to parse magic file={}: {e}",
                            $path.to_string_lossy()
                        ),
                    )
                })?;
                db.load(f).map_err(|e| {
                    syn::Error::new(
                        $span.clone(),
                        format!("database failed to load magic file: {e}"),
                    )
                })?;
            };
        }

        for (s, p) in include.iter() {
            if p.is_dir() {
                for rule_file in wo.walk(p) {
                    let rule_file = rule_file.map_err(|e| {
                        syn::Error::new(*s, format!("failed to list rule file: {e}"))
                    })?;

                    if exclude_set.contains(&rule_file) {
                        continue;
                    }

                    load_file!(s, &rule_file);
                }
            } else if p.is_file() {
                load_file!(s, p);
            }
        }

        // Serialize and save database
        let mut ser = vec![];
        db.serialize(&mut ser).map_err(|e| {
            syn::Error::new(
                struct_name.span(),
                format!("failed to serialize database: {e}"),
            )
        })?;

        fs::write(&database_path, ser).map_err(|e| {
            syn::Error::new(
                struct_name.span(),
                format!("failed to save database file: {e}"),
            )
        })?;
    }

    let str_db_path = database_path.to_string_lossy().to_string();

    // Generate the output: the original function + a print statement
    let output = quote! {
        /// This structure exposes an embedded compiled magic database.
        #struct_vis struct #struct_name;

        impl #struct_name {
            const DB: &[u8] = include_bytes!(#str_db_path);

            /// Opens the embedded magic database and returns a [`magic_rs::MagicDb`]
            #struct_vis fn open() -> Result<magic_rs::MagicDb, magic_rs::Error> {
                magic_rs::MagicDb::deserialize(&mut Self::DB.as_ref())
            }
        }
    };

    Ok(output.into())
}

/// Procedural macro to embed a compiled [`magic_rs::MagicDb`]
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
/// use magic_rs::MagicDb;
///
/// #[magic_embed(include=["magic-db/src/magdir"], exclude=["magic-db/src/magdir/der"])]
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
