[![Crates.io Version](https://img.shields.io/crates/v/magic-embed?style=for-the-badge)](https://crates.io/crates/magic-embed)
[![docs.rs](https://img.shields.io/docsrs/magic-embed?style=for-the-badge)](https://docs.rs/magic-embed)

<!-- cargo-rdme start -->

# `magic-embed`: Compile-time Magic Database Embedding

A procedural macro crate for embedding compiled [`pure_magic`](https://crates.io/crates/pure-magic) databases directly into your Rust binary.
This crate provides a convenient way to bundle file type detection rules with your application,
eliminating the need for external rule files at runtime.

## Features

* **Compile-time Embedding**: Magic rule files are compiled and embedded during build
* **Zero Runtime Dependencies**: No need to distribute separate rule files
* **Flexible Configuration**: Include/exclude specific rule files or directories
* **Seamless Integration**: Works with the [`pure_magic`](https://crates.io/crates/pure-magic)

## Installation

Add `magic-embed` to your `Cargo.toml`:

```toml
[dependencies]
magic-embed = "0.1"  # Replace with the latest version
pure-magic = "0.1"     # Required peer dependency
```

## Usage

Apply the `#[magic_embed]` attribute to a struct to embed a compiled magic database:

```rust
use magic_embed::magic_embed;
use pure_magic::MagicDb;

#[magic_embed(include=["../../magic-db/src/magdir"], exclude=["../../magic-db/src/magdir/der"])]
struct MyMagicDb;

fn main() -> Result<(), pure_magic::Error> {
    let db = MyMagicDb::open()?;
    // Use the database as you would with pure_magic
    Ok(())
}
```

## Attributes

| Attribute | Type       | Required | Description |
|-----------|------------|----------|-------------|
| `include` | String[]   | Yes      | Paths to include in the database (files or directories) |
| `exclude` | String[]   | No       | Paths to exclude from the database |

## Complete Example

```rust
use magic_embed::magic_embed;
use pure_magic::MagicDb;
use std::fs::File;
use std::env::current_exe;

#[magic_embed(
    include=["../../magic-db/src/magdir"],
    exclude=["../../magic-db/src/magdir/der"]
)]
struct AppMagicDb;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open the embedded database
    let db = AppMagicDb::open()?;

    // Use it to detect file types
    let mut file = File::open(current_exe()?)?;
    let magic = db.first_magic(&mut file, None)?;

    println!("Detected: {} (MIME: {})", magic.message(), magic.mime_type());
    Ok(())
}
```

## Build Configuration

To ensure your database is rebuilt when rule files change, create a `build.rs` file:

```rust
// build.rs
fn main() {
    println!("cargo:rerun-if-changed=magic/rules/");
}
```

Replace `magic/rules/` with the path to your actual rule files.

## How It Works

1. **Compile Time**: The macro compiles all specified magic rule files into a binary database
2. **Embedding**: The compiled database is embedded in your binary as a byte array
3. **Runtime**: The `open()` method deserializes the embedded database

## Performance Considerations

- The database is compiled only when source files change
- Embedded databases increase binary size but eliminate runtime file I/O
- Database deserialization happens once at runtime when `open()` is called

## License

This project is licensed under the **GPL-3.0 License**.

<!-- cargo-rdme end -->
