[![Crates.io Version](https://img.shields.io/crates/v/pure-magic?style=for-the-badge)](https://crates.io/crates/pure-magic)
[![docs.rs](https://img.shields.io/docsrs/pure-magic?style=for-the-badge)](https://docs.rs/pure-magic)

<!-- cargo-rdme start -->

# `pure-magic`: A pure and safe Rust Reimplementation of `libmagic`

Unlike many file identification crates, `pure-magic` is highly compatible with the standard
`magic` rule format, allowing seamless reuse of existing
[rules](https://github.com/qjerome/magic-rs/tree/main/magic-db/src/magdir). This makes it an ideal
drop-in replacement for crates relying on **`libmagic` C bindings**, where memory safety is critical.

**Key Features:**
- File type detection
- MIME type inference
- Custom magic rule parsing

## Installation
Add `pure-magic` to your `Cargo.toml`:

```toml
[dependencies]
pure-magic = "0.1"  # Replace with the latest version
```

Or add the latest version with cargo:

```sh
cargo add pure-magic
```

## Quick Start

### Detect File Types Programmatically
```rust
use pure_magic::{MagicDb, MagicSource, DataReader};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut db = MagicDb::new();
    // Create a MagicSource from a file
    let rust_magic = MagicSource::open("../magic-db/src/magdir/rust")?;
    db.load(rust_magic);
    // Verification is not mandatory
    db.verify()?;

    // Open a file and detect its type
    let mut dr = File::open("src/lib.rs")
        .and_then(DataReader::from_file)?;
    let magic = db.first_magic(&mut dr, None)?;

    println!(
        "File type: {} (MIME: {}, strength: {})",
        magic.message(),
        magic.mime_type(),
        magic.strength()
    );
    Ok(())
}
```

### Get All Matching Rules
```rust
use pure_magic::{MagicDb, MagicSource, DataReader};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut db = MagicDb::new();
    // Create a MagicSource from a file
    let rust_magic = MagicSource::open("../magic-db/src/magdir/rust")?;
    db.load(rust_magic);

    // Open a file and detect its type
    let mut dr = File::open("src/lib.rs")
        .and_then(DataReader::from_file)?;

    // Get all matching rules, sorted by strength
    let magics = db.all_magics(&mut dr)?;

    // Must contain rust file magic and default text magic
    assert!(magics.len() > 1);

    for magic in magics {
        println!(
            "Match: {} (strength: {}, source: {})",
            magic.message(),
            magic.strength(),
            magic.source().unwrap_or("unknown")
        );
    }
    Ok(())
}
```

### Serialize a Database to Disk
```rust
use pure_magic::{MagicDb, MagicSource};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut db = MagicDb::new();
    // Create a MagicSource from a file
    let rust_magic = MagicSource::open("../magic-db/src/magdir/rust")?;
    db.load(rust_magic);

    // Serialize the database to a file
    let mut output = File::create("/tmp/compiled.db")?;
    db.serialize(&mut output)?;

    println!("Database saved to file");
    Ok(())
}
```

### Deserialize a Database
```rust
use pure_magic::{MagicDb, MagicSource};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut db = MagicDb::new();
    // Create a MagicSource from a file
    let rust_magic = MagicSource::open("../magic-db/src/magdir/rust")?;
    db.load(rust_magic);

    // Serialize the database in a vector
    let mut ser = vec![];
    db.serialize(&mut ser)?;
    println!("Database saved to vector");

    // We deserialize from slice
    let db = MagicDb::deserialize(&mut ser.as_slice())?;

    assert!(!db.rules().is_empty());

    Ok(())
}
```

## License
This project is dual-licensed under either:
- **GPL-3.0**
- **BSD-2-Clause**

## Contributing
Contributions are welcome! Open an issue or submit a pull request.

## Acknowledgments
- Inspired by the original `libmagic` (part of the `file` command).

<!-- cargo-rdme end -->
