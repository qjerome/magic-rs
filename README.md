# Magic File Type Detection Ecosystem

**A safe Rust implementation of file type detection with 99% compatibility with libmagic rule format**

This ecosystem provides a complete, memory-safe alternative to the traditional `libmagic` implementation, while maintaining near-full compatibility with existing rule files.

## 🔥 Key Advantages

- **99% libmagic Compatible** - Uses the same rule format and syntax
- **Memory Safe** - Pure Rust implementation with no `unsafe` code
- **High Performance** - Optimized parsing and detection
- **Embeddable** - Compile rules directly into your binary
- **Extensible** - Easy to add new file type detection rules

## 📦 Crates Overview

### 1. [`magic-rs`](magic/)
**Core libmagic-compatible detection engine**

- **99% compatible** with original libmagic rule syntax
- Safe Rust implementation (no `unsafe`)
- Powerful APIs:
  - File type detection
  - MIME type identification
  - Access rule strength
  - Polyglot file detection

### 2. [`magic-embed`](magic-embed/)
**Procedural macro for embedding rule databases**

- Compiles libmagic-compatible rules at build time
- Use it to embeds a compiled database of magics in your binary

### 3. [`magic-db`](magic-db/)
**Precompiled libmagic-compatible rule database**

- Contains [magic rules](magic-db/src/magdir/) 
- Ready-to-use with zero configuration


## 🚀 Getting Started: using [`wiza`](wiza/) CLI

### Installation

```sh
cargo install wiza
```

### Basic file identification

```sh
$ wiza /bin/file
/bin/file source:elf strength:431 mime:application/x-pie-executable magic:ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV)
```

## 📜 Rule Compatibility

This project has been built to provide the maximum level of compatibility
with existing `libmagic` rules. So most of the rules you will find in
the [`file`](https://github.com/file/file) repository will directly be 
compatible with this project. You just need to be aware of the current f
few incompatibilities:

- **Ternary printf format is not supported**:  The following extract from ELF
detection will not be supported, however it can be fixed trivially without
relying on ternary formatting.
```
0	name		elf-le
[...]
>16	leshort		3		${x?pie executable:shared object},
!:mime	application/x-${x?pie-executable:sharedlib}
```
- **DER Rule Limitation**: The only major incompatibility is with ASN.1/DER encoding rules, which require specialized test operations not yet implemented in `magic-rs`. All other rule types work identically to libmagic.

## 📚 Documentation

- [magic-rs API Docs](https://docs.rs/magic-rs) - Core detection library
- [magic-embed Docs](https://docs.rs/magic-embed) - Embedding macro
- [magic-db Docs](https://docs.rs/magic-db) - Precompiled database
- [Rule Syntax Guide](https://www.man7.org/linux/man-pages/man4/magic.4.html) - libmagic-compatible rule format

## 🤝 Contributing

We welcome contributions to improve libmagic compatibility:

1. Open an issue with a description of the problem
2. Include sample files that demonstrate the issue
3. Suggest specific rule modifications if possible

You can also contribute by:
- Improving existing rules in the [`src/magdir`](https://github.com/qjerome/magic-rs/magic-db/src/magdir) directory
- Adding support for new file formats
- Helping optimize rule performance or accuracy

## 📄 License

All components are licensed under **GPL-3.0**.

## 🙌 Acknowledgments

- Thanks to all the people behind [file](https://github.com/file/file), project which served as a foundation for this work.
- Thanks to [@adulau](https://github.com/adulau) for supporting this work.
- Thanks to all my colleagues at [CIRCL](https://circl.lu/) who listened to me (without complaining) talking about `magic-rs` almost every single day since I started this project.