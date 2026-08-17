# magic-rs

A safe Rust implementation of file type detection, compatible with the
`libmagic` rule format.

This workspace provides a memory-safe alternative to `libmagic`, parsing
the same magic rule files used by the `file` command, with no `unsafe`
code.

[![CI](https://img.shields.io/github/actions/workflow/status/qjerome/magic-rs/rust.yml?style=for-the-badge)](https://github.com/qjerome/magic-rs/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/pure-magic.svg?style=for-the-badge)](https://crates.io/crates/pure-magic)
[![docs.rs](https://img.shields.io/docsrs/pure-magic?style=for-the-badge)](https://docs.rs/pure-magic)
[![PyPI](https://img.shields.io/pypi/v/pure-magic-rs.svg?style=for-the-badge)](https://pypi.org/project/pure-magic-rs/)
[![License](https://img.shields.io/crates/l/pure-magic.svg?style=for-the-badge)](#license)

## Advantages over libmagic

- **Memory safety.** `pure-magic` is built with `#![forbid(unsafe_code)]`,
  so it can't be affected by the memory-corruption bugs (buffer overreads,
  use-after-free) that have periodically been reported against the C
  implementation. This matters in particular when scanning untrusted or
  adversarial input, which is `file`'s primary use case.
- **Self-contained, purpose-built binaries.** `magic-embed` compiles a
  chosen set of rule files directly into the binary at build time, via
  `include`/`exclude` paths. You can embed only the formats a given tool
  needs, instead of the entire `magdir`, which keeps the binary lean and
  removes the runtime dependency on `libmagic.so` or an external
  `magic.mgc` — and with it, the risk of a binary and its shared library
  disagreeing on the compiled database version.
- **Straightforward cross-compilation.** No C toolchain or autotools
  build step; anything `cargo` can target, `pure-magic` can target.
- **Native Python bindings.** `pure-magic-rs` ships as a self-contained
  wheel with an embedded database, with no `libmagic` system dependency
  to install or link against.

These come with a trade-off: see [Differences from
libmagic](#differences-from-libmagic) below for what's out of scope.

## Crates

| Crate | Description |
| --- | --- |
| [`pure-magic`](pure-magic/) | Core detection engine: parses magic rules and evaluates them against a byte stream. |
| [`magic-embed`](magic-embed/) | Procedural macro to compile a rule database into a binary at build time. |
| [`magic-db`](magic-db/) | Precompiled database built from the [rules](magic-db/src/magdir/) shipped with `file`. |
| [`wiza`](wiza/) | Command-line tool built on `pure-magic` and `magic-db`. |
| [`pure-magic-rs`](python/) | Python bindings for `pure-magic`, published on PyPI. |

## Getting started

Install the `wiza` CLI:

```sh
cargo install wiza
```

Identify a file:

```sh
$ wiza /bin/file
/bin/file source:elf strength:431 mime:application/x-pie-executable magic:ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV)
```

Python bindings are also available as [`pure-magic-rs`](python/) on PyPI,
with an embedded database so no separate rule files are needed:

```sh
pip install pure-magic-rs
```

```python
from pure_magic_rs import MagicDb

db = MagicDb()
result = db.best_magic_file("example.png")
print(result.message, result.mime_type)
```

See the [Python package README](python/README.md) for the full API.

## Rule compatibility

Most rules from the [`file`](https://github.com/file/file) repository work
unmodified against `pure-magic`. Two known gaps:

- **Ternary printf formatting is not supported.** Rules using
  `${cond?a:b}` need to be rewritten, which is usually straightforward.
  For example, this extract from the ELF rules:

  ```
  0	name		elf-le
  [...]
  >16	leshort		3		${x?pie executable:shared object},
  !:mime	application/x-${x?pie-executable:sharedlib}
  ```

- **DER/ASN.1 rules are not implemented.** They require dedicated parsing
  that `pure-magic` doesn't yet provide. Everything else behaves the same
  as `libmagic`.

## Differences from libmagic

`libmagic` goes beyond magic-byte matching for some formats. For ELF
binaries in particular, it parses program and section headers to report
the dynamic linker path, build ID, and similar metadata:

```
$ file /bin/ls
/bin/ls: ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, BuildID[sha1]=c988ae960e91ea3f9f7b9cbbc2e3e4ffc0353796, for GNU/Linux 4.4.0, stripped
```

```
$ wiza /bin/ls
/bin/ls source:elf strength:436 mime:application/x-pie-executable magic:ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV)
```

`magic-rs` intentionally stops at what the magic rule language can
express. Structural binary parsing — ELF section and program headers,
build metadata, and the equivalents for COFF, Mach-O, PE, PDF, and so
on — is out of scope. There's no principled place to draw that line once
you start walking binary structures for one format, so the boundary is
drawn at the rule format itself.

## Documentation

- [pure-magic](https://docs.rs/pure-magic)
- [magic-embed](https://docs.rs/magic-embed)
- [magic-db](https://docs.rs/magic-db)
- [Magic rule syntax (man page)](https://www.man7.org/linux/man-pages/man4/magic.4.html)

## Contributing

Bug reports should include a sample file (or a minimal reproduction)
demonstrating the mismatch with `libmagic`, and, where possible, a
suggested rule fix.

Contributions are also welcome for:

- Fixes and additions to the [rule database](./magic-db/src/magdir/)
- New file format support
- Performance improvements to rule evaluation

## License

Dual-licensed under [GPL-3.0](LICENSE-GPL) or [BSD-2-Clause](LICENSE-BSD),
at your option.

## Acknowledgments

- [file](https://github.com/file/file), whose magic rule format and
  database this project builds on.
- [@adulau](https://github.com/adulau) for supporting this work.
- My colleagues at [CIRCL](https://circl.lu/) for their patience
  listening to me talk about `pure-magic` almost every day since I
  started this project.
