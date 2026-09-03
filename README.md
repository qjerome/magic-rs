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
- **Single-pass, structured results.** One `first_magic_file`/
  `first_magic_slice` call returns a `Magic` with message, MIME type,
  extensions, and strength together. `libmagic`'s C API ties these to
  mutually exclusive flags (`MAGIC_NONE`, `MAGIC_MIME_TYPE`,
  `MAGIC_EXTENSION`) — getting all three means reconfiguring the cookie
  with `magic_setflags()` and rescanning the same data 2-3 times.

These come with a trade-off: see [Differences from
libmagic](#differences-from-libmagic) below for what's out of scope.

## Benchmarks

### APIs

This benchmark measures the speed of `pure-magic` vs `libmagic`, via the
[`magic` crate](https://crates.io/crates/magic).

| Benchmark | API | pure_magic | libmagic | Speedup |
| --- | --- | --- | --- | --- |
| single_large_file | file | **1.64 ms** | 4.17 ms | pure_magic 2.54x |
| medium_file | file | **190.23 µs** | 318.31 µs | pure_magic 1.67x |
| many_small_files_with_ext | file | **96.26 ms** | 287.78 ms | pure_magic 2.99x |
| many_small_files_no_ext | file | **236.69 ms** | 290.14 ms | pure_magic 1.23x |
| single_large_file | buffer | **108.70 µs** | 254.24 µs | pure_magic 2.34x |
| medium_file | buffer | **44.68 µs** | 247.55 µs | pure_magic 5.54x |
| many_small_files_with_ext | buffer | **82.36 ms** | 268.93 ms | pure_magic 3.27x |
| many_small_files_no_ext | buffer | **228.93 ms** | 289.25 ms | pure_magic 1.26x |

### CLI

CLI benchmarks were run with [`hyperfine`](https://github.com/sharkdp/hyperfine),
across corpus sizes sampled from a broad range of file types found in
the user's home directory.

|  files |        wiza (ms) |        file (ms) |  file/wiza | wiza ms/file | file ms/file | faster |
| -------|------------------|------------------|------------|--------------|--------------|------- |
|     50 |       61.1 ± 2.3 |   **38.0 ± 1.5** |      0.62x |        1.222 |    **0.760** |   file |
|    100 |       89.4 ± 5.3 |   **65.9 ± 3.6** |      0.74x |        0.894 |    **0.659** |   file |
|    250 |  **156.1 ± 6.4** |      176.9 ± 6.1 |      1.13x |    **0.624** |        0.708 |   wiza |
|    500 | **280.3 ± 13.4** |     473.6 ± 26.2 |      1.69x |    **0.561** |        0.947 |   wiza |
|   1000 | **492.5 ± 27.3** |     759.6 ± 29.9 |      1.54x |    **0.492** |        0.760 |   wiza |
|   2000 | **993.1 ± 55.9** |    1530.3 ± 78.6 |      1.54x |    **0.497** |        0.765 |   wiza |
|   4000 | **2552.0 ± 86.0** |    3678.0 ± 59.0 |      1.44x |    **0.638** |        0.919 |   wiza |
|   8000 | **6100.0 ± 69.0** |   8501.0 ± 102.0 |      1.39x |    **0.762** |        1.063 |   wiza |


What explains the crossover between `wiza` and `file`? `file` `mmap`s its
precompiled `.mgc` database and casts the mapped bytes directly into its
rule structs. `wiza` instead deserializes its entire embedded
database into owned Rust values up front, so every invocation pays that
full cost before it can evaluate a single rule, even when scanning just
one file. Once that fixed cost is amortized across the corpus (around
250 files in the table above), `wiza` is consistently faster. This
matches the API benchmarks above, which don't include database loading
at all, since the database is built once outside the timed loop.

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

- **Ternary message formatting (`${x?a:b}`) is not supported.** `x` is
  not a general variable — it's the single hardcoded case libmagic's
  `varexpand()` recognizes, meaning "does the scanned file have its
  Unix execute permission bit set on disk". That's a property of the
  file's metadata, not its content, and `pure-magic` deliberately
  doesn't check it: it would only ever be meaningful when scanning a
  real file from disk (never for in-memory buffers), for a
  single-purpose heuristic found in exactly one rule upstream (`elf`).
  Rules using it need to be rewritten to drop the condition and
  pick one fixed message.

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

A second, different category is `libmagic`'s `${x?a:b}` ternary message
formatting (see [Rule compatibility](#rule-compatibility)).
In ternary formatting `x` reads the scanned file's Unix execute
permission bit from `stat()`, so it isn't a function of the file's bytes
at all. The same content can report two different types depending on how
it's scanned: `chmod +x` changes the answer for the same path, and
`libmagic`'s buffer-scanning API (`magic_buffer()` for in-memory data)
never has a real file descriptor to `stat()`, so it can never find the
permission bit. We believe a magic byte matcher shouldn't give a
different answer for identical bytes based on external state, so
`pure-magic` doesn't replicate it.

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
