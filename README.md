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
- **A real "give me the best answer" mode.** For speed, both tools
  normally stop at the first rule that matches, which is usually but
  not always the most accurate one. `pure-magic`'s `best_magic` checks
  every rule and returns the one that actually fits best.

These come with a trade-off: see [Differences from
libmagic](#differences-from-libmagic) below for what's out of scope.

## Benchmarks

### APIs

This benchmark measures the speed of `pure-magic` vs `libmagic`, via the
[`magic` crate](https://crates.io/crates/magic).

| Benchmark | API | pure_magic (first) | libmagic | Speedup | pure_magic (best) |
| --- | --- | --- | --- | --- | --- |
| single_large_file | file | **2.05 ms** | 4.78 ms | pure_magic 2.33x | 2.08 ms (1.0x first) |
| medium_file | file | **147.78 µs** | 180.48 µs | pure_magic 1.22x | 186.88 µs (1.3x first) |
| many_small_files_with_ext | file | **53.20 ms** | 161.95 ms | pure_magic 3.04x | 175.81 ms (3.3x first) |
| many_small_files_no_ext | file | **155.79 ms** | 162.12 ms | pure_magic 1.04x | 188.07 ms (1.2x first) |
| single_large_file | buffer | **71.50 µs** | 134.45 µs | pure_magic 1.88x | 92.38 µs (1.3x first) |
| medium_file | buffer | **28.21 µs** | 124.24 µs | pure_magic 4.40x | 44.72 µs (1.6x first) |
| many_small_files_with_ext | buffer | **42.75 ms** | 144.88 ms | pure_magic 3.39x | 143.78 ms (3.4x first) |
| many_small_files_no_ext | buffer | **135.70 ms** | 145.14 ms | pure_magic 1.07x | 155.11 ms (1.1x first) |

### CLI

CLI benchmarks were run with [`hyperfine`](https://github.com/sharkdp/hyperfine),
across corpus sizes sampled from a broad range of file types found in
the user's home directory.

|  files |        wiza (ms) |        file (ms) |  file/wiza | wiza ms/file | file ms/file | faster |
| -------|------------------|------------------|------------|--------------|--------------|------- |
|     50 |       61.9 ± 4.1 |   **36.1 ± 2.7** |      0.58x |        1.238 |    **0.722** |   file |
|    100 |       66.1 ± 3.0 |   **61.7 ± 4.6** |      0.93x |        0.661 |    **0.617** |   file |
|    250 |   **90.5 ± 2.1** |      109.2 ± 2.4 |      1.21x |    **0.362** |        0.437 |   wiza |
|    500 |  **143.9 ± 2.0** |      215.0 ± 4.8 |      1.49x |    **0.288** |        0.430 |   wiza |
|   1000 | **267.3 ± 30.7** |      399.3 ± 7.0 |      1.49x |    **0.267** |        0.399 |   wiza |
|   2000 | **600.1 ± 28.5** |     775.9 ± 11.4 |      1.29x |    **0.300** |        0.388 |   wiza |
|   4000 | **980.5 ± 68.3** |    1596.9 ± 30.1 |      1.63x |    **0.245** |        0.399 |   wiza |
|   8000 | **1842.0 ± 6.0** |    3154.0 ± 35.0 |      1.71x |    **0.230** |        0.394 |   wiza |

What explains the crossover between `wiza` and `file`? `file` memory maps its
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
