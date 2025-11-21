[![Crates.io Version](https://img.shields.io/crates/v/wiza?style=for-the-badge)](https://crates.io/crates/wiza)

<!-- cargo-rdme start -->

# wiza - The File Wizard ✨

**What is zat?** Now you know.
A Rust-powered alternative to the `file` command, with JSON support, custom rules, and a sprinkle of magic.

## Features

- **Fast file type detection** using magic numbers and file signatures
- **Polyglot file detection** - identify files that contain multiple valid formats
- **JSON output** for programmatic use
- **Recursive directory scanning** with `-R` flag
- **File extension based acceleration** for faster matching
- **Embedded database** with common file type rules
- **Rule compilation** with the `compile` subcommand

## Installation

```sh
cargo install wiza
```

## Usage

### Basic file identification

```sh
$ wiza /bin/file
/bin/file source:elf strength:431 mime:application/x-pie-executable magic:ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV)
```

### Polyglot file detection

`wiza` can detect polyglot files - files that are valid in multiple formats.

To detect polyglot files, use the `--all` flag to show all matching rules:

```sh
$ wiza --all private/polyglot-database/files/resume_iso.pdf
resume_iso.pdf source:filesystems strength:495 mime:application/x-iso9660-image magic:ISO 9660 CD-ROM filesystem data (DOS/MBR boot sector) 'ISOIMAGE' (bootable)
resume_iso.pdf source:filesystems strength:155 mime:application/octet-stream magic:DOS/MBR boot sector
resume_iso.pdf source:pdf strength:141 mime:application/pdf magic:PDF document, version ë.ë
resume_iso.pdf source:filesystems strength:137 mime:application/octet-stream magic:DOS/MBR boot sector
resume_iso.pdf source:hardcoded strength:0 mime:application/octet-stream magic:data
```

### JSON output

```sh
$ wiza -j /bin/file | jq '.'
{
   "path": "/bin/file",
   "source": "elf",
   "magic": "ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV)",
   "mime-type": "application/x-pie-executable",
   "creator-code": null,
   "strength": 431,
   "extensions": [
       "so"
   ]
}
```

### Compile custom rules

```sh
$ wiza compile --rules custom_rules/ --output my_rules.db
$ wiza --db my_rules.db custom_file
```

## Backward compatibility with `file` / `libmagic` rule format

Create magic rule files following the standard [magic](https://www.man7.org/linux/man-pages/man4/magic.4.html) format. Example:

```text
# my_rules/xyz
0       string          MYTYPE      My custom file type
>4      byte            1           version 1
>4      byte            2           version 2
```

Then compile with:

```sh
wiza compile --rules my_rules/ --output my_rules.db
```

## Why the name "wiza"?

`wiza` is a playful combination of:
- **"Wizard"**: Ties to the "magic" of file type detection (using magic numbers)
- **"What is zat?"**: A conversational way to ask "What is that?" (for French speakers)

## License

This project is licensed under the **GPL-3.0 License**.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Inspired by the Unix `file` command but with modern improvements

<!-- cargo-rdme end -->
