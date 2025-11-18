//! # wiza - The File Wizard ✨
//!
//! **What is zat?** Now you know.
//! A Rust-powered alternative to the `file` command, with JSON support, custom rules, and a sprinkle of magic.
//!
//! ## Features
//!
//! - **Fast file type detection** using magic numbers and file signatures
//! - **Polyglot file detection** - identify files that contain multiple valid formats
//! - **JSON output** for programmatic use
//! - **Recursive directory scanning** with `-R` flag
//! - **File extension based acceleration** for faster matching
//! - **Embedded database** with common file type rules
//! - **Rule compilation** with the `compile` subcommand
//!
//! ## Installation
//!
//! ```sh
//! cargo install wiza
//! ```
//!
//! ## Usage
//!
//! ### Basic file identification
//!
//! ```sh
//! $ wiza /bin/file
//! /bin/file source:elf strength:431 mime:application/x-pie-executable magic:ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV)
//! ```
//!
//! ### Polyglot file detection
//!
//! `wiza` can detect polyglot files - files that are valid in multiple formats.
//!
//! To detect polyglot files, use the `--all` flag to show all matching rules:
//!
//! ```sh
//! $ wiza --all private/polyglot-database/files/resume_iso.pdf
//! resume_iso.pdf source:filesystems strength:495 mime:application/x-iso9660-image magic:ISO 9660 CD-ROM filesystem data (DOS/MBR boot sector) 'ISOIMAGE' (bootable)
//! resume_iso.pdf source:filesystems strength:155 mime:application/octet-stream magic:DOS/MBR boot sector
//! resume_iso.pdf source:pdf strength:141 mime:application/pdf magic:PDF document, version ë.ë
//! resume_iso.pdf source:filesystems strength:137 mime:application/octet-stream magic:DOS/MBR boot sector
//! resume_iso.pdf source:hardcoded strength:0 mime:application/octet-stream magic:data
//! ```
//!
//! ### JSON output
//!
//! ```sh
//! $ wiza -j /bin/file | jq '.'
//! {
//!    "path": "/bin/file",
//!    "source": "elf",
//!    "magic": "ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV)",
//!    "mime-type": "application/x-pie-executable",
//!    "creator-code": null,
//!    "strength": 431,
//!    "extensions": [
//!        "so"
//!    ]
//! }
//! ```
//!
//! ### Compile custom rules
//!
//! ```sh
//! $ wiza compile --rules custom_rules/ --output my_rules.db
//! $ wiza --db my_rules.db custom_file
//! ```
//!
//! ## Backward compatibility with `file` / `libmagic` rule format
//!
//! Create magic rule files following the standard [magic](https://www.man7.org/linux/man-pages/man4/magic.4.html) format. Example:
//!
//! ```text
//! # my_rules/xyz
//! 0       string          MYTYPE      My custom file type
//! >4      byte            1           version 1
//! >4      byte            2           version 2
//! ```
//!
//! Then compile with:
//!
//! ```sh
//! wiza compile --rules my_rules/ --output my_rules.db
//! ```
//!
//! ## Why the name "wiza"?
//!
//! `wiza` is a playful combination of:
//! - **"Wizard"**: Ties to the "magic" of file type detection (using magic numbers)
//! - **"What is zat?"**: A conversational way to ask "What is that?" (for French speakers)
//!
//! ## License
//!
//! This project is licensed under the **GPL-3.0 License**.
//!
//! ## Contributing
//!
//! Contributions are welcome! Please open an issue or submit a pull request.
//!
//! ## Acknowledgments
//!
//! - Inspired by the Unix `file` command but with modern improvements

use std::{
    borrow::Cow,
    collections::HashSet,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::anyhow;
use clap::{CommandFactory, FromArgMatches, Parser, Subcommand, builder::styling};
use fs_walk::WalkOptions;
use pure_magic::{Magic, MagicDb, MagicSource};
use serde_derive::Serialize;
use tracing::{debug, error, info};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
struct Cli {
    #[clap(subcommand)]
    command: Option<Command>,
    /// Hide log messages
    #[arg(short, long)]
    silent: bool,
    /// Show all magic rules matching
    /// not only the first one
    #[arg(short, long)]
    all: bool,
    /// Disable file extension acceleration (matches first
    /// rules where the file extension is defined)
    #[arg(long)]
    no_accel: bool,
    /// Output the result as JSON instead of text
    #[arg(short, long)]
    json: bool,
    /// Path to magic rule file or directory
    /// containing rule files to compile
    #[arg(short, long)]
    rules: Vec<PathBuf>,
    /// Walk directories recursively while scanning
    #[arg(short = 'R', long)]
    recursive: bool,
    /// Path to compiled rules database
    #[arg(short, long)]
    db: Option<PathBuf>,
    /// Files or directory to scan
    paths: Vec<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compile magic rules into binary format
    Compile(CompileOpt),
    /// Show rules' information
    Show(ShowOpt),
}

impl Command {
    fn scan(o: ScanOpt) -> Result<(), anyhow::Error> {
        let db = if let Some(db) = o.db {
            let start = Instant::now();
            let db = MagicDb::deserialize(&mut File::open(&db).map_err(|e| {
                anyhow!("failed to open database file {}: {e}", db.to_string_lossy())
            })?)
            .map_err(|e| anyhow!("failed to deserialize database: {e}"))?;
            info!("Time to deserialize database: {:?}", start.elapsed());
            db
        } else if !o.rules.is_empty() {
            let mut db = MagicDb::new();

            let start = Instant::now();
            db_load_rules(&mut db, &o.rules, o.silent)?;
            info!("Time to parse rule files: {:?}", start.elapsed());
            db
        } else {
            magic_db::CompiledDb::open()
                .map_err(|e| anyhow!("failed to open embedded database: {e}"))?
        };

        for item in o.paths {
            let mut wo = WalkOptions::new();
            wo.files().sort(true);

            if !o.recursive {
                wo.max_depth(0);
            }

            for f in wo.walk(item).flatten() {
                debug!("scanning file: {}", f.to_string_lossy());
                let Ok(mut file) = File::open(&f)
                    .inspect_err(|e| error!("failed to open file={}: {e}", f.to_string_lossy()))
                else {
                    continue;
                };

                if o.all {
                    let Ok(magics) = db.magic_all(&mut file).inspect_err(|e| {
                        error!("failed to get magic file={}: {e}", f.to_string_lossy())
                    }) else {
                        continue;
                    };

                    if o.json {
                        let amr = SerAllMagicResult {
                            path: f,
                            magics: magics
                                .iter()
                                .map(|m| {
                                    SerMagicResult::from_path_and_magic(Option::<PathBuf>::None, m)
                                })
                                .collect(),
                        };

                        let Ok(json) = serde_json::to_string(&amr)
                            .inspect_err(|e| error!("failed to serialize magic: {e}"))
                        else {
                            continue;
                        };

                        println!("{json}")
                    } else {
                        for magic in magics {
                            println!(
                                "{} source:{} strength:{} mime:{} magic:{}",
                                f.to_string_lossy(),
                                magic.source().unwrap_or(&Cow::Borrowed("unknown")),
                                magic.strength(),
                                magic.mime_type(),
                                magic.message()
                            )
                        }
                    }
                } else {
                    let ext: Option<&str> = if o.no_accel {
                        None
                    } else {
                        f.extension().and_then(|e| e.to_str())
                    };

                    let Ok(magic) = db.magic_first(&mut file, ext).inspect_err(|e| {
                        error!("failed to get magic file={}: {e}", f.to_string_lossy())
                    }) else {
                        continue;
                    };

                    if !o.json {
                        println!(
                            "{} source:{} strength:{} mime:{} magic:{}",
                            f.to_string_lossy(),
                            magic.source().unwrap_or(&Cow::Borrowed("none")),
                            magic.strength(),
                            magic.mime_type(),
                            magic.message()
                        )
                    } else {
                        let mr = SerMagicResult::from_path_and_magic(Some(f), &magic);
                        let Ok(json) = serde_json::to_string(&mr)
                            .inspect_err(|e| error!("failed to serialize magic: {e}"))
                        else {
                            continue;
                        };

                        println!("{json}");
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Parser)]
struct ScanOpt {
    /// Hide log messages
    #[arg(short, long)]
    silent: bool,
    /// Show all magic rules matching
    /// not only the first one
    #[arg(short, long)]
    all: bool,
    /// Enable file extension acceleration. Matches first the
    /// rules where file extension is defined.
    #[arg(long)]
    no_accel: bool,
    /// Output the result as JSON instead of text
    #[arg(short, long)]
    json: bool,
    /// Path to magic rule file or directory
    /// containing rule files to compile
    #[arg(short, long)]
    rules: Vec<PathBuf>,
    /// Walk directories recursively while scanning
    #[arg(short = 'R', long)]
    recursive: bool,
    /// Path to compiled rules database
    #[arg(short, long)]
    db: Option<PathBuf>,
    /// Files or directory to scan
    paths: Vec<PathBuf>,
}

#[derive(Debug, Parser)]
struct CompileOpt {
    /// Path to magic rule file or directory
    /// containing rule files to compile
    #[arg(short, long)]
    rules: Vec<PathBuf>,
    /// Path to compiled rules database
    output: PathBuf,
}

#[derive(Debug, Parser)]
struct ShowOpt {
    /// Path to magic rule file or directory
    /// containing rule files to use
    #[arg(short, long)]
    rules: Vec<PathBuf>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
struct SerMagicResult<'m> {
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<PathBuf>,
    source: Option<Cow<'m, str>>,
    magic: String,
    mime_type: &'m str,
    creator_code: Option<Cow<'m, str>>,
    strength: u64,
    extensions: &'m HashSet<Cow<'m, str>>,
}

#[derive(Debug, Serialize)]
struct SerAllMagicResult<'m> {
    path: PathBuf,
    magics: Vec<SerMagicResult<'m>>,
}

impl<'m> SerMagicResult<'m> {
    fn from_path_and_magic<P: AsRef<Path>>(p: Option<P>, m: &'m Magic<'_>) -> Self {
        Self {
            path: p.map(|p| p.as_ref().to_path_buf()),
            source: m.source().map(|s| s.into()),
            magic: m.message(),
            mime_type: m.mime_type(),
            creator_code: m.creator_code().map(|s| s.into()),
            strength: m.strength(),
            extensions: m.extensions(),
        }
    }
}

fn db_load_rules(
    db: &mut MagicDb,
    rules: &[PathBuf],
    silent: bool,
) -> Result<(), pure_magic::Error> {
    for rule in rules {
        if rule.is_dir() {
            let walker = WalkOptions::new()
                .files()
                .max_depth(0)
                .sort(true)
                .walk(rule);
            for p in walker.flatten() {
                info!("loading magic rule: {}", p.to_string_lossy());
                let magic = MagicSource::open(&p).inspect_err(|e| {
                    if !silent {
                        error!("{} {e}", p.to_string_lossy())
                    }
                });
                // FIXME: we ignore error for the moment
                if magic.is_err() {
                    continue;
                }
                let _ = db.load(magic?)?;
            }
        } else {
            info!("loading magic rule: {}", rule.to_string_lossy());
            db.load(MagicSource::open(rule)?)?;
        }
    }

    Ok(())
}

fn main() -> Result<(), anyhow::Error> {
    let c = {
        let c: clap::Command = Cli::command();
        let styles = styling::Styles::styled()
            .header(styling::AnsiColor::Green.on_default() | styling::Effects::BOLD)
            .usage(styling::AnsiColor::Green.on_default() | styling::Effects::BOLD)
            .literal(styling::AnsiColor::Blue.on_default() | styling::Effects::BOLD)
            .placeholder(styling::AnsiColor::Cyan.on_default());

        c.styles(styles).help_template(
            r#"{about-with-newline}
{author-with-newline}
{usage-heading} {usage}
            
{all-args}"#,
        )
    };

    let cli: Cli = Cli::from_arg_matches(&c.get_matches())?;

    // Initialize the tracing subscriber
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    match cli.command {
        Some(Command::Show(o)) => {
            let db = if !o.rules.is_empty() {
                let mut db = MagicDb::new();
                db_load_rules(&mut db, &o.rules, false)?;
                db
            } else {
                magic_db::CompiledDb::open()
                    .map_err(|e| anyhow!("failed to open embedded database: {e}"))?
            };

            for r in db.rules() {
                println!(
                    "{}:{} score={} text={} extensions={}",
                    r.source().unwrap_or("unknown"),
                    r.line(),
                    r.score(),
                    r.is_text(),
                    {
                        let mut v: Vec<&str> = r.extensions().iter().map(|s| s.as_ref()).collect();
                        v.sort();
                        v.join("/")
                    }
                )
            }
        }

        Some(Command::Compile(o)) => {
            let mut db = MagicDb::new();

            let mut start = Instant::now();
            for rule in o.rules {
                if rule.is_dir() {
                    let walker = WalkOptions::new()
                        .files()
                        .max_depth(1)
                        .sort(true)
                        .walk(rule);
                    for p in walker.flatten() {
                        info!("loading magic rule: {}", p.to_string_lossy());
                        let magic = MagicSource::open(&p)
                            .inspect_err(|e| error!("{} {e}", p.to_string_lossy()));
                        // FIXME: we ignore error for the moment
                        if magic.is_err() {
                            continue;
                        }
                        let _ = db.load(magic?)?;
                    }
                } else {
                    info!("loading magic rule: {}", rule.to_string_lossy());
                    db.load(MagicSource::open(rule)?)?;
                }
            }

            info!("Time to parse rule files: {:?}", start.elapsed());
            start = Instant::now();

            let mut o = File::create(&o.output)
                .map_err(|e| anyhow!("failed at creating {}: {e}", o.output.to_string_lossy()))?;

            let mut bytes = vec![];
            db.serialize(&mut bytes)
                .map_err(|e| anyhow!("failed to serialize database: {e}"))?;

            o.write_all(&bytes)
                .map_err(|e| anyhow!("failed to save database: {e}"))?;

            info!("Time to serialize and save database: {:?}", start.elapsed());
        }

        None => Command::scan(ScanOpt {
            silent: cli.silent,
            all: cli.all,
            no_accel: cli.no_accel,
            json: cli.json,
            rules: cli.rules,
            recursive: cli.recursive,
            db: cli.db,
            paths: cli.paths,
        })?,
    }

    Ok(())
}
