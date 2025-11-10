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
use magic_embed::magic_embed;
use magic_rs::{Magic, MagicDb, MagicFile};
use serde_derive::Serialize;
use tracing::{debug, error, info};
use tracing_subscriber::EnvFilter;

#[magic_embed(include=["magic/src/magdir"], exclude=["magic/src/magdir/der"])]
struct EmbeddedMagicDb;

#[derive(Parser)]
struct Cli {
    #[clap(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Scan paths and attempt at classifying files
    Scan(ScanOpt),
    /// Compile magic rules into binary format
    Compile(CompileOpt),
    /// Show rules' information
    Show(ShowOpt),
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
    files: Vec<PathBuf>,
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
    strength: Option<u64>,
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
            source: m.source().cloned(),
            magic: m.message(),
            mime_type: m.mime_type(),
            creator_code: m.creator_code().cloned(),
            strength: m.strength(),
            extensions: m.exts(),
        }
    }
}

fn db_load_rules(db: &mut MagicDb, rules: &[PathBuf], silent: bool) -> Result<(), magic_rs::Error> {
    for rule in rules {
        if rule.is_dir() {
            let walker = WalkOptions::new()
                .files()
                .max_depth(0)
                .sort(true)
                .walk(rule);
            for p in walker.flatten() {
                info!("loading magic rule: {}", p.to_string_lossy());
                let magic = MagicFile::open(&p).inspect_err(|e| {
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
            db.load(MagicFile::open(rule)?)?;
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
                EmbeddedMagicDb::open().unwrap()
            };

            for r in db.rules() {
                println!(
                    "{}:{} score={} text={} extensions={}",
                    r.source().unwrap_or("unknown"),
                    r.line(),
                    r.score(),
                    r.is_text(),
                    {
                        let mut v: Vec<&str> =
                            r.extensions().into_iter().map(|s| s.as_ref()).collect();
                        v.sort();
                        v.join("/")
                    }
                )
            }
        }
        Some(Command::Scan(o)) => {
            let db = if let Some(db) = o.db {
                let start = Instant::now();
                let db = MagicDb::deserialize_reader(&mut File::open(&db).map_err(|e| {
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
                EmbeddedMagicDb::open().unwrap()
            };

            for item in o.files {
                let mut wo = WalkOptions::new();
                wo.files().sort(true);

                if !o.recursive {
                    wo.max_depth(0);
                }

                for f in wo.walk(item).flatten() {
                    debug!("scanning file: {}", f.to_string_lossy());
                    let Ok(mut file) = File::open(&f).inspect_err(|e| {
                        error!("failed to open file={}: {e}", f.to_string_lossy())
                    }) else {
                        continue;
                    };

                    if o.all {
                        let Ok(mut magics) = db.magic_all(&mut file).inspect_err(|e| {
                            error!("failed to get magic file={}: {e}", f.to_string_lossy())
                        }) else {
                            continue;
                        };

                        // we sort only if needed
                        magics.sort_by(|a, b| b.0.cmp(&a.0));

                        if o.json {
                            let amr = SerAllMagicResult {
                                path: f,
                                magics: magics
                                    .iter()
                                    .map(|(_, m)| {
                                        SerMagicResult::from_path_and_magic(
                                            Option::<PathBuf>::None,
                                            m,
                                        )
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
                            for (strength, magic) in magics {
                                println!(
                                    "file:{} source:{} strength:{strength} mime:{} magic:{}",
                                    f.to_string_lossy(),
                                    magic.source().unwrap_or(&Cow::Borrowed("unknown")),
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
                                "file:{} source:{} strength:{} mime:{} magic:{}",
                                f.to_string_lossy(),
                                magic.source().unwrap_or(&Cow::Borrowed("none")),
                                magic.strength().unwrap_or_default(),
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
                        let magic = MagicFile::open(&p)
                            .inspect_err(|e| error!("{} {e}", p.to_string_lossy()));
                        // FIXME: we ignore error for the moment
                        if magic.is_err() {
                            continue;
                        }
                        let _ = db.load(magic?)?;
                    }
                } else {
                    info!("loading magic rule: {}", rule.to_string_lossy());
                    db.load(MagicFile::open(rule)?)?;
                }
            }

            info!("Time to parse rule files: {:?}", start.elapsed());
            start = Instant::now();

            let mut o = File::create(&o.output)
                .map_err(|e| anyhow!("failed at creating {}: {e}", o.output.to_string_lossy()))?;

            let bytes = db
                .serialize()
                .map_err(|e| anyhow!("failed to serialize database: {e}"))?;

            o.write_all(&bytes)
                .map_err(|e| anyhow!("failed to save database: {e}"))?;

            info!("Time to serialize and save database: {:?}", start.elapsed());
        }
        None => {}
    }

    Ok(())
}
