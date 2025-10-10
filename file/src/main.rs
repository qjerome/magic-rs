use std::{borrow::Cow, fs::File, io::Write, path::PathBuf, time::Instant};

use anyhow::anyhow;
use clap::{CommandFactory, FromArgMatches, Parser, Subcommand, builder::styling};
use fs_walk::WalkOptions;
use lazy_cache::LazyCache;
use magic_rs::{FILE_BYTES_MAX, MagicDb, MagicFile};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
struct Cli {
    #[clap(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    Test(TestOpt),
    Parse(ParseOpt),
    Compile(CompileOpt),
}

#[derive(Debug, Parser)]
struct ParseOpt {
    rules: Vec<PathBuf>,
}

#[derive(Debug, Parser)]
struct TestOpt {
    /// Hide log messages
    #[arg(short, long)]
    silent: bool,
    /// Show all magic rules matching
    /// not only the first one
    #[arg(long)]
    all: bool,
    /// Enable file extension acceleration. Matches first the
    /// rules where file extension is defined.
    #[arg(long)]
    no_accel: bool,
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
        Some(Command::Parse(o)) => {
            for f in o.rules {
                let _ = MagicFile::open(f)?;

                /*for r in m.rules() {
                    println!("# new rule");
                    for e in r.entries() {
                        println!("{:?}", e)
                    }
                    println!()

                }

                for d in m.dep_rules() {
                    println!("# dependency rule name: {}", d.name());
                    for e in d.rule().entries() {
                        println!("{:?}", e)
                    }
                    println!()
                }*/
            }
        }

        Some(Command::Test(o)) => {
            let db = if let Some(db) = o.db {
                let start = Instant::now();
                let db = MagicDb::deserialize_reader(&mut File::open(&db).map_err(|e| {
                    anyhow!("failed to open database file {}: {e}", db.to_string_lossy())
                })?)
                .map_err(|e| anyhow!("failed to deserialize database: {e}"))?;
                println!("Time to deserialize database: {:?}", start.elapsed());
                db
            } else {
                let mut db = MagicDb::new();

                let start = Instant::now();
                for rule in o.rules {
                    if rule.is_dir() {
                        let walker = WalkOptions::new()
                            .files()
                            .max_depth(1)
                            .sort(true)
                            .walk(rule);
                        for p in walker.flatten() {
                            info!("loading magic rule: {}", p.to_string_lossy());
                            let magic = MagicFile::open(&p).inspect_err(|e| {
                                if !o.silent {
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
                println!("Time to parse rule files: {:?}", start.elapsed());
                db
            };

            for item in o.files {
                let mut wo = WalkOptions::new();
                wo.files().sort(true);

                if !o.recursive {
                    wo.max_depth(1);
                }

                for f in wo.walk(item).flatten() {
                    info!("scanning file: {}", f.to_string_lossy());
                    let start = Instant::now();
                    let Ok(mut haystack) = LazyCache::<File>::open(&f)
                        .inspect_err(|e| error!("cannot open file={}: {e}", f.to_string_lossy()))
                        .map(|lc| lc.with_hot_cache(2 * FILE_BYTES_MAX).unwrap())
                    else {
                        continue;
                    };

                    if o.all {
                        let Ok(mut magics) = db.magic_all(&mut haystack).inspect_err(|e| {
                            error!("failed to get magic file={}: {e}", f.to_string_lossy())
                        }) else {
                            continue;
                        };
                        // we sort only if needed
                        magics.sort_by(|a, b| b.0.cmp(&a.0));
                        for (strength, magic) in magics {
                            println!(
                                "file:{} source:{} strength:{strength} mime:{} magic:{}",
                                f.to_string_lossy(),
                                magic.source().unwrap_or(&Cow::Borrowed("unknown")),
                                magic.mimetype(),
                                magic.message()
                            )
                        }
                    } else {
                        let ext: Option<&str> = if o.no_accel {
                            None
                        } else {
                            // files without extension must set have an empty string extension to benefit from
                            // file extension acceleration
                            Some(f.extension().and_then(|e| e.to_str()).unwrap_or_default())
                        };

                        let Ok(magic) = db.magic_first(&mut haystack, ext).inspect_err(|e| {
                            error!("failed to get magic file={}: {e}", f.to_string_lossy())
                        }) else {
                            continue;
                        };

                        let elapsed = start.elapsed();
                        println!(
                            "time_ns:{:?} time:{:?} file:{} source:{} strength:{} mime:{} magic:{}",
                            elapsed.as_nanos(),
                            elapsed,
                            f.to_string_lossy(),
                            magic.source().unwrap_or(&Cow::Borrowed("none")),
                            magic.strength().unwrap_or_default(),
                            magic.mimetype(),
                            magic.message()
                        )
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

            println!("Time to parse rule files: {:?}", start.elapsed());
            start = Instant::now();

            let mut o = File::create(&o.output)
                .map_err(|e| anyhow!("failed at creating {}: {e}", o.output.to_string_lossy()))?;

            let bytes = db
                .serialize()
                .map_err(|e| anyhow!("failed to serialize database: {e}"))?;

            o.write_all(&bytes)
                .map_err(|e| anyhow!("failed to save database: {e}"))?;

            println!("Time to serialize and save database: {:?}", start.elapsed());
        }
        None => {}
    }

    Ok(())
}
