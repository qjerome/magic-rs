use std::{fs::File, path::PathBuf, time::Instant};

use clap::{CommandFactory, FromArgMatches, Parser, Subcommand, builder::styling};
use fs_walk::WalkOptions;
use lazy_cache::LazyCache;
use magic_rs::{MagicDb, MagicFile};
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
    #[arg(long)]
    all: bool,
    #[arg(short, long)]
    rules: Vec<PathBuf>,
    files: Vec<PathBuf>,
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
        .init();

    match cli.command {
        Some(Command::Parse(o)) => {
            for f in o.rules {
                let m = MagicFile::open(f)?;

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
            let mut db = MagicDb::new();

            let start = Instant::now();
            for rule in o.rules {
                if rule.is_dir() {
                    let wo = WalkOptions::new().files().max_depth(1);
                    for p in wo.walk(rule).flatten() {
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
            println!("Time parse rule files: {:?}", start.elapsed());

            let start = Instant::now();
            for f in o.files {
                let mut haystack = LazyCache::<File>::open(&f, 4096, 4 << 20).unwrap();
                let mut magics = db.magic(&mut haystack)?;
                magics.sort_by(|a, b| b.0.cmp(&a.0));

                if o.all {
                    for (strength, magic) in magics {
                        println!(
                            "file:{} strength:{strength} mime:{} magic:{}",
                            f.to_string_lossy(),
                            magic.mimetype(),
                            magic.message()
                        )
                    }
                } else {
                    if let Some((strength, magic)) = magics.first() {
                        println!(
                            "file:{} strength:{strength} mime:{} magic:{}",
                            f.to_string_lossy(),
                            magic.mimetype(),
                            magic.message()
                        )
                    }
                }
            }
            println!("Time to scan file: {:?}", start.elapsed());
        }
        None => {}
    }

    Ok(())
}
