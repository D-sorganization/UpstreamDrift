//! `upstream-codemap` — standalone Rust CLI for indexing the repo.
//!
//! Subcommands mirror the Python `codemap.cli` surface so callers can swap
//! between the two without behaviour changes.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use rusqlite::params;
use serde::Serialize;

use upstream_codemap::{db, indexer, watcher};

#[derive(Parser, Debug)]
#[command(name = "upstream-codemap", about = "Repo-aware code map (Rust).")]
struct Cli {
    /// Repo root (default: auto-discover via git).
    #[arg(long, global = true)]
    repo: Option<PathBuf>,

    /// Verbose logging.
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// (Re)build the index.
    Rebuild(RebuildArgs),
    /// BM25 search across the symbol index.
    Search(SearchArgs),
    /// Find callers of a qualified symbol.
    #[command(name = "who-calls")]
    WhoCalls(WhoCallsArgs),
    /// Export the index as JSONL.
    Export(ExportArgs),
    /// Show repo index stats.
    Info,
    /// Watch the repo and re-index changed files.
    Watch(WatchArgs),
}

#[derive(Args, Debug)]
struct RebuildArgs {
    /// Only re-parse files changed since this git ref.
    #[arg(long)]
    since: Option<String>,
}

#[derive(Args, Debug)]
struct SearchArgs {
    /// Search terms.
    query: Vec<String>,
    /// Max hits to return.
    #[arg(short = 'k', default_value_t = 20)]
    k: u32,
    /// Filter by kind (function, class, method, ...).
    #[arg(long)]
    kind: Option<String>,
    /// JSON output.
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug)]
struct WhoCallsArgs {
    /// Qualified symbol name (e.g. Foo.bar).
    qualified: String,
    /// JSON output.
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug)]
struct ExportArgs {
    /// Output path (default: .codemap/exports/code_map.jsonl.gz).
    #[arg(long)]
    jsonl: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct WatchArgs {
    /// Debounce window in milliseconds.
    #[arg(long, default_value_t = 500)]
    debounce_ms: u64,
}

fn discover_repo(arg: Option<&Path>) -> PathBuf {
    if let Some(p) = arg {
        return p.to_path_buf();
    }
    if let Ok(out) = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return PathBuf::from(s);
            }
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let repo = discover_repo(cli.repo.as_deref());
    match cli.cmd {
        Cmd::Rebuild(a) => cmd_rebuild(&repo, a.since.as_deref()),
        Cmd::Search(a) => cmd_search(&repo, a),
        Cmd::WhoCalls(a) => cmd_who_calls(&repo, a),
        Cmd::Export(a) => cmd_export(&repo, a.jsonl),
        Cmd::Info => cmd_info(&repo),
        Cmd::Watch(a) => watcher::watch(&repo, Duration::from_millis(a.debounce_ms)),
    }
}

fn cmd_rebuild(repo: &Path, since: Option<&str>) -> Result<()> {
    let stats = indexer::rebuild(repo, since)?;
    println!(
        "indexed {} files ({} unchanged), {} symbols in {:.2}s",
        stats.files_parsed, stats.files_skipped_unchanged, stats.symbols_inserted, stats.elapsed_s,
    );
    if !stats.errors.is_empty() {
        eprintln!(
            "  {} errors (first: {})",
            stats.errors.len(),
            stats.errors[0]
        );
    }
    Ok(())
}

#[derive(Serialize)]
struct HitOut {
    score: f64,
    path: String,
    kind: String,
    name: String,
    qualified: String,
    sig: String,
    docstring: String,
    start_line: i64,
    end_line: i64,
}

fn cmd_search(repo: &Path, args: SearchArgs) -> Result<()> {
    let query = args.query.join(" ");
    let conn = db::open_db(repo)?;
    let mut sql = String::from(
        "SELECT s.path, s.kind, s.name, s.qualified, s.sig, s.docstring, \
         s.start_line, s.end_line, bm25(symbols_fts) AS score \
         FROM symbols s JOIN symbols_fts f ON f.rowid = s.id \
         WHERE symbols_fts MATCH ?1",
    );
    if args.kind.is_some() {
        sql.push_str(" AND s.kind = ?2");
    }
    sql.push_str(" ORDER BY score LIMIT ?");
    sql.push_str(if args.kind.is_some() { "3" } else { "2" });

    let mut stmt = conn.prepare(&sql)?;
    let k = args.k as i64;
    let hits: Vec<HitOut> = if let Some(kind) = &args.kind {
        stmt.query_map(params![query, kind, k], row_to_hit)?
            .flatten()
            .collect()
    } else {
        stmt.query_map(params![query, k], row_to_hit)?
            .flatten()
            .collect()
    };

    if args.json {
        let json = serde_json::to_string_pretty(&hits)?;
        println!("{json}");
    } else if hits.is_empty() {
        println!("(no matches)");
    } else {
        for h in &hits {
            println!("[{:7.2}] {:8} {}", h.score, h.kind, h.qualified);
            println!(
                "           {}:{}-{}  {}",
                h.path, h.start_line, h.end_line, h.sig
            );
        }
    }
    Ok(())
}

fn row_to_hit(r: &rusqlite::Row) -> rusqlite::Result<HitOut> {
    Ok(HitOut {
        path: r.get(0)?,
        kind: r.get(1)?,
        name: r.get(2)?,
        qualified: r.get(3)?,
        sig: r.get(4)?,
        docstring: r.get(5)?,
        start_line: r.get(6)?,
        end_line: r.get(7)?,
        score: r.get::<_, f64>(8)?,
    })
}

fn cmd_who_calls(repo: &Path, args: WhoCallsArgs) -> Result<()> {
    let conn = db::open_db(repo)?;
    // The Python implementation searches calls_out for the suffix match.
    // Mirror that: any symbol whose calls_out JSON array contains `qualified`
    // or its trailing identifier counts as a caller.
    let needle = args.qualified.clone();
    let short = needle.rsplit_once('.').map(|(_, x)| x).unwrap_or(&needle);
    let mut stmt = conn.prepare(
        "SELECT path, kind, qualified, start_line FROM symbols \
         WHERE calls_out LIKE ?1 OR calls_out LIKE ?2",
    )?;
    let pat1 = format!("%\"{}\"%", needle);
    let pat2 = format!("%\"{}\"%", short);
    let rows = stmt.query_map(params![pat1, pat2], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, i64>(3)?,
        ))
    })?;
    #[derive(Serialize)]
    struct Caller {
        path: String,
        kind: String,
        qualified: String,
        start_line: i64,
    }
    let callers: Vec<Caller> = rows
        .flatten()
        .map(|(path, kind, qualified, start_line)| Caller {
            path,
            kind,
            qualified,
            start_line,
        })
        .collect();
    if args.json {
        println!("{}", serde_json::to_string_pretty(&callers)?);
    } else if callers.is_empty() {
        println!("(no callers found)");
    } else {
        for c in &callers {
            println!("{:8} {}  {}:{}", c.kind, c.qualified, c.path, c.start_line);
        }
    }
    Ok(())
}

fn cmd_export(repo: &Path, out: Option<PathBuf>) -> Result<()> {
    let conn = db::open_db(repo)?;
    let out_path = out.unwrap_or_else(|| {
        repo.join(".codemap")
            .join("exports")
            .join("code_map.jsonl.gz")
    });
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let gz = out_path.extension().and_then(|s| s.to_str()) == Some("gz");
    let file = std::fs::File::create(&out_path)
        .with_context(|| format!("failed to create {}", out_path.display()))?;
    let mut writer: Box<dyn Write> = if gz {
        Box::new(BufWriter::new(flate2::write::GzEncoder::new(
            file,
            flate2::Compression::default(),
        )))
    } else {
        Box::new(BufWriter::new(file))
    };
    let mut stmt = conn.prepare(
        "SELECT id, path, kind, name, qualified, sig, docstring, \
         start_line, end_line, calls_out, hash FROM symbols",
    )?;
    let rows = stmt.query_map([], |r| {
        let id: i64 = r.get(0)?;
        let path: String = r.get(1)?;
        let kind: String = r.get(2)?;
        let name: String = r.get(3)?;
        let qualified: String = r.get(4)?;
        let sig: String = r.get(5)?;
        let docstring: String = r.get(6)?;
        let start_line: i64 = r.get(7)?;
        let end_line: i64 = r.get(8)?;
        let calls_out: String = r.get(9)?;
        let hash: String = r.get(10)?;
        Ok((
            id, path, kind, name, qualified, sig, docstring, start_line, end_line, calls_out, hash,
        ))
    })?;
    let mut n = 0u64;
    for row in rows.flatten() {
        let (id, path, kind, name, qualified, sig, docstring, sl, el, calls, hash) = row;
        let rec = serde_json::json!({
            "id": id,
            "path": path,
            "kind": kind,
            "name": name,
            "qualified": qualified,
            "sig": sig,
            "docstring": docstring,
            "start_line": sl,
            "end_line": el,
            "calls_out": calls,
            "hash": hash,
        });
        writeln!(writer, "{}", rec)?;
        n += 1;
    }
    writer.flush()?;
    println!("exported {n} symbols -> {}", out_path.display());
    Ok(())
}

fn cmd_info(repo: &Path) -> Result<()> {
    let info = indexer::repo_info(repo)?;
    println!("repo:       {}", info.repo_root);
    println!("files:      {}", info.files);
    println!("symbols:    {}", info.symbols);
    println!("db size:    {:.1} KiB", info.db_size_bytes as f64 / 1024.0);
    println!(
        "last cmt:   {}",
        info.last_commit.as_deref().unwrap_or("(unknown)")
    );
    println!("languages:");
    for (lang, n) in info.languages {
        println!("  {:10} {}", lang, n);
    }
    Ok(())
}
