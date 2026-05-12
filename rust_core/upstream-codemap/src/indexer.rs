//! Walk a repo, parse files, persist symbols into SQLite.
//!
//! Mirrors `codemap.indexer.rebuild` so the resulting `.codemap/index.db` is
//! interchangeable with the Python implementation's output.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use rayon::prelude::*;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::db;
use crate::parsers::{self, ParseResult};

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct RebuildStats {
    pub files_seen: u64,
    pub files_parsed: u64,
    pub files_skipped_unchanged: u64,
    pub symbols_inserted: u64,
    pub symbols_deleted: u64,
    pub elapsed_s: f64,
    pub errors: Vec<String>,
}

const DEFAULT_SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    "target",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".codemap",
    ".idea",
    ".vscode",
    "htmlcov",
    "site-packages",
];

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn hash_bytes(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}

fn current_commit(repo_root: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo_root)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn git_changed_files(repo_root: &Path, since: &str) -> Vec<String> {
    let out = std::process::Command::new("git")
        .args(["diff", "--name-only", &format!("{since}..HEAD")])
        .current_dir(repo_root)
        .output();
    let Ok(out) = out else { return vec![] };
    if !out.status.success() {
        return vec![];
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn walk_repo(repo_root: &Path) -> Vec<(PathBuf, String)> {
    let skip: HashSet<&str> = DEFAULT_SKIP_DIRS.iter().copied().collect();
    let mut builder = WalkBuilder::new(repo_root);
    builder
        .hidden(false)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(false)
        .ignore(true)
        .require_git(false)
        .filter_entry(move |e| {
            if let Some(name) = e.file_name().to_str() {
                if e.file_type().map(|t| t.is_dir()).unwrap_or(false) && skip.contains(name) {
                    return false;
                }
            }
            true
        });

    let mut out = Vec::new();
    for entry in builder.build().flatten() {
        let path = entry.path();
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        if parsers::language_for(path).is_none() {
            continue;
        }
        let Ok(rel) = path.strip_prefix(repo_root) else {
            continue;
        };
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        out.push((path.to_path_buf(), rel_str));
    }
    out
}

struct ParsedFile {
    rel: String,
    abs_path: PathBuf,
    file_hash: String,
    mtime: f64,
    size: u64,
    parsed: ParseResult,
    bytes: Vec<u8>,
}

fn parse_one(abs_path: &Path, rel: &str) -> Option<ParsedFile> {
    let bytes = std::fs::read(abs_path).ok()?;
    let file_hash = hash_bytes(&bytes);
    let parsed = parsers::dispatch(abs_path, &bytes)?;
    let (mtime, size) = match abs_path.metadata() {
        Ok(meta) => {
            let m = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs_f64())
                .unwrap_or_else(now_secs);
            (m, meta.len())
        }
        Err(_) => (now_secs(), bytes.len() as u64),
    };
    Some(ParsedFile {
        rel: rel.to_string(),
        abs_path: abs_path.to_path_buf(),
        file_hash,
        mtime,
        size,
        parsed,
        bytes,
    })
}

fn slice_lines(data: &[u8], start: u32, end: u32) -> Vec<u8> {
    let text = String::from_utf8_lossy(data);
    let lines: Vec<&str> = text.lines().collect();
    let s = (start as usize).saturating_sub(1);
    let e = (end as usize).min(lines.len());
    if s >= lines.len() || s >= e {
        return Vec::new();
    }
    lines[s..e].join("\n").into_bytes()
}

fn upsert_file(tx: &Connection, pf: &ParsedFile, stats: &mut RebuildStats) -> Result<()> {
    let rel = pf.rel.as_str();
    // Delete prior symbols.
    let deleted = tx.execute("DELETE FROM symbols WHERE path = ?1", params![rel])?;
    stats.symbols_deleted += deleted as u64;

    tx.execute(
        "INSERT OR REPLACE INTO files(\
             path, language, hash, mtime, size, imports, indexed_at\
         ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            rel,
            pf.parsed.language,
            pf.file_hash,
            pf.mtime,
            pf.size as i64,
            serde_json::to_string(&pf.parsed.imports).unwrap_or_else(|_| "[]".into()),
            now_secs(),
        ],
    )?;

    let mut stmt = tx.prepare(
        "INSERT INTO symbols(path, kind, name, qualified, sig, docstring, \
         start_line, end_line, calls_out, hash) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
    )?;
    for sym in &pf.parsed.symbols {
        let slice = slice_lines(&pf.bytes, sym.start_line, sym.end_line);
        let sym_hash = hash_bytes(&slice);
        stmt.execute(params![
            rel,
            sym.kind,
            sym.name,
            sym.qualified,
            sym.sig,
            sym.docstring,
            sym.start_line as i64,
            sym.end_line as i64,
            serde_json::to_string(&sym.calls_out).unwrap_or_else(|_| "[]".into()),
            sym_hash,
        ])?;
        stats.symbols_inserted += 1;
    }
    Ok(())
}

/// Index `repo_root` from scratch (or incrementally if `since` is given).
pub fn rebuild(repo_root: &Path, since: Option<&str>) -> Result<RebuildStats> {
    let start = Instant::now();
    let repo = repo_root
        .canonicalize()
        .unwrap_or_else(|_| repo_root.to_path_buf());

    let mut conn = db::open_db(&repo)?;
    let mut stats = RebuildStats::default();

    // Collect candidate (abs, rel) pairs.
    let candidates: Vec<(PathBuf, String)> = match since {
        Some(rev) => {
            let changed = git_changed_files(&repo, rev);
            if changed.is_empty() {
                walk_repo(&repo)
            } else {
                let mut pairs = Vec::new();
                for rel in changed {
                    let abs_p = repo.join(&rel);
                    if !abs_p.exists() {
                        let deleted =
                            conn.execute("DELETE FROM symbols WHERE path = ?1", params![rel])?;
                        conn.execute("DELETE FROM files WHERE path = ?1", params![rel])?;
                        stats.symbols_deleted += deleted as u64;
                        continue;
                    }
                    if parsers::language_for(&abs_p).is_none() {
                        continue;
                    }
                    pairs.push((abs_p, rel.replace('\\', "/")));
                }
                pairs
            }
        }
        None => walk_repo(&repo),
    };

    // Filter out unchanged-by-hash files before doing the expensive parse.
    // We read the file twice in the unchanged case (once for hash here, once
    // in parse_one if it's new), but the bytes are warm in OS cache and the
    // SQLite roundtrip dominates anyway. To minimise round-trips, look up the
    // existing hash for every candidate in one pass.
    let mut existing: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    {
        let mut stmt = conn.prepare("SELECT path, hash FROM files")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        for row in rows.flatten() {
            existing.insert(row.0, row.1);
        }
    }

    stats.files_seen = candidates.len() as u64;

    // Parse in parallel; skip unchanged.
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let unchanged: Mutex<u64> = Mutex::new(0);

    let parsed: Vec<ParsedFile> = candidates
        .par_iter()
        .filter_map(|(abs_path, rel)| {
            let bytes = match std::fs::read(abs_path) {
                Ok(b) => b,
                Err(e) => {
                    errors.lock().unwrap().push(format!("{rel}: {e}"));
                    return None;
                }
            };
            let h = hash_bytes(&bytes);
            if let Some(prev) = existing.get(rel) {
                if prev == &h {
                    *unchanged.lock().unwrap() += 1;
                    return None;
                }
            }
            let parsed = parsers::dispatch(abs_path, &bytes)?;
            let (mtime, size) = match abs_path.metadata() {
                Ok(meta) => {
                    let m = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                        .map(|d| d.as_secs_f64())
                        .unwrap_or_else(now_secs);
                    (m, meta.len())
                }
                Err(_) => (now_secs(), bytes.len() as u64),
            };
            Some(ParsedFile {
                rel: rel.clone(),
                abs_path: abs_path.clone(),
                file_hash: h,
                mtime,
                size,
                parsed,
                bytes,
            })
        })
        .collect();

    stats.files_skipped_unchanged = *unchanged.lock().unwrap();
    stats.errors.extend(errors.into_inner().unwrap());

    let tx = conn.transaction()?;
    for pf in &parsed {
        if let Err(e) = upsert_file(&tx, pf, &mut stats) {
            stats.errors.push(format!("{}: {e}", pf.rel));
        } else {
            stats.files_parsed += 1;
        }
        let _ = &pf.abs_path; // silence unused
    }
    tx.commit()?;

    // Refresh manifest.
    let manifest = serde_json::json!({
        "repo_root": repo.to_string_lossy(),
        "schema_version": crate::SCHEMA_VERSION,
        "last_indexed": now_secs(),
        "last_commit": current_commit(&repo),
        "files": stats.files_parsed,
        "symbols": stats.symbols_inserted,
    });
    let mp = db::manifest_path(&repo);
    if let Some(parent) = mp.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(
        &mp,
        serde_json::to_string_pretty(&manifest).unwrap_or_else(|_| "{}".into()),
    )
    .with_context(|| format!("failed to write manifest {}", mp.display()))?;

    stats.elapsed_s = start.elapsed().as_secs_f64();
    Ok(stats)
}

/// Re-parse a single file (used by `watch` mode for incremental updates).
pub fn reindex_file(repo_root: &Path, rel: &str) -> Result<()> {
    let repo = repo_root
        .canonicalize()
        .unwrap_or_else(|_| repo_root.to_path_buf());
    let abs_p = repo.join(rel);
    if !abs_p.exists() {
        let conn = db::open_db(&repo)?;
        conn.execute("DELETE FROM symbols WHERE path = ?1", params![rel])?;
        conn.execute("DELETE FROM files WHERE path = ?1", params![rel])?;
        return Ok(());
    }
    if parsers::language_for(&abs_p).is_none() {
        return Ok(());
    }
    let Some(pf) = parse_one(&abs_p, rel) else {
        return Ok(());
    };
    let mut conn = db::open_db(&repo)?;
    let tx = conn.transaction()?;
    let mut stats = RebuildStats::default();
    upsert_file(&tx, &pf, &mut stats)?;
    tx.commit()?;
    Ok(())
}

/// Helper for the `info` subcommand.
#[derive(Debug, Serialize)]
pub struct RepoInfo {
    pub repo_root: String,
    pub files: i64,
    pub symbols: i64,
    pub languages: Vec<(String, i64)>,
    pub db_size_bytes: u64,
    pub last_indexed: Option<f64>,
    pub last_commit: Option<String>,
}

pub fn repo_info(repo_root: &Path) -> Result<RepoInfo> {
    let repo = repo_root
        .canonicalize()
        .unwrap_or_else(|_| repo_root.to_path_buf());
    let conn = db::open_db(&repo)?;
    let files: i64 = conn
        .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
        .unwrap_or(0);
    let symbols: i64 = conn
        .query_row("SELECT COUNT(*) FROM symbols", [], |r| r.get(0))
        .unwrap_or(0);
    let mut stmt = conn.prepare("SELECT language, COUNT(*) FROM files GROUP BY language")?;
    let mut langs = Vec::new();
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
    for row in rows.flatten() {
        langs.push(row);
    }
    langs.sort_by_key(|kv| std::cmp::Reverse(kv.1));
    let db_size = std::fs::metadata(db::db_path(&repo))
        .map(|m| m.len())
        .unwrap_or(0);

    let mut last_indexed = None;
    let mut last_commit = None;
    if let Ok(s) = std::fs::read_to_string(db::manifest_path(&repo)) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            last_indexed = v.get("last_indexed").and_then(|x| x.as_f64());
            last_commit = v
                .get("last_commit")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string());
        }
    }
    Ok(RepoInfo {
        repo_root: repo.to_string_lossy().to_string(),
        files,
        symbols,
        languages: langs,
        db_size_bytes: db_size,
        last_indexed,
        last_commit,
    })
}
