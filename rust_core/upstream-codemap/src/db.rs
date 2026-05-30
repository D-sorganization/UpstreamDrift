//! SQLite + FTS5 schema mirror of `codemap.db`.
//!
//! Schema mirrors `src/shared/python/codemap/db.py` exactly so a Rust-produced
//! `.codemap/index.db` is queryable by the existing Python `codemap.api`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::Connection;

use crate::{DB_DIR_NAME, DB_FILE_NAME, MANIFEST_FILE_NAME, SCHEMA_VERSION};

/// Returns the canonical index DB path for `repo_root`.
pub fn db_path(repo_root: &Path) -> PathBuf {
    repo_root.join(DB_DIR_NAME).join(DB_FILE_NAME)
}

/// Returns the manifest JSON path for `repo_root`.
pub fn manifest_path(repo_root: &Path) -> PathBuf {
    repo_root.join(DB_DIR_NAME).join(MANIFEST_FILE_NAME)
}

/// Open (and lazily initialise) the index DB for `repo_root`.
pub fn open_db(repo_root: &Path) -> Result<Connection> {
    let target = db_path(repo_root);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create db parent dir {}", parent.display()))?;
        // Drop a sibling .gitignore so the index isn't accidentally committed.
        let gi = parent.join(".gitignore");
        if !gi.exists() {
            std::fs::write(&gi, "*\n").ok();
        }
    }
    let conn = Connection::open(&target)
        .with_context(|| format!("failed to open {}", target.display()))?;
    conn.pragma_update_and_check(None, "journal_mode", "WAL", |_| Ok(()))?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    init_schema(&conn)?;
    Ok(conn)
}

/// Create tables if missing. Idempotent.
pub fn init_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            path        TEXT PRIMARY KEY,
            language    TEXT NOT NULL,
            hash        TEXT NOT NULL,
            mtime       REAL NOT NULL,
            size        INTEGER NOT NULL,
            imports     TEXT NOT NULL DEFAULT '[]',
            indexed_at  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS symbols (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            path        TEXT NOT NULL,
            kind        TEXT NOT NULL,
            name        TEXT NOT NULL,
            qualified   TEXT NOT NULL,
            sig         TEXT NOT NULL DEFAULT '',
            docstring   TEXT NOT NULL DEFAULT '',
            start_line  INTEGER NOT NULL,
            end_line    INTEGER NOT NULL,
            calls_out   TEXT NOT NULL DEFAULT '[]',
            hash        TEXT NOT NULL,
            FOREIGN KEY (path) REFERENCES files(path) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_symbols_qualified ON symbols(qualified);
        CREATE INDEX IF NOT EXISTS idx_symbols_name      ON symbols(name);
        CREATE INDEX IF NOT EXISTS idx_symbols_path      ON symbols(path);
        CREATE INDEX IF NOT EXISTS idx_symbols_kind      ON symbols(kind);

        CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
            name, qualified, sig, docstring, co,
            content='symbols', content_rowid='id',
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
            INSERT INTO symbols_fts(rowid, name, qualified, sig, docstring, co)
            VALUES (new.id, new.name, new.qualified,
                    new.sig, new.docstring, new.calls_out);
        END;

        CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
            INSERT INTO symbols_fts(
                symbols_fts, rowid, name, qualified, sig, docstring, co
            ) VALUES('delete', old.id, old.name, old.qualified,
                     old.sig, old.docstring, old.calls_out);
        END;

        CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
            INSERT INTO symbols_fts(
                symbols_fts, rowid, name, qualified, sig, docstring, co
            ) VALUES('delete', old.id, old.name, old.qualified,
                     old.sig, old.docstring, old.calls_out);
            INSERT INTO symbols_fts(rowid, name, qualified, sig, docstring, co)
            VALUES (new.id, new.name, new.qualified,
                    new.sig, new.docstring, new.calls_out);
        END;
        "#,
    )?;

    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES(?1, ?2)",
        rusqlite::params!["schema_version", SCHEMA_VERSION.to_string()],
    )
    .map_err(|e| anyhow::anyhow!("Failed executing INSERT OR REPLACE INTO meta: {:?}", e))?;
    Ok(())
}

/// Read the meta schema_version, or 0 if missing.
pub fn get_schema_version(conn: &Connection) -> i64 {
    conn.query_row(
        "SELECT value FROM meta WHERE key = 'schema_version'",
        [],
        |r| r.get::<_, String>(0),
    )
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(0)
}
