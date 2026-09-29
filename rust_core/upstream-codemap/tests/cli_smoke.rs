//! End-to-end smoke test: rebuild a tiny fixture repo and query it.

use std::fs;
use std::path::Path;

use rusqlite::Connection;
use upstream_codemap::{db, indexer};

fn write_fixture(root: &Path) {
    fs::create_dir_all(root).unwrap();
    fs::write(
        root.join("hello.py"),
        b"\"\"\"hi\"\"\"\n\nclass Greeter:\n    def say_hi(self):\n        print('hi')\n",
    )
    .unwrap();
    fs::write(
        root.join("lib.rs"),
        b"pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
    )
    .unwrap();
    fs::write(root.join("notes.md"), b"# Title\n\n## Sub\n").unwrap();
}

#[test]
fn rebuild_then_query() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    write_fixture(root);

    let stats = indexer::rebuild(root, None).unwrap_or_else(|e| panic!("rebuild failed: {:?}", e));
    assert!(stats.files_parsed >= 3, "expected >=3 files, got {stats:?}");
    assert!(stats.symbols_inserted >= 4);
    assert!(stats.elapsed_s >= 0.0);

    // Query the resulting DB directly.
    let dbp = db::db_path(root);
    assert!(dbp.exists(), "db not created at {}", dbp.display());
    let conn = Connection::open(&dbp).unwrap();

    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM symbols WHERE name = 'say_hi'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 1, "say_hi should be indexed once");

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM symbols WHERE name = 'add'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(count, 1);

    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM symbols WHERE name = 'Title'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);

    // FTS query: searching for 'say_hi' should produce a hit.
    let mut stmt = conn
        .prepare(
            "SELECT s.qualified FROM symbols s JOIN symbols_fts f \
             ON f.rowid = s.id WHERE symbols_fts MATCH ?1",
        )
        .unwrap();
    let hits: Vec<String> = stmt
        .query_map(["say_hi"], |r| r.get::<_, String>(0))
        .unwrap()
        .flatten()
        .collect();
    assert!(
        hits.iter().any(|h| h.contains("say_hi")),
        "FTS5 should find say_hi: {hits:?}",
    );
}

#[test]
fn rebuild_skips_unchanged_files() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    write_fixture(root);

    let first = indexer::rebuild(root, None).unwrap_or_else(|e| panic!("first failed: {:?}", e));
    assert!(first.files_parsed >= 3);

    let second = indexer::rebuild(root, None).expect("second");
    assert!(
        second.files_skipped_unchanged >= 3,
        "expected most files unchanged on second rebuild: {second:?}",
    );
    assert_eq!(second.files_parsed, 0, "no files should re-parse");
}
