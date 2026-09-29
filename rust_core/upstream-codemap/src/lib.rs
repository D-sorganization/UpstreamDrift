//! Library entry for `upstream-codemap`.
//!
//! The crate is primarily a binary, but the modules are re-exported so that
//! integration tests and the optional pyo3 facade can drive the same code.

pub mod db;
pub mod indexer;
pub mod parsers;
pub mod watcher;

pub use indexer::{rebuild, RebuildStats};

/// Schema version mirrored from `codemap.db.SCHEMA_VERSION`.
pub const SCHEMA_VERSION: i64 = 1;

/// Directory name (under repo root) for the index db.
pub const DB_DIR_NAME: &str = ".codemap";
/// Database file name inside `DB_DIR_NAME`.
pub const DB_FILE_NAME: &str = "index.db";
/// Manifest JSON file name inside `DB_DIR_NAME`.
pub const MANIFEST_FILE_NAME: &str = "manifest.json";
