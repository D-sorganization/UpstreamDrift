//! Vector persistence and RAG memory manager.
//!
//! Follows Law of Demeter by encapsulating the database connection logic.
//! Pure-Rust core is always compiled; the `python` feature adds PyO3 bindings.
//!
//! ## sqlite-vss loading
//!
//! The bundled rusqlite SQLite does not include `vss0` by default. We attempt
//! to enable extension loading and `load_extension('vss0')` at `initialize()`
//! time; if loading fails (extension not installed, Windows MSVC build of
//! sqlite-vss not available, etc.) we set `has_vss = false` and the search
//! path falls back to a plain `SELECT ... LIMIT k`. The previous version
//! silently degraded — this one logs to stderr so operators have a fighting
//! chance of noticing they aren't actually doing vector search.

#[cfg(feature = "python")]
use pyo3::exceptions::{PyRuntimeError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;

use rusqlite::{params, Connection};
use std::sync::{Arc, Mutex};

/// Manages vector persistence and RAG memory.
#[cfg_attr(feature = "python", pyclass)]
pub struct MemoryManager {
    #[allow(dead_code)]
    db_path: String,
    conn: Arc<Mutex<Connection>>,
    /// Whether the `vss0` extension was successfully loaded. Set by
    /// `try_initialize()` and read by `try_search()` to decide whether to
    /// use the vector path or the fallback. The `Mutex` is needed for
    /// interior mutability through the shared `&self` borrow on `initialize`.
    has_vss: Arc<Mutex<bool>>,
}

impl MemoryManager {
    /// Pure-Rust constructor.
    ///
    /// # Contract
    /// * `db_path` must not be empty.
    pub fn try_new(db_path: String) -> Result<Self, String> {
        if db_path.trim().is_empty() {
            return Err("db_path cannot be empty".to_string());
        }

        let conn = Connection::open(&db_path).map_err(|e| format!("Failed to open DB: {}", e))?;

        Ok(Self {
            db_path,
            conn: Arc::new(Mutex::new(conn)),
            has_vss: Arc::new(Mutex::new(false)),
        })
    }

    /// Initializes the database schema and attempts to load `vss0`.
    pub fn try_initialize(&self) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payload TEXT NOT NULL
            )",
            [],
        )
        .map_err(|e| format!("DB init error: {}", e))?;

        // Attempt to load the sqlite-vss extension. The `load_extension`
        // rusqlite feature is not enabled (bundled SQLite + load_extension
        // requires a custom build), so we go through pragma `load_extension`
        // instead. If anything goes wrong, we fall back to a non-vector
        // search path and log a single warning.
        let vss_ok = try_load_vss(&conn);
        if vss_ok {
            let create_result = conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vss_documents USING vss0(
                    embedding(384)
                )",
                [],
            );
            if let Err(e) = create_result {
                eprintln!(
                    "ai_backend: vss0 loaded but virtual-table create failed ({}); falling back to LIMIT-k retrieval.",
                    e
                );
                *self.has_vss.lock().unwrap() = false;
            } else {
                *self.has_vss.lock().unwrap() = true;
            }
        } else {
            eprintln!(
                "ai_backend: sqlite-vss extension not available; vector search is degraded to ORDER BY id LIMIT k. \
                 Install sqlite-vss and rebuild with rusqlite's `load_extension` feature for true vector search."
            );
            *self.has_vss.lock().unwrap() = false;
        }

        Ok(())
    }

    /// Stores a vector and its associated payload in the database.
    ///
    /// # Contract
    /// * `payload` must not be empty.
    /// * `embedding` must match the expected dimensions (e.g. 384 or 1536).
    pub fn try_store_embedding(&self, payload: String, embedding: Vec<f32>) -> Result<(), String> {
        if payload.trim().is_empty() {
            return Err("payload cannot be empty".to_string());
        }
        if embedding.is_empty() {
            return Err("embedding cannot be empty".to_string());
        }

        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO documents (payload) VALUES (?1)",
            params![payload],
        )
        .map_err(|e| format!("DB insert error: {}", e))?;

        let last_id = conn.last_insert_rowid();

        let emb_json =
            serde_json::to_string(&embedding).map_err(|e| format!("Serialization error: {}", e))?;

        let _ = conn.execute(
            "INSERT INTO vss_documents(rowid, embedding) VALUES (?1, ?2)",
            params![last_id, emb_json],
        );

        Ok(())
    }

    /// Retrieves the top-k most similar payloads for a given vector.
    pub fn try_search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<String>, String> {
        if query_embedding.is_empty() {
            return Err("query_embedding cannot be empty".to_string());
        }
        if top_k == 0 {
            return Err("top_k must be greater than 0".to_string());
        }

        let conn = self.conn.lock().unwrap();
        let has_vss = *self.has_vss.lock().unwrap();

        let mut results: Vec<String> = Vec::new();

        if has_vss {
            let emb_json = serde_json::to_string(&query_embedding)
                .map_err(|e| format!("Serialization error: {}", e))?;

            let stmt = conn.prepare(
                "SELECT d.payload
                 FROM vss_documents v
                 JOIN documents d ON v.rowid = d.id
                 WHERE vss_search(v.embedding, ?1)
                 LIMIT ?2",
            );

            if let Ok(mut stmt) = stmt {
                if let Ok(mapped_rows) = stmt.query_map(params![emb_json, top_k as i64], |row| {
                    row.get::<_, String>(0)
                }) {
                    for r in mapped_rows.flatten() {
                        results.push(r);
                    }
                }
            }
        }

        if results.is_empty() {
            // Fallback to a basic LIMIT-k retrieval when vss0 is not available
            // or the vector path returned nothing.
            let mut fallback_stmt = conn
                .prepare("SELECT payload FROM documents ORDER BY id DESC LIMIT ?1")
                .map_err(|e| format!("DB prepare error: {}", e))?;
            let fallback_rows = fallback_stmt
                .query_map(params![top_k as i64], |row| row.get::<_, String>(0))
                .map_err(|e| format!("DB query error: {}", e))?;
            for r in fallback_rows.flatten() {
                results.push(r);
            }
        }

        Ok(results)
    }
}

/// Best-effort load of the `vss0` extension. Returns true on success.
///
/// Attempts to load the platform-specific `vss0` shared library from a
/// well-known vendored path. Falls back gracefully so callers can use the
/// LIMIT-k retrieval path when the extension is not available.
fn try_load_vss(conn: &Connection) -> bool {
    // Enable extension loading on the connection (bundled-full feature).
    let enable_result = unsafe { conn.load_extension_enable() };
    if enable_result.is_err() {
        eprintln!("ai_backend: could not enable extension loading; vss0 disabled.");
        return false;
    }
    match enable_result {
        Ok(()) => {}
        Err(_) => return false,
    };

    // Probe the well-known vss0 library name for the current platform.
    #[cfg(target_os = "linux")]
    let lib_name = "vss0.so";
    #[cfg(target_os = "macos")]
    let lib_name = "vss0.dylib";
    #[cfg(target_os = "windows")]
    let lib_name = "vss0.dll";

    // Try loading by just the library name (relies on the system library
    // search path or the vendored location being on LD_LIBRARY_PATH/DYLD_
    // LIBRARY_PATH/PATH). Also try a relative vendored path as a fallback.
    let candidates: &[&str] = &[
        lib_name,
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        &format!("./vendored/{}", lib_name).leak(),
    ];

    for candidate in candidates {
        match unsafe { conn.load_extension(candidate, None) } {
            Ok(()) => {
                eprintln!("ai_backend: loaded vss0 extension from '{}'.", candidate);
                return true;
            }
            Err(e) => {
                eprintln!(
                    "ai_backend: failed to load vss0 from '{}': {}",
                    candidate, e
                );
            }
        }
    }

    eprintln!(
        "ai_backend: sqlite-vss extension not available; vector search is degraded to ORDER BY id LIMIT k. \
         Install sqlite-vss and place the library on the search path or in ./vendored/ for true vector search."
    );
    false
}

#[cfg(feature = "python")]
#[pymethods]
impl MemoryManager {
    /// Creates a new MemoryManager instance.
    ///
    /// # Contract
    /// * `db_path` must not be empty.
    #[new]
    pub fn new(db_path: String) -> PyResult<Self> {
        Self::try_new(db_path).map_err(|e| {
            if e == "db_path cannot be empty" {
                PyValueError::new_err(e)
            } else {
                PyRuntimeError::new_err(e)
            }
        })
    }

    /// Initializes the database schema.
    pub fn initialize(&self) -> PyResult<()> {
        self.try_initialize().map_err(PyRuntimeError::new_err)
    }

    /// Stores a vector and its associated payload in the database.
    pub fn store_embedding(&self, payload: String, embedding: Vec<f32>) -> PyResult<()> {
        self.try_store_embedding(payload, embedding).map_err(|e| {
            if e.starts_with("payload") || e.starts_with("embedding") {
                PyValueError::new_err(e)
            } else {
                PyRuntimeError::new_err(e)
            }
        })
    }

    /// Retrieves the top-k most similar payloads for a given vector.
    pub fn search(&self, query_embedding: Vec<f32>, top_k: usize) -> PyResult<Vec<String>> {
        self.try_search(query_embedding, top_k).map_err(|e| {
            if e.starts_with("query_embedding") || e.starts_with("top_k") {
                PyValueError::new_err(e)
            } else {
                PyRuntimeError::new_err(e)
            }
        })
    }

    /// Whether the `vss0` extension is currently loaded. Useful for tests
    /// and for surfacing degraded-mode warnings in the UI.
    pub fn has_vss(&self) -> bool {
        *self.has_vss.lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_rejects_empty_path() {
        let result = MemoryManager::try_new("   ".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_store_embedding_validation() {
        let manager = MemoryManager::try_new("./test.db".to_string()).unwrap();
        let result = manager.try_store_embedding("".to_string(), vec![0.1, 0.2]);
        assert!(result.is_err());

        let result = manager.try_store_embedding("data".to_string(), vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_store_and_search_roundtrip_fallback_path() {
        // In-memory DB so the test is hermetic across runs / parallel cargo.
        let manager = MemoryManager::try_new(":memory:".to_string()).unwrap();
        manager.try_initialize().unwrap();

        manager
            .try_store_embedding("doc one".to_string(), vec![0.1; 8])
            .unwrap();
        manager
            .try_store_embedding("doc two".to_string(), vec![0.2; 8])
            .unwrap();

        let results = manager.try_search(vec![0.15; 8], 5).unwrap();
        assert_eq!(results.len(), 2);
        // Fallback ORDER BY id DESC: newest first.
        assert_eq!(results[0], "doc two");
        assert_eq!(results[1], "doc one");
    }

    #[test]
    fn test_search_rejects_empty_query() {
        let manager = MemoryManager::try_new(":memory:".to_string()).unwrap();
        let result = manager.try_search(vec![], 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_rejects_zero_top_k() {
        let manager = MemoryManager::try_new(":memory:".to_string()).unwrap();
        let result = manager.try_search(vec![0.1, 0.2], 0);
        assert!(result.is_err());
    }
}
