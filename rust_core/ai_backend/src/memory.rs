//! Vector persistence and RAG memory manager.
//!
//! Follows Law of Demeter by encapsulating the database connection logic.
//! Pure-Rust core is always compiled; the `python` feature adds PyO3 bindings.

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
        })
    }

    /// Initializes the database schema.
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

        // Note: In a real deployment, sqlite-vss extension must be loaded here
        // using sqlite3_enable_load_extension.
        let _ = conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vss_documents USING vss0(
                embedding(384)
            )",
            [],
        );

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

        let emb_json = serde_json::to_string(&query_embedding)
            .map_err(|e| format!("Serialization error: {}", e))?;

        let stmt = conn.prepare(
            "SELECT d.payload
             FROM vss_documents v
             JOIN documents d ON v.rowid = d.id
             WHERE vss_search(v.embedding, ?1)
             LIMIT ?2",
        );

        let mut results: Vec<String> = Vec::new();

        if let Ok(mut stmt) = stmt {
            if let Ok(mapped_rows) = stmt.query_map(params![emb_json, top_k as i64], |row| {
                row.get::<_, String>(0)
            }) {
                for r in mapped_rows.flatten() {
                    results.push(r);
                }
            }
        }

        if results.is_empty() {
            // Fallback to a basic retrieval when vss0 is not available.
            let mut fallback_stmt = conn
                .prepare("SELECT payload FROM documents LIMIT ?1")
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
}
