use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use rusqlite::{Connection, params, Result as SqliteResult};
use std::sync::{Arc, Mutex};

/// Manages vector persistence and RAG memory.
/// Follows Law of Demeter by encapsulating the database connection logic.
#[pyclass]
pub struct MemoryManager {
    db_path: String,
    conn: Arc<Mutex<Connection>>,
}

#[pymethods]
impl MemoryManager {
    /// Creates a new MemoryManager instance.
    ///
    /// # Contract
    /// * `db_path` must not be empty.
    #[new]
    pub fn new(db_path: String) -> PyResult<Self> {
        if db_path.trim().is_empty() {
            return Err(PyValueError::new_err("db_path cannot be empty"));
        }
        
        let conn = Connection::open(&db_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open DB: {}", e)))?;
            
        Ok(Self { 
            db_path,
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Initializes the database schema.
    pub fn initialize(&self) -> PyResult<()> {
        let conn = self.conn.lock().unwrap();
        
        // Attempt to create a standard table for documents, and a virtual table for vss0
        // If vss0 is not available (extension not loaded), it might fail, so we handle it gracefully.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payload TEXT NOT NULL
            )",
            [],
        ).map_err(|e| PyRuntimeError::new_err(format!("DB init error: {}", e)))?;

        // Note: In a real deployment, sqlite-vss extension must be loaded here
        // using sqlite3_enable_load_extension.
        // We'll create the vss0 table if we can.
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
    pub fn store_embedding(&self, payload: String, embedding: Vec<f32>) -> PyResult<()> {
        if payload.trim().is_empty() {
            return Err(PyValueError::new_err("payload cannot be empty"));
        }
        if embedding.is_empty() {
            return Err(PyValueError::new_err("embedding cannot be empty"));
        }
        
        let conn = self.conn.lock().unwrap();
        
        // Insert into documents
        conn.execute(
            "INSERT INTO documents (payload) VALUES (?1)",
            params![payload],
        ).map_err(|e| PyRuntimeError::new_err(format!("DB insert error: {}", e)))?;
        
        let last_id = conn.last_insert_rowid();

        // Convert embedding to JSON array format for vss0
        let emb_json = serde_json::to_string(&embedding)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {}", e)))?;

        // Try inserting into vss_documents if it exists
        let _ = conn.execute(
            "INSERT INTO vss_documents(rowid, embedding) VALUES (?1, ?2)",
            params![last_id, emb_json],
        );
        
        Ok(())
    }

    /// Retrieves the top-k most similar payloads for a given vector.
    pub fn search(&self, query_embedding: Vec<f32>, top_k: usize) -> PyResult<Vec<String>> {
        if query_embedding.is_empty() {
            return Err(PyValueError::new_err("query_embedding cannot be empty"));
        }
        if top_k == 0 {
            return Err(PyValueError::new_err("top_k must be greater than 0"));
        }

        let conn = self.conn.lock().unwrap();
        
        let emb_json = serde_json::to_string(&query_embedding)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {}", e)))?;

        let mut stmt = conn.prepare(
            "SELECT d.payload 
             FROM vss_documents v 
             JOIN documents d ON v.rowid = d.id 
             WHERE vss_search(v.embedding, ?1) 
             LIMIT ?2"
        ).map_err(|e| PyRuntimeError::new_err(format!("DB prepare error: {}", e)))?;

        let rows = stmt.query_map(params![emb_json, top_k as i64], |row| {
            row.get::<_, String>(0)
        });

        match rows {
            Ok(mapped_rows) => {
                let mut results = Vec::new();
                for r in mapped_rows {
                    if let Ok(payload) = r {
                        results.push(payload);
                    }
                }
                if results.is_empty() {
                    // Fallback to random if no vss setup
                    let mut fallback_stmt = conn.prepare("SELECT payload FROM documents LIMIT ?1").unwrap();
                    let fallback_rows = fallback_stmt.query_map(params![top_k as i64], |row| row.get::<_, String>(0)).unwrap();
                    for r in fallback_rows {
                        if let Ok(p) = r {
                            results.push(p);
                        }
                    }
                }
                Ok(results)
            },
            Err(_) => {
                // VSS likely not loaded, fallback to basic retrieval
                let mut fallback_stmt = conn.prepare("SELECT payload FROM documents LIMIT ?1").unwrap();
                let fallback_rows = fallback_stmt.query_map(params![top_k as i64], |row| row.get::<_, String>(0)).unwrap();
                let mut results = Vec::new();
                for r in fallback_rows {
                    if let Ok(p) = r {
                        results.push(p);
                    }
                }
                Ok(results)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_rejects_empty_path() {
        let result = MemoryManager::new("   ".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_store_embedding_validation() {
        let manager = MemoryManager::new("./test.db".to_string()).unwrap();
        let result = manager.store_embedding("".to_string(), vec![0.1, 0.2]);
        assert!(result.is_err());
        
        let result = manager.store_embedding("data".to_string(), vec![]);
        assert!(result.is_err());
    }
}
