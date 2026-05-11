use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

/// Manages vector persistence and RAG memory.
/// Follows Law of Demeter by encapsulating the database connection logic.
#[pyclass]
pub struct MemoryManager {
    db_path: String,
    // Future: SQLite connection handle or Qdrant client
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
            return Err(pyo3::exceptions::PyValueError::new_err("db_path cannot be empty"));
        }
        Ok(Self { db_path })
    }

    /// Initializes the database schema.
    pub fn initialize(&self) -> PyResult<()> {
        // Dummy implementation for now.
        // In reality, this would CREATE VIRTUAL TABLE using sqlite-vss.
        Ok(())
    }

    /// Stores a vector and its associated payload in the database.
    ///
    /// # Contract
    /// * `payload` must not be empty.
    /// * `embedding` must match the expected dimensions (e.g. 384 or 1536).
    pub fn store_embedding(&self, payload: String, embedding: Vec<f32>) -> PyResult<()> {
        if payload.trim().is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("payload cannot be empty"));
        }
        if embedding.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("embedding cannot be empty"));
        }
        
        // Dummy insert
        Ok(())
    }

    /// Retrieves the top-k most similar payloads for a given vector.
    pub fn search(&self, query_embedding: Vec<f32>, top_k: usize) -> PyResult<Vec<String>> {
        if query_embedding.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("query_embedding cannot be empty"));
        }
        if top_k == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("top_k must be greater than 0"));
        }

        // Dummy search
        Ok(vec!["Matched payload 1".to_string()])
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
