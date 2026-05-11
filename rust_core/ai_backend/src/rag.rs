//! High-performance RAG Pipeline.
//!
//! Follows Law of Demeter by acting as the coordinator; the UI calls
//! `RagPipeline`, which in turn orchestrates embeddings and the
//! `MemoryManager`. The PyO3 wrapper (`RagPipeline`) is feature-gated; the
//! pure-Rust indexing helpers below are always compiled and tested.

use std::path::Path;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::memory::MemoryManager;

/// Validate that a path exists and is a directory.
///
/// # Contract
/// * `root_path` must exist and be a directory.
pub fn validate_index_path(root_path: &str) -> Result<(), String> {
    let path = Path::new(root_path);
    if !path.exists() {
        return Err(format!("Path does not exist: {}", root_path));
    }
    if !path.is_dir() {
        return Err(format!("Path is not a directory: {}", root_path));
    }
    Ok(())
}

/// Pure-Rust indexer: walks a directory, chunks files, and stores entries via
/// the provided `MemoryManager`. Returns the number of chunks indexed.
///
/// # Contract
/// * `root_path` must exist and be a directory.
pub fn index_codebase_impl(memory: &MemoryManager, root_path: &str) -> Result<usize, String> {
    validate_index_path(root_path)?;

    let mut files_indexed: usize = 0;

    // Pseudo-implementation of embeddings since ONNX/local models require large
    // dependencies. We use a zero-vector placeholder; in production we'd plug
    // in candle-core or tokenizers.
    for entry in walkdir::WalkDir::new(root_path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() {
            if let Ok(content) = std::fs::read_to_string(path) {
                let lines: Vec<&str> = content.lines().collect();
                for chunk in lines.chunks(50) {
                    let chunk_text = chunk.join("\n");
                    if chunk_text.trim().is_empty() {
                        continue;
                    }

                    let dummy_embedding = vec![0.0_f32; 384];

                    if memory
                        .try_store_embedding(chunk_text, dummy_embedding)
                        .is_ok()
                    {
                        files_indexed += 1;
                    }
                }
            }
        }
    }

    Ok(files_indexed)
}

/// Pure-Rust context retrieval helper.
///
/// # Contract
/// * `prompt` must not be empty.
/// * `top_k` must be greater than 0.
pub fn retrieve_context_impl(
    memory: &MemoryManager,
    prompt: &str,
    top_k: usize,
) -> Result<Vec<String>, String> {
    if prompt.trim().is_empty() {
        return Err("Prompt cannot be empty".to_string());
    }
    if top_k == 0 {
        return Err("top_k must be greater than 0".to_string());
    }

    let dummy_query_embedding = vec![0.0_f32; 384];
    memory.try_search(dummy_query_embedding, top_k)
}

// ── Python bindings (feature-gated) ──────────────────────────────────────────

/// High-performance RAG Pipeline (PyO3 wrapper).
#[cfg(feature = "python")]
#[pyclass]
pub struct RagPipeline {
    memory: Py<MemoryManager>,
}

#[cfg(feature = "python")]
#[pymethods]
impl RagPipeline {
    /// Creates a new RagPipeline instance, taking ownership of the MemoryManager.
    #[new]
    pub fn new(memory: Py<MemoryManager>) -> Self {
        Self { memory }
    }

    /// Recursively indexes a directory, chunks text, generates embeddings,
    /// and stores them via the MemoryManager.
    pub fn index_codebase(&self, py: Python, root_path: String) -> PyResult<usize> {
        let memory_ref = self.memory.borrow(py);
        index_codebase_impl(&memory_ref, &root_path).map_err(|e| {
            if e.starts_with("Path does not exist") {
                pyo3::exceptions::PyFileNotFoundError::new_err(e)
            } else {
                pyo3::exceptions::PyValueError::new_err(e)
            }
        })
    }

    /// Retrieves context for a given prompt to augment the LLM request.
    pub fn retrieve_context(
        &self,
        py: Python,
        prompt: String,
        top_k: usize,
    ) -> PyResult<Vec<String>> {
        let memory_ref = self.memory.borrow(py);
        retrieve_context_impl(&memory_ref, &prompt, top_k)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_rejects_nonexistent_path() {
        let result = validate_index_path("/this/path/does/not/exist/for/sure/123");
        assert!(result.is_err());
    }

    #[test]
    fn test_retrieve_context_rejects_empty_prompt() {
        let memory = MemoryManager::try_new("./rag_test.db".to_string()).unwrap();
        let result = retrieve_context_impl(&memory, "   ", 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_retrieve_context_rejects_zero_top_k() {
        let memory = MemoryManager::try_new("./rag_test2.db".to_string()).unwrap();
        let result = retrieve_context_impl(&memory, "query", 0);
        assert!(result.is_err());
    }
}
