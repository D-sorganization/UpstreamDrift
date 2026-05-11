use pyo3::prelude::*;
use std::path::Path;
use crate::memory::MemoryManager;

/// High-performance RAG Pipeline.
/// Follows Law of Demeter by acting as the coordinator; the UI calls RagPipeline,
/// which in turn orchestrates embeddings and the MemoryManager.
#[pyclass]
pub struct RagPipeline {
    memory: Py<MemoryManager>,
}

#[pymethods]
impl RagPipeline {
    /// Creates a new RagPipeline instance, taking ownership of the MemoryManager.
    #[new]
    pub fn new(memory: Py<MemoryManager>) -> Self {
        Self { memory }
    }

    /// Recursively indexes a directory, chunks text, generates embeddings,
    /// and stores them via the MemoryManager.
    ///
    /// # Contract
    /// * `root_path` must exist and be a directory.
    pub fn index_codebase(&self, py: Python, root_path: String) -> PyResult<usize> {
        let path = Path::new(&root_path);
        if !path.exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!("Path does not exist: {}", root_path)));
        }
        if !path.is_dir() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Path is not a directory: {}", root_path)));
        }

        let mut files_indexed = 0;
        
        // Pseudo-implementation of embeddings since ONNX/local models require large dependencies:
        // We'll generate a deterministic pseudo-random embedding based on text length for now,
        // or zeroes. In a real environment, we'd use candle-core or tokenizers.
        
        let memory_ref = self.memory.borrow(py);
        
        for entry in walkdir::WalkDir::new(&root_path).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() {
                // Read file
                if let Ok(content) = std::fs::read_to_string(path) {
                    // Chunk contents roughly by lines
                    let lines: Vec<&str> = content.lines().collect();
                    for chunk in lines.chunks(50) {
                        let chunk_text = chunk.join("\n");
                        if chunk_text.trim().is_empty() { continue; }
                        
                        // Generate dummy embedding for now
                        let dummy_embedding = vec![0.0; 384]; 
                        
                        if memory_ref.store_embedding(chunk_text, dummy_embedding).is_ok() {
                            files_indexed += 1;
                        }
                    }
                }
            }
        }

        Ok(files_indexed)
    }

    /// Retrieves context for a given prompt to augment the LLM request.
    ///
    /// # Contract
    /// * `prompt` must not be empty.
    pub fn retrieve_context(&self, py: Python, prompt: String, top_k: usize) -> PyResult<Vec<String>> {
        if prompt.trim().is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("Prompt cannot be empty"));
        }
        if top_k == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("top_k must be greater than 0"));
        }

        // Pseudo-implementation:
        // 1. Generate embedding for prompt.
        let dummy_query_embedding = vec![0.0; 384];

        // 2. Query memory manager.
        let memory_ref = self.memory.borrow(py);
        memory_ref.search(dummy_query_embedding, top_k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_rejects_nonexistent_path() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let memory = Py::new(py, MemoryManager::new("./dummy.db".to_string()).unwrap()).unwrap();
            let pipeline = RagPipeline::new(memory);
            let result = pipeline.index_codebase(py, "/this/path/does/not/exist/for/sure/123".to_string());
            assert!(result.is_err());
        });
    }
}
