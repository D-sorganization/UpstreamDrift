//! High-performance RAG Pipeline.
//!
//! Follows Law of Demeter by acting as the coordinator; the UI calls
//! `RagPipeline`, which in turn orchestrates embeddings and the
//! `MemoryManager`. The PyO3 wrapper (`RagPipeline`) is feature-gated; the
//! pure-Rust indexing helpers below are always compiled and tested.
//!
//! ## Embeddings
//!
//! Real embeddings via [`crate::embeddings`] (OpenAI-compatible endpoint).
//! The pre-#5220 placeholder `vec![0.0; 384]` is removed. If the embedding
//! call fails, the chunk is skipped and a warning is logged via `eprintln!`
//! (the crate does not yet take a `log` dependency); a follow-up will route
//! these through a proper logger.

use std::path::Path;
use std::sync::Arc;

use reqwest::Client;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::config::AIConfig;
use crate::embeddings;
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
/// Real embeddings are obtained via the configured provider; chunks for which
/// the embedding call fails are skipped (not fatal, so a transient provider
/// blip doesn't abort an indexing run).
///
/// # Contract
/// * `root_path` must exist and be a directory.
pub fn index_codebase_impl(
    memory: &MemoryManager,
    config: &AIConfig,
    rt: &tokio::runtime::Runtime,
    client: Arc<Client>,
    root_path: &str,
) -> Result<usize, String> {
    validate_index_path(root_path)?;

    let mut indexed: usize = 0;

    for entry in walkdir::WalkDir::new(root_path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        let lines: Vec<&str> = content.lines().collect();
        for chunk in lines.chunks(50) {
            let chunk_text = chunk.join("\n");
            if chunk_text.trim().is_empty() {
                continue;
            }

            let client = Arc::clone(&client);
            let embedding = rt.block_on(embeddings::embed_one(client, config, &chunk_text));
            let embedding = match embedding {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "ai_backend: embedding failed for chunk in {} ({} chars): {}",
                        path.display(),
                        chunk_text.len(),
                        e
                    );
                    continue;
                }
            };

            if memory.try_store_embedding(chunk_text, embedding).is_ok() {
                indexed += 1;
            }
        }
    }

    Ok(indexed)
}

/// Pure-Rust context retrieval helper.
///
/// # Contract
/// * `prompt` must not be empty.
/// * `top_k` must be greater than 0.
pub fn retrieve_context_impl(
    memory: &MemoryManager,
    config: &AIConfig,
    rt: &tokio::runtime::Runtime,
    client: Arc<Client>,
    prompt: &str,
    top_k: usize,
) -> Result<Vec<String>, String> {
    if prompt.trim().is_empty() {
        return Err("Prompt cannot be empty".to_string());
    }
    if top_k == 0 {
        return Err("top_k must be greater than 0".to_string());
    }

    let query_embedding = rt.block_on(embeddings::embed_one(client, config, prompt))?;
    memory.try_search(query_embedding, top_k)
}

// ── Python bindings (feature-gated) ──────────────────────────────────────────

/// High-performance RAG Pipeline (PyO3 wrapper).
///
/// Owns a `MemoryManager` reference + a private reqwest client and Tokio
/// runtime so embeddings can be generated synchronously from the Python
/// caller's perspective.
#[cfg(feature = "python")]
#[pyclass]
pub struct RagPipeline {
    memory: Py<MemoryManager>,
    config: AIConfig,
    client: Arc<Client>,
    rt: Arc<tokio::runtime::Runtime>,
}

#[cfg(feature = "python")]
#[pymethods]
impl RagPipeline {
    /// Creates a new RagPipeline. Requires the `AIConfig` so the pipeline
    /// can call the embedding endpoint; without it, embeddings would have
    /// to fall back to the (removed) zero-vector placeholder.
    #[new]
    pub fn new(memory: Py<MemoryManager>, config: AIConfig) -> PyResult<Self> {
        config
            .validate()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build Tokio runtime: {}",
                    e
                ))
            })?;
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build reqwest Client: {}",
                    e
                ))
            })?;
        Ok(Self {
            memory,
            config,
            client: Arc::new(client),
            rt: Arc::new(rt),
        })
    }

    /// Recursively indexes a directory, chunks text, generates embeddings,
    /// and stores them via the MemoryManager.
    pub fn index_codebase(&self, py: Python, root_path: String) -> PyResult<usize> {
        let memory_ref = self.memory.borrow(py);
        index_codebase_impl(
            &memory_ref,
            &self.config,
            &self.rt,
            Arc::clone(&self.client),
            &root_path,
        )
        .map_err(|e| {
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
        retrieve_context_impl(
            &memory_ref,
            &self.config,
            &self.rt,
            Arc::clone(&self.client),
            &prompt,
            top_k,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AIConfig {
        AIConfig::try_new(
            "k".into(),
            "http://localhost:1".into(),
            "m".into(),
            ":memory:".into(),
        )
        .unwrap()
    }

    fn test_rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn test_validate_rejects_nonexistent_path() {
        let result = validate_index_path("/this/path/does/not/exist/for/sure/123");
        assert!(result.is_err());
    }

    #[test]
    fn test_retrieve_context_rejects_empty_prompt() {
        let memory = MemoryManager::try_new("./rag_test.db".to_string()).unwrap();
        let rt = test_rt();
        let client = Arc::new(Client::new());
        let result = retrieve_context_impl(&memory, &test_config(), &rt, client, "   ", 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_retrieve_context_rejects_zero_top_k() {
        let memory = MemoryManager::try_new("./rag_test2.db".to_string()).unwrap();
        let rt = test_rt();
        let client = Arc::new(Client::new());
        let result = retrieve_context_impl(&memory, &test_config(), &rt, client, "query", 0);
        assert!(result.is_err());
    }
}
