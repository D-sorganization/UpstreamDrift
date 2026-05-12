//! High-performance RAG Pipeline.
//!
//! Follows Law of Demeter by acting as the coordinator; the UI calls
//! `RagPipeline`, which in turn orchestrates embeddings and the
//! `MemoryManager`. The PyO3 wrapper (`RagPipeline`) is feature-gated; the
//! pure-Rust indexing helpers below are always compiled and tested.
//!
//! ## Embeddings
//!
//! Two embedding backends are supported:
//!
//! 1. **HTTP** (default) — [`crate::embeddings`] posts to an
//!    OpenAI-compatible `/embeddings` endpoint. Works against OpenAI,
//!    LM Studio, Ollama-with-/v1, Together, etc.
//! 2. **Local ONNX** (opt-in via `local-embeddings` feature) —
//!    [`crate::local_embed::LocalEmbedder`] runs `all-MiniLM-L6-v2` via
//!    `ort` + `tokenizers` with no network call. Selected automatically
//!    when [`RagPipeline`] is constructed with `use_local_embeddings=True`.
//!
//! ## Indexing
//!
//! [`index_codebase_impl`] walks the target directory with the [`ignore`]
//! crate (so `.gitignore` is honored without re-implementing it), filters
//! to a configured set of text-ish file extensions, splits each file into
//! line-window chunks with overlap, and stores each chunk's embedding via
//! the `MemoryManager`. Idempotency comes from the per-payload content
//! hash inside `MemoryManager::try_store_embedding`.

use std::path::Path;
use std::sync::Arc;

use ignore::WalkBuilder;
use reqwest::Client;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::config::AIConfig;
use crate::embeddings;
use crate::memory::MemoryManager;

/// Default chunk window size, in **lines**, fed to the embedder per call.
/// 40 lines of typical Python/Rust source is roughly 300–500 tokens, well
/// under the MiniLM 256-token limit; longer chunks get split further by the
/// tokenizer's truncation. Tuned by experiment, not by deep optimization.
pub const DEFAULT_CHUNK_LINES: usize = 40;

/// Default line overlap between adjacent chunks. Preserves cross-chunk
/// context for symbols that straddle a window boundary.
pub const DEFAULT_CHUNK_OVERLAP: usize = 8;

/// Allowed file extensions (lower-case, without leading dot). Anything else
/// is skipped — saves embedding cost on lock files, generated outputs, etc.
const ALLOWED_EXTENSIONS: &[&str] = &[
    "py", "rs", "toml", "md", "rst", "txt", "json", "yaml", "yml", "ts", "tsx", "js", "jsx",
    "html", "css", "go", "java", "kt", "swift", "cpp", "c", "h", "hpp", "rb",
];

/// Maximum file size (bytes) to read. Larger files are skipped to avoid
/// pulling vendored blobs into the vector store.
const MAX_FILE_BYTES: u64 = 512 * 1024;

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

/// Split text into overlapping line-window chunks.
///
/// # Contract
/// * `chunk_lines` must be > 0.
pub fn chunk_text(text: &str, chunk_lines: usize, overlap: usize) -> Vec<String> {
    if chunk_lines == 0 {
        return Vec::new();
    }
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let step = chunk_lines.saturating_sub(overlap).max(1);
    let mut out = Vec::new();
    let mut start = 0;
    while start < lines.len() {
        let end = (start + chunk_lines).min(lines.len());
        let chunk = lines[start..end].join("\n");
        if !chunk.trim().is_empty() {
            out.push(chunk);
        }
        if end == lines.len() {
            break;
        }
        start += step;
    }
    out
}

fn is_allowed(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    let ext_lower = ext.to_ascii_lowercase();
    ALLOWED_EXTENSIONS.iter().any(|e| *e == ext_lower)
}

/// Embedder abstraction so the indexing path doesn't care whether it's
/// calling an HTTP service or a local ONNX session.
pub trait Embedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, String>;
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
}

/// HTTP embedder bound to a runtime + reqwest client + AIConfig.
pub struct HttpEmbedder<'a> {
    pub client: Arc<Client>,
    pub config: &'a AIConfig,
    pub rt: &'a tokio::runtime::Runtime,
}

impl<'a> Embedder for HttpEmbedder<'a> {
    fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let client = Arc::clone(&self.client);
        self.rt
            .block_on(embeddings::embed_one(client, self.config, text))
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let client = Arc::clone(&self.client);
        self.rt
            .block_on(embeddings::embed_batch(client, self.config, texts))
    }
}

#[cfg(feature = "local-embeddings")]
impl Embedder for crate::local_embed::LocalEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        crate::local_embed::LocalEmbedder::embed(self, text)
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        crate::local_embed::LocalEmbedder::embed_batch(self, texts)
    }
}

/// Pure-Rust indexer using any [`Embedder`]. Walks the directory with
/// `.gitignore` respected, chunks each file, and stores each chunk via the
/// `MemoryManager`. Returns the number of chunks successfully indexed
/// (excludes chunks that failed embedding or were dedupe-skipped at the
/// memory layer).
///
/// # Contract
/// * `root_path` must exist and be a directory.
pub fn index_with_embedder<E: Embedder>(
    memory: &MemoryManager,
    embedder: &E,
    root_path: &str,
) -> Result<usize, String> {
    validate_index_path(root_path)?;

    // `require_git(false)` lets `.gitignore` files take effect outside a
    // git repository — relevant for indexing extracted archives, vendor
    // dirs, or test fixtures. Default `ignore` behaviour skips .gitignore
    // unless the tree contains a `.git/` dir, which surprised the tests.
    let walker = WalkBuilder::new(root_path)
        .standard_filters(true)
        .git_ignore(true)
        .git_exclude(true)
        .require_git(false)
        .hidden(true)
        .parents(true)
        .build();

    let mut indexed: usize = 0;

    for entry in walker.flatten() {
        let path = entry.path();
        if !path.is_file() || !is_allowed(path) {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if meta.len() > MAX_FILE_BYTES {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        let chunks = chunk_text(&content, DEFAULT_CHUNK_LINES, DEFAULT_CHUNK_OVERLAP);
        if chunks.is_empty() {
            continue;
        }

        // Batch the embeddings per file so the HTTP path makes far fewer
        // round-trips. The local-ONNX path is fast either way.
        let payloads_for_log = format!("{} ({} chunks)", path.display(), chunks.len());
        let embeddings_batch = match embedder.embed_batch(&chunks) {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "ai_backend: embed batch failed for {}: {}",
                    payloads_for_log, e
                );
                continue;
            }
        };
        if embeddings_batch.len() != chunks.len() {
            eprintln!(
                "ai_backend: embedder returned {} vectors for {} chunks in {}; skipping",
                embeddings_batch.len(),
                chunks.len(),
                path.display()
            );
            continue;
        }
        for (chunk, emb) in chunks.into_iter().zip(embeddings_batch) {
            // Tag chunks with their source path so retrieval surfaces the
            // file location alongside the snippet. Keeps payload one string
            // (matches MemoryManager's payload-is-string contract).
            let payload = format!("// {}\n{}", path.display(), chunk);
            if memory.try_store_embedding(payload, emb).is_ok() {
                indexed += 1;
            }
        }
    }

    Ok(indexed)
}

/// Convenience wrapper: index with the HTTP embedder.
pub fn index_codebase_impl(
    memory: &MemoryManager,
    config: &AIConfig,
    rt: &tokio::runtime::Runtime,
    client: Arc<Client>,
    root_path: &str,
) -> Result<usize, String> {
    let embedder = HttpEmbedder { client, config, rt };
    index_with_embedder(memory, &embedder, root_path)
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
/// Owns a `MemoryManager` reference, a private reqwest client + Tokio runtime
/// for the HTTP embedder, and optionally a `LocalEmbedder` instance for the
/// `local-embeddings` path. The constructor's `use_local_embeddings` flag
/// chooses which backend embeds chunks and queries.
#[cfg(feature = "python")]
#[pyclass]
pub struct RagPipeline {
    memory: Py<MemoryManager>,
    config: AIConfig,
    client: Arc<Client>,
    rt: Arc<tokio::runtime::Runtime>,
    #[cfg(feature = "local-embeddings")]
    local: Option<Arc<crate::local_embed::LocalEmbedder>>,
    use_local: bool,
}

#[cfg(feature = "python")]
#[pymethods]
impl RagPipeline {
    /// Creates a new RagPipeline.
    ///
    /// When `use_local_embeddings` is true and the crate was compiled with
    /// the `local-embeddings` feature, [`crate::local_embed::LocalEmbedder`]
    /// is loaded from the cache directory (downloading the ONNX model and
    /// tokenizer on first use). Otherwise the HTTP endpoint configured in
    /// `AIConfig` is used.
    #[new]
    #[pyo3(signature = (memory, config, use_local_embeddings=false))]
    pub fn new(
        memory: Py<MemoryManager>,
        config: AIConfig,
        use_local_embeddings: bool,
    ) -> PyResult<Self> {
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

        #[cfg(feature = "local-embeddings")]
        let local = if use_local_embeddings {
            Some(Arc::new(
                crate::local_embed::LocalEmbedder::from_default_cache()
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?,
            ))
        } else {
            None
        };

        #[cfg(not(feature = "local-embeddings"))]
        if use_local_embeddings {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Local ONNX embeddings requested but ai_backend was built without \
                 the `local-embeddings` feature. Rebuild with: \
                 `cd rust_core/ai_backend && maturin develop --features python,local-embeddings`",
            ));
        }

        Ok(Self {
            memory,
            config,
            client: Arc::new(client),
            rt: Arc::new(rt),
            #[cfg(feature = "local-embeddings")]
            local,
            use_local: use_local_embeddings,
        })
    }

    /// Whether the pipeline is using local ONNX embeddings.
    pub fn uses_local_embeddings(&self) -> bool {
        self.use_local
    }

    /// Recursively indexes a directory, chunks text, generates embeddings,
    /// and stores them via the MemoryManager.
    ///
    /// Holds the GIL while indexing — releasing it across `py.allow_threads`
    /// is blocked by the `PyRef<MemoryManager>` borrow not implementing
    /// `Ungil`. Indexing is bound by the embedder (HTTP latency or ONNX
    /// inference), neither of which contends on the GIL, so the practical
    /// cost is small.
    pub fn index_codebase(&self, py: Python, root_path: String) -> PyResult<usize> {
        let memory_ref = self.memory.borrow(py);
        let result = {
            #[cfg(feature = "local-embeddings")]
            {
                if let Some(local) = self.local.as_ref() {
                    index_with_embedder(&memory_ref, local.as_ref(), &root_path)
                } else {
                    index_codebase_impl(
                        &memory_ref,
                        &self.config,
                        &self.rt,
                        Arc::clone(&self.client),
                        &root_path,
                    )
                }
            }
            #[cfg(not(feature = "local-embeddings"))]
            {
                index_codebase_impl(
                    &memory_ref,
                    &self.config,
                    &self.rt,
                    Arc::clone(&self.client),
                    &root_path,
                )
            }
        };
        result.map_err(|e| {
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

        #[cfg(feature = "local-embeddings")]
        {
            if let Some(local) = self.local.as_ref() {
                if prompt.trim().is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Prompt cannot be empty",
                    ));
                }
                if top_k == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "top_k must be greater than 0",
                    ));
                }
                let q = local
                    .embed(&prompt)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                return memory_ref
                    .try_search(q, top_k)
                    .map_err(pyo3::exceptions::PyValueError::new_err);
            }
        }

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
        let memory = MemoryManager::try_new(":memory:".to_string()).unwrap();
        memory.try_initialize().unwrap();
        let rt = test_rt();
        let client = Arc::new(Client::new());
        let result = retrieve_context_impl(&memory, &test_config(), &rt, client, "   ", 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_retrieve_context_rejects_zero_top_k() {
        let memory = MemoryManager::try_new(":memory:".to_string()).unwrap();
        memory.try_initialize().unwrap();
        let rt = test_rt();
        let client = Arc::new(Client::new());
        let result = retrieve_context_impl(&memory, &test_config(), &rt, client, "query", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_chunk_text_emits_overlapping_windows() {
        let body = (0..100)
            .map(|i| format!("line{}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_text(&body, 40, 8);
        assert!(chunks.len() > 1);
        // Second chunk should overlap with first: starts at line (40 - 8) = 32.
        assert!(chunks[1].contains("line32"));
        assert!(chunks[1].contains("line39"));
    }

    #[test]
    fn test_chunk_text_handles_empty() {
        assert!(chunk_text("", 40, 8).is_empty());
        assert!(chunk_text("   \n\n   ", 40, 8).is_empty());
    }

    #[test]
    fn test_is_allowed_extension() {
        assert!(is_allowed(Path::new("foo.py")));
        assert!(is_allowed(Path::new("a/b/c.rs")));
        assert!(!is_allowed(Path::new("blob.bin")));
        assert!(!is_allowed(Path::new("no_ext")));
    }

    /// Fixture embedder that maps text → a deterministic dim-2 vector by
    /// hashing the byte sum. Lets us test the indexing pipeline without
    /// pulling in ORT or hitting the network.
    struct StubEmbedder;
    impl Embedder for StubEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
            let s: u32 = text.bytes().map(|b| b as u32).sum();
            Ok(vec![(s % 1000) as f32, ((s / 1000) % 1000) as f32])
        }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
            texts.iter().map(|t| self.embed(t)).collect()
        }
    }

    #[test]
    fn test_index_with_embedder_walks_dir() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.py"), "alpha\nbeta\ngamma\n").unwrap();
        std::fs::write(tmp.path().join("b.rs"), "fn x() {}\nfn y() {}\n").unwrap();
        std::fs::write(tmp.path().join("c.bin"), vec![0u8; 32]).unwrap(); // skipped

        let memory = MemoryManager::try_new(":memory:".to_string()).unwrap();
        memory.try_initialize().unwrap();
        let stub = StubEmbedder;
        let n = index_with_embedder(&memory, &stub, tmp.path().to_str().unwrap()).unwrap();
        assert!(n >= 2, "expected at least 2 chunks indexed, got {}", n);
        assert!(memory.try_count().unwrap() >= 2);
    }

    #[test]
    fn test_index_with_embedder_respects_gitignore() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(".gitignore"), "ignored.py\n").unwrap();
        std::fs::write(tmp.path().join("keep.py"), "keep me\n").unwrap();
        std::fs::write(tmp.path().join("ignored.py"), "ignore me\n").unwrap();

        let memory = MemoryManager::try_new(":memory:".to_string()).unwrap();
        memory.try_initialize().unwrap();
        let stub = StubEmbedder;
        index_with_embedder(&memory, &stub, tmp.path().to_str().unwrap()).unwrap();

        // Search should only ever return content from `keep.py`.
        let hits = memory.try_search(vec![1.0, 0.0], 10).unwrap();
        for hit in &hits {
            assert!(
                !hit.contains("ignore me"),
                "ignored.py content leaked into index: {}",
                hit
            );
        }
        assert!(hits.iter().any(|h| h.contains("keep me")));
    }
}
