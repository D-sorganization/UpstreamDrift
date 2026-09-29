//! Configuration for the AI client and RAG system.
//!
//! Maintains Design by Contract (DbC) by ensuring valid configurations
//! at instantiation time. The pure-Rust core (`AIConfig` struct + `validate`)
//! is always compiled; PyO3 bindings are added under the `python` feature.

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Default chat completion path appended to `base_url` when none is supplied
/// (OpenAI-compatible providers: OpenAI, Azure OpenAI, Together, Groq, Ollama
/// with `/v1`, vLLM, LM Studio, etc.).
pub const DEFAULT_CHAT_PATH: &str = "/chat/completions";

/// Default embedding path appended to `base_url` when none is supplied
/// (OpenAI-compatible providers).
pub const DEFAULT_EMBED_PATH: &str = "/embeddings";

/// Default embedding model — `text-embedding-3-small` is a reasonable default
/// for OpenAI-compatible providers and produces 1536-dim vectors. Callers
/// targeting other providers (Voyage, Cohere, local Ollama) should override.
pub const DEFAULT_EMBED_MODEL: &str = "text-embedding-3-small";

/// Configuration for the AI client and RAG system.
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug)]
pub struct AIConfig {
    pub api_key: String,
    pub base_url: String,
    pub model_name: String,
    pub db_path: String,
    /// Path suffix appended to `base_url` for chat completions. Defaults to
    /// `/chat/completions` (OpenAI-compatible). Override for providers that
    /// require a different route (e.g. Anthropic `/v1/messages`).
    pub chat_path: String,
    /// Path suffix appended to `base_url` for embeddings. Defaults to
    /// `/embeddings` (OpenAI-compatible).
    pub embed_path: String,
    /// Embedding model name. Decoupled from `model_name` (which is the chat
    /// model) so chat and embedding can target different models or providers
    /// when the API surface allows it.
    pub embedding_model: String,
}

impl AIConfig {
    /// Validate the public invariants of an `AIConfig`.
    ///
    /// # Contract
    /// * `base_url` must not be empty.
    /// * `model_name` must not be empty.
    pub fn validate(&self) -> Result<(), String> {
        if self.base_url.trim().is_empty() {
            return Err("base_url cannot be empty".to_string());
        }
        if self.model_name.trim().is_empty() {
            return Err("model_name cannot be empty".to_string());
        }
        Ok(())
    }

    /// Pure-Rust constructor used by tests and internal code.
    pub fn try_new(
        api_key: String,
        base_url: String,
        model_name: String,
        db_path: String,
    ) -> Result<Self, String> {
        let config = Self {
            api_key,
            base_url,
            model_name,
            db_path,
            chat_path: DEFAULT_CHAT_PATH.to_string(),
            embed_path: DEFAULT_EMBED_PATH.to_string(),
            embedding_model: DEFAULT_EMBED_MODEL.to_string(),
        };
        config.validate()?;
        Ok(config)
    }

    /// Resolve the full chat-completions URL by joining `base_url` with
    /// `chat_path`. Handles trailing/leading slashes so callers don't have
    /// to think about it. If `chat_path` is empty, returns `base_url`
    /// unmodified (escape hatch for callers that pre-baked the full URL).
    pub fn chat_url(&self) -> String {
        Self::join_url(&self.base_url, &self.chat_path)
    }

    /// Resolve the full embeddings URL.
    pub fn embed_url(&self) -> String {
        Self::join_url(&self.base_url, &self.embed_path)
    }

    fn join_url(base: &str, path: &str) -> String {
        let base = base.trim_end_matches('/');
        let path = path.trim();
        if path.is_empty() {
            return base.to_string();
        }
        if path.starts_with("http://") || path.starts_with("https://") {
            return path.to_string();
        }
        // Handle path segments that may have overlapping prefixes
        // e.g., base="http://localhost:11434/v1" + path="/v1/chat/completions"
        // should produce "http://localhost:11434/v1/chat/completions" not "...v1/v1/..."
        if path.starts_with('/') {
            // Check if path starts with a segment that base already ends with
            let path_segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
            if let Some(first_segment) = path_segments.first() {
                let base_segments: Vec<&str> = base.split('/').filter(|s| !s.is_empty()).collect();
                if let Some(last_segment) = base_segments.last() {
                    if first_segment == last_segment {
                        // Skip the first segment of path since it duplicates the last segment of base
                        let remaining_path = path_segments[1..].join("/");
                        if remaining_path.is_empty() {
                            return base.to_string();
                        }
                        return format!("{}/{}", base, remaining_path);
                    }
                }
            }
            format!("{}{}", base, path)
        } else {
            format!("{}/{}", base, path)
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl AIConfig {
    /// Creates a new AIConfig.
    ///
    /// # Contract
    /// * `base_url` must not be empty.
    /// * `model_name` must not be empty.
    #[new]
    #[pyo3(signature = (api_key, base_url, model_name, db_path, chat_path=None, embed_path=None, embedding_model=None))]
    pub fn new(
        api_key: String,
        base_url: String,
        model_name: String,
        db_path: String,
        chat_path: Option<String>,
        embed_path: Option<String>,
        embedding_model: Option<String>,
    ) -> PyResult<Self> {
        let mut cfg = Self::try_new(api_key, base_url, model_name, db_path)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        if let Some(p) = chat_path {
            cfg.chat_path = p;
        }
        if let Some(p) = embed_path {
            cfg.embed_path = p;
        }
        if let Some(m) = embedding_model {
            cfg.embedding_model = m;
        }
        Ok(cfg)
    }

    /// Resolved chat URL (read-only helper for Python callers).
    #[pyo3(name = "chat_url")]
    fn py_chat_url(&self) -> String {
        self.chat_url()
    }

    /// Resolved embed URL.
    #[pyo3(name = "embed_url")]
    fn py_embed_url(&self) -> String {
        self.embed_url()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_config() {
        let config = AIConfig::try_new(
            "key".to_string(),
            "https://api.openai.com/v1".to_string(),
            "gpt-4".to_string(),
            "./memory.db".to_string(),
        )
        .unwrap();
        assert_eq!(config.model_name, "gpt-4");
        assert_eq!(config.chat_path, DEFAULT_CHAT_PATH);
    }

    #[test]
    fn test_invalid_config_empty_url() {
        let config = AIConfig::try_new(
            "key".to_string(),
            "   ".to_string(),
            "gpt-4".to_string(),
            "./memory.db".to_string(),
        );
        assert!(config.is_err());
    }

    #[test]
    fn test_chat_url_default() {
        let cfg = AIConfig::try_new(
            "k".into(),
            "https://api.openai.com/v1".into(),
            "gpt-4".into(),
            "x".into(),
        )
        .unwrap();
        assert_eq!(cfg.chat_url(), "https://api.openai.com/v1/chat/completions");
        assert_eq!(cfg.embed_url(), "https://api.openai.com/v1/embeddings");
    }

    #[test]
    fn test_chat_url_handles_trailing_slash() {
        let cfg = AIConfig::try_new(
            "k".into(),
            "https://api.openai.com/v1/".into(),
            "m".into(),
            "x".into(),
        )
        .unwrap();
        assert_eq!(cfg.chat_url(), "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn test_chat_url_handles_relative_path_no_leading_slash() {
        let mut cfg = AIConfig::try_new(
            "k".into(),
            "https://api.openai.com/v1".into(),
            "m".into(),
            "x".into(),
        )
        .unwrap();
        cfg.chat_path = "chat/completions".into();
        assert_eq!(cfg.chat_url(), "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn test_chat_url_absolute_override() {
        let mut cfg = AIConfig::try_new(
            "k".into(),
            "https://api.openai.com/v1".into(),
            "m".into(),
            "x".into(),
        )
        .unwrap();
        cfg.chat_path = "https://other.example/v2/chat".into();
        assert_eq!(cfg.chat_url(), "https://other.example/v2/chat");
    }

    #[test]
    fn test_chat_url_empty_path_returns_base() {
        let mut cfg = AIConfig::try_new(
            "k".into(),
            "https://full.example/chat".into(),
            "m".into(),
            "x".into(),
        )
        .unwrap();
        cfg.chat_path = "".into();
        assert_eq!(cfg.chat_url(), "https://full.example/chat");
    }

    #[test]
    fn test_chat_url_handles_duplicate_path_segments() {
        // Test case for issue #5307: users who configure ollama_host with /v1 suffix
        // should not get /v1/v1/... URLs when default paths are applied
        let mut cfg = AIConfig::try_new(
            "k".into(),
            "http://localhost:11434/v1".into(),
            "llama3.1:8b".into(),
            "x".into(),
        )
        .unwrap();
        cfg.chat_path = "/v1/chat/completions".into();
        cfg.embed_path = "/v1/embeddings".into();
        // Should deduplicate the /v1 segment
        assert_eq!(cfg.chat_url(), "http://localhost:11434/v1/chat/completions");
        assert_eq!(cfg.embed_url(), "http://localhost:11434/v1/embeddings");
    }
}
