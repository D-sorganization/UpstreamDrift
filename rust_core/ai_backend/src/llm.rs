//! Core AI Engine: OpenAI-compatible LLM client with real streaming.
//!
//! Follows Law of Demeter (LoD) by fully encapsulating the reqwest Client
//! and async runtime, exposing only high-level business methods to Python.
//!
//! ## Surface
//!
//! - [`AIEngine::try_generate_response`] — blocking, single-shot completion.
//!   Parses the OpenAI-compatible response envelope and returns the
//!   `choices[0].message.content` string.
//! - [`AIEngine::try_stream_response`] — blocking call that returns a `Vec`
//!   of streamed delta chunks. Each chunk is an incremental `content` delta
//!   from the SSE stream. This drains the stream eagerly; the PyO3 wrapper
//!   exposes this as a Python list which `RustAgentAdapter` iterates over.
//!   (Truly-incremental streaming across the PyO3 boundary requires a
//!   `__next__`-style iterator and is tracked as a follow-up — collecting
//!   the deltas server-side and yielding them as multiple chunks is the
//!   pragmatic first step and is a real improvement over "blocking until
//!   complete, then emit one chunk".)

#[cfg(feature = "python")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use futures::StreamExt;
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;

use crate::config::AIConfig;

/// The core AI Engine that manages connections and state to the LLM.
#[cfg_attr(feature = "python", pyclass)]
pub struct AIEngine {
    config: AIConfig,
    client: Arc<Client>,
    rt: Arc<tokio::runtime::Runtime>,
}

impl AIEngine {
    /// Pure-Rust constructor.
    pub fn try_new(config: AIConfig) -> Result<Self, String> {
        config.validate()?;

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("Failed to build Tokio runtime: {}", e))?;

        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| format!("Failed to build reqwest Client: {}", e))?;

        Ok(Self {
            config,
            client: Arc::new(client),
            rt: Arc::new(rt),
        })
    }

    /// Pure-Rust synchronous response generation.
    ///
    /// # Contract
    /// * `prompt` must not be empty.
    pub fn try_generate_response(&self, prompt: String) -> Result<String, String> {
        if prompt.trim().is_empty() {
            return Err("Prompt cannot be empty".to_string());
        }

        let config = self.config.clone();
        let client = Arc::clone(&self.client);

        self.rt
            .block_on(async move { Self::generate_response_async(client, config, prompt).await })
            .map_err(|e| format!("API Request failed: {}", e))
    }

    /// Blocking call that drains an SSE stream and returns the ordered list
    /// of content deltas. See module docs for the streaming-surface caveats.
    ///
    /// # Contract
    /// * `prompt` must not be empty.
    pub fn try_stream_response(&self, prompt: String) -> Result<Vec<String>, String> {
        if prompt.trim().is_empty() {
            return Err("Prompt cannot be empty".to_string());
        }

        let config = self.config.clone();
        let client = Arc::clone(&self.client);

        self.rt
            .block_on(async move { Self::stream_response_async(client, config, prompt).await })
            .map_err(|e| format!("API Stream failed: {}", e))
    }

    async fn generate_response_async(
        client: Arc<Client>,
        config: AIConfig,
        prompt: String,
    ) -> Result<String, String> {
        let url = config.chat_url();
        let payload = serde_json::json!({
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": false,
        });

        let resp = client
            .post(&url)
            .bearer_auth(&config.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("HTTP {}: {}", status, body));
        }

        let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        extract_content(&body)
    }

    async fn stream_response_async(
        client: Arc<Client>,
        config: AIConfig,
        prompt: String,
    ) -> Result<Vec<String>, String> {
        let url = config.chat_url();
        let payload = serde_json::json!({
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": true,
        });

        let resp = client
            .post(&url)
            .bearer_auth(&config.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("HTTP {}: {}", status, body));
        }

        let mut deltas: Vec<String> = Vec::new();
        let mut byte_stream = resp.bytes_stream();
        let mut buf = String::new();

        while let Some(chunk) = byte_stream.next().await {
            let bytes = chunk.map_err(|e| e.to_string())?;
            buf.push_str(&String::from_utf8_lossy(&bytes));

            // SSE frames are separated by \n\n. Drain any complete frames.
            while let Some(idx) = buf.find("\n\n") {
                let frame = buf[..idx].to_string();
                buf.drain(..idx + 2);
                if let Some(delta) = parse_sse_frame(&frame) {
                    if !delta.is_empty() {
                        deltas.push(delta);
                    }
                }
            }
        }

        // Drain trailing partial frame, if any.
        if !buf.trim().is_empty() {
            if let Some(delta) = parse_sse_frame(&buf) {
                if !delta.is_empty() {
                    deltas.push(delta);
                }
            }
        }

        Ok(deltas)
    }
}

/// Extract `choices[0].message.content` from an OpenAI-compatible response.
/// Falls back to returning the raw JSON string if the structure is not as
/// expected so callers can debug provider deviations without a panic.
fn extract_content(body: &serde_json::Value) -> Result<String, String> {
    body.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("Unexpected response shape: {}", body))
}

/// Parse a single SSE frame and return the content delta if present.
/// Returns `None` for the terminator `[DONE]` or frames without content.
fn parse_sse_frame(frame: &str) -> Option<String> {
    for line in frame.lines() {
        let line = line.trim();
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" || data.is_empty() {
            return None;
        }
        let parsed: serde_json::Value = serde_json::from_str(data).ok()?;
        let delta = parsed
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(|c| c.as_str());
        if let Some(d) = delta {
            return Some(d.to_string());
        }
    }
    None
}

#[cfg(feature = "python")]
#[pymethods]
impl AIEngine {
    /// Creates a new AIEngine.
    #[new]
    pub fn new(config: AIConfig) -> PyResult<Self> {
        Self::try_new(config).map_err(PyRuntimeError::new_err)
    }

    /// Synchronous method to send a prompt and get a response.
    ///
    /// # Contract
    /// * `prompt` must not be empty.
    pub fn generate_response(&self, prompt: String) -> PyResult<String> {
        self.try_generate_response(prompt).map_err(|e| {
            if e == "Prompt cannot be empty" {
                pyo3::exceptions::PyValueError::new_err(e)
            } else {
                PyRuntimeError::new_err(e)
            }
        })
    }

    /// Streaming variant: drains the SSE stream and returns the ordered list
    /// of incremental content deltas. The caller is expected to iterate the
    /// list and surface each delta to the UI. See the module docstring for
    /// the surface-shape rationale.
    pub fn stream_response(&self, prompt: String) -> PyResult<Vec<String>> {
        self.try_stream_response(prompt).map_err(|e| {
            if e == "Prompt cannot be empty" {
                pyo3::exceptions::PyValueError::new_err(e)
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
    fn test_engine_rejects_empty_prompt() {
        let config = AIConfig::try_new(
            "key".to_string(),
            "http://local".to_string(),
            "model".to_string(),
            "db".to_string(),
        )
        .unwrap();
        let engine = AIEngine::try_new(config).unwrap();
        let result = engine.try_generate_response("   ".to_string());
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Prompt cannot be empty");
    }

    #[test]
    fn test_extract_content_happy_path() {
        let body = serde_json::json!({
            "choices": [{"message": {"content": "hello"}}]
        });
        assert_eq!(extract_content(&body).unwrap(), "hello");
    }

    #[test]
    fn test_extract_content_unexpected_shape() {
        let body = serde_json::json!({"foo": "bar"});
        assert!(extract_content(&body).is_err());
    }

    #[test]
    fn test_parse_sse_frame_content_delta() {
        let frame = "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}";
        assert_eq!(parse_sse_frame(frame), Some("Hi".to_string()));
    }

    #[test]
    fn test_parse_sse_frame_done() {
        assert_eq!(parse_sse_frame("data: [DONE]"), None);
    }

    #[test]
    fn test_parse_sse_frame_no_content() {
        let frame = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}";
        assert_eq!(parse_sse_frame(frame), None);
    }
}
