//! Core AI Engine that manages connections and state to the LLM.
//!
//! Follows Law of Demeter (LoD) by fully encapsulating the reqwest Client
//! and async runtime, exposing only high-level business methods to Python.

#[cfg(feature = "python")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use reqwest::Client;
use std::sync::Arc;

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
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("Failed to build Tokio runtime: {}", e))?;

        Ok(Self {
            config,
            client: Arc::new(Client::new()),
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

    async fn generate_response_async(
        client: Arc<Client>,
        config: AIConfig,
        prompt: String,
    ) -> Result<String, String> {
        let payload = serde_json::json!({
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}]
        });

        // Use base_url as the endpoint. In a real system, we'd append paths like /chat/completions.
        let resp = client
            .post(&config.base_url)
            .bearer_auth(&config.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if !resp.status().is_success() {
            return Err(format!("HTTP Error: {}", resp.status()));
        }

        let text = resp.text().await.map_err(|e| e.to_string())?;
        Ok(text)
    }
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
}
