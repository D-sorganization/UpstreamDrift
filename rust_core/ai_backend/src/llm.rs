use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::sync::Arc;
use reqwest::Client;

use crate::config::AIConfig;

/// The core AI Engine that manages connections and state to the LLM.
/// Follows Law of Demeter (LoD) by fully encapsulating the reqwest Client
/// and async runtime, exposing only high-level business methods to Python.
#[pyclass]
pub struct AIEngine {
    config: AIConfig,
    client: Arc<Client>,
    rt: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl AIEngine {
    /// Creates a new AIEngine.
    #[new]
    pub fn new(config: AIConfig) -> PyResult<Self> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build Tokio runtime: {}", e)))?;
            
        Ok(Self {
            config,
            client: Arc::new(Client::new()),
            rt: Arc::new(rt),
        })
    }

    /// Synchronous method to send a prompt and get a response.
    ///
    /// # Contract
    /// * `prompt` must not be empty.
    pub fn generate_response(&self, prompt: String) -> PyResult<String> {
        if prompt.trim().is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("Prompt cannot be empty"));
        }

        let config = self.config.clone();
        let client = Arc::clone(&self.client);

        // Execute async logic synchronously for Python integration
        self.rt.block_on(async move {
            Self::generate_response_async(client, config, prompt).await
        }).map_err(|e| PyRuntimeError::new_err(format!("API Request failed: {}", e)))
    }
}

impl AIEngine {
    async fn generate_response_async(client: Arc<Client>, config: AIConfig, prompt: String) -> Result<String, String> {
        let payload = serde_json::json!({
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}]
        });

        // Use base_url as the endpoint. In a real system, we'd append paths like /chat/completions.
        let resp = client.post(&config.base_url)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_rejects_empty_prompt() {
        let config = AIConfig::new("key".to_string(), "http://local".to_string(), "model".to_string(), "db".to_string()).unwrap();
        let engine = AIEngine::new(config).unwrap();
        let result = engine.generate_response("   ".to_string());
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "ValueError: Prompt cannot be empty");
    }
}
