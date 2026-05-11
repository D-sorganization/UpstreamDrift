//! Configuration for the AI client and RAG system.
//!
//! Maintains Design by Contract (DbC) by ensuring valid configurations
//! at instantiation time. The pure-Rust core (`AIConfig` struct + `validate`)
//! is always compiled; PyO3 bindings are added under the `python` feature.

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Configuration for the AI client and RAG system.
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug)]
pub struct AIConfig {
    pub api_key: String,
    pub base_url: String,
    pub model_name: String,
    pub db_path: String,
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
        };
        config.validate()?;
        Ok(config)
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
    pub fn new(
        api_key: String,
        base_url: String,
        model_name: String,
        db_path: String,
    ) -> PyResult<Self> {
        Self::try_new(api_key, base_url, model_name, db_path)
            .map_err(pyo3::exceptions::PyValueError::new_err)
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
}
