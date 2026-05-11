use pyo3::prelude::*;

/// Configuration for the AI client and RAG system.
/// Maintains Design by Contract (DbC) by ensuring valid configurations
/// at instantiation time.
#[pyclass]
#[derive(Clone, Debug)]
pub struct AIConfig {
    #[pyo3(get, set)]
    pub api_key: String,
    #[pyo3(get, set)]
    pub base_url: String,
    #[pyo3(get, set)]
    pub model_name: String,
    #[pyo3(get, set)]
    pub db_path: String,
}

#[pymethods]
impl AIConfig {
    /// Creates a new AIConfig.
    ///
    /// # Contract
    /// * `base_url` must not be empty.
    /// * `model_name` must not be empty.
    #[new]
    pub fn new(api_key: String, base_url: String, model_name: String, db_path: String) -> PyResult<Self> {
        if base_url.trim().is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("base_url cannot be empty"));
        }
        if model_name.trim().is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("model_name cannot be empty"));
        }

        Ok(Self {
            api_key,
            base_url,
            model_name,
            db_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_config() {
        let config = AIConfig::new(
            "key".to_string(),
            "https://api.openai.com/v1".to_string(),
            "gpt-4".to_string(),
            "./memory.db".to_string(),
        ).unwrap();
        assert_eq!(config.model_name, "gpt-4");
    }

    #[test]
    fn test_invalid_config_empty_url() {
        let config = AIConfig::new(
            "key".to_string(),
            "   ".to_string(),
            "gpt-4".to_string(),
            "./memory.db".to_string(),
        );
        assert!(config.is_err());
    }
}
