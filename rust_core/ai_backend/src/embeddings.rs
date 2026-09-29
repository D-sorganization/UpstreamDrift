//! Embedding client for OpenAI-compatible embedding endpoints.
//!
//! Posts to `{config.base_url}{config.embed_path}` with the configured
//! `embedding_model` and returns the `data[0].embedding` vector. Multiple
//! inputs in a single request are supported via [`embed_batch`].
//!
//! ## Why not local ONNX?
//!
//! Local ONNX embeddings (e.g. `all-MiniLM-L6-v2` via `ort` or `candle-onnx`)
//! were considered for offline use, but ship a 20–80 MB model and pull in a
//! heavy native dependency. We use a remote-endpoint embedder as the default
//! and document local-ONNX support as a follow-up (#5167 successor) so this
//! crate stays buildable on Windows MSVC and macOS without external runtimes.

use reqwest::Client;
use std::sync::Arc;

use crate::config::AIConfig;

/// Generate a single embedding via the configured endpoint.
///
/// # Contract
/// * `text` must not be empty.
pub async fn embed_one(
    client: Arc<Client>,
    config: &AIConfig,
    text: &str,
) -> Result<Vec<f32>, String> {
    if text.trim().is_empty() {
        return Err("text cannot be empty".to_string());
    }
    let mut batch = embed_batch(client, config, &[text.to_string()]).await?;
    batch
        .pop()
        .ok_or_else(|| "Provider returned empty embedding batch".to_string())
}

/// Generate embeddings for a batch of inputs.
///
/// # Contract
/// * `inputs` must not be empty.
/// * Each input must not be empty after trimming.
pub async fn embed_batch(
    client: Arc<Client>,
    config: &AIConfig,
    inputs: &[String],
) -> Result<Vec<Vec<f32>>, String> {
    if inputs.is_empty() {
        return Err("inputs cannot be empty".to_string());
    }
    if inputs.iter().any(|s| s.trim().is_empty()) {
        return Err("inputs must not contain empty strings".to_string());
    }

    let url = config.embed_url();
    let payload = serde_json::json!({
        "model": config.embedding_model,
        "input": inputs,
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
    let data = body
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| format!("Unexpected embedding response shape: {}", body))?;

    let mut out: Vec<Vec<f32>> = Vec::with_capacity(data.len());
    for item in data {
        let arr = item
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| format!("Missing embedding array in response: {}", item))?;
        let vec: Vec<f32> = arr
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
        if vec.is_empty() {
            return Err(format!("Empty embedding vector in response: {}", item));
        }
        out.push(vec);
    }

    if out.is_empty() {
        return Err("Provider returned no embeddings".to_string());
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_one_rejects_empty() {
        let cfg =
            AIConfig::try_new("k".into(), "http://local".into(), "m".into(), "x".into()).unwrap();
        let client = Arc::new(Client::new());
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(embed_one(client, &cfg, "  "));
        assert!(result.is_err());
    }

    #[test]
    fn test_embed_batch_rejects_empty_inputs() {
        let cfg =
            AIConfig::try_new("k".into(), "http://local".into(), "m".into(), "x".into()).unwrap();
        let client = Arc::new(Client::new());
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(embed_batch(Arc::clone(&client), &cfg, &[]));
        assert!(result.is_err());
        let result = rt.block_on(embed_batch(client, &cfg, &["".to_string()]));
        assert!(result.is_err());
    }
}
