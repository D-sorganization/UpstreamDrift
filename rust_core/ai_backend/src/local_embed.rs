//! Local ONNX-based sentence embeddings.
//!
//! Uses [`ort`] (ONNX Runtime) + [`tokenizers`] to run a small sentence-
//! transformers model (default: `sentence-transformers/all-MiniLM-L6-v2`,
//! 384-dim on-device; model artifacts are revision-pinned and SHA-256 verified.
//! Gated behind the `local-embeddings` Cargo feature so plain `cargo build`
//! and the default maturin wheel stay slim. Consumers opt in by building
//! with `--features local-embeddings`.

#![cfg(feature = "local-embeddings")]

use ndarray::{Array1, Array2};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Default model identifier (HuggingFace repo). The ONNX-converted variant
/// of `sentence-transformers/all-MiniLM-L6-v2` is hosted by `optimum`.
pub const DEFAULT_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

const MODEL_LOCK: &str = include_str!("../local_embedding_model.lock.json");

/// Expected output dimension for the default model.
pub const DEFAULT_EMBEDDING_DIM: usize = 384;

/// Maximum sequence length the model accepts. Inputs longer than this are
/// truncated by the tokenizer.
const MAX_SEQ_LEN: usize = 256;

/// A reusable local embedder. Loading is expensive (~100 ms) so callers
/// should hold this instance for the lifetime of the indexing job.
pub struct LocalEmbedder {
    // `Session::run` requires `&mut self`. Wrap in a Mutex so the embedder
    // is `Send + Sync` and the same instance can be shared across the
    // PyO3 boundary without forcing the caller to manage interior mutability.
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dim: usize,
}

impl LocalEmbedder {
    /// Construct using the default model, downloading to the cache dir if
    /// necessary. Returns an error if download or model parse fails.
    pub fn from_default_cache() -> Result<Self, String> {
        let lock: serde_json::Value = serde_json::from_str(MODEL_LOCK)
            .map_err(|e| format!("Invalid local embedding model lock: {}", e))?;
        let revision = lock["revision"]
            .as_str()
            .ok_or("Model lock has no revision")?;
        let artifacts = lock["artifacts"]
            .as_object()
            .ok_or("Model lock has no artifacts")?;
        let artifact = |name: &str| -> Result<(String, String, String), String> {
            let item = artifacts
                .get(name)
                .ok_or_else(|| format!("Model lock missing {}", name))?;
            let path = item["path"]
                .as_str()
                .ok_or_else(|| format!("{} has no path", name))?;
            let cache_name = item["cache_name"]
                .as_str()
                .ok_or_else(|| format!("{} has no cache name", name))?;
            let sha256 = item["sha256"]
                .as_str()
                .ok_or_else(|| format!("{} has no SHA-256", name))?;
            Ok((
                format!(
                    "https://huggingface.co/{}/resolve/{}/{}",
                    DEFAULT_MODEL_ID, revision, path
                ),
                cache_name.to_owned(),
                sha256.to_owned(),
            ))
        };
        let (model_url, model_name, model_sha) = artifact("model")?;
        let (tokenizer_url, tokenizer_name, tokenizer_sha) = artifact("tokenizer")?;
        let cache_dir = resolve_cache_dir()?;
        fs::create_dir_all(&cache_dir)
            .map_err(|e| format!("Failed to create cache dir {}: {}", cache_dir.display(), e))?;

        let model_path = cache_dir.join(model_name);
        let tokenizer_path = cache_dir.join(tokenizer_name);

        let offline = std::env::var("UPSTREAM_DRIFT_MODEL_OFFLINE").as_deref() == Ok("1");
        ensure_artifact(&model_url, &model_path, &model_sha, offline)
            .map_err(|e| format!("Model artifact unavailable: {}", e))?;
        ensure_artifact(&tokenizer_url, &tokenizer_path, &tokenizer_sha, offline)
            .map_err(|e| format!("Tokenizer artifact unavailable: {}", e))?;

        Self::from_paths(&model_path, &tokenizer_path)
    }

    /// Construct from explicit file paths. Useful for tests and for callers
    /// that want to bundle the model in a wheel.
    pub fn from_paths(model_path: &Path, tokenizer_path: &Path) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Tokenizer load error: {}", e))?;

        let session = Session::builder()
            .map_err(|e| format!("ORT session builder error: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ORT optimization-level error: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("ORT model load error: {}", e))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dim: DEFAULT_EMBEDDING_DIM,
        })
    }

    /// Output dimension (384 for `all-MiniLM-L6-v2`).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Embed a single string. See [`Self::embed_batch`] for batched calls.
    ///
    /// # Contract
    /// * `text` must not be empty (returns Err otherwise).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        if text.trim().is_empty() {
            return Err("text cannot be empty".to_string());
        }
        let mut batch = self.embed_batch(&[text.to_string()])?;
        Ok(batch.remove(0))
    }

    /// Embed a batch. Each input is truncated to `MAX_SEQ_LEN` tokens; the
    /// model is run once over the padded batch and the last-hidden-state is
    /// mean-pooled (attention-mask weighted) to produce one f32 vector per
    /// input.
    pub fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if inputs.is_empty() {
            return Err("inputs cannot be empty".to_string());
        }

        let encodings = self
            .tokenizer
            .encode_batch(inputs.iter().map(|s| s.as_str()).collect::<Vec<_>>(), true)
            .map_err(|e| format!("Tokenizer encode error: {}", e))?;

        // Find batch max len (post-truncation) so padding is minimal.
        let batch = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(MAX_SEQ_LEN))
            .max()
            .unwrap_or(1)
            .max(1);

        let mut input_ids = Array2::<i64>::zeros((batch, max_len));
        let mut attention_mask = Array2::<i64>::zeros((batch, max_len));
        let mut token_type_ids = Array2::<i64>::zeros((batch, max_len));

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let type_ids = enc.get_type_ids();
            let n = ids.len().min(max_len);
            for j in 0..n {
                input_ids[[i, j]] = ids[j] as i64;
                attention_mask[[i, j]] = mask[j] as i64;
                token_type_ids[[i, j]] = type_ids[j] as i64;
            }
        }

        let inputs_map = ort::inputs![
            "input_ids" => Value::from_array(input_ids.clone()).map_err(|e| format!("ORT input_ids error: {}", e))?,
            "attention_mask" => Value::from_array(attention_mask.clone()).map_err(|e| format!("ORT attention_mask error: {}", e))?,
            "token_type_ids" => Value::from_array(token_type_ids).map_err(|e| format!("ORT token_type_ids error: {}", e))?,
        ];

        // `SessionOutputs` borrows from the `Session`, so we have to extract
        // the hidden-state tensor into an owned ndarray before releasing the
        // mutex guard. Holding the guard for the entire scoring loop would
        // serialize concurrent embed calls; today that's fine (we only
        // construct one embedder), and we can shard later if needed.
        let owned: ndarray::Array3<f32> = {
            let mut session = self
                .session
                .lock()
                .map_err(|_| "ORT session mutex poisoned".to_string())?;
            let outputs = session
                .run(inputs_map)
                .map_err(|e| format!("ORT run error: {}", e))?;
            let (_name, last_hidden) = outputs
                .iter()
                .next()
                .ok_or_else(|| "ORT returned no outputs".to_string())?;
            let view = last_hidden
                .try_extract_array::<f32>()
                .map_err(|e| format!("ORT output extract error: {}", e))?;
            let view = view
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| format!("Expected (batch, seq, hidden) output but got: {}", e))?;
            view.to_owned()
        };

        let view = owned.view();
        let (b, s, h) = view.dim();
        if b != batch || s != max_len {
            return Err(format!(
                "Output shape mismatch: expected ({}, {}, *), got ({}, {}, {})",
                batch, max_len, b, s, h
            ));
        }

        // Mean-pool with attention mask: sum(hidden * mask) / sum(mask).
        let mut results = Vec::with_capacity(batch);
        for bi in 0..batch {
            let mut pooled = Array1::<f32>::zeros(h);
            let mut denom = 0f32;
            for si in 0..max_len {
                let m = attention_mask[[bi, si]] as f32;
                if m == 0.0 {
                    continue;
                }
                denom += m;
                for hi in 0..h {
                    pooled[hi] += view[[bi, si, hi]] * m;
                }
            }
            if denom > 0.0 {
                pooled.mapv_inplace(|v| v / denom);
            }
            // L2-normalize so cosine similarity == dot product downstream.
            let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            pooled.mapv_inplace(|v| v / norm);
            results.push(pooled.into_raw_vec_and_offset().0);
        }

        Ok(results)
    }
}

/// Resolve the model cache directory: `$UPSTREAM_DRIFT_MODEL_CACHE` if set,
/// else `~/.cache/upstream-drift/models/`.
pub fn resolve_cache_dir() -> Result<PathBuf, String> {
    if let Ok(dir) = std::env::var("UPSTREAM_DRIFT_MODEL_CACHE") {
        return Ok(PathBuf::from(dir));
    }
    let home = dirs::cache_dir()
        .or_else(dirs::home_dir)
        .ok_or_else(|| "Could not resolve cache dir (no $HOME)".to_string())?;
    Ok(home.join("upstream-drift").join("models"))
}

/// Download a URL into a file path using a blocking reqwest client.
fn download_to(url: &str, dest: &Path, expected_sha256: &str) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| e.to_string())?;
    let mut resp = client
        .get(url)
        .send()
        .map_err(|e| format!("HTTP error: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {} for {}", resp.status(), url));
    }
    let tmp = dest.with_extension("partial");
    {
        let mut file = fs::File::create(&tmp)
            .map_err(|e| format!("Failed to create {}: {}", tmp.display(), e))?;
        std::io::copy(&mut resp, &mut file).map_err(|e| format!("Write error: {}", e))?;
    }
    if let Err(error) = verify_sha256(&tmp, expected_sha256) {
        let _ = fs::remove_file(&tmp);
        return Err(error);
    }
    fs::rename(&tmp, dest).map_err(|e| {
        format!(
            "Failed to move {} to {}: {}",
            tmp.display(),
            dest.display(),
            e
        )
    })?;
    Ok(())
}
fn ensure_artifact(
    url: &str,
    path: &Path,
    expected_sha256: &str,
    offline: bool,
) -> Result<(), String> {
    if !path.exists() {
        if offline {
            return Err(format!("offline cache is missing {}", path.display()));
        }
        download_to(url, path, expected_sha256)?;
    }
    verify_sha256(path, expected_sha256)
}

fn verify_sha256(path: &Path, expected: &str) -> Result<(), String> {
    let actual = sha256_of_file(path)?;
    if actual != expected {
        return Err(format!(
            "SHA-256 mismatch for {}: {expected} != {actual}",
            path.display()
        ));
    }
    Ok(())
}

/// Compute SHA-256 of a file, hex-encoded. Used for integrity checks when
/// callers want to verify a pinned model artifact.
pub fn sha256_of_file(path: &Path) -> Result<String, String> {
    let mut file =
        fs::File::open(path).map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("Read error: {}", e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_cache_dir_respects_env_override() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("UPSTREAM_DRIFT_MODEL_CACHE", tmp.path());
        let resolved = resolve_cache_dir().unwrap();
        assert_eq!(resolved, tmp.path());
        std::env::remove_var("UPSTREAM_DRIFT_MODEL_CACHE");
    }

    #[test]
    fn test_model_lock_has_immutable_revision_and_sha256_digests() {
        let lock: serde_json::Value = serde_json::from_str(MODEL_LOCK).unwrap();
        assert_eq!(lock["revision"].as_str().unwrap().len(), 40);
        for artifact in lock["artifacts"].as_object().unwrap().values() {
            assert_eq!(artifact["sha256"].as_str().unwrap().len(), 64);
            assert!(artifact["size_bytes"].as_u64().unwrap() > 0);
        }
    }

    #[test]
    fn test_offline_cache_requires_present_digest_verified_artifacts() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("missing");
        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        probe.set_nonblocking(true).unwrap();
        let url = format!("http://{}/model", probe.local_addr().unwrap());
        let error = ensure_artifact(&url, &missing, "00", true).unwrap_err();
        assert!(error.contains("offline cache is missing"));
        assert_eq!(
            probe.accept().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );

        let corrupt = tmp.path().join("corrupt");
        fs::write(&corrupt, b"wrong artifact").unwrap();
        let error = ensure_artifact(&url, &corrupt, "00", true).unwrap_err();
        assert!(error.contains("SHA-256 mismatch"));
    }
}
