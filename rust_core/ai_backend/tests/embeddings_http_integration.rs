//! Integration tests for the embeddings HTTP endpoint.
//!
//! Gated to `not(feature = "python")` — see the module docstring on
//! `llm_http_integration.rs` for the macOS-arm64 linker rationale.
#![cfg(not(feature = "python"))]

use std::sync::Arc;

use ai_backend::config::AIConfig;
use ai_backend::embeddings;
use reqwest::Client;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test(flavor = "multi_thread")]
async fn embed_one_parses_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4], "index": 0}],
            "model": "text-embedding-3-small",
            "object": "list"
        })))
        .mount(&server)
        .await;

    let cfg = AIConfig::try_new(
        "k".into(),
        format!("{}/v1", server.uri()),
        "gpt".into(),
        ":memory:".into(),
    )
    .unwrap();

    let client = Arc::new(Client::new());
    let v = embeddings::embed_one(client, &cfg, "hello").await.unwrap();
    assert_eq!(v.len(), 4);
    assert!((v[0] - 0.1).abs() < 1e-6);
}

#[tokio::test(flavor = "multi_thread")]
async fn embed_batch_returns_multiple_vectors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [
                {"embedding": [0.1, 0.2], "index": 0},
                {"embedding": [0.3, 0.4], "index": 1},
            ]
        })))
        .mount(&server)
        .await;

    let cfg = AIConfig::try_new(
        "k".into(),
        format!("{}/v1", server.uri()),
        "gpt".into(),
        ":memory:".into(),
    )
    .unwrap();

    let client = Arc::new(Client::new());
    let vecs = embeddings::embed_batch(client, &cfg, &["a".into(), "b".into()])
        .await
        .unwrap();
    assert_eq!(vecs.len(), 2);
    assert_eq!(vecs[1], vec![0.3_f32, 0.4]);
}

#[tokio::test(flavor = "multi_thread")]
async fn embed_one_surfaces_http_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server)
        .await;

    let cfg = AIConfig::try_new(
        "k".into(),
        format!("{}/v1", server.uri()),
        "gpt".into(),
        ":memory:".into(),
    )
    .unwrap();

    let client = Arc::new(Client::new());
    let err = embeddings::embed_one(client, &cfg, "hello")
        .await
        .unwrap_err();
    assert!(err.contains("500"));
}
