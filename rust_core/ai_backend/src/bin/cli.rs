//! Offline CLI for the ai_backend vector store.
//!
//! Subcommands:
//! * `index <path>` — recursively index a directory using the configured
//!   embedder (HTTP by default, ONNX with `--local`).
//! * `search <query> [--top-k N]` — embed the query and print the top-k
//!   matching chunks.
//! * `stats` — print stored document count.
//!
//! Designed to pre-warm the local vector store before launching the GUI,
//! and to give operators a way to debug retrieval quality without going
//! through Python.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;

use ai_backend::config::AIConfig;
use ai_backend::memory::MemoryManager;
use ai_backend::rag::{index_with_embedder, HttpEmbedder};
use reqwest::Client;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return ExitCode::from(2);
    }

    let db_path = env::var("AI_BACKEND_DB").unwrap_or_else(|_| "./ai_backend.db".to_string());
    let use_local = args.iter().any(|a| a == "--local");

    let cmd = args[1].as_str();
    let rest: Vec<&String> = args
        .iter()
        .skip(2)
        .filter(|a| a.as_str() != "--local")
        .collect();

    match cmd {
        "index" => {
            if rest.is_empty() {
                eprintln!("error: `index` requires a path argument");
                return ExitCode::from(2);
            }
            let path = rest[0].as_str();
            run_index(&db_path, path, use_local)
        }
        "search" => {
            if rest.is_empty() {
                eprintln!("error: `search` requires a query argument");
                return ExitCode::from(2);
            }
            let query = rest[0].as_str();
            let top_k = parse_top_k(&rest).unwrap_or(5);
            run_search(&db_path, query, top_k, use_local)
        }
        "stats" => run_stats(&db_path),
        "help" | "--help" | "-h" => {
            print_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("error: unknown subcommand `{}`", other);
            print_usage();
            ExitCode::from(2)
        }
    }
}

fn parse_top_k(args: &[&String]) -> Option<usize> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a.as_str() == "--top-k" {
            if let Some(n) = iter.next() {
                return n.parse().ok();
            }
        }
    }
    None
}

fn print_usage() {
    eprintln!(
        "Usage:\n  \
         ai_backend_cli index <path> [--local]\n  \
         ai_backend_cli search <query> [--top-k N] [--local]\n  \
         ai_backend_cli stats\n\n\
         Environment:\n  \
         AI_BACKEND_DB         Path to sqlite DB (default ./ai_backend.db)\n  \
         AI_BACKEND_BASE_URL   HTTP embedder base URL (default http://localhost:11434/v1)\n  \
         AI_BACKEND_API_KEY    HTTP embedder API key (default empty)\n  \
         AI_BACKEND_EMBED_MODEL Embedding model name\n  \
         UPSTREAM_DRIFT_MODEL_CACHE Cache dir for local ONNX models"
    );
}

fn build_http_config() -> AIConfig {
    let base =
        env::var("AI_BACKEND_BASE_URL").unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let key = env::var("AI_BACKEND_API_KEY").unwrap_or_default();
    let model =
        env::var("AI_BACKEND_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());
    let mut cfg = AIConfig::try_new(
        key,
        base,
        "_unused_chat_model_".to_string(),
        "_unused_".to_string(),
    )
    .expect("default config should validate");
    cfg.embedding_model = model;
    cfg
}

fn run_index(db_path: &str, path: &str, use_local: bool) -> ExitCode {
    let memory = match MemoryManager::try_new(db_path.to_string()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = memory.try_initialize() {
        eprintln!("error: {}", e);
        return ExitCode::FAILURE;
    }

    let start = std::time::Instant::now();
    let result = if use_local {
        #[cfg(feature = "local-embeddings")]
        {
            match ai_backend::local_embed::LocalEmbedder::from_default_cache() {
                Ok(local) => index_with_embedder(&memory, &local, path),
                Err(e) => Err(e),
            }
        }
        #[cfg(not(feature = "local-embeddings"))]
        {
            let _ = path;
            Err("Built without --features local-embeddings; rebuild to use --local".to_string())
        }
    } else {
        let cfg = build_http_config();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let client = Arc::new(Client::new());
        let embedder = HttpEmbedder {
            client,
            config: &cfg,
            rt: &rt,
        };
        index_with_embedder(&memory, &embedder, path)
    };

    match result {
        Ok(n) => {
            let elapsed = start.elapsed();
            println!(
                "indexed {} chunks in {:.2?} ({:.1} chunks/s)",
                n,
                elapsed,
                n as f64 / elapsed.as_secs_f64().max(1e-6)
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn run_search(db_path: &str, query: &str, top_k: usize, use_local: bool) -> ExitCode {
    let memory = match MemoryManager::try_new(db_path.to_string()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = memory.try_initialize() {
        eprintln!("error: {}", e);
        return ExitCode::FAILURE;
    }

    let query_vec_result = if use_local {
        #[cfg(feature = "local-embeddings")]
        {
            match ai_backend::local_embed::LocalEmbedder::from_default_cache() {
                Ok(local) => local.embed(query),
                Err(e) => Err(e),
            }
        }
        #[cfg(not(feature = "local-embeddings"))]
        {
            let _ = query;
            Err("Built without --features local-embeddings".to_string())
        }
    } else {
        let cfg = build_http_config();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let client = Arc::new(Client::new());
        rt.block_on(ai_backend::embeddings::embed_one(client, &cfg, query))
    };

    let query_vec = match query_vec_result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    match memory.try_search(query_vec, top_k) {
        Ok(hits) => {
            for (i, h) in hits.iter().enumerate() {
                println!("=== hit {} ===", i + 1);
                println!("{}", h);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn run_stats(db_path: &str) -> ExitCode {
    let memory = match MemoryManager::try_new(db_path.to_string()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = memory.try_initialize() {
        eprintln!("error: {}", e);
        return ExitCode::FAILURE;
    }
    match memory.try_count() {
        Ok(n) => {
            println!("{} documents stored at {}", n, db_path);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::FAILURE
        }
    }
}
