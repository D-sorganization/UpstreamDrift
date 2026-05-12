pub mod config;
pub mod embeddings;
pub mod llm;
#[cfg(feature = "local-embeddings")]
pub mod local_embed;
pub mod memory;
pub mod rag;

// ── Python bindings (feature-gated) ──────────────────────────────────────────
//
// PyO3 is only compiled in when the `python` feature is enabled (via maturin
// for the production wheel). The pure-Rust internals above remain testable
// with `cargo test` without linking libpython.

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use crate::config::AIConfig;
#[cfg(feature = "python")]
use crate::llm::AIEngine;
#[cfg(feature = "python")]
use crate::memory::MemoryManager;
#[cfg(feature = "python")]
use crate::rag::RagPipeline;

/// A Python module implemented in Rust for the UpstreamDrift AI Backend.
#[cfg(feature = "python")]
#[pymodule]
fn ai_backend(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AIConfig>()?;
    m.add_class::<AIEngine>()?;
    m.add_class::<MemoryManager>()?;
    m.add_class::<RagPipeline>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
