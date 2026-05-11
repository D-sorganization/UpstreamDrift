pub mod config;
pub mod llm;
pub mod memory;
pub mod rag;

use pyo3::prelude::*;
use crate::config::AIConfig;
use crate::llm::AIEngine;
use crate::memory::MemoryManager;
use crate::rag::RagPipeline;

/// A Python module implemented in Rust for the UpstreamDrift AI Backend.
#[pymodule]
fn ai_backend(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AIConfig>()?;
    m.add_class::<AIEngine>()?;
    m.add_class::<MemoryManager>()?;
    m.add_class::<RagPipeline>()?;
    Ok(())
}
