//! PyO3 bindings.
//!
//! The wire format between Rust and Python is JSON — `serde_json` handles
//! the [`Robot`] AST in both directions, and the Python facade decodes/
//! encodes via stdlib `json`. This keeps the binding surface tiny and
//! avoids hand-coding a `FromPyObject` for every AST node.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::ast::Robot;
use crate::mjcf_ast::MujocoDocument;

#[pyfunction]
fn parse_urdf(xml: &str) -> PyResult<String> {
    let robot = crate::parser::urdf::parse_urdf_str(xml)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&robot).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn write_urdf(robot_json: &str) -> PyResult<String> {
    let robot: Robot =
        serde_json::from_str(robot_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    crate::writer::urdf::write_urdf(&robot).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn parse_mjcf(xml: &str) -> PyResult<String> {
    let doc = crate::parser::mjcf::parse_mjcf_str(xml)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&doc).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn write_mjcf(doc_json: &str) -> PyResult<String> {
    let doc: MujocoDocument =
        serde_json::from_str(doc_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    crate::writer::mjcf::write_mjcf(&doc).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// `version` lets the Python facade detect ABI/feature breaks and fall back
/// to pure Python without surprising callers.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn upstream_urdf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_urdf, m)?)?;
    m.add_function(wrap_pyfunction!(write_urdf, m)?)?;
    m.add_function(wrap_pyfunction!(parse_mjcf, m)?)?;
    m.add_function(wrap_pyfunction!(write_mjcf, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
