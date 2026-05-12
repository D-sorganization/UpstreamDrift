//! # upstream-mesh — UpstreamDrift Mesh Kernels
//!
//! First slice of issue #5219: a `parry3d`-backed convex hull kernel,
//! shipping behind an optional PyO3 binding so we can incrementally
//! retire the `trimesh` Python dependency in
//! `humanoid_character_builder/mesh/_cg_*.py`.
//!
//! ## Modules
//!
//! - `convex_hull`: Deterministic convex hull (vertices + triangle indices)
//!   computed via `parry3d::transformation::convex_hull`.
//!
//! ## Roadmap
//!
//! Slice 2 (tracked in the follow-up issue cross-linked from #5219):
//! quadric edge-collapse decimation with a streaming/iterator API to
//! prevent the OOM lineage of closed issue #3903; primitive-fitting
//! (sphere / box / cylinder / capsule); mesh metrics and inertia.

pub mod convex_hull;

pub use convex_hull::{compute_convex_hull, ConvexHullError, ConvexHullResult};

// ── Python bindings (feature-gated) ──────────────────────────────────────────

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-facing entry point for `upstream_mesh.compute_convex_hull`.
///
/// Accepts a list of `(x, y, z)` tuples (or any object that converts to
/// `Vec<(f32, f32, f32)>` via PyO3). Returns a Python object exposing
/// `vertices: list[tuple[float, float, float]]` and
/// `indices: list[tuple[int, int, int]]`.
///
/// We intentionally take `Vec<(f32, f32, f32)>` rather than a `numpy.ndarray`
/// for this first slice — it avoids the `numpy` Rust crate dep (and its
/// nalgebra feature-flag matrix) until decimation actually needs it. Slice 2
/// will introduce `numpy` once we benchmark zero-copy paths for million-vertex
/// meshes.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "compute_convex_hull")]
fn compute_convex_hull_py(
    vertices: Vec<(f32, f32, f32)>,
) -> PyResult<convex_hull::PyConvexHullResult> {
    let pts: Vec<[f32; 3]> = vertices.into_iter().map(|(x, y, z)| [x, y, z]).collect();
    convex_hull::compute_convex_hull(&pts)
        .map(convex_hull::PyConvexHullResult::from)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[cfg(feature = "python")]
#[pymodule]
fn upstream_mesh(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<convex_hull::PyConvexHullResult>()?;
    m.add_function(pyo3::wrap_pyfunction!(compute_convex_hull_py, m)?)?;
    Ok(())
}
