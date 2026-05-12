//! Convex hull kernel.
//!
//! Wraps `parry3d::transformation::convex_hull`, which returns a
//! `(Vec<Point<Real>>, Vec<[u32; 3]>)` pair representing the hull's
//! vertices and triangle indices. We expose a small owning result type
//! that's cheap to move into Python and has stable serde shape for
//! disk-cache scenarios in the eventual mesh_processor port.

use nalgebra::Point3;
use serde::{Deserialize, Serialize};

/// Result of a convex hull computation: deduplicated hull vertices plus
/// triangle indices into that vertex array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvexHullResult {
    /// Hull vertices in `(x, y, z)` order. Length equals the number of
    /// extreme points kept by `parry3d`.
    pub vertices: Vec<[f32; 3]>,
    /// Triangle index triples into `vertices`. Each triangle is wound
    /// outward (right-hand rule), matching parry's convention.
    pub indices: Vec<[u32; 3]>,
}

impl ConvexHullResult {
    /// Number of triangles in the hull.
    #[inline]
    pub fn num_triangles(&self) -> usize {
        self.indices.len()
    }

    /// Number of vertices in the hull.
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Signed volume of the hull, computed by summing per-triangle
    /// tetrahedra against the origin.
    ///
    /// For a convex hull with outward-facing triangles this is positive
    /// and equal to the enclosed volume — matches `scipy.spatial.ConvexHull.volume`
    /// and `trimesh.Trimesh.volume` to within float tolerance.
    pub fn volume(&self) -> f64 {
        let mut acc = 0.0_f64;
        for &[i, j, k] in &self.indices {
            let a = self.vertices[i as usize];
            let b = self.vertices[j as usize];
            let c = self.vertices[k as usize];
            // (a · (b × c)) / 6
            let cross_x = (b[1] as f64) * (c[2] as f64) - (b[2] as f64) * (c[1] as f64);
            let cross_y = (b[2] as f64) * (c[0] as f64) - (b[0] as f64) * (c[2] as f64);
            let cross_z = (b[0] as f64) * (c[1] as f64) - (b[1] as f64) * (c[0] as f64);
            acc += (a[0] as f64) * cross_x + (a[1] as f64) * cross_y + (a[2] as f64) * cross_z;
        }
        (acc / 6.0).abs()
    }
}

/// Errors returned by [`compute_convex_hull`].
#[derive(Debug)]
pub enum ConvexHullError {
    /// Fewer than four input points were supplied. A 3-D convex hull
    /// requires at least a tetrahedron's worth of points.
    TooFewPoints(usize),
    /// An input vertex contained a non-finite coordinate.
    NonFiniteInput { index: usize },
}

impl core::fmt::Display for ConvexHullError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooFewPoints(n) => {
                write!(f, "convex hull requires at least 4 input points (got {n})")
            }
            Self::NonFiniteInput { index } => {
                write!(f, "input vertex {index} contained non-finite coordinate")
            }
        }
    }
}

impl std::error::Error for ConvexHullError {}

/// Compute the 3-D convex hull of `vertices`.
///
/// # Preconditions
/// - `vertices.len() >= 4`
/// - Every coordinate is finite (`is_finite`).
///
/// # Returns
/// A [`ConvexHullResult`] whose `vertices` is a deduplicated set of hull
/// extreme points and whose `indices` are outward-facing triangles.
///
/// # Notes
/// Coplanar / colinear inputs may cause `parry3d` to return a hull with
/// one fewer vertex than scipy's QHull — callers comparing against scipy
/// should allow ±1 vertex slack on degenerate seeds.
pub fn compute_convex_hull(vertices: &[[f32; 3]]) -> Result<ConvexHullResult, ConvexHullError> {
    if vertices.len() < 4 {
        return Err(ConvexHullError::TooFewPoints(vertices.len()));
    }
    for (idx, v) in vertices.iter().enumerate() {
        if !v[0].is_finite() || !v[1].is_finite() || !v[2].is_finite() {
            return Err(ConvexHullError::NonFiniteInput { index: idx });
        }
    }

    let points: Vec<Point3<f32>> = vertices
        .iter()
        .map(|&[x, y, z]| Point3::new(x, y, z))
        .collect();

    let (hull_pts, hull_idx) = parry3d::transformation::convex_hull(&points);

    let out_vertices: Vec<[f32; 3]> = hull_pts.into_iter().map(|p| [p.x, p.y, p.z]).collect();
    let out_indices: Vec<[u32; 3]> = hull_idx;

    Ok(ConvexHullResult {
        vertices: out_vertices,
        indices: out_indices,
    })
}

// ── Python wrapper class ─────────────────────────────────────────────────────

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-facing wrapper around [`ConvexHullResult`].
///
/// Exposes the `vertices` and `indices` lists plus convenience accessors
/// matching the surface of `trimesh.Trimesh` we replace in
/// `_cg_convex_hull.py` (`.vertices`, `.faces`, `.volume`).
#[cfg(feature = "python")]
#[pyclass(name = "ConvexHullResult", module = "upstream_mesh", frozen)]
pub struct PyConvexHullResult {
    inner: ConvexHullResult,
}

#[cfg(feature = "python")]
impl From<ConvexHullResult> for PyConvexHullResult {
    fn from(inner: ConvexHullResult) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyConvexHullResult {
    /// Hull vertices as a list of `(x, y, z)` tuples.
    #[getter]
    fn vertices(&self) -> Vec<(f32, f32, f32)> {
        self.inner
            .vertices
            .iter()
            .map(|v| (v[0], v[1], v[2]))
            .collect()
    }

    /// Triangle indices as a list of `(i, j, k)` tuples — alias `faces`
    /// matches `trimesh.Trimesh.faces` for drop-in usage.
    #[getter]
    fn indices(&self) -> Vec<(u32, u32, u32)> {
        self.inner
            .indices
            .iter()
            .map(|t| (t[0], t[1], t[2]))
            .collect()
    }

    /// Alias for [`PyConvexHullResult::indices`] matching trimesh naming.
    #[getter]
    fn faces(&self) -> Vec<(u32, u32, u32)> {
        self.indices()
    }

    /// Number of triangles.
    #[getter]
    fn num_triangles(&self) -> usize {
        self.inner.num_triangles()
    }

    /// Number of vertices.
    #[getter]
    fn num_vertices(&self) -> usize {
        self.inner.num_vertices()
    }

    /// Enclosed volume of the hull (matches `scipy.spatial.ConvexHull.volume`).
    #[getter]
    fn volume(&self) -> f64 {
        self.inner.volume()
    }

    fn __repr__(&self) -> String {
        format!(
            "ConvexHullResult(vertices={}, triangles={})",
            self.inner.num_vertices(),
            self.inner.num_triangles()
        )
    }
}
