//! PyO3 bindings.
//!
//! Slice 1 kept `Vec<(f32, f32, f32)>` to avoid the `numpy` crate dep.
//! Slice 2 (#5248) introduces it because the headline acceptance criterion
//! — 1M-triangle peak-RSS comparison — requires zero-copy `numpy.ndarray`
//! ingestion. The convex-hull entry point retains the tuple-list form for
//! back-compat with the existing python test in
//! `tests/unit/mesh/test_rust_convex_hull.py`.

#![allow(clippy::type_complexity)]

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::convex_hull::{compute_convex_hull, PyConvexHullResult};
use crate::decimation::{decimate_vhacd, DecimateParameters};
use crate::primitives::{fit_aabb, fit_capsule, fit_cylinder, fit_obb, fit_sphere};

// ── Convex hull (back-compat tuple-list form from slice 1) ───────────────────

#[pyfunction]
#[pyo3(name = "compute_convex_hull")]
fn compute_convex_hull_py(vertices: Vec<(f32, f32, f32)>) -> PyResult<PyConvexHullResult> {
    let pts: Vec<[f32; 3]> = vertices.into_iter().map(|(x, y, z)| [x, y, z]).collect();
    compute_convex_hull(&pts)
        .map(PyConvexHullResult::from)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

// ── Convex hull (zero-copy numpy form for million-vertex meshes) ─────────────

/// Numpy zero-copy variant of `compute_convex_hull`.
///
/// Returns `(hull_vertices: (Nh,3) float32, hull_indices: (Mh,3) uint32)`.
#[pyfunction]
#[pyo3(name = "compute_convex_hull_np")]
fn compute_convex_hull_np<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f32>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u32>>)> {
    let view = vertices.as_array();
    if view.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vertices must have shape (N, 3)",
        ));
    }
    let pts: Vec<[f32; 3]> = view
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();

    let hull = compute_convex_hull(&pts)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let n_v = hull.vertices.len();
    let mut flat_v = Vec::with_capacity(n_v * 3);
    for v in &hull.vertices {
        flat_v.extend_from_slice(v);
    }
    let arr_v = ndarray::Array2::from_shape_vec((n_v, 3), flat_v)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let n_i = hull.indices.len();
    let mut flat_i = Vec::with_capacity(n_i * 3);
    for t in &hull.indices {
        flat_i.extend_from_slice(t);
    }
    let arr_i = ndarray::Array2::from_shape_vec((n_i, 3), flat_i)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((arr_v.into_pyarray(py), arr_i.into_pyarray(py)))
}

// ── VHACD decimation ─────────────────────────────────────────────────────────

/// VHACD-based mesh decomposition / decimation.
///
/// Inputs:
/// - `vertices`: `(N, 3)` float32 numpy array.
/// - `indices`:  `(M, 3)` uint32 numpy array.
/// - `resolution`, `max_hulls`, `concavity`, `fill_mode`: VHACD tuning.
///
/// Returns: `list[(vertices: (Nh,3) float32, indices: (Mh,3) uint32)]` — one
/// entry per convex part.
#[pyfunction]
#[pyo3(
    name = "decimate_vhacd",
    signature = (vertices, indices, resolution=64, max_hulls=16, concavity=0.001, fill_mode=0)
)]
#[allow(clippy::type_complexity)]
fn decimate_vhacd_py<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f32>,
    indices: PyReadonlyArray2<'py, u32>,
    resolution: u32,
    max_hulls: usize,
    concavity: f32,
    fill_mode: u8,
) -> PyResult<Vec<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u32>>)>> {
    let v_view = vertices.as_array();
    let i_view = indices.as_array();
    if v_view.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vertices must have shape (N, 3)",
        ));
    }
    if i_view.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "indices must have shape (M, 3)",
        ));
    }
    let pts: Vec<[f32; 3]> = v_view
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    let tris: Vec<[u32; 3]> = i_view
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();

    let params = DecimateParameters {
        resolution,
        max_hulls,
        concavity,
        fill_mode,
    };
    let out = decimate_vhacd(&pts, &tris, &params)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let mut parts_py = Vec::with_capacity(out.parts.len());
    for hull in out.parts {
        let n_v = hull.vertices.len();
        let mut flat_v = Vec::with_capacity(n_v * 3);
        for v in &hull.vertices {
            flat_v.extend_from_slice(v);
        }
        let arr_v = ndarray::Array2::from_shape_vec((n_v, 3), flat_v)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let n_i = hull.indices.len();
        let mut flat_i = Vec::with_capacity(n_i * 3);
        for t in &hull.indices {
            flat_i.extend_from_slice(t);
        }
        let arr_i = ndarray::Array2::from_shape_vec((n_i, 3), flat_i)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        parts_py.push((arr_v.into_pyarray(py), arr_i.into_pyarray(py)));
    }
    Ok(parts_py)
}

// ── Primitive fitting ────────────────────────────────────────────────────────

/// `(center, extents, volume_ratio)` for AABB.
type AabbTuple = ([f32; 3], [f32; 3], f32);
/// `(center, extents, rotation_xyzw, volume_ratio)` for OBB.
type ObbTuple = ([f32; 3], [f32; 3], [f32; 4], f32);
/// `(center, radius, volume_ratio)` for sphere.
type SphereTuple = ([f32; 3], f32, f32);
/// `(center, radius, height, rotation_xyzw, volume_ratio)` for cylinder/capsule.
type CylinderTuple = ([f32; 3], f32, f32, [f32; 4], f32);

fn to_pts(vertices: PyReadonlyArray2<'_, f32>) -> PyResult<Vec<[f32; 3]>> {
    let view = vertices.as_array();
    if view.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vertices must have shape (N, 3)",
        ));
    }
    Ok(view
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect())
}

/// `(center: (3,), extents: (3,), volume_ratio: float)`.
#[pyfunction]
#[pyo3(name = "fit_aabb", signature = (vertices, mesh_volume=None))]
fn fit_aabb_py(
    vertices: PyReadonlyArray2<'_, f32>,
    mesh_volume: Option<f32>,
) -> PyResult<AabbTuple> {
    let pts = to_pts(vertices)?;
    let f = fit_aabb(&pts, mesh_volume)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((f.center, f.extents, f.volume_ratio))
}

/// `(center: (3,), extents: (3,), rotation_xyzw: (4,), volume_ratio: float)`.
#[pyfunction]
#[pyo3(name = "fit_obb", signature = (vertices, mesh_volume=None))]
fn fit_obb_py(vertices: PyReadonlyArray2<'_, f32>, mesh_volume: Option<f32>) -> PyResult<ObbTuple> {
    let pts = to_pts(vertices)?;
    let f = fit_obb(&pts, mesh_volume)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((f.center, f.extents, f.rotation, f.volume_ratio))
}

/// `(center: (3,), radius: float, volume_ratio: float)`.
#[pyfunction]
#[pyo3(name = "fit_sphere", signature = (vertices, mesh_volume=None))]
fn fit_sphere_py(
    vertices: PyReadonlyArray2<'_, f32>,
    mesh_volume: Option<f32>,
) -> PyResult<SphereTuple> {
    let pts = to_pts(vertices)?;
    let f = fit_sphere(&pts, mesh_volume)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((f.center, f.radius, f.volume_ratio))
}

/// `(center: (3,), radius: float, height: float, rotation_xyzw: (4,), volume_ratio: float)`.
#[pyfunction]
#[pyo3(name = "fit_cylinder", signature = (vertices, mesh_volume=None))]
fn fit_cylinder_py(
    vertices: PyReadonlyArray2<'_, f32>,
    mesh_volume: Option<f32>,
) -> PyResult<CylinderTuple> {
    let pts = to_pts(vertices)?;
    let f = fit_cylinder(&pts, mesh_volume)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((f.center, f.radius, f.height, f.rotation, f.volume_ratio))
}

/// `(center: (3,), radius: float, height: float, rotation_xyzw: (4,), volume_ratio: float)`.
#[pyfunction]
#[pyo3(name = "fit_capsule", signature = (vertices, mesh_volume=None))]
fn fit_capsule_py(
    vertices: PyReadonlyArray2<'_, f32>,
    mesh_volume: Option<f32>,
) -> PyResult<CylinderTuple> {
    let pts = to_pts(vertices)?;
    let f = fit_capsule(&pts, mesh_volume)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((f.center, f.radius, f.height, f.rotation, f.volume_ratio))
}

// ── Module init ──────────────────────────────────────────────────────────────

#[pymodule]
fn upstream_mesh(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConvexHullResult>()?;
    m.add_function(wrap_pyfunction!(compute_convex_hull_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_convex_hull_np, m)?)?;
    m.add_function(wrap_pyfunction!(decimate_vhacd_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_aabb_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_obb_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_sphere_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_cylinder_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_capsule_py, m)?)?;
    Ok(())
}
