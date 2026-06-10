//! PyO3 bindings — path in, dict of numpy arrays out.
//!
//! The Python facade in `motion_pipeline/sources/{c3d,bvh,trc}_adapter.py`
//! is responsible for converting these arrays into `MarkerTrajectory` /
//! `JointTrajectory` pydantic objects. We deliberately keep the Rust ABI
//! flat so future facades (e.g. Pose Studio) can reuse the same calls
//! without paying the cost of pydantic construction.

use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;

use crate::{bvh, c3d, trc, JointData, MarkerData};

fn marker_data_to_pydict<'py>(py: Python<'py>, data: MarkerData) -> PyResult<Bound<'py, PyDict>> {
    let n_frames = data.n_frames;
    let n_markers = data.n_markers;
    // Reshape flat positions into a 2D (n_frames, n_markers*3) numpy array.
    let arr = ndarray::Array2::from_shape_vec((n_frames, n_markers * 3), data.positions).map_err(
        |e| pyo3::exceptions::PyValueError::new_err(format!("MarkerData shape mismatch: {e}")),
    )?;
    let py_arr = arr.into_pyarray(py);

    let dict = PyDict::new(py);
    dict.set_item("positions", py_arr)?;
    let labels = PyList::new(py, &data.names)?;
    dict.set_item("labels", labels)?;
    dict.set_item("n_frames", n_frames)?;
    dict.set_item("n_markers", n_markers)?;
    dict.set_item("fps", data.fps)?;
    dict.set_item("units", data.units)?;
    let events = PyList::empty(py);
    for event in &data.events {
        let entry = PyDict::new(py);
        entry.set_item("label", &event.label)?;
        entry.set_item("context", &event.context)?;
        entry.set_item("time_s", event.time_s)?;
        events.append(entry)?;
    }
    dict.set_item("events", events)?;
    if let Some(analog) = data.analog {
        let analog_arr = ndarray::Array2::from_shape_vec(
            (
                analog.n_frames * analog.samples_per_frame,
                analog.n_channels,
            ),
            analog.values,
        )
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("AnalogData shape mismatch: {e}"))
        })?;
        let analog_dict = PyDict::new(py);
        analog_dict.set_item("values", analog_arr.into_pyarray(py))?;
        analog_dict.set_item("labels", PyList::new(py, &analog.labels)?)?;
        analog_dict.set_item("units", PyList::new(py, &analog.units)?)?;
        analog_dict.set_item("n_frames", analog.n_frames)?;
        analog_dict.set_item("samples_per_frame", analog.samples_per_frame)?;
        analog_dict.set_item("n_channels", analog.n_channels)?;
        analog_dict.set_item("rate", analog.rate)?;
        dict.set_item("analog", analog_dict)?;
    } else {
        dict.set_item("analog", py.None())?;
    }
    let force_platforms = PyList::empty(py);
    for platform in &data.force_platforms {
        let entry = PyDict::new(py);
        entry.set_item("type", platform.platform_type)?;
        entry.set_item("channels", PyList::new(py, &platform.channels)?)?;
        let corners = PyList::empty(py);
        for corner in &platform.corners {
            corners.append((corner[0], corner[1], corner[2]))?;
        }
        entry.set_item("corners", corners)?;
        entry.set_item(
            "origin",
            (platform.origin[0], platform.origin[1], platform.origin[2]),
        )?;
        force_platforms.append(entry)?;
    }
    dict.set_item("force_platforms", force_platforms)?;
    Ok(dict)
}

#[pyfunction]
fn parse_c3d<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let data = c3d::parse_c3d_file(&path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("C3D parse error: {e}")))?;
    marker_data_to_pydict(py, data)
}

#[pyfunction]
fn parse_trc<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let data = trc::parse_trc_file(&path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("TRC parse error: {e}")))?;
    marker_data_to_pydict(py, data)
}

fn joint_data_to_pydict<'py>(py: Python<'py>, data: JointData) -> PyResult<Bound<'py, PyDict>> {
    let n_frames = data.n_frames;
    let num_dofs = data.num_dofs.max(1);
    let motion_arr =
        ndarray::Array2::from_shape_vec((n_frames, num_dofs), data.motion).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("JointData shape mismatch: {e}"))
        })?;
    let py_motion: Bound<'_, PyArray2<f32>> = motion_arr.into_pyarray(py);

    let dict = PyDict::new(py);
    dict.set_item("motion", py_motion)?;
    dict.set_item("n_frames", n_frames)?;
    dict.set_item("num_dofs", data.num_dofs)?;
    dict.set_item("fps", data.fps)?;

    // Joint hierarchy as a list of (name, parent_idx_or_None, channels).
    let joints_list = PyList::empty(py);
    for j in &data.joints {
        let entry = PyDict::new(py);
        entry.set_item("name", &j.name)?;
        match j.parent {
            Some(p) => entry.set_item("parent", p)?,
            None => entry.set_item("parent", py.None())?,
        }
        let chans = PyList::new(py, &j.channels)?;
        entry.set_item("channels", chans)?;
        joints_list.append(entry)?;
    }
    dict.set_item("joints", joints_list)?;
    Ok(dict)
}

#[pyfunction]
fn parse_bvh<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let data = bvh::parse_bvh_file(&path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("BVH parse error: {e}")))?;
    joint_data_to_pydict(py, data)
}

#[pymodule]
fn upstream_mocap_io(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_c3d, m)?)?;
    m.add_function(wrap_pyfunction!(parse_trc, m)?)?;
    m.add_function(wrap_pyfunction!(parse_bvh, m)?)?;
    Ok(())
}
