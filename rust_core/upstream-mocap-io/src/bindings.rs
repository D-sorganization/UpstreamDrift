//! PyO3 bindings — path in, dict of numpy arrays out.
//!
//! The Python facade in `motion_pipeline/sources/{c3d,bvh,trc}_adapter.py`
//! is responsible for converting these arrays into `MarkerTrajectory` /
//! `JointTrajectory` pydantic objects. We deliberately keep the Rust ABI
//! flat so future facades (e.g. Pose Studio) can reuse the same calls
//! without paying the cost of pydantic construction.

use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::{bvh, c3d, trc, JointData, MarkerData, ParseError};

#[cfg(unix)]
fn path_contains_nul(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;

    path.as_os_str().as_bytes().contains(&0)
}

#[cfg(windows)]
fn path_contains_nul(path: &Path) -> bool {
    use std::os::windows::ffi::OsStrExt;

    path.as_os_str().encode_wide().any(|unit| unit == 0)
}

#[cfg(not(any(unix, windows)))]
fn path_contains_nul(path: &Path) -> bool {
    path.as_os_str().to_string_lossy().contains('\0')
}

fn validate_input_path(format: &str, path: &Path) -> PyResult<()> {
    if path.as_os_str().is_empty() {
        return Err(PyValueError::new_err(format!(
            "{format} path must be non-empty"
        )));
    }
    if path_contains_nul(path) {
        return Err(PyValueError::new_err(format!(
            "{format} path contains an interior NUL byte: {}",
            path.display()
        )));
    }
    Ok(())
}

fn parse_error_to_pyerr(format: &str, path: &Path, err: ParseError) -> PyErr {
    match err {
        ParseError::Io(io_err) if io_err.kind() == ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(format!(
                "{format} file not found: {} ({io_err})",
                path.display()
            ))
        }
        ParseError::Io(io_err) => PyOSError::new_err(format!(
            "{format} file access error for {}: {io_err}",
            path.display()
        )),
        ParseError::Format(message) => PyValueError::new_err(format!(
            "{format} parse error in {}: {message}",
            path.display()
        )),
    }
}

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
    validate_input_path("C3D", &path)?;
    let data = c3d::parse_c3d_file(&path).map_err(|e| parse_error_to_pyerr("C3D", &path, e))?;
    marker_data_to_pydict(py, data)
}

#[pyfunction]
fn parse_trc<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    validate_input_path("TRC", &path)?;
    let data = trc::parse_trc_file(&path).map_err(|e| parse_error_to_pyerr("TRC", &path, e))?;
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
    validate_input_path("BVH", &path)?;
    let data = bvh::parse_bvh_file(&path).map_err(|e| parse_error_to_pyerr("BVH", &path, e))?;
    joint_data_to_pydict(py, data)
}

#[pymodule]
fn upstream_mocap_io(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_c3d, m)?)?;
    m.add_function(wrap_pyfunction!(parse_trc, m)?)?;
    m.add_function(wrap_pyfunction!(parse_bvh, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn with_python(test: impl FnOnce(Python<'_>)) {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(test);
    }

    fn unique_temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "upstream_mocap_io_bindings_{name}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    fn assert_missing_file_error(format: &str, err: PyErr, py: Python<'_>, path: &Path) {
        assert!(
            err.is_instance_of::<PyFileNotFoundError>(py),
            "{format} missing file should raise FileNotFoundError, got {err:?}"
        );
        let message = err.to_string();
        assert!(message.contains(format), "{message}");
        assert!(
            message.contains(&path.display().to_string()),
            "missing path not named in error: {message}"
        );
    }

    fn assert_malformed_file_error(format: &str, err: PyErr, py: Python<'_>, path: &Path) {
        assert!(
            err.is_instance_of::<PyValueError>(py),
            "{format} malformed file should raise ValueError, got {err:?}"
        );
        let message = err.to_string();
        assert!(message.contains(format), "{message}");
        assert!(message.contains("parse error"), "{message}");
        assert!(
            message.contains(&path.display().to_string()),
            "malformed path not named in error: {message}"
        );
    }

    #[test]
    fn parse_bindings_reject_empty_paths_before_file_access() {
        with_python(|py| {
            let err = parse_trc(py, PathBuf::new()).expect_err("empty path must fail");
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("non-empty"));
        });
    }

    #[test]
    fn parse_bindings_map_missing_files_to_file_not_found() {
        with_python(|py| {
            let c3d_path = unique_temp_path("missing.c3d");
            let trc_path = unique_temp_path("missing.trc");
            let bvh_path = unique_temp_path("missing.bvh");

            let err = parse_c3d(py, c3d_path.clone()).expect_err("missing c3d must fail");
            assert_missing_file_error("C3D", err, py, &c3d_path);

            let err = parse_trc(py, trc_path.clone()).expect_err("missing trc must fail");
            assert_missing_file_error("TRC", err, py, &trc_path);

            let err = parse_bvh(py, bvh_path.clone()).expect_err("missing bvh must fail");
            assert_missing_file_error("BVH", err, py, &bvh_path);
        });
    }

    #[test]
    fn parse_bindings_map_malformed_present_files_to_value_error() {
        with_python(|py| {
            let c3d_path = unique_temp_path("malformed.c3d");
            let trc_path = unique_temp_path("malformed.trc");
            let bvh_path = unique_temp_path("malformed.bvh");

            fs::write(&c3d_path, b"not a c3d").expect("write malformed c3d");
            fs::write(&trc_path, "PathFileType\t4\t(X/Y/Z)\tbad.trc\n")
                .expect("write malformed trc");
            fs::write(&bvh_path, "HIERARCHY\nROOT Hips\n").expect("write malformed bvh");

            let err = parse_c3d(py, c3d_path.clone()).expect_err("malformed c3d must fail");
            assert_malformed_file_error("C3D", err, py, &c3d_path);

            let err = parse_trc(py, trc_path.clone()).expect_err("malformed trc must fail");
            assert_malformed_file_error("TRC", err, py, &trc_path);

            let err = parse_bvh(py, bvh_path.clone()).expect_err("malformed bvh must fail");
            assert_malformed_file_error("BVH", err, py, &bvh_path);

            let _ = fs::remove_file(c3d_path);
            let _ = fs::remove_file(trc_path);
            let _ = fs::remove_file(bvh_path);
        });
    }
}
