//! PyO3 bindings — numpy ndarray in / numpy ndarray out.
//!
//! The Python facade is responsible for type checks. The Rust side accepts
//! float64 contiguous `(n_frames, n_points, n_dims)` arrays via `numpy`'s
//! `PyReadonlyArray3` (with a copy via `.as_array()`).

use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;

use crate::{filter, gap_fill, resample};

#[pyfunction]
#[pyo3(signature = (data, cutoff_hz, order, fps))]
fn butterworth_filter<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f64>,
    cutoff_hz: f64,
    order: usize,
    fps: f64,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let view = data.as_array();
    let out = filter::butterworth_filter(view, cutoff_hz, order, fps);
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (data, window_length, polyorder))]
fn savgol_filter<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f64>,
    window_length: usize,
    polyorder: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let view = data.as_array();
    let out = filter::savgol_filter(view, window_length, polyorder);
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (data, kernel_size))]
fn median_filter<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f64>,
    kernel_size: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let view = data.as_array();
    let out = filter::median_filter(view, kernel_size);
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (data, sigma))]
fn gaussian_filter<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f64>,
    sigma: f64,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let view = data.as_array();
    let out = filter::gaussian_filter(view, sigma);
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (data, process_noise, measurement_noise))]
fn kalman_filter<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f64>,
    process_noise: f64,
    measurement_noise: f64,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let view = data.as_array();
    let out = filter::kalman_filter(view, process_noise, measurement_noise);
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (data, mask, max_gap))]
fn linear_gap_fill<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    max_gap: usize,
) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray2<bool>>)> {
    let view = data.as_array();
    let m = mask.as_array();
    let (out, out_mask) = gap_fill::linear_gap_fill(view, m, max_gap);
    Ok((out.into_pyarray(py), out_mask.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (data, mask, max_gap))]
fn cubic_gap_fill<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    max_gap: usize,
) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray2<bool>>)> {
    let view = data.as_array();
    let m = mask.as_array();
    let (out, out_mask) = gap_fill::cubic_gap_fill(view, m, max_gap);
    Ok((out.into_pyarray(py), out_mask.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (data, mask, max_gap, rank=None))]
fn pca_gap_fill<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    max_gap: usize,
    rank: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, numpy::PyArray1<bool>>,
)> {
    let view = data.as_array();
    let m = mask.as_array();
    let (out, out_mask, success) = gap_fill::pca_gap_fill(view, m, max_gap, rank);
    Ok((
        out.into_pyarray(py),
        out_mask.into_pyarray(py),
        success.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (data, source_timestamps, target_timestamps))]
fn resample_fps<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f64>,
    source_timestamps: PyReadonlyArray1<'py, f64>,
    target_timestamps: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let view = data.as_array();
    let src = source_timestamps.as_slice()?.to_vec();
    let tgt = target_timestamps.as_slice()?.to_vec();
    let out = resample::resample_fps(view, &src, &tgt);
    Ok(out.into_pyarray(py))
}

#[pymodule]
fn upstream_mocap_preproc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(butterworth_filter, m)?)?;
    m.add_function(wrap_pyfunction!(savgol_filter, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_filter, m)?)?;
    m.add_function(wrap_pyfunction!(kalman_filter, m)?)?;
    m.add_function(wrap_pyfunction!(linear_gap_fill, m)?)?;
    m.add_function(wrap_pyfunction!(cubic_gap_fill, m)?)?;
    m.add_function(wrap_pyfunction!(pca_gap_fill, m)?)?;
    m.add_function(wrap_pyfunction!(resample_fps, m)?)?;
    Ok(())
}
