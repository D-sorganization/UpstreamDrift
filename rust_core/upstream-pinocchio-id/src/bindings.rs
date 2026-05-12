//! PyO3 bindings — numpy ndarray in / numpy ndarray out, callback for rnea.
//!
//! The Python facade is responsible for type checks and for constructing the
//! `rnea_callback` (typically a `lambda q, v, a: pin.rnea(model, data, q, v, a)`).
//!
//! Frame-by-frame the Rust driver hands the Python callback three 1-D numpy
//! arrays (q, qdot, qddot). The callback must return a 1-D float64 numpy
//! array of length `n_dof`. The Rust side aggregates results into an
//! `(n_frames, n_dof)` numpy ndarray returned to Python.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::driver::run_inverse_dynamics;
use crate::finite_diff::{finite_diff_qddot, finite_diff_qdot};

/// Compute centred finite-difference qdot from `(q, times)`.
#[pyfunction]
#[pyo3(signature = (q, times))]
fn compute_qdot<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let qv = q.as_array();
    let tv = times.as_array();
    let out = finite_diff_qdot(qv, tv);
    Ok(out.into_pyarray(py))
}

/// Compute non-uniform three-point qddot from `(q, times)`.
#[pyfunction]
#[pyo3(signature = (q, times))]
fn compute_qddot<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let qv = q.as_array();
    let tv = times.as_array();
    let out = finite_diff_qddot(qv, tv);
    Ok(out.into_pyarray(py))
}

/// Run the inverse-dynamics outer loop.
#[pyfunction]
#[pyo3(signature = (q, times, rnea_callback, qdot_override=None, qddot_override=None))]
#[allow(clippy::type_complexity)]
fn inverse_dynamics<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    rnea_callback: Bound<'py, PyAny>,
    qdot_override: Option<PyReadonlyArray2<'py, f64>>,
    qddot_override: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let qv = q.as_array();
    let tv = times.as_array();
    let v_over = qdot_override.as_ref().map(|a| a.as_array());
    let a_over = qddot_override.as_ref().map(|a| a.as_array());

    let result = run_inverse_dynamics(qv, tv, v_over, a_over, |_frame, q_row, v_row, a_row| {
        let q_arr = q_row.to_owned().into_pyarray(py);
        let v_arr = v_row.to_owned().into_pyarray(py);
        let a_arr = a_row.to_owned().into_pyarray(py);
        let res = rnea_callback
            .call1((q_arr, v_arr, a_arr))
            .map_err(|e| format!("{e}"))?;
        let arr: Bound<'py, PyArray1<f64>> = res
            .extract()
            .map_err(|e| format!("rnea must return 1-D float64 ndarray: {e}"))?;
        // SAFETY: we copy out immediately and do not retain `arr`.
        let owned = unsafe { arr.as_array().to_owned() };
        Ok(owned)
    });

    match result {
        Ok(buf) => Ok((
            buf.qdot.into_pyarray(py),
            buf.qddot.into_pyarray(py),
            buf.tau.into_pyarray(py),
        )),
        Err(crate::driver::DriverError::ShapeMismatch { q_rows, n_times }) => {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "q rows ({q_rows}) does not match times length ({n_times})"
            )))
        }
        Err(crate::driver::DriverError::NonFiniteTau { frame }) => {
            Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "RNEA produced non-finite torques at frame {frame}"
            )))
        }
        Err(crate::driver::DriverError::CallbackFailure { frame, message }) => {
            Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "RNEA callback failed at frame {frame}: {message}"
            )))
        }
        Err(crate::driver::DriverError::QdotOverrideShapeMismatch {
            expected_rows,
            expected_cols,
            actual_rows,
            actual_cols,
        }) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "qdot_override shape ({actual_rows}, {actual_cols}) does not match expected ({expected_rows}, {expected_cols})"
        ))),
        Err(crate::driver::DriverError::QddotOverrideShapeMismatch {
            expected_rows,
            expected_cols,
            actual_rows,
            actual_cols,
        }) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "qddot_override shape ({actual_rows}, {actual_cols}) does not match expected ({expected_rows}, {expected_cols})"
        ))),
    }
}

#[pymodule]
fn upstream_pinocchio_id(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_qdot, m)?)?;
    m.add_function(wrap_pyfunction!(compute_qddot, m)?)?;
    m.add_function(wrap_pyfunction!(inverse_dynamics, m)?)?;
    Ok(())
}
