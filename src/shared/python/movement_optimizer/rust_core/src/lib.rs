//! Rust-accelerated hot-path functions for the Movement Optimizer.
//!
//! Provides vectorised inverse dynamics and COM computation using
//! pre-computed coupling coefficients.  Uses rayon for parallel
//! iteration over timesteps when N is large.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_conversion)]

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Parallel threshold: use rayon only if N exceeds this.
const PAR_THRESHOLD: usize = 128;

/// Compute inverse dynamics for all timesteps in batch.
///
/// Parameters are pre-computed scalar constants from LagrangianDynamics:
///   M00, M11, M22  -- diagonal mass-matrix entries
///   a01, a02, a12  -- coupling coefficients
///   g0, g1, g2     -- gravity coefficients
///
/// q, qd, qdd: (N, 3) arrays
/// Returns: torques (N, 3)
#[pyfunction]
fn inverse_dynamics_batch_rs<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f64>,
    qd: PyReadonlyArray2<'py, f64>,
    qdd: PyReadonlyArray2<'py, f64>,
    m00: f64,
    m11: f64,
    m22: f64,
    a01: f64,
    a02: f64,
    a12: f64,
    g0: f64,
    g1: f64,
    g2: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let q = q.as_array();
    let qd = qd.as_array();
    let qdd = qdd.as_array();

    // Validate input shapes: all must be (N, 3)
    if q.shape()[1] != 3 || qd.shape()[1] != 3 || qdd.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "q, qd, qdd must each have exactly 3 columns",
        ));
    }
    let n = q.shape()[0];
    if qd.shape()[0] != n || qdd.shape()[0] != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "q, qd, qdd must have the same number of rows",
        ));
    }

    let mut tau = Array2::<f64>::zeros((n, 3));

    let compute_row = |i: usize| -> [f64; 3] {
        let q0 = q[[i, 0]];
        let q1 = q[[i, 1]];
        let q2 = q[[i, 2]];
        let qd0 = qd[[i, 0]];
        let qd1 = qd[[i, 1]];
        let qd2 = qd[[i, 2]];
        let qdd0 = qdd[[i, 0]];
        let qdd1 = qdd[[i, 1]];
        let qdd2 = qdd[[i, 2]];

        let d01 = q0 - q1;
        let d02 = q0 - q2;
        let d12 = q1 - q2;

        let c01 = d01.cos();
        let c02 = d02.cos();
        let c12 = d12.cos();
        let s01 = d01.sin();
        let s02 = d02.sin();
        let s12 = d12.sin();

        // M(q) * qdd
        let t0 = m00 * qdd0 + a01 * c01 * qdd1 + a02 * c02 * qdd2;
        let t1 = a01 * c01 * qdd0 + m11 * qdd1 + a12 * c12 * qdd2;
        let t2 = a02 * c02 * qdd0 + a12 * c12 * qdd1 + m22 * qdd2;

        // + Coriolis
        let t0 = t0 + a01 * s01 * qd1 * qd1 + a02 * s02 * qd2 * qd2;
        let t1 = t1 - a01 * s01 * qd0 * qd0 + a12 * s12 * qd2 * qd2;
        let t2 = t2 - a02 * s02 * qd0 * qd0 - a12 * s12 * qd1 * qd1;

        // + Gravity
        let t0 = t0 + g0 * q0.sin();
        let t1 = t1 + g1 * q1.sin();
        let t2 = t2 + g2 * q2.sin();

        [t0, t1, t2]
    };

    if n >= PAR_THRESHOLD {
        let rows: Vec<[f64; 3]> = (0..n).into_par_iter().map(compute_row).collect();
        for (i, row) in rows.iter().enumerate() {
            tau[[i, 0]] = row[0];
            tau[[i, 1]] = row[1];
            tau[[i, 2]] = row[2];
        }
    } else {
        for i in 0..n {
            let row = compute_row(i);
            tau[[i, 0]] = row[0];
            tau[[i, 1]] = row[1];
            tau[[i, 2]] = row[2];
        }
    }

    Ok(tau.into_pyarray_bound(py))
}

/// Compute COM x-coordinate for all timesteps.
///
/// Parameters:
///   q: (N, 3) joint angles
///   l0, l1, l2: segment lengths
///   d0, d1, d2: COM distances from proximal joint
///   m0, m1, m2: segment masses
///   m_feet, foot_com_x: foot mass and COM x
///   bar_mass, body_mass: load and total body mass
///   is_squat: true for squat/full_squat, false for deadlift
///   m_arms: arm mass (used for deadlift)
///
/// Returns: com_x (N,)
#[pyfunction]
fn com_x_batch_rs<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f64>,
    l0: f64,
    l1: f64,
    l2: f64,
    d0: f64,
    d1: f64,
    d2: f64,
    m0: f64,
    m1: f64,
    m2: f64,
    m_feet: f64,
    foot_com_x: f64,
    bar_mass: f64,
    body_mass: f64,
    is_squat: bool,
    m_arms: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let q = q.as_array();

    // Validate input shape: q must be (N, 3)
    if q.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "q must have exactly 3 columns",
        ));
    }
    let n = q.shape()[0];
    let total_mass = body_mass + bar_mass;

    let compute_one = |i: usize| -> f64 {
        let sq0 = q[[i, 0]].sin();
        let sq1 = q[[i, 1]].sin();
        let sq2 = q[[i, 2]].sin();

        let knee_x = l0 * sq0;
        let hip_x = knee_x + l1 * sq1;
        let shoulder_x = hip_x + l2 * sq2;

        let c1x = d0 * sq0;
        let c2x = knee_x + d1 * sq1;
        let c3x = hip_x + d2 * sq2;

        let mut num = m_feet * foot_com_x + m0 * c1x + m1 * c2x + m2 * c3x;

        if is_squat {
            num += bar_mass * shoulder_x;
        } else {
            num += (m_arms + bar_mass) * shoulder_x;
        }

        num / total_mass
    };

    let result: Vec<f64> = if n >= PAR_THRESHOLD {
        (0..n).into_par_iter().map(compute_one).collect()
    } else {
        (0..n).map(compute_one).collect()
    };

    Ok(Array1::from_vec(result).into_pyarray_bound(py))
}

/// Generic N-DOF inverse dynamics for all timesteps in batch.
///
/// For an N-DOF serial chain with diagonal mass matrix approximation:
///   tau_i = mass_matrix_diag[i] * qdd_i + gravity_coeffs[i] * sin(q_i)
///
/// This is a simplified but generic version that ignores off-diagonal
/// coupling terms (Coriolis/centripetal).  For the full 3-DOF version
/// with coupling, use `inverse_dynamics_batch_rs`.
///
/// Parameters:
///   q, qd, qdd: (N, n_dof) arrays
///   mass_matrix_diag: (n_dof,) diagonal mass-matrix entries
///   gravity_coeffs: (n_dof,) gravity coefficients
///
/// Returns: torques (N, n_dof)
#[pyfunction]
fn inverse_dynamics_ndof_rs<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f64>,
    _qd: PyReadonlyArray2<'py, f64>,
    qdd: PyReadonlyArray2<'py, f64>,
    mass_matrix_diag: PyReadonlyArray1<'py, f64>,
    gravity_coeffs: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let q = q.as_array();
    let qdd = qdd.as_array();
    let m_diag = mass_matrix_diag.as_array();
    let g_coeffs = gravity_coeffs.as_array();

    let n = q.shape()[0];
    let n_dof = q.shape()[1];

    if m_diag.len() != n_dof || g_coeffs.len() != n_dof {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mass_matrix_diag and gravity_coeffs must have length n_dof",
        ));
    }

    let compute_row = |i: usize| -> Vec<f64> {
        let mut row = Vec::with_capacity(n_dof);
        for j in 0..n_dof {
            let t = m_diag[j] * qdd[[i, j]] + g_coeffs[j] * q[[i, j]].sin();
            row.push(t);
        }
        row
    };

    let mut tau = Array2::<f64>::zeros((n, n_dof));

    if n >= PAR_THRESHOLD {
        let rows: Vec<Vec<f64>> = (0..n).into_par_iter().map(compute_row).collect();
        for (i, row) in rows.iter().enumerate() {
            for j in 0..n_dof {
                tau[[i, j]] = row[j];
            }
        }
    } else {
        for i in 0..n {
            let row = compute_row(i);
            for j in 0..n_dof {
                tau[[i, j]] = row[j];
            }
        }
    }

    Ok(tau.into_pyarray_bound(py))
}

#[pymodule]
fn movement_optimizer_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(inverse_dynamics_batch_rs, m)?)?;
    m.add_function(wrap_pyfunction!(com_x_batch_rs, m)?)?;
    m.add_function(wrap_pyfunction!(inverse_dynamics_ndof_rs, m)?)?;
    Ok(())
}
