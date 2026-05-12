//! PyO3 bindings — numpy-backed batched kernels.
//!
//! Scalar curves and the per-muscle `HillMuscleModel` / `ActivationDynamics`
//! classes are registered directly in `lib.rs`. This module adds the
//! batched / RL-oriented entry points and releases the GIL while their
//! core loops run.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::activation::ActivationDynamics;
use crate::batch;
use crate::model::{HillMuscleModel, MuscleParameters, MuscleState};

fn build_dyn(
    tau_act: Option<f64>,
    tau_deact: Option<f64>,
    min_activation: Option<f64>,
) -> PyResult<ActivationDynamics> {
    let defaults = ActivationDynamics::default();
    ActivationDynamics::new(
        tau_act.unwrap_or(defaults.tau_act),
        tau_deact.unwrap_or(defaults.tau_deact),
        min_activation.unwrap_or(defaults.min_activation),
    )
    .map_err(PyValueError::new_err)
}

fn as_slice_1d<'py, 'a>(
    arr: &'a PyReadonlyArray1<'py, f64>,
    name: &'static str,
) -> PyResult<&'a [f64]> {
    arr.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{name} must be a contiguous 1-D float64 array"))
    })
}

#[pyfunction]
#[pyo3(
    name = "activation_step_batch",
    signature = (u, a, dt, tau_act = None, tau_deact = None, min_activation = None)
)]
fn py_activation_step_batch<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray1<'py, f64>,
    dt: f64,
    tau_act: Option<f64>,
    tau_deact: Option<f64>,
    min_activation: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let dyn_params = build_dyn(tau_act, tau_deact, min_activation)?;
    let u_slice = as_slice_1d(&u, "u")?;
    let a_slice = as_slice_1d(&a, "a")?;
    if u_slice.len() != a_slice.len() {
        return Err(PyValueError::new_err("u and a must have the same length"));
    }
    let m = u_slice.len();
    let dyns = vec![dyn_params; m];
    let mut out = vec![0.0_f64; m];
    py.allow_threads(|| batch::activation_step_batch(u_slice, a_slice, dt, &dyns, &mut out))
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}

/// Compute muscle forces for `M` muscles.
///
/// `activations`, `l_ce`, `v_ce` are all `(M,)` float64.
/// `params` is a `(M, 7)` float64 matrix; columns are
/// `[f_max, l_opt, l_slack, v_max, pennation_angle, damping, force_length_width]`.
/// Returns `(M,)` float64.
#[pyfunction]
#[pyo3(name = "muscle_force_batch", signature = (activations, l_ce, v_ce, params))]
fn py_muscle_force_batch<'py>(
    py: Python<'py>,
    activations: PyReadonlyArray1<'py, f64>,
    l_ce: PyReadonlyArray1<'py, f64>,
    v_ce: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let acts = as_slice_1d(&activations, "activations")?;
    let l = as_slice_1d(&l_ce, "l_ce")?;
    let v = as_slice_1d(&v_ce, "v_ce")?;
    let m = acts.len();
    if l.len() != m || v.len() != m {
        return Err(PyValueError::new_err(
            "activations, l_ce, v_ce must all have the same length",
        ));
    }
    let p_view = params.as_array();
    if p_view.shape() != [m, 7] {
        return Err(PyValueError::new_err(
            "params must have shape (M, 7) matching activations length",
        ));
    }
    let mut models = Vec::with_capacity(m);
    let mut states = Vec::with_capacity(m);
    for i in 0..m {
        let row = p_view.row(i);
        let params_obj = MuscleParameters::new(row[0], row[1], row[2], row[3], row[4], row[5])
            .map_err(|e| PyValueError::new_err(format!("row {i}: {e}")))?;
        let model = HillMuscleModel::new(params_obj, Some(row[6]));
        models.push(model);
        states.push(MuscleState {
            activation: acts[i],
            l_ce: l[i],
            v_ce: v[i],
            l_mt: 0.0,
        });
    }
    let mut out = vec![0.0_f64; m];
    py.allow_threads(|| batch::muscle_force_batch(&models, &states, &mut out))
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}

/// Compute joint torques `tau = R · F`.
#[pyfunction]
#[pyo3(name = "joint_torques_batch", signature = (moment_arms, forces))]
fn py_joint_torques_batch<'py>(
    py: Python<'py>,
    moment_arms: PyReadonlyArray2<'py, f64>,
    forces: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let r_view = moment_arms.as_array();
    let (j, m) = (r_view.shape()[0], r_view.shape()[1]);
    let f = as_slice_1d(&forces, "forces")?;
    if f.len() != m {
        return Err(PyValueError::new_err(
            "forces length must match moment_arms.shape[1]",
        ));
    }
    let r_owned: Vec<f64> = r_view.iter().copied().collect();
    let mut out = vec![0.0_f64; j];
    py.allow_threads(|| batch::joint_torques_batch(&r_owned, j, m, f, &mut out))
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}

/// Single combined RL step.
///
/// Returns `(new_activations: (M,), torques: (J,))`.
#[pyfunction]
#[pyo3(
    name = "step_full",
    signature = (
        excitations, activations, l_ce, v_ce, params, moment_arms, dt,
        tau_act = None, tau_deact = None, min_activation = None
    )
)]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn py_step_full<'py>(
    py: Python<'py>,
    excitations: PyReadonlyArray1<'py, f64>,
    activations: PyReadonlyArray1<'py, f64>,
    l_ce: PyReadonlyArray1<'py, f64>,
    v_ce: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray2<'py, f64>,
    moment_arms: PyReadonlyArray2<'py, f64>,
    dt: f64,
    tau_act: Option<f64>,
    tau_deact: Option<f64>,
    min_activation: Option<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let dyn_params = build_dyn(tau_act, tau_deact, min_activation)?;
    let u = as_slice_1d(&excitations, "excitations")?;
    let a = as_slice_1d(&activations, "activations")?;
    let l = as_slice_1d(&l_ce, "l_ce")?;
    let v = as_slice_1d(&v_ce, "v_ce")?;
    let m = u.len();
    if a.len() != m || l.len() != m || v.len() != m {
        return Err(PyValueError::new_err(
            "excitations, activations, l_ce, v_ce must all have the same length",
        ));
    }
    let p_view = params.as_array();
    if p_view.shape() != [m, 7] {
        return Err(PyValueError::new_err("params must have shape (M, 7)"));
    }
    let r_view = moment_arms.as_array();
    if r_view.shape()[1] != m {
        return Err(PyValueError::new_err(
            "moment_arms.shape[1] must match the number of muscles",
        ));
    }
    let j = r_view.shape()[0];
    let r_owned: Vec<f64> = r_view.iter().copied().collect();
    let mut models = Vec::with_capacity(m);
    let mut states = Vec::with_capacity(m);
    for i in 0..m {
        let row = p_view.row(i);
        let params_obj = MuscleParameters::new(row[0], row[1], row[2], row[3], row[4], row[5])
            .map_err(|e| PyValueError::new_err(format!("row {i}: {e}")))?;
        models.push(HillMuscleModel::new(params_obj, Some(row[6])));
        states.push(MuscleState {
            activation: a[i],
            l_ce: l[i],
            v_ce: v[i],
            l_mt: 0.0,
        });
    }
    let dyns = vec![dyn_params; m];
    let mut a_out = vec![0.0_f64; m];
    let mut tau_out = vec![0.0_f64; j];
    py.allow_threads(|| {
        batch::step_full(
            u,
            a,
            dt,
            &dyns,
            &models,
            &states,
            &r_owned,
            j,
            &mut a_out,
            &mut tau_out,
        )
    })
    .map_err(PyValueError::new_err)?;
    Ok((a_out.into_pyarray(py), tau_out.into_pyarray(py)))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_activation_step_batch, m)?)?;
    m.add_function(wrap_pyfunction!(py_muscle_force_batch, m)?)?;
    m.add_function(wrap_pyfunction!(py_joint_torques_batch, m)?)?;
    m.add_function(wrap_pyfunction!(py_step_full, m)?)?;
    Ok(())
}
