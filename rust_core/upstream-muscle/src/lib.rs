//! # upstream-muscle — UpstreamDrift Hill muscle kernels
//!
//! High-performance Rust implementation of the Hill-type muscle model used
//! by UpstreamDrift's biomechanics stack. Designed to be called from RL
//! training loops (`stable_baselines3`) where releasing the GIL during
//! batched evaluation is the win.
//!
//! ## Implemented surface (UD#5216 complete)
//!
//! Scalar primitives, ported from
//! `src/shared/python/biomechanics/{hill_muscle,activation_dynamics,multi_muscle}.py`:
//!
//! - [`hill::f_l`] — active force-length (Gaussian)
//! - [`hill::f_p`] — passive force-length (PEE exponential spring)
//! - [`hill::f_v`] — force-velocity (Hill hyperbola + eccentric plateau)
//! - [`hill::f_t`] — tendon force (SEE quadratic)
//! - [`model::HillMuscleModel`] — state-bearing force assembly (CE + PEE
//!   + damping projected through pennation)
//! - [`activation::ActivationDynamics`] — first-order excitation →
//!   activation Euler step
//! - [`muscle_equilibrium`] — equilibrium solver (PR #5247)
//! - [`multi_muscle`] — multi-muscle moment summation, HashMap-attachment
//!   API (PR #5247)
//! - [`multi::joint_torques`] — dense `τ = R · F` for RL hot-loops
//! - [`batch`] — Rayon-parallel batched RL inner-loop kernels
//!
//! ## Python facade
//!
//! Building with `--features python` (via `maturin`) compiles the PyO3
//! module `upstream_muscle`. It exposes:
//!
//! - Scalar wrappers around the curves.
//! - `ActivationDynamics`, `MuscleParameters`, `MuscleState`,
//!   `HillMuscleModel` as Python classes (from PR #5246).
//! - `MuscleAttachment`, `MuscleGroup`, `AntagonistPair`,
//!   `PyEquilibriumSolver` (from PR #5247).
//! - Numpy-backed batched APIs (`activation_step_batch`,
//!   `muscle_force_batch`, `joint_torques_batch`, `step_full`).
//!
//! Batched entry points release the GIL via `py.allow_threads`.
//!
//! Numerical parity vs the Python source is asserted within 1e-6 in
//! `tests/parity_*.rs`. Pure-Python fallback lives in
//! `src/shared/python/biomechanics/rust_muscle.py`.

pub mod activation;
pub mod batch;
pub mod hill;
pub mod model;
pub mod multi;
pub mod multi_muscle;
pub mod muscle_equilibrium;

// Convenience re-exports.
pub use activation::ActivationDynamics;
pub use hill::{f_l, f_l_with_width, f_p, f_t, f_v};
pub use model::{HillMuscleModel, MuscleParameters, MuscleState};
pub use multi::{joint_torque, joint_torques};

// ── Python bindings (feature-gated) ──────────────────────────────────────────

#[cfg(feature = "python")]
mod python_api;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Active force-length curve. Equivalent to
/// `HillMuscleModel.force_length_active` in the Python source.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "f_l", signature = (l_norm, width = None))]
fn py_f_l(py: Python<'_>, l_norm: f64, width: Option<f64>) -> f64 {
    py.allow_threads(|| match width {
        Some(w) => hill::f_l_with_width(l_norm, w),
        None => hill::f_l(l_norm),
    })
}

/// Passive (PEE) force-length curve. Equivalent to
/// `HillMuscleModel.force_length_passive`.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "f_p")]
fn py_f_p(py: Python<'_>, l_norm: f64) -> f64 {
    py.allow_threads(|| hill::f_p(l_norm))
}

/// Force-velocity curve. Equivalent to `HillMuscleModel.force_velocity`.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "f_v")]
fn py_f_v(py: Python<'_>, v_norm: f64) -> f64 {
    py.allow_threads(|| hill::f_v(v_norm))
}

/// Tendon (SEE) force-length curve. Equivalent to
/// `HillMuscleModel.tendon_force`.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "f_t")]
fn py_f_t(py: Python<'_>, l_tendon_norm: f64) -> f64 {
    py.allow_threads(|| hill::f_t(l_tendon_norm))
}

/// `upstream_muscle` Python module.
#[cfg(feature = "python")]
#[pymodule]
fn upstream_muscle(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_f_l, m)?)?;
    m.add_function(wrap_pyfunction!(py_f_p, m)?)?;
    m.add_function(wrap_pyfunction!(py_f_v, m)?)?;
    m.add_function(wrap_pyfunction!(py_f_t, m)?)?;
    m.add_class::<activation::ActivationDynamics>()?;
    m.add_class::<model::HillMuscleModel>()?;
    m.add_class::<model::MuscleParameters>()?;
    m.add_class::<model::MuscleState>()?;

    m.add_class::<muscle_equilibrium::PyEquilibriumSolver>()?;
    m.add_function(wrap_pyfunction!(
        muscle_equilibrium::compute_equilibrium_state,
        m
    )?)?;

    m.add_class::<multi_muscle::MuscleAttachment>()?;
    m.add_class::<multi_muscle::MuscleGroup>()?;
    m.add_class::<multi_muscle::AntagonistPair>()?;

    m.add(
        "DEFAULT_FORCE_LENGTH_WIDTH",
        hill::DEFAULT_FORCE_LENGTH_WIDTH,
    )?;
    m.add("DEFAULT_TAU_ACT", activation::DEFAULT_TAU_ACT)?;
    m.add("DEFAULT_TAU_DEACT", activation::DEFAULT_TAU_DEACT)?;
    m.add("DEFAULT_MIN_ACTIVATION", activation::DEFAULT_MIN_ACTIVATION)?;
    python_api::register(m)?;
    Ok(())
}
