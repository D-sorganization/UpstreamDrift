//! # upstream-muscle — UpstreamDrift Hill muscle kernels
//!
//! High-performance Rust implementation of the Hill-type muscle model used
//! by UpstreamDrift's biomechanics stack. Designed to be called from RL
//! training loops (`stable_baselines3`) where releasing the GIL during
//! batched evaluation is the win.
//!
//! ## Implemented slices
//!
//! Pure scalar contractile/passive/tendon curves ported from
//! `src/shared/python/biomechanics/hill_muscle.py`:
//!
//! - [`hill::f_l`] — active force-length (Gaussian)
//! - [`hill::f_p`] — passive force-length (PEE exponential spring)
//! - [`hill::f_v`] — force-velocity (Hill hyperbola + eccentric plateau)
//! - [`hill::f_t`] — tendon force (SEE quadratic)
//! - [`model::HillMuscleModel`] — state-bearing force assembly
//! - [`activation::ActivationDynamics`] — first-order activation dynamics
//!
//! Numerical parity vs the Python source is asserted within 1e-6 in
//! `tests/parity_hill.rs`.
//!
//! ## Out of scope (future slices, tracked separately)
//!
//! - Full muscle equilibrium solver (`muscle_equilibrium.py`)
//! - Multi-muscle moment summation (`multi_muscle.py`)
//! - Batched / `rayon`-parallel API for RL inner loops
//! - OpenSim/MuJoCo parity test corpus
//! - Python facade replacement
//!
//! See the umbrella issue UD#5216 and the slice-1 follow-up for status.

pub mod activation;
pub mod hill;
pub mod model;

// Convenience re-exports so callers can `use upstream_muscle::{f_l, f_v, f_t};`.
pub use activation::ActivationDynamics;
pub use hill::{f_l, f_l_with_width, f_p, f_t, f_v};
pub use model::{HillMuscleModel, MuscleParameters, MuscleState};

// ── Python bindings (feature-gated) ──────────────────────────────────────────
//
// PyO3 is only compiled when the `python` feature is enabled (via maturin
// for the production wheel). The pure-Rust internals above remain testable
// with `cargo test` without linking libpython, mirroring the
// upstream-physics / ai_backend pattern.

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
///
/// ```python
/// from upstream_muscle import f_l, f_v, f_t
/// f = f_l(0.95)
/// ```
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
    m.add(
        "DEFAULT_FORCE_LENGTH_WIDTH",
        hill::DEFAULT_FORCE_LENGTH_WIDTH,
    )?;
    Ok(())
}
