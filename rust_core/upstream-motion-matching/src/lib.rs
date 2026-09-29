//! # upstream-motion-matching — Motion-matching outer-loop kernels
//!
//! First slice of issue #5218: a Rust port of the per-frame finite-
//! difference kernel that today lives in
//! `src/shared/python/motion_pipeline/matching/inverse_dyn_pinocchio.py`
//! (`PinocchioInverseDynMatchingSolver._finite_difference`, lines 70-111).
//!
//! ## Modules
//!
//! - [`finite_diff`] — Uniform-`dt` central-difference qdot / qddot for
//!   `(N_frames, N_dof)` joint-position trajectories.
//!
//! ## Roadmap
//!
//! Slice 2 (tracked in the follow-up issue cross-linked from #5218):
//! per-frame `pin.rnea` driver in Rust, calling back into Pinocchio's
//! C++ via released-GIL Python bindings (amortizes 1 GIL crossing per
//! trajectory instead of N per frame). Slice 3: CMC + MuJoCo torque
//! variants and the full N=1000-frame end-to-end benchmark. Eventual
//! acceptance: ≥3× end-to-end speedup vs the current Python driver and
//! numerical parity within the Pinocchio binding's tolerance.

pub mod finite_diff;

pub use finite_diff::{finite_diff_uniform, FiniteDiffError, FiniteDiffResult};

// ── Python bindings (feature-gated) ──────────────────────────────────────────

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-facing entry point for
/// `upstream_motion_matching.finite_diff_q_to_qdot_qddot`.
///
/// We intentionally take `Vec<Vec<f64>>` rather than a `numpy.ndarray`
/// for this first slice — it avoids the `numpy` Rust crate dep (and its
/// nalgebra feature-flag matrix) until the slice-2 `pin.rnea` driver
/// actually needs zero-copy buffers. `upstream-mesh` made the same
/// choice (see its `lib.rs` rationale comment).
///
/// Returns `(qdot, qddot)` as a tuple of nested float lists with the
/// same shape as the input. The GIL is released for the duration of
/// the numerical work.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "finite_diff_q_to_qdot_qddot")]
#[allow(clippy::type_complexity)]
fn finite_diff_q_to_qdot_qddot_py(
    py: Python<'_>,
    q: Vec<Vec<f64>>,
    dt: f64,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    py.allow_threads(|| {
        finite_diff::finite_diff_uniform(&q, dt)
            .map(|r| (r.qdot, r.qddot))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    })
}

#[cfg(feature = "python")]
#[pymodule]
fn upstream_motion_matching(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(finite_diff_q_to_qdot_qddot_py, m)?)?;
    Ok(())
}
