//! Finite-difference qdot/qddot estimators matching the Python reference
//! in `motion_pipeline/matching/inverse_dyn_pinocchio.py` lines 85-110.
//!
//! Inputs are `(n_frames, n_dof)` `q` arrays and a length-`n_frames` `times`
//! array. The scheme is **non-uniform-dt aware**:
//!
//! - Interior `qdot[i]` uses centred difference over `times[i+1] - times[i-1]`.
//! - First/last `qdot` use one-sided forward/backward differences.
//! - Interior `qddot[i]` uses the non-uniform three-point second derivative:
//!   `qddot = 2*(q[i+1]*dt_b - q[i]*(dt_b+dt_f) + q[i-1]*dt_f) /
//!   (dt_b*dt_f*(dt_b+dt_f))` where `dt_b = t[i]-t[i-1]`, `dt_f = t[i+1]-t[i]`.
//! - Endpoints copy the nearest interior `qddot` value.
//!
//! Edge cases (zero `dt`, fewer than 2 frames) zero out the corresponding
//! row, just like the Python reference.
//!
//! ## Relationship to `upstream-motion-matching::finite_diff` (issue #7660)
//!
//! `upstream-motion-matching` carries a sibling finite-difference kernel. The
//! two are *not* interchangeable: that one assumes a **uniform** `dt` and
//! works on `Vec<Vec<f64>>`, whereas this one is **non-uniform-`dt` aware**
//! and works on `ndarray` views (the shape the Pinocchio driver and PyO3
//! numpy bindings already use). Forcing either to adopt the other's API would
//! be a large, behaviour-changing refactor. What the audit actually flagged
//! was the *inconsistent error contract* — this crate panicked via
//! `assert_eq!` while motion-matching returned a `Result`. That inconsistency
//! is now resolved: both report precondition violations through a `Result`
//! with a `FiniteDiffError` enum. Full extraction of a shared
//! `kinematics-core` crate that unifies the uniform and non-uniform schemes
//! behind one API is deferred (tracked under the finite-difference cluster
//! #7556).
//!
//! Contract for short trajectories (issue #7146): these kernels return
//! all-zero derivatives for `n_frames < 2` (qdot) / `n_frames < 3` (qddot).
//! That makes inverse dynamics degenerate to statics. Callers must enforce the
//! minimum-frame precondition *before* reaching here — the Python entry points
//! (`pose_interchange` reference adapter and `motion_pipeline` solver) do this
//! via `engine_core.finite_difference.require_enough_frames_for_finite_diff`,
//! raising `ValueError` unless explicit qdot/qddot overrides are supplied. The
//! zero-fill is retained only as a defined-but-guarded fallback so the kernels
//! never panic on a degenerate row.

use ndarray::{Array2, ArrayView1, ArrayView2};

/// Error returned by the finite-difference kernels when their shape
/// precondition is violated.
///
/// Standardising on a `Result` contract (rather than `assert_eq!`) keeps the
/// error path consistent with `upstream-motion-matching`'s
/// `FiniteDiffError` and avoids a `panic!` propagating across the PyO3
/// boundary as a `BaseException` (issue #7660, related #7147).
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum FiniteDiffError {
    /// `times.len()` did not equal `q`'s row count (`n_frames`).
    TimesLengthMismatch { n_frames: usize, n_times: usize },
}

impl core::fmt::Display for FiniteDiffError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TimesLengthMismatch { n_frames, n_times } => write!(
                f,
                "times length ({n_times}) must equal q rows / n_frames ({n_frames})"
            ),
        }
    }
}

impl std::error::Error for FiniteDiffError {}

/// Compute centred finite-difference `qdot` matching the Python reference.
///
/// Shape: `q` is `(n_frames, n_dof)`, `times` length `n_frames`; output is
/// `(n_frames, n_dof)`.
///
/// # Errors
/// Returns [`FiniteDiffError::TimesLengthMismatch`] when `times.len()` does
/// not equal `q`'s row count.
pub fn finite_diff_qdot(
    q: ArrayView2<'_, f64>,
    times: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, FiniteDiffError> {
    let (n_frames, n_dof) = q.dim();
    if times.len() != n_frames {
        return Err(FiniteDiffError::TimesLengthMismatch {
            n_frames,
            n_times: times.len(),
        });
    }
    let mut qdot = Array2::<f64>::zeros((n_frames, n_dof));
    if n_frames < 2 {
        return Ok(qdot);
    }
    // Interior centred difference.
    for i in 1..n_frames.saturating_sub(1) {
        let dt = times[i + 1] - times[i - 1];
        if dt > 0.0 {
            for k in 0..n_dof {
                qdot[(i, k)] = (q[(i + 1, k)] - q[(i - 1, k)]) / dt;
            }
        }
    }
    // Endpoints: forward / backward with a 1e-9 floor on dt (matches Python).
    let dt_first = (times[1] - times[0]).max(1e-9);
    let dt_last = (times[n_frames - 1] - times[n_frames - 2]).max(1e-9);
    for k in 0..n_dof {
        qdot[(0, k)] = (q[(1, k)] - q[(0, k)]) / dt_first;
        qdot[(n_frames - 1, k)] = (q[(n_frames - 1, k)] - q[(n_frames - 2, k)]) / dt_last;
    }
    Ok(qdot)
}

/// Compute non-uniform three-point `qddot` matching the Python reference.
///
/// # Errors
/// Returns [`FiniteDiffError::TimesLengthMismatch`] when `times.len()` does
/// not equal `q`'s row count.
pub fn finite_diff_qddot(
    q: ArrayView2<'_, f64>,
    times: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, FiniteDiffError> {
    let (n_frames, n_dof) = q.dim();
    if times.len() != n_frames {
        return Err(FiniteDiffError::TimesLengthMismatch {
            n_frames,
            n_times: times.len(),
        });
    }
    let mut qddot = Array2::<f64>::zeros((n_frames, n_dof));
    if n_frames < 3 {
        return Ok(qddot);
    }
    for i in 1..n_frames - 1 {
        let dt_b = times[i] - times[i - 1];
        let dt_f = times[i + 1] - times[i];
        if dt_b > 0.0 && dt_f > 0.0 {
            let denom = dt_b * dt_f * (dt_b + dt_f);
            for k in 0..n_dof {
                qddot[(i, k)] = 2.0
                    * (q[(i + 1, k)] * dt_b - q[(i, k)] * (dt_b + dt_f) + q[(i - 1, k)] * dt_f)
                    / denom;
            }
        }
    }
    // Endpoints copy nearest interior row.
    for k in 0..n_dof {
        qddot[(0, k)] = qddot[(1, k)];
        qddot[(n_frames - 1, k)] = qddot[(n_frames - 2, k)];
    }
    Ok(qddot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn qdot_constant_motion_is_constant_velocity() {
        let times = array![0.0_f64, 0.1, 0.2, 0.3, 0.4];
        let q = array![[0.0_f64], [0.1], [0.2], [0.3], [0.4]];
        let v = finite_diff_qdot(q.view(), times.view()).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(v[(i, 0)], 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn qddot_quadratic_motion_is_constant_accel() {
        let times = array![0.0_f64, 0.1, 0.2, 0.3, 0.4];
        let q = array![[0.0_f64], [0.01], [0.04], [0.09], [0.16]];
        let a = finite_diff_qddot(q.view(), times.view()).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(a[(i, 0)], 2.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn handles_too_few_frames() {
        let times = array![0.0_f64];
        let q = array![[1.0_f64, 2.0]];
        let v = finite_diff_qdot(q.view(), times.view()).unwrap();
        let a = finite_diff_qddot(q.view(), times.view()).unwrap();
        assert_eq!(v.dim(), (1, 2));
        assert_eq!(a.dim(), (1, 2));
        assert_abs_diff_eq!(v[(0, 0)], 0.0);
        assert_abs_diff_eq!(a[(0, 0)], 0.0);
    }

    #[test]
    fn times_length_mismatch_is_a_result_error_not_a_panic() {
        // Previously this path used `assert_eq!`, panicking. The contract is
        // now a typed `Result` error (issue #7660).
        let times = array![0.0_f64, 0.1];
        let q = array![[0.0_f64], [0.01], [0.04]];
        assert_eq!(
            finite_diff_qdot(q.view(), times.view()).unwrap_err(),
            FiniteDiffError::TimesLengthMismatch {
                n_frames: 3,
                n_times: 2
            }
        );
        assert_eq!(
            finite_diff_qddot(q.view(), times.view()).unwrap_err(),
            FiniteDiffError::TimesLengthMismatch {
                n_frames: 3,
                n_times: 2
            }
        );
    }
}
