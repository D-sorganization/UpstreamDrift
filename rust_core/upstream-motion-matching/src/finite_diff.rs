//! Finite-difference qdot / qddot kernel.
//!
//! Pure-Rust port of the inner loop in
//! `src/shared/python/motion_pipeline/matching/inverse_dyn_pinocchio.py`
//! lines 85-110 (`PinocchioInverseDynMatchingSolver._finite_difference`).
//!
//! Given a `(N_frames, N_dof)` joint-position trajectory `q` sampled at a
//! uniform `dt`, compute the corresponding `qdot` and `qddot` arrays of
//! the same shape using:
//!
//! * **Interior frames** — central differences (second-order accurate).
//! * **Boundary frames** — one-sided differences for `qdot`; copy the
//!   nearest interior `qddot` (matches the Python reference, which lacks
//!   end-point second-derivative samples).
//!
//! The Python implementation supports non-uniform timestamps. For the
//! first slice we restrict to uniform `dt` because:
//!
//! 1. Mocap pipelines in this repo resample to uniform sample rate
//!    upstream (see `motion_pipeline/preprocessing/`).
//! 2. It collapses the qddot formula from the general
//!    `2*(q[i+1]*dt_b - q[i]*(dt_b+dt_f) + q[i-1]*dt_f) /
//!    (dt_b*dt_f*(dt_b+dt_f))` to `(q[i+1] - 2*q[i] + q[i-1]) / dt^2`,
//!    which is what we need for the matching-pipeline benchmark and for
//!    parity vs the Python driver when fed uniform-time trajectories.
//!
//! Slice 2 (tracked in the follow-up issue cross-linked from #5218) will
//! lift this restriction once the PyO3 callback architecture for the
//! per-frame `pin.rnea` driver lands.

use serde::{Deserialize, Serialize};

/// Errors returned by [`finite_diff_uniform`].
#[derive(Debug, PartialEq, Clone)]
pub enum FiniteDiffError {
    /// Trajectory had zero frames. Need at least one row.
    EmptyTrajectory,
    /// Trajectory had fewer than two frames — boundary differences are
    /// undefined, callers should special-case `N=1` themselves.
    SingleFrame,
    /// `dt` was non-positive or non-finite.
    InvalidDt(f64),
    /// One of the `q` rows had a different number of columns than the
    /// first row.
    RaggedRows {
        expected: usize,
        got: usize,
        row: usize,
    },
    /// A `q` value was non-finite (NaN or infinity).
    NonFiniteInput { row: usize, col: usize, value: f64 },
}

impl core::fmt::Display for FiniteDiffError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyTrajectory => write!(f, "trajectory must have at least one frame"),
            Self::SingleFrame => write!(
                f,
                "finite-difference requires at least 2 frames (got 1); \
                 caller must handle single-frame case"
            ),
            Self::InvalidDt(dt) => {
                write!(f, "dt must be finite and positive, got {dt}")
            }
            Self::RaggedRows { expected, got, row } => write!(
                f,
                "q row {row} has {got} columns, expected {expected} \
                 (matching first row)"
            ),
            Self::NonFiniteInput { row, col, value } => {
                write!(f, "q[{row}][{col}] is non-finite ({value})")
            }
        }
    }
}

impl std::error::Error for FiniteDiffError {}

/// Result of [`finite_diff_uniform`]: `(qdot, qddot)` row-major arrays
/// matching the input `q` shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiniteDiffResult {
    /// Joint velocities, shape `(N_frames, N_dof)`.
    pub qdot: Vec<Vec<f64>>,
    /// Joint accelerations, shape `(N_frames, N_dof)`.
    pub qddot: Vec<Vec<f64>>,
}

/// Compute `qdot` and `qddot` from a uniformly sampled `q` trajectory.
///
/// # Arguments
/// * `q` — Row-major `(N_frames, N_dof)` joint positions.
/// * `dt` — Uniform sample interval in seconds. Must be finite and `> 0`.
///
/// # Numerical scheme (matches Python at uniform `dt`)
///
/// Let `N = q.len()` and `D = q[0].len()`.
///
/// * `qdot[0]    = (q[1]   - q[0])    / dt`
/// * `qdot[N-1]  = (q[N-1] - q[N-2])  / dt`
/// * `qdot[i]    = (q[i+1] - q[i-1])  / (2*dt)` for `1 <= i <= N-2`
/// * `qddot[i]   = (q[i+1] - 2*q[i] + q[i-1]) / dt^2` for `1 <= i <= N-2`
/// * `qddot[0]   = qddot[1]`     (only when `N >= 3`, else zero)
/// * `qddot[N-1] = qddot[N-2]`   (same condition)
///
/// For `N == 2` `qddot` is all zeros; this matches the Python branch
/// that only copies `qddot[1] -> qddot[0]` when `len(times) >= 3`.
///
/// # Errors
/// See [`FiniteDiffError`] — empty input, ragged rows, non-finite values
/// or non-positive `dt`.
pub fn finite_diff_uniform(q: &[Vec<f64>], dt: f64) -> Result<FiniteDiffResult, FiniteDiffError> {
    if q.is_empty() {
        return Err(FiniteDiffError::EmptyTrajectory);
    }
    let n = q.len();
    if n < 2 {
        return Err(FiniteDiffError::SingleFrame);
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(FiniteDiffError::InvalidDt(dt));
    }
    let d = q[0].len();
    for (row_idx, row) in q.iter().enumerate() {
        if row.len() != d {
            return Err(FiniteDiffError::RaggedRows {
                expected: d,
                got: row.len(),
                row: row_idx,
            });
        }
        for (col_idx, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(FiniteDiffError::NonFiniteInput {
                    row: row_idx,
                    col: col_idx,
                    value: v,
                });
            }
        }
    }

    let mut qdot = vec![vec![0.0_f64; d]; n];
    let mut qddot = vec![vec![0.0_f64; d]; n];

    let inv_2dt = 0.5 / dt;
    let inv_dt = 1.0 / dt;
    let inv_dt2 = 1.0 / (dt * dt);

    // Interior central differences.
    for i in 1..n - 1 {
        let qm = &q[i - 1];
        let qc = &q[i];
        let qp = &q[i + 1];
        let qd = &mut qdot[i];
        let qdd = &mut qddot[i];
        for j in 0..d {
            qd[j] = (qp[j] - qm[j]) * inv_2dt;
            qdd[j] = (qp[j] - 2.0 * qc[j] + qm[j]) * inv_dt2;
        }
    }

    // Boundary qdot: one-sided.
    {
        let q0 = &q[0];
        let q1 = &q[1];
        let qd0 = &mut qdot[0];
        for j in 0..d {
            qd0[j] = (q1[j] - q0[j]) * inv_dt;
        }
    }
    {
        let qn1 = &q[n - 1];
        let qn2 = &q[n - 2];
        let qd_last = &mut qdot[n - 1];
        for j in 0..d {
            qd_last[j] = (qn1[j] - qn2[j]) * inv_dt;
        }
    }

    // Boundary qddot: copy the nearest interior sample. Matches the
    // Python reference's `if len(times) >= 3` branch — for N == 2 the
    // boundary qddot stay zero.
    if n >= 3 {
        qddot[0] = qddot[1].clone();
        qddot[n - 1] = qddot[n - 2].clone();
    }

    Ok(FiniteDiffResult { qdot, qddot })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn linear_trajectory_has_constant_qdot_and_zero_qddot() {
        // q[i] = [i * 0.1, i * 0.2] sampled at dt=0.01; qdot must be
        // exactly [10, 20] everywhere, qddot exactly zero.
        let dt = 0.01_f64;
        let q: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64 * 0.1, i as f64 * 0.2])
            .collect();
        let r = finite_diff_uniform(&q, dt).unwrap();
        for row in &r.qdot {
            assert_relative_eq!(row[0], 10.0, max_relative = 1e-12);
            assert_relative_eq!(row[1], 20.0, max_relative = 1e-12);
        }
        for row in &r.qddot {
            assert_relative_eq!(row[0], 0.0, epsilon = 1e-9);
            assert_relative_eq!(row[1], 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn quadratic_trajectory_has_constant_qddot() {
        // q[i] = 0.5 * a * (i*dt)^2 with a = 3.0 → qddot must be a in
        // every interior frame. Boundary qddot is copied from interior,
        // so it equals a too.
        let dt = 0.05_f64;
        let a = 3.0_f64;
        let q: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let t = i as f64 * dt;
                vec![0.5 * a * t * t]
            })
            .collect();
        let r = finite_diff_uniform(&q, dt).unwrap();
        for (i, row) in r.qddot.iter().enumerate() {
            assert_relative_eq!(row[0], a, max_relative = 1e-10, epsilon = 1e-10);
            // qdot at interior i should be a * t (central difference is
            // exact for quadratic data).
            if i >= 1 && i <= r.qdot.len() - 2 {
                let t = i as f64 * dt;
                assert_relative_eq!(r.qdot[i][0], a * t, max_relative = 1e-10);
            }
        }
    }

    #[test]
    fn empty_trajectory_errors() {
        let q: Vec<Vec<f64>> = vec![];
        assert_eq!(
            finite_diff_uniform(&q, 0.01).unwrap_err(),
            FiniteDiffError::EmptyTrajectory
        );
    }

    #[test]
    fn single_frame_errors() {
        let q = vec![vec![1.0, 2.0]];
        assert_eq!(
            finite_diff_uniform(&q, 0.01).unwrap_err(),
            FiniteDiffError::SingleFrame
        );
    }

    #[test]
    fn invalid_dt_errors() {
        let q = vec![vec![0.0], vec![1.0]];
        assert!(matches!(
            finite_diff_uniform(&q, 0.0).unwrap_err(),
            FiniteDiffError::InvalidDt(_)
        ));
        assert!(matches!(
            finite_diff_uniform(&q, -0.01).unwrap_err(),
            FiniteDiffError::InvalidDt(_)
        ));
        assert!(matches!(
            finite_diff_uniform(&q, f64::NAN).unwrap_err(),
            FiniteDiffError::InvalidDt(_)
        ));
    }

    #[test]
    fn ragged_rows_error() {
        let q = vec![vec![0.0, 0.0], vec![1.0]];
        assert_eq!(
            finite_diff_uniform(&q, 0.01).unwrap_err(),
            FiniteDiffError::RaggedRows {
                expected: 2,
                got: 1,
                row: 1
            }
        );
    }

    #[test]
    fn non_finite_input_errors() {
        let q = vec![vec![0.0, f64::NAN], vec![1.0, 2.0]];
        assert!(matches!(
            finite_diff_uniform(&q, 0.01).unwrap_err(),
            FiniteDiffError::NonFiniteInput { row: 0, col: 1, .. }
        ));
    }

    #[test]
    fn two_frames_qddot_is_zero() {
        // Special case from the Python reference: with N=2 the boundary
        // qddot stay zero (the `if len(times) >= 3` branch is skipped).
        let q = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
        let r = finite_diff_uniform(&q, 0.1).unwrap();
        assert_relative_eq!(r.qdot[0][0], 10.0);
        assert_relative_eq!(r.qdot[0][1], 20.0);
        assert_relative_eq!(r.qdot[1][0], 10.0);
        assert_relative_eq!(r.qdot[1][1], 20.0);
        assert_eq!(r.qddot[0], vec![0.0, 0.0]);
        assert_eq!(r.qddot[1], vec![0.0, 0.0]);
    }
}
