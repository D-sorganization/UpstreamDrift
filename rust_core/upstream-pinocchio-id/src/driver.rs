//! Outer-loop orchestration for inverse dynamics.
//!
//! Mirrors the structure of `inverse_dyn_pinocchio.py` lines 204-217: walk
//! the trajectory frame-by-frame, look up the per-frame `(q, v, a)` slice,
//! invoke an `rnea`-shaped callback to produce `tau`, validate finiteness,
//! and store the result.
//!
//! The callback abstraction lets us keep this crate free of any direct
//! Pinocchio dependency: the Python binding wraps `pinocchio.rnea` in a
//! closure; tests can use a closed-form analytical inverse-dynamics rule.

use crate::buffers::DriverBuffers;
use crate::finite_diff::{finite_diff_qddot, finite_diff_qdot};
use ndarray::{Array1, ArrayView1, ArrayView2};

/// Error type produced by [`run_inverse_dynamics`].
#[derive(Debug)]
pub enum DriverError {
    /// `q` rows did not match `times` length.
    ShapeMismatch { q_rows: usize, n_times: usize },
    /// The callback produced a non-finite torque on a given frame.
    NonFiniteTau { frame: usize },
    /// The callback signalled failure on a given frame.
    CallbackFailure { frame: usize, message: String },
}

impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { q_rows, n_times } => write!(
                f,
                "q rows ({q_rows}) does not match times length ({n_times})"
            ),
            Self::NonFiniteTau { frame } => {
                write!(f, "RNEA produced non-finite torques at frame {frame}")
            }
            Self::CallbackFailure { frame, message } => {
                write!(f, "RNEA callback failed at frame {frame}: {message}")
            }
        }
    }
}

impl std::error::Error for DriverError {}

/// Orchestrate the per-frame inverse-dynamics loop.
///
/// - `q`: `(n_frames, n_dof)` joint positions.
/// - `times`: length `n_frames`.
/// - `qdot_override` / `qddot_override`: optional pre-computed velocities and
///   accelerations. When `None`, the finite-difference scheme is used.
/// - `rnea_callback(frame, q_row, qdot_row, qddot_row) -> Result<Array1<f64>>`
///   is invoked per frame to compute `tau`.
///
/// Returns the populated [`DriverBuffers`] (whose `tau` field is the result).
pub fn run_inverse_dynamics<F>(
    q: ArrayView2<'_, f64>,
    times: ArrayView1<'_, f64>,
    qdot_override: Option<ArrayView2<'_, f64>>,
    qddot_override: Option<ArrayView2<'_, f64>>,
    mut rnea_callback: F,
) -> Result<DriverBuffers, DriverError>
where
    F: FnMut(
        usize,
        ArrayView1<'_, f64>,
        ArrayView1<'_, f64>,
        ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String>,
{
    let (n_frames, n_dof) = q.dim();
    if times.len() != n_frames {
        return Err(DriverError::ShapeMismatch {
            q_rows: n_frames,
            n_times: times.len(),
        });
    }
    let mut buf = DriverBuffers::new(n_frames, n_dof);

    if let Some(v) = qdot_override {
        buf.qdot.assign(&v);
    } else {
        buf.qdot = finite_diff_qdot(q, times);
    }
    if let Some(a) = qddot_override {
        buf.qddot.assign(&a);
    } else {
        buf.qddot = finite_diff_qddot(q, times);
    }

    for i in 0..n_frames {
        let q_i = q.row(i);
        let v_i = buf.qdot.row(i);
        let a_i = buf.qddot.row(i);
        let tau_i = rnea_callback(i, q_i, v_i, a_i)
            .map_err(|message| DriverError::CallbackFailure { frame: i, message })?;
        for k in 0..n_dof {
            let t = tau_i[k];
            if !t.is_finite() {
                return Err(DriverError::NonFiniteTau { frame: i });
            }
            buf.tau[(i, k)] = t;
        }
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn analytical_unit_mass_tau_matches_qddot() {
        let times = array![0.0_f64, 0.1, 0.2, 0.3, 0.4];
        let q = array![[0.0_f64], [0.01], [0.04], [0.09], [0.16]];
        let res = run_inverse_dynamics(q.view(), times.view(), None, None, |_, _, _, a| {
            Ok(a.to_owned())
        })
        .unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(res.tau[(i, 0)], 2.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn shape_mismatch_is_an_error() {
        let times = array![0.0_f64, 0.1];
        let q = array![[0.0_f64], [0.01], [0.04]];
        let err = run_inverse_dynamics(q.view(), times.view(), None, None, |_, _, _, _| {
            Ok(Array1::<f64>::zeros(1))
        });
        assert!(matches!(err, Err(DriverError::ShapeMismatch { .. })));
    }

    #[test]
    fn non_finite_tau_is_an_error() {
        let times = array![0.0_f64, 0.1, 0.2];
        let q = array![[0.0_f64], [0.01], [0.04]];
        let err = run_inverse_dynamics(q.view(), times.view(), None, None, |_, _, _, _| {
            Ok(array![f64::NAN])
        });
        assert!(matches!(err, Err(DriverError::NonFiniteTau { frame: 0 })));
    }
}
