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

use ndarray::{Array2, ArrayView1, ArrayView2};

/// Compute centred finite-difference `qdot` matching the Python reference.
///
/// Shape: `q` is `(n_frames, n_dof)`, `times` length `n_frames`; output is
/// `(n_frames, n_dof)`.
pub fn finite_diff_qdot(q: ArrayView2<'_, f64>, times: ArrayView1<'_, f64>) -> Array2<f64> {
    let (n_frames, n_dof) = q.dim();
    assert_eq!(
        times.len(),
        n_frames,
        "times length must equal q rows (n_frames)"
    );
    let mut qdot = Array2::<f64>::zeros((n_frames, n_dof));
    if n_frames < 2 {
        return qdot;
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
    qdot
}

/// Compute non-uniform three-point `qddot` matching the Python reference.
pub fn finite_diff_qddot(q: ArrayView2<'_, f64>, times: ArrayView1<'_, f64>) -> Array2<f64> {
    let (n_frames, n_dof) = q.dim();
    assert_eq!(
        times.len(),
        n_frames,
        "times length must equal q rows (n_frames)"
    );
    let mut qddot = Array2::<f64>::zeros((n_frames, n_dof));
    if n_frames < 3 {
        return qddot;
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
    qddot
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
        let v = finite_diff_qdot(q.view(), times.view());
        for i in 0..5 {
            assert_abs_diff_eq!(v[(i, 0)], 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn qddot_quadratic_motion_is_constant_accel() {
        let times = array![0.0_f64, 0.1, 0.2, 0.3, 0.4];
        let q = array![[0.0_f64], [0.01], [0.04], [0.09], [0.16]];
        let a = finite_diff_qddot(q.view(), times.view());
        for i in 0..5 {
            assert_abs_diff_eq!(a[(i, 0)], 2.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn handles_too_few_frames() {
        let times = array![0.0_f64];
        let q = array![[1.0_f64, 2.0]];
        let v = finite_diff_qdot(q.view(), times.view());
        let a = finite_diff_qddot(q.view(), times.view());
        assert_eq!(v.dim(), (1, 2));
        assert_eq!(a.dim(), (1, 2));
        assert_abs_diff_eq!(v[(0, 0)], 0.0);
        assert_abs_diff_eq!(a[(0, 0)], 0.0);
    }
}
