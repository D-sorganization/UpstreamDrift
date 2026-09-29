//! Multi-muscle moment summation — convert muscle forces to joint torques.
//!
//! For a single joint actuated by `M` muscles with moment arms `r_i`, the
//! net joint torque is the dot product `τ = Σ r_i · F_i`. Extending to
//! `J` joints with `M` muscles, the moment-arm matrix `R` is `(J × M)` and
//! the joint torque vector is `τ = R · F`.
//!
//! This is the routine that dominates RL inner loops once muscle forces
//! are in hand. The batched form (`crate::batch::joint_torques_batched`)
//! fans it out across many trajectory steps in parallel.

/// Compute net torque for a single joint given matched-length slices of
/// moment arms and muscle forces. `O(M)`.
///
/// `r[i]` is the moment arm of muscle `i` about the joint (sign convention
/// matches the Python source: positive for flexion, negative for
/// extension). `f[i]` is the corresponding muscle force in newtons.
///
/// # Panics
/// Panics if `r.len() != f.len()` (caller-side bug; the batched APIs
/// validate up-front).
#[inline]
pub fn joint_torque(r: &[f64], f: &[f64]) -> f64 {
    assert_eq!(
        r.len(),
        f.len(),
        "moment arm and force vectors must match in length"
    );
    let mut acc = 0.0;
    for i in 0..r.len() {
        acc += r[i] * f[i];
    }
    acc
}

/// Compute joint torques for `J` joints given the row-major moment-arm
/// matrix `R` (shape `J × M`) and the muscle force vector `f` (length `M`).
///
/// Writes results into `out` (length `J`). Equivalent to `τ = R · f`.
///
/// # Panics
/// Panics if shapes are inconsistent.
pub fn joint_torques(
    r_row_major: &[f64],
    n_joints: usize,
    n_muscles: usize,
    f: &[f64],
    out: &mut [f64],
) {
    assert_eq!(
        r_row_major.len(),
        n_joints * n_muscles,
        "moment-arm matrix has wrong length"
    );
    assert_eq!(f.len(), n_muscles, "force vector length mismatch");
    assert_eq!(out.len(), n_joints, "output vector length mismatch");
    for j in 0..n_joints {
        let row = &r_row_major[j * n_muscles..(j + 1) * n_muscles];
        let mut acc = 0.0;
        for i in 0..n_muscles {
            acc += row[i] * f[i];
        }
        out[j] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn elbow_agonist_antagonist() {
        // Biceps and brachialis are flexors (r > 0); triceps is extensor.
        let r = [0.04, 0.03, -0.035];
        let f = [400.0, 200.0, 240.0];
        // Expected: 0.04*400 + 0.03*200 - 0.035*240 = 16 + 6 - 8.4 = 13.6.
        assert_relative_eq!(joint_torque(&r, &f), 13.6, epsilon = 1e-9);
    }

    #[test]
    fn matrix_form_matches_manual_dot_products() {
        // 2 joints, 3 muscles.
        let r = [0.04, 0.03, -0.035, 0.01, 0.02, 0.0];
        let f = [400.0, 200.0, 240.0];
        let mut out = [0.0; 2];
        joint_torques(&r, 2, 3, &f, &mut out);
        assert_relative_eq!(out[0], 0.04 * 400.0 + 0.03 * 200.0 - 0.035 * 240.0);
        assert_relative_eq!(out[1], 0.01 * 400.0 + 0.02 * 200.0 + 0.0 * 240.0);
    }

    #[test]
    fn empty_inputs_yield_zero() {
        assert_eq!(joint_torque(&[], &[]), 0.0);
    }
}
