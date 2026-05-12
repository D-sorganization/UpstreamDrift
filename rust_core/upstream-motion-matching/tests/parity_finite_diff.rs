//! Pure-Rust parity / regression tests for [`finite_diff_uniform`].
//!
//! The cross-language parity check vs the Python
//! `PinocchioInverseDynMatchingSolver._finite_difference` lives in
//! `tests/unit/motion_pipeline/test_rust_finite_diff_parity.py` (only
//! runnable when the crate is built with `--features python`). Here we
//! cover analytic-trajectory regressions that don't depend on Python.

use approx::assert_relative_eq;
use upstream_motion_matching::finite_diff_uniform;

fn sine_traj(n: usize, d: usize, dt: f64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let t = i as f64 * dt;
            (0..d)
                .map(|j| {
                    let f = 1.0 + 0.5 * j as f64;
                    (2.0 * std::f64::consts::PI * f * t + 0.1 * j as f64).sin()
                })
                .collect()
        })
        .collect()
}

#[test]
fn sine_trajectory_qdot_close_to_analytic_derivative() {
    // For dense enough sampling, the central-difference qdot should
    // approach the analytic derivative within O(dt^2). We use 1000
    // frames at 240 Hz with f = 1 Hz on DOF 0; the truncation error is
    // ~ (2*pi)^3 * dt^2 / 6 ≈ 0.001.
    let dt = 1.0 / 240.0;
    let q = sine_traj(1000, 1, dt);
    let r = finite_diff_uniform(&q, dt).unwrap();

    for i in 1..999 {
        let t = i as f64 * dt;
        let analytic = 2.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * t).cos();
        // 1e-3 tolerance: dominated by O(dt^2) truncation, not bits.
        assert_relative_eq!(r.qdot[i][0], analytic, max_relative = 1e-3, epsilon = 1e-3);
    }
}

#[test]
fn output_shape_matches_input_shape() {
    let dt = 1.0 / 100.0;
    let q = sine_traj(50, 7, dt);
    let r = finite_diff_uniform(&q, dt).unwrap();
    assert_eq!(r.qdot.len(), 50);
    assert_eq!(r.qddot.len(), 50);
    for row in &r.qdot {
        assert_eq!(row.len(), 7);
    }
    for row in &r.qddot {
        assert_eq!(row.len(), 7);
    }
}

#[test]
fn boundary_qddot_copies_neighbor_when_n_ge_3() {
    // Quadratic data: interior qddot is exactly the constant. The
    // boundary frames must equal the interior — that's the reference
    // implementation's `if len(times) >= 3` branch.
    let dt = 0.01;
    let a = 7.0;
    let q: Vec<Vec<f64>> = (0..30)
        .map(|i| {
            let t = i as f64 * dt;
            vec![0.5 * a * t * t, -0.25 * a * t * t]
        })
        .collect();
    let r = finite_diff_uniform(&q, dt).unwrap();
    assert_relative_eq!(r.qddot[0][0], a, max_relative = 1e-10);
    assert_relative_eq!(r.qddot[0][1], -0.5 * a, max_relative = 1e-10);
    assert_relative_eq!(r.qddot[29][0], a, max_relative = 1e-10);
    assert_relative_eq!(r.qddot[29][1], -0.5 * a, max_relative = 1e-10);
}
