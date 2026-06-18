//! Integration test for the inverse-dynamics driver (issue #7663).
//!
//! Feeds a fake analytical RNEA callback through `run_inverse_dynamics` to
//! exercise the per-frame orchestration loop end-to-end. The `unsafe`
//! numpy-borrow in `bindings.rs` only runs under the `python` feature; here we
//! drive the same callback-shaped contract (`FnMut(frame, q, v, a) -> Result`)
//! that the PyO3 layer wraps, covering finite-difference → callback → tau
//! aggregation and the error paths.

use ndarray::{array, Array1};
use upstream_pinocchio_id::driver::{run_inverse_dynamics, DriverError};

/// Unit-mass point: tau == qddot. With quadratic q the acceleration is
/// constant, so every frame's tau must equal it.
#[test]
fn fake_rnea_unit_mass_recovers_constant_acceleration() {
    let times = array![0.0_f64, 0.1, 0.2, 0.3, 0.4];
    let q = array![[0.0_f64], [0.01], [0.04], [0.09], [0.16]];

    let mut frames_seen = Vec::new();
    let buf = run_inverse_dynamics(q.view(), times.view(), None, None, |frame, qr, vr, ar| {
        frames_seen.push(frame);
        // Sanity: callback receives correctly-shaped rows.
        assert_eq!(qr.len(), 1);
        assert_eq!(vr.len(), 1);
        assert_eq!(ar.len(), 1);
        // "RNEA" for a unit point mass: tau = a.
        Ok(ar.to_owned())
    })
    .expect("driver should succeed");

    assert_eq!(frames_seen, vec![0, 1, 2, 3, 4]);
    for i in 0..5 {
        assert!(
            (buf.tau[(i, 0)] - 2.0).abs() < 1e-9,
            "tau[{i}] = {}",
            buf.tau[(i, 0)]
        );
    }
}

/// A callback that signals failure must surface as `CallbackFailure`, never a
/// panic.
#[test]
fn fake_rnea_callback_failure_is_reported() {
    let times = array![0.0_f64, 0.1, 0.2];
    let q = array![[0.0_f64], [0.01], [0.04]];
    let err = run_inverse_dynamics(q.view(), times.view(), None, None, |frame, _, _, _| {
        if frame == 1 {
            Err("boom".to_string())
        } else {
            Ok(Array1::<f64>::zeros(1))
        }
    });
    match err {
        Err(DriverError::CallbackFailure { frame, message }) => {
            assert_eq!(frame, 1);
            assert_eq!(message, "boom");
        }
        Err(other) => panic!("expected CallbackFailure, got {other:?}"),
        Ok(_) => panic!("expected CallbackFailure, got Ok"),
    }
}

/// Overrides bypass finite differencing; the callback then sees exactly the
/// supplied qdot/qddot.
#[test]
fn fake_rnea_with_overrides_uses_supplied_derivatives() {
    let times = array![0.0_f64, 0.1, 0.2];
    let q = array![[0.0_f64], [1.0], [2.0]];
    let qdot = array![[5.0_f64], [5.0], [5.0]];
    let qddot = array![[0.0_f64], [0.0], [0.0]];
    let buf = run_inverse_dynamics(
        q.view(),
        times.view(),
        Some(qdot.view()),
        Some(qddot.view()),
        |_, _, v, a| {
            assert_eq!(v[0], 5.0);
            assert_eq!(a[0], 0.0);
            Ok(a.to_owned())
        },
    )
    .expect("driver should succeed with overrides");
    for i in 0..3 {
        assert_eq!(buf.qdot[(i, 0)], 5.0);
    }
}
