//! Parity anchors for the Python physics facades backed by `upstream-physics`.
//!
//! These fixtures cover the three public physics surfaces called out in the
//! Rust audit: RK4 integration, aerodynamics, and ball-flight orchestration.
//! Expected values were generated from the Python reference formulas used by
//! `src.shared.python.physics.*` and are intentionally crate-level tests so the
//! CI Rust gate cannot pass without exercising the published crate surface.

use upstream_physics::{
    aerodynamics::{compute_aero_forces, AeroBallProperties, AirProperties},
    ball_flight::simulate_ball_trajectory,
    rk4::{integrate, IntegratorConfig},
    Vector3,
};

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: expected {expected:.15}, got {actual:.15}, diff {:.3e}",
        (actual - expected).abs()
    );
}

#[test]
fn rk4_decay_matches_python_reference_fixture() {
    let config = IntegratorConfig {
        dt: 0.025,
        max_steps: 1_000,
    };

    let result = integrate(
        |_t, y| vec![-0.75 * y[0]],
        0.0,
        1.5,
        &[2.0],
        &config,
        None::<fn(f64, &[f64]) -> bool>,
    );

    assert!(result.completed);
    assert_eq!(result.steps_taken, 61);
    assert_close(
        result.states[result.states.len() - 1],
        0.649_304_935_480_908_6,
        1e-12,
        "rk4 final y",
    );
}

#[test]
fn aerodynamics_forces_match_python_reference_fixture() {
    let velocity = Vector3::new(62.0, -4.0, 18.0);
    let spin = Vector3::new(12.0, 265.0, -35.0);
    let forces = compute_aero_forces(
        &velocity,
        &spin,
        &AeroBallProperties::default(),
        &AirProperties::default(),
    );

    assert_close(forces.drag.x, -0.896_370_706_135_275_8, 1e-12, "drag.x");
    assert_close(forces.drag.y, 0.057_830_368_137_759_73, 1e-12, "drag.y");
    assert_close(forces.drag.z, -0.260_236_656_619_918_8, 1e-12, "drag.z");
    assert_close(forces.lift.x, 0.230_668_138_141_559_87, 1e-12, "lift.x");
    assert_close(forces.lift.y, -0.118_871_312_657_831_92, 1e-12, "lift.y");
    assert_close(forces.lift.z, -0.820_939_434_189_335_5, 1e-12, "lift.z");
    assert_close(forces.magnus.x, 0.034_732_839_829_469_9, 1e-12, "magnus.x");
    assert_close(
        forces.magnus.y,
        -0.017_899_040_136_741_94,
        1e-12,
        "magnus.y",
    );
    assert_close(
        forces.magnus.z,
        -0.123_612_901_665_227_87,
        1e-12,
        "magnus.z",
    );
}

#[test]
fn ball_flight_matches_python_reference_fixture() {
    let config = IntegratorConfig {
        dt: 0.02,
        max_steps: 1_000,
    };

    let result = simulate_ball_trajectory(
        [0.0, 0.0, 0.5],
        [32.0, 0.5, 18.0],
        [0.0, 1.0, 0.0],
        220.0,
        [0.0, 0.0, -9.81],
        [-2.0, 0.0, 0.0],
        &AeroBallProperties::default(),
        &AirProperties::default(),
        &config,
    );

    assert!(result.completed);
    let final_point = result.points.last().expect("trajectory has a final point");
    assert_eq!(result.steps, 104);
    assert_close(
        final_point.t,
        2.080_000_000_000_001_4,
        1e-12,
        "landing time",
    );
    assert_close(final_point.x, 56.845_911_438_624_14, 1e-10, "landing x");
    assert_close(final_point.y, 0.864_213_713_705_367, 1e-12, "landing y");
    assert_close(final_point.z, -0.197_274_619_688_210_97, 1e-12, "landing z");
}
