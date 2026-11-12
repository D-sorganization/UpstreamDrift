//! Acceptance contract for issue #15045 (SCI-07 / SCI-08).

use approx::assert_relative_eq;
use upstream_muscle::activation::{
    ActivationDynamics, DEFAULT_MAX_ACTIVATION_INTEGRATION_DT, DEFAULT_MIN_ACTIVATION,
    DEFAULT_TAU_ACT,
};
use upstream_muscle::hill::{
    f_t_with_reference_strain, f_v, DEFAULT_TENDON_STRAIN_AT_FMAX, HILL_A_CONCENTRIC,
};

/// Hill concentric hyperbola: at normalized velocity `v = -1` (full Vmax shortening) force is zero.
#[inline]
fn f_v_hill_concentric_at(v: f64) -> f64 {
    (1.0 + v) / (1.0 - v / HILL_A_CONCENTRIC)
}

#[test]
fn f_v_is_zero_at_and_beyond_max_shortening() {
    assert_relative_eq!(f_v(-1.0), 0.0, epsilon = 1e-15);
    assert_relative_eq!(f_v(-1.5), 0.0, epsilon = 1e-15);
    // Derived limit: uncorrected hyperbola is negative past v = -1.
    assert!(f_v_hill_concentric_at(-1.5) < 0.0);
}

#[test]
fn f_v_concentric_matches_hyperbola_inside_physical_range() {
    let v = -0.5;
    assert_relative_eq!(f_v(v), f_v_hill_concentric_at(v), epsilon = 1e-12);
}

#[test]
fn f_v_force_velocity_join_is_c0_not_c1_documented_kink() {
    let h = 1e-8;
    let left = (f_v(-h) - f_v(-2.0 * h)) / h;
    let right = (f_v(h) - f_v(0.0)) / h;
    assert_relative_eq!(f_v(0.0), 1.0, epsilon = 1e-12);
    assert!(
        (left - right).abs() > 1e-3,
        "piecewise Hill model has a documented C¹ kink at v=0 (left slope {left}, right {right})"
    );
}

#[test]
fn f_t_toe_linear_junction_is_c0_parity_with_python() {
    let eps_0 = DEFAULT_TENDON_STRAIN_AT_FMAX;
    let eps_toe = 0.609 * eps_0;
    let l_toe = 1.0 + eps_toe;
    let h = 1e-8;
    assert_relative_eq!(
        f_t_with_reference_strain(l_toe - h, eps_0),
        f_t_with_reference_strain(l_toe + h, eps_0),
        epsilon = 1e-6
    );
}

/// Closed-form rise over one interval when τ is frozen at the activation floor.
fn expected_floor_exponential_step(u: f64, a0: f64, dt: f64) -> f64 {
    let tau = DEFAULT_TAU_ACT * (0.5 + 1.5 * a0);
    u + (a0 - u) * (-dt / tau).exp()
}

/// Single forward-Euler step with output clamp (pre-#15045 behaviour).
fn naive_euler_clamped(dynamics: &ActivationDynamics, u: f64, a: f64, dt: f64) -> f64 {
    let dadt = dynamics.compute_derivative(u, a);
    (a + dadt * dt).clamp(dynamics.min_activation, 1.0)
}

#[test]
fn activation_one_ms_matches_exponential_formula() {
    let dynamics = ActivationDynamics::new(DEFAULT_TAU_ACT, 0.040, DEFAULT_MIN_ACTIVATION).unwrap();
    let dt = 0.001;
    let u = 1.0;
    let a0 = DEFAULT_MIN_ACTIVATION;
    let got = dynamics.update(u, a0, dt).expect("update");
    let expected = expected_floor_exponential_step(u, a0, dt);
    assert_relative_eq!(got, expected, epsilon = 1e-9);
}

#[test]
fn activation_mocap_steps_remain_below_clamp_and_rise() {
    let dynamics = ActivationDynamics::new(DEFAULT_TAU_ACT, 0.040, DEFAULT_MIN_ACTIVATION).unwrap();
    let u = 1.0;
    let a0 = DEFAULT_MIN_ACTIVATION;
    let mut prev = a0;
    for dt in [0.001, 0.010, 0.020] {
        let got = dynamics.update(u, a0, dt).expect("update");
        assert!(got > prev);
        assert!(got < 1.0);
        prev = got;
    }
}

#[test]
fn activation_avoids_euler_overshoot_clamp_at_mocap_dt() {
    let dynamics = ActivationDynamics::new(DEFAULT_TAU_ACT, 0.040, DEFAULT_MIN_ACTIVATION).unwrap();
    let dt = 0.020;
    let naive = naive_euler_clamped(&dynamics, 1.0, DEFAULT_MIN_ACTIVATION, dt);
    let integrated = dynamics
        .update(1.0, DEFAULT_MIN_ACTIVATION, dt)
        .expect("update");
    assert_relative_eq!(naive, 1.0, epsilon = 1e-12);
    assert!(
        integrated < 1.0 - 1e-6,
        "20 ms step must not ride the output clamp (got {integrated})"
    );
    assert!(integrated < naive - 0.1);
}

#[test]
fn activation_integration_substep_cap_is_half_min_tau_at_floor() {
    let tau_floor = DEFAULT_TAU_ACT * (0.5 + 1.5 * DEFAULT_MIN_ACTIVATION);
    let expected_cap = 0.5 * tau_floor;
    assert_relative_eq!(
        DEFAULT_MAX_ACTIVATION_INTEGRATION_DT,
        expected_cap,
        epsilon = 1e-6
    );
}
