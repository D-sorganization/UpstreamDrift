//! Generic RK4 (Runge-Kutta 4th order) numerical integrator.
//!
//! This module provides a configurable RK4 integrator for solving
//! ordinary differential equations (ODEs) of the form:
//!
//!   dy/dt = f(t, y)
//!
//! The integrator supports:
//! - Fixed-step integration
//! - Early termination conditions
//! - State history recording
//!
//! # Design by Contract
//!
//! - `dt > 0` (positive time step)
//! - `t_end > t_start` (forward integration)
//! - State vector must be finite (no NaN/Inf)

use serde::{Deserialize, Serialize};

/// Configuration for the RK4 integrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct IntegratorConfig {
    /// Fixed time step [s].
    pub dt: f64,
    /// Maximum number of steps (safety limit).
    pub max_steps: usize,
}

impl Default for IntegratorConfig {
    fn default() -> Self {
        Self {
            dt: 0.001,
            max_steps: 100_000,
        }
    }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl IntegratorConfig {
    #[new]
    #[pyo3(signature = (dt=0.001, max_steps=100_000))]
    fn py_new(dt: f64, max_steps: usize) -> Self {
        Self { dt, max_steps }
    }
}

/// Result of an RK4 integration run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct IntegrationResult {
    /// Time values at each recorded step.
    pub times: Vec<f64>,
    /// State vectors at each recorded step (flattened: [n_steps * state_dim]).
    pub states: Vec<f64>,
    /// Dimension of the state vector.
    pub state_dim: usize,
    /// Number of steps taken.
    pub steps_taken: usize,
    /// Whether integration completed normally.
    pub completed: bool,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl IntegrationResult {
    /// Number of time steps recorded.
    #[getter]
    fn num_points(&self) -> usize {
        self.times.len()
    }

    /// Get state at step index as a Vec.
    fn get_state(&self, index: usize) -> Vec<f64> {
        let start = index * self.state_dim;
        let end = start + self.state_dim;
        self.states[start..end].to_vec()
    }
}

// ── WASM bindings ────────────────────────────────────────────────────────────

#[cfg(feature = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen]
impl IntegratorConfig {
    /// Create integrator configuration.
    #[wasm_bindgen(constructor)]
    pub fn wasm_new(dt: f64, max_steps: usize) -> Self {
        Self { dt, max_steps }
    }
}

/// Integrate an ODE using RK4 with a closure-based derivative function.
///
/// # Arguments
/// * `f` - Derivative function: `f(t, state) -> d_state/dt`
/// * `t_start` - Initial time
/// * `t_end` - Final time
/// * `y0` - Initial state vector
/// * `config` - Integrator configuration
/// * `terminate` - Optional early termination: `terminate(t, state) -> bool`
///
/// # Panics (debug only)
/// Panics if `dt <= 0`, `t_end <= t_start`, or initial state contains NaN.
pub fn integrate<F, T>(
    f: F,
    t_start: f64,
    t_end: f64,
    y0: &[f64],
    config: &IntegratorConfig,
    terminate: Option<T>,
) -> IntegrationResult
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    T: Fn(f64, &[f64]) -> bool,
{
    // DbC: Precondition validation
    debug_assert!(config.dt > 0.0, "Time step must be positive");
    debug_assert!(t_end > t_start, "t_end must be greater than t_start");
    debug_assert!(
        y0.iter().all(|v| v.is_finite()),
        "Initial state must be finite"
    );

    let state_dim = y0.len();
    let mut t = t_start;
    let mut y = y0.to_vec();
    let dt = config.dt;

    // Pre-allocate with estimated capacity
    let estimated_steps = ((t_end - t_start) / dt).ceil() as usize + 1;
    let capacity = estimated_steps.min(config.max_steps);
    let mut times = Vec::with_capacity(capacity);
    let mut states = Vec::with_capacity(capacity * state_dim);

    // Record initial state
    times.push(t);
    states.extend_from_slice(&y);

    let mut steps = 0usize;
    let mut k1 = vec![0.0; state_dim];
    let mut k2 = vec![0.0; state_dim];
    let mut k3 = vec![0.0; state_dim];
    let mut k4 = vec![0.0; state_dim];
    let mut y_temp = vec![0.0; state_dim];

    while t < t_end && steps < config.max_steps {
        // Clamp final step
        let h = if t + dt > t_end { t_end - t } else { dt };

        // k1 = f(t, y)
        let k1_val = f(t, &y);
        k1.copy_from_slice(&k1_val);

        // k2 = f(t + h/2, y + h/2 * k1)
        for i in 0..state_dim {
            y_temp[i] = y[i] + 0.5 * h * k1[i];
        }
        let k2_val = f(t + 0.5 * h, &y_temp);
        k2.copy_from_slice(&k2_val);

        // k3 = f(t + h/2, y + h/2 * k2)
        for i in 0..state_dim {
            y_temp[i] = y[i] + 0.5 * h * k2[i];
        }
        let k3_val = f(t + 0.5 * h, &y_temp);
        k3.copy_from_slice(&k3_val);

        // k4 = f(t + h, y + h * k3)
        for i in 0..state_dim {
            y_temp[i] = y[i] + h * k3[i];
        }
        let k4_val = f(t + h, &y_temp);
        k4.copy_from_slice(&k4_val);

        // y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        for i in 0..state_dim {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h;
        steps += 1;

        // Record state
        times.push(t);
        states.extend_from_slice(&y);

        // Check termination
        if let Some(ref term) = terminate {
            if term(t, &y) {
                return IntegrationResult {
                    times,
                    states,
                    state_dim,
                    steps_taken: steps,
                    completed: false,
                };
            }
        }
    }

    IntegrationResult {
        times,
        states,
        state_dim,
        steps_taken: steps,
        completed: t >= t_end - 1e-12,
    }
}

// ── Tests (TDD) ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Simple exponential decay dy/dt = -y, y(0) = 1.
    /// Exact solution: y(t) = exp(-t).
    #[test]
    fn test_exponential_decay() {
        let config = IntegratorConfig {
            dt: 0.001,
            max_steps: 10_000,
        };

        let result = integrate(
            |_t, y| vec![-y[0]],
            0.0,
            1.0,
            &[1.0],
            &config,
            None::<fn(f64, &[f64]) -> bool>,
        );

        assert!(result.completed);
        let final_y = result.states[result.states.len() - 1];
        let exact = (-1.0_f64).exp();
        assert!(
            (final_y - exact).abs() < 1e-8,
            "Expected ~{exact:.10}, got {final_y:.10}"
        );
    }

    /// Test 2: Harmonic oscillator: x'' + x = 0, x(0)=1, x'(0)=0.
    /// State: [x, v], derivatives: [v, -x].
    /// Exact: x(t) = cos(t), v(t) = -sin(t).
    #[test]
    fn test_harmonic_oscillator() {
        let config = IntegratorConfig {
            dt: 0.0001,
            max_steps: 200_000,
        };

        let result = integrate(
            |_t, y| vec![y[1], -y[0]],
            0.0,
            std::f64::consts::TAU, // Full period (2π)
            &[1.0, 0.0],
            &config,
            None::<fn(f64, &[f64]) -> bool>,
        );

        assert!(result.completed);
        let n = result.times.len();

        // After full period, should return to initial state
        let final_x = result.states[(n - 1) * 2];
        let final_v = result.states[(n - 1) * 2 + 1];

        assert!(
            (final_x - 1.0).abs() < 1e-5,
            "x should return to 1.0 after full period, got {final_x}"
        );
        assert!(
            final_v.abs() < 1e-5,
            "v should return to 0.0 after full period, got {final_v}"
        );
    }

    /// Test 3: Linear ODE dy/dt = 1, y(0) = 0. Exact: y(t) = t.
    #[test]
    fn test_constant_derivative() {
        let config = IntegratorConfig {
            dt: 0.01,
            max_steps: 1_000,
        };

        let result = integrate(
            |_t, _y| vec![1.0],
            0.0,
            5.0,
            &[0.0],
            &config,
            None::<fn(f64, &[f64]) -> bool>,
        );

        assert!(result.completed);
        let final_y = result.states[result.states.len() - 1];
        assert!((final_y - 5.0).abs() < 1e-10, "Expected 5.0, got {final_y}");
    }

    /// Test 4: Verify early termination when state crosses threshold.
    #[test]
    fn test_early_termination() {
        let config = IntegratorConfig {
            dt: 0.01,
            max_steps: 10_000,
        };

        // dy/dt = 1 (linear growth), terminate when y >= 3.0
        let result = integrate(
            |_t, _y| vec![1.0],
            0.0,
            100.0, // Would go to 100 without termination
            &[0.0],
            &config,
            Some(|_t: f64, y: &[f64]| y[0] >= 3.0),
        );

        assert!(!result.completed, "Should have terminated early");
        let final_y = result.states[result.states.len() - 1];
        assert!(
            final_y >= 3.0,
            "Should have crossed threshold, got {final_y}"
        );
        assert!(final_y < 4.0, "Should stop near threshold, got {final_y}");
    }

    /// Test 5: Verify max_steps safety limit.
    #[test]
    fn test_max_steps_limit() {
        let config = IntegratorConfig {
            dt: 0.01,
            max_steps: 10,
        };

        let result = integrate(
            |_t, _y| vec![1.0],
            0.0,
            1000.0,
            &[0.0],
            &config,
            None::<fn(f64, &[f64]) -> bool>,
        );

        assert!(!result.completed, "Should not complete with max_steps=10");
        assert_eq!(result.steps_taken, 10);
    }

    /// Test 6: Verify state history is recorded correctly.
    #[test]
    fn test_state_history_recording() {
        let config = IntegratorConfig {
            dt: 0.1,
            max_steps: 100,
        };

        let result = integrate(
            |_t, _y| vec![1.0, -1.0],
            0.0,
            1.0,
            &[0.0, 10.0],
            &config,
            None::<fn(f64, &[f64]) -> bool>,
        );

        assert_eq!(result.state_dim, 2);
        // 10 steps + initial = 11, possibly +1 from final step clamping
        assert!(result.times.len() >= 11 && result.times.len() <= 12);
        assert_eq!(result.states.len(), result.times.len() * 2);

        // Initial state
        assert!((result.states[0] - 0.0).abs() < 1e-12);
        assert!((result.states[1] - 10.0).abs() < 1e-12);
    }

    /// Test 7: Projectile motion under gravity (2D: [x, y, vx, vy]).
    /// Validates physics-relevant ODE solving.
    #[test]
    fn test_projectile_motion() {
        let g = 9.81;
        let config = IntegratorConfig {
            dt: 0.001,
            max_steps: 100_000,
        };

        // Launch at 45°, 20 m/s
        let v0 = 20.0;
        let angle = std::f64::consts::FRAC_PI_4;
        let vx0 = v0 * angle.cos();
        let vy0 = v0 * angle.sin();

        let result = integrate(
            move |_t, _y| vec![_y[2], _y[3], 0.0, -g],
            0.0,
            3.0,
            &[0.0, 0.0, vx0, vy0],
            &config,
            Some(|t: f64, y: &[f64]| t > 0.1 && y[1] < 0.0), // Ground hit
        );

        // Check max height is approximately v0² sin²(θ) / (2g) ≈ 10.19m
        let expected_max_h = vy0 * vy0 / (2.0 * g);
        let max_h = (0..result.times.len())
            .map(|i| result.states[i * 4 + 1])
            .fold(0.0_f64, f64::max);

        assert!(
            (max_h - expected_max_h).abs() < 0.1,
            "Max height: expected {expected_max_h:.2}, got {max_h:.2}"
        );

        // Check range is approximately v0² sin(2θ) / g ≈ 40.77m
        let expected_range = v0 * v0 * (2.0 * angle).sin() / g;
        let n = result.times.len();
        let final_x = result.states[(n - 1) * 4];
        assert!(
            (final_x - expected_range).abs() < 1.0,
            "Range: expected {expected_range:.2}, got {final_x:.2}"
        );
    }

    /// Test 8: Zero-length integration returns initial state only.
    #[test]
    fn test_zero_duration() {
        let config = IntegratorConfig::default();

        let result = integrate(
            |_t, _y| vec![1.0],
            0.0,
            0.0001, // Very short, 1 step
            &[42.0],
            &config,
            None::<fn(f64, &[f64]) -> bool>,
        );

        assert!(result.times.len() >= 2); // At least initial + 1 step
        assert!((result.states[0] - 42.0).abs() < 1e-12);
    }
}
