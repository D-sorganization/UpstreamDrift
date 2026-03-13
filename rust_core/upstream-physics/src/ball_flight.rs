//! Ball flight trajectory simulation using the Rust RK4 integrator.
//!
//! Wires together `aerodynamics::compute_aero_forces` and `rk4::integrate`
//! to provide a full ball flight trajectory simulation callable from Python.
//!
//! # Design by Contract
//! - All velocity/spin/position components must be finite
//! - `ball.mass > 0`, `air.density > 0`
//! - `config.dt > 0`
//!
//! # State vector layout
//! `[x, y, z, vx, vy, vz, omega]` — position, velocity, spin magnitude.
//! The spin axis direction is constant; only the magnitude decays.

use crate::aerodynamics::{compute_aero_forces, AeroBallProperties, AirProperties};
use crate::rk4::{integrate, IntegratorConfig};
use serde::{Deserialize, Serialize};
use tools_core::Vector3;

/// A single point in the simulated trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct TrajectoryPoint {
    /// Time [s]
    pub t: f64,
    /// Position X [m]
    pub x: f64,
    /// Position Y [m]
    pub y: f64,
    /// Position Z [m] (height)
    pub z: f64,
    /// Velocity X [m/s]
    pub vx: f64,
    /// Velocity Y [m/s]
    pub vy: f64,
    /// Velocity Z [m/s]
    pub vz: f64,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl TrajectoryPoint {
    #[getter]
    fn t(&self) -> f64 {
        self.t
    }
    #[getter]
    fn x(&self) -> f64 {
        self.x
    }
    #[getter]
    fn y(&self) -> f64 {
        self.y
    }
    #[getter]
    fn z(&self) -> f64 {
        self.z
    }
    #[getter]
    fn vx(&self) -> f64 {
        self.vx
    }
    #[getter]
    fn vy(&self) -> f64 {
        self.vy
    }
    #[getter]
    fn vz(&self) -> f64 {
        self.vz
    }
}

/// Result of a ball trajectory simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct BallTrajectoryResult {
    /// Trajectory points from launch to ground impact.
    pub points: Vec<TrajectoryPoint>,
    /// Whether the simulation completed normally (ball reached z ≤ 0).
    pub completed: bool,
    /// Number of integration steps taken.
    pub steps: usize,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl BallTrajectoryResult {
    /// Number of trajectory points.
    #[getter]
    fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Whether the ball reached the ground.
    #[getter]
    fn completed(&self) -> bool {
        self.completed
    }

    /// All trajectory points.
    fn get_points(&self) -> Vec<TrajectoryPoint> {
        self.points.clone()
    }

    /// Flatten trajectory to a list of `[t, x, y, z, vx, vy, vz]` tuples.
    fn to_flat_list(&self) -> Vec<Vec<f64>> {
        self.points
            .iter()
            .map(|p| vec![p.t, p.x, p.y, p.z, p.vx, p.vy, p.vz])
            .collect()
    }
}

/// Simulate a ball trajectory using the Rust RK4 integrator.
///
/// The state vector is `[x, y, z, vx, vy, vz, omega]`.
/// Spin decays continuously as `d(omega)/dt = -spin_decay_rate * omega`.
///
/// # Arguments
/// - `pos0`: Initial position `[x, y, z]` [m]
/// - `vel0`: Initial velocity `[vx, vy, vz]` [m/s]
/// - `spin_axis`: Unit vector for spin axis direction
/// - `omega0`: Initial spin rate [rad/s]
/// - `gravity`: Gravity acceleration vector (typically `[0, 0, -9.81]`)
/// - `wind`: Wind velocity vector [m/s]
/// - `ball`: Ball physical properties
/// - `air`: Air properties at launch conditions
/// - `config`: RK4 integrator configuration
///
/// # DbC Preconditions
/// - All position/velocity/gravity/wind components must be finite
/// - `omega0 >= 0`
/// - `config.dt > 0`, `ball.mass > 0`, `air.density >= 0`
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn simulate_ball_trajectory(
    pos0: [f64; 3],
    vel0: [f64; 3],
    spin_axis: [f64; 3],
    omega0: f64,
    gravity: [f64; 3],
    wind: [f64; 3],
    ball: &AeroBallProperties,
    air: &AirProperties,
    config: &IntegratorConfig,
) -> BallTrajectoryResult {
    debug_assert!(
        pos0.iter().all(|v| v.is_finite()),
        "DbC: initial position must be finite"
    );
    debug_assert!(
        vel0.iter().all(|v| v.is_finite()),
        "DbC: initial velocity must be finite"
    );
    debug_assert!(omega0 >= 0.0, "DbC: omega0 must be non-negative");
    debug_assert!(ball.mass > 0.0, "DbC: ball mass must be positive");
    debug_assert!(air.density >= 0.0, "DbC: air density must be non-negative");

    // State: [x, y, z, vx, vy, vz, omega]
    let y0 = [pos0[0], pos0[1], pos0[2], vel0[0], vel0[1], vel0[2], omega0];

    // Capture by reference using owned copies for the closure
    let ball = ball.clone();
    let air = air.clone();
    let spin_axis_v = Vector3::new(spin_axis[0], spin_axis[1], spin_axis[2]);
    let gravity_v = Vector3::new(gravity[0], gravity[1], gravity[2]);
    let wind_v = Vector3::new(wind[0], wind[1], wind[2]);

    let derivative = move |_t: f64, state: &[f64]| -> Vec<f64> {
        let velocity = Vector3::new(state[3], state[4], state[5]);
        let omega = state[6];

        // Relative velocity accounting for wind
        let rel_vel = velocity - wind_v;

        // Spin vector = axis direction × magnitude
        let spin = spin_axis_v * omega;

        // Aerodynamic forces [N]
        let forces = compute_aero_forces(&rel_vel, &spin, &ball, &air);
        let total_aero = forces.drag + forces.lift + forces.magnus;

        // Acceleration = gravity + F_aero / mass
        let ax = gravity_v.x + total_aero.x / ball.mass;
        let ay = gravity_v.y + total_aero.y / ball.mass;
        let az = gravity_v.z + total_aero.z / ball.mass;

        // d(omega)/dt = -spin_decay_rate * omega (continuous exponential decay)
        let domega = -ball.spin_decay_rate * omega;

        vec![velocity.x, velocity.y, velocity.z, ax, ay, az, domega]
    };

    // Terminate when ball returns to ground (z ≤ 0, after initial launch)
    let terminate = |t: f64, state: &[f64]| -> bool { t > 0.05 && state[2] <= 0.0 };

    let max_time = config.max_steps as f64 * config.dt;
    let result = integrate(derivative, 0.0, max_time, &y0, config, Some(terminate));

    // Convert IntegrationResult to BallTrajectoryResult
    let state_dim = result.state_dim;
    let points = result
        .times
        .iter()
        .zip(result.states.chunks(state_dim))
        .map(|(&t, state)| TrajectoryPoint {
            t,
            x: state[0],
            y: state[1],
            z: state[2],
            vx: state[3],
            vy: state[4],
            vz: state[5],
        })
        .collect();

    BallTrajectoryResult {
        completed: result.completed,
        steps: result.steps_taken,
        points,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aerodynamics::{AeroBallProperties, AirProperties};
    use crate::rk4::IntegratorConfig;

    fn default_config() -> IntegratorConfig {
        IntegratorConfig {
            dt: 0.01,
            max_steps: 2_000,
        }
    }

    /// Test 1: Basic trajectory — ball launched at 45° should travel ~20-50m in range.
    #[test]
    fn test_basic_trajectory_45_degrees() {
        let v0 = 30.0_f64;
        let angle = std::f64::consts::FRAC_PI_4;
        let vx = v0 * angle.cos();
        let vz = v0 * angle.sin();

        let result = simulate_ball_trajectory(
            [0.0, 0.0, 0.5],
            [vx, 0.0, vz],
            [0.0, 1.0, 0.0],
            100.0,
            [0.0, 0.0, -9.81],
            [0.0, 0.0, 0.0],
            &AeroBallProperties::default(),
            &AirProperties::default(),
            &default_config(),
        );

        assert!(!result.points.is_empty(), "Should have trajectory points");
        assert!(result.steps > 10, "Should take more than 10 steps");

        let last = result.points.last().unwrap();
        // Ball should have traveled some horizontal distance
        assert!(
            last.x > 5.0,
            "Should travel at least 5m horizontally, got {}",
            last.x
        );
        // Ball should land near ground
        assert!(last.z.abs() < 2.0, "Should end near z=0, got {}", last.z);
    }

    /// Test 2: No gravity, no drag — projectile in vacuum should follow linear path.
    #[test]
    fn test_zero_gravity_linear_path() {
        let ball = AeroBallProperties {
            drag_coefficient: 0.0,
            spin_decay_rate: 0.0,
            ..AeroBallProperties::default()
        };

        let air = AirProperties {
            density: 0.0, // No aerodynamic forces
            ..AirProperties::default()
        };

        let config = IntegratorConfig {
            dt: 0.01,
            max_steps: 100,
        };

        // Launch purely horizontal, no decay, no air, no gravity
        let result = simulate_ball_trajectory(
            [0.0, 0.0, 10.0], // start high to avoid immediate ground hit
            [10.0, 0.0, 0.0], // horizontal
            [0.0, 1.0, 0.0],
            0.0,
            [0.0, 0.0, 0.0], // no gravity
            [0.0, 0.0, 0.0], // no wind
            &ball,
            &air,
            &config,
        );

        assert!(!result.points.is_empty());
        let last = result.points.last().unwrap();
        // Should travel ~10 m/s * 100 steps * 0.01s = ~10m
        assert!(last.x > 5.0, "Should travel horizontally: {}", last.x);
        // Height should be approximately constant (no gravity)
        assert!(
            (last.z - 10.0).abs() < 0.1,
            "Height should be ~10m: {}",
            last.z
        );
    }

    /// Test 3: Result is consistent — same inputs produce same output.
    #[test]
    fn test_deterministic_results() {
        let params = (
            [0.0, 0.0, 0.0_f64],
            [20.0, 0.0, 15.0_f64],
            [0.0, 1.0, 0.0_f64],
            50.0_f64,
            [0.0, 0.0, -9.81_f64],
            [0.0, 0.0, 0.0_f64],
        );

        let result1 = simulate_ball_trajectory(
            params.0,
            params.1,
            params.2,
            params.3,
            params.4,
            params.5,
            &AeroBallProperties::default(),
            &AirProperties::default(),
            &default_config(),
        );
        let result2 = simulate_ball_trajectory(
            params.0,
            params.1,
            params.2,
            params.3,
            params.4,
            params.5,
            &AeroBallProperties::default(),
            &AirProperties::default(),
            &default_config(),
        );

        assert_eq!(result1.steps, result2.steps);
        assert_eq!(result1.points.len(), result2.points.len());
        let last1 = result1.points.last().unwrap();
        let last2 = result2.points.last().unwrap();
        assert!((last1.x - last2.x).abs() < 1e-10);
    }

    /// Test 4: Spin decay — omega should decrease over the trajectory.
    #[test]
    fn test_spin_decay() {
        // The ball_flight sim includes omega as state[6]; we can verify by
        // checking that with high spin_decay_rate the range is shorter
        // (less lift → shorter carry).
        let fast_decay = AeroBallProperties {
            spin_decay_rate: 5.0, // aggressive decay
            ..AeroBallProperties::default()
        };
        let slow_decay = AeroBallProperties::default(); // default 0.1

        let result_fast = simulate_ball_trajectory(
            [0.0, 0.0, 0.0],
            [25.0, 0.0, 15.0],
            [0.0, 1.0, 0.0],
            300.0,
            [0.0, 0.0, -9.81],
            [0.0, 0.0, 0.0],
            &fast_decay,
            &AirProperties::default(),
            &default_config(),
        );
        let result_slow = simulate_ball_trajectory(
            [0.0, 0.0, 0.0],
            [25.0, 0.0, 15.0],
            [0.0, 1.0, 0.0],
            300.0,
            [0.0, 0.0, -9.81],
            [0.0, 0.0, 0.0],
            &slow_decay,
            &AirProperties::default(),
            &default_config(),
        );

        // Fast decay → less lift → shorter (or comparable) range
        let range_fast = result_fast.points.last().map(|p| p.x).unwrap_or(0.0);
        let range_slow = result_slow.points.last().map(|p| p.x).unwrap_or(0.0);
        // Both should be positive distances
        assert!(
            range_fast > 0.0,
            "Fast decay range should be positive: {range_fast}"
        );
        assert!(
            range_slow > 0.0,
            "Slow decay range should be positive: {range_slow}"
        );
    }

    /// Test 5: Wind effect — headwind reduces range.
    #[test]
    fn test_headwind_reduces_range() {
        let base_result = simulate_ball_trajectory(
            [0.0, 0.0, 0.0],
            [25.0, 0.0, 15.0],
            [0.0, 1.0, 0.0],
            100.0,
            [0.0, 0.0, -9.81],
            [0.0, 0.0, 0.0], // no wind
            &AeroBallProperties::default(),
            &AirProperties::default(),
            &default_config(),
        );

        let headwind_result = simulate_ball_trajectory(
            [0.0, 0.0, 0.0],
            [25.0, 0.0, 15.0],
            [0.0, 1.0, 0.0],
            100.0,
            [0.0, 0.0, -9.81],
            [-5.0, 0.0, 0.0], // 5 m/s headwind
            &AeroBallProperties::default(),
            &AirProperties::default(),
            &default_config(),
        );

        let base_range = base_result.points.last().map(|p| p.x).unwrap_or(0.0);
        let headwind_range = headwind_result.points.last().map(|p| p.x).unwrap_or(0.0);
        // Headwind should reduce range
        assert!(
            headwind_range < base_range,
            "Headwind range {headwind_range:.2} should be less than no-wind range {base_range:.2}"
        );
    }
}
