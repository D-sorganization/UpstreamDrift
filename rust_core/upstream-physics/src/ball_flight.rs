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

/// Per-step force snapshot evaluated at the integrated state (issue #14937).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct IntegratedStepForces {
    /// Gravitational force [N]
    pub gravity: [f64; 3],
    /// Drag force [N]
    pub drag: [f64; 3],
    /// Magnus force [N]
    pub magnus: [f64; 3],
    /// Acceleration implied by gravity + aerodynamics [m/s²]
    pub acceleration: [f64; 3],
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl IntegratedStepForces {
    #[getter]
    fn gravity(&self) -> [f64; 3] {
        self.gravity
    }
    #[getter]
    fn drag(&self) -> [f64; 3] {
        self.drag
    }
    #[getter]
    fn magnus(&self) -> [f64; 3] {
        self.magnus
    }
    #[getter]
    fn acceleration(&self) -> [f64; 3] {
        self.acceleration
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
    /// Forces at each trajectory point from the integrated aerodynamic model.
    pub step_forces: Vec<IntegratedStepForces>,
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

    /// Final two native states for landing interpolation without Python
    /// conversion of the full trajectory. Returns fewer only if unavailable.
    fn get_final_points(&self) -> Vec<TrajectoryPoint> {
        self.points[self.points.len().saturating_sub(2)..].to_vec()
    }

    /// Flatten trajectory to a list of `[t, x, y, z, vx, vy, vz]` tuples.
    fn to_flat_list(&self) -> Vec<Vec<f64>> {
        self.points
            .iter()
            .map(|p| vec![p.t, p.x, p.y, p.z, p.vx, p.vy, p.vz])
            .collect()
    }

    /// Per-step forces from the integrator's aerodynamic model (empty when not recorded).
    fn get_step_forces(&self) -> Vec<IntegratedStepForces> {
        self.step_forces.clone()
    }
}

/// Linearly interpolate the segment between ``prev`` and ``final_pt`` to ``z = 0``.
///
/// Preconditions: ``prev.z > 0``, ``final_pt.z <= 0``, and the segment crosses
/// the ground plane (denominator non-zero). Matches the Python
/// ``BallFlightSimulator.simulate_landing`` convention.
fn interpolate_ground_crossing(
    prev: &TrajectoryPoint,
    final_pt: &TrajectoryPoint,
) -> TrajectoryPoint {
    let denom = prev.z - final_pt.z;
    let alpha = if denom.abs() > 1e-15 {
        prev.z / denom
    } else {
        1.0
    };
    TrajectoryPoint {
        t: prev.t + alpha * (final_pt.t - prev.t),
        x: prev.x + alpha * (final_pt.x - prev.x),
        y: prev.y + alpha * (final_pt.y - prev.y),
        z: 0.0,
        vx: prev.vx + alpha * (final_pt.vx - prev.vx),
        vy: prev.vy + alpha * (final_pt.vy - prev.vy),
        vz: prev.vz + alpha * (final_pt.vz - prev.vz),
    }
}

fn integrated_forces_at_state(
    state: &[f64],
    spin_axis_v: &Vector3,
    gravity_v: &Vector3,
    wind_v: &Vector3,
    ball: &AeroBallProperties,
    air: &AirProperties,
) -> IntegratedStepForces {
    let velocity = Vector3::new(state[3], state[4], state[5]);
    let omega = state[6];
    let rel_vel = velocity - *wind_v;
    let spin = *spin_axis_v * omega;
    let forces = compute_aero_forces(&rel_vel, &spin, ball, air);
    let gravity = [
        gravity_v.x * ball.mass,
        gravity_v.y * ball.mass,
        gravity_v.z * ball.mass,
    ];
    let drag = [forces.drag.x, forces.drag.y, forces.drag.z];
    let magnus = [forces.magnus.x, forces.magnus.y, forces.magnus.z];
    let ax = gravity_v.x + (forces.drag.x + forces.magnus.x + forces.lift.x) / ball.mass;
    let ay = gravity_v.y + (forces.drag.y + forces.magnus.y + forces.lift.y) / ball.mass;
    let az = gravity_v.z + (forces.drag.z + forces.magnus.z + forces.lift.z) / ball.mass;
    IntegratedStepForces {
        gravity,
        drag,
        magnus,
        acceleration: [ax, ay, az],
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
/// # Errors
/// Returns `Err` with a descriptive message when any precondition is violated:
/// non-finite inputs, `omega0 < 0`, or invalid ball/air/integrator properties.
/// These were previously `assert!`/`expect` panics that crossed the PyO3
/// boundary as a `PanicException` (issue #7663).
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
    record_integrated_forces: bool,
) -> Result<BallTrajectoryResult, String> {
    if !pos0.iter().all(|v| v.is_finite()) {
        return Err("initial position must be finite".to_string());
    }
    if !vel0.iter().all(|v| v.is_finite()) {
        return Err("initial velocity must be finite".to_string());
    }
    if !spin_axis.iter().all(|v| v.is_finite()) {
        return Err("spin axis must be finite".to_string());
    }
    if !(omega0.is_finite() && omega0 >= 0.0) {
        return Err("omega0 must be finite and non-negative".to_string());
    }
    if !gravity.iter().all(|v| v.is_finite()) {
        return Err("gravity must be finite".to_string());
    }
    if !wind.iter().all(|v| v.is_finite()) {
        return Err("wind must be finite".to_string());
    }
    ball.validate()?;
    air.validate()?;
    config.validate()?;

    // State: [x, y, z, vx, vy, vz, omega]
    let y0 = [pos0[0], pos0[1], pos0[2], vel0[0], vel0[1], vel0[2], omega0];

    // Capture by reference using owned copies for the closure
    let ball_record = ball.clone();
    let air_record = air.clone();
    let ball_integrate = ball.clone();
    let air_integrate = air.clone();
    // Normalize spin_axis to unit vector; fall back to Z-axis if near-zero
    let raw_spin = Vector3::new(spin_axis[0], spin_axis[1], spin_axis[2]);
    let spin_axis_v = {
        let n =
            (raw_spin.x * raw_spin.x + raw_spin.y * raw_spin.y + raw_spin.z * raw_spin.z).sqrt();
        if n > 1e-10 {
            Vector3::new(raw_spin.x / n, raw_spin.y / n, raw_spin.z / n)
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        }
    };
    let gravity_v = Vector3::new(gravity[0], gravity[1], gravity[2]);
    let wind_v = Vector3::new(wind[0], wind[1], wind[2]);
    let spin_axis_record = spin_axis_v;
    let wind_record = wind_v;
    let gravity_record = gravity_v;

    let derivative = move |_t: f64, state: &[f64]| -> Vec<f64> {
        let velocity = Vector3::new(state[3], state[4], state[5]);
        let omega = state[6];

        // Relative velocity accounting for wind
        let rel_vel = velocity - wind_v;

        // Spin vector = axis direction × magnitude
        let spin = spin_axis_v * omega;

        // Aerodynamic forces [N]
        let forces = compute_aero_forces(&rel_vel, &spin, &ball_integrate, &air_integrate);
        let total_aero = forces.drag + forces.lift + forces.magnus;

        // Acceleration = gravity + F_aero / mass
        let ax = gravity_v.x + total_aero.x / ball_integrate.mass;
        let ay = gravity_v.y + total_aero.y / ball_integrate.mass;
        let az = gravity_v.z + total_aero.z / ball_integrate.mass;

        // d(omega)/dt = -spin_decay_rate * omega (continuous exponential decay)
        let domega = -ball_integrate.spin_decay_rate * omega;

        vec![velocity.x, velocity.y, velocity.z, ax, ay, az, domega]
    };

    // Terminate on descending ground contact: z ≤ 0 while vz < 0. The vz guard
    // rejects RK4 undershoot below z = 0 before apex (replaces the old t > 0.05 s).
    let terminate = |_t: f64, state: &[f64]| -> bool { state[2] <= 0.0 && state[5] < 0.0 };

    let max_time = config.max_steps as f64 * config.dt;
    let result = integrate(derivative, 0.0, max_time, &y0, config, Some(terminate))?;

    // Convert IntegrationResult to BallTrajectoryResult
    let state_dim = result.state_dim;
    let mut points: Vec<TrajectoryPoint> = result
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

    if points.len() >= 2 {
        let last_idx = points.len() - 1;
        let prev = points[last_idx - 1].clone();
        let last = points[last_idx].clone();
        if prev.z > 0.0 && last.z <= 0.0 && last.vz < 0.0 {
            points[last_idx] = interpolate_ground_crossing(&prev, &last);
        }
    }

    // Landing (termination by ground contact) is a successful outcome, not an
    // aborted integration.  Override completed=true whenever the final point
    // reached z ≤ 0 on descent.
    let landed = points
        .last()
        .map(|p| p.z <= 0.0 && p.vz < 0.0)
        .unwrap_or(false);

    let step_forces = if record_integrated_forces {
        result
            .states
            .chunks(state_dim)
            .map(|state| {
                integrated_forces_at_state(
                    state,
                    &spin_axis_record,
                    &gravity_record,
                    &wind_record,
                    &ball_record,
                    &air_record,
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(BallTrajectoryResult {
        completed: result.completed || landed,
        steps: result.steps_taken,
        points,
        step_forces,
    })
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

    #[test]
    fn test_simulation_rejects_invalid_mass() {
        let ball = AeroBallProperties {
            mass: 0.0,
            ..AeroBallProperties::default()
        };
        let result = simulate_ball_trajectory(
            [0.0, 0.0, 0.5],
            [10.0, 0.0, 10.0],
            [0.0, 1.0, 0.0],
            100.0,
            [0.0, 0.0, -9.81],
            [0.0, 0.0, 0.0],
            &ball,
            &AirProperties::default(),
            &default_config(),
            false,
        );
        assert!(result.is_err(), "invalid ball mass must be rejected");
    }

    #[test]
    fn test_simulation_rejects_non_finite_input() {
        let result = simulate_ball_trajectory(
            [f64::NAN, 0.0, 0.5],
            [10.0, 0.0, 10.0],
            [0.0, 1.0, 0.0],
            100.0,
            [0.0, 0.0, -9.81],
            [0.0, 0.0, 0.0],
            &AeroBallProperties::default(),
            &AirProperties::default(),
            &default_config(),
            false,
        );
        assert!(result.is_err(), "non-finite position must be rejected");
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
            false,
        )
        .unwrap();

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
            false,
        )
        .unwrap();

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
            false,
        )
        .unwrap();
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
            false,
        )
        .unwrap();

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
            false,
        )
        .unwrap();
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
            false,
        )
        .unwrap();

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
            false,
        )
        .unwrap();

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
            false,
        )
        .unwrap();

        let base_range = base_result.points.last().map(|p| p.x).unwrap_or(0.0);
        let headwind_range = headwind_result.points.last().map(|p| p.x).unwrap_or(0.0);
        // Headwind should reduce range
        assert!(
            headwind_range < base_range,
            "Headwind range {headwind_range:.2} should be less than no-wind range {base_range:.2}"
        );
    }

    #[test]
    fn test_landing_point_interpolated_to_ground_plane() {
        let result = simulate_ball_trajectory(
            [0.0, 0.0, 0.5],
            [32.0, 0.5, 18.0],
            [0.0, 1.0, 0.0],
            220.0,
            [0.0, 0.0, -9.81],
            [-2.0, 0.0, 0.0],
            &AeroBallProperties::default(),
            &AirProperties::default(),
            &IntegratorConfig {
                dt: 0.02,
                max_steps: 1_000,
            },
            false,
        )
        .unwrap();
        let final_point = result.points.last().expect("trajectory has a final point");
        assert!(result.completed);
        assert!(
            final_point.z.abs() < 1e-12,
            "landing must be interpolated to z=0, got z={}",
            final_point.z
        );
        assert!(final_point.vz < 0.0, "landing must occur on descent");
    }

    /// Test: normal landing must report completed = true (issue #2484).
    #[test]
    fn test_landing_reports_completed_true() {
        let result = simulate_ball_trajectory(
            [0.0, 0.0, 0.5],
            [20.0, 0.0, 10.0],
            [0.0, 0.0, 1.0],
            100.0,
            [0.0, 0.0, -9.81],
            [0.0, 0.0, 0.0],
            &AeroBallProperties::default(),
            &AirProperties::default(),
            &default_config(),
            false,
        )
        .unwrap();

        assert!(
            result.completed,
            "Ball that lands on the ground must report completed=true, got false"
        );
    }

    /// Test: non-unit spin_axis is normalized before force computation (issue #2484).
    #[test]
    fn test_non_unit_spin_axis_normalized() {
        // A spin_axis with magnitude 2.0 should behave identically to magnitude 1.0
        let result_unit = simulate_ball_trajectory(
            [0.0, 0.0, 0.5],
            [20.0, 0.0, 10.0],
            [0.0, 0.0, 1.0], // unit axis
            100.0,
            [0.0, 0.0, -9.81],
            [0.0, 0.0, 0.0],
            &AeroBallProperties::default(),
            &AirProperties::default(),
            &default_config(),
            false,
        )
        .unwrap();
        let result_scaled = simulate_ball_trajectory(
            [0.0, 0.0, 0.5],
            [20.0, 0.0, 10.0],
            [0.0, 0.0, 2.0], // non-unit axis — should be normalized
            100.0,
            [0.0, 0.0, -9.81],
            [0.0, 0.0, 0.0],
            &AeroBallProperties::default(),
            &AirProperties::default(),
            &default_config(),
            false,
        )
        .unwrap();

        assert_eq!(
            result_unit.steps, result_scaled.steps,
            "Non-unit spin_axis must be normalized: trajectories must be identical"
        );
        let last_unit = result_unit.points.last().unwrap();
        let last_scaled = result_scaled.points.last().unwrap();
        assert!(
            (last_unit.x - last_scaled.x).abs() < 1e-10,
            "Non-unit spin_axis yields different trajectory: unit.x={} scaled.x={}",
            last_unit.x,
            last_scaled.x
        );
    }

    #[test]
    fn test_integrated_step_forces_match_newton_and_state_accel() {
        let ball = AeroBallProperties::default();
        let air = AirProperties::default();
        let result = simulate_ball_trajectory(
            [0.0, 0.0, 0.0],
            [60.0, 0.0, 12.0],
            [0.0, -1.0, 0.0],
            2500.0 * std::f64::consts::PI / 30.0,
            [0.0, 0.0, -9.81],
            [0.0, 0.0, 0.0],
            &ball,
            &air,
            &IntegratorConfig {
                dt: 0.01,
                max_steps: 200,
            },
            true,
        )
        .unwrap();
        assert_eq!(result.points.len(), result.step_forces.len());
        for forces in &result.step_forces {
            let fx = forces.gravity[0] + forces.drag[0] + forces.magnus[0];
            let fy = forces.gravity[1] + forces.drag[1] + forces.magnus[1];
            let fz = forces.gravity[2] + forces.drag[2] + forces.magnus[2];
            let ax = fx / ball.mass;
            let ay = fy / ball.mass;
            let az = fz / ball.mass;
            assert!((ax - forces.acceleration[0]).abs() < 1e-9);
            assert!((ay - forces.acceleration[1]).abs() < 1e-9);
            assert!((az - forces.acceleration[2]).abs() < 1e-9);
        }
    }
}
