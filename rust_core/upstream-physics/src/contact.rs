//! Contact dynamics for ball-surface impacts.
//!
//! Implements a coefficient-of-restitution (COR) model with
//! spin transfer for golf ball impacts on various surfaces.
//!
//! # Design by Contract
//! - COR must be in [0, 1]
//! - Friction coefficient must be non-negative
//! - Incoming speed must be positive

use serde::{Deserialize, Serialize};
use tools_core::Vector3;

/// Parameters for a contact surface.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct ContactParameters {
    /// Coefficient of restitution (0 = perfectly inelastic, 1 = perfectly elastic).
    pub cor: f64,
    /// Kinetic friction coefficient.
    pub friction: f64,
    /// Surface normal vector (unit vector pointing away from surface).
    pub normal: Vector3,
}

impl Default for ContactParameters {
    fn default() -> Self {
        Self {
            cor: 0.78,                           // Typical golf green COR
            friction: 0.4,                       // Typical grass friction
            normal: Vector3::new(0.0, 1.0, 0.0), // Flat ground (Y-up)
        }
    }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl ContactParameters {
    #[new]
    #[pyo3(signature = (cor=0.78, friction=0.4))]
    fn py_new(cor: f64, friction: f64) -> Self {
        Self {
            cor,
            friction,
            normal: Vector3::new(0.0, 1.0, 0.0),
        }
    }
}

/// Result of a contact calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct ContactResult {
    /// Post-impact velocity.
    pub velocity: Vector3,
    /// Post-impact spin rate [rad/s].
    pub spin_rate: f64,
    /// Energy lost during impact [J].
    pub energy_lost: f64,
    /// Whether the ball is rolling (speed below threshold).
    pub is_rolling: bool,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl ContactResult {
    #[getter]
    fn speed(&self) -> f64 {
        self.velocity.magnitude()
    }
}

/// Calculate the post-impact state of a ball hitting a surface.
///
/// Uses a simple COR model:
/// - Normal component is reflected and scaled by COR
/// - Tangential component is reduced by friction
/// - Spin is modified based on surface interaction
///
/// # Arguments
/// * `velocity` - Incoming velocity vector [m/s]
/// * `spin_rate` - Incoming spin rate [rad/s]
/// * `params` - Surface contact parameters
///
/// # Returns
/// `ContactResult` with post-impact state
pub fn calculate_impact(
    velocity: &Vector3,
    spin_rate: f64,
    params: &ContactParameters,
) -> ContactResult {
    let speed = velocity.magnitude();

    // DbC: validate inputs
    debug_assert!(
        params.cor >= 0.0 && params.cor <= 1.0,
        "COR must be in [0, 1]"
    );
    debug_assert!(params.friction >= 0.0, "Friction must be non-negative");

    // Rolling threshold
    if speed < 0.5 {
        return ContactResult {
            velocity: *velocity,
            spin_rate: 0.0,
            energy_lost: 0.0,
            is_rolling: true,
        };
    }

    let normal = &params.normal;
    let normal_mag = normal.magnitude();
    if normal_mag < 1e-10 {
        // Degenerate normal, return unchanged
        return ContactResult {
            velocity: *velocity,
            spin_rate,
            energy_lost: 0.0,
            is_rolling: false,
        };
    }

    // Normalize the surface normal
    let n = Vector3::new(
        normal.x / normal_mag,
        normal.y / normal_mag,
        normal.z / normal_mag,
    );

    // Decompose velocity into normal and tangential components
    let v_dot_n = velocity.x * n.x + velocity.y * n.y + velocity.z * n.z;

    let v_normal = Vector3::new(v_dot_n * n.x, v_dot_n * n.y, v_dot_n * n.z);

    let v_tangential = Vector3::new(
        velocity.x - v_normal.x,
        velocity.y - v_normal.y,
        velocity.z - v_normal.z,
    );

    // Apply COR to normal component (reflect)
    let v_normal_out = Vector3::new(
        -params.cor * v_normal.x,
        -params.cor * v_normal.y,
        -params.cor * v_normal.z,
    );

    // Apply friction to tangential component
    let friction_factor = (1.0 - params.friction).max(0.0);
    let v_tangential_out = Vector3::new(
        friction_factor * v_tangential.x,
        friction_factor * v_tangential.y,
        friction_factor * v_tangential.z,
    );

    // Combine
    let v_out = Vector3::new(
        v_normal_out.x + v_tangential_out.x,
        v_normal_out.y + v_tangential_out.y,
        v_normal_out.z + v_tangential_out.z,
    );

    // Energy calculation
    let ke_in = 0.5 * speed * speed; // Per unit mass
    let speed_out = v_out.magnitude();
    let ke_out = 0.5 * speed_out * speed_out;
    let energy_lost = ke_in - ke_out;

    // Spin modification: reduce by COR-weighted factor
    let spin_out = spin_rate * params.cor * 0.9;

    ContactResult {
        velocity: v_out,
        spin_rate: spin_out,
        energy_lost: energy_lost.max(0.0),
        is_rolling: speed_out < 0.5,
    }
}

// ── Tests (TDD) ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Perfectly elastic bounce (COR=1) conserves normal speed.
    #[test]
    fn test_elastic_bounce() {
        let params = ContactParameters {
            cor: 1.0,
            friction: 0.0,
            normal: Vector3::new(0.0, 1.0, 0.0),
        };

        let v_in = Vector3::new(0.0, -10.0, 0.0); // Straight down
        let result = calculate_impact(&v_in, 0.0, &params);

        // Should bounce back at same speed
        assert!(
            (result.velocity.y - 10.0).abs() < 1e-6,
            "Expected vy=10.0, got {}",
            result.velocity.y
        );
        assert!(result.energy_lost.abs() < 1e-6, "No energy should be lost");
    }

    /// Test 2: Perfectly inelastic impact (COR=0) kills normal velocity.
    #[test]
    fn test_inelastic_bounce() {
        let params = ContactParameters {
            cor: 0.0,
            friction: 0.0,
            normal: Vector3::new(0.0, 1.0, 0.0),
        };

        let v_in = Vector3::new(5.0, -10.0, 0.0);
        let result = calculate_impact(&v_in, 100.0, &params);

        // Normal component should be zero
        assert!(
            result.velocity.y.abs() < 1e-6,
            "Normal velocity should be zero"
        );
        // Tangential preserved (no friction)
        assert!(
            (result.velocity.x - 5.0).abs() < 1e-6,
            "Tangential velocity preserved"
        );
    }

    /// Test 3: Friction reduces tangential velocity.
    #[test]
    fn test_friction_reduces_tangential() {
        let params = ContactParameters {
            cor: 0.8,
            friction: 0.5,
            normal: Vector3::new(0.0, 1.0, 0.0),
        };

        let v_in = Vector3::new(10.0, -5.0, 0.0);
        let result = calculate_impact(&v_in, 0.0, &params);

        // Tangential should be reduced by friction_factor = 1 - 0.5 = 0.5
        assert!(
            (result.velocity.x - 5.0).abs() < 1e-6,
            "Expected vx=5.0, got {}",
            result.velocity.x
        );
    }

    /// Test 4: Typical golf green impact.
    #[test]
    fn test_typical_green_impact() {
        let params = ContactParameters::default(); // COR=0.78, friction=0.4

        let v_in = Vector3::new(15.0, -8.0, 0.0);
        let result = calculate_impact(&v_in, 3000.0, &params);

        // Post-impact should have reduced speed
        assert!(result.velocity.magnitude() < v_in.magnitude());
        // Energy should be lost
        assert!(result.energy_lost > 0.0);
        // Spin should decrease
        assert!(result.spin_rate < 3000.0);
    }

    /// Test 5: Very slow ball should be rolling.
    #[test]
    fn test_rolling_threshold() {
        let params = ContactParameters::default();

        let v_in = Vector3::new(0.3, -0.1, 0.0); // Below threshold
        let result = calculate_impact(&v_in, 100.0, &params);

        assert!(result.is_rolling, "Ball should be rolling at low speed");
    }

    /// Test 6: Energy conservation: lost energy is non-negative.
    #[test]
    fn test_energy_non_negative() {
        let params = ContactParameters {
            cor: 0.6,
            friction: 0.3,
            normal: Vector3::new(0.0, 1.0, 0.0),
        };

        let v_in = Vector3::new(20.0, -15.0, 5.0);
        let result = calculate_impact(&v_in, 5000.0, &params);

        assert!(
            result.energy_lost >= 0.0,
            "Energy lost must be non-negative"
        );
    }

    /// Test 7: Angled surface normal.
    #[test]
    fn test_angled_surface() {
        // 30-degree slope
        let angle = std::f64::consts::FRAC_PI_6;
        let params = ContactParameters {
            cor: 0.8,
            friction: 0.2,
            normal: Vector3::new(angle.sin(), angle.cos(), 0.0),
        };

        let v_in = Vector3::new(0.0, -10.0, 0.0);
        let result = calculate_impact(&v_in, 0.0, &params);

        // Ball should deflect sideways on angled surface
        assert!(
            result.velocity.x.abs() > 0.1,
            "Should deflect on angled surface"
        );
    }
}
