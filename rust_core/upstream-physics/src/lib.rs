//! # upstream-physics — UpstreamDrift Physics Kernels
//!
//! Provides high-performance, native Rust physics calculations
//! for the UpstreamDrift simulation platform.
//!
//! ## Modules
//!
//! - `rk4`: Generic RK4 numerical integrator with adaptive step control
//! - `contact`: Impact dynamics and ground reaction forces
//! - `swing_plane`: Swing plane fitting and analysis
//!
//! ## Design Principles
//!
//! - **TDD**: Tests written before implementation
//! - **DbC**: Precondition validation via `debug_assert!`
//! - **DRY**: Consumes `tools-core` for math primitives

pub mod aerodynamics;
pub mod ball_flight;
pub mod contact;
pub mod rk4;
pub mod swing_plane;

// Re-export primary types from tools-core for convenience.
pub use tools_core::Vector3;

// ── Python bindings (feature-gated) ──────────────────────────────────────────

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Simulate a ball trajectory from Python.
///
/// Returns a `BallTrajectoryResult` with all trajectory points.
///
/// # Arguments
/// - `pos0`: `[x, y, z]` initial position [m]
/// - `vel0`: `[vx, vy, vz]` initial velocity [m/s]
/// - `spin_axis`: unit vector for spin axis direction
/// - `omega0`: initial spin rate [rad/s]
/// - `gravity`: gravity vector (e.g. `[0, 0, -9.81]`)
/// - `wind`: wind velocity [m/s]
/// - `ball`: `AeroBallProperties` instance
/// - `air`: `AirProperties` instance
/// - `config`: `IntegratorConfig` instance
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (pos0, vel0, spin_axis, omega0, gravity, wind, ball, air, config))]
fn simulate_ball_trajectory_py(
    pos0: [f64; 3],
    vel0: [f64; 3],
    spin_axis: [f64; 3],
    omega0: f64,
    gravity: [f64; 3],
    wind: [f64; 3],
    ball: aerodynamics::AeroBallProperties,
    air: aerodynamics::AirProperties,
    config: rk4::IntegratorConfig,
) -> ball_flight::BallTrajectoryResult {
    ball_flight::simulate_ball_trajectory(
        pos0, vel0, spin_axis, omega0, gravity, wind, &ball, &air, &config,
    )
}

#[cfg(feature = "python")]
#[pymodule]
fn upstream_physics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // RK4 Integrator
    m.add_class::<rk4::IntegratorConfig>()?;
    m.add_class::<rk4::IntegrationResult>()?;

    // Contact model
    m.add_class::<contact::ContactParameters>()?;
    m.add_class::<contact::ContactResult>()?;

    // Swing plane
    m.add_class::<swing_plane::SwingPlaneResult>()?;

    // Aerodynamics
    m.add_class::<aerodynamics::AirProperties>()?;
    m.add_class::<aerodynamics::AeroBallProperties>()?;
    m.add_class::<aerodynamics::AeroForces>()?;

    // Ball flight trajectory simulation (wires aerodynamics + RK4)
    m.add_class::<ball_flight::TrajectoryPoint>()?;
    m.add_class::<ball_flight::BallTrajectoryResult>()?;
    m.add_function(pyo3::wrap_pyfunction!(simulate_ball_trajectory_py, m)?)?;

    Ok(())
}

// ── WASM bindings (feature-gated) ────────────────────────────────────────────

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Create a default IntegratorConfig for browser-side physics (WASM).
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "createIntegratorConfig")]
pub fn create_integrator_config(dt: f64, max_steps: u32) -> Result<JsValue, JsValue> {
    let config = rk4::IntegratorConfig {
        dt,
        max_steps: max_steps as usize,
    };
    serde_wasm_bindgen::to_value(&config)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize IntegratorConfig: {e}")))
}

/// Create default ContactParameters for browser-side physics (WASM).
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "createContactParameters")]
pub fn create_contact_parameters(cor: f64, friction: f64) -> Result<JsValue, JsValue> {
    let params = contact::ContactParameters {
        cor,
        friction,
        ..Default::default()
    };
    serde_wasm_bindgen::to_value(&params)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize ContactParameters: {e}")))
}

/// Compute aerodynamic forces for browser-side physics (WASM).
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "computeAeroForces")]
pub fn wasm_compute_aero_forces(
    vx: f64,
    vy: f64,
    vz: f64,
    sx: f64,
    sy: f64,
    sz: f64,
    air_density: f64,
) -> Result<JsValue, JsValue> {
    let velocity = Vector3::new(vx, vy, vz);
    let spin = Vector3::new(sx, sy, sz);
    let ball = aerodynamics::AeroBallProperties::default();
    let air = aerodynamics::AirProperties {
        density: air_density,
        ..Default::default()
    };
    let forces = aerodynamics::compute_aero_forces(&velocity, &spin, &ball, &air);
    serde_wasm_bindgen::to_value(&forces)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize AeroForces: {e}")))
}
