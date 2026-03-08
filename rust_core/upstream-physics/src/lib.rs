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

pub mod contact;
pub mod rk4;
pub mod swing_plane;

// Re-export primary types from tools-core for convenience.
pub use tools_core::Vector3;

// ── Python bindings (feature-gated) ──────────────────────────────────────────

#[cfg(feature = "python")]
use pyo3::prelude::*;

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

    Ok(())
}
