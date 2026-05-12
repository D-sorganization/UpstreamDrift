//! `AerodynamicsEngine` — unified force orchestrator for the Rust kernel.
//!
//! Mirrors `src/shared/python/physics/aerodynamics/_engine.py` so the
//! RK4 inner loop can compute the full aerodynamic acceleration without
//! crossing the Python/C boundary.
//!
//! Composition (orthogonal, all toggleable):
//!   F_total = (drag_enabled ? drag(v - wind)) +
//!             (lift_enabled ? lift(v - wind, spin)) +
//!             (magnus_enabled ? magnus(v - wind, spin))
//!
//! Wind is supplied by an optional `WindModel`. When `None` the relative
//! velocity equals the ball velocity (no wind).
//!
//! # Design by Contract
//! - `velocity`, `spin`, `position` components must be finite.
//! - `mass` (for `acceleration`) must be finite and positive.
//! - `t` must be finite.
//! - Returned forces are finite.

use serde::{Deserialize, Serialize};
use tools_core::Vector3;

use crate::aerodynamics::{
    compute_drag, compute_lift, compute_magnus, AeroBallProperties, AeroForces, AirProperties,
};
use crate::wind::WindModel;

/// Aerodynamics engine configuration toggles.
///
/// Each individual toggle is gated by `enabled` (master switch),
/// matching the Python `AerodynamicsConfig.is_*_active()` semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct AeroEngineConfig {
    /// Master switch — turns off all aerodynamic forces when false.
    pub enabled: bool,
    pub drag_enabled: bool,
    pub lift_enabled: bool,
    pub magnus_enabled: bool,
}

impl Default for AeroEngineConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            drag_enabled: true,
            lift_enabled: true,
            magnus_enabled: true,
        }
    }
}

impl AeroEngineConfig {
    pub fn drag_active(&self) -> bool {
        self.enabled && self.drag_enabled
    }
    pub fn lift_active(&self) -> bool {
        self.enabled && self.lift_enabled
    }
    pub fn magnus_active(&self) -> bool {
        self.enabled && self.magnus_enabled
    }
}

/// `AerodynamicsEngine` — combines drag/lift/magnus + optional wind.
///
/// The engine owns its ball/air parameters by value; callers that want
/// to change parameters mid-simulation should build a fresh engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct AerodynamicsEngine {
    pub config: AeroEngineConfig,
    pub ball: AeroBallProperties,
    pub air: AirProperties,
    pub wind: Option<WindModel>,
}

impl AerodynamicsEngine {
    /// Construct an engine, validating ball and air properties up-front.
    pub fn new(
        config: AeroEngineConfig,
        ball: AeroBallProperties,
        air: AirProperties,
        wind: Option<WindModel>,
    ) -> Result<Self, String> {
        ball.validate()?;
        air.validate()?;
        Ok(Self {
            config,
            ball,
            air,
            wind,
        })
    }

    /// Compute the full force breakdown (drag/lift/magnus).
    ///
    /// `position` and `t` are used only when a `WindModel` is attached.
    #[must_use]
    pub fn compute_forces(
        &self,
        velocity: &Vector3,
        spin: &Vector3,
        t: f64,
        position: &Vector3,
    ) -> AeroForces {
        debug_assert!(velocity.x.is_finite() && velocity.y.is_finite() && velocity.z.is_finite());
        debug_assert!(spin.x.is_finite() && spin.y.is_finite() && spin.z.is_finite());
        debug_assert!(t.is_finite());

        let rel_velocity = match &self.wind {
            Some(wm) => {
                let w = wm.wind_at(t, position);
                Vector3::new(velocity.x - w.x, velocity.y - w.y, velocity.z - w.z)
            }
            None => *velocity,
        };

        let drag = if self.config.drag_active() {
            compute_drag(&rel_velocity, &self.ball, &self.air)
        } else {
            Vector3::zero()
        };

        let lift = if self.config.lift_active() {
            compute_lift(&rel_velocity, spin, &self.ball, &self.air)
        } else {
            Vector3::zero()
        };

        let magnus = if self.config.magnus_active() {
            compute_magnus(&rel_velocity, spin, &self.ball, &self.air)
        } else {
            Vector3::zero()
        };

        AeroForces { drag, lift, magnus }
    }

    /// Sum of drag + lift + magnus.
    #[must_use]
    pub fn compute_total_force(
        &self,
        velocity: &Vector3,
        spin: &Vector3,
        t: f64,
        position: &Vector3,
    ) -> Vector3 {
        let f = self.compute_forces(velocity, spin, t, position);
        Vector3::new(
            f.drag.x + f.lift.x + f.magnus.x,
            f.drag.y + f.lift.y + f.magnus.y,
            f.drag.z + f.lift.z + f.magnus.z,
        )
    }

    /// Acceleration F_total / mass.
    ///
    /// # Preconditions
    /// - `mass` must be finite and strictly positive.
    #[must_use]
    pub fn compute_acceleration(
        &self,
        velocity: &Vector3,
        spin: &Vector3,
        mass: f64,
        t: f64,
        position: &Vector3,
    ) -> Vector3 {
        assert!(
            mass.is_finite() && mass > 0.0,
            "DbC: mass must be finite and positive"
        );
        let total = self.compute_total_force(velocity, spin, t, position);
        Vector3::new(total.x / mass, total.y / mass, total.z / mass)
    }
}

// ── Python bindings ──────────────────────────────────────────────────────────

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl AeroEngineConfig {
    #[new]
    #[pyo3(signature = (enabled=true, drag_enabled=true, lift_enabled=true, magnus_enabled=true))]
    fn py_new(enabled: bool, drag_enabled: bool, lift_enabled: bool, magnus_enabled: bool) -> Self {
        Self {
            enabled,
            drag_enabled,
            lift_enabled,
            magnus_enabled,
        }
    }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl AerodynamicsEngine {
    #[new]
    #[pyo3(signature = (config, ball, air, wind=None))]
    fn py_new(
        config: AeroEngineConfig,
        ball: AeroBallProperties,
        air: AirProperties,
        wind: Option<WindModel>,
    ) -> pyo3::PyResult<Self> {
        Self::new(config, ball, air, wind).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Compute the full aerodynamic force breakdown.
    ///
    /// Returns `AeroForces` (drag/lift/magnus vectors as length-3 lists via
    /// the existing getters).
    #[pyo3(name = "compute_forces", signature = (velocity, spin, t=0.0, position=[0.0, 0.0, 0.0]))]
    fn py_compute_forces(
        &self,
        velocity: [f64; 3],
        spin: [f64; 3],
        t: f64,
        position: [f64; 3],
    ) -> AeroForces {
        let v = Vector3::new(velocity[0], velocity[1], velocity[2]);
        let s = Vector3::new(spin[0], spin[1], spin[2]);
        let p = Vector3::new(position[0], position[1], position[2]);
        self.compute_forces(&v, &s, t, &p)
    }

    /// Compute total aerodynamic force as `[fx, fy, fz]`.
    #[pyo3(name = "compute_total_force", signature = (velocity, spin, t=0.0, position=[0.0, 0.0, 0.0]))]
    fn py_compute_total_force(
        &self,
        velocity: [f64; 3],
        spin: [f64; 3],
        t: f64,
        position: [f64; 3],
    ) -> [f64; 3] {
        let v = Vector3::new(velocity[0], velocity[1], velocity[2]);
        let s = Vector3::new(spin[0], spin[1], spin[2]);
        let p = Vector3::new(position[0], position[1], position[2]);
        let f = self.compute_total_force(&v, &s, t, &p);
        [f.x, f.y, f.z]
    }

    /// Compute acceleration as `[ax, ay, az]`.
    #[pyo3(name = "compute_acceleration", signature = (velocity, spin, mass, t=0.0, position=[0.0, 0.0, 0.0]))]
    fn py_compute_acceleration(
        &self,
        velocity: [f64; 3],
        spin: [f64; 3],
        mass: f64,
        t: f64,
        position: [f64; 3],
    ) -> pyo3::PyResult<[f64; 3]> {
        if !mass.is_finite() || mass <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mass must be finite and positive",
            ));
        }
        let v = Vector3::new(velocity[0], velocity[1], velocity[2]);
        let s = Vector3::new(spin[0], spin[1], spin[2]);
        let p = Vector3::new(position[0], position[1], position[2]);
        let a = self.compute_acceleration(&v, &s, mass, t, &p);
        Ok([a.x, a.y, a.z])
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wind::{TurbulenceModel, WindConfig};

    fn make_engine(wind: Option<WindModel>) -> AerodynamicsEngine {
        AerodynamicsEngine::new(
            AeroEngineConfig::default(),
            AeroBallProperties::default(),
            AirProperties::default(),
            wind,
        )
        .unwrap()
    }

    #[test]
    fn test_default_engine_drag_opposes_velocity() {
        let engine = make_engine(None);
        let v = Vector3::new(50.0, 0.0, 0.0);
        let s = Vector3::zero();
        let f = engine.compute_total_force(&v, &s, 0.0, &Vector3::zero());
        assert!(f.x < 0.0);
    }

    #[test]
    fn test_master_disable_zeroes_total() {
        let cfg = AeroEngineConfig {
            enabled: false,
            ..AeroEngineConfig::default()
        };
        let engine = AerodynamicsEngine::new(
            cfg,
            AeroBallProperties::default(),
            AirProperties::default(),
            None,
        )
        .unwrap();
        let v = Vector3::new(50.0, 0.0, 0.0);
        let s = Vector3::new(0.0, 300.0, 0.0);
        let f = engine.compute_total_force(&v, &s, 0.0, &Vector3::zero());
        assert!(f.x.abs() < 1e-12 && f.y.abs() < 1e-12 && f.z.abs() < 1e-12);
    }

    #[test]
    fn test_drag_disabled_zeroes_drag_only() {
        let cfg = AeroEngineConfig {
            drag_enabled: false,
            ..AeroEngineConfig::default()
        };
        let engine = AerodynamicsEngine::new(
            cfg,
            AeroBallProperties::default(),
            AirProperties::default(),
            None,
        )
        .unwrap();
        let v = Vector3::new(50.0, 0.0, 0.0);
        let s = Vector3::new(0.0, 300.0, 0.0);
        let parts = engine.compute_forces(&v, &s, 0.0, &Vector3::zero());
        assert!(parts.drag.x.abs() < 1e-12);
        // lift/magnus should still produce nonzero magnitude
        let lm = parts.lift.x * parts.lift.x
            + parts.lift.y * parts.lift.y
            + parts.lift.z * parts.lift.z
            + parts.magnus.x * parts.magnus.x
            + parts.magnus.y * parts.magnus.y
            + parts.magnus.z * parts.magnus.z;
        assert!(lm > 0.0);
    }

    #[test]
    fn test_wind_subtraction_changes_drag_direction() {
        // Headwind larger than ball speed flips relative velocity → drag flips sign.
        let cfg = WindConfig {
            base_velocity: Vector3::new(100.0, 0.0, 0.0),
            ..WindConfig::default()
        };
        let wind = WindModel::new(cfg, TurbulenceModel::disabled()).unwrap();
        let engine = make_engine(Some(wind));
        let v = Vector3::new(20.0, 0.0, 0.0);
        let s = Vector3::zero();
        let f = engine.compute_total_force(&v, &s, 0.0, &Vector3::zero());
        // v - wind = -80 → drag opposes that → positive x
        assert!(
            f.x > 0.0,
            "expected positive drag with strong headwind, got {}",
            f.x
        );
    }

    #[test]
    fn test_acceleration_scales_inverse_mass() {
        let engine = make_engine(None);
        let v = Vector3::new(50.0, 0.0, 0.0);
        let s = Vector3::zero();
        let a1 = engine.compute_acceleration(&v, &s, 0.04593, 0.0, &Vector3::zero());
        let a2 = engine.compute_acceleration(&v, &s, 0.09186, 0.0, &Vector3::zero());
        // Doubling mass halves acceleration
        assert!(
            (a1.x - 2.0 * a2.x).abs() < 1e-9,
            "a1.x={} a2.x={}",
            a1.x,
            a2.x
        );
    }

    #[test]
    #[should_panic(expected = "DbC: mass must be finite and positive")]
    fn test_acceleration_rejects_zero_mass() {
        let engine = make_engine(None);
        let v = Vector3::new(50.0, 0.0, 0.0);
        let s = Vector3::zero();
        let _ = engine.compute_acceleration(&v, &s, 0.0, 0.0, &Vector3::zero());
    }
}
