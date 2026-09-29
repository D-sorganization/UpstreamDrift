//! Wind model for golf-ball aerodynamics.
//!
//! Mirrors `src/shared/python/physics/aerodynamics/_wind.py` so the
//! Rust trajectory kernel can evaluate wind contributions inside the
//! RK4 inner loop without crossing the Python/C boundary.
//!
//! The model has three orthogonal contributions:
//! - constant base velocity,
//! - altitude-dependent gradient (wind shear),
//! - small-scale turbulence via a deterministic sum-of-sinusoids field.
//!
//! Random gusts (Poisson spawn with sinusoidal envelopes) are *not*
//! ported here because they require non-deterministic RNG state which
//! breaks parity testing. Callers that need gusts should continue to
//! use the Python facade with `gusts_enabled=False` in the Rust path
//! and add the gust contribution in Python on top of the Rust wind.
//!
//! # Design by Contract
//! - All public inputs must be finite.
//! - Turbulence coefficients/phases/frequencies are fixed-length arrays
//!   so the inner loop has no heap allocation.
//! - Output wind vector is always finite.

use serde::{Deserialize, Serialize};
use tools_core::Vector3;

/// Number of sinusoidal modes per axis used to model turbulence.
///
/// Matches the Python `TurbulenceModel` default of 10 modes.
pub const TURBULENCE_MODES: usize = 10;

/// Configuration for the wind model.
///
/// All fields are public so callers can construct a fully-typed config
/// from Python or unit tests. Validation runs in `validate()` and is
/// also invoked by every `pyclass` constructor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct WindConfig {
    /// Constant base wind velocity [m/s] (world frame).
    pub base_velocity: Vector3,
    /// If true, scale `base_velocity` by an altitude-dependent factor.
    pub altitude_gradient: bool,
    /// Relative wind speed increase per 10 m of altitude (e.g. 0.05 → +5%).
    pub gradient_factor: f64,
    /// Small-scale turbulence intensity (set 0.0 to disable).
    pub turbulence_intensity: f64,
}

impl Default for WindConfig {
    fn default() -> Self {
        Self {
            base_velocity: Vector3::zero(),
            altitude_gradient: false,
            gradient_factor: 0.05,
            turbulence_intensity: 0.0,
        }
    }
}

impl WindConfig {
    /// Validate public fields in all build modes.
    pub fn validate(&self) -> Result<(), String> {
        if !self.base_velocity.x.is_finite()
            || !self.base_velocity.y.is_finite()
            || !self.base_velocity.z.is_finite()
        {
            return Err("WindConfig.base_velocity must be finite".to_string());
        }
        if !self.gradient_factor.is_finite() {
            return Err("WindConfig.gradient_factor must be finite".to_string());
        }
        if !self.turbulence_intensity.is_finite() || self.turbulence_intensity < 0.0 {
            return Err(format!(
                "WindConfig.turbulence_intensity must be finite and non-negative, got {}",
                self.turbulence_intensity,
            ));
        }
        Ok(())
    }
}

/// Deterministic small-scale turbulence model.
///
/// Identical structure to the Python `TurbulenceModel`: each spatial
/// axis is a sum of `TURBULENCE_MODES` sinusoids in time. Coefficients,
/// phases and frequencies are stored as fixed-size arrays so callers
/// can construct a model with explicit reproducible parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurbulenceModel {
    pub intensity: f64,
    pub coeffs: [[f64; TURBULENCE_MODES]; 3],
    pub phases: [[f64; TURBULENCE_MODES]; 3],
    pub freqs: [f64; TURBULENCE_MODES],
}

impl TurbulenceModel {
    /// Create a zero-intensity (no-op) turbulence model.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            intensity: 0.0,
            coeffs: [[0.0; TURBULENCE_MODES]; 3],
            phases: [[0.0; TURBULENCE_MODES]; 3],
            freqs: [1.0; TURBULENCE_MODES],
        }
    }

    /// Validate public fields.
    pub fn validate(&self) -> Result<(), String> {
        if !self.intensity.is_finite() || self.intensity < 0.0 {
            return Err(format!(
                "TurbulenceModel.intensity must be finite and non-negative, got {}",
                self.intensity,
            ));
        }
        for f in &self.freqs {
            if !f.is_finite() {
                return Err("TurbulenceModel.freqs must be finite".to_string());
            }
        }
        for axis in &self.coeffs {
            for c in axis {
                if !c.is_finite() {
                    return Err("TurbulenceModel.coeffs must be finite".to_string());
                }
            }
        }
        for axis in &self.phases {
            for p in axis {
                if !p.is_finite() {
                    return Err("TurbulenceModel.phases must be finite".to_string());
                }
            }
        }
        Ok(())
    }

    /// Evaluate the turbulence perturbation at time `t` (seconds).
    ///
    /// Matches the Python reference exactly:
    ///   perturbation_i = (sum_j c_ij * sin(freq_j * t + phase_ij)) / N * intensity
    #[must_use]
    pub fn perturbation(&self, t: f64) -> Vector3 {
        if self.intensity.abs() < 1e-10 {
            return Vector3::zero();
        }
        let mut p = [0.0f64; 3];
        for (axis_idx, axis_coeffs) in self.coeffs.iter().enumerate() {
            let phases = &self.phases[axis_idx];
            let mut accum = 0.0;
            for (j, &freq) in self.freqs.iter().enumerate() {
                accum += axis_coeffs[j] * (freq * t + phases[j]).sin();
            }
            p[axis_idx] = accum / TURBULENCE_MODES as f64 * self.intensity;
        }
        Vector3::new(p[0], p[1], p[2])
    }
}

/// Combined wind model: constant + altitude gradient + turbulence.
///
/// This is the deterministic subset of the Python `WindModel`. The
/// stochastic gust spawner is intentionally excluded from the Rust
/// kernel — see module docs for rationale.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct WindModel {
    pub config: WindConfig,
    pub turbulence: TurbulenceModel,
}

impl WindModel {
    /// Construct a wind model from a config and an explicit turbulence model.
    pub fn new(config: WindConfig, turbulence: TurbulenceModel) -> Result<Self, String> {
        config.validate()?;
        turbulence.validate()?;
        Ok(Self { config, turbulence })
    }

    /// Evaluate wind at time `t` and ball position `position`.
    ///
    /// Matches `WindModel.get_wind_at` in Python with `gusts_enabled=False`.
    #[must_use]
    pub fn wind_at(&self, t: f64, position: &Vector3) -> Vector3 {
        debug_assert!(t.is_finite());
        debug_assert!(position.x.is_finite() && position.y.is_finite() && position.z.is_finite());

        let mut wind = self.config.base_velocity;

        if self.config.altitude_gradient {
            let altitude = position.z.max(0.0);
            let multiplier = 1.0 + self.config.gradient_factor * (altitude / 10.0);
            wind = Vector3::new(
                wind.x * multiplier,
                wind.y * multiplier,
                wind.z * multiplier,
            );
        }

        let perturbation = self.turbulence.perturbation(t);
        Vector3::new(
            wind.x + perturbation.x,
            wind.y + perturbation.y,
            wind.z + perturbation.z,
        )
    }
}

// ── Python bindings ──────────────────────────────────────────────────────────

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl WindConfig {
    #[new]
    #[pyo3(signature = (
        base_velocity=[0.0, 0.0, 0.0],
        altitude_gradient=false,
        gradient_factor=0.05,
        turbulence_intensity=0.0,
    ))]
    fn py_new(
        base_velocity: [f64; 3],
        altitude_gradient: bool,
        gradient_factor: f64,
        turbulence_intensity: f64,
    ) -> pyo3::PyResult<Self> {
        let cfg = Self {
            base_velocity: Vector3::new(base_velocity[0], base_velocity[1], base_velocity[2]),
            altitude_gradient,
            gradient_factor,
            turbulence_intensity,
        };
        cfg.validate()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(cfg)
    }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl WindModel {
    /// Construct a `WindModel` from a `WindConfig` and raw turbulence arrays.
    ///
    /// `coeffs` and `phases` are flat length-`3*TURBULENCE_MODES` arrays in
    /// row-major (axis, mode) order; `freqs` is length `TURBULENCE_MODES`.
    /// Pass empty/zero arrays plus `turbulence_intensity=0.0` to disable
    /// turbulence entirely.
    #[new]
    #[pyo3(signature = (config, coeffs, phases, freqs))]
    fn py_new(
        config: WindConfig,
        coeffs: Vec<f64>,
        phases: Vec<f64>,
        freqs: Vec<f64>,
    ) -> pyo3::PyResult<Self> {
        if coeffs.len() != 3 * TURBULENCE_MODES {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coeffs must have {} entries (3 * {}), got {}",
                3 * TURBULENCE_MODES,
                TURBULENCE_MODES,
                coeffs.len()
            )));
        }
        if phases.len() != 3 * TURBULENCE_MODES {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "phases must have {} entries (3 * {}), got {}",
                3 * TURBULENCE_MODES,
                TURBULENCE_MODES,
                phases.len()
            )));
        }
        if freqs.len() != TURBULENCE_MODES {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "freqs must have {} entries, got {}",
                TURBULENCE_MODES,
                freqs.len()
            )));
        }
        let mut c = [[0.0; TURBULENCE_MODES]; 3];
        let mut p = [[0.0; TURBULENCE_MODES]; 3];
        let mut f = [0.0; TURBULENCE_MODES];
        for (axis, (c_row, p_row)) in c.iter_mut().zip(p.iter_mut()).enumerate() {
            let base = axis * TURBULENCE_MODES;
            c_row.copy_from_slice(&coeffs[base..base + TURBULENCE_MODES]);
            p_row.copy_from_slice(&phases[base..base + TURBULENCE_MODES]);
        }
        f.copy_from_slice(&freqs[..TURBULENCE_MODES]);
        let turbulence = TurbulenceModel {
            intensity: config.turbulence_intensity,
            coeffs: c,
            phases: p,
            freqs: f,
        };
        Self::new(config, turbulence).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Evaluate wind at `(t, position)` and return `[wx, wy, wz]`.
    #[pyo3(name = "wind_at")]
    fn py_wind_at(&self, t: f64, position: [f64; 3]) -> [f64; 3] {
        let pos = Vector3::new(position[0], position[1], position[2]);
        let w = self.wind_at(t, &pos);
        [w.x, w.y, w.z]
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn disabled_wind() -> WindModel {
        WindModel::new(WindConfig::default(), TurbulenceModel::disabled())
            .expect("default config should validate")
    }

    #[test]
    fn test_wind_config_default_is_zero_wind() {
        let cfg = WindConfig::default();
        cfg.validate().unwrap();
        assert!(cfg.base_velocity.x.abs() < 1e-15);
        assert!(cfg.base_velocity.y.abs() < 1e-15);
        assert!(cfg.base_velocity.z.abs() < 1e-15);
        assert!(!cfg.altitude_gradient);
    }

    #[test]
    fn test_wind_config_rejects_negative_turbulence() {
        let cfg = WindConfig {
            turbulence_intensity: -0.1,
            ..WindConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_wind_zero_at_origin_with_defaults() {
        let model = disabled_wind();
        let w = model.wind_at(0.5, &Vector3::zero());
        assert!(w.x.abs() < 1e-12 && w.y.abs() < 1e-12 && w.z.abs() < 1e-12);
    }

    #[test]
    fn test_wind_constant_base_velocity() {
        let cfg = WindConfig {
            base_velocity: Vector3::new(3.0, -1.0, 0.0),
            ..WindConfig::default()
        };
        let model = WindModel::new(cfg, TurbulenceModel::disabled()).unwrap();
        let w = model.wind_at(0.0, &Vector3::new(0.0, 0.0, 5.0));
        assert!((w.x - 3.0).abs() < 1e-12);
        assert!((w.y - -1.0).abs() < 1e-12);
        assert!(w.z.abs() < 1e-12);
    }

    #[test]
    fn test_wind_altitude_gradient_scales_with_height() {
        let cfg = WindConfig {
            base_velocity: Vector3::new(10.0, 0.0, 0.0),
            altitude_gradient: true,
            gradient_factor: 0.05,
            ..WindConfig::default()
        };
        let model = WindModel::new(cfg, TurbulenceModel::disabled()).unwrap();

        // altitude = 20 m → multiplier = 1 + 0.05 * (20/10) = 1.10
        let w = model.wind_at(0.0, &Vector3::new(0.0, 0.0, 20.0));
        assert!((w.x - 11.0).abs() < 1e-10, "expected 11.0, got {}", w.x);

        // Negative z must clamp at 0 → multiplier = 1.0
        let w_below = model.wind_at(0.0, &Vector3::new(0.0, 0.0, -5.0));
        assert!((w_below.x - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_turbulence_zero_intensity_is_no_op() {
        let t = TurbulenceModel::disabled();
        let p = t.perturbation(1.234);
        assert!(p.x.abs() < 1e-15 && p.y.abs() < 1e-15 && p.z.abs() < 1e-15);
    }

    #[test]
    fn test_turbulence_matches_python_formula() {
        // Identity weights: each axis sums sin(j * t + 0) for j in 1..=10.
        let coeffs = [[1.0; TURBULENCE_MODES]; 3];
        let phases = [[0.0; TURBULENCE_MODES]; 3];
        let mut freqs = [0.0; TURBULENCE_MODES];
        for (j, f) in freqs.iter_mut().enumerate() {
            *f = (j + 1) as f64;
        }
        let turb = TurbulenceModel {
            intensity: 1.0,
            coeffs,
            phases,
            freqs,
        };
        turb.validate().unwrap();

        let t = 0.25;
        let expected: f64 =
            (1..=10).map(|j| ((j as f64) * t).sin()).sum::<f64>() / TURBULENCE_MODES as f64;

        let p = turb.perturbation(t);
        assert!((p.x - expected).abs() < 1e-12);
        assert!((p.y - expected).abs() < 1e-12);
        assert!((p.z - expected).abs() < 1e-12);
    }
}
