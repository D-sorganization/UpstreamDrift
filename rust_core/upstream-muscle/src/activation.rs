//! Activation dynamics for Hill-type muscle models.
//!
//! This is a direct scalar port of
//! `src/shared/python/biomechanics/activation_dynamics.py`.

#[cfg(feature = "python")]
#[allow(unused_imports)]
use pyo3::prelude::*;

/// Default activation time constant, seconds.
pub const DEFAULT_TAU_ACT: f64 = 0.010;

/// Default deactivation time constant, seconds.
pub const DEFAULT_TAU_DEACT: f64 = 0.040;

/// Default lower activation floor.
pub const DEFAULT_MIN_ACTIVATION: f64 = 0.001;

/// First-order neural excitation to muscle activation dynamics.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct ActivationDynamics {
    pub tau_act: f64,
    pub tau_deact: f64,
    pub min_activation: f64,
}

impl Default for ActivationDynamics {
    fn default() -> Self {
        Self {
            tau_act: DEFAULT_TAU_ACT,
            tau_deact: DEFAULT_TAU_DEACT,
            min_activation: DEFAULT_MIN_ACTIVATION,
        }
    }
}

impl ActivationDynamics {
    /// Construct activation dynamics after validating DbC preconditions.
    pub fn new(tau_act: f64, tau_deact: f64, min_activation: f64) -> Result<Self, String> {
        let dynamics = Self {
            tau_act,
            tau_deact,
            min_activation,
        };
        dynamics.validate()?;
        Ok(dynamics)
    }

    /// Validate constructor preconditions from the Python implementation.
    pub fn validate(&self) -> Result<(), String> {
        if self.tau_act <= 0.0 {
            return Err(format!("tau_act must be positive, got {}", self.tau_act));
        }
        if self.tau_deact <= 0.0 {
            return Err(format!(
                "tau_deact must be positive, got {}",
                self.tau_deact
            ));
        }
        if self.min_activation <= 0.0 || self.min_activation >= 1.0 {
            return Err(format!(
                "min_activation must be in (0, 1), got {}",
                self.min_activation
            ));
        }
        Ok(())
    }

    /// Compute `da/dt` after clamping excitation and activation.
    #[inline]
    pub fn compute_derivative(&self, u: f64, a: f64) -> f64 {
        let u_clamped = u.clamp(self.min_activation, 1.0);
        let a_clamped = a.clamp(self.min_activation, 1.0);

        let tau = if u_clamped > a_clamped {
            self.tau_act * (0.5 + 1.5 * a_clamped)
        } else {
            self.tau_deact / (0.5 + 1.5 * a_clamped)
        };

        (u_clamped - a_clamped) / tau
    }

    /// Euler-update activation by one positive timestep.
    pub fn update(&self, u: f64, a: f64, dt: f64) -> Result<f64, String> {
        if dt <= 0.0 {
            return Err(format!("time step dt must be positive, got {dt}"));
        }
        let a_new = a + self.compute_derivative(u, a) * dt;
        Ok(a_new.clamp(self.min_activation, 1.0))
    }

    /// Batch update helper for lower-overhead Python/RL loops.
    pub fn update_batch(&self, u: &[f64], a: &[f64], dt: f64) -> Result<Vec<f64>, String> {
        if u.len() != a.len() {
            return Err(format!(
                "Batch sizes must match: u has {}, a has {}",
                u.len(),
                a.len()
            ));
        }
        u.iter()
            .zip(a.iter())
            .map(|(&u_i, &a_i)| self.update(u_i, a_i, dt))
            .collect()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl ActivationDynamics {
    #[new]
    #[pyo3(signature = (tau_act = DEFAULT_TAU_ACT, tau_deact = DEFAULT_TAU_DEACT, min_activation = DEFAULT_MIN_ACTIVATION))]
    fn py_new(tau_act: f64, tau_deact: f64, min_activation: f64) -> pyo3::PyResult<Self> {
        Self::new(tau_act, tau_deact, min_activation)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[pyo3(name = "compute_derivative")]
    fn py_compute_derivative(&self, u: f64, a: f64) -> f64 {
        self.compute_derivative(u, a)
    }

    #[pyo3(name = "update")]
    fn py_update(&self, u: f64, a: f64, dt: f64) -> pyo3::PyResult<f64> {
        self.update(u, a, dt)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[pyo3(name = "update_batch")]
    fn py_update_batch(&self, u: Vec<f64>, a: Vec<f64>, dt: f64) -> pyo3::PyResult<Vec<f64>> {
        self.update_batch(&u, &a, dt)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}
