//! State-bearing Hill muscle model.
//!
//! Direct scalar port of `src/shared/python/biomechanics/hill_muscle.py`.

#[cfg(feature = "python")]
#[allow(unused_imports)]
use pyo3::prelude::*;

use crate::hill::{f_l_with_width, f_p, f_v};

/// Parameters defining a specific muscle.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct MuscleParameters {
    pub f_max: f64,
    pub l_opt: f64,
    pub l_slack: f64,
    pub v_max: f64,
    pub pennation_angle: f64,
    pub damping: f64,
}

impl MuscleParameters {
    /// Construct parameters after validating the Python dataclass invariants.
    pub fn new(
        f_max: f64,
        l_opt: f64,
        l_slack: f64,
        v_max: f64,
        pennation_angle: f64,
        damping: f64,
    ) -> Result<Self, String> {
        let params = Self {
            f_max,
            l_opt,
            l_slack,
            v_max,
            pennation_angle,
            damping,
        };
        params.validate()?;
        Ok(params)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.f_max <= 0.0 {
            return Err(format!("f_max must be positive, got {}", self.f_max));
        }
        if self.l_opt <= 0.0 {
            return Err(format!("l_opt must be positive, got {}", self.l_opt));
        }
        if self.l_slack <= 0.0 {
            return Err(format!("l_slack must be positive, got {}", self.l_slack));
        }
        Ok(())
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl MuscleParameters {
    #[new]
    #[pyo3(signature = (f_max, l_opt, l_slack, v_max = 10.0, pennation_angle = 0.0, damping = 0.05))]
    fn py_new(
        f_max: f64,
        l_opt: f64,
        l_slack: f64,
        v_max: f64,
        pennation_angle: f64,
        damping: f64,
    ) -> pyo3::PyResult<Self> {
        Self::new(f_max, l_opt, l_slack, v_max, pennation_angle, damping)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

/// Current muscle state.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct MuscleState {
    pub activation: f64,
    pub l_ce: f64,
    pub v_ce: f64,
    pub l_mt: f64,
}

impl Default for MuscleState {
    fn default() -> Self {
        Self {
            activation: 0.0,
            l_ce: 0.0,
            v_ce: 0.0,
            l_mt: 0.0,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl MuscleState {
    #[new]
    #[pyo3(signature = (activation = 0.0, l_ce = 0.0, v_ce = 0.0, l_mt = 0.0))]
    fn py_new(activation: f64, l_ce: f64, v_ce: f64, l_mt: f64) -> Self {
        Self {
            activation,
            l_ce,
            v_ce,
            l_mt,
        }
    }
}

/// Standard state-bearing Hill-type muscle model.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct HillMuscleModel {
    pub params: MuscleParameters,
    pub force_length_width: f64,
}

impl HillMuscleModel {
    pub fn new(params: MuscleParameters, force_length_width: Option<f64>) -> Self {
        Self {
            params,
            force_length_width: force_length_width
                .unwrap_or(crate::hill::DEFAULT_FORCE_LENGTH_WIDTH),
        }
    }

    #[inline]
    pub fn force_length_active(&self, l_norm: f64) -> f64 {
        f_l_with_width(l_norm, self.force_length_width)
    }

    #[inline]
    pub fn force_length_passive(&self, l_norm: f64) -> f64 {
        f_p(l_norm)
    }

    #[inline]
    pub fn force_velocity(&self, v_norm: f64) -> f64 {
        f_v(v_norm)
    }

    #[inline]
    pub fn tendon_force(&self, l_tendon_norm: f64) -> f64 {
        crate::hill::f_t(l_tendon_norm)
    }

    /// Compute total fiber force projected onto the tendon line of action.
    pub fn compute_force(&self, state: &MuscleState) -> Result<f64, String> {
        if !(0.0..=1.0).contains(&state.activation) {
            return Err(format!(
                "activation must be in [0, 1], got {}",
                state.activation
            ));
        }

        let l_norm = state.l_ce / self.params.l_opt;
        let v_norm = state.v_ce / (self.params.v_max * self.params.l_opt);

        let active = self.params.f_max
            * state.activation
            * self.force_length_active(l_norm)
            * self.force_velocity(v_norm);
        let passive = self.params.f_max * self.force_length_passive(l_norm);
        let damping = self.params.damping * state.v_ce;
        let total = (active + passive + damping) * self.params.pennation_angle.cos();

        Ok(total.max(0.0))
    }

    /// Batch helper for lower-overhead Python/RL loops.
    pub fn compute_force_batch(&self, states: &[MuscleState]) -> Result<Vec<f64>, String> {
        states
            .iter()
            .map(|state| self.compute_force(state))
            .collect()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl HillMuscleModel {
    #[new]
    #[pyo3(signature = (params, force_length_width = None))]
    fn py_new(params: MuscleParameters, force_length_width: Option<f64>) -> Self {
        Self::new(params, force_length_width)
    }

    #[getter]
    fn params(&self) -> MuscleParameters {
        self.params
    }

    #[getter]
    fn force_length_width(&self) -> f64 {
        self.force_length_width
    }

    #[pyo3(name = "force_length_active")]
    fn py_force_length_active(&self, l_norm: f64) -> f64 {
        self.force_length_active(l_norm)
    }

    #[pyo3(name = "force_length_passive")]
    fn py_force_length_passive(&self, l_norm: f64) -> f64 {
        self.force_length_passive(l_norm)
    }

    #[pyo3(name = "force_velocity")]
    fn py_force_velocity(&self, v_norm: f64) -> f64 {
        self.force_velocity(v_norm)
    }

    #[pyo3(name = "tendon_force")]
    fn py_tendon_force(&self, l_tendon_norm: f64) -> f64 {
        self.tendon_force(l_tendon_norm)
    }

    #[pyo3(name = "compute_force")]
    fn py_compute_force(&self, state: &MuscleState) -> pyo3::PyResult<f64> {
        self.compute_force(state)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[pyo3(name = "compute_force_batch")]
    fn py_compute_force_batch(&self, states: Vec<MuscleState>) -> pyo3::PyResult<Vec<f64>> {
        self.compute_force_batch(&states)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}
