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
        if self.v_max <= 0.0 {
            return Err(format!("v_max must be positive, got {}", self.v_max));
        }
        // Pennation must lie in [0, pi/2): at pi/2 the fibre is perpendicular to
        // the tendon and `cos(alpha)` collapses to zero, producing zero/garbage
        // muscle force. Negative angles are unphysical.
        if !(0.0..std::f64::consts::FRAC_PI_2).contains(&self.pennation_angle) {
            return Err(format!(
                "pennation_angle must be in [0, pi/2), got {}",
                self.pennation_angle
            ));
        }
        if self.damping < 0.0 {
            return Err(format!(
                "damping must be non-negative, got {}",
                self.damping
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod model_validation_tests {
    use super::MuscleParameters;

    fn valid() -> [f64; 6] {
        // f_max, l_opt, l_slack, v_max, pennation_angle, damping
        [1000.0, 0.15, 0.20, 10.0, 0.0, 0.05]
    }

    fn build(p: [f64; 6]) -> Result<MuscleParameters, String> {
        MuscleParameters::new(p[0], p[1], p[2], p[3], p[4], p[5])
    }

    #[test]
    fn valid_params_construct() {
        assert!(build(valid()).is_ok());
        // Pennation just under pi/2 is allowed.
        let mut p = valid();
        p[4] = std::f64::consts::FRAC_PI_2 - 1e-9;
        assert!(build(p).is_ok());
        // Zero damping is allowed (>= 0).
        let mut p = valid();
        p[5] = 0.0;
        assert!(build(p).is_ok());
    }

    #[test]
    fn non_positive_v_max_rejected() {
        let mut p = valid();
        p[3] = 0.0;
        assert!(build(p).is_err());
        p[3] = -1.0;
        assert!(build(p).is_err());
    }

    #[test]
    fn out_of_range_pennation_rejected() {
        // Negative pennation.
        let mut p = valid();
        p[4] = -0.01;
        assert!(build(p).is_err());
        // Exactly pi/2 (excluded upper bound).
        let mut p = valid();
        p[4] = std::f64::consts::FRAC_PI_2;
        assert!(build(p).is_err());
        // Beyond pi/2.
        let mut p = valid();
        p[4] = 2.0;
        assert!(build(p).is_err());
    }

    #[test]
    fn negative_damping_rejected() {
        let mut p = valid();
        p[5] = -0.01;
        assert!(build(p).is_err());
    }

    #[test]
    fn existing_positive_constraints_still_enforced() {
        let mut p = valid();
        p[0] = 0.0; // f_max
        assert!(build(p).is_err());
        let mut p = valid();
        p[1] = -0.1; // l_opt
        assert!(build(p).is_err());
        let mut p = valid();
        p[2] = 0.0; // l_slack
        assert!(build(p).is_err());
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
