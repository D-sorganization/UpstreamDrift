//! Multi-muscle moment summation — HashMap-attachment API.
//!
//! Direct port of `MuscleGroup` and `AntagonistPair` from
//! `src/shared/python/biomechanics/multi_muscle.py`. The dense-matrix
//! equivalent for RL hot loops lives in [`crate::multi`].

use std::collections::HashMap;

#[cfg(feature = "python")]
#[allow(unused_imports)]
use pyo3::prelude::*;

use crate::model::{HillMuscleModel, MuscleState};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct MuscleAttachment {
    pub muscle_name: String,
    pub moment_arm: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl MuscleAttachment {
    #[new]
    fn py_new(muscle_name: String, moment_arm: f64) -> Self {
        Self {
            muscle_name,
            moment_arm,
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyclass)]
pub struct MuscleGroup {
    pub name: String,
    pub muscles: HashMap<String, HillMuscleModel>,
    pub attachments: HashMap<String, MuscleAttachment>,
}

impl MuscleGroup {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            muscles: HashMap::new(),
            attachments: HashMap::new(),
        }
    }

    pub fn add_muscle(
        &mut self,
        name: impl Into<String>,
        muscle: HillMuscleModel,
        moment_arm: f64,
    ) -> Result<(), String> {
        let name = name.into();
        if name.is_empty() {
            return Err("muscle name must be non-empty".into());
        }
        if moment_arm == 0.0 {
            return Err("moment_arm must be non-zero".into());
        }
        self.muscles.insert(name.clone(), muscle);
        self.attachments.insert(
            name.clone(),
            MuscleAttachment {
                muscle_name: name,
                moment_arm,
            },
        );
        Ok(())
    }

    pub fn compute_net_torque(
        &self,
        activations: &HashMap<String, f64>,
        muscle_states: &HashMap<String, (f64, f64)>,
    ) -> Result<f64, String> {
        let mut net_torque = 0.0;
        for (name, muscle) in &self.muscles {
            let Some(&act_val) = activations.get(name) else {
                continue;
            };
            if !(0.0..=1.0).contains(&act_val) {
                return Err(format!(
                    "activation for '{name}' must be in [0, 1], got {act_val}"
                ));
            }
            let (l_ce, v_ce) = muscle_states
                .get(name)
                .copied()
                .unwrap_or((muscle.params.l_opt, 0.0));
            let state = MuscleState {
                activation: act_val,
                l_ce,
                v_ce,
                l_mt: 0.0,
            };
            let force = muscle.compute_force(&state)?;
            let r = self
                .attachments
                .get(name)
                .map(|a| a.moment_arm)
                .unwrap_or(0.0);
            net_torque += r * force;
        }
        Ok(net_torque)
    }

    pub fn muscle_names(&self) -> Vec<String> {
        self.muscles.keys().cloned().collect()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl MuscleGroup {
    #[new]
    fn py_new(name: String) -> Self {
        Self::new(name)
    }

    #[pyo3(name = "add_muscle")]
    fn py_add_muscle(
        &mut self,
        name: String,
        muscle: HillMuscleModel,
        moment_arm: f64,
    ) -> PyResult<()> {
        self.add_muscle(name, muscle, moment_arm)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[pyo3(name = "compute_net_torque")]
    fn py_compute_net_torque(
        &self,
        activations: HashMap<String, f64>,
        muscle_states: HashMap<String, (f64, f64)>,
    ) -> PyResult<f64> {
        self.compute_net_torque(&activations, &muscle_states)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[getter]
    fn muscle_names_py(&self) -> Vec<String> {
        self.muscle_names()
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyclass)]
pub struct AntagonistPair {
    pub agonist: MuscleGroup,
    pub antagonist: MuscleGroup,
}

impl AntagonistPair {
    pub fn new(agonist: MuscleGroup, antagonist: MuscleGroup) -> Self {
        Self {
            agonist,
            antagonist,
        }
    }

    pub fn compute_net_torque(
        &self,
        agonist_activations: &HashMap<String, f64>,
        antagonist_activations: &HashMap<String, f64>,
        muscle_states: &HashMap<String, (f64, f64)>,
    ) -> Result<f64, String> {
        let t1 = self
            .agonist
            .compute_net_torque(agonist_activations, muscle_states)?;
        let t2 = self
            .antagonist
            .compute_net_torque(antagonist_activations, muscle_states)?;
        Ok(t1 + t2)
    }

    pub fn muscle_names(&self) -> Vec<String> {
        let mut names = self.agonist.muscle_names();
        names.extend(self.antagonist.muscle_names());
        names
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl AntagonistPair {
    #[new]
    fn py_new(agonist: MuscleGroup, antagonist: MuscleGroup) -> Self {
        Self::new(agonist, antagonist)
    }

    #[pyo3(name = "compute_net_torque")]
    fn py_compute_net_torque(
        &self,
        agonist_activations: HashMap<String, f64>,
        antagonist_activations: HashMap<String, f64>,
        muscle_states: HashMap<String, (f64, f64)>,
    ) -> PyResult<f64> {
        self.compute_net_torque(
            &agonist_activations,
            &antagonist_activations,
            &muscle_states,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[getter]
    fn muscle_names_py(&self) -> Vec<String> {
        self.muscle_names()
    }
}
