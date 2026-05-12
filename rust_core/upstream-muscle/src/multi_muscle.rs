use std::collections::HashMap;
use crate::hill_muscle::{HillMuscleModel, MuscleState};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyclass)]
pub struct MuscleAttachment {
    #[pyo3(get, set)]
    pub muscle_name: String,
    #[pyo3(get, set)]
    pub moment_arm: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl MuscleAttachment {
    #[new]
    fn py_new(muscle_name: String, moment_arm: f64) -> Self {
        Self { muscle_name, moment_arm }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyclass)]
pub struct MuscleGroup {
    #[pyo3(get, set)]
    pub name: String,
    pub muscles: HashMap<String, HillMuscleModel>,
    pub attachments: HashMap<String, MuscleAttachment>,
}

#[cfg(feature = "python")]
#[pymethods]
impl MuscleGroup {
    #[new]
    fn py_new(name: String) -> Self {
        Self {
            name,
            muscles: HashMap::new(),
            attachments: HashMap::new(),
        }
    }

    #[pyo3(name = "add_muscle")]
    fn py_add_muscle(&mut self, name: String, muscle: HillMuscleModel, moment_arm: f64) -> pyo3::PyResult<()> {
        if name.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("muscle name must be non-empty"));
        }
        if moment_arm == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("moment_arm must be non-zero"));
        }
        
        self.muscles.insert(name.clone(), muscle);
        self.attachments.insert(name.clone(), MuscleAttachment { muscle_name: name, moment_arm });
        Ok(())
    }

    #[pyo3(name = "compute_net_torque")]
    fn py_compute_net_torque(
        &self,
        activations: HashMap<String, f64>,
        muscle_states: HashMap<String, (f64, f64)>,
    ) -> pyo3::PyResult<f64> {
        let mut net_torque = 0.0;

        for (name, muscle) in &self.muscles {
            if let Some(&act_val) = activations.get(name) {
                if act_val < 0.0 || act_val > 1.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!("activation for '{}' must be in [0, 1]", name)));
                }

                let (l_ce, v_ce) = muscle_states.get(name).copied().unwrap_or((muscle.params.l_opt, 0.0));
                let state = MuscleState {
                    activation: act_val,
                    l_ce,
                    v_ce,
                    l_mt: 0.0,
                };

                let force = muscle.compute_force(&state);
                let r = self.attachments.get(name).map(|a| a.moment_arm).unwrap_or(0.0);
                net_torque += r * force;
            }
        }

        Ok(net_torque)
    }

    #[getter]
    fn get_muscle_names(&self) -> Vec<String> {
        self.muscles.keys().cloned().collect()
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyclass)]
pub struct AntagonistPair {
    pub agonist: MuscleGroup,
    pub antagonist: MuscleGroup,
}

#[cfg(feature = "python")]
#[pymethods]
impl AntagonistPair {
    #[new]
    fn py_new(agonist: MuscleGroup, antagonist: MuscleGroup) -> Self {
        Self { agonist, antagonist }
    }

    #[pyo3(name = "compute_net_torque")]
    fn py_compute_net_torque(
        &self,
        agonist_activations: HashMap<String, f64>,
        antagonist_activations: HashMap<String, f64>,
        muscle_states: HashMap<String, (f64, f64)>,
    ) -> pyo3::PyResult<f64> {
        let tau_agonist = self.agonist.py_compute_net_torque(agonist_activations, muscle_states.clone())?;
        let tau_antagonist = self.antagonist.py_compute_net_torque(antagonist_activations, muscle_states)?;
        
        Ok(tau_agonist + tau_antagonist)
    }

    #[getter]
    fn get_muscle_names(&self) -> Vec<String> {
        let mut names = self.agonist.get_muscle_names();
        names.extend(self.antagonist.get_muscle_names());
        names
    }
}
