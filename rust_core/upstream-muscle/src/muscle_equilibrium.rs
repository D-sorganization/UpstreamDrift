use crate::model::HillMuscleModel;

pub struct EquilibriumSolver<'a> {
    pub muscle: &'a HillMuscleModel,
}

impl<'a> EquilibriumSolver<'a> {
    pub fn new(muscle: &'a HillMuscleModel) -> Self {
        Self { muscle }
    }

    pub fn equilibrium_residual(&self, l_ce: f64, l_mt: f64, activation: f64, v_ce: f64) -> f64 {
        let l_ce_norm = l_ce / self.muscle.params.l_opt;
        let v_ce_norm = v_ce / self.muscle.params.v_max;

        let cos_alpha = self.muscle.params.pennation_angle.cos();
        let l_tendon = l_mt - l_ce * cos_alpha;
        let l_tendon_norm = l_tendon / self.muscle.params.l_slack;

        let f_l = self.muscle.force_length_active(l_ce_norm);
        let f_v = self.muscle.force_velocity(v_ce_norm);
        let f_p = self.muscle.force_length_passive(l_ce_norm);
        let f_t = self.muscle.tendon_force(l_tendon_norm);

        let f_ce = self.muscle.params.f_max * activation * f_l * f_v;
        let f_pee = self.muscle.params.f_max * f_p;
        let f_fiber = (f_ce + f_pee) * cos_alpha;

        let f_tendon_force = self.muscle.params.f_max * f_t;

        f_fiber - f_tendon_force
    }

    /// Solves for l_ce using the Secant method.
    pub fn solve_fiber_length(
        &self,
        l_mt: f64,
        activation: f64,
        v_ce: f64,
        initial_guess: Option<f64>,
    ) -> Result<f64, String> {
        if l_mt <= 0.0 {
            return Err("l_mt must be positive".to_string());
        }
        if !(0.0..=1.0).contains(&activation) {
            return Err("activation must be in [0, 1]".to_string());
        }

        let max_iterations = 100;
        let tolerance = 1e-6;
        let initial = initial_guess.unwrap_or(0.9 * self.muscle.params.l_opt);

        // Secant method
        let mut x0 = initial;
        let mut x1 = initial * 1.01; // slight perturbation

        let mut f0 = self.equilibrium_residual(x0, l_mt, activation, v_ce);
        let mut f1 = self.equilibrium_residual(x1, l_mt, activation, v_ce);

        for _ in 0..max_iterations {
            if f1.abs() < tolerance * self.muscle.params.f_max {
                if x1 <= 0.0 {
                    return Err("Converged to non-positive fiber length".to_string());
                }
                return Ok(x1);
            }

            if (f1 - f0).abs() < 1e-14 {
                break; // avoid division by zero
            }

            let x2 = x1 - f1 * (x1 - x0) / (f1 - f0);

            x0 = x1;
            f0 = f1;
            x1 = x2;
            f1 = self.equilibrium_residual(x1, l_mt, activation, v_ce);
        }

        Err("Muscle equilibrium solver failed to converge".to_string())
    }

    pub fn solve_fiber_velocity(
        &self,
        l_mt: f64,
        v_mt: f64,
        activation: f64,
        l_ce: f64,
        dt: f64,
    ) -> Result<f64, String> {
        if dt <= 0.0 {
            return Err("dt must be positive".to_string());
        }
        if l_ce <= 0.0 {
            return Err("l_ce must be positive".to_string());
        }

        let l_mt_next = l_mt + v_mt * dt;
        let l_ce_next = self.solve_fiber_length(l_mt_next, activation, 0.0, Some(l_ce))?;

        Ok((l_ce_next - l_ce) / dt)
    }
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (muscle, l_mt, v_mt, activation, initial_l_ce = None))]
pub fn compute_equilibrium_state(
    muscle: &HillMuscleModel,
    l_mt: f64,
    v_mt: f64,
    activation: f64,
    initial_l_ce: Option<f64>,
) -> pyo3::PyResult<(f64, f64)> {
    let solver = EquilibriumSolver::new(muscle);

    let l_ce = solver
        .solve_fiber_length(l_mt, activation, 0.0, initial_l_ce)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let v_ce = if v_mt.abs() > 1e-10 {
        solver
            .solve_fiber_velocity(l_mt, v_mt, activation, l_ce, 0.001)
            .unwrap_or(0.0)
    } else {
        0.0
    };

    Ok((l_ce, v_ce))
}

#[cfg(feature = "python")]
#[pyclass(name = "EquilibriumSolver")]
pub struct PyEquilibriumSolver {
    pub muscle: HillMuscleModel,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyEquilibriumSolver {
    #[new]
    fn py_new(muscle: HillMuscleModel) -> Self {
        Self { muscle }
    }

    #[pyo3(name = "solve_fiber_length")]
    #[pyo3(signature = (l_mt, activation, v_ce = 0.0, initial_guess = None))]
    fn py_solve_fiber_length(
        &self,
        l_mt: f64,
        activation: f64,
        v_ce: f64,
        initial_guess: Option<f64>,
    ) -> pyo3::PyResult<f64> {
        let solver = EquilibriumSolver::new(&self.muscle);
        solver
            .solve_fiber_length(l_mt, activation, v_ce, initial_guess)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    #[pyo3(name = "solve_fiber_velocity")]
    #[pyo3(signature = (l_mt, v_mt, activation, l_ce, dt = 0.001))]
    fn py_solve_fiber_velocity(
        &self,
        l_mt: f64,
        v_mt: f64,
        activation: f64,
        l_ce: f64,
        dt: f64,
    ) -> pyo3::PyResult<f64> {
        let solver = EquilibriumSolver::new(&self.muscle);
        solver
            .solve_fiber_velocity(l_mt, v_mt, activation, l_ce, dt)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
}
