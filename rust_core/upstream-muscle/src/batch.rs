//! Batched / parallel kernels for RL inner loops.
//!
//! The expected hot path is: for each environment step, given `M` muscles
//! and their per-muscle state and parameters, compute (a) the new
//! activation from the excitation, (b) the muscle force, and (c) the
//! resulting joint torques. The Python equivalents (in
//! `activation_dynamics.py` / `hill_muscle.py` / `multi_muscle.py`)
//! interpret per-call which is fine for one muscle and a kernel-eater
//! at `M = 1000+`.
//!
//! These functions operate on plain `&[f64]` / typed slices so they're
//! trivially callable from numpy-backed PyO3 wrappers (`crate::python_api`)
//! and from native Rust callers. Parallelism is via `rayon` chunking; the
//! GIL is released by the PyO3 wrapper before invoking them.
//!
//! These build on the scalar primitives introduced by PR #5246
//! (`ActivationDynamics::update`, `HillMuscleModel::compute_force`).

use rayon::prelude::*;

use crate::activation::ActivationDynamics;
use crate::model::{HillMuscleModel, MuscleState};

/// Heuristic chunk size below which we skip Rayon and run serially —
/// avoids parallel overhead drowning the work for small `M`.
const PARALLEL_THRESHOLD: usize = 128;

/// Advance activation for `M` muscles by one Euler step.
///
/// All slices must have length `M`. Each `dyn_params[i]` is the
/// `ActivationDynamics` value for muscle `i` (pass repeated copies of the
/// same struct if all muscles share parameters).
///
/// Returns `Err` if any sub-step fails its preconditions (which can only
/// happen for non-positive `dt`; per-muscle clamps make the rest
/// total-functional).
pub fn activation_step_batch(
    u: &[f64],
    a_in: &[f64],
    dt: f64,
    dyn_params: &[ActivationDynamics],
    a_out: &mut [f64],
) -> Result<(), String> {
    let m = u.len();
    if a_in.len() != m || a_out.len() != m || dyn_params.len() != m {
        return Err(format!(
            "length mismatch: u={}, a_in={}, a_out={}, dyn_params={}",
            m,
            a_in.len(),
            a_out.len(),
            dyn_params.len()
        ));
    }
    if dt <= 0.0 {
        return Err(format!("dt must be positive, got {dt}"));
    }

    if m < PARALLEL_THRESHOLD {
        for i in 0..m {
            a_out[i] = dyn_params[i].update(u[i], a_in[i], dt)?;
        }
        return Ok(());
    }
    let results: Result<Vec<f64>, String> = (0..m)
        .into_par_iter()
        .map(|i| dyn_params[i].update(u[i], a_in[i], dt))
        .collect();
    let results = results?;
    a_out.copy_from_slice(&results);
    Ok(())
}

/// Compute fiber-projected muscle forces for `M` muscles.
///
/// All slices must have length `M`. `f_out[i]` receives the total force
/// for muscle `i` (equivalent to `HillMuscleModel::compute_force`).
pub fn muscle_force_batch(
    models: &[HillMuscleModel],
    states: &[MuscleState],
    f_out: &mut [f64],
) -> Result<(), String> {
    let m = models.len();
    if states.len() != m || f_out.len() != m {
        return Err(format!(
            "length mismatch: models={}, states={}, f_out={}",
            m,
            states.len(),
            f_out.len()
        ));
    }

    if m < PARALLEL_THRESHOLD {
        for i in 0..m {
            f_out[i] = models[i].compute_force(&states[i])?;
        }
        return Ok(());
    }
    let results: Result<Vec<f64>, String> = (0..m)
        .into_par_iter()
        .map(|i| models[i].compute_force(&states[i]))
        .collect();
    f_out.copy_from_slice(&results?);
    Ok(())
}

/// Compute joint torques for `J` joints from `M` muscle forces and a
/// row-major moment-arm matrix `R` of shape `(J, M)`.
///
/// `tau_out` is `J`-long, `r` is `J*M`-long, `f` is `M`-long.
pub fn joint_torques_batch(
    r: &[f64],
    j: usize,
    m: usize,
    f: &[f64],
    tau_out: &mut [f64],
) -> Result<(), String> {
    if r.len() != j * m {
        return Err(format!(
            "moment-arm matrix has wrong length: expected {}, got {}",
            j * m,
            r.len()
        ));
    }
    if f.len() != m {
        return Err(format!(
            "force vector length mismatch: expected {}, got {}",
            m,
            f.len()
        ));
    }
    if tau_out.len() != j {
        return Err(format!(
            "output length mismatch: expected {}, got {}",
            j,
            tau_out.len()
        ));
    }

    if j < PARALLEL_THRESHOLD {
        for jj in 0..j {
            let row = &r[jj * m..(jj + 1) * m];
            let mut acc = 0.0;
            for i in 0..m {
                acc += row[i] * f[i];
            }
            tau_out[jj] = acc;
        }
        return Ok(());
    }
    tau_out.par_iter_mut().enumerate().for_each(|(jj, out)| {
        let row = &r[jj * m..(jj + 1) * m];
        let mut acc = 0.0;
        for i in 0..m {
            acc += row[i] * f[i];
        }
        *out = acc;
    });
    Ok(())
}

/// One-shot RL step: excitation → activation update → muscle force →
/// joint torque, for `M` muscles and `J` joints.
///
/// Results land in `a_out` (length `M`) and `tau_out` (length `J`). The
/// muscle-force calculation uses the freshly updated activation values
/// from `a_out`.
#[allow(clippy::too_many_arguments)]
pub fn step_full(
    u: &[f64],
    a_in: &[f64],
    dt: f64,
    dyn_params: &[ActivationDynamics],
    models: &[HillMuscleModel],
    states: &[MuscleState],
    moment_arms: &[f64],
    n_joints: usize,
    a_out: &mut [f64],
    tau_out: &mut [f64],
) -> Result<(), String> {
    let m = u.len();
    activation_step_batch(u, a_in, dt, dyn_params, a_out)?;

    // Patch states' activation with the freshly computed value.
    let updated_states: Vec<MuscleState> = (0..m)
        .map(|i| MuscleState {
            activation: a_out[i],
            l_ce: states[i].l_ce,
            v_ce: states[i].v_ce,
            l_mt: states[i].l_mt,
        })
        .collect();
    let mut forces = vec![0.0_f64; m];
    muscle_force_batch(models, &updated_states, &mut forces)?;
    joint_torques_batch(moment_arms, n_joints, m, &forces, tau_out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{HillMuscleModel, MuscleParameters};

    fn default_dyn() -> ActivationDynamics {
        ActivationDynamics::default()
    }

    fn default_model() -> HillMuscleModel {
        let p = MuscleParameters::new(1000.0, 0.15, 0.20, 10.0, 0.0, 0.05).expect("valid params");
        HillMuscleModel::new(p, None)
    }

    #[test]
    fn activation_batch_matches_scalar_calls() {
        let m = 256;
        let u: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64)).collect();
        let a_in: Vec<f64> = vec![0.1; m];
        let dyns: Vec<ActivationDynamics> = vec![default_dyn(); m];
        let mut a_out = vec![0.0; m];
        activation_step_batch(&u, &a_in, 0.001, &dyns, &mut a_out).unwrap();
        for i in 0..m {
            let expected = dyns[i].update(u[i], a_in[i], 0.001).unwrap();
            assert!((a_out[i] - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn force_batch_matches_scalar_calls() {
        let m = 256;
        let models: Vec<HillMuscleModel> = vec![default_model(); m];
        let states: Vec<MuscleState> = (0..m)
            .map(|i| MuscleState {
                activation: 0.5,
                l_ce: 0.10 + 0.001 * i as f64,
                v_ce: 0.0,
                l_mt: 0.30,
            })
            .collect();
        let mut f_out = vec![0.0; m];
        muscle_force_batch(&models, &states, &mut f_out).unwrap();
        for i in 0..m {
            let expected = models[i].compute_force(&states[i]).unwrap();
            assert!((f_out[i] - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn joint_torque_batch_matches_serial() {
        let j = 4;
        let m = 200;
        let r: Vec<f64> = (0..j * m).map(|k| (k as f64).sin() * 0.05).collect();
        let f: Vec<f64> = (0..m).map(|k| 10.0 + k as f64).collect();
        let mut tau_par = vec![0.0; j];
        let mut tau_ser = vec![0.0; j];
        joint_torques_batch(&r, j, m, &f, &mut tau_par).unwrap();
        crate::multi::joint_torques(&r, j, m, &f, &mut tau_ser);
        for jj in 0..j {
            assert!((tau_par[jj] - tau_ser[jj]).abs() < 1e-9);
        }
    }

    #[test]
    fn step_full_smoke() {
        let m = 4;
        let j = 1;
        let u = vec![1.0; m];
        let a_in = vec![0.0; m];
        let dyns = vec![default_dyn(); m];
        let models = vec![default_model(); m];
        let states = vec![
            MuscleState {
                activation: 0.0,
                l_ce: 0.15,
                v_ce: 0.0,
                l_mt: 0.35,
            };
            m
        ];
        let moment_arms = vec![0.04; m];
        let mut a_out = vec![0.0; m];
        let mut tau_out = vec![0.0; j];
        step_full(
            &u,
            &a_in,
            0.01,
            &dyns,
            &models,
            &states,
            &moment_arms,
            j,
            &mut a_out,
            &mut tau_out,
        )
        .unwrap();
        assert!(a_out.iter().all(|&a| a > 0.0));
        assert!(tau_out[0] > 0.0);
    }
}
