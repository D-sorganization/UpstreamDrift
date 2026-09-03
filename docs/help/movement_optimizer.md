---
title: Movement Optimizer
tile_id: movement_optimizer
status: complete
---

# Movement Optimizer

## Purpose

Movement Optimizer computes optimized barbell-exercise trajectories for a
sagittal-plane three-link body model (shin, thigh, trunk) using Lagrangian
inverse dynamics, and reports the joint torques, mechanical power, centre-of-mass
balance and L5/S1 spinal loads implied by the resulting motion.

The tile declares the capabilities `trajectory_optimization`,
`cross_engine_analysis` and `biomechanics`. It is a sibling-repository tile:
the registry gives it `source_root: Movement_Optimizer` and
`path: src/movement_optimizer/__main__.py`, so `resolve_tile_target`
(`src/shared/python/config/tile_target_resolution.py`) resolves it as
`KIND_SIBLING` against a `Movement_Optimizer` checkout beside this one. A
synchronized copy of the same package is vendored in this repository at
`src/shared/python/movement_optimizer/` and is the source read for this page.

## Inputs

`__main__.py` dispatches three modes: `--gui` (the default), `--headless
--exercise <name> [--output <path>]`, and `--list-exercises`.

| Input | Flag | Unit | Default | Accepted range |
| --- | --- | --- | --- | --- |
| Exercise | `--exercise` | identifier | required in headless mode | `squat`, `full_squat`, `deadlift`, `bench_press`, `clean`, `jerk`, `snatch` |
| Body mass | `--body-mass` | kg | 75.0 | 30.0 to 300.0 |
| Height | `--height` | m | 1.75 | 1.4 to 2.2 |
| Bar mass | `--bar-mass` | kg | 60.0 | 0.0 to 500.0 |
| Movement duration | `--duration` | s | 2.0 | 0.5 to 10.0 |
| Smoothness weight | `--smoothness` | dimensionless | 1.0 | 0.1 to 100.0 |
| Output path | `--output` | filesystem path | print to stdout | any writable JSON path |
| Verbosity | `--verbose` | flag | off | - |

Ranges are the `*_RANGE` constants in `validation.py`, enforced by
`validate_all`. The exercise identifier is looked up in `EXERCISE_FACTORIES`
(`cli.py`), which maps each name to a configuration factory in `models.py` or
`exercises/`.

## Outputs

`TrajectoryOptimizer.optimize` returns an `OptimizationResult`
(`trajectory/result.py`), serialized as JSON in headless mode:

| Output | Shape | Unit |
| --- | --- | --- |
| `t` time grid | (N,) | s |
| `q` joint angles | (N, n_dof) | rad |
| `qd` joint velocities | (N, n_dof) | rad/s |
| `qdd` joint accelerations | (N, n_dof) | rad/s^2 |
| `torques` joint torques | (N, n_dof) | N m |
| `power` per-joint mechanical power | (N, n_dof) | W |
| `com` whole-body centre-of-mass path (x, y) | (N, 2) | m |
| `bar` barbell path (x, y) | (N, 2) | m |
| `success` | scalar | boolean (converged and all hard constraints satisfied) |
| `cost` final scalar cost | scalar | dimensionless (lower is better) |
| `com_horizontal_range_cm` peak-to-peak horizontal COM excursion | scalar | cm |
| `elapsed_s` wall-clock optimisation time | scalar | s |
| `n_evals` cost evaluations across all starts | scalar | count |
| `n_joint_limit_violations` samples outside `q_bounds` | scalar | count |

Spinal loading is computed separately by `spine_loads.py`: compression and
anterior-posterior shear at the L5/S1 junction, in newtons, from the
sagittal-plane three-link model. The joint is placed at the base of the torso
segment, which is the hip joint in this model. `NIOSH_COMPRESSION_LIMIT` is
declared as 3400.0 N, the NIOSH recommended compression limit for occupational
lifting, and is the reference the results are read against.

The GUI additionally offers stick-figure animation with playback, trial
comparison overlays, and export to CSV, PNG, PDF and animated GIF.

## Method

The body is a three-link planar chain (shin, thigh, trunk) in the sagittal
plane. Torques are recovered by Lagrangian inverse dynamics; the trajectory is
found by multi-start parallel SLSQP with centre-of-mass balance constraints, so
`success` requires both optimiser convergence and the COM staying inside the
inner base of support with joint angles inside their bounds. A Hill-type muscle
model supplies torque-angle-velocity capacity and sticking-point detection.

Mass distribution above L5/S1 depends on the exercise family: for `squat` and
`full_squat` the torso segment already includes the arms, so `_mass_above_l5`
reads `body.m_squat[2]`; for the deadlift family the torso is trunk plus head
only and it reads `body.m_deadlift[2]`.

An optional Rust extension (`rust_core`, built with PyO3 and maturin)
accelerates the hot-path inverse dynamics; the pure-Python path is used when it
is not built.

The package publishes a `tool_pack/v1` manifest (`tool_pack.yaml`) and a
`biomech.tool_pack` entry point so the launcher can spawn it inside the unified
Biomechanics category, with `manifest()`, `list_exercises()` and
`run_headless()` as the programmatic surface.

Defining modules: `trajectory/` (optimizer and result), `models.py` (body model
and squat, full-squat, deadlift, bench-press configurations), `exercises/`
(clean, jerk, snatch, and additionally gait and sit-to-stand configurations),
`spine_loads.py`, `strength.py` (Hill model), `validation.py`, `cli.py`,
`__main__.py`.

## Limitations

- Two dimensions only. The model is a sagittal-plane three-link chain. The
  3-D selector in the GUI is a disabled placeholder reserved for future work;
  frontal-plane and transverse-plane mechanics, asymmetry and axial rotation
  are outside the model.
- Three links only. Shin, thigh and trunk. There is no separate head, no arm
  chain, no foot segment and no spine articulation beyond the single L5/S1
  estimate.
- Spinal load is an estimate at one level. Compression and anterior-posterior
  shear at L5/S1 only; no lateral shear, no axial torsion, no other vertebral
  level. The NIOSH 3400 N figure is an occupational-lifting limit, not a
  clinical injury threshold for barbell training.
- Not a physics-engine simulation. It does not use MuJoCo, Drake, Pinocchio or
  JaxSim, so its results are not directly comparable with the engine dashboards
  despite the `cross_engine_analysis` capability label.
- Optimizer results are not unique. Multi-start SLSQP is a local method;
  `success` means constraints were satisfied at a converged local optimum, not
  that the trajectory is globally optimal or physiologically preferred.
- No contact or ground-reaction model is exposed. Balance is enforced as a COM
  constraint against a base of support, not through measured or simulated
  ground reaction forces.
- The tile needs a sibling checkout. Without a `Movement_Optimizer` repository
  beside this one, the tile resolves as unavailable on this machine even though
  a vendored copy of the package exists in `src/shared/python/`.

## See Also
- [Exercise Dashboard](biomech_exercise.md)
- [Analysis Tools calculation sheet](analysis_tools_api.md)
- [Biomechanics workspace architecture](../architecture/biomech_workspace.md)
- [Project Map](../architecture/PROJECT_MAP.md)
