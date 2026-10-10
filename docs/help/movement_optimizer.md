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

| Input             | Flag           | Unit            | Default                   | Accepted range                                                              |
| ----------------- | -------------- | --------------- | ------------------------- | --------------------------------------------------------------------------- |
| Exercise          | `--exercise`   | identifier      | required in headless mode | `squat`, `full_squat`, `deadlift`, `bench_press`, `clean`, `jerk`, `snatch` |
| Body mass         | `--body-mass`  | kg              | 75.0                      | 30.0 to 300.0                                                               |
| Height            | `--height`     | m               | 1.75                      | 1.4 to 2.2                                                                  |
| Bar mass          | `--bar-mass`   | kg              | 60.0                      | 0.0 to 500.0                                                                |
| Movement duration | `--duration`   | s               | 2.0                       | 0.5 to 10.0                                                                 |
| Smoothness weight | `--smoothness` | dimensionless   | 1.0                       | 0.1 to 100.0                                                                |
| Output path       | `--output`     | filesystem path | print to stdout           | any writable JSON path                                                      |
| Verbosity         | `--verbose`    | flag            | off                       | -                                                                           |

Ranges are the `*_RANGE` constants in `validation.py`, enforced by
`validate_all`. The exercise identifier is looked up in `EXERCISE_FACTORIES`
(`cli.py`), which maps each name to a configuration factory in `models.py` or
`exercises/`.

## Outputs

`TrajectoryOptimizer.optimize` returns an `OptimizationResult`
(`trajectory/result.py`), serialized as JSON in headless mode:

| Output                                                          | Shape      | Unit                                                   |
| --------------------------------------------------------------- | ---------- | ------------------------------------------------------ |
| `t` time grid                                                   | (N,)       | s                                                      |
| `q` joint angles                                                | (N, n_dof) | rad                                                    |
| `qd` joint velocities                                           | (N, n_dof) | rad/s                                                  |
| `qdd` joint accelerations                                       | (N, n_dof) | rad/s^2                                                |
| `torques` joint torques                                         | (N, n_dof) | N m                                                    |
| `power` per-joint mechanical power                              | (N, n_dof) | W                                                      |
| `com` whole-body centre-of-mass path (x, y)                     | (N, 2)     | m                                                      |
| `bar` barbell path (x, y)                                       | (N, 2)     | m                                                      |
| `success`                                                       | scalar     | boolean (converged and all hard constraints satisfied) |
| `cost` final scalar cost                                        | scalar     | dimensionless (lower is better)                        |
| `com_horizontal_range_cm` peak-to-peak horizontal COM excursion | scalar     | cm                                                     |
| `elapsed_s` wall-clock optimisation time                        | scalar     | s                                                      |
| `n_evals` cost evaluations across all starts                    | scalar     | count                                                  |
| `n_joint_limit_violations` samples outside `q_bounds`           | scalar     | count                                                  |

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

## Lifting Models: Modelling Reference

This section documents the five classical barbell lifts — back squat, deadlift, bench press, snatch, and clean and jerk — as generated by the four per-engine model packs (`OpenSim_Models`, `MuJoCo_Models`, `Drake_Models`, `Pinocchio_Models`) and as analyzed by Movement Optimizer (above). Each pack also ships `gait` and `sit_to_stand`, out of scope here. Every claim is cited to its source file; anything not found in the sources is marked "not yet documented" rather than guessed.

### Body Model

All four packs build a full-body model from `body_mass` and `height`, scaled to Winter (2009) anthropometrics, per `anthropometrics: winter_2009` in each pack's `model_pack.yaml`. OpenSim's is a 15-segment body (pelvis, torso, head, bilateral arms/legs) with ball joints at hip/shoulder/lumbar, pin joints at knee/elbow/neck, and custom 2-DOF joints at ankle/wrist (`OpenSim_Models/src/opensim_models/shared/body/body_model.py`). Every pack gives the pelvis an unconstrained 6-DOF root joint to the world: OpenSim's `FreeJoint` named `ground_pelvis` (`.../shared/body/axial_skeleton.py`), MuJoCo's pelvis `freejoint` (`MuJoCo_Models/.../shared/body/axial_skeleton.py`), Drake's `floating` pelvis joint, configurable to `fixed` (`Drake_Models/src/drake_models/shared/body/body_model.py`), and Pinocchio's root built with `pin.JointModelFreeFlyer()` (`Pinocchio_Models/src/pinocchio_models/exercises/base.py`). Movement Optimizer uses a different, simpler body: a planar three-link shin/thigh/trunk chain (see "Method" above), not these full-body models.

### Barbell and Plates

Each pack models an Olympic barbell to IWF/IPF dimensions: a men's bar is 2.20 m long, 1.31 m between collars, 28 mm shaft, 50 mm sleeves, 20 kg unloaded; a women's bar is 2.01 m, 25 mm shaft, 15 kg (`OpenSim_Models/src/opensim_models/shared/barbell/barbell_model.py`, `BarbellSpec`). The bar is three rigid bodies — left sleeve, shaft, right sleeve — joined by welds, with plates added as sleeve mass via `plate_mass_per_side` (same file). MuJoCo, Drake and Pinocchio each carry an equivalent `BarbellSpec`/barbell builder under their own `shared/barbell/`.

### Grip Interface

The bar attaches to the hands with a rigid weld/fixed joint, not a compliant contact. OpenSim's `attach_barbell_to_hands` welds both hands to the shaft at a configurable `grip_offset` half-width (`OpenSim_Models/src/opensim_models/exercises/base.py`); MuJoCo's `add_weld_constraint` does the same (`MuJoCo_Models/src/mujoco_models/exercises/base.py`). Drake's base builder currently welds the barbell to the left hand only — its own docstring calls this weld "SDF 1.8 kinematic-tree-safe" for a single attachment point (`Drake_Models/src/drake_models/exercises/base.py`); Pinocchio's default `attach_barbell` also welds via `add_fixed_joint` (`Pinocchio_Models/src/pinocchio_models/exercises/base.py`). Grip half-width is exercise-specific (OpenSim's snatch ~0.58 m, clean ~0.25 m, see `.../exercises/constants.py`). The audit linked below tracks where the right-hand weld is missing and where grip half-width mismatches hand spacing.

### Foot and Bench Contact

OpenSim uses four Hunt-Crossley contact spheres per foot (heel/toe, medial/lateral) against a ground half-space (`OpenSim_Models/src/opensim_models/shared/body/foot_contact.py`). For the bench press it adds a near-zero-mass bench body welded to ground, with the pelvis welded supine on it at the IPF-standard 0.43 m bench height (`.../exercises/bench_press/bench_press_model.py`). MuJoCo places a ground plane geom in `worldbody` and uses its built-in contact solver, with `<exclude>` pairs to suppress adjacent-segment self-collision (`MuJoCo_Models/src/mujoco_models/exercises/base.py`). Pinocchio defines contact frames and a Coulomb friction coefficient for its own constraint-based contact — a rectangular foot sole (0.26 m x 0.10 m) with corner points — rather than built-in collision detection (`Pinocchio_Models/src/pinocchio_models/shared/contact/contact_model.py`). Do not assume contact-_force_ parity from these geometry definitions alone; see the audit below.

### Phase Definitions and Start Poses

MuJoCo's optimizer layer defines each lift as a named `Phase(name, t, pose)` sequence (`MuJoCo_Models/src/mujoco_models/optimization/objective_data/*.py`): back squat is `descent_start -> half_depth -> bottom -> drive -> lockout`; deadlift is `floor -> below_knee -> above_knee -> lockout`; snatch is `first_pull -> transition -> second_pull -> turnover -> catch -> recovery`; clean and jerk is `clean_pull -> clean_transition -> clean_extension -> clean_catch -> jerk_dip -> jerk_drive -> jerk_catch -> recovery`. Floor-pull start poses (deadlift, snatch, clean and jerk) share ~80 degree hip flexion and ~60 degree knee flexion defaults in OpenSim and MuJoCo (`.../exercises/constants.py`, `MuJoCo_Models/.../exercises/base.py`). A shared phase/root-joint standard across all four engines does not exist yet (Repository_Management#2025, #2024, per GitHub issue #11740) and is also an audit finding in the chapter linked below.

### What the Optimizer Computes

Movement Optimizer (`Tools/src/movement_optimizer`, vendored at `src/shared/python/movement_optimizer/`) is the only tool here producing full lift trajectories with inverse dynamics: joint angles, velocities, accelerations, torques, power, and centre-of-mass/bar paths from multi-start SLSQP under balance and joint-limit constraints (see "Method" above). Its Hill-type muscle model gives torque-angle-velocity capacity and sticking-point detection for its single three-link chain; it does not distribute load across named muscles. None of the four engine packs define muscle actuators: a search of `OpenSim_Models/src/opensim_models` for `Muscle`/`Thelen`/`Millard`/`actuator` classes found none, consistent with the epic's note that "muscle content in the OpenSim pack is unverified" (GitHub issue #11740). Full-body musculoskeletal lifts with muscle forces are future work under LIFT-6 (GitHub issue #11746); muscle-redundancy solving is not yet documented anywhere in this fleet.

### Spine Load

Movement Optimizer is the only source of a spine-load figure. It computes compression and anterior-posterior shear at the L5/S1 junction from its three-link model, with the joint at the base of the torso segment (the hip joint in this model), reported against `NIOSH_COMPRESSION_LIMIT = 3400.0 N` — an occupational-lifting limit, not a clinical injury threshold (see "Outputs"/"Limitations" above, `spine_loads.py`). None of the four engine packs compute a spine load; the chapter below notes they are not evaluated for joint moments or spine loads at all. LIFT-7 (GitHub issue #11747) tracks adding cross-engine spine-load analysis.

### Known Parity Findings and Limitations

A cross-engine audit compares the four packs on the same inputs (reference lifter 80 kg / 1.78 m, competition bar plus 100 kg) in `manuals/upstreamdrift/chapters/15-lift-pack-audit.qmd`. That chapter is evidence class "derived, provisional" and not approved; read it, and its recorded baseline at `docs/development/lifting/PACK_PARITY_BASELINE.md`, for the measured discrepancies — segment/hand/foot position parity, bar-centre offsets between packs, lifter-only centre-of-mass differences, which packs weld only one hand to the bar, grip-width mismatch, and floor-pull start poses that miss the plate radius. Discrepancies are derived programmatically (`src/shared/python/lifting/pack_audit/gaps.py`) and each maps to a tracked issue in its pack repository or a LIFT child of epic #11740. The chapter also records what it does not cover: body-frame origins only, no contact force or dynamics, and no lift motion-capture or force-plate data exist to validate any pack against.

### Where to Find Each Pack

| Pack      | Repository                         | Exercises root                    | Manifest          |
| --------- | ---------------------------------- | --------------------------------- | ----------------- |
| OpenSim   | `D-sorganization/OpenSim_Models`   | `src/opensim_models/exercises/`   | `model_pack.yaml` |
| MuJoCo    | `D-sorganization/MuJoCo_Models`    | `src/mujoco_models/exercises/`    | `model_pack.yaml` |
| Drake     | `D-sorganization/Drake_Models`     | `src/drake_models/exercises/`     | `model_pack.yaml` |
| Pinocchio | `D-sorganization/Pinocchio_Models` | `src/pinocchio_models/exercises/` | `model_pack.yaml` |

Each root `model_pack.yaml` declares the `model_pack/v1` schema, engine, engine-version constraint, anthropometrics source, and exercise IDs with paths (all four read directly for this section). Movement Optimizer lives at `Tools/src/movement_optimizer`, vendored here at `src/shared/python/movement_optimizer/`, documented in full above.

## See Also

- [Exercise Dashboard](biomech_exercise.md)
- [Analysis Tools calculation sheet](analysis_tools_api.md)
- [Biomechanics workspace architecture](../architecture/biomech_workspace.md)
- [Project Map](../architecture/PROJECT_MAP.md)
- [Lift pack audit design-manual chapter](../../manuals/upstreamdrift/chapters/15-lift-pack-audit.qmd)
- [Lift model packs exploratory modelling reference (LaTeX)](../research/lift_models/README.md)
