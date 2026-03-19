# Issue: [MuJoCo] Perturbation Analysis: Core Module Implementation

## Labels
`perturbation-analysis`, `physics-engine`, `mujoco`, `phase-1`

## Summary

Implement perturbation analysis for the MuJoCo physics engine following the unified
`PerturbationAnalyzer` protocol. MuJoCo's fast C-level simulation enables large-scale
Monte Carlo batches, and its native support for actuator models, tendons, and contact
dynamics makes it the most feature-rich engine for studying real-world perturbation
sensitivity.

## Motivation

MuJoCo is the most widely-used engine in the project and supports the richest actuator
and contact models. Its C-level speed enables running thousands of perturbation trials
in minutes, making it ideal for:
- High-resolution sensitivity maps across the torque profile parameter space
- Studying how contact dynamics (ground reaction forces) amplify perturbations
- Evaluating tendon-driven and muscle-driven actuator models under noise
- Leveraging existing `StabilityMetricsMixin` for CoM/CoP postural stability

For golf swing analysis: quantify how actuator noise propagates through a full humanoid
model with ground contact, and how postural stability metrics (CoM-CoP) respond to
torque perturbations — connecting swing consistency to balance.

## Requirements

### Core Protocol Implementation
- [ ] Create `src/engines/physics_engines/mujoco/python/perturbation/analyzer.py`
- [ ] Implement `MuJoCoPerturbationAnalyzer` class conforming to `PerturbationAnalyzer` protocol
- [ ] Use `MuJoCoPhysicsEngine` for simulation
- [ ] Support MJCF/XML model loading via existing engine infrastructure

### Torque Profile Handling
- [ ] Accept torque profiles as:
  - Polynomial coefficients per actuator (for parity with pendulum)
  - Time-series arrays for `data.ctrl` (N_timesteps × n_actuators)
  - Callable `ctrl_func(t, state) → ctrl` for feedback controllers
- [ ] Map actuator names to `data.ctrl` indices via `mj_name2id()`
- [ ] Support different actuator types: motor, position, velocity, muscle

### Perturbation Implementation
- [ ] Import shared noise generation from `src/shared/python/perturbation/noise.py`
- [ ] Additive perturbation on `data.ctrl`: `ctrl_perturbed = ctrl_base + amp × noise`
- [ ] Multiplicative perturbation: `ctrl_perturbed = ctrl_base × (1 + amp × noise)`
- [ ] Per-actuator independent noise with optional correlation
- [ ] Time-varying noise applied at each simulation step
- [ ] Seed-based reproducibility via `np.random.Generator`
- [ ] Optional: perturb `data.qfrc_applied` for direct joint torque perturbation

### Simulation Loop
- [ ] Reset via `mujoco.mj_resetData()` before each trial
- [ ] Set initial state via `data.qpos`, `data.qvel`
- [ ] Step via `mujoco.mj_step()` for each timestep
- [ ] Record trajectory: positions (`data.qpos`), velocities (`data.qvel`),
  controls (`data.ctrl`), sensor data (`data.sensordata`)
- [ ] Handle simulation failures (divergence, warnings, NaN detection)
- [ ] Optional: use `mj_resetDataKeyframe()` for specific starting poses

### Metric Extraction
- [ ] Use `data.xpos[body_id]` for end-effector position
- [ ] Use `data.cvel[body_id]` or finite differences for end-effector velocity
- [ ] Compute all mandatory metrics per §4.2 of guidelines
- [ ] Add MuJoCo-specific optional metrics:
  - `com_position_final` — center of mass position
  - `com_cop_distance` — via existing `StabilityMetricsMixin`
  - `inclination_angle` — postural stability angle
  - `contact_force_total` — total ground reaction force
  - `actuator_force_total` — total actuator effort
  - `sensor_*` — any defined sensor values

### Integration with StabilityMetricsMixin
- [ ] Use existing `compute_stability_metrics()` for CoM/CoP metrics
- [ ] Report stability metrics as optional perturbation outcomes
- [ ] Correlate stability degradation with perturbation amplitude

### Statistics & Reporting
- [ ] Use shared `MetricStatistics` and `variability_summary()` from shared module
- [ ] Compute Robustness Score
- [ ] JSON export per schema in guidelines §8.1
- [ ] HDF5 export for full trajectory data (compact due to large state vectors)

### Comparison
- [ ] Implement `compare_profiles()` for two control strategies
- [ ] Mann-Whitney U test per metric
- [ ] ComparisonReport with CV ratios and winner determination
- [ ] Support comparing different actuator models (motor vs muscle) on same skeleton

### Dashboard Integration
- [ ] Wire into existing `mujoco_dashboard.py` launcher
- [ ] Add perturbation analysis tab/panel to MuJoCo GUI
- [ ] Display real-time progress for batch runs
- [ ] Render histogram of outcome distributions

### Testing
- [ ] Unit test: zero-amplitude → identical results (CV=0)
- [ ] Unit test: seed reproducibility
- [ ] Unit test: monotonicity (amplitude ↑ → CV ↑)
- [ ] Unit test: per-actuator noise independence
- [ ] Integration test: full batch on humanoid golf model
- [ ] Integration test: comparison of two control profiles
- [ ] Validation test: match pendulum engine results on equivalent 2-DOF model
- [ ] Performance test: 1000 trials completes within acceptable time

## Acceptance Criteria

- `MuJoCoPerturbationAnalyzer` passes protocol type check
- All mandatory metrics computed correctly
- StabilityMetricsMixin integration reports CoM/CoP metrics per trial
- JSON export validates against schema
- Results on 2-DOF model match pendulum engine within statistical tolerance
- Batch of 1000 trials on humanoid model completes in < 5 minutes
- All tests pass

## Parity Checklist
- [ ] Implements `PerturbationAnalyzer` protocol
- [ ] Supports white, pink, and brown noise
- [ ] Supports additive and multiplicative perturbation
- [ ] Reports all mandatory metrics (§4.2 of guidelines)
- [ ] Uses `PerturbationConfig` dataclass for configuration
- [ ] Returns `PerturbationSummary` with `MetricStatistics`
- [ ] Reproducible with seed parameter
- [ ] Batch runner with progress reporting
- [ ] Unit tests with synthetic known-sensitivity cases
- [ ] Design by Contract: pre/postconditions documented
- [ ] JSON export compatible with cross-engine comparison schema

## Dependencies
- Issue #006: Pendulum reference implementation (for shared utilities)
- `src/shared/python/perturbation/` shared module must exist

## References
- Guidelines: `docs/perturbation_analysis_parity_guidelines.md`
- MuJoCo engine: `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/physics_engine.py`
- Stability metrics: `src/shared/python/analysis/stability_metrics.py`
- Engine protocol: `src/shared/python/engine_core/interfaces.py`
- MuJoCo documentation: https://mujoco.readthedocs.io/
