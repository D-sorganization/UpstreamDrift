# Issue: [Drake] Perturbation Analysis: Core Module Implementation

## Labels

`perturbation-analysis`, `physics-engine`, `drake`, `phase-1`

## Summary

Implement perturbation analysis for the Drake physics engine following the unified
`PerturbationAnalyzer` protocol. Drake's systems-level architecture enables signal-based
perturbation injection at the actuator input port, providing a natural framework for
studying how control signal noise propagates through the plant.

## Motivation

Drake's `MultibodyPlant` within a `Diagram` systems framework provides unique advantages:

- Signal-level perturbation injection via input ports (natural control systems approach)
- Built-in `Simulator` with configurable integrators and error control
- `LinearQuadraticRegulator` for linearized sensitivity analysis
- Mature contact handling for ground-interaction scenarios

For golf swing analysis: use Drake's systems framework to inject realistic actuator
noise and study how the closed kinematic chain's dynamics amplify or attenuate
perturbations at different phases of the swing.

## Requirements

### Core Protocol Implementation

- [ ] Create `src/engines/physics_engines/drake/python/perturbation/analyzer.py`
- [ ] Implement `DrakePerturbationAnalyzer` class conforming to `PerturbationAnalyzer` protocol
- [ ] Use `DrakePhysicsEngine` with `MultibodyPlant` for simulation
- [ ] Support URDF/SDF model loading via existing engine infrastructure

### Torque Profile Handling

- [ ] Accept torque profiles as:
  - Polynomial coefficients per joint (for parity with pendulum)
  - Time-series arrays via `PiecewisePolynomial` or `TrajectorySource`
  - Callable `torque_func(t) → τ` wrapped in a Drake `VectorSystem`
- [ ] Map joint names to Drake actuator indices via `plant.GetJointActuatorByName()`
- [ ] Validate actuator dimensions match plant configuration

### Signal-Level Perturbation (Drake-specific)

- [ ] Create `PerturbationSource` as a Drake `LeafSystem` that adds noise to control signal
- [ ] Wire: `torque_source → perturbation_source → plant.actuation_input_port`
- [ ] Support per-actuator independent noise injection
- [ ] Allow perturbation to be applied at specific time windows (e.g., only during downswing)

### Perturbation Implementation

- [ ] Import shared noise generation from `src/shared/python/perturbation/noise.py`
- [ ] Additive: inject noise signal added to base torque
- [ ] Multiplicative: inject gain perturbation on base torque
- [ ] Per-joint noise with configurable correlation structure
- [ ] Seed-based reproducibility via `np.random.Generator`

### Simulation Loop

- [ ] Create fresh `Simulator` context per trial (or reset via `SetDefaultContext`)
- [ ] Set initial conditions via `plant.SetPositions()` and `plant.SetVelocities()`
- [ ] Advance simulation with `simulator.AdvanceTo(t_end)`
- [ ] Record trajectory at specified sample rate via `Simulator` publish triggers
- [ ] Handle simulation failures (integrator divergence, contact issues)

### Metric Extraction

- [ ] Use `plant.CalcPointsPositions()` for end-effector position
- [ ] Use `plant.CalcJacobianSpatialVelocity()` for end-effector velocity
- [ ] Compute all mandatory metrics per §4.2 of guidelines
- [ ] Add Drake-specific optional metrics:
  - `contact_force_total` — total contact force magnitude (if applicable)
  - `integrator_error_estimate` — simulation accuracy indicator
  - `constraint_violation` — joint limit or loop closure violation

### Linearized Sensitivity (Bonus — Drake-specific)

- [ ] Linearize plant about nominal trajectory using `Linearize()`
- [ ] Compute input-output sensitivity matrix from linearized system
- [ ] Compare linearized prediction with Monte Carlo results
- [ ] Optionally compute LQR cost-to-go as robustness measure

### Statistics & Reporting

- [ ] Use shared `MetricStatistics` and `variability_summary()` from shared module
- [ ] Compute Robustness Score
- [ ] JSON export per schema in guidelines §8.1
- [ ] HDF5 export for full trajectory data

### Comparison

- [ ] Implement `compare_profiles()` for two torque profiles
- [ ] Mann-Whitney U test per metric
- [ ] ComparisonReport with CV ratios and winner determination

### Testing

- [ ] Unit test: zero-amplitude → identical results (CV=0)
- [ ] Unit test: seed reproducibility
- [ ] Unit test: monotonicity (amplitude ↑ → CV ↑)
- [ ] Unit test: PerturbationSource system produces correct signal
- [ ] Integration test: full batch on double pendulum model
- [ ] Integration test: comparison of two torque profiles
- [ ] Validation test: match pendulum engine results on equivalent 2-DOF model

## Acceptance Criteria

- `DrakePerturbationAnalyzer` passes protocol type check
- `PerturbationSource` LeafSystem correctly injects noise at signal level
- All mandatory metrics computed correctly
- JSON export validates against schema
- Results on 2-DOF model match pendulum engine within statistical tolerance
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
- Drake engine: `src/engines/physics_engines/drake/python/drake_physics_engine.py`
- Engine protocol: `src/shared/python/engine_core/interfaces.py`
- Drake documentation: https://drake.mit.edu/
