# Issue: [Pinocchio] Perturbation Analysis: Core Module Implementation

## Labels
`perturbation-analysis`, `physics-engine`, `pinocchio`, `phase-1`

## Summary

Implement perturbation analysis for the Pinocchio physics engine following the unified
`PerturbationAnalyzer` protocol. Pinocchio's analytical dynamics capabilities enable
both Monte Carlo perturbation analysis and complementary Jacobian-based sensitivity
analysis for validation.

## Motivation

Pinocchio provides clean, efficient rigid-body dynamics algorithms (RNEA, CRBA, ABA)
with analytical Jacobians. This makes it ideal for:
- Validating Monte Carlo perturbation results against analytical sensitivity predictions
- Computing how small torque changes propagate through the kinematic chain
- Serving as a "clean-room" analytical reference alongside the pendulum implementation

For golf swing analysis: quantify how torque variations at the shoulder vs wrist
differentially affect clubhead speed consistency using Pinocchio's analytical framework.

## Requirements

### Core Protocol Implementation
- [ ] Create `src/engines/physics_engines/pinocchio/python/perturbation/analyzer.py`
- [ ] Implement `PinocchioPerturbationAnalyzer` class conforming to `PerturbationAnalyzer` protocol
- [ ] Use `PinocchioPhysicsEngine` for forward simulation
- [ ] Support URDF model loading via existing engine infrastructure

### Torque Profile Handling
- [ ] Accept torque profiles as:
  - Polynomial coefficients per joint (for parity with pendulum)
  - Time-series arrays τ(t) ∈ ℝ^(N_timesteps × nv)
  - Callable `torque_func(t) → τ` for arbitrary profiles
- [ ] Map joint names/indices to Pinocchio model joint IDs
- [ ] Validate torque dimensions match model DOF

### Perturbation Implementation
- [ ] Import shared noise generation from `src/shared/python/perturbation/noise.py`
- [ ] Additive perturbation: `τ_perturbed(t) = τ_base(t) + amplitude × noise(t)`
- [ ] Multiplicative perturbation: `τ_perturbed(t) = τ_base(t) × (1 + amplitude × noise(t))`
- [ ] Per-joint noise: independent noise stream per DOF
- [ ] Seed-based reproducibility via `np.random.Generator`

### Simulation Loop
- [ ] Reset to initial configuration before each trial
- [ ] Forward integrate using engine's `step()` method
- [ ] Record full trajectory (q, v, a, τ at each timestep)
- [ ] Handle simulation failures gracefully (divergence, NaN)

### Metric Extraction
- [ ] Use `pin.forwardKinematics(model, data, q)` for end-effector position
- [ ] Use `pin.computeFrameJacobian()` × v for end-effector velocity
- [ ] Compute all mandatory metrics per §4.2 of guidelines
- [ ] Add Pinocchio-specific optional metrics:
  - `jacobian_condition_number` — sensitivity indicator at end of motion
  - `manipulability_index` — sqrt(det(J × J^T)) at end of motion

### Analytical Sensitivity (Bonus — Pinocchio-specific)
- [ ] Implement first-order sensitivity: ∂(end_effector)/∂τ via Jacobian
- [ ] Compare analytical sensitivity direction with Monte Carlo dispersion
- [ ] Report agreement metric between analytical and Monte Carlo results
- [ ] Use `pin.computeABADerivatives()` for acceleration sensitivity

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
- [ ] Unit test: analytical vs Monte Carlo sensitivity direction agreement
- [ ] Integration test: full batch on double pendulum URDF
- [ ] Integration test: comparison of two torque profiles
- [ ] Validation test: match pendulum engine results on equivalent 2-DOF model

## Acceptance Criteria

- `PinocchioPerturbationAnalyzer` passes protocol type check
- All mandatory metrics computed correctly (validated against hand calculations)
- Analytical sensitivity direction agrees with Monte Carlo dispersion (cosine similarity > 0.8)
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
- Pinocchio engine: `src/engines/physics_engines/pinocchio/python/pinocchio_physics_engine.py`
- Engine protocol: `src/shared/python/engine_core/interfaces.py`
- Pinocchio documentation: https://stack-of-tasks.github.io/pinocchio/
