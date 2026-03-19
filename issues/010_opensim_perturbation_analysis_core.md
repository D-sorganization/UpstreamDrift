# Issue: [OpenSim] Perturbation Analysis: Core Module Implementation

## Labels
`perturbation-analysis`, `physics-engine`, `opensim`, `phase-1`

## Summary

Implement perturbation analysis for the OpenSim physics engine following the unified
`PerturbationAnalyzer` protocol. OpenSim's musculoskeletal modeling adds a unique
dimension: perturbation can be applied at the **muscle excitation** level (physiological)
in addition to joint torques, revealing how muscle redundancy affects movement consistency.

## Motivation

OpenSim is the gold standard for musculoskeletal simulation. Its muscle-driven dynamics
introduce a critical layer between neural commands and joint torques:

```
Neural excitation → Muscle activation → Muscle force → Joint torque → Motion
```

This redundancy (many muscles per joint) means the same joint torque can arise from
different muscle activation patterns. Perturbation analysis at the muscle level reveals:
- Which muscle activation patterns are more robust to neural noise
- How muscle co-contraction affects movement consistency
- Whether physiological movement strategies are inherently more stable
- The metabolic cost of robustness (co-contraction costs energy)

For golf swing analysis: compare two players' muscle activation strategies — a player
who achieves consistent clubhead speed through high co-contraction (metabolically
expensive but robust) vs one using minimal activation (efficient but sensitive).

## Requirements

### Core Protocol Implementation
- [ ] Create `src/engines/physics_engines/opensim/python/perturbation/analyzer.py`
- [ ] Implement `OpenSimPerturbationAnalyzer` class conforming to `PerturbationAnalyzer` protocol
- [ ] Use `OpenSimPhysicsEngine` for musculoskeletal simulation
- [ ] Support .osim model loading via existing engine infrastructure

### Torque Profile Handling (Dual Mode)
- [ ] **Joint torque mode** (for parity):
  - Polynomial coefficients per joint
  - Time-series joint torque arrays
  - Perturbed the same way as other engines
- [ ] **Muscle excitation mode** (OpenSim-specific):
  - Excitation profiles e(t) ∈ [0,1]^n_muscles per muscle
  - Time-series or polynomial excitation curves
  - Clamped to physiological range [0, 1] after perturbation
- [ ] Map muscle names to OpenSim muscle indices
- [ ] Map joint names to OpenSim coordinate indices

### Perturbation at Muscle Level
- [ ] Additive noise on excitation: `e_perturbed = clamp(e_base + amp × noise, 0, 1)`
- [ ] Multiplicative noise: `e_perturbed = clamp(e_base × (1 + amp × noise), 0, 1)`
- [ ] Per-muscle independent noise
- [ ] Optional: correlated noise across synergist muscle groups
- [ ] Respect physiological bounds (excitation ∈ [0, 1])

### Perturbation at Joint Level
- [ ] Same additive/multiplicative as other engines
- [ ] Applied via `prescribedForce` or reserve actuators
- [ ] Enables direct comparison with non-musculoskeletal engines

### Simulation Loop
- [ ] Reset model state before each trial
- [ ] Set initial coordinates and speeds
- [ ] Apply perturbed excitations/torques via OpenSim controllers
- [ ] Forward integrate using OpenSim's `Manager`
- [ ] Record trajectory: coordinates, speeds, muscle states, forces
- [ ] Handle simulation failures (muscle equilibrium divergence, etc.)

### Metric Extraction
- [ ] Compute all mandatory metrics per §4.2 of guidelines
- [ ] Use OpenSim's `BodyKinematics` analysis for end-effector tracking
- [ ] Add OpenSim-specific optional metrics:
  - `muscle_activation_final` — activation vector at end of motion
  - `muscle_activation_mean` — mean activation during motion (per muscle)
  - `co_contraction_index` — ratio of antagonist to agonist activation
  - `metabolic_cost` — total metabolic energy expenditure
  - `reserve_actuator_force` — residual force needed (model quality indicator)
  - `muscle_force_cv` — per-muscle force variability across trials
  - `tendon_slack_length_sensitivity` — how tendon properties affect outcome

### Statistics & Reporting
- [ ] Use shared `MetricStatistics` and `variability_summary()` from shared module
- [ ] Compute Robustness Score
- [ ] Separate RS for joint-level and muscle-level perturbation
- [ ] JSON export per schema in guidelines §8.1
- [ ] Include muscle-level statistics in export

### Comparison
- [ ] Implement `compare_profiles()` for two excitation/torque profiles
- [ ] Compare joint-level consistency vs muscle-level consistency
- [ ] Mann-Whitney U test per metric
- [ ] ComparisonReport with muscle-level breakdown
- [ ] "Robustness cost" metric: RS × metabolic_cost trade-off

### Testing
- [ ] Unit test: zero-amplitude → identical results (CV=0)
- [ ] Unit test: seed reproducibility
- [ ] Unit test: monotonicity (amplitude ↑ → CV ↑)
- [ ] Unit test: muscle excitation clamping to [0, 1]
- [ ] Unit test: joint-level perturbation matches other engines on simple model
- [ ] Integration test: full batch on arm26 or gait model
- [ ] Integration test: muscle-level vs joint-level perturbation comparison
- [ ] Validation test: match pendulum engine results on equivalent 2-DOF model
  (joint torque mode only)

## Acceptance Criteria

- `OpenSimPerturbationAnalyzer` passes protocol type check
- Both joint-torque and muscle-excitation perturbation modes functional
- Muscle excitations always clamped to [0, 1] after perturbation
- All mandatory metrics computed correctly
- Muscle-specific metrics (co-contraction, metabolic cost) reported
- JSON export validates against schema with muscle-level extensions
- Joint-level results on simple model match pendulum engine within tolerance
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
- OpenSim engine: `src/engines/physics_engines/opensim/python/opensim_physics_engine.py`
- OpenSim muscle analysis: `src/engines/physics_engines/opensim/python/muscle_analysis.py`
- Engine protocol: `src/shared/python/engine_core/interfaces.py`
- OpenSim documentation: https://opensim.stanford.edu/
