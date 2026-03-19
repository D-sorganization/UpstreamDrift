# Issue: [Pendulum Models] Perturbation Analysis: Core Module Formalization

## Labels
`perturbation-analysis`, `physics-engine`, `pendulum`, `phase-1`

## Summary

Formalize and extend the existing pendulum perturbation analysis implementation
(`src/shared/python/pendulum_simulator/perturbation_analysis.py`) to conform to the
unified `PerturbationAnalyzer` protocol defined in the Perturbation Analysis Parity
Guidelines. This serves as the **reference implementation** that all other engines
will follow.

## Motivation

The pendulum simulator already has a working perturbation analysis with noise generation,
batch Monte Carlo, and variability summary. However, it needs to be wrapped in the
unified protocol, extended with multiplicative perturbation mode, and enhanced with the
full mandatory metric set to serve as the gold-standard reference for all other engines.

This enables answering: "Given two golf swing torque profiles on the double pendulum,
which one produces more consistent clubhead speed when joint torques vary?"

## Requirements

### Core Protocol Compliance
- [ ] Create `PendulumPerturbationAnalyzer` class implementing `PerturbationAnalyzer` protocol
- [ ] Wrap existing `batch_perturb_and_simulate()` inside `run_batch()` method
- [ ] Wrap existing `perturb_torque_coeffs()` inside `perturb_torque()` method
- [ ] Implement `set_base_torque_profile()` accepting polynomial coefficient dicts
- [ ] Implement `extract_metrics()` returning all mandatory metrics

### Perturbation Modes
- [ ] Retain existing additive perturbation on polynomial coefficients
- [ ] Add multiplicative perturbation mode: `coeff *= (1 + amplitude * noise)`
- [ ] Add combined (both) mode applying additive then multiplicative
- [ ] Expose mode selection via `PerturbationConfig.perturb_mode`

### Mandatory Metrics
- [ ] `end_effector_position_final` — tip position at motion end (2D for pendulum)
- [ ] `end_effector_velocity_final` — tip velocity at motion end
- [ ] `end_effector_speed_final` — scalar tip speed at motion end
- [ ] `peak_end_effector_speed` — maximum tip speed during motion
- [ ] `total_energy_final` — kinetic + potential energy at end
- [ ] `joint_angles_final` — [θ1, φ] at motion end
- [ ] `joint_velocities_final` — [θ̇1, φ̇] at motion end
- [ ] `trajectory_rmse` — RMSE vs nominal trajectory
- [ ] `trajectory_max_deviation` — max pointwise deviation from nominal
- [ ] `motion_duration` — time to complete (fixed for pendulum, but report it)

### Statistics & Reporting
- [ ] Extend `variability_summary()` to return `MetricStatistics` dataclass
- [ ] Add median, IQR, p5, p95 percentiles to summary
- [ ] Compute Robustness Score (RS = 1/(1+CV_weighted))
- [ ] JSON export matching schema in guidelines §8.1

### Comparison
- [ ] Implement `compare_profiles()` for two torque coefficient sets
- [ ] Statistical test: Mann-Whitney U for each metric
- [ ] Return `ComparisonReport` with winner determination and confidence

### Shared Utilities Extraction
- [ ] Extract `PerturbationConfig`, `PerturbationResult`, `PerturbationSummary`,
  `MetricStatistics` to `src/shared/python/perturbation/config.py`
- [ ] Extract noise generation to `src/shared/python/perturbation/noise.py`
- [ ] Extract statistics computation to `src/shared/python/perturbation/statistics.py`
- [ ] Extract Robustness Score to `src/shared/python/perturbation/robustness_score.py`

### Testing
- [ ] Unit tests: zero-amplitude → CV=0
- [ ] Unit tests: seed reproducibility
- [ ] Unit tests: increasing amplitude → increasing CV (monotonicity)
- [ ] Unit tests: noise distribution characteristics (white/pink/brown)
- [ ] Integration test: full batch run with 100 trials
- [ ] Integration test: comparison report for two known profiles

### GUI Updates
- [ ] Update `perturbation_panel.py` to use new protocol-based analyzer
- [ ] Add perturbation mode selector (additive/multiplicative/both)
- [ ] Add comparison button to run two profiles side-by-side
- [ ] Display Robustness Score in results panel

## Acceptance Criteria

- `PendulumPerturbationAnalyzer` passes protocol type check
- All mandatory metrics computed and validated against hand-calculated values
- Existing GUI perturbation panel works with new analyzer (backward compatible)
- JSON export validates against schema
- All unit and integration tests pass
- Shared utility modules importable by other engine implementations

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
- None (this is the reference implementation)

## References
- Guidelines: `docs/perturbation_analysis_parity_guidelines.md`
- Existing implementation: `src/shared/python/pendulum_simulator/perturbation_analysis.py`
- GUI panel: `src/shared/python/pendulum_simulator/gui/perturbation_panel.py`
- Simulation runner: `src/shared/python/pendulum_simulator/simulation.py`
- Torque utilities: `src/shared/python/pendulum_simulator/torque_utils.py`
