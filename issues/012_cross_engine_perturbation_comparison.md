# Issue: [All Engines] Perturbation Analysis: Cross-Engine Comparison Framework

## Labels
`perturbation-analysis`, `cross-engine`, `phase-3`

## Summary

Build the unified cross-engine comparison framework that runs identical perturbation
analyses across all six engines (Pendulum, Pinocchio, Drake, MuJoCo, OpenSim, MyoSuite),
validates consistency of robustness rankings, and produces combined reports and
dashboards for comparing movement strategies across physics backends.

## Motivation

The core question: **"Is a robust movement strategy robust regardless of which physics
engine simulates it?"** If two golf swings are compared and Swing A is more consistent
in MuJoCo, it should also be more consistent in Drake and Pinocchio (for equivalent
rigid-body models). Any disagreement signals a model discrepancy worth investigating.

This framework also enables:
- Selecting the right engine for a given analysis (speed vs fidelity trade-off)
- Validating engine implementations against each other
- Publishing cross-engine robustness benchmarks

## Requirements

### Cross-Engine Runner
- [ ] Create `src/shared/python/perturbation/cross_engine_runner.py`
- [ ] Accept list of engine names and a shared `PerturbationConfig`
- [ ] Load equivalent models in each engine (model mapping registry)
- [ ] Map torque profiles across engine-specific representations
- [ ] Run `PerturbationAnalyzer.run_batch()` on each engine
- [ ] Collect all `PerturbationSummary` results

### Model Equivalence Registry
- [ ] Create mapping: one logical model → per-engine model files
  - e.g., "double_pendulum_golf" → {pinocchio: "dp_golf.urdf", drake: "dp_golf.sdf", ...}
- [ ] Validate kinematic chain equivalence (same DOF, same topology)
- [ ] Validate inertial property equivalence (masses, inertias within tolerance)
- [ ] Report any discrepancies before running comparison

### Torque Profile Translation
- [ ] Convert polynomial coefficients to engine-specific representations
- [ ] Verify that nominal (unperturbed) trajectories match across engines
  (within integration tolerance)
- [ ] Report nominal trajectory divergence as a pre-check

### Consistency Validation
- [ ] Compare Robustness Scores across engines for the same profile
- [ ] Flag engines where RS differs by more than configurable threshold
- [ ] Compare robustness **rankings** (A vs B) across engines
- [ ] Report ranking agreement percentage
- [ ] Statistical test for ranking consistency (Kendall's W)

### Combined Reporting
- [ ] Create `CrossEngineComparisonReport` dataclass:
  - Per-engine summaries
  - Robustness Score table (engines × profiles)
  - Ranking agreement matrix
  - Flagged discrepancies
- [ ] JSON export combining all engine results
- [ ] Markdown summary table for quick review
- [ ] CSV export for external analysis tools

### Dashboard / Visualization
- [ ] Combined histogram overlay: same metric across all engines
- [ ] Robustness Score bar chart: engines side by side
- [ ] Heatmap: metrics × engines showing CV values
- [ ] Scatter plot: RS_engine_A vs RS_engine_B for correlation
- [ ] Integration with existing launcher dashboards (pinocchio_dashboard,
  drake_dashboard, mujoco_dashboard)

### Benchmark Suite
- [ ] Define standard benchmark models and torque profiles
- [ ] Automated regression test: run all engines, check ranking consistency
- [ ] Performance benchmarking: time per trial per engine
- [ ] CI integration: run benchmark on PR to detect regressions

### Testing
- [ ] Unit test: model equivalence checker catches mismatched DOF
- [ ] Unit test: torque profile translation preserves values
- [ ] Integration test: full cross-engine run on 2-DOF pendulum
- [ ] Integration test: ranking consistency for known A-better-than-B case
- [ ] Validation test: all engines agree on zero-perturbation baseline

## Acceptance Criteria

- Cross-engine runner successfully executes on all 6 engines
- Nominal trajectory agreement within 1% RMSE for equivalent rigid-body models
- Robustness ranking consistent across rigid-body engines (Pendulum, Pinocchio, Drake, MuJoCo)
- Combined report generates correctly in JSON, Markdown, and CSV
- Dashboard renders multi-engine visualization
- Benchmark suite runs in CI
- All tests pass

## Dependencies
- Issue #006: Pendulum perturbation analysis (reference)
- Issue #007: Pinocchio perturbation analysis
- Issue #008: Drake perturbation analysis
- Issue #009: MuJoCo perturbation analysis
- Issue #010: OpenSim perturbation analysis
- Issue #011: MyoSuite perturbation analysis

## References
- Guidelines: `docs/perturbation_analysis_parity_guidelines.md`
- Engine protocol: `src/shared/python/engine_core/interfaces.py`
- Existing dashboards: `src/launchers/pinocchio_dashboard.py`, `drake_dashboard.py`, `mujoco_dashboard.py`
