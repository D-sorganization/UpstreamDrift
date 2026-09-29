# Perturbation Analysis Parity Guidelines

## Comprehensive Framework for Cross-Engine Perturbation & Sensitivity Analysis

**Version**: 1.0
**Date**: 2026-03-19
**Scope**: Pinocchio, Drake, MuJoCo, Pendulum Models, OpenSim, MyoSuite

---

## 1. Executive Summary

This document defines the guidelines for implementing perturbation analysis across all
six physics engine backends in UpstreamDrift. The goal is to quantify **movement strategy
robustness** — answering questions like "why is one golf swing more consistent than
another?" — by applying controlled perturbations to joint torque inputs and measuring
how simulation outputs diverge.

All engines must implement the same analysis interface, produce comparable metrics, and
support cross-engine comparison through a unified reporting layer.

---

## 2. Motivation & Use Cases

### 2.1 Core Question

Given two movement solutions (e.g., two golf swing torque profiles), which one produces
more consistent outcomes when subjected to small, realistic variations in joint torque?

### 2.2 Applications

- **Golf swing consistency**: Compare amateur vs professional torque profiles for
  sensitivity to perturbation
- **Rehabilitation**: Identify movement strategies that are robust to muscle weakness
- **Robot control**: Evaluate control policies for tolerance to actuator noise
- **Movement optimization**: Add stability-against-perturbation as an objective in
  trajectory optimization
- **Cross-engine validation**: Verify that sensitivity rankings are physics-engine
  independent (a robust swing in MuJoCo should also be robust in Drake)

---

## 3. Existing Implementation Reference

The pendulum simulator (`src/shared/python/pendulum_simulator/perturbation_analysis.py`)
provides the reference implementation with:

- **Noise generation**: White, pink, and brown noise types
- **Torque coefficient perturbation**: Additive noise on polynomial coefficients
- **Batch Monte Carlo simulation**: N trials with perturbed inputs
- **Variability summary**: Mean, std, CV, min, max of outcome metrics
- **GUI panel**: `perturbation_panel.py` with threading, progress, and histogram display

All new implementations must maintain **functional parity** with this reference while
adapting to engine-specific capabilities.

---

## 4. Unified Perturbation Analysis Interface

### 4.1 Core Protocol

Every engine's perturbation module MUST implement this protocol:

```python
from typing import Protocol, Any
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration for perturbation analysis runs."""
    n_trials: int              # Number of Monte Carlo trials (default: 100)
    noise_type: str            # 'white', 'pink', or 'brown'
    noise_amplitude: float     # Relative amplitude (e.g., 0.05 = 5%)
    seed: int | None           # Random seed for reproducibility (None = random)
    perturb_mode: str          # 'additive', 'multiplicative', or 'both'
    metric_keys: list[str]     # Which outcome metrics to track

@dataclass
class PerturbationResult:
    """Results from a single perturbation trial."""
    trial_id: int
    perturbed_input: dict[str, np.ndarray]  # The perturbed torque profile
    metrics: dict[str, float]               # Outcome metrics for this trial
    trajectory: np.ndarray | None           # Optional full trajectory (N x state_dim)
    success: bool                           # Whether simulation completed

@dataclass
class PerturbationSummary:
    """Aggregate statistics across all trials."""
    config: PerturbationConfig
    engine_name: str
    n_successful: int
    n_failed: int
    metric_stats: dict[str, MetricStatistics]  # Per-metric statistics
    failure_rate: float                         # n_failed / n_trials
    raw_results: list[PerturbationResult]       # All individual results

@dataclass(frozen=True)
class MetricStatistics:
    """Statistical summary for a single metric."""
    mean: float
    std: float
    cv: float       # Coefficient of variation (std/|mean|)
    median: float
    iqr: float      # Interquartile range
    min: float
    max: float
    p5: float       # 5th percentile
    p95: float      # 95th percentile
    values: np.ndarray  # All raw values for custom analysis

class PerturbationAnalyzer(Protocol):
    """Protocol that every engine perturbation module must satisfy."""

    def configure(self, config: PerturbationConfig) -> None:
        """Set up the analysis configuration."""
        ...

    def set_base_torque_profile(
        self, profile: dict[str, Any]
    ) -> None:
        """Set the nominal (unperturbed) torque input.

        The profile format is engine-specific but must include:
        - Joint names/indices mapping
        - Torque values (coefficients, spline points, or time series)
        """
        ...

    def perturb_torque(
        self, base_profile: dict[str, Any], rng: np.random.Generator
    ) -> dict[str, Any]:
        """Generate a single perturbed torque profile.

        Preconditions:
            - base_profile is valid for this engine
            - rng is a seeded Generator instance
        Postconditions:
            - Return value has same structure as base_profile
            - Perturbation magnitude is bounded by config.noise_amplitude
        """
        ...

    def run_single_trial(
        self, trial_id: int, perturbed_profile: dict[str, Any]
    ) -> PerturbationResult:
        """Execute one simulation with perturbed inputs.

        Must reset engine state before each trial.
        """
        ...

    def extract_metrics(
        self, trajectory: np.ndarray, engine_state: Any
    ) -> dict[str, float]:
        """Extract outcome metrics from a completed simulation.

        Required metrics (all engines):
            - end_effector_position_final: 3D position at end of motion
            - end_effector_velocity_final: 3D velocity at end of motion
            - end_effector_speed_final: scalar speed at end of motion
            - peak_end_effector_speed: maximum speed during motion
            - total_energy_final: total mechanical energy at end
            - joint_angle_final: joint angles at end (per-joint)
            - trajectory_rmse: RMSE vs nominal trajectory

        Optional metrics (engine-specific):
            - muscle_activation_* (OpenSim, MyoSuite)
            - contact_force_* (MuJoCo, Drake)
            - com_cop_distance (humanoid models)
        """
        ...

    def run_batch(self) -> PerturbationSummary:
        """Execute full batch of n_trials perturbation simulations.

        Returns aggregate statistics and all individual results.
        """
        ...

    def compare_profiles(
        self,
        profile_a: dict[str, Any],
        profile_b: dict[str, Any],
        config: PerturbationConfig | None = None,
    ) -> dict[str, Any]:
        """Compare two movement strategies for robustness.

        Returns:
            - summary_a: PerturbationSummary for profile A
            - summary_b: PerturbationSummary for profile B
            - comparison: dict with relative robustness metrics
                - cv_ratio: CV_A / CV_B per metric (>1 means A is less consistent)
                - failure_rate_ratio
                - statistical_tests: p-values from Mann-Whitney U or similar
        """
        ...
```

### 4.2 Required Outcome Metrics (Mandatory for All Engines)

| Metric Key                    | Description                              | Units               |
| ----------------------------- | ---------------------------------------- | ------------------- |
| `end_effector_position_final` | End-effector 3D position at motion end   | meters (3D vector)  |
| `end_effector_velocity_final` | End-effector 3D velocity at motion end   | m/s (3D vector)     |
| `end_effector_speed_final`    | End-effector scalar speed at motion end  | m/s                 |
| `peak_end_effector_speed`     | Maximum end-effector speed during motion | m/s                 |
| `total_energy_final`          | Total mechanical energy at motion end    | Joules              |
| `joint_angles_final`          | All joint angles at motion end           | radians (per-joint) |
| `joint_velocities_final`      | All joint velocities at motion end       | rad/s (per-joint)   |
| `trajectory_rmse`             | RMSE of perturbed vs nominal trajectory  | mixed (state units) |
| `trajectory_max_deviation`    | Max deviation from nominal trajectory    | mixed (state units) |
| `motion_duration`             | Time to complete motion (if variable)    | seconds             |

### 4.3 Required Noise Models (All Engines Must Support)

1. **White noise**: Independent Gaussian samples — models random actuator error
2. **Pink noise (1/f)**: Correlated low-frequency drift — models systematic bias
3. **Brown noise**: Integrated white noise — models slow parameter drift

### 4.4 Required Perturbation Modes

1. **Additive**: `τ_perturbed = τ_base + amplitude × noise`
2. **Multiplicative**: `τ_perturbed = τ_base × (1 + amplitude × noise)`
3. **Both**: Apply additive and multiplicative simultaneously

---

## 5. Engine-Specific Implementation Notes

### 5.1 Pendulum Models (Reference Implementation)

**Location**: `src/engines/physics_engines/pendulum/`
**Existing work**: `src/shared/python/pendulum_simulator/perturbation_analysis.py`

**Approach**:

- Extend existing `batch_perturb_and_simulate()` to match unified protocol
- Add multiplicative perturbation mode
- Add trajectory RMSE and max deviation metrics
- Wrap in `PendulumPerturbationAnalyzer` class implementing protocol
- Leverage existing `make_polynomial_torque()` for coefficient perturbation
- Adapt GUI panel to new protocol interface

**Torque representation**: Polynomial coefficients per joint
**End-effector**: Tip of last pendulum segment (clubhead for golf model)
**State**: [θ1, φ, θ̇1, φ̇] — compact, well-understood

### 5.2 Pinocchio

**Location**: `src/engines/physics_engines/pinocchio/`

**Approach**:

- Create `pinocchio_perturbation_analysis.py` in engine directory
- Use `PinocchioPhysicsEngine` for simulation (inherits `BasePhysicsEngine`)
- Perturb torque profiles via `compute_inverse_dynamics()` → add noise → forward simulate
- Use Pinocchio's `pin.forwardKinematics()` for end-effector tracking
- Leverage `pin.computeJointJacobians()` for sensitivity analysis (Jacobian-based)

**Torque representation**: Time-series τ(t) ∈ ℝ^nv or polynomial coefficients
**End-effector**: Specified body frame (e.g., hand or club)
**Unique capability**: Analytical Jacobian-based sensitivity (complement Monte Carlo)

### 5.3 Drake

**Location**: `src/engines/physics_engines/drake/`

**Approach**:

- Create `drake_perturbation_analysis.py` in engine directory
- Use `DrakePhysicsEngine` with `MultibodyPlant` for simulation
- Perturb via `plant.get_actuation_input_port()` signal injection
- Use Drake's `CalcPointsPositions()` for end-effector tracking
- Leverage Drake's `LinearQuadraticRegulator` for linearized sensitivity (optional)

**Torque representation**: Time-series via Drake input port or polynomial
**End-effector**: Specified body frame queried from plant
**Unique capability**: Drake's systems framework enables signal-level perturbation injection

### 5.4 MuJoCo

**Location**: `src/engines/physics_engines/mujoco/`

**Approach**:

- Create `mujoco_perturbation_analysis.py` in engine directory
- Use `MuJoCoPhysicsEngine` for simulation
- Perturb `data.ctrl` (control inputs) directly
- Use `mj_name2id()` + `data.xpos`/`data.xvelp` for end-effector tracking
- Leverage existing `StabilityMetricsMixin` for CoM/CoP metrics

**Torque representation**: `data.ctrl` array (direct actuator commands)
**End-effector**: Named body site or geom
**Unique capability**: Fast C-level simulation, contact dynamics, muscle actuators

### 5.5 OpenSim

**Location**: `src/engines/physics_engines/opensim/`

**Approach**:

- Create `opensim_perturbation_analysis.py` in engine directory
- Use `OpenSimPhysicsEngine` for musculoskeletal simulation
- Perturb **muscle excitations** (not raw torques) — more physiologically meaningful
- Also support joint torque perturbation for parity
- Track both joint-level and muscle-level outcome metrics

**Torque representation**: Muscle excitations e(t) ∈ [0,1]^n_muscles OR joint torques
**End-effector**: Marker or body in musculoskeletal model
**Unique metrics**: Muscle activation patterns, metabolic cost, co-contraction index
**Unique capability**: Muscle redundancy means same joint torque can come from different
muscle activation patterns — perturbation analysis reveals which activations are more
robust

### 5.6 MyoSuite

**Location**: `src/engines/physics_engines/myosuite/`

**Approach**:

- Create `myosuite_perturbation_analysis.py` in engine directory
- Use `MyoSuitePhysicsEngine` for muscle-driven simulation
- Perturb muscle activations via environment action space
- Track muscle-specific and joint-level metrics
- Interface with MyoSuite's gym-style `env.step()` API

**Torque representation**: Muscle activation actions a(t) ∈ [0,1]^n_muscles
**End-effector**: Task-specific (e.g., fingertip, hand)
**Unique metrics**: Muscle fatigue, activation smoothness, synergy decomposition
**Unique capability**: RL-trained policies can be evaluated for perturbation robustness

---

## 6. Cross-Engine Comparison Framework

### 6.1 Comparison Protocol

To compare the same movement solution across engines, all engines must:

1. **Load equivalent models**: Same kinematic chain, same inertial properties
2. **Apply equivalent torque profiles**: Mapped through engine-specific representations
3. **Run identical perturbation configs**: Same n_trials, noise_type, amplitude, seed
4. **Report standardized metrics**: Using the mandatory metric set from §4.2
5. **Produce comparable output format**: JSON/HDF5 with consistent schema

### 6.2 Robustness Score

A unified **Robustness Score (RS)** enables direct comparison:

```
RS = 1 / (1 + CV_weighted)

where CV_weighted = Σ_i w_i × CV_i

CV_i = coefficient of variation for metric i
w_i  = weight for metric i (configurable, default uniform)
```

Properties:

- RS ∈ (0, 1] — higher is more robust
- RS = 1 means zero variation (perfectly consistent)
- RS ≈ 0 means extreme variation (highly sensitive)

### 6.3 Pairwise Comparison Report

When comparing two movement strategies A and B:

```python
@dataclass
class ComparisonReport:
    engine_name: str
    profile_a_label: str
    profile_b_label: str
    summary_a: PerturbationSummary
    summary_b: PerturbationSummary
    robustness_score_a: float
    robustness_score_b: float
    cv_ratios: dict[str, float]          # CV_A / CV_B per metric
    statistical_tests: dict[str, float]   # p-values per metric
    winner: str                           # 'A', 'B', or 'inconclusive'
    confidence: float                     # Confidence in winner determination
```

### 6.4 Cross-Engine Consistency Check

Run the same comparison across engines and verify:

- Robustness ranking (A vs B) is consistent across engines
- CV magnitudes are within expected scaling factors
- Flag any engine where the ranking disagrees (potential model discrepancy)

---

## 7. GitHub Issue Generation Guidelines

### 7.1 Issue Structure (Required Sections)

Every perturbation analysis issue MUST include:

1. **Title**: `[Engine] Perturbation Analysis: <specific scope>`
2. **Labels**: `perturbation-analysis`, `physics-engine`, engine-specific label
3. **Summary**: 2-3 sentence description of the task
4. **Motivation**: Why this is needed, what questions it answers
5. **Requirements**: Checklist of specific deliverables
6. **Acceptance Criteria**: How to verify the implementation is correct
7. **Parity Checklist**: Cross-reference to unified interface compliance
8. **Dependencies**: Links to prerequisite issues
9. **References**: Links to this guidelines document and reference implementation

### 7.2 Issue Naming Convention

```
[<Engine>] Perturbation Analysis: <Phase Description>
```

Examples:

- `[Pinocchio] Perturbation Analysis: Core Module Implementation`
- `[Drake] Perturbation Analysis: Cross-Engine Comparison Integration`
- `[All Engines] Perturbation Analysis: Unified Comparison Dashboard`

### 7.3 Issue Phasing (Per Engine)

Each engine should have issues in this order:

**Phase 1 — Core Module** (one issue per engine):

- Implement `PerturbationAnalyzer` protocol for the engine
- Noise generation (white, pink, brown)
- Additive and multiplicative perturbation modes
- Batch Monte Carlo runner
- Variability summary with all mandatory metrics
- Unit tests with known-sensitivity test cases

**Phase 2 — Comparison & Reporting** (one issue per engine):

- `compare_profiles()` implementation
- Robustness Score computation
- Statistical tests (Mann-Whitney U, bootstrap CI)
- JSON/HDF5 export of results
- Integration with existing engine dashboard

**Phase 3 — Cross-Engine Integration** (shared issues):

- Unified comparison runner across all engines
- Cross-engine consistency validation
- Combined dashboard / reporting UI
- Benchmark suite for regression testing

### 7.4 Parity Checklist (Include in Every Issue)

```markdown
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
```

### 7.5 Cross-Engine Validation Test Cases

Every engine must pass these validation scenarios:

1. **Zero perturbation**: amplitude=0 → CV=0 for all metrics
2. **Symmetry**: Results statistically equivalent for seed S and seed S (reproducibility)
3. **Monotonicity**: Increasing amplitude → increasing CV (on average)
4. **Known sensitivity**: For a 2-DOF system with analytical solution, verify CV matches
   theoretical prediction within tolerance

---

## 8. Data Schema for Results

### 8.1 JSON Export Format

```json
{
  "schema_version": "1.0",
  "engine": "pinocchio",
  "model": "double_pendulum_golf",
  "config": {
    "n_trials": 200,
    "noise_type": "white",
    "noise_amplitude": 0.05,
    "perturb_mode": "additive",
    "seed": 42
  },
  "summary": {
    "n_successful": 198,
    "n_failed": 2,
    "failure_rate": 0.01,
    "metrics": {
      "end_effector_speed_final": {
        "mean": 45.2,
        "std": 1.8,
        "cv": 0.0398,
        "median": 45.1,
        "iqr": 2.4,
        "min": 40.1,
        "max": 50.3,
        "p5": 42.1,
        "p95": 48.3
      }
    },
    "robustness_score": 0.962
  },
  "trials": [
    {
      "trial_id": 0,
      "success": true,
      "metrics": { "end_effector_speed_final": 45.3 },
      "perturbed_input_hash": "abc123"
    }
  ]
}
```

### 8.2 HDF5 Structure (for trajectory data)

```
/perturbation_analysis/
    /config          (attributes: n_trials, noise_type, etc.)
    /base_profile    (dataset: nominal torque profile)
    /trials/
        /0/
            /perturbed_input   (dataset: perturbed torque)
            /trajectory        (dataset: N x state_dim)
            /metrics           (attributes: metric_name → value)
        /1/
            ...
    /summary/
        /metric_values     (dataset: n_trials x n_metrics)
        /metric_stats      (attributes per metric)
```

---

## 9. Testing Strategy

### 9.1 Unit Tests (Per Engine)

- Noise generator produces correct distribution characteristics
- Perturbation preserves torque profile structure
- Zero-amplitude perturbation returns identical results
- Metrics computation matches hand-calculated values
- Seed reproducibility: same seed → identical results

### 9.2 Integration Tests (Per Engine)

- Full batch run completes without errors
- Results schema validates against JSON schema
- Comparison report generates correctly for two profiles
- Dashboard integration renders histogram and statistics

### 9.3 Cross-Engine Tests (Shared)

- Same 2-DOF model in all engines produces consistent sensitivity rankings
- Robustness Score ordering is preserved across engines
- Data export/import round-trips correctly between engines
- Combined dashboard displays multi-engine results correctly

---

## 10. Implementation Priority

| Priority | Engine          | Rationale                                                    |
| -------- | --------------- | ------------------------------------------------------------ |
| 1        | Pendulum Models | Reference implementation exists; extend and formalize        |
| 2        | Pinocchio       | Clean analytical dynamics; good for validation               |
| 3        | MuJoCo          | Most widely used; fast simulation enables large batches      |
| 4        | Drake           | Strong systems framework; good for signal-level perturbation |
| 5        | OpenSim         | Muscle-level perturbation adds unique physiological insight  |
| 6        | MyoSuite        | RL policy evaluation; depends on OpenSim patterns            |

---

## 11. File Organization Convention

Each engine's perturbation module should follow this structure:

```
src/engines/physics_engines/<engine>/python/
    perturbation/
        __init__.py
        analyzer.py              # PerturbationAnalyzer implementation
        noise.py                 # Noise generation (can import shared)
        metrics.py               # Engine-specific metric extraction
        comparison.py            # Profile comparison logic
        export.py                # JSON/HDF5 export
        tests/
            test_analyzer.py
            test_noise.py
            test_metrics.py
            test_comparison.py
            conftest.py          # Shared fixtures
```

Shared utilities should live in:

```
src/shared/python/perturbation/
    __init__.py
    config.py                    # PerturbationConfig, PerturbationResult, etc.
    noise.py                     # Shared noise generation
    statistics.py                # MetricStatistics, variability_summary
    comparison.py                # Cross-engine comparison utilities
    schema.py                    # JSON schema validation
    robustness_score.py          # RS computation
```

---

## 12. Glossary

| Term                              | Definition                                                  |
| --------------------------------- | ----------------------------------------------------------- | ---- | -------------------------------------- |
| **CV (Coefficient of Variation)** | std /                                                       | mean | — dimensionless measure of variability |
| **Robustness Score (RS)**         | 1/(1+CV_weighted) — unified robustness metric ∈ (0,1]       |
| **Perturbation amplitude**        | Scale factor for noise magnitude relative to nominal input  |
| **White noise**                   | Independent, identically distributed Gaussian samples       |
| **Pink noise (1/f)**              | Power spectral density ∝ 1/f — correlated temporal noise    |
| **Brown noise**                   | Integrated white noise — random walk / drift                |
| **Monte Carlo**                   | Repeated random sampling to estimate statistical properties |
| **ZTCF**                          | Zero-Torque Counterfactual — passive dynamics prediction    |
| **ZVCF**                          | Zero-Velocity Counterfactual — with control but no momentum |

---

## 13. References

- Existing perturbation analysis: `src/shared/python/pendulum_simulator/perturbation_analysis.py`
- Stability metrics: `src/shared/python/analysis/stability_metrics.py`
- Engine protocol: `src/shared/python/engine_core/interfaces.py`
- Base engine: `src/shared/python/engine_core/base_physics_engine.py`
- Pendulum simulation: `src/shared/python/pendulum_simulator/simulation.py`
- Torque utilities: `src/shared/python/pendulum_simulator/torque_utils.py`
