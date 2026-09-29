# Issue: Round-Trip Validation of Surrogate Inversion Against Simscape (Option 2)

## Summary

Implement a validation harness that compares surrogate-predicted kinematics
against ground-truth Simscape kinematics across a held-out validation set, then
reports per-trial residuals. This is the only honest measure of whether Option 2
is actually solving the problem or just fitting the surrogate.

## Motivation

See `motion_matching/shared/VISUALIZATION_SPEC.md` §"Surrogate-vs-truth residuals"
and "Round-trip residuals". A surrogate can be highly accurate on `f_θ(coeffs)`
yet fail on inverted swings because the inversion drifts into a region where
the surrogate is over-confident. Without this validation, Option 2's reported
RMSEs are unreliable.

## Dependencies

- #018 (`simulate_with_coefficients.m`) — ground-truth Simscape forward.
- #028 (`SwingSurrogate`) — the surrogate under test.
- #029 (`invert_via_surrogate`) — the inversion under test.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option2_nn_surrogate\python\round_trip_validation.py`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option2_nn_surrogate\python\matlab_bridge.py` (thin wrapper around MATLAB Engine for Python that calls `simulate_with_coefficients.m`)
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option2_nn_surrogate\python\plot_surrogate_residuals.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option2\test_round_trip_validation.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option2\test_matlab_bridge.py`

## Public API

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class RoundTripConfig:
    surrogate_checkpoint: Path
    validation_dataset: Path
    n_trials: int = 100
    bridge: Literal["matlab_engine", "subprocess", "stub"] = "matlab_engine"
    seed: int = 42
    output_dir: Path = Path("results/option2_validation")


@dataclass(frozen=True)
class RoundTripReport:
    surrogate_predicted: np.ndarray   # (n_trials, n_timesteps, kinematic_dim)
    simscape_truth: np.ndarray        # (n_trials, n_timesteps, kinematic_dim)
    coefficients: np.ndarray          # (n_trials, n_joints*7)
    rmse_per_trial_m: np.ndarray      # (n_trials,)
    rmse_overall_m: float
    histogram_path: Path


def round_trip_validation(config: RoundTripConfig) -> RoundTripReport:
    """For each validation trial:
       1. Read coefficients from validation set.
       2. Predict kinematics with the surrogate.
       3. Run the same coefficients through Simscape (via matlab_bridge).
       4. Record per-trial RMSE.
       Then write a histogram + summary CSV."""
```

## Required tests (TDD)

- `test_round_trip_returns_report_with_per_trial_rmse_and_overall_rmse`
- `test_round_trip_runs_n_trials_simscape_calls_via_matlab_bridge`
- `test_round_trip_overall_rmse_is_within_5mm_for_well_trained_surrogate`
- `test_round_trip_writes_histogram_png_to_output_dir`
- `test_round_trip_writes_summary_csv_with_per_trial_rmse`
- `test_matlab_bridge_calls_simulate_with_coefficients_not_a_separate_simscape_call`
- `test_matlab_bridge_returns_canonical_sim_out_struct`
- `test_matlab_bridge_handles_matlab_engine_unavailable_with_clear_error`
- `test_matlab_bridge_subprocess_fallback_when_engine_unavailable`
- `test_matlab_bridge_stub_returns_random_kinematics_for_unit_tests_without_matlab`
- `test_round_trip_seed_reproducibility_for_validation_subset_selection`
- `test_round_trip_stub_bridge_runs_without_matlab_for_ci`

## DbC contract

Preconditions:

- `config.surrogate_checkpoint` exists.
- `config.validation_dataset` exists.
- `config.n_trials >= 1`.

Postconditions:

- `report.rmse_per_trial_m.shape == (n_trials,)`.
- `report.rmse_overall_m == sqrt(mean(rmse_per_trial_m**2))`.
- `report.histogram_path` exists on disk.

## Acceptance Criteria

- [ ] `round_trip_validation` works end-to-end for the synthetic dataset fixture.
- [ ] `matlab_bridge.py` supports three modes: `matlab_engine`, `subprocess`, `stub`.
- [ ] All listed tests pass; `stub` mode used in CI to avoid MATLAB dependency.
- [ ] Histogram and summary CSV produced as artefacts.
- [ ] DbC decorators applied; postconditions enforced.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines.
- [ ] No `print()`; use `get_logger`.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option2`, `python`, `tdd`, `dbc`

## Effort estimate

M (1-3 days). The MATLAB Engine plumbing is the time sink (and is largely
reusable for #037).
