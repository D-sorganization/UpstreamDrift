# Issue: Implement Hybrid Surrogate-Then-fmincon Polish (Option 2 → Option 1 Handoff)

## Summary

Implement the hybrid pipeline that uses Option 2's surrogate inversion for a
fast initial estimate, then hands off to Option 1's `fit_swing_fmincon` for
final polish. This combines Option 2's speed with Option 1's exactness.

## Motivation

See `motion_matching/README.md`. The surrogate's inversion may settle near the
correct basin but with the surrogate's bias baked in; an fmincon polish using
the **true** Simscape forward removes that bias for the final fit. This pattern
is the most likely production path: train once, fit many.

## Dependencies

- #024 (`fit_swing_fmincon.m`) — polish step.
- #029 (`invert_via_surrogate.py`) — initial estimate.
- #030 (`matlab_bridge.py`) — Python ↔ MATLAB callout, reused.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option2_nn_surrogate\python\hybrid_surrogate_polish.py`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option2_nn_surrogate\python\result_io.py` (writes the canonical result struct in a format `leaderboard.m` can read)
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option2\test_hybrid_surrogate_polish.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option2\test_result_io.py`

## Public API

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class HybridConfig:
    surrogate_inversion: InversionConfig
    fmincon_options_overrides: dict
    output_dir: Path
    bridge: Literal["matlab_engine", "subprocess"] = "matlab_engine"


@dataclass(frozen=True)
class HybridResult:
    initial_estimate: InversionResult
    polished: dict           # mirrors MATLAB fit_swing_fmincon result struct
    final_rmse_m: float      # from polished step
    surrogate_rmse_m: float  # before polish, for comparison
    improvement_mm: float    # surrogate_rmse_m - final_rmse_m, in mm
    duration_s: float
    output_mat_path: Path    # written to be readable by leaderboard.m


def hybrid_surrogate_polish(target: "ClubTarget", config: HybridConfig) -> HybridResult:
    """1) Run invert_via_surrogate to get a warm start.
       2) Pass warm start as x0 to fit_swing_fmincon via matlab_bridge.
       3) Write a result .mat to output_dir compatible with leaderboard.m."""
```

## Required tests (TDD)

- `test_hybrid_runs_inversion_then_fmincon_polish_in_order`
- `test_hybrid_polished_rmse_less_than_or_equal_to_surrogate_rmse`
- `test_hybrid_writes_result_mat_compatible_with_leaderboard_m`
- `test_hybrid_result_solver_field_equals_surrogate_plus_fmincon`
- `test_hybrid_records_both_initial_estimate_and_polished_result`
- `test_hybrid_handles_inversion_failure_with_clear_error_no_polish_attempted`
- `test_hybrid_seed_reproducibility_full_pipeline`
- `test_result_io_writes_provenance_fields_per_coding_standards`
- `test_result_io_round_trips_through_loadmat_with_no_data_loss`
- `test_hybrid_outperforms_surrogate_only_by_at_least_2mm_on_validation_set`

## DbC contract

Preconditions:

- `target` is a `ClubTarget`.
- `config.surrogate_inversion.surrogate_checkpoint` exists.
- `config.output_dir` is creatable.

Postconditions:

- `result.final_rmse_m <= result.surrogate_rmse_m`.
- `result.output_mat_path` exists on disk and contains all provenance fields per
  `CODING_STANDARDS.md`.
- `result.polished["solver"] == "surrogate+fmincon"`.

## Acceptance Criteria

- [ ] `hybrid_surrogate_polish` runs end-to-end on the synthetic fixture.
- [ ] All listed tests pass.
- [ ] Result `.mat` loads in MATLAB and renders correctly via `leaderboard.m`.
- [ ] DbC decorators applied; postconditions enforced.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines.
- [ ] No `print()`; use `get_logger`.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option2`, `python`, `matlab`, `tdd`, `dbc`

## Effort estimate

M (1-3 days). Mostly glue, but the result-struct schema sync between Python
and MATLAB needs care.
