# Issue: Implement invert_via_surrogate.py — Adam-on-Coefficients Inversion (Option 2)

## Summary

Implement the differentiable inversion that minimises the Python cost function
(#016) over coefficient vectors using gradient descent through the trained
surrogate from #028. Adam optimizer, projected onto the coefficient bounds at
each step.

## Motivation

See `motion_matching/README.md` "Why four options in parallel" — Option 2's payoff
is sub-second-per-fit inversion once the surrogate is trained. This is the
inversion call.

## Dependencies

- #016 (`cost.py`) — the J(θ) being minimised.
- #017 (`ClubTarget`) — input target type.
- #028 (`SwingSurrogate`) — provides differentiable forward.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option2_nn_surrogate\python\invert_via_surrogate.py`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option2_nn_surrogate\python\projected_adam.py` (custom Adam variant that projects onto coefficient bounds)
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option2\test_invert_via_surrogate.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option2\test_projected_adam.py`

## Public API

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

@dataclass(frozen=True)
class InversionConfig:
    surrogate_checkpoint: Path
    n_iterations: int = 500
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "auto"
    cost_options: "CostOptions" = field(default_factory=CostOptions)
    starting_strategy: Literal["nominal", "random", "warm_from_dataset"] = "nominal"
    log_every: int = 10


@dataclass(frozen=True)
class InversionResult:
    coefficients: np.ndarray   # (n_joints * 7,)
    final_J: float
    history: list[dict]        # per-iteration {iter, J, terms, grad_norm}
    surrogate_checkpoint_sha: str
    duration_s: float


@precondition(lambda target, config: config.surrogate_checkpoint.exists(),
              "surrogate checkpoint must exist")
@postcondition(lambda result: result.final_J >= 0 and np.isfinite(result.final_J),
               "final J must be finite and non-negative")
def invert_via_surrogate(target: "ClubTarget", config: InversionConfig) -> InversionResult:
    """Adam-on-coefficients inversion using a trained SwingSurrogate."""
```

## Required tests (TDD)

- `test_inversion_returns_coefficients_within_bounds_from_generateRandomCoefficients`
- `test_inversion_decreases_J_monotonically_for_synthetic_target`
- `test_inversion_recovers_theta_truth_within_5_percent_for_synthetic_target_at_10000_iters`
- `test_projected_adam_clamps_updates_to_coefficient_bounds`
- `test_projected_adam_matches_torch_adam_when_no_projection_active`
- `test_inversion_seed_reproducibility`
- `test_inversion_history_has_one_entry_per_log_every_step`
- `test_inversion_starting_strategy_warm_from_dataset_uses_nearest_neighbour_in_kinematic_space`
- `test_inversion_records_surrogate_checkpoint_sha_for_provenance`
- `test_inversion_rejects_missing_surrogate_checkpoint_with_clear_error`
- `test_inversion_runs_in_under_5_seconds_for_500_iterations_on_cpu`

## DbC contract

Preconditions:

- `target` is a `ClubTarget`.
- `config.surrogate_checkpoint` exists.
- `config.n_iterations >= 1`.
- `config.learning_rate > 0`.

Postconditions:

- `result.coefficients` is a finite vector within coefficient bounds.
- `result.final_J >= 0` and finite.
- `len(result.history) == config.n_iterations // config.log_every + 1`.

## Acceptance Criteria

- [ ] `invert_via_surrogate.py` minimises `cost.py`'s J via differentiable forward.
- [ ] All listed tests pass.
- [ ] DbC decorators applied; postconditions enforced.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] Coefficient bound projection verified on synthetic out-of-bounds initial guess.
- [ ] No file exceeds 1200 lines.
- [ ] No `print()`; use `get_logger`.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option2`, `python`, `tdd`, `dbc`

## Effort estimate

M (1-3 days).
