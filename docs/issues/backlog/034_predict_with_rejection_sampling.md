# Issue: Implement predict_with_rejection_sampling.py — Sample-and-Validate Inference (Option 3)

## Summary

Implement the inference pipeline for the trained CVAE: sample `K` candidate
coefficient vectors per target, simulate each through Simscape (or via the
surrogate from #028 for speed), and return the candidate that minimises the
Python cost function from #016.

## Motivation

See `motion_matching/README.md`. A CVAE produces a distribution over coefficients
for each input — multiple may be plausible. The honest inference is "sample K,
score K, return best", which gives O(K) cost evaluations instead of an
optimization loop. With Simscape forward in the loop, this is the only
ms-per-fit option that gracefully handles multi-modality.

## Dependencies

- #016 (`cost.py`) — scoring function.
- #018 (`simulate_with_coefficients.m`) — ground-truth forward (via #030's bridge).
- #028 (`SwingSurrogate`) — fast forward, optional.
- #032 (`SwingInverseCVAE`) — the trained CVAE.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option3_inverse_nn\python\predict_with_rejection_sampling.py`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option3_inverse_nn\python\sample_scorer.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option3\test_predict_with_rejection_sampling.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option3\test_sample_scorer.py`

## Public API

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np

@dataclass(frozen=True)
class RejectionSamplingConfig:
    cvae_checkpoint: Path
    n_samples: int = 32
    scorer_forward: Literal["simscape", "surrogate", "both"] = "surrogate"
    surrogate_checkpoint: Path | None = None  # required when scorer_forward != "simscape"
    cost_options: "CostOptions" = field(default_factory=CostOptions)
    seed: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class RejectionSamplingResult:
    best_coefficients: np.ndarray   # (n_joints * 7,)
    all_candidates: np.ndarray      # (n_samples, n_joints * 7)
    all_J: np.ndarray               # (n_samples,)
    best_J: float
    scorer_forward: str
    duration_s: float


def predict_with_rejection_sampling(
    target: "ClubTarget",
    config: RejectionSamplingConfig,
) -> RejectionSamplingResult:
    """Sample n_samples coefficient candidates from the CVAE, score each, return best."""
```

## Required tests (TDD)

- `test_returns_result_with_n_samples_candidates_and_J_array`
- `test_best_coefficients_minimize_J_among_all_candidates`
- `test_best_J_equals_min_of_all_J`
- `test_scorer_forward_simscape_calls_matlab_bridge`
- `test_scorer_forward_surrogate_uses_swing_surrogate_no_matlab`
- `test_scorer_forward_both_uses_surrogate_for_initial_filter_then_simscape_for_top_k`
- `test_seed_reproducibility_for_fixed_n_samples`
- `test_n_samples_diversity_pairwise_l2_distance_above_threshold`
- `test_runs_in_under_1_second_for_n_samples_32_with_surrogate_scorer_on_cpu`
- `test_rejects_missing_cvae_checkpoint_with_clear_error`
- `test_rejects_simscape_scorer_when_surrogate_checkpoint_required_for_both_mode`
- `test_records_cvae_checkpoint_sha_and_surrogate_checkpoint_sha_for_provenance`

## DbC contract

Preconditions:

- `target` is a `ClubTarget`.
- `config.cvae_checkpoint` exists.
- When `scorer_forward in {"surrogate","both"}`, `config.surrogate_checkpoint` exists.
- `config.n_samples >= 1`.

Postconditions:

- `result.best_J == result.all_J.min()`.
- `result.best_coefficients` is within coefficient bounds.
- `result.all_J.shape == (n_samples,)`.

## Acceptance Criteria

- [ ] `predict_with_rejection_sampling` works end-to-end on the synthetic fixture.
- [ ] All listed tests pass.
- [ ] All three scorer modes implemented (`simscape`, `surrogate`, `both`).
- [ ] DbC decorators applied; postconditions enforced.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines.
- [ ] No `print()`; use `get_logger`.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option3`, `python`, `tdd`, `dbc`

## Effort estimate

M (1-3 days).
