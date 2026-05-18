# Issue: Implement Python cost.py Mirror with Cross-Check Against MATLAB

## Summary

Implement the Python mirror of the cost function at
`src/shared/python/motion_matching/cost.py`. It must produce **numerically
identical** results to the MATLAB `compute_cost` from #015 on the same inputs,
verified by a cross-check test that loads MATLAB outputs and compares to the
Python implementation.

## Motivation

See `motion_matching/shared/COST_FUNCTION_SPEC.md` §"What the function signature
looks like". Options 2, 3, and 4 all need a Python cost function to drive
gradient-based inversion or to share a common metric with Option 1's MATLAB code.
If the two implementations diverge, cross-option leaderboard comparisons (#023)
become meaningless.

## Dependencies

- #015 (MATLAB reference implementation; cross-check oracle).

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\shared\python\motion_matching\__init__.py`
- `C:\Users\diete\Repositories\UpstreamDrift\src\shared\python\motion_matching\cost.py`
- `C:\Users\diete\Repositories\UpstreamDrift\src\shared\python\motion_matching\options.py` (`CostOptions` dataclass)
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\test_cost.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\test_cost_cross_check_matlab.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\fixtures\matlab_cost_reference.json` (precomputed MATLAB outputs for fixed `(theta, target, opts)` triples)

## Public API

Mirror of MATLAB signature, using numpy and a frozen dataclass for `opts`:

```python
from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np

@dataclass(frozen=True)
class CostOptions:
    w_position: float = 1.0
    w_orientation: float = 0.1
    w_anchor_impact: float = 10.0
    regularizer: Literal["total_work", "peak_power", "torque_l2", "coeff_l2"] = "total_work"
    lambda_: float = 1e-4
    q_orientation_repr: Literal["quaternion", "rotmat"] = "quaternion"
    time_alignment: Literal["impact", "address", "none"] = "impact"
    resample_to_hz: float = 1000.0


@dataclass(frozen=True)
class CostTerms:
    position: float
    orientation: float
    impact_anchor: float
    regularizer: float
    total: float


@precondition(lambda theta, target, sim_fn, opts: theta.ndim == 1 and np.all(np.isfinite(theta)),
              "theta must be a finite 1-D vector")
@postcondition(lambda result: result[0] >= 0 and np.isfinite(result[0]),
               "J must be finite and non-negative")
def compute_cost(
    theta: np.ndarray,
    target: "ClubTarget",
    sim_fn: Callable[[np.ndarray], "SimOutput"],
    opts: CostOptions = CostOptions(),
) -> tuple[float, CostTerms]:
    """Scalar swing-matching cost mirroring compute_cost.m."""


def compute_total_work(sim_out: "SimOutput") -> float:
    """Integrates Σ|τ·ω|·dt across joints. Postcondition: W >= 0."""
```

## Required tests (TDD)

- `test_zero_residual_yields_only_regularizer_term`
- `test_position_term_is_mean_squared_butt_plus_clubhead_distance`
- `test_orientation_term_uses_geodesic_quaternion_distance_with_abs`
- `test_quaternion_sign_flip_does_not_change_orientation_term`
- `test_regularizer_modes_match_spec_formulas`
- `test_terms_total_equals_J_within_eps`
- `test_compute_total_work_zero_for_zero_torque`
- `test_compute_total_work_matches_handcalc_for_constant_torque_constant_omega`
- `test_precondition_rejects_nan_theta`
- `test_postcondition_J_nonnegative_for_random_finite_inputs`

Cross-check tests (use the fixture file):

- `test_cross_check_position_term_matches_matlab_within_1e_minus_10`
- `test_cross_check_orientation_term_matches_matlab_within_1e_minus_10`
- `test_cross_check_total_work_matches_matlab_within_1e_minus_10`
- `test_cross_check_J_total_matches_matlab_within_1e_minus_10`
- `test_cross_check_covers_all_four_regularizer_modes`

## DbC contract

Use existing decorators from `src.shared.python.core.contracts`:

Preconditions:

- `theta` is a 1-D finite numpy array of length `n_joints * 7`.
- `target` is a `ClubTarget` instance (from #017).
- `sim_fn` is callable.
- `opts` is a `CostOptions` instance.

Postconditions:

- Return tuple `(J, terms)` where `J` is finite and non-negative.
- `terms.total == J` to within `1e-12`.
- Every field of `terms` is non-negative.

## Acceptance Criteria

- [ ] `cost.py` produces identical numbers to MATLAB on the fixture inputs.
- [ ] All listed unit and cross-check tests pass.
- [ ] `@precondition` / `@postcondition` decorators applied per spec.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines.
- [ ] No `print()` in `src/`; use `get_logger`.
- [ ] No TODO/FIXME without a tracked issue link.
- [ ] Fixture JSON regeneration script committed under `scripts/regen_cost_fixture.m`.

## Labels

`motion-matching`, `shared`, `python`, `tdd`, `dbc`, `infra`

## Effort estimate

M (1-3 days). Mirroring math is fast; debugging `1e-10`-level cross-check
failures (quaternion sign, integration scheme, dt vs sample_rate) is the time sink.
