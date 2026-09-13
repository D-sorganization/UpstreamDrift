"""Measured resolution floors for central-difference derivative audits.

A central difference of two replays cannot resolve derivative components below
the replay error divided by the step. These helpers turn a *measured* per-replay
error into an explicit floor and classify each audited block without widening
the existing relative gate. A block that agrees inside the floor is reported as
unresolved at that step, never as a pass; a declared structural zero is accepted
only when both analytic and central magnitudes sit inside the floor.
"""

from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Verdict = Literal["passed", "structural_zero", "unresolved_at_step", "failed"]


class DerivativeBlockVerdict(NamedTuple):
    """One audited block at one step, with its measured resolution floor."""

    analytic_l2: float
    central_l2: float
    absolute_l2_error: float
    relative_l2_error: float
    resolution_floor: float
    gate: float
    expected_zero: bool
    verdict: Verdict


def central_difference_floor(replay_error: float, step: float) -> float:
    """Return the L2 error a central difference can carry from replay error.

    Each of the two replays may differ from the exact response by replay_error,
    so the estimate (f(+h) - f(-h)) / (2h) is uncertain by replay_error / h.
    """
    if (
        not np.isfinite(replay_error)
        or replay_error < 0
        or not np.isfinite(step)
        or step <= 0
    ):
        raise ValueError("Replay error must be nonnegative and step positive")
    return float(replay_error / step)


def classify_derivative_block(
    analytic: Array,
    central: Array,
    *,
    step: float,
    replay_error: float,
    gate: float = 1e-3,
    expected_zero: bool = False,
    floor_safety_factor: float = 1.0,
) -> DerivativeBlockVerdict:
    """Compare one analytic block against its central estimate at one step.

    The relative gate is unchanged. A pass additionally requires that the floor
    itself resolves the gate, so agreement inside numerical noise is reported as
    unresolved. Failures are disagreements the floor cannot explain. The
    optional safety factor (at least one) widens only the floor, never the gate:
    a replay error measured from one pair of replays bounds the per-replay
    non-reproducible part only up to the unknown split between the pair.
    """
    a, c = np.asarray(analytic, dtype=float), np.asarray(central, dtype=float)
    if (
        a.shape != c.shape
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(c).all()
        or not np.isfinite(gate)
        or gate <= 0
        or not np.isfinite(floor_safety_factor)
        or floor_safety_factor < 1
    ):
        raise ValueError(
            "Blocks must be finite and equal-shaped, gate positive, safety factor >= 1"
        )
    floor = floor_safety_factor * central_difference_floor(replay_error, step)
    analytic_l2, central_l2 = float(np.linalg.norm(a)), float(np.linalg.norm(c))
    error = float(np.linalg.norm(c - a))
    relative = error / analytic_l2 if analytic_l2 > 0 else float("inf")
    verdict: Verdict
    if expected_zero:
        verdict = (
            "structural_zero"
            if analytic_l2 <= floor and central_l2 <= floor
            else "failed"
        )
    elif relative <= gate and floor <= gate * analytic_l2:
        verdict = "passed"
    elif error <= floor:
        verdict = "unresolved_at_step"
    else:
        verdict = "failed"
    return DerivativeBlockVerdict(
        analytic_l2, central_l2, error, relative, floor, gate, expected_zero, verdict
    )


def qualify_direction(verdicts: Sequence[DerivativeBlockVerdict]) -> bool:
    """One block across steps qualifies with a resolved pass and no failure."""
    outcomes = [v.verdict for v in verdicts]
    return (
        bool(outcomes)
        and "failed" not in outcomes
        and any(outcome in ("passed", "structural_zero") for outcome in outcomes)
    )


def cross_block_norm_bound(
    state_jacobian: Array,
    state_scales: Array,
    direction: Array,
    *,
    block: slice,
    other: slice,
) -> float:
    """Bound the other-block response implied by scaled chart orthonormality.

    If J = diag(scales) N with N orthonormal up to defect delta, then for any
    direction d the scaled response norms of the two blocks share the budget
    (1 + delta) * |d|^2. A direction that spends that budget in one block leaves
    the other structurally zero. The bound is independent of the finite
    difference and is measured from the supplied Jacobian itself.
    """
    j = np.asarray(state_jacobian, dtype=float)
    s = np.asarray(state_scales, dtype=float)
    d = np.asarray(direction, dtype=float)
    if (
        j.ndim != 2
        or s.shape != (j.shape[0],)
        or d.shape != (j.shape[1],)
        or not np.isfinite(j).all()
        or not np.isfinite(s).all()
        or not np.isfinite(d).all()
        or np.any(s <= 0)
    ):
        raise ValueError("Invalid chart Jacobian, positive scales or direction")
    n = j / s[:, None]
    defect = float(np.linalg.norm(n.T @ n - np.eye(j.shape[1]), ord=2))
    scaled = n @ d
    budget = (1 + defect) * float(d @ d) - float(scaled[block] @ scaled[block])
    return float(np.max(s[other]) * np.sqrt(max(budget, 0.0)))
