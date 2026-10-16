"""Settings and result types of the noise-aware allocation (NSX-92, #12721).

Value objects only; the solves live in :mod:`.noise_aware_allocation`.  The
result carries the status vocabulary that keeps the claim honest:

* ``OPTIMAL``: convex problem (M1 only, frozen kernels), global optimum
  certified by the solver's KKT residuals.
* ``LOCAL_STATIONARY``: M2 makes the variance non-convex; the point is a
  first-order (KKT) stationary point certified by a vanishing Gauss-Newton step
  and the solver's residuals.  It is NOT claimed to be globally optimal.
* ``OBJECTIVE_FLAT_INVARIANT``: no declared mechanism makes the variance depend
  on the allocation (rule V1); there is nothing to minimise and the variance is
  an ``INVARIANT`` quantity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.shared.python.loop_nullspace.evidence import Array, QuantityRecord
from src.shared.python.loop_nullspace.solvers import KKTResiduals

__all__ = [
    "NON_CLAIM",
    "NoiseAwareResult",
    "NoiseAwareSettings",
    "NoiseAwareStatus",
]

NON_CLAIM = (
    "Conditional on the declared mechanisms and noise amplitudes (hypotheses, "
    "not measurements) and on the linear tangent model at the nominal motion. "
    "The optimum is a hypothesis-conditioned representative, not an estimate of "
    "what a golfer does or a physiological claim; a ranking is not a causal "
    "statement (V8)."
)


class NoiseAwareStatus(str, Enum):
    """Termination class of a noise-aware solve (see the module docstring)."""

    OPTIMAL = "optimal"
    LOCAL_STATIONARY = "local_stationary"
    OBJECTIVE_FLAT_INVARIANT = "objective_flat_invariant"
    INFEASIBLE = "infeasible"
    NOT_CONVERGED = "not_converged"
    SOLVER_FAILURE = "solver_failure"


def _positive(value: float, name: str) -> float:
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class NoiseAwareSettings:
    """Solve options; every tolerance is derived, not tuned.

    ``tie_break_effort_weight`` (>= 0): added capacity effort, in units of the
    normalised variance, used only when the variance has no curvature on some
    retained torque direction (NSX-23 needs strict convexity); ``0`` refuses
    instead.  ``effort_cap``: bound ``J_effort <= cap`` for matched-effort
    comparisons (V4).  ``step_tol``: relative Gauss-Newton step below which the
    iterate counts as stationary; the conic solver is run at ``1e-11`` scaled
    residual so primal points are good to about ``1e-9`` and a step below
    ``1e-8`` is indistinguishable from solver noise.  ``kkt_tol``: largest
    scaled KKT residual accepted for a certificate (``1e-6``, five orders above
    the solver tolerance to absorb the ``mu`` damping bias).  ``flat_tol``:
    ``||J||_F <= flat_tol * max(1, sd)`` marks an allocation-independent
    variance.  ``starts``: extra stacked-``z`` starting points (non-convex case
    only) used to expose several local optima.
    """

    tie_break_effort_weight: float = 0.0
    effort_cap: float | None = None
    max_iterations: int = 80
    step_tol: float = 1e-8
    kkt_tol: float = 1e-6
    flat_tol: float = 1e-12
    outcome_units: str = "rad"
    starts: tuple[Array, ...] = ()

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.tie_break_effort_weight)
            and self.tie_break_effort_weight >= 0.0
        ):
            raise ValueError("tie_break_effort_weight must be finite and >= 0")
        if self.effort_cap is not None:
            _positive(self.effort_cap, "effort_cap")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        for name in ("step_tol", "kkt_tol", "flat_tol"):
            _positive(getattr(self, name), name)
        if not self.outcome_units:
            raise ValueError("outcome_units must be non-empty")
        object.__setattr__(
            self, "starts", tuple(np.asarray(s, dtype=float) for s in self.starts)
        )


@dataclass(frozen=True)
class NoiseAwareResult:
    """Minimum-variance allocation with its certificate and typed variance.

    On ``INFEASIBLE``/``SOLVER_FAILURE``/``NOT_CONVERGED`` only ``status`` and
    ``reason`` are reliable.  ``variance`` is the *linear* outcome variance;
    ``variance_parts`` splits it (initial, additive, signal dependent);
    ``reference_variance`` is the variance of the minimum-effort allocation, so
    the benefit is ``reference_variance - variance`` and is never negative for
    a minimiser.  ``gradient_norm`` is the infinity norm of the exact gradient
    at the reported point projected on nothing (it vanishes only at an interior
    optimum or a flat objective).  ``convex`` is a property of the problem;
    ``global_optimum_certified`` is True only for ``OPTIMAL``.
    """

    status: NoiseAwareStatus
    reason: str = ""
    z: tuple[Array, ...] = ()
    x: tuple[Array, ...] = ()
    torque: Array | None = None
    variance: float | None = None
    variance_parts: dict[str, float] = field(default_factory=dict)
    reference_variance: float | None = None
    reference_torque: Array | None = None
    reference_z: tuple[Array, ...] = ()
    effort: float | None = None
    reference_effort: float | None = None
    gradient_norm: float | None = None
    kkt_residuals: KKTResiduals | None = None
    step_norm: float | None = None
    iterations: int = 0
    convex: bool = True
    global_optimum_certified: bool = False
    local_optima: tuple[float, ...] = ()
    unique_torque: bool = True
    tie_break_applied: bool = False
    mechanisms: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    unverified_hypothesis_ids: tuple[str, ...] = ()
    wrap_probability_bound: float | None = None
    record: QuantityRecord | None = None
    non_claim: str = NON_CLAIM
