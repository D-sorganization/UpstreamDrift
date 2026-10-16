"""Objective specification and result types for trajectory allocation (NSX-23).

Value objects only; the solve lives in :mod:`.allocation`.  The objective is

``J = sum_k w_k [ a_e |S^-1 tau_k|^2 + a_g sum_i (c+_i [p_ki]_+ + c-_i [-p_ki]_+) ]
      + a_r sum_k |S^-1 (tau_{k+1} - tau_k)|^2 / dt_k
      + a_ref sum_k w_k |S^-1 (tau_k - tau_ref,k)|^2``

with ``S = diag(torque_scale)`` (convention ``C-METRIC-CAPACITY``), trapezoid
weights ``w_k`` over ``times_s`` and joint power ``p_ki = v_ki tau_ki``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.evidence import (
    Array,
    QuantityRecord,
    readonly_float,
)
from src.shared.python.loop_nullspace.solvers import KKTResiduals, SolveStatus

__all__ = [
    "AllocationResult",
    "ConvexityReport",
    "GrossPowerSpec",
    "NotStrictlyConvexError",
    "ObjectiveSpec",
    "PowerReduction",
    "VarianceTerm",
]


class NotStrictlyConvexError(ValueError):
    """The objective has no curvature on the retained torque space."""


def _vector(value: Any, name: str) -> Array:
    arr = readonly_float(value, name)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _weight(value: float, name: str) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class GrossPowerSpec:
    """Gross joint power with separate positive/negative costs (R6).

    ``H-GROSS-POWER-COST``: ``c_plus`` weights positive power and ``c_minus``
    negative power, per actuated coordinate.  ``joint_velocity`` is
    ``(frames, n_a)``.  ``allow_reduction=False`` forces the unreduced epigraph
    (used to check the reduction numerically).
    """

    weight: float
    c_plus: Array
    c_minus: Array
    joint_velocity: Array
    allow_reduction: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError(f"gross-power weight must be positive, got {self.weight}")
        cp = _vector(self.c_plus, "c_plus")
        cm = _vector(self.c_minus, "c_minus")
        if cp.shape != cm.shape or cp.size == 0:
            raise ValueError("c_plus and c_minus must be non-empty and equal length")
        if np.any(cp < 0.0):
            raise ValueError("c_plus must be non-negative")
        if np.any(cm < 0.0):
            raise ValueError("c_minus must be non-negative")
        vel = readonly_float(self.joint_velocity, "joint_velocity")
        if vel.ndim != 2 or vel.shape[1] != cp.size:
            raise ValueError(
                f"joint_velocity must have shape (frames, {cp.size}), got {vel.shape}"
            )
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "c_plus", cp)
        object.__setattr__(self, "c_minus", cm)
        object.__setattr__(self, "joint_velocity", vel)


@dataclass(frozen=True)
class VarianceTerm:
    """Signal-dependent-noise variance as a torque quadratic (NSX-92, M1).

    With the outcome kernels frozen, the M1 contribution to the outcome
    variance is ``sum_tj coeff[t, j] tau[t, j]^2`` with
    ``coeff = k_j^2 S_in[t, j]^2 / dt`` (``loop_nullspace.tex``
    ``sec:var-mechanisms``): a *diagonal* convex quadratic in the torque, so it
    enters the allocation exactly like the effort term.  ``weight`` scales it;
    allocation-invariant variance parts are constants and live outside the
    objective.  ``hypothesis_ids`` name the mechanism it rests on.
    """

    weight: float
    coeff: Array
    hypothesis_ids: tuple[str, ...] = ("H-NOISE-SIGNAL-DEPENDENT",)

    def __post_init__(self) -> None:
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError(f"variance weight must be positive, got {self.weight!r}")
        c = readonly_float(self.coeff, "coeff")
        if c.ndim != 2 or c.size == 0:
            raise ValueError(f"coeff must be a non-empty (frames, n_a), got {c.shape}")
        if np.any(c < 0.0):
            raise ValueError("coeff must be non-negative (a variance weight)")
        if not self.hypothesis_ids:
            raise ValueError("a variance term must name its hypothesis ids")
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "coeff", c)
        object.__setattr__(self, "hypothesis_ids", tuple(self.hypothesis_ids))


@dataclass(frozen=True)
class ObjectiveSpec:
    """Weights of the allocation objective (see the module docstring).

    Raises:
        ValueError: If the times are not strictly increasing, a scale is not
            positive, or a weight is negative or non-finite.
    """

    times_s: Array
    torque_scale: Array
    effort_weight: float = 1.0
    rate_weight: float = 0.0
    reference_weight: float = 0.0
    gross_power: GrossPowerSpec | None = None
    variance: VarianceTerm | None = None

    def __post_init__(self) -> None:
        t = _vector(self.times_s, "times_s")
        if t.size == 0 or (t.size > 1 and np.any(np.diff(t) <= 0.0)):
            raise ValueError("times_s must be non-empty and strictly increasing")
        s = _vector(self.torque_scale, "torque_scale")
        if s.size == 0 or np.any(s <= 0.0):
            raise ValueError("torque_scale must be non-empty with positive entries")
        object.__setattr__(self, "times_s", t)
        object.__setattr__(self, "torque_scale", s)
        for name in ("effort_weight", "rate_weight", "reference_weight"):
            object.__setattr__(self, name, _weight(getattr(self, name), name))
        if self.gross_power is not None and not isinstance(
            self.gross_power, GrossPowerSpec
        ):
            raise TypeError("gross_power must be a GrossPowerSpec or None")
        if self.variance is not None:
            if not isinstance(self.variance, VarianceTerm):
                raise TypeError("variance must be a VarianceTerm or None")
            coeff = self.variance.coeff
            if coeff.shape != (t.size, s.size):
                raise ValueError(
                    f"variance coeff must have shape {(t.size, s.size)}, "
                    f"got {coeff.shape}"
                )


@dataclass(frozen=True)
class PowerReduction:
    """Detected reduction of the c-plus/c-minus gross-power cost (NSX-00 identity).

    ``sum(c+[p]+ + c-[-p]+) = 1/2 (c+ + c-) sum|p| + 1/2 (c+ - c-) P_net``.
    ``ABS_POWER_ONLY``: ``c+ = c-`` for every joint, so the net-work term
    vanishes.  ``UNIFORM_NET_PLUS_ABS``: uniform weights and ``P_net`` constant
    over the feasible family (``v^T d tau/dz = 0``), so the ratio ``c-/c+`` is a
    dead parameter.  Otherwise ``NONE`` (per-joint heterogeneous weights).
    """

    kind: str
    uniform: bool
    net_power_invariant: bool
    ratio_is_dead_parameter: bool
    reduced_form_used: bool
    net_work_offset: float = 0.0


@dataclass(frozen=True)
class ConvexityReport:
    """Curvature of the quadratic part on the retained (torque-visible) space.

    ``required_rank`` is the number of z-directions that change a torque;
    ``gauge_dimension`` counts the others (multiplier/f_g gauge).
    """

    rank: int
    required_rank: int
    gauge_dimension: int
    min_curvature_ratio: float
    enabled_terms: tuple[str, ...]


@dataclass(frozen=True)
class AllocationResult:
    """Hypothesis-conditioned representative allocation; never a load estimate.

    On a non-OPTIMAL ``status`` only ``status``, ``reason``, ``solve_time_s``
    and ``iterations`` carry data.  ``row_duals`` maps ``frame<k>:<label>`` to
    the (non-negative) multiplier of a polyhedral row, ``cone_duals`` the
    second-order-cone multiplier vector of an exact cone.
    """

    status: SolveStatus
    reason: str = ""
    times_s: Array | None = None
    z: tuple[Array, ...] = ()
    x: tuple[Array, ...] = ()
    torque: Array | None = None
    objective_terms: dict[str, float] = field(default_factory=dict)
    objective_total: float | None = None
    kkt_residuals: KKTResiduals | None = None
    row_duals: dict[str, float] = field(default_factory=dict)
    cone_duals: dict[str, Array] = field(default_factory=dict)
    solve_time_s: float = 0.0
    iterations: int = 0
    power_reduction: PowerReduction | None = None
    convexity: ConvexityReport | None = None
    record: QuantityRecord | None = None
    reference_used: bool = False
    unverified_hypothesis_ids: tuple[str, ...] = ()
    non_claim: str = (
        "The optimum is a hypothesis-conditioned representative of the feasible "
        "family, not an estimate of the true hand loads or joint torques."
    )
