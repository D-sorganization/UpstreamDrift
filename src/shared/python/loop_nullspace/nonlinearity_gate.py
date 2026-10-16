"""Two-amplitude nonlinearity ratio and the ``NOT_LINEAR_VALID`` flag (NSX-90).

``loop_nullspace.tex`` (``sec:var-nonclaims``, rule V7) says linear variance
predictions are "conditional on a nonlinearity gate" and gives neither the gate
statistic nor its threshold.  This module therefore fixes **no threshold**:
``gate_threshold`` is a required keyword with no default, and the caller owns
the number (and its provenance).  The statistic below is a declared choice, not
a specification value.

Statistic.  For a probe direction ``d`` in the stacked input space
``(x0, w_0, ..., w_{T-1}, ...)`` and a nonlinear rollout ``y(theta)`` with
nominal ``y(0)``, let ``D(a) = y(a d) - y(0)``.  The response is exactly linear
iff ``D(2 eps) = 2 D(eps)``, so per output component

    r = |D(2 eps) - 2 D(eps)| / max(|2 D(eps)|, response_floor),

which is ``O(eps)`` for a smooth map.  ``response_floor`` is a numerical
resolution only (it keeps a zero response from dividing by zero); it is not a
validity limit.  Probes are the adjoint worst-case direction of each outcome
(unit Mahalanobis length, response equal to the predicted standard deviation)
and, optionally, one standard deviation along each input coordinate.  A record
whose maximum ratio exceeds ``gate_threshold``, or whose rollout fails to give
finite values (retained failure), is ``NOT_LINEAR_VALID``.

Non-claim: a passing gate says the linear model is locally adequate along the
probes; it does not cover saturation, slip or missed impacts that the probes do
not reach (use paired nonlinear Monte Carlo, NSX-91).  Pure numpy.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.variance_propagation import (
    AdjointKernels,
    stacked_covariance,
    stacked_sensitivity,
)

__all__ = [
    "DEFAULT_RESPONSE_FLOOR",
    "LinearValidity",
    "NonlinearityReport",
    "Rollout",
    "assess_nonlinearity",
    "probe_directions",
]

Rollout = Callable[[Array], ArrayLike]
DEFAULT_RESPONSE_FLOOR = 1e-12


class LinearValidity(str, Enum):
    """Whether the linear variance prediction passed the nonlinearity gate."""

    LINEAR_VALID = "LINEAR_VALID"
    NOT_LINEAR_VALID = "NOT_LINEAR_VALID"
    NOT_ASSESSED = "NOT_ASSESSED"


@dataclass(frozen=True)
class NonlinearityReport:
    """Result of the two-amplitude gate.

    ``ratios`` has shape ``(n_probe, ny)`` (``inf`` for a failed probe);
    ``max_ratio`` is its column maximum; ``exceeded[i]`` is true when output
    ``i`` is above ``gate_threshold`` (or non-finite).
    """

    ratios: Array
    max_ratio: Array
    exceeded: tuple[bool, ...]
    gate_threshold: float
    amplitude: float
    n_failed: int

    @property
    def validity(self) -> LinearValidity:
        if any(self.exceeded):
            return LinearValidity.NOT_LINEAR_VALID
        return LinearValidity.LINEAR_VALID


def _positive(value: float, name: str) -> float:
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return float(value)


def probe_directions(
    kernels: AdjointKernels,
    p0: ArrayLike,
    sigma: ArrayLike,
    *,
    include_axes: bool = True,
) -> Array:
    """Probe directions in the stacked input space, shape ``(n_probe, m)``.

    For outcome row ``s`` the worst-case direction is ``d = Sigma s^T /
    sqrt(s Sigma s^T)``: ``d^T Sigma^+ d = 1`` and ``s d`` equals the predicted
    standard deviation.  Outcomes with zero predicted variance contribute no
    such probe.  ``include_axes`` adds ``sqrt(Sigma_jj) e_j`` for every input
    coordinate with positive variance.
    """
    cov = stacked_covariance(p0, sigma)
    sens = stacked_sensitivity(kernels)
    rows = []
    for s_row in sens:
        spread = cov @ s_row
        var = float(s_row @ spread)
        if var > 0.0:
            rows.append(spread / math.sqrt(var))
    if include_axes:
        for j, var_j in enumerate(np.diag(cov)):
            if var_j > 0.0:
                axis = np.zeros(cov.shape[0])
                axis[j] = math.sqrt(float(var_j))
                rows.append(axis)
    if not rows:
        raise ValueError("no input direction has positive variance")
    return np.array(rows)


def _evaluate(rollout: Rollout, theta: Array) -> Array | None:
    try:
        out = np.asarray(rollout(theta), dtype=np.float64).reshape(-1)
    except ArithmeticError:
        return None
    return out if bool(np.all(np.isfinite(out))) else None


def assess_nonlinearity(
    rollout: Rollout,
    directions: ArrayLike,
    *,
    amplitude: float,
    gate_threshold: float,
    response_floor: float = DEFAULT_RESPONSE_FLOOR,
) -> NonlinearityReport:
    """Run the two-amplitude finite-difference gate on a nonlinear rollout.

    ``rollout(theta)`` maps a stacked input vector to the outcome vector (the
    deviation from the nominal outcome is formed here from ``rollout(0)``).

    Preconditions (``ValueError``): ``directions`` finite 2-D ``(n_probe, m)``;
    ``amplitude``, ``gate_threshold`` and ``response_floor`` positive and
    finite; the nominal ``rollout(0)`` finite.  ``gate_threshold`` has no
    default by design.  Postconditions: ``ratios >= 0``; a probe whose rollout
    is non-finite at either amplitude counts in ``n_failed`` and its ratio is
    ``inf`` (flagged, never dropped).
    """
    dirs = np.asarray(directions, dtype=np.float64)
    if dirs.ndim != 2 or dirs.shape[0] == 0 or not np.all(np.isfinite(dirs)):
        raise ValueError("directions must be a finite 2-D array (n_probe, m)")
    eps = _positive(amplitude, "amplitude")
    threshold = _positive(gate_threshold, "gate_threshold")
    floor = _positive(response_floor, "response_floor")
    nominal = _evaluate(rollout, np.zeros(dirs.shape[1]))
    if nominal is None:
        raise ValueError("the nominal rollout (theta = 0) must be finite")
    ratios = np.full((dirs.shape[0], nominal.size), np.inf)
    n_failed = 0
    for i, direction in enumerate(dirs):
        one = _evaluate(rollout, eps * direction)
        two = _evaluate(rollout, 2.0 * eps * direction)
        if one is None or two is None or one.size != nominal.size:
            n_failed += 1
            continue
        d_one, d_two = one - nominal, two - nominal
        ratios[i] = np.abs(d_two - 2.0 * d_one) / np.maximum(np.abs(2.0 * d_one), floor)
    max_ratio = ratios.max(axis=0)
    exceeded = tuple(bool(not r <= threshold) for r in max_ratio)
    return NonlinearityReport(
        ratios=ratios,
        max_ratio=max_ratio,
        exceeded=exceeded,
        gate_threshold=threshold,
        amplitude=eps,
        n_failed=n_failed,
    )
