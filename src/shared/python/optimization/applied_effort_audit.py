"""Sampled total actuator torque and slew feasibility for swing optimization.

Inputs are actual applied torques after feedforward, feedback and saturation.
This audit does not establish inter-sample, work, ROM or tracking feasibility.
See docs/research/applied_effort_audit/README.md for the numerical contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.optimization._finite_array import finite_real_array


@dataclass(frozen=True)
class AppliedEffortAudit:
    """Per-joint minimum signed margins; negative means a limit was exceeded."""

    torque_margin_nm: tuple[float, ...]
    slew_margin_nm_s: tuple[float, ...]

    @property
    def feasible(self) -> bool:
        """Whether both sampled constraints hold, including their boundaries."""
        return all(m >= 0.0 for m in self.torque_margin_nm + self.slew_margin_nm_s)


def _real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return a finite real array, or raise ValueError naming ``name``."""
    return finite_real_array(value, name)


def audit_applied_effort(
    *,
    times_s: ArrayLike,
    applied_torque_nm: ArrayLike,
    torque_lower_nm: ArrayLike,
    torque_upper_nm: ArrayLike,
    slew_limit_nm_s: ArrayLike,
) -> AppliedEffortAudit:
    """Return finite minimum torque and secant-slew margins for each joint.

    Preconditions: finite real inputs; strictly increasing 1D times (N >= 2);
    total applied torque shape (N, J), J >= 1; bounds and slew shape (J,);
    lower <= upper and slew >= 0. Units are seconds, Nm and Nm/s.

    Postconditions: immutable J-element margins, nonnegative exactly when all
    samples and adjacent-sample secant rates meet their respective limits.
    Overflow in derived quantities raises ValueError rather than passing.
    The caller supplies torques in a consistent actuator order and sign frame.
    ZOH jumps and continuous-time derivatives require separate verification.
    """
    times = _real_array(times_s, "times_s")
    torque = _real_array(applied_torque_nm, "applied_torque_nm")
    if times.ndim != 1:
        raise ValueError("times_s must be one-dimensional")
    if times.size < 2:
        raise ValueError("times_s must have at least two samples")
    if torque.ndim != 2 or torque.shape[0] != times.size or torque.shape[1] == 0:
        raise ValueError("applied_torque_nm must have shape (N, J), J >= 1")
    lower = _real_array(torque_lower_nm, "torque_lower_nm")
    upper = _real_array(torque_upper_nm, "torque_upper_nm")
    slew = _real_array(slew_limit_nm_s, "slew_limit_nm_s")
    for name, vector in (("lower", lower), ("upper", upper), ("slew", slew)):
        if vector.shape != (torque.shape[1],):
            raise ValueError(f"{name} must have shape (J,)")
    if np.any(lower > upper):
        raise ValueError("torque lower bounds must not exceed upper bounds")
    if np.any(slew < 0.0):
        raise ValueError("slew limits must be nonnegative")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        dt = np.diff(times)
        if not np.all(np.isfinite(dt)):
            raise ValueError("time intervals must be finite")
        if np.any(dt <= 0.0):
            raise ValueError("times_s must be strictly increasing")
        torque_margin = np.minimum(torque - lower, upper - torque).min(axis=0)
        rate = np.abs(np.diff(torque, axis=0) / dt[:, None])
        slew_margin = slew - rate.max(axis=0)
    if not (np.all(np.isfinite(torque_margin)) and np.all(np.isfinite(slew_margin))):
        raise ValueError("derived torque and slew margins must be finite")
    return AppliedEffortAudit(
        tuple(torque_margin.tolist()), tuple(slew_margin.tolist())
    )
