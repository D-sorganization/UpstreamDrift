"""Torque-history margin validation for feasible strategies (#15494).

Checks a *sampled* joint-torque history against declared torque limits, an
optional slew limit and optional joint velocities.  It does not generate
strategies, solve an allocation, forward-replay a swing or certify clinical
safety; a pass only means the supplied samples satisfy the declared priors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import check_shape, readonly_float

__all__ = ["torque_history_margins"]


def _trapezoid(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Per-column trapezoid integral of ``values`` (n, nu) over ``times``."""
    dt = np.diff(times)[:, None]
    return np.sum(0.5 * (values[1:] + values[:-1]) * dt, axis=0)


def torque_history_margins(
    times_s: ArrayLike,
    torque_nm: ArrayLike,
    *,
    tau_lo: ArrayLike,
    tau_hi: ArrayLike,
    safety_fraction: float = 1.0,
    max_slew_nm_s: float | None = None,
    velocity_rad_s: ArrayLike | None = None,
) -> dict[str, Any]:
    """Report limit margins, slew and mechanical work of a torque history.

    Preconditions: ``torque_nm`` is ``(n, nu)`` in N m on a strictly increasing
    ``times_s`` (s); ``tau_lo < 0 < tau_hi`` per joint (N m); the effective
    envelope is ``safety_fraction`` times the limits, with the fraction in
    ``(0, 1]``.  ``velocity_rad_s`` (optional, ``(n, nu)``) enables work.
    The history is read as piecewise linear, so checking the samples also
    bounds every point between them.

    Postconditions: violations are listed, never clipped.  ``margin_nm`` is the
    per-joint minimum signed distance to the effective envelope (negative means
    violated); ``narrowest`` names the joint/sample attaining the global
    minimum.  ``slew_ok`` and the work fields are ``None`` when their inputs
    were not declared.  Signed work is the trapezoid integral of ``tau*omega``;
    positive work integrates its positive part.  The result is JSON-compatible.

    Raises:
        ValueError: for non-finite, mis-shaped or unordered inputs, limits not
            straddling zero, or an out-of-range fraction or slew limit.
    """
    times = readonly_float(times_s, "times_s")
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("times_s must be 1-D, strictly increasing, length >= 2")
    tau = readonly_float(torque_nm, "torque_nm")
    if tau.ndim != 2:
        raise ValueError("torque_nm must be 2-D (samples, joints)")
    check_shape(tau, (times.size, tau.shape[1]), "torque_nm")
    nu = tau.shape[1]
    lo = readonly_float(tau_lo, "tau_lo")
    hi = readonly_float(tau_hi, "tau_hi")
    check_shape(lo, (nu,), "tau_lo")
    check_shape(hi, (nu,), "tau_hi")
    if not (np.all(lo < 0.0) and np.all(hi > 0.0)):
        raise ValueError("require tau_lo < 0 < tau_hi for every joint")
    if not (np.isfinite(safety_fraction) and 0.0 < safety_fraction <= 1.0):
        raise ValueError("safety_fraction must be in (0, 1]")
    if max_slew_nm_s is not None and not (
        np.isfinite(max_slew_nm_s) and max_slew_nm_s > 0.0
    ):
        raise ValueError("max_slew_nm_s must be finite and positive")

    slack = np.minimum(safety_fraction * hi - tau, tau - safety_fraction * lo)
    sample, joint = np.unravel_index(int(np.argmin(slack)), slack.shape)
    violations = [
        {"joint": int(j), "sample": int(k), "excess_nm": float(-slack[k, j])}
        for k, j in zip(*np.nonzero(slack < 0.0), strict=True)
    ]
    peak_slew = np.max(np.abs(np.diff(tau, axis=0)) / np.diff(times)[:, None], axis=0)
    slew_ok = None
    if max_slew_nm_s is not None:
        slew_ok = bool(np.all(peak_slew <= max_slew_nm_s))

    signed_work = positive_work = None
    if velocity_rad_s is not None:
        omega = readonly_float(velocity_rad_s, "velocity_rad_s")
        check_shape(omega, tau.shape, "velocity_rad_s")
        power = tau * omega
        signed_work = _trapezoid(power, times).tolist()
        positive_work = _trapezoid(np.maximum(power, 0.0), times).tolist()

    return {
        "within_limits": not violations,
        "safety_fraction": float(safety_fraction),
        "margin_nm": np.min(slack, axis=0).tolist(),
        "narrowest": {
            "joint": int(joint),
            "sample": int(sample),
            "margin_nm": float(slack[sample, joint]),
        },
        "violations": violations,
        "peak_slew_nm_s": peak_slew.tolist(),
        "slew_ok": slew_ok,
        "signed_work_j": signed_work,
        "positive_work_j": positive_work,
    }
