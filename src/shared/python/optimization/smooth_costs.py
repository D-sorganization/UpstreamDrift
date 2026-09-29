"""Smooth surrogates for the swing objectives no gradient solver could use.

Part of epic #8390 (B1/#8396). The legacy formulations are piecewise
constant: :func:`.._swing_objectives.compute_injury_risk` scores through
hard thresholds (+10/+15/+20 steps) and the kinematic-sequencing constraint
locates peaks with ``np.argmax`` — finite-difference gradients are zero
almost everywhere, which silently degrades SLSQP today and would break the
Drake/CasADi/Crocoddyl backends identically.

This module provides differentiable counterparts with the same structure
and thresholds, built from three primitives:

- ``softplus_excess`` — smooth hinge on threshold crossings,
- ``smooth_abs_max`` — log-sum-exp softmax bound on ``max |x|``,
- ``smooth_peak_time`` — softmax-weighted time of peak ``|velocity|``.

The hard scores remain the reporting truth; consistency tests pin the
surrogates to agree with the legacy ranking on separated cases.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "smooth_abs_max",
    "smooth_injury_risk",
    "smooth_kinematic_sequence_penalty",
    "smooth_peak_time",
    "softplus_excess",
]

# Legacy thresholds mirrored from _swing_objectives.compute_injury_risk.
_VELOCITY_LIMIT_RAD_S = 20.0
_TORQUE_LIMIT_FRACTION = 0.8
_TRUNK_ROTATION_LIMIT_RAD = 1.2
_VELOCITY_RISK = 10.0
_TORQUE_RISK = 15.0
_TRUNK_RISK = 20.0
_RISK_CAP = 100.0

_DEFAULT_SHARPNESS = 8.0
_DEFAULT_BETA = 60.0


def softplus_excess(
    value: float, threshold: float, *, sharpness: float = _DEFAULT_SHARPNESS
) -> float:
    """Smooth hinge: ~0 below ``threshold``, growing smoothly above it.

    Normalized so the output approaches 1 as ``value`` clears the threshold
    by a few ``1/sharpness`` widths (a differentiable stand-in for the
    legacy 0/1 indicator).

    Args:
        value: Observed quantity.
        threshold: Crossing point of the legacy step.
        sharpness: Transition steepness (>0); higher is closer to a step.
    """
    if sharpness <= 0:
        raise ValueError("sharpness must be positive")
    z = sharpness * (value - threshold)
    # Numerically-stable logistic: equals the step's height in the limit.
    return float(1.0 / (1.0 + np.exp(-z)))


def smooth_abs_max(values: np.ndarray, *, beta: float = _DEFAULT_BETA) -> float:
    """Differentiable upper bound on ``max(|values|)`` via log-sum-exp.

    Converges to the true max as ``beta`` grows; the bias is bounded by
    ``log(n)/beta``.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    arr = np.abs(np.asarray(values, dtype=float)).reshape(-1)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    scale = float(np.max(arr))
    if scale == 0.0:
        return 0.0
    return float(scale + np.log(np.sum(np.exp(beta * (arr - scale)))) / beta)


def smooth_injury_risk(
    joint_velocities: dict[str, np.ndarray],
    joint_torques: dict[str, np.ndarray],
    joint_angles: dict[str, np.ndarray],
    torque_limits: dict[str, float],
    *,
    sharpness: float = _DEFAULT_SHARPNESS,
    beta: float = _DEFAULT_BETA,
) -> float:
    """Differentiable analogue of ``compute_injury_risk``.

    Mirrors the legacy structure term for term — per-joint velocity spikes,
    per-joint torque saturation, trunk over-rotation — replacing each hard
    step with a logistic on the smooth max, and the ``min(risk, 100)`` cap
    with a ``tanh`` soft cap.

    Returns:
        Smooth risk score in [0, 100).
    """
    if joint_velocities is None or joint_torques is None or joint_angles is None:
        raise ValueError("trajectory channels must be provided")
    risk = 0.0
    for vel in joint_velocities.values():
        peak = smooth_abs_max(vel, beta=beta)
        risk += _VELOCITY_RISK * softplus_excess(
            peak, _VELOCITY_LIMIT_RAD_S, sharpness=sharpness
        )
    for joint, torque in joint_torques.items():
        limit = torque_limits.get(joint, 100.0)
        peak = smooth_abs_max(torque, beta=beta)
        risk += _TORQUE_RISK * softplus_excess(
            peak, _TORQUE_LIMIT_FRACTION * limit, sharpness=sharpness
        )
    trunk = joint_angles.get("trunk_rotation")
    if trunk is not None and np.asarray(trunk).size > 0:
        peak = smooth_abs_max(trunk, beta=beta)
        risk += _TRUNK_RISK * softplus_excess(
            peak, _TRUNK_ROTATION_LIMIT_RAD, sharpness=sharpness
        )
    return float(_RISK_CAP * np.tanh(risk / _RISK_CAP))


def smooth_peak_time(
    times: np.ndarray, velocity: np.ndarray, *, beta: float = _DEFAULT_BETA
) -> float:
    """Softmax-weighted time of peak ``|velocity|`` (argmax surrogate).

    Args:
        times: Sample times, shape ``(T,)``.
        velocity: Velocity trace, shape ``(T,)``.
        beta: Softmax temperature; higher concentrates on the true peak.
    """
    t = np.asarray(times, dtype=float).reshape(-1)
    v = np.abs(np.asarray(velocity, dtype=float).reshape(-1))
    if t.shape != v.shape or t.size == 0:
        raise ValueError("times and velocity must be non-empty and same length")
    if beta <= 0:
        raise ValueError("beta must be positive")
    z = beta * (v - np.max(v))
    w = np.exp(z)
    return float(np.dot(w, t) / np.sum(w))


def smooth_kinematic_sequence_penalty(
    times: np.ndarray,
    ordered_velocities: list[np.ndarray],
    *,
    beta: float = _DEFAULT_BETA,
    margin: float = 0.0,
) -> float:
    """Differentiable proximal-to-distal sequencing penalty.

    Zero when each segment's (smooth) peak time is non-decreasing along the
    chain; otherwise grows smoothly with the size of each ordering
    violation. Replaces the ``argmax``-based constraint in
    ``_swing_constraints.kinematic_sequence_constraint``.

    Args:
        times: Shared sample times, shape ``(T,)``.
        ordered_velocities: Velocity traces ordered proximal → distal.
        beta: Peak-softmax temperature.
        margin: Optional required separation between successive peaks [s].
    """
    if len(ordered_velocities) < 2:
        raise ValueError("need at least two segments to order")
    peaks = [smooth_peak_time(times, v, beta=beta) for v in ordered_velocities]
    penalty = 0.0
    for earlier, later in zip(peaks[:-1], peaks[1:], strict=True):
        violation = (earlier + margin) - later
        # Softplus is smooth everywhere: ~0 for well-ordered peaks,
        # log(2)/k at an exact tie, asymptotically linear in the violation.
        penalty += float(np.logaddexp(0.0, _DEFAULT_SHARPNESS * violation)) / (
            _DEFAULT_SHARPNESS
        )
    return penalty
