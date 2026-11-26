"""Preregistered admission block for a canonical-v2 solution bundle.

Part of #15808 WP-0c. Legacy bundles omit the block. A declared block must
name its controller, stay inside the recorded interval, carry per-frame
validity, and keep the five torque channels in the accounting identity.
Fitting and replay are out of scope; this module only fails closed.
"""

from __future__ import annotations

import math
from typing import Any, NoReturn

import numpy as np

# Two degrees, stored in radians so the bundle stays in SI.
_REPLAY_JOINT_RMS_RAD = math.radians(2.0)

PREREGISTERED_TOLERANCES: dict[str, float] = {
    "root_force_bw_fraction": 0.02,
    "root_moment_bw_height_fraction": 0.02,
    "impact_impulse_error_fraction": 0.05,
    "stance_slip_m": 0.005,
    "grip_weld_gap_m": 0.002,
    "replay_joint_rms_rad": _REPLAY_JOINT_RMS_RAD,
    "feedback_rms_capacity_fraction": 0.05,
}

TORQUE_CHANNELS = (
    "tau_feedforward",
    "tau_intervention",
    "tau_balance_correction",
    "tau_saturation",
    "tau_realized",
)
FRAME_VALID = "frame_valid"
_ADMISSION_KEYS = frozenset({"controller", "interval", "tolerances"})
_CONTROLLER_KEYS = frozenset({"name", "feedback", "capacity_nm"})
_INTERVAL_KEYS = frozenset({"t_start", "t_end"})
_IDENTITY_ATOL = 1e-9
_TOLERANCE_ATOL = 1e-12


def _fail(message: str) -> NoReturn:
    """Raise ``SolutionBundleError`` for ``message``."""
    from src.shared.python.optimization.solution_bundle import SolutionBundleError

    raise SolutionBundleError(message)


def _finite_number(value: object) -> bool:
    """Return whether ``value`` is a finite int or float, excluding bool."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_mapping(value: object, what: str, keys: frozenset[str]) -> dict[str, Any]:
    """Return ``value`` when it is an object with exactly ``keys``."""
    if not isinstance(value, dict):
        _fail(f"{what} must be an object")
    extra = sorted(set(value) - keys)
    missing = sorted(keys - set(value))
    if extra or missing:
        _fail(
            f"{what} keys must be {sorted(keys)}, got extra {extra} missing {missing}"
        )
    return value


def _validate_controller(controller: object) -> None:
    """Require a named controller with positive finite capacity."""
    body = _require_mapping(controller, "controller", _CONTROLLER_KEYS)
    for key in ("name", "feedback"):
        if not isinstance(body[key], str) or not body[key].strip():
            _fail(f"controller.{key} must be a non-empty string")
    capacity = body["capacity_nm"]
    if not _finite_number(capacity) or float(capacity) <= 0.0:
        _fail("controller.capacity_nm must be a finite number greater than 0")


def _validate_interval(interval: object, time: np.ndarray) -> tuple[float, float]:
    """Return an admitted interval that lies inside ``time``."""
    body = _require_mapping(interval, "interval", _INTERVAL_KEYS)
    start, end = body["t_start"], body["t_end"]
    if not _finite_number(start) or not _finite_number(end) or not start < end:
        _fail("interval needs finite t_start < t_end")
    if start < float(time[0]) or end > float(time[-1]):
        _fail("interval must lie inside the recorded time base")
    return float(start), float(end)


def _validate_tolerances(tolerances: object) -> None:
    """Require exactly the preregistered tolerance values."""
    body = _require_mapping(
        tolerances, "tolerances", frozenset(PREREGISTERED_TOLERANCES)
    )
    for key, expected in PREREGISTERED_TOLERANCES.items():
        value = body[key]
        if not _finite_number(value) or not math.isclose(
            float(value), expected, rel_tol=0.0, abs_tol=_TOLERANCE_ATOL
        ):
            _fail(f"tolerances.{key} must stay at the preregistered value {expected}")


def _validate_flags(
    flags: np.ndarray, time: np.ndarray, start: float, end: float
) -> None:
    """Require a boolean mask that is false outside the admitted interval."""
    if flags.dtype != np.dtype(bool) or flags.shape != (time.shape[0],):
        _fail("frame_valid must be a boolean mask with one flag per sample")
    outside = (time < start) | (time > end)
    if np.any(flags[outside]):
        _fail("frame_valid must be false outside the admitted interval")
    if not np.any(flags):
        _fail("frame_valid must mark at least one admitted sample")


def _as_channel(
    arrays: dict[str, np.ndarray], name: str, n: int, joints: int
) -> np.ndarray:
    """Return one finite torque channel with shape ``(n, joints)``."""
    if name not in arrays:
        _fail(f"admission is missing torque channel {name}")
    value = np.asarray(arrays[name])
    if value.dtype.kind not in "fiu" or value.shape != (n, joints):
        _fail(f"{name} must have numeric shape {(n, joints)}")
    if not np.all(np.isfinite(value)):
        _fail(f"{name} contains non-finite values")
    return value


def validate_admission(
    admission: object,
    *,
    time: np.ndarray,
    n_joints: int,
    arrays: dict[str, np.ndarray],
) -> None:
    """Raise ``SolutionBundleError`` unless ``admission`` matches the contract.

    ``None`` is a legacy recording. It must not carry validity flags or the
    intervention torque channels. A declared block is checked in full.
    """
    channel_names = (*TORQUE_CHANNELS, FRAME_VALID)
    if admission is None:
        present = sorted(name for name in channel_names if name in arrays)
        if present:
            _fail(
                "admission channels require an admission block: " + ", ".join(present)
            )
        return
    if not isinstance(admission, dict):
        _fail("admission must be an object or null")
    extra = sorted(set(admission) - _ADMISSION_KEYS)
    if extra:
        _fail(f"admission has unknown keys {extra}")
    _validate_controller(admission.get("controller"))
    start, end = _validate_interval(admission.get("interval"), time)
    _validate_tolerances(admission.get("tolerances"))
    n = int(time.shape[0])
    if FRAME_VALID not in arrays:
        _fail("admission is missing frame_valid")
    _validate_flags(np.asarray(arrays[FRAME_VALID]), time, start, end)
    channels = [_as_channel(arrays, name, n, n_joints) for name in TORQUE_CHANNELS]
    feed, intervention, balance, saturation, realized = channels
    summed = feed + intervention + balance + saturation
    if not np.allclose(realized, summed, rtol=0.0, atol=_IDENTITY_ATOL):
        _fail("tau_realized must equal the sum of the other torque channels")
