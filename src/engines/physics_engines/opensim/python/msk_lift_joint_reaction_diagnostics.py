"""Diagnostics and CI fingerprints for squat joint-reaction acceleration checks."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.engine_core.opensim_runtime import import_opensim

# Bumped when joint-reaction kinematics or realization order changes; CI asserts this.
MSK_LIFT_JR_KINEMATICS_REVISION = 6


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


__all__ = [
    "MSK_LIFT_JR_KINEMATICS_REVISION",
    "joint_reaction_source_fingerprint",
    "record_max_udot_error",
]


def joint_reaction_source_fingerprint() -> dict[str, Any]:
    """Runtime and source paths for squat joint-reaction acceleration checks."""
    import scipy

    osim = import_opensim()
    parent = Path(__file__).resolve().parent
    kin_path = parent / "msk_lift_simbody_kinematics.py"
    frame_path = parent / "msk_lift_joint_reaction_frame.py"
    return {
        "kinematics_revision": MSK_LIFT_JR_KINEMATICS_REVISION,
        "joint_reactions_module": str(parent / "msk_lift_joint_reactions.py"),
        "joint_reaction_frame_module": str(frame_path),
        "simbody_kinematics_module": str(kin_path),
        "simbody_kinematics_sha256": _source_sha256(kin_path),
        "joint_reaction_frame_sha256": _source_sha256(frame_path),
        "opensim_version": str(osim.GetVersion()),
        "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__),
        "python_executable": sys.executable,
        "constraints_enforced_for_id": False,
        "realization_order": (
            "disable couplers; coordinates+setTime; muscle+residual overrides; "
            "realizeDynamics; InverseDynamicsSolver; residual torques; "
            "realizeAcceleration"
        ),
    }


def record_max_udot_error(
    context: dict[str, Any],
    *,
    row: int,
    time_s: float,
    coordinates: tuple[str, ...],
    u_index: dict[str, int],
    expected_udot: np.ndarray,
    actual_udot: Any,
) -> float:
    """Return max |UDot - expected| and update context['acceleration_worst'] when larger."""
    acc_err = 0.0
    worst = context.setdefault("acceleration_worst", {"error_rad_s2": 0.0})
    for name, acc in zip(coordinates, expected_udot, strict=True):
        err = abs(actual_udot.get(u_index[name]) - acc)
        if err > acc_err:
            acc_err = err
        if err > float(worst.get("error_rad_s2", 0.0)):
            worst.update(
                {
                    "error_rad_s2": float(err),
                    "frame_index": int(row),
                    "time_s": float(time_s),
                    "coordinate": str(name),
                    "mobility_index": int(u_index[name]),
                    "target_udot_rad_s2": float(acc),
                    "realized_udot_rad_s2": float(actual_udot.get(u_index[name])),
                }
            )
    return float(acc_err)
