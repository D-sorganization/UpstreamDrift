"""OpenSim address foot toe-out seed (OSV-4, #11730).

OpenSim's ``hip_rotation_*`` is positive = internal rotation (toe-in) for BOTH
legs (the left axes are mirrored), unlike the shared full-body spec whose left
axis is not mirrored. Foot yaw also depends on the flexion/knee/ankle of the
nominal address pose (a straight ``-theta`` lands a few degrees off), so the
rotation is solved against OpenSim forward kinematics when the runtime exists.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.motion_matching.foot_progression import (
    model_long_axis,
    progression_angle_deg,
)

__all__ = ["apply_toe_out_seed", "opensim_foot_progression_deg"]

#: OpenSim world: x forward, y up, z to the golfer's right. The lead (left)
#: foot is on -z, so the target direction is -z.
_UP = np.array([0.0, 1.0, 0.0])
_TARGET = np.array([0.0, 0.0, -1.0])
_ROLES = {"right": "trail", "left": "lead"}
_SUFFIX = {"right": "r", "left": "l"}
TOLERANCE_DEG = 0.1
MAX_PASSES = 6


def opensim_foot_progression_deg(
    model_path: Path | str, q_dict: Mapping[str, float]
) -> dict[str, float]:
    """Toe-out per foot (deg) of the OpenSim model at ``q_dict`` (radians).

    Raises ``ImportError`` when the OpenSim runtime is missing.
    """
    import opensim as osim

    model = osim.Model(str(model_path))
    state = model.initSystem()
    coords = model.updCoordinateSet()
    for name, value in q_dict.items():
        coords.get(name).setValue(state, float(value), False)
    model.realizePosition(state)

    def position(body: str) -> np.ndarray:
        v = model.getBodySet().get(body).getPositionInGround(state)
        return np.array([v.get(i) for i in range(3)])

    out: dict[str, float] = {}
    for side, sfx in _SUFFIX.items():
        axis = model_long_axis(position(f"calcn_{sfx}"), position(f"toes_{sfx}"), _UP)
        out[side] = progression_angle_deg(
            axis, target_axis=_TARGET, up=_UP, foot_role=_ROLES[side]
        )
    return out


def apply_toe_out_seed(
    model_path: Path | str,
    q_dict: MutableMapping[str, float],
    toe_out_deg: Mapping[str, float],
) -> dict[str, Any]:
    """Set ``hip_rotation_{r,l}`` in ``q_dict`` (radians) for the requested toe-out.

    ``toe_out_deg`` maps ``left``/``right`` to degrees. Returns a receipt block
    with the achieved angles; ``method`` is ``fk`` when solved against OpenSim
    forward kinematics and ``direct`` (angle = -hip_rotation, unverified) when
    the runtime is unavailable.
    """
    for side in _SUFFIX:
        if side not in toe_out_deg or not math.isfinite(toe_out_deg[side]):
            raise ValueError(f"toe_out_deg needs a finite '{side}' value")
    for side, sfx in _SUFFIX.items():
        q_dict[f"hip_rotation_{sfx}"] = -math.radians(float(toe_out_deg[side]))
    try:
        for _ in range(MAX_PASSES):
            achieved = opensim_foot_progression_deg(model_path, q_dict)
            errors = {s: toe_out_deg[s] - achieved[s] for s in _SUFFIX}
            if max(abs(e) for e in errors.values()) <= TOLERANCE_DEG:
                break
            for side, sfx in _SUFFIX.items():
                # internal rotation is positive: more toe-out needs less rotation
                q_dict[f"hip_rotation_{sfx}"] -= math.radians(errors[side])
        method = "fk"
    except ImportError:
        achieved, method = {}, "direct"
    return {
        "method": method,
        "target_deg": {s: float(toe_out_deg[s]) for s in _SUFFIX},
        "model_deg": {s: float(v) for s, v in achieved.items()},
        "hip_rotation_rad": {
            s: float(q_dict[f"hip_rotation_{sfx}"]) for s, sfx in _SUFFIX.items()
        },
    }
