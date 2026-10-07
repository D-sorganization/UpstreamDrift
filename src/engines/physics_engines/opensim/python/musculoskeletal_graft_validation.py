"""Validate the muscle graft against the scaled generic Rajagopal model.

Part of issue #11617 (epic #11605), phase 2.  For sampled reference poses the
spec-skeleton leg pose is converted to Rajagopal coordinates (hip angles from the
femur orientation relative to the recovered pelvis frame, the remaining leg
coordinates by name with the sign map below) and every muscle length is compared
between the grafted model and the generic model.  Disagreement therefore measures
the graft (frames, scale, hip-centre mapping) plus the spec's knee simplification
(walker-knee coupled rotations and translations are dropped by the spec).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python import musculoskeletal_graft as graft
from src.engines.physics_engines.opensim.python.musculoskeletal_spec_bundle import (
    SpecBundle,
)
from src.shared.python.contracts import require

#: Rajagopal coordinate = sign * spec coordinate, per side.  The spec's left knee
#: hinge runs the opposite way to Rajagopal's; the left hip adduction/rotation
#: signs reflect Rajagopal's mirrored left hip axes (applied to the geometric
#: Euler angles).  Right side is the identity.
RAJAGOPAL_SIGNS: dict[str, dict[str, float]] = {
    "r": {
        "knee": 1.0,
        "ankle": 1.0,
        "subtalar": 1.0,
        "mtp": 1.0,
        "add": 1.0,
        "rot": 1.0,
    },
    "l": {
        "knee": -1.0,
        "ankle": 1.0,
        "subtalar": 1.0,
        "mtp": 1.0,
        "add": -1.0,
        "rot": -1.0,
    },
}
_SINGLE = {
    "knee": "knee_angle_",
    "ankle": "ankle_angle_",
    "subtalar": "subtalar_angle_",
    "mtp": "mtp_angle_",
}


def _rotation(model: Any, state: Any, body: str) -> np.ndarray:
    t = model.getBodySet().get(body).getTransformInGround(state)
    return np.array([[t.R().get(i, j) for j in range(3)] for i in range(3)])


def _set(model: Any, state: Any, name: str, value: float) -> None:
    model.getCoordinateSet().get(name).setValue(state, float(value), False)


def muscle_length_agreement(
    model: Any,
    base_scaled: Any,
    bundle: SpecBundle,
    pelvis_rotation: np.ndarray,
    frames: list[int],
) -> dict[str, Any]:
    """Compare grafted and generic muscle lengths at ``frames`` of ``bundle``.

    Returns RMS/max absolute length difference (mm) per side over all muscles of
    that side and frames, plus the worst muscles.
    """
    require(len(frames) > 0, "need at least one frame")
    state = model.initSystem()
    base_state = base_scaled.initSystem()
    errors: dict[str, list[tuple[str, float]]] = {"r": [], "l": []}
    for k in frames:
        q = bundle.q[k]
        for i, name in enumerate(bundle.coordinate_order):
            _set(model, state, name, q[i])
        for side in graft.SIDES:
            _set(
                model,
                state,
                f"knee_angle_{side}_beta",
                q[bundle.index(f"knee_angle_{side}")],
            )
        model.realizePosition(state)
        for c in base_scaled.getCoordinateSet():
            if not c.get_locked():
                c.setValue(base_state, 0.0, False)
        for side in graft.SIDES:
            sign = RAJAGOPAL_SIGNS[side]
            rel = (
                pelvis_rotation.T
                @ _rotation(model, state, "Hip").T
                @ _rotation(model, state, f"femur_{side}")
            )
            flex, add, rot = Rotation.from_matrix(rel).as_euler("ZXY")
            _set(base_scaled, base_state, f"hip_flexion_{side}", flex)
            _set(base_scaled, base_state, f"hip_adduction_{side}", sign["add"] * add)
            _set(base_scaled, base_state, f"hip_rotation_{side}", sign["rot"] * rot)
            for key, prefix in _SINGLE.items():
                _set(
                    base_scaled,
                    base_state,
                    prefix + side,
                    sign[key] * q[bundle.index(prefix + side)],
                )
            _set(
                base_scaled,
                base_state,
                f"knee_angle_{side}_beta",
                sign["knee"] * q[bundle.index(f"knee_angle_{side}")],
            )
        base_scaled.realizePosition(base_state)
        for muscle in model.getMuscles():
            side = graft.side_of(muscle.getName())
            reference = base_scaled.getMuscles().get(muscle.getName())
            diff = 1000.0 * (muscle.getLength(state) - reference.getLength(base_state))
            errors[str(side)].append((muscle.getName(), float(diff)))
    out: dict[str, Any] = {"frames": list(frames)}
    for side, rows in errors.items():
        values = np.array([v for _, v in rows])
        worst = sorted(rows, key=lambda r: -abs(r[1]))[:5]
        out[side] = {
            "rms_mm": float(np.sqrt(np.mean(values**2))),
            "max_abs_mm": float(np.abs(values).max()),
            "worst": [(n, round(v, 2)) for n, v in worst],
        }
    return out
