"""Hip-calibration-aware hip retarget for MyoSuite legs (#12052, OSV-6 #11737).

The ground-support pipeline fits leg coordinates in a hip-calibrated spec:
``apply_hip_calibration`` rewrites each ``hip_{l,r}`` ``parent_to_base`` with
the functional hip centre, the calibrated pelvis axes and a zero-twist
rotation. The same ``hip_flexion/adduction/rotation`` numbers then describe a
different femur orientation than in stock MyoSuite, so a coordinate-value map
turns the address feet 33-45 degrees.

This module maps the *orientation* instead. In the stock (uncalibrated) spec
the femur body frame (follower frame times ``child_to_follower`` inverse) is
the MyoSuite femur frame, and the spec pelvis (``LowerTorso`` follower) frame
is the MyoSuite pelvis frame rotated by ``MYOSUITE_PELVIS_IN_SPEC_PELVIS``
(OpenSim Y-up to the spec's Z-up). So:

    R_myo(pelvis -> femur) = P^T  R_spec(pelvis follower -> femur body)

which is then decomposed into MyoSuite's hinge order (flexion, adduction,
rotation; intrinsic, axes from ``myolegs``). On the stock spec this is the
identity map; on a calibrated spec it absorbs the rewritten hip frames.
Knee, ankle, subtalar and mtp frames are untouched by the calibration and keep
the coordinate map.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

Array: TypeAlias = NDArray[np.float64]

#: MyoSuite pelvis frame expressed in the spec pelvis frame (Rx(+90 deg)); the
#: ``myolegs`` pelvis ``body_quat`` (0.7071, 0.7071, 0, 0).
MYOSUITE_PELVIS_IN_SPEC_PELVIS: Array = Rotation.from_euler(
    "x", 90.0, degrees=True
).as_matrix()

_HIP = ("flexion", "adduction", "rotation")
#: Hinge axes of ``hip_{flexion,adduction,rotation}_{side}`` in the MyoSuite
#: femur body frame, in MJCF order (``myolegs.xml``; femur ``body_quat`` is 1).
MYOSUITE_HIP_AXES: dict[str, tuple[tuple[float, float, float], ...]] = {
    "r": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "l": ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
}
#: Smallest |cos(adduction)| accepted before the ZXY decomposition is singular.
_GIMBAL_COS = 1e-6


def _side(side: str) -> str:
    if side not in MYOSUITE_HIP_AXES:
        raise ValueError(f"side must be 'r' or 'l', got {side!r}")
    return side


def _rotation(transform: Any) -> Array:
    return np.asarray(transform, dtype=np.float64)[:3, :3]


def spec_femur_in_pelvis(
    spec: Mapping[str, Any], coords: Mapping[str, float], side: str
) -> Array:
    """Femur body frame in the pelvis follower frame of ``spec`` (3x3).

    ``coords`` holds the hip coordinates in radians. Raises ``ValueError`` for
    an unknown side, a missing hip joint or coordinate, or a hip primitive
    that is not a rotation.
    """
    _side(side)
    joints = {str(j["child"]): j for j in spec["joints"]}
    hip = joints.get(f"femur_{side}")
    if hip is None:
        raise ValueError(f"spec has no joint for femur_{side}")
    parent = joints.get(str(hip["parent"]))
    rot = np.eye(3) if parent is None else _rotation(parent["child_to_follower"]).T
    rot = rot @ _rotation(hip["parent_to_base"])
    for prim in hip["primitives"]:
        kind, name = str(prim["primitive"]), str(prim["coordinate"])
        if kind[0] != "R":
            raise ValueError(f"hip_{side} primitive {kind} is not a rotation")
        if name not in coords:
            raise ValueError(f"coords lack {name}")
        axis = np.eye(3)["xyz".index(kind[1])]
        rot = rot @ Rotation.from_rotvec(axis * float(coords[name])).as_matrix()
    return rot @ _rotation(hip["child_to_follower"]).T


def myosuite_hip_angles(rotation: Array, side: str) -> dict[str, float]:
    """MyoSuite hip coordinates (rad) that produce ``rotation`` (femur in pelvis).

    Postcondition: the hinges of ``MYOSUITE_HIP_AXES[side]`` applied in order
    reproduce ``rotation``. Raises ``ValueError`` near adduction of +-90 deg.
    """
    axes = MYOSUITE_HIP_AXES[_side(side)]
    rot = np.asarray(rotation, dtype=np.float64)
    if rot.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {rot.shape}")
    # Both sides are Z X Y about the body axes; only the X and Y signs differ.
    # Rz Rx Ry has R[2] = (-cos(x) sin(y), sin(x), cos(x) cos(y)).
    if np.hypot(rot[2, 0], rot[2, 2]) < _GIMBAL_COS:
        raise ValueError("hip adduction at +-90 deg: gimbal lock")
    flexion, adduction, rotation_y = Rotation.from_matrix(rot).as_euler("ZXY")
    signs = (axes[0][2], axes[1][0], axes[2][1])
    values = (flexion, adduction, rotation_y)
    return {
        f"hip_{name}_{side}": float(sign * value)
        for name, sign, value in zip(_HIP, signs, values, strict=True)
    }


def calibrated_hip_targets(
    spec: Mapping[str, Any], coords: Mapping[str, float]
) -> dict[str, float]:
    """MyoSuite ``hip_*_{r,l}`` (rad) with the femur orientation of ``spec``.

    ``coords`` maps the spec's coordinate names to radians and must carry all
    six hip coordinates. On an uncalibrated spec the result equals the input.
    """
    names = [f"hip_{n}_{s}" for s in ("r", "l") for n in _HIP]
    missing = [name for name in names if name not in coords]
    if missing:
        raise ValueError(f"coords lack hip coordinates {missing}")
    out: dict[str, float] = {}
    for side in ("r", "l"):
        femur = MYOSUITE_PELVIS_IN_SPEC_PELVIS.T @ spec_femur_in_pelvis(
            spec, coords, side
        )
        out.update(myosuite_hip_angles(femur, side))
    return out
