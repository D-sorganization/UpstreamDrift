"""Foot-joint tracking measures of a bar-loaded lift run (OSL-16, #13104).

Under a 100 kg bar the toe (mtp), subtalar and ankle coordinates of the
tracked squat ran away from the IK they track: the stiffest loaded foot
contact mode lay outside the RK4 stability region of the default substep (see
:mod:`~src.shared.python.motion_matching.same_input.stiffness`).  These
measures make that failure, and its fix, a number:

* :func:`foot_joint_error_rad` is the largest absolute error of each foot
  coordinate against the reference, over the run;
* :func:`foot_point_deviation_m` turns those errors into a length: each foot
  point (the side's contact-sphere centres and a toe-tip point) is expressed
  in its tibia frame, so the root and leg errors cancel and only the foot
  joints move it; the largest distance from the reference foot point over the
  run, per side, is compared with that side's LIFT-5 ankle IK residual bound.

The penalty contact's static sink under body plus bar moves the whole lifter
(about 3 mm at rest); it does not enter the tibia-frame measure.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

SIDES = ("r", "l")
FOOT_JOINTS: tuple[str, ...] = tuple(
    f"{joint}_{side}"
    for side in SIDES
    for joint in ("ankle_angle", "subtalar_angle", "mtp_angle")
)
#: Toe-tip proxy on the toes body (m, toes frame): 6 cm forward of the mtp
#: joint, about the phalanx length of the Rajagopal toes segment, so mtp
#: rotation shows as a length the way the calcn spheres show ankle and
#: subtalar rotation.
TOE_TIP_M = (0.06, 0.0, 0.0)

__all__ = [
    "FOOT_JOINTS",
    "TOE_TIP_M",
    "foot_joint_error_rad",
    "foot_point_deviation_m",
]


def _checked(order: Sequence[str], q_run: Array, q_ref: Array) -> tuple[Array, Array]:
    run = np.asarray(q_run, dtype=float)
    ref = np.asarray(q_ref, dtype=float)
    if run.ndim != 2 or run.shape[1] != len(order) or run.shape[0] < 1:
        raise ValueError("q_run must be (rows, len(order))")
    if run.shape[0] > ref.shape[0] or ref.shape[1:] != run.shape[1:]:
        raise ValueError("q_ref must have at least q_run's rows and its columns")
    if not (np.isfinite(run).all() and np.isfinite(ref).all()):
        raise ValueError("q_run and q_ref must be finite")
    return run, ref[: run.shape[0]]


def foot_joint_error_rad(
    order: Sequence[str], q_run: Array, q_ref: Array
) -> dict[str, float]:
    """Largest ``|q_run - q_ref|`` (rad) of each foot coordinate over the run.

    ``q_ref`` may be longer than ``q_run`` (a run that stopped early is
    compared on its reached rows).  Raises ``ValueError`` for mismatched or
    nonfinite arrays or a coordinate order without the foot joints.
    """
    run, ref = _checked(order, q_run, q_ref)
    missing = [name for name in FOOT_JOINTS if name not in order]
    if missing:
        raise ValueError(f"coordinate order lacks foot joints {missing}")
    index = list(order)
    return {
        name: float(np.abs(run[:, index.index(name)] - ref[:, index.index(name)]).max())
        for name in FOOT_JOINTS
    }


def _tibia_frame_points(model: object, order: Sequence[str], q: Array) -> dict:
    """Foot points of each side in that side's tibia frame (m)."""
    model.frame_poses(dict(zip(order, map(float, q), strict=True)))  # type: ignore[attr-defined]
    mj, data = model.model, model.data  # type: ignore[attr-defined]
    spheres = model._spheres  # type: ignore[attr-defined]  # noqa: SLF001 - sphere sites
    points = {}
    for side in SIDES:
        tibia = mj.body(f"tibia_{side}").id
        rot, origin = data.xmat[tibia].reshape(3, 3), data.xpos[tibia]
        toes = mj.body(f"toes_{side}").id
        world = [
            data.site_xpos[info["site_id"]]
            for name, info in spheres.items()
            if name.endswith(f"_{side}")
        ]
        world.append(data.xpos[toes] + data.xmat[toes].reshape(3, 3) @ TOE_TIP_M)
        points[side] = np.array([rot.T @ (p - origin) for p in world])
    return points


def foot_point_deviation_m(
    spec_bytes: bytes, order: Sequence[str], q_run: Array, q_ref: Array
) -> dict[str, float]:
    """Largest tibia-frame foot-point distance (m) of ``q_run`` from ``q_ref``.

    Returns ``{"r": ..., "l": ...}``: per side, the maximum over rows and foot
    points (contact-sphere centres and :data:`TOE_TIP_M`) of the distance
    between the run's and the reference's point, both in the tibia frame
    (MuJoCo forward kinematics of ``spec_bytes``).
    """
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )

    run, ref = _checked(order, q_run, q_ref)
    model = NativeMujocoFullBodyModel(bytes(spec_bytes))
    peak = dict.fromkeys(SIDES, 0.0)
    for here, there in zip(run, ref, strict=True):
        a = _tibia_frame_points(model, order, here)
        b = _tibia_frame_points(model, order, there)
        for side in SIDES:
            gap = float(np.linalg.norm(a[side] - b[side], axis=1).max())
            peak[side] = max(peak[side], gap)
    return peak
