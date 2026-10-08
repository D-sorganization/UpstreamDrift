"""MuJoCo and MyoSuite ClubheadSeries adapter (GCV-16).

Forward kinematics of the club body plus ``mj_jac`` at the body origin give the
origin velocity and body angular velocity; the shared kernel transports them
to the face centre, so no visual-only site is required.  MyoSuite simulations
wrap a MuJoCo model, so :func:`clubhead_series_from_myosuite` delegates here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..clubhead_series import ClubheadSeries
from .club_face import (
    NATIVE_CLUB_FACE,
    ClubFaceSpec,
    check_trajectory,
    rigid_body_series,
)


def clubhead_series_from_mujoco(
    model: Any,
    times_s: object,
    qpos: object,
    qvel: object,
    body_name: str,
    spec: ClubFaceSpec = NATIVE_CLUB_FACE,
) -> ClubheadSeries:
    """Build a ClubheadSeries from a MuJoCo rollout of ``(qpos, qvel)``.

    Raises ``ValueError`` for unknown bodies or mis-sized trajectories.
    """
    import mujoco

    t, q, v = check_trajectory(times_s, qpos, qvel, "qpos")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body < 0:
        raise ValueError(f"MuJoCo model has no body named {body_name!r}")
    if q.shape[1] != model.nq or v.shape[1] != model.nv:
        raise ValueError(
            f"trajectory widths ({q.shape[1]}, {v.shape[1]}) != model "
            f"(nq={model.nq}, nv={model.nv})"
        )
    data = mujoco.MjData(model)
    n = t.shape[0]
    pos, rot, lin, ang = (
        np.zeros((n, 3)),
        np.zeros((n, 3, 3)),
        np.zeros((n, 3)),
        np.zeros((n, 3)),
    )
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for i in range(n):
        data.qpos[:] = q[i]
        data.qvel[:] = v[i]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        point = data.xpos[body].copy()
        mujoco.mj_jac(model, data, jacp, jacr, point, body)
        pos[i], rot[i] = point, data.xmat[body].reshape(3, 3)
        lin[i], ang[i] = jacp @ v[i], jacr @ v[i]
    return rigid_body_series(t, pos, rot, lin, ang, spec)


def clubhead_series_from_myosuite(
    sim: Any,
    times_s: object,
    qpos: object,
    qvel: object,
    body_name: str,
    spec: ClubFaceSpec = NATIVE_CLUB_FACE,
) -> ClubheadSeries:
    """MyoSuite wrapper: its ``sim.model`` is a MuJoCo model."""
    model = getattr(sim, "model", None)
    if model is None:
        raise ValueError("MyoSuite sim must expose a .model attribute")
    inner = getattr(model, "_model", model)
    return clubhead_series_from_mujoco(inner, times_s, qpos, qvel, body_name, spec)
