"""Pinocchio ClubheadSeries adapter (GCV-16)."""

from __future__ import annotations

import numpy as np

from ..clubhead_series import ClubheadSeries
from .club_face import (
    NATIVE_CLUB_FACE,
    ClubFaceSpec,
    check_trajectory,
    rigid_body_series,
)


def clubhead_series_from_pinocchio(
    model: object,
    times_s: object,
    q: object,
    v: object,
    frame_name: str,
    spec: ClubFaceSpec = NATIVE_CLUB_FACE,
) -> ClubheadSeries:
    """``updateFramePlacements`` + ``getFrameVelocity(LOCAL_WORLD_ALIGNED)``.

    The frame must be the club body frame (origin at the head).
    """
    import pinocchio as pin

    t, qa, va = check_trajectory(times_s, q, v)
    if not model.existFrame(frame_name):
        raise ValueError(f"Pinocchio model has no frame named {frame_name!r}")
    if qa.shape[1] != model.nq or va.shape[1] != model.nv:
        raise ValueError(
            f"trajectory widths ({qa.shape[1]}, {va.shape[1]}) != model "
            f"(nq={model.nq}, nv={model.nv})"
        )
    fid = model.getFrameId(frame_name)
    data = model.createData()
    n = t.shape[0]
    pos, rot, lin, ang = (
        np.zeros((n, 3)),
        np.zeros((n, 3, 3)),
        np.zeros((n, 3)),
        np.zeros((n, 3)),
    )
    for i in range(n):
        pin.forwardKinematics(model, data, qa[i], va[i])
        pin.updateFramePlacements(model, data)
        placement = data.oMf[fid]
        vel = pin.getFrameVelocity(model, data, fid, pin.LOCAL_WORLD_ALIGNED)
        pos[i], rot[i] = placement.translation, placement.rotation
        lin[i], ang[i] = vel.linear, vel.angular
    return rigid_body_series(t, pos, rot, lin, ang, spec)
