"""Drake ClubheadSeries adapter (GCV-16)."""

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


def clubhead_series_from_drake(
    plant: Any,
    times_s: object,
    q: object,
    v: object,
    body_name: str,
    spec: ClubFaceSpec = NATIVE_CLUB_FACE,
) -> ClubheadSeries:
    """Body pose and spatial velocity of a finalized ``MultibodyPlant`` body."""
    t, qa, va = check_trajectory(times_s, q, v)
    if not plant.HasBodyNamed(body_name):
        raise ValueError(f"Drake plant has no body named {body_name!r}")
    if qa.shape[1] != plant.num_positions() or va.shape[1] != plant.num_velocities():
        raise ValueError(
            f"trajectory widths ({qa.shape[1]}, {va.shape[1]}) != plant "
            f"({plant.num_positions()}, {plant.num_velocities()})"
        )
    body = plant.GetBodyByName(body_name)
    context = plant.CreateDefaultContext()
    n = t.shape[0]
    pos, rot, lin, ang = (
        np.zeros((n, 3)),
        np.zeros((n, 3, 3)),
        np.zeros((n, 3)),
        np.zeros((n, 3)),
    )
    for i in range(n):
        plant.SetPositions(context, qa[i])
        plant.SetVelocities(context, va[i])
        pose = plant.EvalBodyPoseInWorld(context, body)
        vel = plant.EvalBodySpatialVelocityInWorld(context, body)
        pos[i], rot[i] = pose.translation(), pose.rotation().matrix()
        lin[i], ang[i] = vel.translational(), vel.rotational()
    return rigid_body_series(t, pos, rot, lin, ang, spec)
