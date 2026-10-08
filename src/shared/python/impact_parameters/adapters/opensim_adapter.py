"""OpenSim ClubheadSeries adapter (GCV-16).

Requires a model with a club body (OSV epic).  Coordinates are set by name from
a ``{coordinate: values}`` mapping; speeds come from the matching ``speeds``
mapping so the body kinematics reflect the same rollout as the other engines.
"""

from __future__ import annotations

from collections.abc import Mapping

from typing import Any

import numpy as np

from ..clubhead_series import ClubheadSeries
from .club_face import (
    NATIVE_CLUB_FACE,
    ClubFaceSpec,
    empty_pose_twist,
    rigid_body_series,
)


def clubhead_series_from_opensim(
    model: Any,
    times_s: object,
    coordinate_values: Mapping[str, object],
    coordinate_speeds: Mapping[str, object],
    body_name: str,
    spec: ClubFaceSpec = NATIVE_CLUB_FACE,
) -> ClubheadSeries:
    """Body kinematics in ground from an initialised OpenSim model."""
    t = np.asarray(times_s, dtype=float)
    if t.ndim != 1 or t.shape[0] < 2:
        raise ValueError("times_s must be 1-D with at least 2 samples")
    if set(coordinate_values) != set(coordinate_speeds):
        raise ValueError("coordinate_values and coordinate_speeds need the same keys")
    series = {k: np.asarray(x, dtype=float) for k, x in coordinate_values.items()}
    rates = {k: np.asarray(x, dtype=float) for k, x in coordinate_speeds.items()}
    for key in series:
        if series[key].shape != t.shape or rates[key].shape != t.shape:
            raise ValueError(f"coordinate {key!r} must have one value per time sample")
    if not model.getBodySet().hasComponent(body_name):
        raise ValueError(f"OpenSim model has no body named {body_name!r}")
    body = model.getBodySet().get(body_name)
    coords = model.getCoordinateSet()
    state = model.initSystem()
    n = t.shape[0]
    pos, rot, lin, ang = empty_pose_twist(n)
    for i in range(n):
        for key in series:
            coord = coords.get(key)
            coord.setValue(state, float(series[key][i]), False)
            coord.setSpeedValue(state, float(rates[key][i]))
        model.realizeVelocity(state)
        tf = body.getTransformInGround(state)
        vel = body.getVelocityInGround(state)
        pos[i] = [tf.p().get(k) for k in range(3)]
        rmat = tf.R()
        rot[i] = [[rmat.get(r, c) for c in range(3)] for r in range(3)]
        ang[i] = [vel.get(0).get(k) for k in range(3)]
        lin[i] = [vel.get(1).get(k) for k in range(3)]
    return rigid_body_series(t, pos, rot, lin, ang, spec)
