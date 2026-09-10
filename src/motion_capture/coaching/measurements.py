"""Geometric readouts from the same registered motion used for display."""

import numpy as np
import numpy.typing as npt

from src.motion_capture.reference.model import ReferenceMotion
from src.motion_capture.reference.registration import (
    ReferenceRegistration,
    sample_reference_motion,
)

from .geometry import ReferenceGeometry, ReferencePlane, ReferencePoint


def reference_distances(
    motion: ReferenceMotion,
    registration: ReferenceRegistration,
    geometry: ReferenceGeometry,
    scene_times: npt.ArrayLike,
    joint_name: str,
    reference_id: str,
    *,
    scene_id: str,
) -> npt.NDArray[np.float64]:
    """Return metres to a point or signed metres to an infinite plane.

    Inputs refer to one explicit ADR-0041 world scene. Missing or out-of-range
    motion remains NaN; neither visibility nor plane extent changes distances.
    """
    if geometry.scene_id != scene_id:
        raise ValueError("Reference geometry belongs to another scene")
    times = np.asarray(scene_times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("Scene times must be a finite one-dimensional array")
    if joint_name not in motion.joint_names:
        raise ValueError("Unknown motion landmark")
    references: tuple[ReferencePlane | ReferencePoint, ...] = (
        *geometry.planes,
        *geometry.points,
    )
    reference = next(
        (item for item in references if item.id == reference_id),
        None,
    )
    if reference is None:
        raise ValueError("Unknown scene reference")
    points, valid = sample_reference_motion(motion, registration, times)
    index = motion.joint_names.index(joint_name)
    selected = points[:, index].copy()
    selected[~valid[:, index]] = np.nan
    if isinstance(reference, ReferencePlane):
        return reference.distances(selected)
    return np.asarray(
        np.linalg.norm(selected - reference.position_m, axis=1), dtype=np.float64
    )
