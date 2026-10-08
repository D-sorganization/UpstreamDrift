"""Simscape logged-clubhead ClubheadSeries adapter (GCV-16).

Reads the logged clubhead position and velocity (see
``compute_clubhead_speed_mph.m`` / ``detect_clubhead_impact.m``).  When no
orientation is logged the face fields are unavailable with a reason, never
invented.
"""

from __future__ import annotations

from ..clubhead_series import ClubheadSeries
from .club_face import NATIVE_CLUB_FACE, ClubFaceSpec, rigid_body_series

NO_ORIENTATION_REASON = "Simscape log has no clubhead orientation; face unobservable"


def clubhead_series_from_simscape(
    times_s: object,
    position_m: object,
    velocity_mps: object,
    rotations: object | None = None,
    angular_velocity_rps: object | None = None,
    spec: ClubFaceSpec = NATIVE_CLUB_FACE,
) -> ClubheadSeries:
    """Position/velocity logs, optionally with orientation (face enabled)."""
    if rotations is not None:
        if angular_velocity_rps is None:
            raise ValueError("angular_velocity_rps is required with rotations")
        return rigid_body_series(
            times_s, position_m, rotations, velocity_mps, angular_velocity_rps, spec
        )
    return ClubheadSeries(
        times_s=times_s,
        face_center_m=position_m,
        velocity_mps=velocity_mps,
        face_unobservable_reason=NO_ORIENTATION_REASON,
    )
