"""3D-to-2D target projection utility using calibrated rigid swing planes (TB-03 #10588).

Projects a 3D MultiSourceTarget or ClubTarget onto a rigid 2D swing plane without
distorting in-plane segment geometry or dropping coordinates naively.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np

from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.provider import MultiSourceTarget
from src.shared.python.tour_baselines.plane import (
    PlaneFitDiagnostics,
    RigidSwingPlane,
    fit_rigid_swing_plane,
)

logger = logging.getLogger(__name__)


def project_target_to_calibrated_plane(
    target: MultiSourceTarget | ClubTarget,
    plane: RigidSwingPlane,
) -> tuple[ClubTarget, PlaneFitDiagnostics]:
    """Project a 3D target onto an existing calibrated rigid swing plane."""
    club = target.club if isinstance(target, MultiSourceTarget) else target
    if club is None:
        raise ValueError("target.club must be set")

    butt_arr = np.asarray(club.butt, dtype=float)
    head_arr = np.asarray(club.clubhead, dtype=float)

    # Project butt and clubhead
    butt_2d, butt_res = plane.project_points(butt_arr)
    head_2d, head_res = plane.project_points(head_arr)

    # Unproject onto plane (with w=0) to maintain 3D world representation
    butt_proj_3d = plane.unproject_points(butt_2d, w=0.0)
    head_proj_3d = plane.unproject_points(head_2d, w=0.0)

    # Compute overall diagnostics
    all_res = np.concatenate([butt_res, head_res])
    finite_res = all_res[np.isfinite(all_res)]
    rmse = float(np.sqrt(np.mean(finite_res**2))) if len(finite_res) > 0 else 0.0
    max_res = float(np.max(np.abs(finite_res))) if len(finite_res) > 0 else 0.0

    diag = PlaneFitDiagnostics(
        rmse_m=rmse,
        max_residual_m=max_res,
        sample_count=len(finite_res),
        singular_values=(0.0, 0.0, 0.0),
        per_marker_rmse_m={
            "butt": float(np.sqrt(np.mean(butt_res[np.isfinite(butt_res)] ** 2))),
            "clubhead": float(np.sqrt(np.mean(head_res[np.isfinite(head_res)] ** 2))),
        },
    )

    projected_club = dataclasses.replace(
        club,
        butt=butt_proj_3d,
        clubhead=head_proj_3d,
    )
    return projected_club, diag


def project_to_2d(
    target: MultiSourceTarget | ClubTarget,
    plane: RigidSwingPlane | None = None,
) -> ClubTarget:
    """Project a 3D target onto the rigid 2D swing plane.

    If no plane is provided, a single rigid swing plane is calibrated from all
    valid butt and clubhead observations.
    """
    club = target.club if isinstance(target, MultiSourceTarget) else target
    if club is None:
        raise ValueError("target.club must be set")

    if plane is None:
        # Fit rigid plane from combined butt and clubhead points
        butt_arr = np.asarray(club.butt, dtype=float)
        head_arr = np.asarray(club.clubhead, dtype=float)
        combined_pts = np.vstack([butt_arr, head_arr])
        fitted_plane, _ = fit_rigid_swing_plane(combined_pts)
        plane = fitted_plane

    projected_club, _ = project_target_to_calibrated_plane(target, plane)
    return projected_club
