"""3D-to-2D target projection utility.

Projects a 3D MultiSourceTarget or ClubTarget onto a 2D swing plane.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.provider import MultiSourceTarget


def project_to_2d(target: MultiSourceTarget | ClubTarget) -> ClubTarget:
    """Project a 3D target onto the 2D swing plane.

    Returns a new ClubTarget with the z-coordinates zeroed out or mapped appropriately.
    """
    club = target.club if isinstance(target, MultiSourceTarget) else target
    if club is None:
        raise ValueError("target.club must be set")

    # Example projection: just take x and y, set z to 0.
    # In a real scenario, this would fit a plane to the swing and project onto it.

    def proj(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        if out.shape[-1] == 3:
            out[..., 2] = 0.0
        return out

    import dataclasses

    projected = dataclasses.replace(
        club,
        butt=proj(club.butt),
        clubhead=proj(club.clubhead),
    )
    return projected
