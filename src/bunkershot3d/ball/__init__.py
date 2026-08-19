"""Ball model for bunkershot3d (issue #8613).

Models the golf ball in a sand bed:
- Ball lie (position, depth, type)
- Sand-mediated momentum transfer (splash shot)
- Direct club-ball contact (thin/blade shot)
- Launch conditions for flight handoff
"""

from .lie import (
    BallLie,
    BallLieType,
    BallProperties,
    compute_exposed_cap_area,
    compute_submersion_depth,
)
from .splash import (
    BallLaunchResult,
    ContactType,
    SplashTransferResult,
    compute_ball_launch_from_splash,
    compute_sand_ejecta_velocity,
    compute_splash_impulse,
)
from .pipeline import (
    BunkerShotState,
    compute_bunker_launch,
    to_post_impact_state,
)

__all__ = [
    "BallLaunchResult",
    "BallLie",
    "BallLieType",
    "BallProperties",
    "BunkerShotState",
    "ContactType",
    "SplashTransferResult",
    "compute_ball_launch_from_splash",
    "compute_bunker_launch",
    "compute_exposed_cap_area",
    "compute_sand_ejecta_velocity",
    "compute_splash_impulse",
    "compute_submersion_depth",
    "to_post_impact_state",
]
