"""Ball model for bunkershot3d (issues #8613, #8657).

Models the golf ball in a sand bed:
- Ball lie (position, depth, type)
- Sand-mediated momentum transfer (splash shot), driven by the impulse the F0
  solver delivered and the divot mass the metrics layer measured
- Launch conditions for flight handoff, with the validity verdict and
  provenance a carry number may only be quoted under
"""

from .lie import (
    BallLie,
    BallLieType,
    BallProperties,
    compute_exposed_cap_area,
    compute_exposed_cap_fraction,
    compute_submersion_depth,
)
from .splash import (
    BALL_LAUNCH_MEASUREMENT_GAP,
    BALL_LAUNCH_UNCALIBRATED_REASON,
    BALL_MOMENTUM_TRANSFER_EFFICIENCY,
    DEFAULT_MOMENTUM_TRANSFER,
    BallLaunchResult,
    ContactType,
    MomentumTransfer,
    SandDelivery,
    SplashTransferResult,
    compute_ball_launch_from_splash,
    compute_sand_ejecta_velocity,
    compute_splash_impulse,
    launch_verdict,
    momentum_transfer_provenance,
)
from .pipeline import (
    BunkerShotState,
    compute_bunker_launch,
    to_post_impact_state,
)

__all__ = [
    "BALL_LAUNCH_MEASUREMENT_GAP",
    "BALL_LAUNCH_UNCALIBRATED_REASON",
    "BALL_MOMENTUM_TRANSFER_EFFICIENCY",
    "DEFAULT_MOMENTUM_TRANSFER",
    "BallLaunchResult",
    "BallLie",
    "BallLieType",
    "BallProperties",
    "BunkerShotState",
    "ContactType",
    "MomentumTransfer",
    "SandDelivery",
    "SplashTransferResult",
    "compute_ball_launch_from_splash",
    "compute_bunker_launch",
    "compute_exposed_cap_area",
    "compute_exposed_cap_fraction",
    "compute_sand_ejecta_velocity",
    "compute_splash_impulse",
    "compute_submersion_depth",
    "launch_verdict",
    "momentum_transfer_provenance",
    "to_post_impact_state",
]
