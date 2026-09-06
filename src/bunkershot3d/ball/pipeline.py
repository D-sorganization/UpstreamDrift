"""Pipeline integration for bunkershot3d ball model (issues #8613, #8657).

Provides the handoff between bunkershot3d and the existing
``SwingBallFlightPipeline`` infrastructure: a bunker shot in, a
:class:`~src.shared.python.physics.impact_model.PostImpactState` out, ready
for flight simulation.

Everything measured arrives through
:class:`~bunkershot3d.ball.splash.SandDelivery` -- the solver's impulse and
entry/exit speeds, and the metrics layer's divot mass. What remains on
:class:`BunkerShotState` is the club's declared geometry and mass and the lie,
none of which the solver measures. The old ``entry_depth_m``,
``sole_width_m`` and ``sole_length_m`` fields existed only to feed the deleted
box-volume estimate of displaced sand (issue #8657) and are gone with it.

Usage::

    state = BunkerShotState(club_loft_deg=56.0, ball_lie=..., delivery=...)
    result = compute_bunker_launch(state)
    post = to_post_impact_state(result, state)
    # Then use SwingBallFlightPipeline with post_state
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.shared.python.core.contracts import require
from src.shared.python.physics.impact_model import PostImpactState

from .lie import BallLie, BallProperties
from .splash import (
    DEFAULT_MOMENTUM_TRANSFER,
    BallLaunchResult,
    MomentumTransfer,
    SandDelivery,
    compute_ball_launch_from_splash,
)

__all__ = [
    "BunkerShotState",
    "compute_bunker_launch",
    "to_post_impact_state",
]


@dataclass(slots=True)
class BunkerShotState:
    """Complete specification of a bunker shot.

    Attributes:
        club_loft_deg: Effective loft at delivery [degrees].
        ball_lie: Ball position and burial in sand.
        delivery: What the solver and metrics layer measured about the strike.
        club_mass_kg: Club head mass [kg].
        ball: Ball properties.
        transfer: The uncalibrated sand-to-ball partition parameters.
    """

    club_loft_deg: float
    ball_lie: BallLie
    delivery: SandDelivery
    club_mass_kg: float = 0.30  # 300g wedge head
    ball: BallProperties = field(default_factory=BallProperties)
    transfer: MomentumTransfer = DEFAULT_MOMENTUM_TRANSFER

    def __post_init__(self) -> None:
        """Validate the declared club properties.

        Raises:
            ValueError: If the delivery is not a measured strike. The launch
                may not fall back on a defaulted one.
        """
        require(0 < self.club_loft_deg < 90, "loft must be in (0, 90) degrees")
        require(self.club_mass_kg > 0, "club mass must be positive")
        if not isinstance(self.delivery, SandDelivery):
            raise ValueError(
                "a bunker shot must carry the strike the solver measured; got "
                f"{type(self.delivery).__name__}. Ball launch is derived from "
                "the delivered impulse and the divot mass (issue #8657), and "
                "neither has a sensible default"
            )


def compute_bunker_launch(state: BunkerShotState) -> BallLaunchResult:
    """Compute ball launch conditions from bunker shot.

    This is the main entry point for bunker shot physics. Thin/blade direct
    contact remains out of scope, so the splash transfer is always used.

    Args:
        state: Complete bunker shot specification.

    Returns:
        The launch, its validity verdict and the provenance of the partition.
    """
    return compute_ball_launch_from_splash(
        lie=state.ball_lie,
        ball=state.ball,
        delivery=state.delivery,
        club_loft_rad=math.radians(state.club_loft_deg),
        club_mass_kg=state.club_mass_kg,
        transfer=state.transfer,
    )


def to_post_impact_state(
    result: BallLaunchResult,
    state: BunkerShotState,
) -> PostImpactState:
    """Convert a launch result to the flight pipeline's input contract.

    The clubhead's state after the sand is the solver's, not an estimate: it
    left the bed at :attr:`~bunkershot3d.ball.splash.SandDelivery.exit_speed_m_s`
    after :attr:`~bunkershot3d.ball.splash.SandDelivery.contact_duration_s` of
    engagement. Only the *direction* it leaves in is a convention, taken from
    the loft as the launch direction is.

    Args:
        result: Ball launch result from the bunker shot.
        state: Original bunker shot state.

    Returns:
        PostImpactState ready for flight simulation.
    """
    delivery = state.delivery
    club_loft_rad = math.radians(state.club_loft_deg)
    ball_angular_velocity = np.array(result.ball_angular_velocity, dtype=float)

    if delivery.exit_velocity_m_s is not None:
        club_velocity = np.array(delivery.exit_velocity_m_s, dtype=float)
    else:
        club_speed_out = delivery.exit_speed_m_s
        club_velocity = np.array(
            [
                club_speed_out * math.cos(club_loft_rad * 0.3),  # Forward
                0.0,  # No lateral
                -club_speed_out * math.sin(club_loft_rad * 0.3),  # Down through sand
            ],
            dtype=float,
        )

    if delivery.exit_angular_velocity_rad_s is not None:
        clubhead_angular_velocity = np.array(
            delivery.exit_angular_velocity_rad_s, dtype=float
        )
    else:
        clubhead_angular_velocity = np.zeros(3, dtype=float)

    ball_ke = 0.5 * state.ball.mass_kg * result.ball_speed_m_s**2
    ball_rot_ke = (
        0.5 * state.ball.moi_kg_m2 * float(np.linalg.norm(ball_angular_velocity)) ** 2
    )

    return PostImpactState(
        ball_velocity=np.array(result.ball_velocity, dtype=float),
        ball_angular_velocity=ball_angular_velocity,
        clubhead_velocity=club_velocity,
        clubhead_angular_velocity=clubhead_angular_velocity,
        contact_duration=delivery.contact_duration_s,
        energy_transfer=float(ball_ke + ball_rot_ke),
        impact_location=np.zeros(2, dtype=float),
    )
