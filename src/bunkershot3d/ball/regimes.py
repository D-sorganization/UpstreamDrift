"""Which contact regime a strike fell into, and which of them may launch.

Issue #9544.  The splash partition in :mod:`.splash` is the only ball
launch this package has, and it is a model of exactly one thing: a head
that enters the sand *behind* the ball, passes under it, and throws sand
at it.  Three other outcomes are reachable from the same delivery
sweep, and none of them has a launch in common with a splash:

* **No hit** -- the head never engaged the bed, or cleared it behind a
  ball that is sunk below the surface.  Nothing moved the ball.
* **Direct strike** (thin, bladed) -- the leading edge reaches the ball's
  footprint before it reaches the sand, or the sole leaves the sand
  behind a ball still standing proud, so the face meets the ball with
  no sand between them.  That is an impact problem, not a splash, and
  the canonical impact contract lives in Tools; it is out of scope here.
* **Buried, no release** -- the head engaged and never came back out.
  There is no exit crossing, no divot, and no meaningful launch.

This module decides which of the four a strike was, from what the F0
march and the divot measurement already report, and
:func:`~.pipeline.compute_bunker_launch` refuses every regime but the
splash.  A refusal, not a fallback: routing a direct strike through the
splash partition would print a carry for a shot the model does not
describe.

What the classifier does **not** know
--------------------------------------

F0 is a flat half space with no crater and no free-surface evolution, so
the thickness of the sand cushion between face and ball is not a
quantity it has.  The discriminants below are the sole path's stations
relative to the ball's footprint -- entry behind it, exit past it -- and
they are declared, qualitative conventions.  A splash verdict says the
geometry *permits* a splash; it does not measure one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..exceptions import BunkerShot3DValueError
from .lie import BallLie, BallProperties
from .splash import ContactType

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..metrics.divot import DivotMetrics
    from ..solvers.shot import ShotResult

__all__ = [
    "RegimeClassification",
    "StrikeOutcome",
    "UnsupportedContactRegimeError",
    "classify_contact_regime",
]


class UnsupportedContactRegimeError(BunkerShot3DValueError):
    """The strike fell into a regime that has no launch model.

    A ``ValueError`` so the workbench reports it as a missing metric --
    "carry: ..." -- rather than a crash, and so no carry is printed.
    """


@dataclass(frozen=True, slots=True)
class StrikeOutcome:
    """The measured facts a regime is decided on.

    Attributes:
        engaged: Whether any sample of the march had an engaged element.
        exited: Whether the record ends with the sole above the surface.
        entry_distance_behind_ball_m: Where the sole entered the sand,
            positive behind the ball (the delivery-data convention, see
            :class:`~bunkershot3d.metrics.divot.DivotMetrics`). ``None``
            when no divot could be measured.
        exit_distance_past_ball_m: Where the sole left the sand, positive
            past the ball. ``None`` when no divot could be measured.
    """

    engaged: bool
    exited: bool
    entry_distance_behind_ball_m: float | None
    exit_distance_past_ball_m: float | None

    @classmethod
    def from_shot(cls, shot: ShotResult, divot: DivotMetrics | None) -> StrikeOutcome:
        """Read the outcome off a shot trace and its divot, if it has one.

        Args:
            shot: The F0 march.
            divot: The divot measured on it, or ``None`` when the metrics
                layer could not measure one.

        Returns:
            The outcome.
        """
        return cls(
            engaged=shot.contact_duration_s > 0.0,
            exited=shot.exited,
            entry_distance_behind_ball_m=(
                None if divot is None else float(divot.entry_distance_behind_ball_m)
            ),
            exit_distance_past_ball_m=(
                None if divot is None else float(divot.exit_distance_past_ball_m)
            ),
        )


@dataclass(frozen=True, slots=True)
class RegimeClassification:
    """Which regime the strike fell into, and why.

    Attributes:
        regime: The regime.
        reasons: What decided it, and what the verdict may not be read as.
    """

    regime: ContactType
    reasons: tuple[str, ...]

    @property
    def launch_derivable(self) -> bool:
        """Whether the splash partition applies to this strike."""
        return self.regime is ContactType.SPLASH


_F0_CUSHION_LIMITATION = (
    "the F0 half space has no crater and no free surface, so the sand cushion "
    "between face and ball is not measured; this regime is decided from the "
    "sole path's stations against the ball's footprint, which is a declared "
    "geometric convention (issue #9544)"
)


def classify_contact_regime(
    outcome: StrikeOutcome, *, lie: BallLie, ball: BallProperties
) -> RegimeClassification:
    """Decide which of the four regimes a strike fell into.

    The rules, in the order they are applied:

    1. Never engaged -> ``NO_HIT``.
    2. Engaged but never exited -> ``BURIED_NO_RELEASE``.
    3. Sole entered at or inside the ball's footprint (entry distance
       behind the ball no more than one radius) -> ``THIN``: the leading
       edge meets the ball before the sand does.
    4. Sole left the sand at or behind the footprint's trailing edge:
       ``THIN`` if the ball still stands proud of the surface, since the
       face then meets it out of the sand; ``NO_HIT`` if the ball is sunk
       a full diameter, since the head clears it.
    5. Otherwise ``SPLASH``.

    Args:
        outcome: The measured facts.
        lie: The ball's burial, ``depth_m`` being how far its underside
            sits below the surface.
        ball: The ball's dimensions.

    Returns:
        The regime and the reasons that decided it.

    Raises:
        ValueError: If the strike engaged and exited but no divot stations
            were supplied, since the regime cannot then be decided.
    """
    if not isinstance(outcome, StrikeOutcome):
        raise ValueError(
            f"outcome must be a StrikeOutcome, got {type(outcome).__name__}"
        )
    if not outcome.engaged:
        return RegimeClassification(
            ContactType.NO_HIT,
            ("the head never engaged the bed, so nothing reached the ball",),
        )
    if not outcome.exited:
        return RegimeClassification(
            ContactType.BURIED_NO_RELEASE,
            (
                "the head engaged the bed and never brought its sole back above "
                "the surface, so there is no exit crossing, no divot and no "
                "release; a longer window does not change that for a head "
                "that has stopped",
            ),
        )
    entry = outcome.entry_distance_behind_ball_m
    exit_ = outcome.exit_distance_past_ball_m
    if entry is None or exit_ is None:
        raise ValueError(
            "the strike engaged and exited but carries no divot stations, so "
            "its contact regime cannot be decided; measure the divot first"
        )
    radius = ball.radius_m
    if entry <= radius:
        return RegimeClassification(
            ContactType.THIN,
            (
                f"the sole entered the sand {entry * 1e3:.3g} mm behind the "
                f"ball, inside its {radius * 1e3:.3g} mm footprint, so the "
                "leading edge meets the ball before it meets the sand; this is "
                "a direct (thin) strike and not a splash, and no launch is "
                "derived for it",
                _F0_CUSHION_LIMITATION,
            ),
        )
    if exit_ <= -radius:
        proud_m = ball.diameter_m - lie.depth_m
        if proud_m > 0.0:
            return RegimeClassification(
                ContactType.THIN,
                (
                    f"the sole left the sand {-exit_ * 1e3:.3g} mm short of the "
                    f"ball, which stands {proud_m * 1e3:.3g} mm proud of the "
                    "surface, so the face meets the ball out of the sand; this "
                    "is a direct strike and not a splash",
                    _F0_CUSHION_LIMITATION,
                ),
            )
        return RegimeClassification(
            ContactType.NO_HIT,
            (
                f"the sole left the sand {-exit_ * 1e3:.3g} mm short of a ball "
                "sunk to the surface, so the head cleared it and nothing "
                "reached the ball",
                _F0_CUSHION_LIMITATION,
            ),
        )
    return RegimeClassification(
        ContactType.SPLASH,
        (
            f"the sole entered {entry * 1e3:.3g} mm behind the ball and left "
            f"{exit_ * 1e3:.3g} mm past it, so the geometry permits a splash",
            _F0_CUSHION_LIMITATION,
        ),
    )
