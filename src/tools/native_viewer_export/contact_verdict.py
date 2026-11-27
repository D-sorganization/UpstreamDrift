"""The one "did the face strike the ball" decision of an export (#15150, C22).

The Impact Factors panel and the ball flight both read it, so the panel never
says "no contact" next to a drawn flight. The rule is the impact card's own:
a contact point further than ``impact_card.MAX_FACE_OFFSET_M`` from the face
centre is off the face, so the model face misses the ball.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.shared.python.impact_parameters import ClubheadSeries
from src.tools.native_viewer_export import impact_card

if TYPE_CHECKING:
    from src.tools.native_viewer_export.ball import BallPlacement


@dataclass(frozen=True)
class ContactVerdict:
    """Whether the face struck the ball (``False``: it passed without contact)."""

    contact: bool


def verdict_from_card(card: object, ball: BallPlacement | None) -> ContactVerdict:
    """The verdict of an impact ``card`` computed with ``ball``.

    Without a ball nothing can be missed, so ``contact`` is ``True``, as it is
    for a card without a location row (nothing says the face missed).
    """
    if ball is None:
        return ContactVerdict(True)
    rows = getattr(card, "rows", ())
    row = next((r for r in rows if r.key == "impact_location"), None)
    reason = "" if row is None else str(row.reason)
    return ContactVerdict(not reason.startswith(impact_card.OFF_FACE))


def verdict_for(
    series: ClubheadSeries, impact_time_s: float, ball: BallPlacement | None
) -> ContactVerdict:
    """The verdict of ``series`` at ``impact_time_s``.

    Raises what :func:`impact_card.card_from_series` raises for a series it
    cannot use. Prefer the verdict the panel already holds over this call.
    """
    if ball is None:
        return ContactVerdict(True)
    card = impact_card.card_from_series(series, impact_time_s, ball)
    return verdict_from_card(card, ball)
