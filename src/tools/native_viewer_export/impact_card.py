"""Impact parameter card for native exports (GCV-19, #11725).

One card per swing from the same-input pose ``q`` that every engine's viewer
displays. The MuJoCo plant of the specification export places the spec club
body, :mod:`club_pose` gives the true face centre and axes (the same face the
decorative ball uses), central differences of the face-centre positions give
the velocity, and the GCV-15 extractor plus GCV-17 ``build_impact_card`` give
the rows. The card is the model the API route, the PyQt widget and the web
panel already share.

The specification has no target line, so the target direction is the
horizontal part of the address (state 0) face normal: the face is square at
address (the rule of ``motion_matching.pipeline.gaze_residual.plan_gaze``).
That is the normal of the address-square head, whatever head roll is drawn.
Right-handed, Z-up. Unavailable is never zero: any failure returns a reason.
With a ball the card reads the club where its face reaches the ball
(:mod:`impact_state`, #15150), not at the sample before the impact time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.impact_parameters import (
    BallObservation,
    ClubheadSeries,
    ImpactParameters,
    TargetFrame,
    extract_impact_parameters,
)
from src.shared.python.impact_parameters.panel_model import (
    MPS_TO_MPH,
    CardRow,
    ImpactCard,
    build_impact_card,
)
from src.tools.native_viewer_export.club_pose import ClubPoseModel
from src.tools.native_viewer_export.display_units import load_display_units
from src.tools.native_viewer_export.impact_state import series_at_contact

if TYPE_CHECKING:
    from src.tools.native_viewer_export.ball import BallPlacement
    from src.tools.native_viewer_export.core import SwingInput

TARGET_SOURCE = "address_face_normal"
#: Largest contact offset from the face centre still counted as a strike. It is
#: wider than half the face height of every club, so a contact further out
#: lies off the face (the model face passed the ball without striking it).
MAX_FACE_OFFSET_M = 0.050
#: The contact point is carried along the head velocity only when the face
#: closes on the ball at more than this fraction of the head speed (a numerical
#: guard against a face moving edge-on; no physical quantity).
MIN_CLOSING_FRACTION = 0.1
#: Failures that become an "unavailable" reason rather than an exception
#: (``BackendUnavailable`` is a ``RuntimeError``).
_EXPECTED_ERRORS = (ValueError, KeyError, RuntimeError, OSError, TypeError)
#: start of the reason given for a contact point that lies off the face
OFF_FACE = "ball contact off the face"


def _backend_unavailable(message: str) -> Exception:
    from src.tools.native_viewer_export.core import BackendUnavailable

    return BackendUnavailable(message)


#: ``extras["impact_state"]``: where on the swing the card rows were read.
STATE_AT_CONTACT = "pre_contact_fit_at_ball_contact"
STATE_LAST_SAMPLE = "last_sample_before_impact_time"


@dataclass(frozen=True)
class ModelClubSeries(ClubheadSeries):
    """Face series of a model; ``target_normal`` is the world normal of its
    address-square head at the first state, which the target line follows."""

    target_normal: tuple[float, float, float] | None = None


def _club_series(swing: SwingInput) -> ClubheadSeries:
    """Face series of the displayed states ``swing.q`` (unresampled source states).

    Velocity is ``np.gradient`` of the world face-centre positions over
    ``swing.time_s``. Raises ``BackendUnavailable`` without MuJoCo and
    ``ValueError`` for a model without a club or fewer than 3 states.
    """
    n = swing.n_states
    if n < 3:
        raise ValueError("a swing needs at least 3 states for an impact card")
    try:
        model = ClubPoseModel(swing.bundle.spec_bytes)
    except ImportError as exc:
        raise _backend_unavailable(f"mujoco is not available: {exc}") from exc
    names = swing.bundle.coordinate_order
    faces = [
        model.face_world(dict(zip(names, map(float, row), strict=True)))
        for row in swing.q
    ]
    times = np.array([swing.time_s(i) for i in range(n)], dtype=float)
    centre = np.array([f.centre_m for f in faces])
    address = dict(zip(names, map(float, swing.q[0]), strict=True))
    square_x, square_y, square_z = model.address_square_normal_world(address).tolist()
    return ModelClubSeries(
        target_normal=(square_x, square_y, square_z),
        times_s=times,
        face_center_m=centre,
        velocity_mps=np.gradient(centre, times, axis=0),
        face_normal=np.array([f.normal for f in faces]),
        toe_axis=np.array([f.toe for f in faces]),
        grip_axis=np.array([f.grip for f in faces]),
    )


def address_target_frame(series: ClubheadSeries) -> TargetFrame:
    """Right-handed Z-up target frame along the horizontal address face normal
    (of the address-square head for a :class:`ModelClubSeries`, #15150)."""
    stated = getattr(series, "target_normal", None)
    normal = np.asarray(series.face_normal)[0] if stated is None else np.array(stated)
    horizontal = np.array([normal[0], normal[1], 0.0])
    length = float(np.linalg.norm(horizontal))
    if length < 1e-6:
        raise ValueError("address face normal has no horizontal part: no target line")
    x, y, _ = horizontal / length
    return TargetFrame(target_dir=(float(x), float(y), 0.0), handedness="right")


def address_target_direction(swing: SwingInput) -> np.ndarray:
    """Unit horizontal target-line direction of ``swing`` (z component 0).

    Raises ``ValueError`` when the address face normal has no horizontal part.
    """
    x, y, _ = address_target_frame(_club_series(swing)).target_dir
    return np.array([x, y, 0.0]) / math.hypot(x, y)


def ball_contact_point(
    series: ClubheadSeries, index: int, ball: BallPlacement
) -> np.ndarray:
    """Point of the ball surface facing the face at sample ``index``.

    ``ball_centre - radius * n_face``: the face normal points from the face to
    the ball, so the ball surface point facing the club lies one radius back
    along the normal, carried back along the head velocity onto the face plane
    of the sample (#15150): the sample lies up to one step of head travel short
    of the ball, and an open face would read that travel as a toe or heel
    miss. Raises ``ValueError`` without a face normal series.
    """
    if series.face_normal is None:
        raise ValueError("face normal unobservable: no ball contact point")
    normal = np.asarray(series.face_normal[index], dtype=float)
    length = float(np.linalg.norm(normal))
    if length < 1e-9:
        raise ValueError("face normal has zero length: no ball contact point")
    unit = normal / length
    point = np.asarray(ball.position_m, dtype=float) - ball.radius_m * unit
    velocity = np.asarray(series.velocity_mps[index], dtype=float)
    closing = float(velocity @ unit)
    if closing > MIN_CLOSING_FRACTION * float(np.linalg.norm(velocity)):
        gap = float((point - np.asarray(series.face_center_m[index])) @ unit)
        point = point - velocity * (gap / closing)
    return point


def _on_face_only(params: ImpactParameters) -> ImpactParameters:
    """``params`` with an off-face contact replaced by an "unavailable" reason.

    A contact point further than :data:`MAX_FACE_OFFSET_M` from the face centre
    is not on any club face: the model face passed the ball without striking
    it, so no strike location is claimed. The measured miss is kept in the
    reason. Postcondition: a returned toe/high pair lies within the bound.
    """
    if params.toe_mm is None or params.high_mm is None:
        return params
    if math.hypot(params.toe_mm, params.high_mm) <= MAX_FACE_OFFSET_M * 1000.0:
        return params
    reason = (
        f"{OFF_FACE} (toe {params.toe_mm:+.0f} mm, high "
        f"{params.high_mm:+.0f} mm from the face centre): the model face misses "
        "the ball"
    )
    unavailable = {**params.unavailable, "toe_mm": reason, "high_mm": reason}
    return replace(params, toe_mm=None, high_mm=None, unavailable=unavailable)


def contact_index(series: ClubheadSeries, impact_time_s: float) -> int:
    """First sample at or after ``impact_time_s`` (at least 1, at most the last).

    The clubhead state of the impact is the sample before it.
    """
    later = np.nonzero(series.times_s >= impact_time_s)[0]
    return max(int(later[0]) if later.size else len(series) - 1, 1)


#: note of a smash factor computed from the model ball speed
SMASH_NOTE = "model"


def smash_factor(ball_speed_mph: float, clubhead_speed_mph: float) -> float:
    """Ball speed over clubhead speed (same unit, dimensionless).

    Raises ``ValueError`` unless both speeds are finite and positive.
    """
    for name, value in (("ball", ball_speed_mph), ("clubhead", clubhead_speed_mph)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} speed must be finite and positive, got {value}")
    return ball_speed_mph / clubhead_speed_mph


def with_model_smash(card: ImpactCard, ball_speed_mph: float | None) -> ImpactCard:
    """``card`` with the smash factor of the flight model's ball speed.

    ``ball_speed_mph`` is the ball speed of the impact solve that drives the
    tracer (``FlightSummary.ball_speed_mph``). ``None`` returns ``card``
    unchanged (the row keeps its "unavailable" reason, never a zero), as does a
    card whose clubhead speed is not available. The row carries ``SMASH_NOTE``.
    """
    if ball_speed_mph is None:
        return card
    speed = next((r.value for r in card.rows if r.key == "clubhead_speed"), None)
    if speed is None:
        return card
    scale = 1.0 if card.units == "mph" else MPS_TO_MPH  # card speed -> mph
    value = smash_factor(ball_speed_mph, float(speed) * scale)
    rows = tuple(
        CardRow(r.key, r.label, r.unit, value, note=SMASH_NOTE)
        if r.key == "smash_factor"
        else r
        for r in card.rows
    )
    return replace(card, rows=rows)


def card_from_series(
    series: ClubheadSeries,
    impact_time_s: float,
    ball: BallPlacement | None = None,
    speed_unit: str | None = None,
) -> ImpactCard:
    """Card of ``series`` at ``impact_time_s`` (last sample before it).

    ``speed_unit`` (mph or m/s; default: the display units settings) is the
    unit of the speed rows; ``ValueError`` for another.

    With a ``ball`` the rows are read where the face reaches it
    (:func:`impact_state.series_at_contact`), never across the strike; the
    toe/high rows are the offsets of the ball surface point facing the face
    there from the face centre. ``extras["ball_source"]`` records where the
    placement came from and ``extras["impact_state"]`` which state was read.
    Without a ball those rows stay unavailable.

    Raises ``ValueError`` for a non-finite time, an address normal without a
    horizontal part, or a club head slower than the extractor minimum.
    """
    if not math.isfinite(impact_time_s):
        raise ValueError(f"impact_time_s must be finite, got {impact_time_s}")
    contact = contact_index(series, impact_time_s)
    observation = None
    if ball is not None:
        # the extractor's impact state is the last pre-contact sample
        observation = BallObservation(
            contact_m=ball_contact_point(series, contact - 1, ball)
        )
    frame = address_target_frame(series)
    moved = None
    if ball is not None:
        moved = series_at_contact(series, ball.position_m, ball.radius_m, impact_time_s)
    if moved is None:
        params = extract_impact_parameters(
            series, frame, contact_index=contact, ball=observation
        )
    else:
        params = _parameters_at_contact(*moved, frame, ball)
    card = build_impact_card(
        _on_face_only(params), units=speed_unit or load_display_units().speed_unit
    )
    card.extras["impact_state"] = STATE_AT_CONTACT if moved else STATE_LAST_SAMPLE
    card.extras["target_source"] = TARGET_SOURCE
    if ball is not None:
        card.extras["ball_source"] = ball.source
    return card


def _parameters_at_contact(
    series: ClubheadSeries, index: int, frame: TargetFrame, ball: BallPlacement | None
) -> ImpactParameters:
    """Parameters of contact sample ``index``; the ball touches its face plane."""
    if ball is None:
        raise ValueError("a contact state needs the ball it was found with")
    facing = np.asarray(series.face_normal)[index]
    touch = BallObservation(np.asarray(ball.position_m) - ball.radius_m * facing)
    return extract_impact_parameters(series, frame, impact_index=index, ball=touch)


def impact_card_for(
    swing: SwingInput,
    impact_time_s: float,
    ball: BallPlacement | None = None,
    speed_unit: str | None = None,
) -> ImpactCard | str:
    """Impact card of ``swing`` at ``impact_time_s``, or the reason it is unavailable.

    ``ball`` is the decorative ball placement (``swing.ball`` of the export);
    without it impact location is unavailable. Preconditions: ``impact_time_s``
    is finite (``ValueError``). Postcondition:
    a returned ``ImpactCard`` carries computed values or per-row reasons, never
    zero fill; MuJoCo missing, no club, too-slow club head or an extractor
    failure give a non-empty reason string.
    """
    if not math.isfinite(impact_time_s):
        raise ValueError(f"impact_time_s must be finite, got {impact_time_s}")
    try:
        return card_from_series(_club_series(swing), impact_time_s, ball, speed_unit)
    except _EXPECTED_ERRORS as exc:
        return f"impact card unavailable: {exc}"
