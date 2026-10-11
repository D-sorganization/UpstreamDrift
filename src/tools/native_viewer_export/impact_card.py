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
Right-handed, Z-up. Unavailable is never zero: any failure returns a reason.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.impact_parameters import (
    BallObservation,
    ClubheadSeries,
    TargetFrame,
    extract_impact_parameters,
)
from src.shared.python.impact_parameters.panel_model import (
    ImpactCard,
    build_impact_card,
)
from src.tools.native_viewer_export.club_pose import ClubPoseModel

if TYPE_CHECKING:
    from src.tools.native_viewer_export.ball import BallPlacement
    from src.tools.native_viewer_export.core import SwingInput

TARGET_SOURCE = "address_face_normal"
#: Failures that become an "unavailable" reason rather than an exception
#: (``BackendUnavailable`` is a ``RuntimeError``).
_EXPECTED_ERRORS = (ValueError, KeyError, RuntimeError, OSError, TypeError)


def _backend_unavailable(message: str) -> Exception:
    from src.tools.native_viewer_export.core import BackendUnavailable

    return BackendUnavailable(message)


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
    return ClubheadSeries(
        times_s=times,
        face_center_m=centre,
        velocity_mps=np.gradient(centre, times, axis=0),
        face_normal=np.array([f.normal for f in faces]),
        toe_axis=np.array([f.toe for f in faces]),
        grip_axis=np.array([f.grip for f in faces]),
    )


def address_target_frame(series: ClubheadSeries) -> TargetFrame:
    """Right-handed Z-up target frame along the horizontal address face normal."""
    normal = np.asarray(series.face_normal)[0]
    horizontal = np.array([normal[0], normal[1], 0.0])
    length = float(np.linalg.norm(horizontal))
    if length < 1e-6:
        raise ValueError("address face normal has no horizontal part: no target line")
    x, y, _ = horizontal / length
    return TargetFrame(target_dir=(float(x), float(y), 0.0), handedness="right")


def ball_contact_point(
    series: ClubheadSeries, index: int, ball: BallPlacement
) -> np.ndarray:
    """Point of the ball surface facing the face at sample ``index``.

    ``ball_centre - radius * n_face``: the face normal points from the face to
    the ball, so the ball surface point facing the club lies one radius back
    along the normal. Raises ``ValueError`` without a face normal series.
    """
    if series.face_normal is None:
        raise ValueError("face normal unobservable: no ball contact point")
    normal = np.asarray(series.face_normal[index], dtype=float)
    length = float(np.linalg.norm(normal))
    if length < 1e-9:
        raise ValueError("face normal has zero length: no ball contact point")
    centre = np.asarray(ball.position_m, dtype=float)
    return centre - ball.radius_m * normal / length


def card_from_series(
    series: ClubheadSeries,
    impact_time_s: float,
    ball: BallPlacement | None = None,
) -> ImpactCard:
    """Card of ``series`` at ``impact_time_s`` (last sample before it).

    With a ``ball`` the contact point is its surface point facing the face at
    the impact sample and the toe/high rows are filled (offsets of that point
    from the face centre along the face toe and high axes); ``extras["ball_source"]``
    records where the placement came from. Without one those rows stay
    unavailable.

    Raises ``ValueError`` for a non-finite time, an address normal without a
    horizontal part, or a club head slower than the extractor minimum.
    """
    if not math.isfinite(impact_time_s):
        raise ValueError(f"impact_time_s must be finite, got {impact_time_s}")
    later = np.nonzero(series.times_s >= impact_time_s)[0]
    contact = max(int(later[0]) if later.size else len(series) - 1, 1)
    observation = None
    if ball is not None:
        # the extractor's impact state is the last pre-contact sample
        observation = BallObservation(
            contact_m=ball_contact_point(series, contact - 1, ball)
        )
    params = extract_impact_parameters(
        series, address_target_frame(series), contact_index=contact, ball=observation
    )
    card = build_impact_card(params)
    card.extras["target_source"] = TARGET_SOURCE
    if ball is not None:
        card.extras["ball_source"] = ball.source
    return card


def impact_card_for(
    swing: SwingInput,
    impact_time_s: float,
    ball: BallPlacement | None = None,
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
        return card_from_series(_club_series(swing), impact_time_s, ball)
    except _EXPECTED_ERRORS as exc:
        return f"impact card unavailable: {exc}"
