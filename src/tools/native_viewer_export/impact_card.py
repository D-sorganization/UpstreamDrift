"""Impact parameter card for native exports (GCV-19, #11725).

One card per swing from the same-input pose ``q`` that every engine's viewer
displays: the MuJoCo plant of the specification export places the club body, the
shared GCV-16 MuJoCo adapter turns it into a ``ClubheadSeries`` and the GCV-15
extractor and GCV-17 ``build_impact_card`` give the rows. The card is the model
the API route, the PyQt widget and the web panel already share.

Unavailable is never zero: any failure returns a reason string instead of a card.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.impact_parameters import (
    ClubheadSeries,
    TargetFrame,
    extract_impact_parameters,
)
from src.shared.python.impact_parameters.panel_model import (
    ImpactCard,
    build_impact_card,
)

if TYPE_CHECKING:
    from src.tools.native_viewer_export.core import SwingInput

CLUB_BODY_SUFFIX = "Clubface Vector"
#: Failures that become an "unavailable" reason rather than an exception
#: (``BackendUnavailable`` is a ``RuntimeError``).
_EXPECTED_ERRORS = (ValueError, KeyError, RuntimeError, OSError, TypeError)


def _backend_unavailable(message: str) -> Exception:
    from src.tools.native_viewer_export.core import BackendUnavailable

    return BackendUnavailable(message)


def _club_body_name(model: object) -> str:
    import mujoco

    for i in range(int(model.nbody)):  # type: ignore[attr-defined]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.endswith(CLUB_BODY_SUFFIX):
            return str(name)
    raise ValueError(f"model has no club body ('...{CLUB_BODY_SUFFIX}')")


def _club_series(swing: SwingInput) -> ClubheadSeries:
    """Club-head series of ``swing.q`` through the MuJoCo plant.

    Velocities are central differences of the displayed ``q`` (the pose is the
    only quantity every engine shares). Raises ``BackendUnavailable`` without
    MuJoCo and ``ValueError`` for a model without a club.
    """
    try:
        from src.engines.physics_engines.mujoco.python.overlay_source import (
            MujocoOverlaySource,
        )
    except ImportError as exc:
        raise _backend_unavailable(f"mujoco is not available: {exc}") from exc
    from src.shared.python.impact_parameters.adapters import (
        clubhead_series_from_mujoco,
    )

    model = MujocoOverlaySource(swing.bundle.spec_bytes).model
    body = _club_body_name(model)
    n = swing.n_states
    if n < 3:
        raise ValueError("a swing needs at least 3 states for an impact card")
    times = np.array([swing.time_s(i) for i in range(n)], dtype=float)
    qpos = np.zeros((n, model.nq))
    for col, name in enumerate(swing.bundle.coordinate_order):
        qpos[:, model.joint(name).qposadr[0]] = swing.q[:, col]
    qvel = np.zeros((n, model.nv))
    rates = np.gradient(swing.q, times, axis=0)
    for col, name in enumerate(swing.bundle.coordinate_order):
        qvel[:, model.joint(name).dofadr[0]] = rates[:, col]
    return clubhead_series_from_mujoco(model, times, qpos, qvel, body)


def impact_card_for(swing: SwingInput, impact_time_s: float) -> ImpactCard | str:
    """Impact card of ``swing`` at ``impact_time_s``, or the reason it is unavailable.

    The impact state is the last sample before ``impact_time_s``. Preconditions:
    ``impact_time_s`` is finite (``ValueError``). Postcondition: a returned
    ``ImpactCard`` carries computed values or per-row reasons, never zero fill;
    MuJoCo missing, no club, too-slow club head or an extractor failure give a
    non-empty reason string.
    """
    if not math.isfinite(impact_time_s):
        raise ValueError(f"impact_time_s must be finite, got {impact_time_s}")
    try:
        series = _club_series(swing)
        later = np.nonzero(series.times_s >= impact_time_s)[0]
        contact = int(later[0]) if later.size else len(series) - 1
        contact = max(contact, 1)
        params = extract_impact_parameters(series, TargetFrame(), contact_index=contact)
        return build_impact_card(params)
    except _EXPECTED_ERRORS as exc:
        return f"impact card unavailable: {exc}"
