"""The rendered frame on the exact instant the face reaches the ball (#15150).

Contact falls between capture samples. The exporter renders a frame at that
instant (:func:`impact_state.series_at_contact`), counts the HUD's milliseconds
from it, keeps the ball on its tee through it and holds it for a settable number
of video frames (``contact_frame_settings.json``).
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tools.native_viewer_export import impact_card, impact_factors
from src.tools.native_viewer_export.impact_state import series_at_contact

SETTINGS_PATH = Path(__file__).with_name("contact_frame_settings.json")
_KEYS = ("impact_window_hold_frames", "full_speed_hold_frames")


def _check_hold(frames: Any) -> int:
    if type(frames) is not int or frames < 0:
        raise ValueError(f"hold frames must be an integer >= 0, got {frames!r}")
    return frames


def hold_count(override: int | None, windowed: bool) -> int:
    """Frames the contact frame is repeated: ``override`` or the file default.

    ``windowed`` selects the impact-window default over the full-speed one.
    Raises ``ValueError`` for a negative or non-integer count.
    """
    if override is not None:
        return _check_hold(override)
    raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    return _check_hold(raw[_KEYS[0] if windowed else _KEYS[1]])


def hold_order(times_s: Any, contact_s: float | None, count: int) -> Any:
    """Row order repeating the row at ``contact_s`` ``count`` more times (else identity)."""
    rows = np.arange(len(times_s))
    if contact_s is None or count <= 0:
        return rows
    hit = np.nonzero(np.asarray(times_s) == contact_s)[0]
    if hit.size == 0:
        return rows
    at = int(hit[0])
    return np.concatenate([rows[: at + 1], np.full(count, at), rows[at + 1 :]])


def contact_time_s(series: Any, ball: Any, impact_s: float) -> float | None:
    """Contact instant of ``series`` with ``ball`` near ``impact_s``, or ``None``."""
    found = series_at_contact(series, ball.position_m, ball.radius_m, impact_s)
    if found is None:
        return None
    moved, at = found
    return float(moved.times_s[at])


def refine(swing: Any, impact_s: float) -> tuple[Any, float | None]:
    """``swing`` with its ball on the tee to the contact instant, and that time (or ``None``)."""
    if swing.ball is None:
        return swing, None
    try:
        series = impact_factors.club_series(swing)
        when = contact_time_s(series, swing.ball, impact_s)
    except impact_card._EXPECTED_ERRORS:
        return swing, None
    if when is None:
        return swing, None
    return replace(swing, ball=replace(swing.ball, hide_after_s=when)), when
