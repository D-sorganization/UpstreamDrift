"""Single integration point for the impact time of an adapter series (GCV-16).

Today this uses the shared peak-speed detector
(``motion_matching.loaders._align.detect_impact_index``).  The checked
closest-approach rule lands with OSV-8/OSV-10 in ``model_appearance/club_face``;
when it is on ``main``, pass it as ``closest_approach`` (or replace the marked
line) and nothing else changes.
"""

from __future__ import annotations

from collections.abc import Callable

from ..clubhead_series import ClubheadSeries


def select_impact_index(
    series: ClubheadSeries,
    closest_approach: Callable[[ClubheadSeries], int] | None = None,
) -> int:
    """Impact sample index; ``closest_approach`` overrides the speed detector."""
    if closest_approach is not None:
        idx = int(closest_approach(series))
        if not 0 <= idx < len(series):
            raise ValueError(
                f"closest_approach returned {idx}, outside [0, {len(series)})"
            )
        return idx
    # INTEGRATION POINT (OSV-8/OSV-10): replace with the club_face closest-approach rule.
    from src.shared.python.motion_matching.loaders._align import detect_impact_index

    return int(detect_impact_index(series.times_s, series.face_center_m))
