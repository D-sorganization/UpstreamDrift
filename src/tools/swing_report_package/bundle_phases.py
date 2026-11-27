"""Key phases of a swing bundle for the package runner (#15724, SRP-5).

Lazily feeds :func:`~.phases.detect_phases` MuJoCo points of the reference
swing: ``grip`` and ``clubhead`` (face centre) from the exporter's club model,
the rest from the bodies named by ``phase_bodies`` in ``runner_settings.json``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.tools.swing_report_package.phases import KeyPhase, detect_phases
from src.tools.swing_report_package.runner import load_runner_settings

__all__ = ["bundle_phases"]


def _body_tracks(
    bundle: Path, swing: str, club: str, bodies: Mapping[str, str]
) -> tuple[Sequence[float], dict[str, Any], float]:
    """State times, world points (club points plus ``bodies``), impact (s)."""
    from src.tools.native_viewer_export.club_pose import ClubPoseModel
    from src.tools.native_viewer_export.core import load_swing_input
    from src.tools.native_viewer_export.overlay import detect_impact_time_s

    sw = load_swing_input(Path(bundle), swing=swing, club=club)
    model = ClubPoseModel(sw.bundle.spec_bytes)
    order = sw.bundle.coordinate_order
    rows = []
    for q in sw.q:
        coords = dict(zip(order, map(float, q), strict=True))
        row = model.body_positions(coords, list(bodies.values()))
        named: dict[str, Any] = {k: row[b] for k, b in bodies.items()}
        named["grip"] = model.grip_midpoint_world(coords)
        named["clubhead"] = model.face_world(coords).centre_m
        rows.append(named)
    points = {k: np.asarray([r[k] for r in rows], dtype=float) for k in rows[0]}
    times = [sw.time_s(i) for i in range(sw.n_states)]
    return times, points, detect_impact_time_s(sw)


def _top_time_s(time_s: Sequence[float], clubhead: Any) -> float:
    """Top of the backswing: the slowest clubhead state before the peak."""
    from src.shared.python.model_appearance.club_face import top_of_backswing_index

    t = np.asarray(time_s, dtype=float)
    return float(t[top_of_backswing_index(t, np.asarray(clubhead, dtype=float))])


def bundle_phases(
    bundle: Path | str,
    swing: str,
    club: str,
    *,
    impact_time_s: float | None = None,
    settings: Mapping[str, Any] | None = None,
) -> Callable[[str], tuple[KeyPhase, ...]]:
    """A ``phases_of`` for :func:`~.runner.run_package` reading ``bundle``;
    ``impact_time_s`` overrides the detected impact.

    Raises:
        ValueError: the settings name no ``phase_bodies`` (at once); a phase
            the detector cannot find (when called).
    """
    look = load_runner_settings() if settings is None else settings
    bodies = dict(look.get("phase_bodies") or {})
    if not bodies:
        raise ValueError("runner settings need 'phase_bodies' for key phases")

    def phases_of(_swing: str) -> tuple[KeyPhase, ...]:
        t, points, detected = _body_tracks(Path(bundle), swing, club, bodies)
        impact = detected if impact_time_s is None else impact_time_s
        return detect_phases(t, points, impact, _top_time_s(t, points["clubhead"]))

    return phases_of
