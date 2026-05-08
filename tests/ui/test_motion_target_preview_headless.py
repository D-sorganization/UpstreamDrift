"""Headless smoke test for the multi-source motion-target preview GUI.

Verifies the matcher GUI can:

1. Load a ``BodyTarget`` and a ``ClubTarget`` from the same C3D file.
2. Scrub the timeline from sample 0 to the last sample without raising.
3. Report an artist count that matches the configured layer-visibility.

Headless invariants:

* ``QT_QPA_PLATFORM=offscreen`` (set at module import).
* matplotlib backend forced to ``Agg``.
* The whole test must complete in under 30 s.

This test is skipped on the wave-4 branch until the animated-preview /
source-toggle work in issues #4481 (body) and #4482 (animated preview)
lands. The skip is loud rather than silent so it shows up in test reports.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

# Force headless before any Qt / matplotlib import surface is touched.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

pytestmark = [pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER_C3D = REPO_ROOT / "data" / "C3D_TA_Driver.c3d"
TIME_BUDGET_S = 30.0


def test_motion_target_preview_scrub_timeline() -> None:
    """Open the preview, load both targets, scrub the timeline end-to-end."""
    if not DRIVER_C3D.exists():
        pytest.skip(f"driver C3D fixture not present: {DRIVER_C3D}")

    matcher_mod = pytest.importorskip(
        "src.shared.python.motion_matching.ui.motion_target_preview",
        reason="waiting on #4482 (animated preview / source-toggle GUI)",
    )
    body_loader_mod = pytest.importorskip(
        "src.shared.python.motion_matching.load_body_target",
        reason="waiting on #4481 (load_body_target)",
    )
    club_loader_mod = pytest.importorskip(
        "src.shared.python.motion_matching.load_club_target",
        reason="load_club_target dispatcher not present on this branch",
    )
    target_mod = pytest.importorskip(
        "src.shared.python.motion_matching.target",
        reason="target module not present on this branch",
    )
    pytest.importorskip(
        "PySide6", reason="Qt bindings not installed in this environment"
    )

    opts = target_mod.AlignOptions()
    club = club_loader_mod.load_club_target(DRIVER_C3D, opts=opts)
    body = body_loader_mod.load_body_target(  # type: ignore[attr-defined]
        DRIVER_C3D, opts=opts, impact_source=club
    )

    start = time.monotonic()
    window = matcher_mod.MotionTargetPreviewWindow()  # type: ignore[attr-defined]
    try:
        window.load_targets(body=body, club=club)  # type: ignore[attr-defined]
        n_frames = int(club.time.shape[0])
        for idx in range(0, n_frames, max(1, n_frames // 20)):
            window.set_timeline_index(idx)  # type: ignore[attr-defined]
        window.set_timeline_index(n_frames - 1)  # type: ignore[attr-defined]

        # Artist count must match the visible-layer expectation.
        expected = int(window.expected_artist_count())  # type: ignore[attr-defined]
        actual = int(window.current_artist_count())  # type: ignore[attr-defined]
        assert actual == expected, (
            f"artist count {actual} != expected {expected} after end-to-end scrub"
        )
    finally:
        close = getattr(window, "close", None)
        if callable(close):
            close()

    elapsed = time.monotonic() - start
    assert elapsed < TIME_BUDGET_S, (
        f"headless smoke exceeded {TIME_BUDGET_S} s budget (took {elapsed:.1f} s)"
    )
