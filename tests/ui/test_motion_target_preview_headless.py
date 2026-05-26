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
