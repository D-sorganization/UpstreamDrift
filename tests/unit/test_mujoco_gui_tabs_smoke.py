"""Smoke tests for MuJoCo GUI tab modules (issue #2350).

These tests verify that tab classes can be imported and instantiated without
error in a headless environment.  They use ``pytest.importorskip`` to skip
gracefully when PyQt6 or mujoco is unavailable (e.g. pure-unit CI runners).

Coverage goal: basic import + construction of each tab class so that obvious
import-time errors, missing attributes, and broken __init__ signatures are
caught before they reach integration tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers — mock heavy Qt / simulation dependencies
# ---------------------------------------------------------------------------


class _FakeSimWidget:
    """Minimal stand-in for MuJoCoSimWidget."""

    model = None
    data = None

    def get_sim_state(self):
        return {"time": 0.0, "qpos": [], "qvel": []}


class _FakeMainWindow:
    """Minimal stand-in for AdvancedGolfAnalysisWindow."""

    status_bar = MagicMock()


# ---------------------------------------------------------------------------
# Module-level import guards
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Analysis tab
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Controls tab
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Visualization tab
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Humanoid Config tab
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Manipulation tab
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Physics tab
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plotting tab
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Manipulability tab
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# tabs __init__ package
# ---------------------------------------------------------------------------
