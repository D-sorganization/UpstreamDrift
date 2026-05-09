"""GUI Component Tests - MuJoCo Simulation Widget and Launchers.

This module tests GUI components with appropriate mocking for headless environments.
Tests verify actual behavior, not just code execution.

Note: These tests require mujoco and PyQt6 to be installed. Tests are skipped
if dependencies are missing rather than using extensive mocking.
"""

import os

# Ensure offscreen platform BEFORE any Qt imports so that a QApplication can be
# created even when no X server / Wayland display is available (headless CI).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MUJOCO_GL", "osmesa")

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# sympy is needed by some transitive imports; skip module if unavailable
pytest.importorskip("sympy", reason="sympy not installed")

from src.shared.python.engine_core.engine_availability import (
    PYQT6_AVAILABLE,
    skip_if_unavailable,
)
from src.shared.python.gui_pkg.gui_utils import get_qapp

if PYQT6_AVAILABLE:
    pass


def _load_model_or_skip(widget, xml_string: str) -> None:
    """Call *widget.load_model_from_xml* and skip the test on GL errors.

    In headless CI environments without EGL / OSMesa the MuJoCo renderer
    cannot create an OpenGL context, raising ``mujoco.FatalError``.
    """
    try:
        widget.load_model_from_xml(xml_string)
    except Exception as exc:  # noqa: BLE001
        # mujoco.FatalError (gladLoadGL) or RuntimeError from renderer init
        if "gladLoadGL" in str(exc) or "OpenGL" in str(exc):
            pytest.skip(f"MuJoCo GL unavailable (headless environment): {exc}")
        raise


@pytest.fixture(scope="module")
def qapp() -> Generator[Any, None, None]:
    """Create a QApplication instance for tests that need it."""
    if not PYQT6_AVAILABLE:
        pytest.skip("PyQt6 not available")
    try:
        app = get_qapp()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qt initialisation failed (headless environment?): {exc}")
    yield app
