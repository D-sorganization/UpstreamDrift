"""Integration test for the C3D viewer entry-point wrapper.

Pins the package-pivot wrapper added in PR #4595. The wrapper inserts the
engine ``src/`` onto ``sys.path`` so the viewer's relative imports resolve
when invoked as a flat script.

Marked ``slow`` and ``requires_gl`` so the fast headless suite skips it
even though the test runs under ``QT_QPA_PLATFORM=offscreen``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.requires_gl]

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = (
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "python"
    / "src"
    / "apps"
    / "run_c3d_viewer.py"
)


def _check_viewer_deps_available() -> bool:
    """Check if optional C3D viewer dependencies are available.

    The C3D viewer requires pandas and GUI stack. Skip the test if these
    optional dependencies are not installed rather than hard-failing.
    """
    try:
        import pandas  # noqa: F401
    except ImportError:
        return False
    try:
        from PyQt5 import QtWidgets  # noqa: F401
    except ImportError:
        try:
            from PyQt6 import QtWidgets  # noqa: F401
        except ImportError:
            return False
    return True
