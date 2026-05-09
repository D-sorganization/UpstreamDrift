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


def test_run_c3d_viewer_imports_cleanly() -> None:
    """Spawning the wrapper must not produce an ``ImportError`` traceback.

    Skipped when optional C3D viewer dependencies (pandas, PyQt) are unavailable.
    """
    if not WRAPPER.exists():
        pytest.skip(f"wrapper missing: {WRAPPER}")

    if not _check_viewer_deps_available():
        pytest.skip(
            "C3D viewer optional dependencies (pandas, PyQt) not available. "
            "Install with: pip install pandas PyQt5"
        )

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    proc = subprocess.Popen(
        [sys.executable, str(WRAPPER)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(REPO_ROOT),
    )
    try:
        try:
            proc.wait(timeout=5)
            stdout, stderr = proc.communicate(timeout=2)
            err_text = stderr.decode("utf-8", errors="replace")
            # If the process exited within 5 s it must NOT have hit an
            # ImportError or other Traceback at module-import time.
            assert "Traceback" not in err_text, err_text
            assert "ImportError" not in err_text, err_text
        except subprocess.TimeoutExpired:
            # Still alive after 5 s -> imports cleared the bar; tear down.
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            stderr_bytes = proc.stderr.read() if proc.stderr else b""
            err_text = stderr_bytes.decode("utf-8", errors="replace")
            assert "ImportError" not in err_text, err_text
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
