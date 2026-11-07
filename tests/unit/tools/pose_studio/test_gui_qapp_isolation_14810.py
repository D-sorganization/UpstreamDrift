"""Regression test for issue #14810: pose_studio GUI tests must not depend on order.

``test_gui.py`` patches ``gui.QtWidgets.QApplication`` with a mock and then
builds real widgets. Qt aborts the process when no real ``QApplication`` exists
yet, so the file passed only when an earlier test on the same xdist worker had
created one. Running the file in a fresh interpreter reproduces the
merge-queue crash.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("PyQt6.QtWidgets")

pytestmark = pytest.mark.unit

_TARGET = Path(__file__).with_name("test_gui.py")
_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_gui_tests_pass_without_a_prior_qapplication() -> None:
    """``test_gui.py`` must pass in a process where nothing made a QApplication."""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "--noconftest",
            "-q",
            str(_TARGET),
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert " failed" not in result.stdout
