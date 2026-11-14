"""Regression test for issue #14812: pose_studio widget tests must not depend on order or mock pollution.

``test_widgets.py`` builds ``UnitsBadge`` and ``EnginePicker`` widgets. Without
a real ``QApplication``, Qt aborts the process (rc 134), and its assertions only
held when earlier tests mocked PyQt6. Running the file in a fresh interpreter
with ``--noconftest`` reproduces the failure red-first.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("PyQt6.QtWidgets")

pytestmark = pytest.mark.unit

_TARGET = Path(__file__).with_name("test_widgets.py")
_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_widgets_tests_pass_without_a_prior_qapplication() -> None:
    """``test_widgets.py`` must pass in a process where nothing made a QApplication."""
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
