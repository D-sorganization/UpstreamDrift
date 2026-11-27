"""Regression test for issue #14807: Drake UI tests must not depend on test order.

``test_drake_ui_abort_regression_11934.py`` builds a ``QMainWindow`` in every
test. Qt aborts the process when no ``QApplication`` exists yet, so a test that
does not request ``qapp`` passed only when an earlier test on the same xdist
worker had created one. Running just those tests in a fresh interpreter, where
nothing has created a ``QApplication``, reproduces the merge-queue crash.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("PyQt6.QtWidgets")

pytestmark = pytest.mark.unit

_TARGET = Path(__file__).with_name("test_drake_ui_abort_regression_11934.py")
_REPO_ROOT = Path(__file__).resolve().parents[5]


def test_tests_without_prior_qapplication_do_not_abort() -> None:
    """The tests that never asked for ``qapp`` must pass in a fresh process."""
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
            "-k",
            "guards_uninitialized or force_glyphs or resolve_induced",
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "3 passed" in result.stdout
