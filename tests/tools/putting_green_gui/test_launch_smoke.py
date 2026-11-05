"""Launch smoke test for the PyQt6 putting demo (#14448).

Starts the real app offscreen in a subprocess, runs one putt through the
shared putting core and checks it exits cleanly. Also pins the documented
one-line launch command and the clear failure when the core is missing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.tools.putting_green_gui import __main__ as launch  # noqa: E402

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCH_COMMAND = "python -m src.tools.putting_green_gui"


def test_smoke_run_does_one_putt_and_exits_cleanly() -> None:
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    proc = subprocess.run(
        [sys.executable, "-m", "src.tools.putting_green_gui", "--smoke"],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "Putting Simulation" in proc.stdout
    assert "Total roll:" in proc.stdout


def test_missing_core_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A ``None`` entry makes ``import`` raise ImportError for that module.
    monkeypatch.setitem(sys.modules, launch.CORE_MODULE, None)
    assert launch.main(["--smoke"]) == launch.EXIT_CORE_MISSING
    err = capsys.readouterr().err
    assert launch.CORE_MODULE in err
    assert "putting core" in err.lower()


def test_launch_command_is_documented() -> None:
    doc = (_REPO_ROOT / "docs/help/putting_green.md").read_text(encoding="utf-8")
    assert _LAUNCH_COMMAND in doc
    assert f"{_LAUNCH_COMMAND} --smoke" in doc
