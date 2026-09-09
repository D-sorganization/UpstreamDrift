"""Real subprocess terminal signals restore the capture command lifecycle."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
from PyQt6.QtCore import QEventLoop, QTimer
from src.tools.capture_rig.process_runner import RigProcessRunner
from tests.tools.capture_rig.test_gui import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def wait_for_completion(
    runner: RigProcessRunner, start: Callable[[], None]
) -> list[int]:
    loop = QEventLoop()
    codes: list[int] = []

    def completed(code: int) -> None:
        codes.append(code)
        loop.quit()

    runner.finished.connect(completed)
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    timeout.start(5000)
    start()
    if not codes:
        loop.exec()
    timeout.stop()
    runner.finished.disconnect(completed)
    return codes


def test_missing_program_reports_failure_and_allows_retry() -> None:
    _app()
    runner = RigProcessRunner()
    output: list[str] = []
    runner.output.connect(output.append)
    codes = wait_for_completion(
        runner, lambda: runner.run(["missing-capture-program-9857"])
    )
    assert len(codes) == 1 and codes[0] != 0
    assert "could not start" in "".join(output).lower()
    assert not runner.busy
    assert wait_for_completion(
        runner, lambda: runner.run([sys.executable, "-c", "pass"])
    ) == [0]


def test_cancel_emits_one_completion() -> None:
    _app()
    runner = RigProcessRunner()

    def start() -> None:
        runner.run([sys.executable, "-c", "import time; time.sleep(30)"])
        QTimer.singleShot(100, runner.stop)

    codes = wait_for_completion(runner, start)
    assert len(codes) == 1 and codes[0] != 0
    assert not runner.busy
