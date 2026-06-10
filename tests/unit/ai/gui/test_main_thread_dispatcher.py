"""Tests for MainThreadToolDispatcher (GUI-thread marshalling).

Verifies that a tool thunk dispatched from a background QThread actually
executes on the dispatcher's owning (GUI) thread, that a same-thread call
runs inline (no deadlock), and that exceptions propagate to the caller.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Generator
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QThread  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.ai.gui.assistant_widgets import (  # noqa: E402
    MainThreadToolDispatcher,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_gl]


@pytest.fixture(scope="module")
def app() -> Generator[QApplication, None, None]:
    instance = QApplication.instance() or QApplication([])
    assert isinstance(instance, QApplication)
    yield instance


def test_same_thread_call_runs_inline(app: QApplication) -> None:
    dispatcher = MainThreadToolDispatcher()
    out = dispatcher(lambda: threading.get_ident())
    assert out == threading.get_ident()


def test_exception_propagates_inline(app: QApplication) -> None:
    dispatcher = MainThreadToolDispatcher()

    def boom() -> None:
        raise ValueError("boom-inline")

    with pytest.raises(ValueError, match="boom-inline"):
        dispatcher(boom)


def test_worker_thread_marshals_to_owning_thread(app: QApplication) -> None:
    dispatcher = MainThreadToolDispatcher()  # owned by this (GUI) thread
    gui_ident = threading.get_ident()
    captured: dict[str, int] = {}

    class _Worker(QThread):
        def run(self) -> None:
            captured["ran_on"] = dispatcher(lambda: threading.get_ident())
            captured["worker_ident"] = threading.get_ident()

    worker = _Worker()
    worker.start()

    deadline = time.monotonic() + 10.0
    while not worker.isFinished():
        app.processEvents()
        if time.monotonic() > deadline:
            worker.terminate()
            worker.wait()
            pytest.fail("dispatcher did not complete within 10s (event loop stall)")
    worker.wait()

    assert captured["worker_ident"] != gui_ident  # worker really was off-thread
    assert captured["ran_on"] == gui_ident  # thunk ran on the GUI thread


def test_worker_thread_exception_propagates(app: QApplication) -> None:
    dispatcher = MainThreadToolDispatcher()
    captured: dict[str, BaseException] = {}

    def raise_worker_error() -> Any:
        raise ValueError("boom-worker")

    class _Worker(QThread):
        def run(self) -> None:
            try:
                dispatcher(raise_worker_error)
            except ValueError as exc:  # noqa: BLE001 - capture for assertion
                captured["error"] = exc

    worker = _Worker()
    worker.start()
    deadline = time.monotonic() + 10.0
    while not worker.isFinished():
        app.processEvents()
        if time.monotonic() > deadline:
            worker.terminate()
            worker.wait()
            pytest.fail("dispatcher did not complete within 10s")
    worker.wait()

    assert isinstance(captured.get("error"), ValueError)
    assert "boom-worker" in str(captured["error"])
