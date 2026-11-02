"""The rust stream worker QThread never outlives the generator (#14321).

UD pair of Tools#5951: ``_RustStreamWorker.run()`` emits ``stream_finished``
before it returns, so the event loop can quit while the thread is still
alive. Destroying that QThread later aborts the interpreter ("QThread:
Destroyed while thread is still running").
"""

from __future__ import annotations

import queue
import time
from types import SimpleNamespace

import pytest

from src.shared.python.ai.adapters import rust_adapter

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")


class _StubEngine:
    def stream_response(self, _prompt: str) -> list[str]:
        return ["a", "b"]


@pytest.fixture()
def qapp():  # type: ignore[no-untyped-def]
    """A QApplication so QEventLoop and cross-thread signals work."""
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_worker_thread_is_finished_when_generator_completes(qapp: object) -> None:
    """A worker still unwinding after ``stream_finished`` is joined."""
    worker_cls = rust_adapter._make_rust_stream_worker_class()
    assert worker_cls is not None

    class _SlowUnwindWorker(worker_cls):  # type: ignore[misc, valid-type]
        def run(self) -> None:
            super().run()
            time.sleep(0.3)

    holder = SimpleNamespace(engine=_StubEngine())

    out = list(
        rust_adapter.RustAgentAdapter._stream_with_worker(
            holder,  # type: ignore[arg-type]
            _SlowUnwindWorker,
            "hi",
            queue.Queue(),
        )
    )

    assert [chunk.content for chunk in out] == ["a", "b"]
    assert holder._active_worker.isFinished()
