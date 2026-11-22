"""Offscreen: a load never freezes the Model Explorer (#15150, G3).

Fast fakes replace the scene build and club-head speed to check the threading
contract: work off the GUI thread, progress shown, errors reported.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6", reason="the explorer needs PyQt6")

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget  # noqa: E402

from src.tools.model_explorer import results_loader_worker as worker  # noqa: E402
from src.tools.model_explorer.golfer_playback import TimeSeries  # noqa: E402

pytestmark = [pytest.mark.ui, pytest.mark.integration]

HOST = "src.tools.model_explorer.results_playback_host"
SLOW_LOAD_S = 0.6
TICK_MS = 20
CLUB = TimeSeries("Club-head speed", "mph", np.zeros(4))


class _FakeSession:
    title, frame_count, current_frame, current_time_s = "Fake run", 4, 0, 0.0
    pose_source, markers, closed = object(), None, False

    def scrub(self, index: int) -> None:
        self.current_frame = index

    def close(self) -> None:
        self.closed = True


class _FakePanel(QWidget):
    def __init__(self, session: Any) -> None:
        super().__init__()
        self.added: list[TimeSeries] = []
        self.add_series = self.added.append
        self.set_heading = self.stop = lambda *_args: None


@pytest.fixture
def qapp() -> QApplication:
    inst = QApplication.instance()
    return inst if isinstance(inst, QApplication) else QApplication(["async"])


@pytest.fixture
def fake_work(qapp: QApplication) -> Generator[list[Any], None, None]:
    made: list[Any] = []

    def _prepare(path: Path, engine: str | None) -> _FakeSession:
        made.append(threading.get_ident())
        time.sleep(SLOW_LOAD_S)  # releases the GIL, like the real C/numpy work
        if not Path(path).name.startswith("ok"):
            raise OSError(f"cannot read {Path(path).name}")
        made.append(_FakeSession())
        return made[-1]

    with (
        patch.object(worker, "prepare_results", _prepare),
        patch.object(worker, "club_head_speed", lambda _pose: CLUB),
        patch(f"{HOST}.ResultsPlaybackHostMixin._playback_panel_for", _FakePanel),
    ):
        yield made


@pytest.fixture
def explorer(fake_work: list[Any]) -> Generator[Any, None, None]:
    from src.tools.model_explorer.gui import MainWidget

    widget = MainWidget()
    with patch.object(QMessageBox, "warning", return_value=0) as box:
        widget.box = box
        yield widget
    widget.cleanup()
    widget.deleteLater()


def _ticks_while(condition: Any) -> int:
    ticks, timer = [0], QTimer()
    timer.timeout.connect(lambda: ticks.__setitem__(0, ticks[0] + 1))
    timer.start(TICK_MS)
    deadline = time.monotonic() + worker.ResultsLoadSettings().wait_timeout_s
    while condition() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(TICK_MS / 4000)
    return ticks[0]


def test_settings_reject_a_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="wait_timeout_s"):
        worker.ResultsLoadSettings(wait_timeout_s=0.0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ok_run.npz", ["loaded", "club", "finished"]),
        ("broken.npz", ["failed cannot read broken.npz", "finished"]),
    ],
)
def test_loader_works_off_the_gui_thread_in_order(
    fake_work: list[Any], name: str, expected: list[str]
) -> None:
    loader, events = worker.ResultsLoader(name), []
    loader.loaded.connect(lambda _s: events.append("loaded"))
    loader.club_ready.connect(lambda _c: events.append("club"))
    loader.failed.connect(lambda m: events.append(f"failed {m}"))
    loader.finished.connect(lambda: events.append("finished"))
    loader.start()
    _ticks_while(lambda: "finished" not in events)
    assert events == expected
    assert fake_work[0] != threading.get_ident(), "loaded on the GUI thread"


def test_explorer_stays_responsive_and_shows_progress_while_loading(
    explorer: Any,
) -> None:
    assert explorer.load_results("ok_driver.npz", wait=False) is True
    assert explorer.is_loading_results
    assert "Loading ok_driver.npz" in explorer.status_bar.currentMessage()
    ticks = _ticks_while(lambda: explorer.playback_panel is None)
    assert ticks >= (SLOW_LOAD_S * 1000 / TICK_MS) / 2, "GUI thread was blocked"
    assert explorer.wait_for_results_load() is True
    assert explorer.playback_panel.added == [CLUB]  # club speed came later


def test_a_new_load_supersedes_one_in_flight(
    explorer: Any, fake_work: list[Any]
) -> None:
    assert explorer.load_results("ok_first.npz", wait=False) is True
    assert explorer.load_results("ok_second.npz") is True
    assert "ok_second.npz" in explorer.status_bar.currentMessage()
    explorer.cleanup()
    first, second = (s for s in fake_work if isinstance(s, _FakeSession))
    assert first.closed, "the superseded run's session leaked"
    assert second.closed


def test_load_errors_reach_the_user(explorer: Any) -> None:
    assert explorer.load_results("broken.npz") is False
    assert "cannot read broken.npz" in str(explorer.box.call_args)
    assert explorer.results_frame_count == 0
    assert not explorer.is_loading_results
