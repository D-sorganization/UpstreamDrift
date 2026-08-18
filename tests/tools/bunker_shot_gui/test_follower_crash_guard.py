"""A follower that refuses a frame must not take the process with it.

Followers refuse an out-of-range index rather than clamping it, and that
contract is correct: a clamped index would leave one view showing a
different moment from the one driving it.

Delivering the refusal through a Qt signal is what makes it dangerous. An
exception escaping a slot unwinds through the C++ event loop, which aborts
the interpreter with ``0xC0000409`` and **no Python traceback**, so the
mechanism meant to make a desynchronisation obvious instead makes it
undiagnosable. These tests pin the fix: the refusal survives, its diagnosis
improves, and the rest of the workbench keeps running.

The crash is genuine rather than theoretical -- it was hit while wiring the
sand-slice view, and the same shape exists for every other follower.
"""

from __future__ import annotations

import logging
import os
import sys

import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.bunker_shot_gui.widgets import SoleLoadFieldWidget  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="session", autouse=True)
def qapp() -> QApplication:
    """One offscreen QApplication for the module."""
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv[:1])
    return application


class _Refuses:
    """A follower that refuses every frame, the way a real one refuses a bad index."""

    def __init__(self) -> None:
        self.calls = 0

    def set_frame(self, frame: int) -> None:
        self.calls += 1
        raise ValueError(f"frame {frame} is outside the shot")


class _Accepts:
    """A follower that accepts, to prove one bad follower does not stop the rest."""

    def __init__(self) -> None:
        self.frames: list[int] = []

    def set_frame(self, frame: int) -> None:
        self.frames.append(int(frame))


class TestARefusalDoesNotEscapeTheSignal:
    def test_emitting_to_a_refusing_follower_does_not_raise(self) -> None:
        transport = SoleLoadFieldWidget("guard")
        transport.link(_Refuses())
        # Unguarded, this is the call that aborts the interpreter.
        transport.frame_changed.emit(0)

    def test_the_refusal_is_logged_with_its_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        transport = SoleLoadFieldWidget("guard")
        transport.link(_Refuses())
        with caplog.at_level(logging.ERROR):
            transport.frame_changed.emit(3)
        assert any(r.exc_info for r in caplog.records), (
            "the refusal must carry its traceback; a bare message would be no "
            "better a diagnosis than the crash it replaces"
        )
        assert "_Refuses" in caplog.text
        assert "3" in caplog.text

    def test_the_refusal_is_not_swallowed_silently(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        transport = SoleLoadFieldWidget("guard")
        transport.link(_Refuses())
        with caplog.at_level(logging.ERROR):
            transport.frame_changed.emit(0)
        assert caplog.records, (
            "a silent pass would hide the desynchronisation the raise exists "
            "to expose, which is the wrong trade in the other direction"
        )


class TestOneBadFollowerDoesNotStopTheRest:
    def test_a_healthy_follower_still_receives_frames(self) -> None:
        transport = SoleLoadFieldWidget("guard")
        healthy = _Accepts()
        transport.link(_Refuses())
        transport.link(healthy)
        transport.frame_changed.emit(1)
        transport.frame_changed.emit(2)
        assert healthy.frames == [1, 2]

    def test_the_refusing_follower_stops_being_called(self) -> None:
        transport = SoleLoadFieldWidget("guard")
        refuses = _Refuses()
        transport.link(refuses)
        for frame in range(5):
            transport.frame_changed.emit(frame)
        assert refuses.calls == 1, (
            "a follower that has desynchronised should be dropped, not "
            "re-invoked on every tick to log the same failure repeatedly"
        )
