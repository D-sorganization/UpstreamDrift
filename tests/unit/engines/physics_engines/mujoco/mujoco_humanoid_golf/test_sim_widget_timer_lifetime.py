"""``MuJoCoSimWidget`` must not outlive its own simulation timer (UD #9474).

Why this matters
----------------
``MuJoCoSimWidget.__init__`` starts a 60 fps ``QTimer`` unconditionally, and
until this change the class had no ``closeEvent``, no ``hideEvent`` and no
teardown method. ``QWidget.close()`` only *hides* a widget, so every sim widget
a test constructed kept stepping MuJoCo and rendering frames for the remaining
lifetime of the ``QApplication`` -- which, because the unit lane's
``QApplication`` is a process-wide singleton, means the rest of the pytest
session.

In CI that leaked timer fired during an unrelated later test, reached
``SimRenderingMixin._add_force_torque_overlays`` -> ``get_cv2()``, and the
``import cv2`` there raised ``AttributeError`` (a half-initialised OpenCV, not
an ``ImportError``). An exception escaping a Qt slot is fatal under PyQt6, so
the interpreter aborted:

    AttributeError: partially initialized module 'cv2' has no attribute
    'mat_wrapper' (most likely due to a circular import)
    Fatal Python error: Aborted

The lane produced no test summary at all. Both halves are covered here: the
widget must stop its own timer when it is closed, and ``get_cv2`` must honour
its documented contract of returning the module or ``None``.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]

pytest.importorskip("PyQt6", reason="PyQt6 required for Qt lifetime tests")
mujoco = pytest.importorskip("mujoco", reason="MuJoCo required for the sim widget")

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (  # noqa: E402
    sim_rendering_mixin,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_widget import (  # noqa: E402
    MuJoCoSimWidget,
)
from src.shared.python.ui.qt.utils import get_qapp  # noqa: E402


@pytest.fixture
def qapp():
    try:
        return get_qapp()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qt initialisation failed (headless environment?): {exc}")


@pytest.fixture
def widget(qapp):
    """A sim widget that is always torn down, even when a test fails."""
    made = MuJoCoSimWidget(width=64, height=48, fps=30)
    try:
        yield made
    finally:
        made.stop_simulation()
        made.deleteLater()


def test_timer_is_running_after_construction(widget) -> None:
    """Baseline: the widget really does start a timer in ``__init__``."""
    assert widget.timer.isActive()


def test_close_stops_the_simulation_timer(widget) -> None:
    """Closing the widget must stop the timer it owns.

    Before UD #9474 this failed: ``close()`` only hid the widget, so the
    timer kept firing ``_on_timer`` for the rest of the process.
    """
    assert widget.timer.isActive()

    widget.close()

    assert not widget.timer.isActive(), (
        "close() left the simulation timer running; the widget will keep "
        "stepping MuJoCo during unrelated later tests"
    )
    assert widget.running is False


def test_stop_simulation_is_idempotent(widget) -> None:
    """Teardown runs from several paths, so it must tolerate repetition."""
    widget.stop_simulation()
    widget.stop_simulation()

    assert not widget.timer.isActive()
    assert widget.running is False


def test_close_is_idempotent(widget) -> None:
    widget.close()
    widget.close()

    assert not widget.timer.isActive()


class TestGetCv2Contract:
    """``get_cv2`` promises "the cv2 module, or None". It must keep that."""

    @staticmethod
    @pytest.fixture(autouse=True)
    def _reset_cv2_cache():
        original = dict(sim_rendering_mixin._cv2_state)
        sim_rendering_mixin._cv2_state.update({"lib": None, "invalid": False})
        try:
            yield
        finally:
            sim_rendering_mixin._cv2_state.clear()
            sim_rendering_mixin._cv2_state.update(original)

    @staticmethod
    def test_returns_none_when_opencv_is_half_initialised(monkeypatch) -> None:
        """A partially initialised OpenCV raises AttributeError, not ImportError.

        This is the exact CI failure: cv2's own ``bootstrap()`` reached
        ``cv2.typing`` before ``cv2.mat_wrapper`` was bound and raised
        ``AttributeError``. Before UD #9474 ``get_cv2`` guarded only
        ``ImportError``, so the AttributeError escaped through the Qt slot and
        aborted the interpreter.
        """
        real_import = __import__

        def broken_cv2_import(name, *args, **kwargs):
            if name == "cv2":
                raise AttributeError(
                    "partially initialized module 'cv2' has no attribute "
                    "'mat_wrapper' (most likely due to a circular import)"
                )
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", broken_cv2_import)

        assert sim_rendering_mixin.get_cv2() is None

    @staticmethod
    def test_returns_none_when_opencv_is_absent(monkeypatch) -> None:
        real_import = __import__

        def missing_cv2_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("No module named 'cv2'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", missing_cv2_import)

        assert sim_rendering_mixin.get_cv2() is None

    @staticmethod
    def test_failure_is_remembered_so_the_import_is_attempted_once(
        monkeypatch,
    ) -> None:
        """A broken OpenCV must not be re-imported on every rendered frame."""
        real_import = __import__
        attempts = []

        def broken_cv2_import(name, *args, **kwargs):
            if name == "cv2":
                attempts.append(name)
                raise AttributeError("half-initialised")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", broken_cv2_import)

        assert sim_rendering_mixin.get_cv2() is None
        assert sim_rendering_mixin.get_cv2() is None
        assert len(attempts) == 1
