"""Unit tests for ``fsp_metrics_widget`` (Phase 3 of the FSP epic, #5504).

The module must import cleanly even when PyQt6 is not installed; the
class falls back to a no-op stub in that case.  When Qt is available we
verify that ``set_result`` writes the expected text into the labels.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

try:
    from PyQt6.QtWidgets import QApplication  # noqa: F401

    HAS_QT = True
except ImportError:  # pragma: no cover - environment-dependent
    HAS_QT = False


_MODULE_PATH = "src.shared.python.ui.qt.widgets.fsp_metrics_widget"


# ---------------------------------------------------------------------------
# Headless import
# ---------------------------------------------------------------------------


def test_widget_class_is_importable_headless() -> None:
    """The widget module imports cleanly even without Qt — headless-safe."""
    mod = importlib.import_module(_MODULE_PATH)
    assert hasattr(mod, "FspMetricsWidget")


def test_widget_can_be_instantiated_headless() -> None:
    """Stub instantiation must not raise on environments without Qt."""
    mod = importlib.import_module(_MODULE_PATH)
    # On Qt-less environments the class is a no-op stub. On Qt-enabled
    # environments we still need a running QApplication, which is set up
    # by the qt_app fixture below.
    if not HAS_QT:
        widget = mod.FspMetricsWidget()
        assert widget is not None


# ---------------------------------------------------------------------------
# Qt-only behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
def qt_app():
    if not HAS_QT:
        pytest.skip("PyQt6 not installed")
    from PyQt6.QtWidgets import QApplication as _QA

    app = _QA.instance() or _QA([])
    yield app


class _FakeResult:
    def __init__(self, slope: float = 12.5, direction: float = -3.0) -> None:
        self.slope_deg = slope
        self.direction_deg = direction
        self.clubhead_deviations = np.array([0.01, -0.02, 0.005, 0.0])
        self.hand_deviations = np.array([0.0, 0.0, 0.0, 0.0])


def _last_set_text(label: object) -> str:
    """Return the last argument passed to ``setText`` -- robust to both
    real PyQt6 and the conftest's ``DummyWidget`` mock."""
    set_text = getattr(label, "setText", None)
    if set_text is not None and hasattr(set_text, "call_args"):
        call_args = set_text.call_args
        if call_args is not None and call_args.args:
            return str(call_args.args[0])
    # Real PyQt6 path -- ``text()`` returns the current text.
    text = getattr(label, "text", None)
    if callable(text):
        return str(text())
    return ""


@pytest.mark.skipif(not HAS_QT, reason="PyQt6 not installed")
class TestFspMetricsWidgetWithQt:
    """Tests for FspMetricsWidget that require a live PyQt6 installation."""

    def test_set_result_updates_slope_label(self, qt_app) -> None:
        from src.shared.python.ui.qt.widgets.fsp_metrics_widget import FspMetricsWidget

        widget = FspMetricsWidget()
        widget.set_result(_FakeResult(slope=12.5, direction=-3.0))
        assert "12.5" in _last_set_text(widget._slope_label)
        assert "-3.0" in _last_set_text(widget._direction_label)

    def test_set_result_summarises_deviations(self, qt_app) -> None:
        from src.shared.python.ui.qt.widgets.fsp_metrics_widget import FspMetricsWidget

        widget = FspMetricsWidget()
        widget.set_result(_FakeResult())
        text = _last_set_text(widget._chart_placeholder).lower()
        # Expect both "mean" and "max" in the formatted summary.
        assert "mean" in text
        assert "max" in text

    def test_set_result_tolerates_missing_deviations(self, qt_app) -> None:
        from src.shared.python.ui.qt.widgets.fsp_metrics_widget import FspMetricsWidget

        class _NoDevs:
            slope_deg = 5.0
            direction_deg = 1.0

        widget = FspMetricsWidget()
        widget.set_result(_NoDevs())  # must not raise
        assert "5.0" in _last_set_text(widget._slope_label)
