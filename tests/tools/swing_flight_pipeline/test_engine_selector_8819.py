"""Engine-selector honesty tests for the Swing→Flight Pipeline GUI (#8819).

The "Physics Engine Source" combo used to be cosmetic: any selection was
stamped onto manually-built numbers.  Now:

* unimplemented engine entries are visible but DISABLED, with a tooltip
  explaining why;
* the result's ``engine_name`` comes from the provider that actually
  produced the swing state;
* a provider that lies about its engine raises a contract error.
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.physics.swing_state_providers import (  # noqa: E402
    REASON_NOT_IMPLEMENTED,
    REASON_NOT_INSTALLED,
    MuJoCoSwingStateProvider,
)

_MUJOCO_AVAILABLE = MuJoCoSwingStateProvider().is_available()
from src.tools.swing_flight_pipeline.gui import SwingFlightWidget  # noqa: E402


pytestmark = pytest.mark.unit


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


@pytest.fixture
def widget():
    w = SwingFlightWidget()
    yield w
    w.cleanup()
    w.deleteLater()


def _item(widget: SwingFlightWidget, name: str):
    combo = widget._engine_combo
    index = [combo.itemText(i) for i in range(combo.count())].index(name)
    return index, combo.model().item(index)


@pytest.mark.parametrize("engine", ["drake", "pinocchio"])
def test_unimplemented_engines_are_disabled(widget, engine):
    _, item = _item(widget, engine)
    assert item is not None
    assert not item.isEnabled(), f"{engine} must be disabled until implemented"


def test_mujoco_entry_enabled_when_engine_available(widget):
    """#8975: MuJoCo sourcing is implemented — the entry enables whenever
    the mujoco package (and its MJCF asset) is importable."""
    _, item = _item(widget, "mujoco")
    assert item is not None
    assert item.isEnabled() == _MUJOCO_AVAILABLE


def test_manual_entry_is_enabled_and_default(widget):
    _, item = _item(widget, "manual")
    assert item.isEnabled()
    assert widget._engine_combo.currentText() == "manual"


@pytest.mark.parametrize("engine", ["drake", "pinocchio"])
def test_disabled_entries_carry_reason_tooltip(widget, engine):
    index, _ = _item(widget, engine)
    tooltip = widget._engine_combo.itemData(index, Qt.ItemDataRole.ToolTipRole)
    assert tooltip in (REASON_NOT_IMPLEMENTED, REASON_NOT_INSTALLED)


def test_mujoco_tooltip_honest(widget):
    """Enabled mujoco carries no misleading 'unavailable' tooltip; when the
    package is missing the tooltip says exactly that."""
    index, _ = _item(widget, "mujoco")
    tooltip = widget._engine_combo.itemData(index, Qt.ItemDataRole.ToolTipRole)
    if _MUJOCO_AVAILABLE:
        assert tooltip not in (REASON_NOT_IMPLEMENTED, REASON_NOT_INSTALLED)
    else:
        assert tooltip == REASON_NOT_INSTALLED


def test_result_engine_name_matches_selected_provider(widget):
    """End-to-end: the reported engine is the provider that ran (#8819)."""
    pytest.importorskip("src.shared.python.physics.impact_model")
    widget._engine_combo.setCurrentText("manual")

    class _CapturingPipeline:
        last_swing = None

        def run(self, swing):
            _CapturingPipeline.last_swing = swing
            raise RuntimeError("stop after swing-state production")

    import src.shared.python.physics.swing_ball_flight_pipeline as pipe_mod

    original = pipe_mod.SwingBallFlightPipeline
    pipe_mod.SwingBallFlightPipeline = _CapturingPipeline  # type: ignore[misc]
    try:
        widget._run_pipeline()
    finally:
        pipe_mod.SwingBallFlightPipeline = original  # type: ignore[misc]

    swing = _CapturingPipeline.last_swing
    assert swing is not None
    assert swing.engine_name == "manual"


def test_selecting_unavailable_provider_surfaces_error(widget):
    """Programmatically forcing a disabled provider must fail loudly."""
    widget._engine_combo.setCurrentText("drake")
    widget._run_pipeline()
    text = widget._results_text.toPlainText()
    assert "Pipeline error" in text
    assert "not available" in text
