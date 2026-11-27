"""Unit tests for Pose Studio presentational widgets (UnitsBadge, EnginePicker)."""

from __future__ import annotations

from PyQt6.QtWidgets import QApplication
import pytest

from src.shared.python.theme.style_constants import Styles
from src.tools.pose_studio.core import SUPPORTED_ENGINES, EngineStatus
from src.tools.pose_studio.widgets.engine_picker import EnginePicker
from src.tools.pose_studio.widgets.units_badge import UnitsBadge

pytestmark = [pytest.mark.unit, pytest.mark.ui]


@pytest.fixture(scope="module", autouse=True)
def qapp() -> QApplication:
    """Ensure a real QApplication exists before any Qt widget is built (#14812)."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_units_badge_set_engine() -> None:
    badge = UnitsBadge()
    # Default engine is drake
    assert badge.text() == "Drake URDF / RPY (rad)"
    assert "The active engine's native pose convention" in badge.toolTip()
    assert badge.styleSheet() == Styles.STATUS_INFO

    # Explicit engine updates reflect their native conventions
    badge.set_engine("mujoco")
    assert badge.text() == "MuJoCo MJCF / Euler (rad)"

    badge.set_engine("pinocchio")
    assert badge.text() == "Pinocchio URDF / RPY (rad)"

    badge.set_engine("opensim")
    assert badge.text() == "OpenSim .osim / coordinates (rad)"

    badge.set_engine("simscape")
    assert badge.text() == "Simscape Parameters / RPY (deg)"

    # Unknown engine fallback
    badge.set_engine("custom")
    assert badge.text() == "custom (unknown)"

    # Type contract: non-string engine_name raises TypeError
    with pytest.raises(TypeError, match="engine_name must be str"):
        badge.set_engine(123)  # type: ignore[arg-type]


def test_engine_picker_initialization() -> None:
    picker = EnginePicker()
    # Default initial engine is drake
    assert picker.current_engine() == "drake"
    assert picker.combo.currentText() == "drake"
    assert picker.combo.count() == len(SUPPORTED_ENGINES)
    assert picker.status_pill.text() == EngineStatus.MOCK.value

    # Custom initial engine
    picker_mujoco = EnginePicker(initial_engine="mujoco")
    assert picker_mujoco.current_engine() == "mujoco"
    assert picker_mujoco.combo.currentText() == "mujoco"

    # Invalid initial engine raises ValueError
    with pytest.raises(ValueError, match="not in SUPPORTED_ENGINES"):
        EnginePicker(initial_engine="invalid_engine")


def test_engine_picker_set_engine_and_status() -> None:
    picker = EnginePicker()

    # Programmatic set_engine updates the combo box and current_engine
    picker.set_engine("mujoco")
    assert picker.current_engine() == "mujoco"
    assert picker.combo.currentText() == "mujoco"

    with pytest.raises(ValueError, match="not in SUPPORTED_ENGINES"):
        picker.set_engine("not_supported")

    # set_status updates the status pill text and style
    picker.set_status(EngineStatus.LIVE)
    assert picker.status_pill.text() == EngineStatus.LIVE.value
    assert picker.status_pill.styleSheet() == Styles.STATUS_SUCCESS_BOLD

    picker.set_status(EngineStatus.ERROR)
    assert picker.status_pill.text() == EngineStatus.ERROR.value
    assert picker.status_pill.styleSheet() == Styles.STATUS_ERROR_BOLD

    picker.set_status(EngineStatus.MOCK)
    assert picker.status_pill.text() == EngineStatus.MOCK.value
    assert picker.status_pill.styleSheet() == Styles.STATUS_WARNING

    with pytest.raises(TypeError, match="status must be an EngineStatus"):
        picker.set_status("mock")  # type: ignore[arg-type]


def test_engine_picker_signal_emission() -> None:
    picker = EnginePicker()
    emitted: list[str] = []
    picker.engine_selected.connect(emitted.append)

    picker.combo.setCurrentText("mujoco")
    assert emitted == ["mujoco"]
