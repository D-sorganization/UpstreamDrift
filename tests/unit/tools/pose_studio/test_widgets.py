from __future__ import annotations

import pytest

from src.tools.pose_studio.widgets.engine_picker import EnginePicker
from src.tools.pose_studio.widgets.units_badge import UnitsBadge
from src.tools.pose_studio.core import EngineStatus


def test_units_badge_set_engine() -> None:
    badge = UnitsBadge()
    badge.set_engine("drake")
    badge.setText.assert_called_with("Drake URDF / RPY (rad)")  # type: ignore

    # We can check that the layout or internal state reflects the new engine
    # In a full UI test, we would check the displayed text, but the mock
    # prevents deep Qt widget introspection easily.


def test_engine_picker_initialization() -> None:
    picker = EnginePicker()
    # It sets the combo box to the default "drake"
    picker.combo.setCurrentText.assert_called_with("drake")  # type: ignore

    # current_engine delegates to combo.currentText()
    picker.combo.currentText.return_value = "drake"  # type: ignore
    assert picker.current_engine() == "drake"
