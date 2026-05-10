from __future__ import annotations

import pytest
import numpy as np

from src.tools.pose_studio.widgets.joint_panel import JointPanel
from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)


def test_joint_panel_initialization() -> None:
    panel = JointPanel()
    assert panel is not None
    assert len(panel._spinboxes) == len(REFERENCE_GOLFER_FIELDS)
    assert len(panel._sliders) == len(REFERENCE_GOLFER_FIELDS)


def test_joint_panel_set_angles() -> None:
    panel = JointPanel()
    # Test setting a couple of specific angles
    angles = {
        REFERENCE_GOLFER_FIELDS[0]: 45.0,
        REFERENCE_GOLFER_FIELDS[1]: -30.0,
    }
    panel.set_angles(angles)

    # Check that spinbox values were set
    # The mock will receive a setValue call
    panel._spinboxes[REFERENCE_GOLFER_FIELDS[0]].setValue.assert_called_with(45.0)  # type: ignore
    panel._sliders[REFERENCE_GOLFER_FIELDS[0]].setValue.assert_called_with(450)  # type: ignore

    panel._spinboxes[REFERENCE_GOLFER_FIELDS[1]].setValue.assert_called_with(-30.0)  # type: ignore
    panel._sliders[REFERENCE_GOLFER_FIELDS[1]].setValue.assert_called_with(-300)  # type: ignore


def test_joint_panel_set_show_radians() -> None:
    panel = JointPanel()

    # Ensure invalid type raises TypeError
    with pytest.raises(TypeError):
        panel.set_show_radians("true")  # type: ignore

    # Toggle to radians
    # We mock the slider to return a specific value so we can check the math
    joint = REFERENCE_GOLFER_FIELDS[0]
    for slider in panel._sliders.values():
        slider.value.return_value = 0  # type: ignore
    panel._sliders[joint].value.return_value = 450  # type: ignore

    panel.set_show_radians(True)
    assert panel._show_radians is True

    # Should have called setSuffix with " rad"
    panel._spinboxes[joint].setSuffix.assert_called_with(" rad")  # type: ignore

    # Should have set the value to radians(45)
    expected_rad = float(np.radians(45.0))
    panel._spinboxes[joint].setValue.assert_called_with(expected_rad)  # type: ignore

    # Toggle back to degrees
    panel._sliders[joint].value.return_value = 450  # type: ignore
    panel.set_show_radians(False)
    assert panel._show_radians is False

    # Should have called setSuffix with " deg"
    panel._spinboxes[joint].setSuffix.assert_called_with(" deg")  # type: ignore

    # Should have set the value to 45.0
    panel._spinboxes[joint].setValue.assert_called_with(45.0)  # type: ignore


def test_joint_panel_joint_widgets() -> None:
    panel = JointPanel()
    widgets = panel.joint_widgets()

    for name in REFERENCE_GOLFER_FIELDS:
        assert f"{name}__spin" in widgets
        assert f"{name}__slider" in widgets

    assert len(widgets) == len(REFERENCE_GOLFER_FIELDS) * 2


def test_joint_panel_on_slider_changed() -> None:
    panel = JointPanel()
    joint = REFERENCE_GOLFER_FIELDS[0]

    # Trigger slider change to 450 (45.0 degrees)
    panel._on_slider_changed(joint, 450)

    panel._spinboxes[joint].setValue.assert_called_with(45.0)  # type: ignore


def test_joint_panel_on_spinbox_changed() -> None:
    panel = JointPanel()
    joint = REFERENCE_GOLFER_FIELDS[0]

    # Trigger spinbox change to 45.0 degrees
    panel._on_spinbox_changed(joint, 45.0)

    panel._sliders[joint].setValue.assert_called_with(450)  # type: ignore
