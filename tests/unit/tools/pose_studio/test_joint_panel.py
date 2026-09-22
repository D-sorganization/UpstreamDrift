from __future__ import annotations

import pytest
import numpy as np

pytestmark = [pytest.mark.unit, pytest.mark.ui]

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


def test_joint_panel_set_limits_rejects_non_mapping() -> None:
    panel = JointPanel()
    with pytest.raises(TypeError):
        panel.set_limits([("a", (0.0, 1.0))])  # type: ignore


def test_joint_panel_set_limits_rejects_inverted_range() -> None:
    panel = JointPanel()
    joint = REFERENCE_GOLFER_FIELDS[0]
    with pytest.raises(ValueError):
        panel.set_limits({joint: (10.0, -10.0)})


def test_joint_panel_set_limits_re_ranges_slider_and_spinbox() -> None:
    panel = JointPanel()
    joint = REFERENCE_GOLFER_FIELDS[0]

    panel.set_limits({joint: (-30.0, 30.0)})

    panel._sliders[joint].setMinimum.assert_called_with(-300)  # type: ignore
    panel._sliders[joint].setMaximum.assert_called_with(300)  # type: ignore
    panel._spinboxes[joint].setMinimum.assert_called_with(-30.0)  # type: ignore
    panel._spinboxes[joint].setMaximum.assert_called_with(30.0)  # type: ignore


def test_joint_panel_set_limits_defaults_unlisted_joints() -> None:
    panel = JointPanel()
    other_joint = REFERENCE_GOLFER_FIELDS[1]

    panel.set_limits({REFERENCE_GOLFER_FIELDS[0]: (-30.0, 30.0)})

    panel._sliders[other_joint].setMinimum.assert_called_with(-1800)  # type: ignore
    panel._sliders[other_joint].setMaximum.assert_called_with(1800)  # type: ignore


def test_joint_panel_set_limits_does_not_read_or_emit() -> None:
    """set_limits only re-ranges min/max; Qt clamps the value itself, so
    this must not read slider.value() (real widgets return an int; the
    unit-test double does not) or emit angle_edited."""
    panel = JointPanel()
    joint = REFERENCE_GOLFER_FIELDS[0]

    seen: list[tuple[str, float]] = []
    panel.angle_edited.connect(lambda name, value: seen.append((name, value)))

    panel.set_limits({joint: (-30.0, 30.0)})

    panel._sliders[joint].value.assert_not_called()  # type: ignore
    assert seen == []


def test_joint_panel_set_error_applies_and_clears_border() -> None:
    panel = JointPanel()
    joint = REFERENCE_GOLFER_FIELDS[0]

    panel.set_error(joint, True)
    style_call = panel._spinboxes[joint].setStyleSheet.call_args  # type: ignore
    assert "border" in style_call.args[0]

    panel.set_error(joint, False)
    panel._spinboxes[joint].setStyleSheet.assert_called_with("")  # type: ignore


def test_joint_panel_set_error_rejects_unknown_joint() -> None:
    panel = JointPanel()
    with pytest.raises(KeyError):
        panel.set_error("not-a-joint", True)
