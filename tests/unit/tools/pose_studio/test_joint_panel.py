from __future__ import annotations

import sys

import pytest
import numpy as np

if "PySide6" in sys.modules:
    pytest.skip(
        "PySide6 already loaded — PyQt6 DLLs unavailable", allow_module_level=True
    )

try:
    from PyQt6.QtWidgets import QApplication  # noqa: F401

    _HAVE_QT = True
except Exception:  # noqa: BLE001
    _HAVE_QT = False

if not _HAVE_QT:  # pragma: no cover - environment-dependent
    pytest.skip("PyQt6.QtWidgets unavailable", allow_module_level=True)

pytestmark = [pytest.mark.unit, pytest.mark.ui]

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.pose_studio.widgets.joint_panel import JointPanel
from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)


@pytest.fixture(scope="module", autouse=True)
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_joint_panel_initialization() -> None:
    panel = JointPanel()
    assert panel is not None
    assert len(panel._spinboxes) == len(REFERENCE_GOLFER_FIELDS)
    assert len(panel._sliders) == len(REFERENCE_GOLFER_FIELDS)


def test_joint_panel_set_angles() -> None:
    panel = JointPanel()
    angles = {
        REFERENCE_GOLFER_FIELDS[0]: 45.0,
        REFERENCE_GOLFER_FIELDS[1]: -30.0,
    }
    panel.set_angles(angles)

    assert panel._spinboxes[REFERENCE_GOLFER_FIELDS[0]].value() == pytest.approx(45.0)
    assert panel._sliders[REFERENCE_GOLFER_FIELDS[0]].value() == 450
    assert panel._spinboxes[REFERENCE_GOLFER_FIELDS[1]].value() == pytest.approx(-30.0)
    assert panel._sliders[REFERENCE_GOLFER_FIELDS[1]].value() == -300


def test_joint_panel_set_show_radians() -> None:
    panel = JointPanel()

    with pytest.raises(TypeError):
        panel.set_show_radians("true")  # type: ignore

    joint = REFERENCE_GOLFER_FIELDS[0]
    panel._sliders[joint].setValue(450)

    panel.set_show_radians(True)
    assert panel._show_radians is True
    assert panel._spinboxes[joint].suffix() == " rad"
    # Spinbox uses 4 decimal places in radian display mode.
    assert panel._spinboxes[joint].value() == pytest.approx(
        float(np.radians(45.0)), abs=1e-3
    )

    panel.set_show_radians(False)
    assert panel._show_radians is False
    assert panel._spinboxes[joint].suffix() == " deg"
    assert panel._spinboxes[joint].value() == pytest.approx(45.0)


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

    panel._on_slider_changed(joint, 450)

    assert panel._spinboxes[joint].value() == pytest.approx(45.0)


def test_joint_panel_on_spinbox_changed() -> None:
    panel = JointPanel()
    joint = REFERENCE_GOLFER_FIELDS[0]

    panel._on_spinbox_changed(joint, 45.0)

    assert panel._sliders[joint].value() == 450


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

    assert panel._sliders[joint].minimum() == -300
    assert panel._sliders[joint].maximum() == 300
    assert panel._spinboxes[joint].minimum() == pytest.approx(-30.0)
    assert panel._spinboxes[joint].maximum() == pytest.approx(30.0)


def test_joint_panel_set_limits_defaults_unlisted_joints() -> None:
    panel = JointPanel()
    other_joint = REFERENCE_GOLFER_FIELDS[1]

    panel.set_limits({REFERENCE_GOLFER_FIELDS[0]: (-30.0, 30.0)})

    assert panel._sliders[other_joint].minimum() == -1800
    assert panel._sliders[other_joint].maximum() == 1800


def test_joint_panel_set_limits_does_not_read_or_emit() -> None:
    """set_limits only re-ranges min/max and must not emit angle_edited."""
    panel = JointPanel()
    joint = REFERENCE_GOLFER_FIELDS[0]

    seen: list[tuple[str, float]] = []
    panel.angle_edited.connect(lambda name, value: seen.append((name, value)))

    panel.set_limits({joint: (-30.0, 30.0)})

    assert seen == []


def test_joint_panel_set_error_applies_and_clears_border() -> None:
    panel = JointPanel()
    joint = REFERENCE_GOLFER_FIELDS[0]

    panel.set_error(joint, True)
    assert "border" in panel._spinboxes[joint].styleSheet()

    panel.set_error(joint, False)
    assert panel._spinboxes[joint].styleSheet() == ""


def test_joint_panel_set_error_rejects_unknown_joint() -> None:
    panel = JointPanel()
    with pytest.raises(KeyError):
        panel.set_error("not-a-joint", True)
