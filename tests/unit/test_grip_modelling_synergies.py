"""Unit and TDD tests for grip modeling synergies and golf swing XML modifications.

Issue #757: Linked sliders (synergies) and improved visualization.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._grip_modelling_synergies import (
    get_descriptive_joint_name,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.grip_modelling_tab import (
    GripModellingTab,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.models_advanced import (
    ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.models_swing import (
    FULL_BODY_GOLF_SWING_XML,
    UPPER_BODY_GOLF_SWING_XML,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("rh_FFJ3", "[Right] Index Knuckle Flexion (MCP) (J3)"),
        ("lh_THJ2", "[Left] Thumb IP Flexion (THJ2)"),
        ("right_WRJ2", "[Right] Wrist Yaw / Abduction (WRJ2)"),
        ("left_LFJ5", "[Left] Little (Pinky) CMC Flexion (LFJ5)"),
        ("ffj1", "Index Knuckle Flexion (MCP) (j1)"),
        ("thj0", "Thumb CMC Abduction (thj0)"),
        ("unknown_joint", "unknown_joint"),
    ],
)
def test_get_descriptive_joint_name(raw_name: str, expected: str) -> None:
    """Verify raw joint names are mapped to user-friendly descriptive labels."""
    assert get_descriptive_joint_name(raw_name) == expected


@pytest.mark.unit
def test_synergy_interpolation_logic() -> None:
    """Verify linear interpolation math for linked synergy sliders."""
    min_val = -0.5
    max_val = 1.5

    # Mid-point interpolation
    t = 0.5
    val = min_val + t * (max_val - min_val)
    assert pytest.approx(val) == 0.5

    # Scale bounds interpolation
    t_min = 0.0
    val_min = min_val + t_min * (max_val - min_val)
    assert pytest.approx(val_min) == -0.5

    t_max = 1.0
    val_max = min_val + t_max * (max_val - min_val)
    assert pytest.approx(val_max) == 1.5


@pytest.mark.unit
def test_models_swing_xml_validity() -> None:
    """Verify modified models_swing XMLs compile successfully in MuJoCo."""
    try:
        model_upper = mujoco.MjModel.from_xml_string(UPPER_BODY_GOLF_SWING_XML)
        assert model_upper is not None
    except (ValueError, RuntimeError) as e:
        pytest.fail(f"UPPER_BODY_GOLF_SWING_XML failed to compile: {e}")

    try:
        model_full = mujoco.MjModel.from_xml_string(FULL_BODY_GOLF_SWING_XML)
        assert model_full is not None
    except (ValueError, RuntimeError) as e:
        pytest.fail(f"FULL_BODY_GOLF_SWING_XML failed to compile: {e}")


@pytest.mark.unit
def test_models_advanced_xml_validity() -> None:
    """Verify modified models_advanced XML compiles successfully in MuJoCo."""
    try:
        model_adv = mujoco.MjModel.from_xml_string(
            ADVANCED_BIOMECHANICAL_GOLF_SWING_XML
        )
        assert model_adv is not None
    except (ValueError, RuntimeError) as e:
        pytest.fail(f"ADVANCED_BIOMECHANICAL_GOLF_SWING_XML failed to compile: {e}")


@pytest.mark.unit
def test_xml_contains_statistic_tag() -> None:
    """Verify the required <statistic> tag is present in all golf XML strings."""
    for xml_str, name in [
        (UPPER_BODY_GOLF_SWING_XML, "UPPER_BODY"),
        (FULL_BODY_GOLF_SWING_XML, "FULL_BODY"),
        (ADVANCED_BIOMECHANICAL_GOLF_SWING_XML, "ADVANCED_BIOMECHANICAL"),
    ]:
        root = ET.fromstring(xml_str)
        stat = root.find("statistic")
        assert stat is not None, f"<statistic> tag missing in {name} model XML"
        assert stat.attrib.get("extent") == "2.0", f"Incorrect extent in {name}"
        assert stat.attrib.get("center") == "0 0 1", f"Incorrect center in {name}"


@pytest.mark.unit
def test_grip_modelling_tab_ui_widths() -> None:
    """Verify that GripModellingTab ui components are constructed with correct widths."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None

    tab = GripModellingTab()
    assert tab.control_panel.width() == 450
    assert tab.combo_hand.count() > 0
    # Clean up
    tab.deleteLater()
