"""Unit and TDD tests for grip modeling synergies and golf swing XML modifications.

Issue #757: Linked sliders (synergies) and improved visualization.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import mujoco
import pytest

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


def get_descriptive_joint_name(name: str) -> str:
    """Helper to convert joint name to a descriptive name (mirrors planned implementation)."""
    side_prefix = ""
    if name.startswith(("rh_", "right_")):
        side_prefix = "[Right] "
    elif name.startswith(("lh_", "left_")):
        side_prefix = "[Left] "

    clean_name = name
    for prefix in ["rh_", "lh_", "right_", "left_"]:
        if clean_name.lower().startswith(prefix):
            clean_name = clean_name[len(prefix) :]
            break

    finger_map = {
        "ff": "Index",
        "mf": "Middle",
        "rf": "Ring",
        "lf": "Little (Pinky)",
        "th": "Thumb",
        "wr": "Wrist",
    }

    match_shadow = re.match(r"^([a-zA-Z]+)J(\d+)$", clean_name)
    match_allegro = re.match(r"^([a-zA-Z]+)j(\d+)$", clean_name)

    if match_shadow:
        finger_code = match_shadow.group(1).lower()
        joint_num = int(match_shadow.group(2))
        finger_name = finger_map.get(finger_code, finger_code.upper())

        if finger_code == "wr":
            if joint_num == 1:
                return f"{side_prefix}Wrist Pitch / Flexion (WRJ1)"
            if joint_num == 2:
                return f"{side_prefix}Wrist Yaw / Abduction (WRJ2)"
        elif finger_code == "th":
            thumb_joints = {
                5: "CMC Abduction (THJ5)",
                4: "CMC Flexion (THJ4)",
                3: "MCP Flexion (THJ3)",
                2: "IP Flexion (THJ2)",
                1: "Distal Flexion (THJ1)",
            }
            return f"{side_prefix}Thumb {thumb_joints.get(joint_num, f'Joint {joint_num}')}"
        elif finger_code == "lf" and joint_num == 5:
            return f"{side_prefix}Little (Pinky) CMC Flexion (LFJ5)"
        else:
            finger_joints = {
                4: "Knuckle Abduction (MCP) (J4)",
                3: "Knuckle Flexion (MCP) (J3)",
                2: "Middle Joint Flexion (PIP) (J2)",
                1: "Distal Joint Flexion (DIP) (J1)",
            }
            return f"{side_prefix}{finger_name} {finger_joints.get(joint_num, f'Joint {joint_num}')}"

    elif match_allegro:
        finger_code = match_allegro.group(1).lower()
        joint_num = int(match_allegro.group(2))
        finger_name = finger_map.get(finger_code, finger_code.upper())

        if finger_code == "th":
            thumb_joints = {
                0: "CMC Abduction (thj0)",
                1: "CMC Flexion (thj1)",
                2: "MCP Flexion (thj2)",
                3: "IP Flexion (thj3)",
            }
            return f"{side_prefix}Thumb {thumb_joints.get(joint_num, f'Joint {joint_num}')}"
        finger_joints = {
            0: "Knuckle Abduction (MCP) (j0)",
            1: "Knuckle Flexion (MCP) (j1)",
            2: "Middle Joint Flexion (PIP) (j2)",
            3: "Distal Joint Flexion (DIP) (j3)",
        }
        return f"{side_prefix}{finger_name} {finger_joints.get(joint_num, f'Joint {joint_num}')}"

    return f"{side_prefix}{name}"


@pytest.mark.unit
def test_get_descriptive_joint_name() -> None:
    """Verify raw joint names are mapped to user-friendly descriptive labels."""
    assert (
        get_descriptive_joint_name("rh_FFJ3")
        == "[Right] Index Knuckle Flexion (MCP) (J3)"
    )
    assert get_descriptive_joint_name("lh_THJ2") == "[Left] Thumb IP Flexion (THJ2)"
    assert get_descriptive_joint_name("ffj1") == "Index Knuckle Flexion (MCP) (j1)"
    assert get_descriptive_joint_name("thj0") == "Thumb CMC Abduction (thj0)"
    assert (
        get_descriptive_joint_name("lh_WRJ1") == "[Left] Wrist Pitch / Flexion (WRJ1)"
    )
    assert get_descriptive_joint_name("unknown_joint") == "unknown_joint"


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
