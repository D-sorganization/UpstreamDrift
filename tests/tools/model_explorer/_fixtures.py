"""Small URDF fixtures used by model_explorer tests."""

from __future__ import annotations

SIMPLE_URDF = """<?xml version="1.0"?>
<robot name="simple">
    <material name="red">
        <color rgba="1 0 0 1"/>
    </material>
    <link name="base">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
        </inertial>
        <visual>
            <geometry><box size="0.1 0.1 0.1"/></geometry>
        </visual>
    </link>
    <link name="arm">
        <inertial>
            <mass value="0.5"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
    <link name="hand"/>
    <joint name="base_to_arm" type="revolute">
        <parent link="base"/>
        <child link="arm"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.0" upper="1.0" effort="10" velocity="1"/>
    </joint>
    <joint name="arm_to_hand" type="fixed">
        <parent link="arm"/>
        <child link="hand"/>
    </joint>
</robot>
"""

# Branching URDF with two end effectors
BRANCH_URDF = """<?xml version="1.0"?>
<robot name="branchy">
    <link name="root"/>
    <link name="L1"/>
    <link name="L2"/>
    <link name="left_hand"/>
    <link name="right_gripper"/>
    <joint name="j1" type="fixed">
        <parent link="root"/>
        <child link="L1"/>
    </joint>
    <joint name="j2" type="fixed">
        <parent link="root"/>
        <child link="L2"/>
    </joint>
    <joint name="j3" type="fixed">
        <parent link="L1"/>
        <child link="left_hand"/>
    </joint>
    <joint name="j4" type="fixed">
        <parent link="L2"/>
        <child link="right_gripper"/>
    </joint>
</robot>
"""

# URDF with end-effector style subtree under "wrist"
EE_URDF = """<?xml version="1.0"?>
<robot name="ee_test">
    <link name="base"/>
    <link name="wrist"/>
    <link name="finger_a"/>
    <link name="finger_b"/>
    <joint name="base_wrist" type="revolute">
        <parent link="base"/>
        <child link="wrist"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/>
    </joint>
    <joint name="wrist_a" type="prismatic">
        <parent link="wrist"/>
        <child link="finger_a"/>
        <axis xyz="1 0 0"/>
        <limit lower="-0.05" upper="0.05" effort="1" velocity="1"/>
    </joint>
    <joint name="wrist_b" type="prismatic">
        <parent link="wrist"/>
        <child link="finger_b"/>
        <axis xyz="-1 0 0"/>
        <limit lower="-0.05" upper="0.05" effort="1" velocity="1"/>
    </joint>
</robot>
"""


def make_segment(
    name: str,
    parent: str | None = None,
    shape: str = "Box",
    mass: float = 1.0,
) -> dict:
    """Build a minimal valid segment dict for URDFBuilder."""
    seg: dict = {
        "name": name,
        "geometry": {
            "shape": shape,
            "dimensions": {"length": 0.5, "width": 0.1, "height": 0.1},
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        },
        "physics": {
            "mass": mass,
            "inertia": {"ixx": 0.01, "iyy": 0.01, "izz": 0.01},
            "material": {
                "name": f"mat_{name}",
                "color": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0},
            },
        },
        "joint": {
            "type": "revolute",
            "axis": {"x": 0, "y": 0, "z": 1},
            "limits": {"lower": -90, "upper": 90, "velocity": 1.0, "effort": 10.0},
        },
    }
    if parent is not None:
        seg["parent"] = parent
    return seg
