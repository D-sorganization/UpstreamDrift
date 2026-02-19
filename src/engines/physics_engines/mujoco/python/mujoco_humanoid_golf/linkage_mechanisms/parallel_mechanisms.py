# ruff: noqa: E501
"""Parallel mechanism generators for MuJoCo.

Includes Delta robot, 5-bar parallel manipulator, Stewart platform,
and pantograph mechanisms.

Extracted from __init__.py for SRP (#1485).
"""

import numpy as np

from src.shared.python.core.constants import GRAVITY_M_S2


def generate_pantograph_xml(scale_factor: float = 2.0) -> str:
    """Generate a pantograph mechanism (geometric scaling/copying).

    Parameters
    ----------
    scale_factor : float
        Scaling factor for the output point relative to input
    """
    return f"""
<mujoco model="pantograph">
    <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="1 1"/>
        <material name="arm_mat" rgba="0.1 0.7 0.9 1"/>
        <material name="input_mat" rgba="0.9 0.7 0.1 1"/>
        <material name="output_mat" rgba="0.9 0.1 0.1 1"/>
    </asset>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1" material="grid"/>

        <!-- Fixed pivot -->
        <geom name="fixed_pivot" type="sphere" pos="0 0 0.5" size="0.12"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

        <!-- First arm (input side) -->
        <body name="arm1" pos="0 0 0.5">
            <joint name="arm1_joint" type="hinge" axis="0 0 1" damping="0.2"/>
            <geom type="capsule" fromto="0 0 0  1.5 1.0 0" size="0.05"
                  material="arm_mat" mass="0.5"/>
            <geom type="sphere" pos="1.5 1.0 0" size="0.09" rgba="0.2 0.8 0.8 1"
                  contype="0" conaffinity="0"/>

            <!-- Input point -->
            <body name="input_point" pos="0.75 0.5 0">
                <geom type="sphere" size="0.12" material="input_mat" mass="0.3"/>
                <geom type="cylinder" fromto="0 0 0  0 0 0.6" size="0.025"
                      rgba="0.9 0.7 0.1 0.6" contype="0" conaffinity="0"/>
            </body>

            <!-- Second arm segment -->
            <body name="arm2" pos="1.5 1.0 0">
                <joint name="arm2_joint" type="hinge" axis="0 0 1" damping="0.15"/>
                <geom type="capsule" fromto="0 0 0  {1.5 * scale_factor} {1.0 * scale_factor} 0"
                      size="0.05" material="arm_mat" mass="0.5"/>

                <!-- Output point (scaled) -->
                <body name="output_point" pos="{0.75 * scale_factor} {0.5 * scale_factor} 0">
                    <geom type="sphere" size="0.15" material="output_mat" mass="0.4"/>
                    <geom type="cylinder" fromto="0 0 0  0 0 0.8" size="0.03"
                          rgba="0.9 0.1 0.1 0.6" contype="0" conaffinity="0"/>
                </body>
            </body>
        </body>

        <!-- Parallel arm structure -->
        <body name="arm3" pos="0 0 0.5">
            <joint name="arm3_joint" type="hinge" axis="0 0 1" damping="0.2"/>
            <geom type="capsule" fromto="0 0 0.05  1.5 1.0 0.05" size="0.045"
                  rgba="0.1 0.6 0.8 0.8" mass="0.5"/>
        </body>
    </worldbody>

    <equality>
        <!-- Maintain parallelogram structure -->
        <joint joint1="arm1_joint" joint2="arm3_joint" polycoef="0 1 0 0 0"/>
    </equality>

    <actuator>
        <motor name="arm1_motor" joint="arm1_joint" gear="15" ctrllimited="true" ctrlrange="-5 5"/>
    </actuator>
</mujoco>
"""


def _delta_robot_arm_xml(
    arm_num: int,
    base_radius: float,
    angle_deg: float,
    arm_length: float,
    forearm_length: float,
) -> str:
    angle_rad = np.radians(angle_deg)
    bx = base_radius * np.cos(angle_rad)
    by = base_radius * np.sin(angle_rad)
    euler_attr = f' euler="0 0 {angle_deg}"' if angle_deg != 0 else ""
    return f"""
        <!-- Arm {arm_num} ({angle_deg} degrees) -->
        <body name="base{arm_num}" pos="{bx} {by} 2">
            <geom type="sphere" size="0.15" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
            <body name="arm{arm_num}" pos="0 0 0"{euler_attr}>
                <joint name="joint{arm_num}" type="hinge" axis="0 1 0" range="-120 120" damping="1.0"/>
                <geom type="capsule" fromto="0 0 0  0 0 {-arm_length}" size="0.08"
                      material="arm_mat" mass="1.0"/>

                <body name="forearm{arm_num}a" pos="0 {-0.15} {-arm_length}">
                    <joint name="elbow{arm_num}a" type="hinge" axis="0 1 0" damping="0.5"/>
                    <geom type="capsule" fromto="0 0 0  0 0 {-forearm_length}" size="0.05"
                          material="forearm_mat" mass="0.5"/>
                </body>
                <body name="forearm{arm_num}b" pos="0 {0.15} {-arm_length}">
                    <joint name="elbow{arm_num}b" type="hinge" axis="0 1 0" damping="0.5"/>
                    <geom type="capsule" fromto="0 0 0  0 0 {-forearm_length}" size="0.05"
                          material="forearm_mat" mass="0.5"/>
                </body>
            </body>
        </body>"""


def _delta_robot_equality_xml(forearm_length: float) -> str:
    lines = [
        "    <equality>",
        "        <!-- Connect forearms to platform -->",
    ]
    for arm_num in range(1, 4):
        lines.append(
            f'        <connect body1="forearm{arm_num}a" body2="platform" anchor="0 0 {-forearm_length}"/>'
        )
        lines.append(
            f'        <connect body1="forearm{arm_num}b" body2="platform" anchor="0 0 {-forearm_length}"/>'
        )
    lines.append("")
    lines.append("        <!-- Keep forearm pairs parallel -->")
    for arm_num in range(1, 4):
        lines.append(
            f'        <joint joint1="elbow{arm_num}a" joint2="elbow{arm_num}b" polycoef="0 1 0 0 0"/>'
        )
    lines.append("    </equality>")
    return "\n".join(lines)


def generate_delta_robot_xml(
    base_radius: float = 2.0, platform_radius: float = 0.5
) -> str:
    """Generate a 3-DOF Delta parallel robot (high-speed pick-and-place).

    Parameters
    ----------
    base_radius : float
        Radius of the fixed base triangle
    platform_radius : float
        Radius of the moving platform triangle
    """
    arm_length = 2.0
    forearm_length = 3.0

    arms_xml = ""
    for arm_num, angle_deg in enumerate([0, 120, 240], start=1):
        arms_xml += _delta_robot_arm_xml(
            arm_num, base_radius, angle_deg, arm_length, forearm_length
        )

    equality_xml = _delta_robot_equality_xml(forearm_length)

    return f"""
<mujoco model="delta_robot">
    <option timestep="0.001" gravity="0 0 -{GRAVITY_M_S2}"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="1 1"/>
        <material name="base_mat" rgba="0.3 0.3 0.3 1"/>
        <material name="arm_mat" rgba="0.9 0.1 0.1 1"/>
        <material name="forearm_mat" rgba="0.1 0.7 0.1 1"/>
        <material name="platform_mat" rgba="0.1 0.1 0.9 1"/>
    </asset>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1" material="grid"/>

        <!-- Base frame -->
        <geom type="cylinder" pos="0 0 2" size="{base_radius} 0.2"
              material="base_mat" contype="0" conaffinity="0"/>
{arms_xml}

        <!-- End effector platform -->
        <body name="platform" pos="0 0 {2 - arm_length - forearm_length}">
            <freejoint/>
            <geom type="cylinder" size="{platform_radius} 0.1" material="platform_mat" mass="0.5"/>
            <geom type="sphere" pos="0 0 -0.3" size="0.2" rgba="1 0.5 0 1" mass="0.3"/>
        </body>
    </worldbody>

{equality_xml}

    <actuator>
        <motor name="motor1" joint="joint1" gear="50" ctrllimited="true" ctrlrange="-10 10"/>
        <motor name="motor2" joint="joint2" gear="50" ctrllimited="true" ctrlrange="-10 10"/>
        <motor name="motor3" joint="joint3" gear="50" ctrllimited="true" ctrlrange="-10 10"/>
    </actuator>
</mujoco>
"""


def generate_five_bar_parallel_xml(link_length: float = 1.5) -> str:
    """Generate a 5-bar parallel manipulator (2-DOF planar robot).

    Parameters
    ----------
    link_length : float
        Length of the links
    """
    base_width = 2.0

    return f"""
<mujoco model="five_bar_parallel">
    <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="1 1"/>
        <material name="link1_mat" rgba="0.9 0.1 0.1 1"/>
        <material name="link2_mat" rgba="0.1 0.7 0.1 1"/>
        <material name="effector_mat" rgba="1 0.5 0 1"/>
    </asset>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1" material="grid"/>

        <!-- Base -->
        <geom type="box" pos="0 0 0.25" size="{base_width / 2} 0.15 0.25"
              rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>

        <!-- Left base pivot -->
        <geom name="left_pivot" type="sphere" pos="{-base_width / 2} 0 0.5" size="0.12"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

        <!-- Right base pivot -->
        <geom name="right_pivot" type="sphere" pos="{base_width / 2} 0 0.5" size="0.12"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

        <!-- Left arm chain -->
        <body name="left_link1" pos="{-base_width / 2} 0 0.5">
            <joint name="left_joint1" type="hinge" axis="0 0 1" damping="0.5"/>
            <geom type="capsule" fromto="0 0 0  {link_length} 0 0" size="0.06"
                  material="link1_mat" mass="0.8"/>
            <geom type="sphere" pos="{link_length} 0 0" size="0.09" rgba="1 0.5 0.5 1"
                  contype="0" conaffinity="0"/>

            <body name="left_link2" pos="{link_length} 0 0">
                <joint name="left_joint2" type="hinge" axis="0 0 1" damping="0.3"/>
                <geom type="capsule" fromto="0 0 0  {link_length} 0 0" size="0.05"
                      material="link2_mat" mass="0.6"/>
            </body>
        </body>

        <!-- Right arm chain -->
        <body name="right_link1" pos="{base_width / 2} 0 0.5">
            <joint name="right_joint1" type="hinge" axis="0 0 1" damping="0.5"/>
            <geom type="capsule" fromto="0 0 0  {-link_length} 0 0" size="0.06"
                  material="link1_mat" mass="0.8"/>
            <geom type="sphere" pos="{-link_length} 0 0" size="0.09" rgba="1 0.5 0.5 1"
                  contype="0" conaffinity="0"/>

            <body name="right_link2" pos="{-link_length} 0 0">
                <joint name="right_joint2" type="hinge" axis="0 0 1" damping="0.3"/>
                <geom type="capsule" fromto="0 0 0  {-link_length} 0 0" size="0.05"
                      material="link2_mat" mass="0.6"/>
            </body>
        </body>

        <!-- End effector -->
        <body name="end_effector" pos="0 0 0.5">
            <freejoint/>
            <geom type="sphere" size="0.15" material="effector_mat" mass="0.5"/>
            <geom type="cylinder" fromto="0 0 0  0 0 -0.5" size="0.03"
                  rgba="1 0.5 0 0.7" contype="0" conaffinity="0"/>
        </body>
    </worldbody>

    <equality>
        <!-- Connect chains to end effector -->
        <connect body1="left_link2" body2="end_effector" anchor="{link_length} 0 0"/>
        <connect body1="right_link2" body2="end_effector" anchor="{-link_length} 0 0"/>
    </equality>

    <actuator>
        <motor name="left_motor" joint="left_joint1" gear="25" ctrllimited="true" ctrlrange="-8 8"/>
        <motor name="right_motor" joint="right_joint1" gear="25" ctrllimited="true" ctrlrange="-8 8"/>
    </actuator>
</mujoco>
"""


def _stewart_base_attachment_xml(base_radius: float) -> str:
    """Generate XML for the 6 base attachment point spheres."""
    angles = [0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3]
    lines = []
    for angle in angles:
        x = base_radius * np.cos(angle)
        y = base_radius * np.sin(angle)
        lines.append(
            f'            <geom type="sphere" pos="{x} {y} 0.15"'
            f'\n                  size="0.1" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>'
        )
    return "\n".join(lines)


def _stewart_leg_xml(
    leg_num: int, base_radius: float, angle: float, leg_min: float, leg_max: float
) -> str:
    """Generate XML for a single Stewart platform leg (lower + upper)."""
    x = base_radius * np.cos(angle)
    y = base_radius * np.sin(angle)
    return f"""
            <!-- Leg {leg_num} -->
            <body name="leg{leg_num}_lower" pos="{x} {y} 0.15">
                <joint name="leg{leg_num}_base_ball" type="ball" damping="0.5"/>
                <geom type="capsule" fromto="0 0 0  0 0 1.2" size="0.05" material="leg_mat" mass="0.5"/>

                <body name="leg{leg_num}_upper" pos="0 0 1.2">
                    <joint name="leg{leg_num}_extend" type="slide" axis="0 0 1" range="{leg_min - 1.2} {leg_max - 1.2}" damping="2.0"/>
                    <geom type="capsule" fromto="0 0 0  0 0 1.0" size="0.045"
                          rgba="0.1 0.6 0.8 1" mass="0.4"/>
                </body>
            </body>"""


def _stewart_platform_attachment_xml(platform_radius: float) -> str:
    """Generate XML for the 6 platform attachment point spheres."""
    angles = [
        np.pi / 6,
        np.pi / 2,
        5 * np.pi / 6,
        7 * np.pi / 6,
        3 * np.pi / 2,
        11 * np.pi / 6,
    ]
    lines = []
    for angle in angles:
        x = platform_radius * np.cos(angle)
        y = platform_radius * np.sin(angle)
        lines.append(
            f'            <geom type="sphere"'
            f'\n                  pos="{x} {y} -0.12"'
            f'\n                  size="0.08" rgba="0.8 0.6 0.1 1" contype="0" conaffinity="0"/>'
        )
    return "\n".join(lines)


def _stewart_equality_xml(platform_radius: float) -> str:
    """Generate XML for leg-to-platform connect constraints."""
    angles = [
        np.pi / 6,
        np.pi / 2,
        5 * np.pi / 6,
        7 * np.pi / 6,
        3 * np.pi / 2,
        11 * np.pi / 6,
    ]
    lines = []
    for i, angle in enumerate(angles, start=1):
        x = platform_radius * np.cos(angle)
        y = platform_radius * np.sin(angle)
        lines.append(
            f'        <connect body1="leg{i}_upper" body2="platform" anchor="0 0 1.0"'
            f'\n                 relpose="{x}'
            f'\n                          {y} -0.12  1 0 0 0"/>'
        )
    return "\n".join(lines)


def _stewart_actuator_xml() -> str:
    """Generate XML for the 6 leg motor actuators."""
    lines = []
    for i in range(1, 7):
        lines.append(
            f'        <motor name="leg{i}_motor" joint="leg{i}_extend" gear="100"'
            f'\n               ctrllimited="true" ctrlrange="-15 15"/>'
        )
    return "\n".join(lines)


def generate_stewart_platform_xml(
    base_radius: float = 1.5, platform_radius: float = 0.8
) -> str:
    """Generate a Stewart platform (6-DOF parallel manipulator).

    Parameters
    ----------
    base_radius : float
        Radius of the base hexagon
    platform_radius : float
        Radius of the platform hexagon
    """
    leg_min = 1.5
    leg_max = 3.0

    base_attachments = _stewart_base_attachment_xml(base_radius)

    leg_angles = [0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3]
    legs_xml = ""
    for i, angle in enumerate(leg_angles, start=1):
        legs_xml += _stewart_leg_xml(i, base_radius, angle, leg_min, leg_max)

    platform_attachments = _stewart_platform_attachment_xml(platform_radius)
    equality_xml = _stewart_equality_xml(platform_radius)
    actuator_xml = _stewart_actuator_xml()

    return f"""
<mujoco model="stewart_platform">
    <option timestep="0.001" gravity="0 0 -{GRAVITY_M_S2}"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="1 1"/>
        <material name="base_mat" rgba="0.3 0.3 0.3 1"/>
        <material name="leg_mat" rgba="0.1 0.7 0.9 1"/>
        <material name="platform_mat" rgba="0.9 0.7 0.1 1"/>
    </asset>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1" material="grid"/>

        <!-- Base platform -->
        <body name="base" pos="0 0 0.2">
            <geom type="cylinder" size="{base_radius} 0.15" material="base_mat" mass="5.0"/>

            <!-- Base attachment points (6 around hexagon) -->
{base_attachments}
{legs_xml}
        </body>

        <!-- Moving platform -->
        <body name="platform" pos="0 0 2.5">
            <freejoint/>
            <geom type="cylinder" size="{platform_radius} 0.12" material="platform_mat" mass="2.0"/>
            <geom type="sphere" pos="0 0 0.2" size="0.15" rgba="1 0.5 0 1" mass="0.5"/>

            <!-- Platform attachment points -->
{platform_attachments}
        </body>
    </worldbody>

    <equality>
        <!-- Connect leg tops to platform -->
{equality_xml}
    </equality>

    <actuator>
{actuator_xml}
    </actuator>
</mujoco>
"""
