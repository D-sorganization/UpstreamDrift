"""MJCF models for golf swing systems.

Demo/utility models and club XML generation live here.
Pendulum models are in pendulum_models_xml.py.
Golf swing models are in golf_swing_models_xml.py.
"""

from __future__ import annotations

from typing import Any, cast

from src.shared.python.core.constants import (
    DEFAULT_TIME_STEP,
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
)

# Re-export pendulum models for backward compatibility
from .golf_swing_models_xml import (  # noqa: F401
    ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
    CLUB_CONFIGS,
    FULL_BODY_GOLF_SWING_XML,
    UPPER_BODY_GOLF_SWING_XML,
)
from .pendulum_models_xml import (  # noqa: F401
    CHAOTIC_PENDULUM_XML,
    DOUBLE_PENDULUM_XML,
    TRIPLE_PENDULUM_XML,
)

# Convert to float for use in f-strings
_BALL_MASS = float(GOLF_BALL_MASS_KG)
_BALL_RADIUS = float(GOLF_BALL_RADIUS_M)
_BALL_RADIUS_INNER = _BALL_RADIUS * 0.998  # For dimple visualization
_TIME_STEP = float(DEFAULT_TIME_STEP)


# ==============================================================================
# TWO-LINK INCLINED PLANE MODEL WITH UNIVERSAL JOINT AT WRIST
# ==============================================================================
TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML = rf"""
<mujoco model="two_link_inclined_universal">
  <option timestep="0.001" gravity="0 0 -{GRAVITY_M_S2}"
          integrator="RK4" solver="Newton"/>

  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>

  <visual>
    <global offwidth="1024" offheight="1024"/>
    <map znear="0.01" zfar="20"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <material name="arm_mat" rgba="0.8 0.3 0.2 1"/>
    <material name="club_mat" rgba="0.2 0.8 0.3 1"/>
    <material name="incline_mat" rgba="0.6 0.6 0.7 0.5"/>
    <material name="ground_mat" rgba="0.4 0.6 0.3 1"/>
  </asset>

  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="5 5 0.1" material="ground_mat"/>

    <!-- Inclined reference plane (visual only, 15 degrees) -->
    <geom name="incline_ref" type="box" size="2 2 0.01"
          pos="0 0 0.8" euler="0.262 0 0"
          material="incline_mat" contype="0" conaffinity="0"/>

    <!-- Cameras -->
    <camera name="side" pos="-3 -1 1.5" euler="0.15 0 0.3"/>
    <camera name="front" pos="0 -3 1.5" euler="0.15 0 1.57"/>
    <camera name="incline_view" pos="-2 -2 2" euler="0.4 0 0.785"/>

    <!-- Shoulder/base on inclined orientation -->
    <body name="shoulder_base" pos="0 0 1.2" euler="0.262 0 0">
      <!-- Shoulder hinge (rotation about incline normal) -->
      <joint name="shoulder" type="hinge" axis="0 0 1" limited="false"
             damping="1.0" armature="0.02"/>

      <geom name="shoulder_marker" type="sphere" size="0.04" rgba="1 0 0 0.8"/>
      <site name="shoulder_site" pos="0 0 0" size="0.01"/>

      <!-- Upper arm -->
      <body name="upper_arm" pos="0 0 0">
        <geom name="upper_arm_geom" type="capsule"
              fromto="0 0 0 0.5 0 0" size="0.03"
              material="arm_mat" mass="1.5"/>

        <!-- Wrist with UNIVERSAL JOINT (2 perpendicular hinges) -->
        <body name="wrist_body" pos="0.5 0 0">
          <!-- Universal joint axis 1: flex/extend in local y-z plane -->
          <joint name="wrist_universal_1" type="hinge" axis="0 1 0"
                 range="-1.57 1.57" damping="0.5" armature="0.01"/>

          <!-- Universal joint axis 2: lateral deviation perpendicular to axis 1 -->
          <joint name="wrist_universal_2" type="hinge" axis="1 0 0"
                 range="-1.57 1.57" damping="0.5" armature="0.01"/>

          <geom name="wrist_marker" type="sphere" size="0.03" rgba="0 1 0 0.8"/>
          <site name="wrist_site" pos="0 0 0" size="0.01"/>

          <!-- Club shaft -->
          <body name="club" pos="0 0 0">
            <geom name="club_shaft" type="capsule"
                  fromto="0 0 0 0.9 0 0" size="0.015"
                  material="club_mat" mass="0.3"/>

            <!-- Club head -->
            <body name="clubhead" pos="0.9 0 0">
              <geom name="clubhead_geom" type="box"
                    size="0.05 0.03 0.025"
                    rgba="0.1 0.1 0.1 1" mass="0.2"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <sensor>
    <!-- Constraint force sensors for universal joint -->
    <jointpos name="shoulder_pos" joint="shoulder"/>
    <jointvel name="shoulder_vel" joint="shoulder"/>
    <jointpos name="wrist_u1_pos" joint="wrist_universal_1"/>
    <jointvel name="wrist_u1_vel" joint="wrist_universal_1"/>
    <jointpos name="wrist_u2_pos" joint="wrist_universal_2"/>
    <jointvel name="wrist_u2_vel" joint="wrist_universal_2"/>

    <!-- Torque sensors (sites now defined within body hierarchy) -->
    <torque name="shoulder_torque" site="shoulder_site"/>
    <torque name="wrist_torque" site="wrist_site"/>
  </sensor>

  <actuator>
    <motor name="shoulder_motor" joint="shoulder" gear="50"
           ctrllimited="true" ctrlrange="-80 80"/>
    <motor name="wrist_u1_motor" joint="wrist_universal_1" gear="20"
           ctrllimited="true" ctrlrange="-30 30"/>
    <motor name="wrist_u2_motor" joint="wrist_universal_2" gear="20"
           ctrllimited="true" ctrlrange="-30 30"/>
  </actuator>
</mujoco>
"""


# ==============================================================================
# GIMBAL JOINT DEMONSTRATION MODEL
# ==============================================================================
GIMBAL_JOINT_DEMO_XML = rf"""
<mujoco model="gimbal_joint_demo">
  <option timestep="0.001" gravity="0 0 -{GRAVITY_M_S2}"
          integrator="RK4" solver="Newton"/>

  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>

  <visual>
    <map znear="0.01" zfar="20"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <material name="outer_ring" rgba="0.8 0.2 0.2 0.7"/>
    <material name="middle_ring" rgba="0.2 0.8 0.2 0.7"/>
    <material name="inner_ring" rgba="0.2 0.2 0.8 0.7"/>
    <material name="payload" rgba="0.8 0.8 0.2 1"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.5 0.5 0.5 1"/>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>

    <camera name="perspective" pos="-2 -2 1.5" euler="0.4 0 0.785"/>
    <camera name="side" pos="-3 0 1" euler="0.1 0 0"/>
    <camera name="top" pos="0 0 3" euler="0 0 0"/>

    <!-- Fixed support -->
    <body name="support" pos="0 0 1.0">
      <geom name="support_post" type="cylinder"
            fromto="0 0 -0.5 0 0 0" size="0.02"
            rgba="0.3 0.3 0.3 1"/>

      <!-- GIMBAL: Outer ring - rotates about Z axis -->
      <body name="gimbal_outer" pos="0 0 0">
        <joint name="gimbal_z" type="hinge" axis="0 0 1"
               limited="false" damping="0.3" armature="0.01"/>

        <geom name="outer_ring_1" type="capsule"
              fromto="-0.3 0 0 0.3 0 0" size="0.015"
              material="outer_ring"/>
        <geom name="outer_ring_2" type="capsule"
              fromto="0 -0.3 0 0 0.3 0" size="0.015"
              material="outer_ring"/>

        <!-- Middle ring - rotates about Y axis (perpendicular to outer) -->
        <body name="gimbal_middle" pos="0 0 0" euler="0 0 0">
          <joint name="gimbal_y" type="hinge" axis="0 1 0"
                 limited="false" damping="0.3" armature="0.01"/>

          <geom name="middle_ring_1" type="capsule"
                fromto="-0.25 0 0 0.25 0 0" size="0.013"
                material="middle_ring"/>
          <geom name="middle_ring_2" type="capsule"
                fromto="0 0 -0.25 0 0 0.25" size="0.013"
                material="middle_ring"/>

          <!-- Inner ring - rotates about X axis (perpendicular to middle) -->
          <body name="gimbal_inner" pos="0 0 0" euler="0 0 0">
            <joint name="gimbal_x" type="hinge" axis="1 0 0"
                   limited="false" damping="0.3" armature="0.01"/>

            <geom name="inner_ring_1" type="capsule"
                  fromto="0 -0.20 0 0 0.20 0" size="0.011"
                  material="inner_ring"/>
            <geom name="inner_ring_2" type="capsule"
                  fromto="0 0 -0.20 0 0 0.20" size="0.011"
                  material="inner_ring"/>

            <!-- Payload (club or sensor platform) -->
            <body name="payload" pos="0 0 0">
              <geom name="payload_geom" type="box"
                    size="0.15 0.08 0.04"
                    material="payload" mass="0.5"/>

              <!-- Orientation indicator -->
              <geom name="payload_arrow" type="capsule"
                    fromto="0 0 0 0 0.12 0" size="0.01"
                    rgba="1 0 0 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <sensor>
    <jointpos name="gimbal_z_pos" joint="gimbal_z"/>
    <jointvel name="gimbal_z_vel" joint="gimbal_z"/>
    <jointpos name="gimbal_y_pos" joint="gimbal_y"/>
    <jointvel name="gimbal_y_vel" joint="gimbal_y"/>
    <jointpos name="gimbal_x_pos" joint="gimbal_x"/>
    <jointvel name="gimbal_x_vel" joint="gimbal_x"/>

    <framepos name="payload_pos" objtype="body" objname="payload"/>
    <framequat name="payload_quat" objtype="body" objname="payload"/>
  </sensor>

  <actuator>
    <motor name="gimbal_z_motor" joint="gimbal_z" gear="10"
           ctrllimited="true" ctrlrange="-15 15"/>
    <motor name="gimbal_y_motor" joint="gimbal_y" gear="10"
           ctrllimited="true" ctrlrange="-15 15"/>
    <motor name="gimbal_x_motor" joint="gimbal_x" gear="10"
           ctrllimited="true" ctrlrange="-15 15"/>
  </actuator>
</mujoco>
"""


# ==============================================================================
# ADVANCED FULL-BODY MODEL WITH SPECIFIED JOINT TYPES
# ==============================================================================
def generate_flexible_club_xml(club_type: str = "driver", num_segments: int = 3) -> str:
    """Generate XML for a flexible golf club with specified number of segments.

    Args:
        club_type: "driver", "iron_7", or "wedge"
        num_segments: Number of flexible shaft segments (1-5)

    Returns:
        XML string for the club body

    Raises:
        ValueError: If club_type is not in CLUB_CONFIGS or num_segments is invalid
    """
    if club_type not in CLUB_CONFIGS:
        valid_types = ", ".join(CLUB_CONFIGS.keys())
        msg = f"Invalid club_type '{club_type}'. Must be one of: {valid_types}"
        raise ValueError(msg)

    if not (1 <= num_segments <= 5):
        msg = f"num_segments must be between 1 and 5, got {num_segments}"
        raise ValueError(msg)

    config = _extract_club_config(club_type)

    seg_length = config["shaft_length"] / num_segments
    seg_mass = config["shaft_mass"] / num_segments

    xml_parts = _generate_grip_xml(club_type, num_segments, config)
    xml_parts.extend(
        _generate_shaft_segments_xml(num_segments, config, seg_length, seg_mass)
    )
    xml_parts.extend(_generate_clubhead_xml(num_segments, config, seg_length))

    for i in range(num_segments):
        indent = "  " * (num_segments - i)
        xml_parts.append(f"{indent}</body>")
    xml_parts.append("</body>")

    return "\n".join(xml_parts)


def _extract_club_config(club_type: str) -> dict[str, Any]:
    raw = CLUB_CONFIGS[club_type]
    flex_stiffness = raw["flex_stiffness"]
    if not isinstance(flex_stiffness, list):
        msg = "flex_stiffness must be a list"
        raise TypeError(msg)
    head_size = raw["head_size"]
    if not isinstance(head_size, list):
        msg = "head_size must be a list"
        raise TypeError(msg)
    return {
        "grip_length": cast("float", raw["grip_length"]),
        "grip_radius": cast("float", raw["grip_radius"]),
        "grip_mass": cast("float", raw["grip_mass"]),
        "shaft_length": cast("float", raw["shaft_length"]),
        "shaft_radius": cast("float", raw["shaft_radius"]),
        "shaft_mass": cast("float", raw["shaft_mass"]),
        "head_mass": cast("float", raw["head_mass"]),
        "club_loft": cast("float", raw["club_loft"]),
        "flex_stiffness": flex_stiffness,
        "head_size": head_size,
    }


def _generate_grip_xml(
    club_type: str, num_segments: int, config: dict[str, Any]
) -> list[str]:
    grip_length = config["grip_length"]
    grip_radius = config["grip_radius"]
    grip_mass = config["grip_mass"]
    club_loft = config["club_loft"]

    g_ixx = grip_mass * grip_length**2 / 12
    g_izz = grip_mass * grip_radius**2 / 2

    return [
        f"<!-- {club_type.upper()} - {num_segments} segment flexible shaft -->",
        f'<body name="club_grip" pos="0 0 -0.10" euler="0 -{club_loft:.3f} 0">',
        f'  <inertial pos="0 0 -{grip_length / 2:.4f}" mass="{grip_mass:.4f}"',
        f'            diaginertia="{g_ixx:.8f} {g_ixx:.8f} {g_izz:.8f}"/>',
        '  <geom name="grip_geom" type="capsule"',
        f'        fromto="0 0 0 0 0 -{grip_length:.4f}"',
        f'        size="{grip_radius:.4f}" material="club_grip_mat"/>',
    ]


def _generate_shaft_segments_xml(
    num_segments: int,
    config: dict[str, Any],
    seg_length: float,
    seg_mass: float,
) -> list[str]:
    grip_length = config["grip_length"]
    shaft_radius = config["shaft_radius"]
    flex_stiffness = config["flex_stiffness"]

    xml_parts: list[str] = []
    for i in range(num_segments):
        seg_name = f"shaft_seg{i + 1}"
        is_first = i == 0

        stiffness_idx = min(i, 2)
        stiffness = float(flex_stiffness[stiffness_idx])

        if is_first:
            xml_parts.append(f"\n  <!-- Shaft Segment {i + 1} (upper) -->")
            xml_parts.append(
                f'  <body name="{seg_name}" pos="0 0 -{grip_length:.4f}">',
            )
        else:
            xml_parts.append(f"\n    <!-- Shaft Segment {i + 1} -->")
            xml_parts.append(
                f'    <body name="{seg_name}" pos="0 0 -{seg_length:.4f}">',
            )

        indent = "  " if is_first else "    "
        damping = max(0.4 - i * 0.1, 0.05)
        xml_parts.extend(
            [
                f'{indent}  <joint name="{seg_name}_flex" type="hinge" axis="1 0 0"',
                f'{indent}         range="-0.{15 + i * 5} 0.{15 + i * 5}" '
                f'damping="{damping:.2f}" stiffness="{stiffness}" armature="0.001"/>',
                f'{indent}  <inertial pos="0 0 -{seg_length / 2:.4f}" '
                f'mass="{seg_mass:.4f}"',
                f'{indent}            diaginertia="{seg_mass * seg_length**2 / 12:.8f} '
                f"{seg_mass * seg_length**2 / 12:.8f} "
                f'{seg_mass * shaft_radius**2 / 2:.8f}"/>',
                f'{indent}  <geom name="{seg_name}_geom" type="capsule" '
                f'fromto="0 0 0 0 0 -{seg_length:.4f}"',
                f'{indent}        size="{shaft_radius:.4f}" '
                f'material="club_shaft_mat"/>',
            ],
        )
    return xml_parts


def _generate_clubhead_xml(
    num_segments: int,
    config: dict[str, Any],
    seg_length: float,
) -> list[str]:
    head_mass = config["head_mass"]
    club_loft = config["club_loft"]
    head_size = config["head_size"]

    indent = "  " + "  " * num_segments
    h_w = float(head_size[0])
    h_h = float(head_size[1])
    h_d = float(head_size[2])

    ixx = head_mass * h_w**2 / 12
    iyy = head_mass * h_h**2 / 12
    izz = head_mass * h_d**2 / 12

    return [
        f"\n{indent}<!-- Club Head -->",
        f'{indent}<body name="hosel" pos="0 0 -{seg_length:.4f}"',
        f'{indent}      euler="0 {club_loft:.3f} 0">',
        f'{indent}  <inertial pos="0 0.02 -0.01" mass="0.010"',
        f'{indent}            diaginertia="0.000005 0.000005 0.000002"/>',
        f'{indent}  <geom name="hosel_geom" type="cylinder" '
        f'fromto="0 0 0 0 0.030 -0.005"',
        f'{indent}        size="0.008" material="club_head_mat"/>',
        f'{indent}  <body name="clubhead" pos="0 0.040 -0.008">',
        f'{indent}    <inertial pos="0 {h_h / 2:.4f} 0.002" mass="{head_mass:.4f}"',
        f'{indent}              diaginertia="{ixx:.6f} {iyy:.6f} {izz:.6f}"/>',
        f'{indent}    <geom name="head_body" type="box"',
        f'{indent}          size="{h_w:.4f} {h_h:.4f} {h_d:.4f}"',
        f'{indent}          pos="0 {h_h:.4f} 0" material="club_head_mat"/>',
        f'{indent}    <geom name="face" type="box"',
        f'{indent}          size="{h_w + 0.001:.4f} 0.003 {h_d + 0.001:.4f}"',
        f'{indent}          pos="0 {h_h * 2 + 0.003:.4f} 0" '
        f'rgba="0.85 0.15 0.15 0.9"/>',
        f"{indent}  </body>",
        f"{indent}</body>",
    ]


def generate_rigid_club_xml(club_type: str = "driver") -> str:
    """Generate XML for a rigid (non-flexible) golf club.

    Args:
        club_type: "driver", "iron_7", or "wedge"

    Returns:
        XML string for the club body

    Raises:
        ValueError: If club_type is not in CLUB_CONFIGS
    """
    if club_type not in CLUB_CONFIGS:
        valid_types = ", ".join(CLUB_CONFIGS.keys())
        msg = f"Invalid club_type '{club_type}'. Must be one of: {valid_types}"
        raise ValueError(
            msg,
        )

    config = CLUB_CONFIGS[club_type]

    # Extract typed values for type checking
    grip_length = cast("float", config["grip_length"])
    grip_radius = cast("float", config["grip_radius"])
    grip_mass = cast("float", config["grip_mass"])
    shaft_radius = cast("float", config["shaft_radius"])
    shaft_mass = cast("float", config["shaft_mass"])
    total_length = cast("float", config["total_length"])
    head_mass = cast("float", config["head_mass"])
    club_loft = cast("float", config["club_loft"])
    head_size = config["head_size"]
    if not isinstance(head_size, list):
        msg = "head_size must be a list"
        raise TypeError(msg)

    # Calculate properties for cleaner XML generation
    h_w = float(head_size[0])  # width (x)
    h_h = float(head_size[1])  # height (y/depth)
    h_d = float(head_size[2])  # depth (z/height)

    # Inertia tensor diagonal elements (box approximation)
    ixx = head_mass * h_w**2 / 12
    iyy = head_mass * h_h**2 / 12
    izz = head_mass * h_d**2 / 12

    return f"""<!-- {club_type.upper()} - Rigid shaft -->
<body name="club" pos="0 0 -0.10" euler="0 -{club_loft:.3f} 0">
  <!-- Grip -->
  <geom name="club_grip" type="capsule"
        fromto="0 0 0 0 0 -{grip_length:.4f}" size="{grip_radius:.4f}"
        material="club_grip_mat" mass="{grip_mass:.4f}"/>

  <!-- Shaft (rigid) -->
  <geom name="club_shaft" type="capsule"
        fromto="0 0 -{grip_length:.4f} 0 0 -{total_length:.4f}"
        size="{shaft_radius:.4f}"
        material="club_shaft_mat" mass="{shaft_mass:.4f}"/>

  <!-- Club Head -->
  <body name="clubhead" pos="0 0 -{total_length:.4f}">
    <inertial pos="0 {h_h / 2:.4f} 0.002" mass="{head_mass:.4f}"
              diaginertia="{ixx:.6f} {iyy:.6f} {izz:.6f}"/>
    <geom name="head_body" type="box" size="{h_w:.4f} {h_h:.4f} {h_d:.4f}"
          pos="0 {h_h:.4f} 0" material="club_head_mat"/>
    <geom name="face" type="box" size="{h_w + 0.001:.4f} 0.003 {h_d + 0.001:.4f}"
          pos="0 {h_h * 2 + 0.003:.4f} 0" rgba="0.85 0.15 0.15 0.9"/>
  </body>
</body>"""
