"""External model references and golf club XML generators.

Includes MyoSuite model paths, CMU Humanoid loader, and flexible/rigid
club XML generators driven by CLUB_CONFIGS.
"""

from __future__ import annotations

from typing import cast

from src.shared.python.equipment import CLUB_CONFIGS

# ==============================================================================

# ==============================================================================
# MyoSuite Musculoskeletal Model Paths
# ==============================================================================
# These models use the MyoSuite musculoskeletal framework with realistic
# muscle-tendon units, biomechanical constraints, and physiological parameters.
# Models are loaded from external XML files in the myo_sim directory.

# Upper body model (19 DOF, 20 actuators): Torso + head + both arms
MYOUPPERBODY_PATH = "myo_sim/body/myoupperbody.xml"

# Full body model (52 DOF, 290 actuators): Complete musculoskeletal system
# Includes torso, head, arms, and legs with muscle-tendon units
MYOBODY_PATH = "myo_sim/body/myobody.xml"

# Simplified arm model (bilateral, 14 DOF): Both arms with simplified torso
MYOARM_SIMPLE_PATH = "myo_sim/arm/myoarm_simple.xml"


# ==============================================================================
# DeepMind Control Suite - CMU Humanoid Model
# ==============================================================================
# The humanoid CM (Carnegie Mellon University) model is from the DeepMind
# Control Suite. It provides a realistic full-body humanoid based on CMU
# motion capture data with 16 actuated joints.

HUMANOID_CM_JOINTS = [
    "Lower Back",
    "Upper Back",
    "R Tibia",
    "L Tibia",
    "R Femur",
    "L Femur",
    "R Foot",
    "L Foot",
    "R Humerus RX",
    "L Humerus RX",
    "R Humerus RZ",
    "L Humerus RZ",
    "R Humerus RY",
    "L Humerus RY",
    "R Radius",
    "L Radius",
]


def load_humanoid_cm_xml() -> str | None:
    """Load the CMU Humanoid MJCF XML from dm_control if available.

    The humanoid CM (CMU) model is the original MuJoCo humanoid from the
    DeepMind Control Suite, based on CMU motion capture anthropometry.

    Returns:
        XML string if dm_control is installed, None otherwise.
    """
    try:
        from dm_control import suite

        env = suite.load(domain_name="humanoid", task_name="stand")
        return env.physics.model.get_xml()
    except ImportError:
        return None


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
        raise ValueError(
            msg,
        )

    if not (1 <= num_segments <= 5):
        msg = f"num_segments must be between 1 and 5, got {num_segments}"
        raise ValueError(msg)

    config = CLUB_CONFIGS[club_type]

    # Extract typed values for type checking
    grip_length = cast("float", config["grip_length"])
    grip_radius = cast("float", config["grip_radius"])
    grip_mass = cast("float", config["grip_mass"])
    shaft_length = cast("float", config["shaft_length"])
    shaft_radius = cast("float", config["shaft_radius"])
    shaft_mass = cast("float", config["shaft_mass"])
    head_mass = cast("float", config["head_mass"])
    club_loft = cast("float", config["club_loft"])
    flex_stiffness = config["flex_stiffness"]
    if not isinstance(flex_stiffness, list):
        msg = "flex_stiffness must be a list"
        raise TypeError(msg)
    head_size = config["head_size"]
    if not isinstance(head_size, list):
        msg = "head_size must be a list"
        raise TypeError(msg)

    # Calculate segment properties
    seg_length = shaft_length / num_segments
    seg_mass = shaft_mass / num_segments

    # Grip inertia properties
    g_ixx = grip_mass * grip_length**2 / 12
    g_izz = grip_mass * grip_radius**2 / 2

    xml_parts = [
        f"<!-- {club_type.upper()} - {num_segments} segment flexible shaft -->",
        f'<body name="club_grip" pos="0 0 -0.10" euler="0 -{club_loft:.3f} 0">',
        f'  <inertial pos="0 0 -{grip_length / 2:.4f}" mass="{grip_mass:.4f}"',
        f'            diaginertia="{g_ixx:.8f} {g_ixx:.8f} {g_izz:.8f}"/>',
        '  <geom name="grip_geom" type="capsule"',
        f'        fromto="0 0 0 0 0 -{grip_length:.4f}"',
        f'        size="{grip_radius:.4f}" material="club_grip_mat"/>',
    ]

    # Generate shaft segments
    for i in range(num_segments):
        seg_name = f"shaft_seg{i + 1}"
        is_first = i == 0

        stiffness_idx = min(i, 2)  # Use up to 3 stiffness values
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
        # Ensure minimum damping to prevent instability (never less than 0.05)
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

    # Add clubhead
    indent = "  " + "  " * num_segments
    # Calculate properties for cleaner XML generation
    h_w = float(head_size[0])
    h_h = float(head_size[1])
    h_d = float(head_size[2])

    ixx = head_mass * h_w**2 / 12
    iyy = head_mass * h_h**2 / 12
    izz = head_mass * h_d**2 / 12

    xml_parts.extend(
        [
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
        ],
    )

    # Close all body tags
    for i in range(num_segments):
        indent = "  " * (num_segments - i)
        xml_parts.append(f"{indent}</body>")

    xml_parts.append("</body>")

    return "\n".join(xml_parts)


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
