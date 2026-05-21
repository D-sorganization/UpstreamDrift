"""MuJoCo MJCF XML generator for the lower body golf-swing model."""

from __future__ import annotations

import textwrap


def build_lower_body_xml(
    thigh_mass: float = 10.0,
    calf_mass: float = 4.5,
    foot_mass: float = 1.0,
    pelvis_mass: float = 20.0,
    thigh_length: float = 0.45,
    calf_length: float = 0.45,
    pelvis_width: float = 0.3,
) -> str:
    """Build an MJCF XML string describing the lower body model.

    The model has a 6-DOF free-joint pelvis, gimbal (3 axes) hips, revolute
    knees and universal ankles, plus contact-based foot-ground interaction.

    Args:
        thigh_mass: Mass of each thigh ellipsoid (kg).
        calf_mass: Mass of each calf ellipsoid (kg).
        foot_mass: Mass of each foot ellipsoid (kg).
        pelvis_mass: Mass of the pelvis body (kg).
        thigh_length: Length of each thigh (m).
        calf_length: Length of each calf (m).
        pelvis_width: Side-to-side width of the pelvis (m); hips are placed
            at ``pelvis_width / 2`` on either side of the body centreline.

    Returns:
        A dedented MJCF XML string ready for ``mujoco.MjModel.from_xml_string``.

    Raises:
        TypeError: If any argument is not a real number.
        ValueError: If any argument is not strictly positive.
    """
    _validate_positive_float("thigh_mass", thigh_mass)
    _validate_positive_float("calf_mass", calf_mass)
    _validate_positive_float("foot_mass", foot_mass)
    _validate_positive_float("pelvis_mass", pelvis_mass)
    _validate_positive_float("thigh_length", thigh_length)
    _validate_positive_float("calf_length", calf_length)
    _validate_positive_float("pelvis_width", pelvis_width)

    hip_offset = pelvis_width / 2.0
    pelvis_z = thigh_length + calf_length + 0.1
    pelvis_geoms = _pelvis_anatomical_geoms(hip_offset, pelvis_mass)

    xml = f"""
    <mujoco model="lower_body_model">
        <compiler angle="degree" coordinate="local"/>
        <option gravity="0 0 -9.81" timestep="0.005"/>
        <size njmax="500" nconmax="100"/>
        <asset>
            <material name="matplane" reflectance="0.3" specular="1" shininess="1" rgba="0.8 0.9 0.8 1"/>
            <material name="matgeom" rgba="0.8 0.6 0.4 1"/>
            <material name="matfoot" rgba="0.2 0.2 0.2 1"/>
        </asset>

        <worldbody>
            <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
            <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1" material="matplane" condim="3"/>

            <body name="pelvis" pos="0 0 {pelvis_z}">
                <freejoint name="root"/>
{pelvis_geoms}

                <!-- RIGHT LEG -->
{_build_leg_xml("r", hip_offset, thigh_length, calf_length, thigh_mass, calf_mass, foot_mass)}

                <!-- LEFT LEG -->
{_build_leg_xml("l", hip_offset, thigh_length, calf_length, thigh_mass, calf_mass, foot_mass)}
            </body>
        </worldbody>

        <actuator>
{_build_leg_actuators_xml("r")}

{_build_leg_actuators_xml("l")}
        </actuator>

        <sensor>
            <touch name="r_foot_touch" site="r_foot_center"/>
            <touch name="l_foot_touch" site="l_foot_center"/>
        </sensor>
    </mujoco>
    """

    return textwrap.dedent(xml).strip()


def _validate_positive_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if value <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {value}")


def _build_leg_xml(
    side: str,
    hip_offset: float,
    thigh_length: float,
    calf_length: float,
    thigh_mass: float,
    calf_mass: float,
    foot_mass: float,
) -> str:
    """Return the MJCF fragment for one leg (thigh -> calf -> foot).

    ``side`` is ``"r"`` or ``"l"``. The right leg is placed at ``-hip_offset``
    on the body Y axis and the left at ``+hip_offset``; everything else is
    identical, so this helper is the single source of truth for leg
    geometry, joint axes, joint ranges, masses, and naming.
    """
    if side not in ("r", "l"):
        raise ValueError(f"side must be 'r' or 'l', got {side!r}")
    y_sign = "-" if side == "r" else ""
    return (
        f'                <body name="{side}_thigh" pos="0 {y_sign}{hip_offset} -0.05">\n'
        f'                    <joint name="{side}_hip_x" type="hinge" axis="1 0 0" range="-90 90"/>\n'
        f'                    <joint name="{side}_hip_y" type="hinge" axis="0 1 0" range="-90 90"/>\n'
        f'                    <joint name="{side}_hip_z" type="hinge" axis="0 0 1" range="-90 90"/>\n'
        f'                    <geom type="ellipsoid" size="0.08 0.08 {thigh_length / 2}" pos="0 0 -{thigh_length / 2}" mass="{thigh_mass}" material="matgeom"/>\n'
        f'                    <body name="{side}_calf" pos="0 0 -{thigh_length}">\n'
        f'                        <joint name="{side}_knee" type="hinge" axis="0 1 0" range="0 150"/>\n'
        f'                        <geom type="ellipsoid" size="0.06 0.06 {calf_length / 2}" pos="0 0 -{calf_length / 2}" mass="{calf_mass}" material="matgeom"/>\n'
        f'                        <body name="{side}_foot" pos="0 0 -{calf_length}">\n'
        f'                            <joint name="{side}_ankle_x" type="hinge" axis="1 0 0" range="-60 60"/>\n'
        f'                            <joint name="{side}_ankle_y" type="hinge" axis="0 1 0" range="-60 60"/>\n'
        f'                            <geom name="{side}_foot_geom" type="ellipsoid" size="0.13 0.05 0.04" pos="0.06 0 -0.04" mass="{foot_mass}" material="matfoot" condim="3"/>\n'
        f'                            <site name="{side}_foot_center" type="sphere" size="0.01" pos="0.06 0 -0.04" rgba="1 0 0 1"/>\n'
        f"                        </body>\n"
        f"                    </body>\n"
        f"                </body>"
    )


def _build_leg_actuators_xml(side: str) -> str:
    """Return the six motor declarations for one leg (hip x/y/z, knee, ankle x/y)."""
    if side not in ("r", "l"):
        raise ValueError(f"side must be 'r' or 'l', got {side!r}")
    joints = (
        f"{side}_hip_x",
        f"{side}_hip_y",
        f"{side}_hip_z",
        f"{side}_knee",
        f"{side}_ankle_x",
        f"{side}_ankle_y",
    )
    return "\n".join(
        f'            <motor name="act_{joint}" joint="{joint}" gear="1" '
        f'ctrllimited="true" ctrlrange="-500 500"/>'
        for joint in joints
    )


# Anatomical pelvis marker geometry.
#
# All visual markers are declared with mass="0" and contype="0" conaffinity="0"
# so they don't contribute to inertia or generate contacts. All mass is held by
# the single "pelvis_body" ellipsoid, preserving the dynamics of the original
# simple pelvis exactly while making pelvic tilt visually unambiguous.
#
# Coordinate convention for the pelvis body frame: +X forward, +Y left, +Z up.
#
# Landmarks (relative to the pelvis body origin):
#   sacrum         — posterior-superior midline
#   iliac wings    — bilateral flattened ellipsoids forming the "butterfly" top
#   ASIS markers   — anterior-superior iliac spines, bright so tilt reads clearly
#   pubic symphysis — anterior-inferior midline
_PELVIS_SEMI_X = 0.15  # Semi-axis of the inertial host ellipsoid (forward).
_PELVIS_SEMI_Z = 0.12  # Semi-axis of the inertial host ellipsoid (up).


def _pelvis_anatomical_geoms(hip_offset: float, pelvis_mass: float) -> str:
    """Return the indented MJCF geom fragment for an anatomical pelvis shape."""
    host_semi_y = hip_offset + 0.05

    # Landmark positions scale laterally with hip_offset so the markers track
    # pelvis_width. X/Z positions are fixed relative to the host ellipsoid.
    sacrum_pos_x = -0.10
    sacrum_pos_z = 0.04
    ilium_pos_x = 0.02
    ilium_pos_y = 0.65 * hip_offset
    ilium_pos_z = 0.04
    asis_pos_x = 0.13
    asis_pos_y = 0.80 * hip_offset
    asis_pos_z = 0.06
    pubis_pos_x = 0.12
    pubis_pos_z = -0.08

    return (
        f'                <geom name="pelvis_body" type="ellipsoid" '
        f'size="{_PELVIS_SEMI_X} {host_semi_y} {_PELVIS_SEMI_Z}" '
        f'mass="{pelvis_mass}" rgba="0.82 0.72 0.55 0.35" '
        f'contype="0" conaffinity="0"/>\n'
        f'                <geom name="pelvis_sacrum" type="ellipsoid" '
        f'size="0.035 0.05 0.08" pos="{sacrum_pos_x} 0 {sacrum_pos_z}" '
        f'mass="0" rgba="0.55 0.40 0.28 1" contype="0" conaffinity="0"/>\n'
        f'                <geom name="pelvis_r_ilium" type="ellipsoid" '
        f'size="0.10 0.025 0.10" pos="{ilium_pos_x} -{ilium_pos_y} {ilium_pos_z}" '
        f'mass="0" rgba="0.95 0.90 0.78 1" contype="0" conaffinity="0"/>\n'
        f'                <geom name="pelvis_l_ilium" type="ellipsoid" '
        f'size="0.10 0.025 0.10" pos="{ilium_pos_x} {ilium_pos_y} {ilium_pos_z}" '
        f'mass="0" rgba="0.95 0.90 0.78 1" contype="0" conaffinity="0"/>\n'
        f'                <geom name="pelvis_r_asis" type="sphere" '
        f'size="0.025" pos="{asis_pos_x} -{asis_pos_y} {asis_pos_z}" '
        f'mass="0" rgba="0.95 0.15 0.15 1" contype="0" conaffinity="0"/>\n'
        f'                <geom name="pelvis_l_asis" type="sphere" '
        f'size="0.025" pos="{asis_pos_x} {asis_pos_y} {asis_pos_z}" '
        f'mass="0" rgba="0.95 0.15 0.15 1" contype="0" conaffinity="0"/>\n'
        f'                <geom name="pelvis_pubis" type="ellipsoid" '
        f'size="0.03 0.04 0.025" pos="{pubis_pos_x} 0 {pubis_pos_z}" '
        f'mass="0" rgba="0.85 0.78 0.65 1" contype="0" conaffinity="0"/>'
    )
