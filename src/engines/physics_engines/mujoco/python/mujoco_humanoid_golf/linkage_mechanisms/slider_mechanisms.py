# ruff: noqa: E501
"""Slider mechanism generators for MuJoCo.

Includes slider-crank and Scotch yoke mechanisms.

Extracted from __init__.py for SRP (#1485).
"""

from src.shared.python.core.constants import GRAVITY_M_S2


def _slider_crank_assets_xml() -> str:
    """Generate the asset section for the slider-crank XML."""
    return """
    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
                 rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
        <material name="crank_mat" rgba="0.9 0.1 0.1 1"/>
        <material name="rod_mat" rgba="0.1 0.7 0.1 1"/>
        <material name="slider_mat" rgba="0.1 0.1 0.9 1"/>
    </asset>"""


def _slider_crank_worldbody_xml(
    orientation: str,
    crank_length: float,
    rod_length: float,
    slider_axis: str,
    slider_start: float,
    slider_end: float,
) -> str:
    """Generate the worldbody section for the slider-crank XML."""
    return f"""
    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1" material="grid"/>

        <!-- Fixed crank pivot -->
        <geom name="crank_pivot" type="sphere" pos="0 0 1" size="0.1"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

        <!-- Slider guide rail -->
        <geom type="cylinder"
              fromto="{slider_start if orientation == "horizontal" else 0} 0 {1 if orientation == "horizontal" else slider_start}
                      {slider_end if orientation == "horizontal" else 0} 0 {1 if orientation == "horizontal" else slider_end}"
              size="0.04" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>

        <!-- Crank -->
        <body name="crank" pos="0 0 1">
            <joint name="crank_joint" type="hinge" axis="0 1 0" damping="0.2"/>
            <geom type="capsule"
                  fromto="0 0 0
                          {crank_length if orientation == "horizontal" else 0} 0
                          {0 if orientation == "horizontal" else crank_length}"
                  size="0.06" material="crank_mat" mass="1.0"/>
            <geom name="crank_end" type="sphere"
                  pos="{crank_length if orientation == "horizontal" else 0} 0
                       {0 if orientation == "horizontal" else crank_length}"
                  size="0.08" rgba="1 0.5 0 1" contype="0" conaffinity="0"/>

            <!-- Connecting rod -->
            <body name="connecting_rod"
                  pos="{crank_length if orientation == "horizontal" else 0} 0
                       {0 if orientation == "horizontal" else crank_length}">
                <joint name="rod_crank_joint" type="hinge" axis="0 1 0" damping="0.1"/>
                <geom type="capsule"
                  fromto="0 0 0
                          {rod_length if orientation == "horizontal" else 0} 0
                          {0 if orientation == "horizontal" else rod_length}"
                  size="0.05" material="rod_mat" mass="0.5"/>
            </body>
        </body>

        <!-- Slider -->
        <body name="slider"
              pos="{rod_length + crank_length if orientation == "horizontal" else 0} 0
                   {1 if orientation == "horizontal" else rod_length + crank_length}">
            <joint name="slider_joint" type="slide" axis="{slider_axis}" damping="1.0"
                   range="{slider_start} {slider_end}"/>
            <geom type="box" size="0.15 0.1 0.15" material="slider_mat" mass="2.0"/>
            <geom name="slider_marker" type="sphere" pos="0 0 0.2" size="0.1"
                  rgba="1 0 1 1" contype="0" conaffinity="0"/>
        </body>
    </worldbody>"""


def _slider_crank_constraints_xml(orientation: str, rod_length: float) -> str:
    """Generate the equality and actuator sections for the slider-crank XML."""
    return f"""
    <equality>
        <connect body1="connecting_rod" body2="slider"
                 anchor="{rod_length if orientation == "horizontal" else 0} 0
                         {0 if orientation == "horizontal" else rod_length}"/>
    </equality>

    <actuator>
        <motor name="crank_motor" joint="crank_joint" gear="30"
              ctrllimited="true" ctrlrange="-10 10"/>
    </actuator>"""


def generate_slider_crank_xml(
    crank_length: float = 1.0,
    rod_length: float = 3.0,
    orientation: str = "horizontal",
) -> str:
    """Generate a slider-crank mechanism (basis for piston engines).

    Parameters
    ----------
    crank_length : float
        Length of the rotating crank
    rod_length : float
        Length of the connecting rod
    orientation : str
        "horizontal" or "vertical" slider direction
    """
    slider_axis = "1 0 0" if orientation == "horizontal" else "0 0 1"
    slider_start = -rod_length - crank_length
    slider_end = rod_length + crank_length

    assets = _slider_crank_assets_xml()
    worldbody = _slider_crank_worldbody_xml(
        orientation, crank_length, rod_length, slider_axis, slider_start, slider_end
    )
    constraints = _slider_crank_constraints_xml(orientation, rod_length)

    return f"""
<mujoco model="slider_crank">
    <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}"/>
{assets}
{worldbody}
{constraints}
</mujoco>
"""


def generate_scotch_yoke_xml(crank_radius: float = 1.0) -> str:
    """Generate a Scotch yoke mechanism (perfect harmonic motion).

    Parameters
    ----------
    crank_radius : float
        Radius of the rotating crank
    """
    return f"""
<mujoco model="scotch_yoke">
    <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
                 rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="1 1"/>
        <material name="crank_mat" rgba="0.9 0.1 0.1 1"/>
        <material name="yoke_mat" rgba="0.1 0.7 0.9 1"/>
    </asset>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1" material="grid"/>

        <!-- Central pivot -->
        <geom name="center_pivot" type="sphere" pos="0 0 1.5" size="0.12"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

        <!-- Vertical guide rails for yoke -->
        <geom type="cylinder" fromto="0 -0.3 0  0 -0.3 3" size="0.03"
              rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
        <geom type="cylinder" fromto="0 0.3 0  0 0.3 3" size="0.03"
              rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>

        <!-- Crank -->
        <body name="crank" pos="0 0 1.5">
            <joint name="crank_joint" type="hinge" axis="0 1 0" damping="0.1"/>
            <geom type="cylinder" fromto="0 0 0  0 0 {crank_radius}" size="0.04"
                  material="crank_mat" mass="0.5"/>
            <geom type="sphere" pos="0 0 {crank_radius}" size="0.08"
                  rgba="1 0.5 0 1" mass="0.2" contype="0" conaffinity="0"/>

            <!-- Pin on crank end -->
            <body name="crank_pin" pos="0 0 {crank_radius}">
                <geom type="sphere" size="0.06" rgba="1 1 0 1" mass="0.1"/>
            </body>
        </body>

        <!-- Yoke (slider) -->
        <body name="yoke" pos="0 0 1.5">
            <joint name="yoke_joint" type="slide" axis="0 0 1" damping="2.0"
                   range="{-crank_radius - 0.5} {crank_radius + 0.5}"/>
            <geom type="box" size="0.4 0.35 0.1" material="yoke_mat" mass="2.0"/>
            <!-- Slot visualization (simplified) -->
            <geom type="box" pos="0 0 0" size="0.35 0.08 0.08"
                  rgba="0.2 0.2 0.2 0.5" contype="0" conaffinity="0"/>
        </body>
    </worldbody>

    <equality>
        <!-- Connect crank pin to yoke slot (simplified as weld in y direction) -->
        <weld body1="crank_pin" body2="yoke" solref="0.001 1"
              solimp="0.9 0.95 0.001"/>
    </equality>

    <actuator>
        <motor name="crank_motor" joint="crank_joint" gear="20"
              ctrllimited="true" ctrlrange="-8 8"/>
    </actuator>
</mujoco>
"""
