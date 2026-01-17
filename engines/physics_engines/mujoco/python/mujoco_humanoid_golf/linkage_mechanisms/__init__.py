# ruff: noqa: E501
"""
Linkage Mechanisms Library for MuJoCo Physics Exploration

This module provides a comprehensive collection of classic linkage mechanisms
and parallel manipulators for educational and research purposes.

Categories:
- Four-bar linkages (Grashof types, parallel, antiparallel)
- Slider mechanisms (slider-crank, Scotch yoke, slider-rocker)
- Special mechanisms (Geneva drive, Oldham coupling)
- Straight-line mechanisms (Peaucellier, Chebyshev, Watt, Roberts)
- Parallel mechanisms (Delta robot, Stewart platform, 5-bar parallel)
- Scaling mechanisms (Pantograph)
- Cam and follower systems
"""

import numpy as np
from shared.python.constants import GRAVITY_M_S2


def generate_four_bar_linkage_xml(
    link_lengths: list[float] | None = None,
    link_type: str = "grashof_crank_rocker",
) -> str:
    """
    Generate a four-bar linkage mechanism.

    Parameters:
    -----------
    link_lengths : list of 4 floats, optional
        Lengths of [ground, crank, coupler, follower]. If None, uses defaults.
    link_type : str
        Type of four-bar: "grashof_crank_rocker", "grashof_double_crank",
        "grashof_double_rocker", "non_grashof", "parallel", "antiparallel"

    Returns:
    --------
    str : MuJoCo XML string
    """
    # Default configurations for different types
    configs = {
        "grashof_crank_rocker": [4.0, 1.0, 3.5, 3.0],  # s + l < p + q
        "grashof_double_crank": [4.0, 2.0, 4.5, 3.5],  # Parallelogram-like
        "grashof_double_rocker": [4.0, 3.8, 2.5, 3.0],  # Both rock
        "non_grashof": [4.0, 4.5, 2.0, 3.0],  # s + l > p + q
        "parallel": [4.0, 2.0, 4.0, 2.0],  # Parallelogram
        "antiparallel": [4.0, 2.0, 4.0, 2.0],  # Crossed linkage
    }

    if link_lengths is None:
        link_lengths = configs.get(link_type, configs["grashof_crank_rocker"])

    ground, crank, coupler, follower = link_lengths

    # Crank position at origin, follower at (ground, 0)

    # Initial coupler end position (approximate)
    crank * np.cos(np.pi / 4)
    crank * np.sin(np.pi / 4)

    return f"""
<mujoco model="four_bar_linkage_{link_type}">
    <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
        <map stiffness="700" shadowscale="0.5" fogstart="10" fogend="15"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
                 rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
        <material name="crank_mat" rgba="0.9 0.1 0.1 1"/>
        <material name="coupler_mat" rgba="0.1 0.7 0.1 1"/>
        <material name="follower_mat" rgba="0.1 0.1 0.9 1"/>
        <material name="ground_mat" rgba="0.3 0.3 0.3 1"/>
    </asset>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" specular="0.3 0.3 0.3"
              pos="0 0 5" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1" material="grid"/>

        <!-- Ground link visualization -->
        <geom type="cylinder" fromto="0 0 0.5  {ground} 0 0.5" size="0.03"
              material="ground_mat" contype="0" conaffinity="0"/>

        <!-- Fixed pivot points -->
        <geom name="pivot_crank" type="sphere" pos="0 0 0.5" size="0.08"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
        <geom name="pivot_follower" type="sphere" pos="{ground} 0 0.5" size="0.08"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

        <!-- Crank link -->
        <body name="crank" pos="0 0 0.5">
            <joint name="crank_joint" type="hinge" axis="0 0 1" damping="0.1"/>
            <geom type="capsule"
                  fromto="0 0 0  {crank * np.cos(np.pi / 4)} {crank * np.sin(np.pi / 4)} 0"
                  size="0.05" material="crank_mat" mass="0.5"/>
            <geom name="crank_end" type="sphere"
                  pos="{crank * np.cos(np.pi / 4)} {crank * np.sin(np.pi / 4)} 0"
                  size="0.07" rgba="1 0.5 0 1" contype="0" conaffinity="0"/>

            <!-- Coupler link attached to crank end -->
            <body name="coupler" pos="{crank * np.cos(np.pi / 4)} {crank * np.sin(np.pi / 4)} 0">
                <joint name="coupler_crank_joint" type="hinge" axis="0 0 1" damping="0.05"/>
                <geom type="capsule" fromto="0 0 0  {coupler * 0.7} {coupler * 0.3} 0"
                      size="0.05" material="coupler_mat" mass="0.5"/>
                <geom name="coupler_point" type="sphere" pos="{coupler * 0.35} {coupler * 0.15} 0"
                      size="0.06" rgba="1 1 0 1" contype="0" conaffinity="0"/>
            </body>
        </body>

        <!-- Follower link -->
        <body name="follower" pos="{ground} 0 0.5">
            <joint name="follower_joint" type="hinge" axis="0 0 1" damping="0.1"/>
            <geom type="capsule" fromto="0 0 0  {-follower * 0.7} {follower * 0.3} 0"
                  size="0.05" material="follower_mat" mass="0.5"/>
            <geom name="follower_end" type="sphere" pos="{-follower * 0.7} {follower * 0.3} 0"
                  size="0.07" rgba="0.5 0 1 1" contype="0" conaffinity="0"/>
        </body>
    </worldbody>

    <actuator>
        <motor name="crank_motor" joint="crank_joint" gear="20"
              ctrllimited="true" ctrlrange="-5 5"/>
    </actuator>
</mujoco>
"""


def generate_slider_crank_xml(
    crank_length: float = 1.0,
    rod_length: float = 3.0,
    orientation: str = "horizontal",
) -> str:
    """
    Generate a slider-crank mechanism (basis for piston engines).

    Parameters:
    -----------
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

    return f"""
<mujoco model="slider_crank">
    <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}"/>

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
    </asset>

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
    </worldbody>

    <equality>
        <connect body1="connecting_rod" body2="slider"
                 anchor="{rod_length if orientation == "horizontal" else 0} 0
                         {0 if orientation == "horizontal" else rod_length}"/>
    </equality>

    <actuator>
        <motor name="crank_motor" joint="crank_joint" gear="30"
              ctrllimited="true" ctrlrange="-10 10"/>
    </actuator>
</mujoco>
"""


def generate_scotch_yoke_xml(crank_radius: float = 1.0) -> str:
    """
    Generate a Scotch yoke mechanism (perfect harmonic motion).

    Parameters:
    -----------
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


def generate_geneva_mechanism_xml(num_slots: int = 6, drive_radius: float = 2.0) -> str:
    """
    Generate a Geneva mechanism (intermittent motion).

    Parameters:
    -----------
    num_slots : int
        Number of slots in the Geneva wheel (typically 4, 6, or 8)
    drive_radius : float
        Radius of the drive wheel
    """
    return f"""
<mujoco model="geneva_mechanism">
    <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
                 rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="1 1"/>
        <material name="drive_mat" rgba="0.9 0.1 0.1 1"/>
        <material name="geneva_mat" rgba="0.1 0.7 0.1 1"/>
    </asset>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1" material="grid"/>

        <!-- Drive wheel (continuous rotation) -->
        <body name="drive_wheel" pos="0 0 1">
            <joint name="drive_joint" type="hinge" axis="0 0 1" damping="0.5"/>
            <geom type="cylinder" size="{drive_radius * 0.8} 0.1" material="drive_mat" mass="1.0"/>
            <!-- Drive pin -->
            <geom type="cylinder"
                  fromto="{drive_radius * 0.5} 0 0.11  {drive_radius * 0.5} 0 0.3"
                  size="0.1" rgba="1 0.5 0 1" mass="0.3"/>
            <geom type="sphere" pos="{drive_radius * 0.5} 0 0.3" size="0.12"
                  rgba="1 1 0 1" mass="0.1"/>
        </body>

        <!-- Geneva wheel (intermittent rotation) -->
        <body name="geneva_wheel" pos="{drive_radius * 1.5} 0 1">
            <joint name="geneva_joint" type="hinge" axis="0 0 1" damping="1.0"/>
            <geom type="cylinder" size="{drive_radius} 0.15" material="geneva_mat" mass="2.0"/>
            <!-- Simplified slot representations as visual markers -->
            <geom type="box" pos="{drive_radius * 0.7} 0 0.16" size="0.15 0.12 0.05"
                  rgba="0.2 0.9 0.2 0.7" contype="0" conaffinity="0"/>
            <geom type="box" pos="{-drive_radius * 0.7} 0 0.16" size="0.15 0.12 0.05"
                  rgba="0.2 0.9 0.2 0.7" contype="0" conaffinity="0"/>
            <geom type="box" pos="0 {drive_radius * 0.7} 0.16" size="0.12 0.15 0.05"
                  rgba="0.2 0.9 0.2 0.7" contype="0" conaffinity="0"/>
            <geom type="box" pos="0 {-drive_radius * 0.7} 0.16" size="0.12 0.15 0.05"
                  rgba="0.2 0.9 0.2 0.7" contype="0" conaffinity="0"/>
        </body>

        <!-- Center pivots -->
        <geom name="drive_pivot" type="sphere" pos="0 0 1" size="0.15"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
        <geom name="geneva_pivot" type="sphere" pos="{drive_radius * 1.5} 0 1" size="0.15"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
    </worldbody>

    <actuator>
        <motor name="drive_motor" joint="drive_joint" gear="25"
              ctrllimited="true" ctrlrange="-5 5"/>
    </actuator>
</mujoco>
"""


def generate_peaucellier_linkage_xml(scale: float = 1.0) -> str:
    """
    Generate Peaucellier-Lipkin linkage (exact straight-line mechanism).

    This is one of the first planar linkages capable of transforming
    rotary motion into perfect straight-line motion.
    """
    return f"""
<mujoco model="peaucellier_linkage">
    <option timestep="0.001" gravity="0 0 -{GRAVITY_M_S2}"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
                 rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="1 1"/>
        <material name="link_mat" rgba="0.1 0.7 0.9 1"/>
        <material name="tracer_mat" rgba="1 0.1 0.1 1"/>
    </asset>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1" material="grid"/>

        <!-- Fixed anchor points -->
        <geom name="anchor_O" type="sphere" pos="0 0 1" size="0.1"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
        <geom name="anchor_A" type="sphere" pos="{2.5 * scale} 0 1" size="0.1"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

        <!-- Drive arm from O -->
        <body name="drive_arm" pos="0 0 1">
            <joint name="drive_joint" type="hinge" axis="0 0 1" damping="0.2"/>
            <geom type="capsule" fromto="0 0 0  {1.5 * scale} 0 0" size="0.04"
                  rgba="0.9 0.5 0.1 1" mass="0.5"/>

            <!-- Point B on drive arm -->
            <body name="point_B" pos="{1.5 * scale} 0 0">
                <geom type="sphere" size="0.08" rgba="1 0.7 0 1" mass="0.2" contype="0" conaffinity="0"/>
            </body>
        </body>

        <!-- Link from A to B -->
        <body name="link_AB" pos="{2.5 * scale} 0 1">
            <joint name="joint_A" type="hinge" axis="0 0 1" damping="0.1"/>
            <geom type="capsule" fromto="0 0 0  {-1.8 * scale} 0 0" size="0.04"
                  material="link_mat" mass="0.4"/>
        </body>

        <!-- Rhombus linkage (simplified representation) -->
        <body name="rhombus_center" pos="{1.2 * scale} 0 1">
            <joint name="rhombus_joint" type="hinge" axis="0 0 1" damping="0.1"/>
            <geom type="capsule" fromto="{-0.5 * scale} {0.8 * scale} 0  {0.5 * scale} {-0.8 * scale} 0"
                  size="0.03" material="link_mat" mass="0.3"/>
            <geom type="capsule" fromto="{-0.5 * scale} {-0.8 * scale} 0  {0.5 * scale} {0.8 * scale} 0"
                  size="0.03" material="link_mat" mass="0.3"/>

            <!-- Tracer point (generates straight line) -->
            <body name="tracer" pos="{1.2 * scale} 0 0">
                <geom type="sphere" size="0.12" material="tracer_mat" mass="0.3"/>
                <geom type="cylinder" fromto="0 0 0  0 0 0.5" size="0.02"
                      rgba="1 0 0 0.5" contype="0" conaffinity="0"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor name="drive_motor" joint="drive_joint" gear="15" ctrllimited="true" ctrlrange="-5 5"/>
    </actuator>
</mujoco>
"""


def generate_chebyshev_linkage_xml() -> str:
    """
    Generate Chebyshev linkage (approximate straight-line mechanism).

    Produces an approximate straight line over a portion of its path,
    useful for walking mechanisms.
    """
    return f"""
<mujoco model="chebyshev_linkage">
    <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker"
                 rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="1 1"/>
        <material name="link1_mat" rgba="0.9 0.1 0.1 1"/>
        <material name="link2_mat" rgba="0.1 0.7 0.1 1"/>
        <material name="link3_mat" rgba="0.1 0.1 0.9 1"/>
    </asset>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1" material="grid"/>

        <!-- Ground link (4 units) -->
        <geom type="cylinder" fromto="-2 0 0.5  2 0 0.5" size="0.04"
              rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>

        <!-- Left pivot -->
        <geom name="left_pivot" type="sphere" pos="-2 0 0.5" size="0.1"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

        <!-- Right pivot -->
        <geom name="right_pivot" type="sphere" pos="2 0 0.5" size="0.1"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

        <!-- Left crank (2 units) -->
        <body name="left_crank" pos="-2 0 0.5">
            <joint name="left_crank_joint" type="hinge" axis="0 0 1" damping="0.15"/>
            <geom type="capsule" fromto="0 0 0  1.4 1.4 0" size="0.05"
                  material="link1_mat" mass="0.6"/>
            <geom type="sphere" pos="1.4 1.4 0" size="0.08" rgba="1 0.5 0 1"
                  contype="0" conaffinity="0"/>

            <!-- Coupler attached to left crank (2.5 units) -->
            <body name="coupler" pos="1.4 1.4 0">
                <joint name="coupler_left_joint" type="hinge" axis="0 0 1" damping="0.1"/>
                <geom type="capsule" fromto="0 0 0  2.0 -0.5 0" size="0.05"
                      material="link2_mat" mass="0.7"/>

                <!-- Foot/tracer point -->
                <body name="foot" pos="1.0 -0.25 0">
                    <geom type="sphere" size="0.15" rgba="1 0.1 0.1 1" mass="0.5"/>
                    <geom type="cylinder" fromto="0 0 0  0 0 -0.7" size="0.03"
                          rgba="1 0 0 0.7" contype="0" conaffinity="0"/>
                </body>
            </body>
        </body>

        <!-- Right rocker (2 units) -->
        <body name="right_rocker" pos="2 0 0.5">
            <joint name="right_rocker_joint" type="hinge" axis="0 0 1" damping="0.15"/>
            <geom type="capsule" fromto="0 0 0  -1.4 1.4 0" size="0.05"
                  material="link3_mat" mass="0.6"/>
            <geom type="sphere" pos="-1.4 1.4 0" size="0.08" rgba="0.5 0 1 1"
                  contype="0" conaffinity="0"/>
        </body>
    </worldbody>

    <equality>
        <!-- Connect coupler to right rocker -->
        <connect body1="coupler" body2="right_rocker" anchor="2.0 -0.5 0"/>
    </equality>

    <actuator>
        <motor name="left_crank_motor" joint="left_crank_joint" gear="20"
               ctrllimited="true" ctrlrange="-6 6"/>
    </actuator>
</mujoco>
"""


def generate_pantograph_xml(scale_factor: float = 2.0) -> str:
    """
    Generate a pantograph mechanism (geometric scaling/copying).

    Parameters:
    -----------
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


def generate_delta_robot_xml(
    base_radius: float = 2.0, platform_radius: float = 0.5
) -> str:
    """
    Generate a 3-DOF Delta parallel robot (high-speed pick-and-place).

    Parameters:
    -----------
    base_radius : float
        Radius of the fixed base triangle
    platform_radius : float
        Radius of the moving platform triangle
    """
    arm_length = 2.0
    forearm_length = 3.0

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

        <!-- Arm 1 (0 degrees) -->
        <body name="base1" pos="{base_radius} 0 2">
            <geom type="sphere" size="0.15" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
            <body name="arm1" pos="0 0 0">
                <joint name="joint1" type="hinge" axis="0 1 0" range="-120 120" damping="1.0"/>
                <geom type="capsule" fromto="0 0 0  0 0 {-arm_length}" size="0.08"
                      material="arm_mat" mass="1.0"/>

                <!-- Forearm 1 (parallel linkage) -->
                <body name="forearm1a" pos="0 {-0.15} {-arm_length}">
                    <joint name="elbow1a" type="hinge" axis="0 1 0" damping="0.5"/>
                    <geom type="capsule" fromto="0 0 0  0 0 {-forearm_length}" size="0.05"
                          material="forearm_mat" mass="0.5"/>
                </body>
                <body name="forearm1b" pos="0 {0.15} {-arm_length}">
                    <joint name="elbow1b" type="hinge" axis="0 1 0" damping="0.5"/>
                    <geom type="capsule" fromto="0 0 0  0 0 {-forearm_length}" size="0.05"
                          material="forearm_mat" mass="0.5"/>
                </body>
            </body>
        </body>

        <!-- Arm 2 (120 degrees) -->
        <body name="base2" pos="{base_radius * np.cos(2 * np.pi / 3)} {base_radius * np.sin(2 * np.pi / 3)} 2">
            <geom type="sphere" size="0.15" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
            <body name="arm2" pos="0 0 0" euler="0 0 120">
                <joint name="joint2" type="hinge" axis="0 1 0" range="-120 120" damping="1.0"/>
                <geom type="capsule" fromto="0 0 0  0 0 {-arm_length}" size="0.08"
                      material="arm_mat" mass="1.0"/>

                <body name="forearm2a" pos="0 {-0.15} {-arm_length}">
                    <joint name="elbow2a" type="hinge" axis="0 1 0" damping="0.5"/>
                    <geom type="capsule" fromto="0 0 0  0 0 {-forearm_length}" size="0.05"
                          material="forearm_mat" mass="0.5"/>
                </body>
                <body name="forearm2b" pos="0 {0.15} {-arm_length}">
                    <joint name="elbow2b" type="hinge" axis="0 1 0" damping="0.5"/>
                    <geom type="capsule" fromto="0 0 0  0 0 {-forearm_length}" size="0.05"
                          material="forearm_mat" mass="0.5"/>
                </body>
            </body>
        </body>

        <!-- Arm 3 (240 degrees) -->
        <body name="base3" pos="{base_radius * np.cos(4 * np.pi / 3)} {base_radius * np.sin(4 * np.pi / 3)} 2">
            <geom type="sphere" size="0.15" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
            <body name="arm3" pos="0 0 0" euler="0 0 240">
                <joint name="joint3" type="hinge" axis="0 1 0" range="-120 120" damping="1.0"/>
                <geom type="capsule" fromto="0 0 0  0 0 {-arm_length}" size="0.08"
                      material="arm_mat" mass="1.0"/>

                <body name="forearm3a" pos="0 {-0.15} {-arm_length}">
                    <joint name="elbow3a" type="hinge" axis="0 1 0" damping="0.5"/>
                    <geom type="capsule" fromto="0 0 0  0 0 {-forearm_length}" size="0.05"
                          material="forearm_mat" mass="0.5"/>
                </body>
                <body name="forearm3b" pos="0 {0.15} {-arm_length}">
                    <joint name="elbow3b" type="hinge" axis="0 1 0" damping="0.5"/>
                    <geom type="capsule" fromto="0 0 0  0 0 {-forearm_length}" size="0.05"
                          material="forearm_mat" mass="0.5"/>
                </body>
            </body>
        </body>

        <!-- End effector platform -->
        <body name="platform" pos="0 0 {2 - arm_length - forearm_length}">
            <freejoint/>
            <geom type="cylinder" size="{platform_radius} 0.1" material="platform_mat" mass="0.5"/>
            <geom type="sphere" pos="0 0 -0.3" size="0.2" rgba="1 0.5 0 1" mass="0.3"/>
        </body>
    </worldbody>

    <equality>
        <!-- Connect forearms to platform -->
        <connect body1="forearm1a" body2="platform" anchor="0 0 {-forearm_length}"/>
        <connect body1="forearm1b" body2="platform" anchor="0 0 {-forearm_length}"/>
        <connect body1="forearm2a" body2="platform" anchor="0 0 {-forearm_length}"/>
        <connect body1="forearm2b" body2="platform" anchor="0 0 {-forearm_length}"/>
        <connect body1="forearm3a" body2="platform" anchor="0 0 {-forearm_length}"/>
        <connect body1="forearm3b" body2="platform" anchor="0 0 {-forearm_length}"/>

        <!-- Keep forearm pairs parallel -->
        <joint joint1="elbow1a" joint2="elbow1b" polycoef="0 1 0 0 0"/>
        <joint joint1="elbow2a" joint2="elbow2b" polycoef="0 1 0 0 0"/>
        <joint joint1="elbow3a" joint2="elbow3b" polycoef="0 1 0 0 0"/>
    </equality>

    <actuator>
        <motor name="motor1" joint="joint1" gear="50" ctrllimited="true" ctrlrange="-10 10"/>
        <motor name="motor2" joint="joint2" gear="50" ctrllimited="true" ctrlrange="-10 10"/>
        <motor name="motor3" joint="joint3" gear="50" ctrllimited="true" ctrlrange="-10 10"/>
    </actuator>
</mujoco>
"""


def generate_five_bar_parallel_xml(link_length: float = 1.5) -> str:
    """
    Generate a 5-bar parallel manipulator (2-DOF planar robot).

    Parameters:
    -----------
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


def generate_stewart_platform_xml(
    base_radius: float = 1.5, platform_radius: float = 0.8
) -> str:
    """
    Generate a Stewart platform (6-DOF parallel manipulator).

    Parameters:
    -----------
    base_radius : float
        Radius of the base hexagon
    platform_radius : float
        Radius of the platform hexagon
    """
    leg_min = 1.5
    leg_max = 3.0

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
            <geom type="sphere" pos="{base_radius * np.cos(0)} {base_radius * np.sin(0)} 0.15"
                  size="0.1" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
            <geom type="sphere" pos="{base_radius * np.cos(np.pi / 3)} {base_radius * np.sin(np.pi / 3)} 0.15"
                  size="0.1" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
            <geom type="sphere" pos="{base_radius * np.cos(2 * np.pi / 3)} {base_radius * np.sin(2 * np.pi / 3)} 0.15"
                  size="0.1" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
            <geom type="sphere" pos="{base_radius * np.cos(np.pi)} {base_radius * np.sin(np.pi)} 0.15"
                  size="0.1" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
            <geom type="sphere" pos="{base_radius * np.cos(4 * np.pi / 3)} {base_radius * np.sin(4 * np.pi / 3)} 0.15"
                  size="0.1" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
            <geom type="sphere" pos="{base_radius * np.cos(5 * np.pi / 3)} {base_radius * np.sin(5 * np.pi / 3)} 0.15"
                  size="0.1" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

            <!-- Leg 1 -->
            <body name="leg1_lower" pos="{base_radius * np.cos(0)} {base_radius * np.sin(0)} 0.15">
                <joint name="leg1_base_ball" type="ball" damping="0.5"/>
                <geom type="capsule" fromto="0 0 0  0 0 1.2" size="0.05" material="leg_mat" mass="0.5"/>

                <body name="leg1_upper" pos="0 0 1.2">
                    <joint name="leg1_extend" type="slide" axis="0 0 1" range="{leg_min - 1.2} {leg_max - 1.2}" damping="2.0"/>
                    <geom type="capsule" fromto="0 0 0  0 0 1.0" size="0.045"
                          rgba="0.1 0.6 0.8 1" mass="0.4"/>
                </body>
            </body>

            <!-- Leg 2 -->
            <body name="leg2_lower" pos="{base_radius * np.cos(np.pi / 3)} {base_radius * np.sin(np.pi / 3)} 0.15">
                <joint name="leg2_base_ball" type="ball" damping="0.5"/>
                <geom type="capsule" fromto="0 0 0  0 0 1.2" size="0.05" material="leg_mat" mass="0.5"/>

                <body name="leg2_upper" pos="0 0 1.2">
                    <joint name="leg2_extend" type="slide" axis="0 0 1" range="{leg_min - 1.2} {leg_max - 1.2}" damping="2.0"/>
                    <geom type="capsule" fromto="0 0 0  0 0 1.0" size="0.045"
                          rgba="0.1 0.6 0.8 1" mass="0.4"/>
                </body>
            </body>

            <!-- Leg 3 -->
            <body name="leg3_lower" pos="{base_radius * np.cos(2 * np.pi / 3)} {base_radius * np.sin(2 * np.pi / 3)} 0.15">
                <joint name="leg3_base_ball" type="ball" damping="0.5"/>
                <geom type="capsule" fromto="0 0 0  0 0 1.2" size="0.05" material="leg_mat" mass="0.5"/>

                <body name="leg3_upper" pos="0 0 1.2">
                    <joint name="leg3_extend" type="slide" axis="0 0 1" range="{leg_min - 1.2} {leg_max - 1.2}" damping="2.0"/>
                    <geom type="capsule" fromto="0 0 0  0 0 1.0" size="0.045"
                          rgba="0.1 0.6 0.8 1" mass="0.4"/>
                </body>
            </body>

            <!-- Leg 4 -->
            <body name="leg4_lower" pos="{base_radius * np.cos(np.pi)} {base_radius * np.sin(np.pi)} 0.15">
                <joint name="leg4_base_ball" type="ball" damping="0.5"/>
                <geom type="capsule" fromto="0 0 0  0 0 1.2" size="0.05" material="leg_mat" mass="0.5"/>

                <body name="leg4_upper" pos="0 0 1.2">
                    <joint name="leg4_extend" type="slide" axis="0 0 1" range="{leg_min - 1.2} {leg_max - 1.2}" damping="2.0"/>
                    <geom type="capsule" fromto="0 0 0  0 0 1.0" size="0.045"
                          rgba="0.1 0.6 0.8 1" mass="0.4"/>
                </body>
            </body>

            <!-- Leg 5 -->
            <body name="leg5_lower" pos="{base_radius * np.cos(4 * np.pi / 3)} {base_radius * np.sin(4 * np.pi / 3)} 0.15">
                <joint name="leg5_base_ball" type="ball" damping="0.5"/>
                <geom type="capsule" fromto="0 0 0  0 0 1.2" size="0.05" material="leg_mat" mass="0.5"/>

                <body name="leg5_upper" pos="0 0 1.2">
                    <joint name="leg5_extend" type="slide" axis="0 0 1" range="{leg_min - 1.2} {leg_max - 1.2}" damping="2.0"/>
                    <geom type="capsule" fromto="0 0 0  0 0 1.0" size="0.045"
                          rgba="0.1 0.6 0.8 1" mass="0.4"/>
                </body>
            </body>

            <!-- Leg 6 -->
            <body name="leg6_lower" pos="{base_radius * np.cos(5 * np.pi / 3)} {base_radius * np.sin(5 * np.pi / 3)} 0.15">
                <joint name="leg6_base_ball" type="ball" damping="0.5"/>
                <geom type="capsule" fromto="0 0 0  0 0 1.2" size="0.05" material="leg_mat" mass="0.5"/>

                <body name="leg6_upper" pos="0 0 1.2">
                    <joint name="leg6_extend" type="slide" axis="0 0 1" range="{leg_min - 1.2} {leg_max - 1.2}" damping="2.0"/>
                    <geom type="capsule" fromto="0 0 0  0 0 1.0" size="0.045"
                          rgba="0.1 0.6 0.8 1" mass="0.4"/>
                </body>
            </body>
        </body>

        <!-- Moving platform -->
        <body name="platform" pos="0 0 2.5">
            <freejoint/>
            <geom type="cylinder" size="{platform_radius} 0.12" material="platform_mat" mass="2.0"/>
            <geom type="sphere" pos="0 0 0.2" size="0.15" rgba="1 0.5 0 1" mass="0.5"/>

            <!-- Platform attachment points -->
            <geom type="sphere"
                  pos="{platform_radius * np.cos(np.pi / 6)} {platform_radius * np.sin(np.pi / 6)} -0.12"
                  size="0.08" rgba="0.8 0.6 0.1 1" contype="0" conaffinity="0"/>
            <geom type="sphere"
                  pos="{platform_radius * np.cos(np.pi / 2)} {platform_radius * np.sin(np.pi / 2)} -0.12"
                  size="0.08" rgba="0.8 0.6 0.1 1" contype="0" conaffinity="0"/>
            <geom type="sphere"
                  pos="{platform_radius * np.cos(5 * np.pi / 6)} {platform_radius * np.sin(5 * np.pi / 6)} -0.12"
                  size="0.08" rgba="0.8 0.6 0.1 1" contype="0" conaffinity="0"/>
            <geom type="sphere"
                  pos="{platform_radius * np.cos(7 * np.pi / 6)} {platform_radius * np.sin(7 * np.pi / 6)} -0.12"
                  size="0.08" rgba="0.8 0.6 0.1 1" contype="0" conaffinity="0"/>
            <geom type="sphere"
                  pos="{platform_radius * np.cos(3 * np.pi / 2)} {platform_radius * np.sin(3 * np.pi / 2)} -0.12"
                  size="0.08" rgba="0.8 0.6 0.1 1" contype="0" conaffinity="0"/>
            <geom type="sphere"
                  pos="{platform_radius * np.cos(11 * np.pi / 6)} {platform_radius * np.sin(11 * np.pi / 6)} -0.12"
                  size="0.08" rgba="0.8 0.6 0.1 1" contype="0" conaffinity="0"/>
        </body>
    </worldbody>

    <equality>
        <!-- Connect leg tops to platform -->
        <connect body1="leg1_upper" body2="platform" anchor="0 0 1.0"
                 relpose="{platform_radius * np.cos(np.pi / 6)}
                          {platform_radius * np.sin(np.pi / 6)} -0.12  1 0 0 0"/>
        <connect body1="leg2_upper" body2="platform" anchor="0 0 1.0"
                 relpose="{platform_radius * np.cos(np.pi / 2)}
                          {platform_radius * np.sin(np.pi / 2)} -0.12  1 0 0 0"/>
        <connect body1="leg3_upper" body2="platform" anchor="0 0 1.0"
                 relpose="{platform_radius * np.cos(5 * np.pi / 6)}
                          {platform_radius * np.sin(5 * np.pi / 6)} -0.12  1 0 0 0"/>
        <connect body1="leg4_upper" body2="platform" anchor="0 0 1.0"
                 relpose="{platform_radius * np.cos(7 * np.pi / 6)}
                          {platform_radius * np.sin(7 * np.pi / 6)} -0.12  1 0 0 0"/>
        <connect body1="leg5_upper" body2="platform" anchor="0 0 1.0"
                 relpose="{platform_radius * np.cos(3 * np.pi / 2)}
                          {platform_radius * np.sin(3 * np.pi / 2)} -0.12  1 0 0 0"/>
        <connect body1="leg6_upper" body2="platform" anchor="0 0 1.0"
                 relpose="{platform_radius * np.cos(11 * np.pi / 6)}
                          {platform_radius * np.sin(11 * np.pi / 6)} -0.12  1 0 0 0"/>
    </equality>

    <actuator>
        <motor name="leg1_motor" joint="leg1_extend" gear="100"
               ctrllimited="true" ctrlrange="-15 15"/>
        <motor name="leg2_motor" joint="leg2_extend" gear="100"
               ctrllimited="true" ctrlrange="-15 15"/>
        <motor name="leg3_motor" joint="leg3_extend" gear="100"
               ctrllimited="true" ctrlrange="-15 15"/>
        <motor name="leg4_motor" joint="leg4_extend" gear="100"
               ctrllimited="true" ctrlrange="-15 15"/>
        <motor name="leg5_motor" joint="leg5_extend" gear="100"
               ctrllimited="true" ctrlrange="-15 15"/>
        <motor name="leg6_motor" joint="leg6_extend" gear="100"
               ctrllimited="true" ctrlrange="-15 15"/>
    </actuator>
</mujoco>
"""


def generate_watt_linkage_xml() -> str:
    """
    Generate Watt's linkage (approximate straight-line mechanism).

    Used historically in steam engines for straight-line guidance.
    """
    return f"""
<mujoco model="watt_linkage">
    <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker"
                 rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="1 1"/>
        <material name="link_mat" rgba="0.7 0.3 0.1 1"/>
        <material name="tracer_mat" rgba="1 0.1 0.1 1"/>
    </asset>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1" material="grid"/>

        <!-- Ground link -->
        <geom type="cylinder" fromto="-2.5 0 0.5  2.5 0 0.5" size="0.04"
              rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>

        <!-- Left pivot -->
        <geom name="left_pivot" type="sphere" pos="-2.5 0 0.5" size="0.12"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

        <!-- Right pivot -->
        <geom name="right_pivot" type="sphere" pos="2.5 0 0.5" size="0.12"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

        <!-- Left oscillating link -->
        <body name="left_link" pos="-2.5 0 0.5">
            <joint name="left_joint" type="hinge" axis="0 0 1" damping="0.2"/>
            <geom type="capsule" fromto="0 0 0  2.0 1.5 0" size="0.06"
                  material="link_mat" mass="0.7"/>
            <geom type="sphere" pos="2.0 1.5 0" size="0.1" rgba="1 0.6 0.2 1"
                  contype="0" conaffinity="0"/>

            <!-- Central link attached to left -->
            <body name="central_link" pos="2.0 1.5 0">
                <joint name="central_left_joint" type="hinge" axis="0 0 1" damping="0.15"/>
                <geom type="capsule" fromto="0 0 0  2.0 -0.5 0" size="0.055"
                      rgba="0.1 0.7 0.5 1" mass="0.6"/>

                <!-- Tracer point at center of central link -->
                <body name="tracer" pos="1.0 -0.25 0">
                    <geom type="sphere" size="0.15" material="tracer_mat" mass="0.4"/>
                    <geom type="cylinder" fromto="0 0 0  0 0 -0.8" size="0.03"
                          rgba="1 0 0 0.6" contype="0" conaffinity="0"/>
                </body>
            </body>
        </body>

        <!-- Right oscillating link -->
        <body name="right_link" pos="2.5 0 0.5">
            <joint name="right_joint" type="hinge" axis="0 0 1" damping="0.2"/>
            <geom type="capsule" fromto="0 0 0  -2.0 1.5 0" size="0.06"
                  material="link_mat" mass="0.7"/>
            <geom type="sphere" pos="-2.0 1.5 0" size="0.1" rgba="1 0.6 0.2 1"
                  contype="0" conaffinity="0"/>
        </body>
    </worldbody>

    <equality>
        <!-- Connect central link to right oscillator -->
        <connect body1="central_link" body2="right_link" anchor="2.0 -0.5 0"/>
    </equality>

    <actuator>
        <motor name="left_motor" joint="left_joint" gear="20"
               ctrllimited="true" ctrlrange="-6 6"/>
    </actuator>
</mujoco>
"""


def generate_oldham_coupling_xml(offset: float = 0.5) -> str:
    """
    Generate Oldham coupling (parallel shaft coupler with offset).

    Parameters:
    -----------
    offset : float
        Offset distance between the two parallel shafts
    """
    return f"""
<mujoco model="oldham_coupling">
    <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker"
                 rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="1 1"/>
        <material name="input_mat" rgba="0.9 0.1 0.1 1"/>
        <material name="middle_mat" rgba="0.1 0.7 0.1 1"/>
        <material name="output_mat" rgba="0.1 0.1 0.9 1"/>
    </asset>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1" material="grid"/>

        <!-- Input shaft -->
        <body name="input_shaft" pos="0 0 1.5">
            <joint name="input_joint" type="hinge" axis="0 0 1" damping="0.3"/>
            <geom type="cylinder" fromto="0 0 -0.8  0 0 0" size="0.08"
                  rgba="0.7 0.7 0.7 1" mass="0.5"/>
            <geom type="cylinder" size="0.4 0.1" material="input_mat" mass="1.0"/>
            <!-- Tongue slot (simplified as box) -->
            <geom type="box" pos="0 0 0.12" size="0.35 0.08 0.03"
                  rgba="0.9 0.3 0.3 0.7" contype="0" conaffinity="0"/>
        </body>

        <!-- Middle disc (floating, has two perpendicular slots) -->
        <body name="middle_disc" pos="0 0 1.7">
            <freejoint/>
            <geom type="cylinder" size="0.45 0.08" material="middle_mat" mass="0.8"/>
            <!-- Tongues/slots visualization -->
            <geom type="box" pos="0 0 0.09" size="0.4 0.06 0.02"
                  rgba="0.3 0.9 0.3 0.7" contype="0" conaffinity="0"/>
            <geom type="box" pos="0 0 0.09" size="0.06 0.4 0.02"
                  rgba="0.3 0.9 0.3 0.7" contype="0" conaffinity="0"/>
        </body>

        <!-- Output shaft (offset) -->
        <body name="output_shaft" pos="{offset} 0 1.9">
            <joint name="output_joint" type="hinge" axis="0 0 1" damping="0.3"/>
            <geom type="cylinder" size="0.4 0.1" material="output_mat" mass="1.0"/>
            <geom type="cylinder" fromto="0 0 0.1  0 0 0.9" size="0.08"
                  rgba="0.7 0.7 0.7 1" mass="0.5"/>
            <!-- Tongue slot -->
            <geom type="box" pos="0 0 -0.12" size="0.08 0.35 0.03"
                  rgba="0.3 0.3 0.9 0.7" contype="0" conaffinity="0"/>
        </body>
    </worldbody>

    <equality>
        <!-- Simplified coupling constraints -->
        <weld body1="input_shaft" body2="middle_disc"
              solref="0.002 1" solimp="0.9 0.95 0.001" relpose="0 0 0.2  1 0 0 0"/>
        <weld body1="middle_disc" body2="output_shaft"
              solref="0.002 1" solimp="0.9 0.95 0.001" relpose="{-offset} 0 0.2  1 0 0 0"/>
    </equality>

    <actuator>
        <motor name="input_motor" joint="input_joint" gear="25"
               ctrllimited="true" ctrlrange="-8 8"/>
    </actuator>
</mujoco>
"""


# Catalog of all available mechanisms for GUI integration
LINKAGE_CATALOG = {
    "Four-Bar: Grashof Crank-Rocker": {
        "xml": generate_four_bar_linkage_xml(link_type="grashof_crank_rocker"),
        "actuators": ["crank_motor"],
        "category": "Four-Bar Linkages",
        "description": "Classic crank-rocker mechanism (Grashof condition satisfied)",
    },
    "Four-Bar: Double Crank": {
        "xml": generate_four_bar_linkage_xml(link_type="grashof_double_crank"),
        "actuators": ["crank_motor"],
        "category": "Four-Bar Linkages",
        "description": "Both output links can rotate fully (Grashof)",
    },
    "Four-Bar: Double Rocker": {
        "xml": generate_four_bar_linkage_xml(link_type="grashof_double_rocker"),
        "actuators": ["crank_motor"],
        "category": "Four-Bar Linkages",
        "description": "Both output links oscillate",
    },
    "Four-Bar: Parallelogram": {
        "xml": generate_four_bar_linkage_xml(link_type="parallel"),
        "actuators": ["crank_motor"],
        "category": "Four-Bar Linkages",
        "description": "Special case with parallel opposite links",
    },
    "Slider-Crank (Horizontal)": {
        "xml": generate_slider_crank_xml(orientation="horizontal"),
        "actuators": ["crank_motor"],
        "category": "Slider Mechanisms",
        "description": "Piston engine mechanism, converts rotation to linear motion",
    },
    "Slider-Crank (Vertical)": {
        "xml": generate_slider_crank_xml(orientation="vertical"),
        "actuators": ["crank_motor"],
        "category": "Slider Mechanisms",
        "description": "Vertical orientation slider-crank",
    },
    "Scotch Yoke": {
        "xml": generate_scotch_yoke_xml(),
        "actuators": ["crank_motor"],
        "category": "Slider Mechanisms",
        "description": "Perfect simple harmonic motion generator",
    },
    "Geneva Mechanism": {
        "xml": generate_geneva_mechanism_xml(),
        "actuators": ["drive_motor"],
        "category": "Special Mechanisms",
        "description": "Intermittent motion converter (used in film projectors)",
    },
    "Oldham Coupling": {
        "xml": generate_oldham_coupling_xml(),
        "actuators": ["input_motor"],
        "category": "Special Mechanisms",
        "description": "Couples parallel shafts with offset",
    },
    "Peaucellier-Lipkin Linkage": {
        "xml": generate_peaucellier_linkage_xml(),
        "actuators": ["drive_motor"],
        "category": "Straight-Line Mechanisms",
        "description": "Exact straight-line mechanism (mathematical perfection)",
    },
    "Chebyshev Linkage": {
        "xml": generate_chebyshev_linkage_xml(),
        "actuators": ["left_crank_motor"],
        "category": "Straight-Line Mechanisms",
        "description": "Approximate straight-line (walking mechanism)",
    },
    "Watt's Linkage": {
        "xml": generate_watt_linkage_xml(),
        "actuators": ["left_motor"],
        "category": "Straight-Line Mechanisms",
        "description": "Steam engine straight-line guidance (historical)",
    },
    "Pantograph": {
        "xml": generate_pantograph_xml(),
        "actuators": ["arm1_motor"],
        "category": "Scaling Mechanisms",
        "description": "Geometric scaling and copying mechanism",
    },
    "Delta Robot (3-DOF Parallel)": {
        "xml": generate_delta_robot_xml(),
        "actuators": ["motor1", "motor2", "motor3"],
        "category": "Parallel Mechanisms",
        "description": "High-speed pick-and-place robot",
    },
    "5-Bar Parallel Manipulator": {
        "xml": generate_five_bar_parallel_xml(),
        "actuators": ["left_motor", "right_motor"],
        "category": "Parallel Mechanisms",
        "description": "2-DOF planar parallel robot",
    },
    "Stewart Platform (6-DOF)": {
        "xml": generate_stewart_platform_xml(),
        "actuators": [
            "leg1_motor",
            "leg2_motor",
            "leg3_motor",
            "leg4_motor",
            "leg5_motor",
            "leg6_motor",
        ],
        "category": "Parallel Mechanisms",
        "description": "Flight simulator platform, full 6-DOF control",
    },
}


# Public API
__all__ = [
    "LINKAGE_CATALOG",
    "generate_chebyshev_linkage_xml",
    "generate_delta_robot_xml",
    "generate_five_bar_parallel_xml",
    "generate_four_bar_linkage_xml",
    "generate_geneva_mechanism_xml",
    "generate_oldham_coupling_xml",
    "generate_pantograph_xml",
    "generate_peaucellier_linkage_xml",
    "generate_scotch_yoke_xml",
    "generate_slider_crank_xml",
    "generate_stewart_platform_xml",
    "generate_watt_linkage_xml",
]
