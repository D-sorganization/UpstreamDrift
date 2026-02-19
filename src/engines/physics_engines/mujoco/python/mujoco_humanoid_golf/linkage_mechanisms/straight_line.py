# ruff: noqa: E501
"""Straight-line mechanism generators for MuJoCo.

Includes Peaucellier-Lipkin, Chebyshev, and Watt linkages.

Extracted from __init__.py for SRP (#1485).
"""

from src.shared.python.core.constants import GRAVITY_M_S2


def generate_peaucellier_linkage_xml(scale: float = 1.0) -> str:
    """Generate Peaucellier-Lipkin linkage (exact straight-line mechanism).

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
    """Generate Chebyshev linkage (approximate straight-line mechanism).

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


def generate_watt_linkage_xml() -> str:
    """Generate Watt's linkage (approximate straight-line mechanism).

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
