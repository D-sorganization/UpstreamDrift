# ruff: noqa: E501
"""Special mechanism generators for MuJoCo.

Includes Geneva drive and Oldham coupling mechanisms.

Extracted from __init__.py for SRP (#1485).
"""

from src.shared.python.core.constants import GRAVITY_M_S2


def generate_geneva_mechanism_xml(num_slots: int = 6, drive_radius: float = 2.0) -> str:
    """Generate a Geneva mechanism (intermittent motion).

    Parameters
    ----------
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


def generate_oldham_coupling_xml(offset: float = 0.5) -> str:
    """Generate Oldham coupling (parallel shaft coupler with offset).

    Parameters
    ----------
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
