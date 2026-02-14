"""MJCF XML model definitions.

Split from head_models.py for maintainability.
"""

from __future__ import annotations

from src.shared.python.core.constants import (
    DEFAULT_TIME_STEP,
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
)

# Convert to float for use in f-strings
_BALL_MASS = float(GOLF_BALL_MASS_KG)
_BALL_RADIUS = float(GOLF_BALL_RADIUS_M)
_BALL_RADIUS_INNER = _BALL_RADIUS * 0.998
_TIME_STEP = float(DEFAULT_TIME_STEP)

CHAOTIC_PENDULUM_XML = rf"""<mujoco model="chaotic_driven_pendulum">
  <option timestep="{_TIME_STEP}" gravity="0 0 -{GRAVITY_M_S2}" integrator="RK4"/>

  <visual>
    <global offwidth="1024" offheight="1024"/>
    <map znear="0.01" zfar="10"/>
    <headlight diffuse="0.8 0.8 0.8"/>
  </visual>

  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- Camera positioned to view pendulum motion -->
    <camera name="side" pos="-2.5 0 0.8" euler="0 0 0"/>
    <camera name="front" pos="0 -2.5 0.8" euler="0 0 1.57"/>

    <!-- Driven base (oscillates horizontally to induce chaos) -->
    <body name="drive_base" pos="0 0 1.0">
      <!-- Horizontal slider for base excitation (x-axis) -->
      <joint name="base_drive_x" type="slide" axis="1 0 0"
             limited="false" damping="0.1"/>

      <!-- Vertical support column -->
      <geom name="support_column" type="cylinder"
            fromto="0 0 0 0 0 -0.3" size="0.02"
            rgba="0.5 0.5 0.5 1"/>

      <!-- Pivot point visualization -->
      <geom name="pivot" type="sphere" size="0.03"
            rgba="0.2 0.2 0.8 1" pos="0 0 0"/>

      <!-- Pendulum arm -->
      <body name="pendulum" pos="0 0 0">
        <joint name="pendulum_hinge" type="hinge" axis="0 1 0"
               range="-3 3" limited="false" damping="0.05"/>

        <!-- Pendulum rod -->
        <geom name="pendulum_rod" type="capsule"
              fromto="0 0 0 0 0 -0.8" size="0.015"
              rgba="0.8 0.2 0.2 1" mass="0.05"/>

        <!-- Pendulum bob (concentrated mass at end) -->
        <body name="bob" pos="0 0 -0.8">
          <geom name="bob_geom" type="sphere" size="0.08"
                rgba="0.9 0.1 0.1 1" mass="1.0"/>
        </body>
      </body>
    </body>

    <!-- Reference markers for visualization -->
    <geom name="equilibrium_marker" type="cylinder"
          fromto="0 0 0.2 0 0 0.25" size="0.005"
          rgba="0 1 0 0.3"/>
  </worldbody>

  <actuator>
    <!-- Actuator to drive the base oscillation (for forcing) -->
    <motor name="base_drive_motor" joint="base_drive_x" gear="10"
           ctrllimited="true" ctrlrange="-20 20"/>

    <!-- Actuator for pendulum control (for control experiments) -->
    <motor name="pendulum_motor" joint="pendulum_hinge" gear="5"
           ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


DOUBLE_PENDULUM_XML = rf"""    <mujoco model="golf_double_pendulum">
  <option timestep="0.001" gravity="0 0 -{GRAVITY_M_S2}" integrator="RK4"/>

  <visual>
    <global offwidth="1024" offheight="1024"/>
    <map znear="0.01" zfar="20"/>
    <headlight diffuse="0.8 0.8 0.8"/>
  </visual>

  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- Side camera for a classic golf view -->
    <camera name="side" pos="-3 0 1.3" euler="0 20 0"/>

    <!-- Shoulder at about chest height -->
    <body name="shoulder_body" pos="0 0 1.2">
      <joint name="shoulder" type="hinge" axis="0 0 1" limited="false"/>

      <!-- Upper arm / combined arms segment -->
      <geom name="upper_arm" type="capsule"
            fromto="0 0 0 0.4 0 0" size="0.03"
            rgba="0.8 0.2 0.2 1"/>

      <!-- Club body directly from shoulder (double pendulum) -->
      <body name="club_body" pos="0.4 0 0">
        <joint name="wrist" type="hinge" axis="0 0 1" limited="false"/>

        <!-- Golf club shaft -->
        <geom name="club_shaft" type="capsule"
              fromto="0 0 0 1.0 0 0" size="0.015"
              rgba="0.2 0.8 0.2 1"/>

        <!-- Clubhead as a small box at the end -->
        <geom name="club_head" type="box" size="0.05 0.03 0.02"
              pos="1.0 0 0" rgba="0.1 0.1 0.1 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="shoulder_motor" joint="shoulder" gear="1"/>
    <motor name="wrist_motor" joint="wrist" gear="1"/>
  </actuator>
</mujoco>
"""


TRIPLE_PENDULUM_XML = rf"""    <mujoco model="golf_triple_pendulum">
  <option timestep="0.001" gravity="0 0 -{GRAVITY_M_S2}" integrator="RK4"/>

  <visual>
    <global offwidth="1024" offheight="1024"/>
    <map znear="0.01" zfar="20"/>
    <headlight diffuse="0.8 0.8 0.8"/>
  </visual>

  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- Side camera for a classic golf view -->
    <camera name="side" pos="-3 0 1.3" euler="0 20 0"/>

    <!-- Shoulder at about chest height -->
    <body name="shoulder_body" pos="0 0 1.2">
      <joint name="shoulder" type="hinge" axis="0 0 1" limited="false"/>

      <!-- Upper arm -->
      <geom name="upper_arm" type="capsule"
            fromto="0 0 0 0.35 0 0" size="0.03"
            rgba="0.8 0.2 0.2 1"/>

      <body name="forearm_body" pos="0.35 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" limited="false"/>

        <!-- Forearm -->
        <geom name="forearm" type="capsule"
              fromto="0 0 0 0.35 0 0" size="0.025"
              rgba="0.2 0.2 0.8 1"/>

        <body name="club_body" pos="0.35 0 0">
          <joint name="wrist" type="hinge" axis="0 0 1" limited="false"/>

          <!-- Golf club shaft -->
          <geom name="club_shaft" type="capsule"
                fromto="0 0 0 1.0 0 0" size="0.015"
                rgba="0.2 0.8 0.2 1"/>

          <!-- Clubhead -->
          <geom name="club_head" type="box" size="0.05 0.03 0.02"
                pos="1.0 0 0" rgba="0.1 0.1 0.1 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="shoulder_motor" joint="shoulder" gear="1"/>
    <motor name="elbow_motor" joint="elbow" gear="1"/>
    <motor name="wrist_motor" joint="wrist" gear="1"/>
  </actuator>
</mujoco>
"""
