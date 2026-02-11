"""MJCF pendulum models for golf swing analysis.

Includes chaotic driven pendulum, double/triple pendulum golf analogues,
two-link inclined plane with universal joint, and gimbal joint demonstration.
"""

from __future__ import annotations

from src.shared.python import constants

GRAVITY_M_S2 = float(constants.GRAVITY_M_S2)
DEFAULT_TIME_STEP = float(constants.DEFAULT_TIME_STEP)


CHAOTIC_PENDULUM_XML = rf"""<mujoco model="chaotic_driven_pendulum">
  <option timestep="{DEFAULT_TIME_STEP}" gravity="0 0 -{GRAVITY_M_S2}"
          integrator="RK4"/>

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
  <option timestep="{DEFAULT_TIME_STEP}" gravity="0 0 -{GRAVITY_M_S2}"
          integrator="RK4"/>

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
  <option timestep="{DEFAULT_TIME_STEP}" gravity="0 0 -{GRAVITY_M_S2}"
          integrator="RK4"/>

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


# ==============================================================================
# TWO-LINK INCLINED PLANE MODEL WITH UNIVERSAL JOINT AT WRIST
# ==============================================================================
TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML = rf"""
<mujoco model="two_link_inclined_universal">
  <option timestep="{DEFAULT_TIME_STEP}" gravity="0 0 -{GRAVITY_M_S2}"
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
  <option timestep="{DEFAULT_TIME_STEP}" gravity="0 0 -{GRAVITY_M_S2}"
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
