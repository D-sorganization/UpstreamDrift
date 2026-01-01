"""MJCF models for golf swing systems.

Includes simple pendulum models and biomechanically realistic golf swing models
with anthropometric body segments and two-handed grip.
"""

from __future__ import annotations

from typing import cast

from shared.python import constants

GRAVITY_M_S2 = float(constants.GRAVITY_M_S2)

CHAOTIC_PENDULUM_XML = rf"""<mujoco model="chaotic_driven_pendulum">
  <option timestep="0.001" gravity="0 0 -{GRAVITY_M_S2}" integrator="RK4"/>

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


UPPER_BODY_GOLF_SWING_XML = rf"""<mujoco model="golf_upper_body_swing">
  <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}" integrator="RK4"/>

  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>

  <visual>
    <global offwidth="1024" offheight="1024"/>
    <map znear="0.01" zfar="50"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <!-- Define materials for better visualization -->
    <material name="torso_mat" rgba="0.7 0.5 0.4 1"/>
    <material name="arm_left_mat" rgba="0.6 0.4 0.3 1"/>
    <material name="arm_right_mat" rgba="0.6 0.4 0.3 1"/>
    <material name="club_mat" rgba="0.2 0.2 0.2 1"/>
    <material name="grip_mat" rgba="0.1 0.1 0.1 1"/>
    <material name="ground_mat" rgba="0.4 0.6 0.3 1"/>
  </asset>

  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="10 10 0.1" material="ground_mat"/>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>

    <!-- Camera views -->
    <camera name="side" pos="-4 -1.5 1.3" euler="0.1 0 0.3" mode="fixed"/>
    <camera name="front" pos="0 -4 1.3" euler="0.1 0 1.57" mode="fixed"/>
    <camera name="top" pos="0 0 5" euler="0 0 0" mode="fixed"/>

    <!-- Pelvis (fixed base at hip height) -->
    <body name="pelvis" pos="0 0 0.95">
      <geom name="pelvis_geom" type="box" size="0.15 0.08 0.08"
            rgba="0.6 0.5 0.4 1" mass="10"/>
      <geom name="pelvis_marker" type="sphere" size="0.03"
            pos="0 0 0" rgba="1 0 0 0.5"/>

      <!-- Torso connected via spine rotation joint -->
      <body name="torso" pos="0 0 0.1">
        <joint name="spine_rotation" type="hinge" axis="0 0 1"
               range="-1.57 1.57" damping="2.0"/>

        <!-- Torso (spine) segment -->
        <geom name="lower_torso" type="capsule" fromto="0 0 0 0 0 0.25"
              size="0.12" material="torso_mat" mass="15"/>
        <geom name="upper_torso" type="capsule" fromto="0 0 0.25 0 0 0.50"
              size="0.14" material="torso_mat" mass="15"/>

        <!-- Left shoulder -->
        <body name="left_shoulder" pos="-0.2 0 0.50" euler="0 0 0">
          <joint name="left_shoulder_swing" type="hinge" axis="0 1 0"
                 range="-2.0 2.8" damping="1.5"/>
          <joint name="left_shoulder_lift" type="hinge" axis="1 0 0"
                 range="-1.5 1.5" damping="1.5"/>

          <!-- Left upper arm -->
          <geom name="left_upper_arm" type="capsule"
                fromto="0 0 0 0.25 0 -0.05" size="0.035"
                material="arm_left_mat" mass="2.5"/>

          <!-- Left elbow -->
          <body name="left_elbow" pos="0.25 0 -0.05">
            <joint name="left_elbow" type="hinge" axis="0 1 0"
                   range="-2.4 0" damping="1.0"/>

            <!-- Left forearm -->
            <geom name="left_forearm" type="capsule"
                  fromto="0 0 0 0.25 0 0" size="0.03"
                  material="arm_left_mat" mass="1.5"/>

            <!-- Left hand/wrist -->
            <body name="left_hand" pos="0.25 0 0">
              <joint name="left_wrist" type="hinge" axis="0 1 0"
                     range="-1.57 1.57" damping="0.5"/>
              <geom name="left_hand_geom" type="box"
                    size="0.04 0.02 0.08"
                    rgba="0.9 0.7 0.6 1" mass="0.4"/>
            </body>
          </body>
        </body>

        <!-- Right shoulder -->
        <body name="right_shoulder" pos="0.2 0 0.50" euler="0 0 0">
          <joint name="right_shoulder_swing" type="hinge" axis="0 1 0"
                 range="-2.8 2.0" damping="1.5"/>
          <joint name="right_shoulder_lift" type="hinge" axis="1 0 0"
                 range="-1.5 1.5" damping="1.5"/>

          <!-- Right upper arm -->
          <geom name="right_upper_arm" type="capsule"
                fromto="0 0 0 0.25 0 -0.05" size="0.035"
                material="arm_right_mat" mass="2.5"/>

          <!-- Right elbow -->
          <body name="right_elbow" pos="0.25 0 -0.05">
            <joint name="right_elbow" type="hinge" axis="0 1 0"
                   range="-2.4 0" damping="1.0"/>

            <!-- Right forearm -->
            <geom name="right_forearm" type="capsule"
                  fromto="0 0 0 0.25 0 0" size="0.03"
                  material="arm_right_mat" mass="1.5"/>

            <!-- Right hand/wrist connected to club -->
            <body name="right_hand" pos="0.25 0 0">
              <joint name="right_wrist" type="hinge" axis="0 1 0"
                     range="-1.57 1.57" damping="0.5"/>
              <geom name="right_hand_geom" type="box"
                    size="0.04 0.02 0.08"
                    rgba="0.9 0.7 0.6 1" mass="0.4"/>

              <!-- Golf club attached to right hand
                   (left hand connects via equality constraint) -->
              <body name="club" pos="0 0 -0.08" euler="0 -0.3 0">
                <joint name="club_wrist" type="hinge" axis="0 1 0"
                       range="-1.0 1.0" damping="0.3"/>

                <!-- Club grip -->
                <geom name="club_grip" type="capsule"
                      fromto="0 0 0 0 0 -0.25" size="0.015"
                      material="grip_mat" mass="0.1"/>

                <!-- Club shaft -->
                <geom name="club_shaft" type="capsule"
                      fromto="0 0 -0.25 0 0 -1.05" size="0.012"
                      material="club_mat" mass="0.25"/>

                <!-- Club head (driver) -->
                <body name="clubhead" pos="0 0 -1.05">
                  <geom name="clubhead_geom" type="box"
                        size="0.055 0.04 0.03"
                        rgba="0.15 0.15 0.15 1" mass="0.2"/>
                  <!-- Club face indicator -->
                  <geom name="clubface" type="box"
                        size="0.056 0.041 0.005"
                        pos="0 0.041 0"
                        rgba="0.8 0.2 0.2 0.7"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Ball positioned at address -->
    <body name="ball" pos="0 0.1 0.02">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.02135"
            rgba="1 1 1 1" mass="0.04593"
            condim="3" friction="0.8 0.005 0.0001"/>
    </body>
  </worldbody>

  <!-- Equality constraints to connect left hand to club -->
  <equality>
    <weld body1="left_hand" body2="club"
          relpose="0 0 -0.16 1 0 0 0" active="true"/>
  </equality>

  <actuator>
    <!-- Torso -->
    <motor name="spine_rotation_motor" joint="spine_rotation" gear="100"
           ctrllimited="true" ctrlrange="-100 100"/>

    <!-- Left arm -->
    <motor name="left_shoulder_swing_motor" joint="left_shoulder_swing" gear="50"
           ctrllimited="true" ctrlrange="-80 80"/>
    <motor name="left_shoulder_lift_motor" joint="left_shoulder_lift" gear="50"
           ctrllimited="true" ctrlrange="-80 80"/>
    <motor name="left_elbow_motor" joint="left_elbow" gear="40"
           ctrllimited="true" ctrlrange="-60 60"/>
    <motor name="left_wrist_motor" joint="left_wrist" gear="20"
           ctrllimited="true" ctrlrange="-30 30"/>

    <!-- Right arm -->
    <motor name="right_shoulder_swing_motor" joint="right_shoulder_swing" gear="50"
           ctrllimited="true" ctrlrange="-80 80"/>
    <motor name="right_shoulder_lift_motor" joint="right_shoulder_lift" gear="50"
           ctrllimited="true" ctrlrange="-80 80"/>
    <motor name="right_elbow_motor" joint="right_elbow" gear="40"
           ctrllimited="true" ctrlrange="-60 60"/>
    <motor name="right_wrist_motor" joint="right_wrist" gear="20"
           ctrllimited="true" ctrlrange="-30 30"/>

    <!-- Club -->
    <motor name="club_wrist_motor" joint="club_wrist" gear="15"
           ctrllimited="true" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""


FULL_BODY_GOLF_SWING_XML = rf"""
<mujoco model="golf_full_body_swing">
  <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}" integrator="RK4"/>

  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>

  <visual>
    <global offwidth="1024" offheight="1024"/>
    <map znear="0.01" zfar="50"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <material name="torso_mat" rgba="0.7 0.5 0.4 1"/>
    <material name="arm_mat" rgba="0.6 0.4 0.3 1"/>
    <material name="leg_mat" rgba="0.5 0.5 0.6 1"/>
    <material name="club_mat" rgba="0.2 0.2 0.2 1"/>
    <material name="grip_mat" rgba="0.1 0.1 0.1 1"/>
    <material name="ground_mat" rgba="0.4 0.6 0.3 1"/>
  </asset>

  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="10 10 0.1" material="ground_mat"/>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>

    <!-- Camera views -->
    <camera name="side" pos="-5 -2 1.5" euler="0.15 0 0.35" mode="fixed"/>
    <camera name="front" pos="0 -5 1.5" euler="0.15 0 1.57" mode="fixed"/>
    <camera name="top" pos="0 0 6" euler="0 0 0" mode="fixed"/>
    <camera name="follow" pos="-3 -1 2" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>

    <!-- Left foot (grounded) -->
    <body name="left_foot" pos="-0.15 0 0.05">
      <geom name="left_foot_geom" type="box" size="0.08 0.12 0.04"
            rgba="0.3 0.3 0.3 1" mass="1.0"/>

      <!-- Left ankle -->
      <body name="left_shin" pos="0 0 0.05">
        <joint name="left_ankle" type="hinge" axis="1 0 0"
               range="-0.8 0.8" damping="1.5"/>

        <!-- Left shin -->
        <geom name="left_shin_geom" type="capsule"
              fromto="0 0 0 0 0 0.43" size="0.04"
              material="leg_mat" mass="4.0"/>

        <!-- Left knee -->
        <body name="left_thigh" pos="0 0 0.43">
          <joint name="left_knee" type="hinge" axis="1 0 0"
                 range="-2.0 0" damping="2.0"/>

          <!-- Left thigh -->
          <geom name="left_thigh_geom" type="capsule"
                fromto="0 0 0 0 0 0.45" size="0.055"
                material="leg_mat" mass="8.0"/>

          <!-- Left hip connection point -->
          <site name="left_hip_site" pos="0 0 0.45"/>
        </body>
      </body>
    </body>

    <!-- Right foot (grounded) -->
    <body name="right_foot" pos="0.15 0 0.05">
      <geom name="right_foot_geom" type="box" size="0.08 0.12 0.04"
            rgba="0.3 0.3 0.3 1" mass="1.0"/>

      <!-- Right ankle -->
      <body name="right_shin" pos="0 0 0.05">
        <joint name="right_ankle" type="hinge" axis="1 0 0"
               range="-0.8 0.8" damping="1.5"/>

        <!-- Right shin -->
        <geom name="right_shin_geom" type="capsule"
              fromto="0 0 0 0 0 0.43" size="0.04"
              material="leg_mat" mass="4.0"/>

        <!-- Right knee -->
        <body name="right_thigh" pos="0 0 0.43">
          <joint name="right_knee" type="hinge" axis="1 0 0"
                 range="-2.0 0" damping="2.0"/>

          <!-- Right thigh -->
          <geom name="right_thigh_geom" type="capsule"
                fromto="0 0 0 0 0 0.45" size="0.055"
                material="leg_mat" mass="8.0"/>

          <!-- Right hip connection point -->
          <site name="right_hip_site" pos="0 0 0.45"/>
        </body>
      </body>
    </body>

    <!-- Pelvis (connected to both legs via equality constraints) -->
    <body name="pelvis" pos="0 0 0.93">
      <freejoint/>
      <geom name="pelvis_geom" type="box" size="0.16 0.09 0.09"
            rgba="0.6 0.5 0.4 1" mass="12"/>

      <!-- Torso connected via spine joints -->
      <body name="lower_torso" pos="0 0 0.1">
        <joint name="spine_bend" type="hinge" axis="1 0 0"
               range="-0.5 0.8" damping="3.0"/>

        <geom name="lower_torso_geom" type="capsule"
              fromto="0 0 0 0 0 0.25" size="0.12"
              material="torso_mat" mass="12"/>

        <body name="upper_torso" pos="0 0 0.25">
          <joint name="spine_rotation" type="hinge" axis="0 0 1"
                 range="-1.8 1.8" damping="3.0"/>

          <geom name="upper_torso_geom" type="capsule"
                fromto="0 0 0 0 0 0.25" size="0.14"
                material="torso_mat" mass="15"/>

          <!-- Head (for reference) -->
          <body name="head" pos="0 0 0.30">
            <geom name="head_geom" type="sphere" size="0.1"
                  rgba="0.9 0.7 0.6 1" mass="5"/>
          </body>

          <!-- Left shoulder -->
          <body name="left_shoulder" pos="-0.20 0 0.20">
            <joint name="left_shoulder_swing" type="hinge" axis="0 1 0"
                   range="-2.2 3.0" damping="1.5"/>
            <joint name="left_shoulder_lift" type="hinge" axis="1 0 0"
                   range="-1.5 1.5" damping="1.5"/>

            <geom name="left_upper_arm" type="capsule"
                  fromto="0 0 0 0.28 0 -0.06" size="0.035"
                  material="arm_mat" mass="2.5"/>

            <body name="left_elbow" pos="0.28 0 -0.06">
              <joint name="left_elbow" type="hinge" axis="0 1 0"
                     range="-2.5 0" damping="1.0"/>

              <geom name="left_forearm" type="capsule"
                    fromto="0 0 0 0.26 0 0" size="0.03"
                    material="arm_mat" mass="1.5"/>

              <body name="left_hand" pos="0.26 0 0">
                <joint name="left_wrist" type="hinge" axis="0 1 0"
                       range="-1.6 1.6" damping="0.5"/>
                <geom name="left_hand_geom" type="box"
                      size="0.04 0.02 0.09"
                      rgba="0.9 0.7 0.6 1" mass="0.4"/>
              </body>
            </body>
          </body>

          <!-- Right shoulder -->
          <body name="right_shoulder" pos="0.20 0 0.20">
            <joint name="right_shoulder_swing" type="hinge" axis="0 1 0"
                   range="-3.0 2.2" damping="1.5"/>
            <joint name="right_shoulder_lift" type="hinge" axis="1 0 0"
                   range="-1.5 1.5" damping="1.5"/>

            <geom name="right_upper_arm" type="capsule"
                  fromto="0 0 0 0.28 0 -0.06" size="0.035"
                  material="arm_mat" mass="2.5"/>

            <body name="right_elbow" pos="0.28 0 -0.06">
              <joint name="right_elbow" type="hinge" axis="0 1 0"
                     range="-2.5 0" damping="1.0"/>

              <geom name="right_forearm" type="capsule"
                    fromto="0 0 0 0.26 0 0" size="0.03"
                    material="arm_mat" mass="1.5"/>

              <body name="right_hand" pos="0.26 0 0">
                <joint name="right_wrist" type="hinge" axis="0 1 0"
                       range="-1.6 1.6" damping="0.5"/>
                <geom name="right_hand_geom" type="box"
                      size="0.04 0.02 0.09"
                      rgba="0.9 0.7 0.6 1" mass="0.4"/>

                <!-- Club attached to right hand -->
                <body name="club" pos="0 0 -0.09" euler="0 -0.35 0">
                  <joint name="club_wrist" type="hinge" axis="0 1 0"
                         range="-1.2 1.2" damping="0.3"/>

                  <geom name="club_grip" type="capsule"
                        fromto="0 0 0 0 0 -0.28" size="0.015"
                        material="grip_mat" mass="0.1"/>

                  <geom name="club_shaft" type="capsule"
                        fromto="0 0 -0.28 0 0 -1.08" size="0.012"
                        material="club_mat" mass="0.28"/>

                  <body name="clubhead" pos="0 0 -1.08">
                    <geom name="clubhead_geom" type="box"
                          size="0.06 0.04 0.032"
                          rgba="0.15 0.15 0.15 1" mass="0.2"/>
                    <geom name="clubface" type="box"
                          size="0.061 0.041 0.005"
                          pos="0 0.041 0"
                          rgba="0.9 0.2 0.2 0.7"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Golf ball -->
    <body name="ball" pos="0 0.15 0.02135">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.02135"
            rgba="1 1 1 1" mass="0.04593"
            condim="3" friction="0.8 0.005 0.0001"/>
    </body>
  </worldbody>

  <!-- Connect pelvis to legs -->
  <equality>
    <connect body1="pelvis" body2="left_thigh" anchor="0 0 0"/>
    <connect body1="pelvis" body2="right_thigh" anchor="0 0 0"/>
    <weld body1="left_hand" body2="club" relpose="0 0 -0.18 1 0 0 0"/>
  </equality>

  <actuator>
    <!-- Legs -->
    <motor name="left_ankle_motor" joint="left_ankle" gear="30"
           ctrllimited="true" ctrlrange="-40 40"/>
    <motor name="left_knee_motor" joint="left_knee" gear="80"
           ctrllimited="true" ctrlrange="-120 120"/>
    <motor name="right_ankle_motor" joint="right_ankle" gear="30"
           ctrllimited="true" ctrlrange="-40 40"/>
    <motor name="right_knee_motor" joint="right_knee" gear="80"
           ctrllimited="true" ctrlrange="-120 120"/>

    <!-- Torso -->
    <motor name="spine_bend_motor" joint="spine_bend" gear="120"
           ctrllimited="true" ctrlrange="-100 100"/>
    <motor name="spine_rotation_motor" joint="spine_rotation" gear="120"
           ctrllimited="true" ctrlrange="-100 100"/>

    <!-- Left arm -->
    <motor name="left_shoulder_swing_motor" joint="left_shoulder_swing" gear="50"
           ctrllimited="true" ctrlrange="-80 80"/>
    <motor name="left_shoulder_lift_motor" joint="left_shoulder_lift" gear="50"
           ctrllimited="true" ctrlrange="-80 80"/>
    <motor name="left_elbow_motor" joint="left_elbow" gear="40"
           ctrllimited="true" ctrlrange="-60 60"/>
    <motor name="left_wrist_motor" joint="left_wrist" gear="20"
           ctrllimited="true" ctrlrange="-30 30"/>

    <!-- Right arm -->
    <motor name="right_shoulder_swing_motor" joint="right_shoulder_swing" gear="50"
           ctrllimited="true" ctrlrange="-80 80"/>
    <motor name="right_shoulder_lift_motor" joint="right_shoulder_lift" gear="50"
           ctrllimited="true" ctrlrange="-80 80"/>
    <motor name="right_elbow_motor" joint="right_elbow" gear="40"
           ctrllimited="true" ctrlrange="-60 60"/>
    <motor name="right_wrist_motor" joint="right_wrist" gear="20"
           ctrllimited="true" ctrlrange="-30 30"/>

    <!-- Club -->
    <motor name="club_wrist_motor" joint="club_wrist" gear="15"
           ctrllimited="true" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""


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
ADVANCED_BIOMECHANICAL_GOLF_SWING_XML = rf"""
<mujoco model="advanced_biomechanical_golf_swing">
  <option timestep="0.001" gravity="0 0 -{GRAVITY_M_S2}"
          integrator="RK4" solver="Newton" iterations="50"/>

  <compiler angle="radian" coordinate="local"
            inertiafromgeom="false" balanceinertia="true"/>

  <visual>
    <global offwidth="1024" offheight="1024"/>
    <map znear="0.01" zfar="50"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/>
    <quality shadowsize="8192"/>
    <scale framelength="0.2" framewidth="0.005"/>
  </visual>

  <default>
    <geom condim="3" friction="0.9 0.005 0.0001"/>
    <joint damping="0.5" armature="0.01"/>
  </default>

  <asset>
    <material name="torso_mat" rgba="0.75 0.55 0.45 1"
              specular="0.3" shininess="0.2"/>
    <material name="arm_mat" rgba="0.65 0.45 0.35 1"
              specular="0.3" shininess="0.2"/>
    <material name="leg_mat" rgba="0.55 0.55 0.65 1"
              specular="0.3" shininess="0.2"/>
    <material name="scapula_mat" rgba="0.85 0.75 0.65 0.8"
              specular="0.4" shininess="0.3"/>
    <material name="club_grip_mat" rgba="0.1 0.1 0.1 1"
              specular="0.1" shininess="0.05"/>
    <material name="club_shaft_mat" rgba="0.3 0.3 0.35 1"
              specular="0.8" shininess="0.6"/>
    <material name="club_head_mat" rgba="0.15 0.15 0.18 1"
              specular="0.9" shininess="0.8"/>
    <material name="ground_mat" rgba="0.35 0.55 0.25 1"/>
  </asset>

  <worldbody>
    <!-- Ground -->
    <geom name="floor" type="plane" size="15 15 0.1" material="ground_mat"/>
    <light pos="3 -3 4" dir="-0.3 0.3 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <light pos="-3 -3 4" dir="0.3 0.3 -1" directional="true" diffuse="0.4 0.4 0.4"/>

    <!-- Cameras -->
    <camera name="side" pos="-5 -2 1.6" euler="0.15 0 0.35" mode="fixed"/>
    <camera name="front" pos="0 -6 1.6" euler="0.15 0 1.57" mode="fixed"/>
    <camera name="top" pos="0 0 8" euler="0 0 0" mode="fixed"/>
    <camera name="follow" pos="-4 -2 2.5" xyaxes="0 -1 0 0.5 0 1" mode="trackcom"/>
    <camera name="dtl" pos="-2 -2 1.2" xyaxes="0.7 -0.7 0 0.3 0.3 1" mode="fixed"/>

    <!-- Left Foot -->
    <body name="left_foot" pos="-0.12 0 0.04">
      <inertial pos="0.05 0 -0.01" mass="1.2"
                diaginertia="0.0012 0.0035 0.0039"/>
      <geom name="left_foot_geom" type="box" size="0.09 0.14 0.03"
            rgba="0.25 0.25 0.25 1"/>

      <!-- Left Ankle - 2 DOF -->
      <body name="left_talus" pos="0 -0.05 0.04">
        <joint name="left_ankle_plantar" type="hinge" axis="1 0 0"
               range="-0.7 0.9" damping="2.0" armature="0.02"/>
        <joint name="left_ankle_inversion" type="hinge" axis="0 1 0.2"
               range="-0.6 0.6" damping="1.5" armature="0.015"/>

        <!-- Shin -->
        <inertial pos="0 0 0.215" mass="3.8"
                  diaginertia="0.0504 0.0504 0.0048"/>
        <geom name="left_shin_geom" type="capsule" fromto="0 0 0 0 0 0.43"
              size="0.04" material="leg_mat"/>

        <!-- Left Knee - 1 DOF -->
        <body name="left_femur" pos="0 0 0.43">
          <joint name="left_knee" type="hinge" axis="1 0 0"
                 range="-2.3 0.1" damping="3.0" armature="0.03"/>

          <inertial pos="0 0 0.215" mass="9.800"
                    diaginertia="0.1268 0.1268 0.0145"/>
          <geom name="left_thigh_geom" type="capsule" fromto="0 0 0 0 0 0.43"
                size="0.06" material="leg_mat"/>

          <site name="left_hip_site" pos="0 0 0.43"/>
        </body>
      </body>
    </body>

    <!-- Right Foot -->
    <body name="right_foot" pos="0.12 0 0.04">
      <inertial pos="0.05 0 -0.01" mass="1.2"
                diaginertia="0.0012 0.0035 0.0039"/>
      <geom name="right_foot_geom" type="box" size="0.09 0.14 0.03"
            rgba="0.25 0.25 0.25 1"/>

      <!-- Right Ankle - 2 DOF -->
      <body name="right_talus" pos="0 -0.05 0.04">
        <joint name="right_ankle_plantar" type="hinge" axis="1 0 0"
               range="-0.7 0.9" damping="2.0" armature="0.02"/>
        <joint name="right_ankle_inversion" type="hinge" axis="0 1 0.2"
               range="-0.6 0.6" damping="1.5" armature="0.015"/>

        <!-- Shin -->
        <inertial pos="0 0 0.215" mass="3.8"
                  diaginertia="0.0504 0.0504 0.0048"/>
        <geom name="right_shin_geom" type="capsule" fromto="0 0 0 0 0 0.43"
              size="0.04" material="leg_mat"/>

        <!-- Right Knee - 1 DOF -->
        <body name="right_femur" pos="0 0 0.43">
          <joint name="right_knee" type="hinge" axis="1 0 0"
                 range="-2.3 0.1" damping="3.0" armature="0.03"/>

          <inertial pos="0 0 0.215" mass="9.800"
                    diaginertia="0.1268 0.1268 0.0145"/>
          <geom name="right_thigh_geom" type="capsule" fromto="0 0 0 0 0 0.43"
                size="0.06" material="leg_mat"/>

          <site name="right_hip_site" pos="0 0 0.43"/>
        </body>
      </body>
    </body>

    <!-- Pelvis -->
    <body name="pelvis" pos="0 0 0.90">
      <freejoint/>
      <inertial pos="0 0 0" mass="11.7"
                diaginertia="0.0956 0.0929 0.0614"/>
      <geom name="pelvis_geom" type="box" size="0.17 0.10 0.09"
            material="torso_mat"/>

      <!-- Lower Spine -->
      <body name="lumbar" pos="0 0 0.10">
        <!-- Spine - 3 DOF: lateral bend, sagittal bend, axial rotation -->
        <joint name="spine_lateral" type="hinge" axis="0 1 0"
               range="-0.5 0.5" damping="4.0" armature="0.04"/>
        <joint name="spine_sagittal" type="hinge" axis="1 0 0"
               range="-0.7 1.0" damping="4.5" armature="0.045"/>
        <joint name="spine_rotation" type="hinge" axis="0 0 1"
               range="-1.9 1.9" damping="3.5" armature="0.035"/>

        <inertial pos="0 0 0.125" mass="13.2"
                  diaginertia="0.1139 0.1139 0.0221"/>
        <geom name="lumbar_geom" type="capsule" fromto="0 0 0 0 0 0.15"
              size="0.12" material="torso_mat"/>

        <!-- Upper Thorax -->
        <body name="thorax" pos="0 0 0.15">
          <inertial pos="0 0 0.15" mass="17.5"
                    diaginertia="0.1996 0.1996 0.0384"/>
          <geom name="thorax_geom" type="capsule" fromto="0 0 0 0 0 0.30"
                size="0.14" material="torso_mat"/>

          <!-- Head and Neck -->
          <body name="neck" pos="0 0 0.30">
            <inertial pos="0 0 0.12" mass="5.6"
                      diaginertia="0.0268 0.0268 0.0102"/>
            <geom name="neck_geom" type="capsule" fromto="0 0 0 0 0 0.08"
                  size="0.045" material="torso_mat"/>
            <geom name="head_geom" type="sphere" pos="0 0 0.16" size="0.11"
                  rgba="0.9 0.7 0.6 1"/>
          </body>

          <!-- LEFT SCAPULA - 2 DOF (elevation/depression, protraction/retraction) -->
          <body name="left_scapula" pos="-0.08 0 0.25" euler="0 0 -0.2">
            <joint name="left_scapula_elevation" type="hinge" axis="0 1 0"
                   range="-0.5 0.8" damping="1.2" armature="0.012"/>
            <joint name="left_scapula_protraction" type="hinge" axis="0 0 1"
                   range="-0.7 0.7" damping="1.0" armature="0.01"/>

            <inertial pos="-0.06 0 0" mass="0.8"
                      diaginertia="0.0008 0.0012 0.0008"/>
            <geom name="left_scapula_geom" type="box" size="0.06 0.04 0.10"
                  material="scapula_mat"/>

            <!-- LEFT SHOULDER - 3 DOF
                 (flexion/extension, abduction/adduction, rotation) -->
            <body name="left_humerus" pos="-0.08 0 0" euler="0 0 0">
              <joint name="left_shoulder_flexion" type="hinge" axis="0 1 0"
                     range="-1.0 3.0" damping="2.5" armature="0.025"/>
              <joint name="left_shoulder_abduction" type="hinge" axis="1 0 0"
                     range="-0.5 2.8" damping="2.5" armature="0.025"/>
              <joint name="left_shoulder_rotation" type="hinge" axis="0 0 1"
                     range="-1.6 1.6" damping="2.0" armature="0.02"/>

              <inertial pos="0.16 0 0" mass="2.1"
                        diaginertia="0.0116 0.0116 0.0018"/>
              <geom name="left_humerus_geom" type="capsule" fromto="0 0 0 0.32 0 -0.02"
                    size="0.036" material="arm_mat"/>

              <!-- LEFT ELBOW - 1 DOF -->
              <body name="left_radius" pos="0.32 0 -0.02">
                <joint name="left_elbow" type="hinge" axis="0 1 0"
                       range="-2.6 0" damping="1.8" armature="0.018"/>

                <inertial pos="0.14 0 0" mass="1.3"
                          diaginertia="0.0064 0.0064 0.0009"/>
                <geom name="left_forearm_geom" type="capsule" fromto="0 0 0 0.28 0 0"
                      size="0.032" material="arm_mat"/>

                <!-- LEFT WRIST - 2 DOF (flexion/extension, radial/ulnar deviation) -->
                <body name="left_hand" pos="0.28 0 0">
                  <joint name="left_wrist_flexion" type="hinge" axis="0 1 0"
                         range="-1.4 1.4" damping="0.8" armature="0.008"/>
                  <joint name="left_wrist_deviation" type="hinge" axis="1 0 0"
                         range="-0.6 0.5" damping="0.6" armature="0.006"/>

                  <inertial pos="0.045 0 0" mass="0.45"
                            diaginertia="0.0008 0.0008 0.0002"/>
                  <geom name="left_hand_geom" type="box" size="0.045 0.025 0.10"
                        rgba="0.9 0.7 0.6 1"/>
                </body>
              </body>
            </body>
          </body>

          <!-- RIGHT SCAPULA - 2 DOF -->
          <body name="right_scapula" pos="0.08 0 0.25" euler="0 0 0.2">
            <joint name="right_scapula_elevation" type="hinge" axis="0 1 0"
                   range="-0.5 0.8" damping="1.2" armature="0.012"/>
            <joint name="right_scapula_protraction" type="hinge" axis="0 0 1"
                   range="-0.7 0.7" damping="1.0" armature="0.01"/>

            <inertial pos="0.06 0 0" mass="0.8"
                      diaginertia="0.0008 0.0012 0.0008"/>
            <geom name="right_scapula_geom" type="box" size="0.06 0.04 0.10"
                  material="scapula_mat"/>

            <!-- RIGHT SHOULDER - 3 DOF -->
            <body name="right_humerus" pos="0.08 0 0">
              <joint name="right_shoulder_flexion" type="hinge" axis="0 1 0"
                     range="-1.0 3.0" damping="2.5" armature="0.025"/>
              <joint name="right_shoulder_abduction" type="hinge" axis="1 0 0"
                     range="-2.8 0.5" damping="2.5" armature="0.025"/>
              <joint name="right_shoulder_rotation" type="hinge" axis="0 0 1"
                     range="-1.6 1.6" damping="2.0" armature="0.02"/>

              <inertial pos="0.16 0 0" mass="2.1"
                        diaginertia="0.0116 0.0116 0.0018"/>
              <geom name="right_humerus_geom" type="capsule" fromto="0 0 0 0.32 0 -0.02"
                    size="0.036" material="arm_mat"/>

              <!-- RIGHT ELBOW - 1 DOF -->
              <body name="right_radius" pos="0.32 0 -0.02">
                <joint name="right_elbow" type="hinge" axis="0 1 0"
                       range="-2.6 0" damping="1.8" armature="0.018"/>

                <inertial pos="0.14 0 0" mass="1.3"
                          diaginertia="0.0064 0.0064 0.0009"/>
                <geom name="right_forearm_geom" type="capsule" fromto="0 0 0 0.28 0 0"
                      size="0.032" material="arm_mat"/>

                <!-- RIGHT WRIST - 2 DOF -->
                <body name="right_hand" pos="0.28 0 0">
                  <joint name="right_wrist_flexion" type="hinge" axis="0 1 0"
                         range="-1.4 1.4" damping="0.8" armature="0.008"/>
                  <joint name="right_wrist_deviation" type="hinge" axis="1 0 0"
                         range="-0.5 0.6" damping="0.6" armature="0.006"/>

                  <inertial pos="0.045 0 0" mass="0.45"
                            diaginertia="0.0008 0.0008 0.0002"/>
                  <geom name="right_hand_geom" type="box" size="0.045 0.025 0.10"
                        rgba="0.9 0.7 0.6 1"/>

                  <!-- GOLF CLUB - FLEXIBLE SHAFT -->
                  <body name="club_grip" pos="0 0 -0.10" euler="0 -0.4 0">
                    <inertial pos="0 0 -0.14" mass="0.050"
                              diaginertia="0.00008 0.00008 0.000001"/>
                    <geom name="grip_geom" type="capsule"
                          fromto="0 0 0 0 0 -0.28"
                          size="0.0145" material="club_grip_mat"/>

                    <!-- Shaft Segment 1 (upper) -->
                    <body name="shaft_upper" pos="0 0 -0.28">
                      <joint name="shaft_flex_upper" type="hinge" axis="1 0 0"
                             range="-0.15 0.15" damping="0.4"
                             stiffness="150" armature="0.001"/>

                      <inertial pos="0 0 -0.13" mass="0.045"
                                diaginertia="0.00006 0.00006 0.000001"/>
                      <geom name="shaft_upper_geom" type="capsule"
                            fromto="0 0 0 0 0 -0.26"
                            size="0.0062" material="club_shaft_mat"/>

                      <!-- Shaft Segment 2 (middle) -->
                      <body name="shaft_middle" pos="0 0 -0.26">
                        <joint name="shaft_flex_middle" type="hinge" axis="1 0 0"
                               range="-0.2 0.2" damping="0.3"
                               stiffness="120" armature="0.001"/>

                        <inertial pos="0 0 -0.13" mass="0.055"
                                  diaginertia="0.00007 0.00007 0.000001"/>
                        <geom name="shaft_middle_geom" type="capsule"
                              fromto="0 0 0 0 0 -0.26"
                              size="0.0060" material="club_shaft_mat"/>

                        <!-- Shaft Segment 3 (lower - tip) -->
                        <body name="shaft_tip" pos="0 0 -0.26">
                          <joint name="shaft_flex_tip" type="hinge" axis="1 0 0"
                                 range="-0.25 0.25" damping="0.2"
                                 stiffness="100" armature="0.001"/>

                          <inertial pos="0 0 -0.12" mass="0.058"
                                    diaginertia="0.00008 0.00008 0.000001"/>
                          <geom name="shaft_tip_geom" type="capsule"
                                fromto="0 0 0 0 0 -0.24"
                                size="0.0058" material="club_shaft_mat"/>

                          <!-- Hosel and Club Head -->
                          <body name="hosel" pos="0 0 -0.24" euler="0 0.17 0">
                            <inertial pos="0 0.02 -0.01" mass="0.010"
                                      diaginertia="0.000005 0.000005 0.000002"/>
                            <geom name="hosel_geom" type="cylinder"
                                  fromto="0 0 0 0 0.035 -0.005"
                                  size="0.008" material="club_head_mat"/>

                            <!-- Driver Head -->
                            <body name="clubhead" pos="0 0.050 -0.010">
                              <inertial pos="0 0.025 0.002" mass="0.198"
                                        diaginertia="0.00035 0.00025 0.00018"/>

                              <!-- Club head body -->
                              <geom name="head_body" type="box" size="0.062 0.048 0.038"
                                    pos="0 0.025 0" material="club_head_mat"/>

                              <!-- Face -->
                              <geom name="face" type="box" size="0.063 0.003 0.039"
                                    pos="0 0.051 0" rgba="0.85 0.15 0.15 0.9"/>

                              <!-- Crown (top) -->
                              <geom name="crown" type="box" size="0.055 0.042 0.002"
                                    pos="0 0.025 0.040" rgba="0.12 0.12 0.15 1"/>

                              <!-- Sole (bottom) -->
                              <geom name="sole" type="box" size="0.056 0.043 0.003"
                                    pos="0 0.025 -0.041" rgba="0.18 0.18 0.20 1"/>

                              <!-- Alignment aid -->
                              <geom name="alignment" type="box" size="0.002 0.035 0.001"
                                    pos="0 0.025 0.043" rgba="1 1 1 0.95"/>
                            </body>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Golf Ball -->
    <body name="ball" pos="0 0.20 0.02135">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.04593"
                diaginertia="0.000017 0.000017 0.000017"/>
      <geom name="ball_geom" type="sphere" size="0.02135"
            rgba="1 1 1 1" condim="3" friction="0.8 0.005 0.0001"/>
      <!-- Dimple visualization (cosmetic) -->
      <geom name="ball_detail" type="sphere" size="0.0214"
            rgba="0.95 0.95 0.95 0.3"/>
    </body>
  </worldbody>

  <!-- Connect pelvis to legs via equality constraints -->
  <equality>
    <connect body1="pelvis" body2="left_femur" anchor="-0.12 0 0"/>
    <connect body1="pelvis" body2="right_femur" anchor="0.12 0 0"/>
    <!-- Left hand grips club -->
    <weld body1="left_hand" body2="club_grip"
          relpose="0 0 -0.18 1 0 0 0" active="true"/>
  </equality>

  <actuator>
    <!-- Left Leg (3 DOF) -->
    <motor name="l_ankle_plantar" joint="left_ankle_plantar"
           gear="40" ctrllimited="true" ctrlrange="-50 50"/>
    <motor name="l_ankle_invert" joint="left_ankle_inversion"
           gear="30" ctrllimited="true" ctrlrange="-35 35"/>
    <motor name="l_knee" joint="left_knee"
           gear="120" ctrllimited="true" ctrlrange="-150 150"/>

    <!-- Right Leg (3 DOF) -->
    <motor name="r_ankle_plantar" joint="right_ankle_plantar"
           gear="40" ctrllimited="true" ctrlrange="-50 50"/>
    <motor name="r_ankle_invert" joint="right_ankle_inversion"
           gear="30" ctrllimited="true" ctrlrange="-35 35"/>
    <motor name="r_knee" joint="right_knee"
           gear="120" ctrllimited="true" ctrlrange="-150 150"/>

    <!-- Spine (3 DOF) -->
    <motor name="spine_lateral" joint="spine_lateral"
           gear="140" ctrllimited="true" ctrlrange="-120 120"/>
    <motor name="spine_sagittal" joint="spine_sagittal"
           gear="150" ctrllimited="true" ctrlrange="-130 130"/>
    <motor name="spine_rotation" joint="spine_rotation"
           gear="130" ctrllimited="true" ctrlrange="-110 110"/>

    <!-- Left Scapula (2 DOF) -->
    <motor name="l_scap_elev" joint="left_scapula_elevation"
           gear="35" ctrllimited="true" ctrlrange="-40 40"/>
    <motor name="l_scap_prot" joint="left_scapula_protraction"
           gear="30" ctrllimited="true" ctrlrange="-35 35"/>

    <!-- Left Shoulder (3 DOF) -->
    <motor name="l_shldr_flex" joint="left_shoulder_flexion"
           gear="70" ctrllimited="true" ctrlrange="-90 90"/>
    <motor name="l_shldr_abd" joint="left_shoulder_abduction"
           gear="70" ctrllimited="true" ctrlrange="-90 90"/>
    <motor name="l_shldr_rot" joint="left_shoulder_rotation"
           gear="50" ctrllimited="true" ctrlrange="-65 65"/>

    <!-- Left Elbow (1 DOF) -->
    <motor name="l_elbow" joint="left_elbow"
           gear="60" ctrllimited="true" ctrlrange="-75 75"/>

    <!-- Left Wrist (2 DOF) -->
    <motor name="l_wrist_flex" joint="left_wrist_flexion"
           gear="25" ctrllimited="true" ctrlrange="-35 35"/>
    <motor name="l_wrist_dev" joint="left_wrist_deviation"
           gear="20" ctrllimited="true" ctrlrange="-28 28"/>

    <!-- Right Scapula (2 DOF) -->
    <motor name="r_scap_elev" joint="right_scapula_elevation"
           gear="35" ctrllimited="true" ctrlrange="-40 40"/>
    <motor name="r_scap_prot" joint="right_scapula_protraction"
           gear="30" ctrllimited="true" ctrlrange="-35 35"/>

    <!-- Right Shoulder (3 DOF) -->
    <motor name="r_shldr_flex" joint="right_shoulder_flexion"
           gear="70" ctrllimited="true" ctrlrange="-90 90"/>
    <motor name="r_shldr_abd" joint="right_shoulder_abduction"
           gear="70" ctrllimited="true" ctrlrange="-90 90"/>
    <motor name="r_shldr_rot" joint="right_shoulder_rotation"
           gear="50" ctrllimited="true" ctrlrange="-65 65"/>

    <!-- Right Elbow (1 DOF) -->
    <motor name="r_elbow" joint="right_elbow"
           gear="60" ctrllimited="true" ctrlrange="-75 75"/>

    <!-- Right Wrist (2 DOF) -->
    <motor name="r_wrist_flex" joint="right_wrist_flexion"
           gear="25" ctrllimited="true" ctrlrange="-35 35"/>
    <motor name="r_wrist_dev" joint="right_wrist_deviation"
           gear="20" ctrllimited="true" ctrlrange="-28 28"/>

    <!-- Flexible Shaft (3 DOF) - typically controlled passively,
         but can be actuated for study -->
    <motor name="shaft_upper" joint="shaft_flex_upper"
           gear="2" ctrllimited="true" ctrlrange="-5 5"/>
    <motor name="shaft_middle" joint="shaft_flex_middle"
           gear="2" ctrllimited="true" ctrlrange="-5 5"/>
    <motor name="shaft_tip" joint="shaft_flex_tip"
           gear="2" ctrllimited="true" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""


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


# GOLF CLUB CONFIGURATIONS
# ==============================================================================

# Golf club parameters (realistic values)
CLUB_CONFIGS: dict[str, dict[str, float | list[float]]] = {
    "driver": {
        "grip_length": 0.28,
        "grip_radius": 0.0145,
        "grip_mass": 0.050,
        "shaft_length": 1.10,  # Total shaft length
        "shaft_radius": 0.0062,
        "shaft_mass": 0.065,
        "head_mass": 0.198,
        "head_size": [0.062, 0.048, 0.038],
        "total_length": 1.16,
        "club_loft": 0.17,  # 10 degrees
        "flex_stiffness": [180, 150, 120],  # Upper, middle, lower
    },
    "iron_7": {
        "grip_length": 0.26,
        "grip_radius": 0.0140,
        "grip_mass": 0.048,
        "shaft_length": 0.94,
        "shaft_radius": 0.0058,
        "shaft_mass": 0.072,
        "head_mass": 0.253,
        "head_size": [0.038, 0.025, 0.045],
        "total_length": 0.95,
        "club_loft": 0.56,  # 32 degrees
        "flex_stiffness": [220, 200, 180],
    },
    "wedge": {
        "grip_length": 0.25,
        "grip_radius": 0.0138,
        "grip_mass": 0.045,
        "shaft_length": 0.89,
        "shaft_radius": 0.0056,
        "shaft_mass": 0.078,
        "head_mass": 0.288,
        "head_size": [0.032, 0.022, 0.048],
        "total_length": 0.90,
        "club_loft": 0.96,  # 55 degrees
        "flex_stiffness": [240, 220, 200],
    },
}


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
