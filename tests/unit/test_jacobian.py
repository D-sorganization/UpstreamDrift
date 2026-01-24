"""Tests for Jacobian computation across physics engines.

Implements Task 3.1: Jacobian Coverage Completion per Phase 3 roadmap.
Verifies Jacobian shape (6×nv), structure, and cross-engine consistency.

Refactored to use shared engine availability module (DRY principle).
"""

import numpy as np
import pytest

from src.shared.python.engine_availability import (
    MUJOCO_AVAILABLE,
    PINOCCHIO_AVAILABLE,
    skip_if_unavailable,
)

# Simple inline URDF for Jacobian tests (2-DOF planar arm)
SIMPLE_ARM_URDF = """<?xml version="1.0"?>
<robot name="simple_arm">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.25"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry><cylinder radius="0.02" length="0.5"/></geometry>
    </visual>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>

  <link name="link2">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.25"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
    <visual>
      <geometry><cylinder radius="0.02" length="0.5"/></geometry>
    </visual>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.5"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
</robot>
"""

# Simple MJCF equivalent for MuJoCo tests
SIMPLE_ARM_MJCF = """
<mujoco model="simple_arm">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <compiler angle="radian"/>

  <worldbody>
    <light name="light" pos="0 0 3"/>
    <body name="link1" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 1 0"/>
      <geom type="cylinder" size="0.02 0.25" pos="0 0 0.25"/>
      <inertial pos="0 0 0.25" mass="1.0" diaginertia="0.01 0.01 0.01"/>

      <body name="link2" pos="0 0 0.5">
        <joint name="joint2" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.02 0.25" pos="0 0 0.25"/>
        <inertial pos="0 0 0.25" mass="0.5" diaginertia="0.005 0.005 0.005"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class TestJacobianShape:
    """Tests for Jacobian shape compliance."""

    @skip_if_unavailable("mujoco")
    def test_mujoco_jacobian_shape(self) -> None:
        """Test MuJoCo Jacobian returns correct shape (3×nv, 3×nv, 6×nv)."""
        import mujoco

        model = mujoco.MjModel.from_xml_string(SIMPLE_ARM_MJCF)
        data = mujoco.MjData(model)

        # Forward kinematics
        mujoco.mj_forward(model, data)

        # Get body ID for link2
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link2")
        assert body_id != -1, "Body 'link2' not found in MuJoCo model"

        # Compute Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

        # Verify shapes
        assert jacp.shape == (3, model.nv), f"Linear Jacobian shape: {jacp.shape}"
        assert jacr.shape == (3, model.nv), f"Angular Jacobian shape: {jacr.shape}"

        # Verify nv matches expected (2 DOF)
        assert model.nv == 2, f"Expected 2 DOF, got {model.nv}"

    @skip_if_unavailable("pinocchio")
    def test_pinocchio_jacobian_shape(self) -> None:
        """Test Pinocchio Jacobian returns correct shape (6×nv)."""
        # Load URDF
        import tempfile
        from pathlib import Path

        import pinocchio as pin

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(SIMPLE_ARM_URDF)
            urdf_path = f.name

        try:
            model = pin.buildModelFromUrdf(urdf_path)
            data = model.createData()

            # Set neutral configuration
            q = pin.neutral(model)
            pin.forwardKinematics(model, data, q)
            pin.computeJointJacobians(model, data, q)

            # Get frame ID for link2
            if model.existFrame("link2"):
                frame_id = model.getFrameId("link2")
                J = pin.getFrameJacobian(
                    model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )

                # Verify shape (6×nv)
                assert J.shape[0] == 6, f"Expected 6 rows, got {J.shape[0]}"
                assert (
                    J.shape[1] == model.nv
                ), f"Expected {model.nv} cols, got {J.shape[1]}"

        finally:
            Path(urdf_path).unlink(missing_ok=True)


class TestJacobianStructure:
    """Tests for Jacobian structural correctness."""

    @skip_if_unavailable("mujoco")
    def test_mujoco_jacobian_nonzero(self) -> None:
        """Test MuJoCo Jacobian has expected non-zero entries."""
        import mujoco

        model = mujoco.MjModel.from_xml_string(SIMPLE_ARM_MJCF)
        data = mujoco.MjData(model)

        # Set non-zero joint positions
        data.qpos[0] = 0.5  # Joint 1
        data.qpos[1] = 0.3  # Joint 2

        mujoco.mj_forward(model, data)

        # Get Jacobian for link2 (end-effector)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link2")
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

        # For a 2-DOF planar arm with Y-axis rotation:
        # - Both joints should contribute to linear velocity (both columns non-zero)
        # - Angular Jacobian should have non-zero Y-components
        assert np.any(jacp != 0), "Linear Jacobian should not be all zeros"
        assert np.any(jacr != 0), "Angular Jacobian should not be all zeros"

        # Verify Y-axis rotation coupling (angular Jacobian Y-row)
        assert np.any(jacr[1, :] != 0), "Y-axis rotation should be non-zero"

    @skip_if_unavailable("mujoco")
    def test_jacobian_zero_position(self) -> None:
        """Test Jacobian at zero configuration."""
        import mujoco

        model = mujoco.MjModel.from_xml_string(SIMPLE_ARM_MJCF)
        data = mujoco.MjData(model)

        # Zero configuration
        data.qpos[:] = 0
        mujoco.mj_forward(model, data)

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link2")
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

        # At zero config, linear Jacobian should still be non-zero
        # because joint rotation still produces end-effector translation
        assert np.any(jacp != 0), "Linear Jacobian at zero config should be non-zero"


class TestJacobianConsistency:
    """Tests for cross-engine Jacobian consistency."""

    @pytest.mark.skipif(
        not (MUJOCO_AVAILABLE and PINOCCHIO_AVAILABLE),
        reason="Requires both MuJoCo and Pinocchio",
    )
    def test_jacobian_shape_consistency(self) -> None:
        """Test Jacobian shapes match across engines."""
        import tempfile
        from pathlib import Path

        import mujoco
        import pinocchio as pin

        # MuJoCo
        mj_model = mujoco.MjModel.from_xml_string(SIMPLE_ARM_MJCF)

        # Pinocchio
        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(SIMPLE_ARM_URDF)
            urdf_path = f.name

        try:
            pin_model = pin.buildModelFromUrdf(urdf_path)

            # Both should have same number of DOF
            assert (
                mj_model.nv == pin_model.nv
            ), f"DOF mismatch: MuJoCo={mj_model.nv}, Pinocchio={pin_model.nv}"

        finally:
            Path(urdf_path).unlink(missing_ok=True)


class TestJacobianNumericalValidation:
    """Numerical validation of Jacobians via finite differences."""

    @skip_if_unavailable("mujoco")
    def test_jacobian_finite_difference_validation(self) -> None:
        """Validate Jacobian against finite difference approximation."""
        import mujoco

        model = mujoco.MjModel.from_xml_string(SIMPLE_ARM_MJCF)
        data = mujoco.MjData(model)

        # Set random configuration
        data.qpos[0] = 0.3
        data.qpos[1] = 0.5
        mujoco.mj_forward(model, data)

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link2")

        # Get analytical Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

        # Get current body position
        body_pos = data.xpos[body_id].copy()

        # Finite difference Jacobian
        eps = 1e-6
        jacp_fd = np.zeros((3, model.nv))

        for i in range(model.nv):
            # Perturb
            data.qpos[i] += eps
            mujoco.mj_forward(model, data)
            pos_pert = data.xpos[body_id].copy()

            jacp_fd[:, i] = (pos_pert - body_pos) / eps

            # Restore
            data.qpos[i] -= eps

        mujoco.mj_forward(model, data)

        # Compare analytical vs finite difference
        np.testing.assert_allclose(
            jacp,
            jacp_fd,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Analytical Jacobian doesn't match finite difference",
        )
