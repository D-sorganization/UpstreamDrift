"""Ground-truth regression tests for MuJoCo dynamics defects #8021, #8008, #8015.

Each test compares the production code against an independently derived truth
value (``mj_inverse`` / finite differences), not against a previously recorded
output of the same code path.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_mujoco_available = importlib.util.find_spec("mujoco") is not None

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _mujoco_available, reason="mujoco not installed"),
]

TWO_LINK_XML = """
<mujoco>
  <option gravity="0 0 -9.80665"/>
  <worldbody>
    <body name="l1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="1 0 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.05"/>
      <body name="l2" pos="0 0 -0.5">
        <joint name="j2" type="hinge" axis="1 0 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.05"/>
        <body name="club_head" pos="0 0 -0.5" euler="0 30 45">
          <geom type="sphere" size="0.06"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1"/>
    <motor joint="j2"/>
  </actuator>
</mujoco>
"""

FLOATING_BASE_XML = """
<mujoco>
  <option gravity="0 0 -9.80665"/>
  <worldbody>
    <body name="root" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="link" pos="0.1 0 0">
        <joint name="h1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
        <body name="tip" pos="0.3 0 0">
          <joint name="b1" type="ball"/>
          <geom type="sphere" size="0.04"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def two_link():
    import mujoco

    model = mujoco.MjModel.from_xml_string(TWO_LINK_XML)
    return model, mujoco.MjData(model)


class TestRecursiveNewtonEulerGroundTruth:
    """#8021 — ``compute`` returned all zeros for every input."""

    def test_matches_mj_inverse(self, two_link) -> None:
        import mujoco

        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._id_core import (
            RecursiveNewtonEuler,
        )

        model, data = two_link
        qpos = np.array([0.3, -0.5])
        qvel = np.array([1.2, -0.7])
        qacc = np.array([2.0, 1.0])

        truth_data = mujoco.MjData(model)
        truth_data.qpos[:] = qpos
        truth_data.qvel[:] = qvel
        mujoco.mj_forward(model, truth_data)
        truth_data.qacc[:] = qacc
        mujoco.mj_inverse(model, truth_data)
        truth = truth_data.qfrc_inverse.copy()

        got = RecursiveNewtonEuler(model, data).compute(qpos, qvel, qacc)

        assert np.any(np.abs(truth) > 1e-6), "degenerate fixture: truth is zero"
        np.testing.assert_allclose(got, truth, atol=1e-8)

    def test_depends_on_qacc(self, two_link) -> None:
        """flg_acc=0 would make the result independent of qacc."""
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._id_core import (
            RecursiveNewtonEuler,
        )

        model, data = two_link
        rne = RecursiveNewtonEuler(model, data)
        qpos = np.array([0.3, -0.5])
        qvel = np.array([1.2, -0.7])

        a = rne.compute(qpos, qvel, np.zeros(2))
        b = rne.compute(qpos, qvel, np.array([2.0, 1.0]))
        assert not np.allclose(a, b)

    def test_rejects_wrong_sized_input(self, two_link) -> None:
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._id_core import (
            RecursiveNewtonEuler,
        )

        model, data = two_link
        rne = RecursiveNewtonEuler(model, data)
        with pytest.raises(ValueError, match="nv"):
            rne.compute(np.zeros(model.nq), np.zeros(model.nv + 1), np.zeros(model.nv))


class TestTaskSpaceInducedAcceleration:
    """#8008 — club acceleration was always zero (``data.cacc`` never populated)."""

    @staticmethod
    def _finite_difference_accel(model, data, body_id: int) -> np.ndarray:
        import mujoco

        dt = 1e-7

        def linear_velocity(d):
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, d, jacp, None, body_id)
            return jacp @ d.qvel

        d0 = mujoco.MjData(model)
        d0.qpos[:] = data.qpos
        d0.qvel[:] = data.qvel
        d0.ctrl[:] = data.ctrl
        mujoco.mj_forward(model, d0)
        v0 = linear_velocity(d0)

        d1 = mujoco.MjData(model)
        d1.qpos[:] = d0.qpos
        d1.qvel[:] = d0.qvel + dt * d0.qacc
        mujoco.mj_integratePos(model, d1.qpos, d1.qvel, dt)
        d1.ctrl[:] = d0.ctrl
        mujoco.mj_forward(model, d1)
        return (linear_velocity(d1) - v0) / dt

    def test_total_matches_finite_difference(self, two_link) -> None:
        import mujoco

        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.rigid_body_dynamics.induced_acceleration import (
            MuJoCoInducedAccelerationAnalyzer,
        )

        model, data = two_link
        data.qpos[:] = [0.4, -0.3]
        data.qvel[:] = [1.5, -2.0]
        data.ctrl[:] = [0.5, -0.2]
        mujoco.mj_forward(model, data)

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "club_head")
        truth = self._finite_difference_accel(model, data, body_id)
        assert np.linalg.norm(truth) > 1e-3, "degenerate fixture: truth is zero"

        mujoco.mj_forward(model, data)
        result = MuJoCoInducedAccelerationAnalyzer(
            model, data
        ).compute_task_space_components("club_head")
        assert result is not None
        np.testing.assert_allclose(result["total"], truth, atol=1e-4)

    def test_components_sum_to_total(self, two_link) -> None:
        import mujoco

        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.rigid_body_dynamics.induced_acceleration import (
            MuJoCoInducedAccelerationAnalyzer,
        )

        model, data = two_link
        data.qpos[:] = [0.4, -0.3]
        data.qvel[:] = [1.5, -2.0]
        data.ctrl[:] = [0.5, -0.2]
        mujoco.mj_forward(model, data)

        result = MuJoCoInducedAccelerationAnalyzer(
            model, data
        ).compute_task_space_components("club_head")
        assert result is not None
        parts = (
            result["gravity"]
            + result["velocity"]
            + result["control"]
            + result["constraint"]
        )
        np.testing.assert_allclose(parts, result["total"], atol=1e-9)

    def test_planar_mechanism_has_no_out_of_plane_acceleration(self, two_link) -> None:
        """The old ``xmat @ cacc`` rotation injected a spurious X component."""
        import mujoco

        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.rigid_body_dynamics.induced_acceleration import (
            MuJoCoInducedAccelerationAnalyzer,
        )

        model, data = (
            two_link  # both hinges about X -> motion confined to the y-z plane
        )
        data.qpos[:] = [0.4, -0.3]
        data.qvel[:] = [1.5, -2.0]
        mujoco.mj_forward(model, data)

        result = MuJoCoInducedAccelerationAnalyzer(
            model, data
        ).compute_task_space_components("club_head")
        assert result is not None
        assert abs(result["total"][0]) < 1e-9


class TestInverseKinematicsFloatingBase:
    """#8015 — IK added an nv tangent step to an nq configuration."""

    def test_ik_runs_on_floating_base_model(self) -> None:
        import mujoco

        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.advanced_kinematics import (
            AdvancedKinematicsAnalyzer,
        )

        model = mujoco.MjModel.from_xml_string(FLOATING_BASE_XML)
        data = mujoco.MjData(model)
        assert model.nq != model.nv, "fixture must exercise the nq != nv path"
        mujoco.mj_forward(model, data)

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tip")
        target = data.xpos[body_id].copy() + np.array([0.02, 0.01, -0.02])

        analyzer = AdvancedKinematicsAnalyzer(model, data)
        q, success, _ = analyzer.solve_inverse_kinematics(body_id, target)

        assert len(q) == model.nq
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        residual = float(np.linalg.norm(data.xpos[body_id] - target))
        assert success
        assert residual < 1e-3

    def test_quaternion_stays_normalized(self) -> None:
        """Element-wise addition to a quaternion would break unit norm."""
        import mujoco

        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.advanced_kinematics import (
            AdvancedKinematicsAnalyzer,
        )

        model = mujoco.MjModel.from_xml_string(FLOATING_BASE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tip")
        target = data.xpos[body_id].copy() + np.array([0.3, 0.2, -0.15])

        analyzer = AdvancedKinematicsAnalyzer(model, data)
        q, _, _ = analyzer.solve_inverse_kinematics(body_id, target)

        # free joint quaternion occupies qpos[3:7]
        assert abs(float(np.linalg.norm(q[3:7])) - 1.0) < 1e-6

    def test_mocap_retargeting_ik_runs_on_floating_base(self) -> None:
        import mujoco

        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._mocap_data import (
            MarkerSet,
            MotionCaptureFrame,
        )
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._mocap_retargeting import (
            MotionRetargeting,
        )

        model = mujoco.MjModel.from_xml_string(FLOATING_BASE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tip")
        target = data.xpos[body_id].copy() + np.array([0.01, 0.0, -0.01])

        marker_set = MarkerSet(
            markers={"TIP": "tip"}, marker_offsets={"TIP": np.zeros(3)}
        )
        retargeter = MotionRetargeting(model, data, marker_set)
        assert retargeter.marker_to_body_id == {"TIP": body_id}
        frame = MotionCaptureFrame(time=0.0, marker_positions={"TIP": target})

        q, _ = retargeter._solve_frame_ik(frame, ["TIP"], data.qpos.copy(), 50)
        assert len(q) == model.nq
        assert abs(float(np.linalg.norm(q[3:7])) - 1.0) < 1e-6
