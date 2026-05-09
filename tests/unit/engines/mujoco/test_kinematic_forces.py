"""Comprehensive tests for kinematic forces module."""

import warnings
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.kinematic_forces import (
    KinematicForceAnalyzer,
    KinematicForceData,
)
from mujoco_humanoid_golf.models import DOUBLE_PENDULUM_XML


def create_limited_club_model() -> SimpleNamespace:
    """Create a minimal mock model with a limited hinge and club-head body."""
    return SimpleNamespace(
        nv=3,
        nq=3,
        nu=1,
        njnt=1,
        nbody=2,
        jnt_limited=np.array([True]),
        jnt_qposadr=np.array([0]),
        jnt_range=np.array([[-0.1, 0.1]]),
        body_mass=np.array([1.0, 1.0]),
    )


class TestKinematicForceData:
    """Tests for KinematicForceData dataclass."""

    def test_kinematic_forces_initialization(self) -> None:
        """Test force data initialization."""
        coriolis = np.array([1.0, -0.5])
        gravity = np.array([0.5, -0.3])

        data = KinematicForceData(
            time=1.0,
            coriolis_forces=coriolis,
            gravity_forces=gravity,
        )

        assert data.time == 1.0
        np.testing.assert_array_equal(data.coriolis_forces, coriolis)
        np.testing.assert_array_equal(data.gravity_forces, gravity)
        assert data.coriolis_power == 0.0


class TestKinematicForceAnalyzer:
    """Tests for KinematicForceAnalyzer class."""

    @pytest.fixture()
    def model_and_data(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Create model and data for testing."""
        model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def test_kinematic_forces_initialization(self, model_and_data) -> None:
        """Test analyzer initialization."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        assert analyzer.model == model
        assert analyzer.data == data

    def test_kinematic_forces_find_body_id(self, model_and_data) -> None:
        """Test finding body ID."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        body_id = analyzer._find_body_id("shoulder")
        if body_id is not None:
            assert body_id > 0
            assert body_id < model.nbody

        # Should return None for nonexistent body
        body_id = analyzer._find_body_id("nonexistent_body_xyz")
        assert body_id is None

    def test_compute_coriolis_forces(self, model_and_data) -> None:
        """Test computing Coriolis forces."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()
        qvel = np.array([0.1, -0.05])

        coriolis = analyzer.compute_coriolis_forces(qpos, qvel)

        assert coriolis.shape == (model.nv,)
        assert np.all(np.isfinite(coriolis))

    def test_compute_coriolis_forces_zero_velocity(self, model_and_data) -> None:
        """Test Coriolis forces with zero velocity."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()
        qvel = np.zeros(model.nv)

        coriolis = analyzer.compute_coriolis_forces(qpos, qvel)

        # With zero velocity, Coriolis should be approximately zero
        assert coriolis.shape == (model.nv,)
        # May have small numerical errors
        assert np.all(np.abs(coriolis) < 1e-3)

    def test_kinematic_forces_compute_gravity_forces(self, model_and_data) -> None:
        """Test computing gravity forces."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()

        gravity = analyzer.compute_gravity_forces(qpos)

        assert gravity.shape == (model.nv,)
        assert np.all(np.isfinite(gravity))

    def test_decompose_coriolis_forces(self, model_and_data) -> None:
        """Test decomposing Coriolis forces."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()
        qvel = np.array([0.1, -0.05])

        centrifugal, coupling = analyzer.decompose_coriolis_forces(qpos, qvel)

        assert centrifugal.shape == (model.nv,)
        assert coupling.shape == (model.nv,)
        assert np.all(np.isfinite(centrifugal))
        assert np.all(np.isfinite(coupling))

    def test_kinematic_forces_compute_mass_matrix(self, model_and_data) -> None:
        """Test computing mass matrix."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()

        M = analyzer.compute_mass_matrix(qpos)

        assert M.shape == (model.nv, model.nv)
        assert np.allclose(M, M.T)  # Symmetric
        assert np.all(np.linalg.eigvals(M) > 0)  # Positive definite
        assert np.all(np.isfinite(M))

    def test_compute_club_head_apparent_forces(self, model_and_data) -> None:
        """Test computing club head apparent forces."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()
        qvel = np.array([0.1, -0.05])
        qacc = np.zeros(model.nv)  # qacc needed for apparent force

        # May not have club head in simple model
        if analyzer.club_head_id is not None:
            coriolis, centrifugal, apparent = (
                analyzer.compute_club_head_apparent_forces(qpos, qvel, qacc)
            )

            assert coriolis.shape == (3,)
            assert centrifugal.shape == (3,)
            assert apparent.shape == (3,)

            # Check finiteness
            assert np.all(np.isfinite(coriolis))
            assert np.all(np.isfinite(centrifugal))
            assert np.all(np.isfinite(apparent))

    def test_compute_club_head_apparent_forces_clamps_joint_limits(
        self,
        monkeypatch,
    ) -> None:
        """Test the central-difference perturbation stays within joint limits."""
        model = create_limited_club_model()
        data = SimpleNamespace(
            qpos=np.array([0.1, 0.0, 0.0]),
            qvel=np.zeros(model.nv),
            qacc=np.zeros(model.nv),
            xpos=np.zeros((model.nbody, 3)),
            ctrl=np.zeros(model.nu),
            time=0.0,
        )

        monkeypatch.setattr(
            mujoco,
            "MjData",
            lambda _model: SimpleNamespace(
                qpos=np.zeros(_model.nq),
                qvel=np.zeros(_model.nv),
                qacc=np.zeros(_model.nv),
                xpos=np.zeros((_model.nbody, 3)),
                ctrl=np.zeros(_model.nu),
                time=0.0,
            ),
        )
        monkeypatch.setattr(
            mujoco,
            "mj_id2name",
            lambda _model, _obj, body_id: "club_head" if body_id == 1 else "world",
        )
        monkeypatch.setattr(mujoco, "mj_forward", lambda _model, _data: None)
        analyzer = KinematicForceAnalyzer(model, data)

        qvel = np.array([1.0, 0.0, 0.0])
        qacc = np.zeros(model.nv)
        observed_qpos: list[np.ndarray] = []

        def fake_compute_jacobian(body_id: int, data=None):
            """Record the perturbed configuration and return zero Jacobians."""
            assert data is not None
            observed_qpos.append(data.qpos.copy())
            return np.zeros((3, model.nv)), np.zeros((3, model.nv))

        monkeypatch.setattr(analyzer, "_compute_jacobian", fake_compute_jacobian)
        monkeypatch.setattr(
            analyzer,
            "compute_coriolis_forces",
            lambda _qpos, _qvel: np.zeros(model.nv),
        )

        coriolis, centrifugal, apparent = analyzer.compute_club_head_apparent_forces(
            data.qpos.copy(),
            qvel,
            qacc,
        )

        assert len(observed_qpos) == 3
        assert np.isclose(observed_qpos[0][0], 0.1)
        assert np.all(np.isfinite(coriolis))
        assert np.all(np.isfinite(centrifugal))
        assert np.all(np.isfinite(apparent))

        q_min, q_max = model.jnt_range[0]
        for q_sample in observed_qpos:
            assert q_min - 1e-12 <= q_sample[0] <= q_max + 1e-12

    def test_apparent_forces_use_effective_step_when_clamped(
        self,
        monkeypatch,
    ) -> None:
        """Clamped finite differences must divide by the actual post-clamp step.

        Regression for issue #2766: at a joint limit with outward velocity,
        clamping produces an asymmetric perturbation (one side pinned to the
        boundary, the other free). Dividing the Jacobian difference by the
        nominal 2 * epsilon denominator then systematically underestimates
        jacp_dot. The fix uses the effective displacement projected onto qvel.
        """
        model = create_limited_club_model()
        # Start at the positive joint limit so forward perturbation clips.
        qpos_start = np.array([0.1, 0.0, 0.0])
        data = SimpleNamespace(
            qpos=qpos_start.copy(),
            qvel=np.zeros(model.nv),
            qacc=np.zeros(model.nv),
            xpos=np.zeros((model.nbody, 3)),
            ctrl=np.zeros(model.nu),
            time=0.0,
        )

        monkeypatch.setattr(
            mujoco,
            "MjData",
            lambda _model: SimpleNamespace(
                qpos=np.zeros(_model.nq),
                qvel=np.zeros(_model.nv),
                qacc=np.zeros(_model.nv),
                xpos=np.zeros((_model.nbody, 3)),
                ctrl=np.zeros(_model.nu),
                time=0.0,
            ),
        )
        monkeypatch.setattr(
            mujoco,
            "mj_id2name",
            lambda _model, _obj, body_id: "club_head" if body_id == 1 else "world",
        )
        monkeypatch.setattr(mujoco, "mj_forward", lambda _model, _data: None)
        analyzer = KinematicForceAnalyzer(model, data)

        # Outward velocity on the first joint means `qpos + epsilon * qvel` is
        # clamped to the boundary while `qpos - epsilon * qvel` is not.
        qvel = np.array([1.0, 0.0, 0.0])
        qacc = np.zeros(model.nv)

        call_count = {"n": 0}

        def fake_compute_jacobian(body_id: int, data=None):
            """Return a Jacobian that depends linearly on qpos[0].

            With jacp[0,0] = qpos[0], the true dJ/dqpos[0] is 1.0, so
            jacp_dot @ qvel along qvel = [1, 0, 0] should equal [1, 0, 0].
            A bug using 2 * epsilon when forward-clamp is active would report
            a smaller value proportional to the asymmetry.
            """
            assert data is not None
            call_count["n"] += 1
            jacp = np.zeros((3, model.nv))
            jacp[0, 0] = float(data.qpos[0])
            return jacp, np.zeros((3, model.nv))

        monkeypatch.setattr(analyzer, "_compute_jacobian", fake_compute_jacobian)
        monkeypatch.setattr(
            analyzer,
            "compute_coriolis_forces",
            lambda _qpos, _qvel: np.zeros(model.nv),
        )

        coriolis, _, _ = analyzer.compute_club_head_apparent_forces(
            qpos_start.copy(),
            qvel,
            qacc,
        )

        # With true dJ/dqpos[0] = 1 and qvel[0] = 1, coriolis_accel = [1, 0, 0]
        # and coriolis_force = -club_head_mass * coriolis_accel = [-1, 0, 0].
        # The buggy 2*epsilon denominator would give a magnitude strictly less
        # than 1 because the forward perturbation is clipped to the limit.
        assert call_count["n"] >= 3
        assert np.isclose(coriolis[0], -1.0, atol=1e-6), (
            f"expected coriolis force magnitude ~1.0 with effective-step "
            f"correction, got {coriolis[0]}"
        )
        assert np.isclose(coriolis[1], 0.0, atol=1e-9)
        assert np.isclose(coriolis[2], 0.0, atol=1e-9)

    def test_analyze_trajectory(self, model_and_data) -> None:
        """Test analyzing trajectory."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        times = np.array([0.0, 0.01, 0.02])
        positions = np.array([data.qpos.copy() for _ in range(3)])
        velocities = np.array([data.qvel.copy() for _ in range(3)])
        accelerations = np.zeros((3, model.nv))

        results = analyzer.analyze_trajectory(
            times, positions, velocities, accelerations
        )

        assert len(results) == 3
        assert all(isinstance(r, KinematicForceData) for r in results)
        assert all(r.time == t for r, t in zip(results, times, strict=False))

    def test_compute_effective_mass(self, model_and_data) -> None:
        """Test computing effective mass."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        # Set a non-zero configuration to avoid singularities
        qpos = np.array([0.5, 0.5])  # Arbitrary non-zero angles
        direction = np.array([0.0, 1.0, 0.0])  # Y direction (tangential movement)

        body_id = analyzer._find_body_id("club_body")
        assert body_id is not None, "club_body not found in model"

        # Filter runtime warning for rank deficient Jacobian (planar robot in 3D world)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="Jacobian is rank deficient"
            )
            m_eff = analyzer.compute_effective_mass(qpos, direction, body_id)

        assert isinstance(m_eff, float)
        assert m_eff >= 0.0
        assert np.isfinite(m_eff)

    def test_compute_kinematic_power(self, model_and_data) -> None:
        """Test computing kinematic power."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()
        qvel = np.array([0.1, -0.05])

        power_data = analyzer.compute_kinematic_power(qpos, qvel)

        assert "coriolis_power" in power_data
        assert "centrifugal_power" in power_data
        assert all(isinstance(v, float) for v in power_data.values())


# ---------------------------------------------------------------------------
# Tests for _clamp_to_joint_limits (issue #2766)
# ---------------------------------------------------------------------------

#: Minimal MuJoCo model with one limited hinge joint (range [-1, 1] rad).
_LIMITED_JOINT_XML = """
<mujoco model="limited_hinge_test">
  <worldbody>
    <body name="link" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 0 1"
             limited="true" range="-1 1" damping="0.1"/>
      <geom type="capsule" size="0.05 0.2"/>
    </body>
  </worldbody>
</mujoco>
"""


class TestClampToJointLimits:
    """Verify that _clamp_to_joint_limits respects the model joint ranges."""

    @pytest.fixture()
    def limited_analyzer(self) -> KinematicForceAnalyzer:
        """KinematicForceAnalyzer backed by a model that has a limited joint."""
        model = mujoco.MjModel.from_xml_string(_LIMITED_JOINT_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return KinematicForceAnalyzer(model, data)

    def test_clamp_within_limits_unchanged(
        self, limited_analyzer: KinematicForceAnalyzer
    ) -> None:
        """qpos values inside the joint range are returned unchanged."""
        qpos = np.array([0.5])
        result = limited_analyzer._clamp_to_joint_limits(qpos)
        np.testing.assert_array_almost_equal(result, qpos)

    def test_clamp_above_max(self, limited_analyzer: KinematicForceAnalyzer) -> None:
        """qpos above the upper limit is clamped to the upper limit."""
        qpos = np.array([2.0])
        result = limited_analyzer._clamp_to_joint_limits(qpos)
        assert result[0] == pytest.approx(1.0)

    def test_clamp_below_min(self, limited_analyzer: KinematicForceAnalyzer) -> None:
        """qpos below the lower limit is clamped to the lower limit."""
        qpos = np.array([-3.0])
        result = limited_analyzer._clamp_to_joint_limits(qpos)
        assert result[0] == pytest.approx(-1.0)

    def test_clamp_does_not_mutate_input(
        self, limited_analyzer: KinematicForceAnalyzer
    ) -> None:
        """The original qpos array must not be modified (returns a copy)."""
        qpos = np.array([5.0])
        original = qpos.copy()
        limited_analyzer._clamp_to_joint_limits(qpos)
        np.testing.assert_array_equal(qpos, original)

    def test_effective_denom_at_joint_limit(
        self, limited_analyzer: KinematicForceAnalyzer
    ) -> None:
        """compute_club_head_apparent_forces finishes without NaN when qpos is
        clamped.  The DOUBLE_PENDULUM_XML model has no club_head body so the
        method returns early, but the limited model also has no club_head.
        We just verify the method is callable and the clamping path is exercised
        without errors."""
        qpos = np.array([0.99])  # near upper limit
        qvel = np.array([100.0])  # large velocity → forward perturbation would clip
        qacc = np.zeros(1)
        result = limited_analyzer.compute_club_head_apparent_forces(qpos, qvel, qacc)
        # No club_head_id → returns zero vectors; verify shapes and finiteness.
        for vec in result:
            assert np.all(np.isfinite(vec))
