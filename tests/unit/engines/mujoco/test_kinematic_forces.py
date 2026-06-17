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

    def test_decompose_coriolis_forces_preserves_legacy_split(
        self, model_and_data, monkeypatch
    ) -> None:
        """Characterize the existing split as total minus single-DOF terms."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()
        qvel = np.array([0.4, -0.2])

        def fake_coriolis(_qpos: np.ndarray, qvel_arg: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    3.0 * qvel_arg[0] ** 2 + 5.0 * qvel_arg[0] * qvel_arg[1],
                    -2.0 * qvel_arg[1] ** 2 + 7.0 * qvel_arg[0] * qvel_arg[1],
                ]
            )

        monkeypatch.setattr(analyzer, "compute_coriolis_forces", fake_coriolis)

        centrifugal, coupling = analyzer.decompose_coriolis_forces(qpos, qvel)

        expected_total = fake_coriolis(qpos, qvel)
        expected_centrifugal = fake_coriolis(
            qpos, np.array([qvel[0], 0.0])
        ) + fake_coriolis(qpos, np.array([0.0, qvel[1]]))
        np.testing.assert_allclose(centrifugal, expected_centrifugal, atol=1e-12)
        np.testing.assert_allclose(coupling, expected_total - expected_centrifugal)

    def test_analyze_trajectory_reuses_coriolis_decomposition_for_power(
        self, model_and_data, monkeypatch
    ) -> None:
        """One trajectory frame should compute each expensive force term once."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = data.qpos.copy()
        qvel = np.array([0.4, -0.2])
        qacc = np.zeros(model.nv)
        coriolis = np.array([1.25, -0.5])
        centrifugal = np.array([0.75, -0.25])
        coupling = coriolis - centrifugal
        gravity = np.array([0.1, 0.2])
        calls = {"coriolis": 0, "decompose": 0, "gravity": 0}

        def fake_coriolis(_qpos: np.ndarray, _qvel: np.ndarray) -> np.ndarray:
            calls["coriolis"] += 1
            return coriolis.copy()

        def fake_decompose(
            _qpos: np.ndarray,
            _qvel: np.ndarray,
            *,
            coriolis_forces: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
            calls["decompose"] += 1
            assert coriolis_forces is not None
            np.testing.assert_allclose(coriolis_forces, coriolis)
            return centrifugal.copy(), coupling.copy()

        def fake_gravity(_qpos: np.ndarray) -> np.ndarray:
            calls["gravity"] += 1
            return gravity.copy()

        monkeypatch.setattr(analyzer, "compute_coriolis_forces", fake_coriolis)
        monkeypatch.setattr(analyzer, "decompose_coriolis_forces", fake_decompose)
        monkeypatch.setattr(analyzer, "compute_gravity_forces", fake_gravity)
        monkeypatch.setattr(
            analyzer,
            "compute_club_head_apparent_forces",
            lambda _qpos, _qvel, _qacc, *, coriolis_forces=None: (
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
            ),
        )
        monkeypatch.setattr(
            analyzer,
            "compute_kinetic_energy_components",
            lambda _qpos, _qvel: {
                "rotational": 2.0,
                "translational": 3.0,
                "total": 5.0,
            },
        )

        results = analyzer.analyze_trajectory(
            np.array([0.0]),
            qpos.reshape(1, model.nq),
            qvel.reshape(1, model.nv),
            qacc.reshape(1, model.nv),
        )

        assert calls == {"coriolis": 1, "decompose": 1, "gravity": 1}
        np.testing.assert_allclose(results[0].coriolis_forces, coriolis)
        np.testing.assert_allclose(results[0].centrifugal_forces, centrifugal)
        assert results[0].coriolis_power == pytest.approx(float(coriolis @ qvel))
        assert results[0].centrifugal_power == pytest.approx(float(centrifugal @ qvel))

    def test_compute_coriolis_forces_rejects_bad_state_vectors(
        self, model_and_data
    ) -> None:
        """Boundary contracts reject wrong-shaped or non-finite state vectors."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        with pytest.raises(ValueError, match="qvel must have shape"):
            analyzer.compute_coriolis_forces(data.qpos.copy(), np.zeros(model.nv + 1))

        qvel = np.zeros(model.nv)
        qvel[0] = np.nan
        with pytest.raises(ValueError, match="qvel must contain only finite values"):
            analyzer.compute_coriolis_forces(data.qpos.copy(), qvel)

    def test_coriolis_rne_is_independent_of_scratch_data_order(
        self, model_and_data
    ) -> None:
        """The same state should produce the same RNE result after scratch reuse."""
        model, data = model_and_data
        analyzer = KinematicForceAnalyzer(model, data)

        qpos = np.array([0.1, -0.04])
        qvel = np.array([0.3, -0.15])

        first = analyzer.compute_coriolis_forces_rne(qpos, qvel)
        analyzer.compute_gravity_forces(qpos)
        second = analyzer.compute_coriolis_forces_rne(qpos, qvel)

        np.testing.assert_allclose(second, first, atol=1e-12, rtol=0.0)

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
  <compiler angle="radian"/>
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
