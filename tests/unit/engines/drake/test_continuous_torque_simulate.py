"""Unit and contract tests for Drake continuous-torque simulation harness.

Covers Issue #9968 (Cross-Engine Equivalency WP4: Drake MultibodyPlant
Simulation Harness and Parity Driver):
- Forward-dynamics rollout under continuous 6th-order Bernstein polynomial torques.
- Energy balance accounting (kinetic + potential energy).
- Standard SimOut dataclass matching Simscape contract.
- Canonical coordinate names and EngineJointMap DOF resolution.
- Clean fallback when Drake C++ runtime (pydrake) is unavailable.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.typing import NDArray
import pytest

from src.engines.physics_engines.drake.python.simulate_with_coefficients import (
    CANONICAL_COORDINATE_NAMES,
    CLUBHEAD_FRAME_NAME,
    COEFFS_PER_JOINT,
    DEFAULT_GOLFER_URDF,
    EngineJointMap,
    GRIP_FRAME_NAME,
    POLY_BOUNDS,
    POLY_DEGREE,
    SimOptions,
    SimOut,
    SynthesizeOptions,
    evaluate_bernstein_torque,
    evaluate_polynomial_torque,
    evaluate_torque_polynomial,
    get_drake_canonical_joint_map,
    is_drake_available,
    polynomial_torque_bounds,
    simulate_with_coefficients,
    synthesize_target_from_coefficients,
)
from src.shared.python.math_utils.quaternion import rotmat_to_quat
from src.shared.python.motion_matching.piecewise_polynomial import (
    bernstein_to_power_matrix,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Canonical Coordinates & Parameter Bounds
# --------------------------------------------------------------------------- #


def test_canonical_coordinate_names_count_and_order() -> None:
    """Canonical actuation channels must have exactly 27 coordinates matching spec §3."""
    assert len(CANONICAL_COORDINATE_NAMES) == 27
    assert CANONICAL_COORDINATE_NAMES[0] == "TranslationInputX"
    assert CANONICAL_COORDINATE_NAMES[1] == "TranslationInputY"
    assert CANONICAL_COORDINATE_NAMES[2] == "TranslationInputZ"
    assert CANONICAL_COORDINATE_NAMES[3] == "HipInputX"
    assert CANONICAL_COORDINATE_NAMES[4] == "HipInputY"
    assert CANONICAL_COORDINATE_NAMES[5] == "HipInputZ"
    assert CANONICAL_COORDINATE_NAMES[6] == "SpineInputX"
    assert CANONICAL_COORDINATE_NAMES[7] == "SpineInputY"
    assert CANONICAL_COORDINATE_NAMES[8] == "TorsoInput"
    assert CANONICAL_COORDINATE_NAMES[25] == "RWInputX"
    assert CANONICAL_COORDINATE_NAMES[26] == "RWInputY"
    assert GRIP_FRAME_NAME == "mid_hands"
    assert CLUBHEAD_FRAME_NAME == "club_head"


def test_polynomial_torque_bounds_shapes_and_values() -> None:
    """polynomial_torque_bounds returns symmetric arrays with shape (n_joints * 7,)."""
    n_joints = 4
    lb, ub = polynomial_torque_bounds(n_joints)
    assert lb.shape == (n_joints * COEFFS_PER_JOINT,)
    assert ub.shape == (n_joints * COEFFS_PER_JOINT,)
    np.testing.assert_allclose(lb, -ub)
    expected_one_joint = np.asarray(POLY_BOUNDS, dtype=np.float64)
    np.testing.assert_allclose(ub[:7], expected_one_joint)

    with pytest.raises(ValueError, match="n_joints must be > 0"):
        polynomial_torque_bounds(0)


# --------------------------------------------------------------------------- #
# Bernstein & Power Polynomial Evaluation Tests
# --------------------------------------------------------------------------- #


def test_evaluate_bernstein_torque_endpoints() -> None:
    """In Bernstein basis: tau(0) = c_0 and tau(T) = c_6."""
    coeffs = np.array(
        [
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ],
        dtype=np.float64,
    )
    t0 = 0.0
    t_final = 0.35

    tau_start = evaluate_bernstein_torque(coeffs, t0, T_s=t_final, t0=t0)
    np.testing.assert_allclose(tau_start, coeffs[:, 0], atol=1e-12)

    tau_end = evaluate_bernstein_torque(coeffs, t_final, T_s=t_final, t0=t0)
    np.testing.assert_allclose(tau_end, coeffs[:, -1], atol=1e-12)


def test_evaluate_bernstein_torque_partition_of_unity() -> None:
    """Sum of Bernstein basis polynomials is identically 1 (partition of unity)."""
    n_joints = 2
    coeffs = np.ones((n_joints, COEFFS_PER_JOINT), dtype=np.float64)
    t_final = 1.0

    for t in np.linspace(0.0, t_final, 11):
        tau = evaluate_bernstein_torque(coeffs, float(t), T_s=t_final)
        np.testing.assert_allclose(tau, np.ones(n_joints), atol=1e-12)


def test_evaluate_bernstein_torque_matches_power_conversion() -> None:
    """Direct Bernstein evaluation matches power-basis Horner evaluation via M_bernstein."""
    rng = np.random.default_rng(seed=42)
    n_joints = 5
    coeffs = rng.standard_normal((n_joints, COEFFS_PER_JOINT))
    t_final = 0.4
    t0 = 0.05

    m_bern = bernstein_to_power_matrix(POLY_DEGREE)
    p_coeffs = coeffs @ m_bern
    scale_powers = t_final ** np.arange(COEFFS_PER_JOINT, dtype=np.float64)
    power_coeffs = p_coeffs / scale_powers[None, :]

    test_times = np.linspace(t0, t0 + t_final, 15)
    for t in test_times:
        direct = evaluate_bernstein_torque(coeffs, float(t), T_s=t_final, t0=t0)
        via_power = evaluate_torque_polynomial(
            power_coeffs.flatten(), float(t - t0), n_joints, basis="power"
        )
        np.testing.assert_allclose(direct, via_power, rtol=1e-11, atol=1e-12)


def test_evaluate_bernstein_torque_preconditions() -> None:
    """DbC guards on bad coefficient shapes or invalid horizons."""
    with pytest.raises(ValueError, match="2D"):
        evaluate_bernstein_torque(np.zeros(7), 0.1)

    with pytest.raises(ValueError, match="columns"):
        evaluate_bernstein_torque(np.zeros((2, 6)), 0.1)

    with pytest.raises(ValueError, match="finite"):
        evaluate_bernstein_torque(np.zeros((2, 7)), float("nan"))

    with pytest.raises(ValueError, match="positive"):
        evaluate_bernstein_torque(np.zeros((2, 7)), 0.1, T_s=0.0)


def test_evaluate_torque_polynomial_bases() -> None:
    """evaluate_torque_polynomial supports both power and bernstein bases."""
    n_joints = 2
    theta = np.zeros(n_joints * COEFFS_PER_JOINT)
    theta[0] = 5.0
    theta[6] = 20.0
    theta[7] = -3.0
    theta[13] = 10.0

    tau_power = evaluate_torque_polynomial(theta, 0.0, n_joints, basis="power")
    assert tau_power[0] == 5.0
    assert tau_power[1] == -3.0

    tau_bern = evaluate_torque_polynomial(
        theta, 0.5, n_joints, basis="bernstein", T_s=0.5
    )
    assert tau_bern[0] == 20.0
    assert tau_bern[1] == 10.0


# --------------------------------------------------------------------------- #
# SimOptions Preconditions and Contract
# --------------------------------------------------------------------------- #


def test_sim_options_contract_and_defaults() -> None:
    """SimOptions sets defaults and accepts both power and bernstein bases."""
    opts = SimOptions()
    assert opts.simulation_time_s == 0.3
    assert opts.sample_rate_hz == 1000.0
    assert opts.time_step_s == 1e-3
    assert opts.basis == "power"
    assert opts.compute_energy is True
    assert opts.compute_qdd is True

    # Check alias reconciliation
    opts_bern = SimOptions(
        T_s=0.25, basis="bernstein", output_rate_hz=2000.0, dt=0.0005
    )
    assert opts_bern.simulation_time_s == 0.25
    assert opts_bern.T_s == 0.25
    assert opts_bern.t_final == 0.25
    assert opts_bern.sample_rate_hz == 2000.0
    assert opts_bern.output_rate_hz == 2000.0
    assert opts_bern.time_step_s == 0.0005
    assert opts_bern.dt == 0.0005
    assert opts_bern.basis == "bernstein"

    with pytest.raises(ValueError, match="basis"):
        SimOptions(basis="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="T_s"):
        SimOptions(T_s=-0.1)

    with pytest.raises(ValueError, match="gravity"):
        SimOptions(gravity=(0.0, 0.0))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# SimOut Simscape Contract & Properties
# --------------------------------------------------------------------------- #


def test_sim_out_canonical_contract_and_properties() -> None:
    """SimOut exposes both Simscape standard fields and Drake property aliases."""
    n_samples = 21
    nv = 4
    time = np.linspace(0.0, 0.02, n_samples)
    q = np.zeros((n_samples, nv))
    qd = np.zeros((n_samples, nv))
    qdd = np.ones((n_samples, nv)) * 2.5
    tau = np.ones((n_samples, nv)) * 10.0
    grip = np.tile(np.array([0.1, 0.2, 0.3]), (n_samples, 1))
    grip_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_samples, 1))
    clubhead = np.tile(np.array([0.5, 0.6, 0.7]), (n_samples, 1))
    club_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_samples, 1))
    ke = np.linspace(0.0, 10.0, n_samples)
    pe = np.linspace(5.0, 15.0, n_samples)

    out = SimOut(
        time=time,
        q=q,
        qd=qd,
        qdd=qdd,
        tau=tau,
        grip=grip,
        grip_quat=grip_quat,
        clubhead=clubhead,
        club_quat=club_quat,
        solver_status="success",
        duration_s=0.045,
        kinetic_energy=ke,
        potential_energy=pe,
        meta={"test": True},
    )

    # Standard Simscape / cross-engine contract fields
    assert np.array_equal(out.time, time)
    assert np.array_equal(out.q, q)
    assert np.array_equal(out.qd, qd)
    assert np.array_equal(out.qdd, qdd)
    assert np.array_equal(out.tau, tau)
    assert np.array_equal(out.grip, grip)
    assert np.array_equal(out.grip_quat, grip_quat)
    assert np.array_equal(out.clubhead, clubhead)
    assert np.array_equal(out.club_quat, club_quat)
    assert out.solver_status == "success"
    assert out.duration_s == 0.045
    assert np.array_equal(out.kinetic_energy, ke)
    assert np.array_equal(out.potential_energy, pe)
    assert out.meta["test"] is True
    assert out.metadata["test"] is True

    # Property aliases
    assert np.array_equal(out.t, time)
    assert np.array_equal(out.grip_position, grip)
    assert np.array_equal(out.clubhead_position, clubhead)
    assert out.grip_rotation.shape == (n_samples, 3, 3)
    assert out.clubhead_rotation.shape == (n_samples, 3, 3)
    np.testing.assert_allclose(out.grip_rotation[0], np.eye(3), atol=1e-12)


def test_sim_out_legacy_keyword_compatibility() -> None:
    """SimOut can be instantiated with legacy keyword arguments without error."""
    n_samples = 11
    nv = 2
    t = np.linspace(0.0, 0.01, n_samples)
    q = np.zeros((n_samples, nv))
    qd = np.zeros((n_samples, nv))
    tau = np.zeros((n_samples, nv))
    grip_pos = np.ones((n_samples, 3))
    grip_rot = np.broadcast_to(np.eye(3), (n_samples, 3, 3)).copy()
    club_pos = np.ones((n_samples, 3)) * 2.0
    club_rot = np.broadcast_to(np.eye(3), (n_samples, 3, 3)).copy()

    out = SimOut(
        t=t,
        q=q,
        qd=qd,
        tau=tau,
        grip_position=grip_pos,
        grip_rotation=grip_rot,
        clubhead_position=club_pos,
        clubhead_rotation=club_rot,
        kinetic_energy=np.zeros(n_samples),
        potential_energy=np.zeros(n_samples),
    )

    assert np.array_equal(out.time, t)
    assert np.array_equal(out.grip, grip_pos)
    assert np.array_equal(out.clubhead, club_pos)
    assert out.grip_quat.shape == (n_samples, 4)
    assert out.club_quat.shape == (n_samples, 4)
    np.testing.assert_allclose(out.grip_quat[:, 0], 1.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# EngineJointMap Contract Tests
# --------------------------------------------------------------------------- #


def test_get_drake_canonical_joint_map() -> None:
    """get_drake_canonical_joint_map maps canonical coordinates to DOF indices."""
    mock_plant = MagicMock()

    def mock_has_joint(name: str) -> bool:
        return name in {
            "pelvis_to_lumbar1_intermediate",
            "lumbar1_intermediate_to_lumbar1",
            "lumbar3_to_thorax1",
        }

    mock_plant.HasJointNamed.side_effect = mock_has_joint

    class MockJoint:
        def __init__(self, idx: int) -> None:
            self._idx = idx

        def velocity_start(self) -> int:
            return self._idx

    mock_plant.GetJointByName.side_effect = lambda name: {
        "pelvis_to_lumbar1_intermediate": MockJoint(6),
        "lumbar1_intermediate_to_lumbar1": MockJoint(7),
        "lumbar3_to_thorax1": MockJoint(8),
    }[name]

    mock_plant.HasJointActuatorNamed.return_value = False

    joint_map = get_drake_canonical_joint_map(mock_plant)
    assert isinstance(joint_map, EngineJointMap)
    assert len(joint_map.coordinate_names) == 27
    assert len(joint_map.engine_dof_indices) == 27
    assert len(joint_map.sign_flips) == 27

    # SpineInputX (index 6) maps to pelvis_to_lumbar1_intermediate -> velocity_start=6
    assert joint_map.coordinate_names[6] == "SpineInputX"
    assert joint_map.engine_dof_indices[6] == 6

    # SpineInputY (index 7) maps to lumbar1_intermediate_to_lumbar1 -> velocity_start=7
    assert joint_map.coordinate_names[7] == "SpineInputY"
    assert joint_map.engine_dof_indices[7] == 7

    # Unmapped/absent joint maps to -1
    assert joint_map.engine_dof_indices[0] == -1


# --------------------------------------------------------------------------- #
# Runtime Availability Probe & Fallback
# --------------------------------------------------------------------------- #


def test_is_drake_available_probe() -> None:
    """is_drake_available returns a boolean reflecting pydrake installation."""
    result = is_drake_available()
    assert isinstance(result, bool)

    with patch.dict(sys.modules, {"pydrake": None}):
        assert is_drake_available() is False


def test_simulate_with_coefficients_missing_runtime_raises_import_error() -> None:
    """When Drake is unavailable, simulate_with_coefficients raises clean ImportError."""
    theta = np.zeros(19 * COEFFS_PER_JOINT)
    with patch(
        "src.engines.physics_engines.drake.python.motion_matching.simulate.is_drake_available",
        return_value=False,
    ):
        with pytest.raises(
            ImportError, match="Drake runtime \\(pydrake\\) is required"
        ):
            simulate_with_coefficients(theta)


# --------------------------------------------------------------------------- #
# Mocked Drake Forward Rollout with Bernstein Torque & Energy Accounting
# --------------------------------------------------------------------------- #


@pytest.fixture
def _mock_drake_harness() -> Generator[dict[str, Any], None, None]:
    """Provide a full mocked pydrake environment for forward-dynamics testing."""
    keys = [
        "pydrake",
        "pydrake.all",
        "pydrake.multibody",
        "pydrake.multibody.plant",
        "pydrake.multibody.parsing",
        "pydrake.multibody.tree",
        "pydrake.systems",
        "pydrake.systems.analysis",
        "pydrake.systems.framework",
        "pydrake.systems.primitives",
        "pydrake.math",
    ]
    mocks = {k: MagicMock() for k in keys}

    plant = MagicMock(name="MultibodyPlant")
    n_q, n_v, n_act = 27, 27, 21
    plant.num_positions.return_value = n_q
    plant.num_velocities.return_value = n_v
    plant.num_actuators.return_value = n_act
    plant.num_multibody_states.return_value = n_q + n_v
    plant.GetPositions.return_value = np.zeros(n_q)
    plant.GetVelocities.return_value = np.zeros(n_v)
    body_mock = MagicMock()
    body_mock.body_frame.return_value = MagicMock(name="BodyFrame")
    plant.HasBodyNamed.return_value = True
    plant.GetBodyByName.return_value = body_mock

    tf_mock = MagicMock()
    tf_mock.translation.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    quat_mock = MagicMock()
    quat_mock.w.return_value = 1.0
    quat_mock.x.return_value = 0.0
    quat_mock.y.return_value = 0.0
    quat_mock.z.return_value = 0.0
    tf_mock.rotation.return_value.ToQuaternion.return_value = quat_mock
    plant.CalcRelativeTransform.return_value = tf_mock

    plant.CalcKineticEnergy.return_value = 1.234
    plant.CalcPotentialEnergy.return_value = 5.678

    scene_graph = MagicMock(name="SceneGraph")
    mocks["pydrake.multibody.plant"].AddMultibodyPlantSceneGraph = MagicMock(
        return_value=(plant, scene_graph)
    )

    class _MockLeafSystem:
        def __init__(self) -> None:
            self._ports: list[Any] = []

        def DeclareVectorOutputPort(self, name: str, model_vec: Any, calc: Any) -> Any:
            p = MagicMock(name=f"Port[{name}]")
            self._ports.append(p)
            return p

        def get_output_port(self, idx: int) -> Any:
            return self._ports[idx] if self._ports else MagicMock()

    framework = mocks["pydrake.systems.framework"]
    framework.LeafSystem = _MockLeafSystem
    framework.BasicVector = MagicMock(side_effect=lambda n: MagicMock())
    framework.DiagramBuilder = MagicMock(return_value=MagicMock())

    sim_inst = MagicMock(name="Simulator")
    diag_ctx = MagicMock(name="DiagramContext")
    sim_inst.get_context.return_value = diag_ctx
    mocks["pydrake.systems.analysis"].Simulator = MagicMock(return_value=sim_inst)
    mocks["pydrake.systems.primitives"].VectorLogSink = MagicMock(
        return_value=MagicMock()
    )

    plant_ctx = MagicMock(name="PlantContext")
    plant.GetMyMutableContextFromRoot.return_value = plant_ctx
    plant.GetMyContextFromRoot.return_value = plant_ctx

    b_inst = framework.DiagramBuilder.return_value
    b_inst.Build.return_value = MagicMock(name="Diagram")
    b_inst.Build.return_value.CreateDefaultContext.return_value = diag_ctx

    with (
        patch.dict(sys.modules, mocks),
        patch(
            "src.engines.physics_engines.drake.python.motion_matching.simulate.is_drake_available",
            return_value=True,
        ),
    ):
        yield {"plant": plant, "simulator": sim_inst, "n_act": n_act}


def test_mocked_simulate_with_bernstein_basis_and_energy(
    _mock_drake_harness: dict[str, Any],
) -> None:
    """Forward dynamics rollout with 6th-order continuous Bernstein polynomial torques."""
    n_act = _mock_drake_harness["n_act"]
    # 7 Bernstein control points per joint
    c_pts = np.zeros((n_act, COEFFS_PER_JOINT), dtype=np.float64)
    c_pts[0, 0] = 10.0
    c_pts[0, -1] = 50.0

    opts = SimOptions(
        simulation_time_s=0.02,
        sample_rate_hz=1000.0,
        basis="bernstein",
        compute_energy=True,
    )
    out = simulate_with_coefficients(c_pts.flatten(), opts)

    assert isinstance(out, SimOut)
    assert out.solver_status == "success"
    n_expected = 21
    assert out.time.shape == (n_expected,)
    assert out.tau.shape == (n_expected, n_act)
    assert out.kinetic_energy.shape == (n_expected,)
    assert out.potential_energy.shape == (n_expected,)

    # At t=0, tau should equal c_0 = 10.0
    assert out.tau[0, 0] == pytest.approx(10.0, abs=1e-5)
    # At t=T_s, tau should equal c_6 = 50.0
    assert out.tau[-1, 0] == pytest.approx(50.0, abs=1e-5)

    # Verify energy values recorded from plant
    assert out.kinetic_energy[0] == pytest.approx(1.234)
    assert out.potential_energy[0] == pytest.approx(5.678)


def test_synthesize_target_from_coefficients_helper(
    _mock_drake_harness: dict[str, Any],
) -> None:
    """synthesize_target_from_coefficients constructs a validated ClubTarget."""
    n_act = _mock_drake_harness["n_act"]
    theta = np.zeros(n_act * COEFFS_PER_JOINT)

    opts = SynthesizeOptions(
        sim_options=SimOptions(simulation_time_s=0.01, sample_rate_hz=1000.0),
        subject_id="test_subject",
        trial_id="test_trial",
    )
    target = synthesize_target_from_coefficients(theta, opts)

    assert target.source.format == "synthetic"
    assert target.source.subject_id == "test_subject"
    assert target.source.trial_id == "test_trial"
    assert target.time.shape[0] == 11
    assert target.butt.shape == (11, 3)
    assert target.clubhead.shape == (11, 3)
    assert target.club_quat.shape == (11, 4)


# --------------------------------------------------------------------------- #
# Live Real Drake MultibodyPlant Test (Run when pydrake is installed)
# --------------------------------------------------------------------------- #


@pytest.mark.requires_drake
def test_live_drake_continuous_torque_rollout() -> None:
    """Live Drake MultibodyPlant execution if pydrake is installed on host."""
    if not is_drake_available():
        pytest.skip("pydrake runtime is not installed on host")

    from pydrake.multibody.plant import MultibodyPlant
    from pydrake.systems.framework import DiagramBuilder

    builder = DiagramBuilder()
    plant = builder.AddSystem(MultibodyPlant(time_step=1e-3))
    urdf = DEFAULT_GOLFER_URDF
    if not urdf.exists():
        pytest.skip(f"Golfer URDF {urdf} does not exist")

    from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
        load_humanoid_into_plant,
    )

    load_humanoid_into_plant(plant, urdf)
    plant.Finalize()
    n_act = int(plant.num_actuators())

    theta = np.zeros(n_act * COEFFS_PER_JOINT)
    theta[0] = 5.0
    theta[6] = 5.0

    opts = SimOptions(
        urdf_path=urdf,
        simulation_time_s=0.01,
        sample_rate_hz=1000.0,
        basis="bernstein",
        compute_energy=True,
    )
    out = simulate_with_coefficients(theta, opts)
    assert out.solver_status == "success"
    assert out.time.shape[0] == 11
    assert out.tau.shape == (11, n_act)
    assert np.all(np.isfinite(out.q))
    assert np.all(np.isfinite(out.qd))
