"""Unit and contract tests for Pinocchio continuous-torque simulation harness.

Covers Issue #9967 (Cross-Engine Equivalency WP3: Pinocchio Articulated-Body
Simulation Harness and Parity Driver):
- Forward-dynamics rollout under continuous 6th-order Bernstein polynomial torques.
- Energy balance accounting (kinetic + potential energy).
- Standard SimOut dataclass matching Simscape contract.
- Canonical coordinate names and EngineJointMap DOF resolution.
- Clean fallback when Pinocchio C++ runtime is unavailable.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
from numpy.typing import NDArray
import pytest

from src.engines.physics_engines.pinocchio.python.simulate_with_coefficients import (
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
    evaluate_bernstein_torque,
    evaluate_polynomial_torque,
    get_pinocchio_canonical_joint_map,
    is_pinocchio_available,
    polynomial_torque_bounds,
    simulate_with_coefficients,
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
# Bernstein Polynomial Evaluation Tests
# --------------------------------------------------------------------------- #


def test_evaluate_bernstein_torque_endpoints() -> None:
    """In Bernstein basis: tau(0) = c_0 and tau(T) = c_6."""
    n_joints = 3
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
        via_power = evaluate_polynomial_torque(power_coeffs, float(t - t0))
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


# --------------------------------------------------------------------------- #
# SimOptions Preconditions and Contract
# --------------------------------------------------------------------------- #


def test_sim_options_contract_and_defaults() -> None:
    """SimOptions sets defaults and accepts both power and bernstein bases."""
    opts = SimOptions()
    assert opts.t_final == 1.0
    assert opts.dt == 1e-3
    assert opts.basis == "power"
    assert opts.T_s == 1.0
    assert opts.compute_energy is True
    assert opts.compute_qdd is True

    # Custom options with Bernstein basis
    opts_bern = SimOptions(T_s=0.25, basis="bernstein", output_rate_hz=2000.0)
    assert opts_bern.t_final == 0.25
    assert opts_bern.T_s == 0.25
    assert opts_bern.dt == 0.0005
    assert opts_bern.basis == "bernstein"

    with pytest.raises(ValueError, match="basis"):
        SimOptions(basis="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="T_s"):
        SimOptions(T_s=-0.1)


# --------------------------------------------------------------------------- #
# SimOut Simscape Contract & Properties
# --------------------------------------------------------------------------- #


def test_sim_out_canonical_contract_and_properties() -> None:
    """SimOut exposes both Simscape standard fields and Pinocchio property aliases."""
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


def test_get_pinocchio_canonical_joint_map() -> None:
    """get_pinocchio_canonical_joint_map maps canonical coordinates to DOF indices."""
    mock_model = MagicMock()

    def mock_exist_joint(name: str) -> bool:
        return name in {
            "pelvis_to_lumbar1_intermediate",
            "lumbar1_intermediate_to_lumbar1",
            "lumbar3_to_thorax1",
        }

    mock_model.existJointName.side_effect = mock_exist_joint

    class MockJoint:
        def __init__(self, idx_v: int) -> None:
            self.idx_v = idx_v

    mock_model.getJointId.side_effect = lambda name: {
        "pelvis_to_lumbar1_intermediate": 1,
        "lumbar1_intermediate_to_lumbar1": 2,
        "lumbar3_to_thorax1": 3,
    }[name]
    mock_model.joints = {1: MockJoint(0), 2: MockJoint(1), 3: MockJoint(2)}

    joint_map = get_pinocchio_canonical_joint_map(mock_model)
    assert isinstance(joint_map, EngineJointMap)
    assert len(joint_map.coordinate_names) == 27
    assert len(joint_map.engine_dof_indices) == 27
    assert len(joint_map.sign_flips) == 27

    # SpineInputX (index 6) maps to pelvis_to_lumbar1_intermediate -> idx_v=0
    assert joint_map.coordinate_names[6] == "SpineInputX"
    assert joint_map.engine_dof_indices[6] == 0

    # Unmapped/absent joint maps to -1
    assert joint_map.engine_dof_indices[0] == -1


# --------------------------------------------------------------------------- #
# Forward Dynamics & Energy Balance Verification
# --------------------------------------------------------------------------- #


def test_rk4_step_energy_conservation_on_harmonic_oscillator() -> None:
    """RK4 integration on conservative system conserves mechanical energy."""
    from src.engines.physics_engines.pinocchio.python.motion_matching.simulate import (
        _rk4_step,
    )

    # Consider 1-DOF harmonic oscillator: qdd = -omega^2 * q.
    # Total mechanical energy: E = 0.5 * qd^2 + 0.5 * omega^2 * q^2.
    omega = 2.0 * np.pi  # 1 Hz

    mock_pin = MagicMock()

    def mock_aba(
        model: Any,
        data: Any,
        q: NDArray[np.float64],
        qd: NDArray[np.float64],
        tau: NDArray[np.float64],
    ) -> None:
        data.ddq = -(omega**2) * q + tau

    mock_pin.aba = mock_aba

    class MockData:
        def __init__(self) -> None:
            self.ddq = np.zeros(1)

    model = MagicMock()
    data = MockData()
    coeffs = np.zeros((1, COEFFS_PER_JOINT), dtype=np.float64)  # Zero torque

    q = np.array([1.0], dtype=np.float64)
    qd = np.array([0.0], dtype=np.float64)
    dt = 1e-3
    n_steps = 1000

    e_initial = float(0.5 * (qd[0] ** 2) + 0.5 * (omega**2) * (q[0] ** 2))

    for step in range(n_steps):
        t = step * dt
        q, qd, tau, qdd = _rk4_step(mock_pin, model, data, coeffs, q, qd, t, dt)

    e_final = float(0.5 * (qd[0] ** 2) + 0.5 * (omega**2) * (q[0] ** 2))

    # Classical RK4 on 1s harmonic oscillation with dt=1ms conserves energy within < 1e-6 relative error
    rel_error = abs(e_final - e_initial) / e_initial
    assert rel_error < 1e-6, f"Energy drift {rel_error:.2e} exceeded threshold"


def test_simulate_with_coefficients_missing_runtime_raises_import_error() -> None:
    """When Pinocchio is unavailable, simulate_with_coefficients raises clean ImportError."""
    if is_pinocchio_available():
        pytest.skip(
            "Pinocchio is installed; testing missing runtime branch not applicable"
        )

    theta = np.zeros(27 * COEFFS_PER_JOINT)
    with pytest.raises(ImportError, match="Pinocchio C\\+\\+ runtime"):
        simulate_with_coefficients(theta)


def test_simulate_with_coefficients_real_urdf_if_available() -> None:
    """Forward dynamics rollout with Bernstein polynomial torques on real Pinocchio."""
    if not is_pinocchio_available():
        pytest.skip("Pinocchio C++ runtime not installed")

    if not DEFAULT_GOLFER_URDF.exists():
        pytest.skip(f"URDF {DEFAULT_GOLFER_URDF} not found")

    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(DEFAULT_GOLFER_URDF))
    nv = int(model.nv)

    # 6th-order continuous Bernstein coefficients
    theta = np.zeros(nv * COEFFS_PER_JOINT, dtype=np.float64)
    # Apply gentle torque to first actuated joint
    theta[0] = 5.0
    theta[6] = 5.0

    opts = SimOptions(
        urdf_path=DEFAULT_GOLFER_URDF,
        t_final=0.01,
        dt=0.001,
        basis="bernstein",
        compute_energy=True,
        compute_qdd=True,
    )
    out = simulate_with_coefficients(theta, opts)

    assert isinstance(out, SimOut)
    assert out.solver_status == "success"
    assert out.time.shape[0] == 11
    assert out.q.shape == (11, model.nq)
    assert out.qd.shape == (11, nv)
    assert out.qdd.shape == (11, nv)
    assert out.tau.shape == (11, nv)
    assert out.grip.shape == (11, 3)
    assert out.grip_quat.shape == (11, 4)
    assert out.clubhead.shape == (11, 3)
    assert out.club_quat.shape == (11, 4)
    assert out.kinetic_energy.shape == (11,)
    assert out.potential_energy.shape == (11,)
    assert np.all(np.isfinite(out.q))
    assert np.all(np.isfinite(out.qd))
    assert np.all(np.isfinite(out.qdd))
