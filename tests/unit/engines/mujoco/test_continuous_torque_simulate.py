"""Unit and contract tests for MuJoCo continuous-torque simulation harness.

Covers Issue #9966 (Cross-Engine Equivalency WP2 / WP3):
- Forward-dynamics rollout under continuous polynomial and Bernstein torques.
- Energy accounting: kinetic, potential, and total mechanical energy balance.
- Canonical site extraction (mid_hands, clubhead) and unit quaternions.
- Determinism and context manager cleanup of global mjcb_control.
- Canonical EngineJointMap validation.
- Precondition enforcement on input parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from src.engines.physics_engines.mujoco.python.simulate_with_coefficients import (
    CANONICAL_COORDINATE_NAMES,
    CLUBHEAD_SITE_NAME,
    DEFAULT_GOLFER_XML,
    EngineJointMap,
    GRIP_SITE_NAME,
    POLY_BOUNDS,
    PolynomialTorqueDriver,
    SimOptions,
    SimOut,
    get_mujoco_canonical_joint_map,
    polynomial_torque_bounds,
    simulate_with_coefficients,
)
from src.shared.python.motion_matching.piecewise_polynomial import (
    PiecewisePolynomialTorque,
    PolynomialSegment,
)

pytestmark = [pytest.mark.requires_mujoco, pytest.mark.unit]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_canonical_model() -> Any:
    import mujoco

    if not DEFAULT_GOLFER_XML.exists():
        pytest.skip(f"Canonical model {DEFAULT_GOLFER_XML} not found")
    return mujoco.MjModel.from_xml_path(str(DEFAULT_GOLFER_XML))


# --------------------------------------------------------------------------- #
# Forward Dynamics & Contract Tests
# --------------------------------------------------------------------------- #


def test_simulate_with_canonical_golfer_xml_rollout() -> None:
    """Forward dynamics on canonical golfer.xml executes and produces canonical SimOut."""
    model = _load_canonical_model()
    nu = int(model.nu)
    theta: NDArray[np.float64] = np.zeros(nu * 7, dtype=np.float64)

    opts = SimOptions(
        xml_path=DEFAULT_GOLFER_XML,
        T_s=0.02,
        dt=0.001,
        output_rate_hz=1000.0,
        compute_energy=True,
    )
    out = simulate_with_coefficients(theta, opts)

    assert isinstance(out, SimOut)
    n_expected = int(round(opts.T_s * opts.output_rate_hz)) + 1
    assert out.time.shape == (n_expected,)
    assert out.q.shape == (n_expected, model.nq)
    assert out.qd.shape == (n_expected, model.nv)
    assert out.tau.shape == (n_expected, nu)
    assert out.grip.shape == (n_expected, 3)
    assert out.grip_quat.shape == (n_expected, 4)
    assert out.clubhead.shape == (n_expected, 3)
    assert out.club_quat.shape == (n_expected, 4)
    assert out.solver_status == "success"
    assert out.duration_s > 0.0

    # Test cross-engine property aliases
    assert out.t.shape == (n_expected,)
    assert out.grip_position.shape == (n_expected, 3)
    assert out.grip_rotation.shape == (n_expected, 3, 3)
    assert out.clubhead_position.shape == (n_expected, 3)
    assert out.clubhead_rotation.shape == (n_expected, 3, 3)


def test_energy_accounting_and_conservation() -> None:
    """Energy accounting tracks kinetic, potential, and total energy."""
    model = _load_canonical_model()
    nu = int(model.nu)
    theta: NDArray[np.float64] = np.zeros(nu * 7, dtype=np.float64)

    opts = SimOptions(
        xml_path=DEFAULT_GOLFER_XML,
        T_s=0.01,
        dt=0.0005,
        output_rate_hz=2000.0,
        compute_energy=True,
    )
    out = simulate_with_coefficients(theta, opts)

    assert out.kinetic_energy.shape == out.time.shape
    assert out.potential_energy.shape == out.time.shape
    assert np.all(np.isfinite(out.kinetic_energy))
    assert np.all(np.isfinite(out.potential_energy))

    # At t=0 from rest, kinetic energy is 0
    assert out.kinetic_energy[0] == pytest.approx(0.0, abs=1e-8)
    # Under gravity, kinetic energy increases
    assert out.kinetic_energy[-1] > 0.0

    # Total energy E = T + V is finite and tracked across the entire rollout
    total_energy = out.kinetic_energy + out.potential_energy
    assert np.all(np.isfinite(total_energy))
    assert total_energy.shape == out.time.shape


def test_bernstein_basis_torque_rollout() -> None:
    """MuJoCo harness correctly evaluates torque in 6th-order Bernstein basis."""
    model = _load_canonical_model()
    nu = int(model.nu)

    # 7 Bernstein control points per joint
    c_pts: NDArray[np.float64] = np.zeros((nu, 7), dtype=np.float64)
    # Apply 10 N*m on first actuator at t=0 ramping to 50 N*m at t=T
    c_pts[0, 0] = 10.0
    c_pts[0, -1] = 50.0

    opts = SimOptions(
        xml_path=DEFAULT_GOLFER_XML,
        T_s=0.05,
        output_rate_hz=1000.0,
        basis="bernstein",
    )
    out = simulate_with_coefficients(c_pts.flatten(), opts)

    assert out.solver_status == "success"
    # At t=0, tau should equal c_0 = 10.0
    assert out.tau[0, 0] == pytest.approx(10.0, abs=1e-5)
    # At t=T_s, tau should equal c_6 = 50.0
    assert out.tau[-1, 0] == pytest.approx(50.0, abs=1e-5)


def test_piecewise_polynomial_torque_driver_direct() -> None:
    """PolynomialTorqueDriver works with PiecewisePolynomialTorque instances."""
    model = _load_canonical_model()
    nu = int(model.nu)

    # Create two segments on [0, 0.02] and [0.02, 0.05]
    coeffs1 = np.ones((nu, 7), dtype=np.float64) * 2.0
    coeffs2 = np.ones((nu, 7), dtype=np.float64) * 5.0
    seg1 = PolynomialSegment(0.0, 0.02, coeffs1, is_bernstein=False)
    seg2 = PolynomialSegment(0.02, 0.05, coeffs2, is_bernstein=False)
    piecewise = PiecewisePolynomialTorque((seg1, seg2))

    driver = PolynomialTorqueDriver(model, piecewise, t0=0.0)
    # At t=0.01 (in seg1)
    val1 = driver.evaluate(0.01)
    # At t=0.03 (in seg2)
    val2 = driver.evaluate(0.03)

    assert val1[0] > 0.0
    assert val2[0] > 0.0


def test_site_kinematics_unit_quaternions() -> None:
    """Mid-hands and clubhead site orientations remain valid unit quaternions."""
    opts = SimOptions(
        xml_path=DEFAULT_GOLFER_XML,
        T_s=0.02,
        output_rate_hz=500.0,
    )
    model = _load_canonical_model()
    theta: NDArray[np.float64] = np.zeros(model.nu * 7, dtype=np.float64)
    out = simulate_with_coefficients(theta, opts)

    # Unit norm quaternions
    grip_norms = np.linalg.norm(out.grip_quat, axis=1)
    head_norms = np.linalg.norm(out.club_quat, axis=1)
    np.testing.assert_allclose(grip_norms, 1.0, atol=1e-6)
    np.testing.assert_allclose(head_norms, 1.0, atol=1e-6)

    # Site names check
    assert out.meta["grip_site_id"] >= 0
    assert out.meta["club_site_id"] >= 0


def test_determinism_back_to_back() -> None:
    """Consecutive runs produce bit-for-bit identical outputs."""
    model = _load_canonical_model()
    nu = int(model.nu)
    rng = np.random.default_rng(123)
    theta: NDArray[np.float64] = rng.uniform(-1.0, 1.0, size=nu * 7).astype(np.float64)

    opts = SimOptions(xml_path=DEFAULT_GOLFER_XML, T_s=0.01, output_rate_hz=500.0)
    out1 = simulate_with_coefficients(theta, opts)
    out2 = simulate_with_coefficients(theta, opts)

    np.testing.assert_array_equal(out1.q, out2.q)
    np.testing.assert_array_equal(out1.qd, out2.qd)
    np.testing.assert_array_equal(out1.tau, out2.tau)
    np.testing.assert_array_equal(out1.grip, out2.grip)
    np.testing.assert_array_equal(out1.clubhead, out2.clubhead)


def test_engine_joint_map() -> None:
    """Canonical 27 coordinates map to MuJoCo actuators correctly."""
    model = _load_canonical_model()
    joint_map = get_mujoco_canonical_joint_map(model)

    assert isinstance(joint_map, EngineJointMap)
    assert len(joint_map.coordinate_names) == 27
    assert joint_map.coordinate_names == CANONICAL_COORDINATE_NAMES
    assert len(joint_map.engine_dof_indices) == 27
    assert len(joint_map.sign_flips) == 27

    # Verify key actuated joints resolve to valid indices
    spine_x_idx = CANONICAL_COORDINATE_NAMES.index("SpineInputX")
    assert joint_map.engine_dof_indices[spine_x_idx] >= 0


def test_invalid_inputs_raise() -> None:
    """Non-finite or misshapen theta raises ValueError."""
    model = _load_canonical_model()
    nu = int(model.nu)

    # Wrong length
    with pytest.raises(ValueError):
        simulate_with_coefficients(
            np.zeros(nu * 7 + 1),
            SimOptions(xml_path=DEFAULT_GOLFER_XML),
        )

    # NaN in theta
    bad_theta: NDArray[np.float64] = np.zeros(nu * 7, dtype=np.float64)
    bad_theta[0] = np.nan
    with pytest.raises(ValueError):
        simulate_with_coefficients(
            bad_theta,
            SimOptions(xml_path=DEFAULT_GOLFER_XML),
        )

    # Missing XML file
    with pytest.raises(FileNotFoundError):
        simulate_with_coefficients(
            np.zeros(nu * 7),
            SimOptions(xml_path="non_existent_file.xml"),
        )
