"""Unit tests for MuJoCoPerturbationAnalyzer (#1980).

All tests run when mujoco is installed.  Tests skip gracefully when not.

Design by Contract
------------------
- Pre: set_base_torque_profile() must be called before run_batch() / perturb_torque().
- Post: run_batch() returns PerturbationSummary with all MANDATORY_METRICS.
- Post: extract_metrics() returns all MANDATORY_METRICS for valid MuJoCoSimResult.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.engines.physics_engines.mujoco.python.perturbation.analyzer import (
    MANDATORY_METRICS,
    MuJoCoSimResult,
)
from src.shared.python.pendulum_simulator.perturbation_analysis import (
    perturb_torque_coeffs,
)
from src.shared.python.perturbation.analyzer_base import ComparisonReport
from src.shared.python.perturbation.config import PerturbationConfig

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

try:
    import mujoco as _mujoco  # noqa: F401

    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False

_skip_no_mujoco = pytest.mark.skipif(
    not _MUJOCO_AVAILABLE, reason="mujoco not installed"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SMALL_CONFIG = PerturbationConfig(
    n_trials=3,
    noise_type="white",
    noise_amplitude=0.05,
    perturb_mode="additive",
    seed=42,
)

_ZERO_PROFILE: dict = {"coeffs": [[0.0, 0.0], [0.0, 0.0]]}


# ---------------------------------------------------------------------------
# Pure-Python helpers (no mujoco required)
# ---------------------------------------------------------------------------


class TestMandatoryMetrics:
    def test_mujoco_perturbation_analyzer_non_empty(self) -> None:
        assert len(MANDATORY_METRICS) >= 10

    def test_mujoco_perturbation_analyzer_contains_required_names(self) -> None:
        required = {
            "end_effector_position_final",
            "end_effector_velocity_final",
            "end_effector_speed_final",
            "peak_end_effector_speed",
            "total_energy_final",
            "joint_angles_final",
            "joint_velocities_final",
            "trajectory_rmse",
            "trajectory_max_deviation",
            "motion_duration",
        }
        assert required.issubset(set(MANDATORY_METRICS))

    def test_mujoco_perturbation_analyzer_no_duplicates(self) -> None:
        assert len(MANDATORY_METRICS) == len(set(MANDATORY_METRICS))


class TestPerturbCoeffsByMode:
    def test_mujoco_perturbation_analyzer_additive_zero_amplitude_no_change(
        self,
    ) -> None:
        coeffs = [[1.0, 2.0], [3.0, 4.0]]
        config = PerturbationConfig(
            n_trials=1, noise_amplitude=0.0, noise_type="white", perturb_mode="additive"
        )
        result = perturb_torque_coeffs(
            coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            perturb_mode=config.perturb_mode,
            seed=0,
        )
        flat_orig = [c for j in coeffs for c in j]
        flat_res = [c for j in result for c in j]
        assert all(abs(a - b) < 1e-12 for a, b in zip(flat_orig, flat_res, strict=True))

    def test_mujoco_perturbation_analyzer_multiplicative_same_shape(self) -> None:
        coeffs = [[1.0, 2.0], [3.0, 4.0]]
        config = PerturbationConfig(
            n_trials=1,
            noise_amplitude=0.1,
            noise_type="white",
            perturb_mode="multiplicative",
        )
        result = perturb_torque_coeffs(
            coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            perturb_mode=config.perturb_mode,
            seed=0,
        )
        assert len(result) == len(coeffs)
        for orig, res in zip(coeffs, result, strict=True):
            assert len(res) == len(orig)

    def test_mujoco_perturbation_analyzer_both_mode_same_shape(self) -> None:
        coeffs = [[1.0, 2.0], [3.0, 4.0]]
        config = PerturbationConfig(
            n_trials=1, noise_amplitude=0.1, noise_type="white", perturb_mode="both"
        )
        result = perturb_torque_coeffs(
            coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            perturb_mode=config.perturb_mode,
            seed=0,
        )
        assert len(result) == len(coeffs)

    def test_mujoco_perturbation_analyzer_reproducible_with_seed(self) -> None:
        coeffs = [[1.0, 2.0], [3.0, 4.0]]
        config = PerturbationConfig(
            n_trials=1, noise_amplitude=0.3, noise_type="white", perturb_mode="additive"
        )
        r1 = perturb_torque_coeffs(
            coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            perturb_mode=config.perturb_mode,
            seed=5,
        )
        r2 = perturb_torque_coeffs(
            coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            perturb_mode=config.perturb_mode,
            seed=5,
        )
        assert r1 == r2


class TestMuJoCoSimResult:
    def _make(self, n: int = 5, nq: int = 2, nv: int = 2) -> MuJoCoSimResult:
        t = np.linspace(0.0, 1.0, n)
        qpos = np.zeros((n, nq))
        qvel = np.zeros((n, nv))
        ee_pos = np.zeros((n, 3))
        ee_vel = np.zeros((n, 3))
        ke = np.ones(n)
        pe = np.zeros(n)
        return MuJoCoSimResult(
            t=t,
            qpos_traj=qpos,
            qvel_traj=qvel,
            ee_pos_traj=ee_pos,
            ee_vel_traj=ee_vel,
            kinetic_energy_traj=ke,
            potential_energy_traj=pe,
        )

    def test_mujoco_perturbation_analyzer_n_steps(self) -> None:
        r = self._make(n=10)
        assert r.n_steps == 10

    def test_mujoco_perturbation_analyzer_trajectory_shapes(self) -> None:
        r = self._make(n=5, nq=2, nv=2)
        assert r.qpos_traj.shape == (5, 2)
        assert r.qvel_traj.shape == (5, 2)


class TestComparisonReport:
    def test_mujoco_perturbation_analyzer_default_fields(self) -> None:
        report = ComparisonReport(winner="A", confidence=0.9)
        assert report.winner == "A"
        assert report.confidence == 0.9
        assert isinstance(report.metric_comparisons, dict)
        assert isinstance(report.pvalues, dict)


# ---------------------------------------------------------------------------
# MuJoCo-dependent tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analyzer():  # type: ignore[no-untyped-def]
    from src.engines.physics_engines.mujoco.python.perturbation.analyzer import (
        MuJoCoPerturbationAnalyzer,
    )

    return MuJoCoPerturbationAnalyzer(t_end=0.1)


@pytest.fixture(scope="module")
def analyzer_with_profile(analyzer):  # type: ignore[no-untyped-def]
    analyzer.set_base_torque_profile(_ZERO_PROFILE)
    return analyzer


# ---------------------------------------------------------------------------
# Polynomial control vectorization parity (#7559)
# ---------------------------------------------------------------------------


def _vectorized_horner_ctrl(coeffs, nu, t_values):
    """Mirror analyzer._simulate's padded-matrix Horner control evaluation."""
    n_actuators_coeff = len(coeffs)
    max_order = 1
    for j in range(min(nu, n_actuators_coeff)):
        max_order = max(max_order, len(coeffs[j]))
    poly_matrix = np.zeros((nu, max_order), dtype=np.float64)
    for j in range(nu):
        if j < n_actuators_coeff and len(coeffs[j]) > 0:
            desc = np.asarray(coeffs[j][::-1], dtype=np.float64)
            poly_matrix[j, max_order - desc.size :] = desc

    out = []
    ctrl = np.zeros(nu, dtype=np.float64)
    for t in t_values:
        ctrl[:] = poly_matrix[:, 0]
        for col in range(1, max_order):
            ctrl *= t
            ctrl += poly_matrix[:, col]
        out.append(ctrl.copy())
    return np.array(out)


def test_vectorized_poly_control_matches_polyval():
    """Vectorized padded Horner must equal per-actuator np.polyval (ragged)."""
    nu = 4
    # Ragged ascending-power coeff lists (some shorter, one empty, one extra).
    coeffs = [
        [1.0, 2.0, 3.0],  # 1 + 2t + 3t^2
        [0.5],  # constant
        [],  # padded to zero
        [0.0, -1.0, 0.0, 4.0],  # -t + 4t^3
    ]
    t_values = [0.0, 0.01, 0.5, 1.0, 2.5]

    got = _vectorized_horner_ctrl(coeffs, nu, t_values)

    expected = np.zeros((len(t_values), nu))
    for ti, t in enumerate(t_values):
        for j in range(nu):
            if j < len(coeffs) and len(coeffs[j]) > 0:
                expected[ti, j] = float(np.polyval(coeffs[j][::-1], t))
            else:
                expected[ti, j] = 0.0

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_vectorized_poly_control_more_actuators_than_coeffs():
    """Actuators beyond provided coeffs must produce zero control."""
    nu = 5
    coeffs = [[2.0, 1.0], [3.0]]  # only 2 actuators specified
    got = _vectorized_horner_ctrl(coeffs, nu, [0.7])
    assert got.shape == (1, nu)
    np.testing.assert_allclose(got[0, 2:], 0.0, atol=1e-12)
    np.testing.assert_allclose(got[0, 0], np.polyval([1.0, 2.0], 0.7), atol=1e-12)
    np.testing.assert_allclose(got[0, 1], 3.0, atol=1e-12)


# ---------------------------------------------------------------------------
# EE finite-difference velocity vectorization parity (#7563)
# ---------------------------------------------------------------------------


def _ee_vel_scalar_reference(t_arr, ee_pos_arr):
    """The exact pre-optimization per-row finite-difference loop."""
    ee_vel = np.zeros_like(ee_pos_arr)
    for i in range(1, len(t_arr)):
        dt_i = max(t_arr[i] - t_arr[i - 1], 1e-12)
        ee_vel[i] = (ee_pos_arr[i] - ee_pos_arr[i - 1]) / dt_i
    return ee_vel


def _ee_vel_vectorized(t_arr, ee_pos_arr):
    """Mirror analyzer._simulate's vectorized EE velocity finite difference."""
    ee_vel = np.zeros_like(ee_pos_arr)
    if len(t_arr) > 1:
        dts = np.maximum(np.diff(t_arr), 1e-12)[:, None]
        ee_vel[1:] = np.diff(ee_pos_arr, axis=0) / dts
    return ee_vel


def test_ee_velocity_vectorized_matches_scalar():
    """Vectorized EE finite difference must equal the per-row loop."""
    rng = np.random.default_rng(7563)
    # Uneven timestamps including a near-duplicate to exercise the 1e-12 clamp.
    t_arr = np.array([0.0, 0.01, 0.01 + 1e-15, 0.05, 0.2, 0.5])
    ee_pos_arr = rng.standard_normal((len(t_arr), 3))

    got = _ee_vel_vectorized(t_arr, ee_pos_arr)
    expected = _ee_vel_scalar_reference(t_arr, ee_pos_arr)

    np.testing.assert_allclose(got, expected, rtol=0, atol=0.0)
    np.testing.assert_array_equal(got[0], np.zeros(3))


def test_ee_velocity_vectorized_single_frame():
    """A single sample yields all-zero velocities (no crash)."""
    t_arr = np.array([0.0])
    ee_pos_arr = np.array([[1.0, 2.0, 3.0]])
    got = _ee_vel_vectorized(t_arr, ee_pos_arr)
    np.testing.assert_array_equal(got, np.zeros((1, 3)))
