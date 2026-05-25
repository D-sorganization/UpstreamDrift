"""Unit tests for PinocchioPerturbationAnalyzer (#1978).

Tests are split into two groups:
1. Pure-Python tests that run without pinocchio installed (helpers, data structures).
2. Pinocchio-dependent tests that are skipped if pinocchio is not available.

Design by Contract
------------------
- Pre: set_base_torque_profile() must be called before run_batch() / perturb_torque().
- Post: run_batch() returns PerturbationSummary with all MANDATORY_METRICS.
- Post: extract_metrics() returns all MANDATORY_METRICS for valid PinocchioSimResult.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.engines.physics_engines.pinocchio.python.perturbation.analyzer import (
    MANDATORY_METRICS,
    PinocchioSimResult,
)
from src.shared.python.pendulum_simulator.perturbation_analysis import (
    perturb_torque_coeffs,
)
from src.shared.python.perturbation.analyzer_base import ComparisonReport
from src.shared.python.perturbation.config import PerturbationConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GOLFER_URDF = (
    Path(__file__).parents[5]
    / "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf"
)

_SMALL_CONFIG = PerturbationConfig(
    n_trials=3,
    noise_type="white",
    noise_amplitude=0.05,
    perturb_mode="additive",
    seed=42,
)


# ---------------------------------------------------------------------------
# Pure-Python helpers (no pinocchio required)
# ---------------------------------------------------------------------------


class TestMandatoryMetrics:
    def test_pinocchio_perturbation_analyzer_non_empty(self) -> None:
        assert len(MANDATORY_METRICS) >= 10

    def test_pinocchio_perturbation_analyzer_contains_required_names(self) -> None:
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

    def test_pinocchio_perturbation_analyzer_no_duplicates(self) -> None:
        assert len(MANDATORY_METRICS) == len(set(MANDATORY_METRICS))


class TestPerturbCoeffsByMode:
    def test_pinocchio_perturbation_analyzer_additive_zero_amplitude_no_change(
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

    def test_multiplicative_returns_same_shape(self) -> None:
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

    def test_both_mode_returns_same_shape(self) -> None:
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

    def test_additive_reproducible_with_same_seed(self) -> None:
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


class TestPinocchioSimResult:
    def _make_sim_result(
        self, n: int = 5, nq: int = 7, nv: int = 6
    ) -> PinocchioSimResult:
        t = np.linspace(0.0, 1.0, n)
        q = np.zeros((n, nq))
        v = np.zeros((n, nv))
        ee_pos = np.zeros((n, 3))
        ee_vel = np.zeros((n, 3))
        ke = np.ones(n)
        pe = np.zeros(n)
        return PinocchioSimResult(
            t=t,
            q_traj=q,
            v_traj=v,
            ee_pos_traj=ee_pos,
            ee_vel_traj=ee_vel,
            kinetic_energy_traj=ke,
            potential_energy_traj=pe,
        )

    def test_pinocchio_perturbation_analyzer_n_steps(self) -> None:
        r = self._make_sim_result(n=10)
        assert r.n_steps == 10

    def test_trajectory_shape(self) -> None:
        r = self._make_sim_result(n=5, nq=7, nv=6)
        assert r.q_traj.shape == (5, 7)
        assert r.v_traj.shape == (5, 6)


class TestComparisonReport:
    def test_pinocchio_perturbation_analyzer_default_fields(self) -> None:
        report = ComparisonReport(winner="A", confidence=0.8)
        assert report.winner == "A"
        assert report.confidence == 0.8
        assert isinstance(report.metric_comparisons, dict)
        assert isinstance(report.pvalues, dict)


# ---------------------------------------------------------------------------
# Pinocchio-dependent tests
# ---------------------------------------------------------------------------

try:
    import pinocchio as _pin  # noqa: F401

    _PINOCCHIO_AVAILABLE = True
except ImportError:
    _PINOCCHIO_AVAILABLE = False


@pytest.fixture(scope="module")
def urdf_path() -> Path:
    if not _GOLFER_URDF.exists():
        pytest.skip(f"Golfer URDF not found: {_GOLFER_URDF}")
    return _GOLFER_URDF


@pytest.fixture(scope="module")
def analyzer(urdf_path: Path):  # type: ignore[no-untyped-def]
    from src.engines.physics_engines.pinocchio.python.perturbation.analyzer import (
        PinocchioPerturbationAnalyzer,
    )

    return PinocchioPerturbationAnalyzer(urdf_path=urdf_path, t_end=0.1, dt=0.01)


@pytest.fixture(scope="module")
def simple_profile(analyzer) -> dict:  # type: ignore[no-untyped-def]
    nv = analyzer._nv
    return {"coeffs": [[0.0, 0.0] for _ in range(nv)]}


@pytest.fixture(scope="module")
def analyzer_with_profile(analyzer, simple_profile):  # type: ignore[no-untyped-def]
    analyzer.set_base_torque_profile(simple_profile)
    return analyzer
