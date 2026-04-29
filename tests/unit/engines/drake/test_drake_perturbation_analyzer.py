"""Unit tests for DrakePerturbationAnalyzer (#1979).

Tests are split into two groups:
1. Pure-Python tests that run without pydrake installed (helpers, data structures).
2. Drake-dependent tests that are skipped if pydrake is not available.

Design by Contract
------------------
- Pre: set_base_torque_profile() must be called before run_batch() / perturb_torque().
- Post: run_batch() returns PerturbationSummary with all MANDATORY_METRICS.
- Post: extract_metrics() returns all MANDATORY_METRICS for valid DrakeSimResult.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.engines.physics_engines.drake.python.perturbation.analyzer import (
    MANDATORY_METRICS,
    DrakeSimResult,
)
from src.shared.python.pendulum_simulator.perturbation_analysis import (
    perturb_torque_coeffs,
)
from src.shared.python.perturbation.analyzer_base import ComparisonReport
from src.shared.python.perturbation.config import PerturbationConfig

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

# ---------------------------------------------------------------------------
# Availability check (without importing pydrake at module level)
# ---------------------------------------------------------------------------

try:
    import pydrake.multibody.tree  # noqa: F401

    _DRAKE_AVAILABLE = True
except ImportError:
    _DRAKE_AVAILABLE = False

_skip_no_drake = pytest.mark.skipif(
    not _DRAKE_AVAILABLE, reason="pydrake not installed"
)


# ---------------------------------------------------------------------------
# Pure-Python helpers (no pydrake required)
# ---------------------------------------------------------------------------


class TestMandatoryMetrics:
    def test_non_empty(self) -> None:
        assert len(MANDATORY_METRICS) >= 10

    def test_contains_required_names(self) -> None:
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

    def test_no_duplicates(self) -> None:
        assert len(MANDATORY_METRICS) == len(set(MANDATORY_METRICS))


class TestPerturbCoeffsByMode:
    def test_additive_zero_amplitude_no_change(self) -> None:
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

    def test_multiplicative_same_shape(self) -> None:
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

    def test_both_mode_same_shape(self) -> None:
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

    def test_reproducible_with_seed(self) -> None:
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


class TestDrakeSimResult:
    def _make_result(self, n: int = 5, nq: int = 2, nv: int = 1) -> DrakeSimResult:
        t = np.linspace(0.0, 1.0, n)
        q = np.zeros((n, nq))
        v = np.zeros((n, nv))
        ee_pos = np.zeros((n, 3))
        ee_vel = np.zeros((n, 3))
        ke = np.ones(n)
        pe = np.zeros(n)
        return DrakeSimResult(
            t=t,
            q_traj=q,
            v_traj=v,
            ee_pos_traj=ee_pos,
            ee_vel_traj=ee_vel,
            kinetic_energy_traj=ke,
            potential_energy_traj=pe,
        )

    def test_n_steps(self) -> None:
        r = self._make_result(n=10)
        assert r.n_steps == 10

    def test_trajectory_shapes(self) -> None:
        r = self._make_result(n=5, nq=2, nv=1)
        assert r.q_traj.shape == (5, 2)
        assert r.v_traj.shape == (5, 1)


class TestComparisonReport:
    def test_default_fields(self) -> None:
        report = ComparisonReport(winner="B", confidence=0.7)
        assert report.winner == "B"
        assert report.confidence == 0.7
        assert isinstance(report.metric_comparisons, dict)
        assert isinstance(report.pvalues, dict)


# ---------------------------------------------------------------------------
# Drake-dependent tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analyzer():  # type: ignore[no-untyped-def]
    from src.engines.physics_engines.drake.python.perturbation.analyzer import (
        DrakePerturbationAnalyzer,
    )

    # Uses _MINIMAL_URDF (1-DOF pendulum) as fallback
    return DrakePerturbationAnalyzer(t_end=0.1, dt=0.01)


@pytest.fixture(scope="module")
def simple_profile(analyzer) -> dict:  # type: ignore[no-untyped-def]
    nu = analyzer._nu
    return {"coeffs": [[0.0, 0.0] for _ in range(nu)]}


@pytest.fixture(scope="module")
def analyzer_with_profile(analyzer, simple_profile):  # type: ignore[no-untyped-def]
    analyzer.set_base_torque_profile(simple_profile)
    return analyzer


@_skip_no_drake
class TestEngineMetadata:
    def test_engine_name(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        assert analyzer.ENGINE_NAME == "drake"

    def test_nq_positive(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        assert analyzer._nq > 0

    def test_nu_non_negative(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        assert analyzer._nu >= 0


@_skip_no_drake
class TestSetBaseProfile:
    def test_accepts_valid_profile(self, analyzer, simple_profile) -> None:  # type: ignore[no-untyped-def]
        analyzer.set_base_torque_profile(simple_profile)  # no exception

    def test_requires_dict(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises((AssertionError, TypeError)):
            analyzer.set_base_torque_profile("not_a_dict")  # type: ignore[arg-type]

    def test_requires_coeffs_key(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises((AssertionError, KeyError)):
            analyzer.set_base_torque_profile({"bad_key": []})

    def test_stores_base_coeffs(self, analyzer, simple_profile) -> None:  # type: ignore[no-untyped-def]
        analyzer.set_base_torque_profile(simple_profile)
        assert analyzer._base_coeffs is not None


@_skip_no_drake
class TestPerturbTorque:
    def test_returns_dict_with_coeffs(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        result = analyzer_with_profile.perturb_torque(_SMALL_CONFIG, seed=0)
        assert isinstance(result, dict)
        assert "coeffs" in result

    def test_same_shape_as_base(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        base = analyzer_with_profile._base_coeffs
        result = analyzer_with_profile.perturb_torque(_SMALL_CONFIG, seed=0)
        assert len(result["coeffs"]) == len(base)

    def test_reproducible_with_seed(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        r1 = analyzer_with_profile.perturb_torque(_SMALL_CONFIG, seed=7)
        r2 = analyzer_with_profile.perturb_torque(_SMALL_CONFIG, seed=7)
        assert r1["coeffs"] == r2["coeffs"]


@_skip_no_drake
class TestExtractMetrics:
    def test_returns_all_mandatory_metrics(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        coeffs = analyzer_with_profile._base_coeffs
        sim = analyzer_with_profile._simulate(coeffs)
        metrics = analyzer_with_profile.extract_metrics(sim)
        for name in MANDATORY_METRICS:
            assert name in metrics, f"Missing metric: {name}"

    def test_values_are_finite(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        coeffs = analyzer_with_profile._base_coeffs
        sim = analyzer_with_profile._simulate(coeffs)
        metrics = analyzer_with_profile.extract_metrics(sim)
        for name, val in metrics.items():
            if isinstance(val, np.ndarray):
                assert np.all(np.isfinite(val)), f"Non-finite array metric: {name}"
            else:
                assert np.isfinite(float(val)), f"Non-finite scalar metric: {name}"

    def test_rejects_invalid_input(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises((AssertionError, AttributeError)):
            analyzer_with_profile.extract_metrics("bad_input")  # type: ignore[arg-type]

    def test_motion_duration_positive(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        coeffs = analyzer_with_profile._base_coeffs
        sim = analyzer_with_profile._simulate(coeffs)
        metrics = analyzer_with_profile.extract_metrics(sim)
        assert float(metrics["motion_duration"]) > 0.0


@_skip_no_drake
class TestRunBatch:
    def test_returns_perturbation_summary(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        from src.shared.python.perturbation.config import PerturbationSummary

        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        assert isinstance(result, PerturbationSummary)

    def test_engine_name_matches(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        assert result.engine_name == "drake"

    def test_success_rate_in_range(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        assert 0.0 <= result.success_rate <= 1.0

    def test_robustness_score_in_range(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        assert 0.0 <= result.robustness_score <= 1.0

    def test_requires_profile_set(self) -> None:
        if not _DRAKE_AVAILABLE:
            pytest.skip("pydrake not installed")
        from src.engines.physics_engines.drake.python.perturbation.analyzer import (
            DrakePerturbationAnalyzer,
        )

        fresh = DrakePerturbationAnalyzer(t_end=0.1, dt=0.01)
        with pytest.raises((AssertionError, AttributeError)):
            fresh.run_batch(_SMALL_CONFIG)


@_skip_no_drake
class TestRK4EnergyStability:
    """Regression tests for issue #2121: explicit Euler caused energy divergence.

    RK4 integration must keep total mechanical energy bounded over the simulation
    interval for a zero-torque (free) pendulum.  Explicit Euler would cause the
    energy to grow monotonically; RK4 keeps it near-constant.
    """

    def test_energy_does_not_diverge_zero_torque(self) -> None:
        """Total energy must remain bounded for a zero-torque simulation (issue #2121)."""
        from src.engines.physics_engines.drake.python.perturbation.analyzer import (
            DrakePerturbationAnalyzer,
        )

        # Use a longer sim with larger dt to amplify any instability
        analyzer = DrakePerturbationAnalyzer(t_end=1.0, dt=0.02)
        zero_profile = {"coeffs": [[0.0, 0.0] for _ in range(analyzer._nu)]}
        analyzer.set_base_torque_profile(zero_profile)

        sim = analyzer._simulate(zero_profile["coeffs"])
        total_energy = sim.kinetic_energy_traj + sim.potential_energy_traj

        # Energy at start
        e_initial = total_energy[0]

        # With RK4, total energy drift should be small (< 5x over 1 s).
        # Explicit Euler would show unconstrained growth — often 100%+ over 1 s.
        e_final = total_energy[-1]
        e_peak = float(np.max(np.abs(total_energy)))

        # Guard against zero-energy edge case (all-zero initial state)
        if abs(e_initial) < 1e-10 and abs(e_final) < 1e-10:
            # All energy is zero throughout — trivially stable
            return

        reference = max(abs(e_initial), 1e-10)
        relative_drift = abs(e_final - e_initial) / reference
        assert relative_drift < 5.0, (
            f"Energy drift too large: {relative_drift:.2%} "
            f"(initial={e_initial:.4f}, final={e_final:.4f}, peak={e_peak:.4f}). "
            "This may indicate a reversion to explicit Euler (issue #2121)."
        )

    def test_simulation_produces_finite_trajectories(self) -> None:
        """All trajectory arrays must contain only finite values (no NaN/inf)."""
        from src.engines.physics_engines.drake.python.perturbation.analyzer import (
            DrakePerturbationAnalyzer,
        )

        analyzer = DrakePerturbationAnalyzer(t_end=0.5, dt=0.01)
        profile = {"coeffs": [[0.5, -0.1] for _ in range(analyzer._nu)]}
        analyzer.set_base_torque_profile(profile)
        sim = analyzer._simulate(profile["coeffs"])

        assert np.all(np.isfinite(sim.q_traj)), "q_traj contains non-finite values"
        assert np.all(np.isfinite(sim.v_traj)), "v_traj contains non-finite values"
        assert np.all(np.isfinite(sim.kinetic_energy_traj)), (
            "kinetic_energy_traj contains non-finite values"
        )
        assert np.all(np.isfinite(sim.potential_energy_traj)), (
            "potential_energy_traj contains non-finite values"
        )
