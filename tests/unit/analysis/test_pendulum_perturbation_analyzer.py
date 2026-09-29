"""Unit tests for PendulumPerturbationAnalyzer (#1977).

Tests the reference implementation of the PerturbationAnalyzer protocol
for the driven double-pendulum model.

Design by Contract
------------------
- Pre: set_base_torque_profile() called before run_batch() / perturb_torque().
- Post: run_batch() returns PerturbationSummary with all MANDATORY_METRICS.
- Post: extract_metrics() returns all MANDATORY_METRICS for valid sim results.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.pendulum_perturbation_analyzer import (
    MANDATORY_METRICS,
    ComparisonReport,
    PendulumPerturbationAnalyzer,
    _perturb_coeffs_by_mode,
)
from src.shared.python.pendulum_simulator.physics import PendulumParams
from src.shared.python.perturbation.config import (
    PerturbationConfig,
    PerturbationSummary,
)
from src.shared.python.perturbation.statistics import MetricStatistics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SIMPLE_COEFFS: list[list[float]] = [[0.0, 0.0], [0.0, 0.0]]

_SIMPLE_PROFILE: dict = {"coeffs": _SIMPLE_COEFFS}

_SMALL_CONFIG = PerturbationConfig(
    n_trials=3,
    noise_type="white",
    noise_amplitude=0.05,
    perturb_mode="additive",
    seed=42,
)

_DEFAULT_PARAMS = PendulumParams(m1=5.0, m2=0.30, L1=0.65, L2=1.10)


@pytest.fixture()
def analyzer() -> PendulumPerturbationAnalyzer:
    return PendulumPerturbationAnalyzer(_DEFAULT_PARAMS, t_end=0.5, dt=0.01)


@pytest.fixture()
def analyzer_with_profile(
    analyzer: PendulumPerturbationAnalyzer,
) -> PendulumPerturbationAnalyzer:
    analyzer.set_base_torque_profile(_SIMPLE_PROFILE)
    return analyzer


# ---------------------------------------------------------------------------
# MANDATORY_METRICS
# ---------------------------------------------------------------------------


def test_mandatory_metrics_non_empty() -> None:
    assert len(MANDATORY_METRICS) >= 10


def test_mandatory_metrics_contains_required_names() -> None:
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


def test_mandatory_metrics_no_duplicates() -> None:
    assert len(MANDATORY_METRICS) == len(set(MANDATORY_METRICS))


# ---------------------------------------------------------------------------
# Engine name
# ---------------------------------------------------------------------------


def test_pendulum_perturbation_analyzer_engine_name(
    analyzer: PendulumPerturbationAnalyzer,
) -> None:
    assert analyzer.ENGINE_NAME == "pendulum_double"


# ---------------------------------------------------------------------------
# set_base_torque_profile
# ---------------------------------------------------------------------------


def test_set_base_torque_profile_accepts_valid_dict(
    analyzer: PendulumPerturbationAnalyzer,
) -> None:
    analyzer.set_base_torque_profile(_SIMPLE_PROFILE)
    # no assertion needed — absence of exception is the contract


def test_set_base_torque_profile_requires_coeffs_key(
    analyzer: PendulumPerturbationAnalyzer,
) -> None:
    with pytest.raises((AssertionError, KeyError, TypeError)):
        analyzer.set_base_torque_profile({"wrong_key": []})


def test_set_base_torque_profile_stores_coeffs(
    analyzer: PendulumPerturbationAnalyzer,
) -> None:
    analyzer.set_base_torque_profile(_SIMPLE_PROFILE)
    assert analyzer._base_coeffs == _SIMPLE_COEFFS


# ---------------------------------------------------------------------------
# perturb_torque
# ---------------------------------------------------------------------------


def test_perturb_torque_returns_dict(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    result = analyzer_with_profile.perturb_torque(_SMALL_CONFIG, seed=0)
    assert isinstance(result, dict)


def test_perturb_torque_has_coeffs_key(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    result = analyzer_with_profile.perturb_torque(_SMALL_CONFIG, seed=0)
    assert "coeffs" in result


def test_perturb_torque_same_shape(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    result = analyzer_with_profile.perturb_torque(_SMALL_CONFIG, seed=0)
    coeffs = result["coeffs"]
    assert len(coeffs) == len(_SIMPLE_COEFFS)
    for orig, pert in zip(_SIMPLE_COEFFS, coeffs, strict=True):
        assert len(pert) == len(orig)


def test_perturb_torque_reproducible_with_seed(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    r1 = analyzer_with_profile.perturb_torque(_SMALL_CONFIG, seed=7)
    r2 = analyzer_with_profile.perturb_torque(_SMALL_CONFIG, seed=7)
    assert r1["coeffs"] == r2["coeffs"]


def test_perturb_torque_different_seeds_differ(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    # Use non-zero coefficients so noise makes a measurable difference
    analyzer_with_profile.set_base_torque_profile({"coeffs": [[1.0, 2.0], [3.0, 4.0]]})
    config = PerturbationConfig(
        n_trials=3, noise_amplitude=0.5, noise_type="white", seed=1
    )
    r3 = analyzer_with_profile.perturb_torque(config, seed=0)
    r4 = analyzer_with_profile.perturb_torque(config, seed=999)
    flat3 = [c for joint in r3["coeffs"] for c in joint]
    flat4 = [c for joint in r4["coeffs"] for c in joint]
    assert flat3 != flat4


# ---------------------------------------------------------------------------
# _perturb_coeffs_by_mode helper
# ---------------------------------------------------------------------------


def test_perturb_coeffs_additive_zero_amplitude() -> None:
    coeffs = [[1.0, 2.0], [3.0, 4.0]]
    config = PerturbationConfig(
        n_trials=1, noise_amplitude=0.0, noise_type="white", perturb_mode="additive"
    )
    result = _perturb_coeffs_by_mode(coeffs, config, seed=0)
    # zero amplitude → no change
    flat_orig = [c for j in coeffs for c in j]
    flat_res = [c for j in result for c in j]
    assert all(abs(a - b) < 1e-12 for a, b in zip(flat_orig, flat_res, strict=True))


def test_perturb_coeffs_multiplicative_returns_same_shape() -> None:
    coeffs = [[1.0, 2.0], [3.0, 4.0]]
    config = PerturbationConfig(
        n_trials=1,
        noise_amplitude=0.1,
        noise_type="white",
        perturb_mode="multiplicative",
    )
    result = _perturb_coeffs_by_mode(coeffs, config, seed=0)
    assert len(result) == len(coeffs)
    for orig, res in zip(coeffs, result, strict=True):
        assert len(res) == len(orig)


def test_perturb_coeffs_both_mode() -> None:
    coeffs = [[1.0, 2.0], [3.0, 4.0]]
    config = PerturbationConfig(
        n_trials=1, noise_amplitude=0.1, noise_type="white", perturb_mode="both"
    )
    result = _perturb_coeffs_by_mode(coeffs, config, seed=0)
    assert len(result) == len(coeffs)


# ---------------------------------------------------------------------------
# extract_metrics
# ---------------------------------------------------------------------------


def test_extract_metrics_returns_all_mandatory(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    sim = analyzer_with_profile._simulate(_SIMPLE_COEFFS)
    metrics = analyzer_with_profile.extract_metrics(sim)
    for name in MANDATORY_METRICS:
        assert name in metrics, f"Missing metric: {name}"


def test_extract_metrics_values_are_finite(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    sim = analyzer_with_profile._simulate(_SIMPLE_COEFFS)
    metrics = analyzer_with_profile.extract_metrics(sim)
    for name, val in metrics.items():
        if isinstance(val, np.ndarray):
            assert np.all(np.isfinite(val)), f"Non-finite array metric: {name}"
        else:
            assert np.isfinite(float(val)), f"Non-finite scalar metric: {name}"


def test_extract_metrics_rejects_invalid_input(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    with pytest.raises((AssertionError, AttributeError, TypeError)):
        analyzer_with_profile.extract_metrics("not a sim result")  # type: ignore[arg-type]


def test_extract_metrics_motion_duration_positive(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    sim = analyzer_with_profile._simulate(_SIMPLE_COEFFS)
    metrics = analyzer_with_profile.extract_metrics(sim)
    assert float(metrics["motion_duration"]) > 0.0


def test_extract_metrics_speed_non_negative(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    sim = analyzer_with_profile._simulate(_SIMPLE_COEFFS)
    metrics = analyzer_with_profile.extract_metrics(sim)
    assert float(metrics["end_effector_speed_final"]) >= 0.0
    assert float(metrics["peak_end_effector_speed"]) >= 0.0


def test_extract_metrics_joint_arrays_shape(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    sim = analyzer_with_profile._simulate(_SIMPLE_COEFFS)
    metrics = analyzer_with_profile.extract_metrics(sim)
    assert isinstance(metrics["joint_angles_final"], np.ndarray)
    assert metrics["joint_angles_final"].shape == (2,)
    assert isinstance(metrics["joint_velocities_final"], np.ndarray)
    assert metrics["joint_velocities_final"].shape == (2,)


def test_extract_metrics_position_is_2d(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    sim = analyzer_with_profile._simulate(_SIMPLE_COEFFS)
    metrics = analyzer_with_profile.extract_metrics(sim)
    pos = metrics["end_effector_position_final"]
    assert isinstance(pos, np.ndarray)
    assert pos.shape == (2,)


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------


def test_run_batch_returns_perturbation_summary(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
    assert isinstance(result, PerturbationSummary)


def test_run_batch_engine_name_matches(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
    assert result.engine_name == analyzer_with_profile.ENGINE_NAME


def test_run_batch_contains_all_mandatory_metrics(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
    # Scalar metrics should all appear in result.metrics
    scalar_metrics = [
        m
        for m in MANDATORY_METRICS
        if m
        not in (
            "end_effector_position_final",
            "end_effector_velocity_final",
            "joint_angles_final",
            "joint_velocities_final",
        )
    ]
    for name in scalar_metrics:
        assert name in result.metrics, f"Missing metric in summary: {name}"


def test_run_batch_success_rate_in_range(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
    assert 0.0 <= result.success_rate <= 1.0


def test_run_batch_robustness_score_in_range(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
    assert 0.0 <= result.robustness_score <= 1.0


def test_run_batch_execution_time_positive(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
    assert result.execution_time_sec >= 0.0


def test_run_batch_metric_statistics_type(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
    for name, val in result.metrics.items():
        if isinstance(val, MetricStatistics):
            assert np.isfinite(val.mean), f"Non-finite mean in {name}"
            assert val.std >= 0.0, f"Negative std in {name}"


def test_run_batch_requires_profile_set(
    analyzer: PendulumPerturbationAnalyzer,
) -> None:
    with pytest.raises((AssertionError, AttributeError, TypeError)):
        analyzer.run_batch(_SMALL_CONFIG)


def test_run_batch_zero_amplitude_high_success_rate(
    analyzer_with_profile: PendulumPerturbationAnalyzer,
) -> None:
    config = PerturbationConfig(
        n_trials=5, noise_amplitude=0.0, noise_type="white", seed=0
    )
    result = analyzer_with_profile.run_batch(config)
    assert result.success_rate == 1.0


def test_run_batch_reproducible_with_seed(
    analyzer: PendulumPerturbationAnalyzer,
) -> None:
    profile = {"coeffs": [[1.0, 0.5, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0]]}
    config = PerturbationConfig(
        n_trials=3, noise_amplitude=0.05, noise_type="white", seed=77
    )
    analyzer.set_base_torque_profile(profile)
    r1 = analyzer.run_batch(config)
    analyzer.set_base_torque_profile(profile)
    r2 = analyzer.run_batch(config)
    assert abs(r1.robustness_score - r2.robustness_score) < 1e-10


# ---------------------------------------------------------------------------
# compare_profiles
# ---------------------------------------------------------------------------


def test_compare_profiles_returns_comparison_report(
    analyzer: PendulumPerturbationAnalyzer,
) -> None:
    scipy = pytest.importorskip("scipy")  # noqa: F841
    profile_a = {"coeffs": [[1.0, 0.0], [0.0, 1.0]]}
    profile_b = {"coeffs": [[0.5, 0.5], [0.5, 0.5]]}
    config = PerturbationConfig(n_trials=3, noise_amplitude=0.05, seed=0)
    result = analyzer.compare_profiles(profile_a, profile_b, config, "A", "B")
    assert isinstance(result, ComparisonReport)


def test_compare_profiles_winner_is_a_or_b(
    analyzer: PendulumPerturbationAnalyzer,
) -> None:
    pytest.importorskip("scipy")
    profile_a = {"coeffs": [[1.0, 0.0], [0.0, 1.0]]}
    profile_b = {"coeffs": [[0.5, 0.5], [0.5, 0.5]]}
    config = PerturbationConfig(n_trials=3, noise_amplitude=0.05, seed=0)
    result = analyzer.compare_profiles(profile_a, profile_b, config, "A", "B")
    assert result.winner in ("A", "B")


def test_compare_profiles_confidence_in_range(
    analyzer: PendulumPerturbationAnalyzer,
) -> None:
    pytest.importorskip("scipy")
    profile_a = {"coeffs": [[1.0, 0.0], [0.0, 1.0]]}
    profile_b = {"coeffs": [[0.5, 0.5], [0.5, 0.5]]}
    config = PerturbationConfig(n_trials=3, noise_amplitude=0.05, seed=0)
    result = analyzer.compare_profiles(profile_a, profile_b, config, "A", "B")
    assert 0.0 <= result.confidence <= 1.0


def test_compare_profiles_has_metric_comparisons(
    analyzer: PendulumPerturbationAnalyzer,
) -> None:
    pytest.importorskip("scipy")
    profile_a = {"coeffs": [[1.0, 0.0], [0.0, 1.0]]}
    profile_b = {"coeffs": [[1.0, 0.0], [0.0, 1.0]]}
    config = PerturbationConfig(n_trials=3, noise_amplitude=0.05, seed=0)
    result = analyzer.compare_profiles(profile_a, profile_b, config, "A", "B")
    assert isinstance(result.metric_comparisons, dict)
    assert len(result.metric_comparisons) > 0
