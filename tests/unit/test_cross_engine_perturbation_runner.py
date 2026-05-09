"""Unit tests for CrossEnginePerturbationRunner (#1983).

Tests cover:
- Pure-Python data structures (no engine required)
- Utility functions: rank_engines(), compute_consistency(), format_report()
- Runner protocol with a mock analyzer stub (no physics engine needed)
- Integration smoke test with real MuJoCo engine (skips if not installed)

Design by Contract
------------------
- Pre: run_all() requires profile to be set first.
- Post: run_all() returns CrossEngineReport with ranking sorted by RS descending.
- Post: rank_engines() assigns contiguous ranks starting at 1.
- Post: compute_consistency() returns one entry per shared metric with >= 2 engines.
"""

from __future__ import annotations

import pytest
from src.shared.python.perturbation.config import (
    PerturbationConfig,
    PerturbationSummary,
)
from src.shared.python.perturbation.cross_engine_runner import (
    SUPPORTED_ENGINES,
    ConsistencyMetrics,
    CrossEnginePerturbationRunner,
    CrossEngineReport,
    EngineRankEntry,
    compute_consistency,
    format_report,
    rank_engines,
)
from src.shared.python.perturbation.statistics import MetricStatistics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_summary(
    engine: str,
    rs: float,
    success_rate: float = 1.0,
    metrics: dict | None = None,
) -> PerturbationSummary:
    """Build a minimal PerturbationSummary for testing."""
    cfg = PerturbationConfig(n_trials=3, noise_amplitude=0.05)
    if metrics is None:
        metrics = {}
    return PerturbationSummary(
        engine_name=engine,
        config=cfg,
        robustness_score=rs,
        metrics=metrics,
        success_rate=success_rate,
        execution_time_sec=0.1,
    )


def _make_metric_stats(mean: float, std: float = 0.01) -> MetricStatistics:
    cv = std / abs(mean) if abs(mean) > 1e-12 else 0.0
    return MetricStatistics(
        mean=mean,
        std=std,
        cv=cv,
        min_val=mean - std,
        max_val=mean + std,
        median=mean,
        iqr=std,
        p5=mean - 2 * std,
        p95=mean + 2 * std,
    )


_ZERO_PROFILE: dict = {"coeffs": [[0.0, 0.0], [0.0, 0.0]]}

_SMALL_CONFIG = PerturbationConfig(
    n_trials=3, noise_amplitude=0.05, noise_type="white", seed=7
)

# ---------------------------------------------------------------------------
# SUPPORTED_ENGINES constant
# ---------------------------------------------------------------------------


class TestSupportedEngines:
    def test_cross_engine_perturbation_runner_non_empty(self) -> None:
        assert len(SUPPORTED_ENGINES) >= 6

    def test_contains_all_six(self) -> None:
        expected = {"pendulum", "pinocchio", "drake", "mujoco", "opensim", "myosuite"}
        assert expected.issubset(set(SUPPORTED_ENGINES))

    def test_cross_engine_perturbation_runner_no_duplicates(self) -> None:
        assert len(SUPPORTED_ENGINES) == len(set(SUPPORTED_ENGINES))


# ---------------------------------------------------------------------------
# EngineRankEntry dataclass
# ---------------------------------------------------------------------------


class TestEngineRankEntry:
    def test_cross_engine_perturbation_runner_defaults(self) -> None:
        entry = EngineRankEntry(
            engine_name="mujoco",
            robustness_score=0.85,
            success_rate=1.0,
            execution_time_sec=2.3,
        )
        assert entry.rank == 0
        assert entry.engine_name == "mujoco"

    def test_rank_can_be_set(self) -> None:
        entry = EngineRankEntry(
            engine_name="drake",
            robustness_score=0.70,
            success_rate=0.9,
            execution_time_sec=1.0,
            rank=2,
        )
        assert entry.rank == 2


# ---------------------------------------------------------------------------
# ConsistencyMetrics dataclass
# ---------------------------------------------------------------------------


class TestConsistencyMetrics:
    def test_fields(self) -> None:
        cm = ConsistencyMetrics(
            metric_name="trajectory_rmse",
            engine_means={"mujoco": 0.1, "drake": 0.12},
            spread=0.02,
            coefficient_of_variation=0.1,
            is_consistent=True,
        )
        assert cm.metric_name == "trajectory_rmse"
        assert cm.is_consistent is True


# ---------------------------------------------------------------------------
# CrossEngineReport dataclass
# ---------------------------------------------------------------------------


class TestCrossEngineReport:
    def _make_report(self) -> CrossEngineReport:
        cfg = PerturbationConfig(n_trials=3)
        summaries = {"mujoco": _make_summary("mujoco", 0.9)}
        ranking = rank_engines(summaries)
        return CrossEngineReport(
            config=cfg,
            summaries=summaries,
            ranking=ranking,
            consistency={},
        )

    def test_cross_engine_perturbation_runner_to_dict_keys(self) -> None:
        report = self._make_report()
        d = report.to_dict()
        assert "config" in d
        assert "summaries" in d
        assert "ranking" in d
        assert "consistency" in d
        assert "failed_engines" in d
        assert "total_time_sec" in d

    def test_failed_engines_default_empty(self) -> None:
        report = self._make_report()
        assert report.failed_engines == []


# ---------------------------------------------------------------------------
# rank_engines()
# ---------------------------------------------------------------------------


class TestRankEngines:
    def test_sorted_descending(self) -> None:
        summaries = {
            "a": _make_summary("a", 0.5),
            "b": _make_summary("b", 0.9),
            "c": _make_summary("c", 0.7),
        }
        ranking = rank_engines(summaries)
        scores = [e.robustness_score for e in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_start_at_one(self) -> None:
        summaries = {"a": _make_summary("a", 0.8), "b": _make_summary("b", 0.6)}
        ranking = rank_engines(summaries)
        assert ranking[0].rank == 1
        assert ranking[1].rank == 2

    def test_single_engine(self) -> None:
        summaries = {"mujoco": _make_summary("mujoco", 0.85)}
        ranking = rank_engines(summaries)
        assert len(ranking) == 1
        assert ranking[0].rank == 1
        assert ranking[0].engine_name == "mujoco"

    def test_empty_summaries(self) -> None:
        ranking = rank_engines({})
        assert ranking == []

    def test_ties_maintain_order_stability(self) -> None:
        summaries = {
            "a": _make_summary("a", 0.7),
            "b": _make_summary("b", 0.7),
        }
        ranking = rank_engines(summaries)
        assert len(ranking) == 2
        assert {e.robustness_score for e in ranking} == {0.7}


# ---------------------------------------------------------------------------
# compute_consistency()
# ---------------------------------------------------------------------------


class TestComputeConsistency:
    def _make_summaries_with_metrics(self) -> dict[str, PerturbationSummary]:
        return {
            "mujoco": _make_summary(
                "mujoco",
                0.9,
                metrics={
                    "trajectory_rmse": _make_metric_stats(0.10, 0.01),
                    "peak_end_effector_speed": _make_metric_stats(2.0, 0.05),
                },
            ),
            "drake": _make_summary(
                "drake",
                0.85,
                metrics={
                    "trajectory_rmse": _make_metric_stats(0.11, 0.01),
                    "peak_end_effector_speed": _make_metric_stats(2.1, 0.05),
                },
            ),
        }

    def test_returns_shared_metrics(self) -> None:
        summaries = self._make_summaries_with_metrics()
        consistency = compute_consistency(summaries)
        assert "trajectory_rmse" in consistency
        assert "peak_end_effector_speed" in consistency

    def test_engine_means_populated(self) -> None:
        summaries = self._make_summaries_with_metrics()
        c = compute_consistency(summaries)["trajectory_rmse"]
        assert "mujoco" in c.engine_means
        assert "drake" in c.engine_means

    def test_spread_non_negative(self) -> None:
        summaries = self._make_summaries_with_metrics()
        for c in compute_consistency(summaries).values():
            assert c.spread >= 0.0

    def test_consistent_when_cv_low(self) -> None:
        # Two engines with nearly identical means → consistent
        summaries = {
            "a": _make_summary(
                "a", 0.8, metrics={"rmse": _make_metric_stats(1.0, 0.001)}
            ),
            "b": _make_summary(
                "b", 0.8, metrics={"rmse": _make_metric_stats(1.001, 0.001)}
            ),
        }
        c = compute_consistency(summaries, threshold=0.2)["rmse"]
        assert c.is_consistent is True

    def test_inconsistent_when_cv_high(self) -> None:
        # Two engines with very different means → inconsistent
        summaries = {
            "a": _make_summary(
                "a", 0.8, metrics={"rmse": _make_metric_stats(1.0, 0.01)}
            ),
            "b": _make_summary(
                "b", 0.8, metrics={"rmse": _make_metric_stats(10.0, 0.01)}
            ),
        }
        c = compute_consistency(summaries, threshold=0.2)["rmse"]
        assert c.is_consistent is False

    def test_single_engine_excluded(self) -> None:
        # Only one engine → cannot compute consistency
        summaries = {
            "mujoco": _make_summary(
                "mujoco", 0.9, metrics={"rmse": _make_metric_stats(0.1)}
            )
        }
        c = compute_consistency(summaries)
        assert "rmse" not in c

    def test_empty_summaries(self) -> None:
        assert compute_consistency({}) == {}


# ---------------------------------------------------------------------------
# format_report()
# ---------------------------------------------------------------------------


class TestFormatReport:
    def _make_report(self) -> CrossEngineReport:
        cfg = PerturbationConfig(n_trials=3)
        summaries = {
            "mujoco": _make_summary("mujoco", 0.9),
            "drake": _make_summary("drake", 0.8),
        }
        ranking = rank_engines(summaries)
        consistency = compute_consistency(summaries)
        return CrossEngineReport(
            config=cfg,
            summaries=summaries,
            ranking=ranking,
            consistency=consistency,
        )

    def test_cross_engine_perturbation_runner_returns_string(self) -> None:
        report = self._make_report()
        text = format_report(report)
        assert isinstance(text, str)

    def test_contains_engine_names(self) -> None:
        report = self._make_report()
        text = format_report(report)
        assert "mujoco" in text
        assert "drake" in text

    def test_contains_header(self) -> None:
        report = self._make_report()
        text = format_report(report)
        assert "Cross-Engine Perturbation Comparison Report" in text

    def test_failed_engines_section(self) -> None:
        cfg = PerturbationConfig(n_trials=3)
        report = CrossEngineReport(
            config=cfg,
            summaries={},
            ranking=[],
            consistency={},
            failed_engines=["pinocchio", "opensim"],
        )
        text = format_report(report)
        assert "pinocchio" in text
        assert "opensim" in text


# ---------------------------------------------------------------------------
# CrossEnginePerturbationRunner — unit tests with mock analyzers
# ---------------------------------------------------------------------------


class _MockAnalyzer:
    """Lightweight mock PerturbationAnalyzer for unit testing the runner."""

    ENGINE_NAME: str = "mock"

    def __init__(self, rs: float = 0.8) -> None:
        self._rs = rs
        self._profile: dict | None = None

    def set_base_torque_profile(self, profile: object) -> None:
        assert isinstance(profile, dict)
        assert "coeffs" in profile
        self._profile = profile  # type: ignore[assignment]

    def perturb_torque(self, config: PerturbationConfig, seed: int) -> dict:
        assert self._profile is not None
        return {"coeffs": self._profile["coeffs"]}  # type: ignore[index]

    def run_batch(self, config: PerturbationConfig) -> PerturbationSummary:
        assert self._profile is not None
        return PerturbationSummary(
            engine_name=self.ENGINE_NAME,
            config=config,
            robustness_score=self._rs,
            metrics={},
            success_rate=1.0,
            execution_time_sec=0.01,
        )

    def extract_metrics(self, sim_result: object) -> dict[str, float]:
        return {}


class TestCrossEnginePerturbationRunner:
    def test_requires_profile_before_run(self) -> None:
        runner = CrossEnginePerturbationRunner(engines=["mujoco"])
        with pytest.raises(AssertionError):
            runner.run_all(_SMALL_CONFIG)

    def test_set_profile_validates_dict(self) -> None:
        runner = CrossEnginePerturbationRunner(engines=["mujoco"])
        with pytest.raises(AssertionError):
            runner.set_profile("not_a_dict")  # type: ignore[arg-type]

    def test_set_profile_requires_coeffs(self) -> None:
        runner = CrossEnginePerturbationRunner(engines=["mujoco"])
        with pytest.raises(AssertionError):
            runner.set_profile({"bad_key": []})

    def test_rejects_unknown_engine(self) -> None:
        with pytest.raises(AssertionError):
            CrossEnginePerturbationRunner(engines=["nonexistent_engine"])

    def test_run_all_with_mocked_analyzers(self) -> None:
        """Test runner orchestration using injected mock analyzers."""
        runner = CrossEnginePerturbationRunner(engines=["mujoco", "drake"])
        runner.set_profile(_ZERO_PROFILE)

        # Inject mocks directly bypassing _load_analyzer
        runner._analyzers["mujoco"] = _MockAnalyzer(rs=0.9)
        runner._analyzers["drake"] = _MockAnalyzer(rs=0.7)

        report = runner.run_all(_SMALL_CONFIG)

        assert isinstance(report, CrossEngineReport)
        assert "mujoco" in report.summaries
        assert "drake" in report.summaries
        assert len(report.ranking) == 2
        assert report.ranking[0].robustness_score >= report.ranking[1].robustness_score
        assert report.ranking[0].rank == 1

    def test_failed_engine_captured(self) -> None:
        """Engine that raises exception goes into failed_engines."""

        class _BrokenAnalyzer(_MockAnalyzer):
            def run_batch(self, config: PerturbationConfig) -> PerturbationSummary:
                msg = "Simulated engine failure"
                raise RuntimeError(msg)

        runner = CrossEnginePerturbationRunner(engines=["mujoco", "drake"])
        runner.set_profile(_ZERO_PROFILE)
        runner._analyzers["mujoco"] = _MockAnalyzer(rs=0.9)
        runner._analyzers["drake"] = _BrokenAnalyzer()

        report = runner.run_all(_SMALL_CONFIG)

        assert "drake" in report.failed_engines
        assert "mujoco" in report.summaries

    def test_ranking_sorted_descending(self) -> None:
        runner = CrossEnginePerturbationRunner(engines=["mujoco", "drake", "pinocchio"])
        runner.set_profile(_ZERO_PROFILE)
        runner._analyzers["mujoco"] = _MockAnalyzer(rs=0.5)
        runner._analyzers["drake"] = _MockAnalyzer(rs=0.9)
        runner._analyzers["pinocchio"] = _MockAnalyzer(rs=0.7)

        report = runner.run_all(_SMALL_CONFIG)
        scores = [e.robustness_score for e in report.ranking]
        assert scores == sorted(scores, reverse=True)

    def test_total_time_non_negative(self) -> None:
        runner = CrossEnginePerturbationRunner(engines=["mujoco"])
        runner.set_profile(_ZERO_PROFILE)
        runner._analyzers["mujoco"] = _MockAnalyzer(rs=0.8)

        report = runner.run_all(_SMALL_CONFIG)
        assert report.total_time_sec >= 0.0

    def test_run_single(self) -> None:
        runner = CrossEnginePerturbationRunner(engines=["mujoco"])
        runner.set_profile(_ZERO_PROFILE)
        runner._analyzers["mujoco"] = _MockAnalyzer(rs=0.75)

        summary = runner.run_single("mujoco", _SMALL_CONFIG)
        assert isinstance(summary, PerturbationSummary)
        assert summary.robustness_score == pytest.approx(0.75)

    def test_run_single_requires_profile(self) -> None:
        runner = CrossEnginePerturbationRunner(engines=["mujoco"])
        with pytest.raises(AssertionError):
            runner.run_single("mujoco", _SMALL_CONFIG)


# ---------------------------------------------------------------------------
# Integration smoke test — real MuJoCo engine (skips if not installed)
# ---------------------------------------------------------------------------

_MUJOCO_ANALYZER_AVAILABLE: bool = False
try:
    import mujoco as _mujoco_check  # noqa: F401  # type: ignore[import-untyped]

    _mujoco_analyzer_mod = __import__(
        "src.engines.physics_engines.mujoco.python.perturbation.analyzer",
        fromlist=[""],
    )
    _MUJOCO_ANALYZER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass

_skip_no_mujoco = pytest.mark.skipif(
    not _MUJOCO_ANALYZER_AVAILABLE,
    reason="mujoco or mujoco perturbation analyzer not installed",
)
