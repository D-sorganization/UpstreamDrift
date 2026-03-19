"""Cross-Engine Perturbation Comparison Framework (#1983).

Runs identical perturbation analyses across all six physics engine backends
(Pendulum, Pinocchio, Drake, MuJoCo, OpenSim, MyoSuite) and produces
unified ranking and consistency reports.

Design by Contract
------------------
- Pre:  engine names must be from ``SUPPORTED_ENGINES``.
- Pre:  all engines must have been given the same logical profile before
        ``run_all()`` is called.
- Post: ``run_all()`` returns a ``CrossEngineReport`` with one
        ``PerturbationSummary`` per engine that succeeded.
- Post: ``rank_engines()`` returns engines sorted by ``robustness_score``
        (descending) — higher is more robust.

DRY
---
Delegates per-engine Monte Carlo batches to each engine's own
``PerturbationAnalyzer.run_batch()``.  All noise logic lives in the shared
``perturbation`` package; this module only orchestrates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.shared.python.perturbation.config import (
    PerturbationConfig,
    PerturbationSummary,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported engine names (must match ENGINE_NAME on each analyzer class)
# ---------------------------------------------------------------------------

SUPPORTED_ENGINES: tuple[str, ...] = (
    "pendulum",
    "pinocchio",
    "drake",
    "mujoco",
    "opensim",
    "myosuite",
)

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class EngineRankEntry:
    """Single entry in the cross-engine robustness ranking.

    Attributes
    ----------
    engine_name : str
    robustness_score : float — higher is more robust
    success_rate : float — fraction of trials that did not error
    execution_time_sec : float
    rank : int — 1 = most robust
    """

    engine_name: str
    robustness_score: float
    success_rate: float
    execution_time_sec: float
    rank: int = 0


@dataclass
class ConsistencyMetrics:
    """Cross-engine consistency statistics for a single scalar metric.

    Attributes
    ----------
    metric_name : str
    engine_means : dict mapping engine_name → mean value
    spread : float — max(means) − min(means) across engines
    coefficient_of_variation : float — std(means) / |mean(means)|
    is_consistent : bool — CV < consistency_threshold
    """

    metric_name: str
    engine_means: dict[str, float]
    spread: float
    coefficient_of_variation: float
    is_consistent: bool


@dataclass
class CrossEngineReport:
    """Full report from a cross-engine perturbation comparison run.

    Attributes
    ----------
    config : PerturbationConfig — shared config used for all engines
    summaries : dict mapping engine_name → PerturbationSummary
    ranking : list of EngineRankEntry, sorted by robustness_score descending
    consistency : dict mapping metric_name → ConsistencyMetrics
    failed_engines : list of engine names that raised exceptions
    total_time_sec : float
    """

    config: PerturbationConfig
    summaries: dict[str, PerturbationSummary]
    ranking: list[EngineRankEntry]
    consistency: dict[str, ConsistencyMetrics]
    failed_engines: list[str] = field(default_factory=list)
    total_time_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "config": {
                "n_trials": self.config.n_trials,
                "noise_type": self.config.noise_type,
                "noise_amplitude": self.config.noise_amplitude,
                "perturb_mode": self.config.perturb_mode,
                "seed": self.config.seed,
            },
            "summaries": {
                name: summary.to_dict() for name, summary in self.summaries.items()
            },
            "ranking": [
                {
                    "rank": e.rank,
                    "engine": e.engine_name,
                    "robustness_score": e.robustness_score,
                    "success_rate": e.success_rate,
                    "execution_time_sec": e.execution_time_sec,
                }
                for e in self.ranking
            ],
            "consistency": {
                name: {
                    "spread": c.spread,
                    "cv": c.coefficient_of_variation,
                    "is_consistent": c.is_consistent,
                    "engine_means": c.engine_means,
                }
                for name, c in self.consistency.items()
            },
            "failed_engines": self.failed_engines,
            "total_time_sec": self.total_time_sec,
        }


# ---------------------------------------------------------------------------
# Engine loader registry
# ---------------------------------------------------------------------------

_ENGINE_LOADER_MAP: dict[str, str] = {
    "pendulum": (
        "src.shared.python.pendulum_simulator.perturbation_analysis"
        "|PendulumPerturbationAnalyzer"
    ),
    "pinocchio": (
        "src.engines.physics_engines.pinocchio.python.perturbation.analyzer"
        "|PinocchioPerturbationAnalyzer"
    ),
    "drake": (
        "src.engines.physics_engines.drake.python.perturbation.analyzer"
        "|DrakePerturbationAnalyzer"
    ),
    "mujoco": (
        "src.engines.physics_engines.mujoco.python.perturbation.analyzer"
        "|MuJoCoPerturbationAnalyzer"
    ),
    "opensim": (
        "src.engines.physics_engines.opensim.python.perturbation.analyzer"
        "|OpenSimPerturbationAnalyzer"
    ),
    "myosuite": (
        "src.engines.physics_engines.myosuite.python.perturbation.analyzer"
        "|MyoSuitePerturbationAnalyzer"
    ),
}


def _load_analyzer(engine_name: str, **kwargs: Any) -> Any:
    """Dynamically load and instantiate a PerturbationAnalyzer for an engine.

    Parameters
    ----------
    engine_name : str
        One of ``SUPPORTED_ENGINES``.
    **kwargs :
        Passed through to the analyzer constructor.

    Returns
    -------
    analyzer instance or raises ImportError / ValueError.
    """
    assert engine_name in SUPPORTED_ENGINES, (
        f"Unsupported engine: {engine_name!r}.  Choose from {SUPPORTED_ENGINES}"
    )

    entry = _ENGINE_LOADER_MAP[engine_name]
    module_path, cls_name = entry.split("|")

    import importlib  # noqa: PLC0415

    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Pendulum-specific loader (different constructor signature)
# ---------------------------------------------------------------------------


def _load_pendulum_analyzer(**kwargs: Any) -> Any:
    """Load PendulumPerturbationAnalyzer with default parameters."""
    # PendulumPerturbationAnalyzer lives in a namespace package excluded from mypy
    import importlib  # noqa: PLC0415

    _ppa_mod = importlib.import_module(  # type: ignore[assignment]
        "src.shared.python.pendulum_simulator.pendulum_perturbation_analyzer"
    )
    _phys_mod = importlib.import_module("src.shared.python.pendulum_simulator.physics")
    PendulumParams = _phys_mod.PendulumParams  # type: ignore[attr-defined]

    params = kwargs.pop("params", PendulumParams(m1=5.0, m2=0.30, L1=0.65, L2=1.10))
    t_end = kwargs.pop("t_end", 1.5)
    dt = kwargs.pop("dt", 0.01)
    return _ppa_mod.PendulumPerturbationAnalyzer(params, t_end=t_end, dt=dt)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


class CrossEnginePerturbationRunner:
    """Orchestrates perturbation analysis across multiple physics engines.

    Usage::

        runner = CrossEnginePerturbationRunner(
            engines=["pendulum", "mujoco"],
            profile={"coeffs": [[0.5, 0.1], [0.3, -0.05]]},
        )
        config = PerturbationConfig(n_trials=20, noise_amplitude=0.05)
        report = runner.run_all(config)
        for entry in report.ranking:
            logger.info("  %d. %s  RS=%.3f", entry.rank, entry.engine_name, entry.robustness_score)

    Design by Contract
    ------------------
    Pre:  ``engines`` must be a non-empty subset of ``SUPPORTED_ENGINES``.
    Pre:  ``profile`` must be a dict with a 'coeffs' key.
    Post: ``run_all()`` returns ``CrossEngineReport`` with rankings.
    """

    def __init__(
        self,
        engines: list[str] | None = None,
        profile: dict | None = None,
        engine_kwargs: dict[str, dict] | None = None,
        consistency_threshold: float = 0.2,
    ) -> None:
        """Initialise the cross-engine runner.

        Parameters
        ----------
        engines : list of str, optional
            Engine names to include.  Defaults to all supported engines.
        profile : dict, optional
            Base torque profile (``{"coeffs": [[...], [...]]}``).
            Can also be set later via ``set_profile()``.
        engine_kwargs : dict, optional
            Per-engine constructor kwargs.  E.g. ``{"mujoco": {"t_end": 2.0}}``.
        consistency_threshold : float
            CV threshold below which a metric is considered consistent
            across engines.  Default 0.2 (20%).
        """
        if engines is None:
            engines = list(SUPPORTED_ENGINES)

        for name in engines:
            assert name in SUPPORTED_ENGINES, (
                f"Unknown engine: {name!r}.  Supported: {SUPPORTED_ENGINES}"
            )

        self._engines = engines
        self._profile = profile
        self._engine_kwargs = engine_kwargs or {}
        self._consistency_threshold = consistency_threshold
        self._analyzers: dict[str, Any] = {}

    def set_profile(self, profile: dict) -> None:
        """Set the base torque profile for all engines.

        Design by Contract
        ------------------
        Pre: profile is a dict with 'coeffs' key.
        """
        assert isinstance(profile, dict), f"profile must be a dict, got {type(profile)}"
        assert "coeffs" in profile, "'coeffs' key missing from profile"
        self._profile = profile

    def _get_or_load_analyzer(self, engine_name: str) -> Any:
        """Load analyzer for engine, caching for reuse."""
        if engine_name not in self._analyzers:
            kwargs = self._engine_kwargs.get(engine_name, {})
            if engine_name == "pendulum":
                analyzer = _load_pendulum_analyzer(**kwargs)
            else:
                analyzer = _load_analyzer(engine_name, **kwargs)
            self._analyzers[engine_name] = analyzer
        return self._analyzers[engine_name]

    def run_all(self, config: PerturbationConfig) -> CrossEngineReport:
        """Run perturbation analysis on all configured engines.

        Design by Contract
        ------------------
        Pre:  ``profile`` has been set (via constructor or ``set_profile()``).
        Post: returned ``CrossEngineReport.summaries`` has one entry per
              engine that completed without error.
        Post: returned ``CrossEngineReport.ranking`` is sorted by
              robustness_score descending.

        Parameters
        ----------
        config : PerturbationConfig
            Shared perturbation config applied to all engines.

        Returns
        -------
        CrossEngineReport
        """
        assert self._profile is not None, (
            "set_profile() must be called before run_all()"
        )

        t_wall_start = time.monotonic()
        summaries: dict[str, PerturbationSummary] = {}
        failed_engines: list[str] = []

        for engine_name in self._engines:
            try:
                analyzer = self._get_or_load_analyzer(engine_name)
                analyzer.set_base_torque_profile(self._profile)
                summary = analyzer.run_batch(config)
                summaries[engine_name] = summary
                logger.info(
                    "%-12s  RS=%.4f  success=%.1f%%  t=%.2fs",
                    engine_name,
                    summary.robustness_score,
                    summary.success_rate * 100,
                    summary.execution_time_sec,
                )
            except Exception:  # noqa: BLE001
                logger.warning("Engine '%s' failed", engine_name, exc_info=True)
                failed_engines.append(engine_name)

        ranking = rank_engines(summaries)
        consistency = compute_consistency(summaries, self._consistency_threshold)
        total_time = time.monotonic() - t_wall_start

        return CrossEngineReport(
            config=config,
            summaries=summaries,
            ranking=ranking,
            consistency=consistency,
            failed_engines=failed_engines,
            total_time_sec=total_time,
        )

    def run_single(
        self, engine_name: str, config: PerturbationConfig
    ) -> PerturbationSummary:
        """Run perturbation analysis on a single engine.

        Design by Contract
        ------------------
        Pre:  ``profile`` has been set.
        Pre:  engine_name in SUPPORTED_ENGINES.
        """
        assert self._profile is not None, (
            "set_profile() must be called before run_single()"
        )
        assert engine_name in SUPPORTED_ENGINES, f"Unknown engine: {engine_name!r}"
        analyzer = self._get_or_load_analyzer(engine_name)
        analyzer.set_base_torque_profile(self._profile)
        return analyzer.run_batch(config)


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------


def rank_engines(
    summaries: dict[str, PerturbationSummary],
) -> list[EngineRankEntry]:
    """Sort engines by robustness_score descending.

    Parameters
    ----------
    summaries : dict mapping engine_name → PerturbationSummary

    Returns
    -------
    list of EngineRankEntry with rank starting at 1.
    """
    entries = [
        EngineRankEntry(
            engine_name=name,
            robustness_score=s.robustness_score,
            success_rate=s.success_rate,
            execution_time_sec=s.execution_time_sec,
        )
        for name, s in summaries.items()
    ]
    entries.sort(key=lambda e: e.robustness_score, reverse=True)
    for i, entry in enumerate(entries):
        entry.rank = i + 1
    return entries


def compute_consistency(
    summaries: dict[str, PerturbationSummary],
    threshold: float = 0.2,
) -> dict[str, ConsistencyMetrics]:
    """Compute cross-engine consistency for each scalar metric.

    For each scalar metric shared across engines, computes:
    - spread = max(means) − min(means)
    - CV = std(means) / |mean(means)|  (0 if mean ≈ 0)
    - is_consistent = CV < threshold

    Parameters
    ----------
    summaries : dict mapping engine_name → PerturbationSummary
    threshold : float
        CV threshold for consistency classification.

    Returns
    -------
    dict mapping metric_name → ConsistencyMetrics
    """
    import numpy as np  # noqa: PLC0415

    if not summaries:
        return {}

    # Collect scalar metric means across engines
    all_metric_names: set[str] = set()
    for summary in summaries.values():
        all_metric_names.update(summary.metrics.keys())

    consistency: dict[str, ConsistencyMetrics] = {}

    for metric_name in sorted(all_metric_names):
        engine_means: dict[str, float] = {}
        for engine_name, summary in summaries.items():
            stats = summary.metrics.get(metric_name)
            if stats is not None and hasattr(stats, "mean"):
                engine_means[engine_name] = float(stats.mean)

        if len(engine_means) < 2:
            continue  # Need at least 2 engines to compare

        means_arr = np.array(list(engine_means.values()))
        spread = float(np.max(means_arr) - np.min(means_arr))
        mean_val = float(np.mean(means_arr))
        std_val = float(np.std(means_arr))
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-12 else 0.0

        consistency[metric_name] = ConsistencyMetrics(
            metric_name=metric_name,
            engine_means=engine_means,
            spread=spread,
            coefficient_of_variation=cv,
            is_consistent=cv < threshold,
        )

    return consistency


def format_report(report: CrossEngineReport) -> str:
    """Format a CrossEngineReport as a human-readable string.

    Parameters
    ----------
    report : CrossEngineReport

    Returns
    -------
    Formatted string suitable for logging or printing.
    """
    lines = [
        "=" * 60,
        "Cross-Engine Perturbation Comparison Report",
        "=" * 60,
        f"Config: n_trials={report.config.n_trials}, "
        f"amplitude={report.config.noise_amplitude}, "
        f"mode={report.config.perturb_mode}",
        f"Total wall time: {report.total_time_sec:.2f}s",
        "",
        "Engine Ranking (by Robustness Score):",
        "-" * 40,
    ]

    for entry in report.ranking:
        lines.append(
            f"  {entry.rank}. {entry.engine_name:<12} "
            f"RS={entry.robustness_score:.4f}  "
            f"success={entry.success_rate * 100:.1f}%  "
            f"t={entry.execution_time_sec:.2f}s"
        )

    if report.failed_engines:
        lines += ["", "Failed Engines:", "-" * 40]
        for name in report.failed_engines:
            lines.append(f"  - {name}")

    if report.consistency:
        n_consistent = sum(1 for c in report.consistency.values() if c.is_consistent)
        n_total = len(report.consistency)
        lines += [
            "",
            f"Metric Consistency ({n_consistent}/{n_total} consistent "
            f"at CV < {0.2:.0%}):",
            "-" * 40,
        ]
        for metric_name, c in sorted(report.consistency.items()):
            status = "CONSISTENT" if c.is_consistent else "INCONSISTENT"
            lines.append(
                f"  {status:<12} {metric_name:<35} CV={c.coefficient_of_variation:.3f}"
            )

    lines.append("=" * 60)
    return "\n".join(lines)
