"""PF-08 (#10438) time-to-accepted-swing benchmark schemas for MS-105.

Publishes measured *service budgets* and cache-key contracts. Never invents
a universal solve-time guarantee — ``guarantee`` is always ``False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import JobStage

__all__ = [
    "SERVICE_TARGETS",
    "StageBenchmarkSample",
    "TimeToAcceptedReport",
    "measure_stage_sample",
    "publish_service_targets",
]

_BENCH_SCHEMA = "time-to-accepted-swing/1.0.0"
_GOVERNING_ISSUE = 10379
_FOLDED_ISSUE = 10438

SERVICE_TARGETS: dict[str, Any] = {
    "schema": _BENCH_SCHEMA,
    "governing_issue": _GOVERNING_ISSUE,
    "folded_from": _FOLDED_ISSUE,
    "guarantee": False,
    "hardware": "installation-independent reference host (publish measured limits)",
    "stages": {
        JobStage.IK.value: {
            "median_budget_s": 5.0,
            "p95_budget_s": None,
            "notes": "budget, not a guarantee",
        },
        JobStage.REALLOCATION.value: {
            "median_budget_s": 1.0,
            "p95_budget_s": None,
            "notes": "budget, not a guarantee",
        },
        JobStage.CANDIDATE.value: {
            "median_budget_s": 10.0,
            "p95_budget_s": None,
            "notes": "budget, not a guarantee",
        },
        JobStage.FIT_REPLAY.value: {
            "median_budget_s": 30.0,
            "p95_budget_s": None,
            "notes": "budget, not a guarantee",
        },
    },
    "cache_invalidation": (
        "caches keyed by model/contact/trajectory/runtime hashes; miss forces cold path"
    ),
}


@dataclass(frozen=True, slots=True)
class StageBenchmarkSample:
    """One measured cold/warm stage sample with memory and cache identity."""

    stage: JobStage
    cold_s: float
    warm_s: float
    peak_memory_mb: float
    cache_key: str
    cache_hit: bool

    def __post_init__(self) -> None:
        if self.cold_s < 0 or self.warm_s < 0:
            raise ValueError("stage times must be >= 0")
        if self.peak_memory_mb < 0:
            raise ValueError("peak_memory_mb must be >= 0")
        if not self.cache_key.strip():
            raise ValueError("cache_key must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "cold_s": self.cold_s,
            "warm_s": self.warm_s,
            "peak_memory_mb": self.peak_memory_mb,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
        }


@dataclass(frozen=True, slots=True)
class TimeToAcceptedReport:
    """Aggregated measured report; universal guarantees are forbidden."""

    samples: tuple[StageBenchmarkSample, ...]
    hardware_label: str
    guarantee: bool = False

    def __post_init__(self) -> None:
        if self.guarantee:
            raise ValueError(
                "universal solve-time guarantee is forbidden; "
                "publish measured budgets only (guarantee=False)"
            )
        if not self.hardware_label.strip():
            raise ValueError("hardware_label must be non-empty")
        object.__setattr__(self, "samples", tuple(self.samples))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _BENCH_SCHEMA,
            "hardware_label": self.hardware_label,
            "guarantee": False,
            "samples": [s.to_dict() for s in self.samples],
            "service_targets": publish_service_targets(),
        }


def measure_stage_sample(
    stage: JobStage,
    *,
    cold_s: float,
    warm_s: float,
    peak_memory_mb: float,
    cache_key: str,
    cache_hit: bool,
) -> StageBenchmarkSample:
    """Build a typed stage sample from measured timings (no invention)."""
    return StageBenchmarkSample(
        stage=stage,
        cold_s=cold_s,
        warm_s=warm_s,
        peak_memory_mb=peak_memory_mb,
        cache_key=cache_key,
        cache_hit=cache_hit,
    )


def publish_service_targets() -> Mapping[str, Any]:
    """Return the frozen service-budget table (not a solve-time guarantee)."""
    return SERVICE_TARGETS
