"""Adequacy-gated summaries for antithetic topology experiments.

Antithetic replicates are dependent within a mirrored pair.  This module uses
the pair—not the individual rollout—as the independent sampling unit and
suppresses probability-like output until the registered count and precision
gates pass.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from statistics import NormalDist

from scripts.research.proximal_distal_energy.event_robustness_study import (
    DelayNoiseTopologyResult,
)
from scripts.research.proximal_distal_energy.event_topology_robustness import (
    GlobalEventTopology,
)


@dataclass(frozen=True, slots=True)
class TopologyAdequacyConfig:
    """Independent-pair count and interval-precision publication gate."""

    required_independent_pairs: int = 96
    maximum_interval_half_width: float = 0.10
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if (
            isinstance(self.required_independent_pairs, bool)
            or not isinstance(self.required_independent_pairs, int)
            or self.required_independent_pairs < 1
        ):
            raise ValueError("required_independent_pairs must be positive")
        if (
            not math.isfinite(self.maximum_interval_half_width)
            or not 0.0 < self.maximum_interval_half_width < 1.0
        ):
            raise ValueError("maximum_interval_half_width must lie in (0, 1)")
        if not math.isfinite(self.confidence) or not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class DelayTopologySummary:
    """Raw topology counts and, when adequate, a pair-preservation interval."""

    delay_s: float
    topology_counts: tuple[tuple[str, int], ...]
    independent_pair_count: int
    preserved_pair_count: int
    adequacy_passed: bool
    preservation_fraction: float | None
    preservation_interval: tuple[float, float] | None


def _topology_signature(topology: GlobalEventTopology) -> tuple[object, ...]:
    return (
        topology.status.value,
        topology.crossing_count,
        tuple(event.direction.value for event in topology.events),
    )


def _wilson_interval(
    successes: int, total: int, *, confidence: float
) -> tuple[float, float]:
    proportion = successes / total
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    half_width = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def summarize_topology_by_delay(
    result: DelayNoiseTopologyResult,
    *,
    config: TopologyAdequacyConfig,
) -> tuple[DelayTopologySummary, ...]:
    """Summarize topology without treating mirrored replicates as independent."""

    if result.replicate_count % 2 != 0:
        raise ValueError("antithetic topology results require an even replicate count")
    pair_count = result.replicate_count // 2
    summaries: list[DelayTopologySummary] = []
    for nominal in result.nominal.outcomes:
        retained = [item for item in result.outcomes if item.delay_s == nominal.delay_s]
        retained.sort(key=lambda item: item.replicate_index)
        if [item.replicate_index for item in retained] != list(
            range(result.replicate_count)
        ):
            raise ValueError("each delay must retain every replicate exactly once")
        nominal_signature = _topology_signature(nominal.topology)
        preserved = sum(
            _topology_signature(retained[index].topology) == nominal_signature
            and _topology_signature(retained[index + pair_count].topology)
            == nominal_signature
            for index in range(pair_count)
        )
        interval = _wilson_interval(preserved, pair_count, confidence=config.confidence)
        half_width = (interval[1] - interval[0]) / 2.0
        adequate = (
            pair_count >= config.required_independent_pairs
            and half_width <= config.maximum_interval_half_width
        )
        counts = Counter(item.topology.status.value for item in retained)
        summaries.append(
            DelayTopologySummary(
                delay_s=nominal.delay_s,
                topology_counts=tuple(sorted(counts.items())),
                independent_pair_count=pair_count,
                preserved_pair_count=preserved,
                adequacy_passed=adequate,
                preservation_fraction=(preserved / pair_count if adequate else None),
                preservation_interval=(interval if adequate else None),
            )
        )
    return tuple(summaries)


__all__ = [
    "DelayTopologySummary",
    "TopologyAdequacyConfig",
    "summarize_topology_by_delay",
]
