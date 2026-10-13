"""Source-backed, per-shot scores for the hypothetical matched-study course.

This is the reusable endpoint-vector boundary for paired interaction reports.
The public Tools scorer supplies scores at every baseline knot; the existing
cache interpolates its returned values and rejects out-of-support endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .scoring import YARDS_PER_METRE, build_broadie_approx_baseline
from .scoring_cache import source_backed_score_cache


@dataclass(frozen=True)
class CourseEndpointScoreVectors:
    """Indexed physical-shot scores and exact course/baseline provenance."""

    scores: dict[str, np.ndarray]
    primary_fraction: dict[str, float]
    primary_fraction_name: str
    scenario_id: str
    start_lie: str
    start_distance_m: float
    api_evaluated_states: int
    baseline: dict[str, str]


def score_course_endpoint_vectors(
    endpoints: dict[str, list[tuple[int, float, float]]],
    *,
    club_id: str,
    target_x_m: float,
    hole_distance_m: float = 400.0,
    fairway_half_width_m: float = 15.0,
    green_radius_m: float = 15.0,
) -> CourseEndpointScoreVectors:
    """Score the same three ordered physical populations as the course report.

    Driver uses a tee start, a straight fairway corridor and rough beyond it;
    irons use a fairway start, a centered circular green and rough beyond it.
    Landings are carry-only finishes without roll, elevation, or hazards.
    """
    if club_id not in {"driver", "seven_iron", "pitching_wedge"}:
        raise ValueError("club_id must be a named illustrative preset")
    dimensions = (target_x_m, hole_distance_m, fairway_half_width_m, green_radius_m)
    if not all(math.isfinite(value) and value > 0 for value in dimensions):
        raise ValueError("course dimensions must be finite and positive")
    if set(endpoints) != {"Straight", "Draw", "Fade"}:
        raise ValueError("all three paired patterns are required")
    indices = [index for index, _, _ in endpoints["Straight"]]
    if (
        len(indices) < 2
        or len(indices) != len(set(indices))
        or any(not isinstance(index, int) or index < 0 for index in indices)
    ):
        raise ValueError("at least two distinct paired shot indices are required")
    for shots in endpoints.values():
        if [index for index, _, _ in shots] != indices:
            raise ValueError("patterns must share ordered shot indices")
        if any(not math.isfinite(x) or not math.isfinite(y) for _, x, y in shots):
            raise ValueError("endpoints must be finite")

    driver = club_id == "driver"
    start_lie = "tee" if driver else "fairway"
    start_distance_m = hole_distance_m if driver else target_x_m
    primary_lie = "fairway" if driver else "green"
    primary_name = "fairway_fraction" if driver else "green_fraction"
    baseline = build_broadie_approx_baseline()
    try:
        cache = source_backed_score_cache(
            baseline,
            start_lie=start_lie,
            start_distance_yards=start_distance_m * YARDS_PER_METRE,
            finish_lies=(primary_lie, "rough"),
        )
        scores: dict[str, np.ndarray] = {}
        primary_fraction: dict[str, float] = {}
        for pattern, shots in endpoints.items():
            xy = np.asarray([(x, y) for _, x, y in shots], dtype=float)
            if driver:
                distances_m = np.hypot(hole_distance_m - xy[:, 0], xy[:, 1])
                primary = np.abs(xy[:, 1]) <= fairway_half_width_m
            else:
                distances_m = np.hypot(xy[:, 0] - target_x_m, xy[:, 1])
                primary = distances_m <= green_radius_m
            distances_yards = distances_m * YARDS_PER_METRE
            scored = np.empty(len(shots), dtype=float)
            for lie, mask in ((primary_lie, primary), ("rough", ~primary)):
                if bool(np.any(mask)):
                    scored[mask] = cache.score_many(lie, distances_yards[mask])
            if not np.all(np.isfinite(scored)):
                raise ValueError("source-backed endpoint scores must be finite")
            scores[pattern] = scored
            primary_fraction[pattern] = float(np.mean(primary))
    except ValueError as exc:
        raise ValueError(
            f"course state outside published baseline support: {exc}"
        ) from exc
    return CourseEndpointScoreVectors(
        scores=scores,
        primary_fraction=primary_fraction,
        primary_fraction_name=primary_name,
        scenario_id="tee_fairway_rough" if driver else "approach_green_rough",
        start_lie=start_lie,
        start_distance_m=start_distance_m,
        api_evaluated_states=cache.api_evaluated_states,
        baseline={
            "id": baseline.baseline_id,
            "version": baseline.version,
            "table_sha256": baseline.table_sha256,
            "source_url": baseline.source_url,
        },
    )
