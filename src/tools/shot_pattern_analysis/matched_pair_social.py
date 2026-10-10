"""Two-condition social views of already paired, supported shot endpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path

import numpy as np

from .academic_plots import write_academic_dispersion_figure

_PATTERNS = ("Straight", "Draw", "Fade")


def write_pair_social_density(
    point_sets: Mapping[str, Sequence[tuple[float, float]] | np.ndarray],
    output_stem: Path,
    *,
    condition_labels: tuple[str, str],
    scenario: str,
    social_club: str,
    target_point_m: tuple[float, float],
    total_n: int,
    delivery_noise: str | None = None,
    source_protocols: tuple[str, str] | None = None,
    target_modes: tuple[str, str] | None = None,
) -> dict[str, Path]:
    """Export three two-panel figures, preserving all supplied coordinates.

    The caller must supply the identical common shot indices from all six
    populations, in the same order. This presentation layer checks shapes,
    counts and finite coordinates; upstream saved-data validation establishes
    shot identities. Each figure shares axes, histogram bins and density scales
    between conditions. Scales can differ between physical-pattern figures.
    No flights, density algorithm, aim transformation or resampling are added.
    """
    if (
        not isinstance(condition_labels, (tuple, list, Sequence))
        or isinstance(condition_labels, (str, bytes))
        or len(condition_labels) != 2
        or any(
            not isinstance(label, str) or not label.strip()
            for label in condition_labels
        )
        or len(set(condition_labels)) != 2
    ):
        raise ValueError("two distinct non-empty condition labels are required")
    if any(
        not isinstance(label, str) or not label.strip()
        for label in (scenario, social_club)
    ):
        raise ValueError("scenario and social club must be non-empty display labels")
    if delivery_noise is not None and (
        not isinstance(delivery_noise, str) or not delivery_noise.strip()
    ):
        raise ValueError("delivery noise must be a non-empty display label")

    if source_protocols is not None:
        if (
            not isinstance(source_protocols, (tuple, list, Sequence))
            or isinstance(source_protocols, (str, bytes))
            or len(source_protocols) != 2
            or any(not isinstance(p, str) or not p.strip() for p in source_protocols)
        ):
            raise ValueError("source_protocols must be a pair of two non-empty strings")

    if target_modes is not None:
        if (
            not isinstance(target_modes, (tuple, list, Sequence))
            or isinstance(target_modes, (str, bytes))
            or len(target_modes) != 2
            or any(not isinstance(m, str) or not m.strip() for m in target_modes)
        ):
            raise ValueError("target_modes must be a pair of two non-empty strings")

    if (
        not isinstance(target_point_m, (tuple, list, Sequence))
        or isinstance(target_point_m, (str, bytes))
        or len(target_point_m) != 2
    ):
        raise ValueError(
            "target_point_m must contain exactly two finite real coordinates"
        )
    for c in target_point_m:
        if (
            isinstance(c, bool)
            or not isinstance(c, (int, float, np.floating, np.integer))
            or not math.isfinite(float(c))
        ):
            raise ValueError(
                "target_point_m must contain exactly two finite real coordinates"
            )
    target_x = float(target_point_m[0])
    target_y = float(target_point_m[1])

    if target_y != 0.0:
        coord_str = f"{target_x:.2f} m downrange, {target_y:.2f} m lateral"
    else:
        coord_str = f"{target_x:.2f} m downrange"

    if target_modes is not None:
        target_str = (
            f"{coord_str} ("
            f"{condition_labels[0]}: {target_modes[0]} · "
            f"{condition_labels[1]}: {target_modes[1]})"
        )
    else:
        target_str = f"{coord_str} (Common Target)"

    protocol_str: str | None = None
    if source_protocols is not None:
        protocol_str = (
            f"{condition_labels[0]}: {source_protocols[0]} · "
            f"{condition_labels[1]}: {source_protocols[1]}"
        )

    if isinstance(total_n, bool) or not isinstance(total_n, int) or total_n < 2:
        raise ValueError("total_n must be an integer >= 2")
    expected = {
        f"{pattern} · {label}" for pattern in _PATTERNS for label in condition_labels
    }
    if set(point_sets) != expected:
        raise ValueError(
            "point sets must cover exactly three patterns and two conditions"
        )
    arrays = {
        name: np.asarray(points, dtype=float) for name, points in point_sets.items()
    }
    if any(
        points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
        for points in arrays.values()
    ):
        raise ValueError("every endpoint population must be a finite N by 2 array")
    counts = {len(points) for points in arrays.values()}
    if len(counts) != 1 or not 2 <= next(iter(counts)) <= total_n:
        raise ValueError(
            "six common populations must have identical counts in [2, total_n]"
        )
    common_n = next(iter(counts))
    outputs: dict[str, Path] = {}
    for pattern in _PATTERNS:
        points = {label: arrays[f"{pattern} · {label}"] for label in condition_labels}
        pair_metadata = {
            "social_club": social_club,
            **({"Protocol": protocol_str} if protocol_str else {}),
            "Matched Target": target_str,
            **({"Delivery Noise": delivery_noise} if delivery_noise else {}),
            "Population": "Same Shot Indices Supported in All Six Populations",
            "Scales": "Shared Within This Figure; Other Pattern Figures May Differ",
            "valid_shots_by_pattern": dict.fromkeys(condition_labels, common_n),
            "total_shots_by_pattern": dict.fromkeys(condition_labels, total_n),
        }
        paths = write_academic_dispersion_figure(
            points,
            output_stem.with_name(f"{output_stem.name}_{pattern.lower()}_social"),
            scenario=f"{pattern}: {scenario}",
            metadata=pair_metadata,
            target_point_m=target_point_m,
            presentation="social",
        )
        outputs.update(
            {f"{pattern.lower()}_{kind}": path for kind, path in paths.items()}
        )
    return outputs


__all__ = ["write_pair_social_density"]
