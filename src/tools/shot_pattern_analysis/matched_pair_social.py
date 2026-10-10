"""Two-condition social views of already paired, supported shot endpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

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
    source_protocols: tuple[str, str] | Sequence[str] | None = None,
    context: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
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
        len(condition_labels) != 2
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
    resolved_protocols = source_protocols
    if resolved_protocols is None and context is not None:
        if "source_protocols" in context:
            resolved_protocols = context["source_protocols"]
        elif "protocols" in context:
            resolved_protocols = context["protocols"]
        elif "first_protocol" in context and "second_protocol" in context:
            resolved_protocols = (
                context["first_protocol"],
                context["second_protocol"],
            )

    if resolved_protocols is not None:
        if (
            not isinstance(resolved_protocols, Sequence)
            or isinstance(resolved_protocols, (str, bytes))
            or len(resolved_protocols) != 2
            or any(not isinstance(p, str) or not p.strip() for p in resolved_protocols)
        ):
            raise ValueError(
                "source_protocols must be a sequence of two non-empty strings"
            )
    target_x = float(target_point_m[0])
    if not math.isfinite(target_x):
        raise ValueError("target_point_m must contain finite coordinates")
    target_context = None
    if context is not None and isinstance(context, Mapping):
        target_context = context.get("target_context") or context.get("target_mode")
    target_str = (
        f"{target_x:.2f} m downrange ({target_context})"
        if target_context
        else f"{target_x:.2f} m downrange (Common Target)"
    )
    protocol_str: str | None = None
    if resolved_protocols is not None:
        protocol_str = (
            f"{condition_labels[0]}: {resolved_protocols[0]} · "
            f"{condition_labels[1]}: {resolved_protocols[1]}"
        )
    elif metadata is not None and "Protocol" in metadata:
        protocol_str = str(metadata["Protocol"])
    elif context is not None and "Protocol" in context:
        protocol_str = str(context["Protocol"])

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
        if metadata:
            for key, val in metadata.items():
                if key not in pair_metadata:
                    pair_metadata[key] = val
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
