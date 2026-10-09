"""Shared, empirical two-dimensional endpoint density grids.

This module intentionally uses histogram probability mass rather than a KDE.
The histogram preserves singular and near-one-dimensional samples without
inventing spatial spread. HDR cutoffs are discrete: tied bins can make the
achieved mass exceed the requested mass.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PatternDensity:
    """Histogram density and discrete highest-density-region summaries."""

    n: int
    density_per_m2: FloatArray
    hdr_density_thresholds: Mapping[float, float]
    hdr_achieved_mass: Mapping[float, float]
    maximum_bin_probability: float


@dataclass(frozen=True)
class DensityComparison:
    """Common coordinates and probability-density scale for several patterns."""

    x_edges_m: FloatArray
    y_edges_m: FloatArray
    x_limits_m: tuple[float, float]
    y_limits_m: tuple[float, float]
    bin_resolution_m: tuple[float, float]
    density_vmax_per_m2: float
    patterns: Mapping[str, PatternDensity]


def _coordinate_array(name: str, points: object) -> FloatArray:
    try:
        values = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"pattern {name!r} points must be finite (x, y) pairs"
        ) from exc
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError(f"pattern {name!r} needs at least one (x, y) point")
    if not np.isfinite(values).all():
        raise ValueError(f"pattern {name!r} points must be finite")
    return values


def _validated_edges(edges: Sequence[float], name: str) -> FloatArray:
    values = np.asarray(edges, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} edges must be a finite one-dimensional sequence")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} edges must be strictly increasing")
    return values


def _hdr_summaries(
    density: FloatArray,
    cell_areas: FloatArray,
    requested_masses: tuple[float, ...],
) -> tuple[dict[float, float], dict[float, float]]:
    bin_probability = density * cell_areas
    flat_order = np.argsort(density.ravel())[::-1]
    ordered_probability = bin_probability.ravel()[flat_order]
    cumulative_probability = np.cumsum(ordered_probability)
    thresholds: dict[float, float] = {}
    achieved: dict[float, float] = {}
    flat_density = density.ravel()
    for requested in requested_masses:
        index = min(
            int(np.searchsorted(cumulative_probability, requested, side="left")),
            len(flat_order) - 1,
        )
        cutoff = float(flat_density[flat_order[index]])
        thresholds[requested] = cutoff
        achieved[requested] = float(np.sum(bin_probability[density >= cutoff]))
    return thresholds, achieved


def shared_histogram_density(
    points_by_pattern: Mapping[str, object],
    *,
    bins: int | tuple[int, int] = 64,
    target_point_m: tuple[float, float] = (0.0, 0.0),
    padding_fraction: float = 0.05,
    constant_axis_half_width_m: float = 0.5,
    region_masses: Sequence[float] = (0.50, 0.80, 0.95),
) -> DensityComparison:
    """Build comparable 2D histogram densities from endpoint coordinates.

    Input coordinates are used as supplied (for example, raw downrange and
    right-positive lateral endpoints). ``target_point_m`` adds the target to
    the common plot bounds and anchors the zero-error marker; it does not
    translate, aim-adjust, or otherwise transform any endpoint.

    Histogram density is ``count / (n * bin area)`` and integrates to one for
    every pattern. No smoothing is applied. For a constant coordinate, the
    plotting grid receives a support window of ``constant_axis_half_width_m``
    on either side, but observations remain in their observed bin; this adds
    no sample variation. HDR cutoffs use whole bins and report the achieved
    mass, which may exceed the requested mass when bins tie.

    Args:
        points_by_pattern: Non-empty mapping of pattern names to N×2 arrays.
        bins: Number of shared bins on both axes or per-axis bin counts.
        target_point_m: Explicit target coordinates in the same frame as data.
        padding_fraction: Fraction of each non-constant data span added per side.
        constant_axis_half_width_m: Display-only half-span for a constant axis.
        region_masses: Requested empirical HDR masses, each strictly between 0
            and 1.

    Raises:
        ValueError: If coordinates, grid controls or HDR probabilities are invalid.
    """
    if not isinstance(points_by_pattern, Mapping) or not points_by_pattern:
        raise ValueError("points_by_pattern must contain at least one pattern")
    points = {
        name: _coordinate_array(name, sample)
        for name, sample in points_by_pattern.items()
        if isinstance(name, str) and name.strip()
    }
    if len(points) != len(points_by_pattern):
        raise ValueError("pattern names must be non-empty strings")
    try:
        target = np.asarray(target_point_m, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_point_m must be a finite (x, y) pair") from exc
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("target_point_m must be a finite (x, y) pair")
    if not math.isfinite(padding_fraction) or padding_fraction < 0:
        raise ValueError("padding_fraction must be finite and non-negative")
    if not math.isfinite(constant_axis_half_width_m) or constant_axis_half_width_m <= 0:
        raise ValueError("constant_axis_half_width_m must be finite and positive")
    if isinstance(bins, int):
        shape = (bins, bins)
    elif isinstance(bins, tuple) and len(bins) == 2:
        shape = bins
    else:
        raise ValueError("bins must be an integer or a pair of integers")
    if any(not isinstance(value, int) or value < 4 for value in shape):
        raise ValueError("each bin count must be an integer >= 4")
    try:
        masses = tuple(float(value) for value in region_masses)
    except (TypeError, ValueError) as exc:
        raise ValueError("region_masses must be probabilities in (0, 1)") from exc
    if (
        not masses
        or len(set(masses)) != len(masses)
        or any(not math.isfinite(value) or not 0 < value < 1 for value in masses)
    ):
        raise ValueError("region_masses must be unique probabilities in (0, 1)")

    all_points = np.concatenate((*points.values(), target.reshape(1, 2)), axis=0)
    lows = np.min(all_points, axis=0)
    highs = np.max(all_points, axis=0)
    spans = highs - lows
    for axis in range(2):
        if spans[axis] == 0:
            lows[axis] -= constant_axis_half_width_m
            highs[axis] += constant_axis_half_width_m
        else:
            pad = spans[axis] * padding_fraction
            lows[axis] -= pad
            highs[axis] += pad

    edges = tuple(
        np.linspace(
            float(lows[axis]), float(highs[axis]), shape[axis] + 1, dtype=np.float64
        )
        for axis in range(2)
    )
    x_edges, y_edges = edges
    cell_areas = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
    estimates: dict[str, PatternDensity] = {}
    for name, sample in points.items():
        counts, _, _ = np.histogram2d(sample[:, 0], sample[:, 1], bins=edges)
        density = counts / (len(sample) * cell_areas)
        max_bin_probability = float(np.max(counts) / len(sample))
        thresholds, achieved = _hdr_summaries(density, cell_areas, masses)
        estimates[name] = PatternDensity(
            n=len(sample),
            density_per_m2=density,
            hdr_density_thresholds=thresholds,
            hdr_achieved_mass=achieved,
            maximum_bin_probability=max_bin_probability,
        )

    resolution = (
        float(np.diff(x_edges)[0]),
        float(np.diff(y_edges)[0]),
    )
    return DensityComparison(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        x_limits_m=(float(x_edges[0]), float(x_edges[-1])),
        y_limits_m=(float(y_edges[0]), float(y_edges[-1])),
        bin_resolution_m=resolution,
        density_vmax_per_m2=max(
            float(np.max(item.density_per_m2)) for item in estimates.values()
        ),
        patterns=estimates,
    )


__all__ = ["DensityComparison", "PatternDensity", "shared_histogram_density"]
