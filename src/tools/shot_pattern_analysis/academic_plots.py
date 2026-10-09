"""Publication-oriented endpoint-density and raw-flight figures.

These functions consume supplied data only. They do not simulate shots, alter
aim points, launch a GUI, or infer player-level precision from face variance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .density import DensityComparison, shared_histogram_density

_COLORS = {
    "Straight": "#286090",
    "Draw": "#c66a19",
    "Fade": "#7b4a9e",
}
_HDR_STYLES = {
    0.50: ("solid", "50% HDR"),
    0.80: ("dashed", "80% HDR"),
    0.95: ("dotted", "95% HDR"),
}


def _validate_metadata(metadata: Mapping[str, Any]) -> str:
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping of scenario inputs")
    if not all(isinstance(key, str) and key.strip() for key in metadata):
        raise ValueError("metadata keys must be non-empty strings")
    values: list[str] = []
    for key, value in metadata.items():
        if key in ("valid_shots_by_pattern", "total_shots_by_pattern"):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metadata value for {key!r} must be finite")
        values.append(f"{key.replace('_', ' ')}: {value}")
    return "  |  ".join(values)


def _count_map(
    metadata: Mapping[str, Any], key: str, pattern_names: Sequence[str]
) -> dict[str, int]:
    raw = metadata.get(key)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping) or set(raw) != set(pattern_names):
        raise ValueError(f"metadata {key} must cover every plotted pattern")
    counts: dict[str, int] = {}
    for name, count in raw.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"metadata {key}[{name!r}] must be a non-negative integer")
        counts[name] = count
    return counts


def _population_caption(
    pattern_names: Sequence[str],
    valid_counts: Mapping[str, int],
    total_counts: Mapping[str, int],
) -> str:
    if not total_counts:
        return "Supported counts: " + "; ".join(
            f"{name} n_valid={valid_counts[name]:,}" for name in pattern_names
        )
    for name in pattern_names:
        if total_counts[name] < valid_counts[name]:
            raise ValueError(f"total shot count is below supported count for {name}")
    return "Valid/Total: " + "; ".join(
        f"{name} {valid_counts[name]:,}/{total_counts[name]:,} valid"
        for name in pattern_names
    )


def _trajectory_population_caption(
    pattern_names: Sequence[str],
    path_counts: Mapping[str, int],
    valid_counts: Mapping[str, int],
    total_counts: Mapping[str, int],
) -> str:
    if valid_counts and total_counts:
        return (
            _population_caption(pattern_names, valid_counts, total_counts)
            + "  |  Paths shown: "
            + "; ".join(f"{name} {path_counts[name]:,}" for name in pattern_names)
        )
    if valid_counts:
        return (
            "Supported counts: "
            + "; ".join(
                f"{name} n_valid={valid_counts[name]:,}" for name in pattern_names
            )
            + "  |  Paths shown: "
            + "; ".join(f"{name} {path_counts[name]:,}" for name in pattern_names)
        )
    if total_counts:
        return (
            "Paths shown: "
            + "; ".join(
                f"{name} {path_counts[name]:,} of {total_counts[name]:,} total"
                for name in pattern_names
            )
            + " (supported count not supplied)"
        )
    return "Paths shown: " + "; ".join(
        f"{name} {path_counts[name]:,}" for name in pattern_names
    )


def _validate_stem(output_stem: Path | str) -> Path:
    path = Path(output_stem)
    if path.suffix:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_formats(figure: Any, output_stem: Path, *, dpi: int) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for extension in ("png", "pdf", "svg"):
        path = output_stem.with_suffix(f".{extension}")
        figure.savefig(
            path,
            format=extension,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="white",
            metadata={"Creator": "UpstreamDrift shot-pattern analysis"},
        )
        outputs[extension] = path
    return outputs


def _matplotlib():
    try:
        import matplotlib
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"matplotlib is required for academic figures: {exc}"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _axis_centers(edges: NDArray[np.float64]) -> NDArray[np.float64]:
    return (edges[:-1] + edges[1:]) / 2.0


def _density_marginals(estimate: DensityComparison, density: NDArray[np.float64]):
    dx, dy = estimate.bin_resolution_m
    marginal_x = np.sum(density, axis=1) * dy
    marginal_y = np.sum(density, axis=0) * dx
    return marginal_x, marginal_y


def write_academic_dispersion_figure(
    points_by_pattern: Mapping[str, object],
    output_stem: Path | str,
    *,
    scenario: str,
    metadata: Mapping[str, Any],
    target_point_m: tuple[float, float],
    bins: int | tuple[int, int] = 64,
    region_masses: Sequence[float] = (0.50, 0.80, 0.95),
    dpi: int = 300,
) -> dict[str, Path]:
    """Export raw endpoint density with marginals and empirical HDR contours.

    The coordinates are plotted exactly as supplied. The target marker and
    common bounds are explicit; no aim correction or translation is applied.
    Each pattern uses the same histogram edges and shared density color scale.
    The marginal curves are normalized histogram densities, not KDE curves.

    Args:
        points_by_pattern: Pattern-name to N×2 endpoint array, in metres.
        output_stem: Path without a format extension; parent directories are made.
        scenario: Concise scenario name shown in the figure caption.
        metadata: Flexible reproducibility inputs (club, mode, delivery,
            speed, SDs, matched residual, and any additional controls).
        target_point_m: Target in the same coordinate frame as the endpoints.
        bins: Common histogram resolution, per axis or as one integer.
        region_masses: Requested empirical highest-density-region probability.

    Returns:
        Paths keyed by ``png``, ``pdf`` and ``svg``.
    """
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("scenario must be a non-empty string")
    if not isinstance(dpi, int) or not 72 <= dpi <= 600:
        raise ValueError("dpi must be an integer in [72, 600]")
    metadata_caption = _validate_metadata(metadata)
    estimate = shared_histogram_density(
        points_by_pattern,
        bins=bins,
        target_point_m=target_point_m,
        region_masses=region_masses,
    )
    names = list(estimate.patterns)
    names_tuple = tuple(names)
    valid_counts = _count_map(metadata, "valid_shots_by_pattern", names_tuple)
    total_counts = _count_map(metadata, "total_shots_by_pattern", names_tuple)
    actual_valid_counts = {name: estimate.patterns[name].n for name in names}
    if valid_counts and valid_counts != actual_valid_counts:
        raise ValueError("metadata valid_shots_by_pattern differs from endpoint counts")
    population = _population_caption(names, actual_valid_counts, total_counts)
    plt = _matplotlib()
    figure = plt.figure(figsize=(16.5, 8.8), facecolor="white")
    outer = figure.add_gridspec(
        1, len(names), left=0.06, right=0.91, top=0.82, bottom=0.17, wspace=0.27
    )
    for name, panel_index in zip(names, range(len(names)), strict=True):
        inner = outer[0, panel_index].subgridspec(
            2, 2, height_ratios=(1, 4), width_ratios=(4, 1), hspace=0.04, wspace=0.04
        )
        axis_top = figure.add_subplot(inner[0, 0])
        axis = figure.add_subplot(inner[1, 0], sharex=axis_top)
        axis_right = figure.add_subplot(inner[1, 1], sharey=axis)
        pattern = estimate.patterns[name]
        density = pattern.density_per_m2
        marginal_x, marginal_y = _density_marginals(estimate, density)
        color = _COLORS.get(name, "#286090")

        mesh = axis.pcolormesh(
            estimate.x_edges_m,
            estimate.y_edges_m,
            density.T,
            cmap="Blues",
            vmin=0.0,
            vmax=estimate.density_vmax_per_m2,
            shading="flat",
            rasterized=True,
        )
        sample = np.asarray(points_by_pattern[name], dtype=float)
        axis.scatter(
            sample[:, 0],
            sample[:, 1],
            s=7,
            color=color,
            alpha=0.30,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        for mass in region_masses:
            mass_value = float(mass)
            threshold = pattern.hdr_density_thresholds[mass_value]
            if threshold <= 0 or threshold >= estimate.density_vmax_per_m2:
                continue
            style, _ = _HDR_STYLES.get(mass_value, ("solid", f"{mass_value:.0%} HDR"))
            axis.contour(
                _axis_centers(estimate.x_edges_m),
                _axis_centers(estimate.y_edges_m),
                density.T,
                levels=[threshold],
                colors=["#222222"],
                linewidths=1.0,
                linestyles=[style],
                zorder=4,
            )

        axis.scatter(
            [target_point_m[0]],
            [target_point_m[1]],
            marker="+",
            s=35,
            color="#111111",
            linewidths=1.0,
            zorder=5,
            label="Target",
        )
        axis.axvline(target_point_m[0], color="#333333", linewidth=0.55, alpha=0.55)
        axis.axhline(target_point_m[1], color="#333333", linewidth=0.55, alpha=0.55)
        axis.set_xlim(estimate.x_limits_m)
        axis.set_ylim(estimate.y_limits_m)
        axis.set_title(f"{name}  (n={pattern.n:,})", fontsize=12, fontweight="bold")
        axis.set_xlabel("Downrange Endpoint (m)")
        if panel_index == 0:
            axis.set_ylabel("Lateral Endpoint (m; Right Positive)")
        axis.grid(color="#666666", linewidth=0.35, alpha=0.25)
        axis.set_facecolor("white")

        axis_top.stairs(marginal_x, estimate.x_edges_m, color=color, linewidth=1.1)
        axis_top.set_xlim(estimate.x_limits_m)
        axis_top.tick_params(axis="x", labelbottom=False, bottom=False)
        axis_top.tick_params(axis="y", labelsize=7, length=2)
        axis_top.set_ylabel("Density", fontsize=8)
        axis_top.grid(axis="y", linewidth=0.3, alpha=0.25)
        axis_right.stairs(
            marginal_y,
            estimate.y_edges_m,
            orientation="horizontal",
            color=color,
            linewidth=1.1,
        )
        axis_right.set_ylim(estimate.y_limits_m)
        axis_right.tick_params(axis="y", labelleft=False, left=False)
        axis_right.tick_params(axis="x", labelsize=7, length=2)
        axis_right.set_xlabel("Density", fontsize=8)
        axis_right.grid(axis="x", linewidth=0.3, alpha=0.25)

    colorbar_axis = figure.add_axes((0.93, 0.28, 0.014, 0.48))
    colorbar = figure.colorbar(mesh, cax=colorbar_axis)
    colorbar.set_label("Endpoint Probability Density (m⁻²)")
    figure.suptitle(
        "Shot-Endpoint Density Conditional on Supported Airborne Shots",
        x=0.06,
        y=0.965,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(0.06, 0.92, scenario, fontsize=12, ha="left")
    figure.text(
        0.06,
        0.885,
        f"{metadata_caption}  |  {population}" if metadata_caption else population,
        fontsize=9,
        ha="left",
        color="#333333",
    )

    hdr_handles = []
    from matplotlib.lines import Line2D

    for mass in region_masses:
        mass_value = float(mass)
        style, label = _HDR_STYLES.get(mass_value, ("solid", f"{mass_value:.0%} HDR"))
        hdr_handles.append(
            Line2D(
                [0], [0], color="#222222", linestyle=style, linewidth=1.2, label=label
            )
        )
    hdr_handles.append(
        Line2D(
            [0],
            [0],
            color="#111111",
            marker="+",
            linestyle="none",
            markersize=6,
            label="Target",
        )
    )
    figure.legend(
        handles=hdr_handles,
        loc="lower center",
        bbox_to_anchor=(0.48, 0.085),
        ncol=len(hdr_handles),
        frameon=False,
        title="Empirical Highest-Density Regions",
    )
    achieved_text = "  |  ".join(
        f"{name}: "
        + ", ".join(
            f"{mass:.0%} request→{estimate.patterns[name].hdr_achieved_mass[float(mass)]:.1%}"
            for mass in region_masses
        )
        for name in names
    )
    dx, dy = estimate.bin_resolution_m
    figure.text(
        0.06,
        0.025,
        f"Empirical histogram only; tied bins can exceed requested HDR mass. "
        f"No KDE or synthetic covariance spread; not calibrated player precision. "
        f"Bin resolution: {dx:.3g} × {dy:.3g} m. {achieved_text}",
        fontsize=8,
        ha="left",
    )
    outputs = _save_formats(figure, _validate_stem(output_stem), dpi=dpi)
    plt.close(figure)
    return outputs


def _validated_trajectories(
    trajectories_by_pattern: Mapping[str, Sequence[object]],
) -> dict[str, list[NDArray[np.float64]]]:
    if not isinstance(trajectories_by_pattern, Mapping) or not trajectories_by_pattern:
        raise ValueError("trajectories_by_pattern must contain at least one pattern")
    validated: dict[str, list[NDArray[np.float64]]] = {}
    for name, paths in trajectories_by_pattern.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("pattern names must be non-empty strings")
        if not paths:
            raise ValueError(f"pattern {name!r} needs at least one trajectory")
        validated_paths = []
        for path in paths:
            try:
                values = np.asarray(path, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"trajectory for {name!r} must be finite (x, y) pairs"
                ) from exc
            if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
                raise ValueError(
                    f"trajectory for {name!r} needs at least two (x, y) points"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"trajectory for {name!r} must be finite")
            validated_paths.append(values)
        validated[name] = validated_paths
    return validated


def write_overhead_trajectory_figure(
    trajectories_by_pattern: Mapping[str, Sequence[object]],
    output_stem: Path | str,
    *,
    target_x_m: float,
    scenario: str,
    metadata: Mapping[str, Any],
    trajectories_are_aimed: bool = False,
    dpi: int = 300,
) -> dict[str, Path]:
    """Export a shared-axis comparison of supplied raw overhead trajectories.

    Trajectories are plotted exactly as supplied in downrange/right-positive
    lateral coordinates. They are not rotated to an aim line or re-simulated.
    Set ``trajectories_are_aimed`` only to trigger a fail-fast contract error;
    this figure is explicitly for raw, un-aimed simulator trajectories.
    """
    if trajectories_are_aimed:
        raise ValueError("overhead trajectory input must be raw, not aim-adjusted")
    if not isinstance(target_x_m, (int, float)) or not math.isfinite(target_x_m):
        raise ValueError("target_x_m must be finite")
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("scenario must be a non-empty string")
    if not isinstance(dpi, int) or not 72 <= dpi <= 600:
        raise ValueError("dpi must be an integer in [72, 600]")
    metadata_caption = _validate_metadata(metadata)
    trajectories = _validated_trajectories(trajectories_by_pattern)
    names_tuple = tuple(trajectories)
    valid_counts = _count_map(metadata, "valid_shots_by_pattern", names_tuple)
    total_counts = _count_map(metadata, "total_shots_by_pattern", names_tuple)
    if valid_counts and any(
        valid_counts[name] < len(trajectories[name]) for name in names_tuple
    ):
        raise ValueError("valid_shots_by_pattern cannot be below displayed paths")
    path_counts = {name: len(trajectories[name]) for name in names_tuple}
    population = _trajectory_population_caption(
        names_tuple, path_counts, valid_counts, total_counts
    )
    all_points = np.concatenate(
        [path for paths in trajectories.values() for path in paths], axis=0
    )
    lows = np.minimum(np.min(all_points, axis=0), (0.0, 0.0))
    highs = np.maximum(np.max(all_points, axis=0), (float(target_x_m), 0.0))
    spans = highs - lows
    for axis in range(2):
        padding = max(float(spans[axis]) * 0.05, 0.25)
        lows[axis] -= padding
        highs[axis] += padding

    plt = _matplotlib()
    names = list(trajectories)
    figure, axes = plt.subplots(
        1,
        len(names),
        figsize=(6.0 * len(names), 6.0),
        sharex=True,
        sharey=True,
        squeeze=False,
        facecolor="white",
    )
    for axis, name in zip(axes[0], names, strict=True):
        color = _COLORS.get(name, "#286090")
        for trajectory in trajectories[name]:
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=color,
                alpha=0.24,
                linewidth=0.8,
                rasterized=True,
            )
        axis.scatter([target_x_m], [0.0], marker="+", color="#111111", s=38, zorder=3)
        axis.axvline(target_x_m, color="#333333", linestyle=":", linewidth=0.8)
        axis.axhline(0.0, color="#555555", linewidth=0.45)
        axis.set_xlim(float(lows[0]), float(highs[0]))
        axis.set_ylim(float(lows[1]), float(highs[1]))
        valid_text = f"; valid={valid_counts[name]:,}" if valid_counts else ""
        total_text = f"/{total_counts[name]:,}" if total_counts and valid_counts else ""
        axis.set_title(
            f"{name}  (paths shown={path_counts[name]:,}{valid_text}{total_text})",
            fontsize=11,
            fontweight="bold",
        )
        axis.set_xlabel("Downrange Position (m)")
        axis.grid(color="#666666", linewidth=0.35, alpha=0.25)
    axes[0][0].set_ylabel("Lateral Position (m; Right Positive)")
    figure.suptitle(
        "Raw Overhead Flight-Trajectory Comparison",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(0.06, 0.92, scenario, fontsize=12, ha="left")
    figure.text(
        0.06,
        0.885,
        f"{metadata_caption}  |  {population}" if metadata_caption else population,
        fontsize=9,
        ha="left",
        color="#333333",
    )
    figure.text(
        0.06,
        0.025,
        "Raw trajectory coordinates; no aim-line rotation. Only the displayed sample paths are plotted; paths are sampled from supported airborne shots.",
        fontsize=9,
        ha="left",
    )
    outputs = _save_formats(figure, _validate_stem(output_stem), dpi=dpi)
    plt.close(figure)
    return outputs


__all__ = ["write_academic_dispersion_figure", "write_overhead_trajectory_figure"]
