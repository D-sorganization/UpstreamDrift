"""Publication-oriented endpoint-density and raw-flight figures.

These functions consume supplied data only. They do not simulate shots, alter
aim points, launch a GUI, or infer player-level precision from face variance.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .density import DensityComparison, shared_histogram_density

_COLORS = {
    "Straight": "#286090",
    "Draw": "#c66a19",
    "Fade": "#7b4a9e",
}
_PILOT_FIGURE_LABEL = "Illustrative Simulation • Assumed Head Properties • Carry Only"
_LineStyle = Literal["solid", "dashed", "dotted"]
_HDR_STYLES = {
    0.50: ("solid", "50% HDR"),
    0.80: ("dashed", "80% HDR"),
    0.95: ("dotted", "95% HDR"),
}
_MAX_OVERLAY_DOTS = 600


def _space_shared_x_ticks(figure: Any, axes: Sequence[Any]) -> None:
    """Keep shared scales and fonts while reserving a four-point label gutter."""
    if not axes:
        return
    limits = axes[0].get_xlim()
    ticks = [float(t) for t in axes[0].get_xticks() if limits[0] <= t <= limits[1]]
    while True:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        crowded = False
        for axis in axes:
            labels = [
                text
                for text in axis.get_xticklabels()
                if text.get_visible() and text.get_text()
            ]
            boxes = sorted(
                (text.get_window_extent(renderer) for text in labels),
                key=lambda box: box.x0,
            )
            crowded |= any(
                (right.x0 - left.x1) * 72.0 / figure.dpi < 4.0
                for left, right in zip(boxes, boxes[1:], strict=False)
            )
        if not crowded:
            return
        # The origin and last in-range tick retain two quantitative anchors
        # when the former interior ticks cannot fit in the narrow marginal.
        reduced = ticks[::2] if len(ticks) > 2 else [float(limits[0]), ticks[-1]]
        if reduced == ticks:
            raise ValueError("shared tick labels cannot fit with a readable gutter")
        ticks = reduced
        for axis in axes:
            axis.set_xticks(ticks)
            axis.set_xlim(limits)


def _space_main_marginal_boundary(
    figure: Any, main_axes: Sequence[Any], right_axes: Sequence[Any]
) -> None:
    """Prune a crowded final main tick without moving data or density axes."""
    if not main_axes:
        return
    limits = main_axes[0].get_xlim()
    ticks = [float(t) for t in main_axes[0].get_xticks()]
    while True:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        crowded = False
        for main, right in zip(main_axes, right_axes, strict=True):
            main_boxes = [
                t.get_window_extent(renderer)
                for t in main.get_xticklabels()
                if t.get_visible() and t.get_text()
            ]
            right_boxes = [
                t.get_window_extent(renderer)
                for t in right.get_xticklabels()
                if t.get_visible() and t.get_text()
            ]
            if main_boxes and right_boxes:
                gap = min(b.x0 for b in right_boxes) - max(b.x1 for b in main_boxes)
                crowded |= gap * 72.0 / figure.dpi < 4.0
        if not crowded:
            return
        retained = ticks[:-1]
        if sum(limits[0] <= tick <= limits[1] for tick in retained) < 2:
            raise ValueError("main and shared tick labels cannot fit readably")
        ticks = retained
        for main in main_axes:
            main.set_xticks(ticks)
            main.set_xlim(limits)


def _validate_metadata(metadata: Mapping[str, Any]) -> str:
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping of scenario inputs")
    if not all(isinstance(key, str) and key.strip() for key in metadata):
        raise ValueError("metadata keys must be non-empty strings")
    social_club = metadata.get("social_club")
    if social_club is not None and (
        not isinstance(social_club, str) or not social_club.strip()
    ):
        raise ValueError("social_club must be a non-empty display name")
    values: list[str] = []
    for key, value in metadata.items():
        if key in ("social_club", "valid_shots_by_pattern", "total_shots_by_pattern"):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metadata value for {key!r} must be finite")
        values.append(f"{key.replace('_', ' ')}: {value}")
    return textwrap.fill("  |  ".join(values), width=145, break_long_words=False)


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


def _validate_plot_envelope(scenario: str, dpi: int, presentation: str) -> None:
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("scenario must be a non-empty string")
    if not isinstance(dpi, int) or not 72 <= dpi <= 600:
        raise ValueError("dpi must be an integer in [72, 600]")
    if presentation not in ("academic", "social"):
        raise ValueError("presentation must be academic or social")


def write_academic_dispersion_figure(
    points_by_pattern: Mapping[str, object],
    output_stem: Path | str,
    *,
    scenario: str,
    metadata: Mapping[str, Any],
    target_point_m: tuple[float, float],
    bins: int | tuple[int, int] = 64,
    region_masses: Sequence[float] = (0.50, 0.80, 0.95),
    presentation: Literal["academic", "social"] = "academic",
    dpi: int = 300,
) -> dict[str, Path]:
    """Export raw endpoint density with marginals and empirical HDR contours.

    The coordinates are plotted exactly as supplied. The target marker and
    common bounds are explicit; no aim correction or translation is applied.
    Each pattern uses the same histogram edges and shared density color scale.
    The marginal curves are normalized histogram densities, not KDE curves.
    Histograms, marginals, and HDR regions use every supported endpoint; the
    colored dot overlay shows at most 600 deterministic samples and discloses
    that count. ``social`` changes only layout and caption density.

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
    _validate_plot_envelope(scenario, dpi, presentation)
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
    overlay_counts = {
        name: min(actual_valid_counts[name], _MAX_OVERLAY_DOTS) for name in names
    }
    plt = _matplotlib()
    from matplotlib import ticker
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D

    is_social = presentation == "social"
    num_panels = len(names)

    if is_social:
        fig_width = 10.8
        if num_panels == 2:
            fig_height = 13.5  # 4:5 Facebook portrait post aspect ratio (1080 x 1350)
        elif num_panels == 3:
            fig_height = 18.0  # 3:5 mobile feed portrait ratio (1080 x 1800)
        elif num_panels == 1:
            fig_height = 10.8  # 1:1 square
        else:
            fig_height = 5.0 + 4.5 * num_panels
        figure = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    else:
        figure = plt.figure(figsize=(16.5, 9.8), facecolor="white")

    FigureCanvasAgg(figure)

    hdr_handles: list[Line2D] = []
    for mass in region_masses:
        mass_value = float(mass)
        style, label = _HDR_STYLES.get(mass_value, ("solid", f"{mass_value:.0%} HDR"))
        style = cast(_LineStyle, style)
        hdr_handles.append(
            Line2D(
                [0],
                [0],
                color="#222222",
                linestyle=style,
                linewidth=1.4 if is_social else 1.2,
                label=label,
            )
        )
    hdr_handles.append(
        Line2D(
            [0],
            [0],
            color="#111111",
            marker="+",
            linestyle="none",
            markersize=7 if is_social else 6,
            label="Target",
        )
    )

    achieved_text = (
        "Achieved HDR Mass By Pattern (Discrete Ties May Exceed Request): "
        + "  |  ".join(
            f"{name} "
            + "/".join(
                f"{estimate.patterns[name].hdr_achieved_mass[float(mass)]:.1%}"
                for mass in region_masses
            )
            for name in names
        )
    )
    dx, dy = estimate.bin_resolution_m
    overlay_caption = "Colored Dots Shown/Valid: " + "; ".join(
        f"{name} {overlay_counts[name]:,}/{actual_valid_counts[name]:,}"
        for name in names
    )
    if is_social:
        footer = (
            "Model-Conditional Simulation • Carry Only • Not Calibrated Player Precision. "
            f"{overlay_caption}. Density, HDR, and marginals use every valid endpoint. "
            "Shared axes and density scale; contours are simulated-sample HDR."
        )
    else:
        footer = (
            f"{_PILOT_FIGURE_LABEL}. HDR mass is sample-derived from simulated endpoints. "
            f"{overlay_caption}; density, HDR, and marginals use every valid endpoint. "
            "Histogram only; no KDE or synthetic covariance spread; not calibrated player precision. "
            f"Bin resolution: {dx:.3g} × {dy:.3g} m. {achieved_text}"
        )

    if is_social:
        # Suptitle
        title_text = f"{metadata.get('social_club', 'Shot')}: Shot Dispersion"
        t_title = figure.text(
            0.08, 0.985, title_text, ha="left", va="top", fontsize=18, fontweight="bold"
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        y_title_bottom = t_title.get_window_extent(renderer).y0 / figure.bbox.height

        # Scenario
        wrapped_scenario = textwrap.fill(scenario, width=70, break_long_words=False)
        t_scenario = figure.text(
            0.08,
            y_title_bottom - 0.010,
            wrapped_scenario,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="top",
        )
        figure.canvas.draw()
        y_scen_bottom = t_scenario.get_window_extent(renderer).y0 / figure.bbox.height

        # Metadata caption with complete population disclosures
        wrapped_metadata = textwrap.fill(
            f"{metadata_caption}  |  {population}" if metadata_caption else population,
            width=78,
            break_long_words=False,
        )
        t_metadata = figure.text(
            0.08,
            y_scen_bottom - 0.008,
            wrapped_metadata,
            fontsize=10.5,
            ha="left",
            va="top",
            color="#333333",
        )
        figure.canvas.draw()
        y_meta_bottom = t_metadata.get_window_extent(renderer).y0 / figure.bbox.height

        # Legend
        legend_top = y_meta_bottom - 0.012
        legend = figure.legend(
            handles=hdr_handles,
            loc="upper center",
            bbox_to_anchor=(0.48, legend_top),
            ncol=len(hdr_handles),
            frameon=False,
            fontsize=10,
            title="Simulated-Sample Highest-Density Regions",
            title_fontsize=10.5,
        )
        figure.canvas.draw()
        y_leg_bottom = legend.get_window_extent(renderer).y0 / figure.bbox.height

        # Footer
        wrapped_footer = textwrap.fill(footer, width=95, break_long_words=False)
        t_footer = figure.text(
            0.08, 0.018, wrapped_footer, fontsize=9.5, ha="left", va="bottom"
        )
        figure.canvas.draw()
        y_foot_top = t_footer.get_window_extent(renderer).y1 / figure.bbox.height

        top_grid = y_leg_bottom - 0.016
        bottom_grid = y_foot_top + 0.028
        outer = figure.add_gridspec(
            num_panels,
            1,
            left=0.10,
            right=0.86,
            top=top_grid,
            bottom=bottom_grid,
            hspace=0.36 if num_panels == 2 else 0.32,
        )
    else:
        outer = figure.add_gridspec(
            1, len(names), left=0.06, right=0.91, top=0.69, bottom=0.17, wspace=0.27
        )
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
            textwrap.fill(
                f"{metadata_caption}  |  {population}"
                if metadata_caption
                else population,
                width=145,
                break_long_words=False,
            ),
            fontsize=9,
            ha="left",
            va="top",
            color="#333333",
        )
        figure.legend(
            handles=hdr_handles,
            loc="upper center",
            bbox_to_anchor=(0.48, 0.775),
            ncol=len(hdr_handles),
            frameon=False,
            title="Simulated-Sample Highest-Density Regions",
        )
        figure.text(
            0.06,
            0.025,
            textwrap.fill(footer, width=210, break_long_words=False),
            fontsize=8.5,
            ha="left",
        )

    top_axes: list[Any] = []
    right_axes: list[Any] = []
    main_axes: list[Any] = []
    meshes: list[Any] = []
    common_top_max = 0.0
    common_right_max = 0.0

    for name, panel_index in zip(names, range(len(names)), strict=True):
        grid_cell = outer[panel_index, 0] if is_social else outer[0, panel_index]
        inner = grid_cell.subgridspec(
            2, 2, height_ratios=(1, 4), width_ratios=(4, 1), hspace=0.06, wspace=0.06
        )
        axis_top = figure.add_subplot(inner[0, 0])
        axis = figure.add_subplot(inner[1, 0], sharex=axis_top)
        axis_right = figure.add_subplot(inner[1, 1], sharey=axis)
        main_axes.append(axis)
        top_axes.append(axis_top)
        right_axes.append(axis_right)

        pattern = estimate.patterns[name]
        density = pattern.density_per_m2
        marginal_x, marginal_y = _density_marginals(estimate, density)
        common_top_max = max(common_top_max, float(np.max(marginal_x)))
        common_right_max = max(common_right_max, float(np.max(marginal_y)))
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
        meshes.append(mesh)

        sample = np.asarray(points_by_pattern[name], dtype=float)
        overlay_indices = np.linspace(
            0, len(sample) - 1, overlay_counts[name], dtype=int
        )
        axis.scatter(
            sample[overlay_indices, 0],
            sample[overlay_indices, 1],
            s=6 if is_social else 5,
            color=color,
            alpha=0.22,
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
            style = cast(_LineStyle, style)
            axis.contour(
                _axis_centers(estimate.x_edges_m),
                _axis_centers(estimate.y_edges_m),
                density.T,
                levels=[threshold],
                colors=["#222222"],
                linewidths=1.2 if is_social else 1.0,
                linestyles=[style],
                zorder=4,
            )

        axis.scatter(
            [target_point_m[0]],
            [target_point_m[1]],
            marker="+",
            s=40 if is_social else 35,
            color="#111111",
            linewidths=1.2 if is_social else 1.0,
            zorder=5,
            label="Target",
        )
        axis.axvline(target_point_m[0], color="#333333", linewidth=0.55, alpha=0.55)
        axis.axhline(target_point_m[1], color="#333333", linewidth=0.55, alpha=0.55)
        axis.set_xlim(estimate.x_limits_m)
        axis.set_ylim(estimate.y_limits_m)

        axis_top.set_title(
            f"{name}  (n={pattern.n:,})",
            fontsize=12.5 if is_social else 12,
            fontweight="bold",
            pad=5,
        )
        axis.set_xlabel("Downrange Endpoint (m)", fontsize=11 if is_social else 10)
        if is_social or panel_index == 0:
            axis.set_ylabel(
                "Lateral Endpoint (m; Right Positive)",
                fontsize=11 if is_social else 10,
            )
        axis.grid(color="#666666", linewidth=0.35, alpha=0.25)
        axis.set_facecolor("white")
        axis.tick_params(axis="both", labelsize=10 if is_social else 8, length=3)
        axis.xaxis.set_major_locator(ticker.MaxNLocator(prune="upper"))

        axis_top.stairs(
            marginal_x,
            estimate.x_edges_m,
            color=color,
            linewidth=1.3 if is_social else 1.1,
        )
        axis_top.set_xlim(estimate.x_limits_m)
        axis_top.tick_params(axis="x", labelbottom=False, bottom=False)
        axis_top.tick_params(axis="y", labelsize=9 if is_social else 7, length=2)
        axis_top.set_ylabel("Density", fontsize=9.5 if is_social else 8)
        axis_top.grid(axis="y", linewidth=0.3, alpha=0.25)
        axis_top.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune="lower"))

        axis_right.stairs(
            marginal_y,
            estimate.y_edges_m,
            orientation="horizontal",
            color=color,
            linewidth=1.3 if is_social else 1.1,
        )
        axis_right.set_ylim(estimate.y_limits_m)
        axis_right.tick_params(axis="y", labelleft=False, left=False)
        axis_right.tick_params(axis="x", labelsize=9 if is_social else 7, length=2)
        axis_right.set_xlabel("Density", fontsize=9.5 if is_social else 8)
        axis_right.grid(axis="x", linewidth=0.3, alpha=0.25)
        axis_right.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune="lower"))

    if common_top_max <= 0 or common_right_max <= 0:
        raise ValueError("marginal density maxima must be positive")
    for axis_top, axis_right in zip(top_axes, right_axes, strict=True):
        axis_top.set_ylim(0.0, common_top_max * 1.05)
        axis_right.set_xlim(0.0, common_right_max * 1.05)

    _space_shared_x_ticks(figure, right_axes)
    _space_shared_x_ticks(figure, main_axes)
    _space_main_marginal_boundary(figure, main_axes, right_axes)

    if is_social:
        # Reserve the actual tick/label footprint, not only the axes rectangle.
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        decorated_bottom = min(axis.get_tightbbox(renderer).y0 for axis in figure.axes)
        footer_top = t_footer.get_window_extent(renderer).y1
        clearance = footer_top + 6.0 * figure.dpi / 72.0 - decorated_bottom
        if clearance > 0:
            bottom_grid += clearance / figure.bbox.height
            if bottom_grid >= top_grid:
                raise ValueError("social annotations leave no room for density panels")
            outer.update(bottom=bottom_grid)
        colorbar_axis = figure.add_axes(
            (0.89, bottom_grid, 0.022, top_grid - bottom_grid)
        )
        colorbar = figure.colorbar(meshes[0], cax=colorbar_axis)
        colorbar.set_label("Endpoint Probability Density (m⁻²)", fontsize=10.5)
        colorbar.ax.tick_params(labelsize=9)
    else:
        colorbar_axis = figure.add_axes((0.93, 0.28, 0.014, 0.48))
        colorbar = figure.colorbar(meshes[0], cax=colorbar_axis)
        colorbar.set_label("Endpoint Probability Density (m⁻²)")

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
    presentation: Literal["academic", "social"] = "academic",
    dpi: int = 300,
) -> dict[str, Path]:
    """Export a shared-axis comparison of supplied raw overhead trajectories.

    Trajectories are plotted exactly as supplied in downrange/right-positive
    lateral coordinates. They are not rotated to an aim line or re-simulated.
    Set ``trajectories_are_aimed`` only to trigger a fail-fast contract error;
    this figure is explicitly for raw, un-aimed simulator trajectories.
    The displayed path count and calculated vertical display magnification are
    always disclosed. ``social`` changes only layout and caption density.
    """
    if trajectories_are_aimed:
        raise ValueError("overhead trajectory input must be raw, not aim-adjusted")
    if not isinstance(target_x_m, (int, float)) or not math.isfinite(target_x_m):
        raise ValueError("target_x_m must be finite")
    _validate_plot_envelope(scenario, dpi, presentation)
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
    population = (
        _population_caption(names_tuple, valid_counts, total_counts)
        if presentation == "social" and valid_counts
        else _trajectory_population_caption(
            names_tuple, path_counts, valid_counts, total_counts
        )
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
        figsize=(16.0, 9.0) if presentation == "social" else (6.0 * len(names), 6.0),
        sharex=True,
        sharey=True,
        squeeze=False,
        facecolor="white",
    )
    figure.subplots_adjust(
        top=0.69 if presentation == "social" else 0.70,
        bottom=0.16,
        left=0.075,
        right=0.98,
        wspace=0.15,
    )
    for axis, name in zip(axes[0], names, strict=True):
        color = _COLORS.get(name, "#286090")
        for trajectory in trajectories[name]:
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=color,
                alpha=0.72,
                linewidth=1.3,
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
            (
                name
                if presentation == "social"
                else f"{name}  (paths shown={path_counts[name]:,}{valid_text}{total_text})"
            ),
            fontsize=11,
            fontweight="bold",
        )
        axis.set_xlabel("Downrange Position (m)")
        axis.grid(color="#666666", linewidth=0.35, alpha=0.25)
    axes[0][0].set_ylabel("Lateral Position (m; Right Positive)")
    figure.suptitle(
        (
            f"{metadata.get('social_club', 'Shot')}: Sample Flight Paths"
            if presentation == "social"
            else "Raw Overhead Flight-Trajectory Comparison"
        ),
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.06,
        0.925,
        scenario,
        fontsize=14 if presentation == "social" else 12,
        ha="left",
        va="top",
    )
    figure.text(
        0.06,
        0.89,
        textwrap.fill(
            f"{metadata_caption}  |  {population}" if metadata_caption else population,
            width=145,
            break_long_words=False,
        ),
        fontsize=10 if presentation == "social" else 9,
        ha="left",
        va="top",
        color="#333333",
    )
    figure.canvas.draw()
    axis_for_scale = axes[0][0]
    origin_px = axis_for_scale.transData.transform((0.0, 0.0))
    x_unit_px = axis_for_scale.transData.transform((1.0, 0.0))
    y_unit_px = axis_for_scale.transData.transform((0.0, 1.0))
    vertical_scale_ratio = (y_unit_px[1] - origin_px[1]) / (x_unit_px[0] - origin_px[0])
    shown_paths_note = (
        "Paths Shown: "
        + "; ".join(f"{name} {path_counts[name]}" for name in names)
        + ". Samples from supported airborne shots."
        if presentation == "social"
        else "Only the displayed paths are plotted; paths are sampled from supported airborne shots."
    )
    overhead_footer = (
        "Model-Conditional Simulation • Carry Only • Not Calibrated Player Precision. "
        if presentation == "social"
        else f"{_PILOT_FIGURE_LABEL}. "
    ) + (
        "Raw trajectories; no aim-line rotation. "
        f"Vertical display magnification: {vertical_scale_ratio:.1f}× versus downrange. "
        + shown_paths_note
    )
    figure.text(
        0.06,
        0.025,
        textwrap.fill(overhead_footer, width=180, break_long_words=False),
        fontsize=10 if presentation == "social" else 9,
        ha="left",
    )
    outputs = _save_formats(figure, _validate_stem(output_stem), dpi=dpi)
    plt.close(figure)
    return outputs


__all__ = ["write_academic_dispersion_figure", "write_overhead_trajectory_figure"]
