"""Academic heatmaps for selected matched-delivery matrix conditions.

Plotting is read-only: callers provide already validated completed summaries.
Unavailable and not-run conditions stay visible in gray hatched matrix cells.
"""

from __future__ import annotations

import math
import re
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .noise import NoiseSpec

PATTERNS = ("Straight", "Draw", "Fade")
CLUBS = ("driver", "seven_iron", "pitching_wedge")
_METRICS = {
    "target_rmse_m": ("Raw-Endpoint Target RMSE (m)", "viridis", "RMSE (m)"),
    "target_hit_fraction": ("Target Containment Fraction", "Greens", "Fraction"),
    "supported_fraction": ("Supported Airborne Shot Fraction", "Greens", "Fraction"),
    "radial_p95_m": ("Radial Target-Error P95 (m)", "viridis", "P95 Error (m)"),
    "conditional_sg": (
        "Conditional Mean Strokes Gained by Pattern (Strokes)",
        "coolwarm",
        "Mean Strokes Gained (Strokes)",
    ),
}
_CLUB_LABELS = {
    "driver": "Driver",
    "seven_iron": "7-Iron",
    "pitching_wedge": "Pitching Wedge",
}


def _config_value(cell: Mapping[str, Any], key: str) -> Any:
    config = cell.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("each matrix cell needs a config object")
    if key in config:
        return config[key]
    club = config.get("club")
    if isinstance(club, Mapping) and key in club:
        return club[key]
    return None


def metric_value(cell: Mapping[str, Any], metric: str, pattern: str) -> float | None:
    """Return one report metric, or None when the declared result is unavailable."""
    if metric not in _METRICS:
        raise ValueError(f"unknown matrix metric: {metric}")
    if pattern not in PATTERNS:
        raise ValueError(f"unknown landing pattern: {pattern}")
    if cell.get("status") != "completed":
        return None
    summary = cell.get("summary")
    if not isinstance(summary, Mapping):
        return None
    pattern_stats = summary.get(pattern)
    if not isinstance(pattern_stats, Mapping):
        return None
    if metric == "target_rmse_m":
        value = pattern_stats.get("target_rmse_m")
    elif metric == "target_hit_fraction":
        value = pattern_stats.get("target_containment_fraction")
    elif metric == "supported_fraction":
        value = pattern_stats.get("supported_fraction")
    elif metric == "radial_p95_m":
        stats = pattern_stats.get("statistics")
        value = (
            stats.get("p95_radial_target_error_m")
            if isinstance(stats, Mapping)
            else None
        )
    else:
        scoring = cell.get("course_scoring")
        if not isinstance(scoring, Mapping) or scoring.get("status") not in (
            "available",
            "conditional_only",
        ):
            return None
        scores = scoring.get("patterns")
        item = scores.get(pattern) if isinstance(scores, Mapping) else None
        value = item.get("mean_strokes_gained") if isinstance(item, Mapping) else None
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"matrix metric {metric} for {pattern} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"matrix metric {metric} for {pattern} must be finite")
    return float(value)


def _validate_cells(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], tuple[float, ...], tuple[float, ...]]:
    if not cells:
        raise ValueError("selected matrix cells must not be empty")
    indexes: dict[tuple[str, float, float], Mapping[str, Any]] = {}
    leans: set[float] = set()
    curves: set[float] = set()
    laws: set[str] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("matrix cells must be mappings")
        name = cell.get("scenario")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("each matrix cell needs a scenario name")
        status = cell.get("status")
        if status not in ("completed", "unavailable", "not_run"):
            raise ValueError(f"invalid cell status for {name}: {status}")
        club = _config_value(cell, "club_id")
        if club not in CLUBS:
            raise ValueError(f"unknown or missing club for matrix cell {name}")
        lean = _config_value(cell, "shaft_lean_deg")
        curve = _config_value(cell, "curve_scale")
        if (
            isinstance(lean, bool)
            or not isinstance(lean, (int, float))
            or not math.isfinite(lean)
            or isinstance(curve, bool)
            or not isinstance(curve, (int, float))
            or not math.isfinite(curve)
        ):
            raise ValueError(f"invalid lean/curve matrix coordinates for {name}")
        key = (club, float(lean), float(curve))
        if key in indexes:
            raise ValueError(f"duplicate matrix cell: {name}")
        indexes[key] = cell
        leans.add(float(lean))
        curves.add(float(curve))
        law = _config_value(cell, "contact_law")
        if law is not None:
            if not isinstance(law, str) or not law:
                raise ValueError(f"invalid contact law in matrix cell {name}")
            laws.add(law)
    if len(laws) > 1:
        raise ValueError("selected cells must use one contact law; facet by law")
    expected = {
        (club, lean, curve) for club in CLUBS for lean in leans for curve in curves
    }
    if set(indexes) != expected:
        raise ValueError(
            "incomplete matrix axes: include explicit unavailable/not-run cells"
        )
    return (
        [indexes[key] for key in sorted(indexes)],
        tuple(sorted(leans)),
        tuple(sorted(curves)),
    )


def _format_metric(metric: str, value: float, stats: Mapping[str, Any]) -> str:
    supported = stats.get("supported_n")
    total = stats.get("total_n")
    count = (
        f"{supported}/{total}"
        if isinstance(supported, int) and isinstance(total, int)
        else ""
    )
    if metric in ("target_hit_fraction", "supported_fraction"):
        return f"{value:.0%}" + (f"\nn={count}" if count else "")
    if metric == "conditional_sg":
        marker = (
            "conditional "
            if supported is not None and total is not None and supported < total
            else ""
        )
        return f"{value:+.2f}" + (f"\n{marker}n={count}" if count else "")
    return f"{value:.2f}" + (f"\nn={count}" if count else "")


def _format_cell_metric(
    cell: Mapping[str, Any], metric: str, pattern: str, value: float
) -> str:
    """Format one annotation with the population used by that metric."""
    summary = cell.get("summary")
    pattern_stats = summary.get(pattern) if isinstance(summary, Mapping) else None
    if not isinstance(pattern_stats, Mapping):
        pattern_stats = {}
    if metric != "conditional_sg":
        return _format_metric(metric, value, pattern_stats)

    scoring = cell.get("course_scoring")
    if not isinstance(scoring, Mapping):
        return _format_metric(metric, value, {})
    paired_n = scoring.get("paired_n")
    total_n = scoring.get("total_n")
    if (
        isinstance(paired_n, int)
        and not isinstance(paired_n, bool)
        and isinstance(total_n, int)
        and not isinstance(total_n, bool)
    ):
        return _format_metric(
            metric,
            value,
            {"supported_n": paired_n, "total_n": total_n},
        )
    return _format_metric(metric, value, {})


def _cell_text_color(cmap: Any, norm: Any, value: float | None) -> str:
    """Choose black or white using the higher linear-sRGB contrast ratio."""
    if value is None or norm is None:
        return "#111827"
    try:
        norm_val = norm(value)
        norm_float = float(np.clip(float(norm_val), 0.0, 1.0))
        rgba = cmap(norm_float)
        r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
        linear = [
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in (r, g, b)
        ]
        luminance = sum(
            channel * weight
            for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True)
        )
        white_contrast = 1.05 / (luminance + 0.05)
        black_contrast = (luminance + 0.05) / 0.05
        return "#ffffff" if white_contrast >= black_contrast else "black"
    except (TypeError, ValueError, AttributeError):
        return "#111827"


def _extract_noise_spec(cell: Mapping[str, Any]) -> NoiseSpec | None:
    config = cell.get("config")
    noise = None
    if isinstance(config, Mapping) and "noise" in config:
        noise = config["noise"]
    elif "noise" in cell:
        noise = cell["noise"]
    if isinstance(noise, NoiseSpec):
        return noise
    if isinstance(noise, Mapping):
        sds = noise.get("standard_deviations")
        corr = noise.get("correlation")
        if isinstance(sds, Sequence) and not isinstance(sds, (str, bytes)):
            try:
                corr_tuple = (
                    tuple(tuple(float(v) for v in row) for row in corr)
                    if isinstance(corr, Sequence) and not isinstance(corr, (str, bytes))
                    else None
                )
                return NoiseSpec(tuple(float(v) for v in sds), corr_tuple)
            except (ValueError, TypeError):
                return None
    return None


def _derive_delivery_noise(cells: Sequence[Mapping[str, Any]]) -> str | None:
    """Format readable primitive SDs and correlation; correlation is shaft-twist/path."""
    specs = [_extract_noise_spec(cell) for cell in cells]
    valid_specs = [item for item in specs if item is not None]
    if not valid_specs:
        return None
    spec = valid_specs[0]
    if len(valid_specs) != len(specs) or any(
        item.standard_deviations != spec.standard_deviations
        or not np.array_equal(item.correlation_matrix(), spec.correlation_matrix())
        for item in valid_specs[1:]
    ):
        raise ValueError(
            "selected matrix cells must use one noise profile; facet by noise"
        )

    sd = spec.standard_deviations
    parts: list[str] = []
    if sd[0]:
        parts.append(f"Shaft Twist SD {sd[0]:g}°")
    if sd[1]:
        parts.append(f"Path SD {sd[1]:g}°")
    if sd[2]:
        parts.append(f"Speed SD {sd[2] / 0.44704:g} mph")
    if sd[3] == sd[4] and sd[3]:
        parts.append(f"Strike SD {sd[3] / 0.0254:g} in on Each Axis")
    else:
        for label, val in (("Horizontal", sd[3]), ("Vertical", sd[4])):
            if val:
                parts.append(f"{label} Strike SD {val / 0.0254:g} in")
    for label, val in (("Shaft Lean", sd[5]), ("Attack Angle", sd[6])):
        if val:
            parts.append(f"{label} SD {val:g}°")

    if not parts:
        parts.append("Zero Primitive Noise")

    names = (
        "Shaft Twist",
        "Path",
        "Speed",
        "Horizontal Strike",
        "Vertical Strike",
        "Shaft Lean",
        "Attack Angle",
    )
    correlation = spec.correlation_matrix()
    corr_parts: list[str] = []
    if sd[0] and sd[1] and abs(correlation[0, 1]) > 1e-6:
        corr_parts.append(f"Shaft Twist–Path Correlation r = {correlation[0, 1]:+g}")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if (i, j) == (0, 1):
                continue
            if sd[i] and sd[j] and abs(correlation[i, j]) > 1e-6:
                corr_parts.append(
                    f"{names[i]}–{names[j]} Correlation r = {correlation[i, j]:+g}"
                )

    if corr_parts:
        return f"{', '.join(parts)} · {', '.join(corr_parts)}"
    active_sds = sum(1 for v in sd if v > 0)
    if active_sds > 1:
        return f"{', '.join(parts)} · Independent Delivery"
    return ", ".join(parts)


def _containment_radius_caption(cells: Sequence[Mapping[str, Any]]) -> str:
    """Disclose the endpoint success geometry separately from course scoring."""
    radii = [_config_value(cell, "target_radius_m") for cell in cells]
    if all(radius is None for radius in radii):
        return "Circular Target Radius: Unspecified"
    if (
        any(
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not math.isfinite(radius)
            or radius <= 0
            for radius in radii
        )
        or len(set(radii)) != 1
    ):
        raise ValueError("selected cells must use one circular target radius")
    return f"Circular Target Radius: {radii[0]:g} m"


def _derive_target_context(cells: Sequence[Mapping[str, Any]]) -> str | None:
    """Identify within-lean nominal targets versus fixed-target control from cell config."""
    has_target_spec = False
    fixed_targets: set[float] = set()
    for cell in cells:
        config = cell.get("config")
        target = None
        found = False
        if isinstance(config, Mapping) and "target_x_m" in config:
            target = config["target_x_m"]
            found = True
        elif "target_x_m" in cell:
            target = cell["target_x_m"]
            found = True
        if found:
            has_target_spec = True
            if (
                isinstance(target, (int, float))
                and not isinstance(target, bool)
                and math.isfinite(target)
            ):
                fixed_targets.add(float(target))

    if not has_target_spec:
        return None
    if fixed_targets:
        if len(fixed_targets) == 1:
            val = next(iter(fixed_targets))
            return f"Fixed-Target Control ({val:g} m)"
        return "Fixed-Target Control"
    return "Within-Lean Nominal Targets"


def _derive_course_context(cells: Sequence[Mapping[str, Any]], club: str) -> str | None:
    """Identify course scoring context for a club from actual course_scoring metadata."""
    for cell in cells:
        club_id = _config_value(cell, "club_id")
        if club_id != club:
            continue
        scoring = cell.get("course_scoring")
        if not isinstance(scoring, Mapping):
            continue
        start_lie = scoring.get("start_lie")
        if not isinstance(start_lie, str):
            continue
        hole_dist = scoring.get("hole_distance_m")
        if (
            hole_dist is None
            or isinstance(hole_dist, bool)
            or not isinstance(hole_dist, (int, float))
            or not math.isfinite(hole_dist)
        ):
            hole_dist = scoring.get("start_distance_m")
        fw_half = scoring.get("fairway_half_width_m")
        green_r = scoring.get("green_radius_m")

        if (
            start_lie.lower() == "tee"
            and fw_half is not None
            and isinstance(fw_half, (int, float))
            and not isinstance(fw_half, bool)
            and hole_dist is not None
            and isinstance(hole_dist, (int, float))
            and not isinstance(hole_dist, bool)
        ):
            return f"Tee, {float(hole_dist):g} m, {float(fw_half) * 2:g} m Fairway"
        if (
            start_lie.lower() == "fairway"
            and green_r is not None
            and isinstance(green_r, (int, float))
            and not isinstance(green_r, bool)
        ):
            return f"Fairway Approach, {float(green_r):g} m Green Radius"
        if (
            green_r is not None
            and isinstance(green_r, (int, float))
            and not isinstance(green_r, bool)
        ):
            return f"{start_lie.title()} Approach, {float(green_r):g} m Green Radius"
        if (
            hole_dist is not None
            and isinstance(hole_dist, (int, float))
            and not isinstance(hole_dist, bool)
        ):
            return f"{start_lie.title()}, {float(hole_dist):g} m"
    return None


def _has_carry_as_final_lie(cells: Sequence[Mapping[str, Any]]) -> bool:
    """Check if course scoring uses carry-as-final-lie approximation."""
    for cell in cells:
        scoring = cell.get("course_scoring")
        if not isinstance(scoring, Mapping):
            continue
        limitations = scoring.get("limitations")
        if isinstance(limitations, Sequence) and not isinstance(
            limitations, (str, bytes)
        ):
            for lim in limitations:
                if isinstance(lim, str) and "final lie" in lim.lower():
                    return True
        scenario_id = scoring.get("scenario_id")
        if scenario_id in ("tee_fairway_rough", "approach_green_rough"):
            return True
    return False


def _draw_metric(
    cells: Sequence[Mapping[str, Any]],
    leans: tuple[float, ...],
    curves: tuple[float, ...],
    metric: str,
    selection_label: str,
    output_stem: Path,
    *,
    dpi: int,
    formats: Sequence[str],
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    target_context = _derive_target_context(cells)
    derived_noise = _derive_delivery_noise(cells)
    carry_final_lie = _has_carry_as_final_lie(cells)
    course_contexts = {club: _derive_course_context(cells, club) for club in CLUBS}

    by_key = {
        (
            _config_value(cell, "club_id"),
            float(_config_value(cell, "shaft_lean_deg")),
            float(_config_value(cell, "curve_scale")),
        ): cell
        for cell in cells
    }
    values = [
        value
        for pattern in PATTERNS
        for cell in cells
        if (value := metric_value(cell, metric, pattern)) is not None
    ]
    title, cmap_name, colorbar_label = _METRICS[metric]
    cmap = plt.get_cmap(cmap_name).with_extremes(bad="#e5e7eb")
    norm: Any = None
    if values:
        if metric == "conditional_sg":
            extent = max(abs(min(values)), abs(max(values)), 1e-9)
            norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)
        elif metric in ("target_hit_fraction", "supported_fraction"):
            norm = plt.Normalize(vmin=0.0, vmax=1.0)
        else:
            low, high = min(values), max(values)
            if math.isclose(low, high):
                high = low + max(abs(low) * 0.05, 1e-6)
            norm = plt.Normalize(vmin=low, vmax=high)

    fig, axes = plt.subplots(3, 3, figsize=(16.5, 10.5), facecolor="white")
    images = []
    for row_index, pattern in enumerate(PATTERNS):
        for col_index, club in enumerate(CLUBS):
            axis = axes[row_index, col_index]
            grid = np.full((len(leans), len(curves)), np.nan, dtype=float)
            for i, lean in enumerate(leans):
                for j, curve in enumerate(curves):
                    cell = by_key[(club, lean, curve)]
                    grid[i, j] = (
                        np.nan
                        if (value := metric_value(cell, metric, pattern)) is None
                        else value
                    )
            masked: NDArray[np.float64] = np.ma.masked_invalid(grid)
            if values and norm is not None:
                image = axis.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
                images.append(image)
            else:
                axis.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
                axis.text(
                    0.5,
                    1.08,
                    "No Available Values",
                    transform=axis.transAxes,
                    ha="center",
                    fontsize=8,
                    color="#555555",
                )
            axis.set_xticks(range(len(curves)), [f"×{curve:g}" for curve in curves])
            axis.set_yticks(range(len(leans)), [f"{lean:g}°" for lean in leans])
            if col_index == 0:
                axis.set_ylabel(f"{pattern}\nShaft Lean")
            if row_index == len(PATTERNS) - 1:
                axis.set_xlabel("Curve Scale")

            club_ctx = course_contexts.get(club) if metric == "conditional_sg" else None
            title_text = (
                f"{_CLUB_LABELS[club]} — {pattern}\n{textwrap.fill(club_ctx, width=44)}"
                if club_ctx
                else f"{_CLUB_LABELS[club]} — {pattern}"
            )
            axis.set_title(
                title_text,
                fontsize=10 if club_ctx else 11,
                fontweight="bold",
            )
            axis.set_xticks(np.arange(-0.5, len(curves), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(leans), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=2)
            axis.tick_params(which="minor", bottom=False, left=False)
            for i, lean in enumerate(leans):
                for j, curve in enumerate(curves):
                    cell = by_key[(club, lean, curve)]
                    value = metric_value(cell, metric, pattern)
                    if value is None:
                        axis.add_patch(
                            Rectangle(
                                (j - 0.5, i - 0.5),
                                1,
                                1,
                                facecolor="#e5e7eb",
                                edgecolor="#777777",
                                linewidth=0.7,
                                hatch="////",
                                zorder=3,
                            )
                        )
                        reason = cell.get("reason") or "metric unavailable"
                        label = (
                            "NOT RUN"
                            if cell.get("status") == "not_run"
                            else "UNAVAILABLE"
                        )
                        annotation = f"{label}\n{textwrap.fill(str(reason), width=30)}"
                        axis.text(
                            j,
                            i,
                            annotation,
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="#111827",
                            zorder=4,
                        )
                    else:
                        annotation = _format_cell_metric(cell, metric, pattern, value)
                        text_color = _cell_text_color(cmap, norm, value)
                        axis.text(
                            j,
                            i,
                            annotation,
                            ha="center",
                            va="center",
                            fontsize=8,
                            color=text_color,
                            zorder=4,
                        )
            axis.set_xlim(-0.5, len(curves) - 0.5)
            axis.set_ylim(len(leans) - 0.5, -0.5)

    fig.suptitle(
        f"Matched-Delivery Matrix: {title}",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )

    clean_label = selection_label
    if derived_noise and "noise SDs=" in clean_label:
        clean_label = re.sub(r";?\s*noise SDs=\[[^\]]*\]", "", clean_label).strip()

    subtitle_parts: list[str] = []
    if clean_label:
        subtitle_parts.append(clean_label)
    if derived_noise:
        subtitle_parts.append(f"Delivery Noise: {derived_noise}")
    if target_context and not any(
        target_context.lower() in p.lower() for p in subtitle_parts
    ):
        subtitle_parts.append(f"Target: {target_context}")

    subtitle_text = "\n".join(
        textwrap.fill(part, width=145, break_long_words=False)
        for part in subtitle_parts
    )
    # Preserve panel sizes while allocating physical space for covariance captions.
    # Extra header lines must not squeeze unavailable-cell reasons or data labels.
    base_height = 10.5
    header_extra = max(0, len(subtitle_text.splitlines()) - 3) * 0.24
    figure_height = base_height + header_extra
    fig.set_size_inches(16.5, figure_height)
    height_scale = base_height / figure_height
    fig.text(
        0.06,
        1.0 - 0.06 * height_scale,
        subtitle_text,
        ha="left",
        va="top",
        fontsize=10,
    )

    caption_notes: list[str] = [
        "Unsupported or infeasible conditions are hatched and labeled. Landing metrics are model-conditional;",
    ]
    if metric == "target_hit_fraction":
        caption_notes.append(_containment_radius_caption(cells) + ".")
    if metric == "conditional_sg":
        caption_notes.append(
            "strokes gained is a hypothetical scoring comparison conditional on supported endpoints, not player-score prediction."
        )
        if any(course_contexts.values()):
            ctx_summary = "; ".join(
                f"{_CLUB_LABELS[c]}: {ctx}" for c, ctx in course_contexts.items() if ctx
            )
            caption_notes.append(f"Course Scoring Context: {ctx_summary}.")
        if carry_final_lie:
            caption_notes.append(
                "Carry-as-Final-Lie Approximation (No Roll or Hazards)."
            )
    else:
        caption_notes.append(
            "strokes gained is a hypothetical scoring comparison conditional on supported endpoints, not player-score prediction."
        )

    if target_context and not any(
        target_context.lower() in n.lower() for n in caption_notes
    ):
        caption_notes.append(f"Target Context: {target_context}.")

    caption_text = textwrap.fill(
        " ".join(caption_notes), width=155, break_long_words=False
    )
    fig.text(
        0.06,
        0.045 * height_scale,
        caption_text,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.89 if images else 0.98,
        top=0.84 * height_scale,
        bottom=0.12 * height_scale,
        hspace=0.48,
        wspace=0.28,
    )
    if images:
        colorbar_axis = fig.add_axes(
            (0.93, 0.23 * height_scale, 0.016, 0.56 * height_scale)
        )
        colorbar = fig.colorbar(images[0], cax=colorbar_axis)
        colorbar.set_label(colorbar_label)
    outputs: dict[str, Path] = {}
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(
            path, format=extension, dpi=dpi, bbox_inches="tight", facecolor="white"
        )
        outputs[extension] = path
    plt.close(fig)
    return outputs


def export_matched_matrix_heatmaps(
    cells: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    selection_label: str,
    dpi: int = 300,
    metrics: Sequence[str] | None = None,
    formats: Sequence[str] = ("png", "pdf", "svg"),
) -> dict[str, Path]:
    """Export five shared-scale metric heatmaps for one selected matrix slice.

    Input records are normalized summary dictionaries from verified bundle
    manifests. Include each expected club × lean × curve cell, even when it is
    explicitly unavailable or not run. Different contact laws must be rendered
    as separate slices so model assumptions are never blended.
    """
    if not isinstance(selection_label, str) or not selection_label.strip():
        raise ValueError("selection_label must describe the selected matrix slice")
    if not isinstance(dpi, int) or not 72 <= dpi <= 600:
        raise ValueError("dpi must be an integer in [72, 600]")
    chosen_metrics = tuple(_METRICS if metrics is None else metrics)
    if not chosen_metrics or any(metric not in _METRICS for metric in chosen_metrics):
        raise ValueError("metrics must contain known metric names")
    chosen_formats = tuple(formats)
    if not chosen_formats or any(
        fmt not in ("png", "pdf", "svg") for fmt in chosen_formats
    ):
        raise ValueError("formats must be selected from png, pdf, svg")
    selected, leans, curves = _validate_cells(cells)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for metric in chosen_metrics:
        paths = _draw_metric(
            selected,
            leans,
            curves,
            metric,
            selection_label,
            output_dir / f"matched_matrix_{metric}",
            dpi=dpi,
            formats=chosen_formats,
        )
        outputs.update(
            {f"{metric}_{extension}": path for extension, path in paths.items()}
        )
    return outputs


__all__ = [
    "_cell_text_color",
    "_derive_course_context",
    "_derive_delivery_noise",
    "_derive_target_context",
    "_has_carry_as_final_lie",
    "export_matched_matrix_heatmaps",
    "metric_value",
]
