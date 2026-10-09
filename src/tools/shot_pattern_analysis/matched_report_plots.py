"""Academic heatmaps for selected matched-delivery matrix conditions.

Plotting is read-only: callers provide already validated completed summaries.
Unavailable and not-run conditions stay visible in gray hatched matrix cells.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
        value = stats.get("p95_radial_target_error_m") if isinstance(stats, Mapping) else None
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
        (club, lean, curve)
        for club in CLUBS
        for lean in leans
        for curve in curves
    }
    if set(indexes) != expected:
        raise ValueError("incomplete matrix axes: include explicit unavailable/not-run cells")
    return [indexes[key] for key in sorted(indexes)], tuple(sorted(leans)), tuple(sorted(curves))


def _format_metric(metric: str, value: float, stats: Mapping[str, Any]) -> str:
    supported = stats.get("supported_n")
    total = stats.get("total_n")
    count = f"{supported}/{total}" if isinstance(supported, int) and isinstance(total, int) else ""
    if metric in ("target_hit_fraction", "supported_fraction"):
        return f"{value:.0%}" + (f"\nn={count}" if count else "")
    if metric == "conditional_sg":
        marker = "conditional " if supported is not None and total is not None and supported < total else ""
        return f"{value:+.2f}" + (f"\n{marker}n={count}" if count else "")
    return f"{value:.2f}" + (f"\nn={count}" if count else "")


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
            axis.set_xlabel("Curve Scale")
            if col_index == 0:
                axis.set_ylabel(f"{pattern}\nShaft Lean")
            else:
                axis.set_ylabel("Shaft Lean")
            axis.set_title(
                f"{_CLUB_LABELS[club]} — {pattern}",
                fontsize=11,
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
                    stats_map = cell.get("summary", {})
                    stats = stats_map.get(pattern, {}) if isinstance(stats_map, Mapping) else {}
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
                        annotation = f"{label}\n{textwrap.fill(str(reason), width=18)}"
                    else:
                        annotation = _format_metric(
                            metric, value, stats if isinstance(stats, Mapping) else {}
                        )
                        axis.text(
                            j,
                            i,
                            annotation,
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="#111827",
                            zorder=4,
                        )
                    if value is None:
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
    fig.text(
        0.06,
        0.94,
        textwrap.fill(selection_label, width=150, break_long_words=False),
        ha="left",
        va="top",
        fontsize=10,
    )
    fig.text(
        0.06,
        0.045,
        "Unsupported or infeasible conditions are hatched and labeled. Landing metrics are model-conditional; "
        "strokes gained is a hypothetical scoring comparison conditional on supported endpoints, not player-score prediction.",
        ha="left",
        fontsize=9,
        color="#333333",
    )
    if images:
        colorbar = fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.78, pad=0.025)
        colorbar.set_label(colorbar_label)
    fig.subplots_adjust(left=0.08, right=0.92 if images else 0.98, top=0.84, bottom=0.12, hspace=0.48, wspace=0.28)
    outputs: dict[str, Path] = {}
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path, format=extension, dpi=dpi, bbox_inches="tight", facecolor="white")
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
    if not chosen_formats or any(fmt not in ("png", "pdf", "svg") for fmt in chosen_formats):
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
        outputs.update({f"{metric}_{extension}": path for extension, path in paths.items()})
    return outputs


__all__ = ["export_matched_matrix_heatmaps", "metric_value"]
