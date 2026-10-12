"""Plotting helpers for the additive matched-delivery supplement report."""

from __future__ import annotations

import hashlib
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .academic_plots import (
    FigureOptions,
    write_academic_dispersion_figure,
    write_overhead_trajectory_figure,
)
from .matched_figure_identity import (
    _slug as _slug,
    pair_comparison_identity as pair_comparison_identity,
    pair_comparison_slug as pair_comparison_slug,
    validate_planned_figure_identities as validate_planned_figure_identities,
)
from .density import DensityComparison, shared_histogram_density
from .matched_matrix import PROFILES
from .matched_pair_social import PairSocialContext, write_pair_social_density
from .matched_report import load_individually_supported_endpoints
from .matched_sensitivity import _supported_endpoints
from .provenance import manifest_files_match
from .noise import NOISE_DISPLAY_LABELS, NoiseSpec

CLOSED_SELECTED_FIGURE_KINDS: frozenset[str] = frozenset(
    {
        "density_png",
        "density_pdf",
        "density_svg",
        "overhead_png",
        "overhead_pdf",
        "overhead_svg",
        "overhead_social_png",
    }
)

__all__ = [
    "CLOSED_SELECTED_FIGURE_KINDS",
    "namespace_figure_paths",
    "pair_comparison_identity",
    "pair_comparison_slug",
    "render_selected_pair_figures",
    "validate_planned_figure_identities",
]

_PATTERNS = ("Straight", "Draw", "Fade")
_StyledCondition = tuple[str, dict[str, Any], Literal["solid", "dashed"]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile(cell_name: str) -> str:
    found = [profile for profile in PROFILES if cell_name.endswith("_" + profile)]
    return found[0] if len(found) == 1 else ""


def _condition(ref: dict[str, Any]) -> dict[str, Any]:
    config = ref["config"]
    club = config["club"]
    name = ref.get("alias_of_primary", ref["cell"])
    profile = _profile(name)
    if not profile:
        raise ValueError(f"cell name lacks a supported profile: {name}")
    summary = ref["summary"]
    target = summary.get("target_x_m")
    if not isinstance(target, (int, float)):
        raise ValueError(f"cell has no resolved physical target: {name}")
    return {
        "club": club["club_id"],
        "lean": float(club["shaft_lean_deg"]),
        "curve": float(config["curve_scale"]),
        "rotation": float(config["noise"]["standard_deviations"][0]),
        "profile": profile,
        "target": round(float(target), 12),
    }


def _noise_profile_description(
    profile: str, noise_config: dict[str, Any] | None = None
) -> str:
    """Describe actual serialized primitive noise, never infer it from a name."""
    if not isinstance(noise_config, dict):
        raise ValueError("caption requires serialized noise parameters")
    spec = NoiseSpec(
        tuple(noise_config["standard_deviations"]), noise_config.get("correlation")
    )
    sd = spec.standard_deviations
    human_profile = (
        "Covariance Profile"
        if profile.startswith("cov_facepath_")
        else re.sub(r"([A-Za-z])([0-9])", r"\1 \2", profile).replace("_", " ").title()
    )
    parts: list[str] = []
    if sd[1]:
        parts.append(f"Path SD {sd[1]:g}°")
    if sd[2]:
        parts.append(f"Speed SD {sd[2] / 0.44704:g} mph")
    if sd[3] == sd[4] and sd[3]:
        parts.append(f"Strike SD {sd[3] / 0.0254:g} in on Each Axis")
    else:
        for label, value in (("Horizontal", sd[3]), ("Vertical", sd[4])):
            if value:
                parts.append(f"{label} Strike SD {value / 0.0254:g} in")
    for label, value in (("Shaft Lean", sd[5]), ("Attack Angle", sd[6])):
        if value:
            parts.append(f"{label} SD {value:g}°")
    names = NOISE_DISPLAY_LABELS
    correlation = spec.correlation_matrix()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if sd[i] and sd[j] and correlation[i, j]:
                parts.append(
                    f"{names[i]}–{names[j]} Correlation r = {correlation[i, j]:+g}"
                )
    return f"{human_profile} ({', '.join(parts)})" if parts else human_profile


def _condition_caption(ref: dict[str, Any], *, include_noise: bool = True) -> str:
    condition = _condition(ref)
    club = str(condition["club"]).replace("_", " ").title()
    profile = _noise_profile_description(
        condition["profile"], ref.get("config", {}).get("noise")
    )
    return (
        f"{club} · "
        + (f"{profile} · " if include_noise else "")
        + f"Lean {condition['lean']:g}° · "
        f"Curve ×{condition['curve']:g} · Shaft Twist SD {condition['rotation']:g}°"
    )


def _pair_social_scenario(
    comparison: str, first: dict[str, Any], second: dict[str, Any]
) -> str:
    """Construct scenario string distinguishing varying panel condition from fixed controls."""
    c1 = _condition(first)
    comparison_title = _comparison_title(comparison)
    fixed_parts: list[str] = []
    if comparison != "across_lean":
        fixed_parts.append(f"Lean {c1['lean']:g}°")
    if comparison != "curve_strength":
        fixed_parts.append(f"Curve ×{c1['curve']:g}")
    if comparison != "shaft_rotation_sd":
        fixed_parts.append(f"Shaft Twist SD {c1['rotation']:g}°")
    fixed_str = " · ".join(fixed_parts)
    return f"Varying {comparison_title} in Panels · Fixed: {fixed_str}"


def _pair_condition_label(comparison: str, ref: dict[str, Any]) -> str:
    condition = _condition(ref)
    if comparison == "shaft_rotation_sd":
        return f"Rotation SD {condition['rotation']:g}°"
    if comparison == "curve_strength":
        return f"Curve ×{condition['curve']:g}"
    if comparison == "across_lean":
        return f"Lean {condition['lean']:g}°"
    return f"{_comparison_title(comparison)} · {_condition_caption(ref)}"


def _comparison_title(comparison: str) -> str:
    return comparison.replace("_", " ").title().replace("Sd", "SD")


def _plot_metadata(
    ref: dict[str, Any], label: str, valid_counts: dict[str, int]
) -> dict[str, Any]:
    config = ref["config"]
    summary = ref["summary"]
    total = {pattern: int(config["n_shots"]) for pattern in _PATTERNS}
    return {
        "source": ref["source_protocol"],
        "cell": ref.get("alias_cell", ref["cell"]),
        "club": config["club"]["club_id"],
        "lean_deg": config["club"]["shaft_lean_deg"],
        "curve_scale": config["curve_scale"],
        "shaft_rotation_sd_deg": config["noise"]["standard_deviations"][0],
        "figure_scope": label,
        "valid_shots_by_pattern": valid_counts,
        "total_shots_by_pattern": total,
        "conditioning": "individually supported endpoints by physical pattern",
    }


def _raw_flight_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep saved flight samples distinct from independently supported endpoints."""
    excluded = {
        "conditioning",
        "Population",
        "valid_shots_by_pattern",
        "total_shots_by_pattern",
    }
    return {
        **{key: value for key, value in metadata.items() if key not in excluded},
        "Population": "Saved Raw Trajectories; Not Paired by Shot Index",
        "figure_scope": "Saved Raw Flight Samples",
    }


def _cell_target_mode(ref: Mapping[str, Any]) -> str:
    """Classify cell target mode to distinguish within-lean matching from controls."""
    if ref.get("execution_kind") == "provenance_alias":
        return "Fixed-Target Control (Alias)"
    config = ref.get("config")
    if isinstance(config, Mapping) and config.get("target_x_m") is not None:
        return "Fixed-Target Control"
    req_config = ref.get("requested_config")
    if isinstance(req_config, Mapping) and req_config.get("target_x_m") is not None:
        return "Fixed-Target Control"
    return "Within-Lean Nominal Target"


def _cell_figure_ledger(
    ref: dict[str, Any],
    label: str,
    target_x: float,
    metadata: dict[str, Any],
    trajectory_paths: dict[str, Any],
    output_dir: Path,
    groups: tuple[tuple[str, dict[str, Path], str], ...],
) -> list[dict[str, Any]]:
    """Record supported endpoint counts separately from raw trajectory samples."""
    ledger: list[dict[str, Any]] = []
    for figure_group, paths, presentation in groups:
        for kind, path in paths.items():
            ledger.append(
                {
                    "figure_kind": f"{figure_group}_{kind}",
                    "presentation": presentation,
                    "source_kind": (
                        "individual_supported_endpoints"
                        if "density" in figure_group
                        else "saved_illustrative_trajectories"
                    ),
                    "source_protocol": ref["source_protocol"],
                    "cell": label,
                    "target_x_m": target_x,
                    "conditioning": (
                        "individually supported endpoints by physical pattern"
                        if "density" in figure_group
                        else "saved raw trajectories, not paired by shot index"
                    ),
                    "valid_n_by_pattern": (
                        metadata["valid_shots_by_pattern"]
                        if "density" in figure_group
                        else None
                    ),
                    "total_n_by_pattern": (
                        metadata["total_shots_by_pattern"]
                        if "density" in figure_group
                        else None
                    ),
                    "saved_trajectory_samples_by_pattern": (
                        {name: len(trajectory_paths[name]) for name in _PATTERNS}
                        if "overhead" in figure_group
                        else None
                    ),
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(path),
                }
            )
    return ledger


def _load_cell_flight_samples(cell_dir: Path, cell: str) -> dict[str, Any]:
    """Require saved trajectories for each physical pattern before rendering."""
    flights_path = cell_dir / "flight_samples.json"
    if not flights_path.is_file():
        raise ValueError(f"figure source has no flight samples: {cell}")
    flight_record = json.loads(flights_path.read_text(encoding="utf-8"))
    trajectory_paths = flight_record.get("trajectories")
    if not isinstance(trajectory_paths, dict) or set(trajectory_paths) != set(
        _PATTERNS
    ):
        raise ValueError(f"flight samples lack pattern trajectories: {cell}")
    return trajectory_paths


def _render_cell_figures(ref: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    if ref.get("status") != "completed":
        return []
    label = str(ref.get("alias_cell", ref["cell"]))
    protocol = ref.get("source_protocol")
    if not protocol or not isinstance(protocol, str) or not protocol.strip():
        raise ValueError(f"cell reference lacks source protocol: {label}")
    cell_dir = Path(ref["source_root"]) / ref["cell"]
    manifest_path = cell_dir / "manifest.json"
    if not manifest_files_match(manifest_path):
        raise ValueError(f"figure source manifest is invalid: {ref['cell']}")
    points, _ = load_individually_supported_endpoints(
        cell_dir / "shots.csv.gz", int(ref["config"]["n_shots"])
    )
    arrays = {name: points[name] for name in _PATTERNS}
    target_x = float(ref["summary"]["target_x_m"])
    stem = output_dir / "cells" / _slug(label)
    valid_counts = {name: len(arrays[name]) for name in _PATTERNS}
    metadata = _plot_metadata(ref, "one-cell supported population", valid_counts)
    social_club = str(ref["config"]["club"]["club_id"]).replace("_", " ").title()
    target_mode = _cell_target_mode(ref)
    social_metadata = {
        "social_club": social_club,
        "Protocol": protocol,
        "Matched Target": f"{target_x:.2f} m downrange ({target_mode})",
        "Delivery Noise": _noise_profile_description(
            _condition(ref)["profile"], ref["config"]["noise"]
        ),
        "Population": "Individually Supported Endpoints by Physical Pattern",
        "valid_shots_by_pattern": valid_counts,
        "total_shots_by_pattern": metadata["total_shots_by_pattern"],
    }
    ledger: list[dict[str, Any]] = []
    density_paths: dict[str, Path] = {}
    density_social_paths: dict[str, Path] = {}
    if all(len(arrays[name]) >= 2 for name in _PATTERNS):
        density_paths = write_academic_dispersion_figure(
            arrays,
            stem.with_name(stem.name + "_density"),
            scenario=f"Cell: {label}",
            metadata=metadata,
            target_point_m=(target_x, 0.0),
            options=FigureOptions(presentation="academic"),
        )
        density_social_paths = write_academic_dispersion_figure(
            arrays,
            stem.with_name(stem.name + "_density_social"),
            scenario=_condition_caption(ref, include_noise=False),
            metadata=social_metadata,
            target_point_m=(target_x, 0.0),
            options=FigureOptions(presentation="social"),
        )
    else:
        ledger.append(
            {
                "figure_kind": "density",
                "source_kind": "individual_supported_endpoints",
                "source_protocol": ref["source_protocol"],
                "cell": label,
                "status": "unavailable",
                "reason": "at least one physical pattern has fewer than two supported endpoints",
                "valid_n_by_pattern": valid_counts,
                "total_n_by_pattern": metadata["total_shots_by_pattern"],
            }
        )
    trajectory_paths = _load_cell_flight_samples(cell_dir, ref["cell"])
    overhead_paths = write_overhead_trajectory_figure(
        trajectory_paths,
        stem.with_name(stem.name + "_overhead"),
        target_x_m=target_x,
        scenario=f"Cell: {label}; Raw Simulator Flight Samples",
        metadata=_raw_flight_metadata(metadata),
        options=FigureOptions(presentation="academic"),
    )
    overhead_social_paths = write_overhead_trajectory_figure(
        trajectory_paths,
        stem.with_name(stem.name + "_overhead_social"),
        target_x_m=target_x,
        scenario=_condition_caption(ref, include_noise=False) + " · Raw Flight Samples",
        metadata=_raw_flight_metadata(social_metadata),
        options=FigureOptions(presentation="social"),
    )
    groups = (
        ("density", density_paths, "academic"),
        ("density_social", density_social_paths, "social"),
        ("overhead", overhead_paths, "academic"),
        ("overhead_social", overhead_social_paths, "social"),
    )
    ledger.extend(
        _cell_figure_ledger(
            ref, label, target_x, metadata, trajectory_paths, output_dir, groups
        )
    )
    return ledger


def _save_figure(
    figure: Any, stem: Path, formats: tuple[str, ...] = ("png", "pdf", "svg")
) -> dict[str, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix, dpi in (("png", 300), ("pdf", None), ("svg", None)):
        if suffix not in formats:
            continue
        path = stem.with_suffix(f".{suffix}")
        options: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi is not None:
            options["dpi"] = dpi
        figure.savefig(path, **options)
        outputs[suffix] = path
    return outputs


def namespace_figure_paths(group: str, paths: dict[str, Path]) -> dict[str, Path]:
    """Keep density and overhead exports distinct when combining ledgers."""
    if group not in {"density", "overhead"}:
        raise ValueError("figure path group must be density or overhead")
    if not paths or any(not kind for kind in paths):
        raise ValueError("figure path suffixes must be nonempty")
    return {f"{group}_{kind}": path for kind, path in paths.items()}


def _displayed_sample_indices(point_count: int, maximum: int = 600) -> np.ndarray:
    """Select a deterministic, evenly spaced dot overlay without thinning density."""
    if point_count < 1 or maximum < 1:
        raise ValueError("point_count and maximum must both be positive")
    return np.linspace(0, point_count - 1, min(point_count, maximum), dtype=int)


@dataclass(frozen=True)
class PairFigureContext:
    """Bind the two saved conditions, display labels, and common physical target."""

    first: dict[str, Any]
    second: dict[str, Any]
    comparison: str
    labels: tuple[str, str]
    target_x: float


@dataclass
class _HeaderStack:
    """Place wrapped figure headers with gaps measured in physical points."""

    figure: Any
    renderer: Any
    y_cursor: float

    def add(
        self, text: str, fontsize: float, gap_pt: float = 3.5, wrap_chars: int = 180
    ) -> Any:
        figure = self.figure
        gap_frac = (gap_pt / 72.0) / (figure.bbox.height / figure.dpi)
        wrapped = textwrap.fill(text, width=wrap_chars, break_long_words=False)
        artist = figure.text(
            0.06,
            self.y_cursor - gap_frac,
            wrapped,
            ha="left",
            va="top",
            fontsize=fontsize,
        )
        self.y_cursor = artist.get_window_extent(self.renderer).y0 / figure.bbox.height
        return artist


def _pair_density_grid_top(
    figure: Any,
    first: dict,
    second: dict,
    comparison: str,
    labels: tuple[str, str],
    target_x: float,
    common_n: int,
) -> float:
    """Lay out captions and legend, returning the available panel ceiling."""
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    first_label, second_label = labels
    canvas = FigureCanvasAgg(figure)
    renderer = canvas.get_renderer()

    t_title = figure.text(
        0.06,
        0.985,
        "Matched Endpoint Density: Two Conditions, Shared Axes and Density Scale",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )
    header = _HeaderStack(
        figure, renderer, t_title.get_window_extent(renderer).y0 / figure.bbox.height
    )

    total_n = int(first["config"]["n_shots"])
    comp_text = (
        f"{comparison} · Target {target_x:.3f} m · Density uses all "
        f"{common_n:,}/{total_n:,} shots common to six populations; "
        f"{min(common_n, 600):,} evenly spaced dots per panel"
    )
    header.add(comp_text, fontsize=8.5, gap_pt=5.0, wrap_chars=180)

    cap1 = f"{first_label}: {_condition_caption(first)} · {first['source_protocol']}"
    header.add(cap1, fontsize=7.7, gap_pt=3.5, wrap_chars=180)

    cap2 = f"{second_label}: {_condition_caption(second)} · {second['source_protocol']}"
    header.add(cap2, fontsize=7.7, gap_pt=3.5, wrap_chars=180)

    disc_text = (
        "Illustrative Simulation | Assumed Head Properties | Carry Only · "
        "Raw coordinates; no aim correction. Matched-condition changes include nominal re-matching."
    )
    header.add(disc_text, fontsize=9.0, gap_pt=3.5, wrap_chars=180)

    leg_gap_frac = (7.0 / 72.0) / (figure.bbox.height / figure.dpi)
    legend_top = header.y_cursor - leg_gap_frac
    hdr_styles: tuple[tuple[float, Literal["solid", "dashed", "dotted"]], ...] = (
        (0.50, "solid"),
        (0.80, "dashed"),
        (0.95, "dotted"),
    )
    hdr = [
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle=style,
            label=f"{mass:.0%} Simulated-Sample HDR",
        )
        for mass, style in hdr_styles
    ]
    hdr.append(
        Line2D(
            [0], [0], marker="+", color="black", linestyle="none", label="Common Target"
        )
    )
    legend = figure.legend(
        handles=hdr,
        loc="upper center",
        bbox_to_anchor=(0.48, legend_top),
        ncol=4,
        frameon=False,
    )
    leg_bottom = legend.get_window_extent(renderer).y0 / figure.bbox.height
    margin_frac = (14.0 / 72.0) / (figure.bbox.height / figure.dpi)
    return min(0.76, leg_bottom - margin_frac)


def _draw_pair_density_panels(
    figure: Any,
    grid: Any,
    estimate: DensityComparison,
    point_sets: dict[str, list[tuple[float, float]]],
    row_conditions: tuple[tuple[str, dict], tuple[str, dict]],
    target_x: float,
) -> None:
    """Render paired populations with their common density normalization."""
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=0.0, vmax=estimate.density_vmax_per_m2)
    color_mesh = None
    pattern_colors = {"Straight": "#286090", "Draw": "#c66a19", "Fade": "#7b4a9e"}
    for row_index, (condition_name, _) in enumerate(row_conditions):
        for col_index, pattern in enumerate(_PATTERNS):
            key = f"{pattern} · {condition_name}"
            sample: np.ndarray = np.asarray(point_sets[key], dtype=float)
            density = estimate.patterns[key].density_per_m2
            axis = figure.add_subplot(grid[row_index, col_index])
            color_mesh = axis.pcolormesh(
                estimate.x_edges_m,
                estimate.y_edges_m,
                density.T,
                cmap="Blues",
                norm=norm,
                shading="flat",
                rasterized=True,
            )
            displayed_indices = _displayed_sample_indices(len(sample))
            axis.scatter(
                sample[displayed_indices, 0],
                sample[displayed_indices, 1],
                s=3,
                color=pattern_colors[pattern],
                alpha=0.25,
                linewidths=0,
                rasterized=True,
            )
            density_summary = estimate.patterns[key]
            styles: dict[float, Literal["solid", "dashed", "dotted"]] = {
                0.50: "solid",
                0.80: "dashed",
                0.95: "dotted",
            }
            centers_x = (estimate.x_edges_m[1:] + estimate.x_edges_m[:-1]) / 2
            centers_y = (estimate.y_edges_m[1:] + estimate.y_edges_m[:-1]) / 2
            for mass, style in styles.items():
                threshold = density_summary.hdr_density_thresholds[mass]
                if 0.0 < threshold < estimate.density_vmax_per_m2:
                    axis.contour(
                        centers_x,
                        centers_y,
                        density.T,
                        levels=[threshold],
                        colors=[pattern_colors[pattern]],
                        linestyles=[style],
                        linewidths=1.0,
                    )
            axis.scatter([target_x], [0.0], marker="+", color="black", s=32)
            axis.axvline(target_x, color="#444444", linestyle=":", linewidth=0.7)
            axis.axhline(0.0, color="#444444", linewidth=0.5)
            axis.set_xlim(estimate.x_limits_m)
            axis.set_ylim(estimate.y_limits_m)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2, linewidth=0.4)
            axis.set_title(
                f"{pattern} · {condition_name}",
                fontsize=10,
            )
            axis.set_xlabel("Downrange (m)")
            if col_index == 0:
                axis.set_ylabel("Lateral (m; Right Positive)")
    if color_mesh is not None:
        colorbar = figure.colorbar(
            color_mesh, ax=figure.axes, fraction=0.018, pad=0.055
        )
        colorbar.set_label("Endpoint Probability Density (m⁻²); Common Scale")


def _write_pair_density_figure(
    point_sets: dict[str, list[tuple[float, float]]],
    *,
    context: PairFigureContext,
    common_n: int,
    output_stem: Path,
    formats: tuple[str, ...] = ("png", "pdf", "svg"),
) -> dict[str, Path]:
    """Plot two conditions in six panels on common bins and density scale."""
    first, second = context.first, context.second
    comparison, target_x = context.comparison, context.target_x
    first_label, second_label = context.labels
    import matplotlib.pyplot as plt

    estimate = shared_histogram_density(
        point_sets, bins=64, target_point_m=(target_x, 0.0)
    )
    figure = plt.figure(figsize=(16, 9), facecolor="white")
    grid_top = _pair_density_grid_top(
        figure,
        first,
        second,
        comparison,
        (first_label, second_label),
        target_x,
        common_n,
    )

    grid = figure.add_gridspec(
        2, 3, left=0.06, right=0.91, top=grid_top, bottom=0.13, hspace=0.36, wspace=0.24
    )
    first_condition, second_condition = _condition(first), _condition(second)
    row_conditions = ((first_label, first_condition), (second_label, second_condition))
    _draw_pair_density_panels(
        figure, grid, estimate, point_sets, row_conditions, target_x
    )
    if formats == ("png", "pdf", "svg"):
        outputs = _save_figure(figure, output_stem)
    else:
        outputs = _save_figure(figure, output_stem, formats=formats)
    plt.close(figure)
    return outputs


def _decorate_pair_overhead(
    figure: Any,
    trajectory_sets: dict[str, list[Any]],
    conditions: tuple[_StyledCondition, _StyledCondition],
    comparison: str,
    target_x: float,
    social: bool,
    colors: dict[str, str],
) -> None:
    """Place overhead captions and the pattern/condition legend."""
    from matplotlib.lines import Line2D

    (first_label, first, _), (second_label, second, _) = conditions
    font_scale = 1.0 if social else 0.92
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(figure)
    renderer = canvas.get_renderer()

    t_title = figure.text(
        0.06,
        0.975,
        "Saved Raw Flight Paths by Pattern and Matched Condition",
        ha="left",
        va="top",
        fontsize=17 * font_scale,
        fontweight="bold",
    )
    header = _HeaderStack(
        figure, renderer, t_title.get_window_extent(renderer).y0 / figure.bbox.height
    )

    wrap_w = 175 if not social else 165
    header.add(
        f"{_comparison_title(comparison)} · Target {target_x:.3f} m",
        fontsize=10 * font_scale,
        gap_pt=5.0,
        wrap_chars=wrap_w,
    )
    header.add(
        f"{first_label}: {_condition_caption(first)} · {first['source_protocol']}",
        fontsize=8.5 * font_scale,
        gap_pt=3.5,
        wrap_chars=wrap_w,
    )
    header.add(
        f"{second_label}: {_condition_caption(second)} · {second['source_protocol']}",
        fontsize=8.5 * font_scale,
        gap_pt=3.5,
        wrap_chars=wrap_w,
    )
    path_counts = {key: len(value) for key, value in trajectory_sets.items()}
    first_counts = ", ".join(
        f"{pattern} {path_counts[f'{pattern} · {first_label}']}"
        for pattern in _PATTERNS
    )
    second_counts = ", ".join(
        f"{pattern} {path_counts[f'{pattern} · {second_label}']}"
        for pattern in _PATTERNS
    )
    header.add(
        f"Saved paths (not paired by shot index): {first_label} {first_counts} · {second_label} {second_counts}",
        fontsize=8.5 * font_scale,
        gap_pt=3.5,
        wrap_chars=wrap_w,
    )
    header.add(
        "Illustrative Simulation | Assumed Head Properties | Carry Only · Lateral display magnified 6×; physical axis limits are shared.",
        fontsize=8.5 * font_scale,
        gap_pt=3.5,
        wrap_chars=wrap_w,
    )
    pattern_handles = [
        Line2D([0], [0], color=colors[pattern], label=pattern) for pattern in _PATTERNS
    ]
    condition_handles = [
        Line2D([0], [0], color="#333333", linestyle=style, label=condition)
        for condition, _, style in conditions
    ]
    leg_gap_frac = (7.0 / 72.0) / (figure.bbox.height / figure.dpi)
    legend_top = header.y_cursor - leg_gap_frac
    figure.legend(
        handles=pattern_handles + condition_handles,
        loc="upper center",
        bbox_to_anchor=(0.57, legend_top),
        ncol=5,
        frameon=False,
        fontsize=9 * font_scale,
    )


def _create_pair_overhead_figure(
    trajectory_sets: dict[str, list[Any]],
    *,
    first: dict,
    second: dict,
    comparison: str,
    first_label: str,
    second_label: str,
    target_x: float,
    social: bool,
):
    import matplotlib.pyplot as plt

    all_points = [
        np.asarray(path, dtype=float)
        for paths in trajectory_sets.values()
        for path in paths
    ]
    lows = np.minimum(np.min(np.concatenate(all_points), axis=0), (0.0, 0.0))
    highs = np.maximum(np.max(np.concatenate(all_points), axis=0), (target_x, 0.0))
    padding = np.maximum((highs - lows) * 0.05, 0.25)
    lows -= padding
    highs += padding

    font_scale = 1.0 if social else 0.92
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 9),
        sharex=True,
        sharey=True,
        squeeze=False,
        facecolor="white",
    )
    figure.subplots_adjust(top=0.70, bottom=0.19, left=0.085, right=0.98, wspace=0.13)
    colors = {"Straight": "#286090", "Draw": "#c66a19", "Fade": "#7b4a9e"}
    conditions: tuple[_StyledCondition, _StyledCondition] = (
        (first_label, first, "solid"),
        (second_label, second, "dashed"),
    )
    for col, pattern in enumerate(_PATTERNS):
        axis = axes[0][col]
        for condition_name, _ref, style in conditions:
            paths = trajectory_sets[f"{pattern} · {condition_name}"]
            for path in paths:
                array = np.asarray(path, dtype=float)
                axis.plot(
                    array[:, 0],
                    array[:, 1],
                    color=colors[pattern],
                    linestyle=style,
                    alpha=0.46 if style == "solid" else 0.72,
                    linewidth=1.0,
                    rasterized=True,
                )
        axis.scatter([target_x], [0.0], marker="+", color="black", s=34, zorder=4)
        axis.axvline(target_x, color="#444444", linestyle=":", linewidth=0.7)
        axis.axhline(0.0, color="#444444", linewidth=0.5)
        axis.set_xlim(float(lows[0]), float(highs[0]))
        axis.set_ylim(float(lows[1]), float(highs[1]))
        axis.set_title(pattern, fontsize=12 * font_scale, fontweight="bold")
        axis.set_xlabel("Downrange Position (m)", fontsize=10 * font_scale)
        if col == 0:
            axis.set_ylabel(
                "Lateral Position (m; Right Positive)", fontsize=10 * font_scale
            )
        axis.grid(alpha=0.2, linewidth=0.4)
        # A disclosed 6× vertical display magnification keeps long carry paths legible.
        axis.set_aspect(6.0, adjustable="box")

    _decorate_pair_overhead(
        figure, trajectory_sets, conditions, comparison, target_x, social, colors
    )
    return plt, figure


def _write_pair_overhead_figure(
    trajectory_sets: dict[str, list[Any]],
    *,
    context: PairFigureContext,
    output_stem: Path,
    academic_formats: tuple[str, ...] = ("png", "pdf", "svg"),
    include_social: bool = True,
) -> dict[str, Path]:
    """Write academic and dedicated 16:9 paired overhead exports."""
    first, second = context.first, context.second
    comparison, target_x = context.comparison, context.target_x
    first_label, second_label = context.labels
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    if academic_formats:
        plt, figure = _create_pair_overhead_figure(
            trajectory_sets,
            first=first,
            second=second,
            comparison=comparison,
            first_label=first_label,
            second_label=second_label,
            target_x=target_x,
            social=False,
        )
        if academic_formats == ("png", "pdf", "svg"):
            outputs = _save_figure(figure, output_stem)
        else:
            outputs = _save_figure(figure, output_stem, formats=academic_formats)
        plt.close(figure)

    if include_social:
        plt, social_figure = _create_pair_overhead_figure(
            trajectory_sets,
            first=first,
            second=second,
            comparison=comparison,
            first_label=first_label,
            second_label=second_label,
            target_x=target_x,
            social=True,
        )
        social_path = output_stem.with_name(output_stem.name + "_social.png")
        social_figure.savefig(social_path, dpi=120, facecolor="white")
        plt.close(social_figure)
        outputs["social_png"] = social_path
    return outputs


@dataclass(frozen=True)
class _PairExportRequest:
    """A validated comparison with its artifact destination and selected formats."""

    context: PairFigureContext
    output_dir: Path
    stem: Path
    figure_kinds: frozenset[str] | None


def _pair_export_formats(
    group: str, figure_kinds: frozenset[str] | None
) -> tuple[str, ...]:
    """Select academic formats in stable PNG, PDF, SVG order."""
    return tuple(
        suffix
        for suffix in ("png", "pdf", "svg")
        if figure_kinds is None or f"{group}_{suffix}" in figure_kinds
    )


def _load_pair_common_points(
    context: PairFigureContext,
) -> tuple[dict[str, list[tuple[float, float]]], int]:
    """Intersect saved supported shot indices across both conditions and patterns."""
    loaded = []
    for ref in (context.first, context.second):
        directory = Path(ref["source_root"]) / ref["cell"]
        endpoints, _ = _supported_endpoints(
            directory / "shots.csv.gz", int(ref["config"]["n_shots"])
        )
        loaded.append(endpoints)
    common = sorted(
        set.intersection(
            *(
                {index for index, _, _ in endpoint_map[pattern]}
                for endpoint_map in loaded
                for pattern in _PATTERNS
            )
        )
    )
    point_sets: dict[str, list[tuple[float, float]]] = {}
    for condition_label, endpoint_map in zip(context.labels, loaded, strict=True):
        by_pattern = {
            name: {idx: (x, y) for idx, x, y in endpoint_map[name]}
            for name in _PATTERNS
        }
        for pattern in _PATTERNS:
            name = f"{pattern} · {condition_label}"
            point_sets[name] = [by_pattern[pattern][index] for index in common]
    return point_sets, len(common)


def _pair_density_ledger(
    request: _PairExportRequest,
    paths: dict[str, Path],
    common_n: int,
    *,
    social: bool = False,
) -> list[dict[str, Any]]:
    """Record the same common population for academic and per-pattern exports."""
    context = request.context
    first, second = context.first, context.second
    ledger: list[dict[str, Any]] = []
    for key, path in paths.items():
        pattern: str | None
        if social:
            pattern_lower, ext = key.split("_", 1)
            pattern = pattern_lower.title()
            kind = f"density_{pattern_lower}_social_{ext}"
        else:
            kind, pattern = key, None
            if request.figure_kinds is not None and kind not in request.figure_kinds:
                continue
        ledger.append(
            {
                "figure_kind": kind,
                "presentation": "social" if social else "academic",
                "physical_pattern": pattern,
                "source_kind": "six_population_common_index_comparison",
                "comparison_kind": context.comparison,
                "first_cell": first["cell"],
                "second_cell": second["cell"],
                "first_protocol": first["source_protocol"],
                "second_protocol": second["source_protocol"],
                "common_supported_n": common_n,
                "total_n_per_population": first["config"]["n_shots"],
                "conditioning": "same shot indices supported in all six populations",
                "target_x_m": context.target_x,
                "relative_path": path.relative_to(request.output_dir).as_posix(),
                "sha256": _sha256(path),
            }
        )
    return ledger


def _unavailable_pair_figure(
    context: PairFigureContext,
    kind: str,
    source_kind: str,
    reason: str,
) -> dict[str, Any]:
    """Retain an explicit unavailable row instead of silently omitting a population."""
    return {
        "figure_kind": kind,
        "source_kind": source_kind,
        "comparison_kind": context.comparison,
        "first_cell": context.first["cell"],
        "second_cell": context.second["cell"],
        "status": "unavailable",
        "reason": reason,
    }


def _render_pair_social_density(
    request: _PairExportRequest,
    point_sets: dict[str, list[tuple[float, float]]],
    common_n: int,
) -> list[dict[str, Any]]:
    """Export the full-report social figures with their original condition captions."""
    context, stem = request.context, request.stem
    first, second = context.first, context.second
    pair_scenario = _pair_social_scenario(context.comparison, first, second)
    social_club = str(first["config"]["club"]["club_id"]).replace("_", " ").title()
    m0, m1 = _cell_target_mode(first), _cell_target_mode(second)
    paths = write_pair_social_density(
        point_sets,
        stem.with_name(stem.name + "_density"),
        context=PairSocialContext(
            condition_labels=context.labels,
            scenario=pair_scenario,
            social_club=social_club,
            target_point_m=(context.target_x, 0.0),
            total_n=int(first["config"]["n_shots"]),
            delivery_noise=_noise_profile_description(
                _condition(first)["profile"], first["config"]["noise"]
            ),
            source_protocols=(first["source_protocol"], second["source_protocol"]),
            target_modes=(m0, m1),
        ),
    )
    return _pair_density_ledger(request, paths, common_n, social=True)


def _render_pair_density(request: _PairExportRequest) -> list[dict[str, Any]]:
    """Render only the requested common-index density exports."""
    point_sets, common_n = _load_pair_common_points(request.context)
    if common_n < 2:
        return [
            _unavailable_pair_figure(
                request.context,
                "paired_density",
                "six_population_common_index_comparison",
                "fewer than two indices supported in all six populations",
            )
        ]
    ledger: list[dict[str, Any]] = []
    formats = _pair_export_formats("density", request.figure_kinds)
    if formats:
        paths = namespace_figure_paths(
            "density",
            _write_pair_density_figure(
                point_sets,
                context=request.context,
                common_n=common_n,
                output_stem=request.stem.with_name(request.stem.name + "_density"),
                formats=formats,
            ),
        )
        ledger.extend(_pair_density_ledger(request, paths, common_n))
    if request.figure_kinds is None:
        ledger.extend(_render_pair_social_density(request, point_sets, common_n))
    return ledger


def _load_pair_trajectories(context: PairFigureContext) -> dict[str, list[Any]]:
    """Read saved raw paths without applying endpoint support conditioning."""
    trajectory_sets: dict[str, list[Any]] = {}
    for label, ref in zip(context.labels, (context.first, context.second), strict=True):
        directory = Path(ref["source_root"]) / ref["cell"]
        flight = json.loads((directory / "flight_samples.json").read_text())
        paths_by_pattern = flight.get("trajectories", {})
        for pattern in _PATTERNS:
            trajectory_sets[f"{pattern} · {label}"] = paths_by_pattern.get(pattern, [])
    return trajectory_sets


def _pair_overhead_ledger(
    request: _PairExportRequest,
    paths: dict[str, Path],
    trajectory_sets: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Record raw trajectory sample counts independently of supported endpoints."""
    context = request.context
    first, second = context.first, context.second
    ledger: list[dict[str, Any]] = []
    for kind, path in paths.items():
        if request.figure_kinds is not None and kind not in request.figure_kinds:
            continue
        ledger.append(
            {
                "figure_kind": kind,
                "presentation": "social" if "social" in kind else "academic",
                "physical_pattern": None,
                "source_kind": "saved_illustrative_trajectory_samples",
                "comparison_kind": context.comparison,
                "first_cell": first["cell"],
                "second_cell": second["cell"],
                "first_protocol": first["source_protocol"],
                "second_protocol": second["source_protocol"],
                "conditioning": "saved raw trajectories, not paired by shot index",
                "target_x_m": context.target_x,
                "saved_trajectory_samples_by_pattern": {
                    key: len(samples) for key, samples in trajectory_sets.items()
                },
                "relative_path": path.relative_to(request.output_dir).as_posix(),
                "sha256": _sha256(path),
            }
        )
    return ledger


def _render_pair_overhead(request: _PairExportRequest) -> list[dict[str, Any]]:
    """Route academic and social overhead exports without thinning saved paths."""
    trajectory_sets = _load_pair_trajectories(request.context)
    if any(not trajectory_sets[name] for name in trajectory_sets):
        return [
            _unavailable_pair_figure(
                request.context,
                "paired_overhead",
                "saved_illustrative_trajectory_samples",
                "one source lacks saved raw trajectory samples",
            )
        ]
    formats = _pair_export_formats("overhead", request.figure_kinds)
    include_social = (
        request.figure_kinds is None or "overhead_social_png" in request.figure_kinds
    )
    if not formats and not include_social:
        return []
    paths = namespace_figure_paths(
        "overhead",
        _write_pair_overhead_figure(
            trajectory_sets,
            context=request.context,
            output_stem=request.stem.with_name(request.stem.name + "_overhead"),
            academic_formats=formats,
            include_social=include_social,
        ),
    )
    return _pair_overhead_ledger(request, paths, trajectory_sets)


def _render_pair_figures_core(
    pair: dict[str, Any],
    output_dir: Path,
    *,
    figure_kinds: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    first, second = pair["first"], pair["second"]
    if first.get("status") != "completed" or second.get("status") != "completed":
        return []
    if first.get("source_protocol") is None or second.get("source_protocol") is None:
        raise ValueError(
            f"pair references lack source protocol: {first.get('cell')} vs {second.get('cell')}"
        )
    comparison, _, _, _, _, target = pair_comparison_identity(pair)
    labels = (
        _pair_condition_label(comparison, first),
        _pair_condition_label(comparison, second),
    )
    request = _PairExportRequest(
        PairFigureContext(first, second, comparison, labels, target),
        output_dir,
        output_dir / "comparisons" / pair_comparison_slug(pair),
        figure_kinds,
    )
    ledger: list[dict[str, Any]] = []
    if figure_kinds is None or bool(
        {"density_png", "density_pdf", "density_svg"} & figure_kinds
    ):
        ledger.extend(_render_pair_density(request))
    if figure_kinds is None or bool(
        {"overhead_png", "overhead_pdf", "overhead_svg", "overhead_social_png"}
        & figure_kinds
    ):
        ledger.extend(_render_pair_overhead(request))
    return ledger


def _render_pair_figures(
    pair: dict[str, Any], output_dir: Path
) -> list[dict[str, Any]]:
    return _render_pair_figures_core(pair, output_dir, figure_kinds=None)


def render_selected_pair_figures(
    pair: dict[str, Any],
    output_dir: Path,
    *,
    figure_kinds: frozenset[str],
) -> list[dict[str, Any]]:
    """Export explicitly requested subset of pair figures without extra files."""
    if not isinstance(figure_kinds, frozenset):
        raise TypeError(
            f"figure_kinds must be a frozenset[str], got {type(figure_kinds).__name__}"
        )
    if not figure_kinds:
        raise ValueError("figure_kinds must not be empty")
    unknown = figure_kinds - CLOSED_SELECTED_FIGURE_KINDS
    if unknown:
        raise ValueError(
            f"unsupported figure_kinds: {sorted(unknown)}; "
            f"must be a subset of {sorted(CLOSED_SELECTED_FIGURE_KINDS)}"
        )
    return _render_pair_figures_core(pair, output_dir, figure_kinds=figure_kinds)
