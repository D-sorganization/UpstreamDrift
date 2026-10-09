"""Plotting helpers for the additive matched-delivery supplement report."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .academic_plots import (
    write_academic_dispersion_figure,
    write_overhead_trajectory_figure,
)
from .density import shared_histogram_density
from .matched_matrix import PROFILES
from .matched_report import load_individually_supported_endpoints
from .matched_sensitivity import _supported_endpoints
from .provenance import manifest_files_match

_PATTERNS = ("Straight", "Draw", "Fade")


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

def _condition_caption(ref: dict[str, Any]) -> str:
    condition = _condition(ref)
    club = str(condition["club"]).replace("_", " ").title()
    return (
        f"{club} · {condition['profile']} · Lean {condition['lean']:g}° · "
        f"Curve ×{condition['curve']:g} · Shaft Twist SD {condition['rotation']:g}°"
    )

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
        "stage": "Final Supplemental Analysis",
        "figure_scope": label,
        "valid_shots_by_pattern": valid_counts,
        "total_shots_by_pattern": total,
        "conditioning": "individually supported endpoints by physical pattern",
    }

def _render_cell_figures(ref: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    if ref.get("status") != "completed":
        return []
    cell_dir = Path(ref["source_root"]) / ref["cell"]
    manifest_path = cell_dir / "manifest.json"
    if not manifest_files_match(manifest_path):
        raise ValueError(f"figure source manifest is invalid: {ref['cell']}")
    points, _ = load_individually_supported_endpoints(
        cell_dir / "shots.csv.gz", int(ref["config"]["n_shots"])
    )
    arrays = {name: points[name] for name in _PATTERNS}
    if any(not arrays[name] for name in _PATTERNS):
        return [
            {
                "figure_kind": "density_and_overhead",
                "source_kind": "unique_supplement_cell",
                "source_protocol": ref["source_protocol"],
                "cell": ref.get("alias_cell", ref["cell"]),
                "status": "unavailable",
                "reason": "at least one physical pattern has no supported endpoints",
            }
        ]
    target_x = float(ref["summary"]["target_x_m"])
    label = str(ref.get("alias_cell", ref["cell"]))
    stem = output_dir / "cells" / _slug(label)
    valid_counts = {name: len(arrays[name]) for name in _PATTERNS}
    metadata = _plot_metadata(ref, "one-cell supported population", valid_counts)
    density_paths = write_academic_dispersion_figure(
        arrays,
        stem.with_name(stem.name + "_density"),
        scenario=f"Supplement Cell: {label}",
        metadata=metadata,
        target_point_m=(target_x, 0.0),
    )
    flights_path = cell_dir / "flight_samples.json"
    if not flights_path.is_file():
        raise ValueError(f"figure source has no flight samples: {ref['cell']}")
    flight_record = json.loads(flights_path.read_text(encoding="utf-8"))
    trajectory_paths = flight_record.get("trajectories")
    if not isinstance(trajectory_paths, dict) or set(trajectory_paths) != set(
        _PATTERNS
    ):
        raise ValueError(f"flight samples lack pattern trajectories: {ref['cell']}")
    overhead_paths = write_overhead_trajectory_figure(
        trajectory_paths,
        stem.with_name(stem.name + "_overhead"),
        target_x_m=target_x,
        scenario=f"Supplement Cell: {label}; Raw Simulator Flight Samples",
        metadata=metadata,
    )
    ledger = []
    for figure_group, paths in (
        ("density", density_paths),
        ("overhead", overhead_paths),
    ):
        for kind, path in paths.items():
            ledger.append(
                {
                    "figure_kind": f"{figure_group}_{kind}",
                    "source_kind": (
                        "individual_supported_endpoints"
                        if figure_group == "density"
                        else "saved_illustrative_trajectories"
                    ),
                    "source_protocol": ref["source_protocol"],
                    "cell": label,
                    "target_x_m": target_x,
                    "valid_n_by_pattern": (
                        metadata["valid_shots_by_pattern"]
                        if figure_group == "density"
                        else None
                    ),
                    "total_n_by_pattern": (
                        metadata["total_shots_by_pattern"]
                        if figure_group == "density"
                        else None
                    ),
                    "saved_trajectory_samples_by_pattern": (
                        {name: len(trajectory_paths[name]) for name in _PATTERNS}
                        if figure_group == "overhead"
                        else None
                    ),
                    "relative_path": str(path.relative_to(output_dir)),
                    "sha256": _sha256(path),
                }
            )
    return ledger

def _slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_.-")
    if not clean:
        raise ValueError("figure label has no safe filename characters")
    return f"{clean[:160]}_{hashlib.sha256(value.encode()).hexdigest()[:10]}"

def _save_figure(figure: Any, stem: Path) -> dict[str, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix, dpi in (("png", 300), ("pdf", None), ("svg", None)):
        path = stem.with_suffix(f".{suffix}")
        options: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi is not None:
            options["dpi"] = dpi
        figure.savefig(path, **options)
        outputs[suffix] = path
    social_path = stem.with_name(stem.name + "_social.png")
    figure.savefig(
        social_path,
        dpi=120,
        facecolor="white",
    )
    outputs["social_png"] = social_path
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

def _write_pair_density_figure(
    point_sets: dict[str, list[tuple[float, float]]],
    *,
    first: dict,
    second: dict,
    comparison: str,
    target_x: float,
    common_n: int,
    output_stem: Path,
) -> dict[str, Path]:
    """Plot two conditions in six panels on common bins and density scale."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize

    estimate = shared_histogram_density(
        point_sets, bins=64, target_point_m=(target_x, 0.0)
    )
    figure = plt.figure(figsize=(16, 9), facecolor="white")
    grid = figure.add_gridspec(
        2, 3, left=0.06, right=0.91, top=0.76, bottom=0.13, hspace=0.24, wspace=0.24
    )
    first_condition, second_condition = _condition(first), _condition(second)
    row_conditions = (("First", first_condition), ("Second", second_condition))
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
            color_mesh, ax=figure.axes, fraction=0.018, pad=0.025
        )
        colorbar.set_label("Endpoint Probability Density (m⁻²); Common Scale")
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
    figure.legend(
        handles=hdr,
        loc="upper center",
        bbox_to_anchor=(0.48, 0.83),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Matched Endpoint Density: Two Conditions, Shared Axes and Density Scale",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    total_n = int(first["config"]["n_shots"])
    figure.text(
        0.06,
        0.915,
        f"{comparison} · Target {target_x:.3f} m · Density uses all {common_n:,}/{total_n:,} shots common to six populations; {min(common_n, 600):,} evenly spaced dots per panel",
        fontsize=8.5,
    )
    figure.text(
        0.06,
        0.895,
        f"First: {_condition_caption(first)} · {first['source_protocol']}",
        fontsize=7.7,
    )
    figure.text(
        0.06,
        0.875,
        f"Second: {_condition_caption(second)} · {second['source_protocol']}",
        fontsize=7.7,
    )
    figure.text(
        0.06,
        0.855,
        "Illustrative Simulation | Assumed Head Properties | Carry Only · Raw coordinates; no aim correction. Matched-condition changes include nominal re-matching.",
        fontsize=9,
    )
    outputs = _save_figure(figure, output_stem)
    plt.close(figure)
    return outputs

def _write_pair_overhead_figure(
    trajectory_sets: dict[str, list[Any]],
    *,
    first: dict,
    second: dict,
    comparison: str,
    target_x: float,
    output_stem: Path,
) -> dict[str, Path]:
    """Compare raw saved trajectory samples in a shared two-row layout."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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
    figure, axes = plt.subplots(
        2, 3, figsize=(16, 9), sharex=True, sharey=True, facecolor="white"
    )
    figure.subplots_adjust(
        top=0.77, bottom=0.12, left=0.07, right=0.98, wspace=0.16, hspace=0.28
    )
    conditions = (("First", first), ("Second", second))
    colors = {"Straight": "#286090", "Draw": "#c66a19", "Fade": "#7b4a9e"}
    for row, (condition_name, ref) in enumerate(conditions):
        condition = _condition(ref)
        for col, pattern in enumerate(_PATTERNS):
            axis = axes[row, col]
            key = f"{pattern} · {condition_name}"
            trajectories = trajectory_sets.get(key, [])
            for path in trajectories:
                array = np.asarray(path, dtype=float)
                axis.plot(
                    array[:, 0],
                    array[:, 1],
                    color=colors[pattern],
                    alpha=0.4,
                    linewidth=0.8,
                    rasterized=True,
                )
            axis.scatter([target_x], [0.0], marker="+", color="black", s=32)
            axis.axvline(target_x, color="#444444", linestyle=":", linewidth=0.7)
            axis.axhline(0.0, color="#444444", linewidth=0.5)
            axis.set_xlim(float(lows[0]), float(highs[0]))
            axis.set_ylim(float(lows[1]), float(highs[1]))
            axis.set_title(
                f"{pattern} · {condition_name}",
                fontsize=10,
            )
            axis.set_xlabel("Downrange Position (m)")
            if col == 0:
                axis.set_ylabel("Lateral Position (m; Right Positive)")
            axis.grid(alpha=0.2, linewidth=0.4)
            axis.set_aspect("equal", adjustable="box")
    figure.suptitle(
        "Raw Overhead Flight Samples by Matched Condition",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.06,
        0.925,
        f"{comparison} · saved illustrative trajectories by condition · target {target_x:.3f} m",
        fontsize=8.5,
    )
    figure.text(
        0.06,
        0.905,
        f"First: {_condition_caption(first)} · {first['source_protocol']}",
        fontsize=7.7,
    )
    figure.text(
        0.06,
        0.885,
        f"Second: {_condition_caption(second)} · {second['source_protocol']}",
        fontsize=7.7,
    )
    path_counts = {key: len(value) for key, value in trajectory_sets.items()}
    first_counts = ", ".join(
        f"{pattern} {path_counts[f'{pattern} · First']}" for pattern in _PATTERNS
    )
    second_counts = ", ".join(
        f"{pattern} {path_counts[f'{pattern} · Second']}" for pattern in _PATTERNS
    )
    figure.text(
        0.06,
        0.865,
        f"Saved trajectory paths (not paired by index): First {first_counts} | Second {second_counts}",
        fontsize=8,
    )
    figure.text(
        0.06,
        0.845,
        "Illustrative Simulation | Assumed Head Properties | Carry Only · Raw paths are shown by condition, not paired by shot index.",
        fontsize=8.5,
    )
    handles = [
        Line2D([0], [0], color=colors[pattern], label=pattern) for pattern in _PATTERNS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.48, 0.825),
        ncol=3,
        frameon=False,
    )
    outputs = _save_figure(figure, output_stem)
    plt.close(figure)
    return outputs

def _render_pair_figures(
    pair: dict[str, Any], output_dir: Path
) -> list[dict[str, Any]]:
    first, second = pair["first"], pair["second"]
    if first.get("status") != "completed" or second.get("status") != "completed":
        return []
    loaded = []
    for ref in (first, second):
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
    comparison = pair["comparison_kind"]
    label = f"{comparison}_{first['cell']}_vs_{second['cell']}"
    stem = output_dir / "comparisons" / _slug(label)
    target = pair["target_x_m"]
    ledger: list[dict[str, Any]] = []
    if len(common) >= 2:
        point_sets: dict[str, list[tuple[float, float]]] = {}
        for condition_label, endpoint_map in zip(("First", "Second"), loaded, strict=True):
            by_pattern = {
                name: {idx: (x, y) for idx, x, y in endpoint_map[name]}
                for name in _PATTERNS
            }
            for pattern in _PATTERNS:
                name = f"{pattern} · {condition_label}"
                point_sets[name] = [by_pattern[pattern][index] for index in common]
        density_paths = namespace_figure_paths(
            "density",
            _write_pair_density_figure(
                point_sets,
                first=first,
                second=second,
                comparison=comparison,
                target_x=target,
                common_n=len(common),
                output_stem=stem.with_name(stem.name + "_density"),
            ),
        )
        ledger.extend(
            {
                "figure_kind": kind,
                "source_kind": "six_population_common_index_comparison",
                "comparison_kind": comparison,
                "first_cell": first["cell"],
                "second_cell": second["cell"],
                "first_protocol": first["source_protocol"],
                "second_protocol": second["source_protocol"],
                "common_supported_n": len(common),
                "total_n_per_population": first["config"]["n_shots"],
                "conditioning": "same shot indices supported in all six populations",
                "target_x_m": target,
                "relative_path": str(path.relative_to(output_dir)),
                "sha256": _sha256(path),
            }
            for kind, path in density_paths.items()
        )
    else:
        ledger.append(
            {
                "figure_kind": "paired_density",
                "source_kind": "six_population_common_index_comparison",
                "comparison_kind": comparison,
                "first_cell": first["cell"],
                "second_cell": second["cell"],
                "status": "unavailable",
                "reason": "fewer than two indices supported in all six populations",
            }
        )
    trajectory_sets: dict[str, list[Any]] = {}
    for label_condition, ref in zip(("First", "Second"), (first, second), strict=True):
        directory = Path(ref["source_root"]) / ref["cell"]
        flight = json.loads((directory / "flight_samples.json").read_text())
        paths_by_pattern = flight.get("trajectories", {})
        for pattern in _PATTERNS:
            key = f"{pattern} · {label_condition}"
            trajectory_sets[key] = paths_by_pattern.get(pattern, [])
    if any(not trajectory_sets[name] for name in trajectory_sets):
        ledger.append(
            {
                "figure_kind": "paired_overhead",
                "source_kind": "saved_illustrative_trajectory_samples",
                "comparison_kind": comparison,
                "first_cell": first["cell"],
                "second_cell": second["cell"],
                "status": "unavailable",
                "reason": "one source lacks saved raw trajectory samples",
            }
        )
    else:
        overhead_paths = namespace_figure_paths(
            "overhead",
            _write_pair_overhead_figure(
            trajectory_sets,
            first=first,
            second=second,
            comparison=comparison,
            target_x=target,
            output_stem=stem.with_name(stem.name + "_overhead"),
            ),
        )
        ledger.extend(
            {
                "figure_kind": kind,
                "source_kind": "saved_illustrative_trajectory_samples",
                "comparison_kind": comparison,
                "first_cell": first["cell"],
                "second_cell": second["cell"],
                "first_protocol": first["source_protocol"],
                "second_protocol": second["source_protocol"],
                "conditioning": "saved raw trajectories, not paired by shot index",
                "target_x_m": target,
                "relative_path": str(path.relative_to(output_dir)),
                "sha256": _sha256(path),
            }
            for kind, path in overhead_paths.items()
        )
    return ledger
