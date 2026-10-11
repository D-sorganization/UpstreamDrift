"""Replot accepted saved shot data in conventional target-view orientation.

This is a presentation-only renderer. Saved coordinates remain [downrange,
lateral] metres; displays use x=lateral (right positive), y=downrange.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

from .summary import render_summary

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PHASE = Path(__file__).resolve().parent.parent
TEX = PHASE.parent / "shot_pattern_analysis.tex"
OUT = Path(__file__).resolve().parent
PATTERNS = ("Straight", "Draw", "Fade")
COLORS = {"Straight": "#2563a6", "Draw": "#c76a23", "Fade": "#804c9e"}
CLUBS = {
    "driver": "Driver",
    "seven_iron": "Seven Iron",
    "pitching_wedge": "Pitching Wedge",
}


def orient_points(saved: object) -> np.ndarray:
    """Return a fresh N×2 [lateral, downrange] array from finite saved metres."""
    points = np.asarray(saved, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("saved points must be a nonempty finite N×2 coordinate array")
    return points[:, [1, 0]].copy()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> str:
    """Use stable study-relative source names in the reproducibility mapping."""
    return str(path.relative_to(PHASE))


def _cell_dir(protocol: str, cell: str) -> Path:
    roots = {
        "original_primary/1": PHASE / "final_v1",
        "original_control/1": PHASE / "control_final_v1",
        "pw-primary-face-bounds/1": PHASE / "supplement_final_v1/pw_primary_face",
    }
    if protocol not in roots:
        raise ValueError(f"unmapped source protocol: {protocol}")
    result = roots[protocol] / cell
    if not result.is_dir():
        raise ValueError(f"saved source missing: {result}")
    manifest = json.loads((result / "manifest.json").read_text())
    for name in ("shots.csv.gz", "flight_samples.json"):
        if _sha(result / name) != manifest["files_sha256"][name]:
            raise ValueError(
                f"saved source hash differs from manifest: {result / name}"
            )
    return result


def _endpoints(directory: Path) -> dict[str, dict[int, np.ndarray]]:
    points: dict[str, dict[int, np.ndarray]] = {name: {} for name in PATTERNS}
    with gzip.open(directory / "shots.csv.gz", "rt", newline="") as stream:
        rows = csv.DictReader(stream)
        for row in rows:
            name = row["pattern"]
            if name not in points:
                raise ValueError(f"unknown pattern: {name}")
            if row["status"] == "supported":
                index = int(row["shot_index"])
                if index in points[name]:
                    raise ValueError("duplicate supported shot index")
                points[name][index] = orient_points(
                    [[float(row["carry_x_m"]), float(row["carry_y_m"])]]
                )[0]
    return points


def _density(
    ax: plt.Axes,
    points: np.ndarray,
    target: float,
    color: str,
    limits: tuple[tuple[float, float], tuple[float, float]],
    vmax: float,
) -> None:
    xlim, ylim = limits
    hist, xedges, yedges = np.histogram2d(
        points[:, 0], points[:, 1], bins=55, range=[xlim, ylim], density=True
    )
    ax.pcolormesh(
        xedges,
        yedges,
        hist.T,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        shading="flat",
        rasterized=True,
    )
    masses = ((0.5, "-"), (0.8, "--"), (0.95, ":"))
    order = np.sort(hist.ravel())[::-1]
    cumulative = np.cumsum(order) / order.sum()
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    for mass, style in masses:
        threshold = order[min(np.searchsorted(cumulative, mass), len(order) - 1)]
        if threshold > 0 and threshold < hist.max():
            ax.contour(
                xc,
                yc,
                hist.T,
                levels=[threshold],
                colors="#202b38",
                linestyles=style,
                linewidths=0.8,
            )
    indices = np.linspace(0, len(points) - 1, min(450, len(points)), dtype=int)
    ax.scatter(
        points[indices, 0],
        points[indices, 1],
        s=3.5,
        alpha=0.18,
        color=color,
        rasterized=True,
    )
    ax.plot(0, target, "+", color="#c51f30", ms=10, mew=1.7)
    ax.axvline(0, color="#536577", lw=0.5, alpha=0.55)
    ax.axhline(target, color="#536577", lw=0.5, alpha=0.55)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.16, linewidth=0.5)


def _density_plate(
    groups: list[tuple[str, np.ndarray]],
    target: float,
    title: str,
    note: str,
    destination: Path,
    rows: int = 1,
) -> None:
    ncols = 3
    figure, axes = plt.subplots(
        rows, ncols, figsize=(10.5, 6.0 if rows == 1 else 8.7), constrained_layout=False
    )
    axes = np.asarray(axes).reshape(rows, ncols)
    all_points = np.vstack([p for _, p in groups])
    xhalf = max(float(np.max(np.abs(all_points[:, 0]))), 1.0) * 1.05
    yhalf = max(float(np.max(np.abs(all_points[:, 1] - target))), 1.0) * 1.05
    limits = ((-xhalf, xhalf), (target - yhalf, target + yhalf))
    hist_max = max(
        np.histogram2d(p[:, 0], p[:, 1], bins=55, range=limits, density=True)[0].max()
        for _, p in groups
    )
    for index, (label, points) in enumerate(groups):
        ax = axes.flat[index]
        pattern = label.split(" · ")[0]
        _density(ax, points, target, COLORS[pattern], limits, hist_max)
        ax.set_title(f"{label}  (n={len(points):,})", fontsize=13, pad=8)
        if index // 3 == rows - 1:
            ax.set_xlabel("Lateral Endpoint (m; Right Positive)", fontsize=11)
        if index % 3 == 0:
            ax.set_ylabel("Downrange Endpoint (m)", fontsize=11)
        ax.tick_params(labelsize=10)
    figure.text(0.05, 0.985, title, ha="left", va="top", fontsize=19, fontweight="bold")
    figure.text(0.05, 0.905, note, ha="left", va="top", fontsize=11)
    figure.text(
        0.05,
        0.050,
        "Red + = Matched Target  ·  Solid/Dashed/Dotted = 50/80/95% Binned Empirical HDR  ·  Equal Metre Scaling",
        fontsize=10,
        color="#34465a",
    )
    figure.text(
        0.05,
        0.025,
        "Blue = Probability Density (m⁻²), Shared Scale · Carry Only · Model-Conditional Simulation; Not Mean Confidence Intervals or Calibrated Player Precision.",
        fontsize=9,
        color="#536577",
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.81 if rows == 1 else 0.83,
        bottom=0.17 if rows == 1 else 0.13,
        wspace=0.27,
        hspace=0.25,
    )
    _save(figure, destination)


def _flight_plate(
    groups: list[tuple[str, list]],
    target: float,
    title: str,
    note: str,
    destination: Path,
    *,
    rows: int = 1,
) -> None:
    fig, axes = plt.subplots(rows, 3, figsize=(10.5, 7.0 if rows == 1 else 8.7))
    axes = np.asarray(axes).reshape(rows, 3)
    oriented = [
        (label, [orient_points(path) for path in paths]) for label, paths in groups
    ]
    all_x = np.concatenate([p[:, 0] for _, paths in oriented for p in paths])
    xhalf = max(np.max(np.abs(all_x)) * 1.10, 1.0)
    ymax = max(np.max(p[:, 1]) for _, paths in oriented for p in paths) * 1.06
    for index, (label, paths) in enumerate(oriented):
        ax = axes.flat[index]
        pattern = label.split(" · ")[0]
        for path in paths:
            ax.plot(path[:, 0], path[:, 1], color=COLORS[pattern], lw=1.0, alpha=0.55)
            ax.plot(path[-1, 0], path[-1, 1], "o", color=COLORS[pattern], ms=2.4)
        ax.plot(0, target, "+", color="#c51f30", ms=10, mew=1.7)
        ax.axvline(0, color="#536577", lw=0.6)
        ax.set_xlim(-xhalf, xhalf)
        ax.set_ylim(0, ymax)
        ax.set_title(label, fontsize=13)
        if index // 3 == rows - 1:
            ax.set_xlabel("Lateral Position (m; Right Positive)", fontsize=11)
        if index % 3 == 0:
            ax.set_ylabel("Downrange Distance (m)", fontsize=11)
        ax.grid(alpha=0.16)
    fig.text(0.05, 0.985, title, ha="left", va="top", fontsize=19, fontweight="bold")
    fig.text(0.05, 0.905, note, va="top", fontsize=11)
    fig.text(
        0.05,
        0.034,
        "Red + = Matched Target  ·  Lateral Scale Is Magnified Relative to Downrange; Read Axis Metres  ·  Carry Only",
        fontsize=10,
        color="#34465a",
    )
    fig.text(
        0.05,
        0.012,
        "Saved Illustrative Flight Paths; Not the Full Population or a Paired Path-by-Path Experiment.",
        fontsize=9,
        color="#536577",
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.81 if rows == 1 else 0.83,
        bottom=0.12 if rows == 1 else 0.10,
        wspace=0.25,
        hspace=0.25,
    )
    _save(fig, destination)


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            stem.with_suffix("." + suffix),
            dpi=180 if suffix == "png" else 200,
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close(fig)


def _selected_pair(comparisons: list[dict], club: str, kind: str) -> dict:
    prefix = (
        f"{club}_lean0_curve1_rotation{'2' if kind == 'across_lean' else '1'}_combined2"
    )
    matches = [
        row
        for row in comparisons
        if row["comparison_kind"] == kind
        and row["sources"][0]["cell"] == prefix
        and (
            kind != "across_lean"
            or row["sources"][1]["cell"].startswith(f"{club}_lean5")
        )
    ]
    if len(matches) != 1 or matches[0]["status"] not in (
        "available",
        "conditional_only",
    ):
        raise ValueError(f"expected one available {club} {kind} source pair")
    return matches[0]


def _render_baselines(mapping: list[dict], old: list[str]) -> None:
    for club in CLUBS:
        cell = f"{club}_lean0_curve1_rotation1_combined2"
        directory = _cell_dir("original_primary/1", cell)
        shots = _endpoints(directory)
        flight_path = directory / "flight_samples.json"
        flights = json.loads(flight_path.read_text())
        target = float(flights["target_x_m"])
        counts = [len(shots[p]) for p in PATTERNS]
        label = CLUBS[club]
        density_stem = OUT / f"{club}_baseline_density"
        _density_plate(
            [(p, np.stack(list(shots[p].values()))) for p in PATTERNS],
            target,
            f"{label}: Endpoint Dispersion",
            f"Matched Target {target:.1f} m · Supported {min(counts):,}–{max(counts):,}/10,000 per Pattern · Shared Axes and Density Scale",
            density_stem,
        )
        mapping.append(
            {
                "original": old[len(mapping)],
                "replacement": str(density_stem.relative_to(PHASE)) + ".pdf",
                "sources": {
                    _source(directory / "shots.csv.gz"): _sha(
                        directory / "shots.csv.gz"
                    )
                },
            }
        )
        overhead_stem = OUT / f"{club}_baseline_flight_paths"
        _flight_plate(
            [(p, flights["trajectories"][p]) for p in PATTERNS],
            target,
            f"{label}: Illustrative Flight Paths",
            f"Matched Target {target:.1f} m · Six Saved Paths per Pattern · Zero-Lean Combined-Noise Condition",
            overhead_stem,
        )
        mapping.append(
            {
                "original": old[len(mapping)],
                "replacement": str(overhead_stem.relative_to(PHASE)) + ".pdf",
                "sources": {_source(flight_path): _sha(flight_path)},
            }
        )


def _render_comparisons(
    mapping: list[dict], old: list[str], comparisons: list[dict], report_path: Path
) -> None:
    for club in CLUBS:
        for kind in ("shaft_rotation_sd", "curve_strength"):
            pair = _selected_pair(comparisons, club, kind)
            dirs = [
                _cell_dir(protocol, ref["cell"])
                for protocol, ref in zip(
                    pair["source_protocols"], pair["sources"], strict=True
                )
            ]
            maps = [_endpoints(directory) for directory in dirs]
            common = sorted(
                set.intersection(*(set(m[p]) for m in maps for p in PATTERNS))
            )
            if len(common) != int(pair["common_supported_n"]):
                raise ValueError(f"common support differs for {club} {kind}")
            labels = (
                ("SD 1°", "SD 2°")
                if kind == "shaft_rotation_sd"
                else ("Curve ×1", "Curve ×2")
            )
            groups = [
                (f"{p} · {label}", np.stack([m[p][i] for i in common]))
                for label, m in zip(labels, maps, strict=True)
                for p in PATTERNS
            ]
            title = f"{CLUBS[club]}: {'Increasing Shaft-Rotation Variability' if kind == 'shaft_rotation_sd' else 'Doubling Curve Strength'}"
            stem = OUT / f"{club}_{kind}_density"
            _density_plate(
                groups,
                float(pair["target_x_m"]),
                title,
                f"Two Ordered Conditions · {len(common):,}/10,000 Common Supported Indices Across All Six Populations · Shared Axes and Density Scale",
                stem,
                rows=2,
            )
            mapping.append(
                {
                    "original": old[len(mapping)],
                    "replacement": str(stem.relative_to(PHASE)) + ".pdf",
                    "sources": {
                        _source(d / "shots.csv.gz"): _sha(d / "shots.csv.gz")
                        for d in dirs
                    },
                    "comparison_receipt": _sha(report_path),
                }
            )


def _render_lean_comparison(
    mapping: list[dict], old: list[str], comparisons: list[dict], report_path: Path
) -> None:
    pair = _selected_pair(comparisons, "driver", "across_lean")
    dirs = [
        _cell_dir(protocol, ref["cell"])
        for protocol, ref in zip(pair["source_protocols"], pair["sources"], strict=True)
    ]
    flights = [json.loads((d / "flight_samples.json").read_text()) for d in dirs]
    groups = [
        (f"{p} · Lean {lean}°", flight["trajectories"][p])
        for lean, flight in zip((0, 5), flights, strict=True)
        for p in PATTERNS
    ]
    stem = OUT / "driver_across_lean_flight_paths"
    _flight_plate(
        groups,
        float(pair["target_x_m"]),
        "Driver: Forward-Lean Comparison",
        "Lean 0° Versus 5° · Shaft-Rotation SD 2° · Curve ×1 · Six Saved Paths per Pattern and Condition",
        stem,
        rows=2,
    )
    mapping.append(
        {
            "original": old[len(mapping)],
            "replacement": str(stem.relative_to(PHASE)) + ".pdf",
            "sources": {
                _source(d / "flight_samples.json"): _sha(d / "flight_samples.json")
                for d in dirs
            },
            "comparison_receipt": _sha(report_path),
        }
    )


def render() -> dict:
    """Render exactly 13 replacement plates and one summary; return source-bound mapping."""
    old = json.loads((OUT / "original_paths.json").read_text())
    if len(old) != 13:
        raise ValueError(f"expected 13 embedded source figures, found {len(old)}")
    mapping: list[dict] = []
    _render_baselines(mapping, old)
    report_path = PHASE / "supplement_report_v1/REPORT_SUPPLEMENT_COMPARISONS.json"
    comparisons = json.loads(report_path.read_text())["comparisons"]
    _render_comparisons(mapping, old, comparisons, report_path)
    _render_lean_comparison(mapping, old, comparisons, report_path)
    summary = render_summary(TEX, OUT, _save, COLORS)
    result = {
        "schema": "shot-pattern-presentation-v2/1",
        "display_axes": {"x": "lateral_m_right_positive", "y": "downrange_m"},
        "coordinate_source": "saved [downrange_m, lateral_m] without translation or aim correction",
        "embedded_figure_mapping": mapping,
        "graphical_summary": str(summary.relative_to(PHASE)) + ".pdf",
        "summary_numbers_source": "shot_pattern_analysis.tex: Selected Relative-SG Ranges and Pointwise Interval Directions",
        "presentation_source_sha256": {
            "render.py": _sha(Path(__file__)),
            "summary.py": _sha(OUT / "summary.py"),
        },
    }
    (OUT / "mapping.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    render()
