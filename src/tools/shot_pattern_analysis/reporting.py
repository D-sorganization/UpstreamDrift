"""Reproducible data export and social-size shot pattern figures."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.shared.python.physics.ball_launch_conditions import EnvironmentalConditions
from src.shared.python.physics.ball_properties import BallProperties
from src.shared.python.physics.impact_model import ImpactParameters

from .core import PATTERNS, AnalysisConfig, AnalysisResult, ShotRecord
from .dispersion_stats import landing_dispersion
from .range_control import equal_range_endpoint, equal_range_summary
from .uncertainty import paired_hit_difference, paired_variance_ratio

_COLORS = {"Straight": "#83d7ff", "Draw": "#f8bf58", "Fade": "#e98ac0"}
_LIMITATION = (
    "Simplified central rigid-body impulse and shaft-axis delivery geometry; "
    "no measured player covariance, off-center gear physics, or ground roll. "
    "Results are model-conditional, not observed player performance."
)


def _scenario_footer(config: AnalysisConfig) -> str:
    return (
        f"Model-Conditional | {config.club_id} | {config.club_speed_mps:g} m/s | "
        f"Loft {config.loft_deg:g}° | AoA {config.attack_angle_deg:+g}° | "
        f"Face SD {config.face_sd_deg:g}° | Curve ×{config.curve_scale:g} | "
        f"{config.delivery_mode}, lie {config.lie_deg:g}°, lean {config.shaft_lean_deg:g}°"
    )


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _variance_comparisons(result: AnalysisResult) -> dict[str, dict]:
    """Compare paired lateral variance against straight, retaining unavailable cases."""
    grouped = {
        pattern.name: [row for row in result.shots if row.pattern == pattern.name]
        for pattern in PATTERNS
    }
    comparisons: dict[str, dict[str, dict[str, float | str | None]]] = {}
    for name in ("Draw", "Fade"):
        comparisons[name] = {}
        for label, field in (("raw", "carry_y_m"), ("aimed", "aimed_y_m")):
            reference = [getattr(row, field) for row in grouped["Straight"]]
            curved = [getattr(row, field) for row in grouped[name]]
            if len(reference) < 10 or np.var(reference, ddof=1) == 0:
                comparisons[name][label] = {
                    "estimate": None,
                    "reason": "Requires >=10 pairs and positive straight variance",
                }
            else:
                comparisons[name][label] = cast(
                    dict[str, float | str | None],
                    paired_variance_ratio(curved, reference, seed=result.config.seed),
                )
    return comparisons


def _hit_comparisons(result: AnalysisResult) -> dict[str, dict]:
    """Bootstrap paired curved-minus-straight target-hit differences."""
    grouped = {
        name: [row for row in result.shots if row.pattern == name] for name in _COLORS
    }
    comparisons: dict[str, dict[str, dict[str, float | str | None]]] = {}
    for name in ("Draw", "Fade"):
        comparisons[name] = {}
        for label, x_field, y_field in (
            ("raw", "carry_x_m", "carry_y_m"),
            ("aimed", "aimed_x_m", "aimed_y_m"),
        ):
            if len(grouped["Straight"]) < 10:
                comparisons[name][label] = {
                    "estimate": None,
                    "reason": "Requires >=10 paired shots",
                }
            else:
                comparisons[name][label] = cast(
                    dict[str, float | str | None],
                    paired_hit_difference(
                        _hit_flags(grouped[name], x_field, y_field, result),
                        _hit_flags(grouped["Straight"], x_field, y_field, result),
                        seed=result.config.seed,
                    ),
                )
    return comparisons


def _hit_flags(
    rows: list[ShotRecord], x_field: str, y_field: str, result: AnalysisResult
) -> np.ndarray:
    return np.array(
        [
            int(
                np.hypot(
                    getattr(row, x_field) - result.target_x_m,
                    getattr(row, y_field),
                )
                <= result.config.target_radius_m
            )
            for row in rows
        ]
    )


def _pattern_label(name: str, config: AnalysisConfig) -> str:
    pattern = next(p for p in PATTERNS if p.name == name)
    return (
        f"{name}  Face {pattern.face_deg * config.curve_scale:+g}° / "
        f"Path {pattern.path_deg * config.curve_scale:+g}°"
    )


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.facecolor": "#122641",
            "figure.facecolor": "#091b30",
            "axes.labelcolor": "#e5edf6",
            "xtick.color": "#bdcbda",
            "ytick.color": "#bdcbda",
            "text.color": "#e5edf6",
            "grid.color": "#45617b",
            "axes.edgecolor": "#6b8094",
        }
    )


def _save_flight(result: AnalysisResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.79, bottom=0.18)
    fig.text(0.07, 0.94, "Shot Shape: Overhead Flight", fontsize=28, weight="bold")
    fig.text(
        0.07,
        0.885,
        f"Straight vs. Draw vs. Fade  |  {result.config.n_shots:,} Simulated Shots per Pattern",
        fontsize=15,
        color="#bed0df",
    )
    for name, paths in result.flight_samples.items():
        color = _COLORS[name]
        for points in paths[1:]:
            if not points:
                continue
            arr = np.asarray(points)
            ax.plot(arr[:, 0], arr[:, 1], color=color, alpha=0.25, lw=1.1)
        nominal = np.asarray(paths[0])
        if len(nominal):
            ax.plot(
                nominal[:, 0],
                nominal[:, 1],
                color=color,
                lw=3.2,
                label=_pattern_label(name, result.config),
            )
            ax.scatter(
                nominal[-1, 0],
                nominal[-1, 1],
                s=90,
                color=color,
                edgecolors="white",
                zorder=5,
            )
    ax.axhline(0, color="#aec3d4", lw=1, linestyle="--", alpha=0.7)
    ax.axvline(result.target_x_m, color="#aec3d4", lw=1, linestyle=":", alpha=0.5)
    ax.set_xlabel("Downrange Carry (m)")
    ax.set_ylabel("Lateral Carry (m; Right Positive)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", facecolor="#183754", edgecolor="#4c6378")
    fig.text(
        0.07,
        0.105,
        f"Nominal path plus up to 9 sampled flights per pattern shown. All {result.config.n_shots:,} per pattern enter the dispersion statistics. Lateral scale expanded.",
        fontsize=11,
        color="#bfd2df",
    )
    fig.text(
        0.07,
        0.065,
        _scenario_footer(result.config),
        fontsize=11,
        color="#f4b986",
    )
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def _equal_range_diagnostic(result: AnalysisResult) -> dict:
    grouped = {
        name: [row for row in result.shots if row.pattern == name] for name in _COLORS
    }
    endpoints = {
        name: [
            equal_range_endpoint(row.aimed_x_m, row.aimed_y_m, result.target_x_m)
            for row in rows
        ]
        for name, rows in grouped.items()
    }
    patterns = {
        name: equal_range_summary(
            [(row.aimed_x_m, row.aimed_y_m) for row in rows],
            result.target_x_m,
            result.config.target_radius_m,
        )
        for name, rows in grouped.items()
    }
    paired: dict[str, dict[str, float | str | None]] = {}
    reference = [y for _, y in endpoints["Straight"]]
    for name in ("Draw", "Fade"):
        curved = [y for _, y in endpoints[name]]
        if len(reference) < 10 or np.var(reference, ddof=1) == 0:
            paired[name] = {
                "estimate": None,
                "reason": "Requires >=10 pairs and positive straight variance",
            }
        else:
            paired[name] = cast(
                dict[str, float | str | None],
                paired_variance_ratio(curved, reference, seed=result.config.seed),
            )
    return {
        "method": "Geometric radial normalization of aimed carry endpoints; no impact or flight re-simulation",
        "target_range_m": result.target_x_m,
        "patterns": patterns,
        "paired_variance_ratios": paired,
    }


def _save_dispersion(
    result: AnalysisResult, path: Path, *, equal_range: bool = False
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=120, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.97, top=0.79, bottom=0.20, wspace=0.09)
    title = (
        "Equal-Range Dispersion Diagnostic"
        if equal_range
        else "Shot Dispersion at Carry"
    )
    fig.text(0.07, 0.94, title, fontsize=28, weight="bold")
    fig.text(
        0.07,
        0.885,
        (
            f"Geometric Equal-Range Diagnostic — Not Re-Simulated  |  All Shots Set to {result.target_x_m:.2f} m"
            if equal_range
            else f"{result.config.n_shots:,} Shots Each  |  Face Variation SD {result.config.face_sd_deg:g}°  |  Nominal Aim Adjusted"
        ),
        fontsize=15,
        color="#bed0df",
    )
    fig.text(
        0.07,
        0.835,
        "  |  ".join(_pattern_label(name, result.config) for name in _COLORS),
        fontsize=11,
        color="#d1deeb",
    )
    all_rows = {
        name: [r for r in result.shots if r.pattern == name] for name in _COLORS
    }
    points = {
        name: [
            equal_range_endpoint(r.aimed_x_m, r.aimed_y_m, result.target_x_m)
            if equal_range
            else (r.aimed_x_m, r.aimed_y_m)
            for r in rows
        ]
        for name, rows in all_rows.items()
    }
    lateral_bound = max(abs(y) for rows in points.values() for _, y in rows)
    lateral_bound = max(25.0, lateral_bound * 1.05)
    downrange_bound = max(
        25.0,
        max(abs(x - result.target_x_m) for rows in points.values() for x, _ in rows)
        * 1.05,
    )
    for ax, (name, rows) in zip(axes, all_rows.items(), strict=True):
        color = _COLORS[name]
        x = [y for _, y in points[name]]
        y = [x - result.target_x_m for x, _ in points[name]]
        ax.scatter(x, y, s=4, color=color, alpha=0.10, linewidths=0, rasterized=True)
        ax.add_patch(
            Circle(
                (0, 0),
                result.config.target_radius_m,
                fill=False,
                color="#d9e9f7",
                lw=1.6,
                ls="--",
            )
        )
        ax.plot(0, 0, marker="+", markersize=14, markeredgewidth=2, color="white")
        ax.set_title(name, fontsize=20, weight="bold", color=color, pad=16)
        ax.set_xlim(-lateral_bound, lateral_bound)
        ax.set_ylim(-downrange_bound, downrange_bound)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlabel("Lateral (m; Right Positive)")
        stats = (
            equal_range_summary(
                [(r.aimed_x_m, r.aimed_y_m) for r in rows],
                result.target_x_m,
                result.config.target_radius_m,
            )
            if equal_range
            else result.summary[name]
        )
        sd_key = "lateral_sd_m" if equal_range else "aimed_lateral_sd_m"
        hit_key = "target_hit_fraction" if equal_range else "aimed_target_hit_fraction"
        sd_text = f"{stats[sd_key]:.2f}" if equal_range else f"{stats[sd_key]:.1f}"
        hit_text = f"{stats[hit_key]:.2%}" if equal_range else f"{stats[hit_key]:.1%}"
        ax.text(
            0.04,
            0.04,
            f"Lateral SD  {sd_text} m\n"
            f"{result.config.target_radius_m:g} m Circle  {hit_text}",
            transform=ax.transAxes,
            va="bottom",
            fontsize=11,
            bbox={"facecolor": "#091b30", "edgecolor": "none", "alpha": 0.85, "pad": 8},
        )
    axes[0].set_ylabel("Downrange Error (m)")
    fig.text(
        0.07,
        0.115,
        (
            "Each existing aimed landing is radially scaled to the straight nominal carry. Bearing is preserved; no flight is re-simulated."
            if equal_range
            else "Each point is one simulated landing. Aim rotates each nominal pattern onto the target line; target distance is fixed."
        ),
        fontsize=11,
        color="#bfd2df",
    )
    fig.text(
        0.07,
        0.075,
        _scenario_footer(result.config),
        fontsize=11,
        color="#f4b986",
    )
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def export_analysis(result: AnalysisResult, output_dir: Path) -> dict[str, Path]:
    """Write raw results, summary, provenance, and three shareable PNG figures."""
    if not isinstance(result, AnalysisResult):
        raise TypeError("result must be AnalysisResult")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "shots_csv": output_dir / "shots.csv",
        "summary_json": output_dir / "summary.json",
        "receipt_json": output_dir / "receipt.json",
        "overhead_png": output_dir / "overhead_flight.png",
        "dispersion_png": output_dir / "dispersion.png",
        "dispersion_equal_range_png": output_dir / "dispersion_equal_range.png",
    }
    with paths["shots_csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(ShotRecord)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in result.shots)
    summary = {
        "title": "Shot Pattern Analysis: Straight, Draw, and Fade",
        "config": asdict(result.config),
        "target_x_m": result.target_x_m,
        "patterns": result.summary,
        "nominal_outcomes": {
            name: asdict(outcome) for name, outcome in result.nominal_outcomes.items()
        },
        "paired_variance_comparisons": _variance_comparisons(result),
        "paired_hit_comparisons": _hit_comparisons(result),
        "equal_range_diagnostic": _equal_range_diagnostic(result),
        "landing_dispersion": {
            name: landing_dispersion(
                [
                    (row.aimed_x_m, row.aimed_y_m)
                    for row in result.shots
                    if row.pattern == name
                ],
                result.target_x_m,
                nominal_point=(result.target_x_m, 0.0),
            )
            for name in _COLORS
        },
        "bootstrap": {
            "method": "Paired percentile bootstrap; curved/straight lateral variance",
            "resamples": 500,
            "confidence": 0.95,
            "scope": "Monte Carlo sampling only; excludes model and parameter uncertainty",
        },
        "limitations": [_LIMITATION],
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2) + "\n")
    package_dir = Path(__file__).parent
    repo_root = Path(__file__).resolve().parents[3]
    source_files: tuple[Path, ...] = (
        package_dir / "core.py",
        package_dir / "physics.py",
        package_dir / "delivery_geometry.py",
        package_dir / "presets.py",
        package_dir / "dispersion_stats.py",
        package_dir / "scoring.py",
        package_dir / "scoring_cache.py",
        package_dir / "scenario_scoring.py",
        package_dir / "reporting.py",
        package_dir / "range_control.py",
        repo_root / "src/shared/python/physics/impact_model/models.py",
        repo_root / "src/shared/python/physics/impact_model/solver.py",
        repo_root / "src/shared/python/physics/impact_model/types.py",
        repo_root / "src/shared/python/physics/ball_simulator.py",
        repo_root / "src/shared/python/physics/ball_launch_conditions.py",
        repo_root / "src/shared/python/physics/ball_properties.py",
        repo_root / "src/shared/python/core/physics_constants.py",
        repo_root / "rust_core/upstream-physics/Cargo.toml",
        repo_root / "Cargo.lock",
    )
    source_files += tuple(
        sorted((repo_root / "rust_core/upstream-physics/src").glob("*.rs"))
    )
    optional_uncertainty = package_dir / "uncertainty.py"
    source_files += (optional_uncertainty,) if optional_uncertainty.exists() else ()
    rust_spec = importlib.util.find_spec("upstream_physics")
    rust_binary = None
    if rust_spec and rust_spec.origin:
        candidate = Path(rust_spec.origin)
        if candidate.suffix == ".so":
            rust_binary = candidate
        else:
            binaries = sorted(candidate.parent.glob("*.so"))
            rust_binary = binaries[0] if binaries else None
    try:
        rust_version = importlib.metadata.version("upstream-physics")
    except importlib.metadata.PackageNotFoundError:
        rust_version = None
    receipt = {
        "seed": result.config.seed,
        "shots_per_pattern": result.config.n_shots,
        "face_distribution": f"Independent normal deviations N(0, {result.config.face_sd_deg:g}°), paired across patterns",
        "physics": {
            "impact_model": "rigid_body",
            "flight_model": "BallFlightSimulator (upstream_physics Rust RK4)",
            "club_attack_deg": result.config.attack_angle_deg,
            "clubhead_mass_kg": result.config.clubhead_mass_kg,
            "delivery_mode": result.config.delivery_mode,
            "lie_deg": result.config.lie_deg,
            "shaft_lean_deg": result.config.shaft_lean_deg,
            "contact_offset_m": [0.0, 0.0],
            "environment": "Default EnvironmentalConditions, zero wind",
            "rust_package_version": rust_version,
            "rust_binary_sha256": _source_hash(rust_binary)
            if rust_binary and rust_binary.is_file()
            else None,
            "impact_parameters": asdict(ImpactParameters()),
            "ball_properties": {
                "mass_kg": BallProperties().mass,
                "radius_m": BallProperties().radius,
            },
            "environment_values": {
                "air_density_kg_m3": EnvironmentalConditions().air_density,
                "gravity_m_s2": EnvironmentalConditions().gravity,
                "temperature_c": EnvironmentalConditions().temperature,
                "wind_velocity_mps": EnvironmentalConditions().wind_velocity.tolist(),
            },
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": {
            str(p.relative_to(repo_root)): _source_hash(p) for p in source_files
        },
        "limitations": [
            _LIMITATION,
            "No player-specific face/path covariance, speed/loft variation, wind, or ground roll.",
        ],
    }
    paths["receipt_json"].write_text(json.dumps(receipt, indent=2) + "\n")
    _save_flight(result, paths["overhead_png"])
    _save_dispersion(result, paths["dispersion_png"])
    _save_dispersion(result, paths["dispersion_equal_range_png"], equal_range=True)
    return paths


def augment_saved_bundle_with_equal_range(output_dir: Path) -> Path:
    """Add the geometric equal-range diagnostic to a completed raw-data bundle.

    This reads saved shot endpoints and does not run impact or flight again.
    Existing numerical outcomes, figures, and receipt are preserved.
    """
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    shots_path = output_dir / "shots.csv"
    summary = json.loads(summary_path.read_text())
    with shots_path.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = tuple(
        ShotRecord(
            **cast(
                Any,
                {
                    field.name: (
                        raw[field.name]
                        if field.name == "pattern"
                        else int(raw[field.name])
                        if field.name == "shot_index"
                        else float(raw[field.name])
                    )
                    for field in fields(ShotRecord)
                },
            )
        )
        for raw in raw_rows
    )
    config = AnalysisConfig(**summary["config"])
    if len(rows) != len(PATTERNS) * config.n_shots:
        raise ValueError("saved shots.csv does not contain the expected shot count")
    result = AnalysisResult(
        config=config,
        shots=rows,
        summary=summary["patterns"],
        flight_samples={},
        target_x_m=float(summary["target_x_m"]),
        nominal_outcomes={},
    )
    summary["equal_range_diagnostic"] = _equal_range_diagnostic(result)
    summary["paired_hit_comparisons"] = _hit_comparisons(result)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    graphic = output_dir / "dispersion_equal_range.png"
    _save_dispersion(result, output_dir / "dispersion.png")
    _save_dispersion(result, graphic, equal_range=True)
    return graphic
