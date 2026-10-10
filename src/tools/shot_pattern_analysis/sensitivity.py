"""Reproducible deterministic delivery diagnostics using the shared solvers."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
import math
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .core import AnalysisConfig, PhysicsProtocol
from .delivery_geometry import delivery_from_face_angle, delivery_from_shaft_rotation
from .physics import ShotPhysics
from .presets import CLUB_PRESETS


def fixed_club_pitch(*, base_loft_deg: float, lie_deg: float, pitch_deg: float) -> dict:
    """Pitch one rigid face/shaft pair toward the target about world +Y.

    Positive pitch advances the grip and reduces loft. Unlike changing the
    shaft axis while holding delivered loft fixed, this preserves the angle
    between face normal and shaft. World +X is target, +Y left, +Z up.
    Returns geometry only, including negative loft without implying a flight.
    """
    if not math.isfinite(pitch_deg) or not 0 <= pitch_deg < 45:
        raise ValueError("pitch must be finite and in [0, 45) degrees")
    face = delivery_from_shaft_rotation(
        0.0, base_loft_deg=base_loft_deg, lie_deg=lie_deg, shaft_lean_deg=0.0
    )
    shaft = np.array(
        [0.0, math.cos(math.radians(lie_deg)), math.sin(math.radians(lie_deg))]
    )
    rotation = Rotation.from_euler("y", pitch_deg, degrees=True)
    rotated = np.asarray(
        rotation.apply(np.array([face.face_normal, shaft])), dtype=float
    )
    normal, axis = rotated
    return {
        "pitch_deg": pitch_deg,
        "dynamic_loft_deg": math.degrees(
            math.atan2(normal[2], math.hypot(normal[0], normal[1]))
        ),
        "face_angle_deg": math.degrees(math.atan2(-normal[1], normal[0])),
        "shaft_lean_deg": math.degrees(math.atan2(axis[0], axis[2])),
        "shaft_elevation_deg": math.degrees(
            math.atan2(axis[2], math.hypot(axis[0], axis[1]))
        ),
        "normal_dot_shaft": float(np.dot(normal, axis)),
        "face_normal": normal.tolist(),
        "shaft_axis": axis.tolist(),
    }


def _flight(engine: PhysicsProtocol, config: AnalysisConfig, face: float) -> dict:
    outcome = engine.simulate(face_deg=face, path_deg=0.0, config=config)
    return {
        "ball_speed_mps": outcome.ball_speed_mps,
        "launch_elevation_deg": outcome.launch_elevation_deg,
        "launch_azimuth_deg": outcome.launch_azimuth_deg,
        "spin_rpm": outcome.spin_rpm,
        "spin_axis_tilt_deg": outcome.spin_axis_tilt_deg,
        "carry_m": outcome.carry_m,
        "forward_m": outcome.carry_x_m,
        "lateral_right_m": outcome.carry_y_m,
    }


def build_sensitivity(
    configs: Mapping[str, AnalysisConfig], *, engine: PhysicsProtocol | None = None
) -> dict:
    """Evaluate face/axis, loft-only, and distinct rigid-club pitch controls.

    Requires at least one named valid configuration. Face/path are relative
    to a square nominal straight shot; path and speed remain fixed. These
    deterministic values are conditional model predictions, not player data.
    """
    if not configs or any(not isinstance(key, str) or not key for key in configs):
        raise ValueError("at least one named club configuration is required")
    if any(not isinstance(value, AnalysisConfig) for value in configs.values()):
        raise ValueError("every configuration must be an AnalysisConfig")
    solver = engine if engine is not None else ShotPhysics()
    result: dict = {
        "schema": "shot-pattern-deterministic-sensitivity/1",
        "status": "illustrative-corrected-central-impact-model-not-empirically-qualified",
        "controls": {
            "face_errors_deg": [-2, -1, 0, 1, 2],
            "axis_lean_deg": [0, 10, 20],
            "loft_offsets_deg": [-1, -0.5, 0, 0.5, 1],
            "path_deg": 0,
            "face_reference": "error from square nominal straight face",
            "axis_grid": "same nominal delivered loft at every shaft lean; changes local face/loft coupling only",
            "pitch_grid": "rigid world-Y pitch of the same face and shaft; mean loft and shaft elevation both change",
            "carry": "radial airborne distance at first ground contact; no roll",
        },
        "clubs": {},
    }
    for name, original in configs.items():
        base = replace(original, delivery_mode="fixed_loft", shaft_lean_deg=0.0)
        nominal = _flight(solver, base, 0.0)
        rows = []
        for lean in (0.0, 10.0, 20.0):
            for mode in ("fixed_loft", "shaft_rotation"):
                config = replace(base, shaft_lean_deg=lean, delivery_mode=mode)
                for error in (-2.0, -1.0, 0.0, 1.0, 2.0):
                    loft = base.loft_deg
                    if mode == "shaft_rotation":
                        loft = delivery_from_face_angle(
                            error,
                            base_loft_deg=base.loft_deg,
                            lie_deg=base.lie_deg,
                            shaft_lean_deg=lean,
                        ).dynamic_loft_deg
                    flight = _flight(solver, config, error)
                    rows.append(
                        {
                            "face_error_deg": error,
                            "shaft_lean_deg": lean,
                            "delivery_mode": mode,
                            "nominal_loft_deg": base.loft_deg,
                            "dynamic_loft_deg": loft,
                            "carry_delta_m": flight["carry_m"] - nominal["carry_m"],
                            **flight,
                        }
                    )
        loft_rows = []
        for offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
            flight = _flight(
                solver, replace(base, loft_deg=base.loft_deg + offset), 0.0
            )
            loft_rows.append(
                {
                    "loft_offset_deg": offset,
                    "dynamic_loft_deg": base.loft_deg + offset,
                    "carry_delta_m": flight["carry_m"] - nominal["carry_m"],
                    **flight,
                }
            )
        pitch_rows = []
        for pitch in (0.0, 10.0, 20.0):
            geometry = fixed_club_pitch(
                base_loft_deg=base.loft_deg, lie_deg=base.lie_deg, pitch_deg=pitch
            )
            loft = geometry["dynamic_loft_deg"]
            if 0 < loft < 45:
                flight = _flight(solver, replace(base, loft_deg=loft), 0.0)
                geometry["flight"] = {
                    "carry_delta_m": flight["carry_m"] - nominal["carry_m"],
                    **flight,
                }
            else:
                geometry["flight"] = None
                geometry["flight_unavailable_reason"] = (
                    "pitched loft outside positive-loft shot-wrapper domain; geometry only"
                )
            pitch_rows.append(geometry)
        result["clubs"][name] = {
            "config": asdict(base),
            "nominal": nominal,
            "face_error_axis_grid": rows,
            "loft_only_grid": loft_rows,
            "fixed_club_pitch_grid": pitch_rows,
        }
    return result


def preset_configs() -> dict[str, AnalysisConfig]:
    """Return the exact current illustrative club inputs for deterministic runs."""
    return {
        key: AnalysisConfig(
            club_id=key,
            club_speed_mps=preset.club_speed_mps,
            loft_deg=preset.loft_deg,
            attack_angle_deg=preset.attack_angle_deg,
            lie_deg=preset.lie_deg,
            clubhead_mass_kg=preset.clubhead_mass_kg,
        )
        for key, preset in CLUB_PRESETS.items()
    }


def scientific_source_hashes() -> dict[str, str]:
    """Hash the shared scientific execution sources and native dependency pin."""
    root = Path(__file__).resolve().parents[3]
    paths = [
        Path(__file__).parent / name
        for name in (
            "sensitivity.py",
            "numerics.py",
            "core.py",
            "physics.py",
            "delivery_geometry.py",
            "presets.py",
        )
    ]
    physics = root / "src/shared/python/physics"
    paths += list((physics / "impact_model").glob("*.py"))
    paths += [
        physics / name
        for name in (
            "ball_simulator.py",
            "ball_launch_conditions.py",
            "ball_properties.py",
            "rust_kernel.py",
        )
    ]
    paths += [
        root / "src/shared/python/core/physics_constants.py",
        root / "Cargo.lock",
        root / "Cargo.toml",
    ]
    paths += list((root / "rust_core/upstream-physics/src").glob("*.rs"))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def runtime_provenance(source_hashes: dict[str, str], command: str) -> dict:
    """Record numerical dependencies and the loaded native extension identity."""
    import upstream_physics

    native_dir = Path(upstream_physics.__file__).resolve().parent
    binaries = sorted(native_dir.glob("*.so")) + sorted(native_dir.glob("*.pyd"))
    if not binaries:
        raise RuntimeError("native extension binary not found for provenance")
    return {
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ("numpy", "scipy", "upstream-physics")
        },
        "source_sha256": source_hashes,
        "native_binary_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in binaries
        },
        "reproduction_command": command,
        "native_build_command": "maturin develop --release --manifest-path rust_core/upstream-physics/Cargo.toml",
        "qualification": "software-level deterministic study; no measured club/golfer validation",
    }


def write_sensitivity(output: Path) -> dict:
    """Write the three-club grid and exact source/native-binary provenance."""
    source_hashes = scientific_source_hashes()
    result = build_sensitivity(preset_configs())
    if scientific_source_hashes() != source_hashes:
        raise RuntimeError("scientific source changed during sensitivity run; retry")
    result["provenance"] = runtime_provenance(
        source_hashes,
        "python3 -m src.tools.shot_pattern_analysis.sensitivity --output docs/research/shot_pattern_analysis/corrected_sensitivity.json",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def render_sensitivity(result: dict, output: Path) -> None:
    """Render the recorded lean-zero controls without rerunning any physics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("#f7f9fc")
    labels = {
        "driver": "Driver",
        "seven_iron": "7-Iron",
        "pitching_wedge": "Pitching Wedge",
    }
    directions = []
    for ax, (name, club) in zip(axes, result["clubs"].items(), strict=True):
        for mode, color, label in (
            ("fixed_loft", "#64748b", "Fixed Loft"),
            ("shaft_rotation", "#1764c0", "Shaft Rotation"),
        ):
            rows = [
                row
                for row in club["face_error_axis_grid"]
                if row["shaft_lean_deg"] == 0 and row["delivery_mode"] == mode
            ]
            x = [row["face_error_deg"] for row in rows]
            y = [row["carry_delta_m"] for row in rows]
            ax.plot(
                x, y, marker="o", markersize=8, linewidth=3, color=color, label=label
            )
            if mode == "shaft_rotation":
                for index in (0, 4):
                    row = rows[index]
                    ax.annotate(
                        f"{row['dynamic_loft_deg']:.2f}° Loft",
                        (x[index], y[index]),
                        xytext=(8 if index == 0 else -8, 15 if y[index] >= 0 else -28),
                        textcoords="offset points",
                        ha="left" if index == 0 else "right",
                        color=color,
                        fontsize=13,
                        fontweight="bold",
                    )
        left, right = rows[0]["carry_delta_m"], rows[-1]["carry_delta_m"]
        directions.append(
            f"{labels.get(name, name)}: {'Long' if left > 0 else 'Short'}-Left/{'Long' if right > 0 else 'Short'}-Right"
        )
        ax.axhline(0, color="#cbd5e1", linewidth=1)
        ax.axvline(0, color="#cbd5e1", linewidth=1)
        ax.set_xlim(-2.35, 2.35)
        ax.set_ylim(-8, 4)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-8, -6, -4, -2, 0, 2, 4])
        ax.grid(axis="y", alpha=0.16)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Face Error from Nominal (°)", labelpad=13)
        ax.set_title(
            f"{labels.get(name, name)}\nNominal Carry: {club['nominal']['carry_m']:.1f} m",
            fontsize=21,
            fontweight="bold",
            pad=24,
        )
        config = club["config"]
        ax.text(
            0.5,
            -0.20,
            f"{config['club_speed_mps']:g} m/s • {config['loft_deg']:g}° Nominal Loft\n{config['attack_angle_deg']:g}° Attack • {config['lie_deg']:g}° Shaft Elevation",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
            color="#475569",
        )
    axes[0].set_ylabel("Carry Change from Nominal (m)", labelpad=12)
    fig.suptitle(
        "Face–Loft Coupling Changes Carry Differently by Club",
        x=0.07,
        y=0.95,
        ha="left",
        fontsize=30,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.07,
        0.89,
        "Negative Face Error Closes Left; Positive Error Opens Right • Shaft Lean: 0°",
        fontsize=18,
        color="#475569",
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.14),
        ncol=2,
        frameon=False,
        fontsize=17,
    )
    fig.text(
        0.07,
        0.11,
        " • ".join(directions),
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.07,
        0.035,
        "Illustrative Delivery Presets • Fixed Path, Speed, and Strike • First Ground Contact, No Roll\nCentral Impact + Native RK4 Flight • No Measured Player Validation • Source: corrected_sensitivity.json",
        fontsize=13,
        color="#475569",
        linespacing=1.6,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.76, bottom=0.35, wspace=0.27)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="render existing JSON without rerunning physics",
    )
    args = parser.parse_args()
    result = (
        json.loads(args.output.read_text())
        if args.render_only
        else write_sensitivity(args.output)
    )
    render_sensitivity(result, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
