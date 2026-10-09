"""Frozen, reproducible corrected-impact club comparison matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
from dataclasses import asdict
from pathlib import Path

from .core import AnalysisConfig, run_analysis
from .presets import CLUB_PRESETS
from .reporting import export_analysis
from .scoring import score_saved_bundle

LOGGER = logging.getLogger(__name__)


def source_snapshot() -> dict:
    """Hash the numerical/scoring sources and loaded native binary before a cell."""
    root = Path(__file__).resolve().parents[3]
    package = Path(__file__).parent
    files = [
        package / name
        for name in (
            "core.py", "physics.py", "delivery_geometry.py", "presets.py",
            "reporting.py", "scoring.py", "scoring_cache.py",
            "scenario_scoring.py", "dispersion_stats.py", "range_control.py",
            "uncertainty.py", "matrix.py",
        )
    ]
    files += [
        root / name
        for name in (
            "src/shared/python/physics/impact_model/models.py",
            "src/shared/python/physics/impact_model/solver.py",
            "src/shared/python/physics/impact_model/types.py",
            "src/shared/python/physics/ball_simulator.py",
            "src/shared/python/physics/ball_launch_conditions.py",
            "src/shared/python/physics/ball_properties.py",
            "src/shared/python/core/physics_constants.py",
            "rust_core/upstream-physics/Cargo.toml", "Cargo.lock",
            "vendor/ud-tools/src/shared/python/launch_monitor/strokes_gained.py",
        )
    ]
    files.extend(sorted((root / "rust_core/upstream-physics/src").glob("*.rs")))
    rust_spec = importlib.util.find_spec("upstream_physics")
    if rust_spec is None or rust_spec.origin is None:
        raise RuntimeError("native upstream_physics must be installed")
    rust_package = Path(rust_spec.origin)
    binaries = (
        [rust_package] if rust_package.suffix == ".so"
        else sorted(rust_package.parent.glob("*.so"))
    )
    if len(binaries) != 1:
        raise RuntimeError("expected exactly one native upstream_physics binary")
    return {
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "native_binary_path": str(binaries[0]),
        "native_binary_sha256": hashlib.sha256(binaries[0].read_bytes()).hexdigest(),
    }


def corrected_matrix() -> tuple[tuple[str, AnalysisConfig], ...]:
    """Three clubs × two delivery modes × two face SDs × two curve sizes."""
    scenarios = []
    for curve_scale in (1.0, 2.0):
        for face_sd_deg in (1.0, 2.0):
            for delivery_mode in ("fixed_loft", "shaft_rotation"):
                for preset in CLUB_PRESETS.values():
                    config = AnalysisConfig(
                        club_id=preset.id,
                        n_shots=10_000,
                        face_sd_deg=face_sd_deg,
                        curve_scale=curve_scale,
                        seed=20_261_008,
                        club_speed_mps=preset.club_speed_mps,
                        loft_deg=preset.loft_deg,
                        attack_angle_deg=preset.attack_angle_deg,
                        clubhead_mass_kg=preset.clubhead_mass_kg,
                        lie_deg=preset.lie_deg,
                        shaft_lean_deg=preset.shaft_lean_deg,
                        delivery_mode=delivery_mode,
                    )
                    name = (
                        f"{preset.id}_{delivery_mode}_sd{face_sd_deg:g}"
                        f"_scale{curve_scale:g}"
                    )
                    scenarios.append((name, config))
    if len(scenarios) != 24 or len({name for name, _ in scenarios}) != 24:
        raise RuntimeError("corrected scenario matrix must have 24 unique cells")
    return tuple(scenarios)


def matrix_shard(
    cells: tuple[tuple[str, AnalysisConfig], ...], *, index: int, count: int
) -> tuple[tuple[str, AnalysisConfig], ...]:
    """Partition immutable cells deterministically across independent workers."""
    if count < 1 or not 0 <= index < count:
        raise ValueError("shard index must be within positive shard count")
    return tuple(cell for position, cell in enumerate(cells) if position % count == index)


def run_corrected_matrix(
    output_root: Path,
    *,
    skip_complete: bool = True,
    shard_index: int = 0,
    shard_count: int = 1,
) -> None:
    """Run every full Monte Carlo cell, preserving each cell's own receipt."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    chosen = {name for name, _ in matrix_shard(corrected_matrix(), index=shard_index, count=shard_count)}
    for number, (name, config) in enumerate(corrected_matrix(), start=1):
        if name not in chosen:
            continue
        destination = output_root / name
        finished = destination / "manifest.json"
        if skip_complete and finished.exists():
            LOGGER.info("[%s/24] Skipped complete %s", number, name)
            continue
        LOGGER.info("[%s/24] Running %s", number, name)
        destination.mkdir(parents=True, exist_ok=True)
        execution = source_snapshot()
        (destination / "run_start.json").write_text(
            json.dumps({"config": asdict(config), **execution}, indent=2) + "\n"
        )

        def on_progress(
            done: int, total: int, cell_number: int = number, cell_name: str = name
        ) -> None:
            if done % 3000 == 0:
                LOGGER.info("[%s/24] %s: %s/%s", cell_number, cell_name, done, total)

        result = run_analysis(
            config,
            progress_callback=on_progress,
        )
        paths = export_analysis(result, destination)
        scoring_path = score_saved_bundle(destination)
        if source_snapshot() != execution:
            raise RuntimeError(f"scientific source changed during {name}; discard and rerun")
        receipt_path = destination / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["execution_source_sha256"] = execution["source_sha256"]
        receipt["execution_native_binary_sha256"] = execution["native_binary_sha256"]
        receipt["source_unchanged_during_run"] = True
        receipt["source_sha256_timing"] = "at_export; verified identical to run_start.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        manifest = {
            "scenario": name,
            "config": asdict(config),
            "files_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (*paths.values(), scoring_path, destination / "strokes_gained_baseline.json", destination / "run_start.json")
            },
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        LOGGER.info("[%s/24] Finished %s", number, name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Run 24 corrected shot-pattern scenarios")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    run_corrected_matrix(
        args.output_root, shard_index=args.shard_index, shard_count=args.shard_count
    )


if __name__ == "__main__":
    main()
