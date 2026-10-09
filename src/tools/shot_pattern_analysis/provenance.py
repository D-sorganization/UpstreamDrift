"""One source snapshot and integrity contract for matrix, CLI, and GUI runs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


def source_snapshot() -> dict:
    """Hash the numerical/scoring sources and loaded native binary before a cell."""
    root = Path(__file__).resolve().parents[3]
    package = Path(__file__).parent
    files = [
        package / name
        for name in (
            "core.py",
            "physics.py",
            "delivery_geometry.py",
            "presets.py",
            "reporting.py",
            "scoring.py",
            "scoring_cache.py",
            "scenario_scoring.py",
            "dispersion_stats.py",
            "range_control.py",
            "uncertainty.py",
            "matrix.py",
            "provenance.py",
            "__main__.py",
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
            "rust_core/upstream-physics/Cargo.toml",
            "Cargo.lock",
            "vendor/ud-tools/src/shared/python/launch_monitor/strokes_gained.py",
        )
    ]
    files.extend(sorted((root / "rust_core/upstream-physics/src").glob("*.rs")))
    rust_spec = importlib.util.find_spec("upstream_physics")
    if rust_spec is None or rust_spec.origin is None:
        raise RuntimeError("native upstream_physics must be installed")
    rust_package = Path(rust_spec.origin)
    binaries = (
        [rust_package]
        if rust_package.suffix == ".so"
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


def assert_source_unchanged(expected: dict, actual: dict) -> None:
    """Reject post-run receipts if scientific source or native binary drifted."""
    if actual != expected:
        raise RuntimeError(
            "scientific source changed during analysis; discard and rerun"
        )


def write_run_start(output_dir: Path, config: dict, execution: dict) -> Path:
    """Persist input and source state before the first simulation shot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_start.json"
    path.write_text(json.dumps({"config": config, **execution}, indent=2) + "\n")
    return path


def stamp_execution_receipt(output_dir: Path, execution: dict) -> Path:
    """Tie an export receipt to its pre-run source snapshot."""
    receipt_path = Path(output_dir) / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["execution_source_sha256"] = execution["source_sha256"]
    receipt["execution_native_binary_sha256"] = execution["native_binary_sha256"]
    receipt["source_unchanged_during_run"] = True
    receipt["source_sha256_timing"] = "at_export; verified identical to run_start.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt_path


def manifest_files_match(manifest_path: Path) -> bool:
    """Return true only when every listed completed-cell file still matches."""
    try:
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        hashes = manifest["files_sha256"]
        if not isinstance(hashes, dict) or not hashes:
            return False
        return all(
            isinstance(name, str)
            and isinstance(expected, str)
            and hashlib.sha256((manifest_path.parent / name).read_bytes()).hexdigest()
            == expected
            for name, expected in hashes.items()
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False
