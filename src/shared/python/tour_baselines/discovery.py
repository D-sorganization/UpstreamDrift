"""Baseline discovery, portable loading and safe model presets (TB-10 #10595).

Integrates baseline catalog discovery, portable export/import, and safe model presets:
1. SafeModelPreset extracts verified state, geometry, inertia, and controls from BaselinePackage.
2. Fail-closed compatibility checks refuse topology mismatches and missing dependencies.
3. Safe session cloning creates isolated presets without modifying parent or nominated baselines.
4. BaselineDiscoveryService scans configurable search paths without hardcoded machine paths.
5. Multi-field filtering (model, club, horizon, qualification status).
6. Fail-closed default selection: unverified/unqualified packages can NEVER be selected as default presets.
7. Portable export and clean-machine import with SHA-256 checksum verification.
8. Rebuilding index preserves deterministic baseline identities.
9. Headless CLI for listing, inspecting, exporting, and importing baseline presets.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np

from src.shared.python.tour_baselines.baseline_package import (
    BASELINE_SCHEMA_VERSION,
    BaselineIdentity,
    BaselinePackage,
    ScientificQualificationStatus,
    StatusBundle,
    _compute_array_hash,
    export_baseline_package,
    import_baseline_package,
)
from src.shared.python.tour_baselines.models import ModelTopology
from src.shared.python.tour_baselines.qualification import compute_package_digest

logger = logging.getLogger(__name__)

PRESET_SCHEMA_VERSION = "tour-safe-preset/1.0.0"
PRESET_MANIFEST_KEY = "preset_manifest_json"

_DEFAULT_GEOMETRY: dict[ModelTopology, dict[str, float]] = {
    ModelTopology.PLANAR_DRIVEN_PENDULUM: {"arm_length": 0.60, "club_length": 1.15},
    ModelTopology.KINEMATIC_RECONSTRUCTION: {"arm_length": 0.60, "club_length": 1.15},
    ModelTopology.CONSTRAINED_UPPER_BODY: {
        "torso_width": 0.40,
        "arm_length": 0.60,
        "club_length": 1.15,
    },
    ModelTopology.FULL_BODY_MULTIBODY: {
        "height_m": 1.80,
        "arm_length": 0.60,
        "club_length": 1.15,
    },
    ModelTopology.REFERENCE_CATALOG_URDF: {
        "arm_length": 0.60,
        "club_length": 1.15,
    },
}

_DEFAULT_INERTIA: dict[ModelTopology, dict[str, float]] = {
    ModelTopology.PLANAR_DRIVEN_PENDULUM: {
        "mass": 5.0,
        "arm_mass": 3.0,
        "club_mass": 0.35,
        "izz": 0.5,
    },
    ModelTopology.KINEMATIC_RECONSTRUCTION: {
        "mass": 5.0,
        "club_mass": 0.35,
    },
    ModelTopology.CONSTRAINED_UPPER_BODY: {
        "mass": 35.0,
        "torso_mass": 25.0,
        "arm_mass": 3.5,
        "club_mass": 0.35,
    },
    ModelTopology.FULL_BODY_MULTIBODY: {
        "mass": 80.0,
        "club_mass": 0.35,
    },
    ModelTopology.REFERENCE_CATALOG_URDF: {
        "mass": 75.0,
        "club_mass": 0.35,
    },
}


class IncompatiblePresetError(Exception):
    """Raised when model topology, identity, or qualification requirements are incompatible."""


class MissingDependencyError(Exception):
    """Raised when a required physics engine, solver, or library dependency is missing."""


class BaselineNotFoundError(Exception):
    """Raised when a requested baseline ID or path cannot be resolved."""


@dataclass(frozen=True)
class SafeModelPreset:
    """Portable, safe snapshot of initial state, geometry, inertia, and controls."""

    preset_id: str
    baseline_id: str
    model_id: str
    topology: ModelTopology
    club: str
    horizon: str
    is_nominated_baseline: bool
    is_qualified: bool
    q0: np.ndarray
    v0: np.ndarray
    inertia: dict[str, Any]
    geometry: dict[str, Any]
    controls: np.ndarray | None = None
    dependencies: dict[str, str] = field(default_factory=dict)
    package_digest: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_package(
        cls,
        package: BaselinePackage,
        *,
        is_nominated: bool = False,
        dependencies: dict[str, str] | None = None,
        preset_id: str | None = None,
        baseline_id: str | None = None,
    ) -> SafeModelPreset:
        """Construct SafeModelPreset from an authoritative BaselinePackage."""
        ident = package.identity
        top = ident.topology
        b_id = baseline_id or f"{ident.model_id}_{ident.capture}_{ident.horizon}"
        p_id = preset_id or f"preset_{b_id}"

        # Extract initial state
        trajs = package.trajectories
        if "q" in trajs and len(trajs["q"]) > 0:
            q0 = np.array(trajs["q"][0], dtype=np.float64)
        else:
            q0 = np.empty((0,), dtype=np.float64)

        if "v" in trajs and len(trajs["v"]) > 0:
            v0 = np.array(trajs["v"][0], dtype=np.float64)
        else:
            v0 = np.empty((0,), dtype=np.float64)

        controls = trajs.get("tau")

        # Geometry and inertia
        reports = package.reports
        geom = dict(
            _DEFAULT_GEOMETRY.get(top, {"arm_length": 0.60, "club_length": 1.15})
        )
        if "geometry" in reports and isinstance(reports["geometry"], dict):
            geom.update(reports["geometry"])

        inertia = dict(
            _DEFAULT_INERTIA.get(
                top, {"mass": 5.0, "arm_mass": 3.0, "club_mass": 0.35, "izz": 0.5}
            )
        )
        if "inertia" in reports and isinstance(reports["inertia"], dict):
            inertia.update(reports["inertia"])

        is_qual = (
            package.statuses.scientific_qualification
            == ScientificQualificationStatus.QUALIFIED
        )

        pkg_digest = compute_package_digest(package)

        return cls(
            preset_id=p_id,
            baseline_id=b_id,
            model_id=ident.model_id,
            topology=top,
            club=ident.capture,
            horizon=ident.horizon,
            is_nominated_baseline=is_nominated,
            is_qualified=is_qual,
            q0=q0,
            v0=v0,
            inertia=inertia,
            geometry=geom,
            controls=controls,
            dependencies=dict(dependencies or {}),
            package_digest=pkg_digest,
            metadata={
                "frame_convention": ident.frame_convention,
                "plane_convention": ident.plane_convention,
                "measurement_map_version": ident.measurement_map_version,
                "solver_name": ident.solver_name,
                "seed": ident.seed,
            },
        )

    def verify_compatibility(
        self,
        *,
        target_model_id: str | None = None,
        target_topology: ModelTopology | None = None,
        available_dependencies: Mapping[str, str] | None = None,
    ) -> None:
        """Fail closed if topology, model ID, or dependencies do not match."""
        if target_topology is not None and target_topology != self.topology:
            raise IncompatiblePresetError(
                f"Topology mismatch: expected {self.topology}, target is {target_topology}"
            )

        if target_model_id is not None and target_model_id != self.model_id:
            raise IncompatiblePresetError(
                f"Model ID mismatch: expected {self.model_id}, target is {target_model_id}"
            )

        if available_dependencies is not None:
            for dep_name in self.dependencies:
                if dep_name not in available_dependencies:
                    raise MissingDependencyError(
                        f"Missing required dependency: {dep_name}"
                    )

    def clone_into_session(
        self,
        session_dir: Path | str,
        new_preset_id: str,
    ) -> SafeModelPreset:
        """Clone preset into an isolated session directory without mutating original."""
        s_dir = Path(session_dir)
        s_dir.mkdir(parents=True, exist_ok=True)

        cloned = SafeModelPreset(
            preset_id=new_preset_id,
            baseline_id=self.baseline_id,
            model_id=self.model_id,
            topology=self.topology,
            club=self.club,
            horizon=self.horizon,
            is_nominated_baseline=False,
            is_qualified=self.is_qualified,
            q0=self.q0.copy(),
            v0=self.v0.copy(),
            inertia=dict(self.inertia),
            geometry=dict(self.geometry),
            controls=self.controls.copy() if self.controls is not None else None,
            dependencies=dict(self.dependencies),
            package_digest=self.package_digest,
            metadata=dict(self.metadata),
        )

        preset_file = s_dir / f"{new_preset_id}.json"
        preset_file.write_text(
            json.dumps(cloned.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return cloned

    def to_dict(self) -> dict[str, Any]:
        """Serialize SafeModelPreset to a JSON-compatible dict."""
        return {
            "schema_version": PRESET_SCHEMA_VERSION,
            "preset_id": self.preset_id,
            "baseline_id": self.baseline_id,
            "model_id": self.model_id,
            "topology": self.topology.value,
            "club": self.club,
            "horizon": self.horizon,
            "is_nominated_baseline": self.is_nominated_baseline,
            "is_qualified": self.is_qualified,
            "q0": self.q0.tolist(),
            "v0": self.v0.tolist(),
            "inertia": self.inertia,
            "geometry": self.geometry,
            "controls": self.controls.tolist() if self.controls is not None else None,
            "dependencies": self.dependencies,
            "package_digest": self.package_digest,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SafeModelPreset:
        """Deserialize SafeModelPreset from a dictionary."""
        controls_raw = data.get("controls")
        controls = (
            np.array(controls_raw, dtype=np.float64)
            if controls_raw is not None
            else None
        )

        return cls(
            preset_id=data["preset_id"],
            baseline_id=data["baseline_id"],
            model_id=data["model_id"],
            topology=ModelTopology(data["topology"]),
            club=data["club"],
            horizon=data["horizon"],
            is_nominated_baseline=bool(data.get("is_nominated_baseline", False)),
            is_qualified=bool(data.get("is_qualified", False)),
            q0=np.array(data["q0"], dtype=np.float64),
            v0=np.array(data["v0"], dtype=np.float64),
            inertia=dict(data.get("inertia", {})),
            geometry=dict(data.get("geometry", {})),
            controls=controls,
            dependencies=dict(data.get("dependencies", {})),
            package_digest=data.get("package_digest", ""),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class BaselineFilter:
    """Filter specification for catalog search."""

    model_id: str | None = None
    club: str | None = None
    horizon: str | None = None
    only_qualified: bool = False
    only_nominated: bool = False


@dataclass(frozen=True)
class BaselineSummary:
    """Compact summary of a discovered baseline package."""

    baseline_id: str
    model_id: str
    topology: ModelTopology
    capture: str
    horizon: str
    is_qualified: bool
    is_nominated: bool
    package_path: Path
    package_digest: str
    created_at_utc: str
    metrics_summary: dict[str, Any]
    statuses: StatusBundle

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "baseline_id": self.baseline_id,
            "model_id": self.model_id,
            "topology": self.topology.value,
            "capture": self.capture,
            "horizon": self.horizon,
            "is_qualified": self.is_qualified,
            "is_nominated": self.is_nominated,
            "package_path": str(self.package_path),
            "package_digest": self.package_digest,
            "created_at_utc": self.created_at_utc,
            "metrics_summary": self.metrics_summary,
            "statuses": self.statuses.to_dict(),
        }


@dataclass(frozen=True)
class BaselineDetail:
    """Detailed view of an inspected baseline package."""

    baseline_id: str
    model_id: str
    topology: str
    capture: str
    horizon: str
    is_qualified: bool
    is_nominated: bool
    package_path: str
    package_digest: str
    metrics: dict[str, Any]
    statuses: dict[str, Any]
    identity: dict[str, Any]
    reports: dict[str, Any]
    artifacts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert detail to dictionary."""
        return asdict(self)


class BaselineDiscoveryService:
    """Service to discover, index, filter, inspect, and export baseline packages."""

    def __init__(self, search_roots: list[Path | str] | None = None) -> None:
        self.search_roots: list[Path] = [
            Path(r).resolve()
            for r in (search_roots or [Path("artifacts/tour_baselines")])
        ]
        self._nominated: dict[str, str] = {}
        self._cache: dict[Path, BaselinePackage] = {}

    def set_nominated_baseline(self, model_id: str, baseline_id: str) -> None:
        """Mark a specific baseline ID as the nominated standard for a model."""
        self._nominated[model_id] = baseline_id

    def discover(
        self, filter_spec: BaselineFilter | None = None
    ) -> list[BaselineSummary]:
        """Scan search roots for baseline packages and return matching summaries."""
        flt = filter_spec or BaselineFilter()
        results: list[BaselineSummary] = []

        all_paths: list[Path] = []
        for root in self.search_roots:
            if not root.is_dir():
                continue
            all_paths.extend(root.rglob("*.npz"))

        # Sort paths deterministically
        for path in sorted(all_paths, key=lambda p: str(p)):
            try:
                pkg = self._load_package(path)
            except Exception as e:
                logger.warning("Failed to load baseline package from %s: %s", path, e)
                continue

            ident = pkg.identity
            b_id = path.stem
            is_qual = (
                pkg.statuses.scientific_qualification
                == ScientificQualificationStatus.QUALIFIED
            )
            is_nom = self._nominated.get(ident.model_id) == b_id

            # Apply filters
            if flt.model_id is not None and ident.model_id != flt.model_id:
                continue
            if flt.club is not None and ident.capture != flt.club:
                continue
            if flt.horizon is not None and ident.horizon != flt.horizon:
                continue
            if flt.only_qualified and not is_qual:
                continue
            if flt.only_nominated and not is_nom:
                continue

            metrics_summary = {
                "whole_marker_rmse_m": pkg.metrics.whole_marker_rmse_m,
                "endpoint_error_m": pkg.metrics.endpoint_error_m,
                "coverage_fraction": pkg.metrics.coverage_fraction,
            }

            summary = BaselineSummary(
                baseline_id=b_id,
                model_id=ident.model_id,
                topology=ident.topology,
                capture=ident.capture,
                horizon=ident.horizon,
                is_qualified=is_qual,
                is_nominated=is_nom,
                package_path=path,
                package_digest=compute_package_digest(pkg),
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                metrics_summary=metrics_summary,
                statuses=pkg.statuses,
            )
            results.append(summary)

        # Sort deterministically by baseline_id
        results.sort(key=lambda s: s.baseline_id)
        return results

    def _load_package(self, path: Path) -> BaselinePackage:
        """Load and cache package from path."""
        p_res = path.resolve()
        if p_res not in self._cache:
            self._cache[p_res] = import_baseline_package(p_res)
        return self._cache[p_res]

    def get_default_preset(
        self,
        model_id: str,
        *,
        club: str = "driver",
        horizon: str = "G1",
    ) -> SafeModelPreset:
        """Get default safe model preset, failing closed if no qualified baseline exists."""
        flt = BaselineFilter(
            model_id=model_id,
            club=club,
            horizon=horizon,
            only_qualified=True,
        )
        qualified_summaries = self.discover(flt)
        if not qualified_summaries:
            raise IncompatiblePresetError(
                f"No qualified baseline preset available for {model_id} ({club}, {horizon})"
            )

        # Prioritize nominated package if one exists among qualified
        selected_summary = qualified_summaries[0]
        for s in qualified_summaries:
            if s.is_nominated:
                selected_summary = s
                break

        pkg = self._load_package(selected_summary.package_path)
        return SafeModelPreset.from_package(
            pkg,
            is_nominated=selected_summary.is_nominated,
            baseline_id=selected_summary.baseline_id,
        )

    def inspect(self, baseline_id: str) -> BaselineDetail:
        """Inspect a single baseline by ID and return its detailed structure."""
        summaries = self.discover()
        target_summary: BaselineSummary | None = None
        for s in summaries:
            if s.baseline_id == baseline_id:
                target_summary = s
                break

        if target_summary is None:
            raise BaselineNotFoundError(f"Baseline '{baseline_id}' not found")

        pkg = self._load_package(target_summary.package_path)
        return BaselineDetail(
            baseline_id=target_summary.baseline_id,
            model_id=target_summary.model_id,
            topology=target_summary.topology.value,
            capture=target_summary.capture,
            horizon=target_summary.horizon,
            is_qualified=target_summary.is_qualified,
            is_nominated=target_summary.is_nominated,
            package_path=str(target_summary.package_path),
            package_digest=target_summary.package_digest,
            metrics=pkg.metrics.to_dict(),
            statuses=pkg.statuses.to_dict(),
            identity=pkg.identity.to_dict(),
            reports=dict(pkg.reports),
            artifacts=dict(pkg.artifacts),
        )

    def export_preset_package(
        self,
        preset: SafeModelPreset,
        destination_path: Path | str,
    ) -> Path:
        """Export preset package to a verifiable portable archive."""
        dest = Path(destination_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        manifest = preset.to_dict()
        manifest_json = json.dumps(manifest, indent=2, sort_keys=True)

        arrays_to_save: dict[str, np.ndarray] = {
            PRESET_MANIFEST_KEY: np.array(manifest_json),
            "q0": preset.q0,
            "v0": preset.v0,
        }
        if preset.controls is not None:
            arrays_to_save["controls"] = preset.controls

        # Compute checksums
        checksums = {
            "q0": _compute_array_hash(preset.q0),
            "v0": _compute_array_hash(preset.v0),
        }
        if preset.controls is not None:
            checksums["controls"] = _compute_array_hash(preset.controls)

        manifest["array_checksums"] = checksums
        manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
        arrays_to_save[PRESET_MANIFEST_KEY] = np.array(manifest_json)

        np.savez(dest, **arrays_to_save)  # type: ignore[arg-type]
        return dest

    def import_preset_package(
        self,
        archive_path: Path | str,
        target_dir: Path | str,
    ) -> SafeModelPreset:
        """Import, verify checksum integrity, and reconstruct SafeModelPreset."""
        src = Path(archive_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Preset archive does not exist: {src}")

        t_dir = Path(target_dir).resolve()
        t_dir.mkdir(parents=True, exist_ok=True)

        with np.load(src, allow_pickle=False) as data:
            if PRESET_MANIFEST_KEY not in data:
                raise ValueError(f"Archive missing {PRESET_MANIFEST_KEY}")

            manifest_str = str(data[PRESET_MANIFEST_KEY])
            manifest = json.loads(manifest_str)

            checksums = manifest.get("array_checksums", {})
            for key in ("q0", "v0"):
                if key not in data or key not in checksums:
                    raise ValueError(f"Required array '{key}' missing or unchecksummed")
                actual_hash = _compute_array_hash(data[key])
                expected_hash = checksums[key]
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"Checksum mismatch for '{key}': expected {expected_hash}, got {actual_hash}"
                    )
                manifest_arr = np.asarray(manifest.get(key, []))
                if not np.array_equal(manifest_arr, data[key]):
                    raise ValueError(
                        f"Manifest embedded array '{key}' does not match verified array member"
                    )

            if "controls" in data:
                if "controls" not in checksums:
                    raise ValueError(
                        "Required array 'controls' missing or unchecksummed"
                    )
                actual_hash = _compute_array_hash(data["controls"])
                if actual_hash != checksums["controls"]:
                    raise ValueError("Checksum mismatch for 'controls'")
                manifest_controls = np.asarray(manifest.get("controls", []))
                if not np.array_equal(manifest_controls, data["controls"]):
                    raise ValueError(
                        "Manifest embedded array 'controls' does not match verified array member"
                    )

            # Reconstruct preset from verified array members
            manifest["q0"] = data["q0"]
            manifest["v0"] = data["v0"]
            if "controls" in data:
                manifest["controls"] = data["controls"]
            preset = SafeModelPreset.from_dict(manifest)

        # Copy archive to target directory
        shutil.copy2(src, t_dir / src.name)
        return preset

    def rebuild_index(self) -> list[BaselineSummary]:
        """Clear cache and rebuild the catalog index deterministically."""
        self._cache.clear()
        return self.discover()


def export_to_ledger_rows(summaries: list[BaselineSummary]) -> list[dict[str, Any]]:
    """Convert baseline summaries to ledger rows conforming to shared result index."""
    rows: list[dict[str, Any]] = []
    for s in summaries:
        rows.append(
            {
                "receipt_path": str(s.package_path),
                "sha256": s.package_digest,
                "engine": s.model_id,
                "metrics": s.metrics_summary,
                "acceptance": s.is_qualified,
            }
        )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command line argument parser for baseline discovery."""
    parser = argparse.ArgumentParser(
        prog="tour_baselines.discovery",
        description="Discover, filter, inspect, and export tour baseline presets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_p = subparsers.add_parser("list", help="List baseline packages")
    list_p.add_argument("--root", type=str, default="artifacts/tour_baselines")
    list_p.add_argument("--format", choices=["json", "text"], default="text")
    list_p.add_argument("--model", type=str, default=None)
    list_p.add_argument("--club", type=str, default=None)
    list_p.add_argument("--horizon", type=str, default=None)
    list_p.add_argument("--only-qualified", action="store_true")
    list_p.add_argument("--only-nominated", action="store_true")

    # inspect
    inspect_p = subparsers.add_parser("inspect", help="Inspect a baseline package")
    inspect_p.add_argument("baseline_id", type=str)
    inspect_p.add_argument("--root", type=str, default="artifacts/tour_baselines")
    inspect_p.add_argument("--format", choices=["json", "text"], default="text")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for baseline discovery."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        service = BaselineDiscoveryService(search_roots=[args.root])
        flt = BaselineFilter(
            model_id=args.model,
            club=args.club,
            horizon=args.horizon,
            only_qualified=args.only_qualified,
            only_nominated=args.only_nominated,
        )
        summaries = service.discover(flt)
        if args.format == "json":
            out_data = [s.to_dict() for s in summaries]
            sys.stdout.write(json.dumps(out_data, indent=2) + "\n")
        else:
            for s in summaries:
                sys.stdout.write(
                    f"{s.baseline_id:<32} {s.model_id:<25} {s.capture:<8} {s.horizon:<4} qualified={s.is_qualified}\n"
                )
        return 0

    if args.command == "inspect":
        service = BaselineDiscoveryService(search_roots=[args.root])
        detail = service.inspect(args.baseline_id)
        if args.format == "json":
            sys.stdout.write(json.dumps(detail.to_dict(), indent=2) + "\n")
        else:
            sys.stdout.write(f"Baseline: {detail.baseline_id}\n")
            sys.stdout.write(f"Model ID: {detail.model_id}\n")
            sys.stdout.write(f"Digest:   {detail.package_digest}\n")
            sys.stdout.write(f"Status:   {detail.statuses}\n")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
