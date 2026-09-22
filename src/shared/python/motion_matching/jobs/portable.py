"""Portable save/reopen/export packages for matching run results (#10379)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    JOBS_SCHEMA,
    AcceptanceState,
    CorruptPackageError,
    HashBundle,
    JobStage,
    JobStatus,
    PortablePackageError,
    RunManifest,
)
from .io_atomic import atomic_write_bytes, atomic_write_json, write_run_manifest

__all__ = ["PortablePackage", "export_portable_package", "import_portable_package"]

_FORBIDDEN_SUFFIXES = {".pkl", ".pickle", ".joblib"}
_MANIFEST_NAME = "manifest.json"
_ASSETS_DIR = "assets"


@dataclass(frozen=True, slots=True)
class PortablePackage:
    """Opened portable package rooted at ``root``."""

    root: Path
    run_id: str
    acceptance: AcceptanceState
    status: JobStatus
    stage: JobStage
    hashes: HashBundle
    provenance: str
    assets: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": JOBS_SCHEMA,
            "run_id": self.run_id,
            "acceptance": self.acceptance.value,
            "status": self.status.value,
            "stage": self.stage.value,
            "hashes": self.hashes.to_dict(),
            "provenance": self.provenance,
            "assets": {
                k: str(v.relative_to(self.root)) for k, v in self.assets.items()
            },
        }


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def _assert_not_pickle(path: Path) -> None:
    if path.suffix.lower() in _FORBIDDEN_SUFFIXES:
        raise PortablePackageError(f"refusing pickle executable payload: {path.name}")


def _constrain_under_root(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    if not candidate.is_relative_to(root_resolved):
        raise PortablePackageError(f"asset path escapes package root: {relative!r}")
    return candidate


def export_portable_package(
    run_root: Path,
    package_dir: Path,
    *,
    asset_paths: Mapping[str, Path],
    input_capture_paths: tuple[Path, ...] = (),
) -> Path:
    """Export a relocated-safe package with relative refs and checksums."""
    run_root = Path(run_root)
    package_dir = Path(package_dir)
    input_set = {p.resolve() for p in input_capture_paths}

    if package_dir.exists():
        existing_manifest = package_dir / _MANIFEST_NAME
        if existing_manifest.exists():
            for capture in input_set:
                # Never overwrite an already-packaged input capture.
                packaged = package_dir / _ASSETS_DIR / capture.name
                if packaged.exists():
                    raise PortablePackageError(
                        "refusing to overwrite input capture in portable package"
                    )

    package_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = package_dir / _ASSETS_DIR
    assets_dir.mkdir(exist_ok=True)

    manifest_src = run_root / "run_manifest.json"
    if not manifest_src.exists():
        raise PortablePackageError("run_manifest.json missing from run root")
    run_manifest = RunManifest.from_dict(
        json.loads(manifest_src.read_text(encoding="utf-8"))
    )

    relative_assets: dict[str, str] = {}
    checksums: dict[str, str] = {}
    for name, src in asset_paths.items():
        src_path = Path(src)
        _assert_not_pickle(src_path)
        if not src_path.exists():
            raise PortablePackageError(f"missing asset {name}: {src_path}")
        dest_rel = f"{_ASSETS_DIR}/{src_path.name}"
        dest = _constrain_under_root(package_dir, dest_rel)
        if src_path.resolve() in input_set and dest.exists():
            raise PortablePackageError(
                "refusing to overwrite input capture in portable package"
            )
        atomic_write_bytes(dest, src_path.read_bytes())
        relative_assets[name] = dest_rel.replace("\\", "/")
        checksums[name] = _sha256_file(dest)

    payload = {
        "schema_version": JOBS_SCHEMA,
        "run_id": run_manifest.run_id,
        "status": run_manifest.status.value,
        "stage": run_manifest.stage.value,
        "acceptance": run_manifest.acceptance.value,
        "hashes": run_manifest.hashes.to_dict(),
        "provenance": run_manifest.provenance,
        "assets": relative_assets,
        "checksums": checksums,
        "blockers": list(run_manifest.blockers),
    }
    atomic_write_json(package_dir / _MANIFEST_NAME, payload)
    # Keep a copy of the run manifest for identity continuity.
    write_run_manifest(package_dir, run_manifest)
    return package_dir


def import_portable_package(package_dir: Path) -> PortablePackage:
    """Open a package, verifying schema, path constraints, and checksums."""
    package_dir = Path(package_dir)
    manifest_path = package_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        raise CorruptPackageError(f"missing {_MANIFEST_NAME}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorruptPackageError("corrupt package manifest") from exc
    schema = raw.get("schema_version")
    if schema != JOBS_SCHEMA:
        raise CorruptPackageError(
            f"package schema_version must be {JOBS_SCHEMA!r}; got {schema!r}"
        )
    assets_raw = dict(raw.get("assets", {}))
    checksums = dict(raw.get("checksums", {}))
    assets: dict[str, Path] = {}
    for name, rel in assets_raw.items():
        if not isinstance(rel, str):
            raise PortablePackageError(f"asset {name} path must be a string")
        path = _constrain_under_root(package_dir, rel)
        if not path.exists():
            raise CorruptPackageError(f"missing asset file for {name}")
        expected = checksums.get(name)
        if expected is not None and _sha256_file(path) != expected:
            raise CorruptPackageError(f"checksum mismatch for asset {name}")
        assets[name] = path
    return PortablePackage(
        root=package_dir.resolve(),
        run_id=str(raw["run_id"]),
        acceptance=AcceptanceState(str(raw["acceptance"])),
        status=JobStatus(str(raw["status"])),
        stage=JobStage(str(raw["stage"])),
        hashes=HashBundle.from_dict(raw["hashes"]),
        provenance=str(raw.get("provenance", "fresh")),
        assets=assets,
    )
