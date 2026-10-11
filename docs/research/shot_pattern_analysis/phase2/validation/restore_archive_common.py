"""Guarded primitives shared by primary/control saved-data restoration audits."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from scripts.research.shot_pattern_analysis.archive_data import verify_archives


def sha(path: Path) -> str:
    """Hash bounded blocks without holding an archive in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Raise a descriptive failure when a restoration invariant is false."""
    if not condition:
        raise ValueError(message)


def inventory(directory: Path) -> dict[str, dict[str, Any]]:
    """Return a symlink-free relative-path, byte-size, and SHA inventory."""
    directory = Path(directory)
    entries = sorted(directory.rglob("*"))
    require(not any(path.is_symlink() for path in entries), "symlink in inventory")
    return {
        path.relative_to(directory.parent).as_posix(): {
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
        for path in entries
        if path.is_file()
    }


@dataclass(frozen=True)
class VerifiedArchiveBundle:
    """Bundle identity returned only after the archive verifier succeeds."""

    path: Path
    manifest: dict[str, Any]
    file_count: int


def verify_archive_bundle(bundle: Path) -> VerifiedArchiveBundle:
    """Verify archive hashes, members, and payloads before reading the manifest."""
    bundle = Path(bundle)
    count = verify_archives(bundle)
    manifest = json.loads((bundle / "archive_manifest.json").read_text())
    return VerifiedArchiveBundle(bundle, manifest, count)


def restore_archive_and_compare(
    verified: VerifiedArchiveBundle,
    dataset_name: str,
    expected_inventory: Mapping[str, Mapping[str, Any]],
    temporary_prefix: str,
    load_report: Callable[[Path], Any],
    original_report: Any,
    restored_bytes_message: str = "restored file bytes",
) -> None:
    """Extract a verified bundle into a fresh temp root and require exact parity."""
    with tempfile.TemporaryDirectory(prefix=temporary_prefix) as temporary:
        destination = Path(temporary)
        require(not any(destination.iterdir()), "restoration directory not fresh")
        for part in verified.manifest["archives"]:
            with ZipFile(verified.path / part) as archive:
                archive.extractall(destination)
        require(
            {path.name for path in destination.iterdir()} == {dataset_name},
            "extra root",
        )
        restored_root = destination / dataset_name
        require(
            inventory(restored_root) == expected_inventory,
            restored_bytes_message,
        )
        require(
            load_report(restored_root) == original_report,
            "full restored report differs",
        )
    require(not destination.exists(), "temporary restoration not cleaned")


def validate_complete_audit(
    audit: Mapping[str, Any], expected_cells: int, expected_rows: int
) -> int:
    """Require a completed independent audit and return its verified row count."""
    require(audit["pass"] is True and audit["complete"] is True, "audit incomplete")
    require(audit["failures"] == [], "scientific audit failures")
    require(
        audit["audited_cells"] == audit["planned_cells"] == expected_cells,
        "audit count",
    )
    require(audit["data_stage"] == "final", "audit stage")
    rows = sum(cell["rows"] for cell in audit["cells"].values())
    require(rows == expected_rows, "audit rows")
    return rows


@dataclass(frozen=True)
class RestorationSnapshot:
    """Frozen inputs that must remain unchanged through a restoration review."""

    root: Path
    original: Path
    original_inventory: Mapping[str, Mapping[str, Any]]
    bundle: Path
    bundle_inventory: Mapping[str, Mapping[str, Any]]
    archive_count: int
    audit_path: Path
    audit_sha256: str
    sources: tuple[Path, ...]
    source_sha256: Mapping[str, str]


def assert_restoration_unchanged(snapshot: RestorationSnapshot) -> None:
    """Recheck every source, dataset, bundle, and audit file after restoration."""
    require(
        inventory(snapshot.original) == snapshot.original_inventory,
        "original changed during review",
    )
    require(
        verify_archives(snapshot.bundle) == snapshot.archive_count,
        "archive count changed",
    )
    require(
        inventory(snapshot.bundle) == snapshot.bundle_inventory,
        "bundle changed during review",
    )
    require(sha(snapshot.audit_path) == snapshot.audit_sha256, "audit receipt changed")
    current_sources = {
        str(path.relative_to(snapshot.root)): sha(path) for path in snapshot.sources
    }
    require(current_sources == snapshot.source_sha256, "review source changed")


def validate_report_binding(
    report: Mapping[str, Any],
    audit: Mapping[str, Any],
    expected_counts: Mapping[str, int],
    planned_cells: int,
) -> None:
    """Bind restored matrix accounting and identities to the independent audit."""
    require(report["counts"] == expected_counts, "matrix status counts")
    require(report["planned_cells"] == planned_cells, "matrix inventory")
    require(report["stage"] == "final", "matrix stage")
    require(report["plan_sha256"] == audit["plan_sha256"], "audit plan binding")
    require(report["execution"] == audit["execution"], "audit execution binding")
    require(
        report["cell_manifest_sha256"]
        == {name: cell["manifest_sha256"] for name, cell in audit["cells"].items()},
        "audit cell binding",
    )
    require(
        {cell["cell"]: cell["status"] for cell in report["cells"]}
        == {name: cell["status"] for name, cell in audit["cells"].items()},
        "audit status binding",
    )
