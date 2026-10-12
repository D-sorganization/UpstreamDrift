"""Package exporter for the swing report contract (#12327, epic #11419).

``export_package`` writes a self-describing, verifiable directory::

    report.json          the validated contract document
    series/*.csv         one file per advertised series (empty cell = missing)
    plots/*.png          render=values plots only
    summary.md           Title Case markdown summary
    MANIFEST.json        sha256 of every other file

The exporter never adds data: it serialises an already validated
``SwingReport`` (fail-closed policy of #10382), so a withheld quantity cannot
reach any file. ``load_package`` is the inverse and enforces I13: the package
re-reads, validates and equals its source.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .contract import Quantity, SwingReport
from .plot_render import plot_path, render_plot_png, values_plots
from .series_csv import (
    SERIES_DIR,
    SeriesCsvError,
    check_series_csv,
    plan_series_files,
    series_csv_text,
)
from .summary_md import render_summary
from .validator import validate_swing_report

logger = logging.getLogger(__name__)

REPORT_NAME = "report.json"
SUMMARY_NAME = "summary.md"
MANIFEST_NAME = "MANIFEST.json"
MANIFEST_VERSION = "1"


class PackageError(ValueError):
    """Raised when a package is malformed, incomplete or tampered with."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(doc: Any) -> bytes:
    text = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _quantity_for(report: SwingReport, quantity_id: str) -> Quantity:
    return next(q for q in report.quantities if q.quantity_id == quantity_id)


def build_package_files(report: SwingReport) -> dict[str, bytes]:
    """Return every package file (except the manifest) as ``path -> bytes``.

    Postcondition: iteration order is deterministic and the mapping is a pure
    function of ``report``.
    """
    files: dict[str, bytes] = {REPORT_NAME: _json_bytes(report.to_dict())}
    for rel, quantity, series in plan_series_files(report):
        files[rel] = series_csv_text(report, quantity, series).encode("utf-8")
    for plot in values_plots(report):
        quantity = _quantity_for(report, plot.quantity_id)
        files[plot_path(plot)] = render_plot_png(plot, quantity)
    files[SUMMARY_NAME] = render_summary(report).encode("utf-8")
    return files


def _manifest_bytes(report: SwingReport, files: dict[str, bytes]) -> bytes:
    entries = [
        {"path": rel, "sha256": _sha256(data), "size_bytes": len(data)}
        for rel, data in sorted(files.items())
    ]
    doc = {
        "manifest_version": MANIFEST_VERSION,
        "report_id": report.report_id,
        "files": entries,
    }
    return _json_bytes(doc)


def _check_target(out_dir: Path) -> None:
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise PackageError(f"output directory {out_dir} exists and is not empty")


def export_package(report: SwingReport, out_dir: Path | str) -> dict[str, str]:
    """Write ``report`` as a package under ``out_dir`` and return ``path -> sha256``.

    Preconditions: ``report`` satisfies the contract (re-validated here) and
    ``out_dir`` is absent or an empty directory.
    Postconditions: ``load_package(out_dir) == report``; nothing is written
    when a precondition fails.

    Raises:
        SwingReportContractError: ``report`` violates the contract.
        PackageError: ``out_dir`` is not empty.
    """
    out = Path(out_dir)
    validate_swing_report(report.to_dict())
    _check_target(out)
    files = build_package_files(report)
    files[MANIFEST_NAME] = _manifest_bytes(report, files)
    for rel, data in files.items():
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return {rel: _sha256(data) for rel, data in files.items() if rel != MANIFEST_NAME}


def _read_manifest(root: Path) -> dict[str, str]:
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise PackageError(f"{MANIFEST_NAME} is missing from {root}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return {e["path"]: e["sha256"] for e in doc["files"]}
    except (ValueError, KeyError, TypeError) as exc:
        raise PackageError(f"{MANIFEST_NAME} is malformed: {exc}") from exc


def _on_disk(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != MANIFEST_NAME
    }


def verify_package(root: Path | str) -> dict[str, str]:
    """Check every file against ``MANIFEST.json``; return ``path -> sha256``.

    Raises:
        PackageError: a listed file is missing or altered, or an unlisted file
            is present.
    """
    base = Path(root)
    listed = _read_manifest(base)
    present = _on_disk(base)
    for rel in sorted(set(listed) - present):
        raise PackageError(f"missing file listed in manifest: {rel}")
    for rel in sorted(present - set(listed)):
        raise PackageError(f"unlisted file in package: {rel}")
    for rel, digest in sorted(listed.items()):
        if _sha256((base / rel).read_bytes()) != digest:
            raise PackageError(f"sha256 mismatch for {rel}")
    return listed


def _check_members(base: Path, report: SwingReport, listed: set[str]) -> None:
    expected = {rel for rel, _, _ in plan_series_files(report)}
    expected |= {plot_path(p) for p in values_plots(report)}
    expected |= {REPORT_NAME, SUMMARY_NAME}
    if listed != expected:
        diff = sorted(listed ^ expected)
        raise PackageError(f"package members do not match report: {diff}")
    for rel, quantity, series in plan_series_files(report):
        try:
            check_series_csv((base / rel).read_bytes(), quantity, series)
        except SeriesCsvError as exc:
            raise PackageError(f"{rel}: {exc}") from exc


def load_package(root: Path | str) -> SwingReport:
    """Verify, re-read and validate a package (I13); return the report.

    Raises:
        PackageError: manifest, member set or series CSV check failed.
        SwingReportContractError: ``report.json`` violates the contract.
    """
    base = Path(root)
    listed = verify_package(base)
    try:
        doc = json.loads((base / REPORT_NAME).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PackageError(f"{REPORT_NAME} is not valid JSON: {exc}") from exc
    report = validate_swing_report(doc)
    _check_members(base, report, set(listed))
    return report


__all__ = [
    "MANIFEST_NAME",
    "PackageError",
    "SERIES_DIR",
    "build_package_files",
    "export_package",
    "load_package",
    "verify_package",
]
