"""Tests for release artifact and SBOM verification (issue #14273)."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import verify_release_artifacts as mod

pytestmark = pytest.mark.unit
VERSION = "2.1.3"


def _create_wheel(path: Path, version: str, *, include_ui: bool = True) -> None:
    meta = f"Metadata-Version: 2.1\nName: upstream-drift\nVersion: {version}\n"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"upstream_drift-{version}.dist-info/METADATA", meta)
        if include_ui:
            zf.writestr("ui/dist/index.html", "<!doctype html>")


def _create_cdx(ver: str, tier: str) -> dict[str, Any]:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "metadata": {"component": {"name": "upstream-drift", "version": ver}},
        "components": [{"name": "numpy", "version": "1.26.4", "scope": tier}],
    }


def _create_spdx(ver: str, tier: str) -> dict[str, Any]:
    return {
        "spdxVersion": "SPDX-2.3",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"upstream-drift-{ver}.{tier}",
        "packages": [{"name": "upstream-drift", "versionInfo": ver}],
    }


def _setup_dist(
    dist_dir: Path,
    version: str = VERSION,
    *,
    include_ui: bool = True,
    tamper_file: str | None = None,
    omit_file: str | None = None,
    extra_file: bool = False,
) -> None:
    dist_dir.mkdir(parents=True, exist_ok=True)
    _create_wheel(
        dist_dir / f"upstream_drift-{version}-py3-none-any.whl",
        version,
        include_ui=include_ui,
    )
    for tier in ("core", "extended", "full"):
        (dist_dir / f"upstream-drift-{version}.cyclonedx.{tier}.json").write_text(
            json.dumps(_create_cdx(version, tier)), encoding="utf-8"
        )
        (dist_dir / f"upstream-drift-{version}.spdx.{tier}.json").write_text(
            json.dumps(_create_spdx(version, tier)), encoding="utf-8"
        )
    if omit_file:
        (dist_dir / omit_file).unlink(missing_ok=True)
    lines = [
        f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n"
        for f in sorted(dist_dir.iterdir())
        if f.is_file()
    ]
    (dist_dir / "SHA256SUMS.txt").write_text("".join(lines), encoding="utf-8")
    if tamper_file:
        (dist_dir / tamper_file).write_text("tampered", encoding="utf-8")
    if extra_file:
        (dist_dir / "extra.bin").write_bytes(b"\x00")


def _update_manifest_hash(dist_dir: Path, filename: str) -> None:
    digest = hashlib.sha256((dist_dir / filename).read_bytes()).hexdigest()
    lines = [
        f"{digest}  {filename}\n" if filename in line else line
        for line in (dist_dir / "SHA256SUMS.txt")
        .read_text(encoding="utf-8")
        .splitlines(keepends=True)
    ]
    (dist_dir / "SHA256SUMS.txt").write_text("".join(lines), encoding="utf-8")


def test_verify_release_bundle_success(tmp_path: Path) -> None:
    _setup_dist(tmp_path / "dist", VERSION)
    report = mod.verify_release_bundle(tmp_path / "dist", version=VERSION)
    assert report["status"] == "passed" and not report["errors"]


def test_checksum_mismatch_fails(tmp_path: Path) -> None:
    _setup_dist(
        tmp_path / "dist",
        VERSION,
        tamper_file=f"upstream-drift-{VERSION}.cyclonedx.core.json",
    )
    report = mod.verify_release_bundle(tmp_path / "dist", version=VERSION)
    assert report["status"] == "failed" and any(
        "mismatch" in e for e in report["errors"]
    )


def test_missing_checksum_manifest_fails(tmp_path: Path) -> None:
    _setup_dist(tmp_path / "dist", VERSION)
    (tmp_path / "dist" / "SHA256SUMS.txt").unlink()
    report = mod.verify_release_bundle(tmp_path / "dist", version=VERSION)
    assert report["status"] == "failed" and any(
        "missing" in e for e in report["errors"]
    )


def test_untracked_extraneous_file_fails(tmp_path: Path) -> None:
    _setup_dist(tmp_path / "dist", VERSION, extra_file=True)
    report = mod.verify_release_bundle(tmp_path / "dist", version=VERSION)
    assert report["status"] == "failed" and any(
        "untracked" in e.lower() for e in report["errors"]
    )


def test_missing_sbom_tier_fails(tmp_path: Path) -> None:
    _setup_dist(
        tmp_path / "dist", VERSION, omit_file=f"upstream-drift-{VERSION}.spdx.full.json"
    )
    report = mod.verify_release_bundle(tmp_path / "dist", version=VERSION)
    assert report["status"] == "failed" and any(
        "missing required sbom" in e.lower() for e in report["errors"]
    )


def test_corrupted_sbom_json_fails(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _setup_dist(dist, VERSION)
    fname = f"upstream-drift-{VERSION}.cyclonedx.extended.json"
    (dist / fname).write_text("{bad json", encoding="utf-8")
    _update_manifest_hash(dist, fname)
    report = mod.verify_release_bundle(dist, version=VERSION)
    assert report["status"] == "failed" and any(
        "invalid json" in e.lower() for e in report["errors"]
    )


def test_invalid_cyclonedx_format_fails(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _setup_dist(dist, VERSION)
    fname = f"upstream-drift-{VERSION}.cyclonedx.core.json"
    (dist / fname).write_text(json.dumps({"bomFormat": "Other"}), encoding="utf-8")
    _update_manifest_hash(dist, fname)
    report = mod.verify_release_bundle(dist, version=VERSION)
    assert report["status"] == "failed" and any(
        "cyclonedx" in e.lower() for e in report["errors"]
    )


def test_invalid_spdx_document_fails(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _setup_dist(dist, VERSION)
    fname = f"upstream-drift-{VERSION}.spdx.core.json"
    (dist / fname).write_text(json.dumps({"spdxVersion": "BAD"}), encoding="utf-8")
    _update_manifest_hash(dist, fname)
    report = mod.verify_release_bundle(dist, version=VERSION)
    assert report["status"] == "failed" and any(
        "spdx" in e.lower() for e in report["errors"]
    )


def test_wheel_missing_ui_bundle_fails(tmp_path: Path) -> None:
    _setup_dist(tmp_path / "dist", VERSION, include_ui=False)
    report = mod.verify_release_bundle(tmp_path / "dist", version=VERSION)
    assert report["status"] == "failed" and any(
        "ui bundle" in e.lower() for e in report["errors"]
    )


def test_version_mismatch_fails(tmp_path: Path) -> None:
    _setup_dist(tmp_path / "dist", VERSION)
    report = mod.verify_release_bundle(tmp_path / "dist", version="2.1.4")
    assert report["status"] == "failed" and any(
        "mismatch" in e.lower() for e in report["errors"]
    )


def test_cli_invocation_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup_dist(tmp_path / "dist", VERSION)
    assert (
        mod.main(["--dist-dir", str(tmp_path / "dist"), "--version", VERSION, "--json"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
