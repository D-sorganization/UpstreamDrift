#!/usr/bin/env python3
"""Verify release artifacts, checksum manifests, and tiered SBOMs (#14273)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from email import message_from_string
from pathlib import Path
from typing import Any

from scripts.security.sbom_tiers import expected_artifact_names

logger = logging.getLogger("verify_release_artifacts")


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_sha256sums(manifest_path: Path) -> dict[str, str]:
    """Parse a SHA256SUMS.txt file into a mapping of filename -> digest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")
    mapping: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            mapping[parts[1].lstrip("*").strip()] = parts[0].lower()
    return mapping


def verify_checksum_manifest(
    dist_dir: Path, manifest_name: str = "SHA256SUMS.txt"
) -> tuple[bool, list[str], dict[str, str]]:
    """Verify checksum manifest against all files in dist_dir."""
    manifest_path = dist_dir / manifest_name
    if not manifest_path.is_file():
        return False, [f"Manifest missing: {manifest_path.name}"], {}
    try:
        manifest = parse_sha256sums(manifest_path)
    except (OSError, ValueError) as exc:
        return False, [f"Failed to parse manifest: {exc}"], {}

    errors: list[str] = []
    if not manifest:
        errors.append(f"{manifest_name} contains no checksum entries")
    for filename, expected_digest in manifest.items():
        target = dist_dir / filename
        if not target.is_file():
            errors.append(f"Referenced file missing: {filename}")
        elif compute_sha256(target) != expected_digest:
            errors.append(f"Checksum mismatch for {filename}")

    dist_files = {
        f.name
        for f in dist_dir.iterdir()
        if f.is_file() and f.name != manifest_name and not f.name.startswith(".")
    }
    for u in sorted(dist_files - set(manifest)):
        errors.append(f"Untracked extraneous artifact found: {u}")
    return not errors, errors, manifest


def verify_wheel_metadata(
    wheel_path: Path, expected_version: str | None = None
) -> tuple[bool, list[str]]:
    """Verify that the wheel is valid zip, contains UI bundle and correct METADATA."""
    if not wheel_path.is_file():
        return False, [f"Wheel missing: {wheel_path.name}"]
    errors: list[str] = []
    try:
        with zipfile.ZipFile(wheel_path, "r") as zf:
            names = set(zf.namelist())
            if "ui/dist/index.html" not in names:
                errors.append(f"Wheel {wheel_path.name} missing UI bundle")
            metas = [n for n in names if n.endswith(".dist-info/METADATA")]
            if not metas:
                errors.append(f"Wheel {wheel_path.name} missing .dist-info/METADATA")
            else:
                msg = message_from_string(zf.read(metas[0]).decode("utf-8", "replace"))
                name, ver = msg.get("Name", ""), msg.get("Version", "")
                if name.replace("_", "-") != "upstream-drift":
                    errors.append(f"Wheel metadata Name mismatch: {name}")
                if expected_version and ver != expected_version:
                    errors.append(
                        f"Wheel version mismatch: expected {expected_version}, got {ver}"
                    )
    except zipfile.BadZipFile as exc:
        errors.append(f"Corrupted wheel zip {wheel_path.name}: {exc}")
    return not errors, errors


def verify_sbom_document(
    sbom_path: Path, expected_version: str | None = None
) -> tuple[bool, list[str]]:
    """Validate JSON syntax and schema requirements for CycloneDX and SPDX SBOMs."""
    if not sbom_path.is_file():
        return False, [f"SBOM missing: {sbom_path.name}"]
    try:
        data = json.loads(sbom_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return False, [f"Invalid JSON in SBOM {sbom_path.name}: {exc}"]

    errors: list[str] = []
    name = sbom_path.name.lower()
    if "cyclonedx" in name or name.endswith(".cdx.json"):
        if data.get("bomFormat") != "CycloneDX":
            errors.append(f"{sbom_path.name} bomFormat is not 'CycloneDX'")
        if not data.get("specVersion"):
            errors.append(f"{sbom_path.name} missing specVersion")
        c_ver = data.get("metadata", {}).get("component", {}).get("version")
        if expected_version and c_ver and c_ver != expected_version:
            errors.append(f"{sbom_path.name} metadata version mismatch")
    elif "spdx" in name:
        if not str(data.get("spdxVersion", "")).startswith("SPDX-"):
            errors.append(f"{sbom_path.name} invalid spdxVersion")
        if data.get("SPDXID") != "SPDXRef-DOCUMENT":
            errors.append(f"{sbom_path.name} invalid SPDXID")
        if not isinstance(data.get("packages"), list) or not data["packages"]:
            errors.append(f"{sbom_path.name} missing packages list")
    else:
        errors.append(f"Unknown SBOM format convention for file: {sbom_path.name}")
    return not errors, errors


def verify_release_bundle(
    dist_dir: Path,
    version: str | None = None,
    required_tiers: tuple[str, ...] = ("core", "extended", "full"),
) -> dict[str, Any]:
    """Verify an entire release bundle directory (checksums, wheels, SBOMs)."""
    if not dist_dir.is_dir():
        return {
            "status": "failed",
            "summary": {"verified_files": 0, "error_count": 1},
            "errors": [f"Release directory not found: {dist_dir}"],
        }

    chk_ok, chk_errs, manifest = verify_checksum_manifest(dist_dir)
    errors: list[str] = list(chk_errs)
    wheels = sorted(dist_dir.glob("upstream_drift-*.whl"))
    target_version = version or (wheels[0].stem.split("-")[1] if wheels else None)

    if not wheels:
        errors.append("No upstream_drift wheel artifact found in dist directory")
    for whl in wheels:
        _, whl_errs = verify_wheel_metadata(whl, expected_version=target_version)
        errors.extend(whl_errs)

    if target_version:
        for expected_name in expected_artifact_names(target_version):
            tier = expected_name.split(".")[-2]
            if tier not in required_tiers:
                continue
            sbom_file = dist_dir / expected_name
            if not sbom_file.is_file():
                errors.append(
                    f"Missing required SBOM for tier '{tier}': {expected_name}"
                )
            else:
                _, s_errs = verify_sbom_document(
                    sbom_file, expected_version=target_version
                )
                errors.extend(s_errs)

    return {
        "status": "passed" if not errors else "failed",
        "summary": {
            "version": target_version,
            "verified_files": len(manifest),
            "error_count": len(errors),
        },
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for release artifact verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = verify_release_bundle(args.dist_dir, version=args.version)
    if args.json:
        print(json.dumps(report, indent=2))
    elif report["status"] == "passed":
        logger.info(
            "Verification PASSED (%d files)", report["summary"]["verified_files"]
        )
    else:
        logger.error(
            "Verification FAILED (%d errors)", report["summary"]["error_count"]
        )
        for err in report["errors"]:
            logger.error("  - %s", err)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
