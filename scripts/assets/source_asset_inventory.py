"""Source asset inventory and selective Git LFS hydration verification.

Part of #14895, parent #14893.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

LFS_SPEC_VERSION = "https://git-lfs.github.com/spec/v1"
LFS_OID_REGEX = re.compile(r"^sha256:([a-f0-9]{64})$")


class AssetHydrationState(str, Enum):
    """State of an asset on disk relative to Git LFS and source expectations."""

    HYDRATED = "hydrated"
    POINTER_ONLY = "pointer_only"
    MISSING = "missing"
    CORRUPTED = "corrupted"
    ORDINARY_BLOB = "ordinary_blob"


class LfsHydrationError(RuntimeError):
    """Raised when an asset fails fail-closed hydration or integrity requirements."""


@dataclass(frozen=True)
class LfsPointer:
    """Represents a validated Git LFS pointer descriptor."""

    oid: str
    size: int
    version: str = LFS_SPEC_VERSION


@dataclass
class HydrationReport:
    """Verification result for a single asset path."""

    path: str
    status: AssetHydrationState
    size_bytes: int
    sha256: str | None = None
    pointer_oid: str | None = None
    error_message: str | None = None


def is_lfs_pointer(content: bytes | str) -> bool:
    """Return True if content matches the Git LFS pointer format specification."""
    return parse_lfs_pointer(content) is not None


def parse_lfs_pointer(content: bytes | str) -> LfsPointer | None:
    """Parse a Git LFS pointer text block; returns None if non-conforming."""
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return None
    else:
        text = content

    v, oid, size = None, None, None
    for line in text.strip().splitlines():
        k, _, val = line.strip().partition(" ")
        if k == "version":
            v = val.strip()
        elif k == "oid":
            m = LFS_OID_REGEX.match(val.strip())
            if m:
                oid = m.group(1)
        elif k == "size" and val.strip().isdigit():
            size = int(val.strip())

    if v == LFS_SPEC_VERSION and oid and size is not None:
        return LfsPointer(oid=oid, size=size, version=v)
    return None


def inspect_asset_hydration(
    path: Path | str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> HydrationReport:
    """Inspect an asset on disk without raising, returning a HydrationReport."""
    p = Path(path)
    if not p.is_file():
        return HydrationReport(
            str(p),
            AssetHydrationState.MISSING,
            0,
            error_message=f"Asset does not exist: {p}",
        )

    try:
        data = p.read_bytes()
    except OSError as err:
        return HydrationReport(
            str(p),
            AssetHydrationState.MISSING,
            0,
            error_message=f"Could not read {p}: {err}",
        )

    size = len(data)
    pointer = parse_lfs_pointer(data)
    if pointer:
        return HydrationReport(
            str(p),
            AssetHydrationState.POINTER_ONLY,
            size,
            pointer_oid=pointer.oid,
            error_message=f"Unhydrated Git LFS pointer: expected {pointer.size} B, oid {pointer.oid}",
        )

    computed = hashlib.sha256(data).hexdigest()
    if expected_size is not None and size != expected_size:
        return HydrationReport(
            str(p),
            AssetHydrationState.CORRUPTED,
            size,
            computed,
            error_message=f"Size mismatch: exp {expected_size}, got {size}",
        )
    if expected_sha256 and computed.lower() != expected_sha256.lower():
        return HydrationReport(
            str(p),
            AssetHydrationState.CORRUPTED,
            size,
            computed,
            error_message=f"SHA-256 mismatch: exp {expected_sha256}, got {computed}",
        )

    return HydrationReport(str(p), AssetHydrationState.HYDRATED, size, sha256=computed)


def verify_asset_hydration(
    path: Path | str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    strict: bool = False,
) -> HydrationReport:
    """Verify an asset on disk; raises LfsHydrationError if strict and invalid."""
    rep = inspect_asset_hydration(path, expected_sha256, expected_size)
    if strict and rep.status not in (
        AssetHydrationState.HYDRATED,
        AssetHydrationState.ORDINARY_BLOB,
    ):
        raise LfsHydrationError(
            f"Verification failed for {rep.path}: {rep.error_message}"
        )
    return rep


def scan_inventory(
    manifest: Mapping[str, Mapping[str, Any]], root: Path | str = "."
) -> list[HydrationReport]:
    """Scan multiple assets defined in an inventory mapping."""
    root_path = Path(root)
    reports: list[HydrationReport] = []
    for rel_path, spec in manifest.items():
        rep = inspect_asset_hydration(
            root_path / rel_path, spec.get("sha256"), spec.get("size")
        )
        reports.append(
            HydrationReport(
                rel_path,
                rep.status,
                rep.size_bytes,
                rep.sha256,
                rep.pointer_oid,
                rep.error_message,
            )
        )
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for asset hydration checks."""
    parser = argparse.ArgumentParser(description="Verify source asset hydration.")
    parser.add_argument("path", nargs="?", help="Asset path to check")
    parser.add_argument("--expected-sha", help="Expected SHA-256")
    parser.add_argument("--expected-size", type=int, help="Expected size in bytes")
    parser.add_argument("--strict", action="store_true", help="Fail closed on errors")
    parser.add_argument("--scan", help="Path to JSON manifest")
    args = parser.parse_args(argv)

    if args.scan:
        with open(args.scan, encoding="utf-8") as f:
            manifest = json.load(f)
        reps = scan_inventory(manifest)
        failed = [
            r
            for r in reps
            if r.status
            not in (AssetHydrationState.HYDRATED, AssetHydrationState.ORDINARY_BLOB)
        ]
        for r in reps:
            print(f"[{r.status.value.upper()}] {r.path} ({r.size_bytes} B)")
        return 1 if (failed and args.strict) else 0

    if not args.path:
        parser.print_help()
        return 2

    r = inspect_asset_hydration(args.path, args.expected_sha, args.expected_size)
    print(f"[{r.status.value.upper()}] {r.path} ({r.size_bytes} B)")
    if r.error_message:
        print(f"  Error: {r.error_message}")
    return (
        1
        if (
            args.strict
            and r.status
            not in (AssetHydrationState.HYDRATED, AssetHydrationState.ORDINARY_BLOB)
        )
        else 0
    )


if __name__ == "__main__":
    sys.exit(main())
