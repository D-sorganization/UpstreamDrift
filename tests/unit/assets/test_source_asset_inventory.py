"""Tests for selective LFS policy contracts and source-asset inventory verification.

Part of #14895, parent #14893.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.assets.source_asset_inventory import (
    AssetHydrationState,
    LfsHydrationError,
    inspect_asset_hydration,
    is_lfs_pointer,
    main,
    parse_lfs_pointer,
    scan_inventory,
    verify_asset_hydration,
)

SAMPLE_OID = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SAMPLE_POINTER_TEXT = f"version https://git-lfs.github.com/spec/v1\noid sha256:{SAMPLE_OID}\nsize 1048576\n"
SAMPLE_POINTER_BYTES = SAMPLE_POINTER_TEXT.encode("utf-8")


@pytest.mark.unit
def test_parse_valid_and_invalid_lfs_pointer() -> None:
    """Validate LFS pointer parser against valid spec and malformed text/binary."""
    assert is_lfs_pointer(SAMPLE_POINTER_TEXT)
    ptr = parse_lfs_pointer(SAMPLE_POINTER_TEXT)
    assert ptr is not None and ptr.oid == SAMPLE_OID and ptr.size == 1048576

    assert not is_lfs_pointer(b"PK\x03\x04zip")
    assert parse_lfs_pointer(b"PK\x03\x04zip") is None
    assert (
        parse_lfs_pointer("version https://git-lfs.github.com/spec/v1\nsize 100\n")
        is None
    )


@pytest.mark.unit
def test_inspect_pointer_and_hydrated_asset(tmp_path: Path) -> None:
    """Classify unhydrated pointer vs hydrated binary payload."""
    ptr_file = tmp_path / "model.onnx"
    ptr_file.write_bytes(SAMPLE_POINTER_BYTES)
    r_ptr = inspect_asset_hydration(ptr_file)
    assert (
        r_ptr.status == AssetHydrationState.POINTER_ONLY
        and r_ptr.pointer_oid == SAMPLE_OID
    )

    data = b"\x00\x01\x02\x03" * 64
    bin_file = tmp_path / "table.parquet"
    bin_file.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    r_bin = inspect_asset_hydration(
        bin_file, expected_sha256=sha, expected_size=len(data)
    )
    assert r_bin.status == AssetHydrationState.HYDRATED and r_bin.sha256 == sha


@pytest.mark.unit
def test_fail_closed_on_missing_corrupt_or_pointer(tmp_path: Path) -> None:
    """Fail closed in strict mode on missing, corrupted, or unhydrated assets."""
    r_miss = inspect_asset_hydration(tmp_path / "missing.bin")
    assert r_miss.status == AssetHydrationState.MISSING
    with pytest.raises(LfsHydrationError, match="does not exist"):
        verify_asset_hydration(tmp_path / "missing.bin", strict=True)

    corrupt = tmp_path / "corrupt.bin"
    corrupt.write_bytes(b"bad")
    with pytest.raises(LfsHydrationError, match="SHA-256 mismatch"):
        verify_asset_hydration(corrupt, expected_sha256="0" * 64, strict=True)

    ptr = tmp_path / "stub.zip"
    ptr.write_bytes(SAMPLE_POINTER_BYTES)
    with pytest.raises(LfsHydrationError, match="Unhydrated Git LFS pointer"):
        verify_asset_hydration(ptr, strict=True)


@pytest.mark.unit
def test_scan_inventory_manifest(tmp_path: Path) -> None:
    """scan_inventory verifies multiple manifest entries correctly."""
    f1, f2 = tmp_path / "a1.bin", tmp_path / "a2.bin"
    f1.write_bytes(b"content")
    f2.write_bytes(SAMPLE_POINTER_BYTES)
    sha1 = hashlib.sha256(b"content").hexdigest()

    manifest = {
        "a1.bin": {"sha256": sha1, "size": 7},
        "a2.bin": {"sha256": SAMPLE_OID, "size": 1048576},
    }
    r_map = {r.path: r for r in scan_inventory(manifest, root=tmp_path)}
    assert r_map["a1.bin"].status == AssetHydrationState.HYDRATED
    assert r_map["a2.bin"].status == AssetHydrationState.POINTER_ONLY


@pytest.mark.unit
def test_cli_execution_modes(tmp_path: Path) -> None:
    """CLI returns expected status codes for help, valid file, and strict error."""
    assert main([]) == 2

    valid = tmp_path / "cli.bin"
    valid.write_bytes(b"ok")
    assert (
        main(
            [
                str(valid),
                "--expected-sha",
                hashlib.sha256(b"ok").hexdigest(),
                "--strict",
            ]
        )
        == 0
    )

    ptr = tmp_path / "cli_ptr.bin"
    ptr.write_bytes(SAMPLE_POINTER_BYTES)
    assert main([str(ptr)]) == 0
    assert main([str(ptr), "--strict"]) == 1
