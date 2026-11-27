"""Unit tests for research artifact release asset fetcher and verification.

Contracts tested:
- Release index schema and path safety validation
- Exact size and SHA-256 integrity verification
- Fail-closed behavior on missing, wrong-size, or corrupted downloads
- Atomic write with zero residual on failure
- Selective path filtering via `only`
- Offline / non-download verification reporting (`verify_assets`)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.shot_pattern_analysis.fetch_release_assets import (
    AssetIndex,
    AssetIntegrityError,
    ReleaseAsset,
    VerificationReport,
    check_asset_status,
    fetch,
    load_index,
    verify_assets,
)

SAMPLE_SHA = hashlib.sha256(b"hello research asset").hexdigest()
SAMPLE_BYTES = len(b"hello research asset")
SAMPLE_PATH = "docs/research/shot_pattern_analysis/test_sample.parquet"


def _make_index(tmp_path: Path, files: dict | None = None) -> Path:
    if files is None:
        files = {
            SAMPLE_PATH: {
                "asset": "test_sample.parquet",
                "bytes": SAMPLE_BYTES,
                "sha256": SAMPLE_SHA,
            }
        }
    data = {
        "schema": "shot-pattern-release-assets/1",
        "release_tag": "test-tag-2026",
        "files": files,
    }
    index_file = tmp_path / "RELEASE_ASSETS.json"
    index_file.write_text(json.dumps(data), encoding="utf-8")
    return index_file


@pytest.mark.unit
def test_load_index_valid(tmp_path: Path) -> None:
    index_file = _make_index(tmp_path)
    index = load_index(index_file)
    assert isinstance(index, AssetIndex)
    assert index.release_tag == "test-tag-2026"
    assert len(index.entries) == 1
    entry = index.entries[0]
    assert entry.path == SAMPLE_PATH
    assert entry.bytes == SAMPLE_BYTES
    assert entry.sha256 == SAMPLE_SHA


@pytest.mark.unit
def test_load_index_invalid_schema(tmp_path: Path) -> None:
    index_file = tmp_path / "bad_schema.json"
    index_file.write_text(
        json.dumps({"schema": "unknown/99", "release_tag": "t", "files": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected release index schema"):
        load_index(index_file)


@pytest.mark.unit
def test_load_index_rejects_unsafe_paths(tmp_path: Path) -> None:
    bad_paths = [
        "../outside.txt",
        "/absolute/path.txt",
        "docs/research/other/file.txt",
        "docs/research/shot_pattern_analysis/../../escape.txt",
    ]
    for bad in bad_paths:
        idx = _make_index(
            tmp_path,
            {bad: {"asset": "a.txt", "bytes": 10, "sha256": "a" * 64}},
        )
        with pytest.raises(ValueError, match="POSIX path under|unsafe asset path"):
            load_index(idx)


@pytest.mark.unit
def test_check_asset_status_matched_missing_corrupted(tmp_path: Path) -> None:
    entry = ReleaseAsset(SAMPLE_PATH, "test.bin", SAMPLE_BYTES, SAMPLE_SHA)
    target = tmp_path / SAMPLE_PATH

    assert check_asset_status(target, entry) == "missing"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"corrupted content of same len")
    assert check_asset_status(target, entry) == "corrupted"

    target.write_bytes(b"hello research asset")
    assert check_asset_status(target, entry) == "matched"


@pytest.mark.unit
def test_verify_assets_report(tmp_path: Path) -> None:
    index_file = _make_index(
        tmp_path,
        {
            f"docs/research/shot_pattern_analysis/file_{i}.bin": {
                "asset": f"file_{i}.bin",
                "bytes": SAMPLE_BYTES,
                "sha256": SAMPLE_SHA,
            }
            for i in range(3)
        },
    )

    report_before = verify_assets(index_file, tmp_path)
    assert isinstance(report_before, VerificationReport)
    assert report_before.total_checked == 3
    assert report_before.matched == 0
    assert len(report_before.missing) == 3
    assert not report_before.is_valid

    p0 = tmp_path / "docs/research/shot_pattern_analysis/file_0.bin"
    p0.parent.mkdir(parents=True, exist_ok=True)
    p0.write_bytes(b"hello research asset")

    p1 = tmp_path / "docs/research/shot_pattern_analysis/file_1.bin"
    p1.write_bytes(b"corrupt payload bytes")

    report_partial = verify_assets(index_file, tmp_path)
    assert report_partial.matched == 1
    assert len(report_partial.corrupted) == 1
    assert len(report_partial.missing) == 1
    assert not report_partial.is_valid


@pytest.mark.unit
def test_fetch_downloads_missing_and_writes_atomically(tmp_path: Path) -> None:
    index_file = _make_index(tmp_path)
    downloaded: list[tuple[str, str]] = []

    def mock_download(tag: str, asset: str) -> bytes:
        downloaded.append((tag, asset))
        return b"hello research asset"

    written = fetch(index_file, tmp_path, download=mock_download)
    assert len(written) == 1
    assert (tmp_path / SAMPLE_PATH).read_bytes() == b"hello research asset"
    assert len(downloaded) == 1

    # Second fetch should skip already matched file
    written_second = fetch(index_file, tmp_path, download=mock_download)
    assert len(written_second) == 0
    assert len(downloaded) == 1


@pytest.mark.unit
def test_fetch_fails_closed_on_corrupt_or_wrong_size(tmp_path: Path) -> None:
    index_file = _make_index(tmp_path)
    target = tmp_path / SAMPLE_PATH

    def bad_hash_download(tag: str, asset: str) -> bytes:
        return b"corrupted bytes here"

    with pytest.raises(AssetIntegrityError, match="does not match expected sha256"):
        fetch(index_file, tmp_path, download=bad_hash_download)

    assert not target.exists(), "Target file must not be created on hash mismatch"

    def bad_size_download(tag: str, asset: str) -> bytes:
        return b"wrong"

    with pytest.raises(AssetIntegrityError, match="does not match expected size"):
        fetch(index_file, tmp_path, download=bad_size_download)

    assert not target.exists(), "Target file must not be created on size mismatch"


@pytest.mark.unit
def test_fetch_respects_only_filter(tmp_path: Path) -> None:
    index_file = _make_index(
        tmp_path,
        {
            "docs/research/shot_pattern_analysis/subset_a/file.bin": {
                "asset": "a.bin",
                "bytes": SAMPLE_BYTES,
                "sha256": SAMPLE_SHA,
            },
            "docs/research/shot_pattern_analysis/subset_b/file.bin": {
                "asset": "b.bin",
                "bytes": SAMPLE_BYTES,
                "sha256": SAMPLE_SHA,
            },
        },
    )

    def mock_download(tag: str, asset: str) -> bytes:
        return b"hello research asset"

    written = fetch(index_file, tmp_path, download=mock_download, only="subset_b")
    assert len(written) == 1
    assert "subset_b" in str(written[0])
    assert not (
        tmp_path / "docs/research/shot_pattern_analysis/subset_a/file.bin"
    ).exists()
