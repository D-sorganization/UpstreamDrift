"""Unit tests for the pinned MyoFullBody asset fetch (issue #11643)."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile

import pytest
from src.shared.python.myofullbody import assets

pytestmark = pytest.mark.unit

COMMIT = "a" * 40


def _archive(files: dict[str, bytes], commit: str = COMMIT) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, payload in files.items():
            info = tarfile.TarInfo(f"myo_sim-{commit}/{name}")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _fake() -> tuple[bytes, assets.AssetManifest]:
    files = {
        "myo_sim/__init__.py": b"VALUE = 1\n",
        "myo_sim/models/a.xml": b"<mujoco/>",
        "LICENSE": b"Apache",
        "README.md": b"ignored",
    }
    data = _archive(files)
    document = assets.generate_manifest(data, COMMIT)
    manifest = assets.AssetManifest(
        document["repository"], COMMIT, document["files"], document["license"]
    )
    return data, manifest


def test_generate_manifest_lists_only_package_files() -> None:
    data, manifest = _fake()
    assert sorted(manifest.files) == [
        "LICENSE",
        "myo_sim/__init__.py",
        "myo_sim/models/a.xml",
    ]
    assert manifest.license["owner_ruling"].startswith("2026-10-07")


def test_fetch_extracts_and_is_idempotent(tmp_path: Path) -> None:
    data, manifest = _fake()
    calls: list[int] = []

    def download() -> bytes:
        calls.append(1)
        return data

    first = assets.fetch(manifest, root=tmp_path, download=download)
    second = assets.fetch(manifest, root=tmp_path, download=download)
    assert first == second == tmp_path / COMMIT
    assert len(calls) == 1  # the verified cache is reused without a download
    assert assets.verify_tree(first, manifest) == []
    assert assets.cached_tree(manifest, tmp_path) == first


def test_tampered_archive_is_rejected_and_nothing_is_written(tmp_path: Path) -> None:
    _, manifest = _fake()
    evil = _archive(
        {
            "myo_sim/__init__.py": b"import os\n",
            "myo_sim/models/a.xml": b"<mujoco/>",
            "LICENSE": b"Apache",
        }
    )
    with pytest.raises(assets.AssetIntegrityError, match="sha256 mismatch"):
        assets.fetch(manifest, root=tmp_path, download=lambda: evil)
    assert not (tmp_path / COMMIT).exists()
    assert assets.cached_tree(manifest, tmp_path) is None


def test_missing_member_is_rejected(tmp_path: Path) -> None:
    _, manifest = _fake()
    short = _archive({"myo_sim/__init__.py": b"VALUE = 1\n", "LICENSE": b"Apache"})
    with pytest.raises(assets.AssetIntegrityError, match="missing"):
        assets.extract_verified(short, manifest, tmp_path / "x")


def test_tampering_with_the_cache_is_detected(tmp_path: Path) -> None:
    data, manifest = _fake()
    tree = assets.fetch(manifest, root=tmp_path, download=lambda: data)
    (tree / "myo_sim" / "models" / "a.xml").write_text("<changed/>")
    (tree / "myo_sim" / "extra.py").write_text("x = 1")
    problems = assets.verify_tree(tree, manifest)
    assert any("sha256 mismatch" in p for p in problems)
    assert any("unexpected python file" in p for p in problems)
    assert assets.cached_tree(manifest, tmp_path) is None
    # a refetch repairs the cache
    repaired = assets.fetch(manifest, root=tmp_path, download=lambda: data)
    assert assets.verify_tree(repaired, manifest) == []


def test_garbage_archive_is_an_integrity_error(tmp_path: Path) -> None:
    _, manifest = _fake()
    with pytest.raises(assets.AssetIntegrityError, match="unreadable"):
        assets.extract_verified(b"not a tarball", manifest, tmp_path / "x")


def test_manifest_rejects_unsafe_paths_and_bad_digests() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        assets.AssetManifest("r", COMMIT, {"../x": "0" * 64}, {})
    with pytest.raises(ValueError, match="bad sha256"):
        assets.AssetManifest("r", COMMIT, {"a": "xyz"}, {})
    with pytest.raises(ValueError, match="at least one"):
        assets.AssetManifest("r", COMMIT, {}, {})


def test_committed_manifest_is_pinned_and_records_the_license() -> None:
    manifest = assets.load_manifest()
    assert manifest.commit == assets.PINNED_COMMIT
    assert len(manifest.commit) == 40
    assert "LICENSE" in manifest.files
    assert any(n.startswith("myo_sim/models/") for n in manifest.files)
    text = json.dumps(manifest.license)
    assert "non-commercial" in text and "Apache-2.0" in text
    assert "MoBL" in text


def test_cache_root_honours_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(assets.CACHE_ENV_VAR, str(tmp_path))
    assert assets.default_cache_root() == tmp_path


def test_myofullbody_loads_and_receipt_counts_416_muscles() -> None:
    pytest.importorskip("mujoco")
    tree = assets.cached_tree()
    if tree is None:
        pytest.skip("MyoFullBody cache absent; run scripts/fetch_myofullbody.py")
    manifest = assets.load_manifest()
    receipt = assets.asset_receipt(tree, manifest)
    inventory = receipt["inventory"]
    assert inventory["muscles"] == assets.EXPECTED_MUSCLES == 416
    assert inventory["joints"] == 123 and inventory["bodies"] == 104
    assert receipt["muscle_count_ok"] is True
