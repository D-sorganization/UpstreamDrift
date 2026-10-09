"""Portable study bundles preserve bytes and detect alteration."""

import json
import zipfile

import pytest

from scripts.research.shot_pattern_analysis.archive_data import (
    build_archives,
    verify_archives,
)

pytestmark = pytest.mark.unit


def test_round_trip_and_reproducibility(tmp_path):
    source = tmp_path / "pilot"
    source.mkdir()
    (source / "matrix_plan.json").write_text('{"cells": {}}\n')
    (source / "nested").mkdir()
    (source / "nested" / "shots.csv.gz").write_bytes(bytes(range(256)) * 4)
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = build_archives(source, first, max_bytes=1300)
    build_archives(source, second, max_bytes=1300)
    assert verify_archives(first) == 2
    assert len(manifest["archives"]) == 2
    restored = tmp_path / "restored"
    for name in manifest["archives"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert (first / name).stat().st_size <= 1300
        with zipfile.ZipFile(first / name) as archive:
            archive.extractall(restored)
    assert (restored / "pilot/nested/shots.csv.gz").read_bytes() == bytes(
        range(256)
    ) * 4
    assert (first / "archive_manifest.json").read_bytes() == (
        second / "archive_manifest.json"
    ).read_bytes()


def test_corrupt_archive_rejected(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    (source / "one").write_bytes(b"abc")
    output = tmp_path / "out"
    manifest = build_archives(source, output)
    archive = output / next(iter(manifest["archives"]))
    archive.write_bytes(archive.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="archive hash"):
        verify_archives(output)


def test_unlisted_member_rejected(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    (source / "one").write_bytes(b"abc")
    output = tmp_path / "out"
    build_archives(source, output)
    manifest = json.loads((output / "archive_manifest.json").read_text())
    manifest["files"] = {}
    (output / "archive_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="member inventory"):
        verify_archives(output)


@pytest.mark.parametrize("kind", ["nested_output", "symlink", "oversize", "pending"])
def test_invalid_input_has_no_published_output(tmp_path, kind):
    source = tmp_path / "data"
    source.mkdir()
    (source / "one").write_bytes(b"x" * 2000)
    output = source / "output" if kind == "nested_output" else tmp_path / "out"
    if kind == "symlink":
        (source / "link").symlink_to(source / "one")
    if kind == "pending":
        (source / ".pending-run").mkdir()
    with pytest.raises(ValueError):
        build_archives(source, output, max_bytes=1000 if kind == "oversize" else 10000)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["replace_earlier", "add_file"])
def test_source_mutation_before_publication_rejected(tmp_path, monkeypatch, mutation):
    from scripts.research.shot_pattern_analysis import archive_data

    source = tmp_path / "data"
    source.mkdir()
    (source / "one").write_bytes(b"initial")
    output = tmp_path / "out"
    original_verify = archive_data.verify_archives

    def mutate_after_verification(path):
        count = original_verify(path)
        target = source / ("one" if mutation == "replace_earlier" else "new")
        target.write_bytes(b"changed")
        return count

    monkeypatch.setattr(archive_data, "verify_archives", mutate_after_verification)
    with pytest.raises(ValueError, match="source changed"):
        archive_data.build_archives(source, output)
    assert not output.exists()


def test_consistent_symlink_archive_rejected(tmp_path):
    import hashlib
    import stat

    source = tmp_path / "data"
    source.mkdir()
    (source / "one").write_bytes(b"elsewhere")
    output = tmp_path / "out"
    manifest = build_archives(source, output)
    name = next(iter(manifest["archives"]))
    path = output / name
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("data/one")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, b"elsewhere")
    manifest["archives"][name] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    (output / "archive_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="regular file"):
        verify_archives(output)


@pytest.mark.parametrize("member", ["C:/escape", "C:escape"])
def test_windows_drive_member_rejected(tmp_path, member):
    import hashlib

    output = tmp_path / "out"
    output.mkdir()
    archive = output / "data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        info = zipfile.ZipInfo(member)
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        bundle.writestr(info, b"abc")
    manifest = {
        "archives": {
            "data.zip": {
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "bytes": archive.stat().st_size,
            }
        },
        "files": {
            member: {
                "archive": "data.zip",
                "sha256": hashlib.sha256(b"abc").hexdigest(),
                "bytes": 3,
            }
        },
    }
    (output / "archive_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe archive member"):
        verify_archives(output)
