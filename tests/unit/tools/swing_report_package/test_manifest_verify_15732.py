"""Fail-closed ``manifest.json`` verification and resume key (#15732, SRP-6)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.tools.swing_report_package import manifest_verify as mv

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
H = "sha256:" + "a" * 64


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(root: Path, **over: Any) -> dict[str, Any]:
    (root / "grid.mp4").write_bytes(b"video")
    m: dict[str, Any] = {
        "manifest_schema_version": 1,
        "kind": "dev",
        "swing": "s1",
        "run": {
            "run_id": H,
            "model_definition_hash": H,
            "subject_profile_hash": H,
            "match_artifact_hash": H,
        },
        "tensor_hash": "b" * 64,
        "settings_hash": "c" * 64,
        "exporter_code_hash": "d" * 64,
        "git": {"commit": "e" * 40, "dirty": False},
        "jobs": [{"key": "k", "status": "done", "outputs": ["grid.mp4"]}],
        "phases": [],
        "artifacts": [{"path": "grid.mp4", "sha256": _sha(b"video")}],
    }
    m.update(over)
    return m


def test_clean_manifest_is_ok(tmp_path: Path) -> None:
    v = mv.verify_manifest(_manifest(tmp_path), tmp_path)
    assert v.ok and v.label == "OK" and v.exit_code == 0 and v.issues == ()


def test_unknown_top_level_field_is_an_error(tmp_path: Path) -> None:
    v = mv.verify_manifest(_manifest(tmp_path, surprise=1), tmp_path)
    assert not v.ok and v.label == "PARTIAL" and v.exit_code != 0
    assert "unknown_field" in {i.code for i in v.issues}


def test_unknown_schema_version(tmp_path: Path) -> None:
    v = mv.verify_manifest(_manifest(tmp_path, manifest_schema_version=99), tmp_path)
    assert "schema_version" in {i.code for i in v.issues}


@pytest.mark.parametrize(
    "ref", ["latest", "main", "feat/x", "C:/models/a.xml", "/m/a", "../a"]
)
def test_mutable_or_path_refs_are_forbidden(tmp_path: Path, ref: str) -> None:
    m = _manifest(tmp_path)
    m["run"]["model_definition_hash"] = ref
    assert "mutable_ref" in {i.code for i in mv.verify_manifest(m, tmp_path).issues}


def test_missing_run_identity_field(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    del m["run"]["match_artifact_hash"]
    assert "missing_field" in {i.code for i in mv.verify_manifest(m, tmp_path).issues}


def test_corrupted_artifact_is_detected(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    (tmp_path / "grid.mp4").write_bytes(b"tampered")
    assert "hash_mismatch" in {i.code for i in mv.verify_manifest(m, tmp_path).issues}


def test_missing_artifact_is_detected(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    (tmp_path / "grid.mp4").unlink()
    assert "missing_artifact" in {
        i.code for i in mv.verify_manifest(m, tmp_path).issues
    }


def test_artifact_path_may_not_escape_root(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    m["artifacts"][0]["path"] = "../outside.mp4"
    assert "bad_path" in {i.code for i in mv.verify_manifest(m, tmp_path).issues}


def test_failed_job_marks_package_partial(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    m["jobs"].append({"key": "j2", "status": "failed", "outputs": []})
    v = mv.verify_manifest(m, tmp_path)
    assert v.label == "PARTIAL" and "incomplete_job" in {i.code for i in v.issues}


def test_dirty_tree_blocked_for_release_but_not_dev(tmp_path: Path) -> None:
    dirty = {
        "commit": "e" * 40,
        "dirty": True,
        "patch_sha256": "f" * 64,
        "untracked": [],
    }
    dev = mv.verify_manifest(_manifest(tmp_path, git=dirty), tmp_path)
    assert dev.ok
    rel = mv.verify_manifest(_manifest(tmp_path, git=dirty, kind="release"), tmp_path)
    assert "dirty_release" in {i.code for i in rel.issues}


def test_dirty_dev_run_needs_patch_and_untracked_list(tmp_path: Path) -> None:
    v = mv.verify_manifest(
        _manifest(tmp_path, git={"commit": "e" * 40, "dirty": True}), tmp_path
    )
    assert "missing_field" in {i.code for i in v.issues}


def test_resume_key_changes_with_each_component(tmp_path: Path) -> None:
    base = _manifest(tmp_path)
    key = mv.resume_key(base)
    assert len(key) == 5
    for field in ("tensor_hash", "settings_hash", "exporter_code_hash"):
        other = copy.deepcopy(base)
        other[field] = "0" * 64
        assert mv.resume_key(other) != key
    other = copy.deepcopy(base)
    other["artifacts"][0]["sha256"] = "0" * 64
    assert mv.resume_key(other) != key
    other = copy.deepcopy(base)
    other["manifest_schema_version"] = 2
    assert mv.resume_key(other) != key


def test_publish_is_atomic_and_round_trips(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    path = mv.publish_manifest(tmp_path, m)
    assert json.loads(path.read_text(encoding="utf-8")) == m
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_publish_refuses_a_manifest_that_does_not_verify(tmp_path: Path) -> None:
    with pytest.raises(mv.ManifestError):
        mv.publish_manifest(tmp_path, _manifest(tmp_path, surprise=1))
    assert not (tmp_path / "manifest.json").exists()
