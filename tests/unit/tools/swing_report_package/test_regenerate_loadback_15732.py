"""Tests for SRP-6 deterministic regeneration CLI and #15660 GUI load-back (#15732)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.motion_matching import model_library_contracts as mlc
from src.shared.python.motion_matching.execution import spec_revisions as sr
from src.tools.swing_report_package import manifest_verify as mv
from src.tools.swing_report_package import regenerate_and_loadback as rl

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
_FIX = Path(__file__).resolve().parents[3] / "fixtures/club_face/spec_driver.json"


def _tag(label: str) -> str:
    """Return deterministic SHA-256 hex digest for a test label."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _spec() -> dict[str, Any]:
    """Build a canonical full-body golfer spec from the shared fixture."""
    spec = json.loads(_FIX.read_text("utf-8"))
    spec["subject"]["hand_visual_backend"] = "skinned_human"
    spec["subject"]["body_visual_backend"] = "skinned_cc0"
    spec["spec_revision"] = {"version": 1, "configuration": "skinned_hand_wrists"}
    return sr.with_current_shaft(spec)


def _manifest(root: Path, **over: Any) -> dict[str, Any]:
    """Create a valid package artifact and write manifest.json under root."""
    (root / "grid.mp4").write_bytes(b"video-bytes")
    h = hashlib.sha256(b"video-bytes").hexdigest()
    run = {"run_id": f"sha256:{_tag('r')}", "model_definition_hash": _tag("m")}
    run.update(subject_profile_hash=_tag("s"), match_artifact_hash=_tag("x"))
    doc: dict[str, Any] = {
        "manifest_schema_version": 1,
        "kind": "dev",
        "swing": "s1",
        "run": run,
        "tensor_hash": _tag("t"),
        "settings_hash": _tag("c"),
        "exporter_code_hash": _tag("e"),
        "git": {"commit": "e" * 40, "dirty": False},
        "jobs": [{"key": "j1", "status": "done", "outputs": ["grid.mp4"]}],
        "phases": [],
        "artifacts": [{"path": "grid.mp4", "sha256": h}],
    }
    doc.update(over)
    (root / "manifest.json").write_text(json.dumps(doc), "utf-8")
    return doc


def _lib(root: Path) -> tuple[Any, Any, Any, Path]:
    """Populate a content-addressed model library with a linked triad."""
    m = mlc.ModelDefinition.from_spec(_spec(), model_id="driver_v2")
    s = mlc.SubjectProfile("subj_01", "right", 1.80, 78.0, _tag("c3d"))
    x = mlc.MatchArtifact("match_0001", m.content_sha256, s.content_sha256, _tag("tr"))
    mp = mlc.save_artifact(root, m)
    mlc.save_artifact(root, s)
    mlc.save_artifact(root, x)
    mlc.ModelLibraryManifest.rebuild_from_directory(root).save(root)
    return m, s, x, mp


def test_corrupted_artifact_partial(tmp_path: Path, capsys: Any) -> None:
    """Corrupted artifact bytes yield PARTIAL status and non-zero CLI exit code."""
    _manifest(tmp_path)
    (tmp_path / "grid.mp4").write_bytes(b"tampered")
    res = rl.regenerate_from_manifest(tmp_path)
    assert res.status == "PARTIAL" and res.exit_code != 0
    assert "hash_mismatch" in {i.code for i in res.issues}
    assert rl.main(["regenerate", "--package-dir", str(tmp_path)]) != 0
    assert json.loads(capsys.readouterr().out)["status"] == "PARTIAL"


@pytest.mark.parametrize(
    ("k", "v"), [("manifest_schema_version", 99), ("extra", 1), ("mut", "latest")]
)
def test_bad_schema_or_mutable_ref(tmp_path: Path, k: str, v: Any) -> None:
    """Incompatible schema, unknown fields, and mutable refs fail closed."""
    doc = _manifest(tmp_path)
    if k == "mut":
        doc["run"]["model_definition_hash"] = v
    else:
        doc[k] = v
    (tmp_path / "manifest.json").write_text(json.dumps(doc), "utf-8")
    with pytest.raises(mv.ManifestError):
        rl.regenerate_from_manifest(tmp_path)
    assert rl.main(["regenerate", "--package-dir", str(tmp_path)]) != 0


def test_dirty_tree_dev_vs_release(tmp_path: Path) -> None:
    """Dirty git state is permitted for dev runs with patch, blocked for release."""
    c, p = "a" * 40, "b" * 64
    dirty = {"commit": c, "dirty": True, "patch_sha256": p, "untracked": []}
    _manifest(tmp_path, kind="dev", git=dirty)
    dev_res = rl.regenerate_from_manifest(tmp_path, release=False)
    assert dev_res.status == "OK" and dev_res.exit_code == 0
    with pytest.raises(mv.ManifestError):
        rl.regenerate_from_manifest(tmp_path, release=True)
    _manifest(tmp_path, kind="release", git=dirty)
    with pytest.raises(mv.ManifestError):
        rl.regenerate_from_manifest(tmp_path)


def test_resume_key_skips_and_runs(tmp_path: Path) -> None:
    """Verified complete jobs skip while incomplete jobs run and atomically publish."""
    jobs = [
        {"key": "j1", "status": "done", "outputs": ["grid.mp4"]},
        {"key": "j2", "status": "failed", "outputs": []},
    ]
    initial_key, ran = mv.resume_key(_manifest(tmp_path, jobs=jobs)), []

    def runner(job: dict[str, Any], root: Path) -> list[str]:
        """Produce output artifact for incomplete job."""
        ran.append(str(job["key"]))
        (root / f"{job['key']}.png").write_bytes(b"png-bytes")
        return [f"{job['key']}.png"]

    res = rl.regenerate_from_manifest(tmp_path, job_runner=runner)
    assert ran == ["j2"] and res.status == "OK" and res.exit_code == 0
    assert res.resume_key != initial_key
    assert [e.action for e in res.journal] == ["skipped", "ran"]
    assert res.journal_path is not None and res.journal_path.is_file()
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    updated = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    assert mv.verify_manifest(updated, tmp_path).ok
    assert res.resume_key == mv.resume_key(updated)
    ran.clear()
    again = rl.regenerate_from_manifest(tmp_path, job_runner=runner)
    assert ran == [] and [e.action for e in again.journal] == ["skipped", "skipped"]
    rl.regenerate_from_manifest(tmp_path, job_runner=runner, tensor_hash=_tag("new-t"))
    assert ran == ["j1", "j2"]


def test_gui_loadback_pinned_and_fail_closed(tmp_path: Path, capsys: Any) -> None:
    """GUI load-back resolves exact pinned artifacts and never substitutes latest."""
    lib, pkg = tmp_path / "library", tmp_path / "package"
    pkg.mkdir()
    m, s, x, mp = _lib(lib)
    run = {"run_id": f"sha256:{_tag('r')}", "model_definition_hash": m.content_sha256}
    run.update(
        subject_profile_hash=s.content_sha256, match_artifact_hash=x.content_sha256
    )
    _manifest(pkg, run=run)
    req = rl.build_gui_loadback_request(pkg, lib)
    assert req.model_definition == m
    assert req.subject_profile == s and req.match_artifact == x
    assert not req.requires_refit
    assert req.artifacts["grid.mp4"] == hashlib.sha256(b"video-bytes").hexdigest()
    rc = rl.main(["loadback", "--package-dir", str(pkg), "--library-root", str(lib)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model_definition_hash"] == m.content_sha256

    mlc.save_artifact(lib, mlc.ModelDefinition.from_spec(_spec(), model_id="driver_v3"))
    orig = mp.read_text("utf-8")
    mp.write_text(json.dumps({**json.loads(orig), "model_id": "tampered"}), "utf-8")
    with pytest.raises(mlc.ArtifactIntegrityError):
        rl.build_gui_loadback_request(pkg, lib)

    mp.write_text(orig, "utf-8")
    mlc.ModelLibraryManifest.rebuild_from_directory(lib).save(lib)
    doc = json.loads((pkg / "manifest.json").read_text("utf-8"))
    doc["run"]["model_definition_hash"] = _tag("missing-model")
    (pkg / "manifest.json").write_text(json.dumps(doc), "utf-8")
    with pytest.raises(mlc.MissingArtifactError):
        rl.build_gui_loadback_request(pkg, lib)
