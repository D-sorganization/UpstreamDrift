"""Unit tests for ModelDefinition, SubjectProfile, MatchArtifact, and resolver."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.model_appearance import canonical_golfer_gate as gate
from src.shared.python.motion_matching import model_library_contracts as mlc
from src.shared.python.motion_matching.execution import spec_revisions as sr

pytestmark = pytest.mark.unit
_FIX = Path(__file__).resolve().parents[3] / "tests/fixtures/club_face/spec_driver.json"
_BASE = json.loads(_FIX.read_text(encoding="utf-8"))


def _canonical_spec() -> dict[str, Any]:
    """Return a deep copy of a canonical full-body golfer spec."""
    s = copy.deepcopy(_BASE)
    s["subject"]["hand_visual_backend"] = "skinned_human"
    s["subject"]["body_visual_backend"] = "skinned_cc0"
    s["spec_revision"] = {"version": 1, "configuration": "skinned_hand_wrists"}
    return sr.with_current_shaft(s)


def _digest(label: str) -> str:
    """Return a deterministic 64-char SHA-256 hex digest for test fixtures."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sample() -> tuple[mlc.ModelDefinition, mlc.SubjectProfile, mlc.MatchArtifact]:
    """Build a valid linked triad of model, subject, and match artifacts."""
    m = mlc.ModelDefinition.from_spec(_canonical_spec(), model_id="driver_v2")
    s = mlc.SubjectProfile("p1", "right", 1.8, 78.0, _digest("c3d"))
    x = mlc.MatchArtifact("m1", m.content_sha256, s.content_sha256, _digest("traj"))
    return m, s, x


@pytest.mark.parametrize(
    ("sec", "key", "val", "part"),
    [
        ("subject", "hand_visual_backend", "capsule", "capsule"),
        ("subject", "body_visual_backend", "loft", "loft"),
        ("root", "spec_revision", {"configuration": "legacy"}, "spec_revision"),
        ("club", "shaft", {"model": "rigid"}, "shaft"),
    ],
)
def test_non_canonical_spec_rejected(sec: str, key: str, val: Any, part: str) -> None:
    """ModelDefinition.from_spec rejects deprecated/rigid specs via gate."""
    spec = _canonical_spec()
    (spec if sec == "root" else spec[sec])[key] = val
    with pytest.raises(gate.NonCanonicalGolferError) as exc:
        mlc.ModelDefinition.from_spec(spec, model_id="bad")
    assert exc.value.part == part


@pytest.mark.parametrize("bad", ["", "latest", "main", "head", "a.json", "d/a.json"])
def test_mutable_and_bare_refs_rejected_fail_closed(tmp_path: Path, bad: str) -> None:
    """Mutable refs and bare filenames without SHA-256 fail closed everywhere."""
    r = mlc.ModelLibraryResolver.from_directory(tmp_path, rebuild=True)
    for fn in (
        lambda: mlc.validate_immutable_ref(bad),
        lambda: mlc.SubjectProfile("s1", "right", 1.75, 75.0, bad),
        lambda: mlc.MatchArtifact("m1", bad, _digest("s"), _digest("t")),
        lambda: r.resolve(bad),
    ):
        with pytest.raises(mlc.MutableReferenceError):
            fn()


def test_sha256_tamper_detection_on_disk(tmp_path: Path) -> None:
    """Modifying any serialized artifact on disk raises ArtifactIntegrityError."""
    m, s, x = _sample()
    paths = [mlc.save_artifact(tmp_path, i) for i in (m, s, x)]
    mlc.ModelLibraryManifest.rebuild_from_directory(tmp_path).save(tmp_path)
    r = mlc.ModelLibraryResolver.from_directory(tmp_path)
    edits = (("model_id", "x"), ("mass_kg", 99.0), ("match_id", "x"))
    for idx, (k, v) in enumerate(edits):
        p, obj = paths[idx], (m, s, x)[idx]
        orig = p.read_text(encoding="utf-8")
        p.write_text(json.dumps({**json.loads(orig), k: v}), encoding="utf-8")
        with pytest.raises(mlc.ArtifactIntegrityError):
            r.resolve(obj.content_sha256)
        p.write_text(orig, encoding="utf-8")


def test_rebuildable_manifest_roundtrip_and_cross_references(tmp_path: Path) -> None:
    """Manifest rebuilds deterministically, verifies cross-refs, fails dangling."""
    m, s, x = _sample()
    for item in (m, s, x):
        mlc.save_artifact(tmp_path, item)
    m_a = mlc.ModelLibraryManifest.rebuild_from_directory(tmp_path)
    m_b = mlc.ModelLibraryManifest.rebuild_from_directory(tmp_path)
    assert m_a.schema_version == mlc.SCHEMA_VERSION
    assert m_a.manifest_sha256 == m_b.manifest_sha256
    assert list(m_a.entries) == sorted(m_a.entries)
    m_a.save(tmp_path)
    loaded = mlc.ModelLibraryManifest.load(tmp_path)
    assert loaded.manifest_sha256 == m_a.manifest_sha256
    r = mlc.ModelLibraryResolver.from_directory(tmp_path)
    for obj in (m, s, x):
        assert r.resolve(obj.content_sha256) == obj
    bad = mlc.MatchArtifact("o", m.content_sha256, _digest("x"), _digest("t"))
    mlc.save_artifact(tmp_path, bad)
    with pytest.raises(mlc.MissingArtifactError):
        mlc.ModelLibraryManifest.rebuild_from_directory(tmp_path)
