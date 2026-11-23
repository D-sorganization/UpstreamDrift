"""Tests for the synchronized-replay export manifest (#15494, Slice C).

The oracle for every hash is ``hashlib.sha256`` applied to bytes written by the
test itself, so the manifest is never checked against its own helper.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from src.shared.python.loop_nullspace.replay_bundle_manifest import (
    MANIFEST_SCHEMA,
    ArtifactRef,
    ArtifactRole,
    BundleCandidate,
    RenderSettings,
    ReplayBundleManifest,
    ReplayStatus,
    verify_against_dir,
)

pytestmark = pytest.mark.unit

_SHA_A = hashlib.sha256(b"a").hexdigest()


def _ref(role: ArtifactRole, name: str, payload: bytes = b"a") -> ArtifactRef:
    return ArtifactRef.from_bytes(role, name, payload)


def _render(**kw: object) -> RenderSettings:
    base = {
        "camera_id": "cam_face_on",
        "timebase_s": 0.001,
        "force_scale_m_per_n": 0.002,
        "torque_scale_m_per_nm": 0.01,
    }
    base.update(kw)
    return RenderSettings(**base)  # type: ignore[arg-type]


def _replayed(cid: str = "min_effort") -> BundleCandidate:
    return BundleCandidate(
        candidate_id=cid,
        strategy="minimum normalized effort",
        status=ReplayStatus.REPLAYED,
        max_residual=1e-9,
        artifacts=(
            _ref(ArtifactRole.VIDEO, f"{cid}/swing.mp4", b"v"),
            _ref(ArtifactRole.RESIDUALS, f"{cid}/res.csv", b"r"),
        ),
    )


def _manifest(*cands: BundleCandidate) -> ReplayBundleManifest:
    return ReplayBundleManifest(
        baseline=_ref(ArtifactRole.BASELINE, "baseline/swing.json", b"b"),
        render=_render(),
        assumptions=("model-conditioned allocation, not measured muscle torque",),
        candidates=cands or (_replayed(),),
    )


def test_artifact_hash_matches_independent_sha256() -> None:
    ref = ArtifactRef.from_bytes(ArtifactRole.VIDEO, "v.mp4", b"abc")
    assert ref.sha256 == hashlib.sha256(b"abc").hexdigest()
    assert ref.size_bytes == 3


@pytest.mark.parametrize("name", ["/abs.mp4", "../x.mp4", "a/../../x", "", "a\\b.mp4"])
def test_unsafe_artifact_names_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        ArtifactRef(ArtifactRole.VIDEO, name, _SHA_A, 1)


def test_bad_digest_rejected() -> None:
    with pytest.raises(ValueError, match="sha256"):
        ArtifactRef(ArtifactRole.VIDEO, "v.mp4", "xyz", 1)


@pytest.mark.parametrize(
    "kw",
    [
        {"timebase_s": 0.0},
        {"timebase_s": float("nan")},
        {"force_scale_m_per_n": -1.0},
        {"torque_scale_m_per_nm": float("inf")},
        {"camera_id": ""},
    ],
)
def test_invalid_render_settings_rejected(kw: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _render(**kw)


def test_replayed_requires_video_and_finite_residual() -> None:
    res = _ref(ArtifactRole.RESIDUALS, "c/r.csv")
    with pytest.raises(ValueError, match="video"):
        BundleCandidate(
            "c", "s", ReplayStatus.REPLAYED, max_residual=1e-9, artifacts=(res,)
        )
    vid = _ref(ArtifactRole.VIDEO, "c/v.mp4")
    for bad in (None, float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="residual"):
            BundleCandidate(
                "c", "s", ReplayStatus.REPLAYED, max_residual=bad, artifacts=(vid, res)
            )


def test_allocation_only_must_not_carry_video() -> None:
    vid = _ref(ArtifactRole.VIDEO, "c/v.mp4")
    res = _ref(ArtifactRole.RESIDUALS, "c/r.csv")
    with pytest.raises(ValueError, match="allocation_only"):
        BundleCandidate(
            "c",
            "s",
            ReplayStatus.ALLOCATION_ONLY,
            max_residual=1e-6,
            artifacts=(vid, res),
        )
    ok = BundleCandidate(
        "c", "s", ReplayStatus.ALLOCATION_ONLY, max_residual=1e-6, artifacts=(res,)
    )
    assert ok.status is ReplayStatus.ALLOCATION_ONLY
    with pytest.raises(ValueError, match="reason"):
        dataclasses.replace(ok, unavailable_reason="x")


def test_unavailable_requires_reason_and_no_artifacts() -> None:
    with pytest.raises(ValueError, match="reason"):
        BundleCandidate("c", "s", ReplayStatus.UNAVAILABLE)
    with pytest.raises(ValueError, match="unavailable"):
        BundleCandidate(
            "c",
            "s",
            ReplayStatus.UNAVAILABLE,
            unavailable_reason="infeasible load",
            artifacts=(_ref(ArtifactRole.RESIDUALS, "c/r.csv"),),
        )
    ok = BundleCandidate(
        "c", "s", ReplayStatus.UNAVAILABLE, unavailable_reason="infeasible load"
    )
    assert ok.unavailable_reason == "infeasible load"


def test_duplicate_candidate_ids_and_artifact_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate candidate"):
        _manifest(_replayed("a"), _replayed("a"))
    clash = BundleCandidate(
        "b",
        "s",
        ReplayStatus.REPLAYED,
        max_residual=0.0,
        artifacts=(
            _ref(ArtifactRole.VIDEO, "a/swing.mp4", b"x"),
            _ref(ArtifactRole.RESIDUALS, "b/r.csv"),
        ),
    )
    with pytest.raises(ValueError, match="duplicate artifact"):
        _manifest(_replayed("a"), clash)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"baseline": _ref(ArtifactRole.VIDEO, "b.mp4")}, "baseline"),
        ({"assumptions": ()}, "assumption"),
        ({"candidates": ()}, "candidate"),
    ],
)
def test_manifest_requires_baseline_role_assumptions_and_candidates(
    override: dict[str, object], match: str
) -> None:
    kw: dict[str, object] = {
        "baseline": _ref(ArtifactRole.BASELINE, "b.json"),
        "render": _render(),
        "assumptions": ("x",),
        "candidates": (_replayed(),),
    }
    kw.update(override)
    with pytest.raises(ValueError, match=match):
        ReplayBundleManifest(**kw)  # type: ignore[arg-type]


def test_json_is_canonical_and_round_trips() -> None:
    m = _manifest(_replayed("b"), _replayed("a"))
    text = m.to_json()
    data = json.loads(text)
    assert data["schema"] == MANIFEST_SCHEMA
    assert [c["candidate_id"] for c in data["candidates"]] == ["a", "b"]
    assert text == _manifest(_replayed("a"), _replayed("b")).to_json()
    assert ReplayBundleManifest.from_json(text) == _manifest(
        _replayed("a"), _replayed("b")
    )
    assert m.digest() == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_from_json_rejects_wrong_schema() -> None:
    data = json.loads(_manifest().to_json())
    data["schema"] = "other/v9"
    with pytest.raises(ValueError, match="schema"):
        ReplayBundleManifest.from_json(json.dumps(data))


def _write_bundle(root: Path, m: ReplayBundleManifest) -> None:
    key = {"video": b"v", "residuals": b"r", "baseline": b"b"}
    for ref in m.all_artifacts():
        target = root / ref.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(key[ref.role.value])


def test_verify_against_dir_clean_then_detects_tamper_and_missing(
    tmp_path: Path,
) -> None:
    m = _manifest(_replayed("a"), _replayed("b"))
    _write_bundle(tmp_path, m)
    assert verify_against_dir(m, tmp_path) == ()

    (tmp_path / "a" / "swing.mp4").write_bytes(b"tampered")
    (tmp_path / "b" / "res.csv").unlink()
    problems = verify_against_dir(m, tmp_path)
    assert any("a/swing.mp4" in p and "sha256" in p for p in problems)
    assert any("b/res.csv" in p and "missing" in p for p in problems)
    assert len(problems) == 2
