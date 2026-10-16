"""Spine model asset admission: SHA-256 gate over owner-provided local files.

Uses a temporary fake asset tree and a test-only manifest; no network and no
real SimTK assets.  Fixture bytes are invented.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.spine.assets import (
    DEFAULT_MANIFEST_PATH,
    SpineAssetRequest,
    SpineAssetStatus,
    admit_spine_asset,
    resolve_spine_asset,
)

pytestmark = pytest.mark.unit

MODEL_BYTES = b"<OpenSimDocument>invented model bytes</OpenSimDocument>"
MESH_BYTES = b"invented mesh bytes"
ENV_VAR = "UD_TEST_SPINE_SOURCE"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(
    tmp_path: Path, model_sha: str | None, mesh_sha: str | None = None
) -> Path:
    entry = {
        "simtk_url": "https://example.invalid/project",
        "env_var": ENV_VAR,
        "model_sha256": model_sha,
        "resource_sha256": {"Geometry/vert.stl": mesh_sha or _sha(MESH_BYTES)},
        "licence_id": {"model": "test-model-licence", "geometry": "test-geo-licence"},
        "licence_text_sha256": None,
        "citation": {"text": "Invented et al.", "doi": "10.0000/test"},
        "owner_accepted_terms_on": None,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "assets": {"fake": entry}}))
    return path


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "private_owner_dir"
    (root / "Geometry").mkdir(parents=True)
    (root / "model.osim").write_bytes(MODEL_BYTES)
    (root / "Geometry" / "vert.stl").write_bytes(MESH_BYTES)
    return root / "model.osim"


def test_resolve_returns_none_when_env_unset(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, _sha(MODEL_BYTES))
    assert resolve_spine_asset("fake", {}, manifest_path=manifest) is None


def test_resolve_returns_request_from_env(tree: Path, tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, _sha(MODEL_BYTES))
    request = resolve_spine_asset("fake", {ENV_VAR: str(tree)}, manifest_path=manifest)
    assert request == SpineAssetRequest("fake", tree)


def test_resolve_unknown_asset_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown spine asset"):
        resolve_spine_asset("nope", {}, manifest_path=_manifest(tmp_path, None))


def test_missing_path_is_missing_never_success(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, _sha(MODEL_BYTES))
    receipt = admit_spine_asset(
        SpineAssetRequest("fake", tmp_path / "absent.osim"), manifest_path=manifest
    )
    assert receipt.status is SpineAssetStatus.MISSING
    assert receipt.model_sha256 is None


def test_null_pinned_hash_is_unpinned(tree: Path, tmp_path: Path) -> None:
    receipt = admit_spine_asset(
        SpineAssetRequest("fake", tree), manifest_path=_manifest(tmp_path, None)
    )
    assert receipt.status is SpineAssetStatus.UNPINNED
    assert receipt.model_sha256 == _sha(MODEL_BYTES)  # observed, not trusted


def test_wrong_model_hash_is_mismatch(tree: Path, tmp_path: Path) -> None:
    receipt = admit_spine_asset(
        SpineAssetRequest("fake", tree),
        manifest_path=_manifest(tmp_path, _sha(b"other")),
    )
    assert receipt.status is SpineAssetStatus.HASH_MISMATCH


def test_wrong_resource_hash_is_mismatch(tree: Path, tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, _sha(MODEL_BYTES), mesh_sha=_sha(b"other"))
    receipt = admit_spine_asset(SpineAssetRequest("fake", tree), manifest_path=manifest)
    assert receipt.status is SpineAssetStatus.HASH_MISMATCH


def test_missing_resource_is_missing(tree: Path, tmp_path: Path) -> None:
    (tree.parent / "Geometry" / "vert.stl").unlink()
    manifest = _manifest(tmp_path, _sha(MODEL_BYTES))
    receipt = admit_spine_asset(SpineAssetRequest("fake", tree), manifest_path=manifest)
    assert receipt.status is SpineAssetStatus.MISSING


def test_correct_hashes_qualify(tree: Path, tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, _sha(MODEL_BYTES))
    receipt = admit_spine_asset(SpineAssetRequest("fake", tree), manifest_path=manifest)
    assert receipt.status is SpineAssetStatus.QUALIFIED
    assert receipt.model_sha256 == _sha(MODEL_BYTES)
    assert receipt.licence_id == "test-model-licence"
    assert [r.relative_name for r in receipt.resources] == ["Geometry/vert.stl"]
    assert receipt.resources[0].sha256 == _sha(MESH_BYTES)


@pytest.mark.parametrize("pin", [None, "0" * 64, "model"])
def test_receipt_never_contains_absolute_paths(
    tree: Path, tmp_path: Path, pin: str | None
) -> None:
    sha = _sha(MODEL_BYTES) if pin == "model" else pin
    receipt = admit_spine_asset(
        SpineAssetRequest("fake", tree), manifest_path=_manifest(tmp_path, sha)
    )
    blob = json.dumps(receipt.as_dict())
    assert str(tree.parent) not in blob
    assert "private_owner_dir" not in blob
    assert "model.osim" in blob  # basename only


def test_request_validates_inputs() -> None:
    with pytest.raises(ValueError):
        SpineAssetRequest("", Path("x"))
    with pytest.raises(TypeError):
        SpineAssetRequest("fake", "not-a-path")  # type: ignore[arg-type]


def test_committed_manifest_schema() -> None:
    data = json.loads(DEFAULT_MANIFEST_PATH.read_text())
    assert set(data["assets"]) == {
        "bruno2015_fullbody_os4_v2",
        "senteler2016_lumbspine_3dof",
        "senteler2016_lumbspine_30dof",
    }
    for asset_id, entry in data["assets"].items():
        assert entry["model_sha256"] is None, asset_id  # pinned in SPN-6b
        assert entry["env_var"].startswith("UD_SPINE_")
        assert entry["simtk_url"].startswith("https://simtk.org/")
        assert entry["citation"]["text"] and entry["citation"]["doi"]
        licences = entry["licence_id"]
        assert licences["model"] and licences["geometry"]
        assert "licence_text_sha256" in entry
        assert "owner_accepted_terms_on" in entry
        assert isinstance(entry["resource_sha256"], dict)


def test_committed_manifest_resolves_without_env() -> None:
    for asset_id in ("bruno2015_fullbody_os4_v2", "senteler2016_lumbspine_3dof"):
        assert resolve_spine_asset(asset_id, {}) is None
