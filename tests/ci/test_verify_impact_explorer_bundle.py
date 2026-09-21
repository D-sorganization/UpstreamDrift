"""Tests for ``scripts/ci/verify_impact_explorer_bundle.py`` (issue #9550).

The verifier must prove that a built Impact Explorer bundle, served through
the ``/impact-explorer-app`` mount, delivers the provider's release
artifacts for the pinned Tools revision: real JavaScript whose SHA-256 matches
the asset manifest, a runtime descriptor stamped with that revision, and a
404 (never the index fallback) for a missing artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.ci import verify_impact_explorer_bundle as mod

pytestmark = pytest.mark.unit

REVISION = "62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1"
JS_BODY = b"console.log('rate of closure');\n"
CSS_BODY = b"body{background:#020617}\n"


def _descriptor(revision: str) -> dict[str, str]:
    return {
        "schema_version": mod.RUNTIME_SCHEMA,
        "mode": "static_inspection",
        "release_revision": revision,
    }


def _index(
    revision: str, script_src: str = "/impact-explorer-app/assets/index-abc.js"
) -> str:
    return (
        "<!doctype html><html><head>"
        '<link rel="stylesheet" href="/impact-explorer-app/assets/index-abc.css">'
        "</head><body>"
        f'<script id="{mod.RUNTIME_ELEMENT_ID}" type="application/json">'
        f"{json.dumps(_descriptor(revision), separators=(',', ':'))}</script>"
        f'<script type="module" src="{script_src}"></script>'
        "</body></html>"
    )


def _write_dist(
    root: Path,
    *,
    revision: str = REVISION,
    index_revision: str | None = None,
    manifest_extra: dict[str, object] | None = None,
) -> Path:
    """Write a release-shaped bundle the way Tools' generateReleaseArtifacts does."""
    dist = root / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "assets" / "index-abc.js").write_bytes(JS_BODY)
    (dist / "assets" / "index-abc.css").write_bytes(CSS_BODY)
    index = _index(index_revision or revision)
    (dist / "index.html").write_text(index, encoding="utf-8")
    (dist / mod.RUNTIME_NAME).write_text(
        json.dumps(_descriptor(revision), indent=2) + "\n", encoding="utf-8"
    )
    assets = []
    for rel, media in (
        ("assets/index-abc.css", "text/css; charset=utf-8"),
        ("assets/index-abc.js", "text/javascript; charset=utf-8"),
        ("index.html", "text/html; charset=utf-8"),
        (mod.RUNTIME_NAME, "application/json; charset=utf-8"),
    ):
        body = (dist / rel).read_bytes()
        assets.append(
            {
                "path": rel,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "media_type": media,
                "executable": False,
            }
        )
    manifest: dict[str, object] = {
        "schema_version": mod.ASSET_MANIFEST_SCHEMA,
        "release_revision": revision,
        "total_bytes": sum(int(a["bytes"]) for a in assets),
        "assets": assets,
    }
    manifest.update(manifest_extra or {})
    (dist / mod.ASSET_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return dist


def test_release_bundle_served_through_the_mount_passes(tmp_path: Path) -> None:
    dist = _write_dist(tmp_path)
    receipt = mod.verify_served_bundle(mod.serve_bundle(dist), REVISION)
    assert receipt["mount"] == mod.MOUNT_PATH
    assert receipt["release_revision"] == REVISION
    assert receipt["javascript_assets"] == 1
    assert receipt["assets_verified"] == 4
    assert receipt["missing_artifact_status"] == 404
    assert "/impact-explorer-app/assets/index-abc.js" in receipt["index_references"]


def test_index_stamped_with_another_revision_fails(tmp_path: Path) -> None:
    dist = _write_dist(tmp_path, index_revision="0" * 40)
    with pytest.raises(mod.BundleVerificationError, match="release_revision"):
        mod.verify_served_bundle(mod.serve_bundle(dist), REVISION)


def test_development_build_without_release_stamp_fails(tmp_path: Path) -> None:
    """A plain `npm run build` (no release artifacts) is not install evidence."""
    dist = _write_dist(tmp_path)
    (dist / mod.RUNTIME_NAME).unlink()
    with pytest.raises(mod.BundleVerificationError, match=mod.RUNTIME_NAME):
        mod.verify_served_bundle(mod.serve_bundle(dist), REVISION)


def test_tampered_asset_fails_manifest_digest(tmp_path: Path) -> None:
    dist = _write_dist(tmp_path)
    (dist / "assets" / "index-abc.js").write_bytes(b"alert('not the build')\n")
    with pytest.raises(mod.BundleVerificationError, match="sha256"):
        mod.verify_served_bundle(mod.serve_bundle(dist), REVISION)


def test_index_referencing_unshipped_asset_fails(tmp_path: Path) -> None:
    dist = _write_dist(tmp_path)
    (dist / "index.html").write_text(
        _index(REVISION, script_src="/impact-explorer-app/assets/ghost.js"),
        encoding="utf-8",
    )
    with pytest.raises(mod.BundleVerificationError, match="ghost.js"):
        mod.verify_served_bundle(mod.serve_bundle(dist), REVISION)


def test_index_built_without_base_path_fails(tmp_path: Path) -> None:
    """Assets requested from host paths would 404 behind the mount."""
    dist = _write_dist(tmp_path)
    (dist / "index.html").write_text(
        _index(REVISION, script_src="./assets/index-abc.js"), encoding="utf-8"
    )
    with pytest.raises(mod.BundleVerificationError, match="base path"):
        mod.verify_served_bundle(mod.serve_bundle(dist), REVISION)


def test_expected_revision_must_be_a_full_commit(tmp_path: Path) -> None:
    dist = _write_dist(tmp_path)
    with pytest.raises(ValueError, match="40 lowercase hex"):
        mod.verify_served_bundle(mod.serve_bundle(dist), "62e8cdb")


def test_missing_dist_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        mod.serve_bundle(tmp_path / "nope")


def test_main_writes_receipt_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dist = _write_dist(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    code = mod.main(
        [
            "--dist",
            str(dist),
            "--expected-revision",
            REVISION,
            "--receipt",
            str(receipt_path),
        ]
    )
    assert code == 0
    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert written["release_revision"] == REVISION
    assert "PASS" in capsys.readouterr().out


def test_main_reports_failure_and_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dist = _write_dist(tmp_path, index_revision="0" * 40)
    code = mod.main(["--dist", str(dist), "--expected-revision", REVISION])
    assert code == 1
    assert "FAIL" in capsys.readouterr().out
