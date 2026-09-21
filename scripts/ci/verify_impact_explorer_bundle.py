"""Verify the served Impact Explorer bundle against its release artifacts.

Issue #9550 (epic #9546), acceptance item 4: a clean supported installation
must serve *real* JavaScript and assets at ``/impact-explorer-app/`` — not
merely an HTTP 200 or the fallback page — and the provider artifact revision
must match the pinned Tools source.

The vendored Tools build (``src/rate_of_closure/web/release/``) stamps a
release revision into ``index.html``, writes ``rate-of-closure-runtime.v1.json``
and a SHA-256 asset manifest ``rate-of-closure-assets.v1.json``. This script
serves a built ``dist`` exactly as ``src.api.local_server`` mounts it
(``StaticFiles(html=True)`` at ``/impact-explorer-app``; the equivalence is
pinned by ``tests/api/test_impact_explorer_mount.py``) and checks what an HTTP
client actually receives:

1. ``index.html`` carries the runtime descriptor for the expected revision and
   references its scripts under the mandated ``/impact-explorer-app/`` base.
2. The runtime descriptor and asset manifest are served and name that revision.
3. Every manifest asset is served with the manifest's byte count and SHA-256,
   and at least one of them is JavaScript.
4. Every asset ``index.html`` references is shipped in the manifest.
5. A missing artifact answers 404 — the mount never masks it with the index.

Usage (CI, after ``npm run build -- --base=/impact-explorer-app/`` and
``ROC_RELEASE_REVISION=<gitlink> node release/generateReleaseArtifacts.mjs``)::

    python3 scripts/ci/verify_impact_explorer_bundle.py \
        --dist vendor/ud-tools/src/rate_of_closure/web/dist \
        --expected-revision "$(git ls-tree HEAD -- vendor/ud-tools | awk '{print $3}')" \
        --receipt impact-explorer-bundle-receipt.json

Requires only ``fastapi`` and ``httpx`` (the test client), not the editable
UpstreamDrift install.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

MOUNT_PATH = "/impact-explorer-app"
RUNTIME_NAME = "rate-of-closure-runtime.v1.json"
ASSET_MANIFEST_NAME = "rate-of-closure-assets.v1.json"
RUNTIME_SCHEMA = "rate-of-closure/web-runtime/v1"
ASSET_MANIFEST_SCHEMA = "rate-of-closure/web-asset-manifest/v1"
RUNTIME_ELEMENT_ID = "rate-of-closure-web-runtime"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUNTIME_BLOCK_RE = re.compile(
    rf'<script id="{RUNTIME_ELEMENT_ID}" type="application/json">(.*?)</script>',
    re.DOTALL,
)
_ASSET_REF_RE = re.compile(r"""(?:src|href)=["']([^"']+)["']""")


class BundleVerificationError(RuntimeError):
    """The served bundle does not match the provider's release artifacts."""


def _require_commit(value: str, label: str) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise ValueError(f"{label} must be 40 lowercase hex chars, got {value!r}")
    return value


def _fetch(client: Any, rel: str) -> Any:
    response = client.get(f"{MOUNT_PATH}/{rel.lstrip('/')}")
    if response.status_code != 200:
        raise BundleVerificationError(
            f"{MOUNT_PATH}/{rel} answered {response.status_code}, expected 200"
        )
    return response


def _fetch_json(client: Any, rel: str, schema: str, revision: str) -> dict[str, Any]:
    try:
        payload = _fetch(client, rel).json()
    except ValueError as exc:
        raise BundleVerificationError(f"{rel} is not JSON: {exc}") from exc
    if payload.get("schema_version") != schema:
        raise BundleVerificationError(
            f"{rel} schema_version {payload.get('schema_version')!r} != {schema!r}"
        )
    if payload.get("release_revision") != revision:
        raise BundleVerificationError(
            f"{rel} release_revision {payload.get('release_revision')!r} "
            f"!= pinned {revision!r}"
        )
    return payload


def _check_index(html: str, revision: str) -> list[str]:
    """Return the mount-relative assets ``index.html`` references."""
    blocks = _RUNTIME_BLOCK_RE.findall(html)
    if len(blocks) != 1:
        raise BundleVerificationError(
            f"index.html must embed exactly one {RUNTIME_ELEMENT_ID} descriptor"
        )
    descriptor = json.loads(blocks[0])
    if descriptor.get("release_revision") != revision:
        raise BundleVerificationError(
            f"index.html release_revision {descriptor.get('release_revision')!r} "
            f"!= pinned {revision!r}"
        )
    refs = [
        ref
        for ref in _ASSET_REF_RE.findall(html)
        if not ref.startswith(("data:", "http://", "https://"))
    ]
    stray = [ref for ref in refs if not ref.startswith(f"{MOUNT_PATH}/")]
    if stray or not refs:
        raise BundleVerificationError(
            f"index.html must reference every asset under the {MOUNT_PATH}/ "
            f"base path (build with --base={MOUNT_PATH}/); offending: {stray}"
        )
    return refs


def _check_assets(client: Any, manifest: dict[str, Any]) -> tuple[int, int]:
    """Fetch each manifest asset through the mount; return (verified, js)."""
    javascript = 0
    assets = manifest.get("assets") or []
    for asset in assets:
        rel = asset["path"]
        response = _fetch(client, rel)
        digest = hashlib.sha256(response.content).hexdigest()
        if digest != asset["sha256"] or len(response.content) != asset["bytes"]:
            raise BundleVerificationError(
                f"{rel}: served sha256/bytes differ from the asset manifest"
            )
        if "javascript" in asset["media_type"]:
            if "javascript" not in response.headers.get("content-type", ""):
                raise BundleVerificationError(
                    f"{rel} served with content-type "
                    f"{response.headers.get('content-type')!r}, not JavaScript"
                )
            javascript += 1
    if javascript == 0:
        raise BundleVerificationError("bundle ships no JavaScript asset")
    return len(assets), javascript


def verify_served_bundle(client: Any, expected_revision: str) -> dict[str, Any]:
    """Verify what an HTTP ``client`` receives under ``/impact-explorer-app``.

    Preconditions: ``client`` answers ``GET`` with ``status_code``, ``headers``,
    ``content``, ``text`` and ``json()``; ``expected_revision`` is a full commit.
    Postconditions: the returned receipt names the mount, the revision, the
    number of manifest assets served with matching digests, how many of them
    are JavaScript, the index references and the missing-artifact status.
    Raises ``BundleVerificationError`` on any deviation.
    """
    _require_commit(expected_revision, "expected_revision")
    index = _fetch(client, "")
    if "text/html" not in index.headers.get("content-type", ""):
        raise BundleVerificationError("index is not served as text/html")
    references = _check_index(index.text, expected_revision)
    _fetch_json(client, RUNTIME_NAME, RUNTIME_SCHEMA, expected_revision)
    manifest = _fetch_json(
        client, ASSET_MANIFEST_NAME, ASSET_MANIFEST_SCHEMA, expected_revision
    )
    shipped = {f"{MOUNT_PATH}/{asset['path']}" for asset in manifest["assets"]}
    unshipped = sorted(set(references) - shipped)
    if unshipped:
        raise BundleVerificationError(
            f"index.html references assets absent from the manifest: {unshipped}"
        )
    verified, javascript = _check_assets(client, manifest)
    missing = client.get(
        f"{MOUNT_PATH}/assets/does-not-exist-{expected_revision[:8]}.js"
    )
    if missing.status_code != 404:
        raise BundleVerificationError(
            f"missing artifact answered {missing.status_code}, expected 404"
        )
    return {
        "mount": MOUNT_PATH,
        "release_revision": expected_revision,
        "assets_verified": verified,
        "javascript_assets": javascript,
        "total_bytes": manifest.get("total_bytes"),
        "index_references": references,
        "missing_artifact_status": missing.status_code,
    }


def serve_bundle(dist: Path) -> Any:
    """Return a test client serving ``dist`` the way ``local_server`` mounts it.

    Precondition: ``dist`` is an existing directory.
    """
    dist = Path(dist)
    if not dist.is_dir():
        raise ValueError(f"bundle directory does not exist: {dist}")
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.mount(
        MOUNT_PATH,
        StaticFiles(directory=str(dist), html=True),
        name="impact_explorer_app",
    )
    return TestClient(app)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 on PASS, 1 on FAIL."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dist", type=Path, required=True, help="Built bundle dir")
    parser.add_argument(
        "--expected-revision",
        required=True,
        help="Pinned Tools gitlink the bundle must be stamped with",
    )
    parser.add_argument(
        "--receipt", type=Path, default=None, help="Write the JSON receipt here"
    )
    args = parser.parse_args(argv)
    try:
        receipt = verify_served_bundle(serve_bundle(args.dist), args.expected_revision)
    except (BundleVerificationError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    if args.receipt is not None:
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {json.dumps(receipt)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
