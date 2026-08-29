"""Build and verify exact-revision companion publication bundles.

This module owns delivery mechanics only. ``companion_catalog`` remains the
single software-fact generator; this layer validates CI authority, packages
its canonical bytes with schemas and digests, and records acquisition evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import jsonschema

from scripts import companion_catalog

REPOSITORY = "D-sorganization/UpstreamDrift"
REPOSITORY_URL = f"https://github.com/{REPOSITORY}"
API_REPOSITORY_URL = f"https://api.github.com/repos/{REPOSITORY}"
ACQUISITION_SCHEMA_ID = (
    "https://upstreamdrift.dev/schemas/"
    "upstreamdrift-companion-acquisition-v1.schema.json"
)
ACQUISITION_CONTRACT_VERSION = "1.0.0"
DEFAULT_OUTPUT_DIR = Path("dist/companion")
MANIFEST_NAME = "upstreamdrift-companion.v1.json"
MANIFEST_SCHEMA_NAME = "upstreamdrift-companion-v1.schema.json"
ACQUISITION_SCHEMA_NAME = "upstreamdrift-companion-acquisition-v1.schema.json"
COMPATIBILITY_POLICY_NAME = "upstreamdrift-companion-compatibility-v1.json"
MANIFEST_SCHEMA_PATH = Path("docs/api/contracts") / MANIFEST_SCHEMA_NAME
ACQUISITION_SCHEMA_PATH = Path("docs/api/contracts") / ACQUISITION_SCHEMA_NAME
COMPATIBILITY_POLICY_PATH = Path("docs/api/contracts") / COMPATIBILITY_POLICY_NAME
_BASE_ASSET_NAMES = (
    MANIFEST_NAME,
    MANIFEST_SCHEMA_NAME,
    ACQUISITION_SCHEMA_NAME,
    COMPATIBILITY_POLICY_NAME,
)
PAYLOAD_ASSET_NAMES = tuple(
    name for base in _BASE_ASSET_NAMES for name in (base, f"{base}.sha256")
)
_RELEASE_TAG = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class PublicationContractError(RuntimeError):
    """Raised when publication or acquisition authority cannot be proven."""


@dataclass(frozen=True)
class AuthorityContext:
    """Validated GitHub Actions identity for an exact source commit."""

    authority: str
    source_commit: str
    repository: str
    event_name: str
    ref: str
    ref_name: str
    run_id: int
    run_attempt: int
    run_url: str


def canonical_json(value: Mapping[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON with one trailing newline."""
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PublicationContractError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def git_head_commit(repo_root: Path) -> str:
    """Return exact lowercase HEAD or fail closed."""
    commit = _run_git(repo_root.resolve(), "rev-parse", "HEAD").lower()
    if not _HEX_40.fullmatch(commit):
        raise PublicationContractError(f"HEAD is not an exact commit: {commit!r}")
    return commit


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise PublicationContractError(f"required CI variable is missing: {name}")
    return value


def _positive_int(env: Mapping[str, str], name: str) -> int:
    raw = _required(env, name)
    try:
        value = int(raw)
    except ValueError as exc:
        raise PublicationContractError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise PublicationContractError(f"{name} must be a positive integer")
    return value


def validate_ci_authority(
    authority: str, *, env: Mapping[str, str], head_commit: str
) -> AuthorityContext:
    """Validate an official protected-main or release-tag push context.

    Preconditions:
        ``head_commit`` is the checked-out exact commit. The environment is a
        GitHub Actions push in the canonical repository.
    Postconditions:
        PR refs, forks, mutable aliases, and CI/HEAD mismatches are refused.
    """
    if authority not in {"protected-main", "tag"}:
        raise ValueError("authority must be 'protected-main' or 'tag'")
    if env.get("GITHUB_ACTIONS") != "true":
        raise PublicationContractError("publication requires GitHub Actions authority")
    event_name = _required(env, "GITHUB_EVENT_NAME")
    if event_name != "push":
        raise PublicationContractError("publication requires an exact push event")
    repository = _required(env, "GITHUB_REPOSITORY")
    if repository != REPOSITORY:
        raise PublicationContractError(
            f"publication repository must be {REPOSITORY}, got {repository!r}"
        )
    if not _HEX_40.fullmatch(head_commit):
        raise PublicationContractError("checked-out HEAD is not an exact commit")
    github_sha = _required(env, "GITHUB_SHA").lower()
    if github_sha != head_commit:
        raise PublicationContractError(
            f"GITHUB_SHA {github_sha!r} does not match checked-out HEAD {head_commit}"
        )
    ref = _required(env, "GITHUB_REF")
    ref_name = _required(env, "GITHUB_REF_NAME")
    if authority == "protected-main":
        if ref != "refs/heads/main" or ref_name != "main":
            raise PublicationContractError(
                "protected main authority requires refs/heads/main"
            )
    elif ref != f"refs/tags/{ref_name}" or not _RELEASE_TAG.fullmatch(ref_name):
        raise PublicationContractError(
            "tag authority requires an exact vX.Y.Z release tag"
        )
    server_url = _required(env, "GITHUB_SERVER_URL").rstrip("/")
    if server_url != "https://github.com":
        raise PublicationContractError("GITHUB_SERVER_URL must be https://github.com")
    run_id = _positive_int(env, "GITHUB_RUN_ID")
    run_attempt = _positive_int(env, "GITHUB_RUN_ATTEMPT")
    return AuthorityContext(
        authority=authority,
        source_commit=head_commit,
        repository=repository,
        event_name=event_name,
        ref=ref,
        ref_name=ref_name,
        run_id=run_id,
        run_attempt=run_attempt,
        run_url=f"{server_url}/{repository}/actions/runs/{run_id}",
    )


def load_compatibility_policy(repo_root: Path) -> dict[str, Any]:
    """Load the tracked compatibility policy document."""
    path = repo_root.resolve() / COMPATIBILITY_POLICY_PATH
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationContractError(f"invalid compatibility policy: {exc}") from exc
    if not isinstance(value, dict):
        raise PublicationContractError("compatibility policy must be a JSON object")
    return value


def _fixture(repo_root: Path, value: object) -> dict[str, Any]:
    if not isinstance(value, str):
        raise PublicationContractError("compatibility fixture path must be a string")
    try:
        relative = companion_catalog.validate_repo_relative(Path(value))
        parsed = json.loads(
            (repo_root.resolve() / Path(*relative.parts)).read_text(encoding="utf-8")
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise PublicationContractError(
            f"invalid compatibility fixture {value!r}: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise PublicationContractError(
            f"compatibility fixture {value!r} must be an object"
        )
    return parsed


def validate_compatibility_policy(repo_root: Path, policy: Mapping[str, Any]) -> None:
    """Validate supported-version fixtures and fail-closed rejection cases."""
    expected_keys = {
        "contract_version",
        "current",
        "previous_supported",
        "fixtures",
        "rejected_fixtures",
        "policy",
    }
    if set(policy) != expected_keys:
        raise PublicationContractError(
            "compatibility policy has unknown or missing keys"
        )
    if policy["contract_version"] != ACQUISITION_CONTRACT_VERSION:
        raise PublicationContractError("compatibility policy contract version is stale")
    current = policy["current"]
    if current != companion_catalog.SCHEMA_VERSION or not isinstance(current, str):
        raise PublicationContractError("compatibility current version is stale")
    previous = policy["previous_supported"]
    fixtures = policy["fixtures"]
    rejected = policy["rejected_fixtures"]
    if not isinstance(previous, list) or not isinstance(fixtures, dict):
        raise PublicationContractError(
            "compatibility versions and fixtures are malformed"
        )
    if not isinstance(rejected, list) or len(rejected) < 2:
        raise PublicationContractError(
            "future and incompatible rejection fixtures are required"
        )
    if any(
        not isinstance(version, str) or not _SEMVER.fullmatch(version)
        for version in previous
    ):
        raise PublicationContractError(
            "previous supported versions must be semantic versions"
        )
    if len(set(previous)) != len(previous) or current in previous:
        raise PublicationContractError("supported schema versions must be unique")
    for version in [current, *previous]:
        if version not in fixtures:
            qualifier = "previous supported" if version in previous else "current"
            raise PublicationContractError(f"missing {qualifier} fixture for {version}")
    schema = json.loads((repo_root.resolve() / MANIFEST_SCHEMA_PATH).read_text("utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for version in [current, *previous]:
        fixture = _fixture(repo_root, fixtures[version])
        if fixture.get("schema_version") != version:
            raise PublicationContractError(f"fixture version does not match {version}")
        errors = sorted(
            validator.iter_errors(fixture), key=lambda error: list(error.path)
        )
        if errors:
            raise PublicationContractError(
                f"supported fixture {version} fails its schema: {errors[0].message}"
            )
    for path in rejected:
        fixture = _fixture(repo_root, path)
        if not list(validator.iter_errors(fixture)):
            raise PublicationContractError(
                f"rejected fixture unexpectedly validates: {path}"
            )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_payload(output_dir: Path, name: str, payload: bytes) -> None:
    destination = output_dir / name
    destination.write_bytes(payload)
    (output_dir / f"{name}.sha256").write_text(
        f"{_sha256(payload)}  {name}\n", encoding="ascii", newline="\n"
    )


def _schema(repo_root: Path, relative: Path) -> dict[str, Any]:
    try:
        schema = json.loads((repo_root / relative).read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
    except (OSError, json.JSONDecodeError, jsonschema.SchemaError) as exc:
        raise PublicationContractError(f"invalid schema {relative}: {exc}") from exc
    return schema


def build_bundle(
    repo_root: Path,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    authority: str,
    env: Mapping[str, str] | None = None,
    require_clean: bool = True,
) -> dict[str, dict[str, int | str]]:
    """Build and verify the canonical publication payload set.

    The authority mode affects only preconditions; the bytes for a given commit
    are identical in protected-main and tag workflows.
    """
    root = repo_root.resolve()
    environment = os.environ if env is None else env
    commit = git_head_commit(root)
    context = validate_ci_authority(authority, env=environment, head_commit=commit)
    if authority == "tag":
        tag_commit = _run_git(root, "rev-parse", f"{context.ref}^{{commit}}").lower()
        if tag_commit != commit:
            raise PublicationContractError(
                f"release tag resolves to {tag_commit}, not checked-out HEAD {commit}"
            )
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    destination.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name
        for path in destination.iterdir()
        if path.name not in PAYLOAD_ASSET_NAMES
    )
    if unexpected:
        raise PublicationContractError(
            "publication output directory contains stale or unexpected files: "
            + ", ".join(unexpected)
        )
    manifest_path = destination / MANIFEST_NAME
    companion_catalog.write_catalog(
        root, output=manifest_path, require_clean=require_clean
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_schema = _schema(root, MANIFEST_SCHEMA_PATH)
    jsonschema.Draft202012Validator(manifest_schema).validate(manifest)
    if manifest["source"]["commit"] != commit:
        raise PublicationContractError("manifest embedded commit does not match HEAD")
    policy = load_compatibility_policy(root)
    validate_compatibility_policy(root, policy)
    acquisition_schema = _schema(root, ACQUISITION_SCHEMA_PATH)
    _write_payload(
        destination, MANIFEST_SCHEMA_NAME, (root / MANIFEST_SCHEMA_PATH).read_bytes()
    )
    _write_payload(
        destination,
        ACQUISITION_SCHEMA_NAME,
        (root / ACQUISITION_SCHEMA_PATH).read_bytes(),
    )
    _write_payload(
        destination,
        COMPATIBILITY_POLICY_NAME,
        (root / COMPATIBILITY_POLICY_PATH).read_bytes(),
    )
    # Check the acquisition schema again through the exact packaged bytes.
    jsonschema.Draft202012Validator.check_schema(acquisition_schema)
    return verify_bundle(destination)


def _parse_sidecar(path: Path, expected_name: str) -> str:
    try:
        line = path.read_text(encoding="ascii")
    except OSError as exc:
        raise PublicationContractError(f"missing digest sidecar: {path.name}") from exc
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\n", line)
    if match is None or match.group(2) != expected_name:
        raise PublicationContractError(f"malformed digest sidecar: {path.name}")
    return match.group(1)


def verify_bundle(bundle_dir: Path) -> dict[str, dict[str, int | str]]:
    """Verify exact filenames, detached hashes, schema, policy, and provenance."""
    root = bundle_dir.resolve()
    actual = {path.name for path in root.iterdir()} if root.is_dir() else set()
    expected = set(PAYLOAD_ASSET_NAMES)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise PublicationContractError(
            f"bundle file set mismatch; missing={missing}, unexpected={unexpected}"
        )
    inventory: dict[str, dict[str, int | str]] = {}
    for name in _BASE_ASSET_NAMES:
        payload = (root / name).read_bytes()
        digest = _sha256(payload)
        if _parse_sidecar(root / f"{name}.sha256", name) != digest:
            raise PublicationContractError(f"digest mismatch for {name}")
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest_schema = json.loads(
        (root / MANIFEST_SCHEMA_NAME).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(manifest_schema).validate(manifest)
    jsonschema.Draft202012Validator.check_schema(
        json.loads((root / ACQUISITION_SCHEMA_NAME).read_text(encoding="utf-8"))
    )
    policy = json.loads((root / COMPATIBILITY_POLICY_NAME).read_text(encoding="utf-8"))
    if policy.get("current") != manifest.get("schema_version"):
        raise PublicationContractError("bundle compatibility policy is stale")
    commit = manifest.get("source", {}).get("commit")
    if not isinstance(commit, str) or not _HEX_40.fullmatch(commit):
        raise PublicationContractError("manifest embedded commit is not exact")
    for name in PAYLOAD_ASSET_NAMES:
        payload = (root / name).read_bytes()
        inventory[name] = {"size": len(payload), "sha256": _sha256(payload)}
    return inventory


def _workflow_payload(context: AuthorityContext) -> dict[str, Any]:
    return {
        "run_id": context.run_id,
        "run_attempt": context.run_attempt,
        "run_url": context.run_url,
        "event_name": context.event_name,
        "ref": context.ref,
    }


def _attestation_payload(attestation_id: str, attestation_url: str) -> dict[str, Any]:
    if not attestation_id:
        raise PublicationContractError("attestation ID is required")
    expected = f"{REPOSITORY_URL}/attestations/{attestation_id}"
    if attestation_url != expected or urlsplit(attestation_url).query:
        raise PublicationContractError(
            "attestation URL is not an exact GitHub identity"
        )
    return {
        "id": attestation_id,
        "url": attestation_url,
        "verification_command": [
            "gh",
            "attestation",
            "verify",
            MANIFEST_NAME,
            "--repo",
            REPOSITORY,
        ],
    }


def _payload_inventory(bundle_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory = verify_bundle(bundle_dir)
    manifest = json.loads((bundle_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    commit = manifest["source"]["commit"]
    payloads = [
        {
            "name": name,
            "size": inventory[name]["size"],
            "sha256": inventory[name]["sha256"],
            "embedded_commit": commit if name == MANIFEST_NAME else None,
        }
        for name in PAYLOAD_ASSET_NAMES
    ]
    return payloads, manifest


def _base_record(
    *,
    context: AuthorityContext,
    manifest: Mapping[str, Any],
    payloads: list[dict[str, Any]],
    channel: str,
    delivery: Mapping[str, Any],
    attestation: Mapping[str, Any],
    limitations: list[str],
) -> dict[str, Any]:
    record = {
        "$schema": ACQUISITION_SCHEMA_ID,
        "contract_version": ACQUISITION_CONTRACT_VERSION,
        "repository": REPOSITORY,
        "source_commit": context.source_commit,
        "schema_version": manifest["schema_version"],
        "generator_version": manifest["source"]["generator"]["version"],
        "channel": channel,
        "workflow": _workflow_payload(context),
        "delivery": dict(delivery),
        "attestation": dict(attestation),
        "payloads": payloads,
        "limitations": limitations,
    }
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / ACQUISITION_SCHEMA_PATH).read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(record)
    return record


def _archive_digest(value: str) -> str:
    digest = value.removeprefix("sha256:")
    if not _HEX_64.fullmatch(digest):
        raise PublicationContractError("Actions artifact digest must be SHA-256")
    return digest


def build_actions_acquisition(
    bundle_dir: Path,
    *,
    env: Mapping[str, str],
    artifact_metadata: Mapping[str, Any],
    attestation_id: str,
    attestation_url: str,
) -> dict[str, Any]:
    """Build evidence for a non-durable exact protected-main Actions artifact."""
    payloads, manifest = _payload_inventory(bundle_dir)
    commit = manifest["source"]["commit"]
    context = validate_ci_authority("protected-main", env=env, head_commit=commit)
    artifact_name = artifact_metadata.get("name")
    artifact_id = artifact_metadata.get("id")
    artifact_url = artifact_metadata.get("url")
    artifact_digest = artifact_metadata.get("digest")
    retention_days = artifact_metadata.get("retention_days")
    if not isinstance(artifact_name, str) or not isinstance(artifact_url, str):
        raise PublicationContractError("artifact name and URL must be strings")
    if not isinstance(artifact_digest, str):
        raise PublicationContractError("artifact digest must be a string")
    if not isinstance(artifact_id, int) or not isinstance(retention_days, int):
        raise PublicationContractError("artifact ID and retention must be integers")
    if artifact_id < 1 or not 1 <= retention_days <= 90:
        raise PublicationContractError("artifact ID and retention must be positive")
    expected_name = f"upstreamdrift-companion-{commit}"
    if artifact_name != expected_name:
        raise PublicationContractError("Actions artifact name is not commit-pinned")
    expected_url = f"{context.run_url}/artifacts/{artifact_id}"
    if artifact_url != expected_url or urlsplit(artifact_url).query:
        raise PublicationContractError(
            "Actions artifact URL is not the exact run identity"
        )
    delivery = {
        "durability": "ephemeral",
        "actions_artifact": {
            "name": artifact_name,
            "artifact_id": artifact_id,
            "url": artifact_url,
            "archive_sha256": _archive_digest(artifact_digest),
            "retention_days": retention_days,
        },
        "release": None,
    }
    return _base_record(
        context=context,
        manifest=manifest,
        payloads=payloads,
        channel="actions",
        delivery=delivery,
        attestation=_attestation_payload(attestation_id, attestation_url),
        limitations=[
            "Actions artifacts expire; no durable release acquisition URL exists yet."
        ],
    )


def build_release_acquisition(
    bundle_dir: Path,
    *,
    env: Mapping[str, str],
    release_metadata: Mapping[str, Any],
    attestation_id: str,
    attestation_url: str,
) -> dict[str, Any]:
    """Build a durable record from an exact draft release and its asset IDs."""
    payloads, manifest = _payload_inventory(bundle_dir)
    context = validate_ci_authority(
        "tag", env=env, head_commit=manifest["source"]["commit"]
    )
    release_id = release_metadata.get("id")
    if not isinstance(release_id, int) or release_id < 1:
        raise PublicationContractError("release ID must be a positive integer")
    if release_metadata.get("tag_name") != context.ref_name:
        raise PublicationContractError("release tag does not match workflow tag")
    if release_metadata.get("draft") is not True:
        raise PublicationContractError(
            "acquisition must be recorded before draft promotion"
        )
    api_url = f"{API_REPOSITORY_URL}/releases/{release_id}"
    html_url = f"{REPOSITORY_URL}/releases/tag/{context.ref_name}"
    if release_metadata.get("url") != api_url:
        raise PublicationContractError(
            "release URL is not an exact numeric API identity"
        )
    raw_assets = release_metadata.get("assets")
    if not isinstance(raw_assets, list):
        raise PublicationContractError("release assets metadata must be an array")
    expected_inventory = verify_bundle(bundle_dir)
    release_assets: list[dict[str, Any]] = []
    for name in PAYLOAD_ASSET_NAMES:
        matches = [asset for asset in raw_assets if asset.get("name") == name]
        if len(matches) != 1:
            raise PublicationContractError(
                f"release asset missing or duplicated: {name}"
            )
        asset = matches[0]
        asset_id = asset.get("id")
        if not isinstance(asset_id, int) or asset_id < 1:
            raise PublicationContractError(f"release asset ID is invalid: {name}")
        expected_api = f"{API_REPOSITORY_URL}/releases/assets/{asset_id}"
        expected_display = (
            f"{REPOSITORY_URL}/releases/download/{context.ref_name}/{name}"
        )
        if asset.get("url") != expected_api or urlsplit(str(asset.get("url"))).query:
            raise PublicationContractError(f"release asset API URL is mutable: {name}")
        if asset.get("size") != expected_inventory[name]["size"]:
            raise PublicationContractError(f"release asset size mismatch: {name}")
        release_assets.append(
            {
                "name": name,
                "asset_id": asset_id,
                "api_url": expected_api,
                "display_url": expected_display,
                "size": expected_inventory[name]["size"],
                "sha256": expected_inventory[name]["sha256"],
            }
        )
    delivery = {
        "durability": "immutable-release",
        "actions_artifact": None,
        "release": {
            "release_id": release_id,
            "tag": context.ref_name,
            "api_url": api_url,
            "html_url": html_url,
            "assets": release_assets,
        },
    }
    return _base_record(
        context=context,
        manifest=manifest,
        payloads=payloads,
        channel="release",
        delivery=delivery,
        attestation=_attestation_payload(attestation_id, attestation_url),
        limitations=[],
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Expose redirects to the contract validator instead of following them."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _request_no_redirect(
    url: str, *, token: str | None
) -> tuple[int, Mapping[str, str], bytes]:
    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": "UpstreamDrift-companion-publication",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=60) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308}:
            return exc.code, dict(exc.headers.items()), b""
        raise PublicationContractError(
            f"release asset request failed with HTTP {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise PublicationContractError(f"release asset request failed: {exc}") from exc


def validate_release_redirect(source_url: str, target_url: str) -> None:
    """Allow one credential-free redirect only to GitHub release storage."""
    source = urlsplit(source_url)
    target = urlsplit(target_url)
    if (
        source.scheme != "https"
        or source.netloc != "api.github.com"
        or not source.path.startswith(f"/repos/{REPOSITORY}/releases/assets/")
    ):
        raise PublicationContractError("release acquisition source is not an asset API")
    hostname = (target.hostname or "").lower()
    allowed = hostname in {
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    } or hostname.endswith(".s3.amazonaws.com")
    if target.scheme != "https" or not allowed or target.fragment:
        raise PublicationContractError("unexpected release asset redirect")


def _download_exact_asset(api_url: str, *, token: str) -> bytes:
    status, headers, payload = _request_no_redirect(api_url, token=token)
    if status == 200:
        return payload
    if status not in {301, 302, 303, 307, 308}:
        raise PublicationContractError(f"unexpected release asset HTTP status {status}")
    location = headers.get("Location") or headers.get("location")
    if not location:
        raise PublicationContractError("release asset redirect has no Location")
    validate_release_redirect(api_url, location)
    final_status, final_headers, final_payload = _request_no_redirect(
        location, token=None
    )
    if final_status != 200:
        detail = final_headers.get("Location") or final_headers.get("location") or ""
        raise PublicationContractError(
            f"unexpected second release redirect/status {final_status}: {detail}"
        )
    return final_payload


def download_release_payloads(
    acquisition: Mapping[str, Any], *, output_dir: Path, token: str
) -> dict[str, dict[str, int | str]]:
    """Acquire exact numeric release assets and verify the reconstructed bundle."""
    if not token:
        raise PublicationContractError("GH_TOKEN is required for release acquisition")
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / ACQUISITION_SCHEMA_PATH).read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(acquisition)
    if acquisition["channel"] != "release":
        raise PublicationContractError(
            "only immutable release records are downloadable"
        )
    release = acquisition["delivery"]["release"]
    assert release is not None
    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise PublicationContractError("release download directory must be empty")
    for asset in release["assets"]:
        payload = _download_exact_asset(asset["api_url"], token=token)
        if len(payload) != asset["size"] or _sha256(payload) != asset["sha256"]:
            raise PublicationContractError(
                f"downloaded release asset differs from record: {asset['name']}"
            )
        (destination / asset["name"]).write_bytes(payload)
    return verify_bundle(destination)


def write_acquisition(record: Mapping[str, Any], output: Path) -> Path:
    """Write a canonical acquisition record and return its detached digest."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(record)
    output.write_bytes(payload)
    digest_path = output.with_suffix(output.suffix + ".sha256")
    digest_path.write_text(
        f"{_sha256(payload)}  {output.name}\n", encoding="ascii", newline="\n"
    )
    return digest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--authority", choices=("protected-main", "tag"), required=True)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    verify = subparsers.add_parser("verify-bundle")
    verify.add_argument("--bundle-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    actions = subparsers.add_parser("record-actions")
    actions.add_argument("--bundle-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    actions.add_argument("--artifact-name", required=True)
    actions.add_argument("--artifact-id", type=int, required=True)
    actions.add_argument("--artifact-url", required=True)
    actions.add_argument("--artifact-digest", required=True)
    actions.add_argument("--retention-days", type=int, required=True)
    actions.add_argument("--attestation-id", required=True)
    actions.add_argument("--attestation-url", required=True)
    actions.add_argument("--output", type=Path, required=True)
    release = subparsers.add_parser("record-release")
    release.add_argument("--bundle-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    release.add_argument("--release-metadata", type=Path, required=True)
    release.add_argument("--attestation-id", required=True)
    release.add_argument("--attestation-url", required=True)
    release.add_argument("--output", type=Path, required=True)
    download = subparsers.add_parser("verify-release-download")
    download.add_argument("--acquisition", type=Path, required=True)
    download.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fail-closed public publication command."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            inventory = build_bundle(
                args.repo_root,
                output_dir=args.output_dir,
                authority=args.authority,
            )
            sys.stdout.write(f"verified {len(inventory)} companion payload files\n")
        elif args.command == "verify-bundle":
            inventory = verify_bundle(args.bundle_dir)
            sys.stdout.write(f"verified {len(inventory)} companion payload files\n")
        elif args.command == "record-actions":
            record = build_actions_acquisition(
                args.bundle_dir,
                env=os.environ,
                artifact_metadata={
                    "name": args.artifact_name,
                    "id": args.artifact_id,
                    "url": args.artifact_url,
                    "digest": args.artifact_digest,
                    "retention_days": args.retention_days,
                },
                attestation_id=args.attestation_id,
                attestation_url=args.attestation_url,
            )
            write_acquisition(record, args.output)
        elif args.command == "record-release":
            metadata = json.loads(args.release_metadata.read_text(encoding="utf-8"))
            record = build_release_acquisition(
                args.bundle_dir,
                env=os.environ,
                release_metadata=metadata,
                attestation_id=args.attestation_id,
                attestation_url=args.attestation_url,
            )
            write_acquisition(record, args.output)
        else:
            acquisition = json.loads(args.acquisition.read_text(encoding="utf-8"))
            inventory = download_release_payloads(
                acquisition,
                output_dir=args.output_dir,
                token=os.environ.get("GH_TOKEN", ""),
            )
            sys.stdout.write(f"acquired and verified {len(inventory)} release files\n")
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        jsonschema.ValidationError,
        PublicationContractError,
        companion_catalog.CatalogAuthorityError,
    ) as exc:
        sys.stderr.write(f"companion publication refused: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
