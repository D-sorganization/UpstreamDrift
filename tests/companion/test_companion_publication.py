"""RED/GREEN publication and acquisition contracts for issue #9192."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/companion"
MANIFEST_SCHEMA = (
    REPO_ROOT / "docs/api/contracts/upstreamdrift-companion-v1.schema.json"
)
ACQUISITION_SCHEMA = (
    REPO_ROOT / "docs/api/contracts/upstreamdrift-companion-acquisition-v1.schema.json"
)
POLICY_PATH = (
    REPO_ROOT / "docs/api/contracts/upstreamdrift-companion-compatibility-v1.json"
)
COMMIT = "1" * 40


def _module():
    from scripts import companion_publication

    return companion_publication


def _ci_env(*, authority: str) -> dict[str, str]:
    env = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REPOSITORY": "D-sorganization/UpstreamDrift",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "123456",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA": COMMIT,
    }
    if authority == "protected-main":
        env.update({"GITHUB_REF": "refs/heads/main", "GITHUB_REF_NAME": "main"})
    else:
        env.update({"GITHUB_REF": "refs/tags/v2.1.1", "GITHUB_REF_NAME": "v2.1.1"})
    return env


@pytest.mark.parametrize("authority", ["protected-main", "tag"])
def test_ci_authority_accepts_only_exact_push_context(authority: str) -> None:
    publication = _module()

    context = publication.validate_ci_authority(
        authority, env=_ci_env(authority=authority), head_commit=COMMIT
    )

    assert context.source_commit == COMMIT
    assert context.authority == authority


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"GITHUB_EVENT_NAME": "pull_request"}, "push event"),
        ({"GITHUB_SHA": "2" * 40}, "does not match"),
        ({"GITHUB_REPOSITORY": "someone/fork"}, "repository"),
        ({"GITHUB_REF": "refs/pull/9192/merge"}, "protected main"),
        ({"GITHUB_ACTIONS": "false"}, "GitHub Actions"),
    ],
)
def test_protected_authority_refuses_pr_fork_and_mismatch(
    mutation: dict[str, str], message: str
) -> None:
    publication = _module()
    env = _ci_env(authority="protected-main")
    env.update(mutation)

    with pytest.raises(publication.PublicationContractError, match=message):
        publication.validate_ci_authority("protected-main", env=env, head_commit=COMMIT)


def test_tag_authority_refuses_non_release_tag() -> None:
    publication = _module()
    env = _ci_env(authority="tag")
    env["GITHUB_REF"] = "refs/tags/latest"
    env["GITHUB_REF_NAME"] = "latest"

    with pytest.raises(publication.PublicationContractError, match="release tag"):
        publication.validate_ci_authority("tag", env=env, head_commit=COMMIT)


def test_compatibility_policy_validates_current_and_rejected_fixtures() -> None:
    publication = _module()
    policy = publication.load_compatibility_policy(REPO_ROOT)

    publication.validate_compatibility_policy(REPO_ROOT, policy)

    assert policy["current"] == "1.0.0"
    assert policy["previous_supported"] == []
    schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
    current = json.loads(
        (FIXTURE_ROOT / "current-v1.0.0.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(current)
    for name in (
        "rejected-future-v2.0.0.json",
        "rejected-incompatible-v1.0.0.json",
    ):
        fixture = json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))
        assert list(jsonschema.Draft202012Validator(schema).iter_errors(fixture))


def test_compatibility_policy_requires_fixture_for_every_previous_version() -> None:
    publication = _module()
    policy = copy.deepcopy(publication.load_compatibility_policy(REPO_ROOT))
    policy["previous_supported"] = ["0.9.0"]

    with pytest.raises(
        publication.PublicationContractError, match="previous supported fixture"
    ):
        publication.validate_compatibility_policy(REPO_ROOT, policy)


def test_compatibility_policy_rejects_stale_current_schema() -> None:
    publication = _module()
    policy = copy.deepcopy(publication.load_compatibility_policy(REPO_ROOT))
    policy["current"] = "1.0.1"

    with pytest.raises(
        publication.PublicationContractError, match="current version is stale"
    ):
        publication.validate_compatibility_policy(REPO_ROOT, policy)


def test_build_bundle_is_canonical_and_self_verifying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = _module()
    head = publication.git_head_commit(REPO_ROOT)
    env = _ci_env(authority="protected-main")
    env["GITHUB_SHA"] = head
    monkeypatch.setenv("GITHUB_SHA", head)

    first = publication.build_bundle(
        REPO_ROOT,
        output_dir=tmp_path / "first",
        authority="protected-main",
        env=env,
        require_clean=False,
    )
    second = publication.build_bundle(
        REPO_ROOT,
        output_dir=tmp_path / "second",
        authority="protected-main",
        env=env,
        require_clean=False,
    )

    assert first == second
    assert set(first) == set(publication.PAYLOAD_ASSET_NAMES)
    for name, metadata in first.items():
        payload = (tmp_path / "first" / name).read_bytes()
        assert metadata["size"] == len(payload)
        assert metadata["sha256"] == hashlib.sha256(payload).hexdigest()
    publication.verify_bundle(tmp_path / "first")


def test_bundle_refuses_missing_renamed_stale_and_bad_digest(tmp_path: Path) -> None:
    publication = _module()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in publication.PAYLOAD_ASSET_NAMES:
        (bundle / name).write_text("placeholder\n", encoding="utf-8")

    with pytest.raises(publication.PublicationContractError):
        publication.verify_bundle(bundle)


def test_actions_record_is_explicitly_ephemeral_and_exact(tmp_path: Path) -> None:
    publication = _module()
    bundle = _build_test_bundle(publication, tmp_path)
    env = _ci_env(authority="protected-main")
    record = publication.build_actions_acquisition(
        bundle,
        env=env,
        artifact_name=f"upstreamdrift-companion-{COMMIT}",
        artifact_id=987,
        artifact_url=(
            "https://github.com/D-sorganization/UpstreamDrift/"
            "actions/runs/123456/artifacts/987"
        ),
        artifact_digest="sha256:" + "b" * 64,
        retention_days=30,
        attestation_id="456",
        attestation_url=(
            "https://github.com/D-sorganization/UpstreamDrift/attestations/456"
        ),
    )

    jsonschema.Draft202012Validator(_acquisition_schema()).validate(record)
    assert record["channel"] == "actions"
    assert record["delivery"]["durability"] == "ephemeral"
    assert record["delivery"]["release"] is None
    assert record["limitations"] == [
        "Actions artifacts expire; no durable release acquisition URL exists yet."
    ]


def test_actions_record_rejects_mutable_or_cross_run_url(tmp_path: Path) -> None:
    publication = _module()
    bundle = _build_test_bundle(publication, tmp_path)

    with pytest.raises(publication.PublicationContractError, match="artifact URL"):
        publication.build_actions_acquisition(
            bundle,
            env=_ci_env(authority="protected-main"),
            artifact_name=f"upstreamdrift-companion-{COMMIT}",
            artifact_id=987,
            artifact_url="https://github.com/D-sorganization/UpstreamDrift/actions/runs/999/artifacts/987",
            artifact_digest="sha256:" + "b" * 64,
            retention_days=30,
            attestation_id="456",
            attestation_url="https://github.com/D-sorganization/UpstreamDrift/attestations/456",
        )


def test_release_record_uses_numeric_api_asset_identity_and_attestation(
    tmp_path: Path,
) -> None:
    publication = _module()
    bundle = _build_test_bundle(publication, tmp_path)
    metadata = _release_metadata(publication, bundle)
    record = publication.build_release_acquisition(
        bundle,
        env=_ci_env(authority="tag"),
        release_metadata=metadata,
        attestation_id="456",
        attestation_url=(
            "https://github.com/D-sorganization/UpstreamDrift/attestations/456"
        ),
    )

    jsonschema.Draft202012Validator(_acquisition_schema()).validate(record)
    assert record["channel"] == "release"
    assert record["delivery"]["durability"] == "immutable-release"
    assert record["delivery"]["release"]["release_id"] == 777
    assert all(
        asset["api_url"].startswith(
            "https://api.github.com/repos/D-sorganization/UpstreamDrift/releases/assets/"
        )
        for asset in record["delivery"]["release"]["assets"]
    )


def test_release_download_allows_one_github_object_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = _module()
    bundle = _build_test_bundle(publication, tmp_path)
    record = publication.build_release_acquisition(
        bundle,
        env=_ci_env(authority="tag"),
        release_metadata=_release_metadata(publication, bundle),
        attestation_id="456",
        attestation_url=(
            "https://github.com/D-sorganization/UpstreamDrift/attestations/456"
        ),
    )
    payload_by_id = {
        str(asset["asset_id"]): (bundle / asset["name"]).read_bytes()
        for asset in record["delivery"]["release"]["assets"]
    }

    def request(url: str, *, token: str | None):
        if url.startswith("https://api.github.com/"):
            assert token == "secret"
            asset_id = url.rsplit("/", 1)[1]
            return (
                302,
                {
                    "Location": (
                        "https://release-assets.githubusercontent.com/"
                        f"download/{asset_id}"
                    )
                },
                b"",
            )
        assert token is None
        return 200, {}, payload_by_id[url.rsplit("/", 1)[1]]

    monkeypatch.setattr(publication, "_request_no_redirect", request)
    inventory = publication.download_release_payloads(
        record, output_dir=tmp_path / "download", token="secret"
    )

    assert set(inventory) == set(publication.PAYLOAD_ASSET_NAMES)


def test_release_download_rejects_untrusted_or_second_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = _module()
    bundle = _build_test_bundle(publication, tmp_path)
    record = publication.build_release_acquisition(
        bundle,
        env=_ci_env(authority="tag"),
        release_metadata=_release_metadata(publication, bundle),
        attestation_id="456",
        attestation_url=(
            "https://github.com/D-sorganization/UpstreamDrift/attestations/456"
        ),
    )

    def request(url: str, *, token: str | None):
        del url, token
        return 302, {"Location": "https://example.invalid/mutable"}, b""

    monkeypatch.setattr(publication, "_request_no_redirect", request)
    with pytest.raises(publication.PublicationContractError, match="unexpected"):
        publication.download_release_payloads(
            record, output_dir=tmp_path / "download", token="secret"
        )


@pytest.mark.parametrize("defect", ["missing", "size", "mutable_url", "unattested"])
def test_release_record_fails_closed_on_asset_or_attestation_defect(
    tmp_path: Path, defect: str
) -> None:
    publication = _module()
    bundle = _build_test_bundle(publication, tmp_path)
    metadata = _release_metadata(publication, bundle)
    attestation_id = "456"
    if defect == "missing":
        metadata["assets"].pop()
    elif defect == "size":
        metadata["assets"][0]["size"] += 1
    elif defect == "mutable_url":
        metadata["assets"][0]["url"] = metadata["assets"][0]["browser_download_url"]
    else:
        attestation_id = ""

    with pytest.raises(publication.PublicationContractError):
        publication.build_release_acquisition(
            bundle,
            env=_ci_env(authority="tag"),
            release_metadata=metadata,
            attestation_id=attestation_id,
            attestation_url="https://github.com/D-sorganization/UpstreamDrift/attestations/456",
        )


def test_existing_workflows_share_publication_command_and_publish_atomically() -> None:
    yaml = pytest.importorskip("yaml")
    workflow_path = REPO_ROOT / ".github/workflows/release.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]

    main_job = jobs["companion-protected-main"]
    release_build = jobs["build"]
    main_command = next(
        step["run"]
        for step in main_job["steps"]
        if step.get("name") == "Build companion publication bundle"
    )
    tag_command = next(
        step["run"]
        for step in release_build["steps"]
        if step.get("name") == "Build companion publication bundle"
    )
    assert "scripts/companion_publication.py build" in main_command
    assert "scripts/companion_publication.py build" in tag_command
    assert "--authority protected-main" in main_command
    assert "--authority tag" in tag_command

    release_job = jobs["create-release"]
    release_step = next(
        step for step in release_job["steps"] if step.get("id") == "github-release"
    )
    assert release_step["with"]["draft"] is True
    assert release_step["with"]["overwrite_files"] is False
    finalize = jobs["record-companion-release"]
    assert "create-release" in finalize["needs"]
    assert any(
        step.get("name") == "Acquire and verify numeric release assets"
        for step in finalize["steps"]
    )
    assert any(
        step.get("name") == "Publish verified release" for step in finalize["steps"]
    )


def _build_test_bundle(publication, tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = json.loads(
        (FIXTURE_ROOT / "current-v1.0.0.json").read_text(encoding="utf-8")
    )
    manifest["source"]["commit"] = COMMIT
    payloads = {
        publication.MANIFEST_NAME: publication.canonical_json(manifest),
        publication.MANIFEST_SCHEMA_NAME: MANIFEST_SCHEMA.read_bytes(),
        publication.ACQUISITION_SCHEMA_NAME: ACQUISITION_SCHEMA.read_bytes(),
        publication.COMPATIBILITY_POLICY_NAME: POLICY_PATH.read_bytes(),
    }
    for name, payload in payloads.items():
        (bundle / name).write_bytes(payload)
        (bundle / f"{name}.sha256").write_text(
            f"{hashlib.sha256(payload).hexdigest()}  {name}\n",
            encoding="ascii",
            newline="\n",
        )
    publication.verify_bundle(bundle)
    return bundle


def _release_metadata(publication, bundle: Path) -> dict[str, object]:
    assets = []
    for index, name in enumerate(publication.PAYLOAD_ASSET_NAMES, start=1000):
        assets.append(
            {
                "id": index,
                "name": name,
                "size": (bundle / name).stat().st_size,
                "url": (
                    "https://api.github.com/repos/D-sorganization/UpstreamDrift/"
                    f"releases/assets/{index}"
                ),
                "browser_download_url": (
                    "https://github.com/D-sorganization/UpstreamDrift/"
                    f"releases/download/v2.1.1/{name}"
                ),
            }
        )
    return {
        "id": 777,
        "tag_name": "v2.1.1",
        "url": "https://api.github.com/repos/D-sorganization/UpstreamDrift/releases/777",
        "html_url": "https://github.com/D-sorganization/UpstreamDrift/releases/tag/v2.1.1",
        "draft": True,
        "assets": assets,
    }


def _acquisition_schema() -> dict[str, object]:
    return json.loads(ACQUISITION_SCHEMA.read_text(encoding="utf-8"))
