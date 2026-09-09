"""RED/GREEN publication and acquisition contracts for issue #9192."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

pytestmark = pytest.mark.unit

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
        artifact_metadata={
            "name": f"upstreamdrift-companion-{COMMIT}",
            "id": 987,
            "url": (
                "https://github.com/D-sorganization/UpstreamDrift/"
                "actions/runs/123456/artifacts/987"
            ),
            "digest": "sha256:" + "b" * 64,
            "retention_days": 30,
        },
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
            artifact_metadata={
                "name": f"upstreamdrift-companion-{COMMIT}",
                "id": 987,
                "url": "https://github.com/D-sorganization/UpstreamDrift/actions/runs/999/artifacts/987",
                "digest": "sha256:" + "b" * 64,
                "retention_days": 30,
            },
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
    assert "-m scripts.companion_publication build" in main_command
    assert "-m scripts.companion_publication build" in tag_command
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


def test_every_companion_cli_job_runs_from_explicit_github_workspace() -> None:
    yaml = pytest.importorskip("yaml")
    workflow_path = REPO_ROOT / ".github/workflows/release.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    companion_jobs = {
        job_id: job
        for job_id, job in workflow["jobs"].items()
        if any(
            "scripts.companion_publication" in step.get("run", "")
            for step in job.get("steps", [])
        )
    }

    assert companion_jobs
    for job_id, job in companion_jobs.items():
        assert job.get("defaults", {}).get("run", {}).get("working-directory") == (
            "${{ github.workspace }}"
        ), f"{job_id} can run outside the checked-out repository"


def test_public_module_cli_is_importable_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.companion_publication", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        encoding="utf-8",
        shell=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Build and verify exact-revision companion publication bundles" in (
        completed.stdout
    )


def _build_test_bundle(publication, tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = json.loads(
        (FIXTURE_ROOT / "current-v1.0.0.json").read_text(encoding="utf-8")
    )
    manifest["source"]["commit"] = COMMIT
    payloads = _payload_bytes(publication, manifest)
    for name, payload in payloads.items():
        (bundle / name).write_bytes(payload)
        (bundle / f"{name}.sha256").write_text(
            f"{hashlib.sha256(payload).hexdigest()}  {name}\n",
            encoding="ascii",
            newline="\n",
        )
    publication.verify_bundle(bundle)
    return bundle


def _payload_bytes(publication, manifest: dict) -> dict[str, bytes]:
    """Derive every base payload from one manifest via the pure helpers."""
    from scripts import companion_catalog

    manifest_bytes = publication.canonical_json(manifest)
    capabilities = companion_catalog._capabilities_payload(manifest, {})
    screenshots = companion_catalog._screenshots_payload(manifest)
    return {
        publication.MANIFEST_NAME: manifest_bytes,
        publication.CONSUMER_MANIFEST_NAME: manifest_bytes,
        publication.CAPABILITIES_NAME: publication.canonical_json(capabilities),
        publication.SCREENSHOTS_NAME: publication.canonical_json(screenshots),
        publication.MANIFEST_SCHEMA_NAME: MANIFEST_SCHEMA.read_bytes(),
        publication.CAPABILITIES_SCHEMA_NAME: (
            REPO_ROOT / publication.CAPABILITIES_SCHEMA_PATH
        ).read_bytes(),
        publication.SCREENSHOTS_SCHEMA_NAME: (
            REPO_ROOT / publication.SCREENSHOTS_SCHEMA_PATH
        ).read_bytes(),
        publication.ACQUISITION_SCHEMA_NAME: ACQUISITION_SCHEMA.read_bytes(),
        publication.COMPATIBILITY_POLICY_NAME: POLICY_PATH.read_bytes(),
    }


def _write_sidecar(bundle: Path, name: str, payload: bytes) -> None:
    (bundle / name).write_bytes(payload)
    (bundle / f"{name}.sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n",
        encoding="ascii",
        newline="\n",
    )


# --- #9416: consumer payloads, byte-identical manifest.json, check command ---


def test_payload_asset_names_derive_from_single_base_list() -> None:
    publication = _module()

    assert publication._BASE_ASSET_NAMES == (
        "upstreamdrift-companion.v1.json",
        "manifest.json",
        "capabilities.json",
        "screenshots.json",
        "upstreamdrift-companion-v1.schema.json",
        "upstreamdrift-companion-capabilities-v1.schema.json",
        "upstreamdrift-companion-screenshots-v1.schema.json",
        "upstreamdrift-companion-acquisition-v1.schema.json",
        "upstreamdrift-companion-compatibility-v1.json",
    )
    assert len(publication.PAYLOAD_ASSET_NAMES) == 18
    assert all(
        f"{name}.sha256" in publication.PAYLOAD_ASSET_NAMES
        for name in publication._BASE_ASSET_NAMES
    )
    schema = _acquisition_schema()
    assert schema["properties"]["payloads"]["minItems"] == 18
    assert schema["properties"]["payloads"]["maxItems"] == 18


def test_bundle_manifest_json_is_byte_identical_to_versioned_manifest(
    tmp_path: Path,
) -> None:
    publication = _module()
    bundle = _build_test_bundle(publication, tmp_path)
    assert (bundle / "manifest.json").read_bytes() == (
        bundle / "upstreamdrift-companion.v1.json"
    ).read_bytes()

    drifted = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    drifted["publication"]["blockers"].append("drifted consumer copy")
    _write_sidecar(bundle, "manifest.json", publication.canonical_json(drifted))

    with pytest.raises(
        publication.PublicationContractError, match="not byte-identical"
    ):
        publication.verify_bundle(bundle)


def test_bundle_refuses_cross_payload_commit_mismatch(tmp_path: Path) -> None:
    publication = _module()
    bundle = _build_test_bundle(publication, tmp_path)
    capabilities = json.loads(
        (bundle / "capabilities.json").read_text(encoding="utf-8")
    )
    capabilities["source"]["commit"] = "2" * 40
    _write_sidecar(
        bundle, "capabilities.json", publication.canonical_json(capabilities)
    )

    with pytest.raises(publication.PublicationContractError, match="embedded commit"):
        publication.verify_bundle(bundle)


def test_bundle_refuses_capabilities_for_unknown_programs(tmp_path: Path) -> None:
    publication = _module()
    bundle = _build_test_bundle(publication, tmp_path)
    capabilities = json.loads(
        (bundle / "capabilities.json").read_text(encoding="utf-8")
    )
    capabilities["programs"].append(
        {
            "id": "ghost",
            "engine_id": None,
            "support_tier": "not_applicable",
            "maturity": "unclassified",
            "availability_state": "conditional",
            "capability_ids": [],
        }
    )
    _write_sidecar(
        bundle, "capabilities.json", publication.canonical_json(capabilities)
    )

    with pytest.raises(publication.PublicationContractError, match="programs"):
        publication.verify_bundle(bundle)


def test_real_bundle_ships_consumer_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = _module()
    head = publication.git_head_commit(REPO_ROOT)
    env = _ci_env(authority="protected-main")
    env["GITHUB_SHA"] = head
    monkeypatch.setenv("GITHUB_SHA", head)

    inventory = publication.build_bundle(
        REPO_ROOT,
        output_dir=tmp_path / "bundle",
        authority="protected-main",
        env=env,
        require_clean=False,
    )

    assert set(inventory) == set(publication.PAYLOAD_ASSET_NAMES)
    bundle = tmp_path / "bundle"
    screenshots = json.loads((bundle / "screenshots.json").read_text("utf-8"))
    capabilities = json.loads((bundle / "capabilities.json").read_text("utf-8"))
    assert screenshots["source"]["commit"] == head
    assert capabilities["source"]["commit"] == head
    assert screenshots["records"]
    assert all(r["status"] == "pending" for r in screenshots["records"])
    assert capabilities["capabilities"]
    payloads, _ = publication._payload_inventory(bundle)
    embedded = {p["name"]: p["embedded_commit"] for p in payloads}
    assert embedded["manifest.json"] == head
    assert embedded["capabilities.json"] == head
    assert embedded["screenshots.json"] == head
    assert embedded["upstreamdrift-companion-v1.schema.json"] is None


def test_check_command_validates_in_memory_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    publication = _module()
    before = (
        sorted(
            p for p in (REPO_ROOT / "dist").rglob("*") if "companion" in p.as_posix()
        )
        if (REPO_ROOT / "dist").is_dir()
        else []
    )

    code = publication.main(["--repo-root", str(REPO_ROOT), "check", "--allow-dirty"])

    captured = capsys.readouterr()
    assert code == 0, captured.err
    head = publication.git_head_commit(REPO_ROOT)
    assert captured.out.startswith(f"companion check ok: commit={head} ")
    assert "screenshots=" in captured.out and "pending=" in captured.out
    after = (
        sorted(
            p for p in (REPO_ROOT / "dist").rglob("*") if "companion" in p.as_posix()
        )
        if (REPO_ROOT / "dist").is_dir()
        else []
    )
    assert after == before
    assert not list(tmp_path.iterdir())


def test_check_command_refuses_dirty_tree_without_allow_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    publication = _module()
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, shell=False)
    (tmp_path / "dirty.txt").write_text("x\n", encoding="utf-8")

    code = publication.main(["--repo-root", str(tmp_path), "check"])

    assert code == 2
    assert "companion publication refused" in capsys.readouterr().err


def test_ci_standard_runs_check_before_workflow_execution() -> None:
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/ci-standard.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["companion-workflows"]["steps"]
    names = [step.get("name") for step in steps]
    check_index = names.index(
        "Check companion manifest builds from a clean checkout (#9416)"
    )
    assert (
        "scripts.companion_publication --repo-root . check"
        in (steps[check_index]["run"])
    )
    assert check_index < names.index("Execute governed companion workflow authority")


def test_release_companion_jobs_pin_pythonpath_and_dependencies() -> None:
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    workspace = "${{ github.workspace }}"
    for job_id in ("companion-protected-main", "record-companion-release"):
        job = jobs[job_id]
        assert job["env"]["PYTHONPATH"] == workspace, job_id
        install = next(
            step["run"]
            for step in job["steps"]
            if step.get("name") == "Install companion publication dependencies"
        )
        assert "jsonschema==" in install and "pyyaml==" in install
    build_steps = [
        step
        for step in jobs["build"]["steps"]
        if "scripts.companion_publication" in step.get("run", "")
    ]
    assert build_steps
    assert all(step["env"]["PYTHONPATH"] == workspace for step in build_steps)


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
