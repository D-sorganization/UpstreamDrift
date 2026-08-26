"""Contracts for the deterministic Canonical Core dependency bootstrap."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "config" / "ci" / "conformance-wheelhouse-v1.json"
LOCK_PATH = REPO_ROOT / "config" / "ci" / "conformance-bootstrap-py311.lock"
ACTION_PATH = REPO_ROOT / ".github" / "actions" / "bootstrap-conformance-dependencies"
WORKFLOW_ROOT = REPO_ROOT / ".github" / "workflows"

EXPECTED_VERSIONS = {
    "annotated-types": "0.7.0",
    "numpy": "2.2.6",
    "pydantic": "2.12.5",
    "pydantic-core": "2.41.5",
    "scipy": "1.14.1",
    "typing-extensions": "4.15.0",
    "typing-inspection": "0.4.2",
}


def _write_fixture_manifest(root: Path, payloads: dict[str, bytes]) -> Path:
    artifacts = []
    for filename, payload in payloads.items():
        artifacts.append(
            {
                "filename": filename,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
    manifest = {
        "schema_version": "upstreamdrift-conformance-wheelhouse/1",
        "python": "3.11",
        "implementation": "cpython",
        "platform_system": "Linux",
        "machine": "x86_64",
        "artifacts": artifacts,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_repository_governs_exact_core_wheel_set() -> None:
    """The bounded wheelhouse must pin every core artifact and its digest."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    lock = LOCK_PATH.read_text(encoding="utf-8")

    assert manifest["schema_version"] == "upstreamdrift-conformance-wheelhouse/1"
    assert manifest["python"] == "3.11"
    assert manifest["implementation"] == "cpython"
    assert manifest["platform_system"] == "Linux"
    assert manifest["machine"] == "x86_64"

    versions = {
        artifact["distribution"]: artifact["version"]
        for artifact in manifest["artifacts"]
    }
    assert versions == EXPECTED_VERSIONS
    for artifact in manifest["artifacts"]:
        assert len(artifact["sha256"]) == 64
        assert artifact["sha256"].isalnum()
        assert artifact["size"] > 0
        assert artifact["filename"].endswith(".whl")
        requirement = f"{artifact['distribution']}=={artifact['version']}"
        assert requirement in lock
        assert f"--hash=sha256:{artifact['sha256']}" in lock
        assert artifact["provenance"]["repository"] == "PyPI"
        assert artifact["provenance"]["json_url"].startswith("https://pypi.org/pypi/")


def test_shared_action_restores_then_installs_verified_wheelhouse() -> None:
    """One pinned composite action must own cache restore and no-index install."""
    action = (ACTION_PATH / "action.yml").read_text(encoding="utf-8")

    assert "actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9" in action
    assert "hashFiles('config/ci/conformance-wheelhouse-v1.json')" in action
    assert "scripts/ci/conformance_dependency_bootstrap.py install" in action
    assert "cache-hit" in action


def test_seed_workflow_is_manual_bounded_and_hash_checked() -> None:
    """Cache population must be manual and use the source-controlled hash lock."""
    workflow = (WORKFLOW_ROOT / "seed-conformance-wheelhouse.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "--require-hashes" in workflow
    assert "--only-binary=:all:" in workflow
    assert "conformance-bootstrap-py311.lock" in workflow
    assert "conformance_dependency_bootstrap.py verify" in workflow
    for approved_runner in range(1, 5):
        assert f"d-sorg-local-Oglaptop-{approved_runner}" in workflow
    assert "d-sorg-local-Oglaptop-5" not in workflow
    assert "cache seeding does not approve package versions" in workflow


@pytest.mark.parametrize(
    "workflow_name,expected_calls",
    [
        ("cross-engine-equivalence.yml", 2),
        ("cross-engine-leaderboard.yml", 1),
        ("cross-engine-leaderboard-publish.yml", 1),
    ],
)
def test_cross_engine_workflows_share_offline_bootstrap(
    workflow_name: str,
    expected_calls: int,
) -> None:
    """Conformance and leaderboard jobs must not carry ad-hoc core installs."""
    workflow = (WORKFLOW_ROOT / workflow_name).read_text(encoding="utf-8")

    action_ref = "./.github/actions/bootstrap-conformance-dependencies"
    assert workflow.count(action_ref) == expected_calls
    assert 'pip install --force-reinstall "pydantic==2.12.5"' not in workflow
    assert (
        'pip install --force-reinstall --no-cache-dir "numpy==2.2.6" "scipy==1.14.1"'
    ) not in workflow


def test_verify_fails_closed_when_wheelhouse_is_missing(tmp_path: Path) -> None:
    """An absent approved cache must fail before pip or tests can execute."""
    from scripts.ci.conformance_dependency_bootstrap import (
        BootstrapError,
        RuntimeContract,
        verify_wheelhouse,
    )

    manifest_path = _write_fixture_manifest(
        tmp_path,
        {"example-1.0-py3-none-any.whl": b"approved"},
    )

    with pytest.raises(BootstrapError, match="approved wheelhouse is missing"):
        verify_wheelhouse(
            manifest_path,
            tmp_path / "missing",
            RuntimeContract.current_for_tests(),
        )


def test_verify_rejects_corrupt_and_unapproved_wheels(tmp_path: Path) -> None:
    """Digest drift and extra wheel files both violate the bounded contract."""
    from scripts.ci.conformance_dependency_bootstrap import (
        BootstrapError,
        RuntimeContract,
        verify_wheelhouse,
    )

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    filename = "example-1.0-py3-none-any.whl"
    manifest_path = _write_fixture_manifest(tmp_path, {filename: b"approved"})
    (wheelhouse / filename).write_bytes(b"corrupt")

    with pytest.raises(BootstrapError, match="sha256 mismatch"):
        verify_wheelhouse(
            manifest_path,
            wheelhouse,
            RuntimeContract.current_for_tests(),
        )

    (wheelhouse / filename).write_bytes(b"approved")
    (wheelhouse / "unapproved-1.0-py3-none-any.whl").write_bytes(b"extra")
    with pytest.raises(BootstrapError, match="unapproved wheel artifacts"):
        verify_wheelhouse(
            manifest_path,
            wheelhouse,
            RuntimeContract.current_for_tests(),
        )


def test_install_uses_no_index_hashes_and_binary_wheels(tmp_path: Path) -> None:
    """Verified artifacts must reach pip only through the no-index boundary."""
    from scripts.ci.conformance_dependency_bootstrap import (
        RuntimeContract,
        install_verified_wheelhouse,
    )

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    filename = "example-1.0-py3-none-any.whl"
    (wheelhouse / filename).write_bytes(b"approved")
    manifest_path = _write_fixture_manifest(tmp_path, {filename: b"approved"})
    lock_path = tmp_path / "requirements.lock"
    digest = hashlib.sha256(b"approved").hexdigest()
    lock_path.write_text(
        f"example==1.0 --hash=sha256:{digest}\n",
        encoding="utf-8",
    )
    runner = Mock()

    install_verified_wheelhouse(
        manifest_path,
        lock_path,
        wheelhouse,
        RuntimeContract.current_for_tests(),
        runner=runner,
    )

    command = runner.call_args.args[0]
    assert command[0:3] == [sys.executable, "-m", "pip"]
    assert "--no-index" in command
    assert "--require-hashes" in command
    assert "--only-binary=:all:" in command
    assert "--force-reinstall" in command
    assert "--find-links" in command
    runner.assert_called_once()


def test_verify_rejects_unsupported_runtime_before_artifact_access(
    tmp_path: Path,
) -> None:
    """A wheelhouse may not be reused across Python or platform boundaries."""
    from scripts.ci.conformance_dependency_bootstrap import (
        BootstrapError,
        RuntimeContract,
        verify_wheelhouse,
    )

    manifest_path = _write_fixture_manifest(
        tmp_path,
        {"example-1.0-py3-none-any.whl": b"approved"},
    )
    runtime = RuntimeContract(
        python="3.12",
        implementation="cpython",
        platform_system="Linux",
        machine="x86_64",
    )

    with pytest.raises(BootstrapError, match="runtime does not match"):
        verify_wheelhouse(manifest_path, tmp_path / "missing", runtime)
