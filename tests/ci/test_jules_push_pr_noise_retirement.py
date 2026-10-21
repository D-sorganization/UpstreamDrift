"""UD-CI-003: Jules push/PR noise workflow retirement batch 1 (#13575)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci.jules_push_pr_noise_retirement import (
    all_retired_active_paths,
    check_not_required_contexts,
    check_repository_state,
    load_config,
    required_contexts,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "scripts" / "config" / "jules_push_pr_noise_retirement.json"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
ARCHIVE = (
    REPO_ROOT / ".github" / "workflows-retired" / "jules-push-pr-noise" / "batch-1"
)
REQUIRED_CHECKS = REPO_ROOT / "scripts" / "config" / "ci-required-checks-live.json"


def test_batch_1_allowlist_matches_archived_basenames() -> None:
    config = load_config(CONFIG)
    batch = config["batches"]["1"]
    retired = batch["retired_active_paths"]
    assert retired == [
        ".github/workflows/Jules-Control-Tower.yml",
        ".github/workflows/Jules-Supersede-Check.yml",
    ]
    for path in retired:
        assert (ARCHIVE / Path(path).name).is_file()


def test_retired_paths_absent_from_active_workflows_dir() -> None:
    config = load_config(CONFIG)
    for path in all_retired_active_paths(config):
        assert not (REPO_ROOT / path).is_file(), path


def test_k1_k2_keep_set_present_and_only_jules_active_files() -> None:
    config = load_config(CONFIG)
    keep = set(config["k1_k2_keep"])
    assert keep == {
        "Jules-Redundant-PR-Closer.yml",
        "Jules-Redundant-Issue-Closer.yml",
        "Jules-Auto-Assign-Issues.yml",
        "Jules-Issue-Mention-Handler.yml",
        "Jules-Diff-Verifier.yml",
    }
    for name in keep:
        assert (WORKFLOWS / name).is_file()
    jules_active = sorted(WORKFLOWS.glob("Jules-*.yml"))
    assert {p.name for p in jules_active} == keep


def test_repository_state_guard_passes() -> None:
    config = load_config(CONFIG)
    assert check_repository_state(REPO_ROOT, config) == []


def test_retired_workflows_not_required_contexts() -> None:
    config = load_config(CONFIG)
    if not REQUIRED_CHECKS.is_file():
        pytest.skip("ci-required-checks-live.json not yet on UpstreamDrift main")
    findings = check_not_required_contexts(config)
    assert findings == []
    payload = json.loads(REQUIRED_CHECKS.read_text(encoding="utf-8"))
    required = payload.get("required_contexts", payload.get("contexts", []))
    for name in required_contexts(config):
        assert name not in required


def test_disable_workflow_exists_for_post_merge() -> None:
    path = WORKFLOWS / "jules-push-pr-noise-disable.yml"
    text = path.read_text(encoding="utf-8")
    assert "jules_push_pr_noise_retirement.py" in text
    assert "--disable" in text
    assert "branches: [main]" in text or "branches:\n      - main" in text


def test_disable_workflow_on_lint_cancel_false_allowlist() -> None:
    lint = (WORKFLOWS / "lint-workflow-files.yml").read_text(encoding="utf-8")
    allowlist = next(
        line for line in lint.splitlines() if "nightly-publish.yml" in line
    )
    assert "jules-push-pr-noise-disable.yml" in allowlist
