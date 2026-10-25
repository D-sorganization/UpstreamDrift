"""Pinned export of live branch-protection required contexts (RM #2216 / #13572)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci.jules_push_pr_noise_retirement import (
    check_not_required_contexts,
    load_config,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_CHECKS = REPO_ROOT / "scripts" / "config" / "ci-required-checks-live.json"
JULES_RETIREMENT = (
    REPO_ROOT / "scripts" / "config" / "jules_push_pr_noise_retirement.json"
)
LEADERBOARD_PUBLISH_CONTEXT = "publish cross-engine leaderboard JSON"


def _payload() -> dict:
    assert REQUIRED_CHECKS.is_file(), "ci-required-checks-live.json must be present"
    return json.loads(REQUIRED_CHECKS.read_text(encoding="utf-8"))


def test_required_contexts_export_is_a_non_empty_list() -> None:
    payload = _payload()
    required = payload.get("required_contexts", payload.get("contexts"))
    assert isinstance(required, list)
    assert required, "branch protection must name at least one required context"
    assert all(isinstance(name, str) and name.strip() for name in required)


def test_quality_gate_is_the_live_required_context() -> None:
    payload = _payload()
    required = payload["required_contexts"]
    assert required == ["quality-gate"]


def test_leaderboard_publish_is_not_a_required_context() -> None:
    required = _payload()["required_contexts"]
    assert LEADERBOARD_PUBLISH_CONTEXT not in required
    assert "cross-engine-leaderboard-publish" not in required


def test_retired_jules_noise_workflows_not_required() -> None:
    config = load_config(JULES_RETIREMENT)
    assert check_not_required_contexts(config) == []
