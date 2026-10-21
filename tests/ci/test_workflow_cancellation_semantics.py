"""Workflow cancellation semantics for merge queue safety (UD-CI-032, issue #13573)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "ci" / "audit_workflow_cancellation_semantics.py"
PR_ONLY_CANCEL = "${{ github.event_name == 'pull_request' }}"
READONLY_QUEUE_IGNORE = "gh-readonly-queue/**"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_workflow_cancellation_semantics_under_test", AUDIT_SCRIPT
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit_module():
    return _load_audit_module()


def _load(name: str) -> dict:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _on(data: dict) -> dict:
    on = data.get("on", data.get(True, {}))
    assert isinstance(on, dict)
    return on


def test_inventory_audit_passes_for_every_active_workflow(audit_module) -> None:
    """Every workflow must satisfy cancel-in-progress and branches-ignore rules."""
    paths = audit_module.iter_workflow_files(WORKFLOWS)
    violations = audit_module.audit_workflow_paths(paths)
    assert violations == []


def test_ci_standard_never_cancels_merge_queue_or_main_runs() -> None:
    data = _load("ci-standard.yml")
    concurrency = data["concurrency"]
    assert concurrency["cancel-in-progress"] == PR_ONLY_CANCEL
    group = concurrency["group"]
    assert "github.event_name" in group
    assert "merge_group" in _on(data)


def test_push_triggers_do_not_combine_branches_and_branches_ignore() -> None:
    offenders: list[str] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        push = _on(data).get("push")
        if isinstance(push, dict) and "branches" in push and "branches-ignore" in push:
            offenders.append(path.name)
    assert offenders == []


def test_readonly_queue_push_compliance(audit_module) -> None:
    """``branches: [main]`` satisfies queue safety without ``branches-ignore``."""
    for path in audit_module.iter_workflow_files(WORKFLOWS):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        on = audit_module._workflow_on(data)
        push = audit_module._push_trigger(on)
        push_branches = audit_module._push_branches(on)
        assert audit_module._readonly_queue_push_compliance(push, push_branches) == []


def test_workflow_lint_enforces_merge_queue_cancellation_audit() -> None:
    lint = (WORKFLOWS / "lint-workflow-files.yml").read_text(encoding="utf-8")
    assert "audit_workflow_cancellation_semantics.py" in lint
    assert "merge-queue cancellation semantics" in lint


def test_inventory_lists_cancel_and_readonly_queue_compliance(audit_module) -> None:
    """Inventory helper returns structured rows for reporting."""
    rows = []
    for path in audit_module.iter_workflow_files(WORKFLOWS):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        on = audit_module._workflow_on(data)
        push = audit_module._push_trigger(on)
        push_branches = audit_module._push_branches(on)
        concurrency = data.get("concurrency") or {}
        rows.append(
            {
                "workflow": path.name,
                "cancel_in_progress": concurrency.get("cancel-in-progress"),
                "readonly_queue_compliant": not audit_module._readonly_queue_push_compliance(
                    push, push_branches
                ),
            }
        )
    assert rows, "expected at least one workflow row"
    assert all("cancel_in_progress" in row for row in rows)
