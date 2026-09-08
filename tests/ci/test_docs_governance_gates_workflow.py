"""Protected CI contracts for merge-blocking docs governance (issue: PR #9418).

Docs governance used to live only in docs-governance.yml, whose
`doc-governance` check is not a required status context — so auto-merge fired
as soon as `quality-gate` passed and could land a governance regression the
docs check reported seconds later (PR #9418 pushed an ADR past the doc-size
budget red onto main). These tests pin the fold-in of docs governance into the
required `quality-gate` aggregate: the changed-paths classification, the
dedicated gate job, the fail-closed aggregate wiring, and the shared composite
action that keeps ci-standard.yml and docs-governance.yml running identical
steps.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci-standard.yml"
DOCS_WORKFLOW = ROOT / ".github/workflows/docs-governance.yml"
ACTION = ROOT / ".github/actions/docs-governance-checks/action.yml"
pytestmark = pytest.mark.unit


def _workflow() -> tuple[str, dict[str, object]]:
    text = WORKFLOW.read_text(encoding="utf-8")
    return text, yaml.safe_load(text)


def test_docs_governance_paths_drive_a_dedicated_protected_job() -> None:
    _, workflow = _workflow()
    jobs = workflow["jobs"]
    changed = jobs["changed-paths"]
    gates = jobs["docs-governance-gates"]

    assert changed["outputs"]["docs"] == "${{ steps.detect.outputs.docs }}"
    detect = next(
        step["run"] for step in changed["steps"] if step.get("id") == "detect"
    )
    for path in (
        "docs/*",
        "manuals/upstreamdrift/*",
        "scripts/check_doc_size_budget.py",
        "scripts/config/doc_size_budget.json",
        "scripts/check_document_title_case.py",
        ".github/actions/docs-governance-checks/*",
    ):
        assert path in detect
    assert 'echo "docs=true"' in detect
    assert gates["needs"] == ["pick-runner", "changed-paths"]
    assert gates["if"] == "needs.changed-paths.outputs.docs == 'true'"
    assert any(
        step.get("uses") == "./.github/actions/docs-governance-checks"
        for step in gates["steps"]
    )


def test_required_quality_gate_fails_closed_on_docs_governance_result() -> None:
    _, workflow = _workflow()
    gate = workflow["jobs"]["quality-gate"]
    script = gate["steps"][0]["run"]

    assert "docs-governance-gates" in gate["needs"]
    assert gate["steps"][0]["env"]["DOCS_GOVERNANCE_GATES"] == (
        "${{ needs.docs-governance-gates.result }}"
    )
    assert gate["steps"][0]["env"]["DOCS_CHANGED"] == (
        "${{ needs.changed-paths.outputs.docs }}"
    )
    assert 'docs_expected="success"' in script
    assert 'docs_expected="skipped"' in script
    assert 'DOCS_GOVERNANCE_GATES" != "$docs_expected' in script
    assert script.index("docs-governance-gates reported") < script.index(
        'if [ "$failed" -ne 0 ]'
    )


def test_both_docs_governance_surfaces_share_the_composite_action() -> None:
    """docs-governance.yml and the required gate must run identical steps."""
    docs_workflow = yaml.safe_load(DOCS_WORKFLOW.read_text(encoding="utf-8"))
    doc_steps = docs_workflow["jobs"]["doc-governance"]["steps"]
    assert any(
        step.get("uses") == "./.github/actions/docs-governance-checks"
        for step in doc_steps
    )

    action = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    commands = "\n".join(step.get("run", "") for step in action["runs"]["steps"])
    for command in (
        "scripts/check_docs_governance.py",
        "scripts.check_design_manual_governance",
        "scripts/check_doc_catalog.py",
        "scripts/check_doc_size_budget.py",
        "scripts/check_document_title_case.py",
        "tests/scripts/test_doc_governance_checks.py",
        "tests/scripts/test_design_manual_governance_contract.py",
        "tests/scripts/test_document_title_case.py",
    ):
        assert command in commands


def test_title_case_step_skips_without_a_diff_base() -> None:
    """Schedule/dispatch runs have no base ref; a full-tree title-case scan
    reports thousands of grandfathered violations and would permanently fail
    the required gate, so the composite must skip rather than scan."""
    action = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    title_step = next(
        step
        for step in action["runs"]["steps"]
        if step.get("name") == "Validate document title capitalization"
    )
    assert 'if [ -z "$CHANGED_FROM" ]' in title_step["run"]
    assert "exit 0" in title_step["run"]
