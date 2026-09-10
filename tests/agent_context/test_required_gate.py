"""A navigation failure must propagate into the protected required check."""

from pathlib import Path

import pytest
import yaml


pytestmark = pytest.mark.unit


def test_context_gate_is_required_even_for_docs_only_changes() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github/workflows/ci-standard.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    assert "if" not in jobs["agent-context"]
    assert "agent-context" in jobs["quality-gate"]["needs"]
    assert jobs["quality-gate"]["if"] == "always()"
    steps = jobs["quality-gate"]["steps"]
    assertion = next(
        step for step in steps if step.get("name") == "Require Current Agent Context"
    )
    assert assertion["env"]["CONTEXT_RESULT"].endswith("needs.agent-context.result }}")
    assert assertion["run"] == 'test "$CONTEXT_RESULT" = "success"'
    assert not assertion.get("continue-on-error", False)


def test_candidate_provider_cannot_merge_before_tools_publication() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github/workflows/ci-standard.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["agent-context"]["steps"]
    publication = next(
        step for step in steps if step.get("name") == "Require Published Tools Revision"
    )
    assert (
        "merge-base --is-ancestor HEAD refs/remotes/origin/main" in publication["run"]
    )
    assert not publication.get("continue-on-error", False)
    assert "if" not in publication
    names = [step.get("name") for step in steps]
    assert names.index("Validate Real Navigation Tasks") < names.index(
        "Require Published Tools Revision"
    )
