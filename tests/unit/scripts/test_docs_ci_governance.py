from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

# Anchored to the repo, not the process working directory: these assertions were
# silently unenforced because a relative path raised FileNotFoundError under the
# suite's chdir, so both tests in this file failed for a reason unrelated to what
# they check.
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def test_docs_ci_runs_docs_governance() -> None:
    """Docs-only CI must enforce ADR numbering and catalog governance."""
    workflow = yaml.safe_load((WORKFLOWS / "docs-ci.yml").read_text(encoding="utf-8"))
    # `docs-quality-gate`, not `quality-gate`: this job used to share the name
    # of the CI Standard aggregate, so both published check runs under the one
    # required context and branch protection accepted whichever reported.
    quality_gate_steps = workflow["jobs"]["docs-quality-gate"]["steps"]

    matching_steps = [
        step for step in quality_gate_steps if step.get("name") == "Docs governance"
    ]

    assert matching_steps == [
        {
            "name": "Docs governance",
            "run": "python3 scripts/check_docs_governance.py",
        }
    ]


def test_docs_governance_workflow_installs_pytest_asyncio() -> None:
    """The isolated docs-governance venv must satisfy pyproject pytest options."""
    workflow = yaml.safe_load(
        (WORKFLOWS / "docs-governance.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["doc-governance"]["steps"]
    install_steps = [
        step for step in steps if step.get("name") == "Install test dependencies"
    ]

    assert len(install_steps) == 1
    install_command = install_steps[0]["run"]
    assert "pytest pytest-asyncio pytest-timeout" in install_command
