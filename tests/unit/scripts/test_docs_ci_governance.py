from __future__ import annotations

from pathlib import Path

import yaml


def test_docs_ci_runs_docs_governance() -> None:
    """Docs-only CI must enforce ADR numbering and catalog governance."""
    workflow = yaml.safe_load(Path(".github/workflows/docs-ci.yml").read_text())
    quality_gate_steps = workflow["jobs"]["quality-gate"]["steps"]

    matching_steps = [
        step for step in quality_gate_steps if step.get("name") == "Docs governance"
    ]

    assert matching_steps == [
        {
            "name": "Docs governance",
            "run": "python3 scripts/check_docs_governance.py",
        }
    ]

    test_matching_steps = [
        step
        for step in quality_gate_steps
        if step.get("name") == "Run documentation governance tests"
    ]

    assert test_matching_steps == [
        {
            "name": "Run documentation governance tests",
            "run": "python -m pytest tests/scripts/test_doc_governance_checks.py -q",
        }
    ]
